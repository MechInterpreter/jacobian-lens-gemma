# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""The phrase reconstructor: phrase tokens → ``q_hat`` in J-space.

This is the half of the cycle that makes the experiment falsifiable. A trained
adapter will emit *something* for every cone; only an independently trained map
back from the emitted phrase to J-space can say whether that something is the
right thing — and only if it was trained without ever seeing the phrases it is
later asked about.

Design:

* **Input** is a frozen Gemma-native representation: the rows of
  ``embed_tokens`` for the phrase's own token ids, plus (optionally) a learned
  embedding of the source layer index. No trainable embedding table, no
  tokenizer surgery — the phrase enters exactly as Gemma reads it.
* **Trainable part** is small: an input projection, a shallow transformer
  encoder, attention pooling, and an output head to ``d_model``. Parameter
  counts are reported, not estimated.
* **Output** is unit-norm and has exactly ``d_model`` dimensions, the same space
  as ``q``. Nothing is projected between them.
* **Target** is a phrase prototype computed from **training-split cones only**.

The gate (:func:`reconstructor_gate`) is a stop condition, not a report: if the
reconstructor cannot separate correct phrases from hard distractors on
concept-disjoint validation data, the adapter is never trained.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn

from jlens.autoencoder.checkpoints import parameter_counts
from jlens.autoencoder.config import ReconstructorConfig
from jlens.autoencoder.dataset import JSpaceLanguageDataset
from jlens.autoencoder.errors import AutoencoderError
from jlens.autoencoder.geometry import (
    auroc,
    nonnegative_scale_fit,
    rank_of,
    top_k_accuracy,
    unit,
)


class PhraseEmbedder:
    """Frozen Gemma token embeddings for phrases, padded and masked.

    Caches by token-id tuple: a phrase's embedding rows never change, and the
    pilot asks for the same few hundred phrases thousands of times.
    """

    def __init__(self, model, *, max_phrase_tokens: int, device: torch.device | str | None = None):
        embed = getattr(model, "_embed_tokens", None)
        if embed is None:
            raise AutoencoderError("PhraseEmbedder requires a model exposing _embed_tokens")
        self._embed = embed
        self.max_phrase_tokens = int(max_phrase_tokens)
        self.device = torch.device(device) if device is not None else embed.weight.device
        self.d_model = int(embed.weight.shape[1])
        self._cache: dict[tuple[int, ...], torch.Tensor] = {}

    @torch.no_grad()
    def _rows(self, token_ids: Sequence[int]) -> torch.Tensor:
        key = tuple(int(t) for t in token_ids)
        if not key:
            raise AutoencoderError("cannot embed an empty phrase")
        if len(key) > self.max_phrase_tokens:
            raise AutoencoderError(
                f"phrase has {len(key)} tokens, above max_phrase_tokens="
                f"{self.max_phrase_tokens}"
            )
        cached = self._cache.get(key)
        if cached is None:
            ids = torch.tensor([key], dtype=torch.long, device=self._embed.weight.device)
            cached = self._embed(ids)[0].detach().float().cpu()
            self._cache[key] = cached
        return cached

    def batch(self, phrases: Sequence[Sequence[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        """``(embeddings [B, T, d_model], mask [B, T])`` for a batch of id lists.

        ``mask`` is ``True`` at real tokens. Padding rows are zero **and**
        masked, so a padded position cannot contribute through either path.
        """
        rows = [self._rows(ids) for ids in phrases]
        length = max(r.shape[0] for r in rows)
        embeddings = torch.zeros(len(rows), length, self.d_model, dtype=torch.float32)
        mask = torch.zeros(len(rows), length, dtype=torch.bool)
        for index, row in enumerate(rows):
            embeddings[index, : row.shape[0]] = row
            mask[index, : row.shape[0]] = True
        return embeddings, mask


class PhraseReconstructor(nn.Module):
    """Phrase token embeddings (+ source layer) → unit-norm ``q_hat``."""

    def __init__(
        self,
        *,
        d_model: int,
        config: ReconstructorConfig,
        n_model_layers: int = 64,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.config = config
        hidden = int(config.hidden_dim)
        self.input_projection = nn.Linear(self.d_model, hidden)
        self.input_norm = nn.LayerNorm(hidden)
        self.position = nn.Parameter(torch.zeros(int(config.max_phrase_tokens), hidden))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=int(config.n_heads),
            dim_feedforward=hidden * 2,
            dropout=float(config.dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=int(config.n_layers), enable_nested_tensor=False
        )
        self.pool_query = nn.Parameter(torch.randn(hidden) * hidden**-0.5)
        self.use_source_layer_embedding = bool(config.use_source_layer_embedding)
        if self.use_source_layer_embedding:
            self.layer_embedding = nn.Embedding(int(n_model_layers), hidden)
            nn.init.zeros_(self.layer_embedding.weight)
        self.output_norm = nn.LayerNorm(hidden)
        self.output_head = nn.Linear(hidden, self.d_model)
        nn.init.normal_(self.position, std=0.02)

    def forward(
        self,
        embeddings: torch.Tensor,
        mask: torch.Tensor,
        source_layer: int | torch.Tensor,
    ) -> torch.Tensor:
        """``[B, T, d_model]`` + ``[B, T]`` bool mask → ``[B, d_model]`` unit vectors."""
        if embeddings.ndim != 3:
            raise AutoencoderError(
                f"embeddings must be [B, T, d_model], got {tuple(embeddings.shape)}"
            )
        if embeddings.shape[-1] != self.d_model:
            raise AutoencoderError(
                f"embedding width {embeddings.shape[-1]} != d_model {self.d_model}"
            )
        if mask.shape != embeddings.shape[:2]:
            raise AutoencoderError(
                f"mask {tuple(mask.shape)} does not match embeddings "
                f"{tuple(embeddings.shape[:2])}"
            )
        batch, length, _ = embeddings.shape
        if length > self.position.shape[0]:
            raise AutoencoderError(
                f"phrase length {length} exceeds max_phrase_tokens "
                f"{self.position.shape[0]}"
            )
        hidden = self.input_norm(self.input_projection(embeddings)) + self.position[:length]
        encoded = self.encoder(hidden, src_key_padding_mask=~mask)
        # Attention pooling with a learned query; masked positions get -inf so
        # padding can never receive attention mass.
        scores = (encoded @ self.pool_query) / math.sqrt(encoded.shape[-1])
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        pooled = (encoded * weights).sum(dim=1)
        if self.use_source_layer_embedding:
            if not torch.is_tensor(source_layer):
                source_layer = torch.full(
                    (batch,), int(source_layer), dtype=torch.long, device=pooled.device
                )
            pooled = pooled + self.layer_embedding(source_layer.to(pooled.device))
        return unit(self.output_head(self.output_norm(pooled)))


def _prototype_targets(
    dataset: JSpaceLanguageDataset, split: str
) -> tuple[list[str], torch.Tensor, dict]:
    return dataset.prototypes(split)


def build_reconstructor_training(
    dataset: JSpaceLanguageDataset,
    *,
    config: ReconstructorConfig,
    device: torch.device | str = "cpu",
    seed_global: bool = True,
) -> tuple[PhraseReconstructor, torch.optim.Optimizer, torch.Generator]:
    """The model, optimizer, and sampler generator a training run starts from.

    Split out of :func:`train_reconstructor` so a resuming caller can create the
    objects, restore a checkpoint into them, and hand them back — the alternative
    (letting the trainer construct them, then overwriting) puts a fresh
    ``manual_seed`` between construction and restoration and quietly moves the
    RNG the resumed epoch depends on.

    ``seed_global=False`` is what a resume passes: the global stream comes from
    the checkpoint, not from the config.
    """
    if seed_global:
        torch.manual_seed(int(config.seed))
    model = PhraseReconstructor(d_model=dataset.d_model, config=config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config.learning_rate), weight_decay=float(config.weight_decay)
    )
    generator = torch.Generator().manual_seed(int(config.seed) + 1)
    return model, optimizer, generator


def train_reconstructor(
    dataset: JSpaceLanguageDataset,
    embedder: PhraseEmbedder,
    *,
    config: ReconstructorConfig,
    source_layer: int,
    phrase_token_ids: dict[str, list[int]],
    device: torch.device | str = "cpu",
    log: object = None,
    on_epoch=None,
    model: PhraseReconstructor | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    generator: torch.Generator | None = None,
    start_epoch: int = 0,
    start_batch: int = 0,
    global_step: int = 0,
    history: list[dict] | None = None,
    resume_order: list[int] | None = None,
    resume_partial: dict | None = None,
    checkpoint=None,
    checkpoint_every_steps: int = 0,
    guard=None,
) -> tuple[PhraseReconstructor, dict]:
    """Train the reconstructor on **training-split prototypes only**.

    Loss is a cosine regression term plus an in-batch InfoNCE term. The
    contrastive term is not decoration: the gate is a *retrieval* test, and a
    pure regression objective happily collapses every phrase onto the dataset
    mean, which scores well on cosine and at chance on retrieval.

    ``on_epoch(epoch, metrics)`` is called after every epoch — the hook the
    Colab notebook uses for live monitoring.

    **Resume.** Called with no ``model``/``optimizer``/``generator`` this trains
    from scratch exactly as before. A resuming caller passes restored objects
    plus the position ``(start_epoch, start_batch, global_step)`` and
    ``resume_order`` — the shuffle the interrupted epoch was already using.
    Re-deriving that order from the generator would be wrong: by the time the
    checkpoint was written the generator had already produced it and moved on.

    ``checkpoint(reason, epoch, batch_index, global_step, order, partial,
    metrics, history)`` writes a resume point and returns its path;
    ``checkpoint_every_steps`` sets the periodic cadence (0 disables). ``guard``
    is an :class:`~jlens.autoencoder.state.InterruptGuard`: when it reports a
    stop, this writes one final ``keyboard_interrupt`` checkpoint and raises
    :class:`~jlens.autoencoder.state.StageInterrupted` at a batch boundary.

    ``resume_partial`` carries the interrupted epoch's running loss sums. Without
    it the resumed epoch would divide by its own (smaller) batch count and report
    a mean over the tail only — parameters identical to an uninterrupted run,
    history quietly not.
    """
    resuming = model is not None or optimizer is not None or generator is not None
    phrases, prototypes, dispersion = _prototype_targets(dataset, "train")
    if len(phrases) < 2:
        raise AutoencoderError(
            f"training split has {len(phrases)} phrase(s); the contrastive term "
            f"needs at least 2"
        )
    if model is None or optimizer is None or generator is None:
        built_model, built_optimizer, built_generator = build_reconstructor_training(
            dataset, config=config, device=device, seed_global=not resuming
        )
        model = model or built_model
        optimizer = optimizer or built_optimizer
        generator = generator or built_generator
    targets = prototypes.to(device)
    token_lists = [phrase_token_ids[p] for p in phrases]
    n = len(phrases)
    history = list(history or [])
    interrupted_at: str | None = None

    def write_checkpoint(reason, epoch, batch_index, order, partial, metrics):
        if checkpoint is None:
            return None
        return checkpoint(
            reason=reason,
            epoch=int(epoch),
            batch_index=int(batch_index),
            global_step=int(global_step),
            order=list(order),
            partial=dict(partial),
            metrics=dict(metrics or {}),
            history=list(history),
        )

    for epoch in range(int(start_epoch), int(config.epochs)):
        model.train()
        if epoch == int(start_epoch) and resume_order is not None:
            order = list(resume_order)
        else:
            order = torch.randperm(n, generator=generator).tolist()
        first_batch = int(start_batch) if epoch == int(start_epoch) else 0
        carried = dict(resume_partial or {}) if epoch == int(start_epoch) else {}
        epoch_loss = float(carried.get("epoch_loss", 0.0))
        epoch_cosine = float(carried.get("epoch_cosine", 0.0))
        n_batches = int(carried.get("n_batches", 0))
        starts = list(range(0, n, int(config.batch_size)))
        for batch_index, start in enumerate(starts):
            if batch_index < first_batch:
                continue
            partial = {
                "epoch_loss": epoch_loss,
                "epoch_cosine": epoch_cosine,
                "n_batches": n_batches,
            }
            if guard is not None and guard.should_stop():
                interrupted_at = write_checkpoint(
                    "keyboard_interrupt", epoch, batch_index, order, partial, {}
                )
                break
            batch_indices = order[start : start + int(config.batch_size)]
            if len(batch_indices) < 2:
                continue
            embeddings, mask = embedder.batch([token_lists[i] for i in batch_indices])
            predicted = model(embeddings.to(device), mask.to(device), source_layer)
            batch_targets = targets[batch_indices]
            cosine_term = (predicted * batch_targets).sum(dim=-1)
            loss = (1.0 - cosine_term).mean()
            if config.contrastive_weight > 0:
                logits = (predicted @ batch_targets.T) / float(config.temperature)
                labels = torch.arange(len(batch_indices), device=logits.device)
                loss = loss + float(config.contrastive_weight) * (
                    nn.functional.cross_entropy(logits, labels)
                    + nn.functional.cross_entropy(logits.T, labels)
                ) / 2.0
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if config.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), float(config.grad_clip))
            optimizer.step()
            epoch_loss += float(loss.detach())
            epoch_cosine += float(cosine_term.detach().mean())
            n_batches += 1
            global_step += 1
            if (
                checkpoint_every_steps
                and global_step % int(checkpoint_every_steps) == 0
                and batch_index + 1 < len(starts)
            ):
                write_checkpoint(
                    "periodic",
                    epoch,
                    batch_index + 1,
                    order,
                    {
                        "epoch_loss": epoch_loss,
                        "epoch_cosine": epoch_cosine,
                        "n_batches": n_batches,
                    },
                    {},
                )
        if interrupted_at is not None:
            break
        metrics = {
            "epoch": epoch,
            "loss": epoch_loss / max(1, n_batches),
            "train_prototype_cosine": epoch_cosine / max(1, n_batches),
        }
        history.append(metrics)
        if log is not None:
            log.info(
                "reconstructor epoch %d loss=%.4f cos=%.4f",
                epoch,
                metrics["loss"],
                metrics["train_prototype_cosine"],
            )
        write_checkpoint("epoch_complete", epoch, 0, order, {}, metrics)
        if on_epoch is not None:
            on_epoch(epoch, metrics)
    if interrupted_at is not None or (guard is not None and guard.should_stop()):
        from jlens.autoencoder.state import StageInterrupted

        raise StageInterrupted(
            "reconstructor: stopped at a batch boundary",
            stage="reconstructor",
            checkpoint_path=interrupted_at,
        )
    # Counts are taken *before* freezing: after ``requires_grad_(False)`` every
    # parameter reads as frozen, and a summary saying "trainable: 0" for the
    # module that was just trained is a report nobody can use.
    counts = parameter_counts(model)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    summary = {
        "n_train_phrases": n,
        "history": history,
        "train_prototype_dispersion": dispersion,
        "frozen_after_training": True,
        "global_step": int(global_step),
        "resume_granularity": "batch",
        **counts,
    }
    return model, summary


@torch.no_grad()
def phrase_vectors(
    model: PhraseReconstructor,
    embedder: PhraseEmbedder,
    phrases: Sequence[str],
    phrase_token_ids: dict[str, list[int]],
    *,
    source_layer: int,
    device: torch.device | str = "cpu",
    batch_size: int = 64,
) -> torch.Tensor:
    """``[P, d_model]`` unit ``q_hat`` for a list of phrase surfaces."""
    outputs: list[torch.Tensor] = []
    for start in range(0, len(phrases), int(batch_size)):
        chunk = list(phrases[start : start + int(batch_size)])
        embeddings, mask = embedder.batch([phrase_token_ids[p] for p in chunk])
        outputs.append(model(embeddings.to(device), mask.to(device), source_layer).cpu())
    if not outputs:
        raise AutoencoderError("phrase_vectors called with no phrases")
    return torch.cat(outputs)


def hard_distractors(
    prototypes: torch.Tensor,
    phrases: Sequence[str],
    *,
    n_distractors: int,
) -> dict[str, list[str]]:
    """For each phrase, the ``n_distractors`` **most cone-similar** other phrases.

    Hard by construction: these are the phrases whose cones a reconstructor is
    most likely to confuse with the correct one. Random distractors would make
    the retrieval number look good for reasons that have nothing to do with the
    hypothesis.
    """
    similarity = unit(prototypes) @ unit(prototypes).T
    similarity.fill_diagonal_(float("-inf"))
    k = min(int(n_distractors), len(phrases) - 1)
    if k < 1:
        raise AutoencoderError(
            f"cannot build distractors from {len(phrases)} phrase(s); at least 2 needed"
        )
    top = similarity.topk(k, dim=-1).indices
    return {
        phrases[i]: [phrases[int(j)] for j in top[i]] for i in range(len(phrases))
    }


@torch.no_grad()
def evaluate_reconstructor(
    model: PhraseReconstructor,
    dataset: JSpaceLanguageDataset,
    embedder: PhraseEmbedder,
    *,
    split: str,
    config: ReconstructorConfig,
    source_layer: int,
    phrase_token_ids: dict[str, list[int]],
    device: torch.device | str = "cpu",
    extra_distractors: Sequence[str] = (),
) -> dict:
    """Measure the reconstructor on one split, against hard distractors.

    Reported per split:

    * ``mean_cosine`` / ``mean_explained_fraction`` — the round-trip fit for the
      *correct* phrase against each occurrence's own cone.
    * ``top1`` / ``top5`` retrieval and mean rank among hard distractors.
    * ``auroc`` separating correct-phrase from distractor scores.
    * ``specificity_margin`` — the correct phrase's score against its own cone
      minus its best score against an unrelated cone. A reconstructor that has
      learned "plausible English noun phrase" scores high cosine and ~zero
      margin; that is the distinction this number exists to expose.
    """
    indices = dataset.indices_for_split(split)
    if not indices:
        raise AutoencoderError(f"split {split!r} has no records")
    phrases = dataset.phrases_for_split(split)
    pool = list(dict.fromkeys([*phrases, *extra_distractors]))
    missing = [p for p in pool if p not in phrase_token_ids]
    if missing:
        raise AutoencoderError(f"no token ids for phrase(s) {missing[:5]}")
    vectors = phrase_vectors(
        model, embedder, pool, phrase_token_ids, source_layer=source_layer, device=device
    )
    index_of = {phrase: i for i, phrase in enumerate(pool)}
    _, prototypes, dispersion = dataset.prototypes(split)
    distractors = hard_distractors(prototypes, phrases, n_distractors=config.n_distractors)
    # Extra distractors (the confabulation attractors) are hard for a different
    # reason: they are famous, so a language-model-ish reconstructor rates them
    # highly for everything. They join every candidate list.
    extra = [p for p in extra_distractors if p not in set(phrases)]

    cosines: list[float] = []
    explained: list[float] = []
    ranks: list[int] = []
    positive_scores: list[float] = []
    negative_scores: list[float] = []
    margins: list[float] = []
    per_record: list[dict] = []
    cones = dataset.cones
    for record_index in indices:
        record = dataset.records[record_index]
        q = cones[record_index]
        correct = record["phrase"]
        q_hat = vectors[index_of[correct]]
        fit = nonnegative_scale_fit(q, q_hat)
        cosines.append(fit["cosine"])
        explained.append(fit["explained_fraction"])
        candidates = [*distractors[correct], *extra]
        candidate_scores = [
            float(torch.dot(unit(q), vectors[index_of[c]])) for c in candidates
        ]
        rank = rank_of(fit["cosine"], candidate_scores)
        ranks.append(rank)
        positive_scores.append(fit["cosine"])
        negative_scores.extend(candidate_scores)
        # Specificity: the same phrase judged against other examples' cones.
        unrelated = [
            float(torch.dot(unit(cones[j]), q_hat))
            for j in indices
            if dataset.records[j]["phrase_id"] != record["phrase_id"]
        ]
        margin = fit["cosine"] - (max(unrelated) if unrelated else 0.0)
        margins.append(margin)
        per_record.append(
            {
                "record_index": record_index,
                "phrase": correct,
                "cosine": fit["cosine"],
                "explained_fraction": fit["explained_fraction"],
                "alpha": fit["alpha"],
                "rank": rank,
                "n_candidates": len(candidates) + 1,
                "specificity_margin": margin,
            }
        )
    return {
        "split": split,
        "n_records": len(indices),
        "n_phrases": len(phrases),
        "n_candidates_per_record": config.n_distractors + len(extra) + 1,
        "mean_cosine": sum(cosines) / len(cosines),
        "mean_explained_fraction": sum(explained) / len(explained),
        "mean_specificity_margin": sum(margins) / len(margins),
        "top1_retrieval": top_k_accuracy(ranks, 1),
        "top5_retrieval": top_k_accuracy(ranks, 5),
        "mean_rank": sum(ranks) / len(ranks),
        "auroc_correct_vs_distractor": auroc(positive_scores, negative_scores),
        "prototype_dispersion": dispersion,
        "per_record": per_record,
    }


def reconstructor_gate(metrics: dict, *, config: ReconstructorConfig) -> dict:
    """The stop condition before the verbalizer is trained.

    Returns a verdict dict with an explicit ``passed`` flag and per-criterion
    detail. A ``None`` AUROC (not computable) **fails** — an unmeasurable gate
    is not a passed gate.
    """
    observed_auroc = metrics.get("auroc_correct_vs_distractor")
    observed_top5 = metrics.get("top5_retrieval")
    criteria = [
        {
            "name": "auroc_correct_vs_distractor",
            "observed": observed_auroc,
            "threshold": float(config.gate_auroc_min),
            "passed": observed_auroc is not None and observed_auroc >= config.gate_auroc_min,
        },
        {
            "name": "top5_retrieval",
            "observed": observed_top5,
            "threshold": float(config.gate_top5_min),
            "passed": observed_top5 is not None and observed_top5 >= config.gate_top5_min,
        },
    ]
    passed = all(c["passed"] for c in criteria)
    return {
        "gate": "reconstructor",
        "split": metrics.get("split"),
        "passed": bool(passed),
        "criteria": criteria,
        "verdict": "GO" if passed else "NO-GO",
        "message": (
            "reconstructor separates correct phrases from hard distractors on "
            "concept-disjoint data; the verbalizer may be trained"
            if passed
            else "reconstructor cannot distinguish correct phrases from hard "
            "distractors on concept-disjoint data — STOP, do not train the "
            "verbalizer (failure attributed to: phrase_reconstructor)"
        ),
    }
