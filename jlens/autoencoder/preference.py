# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Reconstructor-guided optimization of the cone adapter.

The reconstructor must *optimize* the verbalizer, not merely filter it
afterwards — a reranker applied at inference can only pick from what the beam
already contains, and would leave the central claim ("the reconstruction score
is a training signal for verbalization") untested.

The objective is deliberately offline and pairwise rather than full RL:

1. Beam candidates are generated once per example per epoch, without gradients.
2. Each candidate is mapped through the **frozen** reconstructor to ``q_hat``
   and scored against the example's own ``q`` and against unrelated cones.
3. Pairs with a reward gap become a DPO-style preference loss against a frozen
   copy of the warm-start adapter, so the update is anchored and cannot drift
   into degenerate text that happens to score well.

Nothing here backpropagates into Gemma or the reconstructor; both are asserted
frozen before and after every step. A REINFORCE refinement exists and is off by
default (:class:`~jlens.autoencoder.config.PolicyGradientConfig`).
"""

from __future__ import annotations

import copy
import os
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn

from jlens.autoencoder.adapter import ConeAdapter
from jlens.autoencoder.checkpoints import state_dict_sha256
from jlens.autoencoder.conditioning import (
    ConditioningBackend,
    assert_gemma_frozen,
    assert_no_frozen_parameters_in_optimizer,
)
from jlens.autoencoder.config import PreferenceConfig
from jlens.autoencoder.errors import AutoencoderError
from jlens.autoencoder.geometry import nonnegative_scale_fit, unit
from jlens.autoencoder.prompting import VerbalizerPrompt
from jlens.autoencoder.state import (
    atomic_write_json,
    iter_valid_files,
    read_json_if_valid,
    sha256_of_json,
)
from jlens.autoencoder.verbalizer import (
    Candidate,
    assert_no_gemma_gradients,
    beam_search,
    sequence_logprobs,
)
from jlens.generative import tensor_sha256

BEAM_CACHE_SCHEMA = "jlens.autoencoder.preference.beam_cache.v1"


def normalize_candidate_text(text: str) -> str:
    """Comparison form for duplicate detection and exact match."""
    return " ".join(str(text).split()).casefold()


def beam_cache_key(
    *,
    q: torch.Tensor,
    adapter_sha256: str,
    prompt_id: str,
    beam_width: int,
    max_new_tokens: int,
    stop_token_ids: Sequence[int],
) -> str:
    """Identity of one beam generation.

    Beam search is deterministic given ``(q, adapter weights, prompt, decoding
    settings)``, so those four things — and nothing else — are the key. The
    adapter hash is the load-bearing part: it changes on every optimizer step, so
    a cache hit is proof the weights are the ones that produced the entry, not an
    assumption that they are.
    """
    return sha256_of_json(
        {
            "schema": BEAM_CACHE_SCHEMA,
            "q_sha256": tensor_sha256(q.detach().float().cpu()),
            "adapter_sha256": adapter_sha256,
            "prompt_id": prompt_id,
            "beam_width": int(beam_width),
            "max_new_tokens": int(max_new_tokens),
            "stop_token_ids": [int(t) for t in stop_token_ids],
        }
    )


class BeamCache:
    """Disk-backed cache of generated beams, keyed by :func:`beam_cache_key`.

    Exists for one specific loss: an interruption *during* the candidate
    generation of a batch. Those beams were the expensive part, the adapter has
    not moved since the last checkpoint, and regenerating them on resume is pure
    waste. Between batches the adapter changes and the keys miss, which is
    correct — a stale beam would be a beam from a different policy.

    ``capacity`` bounds the directory: entries are dropped oldest-first by write
    order, because the useful ones are always the most recent adapter's.
    """

    def __init__(self, directory: str | None, *, capacity: int = 512, enabled: bool = True):
        self.directory = directory
        self.capacity = int(capacity)
        self.enabled = bool(enabled and directory)
        self.hits = 0
        self.misses = 0
        self.writes = 0
        if self.enabled:
            os.makedirs(self.directory, exist_ok=True)

    def _path(self, key: str) -> str:
        return os.path.join(self.directory, f"beams_{key.split(':')[-1][:32]}.json")

    def get(self, key: str) -> list[Candidate] | None:
        if not self.enabled:
            return None
        payload = read_json_if_valid(self._path(key))
        if not payload or payload.get("key") != key or payload.get("schema") != BEAM_CACHE_SCHEMA:
            self.misses += 1
            return None
        self.hits += 1
        return [
            Candidate(
                token_ids=tuple(int(t) for t in entry["token_ids"]),
                text=str(entry["text"]),
                logprob=float(entry["sequence_logprob"]),
                mean_logprob=float(entry["mean_token_logprob"]),
                n_tokens=int(entry["n_tokens"]),
                finished=bool(entry["finished"]),
                beam_rank=int(entry["beam_rank"]),
            )
            for entry in payload.get("candidates", [])
        ]

    def put(self, key: str, candidates: Sequence[Candidate]) -> None:
        if not self.enabled:
            return
        atomic_write_json(
            self._path(key),
            {
                "schema": BEAM_CACHE_SCHEMA,
                "key": key,
                "candidates": [c.to_dict() for c in candidates],
            },
        )
        self.writes += 1
        self._trim()

    def _trim(self) -> None:
        paths = iter_valid_files(self.directory, suffix=".json")
        if len(paths) <= self.capacity:
            return
        paths.sort(key=lambda p: os.path.getmtime(p))
        for path in paths[: len(paths) - self.capacity]:
            try:
                os.remove(path)
            except OSError:  # pragma: no cover - a racing reader is harmless
                pass

    def stats(self) -> dict:
        return {
            "enabled": self.enabled,
            "directory": self.directory,
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
            "capacity": self.capacity,
        }


@dataclass(frozen=True)
class ScoredCandidate:
    """A candidate with its reconstruction geometry and reward decomposition."""

    candidate: Candidate
    q_hat: torch.Tensor
    cosine: float
    alpha: float
    explained_fraction: float
    unrelated_best: float
    margin: float
    brevity_penalty: float
    duplicate_penalty: float
    reward: float
    accepted: bool

    def to_dict(self) -> dict:
        return {
            **self.candidate.to_dict(),
            "cosine": self.cosine,
            "alpha": self.alpha,
            "explained_fraction": self.explained_fraction,
            "unrelated_best_cosine": self.unrelated_best,
            "specificity_margin": self.margin,
            "brevity_penalty": self.brevity_penalty,
            "duplicate_penalty": self.duplicate_penalty,
            "reward": self.reward,
            "accepted": self.accepted,
        }


@torch.no_grad()
def reconstruct_candidates(
    reconstructor,
    embedder,
    candidates: Sequence[Candidate],
    *,
    source_layer: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """``[C, d_model]`` unit ``q_hat`` for a list of candidates.

    A candidate with no tokens (the model stopped immediately) contributes a
    zero row rather than an error: "said nothing" must score zero, not crash the
    run.
    """
    usable = [i for i, c in enumerate(candidates) if c.n_tokens > 0]
    width = embedder.d_model
    output = torch.zeros(len(candidates), width)
    if not usable:
        return output
    embeddings, mask = embedder.batch([candidates[i].token_ids for i in usable])
    vectors = reconstructor(embeddings.to(device), mask.to(device), source_layer).cpu()
    for row, index in enumerate(usable):
        output[index] = vectors[row]
    return output


def score_candidates(
    candidates: Sequence[Candidate],
    q: torch.Tensor,
    q_hats: torch.Tensor,
    unrelated_cones: torch.Tensor,
    *,
    config: PreferenceConfig,
    accept_recon_min: float,
    accept_margin_min: float,
) -> list[ScoredCandidate]:
    """Turn candidates into rewards.

    ``reward = w_r * recon + w_m * margin - brevity - duplicate``, where ``recon``
    is the clamped cosine between ``q`` and the candidate's ``q_hat`` and
    ``margin`` is that minus the best cosine any *unrelated* cone gives the same
    candidate. The margin term is what stops the optimizer from converging on a
    generically plausible phrase that every cone likes.
    """
    if len(candidates) != q_hats.shape[0]:
        raise AutoencoderError(
            f"{len(candidates)} candidates but {q_hats.shape[0]} reconstructions"
        )
    unrelated_unit = unit(unrelated_cones.float()) if unrelated_cones.numel() else None
    seen: dict[str, int] = {}
    scored: list[ScoredCandidate] = []
    for index, candidate in enumerate(candidates):
        q_hat = q_hats[index]
        if candidate.n_tokens == 0 or float(q_hat.norm()) == 0.0:
            fit = {"cosine": 0.0, "alpha": 0.0, "explained_fraction": 0.0}
            unrelated_best = 0.0
        else:
            fit = nonnegative_scale_fit(q.float(), q_hat)
            # ``unit(q_hat)`` explicitly: the reconstructor happens to emit unit
            # vectors, but ``margin`` is a difference of cosines and must not
            # silently become scale-dependent if a caller passes something else.
            unrelated_best = (
                float((unrelated_unit @ unit(q_hat)).max())
                if unrelated_unit is not None
                else 0.0
            )
        recon = max(0.0, float(fit["cosine"]))
        margin = float(fit["cosine"]) - unrelated_best
        key = normalize_candidate_text(candidate.text)
        duplicates = seen.get(key, 0)
        seen[key] = duplicates + 1
        brevity = float(config.brevity_penalty) * max(
            0, candidate.n_tokens - int(config.brevity_target_tokens)
        )
        duplicate = float(config.duplicate_penalty) * duplicates
        reward = (
            float(config.weight_reconstruction) * recon
            + float(config.weight_margin) * margin
            - brevity
            - duplicate
        )
        scored.append(
            ScoredCandidate(
                candidate=candidate,
                q_hat=q_hat,
                cosine=float(fit["cosine"]),
                alpha=float(fit["alpha"]),
                explained_fraction=float(fit["explained_fraction"]),
                unrelated_best=unrelated_best,
                margin=margin,
                brevity_penalty=brevity,
                duplicate_penalty=duplicate,
                reward=reward,
                accepted=bool(recon >= accept_recon_min and margin >= accept_margin_min),
            )
        )
    return scored


def build_preference_pairs(
    scored: Sequence[ScoredCandidate], *, reward_gap: float, max_pairs: int
) -> list[tuple[int, int]]:
    """Index pairs ``(preferred, rejected)`` with a reward gap of at least
    ``reward_gap``, largest gaps first, capped at ``max_pairs``.

    Deterministic: ties in gap break toward the earlier beam ranks, so two runs
    with identical beams train on identical pairs.
    """
    order = sorted(range(len(scored)), key=lambda i: (-scored[i].reward, i))
    pairs: list[tuple[float, int, int]] = []
    for a_position, a in enumerate(order):
        for b in order[a_position + 1 :]:
            gap = scored[a].reward - scored[b].reward
            if gap >= float(reward_gap):
                pairs.append((gap, a, b))
    pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [(a, b) for _gap, a, b in pairs[: int(max_pairs)]]


def sample_unrelated_cones(
    dataset, index: int, *, n: int, generator: torch.Generator
) -> torch.Tensor:
    """``[n, d_model]`` cones from **other phrases** in the same split.

    Same-split on purpose: an "unrelated" cone drawn from a different split
    would confound specificity with distribution shift.
    """
    split = dataset.records[index]["split"]
    phrase_id = dataset.records[index]["phrase_id"]
    pool = [
        i
        for i in dataset.indices_for_split(split)
        if dataset.records[i]["phrase_id"] != phrase_id
    ]
    if not pool:
        raise AutoencoderError(
            f"no unrelated cone available in split {split!r}; the specificity "
            f"margin cannot be computed"
        )
    count = min(int(n), len(pool))
    picked = torch.randperm(len(pool), generator=generator)[:count].tolist()
    return dataset.cones[[pool[i] for i in picked]]


def train_preference(
    model,
    adapter: ConeAdapter,
    reconstructor,
    embedder,
    dataset,
    prompt: VerbalizerPrompt,
    *,
    config: PreferenceConfig,
    conditioner: ConditioningBackend,
    stop_token_ids: Sequence[int],
    pad_token_id: int,
    beam_width: int,
    max_new_tokens: int,
    source_layer: int,
    accept_recon_min: float,
    accept_margin_min: float,
    device: torch.device | str = "cpu",
    split: str = "train",
    log: object = None,
    on_epoch=None,
    start_epoch: int = 0,
    start_batch: int = 0,
    global_step: int = 0,
    optimizer: torch.optim.Optimizer | None = None,
    generator: torch.Generator | None = None,
    history: list[dict] | None = None,
    resume_order: list[int] | None = None,
    resume_partial: dict | None = None,
    reference_state: dict | None = None,
    beam_cache: BeamCache | None = None,
    checkpoint=None,
    checkpoint_every_steps: int = 0,
    guard=None,
) -> tuple[ConeAdapter, dict]:
    """DPO-style preference optimization of the adapter against the frozen
    reconstructor's reward.

    The reference policy is the **warm-started** adapter, so the KL anchor is the
    supervised model rather than a moving target. On a fresh run that is a deep
    copy of ``adapter`` as it enters here; on a resume it must come from
    ``reference_state`` (the preserved ``adapter_warm.pt``), because by then
    ``adapter`` has already taken preference steps and copying it would anchor
    the run to a policy the uninterrupted run never had.

    **Resume.** As in the warm start: position, sampler order, generator state,
    and the interrupted epoch's running sums. ``beam_cache`` reuses candidate
    beams that were generated before the interruption but never trained on.
    """
    assert_gemma_frozen(model, where="preference training (before)")
    for parameter in reconstructor.parameters():
        if parameter.requires_grad:
            raise AutoencoderError(
                "the reconstructor has trainable parameters during preference "
                "training; it must be frozen before the adapter is optimized"
            )
    adapter = adapter.to(device)
    reference = copy.deepcopy(adapter).to(device).eval()
    if reference_state is not None:
        reference.load_state_dict({k: v.to(device) for k, v in reference_state.items()})
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    adapter.train()
    if optimizer is None:
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=float(config.learning_rate))
    optimizer_report = assert_no_frozen_parameters_in_optimizer(
        optimizer, frozen_modules=[model, reconstructor, reference], trainable=adapter
    )
    indices = dataset.indices_for_split(split)
    if generator is None:
        generator = torch.Generator().manual_seed(int(config.seed))
    history = list(history or [])
    cache = beam_cache or BeamCache(None, enabled=False)
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
        if epoch == int(start_epoch) and resume_order is not None:
            order = list(resume_order)
        else:
            order = [
                indices[i] for i in torch.randperm(len(indices), generator=generator).tolist()
            ]
        first_batch = int(start_batch) if epoch == int(start_epoch) else 0
        carried = dict(resume_partial or {}) if epoch == int(start_epoch) else {}
        epoch_loss = float(carried.get("epoch_loss", 0.0))
        n_updates = int(carried.get("n_updates", 0))
        n_pairs_total = int(carried.get("n_pairs_total", 0))
        n_abstained = int(carried.get("n_abstained", 0))
        mean_reward: list[float] = list(carried.get("mean_reward", []))
        starts = list(range(0, len(order), int(config.batch_size)))
        for batch_index, start in enumerate(starts):
            if batch_index < first_batch:
                continue
            partial = {
                "epoch_loss": epoch_loss,
                "n_updates": n_updates,
                "n_pairs_total": n_pairs_total,
                "n_abstained": n_abstained,
                "mean_reward": list(mean_reward),
            }
            if guard is not None and guard.should_stop():
                interrupted_at = write_checkpoint(
                    "keyboard_interrupt", epoch, batch_index, order, partial, {}
                )
                break
            batch_indices = order[start : start + int(config.batch_size)]
            sequences: list[list[int]] = []
            memory_rows: list[torch.Tensor] = []
            pair_slots: list[tuple[int, int]] = []
            adapter_sha = state_dict_sha256(adapter) if cache.enabled else ""
            with torch.no_grad():
                for record_index in batch_indices:
                    q = dataset.cones[record_index]
                    key = (
                        beam_cache_key(
                            q=q,
                            adapter_sha256=adapter_sha,
                            prompt_id=prompt.prompt_id,
                            beam_width=int(beam_width),
                            max_new_tokens=int(max_new_tokens),
                            stop_token_ids=stop_token_ids,
                        )
                        if cache.enabled
                        else ""
                    )
                    candidates = cache.get(key) if cache.enabled else None
                    if candidates is None:
                        memory = adapter(q.unsqueeze(0).to(device))
                        candidates = beam_search(
                            model,
                            prompt,
                            memory.detach(),
                            conditioner=conditioner,
                            beam_width=int(beam_width),
                            max_new_tokens=int(max_new_tokens),
                            stop_token_ids=stop_token_ids,
                            pad_token_id=pad_token_id,
                        )
                        if cache.enabled:
                            cache.put(key, candidates)
                    q_hats = reconstruct_candidates(
                        reconstructor,
                        embedder,
                        candidates,
                        source_layer=source_layer,
                        device=device,
                    )
                    unrelated = sample_unrelated_cones(
                        dataset, record_index, n=config.n_unrelated_cones, generator=generator
                    )
                    scored = score_candidates(
                        candidates,
                        q,
                        q_hats,
                        unrelated,
                        config=config,
                        accept_recon_min=accept_recon_min,
                        accept_margin_min=accept_margin_min,
                    )
                    mean_reward.append(
                        max((s.reward for s in scored), default=0.0)
                    )
                    if not any(s.accepted for s in scored):
                        n_abstained += 1
                    pairs = build_preference_pairs(
                        scored,
                        reward_gap=config.reward_gap,
                        max_pairs=config.max_pairs_per_example,
                    )
                    n_pairs_total += len(pairs)
                    for preferred, rejected in pairs:
                        for choice in (preferred, rejected):
                            tokens = list(scored[choice].candidate.token_ids)
                            if not tokens:
                                tokens = [int(stop_token_ids[0])]
                            sequences.append([*tokens, int(stop_token_ids[0])])
                            memory_rows.append(q)
                        pair_slots.append((len(sequences) - 2, len(sequences) - 1))
            if not pair_slots:
                continue
            q_batch = torch.stack(memory_rows).to(device)
            memory = adapter(q_batch)
            scored_policy = sequence_logprobs(
                model,
                prompt,
                memory,
                sequences,
                conditioner=conditioner,
                pad_token_id=pad_token_id,
            )
            with torch.no_grad():
                reference_memory = reference(q_batch)
                scored_reference = sequence_logprobs(
                    model,
                    prompt,
                    reference_memory,
                    sequences,
                    conditioner=conditioner,
                    pad_token_id=pad_token_id,
                )
            key = "mean" if config.length_normalize else "total"
            policy = scored_policy[key]
            reference_scores = scored_reference[key]
            preferred_slots = torch.tensor([p for p, _ in pair_slots], device=policy.device)
            rejected_slots = torch.tensor([r for _, r in pair_slots], device=policy.device)
            advantage = (
                policy[preferred_slots] - reference_scores[preferred_slots]
            ) - (policy[rejected_slots] - reference_scores[rejected_slots])
            loss = -nn.functional.logsigmoid(float(config.beta) * advantage).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            assert_no_gemma_gradients(model, where="preference backward")
            optimizer.step()
            epoch_loss += float(loss.detach())
            n_updates += 1
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
                        "n_updates": n_updates,
                        "n_pairs_total": n_pairs_total,
                        "n_abstained": n_abstained,
                        "mean_reward": list(mean_reward),
                    },
                    {},
                )
        if interrupted_at is not None:
            break
        metrics = {
            "epoch": epoch,
            "loss": epoch_loss / max(1, n_updates),
            "n_updates": n_updates,
            "n_pairs": n_pairs_total,
            "n_examples": len(order),
            "abstention_rate": n_abstained / max(1, len(order)),
            "mean_best_reward": sum(mean_reward) / max(1, len(mean_reward)),
        }
        history.append(metrics)
        if log is not None:
            log.info(
                "preference epoch %d loss=%.4f pairs=%d best_reward=%.4f",
                epoch,
                metrics["loss"],
                n_pairs_total,
                metrics["mean_best_reward"],
            )
        write_checkpoint("epoch_complete", epoch, 0, order, {}, metrics)
        if on_epoch is not None:
            on_epoch(epoch, metrics, optimizer)
    if interrupted_at is not None or (guard is not None and guard.should_stop()):
        from jlens.autoencoder.state import StageInterrupted

        raise StageInterrupted(
            "adapter_preference: stopped at a batch boundary",
            stage="adapter_preference",
            checkpoint_path=interrupted_at,
        )
    adapter.eval()
    assert_gemma_frozen(model, where="preference training (after)")
    summary = {
        "split": split,
        "objective": "offline pairwise DPO against the frozen warm-start adapter",
        "policy_gradient_enabled": bool(config.policy_gradient.enabled),
        "history": history,
        "optimizer": optimizer_report,
        "global_step": int(global_step),
        "resume_granularity": "batch",
        "beam_cache": cache.stats(),
        "reference_policy": (
            "restored warm-start adapter" if reference_state is not None else "adapter at entry"
        ),
    }
    if config.policy_gradient.enabled:
        summary["policy_gradient"] = train_policy_gradient(
            model,
            adapter,
            reconstructor,
            embedder,
            dataset,
            prompt,
            config=config,
            conditioner=conditioner,
            stop_token_ids=stop_token_ids,
            pad_token_id=pad_token_id,
            beam_width=beam_width,
            max_new_tokens=max_new_tokens,
            source_layer=source_layer,
            accept_recon_min=accept_recon_min,
            accept_margin_min=accept_margin_min,
            device=device,
            split=split,
            log=log,
        )
    return adapter, summary


def train_policy_gradient(
    model,
    adapter: ConeAdapter,
    reconstructor,
    embedder,
    dataset,
    prompt: VerbalizerPrompt,
    *,
    config: PreferenceConfig,
    conditioner: ConditioningBackend,
    stop_token_ids: Sequence[int],
    pad_token_id: int,
    beam_width: int,
    max_new_tokens: int,
    source_layer: int,
    accept_recon_min: float,
    accept_margin_min: float,
    device: torch.device | str = "cpu",
    split: str = "train",
    log: object = None,
) -> dict:
    """Optional REINFORCE refinement over the beam candidates.

    Off by default. It is included because the brief asks for it to exist, and
    disabled because a high-variance policy gradient on top of a small adapter is
    a good way to produce a result that cannot be reproduced — the pairwise
    objective above is the one this study's conclusions rest on.
    """
    pg = config.policy_gradient
    if not pg.enabled:
        return {"enabled": False, "ran": False}
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=float(pg.learning_rate))
    assert_no_frozen_parameters_in_optimizer(
        optimizer, frozen_modules=[model, reconstructor], trainable=adapter
    )
    indices = dataset.indices_for_split(split)
    generator = torch.Generator().manual_seed(int(config.seed) + 101)
    history: list[dict] = []
    for epoch in range(int(pg.epochs)):
        epoch_loss = 0.0
        n_updates = 0
        for record_index in indices:
            q = dataset.cones[record_index]
            with torch.no_grad():
                memory = adapter(q.unsqueeze(0).to(device))
                candidates = beam_search(
                    model,
                    prompt,
                    memory,
                    conditioner=conditioner,
                    beam_width=int(beam_width),
                    max_new_tokens=int(max_new_tokens),
                    stop_token_ids=stop_token_ids,
                    pad_token_id=pad_token_id,
                )
                q_hats = reconstruct_candidates(
                    reconstructor, embedder, candidates, source_layer=source_layer, device=device
                )
                unrelated = sample_unrelated_cones(
                    dataset, record_index, n=config.n_unrelated_cones, generator=generator
                )
                scored = score_candidates(
                    candidates,
                    q,
                    q_hats,
                    unrelated,
                    config=config,
                    accept_recon_min=accept_recon_min,
                    accept_margin_min=accept_margin_min,
                )
            usable = [s for s in scored if s.candidate.n_tokens > 0]
            if len(usable) < 2:
                continue
            rewards = torch.tensor([s.reward for s in usable], dtype=torch.float32)
            baseline = rewards.mean() if pg.baseline == "mean" else torch.zeros(())
            sequences = [
                [*s.candidate.token_ids, int(stop_token_ids[0])] for s in usable
            ]
            memory = adapter(q.unsqueeze(0).to(device))
            scores = sequence_logprobs(
                model,
                prompt,
                memory,
                sequences,
                conditioner=conditioner,
                pad_token_id=pad_token_id,
            )
            advantage = (rewards - baseline).to(scores["mean"].device)
            loss = -(advantage * scores["mean"]).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            assert_no_gemma_gradients(model, where="policy gradient backward")
            optimizer.step()
            epoch_loss += float(loss.detach())
            n_updates += 1
        history.append({"epoch": epoch, "loss": epoch_loss / max(1, n_updates), "n_updates": n_updates})
        if log is not None:
            log.info("policy gradient epoch %d loss=%.4f", epoch, history[-1]["loss"])
    return {"enabled": True, "ran": True, "history": history}
