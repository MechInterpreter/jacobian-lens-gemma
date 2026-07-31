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
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn

from jlens.autoencoder.adapter import ConeAdapter
from jlens.autoencoder.conditioning import (
    ConditioningBackend,
    assert_gemma_frozen,
    assert_no_frozen_parameters_in_optimizer,
)
from jlens.autoencoder.config import PreferenceConfig
from jlens.autoencoder.errors import AutoencoderError
from jlens.autoencoder.geometry import nonnegative_scale_fit, unit
from jlens.autoencoder.prompting import VerbalizerPrompt
from jlens.autoencoder.verbalizer import (
    Candidate,
    assert_no_gemma_gradients,
    beam_search,
    sequence_logprobs,
)


def normalize_candidate_text(text: str) -> str:
    """Comparison form for duplicate detection and exact match."""
    return " ".join(str(text).split()).casefold()


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
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[ConeAdapter, dict]:
    """DPO-style preference optimization of the adapter against the frozen
    reconstructor's reward.

    The reference policy is a frozen deep copy of the adapter as it enters this
    function (i.e. the warm-started adapter), so the KL anchor is the supervised
    model rather than a moving target.
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
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    adapter.train()
    if optimizer is None:
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=float(config.learning_rate))
    optimizer_report = assert_no_frozen_parameters_in_optimizer(
        optimizer, frozen_modules=[model, reconstructor, reference], trainable=adapter
    )
    indices = dataset.indices_for_split(split)
    generator = torch.Generator().manual_seed(int(config.seed))
    history: list[dict] = []
    for epoch in range(int(start_epoch), int(config.epochs)):
        order = [indices[i] for i in torch.randperm(len(indices), generator=generator).tolist()]
        epoch_loss = 0.0
        n_updates = 0
        n_pairs_total = 0
        n_abstained = 0
        mean_reward: list[float] = []
        for start in range(0, len(order), int(config.batch_size)):
            batch_indices = order[start : start + int(config.batch_size)]
            sequences: list[list[int]] = []
            memory_rows: list[torch.Tensor] = []
            pair_slots: list[tuple[int, int]] = []
            with torch.no_grad():
                for record_index in batch_indices:
                    q = dataset.cones[record_index]
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
        if on_epoch is not None:
            on_epoch(epoch, metrics, optimizer)
    adapter.eval()
    assert_gemma_frozen(model, where="preference training (after)")
    summary = {
        "split": split,
        "objective": "offline pairwise DPO against the frozen warm-start adapter",
        "policy_gradient_enabled": bool(config.policy_gradient.enabled),
        "history": history,
        "optimizer": optimizer_report,
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
