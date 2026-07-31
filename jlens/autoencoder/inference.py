# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Inference: verbalize one cone, and say how much to believe the answer.

Every field the brief requires is returned for every call — top-K candidates,
raw Gemma sequence log-probability, the reconstruction's hash and norm, cosine,
scale-fitted explained fraction, specificity margin, final rank, and an
accept/abstain verdict.

Two abstention paths, because they mean different things:

* ``abstained_below_threshold`` — the best candidate's reconstruction or margin
  is too low. Nothing said anything useful.
* ``abstained_flat`` — several candidates score within ``flat_margin`` of the
  best. The scorer cannot distinguish them, and forcing a top-1 would
  manufacture a confident answer out of a tie. The brief asks for exactly this:
  do not force a single phrase when scores are flat.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from jlens.autoencoder.conditioning import ConditioningBackend
from jlens.autoencoder.config import PreferenceConfig
from jlens.autoencoder.errors import AutoencoderError
from jlens.autoencoder.preference import (
    ScoredCandidate,
    normalize_candidate_text,
    reconstruct_candidates,
    score_candidates,
)
from jlens.autoencoder.prompting import VerbalizerPrompt
from jlens.autoencoder.verbalizer import beam_search
from jlens.generative import tensor_sha256

#: Default flatness tolerance: candidates within this reward of the best are
#: treated as indistinguishable.
DEFAULT_FLAT_MARGIN = 0.01


@torch.no_grad()
def first_token_rank(
    model,
    prompt: VerbalizerPrompt,
    memory: torch.Tensor,
    target_token_id: int,
    *,
    conditioner: ConditioningBackend,
) -> dict:
    """Rank (0 = argmax) and log-probability of ``target_token_id`` at the
    prompt-final position under ``memory``.

    Independent of beam search: it measures whether the memory moved the
    distribution toward the right first token at all, which stays informative
    even when no beam contains the phrase.
    """
    if memory.ndim == 2:
        memory = memory.unsqueeze(0)
    ids = prompt.input_ids(batch=1, device=_prompt_device(model))
    with conditioner.conditioned(
        model, memory=memory, span=(prompt.memory_start, prompt.memory_end)
    ):
        logits = model.logits_from_ids(ids, n_last=1)[0, -1].float()
    log_p = torch.log_softmax(logits, dim=-1)
    target = int(target_token_id)
    if not 0 <= target < log_p.shape[0]:
        raise AutoencoderError(
            f"target token id {target} out of range for vocabulary {log_p.shape[0]}"
        )
    rank = int((log_p > log_p[target]).sum())
    return {
        "first_token_id": target,
        "first_token_rank": rank,
        "first_token_logprob": float(log_p[target]),
        "top1_token_id": int(log_p.argmax()),
    }


def _prompt_device(model) -> torch.device:
    return model._embed_tokens.weight.device


def verbalize_cone(
    model,
    adapter,
    reconstructor,
    embedder,
    q: torch.Tensor,
    prompt: VerbalizerPrompt,
    *,
    conditioner: ConditioningBackend,
    unrelated_cones: torch.Tensor,
    reward_config: PreferenceConfig,
    stop_token_ids: Sequence[int],
    pad_token_id: int,
    beam_width: int,
    max_new_tokens: int,
    source_layer: int,
    accept_recon_min: float,
    accept_margin_min: float,
    flat_margin: float = DEFAULT_FLAT_MARGIN,
    device: torch.device | str = "cpu",
    memory: torch.Tensor | None = None,
) -> dict:
    """Verbalize one cone and report the full evidence for the answer.

    ``memory`` may be supplied directly (that is how the baselines reuse this
    path with a substituted memory); otherwise it is ``adapter(q)``.
    """
    with torch.no_grad():
        if memory is None:
            memory = adapter(q.unsqueeze(0).to(device))
        elif memory.ndim == 2:
            memory = memory.unsqueeze(0)
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
        scored = score_candidates(
            candidates,
            q,
            q_hats,
            unrelated_cones,
            config=reward_config,
            accept_recon_min=accept_recon_min,
            accept_margin_min=accept_margin_min,
        )
    ranked = sorted(range(len(scored)), key=lambda i: (-scored[i].reward, i))
    entries = []
    for final_rank, index in enumerate(ranked):
        item: ScoredCandidate = scored[index]
        entries.append(
            {
                **item.to_dict(),
                "final_rank": final_rank,
                "raw_beam_rank": item.candidate.beam_rank,
                "q_hat_norm": float(item.q_hat.norm()),
                "q_hat_sha256": tensor_sha256(item.q_hat),
            }
        )
    best = entries[0] if entries else None
    runners_up = [e for e in entries[1:] if best is not None and best["reward"] - e["reward"] < float(flat_margin)]
    if best is None:
        verdict = "abstained_no_candidates"
    elif not best["accepted"]:
        verdict = "abstained_below_threshold"
    elif runners_up:
        verdict = "abstained_flat"
    else:
        verdict = "accepted"
    return {
        "schema": "jlens.autoencoder.inference.v1",
        "verdict": verdict,
        "accepted": verdict == "accepted",
        "n_candidates": len(entries),
        "n_flat_runners_up": len(runners_up),
        "flat_margin": float(flat_margin),
        "accept_recon_min": float(accept_recon_min),
        "accept_margin_min": float(accept_margin_min),
        "q_norm": float(q.norm()),
        "q_sha256": tensor_sha256(q),
        "memory_rms": float(memory.detach().float().pow(2).mean().sqrt()),
        "top_candidate": best,
        "candidates": entries,
    }


def exact_match(result: dict, phrase: str, *, top_k: int = 1) -> bool:
    """Whether ``phrase`` appears among the top-``k`` candidates by final rank.

    Comparison is on the normalized surface (case-folded, whitespace-collapsed),
    which is the same normalization the duplicate penalty uses.
    """
    target = normalize_candidate_text(phrase)
    for entry in result["candidates"][: int(top_k)]:
        if normalize_candidate_text(entry["text"]) == target:
            return True
    return False


def substring_recovery(result: dict, phrase: str, *, top_k: int = 1) -> float:
    """Best normalized word-overlap between ``phrase`` and the top-``k``
    candidates, in ``[0, 1]``.

    Partial credit matters here: "Barrier Reef" for "Great Barrier Reef" is a
    materially different outcome from "photosynthesis", and an exact-match-only
    report would score them identically.
    """
    target_words = normalize_candidate_text(phrase).split()
    if not target_words:
        return 0.0
    best = 0.0
    for entry in result["candidates"][: int(top_k)]:
        words = set(normalize_candidate_text(entry["text"]).split())
        overlap = sum(1 for w in target_words if w in words) / len(target_words)
        best = max(best, overlap)
    return best
