# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""The small numeric core shared by the reconstructor, the reward, and the
evaluator.

Kept in one place so "explained fraction" means exactly one thing everywhere in
this experiment. The scale-invariant fit is the definition the brief fixes::

    alpha* = argmin_{alpha >= 0} || q - alpha * q_hat ||^2
           = max(0, <q, q_hat>) / ||q_hat||^2

from which ``1 - ||q - alpha* q_hat||^2 / ||q||^2 = max(0, cos(q, q_hat))^2``.
Both the raw cosine and the scale-fitted explained fraction are reported
everywhere; the second is a monotone function of the first for a single pair,
but it is the quantity that answers "how much of ``q`` did the phrase recover?"
and the two are not interchangeable when averaged.
"""

from __future__ import annotations

import torch

from jlens.autoencoder.errors import AutoencoderError


def unit(vectors: torch.Tensor, *, eps: float = 1e-30) -> torch.Tensor:
    """L2-normalize along the last dimension. Zero rows stay zero."""
    return vectors / vectors.norm(dim=-1, keepdim=True).clamp_min(eps)


def cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Row-wise cosine similarity, broadcasting over leading dimensions."""
    return (unit(a) * unit(b)).sum(dim=-1)


def nonnegative_scale_fit(q: torch.Tensor, q_hat: torch.Tensor) -> dict:
    """Fit ``alpha >= 0`` minimizing ``||q - alpha q_hat||^2`` for one pair.

    Returns ``alpha``, the fitted residual norm, the raw cosine, and the
    scale-fitted explained fraction. A zero ``q_hat`` yields ``alpha = 0`` and an
    explained fraction of 0 rather than an error: an adapter that emits nothing
    should score nothing, not crash the evaluation.
    """
    if q.shape != q_hat.shape or q.ndim != 1:
        raise AutoencoderError(
            f"nonnegative_scale_fit expects two matching 1-D vectors, got "
            f"{tuple(q.shape)} and {tuple(q_hat.shape)}"
        )
    q = q.detach().float()
    q_hat = q_hat.detach().float()
    q_norm_sq = float(q.pow(2).sum())
    hat_norm_sq = float(q_hat.pow(2).sum())
    if hat_norm_sq == 0.0 or q_norm_sq == 0.0:
        return {
            "alpha": 0.0,
            "cosine": 0.0,
            "explained_fraction": 0.0,
            "residual_norm": float(q.norm()),
            "q_norm": float(q.norm()),
            "q_hat_norm": float(q_hat.norm()),
        }
    inner = float((q * q_hat).sum())
    alpha = max(0.0, inner) / hat_norm_sq
    residual = q - alpha * q_hat
    cos = inner / (q_norm_sq**0.5 * hat_norm_sq**0.5)
    return {
        "alpha": float(alpha),
        "cosine": float(cos),
        "explained_fraction": float(max(0.0, cos) ** 2),
        "residual_norm": float(residual.norm()),
        "q_norm": float(q_norm_sq**0.5),
        "q_hat_norm": float(hat_norm_sq**0.5),
    }


def batched_explained_fraction(q: torch.Tensor, q_hat: torch.Tensor) -> torch.Tensor:
    """``max(0, cos)^2`` per row — the scale-fitted explained fraction."""
    return cosine(q, q_hat).clamp_min(0.0).pow(2)


def auroc(positive: list[float] | torch.Tensor, negative: list[float] | torch.Tensor) -> float | None:
    """Area under the ROC curve for two score populations (rank/Mann-Whitney).

    Ties contribute 0.5, which is what makes this the honest statistic for a
    scorer that returns the same value for many candidates — a naive
    "fraction of pairs where positive > negative" would flatter such a scorer.
    Returns ``None`` when either population is empty; a caller must then report
    "not computable", not 0.5.
    """
    pos = torch.as_tensor(list(positive), dtype=torch.float64).flatten()
    neg = torch.as_tensor(list(negative), dtype=torch.float64).flatten()
    if pos.numel() == 0 or neg.numel() == 0:
        return None
    combined = torch.cat([pos, neg])
    order = combined.argsort()
    ranks = torch.empty_like(combined)
    ranks[order] = torch.arange(1, combined.numel() + 1, dtype=torch.float64)
    # Average ranks within tie groups.
    unique, inverse, counts = torch.unique(combined, return_inverse=True, return_counts=True)
    summed = torch.zeros(unique.numel(), dtype=torch.float64)
    summed.index_add_(0, inverse, ranks)
    ranks = (summed / counts)[inverse]
    rank_sum_positive = float(ranks[: pos.numel()].sum())
    n_pos, n_neg = float(pos.numel()), float(neg.numel())
    u = rank_sum_positive - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


def rank_of(target_score: float, other_scores: list[float] | torch.Tensor) -> int:
    """0-based rank of ``target_score`` among itself and ``other_scores``,
    descending, with ties counted **against** the target.

    Ties against the target on purpose: a scorer that gives every candidate the
    same value has not retrieved anything, and should not be credited with
    rank 0.
    """
    others = torch.as_tensor(list(other_scores), dtype=torch.float64).flatten()
    return int((others >= float(target_score)).sum())


def top_k_accuracy(ranks: list[int], k: int) -> float:
    """Fraction of ranks strictly below ``k``. Empty input is an error, not 0.0."""
    if not ranks:
        raise AutoencoderError("top_k_accuracy called with no ranks")
    return sum(1 for r in ranks if r < int(k)) / len(ranks)


def mean_or_none(values: list[float]) -> float | None:
    present = [float(v) for v in values if v is not None]
    return sum(present) / len(present) if present else None
