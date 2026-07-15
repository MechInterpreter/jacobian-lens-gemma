# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Negative controls for the Jacobian lens (model-agnostic).

The question a control answers: does the J-lens read out interpretable tokens
because ``J_l`` encodes *meaningful learned transport*, or would any matrix of
the right shape and scale (or the raw vocabulary geometry) do the same?

Controls, strongest first:

- :func:`permute_rows` — the **primary control**: a row-permuted copy of the
  *fitted* ``J_l``. Row ``i`` of ``J_l`` is the gradient pattern of target
  dimension ``i``; permuting rows keeps every marginal statistic of the fitted
  matrix (row norms as a set, Frobenius norm, entry distribution) while
  destroying the source→target dimension correspondence. If readouts survive
  this, they never depended on the learned transport.
- **Wrong-layer application** (:meth:`ControlSuite.wrong_layer_pairs`):
  transport ``h_l`` with ``J_{l'}`` fitted at a different layer — meaningful
  transport should be layer-specific, at least far from the final layers.
- :func:`scale_matched_random` — i.i.d. Gaussian matrix rescaled to
  ``||J_l||_F``: tests shape+scale alone.
- Identity / logit lens — upstream ``JacobianLens.apply(use_jacobian=False)``.

Metrics (:func:`topk_overlap`, :func:`ranks_of_targets`) are computed on
logits; note that a monotonic final-logit softcap (Gemma's
``30·tanh(x/30)``) cannot change rankings, so these metrics are identical for
pre-softcap and softcapped logits.
"""

from __future__ import annotations

import torch

from jlens.lens import JacobianLens


def scale_matched_random(J: torch.Tensor, *, seed: int) -> torch.Tensor:
    """An i.i.d. Gaussian matrix with the same shape and Frobenius norm as ``J``."""
    generator = torch.Generator().manual_seed(seed)
    random = torch.randn(J.shape, generator=generator, dtype=torch.float32)
    return random * (J.float().norm() / random.norm())


def permute_rows(J: torch.Tensor, *, seed: int) -> torch.Tensor:
    """A copy of ``J`` with its rows (target dimensions) randomly permuted.

    Preserves the fitted matrix's entries exactly; destroys which source
    pattern feeds which target dimension. A fixed point-free-ish permutation is
    not enforced; with ``d_model >> 1`` the expected number of fixed points is 1.
    """
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(J.shape[0], generator=generator)
    return J[perm].clone()


def control_lens(
    lens: JacobianLens,
    kind: str,
    *,
    seed: int = 0,
) -> JacobianLens:
    """A :class:`JacobianLens` whose matrices are controls derived from ``lens``.

    Args:
        lens: The fitted lens.
        kind: ``"permuted"`` (primary; :func:`permute_rows`) or ``"random"``
            (:func:`scale_matched_random`).
        seed: Base seed; layer ``l`` uses ``seed + l`` so layers differ.
    """
    makers = {"permuted": permute_rows, "random": scale_matched_random}
    if kind not in makers:
        raise ValueError(f"unknown control kind {kind!r}; expected {sorted(makers)}")
    jacobians = {
        layer: makers[kind](J, seed=seed + layer)
        for layer, J in lens.jacobians.items()
    }
    return JacobianLens(
        jacobians=jacobians, n_prompts=lens.n_prompts, d_model=lens.d_model
    )


def wrong_layer_lens(lens: JacobianLens) -> JacobianLens:
    """A lens whose fitted layers are cyclically reassigned (``J`` fitted at
    layer ``l`` is applied at the next fitted layer), so each residual is
    transported with a genuine-but-mismatched Jacobian."""
    layers = lens.source_layers
    if len(layers) < 2:
        raise ValueError("wrong_layer_lens needs a lens fitted at >= 2 layers")
    shifted = {
        layers[i]: lens.jacobians[layers[(i + 1) % len(layers)]].clone()
        for i in range(len(layers))
    }
    return JacobianLens(
        jacobians=shifted, n_prompts=lens.n_prompts, d_model=lens.d_model
    )


def topk_overlap(logits_a: torch.Tensor, logits_b: torch.Tensor, k: int) -> float:
    """Mean Jaccard-free overlap ``|topk(a) ∩ topk(b)| / k`` over leading dims.

    Both tensors are ``[..., vocab]`` with identical leading shapes.
    """
    top_a = logits_a.topk(k, dim=-1).indices
    top_b = logits_b.topk(k, dim=-1).indices
    matches = (top_a.unsqueeze(-1) == top_b.unsqueeze(-2)).any(-1).sum(-1).float()
    return float((matches / k).mean())


def ranks_of_targets(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    """Rank (0 = argmax) of ``target_ids[i]`` under ``logits[i]``.

    Args:
        logits: ``[n, vocab]``.
        target_ids: ``[n]`` token ids.
    """
    target_scores = logits.gather(-1, target_ids.unsqueeze(-1))
    return (logits > target_scores).sum(-1)
