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

from collections.abc import Sequence

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
    transported with a genuine-but-mismatched Jacobian.

    .. note:: Superseded for new evaluations. The cyclic shift conflates
        qualitatively different substitutions — adjacent late layers get a
        near-equivalent transport while the last fitted layer gets the most
        distant one (see ``docs/pilot_report.md``, "Audit of the wrong-layer
        control"). Kept unchanged because the smoke and pilot artifacts were
        produced with it; use :func:`layer_mapped_lens` with
        :func:`adjacent_layer_mapping` / :func:`distant_layer_mapping` /
        :func:`shuffled_layer_mapping` going forward.
    """
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


def wrong_layer_mapping(layers: Sequence[int]) -> dict[int, int]:
    """The layer reassignment :func:`wrong_layer_lens` performs, made
    explicit: ``{applied_layer: layer_whose_J_is_used}`` under the cyclic
    shift. Exists so the historical control's provenance can be recorded and
    audited without re-deriving it from the implementation."""
    layers = sorted(layers)
    if len(layers) < 2:
        raise ValueError("wrong_layer_mapping needs >= 2 layers")
    return {
        layers[i]: layers[(i + 1) % len(layers)] for i in range(len(layers))
    }


def adjacent_layer_mapping(layers: Sequence[int]) -> dict[int, int]:
    """Map each fitted layer to its *nearest other* fitted layer (ties break
    toward the deeper layer). A weak mismatch control: transports each
    residual with the most similar genuine-but-wrong Jacobian available."""
    layers = sorted(layers)
    if len(layers) < 2:
        raise ValueError("adjacent_layer_mapping needs >= 2 layers")
    mapping: dict[int, int] = {}
    for layer in layers:
        others = [l for l in layers if l != layer]
        mapping[layer] = min(others, key=lambda l: (abs(l - layer), -l))
    return mapping


def distant_layer_mapping(layers: Sequence[int]) -> dict[int, int]:
    """Map each fitted layer to the *farthest* fitted layer (ties break
    toward the shallower layer). A strong mismatch control: transports each
    residual with the least similar genuine Jacobian available."""
    layers = sorted(layers)
    if len(layers) < 2:
        raise ValueError("distant_layer_mapping needs >= 2 layers")
    mapping: dict[int, int] = {}
    for layer in layers:
        others = [l for l in layers if l != layer]
        mapping[layer] = max(others, key=lambda l: (abs(l - layer), -l))
    return mapping


def shuffled_layer_mapping(layers: Sequence[int], *, seed: int) -> dict[int, int]:
    """A seeded derangement of the fitted layers (no layer keeps its own
    Jacobian), deterministic for a given ``(layers, seed)``. Mixes distances
    without the cyclic shift's systematic wraparound artifact."""
    layers = sorted(layers)
    if len(layers) < 2:
        raise ValueError("shuffled_layer_mapping needs >= 2 layers")
    generator = torch.Generator().manual_seed(seed)
    while True:
        perm = torch.randperm(len(layers), generator=generator).tolist()
        if all(perm[i] != i for i in range(len(layers))):
            return {layers[i]: layers[perm[i]] for i in range(len(layers))}


def layer_mapped_lens(
    lens: JacobianLens, mapping: dict[int, int]
) -> JacobianLens:
    """A lens where the residual at layer ``l`` is transported with the
    Jacobian fitted at ``mapping[l]``. Every key and value must be a fitted
    layer; the mapping is the control's provenance and should be recorded in
    output metadata (see :func:`mapping_provenance`)."""
    fitted = set(lens.source_layers)
    bad = sorted(set(mapping) - fitted) + sorted(set(mapping.values()) - fitted)
    if bad:
        raise ValueError(
            f"mapping references non-fitted layers {sorted(set(bad))}; "
            f"fitted layers are {lens.source_layers}"
        )
    if set(mapping) != fitted:
        raise ValueError(
            f"mapping keys {sorted(mapping)} must cover exactly the fitted "
            f"layers {lens.source_layers}"
        )
    remapped = {
        layer: lens.jacobians[mapping[layer]].clone() for layer in mapping
    }
    return JacobianLens(
        jacobians=remapped, n_prompts=lens.n_prompts, d_model=lens.d_model
    )


def mapping_provenance(mapping: dict[int, int]) -> list[dict]:
    """Serializable provenance rows for a layer-mapped control:
    which fitted Jacobian was used at which layer, and how far apart the
    intended and substituted layers are."""
    return [
        {
            "applied_at_layer": layer,
            "jacobian_fitted_at_layer": source,
            "layer_distance": abs(layer - source),
        }
        for layer, source in sorted(mapping.items())
    ]


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
