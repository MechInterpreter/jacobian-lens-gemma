# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Finite-difference validation of the Jacobian estimator.

The fitted lens is an *averaged* Jacobian and must never be tested as if it
were an exact local derivative. These tests therefore restrict the estimator
to a **single valid source position** (skip_first = seq_len - 2, so exactly
one position carries a cotangent and the position-mean is over one term).
There the estimator equals the exact local Jacobian
``J = ∂h_final,p / ∂h_l,p`` and must match a finite perturbation of the same
source activation: inject ``ε·v`` at the block-l output via a forward hook and
compare ``(h_final(h+εv) − h_final(h)) / ε`` at position ``p`` against
``J v``.

Run on the linear TinyDecoder (exact to machine precision) and on a nonlinear
variant (first-order agreement, tolerance O(ε)). A third test documents the
averaging point directly: with several valid positions the estimator is a sum
over targets / mean over sources, and differs from any single-position local
Jacobian.
"""

from __future__ import annotations

import torch
from torch import nn

from jlens.fitting import jacobian_for_prompt

from .tiny import TinyDecoder


class _NonlinearBlock(nn.Module):
    def __init__(self, d_model: int, seed: int) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.linear = nn.Linear(d_model, d_model, bias=False)
        with torch.no_grad():
            self.linear.weight.mul_(0.3)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + torch.tanh(self.linear(hidden))


def _nonlinear_decoder(n_layers: int = 4, d_model: int = 8) -> TinyDecoder:
    model = TinyDecoder(n_layers=n_layers, d_model=d_model)
    model.layers = nn.ModuleList(
        _NonlinearBlock(d_model, seed=i) for i in range(n_layers)
    )
    # float64 so finite-difference truncation error is observable above
    # roundoff (in fp32 both sit at ~1e-5 and cannot be told apart).
    model.double()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def _single_position_setup(model: TinyDecoder, prompt: str, source_layer: int):
    """Jacobian restricted to one valid source position; returns (J, p, ids)."""
    input_ids = model.encode(prompt, max_length=64)
    seq_len = input_ids.shape[1]
    skip_first = seq_len - 2  # exactly one valid position: p = seq_len - 2
    jacobians, _, n_valid = jacobian_for_prompt(
        model,
        prompt,
        source_layers=[source_layer],
        dim_batch=4,
        max_seq_len=64,
        skip_first=skip_first,
    )
    assert n_valid == 1
    return jacobians[source_layer], seq_len - 2, input_ids


def _final_residual_with_bump(
    model: TinyDecoder,
    input_ids: torch.Tensor,
    source_layer: int,
    position: int,
    bump: torch.Tensor | None,
) -> torch.Tensor:
    """Forward pass, optionally adding ``bump`` to the block output at
    ``(0, position)``, returning the final-block residual at ``position``."""

    def hook(module, inputs, output):
        if bump is None:
            return output
        patched = output.clone()
        patched[0, position] += bump
        return patched

    handle = model.layers[source_layer].register_forward_hook(hook)
    try:
        with torch.no_grad():
            hidden = model.forward(input_ids).last_hidden_state
    finally:
        handle.remove()
    return hidden[0, position].double()


def test_fd_matches_exactly_on_linear_model():
    model = TinyDecoder(n_layers=4, d_model=8)
    for param in model.parameters():
        param.requires_grad_(False)
    prompt = "the quick brown fox jumps over the lazy dog"
    J, p, input_ids = _single_position_setup(model, prompt, source_layer=1)

    torch.manual_seed(7)
    v = torch.randn(8)
    v /= v.norm()
    epsilon = 1e-2  # blocks are linear: FD is exact for any epsilon
    base = _final_residual_with_bump(model, input_ids, 1, p, None)
    bumped = _final_residual_with_bump(model, input_ids, 1, p, epsilon * v)
    fd_direction = (bumped - base) / epsilon
    torch.testing.assert_close(J.double() @ v.double(), fd_direction, rtol=1e-4, atol=1e-4)


def test_fd_first_order_on_nonlinear_model():
    model = _nonlinear_decoder()
    prompt = "the quick brown fox jumps over the lazy dog"
    J, p, input_ids = _single_position_setup(model, prompt, source_layer=1)

    torch.manual_seed(11)
    v = torch.randn(8, dtype=torch.float64)
    v /= v.norm()
    base = _final_residual_with_bump(model, input_ids, 1, p, None)

    # jacobian_for_prompt accumulates J in fp32, so its ~1e-7 relative noise
    # is the floor; truncation error O(epsilon) sits well above it here.
    Jv = J.double() @ v
    errors = {}
    for epsilon in (1e-2, 1e-3):
        bumped = _final_residual_with_bump(model, input_ids, 1, p, epsilon * v)
        fd_direction = (bumped - base) / epsilon
        errors[epsilon] = float((Jv - fd_direction).norm() / fd_direction.norm())
    assert errors[1e-3] < 1e-3, f"first-order mismatch: {errors}"
    # The error must shrink with epsilon (i.e. it is truncation error, not an
    # orientation/indexing bug).
    assert errors[1e-3] < errors[1e-2], f"error not shrinking: {errors}"


def test_averaged_estimator_differs_from_local_jacobian():
    """The default estimator (many source positions, summed future targets) is
    NOT a local derivative; document that by construction."""
    model = _nonlinear_decoder()
    prompt = "the quick brown fox jumps over the lazy dog and runs away"
    J_local, _, _ = _single_position_setup(model, prompt, source_layer=1)
    averaged, _, n_valid = jacobian_for_prompt(
        model, prompt, source_layers=[1], dim_batch=4, max_seq_len=64, skip_first=2
    )
    assert n_valid > 1
    assert not torch.allclose(averaged[1], J_local, atol=1e-4)
