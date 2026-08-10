# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
"""Dense R-lens rules: exact forwards, specified backwards, safe restoration."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from jlens.relprop import (
    R_LENS_METHOD,
    RelPropArchitectureError,
    audit_dense_relprop_architecture,
    dense_relprop_backward,
)


class Norm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.eps = 1e-6
        self.with_scale = True
        self.weight = nn.Parameter(torch.linspace(0.8, 1.2, d_model))

    def forward(self, x):
        mean_squared = x.float().pow(2).mean(-1, keepdim=True) + self.eps
        out = x.float() * torch.pow(mean_squared, -0.5)
        return (out * self.weight.float()).type_as(x)


class MLP(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_model, bias=False)
        self.up_proj = nn.Linear(d_model, d_model, bias=False)
        self.down_proj = nn.Linear(d_model, d_model, bias=False)
        self.act_fn = torch.nn.functional.silu

    def forward(self, x):
        return self.down_proj(
            self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        )


class Block(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.enable_moe_block = False
        self.input_layernorm = Norm(d_model)
        self.post_attention_layernorm = Norm(d_model)
        self.pre_feedforward_layernorm = Norm(d_model)
        self.post_feedforward_layernorm = Norm(d_model)
        self.mlp = MLP(d_model)
        self.hidden_size_per_layer_input = 0

    def forward(self, x):
        residual = x
        x = self.input_layernorm(x)
        x = self.post_attention_layernorm(x)
        x = residual + x
        residual = x
        x = self.pre_feedforward_layernorm(x)
        x = self.mlp(x)
        x = self.post_feedforward_layernorm(x)
        return residual + x


def model():
    torch.manual_seed(3)
    return SimpleNamespace(layers=nn.ModuleList([Block(4), Block(4)]))


def test_method_has_no_invented_lambda_grid():
    assert R_LENS_METHOD.lambda_grid == ()
    assert R_LENS_METHOD.requires_distinct_backward_accumulation is True
    assert R_LENS_METHOD.algebraic_transform_of_raw_jacobian is False


def test_context_is_an_exact_forward_noop_and_changes_backward():
    m = model()
    x = torch.tensor([[[-1.0, -0.25, 0.5, 1.5]]], requires_grad=True)
    raw = m.layers[0](x)
    raw_grad = torch.autograd.grad(raw.sum(), x)[0]

    x_r = x.detach().clone().requires_grad_(True)
    with dense_relprop_backward(m) as audit:
        relevance = m.layers[0](x_r)
        relevance_grad = torch.autograd.grad(relevance.sum(), x_r)[0]
    assert audit["passed"] is True
    torch.testing.assert_close(relevance, raw, rtol=0, atol=0)
    assert not torch.allclose(relevance_grad, raw_grad)


def test_context_restores_every_forward_after_exception():
    m = model()
    originals = [
        (layer.mlp.forward, layer.input_layernorm.forward) for layer in m.layers
    ]
    with pytest.raises(RuntimeError, match="boom"):
        with dense_relprop_backward(m):
            raise RuntimeError("boom")
    for layer, (mlp_forward, norm_forward) in zip(m.layers, originals, strict=True):
        assert layer.mlp.forward == mlp_forward
        assert layer.input_layernorm.forward == norm_forward


def test_nested_context_is_refused():
    m = model()
    with dense_relprop_backward(m):
        with pytest.raises(RuntimeError, match="already active"):
            with dense_relprop_backward(m):
                pass


def test_architecture_audit_refuses_moe():
    m = model()
    m.layers[1].enable_moe_block = True
    with pytest.raises(RelPropArchitectureError, match="MoE"):
        audit_dense_relprop_architecture(m)


def test_half_rule_splits_product_branch_relevance_evenly():
    m = model()
    mlp = m.layers[0].mlp
    with torch.no_grad():
        for matrix in (mlp.gate_proj.weight, mlp.up_proj.weight, mlp.down_proj.weight):
            matrix.copy_(torch.eye(4))
    x = torch.tensor([[[0.0, 0.5, -1.0, 2.0]]], requires_grad=True)
    with dense_relprop_backward(m):
        y = mlp(x)
        grad = torch.autograd.grad(y.sum(), x)[0]
    activated = torch.nn.functional.silu(x.detach())
    factor = torch.where(
        x.detach() != 0,
        activated / x.detach(),
        torch.full_like(x.detach(), 0.5),
    )
    expected = 0.5 * x.detach() * factor + 0.5 * activated
    torch.testing.assert_close(grad, expected)
