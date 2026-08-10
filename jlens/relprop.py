# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Relevance-propagation backward rules for the dense-model R-lens.

The R-lens is fitted with the same estimator as the J-lens, but replaces parts
of the ordinary backward pass with the dense-model rules described in
"R-lens: Making J-lens More Faithful on Early Layers" (Blank, Bhatia, Nanda,
2026):

* detach the denominator of residual-stream RMSNorms (LN rule);
* use the activation value divided by its input as the gated-MLP activation's
  local coefficient (identity rule);
* split relevance evenly between the two multiplicative MLP branches (half
  rule);
* leave linear layers, attention, and q/k norms on ordinary autograd.

This module changes no parameter and no forward value.  It temporarily swaps
``forward`` methods only while an R-lens graph is being built, then restores
the exact original methods even if the forward raises.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from types import MethodType

import torch

from jlens.mmpilot.store import payload_checksum

__all__ = [
    "R_LENS_METHOD",
    "R_LENS_METHOD_VERSION",
    "R_LENS_SOURCE_URL",
    "RelPropArchitectureError",
    "RelPropMethod",
    "audit_dense_relprop_architecture",
    "dense_relprop_backward",
]

R_LENS_SOURCE_URL = (
    "https://www.lesswrong.com/posts/nv8oedrnLXKRzNEL9/"
    "r-lens-making-j-lens-more-faithful-on-early-layers"
)
R_LENS_METHOD_VERSION = "jlens.relprop.dense_r_lens.v1"


class RelPropArchitectureError(RuntimeError):
    """The loaded decoder does not expose the modules the frozen rules need."""


@dataclass(frozen=True)
class RelPropMethod:
    """The complete, serialisable identity of the implemented R-lens rules."""

    version: str = R_LENS_METHOD_VERSION
    source_url: str = R_LENS_SOURCE_URL
    source_date: str = "2026-08-05"
    model_family: str = "dense decoder"
    rmsnorm_rule: str = "detach normalization denominator"
    gated_activation_rule: str = "x * stop_grad(activation(x) / x)"
    multiplicative_gate_rule: str = "half relevance to each MLP branch"
    linear_layers: str = "ordinary autograd (LRP 0-rule is identical)"
    attention: str = "ordinary autograd"
    qk_norms: str = "ordinary autograd"
    moe_routing: str = "unsupported"
    lambda_grid: tuple[float, ...] = ()
    algebraic_transform_of_raw_jacobian: bool = False
    requires_distinct_backward_accumulation: bool = True

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["lambda_grid"] = list(payload["lambda_grid"])
        return payload

    @property
    def digest(self) -> str:
        return payload_checksum(self.to_dict())


R_LENS_METHOD = RelPropMethod()


class _RelPropGatedProduct(torch.autograd.Function):
    """Exact forward product with identity-rule/half-rule derivatives."""

    @staticmethod
    def forward(ctx, gate_pre, gate_activated, up):
        # Save values, not graphs. Returning the ordinary product makes the
        # R-lens a strict forward no-op rather than merely numerically close.
        ctx.save_for_backward(gate_pre, gate_activated, up)
        return gate_activated * up

    @staticmethod
    def backward(ctx, grad_output):
        gate_pre, gate_activated, up = ctx.saved_tensors
        # GELU and SiLU both have the limit activation(x)/x -> 1/2 at zero.
        # torch.where avoids a 0/0 without altering any nonzero coefficient.
        factor = torch.where(
            gate_pre != 0,
            gate_activated / gate_pre,
            torch.full_like(gate_pre, 0.5),
        )
        gate_grad = 0.5 * grad_output * up * factor
        up_grad = 0.5 * grad_output * gate_activated
        # gate_activated is supplied only to preserve the exact forward value;
        # its ordinary activation derivative must not also propagate.
        return gate_grad, None, up_grad


_RESIDUAL_NORM_NAMES = (
    "input_layernorm",
    "post_attention_layernorm",
    "pre_feedforward_layernorm",
    "post_feedforward_layernorm",
    "post_per_layer_input_norm",
)


def _decoder_layers(model) -> list:
    layers = list(getattr(model, "layers", ()))
    if not layers:
        raise RelPropArchitectureError("model.layers is empty or unavailable")
    return layers


def audit_dense_relprop_architecture(model) -> dict:
    """Fail closed unless every decoder block supports the frozen dense rules."""
    layers = _decoder_layers(model)
    mock_linear = all(
        hasattr(layer, "linear") and not hasattr(layer, "mlp") for layer in layers
    ) and type(model).__name__ == "MockCalibrationModel"
    if mock_linear:
        payload = {
            "schema": "jlens.relprop.architecture_audit.v1",
            "method": R_LENS_METHOD.to_dict(),
            "method_digest": R_LENS_METHOD.digest,
            "n_layers": len(layers),
            "layers": [
                {"layer": index, "mock_linear_passthrough": True}
                for index in range(len(layers))
            ],
            "mock_only": True,
            "mock_note": (
                "the tiny MOCK stack has no norm or gated MLP; the R arm is a "
                "labelled passthrough that tests orchestration, not RelP math"
            ),
            "qk_norms_modified": False,
            "attention_modified": False,
            "passed": True,
        }
        payload["audit_checksum"] = payload_checksum(payload)
        return payload
    rows = []
    for index, layer in enumerate(layers):
        if bool(getattr(layer, "enable_moe_block", False)):
            raise RelPropArchitectureError(
                f"layer {index} enables MoE routing; the dense R-lens rules "
                "cannot be silently applied to it"
            )
        missing = [
            name
            for name in _RESIDUAL_NORM_NAMES[:4]
            if not hasattr(layer, name)
        ]
        mlp = getattr(layer, "mlp", None)
        for name in ("gate_proj", "up_proj", "down_proj", "act_fn"):
            if mlp is None or not hasattr(mlp, name):
                missing.append(f"mlp.{name}")
        if missing:
            raise RelPropArchitectureError(
                f"layer {index} lacks required dense R-lens component(s): {missing}"
            )
        rows.append(
            {
                "layer": index,
                "residual_norms": [
                    name for name in _RESIDUAL_NORM_NAMES if hasattr(layer, name)
                ],
                "mlp_class": type(mlp).__name__,
                "has_per_layer_input": bool(
                    getattr(layer, "hidden_size_per_layer_input", 0)
                ),
            }
        )
    payload = {
        "schema": "jlens.relprop.architecture_audit.v1",
        "method": R_LENS_METHOD.to_dict(),
        "method_digest": R_LENS_METHOD.digest,
        "n_layers": len(layers),
        "layers": rows,
        "residual_norm_scope": list(_RESIDUAL_NORM_NAMES),
        "qk_norms_modified": False,
        "attention_modified": False,
        "ple_gate_modified": False,
        "ple_note": (
            "Gemma 4 per-layer-input gating is not one of the gated MLP rules "
            "specified by the source method; its backward remains ordinary"
        ),
        "passed": True,
    }
    payload["audit_checksum"] = payload_checksum(payload)
    return payload


def _relprop_norm_forward(module, hidden_states):
    states = hidden_states.float()
    mean_squared = states.pow(2).mean(-1, keepdim=True) + module.eps
    denominator = torch.pow(mean_squared, -0.5).detach()
    output = states * denominator
    if bool(getattr(module, "with_scale", True)):
        output = output * module.weight.float()
    return output.type_as(hidden_states)


def _relprop_mlp_forward(module, hidden_states):
    gate_pre = module.gate_proj(hidden_states)
    gate_activated = module.act_fn(gate_pre)
    up = module.up_proj(hidden_states)
    product = _RelPropGatedProduct.apply(gate_pre, gate_activated, up)
    return module.down_proj(product)


@contextmanager
def dense_relprop_backward(model) -> Iterator[dict]:
    """Install the frozen dense R-lens rules for exactly one graph build.

    The yielded architecture audit is suitable for persistence beside the
    resulting accumulator. Nested installation is refused because restoring a
    nested set of instance methods is unnecessarily ambiguous.
    """
    if getattr(model, "_jlens_relprop_active", False):
        raise RuntimeError("dense_relprop_backward is already active on this model")
    audit = audit_dense_relprop_architecture(model)
    patches: list[tuple[object, str, object]] = []
    model._jlens_relprop_active = True
    try:
        for layer in _decoder_layers(model):
            if audit.get("mock_only"):
                continue
            for name in _RESIDUAL_NORM_NAMES:
                norm = getattr(layer, name, None)
                if norm is not None:
                    patches.append((norm, "forward", norm.forward))
                    norm.forward = MethodType(_relprop_norm_forward, norm)
            mlp = layer.mlp
            patches.append((mlp, "forward", mlp.forward))
            mlp.forward = MethodType(_relprop_mlp_forward, mlp)
        yield audit
    finally:
        for module, name, original in reversed(patches):
            setattr(module, name, original)
        model._jlens_relprop_active = False
