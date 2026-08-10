# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Predeclared raw-J and R-lens arms for the L27–L31 transition study."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass

from jlens.calibration.adjacent import select_earliest_confirmed_layer
from jlens.mmpilot.store import payload_checksum
from jlens.relprop import R_LENS_METHOD

__all__ = [
    "DUAL_ARM_RUNTIME_ANCHOR",
    "LENS_ARM_PROTOCOL",
    "LENS_ARM_PROTOCOL_VERSION",
    "RAW_J_ARM",
    "R_LENS_ARM",
    "LensArm",
    "dual_arm_fitting_budget",
    "format_dual_arm_fitting_budget",
    "select_both_lens_arms",
]

LENS_ARM_PROTOCOL_VERSION = "jlens.calibration.raw_j_and_r_lens_arms.v1"


@dataclass(frozen=True)
class LensArm:
    name: str
    label: str
    scientific_role: str
    backward_rule: str
    method_digest: str
    space_name: str
    primary: bool

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def digest(self) -> str:
        return payload_checksum(self.to_dict())


RAW_J_ARM = LensArm(
    name="raw_j",
    label="RAW_J_LENS",
    scientific_role="primary Anthropic-faithful arm",
    backward_rule="ordinary_autograd",
    method_digest=payload_checksum(
        {
            "version": "anthropic_j_lens_raw_average_jacobian.v1",
            "estimator": "ordinary input-output Jacobian averaged over prompts",
        }
    ),
    space_name="J_SPACE",
    primary=True,
)

R_LENS_ARM = LensArm(
    name="r_lens",
    label="R_LENS",
    scientific_role="secondary RelP extension arm",
    backward_rule=R_LENS_METHOD.version,
    method_digest=R_LENS_METHOD.digest,
    space_name="R_SPACE",
    primary=False,
)

LENS_ARM_PROTOCOL = {
    "version": LENS_ARM_PROTOCOL_VERSION,
    "arms": [RAW_J_ARM.to_dict(), R_LENS_ARM.to_dict()],
    "fit_order": [RAW_J_ARM.name, R_LENS_ARM.name],
    "same_text_prompts": True,
    "same_development_prompts": True,
    "same_untouched_confirmation_prompts": True,
    "confirmation_retuning_allowed": False,
    "raw_j_is_primary": True,
    "r_lens_is_algebraic_transform_of_raw_j": False,
    "lambda_grid": [],
}
LENS_ARM_PROTOCOL["digest"] = payload_checksum(LENS_ARM_PROTOCOL)


# The earlier 7.1-minute number measured fitting an already accumulated
# Jacobian and is not a wall-clock anchor for fresh Jacobian accumulation.
# The operator subsequently measured about 80 seconds per prompt for a fresh
# span from L8 to L41 on one L4.  Use that direct observation for planning this
# notebook's two independent backward accumulations.
DUAL_ARM_RUNTIME_ANCHOR = {
    "seconds_per_prompt": 80.0,
    "backward_span_blocks": 33,
    "device": "one NVIDIA L4",
    "measurement": "fresh research-grade Jacobian accumulation",
}


def dual_arm_fitting_budget(
    *,
    scale: int,
    layers: tuple[int, ...],
    target_layer: int,
) -> dict:
    """Conservative planning budget for separate raw-J and RelP fits.

    The arms cannot share an accumulator because their backward rules differ.
    The estimate scales the measured per-prompt time by backward span, then
    reports a deliberately broad 2--4 hour band per arm on an L4.
    """
    span = int(target_layer) - min(int(layer) for layer in layers)
    measured_seconds = (
        DUAL_ARM_RUNTIME_ANCHOR["seconds_per_prompt"]
        * span
        / DUAL_ARM_RUNTIME_ANCHOR["backward_span_blocks"]
        * int(scale)
    )
    empirical_hours = measured_seconds / 3600.0
    per_arm_low = max(2.0, empirical_hours * 0.8)
    per_arm_high = max(4.0, empirical_hours * 1.6)
    payload = {
        "schema": "jlens.calibration.dual_arm_fitting_budget.v1",
        "scale": int(scale),
        "layers": [int(layer) for layer in layers],
        "target_layer": int(target_layer),
        "backward_span_blocks": span,
        "n_independent_accumulations": 2,
        "raw_j_backward_rule": RAW_J_ARM.backward_rule,
        "r_lens_backward_rule": R_LENS_ARM.backward_rule,
        "accumulators_shared": False,
        "anchor": dict(DUAL_ARM_RUNTIME_ANCHOR),
        "per_arm_hours_low": round(per_arm_low, 2),
        "per_arm_hours_high": round(per_arm_high, 2),
        "total_hours_low": round(2 * per_arm_low, 2),
        "total_hours_high": round(2 * per_arm_high, 2),
        "raw_j_forward_passes": int(scale),
        "r_lens_forward_passes": int(scale),
        "total_forward_passes": 2 * int(scale),
        "total_backward_block_steps": 2 * int(scale) * span,
    }
    payload["budget_checksum"] = payload_checksum(payload)
    return payload


def format_dual_arm_fitting_budget(budget: Mapping) -> str:
    """Render the budget before either expensive fitting arm can run."""
    return f"""\
DUAL-ARM FITTING BUDGET -- raw J primary, R-lens secondary

  layers / target       {budget['layers']} -> L{budget['target_layer']}
  prompts per arm       {budget['scale']:,}
  independent fits      {budget['n_independent_accumulations']} (no shared accumulator)
  total forward passes  {budget['total_forward_passes']:,}
  backward block steps  {budget['total_backward_block_steps']:,}

  L4 planning time      {budget['per_arm_hours_low']:.1f}-{budget['per_arm_hours_high']:.1f} h per arm
                        {budget['total_hours_low']:.1f}-{budget['total_hours_high']:.1f} h total

Raw J uses ordinary autograd. R-lens repeats the same prompts with the frozen
RelP backward rules. The two outputs are evaluated independently and never
pooled. This is a conservative planning range, not a completion promise.
"""


def select_both_lens_arms(
    *,
    raw_confirmation: Mapping[int, Mapping],
    r_confirmation: Mapping[int, Mapping],
    raw_development: Mapping[int, Mapping] | None = None,
    r_development: Mapping[int, Mapping] | None = None,
) -> dict:
    """Apply the unchanged earliest-passer rule independently to both arms."""
    raw = select_earliest_confirmed_layer(
        raw_confirmation, development=raw_development
    )
    relevance = select_earliest_confirmed_layer(
        r_confirmation, development=r_development
    )
    if raw.get("selected_layer") is not None:
        headline = "RAW_J_LENS_EARLY_LAYER_GO"
    elif relevance.get("selected_layer") is not None:
        headline = "R_LENS_EARLY_LAYER_GO"
    else:
        headline = "BOTH_LENS_ARMS_NO_GO"
    payload = {
        "schema": "jlens.calibration.dual_lens_selection.v1",
        "protocol": LENS_ARM_PROTOCOL,
        "raw_j": raw,
        "r_lens": relevance,
        "raw_j_selected_layer": raw.get("selected_layer"),
        "r_lens_selected_layer": relevance.get("selected_layer"),
        "headline": headline,
        "arms_pooled": False,
        "claim_boundary": (
            "A result supported only by the R-lens is an R-space result, not "
            "evidence that the raw Anthropic J-lens passed at that layer."
        ),
    }
    payload["selection_checksum"] = payload_checksum(payload)
    return payload
