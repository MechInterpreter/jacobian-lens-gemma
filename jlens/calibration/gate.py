# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""The predeclared validity gate for calibrated layers, and target diversity.

This gate is the localization Stage-B gate re-parameterized, not a new idea.
Every rank convention, every tie definition and every threshold's *meaning*
comes from :mod:`jlens.mmlocalize.lens_validity`, and the scoring rows are
produced by its :func:`~jlens.mmlocalize.lens_validity.tie_aware_row` — so a
number here means exactly what the same number meant in the completed
localization run. What changes:

* **128 held-out prompts instead of 32**, at both validation and confirmation.
  Validation is a forward-pass workload, so four times the sample costs minutes.
* **One added blocking clause: target-token share.** A distinct-count floor
  alone is satisfiable by a sample that is 60% ``the`` plus a long tail. Over
  128 prompts that would pass a "24 distinct targets" test while measuring
  almost nothing, so no single target may exceed 25% of prompts.

The added clause is strictly harder than the gate it extends. Nothing is
relaxed.

Target selection never consults a lens. :func:`select_diverse_validation_prompts`
takes a callable that runs the frozen model's ordinary output path and nothing
else; there is no parameter through which a J-lens could enter. This is
structural, not a convention — "do not select examples using J-lens
performance" is enforced by the signature.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass

from jlens.mmlocalize.lens_validity import (
    CONTROL_VARIANTS,
    CRITERION_RANK_CONVENTION,
    DIAGNOSTIC_VARIANTS,
    LAYER_ELIGIBLE,
    LAYER_INELIGIBLE,
    LEGACY_VALIDITY_GATE,
    NOISE_CONTROLS,
    RANK_CONVENTIONS,
    READOUT_VARIANTS,
    InsufficientTargetDiversityError,
    LegacyValidityGate,
    evaluate_legacy_gate,
    fold_mrr,
    select_target_diverse_prompts,
    summarize_layer,
    tie_aware_row,
)
from jlens.mmpilot.store import payload_checksum

__all__ = [
    "CALIBRATION_GATE",
    "CALIBRATION_VALIDITY_PROTOCOL",
    "CONTROL_SEED",
    "CONTROL_VARIANTS",
    "DIAGNOSTIC_VARIANTS",
    "LAYER_ELIGIBLE",
    "LAYER_INELIGIBLE",
    "N_CONFIRMATION_PROMPTS",
    "N_VALIDATION_PROMPTS",
    "RANK_CONVENTIONS",
    "READOUT_VARIANTS",
    "CalibrationGate",
    "InsufficientTargetDiversityError",
    "audit_target_diversity",
    "evaluate_calibration_layer",
    "evaluate_calibration_layers",
    "eligible_layers",
    "gate_text",
    "select_diverse_validation_prompts",
    "tie_aware_row",
]

#: Protocol tag. Distinct from the localization, v2 and layer-32 tags: results
#: under this gate must never be mistaken for results under those.
CALIBRATION_VALIDITY_PROTOCOL = "research-grade-calibration-tie-aware-native-readout-v1"

N_VALIDATION_PROMPTS = 128
N_CONFIRMATION_PROMPTS = 128

#: Matches every earlier run's control seed, so a control number here means
#: what a control number meant there.
CONTROL_SEED = 1234

VALIDATION_PROMPT_SEED = 20260805
CONFIRMATION_PROMPT_SEED = 20260806

TARGET_SELECTION_PROTOCOL = "target-token-stratified-stable-rank-v1"


@dataclass(frozen=True)
class CalibrationGate:
    """The pass rule for one layer at one scale, fixed before any number exists.

    Every threshold is stated against the **midrank** convention. See
    :mod:`jlens.mmlocalize.lens_validity` for why midrank is the only rank that
    can carry a threshold in the presence of ties.

    Attributes:
        max_target_token_share: The clause this gate adds. Ceiling on the
            fraction of prompts sharing a single target token, so a
            distinct-count floor cannot be satisfied by a dominated sample.
    """

    n_prompts: int = N_VALIDATION_PROMPTS
    rank_convention: str = CRITERION_RANK_CONVENTION
    max_tied_at_max_rate: float = 0.50
    min_noise_control_mrr_ratio: float = 1.5
    min_noise_control_mrr_margin: float = 0.10
    min_wrong_layer_mrr_margin: float = 0.15
    max_median_midrank: float = 5.0
    min_top_k_inclusion: float = 0.50
    top_k: int = 10
    n_folds: int = 4
    min_fold_mrr_fraction: float = 0.50
    min_distinct_target_tokens: int = 24
    max_target_token_share: float = 0.25
    controls: tuple[str, ...] = CONTROL_VARIANTS
    noise_controls: tuple[str, ...] = NOISE_CONTROLS
    wrong_layer_mapping: str = "distant_layer_mapping"
    version: str = CALIBRATION_VALIDITY_PROTOCOL

    def to_dict(self) -> dict:
        payload = asdict(self)
        for key in ("controls", "noise_controls"):
            payload[key] = list(payload[key])
        return payload

    @property
    def digest(self) -> str:
        """Checksum of the gate, bound into the resume fingerprint."""
        return payload_checksum(self.to_dict())

    def for_confirmation(self) -> CalibrationGate:
        """The same gate, unchanged, at the confirmation sample size.

        Deliberately not a *different* rule: publication must clear the rule
        the study was designed around, on data it never saw.
        """
        return CalibrationGate(**{**asdict(self), "n_prompts": N_CONFIRMATION_PROMPTS})


#: The single gate this workflow runs under.
CALIBRATION_GATE = CalibrationGate()


def gate_text(
    gate: CalibrationGate = CALIBRATION_GATE,
    legacy: LegacyValidityGate = LEGACY_VALIDITY_GATE,
) -> str:
    """The criterion block, printed before any result-producing cell runs."""
    return f"""\
PREDECLARED CALIBRATION VALIDITY GATE — {gate.version}
Fixed before any number is computed. Not revisable after seeing results.
Criterion digest: {gate.digest}

Every threshold is stated against the MIDRANK convention:
    midrank = (tokens strictly above the target)
              + (tokens tied with the target + 1) / 2
Optimistic and pessimistic ranks are computed and reported; nothing is decided
on them.

A layer PASSES at a scale point only if ALL of the following hold, over exactly
{gate.n_prompts} held-out prompts the lens never saw:

  1. coverage and non-degeneracy
       every variant scored on all {gate.n_prompts} prompts, every metric finite,
       at least {gate.min_distinct_target_tokens} distinct target tokens,
       NO single target token above {gate.max_target_token_share:.0%} of prompts,
       and J-lens tied-at-maximum rate at most {gate.max_tied_at_max_rate:.2f}.
  2. beats the noise controls
       J-lens MRR >= {gate.min_noise_control_mrr_ratio:.1f}x AND >= +{gate.min_noise_control_mrr_margin:.2f}
       over BOTH the row-permuted and the norm-matched-random control.
  3. beats the wrong-layer control
       J-lens MRR >= wrong-layer MRR + {gate.min_wrong_layer_mrr_margin:.2f},
       using {gate.wrong_layer_mapping} (not the deprecated cyclic control).
  4. rank and top-k
       median midrank <= {gate.max_median_midrank:.1f} AND top-{gate.top_k} inclusion
       >= {gate.min_top_k_inclusion:.2f}, inclusion at the PESSIMISTIC rank so a tie block
       cannot manufacture membership.
  5. stability across held-out subsets
       over {gate.n_folds} fixed folds (prompt i -> fold i mod {gate.n_folds}), every fold's
       J-lens MRR beats every control on that fold, and no fold falls below
       {gate.min_fold_mrr_fraction:.2f}x the overall MRR.

WHAT THIS GATE ADDS over the localization Stage-B gate: clause 1's target-share
ceiling. A distinct-count floor alone is satisfiable by a sample dominated by
one common token plus a long tail, which would pass while measuring almost
nothing. Nothing is relaxed; the added clause is strictly harder.

Reported, never blocking: unique and argmax top-1 agreement, tie-block sizes,
mean top-{gate.top_k} overlap, margin over the strongest non-target token, the
ordinary logit lens as a diagnostic, and the legacy v2 conjunction
({legacy.name}) computed beside this gate for every layer.

PASSING AT A SCALE POINT IS NOT PUBLICATION. Publication requires passing this
same gate on the untouched confirmation set, after the scale is chosen.
"""


# ------------------------------------------------------------ target diversity


def audit_target_diversity(
    target_token_ids: Sequence[int], *, gate: CalibrationGate = CALIBRATION_GATE
) -> dict:
    """Distinct-count and dominance statistics for a set of targets.

    Returns a report with ``passed``; it does not raise. The raising path is
    :func:`select_diverse_validation_prompts`, which refuses *before* any
    readout is scored.
    """
    ids = [int(value) for value in target_token_ids]
    n = len(ids)
    counts: dict[int, int] = {}
    for token_id in ids:
        counts[token_id] = counts.get(token_id, 0) + 1
    top_id, top_count = (
        max(counts.items(), key=lambda item: (item[1], -item[0]))
        if counts
        else (None, 0)
    )
    max_share = (top_count / n) if n else 0.0
    distinct_ok = len(counts) >= gate.min_distinct_target_tokens
    share_ok = max_share <= gate.max_target_token_share
    report = {
        "n_prompts": n,
        "n_distinct_target_tokens": len(counts),
        "min_distinct_target_tokens": gate.min_distinct_target_tokens,
        "distinct_ok": bool(distinct_ok),
        "most_common_target_token_id": top_id,
        "most_common_target_count": int(top_count),
        "max_target_token_share": round(max_share, 6),
        "max_target_token_share_allowed": gate.max_target_token_share,
        "share_ok": bool(share_ok),
        "passed": bool(distinct_ok and share_ok),
        "counts": {str(key): value for key, value in sorted(counts.items())},
    }
    report["diversity_checksum"] = payload_checksum(
        {key: value for key, value in report.items() if key != "counts"}
    )
    return report


def select_diverse_validation_prompts(
    pool: Sequence[str],
    *,
    n_prompts: int,
    gate: CalibrationGate = CALIBRATION_GATE,
    excluded: Mapping[str, str] | None = None,
    seed: int = VALIDATION_PROMPT_SEED,
    target_token_for_prompt: Callable[[str], int],
) -> tuple[list[str], dict]:
    """Choose a deterministic held-out set meeting both diversity clauses.

    ``target_token_for_prompt`` runs the frozen model's ordinary output path.
    **There is no parameter through which a J-lens or a candidate layer could
    be supplied**, so selection cannot be contaminated by the thing under test.

    Raises:
        InsufficientTargetDiversityError: If the pool cannot meet the distinct
            floor, or if the selected set is dominated by one target token.
            The threshold is never lowered to make a run proceed.
    """
    prompts, manifest = select_target_diverse_prompts(
        pool,
        n_prompts=n_prompts,
        min_distinct_target_tokens=gate.min_distinct_target_tokens,
        excluded=dict(excluded or {}),
        seed=seed,
        target_token_for_prompt=target_token_for_prompt,
    )
    diversity = audit_target_diversity(
        [row["target_token_id"] for row in manifest["prompts"]], gate=gate
    )
    if not diversity["share_ok"]:
        raise InsufficientTargetDiversityError(
            f"the selected {len(prompts)}-prompt set is dominated by target token "
            f"{diversity['most_common_target_token_id']} "
            f"({diversity['most_common_target_count']}/{diversity['n_prompts']} = "
            f"{diversity['max_target_token_share']:.1%}); the fixed gate allows at "
            f"most {gate.max_target_token_share:.0%}. Refusing before any J-lens "
            "readout is scored — this threshold is not lowered."
        )
    manifest = dict(manifest)
    manifest["gate_version"] = gate.version
    manifest["gate_digest"] = gate.digest
    manifest["diversity"] = diversity
    manifest["selected_by_jlens_performance"] = False
    manifest["selection_checksum"] = payload_checksum(
        {key: value for key, value in manifest.items() if key != "selection_checksum"}
    )
    return prompts, manifest


# ------------------------------------------------------------------- verdicts


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def evaluate_calibration_layer(
    rows: Sequence[Mapping],
    *,
    layer: int,
    scale: int,
    stage: str = "validation",
    gate: CalibrationGate = CALIBRATION_GATE,
    legacy: LegacyValidityGate = LEGACY_VALIDITY_GATE,
) -> dict:
    """Apply the gate to one layer at one scale point.

    Every clause reports its own flag and its own numbers, so a failure names
    the clause rather than the layer.
    """
    layer_rows = [row for row in rows if int(row["layer"]) == int(layer)]
    metrics = summarize_layer(layer_rows, layer=layer, top_k=gate.top_k)
    jlens = metrics["j_lens"]
    noise = {name: metrics[name] for name in gate.noise_controls}
    wrong_layer = metrics["wrong_layer"]

    targets = [
        int(row["target_token_id"]) for row in layer_rows if row["variant"] == "j_lens"
    ]
    diversity = audit_target_diversity(targets, gate=gate)

    all_finite = all(
        _finite(value)
        for variant in metrics.values()
        for key, value in variant.items()
        if key != "n_prompts"
    )
    full_coverage = all(
        variant["n_prompts"] == gate.n_prompts for variant in metrics.values()
    )
    inclusion_key = f"top{gate.top_k}_inclusion"

    folds = {
        variant: fold_mrr(layer_rows, variant=variant, n_folds=gate.n_folds)
        for variant in ("j_lens", *gate.controls)
    }
    fold_beats_controls = {
        fold: all(folds["j_lens"][fold] > folds[control][fold] for control in gate.controls)
        for fold in range(gate.n_folds)
    }
    overall_mrr = jlens["mean_reciprocal_rank"]
    fold_floor = gate.min_fold_mrr_fraction * overall_mrr
    fold_above_floor = {
        fold: bool(value >= fold_floor) for fold, value in folds["j_lens"].items()
    }

    checks: list[dict] = [
        {
            "check": "coverage_and_nondegeneracy",
            "passed": bool(
                all_finite
                and full_coverage
                and diversity["passed"]
                and jlens["tied_at_max_rate"] <= gate.max_tied_at_max_rate
            ),
            "detail": (
                f"finite={all_finite}, every variant on {gate.n_prompts} "
                f"prompts={full_coverage}, distinct targets="
                f"{diversity['n_distinct_target_tokens']} "
                f"(need >= {gate.min_distinct_target_tokens}), max target share="
                f"{diversity['max_target_token_share']:.3f} "
                f"(ceiling {gate.max_target_token_share:.2f}), tied-at-max rate "
                f"{jlens['tied_at_max_rate']:.3f} (ceiling "
                f"{gate.max_tied_at_max_rate:.2f}; mean tie block "
                f"{jlens['mean_n_tied_at_max']:.1f} tokens)"
            ),
        },
        {
            "check": "mrr_beats_noise_controls",
            "passed": all(
                overall_mrr
                >= gate.min_noise_control_mrr_ratio * control["mean_reciprocal_rank"]
                and overall_mrr
                >= control["mean_reciprocal_rank"] + gate.min_noise_control_mrr_margin
                for control in noise.values()
            ),
            "detail": (
                f"J-lens MRR {overall_mrr:.5f} vs "
                + ", ".join(
                    f"{name}={control['mean_reciprocal_rank']:.5f}"
                    for name, control in noise.items()
                )
                + f" (need >= {gate.min_noise_control_mrr_ratio:.1f}x and "
                f">= +{gate.min_noise_control_mrr_margin:.2f})"
            ),
        },
        {
            "check": "mrr_beats_wrong_layer_by_margin",
            "passed": bool(
                overall_mrr
                >= wrong_layer["mean_reciprocal_rank"] + gate.min_wrong_layer_mrr_margin
            ),
            "detail": (
                f"J-lens MRR {overall_mrr:.5f} vs wrong-layer "
                f"{wrong_layer['mean_reciprocal_rank']:.5f} "
                f"({gate.wrong_layer_mapping}); need "
                f"+{gate.min_wrong_layer_mrr_margin:.2f}"
            ),
        },
        {
            "check": "median_rank_and_top_k",
            "passed": bool(
                jlens["median_midrank"] <= gate.max_median_midrank
                and jlens[inclusion_key] >= gate.min_top_k_inclusion
            ),
            "detail": (
                f"median midrank {jlens['median_midrank']:.2f} (ceiling "
                f"{gate.max_median_midrank:.1f}), top-{gate.top_k} inclusion "
                f"{jlens[inclusion_key]:.3f} (floor {gate.min_top_k_inclusion:.2f}); "
                f"optimistic median {jlens['median_optimistic_rank']:.2f}, "
                f"pessimistic {jlens['median_pessimistic_rank']:.2f}"
            ),
        },
        {
            "check": "stable_across_heldout_subsets",
            "passed": bool(
                all(fold_beats_controls.values()) and all(fold_above_floor.values())
            ),
            "detail": (
                f"{sum(fold_beats_controls.values())}/{gate.n_folds} folds beat every "
                f"control, {sum(fold_above_floor.values())}/{gate.n_folds} at or above "
                f"{fold_floor:.4f} ({gate.min_fold_mrr_fraction:.2f}x overall MRR); "
                + ", ".join(
                    f"f{fold}={value:.4f}"
                    for fold, value in sorted(folds["j_lens"].items())
                )
            ),
        },
    ]

    passed = all(check["passed"] for check in checks)
    legacy_result = evaluate_legacy_gate(metrics, legacy=legacy, top_k=gate.top_k)
    return {
        "protocol": gate.version,
        "stage": stage,
        "scale": int(scale),
        "layer": int(layer),
        "status": LAYER_ELIGIBLE if passed else LAYER_INELIGIBLE,
        "passed": passed,
        "rank_convention": gate.rank_convention,
        "rank_conventions_reported": list(RANK_CONVENTIONS),
        "gate": gate.to_dict(),
        "gate_digest": gate.digest,
        "checks": checks,
        "failed_checks": [check["check"] for check in checks if not check["passed"]],
        "metrics": metrics,
        "fold_mrr": {name: dict(sorted(value.items())) for name, value in folds.items()},
        "fold_beats_all_controls": dict(sorted(fold_beats_controls.items())),
        "target_diversity": diversity,
        "legacy_gate": legacy_result,
        "gates_agree": bool(passed == legacy_result["passed"]),
        "diagnostic_variants": list(DIAGNOSTIC_VARIANTS),
        "secondary_metrics_are_non_blocking": True,
        "publication_requires_confirmation": True,
    }


def evaluate_calibration_layers(
    rows: Sequence[Mapping],
    *,
    layers: Sequence[int],
    scale: int,
    stage: str = "validation",
    gate: CalibrationGate = CALIBRATION_GATE,
) -> dict[int, dict]:
    """One verdict per layer, keyed by physical layer."""
    return {
        int(layer): evaluate_calibration_layer(
            rows, layer=layer, scale=scale, stage=stage, gate=gate
        )
        for layer in layers
    }


def eligible_layers(results: Mapping[int, Mapping]) -> list[int]:
    """The layers that passed, shallow to deep."""
    return sorted(int(layer) for layer, result in results.items() if result["passed"])
