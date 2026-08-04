# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Stages C and D: the paired depth contrast, the budget, and the rubric.

One question, asked of four predetermined layers on one frozen set of
photographs: **among these layers, what is the earliest at which the confirmed
text-image transfer is present under its own controls?**

Three properties of the rubric are worth stating plainly.

**Layer 38 is the anchor, not a competitor.** If the established result does not
reproduce on the localization subset, no earlier layer's number means anything —
there is nothing to be earlier *than*. That case is
:data:`INCONCLUSIVE_LAYER_LOCALIZATION`, never a quiet report of whichever layer
happened to look best.

**An ineligible layer is not a negative result.** A layer whose lens could not be
shown to read anything out was never causally tested, so "no transfer found
there" is not something this study observed. When no earlier layer is eligible
the verdict is inconclusive and names lens validity as the blocker — it is
emphatically not :data:`LATE_ONLY_SUPPORTED`, which is a claim about layers that
*were* tested and came back empty.

**The comparison is paired.** :func:`paired_layer_comparison` contrasts each
earlier layer with layer 38 photograph by photograph, on the same images, so a
depth difference is not partly a difference in which pictures a layer received.

And the limit that survives every outcome: this reports the earliest *tested*
layer with evidence. The earliest layer in the model is a different quantity,
and four layers cannot measure it.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

from jlens.mmlocalize.layers import (
    LOCALIZATION_LAYERS,
    REFERENCE_LAYER,
    layer_ref,
)
from jlens.mmlocalize.targets import (
    CONCEPT_CONDITIONING_LIMITATION,
    LOCALIZATION_CONCEPTS,
    POLICY_REUSED_PAIRED,
    REUSED_POLICY_LIMITATION,
)
from jlens.mmpilot.report import FAIL, NOT_EVALUATED, PASS, criterion

EARLY_TRANSFER_CONFIRMED = "EARLY_TRANSFER_CONFIRMED"
LATE_ONLY_SUPPORTED = "LATE_ONLY_SUPPORTED"
INCONCLUSIVE_LAYER_LOCALIZATION = "INCONCLUSIVE_LAYER_LOCALIZATION"

#: Bound into the run fingerprint.
LOCALIZATION_VERDICT_VERSION = "mmlocalize.localization_verdict.v1"

#: The two off-diagonal directions. Same-modality cells are not run.
OFF_DIAGONAL_PAIRS = ("text->image", "image->text")

#: Printed with every verdict, whatever it says.
DEPTH_SCOPE_LIMITATION = (
    "This reports the EARLIEST TESTED LAYER WITH EVIDENCE among physical layers "
    f"{list(LOCALIZATION_LAYERS)}. It does not report the earliest layer in the "
    "model. Four layers sampled at roughly 48%, 62%, 76% and 90% of depth cannot "
    "resolve where a signal begins; a layer between two tested ones is untested, "
    "and a layer shallower than the shallowest tested is unexamined."
)


@dataclass(frozen=True)
class LocalizationThresholds:
    """The rubric's numbers, stated once and printed in the report."""

    concepts: tuple[str, ...] = LOCALIZATION_CONCEPTS
    min_concepts_transferring: int = 1
    min_fraction_expected_sign: float = 0.75
    control_separation_factor: float = 1.5
    norm_ratio_bounds: tuple[float, float] = (0.5, 2.0)
    required_positive_images_per_cell: int = 4
    required_negative_images_per_cell: int = 4
    #: Distinct photographs a layer must actually have contributed before its
    #: cell is allowed to decide anything.
    min_images_for_a_decidable_cell: int = 4
    version: str = LOCALIZATION_VERDICT_VERSION

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["concepts"] = list(self.concepts)
        payload["norm_ratio_bounds"] = list(self.norm_ratio_bounds)
        return payload


# ------------------------------------------------------------- pass budget


@dataclass(frozen=True)
class LocalizationBudget:
    """What the configuration costs, before anything is loaded."""

    n_concepts: int
    n_candidates: int
    n_modalities: int
    n_layers_captured: int
    n_eligible_causal_layers: int
    n_total_groups: int
    n_capability_groups: int
    n_causal_cells: int
    n_targets_per_cell: int
    n_conditions_per_target: int
    n_validation_prompts: int
    validation_target_discovery_passes: int
    text_validation_passes: int
    capability_passes: int
    activation_passes: int
    causal_clean_passes: int
    causal_intervention_passes: int
    total_passes: int
    recalibration_passes: int
    estimated_units: dict
    estimated_drive_bytes: int

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def estimated_drive_mb(self) -> float:
        return self.estimated_drive_bytes / (1024 * 1024)


def estimate_localization_passes(
    *,
    n_concepts: int,
    modalities: Sequence[str],
    n_total_groups: int,
    n_capability_groups: int,
    n_layers_captured: int,
    n_eligible_causal_layers: int,
    n_targets_per_cell: int,
    alphas: Sequence[float],
    n_validation_prompts: int,
    n_validation_discovery_prompts: int = 0,
    n_control_kinds_with_alpha: int = 4,
    n_candidate_orders: int = 2,
    n_readout_variants: int = 5,
    recalibration_enabled: bool = False,
    n_recalibration_prompts: int = 256,
    d_model: int = 2560,
) -> LocalizationBudget:
    """Count every forward pass the configuration implies. No model is touched.

    Two costs here differ from the robustness study's and are worth reading.

    Activation capture does **not** multiply by layers: one forward pass records
    every requested layer's residual, so four layers cost what one did. That is
    why a four-layer diagnostic is affordable at all.

    Causal cost **does** multiply by eligible layers, and by nothing else that
    can be predicted in advance — which is why this budget is recomputed after
    Stage B, when the eligible set is known, rather than guessed before it.
    """
    n_modalities = len(modalities)
    n_candidates = n_concepts
    positive_alphas = [a for a in alphas if a > 0]
    n_conditions = 1 + len(positive_alphas) * n_control_kinds_with_alpha
    # Off-diagonal only: each source modality drives the other one.
    n_cells = len(LOCALIZATION_CONCEPTS) * n_modalities * (n_modalities - 1)

    # One forward per (prompt, layer-independent capture) per readout variant is
    # not needed: the variants re-read one captured residual. The model cost is
    # one teacher-forced forward per validation prompt.
    target_discovery = int(n_validation_discovery_prompts)
    text_validation = n_validation_prompts
    capability = n_capability_groups * n_modalities * n_candidate_orders * n_candidates
    activation = n_total_groups * n_modalities  # every layer in one pass
    clean = n_cells * n_eligible_causal_layers * n_targets_per_cell * n_candidates
    intervention = (
        n_cells * n_eligible_causal_layers * n_targets_per_cell * n_conditions * n_candidates
    )
    recalibration = n_recalibration_prompts if recalibration_enabled else 0

    units = {
        "validation_target_discovery": target_discovery,
        "text_validation": n_validation_prompts,
        "capability": n_capability_groups * n_modalities,
        "activation": n_total_groups * n_modalities * n_layers_captured,
        "jspace": n_total_groups * n_modalities * n_layers_captured,
        "direction": n_concepts * n_modalities * 2 * n_eligible_causal_layers,
        "intervention": n_cells * n_eligible_causal_layers * n_targets_per_cell * n_conditions,
        "metric": 4 + n_layers_captured,
    }
    bytes_per = {
        "validation_target_discovery": 0,
        "text_validation": n_readout_variants * 1_200,
        "capability": 12_000,
        "activation": d_model * 20 + 2_000,
        "jspace": 4_000,
        "direction": d_model * 20 + 2_000,
        "intervention": 4_000,
        "metric": 200_000,
    }
    estimated_bytes = sum(count * bytes_per[name] for name, count in units.items())

    return LocalizationBudget(
        n_concepts=n_concepts,
        n_candidates=n_candidates,
        n_modalities=n_modalities,
        n_layers_captured=int(n_layers_captured),
        n_eligible_causal_layers=int(n_eligible_causal_layers),
        n_total_groups=n_total_groups,
        n_capability_groups=n_capability_groups,
        n_causal_cells=n_cells,
        n_targets_per_cell=n_targets_per_cell,
        n_conditions_per_target=n_conditions,
        n_validation_prompts=int(n_validation_prompts),
        validation_target_discovery_passes=target_discovery,
        text_validation_passes=text_validation,
        capability_passes=capability,
        activation_passes=activation,
        causal_clean_passes=clean,
        causal_intervention_passes=intervention,
        total_passes=(
            target_discovery + text_validation + capability + activation + clean + intervention
        ),
        recalibration_passes=recalibration,
        estimated_units=units,
        estimated_drive_bytes=estimated_bytes,
    )


def format_budget(budget: LocalizationBudget, *, seconds_per_pass: float = 0.5) -> str:
    """The block the notebook prints before asking for confirmation."""
    hours = budget.total_passes * seconds_per_pass / 3600.0
    lines = [
        "=" * 72,
        "MODEL PASS BUDGET — read this before confirming",
        "=" * 72,
        f"  concepts scored per question    {budget.n_candidates}",
        f"  modalities                      {budget.n_modalities}",
        f"  layers captured                 {budget.n_layers_captured} "
        "(ONE forward pass records all of them)",
        f"  layers eligible for causal work  {budget.n_eligible_causal_layers} "
        "(decided by Stage B, not assumed)",
        f"  synchronized groups             {budget.n_total_groups}",
        f"  off-diagonal cells per layer    {budget.n_causal_cells}",
        f"  targets per cell                {budget.n_targets_per_cell}",
        f"  conditions per target           {budget.n_conditions_per_target}",
        "",
        f"  target-discovery passes         "
        f"{budget.validation_target_discovery_passes:>8,}",
        f"  text lens-validation passes     {budget.text_validation_passes:>8,}",
        f"  capability passes               {budget.capability_passes:>8,}",
        f"  activation passes               {budget.activation_passes:>8,}",
        f"  causal clean-scoring passes     {budget.causal_clean_passes:>8,}",
        f"  causal intervention passes      {budget.causal_intervention_passes:>8,}",
        f"  TOTAL model forward passes      {budget.total_passes:>8,}",
        "",
        f"  text recalibration              {budget.recalibration_passes:>8,} "
        + (
            "(enabled — a separate, explicit choice)"
            if budget.recalibration_passes
            else "(disabled; the frozen v2 artifact is evaluated as-is)"
        ),
        "",
        f"  estimated wall clock            ~{hours:.1f} h at "
        f"{seconds_per_pass:.2f} s/pass on one L4",
        f"  estimated Drive footprint       ~{budget.estimated_drive_mb:.0f} MB",
        "  estimated units: "
        + ", ".join(f"{k}={v}" for k, v in sorted(budget.estimated_units.items())),
    ]
    return "\n".join(lines)


# ------------------------------------------------------ the paired comparison


def paired_layer_comparison(
    interventions: Mapping,
    *,
    layers: Sequence[int] = LOCALIZATION_LAYERS,
    reference_layer: int = REFERENCE_LAYER,
    concepts: Sequence[str] = LOCALIZATION_CONCEPTS,
    control_kind: str = "source_concept",
) -> list[dict]:
    """Photograph-by-photograph contrasts between each layer and the reference.

    For every (concept, direction, alpha) the strongest-alpha source-concept row
    is taken at each layer, its ``per_image`` effects are matched on
    ``image_id``, and the paired differences are summarised. Only photographs
    present at **both** layers are used, which is all of them when the frozen
    target set was honoured — and the count is reported so a reader can check
    that rather than assume it.

    A paired difference is the right statistic here because the same photograph
    appears at every layer: an unpaired comparison would fold each image's own
    difficulty into the depth contrast.
    """
    rows = list(interventions.get("rows", []))
    out: list[dict] = []
    for concept in concepts:
        for pair in OFF_DIAGONAL_PAIRS:
            reference_row = _strongest_row(
                rows, concept=concept, pair=pair, layer=reference_layer,
                control_kind=control_kind,
            )
            if reference_row is None:
                continue
            alpha = float(reference_row["alpha"])
            reference_images = _per_image_effects(reference_row)
            for layer in sorted(int(x) for x in layers):
                if int(layer) == int(reference_layer):
                    continue
                row = _row_at(
                    rows, concept=concept, pair=pair, layer=layer,
                    control_kind=control_kind, alpha=alpha,
                )
                if row is None:
                    out.append(
                        {
                            "concept": concept,
                            "pair": pair,
                            "layer": int(layer),
                            "reference_layer": int(reference_layer),
                            "alpha": alpha,
                            "paired": False,
                            "reason": (
                                "no matching row at this layer — it was not "
                                "causally tested (ineligible), so there is "
                                "nothing to pair"
                            ),
                        }
                    )
                    continue
                layer_images = _per_image_effects(row)
                shared = sorted(set(reference_images) & set(layer_images))
                deltas = [layer_images[image] - reference_images[image] for image in shared]
                out.append(
                    {
                        "concept": concept,
                        "pair": pair,
                        "layer": int(layer),
                        "layer_normalized": layer_ref(int(layer)).normalized,
                        "reference_layer": int(reference_layer),
                        "alpha": alpha,
                        "paired": True,
                        "n_paired_images": len(shared),
                        "n_images_layer": len(layer_images),
                        "n_images_reference": len(reference_images),
                        "layer_effect": _mean([layer_images[i] for i in shared]),
                        "reference_effect": _mean([reference_images[i] for i in shared]),
                        "mean_paired_delta": _mean(deltas),
                        "median_paired_delta": (
                            float(statistics.median(deltas)) if deltas else None
                        ),
                        "fraction_images_layer_exceeds_reference": (
                            _mean([1.0 if d > 0 else 0.0 for d in deltas])
                        ),
                        "paired_image_ids": shared,
                    }
                )
    return out


def _strongest_row(
    rows: Sequence[Mapping], *, concept: str, pair: str, layer: int, control_kind: str
) -> dict | None:
    candidates = [
        row
        for row in rows
        if row["concept"] == concept
        and row["pair"] == pair
        and int(row["layer"]) == int(layer)
        and row["control_kind"] == control_kind
        and float(row["alpha"]) > 0
    ]
    return (
        max(candidates, key=lambda row: float(row["mean_signed_target_effect"]))
        if candidates
        else None
    )


def _row_at(
    rows: Sequence[Mapping],
    *,
    concept: str,
    pair: str,
    layer: int,
    control_kind: str,
    alpha: float,
) -> dict | None:
    for row in rows:
        if (
            row["concept"] == concept
            and row["pair"] == pair
            and int(row["layer"]) == int(layer)
            and row["control_kind"] == control_kind
            and float(row["alpha"]) == float(alpha)
        ):
            return row
    return None


def _per_image_effects(row: Mapping) -> dict[str, float]:
    return {
        str(image_id): float(block["mean_signed_target_effect"])
        for image_id, block in (row.get("per_image") or {}).items()
        if block.get("mean_signed_target_effect") is not None
    }


def _mean(values: Sequence[float]) -> float | None:
    values = [float(v) for v in values if v is not None]
    return statistics.fmean(values) if values else None


# -------------------------------------------------------------- causal cells


def evaluate_layer_cells(
    interventions: Mapping,
    *,
    layer: int,
    concepts: Sequence[str] = LOCALIZATION_CONCEPTS,
    thresholds: LocalizationThresholds | None = None,
) -> list[dict]:
    """One verdict per (concept, direction) at one layer, against its controls.

    Every condition is required. A cell whose mean cleared its controls but had
    the wrong sign on half its photographs has not transferred, and neither has
    one whose four distinct images turned out to be fewer.
    """
    thresholds = thresholds or LocalizationThresholds()
    rows = [row for row in interventions.get("rows", []) if int(row["layer"]) == int(layer)]
    low, high = thresholds.norm_ratio_bounds
    out: list[dict] = []
    for concept in concepts:
        for pair in OFF_DIAGONAL_PAIRS:
            best = _strongest_row(
                rows, concept=concept, pair=pair, layer=layer,
                control_kind="source_concept",
            )
            if best is None:
                out.append(
                    {
                        "concept": concept,
                        "pair": pair,
                        "layer": int(layer),
                        "evaluated": False,
                        "passes": False,
                        "reasons": ["no source-concept row at this layer"],
                    }
                )
                continue
            alpha = float(best["alpha"])
            random_row = _row_at(
                rows, concept=concept, pair=pair, layer=layer,
                control_kind="random_norm_matched", alpha=alpha,
            )
            unrelated_row = _row_at(
                rows, concept=concept, pair=pair, layer=layer,
                control_kind="unrelated_concept", alpha=alpha,
            )
            raw_row = _row_at(
                rows, concept=concept, pair=pair, layer=layer,
                control_kind="raw_residual_difference", alpha=alpha,
            )
            effect = float(best["mean_signed_target_effect"])
            controls = [
                float(row["mean_signed_target_effect"])
                for row in (random_row, unrelated_row)
                if row is not None
            ]
            strongest = max(controls) if controls else 0.0

            reasons: list[str] = []
            if effect <= 0:
                reasons.append(f"effect {effect:+.4f} is not positive")
            if best["fraction_expected_sign"] < thresholds.min_fraction_expected_sign:
                reasons.append(
                    f"expected-sign fraction {best['fraction_expected_sign']:.2f} < "
                    f"{thresholds.min_fraction_expected_sign}"
                )
            if len(controls) < 2:
                reasons.append("a matched random or unrelated-concept control is missing")
            elif not (
                effect > float(random_row["mean_signed_target_effect"])
                and effect > float(unrelated_row["mean_signed_target_effect"])
                and effect >= thresholds.control_separation_factor * max(strongest, 0.0)
            ):
                reasons.append(
                    f"effect {effect:+.4f} does not clear "
                    f"{thresholds.control_separation_factor}x the strongest control "
                    f"({strongest:+.4f})"
                )
            ratio = float(best["mean_activation_norm_ratio"])
            if not low <= ratio <= high:
                reasons.append(f"activation norm ratio {ratio:.3f} outside [{low}, {high}]")
            unrelated_change = float(best["mean_abs_unrelated_change"])
            if unrelated_change >= abs(effect):
                reasons.append(
                    f"unrelated candidates moved {unrelated_change:.4f}, not less than "
                    f"the target's {abs(effect):.4f} — the edit looks global"
                )
            n_positive = int(best.get("n_positive_images", 0))
            n_negative = int(best.get("n_negative_images", 0))
            if n_positive < thresholds.required_positive_images_per_cell:
                reasons.append(
                    f"{n_positive} distinct positive image(s) < "
                    f"{thresholds.required_positive_images_per_cell}"
                )
            if n_negative < thresholds.required_negative_images_per_cell:
                reasons.append(
                    f"{n_negative} distinct negative image(s) < "
                    f"{thresholds.required_negative_images_per_cell}"
                )
            if int(best.get("n_distinct_images", 0)) < thresholds.min_images_for_a_decidable_cell:
                reasons.append(
                    f"{best.get('n_distinct_images')} distinct photograph(s) is too "
                    f"few for this cell to decide anything "
                    f"(need {thresholds.min_images_for_a_decidable_cell})"
                )

            beats_raw = raw_row is None or effect >= float(
                raw_row["mean_signed_target_effect"]
            )
            out.append(
                {
                    "concept": concept,
                    "pair": pair,
                    "layer": int(layer),
                    "layer_normalized": layer_ref(int(layer)).normalized,
                    "evaluated": True,
                    "alpha": alpha,
                    "mean_signed_target_effect": effect,
                    "mean_signed_margin_effect": best.get("mean_signed_margin_effect"),
                    "fraction_expected_sign": best["fraction_expected_sign"],
                    "mean_activation_norm_ratio": ratio,
                    "mean_abs_unrelated_change": unrelated_change,
                    "n_prediction_changes": best.get("n_prediction_changes"),
                    "n_distinct_images": best.get("n_distinct_images"),
                    "n_positive_images": n_positive,
                    "n_negative_images": n_negative,
                    "random_control": (random_row or {}).get("mean_signed_target_effect"),
                    "unrelated_control": (unrelated_row or {}).get(
                        "mean_signed_target_effect"
                    ),
                    "raw_residual_control": (raw_row or {}).get(
                        "mean_signed_target_effect"
                    ),
                    "jspace_beats_raw_direction": beats_raw,
                    "passes": not reasons,
                    "reasons": reasons,
                }
            )
    return out


def representational_rows(representational: Mapping, *, layer: int) -> list[dict]:
    """The two cross-modal directions at one layer, with what each is judged on."""
    out = []
    for pair in OFF_DIAGONAL_PAIRS:
        entry = (representational.get("pairs") or {}).get(pair)
        if not entry:
            continue
        jspace = entry["jspace_retrieval"]["top1_accuracy"]
        shuffled = entry["shuffled_control"]["p95_top1_accuracy"]
        jspace_gap = entry["jspace_separation"]["gap"]
        raw_gap = entry["raw_residual_separation"]["gap"]
        out.append(
            {
                "layer": int(layer),
                "layer_normalized": layer_ref(int(layer)).normalized,
                "pair": pair,
                "n_queries": entry["jspace_retrieval"]["n_queries"],
                "jspace_top1": jspace,
                "jspace_mrr": entry["jspace_retrieval"]["mrr"],
                "shuffled_p95": shuffled,
                # Accuracy is discrete at 1/n_queries, so a fixed additive margin
                # can demand an accuracy above 1.0. The stated control is: beat it.
                "beats_shuffled": jspace > shuffled,
                "jspace_separation_gap": jspace_gap,
                "raw_separation_gap": raw_gap,
                "support_overlap_gap": entry["jspace_support_overlap"]["gap"],
                "raw_top1": entry["raw_residual_retrieval"]["top1_accuracy"],
                "jspace_separation_beats_raw": jspace_gap > raw_gap,
                "n_excluded_same_image_different_group": (
                    entry.get("exclusions", {}).get(
                        "n_excluded_same_image_different_group"
                    )
                ),
            }
        )
    return out


# ------------------------------------------------------------------- verdict


def localization_verdict(
    *,
    validity: Mapping[int, Mapping],
    representational: Mapping[int, Mapping],
    interventions: Mapping,
    target_manifest: Mapping,
    layers: Sequence[int] = LOCALIZATION_LAYERS,
    reference_layer: int = REFERENCE_LAYER,
    concepts: Sequence[str] = LOCALIZATION_CONCEPTS,
    thresholds: LocalizationThresholds | None = None,
) -> dict:
    """The localization decision.

    Args:
        validity: ``{layer: Stage B result}``. Eligibility is read from here and
            nowhere else — a layer is never treated as testable because it
            produced a number.
        representational: ``{layer: stage_representational report}``.
        interventions: The **image-level** aggregate. This study's unit is the
            photograph; a group-level table would let one image count twice.
    """
    thresholds = thresholds or LocalizationThresholds()
    layers = [int(layer) for layer in layers]
    earlier = [layer for layer in layers if layer < int(reference_layer)]

    per_layer: dict[int, dict] = {}
    for layer in layers:
        eligible = bool(validity.get(layer, {}).get("eligible"))
        rep_rows = representational_rows(
            representational.get(layer, {}) or {}, layer=layer
        )
        cells = (
            evaluate_layer_cells(
                interventions, layer=layer, concepts=concepts, thresholds=thresholds
            )
            if eligible
            else []
        )
        passing = [cell for cell in cells if cell["passes"]]
        per_layer[layer] = {
            "layer": layer,
            "layer_normalized": layer_ref(layer).normalized,
            "lens_eligible": eligible,
            "lens_failed_checks": list(validity.get(layer, {}).get("failed_checks", [])),
            "representational": rep_rows,
            "representation_beats_shuffled_both_directions": bool(
                rep_rows and all(row["beats_shuffled"] for row in rep_rows)
            ),
            "jspace_separation_beats_raw": bool(
                rep_rows and all(row["jspace_separation_beats_raw"] for row in rep_rows)
            ),
            "causal_cells": cells,
            "causally_tested": eligible,
            "concepts_transferring": sorted({cell["concept"] for cell in passing}),
            "passing_cells": [
                {"concept": cell["concept"], "pair": cell["pair"]} for cell in passing
            ],
            "raw_direction_exceptions": [
                {"concept": cell["concept"], "pair": cell["pair"]}
                for cell in passing
                if not cell["jspace_beats_raw_direction"]
            ],
            "skipped_because": (
                None
                if eligible
                else (
                    "the lens did not pass the independent layer-validity gate at "
                    "this layer, so no causal claim may rest on it. Its diagnostic "
                    "results above are preserved; this is not evidence that "
                    "transfer is absent here — it was never tested."
                )
            ),
        }

    def _layer_transfers(layer: int) -> bool:
        entry = per_layer[layer]
        return bool(
            entry["lens_eligible"]
            and entry["representation_beats_shuffled_both_directions"]
            and len(entry["concepts_transferring"]) >= thresholds.min_concepts_transferring
        )

    reference = per_layer.get(int(reference_layer), {})
    reference_reproduces = _layer_transfers(int(reference_layer))
    eligible_earlier = [layer for layer in earlier if per_layer[layer]["lens_eligible"]]
    transferring_earlier = [layer for layer in eligible_earlier if _layer_transfers(layer)]
    earliest = min(transferring_earlier) if transferring_earlier else None

    criteria: dict[str, dict] = {}
    criteria["reference_layer_reproduces"] = criterion(
        PASS if reference_reproduces else FAIL,
        {
            "layer": int(reference_layer),
            "lens_eligible": reference.get("lens_eligible"),
            "representation_beats_shuffled": reference.get(
                "representation_beats_shuffled_both_directions"
            ),
            "concepts_transferring": reference.get("concepts_transferring"),
            "reading": (
                "layer 38 is the anchor. Without it there is nothing for an "
                "earlier layer to be earlier than, and no verdict about depth "
                "can be drawn from this run."
            ),
        },
    )
    criteria["earlier_layers_were_eligible"] = criterion(
        PASS if eligible_earlier else FAIL,
        {
            "earlier_layers": earlier,
            "eligible": eligible_earlier,
            "per_layer_failed_checks": {
                layer: per_layer[layer]["lens_failed_checks"] for layer in earlier
            },
            "reading": (
                "an ineligible layer was never causally tested. 'No transfer "
                "found there' is not an observation this study made about it."
            ),
        },
    )
    causal_skip = (
        None if eligible_earlier else "no earlier layer passed the lens-validity gate"
    )
    criteria["earlier_layer_transfers_under_controls"] = criterion(
        NOT_EVALUATED
        if causal_skip
        else (PASS if transferring_earlier else FAIL),
        {
            "eligible_earlier_layers": eligible_earlier,
            "transferring": transferring_earlier,
            "earliest_with_evidence": earliest,
            "per_layer": {
                layer: {
                    "concepts_transferring": per_layer[layer]["concepts_transferring"],
                    "representation_beats_shuffled": per_layer[layer][
                        "representation_beats_shuffled_both_directions"
                    ],
                    "cells": per_layer[layer]["causal_cells"],
                }
                for layer in eligible_earlier
            },
            "required_concepts": thresholds.min_concepts_transferring,
        },
        skipped_because=causal_skip,
    )
    criteria["targets_identical_at_every_layer"] = criterion(
        PASS if target_manifest.get("same_targets_at_every_layer") else FAIL,
        {
            "target_checksum": target_manifest.get("target_checksum"),
            "policy": target_manifest.get("policy"),
            "reading": (
                "a depth contrast on different photographs per layer is partly a "
                "contrast between photographs"
            ),
        },
    )
    criteria["concepts_are_conditioned_not_sampled"] = criterion(
        PASS,
        {
            "concepts": list(concepts),
            "reading": CONCEPT_CONDITIONING_LIMITATION,
        },
    )

    statuses = {name: entry["status"] for name, entry in criteria.items()}

    if not reference_reproduces:
        verdict = INCONCLUSIVE_LAYER_LOCALIZATION
        rationale = (
            f"Layer {reference_layer} did not reproduce on the localization "
            "subset, so there is no established effect for an earlier layer to be "
            "earlier than. Nothing about depth can be concluded from this run — "
            "including about the earlier layers, whose numbers are reported but "
            "cannot be interpreted against a missing anchor."
        )
    elif transferring_earlier:
        verdict = EARLY_TRANSFER_CONFIRMED
        rationale = (
            f"Physical layer {earliest} (~normalized "
            f"{layer_ref(int(earliest)).normalized}) passed the independent "
            "lens-validity gate, beat the shuffled representation control in both "
            "directions, and produced an expected-sign off-diagonal causal effect "
            f"for {per_layer[earliest]['concepts_transferring']} that exceeded its "
            "matched random and external unrelated-concept controls with sane "
            f"activation norms — while layer {reference_layer} reproduced. This is "
            "the earliest TESTED layer with evidence, not the earliest layer in "
            "the model."
        )
    elif eligible_earlier:
        verdict = LATE_ONLY_SUPPORTED
        rationale = (
            f"Layer {reference_layer} reproduced on the localization subset, and "
            f"the earlier layer(s) that passed the lens-validity gate "
            f"({eligible_earlier}) did not produce controlled off-diagonal "
            "transfer. Within this design the effect is supported only near "
            "answer-language convergence. Untested layers, and layers between the "
            "tested ones, are not covered by this statement."
        )
    else:
        verdict = INCONCLUSIVE_LAYER_LOCALIZATION
        rationale = (
            f"Layer {reference_layer} reproduced, but no earlier layer passed the "
            "independent lens-validity gate, so no earlier layer was causally "
            "tested. This is NOT evidence that transfer is late-only: it is the "
            "absence of a usable readout earlier in the stack, which is a fact "
            "about the frozen text-calibrated lens at those layers rather than "
            "about the model's representations."
        )

    return {
        "schema": "jlens.mmlocalize.localization_verdict.v1",
        "verdict": verdict,
        "rationale": rationale,
        "criteria": criteria,
        "criteria_status": statuses,
        "layers": layers,
        "reference_layer": int(reference_layer),
        "earlier_layers": earlier,
        "eligible_earlier_layers": eligible_earlier,
        "earlier_layers_transferring": transferring_earlier,
        "earliest_tested_layer_with_evidence": earliest,
        "reference_layer_reproduces": reference_reproduces,
        "per_layer": per_layer,
        "paired_layer_comparison": paired_layer_comparison(
            interventions,
            layers=layers,
            reference_layer=reference_layer,
            concepts=concepts,
        ),
        "concepts": list(concepts),
        "thresholds": thresholds.to_dict(),
        "target_policy": target_manifest.get("policy"),
        "target_checksum": target_manifest.get("target_checksum"),
        "depth_scope_limitation": DEPTH_SCOPE_LIMITATION,
        "concept_conditioning_limitation": CONCEPT_CONDITIONING_LIMITATION,
        "target_policy_limitation": (
            REUSED_POLICY_LIMITATION
            if target_manifest.get("policy") == POLICY_REUSED_PAIRED
            else None
        ),
        "intervention_limitation": (
            "Interventions add and subtract a direction on the residual stream at "
            "the final prompt token. That is not erasure and not projection "
            "ablation."
        ),
    }


# -------------------------------------------------------------------- report


def render_report(
    *,
    run_dir: str,
    verdict: Mapping,
    validity: Mapping[int, Mapping],
    budget: Mapping | None,
    resume: Mapping | None,
    mode: str,
) -> str:
    """The localization report. Layer tables first, verdict after."""
    mode_note = (
        "  \n- **MOCK run: pipeline evidence only, not scientific evidence.**"
        if mode.lower() == "mock"
        else ""
    )
    lines = [
        f"# Layer localization — {verdict['verdict']}",
        "",
        f"- mode: **{mode}**{mode_note}",
        f"- run directory: `{run_dir}`",
        f"- layers (physical): {verdict['layers']} "
        f"(~normalized {[layer_ref(x).normalized for x in verdict['layers']]})",
        f"- reference layer: {verdict['reference_layer']}",
        f"- concepts: {verdict['concepts']}",
        f"- target policy: `{verdict['target_policy']}`",
        f"- target checksum: `{verdict['target_checksum']}`",
        "",
        "## Verdict",
        "",
        f"**{verdict['verdict']}** — {verdict['rationale']}",
        "",
        "## Stage B — layer validity (text only, held-out, tie-aware)",
        "",
        "Ranks are **midranks**. Unique top-1 and the tied-at-maximum rate are "
        "reported because they are the pair of numbers the old gate confused: "
        "under a tie block, argmax reports the tie-break rule.",
        "",
        "| layer | norm | eligible | MRR | median midrank | median optimistic | "
        "unique top-1 | tied-at-max | top-10 incl | old gate | failed |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for layer in verdict["layers"]:
        result = validity.get(layer) or validity.get(int(layer))
        if not result:
            lines.append(f"| {layer} | | not evaluated | | | | | | | | |")
            continue
        metrics = result["metrics"]["j_lens"]
        lines.append(
            f"| {layer} | ~{layer_ref(int(layer)).normalized} | "
            f"{'yes' if result['eligible'] else 'NO'} | "
            f"{metrics['mean_reciprocal_rank']:.4f} | "
            f"{metrics['median_midrank']:.2f} | "
            f"{metrics['median_optimistic_rank']:.2f} | "
            f"{metrics['unique_top1_agreement']:.3f} | "
            f"{metrics['tied_at_max_rate']:.3f} | "
            f"{metrics.get('top10_inclusion', float('nan')):.3f} | "
            f"{'pass' if result['legacy_gate']['passed'] else 'fail'} | "
            f"{', '.join(result['failed_checks']) or '-'} |"
        )

    lines += [
        "",
        "## Stage C and D — per layer",
        "",
    ]
    for layer in verdict["layers"]:
        entry = verdict["per_layer"][layer]
        lines += [
            f"### Physical layer {layer} (~normalized {entry['layer_normalized']})",
            "",
            f"- lens eligible: **{'yes' if entry['lens_eligible'] else 'no'}**",
        ]
        if entry["representational"]:
            lines += [
                "",
                "| direction | queries | J top-1 | J MRR | shuffled p95 | beats shuffled | "
                "J gap | raw gap |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
            for row in entry["representational"]:
                lines.append(
                    f"| {row['pair']} | {row['n_queries']} | {row['jspace_top1']:.3f} | "
                    f"{row['jspace_mrr']:.3f} | {row['shuffled_p95']:.3f} | "
                    f"{'yes' if row['beats_shuffled'] else 'no'} | "
                    f"{row['jspace_separation_gap']:+.4f} | "
                    f"{row['raw_separation_gap']:+.4f} |"
                )
        if not entry["lens_eligible"]:
            lines += ["", f"- causal stage **skipped**: {entry['skipped_because']}", ""]
            continue
        lines += [
            "",
            "| concept | direction | alpha | +img | -img | effect | sign frac | random | "
            "unrelated | raw | norm | passes |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for cell in entry["causal_cells"]:
            if not cell.get("evaluated"):
                lines.append(
                    f"| {cell['concept']} | {cell['pair']} | | | | | | | | | | "
                    "not evaluated |"
                )
                continue
            lines.append(
                f"| {cell['concept']} | {cell['pair']} | {cell['alpha']:g} | "
                f"{cell['n_positive_images']} | {cell['n_negative_images']} | "
                f"{cell['mean_signed_target_effect']:+.4f} | "
                f"{cell['fraction_expected_sign']:.2f} | "
                f"{_number(cell['random_control'])} | "
                f"{_number(cell['unrelated_control'])} | "
                f"{_number(cell['raw_residual_control'])} | "
                f"{cell['mean_activation_norm_ratio']:.3f} | "
                f"{'yes' if cell['passes'] else 'no'} |"
            )
        failures = [
            f"- **{cell['concept']} {cell['pair']}**: " + "; ".join(cell["reasons"])
            for cell in entry["causal_cells"]
            if not cell["passes"] and cell.get("reasons")
        ]
        if failures:
            lines += ["", "Why a cell did not pass:", "", *failures]
        if entry["raw_direction_exceptions"]:
            lines += [
                "",
                "- **The raw difference-in-means direction matched or beat the "
                f"J-space one** in {entry['raw_direction_exceptions']}. Transfer "
                "is unaffected; what is downgraded is the claim that the J-space "
                "decomposition is what carried it.",
            ]
        if not entry["jspace_separation_beats_raw"] and entry["representational"]:
            lines.append(
                "- Raw-residual separation matched or beat J-space separation at "
                "this layer; the representational result stands, the "
                "decomposition's added value does not."
            )
        lines.append("")

    paired = [row for row in verdict["paired_layer_comparison"] if row.get("paired")]
    if paired:
        lines += [
            "## Paired depth contrast (same photographs at every layer)",
            "",
            "| concept | direction | layer | vs | alpha | paired images | layer effect | "
            "reference effect | mean paired delta | frac images layer > reference |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for row in paired:
            lines.append(
                f"| {row['concept']} | {row['pair']} | {row['layer']} | "
                f"{row['reference_layer']} | {row['alpha']:g} | "
                f"{row['n_paired_images']} | {_number(row['layer_effect'])} | "
                f"{_number(row['reference_effect'])} | "
                f"{_number(row['mean_paired_delta'])} | "
                f"{_number(row['fraction_images_layer_exceeds_reference'])} |"
            )
        lines.append("")

    lines += [
        "## Criteria",
        "",
        "| criterion | status |",
        "| --- | --- |",
    ]
    lines += [
        f"| {name} | {status.replace('_', ' ')} |"
        for name, status in verdict["criteria_status"].items()
    ]
    if budget:
        lines += [
            "",
            "## What it cost",
            "",
            f"- model forward passes: {budget.get('total_passes'):,}",
            f"- layers captured: {budget.get('n_layers_captured')} "
            "(one forward pass records all of them)",
            f"- layers causally tested: {budget.get('n_eligible_causal_layers')}",
        ]
    if resume:
        lines += [
            "",
            "## Resume",
            "",
            f"- state: {resume.get('status')}",
            f"- units: {resume.get('completed_units')}",
        ]
    lines += [
        "",
        "## Scope and limits",
        "",
        f"- {verdict['depth_scope_limitation']}",
        f"- {verdict['concept_conditioning_limitation']}",
        f"- {verdict['intervention_limitation']}",
    ]
    if verdict.get("target_policy_limitation"):
        lines.append(f"- {verdict['target_policy_limitation']}")
    lines += [
        "- A layer that failed the lens-validity gate was never causally tested. "
        "Its diagnostic numbers are preserved above; the absence of a causal "
        "result there is not a negative causal result.",
        "- Written text and images only. Spoken audio is excluded by design and "
        "environmental audio is not tested; neither absence is evidence about "
        "either.",
        "",
    ]
    return "\n".join(lines)


def _number(value) -> str:
    return "n/a" if value is None else f"{float(value):+.4f}"


__all__ = [
    "DEPTH_SCOPE_LIMITATION",
    "EARLY_TRANSFER_CONFIRMED",
    "INCONCLUSIVE_LAYER_LOCALIZATION",
    "LATE_ONLY_SUPPORTED",
    "LOCALIZATION_VERDICT_VERSION",
    "OFF_DIAGONAL_PAIRS",
    "LocalizationBudget",
    "LocalizationThresholds",
    "estimate_localization_passes",
    "evaluate_layer_cells",
    "format_budget",
    "localization_verdict",
    "paired_layer_comparison",
    "render_report",
    "representational_rows",
]
