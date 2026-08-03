# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""The bounded six-concept robustness study: budget, rubric, and report.

The four-concept pilot survived image-level correction and returned
``GO_CONFIRMED_AFTER_IMAGE_DEDUP``. This asks the smallest follow-up question
that result earns: **does it replicate?** Six concepts instead of four, eight
distinct photographs per cell instead of two, three focal concepts instead of
two, and the photograph as the independent unit from selection onward rather
than by correction afterwards.

It is deliberately not a framework. One layer (38), two modalities, one frozen
lens, off-diagonal cells only, a three-point alpha sweep. Everything that would
make it bigger — more layers, spoken audio, a new lens, early-layer work — is
out, because none of it is what "does the result replicate" needs.

Two things here are worth stating plainly.

**The rubric refuses to decide from the strongest cell.** The pilot's rubric
took the maximum off-diagonal effect and asked whether *it* beat its controls.
With one concept that is the only thing available; with three it is a way to
report the luckiest cell as the result. So a robustness GO requires at least
two of three focal concepts to transfer in **both** directions, each against
its own matched controls.

**The budget must be confirmed before anything loads.** Every count printed by
:func:`estimate_model_passes` is derived from the configuration, not guessed,
and the notebook will not start the model stages until the user has read them
and set a separate confirmation flag by hand.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

from jlens.mmpilot.report import FAIL, NOT_EVALUATED, PASS, criterion

#: The three outcomes. Named distinctly from the pilot's GO/WEAK GO/NO-GO and
#: from the audit's *_AFTER_IMAGE_DEDUP verdicts, so no report can be mistaken
#: for another study's.
ROBUSTNESS_GO = "ROBUSTNESS_GO"
ROBUSTNESS_WEAK_GO = "ROBUSTNESS_WEAK_GO"
ROBUSTNESS_NO_GO = "ROBUSTNESS_NO_GO"

#: Bound into the run fingerprint.
ROBUSTNESS_VERDICT_VERSION = "mmpilot.robustness_verdict.v1"

_TEXT_IMAGE_PAIRS = ("text->image", "image->text")


@dataclass(frozen=True)
class RobustnessThresholds:
    """The rubric's numbers, stated once and printed in the report."""

    n_concepts_required: int = 6
    n_focal_concepts: int = 3
    min_focal_concepts_transferring: int = 2
    min_fraction_expected_sign: float = 0.75
    control_separation_factor: float = 1.5
    norm_ratio_bounds: tuple[float, float] = (0.5, 2.0)
    required_positive_images_per_cell: int = 8
    required_negative_images_per_cell: int = 8
    capability_threshold: float = 0.7
    version: str = ROBUSTNESS_VERDICT_VERSION

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["norm_ratio_bounds"] = list(self.norm_ratio_bounds)
        return payload


# ------------------------------------------------------------- pass budget


@dataclass(frozen=True)
class PassBudget:
    """What the configuration will actually cost, before anything is loaded."""

    n_concepts: int
    n_modalities: int
    n_candidates: int
    n_capability_groups: int
    n_total_groups: int
    n_causal_cells: int
    n_targets_per_cell: int
    n_conditions_per_target: int
    capability_passes: int
    activation_passes: int
    causal_clean_passes: int
    causal_intervention_passes: int
    total_passes: int
    estimated_units: dict
    estimated_drive_bytes: int

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def estimated_drive_mb(self) -> float:
        return self.estimated_drive_bytes / (1024 * 1024)


def estimate_model_passes(
    *,
    n_concepts: int,
    n_focal_concepts: int,
    modalities: Sequence[str],
    n_total_groups: int,
    n_capability_groups: int,
    n_targets_per_cell: int,
    alphas: Sequence[float],
    n_control_kinds_with_alpha: int = 4,
    n_candidate_orders: int = 2,
    off_diagonal_only: bool = True,
    d_model: int = 2560,
) -> PassBudget:
    """Count every forward pass the configuration implies. No model is touched.

    A "pass" is one teacher-forced forward over prompt plus one candidate
    sequence — the unit the capability gate and every intervention are billed
    in. Scoring six candidates is six passes, which is why widening the concept
    set is not free.

    The causal count assumes the clean scoring of each target is done once and
    shared across that target's conditions, which is what
    :func:`~jlens.mmpilot.pipeline.stage_causal` does.
    """
    n_modalities = len(modalities)
    n_candidates = n_concepts
    positive_alphas = [a for a in alphas if a > 0]
    # One zero-alpha control plus every (control kind, positive alpha) pair.
    n_conditions = 1 + len(positive_alphas) * n_control_kinds_with_alpha
    n_target_modalities = (n_modalities - 1) if off_diagonal_only else n_modalities
    n_cells = n_focal_concepts * n_modalities * n_target_modalities

    capability = n_capability_groups * n_modalities * n_candidate_orders * n_candidates
    activation = n_total_groups * n_modalities
    clean = n_cells * n_targets_per_cell * n_candidates
    intervention = n_cells * n_targets_per_cell * n_conditions * n_candidates

    units = {
        "capability": n_capability_groups * n_modalities,
        "activation": n_total_groups * n_modalities,
        "jspace": n_total_groups * n_modalities,
        "direction": n_concepts * n_modalities * 2,
        "intervention": n_cells * n_targets_per_cell * n_conditions,
        "metric": 4,
    }
    # An activation unit stores d_model floats as JSON text; the rest are small
    # records. These are deliberate over-estimates — a budget that surprises
    # you upward mid-run is worse than one that surprises you downward.
    bytes_per = {
        "capability": 12_000,
        "activation": d_model * 20 + 2_000,
        "jspace": 4_000,
        "direction": d_model * 20 + 2_000,
        "intervention": 4_000,
        "metric": 200_000,
    }
    estimated_bytes = sum(count * bytes_per[name] for name, count in units.items())

    return PassBudget(
        n_concepts=n_concepts,
        n_modalities=n_modalities,
        n_candidates=n_candidates,
        n_capability_groups=n_capability_groups,
        n_total_groups=n_total_groups,
        n_causal_cells=n_cells,
        n_targets_per_cell=n_targets_per_cell,
        n_conditions_per_target=n_conditions,
        capability_passes=capability,
        activation_passes=activation,
        causal_clean_passes=clean,
        causal_intervention_passes=intervention,
        total_passes=capability + activation + clean + intervention,
        estimated_units=units,
        estimated_drive_bytes=estimated_bytes,
    )


def format_budget(budget: PassBudget, *, seconds_per_pass: float = 0.5) -> str:
    """The block the notebook prints before asking for confirmation."""
    hours = budget.total_passes * seconds_per_pass / 3600.0
    lines = [
        "=" * 72,
        "MODEL PASS BUDGET — read this before confirming",
        "=" * 72,
        f"  concepts scored per question   {budget.n_candidates} "
        f"(every candidate is a separate forward pass)",
        f"  modalities                     {budget.n_modalities}",
        f"  synchronized groups            {budget.n_total_groups} "
        "(one per distinct image)",
        f"  capability groups              {budget.n_capability_groups}",
        f"  off-diagonal causal cells      {budget.n_causal_cells}",
        f"  targets per cell               {budget.n_targets_per_cell} "
        "(distinct images)",
        f"  conditions per target          {budget.n_conditions_per_target}",
        "",
        f"  capability passes              {budget.capability_passes:>8,}",
        f"  activation passes              {budget.activation_passes:>8,}",
        f"  causal clean-scoring passes    {budget.causal_clean_passes:>8,}",
        f"  causal intervention passes     {budget.causal_intervention_passes:>8,}",
        f"  TOTAL model forward passes     {budget.total_passes:>8,}",
        "",
        f"  estimated wall clock           ~{hours:.1f} h at "
        f"{seconds_per_pass:.2f} s/pass on one L4",
        f"  estimated Drive footprint      ~{budget.estimated_drive_mb:.0f} MB",
        "  estimated units: "
        + ", ".join(f"{k}={v}" for k, v in sorted(budget.estimated_units.items())),
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------- the rubric


def _cell(rows: Sequence[Mapping], concept: str, pair: str, control_kind: str) -> dict | None:
    """The strongest positive-alpha row for one (concept, pair, control)."""
    candidates = [
        row
        for row in rows
        if row["concept"] == concept
        and row["pair"] == pair
        and row["control_kind"] == control_kind
        and row["alpha"] > 0
    ]
    return (
        max(candidates, key=lambda row: row["mean_signed_target_effect"])
        if candidates
        else None
    )


def _matched(rows: Sequence[Mapping], best: Mapping, control_kind: str) -> dict | None:
    for row in rows:
        if (
            row["concept"] == best["concept"]
            and row["pair"] == best["pair"]
            and row["alpha"] == best["alpha"]
            and row["control_kind"] == control_kind
        ):
            return row
    return None


def evaluate_causal_cells(
    interventions: Mapping,
    *,
    focal_concepts: Sequence[str],
    thresholds: RobustnessThresholds,
) -> list[dict]:
    """One verdict per (focal concept, direction), each against its own controls.

    Every condition in the list is required. A cell that clears the controls on
    a mean but had the wrong sign on half its photographs has not transferred,
    and neither has one whose eight distinct images turned out to be fewer.
    """
    rows = list(interventions.get("rows", []))
    low, high = thresholds.norm_ratio_bounds
    out = []
    for concept in focal_concepts:
        for pair in _TEXT_IMAGE_PAIRS:
            best = _cell(rows, concept, pair, "source_concept")
            if best is None:
                out.append(
                    {
                        "concept": concept,
                        "pair": pair,
                        "evaluated": False,
                        "passes": False,
                        "reasons": ["no source-concept row for this cell"],
                    }
                )
                continue
            random_row = _matched(rows, best, "random_norm_matched")
            unrelated_row = _matched(rows, best, "unrelated_concept")
            raw_row = _matched(rows, best, "raw_residual_difference")
            effect = float(best["mean_signed_target_effect"])
            controls = [
                float(row["mean_signed_target_effect"])
                for row in (random_row, unrelated_row)
                if row is not None
            ]
            strongest = max(controls) if controls else 0.0

            reasons = []
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
            # Reported and allowed to downgrade, never to veto on its own: the
            # raw difference-in-means direction answers "did the decomposition
            # earn its keep", not "did transfer happen".
            beats_raw = raw_row is None or effect >= float(
                raw_row["mean_signed_target_effect"]
            )
            out.append(
                {
                    "concept": concept,
                    "pair": pair,
                    "evaluated": True,
                    "alpha": best["alpha"],
                    "mean_signed_target_effect": effect,
                    "fraction_expected_sign": best["fraction_expected_sign"],
                    "mean_activation_norm_ratio": ratio,
                    "mean_abs_unrelated_change": unrelated_change,
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


def representational_rows(representational: Mapping) -> list[dict]:
    """The two cross-modal directions, with what each is judged on."""
    out = []
    for pair in _TEXT_IMAGE_PAIRS:
        entry = (representational.get("pairs") or {}).get(pair)
        if not entry:
            continue
        jspace = entry["jspace_retrieval"]["top1_accuracy"]
        shuffled = entry["shuffled_control"]["p95_top1_accuracy"]
        jspace_gap = entry["jspace_separation"]["gap"]
        raw_gap = entry["raw_residual_separation"]["gap"]
        exclusions = entry.get("exclusions", {})
        out.append(
            {
                "pair": pair,
                "n_queries": entry["jspace_retrieval"]["n_queries"],
                "jspace_top1": jspace,
                "jspace_mrr": entry["jspace_retrieval"]["mrr"],
                "shuffled_p95": shuffled,
                # Accuracy is discrete at 1/n_queries. A fixed additive margin
                # above p95 can demand an accuracy above 1.0, so the stated
                # control is: strictly beat it.
                "beats_shuffled": jspace > shuffled,
                "jspace_separation_gap": jspace_gap,
                "raw_separation_gap": raw_gap,
                "support_overlap_gap": entry["jspace_support_overlap"]["gap"],
                "raw_top1": entry["raw_residual_retrieval"]["top1_accuracy"],
                "jspace_separation_beats_raw": jspace_gap > raw_gap,
                "eligible_targets": exclusions.get("eligible_targets"),
                "n_excluded_same_image_different_group": exclusions.get(
                    "n_excluded_same_image_different_group"
                ),
            }
        )
    return out


def robustness_verdict(
    *,
    capability: Mapping,
    representational: Mapping,
    interventions: Mapping,
    selected_concepts: Sequence[str],
    focal_concepts: Sequence[str],
    unrelated_controls: Mapping[str, str],
    blocked_modalities: Sequence[str] = (),
    thresholds: RobustnessThresholds | None = None,
) -> dict:
    """The robustness decision. Replication first, strongest cell never.

    ``interventions`` must be the **image-level** aggregate: this study's unit
    is the photograph, and a group-level table would let one image's repeated
    captions count more than once — which is the thing the whole design exists
    to avoid.
    """
    thresholds = thresholds or RobustnessThresholds()
    criteria: dict[str, dict] = {}

    retained = list(capability.get("text_image_retained_concepts", []))
    missing = [c for c in selected_concepts if c not in retained]
    capability_passed = len(missing) == 0 and len(selected_concepts) >= (
        thresholds.n_concepts_required
    )
    criteria["six_way_capability"] = criterion(
        PASS if capability_passed else FAIL,
        {
            "selected_concepts": list(selected_concepts),
            "retained_text_image_concepts": retained,
            "concepts_failing_the_gate": missing,
            "required": thresholds.n_concepts_required,
            "threshold": capability.get("threshold"),
            "per_concept": capability.get("per_concept", {}),
            "reading": (
                "every selected concept must pass; a concept is never replaced "
                "after its capability result is seen"
            ),
        },
    )

    rows = representational_rows(representational)
    criteria["representation_beats_shuffled"] = criterion(
        (PASS if rows and all(row["beats_shuffled"] for row in rows) else FAIL)
        if rows
        else NOT_EVALUATED,
        {"directions": rows},
        skipped_because=None if rows else "no cross-modal retrieval pair was computed",
    )
    criteria["jspace_separation_beats_raw"] = criterion(
        (PASS if rows and all(row["jspace_separation_beats_raw"] for row in rows) else FAIL)
        if rows
        else NOT_EVALUATED,
        {"directions": rows},
        skipped_because=None if rows else "no cross-modal retrieval pair was computed",
    )

    cells = evaluate_causal_cells(
        interventions, focal_concepts=focal_concepts, thresholds=thresholds
    )
    by_concept = {
        concept: {
            cell["pair"]: cell["passes"] for cell in cells if cell["concept"] == concept
        }
        for concept in focal_concepts
    }
    bidirectional = sorted(
        concept
        for concept, pairs in by_concept.items()
        if all(pairs.get(pair) for pair in _TEXT_IMAGE_PAIRS)
    )
    unidirectional = sorted(
        concept
        for concept, pairs in by_concept.items()
        if any(pairs.get(pair) for pair in _TEXT_IMAGE_PAIRS)
    )
    causal_skip = "no intervention rows were produced" if not cells else None
    criteria["bidirectional_causal_replication"] = criterion(
        NOT_EVALUATED
        if causal_skip
        else (
            PASS
            if len(bidirectional) >= thresholds.min_focal_concepts_transferring
            else FAIL
        ),
        {
            "focal_concepts": list(focal_concepts),
            "concepts_transferring_both_directions": bidirectional,
            "concepts_transferring_either_direction": unidirectional,
            "required": thresholds.min_focal_concepts_transferring,
            "per_concept": by_concept,
            "cells": cells,
            "reading": (
                "the decision is replication across concepts and directions, "
                "never the single strongest cell"
            ),
        },
        skipped_because=causal_skip,
    )

    passing = [cell for cell in cells if cell["passes"]]
    criteria["distinct_images_per_cell"] = criterion(
        NOT_EVALUATED
        if causal_skip
        else (
            PASS
            if all(
                cell.get("n_positive_images", 0)
                >= thresholds.required_positive_images_per_cell
                and cell.get("n_negative_images", 0)
                >= thresholds.required_negative_images_per_cell
                for cell in cells
                if cell["evaluated"]
            )
            else FAIL
        ),
        {
            "required_positive": thresholds.required_positive_images_per_cell,
            "required_negative": thresholds.required_negative_images_per_cell,
            "per_cell": [
                {
                    "concept": cell["concept"],
                    "pair": cell["pair"],
                    "n_positive_images": cell.get("n_positive_images"),
                    "n_negative_images": cell.get("n_negative_images"),
                }
                for cell in cells
                if cell["evaluated"]
            ],
        },
        skipped_because=causal_skip,
    )

    raw_exceptions = [
        {"concept": cell["concept"], "pair": cell["pair"]}
        for cell in passing
        if not cell["jspace_beats_raw_direction"]
    ]
    criteria["jspace_direction_beats_raw_direction"] = criterion(
        NOT_EVALUATED if causal_skip else (PASS if not raw_exceptions else FAIL),
        {
            "exceptions": raw_exceptions,
            "reading": (
                "a raw difference-in-means direction matching the J-space one "
                "does not undo transfer; it downgrades the claim that the "
                "decomposition earned its keep"
            ),
        },
        skipped_because=causal_skip,
    )

    criteria["external_unrelated_control_is_external"] = criterion(
        PASS
        if unrelated_controls
        and all(
            control not in focal_concepts for control in unrelated_controls.values()
        )
        else FAIL,
        {
            "assignment": dict(sorted(unrelated_controls.items())),
            "focal_concepts": list(focal_concepts),
            "reading": (
                "in a forced choice among the focal concepts the only "
                "alternative is the target's direct contrast, which is not "
                "unrelated to it"
            ),
        },
    )

    criteria["spoken_audio_excluded_by_design"] = criterion(
        PASS,
        {
            "blocked_modalities": list(blocked_modalities),
            "reading": (
                "spoken audio is outside this study by design, not by failure; "
                "environmental audio is not tested at all"
            ),
        },
    )

    statuses = {name: entry["status"] for name, entry in criteria.items()}
    representation_replicates = (
        statuses["six_way_capability"] == PASS
        and statuses["representation_beats_shuffled"] == PASS
        and statuses["jspace_separation_beats_raw"] == PASS
    )
    go_requirements = (
        "six_way_capability",
        "representation_beats_shuffled",
        "jspace_separation_beats_raw",
        "bidirectional_causal_replication",
        "distinct_images_per_cell",
        "external_unrelated_control_is_external",
    )
    all_required = all(statuses.get(name) == PASS for name in go_requirements)

    if all_required and not raw_exceptions:
        verdict, rationale = ROBUSTNESS_GO, (
            f"{len(bidirectional)} of {len(focal_concepts)} focal concepts "
            "transferred in both directions against their own matched random and "
            "external unrelated controls, six-way capability held for every "
            "selected concept, and image-disjoint retrieval beat the shuffled "
            "control in both directions. The photograph was the independent "
            "unit from selection onward."
        )
    elif all_required:
        verdict, rationale = ROBUSTNESS_WEAK_GO, (
            "Every replication requirement was met, but the raw "
            "difference-in-means direction matched or beat the J-space "
            f"direction in {raw_exceptions}. Transfer replicated; the claim "
            "that the J-space decomposition is what carried it is downgraded."
        )
    elif representation_replicates and len(unidirectional) >= (
        thresholds.min_focal_concepts_transferring
    ):
        verdict, rationale = ROBUSTNESS_WEAK_GO, (
            f"Representation replicated and {len(unidirectional)} focal concepts "
            f"transferred in one direction, but only {len(bidirectional)} did so "
            "in both, or control separation was incomplete."
        )
    else:
        failed = sorted(name for name in go_requirements if statuses.get(name) == FAIL)
        verdict, rationale = ROBUSTNESS_NO_GO, (
            f"The result did not replicate at six concepts. Failing: {failed}."
        )

    return {
        "schema": "jlens.mmpilot.robustness_verdict.v1",
        "verdict": verdict,
        "rationale": rationale,
        "criteria": criteria,
        "criteria_status": statuses,
        "go_requirements": list(go_requirements),
        "causal_cells": cells,
        "representational_directions": rows,
        "concepts_transferring_both_directions": bidirectional,
        "concepts_transferring_either_direction": unidirectional,
        "raw_direction_exceptions": raw_exceptions,
        "selected_concepts": list(selected_concepts),
        "focal_concepts": list(focal_concepts),
        "unrelated_controls": dict(sorted(unrelated_controls.items())),
        "thresholds": thresholds.to_dict(),
        "late_layer_limitation": (
            "Layer 38 is late in the decoder and is the only validated layer. A "
            "final-prompt-token edit there cannot establish that the effect "
            "precedes answer-language convergence, so this study does not claim "
            "pre-convergence semantics, however well it replicates."
        ),
        "scope_limitation": (
            "Written text and images only. Spoken audio is excluded by design "
            "and environmental audio is not tested; neither absence is evidence "
            "about either."
        ),
    }


def render_report(
    *,
    run_dir: str,
    verdict: Mapping,
    budget: Mapping | None,
    resume: Mapping | None,
    mode: str,
) -> str:
    """The robustness report. Replication tables first, verdict after."""
    thresholds = verdict["thresholds"]
    mode_note = (
        "  \n- **MOCK run: pipeline evidence only, not scientific evidence.**"
        if mode.lower() == "mock"
        else ""
    )
    lines = [
        f"# Six-concept robustness study — {verdict['verdict']}",
        "",
        f"- mode: **{mode}**{mode_note}",
        f"- run directory: `{run_dir}`",
        f"- selected concepts (ranking order): {verdict['selected_concepts']}",
        f"- focal causal concepts (first three, fixed before any model ran): "
        f"{verdict['focal_concepts']}",
        f"- external unrelated controls: {verdict['unrelated_controls']}",
        "",
        "## Verdict",
        "",
        f"**{verdict['verdict']}** — {verdict['rationale']}",
        "",
        "## Representation (image-disjoint)",
        "",
        "| direction | queries | J top-1 | J MRR | shuffled p95 | beats shuffled | "
        "J gap | raw gap | J beats raw | extra same-image exclusions |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in verdict["representational_directions"]:
        lines.append(
            f"| {row['pair']} | {row['n_queries']} | {row['jspace_top1']:.3f} | "
            f"{row['jspace_mrr']:.3f} | {row['shuffled_p95']:.3f} | "
            f"{'yes' if row['beats_shuffled'] else 'no'} | "
            f"{row['jspace_separation_gap']:+.4f} | {row['raw_separation_gap']:+.4f} | "
            f"{'yes' if row['jspace_separation_beats_raw'] else 'no'} | "
            f"{row['n_excluded_same_image_different_group']} |"
        )

    lines += [
        "",
        "## Causal replication (image is the unit)",
        "",
        "Every cell is judged against **its own** matched controls. The decision "
        "below is replication across concepts and directions — the strongest "
        "cell decides nothing on its own.",
        "",
        "| concept | direction | alpha | +img | -img | effect | sign frac | random | "
        "unrelated | raw | norm | passes |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for cell in verdict["causal_cells"]:
        if not cell.get("evaluated"):
            lines.append(
                f"| {cell['concept']} | {cell['pair']} | | | | | | | | | | "
                f"not evaluated |"
            )
            continue
        lines.append(
            f"| {cell['concept']} | {cell['pair']} | {cell['alpha']:g} | "
            f"{cell['n_positive_images']} | {cell['n_negative_images']} | "
            f"{cell['mean_signed_target_effect']:+.4f} | "
            f"{cell['fraction_expected_sign']:.2f} | "
            f"{_number(cell['random_control'])} | {_number(cell['unrelated_control'])} | "
            f"{_number(cell['raw_residual_control'])} | "
            f"{cell['mean_activation_norm_ratio']:.3f} | "
            f"{'yes' if cell['passes'] else 'no'} |"
        )
    failures = [
        f"- **{cell['concept']} {cell['pair']}**: " + "; ".join(cell["reasons"])
        for cell in verdict["causal_cells"]
        if not cell["passes"] and cell.get("reasons")
    ]
    if failures:
        lines += ["", "Why a cell did not pass:", "", *failures]

    lines += [
        "",
        f"- transferred in **both** directions: "
        f"{verdict['concepts_transferring_both_directions']} "
        f"(required {thresholds['min_focal_concepts_transferring']})",
        f"- transferred in either direction: "
        f"{verdict['concepts_transferring_either_direction']}",
        "",
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
            f"- off-diagonal causal cells: {budget.get('n_causal_cells')}",
            f"- candidates scored per question: {budget.get('n_candidates')}",
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
        f"- {verdict['late_layer_limitation']}",
        f"- {verdict['scope_limitation']}",
        "- Interventions add and subtract a direction on the residual stream. "
        "That is not erasure and not projection ablation.",
        "- One photograph contributes one synchronized group. Repeated captions "
        "of an image were excluded at selection, not averaged away afterwards.",
        "- This replicates the pilot at six concepts. It does not extend it to "
        "new layers, new modalities, or a new lens.",
        "",
    ]
    return "\n".join(lines)


def _number(value) -> str:
    return "n/a" if value is None else f"{float(value):+.4f}"


__all__ = [
    "ROBUSTNESS_GO",
    "ROBUSTNESS_NO_GO",
    "ROBUSTNESS_VERDICT_VERSION",
    "ROBUSTNESS_WEAK_GO",
    "PassBudget",
    "RobustnessThresholds",
    "estimate_model_passes",
    "evaluate_causal_cells",
    "format_budget",
    "render_report",
    "representational_rows",
    "robustness_verdict",
]
