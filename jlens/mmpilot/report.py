# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""The GO / WEAK GO / NO-GO decision, as an explicit rubric.

The rubric is evaluated in code so the recommendation cannot drift with how
the numbers are read. Each criterion carries the evidence that decided it, and
the Markdown report answers the seven questions the pilot was commissioned to
answer — including the ones whose honest answer is "no".

A pipeline that merely executes is not evidence. MOCK runs are labelled as
such in every artifact and can never produce a scientific GO.

Three states, not two
---------------------

A criterion is ``PASS``, ``FAIL``, or ``NOT_EVALUATED``. The third exists
because a boolean cannot tell "we measured this and it was bad" apart from "we
never measured this". The first real run stopped at the behavioral gate, so no
J-space code was ever computed — and the report said the lens had not
reconstructed activations well enough, which was a claim about a measurement
that did not happen. A stage that was skipped is reported as skipped, with the
precondition that stopped it named.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

#: Criterion states. ``passed`` is kept alongside them for callers that only
#: ask the yes/no question, and is True only for :data:`PASS`.
PASS = "PASS"
FAIL = "FAIL"
NOT_EVALUATED = "NOT_EVALUATED"

#: Thresholds the rubric applies. Explicit so the report can print them.
#:
#: There is deliberately **no** absolute reconstruction threshold. An earlier
#: version of this rubric required the frozen lens to explain 50% of a held-out
#: activation's variance. That number has no basis: the published J-space result
#: reports a median of roughly 6-7% of a concept vector's variance in its top-k
#: J-space component, and excess over a same-size random control that never
#: exceeds about 10%. A 50% gate would fail the published result. The lens
#: criterion is now excess over matched random controls — see
#: :mod:`jlens.mmpilot.reconstruction`.
DEFAULT_THRESHOLDS = {
    "min_concepts_text_image": 2,
    "min_median_excess_explained_fraction": 0.0,
    "min_layers_above_random": 1,
    "min_retrieval_margin_over_shuffled": 0.05,
    "min_fraction_expected_sign": 0.75,
    "control_separation_factor": 1.5,
    "norm_ratio_bounds": [0.5, 2.0],
}

_TEXT_IMAGE_PAIRS = ("text->image", "image->text")


def _rows(summary: Mapping, **filters) -> list[dict]:
    out = []
    for row in summary.get("rows", []):
        if all(row.get(key) == value for key, value in filters.items()):
            out.append(dict(row))
    return out


def _best_effect_row(summary: Mapping, *, control_kind: str) -> dict | None:
    """Strongest off-diagonal text/image row for one control kind."""
    candidates = [
        row
        for row in summary.get("rows", [])
        if row["control_kind"] == control_kind
        and row["off_diagonal"]
        and row["alpha"] > 0
        and f"{row['source_modality']}->{row['target_modality']}" in _TEXT_IMAGE_PAIRS
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda row: row["mean_signed_target_effect"])


def criterion(status: str, evidence: Mapping, *, skipped_because: str | None = None) -> dict:
    """One criterion entry. ``passed`` is True only for :data:`PASS`."""
    entry = {
        "status": status,
        "passed": status == PASS,
        "evaluated": status != NOT_EVALUATED,
        "evidence": dict(evidence),
    }
    if status == NOT_EVALUATED:
        entry["not_evaluated_reason"] = skipped_because or "stage did not run"
    return entry


def evaluate_criteria(
    *,
    capability: Mapping,
    lens_validation: Mapping | None,
    code_stats: Mapping,
    representational: Mapping,
    interventions: Mapping,
    reconstruction_control: Mapping | None = None,
    blocked_modalities: Sequence[str] = (),
    thresholds: Mapping | None = None,
) -> dict:
    """Evaluate every rubric criterion.

    Returns ``{name: {status, passed, evaluated, evidence}}`` where ``status``
    is ``PASS``, ``FAIL``, or ``NOT_EVALUATED``. A criterion whose stage never
    ran is ``NOT_EVALUATED``; it is never reported as a failure, and the
    precondition that stopped it is named in ``not_evaluated_reason``.
    """
    limits = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    criteria: dict[str, dict] = {}

    retained = list(capability.get("text_image_retained_concepts", []))
    behavioral_passed = len(retained) >= limits["min_concepts_text_image"]
    criteria["behavioral_capability"] = criterion(
        PASS if behavioral_passed else FAIL,
        {
            "retained_text_image_concepts": retained,
            "required": limits["min_concepts_text_image"],
            "per_concept": capability.get("per_concept", {}),
        },
    )

    # What stopped the run, named once and reused by every skipped criterion.
    precondition = (
        "the behavioral capability gate failed, so no activations were captured"
        if not behavioral_passed
        else "an earlier scientific precondition failed"
    )

    # ------------------------------------------------ lens sanity vs. random
    #
    # The question is NOT "did the lens explain a lot of the activation". A
    # sparse workspace is expected to explain little; the published result puts
    # the median around 6-7%. The question is whether it explained more than
    # random directions matched to it in count, norm, width and pursuit.
    control = dict(reconstruction_control or {})
    by_layer = control.get("by_layer", {}) or {}
    layers_above = list(control.get("layers_above_random", []))
    primary_layer = control.get("primary_layer")
    primary_entry = by_layer.get(str(primary_layer)) if primary_layer is not None else None
    healthy = bool(by_layer) and all(
        entry.get("all_finite") and entry.get("all_nondegenerate")
        for entry in by_layer.values()
    )
    sanity_evidence = {
        "n_codes": code_stats.get("n"),
        "n_control_records": control.get("n_records", 0),
        # Descriptive: reported, never gated on.
        "absolute_median_explained_fraction": code_stats.get("median_explained_fraction"),
        "absolute_heldout_text_median_explained_fraction": code_stats.get(
            "heldout_text_median_explained_fraction"
        ),
        # The criterion.
        "primary_layer": primary_layer,
        "primary_layer_median_excess_explained_fraction": (primary_entry or {}).get(
            "median_excess_explained_fraction"
        ),
        "primary_layer_median_excess_over_random_bound": (primary_entry or {}).get(
            "median_excess_over_random_bound"
        ),
        "layers_above_random": layers_above,
        "required_layers_above_random": limits["min_layers_above_random"],
        "min_median_excess": limits["min_median_excess_explained_fraction"],
        "pursuit_outputs_healthy": healthy,
        "by_layer": by_layer,
        "control_config_hash": control.get("control_config_hash"),
        "lens_checksum": (lens_validation or {}).get("lens_checksum"),
        "absolute_reconstruction_is_not_gated": (
            "no absolute explained-fraction threshold is applied; the published "
            "J-space result reports a median around 6-7%, so a high absolute "
            "bar would reject a working lens"
        ),
        "control_limitation": (control.get("config", {}) or {}).get("interpretation"),
    }
    if not control.get("n_records"):
        criteria["lens_sanity_above_random"] = criterion(
            NOT_EVALUATED,
            sanity_evidence,
            skipped_because=(
                "no J-space codes and no matched-random control results: the "
                f"reconstruction-control stage was skipped because {precondition}. "
                "Whether the lens beats random directions is unknown — this is "
                "not a finding that it does not."
            ),
        )
    else:
        passed = bool(
            lens_validation
            and healthy
            and len(layers_above) >= limits["min_layers_above_random"]
            and (
                primary_entry is None
                or primary_entry.get("above_random")
                or len(layers_above) >= limits["min_layers_above_random"]
            )
        )
        criteria["lens_sanity_above_random"] = criterion(
            PASS if passed else FAIL, sanity_evidence
        )

    pairs = representational.get("pairs", {})
    structure_rows = []
    for key in _TEXT_IMAGE_PAIRS:
        entry = pairs.get(key)
        if not entry:
            continue
        jspace = entry["jspace_retrieval"]["top1_accuracy"]
        shuffled = entry["shuffled_control"]["p95_top1_accuracy"]
        raw = entry["raw_residual_retrieval"]["top1_accuracy"]
        jspace_gap = entry["jspace_separation"]["gap"]
        raw_gap = entry["raw_residual_separation"]["gap"]
        structure_rows.append(
            {
                "pair": key,
                "jspace_top1": jspace,
                "shuffled_p95_top1": shuffled,
                "raw_top1": raw,
                "jspace_gap": jspace_gap,
                "raw_gap": raw_gap,
                "beats_shuffled": jspace
                >= shuffled + limits["min_retrieval_margin_over_shuffled"],
                "beats_raw": (jspace > raw)
                or (jspace == raw and jspace_gap > raw_gap),
            }
        )
    representation_evaluable = len(retained) >= 2
    structure_evidence = {
        "evaluable": representation_evaluable,
        "reason": None
        if representation_evaluable
        else "fewer than two retained concepts makes retrieval and label shuffling trivial",
        "text_image_pairs": structure_rows,
    }
    if not structure_rows:
        structure_status, structure_skip = NOT_EVALUATED, (
            "no text/image retrieval pair was computed: the representational "
            f"stage was skipped because {precondition}"
        )
    elif not representation_evaluable:
        structure_status, structure_skip = NOT_EVALUATED, (
            "fewer than two retained concepts makes retrieval and label "
            "shuffling trivial, so the test was not run as a test"
        )
    else:
        structure_status, structure_skip = (
            PASS if all(row["beats_shuffled"] for row in structure_rows) else FAIL
        ), None
    criteria["representational_structure"] = criterion(
        structure_status, structure_evidence, skipped_because=structure_skip
    )
    criteria["jspace_beats_raw_residual"] = criterion(
        structure_status
        if structure_status == NOT_EVALUATED
        else (PASS if all(row["beats_raw"] for row in structure_rows) else FAIL),
        {"evaluable": representation_evaluable, "text_image_pairs": structure_rows},
        skipped_because=structure_skip,
    )

    # An intervention stage that never ran leaves no rows at all. Rows present
    # but no off-diagonal cell is a real null result, and stays a FAIL.
    intervention_rows = list(interventions.get("rows", []))
    causal_skip = (
        "no intervention was run: the causal stage was skipped because "
        f"{precondition}"
        if not intervention_rows
        else None
    )

    best = _best_effect_row(interventions, control_kind="source_concept")
    criteria["causal_transfer_sign"] = criterion(
        NOT_EVALUATED
        if causal_skip
        else (
            PASS
            if best
            and best["mean_signed_target_effect"] > 0
            and best["fraction_expected_sign"] >= limits["min_fraction_expected_sign"]
            else FAIL
        ),
        {"best_off_diagonal_row": best, "required_fraction": limits["min_fraction_expected_sign"]},
        skipped_because=causal_skip,
    )

    # The rubric's specificity test is against the random and unrelated-concept
    # controls. The raw-residual direction is a *baseline* — it answers "did the
    # J-space decomposition buy anything", not "is the effect real" — so it is
    # measured and reported here but never allowed to veto specificity.
    control_rows: dict[str, dict | None] = {}
    specificity_passed = False
    if best:
        for kind in ("random_norm_matched", "unrelated_concept", "raw_residual_difference"):
            matched = _rows(
                interventions,
                concept=best["concept"],
                source_modality=best["source_modality"],
                target_modality=best["target_modality"],
                layer=best["layer"],
                control_kind=kind,
                alpha=best["alpha"],
            )
            control_rows[kind] = matched[0] if matched else None
        blocking = [
            control_rows.get(kind)
            for kind in ("random_norm_matched", "unrelated_concept")
        ]
        if all(row is not None for row in blocking):
            strongest = max(row["mean_signed_target_effect"] for row in blocking)
            specificity_passed = best["mean_signed_target_effect"] >= (
                limits["control_separation_factor"] * max(strongest, 0.0)
            ) and best["mean_signed_target_effect"] > 0
    criteria["control_specificity"] = criterion(
        NOT_EVALUATED if causal_skip else (PASS if specificity_passed else FAIL),
        {
            "best_row": best,
            "blocking_controls": ["random_norm_matched", "unrelated_concept"],
            "matched_controls": control_rows,
            "factor_required": limits["control_separation_factor"],
        },
        skipped_because=causal_skip,
    )

    raw_row = control_rows.get("raw_residual_difference")
    criteria["jspace_direction_beats_raw_direction"] = criterion(
        NOT_EVALUATED
        if causal_skip
        else (
            PASS
            if best
            and raw_row is not None
            and best["mean_signed_target_effect"] >= raw_row["mean_signed_target_effect"]
            else FAIL
        ),
        {
            "jspace_effect": (best or {}).get("mean_signed_target_effect"),
            "raw_residual_effect": (raw_row or {}).get("mean_signed_target_effect"),
            "reading": (
                "reported, not blocking: a raw difference-in-means direction "
                "beating the J-space direction is informative about whether the "
                "decomposition earns its keep, not about whether transfer happened"
            ),
        },
        skipped_because=causal_skip,
    )

    low, high = limits["norm_ratio_bounds"]
    norm_ok = bool(best) and low <= best["mean_activation_norm_ratio"] <= high
    criteria["activation_norm_sanity"] = criterion(
        NOT_EVALUATED if causal_skip else (PASS if norm_ok else FAIL),
        {
            "mean_activation_norm_ratio": (best or {}).get("mean_activation_norm_ratio"),
            "bounds": [low, high],
        },
        skipped_because=causal_skip,
    )

    specific_not_global = bool(best) and best["mean_abs_unrelated_change"] < abs(
        best["mean_signed_target_effect"]
    )
    criteria["effect_specificity_not_global"] = criterion(
        NOT_EVALUATED if causal_skip else (PASS if specific_not_global else FAIL),
        {
            "mean_signed_target_effect": (best or {}).get("mean_signed_target_effect"),
            "mean_abs_unrelated_change": (best or {}).get(
                "mean_abs_unrelated_change"
            ),
        },
        skipped_because=causal_skip,
    )
    criteria["effect_precedes_output_convergence"] = criterion(
        NOT_EVALUATED
        if causal_skip
        else (
            PASS
            if bool((best or {}).get("pre_output_convergence_validated", False))
            else FAIL
        ),
        {
            "layer": (best or {}).get("layer"),
            "mean_signed_target_effect": (best or {}).get("mean_signed_target_effect"),
            "mean_abs_unrelated_change": (best or {}).get("mean_abs_unrelated_change"),
            "reading": (
                "not established by a final-prompt intervention alone; a late "
                "decoder layer may already contain an answer-scoring direction"
            ),
        },
        skipped_because=causal_skip,
    )

    # An observation about this checkpoint/processor, available whether or not
    # any downstream stage ran. Always evaluated.
    criteria["spoken_audio_available"] = criterion(
        PASS if "spoken_audio" not in blocked_modalities else FAIL,
        {"blocked_modalities": list(blocked_modalities)},
    )
    return criteria


def decide(criteria: Mapping) -> dict:
    """Apply the rubric. Returns ``{recommendation, rationale, next_experiment}``.

    A ``NOT_EVALUATED`` criterion never contributes a failure to the rationale.
    The recommendation may still be NO-GO — an unanswered question is not a
    reason to spend GPU hours — but the report says the stage was skipped, not
    that it went badly.
    """

    def status(name: str) -> str:
        return str(criteria.get(name, {}).get("status", FAIL))

    def ok(name: str) -> bool:
        return status(name) == PASS

    def failed(name: str) -> bool:
        return status(name) == FAIL

    skipped = sorted(
        name for name in criteria if criteria[name].get("status") == NOT_EVALUATED
    )
    skipped_note = (
        " Not evaluated (skipped, not failed): " + ", ".join(skipped) + "."
        if skipped
        else ""
    )
    representational = ok("behavioral_capability") and ok("representational_structure")
    causal = ok("causal_transfer_sign") and ok("control_specificity")
    healthy = ok("activation_norm_sanity") and ok("effect_specificity_not_global")

    if failed("behavioral_capability") or failed("lens_sanity_above_random"):
        failures = []
        if failed("behavioral_capability"):
            failures.append(
                "fewer than the required number of concepts passed the "
                "behavioral gate for both text and image"
            )
        if failed("lens_sanity_above_random"):
            # Note what this does and does not say: the lens reconstruction was
            # indistinguishable from matched random directions. It is never a
            # statement about the absolute share of variance explained.
            failures.append(
                "the frozen lens did not reconstruct held-out activations any "
                "better than random directions matched to it in count, norm and "
                "pursuit, so its coordinates are not carrying lens-specific "
                "structure here (this is not a claim about the absolute share "
                "of variance explained, which is expected to be small)"
            )
        return {
            "recommendation": "NO-GO",
            "rationale": (
                "The precondition failed: " + "; ".join(failures) + "." + skipped_note
            ),
            "next_experiment": (
                "Run the CPU-only synchronized-evidence audit first: a positive "
                "requires the COCO object annotation AND the concept in the "
                "written caption, and a group carrying only the annotation asks "
                "the text arm about a concept its caption never states. Re-rank "
                "concepts on valid synchronized positives only, and spend GPU "
                "time only if at least two concepts still clear the thresholds."
            ),
        }
    if skipped and not representational:
        # Behavioral capability held, but the run did not reach the stages that
        # would answer the question. Absence of evidence, reported as absence.
        return {
            "recommendation": "NO-GO",
            "rationale": (
                "The pilot did not reach the stages that answer its question. "
                "No criterion below the behavioral gate produced a measurement, "
                "so nothing here is evidence for or against cross-modal "
                "transfer." + skipped_note
            ),
            "next_experiment": (
                "Resume the run in the same directory: every checksum-valid "
                "unit is reused, so only the skipped stages are recomputed."
            ),
        }
    if representational and causal and healthy:
        return {
            "recommendation": "GO",
            "rationale": (
                "Concepts are behaviorally readable from both text and image, "
                "cross-modal J-space structure beats the shuffled control, and a "
                "source-derived direction moved a held-out target-modality "
                "example in the intended direction by more than the random, "
                "unrelated-concept, and raw-residual controls, without wrecking "
                "activation norms or shifting every candidate equally."
            ),
            "next_experiment": (
                "Scale to more concepts, more groups per concept, and both "
                "layers before adding any new machinery."
            ),
        }
    if representational:
        return {
            "recommendation": "WEAK GO",
            "rationale": (
                "Representation-level cross-modal structure is present, but the "
                "causal transfer is weak, inconsistent, or not separated from "
                "the controls." + skipped_note
            ),
            "next_experiment": (
                "Smallest next step: hold the concept and layer that produced "
                "the strongest off-diagonal effect fixed, and run only that "
                "cell with more held-out target examples and a denser alpha "
                "sweep. That resolves whether the effect is real or noise "
                "without rebuilding anything."
            ),
        }
    return {
        "recommendation": "NO-GO",
        "rationale": (
            "Cross-modal structure in the frozen J-space was not distinguishable "
            "from the shuffled-label control, so there is nothing for a causal "
            "test to be about." + skipped_note
        ),
        "next_experiment": (
            "Either the lens layer is wrong for multimodal states or the "
            "final-prompt-token position does not carry the concept. Test that "
            "with a cheap layer sweep on the text condition alone before "
            "investing in a larger framework."
        ),
    }


def code_statistics(codes: Sequence[Mapping]) -> dict:
    """Median/mean reconstruction quality and sparsity over all J-space codes."""
    if not codes:
        return {"n": 0, "median_explained_fraction": None}
    explained = sorted(float(code["explained_fraction"]) for code in codes)
    actives = sorted(int(code["n_active"]) for code in codes)
    by_modality = {}
    for modality in sorted(
        {str(code.get("modality")) for code in codes if code.get("modality")}
    ):
        values = sorted(
            float(code["explained_fraction"])
            for code in codes
            if code.get("modality") == modality
        )
        by_modality[modality] = {
            "n": len(values),
            "median_explained_fraction": values[len(values) // 2],
            "mean_explained_fraction": sum(values) / len(values),
        }
    heldout_text = sorted(
        float(code["explained_fraction"])
        for code in codes
        if code.get("modality") == "text" and code.get("split") == "test"
    )
    return {
        "n": len(codes),
        "median_explained_fraction": explained[len(explained) // 2],
        "min_explained_fraction": explained[0],
        "mean_explained_fraction": sum(explained) / len(explained),
        "median_n_active": actives[len(actives) // 2],
        "convergence_statuses": sorted({str(code["convergence_status"]) for code in codes}),
        "by_modality": by_modality,
        "text_median_explained_fraction": by_modality.get("text", {}).get(
            "median_explained_fraction"
        ),
        "heldout_text_median_explained_fraction": (
            heldout_text[len(heldout_text) // 2] if heldout_text else None
        ),
    }


def reconstruction_sentence(criteria: Mapping, code_stats: Mapping) -> str:
    """The one paragraph that must never conflate absolute with excess.

    Absolute explained fraction, excess over matched random, occupancy and
    causal usefulness are four different things. This says the first two and
    points at the rest.
    """
    entry = criteria.get("lens_sanity_above_random", {})
    evidence = entry.get("evidence", {})
    absolute = evidence.get("absolute_heldout_text_median_explained_fraction")
    if absolute is None:
        absolute = code_stats.get("median_explained_fraction")
    if entry.get("status") == NOT_EVALUATED:
        return (
            "J-space reconstruction was **NOT EVALUATED**: no codes and no "
            "matched-random controls were produced, because "
            f"{entry.get('not_evaluated_reason', 'an earlier gate stopped the run')} "
            "Nothing here says the lens reconstructed poorly."
        )
    excess = evidence.get("primary_layer_median_excess_explained_fraction")
    over_bound = evidence.get("primary_layer_median_excess_over_random_bound")
    verdict = entry.get("status")
    occupancy_by_layer = {
        layer: value.get("median_estimated_occupancy")
        for layer, value in (evidence.get("by_layer") or {}).items()
    }
    return (
        f"J-space reconstructed {_percent(absolute)} of total activation "
        "variance. Low absolute reconstruction is expected for a sparse "
        "workspace and is not gated on. Its median excess over matched random "
        f"controls was {_percent(excess)} at the primary layer "
        f"({_percent(over_bound)} beyond the control's upper bound), and layers "
        f"clearing the control: {evidence.get('layers_above_random')}. "
        f"Estimated occupancy per layer: {occupancy_by_layer}. "
        f"So the lens sanity criterion is **{verdict}**. "
        "Reconstruction says nothing on its own about causal usefulness — that "
        "is what the intervention criteria below are for."
    )


def _percent(value) -> str:
    return "unknown" if value is None else f"{100.0 * float(value):.2f}%"


def gonogo_report(
    *,
    mode: str,
    run_dir: str,
    capability: Mapping,
    lens_validation: Mapping | None,
    code_stats: Mapping,
    representational: Mapping,
    interventions: Mapping,
    invariance: Mapping | None,
    reconstruction_control: Mapping | None = None,
    blocked_modalities: Sequence[str] = (),
    manifest_audit: Mapping | None = None,
    thresholds: Mapping | None = None,
) -> tuple[str, dict]:
    """Build the Markdown report and the JSON summary.

    Returns ``(markdown, summary)``. In MOCK mode the recommendation is
    reported as pipeline evidence only and is prefixed accordingly.
    """
    criteria = evaluate_criteria(
        capability=capability,
        lens_validation=lens_validation,
        code_stats=code_stats,
        representational=representational,
        interventions=interventions,
        reconstruction_control=reconstruction_control,
        blocked_modalities=blocked_modalities,
        thresholds=thresholds,
    )
    decision = decide(criteria)
    is_mock = mode.lower() == "mock"
    is_tiny_smoke = mode.lower() == "tiny_smoke"
    is_scientific = not (is_mock or is_tiny_smoke)
    not_evaluated = {
        name: entry.get("not_evaluated_reason")
        for name, entry in criteria.items()
        if entry.get("status") == NOT_EVALUATED
    }
    summary = {
        "schema": "jlens.mmpilot.gonogo.v2",
        "mode": mode,
        "run_dir": run_dir,
        "scientific_evidence": is_scientific,
        "recommendation": decision["recommendation"],
        "rationale": decision["rationale"],
        "next_experiment": decision["next_experiment"],
        "criteria": criteria,
        "criteria_status": {name: entry["status"] for name, entry in criteria.items()},
        "not_evaluated": not_evaluated,
        "thresholds": {**DEFAULT_THRESHOLDS, **(thresholds or {})},
        "blocked_modalities": list(blocked_modalities),
        "capability": capability,
        "lens_validation": lens_validation,
        "code_statistics": code_stats,
        "reconstruction_control": reconstruction_control,
        "representational": representational,
        "interventions": interventions,
        "invariance": invariance,
        "manifest_audit": manifest_audit,
    }

    def yes_no(name: str) -> str:
        status = criteria.get(name, {}).get("status", FAIL)
        return {PASS: "yes", FAIL: "no", NOT_EVALUATED: "NOT EVALUATED"}[status]

    def is_skipped(name: str) -> bool:
        return criteria.get(name, {}).get("status") == NOT_EVALUATED

    best = criteria["causal_transfer_sign"]["evidence"].get("best_off_diagonal_row")
    if is_tiny_smoke:
        mode_note = "  \n- **TINY_SMOKE run: plumbing validation only, NOT scientifically meaningful, never used for GO/NO-GO research verdict.**"
    elif is_mock:
        mode_note = "  \n- **MOCK run: pipeline evidence only, not scientific evidence.**"
    else:
        mode_note = ""

    lines = [
        f"# Multimodal J-space transfer pilot — {decision['recommendation']}",
        "",
        f"- mode: **{mode}**{mode_note}",
        f"- run directory: `{run_dir}`",
        f"- modalities blocked: {list(blocked_modalities) or 'none'}",
        "",
        "## Recommendation",
        "",
        f"**{decision['recommendation']}** — {decision['rationale']}",
        "",
        f"Smallest next experiment: {decision['next_experiment']}",
        "",
        "## The seven questions",
        "",
        f"1. **Did Gemma behaviorally recognize the concepts in each modality?** "
        f"{yes_no('behavioral_capability')} — concepts passing for both text and "
        f"image: {criteria['behavioral_capability']['evidence']['retained_text_image_concepts']}.",
        f"2. **Did the frozen J-lens reconstruct held-out activations better "
        f"than matched random directions?** {yes_no('lens_sanity_above_random')} "
        f"— {reconstruction_sentence(criteria, code_stats)}",
        f"3. **Was J-space structure stronger than raw residual space?** "
        f"{yes_no('jspace_beats_raw_residual')} (structure above the shuffled "
        f"control: {yes_no('representational_structure')}).",
        f"4. **Did a source-derived direction causally transfer to another "
        f"modality?** {yes_no('causal_transfer_sign')}"
        + (
            f" — strongest off-diagonal cell: {best['pair']} concept "
            f"{best['concept']!r} at layer {best['layer']}, alpha {best['alpha']}, "
            f"mean signed effect {best['mean_signed_target_effect']:.4f}."
            if best
            else (
                " — the intervention stage did not run."
                if is_skipped("causal_transfer_sign")
                else " — no off-diagonal cell produced an effect."
            )
        ),
        f"5. **Were the effects larger and more specific than controls?** "
        f"{yes_no('control_specificity')}.",
        f"6. **Did results occur before obvious output-language convergence?** "
        f"{yes_no('effect_precedes_output_convergence')} — a final-prompt edit "
        f"does not by itself establish that the effect precedes answer-language convergence.",
        f"7. **Is there enough signal to justify the larger framework?** "
        f"{decision['recommendation']}.",
        "",
        "## Criteria",
        "",
        "`NOT EVALUATED` means the stage never ran. It is not a failed test and "
        "must not be read as one.",
        "",
        "| criterion | status | note |",
        "| --- | --- | --- |",
    ]
    lines += [
        f"| {name} | {entry['status'].replace('_', ' ')} | "
        f"{entry.get('not_evaluated_reason', '') or ''} |"
        for name, entry in criteria.items()
    ]
    if not_evaluated:
        lines += [
            "",
            "## Not evaluated",
            "",
            "These criteria produced no measurement. Nothing below is evidence "
            "for or against the hypothesis:",
            "",
        ]
        lines += [f"- **{name}** — {reason}" for name, reason in not_evaluated.items()]
    lines += [
        "",
        "## Scope and interpretation",
        "",
        "- SpokenCOCO supplies images, written captions, and spoken readings of "
        "those captions, so this tests transfer among visual, written-linguistic, "
        "and spoken-linguistic evidence. It says nothing about environmental audio.",
        "- Interventions add and subtract a direction on the residual stream. "
        "That is not erasure and not projection ablation.",
        "- Retrieval excludes each query's own synchronized group, so a hit is "
        "not the dataset's own pairing being read back.",
        "- There is **no absolute reconstruction threshold**. The published "
        "J-space result reports a median around 6-7% of a concept vector's "
        "variance in its top-k J-space component, so a high absolute bar would "
        "reject a working lens. The criterion is excess over matched random "
        "controls.",
        "- The random control matches atom count, atom norms, hidden width, "
        "dtype, device and pursuit settings — but not the selection pool. The "
        "J-lens picks its atoms from the whole vocabulary, so clearing this "
        "control is a necessary condition, not a strong one.",
        "- Absolute reconstruction, excess over random, occupancy and causal "
        "usefulness are four separate readings. None implies another.",
        "- A positive requires **both** the COCO object annotation and the "
        "concept (or an approved synonym) in the written caption. A group "
        "carrying only the annotation asks the text arm about something its "
        "caption never states, and is not a valid synchronized positive.",
        "",
        "## Raw summary",
        "",
        "```json",
        json.dumps(
            {
                "recommendation": summary["recommendation"],
                "criteria": {k: v["status"] for k, v in criteria.items()},
                "not_evaluated": not_evaluated,
                "code_statistics": code_stats,
                "intervention_rows": interventions.get("rows", [])[:12],
            },
            indent=2,
            default=str,
        ),
        "```",
        "",
    ]
    return "\n".join(lines), summary
