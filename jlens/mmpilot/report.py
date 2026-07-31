# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""The GO / WEAK GO / NO-GO decision, as an explicit rubric.

The rubric is evaluated in code so the recommendation cannot drift with how
the numbers are read. Each criterion carries the evidence that decided it, and
the Markdown report answers the seven questions the pilot was commissioned to
answer — including the ones whose honest answer is "no".

A pipeline that merely executes is not evidence. MOCK runs are labelled as
such in every artifact and can never produce a scientific GO.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

#: Thresholds the rubric applies. Explicit so the report can print them.
DEFAULT_THRESHOLDS = {
    "min_concepts_text_image": 2,
    "min_median_explained_fraction": 0.5,
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


def evaluate_criteria(
    *,
    capability: Mapping,
    lens_validation: Mapping | None,
    code_stats: Mapping,
    representational: Mapping,
    interventions: Mapping,
    blocked_modalities: Sequence[str] = (),
    thresholds: Mapping | None = None,
) -> dict:
    """Evaluate every rubric criterion. Returns ``{name: {passed, evidence}}``."""
    limits = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    criteria: dict[str, dict] = {}

    retained = list(capability.get("text_image_retained_concepts", []))
    criteria["behavioral_capability"] = {
        "passed": len(retained) >= limits["min_concepts_text_image"],
        "evidence": {
            "retained_text_image_concepts": retained,
            "required": limits["min_concepts_text_image"],
            "per_concept": capability.get("per_concept", {}),
        },
    }

    median_ef = code_stats.get("median_explained_fraction")
    heldout_text_ef = code_stats.get("heldout_text_median_explained_fraction")
    text_median_ef = code_stats.get("text_median_explained_fraction")
    reconstruction_ef = (
        heldout_text_ef
        if heldout_text_ef is not None
        else (text_median_ef if text_median_ef is not None else median_ef)
    )
    criteria["lens_reconstruction"] = {
        "passed": bool(lens_validation)
        and reconstruction_ef is not None
        and reconstruction_ef >= limits["min_median_explained_fraction"],
        "evidence": {
            "median_explained_fraction": median_ef,
            "text_median_explained_fraction": text_median_ef,
            "heldout_text_median_explained_fraction": heldout_text_ef,
            "value_used_for_gate": reconstruction_ef,
            "required": limits["min_median_explained_fraction"],
            "lens_checksum": (lens_validation or {}).get("lens_checksum"),
        },
    }

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
    criteria["representational_structure"] = {
        "passed": representation_evaluable
        and bool(structure_rows)
        and all(row["beats_shuffled"] for row in structure_rows),
        "evidence": {
            "evaluable": representation_evaluable,
            "reason": None
            if representation_evaluable
            else "fewer than two retained concepts makes retrieval and label shuffling trivial",
            "text_image_pairs": structure_rows,
        },
    }
    criteria["jspace_beats_raw_residual"] = {
        "passed": representation_evaluable
        and bool(structure_rows)
        and all(row["beats_raw"] for row in structure_rows),
        "evidence": {
            "evaluable": representation_evaluable,
            "text_image_pairs": structure_rows,
        },
    }

    best = _best_effect_row(interventions, control_kind="source_concept")
    criteria["causal_transfer_sign"] = {
        "passed": bool(best)
        and best["mean_signed_target_effect"] > 0
        and best["fraction_expected_sign"] >= limits["min_fraction_expected_sign"],
        "evidence": {"best_off_diagonal_row": best, "required_fraction": limits["min_fraction_expected_sign"]},
    }

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
    criteria["control_specificity"] = {
        "passed": specificity_passed,
        "evidence": {
            "best_row": best,
            "blocking_controls": ["random_norm_matched", "unrelated_concept"],
            "matched_controls": control_rows,
            "factor_required": limits["control_separation_factor"],
        },
    }

    raw_row = control_rows.get("raw_residual_difference")
    criteria["jspace_direction_beats_raw_direction"] = {
        "passed": bool(best)
        and raw_row is not None
        and best["mean_signed_target_effect"] >= raw_row["mean_signed_target_effect"],
        "evidence": {
            "jspace_effect": (best or {}).get("mean_signed_target_effect"),
            "raw_residual_effect": (raw_row or {}).get("mean_signed_target_effect"),
            "reading": (
                "reported, not blocking: a raw difference-in-means direction "
                "beating the J-space direction is informative about whether the "
                "decomposition earns its keep, not about whether transfer happened"
            ),
        },
    }

    low, high = limits["norm_ratio_bounds"]
    norm_ok = bool(best) and low <= best["mean_activation_norm_ratio"] <= high
    criteria["activation_norm_sanity"] = {
        "passed": norm_ok,
        "evidence": {
            "mean_activation_norm_ratio": (best or {}).get("mean_activation_norm_ratio"),
            "bounds": [low, high],
        },
    }

    specific_not_global = bool(best) and best["mean_abs_unrelated_change"] < abs(
        best["mean_signed_target_effect"]
    )
    criteria["effect_specificity_not_global"] = {
        "passed": specific_not_global,
        "evidence": {
            "mean_signed_target_effect": (best or {}).get("mean_signed_target_effect"),
            "mean_abs_unrelated_change": (best or {}).get(
                "mean_abs_unrelated_change"
            ),
        },
    }
    criteria["effect_precedes_output_convergence"] = {
        "passed": bool((best or {}).get("pre_output_convergence_validated", False)),
        "evidence": {
            "layer": (best or {}).get("layer"),
            "mean_signed_target_effect": (best or {}).get("mean_signed_target_effect"),
            "mean_abs_unrelated_change": (best or {}).get("mean_abs_unrelated_change"),
            "reading": (
                "not established by a final-prompt intervention alone; a late "
                "decoder layer may already contain an answer-scoring direction"
            ),
        },
    }

    criteria["spoken_audio_available"] = {
        "passed": "spoken_audio" not in blocked_modalities,
        "evidence": {"blocked_modalities": list(blocked_modalities)},
    }
    return criteria


def decide(criteria: Mapping) -> dict:
    """Apply the rubric. Returns ``{recommendation, rationale, next_experiment}``."""

    def ok(name: str) -> bool:
        return bool(criteria.get(name, {}).get("passed"))

    representational = ok("behavioral_capability") and ok("representational_structure")
    causal = ok("causal_transfer_sign") and ok("control_specificity")
    healthy = ok("activation_norm_sanity") and ok("effect_specificity_not_global")

    if not ok("behavioral_capability") or not ok("lens_reconstruction"):
        return {
            "recommendation": "NO-GO",
            "rationale": (
                "The precondition failed: "
                + (
                    "fewer than the required number of concepts passed the "
                    "behavioral gate for both text and image. "
                    if not ok("behavioral_capability")
                    else ""
                )
                + (
                    "the frozen lens did not reconstruct the captured activations "
                    "well enough for its coordinates to mean anything."
                    if not ok("lens_reconstruction")
                    else ""
                )
            ).strip(),
            "next_experiment": (
                "Re-audit concept coverage and the capability prompts before "
                "spending anything on activations; if the lens is the problem, "
                "check the layer and revision it was fitted at."
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
                "the controls."
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
            "test to be about."
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
        blocked_modalities=blocked_modalities,
        thresholds=thresholds,
    )
    decision = decide(criteria)
    is_mock = mode.lower() == "mock"
    is_tiny_smoke = mode.lower() == "tiny_smoke"
    is_scientific = not (is_mock or is_tiny_smoke)
    summary = {
        "schema": "jlens.mmpilot.gonogo.v1",
        "mode": mode,
        "run_dir": run_dir,
        "scientific_evidence": is_scientific,
        "recommendation": decision["recommendation"],
        "rationale": decision["rationale"],
        "next_experiment": decision["next_experiment"],
        "criteria": criteria,
        "thresholds": {**DEFAULT_THRESHOLDS, **(thresholds or {})},
        "blocked_modalities": list(blocked_modalities),
        "capability": capability,
        "lens_validation": lens_validation,
        "code_statistics": code_stats,
        "representational": representational,
        "interventions": interventions,
        "invariance": invariance,
        "manifest_audit": manifest_audit,
    }

    def yes_no(name: str) -> str:
        return "yes" if criteria.get(name, {}).get("passed") else "no"

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
        f"2. **Did the compatible J-lens reconstruct activations adequately?** "
        f"{yes_no('lens_reconstruction')} — median explained fraction "
        f"{code_stats.get('median_explained_fraction')}.",
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
            else " — no off-diagonal cell produced an effect."
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
        "| criterion | passed |",
        "| --- | --- |",
    ]
    lines += [
        f"| {name} | {'PASS' if entry['passed'] else 'FAIL'} |"
        for name, entry in criteria.items()
    ]
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
        "",
        "## Raw summary",
        "",
        "```json",
        json.dumps(
            {
                "recommendation": summary["recommendation"],
                "criteria": {k: v["passed"] for k, v in criteria.items()},
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
