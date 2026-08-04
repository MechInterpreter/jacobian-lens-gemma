# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Stages C and D: the budget, the paired depth contrast, and the rubric.

The distinction these tests are mostly about: **an ineligible layer is not a
negative result.** A layer whose lens could not be validated was never causally
tested, so reporting it as "no transfer here" would convert an untested layer
into evidence. That case must reach ``INCONCLUSIVE_LAYER_LOCALIZATION``, never
``LATE_ONLY_SUPPORTED``.
"""

import pytest

from jlens.mmlocalize.layers import LOCALIZATION_LAYERS, REFERENCE_LAYER
from jlens.mmlocalize.targets import (
    LOCALIZATION_CONCEPTS,
    POLICY_FRESH_DISJOINT,
    POLICY_REUSED_PAIRED,
    REUSED_POLICY_LIMITATION,
)
from jlens.mmlocalize.verdict import (
    DEPTH_SCOPE_LIMITATION,
    EARLY_TRANSFER_CONFIRMED,
    INCONCLUSIVE_LAYER_LOCALIZATION,
    LATE_ONLY_SUPPORTED,
    OFF_DIAGONAL_PAIRS,
    LocalizationThresholds,
    estimate_localization_passes,
    evaluate_layer_cells,
    format_budget,
    localization_verdict,
    paired_layer_comparison,
    render_report,
    representational_rows,
)

# ------------------------------------------------------------ the budget


def _budget(**overrides):
    kwargs = {
        "n_concepts": 6,
        "modalities": ("text", "image"),
        "n_total_groups": 48,
        "n_capability_groups": 32,
        "n_layers_captured": 4,
        "n_eligible_causal_layers": 4,
        "n_targets_per_cell": 8,
        "alphas": (0.0, 0.25, 0.5),
        "n_validation_prompts": 32,
    }
    kwargs.update(overrides)
    return estimate_localization_passes(**kwargs)


def test_capturing_four_layers_costs_what_capturing_one_costs():
    """One forward pass records every requested layer — the whole reason a
    four-layer diagnostic is affordable."""
    one = _budget(n_layers_captured=1)
    four = _budget(n_layers_captured=4)
    assert one.activation_passes == four.activation_passes
    # Storage does scale, because there are four residuals to write.
    assert four.estimated_units["activation"] == 4 * one.estimated_units["activation"]


def test_causal_cost_scales_with_eligible_layers_only():
    one = _budget(n_eligible_causal_layers=1)
    four = _budget(n_eligible_causal_layers=4)
    assert four.causal_intervention_passes == 4 * one.causal_intervention_passes
    assert four.causal_clean_passes == 4 * one.causal_clean_passes
    assert one.text_validation_passes == four.text_validation_passes


def test_no_eligible_layer_means_no_causal_cost():
    budget = _budget(n_eligible_causal_layers=0)
    assert budget.causal_intervention_passes == 0
    assert budget.causal_clean_passes == 0
    assert budget.total_passes > 0          # validation and capture still happen


def test_the_total_is_the_sum_of_its_parts():
    budget = _budget()
    assert budget.total_passes == (
        budget.validation_target_discovery_passes
        + budget.text_validation_passes
        + budget.capability_passes
        + budget.activation_passes
        + budget.causal_clean_passes
        + budget.causal_intervention_passes
    )


def test_target_discovery_is_budgeted_separately():
    without_discovery = _budget()
    with_discovery = _budget(n_validation_discovery_prompts=320)
    assert with_discovery.validation_target_discovery_passes == 320
    assert with_discovery.total_passes == without_discovery.total_passes + 320
    assert with_discovery.estimated_units["validation_target_discovery"] == 320


def test_only_off_diagonal_cells_are_counted():
    budget = _budget()
    # 2 concepts x 2 source modalities x 1 target modality.
    assert budget.n_causal_cells == len(LOCALIZATION_CONCEPTS) * 2 * 1


def test_conditions_per_target_cover_the_zero_and_all_four_controls():
    # 1 zero-alpha condition + 2 positive alphas x 4 control kinds.
    assert _budget().n_conditions_per_target == 9


def test_recalibration_is_zero_unless_explicitly_enabled():
    assert _budget().recalibration_passes == 0
    assert _budget(recalibration_enabled=True).recalibration_passes == 256


def test_the_printed_budget_names_every_cost_and_the_eligibility_caveat():
    text = format_budget(_budget())
    for needle in (
        "TOTAL model forward passes",
        "target-discovery passes",
        "text lens-validation passes",
        "ONE forward pass records all of them",
        "decided by Stage B, not assumed",
        "estimated Drive footprint",
    ):
        assert needle in text
    assert "the frozen v2 artifact is evaluated as-is" in format_budget(_budget())


# ----------------------------------------------------- synthetic evidence


def _cell_row(
    *,
    concept="cat",
    pair="text->image",
    layer=38,
    control_kind="source_concept",
    alpha=0.5,
    effect=0.9,
    per_image=None,
):
    images = per_image or {f"img{i}": effect for i in range(8)}
    return {
        "concept": concept,
        "pair": pair,
        "source_modality": pair.split("->")[0],
        "target_modality": pair.split("->")[1],
        "layer": layer,
        "control_kind": control_kind,
        "alpha": alpha,
        "mean_signed_target_effect": effect,
        "mean_signed_margin_effect": effect,
        "fraction_expected_sign": 1.0,
        "mean_activation_norm_ratio": 1.0,
        "mean_abs_unrelated_change": abs(effect) * 0.1,
        "n_prediction_changes": 3,
        "n_distinct_images": len(images),
        "n_positive_images": 4,
        "n_negative_images": 4,
        "per_image": {
            image: {"mean_signed_target_effect": value} for image, value in images.items()
        },
    }


def _interventions(layers=LOCALIZATION_LAYERS, *, effect=0.9, concepts=LOCALIZATION_CONCEPTS):
    rows = []
    for layer in layers:
        for concept in concepts:
            for pair in OFF_DIAGONAL_PAIRS:
                rows.append(
                    _cell_row(concept=concept, pair=pair, layer=layer, effect=effect)
                )
                for kind, control_effect in (
                    ("random_norm_matched", 0.0),
                    ("unrelated_concept", 0.05),
                    ("raw_residual_difference", 0.2),
                ):
                    rows.append(
                        _cell_row(
                            concept=concept,
                            pair=pair,
                            layer=layer,
                            control_kind=kind,
                            effect=control_effect,
                        )
                    )
    return {"rows": rows}


def _representational(top1=0.9, shuffled=0.3, jgap=0.5, rawgap=0.2):
    return {
        "pairs": {
            pair: {
                "jspace_retrieval": {"top1_accuracy": top1, "mrr": top1, "n_queries": 16},
                "shuffled_control": {"p95_top1_accuracy": shuffled},
                "jspace_separation": {"gap": jgap},
                "raw_residual_separation": {"gap": rawgap},
                "jspace_support_overlap": {"gap": 0.3},
                "raw_residual_retrieval": {"top1_accuracy": 0.4},
                "exclusions": {"n_excluded_same_image_different_group": 0},
            }
            for pair in OFF_DIAGONAL_PAIRS
        }
    }


def _validity(eligible_layers):
    return {
        layer: {
            "eligible": layer in eligible_layers,
            "failed_checks": [] if layer in eligible_layers else ["median_rank_and_top_k"],
            "metrics": {"j_lens": {}},
        }
        for layer in LOCALIZATION_LAYERS
    }


def _manifest(policy=POLICY_FRESH_DISJOINT):
    return {
        "same_targets_at_every_layer": True,
        "target_checksum": "sha256:frozen",
        "policy": policy,
    }


def _verdict(eligible, *, interventions=None, representational=None, manifest=None):
    return localization_verdict(
        validity=_validity(eligible),
        representational={layer: representational or _representational() for layer in LOCALIZATION_LAYERS},
        interventions=interventions or _interventions(),
        target_manifest=manifest or _manifest(),
    )


# ------------------------------------------------------------ causal cells


def test_a_clean_cell_passes_against_its_own_controls():
    cells = evaluate_layer_cells(_interventions(), layer=38)
    assert len(cells) == len(LOCALIZATION_CONCEPTS) * len(OFF_DIAGONAL_PAIRS)
    assert all(cell["passes"] for cell in cells)
    assert all(cell["layer_normalized"] == 90 for cell in cells)


def test_a_cell_whose_effect_does_not_clear_its_controls_fails():
    interventions = _interventions(effect=0.05)
    cells = evaluate_layer_cells(interventions, layer=38)
    assert not any(cell["passes"] for cell in cells)
    assert any("does not clear" in " ".join(cell["reasons"]) for cell in cells)


def test_a_missing_unrelated_control_fails_the_cell_explicitly():
    """The defect the MOCK run surfaced: a control that was never estimated is
    not a control, and must not read as a passing cell."""
    interventions = _interventions()
    interventions["rows"] = [
        row for row in interventions["rows"] if row["control_kind"] != "unrelated_concept"
    ]
    cells = evaluate_layer_cells(interventions, layer=38)
    assert not any(cell["passes"] for cell in cells)
    assert any("control is missing" in " ".join(cell["reasons"]) for cell in cells)


def test_too_few_photographs_make_a_cell_undecidable():
    rows = _interventions()["rows"]
    for row in rows:
        row["n_positive_images"] = 1
        row["n_negative_images"] = 1
        row["n_distinct_images"] = 2
    cells = evaluate_layer_cells({"rows": rows}, layer=38)
    assert not any(cell["passes"] for cell in cells)
    assert any("distinct positive image" in " ".join(c["reasons"]) for c in cells)


def test_a_global_looking_edit_fails_specificity():
    rows = _interventions()["rows"]
    for row in rows:
        row["mean_abs_unrelated_change"] = 10.0
    cells = evaluate_layer_cells({"rows": rows}, layer=38)
    assert any("looks global" in " ".join(cell["reasons"]) for cell in cells)


def test_an_insane_activation_norm_fails_the_cell():
    rows = _interventions()["rows"]
    for row in rows:
        row["mean_activation_norm_ratio"] = 9.0
    cells = evaluate_layer_cells({"rows": rows}, layer=38)
    assert any("norm ratio" in " ".join(cell["reasons"]) for cell in cells)


def test_a_layer_with_no_rows_is_reported_as_not_evaluated():
    cells = evaluate_layer_cells(_interventions(layers=(38,)), layer=20)
    assert all(cell["evaluated"] is False for cell in cells)
    assert all(cell["passes"] is False for cell in cells)


# --------------------------------------------------- the paired comparison


def test_the_paired_contrast_matches_photographs_across_layers():
    rows = paired_layer_comparison(_interventions())
    paired = [row for row in rows if row["paired"]]
    assert paired, "the frozen target set must make every layer pairable"
    for row in paired:
        assert row["n_paired_images"] == 8
        assert row["reference_layer"] == REFERENCE_LAYER
        assert row["layer"] != REFERENCE_LAYER
        assert row["mean_paired_delta"] == pytest.approx(0.0)


def test_only_photographs_present_at_both_layers_are_paired():
    interventions = _interventions(layers=(32, 38))
    for row in interventions["rows"]:
        if int(row["layer"]) == 32:
            row["per_image"] = {
                image: block
                for image, block in list(row["per_image"].items())[:5]
            }
    rows = [r for r in paired_layer_comparison(interventions, layers=(32, 38)) if r["paired"]]
    assert rows and all(row["n_paired_images"] == 5 for row in rows)


def test_an_untested_layer_is_reported_as_unpairable_not_as_zero():
    """A layer that was skipped has no row; the contrast must say so rather than
    silently contributing a delta of zero."""
    rows = paired_layer_comparison(_interventions(layers=(38,)))
    assert rows and all(row["paired"] is False for row in rows)
    assert all("nothing to pair" in row["reason"] for row in rows)


def test_a_stronger_earlier_layer_shows_a_positive_paired_delta():
    interventions = _interventions(layers=(32, 38))
    for row in interventions["rows"]:
        if int(row["layer"]) == 32 and row["control_kind"] == "source_concept":
            row["per_image"] = {
                image: {"mean_signed_target_effect": 1.4}
                for image in row["per_image"]
            }
    rows = [r for r in paired_layer_comparison(interventions, layers=(32, 38)) if r["paired"]]
    assert all(row["mean_paired_delta"] == pytest.approx(0.5) for row in rows)
    assert all(row["fraction_images_layer_exceeds_reference"] == 1.0 for row in rows)


# ---------------------------------------------------------- the verdict


def test_an_earlier_eligible_layer_that_transfers_confirms_early_transfer():
    verdict = _verdict(eligible=[32, 38])
    assert verdict["verdict"] == EARLY_TRANSFER_CONFIRMED
    assert verdict["earliest_tested_layer_with_evidence"] == 32
    assert verdict["reference_layer_reproduces"] is True
    assert "earliest TESTED layer" in verdict["rationale"]


def test_the_earliest_transferring_layer_is_reported_not_the_strongest():
    verdict = _verdict(eligible=[20, 26, 32, 38])
    assert verdict["earliest_tested_layer_with_evidence"] == 20


def test_eligible_earlier_layers_without_transfer_support_late_only():
    interventions = _interventions()
    weakened = []
    for row in interventions["rows"]:
        if int(row["layer"]) != REFERENCE_LAYER and row["control_kind"] == "source_concept":
            row = {**row, "mean_signed_target_effect": 0.0,
                   "per_image": {k: {"mean_signed_target_effect": 0.0}
                                 for k in row["per_image"]}}
        weakened.append(row)
    verdict = _verdict(eligible=[26, 32, 38], interventions={"rows": weakened})
    assert verdict["verdict"] == LATE_ONLY_SUPPORTED
    assert verdict["earliest_tested_layer_with_evidence"] is None
    assert "did not produce controlled off-diagonal transfer" in verdict["rationale"]


def test_no_eligible_earlier_layer_is_inconclusive_not_late_only():
    """The distinction the whole rubric turns on."""
    verdict = _verdict(eligible=[38])
    assert verdict["verdict"] == INCONCLUSIVE_LAYER_LOCALIZATION
    assert verdict["verdict"] != LATE_ONLY_SUPPORTED
    assert "NOT evidence that transfer is late-only" in verdict["rationale"]
    assert "about the frozen text-calibrated lens" in verdict["rationale"]


def test_a_reference_layer_that_does_not_reproduce_is_inconclusive():
    verdict = _verdict(
        eligible=[20, 26, 32, 38], representational=_representational(top1=0.1, shuffled=0.9)
    )
    assert verdict["verdict"] == INCONCLUSIVE_LAYER_LOCALIZATION
    assert verdict["reference_layer_reproduces"] is False
    assert "nothing about depth can be concluded" in verdict["rationale"].lower()


def test_an_ineligible_layer_is_skipped_causally_and_says_why():
    verdict = _verdict(eligible=[38])
    for layer in (20, 26, 32):
        entry = verdict["per_layer"][layer]
        assert entry["causally_tested"] is False
        assert entry["causal_cells"] == []
        assert "never tested" in entry["skipped_because"]
        # Its diagnostics survive.
        assert entry["representational"]


def test_an_ineligible_layer_keeps_its_representational_diagnostics():
    verdict = _verdict(eligible=[38])
    assert verdict["per_layer"][20]["representation_beats_shuffled_both_directions"] is True


def test_every_verdict_carries_the_depth_and_conditioning_limits():
    for eligible in ([38], [32, 38], [20, 26, 32, 38]):
        verdict = _verdict(eligible=eligible)
        assert verdict["depth_scope_limitation"] == DEPTH_SCOPE_LIMITATION
        assert "earliest layer in the model" in verdict["depth_scope_limitation"]
        assert "concept-general prevalence" in verdict["concept_conditioning_limitation"]
        assert "not erasure" in verdict["intervention_limitation"]


def test_the_reused_target_policy_limitation_is_carried_into_the_verdict():
    verdict = _verdict(eligible=[32, 38], manifest=_manifest(POLICY_REUSED_PAIRED))
    assert verdict["target_policy_limitation"] == REUSED_POLICY_LIMITATION
    assert _verdict(eligible=[32, 38])["target_policy_limitation"] is None


def test_drifting_targets_fail_the_pairing_criterion():
    verdict = _verdict(
        eligible=[32, 38],
        manifest={"same_targets_at_every_layer": False, "target_checksum": "x",
                  "policy": POLICY_FRESH_DISJOINT},
    )
    assert verdict["criteria_status"]["targets_identical_at_every_layer"] == "FAIL"


def test_the_thresholds_are_the_commissioned_ones():
    thresholds = LocalizationThresholds()
    assert thresholds.concepts == ("cat", "toilet")
    assert thresholds.required_positive_images_per_cell == 4
    assert thresholds.required_negative_images_per_cell == 4
    assert thresholds.min_fraction_expected_sign == 0.75


# ------------------------------------------------------------- the report


def _validity_with_metrics(eligible):
    return {
        layer: {
            "eligible": layer in eligible,
            "failed_checks": [] if layer in eligible else ["median_rank_and_top_k"],
            "legacy_gate": {"passed": layer in eligible},
            "metrics": {
                "j_lens": {
                    "mean_reciprocal_rank": 0.9,
                    "median_midrank": 1.0,
                    "median_optimistic_rank": 1.0,
                    "unique_top1_agreement": 0.8,
                    "tied_at_max_rate": 0.1,
                    "top10_inclusion": 0.9,
                }
            },
        }
        for layer in LOCALIZATION_LAYERS
    }


def test_the_report_shows_both_numbering_systems_and_both_gates():
    verdict = _verdict(eligible=[32, 38])
    report = render_report(
        run_dir="/runs/x",
        verdict=verdict,
        validity=_validity_with_metrics([32, 38]),
        budget=_budget().to_dict(),
        resume={"status": "starting", "completed_units": {}},
        mode="localization",
    )
    assert "~normalized [48, 62, 76, 90]" in report
    assert "median midrank" in report and "median optimistic" in report
    assert "old gate" in report
    assert "tied-at-max" in report
    assert EARLY_TRANSFER_CONFIRMED in report
    assert "Paired depth contrast" in report
    assert "earliest layer in the model" in report


def test_the_report_marks_a_mock_run_as_pipeline_evidence_only():
    report = render_report(
        run_dir="/runs/x",
        verdict=_verdict(eligible=[38]),
        validity=_validity_with_metrics([38]),
        budget=None,
        resume=None,
        mode="mock",
    )
    assert "not scientific evidence" in report


def test_the_report_names_a_skipped_layer_as_untested():
    report = render_report(
        run_dir="/runs/x",
        verdict=_verdict(eligible=[38]),
        validity=_validity_with_metrics([38]),
        budget=None,
        resume=None,
        mode="localization",
    )
    assert "causal stage **skipped**" in report
    assert "not a negative causal result" in report


def test_representational_rows_carry_the_layer_and_its_normalized_depth():
    rows = representational_rows(_representational(), layer=32)
    assert [row["pair"] for row in rows] == list(OFF_DIAGONAL_PAIRS)
    assert all(row["layer"] == 32 and row["layer_normalized"] == 76 for row in rows)
    assert all(row["beats_shuffled"] for row in rows)
