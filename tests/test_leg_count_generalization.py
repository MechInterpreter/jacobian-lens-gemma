from __future__ import annotations

import pytest

from jlens.mmpilot.leg_count_generalization import (
    CONDITIONS,
    CROSS_CONDITION,
    MODALITIES,
    NOVEL_CONDITIONS,
    NOVEL_CONTROL_CONDITIONS,
    NOVEL_SCORED_CONDITIONS,
    NOVEL_TARGETS,
    TARGET_ANSWERS,
    LegCountGeneralizationRefused,
    confirmation_report,
    derive_cross_target_rows,
    development_report,
    frozen_design,
    leg_count_answer_matches,
    novel_confirmation_report,
    novel_development_report,
    novel_frozen_design,
)


def _rows(*, targets=("cat", "ant"), n=12, successes=9):
    leverage = []
    trials = []
    for target in targets:
        for modality in MODALITIES:
            for index in range(n):
                leverage.append(
                    {
                        "target": target,
                        "modality": modality,
                        "group_id": f"g{index}",
                        "success": index < successes,
                        "integrity_pass": True,
                    }
                )
                for condition in CONDITIONS:
                    trials.append(
                        {
                            "target": target,
                            "modality": modality,
                            "group_id": f"g{index}",
                            "condition": condition,
                            "success": condition == "exact" and index < successes,
                            "integrity_pass": True,
                        }
                    )
    return leverage, trials


def test_frozen_design_uses_distinct_answers_and_no_refit():
    design = frozen_design()
    assert design["target_answers"] == {"cat": "4", "ant": "6", "spider": "8"}
    assert design["layers"] == list(range(16, 41))
    assert design["alpha"] == 1.0
    assert design["lens_refitted"] is False
    assert design["development"] == {
        "candidates": 9,
        "recruited": 6,
        "minimum_exact_rate": 0.50,
        "minimum_control_margin": 0.25,
        "minimum_answer_leverage_rate": 0.75,
    }
    assert design["confirmation"]["candidates"] == 22
    assert design["confirmation"]["recruited"] == 12


def test_number_answers_accept_digit_or_word_without_candidate_scoring():
    assert leg_count_answer_matches("6", "6")
    assert leg_count_answer_matches(" six ", "6")
    assert not leg_count_answer_matches("eight", "6")


def test_development_selects_only_passing_novel_targets():
    leverage, trials = _rows(n=6, successes=5)
    report = development_report(leverage, trials, expected_n=6)
    assert report["verdict"] == "LEG_COUNT_GENERALIZATION_DEVELOPMENT_GO"
    assert report["selected_novel_targets"] == ["ant"]
    assert report["fresh_confirmation_opened"] is False


def test_answer_leverage_null_does_not_veto_identity_recomputation():
    leverage, trials = _rows(n=6, successes=5)
    for row in leverage:
        row["success"] = False
    report = development_report(leverage, trials, expected_n=6)
    assert report["verdict"] == "LEG_COUNT_GENERALIZATION_DEVELOPMENT_GO"
    assert report["calibration_passed"] is True
    assert report["selected_novel_targets"] == ["ant"]
    assert all(
        row["answer_leverage_is_diagnostic_only"]
        for row in report["target_results"]
    )


def test_confirmation_requires_calibration_and_holm_passing_novel_target():
    leverage, trials = _rows(n=12, successes=9)
    report = confirmation_report(
        leverage, trials, frozen_targets=("cat", "ant")
    )
    assert report["verdict"] == "FRESH_MULTIMODAL_LEG_COUNT_GENERALIZATION_GO"
    assert report["calibration_passed"] is True
    assert report["novel_passing_targets"] == ["ant"]
    assert all(row["holm_adjusted_p"] <= 0.05 for row in report["paired_comparisons"])


def test_confirmation_no_go_when_novel_target_does_not_replicate():
    leverage, trials = _rows(n=12, successes=9)
    for row in trials:
        if row["target"] == "ant":
            row["success"] = False
    report = confirmation_report(
        leverage, trials, frozen_targets=("cat", "ant")
    )
    assert report["verdict"] == "FRESH_MULTIMODAL_LEG_COUNT_GENERALIZATION_NO_GO"


def _novel_rows(
    *,
    targets=("ant", "spider"),
    n=12,
    exact_successes=10,
    control_surface="2",
):
    """Executed rows only.  The cross-target control is derived from these."""

    rows = []
    for target in targets:
        for modality in MODALITIES:
            for index in range(n):
                for condition in NOVEL_CONDITIONS:
                    hit = condition == "exact" and index < exact_successes
                    rows.append(
                        {
                            "target": target,
                            "modality": modality,
                            "group_id": f"g{index}",
                            "condition": condition,
                            "patched_surface": (
                                TARGET_ANSWERS[target] if hit else control_surface
                            ),
                            "success": hit,
                            "integrity_pass": True,
                        }
                    )
    return rows


def test_novel_development_scores_six_controls_including_cross_target():
    rows = _novel_rows(n=6, exact_successes=4)
    report = novel_development_report(rows, expected_n=6)
    assert report["verdict"] == "LEG_COUNT_NOVEL_TARGET_DEVELOPMENT_GO"
    assert report["passing_novel_targets"] == ["ant", "spider"]
    assert report["selected_confirmation_targets"] == ["ant", "spider"]
    assert report["selected_confirmation_target"] == "ant"
    assert set(report["effect_cells"][0]["conditions"]) == set(
        NOVEL_SCORED_CONDITIONS
    )
    assert CROSS_CONDITION in NOVEL_CONTROL_CONDITIONS


def test_cross_target_control_is_derived_from_the_other_identity():
    rows = _novel_rows(n=6, exact_successes=6)
    derived = derive_cross_target_rows(rows, target="ant", donor_target="spider")
    assert len(derived) == 6 * len(MODALITIES)
    # Spider's exact exchange answered 8, so it never counts as a six.
    assert not any(row["success"] for row in derived)
    assert {row["condition"] for row in derived} == {CROSS_CONDITION}


def test_a_generic_perturbation_that_hits_every_answer_cannot_pass():
    """The failure the four-legged pilot could not rule out.

    Here the exchange is 'successful' for both identities on the same photos
    because it drives whatever answer is being scored.  The cross-target
    control catches it: an identity swap toward spider that also reads as six
    means the answer is not tracking the identity.
    """

    rows = _novel_rows(n=6, exact_successes=6)
    for row in rows:
        if row["condition"] == "exact":
            # Whichever identity was inserted, the model lands on six.  Against
            # zero, unrelated and every random seed this looks like a clean 6/6
            # win for ant.  Only the cross-target control sees that the spider
            # exchange produced the same six.
            row["patched_surface"] = "6"
            row["success"] = row["target"] == "ant"
    report = novel_development_report(rows, expected_n=6)
    assert report["verdict"] == "LEG_COUNT_NOVEL_TARGET_DEVELOPMENT_NO_GO"
    assert report["passing_novel_targets"] == []
    ant_text = next(
        cell
        for cell in report["effect_cells"]
        if cell["target"] == "ant" and cell["modality"] == "text"
    )
    assert ant_text["conditions"]["exact"]["rate"] == 1.0
    assert ant_text["conditions"]["random_0"]["rate"] == 0.0
    assert ant_text["conditions"][CROSS_CONDITION]["rate"] == 1.0


def test_confusion_table_reports_the_answer_each_identity_produced():
    report = novel_development_report(
        _novel_rows(n=6, exact_successes=6), expected_n=6
    )
    by_target = {
        (row["target"], row["modality"]): row["answers"]
        for row in report["target_answer_confusion"]
    }
    assert by_target[("ant", "text")] == {"6": 6}
    assert by_target[("spider", "text")] == {"8": 6}
    assert all(
        row["all_targets_correct"] == 6 for row in report["double_dissociation"]
    )


def test_answer_leverage_is_reported_but_never_gates():
    rows = _novel_rows(n=6, exact_successes=4)
    leverage = [
        {
            "target": target,
            "modality": modality,
            "group_id": f"g{index}",
            "success": False,
            "integrity_pass": True,
        }
        for target in ("ant", "spider")
        for modality in MODALITIES
        for index in range(6)
    ]
    report = novel_development_report(rows, leverage, expected_n=6)
    assert report["verdict"] == "LEG_COUNT_NOVEL_TARGET_DEVELOPMENT_GO"
    assert all(
        row["rate"] == 0.0 and row["gating"] is False
        for row in report["answer_leverage_diagnostic"]
    )


def test_novel_confirmation_holm_family_is_eighteen_and_secondary_never_gates():
    rows = _novel_rows(n=12, exact_successes=10)
    report = novel_confirmation_report(
        rows, target="ant", secondary_targets=("spider",), expected_n=12
    )
    assert report["verdict"] == (
        "FRESH_MULTIMODAL_NOVEL_LEG_COUNT_GENERALIZATION_GO"
    )
    assert len(report["paired_comparisons"]) == 18
    assert all(
        row["holm_adjusted_p"] <= 0.05 for row in report["paired_comparisons"]
    )
    assert [family["target"] for family in report["holm_families"]] == [
        "ant",
        "spider",
    ]
    assert [family["gating"] for family in report["holm_families"]] == [True, False]


def test_a_failing_secondary_does_not_sink_the_primary():
    rows = _novel_rows(n=12, exact_successes=10)
    for row in rows:
        if row["target"] == "spider" and row["condition"] == "exact":
            row["success"] = False
            row["patched_surface"] = "2"
    report = novel_confirmation_report(
        rows, target="ant", secondary_targets=("spider",), expected_n=12
    )
    assert report["verdict"] == (
        "FRESH_MULTIMODAL_NOVEL_LEG_COUNT_GENERALIZATION_GO"
    )
    results = {row["target"]: row for row in report["target_results"]}
    assert results["ant"]["passed"] is True
    assert results["spider"]["passed"] is False


def test_confirmation_no_go_when_the_primary_misses_one_modality():
    rows = _novel_rows(n=12, exact_successes=10)
    for row in rows:
        if (
            row["target"] == "ant"
            and row["modality"] == "spoken_audio"
            and row["condition"] == "exact"
        ):
            row["success"] = False
            row["patched_surface"] = "2"
    report = novel_confirmation_report(
        rows, target="ant", secondary_targets=("spider",), expected_n=12
    )
    assert report["verdict"] == (
        "FRESH_MULTIMODAL_NOVEL_LEG_COUNT_GENERALIZATION_NO_GO"
    )


def test_the_novel_design_is_frozen_apart_from_the_cat_calibration():
    novel = novel_frozen_design()
    assert novel["design_digest"] != frozen_design()["design_digest"]
    assert novel["novel_targets"] == ["ant", "spider"]
    assert novel["target_answers"] == {"ant": "6", "spider": "8"}
    assert novel["answer_leverage_role"] == "diagnostic_only_never_gating"
    assert novel_frozen_design()["design_digest"] == novel["design_digest"]


def test_scoring_refuses_a_verdict_its_own_surface_does_not_support():
    rows = _novel_rows(n=6, exact_successes=6)
    for row in rows:
        if row["target"] == "ant" and row["condition"] == "random_0":
            row["success"] = True  # surface still reads "2"
    with pytest.raises(LegCountGeneralizationRefused):
        novel_development_report(rows, expected_n=6)


def test_development_hands_the_confirmation_stage_the_keys_it_reads():
    """The stage boundary, end to end.

    Stage 7C consumes the development report and freezes a design; Stage 7D
    consumes that design.  A rename on either side is silent until an A100 is
    already running, so the contract is pinned here instead.
    """

    development = novel_development_report(
        _novel_rows(n=6, exact_successes=4), expected_n=6
    )
    frozen = list(development["selected_confirmation_targets"])
    assert frozen, "development must hand at least one target forward"

    # Stage 7C: the priority target gates, the rest support, and every
    # predeclared identity is executed so the cross-target control exists.
    primary, *secondary = frozen
    donors = [name for name in NOVEL_TARGETS if name not in frozen]
    executed = [*frozen, *donors]
    assert sorted(executed) == sorted(NOVEL_TARGETS)

    # Stage 7D: score the untouched split under exactly that design.
    report = novel_confirmation_report(
        _novel_rows(targets=tuple(executed), n=12, exact_successes=10),
        target=primary,
        secondary_targets=secondary,
        expected_n=12,
    )
    assert report["verdict"] == (
        "FRESH_MULTIMODAL_NOVEL_LEG_COUNT_GENERALIZATION_GO"
    )
    assert report["target_results"][0]["role"] == "primary"
    assert len(report["double_dissociation"]) == len(MODALITIES)


def test_dissociation_still_reports_an_identity_that_missed_selection():
    rows = _novel_rows(n=12, exact_successes=10)
    report = novel_confirmation_report(rows, target="ant", expected_n=12)
    assert report["secondary_targets"] == []
    # Spider was executed only as ant's cross-target control, and its answers
    # still belong in the descriptive table.
    assert {row["target"] for row in report["target_answer_confusion"]} == {
        "ant",
        "spider",
    }
    assert len(report["double_dissociation"]) == len(MODALITIES)
