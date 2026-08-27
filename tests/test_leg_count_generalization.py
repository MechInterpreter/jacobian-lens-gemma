from __future__ import annotations

from jlens.mmpilot.leg_count_generalization import (
    CONDITIONS,
    MODALITIES,
    confirmation_report,
    development_report,
    frozen_design,
    leg_count_answer_matches,
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
