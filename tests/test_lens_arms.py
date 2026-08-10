from jlens.calibration.lens_arms import (
    LENS_ARM_PROTOCOL,
    R_LENS_ARM,
    RAW_J_ARM,
    dual_arm_fitting_budget,
    format_dual_arm_fitting_budget,
    select_both_lens_arms,
)


def _result(passing):
    return {
        layer: {
            "passed": layer in passing,
            "metrics": {},
            "failed_checks": [] if layer in passing else ["gate"],
        }
        for layer in (27, 28, 29, 30, 31)
    }


def test_raw_is_primary_and_arms_are_never_pooled():
    assert RAW_J_ARM.primary is True
    assert R_LENS_ARM.primary is False
    assert LENS_ARM_PROTOCOL["raw_j_is_primary"] is True
    assert LENS_ARM_PROTOCOL["lambda_grid"] == []


def test_each_arm_selects_its_own_earliest_passer():
    result = select_both_lens_arms(
        raw_confirmation=_result({30, 31}),
        r_confirmation=_result({28, 29}),
    )
    assert result["raw_j_selected_layer"] == 30
    assert result["r_lens_selected_layer"] == 28
    assert result["headline"] == "RAW_J_LENS_EARLY_LAYER_GO"
    assert result["arms_pooled"] is False


def test_r_only_success_is_not_called_raw_j_success():
    result = select_both_lens_arms(
        raw_confirmation=_result(set()),
        r_confirmation=_result({29}),
    )
    assert result["headline"] == "R_LENS_EARLY_LAYER_GO"
    assert "R-space" in result["claim_boundary"]


def test_dual_arm_budget_counts_two_independent_accumulations():
    budget = dual_arm_fitting_budget(
        scale=250, layers=(27, 28, 29, 30, 31), target_layer=41
    )
    assert budget["n_independent_accumulations"] == 2
    assert budget["accumulators_shared"] is False
    assert budget["total_forward_passes"] == 500
    assert budget["total_backward_block_steps"] == 7000
    assert budget["total_hours_low"] >= 4.0
    assert budget["total_hours_high"] >= 8.0
    rendered = format_dual_arm_fitting_budget(budget)
    assert "no shared accumulator" in rendered
    assert "4.0-8.0 h total" in rendered
