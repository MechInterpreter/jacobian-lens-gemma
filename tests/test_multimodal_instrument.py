"""Unit tests for the swap-instrument integrity gate.

These cover the clause-by-clause behaviour of
:mod:`jlens.mmpilot.multimodal_instrument`, the realization correction it
depends on, and the read-only amendments that reclassify the completed flawed
run. The notebook-level counterparts live in
``tests/test_multimodal_followup_realpath.py``, which executes the corrected
Stage 5B1RC cell end to end against a synthetic backend.

**Nothing here is a scientific result.** Every tensor is synthetic.
"""

from __future__ import annotations

import json

import pytest
import torch

from jlens.mmpilot.coordinate_swap import (
    ModelDtypeRealizationPolicy,
    swap_coordinates,
)
from jlens.mmpilot.multimodal_followup import (
    MultimodalFollowupRefused,
    corrected_exploratory_verdict,
    direct_answer_matching_defect_amendment,
    direct_answer_trial_row,
    generation_trial_row,
    instrument_defect_amendment,
    legacy_confirmation_realization_audit,
    legacy_confirmation_replication_verdict,
)
from jlens.mmpilot.multimodal_instrument import (
    INSTRUMENT_STATES,
    INTEGRITY_CLAUSES,
    MODEL_DTYPE_REALIZATION,
    POST_CAST_TOLERANCE,
    InstrumentIntegrityRefused,
    cell_integrity,
    instrument_state,
    realization_policy_digest,
    trial_integrity,
)
from jlens.mmpilot.store import payload_checksum
from jlens.mmpilot.workspace_replication import summarize_swap_diagnostics

_BAND = list(range(16, 41))


def _clean_row(**overrides) -> dict:
    """One faithfully realized alpha=1 exact trial row."""
    row = {
        "condition": "exact",
        "alpha": 1.0,
        "layers_patched": list(_BAND),
        "all_prompt_positions_patched": True,
        "all_hooks_fired": True,
        "all_finite": True,
        "all_layers_are_exact_alpha_one_exchange_before_cast": True,
        "all_model_dtype_realizations_converged": True,
        "max_coordinate_update_error": 0.001,
        "max_orthogonal_residual_drift": 0.001,
        "max_activation_norm_ratio": 1.02,
        "max_update_to_activation_norm_ratio": 0.2,
        "teacher_forcing_used": False,
        "candidate_list_supplied": False,
    }
    row.update(overrides)
    return row


# ------------------------------------------------------------ clause by clause


def test_a_clean_row_satisfies_every_clause() -> None:
    result = trial_integrity(_clean_row(), layers=_BAND)
    assert result["passed"] is True
    assert result["failed_clauses"] == []
    assert result["missing_fields"] == []
    assert set(result["clauses"]) == set(INTEGRITY_CLAUSES)


def test_a_coordinate_error_of_0_020_passes_the_gate() -> None:
    """The boundary is inclusive, and it is the frozen 0.02."""
    result = trial_integrity(
        _clean_row(max_coordinate_update_error=0.020), layers=_BAND
    )
    assert POST_CAST_TOLERANCE == 0.02
    assert result["passed"] is True


def test_a_coordinate_error_of_0_021_fails_the_gate() -> None:
    result = trial_integrity(
        _clean_row(max_coordinate_update_error=0.021), layers=_BAND
    )
    assert result["passed"] is False
    assert result["failed_clauses"] == [
        "post_cast_coordinate_error_within_tolerance"
    ]


def test_the_residual_drift_gate_is_enforced_at_the_same_tolerance() -> None:
    assert trial_integrity(
        _clean_row(max_orthogonal_residual_drift=0.020), layers=_BAND
    )["passed"]
    failed = trial_integrity(
        _clean_row(max_orthogonal_residual_drift=0.021), layers=_BAND
    )
    assert failed["failed_clauses"] == [
        "post_cast_residual_drift_within_tolerance"
    ]


@pytest.mark.parametrize(
    "field, clause",
    [
        ("all_finite", "all_finite"),
        ("all_hooks_fired", "all_hooks_fired"),
        (
            "all_layers_are_exact_alpha_one_exchange_before_cast",
            "alpha_one_exact_exchange_before_cast",
        ),
        (
            "all_model_dtype_realizations_converged",
            "model_dtype_realization_converged",
        ),
        (
            "max_coordinate_update_error",
            "post_cast_coordinate_error_within_tolerance",
        ),
        (
            "max_orthogonal_residual_drift",
            "post_cast_residual_drift_within_tolerance",
        ),
    ],
)
def test_a_missing_diagnostic_fails_its_clause(field: str, clause: str) -> None:
    """Absent evidence is never a pass.

    ``float(row.get(key) or 0.0)`` turned a missing post-cast error into a
    perfect one, which is how an unrealized intervention scored
    ``integrity_pass = true``.
    """
    row = _clean_row()
    row.pop(field)
    result = trial_integrity(row, layers=_BAND)
    assert result["passed"] is False
    assert clause in result["failed_clauses"]
    assert field in result["missing_fields"]


def test_the_exact_before_cast_clause_is_required_at_alpha_one() -> None:
    failed = trial_integrity(
        _clean_row(all_layers_are_exact_alpha_one_exchange_before_cast=False),
        layers=_BAND,
    )
    assert failed["failed_clauses"] == ["alpha_one_exact_exchange_before_cast"]


def test_the_exact_before_cast_clause_is_not_applicable_at_alpha_zero() -> None:
    """The parity arm multiplies the update by exactly zero.

    Its ``alpha_one_is_exact_exchange`` flag is False by construction, so
    requiring the clause would fail every valid zero control.
    """
    row = _clean_row(
        condition="zero",
        alpha=0.0,
        all_layers_are_exact_alpha_one_exchange_before_cast=False,
    )
    result = trial_integrity(row, layers=_BAND)
    assert result["passed"] is True
    assert "alpha_one_exact_exchange_before_cast" in result["not_applicable"]


def test_a_wrong_layer_list_fails() -> None:
    result = trial_integrity(
        _clean_row(layers_patched=list(range(16, 40))), layers=_BAND
    )
    assert result["failed_clauses"] == ["expected_layers_patched"]


def test_unpatched_prompt_positions_fail() -> None:
    result = trial_integrity(
        _clean_row(all_prompt_positions_patched=False), layers=_BAND
    )
    assert result["failed_clauses"] == ["all_prompt_positions_patched"]


def test_teacher_forcing_or_a_candidate_list_fails() -> None:
    assert "no_teacher_forcing" in trial_integrity(
        _clean_row(teacher_forcing_used=True), layers=_BAND
    )["failed_clauses"]
    assert "no_candidate_list" in trial_integrity(
        _clean_row(candidate_list_supplied=True), layers=_BAND
    )["failed_clauses"]


def test_activation_and_update_limits_are_part_of_the_same_gate() -> None:
    assert "activation_norm_ratio_within_limit" in trial_integrity(
        _clean_row(max_activation_norm_ratio=1.3), layers=_BAND
    )["failed_clauses"]
    assert "update_to_activation_ratio_within_limit" in trial_integrity(
        _clean_row(max_update_to_activation_norm_ratio=0.9), layers=_BAND
    )["failed_clauses"]


def test_an_empty_band_is_refused_rather_than_passed_vacuously() -> None:
    with pytest.raises(InstrumentIntegrityRefused):
        trial_integrity(_clean_row(), layers=[])


def test_an_empty_cell_fails() -> None:
    result = cell_integrity([], layers=_BAND)
    assert result["passed"] is False
    assert result["n_trials"] == 0


def test_the_direct_answer_arm_is_gated_on_its_own_realization_field() -> None:
    row = _clean_row(condition="direct_answer", alpha=1.0)
    row.pop("all_layers_are_exact_alpha_one_exchange_before_cast")
    row.pop("max_coordinate_update_error")
    row.pop("max_orthogonal_residual_drift")
    row["max_relative_norm_match_error"] = 0.002
    row["max_relative_cumulative_band_displacement_match_error"] = 0.002
    assert trial_integrity(row, layers=_BAND)["passed"] is True
    row["max_relative_norm_match_error"] = 0.021
    assert "post_cast_coordinate_error_within_tolerance" in trial_integrity(
        row, layers=_BAND
    )["failed_clauses"]


def test_per_layer_matched_but_cumulatively_unmatched_control_fails() -> None:
    """Regression for the exact hole in the completed Stage 6C run."""
    row = _clean_row(condition="direct_answer", alpha=1.0)
    row.pop("all_layers_are_exact_alpha_one_exchange_before_cast")
    row.pop("max_coordinate_update_error")
    row.pop("max_orthogonal_residual_drift")
    row["max_relative_norm_match_error"] = 1e-7
    row["max_relative_cumulative_band_displacement_match_error"] = 2.1
    result = trial_integrity(row, layers=_BAND)
    assert result["passed"] is False
    assert result["failed_clauses"] == [
        "cumulative_band_displacement_norm_matched"
    ]


# ----------------------------------------------------------- the state machine


def test_a_broken_instrument_can_never_be_called_a_scientific_null() -> None:
    for controls_moved in (False, True):
        for effect in (False, True):
            for direct in (None, False, True):
                state = instrument_state(
                    integrity_passed=False,
                    controls_moved=controls_moved,
                    effect_present=effect,
                    direct_answer_available=direct is not None,
                    direct_answer_passed=direct,
                )
                assert state == "INSTRUMENT_FAILURE"


def test_the_positive_control_alone_can_never_produce_a_go() -> None:
    state = instrument_state(
        integrity_passed=True,
        controls_moved=False,
        effect_present=False,
        direct_answer_available=True,
        direct_answer_passed=True,
    )
    assert state == "SCIENTIFIC_NULL"
    assert state != "EFFECT_GO"


def test_both_arms_failing_is_inconclusive_not_a_null() -> None:
    assert instrument_state(
        integrity_passed=True,
        controls_moved=False,
        effect_present=False,
        direct_answer_available=True,
        direct_answer_passed=False,
    ) == "INCONCLUSIVE"


def test_a_missing_positive_control_is_inconclusive() -> None:
    assert instrument_state(
        integrity_passed=True,
        controls_moved=False,
        effect_present=False,
        direct_answer_available=False,
        direct_answer_passed=None,
    ) == "INCONCLUSIVE"


def test_moving_controls_outrank_a_present_effect() -> None:
    assert instrument_state(
        integrity_passed=True,
        controls_moved=True,
        effect_present=True,
        direct_answer_available=True,
        direct_answer_passed=True,
    ) == "CONTROL_FAILURE"


def test_every_state_is_declared() -> None:
    produced = {
        instrument_state(
            integrity_passed=integrity,
            controls_moved=controls,
            effect_present=effect,
            direct_answer_available=direct is not None,
            direct_answer_passed=direct,
        )
        for integrity in (True, False)
        for controls in (True, False)
        for effect in (True, False)
        for direct in (None, True, False)
    }
    assert produced == set(INSTRUMENT_STATES)


# ------------------------------------------------- the realization correction


def _near_degenerate_basis(d_model: int, seed: int) -> torch.Tensor:
    """Two nearly parallel unit directions, as two related concepts give."""
    generator = torch.Generator().manual_seed(seed)
    a = torch.randn(d_model, generator=generator, dtype=torch.float64)
    a = a / a.norm()
    b = a + 0.05 * torch.randn(d_model, generator=generator, dtype=torch.float64)
    b = b / b.norm()
    return torch.stack([a, b], dim=1)


def _activation(V: torch.Tensor, *, scale: float, seed: int) -> torch.Tensor:
    """An activation with a large orthogonal part and order-one coordinates.

    That is the regime a real residual stream is in, and it is what makes a
    bf16 cast move the intended coordinates: the rounding quantum is set by the
    vector's overall magnitude, not by the two coordinates being exchanged.
    """
    generator = torch.Generator().manual_seed(seed)
    d_model = V.shape[0]
    base = torch.randn(4, d_model, generator=generator, dtype=torch.float64) * scale
    coordinates = torch.linalg.solve(V.T @ V, V.T @ base.T).T
    small = torch.randn(4, 2, generator=generator, dtype=torch.float64) * 0.5
    return (base - coordinates @ V.T + small @ V.T).to(torch.bfloat16)


def test_the_bounded_correction_converges_under_simulated_bf16_rounding() -> None:
    """The cast breaks the exchange; the policy repairs it and says so."""
    V = _near_degenerate_basis(256, seed=0)
    h = _activation(V, scale=6.0, seed=0)

    _patched, uncorrected = swap_coordinates(h, V, alpha=1.0)
    assert uncorrected["model_dtype_realization"] is None
    assert (
        uncorrected["max_post_cast_relative_coordinate_update_error"]
        > POST_CAST_TOLERANCE
    ), "fixture no longer reproduces an out-of-tolerance cast"

    _patched, corrected = swap_coordinates(
        h, V, alpha=1.0, realization_policy=MODEL_DTYPE_REALIZATION
    )
    assert corrected["model_dtype_realization"] == MODEL_DTYPE_REALIZATION.to_dict()
    assert corrected["model_dtype_corrections_applied"] >= 1
    assert corrected["model_dtype_realization_converged"] is True
    assert (
        corrected["max_post_cast_relative_coordinate_update_error"]
        <= POST_CAST_TOLERANCE
    )
    assert (
        corrected["max_post_cast_relative_coordinate_update_error"]
        < uncorrected["max_post_cast_relative_coordinate_update_error"]
    )


def test_a_cast_the_policy_cannot_repair_reports_non_convergence() -> None:
    """The honest outcome when the correction budget is not enough.

    This is the regime the completed flawed run was in. The policy reduces the
    error and still cannot reach 0.02, so it reports ``converged = False``
    instead of pretending. The verdict then names ``INSTRUMENT_FAILURE``.
    """
    V = _near_degenerate_basis(256, seed=0)
    h = _activation(V, scale=20.0, seed=0)
    _patched, record = swap_coordinates(
        h, V, alpha=1.0, realization_policy=MODEL_DTYPE_REALIZATION
    )
    assert record["model_dtype_realization_converged"] is False
    assert (
        record["max_post_cast_relative_coordinate_update_error"]
        > POST_CAST_TOLERANCE
    )
    row = _clean_row(
        all_model_dtype_realizations_converged=False,
        max_coordinate_update_error=float(
            record["max_post_cast_relative_coordinate_update_error"]
        ),
    )
    assert trial_integrity(row, layers=_BAND)["passed"] is False


def test_the_policy_payload_and_digest_travel_with_every_saved_trial() -> None:
    V = _near_degenerate_basis(256, seed=0)
    h = _activation(V, scale=6.0, seed=0)
    _patched, record = swap_coordinates(
        h, V, alpha=1.0, realization_policy=MODEL_DTYPE_REALIZATION
    )
    stats = {
        layer: {
            "n_forward_passes": 1,
            "n_positions": int(record["n_positions"]),
            "positions": [0, 1, 2, 3],
            "basis": {"diagnostics": {"condition_number": 3.0, "numerical_rank": 2}},
            "swap_history": [record],
        }
        for layer in _BAND
    }
    summary = summarize_swap_diagnostics(stats, expected_forward_passes=1)
    assert summary["model_dtype_realization_policy_supplied"] is True
    assert summary["model_dtype_realization_policy"] == (
        MODEL_DTYPE_REALIZATION.to_dict()
    )
    row = generation_trial_row(
        {
            "generated_text": "moo",
            "alpha": 1.0,
            "layers_patched": _BAND,
            "all_prompt_positions_patched": True,
            "n_forward_passes": 1,
            "teacher_forcing_used": False,
            "candidate_list_supplied": False,
            "intervention_diagnostics": summary,
        },
        group={"group_id": "g", "image_id": "i"},
        modality="text",
        condition="exact",
        direction=("cat", "cow"),
        answer={"aliases": ["moo"]},
        layers=_BAND,
    )
    assert row["model_dtype_realization_policy"] == MODEL_DTYPE_REALIZATION.to_dict()
    assert row["model_dtype_realization_policy_digest"] == realization_policy_digest()
    assert row["max_model_dtype_corrections_applied"] >= 1
    assert isinstance(row["diagnostic_checksum"], str)
    clauses = trial_integrity(row, layers=_BAND)["clauses"]
    assert clauses["model_dtype_realization_converged"] is True
    assert clauses["post_cast_coordinate_error_within_tolerance"] is True
    assert clauses["post_cast_residual_drift_within_tolerance"] is True
    assert clauses["all_hooks_fired"] is True
    assert clauses["all_finite"] is True
    # This fixture uses a deliberately ill-conditioned basis (two nearly
    # parallel concept directions), so the exchange's update is large relative
    # to the activation and the *magnitude* clause fails. That is the gate
    # working, and it is a separate question from realization fidelity.
    assert clauses["update_to_activation_ratio_within_limit"] is False


def test_a_changed_policy_changes_the_digest() -> None:
    looser = ModelDtypeRealizationPolicy(
        max_corrections=8,
        relative_coordinate_tolerance=0.05,
        relative_residual_tolerance=0.05,
        minimum_scale=1.0,
    )
    assert realization_policy_digest(looser) != realization_policy_digest()


# ---------------------------------------------------------- the real by_layer


def test_by_layer_is_a_dict_keyed_by_string_layer_numbers() -> None:
    """The shape the previous test fake got wrong.

    ``summarize_swap_diagnostics`` returns ``by_layer`` as a mapping keyed by
    the layer number rendered as a string. The old harness fake returned a
    list, so ``generation_trial_row``'s mapping branch never executed under
    test and the verdict bug reached a real GPU run.
    """
    V = _near_degenerate_basis(64, seed=2)
    h = _activation(V, scale=2.0, seed=2)
    _patched, record = swap_coordinates(
        h, V, alpha=1.0, realization_policy=MODEL_DTYPE_REALIZATION
    )
    stats = {
        layer: {
            "n_forward_passes": 2,
            "n_positions": 4,
            "positions": [0, 1, 2, 3],
            "basis": {"diagnostics": {"condition_number": 3.0, "numerical_rank": 2}},
            "swap_history": [record, record],
        }
        for layer in (16, 17, 40)
    }
    summary = summarize_swap_diagnostics(stats, expected_forward_passes=2)
    assert isinstance(summary["by_layer"], dict)
    assert sorted(summary["by_layer"]) == ["16", "17", "40"]
    assert all(isinstance(key, str) for key in summary["by_layer"])
    # and the flattener consumes that shape without help
    row = generation_trial_row(
        {
            "generated_text": "moo",
            "alpha": 1.0,
            "layers_patched": [16, 17, 40],
            "all_prompt_positions_patched": True,
            "n_forward_passes": 2,
            "intervention_diagnostics": summary,
        },
        group={"group_id": "g", "image_id": "i"},
        modality="image",
        condition="exact",
        direction=("cat", "cow"),
        answer={"aliases": ["moo"]},
        layers=[16, 17, 40],
    )
    assert row["max_activation_norm_ratio"] > 0.0
    assert row["all_hooks_fired"] is True


def test_the_hook_path_records_the_policy_it_ran_under() -> None:
    """Through the real hooks on real ``nn.Module`` blocks, not a stand-in.

    ``coordinate_swap_band`` is what installs the correction, and
    ``summarize_swap_diagnostics`` is what a saved trial carries. Driving a
    forward through both is the smallest test that the policy actually reaches
    the tensor the model consumes.
    """
    from jlens.mmpilot.coordinate_swap import (
        ConceptToken,
        build_swap_basis_from_vectors,
        coordinate_swap_band,
    )

    d_model = 256
    V = _near_degenerate_basis(d_model, seed=0)
    bases = {
        layer: build_swap_basis_from_vectors(
            V[:, 0].float(), V[:, 1].float(), layer=layer,
            source=ConceptToken("cat", 1, "cat", "{}"),
            target=ConceptToken("cow", 2, "cow", "{}"),
        )
        for layer in range(3)
    }
    blocks = [torch.nn.Identity() for _ in bases]
    hidden = _activation(V, scale=6.0, seed=0).unsqueeze(0)

    with coordinate_swap_band(
        blocks, bases, alpha=1.0, prompt_len=int(hidden.shape[1]),
        realization_policy=MODEL_DTYPE_REALIZATION,
        record_coordinates=False,
    ) as stats:
        out = hidden
        for block in blocks:
            out = block(out)

    assert out.dtype == hidden.dtype
    summary = summarize_swap_diagnostics(stats, expected_forward_passes=1)
    assert summary["all_hooks_fired"] is True
    assert summary["model_dtype_realization_policy_supplied"] is True
    assert summary["model_dtype_realization_policy"] == (
        MODEL_DTYPE_REALIZATION.to_dict()
    )
    assert summary["max_model_dtype_corrections_applied"] >= 1
    assert isinstance(summary["by_layer"], dict)
    # and the same call without a policy records that fact rather than hiding it
    with coordinate_swap_band(
        blocks, bases, alpha=1.0, prompt_len=int(hidden.shape[1]),
        record_coordinates=False,
    ) as bare_stats:
        out = hidden
        for block in blocks:
            out = block(out)
    bare = summarize_swap_diagnostics(bare_stats, expected_forward_passes=1)
    assert bare["model_dtype_realization_policy_supplied"] is False
    assert bare["model_dtype_realization_policy"] is None
    assert (
        bare["max_post_cast_relative_coordinate_error"]
        > summary["max_post_cast_relative_coordinate_error"]
    )


# ---------------------------------------------------------- the amendments


def test_the_instrument_amendment_recomputes_nothing() -> None:
    amendment = instrument_defect_amendment(
        original_report_path="/drive/runs/x/report.json",
        original_report_checksum="sha256:" + "4" * 64,
        original_run_name="mmnewpropertyrescue_real_6af6affcb145",
        original_verdict="RECRUITED_NEW_PROPERTY_EXPLORATORY_NO_GO",
        omitted_clauses=(
            "all_hooks_fired",
            "post_cast_coordinate_error_within_tolerance",
        ),
        observed_post_cast_relative_errors={"exact": 0.21, "unrelated": 0.29},
        corrected_stage="5B1RC",
    )
    assert amendment["scientific_recompute"] == 0
    assert amendment["scientific_numbers_unchanged"] is True
    assert amendment["original_report_modified"] is False
    assert amendment["original_units_modified"] is False
    assert amendment["verdict_changed_by_prose"] is False
    assert amendment["corrected_classification"] == "INSTRUMENT_INCONCLUSIVE"
    assert amendment["original_verdict_is_reproduced_verbatim"] is True
    assert amendment["original_null_is_readable_as_a_scientific_null"] is False
    recomputed = payload_checksum({
        key: value for key, value in amendment.items()
        if key != "amendment_checksum"
    })
    assert recomputed == amendment["amendment_checksum"]


def test_catdog_matching_amendment_pins_the_completed_run_and_recomputes_nothing() -> None:
    amendment = direct_answer_matching_defect_amendment(
        original_report_path=(
            "/content/drive/MyDrive/jacobian-lens-gemma/runs/mmcatdogdev/"
            "mmcatdogdev_real_bc03b54e8494/catdog_development_report.json"
        ),
        original_report_checksum=(
            "sha256:ccf34c1303c17960edc13653298c4badba462a7cbda5ca90b5fb637d7af04be2"
        ),
        original_run_name="mmcatdogdev_real_bc03b54e8494",
        original_verdict="NEW_PROPERTY_DEVELOPMENT_NO_GO",
        observed_direct_to_exact_ratios={
            "minimum": 2.3,
            "median": 3.1,
            "maximum": 4.7,
        },
        n_trials=24,
        corrected_stage="6C",
        written_utc="2026-08-25T00:00:00+00:00",
    )
    assert amendment["scientific_recompute"] == 0
    assert amendment["original_report_modified"] is False
    assert amendment["original_units_modified"] is False
    assert amendment["corrected_classification"] == "INCONCLUSIVE"
    assert amendment["original_null_is_readable_as_a_scientific_null"] is False
    assert amendment["omitted_integrity_clauses"] == [
        "cumulative_band_displacement_norm_matched"
    ]
    assert amendment["observed_direct_to_exact_cumulative_displacement_ratio"][
        "median"
    ] == 3.1
    recomputed = payload_checksum(
        {key: value for key, value in amendment.items() if key != "amendment_checksum"}
    )
    assert recomputed == amendment["amendment_checksum"]


def test_an_amendment_that_names_nothing_is_refused() -> None:
    with pytest.raises(MultimodalFollowupRefused):
        instrument_defect_amendment(
            original_report_path="",
            original_report_checksum="sha256:x",
            original_run_name="r",
            original_verdict="v",
            omitted_clauses=("all_finite",),
            observed_post_cast_relative_errors={},
            corrected_stage="5B1RC",
        )
    with pytest.raises(MultimodalFollowupRefused):
        instrument_defect_amendment(
            original_report_path="/p",
            original_report_checksum="sha256:x",
            original_run_name="r",
            original_verdict="v",
            omitted_clauses=(),
            observed_post_cast_relative_errors={},
            corrected_stage="5B1RC",
        )


def test_an_amendment_cannot_invent_a_clause() -> None:
    with pytest.raises(MultimodalFollowupRefused):
        instrument_defect_amendment(
            original_report_path="/p",
            original_report_checksum="sha256:x",
            original_run_name="r",
            original_verdict="v",
            omitted_clauses=("everything_was_fine",),
            observed_post_cast_relative_errors={},
            corrected_stage="5B1RC",
        )


def test_the_legacy_audit_neither_reaffirms_nor_invalidates() -> None:
    audit = legacy_confirmation_realization_audit(
        report_checksum="sha256:2bb6dcc1346229573566125bc8d91c782247d55af5091f4215d98bb621472ff7",
        trial_function="jlens.mmpilot.multimodal_lens.unrestricted_swap_trial",
        realization_policy_passed=False,
        stored_diagnostic_fields=[
            "max_activation_norm_ratio",
            "max_coordinate_update_error",
            "max_orthogonal_residual_drift",
            "max_update_to_activation_norm_ratio",
        ],
        enforced_integrity_clauses=[
            "expected_layers_patched",
            "all_prompt_positions_patched",
        ],
        enforced_tolerance=1e-5,
    )
    assert audit["verdict"] == "ARTIFACTS_INSUFFICIENT_REPLICATION_REQUIRED"
    assert audit["artifacts_sufficient"] is False
    assert audit["post_cast_diagnostics_stored"] == []
    assert audit["reaffirms_original_result"] is False
    assert audit["invalidates_original_result"] is False
    assert audit["scientific_recompute"] == 0
    assert audit["required_replication"]["writes_to_original_run"] is False


def test_the_legacy_audit_clears_a_run_whose_artifacts_do_settle_it() -> None:
    audit = legacy_confirmation_realization_audit(
        report_checksum="sha256:" + "1" * 64,
        trial_function="whatever",
        realization_policy_passed=True,
        stored_diagnostic_fields=["max_post_cast_relative_coordinate_error"],
        enforced_integrity_clauses=[
            "post_cast_coordinate_error_within_tolerance",
            "model_dtype_realization_converged",
        ],
        enforced_tolerance=0.02,
    )
    assert audit["verdict"] == "ARTIFACTS_SUFFICIENT_AND_CLEAN"
    assert audit["required_replication"] is None


# -------------------------------------------------- the corrected verdict wiring


def _corrected_rows(*, exact_moves: bool, direct_moves: bool) -> list[dict]:
    rows = []
    for direction in (("cat", "cow"), ("cow", "cat")):
        answer = "moo" if direction[1] == "cow" else "meow"
        source_sound = "meow" if direction[0] == "cat" else "moo"
        for modality in ("text", "image", "spoken_audio"):
            for index in range(8):
                group = {"group_id": f"g{index}", "image_id": f"i{index}"}
                for condition in ("exact", "zero", "random", "unrelated"):
                    moved = condition == "exact" and exact_moves
                    rows.append({
                        **_clean_row(
                            condition=condition,
                            alpha=0.0 if condition == "zero" else 1.0,
                            all_layers_are_exact_alpha_one_exchange_before_cast=(
                                condition != "zero"
                            ),
                        ),
                        "direction": f"{direction[0]}->{direction[1]}",
                        "modality": modality,
                        "group_id": group["group_id"],
                        "image_id": group["image_id"],
                        "generated_text": answer if moved else source_sound,
                        "expected": answer,
                        "success": moved,
                    })
                control = _clean_row(condition="direct_answer", alpha=1.0)
                control.pop("all_layers_are_exact_alpha_one_exchange_before_cast")
                control.pop("max_coordinate_update_error")
                control.pop("max_orthogonal_residual_drift")
                control["max_relative_norm_match_error"] = 0.001
                control["max_relative_cumulative_band_displacement_match_error"] = 0.001
                control["max_activation_norm_ratio"] = 1.0
                rows.append({
                    **control,
                    "direction": f"{direction[0]}->{direction[1]}",
                    "modality": modality,
                    "group_id": group["group_id"],
                    "image_id": group["image_id"],
                    "generated_text": answer if direct_moves else source_sound,
                    "expected": answer,
                    "success": direct_moves,
                })
    return rows


_SOURCE_AUDIT = {
    "verdict": "PROPERTY_AUDIT_NO_GO",
    "family": "animal_sound",
    "audit_digest": "sha256:" + "5" * 64,
}
_LINKAGE = {"verdict": "AUDIO_METADATA_LINKAGE_GO", "audit_digest": "sha256:" + "6" * 64}
_RECRUITMENT = {"complete": True, "selection_digest": "sha256:" + "7" * 64}
_SUPERSEDED = "sha256:467a2862cef70f0b59a75678c6a73c68259f4b29f715a97fbb831914710f660a"


def test_the_corrected_verdict_refuses_a_rerun_without_the_positive_control() -> None:
    rows = [
        row for row in _corrected_rows(exact_moves=False, direct_moves=True)
        if row["condition"] != "direct_answer"
    ]
    with pytest.raises(MultimodalFollowupRefused, match="positive control"):
        corrected_exploratory_verdict(
            rows,
            source_audit=_SOURCE_AUDIT,
            linkage_audit=_LINKAGE,
            recruitment=_RECRUITMENT,
            superseded_report_checksum=_SUPERSEDED,
            layers=_BAND,
        )


def test_the_corrected_verdict_must_pin_what_it_supersedes() -> None:
    with pytest.raises(MultimodalFollowupRefused, match="supersedes"):
        corrected_exploratory_verdict(
            _corrected_rows(exact_moves=False, direct_moves=True),
            source_audit=_SOURCE_AUDIT,
            linkage_audit=_LINKAGE,
            recruitment=_RECRUITMENT,
            superseded_report_checksum="",
            layers=_BAND,
        )


@pytest.mark.parametrize(
    "exact_moves, direct_moves, expected",
    [
        (True, True, "EFFECT_GO"),
        (True, False, "EFFECT_GO"),
        (False, True, "SCIENTIFIC_NULL"),
        (False, False, "INCONCLUSIVE"),
    ],
)
def test_the_corrected_verdict_maps_the_frozen_interpretation_table(
    exact_moves: bool, direct_moves: bool, expected: str
) -> None:
    report = corrected_exploratory_verdict(
        _corrected_rows(exact_moves=exact_moves, direct_moves=direct_moves),
        source_audit=_SOURCE_AUDIT,
        linkage_audit=_LINKAGE,
        recruitment=_RECRUITMENT,
        superseded_report_checksum=_SUPERSEDED,
        layers=_BAND,
    )
    assert report["instrument_state"] == expected
    assert report["verdict"] == f"CORRECTED_RECRUITED_EXPLORATORY_{expected}"
    assert report["supersedes_report_checksum"] == _SUPERSEDED
    assert report["superseded_run_rewritten"] is False
    assert report["alpha_sweep_run"] is False
    assert report["is_confirmation"] is False
    assert json.dumps(report, default=str)


def test_the_corrected_verdict_calls_an_unrealized_exchange_an_instrument_failure() -> None:
    rows = _corrected_rows(exact_moves=False, direct_moves=True)
    for row in rows:
        if row["condition"] != "direct_answer":
            # the worst post-cast relative coordinate error the completed
            # flawed run actually recorded
            row["max_coordinate_update_error"] = 0.21
    report = corrected_exploratory_verdict(
        rows,
        source_audit=_SOURCE_AUDIT,
        linkage_audit=_LINKAGE,
        recruitment=_RECRUITMENT,
        superseded_report_checksum=_SUPERSEDED,
        layers=_BAND,
    )
    assert report["instrument_state"] == "INSTRUMENT_FAILURE"
    assert "SCIENTIFIC_NULL" not in report["verdict"]


# ------------------------------ the confirmed result's realization replication


def _replication_rows(
    *,
    original_error: float,
    corrected_error: float,
    corrected_converged: bool = True,
    reproduces: bool = True,
    corrected_changes_outcome: bool = False,
) -> list[dict]:
    rows = []
    for index in range(4):
        for modality in ("text", "image", "spoken_audio"):
            for condition in ("exact", "zero", "random", "unrelated"):
                stored_success = condition == "exact"
                stored_token = 4 if stored_success else 2
                rows.append({
                    "arm": "uncorrected",
                    "group_id": f"g{index}",
                    "modality": modality,
                    "condition": condition,
                    "layers_patched": list(_BAND),
                    "all_prompt_positions_patched": True,
                    "stored_top_token_id": stored_token,
                    "replayed_top_token_id": (
                        stored_token if reproduces else stored_token + 1
                    ),
                    "stored_success": stored_success,
                    "replayed_success": stored_success,
                    "max_post_cast_relative_coordinate_error": original_error,
                    "max_post_cast_relative_residual_drift": 0.001,
                    "all_model_dtype_realizations_converged": True,
                })
                corrected_success = (
                    not stored_success
                    if corrected_changes_outcome
                    else stored_success
                )
                rows.append({
                    "arm": "corrected",
                    "group_id": f"g{index}",
                    "modality": modality,
                    "condition": condition,
                    "layers_patched": list(_BAND),
                    "all_prompt_positions_patched": True,
                    "stored_top_token_id": stored_token,
                    "replayed_top_token_id": stored_token,
                    "stored_success": stored_success,
                    "replayed_success": corrected_success,
                    "max_post_cast_relative_coordinate_error": corrected_error,
                    "max_post_cast_relative_residual_drift": 0.001,
                    "all_model_dtype_realizations_converged": corrected_converged,
                })
    return rows


_CONFIRMED = "sha256:2bb6dcc1346229573566125bc8d91c782247d55af5091f4215d98bb621472ff7"


def test_a_replication_that_does_not_reproduce_the_stored_tokens_concludes_nothing() -> None:
    report = legacy_confirmation_replication_verdict(
        _replication_rows(original_error=0.21, corrected_error=0.004, reproduces=False),
        original_report_checksum=_CONFIRMED,
        layers=_BAND,
    )
    assert report["verdict"] == "CONFIRMATION_REPLICATION_FAILED"
    assert report["uncorrected_reproduced_stored_tokens"] is False


def test_an_originally_faithful_cast_needs_no_repair() -> None:
    report = legacy_confirmation_replication_verdict(
        _replication_rows(original_error=0.001, corrected_error=0.001),
        original_report_checksum=_CONFIRMED,
        layers=_BAND,
    )
    assert report["verdict"] == "CONFIRMATION_REALIZATION_CLEAN"
    assert report["original_within_tolerance"] is True


def test_a_repaired_instrument_that_preserves_the_outcome_is_named_as_such() -> None:
    report = legacy_confirmation_replication_verdict(
        _replication_rows(original_error=0.21, corrected_error=0.004),
        original_report_checksum=_CONFIRMED,
        layers=_BAND,
    )
    assert report["verdict"] == "CONFIRMATION_REALIZATION_REPAIRED_AND_PRESERVED"
    assert report["n_outcomes_changed"] == 0


def test_a_changed_outcome_is_reported_not_used_to_relabel() -> None:
    report = legacy_confirmation_replication_verdict(
        _replication_rows(
            original_error=0.21, corrected_error=0.004,
            corrected_changes_outcome=True,
        ),
        original_report_checksum=_CONFIRMED,
        layers=_BAND,
    )
    assert report["verdict"] == "CONFIRMATION_REALIZATION_REPAIRED_AND_CHANGED"
    assert report["n_outcomes_changed"] > 0
    assert report["original_verdict_relabelled"] is False
    assert report["original_report_modified"] is False


def test_a_non_convergent_correction_leaves_the_question_open() -> None:
    report = legacy_confirmation_replication_verdict(
        _replication_rows(
            original_error=0.21, corrected_error=0.15, corrected_converged=False,
        ),
        original_report_checksum=_CONFIRMED,
        layers=_BAND,
    )
    assert report["verdict"] == "CONFIRMATION_REALIZATION_INSTRUMENT_FAILURE"


def test_the_replication_refuses_a_single_arm() -> None:
    rows = [
        row for row in _replication_rows(original_error=0.21, corrected_error=0.004)
        if row["arm"] == "corrected"
    ]
    with pytest.raises(MultimodalFollowupRefused, match="both"):
        legacy_confirmation_replication_verdict(
            rows, original_report_checksum=_CONFIRMED, layers=_BAND
        )


def test_the_replication_must_pin_the_report_it_replays() -> None:
    with pytest.raises(MultimodalFollowupRefused, match="pin"):
        legacy_confirmation_replication_verdict(
            _replication_rows(original_error=0.21, corrected_error=0.004),
            original_report_checksum="",
            layers=_BAND,
        )


def test_the_direct_answer_row_is_never_counted_as_a_coordinate_exchange() -> None:
    row = direct_answer_trial_row(
        {
            "generated_text": "moo",
            "alpha": 1.0,
            "layers_patched": _BAND,
            "all_prompt_positions_patched": True,
            "n_forward_passes": 1,
            "intervention_diagnostics": {
                "by_layer": {
                    str(layer): {"max_update_to_activation_ratio": 0.2}
                    for layer in _BAND
                },
                "all_hooks_fired": True,
                "all_finite": True,
                "all_model_dtype_realizations_converged": True,
                "max_relative_norm_match_error": 0.001,
                "max_exact_exchange_cumulative_band_displacement_norm": 3.0,
                "max_direct_answer_cumulative_band_displacement_norm": 3.0,
                "max_relative_cumulative_band_displacement_match_error": 0.001,
                "model_dtype_realization_policy": MODEL_DTYPE_REALIZATION.to_dict(),
            },
        },
        group={"group_id": "g", "image_id": "i"},
        modality="text",
        direction=("cat", "cow"),
        answer={"aliases": ["moo"]},
        layers=_BAND,
    )
    assert row["condition"] == "direct_answer"
    assert row["is_coordinate_exchange"] is False
    assert row["arm"] == "positive_control"
    assert trial_integrity(row, layers=_BAND)["passed"] is True
