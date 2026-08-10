# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""The multimodal half of the L27-L31 study: pins, gates, verdicts, contracts.

The tests here are about the four ways a favourable answer could be reached for
the wrong reason: combining halves measured on different populations, reusing
spent media because a pin was guessed, letting a stale derived gate spend an L4
hour, and calling additive steering a coordinate swap.
"""

import pytest

from jlens.calibration.adjacent import ADJACENT_LENS_GO, ADJACENT_LENS_NO_GO
from jlens.mmpilot.convergence import AMBIGUOUS, CONVERGED, NOT_CONVERGED
from jlens.mmpilot.preconvergence import (
    AMBIGUOUS_CONVERGENCE,
    CAUSAL_TRANSFER_NOT_SUPPORTED,
    COMPLETED_AUDIO_TRANSFER_RUN,
    COMPLETED_FOLLOWUP_RUN,
    CONVERGED_BEFORE_CAUSAL_TEST,
    COORDINATE_SWAP_SCOPE,
    FROZEN_CRITERION_DIGEST,
    LAYER_AMBIGUOUS,
    LAYER_CONVERGED,
    LAYER_NOT_CONVERGED,
    PRECONVERGENCE_RAW_SWITCHES,
    PRECONVERGENCE_RUN_PREFIX,
    PRECONVERGENCE_SUPPORTED,
    REFUSED_INVALID,
    REQUIRED_CAUSAL_CONTROLS,
    REQUIRED_MODALITIES,
    TERMINAL_OUTCOMES,
    TRANSFER_DESCRIPTIVE_ONLY,
    TRANSFER_NOT_EVALUATED,
    TRANSFER_NOT_SUPPORTED,
    TRANSFER_SUPPORTED,
    VERDICT_NAMES,
    PinNotSet,
    PopulationsDiffer,
    PreconvergenceRefused,
    adjacent_lens_integrity,
    assert_causal_controls_recorded,
    assert_completed_population_pins,
    assert_fresh_run_namespace,
    assert_same_population,
    build_summary,
    check_preconvergence_call_contracts,
    convergence_verdict_for_layer,
    derive_preconvergence_gates,
    format_preconvergence_gates,
    format_stage_plan,
    preconvergence_call_contracts,
    preconvergence_fingerprint,
    preconvergence_verdicts,
    refresh_preconvergence_gates,
    render_report,
    stage_four_decision,
    stage_plan,
)
from jlens.mmpilot.stage_gates import MissingStageSwitch

# ------------------------------------------------------------------ the pins

RUNS = "/drive/runs"
PINNED = f"{RUNS}/mml32res_completed_20260809T101112"
ALL_RUNS = (
    f"{RUNS}/{COMPLETED_AUDIO_TRANSFER_RUN}",
    f"{RUNS}/{COMPLETED_FOLLOWUP_RUN}",
    PINNED,
)


def test_all_three_spent_populations_must_be_excluded():
    record = assert_completed_population_pins(ALL_RUNS, resolution_run_dir=PINNED)
    assert record["n_excluded_runs"] == 3
    assert record["pin_was_discovered"] is False
    assert record["pin_was_defaulted"] is False
    assert record["pinned_resolution_run"].startswith("mml32res")


def test_an_omitted_named_run_is_refused():
    with pytest.raises(PinNotSet, match=COMPLETED_AUDIO_TRANSFER_RUN):
        assert_completed_population_pins(ALL_RUNS[1:], resolution_run_dir=PINNED)


@pytest.mark.parametrize("empty", ["", "   ", None])
def test_an_unset_resolution_pin_is_refused_and_never_discovered(empty):
    with pytest.raises(PinNotSet, match="not pinned"):
        assert_completed_population_pins(ALL_RUNS, resolution_run_dir=empty)


def test_a_pin_with_the_wrong_family_is_refused():
    with pytest.raises(PinNotSet, match="does not start with"):
        assert_completed_population_pins(
            (*ALL_RUNS, f"{RUNS}/mml32_something"),
            resolution_run_dir=f"{RUNS}/mml32_something",
        )


def test_a_pin_that_is_not_excluded_excludes_nothing():
    with pytest.raises(PinNotSet, match="not in the exclusion list"):
        assert_completed_population_pins(
            ALL_RUNS[:2] + (f"{RUNS}/mml32res_a",),
            resolution_run_dir=f"{RUNS}/mml32res_b",
        )


def test_the_refusal_explains_why_the_pin_is_manual():
    record = assert_completed_population_pins(ALL_RUNS, resolution_run_dir=PINNED)
    assert "newest" in record["why_the_pin_is_manual"]


# -------------------------------------------------------- the run namespace


def test_a_run_inside_a_completed_namespace_is_refused(tmp_path):
    protected = ("mml32_", "mml32res_", "rgext_")
    with pytest.raises(PreconvergenceRefused, match="completed run namespace"):
        assert_fresh_run_namespace(
            tmp_path / "mml32res_done" / "mmpre_x", protected_prefixes=protected
        )


def test_a_run_that_does_not_name_its_family_is_refused(tmp_path):
    with pytest.raises(PreconvergenceRefused, match="mmpre"):
        assert_fresh_run_namespace(tmp_path / "scratch", protected_prefixes=())


def test_a_well_named_fresh_run_is_accepted(tmp_path):
    root = tmp_path / f"{PRECONVERGENCE_RUN_PREFIX}_mock_abc"
    assert assert_fresh_run_namespace(root, protected_prefixes=("rgext_",)) == str(root)


# ------------------------------------------------------------- stage gates


def _switches(**overrides):
    values = dict.fromkeys(PRECONVERGENCE_RAW_SWITCHES, False)
    values.update(overrides)
    return values


def test_every_gate_is_closed_when_every_switch_is_false():
    assert not any(derive_preconvergence_gates(_switches()).values())


def test_preprocessing_only_closes_every_gate_even_with_model_switches_set():
    gates = derive_preconvergence_gates(
        _switches(
            PREPROCESSING_ONLY=True,
            RUN_LENS_FITTING=True,
            CONFIRM_FITTING_BUDGET=True,
            RUN_MODEL_STAGE=True,
            CONFIRM_MODEL_LOAD=True,
            CONFIRM_STAGE_3_BUDGET=True,
            RUN_STAGE_4_CAUSAL_TRANSFER=True,
            CONFIRM_STAGE_4_BUDGET=True,
        )
    )
    assert not any(gates.values())


def test_each_expensive_stage_needs_its_own_confirmation():
    assert not derive_preconvergence_gates(_switches(RUN_LENS_FITTING=True))[
        "FITTING_ENABLED"
    ]
    assert derive_preconvergence_gates(
        _switches(RUN_LENS_FITTING=True, CONFIRM_FITTING_BUDGET=True)
    )["FITTING_ENABLED"]
    assert not derive_preconvergence_gates(_switches(RUN_MODEL_STAGE=True))[
        "MODEL_STAGE_ENABLED"
    ]
    assert not derive_preconvergence_gates(
        _switches(
            RUN_MODEL_STAGE=True,
            CONFIRM_MODEL_LOAD=True,
            RUN_STAGE_4_CAUSAL_TRANSFER=True,
            CONFIRM_STAGE_4_BUDGET=True,
        )
    )["STAGE_4_REQUESTED"]


def test_a_missing_raw_switch_raises_rather_than_defaulting_to_false():
    partial = _switches()
    partial.pop("RUN_MODEL_STAGE")
    with pytest.raises(MissingStageSwitch, match="RUN_MODEL_STAGE"):
        derive_preconvergence_gates(partial)


def test_refreshing_rederives_a_stale_gate_from_the_raw_switches():
    namespace = _switches(RUN_LENS_FITTING=True, CONFIRM_FITTING_BUDGET=True)
    namespace["FITTING_ENABLED"] = False  # a stale value from an earlier cell
    refresh_preconvergence_gates(namespace)
    assert namespace["FITTING_ENABLED"] is True
    namespace["RUN_LENS_FITTING"] = False
    refresh_preconvergence_gates(namespace)
    assert namespace["FITTING_ENABLED"] is False


def test_the_gate_block_prints_every_raw_switch_and_every_derived_gate():
    switches = _switches()
    text = format_preconvergence_gates(
        derive_preconvergence_gates(switches), switches=switches
    )
    for name in PRECONVERGENCE_RAW_SWITCHES:
        assert name in text
    for name in ("FITTING_ENABLED", "STAGE_3_ENABLED", "STAGE_4_REQUESTED"):
        assert name in text


# -------------------------------------------------------------- stage plan


def test_the_stage_plan_has_all_five_stages_and_only_stage_zero_is_model_free():
    plan = stage_plan()
    assert [entry["stage"] for entry in plan["stages"]] == [0, 1, 2, 3, 4]
    assert plan["stages"][0]["loads_model"] is False
    assert all(entry["loads_model"] for entry in plan["stages"][1:])
    assert plan["required_modalities"] == list(REQUIRED_MODALITIES)
    assert FROZEN_CRITERION_DIGEST in str(plan["stages"][3]["contents"])


def test_the_plan_states_the_coordinate_swap_is_out_of_scope():
    plan = stage_plan()
    assert "OUT OF SCOPE" in plan["coordinate_swap_scope"]
    assert "swap" in format_stage_plan(plan)
    assert "steering" in plan["coordinate_swap_scope"]


def test_the_stage_four_gate_needs_all_four_clauses():
    ok = {
        "lens_verdict": ADJACENT_LENS_GO,
        "convergence_verdict": LAYER_NOT_CONVERGED,
        "controls_passed": True,
        "capability_sufficient": True,
        "requested": True,
        "budget_confirmed": True,
    }
    assert stage_four_decision(**ok)["gate_met"] is True
    for field, bad in (
        ("lens_verdict", ADJACENT_LENS_NO_GO),
        ("convergence_verdict", LAYER_CONVERGED),
        ("convergence_verdict", LAYER_AMBIGUOUS),
        ("controls_passed", False),
        ("capability_sufficient", False),
    ):
        decision = stage_four_decision(**{**ok, field: bad})
        assert decision["gate_met"] is False
        assert decision["gate_overridden"] is True
        assert decision["evidence_status"] == TRANSFER_DESCRIPTIVE_ONLY


def test_stage_four_does_not_run_without_its_own_budget_confirmation():
    decision = stage_four_decision(
        lens_verdict=ADJACENT_LENS_GO,
        convergence_verdict=LAYER_NOT_CONVERGED,
        controls_passed=True,
        capability_sufficient=True,
        requested=True,
        budget_confirmed=False,
    )
    assert decision["runs"] is False


def test_the_stage_four_rule_is_stated_as_an_efficiency_gate():
    decision = stage_four_decision(
        lens_verdict=ADJACENT_LENS_GO,
        convergence_verdict=LAYER_CONVERGED,
        controls_passed=True,
        capability_sufficient=True,
        requested=False,
        budget_confirmed=False,
    )
    assert "UNFAVOURABLE" in decision["rationale"]
    assert "not a filter" in decision["rationale"]


# ------------------------------------------------------- convergence verdict


def _convergence_inputs(classification=NOT_CONVERGED, **overrides):
    payload = {
        "layer": 29,
        "integrity": {"verdict": "PASSED"},
        "convergence": {
            "layer": 29,
            "criterion_digest": FROZEN_CRITERION_DIGEST,
            "classification": {"classification": classification},
            "summary": {
                "per_layer": {
                    "29": {
                        "per_modality": {
                            modality: {"n": 64, "n_distinct_predictions": 5}
                            for modality in REQUIRED_MODALITIES
                        }
                    }
                }
            },
        },
        "controls": {"passed": True, "missing_or_empty": [], "failing": []},
        "disjointness": {"disjoint": True, "failed_families": []},
        "pseudoreplication": {
            "passed": True,
            "n_units": 48,
            "n_distinct_images": 48,
        },
        "sample_plan": {"plan_digest": "sha256:plan"},
        "head_agreement": {
            "passed": True,
            "matches_model_unembed": True,
            "comparison_ran": True,
        },
        "admissibility": {
            "eligible_concepts": ["cat", "zebra"],
            "excluded_concept_names": [],
        },
        "leakage_audit": {"passed": True, "per_modality": {}},
    }
    payload.update(overrides)
    return payload


@pytest.mark.parametrize(
    ("classification", "expected"),
    [
        (NOT_CONVERGED, LAYER_NOT_CONVERGED),
        (CONVERGED, LAYER_CONVERGED),
        (AMBIGUOUS, LAYER_AMBIGUOUS),
    ],
)
def test_each_classification_maps_to_exactly_one_verdict(classification, expected):
    verdict = convergence_verdict_for_layer(**_convergence_inputs(classification))
    assert verdict["verdict"] == expected
    assert verdict["failed_validity_clauses"] == []


def test_a_changed_criterion_digest_refuses_rather_than_classifying():
    inputs = _convergence_inputs()
    inputs["convergence"] = {**inputs["convergence"], "criterion_digest": "sha256:x"}
    verdict = convergence_verdict_for_layer(**inputs)
    assert verdict["verdict"] == REFUSED_INVALID
    assert "criterion_digest_unchanged" in verdict["failed_validity_clauses"]


def test_a_leaked_candidate_refuses_the_whole_measurement():
    inputs = _convergence_inputs()
    inputs["leakage_audit"] = {"passed": False, "per_modality": {"image": "leak"}}
    verdict = convergence_verdict_for_layer(**inputs)
    assert verdict["verdict"] == REFUSED_INVALID
    assert (
        "no_candidate_leaked_into_any_model_visible_prompt"
        in verdict["failed_validity_clauses"]
    )


def test_missing_controls_are_not_an_ambiguous_result():
    inputs = _convergence_inputs(AMBIGUOUS)
    inputs["controls"] = {
        "passed": False,
        "missing_or_empty": ["permuted"],
        "failing": [],
    }
    verdict = convergence_verdict_for_layer(**inputs)
    assert verdict["verdict"] == REFUSED_INVALID
    assert "has observed nothing" in verdict["rationale"]


def test_a_degenerate_modality_cell_refuses():
    inputs = _convergence_inputs()
    inputs["convergence"]["summary"]["per_layer"]["29"]["per_modality"][
        "spoken_audio"
    ] = {"n": 64, "n_distinct_predictions": 1}
    verdict = convergence_verdict_for_layer(**inputs)
    assert verdict["verdict"] == REFUSED_INVALID
    assert "outputs_finite_and_nondegenerate" in verdict["failed_validity_clauses"]


def test_the_measured_layer_is_read_from_the_measurement_not_the_argument():
    # The MOCK world stands in for a deep layer at a shallow index.
    inputs = _convergence_inputs()
    inputs["convergence"] = {**inputs["convergence"], "layer": 1}
    inputs["convergence"]["summary"] = {
        "per_layer": {
            "1": {
                "per_modality": {
                    modality: {"n": 64, "n_distinct_predictions": 5}
                    for modality in REQUIRED_MODALITIES
                }
            }
        }
    }
    verdict = convergence_verdict_for_layer(**inputs)
    assert verdict["measured_layer"] == 1
    assert verdict["layer"] == 29
    assert verdict["verdict"] == LAYER_NOT_CONVERGED


# ----------------------------------------------------------- causal controls


def _all_controls(**overrides):
    controls = {
        name: {"passed": True, "detail": "ok"} for name in REQUIRED_CAUSAL_CONTROLS
    }
    controls.update(overrides)
    return controls


def test_all_seven_required_causal_controls_are_checked():
    record = assert_causal_controls_recorded(_all_controls())
    assert record["passed"] is True
    assert [row["control"] for row in record["rows"]] == list(REQUIRED_CAUSAL_CONTROLS)


@pytest.mark.parametrize("missing", REQUIRED_CAUSAL_CONTROLS)
def test_a_missing_control_record_is_a_failure_never_a_pass(missing):
    controls = _all_controls()
    controls.pop(missing)
    record = assert_causal_controls_recorded(controls)
    assert record["passed"] is False
    assert missing in record["missing_or_empty"]


def test_a_failing_control_is_distinguished_from_a_missing_one():
    record = assert_causal_controls_recorded(
        _all_controls(zero_intervention={"passed": False, "detail": "drifted"})
    )
    assert record["failing"] == ["zero_intervention"]
    assert record["missing_or_empty"] == []


# ---------------------------------------------------------- same population


def test_two_halves_of_one_population_at_one_layer_are_combinable():
    record = assert_same_population(
        convergence_population_digest="sha256:p",
        causal_population_digest="sha256:p",
        convergence_layer=29,
        causal_layer=29,
    )
    assert record["combinable"] is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"causal_population_digest": "sha256:other"},
        {"causal_layer": 31},
        {"causal_population_digest": None},
    ],
)
def test_mismatched_halves_are_refused(kwargs):
    base = {
        "convergence_population_digest": "sha256:p",
        "causal_population_digest": "sha256:p",
        "convergence_layer": 29,
        "causal_layer": 29,
    }
    with pytest.raises(PopulationsDiffer):
        assert_same_population(**{**base, **kwargs})


# ------------------------------------------------------------- the verdicts


def _verdicts(
    lens=ADJACENT_LENS_GO,
    convergence=LAYER_NOT_CONVERGED,
    causal=TRANSFER_SUPPORTED,
    controls_pass=True,
    runs=True,
    overridden=False,
    combinable=True,
    representation_space="J_SPACE",
):
    return preconvergence_verdicts(
        lens_verdict={
            "verdict": lens,
            "selected_layer": 29 if lens == ADJACENT_LENS_GO else None,
            "rationale": "because",
        },
        convergence=(
            {"verdict": convergence, "rationale": "r", "failed_validity_clauses": []}
            if convergence
            else None
        ),
        causal=({"verdict": causal} if causal else None),
        causal_controls={
            "passed": controls_pass,
            "missing_or_empty": [],
            "failing": [],
        },
        stage_four={
            "runs": runs,
            "gate_overridden": overridden,
            "statement": "s",
        },
        same_population={"combinable": combinable},
        lens_arm={
            "label": "R_LENS" if representation_space == "R_SPACE" else "RAW_J_LENS"
        },
        representation_space=representation_space,
    )


def test_the_five_verdict_names_are_always_present():
    payload = _verdicts()
    assert list(payload["verdicts"]) == list(VERDICT_NAMES)
    assert len(VERDICT_NAMES) == 5


def test_the_principal_claim_is_supported_only_when_all_three_hold_together():
    payload = _verdicts()
    assert payload["terminal_outcome"] == PRECONVERGENCE_SUPPORTED
    assert payload["verdicts"]["PRECONVERGENCE_CAUSAL_TRANSFER"]["verdict"] == (
        TRANSFER_SUPPORTED
    )


def test_no_confirmed_lens_stops_the_study():
    payload = _verdicts(lens=ADJACENT_LENS_NO_GO)
    assert payload["terminal_outcome"] == ADJACENT_LENS_NO_GO
    assert payload["verdicts"]["EARLIEST_CONFIRMED_LAYER"]["verdict"] == "NONE"


def test_a_converged_layer_rules_the_claim_out_before_the_causal_test():
    payload = _verdicts(convergence=LAYER_CONVERGED)
    assert payload["terminal_outcome"] == CONVERGED_BEFORE_CAUSAL_TEST


def test_an_ambiguous_layer_supports_nothing():
    payload = _verdicts(convergence=LAYER_AMBIGUOUS)
    assert payload["terminal_outcome"] == AMBIGUOUS_CONVERGENCE


def test_an_invalid_convergence_measurement_refuses():
    payload = _verdicts(convergence=REFUSED_INVALID)
    assert payload["terminal_outcome"] == REFUSED_INVALID


def test_an_overridden_causal_run_is_descriptive_only_and_never_supports():
    payload = _verdicts(overridden=True)
    assert payload["verdicts"]["THREE_MODALITY_CAUSAL_TRANSFER"]["verdict"] == (
        TRANSFER_DESCRIPTIVE_ONLY
    )
    assert payload["terminal_outcome"] == CAUSAL_TRANSFER_NOT_SUPPORTED


def test_failing_causal_controls_cannot_yield_supported():
    payload = _verdicts(controls_pass=False)
    assert payload["verdicts"]["THREE_MODALITY_CAUSAL_TRANSFER"]["verdict"] == (
        TRANSFER_NOT_SUPPORTED
    )
    assert payload["terminal_outcome"] == CAUSAL_TRANSFER_NOT_SUPPORTED


def test_a_not_converged_layer_without_a_causal_measurement_says_so_plainly():
    payload = _verdicts(runs=False, causal=None)
    assert payload["verdicts"]["THREE_MODALITY_CAUSAL_TRANSFER"]["verdict"] == (
        TRANSFER_NOT_EVALUATED
    )
    assert payload["terminal_outcome"] == CAUSAL_TRANSFER_NOT_SUPPORTED
    assert "missing measurement, not a negative result" in payload["statement"]


def test_halves_from_different_populations_are_never_combined():
    payload = _verdicts(combinable=False)
    assert payload["terminal_outcome"] == REFUSED_INVALID
    assert "never combined" in payload["statement"]


def test_every_terminal_outcome_is_reachable():
    reached = {
        _verdicts()["terminal_outcome"],
        _verdicts(lens=ADJACENT_LENS_NO_GO)["terminal_outcome"],
        _verdicts(convergence=LAYER_CONVERGED)["terminal_outcome"],
        _verdicts(convergence=LAYER_AMBIGUOUS)["terminal_outcome"],
        _verdicts(causal=TRANSFER_NOT_SUPPORTED)["terminal_outcome"],
        _verdicts(convergence=REFUSED_INVALID)["terminal_outcome"],
    }
    assert reached == set(TERMINAL_OUTCOMES)


def test_the_principal_claim_names_the_same_layer_and_population_requirement():
    payload = _verdicts()
    joined = " ".join(payload["principal_claim_requires"])
    assert "one and the same physical layer" in joined
    assert "same independent multimodal population" in joined


def test_r_lens_arm_is_reported_as_r_space_and_never_raw_j_space():
    payload = _verdicts(representation_space="R_SPACE")
    assert payload["representation_space"] == "R_SPACE"
    assert payload["arms_pooled"] is False
    assert "confirmed R-lens" in payload["statement"]
    assert "R-space residual steering" in payload["coordinate_swap_scope"]


# ------------------------------------------------------------- lens integrity


def _integrity(**overrides):
    payload = {
        "layer": 29,
        "scale": 250,
        "snapshot": {"checksum": "sha256:s", "n_prompts": 250, "hook_site": "block"},
        "confirmation_verdict": {"layer": 29, "passed": True, "failed_checks": []},
        "invariance": {"passed": True, "modalities": list(REQUIRED_MODALITIES)},
        "calibration_modality": "text-only",
    }
    payload.update(overrides)
    return payload


def test_lens_integrity_passes_on_a_well_formed_fit():
    assert adjacent_lens_integrity(**_integrity())["verdict"] == "PASSED"


def test_a_missing_invariance_record_is_a_refusal_not_a_default():
    record = adjacent_lens_integrity(**_integrity(invariance=None))
    assert record["verdict"] == "REFUSED"
    assert "capture_and_zero_coefficient_invariance" in record["failed_checks"]


def test_a_snapshot_at_the_wrong_scale_is_refused():
    record = adjacent_lens_integrity(
        **_integrity(snapshot={"checksum": "s", "n_prompts": 100, "hook_site": "b"})
    )
    assert "snapshot_prompt_count_matches_scale" in record["failed_checks"]


def test_a_layer_outside_the_interval_is_refused():
    record = adjacent_lens_integrity(
        **_integrity(
            layer=32,
            confirmation_verdict={"layer": 32, "passed": True, "failed_checks": []},
        )
    )
    assert "layer_is_a_predeclared_candidate" in record["failed_checks"]


def test_multimodal_calibration_is_refused():
    record = adjacent_lens_integrity(**_integrity(calibration_modality="image"))
    assert "text_only_calibration" in record["failed_checks"]


# -------------------------------------------------------------- fingerprint


def _fingerprint_fields():
    from jlens.mmpilot.preconvergence import PRECONVERGENCE_FINGERPRINT_FIELDS

    return {name: f"value-{name}" for name in PRECONVERGENCE_FINGERPRINT_FIELDS}


def test_the_fingerprint_binds_every_configuration_field():
    payload = preconvergence_fingerprint(**_fingerprint_fields())
    assert payload["fingerprint_digest"].startswith("sha256:")


def test_a_missing_fingerprint_field_refuses_rather_than_defaulting():
    fields = _fingerprint_fields()
    fields.pop("physical_layer")
    with pytest.raises(PreconvergenceRefused, match="physical_layer"):
        preconvergence_fingerprint(**fields)


def test_an_unknown_fingerprint_field_refuses_too():
    with pytest.raises(PreconvergenceRefused, match="unknown"):
        preconvergence_fingerprint(**_fingerprint_fields(), surprise=1)


@pytest.mark.parametrize(
    "field",
    [
        "physical_layer",
        "candidate_layers",
        "adjacent_gate_digest",
        "untouched_audit_checksum",
        "population_pins_checksum",
        "selected_population_digest",
        "convergence_criterion_digest",
        "candidate_leakage_audit_digest",
        "required_causal_controls",
    ],
)
def test_changing_a_load_bearing_field_moves_the_digest(field):
    base = preconvergence_fingerprint(**_fingerprint_fields())
    moved = preconvergence_fingerprint(**{**_fingerprint_fields(), field: "other"})
    assert base["fingerprint_digest"] != moved["fingerprint_digest"]


# ---------------------------------------------------- pre-download contracts


def test_every_real_branch_symbol_resolves_and_every_call_site_binds():
    contracts = preconvergence_call_contracts()
    assert len(contracts) > 50
    assert check_preconvergence_call_contracts() == []


def test_the_contracts_cover_the_two_entry_points_the_brief_names():
    names = {name for name, *_ in preconvergence_call_contracts()}
    assert "build_real_backend" in names
    assert "assert_audio_protocol" in names


def test_the_forbidden_symbols_are_not_referenced_anywhere_in_the_module():
    import inspect

    from jlens.mmpilot import preconvergence

    source = inspect.getsource(preconvergence)
    assert "load_real_bundle" not in source
    assert "import preflight" not in source
    assert "preflight import preflight" not in source


# ------------------------------------------------------------ summary/report


def test_the_report_names_every_verdict_and_the_terminal_outcome():
    verdicts = _verdicts()
    summary = build_summary(
        fingerprint={"fingerprint_digest": "sha256:f"},
        verdicts=verdicts,
        lens_verdict={"verdict": ADJACENT_LENS_GO, "selected_layer": 29},
        confirmation_manifest={"size": 256},
        untouched_audit={"untouched": True},
        source_layer_record={"disjoint": True},
        fit_record={"n_done": 250},
        convergence={"verdict": LAYER_NOT_CONVERGED},
        convergence_controls={"passed": True},
        capability={"verdict": "AUDIO_CAPABILITY_GO"},
        disjointness={"disjoint": True},
        pseudoreplication={"passed": True},
        pool={},
        exclusion={},
        population_pins={"pinned_resolution_run": "mml32res_x"},
        sample_plan={"plan_digest": "sha256:p"},
        leakage_audit={"passed": True},
        stage_plan_record=stage_plan(),
        stage_four={"runs": True},
        causal={"verdict": TRANSFER_SUPPORTED},
        causal_controls={"passed": True},
        immutability={"unchanged": True},
        cache={},
        resume={},
        mode="real",
    )
    assert summary["primary_verdict"] == PRECONVERGENCE_SUPPORTED
    assert summary["concepts_replaced_after_results"] is False
    assert summary["thresholds_changed_after_results"] is False
    assert summary["interval"]["closed"] is True
    report = render_report(summary)
    for name in VERDICT_NAMES:
        assert name in report
    assert PRECONVERGENCE_SUPPORTED in report
    assert "OUT OF SCOPE" in report


def test_a_mock_report_says_it_proves_nothing_about_gemma():
    summary = build_summary(
        fingerprint={},
        verdicts=_verdicts(),
        lens_verdict={},
        confirmation_manifest={},
        untouched_audit={},
        source_layer_record={},
        fit_record={},
        convergence=None,
        convergence_controls=None,
        capability={},
        disjointness={},
        pseudoreplication={},
        pool={},
        exclusion={},
        population_pins={},
        sample_plan={},
        leakage_audit={},
        stage_plan_record=stage_plan(),
        stage_four={},
        causal=None,
        causal_controls=None,
        immutability={},
        cache={},
        resume={},
        mode="mock",
    )
    assert summary["mock_proves_pipeline_only"] is True
    assert "nothing about" in render_report(summary)


def test_the_module_never_calls_steering_a_coordinate_swap():
    assert "swap" in COORDINATE_SWAP_SCOPE
    assert "OUT OF SCOPE" in COORDINATE_SWAP_SCOPE
    assert "steering" in COORDINATE_SWAP_SCOPE
