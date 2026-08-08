# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""The layer-32 confirmatory follow-up: discovery, comparability, verdicts.

Every test here runs on CPU in milliseconds against tiny fixtures. The point is
not that the code runs — it is that each refusal *fires*, so "we would have
caught that" is a demonstrated fact rather than a claim.
"""

import json
from pathlib import Path

import pytest

from jlens.mmpilot.convergence import (
    AMBIGUOUS,
    CONVERGED,
    LENS_INVALID_LAYERS,
    NOT_CONVERGED,
)
from jlens.mmpilot.l32_followup import (
    ARTIFACT_DISCOVERY_VERSION,
    COMPLETED_STUDY_PROMPT_PROTOCOL,
    CONVERGENCE_PHRASE,
    FALLBACK_SECONDS_PER_PASS,
    FINGERPRINT_FIELDS,
    FORBIDDEN_PHRASES,
    INCONCLUSIVE,
    INTEGRITY_PASSED,
    INTEGRITY_REFUSED,
    INTERVENTION_FAMILY,
    L32_FOLLOWUP_PROTOCOL,
    L32_LAYER,
    MIN_DISTINCT_UNITS,
    NOT_SUPPORTED,
    OPEN_PROMPT_PROTOCOL,
    REFERENCE_LAYER,
    SELECTED_SCALE,
    SUPPORTED,
    TRANSFER_AT_OR_AFTER_CONVERGENCE,
    ArtifactDiscoveryRefused,
    L32FollowupRefused,
    PairedReferenceRequired,
    adjacent_layer_recommendation,
    admissible_supporting_cells,
    assert_native_head_agrees,
    assert_paired_reference_available,
    assert_report_phrasing,
    derive_seconds_per_pass,
    discover_published_l32_lens,
    followup_fingerprint,
    l32_expectations,
    lens_integrity_verdict,
    pre_convergence_verdict,
    prompt_protocol_comparability,
    separate_measured_from_historical,
    validate_discovered_lens,
)
from jlens.mmpilot.mock import (
    MOCK_D_MODEL,
    MOCK_EXTENSION_DEFECTS,
    MOCK_EXTENSION_LAYER,
    MOCK_EXTENSION_SCALE,
    build_mock_extension_run,
)
from jlens.mmpilot.published_lens import PublishedLensRefused, load_published_lenses

LAYER = MOCK_EXTENSION_LAYER
SCALE = MOCK_EXTENSION_SCALE


def _build(tmp_path, name="run", **kwargs):
    kwargs.setdefault("layer", LAYER)
    kwargs.setdefault("scale", SCALE)
    return build_mock_extension_run(
        tmp_path / name, d_model=MOCK_D_MODEL, **kwargs
    )


def _discover(root):
    return discover_published_l32_lens(root, layer=LAYER, expected_scale=SCALE)


@pytest.fixture(scope="module")
def clean_run(tmp_path_factory):
    return _build(tmp_path_factory.mktemp("l32"), "clean")


# ------------------------------------------------------------------ discovery


def test_the_artifact_is_resolved_from_metadata_not_from_a_filename(clean_run):
    found = _discover(clean_run["root"])
    evidence = found.discovery_evidence

    assert found.lens_checksum == clean_run["lens_checksum"]
    assert found.scale == SCALE
    assert evidence["discovery_version"] == ARTIFACT_DISCOVERY_VERSION
    # The chain actually ran: the report was read, the directory was scanned,
    # and exactly one sidecar matched.
    assert Path(evidence["report_path"]).name == "early_layer_extension_report.json"
    assert evidence["extension_sidecars_scanned"]
    assert len(evidence["matching_sidecars"]) == 1
    # The report and the sidecar independently name the same bytes.
    assert evidence["report_checksum_for_layer"] == found.lens_checksum
    assert evidence["file_checksum"] == found.lens_checksum


def test_the_resolved_lens_loads_and_validates(clean_run):
    found = _discover(clean_run["root"])
    record = validate_discovered_lens(found, clean_run["expectations"])

    assert record["passed"] is True
    assert record["failed_checks"] == []
    assert record["layer"] == LAYER
    assert record["confirmation_set_size_expected"] == 256

    loaded = load_published_lenses([found.spec()], clean_run["expectations"])
    assert loaded.layers == (LAYER,)
    assert loaded.n_prompts == SCALE


@pytest.mark.parametrize("defect", MOCK_EXTENSION_DEFECTS)
def test_every_broken_artifact_is_refused(tmp_path, defect):
    built = _build(tmp_path, defect, corrupt=defect)
    with pytest.raises((ArtifactDiscoveryRefused, PublishedLensRefused)):
        validate_discovered_lens(_discover(built["root"]), built["expectations"])


def test_a_wrong_checksum_names_the_disagreement(tmp_path):
    built = _build(tmp_path, "bytes", corrupt="lens_bytes")
    with pytest.raises(ArtifactDiscoveryRefused) as error:
        _discover(built["root"])
    assert "lens_checksum_agreement" in error.value.problems
    assert "file is" in str(error.value)


def test_two_sidecars_for_one_layer_are_refused_not_sorted(tmp_path):
    built = _build(tmp_path, "dup", corrupt="duplicate")
    with pytest.raises(ArtifactDiscoveryRefused) as error:
        _discover(built["root"])
    assert "extension_sidecar_unique" in error.value.problems
    assert "sort order" in str(error.value)


def test_an_unconfirmed_artifact_is_never_discovered(tmp_path):
    built = _build(tmp_path, "unconfirmed", corrupt="unconfirmed")
    with pytest.raises(ArtifactDiscoveryRefused):
        _discover(built["root"])


def test_a_mock_mode_report_publishes_nothing_about_gemma(tmp_path):
    built = _build(tmp_path, "mockmode", mode="mock")
    with pytest.raises(ArtifactDiscoveryRefused, match="MOCK report"):
        _discover(built["root"])


def test_a_no_go_verdict_is_refused(tmp_path):
    built = _build(tmp_path, "nogo", verdict="EARLY_LAYER_CALIBRATION_NO_GO")
    with pytest.raises(ArtifactDiscoveryRefused, match="early_layer_verdict"):
        _discover(built["root"])


def test_invalid_units_in_the_extension_run_refuse_its_publication(tmp_path):
    built = _build(tmp_path, "torn")
    report_path = Path(built["report_path"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["resume"]["invalid_units"] = ["units/fit/one.json"]
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ArtifactDiscoveryRefused, match="torn store"):
        _discover(built["root"])


def test_a_parent_run_that_was_not_proved_immutable_is_refused(tmp_path):
    built = _build(tmp_path, "mutable")
    report_path = Path(built["report_path"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["parent_immutability_proof"]["immutable"] = False
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ArtifactDiscoveryRefused, match="immutable"):
        _discover(built["root"])


def test_an_unknown_report_schema_is_never_guessed_at(tmp_path):
    built = _build(tmp_path, "schema")
    report_path = Path(built["report_path"])
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["schema"] = "some.other.format.v9"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ArtifactDiscoveryRefused, match="never seen that format"):
        _discover(built["root"])


# ----------------------------------------------------------- the scale binding


def test_selected_scale_250_is_bound_and_another_scale_is_refused(clean_run):
    assert SELECTED_SCALE == 250
    with pytest.raises(ArtifactDiscoveryRefused, match="requires 1000"):
        discover_published_l32_lens(
            clean_run["root"], layer=LAYER, expected_scale=1000
        )


def test_the_real_expectations_are_layer_32_at_scale_250():
    expectations = l32_expectations(
        model_repo_id="google/gemma-4-E4B-it", model_revision="f" * 40
    )
    assert expectations.scale_point == 250
    assert expectations.confirmed_layers == (L32_LAYER,)
    # At scale 250 no layer this study loads is on a failed-confirmation list;
    # the scale-100 failure is a fact about scale 100.
    assert expectations.failed_confirmation_layers == ()
    assert expectations.calibration_modality == "text-only"


def test_a_scale_100_lens_cannot_stand_in_for_the_scale_250_one(tmp_path):
    built = _build(tmp_path, "scale100", scale=100)
    found = discover_published_l32_lens(
        built["root"], layer=LAYER, expected_scale=100
    )
    # Discovered under its own scale, but refused against this study's.
    with pytest.raises(PublishedLensRefused, match="fitted_scale"):
        validate_discovered_lens(
            found,
            l32_expectations(
                model_repo_id="mock/gemma-like",
                model_revision="mockrevision0000000000000000000000000000",
                d_model=MOCK_D_MODEL,
                layer=LAYER,
                scale=SCALE,
            ),
        )


# ------------------------------------------------- read-only completed access


def _tree_state(root):
    return {
        str(path.relative_to(root)): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(Path(root).rglob("*"))
        if path.is_file()
    }


def test_discovery_never_writes_into_the_run_it_reads(clean_run):
    before = _tree_state(clean_run["root"])
    found = _discover(clean_run["root"])
    validate_discovered_lens(found, clean_run["expectations"])
    load_published_lenses([found.spec()], clean_run["expectations"])
    assert _tree_state(clean_run["root"]) == before


# --------------------------------------------------------------- comparability


def test_the_completed_candidate_listed_study_requires_a_paired_reference():
    record = prompt_protocol_comparability(
        {"selection_fingerprint": {"capability_protocol": COMPLETED_STUDY_PROMPT_PROTOCOL}}
    )
    assert record["completed_run_protocol"] == COMPLETED_STUDY_PROMPT_PROTOCOL
    assert record["protocols_match"] is False
    assert record["paired_reference_required"] is True
    assert record["reusable_without_rerun"] is False
    assert "candidates listed in the prompt" in record["statement"]


def test_a_matching_protocol_may_reuse_the_completed_results():
    record = prompt_protocol_comparability(
        {"selection_fingerprint": {"capability_protocol": OPEN_PROMPT_PROTOCOL}}
    )
    assert record["protocols_match"] is True
    assert record["paired_reference_required"] is False
    assert record["reusable_without_rerun"] is True


def test_an_unrecorded_protocol_is_treated_as_incomparable_not_as_matching():
    record = prompt_protocol_comparability({})
    assert record["completed_run_protocol_known"] is False
    assert record["paired_reference_required"] is True


def test_the_cross_layer_comparison_is_refused_without_the_reference():
    record = prompt_protocol_comparability(
        {"selection_fingerprint": {"capability_protocol": COMPLETED_STUDY_PROMPT_PROTOCOL}}
    )
    with pytest.raises(PairedReferenceRequired, match="RUN_PAIRED_L35_REFERENCE"):
        assert_paired_reference_available(record, paired_reference_ran=False)
    assert_paired_reference_available(record, paired_reference_ran=True)


def test_measured_and_historical_are_kept_apart():
    record = separate_measured_from_historical(
        measured={"layer": 32},
        historical={"layer": REFERENCE_LAYER},
        comparability=prompt_protocol_comparability(
            {"selection_fingerprint": {"capability_protocol": COMPLETED_STUDY_PROMPT_PROTOCOL}}
        ),
    )
    assert record["may_be_compared_numerically"] is False
    assert record["newly_measured"] != record["read_only_historical_context"]
    assert "not mixed" in record["note"]


# ------------------------------------------------------------------ fingerprint


def _fingerprint_fields():
    return {name: f"value-for-{name}" for name in FINGERPRINT_FIELDS}


def test_the_fingerprint_binds_every_scientific_configuration_field():
    base = followup_fingerprint(**_fingerprint_fields())
    assert base["fingerprint_digest"].startswith("sha256:")

    for name in FINGERPRINT_FIELDS:
        changed = _fingerprint_fields()
        changed[name] = "something-else"
        assert (
            followup_fingerprint(**changed)["fingerprint_digest"]
            != base["fingerprint_digest"]
        ), f"changing {name} did not change the fingerprint"


def test_a_missing_field_is_a_refusal_not_a_none():
    fields = _fingerprint_fields()
    del fields["audio_protocol_fingerprint"]
    with pytest.raises(L32FollowupRefused, match="audio_protocol_fingerprint"):
        followup_fingerprint(**fields)


def test_an_unknown_field_is_refused_too():
    with pytest.raises(L32FollowupRefused, match="unknown"):
        followup_fingerprint(**_fingerprint_fields(), invented_field="x")


def test_the_intervention_family_is_steering_and_is_bound():
    assert INTERVENTION_FAMILY == "source_derived_jspace_steering"
    assert "intervention_family" in FINGERPRINT_FIELDS
    # A run under another family cannot resume from a steering run.
    steering = followup_fingerprint(**_fingerprint_fields())
    swapped = _fingerprint_fields()
    swapped["intervention_family"] = "anthropic_two_coordinate_swap"
    assert (
        followup_fingerprint(**swapped)["fingerprint_digest"]
        != steering["fingerprint_digest"]
    )


# ---------------------------------------------------------------- the budget


def test_an_unreachable_run_gives_a_range_and_never_a_point_estimate(tmp_path):
    record = derive_seconds_per_pass(tmp_path / "not-there")
    assert record["available"] is False
    assert record["fallback_range_seconds"] == list(FALLBACK_SECONDS_PER_PASS)
    assert "median_seconds_per_unit" not in record


def test_too_few_units_gives_a_range_rather_than_a_rate(tmp_path):
    stage = tmp_path / "run" / "units" / "capability"
    stage.mkdir(parents=True)
    for index in range(5):
        (stage / f"{index}.json").write_text("{}", encoding="utf-8")
    record = derive_seconds_per_pass(tmp_path / "run")
    assert record["available"] is False
    assert "too few" in record["reason"]


def test_timing_is_derived_from_unit_mtimes_when_there_are_enough(tmp_path):
    import os

    stage = tmp_path / "run" / "units" / "capability"
    stage.mkdir(parents=True)
    base = 1_700_000_000.0
    for index in range(40):
        path = stage / f"{index}.json"
        path.write_text("{}", encoding="utf-8")
        # Two seconds per unit, plus one long gap that must be dropped as a
        # runtime disconnect rather than counted as compute.
        offset = base + index * 2.0 + (10_000.0 if index >= 30 else 0.0)
        os.utime(path, (offset, offset))
    record = derive_seconds_per_pass(tmp_path / "run", min_samples=10)
    assert record["available"] is True
    assert record["median_seconds_per_unit"] == pytest.approx(2.0)
    assert record["source"] == "unit_file_mtimes"
    assert "understates total elapsed" in record["caveat"]


# ------------------------------------------------------------------ verdict A


def _integrity_inputs(clean_run):
    found = _discover(clean_run["root"])
    return found, validate_discovered_lens(found, clean_run["expectations"])


def test_verdict_a_passes_on_a_good_artifact(clean_run):
    found, validation = _integrity_inputs(clean_run)
    verdict = lens_integrity_verdict(
        validation,
        invariance={"passed": True, "modalities": ["image", "spoken_audio", "text"]},
        discovery=found.to_dict(),
        layer=LAYER,
        scale=SCALE,
    )
    assert verdict["verdict"] == INTEGRITY_PASSED
    assert verdict["failed_checks"] == []


def test_verdict_a_refuses_a_missing_invariance_record_rather_than_defaulting(clean_run):
    found, validation = _integrity_inputs(clean_run)
    verdict = lens_integrity_verdict(
        validation, invariance=None, discovery=found.to_dict(), layer=LAYER, scale=SCALE
    )
    assert verdict["verdict"] == INTEGRITY_REFUSED
    assert "capture_and_zero_coefficient_invariance" in verdict["failed_checks"]


def test_verdict_a_refuses_the_wrong_layer(clean_run):
    found, validation = _integrity_inputs(clean_run)
    verdict = lens_integrity_verdict(
        validation,
        invariance={"passed": True},
        discovery=found.to_dict(),
        layer=LAYER + 7,
        scale=SCALE,
    )
    assert verdict["verdict"] == INTEGRITY_REFUSED
    assert "physical_layer" in verdict["failed_checks"]


# ------------------------------------------------------------ the native head


def test_a_comparison_that_never_ran_is_not_a_comparison_that_passed():
    audit = {"matches_model_unembed": None, "max_abs_difference_vs_model_unembed": None}
    with pytest.raises(L32FollowupRefused, match="never ran"):
        assert_native_head_agrees(audit, required=True)
    # MOCK records it as not run, and says so.
    record = assert_native_head_agrees(audit, required=False)
    assert record["comparison_ran"] is False
    assert record["passed"] is True


def test_a_disagreeing_head_is_refused():
    with pytest.raises(L32FollowupRefused, match="agree"):
        assert_native_head_agrees(
            {"matches_model_unembed": False, "max_abs_difference_vs_model_unembed": 0.2},
            required=True,
        )


def test_an_agreeing_head_passes():
    record = assert_native_head_agrees(
        {"matches_model_unembed": True, "max_abs_difference_vs_model_unembed": 1e-6},
        required=True,
    )
    assert record["passed"] is True
    assert record["comparison_ran"] is True


# ------------------------------------------------------------------ verdict E


def _cell(**overrides):
    cell = {
        "concept": "cat",
        "pair": "spoken_audio->image",
        "evaluated": True,
        "passes": True,
        "counted_toward_verdict": True,
        "capability_admissible": True,
        "mean_signed_target_effect": 0.40,
        "random_control": 0.05,
        "unrelated_control": 0.02,
        "mean_abs_unrelated_change": 0.01,
        "n_positive_images": 8,
        "n_negative_images": 8,
        "reasons": [],
    }
    cell.update(overrides)
    return cell


def _verdict(classification=NOT_CONVERGED, cells=None, **overrides):
    kwargs = {
        "integrity": {"verdict": INTEGRITY_PASSED},
        "causal": {"cells": cells if cells is not None else [_cell()]},
        "convergence": {"classification": classification},
        "controls": {"all_controls_passed": True, "failed_controls": []},
        "capability": {"verdict": "AUDIO_CAPABILITY_GO"},
    }
    kwargs.update(overrides)
    return pre_convergence_verdict(**kwargs)


def test_verdict_e_is_supported_when_every_clause_holds():
    verdict = _verdict()
    assert verdict["verdict"] == SUPPORTED
    assert verdict["failed_clauses"] == []
    assert verdict["n_distinct_units"] >= MIN_DISTINCT_UNITS
    assert CONVERGENCE_PHRASE in verdict["rationale"]
    assert verdict["intervention_family"] == INTERVENTION_FAMILY


def test_a_converged_layer_short_circuits_before_any_other_clause():
    # Deliberately break other clauses too: the CONVERGED branch is checked
    # first and must not be maskable.
    verdict = _verdict(
        classification=CONVERGED,
        integrity={"verdict": INTEGRITY_REFUSED},
        cells=[],
    )
    assert verdict["verdict"] == TRANSFER_AT_OR_AFTER_CONVERGENCE
    assert "checked first and cannot be masked" in verdict["rationale"]


def test_an_ambiguous_layer_can_only_be_inconclusive():
    assert _verdict(classification=AMBIGUOUS)["verdict"] == INCONCLUSIVE
    assert (
        _verdict(classification=AMBIGUOUS, cells=[])["verdict"] == INCONCLUSIVE
    )


def test_an_effect_on_one_photograph_is_not_enough():
    verdict = _verdict(cells=[_cell(n_positive_images=1, n_negative_images=8)])
    assert verdict["verdict"] == NOT_SUPPORTED
    assert verdict["n_distinct_units"] == 1
    assert any("distinct_photographs" in name for name in verdict["failed_clauses"])


def test_an_effect_that_does_not_beat_its_controls_fails():
    verdict = _verdict(cells=[_cell(random_control=0.9)])
    assert verdict["verdict"] == NOT_SUPPORTED
    assert (
        "exceeds_matched_random_and_external_unrelated" in verdict["failed_clauses"]
    )


def test_a_globally_disruptive_edit_is_not_specific():
    verdict = _verdict(cells=[_cell(mean_abs_unrelated_change=0.9)])
    assert "effect_is_specific_not_global_disruption" in verdict["failed_clauses"]


def test_an_insane_activation_norm_fails():
    verdict = _verdict(
        cells=[_cell(reasons=["activation norm ratio 4.100 outside [0.5, 2.0]"])]
    )
    assert "activation_norms_sane" in verdict["failed_clauses"]


def test_a_capability_ineligible_cell_supports_nothing():
    verdict = _verdict(cells=[_cell(counted_toward_verdict=False)])
    assert verdict["n_admissible_off_diagonal_supporting_cells"] == 0
    assert verdict["verdict"] == INCONCLUSIVE


def test_a_same_modality_cell_is_not_off_diagonal():
    assert admissible_supporting_cells({"cells": [_cell(pair="text->text")]}) == []
    assert admissible_supporting_cells({"cells": [_cell(pair="text->image")]})


def test_a_missing_control_record_fails_rather_than_defaults():
    verdict = _verdict(controls={})
    assert "convergence_controls_pass" in verdict["failed_clauses"]


def test_a_failed_capability_gate_fails_verdict_e():
    verdict = _verdict(capability={"verdict": "AUDIO_CAPABILITY_NO_GO"})
    assert "behavioral_capability_valid" in verdict["failed_clauses"]


def test_a_refused_lens_fails_verdict_e():
    verdict = _verdict(integrity={"verdict": INTEGRITY_REFUSED})
    assert "lens_integrity_passed" in verdict["failed_clauses"]


# --------------------------------------------------------------- phrasing


@pytest.mark.parametrize("phrase", FORBIDDEN_PHRASES)
def test_the_forbidden_phrases_are_refused(phrase):
    with pytest.raises(L32FollowupRefused, match="cannot support"):
        assert_report_phrasing(f"The result shows transfer {phrase} in the model.")


def test_the_required_phrase_is_accepted_and_recorded():
    record = assert_report_phrasing(f"Transfer occurs {CONVERGENCE_PHRASE}.")
    assert record["passed"] is True
    assert record["required_phrase_present"] is True


def test_the_forbidden_check_is_case_insensitive():
    with pytest.raises(L32FollowupRefused):
        assert_report_phrasing("This is PRE-LINGUISTIC representation.")


# ----------------------------------------------- the conditional next step


def test_causal_and_not_converged_needs_no_adjacent_sweep():
    record = adjacent_layer_recommendation(
        causal_verdict=SUPPORTED, classification=NOT_CONVERGED
    )
    assert record["fit_l33_l34"] is False
    assert record["recommendation"] == "no_adjacent_sweep_required"
    assert "optional" in record["rationale"]


def test_noncausal_l32_with_a_causal_matched_reference_asks_for_l33_l34():
    record = adjacent_layer_recommendation(
        causal_verdict="UNSUPPORTED",
        classification=NOT_CONVERGED,
        reference_causal=SUPPORTED,
    )
    assert record["fit_l33_l34"] is True
    assert record["recommendation"] == "fit_l33_l34"


def test_a_converged_l32_sends_the_search_earlier_not_later():
    record = adjacent_layer_recommendation(
        causal_verdict=SUPPORTED, classification=CONVERGED
    )
    assert record["fit_l33_l34"] is False
    assert record["recommendation"] == "investigate_layers_earlier_than_32"


def test_an_ambiguous_l32_asks_for_a_convergence_resolution_study():
    record = adjacent_layer_recommendation(
        causal_verdict=SUPPORTED, classification=AMBIGUOUS
    )
    assert record["fit_l33_l34"] is False
    assert record["recommendation"] == "smallest_convergence_resolution_study"


def test_noncausal_l32_without_a_reference_recommends_nothing():
    record = adjacent_layer_recommendation(
        causal_verdict="UNSUPPORTED", classification=NOT_CONVERGED
    )
    assert record["fit_l33_l34"] is False
    assert record["recommendation"] == "no_recommendation_the_premises_are_not_met"
    assert "searching, not localizing" in record["rationale"]


# ------------------------------------------------- the layer-32 lens gate


def test_layer_32_still_needs_a_passing_record_to_be_interpreted():
    """The scale-100 refusal is unchanged; only a real record clears it."""
    from jlens.mmpilot.convergence import (
        LensInvalidLayerError,
        assert_lens_valid_layer,
    )

    assert 32 in LENS_INVALID_LAYERS
    with pytest.raises(LensInvalidLayerError):
        assert_lens_valid_layer(32, audited=(32,))
    assert_lens_valid_layer(
        32, audited=(32,), confirmation_record={"layer": 32, "passed": True}
    )


def test_the_protocol_identifiers_are_pinned():
    assert L32_FOLLOWUP_PROTOCOL == "mmpilot.l32_open_prompt_followup.v1"
    assert OPEN_PROMPT_PROTOCOL == "mmpilot.open_entity_identification.v1"
    assert COMPLETED_STUDY_PROMPT_PROTOCOL == "gemma-it-chat-balanced-options-v1"
    assert L32_LAYER == 32
    assert REFERENCE_LAYER == 35
