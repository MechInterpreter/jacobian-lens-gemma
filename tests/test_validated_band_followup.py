"""The L33-L40 validated-band causal follow-up.

Numbered against the requirements the study was commissioned under. The theme
running through all of them: the completed corrected validation is immutable
selection evidence, L32 is out categorically, and every wrapper here re-labels
an existing computation rather than replacing one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jlens.mmpilot.band_swap import (
    BandDesignRefused,
    BandSwapThresholds,
    band_reasoning_verdict,
    band_trial_record,
    build_band,
    predeclare_suffix_bands,
    summarize_band_cells,
)
from jlens.mmpilot.coordinate_swap import LayerBandError
from jlens.mmpilot.store import RunFingerprint, UnitStore, payload_checksum
from jlens.mmpilot.validated_band_followup import (
    EXCLUDED_FAILED_LAYER,
    EXPECTED_ARTIFACT_CHECKSUMS,
    FOLLOWUP_BAND_END,
    FOLLOWUP_CAPABILITY_NO_GO,
    FOLLOWUP_INTERVENTION_FAMILY,
    FOLLOWUP_PRIMARY_BAND,
    FOLLOWUP_REPORT_NAME,
    FOLLOWUP_STUDY_NAME,
    FOLLOWUP_SUFFIX_STARTS,
    ORIGINAL_VERDICT,
    REPORTING_BOUNDARY,
    FollowupRefused,
    assert_band_hook_integrity,
    assert_corrected_run_unmodified,
    assert_followup_band,
    assert_no_fitting_entry_point,
    corrected_run_digest,
    discover_corrected_band_lenses,
    followup_design_record,
    followup_fingerprint,
    followup_onset_timing,
    followup_pass_budget,
    followup_preflight_record,
    followup_report,
    followup_verdict,
    read_corrected_validation_report,
    read_followup_units,
)
from jlens.mmpilot.validated_band_followup_mock import (
    CAUSAL_SCENARIOS,
    MOCK_DIRECTED_PAIRS,
    MOCK_MODALITIES,
    PREFLIGHT_SCENARIOS,
    mock_band_keys,
    mock_corrected_run,
    mock_followup_records,
    mock_swap_result,
)

THRESHOLDS = BandSwapThresholds(min_images=4)


# ------------------------------------------------------------------ helpers


def _admit(pins, *, require_real_mode=False):
    """The whole preflight chain over a fixture, with its own pins."""
    path, report = read_corrected_validation_report(
        pins["run_dir"],
        expected_report_checksum=pins["expected_report_checksum"],
        expected_protocol_digest=pins["expected_protocol_digest"],
        expected_universe_checksum=pins["expected_universe_checksum"],
        expected_confirmation_manifest_checksum=(
            pins["expected_confirmation_manifest_checksum"]
        ),
        expected_model_repo_id=pins["expected_model_repo_id"],
        expected_model_revision=pins["expected_model_revision"],
        require_real_mode=require_real_mode,
    )
    admission = assert_followup_band(report)
    artifacts, discovery = discover_corrected_band_lenses(
        pins["run_dir"],
        report=report,
        expected_checksums=pins["expected_artifact_checksums"],
    )
    return path, report, admission, artifacts, discovery


def _frozen_design(admission=None, discovery=None):
    primary = build_band(
        FOLLOWUP_PRIMARY_BAND[0], FOLLOWUP_BAND_END,
        usable_layers=FOLLOWUP_PRIMARY_BAND, n_layers=42,
    )
    suffixes = predeclare_suffix_bands(
        starts=FOLLOWUP_SUFFIX_STARTS, end=FOLLOWUP_BAND_END,
        usable_layers=FOLLOWUP_PRIMARY_BAND, n_layers=42,
    )
    return primary, suffixes, followup_design_record(
        primary_band=primary,
        suffix_bands=suffixes,
        admission=admission or {},
        discovery=discovery or {"lens_checksums": {}},
    )


# ---- 1. the original corrected report remains immutable and NO-GO


def test_the_completed_corrected_run_is_read_only_and_proved_unchanged(tmp_path):
    """Requirement 1 and preflight clause 10."""
    pins = mock_corrected_run(tmp_path)
    before = corrected_run_digest(pins["run_dir"])
    _, report, admission, _, _ = _admit(pins)
    after = corrected_run_digest(pins["run_dir"])

    proof = assert_corrected_run_unmodified(before, after)
    assert proof["identical"] is True
    assert proof["read_only"] is True
    assert proof["added"] == [] and proof["removed"] == [] and proof["changed"] == []

    assert report["band_verdict"]["verdict"] == ORIGINAL_VERDICT
    assert admission["original_verdict"] == ORIGINAL_VERDICT
    assert admission["original_stage3_remained_blocked"] is True


def test_a_modified_corrected_run_is_refused(tmp_path):
    pins = mock_corrected_run(tmp_path)
    before = corrected_run_digest(pins["run_dir"])
    published = next(Path(pins["run_dir"]).rglob("lens.corrected.layer35*.pt"))
    published.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(FollowupRefused, match="changed while the follow-up read it"):
        assert_corrected_run_unmodified(
            before, corrected_run_digest(pins["run_dir"])
        )


def test_an_added_file_in_the_completed_run_is_refused(tmp_path):
    pins = mock_corrected_run(tmp_path)
    before = corrected_run_digest(pins["run_dir"])
    (Path(pins["run_dir"]) / "stray.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FollowupRefused):
        assert_corrected_run_unmodified(
            before, corrected_run_digest(pins["run_dir"])
        )


def test_a_report_that_does_not_match_its_own_checksum_is_refused(tmp_path):
    pins = mock_corrected_run(tmp_path)
    path = Path(pins["report_path"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["band_verdict"]["layers_failing"] = []
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(FollowupRefused, match="does not match its own report_checksum"):
        _admit(pins)


# ---- 2 and 3. only exactly L33-L40 admits, and L32 never enters


def test_only_exactly_l33_l40_can_admit_the_followup(tmp_path):
    """Requirement 2."""
    admitted = mock_corrected_run(tmp_path / "ok", scenario="admits_followup")
    _, _, admission, _, _ = _admit(admitted)
    assert admission["followup_band"] == list(FOLLOWUP_PRIMARY_BAND)

    for scenario in ("l32_passes", "different_band"):
        pins = mock_corrected_run(tmp_path / scenario, scenario=scenario)
        with pytest.raises(FollowupRefused, match="not exactly"):
            _admit(pins)


def test_l32_can_never_enter_the_new_band(tmp_path):
    """Requirement 3. Four independent refusals, none of them a threshold."""
    # the module constants
    assert EXCLUDED_FAILED_LAYER == 32
    assert 32 not in FOLLOWUP_PRIMARY_BAND
    assert 32 not in FOLLOWUP_SUFFIX_STARTS

    # the lens gate: no validated artifact exists for it
    with pytest.raises(LayerBandError):
        build_band(32, FOLLOWUP_BAND_END, usable_layers=FOLLOWUP_PRIMARY_BAND)

    # the design record, even when a caller hands over a band containing it
    wide = build_band(32, FOLLOWUP_BAND_END, usable_layers=(32, *FOLLOWUP_PRIMARY_BAND))
    with pytest.raises(FollowupRefused, match="not the validated"):
        followup_design_record(
            primary_band=wide, suffix_bands=(), admission={}, discovery={}
        )
    primary, _, _ = _frozen_design()
    with pytest.raises(FollowupRefused, match="contains L32"):
        followup_design_record(
            primary_band=primary,
            suffix_bands=(wide,),
            admission={},
            discovery={},
            suffix_starts=FOLLOWUP_SUFFIX_STARTS,
        )

    # the discovery, categorically and before anything is read
    pins = mock_corrected_run(tmp_path)
    with pytest.raises(FollowupRefused, match="was requested for the follow-up band"):
        discover_corrected_band_lenses(
            pins["run_dir"],
            report={},
            layers=(32, *FOLLOWUP_PRIMARY_BAND),
            expected_checksums={32: "sha256:x", **pins["expected_artifact_checksums"]},
        )

    # a report that publishes it anyway
    published = mock_corrected_run(tmp_path / "pub", scenario="l32_published")
    with pytest.raises(FollowupRefused, match="publishes layer 32"):
        _admit(published)


# ---- 4, 5, 6. artifact checksums, sidecars, and every mixed-provenance refusal


def test_all_eight_artifact_checksums_are_revalidated(tmp_path):
    """Requirement 4."""
    pins = mock_corrected_run(tmp_path)
    _, _, _, artifacts, discovery = _admit(pins)
    assert sorted(artifacts) == list(FOLLOWUP_PRIMARY_BAND)
    assert len(artifacts) == 8
    for layer, source in artifacts.items():
        assert source.lens_checksum == pins["expected_artifact_checksums"][layer]
        # the checksum is the file's, recomputed, not the sidecar's claim
        from jlens.metadata import file_sha256

        assert source.lens_checksum == file_sha256(source.lens_path)
        assert source.scale == 250
    assert len(discovery["single_scale"]) == 1
    assert discovery["text_only_calibration"] is True
    assert discovery["independently_confirmed_on_one_population"] is True


def test_the_pinned_checksums_are_the_eight_published_ones():
    """Requirement 4. The pins in the module are the ones the study was given."""
    assert sorted(EXPECTED_ARTIFACT_CHECKSUMS) == list(FOLLOWUP_PRIMARY_BAND)
    assert EXPECTED_ARTIFACT_CHECKSUMS[33].endswith("41f3")
    assert EXPECTED_ARTIFACT_CHECKSUMS[40].endswith("3816")
    assert len(set(EXPECTED_ARTIFACT_CHECKSUMS.values())) == 8
    assert 32 not in EXPECTED_ARTIFACT_CHECKSUMS


@pytest.mark.parametrize(
    "scenario",
    ["missing_sidecar", "duplicate_sidecar", "checksum_mismatch", "unpublished_layer"],
)
def test_missing_or_mismatched_sidecars_and_matrices_refuse(tmp_path, scenario):
    """Requirement 5."""
    pins = mock_corrected_run(tmp_path, scenario=scenario)
    with pytest.raises(FollowupRefused):
        _admit(pins)


@pytest.mark.parametrize(
    "scenario",
    [
        "mixed_scale",
        "mixed_fit_prefix",
        "mixed_confirmation_population",
        "mixed_capture_geometry",
        "mixed_estimator",
        "mixed_hook_convention",
        "multimodal_calibration",
    ],
)
def test_mixed_provenance_refuses(tmp_path, scenario):
    """Requirement 6."""
    pins = mock_corrected_run(tmp_path, scenario=scenario)
    with pytest.raises(FollowupRefused):
        _admit(pins)


def test_a_mismatched_model_identity_refuses(tmp_path):
    """Requirement 6, the model half."""
    pins = mock_corrected_run(tmp_path)
    with pytest.raises(FollowupRefused, match="model_revision"):
        read_corrected_validation_report(
            pins["run_dir"],
            expected_report_checksum=pins["expected_report_checksum"],
            expected_protocol_digest=pins["expected_protocol_digest"],
            expected_universe_checksum=pins["expected_universe_checksum"],
            expected_confirmation_manifest_checksum=(
                pins["expected_confirmation_manifest_checksum"]
            ),
            expected_model_repo_id=pins["expected_model_repo_id"],
            expected_model_revision="some-other-revision",
            require_real_mode=False,
        )


def test_a_mock_mode_report_selects_nothing(tmp_path):
    pins = mock_corrected_run(tmp_path)
    with pytest.raises(FollowupRefused, match="a MOCK report selects no band"):
        _admit(pins, require_real_mode=True)


def test_every_commissioned_preflight_case_behaves_as_required(tmp_path):
    for key, scenario in PREFLIGHT_SCENARIOS.items():
        pins = mock_corrected_run(tmp_path / key, scenario=key)
        real_mode = key == "mock_mode_report"
        if scenario.must_be_refused:
            with pytest.raises(FollowupRefused):
                _admit(pins, require_real_mode=real_mode)
        else:
            _admit(pins, require_real_mode=real_mode)


# ---- 7, 8, 9, 10. the bands


def test_the_four_bands_are_exactly_the_predeclared_suffixes():
    """Requirement 7."""
    _, suffixes, design = _frozen_design()
    assert [list(band.layers) for band in suffixes] == [
        list(range(33, 41)),
        list(range(35, 41)),
        list(range(38, 41)),
        [40],
    ]
    assert design["band_keys"] == ["33-40", "35-40", "38-40", "40-40"]
    assert design["predeclared_suffix_starts"] == [33, 35, 38, 40]
    assert design["band_end_layer"] == 40


def test_every_band_is_physically_contiguous():
    """Requirement 8."""
    _, suffixes, _ = _frozen_design()
    for band in suffixes:
        layers = list(band.layers)
        assert layers == list(range(layers[0], layers[-1] + 1))
        assert band.start == layers[0] and band.end == layers[-1]


def test_the_sparse_start_list_is_never_mislabeled_as_the_patched_layers():
    """Requirement 9."""
    _, _, design = _frozen_design()
    note = design["sampled_start_list_is_not_the_patched_layers"]
    assert "band STARTS" in note
    assert "[33, 34, 35, 36, 37, 38, 39, 40]" in note
    assert design["band_start_layers"] != design["primary_band"]["layers"]
    # And the algebra refuses the start list described as a band.
    from jlens.mmpilot.band_swap import assert_contiguous

    with pytest.raises(BandDesignRefused, match="not contiguous"):
        assert_contiguous(FOLLOWUP_SUFFIX_STARTS, what="the band START list")
    # The starts are a sampled grid; the band they name is the full range.
    assert list(design["primary_band"]["layers"]) == list(range(33, 41))


def test_the_design_refuses_a_changed_comparison_topology():
    """Requirement 7's other half: the topology is frozen, not adjustable."""
    primary, suffixes, _ = _frozen_design()
    with pytest.raises(FollowupRefused, match="not the predeclared"):
        followup_design_record(
            primary_band=primary,
            suffix_bands=suffixes,
            admission={},
            discovery={},
            suffix_starts=(33, 34, 35, 36, 37, 38, 39, 40),
        )
    extra = predeclare_suffix_bands(
        starts=(33, 34, 35, 38, 40), end=FOLLOWUP_BAND_END,
        usable_layers=FOLLOWUP_PRIMARY_BAND,
    )
    with pytest.raises(FollowupRefused, match="start at"):
        followup_design_record(
            primary_band=primary, suffix_bands=extra, admission={}, discovery={}
        )


def test_every_band_layer_receives_its_own_coordinate_basis():
    """Requirement 10.

    The fingerprint refuses a band layer with no recorded lens checksum, so a
    patched layer that cannot name the validated artifact its coordinates came
    from is not storable at all.
    """
    _, _, design = _frozen_design()
    checksums = {layer: f"sha256:{layer}" for layer in FOLLOWUP_PRIMARY_BAND}
    fingerprint = followup_fingerprint(
        design=design,
        preflight={"corrected_report_checksum": "sha256:report"},
        lens_checksums=checksums,
        model_repo_id="google/gemma-4-E4B-it",
        model_revision="rev",
        processor_revision="prev",
        transformers_version="5.13.1",
        audio_protocol_fingerprint="sha256:audio",
        prompt_protocol=[{"readout": "property"}],
        candidate_token_ids={"property": {"two": [1], "four": [2]}},
        directed_pairs=list(MOCK_DIRECTED_PAIRS),
        population={"population_digest": "sha256:pop"},
        exclusion={"exclusion_digest": "sha256:excl"},
        thresholds=THRESHOLDS.to_dict(),
        seeds={"selection_seed": "seed"},
        readout_arms=("identity", "property"),
        scoring_rule="top-1 of the scored candidates",
    )
    assert sorted(int(k) for k in fingerprint["artifact_checksums"]) == list(
        FOLLOWUP_PRIMARY_BAND
    )
    incomplete = dict(checksums)
    incomplete.pop(36)
    with pytest.raises(FollowupRefused, match=r"no lens checksum recorded.*\[36\]"):
        followup_fingerprint(
            design=design,
            preflight={},
            lens_checksums=incomplete,
            model_repo_id="m", model_revision="r", processor_revision="p",
            transformers_version="5.13.1", audio_protocol_fingerprint=None,
            prompt_protocol=None, candidate_token_ids={},
            directed_pairs=[], population={}, exclusion={},
            thresholds={}, seeds={}, readout_arms=(),
            scoring_rule="s",
        )


# ---- 11, 12, 13. the hooks


def _result(band, *, prompt_len=12, **overrides):
    result = mock_swap_result(
        band=band, alpha=1.0, prediction="four", clean_prediction="two",
        target_answer="four", candidates=("two", "four"), prompt_len=prompt_len,
    )
    result.update(overrides)
    return result


def test_all_requested_hooks_fire():
    """Requirement 11."""
    band = tuple(range(33, 41))
    record = assert_band_hook_integrity(
        _result(band), band=band, prompt_len=12, expected_forward_passes=2
    )
    assert record["n_hooks"] == 8
    assert record["every_requested_hook_fired"] is True
    assert record["forward_passes_per_hook"] == 2

    missing = _result(band)
    missing["layer_stats"].pop("36")
    with pytest.raises(FollowupRefused, match=r"no hook record for band layer\(s\) \[36\]"):
        assert_band_hook_integrity(missing, band=band, prompt_len=12)

    extra = _result(band)
    extra["layer_stats"]["32"] = dict(extra["layer_stats"]["33"], layer=32)
    with pytest.raises(FollowupRefused, match=r"outside the band"):
        assert_band_hook_integrity(extra, band=band, prompt_len=12)

    uneven = _result(band)
    uneven["layer_stats"]["37"]["n_forward_passes"] = 3
    with pytest.raises(FollowupRefused, match="unequal numbers of times"):
        assert_band_hook_integrity(uneven, band=band, prompt_len=12)

    silent = _result(band)
    silent["layer_stats"]["39"]["n_forward_passes"] = 0
    with pytest.raises(FollowupRefused, match="never fired"):
        assert_band_hook_integrity(silent, band=band, prompt_len=12)

    wrong_count = _result(band)
    with pytest.raises(FollowupRefused, match="one per scored candidate pass"):
        assert_band_hook_integrity(
            wrong_count, band=band, prompt_len=12, expected_forward_passes=3
        )


def test_every_original_prompt_position_is_patched():
    """Requirement 12."""
    band = tuple(range(38, 41))
    record = assert_band_hook_integrity(_result(band, prompt_len=9), band=band, prompt_len=9)
    assert record["n_positions_patched"] == 9
    assert record["every_original_prompt_position_patched"] is True

    partial = _result(band, prompt_len=9)
    partial["layer_stats"]["39"]["positions"] = [0, 1, 2]
    with pytest.raises(FollowupRefused, match="rather than every original"):
        assert_band_hook_integrity(partial, band=band, prompt_len=9)


def test_candidate_token_positions_are_never_patched():
    """Requirement 13."""
    band = tuple(range(40, 41))
    leaked = _result(band, prompt_len=6)
    leaked["layer_stats"]["40"]["positions"] = list(range(8))
    with pytest.raises(FollowupRefused) as error:
        assert_band_hook_integrity(leaked, band=band, prompt_len=6)
    assert "at or beyond" in str(error.value)
    assert "teacher-forced" in str(error.value)

    clean = assert_band_hook_integrity(_result(band, prompt_len=6), band=band, prompt_len=6)
    assert clean["candidate_positions_patched"] == 0
    assert clean["no_candidate_position_patched"] is True


def test_the_stored_record_refuses_a_trial_that_patched_the_wrong_band():
    """Requirement 11, at the record layer, using the real record builder."""
    band = tuple(range(35, 41))
    result = _result(band)
    result["layers_patched"] = [35, 36, 37]
    with pytest.raises(BandDesignRefused, match="hooks fired at"):
        band_trial_record(
            result, band=band, arm="intermediate", condition="swap_alpha1",
            modality="text", source="bird", target="cat", source_answer="two",
            target_answer="four", readout="property", group_id="g", image_id="i",
        )


# ---- 14. the transcript never reaches model inputs


def test_the_transcript_never_reaches_model_inputs():
    """Requirement 14. The guard is the existing one, exercised here."""
    from jlens.mmpilot.prompt_protocol import PromptProtocolError, backend_input_kwargs

    class _Built:
        model_visible_prompt = "A photograph of an animal. How many legs does it have?"
        modality = "spoken_audio"
        media_input = object()
        sampling_rate = 16000
        media_reference = "/drive/audio/x.wav"

    kwargs = backend_input_kwargs(_Built(), transcript="a small brown dog on grass")
    assert "transcript" not in kwargs
    assert all("brown dog" not in str(value) for value in kwargs.values())

    class _Leaky(_Built):
        model_visible_prompt = "The recording says a small brown dog on grass."

    with pytest.raises(PromptProtocolError, match="reached the backend arguments"):
        backend_input_kwargs(_Leaky(), transcript="a small brown dog on grass")


# ---- 15 and 16. alphas and arms


def test_alpha1_and_alpha2_controls_are_intensity_matched_separately():
    """Requirement 15."""
    from jlens.mmpilot.band_swap import controls_for_condition

    assert controls_for_condition("swap_alpha1") == (
        "zero", "random_alpha1", "unrelated_alpha1"
    )
    assert controls_for_condition("swap_alpha2") == (
        "zero", "random_alpha2", "unrelated_alpha2"
    )

    # And the judged cells actually compare like with like.
    records = mock_followup_records("favorable")
    cells = summarize_band_cells(records, thresholds=THRESHOLDS)
    reasoning = band_reasoning_verdict(
        cells, bands=mock_band_keys(), directed_pairs=MOCK_DIRECTED_PAIRS,
        modalities=MOCK_MODALITIES, thresholds=THRESHOLDS,
    )
    for row in reasoning["direction_cells"]:
        for control, comparison in row["control_comparisons"].items():
            if comparison["alpha_matched"] is None or control == "zero":
                continue
            assert comparison["alpha_matched"] == row_alpha(row["condition"]), (
                row["condition"], control, comparison
            )
    assert reasoning["alpha2_sensitivity"]["controls_are_alpha2_matched"] is True


def row_alpha(condition: str) -> float:
    from jlens.mmpilot.band_swap import CONDITION_ALPHA

    return CONDITION_ALPHA[condition]


def test_alpha2_is_never_promoted_to_primary_evidence():
    """Requirement 15's scientific half."""
    records = mock_followup_records("alpha2_only")
    cells = summarize_band_cells(records, thresholds=THRESHOLDS)
    reasoning = band_reasoning_verdict(
        cells, bands=mock_band_keys(), directed_pairs=MOCK_DIRECTED_PAIRS,
        modalities=MOCK_MODALITIES, thresholds=THRESHOLDS,
    )
    verdict = followup_verdict(reasoning, capability_sufficient=True)
    assert verdict["verdict"] == f"{FOLLOWUP_STUDY_NAME}_ALPHA2_SENSITIVITY_ONLY"
    assert verdict["paper_comparable"]["passed"] is False
    assert verdict["alpha2_sensitivity"]["passing_bands"]


def test_intermediate_and_answer_arms_remain_distinct():
    """Requirement 16."""
    from jlens.mmpilot.band_swap import BAND_ARMS

    assert BAND_ARMS == ("intermediate", "answer")
    _, _, design = _frozen_design()
    assert design["arms"] == ["intermediate", "answer"]

    records = mock_followup_records("asymmetric_direction")
    arms = {row["arm"] for row in records}
    assert arms == {"intermediate", "answer"}
    cells = summarize_band_cells(records, thresholds=THRESHOLDS)
    keys = {(row["arm"], row["band_key"], row["condition"]) for row in cells}
    assert ("intermediate", "33-40", "swap_alpha1") in keys
    assert ("answer", "33-40", "swap_alpha1") in keys

    reasoning = band_reasoning_verdict(
        cells, bands=mock_band_keys(), directed_pairs=MOCK_DIRECTED_PAIRS,
        modalities=MOCK_MODALITIES, thresholds=THRESHOLDS,
    )
    timing = followup_onset_timing(
        reasoning, bands=mock_band_keys(), directed_pairs=MOCK_DIRECTED_PAIRS,
        modalities=MOCK_MODALITIES,
    )
    forward = next(r for r in timing["per_direction"] if r["pair"] == "bird->cat")
    assert forward["intermediate_deepest_effective_start"] == 35
    assert forward["answer_deepest_effective_start"] == 40
    assert forward["licensed_separation"] is True
    # each direction is reported before the pooled summary
    assert all(
        row["reported_before_any_pooled_summary"] for row in timing["per_direction"]
    )
    assert timing["pooled_summary"]["reported_after_each_direction"] is True
    assert timing["monotonicity_not_asserted_from_nesting"].startswith(
        "the suffix bands are nested"
    )


def test_a_reverse_direction_null_is_not_hidden_by_the_forward_one():
    """Requirement 16's direction-matching half."""
    records = mock_followup_records("asymmetric_direction")
    cells = summarize_band_cells(records, thresholds=THRESHOLDS)
    reasoning = band_reasoning_verdict(
        cells, bands=mock_band_keys(), directed_pairs=MOCK_DIRECTED_PAIRS,
        modalities=MOCK_MODALITIES, thresholds=THRESHOLDS,
    )
    timing = followup_onset_timing(
        reasoning, bands=mock_band_keys(), directed_pairs=MOCK_DIRECTED_PAIRS,
        modalities=MOCK_MODALITIES,
    )
    reverse = next(r for r in timing["per_direction"] if r["pair"] == "cat->bird")
    assert reverse["intermediate_deepest_effective_start"] is None
    assert reverse["licensed_separation"] is False
    assert len(timing["per_direction"]) == 2


# ---- 17 and 18. the population


def test_the_fresh_population_is_image_disjoint_from_prior_causal_runs():
    """Requirement 17, over the existing selection helper.

    The exclusion is enforced *before* selection: the manifest is filtered, and
    then the population digest is computed over what remains. A photograph a
    completed causal run screened can therefore not be chosen at all.
    """
    from jlens.mmpilot.evidence import EvidenceConfig
    from jlens.mmpilot.paper_reasoning_swap import hidden_animal_population

    groups = [
        {
            "group_id": f"g{i}",
            "image_id": f"img{i}",
            "caption": "a small creature resting on the grass outdoors",
            "image_path": f"/drive/img{i}.jpg",
            "audio_path": f"/drive/img{i}.wav",
            "concept_annotations": ["bird" if i % 2 else "cat"],
        }
        for i in range(24)
    ]
    config = EvidenceConfig(
        lexicon={n: (n,) for n in ("bird", "cat")},
        coco_categories={n: (n,) for n in ("bird", "cat")},
        require_visual_evidence=True,
        require_caption_evidence=False,
    )
    seed = "l33-l40-validated-band-followup-gemma-v1"

    # Without the exclusion, the spent photographs are selectable.
    unfiltered = hidden_animal_population(
        groups, concept_names=("bird", "cat"), evidence_config=config,
        images_per_concept=4, seed=seed,
    )
    spent_images = {f"img{i}" for i in range(12)}
    assert {str(row["image_id"]) for row in unfiltered["groups"]} & spent_images

    # The exclusion is applied to the manifest before selection, exactly as the
    # notebook applies it, so a spent photograph cannot be chosen at all.
    eligible = [row for row in groups if row["image_id"] not in spent_images]
    population = hidden_animal_population(
        eligible, concept_names=("bird", "cat"), evidence_config=config,
        images_per_concept=4, seed=seed,
    )
    chosen = {str(row["image_id"]) for row in population["groups"]}
    assert len(chosen) == 8
    assert not chosen & spent_images
    assert population["population_digest"] != unfiltered["population_digest"]


def test_one_photograph_is_the_independent_unit():
    """Requirement 18.

    Two records for one photograph in one cell is pseudoreplication, and the
    existing aggregation refuses it rather than averaging over it.
    """
    records = mock_followup_records("favorable")
    cells = summarize_band_cells(records, thresholds=THRESHOLDS)
    for cell in cells:
        assert cell["n_images"] == len(set(cell["image_ids"]))
        assert cell["n_images"] >= THRESHOLDS.min_images

    duplicated = [records[0], dict(records[0])]
    with pytest.raises(BandDesignRefused, match="pseudoreplicate"):
        summarize_band_cells(duplicated, thresholds=THRESHOLDS)


# ---- 19 and 20. resume


def _fingerprint(**overrides) -> RunFingerprint:
    base = {
        "mode": FOLLOWUP_INTERVENTION_FAMILY,
        "model_repo_id": "google/gemma-4-E4B-it",
        "model_revision": "rev",
        "processor_revision": "prev",
        "layers": tuple(FOLLOWUP_PRIMARY_BAND),
        "lens_checksum": "sha256:followup-digest",
        "manifest_checksum": "sha256:manifest",
        "split_id": "l33-l40-validated-band-followup-gemma-v1",
        "intervention_config": {"band": list(FOLLOWUP_PRIMARY_BAND), "alpha": [1.0, 2.0]},
        "extra": {"study": FOLLOWUP_STUDY_NAME},
    }
    base.update(overrides)
    return RunFingerprint(**base)


def test_interrupted_execution_resumes_without_recomputing_completed_units(tmp_path):
    """Requirement 19."""
    fingerprint = _fingerprint()
    store = UnitStore(tmp_path / "run", fingerprint)
    assert store.open() == "starting"
    for index in range(5):
        store.save("intervention", f"trial-{index}", {"status": "complete", "i": index})

    # A "crash": a new store object over the same directory, same fingerprint.
    resumed = UnitStore(tmp_path / "run", _fingerprint())
    assert resumed.open() == "resuming"
    recomputed = [i for i in range(5) if not resumed.has("intervention", f"trial-{i}")]
    assert recomputed == []
    resumed.save("intervention", "trial-5", {"status": "complete", "i": 5})
    assert len(resumed.load_all("intervention")) == 6


def test_a_torn_unit_costs_at_most_the_trial_being_executed(tmp_path):
    """Requirement 19's disconnect clause."""
    store = UnitStore(tmp_path / "run", _fingerprint())
    store.open()
    for index in range(3):
        store.save("intervention", f"trial-{index}", {"status": "complete", "i": index})
    torn = store.unit_path("intervention", "trial-2")
    torn.write_text(torn.read_text(encoding="utf-8")[:40], encoding="utf-8")

    assert store.has("intervention", "trial-0")
    assert store.has("intervention", "trial-1")
    assert not store.has("intervention", "trial-2")
    assert len(store.load_all("intervention")) == 2


@pytest.mark.parametrize(
    "change",
    [
        {"layers": (34, 35, 36, 37, 38, 39, 40)},
        {"lens_checksum": "sha256:a-different-artifact"},
        {"manifest_checksum": "sha256:a-different-population"},
        {"split_id": "a-different-selection"},
        {"model_revision": "a-different-model"},
        {"processor_revision": "a-different-processor"},
        {"intervention_config": {"band": list(FOLLOWUP_PRIMARY_BAND), "alpha": [1.0]}},
        {"extra": {"study": "something-else"}},
    ],
)
def test_any_scientifically_relevant_change_refuses_stale_units(tmp_path, change):
    """Requirement 20."""
    from jlens.mmpilot.store import IncompatibleStateError

    store = UnitStore(tmp_path / "run", _fingerprint())
    store.open()
    store.save("intervention", "trial-0", {"status": "complete"})

    changed = UnitStore(tmp_path / "run", _fingerprint(**change))
    with pytest.raises(IncompatibleStateError):
        changed.open()


def test_a_unit_bound_to_another_fingerprint_is_never_reused(tmp_path):
    """Requirement 20, per unit rather than per directory."""
    store = UnitStore(tmp_path / "run", _fingerprint())
    store.open()
    path = store.unit_path("intervention", "trial-0")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"status": "complete"}
    path.write_text(
        json.dumps(
            {
                "schema": "jlens.mmpilot.unit.v1",
                "stage": "intervention",
                "key": "trial-0",
                "fingerprint_digest": "sha256:some-other-run",
                "unit_checksum": payload_checksum(payload),
                "payload": payload,
            }
        ),
        encoding="utf-8",
    )
    assert store.has("intervention", "trial-0") is False


def test_a_followup_run_is_never_readable_as_a_completed_band_run(tmp_path):
    """Requirement 20's report half.

    The completed L32-L40 study's reader opens ``anthropic_band_swap_report.json``
    and the follow-up's opens its own name. Neither can read the other's
    directory, so a follow-up result can never be aggregated as that study's.
    """
    from jlens.mmpilot.band_swap import read_band_units

    run = tmp_path / "run"
    (run / "units" / "intervention").mkdir(parents=True)
    with pytest.raises(FollowupRefused, match="not found"):
        read_followup_units(run)

    (run / FOLLOWUP_REPORT_NAME).write_text(
        json.dumps(
            {
                "schema": "jlens.mmpilot.anthropic_band_swap_report.v1",
                "study_family": "anthropic_contiguous_band_coordinate_swap",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(FollowupRefused, match="declares schema"):
        read_followup_units(run)
    with pytest.raises(BandDesignRefused):
        read_band_units(run)


# ---- 21 and 22. no model, no fitting


def test_cpu_stages_do_not_load_gemma():
    """Requirement 21.

    The preflight, the design, the budget, the aggregation and the timing are
    all reachable without importing a backend that can load a model. This test
    exercises every one of them and asserts the real backend module was never
    imported as a side effect.
    """
    import sys

    for name in ("jlens.mmpilot.real_backend",):
        sys.modules.pop(name, None)

    _, suffixes, design = _frozen_design()
    budget = followup_pass_budget(
        n_pair_concepts=2, n_modalities=3, n_readouts=2, n_candidates_per_readout=2,
        candidate_images_per_concept=24, max_analysis_images_per_cell=8,
        n_bands=len(suffixes), n_arms=2, n_conditions=7,
        band_layer_counts=[len(band.layers) for band in suffixes],
    )
    records = mock_followup_records("favorable")
    cells = summarize_band_cells(records, thresholds=THRESHOLDS)
    reasoning = band_reasoning_verdict(
        cells, bands=mock_band_keys(), directed_pairs=MOCK_DIRECTED_PAIRS,
        modalities=MOCK_MODALITIES, thresholds=THRESHOLDS,
    )
    verdict = followup_verdict(reasoning, capability_sufficient=True)
    timing = followup_onset_timing(
        reasoning, bands=mock_band_keys(), directed_pairs=MOCK_DIRECTED_PAIRS,
        modalities=MOCK_MODALITIES,
    )
    report = followup_report(
        mode="mock", preflight={}, design=design, fingerprint=None,
        population=None, exclusion=None, capability_selection=None,
        capability_sufficient=True, directed_pairs=MOCK_DIRECTED_PAIRS,
        band_keys=mock_band_keys(), thresholds=THRESHOLDS.to_dict(), cells=cells,
        reasoning=reasoning, verdict=verdict, timing=timing, budget=budget,
    )
    assert report["report_checksum"]
    assert "jlens.mmpilot.real_backend" not in sys.modules


def test_no_fitting_entry_point_is_reachable():
    """Requirement 22."""
    import jlens.mmpilot.band_swap as band_swap
    import jlens.mmpilot.coordinate_swap as coordinate_swap
    import jlens.mmpilot.validated_band_followup as followup

    audit = assert_no_fitting_entry_point(followup, band_swap, coordinate_swap)
    assert audit["no_fitting_entry_point_is_reachable"] is True
    assert audit["backward_passes"] == 0

    class _Leaky:
        __name__ = "leaky"

        @staticmethod
        def run_calibration():  # pragma: no cover - never called
            raise AssertionError

    with pytest.raises(FollowupRefused, match="fitting entry point is reachable"):
        assert_no_fitting_entry_point(_Leaky())


def test_the_module_never_imports_a_fitting_module():
    """Requirement 22, structurally."""
    source = Path(
        "jlens/mmpilot/validated_band_followup.py"
    ).read_text(encoding="utf-8")
    imports = [
        line for line in source.splitlines()
        if line.startswith(("import ", "from ")) or line.strip().startswith("import ")
    ]
    joined = "\n".join(imports)
    for forbidden in ("jlens.fitting", "jlens.calibration.run", "run_calibration"):
        assert forbidden not in joined, forbidden
    # The only mention anywhere in the module is the forbidden-symbol list the
    # audit searches for, which is the opposite of a reachable entry point.
    assert source.count("run_calibration") == 1


# ---- 23. the budget


def test_the_exact_pass_budget_is_derived_and_tested():
    """Requirement 23."""
    _, suffixes, _ = _frozen_design()
    budget = followup_pass_budget(
        n_pair_concepts=2,
        n_modalities=3,
        n_readouts=2,
        n_candidates_per_readout=2,
        candidate_images_per_concept=24,
        max_analysis_images_per_cell=8,
        n_bands=len(suffixes),
        n_arms=2,
        n_conditions=7,
        band_layer_counts=[len(band.layers) for band in suffixes],
    )
    assert budget["clean_candidate_passes"] == 576
    assert budget["intervention_candidate_passes"] == 10_752
    assert budget["total"] == 11_328
    assert budget["matches_expected_design"] is True
    assert budget["backward_passes"] == 0
    assert budget["fitting_performed"] is False
    assert budget["hooks_per_trial"] == [8, 6, 3, 1]

    # The arithmetic, factor by factor.
    clean = budget["factors"]["clean"]
    assert (
        clean["pair_concepts"]
        * clean["candidate_images_per_concept"]
        * clean["modalities"]
        * clean["readouts"]
        * clean["candidates_per_readout"]
        == 576
    )
    intervention = budget["factors"]["intervention"]
    assert intervention["analysis_cells"] == 2 * 3 * 8
    assert (
        intervention["analysis_cells"]
        * intervention["bands"]
        * intervention["arms"]
        * intervention["conditions"]
        * intervention["readouts"]
        * intervention["candidates_per_readout"]
        == 10_752
    )


def test_a_changed_factor_is_named_rather_than_absorbed():
    """Requirement 23's other half: a different design does not pass silently."""
    budget = followup_pass_budget(
        n_pair_concepts=2, n_modalities=3, n_readouts=2, n_candidates_per_readout=2,
        candidate_images_per_concept=24, max_analysis_images_per_cell=8,
        n_bands=3, n_arms=2, n_conditions=7, band_layer_counts=[8, 6, 3],
    )
    assert budget["matches_expected_design"] is False
    assert budget["intervention_candidate_passes"] == 8064
    assert budget["expected_intervention_candidate_passes"] == 10_752


# ---- 24. the commissioned causal scenarios


@pytest.mark.parametrize("key", sorted(CAUSAL_SCENARIOS))
def test_every_commissioned_causal_scenario_produces_its_bounded_verdict(key):
    """Requirement 24."""
    scenario = CAUSAL_SCENARIOS[key]
    records = mock_followup_records(scenario)
    if records:
        cells = summarize_band_cells(records, thresholds=THRESHOLDS)
        reasoning = band_reasoning_verdict(
            cells, bands=mock_band_keys(), directed_pairs=MOCK_DIRECTED_PAIRS,
            modalities=MOCK_MODALITIES, thresholds=THRESHOLDS,
        )
    else:
        reasoning = None

    verdict = followup_verdict(
        reasoning, capability_sufficient=scenario.capability_sufficient
    )
    assert verdict["verdict"] == scenario.expected_verdict

    if reasoning is None:
        assert scenario.expected_timing is None
        return
    timing = followup_onset_timing(
        reasoning, bands=mock_band_keys(), directed_pairs=MOCK_DIRECTED_PAIRS,
        modalities=MOCK_MODALITIES,
    )
    assert timing["verdict"] == scenario.expected_timing


def test_a_control_failure_is_never_reported_as_a_result():
    """Requirement 24's most important case."""
    records = mock_followup_records("control_failure")
    cells = summarize_band_cells(records, thresholds=THRESHOLDS)
    primary = next(
        row for row in cells
        if row["condition"] == "swap_alpha1" and row["arm"] == "intermediate"
        and row["readout"] == "property" and row["modality"] == "text"
        and row["band_key"] == "33-40"
    )
    # The raw rate is as high as it gets ...
    assert primary["target_top1_rate"] == 1.0
    reasoning = band_reasoning_verdict(
        cells, bands=mock_band_keys(), directed_pairs=MOCK_DIRECTED_PAIRS,
        modalities=MOCK_MODALITIES, thresholds=THRESHOLDS,
    )
    # ... and the verdict is null, because the matched controls match it.
    verdict = followup_verdict(reasoning, capability_sufficient=True)
    assert verdict["verdict"] == f"{FOLLOWUP_STUDY_NAME}_NULL"
    failing = {
        clause
        for row in reasoning["direction_cells"]
        for clause in row["failed_clauses"]
    }
    assert {"beats_random", "beats_unrelated", "beats_zero"} & failing


def test_a_capability_no_go_is_not_a_null_causal_result():
    """Requirement 24's capability case."""
    verdict = followup_verdict(None, capability_sufficient=False)
    assert verdict["verdict"] == FOLLOWUP_CAPABILITY_NO_GO
    assert verdict["is_a_null_causal_result"] is False
    assert "no causal trial was run" in verdict["why"]

    with pytest.raises(FollowupRefused, match="capability screen was insufficient"):
        followup_verdict({"verdict": "BAND_SWAP_NO_GO"}, capability_sufficient=False)
    with pytest.raises(FollowupRefused, match="no reasoning verdict was produced"):
        followup_verdict(None, capability_sufficient=True)


def test_a_session_with_no_causal_stage_is_neither_null_nor_a_capability_failure():
    """A CPU-only preflight must not print something that reads like a finding."""
    from jlens.mmpilot.validated_band_followup import FOLLOWUP_NOT_RUN

    verdict = followup_verdict(
        None, capability_sufficient=True, causal_stage_ran=False
    )
    assert verdict["verdict"] == FOLLOWUP_NOT_RUN
    assert verdict["is_a_null_causal_result"] is False
    assert verdict["is_a_capability_failure"] is False
    assert verdict["causal_stage_ran"] is False
    assert "no causal evidence of any kind" in verdict["why"]

    # The three non-result states stay distinct.
    capability = followup_verdict(None, capability_sufficient=False)
    assert capability["verdict"] != verdict["verdict"]
    assert capability["is_a_capability_failure"] is True

    with pytest.raises(FollowupRefused, match="causal stage did not run"):
        followup_verdict(
            {"verdict": "BAND_SWAP_NO_GO"},
            capability_sufficient=True,
            causal_stage_ran=False,
        )


def test_an_unknown_underlying_verdict_is_refused_rather_than_invented():
    with pytest.raises(FollowupRefused, match="unknown underlying verdict"):
        followup_verdict({"verdict": "SOMETHING_NEW"}, capability_sufficient=True)


def test_the_verdict_wrapper_relabels_and_changes_no_threshold():
    records = mock_followup_records("favorable")
    cells = summarize_band_cells(records, thresholds=THRESHOLDS)
    reasoning = band_reasoning_verdict(
        cells, bands=mock_band_keys(), directed_pairs=MOCK_DIRECTED_PAIRS,
        modalities=MOCK_MODALITIES, thresholds=THRESHOLDS,
    )
    verdict = followup_verdict(reasoning, capability_sufficient=True)
    assert verdict["relabel_only"] is True
    assert verdict["no_threshold_was_changed"] is True
    assert verdict["threshold_digest"] == reasoning["threshold_digest"]
    assert verdict["underlying_verdict_digest"] == reasoning["verdict_digest"]
    assert verdict["underlying_verdict"] == "BAND_SWAP_PAPER_COMPARABLE_GO"


# ---- the reporting boundary


def test_every_report_states_the_reporting_boundary():
    _, _, design = _frozen_design()
    verdict = followup_verdict(None, capability_sufficient=False)
    report = followup_report(
        mode="real", preflight={}, design=design, fingerprint=None, population=None,
        exclusion=None, capability_selection=None, capability_sufficient=False,
        directed_pairs=[], band_keys=[], thresholds={}, cells=[], reasoning=None,
        verdict=verdict, timing=None,
    )
    assert report["original_l32_l40_verdict"] == ORIGINAL_VERDICT
    assert report["is_a_prospective_causal_followup"] is True
    assert report["is_the_original_l32_l40_confirmatory_band"] is False
    assert report["band_selected_after_lens_validation"] is True
    assert report["original_stage3_remained_blocked"] is True
    assert report["excluded_failed_layer"] == 32
    assert report["supports_no_claim_about_a_band_beginning_at_l32"] is True
    assert report["supports_no_claim_about_layers_earlier_than_33"] is True
    assert (
        report["spokencoco_tests_linguistic_spoken_captions_not_environmental_sound"]
        is True
    )
    assert report["model_outputs_text_image_and_audio_are_input_modalities"] is True
    assert report["alpha2_is_sensitivity_not_primary_evidence"] is True
    assert report["no_lens_was_refitted"] is True
    assert report["no_threshold_was_changed"] is True
    assert report["completed_corrected_run_read_or_modified"] == "read-only"
    assert report["reporting_boundary"] == list(REPORTING_BOUNDARY)
    assert design["reporting_boundary"] == list(REPORTING_BOUNDARY)
    assert verdict["reporting_boundary"] == list(REPORTING_BOUNDARY)


def test_the_boundary_names_every_required_clause():
    joined = " ".join(REPORTING_BOUNDARY)
    for phrase in (
        "selected AFTER",
        "PROSPECTIVE CAUSAL FOLLOW-UP",
        ORIGINAL_VERDICT,
        "L32 is excluded",
        "band beginning at L32",
        "earlier than 33",
        "linguistic spoken captions, not environmental sound",
        "outputs text",
        "alpha=2 is sensitivity evidence",
    ):
        assert phrase in joined, phrase


def test_the_preflight_record_prints_the_four_required_statements(tmp_path):
    from jlens.mmpilot.validated_band_followup import format_followup_preflight

    pins = mock_corrected_run(tmp_path)
    path, report, admission, _, discovery = _admit(pins)
    before = corrected_run_digest(pins["run_dir"])
    preflight = followup_preflight_record(
        report_path=path,
        report=report,
        admission=admission,
        discovery=discovery,
        immutability=assert_corrected_run_unmodified(
            before, corrected_run_digest(pins["run_dir"])
        ),
        corrected_run_dir=pins["run_dir"],
    )
    assert preflight["original_l32_l40_verdict"] == ORIGINAL_VERDICT
    assert preflight["followup_band"] == list(FOLLOWUP_PRIMARY_BAND)
    assert preflight["selection_source"] == "text-only corrected lens validation only"
    assert preflight["no_causal_outcome_selected_the_band"] is True
    assert preflight["no_fitting_will_occur"] is True

    block = format_followup_preflight(preflight)
    assert "ORIGINAL L32-L40 VERDICT REMAINS" in block
    assert ORIGINAL_VERDICT in block
    assert "L33-L40" in block
    assert "no causal outcome selected the band" in block
    assert "no fitting will occur" in block
    for checksum in discovery["lens_checksums"].values():
        assert checksum in block
