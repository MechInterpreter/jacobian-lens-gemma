"""Guards for the multimodal follow-up studies (localization, new property, asymmetry)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jlens.mmpilot import multimodal_followup as followup
from jlens.mmpilot.multimodal_followup import (
    CONTROL_CONDITIONS,
    MODALITIES,
    REQUIRED_CONDITIONS,
    VALIDATED_BAND,
    MultimodalFollowupRefused,
    answer_key,
    artifact_exclusion_audit,
    assert_controls_complete,
    assert_design_frozen,
    assert_lens_reused_not_refitted,
    assert_open_endpoint,
    assert_property_pair_changes_answer,
    asymmetry_replication_design,
    audio_metadata_linkage_audit,
    audit_property_family,
    bands_are_nested_chain,
    development_direction_record,
    exclusion_universe,
    followup_budget,
    freeze_new_property_design,
    leg_count_property_limit,
    load_extra_spent_image_ids,
    localization_budget,
    localization_claim_boundary,
    localization_grid,
    new_property_development_verdict,
    property_answer_matches,
    recruit_all_modality_capable_groups,
    recruited_exploratory_verdict,
    resolve_dominant_answer,
    summarize_localization,
)
from jlens.mmpilot.multimodal_followup_mock import (
    MOCK_LENS_CHECKSUM,
    SCENARIOS,
    MockDisconnect,
    mock_groups,
    run_mock_asymmetry_study,
    run_mock_localization,
    run_mock_new_property_study,
)
from jlens.mmpilot.store import IncompatibleStateError, RunFingerprint, UnitStore


def test_audio_metadata_linkage_audit_is_honest_about_its_boundary(tmp_path) -> None:
    audio = tmp_path / "cow.wav"
    source = tmp_path / "metadata.json"
    audio.write_bytes(b"RIFF")
    source.write_text("{}", encoding="utf-8")
    groups = [{
        "group_id": "g-cow", "image_id": "img-cow", "caption": "a cow",
        "audio_path": str(audio), "audio_record_id": "cow",
        "caption_id": "cap-cow", "source_file": str(source),
        "source_metadata_checksum": "sha256:abc",
        "synchronization_method": "explicit_metadata_fields",
        "media_validation_status": "valid",
    }]
    rows = [{
        "concept": "cow", "group_id": "g-cow", "image_id": "img-cow",
        "modality": "spoken_audio", "generated": "meow", "pass": False,
    }]
    audit = audio_metadata_linkage_audit(groups, rows)
    assert audit["verdict"] == "AUDIO_METADATA_LINKAGE_GO"
    assert audit["metadata_linkage_verified"] is True
    assert audit["waveform_content_independently_transcribed"] is False
    assert audit["model_forwards"] == 0


def test_audio_metadata_linkage_audit_refuses_conflicting_owner(tmp_path) -> None:
    audio = tmp_path / "shared.wav"
    source = tmp_path / "metadata.json"
    audio.write_bytes(b"RIFF")
    source.write_text("{}", encoding="utf-8")
    common = {
        "audio_path": str(audio), "audio_record_id": "shared",
        "caption_id": "cap", "source_file": str(source),
        "source_metadata_checksum": "sha256:abc",
        "synchronization_method": "explicit_metadata_fields",
        "media_validation_status": "valid",
    }
    groups = [
        {**common, "group_id": "g1", "image_id": "i1", "caption": "a cow"},
        {**common, "group_id": "g2", "image_id": "i2", "caption": "another cow"},
    ]
    rows = [{
        "concept": "cow", "group_id": "g1", "image_id": "i1",
        "modality": "spoken_audio", "generated": "meow", "pass": False,
    }]
    assert audio_metadata_linkage_audit(groups, rows)["verdict"] == (
        "AUDIO_METADATA_LINKAGE_NO_GO"
    )


def test_recruited_exploratory_path_is_clean_capability_only() -> None:
    groups = [
        {"group_id": f"g-{concept}-{i}", "image_id": f"i-{concept}-{i}"}
        for concept in ("cat", "cow") for i in range(3)
    ]
    rows = [
        {"concept": concept, "group_id": f"g-{concept}-{i}",
         "image_id": f"i-{concept}-{i}", "modality": modality, "pass": i < 2}
        for concept in ("cat", "cow") for i in range(3) for modality in MODALITIES
    ]
    recruitment = recruit_all_modality_capable_groups(
        groups, rows, concepts=("cat", "cow"), n_per_concept=2
    )
    assert recruitment["complete"] is True
    assert recruitment["eligible_counts"] == {"cat": 2, "cow": 2}
    assert recruitment["causal_outcomes_used_for_selection"] is False

    source = {"family": "animal_sound", "audit_digest": "sha256:source",
              "verdict": "PROPERTY_AUDIT_NO_GO"}
    linkage = {"verdict": "AUDIO_METADATA_LINKAGE_GO",
               "audit_digest": "sha256:link"}
    report = recruited_exploratory_verdict(
        [], source_audit=source, linkage_audit=linkage,
        recruitment=recruitment, layers=(16, 17)
    )
    assert report["verdict"] == "RECRUITED_NEW_PROPERTY_EXPLORATORY_NO_GO"
    assert report["source_aggregate_verdict_unchanged"] is True
    assert report["is_confirmation"] is False

# ----------------------------------------------------- the corrected record


def test_all_six_original_development_directions_are_recognized() -> None:
    record = development_direction_record()
    directions = {row["direction"]: row for row in record["directions"]}
    assert set(directions) == {
        "bird->cat",
        "cat->bird",
        "bird->zebra",
        "zebra->bird",
        "bird->giraffe",
        "giraffe->bird",
    }
    assert all(row["tested"] for row in directions.values())
    assert all(row["trials"] == 24 for row in directions.values())
    assert directions["bird->cat"]["successes"] == 24
    assert directions["cat->bird"]["successes"] == 0
    assert directions["bird->zebra"]["successes"] == 1
    assert directions["bird->giraffe"]["successes"] == 3


def test_cat_to_bird_is_recorded_as_tested_not_untested() -> None:
    record = development_direction_record()
    assert record["reverse_direction_tested"] is True
    assert record["any_direction_untested"] is False
    assert record["asymmetry_established"] is False
    statement = record["accurate_statement"]
    assert "tested in development" in statement
    assert "0 successes in 24 trials" in statement
    assert "has not been" in statement and "fresh data" in statement
    assert "untested" not in statement.replace("independently tested", "")


def test_leg_count_cannot_carry_a_new_property_claim() -> None:
    limit = leg_count_property_limit()
    assert limit["downstream_change_tested_so_far"] == "2 -> 4"
    assert set(limit["directions_sharing_that_change"]) == {
        "bird->cat",
        "bird->zebra",
        "bird->giraffe",
    }
    assert "cat->zebra" in limit["directions_with_no_observable_change"]
    assert limit["cannot_test"] == "generalization to a new downstream property"


def test_only_the_pooled_lens_is_claimed_to_span_the_band() -> None:
    assert "pooled" in followup.POOLED_ONLY_BAND_NOTE
    assert "L33-L40 only" in followup.POOLED_ONLY_BAND_NOTE
    assert list(VALIDATED_BAND) == list(range(16, 41))


# ------------------------------------------------ A. exploratory localization


def test_localization_grid_is_frozen_and_labels_itself_exploratory() -> None:
    grid = localization_grid()
    assert grid["label"] == "exploratory"
    assert grid["is_confirmation"] is False
    assert grid["analysis_rule"]["frozen_before_any_sub_band_outcome"] is True
    assert list(grid["analysis_rule"]["conditions"]) == list(REQUIRED_CONDITIONS)
    assert grid["analysis_rule"]["teacher_forcing_used"] is False
    assert grid["analysis_rule"]["candidate_list_supplied"] is False
    assert grid["grid_digest"].startswith("sha256:")
    for band in grid["bands"]:
        assert set(band["layers"]) <= set(VALIDATED_BAND)
    # every family is populated and the partition family is genuinely disjoint
    partition = [
        band for band in grid["bands"] if band["name"] in grid["families"]["partition"]
    ]
    assert len(partition) == 5
    seen: set[int] = set()
    for band in partition:
        assert not seen & set(band["layers"])
        seen |= set(band["layers"])
    assert seen == set(VALIDATED_BAND)


def test_nested_bands_never_produce_an_exact_onset_claim() -> None:
    grid = localization_grid()
    by_name = {band["name"]: band for band in grid["bands"]}
    nested = [by_name["L16_L40"], by_name["L24_L40"], by_name["L33_L40"]]
    assert bands_are_nested_chain(nested)
    boundary = localization_claim_boundary(nested, grid=grid)
    assert boundary["onset_layer_claimed"] is False
    assert boundary["passing_bands_are_nested_chain"] is True
    assert "nested chain" in boundary["onset_claim"]
    assert boundary["necessity_claimed"] is False

    disjoint = [by_name["L21_L25"], by_name["L36_L40"]]
    assert not bands_are_nested_chain(disjoint)
    boundary = localization_claim_boundary(disjoint, grid=grid)
    assert boundary["onset_layer_claimed"] is False
    assert "not the layer at which the effect begins" in boundary["onset_claim"]


def test_localization_summary_is_labelled_exploratory(tmp_path: Path) -> None:
    report = run_mock_localization(tmp_path / "loc")
    assert report["label"] == "exploratory"
    assert report["is_confirmation"] is False
    assert "spent" in report["population"]
    assert report["verdict"].startswith("EXPLORATORY_")
    assert "confirm" not in report["verdict"].lower().replace("confirmation", "")
    assert report["claim_boundary"]["onset_layer_claimed"] is False
    assert report["claim_boundary"]["necessity_claimed"] is False
    assert report["bands_carrying_effect"]


def test_localization_budget_is_printable_before_model_load() -> None:
    grid = localization_grid()
    budget = localization_budget(grid=grid, n_photographs=8)
    assert budget["lens_fits"] == 0
    assert budget["backward_passes"] == 0
    assert budget["new_media_opened"] == 0
    assert budget["patched_forwards"] == grid["n_bands"] * 8 * 3 * 4
    assert budget["total_forwards"] == budget["patched_forwards"] + 24


def test_localization_summary_refuses_a_missing_control() -> None:
    grid = localization_grid()
    crippled = {
        **grid,
        "analysis_rule": {
            **grid["analysis_rule"],
            "conditions": ["exact", "zero", "random"],
        },
    }
    with pytest.raises(MultimodalFollowupRefused, match="not optional"):
        summarize_localization([], grid=crippled)


# ------------------------------------------------------ B0. property audit


def test_ambiguous_property_answers_are_refused() -> None:
    covering = audit_property_family("body_covering")
    refused = {row["concept"] for row in covering["refused_concepts"]}
    assert {"horse", "cow", "zebra", "giraffe", "elephant"} <= refused
    assert "bird" in covering["admissible_concepts"]
    assert all(row["reason"] for row in covering["refused_concepts"])

    sound = audit_property_family("animal_sound")
    refused_sound = {row["concept"] for row in sound["refused_concepts"]}
    # bird is refused until DOMINANT_ANSWER_RULE resolves it from real data
    assert "bird" in refused_sound
    assert "cat" in sound["admissible_concepts"]


def test_source_and_target_must_have_different_property_answers() -> None:
    record = assert_property_pair_changes_answer("animal_sound", "cat", "cow")
    assert record["changes_property"] is True
    assert record["is_leg_count"] is False
    with pytest.raises(MultimodalFollowupRefused, match="refused by the property audit"):
        assert_property_pair_changes_answer("animal_sound", "cat", "horse")

    # a pair sharing an answer is refused; body_covering's cat/dog both answer
    # "fur", so the check is exercised on a local family of the same shape
    shared = followup.PropertyFamily(
        name="shared_answer_probe",
        question="Q\nAnswer:",
        rationale="test fixture",
        perceptually_available=False,
        perceptual_rationale="test fixture",
        answers=(
            followup.PropertyAnswer("cat", "fur", ("fur",), True, "r"),
            followup.PropertyAnswer("dog", "fur", ("fur",), True, "r"),
            followup.PropertyAnswer("cow", "hide", ("hide",), True, "r"),
        ),
    )
    with pytest.raises(MultimodalFollowupRefused, match="does not change the property"):
        assert_property_pair_changes_answer(shared, "cat", "dog")
    assert assert_property_pair_changes_answer(shared, "cat", "cow")["changes_property"]

    directions = {
        row["direction"]
        for row in audit_property_family("animal_sound")["candidate_directions"]
    }
    assert "cat->cow" in directions
    assert "cat->horse" not in directions


def test_complete_generation_answer_normalization() -> None:
    audit = audit_property_family("body_covering")
    by_concept = {row["concept"]: row for row in audit["concepts"]}
    bird, cat, sheep = by_concept["bird"], by_concept["cat"], by_concept["sheep"]
    assert property_answer_matches("Feathers", bird)
    assert property_answer_matches("  feathers.\n", bird)
    assert property_answer_matches("soft feathers", bird)
    assert property_answer_matches("fur", cat)
    assert not property_answer_matches("fur", bird)
    assert not property_answer_matches("not feathers", bird)
    # a declared alias of one answer counts; an undeclared synonym does not
    assert property_answer_matches("fleece", sheep)
    assert not property_answer_matches("hair", cat)
    with pytest.raises(MultimodalFollowupRefused, match="no declared alias"):
        property_answer_matches("skin", by_concept["elephant"])


def test_property_prompt_is_identical_apart_from_the_text_caption() -> None:
    spec = followup.PROPERTY_FAMILIES["body_covering"]
    question = spec.question
    assert spec.prompt("image", "a caption") == question
    assert spec.prompt("spoken_audio", "a caption") == question
    assert spec.prompt("text", "a caption").endswith(question)
    assert "a caption" in spec.prompt("text", "a caption")


# --------------------------------------------------- exclusions and freezing


def test_all_64_confirmation_candidate_images_are_excluded() -> None:
    candidates = [f"confcand-i{index:03d}" for index in range(64)]
    universe = exclusion_universe(confirmation_candidate_image_ids=candidates)
    assert universe["counts_by_source"]["confirmation_candidates_all_opened"] == 64
    assert set(candidates) <= set(universe["excluded_image_ids"])
    assert universe["candidates_not_only_recruits"] is True
    # the 16 recruits are a subset, so excluding only them is not enough
    recruited_only = exclusion_universe(
        confirmation_candidate_image_ids=candidates[:16]
    )
    assert recruited_only["n_excluded"] == 16 < universe["n_excluded"] == 64

    with pytest.raises(MultimodalFollowupRefused, match="reuses spent photographs"):
        artifact_exclusion_audit(
            [{"group_id": "g", "image_id": candidates[63]}],
            universe=universe,
            label="confirmation",
        )
    audit = artifact_exclusion_audit(
        [{"group_id": "g", "image_id": "brand-new-000"}],
        universe=universe,
        label="confirmation",
    )
    assert audit["disjoint"] is True
    assert audit["read_only"] is True
    assert audit["n_excluded_identities"] == 64


def test_extra_spent_report_paths_widen_exclusion_across_runs(tmp_path: Path) -> None:
    # a run this module has no dedicated loader for (e.g. an abandoned
    # property family) still leaks its opened photographs into the universe
    abandoned = tmp_path / "abandoned_animal_sound_report.json"
    abandoned.write_text(
        json.dumps(
            {
                "rows": [
                    {"group_id": "g1", "image_id": "abandoned-i001"},
                    {"group_id": "g2", "image_id": "abandoned-i002"},
                ],
                "capability_rows": [{"image_id": "abandoned-i003"}],
                "unrelated_field": {"nested": {"image_id": "abandoned-i004"}},
            }
        ),
        encoding="utf-8",
    )
    extra = load_extra_spent_image_ids([abandoned])
    assert extra["checksum_verified"] is False
    assert set(extra["image_ids"]) == {
        "abandoned-i001",
        "abandoned-i002",
        "abandoned-i003",
        "abandoned-i004",
    }
    assert extra["image_ids_by_report"][str(abandoned)] == sorted(extra["image_ids"])

    universe = exclusion_universe(
        extra_image_ids={"manually_declared_extra_runs": extra["image_ids"]}
    )
    assert "abandoned-i002" in universe["excluded_image_ids"]
    with pytest.raises(MultimodalFollowupRefused, match="reuses spent photographs"):
        artifact_exclusion_audit(
            [{"group_id": "g", "image_id": "abandoned-i002"}],
            universe=universe,
            label="fallback_family",
        )

    with pytest.raises(MultimodalFollowupRefused, match="not found"):
        load_extra_spent_image_ids([tmp_path / "does_not_exist.json"])


def test_development_and_confirmation_populations_are_disjoint(tmp_path: Path) -> None:
    result = run_mock_new_property_study(tmp_path / "study", scenario="favorable")
    assert result["confirmation"]["verdict"] == "NEW_PROPERTY_CONFIRMATION_GO"
    audit = result["confirmation"]["exclusion_audit"]
    assert audit["disjoint"] is True
    assert audit["overlap_with_excluded"] == []
    assert audit["counts_by_source"]["new_property_development"] > 0
    assert audit["counts_by_source"]["confirmation_candidates_all_opened"] == 64


def test_confirmation_cannot_open_before_the_design_is_frozen(tmp_path: Path) -> None:
    with pytest.raises(MultimodalFollowupRefused, match="Stage B2 must run"):
        assert_design_frozen(tmp_path / "never_written.json")

    result = run_mock_new_property_study(tmp_path / "study", scenario="favorable")
    path = tmp_path / "study" / "frozen_new_property_design.json"
    assert assert_design_frozen(path)["design_digest"] == result["frozen_design"][
        "design_digest"
    ]

    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["thresholds"]["min_success_rate"] = 0.10
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(MultimodalFollowupRefused, match="edited after it was written"):
        assert_design_frozen(path)


def test_freezing_refuses_a_direction_development_did_not_license() -> None:
    audit = audit_property_family("body_covering")
    empty = new_property_development_verdict([], audit=audit, capability_go=False)
    assert empty["verdict"] == "NEW_PROPERTY_DEVELOPMENT_CAPABILITY_NO_GO"
    with pytest.raises(MultimodalFollowupRefused, match="development returned"):
        freeze_new_property_design(
            development=empty,
            audit=audit,
            direction=("bird", "cat"),
            lens_checksum=MOCK_LENS_CHECKSUM,
            exclusions=exclusion_universe(),
            n_candidates=64,
            n_recruited=16,
            min_success_rate=0.75,
            min_control_margin=0.25,
            min_clean_capability_rate=0.75,
            familywise_alpha=0.05,
            recruitment_rule="clean capability in all three modalities",
            seed="s",
        )


def test_no_fitting_entry_point_is_reachable() -> None:
    source = Path(followup.__file__).read_text(encoding="utf-8")
    for forbidden in ("fit_arm", "combine_layer_shards", "JacobianLens", "backward("):
        assert forbidden not in source
    with pytest.raises(MultimodalFollowupRefused, match="no fitting entry point"):
        assert_lens_reused_not_refitted(
            {"lens_checksum": MOCK_LENS_CHECKSUM, "lens_refitted": True}
        )
    with pytest.raises(MultimodalFollowupRefused, match="nothing here fits"):
        assert_lens_reused_not_refitted(
            {"lens_checksum": MOCK_LENS_CHECKSUM, "backward_passes": 1}
        )
    with pytest.raises(MultimodalFollowupRefused, match="pin the pooled lens"):
        assert_lens_reused_not_refitted({"lens_checksum": ""})
    assert_lens_reused_not_refitted({"lens_checksum": MOCK_LENS_CHECKSUM})
    for budget in (
        localization_budget(grid=localization_grid(), n_photographs=8),
        followup_budget(stage="B1", n_candidates=64, n_recruited=8),
    ):
        assert budget["lens_fits"] == 0
        assert budget["backward_passes"] == 0


def test_no_candidate_list_or_teacher_forcing_anywhere() -> None:
    audit = audit_property_family("body_covering")
    grid = localization_grid()
    design = asymmetry_replication_design(
        lens_checksum=MOCK_LENS_CHECKSUM, exclusions=exclusion_universe()
    )
    for payload in (audit, grid["analysis_rule"], design):
        assert payload["teacher_forcing_used"] is False
        assert payload["candidate_list_supplied"] is False
    with pytest.raises(MultimodalFollowupRefused, match="unrestricted"):
        assert_open_endpoint({"teacher_forcing_used": True})
    with pytest.raises(MultimodalFollowupRefused, match="unrestricted"):
        assert_open_endpoint({"candidate_list_supplied": True})


def test_controls_cannot_be_omitted() -> None:
    assert set(CONTROL_CONDITIONS) == {"zero", "random", "unrelated"}
    assert_controls_complete(REQUIRED_CONDITIONS)
    for dropped in CONTROL_CONDITIONS:
        with pytest.raises(MultimodalFollowupRefused, match="not optional"):
            assert_controls_complete(
                [name for name in REQUIRED_CONDITIONS if name != dropped]
            )
    with pytest.raises(MultimodalFollowupRefused, match="not optional"):
        followup_budget(
            stage="B1",
            n_candidates=8,
            n_recruited=4,
            conditions=("exact", "zero", "random"),
        )


def test_confirmation_refuses_an_unpaired_control(tmp_path: Path) -> None:
    result = run_mock_new_property_study(tmp_path / "study", scenario="favorable")
    design = result["frozen_design"]
    rows = [
        row
        for row in result["confirmation"]["rows"]
        if row["condition"] != "unrelated"
    ]
    with pytest.raises(MultimodalFollowupRefused, match="missing control is refused"):
        followup.confirmation_verdict(
            rows,
            design=design,
            capability_go=True,
            exclusion_audit=result["confirmation"]["exclusion_audit"],
        )


# ---------------------------------------------------------- resume and mocks


def test_configuration_changes_invalidate_resume(tmp_path: Path) -> None:
    def fingerprint(alpha: float) -> RunFingerprint:
        return RunFingerprint(
            mode="mock",
            model_repo_id="mock/gemma-4-E4B-it",
            model_revision="mock-revision",
            processor_revision="mock-revision",
            layers=tuple(VALIDATED_BAND),
            lens_checksum=MOCK_LENS_CHECKSUM,
            manifest_checksum="sha256:" + "1" * 64,
            split_id="split",
            intervention_config={"alpha": alpha, "conditions": list(REQUIRED_CONDITIONS)},
        )

    root = tmp_path / "run"
    assert UnitStore(root, fingerprint(1.0)).open() == "starting"
    assert UnitStore(root, fingerprint(1.0)).open() == "resuming"
    with pytest.raises(IncompatibleStateError):
        UnitStore(root, fingerprint(2.0)).open()


def test_mock_interrupted_run_resumes_exactly(tmp_path: Path) -> None:
    root = tmp_path / "loc"
    with pytest.raises(MockDisconnect):
        run_mock_localization(root, n_photographs=2, interrupt_after=7)
    finished = run_mock_localization(root, n_photographs=2)
    complete = run_mock_localization(tmp_path / "loc_clean", n_photographs=2)

    grid = localization_grid()
    total = grid["n_bands"] * 2 * len(MODALITIES) * len(REQUIRED_CONDITIONS)
    assert complete["n_units_computed_this_session"] == total
    # exactly the seven completed units survived; nothing else was recomputed
    assert finished["n_units_computed_this_session"] == total - 7
    assert finished["cells"] == complete["cells"]
    assert finished["report_checksum"] == complete["report_checksum"]


def test_mock_outcomes_remain_distinct(tmp_path: Path) -> None:
    verdicts = {}
    modes = {}
    for scenario in SCENARIOS:
        result = run_mock_new_property_study(tmp_path / scenario, scenario=scenario)
        verdicts[scenario] = result["development"]["verdict"]
        modes[scenario] = result["development"]["failure_modes"].get("cat->cow")
    assert verdicts["favorable"] == "NEW_PROPERTY_DEVELOPMENT_GO"
    assert verdicts["null"] == "NEW_PROPERTY_DEVELOPMENT_NO_GO"
    assert verdicts["control_failure"] == "NEW_PROPERTY_DEVELOPMENT_CONTROL_FAILURE"
    assert verdicts["capability_no_go"] == "NEW_PROPERTY_DEVELOPMENT_CAPABILITY_NO_GO"
    assert len(set(verdicts.values())) == len(SCENARIOS)
    assert modes["null"] == "no_effect_in_every_modality"
    assert modes["control_failure"] == "controls_also_moved_the_answer"
    # a NO_GO never opens confirmation
    for scenario in ("null", "control_failure", "capability_no_go"):
        result = run_mock_new_property_study(
            tmp_path / f"{scenario}_again", scenario=scenario
        )
        assert result["confirmation"] is None
        assert result["frozen_design"] is None
        assert not (
            tmp_path / f"{scenario}_again" / "frozen_new_property_design.json"
        ).exists()


# ------------------------------------------------------- C. asymmetry design


def test_asymmetry_design_frames_the_outcome_as_replication() -> None:
    design = asymmetry_replication_design(
        lens_checksum=MOCK_LENS_CHECKSUM,
        exclusions=exclusion_universe(
            confirmation_candidate_image_ids=[f"c{i}" for i in range(64)]
        ),
    )
    assert design["direction"] == ["cat", "bird"]
    assert design["layers"] == list(VALIDATED_BAND)
    assert design["alpha"] == 1.0
    assert design["lens_refitted"] is False
    assert list(design["controls"]) == list(CONTROL_CONDITIONS)
    assert design["interpretation"]["cause_identified"] is False
    assert design["n_excluded_identities"] == 64
    assert design["development_record"]["reverse_direction_tested"] is True


def test_asymmetry_null_replicates_without_explaining_itself(tmp_path: Path) -> None:
    null = run_mock_asymmetry_study(tmp_path / "null", scenario="null")["report"]
    assert null["verdict"] == "ASYMMETRY_REPLICATED_NO_REVERSE_EFFECT"
    assert null["reverse_successes"] == 0
    assert null["cause_of_asymmetry_identified"] is False
    assert "does not show" in null["claim_boundary"]

    positive = run_mock_asymmetry_study(tmp_path / "pos", scenario="favorable")["report"]
    assert positive["verdict"] == "ASYMMETRY_DID_NOT_REPLICATE_REVERSE_EFFECT_FOUND"
    assert positive["reverse_successes"] > 0


def test_stage_map_never_promises_confirmation_from_a_spent_population() -> None:
    stages = {row["stage"]: row for row in followup.stage_map()["stages"]}
    assert stages["A"]["confirms"] is False
    assert stages["A"]["label"] == "exploratory/descriptive"
    assert "spent" in stages["A"]["population"]
    assert stages["B3"]["confirms"] is True
    assert "64" in stages["B3"]["population"]
    assert stages["C"]["confirms"] is False
    assert all(row["fits"] == 0 for row in stages.values())


def test_mock_groups_are_deterministic() -> None:
    assert mock_groups("x", 3) == mock_groups("x", 3)
    assert mock_groups("x", 3) != mock_groups("y", 3)


# ------------------------------------------- perceptual availability criterion


def test_perceptually_available_property_is_disqualified() -> None:
    covering = followup.PROPERTY_FAMILIES["body_covering"]
    assert covering.perceptually_available is True
    assert covering.disqualified is True
    assert "visible in the photograph" in covering.perceptual_rationale

    audit = audit_property_family("body_covering")
    assert audit["verdict"] == "PROPERTY_AUDIT_PERCEPTUALLY_AVAILABLE_NO_GO"
    assert audit["family_disqualified"] is True
    assert audit["usable_concepts"] == []
    assert audit["candidate_directions"] == []

    # even a fully capable, well-stocked run cannot resurrect it
    generous = audit_property_family(
        "body_covering",
        available_media=dict.fromkeys(("bird", "cat", "sheep"), 999),
        min_media_per_concept=1,
        clean_capability={
            c: dict.fromkeys(MODALITIES, 1.0) for c in ("bird", "cat", "sheep")
        },
    )
    assert generous["verdict"] == "PROPERTY_AUDIT_PERCEPTUALLY_AVAILABLE_NO_GO"
    assert generous["candidate_directions"] == []

    with pytest.raises(MultimodalFollowupRefused, match="perceptually available"):
        assert_property_pair_changes_answer("body_covering", "bird", "cat")


def test_animal_sound_is_not_perceptually_available() -> None:
    sound = followup.PROPERTY_FAMILIES["animal_sound"]
    assert sound.perceptually_available is False
    assert sound.disqualified is False
    audit = audit_property_family("animal_sound")
    assert audit["verdict"] == "PROPERTY_AUDIT_GO"
    assert {"cat", "cow", "dog", "sheep"} <= set(audit["admissible_concepts"])


# ------------------------------------------------- the dominant-answer rule


def test_dominant_answer_rule_is_declared_before_data() -> None:
    rule = followup.DOMINANT_ANSWER_RULE
    assert rule["declared_before_data"] is True
    assert rule["ties_refused"] is True
    assert "stability, not taxonomic correctness" in rule["standard"]


def test_answer_key_matches_the_scorer_surface_rule() -> None:
    assert answer_key(" Feathers<turn|>") == "feathers"
    assert answer_key("white fur.") == "fur"
    assert answer_key("Moo<turn|>") == "moo"
    assert answer_key("") == ""


def test_dominant_answer_resolves_only_when_stable_in_every_modality() -> None:
    stable = {m: ["Chirp<turn|>"] * 9 + ["Tweet"] for m in MODALITIES}
    got = resolve_dominant_answer(stable, threshold=0.75)
    assert got["resolved"] is True
    assert got["answer"] == "chirp"
    assert all(rate >= 0.75 for rate in got["rates_by_modality"].values())

    # the image route splitting to a species-specific call refuses the concept
    split = {
        "text": ["Chirp"] * 10,
        "spoken_audio": ["Chirp"] * 10,
        "image": ["Quack"] * 6 + ["Chirp"] * 4,
    }
    got = resolve_dominant_answer(split, threshold=0.75)
    assert got["resolved"] is False
    assert "dominates in image" in got["reason"]

    # below threshold everywhere is refused even without a competing winner
    weak = {m: ["Chirp"] * 5 + [f"x{i}" for i in range(5)] for m in MODALITIES}
    assert resolve_dominant_answer(weak, threshold=0.75)["resolved"] is False
    # an exact tie is refused rather than broken arbitrarily
    tie = {m: ["Chirp"] * 5 + ["Tweet"] * 5 for m in MODALITIES}
    assert resolve_dominant_answer(tie, threshold=0.5)["resolved"] is False
    assert resolve_dominant_answer({}, threshold=0.75)["resolved"] is False


def test_bird_sound_answer_is_empirical_not_declared() -> None:
    bird = followup.PROPERTY_FAMILIES["animal_sound"].answer_for("bird")
    assert bird.empirical_answer_required is True
    assert bird.answer == ""

    # with no data the concept cannot be used
    blind = audit_property_family("animal_sound")
    assert "bird" not in blind["usable_concepts"]

    # stable data promotes it and gives it the resolved answer
    completions = {
        "bird": {m: ["Chirp<turn|>"] * 12 for m in MODALITIES},
        "cat": {m: ["Meow<turn|>"] * 12 for m in MODALITIES},
    }
    resolved = audit_property_family(
        "animal_sound",
        available_media={"bird": 99, "cat": 99},
        min_media_per_concept=48,
        clean_capability={c: dict.fromkeys(MODALITIES, 1.0) for c in ("bird", "cat")},
        observed_completions=completions,
    )
    rows = {row["concept"]: row for row in resolved["concepts"]}
    assert rows["bird"]["answer"] == "chirp"
    assert rows["bird"]["empirical_resolution"]["resolved"] is True
    assert "bird->cat" in {d["direction"] for d in resolved["candidate_directions"]}

    # a species-split image route keeps bird out
    completions["bird"]["image"] = ["Quack<turn|>"] * 8 + ["Chirp<turn|>"] * 4
    refused = audit_property_family(
        "animal_sound",
        available_media={"bird": 99, "cat": 99},
        min_media_per_concept=48,
        clean_capability={c: dict.fromkeys(MODALITIES, 1.0) for c in ("bird", "cat")},
        observed_completions=completions,
    )
    assert "bird" not in refused["usable_concepts"]
    assert "bird->cat" not in {d["direction"] for d in refused["candidate_directions"]}


def test_empirical_capability_does_not_need_a_separate_capability_dict() -> None:
    # audit_property_family must resolve and admit an empirical concept from
    # observed_completions alone, with no clean_capability argument at all —
    # scoring "pass" against bird's answer is impossible before this call
    # resolves what that answer even is, so requiring a pre-scored capability
    # dict for it would be circular.
    completions = {
        "bird": {m: ["Chirp<turn|>"] * 12 for m in MODALITIES},
        "cat": {m: ["Meow<turn|>"] * 12 for m in MODALITIES},
    }
    resolved = audit_property_family(
        "animal_sound",
        available_media={"bird": 99, "cat": 99},
        min_media_per_concept=48,
        observed_completions=completions,
    )
    assert resolved["verdict"] == "PROPERTY_AUDIT_GO"
    assert "bird" in resolved["usable_concepts"]
    rows = {row["concept"]: row for row in resolved["concepts"]}
    assert rows["bird"]["capability_by_modality_sufficient"] is True
    assert rows["bird"]["clean_capability"] == rows["bird"]["empirical_resolution"][
        "rates_by_modality"
    ]


def test_unresolved_empirical_pair_is_refused_not_silently_admitted() -> None:
    # before any data exists, bird's declared aliases are empty; the pair
    # check must not read that as "the two answers trivially differ" and let
    # it through — it must refuse until an empirical answer is resolved.
    with pytest.raises(MultimodalFollowupRefused, match="refused by the property audit"):
        assert_property_pair_changes_answer("animal_sound", "bird", "cat")

    resolved_rows = {
        "bird": {
            "answer": "chirp", "aliases": ["chirp"], "admissible": True,
            "reason": "", "empirical_answer_required": True,
        },
    }
    record = assert_property_pair_changes_answer(
        "animal_sound", "bird", "cat", resolved=resolved_rows
    )
    assert record["changes_property"] is True
    assert record["source_answer"]["aliases"] == ["chirp"]


def test_declared_concept_missing_a_modality_stays_gated_when_capability_is_supplied() -> None:
    # regression: a real animal_sound run scored cat=0.60/0.88/0.88,
    # dog=0.29/0.65/0.79, cow=1.00/0.85/0.71 against a 0.75 threshold. Every
    # one of the three fails in at least one modality. A caller that omits
    # clean_capability (as the notebook's first, resolution-only pass does)
    # must not be mistaken for the final, gated verdict: omitting it makes
    # capability_by_modality_sufficient None, and None does not block
    # usability the way False does, so a concept that never had its
    # capability checked would wrongly appear usable.
    rates = {
        "cat": {"text": 0.604, "image": 0.875, "spoken_audio": 0.875},
        "cow": {"text": 1.0, "image": 0.854, "spoken_audio": 0.708},
        "dog": {"text": 0.292, "image": 0.646, "spoken_audio": 0.792},
    }
    ungated = audit_property_family(
        "animal_sound",
        available_media=dict.fromkeys(rates, 99),
        min_media_per_concept=48,
    )
    ungated_rows = {row["concept"]: row for row in ungated["concepts"]}
    for concept in rates:
        assert ungated_rows[concept]["capability_by_modality_sufficient"] is None
    # every one of the three is nonetheless reported "usable" when capability
    # was never actually supplied -- this is the bug, not the fix
    assert set(rates) <= set(ungated["usable_concepts"])

    gated = audit_property_family(
        "animal_sound",
        available_media=dict.fromkeys(rates, 99),
        min_media_per_concept=48,
        clean_capability=rates,
    )
    for concept in rates:
        assert concept not in gated["usable_concepts"]
    assert gated["candidate_directions"] == []
    assert gated["verdict"] == "PROPERTY_AUDIT_NO_GO"


def test_freeze_uses_the_resolved_empirical_answer_not_the_declared_one() -> None:
    audit = audit_property_family(
        "animal_sound",
        available_media={"bird": 99, "cat": 99},
        min_media_per_concept=48,
        observed_completions={
            "bird": {m: ["Chirp<turn|>"] * 12 for m in MODALITIES},
            "cat": {m: ["Meow<turn|>"] * 12 for m in MODALITIES},
        },
    )
    development = new_property_development_verdict(
        [
            {
                "direction": "bird->cat", "modality": modality, "condition": condition,
                "success": condition == "exact", "n": 1,
                "all_prompt_positions_patched": True,
                "layers_patched": list(VALIDATED_BAND),
                "max_activation_norm_ratio": 1.0,
                "max_update_to_activation_norm_ratio": 0.1,
            }
            for modality in MODALITIES
            for condition in REQUIRED_CONDITIONS
        ],
        audit=audit, capability_go=True,
        min_success_rate=1.0, min_control_margin=0.5,
    )
    assert development["verdict"] == "NEW_PROPERTY_DEVELOPMENT_GO"
    design = freeze_new_property_design(
        development=development, audit=audit, direction=("bird", "cat"),
        lens_checksum=MOCK_LENS_CHECKSUM, exclusions=exclusion_universe(),
        n_candidates=64, n_recruited=16, min_success_rate=0.75,
        min_control_margin=0.25, min_clean_capability_rate=0.75,
        familywise_alpha=0.05, recruitment_rule="r", seed="s",
    )
    # this is the fix under test: without it, bird's frozen alias set would
    # be empty (the static declared table), and Stage 5B3 would crash trying
    # to score against it exactly as Stage 5B1 did before the fix
    assert design["answer_aliases"]["bird"] == ["chirp"]
    assert design["answer_aliases"]["cat"] == ["meow", "meows"]
