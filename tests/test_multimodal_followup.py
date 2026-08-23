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
    artifact_exclusion_audit,
    assert_controls_complete,
    assert_design_frozen,
    assert_lens_reused_not_refitted,
    assert_open_endpoint,
    assert_property_pair_changes_answer,
    asymmetry_replication_design,
    audit_property_family,
    bands_are_nested_chain,
    development_direction_record,
    exclusion_universe,
    followup_budget,
    freeze_new_property_design,
    leg_count_property_limit,
    localization_budget,
    localization_claim_boundary,
    localization_grid,
    new_property_development_verdict,
    property_answer_matches,
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
    assert "bird" in refused_sound, "COCO 'bird' has no single conventional sound"
    assert "cat" in sound["admissible_concepts"]


def test_source_and_target_must_have_different_property_answers() -> None:
    record = assert_property_pair_changes_answer("body_covering", "bird", "cat")
    assert record["changes_property"] is True
    assert record["is_leg_count"] is False
    with pytest.raises(MultimodalFollowupRefused, match="does not change the property"):
        assert_property_pair_changes_answer("body_covering", "cat", "dog")
    with pytest.raises(MultimodalFollowupRefused, match="refused by the property audit"):
        assert_property_pair_changes_answer("body_covering", "bird", "zebra")

    directions = {
        row["direction"] for row in audit_property_family("body_covering")["candidate_directions"]
    }
    assert "bird->cat" in directions
    assert "cat->dog" not in directions


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
        modes[scenario] = result["development"]["failure_modes"].get("bird->cat")
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
