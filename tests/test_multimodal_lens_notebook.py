from __future__ import annotations

import ast
import json
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILDER = ROOT / "scripts" / "_build_multimodal_lens_notebook.py"
NOTEBOOK = ROOT / "notebooks" / "multimodal_jspace_matched_jlens_colab.ipynb"


def _notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _source() -> str:
    return "\n".join(
        "".join(cell.get("source", [])) for cell in _notebook()["cells"]
    )


def test_builder_is_byte_identical() -> None:
    before = NOTEBOOK.read_bytes()
    runpy.run_path(str(BUILDER), run_name="__main__")
    assert NOTEBOOK.read_bytes() == before


def test_notebook_is_output_free_and_all_code_parses() -> None:
    notebook = _notebook()
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        assert cell.get("outputs") == []
        assert cell.get("execution_count") is None
        ast.parse("".join(cell["source"]))


def test_notebook_contains_the_four_arm_scientific_contract() -> None:
    source = _source()
    for required in (
        "LENS_ARMS, fit_arm, plan_units",
        "capture_eval_rows",
        "clean_capability_required_in_every_modality_and_endpoint",
        '"controls": ["random", "unrelated"]',
        '"teacher_forcing_used": False',
        '"candidate_list_supplied": False',
        "unrestricted_swap_trial",
        "checkpoint_every=CHECKPOINT_EVERY",
        "answer_equivalence_record",
        "load_completed_causal_source",
        "N_CAUSAL_CANDIDATES_PER_CONCEPT = 96",
        "previously screened images excluded",
        "matched_multimodal_jlens_unrestricted_swap.v3",
        "open_answer_matches",
        "imported read-only",
        "RUN_STAGE3B_ALPHA_SWEEP",
        "ALPHA_SWEEP = (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)",
        '"alpha_selected_after_outcomes": ALPHA_SWEEP_OUTCOME_INFORMED',
        "outcome_informed_stable_range_refinement",
        "load_completed_alpha_sweep_source",
        "matched_multimodal_jlens_unrestricted_alpha_refinement.v6",
        "mmpilot.multimodal_alpha_identity_specificity.v1",
        '"property_endpoint_used_for_selection": False',
        '"exploratory_best_alpha": _exploratory_best_alpha',
        "target_logit_delta",
        "target_rank_improvement",
        "controls_are_intensity_matched",
        "population_reused_without_reselection",
        "alpha1_exact_outcome_parity",
        "Alpha=1 remains primary",
    ):
        assert required in source
    assert "int(_clean_logits.argmax()) == int(_expected)" not in source


def test_notebook_contains_broad_pooled_j_workspace_extension() -> None:
    source = _source()
    for required in (
        "RUN_STAGE3C_BROAD_POOLED_WORKSPACE",
        "BROAD_POOLED_EARLY_LAYERS = tuple(range(16, 33))",
        "BROAD_POOLED_LATE_LAYERS = tuple(range(33, 41))",
        "BROAD_POOLED_BAND = tuple(range(16, 41))",
        "BROAD_POOLED_ALPHAS = (1.0, 2.0)",
        "combine_layer_shards",
        '"fit_distribution": "33 text + 33 image + 33 spoken_audio"',
        '"r_lens_used": False',
        '"positions": "every original prompt position"',
        '"teacher_forcing_used": False',
        '"candidate_list_supplied": False',
        '"endpoint": "unrestricted full-vocabulary next-token top1"',
        "BROAD_POOLED_J_DEVELOPMENT_ALPHA1_GO",
        "BROAD_POOLED_J_DEVELOPMENT_ALPHA2_SENSITIVITY_ONLY",
        "fresh frozen population",
    ):
        assert required in source


def test_notebook_contains_frozen_fresh_multimodal_confirmation() -> None:
    source = _source()
    for required in (
        "RUN_STAGE3D_FRESH_MULTIMODAL_CONFIRMATION",
        "EXPECTED_BROAD_DEVELOPMENT_REPORT_CHECKSUM",
        "sha256:ec1747a78902080ac3fac5f6aa5bc105e36f49ade5b517282ad7c3673da31a42",
        "EXPECTED_BROAD_POOLED_LENS_CHECKSUM",
        "CONFIRMATION_DIRECTION = (\"bird\", \"cat\")",
        "CONFIRMATION_ALPHA = 1.0",
        "CONFIRMATION_IMAGES = 16",
        "load_broad_pooled_development_source",
        "paired_binary_one_sided_p",
        "holm_adjust",
        '"lens_refitted": False',
        '"independent_unit": "photograph with three synchronized modalities"',
        '"multiple_testing": "Holm across 3 modalities x 3 controls"',
        "FRESH_MULTIMODAL_CONFIRMATION_GO",
        "fresh_multimodal_confirmation_report.json",
    ):
        assert required in source
    assert "RUN_STAGE3C_BROAD_POOLED_WORKSPACE and RUN_STAGE3D_FRESH_MULTIMODAL_CONFIRMATION" in source


def test_notebook_contains_no_refit_leg_count_target_generalization() -> None:
    source = _source()
    for required in (
        "RUN_STAGE7A_FREEZE_LEG_GENERALIZATION_POPULATION",
        "RUN_STAGE7B_LEG_GENERALIZATION_DEVELOPMENT",
        "RUN_STAGE7C_FREEZE_LEG_GENERALIZATION_CONFIRMATION",
        "RUN_STAGE7D_LEG_GENERALIZATION_CONFIRMATION",
        'LEG_GENERALIZATION_TARGET_ANSWERS = {"cat": "4", "ant": "6", "spider": "8"}',
        'LEG_GENERALIZATION_SOURCE = "bird"',
        '"lens_refitted": False',
        '"positions": "every_original_prompt_position"',
        "LEG_GENERALIZATION_LAYERS = tuple(range(16, 41))",
        "LEG_GENERALIZATION_ALPHA = 1.0",
        "n_per_concept=31",
        "9 development + 22 confirmation",
        'required=("2", "4", "6", "8"), leading_space=True',
        'variant="single_token_space_prefixed_digit"',
        "preflight_fp32_or_refuse",
        '_observed_dtype != "torch.float32"',
        "load_broad_pooled_development_source",
        "STAGE 7 NO-REFIT LEG-COUNT GENERALIZATION",
        "frozen_population.json",
        "confirmation_design.json",
        "fresh_multimodal_leg_count_generalization_report.json",
        '"fresh_confirmation_opened": False',
        "fitting performed False; backward passes 0",
    ):
        assert required in source, required

    section = source.split("## 20. Stage 7: no-refit target generalization")[1]
    assert "fit_arm(" not in section
    assert "fit_jacobian" not in section
    assert "teacher_forcing" not in section


def test_mock_notebook_executes_end_to_end(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TMP", str(tmp_path))
    monkeypatch.setenv("TEMP", str(tmp_path))
    namespace: dict = {"__name__": "__main__"}
    for index, cell in enumerate(_notebook()["cells"]):
        if cell["cell_type"] == "code":
            exec(
                compile("".join(cell["source"]), f"cell-{index}", "exec"),
                namespace,
            )
    assert namespace["FINAL"]["causal_comparison"]["verdict"] == "MEASURED"
    assert namespace["FINAL"]["cross_evaluation"]["n_rows"] > 0
    assert set(namespace["LENS_CHECKSUMS"]) == {
        "text",
        "image",
        "spoken_audio",
        "pooled",
    }


def test_notebook_contains_the_followup_stage_contract() -> None:
    source = _source()
    for required in (
        "RUN_STAGE5A_BAND_LOCALIZATION",
        "RUN_STAGE5B00_PROPERTY_PROMPT_SCREEN",
        "RUN_STAGE5B0_PROPERTY_AUDIT",
        "RUN_STAGE5B1_NEW_PROPERTY_DEVELOPMENT",
        "RUN_STAGE5B01_AUDIO_LINKAGE_AUDIT",
        "RUN_STAGE3DA_CONFIRMATION_REALIZATION_REPLICATION",
        "RUN_STAGE5B1A_INSTRUMENT_AMENDMENT",
        "RUN_STAGE5B1RC_CORRECTED_EXPLORATORY",
        "RUN_STAGE5B2_FREEZE_NEW_PROPERTY_DESIGN",
        "RUN_STAGE5B3_NEW_PROPERTY_CONFIRMATION",
        "RUN_STAGE5C_ASYMMETRY_REPLICATION",
        "RUN_STAGE6C2_CATDOG_PATH_LOCALIZATION",
        "CATDOG_INCONCLUSIVE_DEVELOPMENT_RUN_DIR",
        "EXPECTED_CATDOG_INCONCLUSIVE_DEVELOPMENT_CHECKSUM",
        "frozen_grid_record",
        "verify_inconclusive_source_report",
        "summarize_path_localization",
        '"conditions": ["direct_answer"]',
        '"exact_exchange_outcomes_used_for_selection": False',
        "catdog_direct_answer_path_localization_report.json",
        "RUN_ARTIFACT_EXCLUSION_AUDIT",
        "load_spent_confirmation_population",
        "load_extra_spent_image_ids",
        "EXTRA_SPENT_REPORT_PATHS",
        "sha256:2bb6dcc1346229573566125bc8d91c782247d55af5091f4215d98bb621472ff7",
        "FRESH_CONFIRMATION_CANDIDATES_OPENED = 64",
        "FRESH_CONFIRMATION_IMAGES_RECRUITED = 16",
        "localization_grid",
        "summarize_localization",
        "assert_lens_reused_not_refitted",
        "assert_property_pair_changes_answer",
        "audit_property_family",
        "freeze_new_property_design",
        "assert_design_frozen",
        "confirmation_verdict",
        "asymmetry_replication_design",
        "asymmetry_replication_verdict",
        "artifact_exclusion_audit",
        "generation_trial_row",
        "audio_metadata_linkage_audit",
        "recruit_all_modality_capable_groups",
        "corrected_exploratory_verdict",
        "direct_answer_trial_row",
        "instrument_defect_amendment",
        "legacy_confirmation_realization_audit",
        "legacy_confirmation_replication_verdict",
        "confirmation_leg_count_prompt",
        "STAGE 3DA CONFIRMATION REALIZATION REPLICATION BUDGET",
        "can relabel the confirmed verdict  no",
        "unrestricted_greedy_direct_answer_trial",
        "MODEL_DTYPE_REALIZATION",
        "realization_policy=MODEL_DTYPE_REALIZATION",
        "realization_policy_digest",
        "EXPECTED_RECRUITED_EXPLORATORY_REPORT_CHECKSUM",
        "sha256:467a2862cef70f0b59a75678c6a73c68259f4b29f715a97fbb831914710f660a",
        "AUDIO_METADATA_LINKAGE_GO",
        "source aggregate verdict     PROPERTY_AUDIT_NO_GO (unchanged)",
        "one checksum-valid JSON per completed",
        "EXACT GENERATIONS",
        "MAXIMUM TOKEN FORWARDS",
        "backward passes            0",
        "expected Drive usage",
        "primary outcome rule",
        "cannot produce a GO",
        "scientific_recompute      0",
        "unrestricted_greedy_swap_trial",
        "NEW_PROPERTY_FAMILY = \"animal_sound\"",
        "NEW_PROPERTY_PROMPT_ID = \"baseline_v1\"",
        "property_prompt_screen_verdict",
        "EXPECTED_PROPERTY_PROMPT_SCREEN_SOURCE_FILE_SHA256",
        "EXPECTED_NEW_PROPERTY_PROMPT_SCREEN_CHECKSUM",
        '"causal_outcomes_used_for_selection": False',
        '"causal_spending_licensed"',
        "DOMINANT_ANSWER_RULE",
        "observed_completions=_completions_by_concept",
        "perceptually_available",
        "NEW_PROPERTY_DIRECTION_PRIORITY",
        "stage_map",
        "followup_budget",
        "localization_budget",
    ):
        assert required in source, required


def test_notebook_never_refits_or_restricts_in_a_followup_stage() -> None:
    source = _source()
    # every follow-up stage pins the completed pooled lens and loads it
    assert source.count("assert_lens_reused_not_refitted") >= 3
    assert "fit_arm" not in source.split("## 12. Read-only artifact")[1]
    followups = source.split("## 12. Read-only artifact")[1]
    assert "teacher_forcing" not in followups.replace('"teacher_forcing_used": False', "")
    assert "candidate list" not in followups.lower().replace(
        "no candidate list", ""
    ).replace("candidate list or teacher forcing", "")


def test_notebook_labels_localization_exploratory_and_claims_no_onset() -> None:
    source = _source()
    assert "**Exploratory and descriptive. Not confirmation, and not promotable.**" in source
    assert "cannot be read as an onset" in source
    assert '"label": "exploratory"' in source or '"exploratory"' in source
    assert "onset_layer_claimed" in source
    assert "already spent" in source


def test_notebook_states_the_corrected_cat_to_bird_record() -> None:
    source = _source()
    assert "Cat to bird **was** tested in development" in source
    assert "0 successes in 24 trials" in source
    assert "development_direction_record" in source
    assert "cat->bird was untested" not in source


def test_notebook_excludes_all_sixty_four_confirmation_candidates() -> None:
    source = _source()
    assert "all 64" in source
    assert "not just" in source or "not only" in source
    assert "confirmation_candidate_image_ids=SPENT_CONFIRMATION[\"candidate_image_ids\"]" in source


def test_notebook_only_claims_the_pooled_lens_spans_the_band() -> None:
    source = _source()
    assert "the text-only, image-only" in source
    assert "cover L33-L40" in source


def test_mock_notebook_exercises_every_followup_outcome(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TMP", str(tmp_path))
    monkeypatch.setenv("TEMP", str(tmp_path))
    namespace: dict = {"__name__": "__main__"}
    for index, cell in enumerate(_notebook()["cells"]):
        if cell["cell_type"] == "code":
            exec(
                compile("".join(cell["source"]), f"cell-{index}", "exec"),
                namespace,
            )
    followup = namespace["MOCK_FOLLOWUP"]
    assert followup["favorable"]["development"] == "NEW_PROPERTY_DEVELOPMENT_GO"
    assert followup["favorable"]["confirmation"] == "NEW_PROPERTY_CONFIRMATION_GO"
    assert followup["null"]["development"] == "NEW_PROPERTY_DEVELOPMENT_NO_GO"
    assert (
        followup["control_failure"]["development"]
        == "NEW_PROPERTY_DEVELOPMENT_CONTROL_FAILURE"
    )
    assert (
        followup["capability_no_go"]["development"]
        == "NEW_PROPERTY_DEVELOPMENT_CAPABILITY_NO_GO"
    )
    assert len({row["development"] for row in followup.values()}) == 4
    for scenario in ("null", "control_failure", "capability_no_go"):
        assert followup[scenario]["confirmation"] is None
    assert namespace["MOCK_LOCALIZATION"]["label"] == "exploratory"
    assert namespace["MOCK_LOCALIZATION"]["is_confirmation"] is False
    assert (
        namespace["MOCK_LOCALIZATION"]["claim_boundary"]["onset_layer_claimed"] is False
    )
    assert namespace["MOCK_ASYMMETRY"]["cause_of_asymmetry_identified"] is False


def test_notebook_records_why_body_covering_was_withdrawn() -> None:
    source = _source()
    assert "visible in the photograph" in source
    assert "body_covering" in source
    assert "DOMINANT_ANSWER_RULE" in source
    # the spent first attempt is wired into the exclusion universe
    assert "mmnewpropertydev_real_baad443fdc39" in source
    assert "EXTRA_SPENT_REPORT_PATHS = (" in source
