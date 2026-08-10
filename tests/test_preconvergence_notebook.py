# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""The L27-L31 pre-convergence notebook: structure, a MOCK run, and resume.

The MOCK run is executed in a **clean interpreter** from a working directory
outside the repository with ``PYTHONPATH`` cleared, so the notebook's own
bootstrap has to make ``jlens`` importable — the same thing a Colab session
does. What it proves is pipeline behaviour: that every stage runs, every gate
holds, every refusal fires where it should, and an interruption at any stage
resumes instead of restarting. It proves nothing about Gemma.
"""

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = (
    REPO_ROOT
    / "notebooks"
    / "research_grade_l27_l31_preconvergence_study_colab.ipynb"
)
BUILDER = REPO_ROOT / "scripts" / "_build_l27_l31_preconvergence_study_notebook.py"
RUNNER = REPO_ROOT / "tests" / "_preconvergence_notebook_runner.py"

#: Every switch that must be committed False.
SWITCHES = (
    "RUN_REAL_PRECONVERGENCE_STUDY",
    "PREPROCESSING_ONLY",
    "RUN_LENS_FITTING",
    "CONFIRM_FITTING_BUDGET",
    "RUN_UNTOUCHED_CONFIRMATION",
    "RUN_MODEL_STAGE",
    "CONFIRM_MODEL_LOAD",
    "CONFIRM_STAGE_3_BUDGET",
    "RUN_STAGE_4_CAUSAL_TRANSFER",
    "CONFIRM_STAGE_4_BUDGET",
    "ALLOW_MANIFEST_REBUILD",
)

#: Cell indices used by the interruption tests. Checked by
#: ``test_the_interruption_points_are_where_the_tests_think_they_are`` so a
#: reordered notebook fails loudly instead of silently testing the wrong thing.
STOP_AFTER = {
    "preprocessing": (14, "run_exclusion_preparation("),
    "fitting": (23, "run_calibration("),
    "capability": (30, "stage_capability("),
    "readout": (33, "run_single_layer_convergence("),
    "causal": (35, "stage_causal("),
}


@pytest.fixture(scope="module")
def notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def code_cells(notebook):
    return [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]


@pytest.fixture(scope="module")
def joined(code_cells):
    return "\n".join(code_cells)


def _short_root(name: str) -> Path:
    """A SHORT scratch root.

    Windows still refuses paths past ~260 characters, and the mock completed
    runs nest a run name, a stage and a unit filename under whatever root pytest
    hands out. A tmp_path here produces paths that are legal on Linux and
    unopenable on Windows, which is a test failing for an irrelevant reason.
    """
    import tempfile

    root = Path(tempfile.gettempdir()) / name
    return root


def _run(root: Path, *arguments, stop_after=None):
    root.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment["MMPILOT_REPO_DIR"] = str(REPO_ROOT)
    environment["MMPILOT_SCRATCH"] = str(root / "s")
    environment["MMPILOT_RUNS_ROOT"] = str(root / "r")
    command = [sys.executable, str(RUNNER), str(NOTEBOOK)]
    if stop_after is not None:
        command += ["--stop-after", str(stop_after)]
    command += list(arguments)
    completed = subprocess.run(
        command, cwd=root, env=environment, capture_output=True, text=True
    )
    lines = [line for line in completed.stdout.splitlines() if line.startswith("{")]
    payload = json.loads(lines[-1]) if lines else {"ok": False, "stdout": completed.stdout}
    if not payload.get("ok"):
        payload.setdefault("stderr", completed.stderr[-3000:])
    return payload


# ============================================================ structure


def test_notebook_is_valid_and_output_free(notebook):
    assert notebook["nbformat"] == 4
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []
            assert cell["execution_count"] is None


def test_the_committed_notebook_matches_its_builder():
    import importlib.util

    spec = importlib.util.spec_from_file_location("_builder", BUILDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = json.dumps(module.build(), indent=1, ensure_ascii=False) + "\n"
    assert NOTEBOOK.read_text(encoding="utf-8") == expected, (
        "the committed notebook is not byte-identical to its generator output; "
        f"run `python {BUILDER.relative_to(REPO_ROOT)}`"
    )


def test_every_code_cell_parses(code_cells):
    for source in code_cells:
        ast.parse(source)


@pytest.mark.parametrize("switch", SWITCHES)
def test_every_switch_is_committed_false(switch, code_cells):
    """The *assignment* must be False. Guidance text may of course say "set it
    to True" — what matters is what a fresh open of the notebook executes."""
    assignments = [
        line.strip()
        for source in code_cells
        for line in source.splitlines()
        if line.startswith(f"{switch} = ")
    ]
    assert assignments, f"{switch} is never assigned"
    assert set(assignments) == {f"{switch} = False"}


def test_the_candidate_interval_is_never_widened_in_the_notebook(joined):
    assert "(27, 28, 29, 30, 31)" in joined
    for outside in ("33", "34"):
        assert f"ADJACENT_CANDIDATE_LAYERS + ({outside}" not in joined


def test_the_frozen_criterion_digest_is_checked_not_recomputed(joined):
    assert "FROZEN_CRITERION_DIGEST" in joined
    assert "criterion_digest_matches" in joined
    assert 'CONVERGENCE["criterion_digest"] != FROZEN_CRITERION_DIGEST' in joined


def test_the_gate_identity_is_asserted_before_the_download(code_cells):
    predownload = [c for c in code_cells if "PRE-DOWNLOAD CHECKS" in c]
    assert len(predownload) == 1
    assert "ADJACENT_GATE is EXTENSION_GATE" in predownload[0]
    assert "check_preconvergence_call_contracts" in predownload[0]
    assert "check_call_contracts" in predownload[0]


def test_the_predownload_cell_precedes_the_model_load(code_cells):
    predownload = next(
        i for i, c in enumerate(code_cells) if "PRE-DOWNLOAD CHECKS" in c
    )
    model = next(i for i, c in enumerate(code_cells) if "build_real_backend(" in c)
    assert predownload < model


def test_every_expensive_cell_rederives_its_gates_from_the_raw_switches(code_cells):
    for source in code_cells:
        spends = any(
            marker in source
            for marker in (
                "run_calibration(",
                "stage_capability(",
                "stage_activations(",
                "stage_causal(",
                "build_real_backend(",
            )
        )
        if spends:
            assert "refresh_gates()" in source, source[:200]


def test_the_three_expensive_stages_each_have_their_own_confirmation(joined):
    for switch, marker in (
        ("CONFIRM_FITTING_BUDGET", "Stage 1 costs"),
        ("CONFIRM_MODEL_LOAD", "~16 GB"),
        ("CONFIRM_STAGE_3_BUDGET", "Stage 3 "),
        ("CONFIRM_STAGE_4_BUDGET", "Stage 4 costs"),
    ):
        assert switch in joined
        assert marker in joined


def test_the_pin_for_the_completed_resolution_run_is_empty_and_never_discovered(
    joined,
):
    assert 'COMPLETED_RESOLUTION_RUN_DIR = ""' in joined
    assert "assert_completed_population_pins" in joined
    for forbidden in ("glob(", "sorted(", "max("):
        assert f'{forbidden}"mml32res' not in joined


def test_no_lens_other_than_this_study_s_own_is_loaded(joined):
    assert "discover_published_l32_lens" not in joined
    assert "load_published_lenses" not in joined
    assert "run_calibration(" in joined


def test_the_parent_accumulator_is_never_seeded(joined):
    assert "seed_extension_checkpoint" not in joined
    assert "assert_new_source_layers" in joined
    assert '"parent_accumulator_seeded": False' in joined


def test_steering_is_never_renamed_as_a_coordinate_swap(joined):
    lowered = joined.lower()
    assert "coordinate_swap_scope" in lowered
    # The only occurrences of "swap" are in the out-of-scope statement.
    assert "coordinate_swap(" not in lowered
    assert "apply_coordinate_swap" not in lowered


def test_the_leakage_audit_is_run_and_persisted_per_modality(joined):
    assert "audited_separately_per_modality" in joined
    for modality in ("text", "image", "spoken_audio"):
        assert f'"{modality}"' in joined
    assert "candidate_leakage_audit.json" in joined
    assert "assert_prompt_leakage_clean" in joined


def test_all_seven_causal_controls_are_named_in_the_notebook(joined):
    from jlens.mmpilot.preconvergence import REQUIRED_CAUSAL_CONTROLS

    for control in REQUIRED_CAUSAL_CONTROLS:
        assert control in joined


def test_completed_runs_are_named_as_protected_and_proven_unchanged(joined):
    assert "PROTECTED_RUN_PREFIXES" in joined
    for prefix in ("mml32res_", "mml32_", "mmaudio_", "rgext_", "rgcalib_"):
        assert f'"{prefix}"' in joined
    assert "assert_sources_unchanged" in joined
    assert "assert_parent_unchanged" in joined


def test_the_interruption_points_are_where_the_tests_think_they_are(code_cells):
    for name, (index, marker) in STOP_AFTER.items():
        assert marker in code_cells[index - 1], f"{name}: cell {index - 1} moved"


# ============================================================ the MOCK run


@pytest.fixture(scope="module")
def mock_run():
    root = _short_root("pcnb_main")
    import shutil

    shutil.rmtree(root, ignore_errors=True)
    payload = _run(root)
    assert payload.get("ok"), payload
    return payload


def test_the_mock_notebook_runs_end_to_end(mock_run):
    assert mock_run["n_cells"] == 38
    assert mock_run["MODE"] == "mock"
    assert mock_run["mock_proves_pipeline_only"] is True


def test_the_predownload_checks_pass_before_anything_expensive(mock_run):
    assert mock_run["PREDOWNLOAD.passed"] is True
    assert mock_run["PREDOWNLOAD.gate_is_extension_gate"] is True
    assert mock_run["PREDOWNLOAD.criterion_digest_matches"] is True
    assert mock_run["PREDOWNLOAD.shared_contract_failures"] == []
    assert mock_run["PREDOWNLOAD.study_contract_failures"] == []


def test_the_accumulator_is_new_and_the_parent_is_not_seeded(mock_run):
    assert mock_run["SOURCE_LAYERS.disjoint"] is True
    assert mock_run["SOURCE_LAYERS.overlap"] == []
    assert mock_run["SOURCE_LAYERS.parent_accumulator_may_be_seeded"] is False
    assert mock_run["FIT_RECORD.parent_accumulator_seeded"] is False
    assert mock_run["FIT_RECORD.layers"] == [27, 28, 29, 30, 31]


def test_the_confirmation_set_is_full_size_and_proved_untouched(mock_run):
    assert mock_run["CONFIRMATION_MANIFEST.size"] == 256
    assert mock_run["UNTOUCHED_AUDIT.untouched"] is True
    assert mock_run["UNTOUCHED_AUDIT.n_exact_hits"] == 0
    assert mock_run["UNTOUCHED_AUDIT.n_near_hits"] == 0
    assert mock_run["UNTOUCHED_AUDIT.n_internal_duplicates"] == 0
    assert set(mock_run["UNTOUCHED_AUDIT.required_disjoint_from"]) == {
        "parent_fit",
        "parent_development",
        "parent_confirmation_opened",
        "adjacent_fit",
        "extension_development_reused_here",
        "extension_confirmation_opened",
    }


def test_the_development_set_is_reused_with_its_role_recorded(mock_run):
    assert mock_run["DEVELOPMENT_ROLE.reused"] is True
    assert mock_run["DEVELOPMENT_ROLE.role_then"] == "development"
    assert mock_run["DEVELOPMENT_ROLE.role_now"] == "development"
    assert mock_run["DEVELOPMENT_ROLE.checksum"].startswith("sha256:")


def test_all_three_spent_populations_are_pinned_and_excluded(mock_run):
    assert mock_run["POPULATION_PINS.n_excluded_runs"] == 3
    assert mock_run["POPULATION_PINS.pin_was_discovered"] is False
    assert mock_run["POPULATION_PINS.pin_was_defaulted"] is False
    assert mock_run["POPULATION_PINS.pinned_resolution_run"].startswith("mml32res")


def test_the_population_is_proven_disjoint_in_every_identity_family(mock_run):
    assert mock_run["DISJOINTNESS.disjoint"] is True
    assert mock_run["DISJOINTNESS.failed_families"] == []
    assert set(mock_run["DISJOINTNESS.n_overlaps"]) >= {
        "image_ids", "group_ids", "audio_paths", "captions",
    }
    assert all(count == 0 for count in mock_run["DISJOINTNESS.n_overlaps"].values())


def test_the_exclusion_really_excluded_something(mock_run):
    counts = mock_run["exclusion_counts"]
    assert counts["image_ids"] > 0
    assert counts["group_ids"] > 0
    assert counts["audio_paths"] > 0


def test_there_is_one_photograph_and_one_recording_per_unit(mock_run):
    assert mock_run["PSEUDOREPLICATION.passed"] is True
    assert (
        mock_run["PSEUDOREPLICATION.n_units"]
        == mock_run["PSEUDOREPLICATION.n_distinct_images"]
        == mock_run["PSEUDOREPLICATION.n_distinct_recordings"]
    )


def test_every_candidate_is_evaluated_and_the_earliest_passer_is_selected(mock_run):
    assert mock_run["SELECTION.candidates"] == [27, 28, 29, 30, 31]
    assert mock_run["SELECTION.evaluated_layers"] == [27, 28, 29, 30, 31]
    assert mock_run["SELECTION.passing_layers"] == [29, 31]
    assert mock_run["SELECTED_LAYER"] == 29
    assert mock_run["LENS_VERDICT.failed_validity_clauses"] == []


def test_the_complete_table_is_reported_including_the_failures(mock_run):
    table = mock_run["confirmation_table"]
    assert [row["layer"] for row in table] == [27, 28, 29, 30, 31]
    failing = [row for row in table if not row["passed"]]
    assert len(failing) == 3
    assert all(row["failed_clauses"] for row in failing)


def test_the_prompt_never_lists_the_candidates(mock_run):
    question = mock_run["OPEN_QUESTION"].lower()
    for concept in mock_run["SELECTED_NAMES"]:
        assert concept.lower() not in question
    assert mock_run["LEAKAGE_AUDIT.passed"] is True
    assert mock_run["LEAKAGE_AUDIT.candidate_order_invariant"] is True
    assert mock_run["LEAKAGE_AUDIT.candidate_set_moves_fingerprint"] is True


def test_the_convergence_controls_all_ran_and_all_passed(mock_run):
    assert mock_run["CONTROLS_RECORD.passed"] is True
    assert mock_run["CONTROLS_RECORD.missing_or_empty"] == []
    assert mock_run["CONTROLS_RECORD.failing"] == []


def test_the_mock_head_agreement_is_recorded_as_not_run_never_as_passed(mock_run):
    assert mock_run["HEAD_AGREEMENT.comparison_ran"] is False


def test_the_terminal_outcome_is_one_of_the_six(mock_run):
    from jlens.mmpilot.preconvergence import TERMINAL_OUTCOMES

    assert mock_run["VERDICTS.terminal_outcome"] in TERMINAL_OUTCOMES
    assert mock_run["primary_verdict"] == mock_run["VERDICTS.terminal_outcome"]


def test_the_mock_world_converges_so_the_claim_is_ruled_out_not_manufactured(mock_run):
    """The MOCK decoder converges by construction, and the study says so."""
    assert mock_run["CONVERGENCE_VERDICT.verdict"] == "LAYER_CONVERGED"
    assert mock_run["VERDICTS.terminal_outcome"] == "CONVERGED_BEFORE_CAUSAL_TEST"
    assert mock_run["verdict.PRECONVERGENCE_CAUSAL_TRANSFER"] == "NOT_EVALUATED"


def test_stage_four_does_not_run_when_its_gate_is_closed(mock_run):
    assert mock_run["STAGE_4.runs"] is False
    assert mock_run["STAGE_4.gate_met"] is False
    assert "same_layer_not_converged_in_every_required_modality" in (
        mock_run["STAGE_4.failed_gate_clauses"]
    )


def test_the_criterion_thresholds_were_not_changed(mock_run):
    assert mock_run["CONVERGENCE_VERDICT.criterion_thresholds_unchanged"] is True


def test_no_concept_was_replaced(mock_run):
    assert mock_run["concepts_replaced_after_results"] is False
    assert mock_run["FEASIBILITY.all_feasible"] is True
    assert mock_run["FEASIBILITY.infeasible_concepts"] == []


def test_the_cache_is_loaded_without_rebuilding_the_join(mock_run):
    assert mock_run["CACHE_LOAD.build_expanded_manifest_called"] is False
    assert mock_run["CACHE_LOAD.compatible"] is True


def test_the_selection_survives_a_permuted_manifest(mock_run):
    assert mock_run["SELECTION_DETERMINISM.deterministic"] is True


def test_the_completed_runs_and_the_parent_were_not_written_to(mock_run):
    assert mock_run["IMMUTABILITY.unchanged"] is True
    assert mock_run["IMMUTABILITY.appeared"] == []
    assert mock_run["IMMUTABILITY.modified"] == []
    assert mock_run["PARENT_IMMUTABILITY.immutable"] is True


def test_the_run_directory_is_in_this_study_s_own_family(mock_run):
    assert Path(mock_run["RUN_DIR"]).name.startswith("mmpre_")


def test_every_required_artifact_is_written(mock_run):
    assert set(mock_run["artifact_names"]) >= {
        "l27_l31_preconvergence_summary.json",
        "l27_l31_preconvergence_report.md",
        "adjacent_lens_table.json",
        "population_manifest.json",
        "disjointness_audit.json",
        "candidate_leakage_audit.json",
        "convergence_tables.json",
        "convergence_controls.json",
        "exclusion_completeness_proof.json",
        "run_state.json",
        "completed_run_immutability.json",
    }


def test_the_mock_run_is_deterministic():
    root = _short_root("pcnb_det")
    import shutil

    shutil.rmtree(root, ignore_errors=True)
    first = _run(root)
    assert first.get("ok"), first
    shutil.rmtree(root, ignore_errors=True)
    second = _run(root)
    assert second.get("ok"), second
    for key in (
        "POPULATION_DIGEST",
        "POOL_DIGEST",
        "RANKING_DIGEST",
        "SELECTED_LAYER",
        "lens_fingerprint_digest",
        "exclusion_digest",
        "VERDICTS.terminal_outcome",
    ):
        assert first[key] == second[key], key
    # `preparation_digest` is deliberately NOT compared across a rebuilt MOCK
    # cache: the MOCK path writes the expanded manifest itself, and
    # `persist_expanded_manifest` stamps it with `created_utc`, so the cache
    # FILE checksum — which the preparation fingerprint binds on purpose —
    # differs between two builds. On the real path the cache is a pre-existing
    # Drive file and the digest is stable; the same-cache case is asserted by
    # `test_preprocessing_only_creates_no_run_directory_and_loads_no_model`.


# ======================================================== the override path


@pytest.fixture(scope="module")
def mock_run_stage_four():
    root = _short_root("pcnb_s4")
    import shutil

    shutil.rmtree(root, ignore_errors=True)
    payload = _run(
        root,
        "RUN_STAGE_4_CAUSAL_TRANSFER=True",
        "CONFIRM_STAGE_4_BUDGET=True",
    )
    assert payload.get("ok"), payload
    return payload


def test_an_overridden_stage_four_runs_and_is_stamped_as_overridden(
    mock_run_stage_four,
):
    assert mock_run_stage_four["STAGE_4.runs"] is True
    assert mock_run_stage_four["STAGE_4.gate_met"] is False
    assert mock_run_stage_four["STAGE_4.gate_overridden"] is True
    assert mock_run_stage_four["STAGE_4.evidence_status"] == "DESCRIPTIVE_ONLY"


def test_all_seven_causal_controls_were_recorded(mock_run_stage_four):
    assert "CAUSAL_CONTROLS.passed" in mock_run_stage_four
    assert mock_run_stage_four["CAUSAL_CONTROLS.missing_or_empty"] == []


def test_an_overridden_stage_four_never_supports_the_principal_claim(
    mock_run_stage_four,
):
    assert mock_run_stage_four["verdict.PRECONVERGENCE_CAUSAL_TRANSFER"] in (
        "DESCRIPTIVE_ONLY",
        "NOT_EVALUATED",
        "NOT_SUPPORTED",
    )
    assert mock_run_stage_four["VERDICTS.terminal_outcome"] != (
        "PRECONVERGENCE_CAUSAL_TRANSFER_SUPPORTED"
    )
    assert "stage_4_causal_report.json" in mock_run_stage_four["artifact_names"]


# ======================================================== preprocessing only


def test_preprocessing_only_creates_no_run_directory_and_loads_no_model():
    root = _short_root("pcnb_prep")
    import shutil

    shutil.rmtree(root, ignore_errors=True)
    payload = _run(root, "PREPROCESSING_ONLY=True")
    assert payload.get("ok"), payload
    assert payload["MODEL_STAGE_ENABLED"] is False
    assert payload["FITTING_ENABLED"] is False
    assert payload["CONFIRMATION_ENABLED"] is False
    assert payload.get("RUN_DIR") is None
    assert payload.get("SELECTED_LAYER") is None
    assert Path(payload["PREP_DIR"]).is_dir()
    # And then a fresh GPU-style session reuses that preparation.
    resumed = _run(root)
    assert resumed.get("ok"), resumed
    assert resumed["PREP.files_reused_from_drive"] > 0
    assert resumed["PREP.files_computed_this_session"] == 0
    assert resumed["preparation_digest"] == payload["preparation_digest"]


# ============================================================ interruption


@pytest.mark.parametrize(
    "stage", ["preprocessing", "fitting", "capability", "readout", "causal"]
)
def test_an_interruption_at_every_stage_resumes_instead_of_restarting(stage):
    root = _short_root(f"pcnb_int_{stage}")
    import shutil

    shutil.rmtree(root, ignore_errors=True)
    stop_after, _marker = STOP_AFTER[stage]
    arguments = (
        ("RUN_STAGE_4_CAUSAL_TRANSFER=True", "CONFIRM_STAGE_4_BUDGET=True")
        if stage == "causal"
        else ()
    )
    interrupted = _run(root, *arguments, stop_after=stop_after)
    assert interrupted.get("ok"), interrupted

    resumed = _run(root, *arguments)
    assert resumed.get("ok"), resumed

    # Nothing is recomputed that was already durable.
    assert resumed["PREP.files_computed_this_session"] == 0
    assert resumed["PREP.files_reused_from_drive"] > 0
    if stage != "preprocessing":
        assert resumed["RESUME.fit_n_done"] == resumed["ADJACENT_SCALE"]
        assert resumed["LENS_RESUME_STATE"] in ("resuming", "resumed")
    if stage in ("readout", "causal"):
        assert resumed["RESUME.units_reused"] > 0
        assert resumed["RUN_STATE"] == "resuming"
    # And the conclusion is the same one the uninterrupted run reaches.
    assert resumed["SELECTED_LAYER"] == 29
    assert resumed["VERDICTS.terminal_outcome"] in (
        "CONVERGED_BEFORE_CAUSAL_TEST",
        "CAUSAL_TRANSFER_NOT_SUPPORTED",
    )


def test_a_changed_scientific_configuration_refuses_the_resume():
    """A different candidate set must not resume this run's directory."""
    from jlens.calibration.adjacent import ADJACENT_PROTOCOL
    from jlens.calibration.state import CalibrationFingerprint

    def fingerprint(layers):
        return CalibrationFingerprint(
            mode="mock",
            protocol_version=ADJACENT_PROTOCOL.version,
            model_repo_id="m",
            model_revision="r",
            tokenizer_revision="r",
            capture_plan_digest="d",
            corpus_manifest_checksum="c",
            gate_digest="g",
            plateau_rule_digest="p",
            scale_points=(250,),
            artifact_format_version="v",
            extra={"candidate_layers": list(layers)},
        )

    from jlens.calibration.adjacent import AdjacentStore

    root = _short_root("pcnb_fp") / "lens"
    import shutil

    shutil.rmtree(root.parent, ignore_errors=True)
    AdjacentStore(root, fingerprint((27, 28, 29, 30, 31))).open()
    with pytest.raises(Exception, match="different configuration"):
        AdjacentStore(root, fingerprint((27, 28))).open()
