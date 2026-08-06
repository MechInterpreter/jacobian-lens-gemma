# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Structure and MOCK execution of the early-layer extension notebook.

The execution tests run every code cell in one namespace from a working
directory where ``jlens`` is not importable, so they check the bootstrap and the
whole extension at once — and, at the committed defaults, that opening the
notebook starts nothing at all.
"""

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = (
    REPO_ROOT
    / "notebooks"
    / "research_grade_early_layer_jlens_extension_colab.ipynb"
)
RUNNER = Path(__file__).resolve().parent / "_early_layer_extension_notebook_runner.py"
BUILDER = REPO_ROOT / "scripts" / "_build_early_layer_extension_notebook.py"
CONFIG_PATH = (
    REPO_ROOT / "configs" / "research_grade_early_layer_jlens_extension_v1.json"
)
PARENT_CONFIG_PATH = (
    REPO_ROOT / "configs" / "research_grade_jlens_calibration_v1.json"
)

REQUIRED_SECTIONS = [
    "bootstrap repository",
    "configuration and staged switches",
    "mount google drive",
    "install and verify dependencies",
    "authentication",
    "frozen protocol, gate and selection rule",
    "parent-run audit and read-only import",
    "model architecture and hook audit",
    "reconstruct the fit ordering and verify the parent prefix",
    "completely fresh development and confirmation sets",
    "target-token diversity audit",
    "compute and storage budget",
    "explicit real-run confirmation",
    "continue the fit from the parent accumulator",
    "evaluate the fresh development set at every scale",
    "compare the scale points and apply the plateau rule",
    "freeze the scale selection",
    "run the untouched final confirmation",
    "publish validated earlier lenses",
    "early-layer verdict, report, and the parent-immutability proof",
]

COMMITTED_SWITCHES = (
    "RUN_REAL_EARLY_LAYER_EXTENSION",
    "RUN_MODEL_STAGES",
    "CONFIRM_PARENT_IMPORT",
    "CONFIRM_250_BUDGET",
    "CONFIRM_1K_BUDGET",
    "RUN_FRESH_DEVELOPMENT",
    "RUN_FINAL_CONFIRMATION",
    "PUBLISH_VALIDATED_EARLY_LENSES",
)


@pytest.fixture(scope="module")
def notebook():
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def _code_cells(payload):
    return [cell for cell in payload["cells"] if cell["cell_type"] == "code"]


def _source(payload):
    return "\n".join("".join(cell["source"]) for cell in payload["cells"])


def _code_source(payload):
    return "\n".join("".join(cell["source"]) for cell in _code_cells(payload))


# --------------------------------------------------------------- structure


def test_notebook_is_valid_and_output_free(notebook):
    assert notebook["nbformat"] == 4
    for index, cell in enumerate(_code_cells(notebook)):
        assert cell["outputs"] == [], f"code cell {index} has stored outputs"
        assert cell["execution_count"] is None, f"code cell {index} has an exec count"


def test_all_code_cells_parse(notebook):
    for index, cell in enumerate(_code_cells(notebook)):
        try:
            ast.parse("".join(cell["source"]))
        except SyntaxError as exc:  # pragma: no cover - failure path
            pytest.fail(f"code cell {index} does not parse: {exc}")


def test_the_commissioned_sections_appear_in_order(notebook):
    headings = re.findall(r"^## (\d+)\. (.+)$", _source(notebook), re.MULTILINE)
    assert [int(number) for number, _ in headings] == list(range(0, 20)), headings
    for (_, title), expected in zip(headings, REQUIRED_SECTIONS, strict=True):
        assert expected in title.lower(), (expected, title)


def test_every_switch_is_committed_false(notebook):
    source = _code_source(notebook)
    for switch in COMMITTED_SWITCHES:
        assert f"{switch} = False" in source, switch
        assert f"{switch} = True" not in source, switch


def test_notebook_matches_its_builder():
    """Regenerate from source and fail on drift."""
    result = subprocess.run(
        [sys.executable, str(BUILDER)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr
    regenerated = subprocess.run(
        ["git", "diff", "--exit-code", "--", str(NOTEBOOK_PATH.relative_to(REPO_ROOT))],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert regenerated.returncode == 0, (
        "the committed notebook does not match its builder; run "
        f"`python {BUILDER.relative_to(REPO_ROOT)}`\n{regenerated.stdout[:3000]}"
    )


def test_notebook_states_the_evidentiary_boundary_up_front(notebook):
    source = re.sub(r"\s+", " ", _source(notebook))
    assert (
        "The scale-100 fitting accumulator is reusable. The scale-100 "
        "confirmation set is not." in source
    )
    assert "not reused, not relabelled and not reset" in source
    assert "it is **excluded** from every new split" in source
    assert "bit-identical" in source


def test_notebook_states_there_is_no_optimizer(notebook):
    source = _source(notebook)
    assert "There is no optimizer and no loss" in source
    assert "not_applicable_estimator_is_a_sample_mean" in _code_source(notebook)


def test_notebook_states_the_budget_as_an_extrapolation(notebook):
    source = _source(notebook)
    assert "7.1 minutes on one L4" in source
    assert "extrapolation, not measurement" in source.lower()
    assert "incremental, not cumulative" in source
    assert "no scale beyond 1,000" in source


def test_notebook_carries_the_mock_disclaimer(notebook):
    source = _source(notebook)
    assert "MOCK success proves pipeline behaviour only" in source
    assert "about layer 32, about layer 26" in source


def test_notebook_never_reads_multimodal_run_results(notebook):
    # The branch name legitimately contains "spokencoco"; it is the checkout to
    # clone, not data to read.
    source = _code_source(notebook).replace(
        "experiment/spokencoco-jspace-pilot", "<branch>"
    )
    for forbidden in (
        "spokencoco",
        "SpokenCOCO",
        "mmpilot_pilot_",
        "mmrobust_",
        "mmlocalize_localization_",
        "audioaudit_",
        "image",
        "audio",
        "autoencoder",
    ):
        assert forbidden not in source, f"notebook code references {forbidden!r}"


def test_the_only_multimodal_import_is_the_pure_tie_aware_scorer(notebook):
    imports = [
        line.strip()
        for line in _code_source(notebook).splitlines()
        if line.strip().startswith(("import ", "from "))
    ]
    multimodal = [line for line in imports if ".mm" in line]
    assert multimodal == [
        "from jlens.mmlocalize.lens_validity import tie_aware_row"
    ], multimodal


def test_the_notebook_never_writes_into_the_parent_run(notebook):
    source = _code_source(notebook)
    # every write goes through the extension store or the extension run dir
    assert "STORE.save(" in source
    assert "RUN_DIR" in source
    for pattern in (
        "PARENT_ROOT /",
        "PARENT.root +",
        'open(PARENT',
    ):
        assert pattern not in source, pattern
    assert "protected_parent_checksums" in source
    assert "assert_parent_unchanged" in source


def test_real_target_discovery_uses_the_lens_model_protocol(notebook):
    source = _code_source(notebook)
    assert "ordinary_next_token_argmax(" in source
    assert "logits_from_ids" not in source


def test_the_selection_is_recorded_before_the_vault_is_opened(notebook):
    source = _code_source(notebook)
    assert source.index('STORE.save("scale_selection"') < source.index(
        "VAULT.unlock(SELECTION)"
    )
    assert "ConfirmationVault(records=SPLITS.confirmation)" in source


# ------------------------------------------------------------ configuration


def test_the_frozen_config_does_not_touch_the_completed_study():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    parent_config = json.loads(PARENT_CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["protocol_version"] != parent_config["protocol_version"]
    assert config["extends_config"] == "configs/research_grade_jlens_calibration_v1.json"
    assert config["extends_config_is_unmodified"] is True
    assert (
        parent_config["protocol_version"]
        == "research-grade-multilayer-text-jlens-calibration-v2-feasible-scales"
    )


def test_the_config_states_every_commissioned_protocol_fact():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    boundary = config["evidentiary_boundary"]
    assert boundary["old_confirmation_set_status"] == "DEVELOPMENT_HISTORY_SPENT"
    assert boundary["old_development_set_status"] == "DEVELOPMENT_HISTORY_SPENT"
    assert "Never reused, never relabelled, never reset" in (
        boundary["old_confirmation_set_handling"]
    )
    assert boundary["parent_run_is_read_only"] is True
    assert boundary["parent_run_written_by_this_extension"] is False
    assert config["scale"]["candidate_scales"] == [250, 1000]
    assert config["scale"]["baseline_scale"] == 100
    assert config["parent_run"]["baseline_scale_status"] == "DESCRIPTIVE_ONLY"
    assert config["why_this_exists"]["primary_earlier_layer"] == 32
    assert config["why_this_exists"]["secondary_earlier_layer"] == 26
    assert config["why_this_exists"]["descriptive_layers"] == [8, 14, 20]
    assert config["why_this_exists"]["already_published_layers"] == [35, 38, 40]
    assert config["validation"]["nothing_is_loosened"] is True
    assert config["validation"]["frozen_before_any_extension_result"] is True
    assert config["corpus"]["multimodal_data_in_fitting"] is False
    assert config["corpus"]["multimodal_data_in_lens_validation"] is False
    assert config["switches"]["every_real_switch_defaults_to_false"] is True
    for switch in COMMITTED_SWITCHES:
        assert config["switches"][switch] is False, switch
    for flag, value in config["scope_boundaries"].items():
        assert value is False, flag


def test_the_config_thresholds_match_the_frozen_gate():
    from jlens.calibration.extension import EXTENSION_GATE

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    thresholds = config["validation"]["thresholds"]
    gate = EXTENSION_GATE.to_dict()
    for key, value in thresholds.items():
        assert gate[key] == value, key
    assert config["validation"]["n_prompts"] == EXTENSION_GATE.n_prompts
    assert config["validation"]["gate_protocol"] == EXTENSION_GATE.version


# --------------------------------------------------------------- execution


def _run_notebook(*overrides, timeout=1800):
    environment = dict(os.environ)
    environment["PYTHONPATH"] = ""
    environment["RGEXT_REPO_DIR"] = str(REPO_ROOT)
    workdir = REPO_ROOT.parent
    result = subprocess.run(
        [sys.executable, str(RUNNER), str(NOTEBOOK_PATH), *overrides],
        capture_output=True,
        text=True,
        cwd=workdir,
        env=environment,
        timeout=timeout,
    )
    lines = [line for line in result.stdout.splitlines() if line.startswith("{")]
    assert lines, (
        f"runner produced no JSON:\n{result.stdout[-3000:]}\n{result.stderr[-3000:]}"
    )
    payload = json.loads(lines[-1])
    assert payload["ok"], payload.get("traceback", payload.get("error"))
    return payload


@pytest.fixture(scope="module")
def default_run():
    return _run_notebook()


@pytest.fixture(scope="module")
def full_run():
    return _run_notebook(
        "RUN_FINAL_CONFIRMATION=True", "PUBLISH_VALIDATED_EARLY_LENSES=True"
    )


def test_committed_defaults_start_nothing(default_run):
    assert default_run["mode"] == "mock"
    assert default_run["model_stages_enabled"] is False
    assert default_run["active_scale_points"] == []
    assert default_run["confirmation_ran"] is False
    assert default_run["vault_status"]["locked"] is True
    assert default_run["vault_status"]["opened"] is False
    assert default_run["n_published"] is None
    assert default_run["verdict"] is None


def test_bootstrap_makes_jlens_importable_from_the_checkout(default_run):
    assert Path(default_run["jlens_file"]).resolve().parent.parent == REPO_ROOT
    assert Path(default_run["repo_path"]).resolve() == REPO_ROOT
    assert len(default_run["commit"]) == 40


def test_the_mock_run_uses_the_real_frozen_grid_and_gate(default_run):
    assert default_run["layers"] == [8, 14, 20, 26, 32, 35, 38, 40]
    assert default_run["active_plan_layers"] == default_run["layers"]
    assert default_run["publishable_layers"] == [26, 32]
    assert default_run["gate_n_prompts"] == 256
    assert default_run["gate_min_distinct_targets"] == 32
    assert default_run["gate_digest"].startswith("sha256:")
    assert default_run["protocol_digest"].startswith("sha256:")
    assert default_run["selection_rule_digest"].startswith("sha256:")


def test_the_parent_is_imported_read_only_and_proved_unchanged(default_run):
    assert default_run["parent_audit_compatible"] is True
    assert default_run["parent_audit_failed_checks"] == []
    assert default_run["parent_confirmation_vault"]["opened"] is True
    assert default_run["parent_immutable"] is True
    assert default_run["parent_files_checked"] >= 6
    assert default_run["seed_action"] == "seeded"
    assert default_run["seed_parent_written"] is False
    assert default_run["parent_accumulator_checksum"].startswith("sha256:")
    assert Path(default_run["parent_root"]) != Path(default_run["run_dir"])


def test_the_fit_prefix_is_verified_before_the_skip(default_run):
    assert default_run["reconstruction_all_match"] is True
    assert default_run["prefix_matches"] is True
    assert default_run["prefix_skip_authorized"] is True


def test_the_continuation_reaches_every_scale_and_equals_a_fresh_fit(default_run):
    assert default_run["snapshot_scales"] == default_run["scales"]
    for scale, n_prompts in default_run["snapshot_n_prompts"].items():
        assert int(scale) == n_prompts  # a snapshot never misnames its own size
    assert default_run["n_fitted"] == max(default_run["scales"])
    equivalence = default_run["equivalence"]
    assert equivalence["bit_identical"] is True
    assert equivalence["differences"] == []


def test_the_fresh_sets_are_the_protocol_sizes_and_leak_free(default_run):
    assert default_run["split_sizes"]["development"] == 256
    assert default_run["split_sizes"]["confirmation"] == 256
    assert default_run["split_leakage_ok"] is True
    assert len(set(default_run["split_checksums"].values())) == 2
    assert set(default_run["excluded_exact"]) == {
        "old_fit",
        "old_development",
        "old_confirmation",
        "new_fit",
    }


def test_target_diversity_clears_the_stricter_floor(default_run):
    assert default_run["diversity_passed"] is True
    assert default_run["confirmation_diversity_passed"] is True
    assert default_run["n_distinct_targets"] >= 32
    assert default_run["max_target_share"] <= 0.25
    assert default_run["confirmation_selected_by_jlens"] is False


def test_the_baseline_scale_is_scored_but_not_selectable(default_run):
    assert len(default_run["development_scales"]) == 3
    assert default_run["development_scales_scored"] == default_run["development_scales"]
    assert default_run["selected_scale"] in default_run["scales"]
    assert default_run["development_scales"][0] not in default_run["scales"]


def test_the_selection_is_made_without_confirmation(default_run):
    assert default_run["selection_confirmation_not_consulted"] is True
    assert default_run["selection_clause"] in (
        "fallback_to_largest",
        "smallest_scale_matching_largest",
        "no_early_layer_development_pass",
    )


def test_resume_state_is_recorded(default_run):
    assert default_run["resume_status_at_open"] == "starting"
    assert default_run["checkpoint_present"] is True
    units = default_run["completed_units"]
    assert units["parent_import"] == 2  # provenance and the immutability proof
    assert units["fresh_splits"] == 1
    assert units["continuation"] == 1
    assert units["scale_selection"] == 1
    assert units["scale_snapshot"] == len(default_run["scales"])
    assert units["validation"] == len(default_run["development_scales"])
    assert units["confirmation"] == 0  # nothing offered the confirmation set
    assert units["publication"] == 0
    assert units["early_layer_verdict"] == 0


# --------------------------------------------- the full path, switches on


def test_confirmation_opens_only_after_a_scale_is_selected(full_run):
    status = full_run["vault_status"]
    assert status["locked"] is False and status["opened"] is True
    assert status["selected_scale"] == full_run["selected_scale"]
    assert status["selection_checksum"].startswith("sha256:")


def test_the_default_scenario_publishes_only_the_confirmed_early_layer(full_run):
    assert full_run["scenario"] == "l32_late_pass"
    assert full_run["verdict"] == "EARLY_LAYER_CALIBRATION_GO"
    assert full_run["verdict_early_layers"] == [32]
    assert full_run["published_layers"] == [32]
    assert full_run["failed_layers"] == [26]
    assert full_run["confirmation_passed"]["32"] is True
    assert full_run["confirmation_passed"]["26"] is False


def test_established_layers_pass_confirmation_but_are_not_republished(full_run):
    for layer in ("35", "38", "40"):
        assert full_run["confirmation_passed"][layer] is True
        assert int(layer) not in full_run["published_layers"]


def test_published_artifacts_carry_the_parent_provenance(full_run):
    run_dir = Path(full_run["run_dir"])
    published = sorted((run_dir / "artifacts" / "published").glob("*.pt"))
    assert len(published) == full_run["n_published"]
    for path in published:
        extension = json.loads(
            path.with_suffix(".extension.json").read_text(encoding="utf-8")
        )
        assert extension["validated"] is True
        assert extension["frozen"] is True
        assert extension["parent_accumulator_checksum"] == (
            full_run["parent_accumulator_checksum"]
        )
        assert extension["parent_accumulator_n_done"] == full_run["parent_n_done"]
        assert extension["old_confirmation_set_reused"] is False
        assert extension["parent_run_written"] is False
        assert extension["development_split_checksum"] == (
            full_run["split_checksums"]["development"]
        )
        assert extension["confirmation_split_checksum"] == (
            full_run["split_checksums"]["confirmation"]
        )
        assert extension["existing_publications_unchanged"] == [35, 38, 40]
        assert extension["artifact_checksum"].startswith("sha256:")
        base = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        assert base["validated"] is True
        assert base["calibration_modality"] == "text-only"
    assert list((run_dir / "artifacts" / "published").glob("*.tmp.*")) == []


def test_the_run_report_and_the_provenance_manifest_are_written(full_run):
    run_dir = Path(full_run["run_dir"])
    report = json.loads(
        (run_dir / "artifacts" / "early_layer_extension_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["mock_proves_pipeline_only"] is True
    assert report["parent_immutability_proof"]["immutable"] is True
    assert report["early_layer_verdict"]["verdict"] == "EARLY_LAYER_CALIBRATION_GO"
    assert report["scale_selection"]["confirmation_not_consulted"] is True
    provenance = json.loads(
        (run_dir / "artifacts" / "parent_provenance.json").read_text(encoding="utf-8")
    )
    assert provenance["read_only"] is True
    assert provenance["parent_written_by_this_extension"] is False
    assert "already opened" in provenance["not_imported"]["old_confirmation_set"]


def test_the_parent_is_still_untouched_after_publication(full_run):
    assert full_run["parent_immutable"] is True


# ------------------------------------------- the other commissioned outcomes


@pytest.mark.parametrize(
    ("scenario", "verdict", "published"),
    [
        ("l32_confirmation_fail", "EARLY_LAYER_CALIBRATION_NO_GO", []),
        ("no_early_layer", "EARLY_LAYER_CALIBRATION_NO_GO", []),
    ],
)
def test_the_negative_scenarios_publish_nothing(scenario, verdict, published):
    payload = _run_notebook(
        "RUN_FINAL_CONFIRMATION=True",
        "PUBLISH_VALIDATED_EARLY_LENSES=True",
        f"MOCK_SCENARIO={scenario}",
    )
    assert payload["scenario"] == scenario
    assert payload["verdict"] == verdict
    assert payload["published_layers"] == published
    assert payload["verdict_early_layers"] == []
    assert "did not yield a validated earlier readout" in payload["verdict_statement"]
    run_dir = Path(payload["run_dir"])
    assert list((run_dir / "artifacts" / "published").glob("*.pt")) == []
