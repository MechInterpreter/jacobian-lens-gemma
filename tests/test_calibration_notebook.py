# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Structure and MOCK execution of the research-grade calibration notebook.

The execution tests run every code cell in one namespace from a working
directory where ``jlens`` is not importable, so they check the bootstrap and the
whole study at once — and, at the committed defaults, that opening the notebook
starts nothing at all.
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
    / "research_grade_multilayer_jlens_calibration_colab.ipynb"
)
RUNNER = Path(__file__).resolve().parent / "_calibration_notebook_runner.py"
BUILDER = REPO_ROOT / "scripts" / "_build_calibration_notebook.py"

REQUIRED_SECTIONS = [
    "bootstrap repository",
    "configuration",
    "mount google drive",
    "install and verify dependencies",
    "authentication",
    "methodology and frozen protocol",
    "corpus audit and deterministic splits",
    "target-token diversity audit",
    "model architecture and hook audit",
    "compute and storage budget",
    "explicit real-run confirmation",
    "collect or resume multilayer calibration data",
    "fit or resume per-layer lenses",
    "validate each layer at the current scale",
    "compare the scale points",
    "apply the predeclared plateau rule",
    "run the untouched final confirmation",
    "publish passing frozen lens artifacts",
    "generate report and resume status",
]

COMMITTED_SWITCHES = (
    "RUN_REAL_CALIBRATION",
    "RUN_MODEL_STAGES",
    "CONFIRM_100_BUDGET",
    "CONFIRM_250_BUDGET",
    "CONFIRM_1K_BUDGET",
    "RUN_FINAL_CONFIRMATION",
    "PUBLISH_VALIDATED_LENSES",
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
    assert [int(number) for number, _ in headings] == list(range(0, 19)), headings
    for (_, title), expected in zip(headings, REQUIRED_SECTIONS, strict=True):
        assert expected in title.lower(), (expected, title)


def test_every_switch_is_committed_false(notebook):
    source = _code_source(notebook)
    for switch in COMMITTED_SWITCHES:
        assert f"{switch} = False" in source, switch
        assert f"{switch} = True" not in source, switch


def test_notebook_matches_its_builder(tmp_path):
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


def test_notebook_states_there_is_no_optimizer(notebook):
    source = _source(notebook)
    assert "no fitting objective and no optimizer" in source
    assert "estimator variance" in source


def test_notebook_states_the_budget_honestly(notebook):
    source = _source(notebook)
    assert "paper-matched 1,000-prompt endpoint" in source
    assert "100 prompts is the upstream usable-scale benchmark" in source
    assert "no scale beyond 1,000" in source
    assert "RUN_OPTIONAL_LARGE_SCALE" not in source
    assert "CONFIRM_OPTIONAL_LARGE_SCALE_BUDGET" not in source


def test_notebook_carries_the_mock_disclaimer(notebook):
    assert "MOCK success proves pipeline behaviour only" in _source(notebook)


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
        "lens.validated.pt",
    ):
        assert forbidden not in source, f"notebook code references {forbidden!r}"


def test_the_only_multimodal_import_is_the_pure_tie_aware_scorer(notebook):
    """Reuse of the scoring function is deliberate; reuse of data is not.

    ``jlens.mmlocalize.lens_validity.tie_aware_row`` is a pure function over two
    logit vectors — sharing it is what makes a number here mean what the same
    number meant in the completed localization run. Any *other* multimodal
    import would be a data path.
    """
    imports = [
        line.strip()
        for line in _code_source(notebook).splitlines()
        if line.strip().startswith(("import ", "from "))
    ]
    multimodal = [line for line in imports if ".mm" in line]
    assert multimodal == [
        "from jlens.mmlocalize.lens_validity import tie_aware_row"
    ], multimodal


def test_real_target_discovery_uses_the_lens_model_protocol(notebook):
    source = _code_source(notebook)
    assert "ordinary_next_token_argmax(" in source
    assert "logits_from_ids" not in source


# --------------------------------------------------------------- execution


def _run_notebook(*overrides, timeout=900):
    environment = dict(os.environ)
    environment["PYTHONPATH"] = ""
    environment["RGCALIB_REPO_DIR"] = str(REPO_ROOT)
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
    assert lines, f"runner produced no JSON:\n{result.stdout[-3000:]}\n{result.stderr[-3000:]}"
    payload = json.loads(lines[-1])
    assert payload["ok"], payload.get("traceback", payload.get("error"))
    return payload


@pytest.fixture(scope="module")
def default_run():
    return _run_notebook()


@pytest.fixture(scope="module")
def full_run():
    return _run_notebook("RUN_FINAL_CONFIRMATION=True", "PUBLISH_VALIDATED_LENSES=True")


def test_committed_defaults_start_nothing(default_run):
    assert default_run["mode"] == "mock"
    assert default_run["model_stages_enabled"] is False
    assert default_run["active_scale_points"] == []
    assert default_run["model_class"] == "MockCalibrationModel"
    assert default_run["confirmation_ran"] is False
    assert default_run["vault_status"]["locked"] is True
    assert default_run["vault_status"]["opened"] is False
    assert default_run["n_published"] is None


def test_bootstrap_makes_jlens_importable_from_the_checkout(default_run):
    assert Path(default_run["jlens_file"]).resolve().parent.parent == REPO_ROOT
    assert Path(default_run["repo_path"]).resolve() == REPO_ROOT
    assert len(default_run["commit"]) == 40


def test_mock_run_uses_the_real_frozen_layer_grid(default_run):
    assert default_run["layers"] == [8, 14, 20, 26, 32, 35, 38, 40]
    assert default_run["active_plan_layers"] == default_run["layers"]
    assert default_run["target_layer"] == 41


def test_splits_are_the_protocol_sizes_and_leak_free(default_run):
    assert default_run["split_sizes"]["validation"] == 128
    assert default_run["split_sizes"]["confirmation"] == 128
    assert default_run["leakage_ok"] is True
    assert default_run["leakage_exact_hits"] == 0
    assert default_run["leakage_near_hits"] == 0
    assert default_run["nesting_exact"] is True
    checksums = set(default_run["split_checksums"].values())
    assert len(checksums) == 3  # three genuinely different partitions


def test_target_diversity_clears_both_clauses(default_run):
    assert default_run["diversity_passed"] is True
    assert default_run["n_distinct_targets"] >= 24
    assert default_run["max_target_share"] <= 0.25


def test_all_layers_are_captured_and_snapshotted(default_run):
    assert default_run["snapshot_scales"] == default_run["scales"]
    for scale, n_prompts in default_run["snapshot_n_prompts"].items():
        assert int(scale) == n_prompts  # a snapshot never misnames its own size
    assert default_run["diagnostic_layers"] == default_run["layers"]


def test_the_four_archetypes_show_up_in_the_scale_table(default_run):
    eligible = {int(k): v for k, v in default_run["eligible_by_scale"].items()}
    scales = sorted(eligible)
    # the late anchor passes everywhere
    assert all(38 in eligible[scale] for scale in scales)
    # the early layer improves with scale and eventually passes
    assert 14 not in eligible[scales[0]]
    assert 14 in eligible[scales[-1]]
    # the degenerate layer and the optimistic-rank-1 layer never pass
    assert all(8 not in eligible[scale] for scale in scales)
    assert all(32 not in eligible[scale] for scale in scales)


def test_comparison_and_plateau_are_computed(default_run):
    n_layers = len(default_run["layers"])
    n_scales = len(default_run["scales"])
    assert default_run["n_comparison_rows"] == n_layers * n_scales
    assert default_run["n_delta_rows"] == n_layers * (n_scales - 1)
    assert default_run["plateau_runs_automatically"] is False


def test_the_optional_extension_does_not_run_even_when_the_rule_fires(default_run):
    assert default_run["plateau_extension_justified"] is True
    assert default_run["plateau_runs_automatically"] is False
    assert default_run["budget"]["scale_points"][-1] == 1_000


def test_budget_is_reported_for_the_real_scale_points(default_run):
    scales = [row["scale"] for row in default_run["budget"]["per_scale"]]
    assert scales == [100, 250, 1_000]
    assert default_run["budget"]["cumulative_not_additive"] is True


def test_report_is_written_and_marked_mock(default_run):
    assert default_run["report_has_mock_banner"] is True
    assert default_run["report_checksum"].startswith("sha256:")


def test_resume_state_is_recorded(default_run):
    assert default_run["resume_status_at_open"] == "starting"
    assert default_run["checkpoint_present"] is True
    units = default_run["completed_units"]
    assert units["scale_snapshot"] == len(default_run["scales"])
    assert units["validation"] == len(default_run["scales"])
    assert units["confirmation"] == 0  # nothing offered the confirmation set
    assert units["publication"] == 0


# --------------------------------------------- the full path, switches on


def test_confirmation_opens_only_after_a_scale_is_selected(full_run):
    status = full_run["vault_status"]
    assert status["locked"] is False and status["opened"] is True
    assert status["selected_scale"] == full_run["selected_scale"]
    assert status["selection_checksum"].startswith("sha256:")


def test_only_confirmed_layers_are_published(full_run):
    passed = {int(k) for k, v in full_run["confirmation_passed"].items() if v}
    assert set(full_run["published_layers"]) == passed
    assert set(full_run["failed_layers"]) == set(full_run["layers"]) - passed
    assert full_run["n_published"] == len(passed)


def test_the_degenerate_and_tie_blocked_layers_are_never_published(full_run):
    for layer in (8, 32):
        assert layer in full_run["failed_layers"]
        assert layer not in full_run["published_layers"]


def test_published_artifacts_exist_on_disk_with_sidecars(full_run):
    run_dir = Path(full_run["run_dir"])
    published = sorted((run_dir / "artifacts" / "published").glob("*.pt"))
    assert len(published) == full_run["n_published"]
    for path in published:
        sidecar = path.with_suffix(".json")
        assert sidecar.is_file()
        artifact = json.loads(sidecar.read_text(encoding="utf-8"))
        assert artifact["validated"] is True
        assert artifact["frozen"] is True
        assert artifact["calibration_modality"] == "text-only"
        assert artifact["spokencoco_used"] is False
        assert artifact["multimodal_data_used"] is False
        assert artifact["objective"] == "not_applicable_estimator_is_a_sample_mean"
        assert artifact["lens_checksum"].startswith("sha256:")
        assert artifact["artifact_checksum"].startswith("sha256:")
    assert list((run_dir / "artifacts" / "published").glob("*.tmp.*")) == []


def test_the_report_records_the_publication(full_run):
    run_dir = Path(full_run["run_dir"])
    report = (run_dir / "artifacts" / "calibration_report.md").read_text(
        encoding="utf-8"
    )
    assert "MOCK RUN" in report
    assert "proves pipeline behaviour only" in report
    assert "Failed layers keep their diagnostics" in report
