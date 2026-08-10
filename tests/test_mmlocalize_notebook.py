# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Structure and MOCK execution of the layer-localization notebook.

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
    REPO_ROOT / "notebooks" / "archive" / "completed_studies" / "multimodal_jspace_layer_localization_colab.ipynb"
)
RUNNER = Path(__file__).resolve().parent / "_mmlocalize_notebook_runner.py"
BUILDER = REPO_ROOT / "scripts" / "_build_localization_notebook.py"

REQUIRED_SECTIONS = [
    "bootstrap repository",
    "configuration",
    "mount google drive",
    "install and verify dependencies",
    "verify the completed robustness run",
    "freeze the localization targets",
    "pass and storage estimate",
    "load and audit gemma",
    "load the frozen lens for the layers under test",
    "text-only layer validity",
    "capability confirmation",
    "capture activations at all four layers",
    "compute j-space codes",
    "paired representational tests",
    "estimate source-only directions",
    "off-diagonal causal interventions",
    "aggregate at the image level",
    "localization verdict and report",
    "resume state",
]


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


def test_the_commissioned_stages_appear_in_order(notebook):
    headings = re.findall(r"^## (\d+)\. (.+)$", _source(notebook), re.MULTILINE)
    assert [int(number) for number, _ in headings] == list(range(1, 20)), headings
    for (_, title), expected in zip(headings, REQUIRED_SECTIONS, strict=True):
        assert expected in title.lower(), (expected, title)


def test_the_committed_notebook_matches_its_generator():
    before = NOTEBOOK_PATH.read_text(encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(BUILDER)], capture_output=True, text=True, cwd=REPO_ROOT
    )
    assert result.returncode == 0, result.stderr
    assert NOTEBOOK_PATH.read_text(encoding="utf-8") == before


def test_every_real_run_flag_defaults_to_false(notebook):
    source = _code_source(notebook)
    for switch in (
        "RUN_REAL_LOCALIZATION",
        "RUN_MODEL_STAGES",
        "CONFIRM_MODEL_PASS_BUDGET",
        "RUN_TEXT_RECALIBRATION",
    ):
        assert re.search(rf"^{switch} = False$", source, re.MULTILINE), switch
        assert not re.search(rf"^{switch} = True$", source, re.MULTILINE), switch


def test_the_layer_set_is_the_predetermined_one_and_is_asserted(notebook):
    source = _code_source(notebook)
    assert "LAYERS = (20, 26, 32, 38)" in source
    assert "assert_immutable_layer_set(LAYERS)" in source
    # The shortcut the brief explicitly rules out.
    assert "LAYERS = (32,)" not in source


def test_the_concepts_are_the_two_that_replicated(notebook):
    source = _code_source(notebook)
    assert "CONCEPTS = LOCALIZATION_CONCEPTS" in source
    prose = " ".join(_source(notebook).split()).lower()
    assert "cat" in prose and "toilet" in prose
    assert "conditioned" in prose


def test_the_model_revision_and_frozen_lens_are_the_validated_triple(notebook):
    source = _code_source(notebook)
    assert 'MODEL_REPO_ID = "google/gemma-4-E4B-it"' in source
    assert "google/gemma-3n-e4b-it" not in source.casefold()
    assert 'MODEL_REVISION = "fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd"' in source
    assert (
        "sha256:4b17bf6086901e633f94d3391f5de6eccd3e735cc24cece63887505d73641c2b"
        in source
    )
    assert "text_jlens_early_layer_recalibration_v2" in source


def test_the_completed_run_is_pinned_by_fingerprint_and_never_written(notebook):
    source = _code_source(notebook)
    assert "mmrobust_robustness_20260804T154417" in source
    assert (
        "sha256:61d0f0e7eb0e2b75831817fa7b9a7f4ebb36d7f4d03fbebce669634390c4c278"
        in source
    )
    assert "verify_completed_run(" in source
    for line in source.splitlines():
        if "COMPLETED_ROBUSTNESS_RUN_DIR" in line or "COMPLETED_EXPANDED" in line:
            assert "write_text" not in line, line
            assert "unlink" not in line, line
    for forbidden in ("shutil.rmtree", "os.remove", "os.unlink"):
        assert forbidden not in source, forbidden


def test_the_run_namespace_is_new_and_cannot_be_the_completed_run(notebook):
    source = _code_source(notebook)
    assert 'f"mmlocalize_{CONFIG.mode}_"' in source
    assert "the completed robustness run; that run is" in source


def test_this_notebook_fits_no_lens(notebook):
    source = _code_source(notebook)
    assert "This notebook does not fit a lens." in source
    assert ".fit(" not in source
    # Recalibration is a plan, never an action taken here.
    assert "format_recalibration_plan()" in source
    assert "check_recalibration_target(" in source


def test_the_validity_gate_is_printed_before_any_result_producing_cell(notebook):
    cells = ["".join(cell["source"]) for cell in _code_cells(notebook)]
    gate = next(i for i, c in enumerate(cells) if "print(gate_text())" in c)
    scoring = next(i for i, c in enumerate(cells) if "tie_aware_row(" in c)
    model = next(i for i, c in enumerate(cells) if "build_real_backend(" in c)
    assert gate < model
    assert gate < scoring


def test_validation_prompts_are_target_diverse_before_lens_scoring(notebook):
    source = _code_source(notebook)
    assert "select_target_diverse_prompts(" in source
    assert "min_distinct_target_tokens=" in source
    assert "LOCALIZATION_VALIDITY_GATE.min_distinct_target_tokens" in source
    assert source.index("select_target_diverse_prompts(") < source.index("tie_aware_row(")
    assert "n_validation_discovery_prompts=POOL_RECORDS" in source


def test_the_targets_are_frozen_before_the_model_cell(notebook):
    cells = ["".join(cell["source"]) for cell in _code_cells(notebook)]
    freeze = next(i for i, c in enumerate(cells) if "freeze_targets(" in c)
    model = next(i for i, c in enumerate(cells) if "build_real_backend(" in c)
    activations = next(i for i, c in enumerate(cells) if "stage_activations(" in c)
    assert freeze < model < activations


def test_the_budget_is_printed_before_the_model_cell(notebook):
    cells = ["".join(cell["source"]) for cell in _code_cells(notebook)]
    budget = next(i for i, c in enumerate(cells) if "format_budget(" in c)
    freeze = next(i for i, c in enumerate(cells) if "freeze_targets(" in c)
    model = next(i for i, c in enumerate(cells) if "build_real_backend(" in c)
    assert freeze < budget < model


def test_the_preflight_runs_before_the_model_is_loaded(notebook):
    cells = ["".join(cell["source"]) for cell in _code_cells(notebook)]
    preflight = next(i for i, c in enumerate(cells) if "localization_preflight(" in c)
    model = next(i for i, c in enumerate(cells) if "build_real_backend(" in c)
    assert preflight < model
    assert "REAL PATH PREFLIGHT" in "".join(
        line for cell in cells for line in cell
    ) or "format_preflight" in cells[preflight]


def test_the_budget_confirmation_gates_every_model_stage(notebook):
    source = _code_source(notebook)
    assert "MODEL_STAGES_ENABLED = bool(RUN_MODEL_STAGES and BUDGET_CONFIRMED)" in source

    cells = ["".join(cell["source"]) for cell in _code_cells(notebook)]
    for needle in (
        "build_real_backend(",
        "localization_preflight(",
        "run_invariance_gate(",
        "load_lens_for_localization(",
        "MockPilotBackend(",
        "tie_aware_row(",
        "stage_capability",
        "stage_activations",
        "stage_codes",
        "stage_representational",
        "stage_directions",
        "stage_causal",
        "getpass.getpass",
        "build_dictionaries",
    ):
        owners = [cell for cell in cells if needle in cell]
        assert owners, needle
        for cell in owners:
            assert "MODEL_STAGES_ENABLED" in cell, needle


def test_the_causal_stage_is_guarded_by_the_eligibility_assertion(notebook):
    cells = ["".join(cell["source"]) for cell in _code_cells(notebook)]
    causal = next(i for i, c in enumerate(cells) if "stage_causal(" in c)
    assert "assert_causally_eligible(" in cells[causal]
    assert "ELIGIBLE_LAYERS" in cells[causal]
    directions = next(i for i, c in enumerate(cells) if "stage_directions(" in c)
    assert "not ELIGIBLE_LAYERS" in cells[directions]


def test_the_representational_stage_runs_at_every_layer_not_only_eligible_ones(notebook):
    cells = ["".join(cell["source"]) for cell in _code_cells(notebook)]
    representational = next(
        i for i, c in enumerate(cells) if "stage_representational(" in c
    )
    assert "for _layer in LAYERS:" in cells[representational]


def test_the_pairing_across_layers_is_asserted_not_assumed(notebook):
    source = _code_source(notebook)
    assert "assert_same_targets_across_layers(" in source


def test_the_invariance_gate_covers_every_layer(notebook):
    source = _code_source(notebook)
    assert "run_invariance_gate(BACKEND, _probe, list(LAYERS))" in source


def test_interpretation_boundaries_are_stated_in_prose(notebook):
    prose = " ".join(_source(notebook).split()).lower()
    assert "not erasure" in prose
    assert "projection ablation" in prose
    assert "earliest tested layer with evidence" in prose
    assert "never the earliest layer in the model" in prose
    assert "eligibility is earned, not declared" in prose
    assert "spoken audio is excluded by design" in prose
    assert "that skip is not a negative result" in prose


def test_the_tie_rationale_is_explained_to_the_reader(notebook):
    prose = " ".join(_source(notebook).split()).lower().replace("`", "").replace("*", "")
    assert "tie block" in prose
    assert "argmax reports the tie-break rule" in prose
    assert "unique top-1 agreements" in prose
    # The reader must be told what the new gate ADDS, not only what it drops.
    assert "adds three blocking clauses" in prose
    assert "drops only the unique-top-1 floor" in prose


def test_bootstrap_comes_before_any_repository_import(notebook):
    cells = ["".join(cell["source"]) for cell in _code_cells(notebook)]
    install = next(
        i for i, c in enumerate(cells) if "pip" in c and "install" in c and "-e" in c
    )
    verify = next(i for i, c in enumerate(cells) if "jlens.__file__" in c)
    first_import = next(
        i for i, c in enumerate(cells) if re.search(r"^\s*from jlens", c, re.MULTILINE)
    )
    assert install < first_import
    assert verify < first_import


def test_bootstrap_defines_only_primitives_and_needs_no_drive(notebook):
    cells = _code_cells(notebook)
    tree = ast.parse("".join(cells[0]["source"]))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            assert isinstance(node.value, ast.Constant) and isinstance(
                node.value.value, str
            )
        else:
            assert isinstance(node, ast.Expr)
    assert not [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]

    sources = ["".join(cell["source"]) for cell in cells]
    verify = next(i for i, c in enumerate(sources) if "jlens.__file__" in c)
    mount = next(i for i, c in enumerate(sources) if "drive.mount" in c)
    assert verify < mount
    for index in range(verify + 1):
        assert "drive" not in sources[index].lower()


# --------------------------------------------------------------- execution


def _clean_environment(tmp_path):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONIOENCODING"] = "utf-8"
    env["MMLOCALIZE_REPO_DIR"] = str(REPO_ROOT)
    env["MMLOCALIZE_SCRATCH"] = str(tmp_path / "scratch")
    env["MMLOCALIZE_RUN_DIR"] = str(tmp_path / "run")
    env["MMLOCALIZE_COMPLETED_RUN_DIR"] = str(tmp_path / "no_completed_run_here")
    return env


def _run_notebook(tmp_path, *overrides, expect_ok=True):
    workdir = tmp_path / "elsewhere"
    workdir.mkdir(exist_ok=True)
    result = subprocess.run(
        [sys.executable, str(RUNNER), str(NOTEBOOK_PATH), *overrides],
        cwd=workdir,
        env=_clean_environment(tmp_path),
        capture_output=True,
        text=True,
        timeout=3600,
    )
    if expect_ok:
        assert result.returncode == 0, (
            f"stdout:\n{result.stdout[-5000:]}\nstderr:\n{result.stderr[-3000:]}"
        )
    return json.loads(result.stdout.strip().splitlines()[-1]), result.stdout


def test_the_test_environment_really_is_clean(tmp_path):
    probe = subprocess.run(
        [sys.executable, "-c", "import jlens"],
        cwd=tmp_path,
        env=_clean_environment(tmp_path),
        capture_output=True,
        text=True,
    )
    assert probe.returncode != 0
    assert "No module named 'jlens'" in probe.stderr


@pytest.fixture(scope="module")
def cpu_path(tmp_path_factory):
    return _run_notebook(tmp_path_factory.mktemp("localize_cpu"))


def test_the_committed_defaults_freeze_and_cost_but_load_nothing(cpu_path):
    report, stdout = cpu_path

    assert report["ok"]
    assert report["run_real"] is False
    assert report["run_model_stages"] is False
    assert report["run_text_recalibration"] is False
    assert report["budget_confirmed"] is False
    assert report["model_stages_enabled"] is False
    assert report["model_is_none"], "no model may load at the committed defaults"
    assert report["verdict"] is None
    assert report["overrides"] == {}
    assert "NOTHING RAN" in stdout
    assert "MODEL STAGES BLOCKED" in stdout


def test_the_layer_set_and_gate_are_printed_before_anything_runs(cpu_path):
    _, stdout = cpu_path
    assert "PREDETERMINED LAYER SET" in stdout
    assert "PREDECLARED LAYER-VALIDITY GATE" in stdout
    assert "MIDRANK" in stdout
    assert "WHY THIS DIFFERS FROM THE OLD GATE" in stdout
    # The gate must appear before the budget, which appears before any model.
    assert stdout.index("PREDECLARED LAYER-VALIDITY GATE") < stdout.index(
        "MODEL PASS BUDGET"
    )


def test_the_targets_are_frozen_with_a_checksum_and_an_audit(cpu_path):
    report, stdout = cpu_path

    assert report["target_checksum"].startswith("sha256:")
    assert report["target_manifest_checksum"].startswith("sha256:")
    assert report["n_target_images"] > 0
    assert report["n_source_images"] > 0
    audit = report["exclusion_audit"]
    assert audit["source_target_overlap"] == []
    assert audit["n_overlap_all"] == 0
    assert audit["fresh_policy_satisfied"] is True
    assert "FROZEN LOCALIZATION TARGETS" in stdout


def test_the_layers_are_the_predetermined_four(cpu_path):
    report, _ = cpu_path
    assert report["layers"] == [20, 26, 32, 38]
    assert report["reference_layer"] == 38
    assert report["concepts"] == ["cat", "toilet"]


def test_the_budget_separates_capture_from_causal_cost(cpu_path):
    report, stdout = cpu_path
    budget = report["budget"]

    assert budget["n_layers_captured"] == 4
    # Four layers, one pass each: capture does not multiply by layers.
    assert budget["activation_passes"] == budget["n_total_groups"] * 2
    assert budget["total_passes"] == (
        budget["validation_target_discovery_passes"]
        + budget["text_validation_passes"]
        + budget["capability_passes"]
        + budget["activation_passes"]
        + budget["causal_clean_passes"]
        + budget["causal_intervention_passes"]
    )
    assert budget["recalibration_passes"] == 0
    assert "ONE forward pass records all of them" in stdout


def test_recalibration_is_off_and_advertised_as_a_separate_choice(cpu_path):
    _, stdout = cpu_path
    assert "recalibration: DISABLED" in stdout
    assert "never overwritten" in stdout or "is never overwritten" in stdout


@pytest.fixture(scope="module")
def full_path(tmp_path_factory):
    return _run_notebook(
        tmp_path_factory.mktemp("localize_full"),
        "RUN_MODEL_STAGES=True",
        "CONFIRM_MODEL_PASS_BUDGET=True",
    )


def test_the_full_mock_study_runs_to_a_verdict(full_path):
    report, stdout = full_path

    assert report["ok"]
    assert Path(report["jlens_file"]).resolve().parent.parent == REPO_ROOT
    assert report["run_real"] is False, "the real study must stay off"
    assert report["model_stages_enabled"] is True
    assert report["model_is_none"], "the real Gemma must not load in MOCK"
    assert report["verdict"] in (
        "EARLY_TRANSFER_CONFIRMED",
        "LATE_ONLY_SUPPORTED",
        "INCONCLUSIVE_LAYER_LOCALIZATION",
    )
    assert "VERDICT:" in stdout
    assert report["resume_status"] == "starting"
    assert report["invariance_passed"] is True
    assert "call-signature contracts were checked and all bind" in stdout


def test_stage_b_scores_every_layer_with_every_variant(full_path):
    report, stdout = full_path

    assert set(report["validity_status"]) == {"20", "26", "32", "38"}
    # 32 prompts x 4 layers x 5 variants.
    assert report["n_validation_rows"] == 32 * 4 * 5
    assert report["rank_conventions"] == ["optimistic", "pessimistic", "midrank"]
    assert report["validity_gate_digest"].startswith("sha256:")
    # ASCII-only substring: the Windows console mangles the em dash in the
    # banner, and this test is about the stage running, not about encoding.
    assert "LAYER VALIDITY (midrank criterion; both gates reported)" in stdout


def test_both_gates_are_reported_for_every_layer(full_path):
    report, stdout = full_path
    assert set(report["legacy_gate_passed"]) == {"20", "26", "32", "38"}
    assert "old gate:" in stdout and "new gate:" in stdout
    assert "gates agree:" in stdout


def test_only_eligible_layers_were_intervened_on(full_path):
    """The skip is structural: an ineligible layer produces no intervention."""
    report, _ = full_path
    assert report["intervention_layers"] == report["eligible_layers"]


def test_the_targets_are_identical_at_every_intervened_layer(full_path):
    report, stdout = full_path
    assert report["criteria_status"]["targets_identical_at_every_layer"] == "PASS"
    if report["intervention_layers"]:
        assert "paired across layers" in stdout


def test_the_paired_depth_contrast_is_produced(full_path):
    report, _ = full_path
    if len(report["eligible_layers"]) > 1:
        assert report["paired_comparison_rows"] > 0


def test_every_intervention_records_its_layer_and_image_identity(full_path):
    report, _ = full_path
    fields = set(report["intervention_fields"])
    for required in (
        "layer",
        "image_id",
        "image_identity_rule_version",
        "target_is_positive",
        "source_split",
        "target_split",
        "unrelated_control_concept",
        "signed_target_effect",
        "signed_margin_effect",
        "activation_norm_ratio",
        "prediction_changed",
    ):
        assert required in fields, required


def test_the_run_fingerprint_binds_the_whole_localization_configuration(full_path):
    report, _ = full_path
    selection = report["selection_fingerprint"]

    assert selection["layer_set_version"].startswith("mmlocalize.layers")
    assert selection["layer_manifest"]["physical_layers"] == [20, 26, 32, 38]
    assert selection["validity_gate_digest"].startswith("sha256:")
    assert selection["target_checksum"] == report["target_checksum"]
    assert selection["localization_concepts"] == ["cat", "toilet"]
    assert selection["source_image_ids"] and selection["target_image_ids"]
    assert not set(selection["source_image_ids"]) & set(selection["target_image_ids"])
    assert len(selection["validation_prompt_hashes"]) == 32
    assert len(selection["validation_prompt_target_ids"]) == 32
    assert selection["validation_prompt_selection_protocol"].startswith(
        "target-token-stratified"
    )
    assert selection["validation_prompt_selection_checksum"].startswith("sha256:")
    assert report["fingerprint_digest"].startswith("sha256:")


def test_the_report_and_summary_are_written(full_path):
    report, _ = full_path
    run_dir = Path(report["run_dir"])
    for name in (
        "localization_report.md",
        "localization_summary.json",
        "localization_target_manifest.json",
        "image_exclusion_audit.json",
        "layer_validity.json",
        "split_provenance.json",
        "pass_budget.json",
    ):
        assert (run_dir / name).is_file(), name
    text = (run_dir / "localization_report.md").read_text(encoding="utf-8")
    assert "~normalized [48, 62, 76, 90]" in text
    assert "not scientific evidence" in text     # MOCK banner


def test_a_second_pass_resumes_rather_than_recomputing(tmp_path_factory):
    root = tmp_path_factory.mktemp("localize_resume")
    first, _ = _run_notebook(
        root, "RUN_MODEL_STAGES=True", "CONFIRM_MODEL_PASS_BUDGET=True"
    )
    second, stdout = _run_notebook(
        root, "RUN_MODEL_STAGES=True", "CONFIRM_MODEL_PASS_BUDGET=True"
    )

    assert first["resume_status"] == "starting"
    assert second["resume_status"] == "resuming"
    assert second["fingerprint_digest"] == first["fingerprint_digest"]
    assert second["verdict"] == first["verdict"]
    assert second["target_checksum"] == first["target_checksum"]
    assert "0 computed" in stdout, "a resumed run must reuse, not recompute"


def test_an_incompatible_configuration_refuses_to_resume(tmp_path_factory, monkeypatch):
    """A run directory built under one layer set, gate or target set is never
    silently reused under another."""
    from jlens.mmpilot.store import IncompatibleStateError, RunFingerprint, UnitStore

    root = tmp_path_factory.mktemp("localize_incompatible")
    base = dict(
        mode="localization",
        model_repo_id="google/gemma-4-E4B-it",
        model_revision="rev",
        processor_revision="rev",
        layers=(20, 26, 32, 38),
        lens_checksum="sha256:lens",
        manifest_checksum="sha256:manifest",
        split_id="spokencoco-localization-v1",
        intervention_config={"causal_layers": [38]},
    )
    original = RunFingerprint(
        **base, selection_config={"validity_gate_digest": "sha256:gate-a"}
    )
    UnitStore(root, original).open()

    for changed in (
        RunFingerprint(**base, selection_config={"validity_gate_digest": "sha256:gate-b"}),
        RunFingerprint(**{**base, "layers": (26, 32, 38)},
                       selection_config={"validity_gate_digest": "sha256:gate-a"}),
        RunFingerprint(**base, selection_config={"target_checksum": "sha256:other"}),
    ):
        with pytest.raises(IncompatibleStateError, match="different configuration"):
            UnitStore(root, changed).open()
