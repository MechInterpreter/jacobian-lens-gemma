# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Structure and MOCK execution of the six-concept robustness notebook.

The execution tests run every code cell in one namespace from a working
directory where ``jlens`` is not importable, so they check the bootstrap and
the whole study at once — and, at the committed defaults, that opening the
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
    REPO_ROOT / "notebooks" / "archive" / "completed_studies" / "multimodal_jspace_spokencoco_robustness_colab.ipynb"
)
RUNNER = Path(__file__).resolve().parent / "_mmpilot_robustness_notebook_runner.py"
BUILDER = REPO_ROOT / "scripts" / "_build_robustness_notebook.py"

REQUIRED_SECTIONS = [
    "bootstrap repository",
    "configuration",
    "mount google drive",
    "install and verify dependencies",
    "load the derived cache and audit media",
    "select six concepts and build the unique-image subset",
    "pass and storage estimate",
    "load and audit gemma",
    "capability gate",
    "validate the frozen lens",
    "extract layer-38 activations",
    "compute j-space codes",
    "image-disjoint representational tests",
    "estimate source-only directions",
    "off-diagonal causal interventions",
    "aggregate at the image level",
    "robustness verdict and report",
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


def test_the_eighteen_commissioned_stages_appear_in_order(notebook):
    headings = re.findall(r"^## (\d+)\. (.+)$", _source(notebook), re.MULTILINE)
    assert [int(number) for number, _ in headings] == list(range(1, 19)), headings
    for (_, title), expected in zip(headings, REQUIRED_SECTIONS, strict=True):
        assert expected in title.lower(), (expected, title)


def test_the_committed_notebook_matches_its_generator():
    before = NOTEBOOK_PATH.read_text(encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(BUILDER)], capture_output=True, text=True, cwd=REPO_ROOT
    )
    assert result.returncode == 0, result.stderr
    assert NOTEBOOK_PATH.read_text(encoding="utf-8") == before


def test_real_backend_import_names_the_implemented_entry_point(notebook):
    """The real branch calls the tested package function, not ad-hoc notebook
    code — that function is what the fake-real-path test executes."""
    source = _source(notebook)
    from jlens.mmpilot import real_backend

    assert "from jlens.mmpilot.real_backend import build_real_backend" in source
    assert "from jlens.mmpilot.real_backend import load_validated_lens" in source
    assert "Gemma4PilotBackend" not in source
    assert callable(real_backend.build_real_backend)
    assert callable(real_backend.load_validated_lens)


def test_confirmed_real_path_explicitly_allows_the_guarded_model_load(notebook):
    source = _source(notebook)
    assert "allow_model_load=True" in source
    assert source.index("if not MODEL_STAGES_ENABLED") < source.index(
        "allow_model_load=True"
    )


def test_model_repo_revision_and_frozen_lens_are_the_validated_triple(notebook):
    source = _source(notebook)
    assert 'MODEL_REPO_ID = "google/gemma-4-E4B-it"' in source
    assert "google/gemma-3n-e4b-it" not in source.casefold()
    assert 'MODEL_REVISION = "fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd"' in source
    assert "text_jlens_early_layer_recalibration_v2/artifacts/lens.validated.pt" in source


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


def test_every_real_run_flag_defaults_to_false(notebook):
    source = _code_source(notebook)
    for switch in (
        "RUN_REAL_ROBUSTNESS",
        "RUN_MODEL_STAGES",
        "TINY_SMOKE",
        "ENABLE_SPOKEN_AUDIO",
        "CONFIRM_MODEL_PASS_BUDGET",
    ):
        assert re.search(rf"^{switch} = False$", source, re.MULTILINE), switch
        assert not re.search(rf"^{switch} = True$", source, re.MULTILINE), switch


def test_the_budget_confirmation_gates_every_model_stage(notebook):
    """The single guarantee: no forward pass happens before the user has read
    the printed cost and said yes to it."""
    source = _code_source(notebook)
    assert "MODEL_STAGES_ENABLED = bool(RUN_MODEL_STAGES and BUDGET_CONFIRMED)" in source

    cells = ["".join(cell["source"]) for cell in _code_cells(notebook)]
    for needle in (
        "build_real_backend(",
        "real_path_preflight(",
        "run_invariance_gate(",
        "load_validated_lens(",
        "MockPilotBackend(",
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


def test_the_budget_is_printed_before_the_model_cell(notebook):
    cells = ["".join(cell["source"]) for cell in _code_cells(notebook)]
    budget = next(i for i, c in enumerate(cells) if "format_budget(" in c)
    model = next(i for i, c in enumerate(cells) if "build_real_backend(" in c)
    selection = next(i for i, c in enumerate(cells) if "build_subset(" in c)
    assert selection < budget < model
    # Ranking and the subset share one cell; the order inside it still matters.
    assert cells[selection].index("rank_concepts(") < cells[selection].index(
        "build_subset("
    )


def test_the_focal_concepts_are_fixed_before_the_model_cell(notebook):
    cells = ["".join(cell["source"]) for cell in _code_cells(notebook)]
    focal = next(i for i, c in enumerate(cells) if "select_focal_concepts(" in c)
    controls = next(i for i, c in enumerate(cells) if "unrelated_control_assignment(" in c)
    model = next(i for i, c in enumerate(cells) if "build_real_backend(" in c)
    capability = next(i for i, c in enumerate(cells) if "stage_capability" in c)
    assert focal < model and controls < model
    assert focal < capability and controls < capability


def test_the_design_is_the_commissioned_one(notebook):
    source = _code_source(notebook)
    for expected in (
        "N_CONCEPTS = 6",
        "N_FOCAL_CONCEPTS = 3",
        "LAYERS = (38,)",
        "CAUSAL_LAYERS = (38,)",
        "ALPHAS = (0.0, 0.25, 0.5)",
        "CAPABILITY_THRESHOLD = 0.7",
        "N_TRAIN_POSITIVE_IMAGES = 8",
        "N_TEST_POSITIVE_IMAGES = 8",
        "N_TRAIN_NEGATIVE_IMAGES = 8",
        "N_TEST_NEGATIVE_IMAGES = 8",
        'subset_profile="image_unique"',
        "image_unique_targets=True",
        "off_diagonal_causal_only=True",
    ):
        assert expected in source, expected


def test_the_validated_lens_is_pinned_and_never_fitted(notebook):
    source = _code_source(notebook)
    assert "text_jlens_early_layer_recalibration_v2" in source
    assert (
        "sha256:4b17bf6086901e633f94d3391f5de6eccd3e735cc24cece63887505d73641c2b"
        in source
    )
    assert "fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd" in source
    assert "validate_lens" in source
    # The checksum guard now lives in the tested loader the notebook calls.
    assert "load_validated_lens(" in source
    assert "expect_checksum=LENS_EXPECT_SHA256" in source
    import inspect

    from jlens.mmpilot import real_backend

    assert "lens other than the validated one" in inspect.getsource(real_backend)
    assert "This notebook does not fit a lens." in source
    assert ".fit(" not in source


def test_the_completed_pilot_is_read_only_and_never_rewritten(notebook):
    source = _code_source(notebook)
    assert "COMPLETED_PILOT_RUN_DIR" in source
    assert "mmpilot_pilot_20260803T160711" in source
    # The pilot's run directory appears only as a read source.
    for line in source.splitlines():
        if "COMPLETED_PILOT_RUN_DIR" in line or "PILOT_EXPANDED" in line:
            assert "write_text" not in line, line
            assert "unlink" not in line, line
    for forbidden in ("shutil.rmtree", "os.remove", "os.unlink"):
        assert forbidden not in source, forbidden


def test_spoken_audio_stays_out_of_the_study(notebook):
    source = _code_source(notebook)
    assert "ENABLE_SPOKEN_AUDIO = False" in source
    assert "spoken audio is outside this study by design" in source
    prose = " ".join(_source(notebook).split()).lower()
    assert "environmental audio is not tested" in prose


def test_interpretation_boundaries_are_stated_in_prose(notebook):
    prose = " ".join(_source(notebook).split()).lower()
    assert "not erasure" in prose
    assert "projection ablation" in prose
    assert "layer 38 is late in the decoder" in prose
    assert "pre-convergence semantics" in prose
    assert "the decision is replication, not the strongest cell" in prose
    assert "never replaced" in prose or "never quietly swapped" in prose


def test_a_failed_concept_is_never_replaced_after_the_fact(notebook):
    source = _code_source(notebook)
    assert "ROBUSTNESS-PROFILE CAPABILITY NO-GO" in source
    assert "selecting on the outcome" in source


# --------------------------------------------------------------- execution


def _clean_environment(tmp_path):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONIOENCODING"] = "utf-8"
    env["MMPILOT_REPO_DIR"] = str(REPO_ROOT)
    env["MMPILOT_SCRATCH"] = str(tmp_path / "scratch")
    env["MMPILOT_RUN_DIR"] = str(tmp_path / "run")
    env["MMPILOT_PILOT_RUN_DIR"] = str(tmp_path / "no_pilot_here")
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
        timeout=2400,
    )
    if expect_ok:
        assert result.returncode == 0, (
            f"stdout:\n{result.stdout[-4000:]}\nstderr:\n{result.stderr[-3000:]}"
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
    return _run_notebook(tmp_path_factory.mktemp("robust_cpu"))


def test_the_committed_defaults_select_and_cost_but_load_nothing(cpu_path):
    report, stdout = cpu_path

    assert report["ok"]
    assert report["run_real"] is False
    assert report["run_model_stages"] is False
    assert report["budget_confirmed"] is False
    assert report["model_stages_enabled"] is False
    assert report["model_is_none"], "no model may load at the committed defaults"
    assert report["verdict"] is None
    assert report["overrides"] == {}
    assert "NOTHING RAN" in stdout
    assert "CONFIRM_MODEL_PASS_BUDGET = True" in stdout
    assert "MODEL STAGES BLOCKED" in stdout


def test_the_cpu_path_selects_six_concepts_in_ranking_order(cpu_path):
    report, _ = cpu_path

    assert len(report["selected_concepts"]) == 6
    ranked = report["ranked_concepts"]
    positions = [ranked.index(name) for name in report["selected_concepts"]]
    assert positions == sorted(positions), "ranking order was not preserved"
    assert report["selected_concepts"] == ranked[:6]


def test_the_focal_concepts_are_the_first_three_and_controls_are_external(cpu_path):
    report, _ = cpu_path

    assert report["focal_concepts"] == report["selected_concepts"][:3]
    assert report["non_focal_concepts"] == report["selected_concepts"][3:]
    assert set(report["unrelated_controls"]) == set(report["focal_concepts"])
    assert all(
        control in report["non_focal_concepts"]
        for control in report["unrelated_controls"].values()
    )


def test_the_subset_is_one_group_per_photograph(cpu_path):
    report, _ = cpu_path

    assert report["n_groups"] == report["n_distinct_images"]
    assert report["n_siblings_excluded"] > 0, "the sibling captions must be recorded"
    assert report["subset_rows_with_sibling_provenance"] == report["n_groups"]
    assert report["leakage_ok"]


def test_the_budget_is_computed_from_the_real_subset(cpu_path):
    report, stdout = cpu_path
    budget = report["budget"]

    assert budget["n_candidates"] == 6
    assert budget["n_causal_cells"] == 6
    assert budget["n_conditions_per_target"] == 9
    assert budget["total_passes"] == (
        budget["capability_passes"]
        + budget["activation_passes"]
        + budget["causal_clean_passes"]
        + budget["causal_intervention_passes"]
    )
    assert "TOTAL model forward passes" in stdout
    assert "estimated Drive footprint" in stdout


@pytest.fixture(scope="module")
def full_path(tmp_path_factory):
    return _run_notebook(
        tmp_path_factory.mktemp("robust_full"),
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
        "ROBUSTNESS_GO",
        "ROBUSTNESS_WEAK_GO",
        "ROBUSTNESS_NO_GO",
    )
    assert "VERDICT:" in stdout
    assert report["resume_status"] == "starting"
    # The invariance gate runs on the MOCK backend too, and the MOCK branch of
    # the preflight cell still binds every real-path call signature.
    assert report["invariance_passed"] is True
    assert "call-signature contracts were checked and all bind" in stdout
    assert "invariance gate passed: True" in stdout


def test_all_six_concepts_clear_the_mock_capability_gate(full_path):
    report, stdout = full_path
    assert report["criteria_status"]["six_way_capability"] == "PASS"
    assert "retained (text+image): " in stdout


def test_the_mock_study_produced_every_unit_kind(full_path):
    report, _ = full_path
    units = report["completed_units"]

    assert units["capability"] > 0
    assert units["activation"] == report["budget"]["estimated_units"]["activation"]
    assert units["jspace"] == units["activation"]
    assert units["intervention"] == report["budget"]["estimated_units"]["intervention"]


def test_image_level_aggregation_is_a_no_op_under_the_unique_image_profile(full_path):
    """The selection repair's whole point: nothing needs averaging back
    together, because nothing was double-counted in the first place."""
    report, stdout = full_path

    assert report["n_pseudoreplicated_rows"] == 0
    assert "no aggregation was needed to make the unit honest" in stdout


def test_every_intervention_records_its_image_identity_and_versions(full_path):
    report, _ = full_path
    fields = set(report["intervention_fields"])

    for required in (
        "group_id",
        "image_id",
        "image_identity_rule_version",
        "target_is_positive",
        "source_split",
        "target_split",
        "target_selection_version",
        "unrelated_control_concept",
    ):
        assert required in fields, required


def test_the_run_fingerprint_carries_the_whole_selection_config(full_path):
    report, _ = full_path
    selection = report["selection_fingerprint"]

    assert selection["n_candidates_scored"] == 6
    assert selection["max_groups_per_image"] == 1
    assert selection["independent_unit"] == "image_id"
    assert selection["focal_concepts"] == report["focal_concepts"]
    assert selection["off_diagonal_causal_only"] is True
    assert report["fingerprint_digest"].startswith("sha256:")


def test_only_off_diagonal_causal_cells_were_run(full_path):
    report, _ = full_path
    # 3 focal x 2 source modalities x 1 target modality x 16 targets x 9 conditions.
    assert report["completed_units"]["intervention"] == 3 * 2 * 16 * 9


def test_a_second_pass_resumes_rather_than_recomputing(tmp_path_factory):
    root = tmp_path_factory.mktemp("robust_resume")
    first, _ = _run_notebook(root, "RUN_MODEL_STAGES=True", "CONFIRM_MODEL_PASS_BUDGET=True")
    second, stdout = _run_notebook(
        root, "RUN_MODEL_STAGES=True", "CONFIRM_MODEL_PASS_BUDGET=True"
    )

    assert first["resume_status"] == "starting"
    assert second["resume_status"] == "resuming"
    assert second["fingerprint_digest"] == first["fingerprint_digest"]
    assert second["verdict"] == first["verdict"]
    assert "0 computed" in stdout, "a resumed run must reuse, not recompute"
