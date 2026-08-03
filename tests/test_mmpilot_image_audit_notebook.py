# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Structure and CPU execution of the image-independence audit notebook.

The execution tests run every code cell in one namespace from a working
directory where ``jlens`` is not importable, so they check the bootstrap and
the audit at once — and, at the committed defaults, that opening the notebook
starts nothing.
"""

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from jlens.mmpilot import mock as K

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = (
    REPO_ROOT / "notebooks" / "mmpilot_image_independence_audit_colab.ipynb"
)
RUNNER = Path(__file__).resolve().parent / "_mmpilot_audit_notebook_runner.py"
BUILDER = REPO_ROOT / "scripts" / "_build_audit_notebook.py"

REQUIRED_SECTIONS = [
    "colab bootstrap",
    "configuration",
    "mount google drive",
    "run the audit",
    "amended verdict",
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


# ------------------------------------------------------------- structure


def test_notebook_is_valid_and_output_free(notebook):
    assert notebook["nbformat"] == 4
    for index, cell in enumerate(_code_cells(notebook)):
        assert cell["outputs"] == [], f"code cell {index} has stored outputs"
        assert cell["execution_count"] is None, f"code cell {index} has an exec count"


def test_all_code_cells_parse_and_use_no_shell_magics(notebook):
    for index, cell in enumerate(_code_cells(notebook)):
        try:
            ast.parse("".join(cell["source"]))
        except SyntaxError as exc:  # pragma: no cover - failure path
            pytest.fail(f"code cell {index} does not parse: {exc}")


def test_sections_appear_in_the_required_order(notebook):
    headings = re.findall(r"^## (\d+)\. (.+)$", _source(notebook), re.MULTILINE)
    assert [int(number) for number, _ in headings] == list(range(0, 5)), headings
    for (_, title), expected in zip(headings, REQUIRED_SECTIONS, strict=True):
        assert expected in title.lower(), (expected, title)


def test_the_committed_notebook_matches_its_generator(tmp_path):
    """The notebook is generated, so a hand edit that skipped the generator
    would silently diverge from the source of truth."""
    before = NOTEBOOK_PATH.read_text(encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(BUILDER)], capture_output=True, text=True, cwd=REPO_ROOT
    )
    assert result.returncode == 0, result.stderr
    assert NOTEBOOK_PATH.read_text(encoding="utf-8") == before


def test_bootstrap_cells_come_before_any_repository_import(notebook):
    cells = ["".join(cell["source"]) for cell in _code_cells(notebook)]
    install = next(
        i for i, cell in enumerate(cells) if "pip" in cell and "install" in cell and "-e" in cell
    )
    verify = next(i for i, cell in enumerate(cells) if "jlens.__file__" in cell)
    first_repo_import = next(
        i for i, cell in enumerate(cells) if re.search(r"^\s*from jlens", cell, re.MULTILINE)
    )
    assert install < first_repo_import, "the editable install must precede any jlens import"
    assert verify < first_repo_import, "`import jlens` must be verified first"


def test_bootstrap_defines_only_primitives_before_cloning(notebook):
    constants = "".join(_code_cells(notebook)[0]["source"])
    tree = ast.parse(constants)
    assigned = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            assert len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
            assert isinstance(node.value, ast.Constant) and isinstance(
                node.value.value, str
            )
            assigned[node.targets[0].id] = node.value.value
        else:
            assert isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
    assert not [
        node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert assigned["BRANCH"] == "experiment/spokencoco-jspace-pilot"
    assert assigned["REPO_DIR"] == "/content/jacobian-lens-gemma"


def test_bootstrap_does_not_need_drive(notebook):
    cells = ["".join(cell["source"]) for cell in _code_cells(notebook)]
    verify = next(i for i, cell in enumerate(cells) if "jlens.__file__" in cell)
    mount = next(i for i, cell in enumerate(cells) if "drive.mount" in cell)
    assert verify < mount
    for index in range(verify + 1):
        assert "drive" not in cells[index].lower()


def test_the_audit_is_off_by_default(notebook):
    source = _code_source(notebook)
    assert re.search(r"^RUN_IMAGE_INDEPENDENCE_AUDIT = False$", source, re.MULTILINE)
    assert not re.search(
        r"^RUN_IMAGE_INDEPENDENCE_AUDIT = True$", source, re.MULTILINE
    )


def test_every_side_effect_is_gated_on_the_flag(notebook):
    cells = ["".join(cell["source"]) for cell in _code_cells(notebook)]
    for needle in ("drive.mount", "run_image_independence_audit("):
        owners = [cell for cell in cells if needle in cell]
        assert owners, needle
        for cell in owners:
            assert "RUN_IMAGE_INDEPENDENCE_AUDIT" in cell, needle


def test_no_model_processor_or_authentication_appears_anywhere(notebook):
    """The central guarantee: running every cell can never load Gemma."""
    source = _code_source(notebook).lower()
    for forbidden in (
        "load_gemma4",
        "autoprocessor",
        "automodel",
        "from_pretrained",
        "getpass",
        "hf_token",
        "huggingface_hub",
        "mockpilotbackend",
        "stage_capability",
        "stage_activations",
        "stage_causal",
        "torch.cuda",
        "build_dictionary",
        "gradient_pursuit",
    ):
        assert forbidden not in source, forbidden
    assert "accelerator" in json.dumps(
        json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))["metadata"]
    )


def test_the_completed_run_directory_is_explicit_and_pinned(notebook):
    source = _code_source(notebook)
    assert "COMPLETED_RUN_DIR" in source
    assert "mmpilot_pilot_20260803T160711" in source
    assert "EXPECTED_RUN_FINGERPRINT" in source
    assert "Refusing to audit a run other than the one named" in source
    # No globbing: auditing a directory nobody named is how the wrong run gets
    # a confident verdict.
    for forbidden in ("glob(", "rglob(", "iterdir()", "sorted(Path"):
        assert forbidden not in source, forbidden


def test_the_notebook_reports_resume_state_and_artifact_paths(notebook):
    source = _code_source(notebook)
    for expected in (
        "audit state:",
        "reused:",
        "computed:",
        "ARTIFACTS WRITTEN",
        "AMENDED VERDICT",
        "PRESERVATION",
        "all originals unchanged",
        "model loaded",
    ):
        assert expected in source, expected


def test_the_stop_banner_says_exactly_what_to_set(notebook):
    source = _code_source(notebook)
    assert "RUN_IMAGE_INDEPENDENCE_AUDIT = True" in source
    assert "NOTHING RAN" in source
    assert "Expected runtime" in source


def test_interpretation_boundaries_are_stated_in_prose(notebook):
    prose = " ".join(_source(notebook).split()).lower()
    assert "not erasure" in prose
    assert "projection ablation" in prose
    assert "pseudoreplication" in prose
    assert "never loads gemma" in prose
    assert "nothing original is overwritten" in prose
    assert "the original verdict is not privileged" in prose


# ------------------------------------------------------------- execution


def _clean_environment(tmp_path):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONIOENCODING"] = "utf-8"
    env["MMPILOT_REPO_DIR"] = str(REPO_ROOT)
    env["MMPILOT_AUDIT_EXPECT_FINGERPRINT"] = ""
    return env


def _run_notebook(tmp_path, *overrides, env=None, expect_ok=True):
    workdir = tmp_path / "elsewhere"
    workdir.mkdir(exist_ok=True)
    result = subprocess.run(
        [sys.executable, str(RUNNER), str(NOTEBOOK_PATH), *overrides],
        cwd=workdir,
        env={**_clean_environment(tmp_path), **(env or {})},
        capture_output=True,
        text=True,
        timeout=900,
    )
    if expect_ok:
        assert result.returncode == 0, (
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr[-3000:]}"
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


def test_the_committed_defaults_run_every_cell_and_audit_nothing(tmp_path):
    report, stdout = _run_notebook(tmp_path)

    assert report["ok"]
    assert report["run_audit"] is False
    assert report["audit_is_none"], "no audit may run at the committed defaults"
    assert report["overrides"] == {}
    assert report["model_modules_imported"] == []
    assert "NOTHING RAN" in stdout
    assert "RUN_IMAGE_INDEPENDENCE_AUDIT = True" in stdout


@pytest.fixture(scope="module")
def notebook_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("nbaudit")
    K.run_mock_pilot(root / "data", root / "run")
    return root / "run"


def test_the_full_cpu_audit_runs_from_a_clean_environment(tmp_path, notebook_run):
    report, stdout = _run_notebook(
        tmp_path,
        "RUN_IMAGE_INDEPENDENCE_AUDIT=True",
        env={"MMPILOT_AUDIT_RUN_DIR": str(notebook_run)},
    )

    assert report["ok"]
    assert Path(report["jlens_file"]).resolve().parent.parent == REPO_ROOT
    assert report["run_audit"] is True
    assert report["verdict"] in (
        "GO_CONFIRMED_AFTER_IMAGE_DEDUP",
        "WEAK_GO_AFTER_IMAGE_DEDUP",
        "NO_GO_AFTER_IMAGE_DEDUP",
    )
    assert report["model_loaded"] is False
    assert report["model_modules_imported"] == [], "no model loader may be imported"
    assert report["all_originals_unchanged"] is True
    assert report["n_groups"] == 2 * report["n_distinct_images"]
    assert report["resume_status"] == "starting"

    assert "SAME-IMAGE DEPENDENCE" in stdout
    assert "CORRECTED REPRESENTATION (image-disjoint)" in stdout
    assert "CORRECTED CAUSATION" in stdout
    assert "AMENDED VERDICT" in stdout
    assert "ARTIFACTS WRITTEN" in stdout
    assert "audit state: starting" in stdout

    for relative in (
        "audits/image_independence_audit.json",
        "metrics/representational_image_disjoint_v1.json",
        "metrics/interventions_image_level_v1.json",
        "summary_image_disjoint_v1.json",
        "report_image_disjoint_v1.md",
    ):
        assert (notebook_run / relative).is_file(), relative


def test_a_second_notebook_pass_resumes_rather_than_recomputing(tmp_path, notebook_run):
    report, stdout = _run_notebook(
        tmp_path,
        "RUN_IMAGE_INDEPENDENCE_AUDIT=True",
        env={"MMPILOT_AUDIT_RUN_DIR": str(notebook_run)},
    )
    assert report["resume_status"] == "resuming"
    assert set(report["reused"]) >= {"audit", "representational", "interventions"}
    assert "audit state: resuming" in stdout


def test_a_run_directory_that_is_not_a_completed_run_stops_the_notebook(
    tmp_path,
):
    empty = tmp_path / "not_a_run"
    empty.mkdir()
    report, _ = _run_notebook(
        tmp_path,
        "RUN_IMAGE_INDEPENDENCE_AUDIT=True",
        env={"MMPILOT_AUDIT_RUN_DIR": str(empty)},
        expect_ok=False,
    )
    assert report["ok"] is False
    assert "not a completed pilot run directory" in report["traceback"]


def test_a_pinned_fingerprint_mismatch_stops_the_notebook(tmp_path, notebook_run):
    report, _ = _run_notebook(
        tmp_path,
        "RUN_IMAGE_INDEPENDENCE_AUDIT=True",
        env={
            "MMPILOT_AUDIT_RUN_DIR": str(notebook_run),
            "MMPILOT_AUDIT_EXPECT_FINGERPRINT": "sha256:not-this-run",
        },
        expect_ok=False,
    )
    assert report["ok"] is False
    assert "Refusing to audit a run other than the one named" in report["traceback"]
