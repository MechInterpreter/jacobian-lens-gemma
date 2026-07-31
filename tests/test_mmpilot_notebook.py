# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Structure and MOCK execution of the SpokenCOCO pilot notebook.

The execution test runs every code cell in one namespace with
``RUN_REAL_PILOT`` left at its committed default, so it checks two things at
once: that the notebook works end to end, and that opening it never starts the
real experiment.
"""

import ast
import json
import os
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = (
    REPO_ROOT / "notebooks" / "multimodal_jspace_spokencoco_pilot_colab.ipynb"
)

REQUIRED_SECTIONS = [
    "configuration",
    "mount google drive",
    "install dependencies",
    "authenticate",
    "audit the model architecture",
    "audit spokencoco",
    "build the pilot subset",
    "run the capability gate",
    "load and validate the frozen j-lens",
    "extract activations",
    "compute j-space codes",
    "run representational tests",
    "estimate concept directions",
    "run causal-transfer interventions",
    "generate the go/no-go report",
    "show resume status",
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


def test_sixteen_sections_appear_in_the_required_order(notebook):
    headings = re.findall(r"^## (\d+)\. (.+)$", _source(notebook), re.MULTILINE)
    assert [int(number) for number, _ in headings] == list(range(1, 17)), headings
    for (_, title), expected in zip(headings, REQUIRED_SECTIONS, strict=True):
        assert expected in title.lower(), (expected, title)


def test_the_real_pilot_is_off_by_default(notebook):
    source = _code_source(notebook)
    assert re.search(r"^RUN_REAL_PILOT = False$", source, re.MULTILINE)
    assert "RUN_REAL_PILOT = True" not in source


def test_every_expensive_action_is_gated_on_the_flag(notebook):
    cells = ["".join(cell["source"]) for cell in _code_cells(notebook)]
    for needle in ("drive.mount", "load_gemma4", "AutoProcessor.from_pretrained",
                   "getpass.getpass"):
        owners = [cell for cell in cells if needle in cell]
        assert owners, needle
        for cell in owners:
            assert "RUN_REAL_PILOT" in cell, needle


def test_configured_drive_paths_match_the_existing_dataset(notebook):
    source = _code_source(notebook)
    assert "/content/drive/MyDrive/datasets/cstf_spokencoco" in source
    assert "/content/drive/MyDrive/datasets/spokencoco_manifest.json" in source
    assert "/content/drive/MyDrive/datasets/cstf_spokencoco_download_cache" in source


def test_notebook_never_downloads_or_rewrites_the_dataset(notebook):
    source = _code_source(notebook)
    for forbidden in ("shutil.rmtree", "os.remove", "download_dataset", "wget", "curl "):
        assert forbidden not in source, forbidden
    assert "never re-downloaded" in _source(notebook) or "not re-downloaded" in _source(notebook)


def test_lens_is_checksum_pinned_and_never_refitted(notebook):
    source = _code_source(notebook)
    assert "LENS_EXPECT_SHA256" in source
    assert "file_sha256" in source
    assert "validate_lens" in source
    assert "fit(" not in source
    assert "This pilot does not fit a lens." in source


def test_hf_token_is_read_with_getpass_and_not_printed(notebook):
    source = _code_source(notebook)
    assert "getpass.getpass" in source
    assert "print(_token)" not in source
    assert 'os.environ["HF_TOKEN"] = _token' in source


def test_invariance_gate_precedes_the_intervention_cell(notebook):
    cells = ["".join(cell["source"]) for cell in _code_cells(notebook)]
    gate = next(i for i, cell in enumerate(cells) if "run_invariance_gate" in cell)
    causal = next(i for i, cell in enumerate(cells) if "stage_causal" in cell)
    assert gate < causal


def test_interpretation_boundaries_are_stated_in_prose(notebook):
    prose = " ".join(_source(notebook).split()).lower()
    assert "not erasure" in prose
    assert "projection ablation" in prose
    assert "environmental audio" in prose
    assert "spoken_audio" in prose
    assert "never a scientific result" in prose or "not evidence about gemma" in prose


def test_no_forbidden_method_is_implemented(notebook):
    source = _code_source(notebook).lower()
    for forbidden in (
        "contrastive",
        "phrase_reconstructor",
        "reconstructor",
        "autoencoder",
        "adapter",
        "logisticregression",
    ):
        assert forbidden not in source, forbidden


def test_mock_execution_of_every_code_cell(notebook, tmp_path, monkeypatch):
    """Execute the notebook's code cells in order, in one namespace."""
    monkeypatch.setenv("MMPILOT_SCRATCH", str(tmp_path / "scratch"))
    monkeypatch.setenv("MMPILOT_RUN_DIR", str(tmp_path / "run"))
    monkeypatch.chdir(REPO_ROOT)
    namespace: dict = {"__name__": "__notebook__"}
    for index, cell in enumerate(_code_cells(notebook)):
        source = "".join(cell["source"])
        try:
            exec(compile(source, f"<cell {index}>", "exec"), namespace)  # noqa: S102
        except Exception as exc:  # pragma: no cover - failure path
            pytest.fail(f"code cell {index} failed: {type(exc).__name__}: {exc}")

    assert namespace["RUN_REAL_PILOT"] is False
    assert namespace["MODEL"] is None, "the real model must not load by default"
    assert namespace["SUMMARY"]["scientific_evidence"] is False
    assert namespace["SUMMARY"]["recommendation"] in ("GO", "WEAK GO", "NO-GO")
    assert namespace["LEAKAGE"]["ok"]
    assert namespace["INVARIANCE"]["passed"]
    assert (Path(os.environ["MMPILOT_RUN_DIR"]) / "report.md").is_file()
    assert (Path(os.environ["MMPILOT_RUN_DIR"]) / "summary.json").is_file()
    assert (Path(os.environ["MMPILOT_RUN_DIR"]) / "derived_manifest.json").is_file()
    assert namespace["STATUS"]["status"] == "starting"
    assert namespace["STATUS"]["completed_units"]["intervention"] > 0
