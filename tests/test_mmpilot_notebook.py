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
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = (
    REPO_ROOT / "notebooks" / "multimodal_jspace_spokencoco_pilot_colab.ipynb"
)

RUNNER = Path(__file__).resolve().parent / "_mmpilot_notebook_runner.py"

REQUIRED_SECTIONS = [
    "colab bootstrap",
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


def test_sections_appear_in_the_required_order(notebook):
    """Bootstrap is section 0; the sixteen commissioned stages follow it."""
    headings = re.findall(r"^## (\d+)\. (.+)$", _source(notebook), re.MULTILINE)
    assert [int(number) for number, _ in headings] == list(range(0, 17)), headings
    for (_, title), expected in zip(headings, REQUIRED_SECTIONS, strict=True):
        assert expected in title.lower(), (expected, title)


def test_bootstrap_cells_come_before_any_repository_import(notebook):
    """The regression this guards: section 1 imported ``jlens`` while the
    package had not been cloned or installed, so a fresh Colab runtime died
    with ModuleNotFoundError on the first executed cell."""
    cells = ["".join(cell["source"]) for cell in _code_cells(notebook)]
    install = next(i for i, cell in enumerate(cells) if "pip" in cell and "install" in cell
                   and "-e" in cell)
    verify = next(i for i, cell in enumerate(cells) if "jlens.__file__" in cell)
    first_repo_import = next(
        i for i, cell in enumerate(cells) if re.search(r"^\s*from jlens", cell, re.MULTILINE)
    )
    assert install < first_repo_import, "the editable install must precede any jlens import"
    assert verify < first_repo_import, "`import jlens` must be verified first"
    # Cells before the verification may only use the standard library.
    for index in range(verify + 1):
        assert not re.search(r"^\s*(from|import) jlens\.", cells[index], re.MULTILINE) or (
            index == verify
        ), f"cell {index} imports from jlens before the bootstrap verified it"


def test_bootstrap_defines_only_primitives_before_cloning(notebook):
    """The first cell may assign string constants and print. Nothing else —
    an import here is what broke a fresh runtime."""
    constants = "".join(_code_cells(notebook)[0]["source"])
    tree = ast.parse(constants)
    assigned = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            assert len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
            assert isinstance(node.value, ast.Constant) and isinstance(node.value.value, str), (
                f"{node.targets[0].id} is not a primitive string constant"
            )
            assigned[node.targets[0].id] = node.value.value
        else:
            assert isinstance(node, ast.Expr) and isinstance(node.value, ast.Call), (
                f"unexpected statement in the constants cell: {ast.dump(node)[:80]}"
            )
    assert not [node for node in ast.walk(tree)
                if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert assigned["BRANCH"] == "experiment/spokencoco-jspace-pilot"
    assert assigned["REPO_DIR"] == "/content/jacobian-lens-gemma"
    assert assigned["REPO_URL"].endswith("jacobian-lens-gemma.git")


def test_clone_cell_is_idempotent_and_verifies_the_branch(notebook):
    cells = ["".join(cell["source"]) for cell in _code_cells(notebook)]
    clone_cell = next(cell for cell in cells if "clone" in cell)
    for expected in ("fetch", "checkout", "reset", "--hard", "rev-parse"):
        assert expected in clone_cell, expected
    assert '(REPO_PATH / ".git").is_dir()' in clone_cell
    assert "refusing to continue against the wrong code" in clone_cell


def test_bootstrap_does_not_need_drive(notebook):
    cells = ["".join(cell["source"]) for cell in _code_cells(notebook)]
    verify = next(i for i, cell in enumerate(cells) if "jlens.__file__" in cell)
    mount = next(i for i, cell in enumerate(cells) if "drive.mount" in cell)
    assert verify < mount
    for index in range(verify + 1):
        assert "drive" not in cells[index].lower()


def test_install_failure_is_reported_not_swallowed(notebook):
    source = _code_source(notebook)
    assert "pip install -e failed" in source
    assert "still not importable after installation" in source
    assert "is shadowing this checkout" in source


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
    """Images live under coco/ and audio under SpokenCOCO/, so the two roots
    are configured separately — a single dataset root resolves neither."""
    config = "".join(_code_cells(notebook)[3]["source"])
    assignments = {
        node.targets[0].id: node.value.value
        for node in ast.parse(config).body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Constant)
    }
    assert assignments["SPOKENCOCO_BASE_ROOT"] == (
        "/content/drive/MyDrive/datasets/cstf_spokencoco"
    )
    assert assignments["IMAGE_MEDIA_ROOT"] == (
        "/content/drive/MyDrive/datasets/cstf_spokencoco/coco"
    )
    assert assignments["AUDIO_MEDIA_ROOT"] == (
        "/content/drive/MyDrive/datasets/cstf_spokencoco/SpokenCOCO"
    )
    assert assignments["DOWNLOAD_CACHE"] == (
        "/content/drive/MyDrive/datasets/cstf_spokencoco_download_cache"
    )
    assert assignments["MANIFEST_PATH"] == (
        "/content/drive/MyDrive/datasets/spokencoco_manifest.json"
    )


def test_all_four_roots_are_offered_in_the_required_order(notebook):
    cell = next(
        "".join(c["source"]) for c in _code_cells(notebook) if "CANDIDATE_ROOTS" in "".join(c["source"])
    )
    order = re.search(r"CANDIDATE_ROOTS = \[(.*?)\]", cell, re.DOTALL).group(1)
    names = [name.strip().rstrip(",") for name in order.strip().splitlines() if name.strip()]
    assert names == [
        "IMAGE_MEDIA_ROOT",
        "AUDIO_MEDIA_ROOT",
        "DOWNLOAD_CACHE",
        "SPOKENCOCO_BASE_ROOT",
    ]


def test_media_root_audit_runs_before_normalization(notebook):
    cells = ["".join(cell["source"]) for cell in _code_cells(notebook)]
    audit_cell = next(i for i, cell in enumerate(cells) if "audit_media_roots" in cell)
    normalize_cell = next(i for i, cell in enumerate(cells) if "normalize_manifest" in cell)
    assert audit_cell <= normalize_cell
    cell = cells[audit_cell]
    assert cell.index("audit_media_roots") < cell.index("normalize_manifest")
    # Normalization must receive the per-role roots, not one shared list.
    assert "image_roots=IMAGE_ROOTS" in cell
    assert "audio_roots=AUDIO_ROOTS" in cell
    assert "expected_roots=EXPECTED_ROOTS" in cell


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


def _clean_environment(tmp_path):
    """An environment where ``jlens`` is not importable: no repo on the path,
    ``PYTHONPATH`` cleared, and a working directory outside the checkout."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONIOENCODING"] = "utf-8"
    env["MMPILOT_REPO_DIR"] = str(REPO_ROOT)
    env["MMPILOT_SCRATCH"] = str(tmp_path / "scratch")
    env["MMPILOT_RUN_DIR"] = str(tmp_path / "run")
    return env


def test_the_test_environment_really_is_clean(tmp_path):
    """Guards the guard: if ``jlens`` were importable here anyway, the
    execution test below would pass even with the bootstrap broken."""
    probe = subprocess.run(
        [sys.executable, "-c", "import jlens"],
        cwd=tmp_path,
        env=_clean_environment(tmp_path),
        capture_output=True,
        text=True,
    )
    assert probe.returncode != 0
    assert "No module named 'jlens'" in probe.stderr


def test_mock_execution_from_a_clean_environment(tmp_path):
    """Execute every code cell in a subprocess that starts without ``jlens``.

    This is the end-to-end check on the bootstrap: cell 0 has to make the
    package importable, and everything after it has to run to a decision.
    """
    workdir = tmp_path / "elsewhere"
    workdir.mkdir()
    result = subprocess.run(
        [sys.executable, str(RUNNER), str(NOTEBOOK_PATH)],
        cwd=workdir,
        env=_clean_environment(tmp_path),
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr[-3000:]}"
    report = json.loads(result.stdout.strip().splitlines()[-1])

    assert report["ok"]
    assert Path(report["jlens_file"]).resolve().parent.parent == REPO_ROOT
    assert Path(report["repo_path"]).resolve() == REPO_ROOT
    assert Path(report["cwd"]).resolve() == REPO_ROOT, "cell 0c must chdir into the repo"
    assert report["commit"]
    assert report["run_real_pilot"] is False
    assert report["model_is_none"], "the real model must not load by default"
    assert report["scientific_evidence"] is False
    assert report["recommendation"] in ("GO", "WEAK GO", "NO-GO")
    assert report["leakage_ok"]
    assert report["invariance_passed"]
    assert report["resume_status"] == "starting"
    assert report["n_interventions"] > 0

    run_dir = Path(tmp_path / "run")
    assert (run_dir / "report.md").is_file()
    assert (run_dir / "summary.json").is_file()
    assert (run_dir / "derived_manifest.json").is_file()
    assert (run_dir / "expanded_manifest.json").is_file()

def test_runtime_expansion_precedes_selection_and_model_scoring(notebook):
    source = _code_source(notebook)
    discovery = source.index("discover_metadata_sources")
    ranking = source.index("concept coverage ranking")
    capability = source.index("stage_capability")
    assert discovery < ranking < capability
    for expected in (
        "BASELINE_COVERAGE_SUFFICIENT",
        "max_files=40",
        "max_depth=3",
        "coco_object_annotation",
        "persist_expanded_manifest",
        "FINAL_MANIFEST_KIND",
        "expanded_derived",
        "GROUPS_PER_CONCEPT = 6",
        "N_CONCEPTS_TO_KEEP = 2",
        "TINY_SMOKE = False",
    ):
        assert expected in source, expected


def test_notebook_does_not_download_or_mutate_original_manifest(notebook):
    source = _code_source(notebook)
    assert "media_redownloaded\": False" in source
    assert "MANIFEST_PATH = \"/content/drive/MyDrive/datasets/spokencoco_manifest.json\"" in source
    assert "write_text" not in "\n".join(line for line in source.splitlines() if "MANIFEST_PATH" in line)
