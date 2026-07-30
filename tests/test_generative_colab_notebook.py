# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Structural checks for ``notebooks/generative_jlens_colab.ipynb``.

Light path only: valid nbformat, no stored outputs, every code cell parses, the
ten required sections are present, and the invariants that make the notebook
safe to hand to a fresh Colab runtime hold — the lens fingerprint is pinned and
gated, the HF token is read with ``getpass`` and never printed, the experiment
writes to the Drive-backed runs root, and nothing deletes a previous run.
"""

import ast
import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "generative_jlens_colab.ipynb"


@pytest.fixture(scope="module")
def notebook():
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def _code_cells(payload):
    return [c for c in payload["cells"] if c["cell_type"] == "code"]


def _source(payload):
    return "\n".join("".join(c["source"]) for c in payload["cells"])


def _code_source(payload):
    return "\n".join("".join(c["source"]) for c in _code_cells(payload))


def test_notebook_is_valid_and_output_free(notebook):
    assert notebook["nbformat"] == 4
    assert notebook["cells"]
    for index, cell in enumerate(_code_cells(notebook)):
        assert cell["outputs"] == [], f"code cell {index} has stored outputs"
        assert cell["execution_count"] is None, f"code cell {index} has an exec count"


def test_all_code_cells_compile(notebook):
    """Also rules out IPython magics: `!cmd` / `%cmd` are not valid Python, so a
    cell that shells out through a magic instead of subprocess fails here."""
    for index, cell in enumerate(_code_cells(notebook)):
        source = "".join(cell["source"])
        try:
            ast.parse(source)
        except SyntaxError as exc:  # pragma: no cover - failure path
            pytest.fail(f"code cell {index} does not parse: {exc}")


def test_all_ten_sections_are_present_in_order(notebook):
    headings = [
        heading
        for heading in re.findall(r"^## (\d+)\. (.+)$", _source(notebook), re.MULTILINE)
    ]
    numbers = [int(number) for number, _ in headings]
    assert numbers == list(range(1, 11)), headings


def test_runtime_and_repo_setup(notebook):
    source = _code_source(notebook)
    # 1. runtime facts
    for expected in ("platform.python_version", "torch.__version__",
                     "transformers.__version__", "torch.version.cuda",
                     "get_device_properties", "nvidia-smi"):
        assert expected in source, expected
    # 2. branch checkout, HEAD assertion, editable install
    assert "experiment/generative-jlens-validation" in source
    assert "HEAD assertion failed" in source
    assert '"install", "-q", "-e", "."' in source or '"-e", "."' in source
    assert "EXPECTED_HEAD" in source
    # The semantic half of the HEAD assertion: the provenance API must exist.
    for name in ("cone_source_role", "expected_cone_source_example_id",
                 "tensor_sha256", "vector_identity"):
        assert name in source, name


def test_hf_token_is_requested_securely_and_never_printed(notebook):
    source = _code_source(notebook)
    assert "getpass.getpass" in source
    assert 'os.environ["HF_TOKEN"]' in source
    # No cell may print the token itself. Masked reporting (length + prefix) is
    # what the notebook does instead; a bare print of the value is a leak.
    for cell in _code_cells(notebook):
        text = "".join(cell["source"])
        assert 'print(os.environ["HF_TOKEN"])' not in text
        assert "print(_token" not in text
    # ...and it is never written anywhere.
    assert "write(_token" not in source
    assert "HF_TOKEN" in source


def test_lens_checksum_is_pinned_and_gated(notebook):
    source = _code_source(notebook)
    assert "91_753_066" in source or "91753066" in source
    assert (
        "7229c7562d1d55420b70abb13f481934649c4b01417bd851e97cedb47c96f474" in source
    )
    assert "pilot_20260715T200437612150_311fd108c23a" in source
    # Verification failure must abort, not warn.
    assert source.count("ABORT") >= 2
    assert "SHA-256 mismatch" in source
    assert "size mismatch" in source


def test_experiment_invocation_matches_the_intended_run(notebook):
    source = _code_source(notebook)
    assert "scripts/validate_benchmark_targets.py" in source
    assert "scripts/run_generative_validation.py" in source
    assert "configs/gemma_generative_validation.yaml" in source
    assert '"-u"' in source  # unbuffered, so streaming is actually live
    assert '"--allow-model-load"' in source
    assert '"--device-map", "cuda"' in source
    assert '"--runs-root", str(RUNS_ROOT)' in source
    assert "LIMIT_EXAMPLES = 2" in source
    assert 'LAYERS = ["14", "21"]' in source
    assert 'RATIOS = ["0.05", "0.1", "0.25"]' in source


def test_run_streams_to_drive_and_reports_outcome(notebook):
    source = _code_source(notebook)
    assert "LOGS_ROOT" in source and "log_path" in source
    assert "elapsed" in source and "return code" in source
    assert "SUCCESS" in source and "FAILURE" in source
    assert "[-100:]" in source  # last 100 log lines on failure


def test_monitor_is_stoppable_and_shows_the_required_signals(notebook):
    source = _code_source(notebook)
    assert "def monitor(" in source
    assert "KeyboardInterrupt" in source
    assert "MONITOR_MAX_SECONDS" in source
    assert "utilization.gpu" in source and "memory.used" in source
    assert "records.jsonl" in source
    assert "[-20:]" in source  # last 20 log lines


def test_inspection_covers_artifacts_provenance_and_the_mandela_check(notebook):
    source = _code_source(notebook)
    for artifact in ("records.jsonl", "gates.json", "prompt_debug.json",
                     "run_metadata.json", "summary.md"):
        assert artifact in source, artifact
    for field in ("source_example_id", "cone_source_example_id",
                  "donor_example_id", "source_activation_sha256",
                  "cone_sha256", "cone_norm", "source_prompt"):
        assert field in source, field
    assert "dev-entity-mandela" in source
    assert "FLAGGED" in source
    # Missing fields are displayed, never fabricated.
    assert "never invented" in _source(notebook) or "never back-filled" in _source(
        notebook
    )


def test_archive_step_is_persistent_and_fingerprinted(notebook):
    source = _code_source(notebook)
    assert "zipfile.ZipFile" in source
    assert "ARCHIVE_ROOT" in source
    assert "sha256_file(ARCHIVE_PATH)" in source
    assert "PERSISTENT PATHS" in source


def test_no_cell_deletes_a_previous_run(notebook):
    """Cells must be rerunnable without destroying earlier results.

    The one permitted unlink is the failed-restore staging file in section 5,
    which is a temporary the same cell created and which never reaches the
    lens path.
    """
    source = _code_source(notebook)
    for forbidden in ("shutil.rmtree", "rm -rf", "os.removedirs"):
        assert forbidden not in source, forbidden
    unlinks = re.findall(r"(\w+)\.unlink\(", source)
    assert set(unlinks) <= {"staging"}, unlinks
    assert "mkdir(parents=True, exist_ok=True)" in source
