# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Structural checks for ``notebooks/jspace_language_autoencoder_colab.ipynb``.

Light path only: valid nbformat, no stored outputs, every code cell parses, the
nine required sections are present in order, and the invariants that make the
notebook safe to hand to a fresh Colab runtime hold — the lens is checksum-gated
for the real pilot, the HF token is read with ``getpass`` and never printed,
stages write to the Drive-backed run directory, smoke mode is the default, and
nothing deletes a previous run.
"""

import ast
import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "jspace_language_autoencoder_colab.ipynb"


@pytest.fixture(scope="module")
def notebook():
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def _code_cells(payload):
    return [c for c in payload["cells"] if c["cell_type"] == "code"]


def _source(payload):
    return "\n".join("".join(c["source"]) for c in payload["cells"])


def _code_source(payload):
    return "\n".join("".join(c["source"]) for c in _code_cells(payload))


def _flat(payload):
    """All prose with whitespace collapsed, so an assertion about a sentence is
    not defeated by where the markdown happens to wrap."""
    return " ".join(_source(payload).split())


def test_notebook_is_valid_and_output_free(notebook):
    assert notebook["nbformat"] == 4
    assert notebook["cells"]
    for index, cell in enumerate(_code_cells(notebook)):
        assert cell["outputs"] == [], f"code cell {index} has stored outputs"
        assert cell["execution_count"] is None, f"code cell {index} has an exec count"


def test_all_code_cells_compile(notebook):
    """Also rules out IPython magics: ``!cmd`` / ``%cmd`` are not valid Python, so
    a cell that shells out through a magic instead of subprocess fails here."""
    for index, cell in enumerate(_code_cells(notebook)):
        source = "".join(cell["source"])
        try:
            ast.parse(source)
        except SyntaxError as exc:  # pragma: no cover - failure path
            pytest.fail(f"code cell {index} does not parse: {exc}")


def test_all_nine_sections_are_present_in_order(notebook):
    headings = re.findall(r"^## (\d+)\. (.+)$", _source(notebook), re.MULTILINE)
    assert [int(number) for number, _ in headings] == list(range(1, 10)), headings


def test_section_titles_match_the_required_workflow(notebook):
    titles = [title.lower() for _, title in re.findall(
        r"^## (\d+)\. (.+)$", _source(notebook), re.MULTILINE
    )]
    for index, expected in enumerate(
        [
            "setup and checksums",
            "dataset smoke build",
            "reconstructor training",
            "reconstructor gate",
            "adapter warm start",
            "reconstructor-guided preference training",
            "held-out evaluation",
            "artifact export",
            "final go/no-go report",
        ]
    ):
        assert expected in titles[index], (expected, titles[index])


def test_runtime_and_repo_setup(notebook):
    source = _code_source(notebook)
    for expected in (
        "platform.python_version",
        "torch.__version__",
        "transformers.__version__",
        "torch.version.cuda",
        "get_device_properties",
        "nvidia-smi",
    ):
        assert expected in source, expected
    assert "experiment/jspace-language-autoencoder" in source
    assert "HEAD assertion failed" in source
    assert '"install", "-q", "-e", "."' in source
    assert "EXPECTED_HEAD" in source


def test_semantic_head_assertion_checks_the_autoencoder_api(notebook):
    source = _code_source(notebook)
    assert "predates the J-space language autoencoder API" in source
    for name in ("AutoencoderConfig", "gonogo_report", "attribute_failure"):
        assert name in source, name


def test_hf_token_is_read_with_getpass_and_never_printed(notebook):
    source = _code_source(notebook)
    assert "getpass.getpass" in source
    assert "print(_token" not in source
    assert "os.environ[\"HF_TOKEN\"] = _token" in source
    # Only the length and a masked prefix may be shown.
    assert "prefix {_value[:3]}***" in source


def test_lens_is_checksum_gated_for_the_real_pilot(notebook):
    source = _code_source(notebook)
    assert "expect_file_sha256" in source
    assert "lens checksum mismatch" in source
    assert "hashlib.sha256" in source


def test_smoke_is_the_default_and_the_pilot_is_opt_in(notebook):
    source = _code_source(notebook)
    assert re.search(r"^SMOKE = True$", source, re.MULTILINE)
    assert "--allow-model-load" in source
    assert "Exact pilot commands" in source


def test_stages_write_to_the_drive_backed_run_directory(notebook):
    source = _code_source(notebook)
    assert "JLANG_ROOT" in source and "MyDrive" in source
    assert '"--output-dir", str(RUN_DIR)' in source
    for script in (
        "build_jspace_language_dataset",
        "train_phrase_reconstructor",
        "train_cone_adapter",
        "evaluate_jspace_language",
    ):
        assert script in source, script


def test_notebook_never_deletes_a_previous_run(notebook):
    source = _code_source(notebook)
    for forbidden in ("shutil.rmtree", "rm -rf", "os.remove(", "unlink("):
        assert forbidden not in source, forbidden


def test_live_monitoring_and_resume_are_exposed(notebook):
    source = _code_source(notebook)
    assert "nvidia-smi" in source
    assert "def follow(" in source and "def launch(" in source
    assert "subprocess.Popen" in source
    text = _flat(notebook)
    assert "resume" in text.lower()
    assert "adapter_epoch" in text


def test_the_gate_section_states_both_thresholds(notebook):
    text = _flat(notebook)
    assert "AUROC >= 0.80" in text
    assert "top-5 retrieval >= 50%" in text
    assert "the verbalizer must not be trained" in text


def test_the_report_section_cannot_hide_a_negative_result(notebook):
    text = _flat(notebook)
    assert "no code path that hides a negative result" in text
    source = _code_source(notebook)
    assert "PRIMARY FAILURE MODE" in source
    assert "failure_attribution" in source


def test_confabulation_attractors_are_reported(notebook):
    text = _flat(notebook)
    for attractor in ("black hole", "photosynthesis", "quantum entanglement", "Great Barrier Reef"):
        assert attractor in text, attractor
