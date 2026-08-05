# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Structure and MOCK execution of the spoken-audio feasibility notebook.

The execution tests run every code cell in one namespace from a working
directory where ``jlens`` is not importable, so they check the bootstrap and the
whole audit at once — and, at the committed defaults, that opening the notebook
loads no model, mounts no Drive, and reads no media.
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
    REPO_ROOT / "notebooks" / "gemma4_native_spoken_audio_feasibility_colab.ipynb"
)
RUNNER = Path(__file__).resolve().parent / "_audio_feasibility_notebook_runner.py"
BUILDER = REPO_ROOT / "scripts" / "_build_audio_feasibility_notebook.py"

REQUIRED_SECTIONS = [
    "bootstrap repository",
    "configuration",
    "mount google drive",
    "install and verify dependencies",
    "authenticate",
    "inspect the pinned model and processor",
    "resolve the native audio protocol",
    "load one or two real spokencoco waveforms",
    "verify placeholder and audio-tower behavior",
    "real waveform vs silence vs a different waveform",
    "complete candidate-sequence scoring",
    "activation capture and invariance",
    "audio ready / audio blocked / audio invalid",
    "resume and status",
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


def test_the_commissioned_sections_appear_in_order(notebook):
    headings = re.findall(r"^## (\d+)\. (.+)$", _source(notebook), re.MULTILINE)
    assert [int(number) for number, _ in headings] == list(range(1, 15)), headings
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
    for switch in ("RUN_REAL_AUDIO_AUDIT", "RUN_MODEL_STAGE", "CONFIRM_MODEL_LOAD"):
        assert re.search(rf"^{switch} = False$", source, re.MULTILINE), switch
        assert not re.search(rf"^{switch} = True$", source, re.MULTILINE), switch


def test_the_pin_is_the_validated_checkpoint(notebook):
    source = _code_source(notebook)
    assert 'MODEL_REPO_ID = "google/gemma-4-E4B-it"' in source
    assert 'MODEL_REVISION = "fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd"' in source
    assert "gemma-3n" not in source.casefold()


def test_the_notebook_never_substitutes_text_for_audio(notebook):
    source = _code_source(notebook)
    for forbidden in ("whisper", "speech_to_text", "transcribe", "asr"):
        assert forbidden not in source.casefold(), forbidden
    prose = " ".join(_source(notebook).split()).lower()
    assert "no filename, transcript or caption is ever handed to the model" in prose
    # The prompt is built for the spoken_audio modality, which carries no caption.
    assert 'build_prompt(QUESTION, modality="spoken_audio")' in source
    assert "caption=" not in source


def test_the_notebook_runs_no_scientific_stage(notebook):
    source = _code_source(notebook)
    for forbidden in (
        "stage_capability(",
        "stage_activations(",
        "stage_codes(",
        "stage_directions(",
        "stage_causal(",
        "estimate_concept_direction(",
        "validate_lens(",
        ".fit(",
    ):
        assert forbidden not in source, forbidden


def test_the_notebook_cannot_write_into_a_completed_run(notebook):
    source = _code_source(notebook)
    assert "new_audit_run_dir(" in source
    for forbidden in ("shutil.rmtree", "os.remove", "os.unlink", ".unlink("):
        assert forbidden not in source, forbidden
    prose = " ".join(_source(notebook).split()).lower()
    assert "no completed pilot, robustness or localization run is opened" in prose


def test_the_verdict_is_computed_not_written_by_hand(notebook):
    source = _code_source(notebook)
    prose = " ".join(_source(notebook).split())
    assert "run_audio_audit(" in source
    assert "AUDIT.verdict" in source
    # The three states are named, and READY carries its caveat in the notebook.
    assert "AUDIO READY is engineering evidence only" in source
    assert "NOT evidence that J-space" in source
    assert "AUDIO INVALID" in prose and "AUDIO BLOCKED" in prose


def test_fingerprint_consequences_are_stated_in_the_notebook(notebook):
    prose = " ".join(_source(notebook).split())
    assert "must change the run fingerprint" in prose.lower()
    assert "protocol_fingerprint" in _code_source(notebook)
    assert "new lens" in prose.lower()


def test_the_protocol_is_resolved_before_the_model_is_loaded(notebook):
    cells = ["".join(cell["source"]) for cell in _code_cells(notebook)]
    resolve = next(i for i, c in enumerate(cells) if "resolve_audio_interface(" in c)
    model = next(i for i, c in enumerate(cells) if "build_real_backend(" in c)
    report = next(i for i, c in enumerate(cells) if "write_audit_report(" in c)
    assert resolve < model < report


def test_the_original_failure_is_reproduced_in_the_notebook(notebook):
    """The notebook shows the two conventions side by side, not just the fix."""
    source = _code_source(notebook)
    assert "PROCESSOR(text=AUDIT_PROMPT, audio=_probe" in source
    assert "encode_audio_prompt(PROCESSOR, AUDIT_PROMPT, _probe)" in source
    assert "the blocker" in source


# ----------------------------------------------------------- MOCK execution


def _clean_environment(tmp_path):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONIOENCODING"] = "utf-8"
    env["AUDIOAUDIT_REPO_DIR"] = str(REPO_ROOT)
    env["AUDIOAUDIT_SCRATCH"] = str(tmp_path / "scratch")
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
        timeout=1800,
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
def mock_run(tmp_path_factory):
    return _run_notebook(tmp_path_factory.mktemp("audio_mock"))


def test_the_committed_defaults_audit_the_mock_world_and_load_nothing(mock_run):
    report, stdout = mock_run
    assert report["ok"]
    assert report["mode"] == "mock"
    assert report["run_real"] is False
    assert report["run_model_stage"] is False
    assert report["confirm_model_load"] is False
    assert report["processor_class"] == "MockAudioProcessor"
    assert "MOCK: no Drive mounted" in stdout


def test_the_mock_run_resolves_the_protocol_without_a_model(mock_run):
    report, _ = mock_run
    assert report["protocol_resolved"] is True
    assert report["call_convention"] == "chat_template_audio_content_block"
    assert report["dynamic_placeholder_count"] is True
    assert report["placeholder_counts"] == [13, 25]
    assert report["protocol_fingerprint"].startswith("sha256:")
    # RUN_MODEL_STAGE is False at the defaults, so no behavior was measured and
    # the honest verdict is BLOCKED rather than a verdict about the model.
    assert report["backend_is_none"] is True
    assert report["verdict"] == "AUDIO_BLOCKED"


def test_the_mock_run_writes_its_own_report(mock_run):
    report, _ = mock_run
    written = Path(report["written"])
    assert written.name == "audio_audit.json"
    assert written.parent.name.startswith("audioaudit_mock_")
    payload = json.loads(written.read_text(encoding="utf-8"))
    assert payload["mode"] == "mock"
    assert (written.parent / "audio_audit.md").is_file()


@pytest.fixture(scope="module")
def mock_model_run(tmp_path_factory):
    """The full MOCK audit, with the model stage on against the mock backend."""
    return _run_notebook(
        tmp_path_factory.mktemp("audio_mock_model"),
        "RUN_MODEL_STAGE=True",
        "CONFIRM_MODEL_LOAD=True",
    )


def test_the_full_mock_audit_reaches_audio_ready(mock_model_run):
    report, _ = mock_model_run
    assert report["ok"]
    assert report["mode"] == "mock"
    assert report["run_real"] is False
    assert report["backend_is_none"] is False
    assert report["verdict"] == "AUDIO_READY", report["failed_checks"]
    assert report["failed_checks"] == []


def test_the_full_mock_audit_ran_every_required_check(mock_model_run):
    from jlens.mmpilot.audio_audit import REQUIRED_CHECKS

    report, _ = mock_model_run
    assert set(report["check_names"]) >= set(REQUIRED_CHECKS)


def test_the_full_mock_audit_captured_activations_and_two_recordings(mock_model_run):
    report, _ = mock_model_run
    assert report["n_waveforms"] == 2
    assert report["activation_layers"], "no residual was captured at any layer"


def test_the_ready_caveat_is_printed_not_only_documented(mock_model_run):
    _, stdout = mock_model_run
    assert "AUDIO READY is engineering evidence only" in stdout
    assert "NOT evidence that J-space" in stdout
    assert "audio protocol fingerprint:" in stdout


def test_mock_execution_is_deterministic(tmp_path_factory, mock_model_run):
    """Same notebook, same overrides, same report — twice."""
    first, _ = mock_model_run
    second, _ = _run_notebook(
        tmp_path_factory.mktemp("audio_mock_again"),
        "RUN_MODEL_STAGE=True",
        "CONFIRM_MODEL_LOAD=True",
    )
    assert second["report_checksum"] == first["report_checksum"]
    assert second["protocol_fingerprint"] == first["protocol_fingerprint"]
    # The run directory is timestamped, so it differs; the content does not.
    assert second["run_dir"] != first["run_dir"]
