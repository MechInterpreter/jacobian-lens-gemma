"""The full-vocabulary notebook: generated, output-free, MOCK-executable, honest.

Requirement 29, plus the notebook-level halves of the requirements that can only
be checked against what the notebook actually does: the CPU/GPU stage
separation, the two explicit confirmations, the budget printed before any model
load, the reporting boundary, and the fact that no completed study's notebook is
touched.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = (
    ROOT
    / "notebooks"
    / "multimodal_jspace_full_vocabulary_causal_validation_colab.ipynb"
)
BUILDER = ROOT / "scripts" / "_build_full_vocabulary_notebook.py"
RUNNER = ROOT / "tests" / "_full_vocabulary_notebook_runner.py"

#: Every completed study's notebook, which this correction must not rewrite.
COMPLETED_NOTEBOOKS = (
    ROOT / "notebooks" / "multimodal_jspace_anthropic_band33_40_swap_colab.ipynb",
    ROOT / "notebooks" / "multimodal_jspace_anthropic_band_swap_colab.ipynb",
    ROOT / "notebooks" / "multimodal_jspace_spokencoco_native_audio_colab.ipynb",
)


def _payload():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code(payload):
    return [cell for cell in payload["cells"] if cell["cell_type"] == "code"]


def _source(payload):
    return "\n".join("".join(cell["source"]) for cell in payload["cells"])


def _code_source(payload):
    return "\n".join("".join(cell["source"]) for cell in _code(payload))


# ---- 29a. generated, output-free, deterministic


def test_notebook_is_output_free_and_every_cell_parses():
    payload = _payload()
    assert payload["metadata"]["accelerator"] == "GPU"
    for cell in _code(payload):
        assert cell["outputs"] == []
        assert cell["execution_count"] is None
        ast.parse("".join(cell["source"]))


def test_notebook_matches_its_builder_byte_for_byte():
    import importlib.util

    spec = importlib.util.spec_from_file_location("_fv_builder", BUILDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rebuilt = json.dumps(module.build(), indent=1, ensure_ascii=False) + "\n"
    assert rebuilt == NOTEBOOK.read_text(encoding="utf-8")


def test_the_builder_writes_only_its_own_target():
    source = BUILDER.read_text(encoding="utf-8")
    assert (
        'TARGET = (\n    ROOT\n    / "notebooks"\n    / '
        '"multimodal_jspace_full_vocabulary_causal_validation_colab.ipynb"\n)'
    ) in source
    assert source.count("TARGET.write_text") == 1


def test_no_completed_notebook_is_touched():
    source = BUILDER.read_text(encoding="utf-8")
    for path in COMPLETED_NOTEBOOKS:
        assert path.name not in source or "untouched" in source
        assert path.exists()


# ---- 29b. the three stages and the two confirmations


def test_the_three_stage_switches_and_two_confirmations_exist():
    source = _code_source(_payload())
    for switch in (
        "RUN_ENDPOINT_AUDIT_CPU = False",
        "RUN_FULL_VOCAB_CAUSAL_GPU = False",
        "RUN_FINAL_REPORT_CPU = False",
        "CONFIRM_MODEL_LOAD = False",
        "CONFIRM_PASS_BUDGET = False",
    ):
        assert switch in source, switch
    assert (
        "GPU_STAGE = bool(RUN_FULL_VOCAB_CAUSAL_GPU and CONFIRM_MODEL_LOAD "
        "and CONFIRM_PASS_BUDGET)"
    ) in source


def test_the_cpu_stages_never_reference_a_model_loader_outside_the_gpu_gate():
    payload = _payload()
    for cell in _code(payload):
        text = "".join(cell["source"])
        if "build_real_backend" not in text:
            continue
        assert "if GPU_STAGE:" in text, (
            "a cell that can load the model is not behind the GPU gate"
        )


def test_the_budget_is_printed_before_any_model_load():
    payload = _payload()
    cells = [("".join(cell["source"])) for cell in _code(payload)]
    budget_cell = next(
        index for index, text in enumerate(cells) if "format_pass_budget(BUDGET)" in text
    )
    load_cell = next(
        index
        for index, text in enumerate(cells)
        if "allow_model_load=True" in text
    )
    assert budget_cell < load_cell


def test_the_budget_cell_refuses_over_the_cap():
    source = _code_source(_payload())
    assert 'if not BUDGET["within_cap"]:' in source
    assert "Stopping before any model" in source
    assert "cap=PASS_CAP" in source


# ---- 29c. the endpoint, stated in the notebook itself


def test_the_notebook_states_the_unrestricted_endpoint():
    source = _source(_payload())
    for phrase in (
        "no answer candidate appended",
        "complete next-token distribution",
        "global rank 1",
        "ENDPOINT_UNRESTRICTED_NEXT_TOKEN",
        "score_unrestricted_next_token",
    ):
        assert phrase in source, phrase


def test_the_notebook_states_the_reporting_boundary():
    source = _source(_payload())
    for phrase in (
        "measurement-correction rerun on the same population",
        "an independent replication",
        "never primary evidence",
        "L32 remains excluded",
        "the research phase ends",
        "spoken captions",
    ):
        assert phrase in source, phrase


def test_the_notebook_never_calls_a_restricted_result_paper_comparable():
    """The phrase may appear only where the notebook names it as prohibited."""
    hits = [
        line
        for line in _source(_payload()).splitlines()
        if "paper-comparable" in line.lower() or "paper_comparable" in line.lower()
    ]
    for line in hits:
        assert any(
            marker in line.lower()
            for marker in ("prohibited", "must not", "superseded", "fails the audit")
        ), line


def test_the_notebook_declares_greedy_generation_secondary():
    source = _source(_payload())
    assert "secondary" in source
    assert "no verdict is derived from" in source


# ---- 29d. it executes end to end in MOCK mode


def test_notebook_runs_end_to_end_in_mock_mode(tmp_path):
    env = dict(os.environ)
    env["PYTHONPATH"] = ""
    env["JLENS_REPO_DIR"] = str(ROOT)
    result = subprocess.run(
        [sys.executable, str(RUNNER), str(NOTEBOOK)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=env,
        timeout=900,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-2000:]
    report = json.loads(result.stdout.strip().splitlines()[-1])
    assert report["ok"] is True
    assert report["real_mode"] is False
    assert report["gpu_stage"] is False
    # 25. no CPU stage loaded Gemma
    assert report["loaded_gemma"] is False
    assert report["torch_cuda_initialized"] is False
    # 30. the audit ran and passed
    assert report["audit_passed"] is True
    assert report["n_overclaims"] == 0
    # amendments
    assert report["n_amendments"] == 2
    # 27. the derived budget is under the cap
    assert report["budget_within_cap"] is True
    assert report["budget_total"] < 5000
    # 28. every commissioned MOCK scenario returned its predeclared verdict
    assert len(report["mock_results"]) == 8
    assert all(row["as_required"] for row in report["mock_results"].values())
    assert report["mock_results"]["restricted_only"]["verdict"] == (
        "FULL_VOCAB_REASONING_NO_GO"
    )
    # 22. the two verdict families stayed separate
    assert report["unrestricted_verdict"] == "FULL_VOCAB_REASONING_ALPHA1_GO"
    assert report["conditional_verdict"] == "NOT_EVALUATED"
    assert report["report_mode"] == "mock"
