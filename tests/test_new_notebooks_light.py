# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Structural light-path checks for the causal smoke and multimodal capture
notebooks: valid nbformat, no stored outputs, compilable code cells, model
loading gated off by default, and required safety phrases present."""

import ast
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = {
    "causal": REPO_ROOT / "notebooks" / "gemma_4_e4b_jspace_causal_smoke.ipynb",
    "multimodal": REPO_ROOT / "notebooks" / "gemma_4_e4b_multimodal_jlens_capture.ipynb",
}


@pytest.fixture(params=sorted(NOTEBOOKS))
def notebook(request):
    path = NOTEBOOKS[request.param]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["_name"] = request.param
    return payload


def _code_cells(payload):
    return [c for c in payload["cells"] if c["cell_type"] == "code"]


def _full_source(payload):
    return "\n".join("".join(c["source"]) for c in payload["cells"])


def test_notebook_is_valid_and_output_free(notebook):
    assert notebook["nbformat"] == 4
    for cell in _code_cells(notebook):
        assert cell["outputs"] == []
        assert cell["execution_count"] is None


def test_all_code_cells_compile(notebook):
    for index, cell in enumerate(_code_cells(notebook)):
        source = "".join(cell["source"])
        try:
            ast.parse(source)
        except SyntaxError as exc:  # pragma: no cover
            pytest.fail(f"code cell {index} does not parse: {exc}")


def test_model_loading_is_gated(notebook):
    source = _full_source(notebook)
    assert 'os.environ.setdefault("JLENS_ALLOW_GEMMA", "1" if IN_COLAB else "0")' in source
    assert "allow_model_load=True" in source  # inside the gated branch only
    assert "if not ALLOW_MODEL_LOAD:" in source
    # The frozen-lens fingerprint gate must be present.
    assert "expect_file_sha256" in source
    assert "refusing" in source


def test_causal_notebook_specific_guarantees():
    source = _full_source(json.loads(NOTEBOOKS["causal"].read_text(encoding="utf-8")))
    assert "assert_run_resumable" in source            # completed-run refusal
    assert "baseline_parity" in source                 # parity gate artifact
    assert "BASELINE PARITY FAILED" in source          # abort path
    assert "completed_condition_ids" in source         # append-safe resume
    assert "condition_plan.json" in source
    assert "isotropic_random_direction" in source
    assert "configs/gemma_jspace_causal_smoke.yaml" in source
    assert "configs/causal_smoke_examples.json" in source or "manifest_path" in source
    assert "explorer_causal_bundle.json" in source


def test_multimodal_notebook_specific_guarantees():
    source = _full_source(json.loads(NOTEBOOKS["multimodal"].read_text(encoding="utf-8")))
    assert "configs/gemma_multimodal_jlens_capture.yaml" in source
    assert "IMAGE_PATH" in source and "AUDIO_PATH" in source
    assert "supports_audio" in source                  # interface detection
    assert "no fake data will be substituted" in source
    assert "multimodal_explorer_bundle.json" in source
    assert "image_record.json" in source and "audio_record.json" in source
    assert "exploratory" in source                     # no invariance claim
    assert "make_cone_record" in source                # reuses pursuit pipeline
    assert "gradient_pursuit" in source
