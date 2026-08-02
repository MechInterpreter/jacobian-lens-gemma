# SPDX-License-Identifier: Apache-2.0
"""Light-path guarantees for the focused text-only recalibration notebook."""

import ast
import json
from pathlib import Path

from jlens.metadata import load_config

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "notebooks" / "gemma_4_e4b_text_jlens_recalibration_colab.ipynb"
CONFIG = ROOT / "configs" / "gemma_text_recalibration.yaml"


def _payload():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _source():
    return "\n".join("".join(cell["source"]) for cell in _payload()["cells"])


def test_recalibration_notebook_is_valid_compilable_and_output_free():
    payload = _payload()
    assert payload["nbformat"] == 4
    for cell in payload["cells"]:
        if cell["cell_type"] != "code":
            continue
        assert cell["execution_count"] is None
        assert cell["outputs"] == []
        ast.parse("".join(cell["source"]))


def test_default_light_path_executes_without_model_or_drive(monkeypatch):
    monkeypatch.chdir(ROOT)
    namespace = {"__name__": "__notebook__"}
    for index, cell in enumerate(_payload()["cells"]):
        if cell["cell_type"] == "code":
            exec(  # noqa: S102 - this intentionally executes committed notebook cells
                compile("".join(cell["source"]), f"<cell {index}>", "exec"), namespace
            )
    assert namespace["RUN_RECALIBRATION"] is False
    assert namespace["MODEL"] is None
    assert namespace["LENS"] is None
    assert namespace["VALIDATION"] is None


def test_recalibration_is_text_only_gated_and_resume_safe():
    source = _source()
    assert "RUN_RECALIBRATION = False" in source
    assert "if RUN_RECALIBRATION:" in source
    assert "apply_chat_template" in source
    assert "add_generation_prompt=True" in source
    assert "config_fingerprint" in source
    assert "incompatible calibration" in source
    assert "checkpoint_path=str(checkpoint)" in source
    assert "SpokenCOCO captions" in source


def test_candidate_is_not_published_without_the_random_control_gate():
    source = _source()
    assert "ReconstructionControlConfig" in source
    assert "max_control_pool_atoms=None" in source
    assert "require_pool_match=True" in source
    assert "layers_above_random" in source
    assert "lens.candidate.pt" in source
    assert "lens.validated.pt" in source
    assert source.index("layers_above_random") < source.index("shutil.copyfile")


def test_recalibration_config_is_narrow_and_pinned():
    config = load_config(str(CONFIG))
    assert config["model"]["revision"] == "fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd"
    assert config["sites"]["source_layers"] == [35, 38]
    assert config["fitting"]["n_prompts"] == 32
    assert config["recalibration"]["heldout_prompts"] == 8
    assert config["recalibration"]["require_layers_above_random"] == 1
