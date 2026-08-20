from __future__ import annotations

import ast
import json
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILDER = ROOT / "scripts" / "_build_workspace_replication_notebook.py"
NOTEBOOK = ROOT / "notebooks" / "multimodal_jspace_workspace_replication_colab.ipynb"


def _notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _source() -> str:
    return "\n".join("".join(cell.get("source", [])) for cell in _notebook()["cells"])


def test_builder_is_byte_identical() -> None:
    before = NOTEBOOK.read_bytes()
    runpy.run_path(str(BUILDER), run_name="__main__")
    assert NOTEBOOK.read_bytes() == before


def test_notebook_is_output_free_and_parses() -> None:
    for cell in _notebook()["cells"]:
        if cell["cell_type"] != "code":
            continue
        assert cell.get("outputs") == []
        assert cell.get("execution_count") is None
        ast.parse("".join(cell["source"]))


def test_notebook_enforces_the_paper_first_order() -> None:
    source = _source()
    for required in (
        "anthropic_text_tasks",
        "text_replication_verdict",
        "capture_source_loading",
        "select_pair_from_loading",
        "freeze_loading_localization",
        "freeze_confirmation_design",
        "assert_fresh_population",
        "MULTIMODAL_PRIMARY_ALPHA = 1.0",
        "MULTIMODAL_SENSITIVITY_ALPHA = 0.75",
        '"teacher_forcing_used": False',
        '"candidate_list_supplied": False',
        "random_two_direction_basis",
        "unrelated_alpha1",
        "position_rule_by_modality",
        "STORE.save(\"activation\"",
        "STORE.save(\"intervention\"",
        "FRESH_MULTIMODAL_DOWNSTREAM_RECOMPUTATION_GO",
    ):
        assert required in source


def test_all_result_switches_default_false() -> None:
    source = _source()
    for name in (
        "RUN_REAL_WORKSPACE_REPLICATION",
        "RUN_STAGE1_TEXT_REPLICATION",
        "RUN_STAGE2_MULTIMODAL_LOADING_DEVELOPMENT",
        "RUN_STAGE3_FREEZE_DESIGN",
        "RUN_STAGE4_FRESH_CONFIRMATION",
        "RUN_STAGE5_WRITE_REPORT",
        "CONFIRM_MODEL_LOAD",
        "CONFIRM_DEVELOPMENT_BUDGET",
        "CONFIRM_CONFIRMATION_BUDGET",
    ):
        assert f"{name} = False" in source
