from __future__ import annotations

import ast
import json
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILDER = ROOT / "scripts" / "_build_multimodal_lens_notebook.py"
NOTEBOOK = ROOT / "notebooks" / "multimodal_jspace_matched_jlens_colab.ipynb"


def _notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _source() -> str:
    return "\n".join(
        "".join(cell.get("source", [])) for cell in _notebook()["cells"]
    )


def test_builder_is_byte_identical() -> None:
    before = NOTEBOOK.read_bytes()
    runpy.run_path(str(BUILDER), run_name="__main__")
    assert NOTEBOOK.read_bytes() == before


def test_notebook_is_output_free_and_all_code_parses() -> None:
    notebook = _notebook()
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        assert cell.get("outputs") == []
        assert cell.get("execution_count") is None
        ast.parse("".join(cell["source"]))


def test_notebook_contains_the_four_arm_scientific_contract() -> None:
    source = _source()
    for required in (
        "LENS_ARMS, fit_arm, plan_units",
        "capture_eval_rows",
        "clean_capability_required_in_every_modality_and_endpoint",
        '"controls": ["random", "unrelated"]',
        '"teacher_forcing_used": False',
        '"candidate_list_supplied": False',
        "unrestricted_swap_trial",
        "checkpoint_every=CHECKPOINT_EVERY",
    ):
        assert required in source


def test_mock_notebook_executes_end_to_end(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TMP", str(tmp_path))
    monkeypatch.setenv("TEMP", str(tmp_path))
    namespace: dict = {"__name__": "__main__"}
    for index, cell in enumerate(_notebook()["cells"]):
        if cell["cell_type"] == "code":
            exec(
                compile("".join(cell["source"]), f"cell-{index}", "exec"),
                namespace,
            )
    assert namespace["FINAL"]["causal_comparison"]["verdict"] == "MEASURED"
    assert namespace["FINAL"]["cross_evaluation"]["n_rows"] > 0
    assert set(namespace["LENS_CHECKSUMS"]) == {
        "text",
        "image",
        "spoken_audio",
        "pooled",
    }
