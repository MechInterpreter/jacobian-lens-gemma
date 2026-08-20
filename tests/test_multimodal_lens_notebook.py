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
        "answer_equivalence_record",
        "load_completed_causal_source",
        "N_CAUSAL_CANDIDATES_PER_CONCEPT = 96",
        "previously screened images excluded",
        "matched_multimodal_jlens_unrestricted_swap.v3",
        "open_answer_matches",
        "imported read-only",
        "RUN_STAGE3B_ALPHA_SWEEP",
        "ALPHA_SWEEP = (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)",
        '"alpha_selected_after_outcomes": ALPHA_SWEEP_OUTCOME_INFORMED',
        "outcome_informed_stable_range_refinement",
        "load_completed_alpha_sweep_source",
        "matched_multimodal_jlens_unrestricted_alpha_refinement.v6",
        "mmpilot.multimodal_alpha_identity_specificity.v1",
        '"property_endpoint_used_for_selection": False',
        '"exploratory_best_alpha": _exploratory_best_alpha',
        "target_logit_delta",
        "target_rank_improvement",
        "controls_are_intensity_matched",
        "population_reused_without_reselection",
        "alpha1_exact_outcome_parity",
        "Alpha=1 remains primary",
    ):
        assert required in source
    assert "int(_clean_logits.argmax()) == int(_expected)" not in source


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
