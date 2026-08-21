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
        "build_assistant_prefill_completion_inputs",
        "text_capability_verdict",
        "text_replication_verdict",
        "unrestricted_greedy_completion",
        "unrestricted_greedy_swap_trial",
        "unrestricted_greedy_direct_answer_trial",
        "text_swap_diagnostic_report",
        "TEXT_DIAGNOSTIC_BANDS",
        "direct_answer_norm_matched",
        "derived_from_stage1",
        "post_cast_max_relative_coordinate_error",
        "model_dtype_realization",
        "TEXT_MODEL_DTYPE_REALIZATION",
        "diagnostic_token_ids",
        "capture_source_loading",
        "select_pair_from_loading",
        "freeze_loading_localization",
        "freeze_confirmation_design",
        "assert_fresh_population",
        "MULTIMODAL_PRIMARY_ALPHA = TEXT_PRIMARY_ALPHA",
        "else 0.75",
        '"teacher_forcing_used": False',
        '"candidate_list_supplied": False',
        "random_two_direction_basis",
        "unrelated_alpha1",
        '"zero"',
        "both_directions_tested",
        "direct_answer_success",
        "paired_text_tests",
        "maximum_completed_forward_passes_lost_on_disconnect",
        "position_rule_by_modality",
        "STORE.save(\"activation\"",
        "STORE.save(\"intervention\"",
        "FRESH_MULTIMODAL_DOWNSTREAM_RECOMPUTATION_GO",
    ):
        assert required in source
    assert "answer_token_table(BACKEND" not in source
    assert 'BACKEND.build_inputs(prompt=task.prompt, modality="text")' not in source
    assert 'STORE.save("capability"' in source
    assert 'TEXT_CAPABILITY["causal_spending_licensed"]' in source
    assert '== "LOADING_FIRST_INSTRUMENT_GO"' in source
    assert "select_loading_instrument" in source
    assert "RUN_STAGE0_FIT_MATCHED_R_LENSES" in source
    assert "dense_relprop_backward" in source
    assert "matched_pooled_r" in source
    assert '"causal_outcome_may_select_band": False' in source
    assert "TEXT_DIAGNOSTIC_BANDS = (tuple(ACTIVE_TEXT_LAYERS),)" in source
    assert "SCIENTIFIC_IMPLEMENTATION_COMMIT" in source
    assert '"c6b5dc144051a13ae163c89d2bfb5a0f955e9288"' in source
    assert 'STUDY_LAYER_WINDOW = "late_jr_l33_l40"' in source
    assert 'STUDY_LAYER_WINDOW == "early_r_l27_l32"' in source
    assert "LAYERS = tuple(range(27, 33))" in source
    assert "late J-lens artifacts are historical controls" in source
    assert 'STUDY_LAYER_WINDOW == "combined_r_l27_l40"' in source
    assert "LAYERS = tuple(range(27, 41))" in source
    assert "combine_disjoint_layer_lenses" in source
    assert "combined_without_refitting" in source
    assert "fitting passes 0" in source
    assert "TEXT_PRIMARY_ALPHA = 2.0" in source
    assert '"exact_primary_swapped_answer_generated"' in source
    assert '"primary_alpha_is_paper_double_strength"' in source
    assert 'STORE.save("intervention", key, trial)' in source
    assert '_design_kwargs["text_diagnostic"] = TEXT_DIAGNOSTIC_REPORT' in source
    assert '_design_kwargs["text_verdict"] = TEXT_VERDICT' in source


def test_all_result_switches_default_false() -> None:
    source = _source()
    for name in (
        "RUN_REAL_WORKSPACE_REPLICATION",
        "RUN_STAGE1_TEXT_REPLICATION",
        "RUN_STAGE1B_TEXT_DIAGNOSTIC",
        "RUN_STAGE2_MULTIMODAL_LOADING_DEVELOPMENT",
        "RUN_STAGE3_FREEZE_DESIGN",
        "RUN_STAGE4_FRESH_CONFIRMATION",
        "RUN_STAGE5_WRITE_REPORT",
        "CONFIRM_MODEL_LOAD",
        "CONFIRM_TEXT_DIAGNOSTIC_BUDGET",
        "CONFIRM_DEVELOPMENT_BUDGET",
        "CONFIRM_CONFIRMATION_BUDGET",
    ):
        assert f"{name} = False" in source


def test_clean_completion_is_not_given_an_intervention_policy() -> None:
    """The realization policy belongs only to calls that install swap hooks."""

    calls = []
    for cell in _notebook()["cells"]:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]))
        calls.extend(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "unrestricted_greedy_completion"
        )
    assert calls
    for call in calls:
        assert "realization_policy" not in {
            keyword.arg for keyword in call.keywords if keyword.arg is not None
        }
