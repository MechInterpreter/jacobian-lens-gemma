# SPDX-License-Identifier: Apache-2.0
"""The committed country-workspace notebook matches its tested builder."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILDER = ROOT / "scripts" / "_build_country_workspace_notebook.py"
NOTEBOOK = (
    ROOT / "notebooks" / "multimodal_country_workspace_generalization_colab.ipynb"
)


def _module():
    spec = importlib.util.spec_from_file_location("country_notebook_builder", BUILDER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source() -> str:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


def test_notebook_is_exact_builder_output() -> None:
    assert json.loads(NOTEBOOK.read_text(encoding="utf-8")) == _module().build()


def test_every_code_cell_compiles() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"cell-{index}", "exec")


def test_default_configuration_spends_nothing() -> None:
    source = _source()
    assert "RUN_REAL_COUNTRY_WORKSPACE = False" in source
    assert "RUN_STAGE0_PREPARE_DATA = False" in source
    assert "RUN_STAGE1_FIT_POOLED_LENS = False" in source
    assert "RUN_STAGE2_CAPABILITY_AND_LOCALIZATION = False" in source
    assert "RUN_STAGE2_DEBUG_COUNTRY_INSTRUMENT = False" in source
    assert "RUN_STAGE2_REFIT_TASK_MATCHED_LENS_IF_NEEDED = False" in source
    assert "RUN_STAGE3_DEVELOPMENT_SWAP = False" in source
    assert "RUN_STAGE4_FRESH_CONFIRMATION = False" in source
    assert "RUN_STAGE3B_FRANCE_CHINA_DOWNSTREAM_DEVELOPMENT = False" in source
    assert "RUN_STAGE4B_FRANCE_CHINA_FRESH_CONFIRMATION = False" in source
    assert "CONFIRM_MODEL_LOAD = False" in source
    assert "CONFIRM_FP32_A100 = False" in source
    assert 'SCIENTIFIC_IMPLEMENTATION_ID = "country-prompt-lens-band-debug-v4.20260826"' in source
    assert '"scientific_implementation_id": SCIENTIFIC_IMPLEMENTATION_ID' in source
    assert "PARENT_LENS_CHECKSUM" in source


def test_real_run_requires_fp32_80gb_a100() -> None:
    source = _source()
    assert 'if "A100" not in name or total_gib < 70' in source
    assert "dtype=torch.float32" in source
    assert 'observed_dtype != "torch.float32"' in source
    assert "LENS.save(str(temporary), dtype=torch.float32)" in source


def test_cpu_preparation_is_pinned_audited_and_resumable() -> None:
    source = _source()
    assert "revision=DATASET_REVISION" in source
    assert "revision=dataset_revision, streaming=True" in source
    assert "pytesseract.image_to_string" in source
    assert 'prep_unit_root = DATA_ROOT / "prep_units" / dataset_revision' in source
    assert '"prep_unit_checksum": payload_checksum(prepared_row)' in source
    assert "prepared population failed its own checksum" in source
    assert "speed = 135 + 3 * ordinal" in source
    assert 'sf.write(audio_path, waveform, 16000, subtype="FLOAT")' in source


def test_scientific_endpoint_and_exchange_are_literal() -> None:
    source = _source()
    assert "unrestricted_greedy_completion" in source
    assert "unrestricted_greedy_swap_trial" in source
    assert '("exact", 1.0, exact)' in source
    assert 'position_rule="all_prompt_positions"' in source
    assert "alpha=1.0" in source
    assert "teacher_forcing" not in source
    assert "candidate_ids" not in source


def test_france_china_followup_is_fixed_and_preserves_parent_no_go() -> None:
    source = _source()
    assert '"France->China" not in full_band.get("passing_directions", ())' in source
    assert 'layers=LAYERS, source=tokens["France"], target=tokens["China"]' in source
    assert 'FRANCE_CHINA_DEVELOPMENT["verdict"] == "COUNTRY_FRANCE_CHINA_DEVELOPMENT_GO"' in source
    assert "fresh confirmation remains unopened" in source
    assert "PARENT TWO-PAIR COUNTRY VERDICT REMAINS UNCHANGED" in source
    assert "RUN_STAGE2_REFIT_TASK_MATCHED_LENS_IF_NEEDED" in source


def test_path_selection_cannot_read_exact_swap_results() -> None:
    source = _source()
    assert "unrestricted_greedy_direct_answer_trial" in source
    assert '"condition": "direct_answer"' in source
    assert "selection_used_exact_swap_outcomes" not in source
    assert "development did not pass both" in source
    assert "clean source capability did not cover two pairs" in source
    assert "Stage 3 requires completed Stage 2 reports" not in source


def test_identity_calibration_not_direct_answer_selects_development_band() -> None:
    source = _source()
    assert 'IDENTITY_CALIBRATION_REPORT["selected"]["band"]' in source
    assert 'LOCALIZATION_REPORT["verdict"] != "COUNTRY_DIRECT_PATHS_GO"' not in source
    assert "DEVELOPMENT NOT LICENSED: direct-answer localization did not pass" not in source
    assert "freeze_identity_calibrated_confirmation_design" in source
    assert '"success": identity_matches(result["generated_text"], target)' in source
    assert '"downstream_facts_hidden_during_band_selection": True' in source


def test_country_prompt_and_task_matched_refit_are_explicit() -> None:
    source = _source()
    assert "instruction=COUNTRY_COMPLETION_INSTRUCTION" in source
    assert '"effective_protocol_digest": payload_checksum' in source
    assert '"base_protocol_digest": BASE_PROTOCOL["protocol_digest"]' in source
    assert "country_identity_task_matched_assistant_prefill" in source
    assert "TASK_LENS_PATH" in source
    assert "RUN_STAGE2_REFIT_TASK_MATCHED_LENS_IF_NEEDED" in source


def test_audio_transcript_is_not_passed_to_backend() -> None:
    source = _source()
    audio_branch = source.split("def build_task_inputs", 1)[1].split(
        "def generated_success", 1
    )[0]
    assert 'kwargs.update(audio=waveform' in audio_branch
    assert "speech_text" not in audio_branch
    assert "transcript" not in audio_branch


def test_every_expensive_unit_has_atomic_resume() -> None:
    source = _source()
    assert "checkpoint_every=CHECKPOINT_EVERY" in source
    assert 'STORE.load("capability", key)' in source
    assert 'STORE.save("capability", key, stored)' in source
    assert 'STORE.load("intervention", key)' in source
    assert 'STORE.save("intervention", key, stored)' in source
    assert "STORE.status_report()" in source
