from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "notebooks" / "multimodal_jspace_anthropic_reasoning_swap_colab.ipynb"
BUILDER = ROOT / "scripts" / "_build_paper_reasoning_swap_notebook.py"


def _payload():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code(payload):
    return [cell for cell in payload["cells"] if cell["cell_type"] == "code"]


def _source(payload):
    return "\n".join("".join(cell["source"]) for cell in payload["cells"])


def test_notebook_is_output_free_and_every_cell_parses():
    payload = _payload()
    assert payload["metadata"]["accelerator"] == "GPU"
    for cell in _code(payload):
        assert cell["outputs"] == []
        assert cell["execution_count"] is None
        ast.parse("".join(cell["source"]))


def test_notebook_matches_generator():
    before = NOTEBOOK.read_bytes()
    subprocess.run([sys.executable, str(BUILDER)], cwd=ROOT, check=True)
    assert NOTEBOOK.read_bytes() == before


def test_design_is_the_paper_comparison_not_the_old_surrogates():
    source = _source(_payload())
    assert "h' = h + V (sigma(pinv(V) h) - pinv(V) h)" in source
    assert '"arms": ["intermediate", "answer"]' in source
    assert 'POSITION_RULE = "all_prompt_positions"' in source
    assert "HIDDEN_ANIMAL_LEGS" in source
    assert "run_swap_condition" in source
    assert "coordinate_swap.v1" in source
    assert "source_derived_jspace_steering" not in source
    assert "NOT_CONVERGED" in source and "not used" in source
    assert "25 evenly spaced" in source
    assert '"bands": [[layer] for layer in grid]' not in source
    assert "independent_layer_record" in source
    assert "repeated_exchange_forbidden" in source
    assert '"one_exchange_per_forward_pass": True' in source
    assert '"position_descriptive_is_blocking": False' in source
    assert '"primary_alpha": (' in source
    assert '2.0 if RUN_INDEPENDENT_ALPHA2_CONFIRMATION else 1.0' in source
    assert '"sensitivity_alpha": (' in source
    assert "mmpaper2_real_" in source


def test_every_expensive_unit_is_fingerprint_gated_and_atomically_saved():
    source = _source(_payload())
    assert "UnitStore(RUN_DIR, fingerprint)" in source
    assert 'STORE.save("capability", key' in source
    assert 'STORE.save("intervention", key' in source
    assert 'STORE.has("capability", key)' in source
    assert 'STORE.has("intervention", key)' in source
    assert "os.replace(tmp, report_path)" in source
    for bound in (
        "population_digest",
        "prior_exclusion_checksum",
        "per_layer_lens_checksums",
        "audio_protocol_fingerprint",
        "threshold_digest",
        "prompt_protocols",
        "conditions",
        "bands",
        "completed_v1_report_checksum",
        "capability_selection_seed",
    ):
        assert bound in source


def test_safe_defaults_execute_without_drive_model_or_cuda():
    payload = _payload()
    namespace = {"__name__": "__main__"}
    old_cwd = os.getcwd()
    try:
        os.chdir(ROOT)
        for cell in _code(payload):
            exec(compile("".join(cell["source"]), str(NOTEBOOK), "exec"), namespace)
    finally:
        os.chdir(old_cwd)
    assert namespace["RUN_REAL_PAPER_SWAP"] is False
    assert namespace["RUN_DIRECTION_MATCHED_AMENDMENT"] is False
    assert namespace["RUN_INDEPENDENT_ALPHA2_CONFIRMATION"] is False
    assert namespace["BACKEND"] is None
    assert namespace["STORE"] is None
    assert namespace["TOTAL_PASSES"] == 0


def test_real_path_is_one_clean_three_switch_run():
    source = _source(_payload())
    assert "RUN_REAL_PAPER_SWAP = False" in source
    assert "CONFIRM_MODEL_LOAD = False" in source
    assert "CONFIRM_PASS_BUDGET = False" in source
    assert "RUN_DIRECTION_MATCHED_AMENDMENT = False" in source
    assert "RUN_INDEPENDENT_ALPHA2_CONFIRMATION = False" in source
    assert source.count("= False") >= 3
    assert "RUN_REAL_PAPER_SWAP = True" not in source
    assert "expected the pinned 125,198-group cache" in source
    assert "build_expanded_manifest" not in source


def test_every_repository_symbol_imported_by_the_real_dictionary_cell_exists():
    from jlens.mmpilot.coordinate_swap import (
        METHOD_VERSION,
        build_swap_basis_from_vectors,
        random_two_direction_basis,
        resolve_concept_token,
    )
    from jlens.mmpilot.paper_reasoning_swap import (
        PAPER_REASONING_SWAP_V2_VERSION,
        amend_paper_v2_report_direction_matching,
        independent_layer_record,
        paper_onset_verdict_v2,
        select_capability_eligible_samples,
    )
    from jlens.mmpilot.store import RunFingerprint, UnitStore, payload_checksum

    assert METHOD_VERSION == "jlens.mmpilot.coordinate_swap.v1"
    assert all(
        callable(value)
        for value in (
            build_swap_basis_from_vectors,
            random_two_direction_basis,
            resolve_concept_token,
            RunFingerprint,
            UnitStore,
            payload_checksum,
            independent_layer_record,
            paper_onset_verdict_v2,
            select_capability_eligible_samples,
            amend_paper_v2_report_direction_matching,
        )
    )
    assert PAPER_REASONING_SWAP_V2_VERSION.endswith(".v2")
    source = _source(_payload())
    assert "build_dictionaries" not in source
    assert "selected lens vectors built" in source


def test_transcript_leakage_guard_is_scoped_to_non_text_modalities():
    source = _source(_payload())
    assert 'offline_transcript = group["caption"] if modality != "text" else None' in source
    assert "BACKEND, built, transcript=offline_transcript" in source
    assert 'build_backend_inputs(BACKEND, built, transcript=group["caption"])' not in source


def test_v2_screens_clean_behavior_before_any_causal_spending():
    source = _source(_payload())
    clean = source.index("select_capability_eligible_samples")
    causal = source.index("run_swap_condition")
    assert clean < causal
    assert 'CAUSAL_SELECTION["all_cells_sufficient"]' in source
    assert 'max_images_per_cell=MAX_ANALYSIS_IMAGES_PER_CELL' in source
    assert 'swap_results_consulted' in source
    assert "except PromptLeakageError as error:" in source
    assert '"status": "leakage_rejected"' in source
    assert '"correct": False' in source
    assert "prompt-leakage rejections" in source


def test_alpha2_has_matched_controls_and_position_is_descriptive_only():
    source = _source(_payload())
    for condition in (
        "swap_alpha1",
        "swap_alpha2",
        "random_alpha1",
        "random_alpha2",
        "unrelated_alpha1",
        "unrelated_alpha2",
        "position_descriptive",
    ):
        assert condition in source
    assert 'condition.startswith("random_")' in source
    assert 'condition.startswith("unrelated_")' in source
    assert "cannot veto a result" in source


def test_completed_run_has_a_cpu_only_direction_matched_amendment():
    source = _source(_payload())
    assert "amend_paper_v2_report_direction_matching" in source
    assert "paper_reasoning_swap_v2_report_direction_matched_v3.json" in source
    assert "scientific_units_recomputed" in source
    assert "The original report and every scientific unit remain unchanged." in source
    assert "mmpaper2_real_04ab55235502" in source
    assert "sha256:b64ce3cec51371769b908d14342fbf42f64a6dccb82f8d235ad81d643815ddc6" in source


def test_independent_alpha2_confirmation_is_fresh_frozen_and_resumable():
    source = _source(_payload())
    assert 'SOURCE_CONCEPTS = ("bird",)' in source
    assert "CANDIDATE_IMAGES_PER_CONCEPT = 128" in source
    assert "MAX_ANALYSIS_IMAGES_PER_CELL = 16" in source
    assert "MIN_ANALYSIS_IMAGES_PER_CELL = 12" in source
    assert "completed v2 pair images excluded" in source
    assert "expected 48 spent pair images in completed v2" in source
    assert "mmpaperconfirm_real" in source
    assert "paper_alpha2_confirmation_verdict" in source
    assert "mmpilot.paper_reasoning_swap_alpha2_confirmation.v2" in source
    assert "paper_reasoning_swap_alpha2_confirmation_report.json" in source
    assert '"independent_alpha2_confirmation": RUN_INDEPENDENT_ALPHA2_CONFIRMATION' in source
    assert '2.0 if RUN_INDEPENDENT_ALPHA2_CONFIRMATION else 1.0' in source
    assert "mmpaperconfirm_real_6b0745c08d84" in source
    assert "expected 288 capability units in the failed screen" in source
    assert "failed capability-screen images excluded" in source
    assert '"causal_results_seen_before_extension": False' in source
    assert '"causal_units_in_parent": 0' in source
    assert "confirmation primary alpha=2 onsets" in source
    assert "secondary alpha=1 onsets" in source
