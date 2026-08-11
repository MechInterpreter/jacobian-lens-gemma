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
    assert "physical_layers_between_samples_are_unpatched" not in source


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
    assert namespace["BACKEND"] is None
    assert namespace["STORE"] is None
    assert namespace["TOTAL_PASSES"] == 0


def test_real_path_is_one_clean_three_switch_run():
    source = _source(_payload())
    assert "RUN_REAL_PAPER_SWAP = False" in source
    assert "CONFIRM_MODEL_LOAD = False" in source
    assert "CONFIRM_PASS_BUDGET = False" in source
    assert source.count("= False") >= 3
    assert "RUN_REAL_PAPER_SWAP = True" not in source
    assert "expected the pinned 125,198-group cache" in source
    assert "build_expanded_manifest" not in source


def test_every_repository_symbol_imported_by_the_real_dictionary_cell_exists():
    from jlens.mmpilot.coordinate_swap import (
        METHOD_VERSION,
        build_swap_bases,
        random_two_direction_basis,
        resolve_concept_token,
    )
    from jlens.mmpilot.pipeline import build_dictionaries
    from jlens.mmpilot.store import RunFingerprint, UnitStore, payload_checksum

    assert METHOD_VERSION == "jlens.mmpilot.coordinate_swap.v1"
    assert all(
        callable(value)
        for value in (
            build_swap_bases,
            random_two_direction_basis,
            resolve_concept_token,
            build_dictionaries,
            RunFingerprint,
            UnitStore,
            payload_checksum,
        )
    )
