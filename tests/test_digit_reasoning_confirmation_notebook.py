from __future__ import annotations

import ast
import importlib.util
import inspect
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = (
    ROOT
    / "notebooks"
    / "multimodal_jspace_digit_reasoning_confirmation_colab.ipynb"
)
BUILDER = ROOT / "scripts" / "_build_digit_reasoning_confirmation_notebook.py"


def _payload():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _source():
    return "\n".join("".join(cell["source"]) for cell in _payload()["cells"])


def test_notebook_is_output_free_and_parses():
    payload = _payload()
    assert payload["metadata"]["accelerator"] == "GPU"
    for cell in payload["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []
            assert cell["execution_count"] is None
            ast.parse("".join(cell["source"]))


def test_notebook_matches_builder_byte_for_byte():
    spec = importlib.util.spec_from_file_location("_digit_builder", BUILDER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    rebuilt = json.dumps(module.build(), indent=1, ensure_ascii=False) + "\n"
    assert rebuilt == NOTEBOOK.read_text(encoding="utf-8")


def test_three_stages_and_two_spend_confirmations_are_visible():
    source = _source()
    for line in (
        "RUN_PREFLIGHT_CPU = False",
        "RUN_DIGIT_CONFIRMATION_GPU = False",
        "RUN_FINAL_REPORT_CPU = False",
        "CONFIRM_MODEL_LOAD = False",
        "CONFIRM_PASS_BUDGET = False",
    ):
        assert line in source
    assert "RUN_DIGIT_CONFIRMATION_GPU and CONFIRM_MODEL_LOAD and CONFIRM_PASS_BUDGET" in source


def test_budget_precedes_model_load_and_no_fitting_is_reachable():
    cells = [
        "".join(cell["source"])
        for cell in _payload()["cells"]
        if cell["cell_type"] == "code"
    ]
    budget = next(i for i, cell in enumerate(cells) if "FINAL CONFIRMATION PASS BUDGET" in cell)
    model = next(i for i, cell in enumerate(cells) if "allow_model_load=True" in cell)
    assert budget < model
    source = _source()
    assert "fitting_performed\": False" in source
    assert '"backward_passes": 0' in source
    assert "fit(" not in source


def test_endpoint_is_digits_alpha2_primary_and_full_vocabulary():
    source = _source()
    for phrase in (
        'DIGIT_ANSWERS: dict[str, str] = {"bird": "2", "cat": "4"}',
        '"2", "4"',
        "PRIMARY_ALPHA = 2.0",
        "score_unrestricted_next_token",
        "greedy_generate",
        "candidate_list_supplied\": False",
        "teacher_forcing_used\": False",
    ):
        # Constants live in the imported module; notebook prose/code carries
        # the operative values and imported names.
        if (
            phrase.startswith("DIGIT_ANSWERS")
            or phrase.startswith("PRIMARY_ALPHA")
            or "candidate_list_supplied" in phrase
            or "teacher_forcing_used" in phrase
        ):
            module_source = (ROOT / "jlens" / "mmpilot" / "digit_reasoning_confirmation.py").read_text(
                encoding="utf-8"
            )
            assert phrase in module_source
        else:
            assert phrase in source


def test_population_excludes_every_completed_causal_family_and_is_never_reused():
    source = _source()
    for run in (
        "mmpaper_real_24be1d028bf1",
        "mmpaper2_real_04ab55235502",
        "mmpaperconfirm_real_6b0745c08d84",
        "mmpaperconfirm_real_a496d5ad7f18",
        "band3340_real_2a72bda9b4ba",
        "mmfv_real_bfb07903e961",
    ):
        assert run in source
    assert "fresh population overlaps a completed causal population" in source
    assert "one synchronized group per photograph" in source


def test_exclusion_avoids_expensive_intervention_scan_unless_needed():
    source = _source()
    assert 'for _stage in (() if _images else ("capability",)):' in source
    assert 'if not _images:\n            _stage = "intervention"' in source
    assert '"identity_source"' in source


def test_atomic_resume_has_separate_score_and_greedy_units():
    source = _source()
    assert 'kind="score"' in source
    assert 'kind="greedy"' in source
    assert 'STORE.save("intervention", _score_key, record)' in source
    assert 'STORE.save("intervention", _greedy_key' in source
    assert "maximum_completed_work_lost_on_disconnect" in source


def test_report_only_path_verifies_every_unit_checksum():
    source = _source()
    assert "def load_valid_units(root, stage, expected_fingerprint_digest):" in source
    assert 'stored.get("unit_checksum") != payload_checksum(payload)' in source
    assert "checksum-invalid/torn units" in source
    assert 'stored.get("fingerprint_digest") != expected_fingerprint_digest' in source


def test_dry_configuration_executes_without_drive_or_model():
    namespace = {"__name__": "__notebook__"}
    for index, cell in enumerate(_payload()["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        exec(compile(source, f"<digit-cell-{index}>", "exec"), namespace)  # noqa: S102
    assert namespace["REAL_MODE"] is False
    assert namespace["GPU_STAGE"] is False
    assert namespace["RUN_DIR"] is None
    assert namespace["REPORT"] is None


def test_real_path_imports_and_keyword_contracts_match_live_package():
    """Catch the import/signature drift that previously wasted Colab loads."""
    from jlens.mmpilot.coordinate_swap import (
        build_swap_basis_from_vectors,
        coordinate_swap_band,
        random_two_direction_basis,
    )
    from jlens.mmpilot.real_backend import (
        build_processor_backend,
        build_real_backend,
    )
    from jlens.mmpilot.tri_modal import assert_audio_protocol
    from jlens.mmpilot.validated_band_followup import (
        discover_corrected_band_lenses,
    )

    inspect.signature(build_processor_backend).bind(
        "repo", revision="rev", token="token"
    )
    inspect.signature(build_real_backend).bind(
        "repo",
        revision="rev",
        token="token",
        device="cuda",
        allow_model_load=True,
        resolve_audio=True,
        expect_n_layers=42,
        expect_d_model=2560,
        expect_vocab_size=262144,
    )
    inspect.signature(assert_audio_protocol).bind(
        object(), expected_fingerprint="sha256:pin"
    )
    inspect.signature(discover_corrected_band_lenses).bind(
        Path("run"), report={}, layers=tuple(range(33, 41))
    )
    assert callable(build_swap_basis_from_vectors)
    assert callable(random_two_direction_basis)
    assert callable(coordinate_swap_band)
