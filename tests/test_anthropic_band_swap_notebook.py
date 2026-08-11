"""The band-swap notebook: generated, output-free, MOCK-executable, honest."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "notebooks" / "multimodal_jspace_anthropic_band_swap_colab.ipynb"
BUILDER = ROOT / "scripts" / "_build_anthropic_band_swap_notebook.py"
HISTORICAL = (
    ROOT / "notebooks" / "multimodal_jspace_anthropic_reasoning_swap_colab.ipynb"
)


def _payload():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code(payload):
    return [cell for cell in payload["cells"] if cell["cell_type"] == "code"]


def _source(payload):
    return "\n".join("".join(cell["source"]) for cell in payload["cells"])


def test_notebook_is_output_free_and_every_cell_parses():
    """Requirement 18."""
    payload = _payload()
    assert payload["metadata"]["accelerator"] == "GPU"
    for cell in _code(payload):
        assert cell["outputs"] == []
        assert cell["execution_count"] is None
        ast.parse("".join(cell["source"]))


def test_notebook_matches_its_generator():
    """Requirement 19. Editing the notebook by hand fails here."""
    before = NOTEBOOK.read_bytes()
    subprocess.run([sys.executable, str(BUILDER)], cwd=ROOT, check=True)
    assert NOTEBOOK.read_bytes() == before


def test_the_historical_single_layer_notebook_is_untouched():
    """Requirement 17, for the artifact that lives in the repository.

    The completed run's notebook is evidence of what was executed. This builder
    writes to a different path and never regenerates it.
    """
    builder = BUILDER.read_text(encoding="utf-8")
    target = next(
        line for line in builder.splitlines() if line.startswith("TARGET = ")
    )
    assert "multimodal_jspace_anthropic_band_swap_colab.ipynb" in target
    assert "reasoning_swap" not in target
    assert builder.count("TARGET.write_text") == 1
    assert HISTORICAL.is_file()
    historical = json.loads(HISTORICAL.read_text(encoding="utf-8"))
    assert "independent_layer_record" in "\n".join(
        "".join(cell["source"]) for cell in historical["cells"]
    )


def test_the_design_is_the_contiguous_band_clamp():
    source = _source(_payload())
    assert "run_swap_condition" in source
    assert "POSITION_RULE = PRIMARY_POSITION_RULE" in source
    assert "position_rule=POSITION_RULE" in source
    assert "every original prompt position" in source
    assert "predeclare_suffix_bands" in source
    assert "assert_contiguous" in source
    assert "BAND_START_LAYERS = (32, 35, 38, 40)" in source
    assert "band STARTS, not a sampled layer grid" in source
    # The refusals the single-layer study carried must not be here.
    assert "len(band) != 1" not in source
    assert "repeated_exchange_forbidden" not in source
    assert "one_exchange_per_forward_pass" not in source
    assert "independent_layer_record" not in source
    assert "anthropic_independent_single_layer_coordinate_swap" not in source


def test_alpha_roles_are_fixed_and_controls_are_intensity_matched():
    """The notebook takes its condition set and its alphas from the module.

    Restating either in the notebook is how they drift, so what is asserted
    here is that it does not: the alphas come from ``PRIMARY_ALPHA`` and
    ``SECONDARY_ALPHA``, every condition's alpha comes from ``CONDITION_ALPHA``,
    and the control basis is chosen by the condition's own prefix.
    """
    source = _source(_payload())
    assert "ALPHAS = (PRIMARY_ALPHA, SECONDARY_ALPHA)" in source
    assert "for condition in BAND_CONDITIONS:" in source
    assert "alpha=CONDITION_ALPHA[condition]" in source
    assert 'condition.startswith("unrelated_")' in source
    assert 'condition.startswith("random_")' in source
    assert 'banks[f"random_{arm}"]' in source
    assert "never swept per sample" in source
    assert "extrapolation" in source
    assert "swept" not in source.replace("never swept per sample", "")


def test_every_expensive_unit_is_fingerprint_gated_and_atomically_saved():
    source = _source(_payload())
    assert "UnitStore(SWAP_RUN_DIR, _fingerprint)" in source
    assert 'SWAP_STORE.save("capability", key' in source
    assert 'SWAP_STORE.save("intervention", key' in source
    assert 'SWAP_STORE.has("intervention", key)' in source
    assert 'BAND_STORE.save(' in source
    assert "checkpoint_every=25" in source
    assert "os.replace(" in source
    # What the run binds itself to, as the notebook passes it in. The shape of
    # the resulting record is pinned in tests/test_anthropic_band_swap.py.
    for bound in (
        "design=DESIGN",
        "lens_checksums=LENS_CHECKSUMS",
        "transformers_version=TRANSFORMERS_VERSION_EXPECTED",
        "audio_protocol_fingerprint=AUDIO_RECORD",
        "prompt_protocol=PROMPT_PROTOCOLS",
        "directed_pairs=DIRECTED_PAIRS",
        "sample_identities=",
        "thresholds=THRESHOLDS.to_dict()",
        "coordinate_swap_method_version=METHOD_VERSION",
        "scoring_rule=",
        "design_digest",
        "inventory_digest",
        "population_digest",
    ):
        assert bound in source, bound


def test_the_completed_runs_are_read_only_and_their_photographs_are_spent():
    """The single-layer runs are read for one thing: what to exclude.

    Their run directories are in `PROTECTED_RUN_DIRS`, so publication refuses to
    write into them, and every photograph they screened is excluded from this
    study's population — those populations were examined while their successors
    were designed, so drawing from them again would select images already known
    to be capability-valid.
    """
    source = _source(_payload())
    assert "PROTECTED_RUN_DIRS" in source
    assert "protected_run_dirs=PROTECTED_RUN_DIRS" in source
    assert '"completed_single_layer_run_read_or_modified": False' in source
    assert "COMPLETED_CAUSAL_RUNS" in source
    for run in ("mmpaper_real_", "mmpaper2_real_", "mmpaperconfirm_real_"):
        assert run in source, run
    assert "does not match its pinned report" in source
    assert "the new population overlaps the exclusion set" in source
    assert "prior_exclusion_digest" in source
    # Read-only: nothing writes into a completed causal run.
    assert "STORE.save" not in source.replace("SWAP_STORE.save", "").replace(
        "BAND_STORE.save", ""
    )


def test_safe_defaults_execute_the_whole_mock_pipeline_without_drive_or_cuda():
    """Requirement 18's other half: the default path runs, and spends nothing."""
    payload = _payload()
    namespace = {"__name__": "__main__"}
    old_cwd = os.getcwd()
    try:
        os.chdir(ROOT)
        for cell in _code(payload):
            exec(compile("".join(cell["source"]), str(NOTEBOOK), "exec"), namespace)
    finally:
        os.chdir(old_cwd)

    assert namespace["MODE"] == "mock"
    assert namespace["RUN_STAGE0_PREFLIGHT"] is False
    assert namespace["RUN_STAGE1_FIT_MISSING_LENSES"] is False
    assert namespace["RUN_STAGE2_CONFIRM_AND_PUBLISH"] is False
    assert namespace["RUN_STAGE3_BAND_SWAP"] is False
    assert namespace["RUN_STAGE4_TIMING"] is False
    assert namespace["SWAP_STORE"] is None
    assert namespace["MODEL"] is None
    assert namespace["POSITION_RULE"] == "all_prompt_positions"
    assert namespace["ALPHAS"] == (1.0, 2.0)
    assert set(namespace["BAND_CONDITIONS"]) == {
        "swap_alpha1", "swap_alpha2", "zero",
        "random_alpha1", "random_alpha2", "unrelated_alpha1", "unrelated_alpha2",
    }

    # Stage 1 ran for real on the synthetic stack, over the real interior grid.
    assert namespace["MOCK_FIT"].n_done == 12
    verdicts = namespace["MOCK_LENS_RESULTS"]
    assert verdicts["all_interior_pass"]["verdict"] == "BAND_INTERIOR_LENS_GO"
    assert verdicts["all_interior_pass"]["full_band_available"] is True
    assert verdicts["one_interior_fails"]["verdict"] == "BAND_INTERIOR_LENS_NO_GO"
    assert verdicts["one_interior_fails"]["largest_admissible_contiguous_band"] == [32, 35]

    # Stages 3 and 4 produced a verdict and the timing comparison.
    assert len(namespace["BAND_RECORDS"]) > 0
    assert namespace["REASONING"]["verdict"].startswith("BAND_SWAP_")
    timing = namespace["TIMING"]
    assert timing["verdict"] == "BAND_ONSET_INTERMEDIATE_EARLIER"
    row = timing["pairs"][0]
    assert (
        row["deepest_effective_start"]["intermediate"]
        < row["deepest_effective_start"]["answer"]
    )
    assert namespace["REPORT"]["mock_proves_pipeline_only"] is True


def test_the_notebook_says_what_a_mock_run_proves():
    source = _source(_payload())
    assert "green MOCK run is evidence about this code and about nothing else" in source
    assert "mock_proves_pipeline_only" in source
    assert "No scientific claim about Gemma 4" in source


def test_every_repository_symbol_the_notebook_imports_exists():
    from jlens.mmpilot.band_lens import (
        band_capture_plan,
        band_layer_verdict,
        band_scale_selection,
        build_band_evaluation_splits,
        discover_extension_scale250_lenses,
        discover_published_band_lenses,
        publish_band_layer,
        verify_reconstructed_extension_splits,
    )
    from jlens.mmpilot.band_swap import (
        band_design_record,
        band_onset_timing,
        band_reasoning_verdict,
        band_swap_fingerprint,
        band_trial_record,
        lens_inventory,
        predeclare_suffix_bands,
        summarize_band_cells,
    )
    from jlens.mmpilot.coordinate_swap import coordinate_swap_band, run_swap_condition

    assert all(
        callable(value)
        for value in (
            band_capture_plan, band_layer_verdict, band_scale_selection,
            build_band_evaluation_splits, discover_extension_scale250_lenses,
            discover_published_band_lenses, publish_band_layer,
            verify_reconstructed_extension_splits, band_design_record,
            band_onset_timing, band_reasoning_verdict, band_swap_fingerprint,
            band_trial_record, lens_inventory, predeclare_suffix_bands,
            summarize_band_cells, coordinate_swap_band, run_swap_condition,
        )
    )
