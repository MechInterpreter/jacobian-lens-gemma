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
    """Requirement 19. Editing the notebook by hand fails here.

    The checked-in bytes are restored before the assertion, so a drifted
    notebook fails the test instead of being silently repaired by running it.
    """
    before = NOTEBOOK.read_bytes()
    subprocess.run([sys.executable, str(BUILDER)], cwd=ROOT, check=True)
    rebuilt = NOTEBOOK.read_bytes()
    if rebuilt != before:
        NOTEBOOK.write_bytes(before)
    assert rebuilt == before, (
        "the checked-in notebook is not what its builder produces; edit "
        f"{BUILDER.name} and regenerate rather than editing the notebook"
    )


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
    # Read-only: nothing writes into a completed causal run. Every store write
    # in the notebook belongs to one of this session's own stores, and the
    # completed band-interior run is protected the same way.
    remaining = source
    for allowed in ("SWAP_STORE.save", "BAND_STORE.save", "CORRECTED_STORE.save"):
        remaining = remaining.replace(allowed, "")
    assert "STORE.save" not in remaining
    assert "SUPERSEDED_BAND_RUN_DIR," in source  # a member of PROTECTED_RUN_DIRS


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

    # The corrected control ran too: three commissioned cases, and the case
    # that matters most is the previously confirmed layer failing.
    assert namespace["RUN_STAGE2C_CORRECTED_PREFLIGHT"] is False
    assert namespace["RUN_STAGE2G_CORRECTED_CONFIRMATION"] is False
    assert namespace["CORRECTED_STORE"] is None
    assert namespace["CORRECTED_VERDICT"] is None
    corrected = namespace["CORRECTED_MOCK_RESULTS"]
    assert set(corrected) == {
        "all_nine_pass", "one_interior_fails", "previously_confirmed_fails"
    }
    assert corrected["all_nine_pass"]["verdict"] == "BAND_CORRECTED_CONTROL_GO"
    assert corrected["all_nine_pass"]["full_band_available"] is True
    assert corrected["all_nine_pass"]["largest_admissible_contiguous_band"] == [32, 40]
    for key, failing, band in (
        ("one_interior_fails", [36], [32, 35]),
        ("previously_confirmed_fails", [35], [36, 40]),
    ):
        assert corrected[key]["verdict"] == "BAND_CORRECTED_CONTROL_NO_GO", key
        assert corrected[key]["full_band_available"] is False, key
        assert corrected[key]["layers_failing"] == failing, key
        assert corrected[key]["largest_admissible_contiguous_band"] == band, key
        assert corrected[key]["stage3_unblocked"] is False, key
    assert namespace["WRONG_LAYER_MAPPING"] == {
        8: 40, 14: 40, 20: 40, 26: 8,
        32: 8, 33: 8, 34: 8, 35: 8, 36: 8, 37: 8, 38: 8, 39: 8, 40: 8,
    }
    assert namespace["SUPERSEDED_WRONG_LAYER_MAPPING"] == {
        33: 39, 34: 39, 36: 33, 37: 33, 39: 33
    }
    assert namespace["CORRECTED_BUDGET"]["backward_passes"] == 0
    assert namespace["CORRECTED_BUDGET"]["fitting_performed"] is False


def test_the_no_go_prose_does_not_tell_the_operator_to_rerun_the_inventory():
    """Requirement 18 of the correction.

    Under `BAND_INTERIOR_LENS_NO_GO` the notebook used to print, unconditionally,
    that re-running section 3 would pick the new lenses up. Nothing was
    published for a failed layer, so the inventory cannot admit it and stage 3
    stays blocked. The instruction is now guarded by the GO branch and the NO-GO
    branch says the opposite.
    """
    source = _source(_payload())
    guard = 'if BAND_VERDICT["verdict"] == "BAND_INTERIOR_LENS_GO":'
    assert guard in source
    before, after = source.split(guard, 1)
    assert "the inventory picks the new lenses up" not in before
    go_branch, no_go_branch = after.split("else:", 1)
    assert "inventory picks the newly published" in go_branch
    for phrase in (
        "STAGE 3 REMAINS BLOCKED",
        "re-running the section 3 inventory cannot admit them",
        "a layer that failed",
    ):
        assert phrase in no_go_branch, phrase
    assert "publishable" not in source


def test_the_corrected_control_is_wired_in_and_gates_the_causal_stage():
    """The correction's notebook requirements: a CPU-only preflight that prints
    the explicit mapping and freezes the protocol, a GPU confirmation over all
    nine band layers, and a stage 3 that will not run without a full-band GO."""
    source = _source(_payload())

    # switches, in the required order and both defaulting to False
    assert "RUN_STAGE2C_CORRECTED_PREFLIGHT = False" in source
    assert "RUN_STAGE2G_CORRECTED_CONFIRMATION = False" in source
    assert "CONFIRM_CORRECTED_READOUT_BUDGET = False" in source

    # the mapping is printed before any confirmation data can be opened
    assert "format_wrong_layer_mapping" in source
    assert "superseded=SUPERSEDED_WRONG_LAYER_MAPPING" in source
    index_mapping = source.index("print(format_wrong_layer_mapping(")
    index_population = source.index("build_corrected_confirmation_population(")
    assert index_mapping < index_population

    # the fixed universe and the nine scored layers
    assert "FIXED_CONTROL_UNIVERSE" in source
    assert "scoring_layers=BAND_SCORING_LAYERS" in source

    # protocol persisted before the population is opened
    index_persist = source.index('CORRECTED_STORE.save("corrected_protocol", "protocol"')
    assert index_persist < index_population
    assert "assert_protocol_persisted(CORRECTED_STORE)" in source

    # the budget is printed in section 5, which runs before section 6 can load
    # a model, and stage 2G will not start without an explicit confirmation
    assert "format_corrected_readout_budget" in source
    assert "corrected_readout_budget(d_model=EXPECT_D_MODEL)" in source
    index_budget = source.index("format_corrected_readout_budget(CORRECTED_BUDGET)")
    assert index_budget < source.index("load_gemma4(")
    assert "stage 2G needs CONFIRM_MODEL_LOAD and CONFIRM_CORRECTED_READOUT_BUDGET" in (
        source
    )

    # resumable readout, atomic units, and units bound to their population
    assert "score_corrected_readout_rows(" in source
    assert 'stage="corrected_readout"' in source
    assert "store=CORRECTED_STORE" in source
    assert 'manifest_checksum=CORRECTED_MANIFEST["manifest_checksum"]' in source
    assert "manifest_checksum=DEVELOPMENT_MANIFEST_CHECKSUM" in source

    # publication into a new versioned directory, never over an artifact
    assert "publish_corrected_layer(" in source
    assert "corrected_dir=CORRECTED_STORE.corrected_dir" in source
    assert "protected_run_dirs=PROTECTED_RUN_DIRS" in source
    assert "SUPERSEDED_BAND_RUN_DIR" in source
    assert "assert_superseded_run_unchanged(" in source

    # stage 3 is blocked without a full-band GO
    assert 'if CORRECTED_VERDICT["full_band_available"]' in source or (
        'if not CORRECTED_VERDICT["full_band_available"]:' in source
    )
    assert "stage 3 is blocked: the corrected confirmation" in source
    assert "a contiguous band needs every one of its nine" in source

    # nothing rewrites the completed run
    assert "immutable historical evidence" in source
    assert "superseded set-dependent-control validation" in source

    # the corrected stages fit nothing: the only call to the fitting entry
    # point is stage 1's, and it is guarded by stage 1's own switch
    assert source.count("run_calibration(") == 3  # 1 real + 2 in the MOCK section
    guarded = source.split("if RUN_STAGE1_FIT_MISSING_LENSES:\n", 1)[1]
    assert guarded.lstrip().startswith("CONTINUATION = run_calibration(")
    corrected_stage = source.split("if RUN_STAGE2G_CORRECTED_CONFIRMATION:", 1)[1]
    corrected_stage = corrected_stage.split("## 11.", 1)[0]
    assert "run_calibration" not in corrected_stage
    assert "jlens.fitting" not in corrected_stage


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
