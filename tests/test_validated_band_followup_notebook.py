"""The L33-L40 follow-up notebook: generated, output-free, MOCK-executable, honest.

Requirement 25, plus the notebook-level halves of the requirements that can only
be checked against what the notebook actually does: the CPU/GPU switch
separation, the budget printed before any model load, the reporting boundary,
and the fact that the completed L32-L40 notebook is not touched.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = (
    ROOT / "notebooks" / "multimodal_jspace_anthropic_band33_40_swap_colab.ipynb"
)
BUILDER = ROOT / "scripts" / "_build_validated_band_followup_notebook.py"
COMPLETED_NOTEBOOK = (
    ROOT / "notebooks" / "multimodal_jspace_anthropic_band_swap_colab.ipynb"
)
COMPLETED_BUILDER = ROOT / "scripts" / "_build_anthropic_band_swap_notebook.py"

#: The builder's own target assignment, so a redirected TARGET fails the test.
TARGET_LINE_MARKER = (
    'TARGET = (\n    ROOT / "notebooks" / '
    '"multimodal_jspace_anthropic_band33_40_swap_colab.ipynb"\n)'
)


def _payload():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code(payload):
    return [cell for cell in payload["cells"] if cell["cell_type"] == "code"]


def _source(payload):
    return "\n".join("".join(cell["source"]) for cell in payload["cells"])


def _code_source(payload):
    return "\n".join("".join(cell["source"]) for cell in _code(payload))


# ---- 25. generated, output-free, deterministic


def test_notebook_is_output_free_and_every_cell_parses():
    payload = _payload()
    assert payload["metadata"]["accelerator"] == "GPU"
    for cell in _code(payload):
        assert cell["outputs"] == []
        assert cell["execution_count"] is None
        ast.parse("".join(cell["source"]))


def test_notebook_matches_its_generator():
    """Editing the notebook by hand fails here.

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


def test_the_completed_l32_l40_notebook_is_untouched():
    """The completed study's notebook and builder are historical evidence.

    This builder writes to a new path and never regenerates that one, which is
    the architectural half of "do not retroactively change the meaning of the
    completed run".
    """
    builder = BUILDER.read_text(encoding="utf-8")
    # This builder writes one file, and it is not the completed study's.
    assert builder.count("TARGET.write_text") == 1
    assert "multimodal_jspace_anthropic_band33_40_swap_colab.ipynb" in builder
    assert "multimodal_jspace_anthropic_band_swap_colab.ipynb" not in builder
    assert str(TARGET_LINE_MARKER) in builder

    # The completed study's builder still writes the completed notebook, and
    # knows nothing about this one.
    assert COMPLETED_NOTEBOOK.is_file() and COMPLETED_BUILDER.is_file()
    completed = COMPLETED_BUILDER.read_text(encoding="utf-8")
    assert "band33_40" not in completed
    assert "validated_band_followup" not in completed
    completed_target = next(
        line for line in completed.splitlines() if line.startswith("TARGET = ")
    )
    assert "multimodal_jspace_anthropic_band_swap_colab.ipynb" in completed_target


def test_the_two_notebooks_write_to_different_run_roots():
    """A follow-up unit can never land in the completed study's directory."""
    source = _code_source(_payload())
    completed = _source(
        json.loads(COMPLETED_NOTEBOOK.read_text(encoding="utf-8"))
    )
    assert 'FOLLOWUP_RUN_ROOT = RUNS_ROOT / "mmband33"' in source
    assert "band3340_real_" in source
    assert "mmband33" not in completed
    assert "band3340_real_" not in completed


# ---- the reporting boundary, in the notebook


def test_the_notebook_states_the_reporting_boundary_before_anything_else():
    payload = _payload()
    source = _source(payload)
    code_source = _code_source(payload)

    assert "format_followup_boundary()" in code_source
    # printed in the configuration cell, before any Drive read or model load
    assert code_source.index("print(format_followup_boundary())") < code_source.index(
        "drive.mount"
    )

    for phrase in (
        "BAND_CORRECTED_CONTROL_NO_GO",
        "prospective causal follow-up",
        "not that study's confirmation",
        "supports no claim about a band beginning at L32",
        "no claim about any layer earlier than 33",
        "linguistic spoken captions, not environmental sound",
        "sensitivity evidence",
        "There is no configuration of this notebook",
    ):
        assert phrase in source, phrase


def test_the_notebook_never_calls_the_followup_the_original_study():
    source = _source(_payload())
    assert "confirmatory band" in source  # only in the negation
    for claim in (
        "confirms the L32-L40",
        "confirmation of the L32-L40",
        "the originally planned L32-L40 study is now",
        "L32 is validated",
    ):
        assert claim not in source, claim
    assert "is a **prospective causal follow-up** over L33-L40" in source
    assert "**never** relabelled as confirmation of" in source
    assert "the failed L32-L40 design" in source


def test_the_four_required_preflight_statements_are_printed():
    source = _code_source(_payload())
    for line in (
        "ORIGINAL L32-L40 VERDICT REMAINS",
        "NEW FOLLOW-UP BAND                 L33-L40",
        "SELECTION SOURCE                   lens validation only",
        "NO CAUSAL OUTCOME SELECTED THE BAND",
        "NO FITTING WILL OCCUR",
    ):
        assert line in source, line


# ---- the design, as the notebook expresses it


def test_the_design_is_the_contiguous_band_clamp_over_l33_l40():
    source = _code_source(_payload())
    assert "run_swap_condition" in source
    assert "POSITION_RULE = PRIMARY_POSITION_RULE" in source
    assert "position_rule=POSITION_RULE" in source
    assert "BAND_START_LAYERS = FOLLOWUP_SUFFIX_STARTS" in source
    assert "Band STARTS, not a sampled layer grid" in source
    assert "predeclare_suffix_bands" in source
    assert "assert_contiguous" in source
    assert "assert_band_hook_integrity" in source
    # the intervention is the existing one, not a re-derivation, and the
    # discarded native direct-readout convergence gate is reported as unused
    assert "residual_intervention" not in source
    assert "direct_answer_vector" not in source
    assert 'TIMING["native_direct_readout_convergence_gate_used"]' in source
    assert "source_derived" not in source


def test_the_notebook_takes_its_alphas_and_conditions_from_the_modules():
    source = _code_source(_payload())
    assert "ALPHAS = (PRIMARY_ALPHA, SECONDARY_ALPHA)" in source
    assert "for condition in BAND_CONDITIONS:" in source
    assert "alpha=CONDITION_ALPHA[condition]" in source
    assert 'condition.startswith("unrelated_")' in source
    assert 'condition.startswith("random_")' in source
    assert 'banks[f"random_{arm}"]' in source
    assert "never compared against an alpha=1 baseline" in _source(_payload())


def test_both_arms_and_both_readouts_are_run():
    source = _code_source(_payload())
    assert 'for arm in ("intermediate", "answer"):' in source
    assert 'READOUT_ARMS = ("identity", "property")' in source
    assert "for readout in READOUT_ARMS:" in source
    assert '"answer": selected_bases(' in source
    assert '"intermediate": selected_bases(layers, source, target)' in source


def test_the_transcript_is_offline_only():
    source = _code_source(_payload())
    assert "build_backend_inputs(BACKEND, built, transcript=offline)" in source
    assert "only so build_backend_inputs can prove it is" in source
    assert "It never reaches the model." in source
    # The transcript reaches exactly two places: the Evidence record, where the
    # offline leakage audit reads it, and build_backend_inputs, which exists to
    # prove it is absent from the backend arguments.
    assert source.count("transcript=") == 2
    assert 'transcript=group["caption"]' in source  # the Evidence record
    assert 'prompt=group["caption"]' not in source


# ---- CPU / GPU separation


def test_cpu_stages_never_load_a_model():
    """Requirement 21, at the notebook level."""
    payload = _payload()
    source = _code_source(payload)

    assert "RUN_PREFLIGHT_CPU = False" in source
    assert "RUN_CAUSAL_GPU = False" in source
    assert "RUN_REPORT_CPU = False" in source
    assert "CONFIRM_MODEL_LOAD = False" in source
    assert "CONFIRM_PASS_BUDGET = False" in source

    # every model-loading call sits inside the GPU stage's guard
    gpu_block = source.split("if RUN_CAUSAL_GPU:\n", 1)[1].split(
        "\nelse:\n    print(\"skipped: RUN_CAUSAL_GPU is False\")", 1
    )[0]
    for loader in ("build_real_backend(", "allow_model_load=True", "torch.cuda"):
        assert source.count(loader) >= 1, loader
    assert "build_real_backend(" in gpu_block
    assert source.count("build_real_backend(") == 1
    assert source.count("allow_model_load=True") == 1
    assert "load_gemma4(" not in source

    # the preflight and the report stages are named as CPU and touch no backend
    preflight = source.split("if RUN_PREFLIGHT_CPU or RUN_CAUSAL_GPU:", 1)[1].split(
        "# The claim that nothing is fitted", 1
    )[0]
    for forbidden in ("build_real_backend", "BACKEND", "cuda", "JacobianLens.load"):
        assert forbidden not in preflight, forbidden


def test_the_budget_is_printed_before_any_model_can_load():
    """Requirement 23, at the notebook level."""
    source = _code_source(_payload())
    assert source.index("format_followup_pass_budget(PASS_BUDGET)") < source.index(
        "build_real_backend("
    )
    assert "followup_pass_budget(" in source
    assert "matches_expected_design" in source
    assert "the derived pass budget is not the expected design budget" in source
    assert "Name the factor that changed before allowing a model to load." in source
    assert "stage B needs CONFIRM_MODEL_LOAD and CONFIRM_PASS_BUDGET" in source
    # The budget is derived, never written down. Scoped to code cells on
    # purpose: the prose may quote the expected total as context, but no code
    # path may produce it from a literal.
    assert "11328" not in source and "11,328" not in source
    assert "10752" not in source and "10,752" not in source
    # The expected totals live in the module and are never overridden here.
    assert "expected_clean=" not in source
    assert "expected_intervention=" not in source
    assert "PASS_BUDGET['expected_total']" in source  # read back, not set


def test_no_fitting_entry_point_appears_in_the_notebook():
    """Requirement 22, at the notebook level."""
    source = _code_source(_payload())
    for forbidden in (
        "run_calibration", "jlens.fitting", "fit_jacobian_lens", ".backward(",
        "requires_grad", "torch.enable_grad",
    ):
        assert forbidden not in source, forbidden
    assert "assert_no_fitting_entry_point(" in source
    assert 'FITTING_AUDIT["backward_passes"]' in source


# ---- the completed runs stay read-only


def test_the_completed_runs_are_read_only_and_their_photographs_are_spent():
    """Requirements 1 and 17, at the notebook level."""
    source = _code_source(_payload())
    assert "PROTECTED_RUN_DIRS" in source
    assert "corrected_run_digest(CORRECTED_RUN_DIR)" in source
    assert "assert_corrected_run_unmodified(" in source
    assert "is inside the completed run" in source
    assert "COMPLETED_CAUSAL_RUNS" in source
    for run in ("mmpaper_real_", "mmpaper2_real_", "mmpaperconfirm_real_"):
        assert run in source, run
    assert "does not match its pinned report" in source
    assert "the new population overlaps the exclusion set" in source
    assert "photograph is the independent unit" in source
    assert "zero_overlap_proof" in source
    assert "one_group_per_photograph" in source
    # image ids, group ids, recording paths and the bytes actually scored
    assert '"image_id": group["image_id"]' in source
    assert '"media_checksum": inputs.media_checksum' in source
    assert "media_checksums=MEDIA_CHECKSUMS" in source
    assert 'SWAP_STORE.save("metric", "followup_media_checksums"' in source

    # Every store write belongs to this session's own store.
    remaining = source.replace("SWAP_STORE.save", "")
    assert "STORE.save" not in remaining
    assert "_STORE.save" not in remaining


def test_the_corrected_run_is_explicitly_configured_never_discovered():
    """Preflight clause 1."""
    source = _code_source(_payload())
    assert 'CORRECTED_RUN_DIR = RUNS_ROOT / "mmband" / ORIGINAL_RUN_NAME' in source
    assert "bandcorr_real_" not in source  # the name comes from the module constant
    for discovery in ("glob(\"bandcorr", "sorted(RUNS_ROOT", "latest", "max(RUNS_ROOT"):
        assert discovery not in source, discovery


def test_the_preflight_pins_every_required_checksum():
    """Preflight clauses 2, 3 and 6."""
    source = _code_source(_payload())
    for pin in (
        "expected_report_checksum=EXPECTED_REPORT_CHECKSUM",
        "expected_protocol_digest=EXPECTED_PROTOCOL_DIGEST",
        "expected_universe_checksum=EXPECTED_UNIVERSE_CHECKSUM",
        "EXPECTED_CONFIRMATION_MANIFEST_CHECKSUM",
        "expected_model_repo_id=MODEL_REPO_ID",
        "expected_model_revision=MODEL_REVISION",
        "expected_checksums=EXPECTED_ARTIFACT_CHECKSUMS",
        "assert_followup_band(CORRECTED_REPORT)",
        "discover_corrected_band_lenses(",
    ):
        assert pin in source, pin


def test_a_cpu_only_session_reports_not_run_rather_than_a_finding():
    source = _code_source(_payload())
    assert "CAUSAL_STAGE_RAN = False" in source
    assert "causal_stage_ran=bool(CAUSAL_STAGE_RAN)" in source
    assert source.count("CAUSAL_STAGE_RAN = True") == 3  # GPU, MOCK, stage-C re-read
    assert "capability_sufficient=bool(CAUSAL_STAGE_RAN and CAPABILITY_SUFFICIENT)" in (
        source
    )


def test_stage_b_is_blocked_without_a_printed_admission():
    source = _code_source(_payload())
    assert "stage B is blocked: the section 3 preflight did not admit" in source
    assert "if PREFLIGHT is None or DESIGN is None:" in source
    assert "is not the set" in source


# ---- resume


def test_every_expensive_unit_is_fingerprint_gated_and_atomically_saved():
    """Requirements 19 and 20, at the notebook level."""
    source = _code_source(_payload())
    assert "UnitStore(SWAP_RUN_DIR, _fingerprint)" in source
    assert 'SWAP_STORE.save("capability", key' in source
    assert 'SWAP_STORE.save("intervention", key' in source
    assert 'SWAP_STORE.has("intervention", key)' in source
    assert 'SWAP_STORE.has("capability", key)' in source
    assert "os.replace(" in source
    assert "computed" in source and "reused" in source
    for bound in (
        "design=DESIGN",
        "preflight=PREFLIGHT",
        "lens_checksums=LENS_CHECKSUMS",
        "transformers_version=TRANSFORMERS_VERSION_EXPECTED",
        "audio_protocol_fingerprint=AUDIO_RECORD",
        "prompt_protocol=PROMPT_PROTOCOLS",
        "candidate_token_ids=CANDIDATE_IDS",
        "directed_pairs=DIRECTED_PAIRS",
        "population=POPULATION",
        "exclusion=EXCLUSION",
        "thresholds=THRESHOLDS.to_dict()",
        "seeds=",
        "readout_arms=READOUT_ARMS",
        "coordinate_swap_method_version=METHOD_VERSION",
        "scoring_rule=",
        "corrected_report_checksum",
    ):
        assert bound in source, bound


def test_drive_reads_use_the_existing_retry_mechanism():
    source = _code_source(_payload())
    assert "drive_media_loaders(journal=RetryJournal())" in source


# ---- the whole thing runs


def test_safe_defaults_execute_the_whole_mock_pipeline_without_drive_or_cuda():
    """Requirement 25's other half: the default path runs, and spends nothing."""
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
    assert namespace["RUN_PREFLIGHT_CPU"] is False
    assert namespace["RUN_CAUSAL_GPU"] is False
    assert namespace["RUN_REPORT_CPU"] is False
    assert namespace["SWAP_STORE"] is None
    assert namespace["PREFLIGHT"] is None
    assert namespace["POSITION_RULE"] == "all_prompt_positions"
    assert namespace["ALPHAS"] == (1.0, 2.0)
    assert namespace["POPULATION"] is None

    # the frozen design
    assert list(namespace["PRIMARY_BAND"].layers) == list(range(33, 41))
    assert namespace["BAND_START_LAYERS"] == (33, 35, 38, 40)
    assert [list(b.layers) for b in namespace["SUFFIX_BANDS"]] == [
        list(range(33, 41)), list(range(35, 41)), list(range(38, 41)), [40],
    ]
    assert namespace["DESIGN"]["band_keys"] == ["33-40", "35-40", "38-40", "40-40"]
    assert 32 not in namespace["DESIGN"]["primary_band"]["layers"]

    # the budget, derived
    budget = namespace["PASS_BUDGET"]
    assert budget["clean_candidate_passes"] == 576
    assert budget["intervention_candidate_passes"] == 10_752
    assert budget["total"] == 11_328
    assert budget["matches_expected_design"] is True
    assert budget["backward_passes"] == 0

    # the preflight scenarios, every one as required
    preflight = namespace["PREFLIGHT_MOCK_RESULTS"]
    assert len(preflight) == 16
    assert all(row["as_required"] for row in preflight.values())
    assert preflight["admits_followup"]["refused"] is False
    assert preflight["l32_passes"]["refused"] is True
    assert preflight["l32_published"]["refused"] is True
    assert all(row["fixture_unchanged_by_reading"] for row in preflight.values())

    # the causal scenarios, every one bounded
    causal = namespace["CAUSAL_MOCK_RESULTS"]
    assert set(causal) == {
        "favorable", "null", "control_failure", "alpha2_only",
        "asymmetric_direction", "capability_no_go",
    }
    assert all(row["as_required"] for row in causal.values())
    assert causal["favorable"]["verdict"].endswith("_GO")
    assert causal["control_failure"]["verdict"].endswith("_NULL")
    assert causal["alpha2_only"]["verdict"].endswith("_ALPHA2_SENSITIVITY_ONLY")
    assert causal["capability_no_go"]["verdict"].endswith("_CAPABILITY_NO_GO")

    # stage B and C ran against the synthetic world
    assert len(namespace["BAND_RECORDS"]) > 0
    assert namespace["VERDICT"]["verdict"].startswith("L33_L40_VALIDATED_BAND_FOLLOWUP")
    assert namespace["TIMING"]["verdict"].startswith("L33_L40_VALIDATED_BAND_FOLLOWUP")
    assert namespace["TIMING"]["native_direct_readout_convergence_gate_used"] is False
    assert namespace["TIMING"]["monotonicity_not_asserted_from_nesting"]
    integrity = namespace["MOCK_HOOK_INTEGRITY"]
    assert integrity["every_requested_hook_fired"] is True
    assert integrity["no_candidate_position_patched"] is True

    # the audit and the report
    assert namespace["FITTING_AUDIT"]["no_fitting_entry_point_is_reachable"] is True
    report = namespace["REPORT"]
    assert report["mock_proves_pipeline_only"] is True
    assert report["original_l32_l40_verdict"] == "BAND_CORRECTED_CONTROL_NO_GO"
    assert report["is_the_original_l32_l40_confirmatory_band"] is False
    assert report["completed_corrected_run_read_or_modified"] == "read-only"


def test_the_notebook_says_what_a_mock_run_proves():
    source = _source(_payload())
    assert "A green MOCK run is evidence about this code and about nothing else" in source
    assert "No scientific claim about Gemma 4" in source
    assert "MOCK RUN COMPLETE" in source
    assert 'MODE = "real" if REAL_MODE else "mock"' in source


def test_every_repository_symbol_the_notebook_imports_exists():
    from jlens.mmpilot.band_swap import (
        band_reasoning_verdict,
        band_trial_record,
        build_band,
        predeclare_suffix_bands,
        summarize_band_cells,
    )
    from jlens.mmpilot.coordinate_swap import coordinate_swap_band, run_swap_condition
    from jlens.mmpilot.paper_reasoning_swap import (
        hidden_animal_population,
        select_capability_eligible_samples,
    )
    from jlens.mmpilot.validated_band_followup import (
        assert_band_hook_integrity,
        assert_corrected_run_unmodified,
        assert_followup_band,
        assert_no_fitting_entry_point,
        corrected_run_digest,
        discover_corrected_band_lenses,
        followup_design_record,
        followup_fingerprint,
        followup_onset_timing,
        followup_pass_budget,
        followup_preflight_record,
        followup_report,
        followup_verdict,
        format_followup_boundary,
        format_followup_pass_budget,
        format_followup_preflight,
        format_followup_verdict,
        read_corrected_validation_report,
        read_followup_units,
    )
    from jlens.mmpilot.validated_band_followup_mock import (
        mock_band_keys,
        mock_corrected_run,
        mock_followup_records,
    )

    assert all(
        callable(value)
        for value in (
            band_reasoning_verdict, band_trial_record, build_band,
            predeclare_suffix_bands, summarize_band_cells, coordinate_swap_band,
            run_swap_condition, hidden_animal_population,
            select_capability_eligible_samples, assert_band_hook_integrity,
            assert_corrected_run_unmodified, assert_followup_band,
            assert_no_fitting_entry_point, corrected_run_digest,
            discover_corrected_band_lenses, followup_design_record,
            followup_fingerprint, followup_onset_timing, followup_pass_budget,
            followup_preflight_record, followup_report, followup_verdict,
            format_followup_boundary, format_followup_pass_budget,
            format_followup_preflight, format_followup_verdict,
            read_corrected_validation_report, read_followup_units,
            mock_band_keys, mock_corrected_run, mock_followup_records,
        )
    )
