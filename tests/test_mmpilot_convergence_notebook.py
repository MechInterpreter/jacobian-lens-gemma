# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Structure and MOCK execution of the output-convergence audit notebook.

The execution tests run every code cell in one namespace from a working
directory where ``jlens`` is not importable, so they check the bootstrap and the
whole audit at once — and, at the committed defaults, that opening the notebook
loads no model and opens no completed run.

**A passing MOCK run is plumbing evidence.** The synthetic world's convergence
trajectory is a knob, so a ``PRE_CONVERGENCE_TRANSFER_SUPPORTED`` there says the
code can detect a separation that was put in on purpose. It is never evidence
about Gemma, and no assertion here treats it as any.
"""

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = (
    REPO_ROOT
    / "notebooks"
    / "multimodal_jspace_output_convergence_audit_colab.ipynb"
)
RUNNER = Path(__file__).resolve().parent / "_mmpilot_convergence_notebook_runner.py"
BUILDER = REPO_ROOT / "scripts" / "_build_convergence_notebook.py"

COMMITTED_SWITCHES = (
    "RUN_REAL_CONVERGENCE_AUDIT",
    "CONFIRM_MODEL_LOAD",
    "RUN_SECONDARY_PROBE",
)

REQUIRED_SECTIONS = [
    "the two switches",
    "the predeclared criterion",
    "mock",
    "stage 1 — provenance and integrity",
    "stage 2 — the frozen evaluation population",
    "stage 3 — load gemma",
    "stages 4–7",
    "immutability of the completed run",
]

PUBLISHED_PINS = {
    35: "sha256:64fb02d718ac48adc1bced99e2eff3c2215052ba144d5dedac05f17936a96ed1",
    38: "sha256:c8508fbf2b916e5d9aaeb8711a30f76414ee16478c5f6cc321e57e2fe846d1c0",
    40: "sha256:8a90f67eeb9bb5db14e6715b8bc516a899da1c3210d0662ec7fa177b5409f7d7",
}


@pytest.fixture(scope="module")
def notebook():
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def _code_cells(payload):
    return [cell for cell in payload["cells"] if cell["cell_type"] == "code"]


def _source(payload):
    return "\n".join("".join(cell["source"]) for cell in payload["cells"])


def _code_source(payload):
    return "\n".join("".join(cell["source"]) for cell in _code_cells(payload))


# ----------------------------------------------------------------- structure


def test_notebook_is_valid_and_output_free(notebook):
    assert notebook["nbformat"] == 4
    for index, cell in enumerate(_code_cells(notebook)):
        assert cell["outputs"] == [], f"code cell {index} has stored outputs"
        assert cell["execution_count"] is None, f"code cell {index} has an exec count"


def test_all_code_cells_parse(notebook):
    for index, cell in enumerate(_code_cells(notebook)):
        try:
            ast.parse("".join(cell["source"]))
        except SyntaxError as exc:  # pragma: no cover - failure path
            pytest.fail(f"code cell {index} does not parse: {exc}")


def test_the_committed_notebook_matches_its_generator():
    before = NOTEBOOK_PATH.read_text(encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(BUILDER)], capture_output=True, text=True, cwd=REPO_ROOT
    )
    assert result.returncode == 0, result.stderr
    assert NOTEBOOK_PATH.read_text(encoding="utf-8") == before


def test_the_sections_appear_in_order(notebook):
    headings = re.findall(r"^## (\d+)\. (.+)$", _source(notebook), re.MULTILINE)
    numbers = [int(number) for number, _ in headings]
    assert numbers == sorted(numbers)
    lowered = [title.lower() for _, title in headings]
    for expected in REQUIRED_SECTIONS:
        assert any(expected in title for title in lowered), expected


def test_every_switch_is_committed_false(notebook):
    """An *assignment* at column zero, which is what the reader edits by hand."""
    source = _code_source(notebook)
    for name in COMMITTED_SWITCHES:
        assignments = re.findall(
            rf"^{re.escape(name)} = (True|False)$", source, re.MULTILINE
        )
        assert assignments == ["False"], (name, assignments)


def test_the_two_commissioned_switch_names_are_present(notebook):
    source = _code_source(notebook)
    assert "RUN_REAL_CONVERGENCE_AUDIT = False" in source
    assert "CONFIRM_MODEL_LOAD = False" in source


def test_the_real_path_pins_the_model_and_the_three_lenses(notebook):
    source = _code_source(notebook)
    assert '"google/gemma-4-E4B-it"' in source
    assert '"fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd"' in source
    for layer, checksum in PUBLISHED_PINS.items():
        assert checksum in source, layer
    assert "jlens.mmpilot.native_spoken_audio.v1" in source


def test_the_run_specific_pins_are_empty_and_refused_when_unset(notebook):
    """The two values that belong to the run, not the repo, are never guessed."""
    source = _code_source(notebook)
    assert 'EXPECTED_RUN_FINGERPRINT_DIGEST = ""' in source
    assert 'EXPECTED_PROCESSOR_REVISION = ""' in source
    assert "refusing to audit an unpinned run" in source
    # And the notebook says outright that echoing the run back is not verifying.
    assert "printing is not verifying" in source.lower()


def test_the_completed_run_is_named_never_discovered(notebook):
    source = _code_source(notebook)
    assert "mmaudio_native_audio_transfer_20260806T144822" in source
    assert "refuses to create or discover" in source


def test_the_audit_directory_must_be_outside_the_completed_run(notebook):
    source = _code_source(notebook)
    assert "is inside the completed run" in source
    assert "never writes into the run it audits" in source


def test_the_notebook_reruns_none_of_the_completed_experiment(notebook):
    """No stage of the completed study may be invoked from here."""
    source = _code_source(notebook)
    for forbidden in (
        "stage_capability",
        "stage_activations",
        "stage_jspace",
        "stage_directions",
        "stage_interventions",
        "estimate_concept_direction",
        "run_condition",
        "gradient_pursuit",
        "build_dictionary",
    ):
        assert forbidden not in source, forbidden


def test_the_interpretation_boundary_is_stated_in_the_notebook(notebook):
    # Markdown emphasis and line wrapping are normalized away so the prose can
    # be written for a reader rather than for this assertion.
    source = re.sub(r"\s+", " ", _source(notebook).lower().replace("*", ""))
    assert "not proof that linguistic information is absent" in source
    assert "before native direct-readout convergence" in source
    assert "never \"pre-linguistic\"" in source
    assert "never \"language-free\"" in source


def test_the_criterion_is_printed_before_any_result_producing_cell(notebook):
    """Section 4 prints the rule; the MOCK world runs in section 5."""
    cells = _code_cells(notebook)
    criterion_cell = next(
        index for index, cell in enumerate(cells) if "print(CRITERION_TEXT)" in "".join(cell["source"])
    )
    mock_cell = next(
        index
        for index, cell in enumerate(cells)
        if "run_mock_convergence_audit(" in "".join(cell["source"])
    )
    audit_cell = next(
        index
        for index, cell in enumerate(cells)
        if "RESULT = run_convergence_audit(" in "".join(cell["source"])
    )
    assert criterion_cell < mock_cell < audit_cell


def test_layer_32_is_named_only_as_never_audited(notebook):
    source = _code_source(notebook)
    assert "FAILED_CONFIRMATION_LAYERS" in source
    assert "assert_lens_valid_layer" in source
    assert "never audited" in _source(notebook).lower()


# ----------------------------------------------------------------- execution


def _run(tmp_path, *overrides, extra_env=None, expect_failure=False):
    tmp_path = Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = ""
    environment["MMPILOT_REPO_DIR"] = str(REPO_ROOT)
    environment.update(extra_env or {})
    result = subprocess.run(
        [sys.executable, str(RUNNER), str(NOTEBOOK_PATH), *overrides],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        env=environment,
    )
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    if expect_failure:
        assert result.returncode != 0, f"expected a refusal:\n{result.stdout[-4000:]}"
        assert payload["ok"] is False
        return payload
    assert result.returncode == 0, (
        f"stdout:\n{result.stdout[-4000:]}\n\nstderr:\n{result.stderr[-2000:]}"
    )
    return payload


@pytest.fixture(scope="module")
def executed(tmp_path_factory):
    return _run(tmp_path_factory.mktemp("nb_default"))


def test_the_committed_defaults_run_the_mock_and_nothing_else(executed):
    assert executed["run_real"] is False
    assert executed["confirm_model_load"] is False
    assert executed["completed_run_opened"] is False
    assert executed["model_is_none"] is True
    assert executed["head_is_none"] is True


def test_no_model_module_is_ever_imported_on_the_mock_path(executed):
    """"No model is loaded" has to be observable, not asserted."""
    assert executed["forbidden_modules_imported"] == []


def test_the_bootstrap_resolves_jlens_to_this_checkout(executed):
    assert executed["repo_path"] == str(REPO_ROOT)
    assert Path(executed["jlens_file"]).resolve().parent.parent == REPO_ROOT.resolve()


def test_the_fixed_design_is_the_confirmed_layers(executed):
    assert executed["layers"] == [35, 38, 40]
    assert executed["primary_layer"] == 35
    assert executed["lens_invalid_layers"] == [32]
    assert executed["modalities"] == ["text", "image", "spoken_audio"]


def test_the_criterion_is_printed_and_digested(executed):
    assert executed["criterion_text_present"] is True
    assert executed["criterion_digest"].startswith("sha256:")


def test_the_mock_exercises_every_verdict_branch(executed):
    matrix = executed["mock_matrix"]
    assert matrix["pre_convergence"] == "PRE_CONVERGENCE_TRANSFER_SUPPORTED"
    assert matrix["converged_early"] == "TRANSFER_AT_OR_AFTER_CONVERGENCE"
    assert matrix["ambiguous"] == "INCONCLUSIVE_CONVERGENCE_TIMING"
    assert matrix["degenerate"] == "INCONCLUSIVE_CONVERGENCE_TIMING"
    assert set(matrix) >= {
        "pre_convergence",
        "converged_early",
        "ambiguous",
        "flat_weak",
        "degenerate",
        "pre_convergence_without_causal_support",
    }


def test_a_weak_readout_alone_does_not_produce_the_claim(executed):
    """The notebook asserts this itself; the test confirms the assertion ran."""
    matrix = executed["mock_matrix"]
    assert matrix["flat_weak"] == "INCONCLUSIVE_CONVERGENCE_TIMING"
    assert (
        matrix["pre_convergence_without_causal_support"]
        == "INCONCLUSIVE_CONVERGENCE_TIMING"
    )


def test_the_mock_audit_reports_a_clean_measurement(executed):
    assert executed["mock_verdict"] == "PRE_CONVERGENCE_TRANSFER_SUPPORTED"
    assert executed["mock_controls_passed"] is True
    assert executed["mock_run_unchanged"] is True
    assert executed["mock_readout_mode"] == "single_token_complete"
    assert executed["mock_norm_convention"] == "rmsnorm_one_plus_weight"


def test_the_capability_filter_is_applied_on_the_mock_path(executed):
    assert executed["mock_admissible_concepts"] == ["cat", "toilet"]
    assert executed["mock_inadmissible_concepts"] == ["zebra"]


def test_the_secondary_probe_never_determines_the_verdict(executed):
    assert executed["mock_probe_ran"] is True
    assert executed["mock_probe_determines_verdict"] is False


def test_every_commissioned_artifact_is_written(executed):
    entries = set(executed["audit_entries"])
    assert {
        "output_convergence_report.md",
        "output_convergence_summary.json",
        "per_sample_direct_readout.jsonl",
        "layer_convergence_table.csv",
        "provenance.json",
        "checksums.json",
        "figure_convergence_versus_layer.svg",
        "figure_causal_versus_convergence.svg",
        "figure_per_modality_trajectories.svg",
    } <= entries


def test_the_audit_writes_nothing_into_the_completed_run(executed):
    entries = executed["completed_run_entries"]
    assert entries
    assert not any(name.startswith("output_convergence") for name in entries)
    assert "readout_units" not in entries


def test_the_layer_table_keeps_the_three_layers_apart(executed):
    assert executed["table_layers"] == [35, 38, 40]
    for column in (
        "causal_transfer_verdict",
        "convergence_classification",
        "lens_validity_gate_passed",
        "n_distinct_target_images",
    ):
        assert column in executed["table_columns"], column


def test_the_verdict_carries_every_commissioned_check(executed):
    assert set(executed["verdict_checks"]) == {
        "integrity",
        "controls",
        "readout_is_interpretable",
        "readout_not_degenerate",
        "trajectory_monotone_within_tolerance",
        "primary_layer_causal_transfer_still_supported",
        "later_validated_layer_clearly_more_converged",
        "principal_evidence_is_capability_admissible_only",
    }
    assert executed["interpretation_boundary_in_verdict"] is True


def test_the_mock_execution_is_deterministic(tmp_path_factory):
    """Same world, same numbers — from two clean interpreters in two directories.

    The audit fingerprint is deliberately *not* compared: it binds the completed
    run's directory, which is a fresh temporary path each time. Determinism is a
    claim about the measurements, and that is what is checked.
    """
    first = _run(tmp_path_factory.mktemp("nb_a"))
    second = _run(tmp_path_factory.mktemp("nb_b"))
    assert first["mock_matrix"] == second["mock_matrix"]
    assert first["mock_verdict"] == second["mock_verdict"]
    assert first["criterion_digest"] == second["criterion_digest"]
    assert first["table_agreements"] == second["table_agreements"]
    assert first["table_classifications"] == second["table_classifications"]


def test_the_real_path_refuses_without_a_pinned_fingerprint(tmp_path_factory):
    """Turning the real switch on is not enough; the run must be pinned."""
    payload = _run(
        tmp_path_factory.mktemp("nb_real"),
        "RUN_REAL_CONVERGENCE_AUDIT=True",
        expect_failure=True,
    )
    # It fails at the completed run, not at some later stage: nothing is
    # computed and no model is touched.
    assert "fingerprint.json" in payload["error"] or "unpinned" in payload["error"]


def test_the_real_path_refuses_to_load_the_model_without_confirmation(
    tmp_path_factory, monkeypatch
):
    """The model gate is separate from the real-mode gate, and says why."""
    source = _code_source(json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8")))
    assert "if not CONFIRM_MODEL_LOAD:" in source
    assert "cannot be reconstructed" in source
    assert "RMSNorm weight" in source
    assert "no forward pass is executed" in source
