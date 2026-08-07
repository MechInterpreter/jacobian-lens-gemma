# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Structure and full MOCK execution of the coordinate-swap notebook.

The execution test runs every code cell in one namespace with the committed
switch left alone, so it checks two things at once: that the notebook works end
to end on CPU, and that opening it starts nothing. A second run flips the switch
and asserts that the notebook *refuses* — there is no real experiment yet, and
the notebook must not improvise one.
"""

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = (
    REPO_ROOT / "notebooks" / "multimodal_jspace_coordinate_swap_mock_colab.ipynb"
)
RUNNER = Path(__file__).resolve().parent / "_coordinate_swap_notebook_runner.py"

REQUIRED_SECTIONS = [
    "colab bootstrap",
    "the switch, and why it refuses",
    "a synthetic, deliberately nonorthogonal j-lens pair",
    "what the implementation refuses",
    "the algebra, checked numerically",
    "the same concept, arriving through three evidence channels",
    "the paper's protocol: every prompt position, over a layer band",
    "downstream recomputation",
    "the controls",
    "the involution, measured",
    "the spec, the fingerprint, atomic storage, and resume",
    "summary",
]


@pytest.fixture(scope="module")
def notebook():
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def _code_cells(payload):
    return [cell for cell in payload["cells"] if cell["cell_type"] == "code"]


def _source(payload):
    return "\n".join("".join(cell["source"]) for cell in payload["cells"])


def _code_source(payload):
    return "\n".join("".join(cell["source"]) for cell in _code_cells(payload))


# ---------------------------------------------------------------- structure


def test_notebook_is_valid_and_output_free(notebook):
    assert notebook["nbformat"] == 4
    for index, cell in enumerate(_code_cells(notebook)):
        assert cell["outputs"] == [], f"code cell {index} has stored outputs"
        assert cell["execution_count"] is None, f"code cell {index} has an exec count"


def test_all_code_cells_parse_and_use_no_shell_magics(notebook):
    for index, cell in enumerate(_code_cells(notebook)):
        source = "".join(cell["source"])
        try:
            ast.parse(source)
        except SyntaxError as exc:  # pragma: no cover - failure path
            pytest.fail(f"code cell {index} does not parse: {exc}")
        for line in source.splitlines():
            assert not line.strip().startswith(("!", "%")), (
                f"code cell {index} uses a shell/line magic: {line!r}"
            )


def test_required_sections_are_present_in_order(notebook):
    text = _source(notebook).lower()
    position = 0
    for heading in REQUIRED_SECTIONS:
        found = text.find(heading, position)
        assert found >= 0, f"missing section {heading!r}"
        position = found


def test_the_notebook_never_loads_gemma(notebook):
    source = _code_source(notebook)
    for forbidden in (
        "AutoProcessor",
        "AutoModel",
        "Gemma4ForConditionalGeneration",
        "from_pretrained",
        "google/gemma",
        "drive.mount",
        "GemmaPilotBackend",
        "huggingface_hub",
    ):
        assert forbidden not in source, f"the MOCK notebook references {forbidden!r}"


def test_the_only_switch_is_committed_false_and_refuses(notebook):
    source = _code_source(notebook)
    assert "RUN_REAL_COORDINATE_SWAP = False" in source
    assert "RUN_REAL_COORDINATE_SWAP = True" not in source
    assert "raise NotImplementedError" in source
    # The refusal has to say *why*, in terms of the recorded calibration record.
    assert "CONTIGUOUS band" in source
    assert "untouched confirmation set" in source


def test_the_notebook_states_the_two_methods_and_the_claim_boundaries(notebook):
    text = _source(notebook)
    assert "source-derived J-space steering" in text
    assert "h + \\alpha\\, V\\,(\\sigma(c) - c)" in text or "V (sigma(c) - c)" in text
    assert "spoken captions" in text
    assert "Behavioral outputs stay **text**" in text
    assert "separate claims" in text
    assert "MOCK SUCCESS IS NOT SCIENTIFIC EVIDENCE" in text
    assert "nothing at all" in text


def test_the_notebook_does_not_touch_a_completed_run(notebook):
    source = _source(notebook)
    for forbidden in ("rgext_real", "MyDrive", "/content/drive", "runs/pilot_", "jlens.calibration"):
        assert forbidden not in source, f"the MOCK notebook references {forbidden!r}"


# ---------------------------------------------------------------- execution


def _run(*overrides):
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment["MMPILOT_REPO_DIR"] = str(REPO_ROOT)
    return subprocess.run(
        [sys.executable, str(RUNNER), str(NOTEBOOK_PATH), *overrides],
        cwd=REPO_ROOT.parent,
        env=environment,
        capture_output=True,
        text=True,
        timeout=900,
    )


@pytest.fixture(scope="module")
def executed():
    result = _run()
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_the_notebook_runs_end_to_end_on_cpu(executed):
    assert executed["ok"] is True
    assert executed["run_real_coordinate_swap"] is False
    assert executed["mode"] == "mock"
    assert executed["status"] == "MOCK_PASSED"
    assert executed["loaded_gemma"] is False
    assert executed["torch_cuda_initialized"] is False


def test_the_mock_result_is_what_the_method_predicts(executed):
    summary = executed["summary"]
    assert summary["ran_real_experiment"] is False
    assert summary["loaded_gemma"] is False
    assert summary["intervention_family"] == "anthropic_coordinate_swap"
    assert summary["position_rule"] == "all_prompt_positions"

    assert summary["algebra"]["alpha_zero_is_bit_exact"] is True
    assert summary["algebra"]["double_swap_recovers"] < 1e-12
    assert summary["algebra"]["coordinates_exchanged"] < 1e-12
    assert summary["algebra"]["orthogonal_drift"] < 1e-12

    assert summary["lens_pair"]["numerical_rank"] == 2
    assert summary["lens_pair"]["cosine"] == pytest.approx(0.45, abs=1e-9)

    # Identity replaced in every evidence channel; property recomputed with it.
    assert summary["identity_predictions"] == {
        "text": "cat", "image": "cat", "spoken_audio": "cat"
    }
    assert summary["property_predictions"] == {
        "text": "four", "image": "four", "spoken_audio": "four"
    }
    # ... but not when the band starts after the property was computed.
    assert summary["layer_band_control"] == {"identity": "cat", "property": "two"}


def test_the_mock_controls_are_all_exercised(executed):
    controls = executed["summary"]["controls"]
    assert controls["coordinate_swap"] == "cat"
    for null_control in (
        "zero",
        "random_two_direction_norm_matched",
        "unrelated_pair_swap",
        "position_control",
        "reverse_swap",
    ):
        assert controls[null_control] == "bird", null_control
    # The direct-answer control moves the property without the identity, which
    # is exactly why it is required beside every downstream result.
    assert executed["summary"]["direct_answer_control"] == {
        "property": "four", "identity": "bird"
    }


def test_the_mock_exercises_every_refusal_and_the_resume_gate(executed):
    summary = executed["summary"]
    assert set(summary["refusals"]) == {
        "identical_source_and_target",
        "ill_conditioned_pair",
        "multi_token_concept",
        "position_rule_without_evidence_span",
        "rank_deficient_pair",
        "transposed_basis",
        "unvalidated_layer_in_band",
    }
    assert summary["resume"]["first_open"] == "starting"
    assert summary["resume"]["second_open"] == "resuming"
    assert summary["resume"]["units"] == 3
    assert len(summary["resume_refusals"]) >= 10


def test_the_band_parity_diagnostic_is_reported(executed):
    parity = executed["summary"]["band_parity"]
    assert [row[1] for row in parity] == ["cat", "bird", "cat", "bird"]


def test_flipping_the_switch_refuses_instead_of_running_anything():
    result = _run("RUN_REAL_COORDINATE_SWAP=True")
    assert result.returncode == 1
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["ok"] is False
    assert "NotImplementedError" in payload["error"]
    assert "does not exist yet" in payload["error"]


# ------------------------------------------------------------ documentation

SCHEMATIC_SVG = REPO_ROOT / "docs" / "assets" / "intervention_methods.svg"
SCHEMATIC_PNG = REPO_ROOT / "docs" / "assets" / "intervention_methods.png"
PROTOCOL_DOC = REPO_ROOT / "docs" / "coordinate_swap_protocol.md"
PILOT_DOC = REPO_ROOT / "docs" / "multimodal_jspace_pilot.md"


def _flat(text: str) -> str:
    """Markdown reflow must not be able to break a documentation check."""
    return " ".join(text.split())


def test_the_schematic_has_an_editable_source_and_a_rendered_png():
    assert SCHEMATIC_SVG.is_file(), "the schematic source must be repo-native text"
    assert SCHEMATIC_PNG.is_file(), "run scripts/render_schematic.py"
    assert SCHEMATIC_PNG.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert (REPO_ROOT / "scripts" / "render_schematic.py").is_file()


def test_the_schematic_shows_both_methods_and_not_the_wrong_equation():
    svg = SCHEMATIC_SVG.read_text(encoding="utf-8")
    # A: the completed pilot's steering.
    assert "source-derived J-space steering" in svg
    assert "h' = h  ±  alpha · v_concept" in svg
    # B: the coordinate swap, in full.
    assert "V = [v_source  v_target]" in svg
    assert "c = pinv(V) h" in svg
    assert "h_patched = h + alpha·V(sigma(c) - c)" in svg
    # The equation this schematic exists to correct must appear only as the
    # thing being replaced, inside the comment that says so.
    body = svg.split("-->", 1)[1]
    assert "d_target" not in body and "delta_target" not in body

    for statement in (
        "The completed three-modality causal result used method A",
        "will use method B",
        "all original prompt",
        "validated layer band",
        "Behavioral outputs remain text",
        "evidence",
        "spoken_audio means spoken captions",
        "Identity replacement and downstream recomputation are separate claims",
    ):
        assert statement in svg, statement


def test_the_protocol_doc_records_the_audit_and_the_boundaries():
    text = _flat(PROTOCOL_DOC.read_text(encoding="utf-8"))
    for required in (
        "row `v`** of `W_U @ J_l`",
        "final_norm_weight=None",
        "atom index **is** the vocabulary token id",
        "float64",
        "rank-deficient",
        "ill-conditioned",
        "all_prompt_positions",
        "final_prompt_token_only",
        "35, 38 and 40",
        "direct-answer-vector control",
        "intervention_family",
        "involution",
        "spoken captions",
        "separate claims",
    ):
        assert required in text, required
    assert "assets/intervention_methods.png" in text


def test_the_pilot_doc_corrects_the_terminology_without_retracting_the_result():
    text = _flat(PILOT_DOC.read_text(encoding="utf-8"))
    section = text.split("## The completed causal result is steering", 1)
    assert len(section) == 2, "the pilot doc must state the A/B distinction"
    body = section[1]
    assert "source-derived positive-minus-negative J-space directions" in body
    assert "stand unchanged" in body
    assert "valid causal steering experiment" in body
    assert "must not be described as one" in body
    assert "h_patched = h + alpha * V (sigma(c) - c)" in body
    assert "No real coordinate-swap run exists" in body
    assert "coordinate_swap_protocol.md" in body
