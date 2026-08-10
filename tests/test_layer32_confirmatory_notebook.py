# SPDX-License-Identifier: Apache-2.0
"""Structural guarantees for the layer-32 confirmatory validation notebook.

These tests exist because the notebook's honesty is mostly a property of its
*order* and its *absences*: the criterion must be printed before any result, no
project module may be imported before the editable install, and there must be
no fitting call anywhere. None of that is visible from a passing run.
"""

import ast
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "notebooks" / "archive" / "completed_studies" / "gemma_4_e4b_layer32_confirmatory_validation_colab.ipynb"


def _payload():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _cells():
    return _payload()["cells"]


def _code_cells():
    return [cell for cell in _cells() if cell["cell_type"] == "code"]


def _source(cell):
    return "".join(cell["source"])


def _full_source():
    return "\n".join(_source(cell) for cell in _cells())


def _code_source():
    return "\n".join(_source(cell) for cell in _code_cells())


# ------------------------------------------------------------------- structure


def test_notebook_is_valid_compilable_and_output_free():
    payload = _payload()
    assert payload["nbformat"] == 4
    for index, cell in enumerate(payload["cells"]):
        if cell["cell_type"] != "code":
            continue
        assert cell["execution_count"] is None, f"cell {index} carries an execution count"
        assert cell["outputs"] == [], f"cell {index} carries stored outputs"
        ast.parse(_source(cell))


def test_notebook_declares_a_gpu_colab_runtime():
    metadata = _payload()["metadata"]
    assert metadata["accelerator"] == "GPU"
    assert metadata["colab"]["gpuType"] == "L4"


# ------------------------------------------------------------------- bootstrap


def test_bootstrap_defines_primitive_constants_before_any_checkout():
    first = _source(_code_cells()[0])
    tree = ast.parse(first)
    assert not [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))], (
        "the first code cell must define primitives only, with no imports"
    )
    for name in ("REPO_URL", "REPO_BRANCH", "REPO_DIR"):
        assert f"{name} = " in first
    assert "experiment/spokencoco-jspace-pilot" in first


def test_checkout_is_idempotent_and_prints_branch_and_commit():
    bootstrap = _source(_code_cells()[1])
    assert "git" in bootstrap and "clone" in bootstrap
    assert "fetch" in bootstrap and "reset" in bootstrap, "a second run must update, not fail"
    assert 'pip", "install", "-e"' in bootstrap or "install\", \"-e\"" in bootstrap
    assert "branch=" in bootstrap and "commit=" in bootstrap
    assert "import jlens" in bootstrap


def test_no_project_module_is_imported_before_the_editable_install():
    """Every ``jlens`` import must live in the bootstrap cell or after it."""
    install_cell = None
    for index, cell in enumerate(_code_cells()):
        if "pip" in _source(cell) and "-e" in _source(cell):
            install_cell = index
            break
    assert install_cell is not None, "no editable install cell found"

    for index, cell in enumerate(_code_cells()[:install_cell]):
        for node in ast.walk(ast.parse(_source(cell))):
            module = None
            if isinstance(node, ast.Import):
                module = node.names[0].name
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
            if module is not None:
                assert not module.startswith("jlens"), (
                    f"code cell {index} imports {module} before the install"
                )


# ------------------------------------------------------------------ gated model


def test_the_run_switch_defaults_to_false():
    source = _code_source()
    assert "RUN_CONFIRMATORY_VALIDATION = False" in source
    assert "RUN_CONFIRMATORY_VALIDATION = True" not in source


def test_every_model_and_drive_cell_is_gated_behind_the_switch():
    for index, cell in enumerate(_code_cells()):
        source = _source(cell)
        needs_gate = any(
            needle in source
            for needle in ("load_gemma4", "drive.mount", "MODEL.forward", "userdata.get")
        )
        if needs_gate:
            assert "if RUN_CONFIRMATORY_VALIDATION:" in source, (
                f"code cell {index} touches the model or Drive without the gate"
            )


def test_default_light_path_executes_without_gemma_or_drive(monkeypatch):
    monkeypatch.chdir(ROOT)
    namespace = {"__name__": "__notebook__"}
    for index, cell in enumerate(_code_cells()):
        exec(  # noqa: S102 - this intentionally executes committed notebook cells
            compile(_source(cell), f"<cell {index}>", "exec"), namespace
        )
    assert namespace["RUN_CONFIRMATORY_VALIDATION"] is False
    for name in ("MODEL", "LENS", "HELDOUT_PROMPTS", "ROWS", "VERDICT", "STORE"):
        assert namespace[name] is None, f"{name} was populated on the light path"


# -------------------------------------------------------------------- no fitting


def test_the_notebook_never_fits_a_lens():
    # Code only: the markdown *names* jlens.fitting.fit in order to say the
    # notebook never calls it.
    source = _code_source()
    for forbidden in ("jlens.fitting", "from jlens import fit", "fit(", "checkpoint_path"):
        assert forbidden not in source, f"the notebook must not fit a lens: found {forbidden!r}"
    assert "JacobianLens.load" in source, "the saved lens must be loaded, not refitted"


def test_the_original_artifacts_are_never_rewritten():
    source = _code_source()
    protected = (
        "lens.validated.pt",
        "validated_lens_manifest.json",
        "native_readout_validation.json",
        "heldout_validation.json",
    )
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = getattr(func, "attr", getattr(func, "id", ""))
            if name in {"write_text", "write_bytes", "save", "copyfile", "replace", "unlink"}:
                rendered = ast.unparse(node)
                assert not any(item in rendered for item in protected), (
                    f"a write targets a protected v2 artifact: {rendered}"
                )
    assert "shutil" not in source


# ----------------------------------------------------------------- layer and set


def test_exactly_layer_32_is_evaluated():
    source = _code_source()
    assert "CONFIRMATORY_LAYER" in source
    assert "LAYER = CONFIRMATORY_LAYER" in source
    assert "layers_evaluated" in source
    assert "at=[LAYER, final_layer]" in source, "only layer 32 and the final layer are recorded"
    assert "source_layers=CONFIG" not in source, "the notebook must not sweep the fitted layers"


def test_thirty_two_new_heldout_prompts_with_overlap_refusal():
    source = _code_source()
    assert "N_CONFIRMATORY_PROMPTS" in source
    assert "excluded_prompt_hashes" in source
    assert "select_confirmatory_prompts" in source
    assert "seed=CONFIRMATORY_PROMPT_SEED" in source
    assert "prompt_manifest.json" in source
    # The rendering-drift guard: if our chat rendering no longer reproduces the
    # v2 prompts, the hash exclusion proves nothing and the run must stop.
    assert "overlap cannot be ruled out" in source


def test_prompts_use_the_gemma_it_chat_protocol():
    source = _code_source()
    assert "apply_chat_template" in source
    assert "add_generation_prompt=True" in source
    assert "chat_instruction" in source


def test_no_spokencoco_or_multimodal_data_is_used():
    source = _full_source().lower()
    for forbidden in ("spokencoco_derived", "mmpilot.store", "image_path", "audio_path", "captions_"):
        assert forbidden not in source, f"the notebook references {forbidden!r}"
    assert "wikitext" in source
    # The only mentions of SpokenCOCO/multimodal are the explicit refusals.
    assert "is not a text-only run directory" in _code_source()


# ------------------------------------------------------------------- criterion


def test_the_criterion_is_predeclared_before_any_result_producing_cell():
    cells = _cells()
    criterion_index = next(
        i for i, cell in enumerate(cells) if "print(CRITERION_TEXT)" in _source(cell)
    )
    for needle in ("MODEL.forward", "evaluate_confirmatory", "select_confirmatory_prompts", "load_gemma4"):
        producing = next(i for i, cell in enumerate(cells) if needle in _source(cell))
        assert criterion_index < producing, (
            f"the criterion must be printed before the cell containing {needle!r}"
        )


def test_the_criterion_is_imported_not_retyped_in_a_result_cell():
    """A criterion typed into the scoring cell could be edited after results
    exist; a module constant bound into the resume fingerprint cannot."""
    source = _code_source()
    assert "CONFIRMATORY_CRITERION" in source
    assert "criterion=CONFIRMATORY_CRITERION" in source
    assert "criterion_digest=CONFIRMATORY_CRITERION.digest" in source
    assert "min_top1_agreement=" not in source, "the criterion must not be redefined inline"
    assert "ConfirmatoryCriterion(" not in source


def test_top10_overlap_is_reported_as_a_non_blocking_secondary():
    source = _code_source()
    assert "secondary_metrics" in source
    assert "mean_top10_overlap" in source or "top10" in source
    assert "* secondary, non-blocking" in source


def test_the_logit_lens_is_recorded_as_a_diagnostic_not_a_control():
    source = _code_source()
    assert "DIAGNOSTIC_VARIANTS" in source
    assert "diagnostics_not_blocking" in source
    assert "logit_lens" in source


def test_all_three_randomised_controls_are_computed():
    source = _code_source()
    assert "build_readout_variants" in source
    assert "CONTROL_VARIANTS" in source
    assert "wrong_layer_mapping" in source, "the wrong-layer substitution must be recorded"
    assert "control_metrics.json" in source


# --------------------------------------------------------------- verify and save


def test_every_precondition_is_verified_before_the_run():
    source = _code_source()
    assert "verify_saved_lens" in source
    assert "LENS_CHECKSUM" in source
    assert "sha256:4b17bf6086901e633f94d3391f5de6eccd3e735cc24cece63887505d73641c2b" in source
    assert "fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd" in source
    assert "expect_d_model" in source
    assert "source_site" in source
    assert "missing required v2 artifact" in source


def test_every_required_artifact_is_saved_under_the_confirmatory_directory():
    source = _code_source()
    assert "layer32_confirmatory_validation" in source
    for artifact in (
        "config.json",
        "prompt_manifest.json",
        "aggregate_metrics.json",
        "control_metrics.json",
        "verdict.json",
        "report.md",
    ):
        assert artifact in source, f"{artifact} is never written"
    assert "STORE.save_result(" in source, "results must be saved one prompt at a time"


def test_resume_status_is_printed_and_reuse_is_counted():
    source = _code_source()
    assert "RESUME_STATUS = STORE.open()" in source
    assert "run state:" in source
    assert "STORE.load_result(" in source
    assert "reused=" in source and "computed=" in source


def test_the_verdict_is_one_of_the_two_declared_strings():
    source = _full_source()
    assert "VALIDATED_FOR_MULTIMODAL_FOLLOWUP" in source
    assert "LAYER32_CONFIRMATORY_NO_GO" in source
    assert 'print("VERDICT:", VERDICT["verdict"])' in _code_source()


def test_a_pass_publishes_a_separate_manifest_and_leaves_the_original_alone():
    source = _code_source()
    assert "layer32_confirmatory_manifest.json" in source
    assert "original_manifest_unmodified" in source
    assert "layer32_independently_confirmed" in source
    publish = next(cell for cell in _code_cells() if "layer32_confirmatory_manifest.json" in _source(cell))
    assert "VERDICT[\"verdict\"] == VERDICT_VALIDATED" in _source(publish), (
        "the confirmatory manifest must only be published on a pass"
    )


def test_a_failure_preserves_results_and_recommends_layer_38_only():
    source = _code_source()
    assert "Results are preserved in" in source
    assert "keep only layer 38" in source.lower()


def test_the_multimodal_selected_layer_is_not_changed():
    source = _code_source()
    assert "selected layer is unchanged" in source
    assert "mmpilot" not in source, "this notebook must not touch the multimodal pipeline"


@pytest.mark.parametrize(
    "phrase",
    [
        "never fits a lens",
        "read-only",
        "32 genuinely new held-out text prompts",
        "cannot veto a primary-metric pass",
    ],
)
def test_the_markdown_states_what_the_run_does_and_does_not_do(phrase):
    assert phrase in _full_source()
