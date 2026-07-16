# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Guards for the layer-21 diagnostic experiment setup: the config must
replicate the pilot recipe exactly (except source_layers=[21]), and the
notebook must keep its scope and stop conditions."""

import json
from pathlib import Path

from jlens.metadata import config_fingerprint, load_config

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "gemma_layer21_diagnostic.yaml"
PILOT_CONFIG_PATH = REPO_ROOT / "configs" / "gemma_text_pilot.yaml"
NOTEBOOK_PATH = (
    REPO_ROOT / "notebooks" / "gemma_4_e4b_layer21_diagnostic.ipynb"
)
JSPACE_CONFIG_PATH = REPO_ROOT / "configs" / "gemma_jspace_pursuit.yaml"

PILOT_REVISION = "fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd"
PILOT_LENS_SHA = (
    "sha256:7229c7562d1d55420b70abb13f481934649c4b01417bd851e97cedb47c96f474"
)


def test_config_validates_and_is_scoped_to_layer21():
    config = load_config(str(CONFIG_PATH))
    assert config["mode"] == "layer21_diagnostic"
    assert config["model"]["allow_model_load"] is False  # gated by default
    assert config["model"]["revision"] == PILOT_REVISION
    assert config["sites"]["source_layers"] == [21]
    assert config["sites"]["target_layer"] == 41
    assert config["pursuit"]["k_values"] == [10]
    assert config_fingerprint(config).startswith("sha256:")


def test_config_replicates_pilot_fitting_recipe():
    config = load_config(str(CONFIG_PATH))
    pilot = load_config(str(PILOT_CONFIG_PATH))
    for key in (
        "prompt_source",
        "n_prompts",
        "max_seq_len",
        "dim_batch",
        "seed",
        "checkpoint_every",
    ):
        assert config["fitting"][key] == pilot["fitting"][key], key
    assert config["positions"] == pilot["positions"]
    assert config["sites"]["target_layer"] == pilot["sites"]["target_layer"]
    assert 21 in pilot["sites"]["source_layers"]
    assert config["model"]["dtype"] == pilot["model"]["dtype"] == "bfloat16"


def test_config_reference_matches_jspace_expectations():
    """The pilot artifact identity (run dir, lens sha, revision) must agree
    with what the completed jspace run was already pinned to."""
    config = load_config(str(CONFIG_PATH))
    ref = config["reference"]
    jspace = json.loads(
        json.dumps(  # yaml via load_jspace_config would validate more keys
            __import__("yaml").safe_load(
                JSPACE_CONFIG_PATH.read_text(encoding="utf-8")
            )
        )
    )
    assert ref["pilot_run_dir_name"] == jspace["lens"]["run_dir_name"]
    assert ref["expect_lens_sha256"] == jspace["lens"]["expect_file_sha256"]
    assert ref["expect_lens_sha256"] == PILOT_LENS_SHA
    assert ref["expect_model_revision"] == jspace["lens"]["expect_model_revision"]


def load_notebook():
    nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    return nb


def code_of(nb) -> str:
    return "\n".join(
        "".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "code"
    )


def test_notebook_has_no_stored_outputs():
    nb = load_notebook()
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []
            assert cell["execution_count"] is None


def test_notebook_scope_and_stop_conditions():
    code = code_of(load_notebook())
    # Branch pin for the Colab bootstrap.
    assert 'BRANCH = "layer21-diagnostic"' in code
    # Stop conditions demanded by the experiment design.
    assert 'source_layers"] != [21]' in code.replace("'", '"')
    assert 'k_values"] != [10]' in code.replace("'", '"')
    assert "run_metadata.json" in code and "COMPLETE" in code
    assert "params_frozen" in code
    assert "resolved != PILOT_REVISION" in code
    assert "hashes != PILOT_PROMPT_HASHES" in code
    assert "validate_finite" in code
    # Lower-memory helpers actually used.
    assert "build_chunk_rows" in code
    # Fresh run directories must never overwrite.
    assert "exist_ok=False" in code
    # Only layer 21 is evaluated / decomposed.
    assert "layers=[21]" in code
    for forbidden in ("layers=[14", "layers=[28", "layers=[35", "layers=[38"):
        assert forbidden not in code


def test_notebook_records_required_diagnostics():
    code = code_of(load_notebook())
    assert "PromptDiagnosticsRecorder" in code
    assert "expected_prompt_hashes=PILOT_PROMPT_HASHES" in code
    assert "per_prompt_diagnostics" in code
    assert "running_accumulation.json" in code
    assert "refit_comparison.json" in code
    assert "execution_record(" in code
