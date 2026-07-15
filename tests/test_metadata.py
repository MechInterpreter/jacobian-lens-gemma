# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Config-validation and metadata tests. The shipped configs must validate."""

import copy
import json
from pathlib import Path

import pytest

from jlens.metadata import (
    UPSTREAM_COMMIT,
    config_fingerprint,
    environment_manifest,
    load_config,
    prompt_hashes,
    validate_config,
    write_metadata,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS = sorted((REPO_ROOT / "configs").glob("gemma_text_*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=[p.stem for p in CONFIGS])
def test_shipped_configs_validate(path):
    config = load_config(str(path))
    assert config["mode"] in ("microsmoke", "smoke", "pilot")
    assert config["model"]["allow_model_load"] is False  # gated by default
    assert config["model"]["expect_d_model"] == 2560
    assert config["model"]["expect_n_layers"] == 42
    # Layer indices structurally sane for a 42-block model.
    assert all(0 <= l < 42 for l in config["sites"]["source_layers"])
    assert max(config["sites"]["source_layers"]) < config["sites"]["target_layer"]


def test_shipped_configs_found():
    assert len(CONFIGS) == 3


def test_shipped_prompt_files_are_consistent():
    fit_prompts = json.loads(
        (REPO_ROOT / "configs/prompts/fit_prompts.json").read_text(encoding="utf-8")
    )
    assert fit_prompts["format"] == "plain_text"
    assert len(fit_prompts["prompts"]) == 8
    assert all(isinstance(p, str) and len(p) > 100 for p in fit_prompts["prompts"])

    eval_prompts = json.loads(
        (REPO_ROOT / "configs/prompts/eval_prompts.json").read_text(encoding="utf-8")
    )
    assert eval_prompts["plain"] and eval_prompts["chat"]
    # Chat prompts are evaluation-only; the fitting corpus has no chat field.
    assert "chat" not in fit_prompts


def _valid_config():
    return load_config(str(CONFIGS[0]))


def test_validate_missing_key():
    config = _valid_config()
    del config["fitting"]["seed"]
    with pytest.raises(ValueError, match="missing key: fitting.seed"):
        validate_config(config)


def test_validate_bad_mode():
    config = _valid_config()
    config["mode"] = "production"
    with pytest.raises(ValueError, match="mode"):
        validate_config(config)


def test_validate_bad_type():
    config = _valid_config()
    config["fitting"]["dim_batch"] = "8"
    with pytest.raises(ValueError, match="fitting.dim_batch"):
        validate_config(config)


def test_validate_seq_len_vs_skip_first():
    config = _valid_config()
    config["fitting"]["max_seq_len"] = config["positions"]["skip_first"]
    with pytest.raises(ValueError, match="no valid positions"):
        validate_config(config)


def test_validate_empty_source_layers():
    config = _valid_config()
    config["sites"]["source_layers"] = []
    with pytest.raises(ValueError, match="source_layers"):
        validate_config(config)


def test_fingerprint_stable_and_sensitive():
    config = _valid_config()
    reordered = copy.deepcopy(config)
    reordered["model"] = dict(reversed(list(reordered["model"].items())))
    assert config_fingerprint(config) == config_fingerprint(reordered)
    changed = copy.deepcopy(config)
    changed["fitting"]["seed"] += 1
    assert config_fingerprint(changed) != config_fingerprint(config)


def test_prompt_hashes_deterministic():
    hashes = prompt_hashes(["alpha", "beta"])
    assert hashes == prompt_hashes(["alpha", "beta"])
    assert len(set(hashes)) == 2 and all(len(h) == 16 for h in hashes)


def test_write_metadata_roundtrip(tmp_path):
    path = tmp_path / "nested" / "meta.json"
    write_metadata(str(path), {"answer": 42})
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["answer"] == 42
    assert "written_utc" in loaded


def test_environment_manifest_pins_upstream():
    manifest = environment_manifest()
    assert manifest["upstream_commit"] == UPSTREAM_COMMIT
    assert len(UPSTREAM_COMMIT) == 40
    assert manifest["torch"] and manifest["transformers"]
    # local_commit is this checkout's HEAD (None only outside git).
    assert manifest["local_commit"] is None or len(manifest["local_commit"]) == 40
