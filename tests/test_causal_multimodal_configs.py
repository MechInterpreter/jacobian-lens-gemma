# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Config validation for the causal smoke run and multimodal capture, plus
the committed example manifest's consistency with committed data."""

import copy
import json
from pathlib import Path

import pytest

from jlens.metadata import (
    config_fingerprint,
    load_causal_config,
    load_multimodal_config,
    validate_causal_config,
    validate_multimodal_config,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CAUSAL_YAML = str(REPO_ROOT / "configs" / "gemma_jspace_causal_smoke.yaml")
MULTIMODAL_YAML = str(REPO_ROOT / "configs" / "gemma_multimodal_jlens_capture.yaml")
MANIFEST = REPO_ROOT / "configs" / "causal_smoke_examples.json"


def test_committed_causal_config_loads():
    config = load_causal_config(CAUSAL_YAML)
    assert config["mode"] == "causal_smoke"
    assert config["intervention"]["layers"] == [35, 38]
    assert 0.0 in [float(m) for m in config["intervention"]["multipliers"]]
    assert config["model"]["allow_model_load"] is False
    assert config_fingerprint(config) == config_fingerprint(config)


def test_causal_config_rejections():
    config = load_causal_config(CAUSAL_YAML)

    broken = copy.deepcopy(config)
    broken["intervention"]["multipliers"] = [-1.0, 1.0]
    with pytest.raises(ValueError, match="must include 0.0"):
        validate_causal_config(broken)

    broken = copy.deepcopy(config)
    broken["intervention"]["layers"] = [35, 40]
    with pytest.raises(ValueError, match="not in"):
        validate_causal_config(broken)

    broken = copy.deepcopy(config)
    broken["mode"] = "causal"
    with pytest.raises(ValueError, match="must be 'causal_smoke'"):
        validate_causal_config(broken)

    broken = copy.deepcopy(config)
    del broken["parity"]
    with pytest.raises(ValueError, match="parity.max_abs_logit_diff_tol"):
        validate_causal_config(broken)


def test_committed_multimodal_config_loads():
    config = load_multimodal_config(MULTIMODAL_YAML)
    assert config["mode"] == "multimodal_capture"
    assert config["capture"]["layers"] == [38]
    assert config["capture"]["k"] == 10
    assert config["capture"]["position"] == -1
    assert config["inputs"]["image_path"] is None  # no committed assets
    assert config["inputs"]["audio_path"] is None
    # Decomposition settings must match the completed jspace run so text and
    # multimodal cones stay comparable.
    assert config["decomposition"] == {
        "normalize_atoms": True,
        "refine_steps": 2,
        "tol_relative_residual": 0.0,
        "fold_final_norm_weight": False,
        "correlation_chunk_size": 65536,
    }


def test_multimodal_config_rejections():
    config = load_multimodal_config(MULTIMODAL_YAML)
    broken = copy.deepcopy(config)
    broken["capture"]["layers"] = [40]
    with pytest.raises(ValueError, match="not in"):
        validate_multimodal_config(broken)
    broken = copy.deepcopy(config)
    broken["mode"] = "capture"
    with pytest.raises(ValueError, match="must be 'multimodal_capture'"):
        validate_multimodal_config(broken)


def test_manifest_examples_are_consistent():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    examples = manifest["examples"]
    assert len(examples) == 4
    slugs = [e["slug"] for e in examples]
    assert len(set(slugs)) == 4
    for example in examples:
        assert example["positions"] == [-1]
        assert len(example["prompt_hash"]) == 16
        assert example["selection_reason"].strip()
    formats = {e["format"] for e in examples}
    assert "chat" in formats and "plain" in formats

    # Manifest examples must exist in the committed demo bundle with matching
    # hashes, so measured causal records attach to explorer examples.
    demo = json.loads(
        (REPO_ROOT / "explorer" / "public" / "data" / "text_demo.json")
        .read_text(encoding="utf-8"))
    by_slug = {e["prompt_slug"]: e for e in demo["examples"]}
    for example in examples:
        assert example["slug"] in by_slug, example["slug"]
        assert by_slug[example["slug"]]["prompt_hash"] == example["prompt_hash"]
