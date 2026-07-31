# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Typed-config behaviour for the J-space language autoencoder."""

from pathlib import Path

import pytest

from jlens.autoencoder.config import AutoencoderConfig, load_autoencoder_config
from jlens.autoencoder.errors import AutoencoderError
from jlens.autoencoder.runner import SMOKE_OVERRIDES, _deep_merge

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "jspace_language_autoencoder.yaml"


def test_shipped_config_loads_and_validates():
    config = load_autoencoder_config(str(CONFIG_PATH))
    assert config.mode == "jspace_language_autoencoder"
    assert config.dataset.source_layer == 14
    assert config.pursuit.k == 10
    assert config.adapter.max_new_tokens == 8
    assert config.adapter.beam_width == 8
    assert 14 in config.lens.expect_source_layers


def test_shipped_config_matches_the_brief_pilot_size():
    config = load_autoencoder_config(str(CONFIG_PATH))
    assert 500 <= config.dataset.n_phrases <= 1000
    assert 3 <= config.dataset.occurrences_per_phrase <= 5
    assert (config.dataset.min_phrase_tokens, config.dataset.max_phrase_tokens) == (2, 6)


def test_policy_gradient_is_disabled_by_default():
    config = load_autoencoder_config(str(CONFIG_PATH))
    assert config.preference.policy_gradient.enabled is False


def test_unknown_key_is_rejected_rather_than_ignored():
    payload = load_autoencoder_config(str(CONFIG_PATH)).to_dict()
    payload["reconstructor"]["hiden_dim"] = 8
    with pytest.raises(AutoencoderError, match="unknown key"):
        AutoencoderConfig.from_dict(payload)


def test_unknown_top_level_key_is_rejected():
    payload = load_autoencoder_config(str(CONFIG_PATH)).to_dict()
    payload["reconstuctor"] = {}
    with pytest.raises(AutoencoderError, match="unknown top-level key"):
        AutoencoderConfig.from_dict(payload)


def test_wrong_mode_is_rejected():
    payload = load_autoencoder_config(str(CONFIG_PATH)).to_dict()
    payload["mode"] = "generative_validation"
    with pytest.raises(AutoencoderError, match="must be"):
        AutoencoderConfig.from_dict(payload)


def test_source_layer_must_be_a_fitted_lens_layer():
    payload = load_autoencoder_config(str(CONFIG_PATH)).to_dict()
    payload["dataset"]["source_layer"] = 41
    with pytest.raises(AutoencoderError, match="expect_source_layers"):
        AutoencoderConfig.from_dict(payload)


def test_phrase_length_bounds_are_enforced():
    payload = load_autoencoder_config(str(CONFIG_PATH)).to_dict()
    payload["dataset"]["max_phrase_tokens"] = 7
    with pytest.raises(AutoencoderError, match="2 <= min <= max <= 6"):
        AutoencoderConfig.from_dict(payload)


def test_adapter_must_be_able_to_emit_the_longest_phrase():
    payload = load_autoencoder_config(str(CONFIG_PATH)).to_dict()
    payload["adapter"]["max_new_tokens"] = 4
    with pytest.raises(AutoencoderError, match="emit a full"):
        AutoencoderConfig.from_dict(payload)


def test_type_errors_name_the_offending_key():
    payload = load_autoencoder_config(str(CONFIG_PATH)).to_dict()
    payload["adapter"]["beam_width"] = "eight"
    with pytest.raises(AutoencoderError, match="adapter.beam_width"):
        AutoencoderConfig.from_dict(payload)


def test_smoke_overrides_produce_a_valid_config():
    base = load_autoencoder_config(str(CONFIG_PATH)).to_dict()
    config = AutoencoderConfig.from_dict(_deep_merge(base, SMOKE_OVERRIDES))
    assert config.dataset.corpus == "mock"
    assert config.dataset.n_phrases == 32
    # The overrides must not disturb the fixed experimental parameters.
    assert config.dataset.source_layer == 14
    assert config.pursuit.k == 10


def test_fingerprint_is_stable_and_content_sensitive():
    config = load_autoencoder_config(str(CONFIG_PATH))
    assert config.fingerprint() == AutoencoderConfig.from_dict(config.to_dict()).fingerprint()
    payload = config.to_dict()
    payload["adapter"]["seed"] = config.adapter.seed + 1
    assert AutoencoderConfig.from_dict(payload).fingerprint() != config.fingerprint()
