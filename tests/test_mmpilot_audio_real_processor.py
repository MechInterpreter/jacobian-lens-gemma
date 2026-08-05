# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""The audio protocol against the **real pinned processor**, opt-in.

:mod:`tests.test_mmpilot_audio` proves the repair against a mock that
reproduces ``Gemma4Processor``'s contract. That is the right default: it is
fast, offline, and deterministic. But the claim being made — *this checkpoint
supports native spoken audio* — is a claim about the real processor, so it
needs to be checkable against the real processor rather than only asserted in
a report.

These tests download four small config files (no weights, ~32 MB of tokenizer)
and run the same resolution the L4 will run. They are **skipped by default**
because they need the network. Enable them with::

    JLENS_REAL_PROCESSOR_TEST=1 pytest tests/test_mmpilot_audio_real_processor.py

Nothing here loads model weights or proves anything about Gemma's behavior. It
establishes only that the input protocol is the one this repository implements.
"""

import os

import pytest

from jlens.mmpilot import audio as A

pytestmark = pytest.mark.skipif(
    os.environ.get("JLENS_REAL_PROCESSOR_TEST") != "1",
    reason="needs the network; set JLENS_REAL_PROCESSOR_TEST=1 to run",
)

REPO_ID = "google/gemma-4-E4B-it"
REVISION = "fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd"


@pytest.fixture(scope="module")
def real():
    transformers = pytest.importorskip("transformers")
    processor = transformers.AutoProcessor.from_pretrained(REPO_ID, revision=REVISION)
    config = transformers.AutoConfig.from_pretrained(REPO_ID, revision=REVISION)
    return processor, config


def test_bare_processor_call_reproduces_the_zero_placeholder_failure(real):
    """The blocker, reproduced: features, no placeholders, and no error."""
    processor, config = real
    samples = A.probe_waveform(1.0, 16_000)
    encoded = processor(text="Answer with exactly one word.", audio=samples, return_tensors="pt")
    assert "input_features" in encoded
    assert int((encoded["input_ids"] == config.audio_token_id).sum()) == 0
    with pytest.raises(A.SpokenAudioUnsupportedError) as excinfo:
        A.verify_audio_encoding(encoded, audio_token_id=config.audio_token_id)
    assert excinfo.value.reason == "features_without_placeholders"


def test_the_content_block_route_produces_a_verified_placeholder_span(real):
    """The repair, against the real processor."""
    processor, config = real
    samples = A.probe_waveform(1.0, 16_000)
    encoded = A.encode_audio_prompt(processor, "Answer with exactly one word.", samples)
    verified = A.verify_audio_encoding(encoded, audio_token_id=config.audio_token_id)
    assert verified["n_placeholders"] == verified["n_expected_from_features"] > 0
    start, end = verified["audio_token_span"]
    assert end - start == verified["n_placeholders"]


def test_the_pilot_concepts_really_are_single_tokens(real):
    """The root cause of the first real audit's FAIL, against the real tokenizer.

    Nothing was wrong with the scorer. The fixture simply could not exercise
    complete-sequence scoring, because these encode to one token each.
    """
    processor, _ = real
    tokenizer = processor.tokenizer
    assert tokenizer(" cat", add_special_tokens=False)["input_ids"] == [5866]
    assert tokenizer(" dog", add_special_tokens=False)["input_ids"] == [4799]


def test_selection_finds_multi_token_candidates_under_the_pinned_tokenizer(real):
    """The repaired fixture, measured rather than assumed."""
    from jlens.mmpilot.audio_audit import select_scoring_candidates

    processor, _ = real

    class _Backend:
        def encode_candidate(self, text):
            return processor.tokenizer(text, add_special_tokens=False)["input_ids"]

    chosen = select_scoring_candidates(_Backend())
    assert chosen == {"traffic light": [8827, 2214], "fire hydrant": [4304, 67175]}
    assert all(len(ids) > 1 for ids in chosen.values())


def test_resolution_reports_the_pinned_protocol(real):
    """The recorded interface names what a later run would have to match."""
    processor, config = real
    resolved = A.resolve_audio_interface(
        processor,
        config,
        model_repo_id=REPO_ID,
        model_revision=REVISION,
        processor_revision=REVISION,
    )
    assert resolved.processor_class == "Gemma4Processor"
    assert resolved.feature_extractor_class == "Gemma4AudioFeatureExtractor"
    assert resolved.audio_token == "<|audio|>"
    assert resolved.audio_token_id == 258881
    assert resolved.sampling_rate == 16_000
    assert resolved.audio_tower_present is True
    assert resolved.call_convention == A.CALL_CONVENTION
    # Placeholder count tracks duration: 0.5 s and 1.0 s must differ.
    assert resolved.dynamic_placeholder_count is True
    assert resolved.notes["placeholder_counts"] == [13, 25]
