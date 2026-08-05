# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""The native spoken-audio path, exercised end to end without a checkpoint.

The pilot blocked ``spoken_audio`` on an observation — *"produced audio
features but zero audio placeholder tokens"* — that is reproducible and whose
cause is a calling convention, not a capability limit. These tests pin the
repair and, just as importantly, pin the failure: :class:`FailureModes` walks
every state in which an audio input is wrong, and each one must be a named
refusal rather than a silent input the model would reject much later.

Everything here runs on CPU in seconds against
:class:`~jlens.mmpilot.mock.MockAudioProcessor`, which reproduces
``Gemma4Processor``'s audio contract exactly — including the silent
zero-placeholder failure of a bare processor call. Waveforms are generated, so
no SpokenCOCO media is needed or committed.
"""

import numpy as np
import pytest
import torch

from jlens.mmpilot import audio as A
from jlens.mmpilot.backend import (
    GemmaPilotBackend,
    ModalityUnsupportedError,
    check_capture_noop,
    check_zero_intervention,
    resolve_processor_interface,
)
from jlens.mmpilot.capability import (
    build_prompt,
    build_question,
    score_candidate_sequences,
)
from jlens.mmpilot.mock import (
    MOCK_AUDIO_TOKEN_ID,
    MOCK_D_MODEL,
    MOCK_VOCAB,
    MockAudioGemmaLike,
    MockAudioProcessor,
    MockAudioTokenizer,
    MockWorld,
    build_mock_audio_backend,
    mock_audio_config,
)
from jlens.mmpilot.store import IncompatibleStateError, RunFingerprint, UnitStore

SR = 16_000
QUESTION = build_question(["cat", "dog"])
PROMPT = build_prompt(QUESTION, modality="spoken_audio")


# --------------------------------------------------------------- a real model
#
# The mock audio world lives in the package, not here: the feasibility
# notebook's MOCK branch runs against the same objects, so a defect in the
# stand-in cannot make these tests pass while the notebook fails.

A_TOKEN_ID = MOCK_AUDIO_TOKEN_ID
AudioMockModel = MockAudioGemmaLike
_Tokenizer = MockAudioTokenizer


def make_backend(**processor_kwargs):
    """A real :class:`GemmaPilotBackend` over the mock processor and model."""
    return build_mock_audio_backend(**processor_kwargs)


@pytest.fixture(scope="module")
def backend_bundle():
    return make_backend()


def wav(seconds=1.0, seed=0):
    return A.probe_waveform(seconds, SR, seed=seed)


# ---------------------------------------------------------------- 1. blocks


def test_content_block_is_the_waveform_and_nothing_else():
    """1. The chat-template content block carries the recording, not a path."""
    samples = wav()
    block = A.audio_content_block(samples)
    assert block["type"] == "audio"
    assert block["audio"] is samples
    # No path, url, filename or transcript may ride along.
    assert set(block) == {"type", "audio"}
    assert A.CONTENT_BLOCK_SCHEMA["type"] == "audio"


def test_chat_template_route_is_what_inserts_the_placeholder():
    """1. The bare processor call is *not* an equivalent convention."""
    processor = MockAudioProcessor()
    samples = wav()
    bare = processor(text=PROMPT, audio=samples)
    assert int((bare["input_ids"] == A_TOKEN_ID).sum()) == 0
    assert "input_features" in bare  # features, no placeholders — the whole bug

    templated = A.encode_audio_prompt(processor, PROMPT, samples)
    assert int((templated["input_ids"] == A_TOKEN_ID).sum()) > 0


# ------------------------------------------------------- 2. waveform contract


@pytest.mark.parametrize(
    "given, expect_channels",
    [
        (np.zeros(SR, dtype=np.float64), 1),
        (np.zeros((SR, 2), dtype=np.float32), 2),
        (torch.zeros(SR), 1),
    ],
)
def test_prepare_waveform_normalizes_dtype_channels_and_shape(given, expect_channels):
    """2. float32, mono, 1-D — whatever the loader handed us."""
    prepared = A.prepare_waveform(given, SR, expected_rate=SR)
    assert prepared.samples.dtype == np.float32
    assert prepared.samples.ndim == 1
    assert prepared.sampling_rate == SR
    assert prepared.n_channels_in == expect_channels
    assert prepared.to_dict()["dtype_out"] == "float32"


def test_sample_rate_mismatch_is_refused_not_resampled():
    """2. Silently reinterpreting 22 kHz as 16 kHz changes the evidence."""
    with pytest.raises(A.SpokenAudioUnsupportedError) as excinfo:
        A.prepare_waveform(np.zeros(22050, dtype=np.float32), 22_050, expected_rate=SR)
    assert excinfo.value.reason == "sampling_rate_mismatch"
    assert "22050" in str(excinfo.value)


@pytest.mark.parametrize(
    "given, sr",
    [
        (np.zeros(0, dtype=np.float32), SR),
        (np.array([np.nan, 0.0], dtype=np.float32), SR),
        (np.zeros((2, 2, 2), dtype=np.float32), SR),
    ],
)
def test_invalid_waveforms_are_refused(given, sr):
    """2. Empty, non-finite and non-audio-shaped input never reach the model."""
    with pytest.raises(A.SpokenAudioUnsupportedError) as excinfo:
        A.prepare_waveform(given, sr, expected_rate=SR)
    assert excinfo.value.reason == "invalid_waveform"


# -------------------------------------------------- 3. placeholders and spans


def test_placeholder_span_is_detected_and_tracks_duration(backend_bundle):
    """3. The span is contiguous, located, and grows with the recording."""
    backend, _, resolved = backend_bundle
    short = backend.build_inputs(prompt=PROMPT, modality="spoken_audio", audio=wav(0.5), sampling_rate=SR)
    long = backend.build_inputs(prompt=PROMPT, modality="spoken_audio", audio=wav(1.0), sampling_rate=SR)

    for built in (short, long):
        start, end = built.modality_token_range
        ids = built.tensors["input_ids"][0]
        assert end > start
        assert torch.all(ids[start:end] == resolved.audio_token_id)
        assert built.audio["n_placeholders"] == end - start

    assert long.audio["n_placeholders"] > short.audio["n_placeholders"]
    assert resolved.dynamic_placeholder_count is True


def test_placeholder_count_equals_what_the_features_imply(backend_bundle):
    """3. The invariant the model enforces, checked before the model sees it."""
    backend, _, _ = backend_bundle
    built = backend.build_inputs(
        prompt=PROMPT, modality="spoken_audio", audio=wav(), sampling_rate=SR
    )
    assert built.audio["n_placeholders"] == built.audio["n_expected_from_features"]
    assert built.audio["n_expected_from_features"] == A.expected_placeholder_count(
        built.tensors["input_features_mask"][0]
    )


# ------------------------------------------------------------- 4-6. refusals


def test_features_without_placeholders_is_refused():
    """4. The exact state that blocked the pilot, now a named refusal."""
    processor = MockAudioProcessor(renders_audio_token=False)
    encoded = A.encode_audio_prompt(processor, PROMPT, wav())
    assert "input_features" in encoded
    with pytest.raises(A.SpokenAudioUnsupportedError) as excinfo:
        A.verify_audio_encoding(encoded, audio_token_id=A_TOKEN_ID)
    assert excinfo.value.reason == "features_without_placeholders"
    assert "zero" in str(excinfo.value) or "no audio placeholder" in str(excinfo.value)


def test_placeholders_without_features_is_refused():
    """5. Placeholders alone would embed literal special tokens as evidence."""
    processor = MockAudioProcessor(emit_features=False)
    encoded = A.encode_audio_prompt(processor, PROMPT, wav())
    assert "input_features" not in encoded
    with pytest.raises(A.SpokenAudioUnsupportedError) as excinfo:
        A.verify_audio_encoding(encoded, audio_token_id=A_TOKEN_ID)
    assert excinfo.value.reason == "placeholders_without_features"


def test_mismatched_placeholder_and_feature_counts_are_refused():
    """6. One placeholder too many is a refusal, not a masked_scatter crash."""
    processor = MockAudioProcessor(placeholder_delta=1)
    encoded = A.encode_audio_prompt(processor, PROMPT, wav())
    with pytest.raises(A.SpokenAudioUnsupportedError) as excinfo:
        A.verify_audio_encoding(encoded, audio_token_id=A_TOKEN_ID)
    assert excinfo.value.reason == "placeholder_feature_count_mismatch"


def test_non_contiguous_placeholder_span_is_refused():
    """6. A broken-up span has no single audio region to record or edit at."""
    processor = MockAudioProcessor(contiguous=False)
    encoded = A.encode_audio_prompt(processor, PROMPT, wav())
    with pytest.raises(A.SpokenAudioUnsupportedError) as excinfo:
        A.verify_audio_encoding(encoded, audio_token_id=A_TOKEN_ID)
    assert excinfo.value.reason == "non_contiguous_placeholder_span"


def test_resolver_refuses_a_checkpoint_without_an_audio_tower():
    """4-6. Component presence is the claim; the tower is the capability."""
    with pytest.raises(A.SpokenAudioUnsupportedError) as excinfo:
        A.resolve_audio_interface(
            MockAudioProcessor(),
            mock_audio_config(audio_tower=False),
            model_repo_id="mock/no-audio",
            model_revision="rev",
            processor_revision="rev",
        )
    assert excinfo.value.reason == "no_audio_tower"


def test_resolver_refuses_a_processor_whose_probe_produces_no_placeholders():
    """4. A probe, not an inspection: the resolver runs the path it certifies."""
    with pytest.raises(A.SpokenAudioUnsupportedError) as excinfo:
        A.resolve_audio_interface(
            MockAudioProcessor(renders_audio_token=False),
            mock_audio_config(),
            model_repo_id="mock/gemma-like",
            model_revision="rev",
            processor_revision="rev",
        )
    assert excinfo.value.reason == "features_without_placeholders"
    assert set(A.REASONS) >= {excinfo.value.reason}


def test_backend_blocks_spoken_audio_without_a_resolved_interface():
    """4. No probed protocol means the channel is blocked, never guessed."""
    processor = MockAudioProcessor()
    model = AudioMockModel()
    interface = resolve_processor_interface(processor, model.config)
    assert interface["supports_audio"] is True  # components exist ...
    backend = GemmaPilotBackend(model, processor, interface, device="cpu")
    assert backend.supports("spoken_audio") is False  # ... which is not support
    with pytest.raises(ModalityUnsupportedError):
        backend.build_inputs(prompt=PROMPT, modality="spoken_audio", audio=wav())


# ------------------------------------------------- 7. the final prompt token


def test_final_prompt_position_is_after_the_expanded_audio_span(backend_bundle):
    """7. Everything is measured and edited at the last *prompt* token."""
    backend, _, _ = backend_bundle
    built = backend.build_inputs(
        prompt=PROMPT, modality="spoken_audio", audio=wav(), sampling_rate=SR
    )
    prompt_len = int(built.tensors["input_ids"].shape[1])
    assert built.prompt_len == prompt_len
    assert built.final_prompt_position == prompt_len - 1
    # It is past the audio span, not inside it.
    assert built.final_prompt_position >= built.modality_token_range[1]
    assert built.audio["final_prompt_position"] == built.final_prompt_position


# ----------------------------------------------- 8. candidate-sequence scoring


def test_complete_multi_token_candidate_sequences_are_scored(backend_bundle):
    """8. The whole answer, not its first token."""
    backend, _, _ = backend_bundle
    built = backend.build_inputs(
        prompt=PROMPT, modality="spoken_audio", audio=wav(), sampling_rate=SR
    )
    candidates = {"cat": [30, 31, 32], "dog": [33, 34]}
    scores = score_candidate_sequences(backend, built, candidates)
    assert scores["cat"]["n_tokens"] == 3
    assert scores["dog"]["n_tokens"] == 2
    for name, row in scores.items():
        assert row["sum_logprob"] < 0.0
        assert row["sum_logprob"] == pytest.approx(
            row["mean_logprob"] * row["n_tokens"], rel=1e-6
        )
        assert row["token_ids"] == candidates[name]


# ------------------------------------------------------ 9. audio-tower firing


def test_audio_tower_is_actually_invoked(backend_bundle):
    """9. A placeholder span proves the text; only a fired hook proves the audio."""
    backend, _, _ = backend_bundle
    built = backend.build_inputs(
        prompt=PROMPT, modality="spoken_audio", audio=wav(), sampling_rate=SR
    )
    report = A.check_audio_tower_invoked(backend, built)
    assert report["invoked"] is True
    assert report["tower_class"] == "MockAudioTower"
    assert report["output_shape"][1] == built.audio["n_placeholders"]


def test_audio_tower_is_not_invoked_for_text(backend_bundle):
    """9. The check would be worthless if it fired on a text-only pass."""
    backend, _, _ = backend_bundle
    text = backend.build_inputs(prompt=PROMPT, modality="text")
    with pytest.raises(A.SpokenAudioUnsupportedError) as excinfo:
        A.check_audio_tower_invoked(backend, text)
    assert excinfo.value.reason == "no_audio_pathway_engaged"


# --------------------------------------------- 10-11. the recording matters


def test_waveform_and_silence_produce_different_logits(backend_bundle):
    """10. Otherwise the audio channel is decorative."""
    backend, _, _ = backend_bundle
    speech = backend.build_inputs(
        prompt=PROMPT, modality="spoken_audio", audio=wav(1.0, seed=1), sampling_rate=SR
    )
    quiet = backend.build_inputs(
        prompt=PROMPT,
        modality="spoken_audio",
        audio=A.silence_waveform(1.0, SR),
        sampling_rate=SR,
    )
    assert speech.audio["n_placeholders"] == quiet.audio["n_placeholders"]
    difference = (
        backend.forward_logits(speech.tensors) - backend.forward_logits(quiet.tensors)
    ).abs().max()
    assert float(difference) > 1e-4


def test_two_different_waveforms_produce_different_logits(backend_bundle):
    """11. And not merely "anything differs from silence"."""
    backend, _, _ = backend_bundle
    first = backend.build_inputs(
        prompt=PROMPT, modality="spoken_audio", audio=wav(1.0, seed=2), sampling_rate=SR
    )
    second = backend.build_inputs(
        prompt=PROMPT, modality="spoken_audio", audio=wav(1.0, seed=3), sampling_rate=SR
    )
    difference = (
        backend.forward_logits(first.tensors) - backend.forward_logits(second.tensors)
    ).abs().max()
    assert float(difference) > 1e-4


# ------------------------------------------------------- 12-13. invariance


def test_capture_hook_preserves_logits_on_an_audio_input(backend_bundle):
    """12. A capture hook that perturbs the pass makes every number meaningless."""
    backend, _, _ = backend_bundle
    built = backend.build_inputs(
        prompt=PROMPT, modality="spoken_audio", audio=wav(), sampling_rate=SR
    )
    report = check_capture_noop(backend, built, [1, 2])
    assert report["passed"] is True
    assert report["max_abs_logit_diff"] == 0.0


def test_zero_coefficient_intervention_preserves_the_clean_result(backend_bundle):
    """13. At the final prompt position, which audio expansion moved."""
    backend, _, _ = backend_bundle
    built = backend.build_inputs(
        prompt=PROMPT, modality="spoken_audio", audio=wav(), sampling_rate=SR
    )
    report = check_zero_intervention(backend, built, 2)
    assert report["passed"] is True
    assert report["resolved_position"] in (
        built.final_prompt_position,
        built.final_prompt_position - built.prompt_len,
    )


# ----------------------------------------------------------- 14. no leakage


def test_spoken_audio_prompt_carries_no_transcript_or_filename():
    """14. The recording is the evidence; the caption is not in the prompt."""
    caption = "a cat sitting on a bench"
    prompt = build_prompt(QUESTION, modality="spoken_audio", caption=caption)
    assert caption not in prompt
    assert A.assert_no_text_leakage(
        prompt, forbidden=[caption, "COCO_train2014_000000419532", "cat_0.wav"]
    )["leaked"] == []


def test_leaked_caption_is_refused():
    """14. And the check really fires when something does leak."""
    caption = "a cat sitting on a bench"
    with pytest.raises(ValueError, match="leaks non-audio evidence"):
        A.assert_no_text_leakage(f"Caption: {caption}\n{QUESTION}", forbidden=[caption])


def test_pipeline_audio_branch_refuses_a_leaking_prompt(monkeypatch):
    """14. Enforced where the group's own caption and paths are known."""
    from jlens.mmpilot import capability as capability_module
    from jlens.mmpilot import pipeline as pipeline_module

    backend, _, _ = make_backend()
    group = {
        "caption": "a cat on a bench",
        "audio_path": "wavs/train/cat_0.wav",
        "image_path": "train2014/cat.jpg",
        "image_id": "419532",
    }
    monkeypatch.setattr(
        capability_module,
        "build_prompt",
        lambda question, *, modality, caption=None: f"Caption: {caption}\n{question}",
    )
    monkeypatch.setattr(pipeline_module, "build_prompt", capability_module.build_prompt)
    with pytest.raises(ValueError, match="leaks non-audio evidence"):
        pipeline_module._build_inputs_for(
            backend,
            group,
            "spoken_audio",
            QUESTION,
            {"load_audio": lambda path: (wav(), SR), "load_image": lambda path: None},
        )


# -------------------------------------------------------- 15-16. fingerprint


def test_artifacts_fingerprint_the_audio_protocol(backend_bundle):
    """15. Every audio input records the protocol it was built under."""
    backend, _, resolved = backend_bundle
    built = backend.build_inputs(
        prompt=PROMPT, modality="spoken_audio", audio=wav(), sampling_rate=SR
    )
    assert built.route["route"] == A.CALL_CONVENTION
    assert built.route["protocol_version"] == A.AUDIO_PROTOCOL_VERSION
    assert built.route["protocol_fingerprint"] == resolved.protocol_fingerprint
    assert resolved.protocol_fingerprint.startswith("sha256:")


@pytest.mark.parametrize(
    "field, value",
    [
        ("model_revision", "some-other-revision"),
        ("transformers_version", "0.0.0"),
        ("sampling_rate", 22_050),
        ("call_convention", "bare_processor_call"),
        ("audio_token_id", 999),
    ],
)
def test_protocol_fingerprint_changes_with_every_bound_field(field, value):
    """15. Revision, library version, protocol and placeholder convention."""
    import dataclasses

    _, _, resolved = make_backend()
    changed = dataclasses.replace(resolved, **{field: value})
    assert changed.protocol_fingerprint != resolved.protocol_fingerprint


def test_probe_numbers_do_not_enter_the_fingerprint():
    """15. Two machines resolving the same protocol must agree on the digest."""
    import dataclasses

    _, _, resolved = make_backend()
    noisy = dataclasses.replace(
        resolved, probes=(), notes={"different": True}, dynamic_placeholder_count=False
    )
    assert noisy.protocol_fingerprint == resolved.protocol_fingerprint


def test_a_run_without_the_audio_protocol_refuses_to_resume_with_it(tmp_path):
    """16. An existing text-image run is not silently reused for audio."""
    _, _, resolved = make_backend()

    def fingerprint(extra):
        return RunFingerprint(
            mode="mock",
            model_repo_id="mock/gemma-like",
            model_revision="mock-rev",
            processor_revision="mock-rev",
            layers=(2, 4),
            lens_checksum="sha256:mock-identity-lens",
            manifest_checksum="sha256:manifest",
            split_id="seed",
            extra=extra,
        )

    UnitStore(tmp_path, fingerprint({})).open()
    with pytest.raises(IncompatibleStateError) as excinfo:
        UnitStore(
            tmp_path,
            fingerprint({"audio_protocol_fingerprint": resolved.protocol_fingerprint}),
        ).open()
    assert "extra" in str(excinfo.value)


# ------------------------------------------------------- 17-18. no collateral


def test_mock_audio_resolution_is_deterministic():
    """17. Same processor, same protocol digest, run after run."""
    first = A.resolve_audio_interface(
        MockAudioProcessor(),
        mock_audio_config(),
        model_repo_id="mock/gemma-like",
        model_revision="mock-rev",
        processor_revision="mock-rev",
    )
    second = A.resolve_audio_interface(
        MockAudioProcessor(),
        mock_audio_config(),
        model_repo_id="mock/gemma-like",
        model_revision="mock-rev",
        processor_revision="mock-rev",
    )
    assert first.to_dict() == second.to_dict()
    assert first.protocol_fingerprint == second.protocol_fingerprint
    assert first.notes["placeholder_counts"] == [13, 25]


def test_text_and_image_inputs_are_unchanged_by_the_audio_repair():
    """18. The completed text-image study must stay byte-for-byte re-derivable."""
    world = MockWorld()

    class ImageProcessor:
        chat_template = None

        def __init__(self):
            from jlens.mmpilot.mock import MockPilotBackend

            self._mock = MockPilotBackend(world)
            self.tokenizer = _Tokenizer()
            self.tokenizer.chat_template = None
            self.image_processor = object()

        def __call__(self, text=None, images=None, return_tensors=None):
            built = self._mock.build_inputs(
                prompt=text,
                modality="image" if images is not None else "text",
                image=images,
            )
            return dict(built.tensors)

    processor = ImageProcessor()
    from jlens.mmpilot.mock import MockGemmaLike

    model = MockGemmaLike(world)
    interface = resolve_processor_interface(processor, model.config)
    plain = GemmaPilotBackend(model, processor, interface, device="cpu")
    with_audio = GemmaPilotBackend(
        model, processor, interface, device="cpu", audio_interface=object()
    )

    for backend in (plain, with_audio):
        assert backend.supports("text") is True
        assert backend.supports("image") is True

    text_a = plain.build_inputs(prompt="Caption: a dog\n" + QUESTION, modality="text")
    text_b = with_audio.build_inputs(prompt="Caption: a dog\n" + QUESTION, modality="text")
    assert torch.equal(text_a.tensors["input_ids"], text_b.tensors["input_ids"])
    assert text_a.route == text_b.route
    assert text_a.route == {"route": "processor_call", "kwargs": ["return_tensors"]}
    assert text_a.modality_token_range is None
    assert text_a.audio is None

    image = torch.zeros(MOCK_D_MODEL)
    image_a = plain.build_inputs(prompt=QUESTION, modality="image", image=image)
    assert image_a.route["route"] == "processor_call"
    assert image_a.route["kwargs"] == ["images", "return_tensors"]
    assert image_a.audio is None


# --------------------------------------------------- the real-path wiring


def test_build_real_backend_resolves_audio_only_when_asked(monkeypatch):
    """``resolve_audio`` is opt-in, and off leaves spoken_audio blocked.

    Runs the actual :func:`~jlens.mmpilot.real_backend.build_real_backend` body
    with only its two network hooks replaced, so the notebook's real branch and
    the installed code cannot drift on this argument.
    """
    from jlens.mmpilot import real_backend as R

    model = AudioMockModel()

    class LensModel:
        _hf_model = model

    class Report:
        def to_dict(self):
            return {"n_layers": 4, "d_model": MOCK_D_MODEL, "vocab_size": MOCK_VOCAB}

    def load(repo_id, *, revision, dtype, device_map, allow_model_load, token):
        return LensModel(), {"model_repo_id": repo_id, "model_revision": revision}

    def verify(model_, *, expect_n_layers, expect_d_model, expect_vocab_size):
        return Report()

    def factory(repo_id, *, revision, token=None):
        processor = MockAudioProcessor()
        processor.tokenizer = _Tokenizer()
        return processor

    monkeypatch.setattr(R, "_loader", lambda: (load, verify))
    monkeypatch.setattr(R, "_processor_factory", lambda: factory)

    common = dict(
        revision="fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd",
        device="cpu",
        allow_model_load=True,
        expect_n_layers=4,
        expect_d_model=MOCK_D_MODEL,
        expect_vocab_size=MOCK_VOCAB,
    )
    off = R.build_real_backend("google/gemma-4-E4B-it", **common)
    assert off.audio_interface is None
    assert off.backend.supports("spoken_audio") is False

    on = R.build_real_backend("google/gemma-4-E4B-it", resolve_audio=True, **common)
    assert on.audio_interface is not None
    assert on.audio_blocked_reason == ""
    assert on.backend.supports("spoken_audio") is True
    assert on.audio_interface.model_revision == common["revision"]


def test_build_real_backend_records_a_blocked_reason_instead_of_crashing(monkeypatch):
    """A blocked audio channel must not take the text-image study down with it."""
    from jlens.mmpilot import real_backend as R

    model = AudioMockModel()

    class LensModel:
        _hf_model = model

    class Report:
        def to_dict(self):
            return {}

    monkeypatch.setattr(
        R,
        "_loader",
        lambda: (
            lambda repo_id, **kw: (
                LensModel(),
                {"model_repo_id": repo_id, "model_revision": kw["revision"]},
            ),
            lambda *a, **k: Report(),
        ),
    )
    monkeypatch.setattr(
        R,
        "_processor_factory",
        lambda: lambda repo_id, *, revision, token=None: MockAudioProcessor(
            renders_audio_token=False
        ),
    )
    bundle = R.build_real_backend(
        "google/gemma-4-E4B-it",
        revision="rev",
        device="cpu",
        allow_model_load=True,
        resolve_audio=True,
    )
    assert bundle.audio_interface is None
    assert "features_without_placeholders" in bundle.audio_blocked_reason
    assert bundle.backend.supports("spoken_audio") is False
    assert bundle.backend.supports("text") is True
