# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""The feasibility audit's three verdicts, and what it refuses to write into.

The verdict logic lives in package code rather than in a notebook cell so it can
be tested against a deliberately broken backend. ``AUDIO_INVALID`` is the state
worth testing hardest: it is the one that looks like success.
"""

import json

import pytest
import torch

from jlens.mmpilot import audio as A
from jlens.mmpilot.audio_audit import (
    AUDIO_BLOCKED,
    AUDIO_INVALID,
    AUDIO_READY,
    REQUIRED_CHECKS,
    AudioAuditResult,
    Check,
    audit_markdown,
    new_audit_run_dir,
    run_audio_audit,
    verdict_from_checks,
    write_audit_report,
)
from tests.test_mmpilot_audio import PROMPT, QUESTION, SR, make_backend, wav

CANDIDATES = {"cat": [30, 31, 32], "dog": [33, 34]}
WAVEFORMS = [("speech_a", wav(1.0, seed=1)), ("speech_b", wav(1.0, seed=2))]
ENVIRONMENT = {"transformers": "5.10.2", "model_revision": "mock-rev"}


def audit(backend, **overrides):
    kwargs = dict(
        prompt=PROMPT,
        waveforms=WAVEFORMS,
        sampling_rate=SR,
        layers=[1, 2],
        candidate_ids=CANDIDATES,
        forbidden_text=["a cat on a bench", "cat_0"],
        mode="mock",
        environment=ENVIRONMENT,
    )
    kwargs.update(overrides)
    return run_audio_audit(backend, **kwargs)


# ------------------------------------------------------------- the verdicts


def test_a_correct_backend_reaches_audio_ready():
    backend, _, _ = make_backend()
    result = audit(backend)
    assert result.verdict == AUDIO_READY, result.failed
    names = {check.name for check in result.checks}
    assert names >= set(REQUIRED_CHECKS)
    assert result.audio_interface["call_convention"] == A.CALL_CONVENTION


def test_an_unresolved_protocol_is_blocked_not_invalid():
    """Nothing was measured, so nothing may be called invalid."""
    from jlens.mmpilot.backend import GemmaPilotBackend, resolve_processor_interface
    from jlens.mmpilot.mock import (
        MockAudioGemmaLike,
        MockAudioProcessor,
        MockAudioTokenizer,
    )

    processor = MockAudioProcessor()
    processor.tokenizer = MockAudioTokenizer()
    model = MockAudioGemmaLike()
    backend = GemmaPilotBackend(
        model, processor, resolve_processor_interface(processor, model.config), device="cpu"
    )
    result = audit(backend)
    assert result.verdict == AUDIO_BLOCKED
    assert result.failed == ["protocol_resolved"]
    assert "did not resolve" in result.blocked_reason


def test_a_leaking_prompt_is_invalid_not_ready():
    """The path works; the evidence isolation does not. That is the trap."""
    backend, _, _ = make_backend()
    caption = "a cat on a bench"
    result = audit(backend, prompt=f"Caption: {caption}\n{QUESTION}", forbidden_text=[caption])
    assert result.verdict == AUDIO_INVALID
    assert "no_transcript_leakage" in result.failed


def test_a_capture_hook_that_moves_logits_is_invalid(monkeypatch):
    backend, _, _ = make_backend()
    from jlens.mmpilot import audio_audit as module
    from jlens.mmpilot.backend import InvarianceError

    def boom(*args, **kwargs):
        raise InvarianceError("capture hook changed logits by 1.0e-01")

    monkeypatch.setattr(module, "check_capture_noop", boom)
    result = audit(backend)
    assert result.verdict == AUDIO_INVALID
    assert "capture_noop" in result.failed


def test_a_model_that_ignores_the_recording_is_invalid():
    """Placeholders and a tower are not enough; the output has to move."""
    backend, _, _ = make_backend()
    original = backend.hf_model.forward

    def deaf(input_ids=None, **kwargs):
        return original(input_ids=input_ids, **{**kwargs, "input_features": None})

    backend.hf_model.forward = deaf
    result = audit(backend)
    assert result.verdict == AUDIO_INVALID
    assert "waveform_differs_from_silence" in result.failed
    assert "waveforms_differ_from_each_other" in result.failed


def test_verdict_requires_every_required_check_to_have_run():
    """A missing required check is invalid, never a silent pass."""
    partial = [Check("protocol_resolved", passed=True, blocking=True)]
    assert verdict_from_checks(partial) == AUDIO_INVALID
    complete = [Check("protocol_resolved", passed=True, blocking=True)] + [
        Check(name, passed=True) for name in REQUIRED_CHECKS
    ]
    assert verdict_from_checks(complete) == AUDIO_READY


def test_blocked_wins_over_invalid():
    checks = [
        Check("protocol_resolved", passed=False, blocking=True),
        Check("capture_noop", passed=False),
    ]
    assert verdict_from_checks(checks) == AUDIO_BLOCKED


def test_audit_needs_two_different_recordings():
    backend, _, _ = make_backend()
    with pytest.raises(ValueError, match="at least two different recordings"):
        audit(backend, waveforms=[("only", wav())])


def test_recordings_of_different_lengths_are_comparable():
    """Two recordings of different durations expand to different sequence
    lengths, so the comparison has to happen at the final prompt token."""
    backend, _, _ = make_backend()
    result = audit(
        backend,
        waveforms=[("short", wav(1.0, seed=1)), ("long", wav(1.5, seed=2))],
    )
    assert result.verdict == AUDIO_READY, result.failed
    pair = next(
        check
        for check in result.checks
        if check.name == "waveforms_differ_from_each_other"
    )
    assert pair.detail["compared_at"] == "final_prompt_token"
    lengths = set(pair.detail["prompt_lens"].values())
    assert len(lengths) == 2, "the two recordings should differ in token length"


# -------------------------------------------------------------- the report


def test_report_states_the_engineering_only_caveat(tmp_path):
    backend, _, _ = make_backend()
    result = audit(backend)
    written = write_audit_report(tmp_path / "run", result)
    markdown = (tmp_path / "run" / "audio_audit.md").read_text(encoding="utf-8")
    assert AUDIO_READY in markdown
    assert "engineering evidence only" in markdown
    assert "says nothing about whether" in markdown
    # The fingerprint consequences are stated in the artifact, not only in docs,
    # and the protocol's own digest is printed so it can be bound into a run.
    assert "change the run fingerprint" in markdown
    assert result.audio_interface["protocol_fingerprint"] in markdown

    payload = json.loads((tmp_path / "run" / "audio_audit.json").read_text(encoding="utf-8"))
    assert payload["verdict"] == AUDIO_READY
    assert payload["report_checksum"].startswith("sha256:")
    assert payload["audio_interface"]["model_revision"] == "mock-rev"
    assert written["json"].endswith("audio_audit.json")


def test_report_checksum_ignores_only_the_timestamp():
    backend, _, _ = make_backend()
    first = audit(backend).to_dict()
    second = audit(backend).to_dict()
    assert first["report_checksum"] == second["report_checksum"]

    changed = AudioAuditResult(
        verdict=AUDIO_READY, checks=[Check("protocol_resolved", passed=True)]
    ).to_dict()
    assert changed["report_checksum"] != first["report_checksum"]


def test_invalid_report_says_not_to_run_the_study(tmp_path):
    backend, _, _ = make_backend()
    caption = "a cat on a bench"
    result = audit(backend, prompt=f"Caption: {caption}\n{QUESTION}", forbidden_text=[caption])
    markdown = audit_markdown(result)
    assert AUDIO_INVALID in markdown
    assert "looks like success" in markdown
    assert "Do not run the study" in markdown


# ------------------------------------------------- never touch a finished run


@pytest.mark.parametrize(
    "name",
    [
        "mmpilot_pilot_20260803T160711",
        "robustness_20260804T101010",
        "mmlocalize_localization_20260804T215140",
        "jspace_20260716T170808536780_e4118850fb70",
    ],
)
def test_audit_refuses_to_write_inside_a_completed_run(tmp_path, name):
    protected = tmp_path / name
    protected.mkdir()
    with pytest.raises(ValueError, match="completed run directory"):
        new_audit_run_dir(protected, mode="real")


def test_audit_creates_its_own_directory_under_a_runs_root(tmp_path):
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    first = new_audit_run_dir(runs_root, mode="mock", tag="a")
    second = new_audit_run_dir(runs_root, mode="mock", tag="b")
    assert first.name.startswith("audioaudit_mock_")
    assert first != second
    assert first.parent == runs_root
    # Existing sibling runs are untouched.
    assert sorted(p.name for p in runs_root.iterdir()) == sorted(
        [first.name, second.name]
    )


def test_capture_and_zero_intervention_use_the_audio_input(tmp_path):
    """The invariance checks must run on the audio input, not on a text probe."""
    backend, _, _ = make_backend()
    result = audit(backend)
    capture = next(check for check in result.checks if check.name == "capture_noop")
    zero = next(check for check in result.checks if check.name == "zero_intervention")
    assert capture.detail["max_abs_logit_diff"] == 0.0
    assert zero.detail["passed"] is True
    built = backend.build_inputs(
        prompt=PROMPT, modality="spoken_audio", audio=wav(1.0, seed=1), sampling_rate=SR
    )
    assert zero.detail["position"] == built.final_prompt_position


def test_scoring_check_requires_a_multi_token_candidate():
    """Single-token candidates would not exercise sequence scoring at all."""
    backend, _, _ = make_backend()
    result = audit(backend, candidate_ids={"cat": [30], "dog": [33]})
    scoring = next(
        check for check in result.checks if check.name == "candidate_sequence_scoring"
    )
    assert scoring.passed is False
    assert result.verdict == AUDIO_INVALID


def test_audio_span_is_recorded_for_every_built_input():
    backend, _, _ = make_backend()
    result = audit(backend)
    span_check = next(check for check in result.checks if check.name == "placeholder_span")
    assert span_check.passed is True
    start, end = span_check.detail["span"]
    assert end - start == span_check.detail["n_placeholders"]
    assert torch.is_tensor(
        backend.build_inputs(
            prompt=PROMPT, modality="spoken_audio", audio=wav(), sampling_rate=SR
        ).tensors["input_features"]
    )
