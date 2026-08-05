# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""The spoken-audio feasibility audit and its three-way verdict.

This answers one engineering question — *can a real waveform be presented to
this checkpoint through its own audio pathway, correctly and without disturbing
anything else?* — and it answers nothing about J-space, about transfer, or about
what Gemma hears. :data:`AUDIO_READY` is a statement about plumbing.

The three verdicts are deliberately not two:

``AUDIO_BLOCKED``
    The pinned checkpoint, processor or library cannot support the required
    path. A property of the software, reported with the exact broken link and a
    migration note. Nothing was measured because nothing could be.

``AUDIO_INVALID``
    The path *was* built, and then failed a check that makes any later number
    untrustworthy: a capture hook that moved the logits, a zero-coefficient edit
    that did not reproduce the clean result, an audio tower that never fired, a
    recording indistinguishable from silence, or a transcript reaching the
    prompt. This is the dangerous state — it looks like success.

``AUDIO_READY``
    Every required check passed. The channel is *technically usable* for a
    future study, and that is the whole claim.

The verdict is computed from a list of checks, in :func:`verdict_from_checks`,
so it is testable without a model and cannot be written by hand in a notebook.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jlens.mmpilot.audio import (
    AUDIO_PROTOCOL_VERSION,
    SpokenAudioUnsupportedError,
    check_audio_tower_invoked,
    silence_waveform,
)
from jlens.mmpilot.backend import (
    InvarianceError,
    check_capture_noop,
    check_zero_intervention,
)
from jlens.mmpilot.store import payload_checksum

AUDIT_VERSION = "jlens.mmpilot.spoken_audio_feasibility.v2"

#: The candidate-scoring **validity** rule, versioned separately from the audit
#: because it is a distinct claim with distinct criteria. v1 conflated two
#: things: it required the fixture to contain a multi-token candidate, and the
#: fixture was the pilot's behavioral concepts, which Gemma tokenizes as single
#: tokens (``" cat"`` -> ``[5866]``, ``" dog"`` -> ``[4799]``). Complete-sequence
#: scoring executed correctly and returned finite scores, and the audit still
#: reported FAIL — a defect in the fixture and the rule, not in the scorer.
SCORING_VALIDITY_RULE = "jlens.mmpilot.scoring_validity.v2"

AUDIO_READY = "AUDIO_READY"
AUDIO_BLOCKED = "AUDIO_BLOCKED"
AUDIO_INVALID = "AUDIO_INVALID"

#: Prefix for the audit's own run directories. Distinct from ``mmpilot_``,
#: ``robustness_`` and ``mmlocalize_`` so this can never be mistaken for, or
#: written into, a completed scientific run.
RUN_PREFIX = "audioaudit"

#: Directory-name fragments belonging to completed studies. Writing into one is
#: refused outright.
PROTECTED_RUN_FRAGMENTS = ("mmpilot", "pilot_", "robustness", "mmlocalize", "jspace_")

#: What must pass for :data:`AUDIO_READY`. Everything else is reported and does
#: not gate.
REQUIRED_CHECKS = (
    "protocol_resolved",
    "placeholder_span",
    "placeholder_feature_agreement",
    "audio_tower_invoked",
    "waveform_differs_from_silence",
    "waveforms_differ_from_each_other",
    "candidate_sequence_scoring",
    "capture_noop",
    "zero_intervention",
    "no_transcript_leakage",
    "final_prompt_position",
)


#: Fixed, predeclared candidate pool, in the order it is searched. COCO category
#: names, chosen because they are the vocabulary this project already works in
#: and because compound names are *likely* to be multi-token — never because any
#: of them is assumed to be. Every length is measured against the live tokenizer
#: in :func:`select_scoring_candidates`; the pool is a search order, not a claim.
#:
#: Single-word entries sit at the end deliberately. They are a fallback for a
#: tokenizer that fuses the compounds, and the selection still refuses unless at
#: least one chosen candidate really is multi-token.
SCORING_CANDIDATE_POOL: tuple[str, ...] = (
    "traffic light",
    "fire hydrant",
    "tennis racket",
    "dining table",
    "parking meter",
    "teddy bear",
    "sports ball",
    "cell phone",
    "wine glass",
    "baseball glove",
    "potted plant",
    "hot dog",
    "refrigerator",
    "umbrella",
    "broccoli",
    "microwave",
)


class ScoringCandidateError(RuntimeError):
    """No pool candidate exercises multi-token scoring under this tokenizer."""


def _is_prefix(a: Sequence[int], b: Sequence[int]) -> bool:
    """Whether either token sequence is a prefix of the other.

    A prefix pair makes the comparison degenerate: the shorter candidate's score
    is a strict sub-sum of the longer one's, so agreement between them says
    nothing about whether the *whole* sequence was scored.
    """
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    return list(long[: len(short)]) == list(short)


def select_scoring_candidates(
    backend: Any,
    *,
    pool: Sequence[str] = SCORING_CANDIDATE_POOL,
    n: int = 2,
) -> dict[str, list[int]]:
    """Pick candidates that actually exercise complete-sequence scoring.

    Deterministic and tokenizer-driven: every pool entry is encoded through the
    backend's own ``encode_candidate``, and the first pair in pool order
    satisfying the constraints is taken. Nothing is assumed about how any phrase
    tokenizes — that assumption is what produced a FAIL on a scorer that was
    working.

    Constraints, in priority order:

    1. Every candidate encodes to at least one token.
    2. Token sequences are distinct.
    3. Neither sequence is a prefix of the other (see :func:`_is_prefix`).
    4. **Both** candidates are multi-token, if any such pair exists.
    5. Otherwise at least one candidate is multi-token.

    Raises:
        ScoringCandidateError: If no pair satisfies even the weaker rule 5 — the
            tokenizer cannot exercise multi-token scoring at all, which is a
            refusal rather than a silently degraded check.
    """
    if n != 2:
        raise ValueError(f"the scoring-validity rule is defined for 2 candidates, got {n}")

    encoded: list[tuple[str, list[int]]] = []
    seen: set[tuple[int, ...]] = set()
    rejected: list[dict] = []
    for phrase in pool:
        ids = [int(i) for i in backend.encode_candidate(f" {phrase}")]
        if not ids:
            rejected.append({"candidate": phrase, "why": "encoded to zero tokens"})
            continue
        key = tuple(ids)
        if key in seen:
            rejected.append({"candidate": phrase, "why": "duplicate token sequence"})
            continue
        seen.add(key)
        encoded.append((phrase, ids))

    def _first_pair(require_both_multi: bool) -> dict[str, list[int]] | None:
        for i in range(len(encoded)):
            for j in range(i + 1, len(encoded)):
                (name_a, ids_a), (name_b, ids_b) = encoded[i], encoded[j]
                multi = (len(ids_a) > 1) + (len(ids_b) > 1)
                if multi < (2 if require_both_multi else 1):
                    continue
                if _is_prefix(ids_a, ids_b):
                    continue
                return {name_a: ids_a, name_b: ids_b}
        return None

    chosen = _first_pair(require_both_multi=True) or _first_pair(require_both_multi=False)
    if chosen is None:
        raise ScoringCandidateError(
            "no pair of pool candidates exercises multi-token scoring under this "
            "tokenizer: every candidate encodes to a single token, or every "
            "multi-token pair is prefix-degenerate. Complete-sequence scoring "
            "cannot be distinguished from first-token scoring here, so this is "
            "refused rather than reported as a pass.\n"
            f"  measured: "
            f"{json.dumps({name: ids for name, ids in encoded}, sort_keys=True)}\n"
            f"  rejected: {json.dumps(rejected, sort_keys=True)}"
        )
    return chosen


@dataclass
class Check:
    """One audited property: what it is, whether it held, and the numbers."""

    name: str
    passed: bool
    detail: dict = field(default_factory=dict)
    blocking: bool = False

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": bool(self.passed),
            "required": self.name in REQUIRED_CHECKS,
            "blocking": bool(self.blocking),
            "detail": dict(self.detail),
        }


def verdict_from_checks(checks: Sequence[Check | Mapping]) -> str:
    """``AUDIO_READY`` / ``AUDIO_BLOCKED`` / ``AUDIO_INVALID`` from the checks.

    A ``blocking`` failure means the path could not be built at all, which is a
    statement about the checkpoint and library — ``AUDIO_BLOCKED``. Any other
    required failure means the path was built and then misbehaved, which is
    ``AUDIO_INVALID``. Blocked wins over invalid: if the protocol never
    resolved, the downstream checks never ran and their absence is not evidence.
    """
    rows = [check if isinstance(check, Mapping) else check.to_dict() for check in checks]
    if any(row["blocking"] and not row["passed"] for row in rows):
        return AUDIO_BLOCKED
    ran = {row["name"] for row in rows}
    failed = [row["name"] for row in rows if row["required"] and not row["passed"]]
    missing = [name for name in REQUIRED_CHECKS if name not in ran]
    if failed or missing:
        return AUDIO_INVALID
    return AUDIO_READY


def check_scoring_validity(
    backend: Any,
    inputs: Any,
    candidate_ids: Mapping[str, Sequence[int]],
    *,
    tolerance: float = 1e-4,
) -> Check:
    """Does the scorer correctly score **complete** candidate sequences?

    This is a question about the mechanism, and deliberately not about the
    model's answer. Which candidate wins is recorded and ignored: the recording
    was never chosen to be about any candidate, so a "wrong" prediction here is
    the expected state, not a defect. Whether Gemma can recognize a concept from
    speech is a *behavioral capability* question belonging to the later
    SpokenCOCO experiment.

    Passes when, and only when:

    - scoring executes and every candidate has a non-empty token sequence;
    - ``n_tokens`` matches the ids actually supplied;
    - at least one candidate is multi-token, and no candidate's tokens are a
      prefix of another's;
    - every per-token term and every aggregate is finite;
    - ``sum_logprob`` equals the sum of the recorded ``token_logprobs`` within
      ``tolerance``, and ``mean_logprob * n_tokens`` agrees with it;
    - reversing the candidate order leaves each candidate's own score unchanged
      within ``tolerance``.
    """
    from jlens.mmpilot.capability import score_candidate_sequences

    forward = dict(candidate_ids)
    reverse = dict(reversed(list(forward.items())))

    detail: dict[str, Any] = {
        "rule": SCORING_VALIDITY_RULE,
        "tolerance": tolerance,
        "candidate_token_ids": {name: [int(i) for i in ids] for name, ids in forward.items()},
        "n_tokens_per_candidate": {name: len(list(ids)) for name, ids in forward.items()},
        "measures": "scoring_validity_only",
        "semantic_prediction_is_not_a_criterion": True,
    }

    try:
        scores = score_candidate_sequences(
            backend, inputs, forward, return_token_logprobs=True
        )
        reordered = score_candidate_sequences(
            backend, inputs, reverse, return_token_logprobs=True
        )
    except Exception as exc:  # noqa: BLE001 - a failure to score is the result
        detail["error"] = f"{type(exc).__name__}: {exc}"
        return Check("candidate_sequence_scoring", passed=False, detail=detail)

    detail["scores"] = scores
    failures: list[str] = []

    sequences = {name: [int(i) for i in ids] for name, ids in forward.items()}
    if not all(sequences.values()):
        failures.append("a candidate encoded to zero tokens")
    if len({tuple(ids) for ids in sequences.values()}) != len(sequences):
        failures.append("candidate token sequences are not distinct")
    names = list(sequences)
    if len(names) == 2 and _is_prefix(sequences[names[0]], sequences[names[1]]):
        failures.append(
            "one candidate's tokens are a prefix of the other's, so agreement "
            "would not distinguish complete-sequence from first-token scoring"
        )
    if not any(len(ids) > 1 for ids in sequences.values()):
        failures.append(
            "no candidate is multi-token, so complete-sequence scoring was never "
            "exercised (this is a fixture defect, not a scorer defect)"
        )

    order_deltas: dict[str, float] = {}
    for name, row in scores.items():
        ids = sequences[name]
        if row["n_tokens"] != len(ids):
            failures.append(f"{name}: n_tokens {row['n_tokens']} != {len(ids)} supplied")
        terms = [float(value) for value in row.get("token_logprobs", [])]
        if len(terms) != len(ids):
            failures.append(f"{name}: {len(terms)} per-token terms for {len(ids)} tokens")
        aggregates = [float(row["sum_logprob"]), float(row["mean_logprob"])]
        if not all(math.isfinite(value) for value in [*terms, *aggregates]):
            failures.append(f"{name}: non-finite per-token or aggregate score")
            continue
        if abs(float(row["sum_logprob"]) - sum(terms)) > tolerance:
            failures.append(
                f"{name}: sum_logprob {row['sum_logprob']!r} != sum of its own "
                f"per-token terms {sum(terms)!r}"
            )
        if abs(float(row["mean_logprob"]) * len(ids) - float(row["sum_logprob"])) > tolerance:
            failures.append(f"{name}: mean_logprob is not sum_logprob / n_tokens")
        delta = abs(float(row["sum_logprob"]) - float(reordered[name]["sum_logprob"]))
        order_deltas[name] = delta
        if delta > tolerance:
            failures.append(
                f"{name}: score moved by {delta:.3e} when the candidate order "
                "changed; scoring must not depend on the order candidates are "
                "presented in"
            )

    detail["order_invariance_max_abs_delta"] = max(order_deltas.values(), default=0.0)
    detail["order_invariance_per_candidate"] = order_deltas
    detail["failures"] = failures
    # Recorded because it is interesting, and explicitly not a criterion.
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1]["sum_logprob"], kv[0]))
    detail["reported_only_argmax"] = ranked[0][0] if ranked else None

    return Check("candidate_sequence_scoring", passed=not failures, detail=detail)


@dataclass
class AudioAuditResult:
    """Everything the feasibility report needs, and its verdict."""

    verdict: str
    checks: list[Check]
    audio_interface: dict | None = None
    blocked_reason: str = ""
    environment: dict = field(default_factory=dict)
    mode: str = "mock"

    def to_dict(self) -> dict:
        scoring = next(
            (
                check.detail
                for check in self.checks
                if check.name == "candidate_sequence_scoring"
            ),
            {},
        )
        payload = {
            "audit_version": AUDIT_VERSION,
            "audio_protocol_version": AUDIO_PROTOCOL_VERSION,
            # The scoring-validity rule and the token sequences it was applied
            # to are part of the audit's fingerprint: a verdict produced under a
            # different rule, or against candidates that tokenize differently,
            # is not the same verdict.
            "scoring_validity_rule": SCORING_VALIDITY_RULE,
            "scoring_candidate_token_ids": scoring.get("candidate_token_ids"),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "mode": self.mode,
            "verdict": self.verdict,
            "blocked_reason": self.blocked_reason,
            "environment": dict(self.environment),
            "audio_interface": self.audio_interface,
            "required_checks": list(REQUIRED_CHECKS),
            "checks": [check.to_dict() for check in self.checks],
        }
        payload["report_checksum"] = payload_checksum(
            {key: value for key, value in payload.items() if key != "created_utc"}
        )
        return payload

    @property
    def failed(self) -> list[str]:
        return [check.name for check in self.checks if not check.passed]


# ------------------------------------------------------------------ the audit


def run_audio_audit(
    backend: Any,
    *,
    prompt: str,
    waveforms: Sequence[tuple[str, Any]],
    sampling_rate: int,
    layers: Sequence[int],
    candidate_ids: Mapping[str, Sequence[int]] | None = None,
    forbidden_text: Sequence[str] = (),
    mode: str = "mock",
    environment: Mapping | None = None,
    logit_tolerance: float = 1e-4,
    score_tolerance: float = 1e-4,
) -> AudioAuditResult:
    """Run every feasibility check against a backend with audio resolved.

    Args:
        waveforms: At least two ``(label, samples)`` pairs of *different*
            recordings. Silence is added automatically as the null comparison.
        candidate_ids: Complete candidate token sequences. ``None`` selects them
            with :func:`select_scoring_candidates`, which measures token lengths
            against the live tokenizer — do not hand in the pilot's behavioral
            concepts, which Gemma encodes as single tokens and which therefore
            cannot exercise complete-sequence scoring at all.
        forbidden_text: Captions, transcripts and file stems that must not
            appear in ``prompt``.

    Never raises for a failed check — a failure is the result. It raises only if
    the caller passed something structurally impossible (fewer than two
    waveforms), or if candidate selection was delegated here and the tokenizer
    cannot supply a multi-token candidate.
    """
    from jlens.mmpilot.audio import assert_no_text_leakage

    if len(waveforms) < 2:
        raise ValueError(
            "the audit needs at least two different recordings; one cannot show "
            "that the model distinguishes them"
        )
    if candidate_ids is None:
        candidate_ids = select_scoring_candidates(backend)

    checks: list[Check] = []
    resolved = getattr(backend, "audio_interface", None)
    blocked_reason = ""

    checks.append(
        Check(
            "protocol_resolved",
            passed=resolved is not None,
            blocking=True,
            detail={
                "call_convention": getattr(resolved, "call_convention", None),
                "protocol_fingerprint": getattr(resolved, "protocol_fingerprint", None),
            },
        )
    )
    if resolved is None:
        return AudioAuditResult(
            verdict=AUDIO_BLOCKED,
            checks=checks,
            blocked_reason="the native spoken-audio protocol did not resolve; "
            "no audio input could be built",
            environment=dict(environment or {}),
            mode=mode,
        )

    # --- build every input first; a build failure is blocking, not invalid.
    built: dict[str, Any] = {}
    try:
        for label, samples in waveforms:
            built[label] = backend.build_inputs(
                prompt=prompt,
                modality="spoken_audio",
                audio=samples,
                sampling_rate=sampling_rate,
            )
        first_label = waveforms[0][0]
        # Silence of the same length, so the comparison isolates the content of
        # the recording rather than its duration (which sets the token count).
        duration_s = built[first_label].audio["waveform"]["n_samples"] / float(
            sampling_rate
        )
        built["silence"] = backend.build_inputs(
            prompt=prompt,
            modality="spoken_audio",
            audio=silence_waveform(duration_s, sampling_rate),
            sampling_rate=sampling_rate,
        )
    except SpokenAudioUnsupportedError as exc:
        checks.append(
            Check(
                "audio_input_built",
                passed=False,
                blocking=True,
                detail={"reason": exc.reason, "message": str(exc), **exc.detail},
            )
        )
        return AudioAuditResult(
            verdict=AUDIO_BLOCKED,
            checks=checks,
            blocked_reason=str(exc),
            audio_interface=resolved.to_record(),
            environment=dict(environment or {}),
            mode=mode,
        )

    checks.append(
        Check("audio_input_built", passed=True, detail={"labels": sorted(built)})
    )

    primary = built[waveforms[0][0]]
    span = primary.modality_token_range
    checks.append(
        Check(
            "placeholder_span",
            passed=bool(span) and span[1] > span[0],
            detail={"span": span, "n_placeholders": primary.audio["n_placeholders"]},
        )
    )
    checks.append(
        Check(
            "placeholder_feature_agreement",
            passed=primary.audio["n_placeholders"]
            == primary.audio["n_expected_from_features"],
            detail={
                "n_placeholders": primary.audio["n_placeholders"],
                "n_expected_from_features": primary.audio["n_expected_from_features"],
            },
        )
    )
    checks.append(
        Check(
            "final_prompt_position",
            passed=(
                primary.final_prompt_position == primary.prompt_len - 1
                and bool(span)
                and primary.final_prompt_position >= span[1]
            ),
            detail={
                "final_prompt_position": primary.final_prompt_position,
                "prompt_len": primary.prompt_len,
                "audio_span": span,
            },
        )
    )

    # --- the tower has to fire.
    try:
        tower = check_audio_tower_invoked(backend, primary)
        checks.append(Check("audio_tower_invoked", passed=True, detail=tower))
    except SpokenAudioUnsupportedError as exc:
        checks.append(
            Check(
                "audio_tower_invoked",
                passed=False,
                detail={"reason": exc.reason, "message": str(exc)},
            )
        )

    # --- the recording has to matter.
    # Compared at the **final prompt token**, not over the whole sequence. Two
    # recordings of different lengths expand to different numbers of audio
    # placeholders and therefore to different sequence lengths, so a whole-tensor
    # comparison is not merely stricter — it is undefined. The final prompt
    # position is also the only position the study ever reads or edits.
    final_logits = {
        label: backend.forward_logits(item.tensors)[0, item.final_prompt_position].float()
        for label, item in built.items()
    }

    def _max_diff(a: str, b: str) -> float:
        return float((final_logits[a] - final_logits[b]).abs().max())

    speech_vs_silence = _max_diff(waveforms[0][0], "silence")
    checks.append(
        Check(
            "waveform_differs_from_silence",
            passed=speech_vs_silence > logit_tolerance,
            detail={
                "compared_at": "final_prompt_token",
                "max_abs_logit_diff": speech_vs_silence,
                "tolerance": logit_tolerance,
            },
        )
    )
    pair_diff = _max_diff(waveforms[0][0], waveforms[1][0])
    checks.append(
        Check(
            "waveforms_differ_from_each_other",
            passed=pair_diff > logit_tolerance,
            detail={
                "compared_at": "final_prompt_token",
                "pair": [waveforms[0][0], waveforms[1][0]],
                "max_abs_logit_diff": pair_diff,
                "tolerance": logit_tolerance,
                "prompt_lens": {
                    label: built[label].prompt_len
                    for label in (waveforms[0][0], waveforms[1][0])
                },
            },
        )
    )

    # --- complete candidate sequences. A question about the mechanism only:
    # which candidate wins is recorded and is not a criterion.
    checks.append(
        check_scoring_validity(backend, primary, candidate_ids, tolerance=score_tolerance)
    )

    # --- the harness must not be the effect.
    for name, runner in (
        ("capture_noop", lambda: check_capture_noop(backend, primary, list(layers), tolerance=logit_tolerance)),
        ("zero_intervention", lambda: check_zero_intervention(backend, primary, int(layers[-1]), tolerance=logit_tolerance)),
    ):
        try:
            checks.append(Check(name, passed=True, detail=runner()))
        except InvarianceError as exc:
            checks.append(Check(name, passed=False, detail={"message": str(exc)}))

    # --- evidence isolation.
    try:
        leakage = assert_no_text_leakage(prompt, forbidden=list(forbidden_text))
        checks.append(Check("no_transcript_leakage", passed=True, detail=leakage))
    except ValueError as exc:
        checks.append(
            Check("no_transcript_leakage", passed=False, detail={"message": str(exc)})
        )

    return AudioAuditResult(
        verdict=verdict_from_checks(checks),
        checks=checks,
        audio_interface=resolved.to_record(),
        blocked_reason=blocked_reason,
        environment=dict(environment or {}),
        mode=mode,
    )


# ----------------------------------------------------------------- reporting


def audit_markdown(result: AudioAuditResult) -> str:
    """The report a human reads, verdict first and caveat immediately after."""
    payload = result.to_dict()
    interface = payload["audio_interface"] or {}
    lines = [
        "# Native spoken-audio feasibility audit",
        "",
        f"## {payload['verdict']}",
        "",
    ]
    if payload["verdict"] == AUDIO_READY:
        lines += [
            "A real waveform reaches this checkpoint's audio tower through its "
            "own native pathway, changes the model's output, and does so without "
            "disturbing the capture or intervention harness.",
            "",
            "**This is engineering evidence only.** It says nothing about whether "
            "J-space concept representations transfer to or from spoken audio. "
            "That is a separate experiment which has not been run.",
        ]
    elif payload["verdict"] == AUDIO_BLOCKED:
        lines += [
            "The pinned checkpoint, processor or Transformers version cannot "
            "support the required native audio path.",
            "",
            f"**Blocked because:** {payload['blocked_reason'] or 'see checks below'}",
        ]
    else:
        lines += [
            "An audio input **was** built, and then failed a check that makes any "
            "downstream number untrustworthy. This state looks like success and "
            "is not. Do not run the study.",
            "",
            "Failed: " + ", ".join(result.failed or ["(none recorded)"]),
        ]

    lines += ["", "## Environment", ""]
    for key in sorted(payload["environment"]):
        lines.append(f"- `{key}`: {payload['environment'][key]}")

    if interface:
        lines += [
            "",
            "## Resolved audio protocol",
            "",
            f"- call convention: `{interface.get('call_convention')}`",
            f"- chat template: `{interface.get('chat_template_convention')}`",
            f"- content block: `{interface.get('content_block_schema')}`",
            f"- sampling rate: {interface.get('sampling_rate')} Hz, "
            f"{interface.get('waveform_dtype')}, {interface.get('waveform_ndim')}-D mono",
            f"- placeholder token: `{interface.get('audio_token')}` "
            f"(id {interface.get('audio_token_id')}), "
            f"bracketed by `{interface.get('boa_token')}` / `{interface.get('eoa_token')}`",
            f"- feature keys: {interface.get('feature_keys')}",
            f"- processor: `{interface.get('processor_class')}` / "
            f"`{interface.get('feature_extractor_class')}`",
            f"- model revision: `{interface.get('model_revision')}`",
            f"- processor revision: `{interface.get('processor_revision')}`",
            f"- transformers: `{interface.get('transformers_version')}`",
            f"- placeholder count tracks duration: "
            f"{interface.get('dynamic_placeholder_count')} "
            f"{interface.get('notes', {}).get('placeholder_counts')}",
        ]

    scoring = next(
        (row for row in payload["checks"] if row["name"] == "candidate_sequence_scoring"),
        None,
    )
    if scoring is not None:
        detail = scoring["detail"]
        lines += [
            "",
            "## Candidate scoring: validity, not capability",
            "",
            "Two different questions, and only the first is asked here.",
            "",
            "**Scoring validity** — does the mechanism correctly score *complete* "
            "candidate token sequences? That is what this check measures: finite "
            "per-token and aggregate terms, an aggregate equal to the sum of its "
            "own per-token terms, at least one genuinely multi-token candidate, "
            "no prefix-degenerate pair, and a score that does not move when the "
            "candidate order changes.",
            "",
            "**Behavioral capability** — can Gemma recognize a concept from the "
            "recording? **Not measured here, and not measurable here.** The "
            "waveform was never selected to be about any candidate, so which "
            "candidate scores highest is meaningless and is deliberately not a "
            "criterion. That question belongs to the SpokenCOCO experiment.",
            "",
            f"- rule: `{detail.get('rule')}`",
            f"- candidates: `{detail.get('candidate_token_ids')}`",
            f"- tokens per candidate: `{detail.get('n_tokens_per_candidate')}`",
            f"- order-invariance max |Δ|: "
            f"{detail.get('order_invariance_max_abs_delta')}",
            f"- highest-scoring candidate (**reported only, not a criterion**): "
            f"`{detail.get('reported_only_argmax')}`",
        ]
        if detail.get("failures"):
            lines += ["", "Failures:"] + [
                f"- {failure}" for failure in detail["failures"]
            ]

    lines += ["", "## Checks", "", "| check | required | result | detail |", "| --- | --- | --- | --- |"]
    for row in payload["checks"]:
        detail = json.dumps(row["detail"], default=str, sort_keys=True)
        if len(detail) > 160:
            detail = detail[:157] + "..."
        lines.append(
            f"| `{row['name']}` | {'yes' if row['required'] else 'no'} | "
            f"{'PASS' if row['passed'] else 'FAIL'} | `{detail}` |"
        )

    lines += [
        "",
        "## Fingerprint consequences",
        "",
        "Any change to the model id, model revision, processor revision, "
        "Transformers version, audio protocol, placeholder convention or "
        "hook-position convention **must** change the run fingerprint. The "
        "resolved protocol's own digest is "
        f"`{interface.get('protocol_fingerprint', '(unresolved)')}`; bind it into "
        "`RunFingerprint.extra` so a run recorded under one protocol refuses to "
        "resume under another.",
        "",
        f"Report checksum: `{payload['report_checksum']}`",
        "",
    ]
    return "\n".join(lines)


def new_audit_run_dir(runs_root: str | Path, *, mode: str, tag: str = "") -> Path:
    """A fresh audit directory. Never inside a completed scientific run.

    Raises:
        ValueError: If ``runs_root`` is itself a completed pilot, robustness,
            localization or J-space run directory. The audit writes new
            artifacts and must not be able to touch finished ones.
    """
    root = Path(runs_root)
    name = root.name.casefold()
    if any(fragment in name for fragment in PROTECTED_RUN_FRAGMENTS):
        raise ValueError(
            f"{root} looks like a completed run directory. The audit writes a "
            "new directory under a runs *root* and never into an existing run."
        )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    digest = hashlib.sha256(f"{stamp}|{mode}|{tag}".encode()).hexdigest()[:8]
    run_dir = root / f"{RUN_PREFIX}_{mode}_{stamp}_{digest}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_audit_report(run_dir: str | Path, result: AudioAuditResult) -> dict:
    """Write ``audio_audit.json`` and ``audio_audit.md``; return their paths."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = result.to_dict()
    json_path = run_dir / "audio_audit.json"
    md_path = run_dir / "audio_audit.md"
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(audit_markdown(result), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path), "payload": payload}


__all__ = [
    "AUDIO_BLOCKED",
    "AUDIO_INVALID",
    "AUDIO_READY",
    "AUDIT_VERSION",
    "SCORING_CANDIDATE_POOL",
    "SCORING_VALIDITY_RULE",
    "ScoringCandidateError",
    "PROTECTED_RUN_FRAGMENTS",
    "REQUIRED_CHECKS",
    "RUN_PREFIX",
    "AudioAuditResult",
    "Check",
    "audit_markdown",
    "check_scoring_validity",
    "new_audit_run_dir",
    "run_audio_audit",
    "select_scoring_candidates",
    "verdict_from_checks",
    "write_audit_report",
]
