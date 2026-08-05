# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Candidate-scoring **validity**, and the fixture defect that faked a failure.

The first real audio audit returned FAIL on `candidate_sequence_scoring` while
scoring executed correctly and returned finite scores for both candidates. The
cause was the fixture and the rule, not the scorer: the audit was handed the
pilot's behavioral concepts, and Gemma encodes ``" cat"`` and ``" dog"`` as
single tokens, so the multi-token path was never exercised — and the rule
reported that as a scorer failure.

These tests pin both halves of the repair. The selection has to *measure* token
lengths rather than assume them, and the rule has to separate three things that
the old one ran together: whether the scorer works, whether the fixture could
show that it works, and which candidate the model happens to prefer. Only the
first two are criteria. The third is never one.
"""

import math

import pytest

from jlens.mmpilot import capability as capability_module
from jlens.mmpilot.audio_audit import (
    AUDIO_INVALID,
    AUDIO_READY,
    SCORING_CANDIDATE_POOL,
    SCORING_VALIDITY_RULE,
    ScoringCandidateError,
    check_scoring_validity,
    select_scoring_candidates,
)
from jlens.mmpilot.capability import score_candidate_sequences
from tests.test_mmpilot_audio import PROMPT, SR, make_backend, wav


@pytest.fixture(scope="module")
def bundle():
    return make_backend()


@pytest.fixture(scope="module")
def audio_input(bundle):
    backend, _, _ = bundle
    return backend.build_inputs(
        prompt=PROMPT, modality="spoken_audio", audio=wav(), sampling_rate=SR
    )


class _FixedTokenizerBackend:
    """A backend whose tokenizer behavior is stated outright, for selection."""

    def __init__(self, table):
        self._table = dict(table)

    def encode_candidate(self, text):
        return list(self._table.get(text.strip(), []))


# ----------------------------------------------- 1-3. selecting the fixture


def test_two_single_token_candidates_cannot_prove_multi_token_scoring(
    bundle, audio_input
):
    """1. The exact shape of the first real audit's fixture.

    Both scores are finite and correct. The check still fails, and the reason it
    gives is about the fixture rather than about the scorer.
    """
    backend, _, _ = bundle
    check = check_scoring_validity(backend, audio_input, {"cat": [30], "dog": [33]})
    assert check.passed is False
    assert any("fixture defect" in failure for failure in check.detail["failures"])
    assert any("no candidate is multi-token" in f for f in check.detail["failures"])
    # The scores themselves were fine — that is the whole point.
    for row in check.detail["scores"].values():
        assert math.isfinite(row["sum_logprob"])


def test_selection_finds_distinct_non_prefix_multi_token_candidates(bundle):
    """2. Deterministic, tokenizer-measured, and it prefers both multi-token."""
    backend, _, _ = bundle
    chosen = select_scoring_candidates(backend)

    assert len(chosen) == 2
    assert all(len(ids) >= 1 for ids in chosen.values())
    assert sum(len(ids) > 1 for ids in chosen.values()) == 2
    sequences = [tuple(ids) for ids in chosen.values()]
    assert sequences[0] != sequences[1]
    shorter, longer = sorted(sequences, key=len)
    assert longer[: len(shorter)] != shorter, "prefix pair would be degenerate"
    assert all(name in SCORING_CANDIDATE_POOL for name in chosen)
    # Deterministic: same backend, same choice.
    assert select_scoring_candidates(backend) == chosen


def test_selection_prefers_both_multi_token_but_accepts_one():
    """2. The weaker rule is a documented fallback, not a silent degradation."""
    only_one = _FixedTokenizerBackend(
        {
            phrase: [100 + index]
            for index, phrase in enumerate(SCORING_CANDIDATE_POOL)
        }
        | {"traffic light": [10, 11]}
    )
    chosen = select_scoring_candidates(only_one)
    assert sum(len(ids) > 1 for ids in chosen.values()) == 1
    assert "traffic light" in chosen


def test_selection_refuses_when_nothing_is_multi_token():
    """3. A tokenizer that fuses every phrase gets a refusal, not a pass."""
    all_single = _FixedTokenizerBackend(
        {phrase: [index + 10] for index, phrase in enumerate(SCORING_CANDIDATE_POOL)}
    )
    with pytest.raises(ScoringCandidateError, match="multi-token"):
        select_scoring_candidates(all_single)


def test_selection_refuses_a_prefix_only_pool():
    """3. Multi-token but prefix-degenerate is still not usable."""
    prefixed = _FixedTokenizerBackend(
        {
            phrase: [10, 11, 12][: 2 + (index % 2)]
            for index, phrase in enumerate(SCORING_CANDIDATE_POOL)
        }
    )
    with pytest.raises(ScoringCandidateError, match="prefix-degenerate"):
        select_scoring_candidates(prefixed)


def test_selection_skips_duplicate_token_sequences():
    """Two names for the same token sequence are one candidate, not two."""
    duplicated = _FixedTokenizerBackend(
        {phrase: [10, 11] for phrase in SCORING_CANDIDATE_POOL}
        | {"fire hydrant": [20, 21]}
    )
    chosen = select_scoring_candidates(duplicated)
    assert list(chosen) == ["traffic light", "fire hydrant"]


# ------------------------------------------------------- 4-8. the PASS rule


def test_finite_complete_multi_token_scores_pass(bundle, audio_input):
    """4. The repaired check on the candidates the audit actually uses."""
    backend, _, _ = bundle
    chosen = select_scoring_candidates(backend)
    check = check_scoring_validity(backend, audio_input, chosen)

    assert check.passed is True, check.detail["failures"]
    assert check.detail["rule"] == SCORING_VALIDITY_RULE
    assert check.detail["measures"] == "scoring_validity_only"
    for name, row in check.detail["scores"].items():
        assert row["n_tokens"] == len(chosen[name])
        assert len(row["token_logprobs"]) == row["n_tokens"]
        assert all(math.isfinite(value) for value in row["token_logprobs"])
        assert row["sum_logprob"] == pytest.approx(sum(row["token_logprobs"]), abs=1e-6)


def test_nonfinite_scores_fail(monkeypatch, bundle, audio_input):
    """5. A NaN anywhere is a failure, not a very small number."""
    backend, _, _ = bundle
    chosen = select_scoring_candidates(backend)
    real = capability_module.score_candidate_sequences

    def poisoned(*args, **kwargs):
        scores = real(*args, **kwargs)
        first = next(iter(scores))
        scores[first]["token_logprobs"][0] = float("nan")
        scores[first]["sum_logprob"] = float("nan")
        return scores

    monkeypatch.setattr(capability_module, "score_candidate_sequences", poisoned)
    check = check_scoring_validity(backend, audio_input, chosen)
    assert check.passed is False
    assert any("non-finite" in failure for failure in check.detail["failures"])


def test_aggregate_inconsistent_with_its_own_per_token_terms_fails(
    monkeypatch, bundle, audio_input
):
    """6. The aggregate must be the sum of the terms it reports."""
    backend, _, _ = bundle
    chosen = select_scoring_candidates(backend)
    real = capability_module.score_candidate_sequences

    def drifted(*args, **kwargs):
        scores = real(*args, **kwargs)
        first = next(iter(scores))
        scores[first]["sum_logprob"] = float(scores[first]["sum_logprob"]) - 1.0
        return scores

    monkeypatch.setattr(capability_module, "score_candidate_sequences", drifted)
    check = check_scoring_validity(backend, audio_input, chosen)
    assert check.passed is False
    assert any("per-token terms" in failure for failure in check.detail["failures"])


def test_candidate_order_invariance_passes_within_tolerance(bundle, audio_input):
    """7. Each candidate scores the same however the set is ordered."""
    backend, _, _ = bundle
    chosen = select_scoring_candidates(backend)
    check = check_scoring_validity(backend, audio_input, chosen)
    assert check.passed is True
    assert check.detail["order_invariance_max_abs_delta"] <= 1e-4
    assert set(check.detail["order_invariance_per_candidate"]) == set(chosen)


def test_candidate_order_instability_fails(monkeypatch, bundle, audio_input):
    """8. A scorer that depends on presentation order is not scoring answers."""
    backend, _, _ = bundle
    chosen = select_scoring_candidates(backend)
    real = capability_module.score_candidate_sequences
    calls = {"n": 0}

    def order_dependent(*args, **kwargs):
        scores = real(*args, **kwargs)
        calls["n"] += 1
        if calls["n"] > 1:  # the reversed-order pass
            for row in scores.values():
                row["sum_logprob"] = float(row["sum_logprob"]) + 5.0
        return scores

    monkeypatch.setattr(capability_module, "score_candidate_sequences", order_dependent)
    check = check_scoring_validity(backend, audio_input, chosen)
    assert check.passed is False
    assert any("candidate order" in failure for failure in check.detail["failures"])
    assert check.detail["order_invariance_max_abs_delta"] == pytest.approx(5.0)


# --------------------------------------------- 9. semantics are not a criterion


def test_semantic_prediction_is_not_part_of_the_verdict(bundle, audio_input):
    """9. Which candidate wins is recorded, labelled, and never a criterion.

    The recording is a generated tone; it is about neither candidate. A rule
    that cared which one won would fail here for a reason that is not a defect.
    """
    backend, _, _ = bundle
    chosen = select_scoring_candidates(backend)
    check = check_scoring_validity(backend, audio_input, chosen)

    assert check.passed is True
    assert check.detail["semantic_prediction_is_not_a_criterion"] is True
    assert check.detail["reported_only_argmax"] in chosen
    # No failure mentions the winner, the answer, or correctness.
    for failure in check.detail["failures"]:
        assert "correct" not in failure.lower()
        assert "argmax" not in failure.lower()


def test_verdict_is_unchanged_when_the_other_candidate_wins(bundle, audio_input):
    """9. Flipping which candidate scores highest changes nothing."""
    backend, _, _ = bundle
    chosen = select_scoring_candidates(backend)
    forward = check_scoring_validity(backend, audio_input, chosen)
    swapped = check_scoring_validity(
        backend, audio_input, dict(reversed(list(chosen.items())))
    )
    assert forward.passed is swapped.passed is True


# --------------------------------------------- 10-11. the audit end to end


def test_a_genuinely_broken_scorer_still_produces_audio_invalid(monkeypatch):
    """10. The repair must not have made the check unable to fail."""
    from jlens.mmpilot.audio_audit import run_audio_audit

    backend, _, _ = make_backend()
    real = capability_module.score_candidate_sequences

    def broken(*args, **kwargs):
        scores = real(*args, **kwargs)
        for row in scores.values():
            row["sum_logprob"] = float("inf")
        return scores

    monkeypatch.setattr(capability_module, "score_candidate_sequences", broken)
    result = run_audio_audit(
        backend,
        prompt=PROMPT,
        waveforms=[("a", wav(1.0, seed=1)), ("b", wav(1.0, seed=2))],
        sampling_rate=SR,
        layers=[1, 2],
        forbidden_text=[],
        mode="mock",
    )
    assert result.verdict == AUDIO_INVALID
    assert "candidate_sequence_scoring" in result.failed


def test_the_audit_selects_its_own_candidates_and_all_checks_pass():
    """11. Every previously passing check still passes, and scoring now does too."""
    from jlens.mmpilot.audio_audit import REQUIRED_CHECKS, run_audio_audit

    backend, _, _ = make_backend()
    result = run_audio_audit(
        backend,
        prompt=PROMPT,
        waveforms=[("a", wav(1.0, seed=1)), ("b", wav(1.0, seed=2))],
        sampling_rate=SR,
        layers=[1, 2],
        forbidden_text=["a cat on a bench"],
        mode="mock",
    )
    assert result.verdict == AUDIO_READY, result.failed
    assert result.failed == []
    names = {check.name for check in result.checks}
    assert names >= set(REQUIRED_CHECKS)
    # The eight checks the first real audit already passed, still passing.
    previously_passing = {
        "placeholder_span",
        "placeholder_feature_agreement",
        "final_prompt_position",
        "audio_tower_invoked",
        "waveform_differs_from_silence",
        "waveforms_differ_from_each_other",
        "capture_noop",
        "zero_intervention",
    }
    by_name = {check.name: check for check in result.checks}
    for name in previously_passing:
        assert by_name[name].passed is True, name


def test_the_rule_and_candidate_ids_are_in_the_audit_fingerprint():
    """A verdict reached under a different rule is not the same verdict."""
    from jlens.mmpilot.audio_audit import run_audio_audit

    backend, _, _ = make_backend()
    result = run_audio_audit(
        backend,
        prompt=PROMPT,
        waveforms=[("a", wav(1.0, seed=1)), ("b", wav(1.0, seed=2))],
        sampling_rate=SR,
        layers=[1, 2],
        mode="mock",
    )
    payload = result.to_dict()
    assert payload["scoring_validity_rule"] == SCORING_VALIDITY_RULE
    assert payload["scoring_candidate_token_ids"] == select_scoring_candidates(backend)
    assert payload["audit_version"].endswith(".v2")

    # Changing the candidate token ids changes the report checksum.
    baseline = payload["report_checksum"]
    other = run_audio_audit(
        backend,
        prompt=PROMPT,
        waveforms=[("a", wav(1.0, seed=1)), ("b", wav(1.0, seed=2))],
        sampling_rate=SR,
        layers=[1, 2],
        candidate_ids={"alpha": [30, 31], "beta": [33, 34, 35]},
        mode="mock",
    )
    assert other.to_dict()["report_checksum"] != baseline


# --------------------------------------------------- the scorer, directly


def test_scorer_reports_per_token_terms_only_when_asked(bundle, audio_input):
    """The scientific stages keep exactly the fields they always had."""
    backend, _, _ = bundle
    candidates = {"traffic light": [30, 31], "fire hydrant": [33, 34]}

    quiet = score_candidate_sequences(backend, audio_input, candidates)
    verbose = score_candidate_sequences(
        backend, audio_input, candidates, return_token_logprobs=True
    )

    for name, row in quiet.items():
        assert set(row) == {"sum_logprob", "mean_logprob", "n_tokens", "token_ids"}
        assert row["sum_logprob"] == pytest.approx(
            verbose[name]["sum_logprob"], abs=1e-9
        )
    for name, row in verbose.items():
        assert row["token_logprobs"] == pytest.approx(
            row["token_logprobs"], abs=0
        )
        assert len(row["token_logprobs"]) == len(candidates[name])
        assert row["sum_logprob"] == pytest.approx(sum(row["token_logprobs"]), abs=1e-9)


def test_scorer_scores_the_whole_sequence_not_the_first_token(bundle, audio_input):
    """A two-token candidate's score must depend on its second token.

    Two candidates sharing a first token and differing in the second: under
    first-token scoring their scores would be identical. The second token's own
    term has to move, and the aggregate with it.
    """
    backend, _, _ = bundle
    first = score_candidate_sequences(
        backend, audio_input, {"c": [30, 31]}, return_token_logprobs=True
    )["c"]
    second = score_candidate_sequences(
        backend, audio_input, {"c": [30, 40]}, return_token_logprobs=True
    )["c"]

    tail_shift = abs(first["token_logprobs"][1] - second["token_logprobs"][1])
    assert tail_shift > 1e-3
    assert abs(first["sum_logprob"] - second["sum_logprob"]) > 1e-3
    # The shared first token dominates neither: its term barely moves, so the
    # difference in the aggregate really does come from the second position.
    head_shift = abs(first["token_logprobs"][0] - second["token_logprobs"][0])
    assert head_shift < tail_shift
