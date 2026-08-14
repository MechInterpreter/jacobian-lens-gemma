# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""The unrestricted next-token endpoint, and the vocabulary of endpoints itself.

Every completed behavioral result in this repository was produced by
:func:`jlens.mmpilot.capability.score_candidate_sequences` and
:func:`jlens.mmpilot.capability.prediction_and_margin`. That pair teacher-forces
each **predeclared** candidate and calls the best-scoring one the "prediction".
It is a mathematically valid conditional sequence-likelihood measurement and a
valid forced-choice preference, and it is **not** the model's output: nothing in
it ever consults the other ~262,000 vocabulary entries.

Anthropic's systematic evaluation asks the other question. The original prompt
is run with no answer appended, the intervention is applied during that forward
pass, and the *complete* next-token distribution at the final prompt position is
inspected: the global argmax is recorded, the target-appropriate token's rank
across the whole vocabulary is recorded, and success is counted only when that
token is global rank 1. Their spider→ant example changes the top output token
from ``8`` to ``6``; a failed trial reports the target's global rank.

This module implements that endpoint, and nothing else about it is negotiable:

* :func:`score_unrestricted_next_token` takes **no candidate list**. Named target
  token ids may be supplied, but only to read ranks and scores back out of a
  distribution that was already computed without them. The same call with a
  different set of names returns the same argmax.
* No answer token is ever appended. ``scored_input_length == prompt_len`` is a
  hard assertion, and the number of appended candidate/completion positions is
  recorded as ``0`` rather than assumed.
* The model's own output head is used. The logits come back from
  :meth:`~jlens.mmpilot.backend.PilotBackend.forward_logits` and are turned into
  log-probabilities with one ``log_softmax``. Final normalization, unembedding
  and logit softcapping are the model's; re-implementing them here would be a
  second thing to keep in step, and the double-norm trap is real.
* The scorer's argmax is checked against a raw ``logits.argmax()`` on the same
  tensor. A disagreement is a refusal, not a warning.

Single-token answers only
=========================

Global argmax is a statement about **one** next token. A multi-token answer has
no global top-1 — the complete-sequence likelihood of ``" mi|cro|wave"`` is not
comparable to the probability mass of a single vocabulary row, and taking only
its first token silently compares prefixes. So :func:`resolve_answer_token`
refuses a multi-token answer for this endpoint, and the caller records it as a
sequence-likelihood diagnostic instead. Nothing is downgraded silently.

Restricted-candidate scoring survives here as a clearly labelled secondary
diagnostic (``target_is_restricted_candidate_top1``), so the corrected run can
report both endpoints from the same population and show where they disagree.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Mapping, Sequence

import torch

from jlens.mmpilot.backend import BuiltInputs, PilotBackend
from jlens.mmpilot.store import payload_checksum

__all__ = [
    "ENDPOINT_CLASSES",
    "ENDPOINT_CONDITIONAL_LOGPROB",
    "ENDPOINT_ENGINEERING",
    "ENDPOINT_GENERATION",
    "ENDPOINT_REPRESENTATIONAL",
    "ENDPOINT_RESTRICTED_CANDIDATE",
    "ENDPOINT_UNRESTRICTED_NEXT_TOKEN",
    "FULL_VOCAB_SCORING_VERSION",
    "GENERATION_VERSION",
    "GREEDY_MATCH_RULE",
    "MultiTokenAnswerError",
    "OUTPUT_HEAD_CONVENTION",
    "UnrestrictedScoringRefused",
    "answer_token_table",
    "greedy_generate",
    "greedy_matches",
    "scoring_contract_digest",
    "normalize_generated_text",
    "resolve_answer_token",
    "restricted_candidate_top1",
    "score_unrestricted_next_token",
    "tie_aware_ranks",
    "token_decoder",
    "unrestricted_trial_record",
]

#: Bound into every unit and every fingerprint this endpoint produces. A change
#: refuses stale units rather than silently mixing two scoring rules.
FULL_VOCAB_SCORING_VERSION = "mmpilot.unrestricted_next_token_scoring.v1"

#: How the distribution is obtained. Named so the artifact says which head
#: produced it, and so nobody later "improves" it by unembedding by hand.
OUTPUT_HEAD_CONVENTION = (
    "model_forward_logits_then_single_log_softmax.v1: the distribution is "
    "log_softmax over the logits the model's own output head returned for the "
    "final prompt position. Final normalization, unembedding and any logit "
    "softcap are the model's and are not re-implemented here."
)

GENERATION_VERSION = "mmpilot.deterministic_greedy_demonstration.v1"

#: Predeclared before any generation runs. Normalization is fixed here so an
#: "exact answer match" can never be loosened after seeing the completions.
GREEDY_MATCH_RULE = (
    "mmpilot.greedy_exact_match.v1: NFKC-normalize, casefold, strip leading and "
    "trailing whitespace and ASCII punctuation, collapse internal whitespace; "
    "the completion matches when the normalized generated text *begins with* "
    "the normalized answer as a whole word. Nothing else counts as a match."
)

# --------------------------------------------------------------- endpoint classes

#: The unrestricted endpoint: the complete next-token distribution decides.
ENDPOINT_UNRESTRICTED_NEXT_TOKEN = "unrestricted_full_vocabulary_next_token"

#: Argmax over a supplied candidate set. A forced-choice preference.
ENDPOINT_RESTRICTED_CANDIDATE = "restricted_candidate_rank"

#: A change in a token's or a sequence's conditional log-probability. A real
#: causal effect on a likelihood; not a statement about what the model emits.
ENDPOINT_CONDITIONAL_LOGPROB = "conditional_token_or_sequence_logprob"

#: Text the model actually produced, free or greedy.
ENDPOINT_GENERATION = "free_or_greedy_generation"

#: Activation-space or J-space measurement. Says nothing about output at all.
ENDPOINT_REPRESENTATIONAL = "activation_or_jspace_representational"

#: Hook mechanics, input-path validation, tokenizer probes, checksums.
ENDPOINT_ENGINEERING = "engineering_or_input_path_validation"

ENDPOINT_CLASSES: tuple[str, ...] = (
    ENDPOINT_UNRESTRICTED_NEXT_TOKEN,
    ENDPOINT_RESTRICTED_CANDIDATE,
    ENDPOINT_CONDITIONAL_LOGPROB,
    ENDPOINT_GENERATION,
    ENDPOINT_REPRESENTATIONAL,
    ENDPOINT_ENGINEERING,
)


class UnrestrictedScoringRefused(RuntimeError):
    """The unrestricted endpoint cannot be measured on this input as asked."""


class MultiTokenAnswerError(UnrestrictedScoringRefused):
    """An answer does not encode as exactly one next token in this context."""


# ------------------------------------------------------------- token resolution


def _encoder(backend: PilotBackend) -> Callable[[str], Sequence[int]]:
    """The tokenizer entry point, preferring a candidate-suffix-free one.

    ``encode_token`` exists on backends whose ``encode_candidate`` deliberately
    appends something (the coordinate-swap MOCK appends a suffix id so complete-
    sequence scoring is exercised). Asking "which single vocabulary row is this
    answer" is a tokenizer question and must not inherit that suffix.
    """
    encode = getattr(backend, "encode_token", None)
    if callable(encode):
        return encode
    encode = getattr(backend, "encode_candidate", None)
    if callable(encode):
        return encode
    raise UnrestrictedScoringRefused(
        "the backend exposes neither encode_token nor encode_candidate; the "
        "answer token cannot be resolved and will not be guessed"
    )


def token_decoder(backend: PilotBackend) -> Callable[[int], str]:
    """The backend's own id→text decoder.

    Raises:
        UnrestrictedScoringRefused: If the backend cannot decode. An argmax
            token id nobody can read is not a reportable global argmax.
    """
    decode = getattr(backend, "decode_token", None)
    if callable(decode):
        return lambda token_id: str(decode(int(token_id)))
    raise UnrestrictedScoringRefused(
        "the backend has no decode_token; the global argmax token could be "
        "recorded as an integer but never decoded, which is not a reportable "
        "output token"
    )


def resolve_answer_token(
    backend: PilotBackend, answer: str, *, leading_space: bool = True
) -> dict:
    """Resolve one answer to exactly one vocabulary row, or refuse.

    The leading space matters: after ``Answer:`` the model continues with
    ``" two"``, not ``"two"``. Both encodings are recorded so the refusal
    message can say which one was ambiguous.

    Raises:
        MultiTokenAnswerError: If the answer does not encode to exactly one
            token. The caller must record the answer as unsupported for the
            unrestricted endpoint and report it as a sequence-likelihood
            diagnostic instead — never by taking the first token.
    """
    encode = _encoder(backend)
    surface = f" {answer}" if leading_space else str(answer)
    ids = [int(value) for value in encode(surface)]
    bare_ids = [int(value) for value in encode(str(answer))]
    record = {
        "answer": str(answer),
        "surface": surface,
        "token_ids": ids,
        "n_tokens": len(ids),
        "bare_token_ids": bare_ids,
        "single_token": len(ids) == 1,
        "leading_space": bool(leading_space),
    }
    if len(ids) != 1:
        raise MultiTokenAnswerError(
            f"answer {answer!r} encodes as {len(ids)} tokens ({ids}) with the "
            f"surface {surface!r}. The unrestricted next-token endpoint is a "
            "statement about one vocabulary row: it is unsupported for this "
            "answer. Do not call complete-sequence teacher-forced likelihood a "
            "global top-1, and do not silently use only the first token — "
            "report it separately as a sequence-likelihood diagnostic."
        )
    record["token_id"] = ids[0]
    return record


def answer_token_table(
    backend: PilotBackend,
    answers: Sequence[str],
    *,
    required: Sequence[str] = (),
    leading_space: bool = True,
) -> dict:
    """Resolve a whole answer set, separating supported from unsupported.

    Args:
        required: Answers whose multi-tokenness must abort the study rather
            than narrow it. ``two`` and ``four`` are required for the reasoning
            experiment — if either is not a single token the experiment is
            refused *before* any model spending.

    Raises:
        MultiTokenAnswerError: If a ``required`` answer is not single-token.
    """
    supported: dict[str, dict] = {}
    unsupported: dict[str, dict] = {}
    encode = _encoder(backend)
    for answer in answers:
        try:
            supported[str(answer)] = resolve_answer_token(
                backend, str(answer), leading_space=leading_space
            )
        except MultiTokenAnswerError:
            surface = f" {answer}" if leading_space else str(answer)
            ids = [int(value) for value in encode(surface)]
            unsupported[str(answer)] = {
                "answer": str(answer),
                "surface": surface,
                "token_ids": ids,
                "n_tokens": len(ids),
                "single_token": False,
                "endpoint_supported": False,
                "reported_as": ENDPOINT_CONDITIONAL_LOGPROB,
            }
    missing = sorted(str(name) for name in required if str(name) in unsupported)
    if missing:
        detail = "; ".join(
            f"{name!r} -> {unsupported[name]['token_ids']}" for name in missing
        )
        raise MultiTokenAnswerError(
            f"the required answer(s) {missing} are not single tokens ({detail}). "
            "The unrestricted reasoning experiment is refused before any model "
            "spending: there is no global top-1 to measure for a multi-token "
            "answer, and substituting a different answer after seeing the "
            "tokenizer would change the frozen design."
        )
    duplicate_ids: dict[int, list[str]] = {}
    for name, row in supported.items():
        duplicate_ids.setdefault(int(row["token_id"]), []).append(name)
    collisions = {
        token_id: sorted(names)
        for token_id, names in duplicate_ids.items()
        if len(names) > 1
    }
    if collisions:
        raise UnrestrictedScoringRefused(
            f"ambiguous answer tokenization: {collisions}. Two answers sharing "
            "one vocabulary row cannot be told apart by a next-token argmax"
        )
    return {
        "scoring_version": FULL_VOCAB_SCORING_VERSION,
        "leading_space": bool(leading_space),
        "supported": supported,
        "unsupported": unsupported,
        "required": [str(name) for name in required],
        "all_single_token": not unsupported,
        "token_ids": {
            name: int(row["token_id"]) for name, row in sorted(supported.items())
        },
    }


# ------------------------------------------------------------------- ranks


def tie_aware_ranks(log_probs: torch.Tensor, token_id: int) -> dict:
    """Rank of ``token_id`` in a 1-D score vector, with ties made explicit.

    ``rank`` counts only strictly-better tokens, so it is the optimistic rank.
    Success in this study additionally requires *uniqueness*, which is why the
    tie count is carried next to it rather than folded away.
    """
    if log_probs.ndim != 1:
        raise UnrestrictedScoringRefused(
            f"expected a 1-D score vector, got shape {tuple(log_probs.shape)}"
        )
    vocab = int(log_probs.shape[0])
    if not 0 <= int(token_id) < vocab:
        raise UnrestrictedScoringRefused(
            f"token id {token_id} is outside the vocabulary of size {vocab}"
        )
    score = log_probs[int(token_id)]
    n_greater = int((log_probs > score).sum())
    n_equal = int((log_probs == score).sum())
    return {
        "token_id": int(token_id),
        "score": float(score),
        "rank": n_greater + 1,
        "rank_optimistic": n_greater + 1,
        "rank_pessimistic": n_greater + n_equal,
        "rank_midrank": n_greater + (n_equal + 1) / 2.0,
        "n_strictly_greater": n_greater,
        "n_tied_including_self": n_equal,
        "is_unique_maximum": n_greater == 0 and n_equal == 1,
        "is_tied_maximum": n_greater == 0,
    }


# ------------------------------------------------------------------ the scorer


def score_unrestricted_next_token(
    backend: PilotBackend,
    inputs: BuiltInputs,
    *,
    target_token_ids: Mapping[str, int] | None = None,
    top_k: int = 10,
    decode: Callable[[int], str] | None = None,
    expected_vocab_size: int | None = None,
) -> dict:
    """The complete next-token distribution at the final prompt position.

    **No candidate list.** ``target_token_ids`` is read *after* the distribution
    exists and cannot influence the argmax; calling this with a different set of
    names — or with none at all — returns the same
    ``global_argmax_token_id``. That independence is the whole point and is
    asserted by the tests.

    One forward pass. No token is appended to ``input_ids``.

    Args:
        target_token_ids: Named vocabulary rows to read ranks and scores for,
            e.g. ``{"target": 1234, "source": 5678}``.
        top_k: How many of the highest-scoring vocabulary rows to record.
        decode: ``id -> text``. Defaults to the backend's ``decode_token``.
        expected_vocab_size: Refuse if the distribution is not this wide. A
            silently different vocabulary means a different output head.

    Raises:
        UnrestrictedScoringRefused: On a non-finite logit, an unexpected
            vocabulary size, an appended position, a missing target id, or any
            disagreement between this function's argmax and ``logits.argmax()``.
    """
    tensors = dict(inputs.tensors)
    input_ids = tensors.get("input_ids")
    if not torch.is_tensor(input_ids):
        raise UnrestrictedScoringRefused("the prepared input has no input_ids tensor")
    scored_input_length = int(input_ids.shape[1])
    prompt_len = int(inputs.prompt_len)
    n_appended = scored_input_length - prompt_len
    # The hard assertion. The endpoint is defined on the untouched prompt.
    if scored_input_length != prompt_len:
        raise UnrestrictedScoringRefused(
            f"scored_input_length {scored_input_length} != prompt_len "
            f"{prompt_len}: {n_appended} position(s) were appended to input_ids. "
            "The unrestricted next-token endpoint scores the original prompt "
            "with no answer candidate appended"
        )

    logits = backend.forward_logits(tensors)
    if logits.ndim != 3 or logits.shape[0] != 1:
        raise UnrestrictedScoringRefused(
            f"expected [1, seq, vocab] logits, got {tuple(logits.shape)}"
        )
    if int(logits.shape[1]) != prompt_len:
        raise UnrestrictedScoringRefused(
            f"the model returned {int(logits.shape[1])} positions for a "
            f"{prompt_len}-token prompt; the processor expanded tokens inside "
            "the model — refusing to score"
        )
    vocab_size = int(logits.shape[2])
    if expected_vocab_size is not None and vocab_size != int(expected_vocab_size):
        raise UnrestrictedScoringRefused(
            f"vocabulary size {vocab_size} != expected {int(expected_vocab_size)}; "
            "a different output head is a different measurement"
        )

    position = int(inputs.final_prompt_position)
    next_logits = logits[0, position].float()
    if not bool(torch.isfinite(next_logits).all()):
        n_bad = int((~torch.isfinite(next_logits)).sum())
        raise UnrestrictedScoringRefused(
            f"{n_bad} non-finite logit(s) at the final prompt position; the "
            "distribution is undefined and will not be reported"
        )

    log_probs = torch.log_softmax(next_logits, dim=-1)
    argmax_id = int(torch.argmax(log_probs))
    raw_argmax_id = int(torch.argmax(next_logits))
    if argmax_id != raw_argmax_id:
        raise UnrestrictedScoringRefused(
            f"scorer argmax {argmax_id} disagrees with logits.argmax() "
            f"{raw_argmax_id}; log_softmax is order-preserving, so this is a "
            "bug and not a rounding artifact"
        )

    decoder = decode if decode is not None else token_decoder(backend)
    k = max(1, min(int(top_k), vocab_size))
    top_values, top_indices = torch.topk(next_logits, k)
    top_tokens = [
        {
            "rank": index + 1,
            "token_id": int(token_id),
            "token": decoder(int(token_id)),
            "logit": float(top_values[index]),
            "logprob": float(log_probs[int(token_id)]),
        }
        for index, token_id in enumerate(top_indices.tolist())
    ]

    named: dict[str, dict] = {}
    for name, token_id in dict(target_token_ids or {}).items():
        if token_id is None:
            raise UnrestrictedScoringRefused(
                f"target {name!r} has no token id; the unrestricted endpoint "
                "cannot report a rank for a token it was not given"
            )
        row = tie_aware_ranks(log_probs, int(token_id))
        row["token"] = decoder(int(token_id))
        row["logit"] = float(next_logits[int(token_id)])
        row["logprob"] = float(log_probs[int(token_id)])
        named[str(name)] = row

    return {
        "endpoint": ENDPOINT_UNRESTRICTED_NEXT_TOKEN,
        "scoring_version": FULL_VOCAB_SCORING_VERSION,
        "output_head_convention": OUTPUT_HEAD_CONVENTION,
        "candidate_list_supplied": False,
        "n_candidate_positions_appended": 0,
        "scored_input_length": scored_input_length,
        "prompt_len": prompt_len,
        "final_prompt_position": position,
        "vocab_size": vocab_size,
        "global_argmax_token_id": argmax_id,
        "global_argmax_token": decoder(argmax_id),
        "global_argmax_logit": float(next_logits[argmax_id]),
        "global_argmax_logprob": float(log_probs[argmax_id]),
        "raw_logits_argmax_token_id": raw_argmax_id,
        "argmax_agrees_with_raw_logits": True,
        "top_k": k,
        "top_tokens": top_tokens,
        "named_tokens": named,
    }


def restricted_candidate_top1(
    candidate_scores: Mapping[str, Mapping], target: str
) -> dict:
    """The **secondary** diagnostic: argmax over a supplied candidate set only.

    Kept so the corrected run can print both endpoints from one population and
    show exactly where they disagree. Every field name says ``restricted``; none
    of them may be reported as the model's output.
    """
    ranked = sorted(
        candidate_scores, key=lambda name: (-float(candidate_scores[name]["sum_logprob"]), name)
    )
    others = [
        float(row["sum_logprob"])
        for name, row in candidate_scores.items()
        if name != target
    ]
    return {
        "endpoint": ENDPOINT_RESTRICTED_CANDIDATE,
        "restricted_candidate_universe": sorted(candidate_scores),
        "n_restricted_candidates": len(ranked),
        "restricted_prediction": ranked[0] if ranked else None,
        "restricted_target_rank": ranked.index(str(target)) + 1 if target in ranked else None,
        "restricted_target_score": float(candidate_scores[target]["sum_logprob"])
        if target in candidate_scores
        else None,
        "restricted_target_margin": (
            float(candidate_scores[target]["sum_logprob"]) - max(others)
            if target in candidate_scores and others
            else None
        ),
        "restricted_ranking": ranked,
        "is_not_the_model_output": True,
    }


def unrestricted_trial_record(
    scored: Mapping,
    *,
    trial_kind: str,
    condition: str,
    arm: str | None = None,
    band: Sequence[int] | None = None,
    layer: int | None = None,
    alpha: float | None = None,
    modality: str,
    readout: str,
    source_answer: str,
    target_answer: str,
    source_token_id: int | None,
    target_token_id: int,
    source_concept: str | None = None,
    target_concept: str | None = None,
    group_id: str,
    image_id: str | None = None,
    recording_id: str | None = None,
    prompt_hash: str | None = None,
    media_checksum: str | None = None,
    hook_integrity: Mapping | None = None,
    model_pins: Mapping | None = None,
    restricted: Mapping | None = None,
    clean: Mapping | None = None,
) -> dict:
    """One stored unrestricted trial, in the shape the unit store persists.

    Everything a verdict may read is derived here from one scored forward pass,
    so the argmax, the rank, the log-probability and the margin cannot drift
    apart. ``restricted`` is the optional secondary diagnostic and is stored
    under names that cannot be mistaken for the primary endpoint.
    """
    named = dict(scored.get("named_tokens") or {})
    if "target" not in named:
        raise UnrestrictedScoringRefused(
            "the scored record has no 'target' named token; an unrestricted "
            "trial without a target rank is not a trial"
        )
    target = named["target"]
    source = named.get("source")
    argmax_id = int(scored["global_argmax_token_id"])
    record = {
        "status": "complete",
        "endpoint": ENDPOINT_UNRESTRICTED_NEXT_TOKEN,
        "scoring_version": scored["scoring_version"],
        "output_head_convention": scored["output_head_convention"],
        "trial_kind": str(trial_kind),
        "condition": str(condition),
        "arm": None if arm is None else str(arm),
        "band": None if band is None else [int(value) for value in band],
        "layer": None if layer is None else int(layer),
        "alpha": None if alpha is None else float(alpha),
        "alpha_is_extrapolation": bool(alpha is not None and float(alpha) > 1.0),
        "modality": str(modality),
        "readout": str(readout),
        "group_id": str(group_id),
        "image_id": None if image_id is None else str(image_id),
        "recording_id": None if recording_id is None else str(recording_id),
        "prompt_hash": prompt_hash,
        "media_checksum": media_checksum,
        # --- the distribution
        "prompt_len": int(scored["prompt_len"]),
        "final_prompt_position": int(scored["final_prompt_position"]),
        "scored_input_length": int(scored["scored_input_length"]),
        "n_candidate_positions_appended": int(scored["n_candidate_positions_appended"]),
        "vocab_size": int(scored["vocab_size"]),
        "global_argmax_token_id": argmax_id,
        "global_argmax_token": scored["global_argmax_token"],
        "top_tokens": list(scored["top_tokens"]),
        # --- the answers, and the concept pair they belong to. The identity and
        # property readouts have different answers for the same directed pair, so
        # aggregation keys on the concepts and never on the answer strings.
        "source_answer": str(source_answer),
        "target_answer": str(target_answer),
        "source_concept": str(source_concept or source_answer),
        "target_concept": str(target_concept or target_answer),
        "source_token_id": None if source_token_id is None else int(source_token_id),
        "target_token_id": int(target_token_id),
        "target_rank": int(target["rank"]),
        "target_rank_optimistic": int(target["rank_optimistic"]),
        "target_rank_pessimistic": int(target["rank_pessimistic"]),
        "target_rank_midrank": float(target["rank_midrank"]),
        "target_logit": float(target["logit"]),
        "target_logprob": float(target["logprob"]),
        "source_rank": None if source is None else int(source["rank"]),
        "source_rank_pessimistic": None if source is None else int(source["rank_pessimistic"]),
        "source_logit": None if source is None else float(source["logit"]),
        "source_logprob": None if source is None else float(source["logprob"]),
        "target_minus_source_logprob": (
            None if source is None else float(target["logprob"]) - float(source["logprob"])
        ),
        "target_is_unique_global_top1": bool(target["is_unique_maximum"]),
        "target_is_tied_global_top1": bool(target["is_tied_maximum"]),
        "global_argmax_is_target": argmax_id == int(target_token_id),
        # --- provenance
        "hook_integrity": dict(hook_integrity) if hook_integrity else None,
        "model_pins": dict(model_pins) if model_pins else None,
    }
    if restricted is not None:
        row = dict(restricted)
        record["target_is_restricted_candidate_top1"] = bool(
            row.get("restricted_prediction") == str(target_answer)
        )
        record["restricted_candidate_diagnostic"] = row
    else:
        record["target_is_restricted_candidate_top1"] = None
        record["restricted_candidate_diagnostic"] = None
    if clean is not None:
        record["clean_global_argmax_token_id"] = int(clean["global_argmax_token_id"])
        record["clean_global_argmax_token"] = clean["global_argmax_token"]
        record["clean_target_rank"] = int(clean["target_rank"])
        record["clean_target_logprob"] = float(clean["target_logprob"])
        record["target_logprob_change"] = float(target["logprob"]) - float(
            clean["target_logprob"]
        )
        record["target_rank_improvement"] = int(clean["target_rank"]) - int(
            target["rank"]
        )
        record["global_argmax_changed"] = argmax_id != int(
            clean["global_argmax_token_id"]
        )
    record["unit_checksum_payload_version"] = FULL_VOCAB_SCORING_VERSION
    return record


# ------------------------------------------------------------------ generation

_PUNCTUATION = re.compile(r"^[\s\W_]+|[\s\W_]+$", flags=re.UNICODE)
_WHITESPACE = re.compile(r"\s+")


def normalize_generated_text(text: str) -> str:
    """The frozen normalization :data:`GREEDY_MATCH_RULE` describes."""
    value = unicodedata.normalize("NFKC", str(text)).casefold()
    value = _WHITESPACE.sub(" ", value)
    return _PUNCTUATION.sub("", value).strip()


def greedy_matches(generated: str, answer: str) -> bool:
    """Whether a completion counts as the answer, under the frozen rule."""
    normalized = normalize_generated_text(generated)
    wanted = normalize_generated_text(answer)
    if not wanted:
        return False
    if normalized == wanted:
        return True
    return normalized.startswith(f"{wanted} ")


@torch.no_grad()
def greedy_generate(
    backend: PilotBackend,
    inputs: BuiltInputs,
    *,
    max_new_tokens: int = 4,
    decode: Callable[[int], str] | None = None,
    answer: str | None = None,
) -> dict:
    """Deterministic greedy continuation — a **secondary** demonstration.

    Temperature 0, no sampling, a fixed small budget, and one forward pass per
    generated token (no cache, so the intervention hooks the caller installed
    fire on every pass exactly as they do for the scored endpoint). This exists
    to show what the model writes; it never overrides, replaces or stands in for
    the full-vocabulary next-token rank, and no verdict may be derived from it
    alone.

    Costs ``max_new_tokens`` forward passes. The caller prints that cost and
    binds it into the fingerprint before spending it.
    """
    budget = int(max_new_tokens)
    if budget < 1:
        raise UnrestrictedScoringRefused("max_new_tokens must be at least 1")
    decoder = decode if decode is not None else token_decoder(backend)
    tensors = dict(inputs.tensors)
    input_ids = tensors["input_ids"]
    if int(input_ids.shape[1]) != int(inputs.prompt_len):
        raise UnrestrictedScoringRefused(
            "greedy generation starts from the untouched prompt; "
            f"input_ids has {int(input_ids.shape[1])} positions for a "
            f"{int(inputs.prompt_len)}-token prompt"
        )

    from jlens.mmpilot.capability import _extend_tensors

    generated: list[int] = []
    n_passes = 0
    for _ in range(budget):
        step_tensors = (
            tensors
            if not generated
            else _extend_tensors(tensors, int(inputs.prompt_len), generated)
        )
        logits = backend.forward_logits(step_tensors)
        n_passes += 1
        step = logits[0, -1].float()
        if not bool(torch.isfinite(step).all()):
            raise UnrestrictedScoringRefused(
                "non-finite logits during greedy generation"
            )
        generated.append(int(torch.argmax(step)))
    text = "".join(decoder(token_id) for token_id in generated)
    record = {
        "endpoint": ENDPOINT_GENERATION,
        "generation_version": GENERATION_VERSION,
        "is_secondary_demonstration": True,
        "temperature": 0.0,
        "do_sample": False,
        "deterministic": True,
        "max_new_tokens": budget,
        "n_forward_passes": n_passes,
        "generated_token_ids": list(generated),
        "generated_text": text,
        "normalized_generated_text": normalize_generated_text(text),
        "match_rule": GREEDY_MATCH_RULE,
    }
    if answer is not None:
        record["answer"] = str(answer)
        record["normalized_answer"] = normalize_generated_text(str(answer))
        record["exact_answer_match"] = greedy_matches(text, str(answer))
    return record


def scoring_contract_digest() -> str:
    """A digest over everything that defines this endpoint's semantics."""
    return payload_checksum(
        {
            "scoring_version": FULL_VOCAB_SCORING_VERSION,
            "output_head_convention": OUTPUT_HEAD_CONVENTION,
            "generation_version": GENERATION_VERSION,
            "greedy_match_rule": GREEDY_MATCH_RULE,
            "endpoint_classes": list(ENDPOINT_CLASSES),
            "no_candidate_list": True,
            "appended_positions": 0,
            "tie_aware": ["optimistic", "pessimistic", "midrank"],
            "success_requires": "unique_global_argmax",
        }
    )
