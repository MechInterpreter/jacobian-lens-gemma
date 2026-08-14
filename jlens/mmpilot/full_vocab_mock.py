# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Commissioned synthetic worlds for the full-vocabulary rerun.

Every scenario here is a **bounded prediction made before the code ran**: a
named world, and the one verdict the real aggregation and the real verdict
functions must return for it. Records go through the real
:func:`jlens.mmpilot.full_vocabulary.unrestricted_trial_record`, the real
:func:`jlens.mmpilot.full_vocab_study.summarize_full_vocab_cells` and the real
:func:`jlens.mmpilot.full_vocab_study.unrestricted_reasoning_verdict`; nothing
is asserted about a shortcut.

The nine cases, and why each one exists:

``full_vocab_go``
    The target token becomes the unique global argmax at α=1 and the controls do
    not. The only case that may return ``FULL_VOCAB_REASONING_ALPHA1_GO``.

``alpha2_only``
    α=1 does nothing; α=2 works. Must never be promoted to the α=1 primary.

``restricted_only``
    The target wins the two-candidate forced choice everywhere and is never the
    global argmax anywhere. **This is the defect the whole study exists to
    correct**, and it must return NO-GO.

``direct_answer_only``
    Only the answer arm — the positive control — passes. A positive control
    passing on its own establishes no intermediate reasoning.

``asymmetric_direction``
    bird→cat works, cat→bird does not. Reported per direction; the pooled
    summary may never hide it.

``control_failure``
    The primary rate is real and its matched controls match it. Not a result.

``capability_no_go``
    The clean unrestricted screen produced too few eligible cells. No causal
    trial ran, so this is not a null result.

``null``
    Nothing moves anywhere.

``provenance_refusal``
    The completed report's checksum does not match its pin, or the canonical
    run's report checksum pin is empty. The family refuses instead of
    approximating.

**A green MOCK run is evidence about this code and about nothing else.** No
Gemma weight is loaded, no real media is read, and no number here is a
measurement.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from jlens.mmpilot.full_vocab_study import (
    CAPABILITY_NO_GO,
    FULL_VOCAB_REASONING_ALPHA1_GO,
    FULL_VOCAB_REASONING_ALPHA2_ONLY,
    FULL_VOCAB_REASONING_NO_GO,
    FULL_VOCAB_REPORT_SCHEMA,
    MODALITIES,
    READOUT_ARMS,
)
from jlens.mmpilot.full_vocabulary import (
    FULL_VOCAB_SCORING_VERSION,
    OUTPUT_HEAD_CONVENTION,
    restricted_candidate_top1,
    unrestricted_trial_record,
)
from jlens.mmpilot.store import payload_checksum
from jlens.mmpilot.validated_band_followup import (
    FOLLOWUP_REPORT_SCHEMA,
    FOLLOWUP_STUDY_NAME,
)

__all__ = [
    "CAUSAL_SCENARIOS",
    "MOCK_ANSWERS",
    "MOCK_DIRECTED_PAIRS",
    "MOCK_N_IMAGES",
    "MOCK_VOCAB",
    "PROVENANCE_SCENARIOS",
    "CausalScenario",
    "ProvenanceScenario",
    "mock_band_followup_report",
    "mock_full_vocab_records",
    "mock_population_reuse",
    "mock_scored",
]

MOCK_VOCAB = 512
MOCK_N_IMAGES = 4
MOCK_PROMPT_LEN = 12

#: Synthetic vocabulary rows. The atom index is the token id, exactly as in a
#: real ``W_U @ J_l`` dictionary.
MOCK_ANSWERS: dict[str, int] = {"bird": 40, "cat": 41, "two": 42, "four": 43}
#: A high-probability distractor, so "the target is not the argmax" has an
#: actual argmax to be.
MOCK_DISTRACTOR_ID = 7
MOCK_DISTRACTOR_TEXT = " the"

MOCK_DIRECTED_PAIRS: tuple[dict, ...] = (
    {"source": "bird", "target": "cat"},
    {"source": "cat", "target": "bird"},
)

_PROPERTY = {"bird": "two", "cat": "four"}


def _decode(token_id: int) -> str:
    for name, ident in MOCK_ANSWERS.items():
        if int(ident) == int(token_id):
            return f" {name}"
    if int(token_id) == MOCK_DISTRACTOR_ID:
        return MOCK_DISTRACTOR_TEXT
    return f"<mock-{int(token_id)}>"


def mock_scored(
    *,
    target_token_id: int,
    source_token_id: int,
    target_is_global_argmax: bool,
    target_logprob: float = -0.20,
    source_logprob: float = -1.40,
    tie: bool = False,
) -> dict:
    """A synthetic ``score_unrestricted_next_token`` payload.

    Deliberately the same shape the real scorer returns, so the real record
    builder, the real aggregation and the real verdicts all run unchanged.
    """
    distractor_logprob = -3.0 if target_is_global_argmax else -0.05
    if tie and target_is_global_argmax:
        distractor_logprob = float(target_logprob)
    ranked = sorted(
        [
            (target_token_id, target_logprob),
            (source_token_id, source_logprob),
            (MOCK_DISTRACTOR_ID, distractor_logprob),
        ],
        key=lambda row: (-row[1], row[0]),
    )
    argmax_id = int(ranked[0][0])
    n_greater_target = sum(1 for _, value in ranked if value > target_logprob)
    n_equal_target = sum(1 for _, value in ranked if value == target_logprob)
    n_greater_source = sum(1 for _, value in ranked if value > source_logprob)
    n_equal_source = sum(1 for _, value in ranked if value == source_logprob)
    return {
        "endpoint": "unrestricted_full_vocabulary_next_token",
        "scoring_version": FULL_VOCAB_SCORING_VERSION,
        "output_head_convention": OUTPUT_HEAD_CONVENTION,
        "candidate_list_supplied": False,
        "n_candidate_positions_appended": 0,
        "scored_input_length": MOCK_PROMPT_LEN,
        "prompt_len": MOCK_PROMPT_LEN,
        "final_prompt_position": MOCK_PROMPT_LEN - 1,
        "vocab_size": MOCK_VOCAB,
        "global_argmax_token_id": argmax_id,
        "global_argmax_token": _decode(argmax_id),
        "global_argmax_logit": float(ranked[0][1]),
        "global_argmax_logprob": float(ranked[0][1]),
        "raw_logits_argmax_token_id": argmax_id,
        "argmax_agrees_with_raw_logits": True,
        "top_k": 3,
        "top_tokens": [
            {
                "rank": index + 1,
                "token_id": int(token_id),
                "token": _decode(int(token_id)),
                "logit": float(value),
                "logprob": float(value),
            }
            for index, (token_id, value) in enumerate(ranked)
        ],
        "named_tokens": {
            "target": {
                "token_id": int(target_token_id),
                "token": _decode(int(target_token_id)),
                "score": float(target_logprob),
                "logit": float(target_logprob),
                "logprob": float(target_logprob),
                "rank": n_greater_target + 1,
                "rank_optimistic": n_greater_target + 1,
                "rank_pessimistic": n_greater_target + n_equal_target,
                "rank_midrank": n_greater_target + (n_equal_target + 1) / 2.0,
                "n_strictly_greater": n_greater_target,
                "n_tied_including_self": n_equal_target,
                "is_unique_maximum": n_greater_target == 0 and n_equal_target == 1,
                "is_tied_maximum": n_greater_target == 0,
            },
            "source": {
                "token_id": int(source_token_id),
                "token": _decode(int(source_token_id)),
                "score": float(source_logprob),
                "logit": float(source_logprob),
                "logprob": float(source_logprob),
                "rank": n_greater_source + 1,
                "rank_optimistic": n_greater_source + 1,
                "rank_pessimistic": n_greater_source + n_equal_source,
                "rank_midrank": n_greater_source + (n_equal_source + 1) / 2.0,
                "n_strictly_greater": n_greater_source,
                "n_tied_including_self": n_equal_source,
                "is_unique_maximum": n_greater_source == 0 and n_equal_source == 1,
                "is_tied_maximum": n_greater_source == 0,
            },
        },
    }


@dataclass(frozen=True)
class CausalScenario:
    """A synthetic world and the one verdict it is predicted to produce."""

    name: str
    expected_verdict: str
    #: ``(band_start, arm, condition, readout, pair, modality) -> bool`` for the
    #: unrestricted endpoint.
    global_top1: Callable[..., bool]
    #: The same signature for the restricted-candidate diagnostic. Defaults to
    #: following the unrestricted result.
    restricted_top1: Callable[..., bool] | None = None
    capability_sufficient: bool = True
    causal_stage_ran: bool = True
    note: str = ""


def _pair_key(pair: Mapping) -> str:
    return f"{pair['source']}->{pair['target']}"


def _null(**_kwargs) -> bool:
    return False


def _always(**_kwargs) -> bool:
    return True


def _favorable(*, arm, condition, readout, **_kwargs) -> bool:
    return (
        arm == "intermediate"
        and condition == "swap_alpha1"
        and readout == "property"
    )


def _alpha2_only(*, arm, condition, readout, **_kwargs) -> bool:
    return (
        arm == "intermediate"
        and condition == "swap_alpha2"
        and readout == "property"
    )


def _direct_answer_only(*, arm, condition, readout, **_kwargs) -> bool:
    return arm == "answer" and condition == "swap_alpha1" and readout == "property"


def _asymmetric(*, arm, condition, readout, pair, **_kwargs) -> bool:
    return (
        arm == "intermediate"
        and condition == "swap_alpha1"
        and readout == "property"
        and pair == "bird->cat"
    )


def _control_failure(*, arm, condition, readout, **_kwargs) -> bool:
    """The primary works and so does every matched control. Not a result."""
    return (
        arm == "intermediate"
        and readout == "property"
        and condition in ("swap_alpha1", "zero", "random_alpha1", "unrelated_alpha1")
    )


CAUSAL_SCENARIOS: dict[str, CausalScenario] = {
    "full_vocab_go": CausalScenario(
        name="full_vocab_go",
        expected_verdict=FULL_VOCAB_REASONING_ALPHA1_GO,
        global_top1=_favorable,
        note="the only world that may return an alpha=1 GO",
    ),
    "alpha2_only": CausalScenario(
        name="alpha2_only",
        expected_verdict=FULL_VOCAB_REASONING_ALPHA2_ONLY,
        global_top1=_alpha2_only,
        note="alpha=2 is a sensitivity condition and is never promoted",
    ),
    "restricted_only": CausalScenario(
        name="restricted_only",
        expected_verdict=FULL_VOCAB_REASONING_NO_GO,
        global_top1=_null,
        restricted_top1=_favorable,
        note=(
            "the target wins the two-candidate forced choice and is never the "
            "global argmax. This is the defect being corrected; it is a NO-GO"
        ),
    ),
    "direct_answer_only": CausalScenario(
        name="direct_answer_only",
        expected_verdict=FULL_VOCAB_REASONING_NO_GO,
        global_top1=_direct_answer_only,
        note="a positive control passing alone establishes no reasoning claim",
    ),
    "asymmetric_direction": CausalScenario(
        name="asymmetric_direction",
        expected_verdict=FULL_VOCAB_REASONING_ALPHA1_GO,
        global_top1=_asymmetric,
        note="one direction only; the per-direction table must show it",
    ),
    "control_failure": CausalScenario(
        name="control_failure",
        expected_verdict=FULL_VOCAB_REASONING_NO_GO,
        global_top1=_control_failure,
        note="a primary rate its own matched controls match is not a result",
    ),
    "capability_no_go": CausalScenario(
        name="capability_no_go",
        expected_verdict=CAPABILITY_NO_GO,
        global_top1=_null,
        capability_sufficient=False,
        note="no causal trial ran; this is never printed as a null result",
    ),
    "null": CausalScenario(
        name="null",
        expected_verdict=FULL_VOCAB_REASONING_NO_GO,
        global_top1=_null,
        note="nothing moves anywhere",
    ),
}


@dataclass(frozen=True)
class ProvenanceScenario:
    """A completed-artifact world and the refusal it must produce."""

    name: str
    expect_refusal: bool
    because: str
    mutate: Callable[[dict], dict] = field(default=lambda report: report)


PROVENANCE_SCENARIOS: dict[str, ProvenanceScenario] = {
    "intact": ProvenanceScenario(
        name="intact",
        expect_refusal=False,
        because="the report matches its pinned checksum and its own body",
    ),
    "wrong_checksum": ProvenanceScenario(
        name="wrong_checksum",
        expect_refusal=True,
        because="the recorded report_checksum is not the pinned one",
        mutate=lambda report: {**report, "report_checksum": "sha256:" + "0" * 64},
    ),
    "edited_body": ProvenanceScenario(
        name="edited_body",
        expect_refusal=True,
        because="the body no longer hashes to its own recorded checksum",
        mutate=lambda report: {**report, "mode": "real", "study_name": "TAMPERED"},
    ),
    "mock_report": ProvenanceScenario(
        name="mock_report",
        expect_refusal=True,
        because="a MOCK report reruns nothing",
        mutate=lambda report: _rechecksum({**report, "mode": "mock"}),
    ),
    "no_population": ProvenanceScenario(
        name="no_population",
        expect_refusal=True,
        because=(
            "the exact historical population cannot be reconstructed, so the "
            "family is refused rather than approximated"
        ),
        mutate=lambda report: _rechecksum({**report, "population_groups": []}),
    ),
}


def _rechecksum(report: Mapping) -> dict:
    payload = {k: v for k, v in dict(report).items() if k != "report_checksum"}
    return {**payload, "report_checksum": payload_checksum(payload)}


def mock_population_reuse(
    *, n_images: int = MOCK_N_IMAGES, modalities: Sequence[str] = MODALITIES
) -> dict:
    """A synthetic reuse payload in the exact shape the real reader returns."""
    groups = []
    selected: dict[str, list[str]] = {}
    for concept in ("bird", "cat"):
        for index in range(int(n_images)):
            group_id = f"g-{concept}-{index}"
            groups.append(
                {
                    "group_id": group_id,
                    "image_id": f"img-{concept}-{index}",
                    "concept": concept,
                    "image_path": f"/mock/{concept}-{index}.jpg",
                    "audio_path": f"/mock/{concept}-{index}.wav",
                }
            )
            for modality in modalities:
                selected.setdefault(f"{concept}|{modality}", []).append(group_id)
    payload = {
        "source_run": "band3340_real_2a72bda9b4ba",
        "source_report_checksum": "sha256:mock-not-a-real-report",
        "reused_verbatim": True,
        "redrawn": False,
        "enlarged": False,
        "capability_reselected": False,
        "is_a_measurement_correction_rerun_on_the_same_population": True,
        "is_an_independent_replication": False,
        "n_groups": len(groups),
        "n_distinct_images": len(groups),
        "group_ids": [row["group_id"] for row in groups],
        "image_ids": [row["image_id"] for row in groups],
        "groups": groups,
        "selected_group_ids": selected,
        "original_population_digest": "sha256:mock-population",
    }
    return {
        **payload,
        "population_reuse_digest": payload_checksum(
            {k: v for k, v in payload.items() if k != "groups"}
        ),
    }


def mock_band_followup_report(
    *,
    fingerprint_pin: str,
    n_images: int = MOCK_N_IMAGES,
) -> dict:
    """A synthetic completed follow-up report the real reader accepts.

    Its ``report_checksum`` is computed **over its own body**, exactly as the
    real writer computes it, and the caller uses that value as the pin. That is
    what lets ``edited_body`` prove the reader catches a body that no longer
    hashes to its recorded checksum, and ``wrong_checksum`` prove it catches a
    recorded checksum that is not the pinned one.
    """
    reuse = mock_population_reuse(n_images=n_images)
    fingerprint_body = {"mock_followup_configuration": True}
    body = {
        "schema": FOLLOWUP_REPORT_SCHEMA,
        "mode": "real",
        "study_name": FOLLOWUP_STUDY_NAME,
        "run_dir": "band3340_real_2a72bda9b4ba",
        "fingerprint": {
            **fingerprint_body,
            "followup_fingerprint_digest": payload_checksum(fingerprint_body),
        },
        "resume": {"fingerprint_digest": str(fingerprint_pin)},
        "population": {"population_digest": "sha256:mock-population"},
        "population_groups": reuse["groups"],
        "capability_selection": {"selected_group_ids": reuse["selected_group_ids"]},
        "followup_verdict": {
            "verdict": "L33_L40_VALIDATED_BAND_FOLLOWUP_ALPHA2_SENSITIVITY_ONLY"
        },
    }
    return _rechecksum(body)


def mock_full_vocab_records(
    scenario: CausalScenario,
    *,
    bands: Sequence[Sequence[int]],
    conditions: Sequence[str],
    arms: Sequence[str],
    modalities: Sequence[str] = MODALITIES,
    readouts: Sequence[str] = READOUT_ARMS,
    directed_pairs: Sequence[Mapping] = MOCK_DIRECTED_PAIRS,
    n_images: int = MOCK_N_IMAGES,
) -> list[dict]:
    """Synthetic trial records, built by the **real** record builder."""
    from jlens.mmpilot.band_swap import CONDITION_ALPHA, band_key

    if not scenario.causal_stage_ran or not scenario.capability_sufficient:
        return []
    records: list[dict] = []
    for pair in directed_pairs:
        source, target = str(pair["source"]), str(pair["target"])
        pair_key = _pair_key(pair)
        for modality in modalities:
            for index in range(int(n_images)):
                for band in bands:
                    key = band_key(band)
                    for arm in arms:
                        for condition in conditions:
                            for readout in readouts:
                                if readout == "property":
                                    source_answer = _PROPERTY[source]
                                    target_answer = _PROPERTY[target]
                                else:
                                    source_answer, target_answer = source, target
                                kwargs = {
                                    "band_start": int(band[0]),
                                    "arm": str(arm),
                                    "condition": str(condition),
                                    "readout": str(readout),
                                    "pair": pair_key,
                                    "modality": str(modality),
                                }
                                hit = bool(scenario.global_top1(**kwargs))
                                restricted_fn = (
                                    scenario.restricted_top1
                                    or scenario.global_top1
                                )
                                restricted_hit = bool(restricted_fn(**kwargs))
                                scored = mock_scored(
                                    target_token_id=MOCK_ANSWERS[target_answer],
                                    source_token_id=MOCK_ANSWERS[source_answer],
                                    target_is_global_argmax=hit,
                                )
                                clean = mock_scored(
                                    target_token_id=MOCK_ANSWERS[target_answer],
                                    source_token_id=MOCK_ANSWERS[source_answer],
                                    target_is_global_argmax=False,
                                )
                                candidate_scores = {
                                    target_answer: {
                                        "sum_logprob": -0.2 if restricted_hit else -2.0
                                    },
                                    source_answer: {"sum_logprob": -1.0},
                                }
                                record = unrestricted_trial_record(
                                    scored,
                                    trial_kind="intervention",
                                    condition=str(condition),
                                    arm=str(arm),
                                    band=list(band),
                                    alpha=float(CONDITION_ALPHA[str(condition)]),
                                    modality=str(modality),
                                    readout=str(readout),
                                    source_answer=source_answer,
                                    target_answer=target_answer,
                                    source_token_id=MOCK_ANSWERS[source_answer],
                                    target_token_id=MOCK_ANSWERS[target_answer],
                                    source_concept=source,
                                    target_concept=target,
                                    group_id=f"g-{source}-{index}",
                                    image_id=f"img-{source}-{index}",
                                    restricted=restricted_candidate_top1(
                                        candidate_scores, target_answer
                                    ),
                                    clean={
                                        "global_argmax_token_id": clean[
                                            "global_argmax_token_id"
                                        ],
                                        "global_argmax_token": clean[
                                            "global_argmax_token"
                                        ],
                                        "target_rank": clean["named_tokens"]["target"][
                                            "rank"
                                        ],
                                        "target_logprob": clean["named_tokens"][
                                            "target"
                                        ]["logprob"],
                                    },
                                )
                                record["band_key"] = key
                                records.append(record)
    return records


#: Named so a MOCK report can never be mistaken for a real one downstream.
MOCK_REPORT_SCHEMA = FULL_VOCAB_REPORT_SCHEMA
