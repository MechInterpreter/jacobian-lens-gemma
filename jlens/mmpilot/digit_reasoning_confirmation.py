# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Prospective full-vocabulary confirmation of the animal reasoning swap.

This protocol exists because the first unrestricted rerun predeclared the word
tokens ``two`` and ``four`` while Gemma's clean greedy answers used the digit
tokens ``2`` and ``4``.  That completed run remains immutable.  This module
defines a new, independent confirmation whose endpoint is frozen to the model's
task-appropriate digit vocabulary rows *before* its fresh population is opened.

The primary intervention is the paper-style two-coordinate exchange at
``alpha=2`` over the independently validated contiguous band L33--L40.  Alpha
one is a prespecified secondary sensitivity.  Success is never inferred from a
candidate list: the target digit must be the unique argmax of the complete
next-token distribution, and a one-token greedy continuation must agree with
that argmax.  Zero, norm-matched random and unrelated-coordinate controls are
paired by photograph, modality and direction.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

from jlens.mmpilot.full_vocabulary import (
    ENDPOINT_GENERATION,
    ENDPOINT_UNRESTRICTED_NEXT_TOKEN,
    FULL_VOCAB_SCORING_VERSION,
    GENERATION_VERSION,
    answer_token_table,
    scoring_contract_digest,
)
from jlens.mmpilot.store import payload_checksum, safe_key

__all__ = [
    "CONFIRMATION_BAND",
    "CONFIRMATION_CONDITIONS",
    "CONFIRMATION_ARM_CONDITIONS",
    "DIGIT_ANSWERS",
    "DIGIT_CONFIRMATION_PROTOCOL_VERSION",
    "DIGIT_CONFIRMATION_REPORT_NAME",
    "DIGIT_CONFIRMATION_RUN_PREFIX",
    "DIGIT_CONFIRMATION_STUDY_NAME",
    "DigitConfirmationRefused",
    "DigitConfirmationThresholds",
    "aggregate_confirmation",
    "confirmation_design",
    "confirmation_fingerprint",
    "confirmation_pass_budget",
    "confirmation_report",
    "confirmation_trial_key",
    "confirmation_verdict",
    "format_confirmation_verdict",
    "resolve_digit_endpoints",
]


DIGIT_CONFIRMATION_STUDY_NAME = "PAPER_STYLE_DIGIT_REASONING_CONFIRMATION"
DIGIT_CONFIRMATION_PROTOCOL_VERSION = (
    "mmpilot.paper_style_digit_reasoning_confirmation.v1"
)
DIGIT_CONFIRMATION_REPORT_SCHEMA = (
    "jlens.mmpilot.paper_style_digit_reasoning_confirmation_report.v1"
)
DIGIT_CONFIRMATION_REPORT_NAME = "digit_reasoning_confirmation_report.json"
DIGIT_CONFIRMATION_RUN_PREFIX = "mmdigitconfirm"

CONFIRMATION_BAND: tuple[int, ...] = tuple(range(33, 41))
CONFIRMATION_MODALITIES: tuple[str, ...] = ("text", "image", "spoken_audio")
CONFIRMATION_DIRECTIONS: tuple[tuple[str, str], ...] = (
    ("bird", "cat"),
    ("cat", "bird"),
)
DIGIT_ANSWERS: dict[str, str] = {"bird": "2", "cat": "4"}
PRIMARY_ALPHA = 2.0
SECONDARY_ALPHA = 1.0
CONFIRMATION_ARMS: tuple[str, ...] = ("intermediate", "answer")
CONFIRMATION_CONDITIONS: tuple[str, ...] = (
    "swap_alpha2",
    "zero",
    "random_alpha2",
    "unrelated_alpha2",
    "swap_alpha1",
)
CONFIRMATION_ARM_CONDITIONS: dict[str, tuple[str, ...]] = {
    "intermediate": CONFIRMATION_CONDITIONS,
    "answer": ("swap_alpha2", "zero"),
}
CONTROL_CONDITIONS: tuple[str, ...] = (
    "zero",
    "random_alpha2",
    "unrelated_alpha2",
)


class DigitConfirmationRefused(RuntimeError):
    """The prospective confirmation cannot be constructed as frozen."""


@dataclass(frozen=True)
class DigitConfirmationThresholds:
    """Frozen gates for the strong three-modality claim.

    Statistical evidence is pooled only after all six modality×direction cells
    are printed.  The pooled paired tests establish that the intervention beats
    each control; every individual cell must still clear the raw success floor
    and beat the corresponding control, so pooling cannot hide a failed channel
    or direction.
    """

    min_images_per_cell: int = 8
    min_primary_success_rate_per_cell: float = 0.50
    min_positive_control_rate_per_cell: float = 0.50
    familywise_alpha: float = 0.05
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 20260813

    def __post_init__(self) -> None:
        if self.min_images_per_cell < 4:
            raise DigitConfirmationRefused("min_images_per_cell must be at least 4")
        for field in (
            "min_primary_success_rate_per_cell",
            "min_positive_control_rate_per_cell",
            "familywise_alpha",
        ):
            value = float(getattr(self, field))
            if not 0.0 < value <= 1.0:
                raise DigitConfirmationRefused(f"{field} must be in (0, 1]")
        if self.bootstrap_samples < 1_000:
            raise DigitConfirmationRefused("bootstrap_samples must be at least 1000")

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def digest(self) -> str:
        return payload_checksum(
            {"protocol": DIGIT_CONFIRMATION_PROTOCOL_VERSION, **self.to_dict()}
        )


def resolve_digit_endpoints(backend) -> dict:
    """Resolve the exact digit rows and refuse a lexicalization mismatch.

    The leading-space continuation convention is the same one used by the
    unrestricted scorer.  Both answers must be one token, distinct, and decode
    to text whose normalized content is the expected digit.
    """
    table = answer_token_table(backend, ("2", "4"), required=("2", "4"))
    decode = getattr(backend, "decode_token", None)
    if not callable(decode):
        raise DigitConfirmationRefused("the backend cannot decode endpoint tokens")
    decoded: dict[str, str] = {}
    for answer, token_id in table["token_ids"].items():
        surface = str(decode(int(token_id))).strip()
        decoded[answer] = surface
        if surface != answer:
            raise DigitConfirmationRefused(
                f"endpoint {answer!r} resolved to token {token_id} decoding as "
                f"{surface!r}; refusing another lexicalization mismatch"
            )
    payload = {
        "endpoint_rule": "exact_single_digit_vocabulary_row.v1",
        "concept_to_answer": dict(DIGIT_ANSWERS),
        "token_ids": dict(table["token_ids"]),
        "decoded": decoded,
        "all_single_token": True,
        "selected_before_population_opened": True,
        "word_tokens_two_four_are_not_endpoints": True,
    }
    return {**payload, "endpoint_digest": payload_checksum(payload)}


def confirmation_design(
    *, thresholds: DigitConfirmationThresholds | None = None
) -> dict:
    """Return the complete frozen design; no result-dependent arguments exist."""
    thresholds = thresholds or DigitConfirmationThresholds()
    payload = {
        "protocol_version": DIGIT_CONFIRMATION_PROTOCOL_VERSION,
        "study_name": DIGIT_CONFIRMATION_STUDY_NAME,
        "primary_endpoint": ENDPOINT_UNRESTRICTED_NEXT_TOKEN,
        "generation_endpoint": ENDPOINT_GENERATION,
        "scoring_version": FULL_VOCAB_SCORING_VERSION,
        "scoring_contract_digest": scoring_contract_digest(),
        "generation_version": GENERATION_VERSION,
        "band": list(CONFIRMATION_BAND),
        "modalities": list(CONFIRMATION_MODALITIES),
        "directions": [
            {"source": source, "target": target}
            for source, target in CONFIRMATION_DIRECTIONS
        ],
        "concept_to_digit_answer": dict(DIGIT_ANSWERS),
        "arms": list(CONFIRMATION_ARMS),
        "conditions": list(CONFIRMATION_CONDITIONS),
        "arm_conditions": {
            arm: list(conditions)
            for arm, conditions in CONFIRMATION_ARM_CONDITIONS.items()
        },
        "blocking_controls": list(CONTROL_CONDITIONS),
        "primary_alpha": PRIMARY_ALPHA,
        "secondary_alpha": SECONDARY_ALPHA,
        "alpha_roles": {
            "2.0": "prospective primary, motivated by the completed pilot",
            "1.0": "secondary sensitivity only",
        },
        "positions": "all_original_prompt_positions",
        "answer_appended": False,
        "candidate_list_supplied": False,
        "teacher_forcing_used": False,
        "greedy_new_tokens": 1,
        "greedy_must_equal_distribution_argmax": True,
        "population": "fresh image-disjoint held-out population",
        "population_may_be_replaced_after_results": False,
        "fitting_performed": False,
        "backward_passes": 0,
        "thresholds": thresholds.to_dict(),
        "threshold_digest": thresholds.digest,
        "claim_if_all_gates_pass": (
            "a paper-style J-lens coordinate exchange of the represented animal "
            "identity causally changes Gemma 4's unrestricted leg-count output "
            "to the target animal's digit across text, image and spoken-caption "
            "inputs on independent held-out examples and against matched controls"
        ),
    }
    return {**payload, "design_digest": payload_checksum(payload)}


def confirmation_trial_key(
    *, group_id: str, modality: str, arm: str, condition: str, kind: str = "trial"
) -> str:
    """Stable checksum-bound unit key; one completed forward result per file."""
    return safe_key(
        "digit-confirm", kind, group_id, modality, arm, condition, "L33-L40"
    )


def confirmation_pass_budget(
    *,
    n_images_per_direction: int,
    capability_candidate_images_per_direction: int = 24,
    include_alpha1: bool = True,
) -> dict:
    """Exact forward-pass budget, including one-token greedy verification."""
    cells = len(CONFIRMATION_DIRECTIONS) * len(CONFIRMATION_MODALITIES)
    selected_clean = cells * int(n_images_per_direction)
    capability_clean = cells * int(capability_candidate_images_per_direction)
    intermediate_conditions = (
        len(CONFIRMATION_ARM_CONDITIONS["intermediate"])
        if include_alpha1
        else 4
    )
    answer_conditions = len(CONFIRMATION_ARM_CONDITIONS["answer"])
    interventions = (
        cells
        * int(n_images_per_direction)
        * (intermediate_conditions + answer_conditions)
    )
    # Greedy is run for clean, primary intermediate, primary answer and zero.
    # Every capability candidate gets one-token greedy parity. Selected trials
    # additionally verify primary-intermediate, primary-answer and zero.
    greedy = capability_clean + selected_clean * 3
    payload = {
        "n_images_per_direction_modality": int(n_images_per_direction),
        "n_modality_direction_cells": cells,
        "capability_candidate_images_per_direction": int(
            capability_candidate_images_per_direction
        ),
        "clean_unrestricted_passes": capability_clean,
        "intervention_unrestricted_passes": interventions,
        "one_token_greedy_verification_passes": greedy,
        "total_forward_passes": capability_clean + interventions + greedy,
        "backward_passes": 0,
        "fitting_performed": False,
        "atomic_resume_unit": "one scored or greedy forward pass",
        "maximum_completed_work_lost_on_disconnect": 0,
    }
    return {**payload, "budget_digest": payload_checksum(payload)}


def confirmation_fingerprint(
    *,
    design: Mapping,
    endpoint: Mapping,
    population: Mapping,
    exclusion: Mapping,
    lens_checksums: Mapping,
    model_pins: Mapping,
    audio_protocol_fingerprint: str,
    prompt_protocol: Sequence[Mapping],
    seeds: Mapping,
) -> dict:
    """Everything whose change must force a new run namespace."""
    payload = {
        "protocol_version": DIGIT_CONFIRMATION_PROTOCOL_VERSION,
        "design": dict(design),
        "endpoint": dict(endpoint),
        "population_digest": population.get("population_digest"),
        "population_group_ids": sorted(
            str(row["group_id"]) for row in population.get("groups", ())
        ),
        "exclusion_digest": exclusion.get("exclusion_digest"),
        "lens_checksums": {str(k): str(v) for k, v in sorted(lens_checksums.items())},
        "model_pins": dict(model_pins),
        "audio_protocol_fingerprint": str(audio_protocol_fingerprint),
        "prompt_protocol": [dict(row) for row in prompt_protocol],
        "seeds": dict(seeds),
    }
    return {**payload, "confirmation_fingerprint_digest": payload_checksum(payload)}


def _success(row: Mapping) -> int:
    return int(
        bool(row.get("target_is_unique_global_top1"))
        and bool(row.get("greedy_first_token_equals_global_argmax", True))
    )


def _paired_rows(
    rows: Sequence[Mapping], treatment: str, control: str
) -> list[tuple[int, int, str]]:
    by_key: dict[tuple[str, str, str, str], dict[str, Mapping]] = {}
    for row in rows:
        if row.get("arm") != "intermediate":
            continue
        key = (
            str(row.get("group_id")),
            str(row.get("modality")),
            str(row.get("source_concept")),
            str(row.get("target_concept")),
        )
        by_key.setdefault(key, {})[str(row.get("condition"))] = row
    pairs = []
    for key, conditions in sorted(by_key.items()):
        if treatment in conditions and control in conditions:
            pairs.append(
                (_success(conditions[treatment]), _success(conditions[control]), "|".join(key))
            )
    return pairs


def _one_sided_paired_pvalue(pairs: Sequence[tuple[int, int, str]]) -> float:
    """Exact one-sided sign test on discordant paired binary outcomes."""
    wins = sum(treatment > control for treatment, control, _ in pairs)
    losses = sum(treatment < control for treatment, control, _ in pairs)
    n = wins + losses
    if n == 0:
        return 1.0
    return min(
        1.0,
        sum(math.comb(n, k) for k in range(wins, n + 1)) / (2**n),
    )


def _bootstrap_difference(
    pairs: Sequence[tuple[int, int, str]], *, samples: int, seed: int
) -> dict:
    if not pairs:
        return {"mean_difference": None, "ci95": [None, None]}
    differences = [float(t - c) for t, c, _ in pairs]
    mean = sum(differences) / len(differences)
    rng = random.Random(int(seed))
    draws = []
    for _ in range(int(samples)):
        draws.append(
            sum(rng.choice(differences) for _ in differences) / len(differences)
        )
    draws.sort()
    lo = draws[int(0.025 * (len(draws) - 1))]
    hi = draws[int(0.975 * (len(draws) - 1))]
    return {"mean_difference": mean, "ci95": [lo, hi]}


def _holm(pvalues: Mapping[str, float], alpha: float) -> dict[str, dict]:
    ordered = sorted((float(value), str(name)) for name, value in pvalues.items())
    out: dict[str, dict] = {}
    running = 0.0
    m = len(ordered)
    for index, (pvalue, name) in enumerate(ordered):
        adjusted = min(1.0, max(running, (m - index) * pvalue))
        running = adjusted
        out[name] = {
            "raw_pvalue": pvalue,
            "holm_adjusted_pvalue": adjusted,
            "passed": adjusted < float(alpha),
        }
    return out


def aggregate_confirmation(
    records: Sequence[Mapping], *, thresholds: DigitConfirmationThresholds | None = None
) -> dict:
    """Aggregate every direction×modality first, then paired pooled controls."""
    thresholds = thresholds or DigitConfirmationThresholds()
    rows = [dict(row) for row in records if row.get("trial_kind") == "trial"]
    cells = []
    for source, target in CONFIRMATION_DIRECTIONS:
        for modality in CONFIRMATION_MODALITIES:
            for arm in CONFIRMATION_ARMS:
                for condition in CONFIRMATION_ARM_CONDITIONS[arm]:
                    selected = [
                        row
                        for row in rows
                        if row.get("source_concept") == source
                        and row.get("target_concept") == target
                        and row.get("modality") == modality
                        and row.get("arm") == arm
                        and row.get("condition") == condition
                    ]
                    successes = sum(_success(row) for row in selected)
                    cells.append(
                        {
                            "source": source,
                            "target": target,
                            "modality": modality,
                            "arm": arm,
                            "condition": condition,
                            "n": len(selected),
                            "n_distinct_images": len(
                                {str(row.get("image_id")) for row in selected}
                            ),
                            "successes": successes,
                            "success_rate": successes / len(selected) if selected else None,
                            "argmax_tokens": sorted(
                                {str(row.get("global_argmax_token")) for row in selected}
                            ),
                        }
                    )

    paired = {}
    raw_pvalues = {}
    for index, control in enumerate(CONTROL_CONDITIONS):
        pairs = _paired_rows(rows, "swap_alpha2", control)
        pvalue = _one_sided_paired_pvalue(pairs)
        raw_pvalues[control] = pvalue
        paired[control] = {
            "n_pairs": len(pairs),
            "treatment_wins": sum(t > c for t, c, _ in pairs),
            "control_wins": sum(t < c for t, c, _ in pairs),
            "ties": sum(t == c for t, c, _ in pairs),
            **_bootstrap_difference(
                pairs,
                samples=thresholds.bootstrap_samples,
                seed=thresholds.bootstrap_seed + index,
            ),
        }
    corrected = _holm(raw_pvalues, thresholds.familywise_alpha)
    for name, result in corrected.items():
        paired[name].update(result)
    payload = {
        "protocol_version": DIGIT_CONFIRMATION_PROTOCOL_VERSION,
        "cells": cells,
        "paired_primary_vs_controls": paired,
        "n_records": len(rows),
        "threshold_digest": thresholds.digest,
    }
    return {**payload, "aggregation_digest": payload_checksum(payload)}


def _find_cell(cells: Sequence[Mapping], **wanted) -> Mapping | None:
    return next(
        (
            row
            for row in cells
            if all(str(row.get(key)) == str(value) for key, value in wanted.items())
        ),
        None,
    )


STRONG_THREE_MODALITY_GO = "DIGIT_REASONING_THREE_MODALITY_CAUSAL_GO"
DIGIT_CONFIRMATION_NO_GO = "DIGIT_REASONING_CONFIRMATION_NO_GO"
DIGIT_CONFIRMATION_CAPABILITY_NO_GO = "DIGIT_REASONING_CAPABILITY_NO_GO"
DIGIT_CONFIRMATION_NOT_EVALUATED = "DIGIT_REASONING_NOT_EVALUATED"


def confirmation_verdict(
    aggregation: Mapping | None,
    *,
    capability: Mapping,
    thresholds: DigitConfirmationThresholds | None = None,
    causal_stage_ran: bool,
) -> dict:
    """Apply the frozen conjunction; no pooled result can mask one failed cell."""
    thresholds = thresholds or DigitConfirmationThresholds()
    capability_cells = list(capability.get("cells") or ())
    capability_pass = bool(capability.get("all_cells_sufficient"))
    clauses: list[dict] = [
        {
            "clause": "clean_digit_capability_in_every_modality_and_direction",
            "passed": capability_pass,
            "detail": capability_cells,
        }
    ]
    if not capability_pass:
        verdict = DIGIT_CONFIRMATION_CAPABILITY_NO_GO
    elif not causal_stage_ran or aggregation is None:
        verdict = DIGIT_CONFIRMATION_NOT_EVALUATED
    else:
        cells = list(aggregation.get("cells") or ())
        for source, target in CONFIRMATION_DIRECTIONS:
            for modality in CONFIRMATION_MODALITIES:
                primary = _find_cell(
                    cells,
                    source=source,
                    target=target,
                    modality=modality,
                    arm="intermediate",
                    condition="swap_alpha2",
                )
                primary_rate = None if primary is None else primary.get("success_rate")
                primary_n = 0 if primary is None else int(primary.get("n_distinct_images", 0))
                cell_ok = (
                    primary is not None
                    and primary_n >= thresholds.min_images_per_cell
                    and float(primary_rate or 0.0)
                    >= thresholds.min_primary_success_rate_per_cell
                )
                control_rates = {}
                for control in CONTROL_CONDITIONS:
                    control_cell = _find_cell(
                        cells,
                        source=source,
                        target=target,
                        modality=modality,
                        arm="intermediate",
                        condition=control,
                    )
                    rate = None if control_cell is None else control_cell.get("success_rate")
                    control_rates[control] = rate
                    cell_ok = cell_ok and rate is not None and float(primary_rate) > float(rate)
                clauses.append(
                    {
                        "clause": f"primary_{source}_to_{target}_{modality}",
                        "passed": bool(cell_ok),
                        "primary_rate": primary_rate,
                        "n_images": primary_n,
                        "control_rates": control_rates,
                    }
                )

                answer = _find_cell(
                    cells,
                    source=source,
                    target=target,
                    modality=modality,
                    arm="answer",
                    condition="swap_alpha2",
                )
                answer_rate = None if answer is None else answer.get("success_rate")
                answer_zero = _find_cell(
                    cells,
                    source=source,
                    target=target,
                    modality=modality,
                    arm="answer",
                    condition="zero",
                )
                answer_zero_rate = (
                    None if answer_zero is None else answer_zero.get("success_rate")
                )
                clauses.append(
                    {
                        "clause": f"direct_answer_positive_control_{source}_to_{target}_{modality}",
                        "passed": bool(
                            answer is not None
                            and int(answer.get("n_distinct_images", 0))
                            >= thresholds.min_images_per_cell
                            and float(answer_rate or 0.0)
                            >= thresholds.min_positive_control_rate_per_cell
                            and answer_zero_rate is not None
                            and float(answer_rate) > float(answer_zero_rate)
                        ),
                        "success_rate": answer_rate,
                        "zero_control_rate": answer_zero_rate,
                    }
                )
        for name, result in dict(
            aggregation.get("paired_primary_vs_controls") or {}
        ).items():
            clauses.append(
                {
                    "clause": f"pooled_paired_primary_beats_{name}",
                    "passed": bool(result.get("passed"))
                    and float((result.get("ci95") or [None])[0] or 0.0) > 0.0,
                    "detail": dict(result),
                }
            )
        verdict = (
            STRONG_THREE_MODALITY_GO
            if all(row["passed"] for row in clauses)
            else DIGIT_CONFIRMATION_NO_GO
        )

    payload = {
        "protocol_version": DIGIT_CONFIRMATION_PROTOCOL_VERSION,
        "verdict": verdict,
        "clauses": clauses,
        "strong_claim_licensed": verdict == STRONG_THREE_MODALITY_GO,
        "licensed_claim": (
            confirmation_design(thresholds=thresholds)["claim_if_all_gates_pass"]
            if verdict == STRONG_THREE_MODALITY_GO
            else None
        ),
        "alpha2_is_primary": True,
        "alpha1_is_secondary_only": True,
        "unrestricted_full_vocabulary": True,
        "teacher_forcing_used": False,
        "threshold_digest": thresholds.digest,
    }
    return {**payload, "verdict_digest": payload_checksum(payload)}


def format_confirmation_verdict(verdict: Mapping) -> str:
    lines = ["=" * 78, f"VERDICT {verdict['verdict']}", "=" * 78]
    for row in verdict.get("clauses", ()):
        lines.append(
            f"  [{'PASS' if row.get('passed') else 'FAIL'}] {row.get('clause')}"
        )
    lines.extend(
        [
            "",
            "This is unrestricted next-token output with one-token greedy parity.",
            "No answer was appended and no candidate list decided the output.",
        ]
    )
    return "\n".join(lines)


def confirmation_report(
    *,
    design: Mapping,
    endpoint: Mapping,
    fingerprint: Mapping,
    population: Mapping,
    exclusion: Mapping,
    capability: Mapping,
    aggregation: Mapping | None,
    verdict: Mapping,
    budget: Mapping,
    resume: Mapping,
) -> dict:
    payload = {
        "schema": DIGIT_CONFIRMATION_REPORT_SCHEMA,
        "protocol_version": DIGIT_CONFIRMATION_PROTOCOL_VERSION,
        "study_name": DIGIT_CONFIRMATION_STUDY_NAME,
        "design": dict(design),
        "endpoint": dict(endpoint),
        "fingerprint": dict(fingerprint),
        "population": dict(population),
        "exclusion": dict(exclusion),
        "capability": dict(capability),
        "aggregation": None if aggregation is None else dict(aggregation),
        "verdict": dict(verdict),
        "budget": dict(budget),
        "resume": dict(resume),
        "completed_word_token_run_unchanged": True,
        "is_independent_confirmation": True,
        "is_post_hoc_reanalysis": False,
    }
    return {**payload, "report_checksum": payload_checksum(payload)}
