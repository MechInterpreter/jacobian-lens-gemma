from __future__ import annotations

from copy import deepcopy

import pytest

from jlens.mmpilot.digit_reasoning_confirmation import (
    CONFIRMATION_ARM_CONDITIONS,
    CONFIRMATION_BAND,
    CONFIRMATION_CONDITIONS,
    DIGIT_ANSWERS,
    DIGIT_CONFIRMATION_CAPABILITY_NO_GO,
    DIGIT_CONFIRMATION_NO_GO,
    DIGIT_CONFIRMATION_NOT_EVALUATED,
    PRIMARY_ALPHA,
    STRONG_THREE_MODALITY_GO,
    DigitConfirmationRefused,
    aggregate_confirmation,
    confirmation_design,
    confirmation_fingerprint,
    confirmation_pass_budget,
    confirmation_trial_key,
    confirmation_verdict,
    resolve_digit_endpoints,
)


class DigitBackend:
    # Matches the pinned Gemma tokenizer behavior observed in Colab: a leading
    # space is its own token, while the bare digit is one vocabulary row.
    tokens = {" 2": [236743, 20], "2": [20], " 4": [236743, 40], "4": [40]}

    def encode_token(self, text):
        return list(self.tokens[text])

    def decode_token(self, token_id):
        return {20: "2", 40: "4"}[int(token_id)]


def test_design_is_the_prospective_digit_protocol():
    design = confirmation_design()
    assert design["band"] == list(range(33, 41)) == list(CONFIRMATION_BAND)
    assert design["primary_alpha"] == PRIMARY_ALPHA == 1.0
    assert "secondary_alpha" not in design
    assert design["concept_to_digit_answer"] == {"bird": "2", "cat": "4"}
    assert design["teacher_forcing_used"] is False
    assert design["candidate_list_supplied"] is False
    assert design["population"] == "fresh image-disjoint held-out population"
    assert design["arm_conditions"] == {
        "intermediate": list(CONFIRMATION_CONDITIONS),
        "answer": ["swap_alpha1", "zero", "random_alpha1"],
    }
    assert CONFIRMATION_ARM_CONDITIONS["answer"] == (
        "swap_alpha1",
        "zero",
        "random_alpha1",
    )


def test_digit_endpoint_is_exact_and_frozen():
    endpoint = resolve_digit_endpoints(DigitBackend())
    assert endpoint["token_ids"] == {"2": 20, "4": 40}
    assert endpoint["concept_to_answer"] == DIGIT_ANSWERS
    assert endpoint["tokenization_surface"] == "bare_digit_without_leading_space"
    assert endpoint["word_tokens_two_four_are_not_endpoints"] is True


def test_digit_endpoint_refuses_another_lexical_mismatch():
    backend = DigitBackend()
    backend.decode_token = lambda _token_id: "four"
    with pytest.raises(DigitConfirmationRefused, match="lexicalization mismatch"):
        resolve_digit_endpoints(backend)


def test_budget_counts_every_forward_and_no_backward():
    budget = confirmation_pass_budget(n_images_per_direction=8)
    assert budget["clean_unrestricted_passes"] == 384
    assert budget["intervention_unrestricted_passes"] == 336
    assert budget["one_token_greedy_verification_passes"] == 528
    assert budget["total_forward_passes"] == 1248
    assert budget["backward_passes"] == 0
    assert budget["maximum_completed_work_lost_on_disconnect"] == 0


def test_trial_keys_are_stable_and_condition_specific():
    first = confirmation_trial_key(
        group_id="g1", modality="image", arm="intermediate", condition="swap_alpha1"
    )
    second = confirmation_trial_key(
        group_id="g1", modality="image", arm="intermediate", condition="zero"
    )
    assert first == confirmation_trial_key(
        group_id="g1", modality="image", arm="intermediate", condition="swap_alpha1"
    )
    assert first != second


def _records(*, primary_success=True, controls_success=False, answer_success=True):
    rows = []
    for source, target in (("bird", "cat"), ("cat", "bird")):
        for modality in ("text", "image", "spoken_audio"):
            for index in range(8):
                for arm in ("intermediate", "answer"):
                    for condition in CONFIRMATION_ARM_CONDITIONS[arm]:
                        success = False
                        if condition == "swap_alpha1" and arm == "intermediate":
                            success = primary_success
                        elif condition == "swap_alpha1" and arm == "answer":
                            success = answer_success
                        elif condition in ("zero", "random_alpha1", "unrelated_alpha1"):
                            success = controls_success
                        rows.append(
                            {
                                "trial_kind": "trial",
                                "group_id": f"{source}-{modality}-{index}",
                                "image_id": f"{source}-{modality}-{index}",
                                "source_concept": source,
                                "target_concept": target,
                                "modality": modality,
                                "arm": arm,
                                "condition": condition,
                                "target_is_unique_global_top1": success,
                                "greedy_first_token_equals_global_argmax": True,
                                "global_argmax_token": DIGIT_ANSWERS[target]
                                if success
                                else DIGIT_ANSWERS[source],
                                "hook_integrity": {
                                    "intervention_diagnostics": {
                                        "by_layer": {
                                            "33": {
                                                "max_update_to_activation_ratio": 0.1,
                                                "min_after_to_before_ratio": 0.95,
                                                "max_after_to_before_ratio": 1.05,
                                            }
                                        },
                                        "all_finite": True,
                                        "max_coordinate_update_error": 0.0,
                                        "all_layers_are_exact_alpha_one_exchange": True,
                                    }
                                },
                            }
                        )
    return rows


def _capability(ok=True):
    return {
        "all_cells_sufficient": ok,
        "cells": [{"cell": "all", "passed": ok}],
    }


def test_favourable_world_licenses_the_strong_claim():
    aggregation = aggregate_confirmation(_records())
    assert set(aggregation["paired_primary_vs_controls"]) == {
        "zero",
        "random_alpha1",
        "unrelated_alpha1",
    }
    assert set(aggregation["paired_direct_answer_vs_controls"]) == {
        "zero",
        "random_alpha1",
    }
    verdict = confirmation_verdict(
        aggregation, capability=_capability(), causal_stage_ran=True
    )
    assert verdict["verdict"] == STRONG_THREE_MODALITY_GO
    assert verdict["strong_claim_licensed"] is True
    assert all(row["passed"] for row in verdict["clauses"])


def test_controls_that_match_treatment_force_no_go():
    aggregation = aggregate_confirmation(_records(controls_success=True))
    verdict = confirmation_verdict(
        aggregation, capability=_capability(), causal_stage_ran=True
    )
    assert verdict["verdict"] == DIGIT_CONFIRMATION_NO_GO
    assert verdict["strong_claim_licensed"] is False


def test_missing_or_inexact_exchange_diagnostics_force_no_go():
    rows = _records()
    rows[0]["hook_integrity"] = {}
    aggregation = aggregate_confirmation(rows)
    verdict = confirmation_verdict(
        aggregation, capability=_capability(), causal_stage_ran=True
    )
    assert verdict["verdict"] == DIGIT_CONFIRMATION_NO_GO
    clause = next(
        row
        for row in verdict["clauses"]
        if row["clause"] == "alpha1_coordinate_exchange_executed_exactly"
    )
    assert clause["passed"] is False
    assert clause["detail"]["n_missing_diagnostics"] == 1

    rows = _records()
    rows[0]["hook_integrity"]["intervention_diagnostics"][
        "max_coordinate_update_error"
    ] = 1e-4
    verdict = confirmation_verdict(
        aggregate_confirmation(rows),
        capability=_capability(),
        causal_stage_ran=True,
    )
    clause = next(
        row
        for row in verdict["clauses"]
        if row["clause"] == "alpha1_coordinate_exchange_executed_exactly"
    )
    assert verdict["verdict"] == DIGIT_CONFIRMATION_NO_GO
    assert clause["passed"] is False
    assert clause["detail"]["max_coordinate_update_error"] == pytest.approx(1e-4)


def test_one_failed_modality_direction_cannot_be_hidden_by_pooling():
    rows = _records()
    for row in rows:
        if (
            row["source_concept"] == "cat"
            and row["modality"] == "spoken_audio"
            and row["arm"] == "intermediate"
            and row["condition"] == "swap_alpha1"
        ):
            row["target_is_unique_global_top1"] = False
    aggregation = aggregate_confirmation(rows)
    verdict = confirmation_verdict(
        aggregation, capability=_capability(), causal_stage_ran=True
    )
    assert verdict["verdict"] == DIGIT_CONFIRMATION_NO_GO
    failed = [row["clause"] for row in verdict["clauses"] if not row["passed"]]
    assert "primary_cat_to_bird_spoken_audio" in failed


def test_direct_answer_arm_is_a_required_positive_control():
    aggregation = aggregate_confirmation(_records(answer_success=False))
    verdict = confirmation_verdict(
        aggregation, capability=_capability(), causal_stage_ran=True
    )
    assert verdict["verdict"] == DIGIT_CONFIRMATION_NO_GO
    assert any(
        not row["passed"] and row["clause"].startswith("direct_answer_positive_control")
        for row in verdict["clauses"]
    )


def test_capability_and_not_evaluated_are_distinct_from_a_null():
    no_capability = confirmation_verdict(
        None, capability=_capability(False), causal_stage_ran=False
    )
    not_run = confirmation_verdict(
        None, capability=_capability(True), causal_stage_ran=False
    )
    assert no_capability["verdict"] == DIGIT_CONFIRMATION_CAPABILITY_NO_GO
    assert not_run["verdict"] == DIGIT_CONFIRMATION_NOT_EVALUATED


def test_greedy_disagreement_invalidates_a_trial_success():
    rows = _records()
    for row in rows:
        if row["condition"] == "swap_alpha1" and row["arm"] == "intermediate":
            row["greedy_first_token_equals_global_argmax"] = False
    verdict = confirmation_verdict(
        aggregate_confirmation(rows),
        capability=_capability(),
        causal_stage_ran=True,
    )
    assert verdict["verdict"] == DIGIT_CONFIRMATION_NO_GO


def test_fingerprint_binds_endpoint_population_exclusion_and_lenses():
    base = dict(
        design=confirmation_design(),
        endpoint={"endpoint_digest": "sha256:endpoint"},
        population={"population_digest": "sha256:population", "groups": [{"group_id": "g"}]},
        exclusion={"exclusion_digest": "sha256:exclusion"},
        lens_checksums={33: "sha256:l33", 40: "sha256:l40"},
        model_pins={"revision": "rev"},
        audio_protocol_fingerprint="sha256:audio",
        prompt_protocol=[{"protocol": "open"}],
        seeds={"selection": 1},
    )
    first = confirmation_fingerprint(**base)
    changed = deepcopy(base)
    changed["endpoint"] = {"endpoint_digest": "sha256:other"}
    second = confirmation_fingerprint(**changed)
    assert (
        first["confirmation_fingerprint_digest"]
        != second["confirmation_fingerprint_digest"]
    )
