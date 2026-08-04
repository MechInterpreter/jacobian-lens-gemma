# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Stage B: tie-aware ranks, the composite gate, and the old gate beside it.

The central case these tests pin is the one layer 32 actually produced in v2:
the target tied at the maximum with a large block of other tokens. Under that
state the optimistic rank is 1, ``argmax`` equality is a coin flip on the
tie-break, and a top-10 slice is an arbitrary ten of the tied tokens. Every
assertion about "unique top-1", "midrank" and "tied-at-max" here exists because
one of those three quantities was being read as if it were another.
"""

import hashlib
import math

import pytest
import torch

from jlens.mmlocalize.lens_validity import (
    CONTROL_VARIANTS,
    CRITERION_RANK_CONVENTION,
    DIAGNOSTIC_VARIANTS,
    LEGACY_VALIDITY_GATE,
    LOCALIZATION_VALIDITY_GATE,
    NOISE_CONTROLS,
    RANK_CONVENTIONS,
    READOUT_VARIANTS,
    RECALIBRATION_PLAN,
    InsufficientTargetDiversityError,
    LayerNotEligibleError,
    LayerValidityGate,
    RecalibrationRefused,
    assert_causally_eligible,
    check_recalibration_target,
    eligible_layers,
    evaluate_all_layers,
    evaluate_layer_validity,
    fold_of,
    gate_text,
    select_target_diverse_prompts,
    summarize_variant,
    tie_aware_row,
)

VOCAB = 200
TARGET = 7


def test_target_diverse_prompt_selection_guarantees_floor_and_is_deterministic():
    pool = [f"prompt {index}" for index in range(40)]

    def target(prompt):
        return int(prompt.rsplit(" ", 1)[1]) % 10

    first, manifest = select_target_diverse_prompts(
        pool,
        n_prompts=16,
        min_distinct_target_tokens=8,
        excluded={},
        seed=17,
        target_token_for_prompt=target,
    )
    second, again = select_target_diverse_prompts(
        list(reversed(pool)),
        n_prompts=16,
        min_distinct_target_tokens=8,
        excluded={},
        seed=17,
        target_token_for_prompt=target,
    )

    assert first == second
    assert manifest == again
    assert manifest["n_selected_distinct_target_tokens"] >= 8
    assert len({row["target_token_id"] for row in manifest["prompts"]}) >= 8


def test_target_diverse_prompt_selection_refuses_impossible_pool_before_lens_scoring():
    pool = [f"prompt {index}" for index in range(40)]
    with pytest.raises(InsufficientTargetDiversityError, match="only 7 distinct"):
        select_target_diverse_prompts(
            pool,
            n_prompts=16,
            min_distinct_target_tokens=8,
            excluded={},
            seed=17,
            target_token_for_prompt=lambda prompt: int(prompt.rsplit(" ", 1)[1]) % 7,
        )


def test_target_diverse_prompt_selection_respects_exclusions():
    pool = [f"prompt {index}" for index in range(30)]
    excluded_prompt = pool[3]
    excluded = {hashlib.sha256(excluded_prompt.encode()).hexdigest(): "prior_fit"}
    selected, manifest = select_target_diverse_prompts(
        pool,
        n_prompts=12,
        min_distinct_target_tokens=8,
        excluded=excluded,
        seed=21,
        target_token_for_prompt=lambda prompt: int(prompt.rsplit(" ", 1)[1]),
    )

    assert excluded_prompt not in selected
    assert manifest["n_selected_distinct_target_tokens"] >= 8


def _actual(target: int = TARGET) -> torch.Tensor:
    logits = torch.zeros(VOCAB)
    logits[target] = 10.0
    return logits


def _row(scores: torch.Tensor, *, variant="j_lens", layer=32, index=0) -> dict:
    return tie_aware_row(
        sample_index=index,
        prompt_sha=f"sha{index:03d}",
        layer=layer,
        variant=variant,
        variant_logits=scores,
        actual_logits=_actual(),
    )


# ----------------------------------------------------- the tie conventions


def test_a_unique_maximum_scores_rank_one_under_every_convention():
    scores = torch.zeros(VOCAB)
    scores[TARGET] = 5.0
    scores[11] = 1.0
    row = _row(scores)

    assert row["unique_top1_agreement"] is True
    assert row["argmax_top1_agreement"] is True
    assert row["tied_at_max"] is False
    assert row["n_tied_at_max"] == 1
    assert row["rank_optimistic"] == row["rank_pessimistic"] == row["rank_midrank"] == 1.0
    assert row["reciprocal_rank"] == 1.0
    assert row["margin_over_best_non_target"] == pytest.approx(4.0)


def test_the_layer_32_pathology_separates_the_three_conventions():
    """A tie block at the maximum: optimistic rank 1, no unique argmax."""
    scores = torch.zeros(VOCAB)
    tied = [TARGET, 11, 13, 17, 19]
    scores[tied] = 3.0
    row = _row(scores)

    assert row["n_tied_at_max"] == len(tied)
    assert row["tied_at_max"] is True
    # The v2 number: nothing scores strictly above the target.
    assert row["rank_optimistic"] == 1.0
    # The honest ones.
    assert row["rank_pessimistic"] == 5.0
    assert row["rank_midrank"] == 3.0
    assert row["reciprocal_rank"] == pytest.approx(1 / 3)
    # The v2 contradiction, reproduced: rank 1 with zero unique agreement.
    assert row["unique_top1_agreement"] is False
    assert row["margin_over_best_non_target"] == 0.0


def test_argmax_agreement_can_be_true_while_unique_agreement_is_false():
    """``argmax`` returns the lowest tied index, so a low-numbered target
    'agrees' for a reason that is a property of the tie-break, not the lens."""
    scores = torch.zeros(VOCAB)
    scores[[TARGET, 60, 80]] = 4.0
    row = _row(scores)

    assert row["argmax_top1_agreement"] is True     # target is the lowest index
    assert row["unique_top1_agreement"] is False
    assert row["predicted_token_id"] == TARGET

    # The same tie with a higher-numbered target flips argmax agreement without
    # changing anything about the lens.
    other = torch.zeros(VOCAB)
    other[[3, 5, TARGET]] = 4.0
    flipped = _row(other)
    assert flipped["argmax_top1_agreement"] is False
    assert flipped["unique_top1_agreement"] is False
    assert flipped["rank_midrank"] == pytest.approx(2.0)


def test_the_mrr_is_computed_from_the_midrank():
    scores = torch.zeros(VOCAB)
    scores[[TARGET, 11, 13]] = 2.0
    row = _row(scores)
    assert row["rank_convention_used_for_mrr"] == CRITERION_RANK_CONVENTION
    assert row["reciprocal_rank"] == pytest.approx(1.0 / row["rank_midrank"])


def test_top_k_inclusion_uses_the_pessimistic_rank():
    """A tie block wider than k must not manufacture top-k membership."""
    scores = torch.zeros(VOCAB)
    scores[[TARGET, *range(50, 70)]] = 1.0     # 21-way tie at the maximum
    row = _row(scores)
    assert row["rank_optimistic"] == 1.0       # would wrongly read as "in top 10"
    assert row["rank_pessimistic"] == 21.0
    assert row["in_top10"] is False


def test_a_target_beaten_outright_ranks_after_its_betters():
    scores = torch.zeros(VOCAB)
    scores[TARGET] = 1.0
    scores[[11, 13, 17]] = 5.0
    row = _row(scores)
    assert row["rank_optimistic"] == 4.0
    assert row["rank_midrank"] == 4.0
    assert row["margin_over_best_non_target"] == pytest.approx(-4.0)


def test_mismatched_logit_shapes_are_refused():
    with pytest.raises(ValueError, match="do not match"):
        tie_aware_row(
            sample_index=0,
            prompt_sha="x",
            layer=32,
            variant="j_lens",
            variant_logits=torch.zeros(10),
            actual_logits=torch.zeros(VOCAB),
        )


def test_every_declared_rank_convention_is_reported():
    row = _row(torch.rand(VOCAB))
    for convention in RANK_CONVENTIONS:
        assert f"rank_{convention}" in row


def test_the_summary_reports_ties_and_all_three_medians():
    rows = [
        _row(_tied(n), index=index)
        for index, n in enumerate((1, 1, 5, 5))
    ]
    summary = summarize_variant(rows)
    assert summary["n_prompts"] == 4
    assert summary["tied_at_max_rate"] == pytest.approx(0.5)
    assert summary["unique_top1_agreement"] == pytest.approx(0.5)
    assert summary["mean_n_tied_at_max"] == pytest.approx(3.0)
    for key in ("median_midrank", "median_optimistic_rank", "median_pessimistic_rank"):
        assert math.isfinite(summary[key])


def _tied(n_tied: int) -> torch.Tensor:
    scores = torch.zeros(VOCAB)
    scores[[TARGET, *range(100, 100 + n_tied - 1)]] = 3.0
    return scores


def test_summarizing_nothing_is_an_error_not_an_empty_pass():
    with pytest.raises(ValueError, match="empty"):
        summarize_variant([])


# ------------------------------------------------------------- the gate


def _rows_for_layer(
    layer: int,
    *,
    n_prompts: int = 32,
    jlens_rank: int = 1,
    control_rank: int = 50,
    wrong_layer_rank: int = 50,
    n_tied: int = 1,
) -> list[dict]:
    """Synthetic rows with a chosen rank profile for each variant."""
    rows: list[dict] = []
    for index in range(n_prompts):
        actual = torch.zeros(VOCAB)
        target = index % 20          # >= 8 distinct target tokens
        actual[target] = 10.0
        for variant in READOUT_VARIANTS:
            rank = {
                "j_lens": jlens_rank,
                "wrong_layer": wrong_layer_rank,
                "logit_lens": control_rank,
            }.get(variant, control_rank)
            scores = torch.zeros(VOCAB)
            # (rank - 1) tokens strictly above the target, then a tie block.
            above = [i for i in range(VOCAB) if i != target][: rank - 1]
            scores[above] = 5.0
            tied_with = [i for i in range(VOCAB) if i != target and i not in above][
                : n_tied - 1
            ]
            scores[target] = 2.0
            scores[tied_with] = 2.0
            rows.append(
                tie_aware_row(
                    sample_index=index,
                    prompt_sha=f"sha{index:03d}",
                    layer=layer,
                    variant=variant,
                    variant_logits=scores,
                    actual_logits=actual,
                )
            )
    return rows


def test_a_clean_layer_is_eligible():
    result = evaluate_layer_validity(_rows_for_layer(38), layer=38)
    assert result["eligible"] is True
    assert result["status"] == "ELIGIBLE"
    assert result["failed_checks"] == []
    assert result["gate_digest"] == LOCALIZATION_VALIDITY_GATE.digest


def test_a_layer_that_only_ties_at_the_maximum_is_refused():
    """The clause the old gate lacked: an optimistic rank of 1 earned by a wide
    tie block is not a readout."""
    rows = _rows_for_layer(32, jlens_rank=1, n_tied=40)
    result = evaluate_layer_validity(rows, layer=32)

    metrics = result["metrics"]["j_lens"]
    assert metrics["median_optimistic_rank"] == 1.0      # looks perfect
    assert metrics["tied_at_max_rate"] == 1.0
    assert metrics["unique_top1_agreement"] == 0.0
    assert result["eligible"] is False
    assert "coverage_and_nondegeneracy" in result["failed_checks"]


def test_a_layer_that_cannot_beat_the_wrong_layer_control_is_refused():
    rows = _rows_for_layer(26, jlens_rank=2, wrong_layer_rank=2)
    result = evaluate_layer_validity(rows, layer=26)
    assert result["eligible"] is False
    assert "mrr_beats_wrong_layer_by_margin" in result["failed_checks"]


def test_a_layer_that_cannot_beat_the_noise_controls_is_refused():
    rows = _rows_for_layer(20, jlens_rank=3, control_rank=3, wrong_layer_rank=90)
    result = evaluate_layer_validity(rows, layer=20)
    assert result["eligible"] is False
    assert "mrr_beats_noise_controls" in result["failed_checks"]


def test_a_layer_with_a_poor_median_rank_is_refused():
    rows = _rows_for_layer(20, jlens_rank=40, control_rank=180, wrong_layer_rank=180)
    result = evaluate_layer_validity(rows, layer=20)
    assert result["eligible"] is False
    assert "median_rank_and_top_k" in result["failed_checks"]


def test_a_short_prompt_set_is_not_a_weaker_pass_but_no_result():
    rows = _rows_for_layer(38, n_prompts=8)
    result = evaluate_layer_validity(rows, layer=38)
    assert result["eligible"] is False
    assert "coverage_and_nondegeneracy" in result["failed_checks"]


def test_degenerate_targets_are_refused():
    rows = []
    for index in range(32):
        actual = torch.zeros(VOCAB)
        actual[3] = 10.0                       # the SAME target every prompt
        for variant in READOUT_VARIANTS:
            scores = torch.zeros(VOCAB)
            scores[3] = 5.0 if variant == "j_lens" else 0.0
            if variant != "j_lens":
                scores[99] = 5.0
            rows.append(
                tie_aware_row(
                    sample_index=index,
                    prompt_sha=f"s{index}",
                    layer=38,
                    variant=variant,
                    variant_logits=scores,
                    actual_logits=actual,
                )
            )
    result = evaluate_layer_validity(rows, layer=38)
    assert result["n_distinct_target_tokens"] == 1
    assert result["eligible"] is False
    assert "coverage_and_nondegeneracy" in result["failed_checks"]


def test_one_lucky_fold_cannot_carry_a_layer():
    """Perfect on one quarter of the prompts, chance on the rest."""
    rows = []
    for index in range(32):
        lucky = fold_of(index) == 0
        rows.extend(
            _rows_for_layer(
                32,
                n_prompts=1,
                jlens_rank=1 if lucky else 60,
                control_rank=60,
                wrong_layer_rank=60,
            )
        )
    # Re-key the sample indices so the folds are the intended ones.
    for position, row in enumerate(rows):
        row["sample"] = position // len(READOUT_VARIANTS)
    result = evaluate_layer_validity(rows, layer=32)
    assert result["eligible"] is False
    assert "stable_across_heldout_subsets" in result["failed_checks"]


def test_folds_are_a_fixed_partition_not_a_resample():
    assert [fold_of(i) for i in range(8)] == [0, 1, 2, 3, 0, 1, 2, 3]
    assert fold_of(31) == 3


# -------------------------------------------------- old gate beside the new


def test_both_gates_are_computed_and_reported_for_every_layer():
    result = evaluate_layer_validity(_rows_for_layer(38), layer=38)
    assert "legacy_gate" in result
    assert result["legacy_gate"]["is_binding"] is False
    assert isinstance(result["legacy_gate"]["passed"], bool)
    assert isinstance(result["gates_agree"], bool)
    checks = {check["check"] for check in result["legacy_gate"]["checks"]}
    assert "legacy_top1_floor" in checks
    assert "legacy_top10_overlap_beats_controls" in checks


def test_the_old_gate_fails_the_tie_case_that_the_new_gate_also_fails():
    """Both gates reject a pure tie block — the new one is not simply laxer."""
    rows = _rows_for_layer(32, jlens_rank=1, n_tied=40)
    result = evaluate_layer_validity(rows, layer=32)
    assert result["eligible"] is False
    assert result["legacy_gate"]["passed"] is False
    assert "legacy_top1_floor" in result["legacy_gate"]["failed_checks"]


def test_the_legacy_gate_never_decides_eligibility():
    """Eligibility comes from the new gate's checks alone."""
    result = evaluate_layer_validity(_rows_for_layer(38), layer=38)
    assert result["eligible"] == all(check["passed"] for check in result["checks"])


def test_the_gate_text_states_the_thresholds_and_the_tie_rationale():
    text = gate_text()
    gate = LOCALIZATION_VALIDITY_GATE
    assert "MIDRANK" in text
    assert f"{gate.max_tied_at_max_rate:.2f}" in text
    assert f"{gate.min_wrong_layer_mrr_margin:.2f}" in text
    assert gate.digest in text
    # The rationale must say what was added, not merely that something changed.
    flowed = " ".join(text.split())
    assert "ADDS a blocking tied-at-maximum ceiling" in flowed
    assert "ADDS a blocking wrong-layer MRR margin" in flowed
    assert "ADDS a blocking fold-stability clause" in flowed
    assert "DROPS only the unique-top-1 floor" in flowed
    assert LEGACY_VALIDITY_GATE.name in text


def test_editing_the_gate_changes_its_digest():
    assert LayerValidityGate().digest == LOCALIZATION_VALIDITY_GATE.digest
    assert LayerValidityGate(max_median_midrank=99.0).digest != (
        LOCALIZATION_VALIDITY_GATE.digest
    )


def test_the_control_and_diagnostic_roles_are_kept_apart():
    assert set(NOISE_CONTROLS) < set(CONTROL_VARIANTS)
    assert "wrong_layer" in CONTROL_VARIANTS
    assert "wrong_layer" not in NOISE_CONTROLS
    assert "logit_lens" in DIAGNOSTIC_VARIANTS
    assert "logit_lens" not in CONTROL_VARIANTS
    assert set(READOUT_VARIANTS) == {"j_lens", *CONTROL_VARIANTS, *DIAGNOSTIC_VARIANTS}


# ------------------------------------------------- eligibility is enforced


def test_eligible_layers_and_the_causal_guard_agree():
    results = evaluate_all_layers(
        [*_rows_for_layer(38), *_rows_for_layer(20, jlens_rank=1, n_tied=40)],
        layers=(20, 38),
    )
    assert eligible_layers(results) == [38]
    assert_causally_eligible(38, results)          # does not raise

    with pytest.raises(LayerNotEligibleError) as error:
        assert_causally_eligible(20, results)
    message = str(error.value)
    assert "coverage_and_nondegeneracy" in message
    assert "skipped" in message


def test_an_unevaluated_layer_cannot_carry_a_causal_claim():
    results = evaluate_all_layers(_rows_for_layer(38), layers=(38,))
    with pytest.raises(LayerNotEligibleError, match="no lens-validity result"):
        assert_causally_eligible(26, results)


# ------------------------------------------------------- recalibration


def test_the_recalibration_plan_is_bounded_and_text_only():
    assert RECALIBRATION_PLAN["n_fitting_prompts"] in (128, 256)
    assert RECALIBRATION_PLAN["layers"] == [20, 26, 32, 38]
    assert RECALIBRATION_PLAN["modality"] == "text-only"
    assert RECALIBRATION_PLAN["multimodal_examples_used"] is False
    assert RECALIBRATION_PLAN["cross_modal_alignment"] is False
    assert RECALIBRATION_PLAN["modality_specific_lens"] is False
    assert RECALIBRATION_PLAN["calibration_uses_cat_or_toilet_targets"] is False
    assert RECALIBRATION_PLAN["heldout_independent_of_fitting"] is True
    assert "v2" in RECALIBRATION_PLAN["never_overwrite"]
    assert "v3" in RECALIBRATION_PLAN["new_artifact_dir"]


def test_overwriting_the_v2_artifact_is_refused():
    with pytest.raises(RecalibrationRefused, match="frozen v2 artifact"):
        check_recalibration_target(
            "runs/text_jlens_early_layer_recalibration_v2/artifacts/lens.validated.pt"
        )
    with pytest.raises(RecalibrationRefused):
        check_recalibration_target(
            "/content/drive/MyDrive/jacobian-lens-gemma/runs/"
            "text_jlens_early_layer_recalibration_v2/artifacts/lens.validated.pt"
        )


def test_a_new_artifact_path_is_accepted():
    record = check_recalibration_target(
        "runs/text_jlens_early_layer_recalibration_v3/artifacts/lens.validated.pt"
    )
    assert record["overwrites_v2"] is False
