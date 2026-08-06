# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the output-convergence timing audit.

The audit's one job is to be hard to fool, so most of what is checked here is a
refusal: a lens-invalid layer, a candidate set that cannot be ranked, a resume
against a different rule, a control that reproduces the primary result, a weak
readout with nothing behind it. The MOCK world exists to make each of those
reachable without Gemma and without a Drive run.
"""

import json
from pathlib import Path

import pytest
import torch

from jlens.mmpilot.convergence import (
    AMBIGUOUS,
    AUDITED_LAYERS,
    CONVERGED,
    CONVERGENCE_CRITERION,
    CONVERGENCE_PROTOCOL,
    CRITERION_TEXT,
    INCONCLUSIVE_CONVERGENCE_TIMING,
    INTERPRETATION_BOUNDARY,
    LENS_INVALID_LAYERS,
    MODALITIES,
    NOT_CONVERGED,
    PRE_CONVERGENCE_TRANSFER_SUPPORTED,
    PRIMARY_LAYER,
    PRIMARY_VARIANT,
    READOUT_FIRST_TOKEN,
    READOUT_SINGLE_TOKEN,
    TRANSFER_AT_OR_AFTER_CONVERGENCE,
    CandidateTokenizationError,
    CompletedRunModified,
    ConvergenceCriterion,
    ConvergenceFingerprint,
    ConvergenceRefused,
    ConvergenceStore,
    IncompatibleStateError,
    LensInvalidLayerError,
    NativeHead,
    assert_lens_valid_layer,
    assert_run_unchanged,
    audit_native_head,
    bootstrap_rate,
    build_population,
    classify_layer,
    clean_predictions_from_interventions,
    direct_readout_row,
    figure_convergence_versus_layer,
    head_from_model,
    image_disjoint_folds,
    layer_convergence_table,
    permuted_activation_assignment,
    permuted_token_assignment,
    protected_file_checksums,
    read_frozen_causal_evidence,
    resolve_candidate_tokens,
    shuffled_label_assignment,
    summarize_cell,
    tie_aware_ranks,
    trajectory_report,
    verify_completed_run,
    write_layer_table_csv,
)
from jlens.mmpilot.convergence_mock import (
    MOCK_INELIGIBLE_CONCEPT,
    MOCK_MODES,
    MockWorldSpec,
    build_mock_completed_run,
    build_mock_head,
    mock_candidate_token_ids,
    mock_verdict_matrix,
    run_mock_convergence_audit,
)

BOOTSTRAP = 120


@pytest.fixture(scope="module")
def audited(tmp_path_factory):
    """One full MOCK audit of the pre-convergence world, reused by many tests."""
    root = tmp_path_factory.mktemp("convergence")
    return run_mock_convergence_audit(
        root / "completed", root / "audit", bootstrap_resamples=BOOTSTRAP
    )


# ------------------------------------------------------- lens-validity refusal


def test_layer_32_is_never_interpreted():
    assert 32 in LENS_INVALID_LAYERS
    with pytest.raises(LensInvalidLayerError, match="failed the calibration"):
        assert_lens_valid_layer(32)


@pytest.mark.parametrize("layer", AUDITED_LAYERS)
def test_confirmed_layers_are_audited(layer):
    assert_lens_valid_layer(layer)


def test_a_layer_outside_the_confirmed_set_is_refused():
    with pytest.raises(LensInvalidLayerError, match="not one of the independently"):
        assert_lens_valid_layer(21)


def test_the_audit_refuses_to_run_over_a_lens_invalid_layer(tmp_path):
    from jlens.mmpilot.convergence import run_convergence_audit

    with pytest.raises(LensInvalidLayerError):
        run_convergence_audit(
            population={"units": []},
            head=None,
            tokenization={},
            head_audit={},
            integrity={},
            completed_summary={},
            store=None,
            layers=(32,),
        )


# ------------------------------------------------------ candidate tokenization


def test_single_token_candidates_give_a_complete_score():
    resolved = resolve_candidate_tokens({"cat": [7], "toilet": [9], "bus": [3]})
    assert resolved["readout_mode"] == READOUT_SINGLE_TOKEN
    assert resolved["all_candidates_single_token"] is True
    assert resolved["candidates"] == ["bus", "cat", "toilet"]
    assert "directly comparable" in resolved["scoring_note"]


def test_a_multi_token_candidate_becomes_a_labelled_first_token_diagnostic():
    resolved = resolve_candidate_tokens({"cat": [7], "toilet": [9, 11]})
    assert resolved["readout_mode"] == READOUT_FIRST_TOKEN
    assert resolved["all_candidates_single_token"] is False
    assert resolved["multi_token_candidates"] == ["toilet"]
    # The label must say plainly that this is not a sequence score.
    assert "FIRST-TOKEN-ONLY" in resolved["scoring_note"]
    assert "no row here is a sequence score" in resolved["scoring_note"]
    assert "never mixed into a readout total" in resolved["scoring_note"]


def test_candidates_sharing_a_first_token_are_refused():
    with pytest.raises(CandidateTokenizationError, match="share a first token"):
        resolve_candidate_tokens({"cat": [7], "catamaran": [7, 12]})


def test_an_empty_or_zero_token_candidate_is_refused():
    with pytest.raises(CandidateTokenizationError, match="empty"):
        resolve_candidate_tokens({})
    with pytest.raises(CandidateTokenizationError, match="zero tokens"):
        resolve_candidate_tokens({"cat": [7], "dog": []})


def test_the_readout_mode_is_stamped_on_every_row(audited):
    modes = {row["readout_mode"] for row in audited["rows"]}
    assert modes == {audited["tokenization"]["readout_mode"]}


# ------------------------------------------------ the exact Gemma convention


def _mock_gemma_model():
    """The repository's CPU mock of ``Gemma4ForConditionalGeneration``.

    Wrapped exactly as the real path wraps the real checkpoint, so the head this
    audit reads is reached through the same attributes.
    """
    from jlens.gemma4 import Gemma4LensModel

    from .mock_gemma4 import MockGemma4ForConditionalGeneration, MockTokenizer

    return Gemma4LensModel(MockGemma4ForConditionalGeneration(), MockTokenizer())


def test_the_audited_head_is_the_model_s_own_unembed():
    """The whole audit rests on this: our path and ``unembed`` must agree.

    Run against the repository's mock of the real module layout, so a change to
    the readout that diverges from the model's own output pathway fails here
    rather than on Drive.
    """
    model = _mock_gemma_model()
    head = head_from_model(model)
    report = audit_native_head(head, model=model)

    assert report["matches_model_unembed"] is True
    assert report["max_abs_difference_vs_model_unembed"] < 1e-2
    assert report["modules_called_not_reimplemented"] is True
    # The mock ships a softcap, and the audited path must apply it because the
    # model's output head does.
    assert report["final_logit_softcapping"] == 30.0
    assert report["softcap_applied"] is True
    assert "tanh" in report["readout_expression"]


def test_a_head_that_disagrees_with_the_model_stops_the_audit():
    model = _mock_gemma_model()
    head = head_from_model(model)
    # Break the head the way a hand-rolled logit lens would: drop the softcap.
    head.softcap = None
    with pytest.raises(ConvergenceRefused, match="disagrees with the model's own"):
        audit_native_head(head, model=model)


def test_the_rmsnorm_weight_convention_is_detected_not_assumed():
    """Gemma applies ``(1 + weight)``; a plain ``weight`` lens is silently wrong.

    The MOCK head implements the ``(1 + weight)`` convention, so the detector
    has to name it without being told.
    """
    head, _ = build_mock_head(MockWorldSpec())
    report = audit_native_head(head)
    assert report["norm_weight_convention"] == "rmsnorm_one_plus_weight"
    residuals = report["norm_convention_residuals"]
    assert residuals["rmsnorm_one_plus_weight"] < residuals["rmsnorm_weight"]


def test_the_softcap_cannot_change_a_ranking():
    """Monotonic by construction, so ranks agree and only values move."""
    head, _ = build_mock_head(MockWorldSpec())
    activation = torch.randn(16, generator=torch.Generator().manual_seed(3))
    capped = head.logits(activation)
    head_uncapped = NativeHead(
        final_norm=head.final_norm,
        lm_head=head.lm_head,
        softcap=None,
        d_model=head.d_model,
        vocab_size=head.vocab_size,
    )
    plain = head_uncapped.logits(activation)
    assert torch.equal(capped.argsort(), plain.argsort())
    assert not torch.allclose(capped, plain)


# ------------------------------------------------------- direct-readout scores


def _row(**overrides):
    head, directions = build_mock_head(MockWorldSpec())
    tokenization = resolve_candidate_tokens(mock_candidate_token_ids(MockWorldSpec()))
    activation = torch.zeros(16)
    activation[directions["cat"]] = 6.0
    defaults = dict(
        activation=activation,
        head=head,
        tokenization=tokenization,
        concept="cat",
        clean_prediction="cat",
        sample_id="s0",
        group_id="g0",
        image_id="i0",
        recording_id=None,
        modality="text",
        layer=35,
        split="test",
        capability_admissible=True,
        activation_checksum="sha256:x",
        head_checksum="sha256:h",
        config_hash="sha256:c",
    )
    defaults.update(overrides)
    return direct_readout_row(**defaults)


def test_a_readout_pointed_at_a_candidate_names_that_candidate():
    row = _row()
    assert row["direct_readout_prediction"] == "cat"
    assert row["target_rank"] == 1.0
    assert row["unique_top1_target"] is True
    assert row["agrees_with_clean_final_prediction_unique"] is True
    assert row["target_margin"] > 0
    assert len(row["candidate_logits"]) == len(row["candidates"]) == 6


def test_every_commissioned_field_is_recorded():
    row = _row()
    for field in (
        "sample_id",
        "group_id",
        "image_id",
        "recording_id",
        "concept",
        "capability_admissible",
        "candidate_token_ids",
        "candidate_logits",
        "candidate_log_probs",
        "direct_readout_prediction",
        "target_rank",
        "target_margin",
        "agrees_with_clean_final_prediction_argmax",
        "agrees_with_ground_truth_argmax",
        "candidate_entropy_nats",
        "top_two_margin",
        "activation_checksum",
        "head_checksum",
        "layer",
        "config_hash",
    ):
        assert field in row, field


def test_the_recording_id_defaults_to_the_group():
    """SpokenCOCO carries one recording per synchronized group."""
    assert _row(recording_id=None)["recording_id"] == "g0"
    assert _row(recording_id="rec-9")["recording_id"] == "rec-9"


def test_a_non_finite_readout_is_refused():
    with pytest.raises(ConvergenceRefused, match="non-finite"):
        _row(activation=torch.full((16,), float("nan")))


def test_a_target_outside_the_candidate_set_is_refused():
    with pytest.raises(ConvergenceRefused, match="not among the fixed candidates"):
        _row(concept="aardvark")


def test_entropy_is_bounded_by_the_uniform_distribution():
    import math

    flat = _row(activation=torch.zeros(16))
    assert flat["candidate_entropy_nats"] <= math.log(6) + 1e-6
    assert flat["candidate_entropy_normalized"] <= 1.0 + 1e-6
    peaked = _row()
    assert peaked["candidate_entropy_nats"] < flat["candidate_entropy_nats"]


# ------------------------------------------------------------ ranks and ties


def test_rank_conventions_agree_when_there_are_no_ties():
    ranks = tie_aware_ranks(torch.tensor([3.0, 1.0, 2.0]), 0)
    assert ranks["rank_optimistic"] == ranks["rank_pessimistic"] == 1.0
    assert ranks["rank_midrank"] == 1.0
    assert ranks["n_tied_at_max"] == 1


def test_a_tie_block_splits_the_three_rank_conventions():
    ranks = tie_aware_ranks(torch.tensor([2.0, 2.0, 2.0, 1.0]), 0)
    assert ranks["rank_optimistic"] == 1.0
    assert ranks["rank_pessimistic"] == 3.0
    assert ranks["rank_midrank"] == 2.0
    assert ranks["n_tied_at_max"] == 3


def test_a_tie_at_the_top_is_not_unique_top1():
    """A tie must not be scored as convergence."""
    head, directions = build_mock_head(MockWorldSpec())
    tokenization = resolve_candidate_tokens(mock_candidate_token_ids(MockWorldSpec()))
    row = direct_readout_row(
        activation=torch.zeros(16),
        head=head,
        tokenization=tokenization,
        concept="cat",
        clean_prediction="cat",
        sample_id="s",
        group_id="g",
        image_id="i",
        recording_id=None,
        modality="text",
        layer=35,
        split="test",
        capability_admissible=True,
        activation_checksum="sha256:x",
        head_checksum="sha256:h",
        config_hash="sha256:c",
    )
    if row["n_tied_at_max"] > 1:
        assert row["unique_top1_target"] is False
        assert row["agrees_with_clean_final_prediction_unique"] is False


# ------------------------------------------------------------------- controls


def test_shuffled_labels_are_a_permutation_of_the_labels():
    rows = [
        {"sample_id": f"s{i}", "concept": concept}
        for i, concept in enumerate(["cat", "cat", "toilet", "toilet"])
    ]
    shuffled = shuffled_label_assignment(rows, seed=1)
    assert sorted(shuffled.values()) == sorted(r["concept"] for r in rows)
    assert set(shuffled) == {r["sample_id"] for r in rows}


def test_shuffled_labels_are_deterministic():
    rows = [{"sample_id": f"s{i}", "concept": "cat" if i % 2 else "toilet"} for i in range(8)]
    assert shuffled_label_assignment(rows, seed=5) == shuffled_label_assignment(rows, seed=5)
    assert shuffled_label_assignment(rows, seed=5) != shuffled_label_assignment(rows, seed=6)


def test_the_token_permutation_gives_every_candidate_another_s_token():
    tokenization = resolve_candidate_tokens(mock_candidate_token_ids(MockWorldSpec()))
    permuted = permuted_token_assignment(tokenization, seed=3)
    original = tokenization["readout_token_ids"]
    assert sorted(permuted.values()) == sorted(original.values())
    # A derangement: no candidate keeps its own token, or the control is a no-op.
    assert all(permuted[name] != original[name] for name in original)


def test_the_activation_permutation_is_a_derangement():
    rows = [{"sample_id": f"s{i}"} for i in range(6)]
    partner = permuted_activation_assignment(rows, seed=2)
    assert sorted(partner.values()) == sorted(partner)
    assert all(key != value for key, value in partner.items())


def test_every_control_collapses_where_the_primary_readout_is_informative():
    """Run the converged world, where the primary result is 1.0 and real."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        result = run_mock_convergence_audit(
            Path(tmp) / "c",
            Path(tmp) / "a",
            spec=MockWorldSpec(mode="converged_early"),
            run_probe=False,
            bootstrap_resamples=BOOTSTRAP,
        )
    assert result["controls"]["all_controls_passed"] is True
    for entry in result["controls"]["per_layer"].values():
        for name, control in entry["controls"].items():
            assert control["primary_is_informative"] is True, name
            assert control["control_value"] < control["primary_value"], name


def test_a_control_at_chance_is_recorded_as_non_informative_not_scored():
    """Where the primary readout is at chance, no result rests on the cell.

    Requiring a permutation to fall *below* chance would be requiring something
    no permutation can do, so the cell is labelled rather than scored.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        result = run_mock_convergence_audit(
            Path(tmp) / "c",
            Path(tmp) / "a",
            spec=MockWorldSpec(mode="flat_weak"),
            run_probe=False,
            bootstrap_resamples=BOOTSTRAP,
        )
    controls = result["controls"]["per_layer"][str(PRIMARY_LAYER)]["controls"]
    assert controls
    for control in controls.values():
        assert control["primary_is_informative"] is False
        assert control["passed"] is True
        assert "chance rate" in control["reason"]


def test_the_wrong_layer_control_absence_is_explained_not_omitted(audited):
    note = audited["controls"]["wrong_layer_control_note"]
    assert "one final normalization module" in note
    assert "permuted-activation control" in note


# ------------------------------------------------------------------ bootstrap


def _bootstrap_rows(values, images):
    return [
        {"image_id": image, "flag": bool(value)}
        for value, image in zip(values, images, strict=True)
    ]


def test_the_image_level_bootstrap_is_deterministic():
    rows = _bootstrap_rows(
        [1, 0, 1, 1, 0, 1], ["a", "a", "b", "b", "c", "c"]
    )
    first = bootstrap_rate(rows, "flag", seed=7, resamples=200)
    second = bootstrap_rate(rows, "flag", seed=7, resamples=200)
    assert first == second
    assert first["n_units"] == 3
    assert first["low"] <= first["point"] <= first["high"]


def test_the_bootstrap_resamples_photographs_not_rows():
    """Two rows of one photograph must move together, or n is over-reported."""
    rows = _bootstrap_rows([1, 1, 0, 0], ["a", "a", "b", "b"])
    result = bootstrap_rate(rows, "flag", seed=1, resamples=200)
    assert result["n_units"] == 2
    assert result["n"] == 4
    # Resampling two units of two identical rows can only ever give 0, 0.5 or 1.
    assert result["low"] in (0.0, 0.5, 1.0)


def test_a_single_unit_yields_no_interval():
    rows = _bootstrap_rows([1, 0], ["a", "a"])
    result = bootstrap_rate(rows, "flag", seed=1, resamples=200)
    assert result["resamples"] == 0
    assert "fewer than two independent units" in result["note"]


def test_image_disjoint_folds_never_split_a_photograph():
    rows = [{"image_id": f"i{i // 2}"} for i in range(8)]
    folds = image_disjoint_folds(rows, n_folds=2)
    assert len(set(folds.values())) == 2
    assert folds == image_disjoint_folds(rows, n_folds=2)


# ------------------------------------------------------- capability filtering


def test_the_ineligible_concept_is_labelled_and_kept_out_of_every_principal_number(
    audited,
):
    population = audited["population"]
    assert population["admissible_concepts"] == ["cat", "toilet"]
    assert population["inadmissible_concepts"] == [MOCK_INELIGIBLE_CONCEPT]
    # Present in the population as a labelled diagnostic ...
    assert any(
        unit["concept"] == MOCK_INELIGIBLE_CONCEPT for unit in population["units"]
    )
    # ... and absent from every principal cell.
    for entry in audited["summary"]["measurements"]["per_layer"].values():
        for cell in entry["per_modality"].values():
            assert MOCK_INELIGIBLE_CONCEPT not in (cell.get("per_concept") or {})
        assert MOCK_INELIGIBLE_CONCEPT not in entry["pooled_principal"]["per_concept"]


def test_the_ineligible_concept_survives_as_descriptive_data(audited):
    descriptive = audited["summary"]["measurements"]["descriptive_inadmissible"]
    assert descriptive["concepts"] == [MOCK_INELIGIBLE_CONCEPT]
    assert descriptive["n_rows"] > 0
    assert "excluded from every principal number" in descriptive["note"]


def test_no_concept_is_reselected(audited):
    assert audited["population"]["reselection_performed"] is False
    assert audited["population"]["focal_concepts"] == list(MockWorldSpec().concepts)


def test_per_concept_results_precede_the_pooled_ones(audited):
    for entry in audited["summary"]["measurements"]["per_layer"].values():
        pooled = entry["pooled_principal"]
        assert set(pooled["per_concept"]) == {"cat", "toilet"}
        assert "reported after the per-concept rows" in entry["pooled_note"]


# ------------------------------------------------------ layer classification


def _layer_summary(**cell):
    base = {
        "clean_agreement_unique": 0.0,
        "clean_agreement_argmax": 0.0,
        "target_accuracy_unique": 0.0,
        "target_accuracy_argmax": 0.0,
        "median_target_rank": 4.0,
        "n": 10,
    }
    base.update(cell)
    return {
        "layer": 35,
        "per_modality": dict.fromkeys(MODALITIES, base),
        "pooled_principal": {**base, "n_distinct_predictions": 4},
    }


def test_a_layer_above_the_upper_bar_is_converged():
    result = classify_layer(
        _layer_summary(
            clean_agreement_unique=0.95,
            clean_agreement_argmax=0.95,
            target_accuracy_unique=0.95,
            target_accuracy_argmax=0.95,
            median_target_rank=1.0,
        )
    )
    assert result["classification"] == CONVERGED


def test_a_layer_below_the_lower_bar_is_not_converged():
    assert classify_layer(_layer_summary())["classification"] == NOT_CONVERGED


def test_the_gap_between_the_bars_is_ambiguous_not_a_weak_conclusion():
    result = classify_layer(
        _layer_summary(
            clean_agreement_unique=0.70,
            clean_agreement_argmax=0.70,
            target_accuracy_unique=0.70,
            target_accuracy_argmax=0.70,
            median_target_rank=1.0,
        )
    )
    assert result["classification"] == AMBIGUOUS


def test_one_converged_modality_does_not_make_a_converged_layer():
    """Stability across text, image and spoken audio is required, not averaged."""
    strong = {
        "clean_agreement_unique": 1.0,
        "clean_agreement_argmax": 1.0,
        "target_accuracy_unique": 1.0,
        "target_accuracy_argmax": 1.0,
        "median_target_rank": 1.0,
        "n": 10,
    }
    weak = {
        "clean_agreement_unique": 0.1,
        "clean_agreement_argmax": 0.1,
        "target_accuracy_unique": 0.1,
        "target_accuracy_argmax": 0.1,
        "median_target_rank": 4.0,
        "n": 10,
    }
    summary = {
        "layer": 35,
        "per_modality": {"text": strong, "image": strong, "spoken_audio": weak},
        "pooled_principal": {**strong, "n_distinct_predictions": 4},
    }
    assert classify_layer(summary)["classification"] == AMBIGUOUS


def test_an_undersized_cell_cannot_be_classified_either_way():
    result = classify_layer(
        _layer_summary(
            n=2,
            clean_agreement_unique=1.0,
            clean_agreement_argmax=1.0,
            target_accuracy_unique=1.0,
            target_accuracy_argmax=1.0,
            median_target_rank=1.0,
        )
    )
    assert result["classification"] == AMBIGUOUS
    assert result["undersized_cells"]


def test_the_criterion_bars_leave_a_real_gap():
    criterion = CONVERGENCE_CRITERION
    assert criterion.not_converged_max_clean_agreement < criterion.converged_min_clean_agreement
    assert criterion.not_converged_max_target_accuracy < criterion.converged_min_target_accuracy
    assert criterion.not_converged_min_median_rank > criterion.converged_max_median_rank


def test_the_criterion_digest_moves_when_a_threshold_moves():
    other = ConvergenceCriterion(converged_min_clean_agreement=0.85)
    assert other.digest != CONVERGENCE_CRITERION.digest


# ------------------------------------------------------------------ trajectory


def _classification(layer, agreement, low, high):
    return {
        "layer": layer,
        "classification": NOT_CONVERGED,
        "pooled_clean_agreement_unique": agreement,
        "pooled_bootstrap": {"low": low, "high": high},
    }


def test_a_clearly_more_converged_later_layer_is_recognised():
    report = trajectory_report(
        {
            35: _classification(35, 0.15, 0.05, 0.25),
            38: _classification(38, 0.60, 0.50, 0.70),
            40: _classification(40, 0.95, 0.90, 1.0),
        }
    )
    assert report["any_later_layer_clearly_more_converged"] is True
    assert report["monotone_within_tolerance"] is True


def test_overlapping_intervals_are_not_a_clear_separation():
    report = trajectory_report(
        {
            35: _classification(35, 0.20, 0.05, 0.65),
            38: _classification(38, 0.45, 0.30, 0.70),
            40: _classification(40, 0.50, 0.35, 0.75),
        }
    )
    assert report["any_later_layer_clearly_more_converged"] is False


def test_a_falling_trajectory_has_no_direction_to_read():
    report = trajectory_report(
        {
            35: _classification(35, 0.80, 0.70, 0.90),
            38: _classification(38, 0.20, 0.10, 0.30),
            40: _classification(40, 0.85, 0.75, 0.95),
        }
    )
    assert report["monotone_within_tolerance"] is False
    assert report["non_monotonic_drops"]


# ------------------------------------------------------------- the layer table


def test_the_layer_table_carries_every_commissioned_column(audited):
    for row in audited["table"]:
        for column in (
            "layer",
            "lens_validity_gate_passed",
            "convergence_classification",
            "clean_agreement_unique",
            "median_target_rank",
            "causal_transfer_verdict",
            "causal_transfer_supported",
            "causal_concepts_supporting",
            "causal_audio_pairs_supporting",
            "control_gaps",
            "activation_norm_ratios",
            "n_distinct_target_images",
            "n_distinct_recordings",
        ):
            assert column in row, column
        for modality in MODALITIES:
            assert f"clean_agreement_unique_{modality}" in row


def test_the_layer_table_keeps_the_three_layers_apart(audited):
    assert [row["layer"] for row in audited["table"]] == list(AUDITED_LAYERS)


def test_the_causal_evidence_is_read_not_recomputed(audited):
    assert audited["causal"]["recomputed"] is False
    assert audited["causal"]["read_only"] is True
    assert audited["provenance"]["causal_evidence_recomputed"] is False


def test_frozen_causal_evidence_reads_both_verdict_locations():
    summary = {
        "verdicts": {
            "C_primary_causal": {
                "layer": 35,
                "verdict": "SUPPORTED",
                "cells": [],
                "audio_cells_supporting_a_claim": [
                    {"concept": "cat", "pair": "text->spoken_audio"}
                ],
            },
            "D_replication": {
                "per_layer": {
                    "38": {"layer": 38, "verdict": "SUPPORTED", "cells": []},
                }
            },
        }
    }
    evidence = read_frozen_causal_evidence(summary)
    assert evidence["per_layer"][35]["source"] == "C_primary_causal"
    assert evidence["per_layer"][38]["source"] == "D_replication.per_layer"
    assert evidence["per_layer"][40]["verdict"] == "NOT_EVALUATED"
    assert evidence["per_layer"][40]["supported"] is False


def test_the_layer_table_writes_a_stable_csv(audited, tmp_path):
    path = write_layer_table_csv(audited["table"], tmp_path / "table.csv")
    first = path.read_text(encoding="utf-8")
    write_layer_table_csv(audited["table"], path)
    assert path.read_text(encoding="utf-8") == first
    assert first.splitlines()[0].startswith("layer,")
    assert len(first.strip().splitlines()) == len(AUDITED_LAYERS) + 1


# ------------------------------------------------------------------- verdicts


@pytest.fixture(scope="module")
def verdict_matrix(tmp_path_factory):
    return mock_verdict_matrix(
        tmp_path_factory.mktemp("matrix"), bootstrap_resamples=BOOTSTRAP
    )


def test_every_world_mode_is_exercised(verdict_matrix):
    assert set(MOCK_MODES).issubset(verdict_matrix)


def test_transfer_before_convergence_is_supported_when_everything_holds(verdict_matrix):
    assert verdict_matrix["pre_convergence"] == PRE_CONVERGENCE_TRANSFER_SUPPORTED


def test_a_converged_layer_35_yields_the_unfavourable_verdict(verdict_matrix):
    assert verdict_matrix["converged_early"] == TRANSFER_AT_OR_AFTER_CONVERGENCE


def test_the_ambiguous_band_yields_inconclusive(verdict_matrix):
    assert verdict_matrix["ambiguous"] == INCONCLUSIVE_CONVERGENCE_TIMING


def test_a_weak_readout_alone_cannot_support_the_claim(verdict_matrix):
    """The failure mode this whole design exists to prevent.

    Every layer is weak, so layer 35 *is* NOT_CONVERGED — and the verdict must
    still refuse, because nothing shows the answer arriving later.
    """
    assert verdict_matrix["flat_weak"] == INCONCLUSIVE_CONVERGENCE_TIMING


def test_a_failed_readout_is_not_a_fact_about_the_representation(verdict_matrix):
    assert verdict_matrix["degenerate"] == INCONCLUSIVE_CONVERGENCE_TIMING


def test_without_the_frozen_causal_result_there_is_no_claim(verdict_matrix):
    assert (
        verdict_matrix["pre_convergence_without_causal_support"]
        == INCONCLUSIVE_CONVERGENCE_TIMING
    )


def test_the_degenerate_world_names_the_failed_readout(tmp_path):
    result = run_mock_convergence_audit(
        tmp_path / "c",
        tmp_path / "a",
        spec=MockWorldSpec(mode="degenerate"),
        run_probe=False,
        bootstrap_resamples=BOOTSTRAP,
    )
    assert "readout_not_degenerate" in result["verdict"]["failed_checks"]


def test_the_converged_branch_is_checked_before_anything_can_mask_it(tmp_path):
    """Even with the causal evidence removed, a converged L35 says so."""
    result = run_mock_convergence_audit(
        tmp_path / "c",
        tmp_path / "a",
        spec=MockWorldSpec(mode="converged_early", causal_supported=False),
        run_probe=False,
        bootstrap_resamples=BOOTSTRAP,
    )
    assert result["verdict"]["verdict"] == TRANSFER_AT_OR_AFTER_CONVERGENCE


def test_the_verdict_states_the_interpretation_boundary(audited):
    verdict = audited["verdict"]
    assert verdict["interpretation_boundary"] == INTERPRETATION_BOUNDARY
    assert "NOT proof" in verdict["failure_of_direct_readout_is_not_proof"]
    assert "pre-linguistic" not in verdict["rationale"].lower()
    assert "language-free" not in verdict["rationale"].lower()


def test_the_criterion_text_forbids_the_overclaim():
    assert "pre-linguistic" in CRITERION_TEXT  # only as a prohibition
    assert "never 'pre-linguistic'" in CRITERION_TEXT
    assert "before native direct-readout convergence" in CRITERION_TEXT


def test_the_report_never_claims_absence_of_linguistic_information(audited):
    report = audited["report"]
    assert "does **not** establish the absence of linguistic information" in report
    lowered = report.lower()
    assert "pre-linguistic" not in lowered.replace("never 'pre-linguistic'", "")


# --------------------------------------------------------------- sensitivity


def test_sensitivity_reports_alternatives_without_moving_the_primary_rule(audited):
    sensitivity = audited["sensitivity"]
    assert sensitivity["primary_rule_unchanged"] is True
    assert len(sensitivity["variants"]) == 3
    digests = {entry["criterion_digest"] for entry in sensitivity["variants"]}
    assert audited["criterion"].digest not in digests
    assert audited["verdict"]["criterion_digest"] == audited["criterion"].digest


# --------------------------------------------------------- secondary probe


def test_the_probe_is_secondary_and_decides_nothing(audited):
    probe = audited["probe"]
    assert probe["ran"] is True
    assert probe["is_secondary_diagnostic"] is True
    assert probe["determines_verdict"] is False
    assert probe["image_disjoint"] is True
    assert INTERPRETATION_BOUNDARY in probe["caveat"]
    assert "probe" not in " ".join(audited["verdict"]["failed_checks"])


# ----------------------------------------------------------- integrity checks


def test_the_completed_run_verifies_against_its_own_pins(tmp_path):
    built = build_mock_completed_run(tmp_path / "run")
    integrity = verify_completed_run(
        run_dir=tmp_path / "run",
        fingerprint_payload=json.loads(
            (tmp_path / "run" / "fingerprint.json").read_text(encoding="utf-8")
        ),
        summary=built["summary"],
        **built["expectations"],
    )
    assert integrity["passed"] is True
    assert all(check["passed"] for check in integrity["checks"])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("expected_model_revision", "d" * 40, "model_revision"),
        ("expected_processor_revision", "e" * 40, "processor_revision"),
        ("expected_audio_protocol_version", "other.v9", "audio_protocol_version"),
        ("expected_combined_lens_checksum", "sha256:wrong", "combined_lens_checksum"),
        ("expected_fingerprint_digest", "sha256:wrong", "run_fingerprint_digest"),
    ],
)
def test_a_mismatched_pin_refuses_the_audit(tmp_path, field, value, message):
    built = build_mock_completed_run(tmp_path / "run")
    expectations = dict(built["expectations"])
    expectations[field] = value
    with pytest.raises(ConvergenceRefused, match=message):
        verify_completed_run(
            run_dir=tmp_path / "run",
            fingerprint_payload=json.loads(
                (tmp_path / "run" / "fingerprint.json").read_text(encoding="utf-8")
            ),
            summary=built["summary"],
            **expectations,
        )


def test_a_mismatched_lens_checksum_refuses_the_audit(tmp_path):
    built = build_mock_completed_run(tmp_path / "run")
    expectations = dict(built["expectations"])
    expectations["expected_lens_checksums"] = {
        **expectations["expected_lens_checksums"],
        35: "sha256:not-the-published-lens",
    }
    with pytest.raises(ConvergenceRefused, match="lens_checksum_L35"):
        verify_completed_run(
            run_dir=tmp_path / "run",
            fingerprint_payload=json.loads(
                (tmp_path / "run" / "fingerprint.json").read_text(encoding="utf-8")
            ),
            summary=built["summary"],
            **expectations,
        )


def test_incomplete_verdict_state_is_refused_not_patched(tmp_path):
    built = build_mock_completed_run(tmp_path / "run")
    summary = {**built["summary"], "verdicts": {}}
    with pytest.raises(ConvergenceRefused, match="capability_filtered_verdicts_present"):
        verify_completed_run(
            run_dir=tmp_path / "run",
            fingerprint_payload=json.loads(
                (tmp_path / "run" / "fingerprint.json").read_text(encoding="utf-8")
            ),
            summary=summary,
            **built["expectations"],
        )


def test_auditing_a_lens_invalid_layer_is_refused_at_verification(tmp_path):
    built = build_mock_completed_run(tmp_path / "run")
    with pytest.raises(LensInvalidLayerError):
        verify_completed_run(
            run_dir=tmp_path / "run",
            fingerprint_payload=json.loads(
                (tmp_path / "run" / "fingerprint.json").read_text(encoding="utf-8")
            ),
            summary=built["summary"],
            layers=(32,),
            **built["expectations"],
        )


# -------------------------------------------------------------- immutability


def test_the_completed_run_is_byte_identical_after_the_audit(audited):
    assert audited["immutability"]["unchanged"] is True
    assert audited["immutability"]["checked_files"]


def test_a_modified_protected_file_is_detected(tmp_path):
    built = build_mock_completed_run(tmp_path / "run")
    before = protected_file_checksums(tmp_path / "run")
    (tmp_path / "run" / "native_audio_transfer_report.md").write_text(
        "tampered", encoding="utf-8"
    )
    with pytest.raises(CompletedRunModified, match="changed during the audit"):
        assert_run_unchanged(before, protected_file_checksums(tmp_path / "run"))
    assert built["run_dir"]


def test_a_file_that_appears_counts_as_a_change(tmp_path):
    build_mock_completed_run(tmp_path / "run")
    (tmp_path / "run" / "run_manifest.json").unlink()
    before = protected_file_checksums(tmp_path / "run")
    (tmp_path / "run" / "run_manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(CompletedRunModified):
        assert_run_unchanged(before, protected_file_checksums(tmp_path / "run"))


def test_the_audit_writes_nothing_into_the_completed_run(audited):
    entries = sorted(Path(audited["completed_run_dir"]).iterdir())
    names = {path.name for path in entries}
    assert "output_convergence_report.md" not in names
    assert "readout_units" not in names
    assert not any(name.startswith("output_convergence") for name in names)


# ------------------------------------------------------------- store / resume


def _fingerprint(**overrides):
    base = dict(
        protocol=CONVERGENCE_PROTOCOL,
        completed_run_fingerprint_digest="sha256:run",
        completed_run_dir="/runs/x",
        model_repo_id="repo",
        model_revision="a" * 40,
        processor_revision="b" * 40,
        layers=(35, 38, 40),
        candidate_digest="sha256:candidates",
        readout_mode=READOUT_SINGLE_TOKEN,
        head_checksum="sha256:head",
        criterion_digest=CONVERGENCE_CRITERION.digest,
        code_version="abc123",
    )
    base.update(overrides)
    return ConvergenceFingerprint(**base)


def test_a_matching_fingerprint_resumes(tmp_path):
    store = ConvergenceStore(tmp_path / "audit", _fingerprint())
    assert store.open() == "starting"
    store.save("k", {"a": 1})
    again = ConvergenceStore(tmp_path / "audit", _fingerprint())
    assert again.open() == "resuming"
    assert again.load("k") == {"a": 1}


@pytest.mark.parametrize(
    "change",
    [
        {"criterion_digest": "sha256:different-rule"},
        {"candidate_digest": "sha256:different-candidates"},
        {"model_revision": "f" * 40},
        {"readout_mode": READOUT_FIRST_TOKEN},
        {"head_checksum": "sha256:different-head"},
        {"layers": (35, 38)},
        {"completed_run_fingerprint_digest": "sha256:another-run"},
    ],
)
def test_an_incompatible_resume_is_refused(tmp_path, change):
    ConvergenceStore(tmp_path / "audit", _fingerprint()).open()
    store = ConvergenceStore(tmp_path / "audit", _fingerprint(**change))
    with pytest.raises(IncompatibleStateError, match="different"):
        store.open()


def test_the_refusal_names_the_field_that_moved(tmp_path):
    ConvergenceStore(tmp_path / "audit", _fingerprint()).open()
    store = ConvergenceStore(tmp_path / "audit", _fingerprint(code_version="zzz"))
    with pytest.raises(IncompatibleStateError, match="code_version"):
        store.open()


def test_a_torn_unit_is_treated_as_missing(tmp_path):
    store = ConvergenceStore(tmp_path / "audit", _fingerprint())
    store.open()
    path = store.save("k", {"a": 1})
    path.write_text('{"schema": "x", "payload"', encoding="utf-8")
    assert store.load("k") is None
    assert store.invalid_units


def test_an_edited_payload_fails_its_own_checksum(tmp_path):
    store = ConvergenceStore(tmp_path / "audit", _fingerprint())
    store.open()
    path = store.save("k", {"a": 1})
    record = json.loads(path.read_text(encoding="utf-8"))
    record["payload"]["a"] = 99
    path.write_text(json.dumps(record), encoding="utf-8")
    assert store.load("k") is None


def test_a_second_audit_reuses_every_row(tmp_path):
    first = run_mock_convergence_audit(
        tmp_path / "c", tmp_path / "a", run_probe=False, bootstrap_resamples=BOOTSTRAP
    )
    second = run_mock_convergence_audit(
        tmp_path / "c", tmp_path / "a", run_probe=False, bootstrap_resamples=BOOTSTRAP
    )
    assert first["units_reused"] == 0
    assert second["units_computed"] == 0
    assert second["units_reused"] == first["units_computed"]
    assert second["verdict"]["verdict"] == first["verdict"]["verdict"]
    assert second["store"].status_report()["status"] == "resuming"


# ------------------------------------------------------------------ artifacts


def test_every_commissioned_artifact_is_written(audited):
    names = {Path(path).name for path in audited["artifacts"]}
    assert {
        "output_convergence_report.md",
        "output_convergence_summary.json",
        "per_sample_direct_readout.jsonl",
        "layer_convergence_table.csv",
        "provenance.json",
        "checksums.json",
        "figure_convergence_versus_layer.svg",
        "figure_causal_versus_convergence.svg",
        "figure_per_modality_trajectories.svg",
    } <= names


def test_the_jsonl_holds_one_row_per_scored_unit(audited):
    path = audited["store"].root / "per_sample_direct_readout.jsonl"
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(audited["rows"])
    parsed = json.loads(lines[0])
    assert parsed["protocol"] == CONVERGENCE_PROTOCOL
    assert parsed["interpretation_boundary"] == INTERPRETATION_BOUNDARY


def test_provenance_records_that_no_forward_pass_ran(audited):
    provenance = audited["provenance"]
    assert provenance["model_forwards_executed"] == 0
    assert provenance["completed_run_read_only"] is True
    assert provenance["criterion_digest"] == audited["criterion"].digest
    assert provenance["native_head_audit"]["modules_called_not_reimplemented"] is True


def test_checksums_cover_the_audit_and_the_protected_run_files(audited):
    checksums = audited["checksums"]
    assert "output_convergence_report.md" in checksums["audit_artifacts"]
    assert checksums["completed_run_protected_files"]
    assert checksums["criterion_digest"] == audited["criterion"].digest


def test_the_figure_is_deterministic_svg(audited):
    first = figure_convergence_versus_layer(audited["table"])
    assert first == figure_convergence_versus_layer(audited["table"])
    assert first.startswith("<svg")
    assert "converged bar" in first and "not-converged bar" in first


# ---------------------------------------------------------------- population


def test_clean_predictions_come_from_the_stored_zero_alpha_units(audited):
    units = audited["population"]["units"]
    assert audited["population"]["n_with_clean_reference"] == len(units)


def test_disagreeing_clean_predictions_are_refused():
    with pytest.raises(ConvergenceRefused, match="disagree about the clean"):
        clean_predictions_from_interventions(
            [
                {"sample_id": "s0", "clean_prediction": "cat"},
                {"sample_id": "s0", "clean_prediction": "dog"},
            ]
        )


def test_a_sample_without_a_clean_reference_keeps_its_other_metrics():
    capability = {
        "per_concept": {
            "cat": dict.fromkeys(
                MODALITIES, {"n": 8, "n_correct": 8, "accuracy": 1.0, "passed": True}
            )
        },
        "threshold": 0.7,
    }
    population = build_population(
        activations=[
            {
                "sample_id": "s0",
                "group_id": "g0",
                "image_id": "i0",
                "concept": "cat",
                "modality": "text",
                "layer": 35,
                "split": "test",
                "activation": [0.0] * 4,
                "activation_checksum": "sha256:a",
            }
        ],
        clean_predictions={},
        capability=capability,
        focal_concepts=["cat"],
    )
    assert population["n_units"] == 1
    assert population["n_with_clean_reference"] == 0
    assert population["units"][0]["clean_final_prediction"] is None


def test_summarize_cell_ignores_a_missing_clean_reference():
    rows = [
        {
            "image_id": "i0",
            "recording_id": "g0",
            "rank_midrank": 1.0,
            "target_margin": 1.0,
            "candidate_entropy_nats": 0.5,
            "top_two_margin": 1.0,
            "direct_readout_prediction": "cat",
            "n_tied_at_max": 1,
            "readout_mode": READOUT_SINGLE_TOKEN,
            "agrees_with_clean_final_prediction_unique": None,
            "agrees_with_clean_final_prediction_argmax": None,
            "agrees_with_ground_truth_unique": True,
            "agrees_with_ground_truth_argmax": True,
            "unique_top1_target": True,
        }
    ]
    summary = summarize_cell(rows)
    assert summary["clean_agreement_unique"] is None
    assert summary["target_accuracy_argmax"] == 1.0
    assert summary["n_with_clean_reference"] == 0


def test_the_population_covers_the_three_modalities_at_every_layer(audited):
    seen = {
        (unit["layer"], unit["modality"]) for unit in audited["population"]["units"]
    }
    assert seen == {(layer, m) for layer in AUDITED_LAYERS for m in MODALITIES}


def test_the_primary_variant_rows_match_the_population(audited):
    primary = [row for row in audited["rows"] if row["variant"] == PRIMARY_VARIANT]
    assert len(primary) == audited["population"]["n_units"]


def test_the_layer_table_is_built_from_the_frozen_evidence(audited):
    table = layer_convergence_table(
        audited["classifications"],
        audited["causal"],
        summary=audited["summary"]["measurements"],
    )
    assert table == audited["table"]
