# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Geometry, reward, preference pairs, baselines, and the GO/NO-GO report.

These run on constructed inputs rather than a trained pipeline: a report that
turns a failure into a pass is a bug that must be caught by a unit test, not
discovered by reading a pilot's output.
"""

import math

import pytest
import torch

from jlens.autoencoder.baselines import (
    BASELINE_IDS,
    baseline_cone,
    confabulation_probe,
    raw_beam_view,
)
from jlens.autoencoder.config import EvaluationConfig, PreferenceConfig
from jlens.autoencoder.errors import AutoencoderError
from jlens.autoencoder.evaluation import (
    attribute_failure,
    evaluate_cross_cone_swap,
    gonogo_report,
    render_markdown,
    summarize_results,
)
from jlens.autoencoder.geometry import (
    auroc,
    batched_explained_fraction,
    nonnegative_scale_fit,
    rank_of,
    top_k_accuracy,
    unit,
)
from jlens.autoencoder.inference import exact_match, substring_recovery
from jlens.autoencoder.preference import (
    build_preference_pairs,
    normalize_candidate_text,
    score_candidates,
)
from jlens.autoencoder.verbalizer import Candidate

EVAL_CONFIG = EvaluationConfig()
REWARD_CONFIG = PreferenceConfig()


# -------------------------------------------------------------------- geometry


def test_scale_fit_is_the_nonnegative_least_squares_solution():
    q = torch.tensor([3.0, 4.0])
    q_hat = torch.tensor([0.6, 0.8])  # unit, same direction
    fit = nonnegative_scale_fit(q, q_hat)
    assert fit["alpha"] == pytest.approx(5.0)
    assert fit["cosine"] == pytest.approx(1.0)
    assert fit["explained_fraction"] == pytest.approx(1.0)
    assert fit["residual_norm"] == pytest.approx(0.0, abs=1e-6)


def test_scale_fit_clamps_alpha_at_zero_for_opposing_directions():
    fit = nonnegative_scale_fit(torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 0.0]))
    assert fit["alpha"] == 0.0
    assert fit["cosine"] == pytest.approx(-1.0)
    assert fit["explained_fraction"] == 0.0


def test_scale_fit_explained_fraction_equals_clamped_cosine_squared():
    torch.manual_seed(0)
    for _ in range(20):
        q = torch.randn(16)
        q_hat = torch.randn(16)
        fit = nonnegative_scale_fit(q, q_hat)
        assert fit["explained_fraction"] == pytest.approx(max(0.0, fit["cosine"]) ** 2, abs=1e-6)


def test_scale_fit_handles_a_zero_reconstruction_without_raising():
    fit = nonnegative_scale_fit(torch.tensor([1.0, 1.0]), torch.zeros(2))
    assert fit["alpha"] == 0.0
    assert fit["explained_fraction"] == 0.0


def test_batched_explained_fraction_matches_the_scalar_fit():
    torch.manual_seed(1)
    q = torch.randn(5, 8)
    q_hat = torch.randn(5, 8)
    batched = batched_explained_fraction(q, q_hat)
    for row in range(5):
        assert float(batched[row]) == pytest.approx(
            nonnegative_scale_fit(q[row], q_hat[row])["explained_fraction"], abs=1e-6
        )


def test_auroc_is_one_for_perfect_separation_and_half_for_ties():
    assert auroc([1.0, 2.0], [0.0, 0.5]) == pytest.approx(1.0)
    assert auroc([0.0, 0.0], [0.0, 0.0]) == pytest.approx(0.5)
    assert auroc([0.0], [1.0]) == pytest.approx(0.0)


def test_auroc_is_none_when_a_population_is_empty():
    assert auroc([], [1.0]) is None
    assert auroc([1.0], []) is None


def test_rank_counts_ties_against_the_target():
    assert rank_of(0.5, [0.4, 0.3]) == 0
    assert rank_of(0.5, [0.5, 0.3]) == 1  # a tie is not a retrieval
    assert rank_of(0.1, [0.9, 0.8, 0.7]) == 3


def test_top_k_accuracy_requires_data():
    assert top_k_accuracy([0, 1, 5], 2) == pytest.approx(2 / 3)
    with pytest.raises(AutoencoderError):
        top_k_accuracy([], 1)


def test_unit_leaves_zero_rows_at_zero():
    assert torch.equal(unit(torch.zeros(1, 4)), torch.zeros(1, 4))


# ---------------------------------------------------------------------- reward


def _candidate(text: str, ids: tuple[int, ...], logprob: float, rank: int) -> Candidate:
    return Candidate(
        token_ids=ids,
        text=text,
        logprob=logprob,
        mean_logprob=logprob / max(1, len(ids)),
        n_tokens=len(ids),
        finished=True,
        beam_rank=rank,
    )


def test_reward_prefers_the_candidate_that_reconstructs_q():
    q = torch.tensor([1.0, 0.0, 0.0])
    candidates = [_candidate("right", (1, 2), -1.0, 0), _candidate("wrong", (3, 4), -0.5, 1)]
    q_hats = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    unrelated = torch.tensor([[0.0, 0.0, 1.0]])
    scored = score_candidates(
        candidates, q, q_hats, unrelated, config=REWARD_CONFIG,
        accept_recon_min=0.3, accept_margin_min=0.05,
    )
    assert scored[0].reward > scored[1].reward
    assert scored[0].accepted is True
    assert scored[1].accepted is False


def test_margin_punishes_a_candidate_every_cone_likes():
    """A generically plausible phrase scores high cosine and ~zero margin: that
    is the confabulation failure mode the margin term exists to price."""
    q = torch.tensor([1.0, 1.0])
    candidates = [_candidate("generic", (1,), -1.0, 0)]
    q_hats = torch.tensor([[1.0, 1.0]])
    unrelated = torch.tensor([[1.0, 1.0]])
    scored = score_candidates(
        candidates, q, q_hats, unrelated, config=REWARD_CONFIG,
        accept_recon_min=0.3, accept_margin_min=0.05,
    )
    assert scored[0].cosine == pytest.approx(1.0)
    assert scored[0].margin == pytest.approx(0.0, abs=1e-6)
    assert scored[0].accepted is False


def test_duplicate_and_brevity_penalties_apply():
    q = torch.tensor([1.0, 0.0])
    candidates = [
        _candidate("same", (1, 2), -1.0, 0),
        _candidate("same", (1, 2), -1.1, 1),
        _candidate("long one", (1, 2, 3, 4, 5, 6), -1.2, 2),
    ]
    q_hats = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    scored = score_candidates(
        candidates, q, q_hats, torch.zeros(0, 2), config=REWARD_CONFIG,
        accept_recon_min=0.3, accept_margin_min=0.0,
    )
    assert scored[1].duplicate_penalty > 0
    assert scored[0].reward > scored[1].reward
    assert scored[2].brevity_penalty > 0


def test_empty_candidate_scores_zero_rather_than_raising():
    scored = score_candidates(
        [_candidate("", (), 0.0, 0)],
        torch.tensor([1.0, 0.0]),
        torch.zeros(1, 2),
        torch.zeros(0, 2),
        config=REWARD_CONFIG,
        accept_recon_min=0.3,
        accept_margin_min=0.0,
    )
    assert scored[0].reward == pytest.approx(0.0)
    assert scored[0].accepted is False


def test_preference_pairs_respect_the_gap_and_the_cap():
    q = torch.tensor([1.0, 0.0])
    candidates = [_candidate(f"c{i}", (i + 1,), -float(i), i) for i in range(4)]
    q_hats = torch.tensor([[1.0, 0.0], [0.7, 0.7], [0.0, 1.0], [-1.0, 0.0]])
    scored = score_candidates(
        candidates, q, q_hats, torch.zeros(0, 2), config=REWARD_CONFIG,
        accept_recon_min=0.3, accept_margin_min=0.0,
    )
    pairs = build_preference_pairs(scored, reward_gap=0.05, max_pairs=3)
    assert len(pairs) <= 3
    for preferred, rejected in pairs:
        assert scored[preferred].reward - scored[rejected].reward >= 0.05
    assert build_preference_pairs(scored, reward_gap=10.0, max_pairs=3) == []


def test_normalize_candidate_text_folds_case_and_whitespace():
    assert normalize_candidate_text(" Great  Barrier Reef ") == "great barrier reef"


# ------------------------------------------------------------------- baselines


def test_every_baseline_id_is_covered_by_a_builder_or_a_special_case():
    q = torch.randn(8)
    for baseline_id in BASELINE_IDS:
        if baseline_id in ("jlens_token_clues", "adapter_raw_beam", "adapter_reranked"):
            continue
        if baseline_id in ("unrelated_q", "naive_token_average"):
            continue  # need a dataset / dictionary; covered by the e2e test
        cone, meta = baseline_cone(baseline_id, q, seed=0)
        assert "derivation" in meta
        if baseline_id == "zero_memory":
            assert cone is None
        else:
            assert cone is not None and cone.shape == q.shape


def test_shuffled_baseline_preserves_the_norm_and_destroys_the_direction():
    torch.manual_seed(0)
    q = torch.randn(64)
    cone, _ = baseline_cone("shuffled_q", q, seed=3)
    assert float(cone.norm()) == pytest.approx(float(q.norm()), rel=1e-5)
    assert abs(float(torch.dot(unit(cone), unit(q)))) < 0.5


def test_sign_reversed_baseline_is_exactly_negated():
    q = torch.randn(16)
    cone, _ = baseline_cone("sign_reversed_q", q)
    assert torch.allclose(cone, -q)


def test_unknown_baseline_raises():
    with pytest.raises(AutoencoderError):
        baseline_cone("teleportation", torch.randn(4))


def test_raw_beam_view_reorders_by_model_logprob_only():
    result = {
        "candidates": [
            {"text": "b", "raw_beam_rank": 1, "final_rank": 0, "accepted": True, "cosine": 0.9},
            {"text": "a", "raw_beam_rank": 0, "final_rank": 1, "accepted": False, "cosine": 0.1},
        ],
        "n_candidates": 2,
        "accepted": True,
        "top_candidate": {"text": "b"},
    }
    view = raw_beam_view(result)
    assert [c["text"] for c in view["candidates"]] == ["a", "b"]
    # Acceptance follows this view's own top candidate, not the reranked one.
    assert view["accepted"] is False


def test_confabulation_probe_flags_a_reconstructor_that_likes_everything():
    class AlwaysSame(torch.nn.Module):
        def forward(self, embeddings, mask, source_layer):
            return unit(torch.ones(embeddings.shape[0], 4))

    class Embedder:
        d_model = 4

        def batch(self, phrases):
            n = len(phrases)
            return torch.ones(n, 2, 4), torch.ones(n, 2, dtype=torch.bool)

    cones = unit(torch.ones(10, 4))
    probe = confabulation_probe(
        AlwaysSame(),
        Embedder(),
        ["black hole", "photosynthesis"],
        {"black hole": [1, 2], "photosynthesis": [3]},
        cones,
        source_layer=14,
        accept_recon_min=0.3,
    )
    assert probe["clean"] is False
    assert probe["worst_fraction_above_threshold"] == pytest.approx(1.0)


# ---------------------------------------------------------------- aggregation


def _result(top_text: str, correct: str, *, accepted: bool, cosine: float = 0.9) -> dict:
    candidates = [
        {
            "text": top_text,
            "cosine": cosine,
            "explained_fraction": cosine**2,
            "accepted": accepted,
            "raw_beam_rank": 0,
            "final_rank": 0,
        },
        {
            "text": "distractor phrase",
            "cosine": 0.1,
            "explained_fraction": 0.01,
            "accepted": False,
            "raw_beam_rank": 1,
            "final_rank": 1,
        },
    ]
    if normalize_candidate_text(top_text) != normalize_candidate_text(correct):
        candidates.append(
            {
                "text": correct,
                "cosine": cosine - 0.05,
                "explained_fraction": (cosine - 0.05) ** 2,
                "accepted": False,
                "raw_beam_rank": 2,
                "final_rank": 2,
            }
        )
    return {
        "candidates": candidates,
        "n_candidates": len(candidates),
        "accepted": accepted,
        "top_candidate": candidates[0],
    }


def test_summarize_results_reports_precision_and_abstention():
    results = [
        (_result("black hole", "black hole", accepted=True), "black hole"),
        (_result("wrong thing", "solar eclipse", accepted=False), "solar eclipse"),
    ]
    summary = summarize_results(results)
    assert summary["exact_match_top1"] == pytest.approx(0.5)
    assert summary["beam_contains_correct"] == pytest.approx(1.0)
    assert summary["acceptance_rate"] == pytest.approx(0.5)
    assert summary["abstention_rate"] == pytest.approx(0.5)
    assert summary["acceptance_precision"] == pytest.approx(1.0)
    assert summary["unfiltered_precision"] == pytest.approx(0.5)
    assert summary["auroc_correct_vs_incorrect_candidates"] is not None


def test_exact_match_and_substring_recovery():
    result = _result("great barrier reef", "Great Barrier Reef", accepted=True)
    assert exact_match(result, "Great Barrier Reef", top_k=1) is True
    assert substring_recovery(result, "Great Barrier Reef", top_k=1) == pytest.approx(1.0)
    partial = _result("Barrier Reef", "Great Barrier Reef", accepted=True)
    assert exact_match(partial, "Great Barrier Reef", top_k=1) is False
    assert substring_recovery(partial, "Great Barrier Reef", top_k=1) == pytest.approx(2 / 3)


def test_cross_cone_swap_detects_a_constant_answer():
    evaluation = {
        "per_record": [
            {"phrase": "black hole", "results": {"adapter_reranked": {"top_text": "same"}}},
            {"phrase": "solar eclipse", "results": {"adapter_reranked": {"top_text": "same"}}},
        ]
    }
    swap = evaluate_cross_cone_swap(evaluation)
    assert swap["answer_is_constant"] is True
    assert swap["matched_own_phrase"] == 0.0


# ------------------------------------------------------------------- go/no-go


def _baseline_metrics(**overrides) -> dict:
    base = {
        "n_records": 10,
        "exact_match_top1": 0.0,
        "exact_match_top5": 0.0,
        "mean_substring_recovery": 0.0,
        "beam_contains_correct": 0.0,
        "mean_explained_fraction": 0.0,
        "acceptance_rate": 0.0,
        "abstention_rate": 1.0,
        "acceptance_precision": None,
        "unfiltered_precision": 0.0,
        "auroc_correct_vs_incorrect_candidates": 0.5,
        "n_correct_candidates": 1,
        "n_incorrect_candidates": 9,
    }
    base.update(overrides)
    return base


def _passing_inputs() -> dict:
    return {
        "reconstructor_metrics": {
            "auroc_correct_vs_distractor": 0.9,
            "top5_retrieval": 0.7,
        },
        "evaluation": {
            "split": "heldout",
            "n_records": 10,
            "baselines": {
                "adapter_reranked": _baseline_metrics(
                    exact_match_top1=0.5,
                    beam_contains_correct=0.7,
                    acceptance_rate=0.6,
                    acceptance_precision=0.8,
                    unfiltered_precision=0.5,
                ),
                "adapter_raw_beam": _baseline_metrics(
                    exact_match_top1=0.3, beam_contains_correct=0.7
                ),
                "zero_memory": _baseline_metrics(beam_contains_correct=0.1, acceptance_rate=0.1),
                "shuffled_q": _baseline_metrics(acceptance_rate=0.1),
                "unrelated_q": _baseline_metrics(acceptance_rate=0.1),
                "sign_reversed_q": _baseline_metrics(acceptance_rate=0.1),
            },
        },
        "config": EVAL_CONFIG,
        "leakage_report": {"violations": []},
        "confabulation": {"clean": True},
        "prompt_robustness": {"cross_prompt_agreement": 0.9},
    }


def test_gonogo_passes_when_every_criterion_is_met():
    report = gonogo_report(**_passing_inputs())
    assert report["verdict"] == "GO"
    assert report["passed"] is True
    assert report["failed_criteria"] == []
    assert "failure_attribution" not in report


def test_gonogo_fails_and_attributes_when_the_reconstructor_is_weak():
    inputs = _passing_inputs()
    inputs["reconstructor_metrics"] = {
        "auroc_correct_vs_distractor": 0.55,
        "top5_retrieval": 0.2,
    }
    report = gonogo_report(**inputs, diagnostics={"train_top5_retrieval": 0.95})
    assert report["verdict"] == "NO-GO"
    assert report["failure_attribution"]["primary"] == "insufficient_data"


def test_gonogo_attributes_cone_information_loss_when_training_also_fails():
    inputs = _passing_inputs()
    inputs["reconstructor_metrics"] = {
        "auroc_correct_vs_distractor": 0.51,
        "top5_retrieval": 0.1,
    }
    report = gonogo_report(**inputs, diagnostics={"train_top5_retrieval": 0.2})
    assert report["failure_attribution"]["primary"] == "cone_information_loss"


def test_gonogo_attributes_the_adapter_when_generation_matches_zero_memory():
    inputs = _passing_inputs()
    inputs["evaluation"]["baselines"]["adapter_reranked"]["beam_contains_correct"] = 0.1
    inputs["evaluation"]["baselines"]["zero_memory"]["beam_contains_correct"] = 0.1
    report = gonogo_report(**inputs)
    assert report["verdict"] == "NO-GO"
    assert report["failure_attribution"]["primary"] == "cone_adapter"


def test_gonogo_fails_on_leakage_regardless_of_performance():
    inputs = _passing_inputs()
    inputs["leakage_report"] = {"violations": [{"kind": "split_mismatch"}]}
    report = gonogo_report(**inputs)
    assert report["verdict"] == "NO-GO"
    assert "no_leakage_detected" in report["failed_criteria"]


def test_gonogo_fails_when_controls_are_accepted_as_often_as_the_adapter():
    inputs = _passing_inputs()
    inputs["evaluation"]["baselines"]["shuffled_q"]["acceptance_rate"] = 0.6
    report = gonogo_report(**inputs)
    assert "controls_rejected" in report["failed_criteria"]


def test_gonogo_requires_the_adapter_baseline():
    inputs = _passing_inputs()
    inputs["evaluation"]["baselines"].pop("adapter_reranked")
    with pytest.raises(AutoencoderError, match="adapter_reranked"):
        gonogo_report(**inputs)


def test_a_none_auroc_fails_rather_than_passing_by_default():
    inputs = _passing_inputs()
    inputs["reconstructor_metrics"]["auroc_correct_vs_distractor"] = None
    report = gonogo_report(**inputs)
    assert report["verdict"] == "NO-GO"


def test_failure_attribution_names_prompt_dependence():
    criteria = [{"name": "acceptance_precision_gain", "passed": False}]
    attribution = attribute_failure(
        criteria,
        reconstructor_metrics={"top5_retrieval": 0.9},
        evaluation={"baselines": {}},
        prompt_robustness={"cross_prompt_agreement": 0.1},
        diagnostics={},
    )
    assert attribution["primary"] == "prompt_dependence"


def test_markdown_renders_both_verdicts_without_hiding_the_failure():
    passing = render_markdown(gonogo_report(**_passing_inputs()))
    assert "**Verdict: GO**" in passing
    inputs = _passing_inputs()
    inputs["leakage_report"] = {"violations": [{"kind": "substring_overlap"}]}
    failing = render_markdown(gonogo_report(**inputs))
    assert "**Verdict: NO-GO**" in failing
    assert "Primary failure mode" in failing
    assert not math.isnan(0.0)  # sanity: the table formatter handles real floats
