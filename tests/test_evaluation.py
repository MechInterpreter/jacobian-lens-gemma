# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Tests for jlens.evaluation: control-suite provenance, aggregate
statistics, and the categorized held-out prompt set (CPU, tiny models)."""

import json
import os

import pytest
import torch

from jlens.evaluation import (
    aggregate_ranks,
    build_control_suite,
    evaluate_suite,
    load_eval_prompts_v2,
)
from jlens.fitting import fit

from .mock_gemma4 import MockTokenizer
from .tiny import TinyDecoder

EVAL_V2_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs",
    "prompts",
    "eval_prompts_v2.json",
)


@pytest.fixture(scope="module")
def fitted():
    model = TinyDecoder(n_layers=4, d_model=8)
    prompts = ["abcdefghij klmnop " * 3, "zyxwvutsrq ponmlk " * 3]
    lens = fit(model, prompts, source_layers=[0, 1, 2], dim_batch=4, max_seq_len=48)
    return model, lens


def test_aggregate_ranks_known_values():
    stats = aggregate_ranks([0, 4, 9, 100])
    assert stats["n"] == 4
    assert stats["median_rank"] == pytest.approx(6.5)
    assert stats["mean_reciprocal_rank"] == pytest.approx(
        (1 / 1 + 1 / 5 + 1 / 10 + 1 / 101) / 4
    )
    assert stats["hit_rate@1"] == pytest.approx(0.25)
    assert stats["hit_rate@5"] == pytest.approx(0.5)
    assert stats["hit_rate@10"] == pytest.approx(0.75)


def test_aggregate_ranks_empty():
    stats = aggregate_ranks([])
    assert stats["n"] == 0
    assert stats["median_rank"] is None
    assert stats["mean_reciprocal_rank"] is None


def test_build_control_suite_names_and_provenance(fitted):
    _, lens = fitted
    suite = build_control_suite(lens, control_seed=1234)
    assert set(suite) == {
        "jlens",
        "logit_lens",
        "permuted",
        "random",
        "adjacent_layer",
        "distant_layer",
        "shuffled_layer",
    }
    assert suite["logit_lens"].use_jacobian is False
    for name, variant in suite.items():
        assert variant.provenance, f"{name} has no provenance"
    # The layer-mapping controls record the exact mapping used.
    for name in ("adjacent_layer", "distant_layer", "shuffled_layer"):
        rows = suite[name].provenance["mapping"]
        assert {row["applied_at_layer"] for row in rows} == set(lens.source_layers)


def test_control_suite_matrices_match_their_claims(fitted):
    """Every variant must hold exactly the matrix its provenance claims."""
    _, lens = fitted
    suite = build_control_suite(lens, control_seed=1234)
    for layer in lens.source_layers:
        J = lens.jacobians[layer]
        torch.testing.assert_close(suite["jlens"].lens.jacobians[layer], J)
        # Row-permuted: identical row multiset, different arrangement.
        P = suite["permuted"].lens.jacobians[layer]
        srt = lambda M: M[torch.argsort(M[:, 0])]  # noqa: E731
        torch.testing.assert_close(srt(P), srt(J))
        assert not torch.equal(P, J)
        # Scale-matched random: same Frobenius norm, different entries.
        R = suite["random"].lens.jacobians[layer]
        torch.testing.assert_close(R.norm(), J.norm())
        assert not torch.allclose(R, J)
    for name in ("adjacent_layer", "distant_layer", "shuffled_layer"):
        for row in suite[name].provenance["mapping"]:
            torch.testing.assert_close(
                suite[name].lens.jacobians[row["applied_at_layer"]],
                lens.jacobians[row["jacobian_fitted_at_layer"]],
            )


def test_build_control_suite_single_layer_omits_layer_mapping():
    from jlens.lens import JacobianLens

    lens = JacobianLens(jacobians={1: torch.eye(8)}, n_prompts=1, d_model=8)
    suite = build_control_suite(lens, control_seed=0)
    assert set(suite) == {"jlens", "logit_lens", "permuted", "random"}


def test_load_eval_prompts_v2_schema_and_rendering():
    rows = load_eval_prompts_v2(EVAL_V2_PATH, MockTokenizer())
    assert len(rows) >= 30
    formats = {row["format"] for row in rows}
    assert formats == {"plain", "chat"}
    categories = {row["category"] for row in rows}
    for expected in (
        "factual",
        "multihop",
        "association",
        "antonym",
        "counting",
        "syntactic",
    ):
        assert expected in categories, f"missing category {expected}"
    # Chat rows are rendered through the tokenizer's chat template.
    chat_rows = [row for row in rows if row["format"] == "chat"]
    assert len(chat_rows) >= 5
    assert all("<user>" in row["text"] for row in chat_rows)
    # Every category with plain prompts has several examples (aggregates
    # need more than one sample).
    plain_by_cat: dict[str, int] = {}
    for row in rows:
        if row["format"] == "plain":
            plain_by_cat[row["category"]] = plain_by_cat.get(row["category"], 0) + 1
    assert all(count >= 4 for count in plain_by_cat.values()), plain_by_cat


def test_eval_prompts_v2_is_valid_json_with_versions():
    with open(EVAL_V2_PATH, encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["version"] == 2
    slugs = [e["slug"] for e in payload["plain"]] + [
        e["slug"] for e in payload["chat"]
    ]
    assert len(slugs) == len(set(slugs)), "duplicate slugs"


@pytest.fixture(scope="module")
def suite_results(fitted):
    model, lens = fitted
    suite = build_control_suite(lens, control_seed=1234)
    prompts = [
        {
            "slug": "p1",
            "category": "factual",
            "format": "plain",
            "text": "abcdefghij klmnop qrstu " * 2,
            "positions": [-2, -1],
        },
        {
            "slug": "p2",
            "category": "counting",
            "format": "plain",
            "text": "one two three four five six seven",
            "positions": [-1],
        },
        {
            "slug": "c1",
            "category": "factual",
            "format": "chat",
            "text": "<user> hello there <assistant>",
            "positions": [-1],
        },
    ]
    results = evaluate_suite(
        model, suite, prompts, layers=lens.source_layers, top_k=5
    )
    return model, lens, suite, prompts, results


def test_evaluate_suite_structure(suite_results):
    _, lens, suite, prompts, results = suite_results
    assert results["n_prompts"] == 3
    assert len(results["examples"]) == 3
    for example in results["examples"]:
        for layer in lens.source_layers:
            per_layer = example["layers"][layer]
            for name in suite:
                assert "topk_overlap_with_model" in per_layer[name]
                assert "rank_of_model_top1" in per_layer[name]
                for rank in per_layer[name]["rank_of_model_top1"]:
                    assert 0 <= rank < 32  # vocab bound
    # Plain and chat aggregated separately; both metric families present.
    assert set(results["aggregates"]) == {"plain", "chat"}
    plain_layer0 = results["aggregates"]["plain"]["0"]
    assert set(plain_layer0) == set(suite)
    for stats in plain_layer0.values():
        assert stats["n"] == 3  # p1 has 2 positions + p2 has 1
        assert "median_rank" in stats and "mean_topk_overlap" in stats
        assert "mean_reciprocal_rank" in stats and "hit_rate@10" in stats
    # Per-category aggregation exists and separates formats.
    assert "factual" in results["aggregates_by_category"]["plain"]
    assert "counting" in results["aggregates_by_category"]["plain"]
    assert "factual" in results["aggregates_by_category"]["chat"]
    # Control provenance is embedded in the output payload.
    assert set(results["provenance"]) == set(suite)


def test_evaluate_suite_is_deterministic(suite_results):
    model, lens, suite, prompts, results = suite_results
    again = evaluate_suite(
        model, suite, prompts, layers=lens.source_layers, top_k=5
    )
    assert json.dumps(results, sort_keys=True) == json.dumps(again, sort_keys=True)


def test_evaluate_suite_jlens_matches_direct_apply(suite_results):
    """The one-forward-pass evaluation path must reproduce what the plain
    JacobianLens.apply readout produces for the fitted lens."""
    model, lens, suite, prompts, results = suite_results
    from jlens.controls import ranks_of_targets

    prompt = prompts[0]
    lens_logits, model_logits, _ = lens.apply(
        model, prompt["text"], layers=[1], positions=prompt["positions"]
    )
    expected = [
        int(r)
        for r in ranks_of_targets(lens_logits[1].float(), model_logits.argmax(-1))
    ]
    recorded = results["examples"][0]["layers"][1]["jlens"]["rank_of_model_top1"]
    assert recorded == expected


def test_evaluate_suite_is_json_serializable(suite_results):
    *_, results = suite_results
    json.dumps(results)  # must not raise
