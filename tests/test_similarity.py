# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Tests for jlens.similarity: metrics, grouping, and frequency statistics
(pure CPU, synthetic records)."""

import math

import pytest

from jlens.similarity import (
    atom_enrichment,
    atom_selection_frequencies,
    coefficient_map,
    inverse_frequency_weights,
    jaccard_similarity,
    output_token_recurrence,
    record_similarity,
    reweighted_map,
    similarity_groups,
    sparse_cosine,
    threshold_sensitivity,
    top_m_atoms,
    top_m_overlap,
    weighted_jaccard,
)


def rec(ids, coeffs, *, layer=1, k=4, fmt="plain", position=-1, slug="s", top1=None):
    """Minimal cone-record stand-in with the fields similarity uses."""
    return {
        "layer": layer,
        "requested_k": k,
        "format": fmt,
        "position": position,
        "prompt_slug": slug,
        "prompt_hash": f"hash-{slug}",
        "effective_token_ids": list(ids),
        "effective_coefficients": list(coeffs),
        "effective_labels": [f"tok{i}" for i in ids],
        "run_provenance": {"model_top1_id": top1},
    }


# ----------------------------------------------------------------- metrics


def test_jaccard_basics_and_empty():
    assert jaccard_similarity([1, 2], [2, 3]) == pytest.approx(1 / 3)
    assert jaccard_similarity([], []) == 1.0
    assert jaccard_similarity([1], []) == 0.0
    assert jaccard_similarity([1, 2], [1, 2]) == 1.0


def test_weighted_jaccard_known_value():
    a = {1: 2.0, 2: 1.0}
    b = {1: 1.0, 3: 1.0}
    # min-sum = 1.0; max-sum = 2 + 1 + 1 = 4.
    assert weighted_jaccard(a, b) == pytest.approx(0.25)
    assert weighted_jaccard({}, {}) == 1.0
    assert weighted_jaccard(a, {}) == 0.0
    assert weighted_jaccard(a, a) == 1.0


def test_weighted_jaccard_rejects_negative():
    with pytest.raises(ValueError):
        weighted_jaccard({1: -1.0}, {1: 1.0})


def test_sparse_cosine_known_value():
    a = {1: 1.0, 2: 1.0}
    b = {1: 1.0}
    assert sparse_cosine(a, b) == pytest.approx(1 / math.sqrt(2))
    assert sparse_cosine(a, {}) == 0.0
    assert sparse_cosine(a, a) == pytest.approx(1.0)


def test_top_m_atoms_deterministic_tie_break():
    r = rec([9, 3, 7], [1.0, 1.0, 2.0])
    # 7 has the largest coefficient; 3 and 9 tie -> lower token id first.
    assert top_m_atoms(r, 2) == [7, 3]
    assert top_m_atoms(r, 10) == [7, 3, 9]
    with pytest.raises(ValueError):
        top_m_atoms(r, 0)


def test_top_m_overlap():
    a = rec([1, 2, 3], [3.0, 2.0, 1.0])
    b = rec([1, 2, 4], [3.0, 2.0, 1.0])
    assert top_m_overlap(a, b, 2) == 1.0
    assert top_m_overlap(a, b, 3) == pytest.approx(2 / 3)


def test_record_similarity_exact_and_unknown_metric():
    a = rec([1, 2], [1.0, 5.0])
    b = rec([2, 1], [9.0, 9.0])
    assert record_similarity(a, b, metric="exact") == 1.0
    assert record_similarity(a, rec([1], [1.0]), metric="exact") == 0.0
    with pytest.raises(ValueError):
        record_similarity(a, b, metric="euclidean")


def test_record_similarity_with_atom_weights():
    a = rec([1, 2], [1.0, 1.0])
    b = rec([1, 3], [1.0, 1.0])
    raw = record_similarity(a, b, metric="weighted_jaccard")
    # Downweighting the shared atom 1 to zero removes all similarity.
    adjusted = record_similarity(
        a, b, metric="weighted_jaccard", atom_weights={1: 0.0}
    )
    assert raw > 0
    assert adjusted == 0.0


def test_reweighted_map_defaults_to_unit_weight():
    assert reweighted_map({1: 2.0, 5: 3.0}, {5: 0.5}) == {1: 2.0, 5: 1.5}


# ---------------------------------------------------------------- grouping


def test_similarity_groups_within_stratum():
    records = [
        rec([1, 2], [1.0, 1.0], slug="a"),
        rec([1, 2], [1.0, 1.0], slug="b"),
        rec([8, 9], [1.0, 1.0], slug="c"),
    ]
    groups = similarity_groups(records, metric="jaccard", threshold=0.99)
    sizes = sorted(g["size"] for g in groups)
    assert sizes == [1, 2]
    grouped = next(g for g in groups if g["size"] == 2)
    assert grouped["record_indices"] == [0, 1]
    assert grouped["stratum"] == {
        "layer": 1,
        "requested_k": 4,
        "format": "plain",
        "position": -1,
    }


def test_similarity_groups_do_not_cross_strata():
    records = [
        rec([1, 2], [1.0, 1.0], layer=1),
        rec([1, 2], [1.0, 1.0], layer=2),
    ]
    groups = similarity_groups(records, metric="jaccard", threshold=0.5)
    assert all(g["size"] == 1 for g in groups)


def test_similarity_groups_transitive_chain_is_one_component():
    # a~b and b~c above threshold, a~c below: still one component — which is
    # why the output is called a similarity group, not a cluster.
    records = [
        rec([1, 2, 3, 4], [1.0] * 4, slug="a"),
        rec([3, 4, 5, 6], [1.0] * 4, slug="b"),
        rec([5, 6, 7, 8], [1.0] * 4, slug="c"),
    ]
    groups = similarity_groups(records, metric="jaccard", threshold=1 / 3)
    assert [g["size"] for g in groups] == [3]


def test_similarity_groups_disjoint_supports_never_compared():
    records = [rec([i], [1.0], slug=str(i)) for i in range(50)]
    groups = similarity_groups(records, metric="jaccard", threshold=0.01)
    assert len(groups) == 50


def test_similarity_groups_threshold_validation():
    with pytest.raises(ValueError):
        similarity_groups([], metric="jaccard", threshold=0.0)
    with pytest.raises(ValueError):
        similarity_groups([], metric="nope", threshold=0.5)


def test_similarity_groups_deterministic():
    records = [
        rec([1, 2], [2.0, 1.0], slug=f"s{i}") for i in range(5)
    ] + [rec([2, 3], [1.0, 2.0], slug=f"t{i}") for i in range(5)]
    a = similarity_groups(records, metric="weighted_jaccard", threshold=0.4)
    b = similarity_groups(records, metric="weighted_jaccard", threshold=0.4)
    assert a == b


def test_threshold_sensitivity_monotone():
    records = [
        rec([1, 2], [1.0, 1.0], slug="a"),
        rec([1, 2], [1.0, 1.0], slug="b"),
        rec([1, 3], [1.0, 1.0], slug="c"),
    ]
    rows = threshold_sensitivity(
        records, metric="jaccard", thresholds=[0.9, 0.3, 0.6]
    )
    assert [row["threshold"] for row in rows] == [0.3, 0.6, 0.9]
    counts = [row["n_records_in_non_singleton_groups"] for row in rows]
    assert counts == sorted(counts, reverse=True)
    assert rows[0]["n_records_in_non_singleton_groups"] == 3
    assert rows[-1]["n_records_in_non_singleton_groups"] == 2


# ----------------------------------------------------------- frequencies


def test_atom_selection_frequencies_counts():
    records = [
        rec([1, 2], [1.0, 1.0], layer=1),
        rec([2, 3], [1.0, 1.0], layer=2),
    ]
    freqs = atom_selection_frequencies(records, strata=("layer",))
    assert freqs["overall"][2] == 2
    assert freqs["by_stratum"][(1,)][1] == 1
    assert (2,) in freqs["by_stratum"]
    assert freqs["record_counts_by_stratum"] == {(1,): 1, (2,): 1}
    assert freqs["labels"][3] == "tok3"


def test_inverse_frequency_weights_monotone():
    weights = inverse_frequency_weights({1: 1, 2: 10, 3: 100}, n_records=100)
    assert weights[1] > weights[2] > weights[3] > 0
    with pytest.raises(ValueError):
        inverse_frequency_weights({}, n_records=0)


def test_atom_enrichment_observed_vs_expected():
    records = [rec([7], [1.0], layer=1) for _ in range(3)] + [
        rec([8], [1.0], layer=2) for _ in range(3)
    ]
    freqs = atom_selection_frequencies(records, strata=("layer",))
    rows = atom_enrichment(freqs)
    layer1 = next(
        r for r in rows if r["stratum"] == {"layer": 1} and r["token_id"] == 7
    )
    assert layer1["observed"] == 3
    assert layer1["expected"] == pytest.approx(1.5)
    assert layer1["observed_over_expected"] == pytest.approx(2.0)
    assert layer1["log_odds"] > 0
    with pytest.raises(ValueError):
        atom_enrichment(freqs, smoothing=0.0)


def test_atom_enrichment_deterministic_order():
    records = [
        rec([5, 3], [2.0, 2.0], layer=1),
        rec([3, 5], [1.0, 1.0], layer=1),
    ]
    freqs = atom_selection_frequencies(records, strata=("layer",))
    rows = atom_enrichment(freqs)
    # Equal observed counts -> ordered by token id.
    assert [r["token_id"] for r in rows] == [3, 5]


def test_output_token_recurrence():
    records = [
        rec([1, 2], [1.0, 1.0], top1=1),
        rec([1, 3], [1.0, 1.0], top1=9),
        rec([2, 3], [1.0, 1.0], top1=2),
    ]
    rows = output_token_recurrence(records)
    by_id = {r["token_id"]: r for r in rows}
    assert by_id[1]["as_output_token"] == 1
    assert by_id[1]["as_non_output_atom"] == 1
    assert by_id[2]["as_output_token"] == 1
    assert 3 not in by_id  # never the record's own output token


def test_coefficient_map_strictness():
    r = rec([1, 2], [0.5, 1.5])
    assert coefficient_map(r) == {1: 0.5, 2: 1.5}
