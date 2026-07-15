# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Tests for jlens.ignition: candidate diagnostics stay explicitly labeled,
separately visible, and heuristically composable only on request."""

import json

import pytest

from jlens.cones import cone_trajectory
from jlens.ignition import (
    DEFAULT_HEURISTIC_WEIGHTS,
    IGNITION_RECORD_SCHEMA,
    candidate_ignition_signals,
    export_transition_records,
    heuristic_candidate_score,
    load_transition_records,
)

from .test_cones import record_for


@pytest.fixture()
def trajectory():
    """Layers 3→7→14→21: unstable early, then stable+concentrated with a
    reconstruction jump at 14 and the output token (id 4) dominant."""
    records = [
        record_for(3, [1, 2, 3], [1.0, 1.0, 1.0], explained=0.05),
        record_for(7, [5, 6], [1.0, 1.0], explained=0.10),
        record_for(14, [4, 6], [4.0, 1.0], explained=0.60),
        record_for(21, [4, 6], [5.0, 1.0], explained=0.70),
    ]
    by_layer = {r["layer"]: r for r in records}
    return cone_trajectory(records), by_layer


def test_signals_are_separately_visible(trajectory):
    transitions, by_layer = trajectory
    signals = candidate_ignition_signals(
        transitions, by_layer, model_top1_id=4, persistence_jaccard_threshold=0.5
    )
    assert len(signals) == 3
    for record in signals:
        assert record["schema"] == IGNITION_RECORD_SCHEMA
        assert "NOT validated" in record["label"]
        for key in (
            "delta_explained_fraction",
            "active_set_jaccard",
            "weighted_similarity",
            "delta_herfindahl",
            "top1_share_to",
            "persistence_length_from_here",
            "output_alignment_to",
        ):
            assert key in record["signals"], key
        assert "caveats" in record

    jump = signals[1]  # 7 -> 14
    assert jump["signals"]["delta_explained_fraction"] == pytest.approx(0.5)
    assert jump["signals"]["top1_share_to"] == pytest.approx(0.8)
    alignment = jump["signals"]["output_alignment_to"]
    assert alignment["in_active_set"] is True
    assert alignment["coefficient_share"] == pytest.approx(0.8)

    stable = signals[2]  # 14 -> 21: same active set
    assert stable["signals"]["active_set_jaccard"] == pytest.approx(1.0)


def test_persistence_run_lengths(trajectory):
    transitions, by_layer = trajectory
    signals = candidate_ignition_signals(
        transitions, by_layer, persistence_jaccard_threshold=0.5
    )
    lengths = [s["signals"]["persistence_length_from_here"] for s in signals]
    # 3->7 disjoint (0.0 < 0.5 stops immediately); 7->14 jaccard 1/3 < 0.5;
    # 14->21 jaccard 1.0.
    assert lengths == [0, 0, 1]


def test_alignment_optional(trajectory):
    transitions, by_layer = trajectory
    signals = candidate_ignition_signals(transitions, by_layer)
    assert all(s["signals"]["output_alignment_to"] is None for s in signals)


def test_rejects_non_transition_records(trajectory):
    _, by_layer = trajectory
    with pytest.raises(ValueError, match="transition"):
        candidate_ignition_signals([{"schema": "nope"}], by_layer)


def test_heuristic_composite_disabled_by_default(trajectory):
    transitions, by_layer = trajectory
    signals = candidate_ignition_signals(transitions, by_layer, model_top1_id=4)
    assert heuristic_candidate_score(signals[1]) is None  # disabled: no score
    scored = heuristic_candidate_score(signals[1], enabled=True)
    assert "HEURISTIC" in scored["label"]
    assert scored["weights"] == DEFAULT_HEURISTIC_WEIGHTS
    assert set(scored["components"]) == set(DEFAULT_HEURISTIC_WEIGHTS)
    assert scored["score"] == pytest.approx(sum(scored["components"].values()))
    custom = heuristic_candidate_score(
        signals[1], enabled=True, weights={"active_set_jaccard": 2.0}
    )
    assert custom["components"] == {
        "active_set_jaccard": 2.0 * signals[1]["signals"]["active_set_jaccard"]
    }
    with pytest.raises(KeyError, match="unknown signal"):
        heuristic_candidate_score(
            signals[1], enabled=True, weights={"not_a_signal": 1.0}
        )


def test_export_round_trip(tmp_path, trajectory):
    transitions, by_layer = trajectory
    signals = candidate_ignition_signals(transitions, by_layer, model_top1_id=4)
    path = str(tmp_path / "ignition.json")
    export_transition_records(signals, path)
    assert load_transition_records(path) == signals
    json.dumps(signals)
    with pytest.raises(ValueError, match="not a candidate-ignition record"):
        export_transition_records([{"schema": "nope"}], path)
