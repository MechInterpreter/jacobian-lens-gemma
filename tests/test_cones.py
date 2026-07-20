# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Tests for jlens.cones: record schema, signatures, and trajectory
utilities (CPU, synthetic records)."""

import json

import pytest
import torch

from jlens.cones import (
    CONE_RECORD_SCHEMA,
    active_set_overlap,
    coefficient_concentration,
    cone_signature,
    cone_trajectory,
    group_by_signature,
    group_by_similarity,
    load_cone_records,
    make_cone_record,
    recurring_signatures,
    save_cone_records,
    weighted_active_set_similarity,
)
from jlens.pursuit import JSpaceDictionary, PursuitSettings, gradient_pursuit

RUN_PROVENANCE = {
    "run_id": "test_run",
    "lens_fingerprint": "sha256:test",
    "model_revision": "deadbeef",
}


def fake_pursuit_record(token_ids, coefficients, *, explained=0.5, k=4):
    """A hand-built jlens.pursuit.result.v1 record for controlled tests."""
    return {
        "schema": "jlens.pursuit.result.v1",
        "requested_k": k,
        "n_selected": len(token_ids),
        "n_iterations": len(token_ids),
        "token_ids": list(token_ids),
        "coefficients": list(coefficients),
        "target_norm": 1.0,
        "residual_norm": (1.0 - explained) ** 0.5,
        "relative_residual": (1.0 - explained) ** 0.5,
        "explained_fraction": explained,
        "stop_reason": "max_atoms",
        "residual_norm_history": [1.0],
        "settings": {"k": k},
        "dictionary_provenance": {"kind": "test"},
    }


def record_for(layer, token_ids, coefficients, *, explained=0.5, position=-1,
               prompt_hash="hash1", slug="p", fmt="plain"):
    return make_cone_record(
        fake_pursuit_record(token_ids, coefficients, explained=explained),
        decoded_labels=[f"tok{i}" for i in token_ids],
        layer=layer,
        position=position,
        input_token_id=7,
        input_token=" x",
        prompt_hash=prompt_hash,
        prompt_slug=slug,
        prompt_format=fmt,
        run_provenance=RUN_PROVENANCE,
    )


# ------------------------------------------------------------- signature


def test_cone_signature_is_deterministic_and_order_invariant():
    a = cone_signature([5, 3, 9])
    b = cone_signature([9, 5, 3, 3])  # order and multiplicity ignored
    assert a == b
    assert a["token_ids"] == [3, 5, 9]
    assert a["digest"].startswith("sha256:")
    assert cone_signature([3, 5]) != a
    assert cone_signature([]) == cone_signature(())


# ---------------------------------------------------------------- record


def test_make_cone_record_schema_and_effective_set():
    record = record_for(14, [10, 20, 30], [2.0, 0.0, 1.0])
    assert record["schema"] == CONE_RECORD_SCHEMA
    for key in (
        "run_provenance",
        "prompt_hash",
        "format",
        "layer",
        "position",
        "input_token",
        "requested_k",
        "selected_token_ids",
        "selected_labels",
        "coefficients",
        "reconstruction",
        "stopping",
        "algorithm_settings",
        "cone_signature",
    ):
        assert key in record, key
    # Zero-coefficient atoms are excluded from the effective set and the
    # signature, but kept in the full selection record.
    assert record["selected_token_ids"] == [10, 20, 30]
    assert record["effective_token_ids"] == [10, 30]
    assert record["effective_labels"] == ["tok10", "tok30"]
    assert record["cone_signature"] == cone_signature([10, 30])
    assert record["run_provenance"]["lens_fingerprint"] == "sha256:test"


def test_make_cone_record_validates_inputs():
    with pytest.raises(ValueError, match="jlens.pursuit.result.v1"):
        make_cone_record(
            {"schema": "wrong"},
            decoded_labels=[],
            layer=0,
            position=-1,
            input_token_id=None,
            input_token=None,
            prompt_hash="h",
            prompt_slug=None,
            prompt_format="plain",
            run_provenance={},
        )
    with pytest.raises(ValueError, match="decoded_labels"):
        make_cone_record(
            fake_pursuit_record([1, 2], [1.0, 1.0]),
            decoded_labels=["only-one"],
            layer=0,
            position=-1,
            input_token_id=None,
            input_token=None,
            prompt_hash="h",
            prompt_slug=None,
            prompt_format="plain",
            run_provenance={},
        )


def test_cone_record_from_real_pursuit_result():
    """End-to-end: gradient pursuit output wraps cleanly into a record."""
    torch.manual_seed(0)
    atoms = torch.eye(8)
    dictionary = JSpaceDictionary(atoms, layer=3)
    result = gradient_pursuit(
        2.0 * atoms[1] + 1.0 * atoms[4], dictionary, PursuitSettings(k=2)
    )
    pursuit_record = result.to_records()[0]
    record = make_cone_record(
        pursuit_record,
        decoded_labels=[f"t{i}" for i in pursuit_record["token_ids"]],
        layer=3,
        position=-1,
        input_token_id=1,
        input_token=" a",
        prompt_hash="abc",
        prompt_slug="e2e",
        prompt_format="plain",
        run_provenance=RUN_PROVENANCE,
    )
    assert sorted(record["effective_token_ids"]) == [1, 4]
    assert record["reconstruction"]["explained_fraction"] == pytest.approx(
        1.0, abs=1e-5
    )
    json.dumps(record)  # JSON-safe


def test_save_load_round_trip(tmp_path):
    records = [record_for(3, [1], [1.0]), record_for(7, [2, 5], [1.0, 0.5])]
    path = str(tmp_path / "cones.json")
    save_cone_records(records, path)
    assert load_cone_records(path) == records
    with pytest.raises(ValueError, match="not a cone record"):
        save_cone_records([{"schema": "nope"}], path)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"schema": "nope"}]), encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected record schema"):
        load_cone_records(str(bad))


# ------------------------------------------------------------ comparison


def test_active_set_overlap_known_values():
    overlap = active_set_overlap([1, 2, 3], [2, 3, 4])
    assert overlap == {"intersection_size": 2, "union_size": 4, "jaccard": 0.5}
    assert active_set_overlap([], [])["jaccard"] == 1.0
    assert active_set_overlap([1], [])["jaccard"] == 0.0


def test_weighted_active_set_similarity_known_values():
    a = record_for(3, [1, 2], [1.0, 1.0])
    b = record_for(7, [1, 2], [2.0, 2.0])  # same direction, scaled
    assert weighted_active_set_similarity(a, b) == pytest.approx(1.0)
    c = record_for(7, [8, 9], [1.0, 1.0])  # disjoint
    assert weighted_active_set_similarity(a, c) == pytest.approx(0.0)
    d = record_for(7, [1, 9], [1.0, 1.0])  # half overlap
    assert weighted_active_set_similarity(a, d) == pytest.approx(0.5)
    empty = record_for(7, [], [])
    assert weighted_active_set_similarity(a, empty) == 0.0


def test_coefficient_concentration_known_values():
    uniform = coefficient_concentration([1.0, 1.0, 1.0, 1.0])
    assert uniform["herfindahl"] == pytest.approx(0.25)
    assert uniform["top1_share"] == pytest.approx(0.25)
    assert uniform["n_nonzero"] == 4
    single = coefficient_concentration([5.0])
    assert single["herfindahl"] == pytest.approx(1.0)
    assert single["top1_share"] == pytest.approx(1.0)
    assert coefficient_concentration([]) == {
        "herfindahl": 0.0,
        "top1_share": 0.0,
        "n_nonzero": 0,
    }


# ------------------------------------------------------------ trajectory


def test_cone_trajectory_transitions():
    records = [
        record_for(3, [1, 2, 3], [1.0, 1.0, 1.0], explained=0.1),
        record_for(7, [2, 3, 4], [1.0, 1.0, 2.0], explained=0.3),
        record_for(14, [4], [3.0], explained=0.8),
    ]
    transitions = cone_trajectory(records)
    assert len(transitions) == 2
    first, second = transitions
    assert (first["layer_from"], first["layer_to"]) == (3, 7)
    assert first["active_set_overlap"]["jaccard"] == pytest.approx(0.5)
    assert first["entered_token_ids"] == [4]
    assert first["exited_token_ids"] == [1]
    assert first["entered_labels"] == ["tok4"]
    assert first["delta_explained_fraction"] == pytest.approx(0.2)
    assert (second["layer_from"], second["layer_to"]) == (7, 14)
    assert second["concentration_to"]["herfindahl"] == pytest.approx(1.0)
    assert second["delta_explained_fraction"] == pytest.approx(0.5)
    json.dumps(transitions)


def test_cone_trajectory_orders_by_layer_and_validates():
    records = [
        record_for(14, [1], [1.0]),
        record_for(3, [1], [1.0]),
    ]
    transitions = cone_trajectory(records)
    assert transitions[0]["layer_from"] == 3
    assert cone_trajectory([]) == []
    mixed = [record_for(3, [1], [1.0]), record_for(7, [1], [1.0], position=-2)]
    with pytest.raises(ValueError, match="one \\(prompt, position\\)"):
        cone_trajectory(mixed)


# ------------------------------------------------------------ recurrence


def test_recurring_signatures_counts_and_orders():
    records = [
        record_for(3, [1, 2], [1.0, 1.0]),
        record_for(7, [2, 1], [0.5, 2.0]),  # same set, different order/coeffs
        record_for(14, [9], [1.0]),
    ]
    table = recurring_signatures(records)
    assert table[0]["count"] == 2
    assert table[0]["token_ids"] == [1, 2]
    assert table[1]["count"] == 1
    assert table[0]["first_seen"]["layer"] == 3


def test_group_by_signature_and_similarity():
    records = [
        record_for(3, [1, 2], [1.0, 1.0]),
        record_for(7, [1, 2], [3.0, 3.0]),
        record_for(14, [8, 9], [1.0, 1.0]),
    ]
    exact = group_by_signature(records)
    assert sorted(len(v) for v in exact.values()) == [1, 2]
    groups = group_by_similarity(records, threshold=0.9)
    assert groups == [[0, 1], [2]]
    # Threshold 0 puts everything with any (even zero) similarity together.
    assert group_by_similarity(records, threshold=0.0) == [[0, 1, 2]]
    with pytest.raises(ValueError, match="threshold"):
        group_by_similarity(records, threshold=1.5)
