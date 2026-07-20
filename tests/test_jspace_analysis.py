# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Tests for jlens.jspace_analysis against a synthetic run directory:
loading, integrity checking, cross-k matching, aggregation, evaluation
summaries, deterministic output, and non-mutation of the source run."""

import hashlib
import json
import os

import pytest

from jlens.cones import cone_signature
from jlens.jspace_analysis import (
    check_cone_record,
    check_integrity,
    cross_k_stability,
    eval_control_collisions,
    eval_control_summary,
    k_marginal_gains,
    load_run,
    match_across_k,
    metrics_table,
    record_key,
    record_metrics,
    stabilization_robustness,
    transition_summary,
    write_csv,
    write_json,
)

LAYERS = [2, 5]
K_VALUES = [2, 3]
PROMPTS = [
    ("alpha", "plain", "hash-alpha", "factual", 101),
    ("beta", "chat", "hash-beta", "antonym", 202),
]
POSITION = -1


def make_record(slug, fmt, prompt_hash, category, top1, layer, k):
    """A valid cone record; k=3 supports nest the k=2 supports."""
    base_ids = [10 + layer, 20 + layer]
    ids = base_ids + [30 + layer] if k == 3 else base_ids
    if slug == "beta":  # different support per prompt, shares one atom
        ids = [10 + layer] + [40 + layer] + ([50 + layer] if k == 3 else [])
    coeffs = [float(len(ids) - i) for i in range(len(ids))]
    if layer == 5 and slug == "alpha":
        ids = list(ids[:-1]) + [top1]  # output token present at late layer
    target = 100.0
    residual = 99.0 - k - layer / 10
    history = [target] + [target - (target - residual) * (i + 1) / len(ids) for i in range(len(ids))]
    return {
        "schema": "jlens.cones.record.v1",
        "run_provenance": {
            "run_id": "testrun",
            "category": category,
            "model_top1_id": top1,
            "model_top1_token": f"tok{top1}",
        },
        "prompt_hash": prompt_hash,
        "prompt_slug": slug,
        "format": fmt,
        "layer": layer,
        "position": POSITION,
        "input_token_id": 1,
        "input_token": "x",
        "requested_k": k,
        "n_selected": len(ids),
        "selected_token_ids": ids,
        "selected_labels": [f"tok{i}" for i in ids],
        "coefficients": coeffs,
        "effective_token_ids": ids,
        "effective_labels": [f"tok{i}" for i in ids],
        "effective_coefficients": coeffs,
        "reconstruction": {
            "target_norm": target,
            "residual_norm": residual,
            "relative_residual": residual / target,
            "explained_fraction": 1 - (residual / target) ** 2,
        },
        "stopping": {
            "stop_reason": "max_atoms",
            "n_iterations": len(ids),
            "residual_norm_history": history,
        },
        "algorithm_settings": {"k": k},
        "dictionary_provenance": {"layer": layer},
        "cone_signature": cone_signature(ids),
    }


def make_transition(slug, fmt, prompt_hash, layer_from, layer_to):
    return {
        "schema": "jlens.cones.transition.v1",
        "prompt_hash": prompt_hash,
        "prompt_slug": slug,
        "format": fmt,
        "position": POSITION,
        "layer_from": layer_from,
        "layer_to": layer_to,
        "active_set_overlap": {"intersection_size": 0, "union_size": 4, "jaccard": 0.0},
        "weighted_similarity": 0.5,
        "explained_fraction_from": 0.01,
        "explained_fraction_to": 0.02,
        "delta_explained_fraction": 0.01,
        "concentration_from": {"herfindahl": 0.5, "top1_share": 0.6, "n_nonzero": 2},
        "concentration_to": {"herfindahl": 0.6, "top1_share": 0.7, "n_nonzero": 2},
        "entered_token_ids": [],
        "entered_labels": [],
        "exited_token_ids": [],
        "exited_labels": [],
    }


def make_ignition(slug, fmt, prompt_hash, layer_from, layer_to, top1):
    return {
        "schema": "jlens.ignition.candidate.v1",
        "label": "candidate ignition diagnostics — NOT validated ignition",
        "prompt_hash": prompt_hash,
        "prompt_slug": slug,
        "format": fmt,
        "position": POSITION,
        "layer_from": layer_from,
        "layer_to": layer_to,
        "signals": {
            "delta_explained_fraction": 0.01,
            "explained_fraction_to": 0.02,
            "active_set_jaccard": 0.0,
            "weighted_similarity": 0.5,
            "delta_herfindahl": 0.1,
            "top1_share_to": 0.7,
            "persistence_length_from_here": 0,
            "output_alignment_to": {
                "model_top1_id": top1,
                "in_active_set": True,
                "coefficient_share": 0.4,
            },
        },
        "entered_labels": [],
        "exited_labels": [],
        "caveats": "test",
    }


def make_eval_example(slug, fmt, category, ranks_by_variant):
    return {
        "slug": slug,
        "category": category,
        "format": fmt,
        "positions": [POSITION],
        "model_top1_ids": [7],
        "layers": {
            str(layer): {
                variant: {
                    "topk_overlap_with_model": 0.1,
                    "rank_of_model_top1": [rank],
                }
                for variant, rank in ranks_by_variant.items()
            }
            for layer in LAYERS
        },
    }


VARIANT_RANKS = {
    "jlens": 0,
    "logit_lens": 4,
    "permuted": 100,
    "random": 200,
    "adjacent_layer": 9,
    "distant_layer": 300,
    "shuffled_layer": 300,
}


@pytest.fixture
def run_dir(tmp_path):
    """A synthetic but schema-complete jspace run directory."""
    root = tmp_path / "run"
    cones_dir = root / "artifacts" / "cones"
    os.makedirs(cones_dir)
    capture_meta = [
        {
            "slug": slug,
            "category": category,
            "format": fmt,
            "position": POSITION,
            "prompt_hash": prompt_hash,
            "seq_len": 5,
            "input_token_id": 1,
            "input_token": "x",
            "model_top1_id": top1,
            "model_top1_token": f"tok{top1}",
        }
        for slug, fmt, prompt_hash, category, top1 in PROMPTS
    ]
    metadata = {
        "run_id": "testrun",
        "run_dir": "/tmp/testrun",
        "config": {
            "mode": "jspace_pursuit",
            "model": {
                "repo_id": "test/model",
                "revision": "rev-1",
                "allow_model_load": False,
            },
            "lens": {"expect_file_sha256": "sha256:feed"},
            "decomposition": {"layers": LAYERS, "k_values": K_VALUES},
        },
        "lens_verification": {"file_sha256": "sha256:feed"},
        "load_info": {"model_repo_id": "test/model", "model_revision": "rev-1"},
        "n_activations_per_layer": len(PROMPTS),
        "capture_meta": capture_meta,
        "environment": {"local_commit": "abc"},
    }
    with open(root / "run_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle)

    for layer in LAYERS:
        for k in K_VALUES:
            records = [
                make_record(slug, fmt, prompt_hash, category, top1, layer, k)
                for slug, fmt, prompt_hash, category, top1 in PROMPTS
            ]
            with open(
                cones_dir / f"cones_layer{layer}_k{k}.json", "w", encoding="utf-8"
            ) as handle:
                json.dump(records, handle)

    for k in K_VALUES:
        transitions = [
            make_transition(slug, fmt, prompt_hash, LAYERS[0], LAYERS[1])
            for slug, fmt, prompt_hash, _, _ in PROMPTS
        ]
        ignitions = [
            make_ignition(slug, fmt, prompt_hash, LAYERS[0], LAYERS[1], top1)
            for slug, fmt, prompt_hash, _, top1 in PROMPTS
        ]
        signatures = [
            {
                "digest": f"sha256:{k}{i}",
                "count": len(LAYERS),
                "token_ids": [1],
                "labels": ["tok1"],
                "first_seen": {"prompt_slug": slug, "layer": LAYERS[0], "position": POSITION},
            }
            for i, (slug, _, _, _, _) in enumerate(PROMPTS)
        ]
        for name, payload in (
            (f"trajectories_k{k}.json", transitions),
            (f"ignition_candidates_k{k}.json", ignitions),
            (f"recurring_signatures_k{k}.json", signatures),
        ):
            with open(root / "artifacts" / name, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)

    evaluation = {
        "results": {
            "layers": LAYERS,
            "top_k": 10,
            "n_prompts": len(PROMPTS),
            "examples": [
                make_eval_example(slug, fmt, category, VARIANT_RANKS)
                for slug, fmt, _, category, _ in PROMPTS
            ],
            "provenance": {
                "adjacent_layer": {
                    "mapping": [
                        {"applied_at_layer": 2, "jacobian_fitted_at_layer": 5},
                        {"applied_at_layer": 5, "jacobian_fitted_at_layer": 2},
                    ]
                },
                "distant_layer": {
                    "mapping": [
                        {"applied_at_layer": 2, "jacobian_fitted_at_layer": 5},
                        {"applied_at_layer": 5, "jacobian_fitted_at_layer": 2},
                    ]
                },
                "shuffled_layer": {
                    "mapping": [
                        {"applied_at_layer": 2, "jacobian_fitted_at_layer": 5},
                        {"applied_at_layer": 5, "jacobian_fitted_at_layer": 2},
                    ]
                },
            },
        }
    }
    with open(root / "artifacts" / "eval_v2_results.json", "w", encoding="utf-8") as handle:
        json.dump(evaluation, handle)
    return str(root)


def tree_digest(path):
    digest = hashlib.sha256()
    for base, _, files in sorted(os.walk(path)):
        for name in sorted(files):
            full = os.path.join(base, name)
            digest.update(name.encode())
            with open(full, "rb") as handle:
                digest.update(handle.read())
    return digest.hexdigest()


def test_load_and_integrity_clean(run_dir):
    art = load_run(run_dir)
    assert sorted(art.cones) == [(2, 2), (2, 3), (5, 2), (5, 3)]
    report = check_integrity(art)
    assert report["clean"], report["issues"]
    # The static-config/allow_model_load inconsistency is a note, not an issue.
    assert any("allow_model_load" in note for note in report["notes"])
    assert report["record_counts"]["cones_layer2_k2"] == 2


def test_integrity_flags_malformed_records(run_dir):
    art = load_run(run_dir)
    record = art.cones[(2, 2)][0]
    record["effective_coefficients"][0] = float("nan")
    report = check_integrity(art)
    assert not report["clean"]
    assert any("nonfinite coefficient" in issue for issue in report["issues"])


def test_check_cone_record_catches_specific_defects():
    good = make_record("alpha", "plain", "hash-alpha", "factual", 101, 2, 2)
    assert check_cone_record(good, 2, 2) == []

    dup = make_record("alpha", "plain", "hash-alpha", "factual", 101, 2, 2)
    dup["effective_token_ids"] = [12, 12]
    dup["cone_signature"] = cone_signature(dup["effective_token_ids"])
    assert any("duplicate token ids" in p for p in check_cone_record(dup, 2, 2))

    negative = make_record("alpha", "plain", "hash-alpha", "factual", 101, 2, 2)
    negative["effective_coefficients"][0] = -1.0
    assert any("non-positive" in p for p in check_cone_record(negative, 2, 2))

    upward = make_record("alpha", "plain", "hash-alpha", "factual", 101, 2, 2)
    upward["stopping"]["residual_norm_history"] = [100.0, 99.0, 99.5]
    assert any("non-increasing" in p for p in check_cone_record(upward, 2, 2))

    mismatch = make_record("alpha", "plain", "hash-alpha", "factual", 101, 2, 2)
    mismatch["cone_signature"] = {"token_ids": [], "digest": "sha256:0000"}
    assert any("digest mismatch" in p for p in check_cone_record(mismatch, 2, 2))

    short = make_record("alpha", "plain", "hash-alpha", "factual", 101, 2, 2)
    short["n_selected"] = 1
    short["selected_token_ids"] = short["selected_token_ids"][:1]
    assert any("max_atoms but n_selected" in p for p in check_cone_record(short, 2, 2))


def test_match_across_k(run_dir):
    art = load_run(run_dir)
    matched = match_across_k(art)
    assert len(matched) == len(PROMPTS) * len(LAYERS)
    for by_k in matched.values():
        assert sorted(by_k) == K_VALUES
    key = record_key(art.cones[(2, 2)][0])
    assert matched[key][2]["requested_k"] == 2

    art.cones[(2, 2)].append(art.cones[(2, 2)][0])
    with pytest.raises(ValueError, match="duplicate record"):
        match_across_k(art)


def test_record_metrics_output_alignment():
    record = make_record("alpha", "plain", "hash-alpha", "factual", 101, 5, 3)
    metrics = record_metrics(record)
    assert metrics["output_token_included"] is True
    assert 0 < metrics["output_token_share"] < 1
    assert metrics["active_set_size"] == 3
    assert metrics["explained_fraction"] == pytest.approx(
        record["reconstruction"]["explained_fraction"]
    )


def test_metrics_table_strata(run_dir):
    art = load_run(run_dir)
    rows = metrics_table(art)
    overall = [
        r
        for r in rows
        if r["layer"] == 2
        and r["k"] == 2
        and (r["format"], r["category"], r["position"]) == ("all", "all", "all")
    ]
    assert len(overall) == 1
    assert overall[0]["n"] == 2
    plain = next(
        r
        for r in rows
        if r["layer"] == 2 and r["k"] == 2 and r["format"] == "plain"
        and r["category"] == "all" and r["position"] == "all"
    )
    assert plain["n"] == 1


def test_k_marginal_gains_positive(run_dir):
    art = load_run(run_dir)
    rows = k_marginal_gains(art)
    all_rows = [r for r in rows if r["format"] == "all"]
    assert all(r["mean_delta_explained_fraction"] > 0 for r in all_rows)
    assert all(r["mean_delta_residual_norm"] < 0 for r in all_rows)
    assert {(r["k_from"], r["k_to"]) for r in all_rows} == {(2, 3)}


def test_cross_k_stability_nested_supports(run_dir):
    art = load_run(run_dir)
    rows = cross_k_stability(art)
    for row in rows:
        if row["format"] == "all":
            # Fixture supports are nested except where the output token
            # replaces the tail atom at the late layer.
            assert row["mean_containment"] >= 0.5
            assert 0 < row["mean_jaccard"] <= 1


def test_transition_summary_and_stabilization(run_dir):
    art = load_run(run_dir)
    rows = transition_summary(art)
    assert rows
    top = next(
        r for r in rows if r["k"] == 2 and r["format"] == "all" and r["cut"] == "all"
    )
    assert top["n"] == 2
    assert top["output_alignment_rate_to"] == 1.0

    stab = stabilization_robustness(art)
    assert all(r["is_final_transition"] for r in stab)
    assert {r["k"] for r in stab} == set(K_VALUES)
    for r in stab:
        assert 0 <= r["frequency_adjusted_similarity"] <= 1
        assert 0 <= r["similarity_without_output_token"] <= 1


def test_eval_control_summary_values(run_dir):
    art = load_run(run_dir)
    rows = eval_control_summary(art)
    jlens = next(
        r
        for r in rows
        if r["layer"] == 2
        and (r["format"], r["category"], r["position"]) == ("all", "all", "all")
        and r["variant"] == "jlens"
    )
    assert jlens["n_ranks"] == 2
    assert jlens["median_rank"] == 0
    assert jlens["hit_rate@1"] == 1.0
    assert jlens["mean_reciprocal_rank"] == pytest.approx(1.0)
    adjacent = next(
        r
        for r in rows
        if r["layer"] == 2
        and (r["format"], r["category"], r["position"]) == ("all", "all", "all")
        and r["variant"] == "adjacent_layer"
    )
    assert adjacent["hit_rate@10"] == 1.0
    assert adjacent["hit_rate@5"] == 0.0


def test_eval_control_collisions(run_dir):
    art = load_run(run_dir)
    report = eval_control_collisions(art)
    # All three mappings coincide in the fixture -> expected identical, and
    # the recorded ranks are identical for distant/shuffled (consistent) but
    # differ for adjacent vs distant (inconsistent -> flagged).
    ds = [
        p
        for p in report["pairs"]
        if (p["variant_a"], p["variant_b"]) == ("distant_layer", "shuffled_layer")
    ]
    assert all(p["expected_identical"] for p in ds)
    assert all(p["consistent"] for p in ds)
    ad = [
        p
        for p in report["pairs"]
        if (p["variant_a"], p["variant_b"]) == ("adjacent_layer", "distant_layer")
    ]
    assert all(p["expected_identical"] and not p["consistent"] for p in ad)


def test_analysis_does_not_mutate_run_dir(run_dir, tmp_path):
    before = tree_digest(run_dir)
    art = load_run(run_dir)
    check_integrity(art)
    metrics_table(art)
    k_marginal_gains(art)
    cross_k_stability(art)
    transition_summary(art)
    stabilization_robustness(art)
    eval_control_summary(art)
    eval_control_collisions(art)
    write_csv(metrics_table(art), str(tmp_path / "out.csv"))
    write_json(check_integrity(art), str(tmp_path / "out.json"))
    assert tree_digest(run_dir) == before


def test_write_csv_deterministic_and_typed(tmp_path):
    rows = [
        {"a": 1, "b": 0.123456789, "c": None, "d": True, "e": "x"},
        {"a": 2, "b": float("inf"), "c": "z", "d": False, "e": "y"},
    ]
    path_a, path_b = str(tmp_path / "a.csv"), str(tmp_path / "b.csv")
    write_csv(rows, path_a)
    write_csv(rows, path_b)
    with open(path_a, encoding="utf-8") as handle:
        content = handle.read()
    assert content == open(path_b, encoding="utf-8").read()
    assert content.splitlines()[0] == "a,b,c,d,e"
    assert "0.12345679" in content
    assert "true" in content and "false" in content
    write_csv([], str(tmp_path / "empty.csv"))
    assert open(tmp_path / "empty.csv", encoding="utf-8").read() == ""
