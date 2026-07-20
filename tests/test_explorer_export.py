# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Tests for the explorer bundle exporter: schema validity, determinism,
stable IDs, path stripping, malformed-input rejection, immutability of
sources, and merge semantics. CPU only, no model."""

import hashlib
import json
from pathlib import Path

import pytest

from jlens.explorer_export import (
    DEFAULT_DEMO_SLUGS,
    ExportError,
    assemble_bundle,
    assert_no_absolute_paths,
    build_text_bundle,
    canonical_json,
    example_id,
    intervention_to_causal_record,
    load_run_artifacts,
    make_provenance,
    merge_bundles,
    validate_bundle,
    write_bundle,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = str(REPO_ROOT / "schemas" / "explorer_bundle.schema.json")
REAL_RUN = REPO_ROOT / "runs" / "jspace_20260716T170808536780_e4118850fb70"

MODEL_REVISION = "fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd"


# ------------------------------------------------------------ fake run


def _cone(slug, prompt_hash, layer, position, token_ids, coefficients):
    labels = [f"tok{t}" for t in token_ids]
    effective = [(t, l, c) for t, l, c in zip(token_ids, labels, coefficients) if c > 0]
    history = [100.0 - 2.0 * i for i in range(len(token_ids) + 1)]
    return {
        "schema": "jlens.cones.record.v1",
        "run_provenance": {
            "run_id": "jspace_fake",
            "run_dir": "/content/drive/MyDrive/fake",  # must be stripped
            "config_fingerprint": "sha256:" + "ab" * 32,
            "lens_fingerprint": "sha256:" + "cd" * 32,
            "lens_path": "/content/drive/MyDrive/fake/lens.pt",
            "model_revision": MODEL_REVISION,
            "local_commit": "e" * 40,
            "upstream_commit": "f" * 40,
        },
        "prompt_hash": prompt_hash,
        "prompt_slug": slug,
        "format": "plain",
        "layer": layer,
        "position": position,
        "input_token_id": 11,
        "input_token": " is",
        "requested_k": 3,
        "n_selected": len(token_ids),
        "selected_token_ids": token_ids,
        "selected_labels": labels,
        "coefficients": coefficients,
        "effective_token_ids": [t for t, _, _ in effective],
        "effective_labels": [l for _, l, _ in effective],
        "effective_coefficients": [c for _, _, c in effective],
        "reconstruction": {
            "target_norm": 100.0,
            "residual_norm": history[-1],
            "relative_residual": history[-1] / 100.0,
            "explained_fraction": 1.0 - (history[-1] / 100.0) ** 2,
        },
        "stopping": {
            "stop_reason": "max_atoms",
            "n_iterations": len(token_ids),
            "residual_norm_history": history,
        },
        "algorithm_settings": {"k": 3, "normalize_atoms": True},
        "dictionary_provenance": {"layer": layer, "n_atoms": 100},
        "cone_signature": {"token_ids": sorted(t for t, _, _ in effective),
                           "digest": "sha256:0011223344556677"},
    }


@pytest.fixture
def fake_run(tmp_path):
    """Minimal jspace-shaped run: 1 example, 2 positions, 2 layers, k=10."""
    run = tmp_path / "jspace_fake"
    (run / "artifacts" / "cones").mkdir(parents=True)
    slug, prompt_hash = "factual-example", "00112233aabbccdd"
    capture = []
    for position in (-2, -1):
        capture.append({
            "slug": slug, "category": "factual", "format": "plain",
            "position": position, "prompt_hash": prompt_hash, "seq_len": 7,
            "input_token_id": 11, "input_token": " is",
            "model_top1_id": 42, "model_top1_token": " answer",
        })
    meta = {
        "written_utc": "2026-07-16T17:09:59+00:00",
        "run_id": "jspace_fake",
        "mode": "jspace_pursuit",
        "config": {
            "model": {"repo_id": "google/gemma-4-E4B-it",
                      "revision": MODEL_REVISION},
            "decomposition": {"layers": [35, 38], "k_values": [10]},
        },
        "config_fingerprint": "sha256:" + "ab" * 32,
        "lens_verification": {"file_sha256": "sha256:" + "cd" * 32},
        "load_info": {"model_revision": MODEL_REVISION},
        "capture_meta": capture,
    }
    (run / "run_metadata.json").write_text(
        json.dumps(meta), encoding="utf-8")
    for layer in (35, 38):
        cones = []
        for position in (-2, -1):
            token_ids = [42, 7, 3] if layer == 38 else [7, 3, 5]
            cones.append(_cone(slug, prompt_hash, layer, position,
                               token_ids, [5.0, 2.0, 0.0]))
        (run / "artifacts" / "cones" / f"cones_layer{layer}_k10.json").write_text(
            json.dumps(cones), encoding="utf-8")
    transitions = []
    for position in (-2, -1):
        transitions.append({
            "schema": "jlens.cones.transition.v1",
            "prompt_hash": prompt_hash, "prompt_slug": slug, "format": "plain",
            "position": position, "layer_from": 35, "layer_to": 38,
            "active_set_overlap": {"intersection_size": 1, "union_size": 3,
                                   "jaccard": 1 / 3},
            "weighted_similarity": 0.5,
            "explained_fraction_from": 0.01, "explained_fraction_to": 0.02,
            "delta_explained_fraction": 0.01,
            "concentration_from": {}, "concentration_to": {},
            "entered_token_ids": [42], "entered_labels": ["tok42"],
            "exited_token_ids": [5], "exited_labels": ["tok5"],
        })
    (run / "artifacts" / "trajectories_k10.json").write_text(
        json.dumps(transitions), encoding="utf-8")
    eval_results = {
        "results": {
            "top_k": 10,
            "examples": [{
                "slug": slug, "positions": [-2, -1],
                "layers": {
                    "35": {"jlens": {"topk_overlap_with_model": 0.1,
                                     "rank_of_model_top1": [4, 1]}},
                    "38": {"jlens": {"topk_overlap_with_model": 0.2,
                                     "rank_of_model_top1": [2, 0]}},
                },
            }],
        },
    }
    (run / "artifacts" / "eval_v2_results.json").write_text(
        json.dumps(eval_results), encoding="utf-8")
    return run


def _hash_tree(root: Path) -> dict:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*")) if p.is_file()
    }


# --------------------------------------------------------------- building


def test_build_is_deterministic_and_schema_valid(fake_run):
    bundle_a, warnings_a = build_text_bundle(str(fake_run))
    bundle_b, _ = build_text_bundle(str(fake_run))
    assert canonical_json(bundle_a) == canonical_json(bundle_b)
    assert warnings_a == ["no prompts file given; prompt_text will be null"]

    import jsonschema

    schema = json.loads(Path(SCHEMA_PATH).read_text(encoding="utf-8"))
    jsonschema.validate(instance=bundle_a, schema=schema)

    assert len(bundle_a["examples"]) == 1
    assert len(bundle_a["cones"]) == 4          # 2 positions x 2 layers
    assert len(bundle_a["pursuit_traces"]) == 4
    assert len(bundle_a["trajectories"]) == 2
    assert bundle_a["provenance"]["created_utc"] == "2026-07-16T17:09:59+00:00"


def test_stable_ids_and_record_content(fake_run):
    bundle, _ = build_text_bundle(str(fake_run))
    example = bundle["examples"][0]
    assert example["example_id"] == "text:factual-example:00112233aabbccdd"
    assert example["example_id"] == example_id("text", "factual-example",
                                               "00112233aabbccdd")
    assert example["strength"]["tag"] == "strong"  # layer-38 rank 0 at -1

    cone38 = next(c for c in bundle["cones"]
                  if c["layer"] == 38 and c["position"] == -1)
    atoms = cone38["selected_atoms"]
    assert atoms[0]["token_id"] == 42 and atoms[0]["is_output_token"]
    assert atoms[2]["coefficient"] == 0.0 and not atoms[2]["is_effective"]
    assert atoms[0]["coefficient_share"] == pytest.approx(5.0 / 7.0)

    trace = next(t for t in bundle["pursuit_traces"]
                 if t["layer"] == 38 and t["position"] == -1)
    assert trace["per_step_coefficients_available"] is False
    assert [s["added_token_id"] for s in trace["steps"]] == [42, 7, 3]
    assert trace["steps"][0]["coefficients_after"] is None
    assert trace["steps"][2]["final_coefficient_zero"] is True
    assert trace["steps"][-1]["residual_norm"] == pytest.approx(94.0)

    record = next(r for r in bundle["layer_records"]
                  if r["layer"] == 38 and r["position"] == -1)
    assert record["rank_of_model_top1"] == 0
    assert record["jlens_topk"] is None  # not persisted; never fabricated

    trajectory = next(t for t in bundle["trajectories"] if t["position"] == -1)
    assert trajectory["output_token_persistence"] == {"in_from": False,
                                                      "in_to": True}
    # L35 effective set is {7, 3}; L38 effective set is {42, 7} (token 3's
    # coefficient is zero there) — so only token 7 is retained.
    assert {a["token_id"] for a in trajectory["retained_atoms"]} == {7}


def test_absolute_paths_stripped_and_checker_raises(fake_run):
    bundle, _ = build_text_bundle(str(fake_run))
    rendered = canonical_json(bundle)
    assert "/content/" not in rendered
    assert "C:\\\\Users" not in rendered and "C:/Users" not in rendered

    with pytest.raises(ExportError, match="absolute local path"):
        assert_no_absolute_paths({"x": [{"y": "/content/drive/foo"}]})
    with pytest.raises(ExportError, match="absolute local path"):
        assert_no_absolute_paths("C:\\Users\\someone\\secret.json")
    assert_no_absolute_paths({"ok": "data/measured/assets/a.png"})


def test_subset_selection_and_unknown_slug_rejection(fake_run):
    bundle, _ = build_text_bundle(str(fake_run), slugs=["factual-example"])
    assert len(bundle["examples"]) == 1
    with pytest.raises(ExportError, match="slugs not present"):
        build_text_bundle(str(fake_run), slugs=["missing-slug"])
    with pytest.raises(ExportError, match="layers"):
        build_text_bundle(str(fake_run), layers=[14])


def test_malformed_artifacts_rejected(fake_run, tmp_path):
    cone_path = fake_run / "artifacts" / "cones" / "cones_layer38_k10.json"
    records = json.loads(cone_path.read_text(encoding="utf-8"))
    records[0]["schema"] = "wrong.schema.v9"
    cone_path.write_text(json.dumps(records), encoding="utf-8")
    with pytest.raises(ExportError, match="unexpected schema"):
        build_text_bundle(str(fake_run))

    with pytest.raises(ExportError, match="not a run directory"):
        load_run_artifacts(str(tmp_path))

    meta_path = fake_run / "run_metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["mode"] = "pilot"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ExportError, match="expected mode"):
        build_text_bundle(str(fake_run))


def test_missing_optional_artifacts_warn_but_build(fake_run):
    (fake_run / "artifacts" / "trajectories_k10.json").unlink()
    (fake_run / "artifacts" / "eval_v2_results.json").unlink()
    bundle, warnings = build_text_bundle(str(fake_run))
    assert bundle["trajectories"] == []
    record = bundle["layer_records"][0]
    assert record["rank_of_model_top1"] is None
    assert any("trajectories" in w for w in warnings)
    assert any("eval_v2_results" in w for w in warnings)


def test_sources_not_mutated(fake_run):
    before = _hash_tree(fake_run)
    build_text_bundle(str(fake_run))
    assert _hash_tree(fake_run) == before


def test_write_bundle_round_trip(fake_run, tmp_path):
    bundle, _ = build_text_bundle(str(fake_run))
    out = tmp_path / "bundle.json"
    fingerprint_a = write_bundle(bundle, str(out), schema_path=SCHEMA_PATH)
    fingerprint_b = write_bundle(bundle, str(out), schema_path=SCHEMA_PATH)
    assert fingerprint_a == fingerprint_b
    validate_bundle(json.loads(out.read_text(encoding="utf-8")), SCHEMA_PATH)


# ---------------------------------------------------------------- merging


def _causal_bundle(status="measured"):
    provenance = make_provenance(
        source_run_ids=["causal_fake"],
        model_repo_id="google/gemma-4-E4B-it",
        model_revision=MODEL_REVISION,
        created_utc="2026-07-17T00:00:00+00:00",
        data_status=status,
        modalities_present=["text"],
    )
    raw = {
        "schema": "jlens.interventions.record.v1",
        "condition_id": "cond_0123456789abcdef",
        "example_id": "text:factual-example:00112233aabbccdd",
        "layer": 38, "position": -1,
        "target_kind": "output_atom_contribution",
        "atom_token_id": 42, "atom_label": " answer", "atom_coefficient": 5.0,
        "multiplier": 1.0, "status": status, "norm_preserving": False,
        "delta_norm": 3.0, "activation_norm": 100.0,
        "delta_to_activation_ratio": 0.03,
        "resolved_position": 6, "control_family": None,
        "matched_target_condition_id": None, "random_seed": None,
        "completion_before": None, "completion_after": None,
        "provenance": {"run_id": "causal_fake"},
        "target_token_id": 42, "target_logit_delta": 1.5,
    }
    return assemble_bundle(
        provenance=provenance,
        causal_records=[intervention_to_causal_record(raw)],
        causal_baseline_parity={"worst_max_abs_logit_diff": 0.001},
    )


def test_causal_bundle_merging(fake_run):
    base, _ = build_text_bundle(str(fake_run))
    merged = merge_bundles(base, _causal_bundle())
    assert len(merged["causal_records"]) == 1
    record = merged["causal_records"][0]
    assert record["condition_id"] == "cond_0123456789abcdef"
    assert "atom_coefficient" not in record  # backend-internal field dropped
    assert record["status"] == "measured"
    assert merged["causal_baseline_parity"]["worst_max_abs_logit_diff"] == 0.001
    assert merged["provenance"]["data_status"] == "measured"
    assert "causal_fake" in merged["provenance"]["source_run_ids"]
    # Base is not mutated.
    assert base["causal_records"] == []

    # Merging the same records again dedupes by condition_id.
    again = merge_bundles(merged, _causal_bundle())
    assert len(again["causal_records"]) == 1


def test_fixture_merge_degrades_bundle_status_but_extra_wins_records(fake_run):
    base, _ = build_text_bundle(str(fake_run))
    merged = merge_bundles(base, _causal_bundle(status="synthetic_fixture"))
    assert merged["provenance"]["data_status"] == "synthetic_fixture"
    assert merged["causal_records"][0]["status"] == "synthetic_fixture"
    # A later measured merge replaces the fixture record (same condition_id).
    remeasured = merge_bundles(merged, _causal_bundle(status="measured"))
    assert remeasured["causal_records"][0]["status"] == "measured"


def test_multimodal_bundle_merging(fake_run):
    base, _ = build_text_bundle(str(fake_run))
    fixture_path = REPO_ROOT / "explorer" / "public" / "data" / "fixtures" / "multimodal_fixture.json"
    multimodal = json.loads(fixture_path.read_text(encoding="utf-8"))
    merged = merge_bundles(base, multimodal)
    modalities = merged["provenance"]["modalities_present"]
    assert "text" in modalities and "image_text" in modalities and "audio_text" in modalities
    assert any(e["modality"] == "audio_text" for e in merged["examples"])
    import jsonschema

    schema = json.loads(Path(SCHEMA_PATH).read_text(encoding="utf-8"))
    jsonschema.validate(instance=merged, schema=schema)

    with pytest.raises(ExportError, match="cannot merge"):
        merge_bundles(base, {"schema": "something.else"})


# ------------------------------------------------- committed artifacts


def test_committed_bundles_validate_against_schema():
    import jsonschema

    schema = json.loads(Path(SCHEMA_PATH).read_text(encoding="utf-8"))
    data_dir = REPO_ROOT / "explorer" / "public" / "data"
    paths = [data_dir / "text_demo.json",
             data_dir / "fixtures" / "causal_fixture.json",
             data_dir / "fixtures" / "multimodal_fixture.json"]
    for path in paths:
        assert path.is_file(), f"missing committed bundle {path}"
        jsonschema.validate(
            instance=json.loads(path.read_text(encoding="utf-8")), schema=schema)

    for name in ("causal", "multimodal"):
        fixture = json.loads(
            (data_dir / "fixtures" / f"{name}_fixture.json").read_text(encoding="utf-8"))
        assert fixture["provenance"]["data_status"] == "synthetic_fixture"


@pytest.mark.skipif(not REAL_RUN.is_dir(), reason="completed jspace run not present")
def test_real_run_export_matches_committed_demo_bundle():
    """When the real run directory exists locally, re-exporting the demo set
    must reproduce the committed text_demo.json byte-for-byte."""
    bundle, _ = build_text_bundle(
        str(REAL_RUN),
        prompts_path=str(REPO_ROOT / "configs" / "prompts" / "eval_prompts_v2.json"),
        analysis_dir=str(REPO_ROOT / "reports" / REAL_RUN.name),
        slugs=list(DEFAULT_DEMO_SLUGS),
        implementation_commit=_committed_implementation_commit(),
    )
    committed = (REPO_ROOT / "explorer" / "public" / "data" / "text_demo.json")
    assert canonical_json(bundle) == committed.read_text(encoding="utf-8")


def _committed_implementation_commit():
    committed = json.loads(
        (REPO_ROOT / "explorer" / "public" / "data" / "text_demo.json")
        .read_text(encoding="utf-8"))
    return committed["provenance"]["implementation_commit"]
