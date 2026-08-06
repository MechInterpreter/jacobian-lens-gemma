# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Amending a completed run's report without invalidating the completed run.

The point of separation being tested: the units were *generated* under one
contract and *reported* under another, and only the second one changed. So

* the raw-generation fingerprint must be untouched, and every completed unit
  must still resume under it;
* the amended artifact must bind to that fingerprint **and** to a digest over
  the exact units it read, so it cannot silently describe a different
  measurement;
* the original report must survive.

The fixture is a miniature of the observed run: three focal concepts, one of
which failed the spoken-audio capability gate at 5/8.
"""

import json
from pathlib import Path

import pytest

from jlens.mmpilot.admissibility import CLAIM_ADMISSIBILITY_RULE_VERSION
from jlens.mmpilot.amend import (
    AMENDED_REPORT_NAME,
    AMENDED_SUMMARY_NAME,
    POSTPROCESSING_VERSION,
    AmendedReportBindingError,
    AmendedReportInputMissing,
    build_amended_report,
    source_unit_digest,
    verify_amended_binding,
    write_amended_report,
)
from jlens.mmpilot.store import RunFingerprint, UnitStore
from jlens.mmpilot.tri_modal import (
    ALL_PAIRS,
    AUDIO_PAIRS,
    TRANSFER_SUPPORTED,
    TriModalThresholds,
    audio_capability_verdict,
    representational_transfer_verdict,
)

MODALITIES = ("text", "image", "spoken_audio")
THRESHOLDS = TriModalThresholds(
    required_positive_images_per_cell=2, required_negative_images_per_cell=2
)
FOCAL = ["zebra", "cat", "toilet"]
COUNTS = {
    "zebra": {"text": 8, "image": 8, "spoken_audio": 5},
    "cat": {"text": 8, "image": 7, "spoken_audio": 8},
    "toilet": {"text": 8, "image": 8, "spoken_audio": 7},
}
CELLS = {
    "zebra": ["spoken_audio->text", "spoken_audio->image"],
    "cat": ["spoken_audio->image"],
    "toilet": list(AUDIO_PAIRS),
}


def _fingerprint(**overrides):
    kwargs = dict(
        mode="native_audio_transfer",
        model_repo_id="google/gemma-4-E4B-it",
        model_revision="fa62d88d",
        processor_revision="fa62d88d",
        layers=(35, 38, 40),
        lens_checksum="sha256:lens",
        manifest_checksum="sha256:manifest",
        split_id="spokencoco-native-audio-v1",
    )
    kwargs.update(overrides)
    return RunFingerprint(**kwargs)


def _capability_summary():
    per_concept = {
        concept: {
            modality: {
                "n": 8,
                "n_correct": n_correct,
                "accuracy": n_correct / 8,
                "passed": n_correct / 8 >= 0.7,
            }
            for modality, n_correct in per_modality.items()
        }
        for concept, per_modality in COUNTS.items()
    }
    return {
        "threshold": 0.7,
        "modalities_evaluated": list(MODALITIES),
        "per_concept": per_concept,
        "retained_concepts": ["cat", "toilet"],
        "text_image_retained_concepts": sorted(per_concept),
        "n_records": 24,
    }


def _representational_report(layer=35):
    def entry():
        return {
            "n_sources": 8,
            "n_targets": 8,
            "jspace_retrieval": {"top1_accuracy": 0.9, "mrr": 0.9, "n_queries": 8},
            "raw_residual_retrieval": {"top1_accuracy": 0.2},
            "jspace_separation": {"gap": 0.4},
            "raw_residual_separation": {"gap": 0.1},
            "jspace_support_overlap": {"gap": 0.3},
            "shuffled_control": {"mean_top1_accuracy": 0.1, "p95_top1_accuracy": 0.3},
            "exclusions": {
                "eligible_targets": 8,
                "n_excluded_same_group": 1,
                "n_excluded_same_image_different_group": 0,
            },
            "n_distinct_source_images": 8,
            "n_distinct_target_images": 8,
        }

    return {"layer": layer, "pairs": {pair: entry() for pair in ALL_PAIRS}}


def _image_level(layer=35):
    rows = []
    for concept, pairs in CELLS.items():
        for pair in pairs:
            source, target = pair.split("->")
            base = {
                "concept": concept,
                "source_modality": source,
                "target_modality": target,
                "pair": pair,
                "off_diagonal": True,
                "layer": layer,
                "alpha": 0.5,
                "n": 8,
                "n_distinct_images": 8,
                "n_positive_images": 4,
                "n_negative_images": 4,
                "fraction_expected_sign": 1.0,
                "mean_abs_unrelated_change": 0.01,
                "mean_activation_norm_ratio": 1.0,
                "n_prediction_changes": 0,
            }
            for kind, effect in (
                ("source_concept", 0.5),
                ("random_norm_matched", 0.01),
                ("unrelated_concept", 0.02),
                ("raw_residual_difference", 0.05),
            ):
                rows.append(
                    {
                        **base,
                        "control_kind": kind,
                        "mean_signed_target_effect": effect,
                        "mean_signed_margin_effect": effect,
                    }
                )
    return {"rows": rows}


@pytest.fixture
def completed_run(tmp_path):
    """A finished run directory: capability, interventions, and stored metrics."""
    root = tmp_path / "mmaudio_run"
    store = UnitStore(root, _fingerprint())
    store.open()

    capability = _capability_summary()
    for concept, per_modality in COUNTS.items():
        for modality, n_correct in per_modality.items():
            for index in range(8):
                store.save(
                    "capability",
                    f"{concept}__{modality}__{index}",
                    {
                        "concept": concept,
                        "modality": modality,
                        "correct": index < n_correct,
                    },
                )
    for row_index, row in enumerate(_image_level()["rows"]):
        store.save("intervention", f"L35__{row_index}", row)

    store.save(
        "metric",
        "audio_capability_verdict",
        audio_capability_verdict(
            capability,
            selected_concepts=sorted(COUNTS),
            modalities=MODALITIES,
            thresholds=THRESHOLDS,
        ),
    )
    store.save(
        "metric",
        "representational_transfer_verdict",
        representational_transfer_verdict(
            {35: _representational_report()},
            thresholds=THRESHOLDS,
            primary_layer=35,
            capability=capability,
            pooled_concepts=sorted(COUNTS),
        ),
    )
    store.save("metric", "interventions_image_level_L35", _image_level())
    (root / "native_audio_transfer_report.md").write_text(
        "# original report — L35_CAUSAL_TRANSFER: SUPPORTED\n", encoding="utf-8"
    )
    return root


def _build(root, **overrides):
    store = UnitStore(root, _fingerprint())
    store.open()
    kwargs = dict(
        primary_layer=35,
        replication_layers=(38, 40),
        focal_concepts=FOCAL,
        thresholds=THRESHOLDS,
        original_report_path=root / "native_audio_transfer_report.md",
        invariance={"passed": True, "per_modality": dict.fromkeys(MODALITIES, {})},
    )
    kwargs.update(overrides)
    return store, build_amended_report(store, **kwargs)


# ------------------------------------ 11: bound to fingerprint and checksums


def test_the_amended_report_binds_to_the_run_fingerprint_and_its_source_units(
    completed_run,
):
    store, amended = _build(completed_run)
    binding = amended["binding"]
    assert binding["run_fingerprint_digest"] == store.fingerprint.digest
    assert binding["raw_generation_fingerprint_unchanged"] is True
    assert binding["postprocessing_version"] == POSTPROCESSING_VERSION
    assert binding["admissibility_rule_version"] == CLAIM_ADMISSIBILITY_RULE_VERSION
    assert binding["model_loaded"] is False
    assert binding["units_written"] is False

    units = binding["source_units"]
    assert units["per_stage"]["capability"]["n_units"] == 3 * 3 * 8
    assert units["per_stage"]["intervention"]["n_units"] == len(_image_level()["rows"])
    assert units["combined_digest"] == source_unit_digest(store)["combined_digest"]
    assert verify_amended_binding(amended["summary"], store)["bound"] is True


def test_the_amended_verdict_is_recomputed_and_excludes_the_ineligible_concept(
    completed_run,
):
    _, amended = _build(completed_run)
    causal = amended["verdicts"]["primary_causal"]
    assert causal["verdict"] == TRANSFER_SUPPORTED
    assert causal["capability_admissibility"]["excluded_concept_names"] == ["zebra"]
    assert not any(
        cell["concept"] == "zebra"
        for cell in causal["audio_cells_supporting_a_claim"]
    )
    # zebra's measurements are still there, in full.
    assert {
        (c["concept"], c["pair"])
        for c in causal["audio_cells_measured_but_inadmissible"]
    } == {("zebra", "spoken_audio->text"), ("zebra", "spoken_audio->image")}
    assert causal["concepts_transferring_both_audio_directions"] == ["toilet"]


def test_the_amended_artifacts_are_written_beside_the_original_never_over_it(
    completed_run,
):
    _, amended = _build(completed_run)
    original = completed_run / "native_audio_transfer_report.md"
    before = original.read_text(encoding="utf-8")

    paths = write_amended_report(amended, run_dir=completed_run)

    assert paths["report"].name == AMENDED_REPORT_NAME
    assert paths["summary"].name == AMENDED_SUMMARY_NAME
    assert original.read_text(encoding="utf-8") == before
    report = paths["report"].read_text(encoding="utf-8")
    assert "AMENDED REPORT — capability-filtered" in report
    assert POSTPROCESSING_VERSION in report
    assert "No model was loaded and no unit was recomputed" in report
    assert "CAPABILITY_INELIGIBLE" in report
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert summary["binding"]["run_fingerprint_digest"]

    with pytest.raises(FileExistsError, match="refusing to write"):
        write_amended_report(
            amended,
            run_dir=completed_run,
            report_name="native_audio_transfer_report.md",
        )


# ---------------------------------------- 12: changed units refuse the reuse


def test_a_changed_intervention_unit_refuses_the_amended_report(completed_run):
    store, amended = _build(completed_run)
    write_amended_report(amended, run_dir=completed_run)

    # Someone edits one measurement. The amended report no longer describes
    # this run.
    unit = store.unit_path("intervention", "L35__0")
    record = json.loads(unit.read_text(encoding="utf-8"))
    record["payload"]["mean_signed_target_effect"] = 99.0
    record["unit_checksum"] = "sha256:" + "0" * 64
    unit.write_text(json.dumps(record), encoding="utf-8")

    fresh = UnitStore(completed_run, _fingerprint())
    fresh.open()
    with pytest.raises(AmendedReportBindingError, match="intervention units"):
        verify_amended_binding(amended["summary"], fresh)


def test_an_added_capability_unit_refuses_the_amended_report(completed_run):
    store, amended = _build(completed_run)
    store.save("capability", "zebra__spoken_audio__99", {"concept": "zebra"})
    fresh = UnitStore(completed_run, _fingerprint())
    fresh.open()
    with pytest.raises(AmendedReportBindingError, match="capability units"):
        verify_amended_binding(amended["summary"], fresh)


def test_a_different_run_fingerprint_refuses_the_amended_report(completed_run):
    _, amended = _build(completed_run)
    other = UnitStore(completed_run, _fingerprint(lens_checksum="sha256:other"))
    with pytest.raises(AmendedReportBindingError, match="run fingerprint"):
        verify_amended_binding(amended["summary"], other)


def test_a_missing_stored_metric_is_refused_rather_than_guessed(completed_run):
    (completed_run / "units" / "metric" / "audio_capability_verdict.json").unlink()
    with pytest.raises(AmendedReportInputMissing, match="audio_capability_verdict"):
        _build(completed_run)


# --------------------------------- 13: the completed units are still resumable


def test_the_completed_units_remain_resumable_after_the_report_is_amended(
    completed_run,
):
    store, amended = _build(completed_run)
    digest_before = store.fingerprint.digest
    counts_before = store.status_report()["completed_units"]
    write_amended_report(amended, run_dir=completed_run)

    # A later run, constructing the fingerprint exactly as the original did.
    resumed = UnitStore(completed_run, _fingerprint())
    assert resumed.open() == "resuming"
    assert resumed.fingerprint.digest == digest_before
    assert resumed.status_report()["completed_units"] == counts_before
    assert resumed.has("capability", "zebra__spoken_audio__0")
    assert resumed.has("intervention", "L35__0")
    assert resumed.load("metric", "interventions_image_level_L35") is not None
    assert not resumed.invalid_units


def test_amending_writes_no_unit_and_touches_no_unit(completed_run):
    store = UnitStore(completed_run, _fingerprint())
    store.open()
    before = {
        stage: store.unit_checksums(stage)
        for stage in ("capability", "intervention", "metric")
    }
    _, amended = _build(completed_run)
    write_amended_report(amended, run_dir=completed_run)
    after = {
        stage: store.unit_checksums(stage)
        for stage in ("capability", "intervention", "metric")
    }
    assert after == before


def test_the_amended_report_does_not_change_the_generation_fingerprint(completed_run):
    # The reporting rule is versioned separately, on purpose: binding it into
    # the generation fingerprint would refuse a completed run because a report
    # bug was fixed.
    store, amended = _build(completed_run)
    stored = json.loads(
        (Path(completed_run) / "fingerprint.json").read_text(encoding="utf-8")
    )
    stored.pop("written_utc", None)
    assert "admissibility" not in json.dumps(stored)
    assert "postprocessing" not in json.dumps(stored)
    assert amended["binding"]["run_fingerprint_digest"] == store.fingerprint.digest
    assert amended["summary"]["admissibility_rule"]["rule_checksum"].startswith(
        "sha256:"
    )
