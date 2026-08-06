# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Reopening a completed run and amending its report, with nothing else in memory.

The bug: the amendment cell opened with ``if STORE is None``, and ``STORE`` only
exists inside the Stage-A model path. The documented CPU procedure — model
stages off, run sections 1–7 and 18b — therefore died on ``NameError``, while the
setup cells had meanwhile created an empty timestamped ``mmaudio_*`` directory
the amendment had no use for.

What is checked here is the contract that replaced it: one *named* completed run,
a fingerprint that must re-derive its own digest and match a pin the operator
wrote down, report metadata read back from the run's own summary rather than
inferred from the runtime, and an amendment that is idempotent — reused when it
still binds, refused when it does not, and never written over an original.

The fixture is a miniature of the observed run: three focal concepts, one of
which failed the spoken-audio capability gate at 5/8.
"""

import hashlib
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from jlens.mmpilot.amend import (
    AMENDED_REPORT_NAME,
    AMENDED_SUMMARY_NAME,
    POSTPROCESSING_VERSION,
    AmendedReportBindingError,
    AmendedReportInputMissing,
    build_amended_report,
    write_amended_report,
)
from jlens.mmpilot.amend_open import (
    MODEL_AND_CAUSAL_SWITCHES,
    ORIGINAL_REPORT_NAME,
    ORIGINAL_SUMMARY_NAME,
    AmendedArtifactsTorn,
    AmendmentModeError,
    ExistingRunNotFound,
    FingerprintMismatch,
    amend_or_reuse,
    amendment_mode_run_dir,
    assert_amendment_mode_exclusive,
    fingerprint_from_dict,
    format_amendment_inputs,
    load_run_fingerprint,
    open_existing_store,
    restore_report_metadata,
)
from jlens.mmpilot.store import RunFingerprint, UnitStore
from jlens.mmpilot.tri_modal import (
    ALL_PAIRS,
    AUDIO_PAIRS,
    THREE_MODALITY_GO,
    TRANSFER_SUPPORTED,
    TriModalThresholds,
    audio_capability_verdict,
    overall_verdict,
    representational_transfer_verdict,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
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


def _fingerprint(**overrides) -> RunFingerprint:
    kwargs = dict(
        mode="native_audio_transfer",
        model_repo_id="google/gemma-4-E4B-it",
        model_revision="fa62d88d",
        processor_revision="fa62d88d",
        layers=(35, 38, 40),
        lens_checksum="sha256:lens",
        manifest_checksum="sha256:manifest",
        split_id="spokencoco-native-audio-v1",
        intervention_config={
            "alphas": [0.0, 0.25, 0.5, 1.0],
            "primary_causal_layer": 35,
            "replication_layers": [38, 40],
            "off_diagonal_causal_only": True,
        },
        selection_config={
            "focal_concepts": list(FOCAL),
            "selected_concepts": [*FOCAL, "bus", "clock", "dog"],
        },
        extra={"modalities": list(MODALITIES)},
    )
    kwargs.update(overrides)
    return RunFingerprint(**kwargs)


def _capability_summary() -> dict:
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


def _representational_report(layer: int = 35) -> dict:
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


def _image_level(layer: int = 35) -> dict:
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


INVARIANCE = {
    "passed": True,
    "modalities": list(MODALITIES),
    "per_modality": {modality: {"passed": True} for modality in MODALITIES},
}
LENS_REPORT = {
    "layers": [35, 38, 40],
    "combined_checksum": "sha256:lens",
    "checksums": {"35": "sha256:a", "38": "sha256:b", "40": "sha256:c"},
}
AUDIO_PROTOCOL = {
    "protocol_version": "jlens.mmpilot.native_spoken_audio.v1",
    "protocol_fingerprint": "sha256:audio",
    "dynamic_placeholder_count": True,
}


def _write_completed_run(root: Path, *, fingerprint: RunFingerprint | None = None) -> Path:
    """A finished run: units, stored metrics, the original report and summary."""
    fingerprint = fingerprint or _fingerprint()
    store = UnitStore(root, fingerprint)
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

    capability_verdict = audio_capability_verdict(
        capability,
        selected_concepts=sorted(COUNTS),
        modalities=MODALITIES,
        thresholds=THRESHOLDS,
    )
    representational = representational_transfer_verdict(
        {35: _representational_report()},
        thresholds=THRESHOLDS,
        primary_layer=35,
        capability=capability,
        pooled_concepts=sorted(COUNTS),
    )
    store.save("metric", "audio_capability_verdict", capability_verdict)
    store.save("metric", "representational_transfer_verdict", representational)
    store.save("metric", "interventions_image_level_L35", _image_level())
    store.save("metric", "invariance", INVARIANCE)

    (root / ORIGINAL_REPORT_NAME).write_text(
        "# original report — L35_CAUSAL_TRANSFER: SUPPORTED\n", encoding="utf-8"
    )
    # Section 18's own summary, which is what the amendment reads its report-only
    # metadata back from.
    overall = overall_verdict(
        capability=capability_verdict,
        representational=representational,
        primary_causal=None,
        replication={"verdict": "NOT_EVALUATED"},
        invariance=INVARIANCE,
        blocked_modalities=[],
        thresholds=THRESHOLDS,
    )
    (root / ORIGINAL_SUMMARY_NAME).write_text(
        json.dumps(
            {
                "run_dir": str(root),
                "mode": fingerprint.mode,
                "fingerprint_digest": fingerprint.digest,
                "selection_fingerprint": dict(fingerprint.selection_config),
                "lens_validation": LENS_REPORT,
                "audio_protocol": AUDIO_PROTOCOL,
                "invariance": INVARIANCE,
                "verdicts": {
                    "A_audio_capability": capability_verdict,
                    "B_representational_transfer": representational,
                    "C_primary_causal": None,
                    "D_replication": {"verdict": "NOT_EVALUATED"},
                    "E_overall": overall,
                },
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return root


@pytest.fixture
def completed_run(tmp_path) -> Path:
    return _write_completed_run(tmp_path / "mmaudio_native_audio_transfer_20260806T001229")


@pytest.fixture
def expected_fingerprint() -> str:
    return _fingerprint().digest


def _open(root, expected):
    return open_existing_store(root, expected_fingerprint=expected)


def _amend(root, expected, **overrides):
    store = _open(root, expected)
    inputs = restore_report_metadata(store)
    inputs.pop("summary_path")
    inputs.pop("restored_from")
    inputs.pop("restored_fields")
    inputs.update(overrides)
    return store, amend_or_reuse(store, **inputs)


def _tree(root: Path) -> dict[str, str]:
    """Every file under ``root``, by relative path, with its content checksum."""
    return {
        str(path.relative_to(root)).replace("\\", "/"): "sha256:"
        + hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


# ------------------------------------------------ 3: faithful reconstruction


def test_the_stored_fingerprint_is_reconstructed_exactly(completed_run):
    stored = json.loads((completed_run / "fingerprint.json").read_text(encoding="utf-8"))
    fingerprint, payload = load_run_fingerprint(completed_run)

    assert fingerprint == _fingerprint()
    assert fingerprint.digest == _fingerprint().digest
    assert payload == stored
    # JSON has no tuples; the reconstruction restores the declared one, which is
    # what makes the digest come back identical rather than merely similar.
    assert fingerprint.layers == (35, 38, 40)
    assert isinstance(fingerprint.layers, tuple)
    assert fingerprint.to_dict()["layers"] == [35, 38, 40]


def test_field_order_in_the_stored_file_does_not_matter(completed_run):
    path = completed_run / "fingerprint.json"
    stored = json.loads(path.read_text(encoding="utf-8"))
    shuffled = dict(reversed(list(stored.items())))
    assert list(shuffled) != list(stored)
    path.write_text(json.dumps(shuffled, indent=2), encoding="utf-8")

    fingerprint, _ = load_run_fingerprint(completed_run)
    assert fingerprint.digest == _fingerprint().digest


def test_only_transport_metadata_is_stripped_before_the_digest(completed_run):
    path = completed_run / "fingerprint.json"
    stored = json.loads(path.read_text(encoding="utf-8"))
    stored["written_utc"] = "2026-08-06T00:12:29+00:00"
    path.write_text(json.dumps(stored, indent=2), encoding="utf-8")

    fingerprint, _ = load_run_fingerprint(completed_run)
    assert fingerprint.digest == _fingerprint().digest
    assert _open(completed_run, fingerprint.digest).status == "resuming"


def test_a_fingerprint_field_this_code_does_not_know_is_refused(completed_run):
    """Dropping it would produce a digest that matched only by omission."""
    path = completed_run / "fingerprint.json"
    stored = json.loads(path.read_text(encoding="utf-8"))
    stored["future_binding"] = {"something": "that bound the run"}
    path.write_text(json.dumps(stored, indent=2), encoding="utf-8")

    with pytest.raises(FingerprintMismatch, match="unknown field"):
        load_run_fingerprint(completed_run)


def test_a_fingerprint_missing_a_required_field_is_refused(completed_run):
    path = completed_run / "fingerprint.json"
    stored = json.loads(path.read_text(encoding="utf-8"))
    del stored["lens_checksum"]
    path.write_text(json.dumps(stored, indent=2), encoding="utf-8")

    with pytest.raises(FingerprintMismatch, match="missing required field"):
        load_run_fingerprint(completed_run)


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda p: p.update(layers=["35", "38", "40"]), id="layers-as-str"),
        pytest.param(lambda p: p.update(selection_config={}), id="empty-selection"),
        pytest.param(lambda p: p.update(intervention_config=None), id="null-config"),
    ],
)
def test_a_fingerprint_that_does_not_re_derive_its_digest_is_refused(tmp_path, mutate):
    """The check that makes "faithful" checkable rather than asserted.

    Each payload here still *reconstructs* into a plausible RunFingerprint — and
    then digests to something the bytes on disk never did. Accepting it would
    mean every later fingerprint comparison checked the wrong object.
    """
    root = tmp_path / "run"
    root.mkdir()
    payload = _fingerprint().to_dict()
    mutate(payload)
    (root / "fingerprint.json").write_text(json.dumps(payload), encoding="utf-8")

    assert fingerprint_from_dict(payload) is not None  # it reconstructs
    with pytest.raises(FingerprintMismatch, match="does not re-derive"):
        load_run_fingerprint(root)


def test_a_fingerprint_file_that_is_not_json_is_refused(completed_run):
    (completed_run / "fingerprint.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(FingerprintMismatch, match="not readable JSON"):
        load_run_fingerprint(completed_run)


# --------------------------------------------------- 4, 5, 6, 8: refusals


def test_a_wrong_expected_fingerprint_is_refused(completed_run):
    with pytest.raises(FingerprintMismatch, match="Refusing to amend a run other"):
        _open(completed_run, "sha256:" + "0" * 64)


def test_a_missing_fingerprint_file_is_refused(completed_run, expected_fingerprint):
    (completed_run / "fingerprint.json").unlink()
    with pytest.raises(ExistingRunNotFound, match="fingerprint.json"):
        _open(completed_run, expected_fingerprint)


def test_a_missing_run_directory_is_refused_and_not_created(tmp_path, expected_fingerprint):
    absent = tmp_path / "never_existed"
    with pytest.raises(ExistingRunNotFound, match="never creates one"):
        _open(absent, expected_fingerprint)
    assert not absent.exists()

    with pytest.raises(ExistingRunNotFound):
        amendment_mode_run_dir(absent)
    assert not absent.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == []


def test_the_run_directory_must_be_named_and_is_never_guessed(tmp_path, expected_fingerprint):
    """No default, no search root, no newest-directory fallback."""
    for older in ("mmaudio_a_20260101T000000", "mmaudio_b_20260806T001229"):
        _write_completed_run(tmp_path / older)

    for empty in (None, "", "   "):
        with pytest.raises(AmendmentModeError, match="AMEND_EXISTING_RUN_DIR"):
            _open(empty, expected_fingerprint)
        with pytest.raises(AmendmentModeError, match="AMEND_EXISTING_RUN_DIR"):
            amendment_mode_run_dir(empty)

    # And the module contains no directory scan to fall back to.
    source = (REPO_ROOT / "jlens" / "mmpilot" / "amend_open.py").read_text(
        encoding="utf-8"
    )
    for forbidden in ("glob(", "iterdir(", "rglob(", "listdir(", "max(", "sorted(runs"):
        assert forbidden not in source, forbidden


def test_the_expected_fingerprint_must_be_given(completed_run):
    for empty in (None, "", "  "):
        with pytest.raises(AmendmentModeError, match="AMEND_EXPECTED_FINGERPRINT"):
            _open(completed_run, empty)


def test_a_directory_with_no_fingerprint_never_becomes_a_started_run(
    tmp_path, expected_fingerprint
):
    root = tmp_path / "empty_but_present"
    root.mkdir()
    with pytest.raises(ExistingRunNotFound):
        _open(root, expected_fingerprint)
    assert sorted(p.name for p in root.iterdir()) == []


def test_the_store_is_opened_resuming(completed_run, expected_fingerprint):
    store = _open(completed_run, expected_fingerprint)
    assert store.status == "resuming"
    assert store.fingerprint.digest == expected_fingerprint
    assert store.root == completed_run


# ------------------------------------------- 9, 10: restored report metadata


def test_the_report_metadata_is_restored_from_the_completed_summary(
    completed_run, expected_fingerprint
):
    store = _open(completed_run, expected_fingerprint)
    restored = restore_report_metadata(store)

    assert restored["lens_report"] == LENS_REPORT
    assert restored["audio_protocol"] == AUDIO_PROTOCOL
    assert restored["invariance"] == INVARIANCE
    assert restored["blocked_modalities"] == []
    # The design comes from the fingerprint the units were produced under.
    assert restored["primary_layer"] == 35
    assert restored["replication_layers"] == (38, 40)
    assert restored["focal_concepts"] == FOCAL
    assert restored["mode"] == "native_audio_transfer"
    assert restored["thresholds"].capability_threshold == THRESHOLDS.capability_threshold
    assert restored["thresholds"].norm_ratio_bounds == THRESHOLDS.norm_ratio_bounds
    assert "restored from the completed run" in format_amendment_inputs(restored)


def test_a_summary_describing_another_run_is_refused(completed_run, expected_fingerprint):
    path = completed_run / ORIGINAL_SUMMARY_NAME
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["fingerprint_digest"] = "sha256:" + "1" * 64
    path.write_text(json.dumps(summary), encoding="utf-8")

    store = _open(completed_run, expected_fingerprint)
    with pytest.raises(FingerprintMismatch, match="describes a different measurement"):
        restore_report_metadata(store)


def test_a_missing_summary_is_refused_rather_than_defaulted(
    completed_run, expected_fingerprint
):
    (completed_run / ORIGINAL_SUMMARY_NAME).unlink()
    store = _open(completed_run, expected_fingerprint)
    with pytest.raises(AmendedReportInputMissing, match="will not invent them"):
        restore_report_metadata(store)


@pytest.mark.parametrize(
    "field",
    ["lens_validation", "audio_protocol", "invariance"],
)
def test_a_report_value_with_no_honest_fallback_is_refused(
    completed_run, expected_fingerprint, field
):
    """An absent invariance gate is not a passed one, and neither is guessed."""
    path = completed_run / ORIGINAL_SUMMARY_NAME
    summary = json.loads(path.read_text(encoding="utf-8"))
    del summary[field]
    path.write_text(json.dumps(summary), encoding="utf-8")

    store = _open(completed_run, expected_fingerprint)
    with pytest.raises(AmendedReportInputMissing, match=field):
        restore_report_metadata(store)


def test_an_absent_blocked_modality_list_is_not_read_as_an_empty_one(
    completed_run, expected_fingerprint
):
    path = completed_run / ORIGINAL_SUMMARY_NAME
    summary = json.loads(path.read_text(encoding="utf-8"))
    del summary["verdicts"]["E_overall"]["criteria"]["three_modality_capability"][
        "evidence"
    ]["blocked_modalities"]
    path.write_text(json.dumps(summary), encoding="utf-8")

    store = _open(completed_run, expected_fingerprint)
    with pytest.raises(AmendedReportInputMissing, match="blocked_modalities"):
        restore_report_metadata(store)


# -------------------------------------- 11, 12, 13, 14, 15: idempotent write


def test_the_first_amendment_writes_both_artifacts_together(
    completed_run, expected_fingerprint
):
    store, result = _amend(completed_run, expected_fingerprint)
    assert result["status"] == "written"
    assert result["paths"]["report"].name == AMENDED_REPORT_NAME
    assert result["paths"]["summary"].name == AMENDED_SUMMARY_NAME
    assert result["paths"]["report"].is_file()
    assert result["paths"]["summary"].is_file()
    assert result["verdicts"]["primary_causal"]["verdict"] == TRANSFER_SUPPORTED
    assert result["binding"]["run_fingerprint_digest"] == store.fingerprint.digest
    assert result["binding"]["model_loaded"] is False
    assert result["binding"]["units_written"] is False
    # zebra was measured and is excluded, not replaced.
    admissibility = result["verdicts"]["primary_causal"]["capability_admissibility"]
    assert admissibility["excluded_concept_names"] == ["zebra"]


def test_matching_existing_artifacts_are_reused_and_never_rewritten(
    completed_run, expected_fingerprint
):
    _amend(completed_run, expected_fingerprint)
    before = _tree(completed_run)

    _, second = _amend(completed_run, expected_fingerprint)

    assert second["status"] == "reused"
    assert second["bound"]["bound"] is True
    assert second["verdicts"]["overall"]["verdict"] == THREE_MODALITY_GO
    assert second["binding"]["postprocessing_version"] == POSTPROCESSING_VERSION
    # Byte-for-byte: an amendment that reaches the same conclusion still has a
    # different amended_at_utc, so "unchanged" has to mean the file, not the text.
    assert _tree(completed_run) == before


def test_a_changed_unit_refuses_the_existing_amendment_rather_than_overwriting(
    completed_run, expected_fingerprint
):
    _amend(completed_run, expected_fingerprint)
    before = _tree(completed_run)

    store = _open(completed_run, expected_fingerprint)
    store.save("capability", "zebra__spoken_audio__99", {"concept": "zebra"})

    with pytest.raises(AmendedReportBindingError, match="capability units"):
        _amend(completed_run, expected_fingerprint)
    assert _tree(completed_run)[AMENDED_REPORT_NAME] == before[AMENDED_REPORT_NAME]
    assert _tree(completed_run)[AMENDED_SUMMARY_NAME] == before[AMENDED_SUMMARY_NAME]


def test_a_changed_admissibility_rule_checksum_refuses_the_existing_amendment(
    completed_run, expected_fingerprint
):
    _amend(completed_run, expected_fingerprint)
    path = completed_run / AMENDED_SUMMARY_NAME
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["binding"]["admissibility_rule_checksum"] = "sha256:" + "2" * 64
    path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(AmendedReportBindingError, match="admissibility rule checksum"):
        _amend(completed_run, expected_fingerprint)


def test_a_changed_run_fingerprint_binding_refuses_the_existing_amendment(
    completed_run, expected_fingerprint
):
    _amend(completed_run, expected_fingerprint)
    path = completed_run / AMENDED_SUMMARY_NAME
    summary = json.loads(path.read_text(encoding="utf-8"))
    summary["binding"]["run_fingerprint_digest"] = "sha256:" + "3" * 64
    path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(AmendedReportBindingError, match="run fingerprint"):
        _amend(completed_run, expected_fingerprint)


@pytest.mark.parametrize("orphan", [AMENDED_REPORT_NAME, AMENDED_SUMMARY_NAME])
def test_exactly_one_amended_artifact_is_refused_as_a_torn_write(
    completed_run, expected_fingerprint, orphan
):
    _amend(completed_run, expected_fingerprint)
    other = (
        AMENDED_SUMMARY_NAME if orphan == AMENDED_REPORT_NAME else AMENDED_REPORT_NAME
    )
    (completed_run / other).unlink()
    before = _tree(completed_run)

    with pytest.raises(AmendedArtifactsTorn, match="torn write"):
        _amend(completed_run, expected_fingerprint)
    assert _tree(completed_run) == before


def test_the_original_report_and_summary_are_never_touched(
    completed_run, expected_fingerprint
):
    originals = {
        name: (completed_run / name).read_bytes()
        for name in (ORIGINAL_REPORT_NAME, ORIGINAL_SUMMARY_NAME)
    }
    _amend(completed_run, expected_fingerprint)
    _amend(completed_run, expected_fingerprint)  # the reuse path too
    for name, content in originals.items():
        assert (completed_run / name).read_bytes() == content

    with pytest.raises(FileExistsError, match="refusing to write"):
        store = _open(completed_run, expected_fingerprint)
        write_amended_report(
            build_amended_report(
                store,
                primary_layer=35,
                replication_layers=(38, 40),
                focal_concepts=FOCAL,
                thresholds=THRESHOLDS,
            ),
            run_dir=completed_run,
            report_name=ORIGINAL_REPORT_NAME,
        )


def test_no_source_unit_file_changes(completed_run, expected_fingerprint):
    units = completed_run / "units"
    before = _tree(units)
    _amend(completed_run, expected_fingerprint)
    _amend(completed_run, expected_fingerprint)
    assert _tree(units) == before

    # And the units still resume under the untouched generation fingerprint.
    resumed = UnitStore(completed_run, _fingerprint())
    assert resumed.open() == "resuming"
    assert not resumed.invalid_units


def test_amending_creates_nothing_outside_the_two_named_artifacts(
    completed_run, expected_fingerprint
):
    before = set(_tree(completed_run))
    _amend(completed_run, expected_fingerprint)
    assert set(_tree(completed_run)) - before == {
        AMENDED_REPORT_NAME,
        AMENDED_SUMMARY_NAME,
    }


# ---------------------------------------------- 16: switch mutual exclusion


def test_amendment_mode_is_refused_beside_every_model_and_causal_switch():
    base = dict.fromkeys(MODEL_AND_CAUSAL_SWITCHES, False)
    assert set(MODEL_AND_CAUSAL_SWITCHES) == {
        "RUN_MODEL_STAGES",
        "CONFIRM_MODEL_LOAD",
        "CONFIRM_REPRESENTATION_BUDGET",
        "RUN_L35_CAUSAL_STAGE",
        "CONFIRM_L35_CAUSAL_BUDGET",
        "RUN_L38_L40_REPLICATION",
        "CONFIRM_REPLICATION_BUDGET",
    }

    allowed = assert_amendment_mode_exclusive(
        {**base, "RUN_AMENDED_REPORT_ONLY": True, "RUN_REAL_AUDIO_TRANSFER": True}
    )
    assert allowed["amendment_only"] is True

    for switch in MODEL_AND_CAUSAL_SWITCHES:
        with pytest.raises(AmendmentModeError, match=switch):
            assert_amendment_mode_exclusive(
                {**base, switch: True, "RUN_AMENDED_REPORT_ONLY": True}
            )
        # Without the amendment switch the same combination is ordinary.
        assert (
            assert_amendment_mode_exclusive(
                {**base, switch: True, "RUN_AMENDED_REPORT_ONLY": False}
            )["amendment_only"]
            is False
        )


def test_a_missing_switch_reads_as_off_rather_than_crashing():
    assert assert_amendment_mode_exclusive({})["amendment_only"] is False


# ------------------------- 1, 2: standalone, in an interpreter with no STORE


STANDALONE = textwrap.dedent(
    '''
    """Everything the amendment needs, in a process that has never seen STORE."""
    import json
    import sys

    from jlens.mmpilot.amend_open import (
        amend_or_reuse,
        assert_no_model_module_imported,
        open_existing_store,
        restore_report_metadata,
    )

    assert "STORE" not in dir(), "the fixture is meaningless if STORE exists"
    for name in ("STORE", "FINGERPRINT", "LENS_REPORT", "AUDIO_PROTOCOL",
                 "INVARIANCE", "BLOCKED_MODALITIES"):
        assert name not in globals(), name

    run_dir, expected = sys.argv[1], sys.argv[2]
    store = open_existing_store(run_dir, expected_fingerprint=expected)
    inputs = restore_report_metadata(store)
    for key in ("summary_path", "restored_from", "restored_fields"):
        inputs.pop(key)
    result = amend_or_reuse(store, **inputs)

    print(json.dumps({
        "status": result["status"],
        "resume": store.status,
        "fingerprint": store.fingerprint.digest,
        "overall": (result["verdicts"]["overall"] or {}).get("verdict"),
        "primary": (result["verdicts"]["primary_causal"] or {}).get("verdict"),
        "modules": assert_no_model_module_imported(),
        "torch_imported": "torch" in sys.modules,
    }))
    '''
)


def _run_standalone(tmp_path, run_dir, expected):
    script = tmp_path / "standalone_amend.py"
    script.write_text(STANDALONE, encoding="utf-8")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPO_ROOT)
    result = subprocess.run(
        [sys.executable, str(script), str(run_dir), expected],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        env=environment,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_the_amendment_runs_in_a_fresh_interpreter_where_store_is_undefined(
    tmp_path, completed_run, expected_fingerprint
):
    payload = _run_standalone(tmp_path, completed_run, expected_fingerprint)
    assert payload["status"] == "written"
    assert payload["resume"] == "resuming"
    assert payload["fingerprint"] == expected_fingerprint
    assert payload["overall"] == THREE_MODALITY_GO
    assert payload["primary"] == TRANSFER_SUPPORTED

    again = _run_standalone(tmp_path, completed_run, expected_fingerprint)
    assert again["status"] == "reused"


def test_the_standalone_amendment_imports_no_model_machinery(
    tmp_path, completed_run, expected_fingerprint
):
    payload = _run_standalone(tmp_path, completed_run, expected_fingerprint)
    # transformers, the Gemma adapter and the real backend are all absent.
    assert payload["modules"] == {"clean": True, "imported": []}
    # torch comes in with `import jlens` itself and is not model machinery; it is
    # reported rather than forbidden so the distinction stays explicit.
    assert payload["torch_imported"] is True
