# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Atomic unit saving, compatible resume, and refusal of incompatible state."""

import json

import pytest

from jlens.mmpilot.store import (
    IncompatibleStateError,
    RunFingerprint,
    UnitStore,
    safe_key,
)


def fingerprint(**overrides) -> RunFingerprint:
    base = dict(
        mode="mock",
        model_repo_id="google/gemma-4-E4B-it",
        model_revision="rev-a",
        processor_revision="rev-a",
        layers=(35, 38),
        lens_checksum="sha256:lens",
        manifest_checksum="sha256:manifest",
        split_id="pilot-v1",
        intervention_config={"alphas": [0.0, 0.5]},
    )
    base.update(overrides)
    return RunFingerprint(**base)


def test_first_open_starts_and_second_resumes(tmp_path):
    store = UnitStore(tmp_path / "run", fingerprint())
    assert store.open() == "starting"
    assert UnitStore(tmp_path / "run", fingerprint()).open() == "resuming"


def test_incompatible_fingerprint_is_refused_with_a_diff(tmp_path):
    UnitStore(tmp_path / "run", fingerprint()).open()
    store = UnitStore(tmp_path / "run", fingerprint(model_revision="rev-b"))
    with pytest.raises(IncompatibleStateError) as excinfo:
        store.open()
    assert "model_revision" in str(excinfo.value)
    assert "rev-b" in str(excinfo.value)


@pytest.mark.parametrize(
    "field, value",
    [
        ("layers", (35, 40)),
        ("lens_checksum", "sha256:other"),
        ("manifest_checksum", "sha256:other"),
        ("split_id", "pilot-v2"),
        ("intervention_config", {"alphas": [1.0]}),
        ("processor_revision", "rev-z"),
    ],
)
def test_every_fingerprint_field_invalidates_the_run(tmp_path, field, value):
    UnitStore(tmp_path / "run", fingerprint()).open()
    with pytest.raises(IncompatibleStateError):
        UnitStore(tmp_path / "run", fingerprint(**{field: value})).open()


def test_saved_unit_round_trips_and_is_reused(tmp_path):
    store = UnitStore(tmp_path / "run", fingerprint())
    store.open()
    store.save("capability", safe_key("g1", "text"), {"correct": True, "margin": 1.5})
    assert store.has("capability", safe_key("g1", "text"))
    assert store.load("capability", safe_key("g1", "text"))["margin"] == 1.5
    assert list(store.load_all("capability")) == [safe_key("g1", "text")]


def test_save_is_atomic_and_leaves_no_temp_files(tmp_path):
    store = UnitStore(tmp_path / "run", fingerprint())
    store.open()
    path = store.save("jspace", "unit", {"a": 1})
    assert path.is_file()
    assert list(path.parent.glob("*.tmp*")) == []


def test_a_torn_unit_is_treated_as_missing_not_trusted(tmp_path):
    store = UnitStore(tmp_path / "run", fingerprint())
    store.open()
    path = store.save("activation", "unit", {"norm": 3.0})
    path.write_text('{"schema": "jlens.mmpilot.unit.v1", "payload": {', encoding="utf-8")
    assert store.load("activation", "unit") is None
    assert store.invalid_units


def test_a_tampered_payload_fails_its_checksum(tmp_path):
    store = UnitStore(tmp_path / "run", fingerprint())
    store.open()
    path = store.save("activation", "unit", {"norm": 3.0})
    record = json.loads(path.read_text(encoding="utf-8"))
    record["payload"]["norm"] = 99.0
    path.write_text(json.dumps(record), encoding="utf-8")
    assert store.load("activation", "unit") is None


def test_units_from_another_fingerprint_are_not_reused(tmp_path):
    first = UnitStore(tmp_path / "a", fingerprint())
    first.open()
    unit_path = first.save("capability", "unit", {"correct": True})
    second = UnitStore(tmp_path / "b", fingerprint(model_revision="rev-b"))
    second.open()
    (second.stage_dir("capability")).mkdir(parents=True, exist_ok=True)
    (second.stage_dir("capability") / "unit.json").write_text(
        unit_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    assert second.load("capability", "unit") is None


def test_unknown_stage_is_rejected(tmp_path):
    store = UnitStore(tmp_path / "run", fingerprint())
    store.open()
    with pytest.raises(ValueError, match="unknown stage"):
        store.save("not_a_stage", "unit", {})


def test_long_keys_stay_unique_and_filesystem_safe():
    a = safe_key("x" * 200, "text")
    b = safe_key("x" * 200, "image")
    assert a != b
    assert len(a) <= 120
    assert all(character.isalnum() or character in "._-" for character in a)


def test_status_report_names_the_resume_state(tmp_path):
    store = UnitStore(tmp_path / "run", fingerprint())
    store.open()
    store.save("capability", "u1", {"ok": True})
    report = store.status_report()
    assert report["status"] == "starting"
    assert report["completed_units"]["capability"] == 1
    assert report["completed_units"]["intervention"] == 0
