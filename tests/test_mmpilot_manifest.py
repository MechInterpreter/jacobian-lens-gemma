# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Manifest inspection, normalization, missing media, and split leakage."""

import json
from pathlib import Path

import pytest

from jlens.mmpilot import manifest as M
from jlens.mmpilot import mock as K


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("spokencoco")
    built = K.build_mock_dataset(root / "data")
    payload = json.loads(Path(built["manifest_path"]).read_text(encoding="utf-8"))
    return {"built": built, "payload": payload, "root": root / "data"}


def _normalized(dataset):
    schema = M.inspect_manifest(dataset["payload"])
    return schema, M.normalize_manifest(
        dataset["payload"],
        schema,
        media_roots=[dataset["root"]],
        source_checksum=M.manifest_checksum(dataset["built"]["manifest_path"]),
        min_complete_groups=8,
    )


def test_schema_is_discovered_not_assumed(dataset):
    schema = M.inspect_manifest(dataset["payload"])
    assert schema.records_key == "data"
    assert schema.nested_key == "captions"
    assert schema.image_field == "image"
    assert schema.audio_field == "wav"
    assert schema.caption_field == "text"
    assert schema.speaker_field == "speaker"
    # The evidence is recorded so a human can audit the mapping.
    assert "image" in schema.evidence and "wav" in schema.evidence["audio"]


def test_unknown_structure_is_refused():
    with pytest.raises(M.ManifestSchemaError):
        M.inspect_manifest({"note": "no records here"})


def test_missing_required_role_is_refused():
    payload = [{"path_a": "x.jpg", "path_b": "y.jpg"}]
    with pytest.raises(M.ManifestSchemaError):
        M.inspect_manifest(payload)


def test_overrides_win_over_inspection(dataset):
    schema = M.inspect_manifest(dataset["payload"], overrides={"caption": "uttid"})
    assert schema.caption_field == "uttid"


def test_normalized_groups_resolve_media_and_audit_counts(dataset):
    _, normalized = _normalized(dataset)
    audit = normalized.audit
    assert audit["n_synchronized_groups"] == len(normalized.groups)
    assert audit["n_missing_image_files"] == 0
    assert audit["n_missing_audio_files"] == 0
    assert audit["n_duplicate_groups"] == 0
    assert audit["speaker_metadata_available"] is True
    assert audit["n_speakers"] >= 2
    for group in normalized.groups:
        assert Path(group["image_path"]).is_file()
        assert Path(group["audio_path"]).is_file()
    assert normalized.conversion["original_manifest_mutated"] is False


def test_missing_media_is_detected_and_reported(dataset, tmp_path):
    payload = json.loads(json.dumps(dataset["payload"]))
    payload["data"][0]["image"] = "images/does_not_exist.jpg"
    schema = M.inspect_manifest(payload)
    normalized = M.normalize_manifest(
        payload,
        schema,
        media_roots=[dataset["root"]],
        source_checksum="sha256:test",
        min_complete_groups=8,
    )
    assert normalized.audit["n_missing_image_files"] == 2  # two captions on that image
    assert "images/does_not_exist.jpg" in normalized.audit["missing_image_examples"]
    assert all(
        group["image_id"] != payload["data"][0]["image_id"] for group in normalized.groups
    )


def test_wrong_dataset_root_refuses_rather_than_continuing(dataset, tmp_path):
    schema = M.inspect_manifest(dataset["payload"])
    with pytest.raises(M.SynchronizationError, match="did not resolve"):
        M.normalize_manifest(
            dataset["payload"],
            schema,
            media_roots=[tmp_path / "nowhere"],
            source_checksum="sha256:test",
            min_complete_groups=1,
        )


def test_too_few_groups_refuses(dataset):
    schema = M.inspect_manifest(dataset["payload"])
    with pytest.raises(M.SynchronizationError, match="synchronized"):
        M.normalize_manifest(
            dataset["payload"],
            schema,
            media_roots=[dataset["root"]],
            source_checksum="sha256:test",
            min_complete_groups=10_000,
        )


def test_concept_coverage_counts_images_not_captions(dataset):
    _, normalized = _normalized(dataset)
    coverage = M.concept_coverage(normalized.groups, K.MOCK_CONCEPTS)
    for concept, entry in coverage.items():
        assert entry["n_images"] == 6, (concept, entry)
        assert entry["n_groups"] == 12, (concept, entry)


def test_split_is_image_group_and_sample_disjoint_but_not_concept_disjoint(dataset):
    _, normalized = _normalized(dataset)
    subset = M.build_subset(normalized.groups, K.MOCK_CONCEPTS, groups_per_concept=6)
    report = M.check_split_leakage(subset)
    assert report["ok"], report
    assert report["image_overlap"] == []
    assert report["group_overlap"] == []
    assert report["audio_overlap"] == []
    # The point of the design: concepts *must* be shared across the split.
    assert sorted(report["shared_concepts_expected"]) == sorted(K.MOCK_CONCEPTS)


def test_all_groups_of_one_image_stay_in_the_same_split(dataset):
    _, normalized = _normalized(dataset)
    subset = M.build_subset(normalized.groups, K.MOCK_CONCEPTS, groups_per_concept=6)
    split_of: dict[str, set] = {}
    for split in ("train", "test"):
        for group in subset["splits"][split]:
            split_of.setdefault(group["image_id"], set()).add(split)
    assert all(len(splits) == 1 for splits in split_of.values())


def test_leakage_check_catches_an_injected_overlap(dataset):
    _, normalized = _normalized(dataset)
    subset = M.build_subset(normalized.groups, K.MOCK_CONCEPTS, groups_per_concept=6)
    subset["splits"]["test"].append(dict(subset["splits"]["train"][0]))
    report = M.check_split_leakage(subset)
    assert not report["ok"]
    assert report["group_overlap"]


def test_subset_selection_is_deterministic(dataset):
    _, normalized = _normalized(dataset)
    first = M.build_subset(normalized.groups, K.MOCK_CONCEPTS, groups_per_concept=6)
    second = M.build_subset(normalized.groups, K.MOCK_CONCEPTS, groups_per_concept=6)
    assert first["concepts"] == second["concepts"]
    assert [g["group_id"] for g in first["splits"]["test"]] == [
        g["group_id"] for g in second["splits"]["test"]
    ]


def test_captions_naming_two_selected_concepts_are_dropped(dataset):
    groups = [
        {
            "group_id": "g1",
            "image_id": "i1",
            "caption": "a dog and a cat on a bus",
            "image_path": "x",
            "audio_path": "y",
            "speaker": None,
            "source_split": None,
        }
    ]
    subset = M.build_subset(groups, K.MOCK_CONCEPTS, groups_per_concept=2)
    assert subset["splits"]["train"] == []
    assert subset["splits"]["test"] == []
