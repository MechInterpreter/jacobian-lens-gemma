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


def test_nonexistent_root_is_refused_immediately(dataset, tmp_path):
    schema = M.inspect_manifest(dataset["payload"])
    with pytest.raises(M.MediaRootError, match="no existing directory"):
        M.normalize_manifest(
            dataset["payload"],
            schema,
            media_roots=[tmp_path / "nowhere"],
            source_checksum="sha256:test",
            min_complete_groups=1,
        )


def test_an_existing_but_wrong_root_reports_both_roles(dataset, tmp_path):
    """The failure the real dataset hit: the root exists, but nothing under it
    matches, and the message has to say which modality lost its files."""
    empty = tmp_path / "empty"
    empty.mkdir()
    schema = M.inspect_manifest(dataset["payload"])
    with pytest.raises(M.SynchronizationError, match="did not resolve") as excinfo:
        M.normalize_manifest(
            dataset["payload"],
            schema,
            media_roots=[empty],
            source_checksum="sha256:test",
            min_complete_groups=1,
        )
    message = str(excinfo.value)
    assert "image roots:" in message and "audio roots:" in message
    assert "image_roots and audio_roots separately" in message


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


# --------------------------------------------------------------- media roots


@pytest.fixture(scope="module")
def sibling_dataset(tmp_path_factory):
    """The real Drive layout: images under coco/, recordings under SpokenCOCO/,
    each addressed by a manifest path relative to its own root."""
    root = tmp_path_factory.mktemp("cstf_spokencoco")
    built = K.build_mock_dataset(root / "data", layout="sibling")
    payload = json.loads(Path(built["manifest_path"]).read_text(encoding="utf-8"))
    return {"built": built, "payload": payload, "base": root / "data"}


def test_sibling_layout_needs_separate_image_and_audio_roots(sibling_dataset):
    """This is the reported failure: every media reference is unresolvable when
    both modalities are looked up under one base root."""
    schema = M.inspect_manifest(sibling_dataset["payload"])
    with pytest.raises(M.SynchronizationError, match="did not resolve"):
        M.normalize_manifest(
            sibling_dataset["payload"],
            schema,
            media_roots=[sibling_dataset["base"]],
            source_checksum="sha256:test",
            min_complete_groups=1,
        )


def test_separate_sibling_roots_resolve_everything(sibling_dataset):
    base = sibling_dataset["base"]
    schema = M.inspect_manifest(sibling_dataset["payload"])
    normalized = M.normalize_manifest(
        sibling_dataset["payload"],
        schema,
        image_roots=[base / "coco", base / "SpokenCOCO", base],
        audio_roots=[base / "coco", base / "SpokenCOCO", base],
        source_checksum="sha256:test",
        min_complete_groups=8,
    )
    assert normalized.audit["n_missing_image_files"] == 0
    assert normalized.audit["n_missing_audio_files"] == 0
    assert normalized.audit["n_synchronized_groups"] > 0
    for group in normalized.groups:
        assert Path(group["image_path"]).is_file()
        assert Path(group["audio_path"]).is_file()


def test_images_resolve_only_under_the_image_root(sibling_dataset):
    base = sibling_dataset["base"]
    schema = M.inspect_manifest(sibling_dataset["payload"])
    normalized = M.normalize_manifest(
        sibling_dataset["payload"],
        schema,
        image_roots=[base / "coco", base / "SpokenCOCO", base],
        audio_roots=[base / "coco", base / "SpokenCOCO", base],
        source_checksum="sha256:test",
        min_complete_groups=8,
    )
    assert list(normalized.audit["resolved_by_root"]["image"]) == [str(base / "coco")]
    assert list(normalized.audit["resolved_by_root"]["audio"]) == [
        str(base / "SpokenCOCO")
    ]
    assert all("coco" in group["image_path"] for group in normalized.groups)
    assert all("SpokenCOCO" in group["audio_path"] for group in normalized.groups)


def test_audio_only_root_cannot_resolve_images(sibling_dataset):
    base = sibling_dataset["base"]
    schema = M.inspect_manifest(sibling_dataset["payload"])
    with pytest.raises(M.SynchronizationError, match="did not resolve"):
        M.normalize_manifest(
            sibling_dataset["payload"],
            schema,
            image_roots=[base / "SpokenCOCO"],
            audio_roots=[base / "SpokenCOCO"],
            source_checksum="sha256:test",
            min_complete_groups=1,
        )


def test_one_root_layouts_still_work(dataset):
    """Backward compatibility: a dataset with both modalities under one root
    needs only ``media_roots``, exactly as before."""
    schema = M.inspect_manifest(dataset["payload"])
    normalized = M.normalize_manifest(
        dataset["payload"],
        schema,
        media_roots=[dataset["root"]],
        source_checksum="sha256:test",
        min_complete_groups=8,
    )
    assert normalized.audit["n_missing_image_files"] == 0
    assert normalized.audit["n_missing_audio_files"] == 0
    assert normalized.audit["media_roots"]["image"] == [str(dataset["root"])]
    assert normalized.audit["media_roots"]["audio"] == [str(dataset["root"])]


def test_ambiguous_duplicate_roots_are_refused(sibling_dataset, tmp_path):
    """Two roots holding *different* files for one manifest path: neither can
    be assumed correct, so normalization refuses instead of picking."""
    base = sibling_dataset["base"]
    decoy = tmp_path / "decoy"
    sample = sibling_dataset["payload"]["data"][0]["image"]
    target = decoy / sample
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"MOCKMEDIA" + b"\x00" * 400)  # a different size
    schema = M.inspect_manifest(sibling_dataset["payload"])
    with pytest.raises(M.MediaRootError, match="several roots"):
        M.normalize_manifest(
            sibling_dataset["payload"],
            schema,
            image_roots=[base / "coco", decoy],
            audio_roots=[base / "SpokenCOCO"],
            source_checksum="sha256:test",
            min_complete_groups=1,
        )


def test_an_identical_mirror_is_not_ambiguous(sibling_dataset, tmp_path):
    """A download cache mirroring the dataset byte-for-byte is not a conflict;
    the first root by configured priority wins and both are recorded."""
    base = sibling_dataset["base"]
    mirror = tmp_path / "cache"
    for record in sibling_dataset["payload"]["data"]:
        source = base / "coco" / record["image"]
        target = mirror / record["image"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    schema = M.inspect_manifest(sibling_dataset["payload"])
    normalized = M.normalize_manifest(
        sibling_dataset["payload"],
        schema,
        image_roots=[base / "coco", mirror],
        audio_roots=[base / "SpokenCOCO"],
        source_checksum="sha256:test",
        min_complete_groups=8,
    )
    assert normalized.audit["n_ambiguous_media"] == 0
    assert all(str(base / "coco") in group["image_path"] for group in normalized.groups)


def test_media_root_audit_reports_which_root_resolves_each_path(sibling_dataset):
    base = sibling_dataset["base"]
    schema = M.inspect_manifest(sibling_dataset["payload"])
    roots = M.resolve_media_roots(
        image_roots=[base / "coco", base / "SpokenCOCO", base],
        audio_roots=[base / "coco", base / "SpokenCOCO", base],
    )
    report = M.audit_media_roots(sibling_dataset["payload"], schema, roots, n_samples=4)
    assert report["ok"]
    assert report["winning_root"]["image"] == str(base / "coco")
    assert report["winning_root"]["audio"] == str(base / "SpokenCOCO")
    for sample in report["samples"]["image"]:
        assert sample["resolved"] is not None
        assert sample["root"] == str(base / "coco")
        assert sample["mode"] == "exact"
    assert report["unresolved"] == {"image": [], "audio": []}


def test_media_root_audit_fails_before_normalization_when_nothing_resolves(
    sibling_dataset,
):
    base = sibling_dataset["base"]
    schema = M.inspect_manifest(sibling_dataset["payload"])
    roots = M.resolve_media_roots(media_roots=[base])
    with pytest.raises(M.MediaRootError, match="no configured image root"):
        M.audit_media_roots(sibling_dataset["payload"], schema, roots, n_samples=4)


def test_media_root_audit_detects_swapped_roots(sibling_dataset):
    """Images resolve under the configured audio root and vice versa."""
    base = sibling_dataset["base"]
    schema = M.inspect_manifest(sibling_dataset["payload"])
    roots = M.resolve_media_roots(
        image_roots=[base / "coco", base / "SpokenCOCO"],
        audio_roots=[base / "coco", base / "SpokenCOCO"],
    )
    with pytest.raises(M.MediaRootError, match="look swapped"):
        M.audit_media_roots(
            sibling_dataset["payload"],
            schema,
            roots,
            expected_roots={
                "image": base / "SpokenCOCO",  # deliberately exchanged
                "audio": base / "coco",
            },
            n_samples=4,
        )


def test_correctly_configured_roots_do_not_trip_the_swap_check(sibling_dataset):
    base = sibling_dataset["base"]
    schema = M.inspect_manifest(sibling_dataset["payload"])
    roots = M.resolve_media_roots(
        image_roots=[base / "coco", base / "SpokenCOCO"],
        audio_roots=[base / "coco", base / "SpokenCOCO"],
    )
    report = M.audit_media_roots(
        sibling_dataset["payload"],
        schema,
        roots,
        expected_roots={"image": base / "coco", "audio": base / "SpokenCOCO"},
        n_samples=4,
    )
    assert report["ok"]


def test_resolve_media_roots_falls_back_and_drops_missing_directories(dataset, tmp_path):
    roots = M.resolve_media_roots(
        media_roots=[dataset["root"], tmp_path / "never_created"]
    )
    assert roots["image"] == roots["audio"] == [dataset["root"]]
    with pytest.raises(M.MediaRootError, match="no media roots given"):
        M.resolve_media_roots()


def test_original_manifest_is_never_mutated_by_resolution(sibling_dataset):
    base = sibling_dataset["base"]
    before = json.dumps(sibling_dataset["payload"], sort_keys=True)
    schema = M.inspect_manifest(sibling_dataset["payload"])
    M.normalize_manifest(
        sibling_dataset["payload"],
        schema,
        image_roots=[base / "coco", base / "SpokenCOCO"],
        audio_roots=[base / "coco", base / "SpokenCOCO"],
        source_checksum="sha256:test",
        min_complete_groups=8,
    )
    assert json.dumps(sibling_dataset["payload"], sort_keys=True) == before
    on_disk = json.loads(
        Path(sibling_dataset["built"]["manifest_path"]).read_text(encoding="utf-8")
    )
    assert json.dumps(on_disk, sort_keys=True) == before


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
