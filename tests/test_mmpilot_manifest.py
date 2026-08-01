# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Manifest inspection, normalization, missing media, and split leakage."""

import errno
import json
import os
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


def _annotated_groups(dataset):
    """Normalized groups carrying their COCO object annotations.

    Selection needs *both* halves of the evidence rule, so a group without its
    visual annotation is not a candidate positive for anything.
    """
    _, normalized = _normalized(dataset)
    return K.attach_object_annotations(normalized.groups, dataset["built"])


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


# ------------------------------------------------- transient Drive failures


class _FlakyStat:
    """Replaces the module's stat indirection to inject failures per path.

    ``failures`` maps a path substring to a list of exceptions raised on
    successive calls; once that list is exhausted the real stat runs. Every
    call is counted, so a test can assert how many retries happened.
    """

    def __init__(self, failures):
        self.failures = {key: list(value) for key, value in failures.items()}
        self.calls = []

    def __call__(self, path):
        self.calls.append(str(path))
        for key, queue in self.failures.items():
            if key in str(path) and queue:
                raise queue.pop(0)
        return os.stat(path)


def _eio(path="/content/drive/x.jpg"):
    return OSError(errno.EIO, "Input/output error", path)


@pytest.fixture
def no_sleep(monkeypatch):
    slept = []
    monkeypatch.setattr(M, "_sleep", slept.append)
    return slept


def test_transient_eio_then_success_resolves_the_file(tmp_path, monkeypatch, no_sleep):
    target = tmp_path / "photo.jpg"
    target.write_bytes(b"pixels")
    flaky = _FlakyStat({"photo.jpg": [_eio(), _eio()]})
    monkeypatch.setattr(M, "_stat", flaky)
    journal = []
    assert M.safe_is_file(target, root=tmp_path, journal=journal) is True
    assert len(journal) == 2, journal
    assert {entry["errno"] for entry in journal} == {errno.EIO}
    assert all(entry["root"] == str(tmp_path) for entry in journal)
    assert len(no_sleep) == 2
    assert no_sleep == sorted(no_sleep), "backoff must not shrink"


def test_persistent_eio_raises_media_io_error_naming_the_remedy(tmp_path, monkeypatch, no_sleep):
    target = tmp_path / "photo.jpg"
    target.write_bytes(b"pixels")
    monkeypatch.setattr(M, "_stat", _FlakyStat({"photo.jpg": [_eio() for _ in range(10)]}))
    with pytest.raises(M.MediaIOError) as excinfo:
        M.safe_is_file(target, root=tmp_path)
    message = str(excinfo.value)
    assert str(target) in message
    assert str(tmp_path) in message
    assert "attempts:        4" in message
    assert "EIO" in message
    assert "force_remount=True" in message


def test_a_transient_failure_is_never_reported_as_a_missing_file(tmp_path, monkeypatch, no_sleep):
    target = tmp_path / "photo.jpg"
    target.write_bytes(b"pixels")
    monkeypatch.setattr(M, "_stat", _FlakyStat({"photo.jpg": [_eio() for _ in range(10)]}))
    with pytest.raises(M.MediaIOError):
        M.probe_path(target, root=tmp_path)
    # The distinction that matters: absent returns None, flaky raises.
    assert M.probe_path(tmp_path / "not_there.jpg") is None


def test_true_missing_file_is_still_missing(tmp_path):
    assert M.safe_is_file(tmp_path / "absent.jpg") is False
    assert M.probe_path(tmp_path / "absent.jpg") is None


def test_enoent_from_a_raw_oserror_counts_as_missing(tmp_path, monkeypatch):
    target = tmp_path / "photo.jpg"
    target.write_bytes(b"pixels")
    monkeypatch.setattr(
        M, "_stat", _FlakyStat({"photo.jpg": [OSError(errno.ENOENT, "nope")]})
    )
    assert M.safe_is_file(target) is False


def test_permission_error_is_not_retried_and_not_called_missing(tmp_path, monkeypatch, no_sleep):
    target = tmp_path / "photo.jpg"
    target.write_bytes(b"pixels")
    flaky = _FlakyStat({"photo.jpg": [PermissionError(errno.EACCES, "denied")]})
    monkeypatch.setattr(M, "_stat", flaky)
    with pytest.raises(M.MediaIOError, match="permission denied"):
        M.safe_is_file(target, root=tmp_path)
    assert len(flaky.calls) == 1, "a permission error must not be retried"
    assert no_sleep == []


def test_unrecognised_oserror_is_refused_rather_than_guessed(tmp_path, monkeypatch, no_sleep):
    target = tmp_path / "photo.jpg"
    target.write_bytes(b"pixels")
    monkeypatch.setattr(
        M, "_stat", _FlakyStat({"photo.jpg": [OSError(errno.EFBIG, "too big")]})
    )
    with pytest.raises(M.MediaIOError, match="unrecognised filesystem error"):
        M.safe_is_file(target)


def test_retries_do_not_duplicate_a_resolution(tmp_path, monkeypatch, no_sleep):
    """A retried probe must still count as one candidate — otherwise a flaky
    mount would look like the same file living under two roots."""
    root_a, root_b = tmp_path / "a", tmp_path / "b"
    for root in (root_a, root_b):
        (root / "sub").mkdir(parents=True)
    (root_a / "sub" / "photo.jpg").write_bytes(b"pixels")
    monkeypatch.setattr(M, "_stat", _FlakyStat({"photo.jpg": [_eio(), _eio()]}))
    journal = []
    resolution = M._resolve_one("sub/photo.jpg", [root_a, root_b], journal=journal)
    assert resolution.resolved == str(root_a / "sub" / "photo.jpg")
    assert resolution.ambiguous is False
    assert len(resolution.candidates) == 1
    assert len(journal) == 2


def test_normalization_survives_a_flaky_mount_and_records_the_retries(
    dataset, monkeypatch, no_sleep
):
    schema = M.inspect_manifest(dataset["payload"])
    first_image = dataset["payload"]["data"][0]["image"].split("/")[-1]
    monkeypatch.setattr(M, "_stat", _FlakyStat({first_image: [_eio(), _eio()]}))
    normalized = M.normalize_manifest(
        dataset["payload"],
        schema,
        media_roots=[dataset["root"]],
        source_checksum="sha256:test",
        min_complete_groups=8,
    )
    assert normalized.audit["n_missing_image_files"] == 0
    assert normalized.audit["n_transient_io_retries"] == 2
    assert normalized.audit["transient_io_examples"][0]["errno"] == errno.EIO


def test_audit_turns_a_persistent_drive_failure_into_an_actionable_error(
    dataset, monkeypatch, no_sleep
):
    schema = M.inspect_manifest(dataset["payload"])
    monkeypatch.setattr(M, "_stat", _FlakyStat({".jpg": [_eio() for _ in range(50)]}))
    roots = M.resolve_media_roots(media_roots=[dataset["root"]])
    with pytest.raises(M.MediaIOError) as excinfo:
        M.audit_media_roots(dataset["payload"], schema, roots, n_samples=2)
    message = str(excinfo.value)
    assert "the image media probe failed while auditing" in message
    assert "force_remount=True" in message
    assert "attempts:        4" in message


def test_audit_reports_retries_that_recovered(dataset, monkeypatch, no_sleep):
    schema = M.inspect_manifest(dataset["payload"])
    first_image = dataset["payload"]["data"][0]["image"].split("/")[-1]
    monkeypatch.setattr(M, "_stat", _FlakyStat({first_image: [_eio()]}))
    roots = M.resolve_media_roots(media_roots=[dataset["root"]])
    report = M.audit_media_roots(dataset["payload"], schema, roots, n_samples=2)
    assert report["ok"]
    assert report["transient_io_retries"] == 1
    assert report["transient_io_examples"][0]["errno_name"] == "EIO"


def test_concept_coverage_counts_images_not_captions(dataset):
    _, normalized = _normalized(dataset)
    coverage = M.concept_coverage(normalized.groups, K.MOCK_CONCEPTS)
    for concept, entry in coverage.items():
        assert entry["n_images"] == 6, (concept, entry)
        assert entry["n_groups"] == 12, (concept, entry)


def test_split_is_image_group_and_sample_disjoint_but_not_concept_disjoint(dataset):
    subset = M.build_subset(_annotated_groups(dataset), K.MOCK_CONCEPTS, groups_per_concept=6)
    report = M.check_split_leakage(subset)
    assert report["ok"], report
    assert report["image_overlap"] == []
    assert report["group_overlap"] == []
    assert report["audio_overlap"] == []
    # The point of the design: concepts *must* be shared across the split.
    assert sorted(report["shared_concepts_expected"]) == sorted(K.MOCK_CONCEPTS)


def test_all_groups_of_one_image_stay_in_the_same_split(dataset):
    subset = M.build_subset(_annotated_groups(dataset), K.MOCK_CONCEPTS, groups_per_concept=6)
    split_of: dict[str, set] = {}
    for split in ("train", "test"):
        for group in subset["splits"][split]:
            split_of.setdefault(group["image_id"], set()).add(split)
    assert all(len(splits) == 1 for splits in split_of.values())


def test_leakage_check_catches_an_injected_overlap(dataset):
    subset = M.build_subset(_annotated_groups(dataset), K.MOCK_CONCEPTS, groups_per_concept=6)
    subset["splits"]["test"].append(dict(subset["splits"]["train"][0]))
    report = M.check_split_leakage(subset)
    assert not report["ok"]
    assert report["group_overlap"]


def test_subset_selection_is_deterministic(dataset):
    groups = _annotated_groups(dataset)
    first = M.build_subset(groups, K.MOCK_CONCEPTS, groups_per_concept=6)
    second = M.build_subset(groups, K.MOCK_CONCEPTS, groups_per_concept=6)
    assert first["concepts"] == second["concepts"]
    assert [g["group_id"] for g in first["splits"]["test"]] == [
        g["group_id"] for g in second["splits"]["test"]
    ]


def test_subset_materializes_only_individually_valid_positive_groups():
    groups = []
    for image_index in range(2):
        common = {
            "image_id": f"cat-image-{image_index}",
            "image_path": f"cat-{image_index}.jpg",
            "speaker": f"speaker-{image_index}",
            "concept_annotations": ["cat"],
            "annotation_source": "coco_object_annotation",
        }
        groups.extend(
            [
                {
                    **common,
                    "group_id": f"a-invalid-{image_index}",
                    "caption": "an animal resting quietly",
                    "audio_path": f"invalid-{image_index}.wav",
                },
                {
                    **common,
                    "group_id": f"z-valid-{image_index}",
                    "caption": "a cat resting quietly",
                    "audio_path": f"valid-{image_index}.wav",
                },
            ]
        )

    subset = M.build_subset(
        groups,
        {"cat": ["cat", "cats"]},
        groups_per_concept=2,
        negatives_per_concept=0,
        max_groups_per_image=2,
    )
    positives = subset["splits"]["train"] + subset["splits"]["test"]
    assert len(positives) == 2
    assert all(group["group_id"].startswith("z-valid") for group in positives)
    assert all(group["evidence"]["is_valid_synchronized_positive"] for group in positives)


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
