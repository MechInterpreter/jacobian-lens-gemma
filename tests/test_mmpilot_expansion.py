# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Manifest expansion, concept ranking, and coverage NO-GO behaviour."""

import json
from pathlib import Path

import pytest

from jlens.mmpilot import expansion as E
from jlens.mmpilot import manifest as M
from jlens.mmpilot import mock as K


@pytest.fixture(scope="module")
def sibling_world(tmp_path_factory):
    """Sibling layout: small hand manifest + fuller SpokenCOCO_train.json."""
    root = tmp_path_factory.mktemp("cstf_spokencoco")
    built = K.build_mock_dataset(
        root / "data",
        layout="sibling",
        manifest_records=8,
    )
    base = root / "data"
    payload = json.loads(Path(built["manifest_path"]).read_text(encoding="utf-8"))
    schema = M.inspect_manifest(payload)
    baseline = M.normalize_manifest(
        payload,
        schema,
        image_roots=[base / "coco", base / "SpokenCOCO", base],
        audio_roots=[base / "coco", base / "SpokenCOCO", base],
        source_checksum=M.manifest_checksum(built["manifest_path"]),
        min_complete_groups=1,
    )
    return {
        "built": built,
        "base": base,
        "payload": payload,
        "schema": schema,
        "baseline": baseline,
        "manifest_path": built["manifest_path"],
        "original_checksum": M.manifest_checksum(built["manifest_path"]),
    }


def test_discover_metadata_finds_fuller_annotation_and_rejects_coco_captions(
    sibling_world,
):
    base = sibling_world["base"]
    sources = E.discover_metadata_sources(
        [base, base / "coco", base / "SpokenCOCO"],
        exclude=[sibling_world["manifest_path"]],
    )
    paths = {Path(source.path).name: source for source in sources}
    assert "SpokenCOCO_train.json" in paths
    assert paths["SpokenCOCO_train.json"].usable
    assert paths["SpokenCOCO_train.json"].n_records > sibling_world["built"][
        "n_records_in_manifest"
    ]
    assert "captions_train2014.json" in paths
    assert not paths["captions_train2014.json"].usable
    assert "no deterministic image-caption-audio schema" in paths["captions_train2014.json"].reason


def test_expansion_adds_groups_without_mutating_original_manifest(sibling_world):
    base = sibling_world["base"]
    manifest_path = Path(sibling_world["manifest_path"])
    before_disk = manifest_path.read_text(encoding="utf-8")
    before_payload = json.dumps(sibling_world["payload"], sort_keys=True)

    sources = E.discover_metadata_sources(
        [base, base / "coco", base / "SpokenCOCO"],
        exclude=[manifest_path],
    )
    result = E.build_expanded_manifest(
        [source for source in sources if source.usable],
        image_roots=[base / "coco", base / "SpokenCOCO", base],
        audio_roots=[base / "coco", base / "SpokenCOCO", base],
        baseline_groups=sibling_world["baseline"].groups,
    )

    assert result.n_groups > len(sibling_world["baseline"].groups)
    assert len(result.baseline_group_ids) == len(sibling_world["baseline"].groups)
    baseline_ids = {group["group_id"] for group in sibling_world["baseline"].groups}
    assert baseline_ids.issubset({group["group_id"] for group in result.groups})

    assert manifest_path.read_text(encoding="utf-8") == before_disk
    assert json.dumps(sibling_world["payload"], sort_keys=True) == before_payload


def test_expanded_manifest_carries_checksums_and_conversion(sibling_world):
    base = sibling_world["base"]
    sources = E.discover_metadata_sources(
        [base, base / "coco", base / "SpokenCOCO"],
        exclude=[sibling_world["manifest_path"]],
    )
    result = E.build_expanded_manifest(
        [source for source in sources if source.usable],
        image_roots=[base / "coco", base / "SpokenCOCO", base],
        audio_roots=[base / "coco", base / "SpokenCOCO", base],
        baseline_groups=sibling_world["baseline"].groups,
    )
    payload = result.to_dict(
        original_checksum=sibling_world["original_checksum"],
        conversion={"converter": "test", "reads_only": True},
    )
    assert payload["original_manifest_mutated"] is False
    assert payload["media_redownloaded"] is False
    assert payload["original_manifest_checksum"] == sibling_world["original_checksum"]
    assert payload["n_groups_in_original"] == len(sibling_world["baseline"].groups)
    assert any(source.checksum for source in result.sources if source.usable)
    for group in result.groups:
        assert Path(group["image_path"]).is_file()
        assert Path(group["audio_path"]).is_file()


def test_rank_concepts_orders_by_feasibility_then_coverage(sibling_world):
    base = sibling_world["base"]
    sources = E.discover_metadata_sources(
        [base, base / "coco", base / "SpokenCOCO"],
        exclude=[sibling_world["manifest_path"]],
    )
    expanded = E.build_expanded_manifest(
        [source for source in sources if source.usable],
        image_roots=[base / "coco", base / "SpokenCOCO", base],
        audio_roots=[base / "coco", base / "SpokenCOCO", base],
        baseline_groups=sibling_world["baseline"].groups,
    )
    rows = E.rank_concepts(expanded.groups, K.MOCK_CONCEPTS, groups_per_concept=6)
    assert rows[0]["feasible"]
    assert sum(row["feasible"] for row in rows) >= 2
    table = E.format_ranking_table(rows)
    assert "concept" in table and "images" in table
    for row in rows:
        if not row["feasible"]:
            assert row["unmet"]


def test_select_concepts_and_subset_are_image_and_group_disjoint(sibling_world):
    base = sibling_world["base"]
    sources = E.discover_metadata_sources(
        [base, base / "coco", base / "SpokenCOCO"],
        exclude=[sibling_world["manifest_path"]],
    )
    expanded = E.build_expanded_manifest(
        [source for source in sources if source.usable],
        image_roots=[base / "coco", base / "SpokenCOCO", base],
        audio_roots=[base / "coco", base / "SpokenCOCO", base],
        baseline_groups=sibling_world["baseline"].groups,
    )
    rows = E.rank_concepts(expanded.groups, K.MOCK_CONCEPTS, groups_per_concept=6)
    chosen = E.select_concepts(rows, n_concepts=2)
    concepts = {name: K.MOCK_CONCEPTS[name] for name in chosen}
    subset = M.build_subset(expanded.groups, concepts, groups_per_concept=6)
    report = M.check_split_leakage(subset)
    assert report["ok"], report


def test_insufficient_local_coverage_raises_dataset_no_go(tmp_path):
    root = tmp_path / "tiny"
    built = K.build_mock_dataset(root, layout="flat", images_per_concept=1, negative_images=1)
    payload = json.loads(Path(built["manifest_path"]).read_text(encoding="utf-8"))
    schema = M.inspect_manifest(payload)
    baseline = M.normalize_manifest(
        payload,
        schema,
        media_roots=[root],
        source_checksum="sha256:test",
        min_complete_groups=1,
    )
    rows = E.rank_concepts(baseline.groups, K.MOCK_CONCEPTS, groups_per_concept=6)
    with pytest.raises(E.DatasetCoverageError, match="DATASET NO-GO") as excinfo:
        E.select_concepts(rows, n_concepts=2)
    message = str(excinfo.value)
    assert "NOT lowered automatically" in message
    assert "TINY_SMOKE" in message


def test_scientific_thresholds_are_not_silently_lowered():
    requirements = E.ConceptRequirements()
    tiny = E.tiny_smoke_requirements()
    assert requirements.min_distinct_images == 6
    assert requirements.min_groups == 6
    assert tiny.min_distinct_images == 2
    assert tiny.min_groups == 2
    assert requirements is not tiny


def test_tiny_smoke_requirements_allow_plumbing_subset(sibling_world):
    base = sibling_world["base"]
    sources = E.discover_metadata_sources(
        [base, base / "coco", base / "SpokenCOCO"],
        exclude=[sibling_world["manifest_path"]],
    )
    expanded = E.build_expanded_manifest(
        [source for source in sources if source.usable],
        image_roots=[base / "coco", base / "SpokenCOCO", base],
        audio_roots=[base / "coco", base / "SpokenCOCO", base],
        baseline_groups=sibling_world["baseline"].groups,
    )
    tiny = E.tiny_smoke_requirements()
    rows = E.rank_concepts(
        expanded.groups,
        K.MOCK_CONCEPTS,
        requirements=tiny,
        groups_per_concept=2,
    )
    chosen = E.select_concepts(rows, n_concepts=2, requirements=tiny)
    assert len(chosen) == 2

def test_discovery_supports_jsonl_csv_and_tsv(tmp_path):
    rows = [{"image": f"i{i}.jpg", "audio": f"a{i}.wav", "caption": f"a dog number {i}", "image_id": str(i)} for i in range(6)]
    (tmp_path / "records.jsonl").write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    for suffix, delimiter in (("csv", ","), ("tsv", "\t")):
        (tmp_path / f"records.{suffix}").write_text(delimiter.join(rows[0]) + "\n" + "\n".join(delimiter.join(row.values()) for row in rows) + "\n", encoding="utf-8")
    sources = E.discover_metadata_sources([tmp_path])
    assert {source.detected_format for source in sources} == {"jsonl", "csv", "tsv"}
    assert all(source.usable for source in sources)
    assert all(source.top_level_schema and source.likely_fields for source in sources)


def test_discovery_is_bounded_by_depth_and_file_count(tmp_path):
    shallow = tmp_path / "meta"
    shallow.mkdir()
    for index in range(4):
        (shallow / f"{index}.json").write_text("[]", encoding="utf-8")
    deep = shallow / "one" / "two"
    deep.mkdir(parents=True)
    (deep / "hidden.json").write_text("[]", encoding="utf-8")
    sources = E.discover_metadata_sources([tmp_path], max_files=2, max_depth=1)
    assert len(sources) == 2
    assert all(Path(source.path).name != "hidden.json" for source in sources)


def test_official_coco_annotations_override_caption_fallback(sibling_world):
    base = sibling_world["base"]
    sources = E.discover_metadata_sources([base], exclude=[sibling_world["manifest_path"]])
    expanded = E.build_expanded_manifest(
        [source for source in sources if source.usable],
        annotation_sources=[source for source in sources if source.source_kind == "coco_object_annotation"],
        image_roots=[base / "coco", base / "SpokenCOCO", base],
        audio_roots=[base / "coco", base / "SpokenCOCO", base],
    )
    annotated = [group for group in expanded.groups if group["concept_annotations"]]
    assert annotated
    assert all(group["annotation_source"] == "coco_object_annotation" for group in annotated)


def test_caption_fallback_has_word_boundaries_not_substrings():
    groups = [{"image_id": "1", "group_id": "g1", "caption": "a caterpillar", "speaker": "s", "concept_annotations": []}]
    rows = E.rank_concepts(groups, {"cat": ["cat"]}, requirements=E.tiny_smoke_requirements(), groups_per_concept=2)
    assert rows[0]["n_distinct_images"] == 0


def test_filename_parser_requires_one_metadata_validated_id():
    assert E.validated_filename_identifier("wavs/spk_cap42.wav", ["cap42", "cap99"]) == "cap42"
    with pytest.raises(E.DatasetCoverageError, match="ambiguous"):
        E.validated_filename_identifier("wavs/spk_cap42.wav", ["cap42", "spk_cap42"])
    with pytest.raises(E.DatasetCoverageError, match="explicit metadata"):
        E.validated_filename_identifier("wavs/unknown.wav", ["cap42"])


def test_persistent_manifest_reuses_only_matching_sources(sibling_world, tmp_path):
    base = sibling_world["base"]
    sources = E.discover_metadata_sources([base], exclude=[sibling_world["manifest_path"]])
    result = E.build_expanded_manifest(
        [source for source in sources if source.usable],
        image_roots=[base / "coco", base / "SpokenCOCO", base],
        audio_roots=[base / "coco", base / "SpokenCOCO", base],
    )
    path = tmp_path / "expanded.json"
    first, status1 = E.persist_expanded_manifest(path, result, original_checksum="sha256:original", conversion={"seed": 1})
    second, status2 = E.persist_expanded_manifest(path, result, original_checksum="sha256:original", conversion={"seed": 1})
    assert "wrote" in status1 and "reused" in status2
    assert first["source_metadata_checksums"] == second["source_metadata_checksums"]
    _, status3 = E.persist_expanded_manifest(path, result, original_checksum="sha256:original", conversion={"seed": 2})
    assert "wrote" in status3


def test_original_manifest_path_is_never_written(sibling_world, tmp_path):
    original = Path(sibling_world["manifest_path"])
    before = original.read_bytes()
    result = E.ExpansionResult([], [], [], [])
    E.persist_expanded_manifest(tmp_path / "expanded.json", result, original_checksum=sibling_world["original_checksum"], conversion={})
    assert original.read_bytes() == before


def test_split_uses_four_train_two_test_and_keeps_images_together(sibling_world):
    base = sibling_world["base"]
    sources = E.discover_metadata_sources([base], exclude=[sibling_world["manifest_path"]])
    expanded = E.build_expanded_manifest([source for source in sources if source.usable], image_roots=[base / "coco", base / "SpokenCOCO", base], audio_roots=[base / "coco", base / "SpokenCOCO", base])
    rows = E.rank_concepts(expanded.groups, K.MOCK_CONCEPTS, groups_per_concept=6)
    chosen = E.select_concepts(rows, n_concepts=2)
    subset = M.build_subset(expanded.groups, {name: K.MOCK_CONCEPTS[name] for name in chosen}, groups_per_concept=6)
    assert all(len(entry["train_images"]) == 4 and len(entry["test_images"]) == 2 for entry in subset["concepts"].values())
    assert M.check_split_leakage(subset)["ok"]

def test_conflicting_audio_to_caption_join_is_rejected(sibling_world):
    base = sibling_world["base"]
    payload = json.loads(Path(sibling_world["built"]["annotation_path"]).read_text(encoding="utf-8"))
    payload["data"][1]["captions"][0]["wav"] = payload["data"][0]["captions"][0]["wav"]
    source = E.MetadataSource(path="synthetic.json", size_bytes=1, checksum="sha256:synthetic", schema=M.inspect_manifest(payload), usable=True, source_kind="synchronized_metadata", payload=payload)
    with pytest.raises(E.DatasetCoverageError, match="conflicting join"):
        E.build_expanded_manifest([source], image_roots=[base / "coco", base / "SpokenCOCO", base], audio_roots=[base / "coco", base / "SpokenCOCO", base])


def test_source_checksum_change_invalidates_persistent_manifest(sibling_world, tmp_path):
    base = sibling_world["base"]
    sources = E.discover_metadata_sources([base], exclude=[sibling_world["manifest_path"]])
    result = E.build_expanded_manifest([source for source in sources if source.usable], image_roots=[base / "coco", base / "SpokenCOCO", base], audio_roots=[base / "coco", base / "SpokenCOCO", base])
    path = tmp_path / "expanded.json"
    E.persist_expanded_manifest(path, result, original_checksum="sha256:o", conversion={"seed": 1})
    usable = next(source for source in result.sources if source.usable)
    usable.checksum = "sha256:changed"
    payload, status = E.persist_expanded_manifest(path, result, original_checksum="sha256:o", conversion={"seed": 1})
    assert "wrote" in status
    assert payload["source_metadata_checksums"][usable.path] == "sha256:changed"


def test_transient_probe_errors_do_not_reduce_expanded_coverage(sibling_world, monkeypatch):
    base = sibling_world["base"]
    sources = E.discover_metadata_sources([base], exclude=[sibling_world["manifest_path"]])
    expected = E.build_expanded_manifest([source for source in sources if source.usable], image_roots=[base / "coco", base / "SpokenCOCO", base], audio_roots=[base / "coco", base / "SpokenCOCO", base]).n_groups
    real_stat = M._stat
    state = {"remaining": 2}
    def flaky(path):
        if state["remaining"] and str(path).endswith(".jpg"):
            state["remaining"] -= 1
            raise OSError(5, "transient EIO")
        return real_stat(path)
    monkeypatch.setattr(M, "_stat", flaky)
    monkeypatch.setattr(M, "_sleep", lambda _seconds: None)
    actual = E.build_expanded_manifest([source for source in sources if source.usable], image_roots=[base / "coco", base / "SpokenCOCO", base], audio_roots=[base / "coco", base / "SpokenCOCO", base]).n_groups
    assert actual == expected

def test_metadata_is_filtered_before_bulk_media_validation():
    rows = [
        {"image": f"i{i}.jpg", "audio": f"a{i}.wav", "caption": ("a cat" if i % 10 == 0 else "a neutral scene"), "image_id": str(i)}
        for i in range(100)
    ]
    schema = M.inspect_manifest(rows)
    bounded, seen, kept = E._bounded_sync_payload(rows, schema, candidate_concepts={"cat": ["cat"]}, object_image_ids=set(), max_records=12)
    assert seen == 100 and kept == 12 and len(bounded) == 12
    assert sum("cat" in row["caption"] for row in bounded) == 10

def test_discovery_prunes_known_media_trees(tmp_path):
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    (annotations / "meta.json").write_text("[]", encoding="utf-8")
    for media_dir in ("wavs", "train2014", "val2014"):
        directory = tmp_path / media_dir / "nested"
        directory.mkdir(parents=True)
        (directory / "must_not_be_seen.json").write_text("[]", encoding="utf-8")
        (directory / "media.wav").write_bytes(b"not metadata")
    sources = E.discover_metadata_sources([tmp_path], max_depth=3)
    assert {Path(source.path).name for source in sources} == {"meta.json"}
