# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""The persistent derived-data cache: when it may be reused, and when it may not.

The audit it caches is expensive and deterministic, so the whole question is
trust. These tests pin down the four states a directory can be in, that a
mismatched fingerprint is refused rather than reused, that an interrupted
publication is never half-read, that a tampered file fails its checksum, and
that nothing user-specific or mutable is written.
"""

import gzip
import json

import pytest

from jlens.mmpilot import cache as K
from jlens.mmpilot import concepts as C
from jlens.mmpilot import evidence as E


def universe_for(*names):
    names = names or ("zebra", "giraffe")
    return C.CategoryUniverse(
        categories=tuple(sorted(names)),
        category_ids={name: (index,) for index, name in enumerate(sorted(names))},
        supercategories={},
        sources=({"path": "annotations/instances_train2014.json", "checksum": "sha256:a"},),
        specs={name: C.lexical_spec(name) for name in sorted(names)},
    )


def fingerprint_for(tmp_path, *, universe=None, thresholds=None, seed="spokencoco-pilot"):
    universe = universe or universe_for()
    return K.build_fingerprint(
        spokencoco_sources={str(tmp_path / "coco" / "SpokenCOCO_train.json"): "sha256:s"},
        coco_annotation_sources={
            str(tmp_path / "coco" / "annotations" / "instances_train2014.json"): "sha256:a"
        },
        original_manifest_checksum="sha256:m",
        evidence_config=E.config_from_specs(universe.specs),
        universe=universe,
        media_roots={
            "image": [tmp_path / "coco"],
            "audio": [tmp_path / "SpokenCOCO"],
        },
        thresholds=thresholds or {"min_distinct_images": 6, "min_groups": 6},
        split_seed=seed,
    )


ARTIFACTS = {
    "metadata.json": {"n_groups": 12},
    "concept_coverage.json": {"rows": [{"concept": "zebra", "feasible": True}]},
    "concept_evidence_index.jsonl.gz": [
        {"group_id": "g0", "image_relpath": "train2014/a.jpg", "audio_relpath": "wavs/a.wav"}
    ],
    "rejected_evidence_counts.json": {"total": {"no_caption_lexical_evidence": 3}},
    "selected_concepts.json": {"selected": ["zebra", "giraffe"]},
    "pilot_subset.json": {"splits": {"train": [], "test": []}},
    "split_provenance.json": {"leakage": {"ok": True}},
}


@pytest.fixture
def cache(tmp_path):
    return K.DerivedCache(
        tmp_path / "drive_cache",
        fingerprint_for(tmp_path),
        staging_root=tmp_path / "staging",
    )


# ---------------------------------------------------------------- fingerprint


def test_the_fingerprint_covers_every_commissioned_input(tmp_path):
    payload = fingerprint_for(tmp_path).to_dict()
    for key in (
        "spokencoco_metadata_checksums",
        "coco_annotation_checksums",
        "original_manifest_checksum",
        "evidence_version",
        "category_universe",
        "lexical_spec_hash",
        "evidence_rule",
        "media_layout",
        "thresholds",
        "split_seed",
        "split_algorithm_version",
    ):
        assert payload[key], key


@pytest.mark.parametrize(
    ("label", "kwargs"),
    [
        ("universe", {"universe": universe_for("zebra", "giraffe", "kite")}),
        ("thresholds", {"thresholds": {"min_distinct_images": 4, "min_groups": 6}}),
        ("seed", {"seed": "a-different-seed"}),
    ],
)
def test_changing_any_fingerprinted_input_changes_the_directory(tmp_path, label, kwargs):
    base = fingerprint_for(tmp_path)
    changed = fingerprint_for(tmp_path, **kwargs)
    assert base.digest != changed.digest, label
    assert base.short != changed.short, label


def test_the_lexical_specification_is_fingerprinted_not_just_the_names(tmp_path):
    """Two universes with the same categories but different lexical policy must
    not share a cache — the derived positives would be different."""
    universe = universe_for("dog")
    relaxed = C.CategoryUniverse(
        categories=universe.categories,
        category_ids=universe.category_ids,
        supercategories={},
        sources=universe.sources,
        # Same category, no exclusion phrase: 'hot dog' would now count.
        specs={"dog": C.LexicalSpec(category="dog", terms=("dog", "dogs"))},
    )
    assert fingerprint_for(tmp_path, universe=universe).digest != fingerprint_for(
        tmp_path, universe=relaxed
    ).digest


def test_the_fingerprint_holds_no_user_specific_absolute_paths(tmp_path):
    payload = json.dumps(fingerprint_for(tmp_path).to_dict())
    assert str(tmp_path) not in payload
    assert "MyDrive" not in payload
    assert "instances_train2014.json" in payload


# -------------------------------------------------------------------- states


def test_an_empty_root_is_a_miss(cache):
    assert cache.state() == K.STATE_MISS
    assert "MISS" in cache.describe()


def test_publishing_then_reading_back_is_a_hit(cache):
    result = cache.publish(ARTIFACTS)
    assert result["published"]
    assert cache.state() == K.STATE_HIT
    assert "HIT" in cache.describe()

    loaded = cache.load()
    assert loaded["metadata.json"] == ARTIFACTS["metadata.json"]
    assert loaded["selected_concepts.json"]["selected"] == ["zebra", "giraffe"]
    assert loaded["concept_evidence_index.jsonl.gz"] == ARTIFACTS[
        "concept_evidence_index.jsonl.gz"
    ]


def test_the_success_marker_is_written_last_and_lists_every_file(cache):
    success = cache.publish(ARTIFACTS)
    assert set(success["artifacts"]) == set(K.ARTIFACT_NAMES)
    for entry in success["artifacts"].values():
        assert entry["checksum"].startswith("sha256:")
        assert entry["size_bytes"] > 0
    assert success["source_metadata_mutated"] is False
    assert success["credentials_stored"] is False


def test_a_directory_without_a_success_marker_is_incomplete_not_usable(cache):
    cache.publish(ARTIFACTS)
    cache.success_path.unlink()
    assert cache.state() == K.STATE_INCOMPLETE
    assert "no valid success marker" in cache.describe()
    with pytest.raises(K.CorruptCacheError, match="incomplete publication"):
        cache.load()


def test_an_interrupted_publication_leaves_nothing_readable(tmp_path, cache, monkeypatch):
    """A crash partway through the copy must not produce a usable directory."""
    real = K.shutil.copyfile
    calls = {"n": 0}

    def flaky(source, destination):
        calls["n"] += 1
        if calls["n"] == 3:
            raise OSError("drive went away")
        return real(source, destination)

    monkeypatch.setattr(K.shutil, "copyfile", flaky)
    with pytest.raises(OSError, match="drive went away"):
        cache.publish(ARTIFACTS)
    assert cache.directory.is_dir()
    assert not cache.success_path.exists()
    assert cache.state() == K.STATE_INCOMPLETE

    # And a later, complete run repairs it rather than reading the fragments.
    monkeypatch.setattr(K.shutil, "copyfile", real)
    assert cache.publish(ARTIFACTS)["published"]
    assert cache.state() == K.STATE_HIT


def test_a_missing_listed_artifact_is_incomplete(cache):
    cache.publish(ARTIFACTS)
    (cache.directory / "pilot_subset.json").unlink()
    assert cache.state() == K.STATE_INCOMPLETE
    with pytest.raises(K.CorruptCacheError, match="listed in _SUCCESS.json but absent"):
        cache.load()


def test_a_tampered_artifact_fails_its_per_file_checksum(cache):
    cache.publish(ARTIFACTS)
    (cache.directory / "concept_coverage.json").write_text(
        json.dumps({"rows": [{"concept": "zebra", "feasible": True, "n_groups": 999}]}),
        encoding="utf-8",
    )
    assert cache.state() == K.STATE_HIT, "the file is still present, so only the checksum can catch it"
    with pytest.raises(K.CorruptCacheError, match="failed its checksum"):
        cache.load()


def test_a_truncated_gzip_index_fails_its_checksum_too(cache):
    cache.publish(ARTIFACTS)
    path = cache.directory / "concept_evidence_index.jsonl.gz"
    path.write_bytes(gzip.compress(b'{"group_id": "tampered"}\n'))
    with pytest.raises(K.CorruptCacheError, match="failed its checksum"):
        cache.load()


def test_a_different_fingerprint_in_the_same_directory_is_refused(tmp_path, cache):
    cache.publish(ARTIFACTS)
    other = K.DerivedCache(
        cache.root, fingerprint_for(tmp_path, seed="other"), staging_root=tmp_path / "s2"
    )
    # Same root, different fingerprint: a different directory, so a clean miss.
    assert other.state() == K.STATE_MISS
    assert other.directory != cache.directory

    # Force the collision the digest prefix normally prevents.
    stored = json.loads(cache.success_path.read_text(encoding="utf-8"))
    stored["fingerprint_digest"] = "sha256:something-else"
    cache.success_path.write_text(json.dumps(stored), encoding="utf-8")
    assert cache.state() == K.STATE_INCOMPATIBLE
    assert "refusing to reuse" in cache.describe().lower()
    with pytest.raises(K.IncompatibleCacheError, match="different fingerprint"):
        cache.load()
    with pytest.raises(K.IncompatibleCacheError, match="refusing to overwrite"):
        cache.publish(ARTIFACTS)


def test_compatibility_is_never_inferred_from_the_directory_name(tmp_path, cache):
    """A directory named for our fingerprint but holding someone else's
    artifacts must be refused, not read."""
    cache.directory.mkdir(parents=True)
    (cache.directory / K.SUCCESS_NAME).write_text(
        json.dumps({"schema_version": K.CACHE_SCHEMA_VERSION,
                    "fingerprint_digest": "sha256:not-ours", "artifacts": {}}),
        encoding="utf-8",
    )
    assert cache.state() == K.STATE_INCOMPATIBLE


def test_an_incomplete_stage_is_refused_before_anything_reaches_drive(cache):
    partial = {name: ARTIFACTS[name] for name in list(ARTIFACTS)[:3]}
    with pytest.raises(K.CacheError, match="refusing to publish an incomplete cache"):
        cache.publish(partial)
    assert not cache.directory.exists()


def test_artifacts_are_staged_locally_before_publication(cache):
    cache.stage(ARTIFACTS)
    assert cache.staging_directory.is_dir()
    assert cache.staging_root != cache.root
    for name in K.ARTIFACT_NAMES:
        assert (cache.staging_directory / name).is_file()
    assert not cache.directory.exists(), "nothing may reach the cache root before publish"


def test_republishing_a_hit_is_a_no_op(cache):
    cache.publish(ARTIFACTS)
    again = cache.publish(ARTIFACTS)
    assert again["published"] is False
    assert again["reason"] == "already published"


def test_the_index_is_one_stream_not_thousands_of_tiny_files(cache):
    cache.publish(
        {**ARTIFACTS, "concept_evidence_index.jsonl.gz": [{"i": i} for i in range(500)]}
    )
    assert len(list(cache.directory.iterdir())) == len(K.ARTIFACT_NAMES) + 1
    assert len(cache.load()["concept_evidence_index.jsonl.gz"]) == 500


# ------------------------------------------------------------ relative paths


def test_media_is_serialized_relative_to_the_configured_roots(tmp_path):
    roots = {"image": [tmp_path / "coco"], "audio": [tmp_path / "SpokenCOCO"]}
    rows = [
        {
            "group_id": "g0",
            "image_path": str(tmp_path / "coco" / "train2014" / "a.jpg"),
            "audio_path": str(tmp_path / "SpokenCOCO" / "wavs" / "train" / "a.wav"),
        }
    ]
    relative = K.to_relative_rows(rows, roots)
    assert relative[0]["image_relpath"] == "train2014/a.jpg"
    assert relative[0]["audio_relpath"] == "wavs/train/a.wav"
    assert "image_path" not in relative[0] and "audio_path" not in relative[0]
    assert str(tmp_path) not in json.dumps(relative)


def test_relative_media_is_re_resolved_against_this_sessions_roots(tmp_path):
    roots = {"image": [tmp_path / "coco"], "audio": [tmp_path / "SpokenCOCO"]}
    image = tmp_path / "coco" / "train2014" / "a.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"x")
    restored = K.to_absolute_rows([{"group_id": "g0", "image_relpath": "train2014/a.jpg"}], roots)
    assert restored[0]["image_path"] == str(image)
    # A file that is no longer there resolves to None rather than vanishing, so
    # the caller can report the gap instead of silently shrinking the subset.
    missing = K.to_absolute_rows([{"group_id": "g1", "image_relpath": "train2014/b.jpg"}], roots)
    assert missing[0]["image_path"] is None


def test_a_subset_round_trips_through_relative_form(tmp_path):
    roots = {"image": [tmp_path / "coco"], "audio": [tmp_path / "SpokenCOCO"]}
    image = tmp_path / "coco" / "train2014" / "a.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"x")
    subset = {
        "concepts": {"zebra": {}},
        "splits": {
            "train": [{"group_id": "g0", "image_path": str(image), "concept": "zebra"}],
            "test": [],
        },
    }
    relative = K.subset_to_relative(subset, roots)
    assert relative["paths_are_relative"] is True
    assert str(tmp_path) not in json.dumps(relative)
    restored = K.subset_to_absolute(relative, roots)
    assert restored["splits"]["train"][0]["image_path"] == str(image)
    assert restored["splits"]["train"][0]["concept"] == "zebra"


def test_a_cache_hit_probes_a_few_media_paths_rather_than_revalidating(tmp_path, cache):
    roots = {"image": [tmp_path / "coco"], "audio": [tmp_path / "SpokenCOCO"]}
    for relative in ("train2014/a.jpg", "train2014/b.jpg"):
        path = tmp_path / "coco" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    rows = [{"group_id": f"g{i}", "image_relpath": "train2014/a.jpg"} for i in range(50)]
    probe = cache.probe_media(rows, roots, n_probes=4)
    assert probe["ok"] and probe["n_probed"] == 4

    gone = cache.probe_media([{"group_id": "g", "image_relpath": "train2014/z.jpg"}], roots)
    assert not gone["ok"] and gone["n_unresolved"] == 1


# ------------------------------------------------------------- immutability


def test_publishing_never_touches_the_source_metadata(tmp_path, cache):
    manifest = tmp_path / "coco" / "annotations" / "instances_train2014.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"categories": [], "annotations": []}), encoding="utf-8")
    before = (manifest.read_bytes(), manifest.stat().st_mtime_ns)

    cache.publish(ARTIFACTS)
    cache.load()

    assert manifest.read_bytes() == before[0]
    assert manifest.stat().st_mtime_ns == before[1]
    assert cache.root not in manifest.parents
