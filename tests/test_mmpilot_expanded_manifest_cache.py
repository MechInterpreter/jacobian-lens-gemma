# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Loading the expanded manifest without rebuilding the evidence join.

``persist_expanded_manifest`` checks compatibility only *after* its caller has
built the whole :class:`~jlens.mmpilot.expansion.ExpansionResult`, so a cache hit
still paid for the miss. ``load_expanded_manifest`` is the compatibility half on
its own, and the central test here is the negative one: it works with
``build_expanded_manifest`` monkeypatched to raise.
"""

import json

import pytest

from jlens.mmpilot import expansion as expansion_module
from jlens.mmpilot.concepts import (
    RECOVERED_UNIVERSE_SOURCE,
    CategoryDiscoveryError,
    universe_from_concept_annotations,
)
from jlens.mmpilot.expansion import (
    DERIVATION_SCHEMA_VERSION,
    ExpandedManifestIncompatible,
    expanded_manifest_compatibility,
    load_expanded_manifest,
    persist_expanded_manifest,
)
from jlens.mmpilot.store import payload_checksum

CONVERSION = {
    "converter": "jlens.mmpilot.expansion.build_expanded_manifest",
    "evidence_rule": "visual_annotation_AND_caption_lexicon",
    "evidence_lexicon_hash": "sha256:lexicon",
    "reads_only": True,
}
ORIGINAL = "sha256:original-manifest"
SOURCES = {"/data/captions.json": "sha256:a", "/data/instances.json": "sha256:b"}


def _groups(n=4):
    return [
        {
            "group_id": f"g_{index}",
            "image_id": f"img_{index}",
            "caption": f"a photo of a cat {index}",
            "audio_path": f"wavs/{index}.wav",
            "image_path": f"images/{index}.jpg",
            "concept_annotations": ["cat", "chair"] if index % 2 else ["zebra"],
        }
        for index in range(n)
    ]


def _cache(tmp_path, **overrides):
    payload = {
        "schema_version": DERIVATION_SCHEMA_VERSION,
        "original_manifest_checksum": ORIGINAL,
        "source_metadata_checksums": dict(SOURCES),
        "conversion": dict(CONVERSION),
        "conversion_hash": payload_checksum(dict(CONVERSION)),
        "n_groups": 4,
        "groups": _groups(),
    }
    payload.update(overrides)
    target = tmp_path / "expanded_manifest.json"
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


def _load(target, **overrides):
    kwargs = {
        "original_checksum": ORIGINAL,
        "expected_sources": SOURCES,
        "conversion": CONVERSION,
    }
    kwargs.update(overrides)
    return load_expanded_manifest(target, **kwargs)


# --------------------------------------------------------------- the hit path


def test_a_compatible_cache_loads_its_groups(tmp_path):
    payload, record = _load(_cache(tmp_path))
    assert record["compatible"] is True
    assert record["failed_clauses"] == []
    assert len(payload["groups"]) == 4
    assert record["manifest_file_checksum"].startswith("sha256:")


def test_the_loader_never_calls_build_expanded_manifest(tmp_path, monkeypatch):
    """The whole point, asserted rather than described."""

    def explode(*args, **kwargs):
        raise AssertionError(
            "build_expanded_manifest was called on a compatible cache path"
        )

    monkeypatch.setattr(expansion_module, "build_expanded_manifest", explode)
    payload, record = _load(_cache(tmp_path))
    assert record["compatible"] is True
    assert payload["n_groups"] == 4


def test_the_expected_group_count_is_checked_when_given(tmp_path):
    payload, record = _load(_cache(tmp_path), expected_group_count=4)
    assert record["compatible"] is True
    assert payload["n_groups"] == 4


def test_the_lexicon_hash_is_checked_when_given(tmp_path):
    _, record = _load(_cache(tmp_path), expected_lexicon_hash="sha256:lexicon")
    assert record["compatible"] is True


def test_the_cache_is_opened_read_only(tmp_path):
    target = _cache(tmp_path)
    before = target.read_bytes()
    _load(target)
    assert target.read_bytes() == before


# ------------------------------------------------------------ the refusals


def test_a_missing_cache_refuses_rather_than_building(tmp_path):
    with pytest.raises(ExpandedManifestIncompatible, match="does not build one"):
        _load(tmp_path / "absent.json")


def test_unreadable_json_refuses(tmp_path):
    target = tmp_path / "expanded_manifest.json"
    target.write_text("{not json", encoding="utf-8")
    with pytest.raises(ExpandedManifestIncompatible, match="not readable JSON"):
        _load(target)


@pytest.mark.parametrize(
    ("override", "clause"),
    [
        ({"schema_version": "jlens.mmpilot.expanded_manifest.v1"}, "schema_version"),
        ({"original_manifest_checksum": "sha256:other"},
         "original_manifest_checksum"),
        ({"source_metadata_checksums": {"/data/captions.json": "sha256:moved"}},
         "source_metadata_checksums"),
        ({"conversion_hash": "sha256:different"}, "conversion_hash"),
    ],
)
def test_each_provenance_clause_refuses_on_its_own(tmp_path, override, clause):
    with pytest.raises(ExpandedManifestIncompatible, match=clause):
        _load(_cache(tmp_path, **override))


def test_a_wrong_group_count_refuses(tmp_path):
    with pytest.raises(ExpandedManifestIncompatible, match="n_groups"):
        _load(_cache(tmp_path), expected_group_count=125198)


def test_a_wrong_lexicon_hash_refuses(tmp_path):
    with pytest.raises(ExpandedManifestIncompatible, match="evidence_lexicon_hash"):
        _load(_cache(tmp_path), expected_lexicon_hash="sha256:other-lexicon")


def test_a_manifest_with_no_groups_refuses(tmp_path):
    target = _cache(tmp_path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    del payload["groups"]
    target.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ExpandedManifestIncompatible, match="no 'groups' list"):
        _load(target)


def test_the_refusal_names_both_values(tmp_path):
    with pytest.raises(ExpandedManifestIncompatible) as error:
        _load(_cache(tmp_path, original_manifest_checksum="sha256:other"))
    message = str(error.value)
    assert ORIGINAL in message
    assert "sha256:other" in message


def test_the_refusal_carries_the_clause_record(tmp_path):
    with pytest.raises(ExpandedManifestIncompatible) as error:
        _load(_cache(tmp_path, conversion_hash="sha256:x"))
    assert error.value.record["failed_clauses"] == ["conversion_hash"]


# ------------------------------------------------------- shared with persist


def test_persist_uses_the_same_compatibility_clauses(tmp_path):
    class _Source:
        def __init__(self, path, checksum):
            self.path, self.checksum = path, checksum

    result = expansion_module.ExpansionResult(
        groups=_groups(),
        sources=[_Source(path, checksum) for path, checksum in SOURCES.items()],
        per_source=[],
    )
    target = _cache(tmp_path)
    _, status = persist_expanded_manifest(
        target, result, original_checksum=ORIGINAL, conversion=CONVERSION
    )
    assert status.startswith("resuming")


def test_the_compatibility_record_is_pure(tmp_path):
    record = expanded_manifest_compatibility(
        None,
        original_checksum=ORIGINAL,
        expected_sources=SOURCES,
        conversion=CONVERSION,
    )
    assert record["compatible"] is False
    assert len(record["failed_clauses"]) == 4


# ------------------------------------------------- recovering the universe


def test_the_universe_is_recovered_from_persisted_annotations():
    universe = universe_from_concept_annotations(_groups())
    assert set(universe.categories) == {"cat", "chair", "zebra"}
    assert universe.lexicon()
    assert universe.sources[0]["path"] == RECOVERED_UNIVERSE_SOURCE


def test_the_recovered_universe_admits_it_has_no_category_ids():
    universe = universe_from_concept_annotations(_groups())
    assert universe.sources[0]["category_ids_available"] is False
    assert all(ids == () for ids in universe.category_ids.values())
    assert "NOT comparable" in universe.sources[0]["note"]


def test_the_recovered_universe_hash_differs_from_a_discovered_one():
    # Not a defect: the two were established from different evidence, and a
    # matching hash would make the cache the authority on ids it never saw.
    from jlens.mmpilot.concepts import CategoryUniverse, lexical_spec

    recovered = universe_from_concept_annotations(_groups())
    discovered = CategoryUniverse(
        categories=recovered.categories,
        category_ids={name: (index + 1,) for index, name in enumerate(recovered.categories)},
        supercategories={},
        sources=(),
        specs={name: lexical_spec(name) for name in recovered.categories},
    )
    assert recovered.universe_hash != discovered.universe_hash
    assert recovered.lexical_hash == discovered.lexical_hash


def test_a_cache_with_no_annotations_refuses():
    groups = [{"group_id": "g", "image_id": "i", "concept_annotations": []}]
    with pytest.raises(CategoryDiscoveryError, match="visual half"):
        universe_from_concept_annotations(groups)


def test_the_recovered_universe_is_capped_at_what_the_cache_contains():
    universe = universe_from_concept_annotations(
        [{"group_id": "g", "image_id": "i", "concept_annotations": ["cat"]}]
    )
    assert universe.categories == ("cat",)
