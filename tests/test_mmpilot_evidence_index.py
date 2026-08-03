from __future__ import annotations

from jlens.mmpilot import evidence as E


def _group(index: int, *, caption: str, labels: list[str]) -> dict:
    return {
        "group_id": f"g-{index}",
        "image_id": f"i-{index // 2}",
        "image_path": f"/images/{index // 2}.jpg",
        "audio_path": f"/audio/{index}.wav",
        "caption": caption,
        "concept_annotations": labels,
    }


def test_index_matches_public_evidence_predicates():
    concepts = {"cat": ("cat", "cats"), "bus": ("bus", "buses")}
    config = E.config_for_concepts(concepts)
    groups = [
        _group(0, caption="A cat watches a bus.", labels=["cat", "bus"]),
        _group(1, caption="A cat sleeps.", labels=["cat"]),
        _group(2, caption="A bus at a stop.", labels=["bus"]),
        _group(3, caption="An empty road.", labels=[]),
        _group(4, caption="A cat is described but not shown.", labels=[]),
    ]

    index = E.build_evidence_index(groups, tuple(concepts), config)
    for image_id, image_groups in index.by_image.items():
        expected_valid = {
            concept
            for concept in concepts
            if any(E.is_valid_positive(group, concept, config) for group in image_groups)
        }
        expected_any = {
            concept
            for concept in concepts
            if any(E.has_any_evidence(group, concept, config) for group in image_groups)
        }
        assert index.valid_concepts_by_image[image_id] == expected_valid
        assert index.evidence_concepts_by_image[image_id] == expected_any


def test_index_narrows_exact_evidence_calls(monkeypatch):
    concepts = {f"object-{i}": (f"word{i}",) for i in range(80)}
    config = E.config_for_concepts(concepts)
    groups = [
        _group(i, caption=f"A word{i % 4} is here.", labels=[f"object-{i % 4}"])
        for i in range(2_000)
    ]
    original = E.group_evidence
    calls = 0

    def counted(group, concept, evidence_config):
        nonlocal calls
        calls += 1
        return original(group, concept, evidence_config)

    monkeypatch.setattr(E, "group_evidence", counted)
    index = E.build_evidence_index(groups, tuple(concepts), config)

    assert len(index.by_image) == 1_000
    assert calls == len(groups)
    assert calls < len(groups) * len(concepts) // 20


def test_exclusion_semantics_are_not_bypassed():
    config = E.EvidenceConfig(
        lexicon={"dog": ("dog", "dogs")},
        coco_categories={"dog": ("dog",)},
        exclusions={"dog": ("hot dog", "hot dogs")},
    )
    groups = [_group(0, caption="Two hot dogs on a plate.", labels=["dog"])]

    index = E.build_evidence_index(groups, ("dog",), config)

    assert index.valid_concepts_by_image["i-0"] == set()
    assert index.evidence_concepts_by_image["i-0"] == {"dog"}
    assert index.caption_group_ids_by_concept["dog"] == set()


def test_restricted_index_reuses_the_full_scan_without_revalidation(monkeypatch):
    concepts = {"cat": ("cat",), "bus": ("bus",), "dog": ("dog",)}
    config = E.config_for_concepts(concepts)
    groups = [
        _group(0, caption="A cat sleeps.", labels=["cat"]),
        _group(2, caption="A bus waits.", labels=["bus"]),
    ]
    index = E.build_evidence_index(groups, tuple(concepts), config)

    def forbidden(*args, **kwargs):
        raise AssertionError("restrict must not revalidate manifest rows")

    monkeypatch.setattr(E, "group_evidence", forbidden)
    restricted = index.restrict(("cat", "dog"))

    assert restricted.valid_concepts_by_image["i-0"] == {"cat"}
    assert restricted.valid_concepts_by_image["i-1"] == set()
    assert restricted.by_image is index.by_image
