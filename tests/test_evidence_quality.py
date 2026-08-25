# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Tests for the frozen animal-evidence-quality gate.

Every rejection case here is modeled directly on one of the four compromised
cow photographs found by inspecting the earlier cat/cow run: a distant speck
(``COCO_train2014_000000481142``), a promotional statue
(``COCO_val2014_000000386718``), an unlabeled competing animal named only in
the caption (``COCO_val2014_000000193162``), and a background competing animal
(``COCO_val2014_000000467776``). The gate exists specifically to catch these.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jlens.mmpilot.evidence_quality import (
    ANIMAL_CATEGORIES,
    DEFAULT_THRESHOLDS,
    EvidenceQualityRefused,
    EvidenceQualityThresholds,
    _depiction_words_near_target,
    build_clean_evidence_index,
    evaluate_image_evidence_quality,
    filter_synchronized_groups,
    freeze_disjoint_populations,
)


def _clean_cat(**overrides) -> dict:
    base = dict(
        target="cat",
        area_by_category={"cat": [50_000.0], "keyboard": [4_000.0]},
        image_area=200_000.0,
        captions=[
            "a cat sitting on a keyboard",
            "a cat resting near a laptop",
            "cats love keyboards apparently",
            "a fluffy cat on a desk",
        ],
    )
    base.update(overrides)
    return base


# ------------------------------------------------------------- clean passes


def test_a_clean_dominant_unambiguous_photo_passes() -> None:
    result = evaluate_image_evidence_quality(**_clean_cat())
    assert result["passed"] is True
    assert result["failed_criteria"] == []
    assert result["animal_species_present"] == ["cat"]


def test_multiple_instances_of_the_target_species_are_fine() -> None:
    """Several cows in one frame is not 'more than one species'."""
    result = evaluate_image_evidence_quality(
        target="cow",
        area_by_category={"cow": [10_000.0, 8_000.0, 6_000.0]},
        image_area=100_000.0,
        captions=["a herd of cows", "several cows crossing a field",
                  "cows walking together", "a group of cows"],
    )
    assert result["passed"] is True


# ------------------------------------------------------- the four real cases


def test_a_distant_speck_fails_prominence() -> None:
    """Modeled on COCO_train2014_000000481142: cow ~0.26% of frame."""
    result = evaluate_image_evidence_quality(
        target="cow",
        area_by_category={"cow": [500.0], "boat": [40_000.0]},
        image_area=190_000.0,
        captions=["a small boat in a body of water",
                   "a boat floating along a river",
                   "a sepia photo of a boat on a calm river",
                   "a boat docked along a riverbank"],
    )
    assert result["passed"] is False
    assert "target_sufficiently_prominent" in result["failed_criteria"]
    # this fixture also fails caption naming -- the boat dominates the words too
    assert "target_consistently_named_in_captions" in result["failed_criteria"]


def test_a_statue_is_caught_by_the_depiction_lexicon() -> None:
    """Modeled on COCO_val2014_000000386718: 'a statue of a cow' / 'a fake cow'."""
    result = evaluate_image_evidence_quality(
        target="cow",
        area_by_category={"cow": [30_000.0], "person": [55_000.0]},
        image_area=180_000.0,
        captions=[
            "a man wearing a tour de france shirt stands beside a statue of a cow",
            "a man standing next to a fake cow smiling for the camera",
            "a male in a black shirt next to a cow and sign",
            "a man poses with a plastic cow at a fair",
        ],
    )
    assert result["passed"] is False
    assert "no_depiction_word_in_captions" in result["failed_criteria"]
    assert "statue" in result["depiction_words_found"]
    assert "fake" in result["depiction_words_found"]
    assert "plastic" in result["depiction_words_found"]


def test_a_competing_animal_named_only_in_caption_is_caught() -> None:
    """Modeled on COCO_val2014_000000193162: caption says 'two dogs and a cow'.

    The COCO object detector never tagged the dog as an instance -- this is
    exactly the case ``concept_annotations``-only filtering misses, and why
    every caption is checked here, not just the object labels.
    """
    result = evaluate_image_evidence_quality(
        target="cow",
        area_by_category={"cow": [25_000.0], "person": [4_000.0], "sheep": [3_000.0]},
        image_area=150_000.0,
        captions=[
            "a man on a three wheeler following behind two dogs and a cow",
            "a black and white cow walking across a lush green field",
            "a man is right behind a cow on green grass",
            "a farmer herding a cow with his dogs",
        ],
    )
    assert result["passed"] is False
    assert "no_competing_animal_word_in_captions" in result["failed_criteria"]
    assert "dog" in result["competing_animal_words"]
    # the sheep IS a detected instance, so criterion 1 fails too
    assert "exactly_one_animal_species" in result["failed_criteria"]
    assert result["animal_species_present"] == ["cow", "sheep"]


def test_a_background_competing_animal_instance_is_caught() -> None:
    """Modeled on COCO_val2014_000000467776: sheep visible with the cow+calf."""
    result = evaluate_image_evidence_quality(
        target="cow",
        area_by_category={"cow": [20_000.0, 8_000.0], "sheep": [500.0, 400.0]},
        image_area=150_000.0,
        captions=["a mother and its calf standing in a field",
                  "a cow and a calf standing in a field",
                  "a cow and calf are standing in a grassy pasture",
                  "two cows in a pasture"],
    )
    assert result["passed"] is False
    assert "exactly_one_animal_species" in result["failed_criteria"]
    assert result["animal_species_present"] == ["cow", "sheep"]


# ---------------------------------------------- the depiction-proximity fix
#
# Found by running the shipped gate against real COCO captions and inspecting
# every "no_depiction_word_in_captions" rejection by hand (~2,400-image
# sample): a bare "does this word appear anywhere in the caption" check
# rejected ~150 real cat/dog photographs because a depiction word described a
# *different* object in the scene, not the animal. Every case below is a
# caption actually seen in that inspection.


@pytest.mark.parametrize(
    "caption, target",
    [
        ("A dog is sitting under a stone arch", "dog"),
        ("A black cat looking at a statue that is sitting in rocks.", "cat"),
        (
            "A cat on a plastic mat in a bathtub with water droplets "
            "falling, with tile on wall.",
            "cat",
        ),
        (
            'A brown and white dog standing next to sign that reads '
            '"beware of dog."',
            "dog",
        ),
        ("A small furry dog snuggles in a plush bed.", "dog"),
        ("a white and woolly dog lying while holding a doll", "dog"),
        # this caption never mentions the animal at all -- a different
        # caption of the same image presumably does, and must not be
        # contaminated by this one
        ("Large brown wooden door on the side of a building.", "dog"),
        ("A picture of a dog that is looking out the window.", "dog"),
        ("This is a photo of a sad looking black lab dog.", "dog"),
        ("a dog sits on a beach inside of a drawn heart", "dog"),
        ("A garden with various statues, plants and a tree.", "dog"),
    ],
)
def test_depiction_words_describing_something_else_do_not_disqualify(
    caption: str, target: str
) -> None:
    assert _depiction_words_near_target(caption, target) == set()


@pytest.mark.parametrize(
    "caption, target, expected_word",
    [
        (
            "A man standing next to a fake cow, and smiling for the camera.",
            "cow", "fake",
        ),
        ("a male in a black shirt next to a cow statue and sign", "cow", "statue"),
        (
            "a man wearing a tour de france shirt stands beside a statue "
            "of a cow",
            "cow", "statue",
        ),
        ("A small stuffed cat sits on the shelf", "cat", "stuffed"),
        ("a toy dog on the table", "dog", "toy"),
        ("a dog figurine painted blue", "dog", "figurine"),
    ],
)
def test_a_depiction_word_adjacent_to_the_target_still_disqualifies(
    caption: str, target: str, expected_word: str
) -> None:
    assert expected_word in _depiction_words_near_target(caption, target)


def test_picture_of_and_photo_of_and_image_of_are_not_in_the_lexicon() -> None:
    """The single biggest false-positive source, removed rather than patched.

    Every genuine depiction these phrases might have caught is already caught
    by its own specific noun ("statue", "toy", "painting", ...) sitting next
    to the animal, so dropping them costs no real coverage -- and they are
    indistinguishable from ordinary COCO caption boilerplate ("A picture of a
    dog...") by adjacency alone.
    """
    from jlens.mmpilot.evidence_quality import DEPICTION_LEXICON

    for phrase in ("picture of", "photo of", "image of"):
        assert phrase not in DEPICTION_LEXICON


# --------------------------------------------------------- individual clauses


def test_person_dominance_fails_independently() -> None:
    result = evaluate_image_evidence_quality(
        target="cat",
        area_by_category={"cat": [10_000.0], "person": [70_000.0]},
        image_area=150_000.0,
        captions=["a cat by a person", "a person with a cat",
                   "someone holding a cat", "a cat and its owner"],
    )
    assert result["passed"] is False
    assert "person_not_dominant" in result["failed_criteria"]
    assert result["person_area_fraction"] == pytest.approx(70_000 / 150_000)


def test_inconsistent_caption_naming_fails_even_with_a_clean_photo() -> None:
    """Detected clearly, but only one of four captions actually says 'cat'."""
    result = evaluate_image_evidence_quality(
        target="cat",
        area_by_category={"cat": [60_000.0]},
        image_area=150_000.0,
        captions=["an animal on a chair", "a furry pet resting",
                   "something sleeping in the sun", "a cat napping"],
    )
    assert result["passed"] is False
    assert "target_consistently_named_in_captions" in result["failed_criteria"]
    assert result["n_captions_naming_target"] == 1
    assert result["required_caption_matches"] == 4  # ceil(0.8*4)=4, floor=1


def test_the_target_word_matches_its_plural() -> None:
    result = evaluate_image_evidence_quality(
        target="cow",
        area_by_category={"cow": [30_000.0]},
        image_area=100_000.0,
        captions=["several cows in a field"] * 4,
    )
    assert "target_consistently_named_in_captions" not in result["failed_criteria"]


def test_zero_captions_never_passes_the_naming_clause() -> None:
    result = evaluate_image_evidence_quality(
        target="cat", area_by_category={"cat": [50_000.0]},
        image_area=100_000.0, captions=[],
    )
    assert "target_consistently_named_in_captions" in result["failed_criteria"]


def test_an_unknown_target_is_refused() -> None:
    with pytest.raises(EvidenceQualityRefused):
        evaluate_image_evidence_quality(
            target="wolf", area_by_category={}, image_area=1.0, captions=[],
        )


def test_zero_image_area_is_refused() -> None:
    with pytest.raises(EvidenceQualityRefused):
        evaluate_image_evidence_quality(
            target="cat", area_by_category={}, image_area=0.0, captions=[],
        )


# ------------------------------------------------------------- the thresholds


def test_thresholds_are_frozen_and_reject_out_of_range_values() -> None:
    with pytest.raises(ValueError):
        EvidenceQualityThresholds(min_area_fraction=0.0)
    with pytest.raises(ValueError):
        EvidenceQualityThresholds(min_area_fraction=1.5)
    with pytest.raises(ValueError):
        EvidenceQualityThresholds(min_caption_matches_floor=0)


def test_thresholds_digest_changes_with_any_field() -> None:
    base = EvidenceQualityThresholds()
    looser = EvidenceQualityThresholds(min_area_fraction=0.01)
    assert base.digest != looser.digest


def test_default_thresholds_are_exactly_the_frozen_values() -> None:
    # documents the frozen numbers so a silent edit is caught by this test
    assert DEFAULT_THRESHOLDS.min_area_fraction == 0.05
    assert DEFAULT_THRESHOLDS.min_caption_match_ratio == 0.80
    assert DEFAULT_THRESHOLDS.max_person_fraction == 0.30
    assert "statue" in DEFAULT_THRESHOLDS.depiction_lexicon
    assert "fake" in DEFAULT_THRESHOLDS.depiction_lexicon


# --------------------------------------------------------- the raw-file index


def _write_coco_fixture(tmp_path: Path, split: str, images: list[dict]) -> tuple[Path, Path]:
    """Build a tiny instances_*.json / captions_*.json pair.

    ``images`` is ``[{id, w, h, objects: {cat_name: [areas]}, captions: [...]}]``.
    """
    categories = {name: idx + 1 for idx, name in enumerate(ANIMAL_CATEGORIES + ("person", "boat"))}
    instances = {
        "images": [{"id": im["id"], "width": im["w"], "height": im["h"],
                     "file_name": f"{split}_{im['id']:012d}.jpg"} for im in images],
        "categories": [{"id": cid, "name": name} for name, cid in categories.items()],
        "annotations": [
            {"image_id": im["id"], "category_id": categories[name], "area": area}
            for im in images
            for name, areas in im["objects"].items()
            for area in areas
        ],
    }
    captions = {
        "annotations": [
            {"image_id": im["id"], "caption": c}
            for im in images for c in im["captions"]
        ]
    }
    inst_path = tmp_path / f"instances_{split}.json"
    cap_path = tmp_path / f"captions_{split}.json"
    inst_path.write_text(json.dumps(instances), encoding="utf-8")
    cap_path.write_text(json.dumps(captions), encoding="utf-8")
    return inst_path, cap_path


def test_build_clean_evidence_index_from_raw_files(tmp_path: Path) -> None:
    inst, cap = _write_coco_fixture(tmp_path, "val2014", [
        {"id": 1, "w": 400, "h": 500, "objects": {"cat": [100_000.0]},
         "captions": ["a cat on a bed"] * 4},
        {"id": 2, "w": 400, "h": 500, "objects": {"cow": [500.0], "boat": [50_000.0]},
         "captions": ["a boat on a river"] * 4},  # fails prominence AND naming
        {"id": 3, "w": 400, "h": 500,
         "objects": {"dog": [90_000.0], "person": [10_000.0]},
         "captions": ["a dog with a person"] * 4},
    ])
    index = build_clean_evidence_index(
        [inst], [cap], targets=("cat", "cow", "dog"),
    )
    assert index["n_candidates_scored"] == 3
    assert index["n_approved"] == {"cat": 1, "cow": 0, "dog": 1}
    assert [row["image_id"] for row in index["approved"]["cat"]] == [1]
    assert index["rejected_counts"]["cow"]["target_sufficiently_prominent"] == 1
    assert index["index_checksum"].startswith("sha256:")
    # thresholds are embedded, not implied
    assert index["thresholds"]["min_area_fraction"] == 0.05


def test_build_clean_evidence_index_refuses_mismatched_split_lists(tmp_path: Path) -> None:
    inst, cap = _write_coco_fixture(tmp_path, "val2014", [])
    with pytest.raises(EvidenceQualityRefused):
        build_clean_evidence_index([inst, inst], [cap], targets=("cat",))


def test_build_clean_evidence_index_refuses_unknown_targets(tmp_path: Path) -> None:
    inst, cap = _write_coco_fixture(tmp_path, "val2014", [])
    with pytest.raises(EvidenceQualityRefused):
        build_clean_evidence_index([inst], [cap], targets=("wolf",))


def test_two_splits_are_merged(tmp_path: Path) -> None:
    inst_v, cap_v = _write_coco_fixture(tmp_path, "val2014", [
        {"id": 10, "w": 300, "h": 300, "objects": {"cat": [50_000.0]},
         "captions": ["a cat"] * 4},
    ])
    inst_t, cap_t = _write_coco_fixture(tmp_path, "train2014", [
        {"id": 20, "w": 300, "h": 300, "objects": {"dog": [50_000.0]},
         "captions": ["a dog"] * 4},
    ])
    index = build_clean_evidence_index(
        [inst_v, inst_t], [cap_v, cap_t], targets=("cat", "dog"),
    )
    assert index["n_approved"] == {"cat": 1, "dog": 1}
    assert index["approved"]["cat"][0]["split"] == "val2014"
    assert index["approved"]["dog"][0]["split"] == "train2014"


# ------------------------------------------------- synchronized-group filter


def test_filter_synchronized_groups_requires_all_media_present() -> None:
    index = {"approved": {"cat": [{"image_id": 12345}]}}
    groups = [
        {"group_id": "g1", "image_id": "COCO_val2014_000000012345",
         "caption": "a cat", "image_path": "/x.jpg", "audio_path": "/x.wav"},
        {"group_id": "g2", "image_id": "COCO_val2014_000000012345",
         "caption": "a cat", "image_path": "/x.jpg", "audio_path": ""},  # no audio
        {"group_id": "g3", "image_id": "COCO_val2014_000000099999",
         "caption": "a cat", "image_path": "/y.jpg", "audio_path": "/y.wav"},  # not approved
    ]
    out = filter_synchronized_groups(index, groups, target="cat")
    assert [g["group_id"] for g in out] == ["g1"]


# ---------------------------------------------------- disjoint population freeze


def _groups(concept: str, n: int) -> list[dict]:
    return [
        {"group_id": f"{concept}-g{i:03d}", "image_id": f"{concept}-i{i:03d}",
         "caption": f"a {concept}", "image_path": f"/{concept}/{i}.jpg",
         "audio_path": f"/{concept}/{i}.wav"}
        for i in range(n)
    ]


def test_freeze_disjoint_populations_produces_non_overlapping_pools() -> None:
    payload = freeze_disjoint_populations(
        {"cat": _groups("cat", 20), "dog": _groups("dog", 20)},
        n_dev_per_concept=8, n_confirm_per_concept=8, seed="test-seed-1",
    )
    assert payload["disjoint"] is True
    assert payload["overlap"] == []
    dev_ids = {row["image_id"] for rows in payload["development"].values() for row in rows}
    conf_ids = {row["image_id"] for rows in payload["confirmation"].values() for row in rows}
    assert dev_ids.isdisjoint(conf_ids)
    assert payload["n_development"] == {"cat": 8, "dog": 8}
    assert payload["n_confirmation"] == {"cat": 8, "dog": 8}
    assert "freeze_digest" in payload


def test_freeze_disjoint_populations_refuses_too_small_a_pool() -> None:
    with pytest.raises(EvidenceQualityRefused):
        freeze_disjoint_populations(
            {"cat": _groups("cat", 10)},
            n_dev_per_concept=8, n_confirm_per_concept=8, seed="s",
        )


def test_freeze_disjoint_populations_is_deterministic_under_the_same_seed() -> None:
    groups = {"cat": _groups("cat", 30)}
    a = freeze_disjoint_populations(
        groups, n_dev_per_concept=5, n_confirm_per_concept=5, seed="fixed"
    )
    b = freeze_disjoint_populations(
        groups, n_dev_per_concept=5, n_confirm_per_concept=5, seed="fixed"
    )
    assert a["freeze_digest"] == b["freeze_digest"]
    assert a["development"] == b["development"]


def test_freeze_disjoint_populations_changes_with_a_different_seed() -> None:
    groups = {"cat": _groups("cat", 30)}
    a = freeze_disjoint_populations(
        groups, n_dev_per_concept=5, n_confirm_per_concept=5, seed="seed-a"
    )
    b = freeze_disjoint_populations(
        groups, n_dev_per_concept=5, n_confirm_per_concept=5, seed="seed-b"
    )
    assert a["development_image_ids"] != b["development_image_ids"]


def test_freeze_disjoint_populations_dedupes_multiple_groups_per_image() -> None:
    dup = _groups("cat", 10)
    # two groups pointing at the same photograph
    dup.append({**dup[0], "group_id": "cat-g000-dup"})
    payload = freeze_disjoint_populations(
        {"cat": dup}, n_dev_per_concept=5, n_confirm_per_concept=4, seed="s",
    )
    all_ids = [row["image_id"] for row in payload["development"]["cat"]]
    all_ids += [row["image_id"] for row in payload["confirmation"]["cat"]]
    assert len(all_ids) == len(set(all_ids))  # every image appears once


def test_freeze_disjoint_populations_rejects_non_positive_counts() -> None:
    with pytest.raises(EvidenceQualityRefused):
        freeze_disjoint_populations({"cat": _groups("cat", 5)},
                                     n_dev_per_concept=0, n_confirm_per_concept=5, seed="s")
