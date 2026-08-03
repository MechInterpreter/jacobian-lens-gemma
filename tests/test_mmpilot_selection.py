# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Image-unique selection: one group per photograph, before anything is spent.

The image-independence audit repaired a completed run's arithmetic. These tests
are about the earlier and more important thing: a run that never selects two
captions of one photograph in the first place, so its ``n`` is honest by
construction rather than by correction.
"""

import random

import pytest

from jlens.mmpilot import manifest as M
from jlens.mmpilot.evidence import config_for_concepts
from jlens.mmpilot.pipeline import PilotConfig, _select_targets
from jlens.mmpilot.selection import (
    CAUSAL_TARGET_SELECTION_VERSION,
    IMAGE_UNIQUE_PROFILE,
    PILOT_PROFILE,
    InsufficientDistinctImagesError,
    SubsetProfile,
    assert_disjoint_images,
    choose_representative_groups,
    select_distinct_image_records,
    select_focal_concepts,
    unrelated_control_assignment,
)

CONCEPTS = {"cat": ("cat", "cats"), "dog": ("dog", "dogs")}


def group(image_id, index, concept="cat", source_split="train2014"):
    """One synchronized group: an image, a caption, and its recording.

    The caption text is unique per (image, caption) — real SpokenCOCO captions
    describe different photographs, and ``check_split_leakage`` rightly treats
    a repeated caption across the split as leakage.
    """
    return {
        "group_id": f"g_{image_id}_{index}",
        "image_id": image_id,
        "caption": f"a photo of a {concept} in scene {image_id} take {index}",
        "image_path": f"/images/{image_id}.jpg",
        "audio_path": f"/audio/{image_id}_{index}.wav",
        "speaker": f"spk-{index % 3}",
        "source_split": source_split,
        "concept_annotations": [concept],
    }


def world(n_images_per_concept=16, captions_per_image=2, n_negatives=32):
    """A dataset where every image carries several captions, as SpokenCOCO does."""
    groups = []
    for concept in sorted(CONCEPTS):
        for image in range(n_images_per_concept):
            # Half the images come from the source train split, half from val,
            # so the source-split-aware branch has something to work with.
            split = "train2014" if image < n_images_per_concept // 2 else "val2014"
            for caption in range(captions_per_image):
                groups.append(
                    group(f"{concept}-img-{image:03d}", caption, concept, split)
                )
    for image in range(n_negatives):
        for caption in range(captions_per_image):
            entry = group(f"neg-img-{image:03d}", caption, "cat")
            entry["caption"] = f"an empty scene {image:03d} take {caption}"
            entry["concept_annotations"] = []
            groups.append(entry)
    return groups


@pytest.fixture(scope="module")
def unique_subset():
    return M.build_subset(
        world(),
        CONCEPTS,
        groups_per_concept=16,
        negatives_per_concept=16,
        seed="robustness-test",
        evidence_config=config_for_concepts(CONCEPTS),
        profile=IMAGE_UNIQUE_PROFILE,
    )


# ------------------------------------------------- one group per photograph


def test_the_pilot_profile_is_unchanged(unique_subset):
    """The completed run's artifacts must stay re-derivable."""
    legacy = M.build_subset(
        world(), CONCEPTS, groups_per_concept=6, evidence_config=config_for_concepts(CONCEPTS)
    )
    rows = [r for split in legacy["splits"].values() for r in split]
    by_image: dict[str, int] = {}
    for row in rows:
        by_image[row["image_id"]] = by_image.get(row["image_id"], 0) + 1

    assert max(by_image.values()) == 2, "the pilot took two captions per image"
    assert PILOT_PROFILE.max_groups_per_image == 2
    # And it does not carry the new provenance, so its rows are byte-identical.
    assert all(
        "excluded_sibling_group_ids" not in (row.get("split_provenance") or {})
        for row in rows
    )


def test_exactly_one_group_per_image_is_selected(unique_subset):
    rows = [r for split in unique_subset["splits"].values() for r in split]
    image_ids = [row["image_id"] for row in rows]

    assert len(image_ids) == len(set(image_ids)), "an image entered twice"
    assert unique_subset["provenance"]["profile"]["max_groups_per_image"] == 1
    assert unique_subset["provenance"]["independent_unit"] == "image_id"


def test_the_excluded_sibling_groups_are_recorded_not_dropped(unique_subset):
    rows = [r for split in unique_subset["splits"].values() for r in split]
    for row in rows:
        provenance = row["split_provenance"]
        assert provenance["excluded_sibling_group_ids"], row["image_id"]
        assert provenance["n_sibling_groups_excluded"] == 1
        assert row["group_id"] not in provenance["excluded_sibling_group_ids"]
        assert "stable_rank" in provenance["representative_selection"]
        assert provenance["representative_selection_reason"]


def test_representative_selection_is_stable_under_manifest_permutation():
    """A re-derived subset must be the same subset.

    Ranking on the content-derived group id rather than on position is what
    makes this true — the manifest's ordering is an accident of how the file
    was written and must not decide which caption represents a photograph.
    """
    baseline = world()
    reference = M.build_subset(
        baseline, CONCEPTS, groups_per_concept=16, negatives_per_concept=16,
        seed="robustness-test", evidence_config=config_for_concepts(CONCEPTS),
        profile=IMAGE_UNIQUE_PROFILE,
    )
    expected = {
        row["image_id"]: row["group_id"]
        for split in reference["splits"].values()
        for row in split
    }
    for seed in (1, 2, 3):
        shuffled = list(baseline)
        random.Random(seed).shuffle(shuffled)
        other = M.build_subset(
            shuffled, CONCEPTS, groups_per_concept=16, negatives_per_concept=16,
            seed="robustness-test", evidence_config=config_for_concepts(CONCEPTS),
            profile=IMAGE_UNIQUE_PROFILE,
        )
        assert {
            row["image_id"]: row["group_id"]
            for split in other["splits"].values()
            for row in split
        } == expected, f"permutation {seed} changed the representative groups"


def test_choose_representative_groups_is_deterministic_and_total():
    candidates = [group("img-1", index) for index in range(5)]
    chosen, excluded, reason = choose_representative_groups(
        candidates, image_id="img-1", seed="s", profile=IMAGE_UNIQUE_PROFILE
    )
    assert len(chosen) == 1
    assert len(excluded) == 4
    assert set(excluded) | {chosen[0]["group_id"]} == {c["group_id"] for c in candidates}
    assert reason
    again, _, _ = choose_representative_groups(
        list(reversed(candidates)), image_id="img-1", seed="s", profile=IMAGE_UNIQUE_PROFILE
    )
    assert again[0]["group_id"] == chosen[0]["group_id"]


# ------------------------------------------------------------ the split


def test_no_train_test_image_overlap(unique_subset):
    leakage = M.check_split_leakage(unique_subset)
    assert leakage["ok"]
    assert leakage["image_overlap"] == []
    assert leakage["group_overlap"] == []


def test_every_sibling_group_of_an_image_stays_on_one_side(unique_subset):
    """Only one sibling is selected, but the excluded ones must not be free to
    reappear on the other side of the split under another concept."""
    train_images = {r["image_id"] for r in unique_subset["splits"]["train"]}
    test_images = {r["image_id"] for r in unique_subset["splits"]["test"]}
    assert not (train_images & test_images)
    excluded_by_split = {
        split: {
            sibling
            for row in rows
            for sibling in row["split_provenance"]["excluded_sibling_group_ids"]
        }
        for split, rows in unique_subset["splits"].items()
    }
    assert not (excluded_by_split["train"] & excluded_by_split["test"])


def test_each_concept_gets_the_stated_distinct_image_counts(unique_subset):
    for concept in CONCEPTS:
        train = {
            r["image_id"] for r in unique_subset["splits"]["train"] if r["concept"] == concept
        }
        test = {
            r["image_id"] for r in unique_subset["splits"]["test"] if r["concept"] == concept
        }
        assert len(train) == 8, concept
        assert len(test) == 8, concept
        assert not (train & test)
    negatives_train = {r["image_id"] for r in unique_subset["splits"]["train"] if not r["concept"]}
    negatives_test = {r["image_id"] for r in unique_subset["splits"]["test"] if not r["concept"]}
    assert len(negatives_train) == 8
    assert len(negatives_test) == 8
    assert not (negatives_train & negatives_test)


def test_a_concept_short_of_the_stated_images_is_refused_not_shrunk():
    with pytest.raises(InsufficientDistinctImagesError, match="needs 8 training"):
        M.build_subset(
            world(n_images_per_concept=6),
            CONCEPTS,
            groups_per_concept=16,
            negatives_per_concept=16,
            evidence_config=config_for_concepts(CONCEPTS),
            profile=IMAGE_UNIQUE_PROFILE,
        )


def test_too_few_matched_negative_images_is_refused():
    with pytest.raises(InsufficientDistinctImagesError, match="matched-negative"):
        M.build_subset(
            world(n_negatives=4),
            CONCEPTS,
            groups_per_concept=16,
            negatives_per_concept=16,
            evidence_config=config_for_concepts(CONCEPTS),
            profile=IMAGE_UNIQUE_PROFILE,
        )


def test_an_image_assigned_to_both_splits_is_refused_under_any_profile():
    """A guard, not a policy. The pilot's source-split prefixes could put one
    photograph on both sides when its captions span both source splits."""
    profile = SubsetProfile(
        name="broken", version="broken.v0", max_groups_per_image=1,
        n_train_positive_images=2, n_test_positive_images=2,
        n_train_negative_images=1, n_test_negative_images=1,
    )
    groups = []
    for index in range(4):
        # Every image has one train-split and one val-split caption.
        groups.append(group(f"cat-img-{index}", 0, "cat", "train2014"))
        groups.append(group(f"cat-img-{index}", 1, "cat", "val2014"))
    for index in range(2):
        entry = group(f"neg-{index}", 0, "cat")
        entry["caption"] = f"an empty scene number {index}"
        entry["concept_annotations"] = []
        groups.append(entry)

    subset = M.build_subset(
        groups, {"cat": ("cat", "cats")}, groups_per_concept=4, negatives_per_concept=1,
        evidence_config=config_for_concepts({"cat": ("cat", "cats")}), profile=profile,
    )
    # The corrected pool logic keeps them disjoint rather than raising.
    assert M.check_split_leakage(subset)["ok"]


# ------------------------------------------------ source-training examples


def code(image_id, concept, *, split="train", modality="text"):
    return {
        "sample_id": f"g_{image_id}:{modality}",
        "group_id": f"g_{image_id}",
        "image_id": image_id,
        "concept": concept,
        "split": split,
        "modality": modality,
    }


def test_distinct_image_selection_takes_one_record_per_image():
    records = [code("i1", "cat"), code("i1", "cat"), code("i2", "cat")]
    records[1]["sample_id"] = "g_i1b:text"
    chosen, images = select_distinct_image_records(
        records, n_required=2, role="test", what="examples"
    )
    assert images == ["i1", "i2"]
    assert len(chosen) == 2


def test_a_short_distinct_image_set_refuses_and_says_what_was_missing():
    records = [code("i1", "cat"), code("i1", "cat")]
    with pytest.raises(InsufficientDistinctImagesError) as error:
        select_distinct_image_records(records, n_required=2, role="cat positives")
    message = str(error.value)
    assert "need 2 distinct image(s)" in message
    assert "1 repeat(s)" in message
    assert "reports an n it never had" in message


def test_positive_and_negative_source_images_are_held_disjoint():
    positives = [code("i1", "cat"), code("i2", "cat")]
    negatives = [code("i2", None), code("i3", None)]
    _, positive_images = select_distinct_image_records(
        positives, n_required=2, role="positives"
    )
    _, negative_images = select_distinct_image_records(
        negatives, n_required=1, role="negatives",
        exclude_images=frozenset(positive_images),
    )
    assert negative_images == ["i3"]
    assert_disjoint_images(
        positive_images, negative_images, left_name="positives", right_name="negatives"
    )


def test_overlapping_image_roles_are_refused():
    with pytest.raises(InsufficientDistinctImagesError, match="share 1 image"):
        assert_disjoint_images(["i1"], ["i1"], left_name="positives", right_name="negatives")


# ---------------------------------------------------------- causal targets


def _target_world():
    groups = {}
    codes = []
    for image, concept in [
        ("p1", "cat"), ("p1", "cat"), ("p2", "cat"), ("p3", "cat"),
        ("n1", None), ("n1", None), ("n2", None), ("n3", None),
    ]:
        suffix = sum(1 for c in codes if c["image_id"] == image)
        group_id = f"g_{image}_{suffix}"
        entry = {
            "group_id": group_id, "image_id": image, "concept": concept,
            "split": "test", "caption": "c", "image_path": "p", "audio_path": "a",
        }
        groups[group_id] = entry
        codes.append(
            {
                "sample_id": f"{group_id}:image", "group_id": group_id,
                "image_id": image, "concept": concept, "split": "test",
                "modality": "image",
            }
        )
    return codes, groups


def test_legacy_target_selection_can_pick_two_captions_of_one_photograph():
    """The behavior the robustness profile exists to replace, pinned so the
    difference is visible rather than asserted."""
    codes, groups = _target_world()
    targets = _select_targets(codes, groups, "cat", "image", 2)
    positive_images = [g["image_id"] for g, sign in targets if sign < 0]

    assert positive_images == ["p1", "p1"], "one photograph counted twice"


def test_image_unique_target_selection_takes_distinct_photographs():
    codes, groups = _target_world()
    targets = _select_targets(
        codes, groups, "cat", "image", 2, image_unique=True, require_exact=True
    )
    positives = [g["image_id"] for g, sign in targets if sign < 0]
    negatives = [g["image_id"] for g, sign in targets if sign > 0]

    assert positives == ["p1", "p2"]
    assert len(set(positives)) == 2
    assert len(set(negatives)) == 2
    assert not (set(positives) & set(negatives))


def test_target_images_are_disjoint_from_the_source_training_images():
    codes, groups = _target_world()
    targets = _select_targets(
        codes, groups, "cat", "image", 2,
        image_unique=True, source_images=frozenset({"p1"}), require_exact=True,
    )
    positives = [g["image_id"] for g, sign in targets if sign < 0]

    assert "p1" not in positives
    assert positives == ["p2", "p3"]


def test_too_few_distinct_target_images_refuses():
    codes, groups = _target_world()
    with pytest.raises(InsufficientDistinctImagesError, match="held-out positives"):
        _select_targets(
            codes, groups, "cat", "image", 8, image_unique=True, require_exact=True
        )


# ------------------------------------------------ concepts and controls


def test_focal_concepts_are_the_first_three_in_ranking_order():
    ranked = ["zebra", "cat", "toilet", "giraffe", "bird", "clock"]
    focal, non_focal = select_focal_concepts(ranked, n_focal=3)

    assert focal == ["zebra", "cat", "toilet"], "ranking order, not alphabetical"
    assert non_focal == ["giraffe", "bird", "clock"]
    assert sorted(focal) != focal, "a sorted result here would be the bug"


def test_ranking_order_is_not_alphabetically_reordered():
    ranked = ["zebra", "cat", "toilet", "giraffe", "bird", "clock"]
    focal, _ = select_focal_concepts(ranked, n_focal=3)
    assert focal != sorted(ranked)[:3]


def test_the_unrelated_control_always_comes_from_outside_the_focal_set():
    focal = ["zebra", "cat", "toilet"]
    non_focal = ["giraffe", "bird", "clock"]
    assignment = unrelated_control_assignment(focal, non_focal)

    assert assignment == {"zebra": "giraffe", "cat": "bird", "toilet": "clock"}
    assert all(control not in focal for control in assignment.values())
    assert len(set(assignment.values())) == 3, "each focal gets its own control"


def test_the_unrelated_control_is_a_pure_function_of_the_ranking():
    """No capability result, activation, or target-test example takes part.

    The rule's only inputs are the two ordered name lists, which is what makes
    it impossible to have chosen the control after seeing how the candidates
    behaved.
    """
    focal, non_focal = ["a", "b"], ["x", "y"]
    first = unrelated_control_assignment(focal, non_focal)
    second = unrelated_control_assignment(list(focal), list(non_focal))
    assert first == second == {"a": "x", "b": "y"}


def test_an_empty_non_focal_set_is_refused():
    with pytest.raises(ValueError, match="direct contrast"):
        unrelated_control_assignment(["a"], [])


def test_too_few_concepts_for_a_focal_set_plus_a_control_is_refused():
    with pytest.raises(ValueError, match="at least one external unrelated control"):
        select_focal_concepts(["a", "b", "c"], n_focal=3)


# ------------------------------------------------------------- the config


def test_the_robustness_config_selects_the_image_unique_policy():
    config = PilotConfig(
        subset_profile="image_unique",
        image_unique_targets=True,
        min_source_positive_images=8,
        min_source_negative_images=8,
        off_diagonal_causal_only=True,
    )
    assert config.profile() is IMAGE_UNIQUE_PROFILE
    assert config.profile().causal_target_selection == CAUSAL_TARGET_SELECTION_VERSION
    assert PilotConfig().profile() is PILOT_PROFILE, "the default stays the pilot's"
