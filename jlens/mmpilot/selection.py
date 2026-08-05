# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Which images, which groups, and which of them are independent — versioned.

The image-independence audit corrected a completed run's arithmetic after the
fact: repeated caption groups of one photograph were averaged within image
before aggregation. That is the right repair for evidence already collected,
but it is the wrong place to fix the problem. A run that spends model passes on
two captions of one photograph has bought one observation at twice the price,
and no amount of downstream averaging gives back the observation it never made.

So selection happens here, before anything is loaded, and **image_id is the
independent unit from the outset**:

* one deterministically chosen synchronized group per image (the sibling groups
  are recorded as excluded, not quietly dropped);
* source-training positives and negatives on distinct, mutually disjoint
  images;
* causal targets on distinct images, disjoint from each other and from the
  source-training images.

Every rule here is **versioned and profiled**. :data:`PILOT_PROFILE` reproduces
the completed four-concept pilot byte for byte — it has to, because that run's
artifacts are still on disk and must stay resumable and re-derivable.
:data:`IMAGE_UNIQUE_PROFILE` is the new behavior, and it is opt-in. A profile
version is bound into the run fingerprint, so a directory built under one
policy can never be resumed under another.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

#: One synchronized group per image, chosen by a seeded stable rank.
REPRESENTATIVE_SELECTION_VERSION = "stable_rank_of_group_within_image.v1"

#: How the pilot chose groups: the lowest ``group_id`` values, up to the cap.
LEGACY_REPRESENTATIVE_SELECTION_VERSION = "lowest_group_id_up_to_cap.v0"

#: Source-training positives/negatives restricted to distinct images.
SOURCE_POSITIVE_SELECTION_VERSION = "distinct_image_positives.v1"
SOURCE_NEGATIVE_SELECTION_VERSION = "distinct_image_negatives.v1"

#: Causal targets deduplicated on image before examples are chosen.
CAUSAL_TARGET_SELECTION_VERSION = "distinct_image_targets_disjoint_from_source.v1"

#: What the pilot did: dedupe on ``group_id`` only.
LEGACY_CAUSAL_TARGET_SELECTION_VERSION = "distinct_group_targets.v0"

#: How an unrelated-concept control is picked. Uses the pre-model ranking and
#: nothing else — no capability result, no activation, no target-test data.
UNRELATED_CONTROL_SELECTION_VERSION = "rotate_non_focal_in_ranking_order.v1"


class InsufficientDistinctImagesError(RuntimeError):
    """Not enough distinct images to satisfy the profile, so nothing proceeds.

    Raised rather than quietly shrinking a set. A concept that supplies six
    photographs where the design asked for eight has not met the design; a run
    that silently continued would report an ``n`` it never had.
    """


def stable_rank(value: str, salt: str) -> str:
    """Deterministic order key. Independent of manifest ordering and of
    Python's per-process hash salt."""
    return hashlib.sha256(f"{salt}|{value}".encode()).hexdigest()


@dataclass(frozen=True)
class SubsetProfile:
    """How many images a split gets, and how one group per image is chosen.

    Args:
        max_groups_per_image: Groups kept per image. ``1`` makes the
            synchronized group and the photograph the same unit.
        representative_selection: Version of the per-image choice rule.
        n_train_positive_images / n_test_positive_images: Distinct positive
            images per concept per split. ``None`` keeps the pilot's derivation
            from ``groups_per_concept``.
        n_train_negative_images / n_test_negative_images: Distinct matched
            negative images per split, shared across concepts.
        record_sibling_exclusions: Write the excluded sibling group ids and the
            selection reason onto every row.
    """

    name: str
    version: str
    max_groups_per_image: int = 2
    representative_selection: str = LEGACY_REPRESENTATIVE_SELECTION_VERSION
    n_train_positive_images: int | None = None
    n_test_positive_images: int | None = None
    n_train_negative_images: int | None = None
    n_test_negative_images: int | None = None
    record_sibling_exclusions: bool = False
    causal_target_selection: str = LEGACY_CAUSAL_TARGET_SELECTION_VERSION
    source_positive_selection: str = SOURCE_POSITIVE_SELECTION_VERSION
    source_negative_selection: str = SOURCE_NEGATIVE_SELECTION_VERSION

    @property
    def image_unique(self) -> bool:
        return self.max_groups_per_image == 1

    @property
    def explicit_image_counts(self) -> bool:
        return self.n_train_positive_images is not None

    def to_dict(self) -> dict:
        return asdict(self)


#: The completed four-concept pilot. Reproduces its subset exactly; do not
#: change any field, or that run stops being re-derivable.
PILOT_PROFILE = SubsetProfile(
    name="pilot",
    version="pilot.v1",
    max_groups_per_image=2,
    representative_selection=LEGACY_REPRESENTATIVE_SELECTION_VERSION,
    causal_target_selection=LEGACY_CAUSAL_TARGET_SELECTION_VERSION,
)

#: The bounded six-concept robustness study: one group per image, eight
#: distinct images in every cell of the design.
IMAGE_UNIQUE_PROFILE = SubsetProfile(
    name="image_unique",
    version="image_unique.v1",
    max_groups_per_image=1,
    representative_selection=REPRESENTATIVE_SELECTION_VERSION,
    n_train_positive_images=8,
    n_test_positive_images=8,
    n_train_negative_images=8,
    n_test_negative_images=8,
    record_sibling_exclusions=True,
    causal_target_selection=CAUSAL_TARGET_SELECTION_VERSION,
)

#: The same policy at MOCK scale. Two distinct images per split instead of
#: eight, so a synthetic world small enough to run on CPU in seconds still
#: exercises every branch of the image-unique rule: one group per photograph,
#: sibling exclusion, disjoint source and target images, distinct-image floors.
#:
#: **It is a plumbing profile and never a scientific one.** Its name is part of
#: the run fingerprint, so a run built under it can never be resumed or
#: confused as one built under :data:`IMAGE_UNIQUE_PROFILE`, and every artifact
#: it produces carries ``mode="mock"``.
IMAGE_UNIQUE_MOCK_PROFILE = SubsetProfile(
    name="image_unique_mock",
    version="image_unique_mock.v1",
    max_groups_per_image=1,
    representative_selection=REPRESENTATIVE_SELECTION_VERSION,
    n_train_positive_images=2,
    n_test_positive_images=2,
    n_train_negative_images=2,
    n_test_negative_images=2,
    record_sibling_exclusions=True,
    causal_target_selection=CAUSAL_TARGET_SELECTION_VERSION,
)

PROFILES = {
    profile.name: profile
    for profile in (PILOT_PROFILE, IMAGE_UNIQUE_PROFILE, IMAGE_UNIQUE_MOCK_PROFILE)
}


# --------------------------------------------------- one group per photograph


def choose_representative_groups(
    candidates: Sequence[Mapping],
    *,
    image_id: str,
    seed: str,
    profile: SubsetProfile,
) -> tuple[list[dict], list[str], str]:
    """``(chosen, excluded_group_ids, reason)`` for one image's groups.

    Under :data:`PILOT_PROFILE` this is the pilot's rule unchanged: the lowest
    ``group_id`` values up to the cap.

    Under an image-unique profile one group is chosen by a seeded stable rank
    over ``group_id`` salted with the image and the split seed. Ranking on the
    id rather than on position makes the choice independent of the order the
    manifest happened to list the captions in, which is the property that
    matters — a re-derived subset must be the same subset.
    """
    ordered = sorted(candidates, key=lambda group: str(group["group_id"]))
    if profile.representative_selection == LEGACY_REPRESENTATIVE_SELECTION_VERSION:
        chosen = ordered[: profile.max_groups_per_image]
        return (
            [dict(group) for group in chosen],
            [str(g["group_id"]) for g in ordered[profile.max_groups_per_image :]],
            f"lowest {profile.max_groups_per_image} group_id(s) for this image",
        )
    ranked = sorted(
        ordered,
        key=lambda group: (
            stable_rank(
                str(group["group_id"]), f"{seed}|representative|{image_id}"
            ),
            str(group["group_id"]),
        ),
    )
    chosen = ranked[: profile.max_groups_per_image]
    return (
        [dict(group) for group in chosen],
        [str(g["group_id"]) for g in ranked[profile.max_groups_per_image :]],
        (
            f"{profile.representative_selection}: lowest "
            f"sha256({seed}|representative|{image_id}|group_id) among this "
            f"image's valid synchronized groups"
        ),
    )


# ------------------------------------------------- distinct-image selections


def group_by_image(records: Sequence[Mapping]) -> dict[str, list[Mapping]]:
    """``{image_id: [record, ...]}`` with each bucket in ``sample_id`` order."""
    out: dict[str, list[Mapping]] = {}
    for record in records:
        out.setdefault(str(record["image_id"]), []).append(record)
    for bucket in out.values():
        bucket.sort(key=lambda record: str(record.get("sample_id") or record["group_id"]))
    return out


def select_distinct_image_records(
    records: Sequence[Mapping],
    *,
    n_required: int | None,
    role: str,
    exclude_images: frozenset[str] = frozenset(),
    what: str = "examples",
) -> tuple[list[Mapping], list[str]]:
    """One record per image, in a deterministic order. ``(records, image_ids)``.

    Args:
        n_required: Refuse unless at least this many distinct images survive.
            ``None`` takes everything available without a floor.
        exclude_images: Images already spent elsewhere in the design — source
            training images when selecting targets, positives when selecting
            negatives.

    Raises:
        InsufficientDistinctImagesError: If fewer than ``n_required`` distinct
            images are available. The message says what was asked for, what was
            found, and how many repeats were discarded.
    """
    by_image = group_by_image(
        [record for record in records if str(record["image_id"]) not in exclude_images]
    )
    chosen = [bucket[0] for _, bucket in sorted(by_image.items())]
    n_discarded = sum(len(bucket) - 1 for bucket in by_image.values())
    if n_required is not None and len(chosen) < n_required:
        raise InsufficientDistinctImagesError(
            f"{role}: {what} need {n_required} distinct image(s) but only "
            f"{len(chosen)} are available "
            f"({len(records)} record(s), {n_discarded} repeat(s) of an image "
            f"discarded, {len(exclude_images)} image(s) excluded as already "
            "spent). Refusing to proceed on a smaller set than the design "
            "states — a silently shrunk cell reports an n it never had."
        )
    if n_required is not None:
        chosen = chosen[:n_required]
    return chosen, [str(record["image_id"]) for record in chosen]


def assert_disjoint_images(
    left: Sequence[str], right: Sequence[str], *, left_name: str, right_name: str
) -> None:
    """Refuse when two roles in the design share a photograph."""
    overlap = sorted(set(left) & set(right))
    if overlap:
        raise InsufficientDistinctImagesError(
            f"{left_name} and {right_name} share {len(overlap)} image(s): "
            f"{overlap[:8]}. The same photograph cannot play two roles in one "
            "cell without making the comparison partly about itself."
        )


# ------------------------------------------------ the unrelated-concept rule


def unrelated_control_assignment(
    focal_concepts: Sequence[str], non_focal_concepts: Sequence[str]
) -> dict[str, str]:
    """``{focal: unrelated control}``, decided before any model runs.

    Both sequences arrive in **pre-model ranking order** and are consumed in
    it. The i-th focal concept takes the ``i % len(non_focal)``-th non-focal
    concept, so each focal gets a different control while the assignment stays
    a pure function of the ranking.

    Nothing here reads a capability result, an activation, a direction, or a
    target-test example. That is the point: a control chosen after seeing how
    the candidates behave is not a control. Note also that the control is drawn
    from *outside* the focal set — in a forced choice among the focal concepts
    the only alternative is the target's direct contrast, which is not
    unrelated to it at all.
    """
    if not non_focal_concepts:
        raise ValueError(
            "an external unrelated control needs at least one non-focal "
            "concept; with none, the only available 'control' is the target's "
            "own direct contrast"
        )
    return {
        concept: non_focal_concepts[index % len(non_focal_concepts)]
        for index, concept in enumerate(focal_concepts)
    }


def select_focal_concepts(
    ranked_concepts: Sequence[str], *, n_focal: int
) -> tuple[list[str], list[str]]:
    """``(focal, non_focal)`` — the first ``n_focal`` in ranking order.

    Ranking order is preserved, never sorted alphabetically: the ranking is the
    deterministic pre-model statement of which concepts the dataset supports
    best, and re-sorting it would substitute a different, arbitrary choice.
    """
    if len(ranked_concepts) < n_focal + 1:
        raise ValueError(
            f"{len(ranked_concepts)} concept(s) cannot supply {n_focal} focal "
            "concepts and at least one external unrelated control"
        )
    return list(ranked_concepts[:n_focal]), list(ranked_concepts[n_focal:])


__all__ = [
    "CAUSAL_TARGET_SELECTION_VERSION",
    "IMAGE_UNIQUE_MOCK_PROFILE",
    "IMAGE_UNIQUE_PROFILE",
    "LEGACY_CAUSAL_TARGET_SELECTION_VERSION",
    "LEGACY_REPRESENTATIVE_SELECTION_VERSION",
    "PILOT_PROFILE",
    "PROFILES",
    "REPRESENTATIVE_SELECTION_VERSION",
    "SOURCE_NEGATIVE_SELECTION_VERSION",
    "SOURCE_POSITIVE_SELECTION_VERSION",
    "UNRELATED_CONTROL_SELECTION_VERSION",
    "InsufficientDistinctImagesError",
    "SubsetProfile",
    "assert_disjoint_images",
    "choose_representative_groups",
    "group_by_image",
    "select_distinct_image_records",
    "select_focal_concepts",
    "stable_rank",
    "unrelated_control_assignment",
]
