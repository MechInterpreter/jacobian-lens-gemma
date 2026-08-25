# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Frozen evidence-quality gate for the cat->dog animal-sound study.

Why this module exists
=======================

Sixteen photographs recruited for the earlier cat/cow animal-sound test were
inspected by eye after the causal run had already spent them. Four of eight
cow photographs were compromised:

* a cow covering 0.26% of the frame, behind a boat and river flowers
  (``COCO_train2014_000000481142``);
* a **fiberglass promotional statue** of a cow at a yogurt stand, captioned
  by COCO's own annotators as *"a statue of a cow"* / *"a fake cow"*
  (``COCO_val2014_000000386718``);
* a cow with a farmer, a sheep, and what the caption calls *"two dogs"*
  sharing the frame (``COCO_val2014_000000193162``);
* a cow and calf with sheep visible in the background
  (``COCO_val2014_000000467776``).

Every one of those was detectable from data already inside COCO's own
instance annotations and captions. The pipeline that selected them
(:func:`jlens.mmpilot.expansion.attach_concept_annotations`) kept only the
*set of category names* present in an image, discarding instance area, and
checked exactly one of an image's five captions rather than all of them. A
cow-shaped statue and a real cow are the same COCO category; nothing looked at
the words *"statue"* or *"fake"* sitting right there in the caption file.

This module is the repair, and it is deliberately **not** folded into
:mod:`jlens.mmpilot.expansion`: it reads the *raw* COCO ``instances_*.json``
and ``captions_*.json`` files directly (bbox area and every caption, not the
lossy category-name projection), and it is frozen — every threshold and
lexical rule below is fixed **before** any group derived from it is scored
against a model, per the mission's requirement that evidence selection never
be informed by a causal or capability outcome.

The five criteria, applied per candidate image
================================================

1. **Exactly one animal species is present.** Two or more disqualifies the
   image outright — there is no principled way to attribute the model's
   answer to one identity when another is also in frame.
2. **The target is sufficiently prominent.** Its single largest instance must
   cover at least :data:`EvidenceQualityThresholds.min_area_fraction` of the
   frame. A speck in a landscape is not evidence of identity.
3. **The target is named consistently across captions.** At least
   :meth:`EvidenceQualityThresholds.min_caption_matches` of the available COCO
   captions must contain the target word (or its plural), checked against
   *every* caption, not one.
4. **No competing animal word appears in any caption**, and **no depiction
   word** (``statue``, ``fake``, ``toy``, ``stuffed``, ``painting``, ...;
   the full list is :data:`DEPICTION_LEXICON`) appears in any caption. Both
   checks run across the complete caption set.
5. **A person does not dominate the frame** — total person-instance area below
   :data:`EvidenceQualityThresholds.max_person_fraction`.

A sixth check — synchronized image, caption, and audio actually present — is
not decidable from COCO alone (COCO carries no audio) and is applied
separately by :func:`filter_synchronized_groups`, once the clean-evidence
index is intersected with the SpokenCOCO-synchronized manifest.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from jlens.mmpilot.store import payload_checksum

__all__ = [
    "ANIMAL_CATEGORIES",
    "DEPICTION_LEXICON",
    "EVIDENCE_QUALITY_VERSION",
    "POPULATION_FREEZE_VERSION",
    "EvidenceQualityRefused",
    "EvidenceQualityThresholds",
    "build_clean_evidence_index",
    "evaluate_image_evidence_quality",
    "filter_synchronized_groups",
    "freeze_disjoint_populations",
]

EVIDENCE_QUALITY_VERSION = "mmpilot.animal_evidence_quality_gate.v1"
POPULATION_FREEZE_VERSION = "mmpilot.disjoint_population_freeze.v1"

#: The ten COCO categories that name an animal species. A second one present
#: in an image, in any role, disqualifies it under criterion 1.
ANIMAL_CATEGORIES: tuple[str, ...] = (
    "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe",
)

#: Words whose presence in a caption means the "animal" is not a live animal:
#: a statue, a toy, a picture of one, a costume. Frozen before any image is
#: scored. Extending this list after seeing which images it would exclude is
#: exactly the "select the evidence to make it work" failure mode this module
#: exists to prevent — do not add to it once a study has opened outcomes.
#: "picture of" / "photo of" / "image of" are deliberately absent. They are
#: COCO's standard boilerplate for narrating an ordinary real photograph ("A
#: picture of a dog that is looking out the window") and are indistinguishable
#: from a genuine depiction ("a picture of a dog statue") by adjacency alone --
#: both put only "a"/"an"/"the" between the phrase and the animal word. Every
#: genuine depiction they might have caught is already caught by its own
#: specific noun ("statue", "toy", "painting", ...) sitting next to the
#: animal, so dropping them costs no real coverage. Found by inspecting real
#: COCO rejections: with these three included, adjacency still fired on
#: "A picture of a dog" as if the dog were fictional.
DEPICTION_LEXICON: tuple[str, ...] = (
    "statue", "statues", "fake", "plastic", "toy", "toys", "stuffed",
    "stuffed animal", "plush", "painting", "painted", "drawing", "drawn",
    "figurine", "sculpture", "model", "poster", "mural", "cartoon",
    "sign", "billboard", "costume",
    "balloon", "carousel", "puppet", "doll", "replica", "ceramic",
    "cardboard", "inflatable", "mascot", "stone", "bronze", "wooden",
    "cutout", "sticker", "logo", "advertisement", "ad for",
)

#: Filler tokens a depiction phrase and the target word are allowed to have
#: between them and still count as describing the *animal* -- "a statue OF A
#: cat", "a picture OF THE dog". Anything else between them ("cat ON A
#: plastic mat", "dog UNDER A stone arch", "dog NEXT TO sign") means the
#: depiction word is describing a different object in the scene, not the
#: animal, and must not disqualify the photo. This is the fix for a real
#: false-positive rate found by inspecting real COCO rejections: with a
#: bare "does this word appear anywhere in the caption" check, ~150 of a
#: ~2,400-image sample were rejected for exactly this reason -- "A dog is
#: sitting under a stone arch" and "A black cat looking at a statue" both
#: describe real animals near an unrelated object, not a fake animal.
_DEPICTION_LINKING_TOKENS = frozenset({"a", "an", "the", "of"})
_TOKEN_PATTERN = re.compile(r"[A-Za-z']+")
#: How many *non-linking* tokens may sit between an adjacent depiction word
#: and the animal (e.g. "a small stuffed cat"). Kept separate from the
#: linking-token allowance so "cat on a plastic mat" (a non-linking "on")
#: still fails regardless of how many linking tokens surround it.
_DEPICTION_MAX_GAP = 3


def _word_pattern(word: str) -> re.Pattern:
    return re.compile(rf"\b{re.escape(word)}s?\b", re.IGNORECASE)


_ANIMAL_PATTERNS = {name: _word_pattern(name) for name in ANIMAL_CATEGORIES}


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_PATTERN.findall(text)]


def _depiction_words_near_target(
    caption: str, target: str, *, lexicon: Sequence[str] = DEPICTION_LEXICON
) -> set[str]:
    """Depiction phrases in ``caption`` that plausibly describe ``target``.

    A phrase counts only if every token between it and an occurrence of the
    target word is a linking token (``a``/``an``/``the``/``of``) -- catching
    "a statue of a cat" and "a stuffed dog" -- or the gap is small and free of
    prepositions describing a *different* relationship ("small stuffed cat").
    A depiction word describing something else in the scene ("on a plastic
    mat", "under a stone arch", "next to sign") is not counted.
    """
    tokens = _tokenize(caption)
    target_positions = [
        i for i, t in enumerate(tokens) if t in (target, f"{target}s")
    ]
    if not target_positions:
        return set()

    hits: set[str] = set()
    for phrase in lexicon:
        phrase_tokens = tuple(phrase.lower().split())
        n = len(phrase_tokens)
        for start in range(len(tokens) - n + 1):
            if tuple(tokens[start:start + n]) != phrase_tokens:
                continue
            end = start + n - 1
            for target_index in target_positions:
                if target_index < start:
                    gap_tokens = tokens[target_index + 1:start]
                elif target_index > end:
                    gap_tokens = tokens[end + 1:target_index]
                else:
                    continue  # overlapping match, not meaningful
                if not gap_tokens:
                    hits.add(phrase)
                elif len(gap_tokens) <= _DEPICTION_MAX_GAP and all(
                    token in _DEPICTION_LINKING_TOKENS for token in gap_tokens
                ):
                    hits.add(phrase)
    return hits


@dataclass(frozen=True)
class EvidenceQualityThresholds:
    """Every numeric and lexical rule the gate applies. Frozen before use.

    Attributes:
        min_area_fraction: The target's largest single instance must cover at
            least this fraction of the image area.
        min_caption_match_ratio: At least this fraction of the image's
            available captions must literally name the target.
        min_caption_matches_floor: The absolute floor on caption matches,
            applied via ``max(floor, ceil(ratio * n_captions))`` so an image
            with only one or two captions is not passed on a single mention.
        max_person_fraction: Total person-instance area must stay below this
            fraction of the frame.
        depiction_lexicon: Caption substrings that disqualify an image
            outright when any caption contains one, checked as a version so a
            later run that changes the wording is a different fingerprint.
    """

    min_area_fraction: float = 0.05
    min_caption_match_ratio: float = 0.80
    min_caption_matches_floor: int = 1
    max_person_fraction: float = 0.30
    depiction_lexicon: tuple[str, ...] = DEPICTION_LEXICON

    def __post_init__(self) -> None:
        for name in ("min_area_fraction", "min_caption_match_ratio", "max_person_fraction"):
            value = float(getattr(self, name))
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1], got {value}")
        if int(self.min_caption_matches_floor) < 1:
            raise ValueError("min_caption_matches_floor must be >= 1")

    def min_caption_matches(self, n_captions: int) -> int:
        if n_captions <= 0:
            return int(self.min_caption_matches_floor)
        return max(
            int(self.min_caption_matches_floor),
            math.ceil(float(self.min_caption_match_ratio) * int(n_captions)),
        )

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["depiction_lexicon"] = list(self.depiction_lexicon)
        payload["version"] = EVIDENCE_QUALITY_VERSION
        return payload

    @property
    def digest(self) -> str:
        return payload_checksum(self.to_dict())


#: The frozen thresholds this study uses. A notebook imports this name rather
#: than constructing its own, so no cell can quietly loosen a criterion.
DEFAULT_THRESHOLDS = EvidenceQualityThresholds()


class EvidenceQualityRefused(RuntimeError):
    """The evidence-quality question was asked in a way that cannot be answered."""


def evaluate_image_evidence_quality(
    *,
    target: str,
    area_by_category: Mapping[str, Sequence[float]],
    image_area: float,
    captions: Sequence[str],
    thresholds: EvidenceQualityThresholds = DEFAULT_THRESHOLDS,
) -> dict:
    """Score one image against the five frozen criteria.

    Args:
        target: The concept this image was recruited for (``"cat"`` or
            ``"dog"``).
        area_by_category: ``{COCO category name: [instance areas in pixels]}``
            for every category with at least one instance in the image,
            straight from ``instances_*.json`` (not the lossy name-only
            projection).
        image_area: The image's ``width * height`` in pixels.
        captions: Every caption COCO records for this image (up to five).
        thresholds: The frozen rule set.

    Returns:
        ``{"passed": bool, "target": ..., "failed_criteria": [...], ...}``
        with every measured quantity recorded regardless of outcome, so a
        rejected image's reasons are auditable.

    Raises:
        EvidenceQualityRefused: If ``target`` is not one of
            :data:`ANIMAL_CATEGORIES`, or ``image_area`` is not positive.
    """
    if target not in ANIMAL_CATEGORIES:
        raise EvidenceQualityRefused(
            f"target {target!r} is not a declared animal category: "
            f"{ANIMAL_CATEGORIES}"
        )
    if not (image_area > 0):
        raise EvidenceQualityRefused(f"image_area must be positive, got {image_area}")

    present_animals = sorted(
        name for name in ANIMAL_CATEGORIES
        if area_by_category.get(name)
    )
    failed: list[str] = []

    single_species = present_animals == [target]
    if not single_species:
        failed.append("exactly_one_animal_species")

    target_areas = sorted(
        (float(a) for a in area_by_category.get(target, ())), reverse=True
    )
    largest_fraction = (target_areas[0] / image_area) if target_areas else 0.0
    prominent = largest_fraction >= float(thresholds.min_area_fraction)
    if not prominent:
        failed.append("target_sufficiently_prominent")

    texts = [str(c) for c in captions]
    n_captions = len(texts)
    matches = sum(1 for c in texts if _ANIMAL_PATTERNS[target].search(c))
    required_matches = thresholds.min_caption_matches(n_captions)
    consistently_named = n_captions > 0 and matches >= required_matches
    if not consistently_named:
        failed.append("target_consistently_named_in_captions")

    competing = sorted(
        name for name in ANIMAL_CATEGORIES
        if name != target
        and any(_ANIMAL_PATTERNS[name].search(c) for c in texts)
    )
    if competing:
        failed.append("no_competing_animal_word_in_captions")

    depiction_hits = sorted({
        word for c in texts for word in _depiction_words_near_target(c, target)
    })
    if depiction_hits:
        failed.append("no_depiction_word_in_captions")

    person_areas = area_by_category.get("person", ())
    person_fraction = sum(float(a) for a in person_areas) / image_area
    person_ok = person_fraction < float(thresholds.max_person_fraction)
    if not person_ok:
        failed.append("person_not_dominant")

    return {
        "version": EVIDENCE_QUALITY_VERSION,
        "target": target,
        "passed": not failed,
        "failed_criteria": failed,
        "animal_species_present": present_animals,
        "n_target_instances": len(target_areas),
        "largest_target_area_fraction": largest_fraction,
        "n_captions": n_captions,
        "n_captions_naming_target": matches,
        "required_caption_matches": required_matches,
        "competing_animal_words": competing,
        "depiction_words_found": depiction_hits,
        "person_area_fraction": person_fraction,
    }


def _read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_clean_evidence_index(
    instances_paths: Sequence[str | Path],
    captions_paths: Sequence[str | Path],
    *,
    targets: Sequence[str] = ("cat", "dog"),
    thresholds: EvidenceQualityThresholds = DEFAULT_THRESHOLDS,
) -> dict:
    """Score every COCO image containing exactly one of ``targets`` as its
    only animal, from the raw instance and caption files.

    Args:
        instances_paths: One or more ``instances_{split}.json`` files (COCO
            object-detection annotations with bbox area).
        captions_paths: The matching ``captions_{split}.json`` files.
        targets: Concepts to index. Every candidate whose sole animal category
            is one of these is scored; everything else is skipped.
        thresholds: The frozen rule set — see :class:`EvidenceQualityThresholds`.

    Returns:
        A payload with ``"approved"`` mapping each target to a list of
        ``{image_id, split, file_name, evidence}`` records (only the images
        that passed), ``"rejected_counts"`` per target and failed criterion,
        the frozen thresholds, and a checksum over the whole thing.

    This never touches a model, a manifest group, or a caption's associated
    audio; it is pure COCO-annotation arithmetic, runnable identically outside
    Colab, on CPU, before anything is loaded.
    """
    if len(instances_paths) != len(captions_paths):
        raise EvidenceQualityRefused(
            "instances_paths and captions_paths must pair one-to-one by split"
        )
    targets = tuple(str(t) for t in targets)
    unknown = sorted(set(targets) - set(ANIMAL_CATEGORIES))
    if unknown:
        raise EvidenceQualityRefused(f"unknown target concept(s): {unknown}")

    approved: dict[str, list[dict]] = {t: [] for t in targets}
    rejected_counts: dict[str, dict[str, int]] = {t: defaultdict(int) for t in targets}
    n_candidates = 0

    for instances_path, captions_path in zip(instances_paths, captions_paths, strict=True):
        instances = _read_json(instances_path)
        captions_payload = _read_json(captions_path)
        split = Path(instances_path).stem.replace("instances_", "")

        categories = {c["id"]: str(c["name"]).lower() for c in instances["categories"]}
        images = {
            im["id"]: {
                "file_name": im.get("file_name", ""),
                "area": float(im.get("width", 0)) * float(im.get("height", 0)),
            }
            for im in instances.get("images", [])
        }
        by_image: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for ann in instances.get("annotations", []):
            name = categories.get(ann.get("category_id"))
            if name is None:
                continue
            by_image[ann["image_id"]][name].append(float(ann.get("area", 0.0)))

        captions_by_image: dict[int, list[str]] = defaultdict(list)
        for row in captions_payload.get("annotations", []):
            captions_by_image[row["image_id"]].append(str(row.get("caption", "")))

        for image_id, category_areas in by_image.items():
            present = sorted(name for name in ANIMAL_CATEGORIES if category_areas.get(name))
            if len(present) != 1 or present[0] not in targets:
                continue
            target = present[0]
            meta = images.get(image_id)
            if meta is None or meta["area"] <= 0:
                continue
            n_candidates += 1
            result = evaluate_image_evidence_quality(
                target=target,
                area_by_category=category_areas,
                image_area=meta["area"],
                captions=captions_by_image.get(image_id, ()),
                thresholds=thresholds,
            )
            if result["passed"]:
                approved[target].append({
                    "image_id": int(image_id),
                    "split": split,
                    "file_name": meta["file_name"],
                    "evidence": result,
                })
            else:
                for clause in result["failed_criteria"]:
                    rejected_counts[target][clause] += 1

        del instances, captions_payload, by_image, captions_by_image

    for target in approved:
        approved[target].sort(key=lambda row: row["image_id"])

    payload = {
        "version": EVIDENCE_QUALITY_VERSION,
        "targets": list(targets),
        "thresholds": thresholds.to_dict(),
        "thresholds_digest": thresholds.digest,
        "n_candidates_scored": n_candidates,
        "n_approved": {t: len(rows) for t, rows in approved.items()},
        "rejected_counts": {t: dict(sorted(c.items())) for t, c in rejected_counts.items()},
        "approved": approved,
    }
    return {**payload, "index_checksum": payload_checksum(payload)}


def filter_synchronized_groups(
    clean_evidence_index: Mapping,
    manifest_groups: Sequence[Mapping],
    *,
    target: str,
) -> list[dict]:
    """Intersect the clean-evidence approval list with synchronized manifest
    groups that actually carry an image, a caption, and an audio recording.

    A COCO image can pass the visual/caption gate and still have no matching
    SpokenCOCO audio recording — this is the sixth criterion from the module
    docstring, decided here because audio presence is a property of the
    synchronized manifest, not of raw COCO.
    """
    approved_ids = {
        int(row["image_id"]) for row in (clean_evidence_index["approved"].get(target) or ())
    }
    out: list[dict] = []
    for group in manifest_groups:
        raw_id = str(group.get("image_id") or "")
        # COCO ids are conventionally "COCO_{split}2014_{12-digit id}"; the
        # split name itself contains digits ("2014"), so only the *trailing*
        # digit run is the image id -- concatenating every digit in the
        # string (as an earlier version of this function did) silently
        # produces a different, wrong number.
        trailing = re.search(r"(\d+)$", raw_id)
        if trailing is None or int(trailing.group(1)) not in approved_ids:
            continue
        if any(
            not str(group.get(key) or "").strip()
            for key in ("group_id", "image_id", "caption", "image_path", "audio_path")
        ):
            continue
        out.append(dict(group))
    return out


def _stable_rank(value: str, seed: str) -> str:
    import hashlib

    return hashlib.sha256(f"{seed}|{value}".encode()).hexdigest()


def freeze_disjoint_populations(
    clean_groups_by_concept: Mapping[str, Sequence[Mapping]],
    *,
    n_dev_per_concept: int,
    n_confirm_per_concept: int,
    seed: str,
) -> dict:
    """Partition each concept's clean-evidence-approved photographs into
    disjoint development and confirmation pools, before any model runs.

    The ordering is a seeded stable hash, not selection on any measured
    outcome — the same rule :func:`jlens.mmpilot.multimodal_lens.
    select_causal_groups` uses for its own fresh-population ordering. The
    first ``n_dev_per_concept`` ranked photographs become the development
    candidate pool; the next ``n_confirm_per_concept`` become the confirmation
    candidate pool. Disjointness is then verified explicitly and recorded,
    not merely assumed from non-overlapping slices.

    Raises:
        EvidenceQualityRefused: If a concept has too few approved photographs
            for both pools combined.

    Returns:
        A payload with ``"development"`` and ``"confirmation"`` group lists
        per concept, the image-id sets for both, an explicit
        ``"disjoint"`` boolean (and the intersection, which must be empty),
        and a checksum. Call this before model loading; its digest belongs in
        the run fingerprint so any later attempt to change the population
        recruits a new run directory rather than mixing populations.
    """
    if n_dev_per_concept <= 0 or n_confirm_per_concept <= 0:
        raise EvidenceQualityRefused(
            "n_dev_per_concept and n_confirm_per_concept must both be positive"
        )
    development: dict[str, list[dict]] = {}
    confirmation: dict[str, list[dict]] = {}
    for concept, groups in clean_groups_by_concept.items():
        needed = int(n_dev_per_concept) + int(n_confirm_per_concept)
        # One representative group per distinct image (a synchronized
        # manifest can carry more than one group per photograph).
        by_image: dict[str, dict] = {}
        for group in groups:
            image_id = str(group.get("image_id") or "")
            if not image_id:
                continue
            candidate = by_image.get(image_id)
            if candidate is None or _stable_rank(
                str(group["group_id"]), f"{seed}|{concept}|sibling"
            ) < _stable_rank(str(candidate["group_id"]), f"{seed}|{concept}|sibling"):
                by_image[image_id] = dict(group)
        ordered = sorted(
            by_image.values(),
            key=lambda row: _stable_rank(str(row["image_id"]), f"{seed}|{concept}|split"),
        )
        if len(ordered) < needed:
            raise EvidenceQualityRefused(
                f"concept {concept!r} has only {len(ordered)} clean-evidence, "
                f"synchronized photographs; {needed} are required "
                f"({n_dev_per_concept} development + {n_confirm_per_concept} "
                "confirmation)"
            )
        development[concept] = ordered[:n_dev_per_concept]
        confirmation[concept] = ordered[
            n_dev_per_concept : n_dev_per_concept + n_confirm_per_concept
        ]

    dev_ids = {
        str(row["image_id"]) for rows in development.values() for row in rows
    }
    confirm_ids = {
        str(row["image_id"]) for rows in confirmation.values() for row in rows
    }
    overlap = sorted(dev_ids & confirm_ids)
    payload = {
        "version": POPULATION_FREEZE_VERSION,
        "seed": str(seed),
        "n_dev_per_concept": int(n_dev_per_concept),
        "n_confirm_per_concept": int(n_confirm_per_concept),
        "development_image_ids": sorted(dev_ids),
        "confirmation_image_ids": sorted(confirm_ids),
        "n_development": {c: len(rows) for c, rows in development.items()},
        "n_confirmation": {c: len(rows) for c, rows in confirmation.items()},
        "overlap": overlap,
        "disjoint": not overlap,
        "frozen_before_model_load": True,
        "selection_uses_causal_outcomes": False,
    }
    if overlap:
        raise EvidenceQualityRefused(
            f"development and confirmation populations are not disjoint: "
            f"{len(overlap)} shared image id(s) {overlap[:5]}"
        )
    digest = payload_checksum(payload)
    return {
        **payload,
        "freeze_digest": digest,
        "development": development,
        "confirmation": confirmation,
    }
