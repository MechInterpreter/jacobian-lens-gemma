# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Reading the *existing* SpokenCOCO manifest, without assuming its schema.

The dataset already sits in Drive and is never re-downloaded, never mutated,
and never trusted to have the field names this code would have picked. So the
flow is: inspect the actual JSON, propose a field mapping with the evidence
that supports it, refuse when the mapping is ambiguous, then write a
*derived* normalized manifest into the run directory alongside the original's
checksum.

A **synchronized group** is one COCO image together with one written caption
and the spoken recording of that same caption. It is the unit everything
downstream splits on: all captions and recordings of one image stay in the
same split, so nothing about a test image can leak through a training caption.

Two manifest shapes are supported, which covers the SpokenCOCO releases in the
wild: flat records (one row per caption, with image/audio/caption fields) and
one level of nesting (one row per image, with a list of caption entries).
Anything else fails loudly with the observed keys printed.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
AUDIO_SUFFIXES = (".wav", ".flac", ".mp3", ".ogg", ".m4a")

_ROLE_NAME_HINTS: dict[str, tuple[str, ...]] = {
    "image": ("image", "img", "photo", "picture", "jpg", "jpeg", "frame"),
    "audio": ("audio", "wav", "flac", "speech", "spoken", "utterance_path", "recording"),
    "caption": ("caption", "text", "transcript", "sentence", "utterance", "label"),
    "image_id": ("image_id", "img_id", "cocoid", "coco_id", "imgid", "image_key"),
    "speaker": ("speaker", "spk", "voice", "talker"),
    "split": ("split", "subset", "partition", "fold"),
}


class ManifestSchemaError(RuntimeError):
    """The manifest's structure could not be resolved reliably."""


class SynchronizationError(RuntimeError):
    """Image / caption / audio could not be synchronized well enough to continue."""


# ------------------------------------------------------------------ schema


@dataclass(frozen=True)
class ManifestSchema:
    """Which manifest field plays which role, and why that was chosen.

    Attributes:
        records_key: Top-level dict key holding the record list, or ``None``
            when the manifest is itself a list.
        nested_key: Record field holding per-caption sub-entries, or ``None``
            for flat records.
        image_field / audio_field / caption_field: Required roles.
        image_id_field / speaker_field / split_field: Optional roles
            (``None`` when the manifest does not carry them).
        evidence: Per-role notes on how the field was picked — printed by the
            notebook so the mapping is auditable rather than magic.
    """

    records_key: str | None
    nested_key: str | None
    image_field: str
    audio_field: str
    caption_field: str
    image_id_field: str | None = None
    speaker_field: str | None = None
    split_field: str | None = None
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _find_records(payload) -> tuple[str | None, list[dict]]:
    if isinstance(payload, list):
        records = [r for r in payload if isinstance(r, dict)]
        if not records:
            raise ManifestSchemaError("manifest is a list but holds no JSON objects")
        return None, records
    if not isinstance(payload, Mapping):
        raise ManifestSchemaError(f"manifest root is {type(payload).__name__}, not list/object")
    candidates = [
        (key, value)
        for key, value in payload.items()
        if isinstance(value, list) and value and isinstance(value[0], Mapping)
    ]
    if not candidates:
        raise ManifestSchemaError(
            f"no list-of-objects field at the manifest root; keys are {sorted(payload)}"
        )
    # The record list is the longest one; ties are ambiguous and refused.
    candidates.sort(key=lambda item: (-len(item[1]), item[0]))
    if len(candidates) > 1 and len(candidates[0][1]) == len(candidates[1][1]):
        raise ManifestSchemaError(
            "several equally long list fields at the manifest root "
            f"({[key for key, _ in candidates]}); pass records_key explicitly"
        )
    key, value = candidates[0]
    return key, [r for r in value if isinstance(r, Mapping)]


def _string_fields(records: Sequence[Mapping], limit: int) -> dict[str, list[str]]:
    """Field -> sample of its string values across the first ``limit`` records."""
    values: dict[str, list[str]] = {}
    for record in records[:limit]:
        for key, value in record.items():
            if isinstance(value, (str, int)) and not isinstance(value, bool):
                values.setdefault(key, []).append(str(value))
    return values


def _name_score(field_name: str, role: str) -> int:
    lowered = field_name.lower()
    for rank, hint in enumerate(_ROLE_NAME_HINTS[role]):
        if hint in lowered:
            return len(_ROLE_NAME_HINTS[role]) - rank
    return 0


def _looks_like_path(values: Sequence[str], suffixes: Sequence[str]) -> bool:
    hits = sum(1 for value in values if value.lower().endswith(tuple(suffixes)))
    return bool(values) and hits >= max(1, int(0.8 * len(values)))


def _looks_like_sentence(values: Sequence[str]) -> bool:
    def sentence(value: str) -> bool:
        return (
            " " in value.strip()
            and len(value.strip()) >= 8
            and not value.lower().endswith(IMAGE_SUFFIXES + AUDIO_SUFFIXES)
            and "/" not in value
            and "\\" not in value
        )

    hits = sum(1 for value in values if sentence(value))
    return bool(values) and hits >= max(1, int(0.8 * len(values)))


def _pick(
    role: str,
    candidates: Sequence[tuple[str, int]],
    *,
    required: bool,
    evidence: dict,
) -> str | None:
    """Choose the single best-scoring field for ``role``; refuse on a tie."""
    ranked = sorted(candidates, key=lambda item: (-item[1], item[0]))
    ranked = [item for item in ranked if item[1] > 0]
    if not ranked:
        if required:
            raise ManifestSchemaError(
                f"no manifest field looks like the {role!r} field; "
                f"pass an explicit override for it"
            )
        evidence[role] = "no candidate field"
        return None
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        raise ManifestSchemaError(
            f"ambiguous {role!r} field: {[name for name, _ in ranked[:4]]} score "
            f"equally; pass an explicit override"
        )
    evidence[role] = f"{ranked[0][0]!r} (score {ranked[0][1]}; runners-up {ranked[1:3]})"
    return ranked[0][0]


def inspect_manifest(
    payload,
    *,
    overrides: Mapping[str, str] | None = None,
    sample_records: int = 200,
) -> ManifestSchema:
    """Resolve the manifest's field mapping by inspection.

    Args:
        payload: The parsed manifest JSON (never modified).
        overrides: Explicit ``role -> field name`` mapping that wins over
            inspection. Roles are the :class:`ManifestSchema` field names
            without the ``_field`` suffix, plus ``records_key`` / ``nested_key``.
        sample_records: How many records to look at when scoring fields.

    Raises:
        ManifestSchemaError: If the structure is unrecognized or a required
            role is missing or ambiguous.
    """
    overrides = dict(overrides or {})
    records_key, records = _find_records(payload)
    if "records_key" in overrides:
        records_key = overrides["records_key"]
        records = [r for r in payload[records_key] if isinstance(r, Mapping)]
    evidence: dict = {
        "records_key": records_key,
        "n_records_seen": len(records),
        "record_keys": sorted(records[0]) if records else [],
    }

    # Nested caption entries: a record field that is a list of objects.
    nested_key = overrides.get("nested_key")
    if nested_key is None:
        nested_candidates = sorted(
            key
            for key, value in records[0].items()
            if isinstance(value, list) and value and isinstance(value[0], Mapping)
        )
        if len(nested_candidates) > 1:
            raise ManifestSchemaError(
                f"several nested list fields {nested_candidates}; pass nested_key"
            )
        nested_key = nested_candidates[0] if nested_candidates else None
    evidence["nested_key"] = nested_key
    if nested_key:
        evidence["nested_keys"] = sorted(records[0][nested_key][0])

    outer = _string_fields(records, sample_records)
    inner: dict[str, list[str]] = {}
    if nested_key:
        flattened = [
            entry
            for record in records[:sample_records]
            for entry in record.get(nested_key, [])
            if isinstance(entry, Mapping)
        ]
        inner = _string_fields(flattened, sample_records * 5)

    def score(role: str, predicate) -> list[tuple[str, int]]:
        out = []
        for name, values in {**outer, **inner}.items():
            points = _name_score(name, role)
            if predicate(values):
                points += 4
            elif points and role in ("image", "audio", "caption"):
                # A name hint alone is not enough for the value-typed roles.
                points = 0
            if points:
                out.append((name, points))
        return out

    schema = ManifestSchema(
        records_key=records_key,
        nested_key=nested_key,
        image_field=overrides.get("image")
        or _pick(
            "image",
            score("image", lambda v: _looks_like_path(v, IMAGE_SUFFIXES)),
            required=True,
            evidence=evidence,
        ),
        audio_field=overrides.get("audio")
        or _pick(
            "audio",
            score("audio", lambda v: _looks_like_path(v, AUDIO_SUFFIXES)),
            required=True,
            evidence=evidence,
        ),
        caption_field=overrides.get("caption")
        or _pick(
            "caption",
            score("caption", _looks_like_sentence),
            required=True,
            evidence=evidence,
        ),
        image_id_field=overrides.get("image_id")
        or _pick(
            "image_id",
            [(n, _name_score(n, "image_id")) for n in {**outer, **inner}],
            required=False,
            evidence=evidence,
        ),
        speaker_field=overrides.get("speaker")
        or _pick(
            "speaker",
            [(n, _name_score(n, "speaker")) for n in {**outer, **inner}],
            required=False,
            evidence=evidence,
        ),
        split_field=overrides.get("split")
        or _pick(
            "split",
            [(n, _name_score(n, "split")) for n in {**outer, **inner}],
            required=False,
            evidence=evidence,
        ),
        evidence=evidence,
    )
    return schema


# ------------------------------------------------------------- normalizing


@dataclass
class NormalizedManifest:
    """Synchronized groups plus the audit that justifies trusting them."""

    groups: list[dict]
    audit: dict
    schema: ManifestSchema
    source_checksum: str
    conversion: dict

    def to_dict(self) -> dict:
        return {
            "schema_version": "jlens.mmpilot.manifest.v1",
            "source_checksum": self.source_checksum,
            "detected_schema": self.schema.to_dict(),
            "conversion": self.conversion,
            "audit": self.audit,
            "groups": self.groups,
        }


def _resolve_media(relative: str, roots: Sequence[Path]) -> str | None:
    candidate = Path(relative)
    if candidate.is_absolute():
        return str(candidate) if candidate.is_file() else None
    for root in roots:
        for probe in (root / candidate, root / candidate.name):
            if probe.is_file():
                return str(probe)
    return None


def _flatten(records: Sequence[Mapping], schema: ManifestSchema) -> list[dict]:
    """One dict per (image, caption, recording) triple, merging nested entries."""
    rows: list[dict] = []
    for record in records:
        if schema.nested_key:
            for entry in record.get(schema.nested_key, []):
                if isinstance(entry, Mapping):
                    merged = {k: v for k, v in record.items() if k != schema.nested_key}
                    merged.update(entry)
                    rows.append(merged)
        else:
            rows.append(dict(record))
    return rows


def normalize_manifest(
    payload,
    schema: ManifestSchema,
    *,
    media_roots: Sequence[str | os.PathLike[str]],
    source_checksum: str,
    max_missing_examples: int = 20,
    min_complete_groups: int = 24,
    max_missing_fraction: float = 0.5,
) -> NormalizedManifest:
    """Build synchronized groups with resolved absolute media paths.

    Args:
        payload: Parsed manifest (read-only).
        schema: Mapping from :func:`inspect_manifest`.
        media_roots: Directories to resolve relative media paths against, in
            priority order (dataset root first, download cache last).
        source_checksum: ``sha256:`` of the original manifest file.
        max_missing_examples: How many missing-file examples to keep in the audit.
        min_complete_groups: Refuse below this many fully synchronized groups.
        max_missing_fraction: Refuse when more than this fraction of rows lose
            a media file — that means the roots are wrong, not that the data
            is slightly incomplete.

    Raises:
        SynchronizationError: When the audit shows synchronization cannot be
            trusted. Nothing downstream runs on a guessed correspondence.
    """
    roots = [Path(root) for root in media_roots]
    records_key, records = _find_records(payload) if schema.records_key is None else (
        schema.records_key,
        [r for r in payload[schema.records_key] if isinstance(r, Mapping)],
    )
    del records_key
    rows = _flatten(records, schema)

    groups: list[dict] = []
    missing_image: list[str] = []
    missing_audio: list[str] = []
    seen: dict[str, int] = {}
    duplicates: list[str] = []
    for index, row in enumerate(rows):
        raw_image = str(row.get(schema.image_field, "") or "")
        raw_audio = str(row.get(schema.audio_field, "") or "")
        caption = str(row.get(schema.caption_field, "") or "").strip()
        image_path = _resolve_media(raw_image, roots) if raw_image else None
        audio_path = _resolve_media(raw_audio, roots) if raw_audio else None
        if image_path is None and raw_image:
            missing_image.append(raw_image)
        if audio_path is None and raw_audio:
            missing_audio.append(raw_audio)
        if not (image_path and audio_path and caption):
            continue
        image_id = str(
            row.get(schema.image_id_field) if schema.image_id_field else ""
        ) or Path(raw_image).stem
        group_id = "g_" + hashlib.sha256(
            f"{image_id}|{caption}|{raw_audio}".encode()
        ).hexdigest()[:16]
        if group_id in seen:
            duplicates.append(group_id)
            continue
        seen[group_id] = index
        groups.append(
            {
                "group_id": group_id,
                "image_id": image_id,
                "caption": caption,
                "image_path": image_path,
                "audio_path": audio_path,
                "speaker": (
                    str(row.get(schema.speaker_field))
                    if schema.speaker_field and row.get(schema.speaker_field) is not None
                    else None
                ),
                "source_split": (
                    str(row.get(schema.split_field))
                    if schema.split_field and row.get(schema.split_field) is not None
                    else None
                ),
                "manifest_row": index,
            }
        )

    groups.sort(key=lambda g: g["group_id"])
    speakers = sorted({g["speaker"] for g in groups if g["speaker"]})
    audit = {
        "n_manifest_records": len(records),
        "n_rows_after_flattening": len(rows),
        "n_valid_image_files": len({g["image_path"] for g in groups}),
        "n_valid_audio_files": len({g["audio_path"] for g in groups}),
        "n_synchronized_groups": len(groups),
        "n_distinct_images": len({g["image_id"] for g in groups}),
        "n_missing_image_files": len(missing_image),
        "n_missing_audio_files": len(missing_audio),
        "missing_image_examples": missing_image[:max_missing_examples],
        "missing_audio_examples": missing_audio[:max_missing_examples],
        "n_duplicate_groups": len(duplicates),
        "duplicate_group_examples": duplicates[:max_missing_examples],
        "available_splits": sorted(
            {g["source_split"] for g in groups if g["source_split"]}
        ),
        "n_speakers": len(speakers),
        "speaker_examples": speakers[:max_missing_examples],
        "speaker_metadata_available": bool(speakers),
        "media_roots": [str(root) for root in roots],
    }

    missing_total = len(missing_image) + len(missing_audio)
    denominator = max(1, 2 * len(rows))
    # Root-level failures are diagnosed first: "the paths are wrong" is a much
    # more useful message than "too few groups", and it is the usual cause.
    if missing_total / denominator > max_missing_fraction:
        raise SynchronizationError(
            f"{missing_total} of {denominator} media references did not resolve "
            f"under {[str(r) for r in roots]} — the dataset root is probably "
            f"wrong. Audit: {json.dumps(audit, indent=2)}"
        )
    if len(groups) < min_complete_groups:
        raise SynchronizationError(
            f"only {len(groups)} fully synchronized image/caption/audio groups "
            f"(need >= {min_complete_groups}). Audit: {json.dumps(audit, indent=2)}"
        )
    return NormalizedManifest(
        groups=groups,
        audit=audit,
        schema=schema,
        source_checksum=source_checksum,
        conversion={
            "converter": "jlens.mmpilot.manifest.normalize_manifest",
            "reads_only": True,
            "original_manifest_mutated": False,
            "group_id_rule": "sha256(image_id|caption|audio_relpath)[:16]",
        },
    )


def manifest_checksum(path: str | os.PathLike[str]) -> str:
    """``sha256:`` of the original manifest file (streamed, read-only)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


# ------------------------------------------------------- concepts & splits


#: Frequent, concrete, visually grounded COCO concepts and the caption words
#: that count as a mention. Candidates only — which ones survive is decided by
#: the manifest audit and then by the behavioral capability gate.
DEFAULT_CONCEPTS: dict[str, tuple[str, ...]] = {
    "dog": ("dog", "dogs", "puppy", "puppies"),
    "cat": ("cat", "cats", "kitten", "kittens"),
    "pizza": ("pizza", "pizzas"),
    "bus": ("bus", "buses", "busses"),
    "train": ("train", "trains"),
    "horse": ("horse", "horses"),
}


def caption_mentions(caption: str, keywords: Iterable[str]) -> bool:
    """Whole-word match of any keyword in ``caption`` (case-insensitive)."""
    lowered = caption.lower()
    return any(
        re.search(rf"\b{re.escape(word.lower())}\b", lowered) for word in keywords
    )


def concept_coverage(
    groups: Sequence[Mapping], concepts: Mapping[str, Sequence[str]]
) -> dict[str, dict]:
    """Per-concept counts of groups, distinct images, and speakers.

    The notebook prints this before choosing concepts, so "is this concept
    represented?" is answered from the manifest rather than assumed.
    """
    out: dict[str, dict] = {}
    for concept, keywords in concepts.items():
        hits = [g for g in groups if caption_mentions(g["caption"], keywords)]
        out[concept] = {
            "n_groups": len(hits),
            "n_images": len({g["image_id"] for g in hits}),
            "n_speakers": len({g["speaker"] for g in hits if g["speaker"]}),
        }
    return out


def _stable_rank(value: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}|{value}".encode()).hexdigest()


def build_subset(
    groups: Sequence[Mapping],
    concepts: Mapping[str, Sequence[str]],
    *,
    groups_per_concept: int = 6,
    negatives_per_concept: int = 6,
    max_groups_per_image: int = 2,
    seed: str = "spokencoco-pilot",
) -> dict:
    """Deterministically select a small, image-disjoint pilot subset.

    Each image is assigned to at most one concept: an image whose caption
    mentions two selected concepts is dropped rather than counted twice. Every
    group belonging to one image lands in the same split, so the split is
    image-disjoint, group-disjoint, and sample-disjoint — but *not*
    concept-disjoint, which is the point: the same concept must appear in
    source-train and in distinct target-test examples.

    ``max_groups_per_image`` bounds how many of an image's captions are taken.
    SpokenCOCO ships roughly five captions per image, and taking all of them
    would multiply every downstream forward pass without adding independent
    images — which is what the statistics actually need.

    Returns a dict with ``concepts``, ``splits`` (``train`` / ``test`` group
    lists), and ``provenance``.
    """
    if groups_per_concept < 2:
        raise ValueError("groups_per_concept must be >= 2 to allow a train/test split")

    by_image: dict[str, list[dict]] = {}
    for group in groups:
        by_image.setdefault(group["image_id"], []).append(dict(group))

    image_concepts: dict[str, set[str]] = {}
    for image_id, image_groups in by_image.items():
        matched = {
            concept
            for concept, keywords in concepts.items()
            if any(caption_mentions(g["caption"], keywords) for g in image_groups)
        }
        image_concepts[image_id] = matched

    selected: dict[str, dict] = {}
    used_images: set[str] = set()
    for concept in sorted(concepts):
        pure = sorted(
            (image_id for image_id, matched in image_concepts.items()
             if matched == {concept} and image_id not in used_images),
            key=lambda image_id: _stable_rank(image_id, f"{seed}|{concept}"),
        )
        chosen = pure[:groups_per_concept]
        used_images.update(chosen)
        n_train = max(1, len(chosen) // 2)
        selected[concept] = {
            "train_images": chosen[:n_train],
            "test_images": chosen[n_train:],
            "n_available_images": len(pure),
        }

    negatives = sorted(
        (image_id for image_id, matched in image_concepts.items()
         if not matched and image_id not in used_images),
        key=lambda image_id: _stable_rank(image_id, f"{seed}|negative"),
    )[: negatives_per_concept * 2]
    n_train_neg = max(1, len(negatives) // 2)
    negative_split = {
        "train_images": negatives[:n_train_neg],
        "test_images": negatives[n_train_neg:],
        "n_available_images": len(negatives),
    }

    def rows(image_ids: Sequence[str], concept: str | None, split: str) -> list[dict]:
        out = []
        for image_id in image_ids:
            chosen = sorted(by_image[image_id], key=lambda g: g["group_id"])[
                :max_groups_per_image
            ]
            for group in chosen:
                out.append({**group, "concept": concept, "split": split,
                            "is_positive": concept is not None})
        return out

    train: list[dict] = []
    test: list[dict] = []
    for concept, entry in selected.items():
        train += rows(entry["train_images"], concept, "train")
        test += rows(entry["test_images"], concept, "test")
    train += rows(negative_split["train_images"], None, "train")
    test += rows(negative_split["test_images"], None, "test")

    return {
        "concepts": selected,
        "negatives": negative_split,
        "splits": {"train": train, "test": test},
        "provenance": {
            "seed": seed,
            "groups_per_concept": groups_per_concept,
            "negatives_per_concept": negatives_per_concept,
            "max_groups_per_image": max_groups_per_image,
            "selection_rule": (
                "images whose captions mention exactly one selected concept; "
                "ordered by sha256(seed|concept|image_id); first half train"
            ),
            "concept_keywords": {k: list(v) for k, v in concepts.items()},
        },
    }


def check_split_leakage(subset: Mapping) -> dict:
    """Verify the split is image-, group-, and sample-disjoint.

    Returns a report; ``ok`` is False when any overlap exists. The notebook
    refuses to continue on a False.
    """
    train, test = subset["splits"]["train"], subset["splits"]["test"]
    image_overlap = sorted({g["image_id"] for g in train} & {g["image_id"] for g in test})
    group_overlap = sorted({g["group_id"] for g in train} & {g["group_id"] for g in test})
    audio_overlap = sorted({g["audio_path"] for g in train} & {g["audio_path"] for g in test})
    caption_overlap = sorted({g["caption"] for g in train} & {g["caption"] for g in test})
    shared_concepts = sorted(
        {g["concept"] for g in train if g["concept"]}
        & {g["concept"] for g in test if g["concept"]}
    )
    return {
        "ok": not (image_overlap or group_overlap or audio_overlap or caption_overlap),
        "image_overlap": image_overlap,
        "group_overlap": group_overlap,
        "audio_overlap": audio_overlap,
        "caption_overlap": caption_overlap,
        "shared_concepts_expected": shared_concepts,
        "n_train_groups": len(train),
        "n_test_groups": len(test),
    }
