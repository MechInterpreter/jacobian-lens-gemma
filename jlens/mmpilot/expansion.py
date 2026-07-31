# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Building a larger pilot manifest from metadata already in Drive.

The hand-made ``spokencoco_manifest.json`` holds 48 records, which leaves no
concept with the six distinct images the pilot's split needs. But SpokenCOCO
ships its own full annotation files (``SpokenCOCO_train.json`` /
``SpokenCOCO_val.json``) next to the media, and those are already on disk. This
module finds them, derives a bigger manifest from them, and — critically —
decides whether the *local* data can support the experiment at its stated
thresholds.

Three rules it does not bend:

1. **The original manifest is never touched.** It is read, checksummed, and
   left alone; the expanded manifest is a new file in the run directory.
2. **Nothing is downloaded.** A group exists only if its image and its
   recording are both already on disk.
3. **Thresholds are never lowered to manufacture a result.** If the local data
   cannot support two concepts at the stated coverage, that is a DATASET NO-GO
   with the ranked table attached, not a quietly relaxed requirement.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from jlens.mmpilot.manifest import (
    ManifestSchema,
    ManifestSchemaError,
    SynchronizationError,
    _stable_rank,
    inspect_manifest,
    manifest_checksum,
    normalize_manifest,
    probe_path,
)
from jlens.mmpilot.store import UnitStore, payload_checksum

#: Metadata formats inspected locally. Discovery is bounded by depth, file count and bytes.
SUPPORTED_SUFFIXES = frozenset({".json", ".jsonl", ".csv", ".tsv"})
DEFAULT_SOURCE_PATTERNS = tuple(f"*{suffix}" for suffix in sorted(SUPPORTED_SUFFIXES))
DISCOVERY_SCHEMA_VERSION = "jlens.mmpilot.metadata_discovery.v2"
DERIVATION_SCHEMA_VERSION = "jlens.mmpilot.expanded_manifest.v2"


#: Directories never searched — the run's own outputs must not feed back in.
EXCLUDED_DIR_NAMES = frozenset({"units", "__pycache__", ".ipynb_checkpoints"})
#: Large media-only trees are pruned before traversal. Metadata belongs beside
#: these directories or under ``annotations/``, never inside media folders.
MEDIA_TREE_DIR_NAMES = frozenset(
    {"wavs", "train2014", "val2014", "test2014", "train2017", "val2017", "images"}
)



class DatasetCoverageError(RuntimeError):
    """The local dataset cannot support the pilot at its stated thresholds."""


@dataclass(frozen=True)
class ConceptRequirements:
    """What a concept must offer to be worth spending model passes on.

    Defaults are the pilot's scientific thresholds. They are constructor
    arguments so a caller can state a different standard *explicitly* — never
    so one can be applied silently when the data falls short.
    """

    min_distinct_images: int = 6
    min_groups: int = 6
    min_train_positives: int = 4
    min_test_positives: int = 2
    min_negatives: int = 6

    def to_dict(self) -> dict:
        return asdict(self)


#: Plumbing-only thresholds for :data:`TINY_SMOKE` runs. Not a scientific
#: standard and never reachable by default — see :func:`tiny_smoke_requirements`.
TINY_SMOKE_REQUIREMENTS = ConceptRequirements(
    min_distinct_images=2,
    min_groups=2,
    min_train_positives=1,
    min_test_positives=1,
    min_negatives=2,
)


def tiny_smoke_requirements() -> ConceptRequirements:
    """The TINY_SMOKE thresholds, for validating plumbing against real media.

    A TINY_SMOKE run exercises real Drive reads, real image and audio decoding,
    and the real processor on a handful of examples. It answers "does this run
    at all", never "does J-space transfer". Runs using these thresholds are
    labelled ``mode="tiny_smoke"`` and are excluded from the research verdict.
    """
    return TINY_SMOKE_REQUIREMENTS


# ------------------------------------------------------------- discovery


@dataclass
class MetadataSource:
    """One candidate annotation file found on disk."""

    path: str
    size_bytes: int
    checksum: str | None = None
    n_records: int = 0
    schema: ManifestSchema | None = None
    usable: bool = False
    reason: str = ""
    detected_format: str = ""
    top_level_schema: dict = field(default_factory=dict)
    likely_fields: dict = field(default_factory=dict)
    source_kind: str = "rejected"
    recursion_depth: int = 0
    parser_version: str = DISCOVERY_SCHEMA_VERSION
    payload: object | None = field(default=None, repr=False)

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "checksum": self.checksum,
            "n_records": self.n_records,
            "usable": self.usable,
            "reason": self.reason,
            "schema": self.schema.to_dict() if self.schema else None,
            "detected_format": self.detected_format,
            "top_level_schema": self.top_level_schema,
            "likely_fields": self.likely_fields,
            "source_kind": self.source_kind,
            "recursion_depth": self.recursion_depth,
            "parser_version": self.parser_version,

        }


def _discover_metadata_sources_v1(
    search_roots: Sequence[str | os.PathLike[str]],
    *,
    patterns: Sequence[str] = DEFAULT_SOURCE_PATTERNS,
    exclude: Sequence[str | os.PathLike[str]] = (),
    max_files: int = 20,
    max_bytes: int = 512 * 1024 * 1024,
) -> list[MetadataSource]:
    """Find annotation files under ``search_roots`` and report which are usable.

    "Usable" means :func:`~jlens.mmpilot.manifest.inspect_manifest` resolves an
    image field, an audio field, and a caption field. COCO's own
    ``captions_*.json`` carries no audio, so it is found, reported, and then
    correctly rejected as unable to form synchronized groups — that rejection
    is information, not a failure.

    Files are returned largest-first, since the fuller annotation file is the
    one worth expanding from.
    """
    excluded = {os.path.normcase(os.path.abspath(str(path))) for path in exclude}
    found: dict[str, Path] = {}
    for root in search_roots:
        root_path = Path(root)
        if not root_path.is_dir():
            continue
        for pattern in patterns:
            for candidate in sorted(root_path.glob(pattern)):
                key = os.path.normcase(os.path.abspath(str(candidate)))
                if key in excluded or not candidate.is_file():
                    continue
                if any(part in EXCLUDED_DIR_NAMES for part in candidate.parts):
                    continue
                found.setdefault(key, candidate)

    sources: list[MetadataSource] = []
    for candidate in sorted(found.values(), key=lambda p: (-p.stat().st_size, str(p))):
        size = candidate.stat().st_size
        source = MetadataSource(path=str(candidate), size_bytes=size)
        if size > max_bytes:
            source.reason = f"skipped: {size / 2**20:.0f} MiB exceeds the {max_bytes / 2**20:.0f} MiB cap"
            sources.append(source)
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
            source.reason = f"unreadable: {type(exc).__name__}"
            sources.append(source)
            continue
        try:
            schema = inspect_manifest(payload)
        except ManifestSchemaError as exc:
            source.reason = f"no synchronizable schema: {exc}"
            sources.append(source)
            continue
        source.schema = schema
        source.checksum = manifest_checksum(candidate)
        source.n_records = _count_rows(payload, schema)
        source.usable = True
        source.reason = "image, audio and caption fields resolved"
        sources.append(source)
        if len([s for s in sources if s.usable]) >= max_files:
            break
    return sources


def _count_rows(payload, schema: ManifestSchema) -> int:
    records = (
        payload
        if isinstance(payload, list)
        else payload.get(schema.records_key or "", [])
    )
    if not schema.nested_key:
        return len(records)
    return sum(
        len(record.get(schema.nested_key, []))
        for record in records
        if isinstance(record, Mapping)
    )



def _load_metadata(path: Path) -> object:
    """Parse one bounded metadata file without making schema assumptions."""
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    delimiter = "\t" if suffix == ".tsv" else ","
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def _top_schema(payload: object) -> dict:
    if isinstance(payload, list):
        sample = payload[0] if payload else None
        return {"type": "list", "length": len(payload), "record_fields": sorted(sample) if isinstance(sample, Mapping) else []}
    if isinstance(payload, Mapping):
        return {"type": "object", "keys": sorted(str(key) for key in payload), "list_fields": {str(key): len(value) for key, value in payload.items() if isinstance(value, list)}}
    return {"type": type(payload).__name__}


def _field_inventory(payload: object) -> list[str]:
    records = payload if isinstance(payload, list) else []
    if isinstance(payload, Mapping):
        records = next((value for value in payload.values() if isinstance(value, list) and value), [])
    fields: set[str] = set()
    for record in records[:20]:
        if not isinstance(record, Mapping):
            continue
        fields.update(str(key) for key in record)
        for value in record.values():
            if isinstance(value, list):
                for nested in value[:5]:
                    if isinstance(nested, Mapping):
                        fields.update(str(key) for key in nested)
    return sorted(fields)


def _likely_fields(payload: object) -> dict:
    fields = _field_inventory(payload)
    roles = {
        "image_id": ("image_id", "imageid", "cocoid", "coco_id"),
        "caption_id": ("caption_id", "captionid", "sentence_id", "id"),
        "caption_text": ("caption", "text", "sentence", "transcript"),
        "audio_path": ("audio", "wav", "audio_path", "wav_path", "file_name"),
        "speaker": ("speaker", "speaker_id", "spkid", "speakerid"),
        "split": ("split", "partition", "subset"),
    }
    return {role: [field for field in fields if field.lower() in names] for role, names in roles.items()}


def _is_coco_objects(payload: object) -> bool:
    return isinstance(payload, Mapping) and isinstance(payload.get("categories"), list) and isinstance(payload.get("annotations"), list) and all(isinstance(item, Mapping) and "category_id" in item and "image_id" in item for item in payload["annotations"][:10])


def _walk_metadata(root: Path, *, max_depth: int, max_candidates: int) -> list[tuple[Path, int]]:
    """Walk only metadata-shaped files to a strict depth/count bound."""
    found: list[tuple[Path, int]] = []
    frontier = [(root, 0)]
    while frontier and len(found) < max_candidates:
        directory, depth = frontier.pop(0)
        try:
            children = sorted(directory.iterdir(), key=lambda path: path.name.lower())
        except OSError:
            continue
        for child in children:
            name = child.name.lower()
            if child.name in EXCLUDED_DIR_NAMES or name in MEDIA_TREE_DIR_NAMES:
                continue
            if child.suffix.lower() in SUPPORTED_SUFFIXES:
                if child.is_file():
                    found.append((child, depth))
                    if len(found) >= max_candidates:
                        break
            elif depth < max_depth and child.is_dir():
                frontier.append((child, depth + 1))
    return found


def discover_metadata_sources(
    search_roots: Sequence[str | os.PathLike[str]],
    *,
    patterns: Sequence[str] = DEFAULT_SOURCE_PATTERNS,
    exclude: Sequence[str | os.PathLike[str]] = (),
    max_files: int = 40,
    max_bytes: int = 512 * 1024 * 1024,
    max_depth: int = 3,
) -> list[MetadataSource]:
    """Inspect a bounded set of local JSON/JSONL/CSV/TSV metadata candidates."""
    del patterns
    excluded = {os.path.normcase(os.path.abspath(str(path))) for path in exclude}
    candidates: dict[str, tuple[Path, int]] = {}
    for root in search_roots:
        root_path = Path(root)
        if not root_path.is_dir():
            continue
        for candidate, depth in _walk_metadata(root_path, max_depth=max_depth, max_candidates=max_files):
            key = os.path.normcase(os.path.abspath(str(candidate)))
            if key not in excluded:
                candidates.setdefault(key, (candidate, depth))
            if len(candidates) >= max_files:
                break
    sources: list[MetadataSource] = []
    for candidate, depth in sorted(candidates.values(), key=lambda item: str(item[0])):
        stat_result = probe_path(candidate, root=candidate.parent)
        if stat_result is None:
            continue
        source = MetadataSource(path=str(candidate), size_bytes=stat_result.st_size, detected_format=candidate.suffix.lower().lstrip("."), recursion_depth=depth)
        if stat_result.st_size > max_bytes:
            source.reason = f"rejected: {stat_result.st_size} bytes exceeds max_bytes={max_bytes}"
            sources.append(source)
            continue
        try:
            payload = _load_metadata(candidate)
        except (json.JSONDecodeError, UnicodeDecodeError, csv.Error, OSError) as exc:
            source.reason = f"rejected: unreadable {type(exc).__name__}: {exc}"
            sources.append(source)
            continue
        source.payload = payload
        source.checksum = manifest_checksum(candidate)
        source.top_level_schema = _top_schema(payload)
        source.likely_fields = _likely_fields(payload)
        if _is_coco_objects(payload):
            source.source_kind = "coco_object_annotation"
            source.n_records = len(payload["annotations"])
            source.reason = "accepted: official COCO object categories and annotations"
            sources.append(source)
            continue
        try:
            source.schema = inspect_manifest(payload)
        except ManifestSchemaError as exc:
            source.reason = f"rejected: no deterministic image-caption-audio schema: {exc}"
            sources.append(source)
            continue
        source.n_records = _count_rows(payload, source.schema)
        source.usable = True
        source.source_kind = "synchronized_metadata"
        source.reason = "accepted: explicit image, caption and audio fields"
        sources.append(source)
    return sources
# ------------------------------------------------------------- expansion


@dataclass
class ExpansionResult:
    """The merged manifest plus the provenance needed to trust it."""

    groups: list[dict]
    sources: list[MetadataSource]
    per_source: list[dict]
    baseline_group_ids: list[str] = field(default_factory=list)

    @property
    def n_groups(self) -> int:
        return len(self.groups)

    def to_dict(self, *, original_checksum: str, conversion: Mapping) -> dict:
        return {
            "schema_version": DERIVATION_SCHEMA_VERSION,
            "original_manifest_checksum": original_checksum,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_metadata_paths": [source.path for source in self.sources],
            "source_metadata_checksums": {source.path: source.checksum for source in self.sources},
            "parser_versions": sorted({source.parser_version for source in self.sources}),
            "original_manifest_mutated": False,
            "media_redownloaded": False,
            "conversion": dict(conversion),
            "conversion_hash": payload_checksum(dict(conversion)),
            "synchronization_method": "explicit_metadata_fields_only",
            "sources": [source.to_dict() for source in self.sources],
            "per_source": self.per_source,
            "n_groups": self.n_groups,
            "n_groups_in_original": len(self.baseline_group_ids),
            "groups": self.groups,
        }


def persist_expanded_manifest(
    path: str | os.PathLike[str],
    result: ExpansionResult,
    *,
    original_checksum: str,
    conversion: Mapping,
) -> tuple[dict, str]:
    """Atomically save, or reuse only when source checksums/config match."""
    path = Path(path)
    expected_sources = {source.path: source.checksum for source in result.sources}
    expected_conversion = payload_checksum(dict(conversion))
    if path.is_file():
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            stored = {}
        compatible = (
            stored.get("schema_version") == DERIVATION_SCHEMA_VERSION
            and stored.get("original_manifest_checksum") == original_checksum
            and stored.get("source_metadata_checksums") == expected_sources
            and stored.get("conversion_hash") == expected_conversion
        )
        if compatible:
            return stored, "resuming: reused checksum-compatible expanded manifest"
    payload = result.to_dict(original_checksum=original_checksum, conversion=conversion)
    UnitStore._write_json(path, payload)
    return payload, "starting: wrote expanded manifest atomically"


def _bounded_sync_payload(
    payload: object,
    schema: ManifestSchema,
    *,
    candidate_concepts: Mapping[str, Sequence[str]],
    object_image_ids: set[str],
    max_records: int,
) -> tuple[object, int, int]:
    """Select metadata records before any media probe; return payload/seen/kept."""
    records = payload if isinstance(payload, list) else payload.get(schema.records_key or "", [])
    if len(records) <= max_records:
        return payload, len(records), len(records)
    keywords = {word.lower() for words in candidate_concepts.values() for word in words}
    positives: list[Mapping] = []
    negatives: list[Mapping] = []
    for record in records:
        if not isinstance(record, Mapping):
            continue
        image_id = str(record.get(schema.image_id_field, "")) if schema.image_id_field else ""
        nested = record.get(schema.nested_key, []) if schema.nested_key else [record]
        captions = [str(item.get(schema.caption_field, "")) for item in nested if isinstance(item, Mapping)]
        is_candidate = image_id in object_image_ids or any(_normalized_words(caption) & keywords for caption in captions)
        (positives if is_candidate else negatives).append(record)
    positives.sort(key=lambda record: _stable_rank(str(record), "metadata-positive"))
    negatives.sort(key=lambda record: _stable_rank(str(record), "metadata-negative"))
    chosen = positives[:max_records]
    chosen += negatives[: max(0, max_records - len(chosen))]
    if isinstance(payload, list):
        return chosen, len(records), len(chosen)
    derived = dict(payload)
    derived[schema.records_key] = chosen
    return derived, len(records), len(chosen)


def build_expanded_manifest(
    sources: Sequence[MetadataSource],
    *,
    image_roots: Sequence[str | os.PathLike[str]],
    audio_roots: Sequence[str | os.PathLike[str]],
    baseline_groups: Sequence[Mapping] = (),
    annotation_sources: Sequence[MetadataSource] = (),
    max_groups: int | None = None,
    candidate_concepts: Mapping[str, Sequence[str]] = (),
    max_metadata_records: int = 20000,
) -> ExpansionResult:
    """Merge every usable source into one set of validated synchronized groups.

    Each source is normalized with the shared resolver, so a group survives
    only when its image *and* its recording are both present on disk. Groups
    are keyed by the same ``group_id`` rule as the original manifest, so
    records the 48-record file already contained are recognised rather than
    duplicated.

    Args:
        max_groups: Stop after this many groups. The pilot needs tens, and a
            full SpokenCOCO split is hundreds of thousands of rows whose media
            mostly is not on disk — checking every one would take hours.

    Raises:
        DatasetCoverageError: If no source yields a single synchronized group.
    """
    merged: dict[str, dict] = {}
    for group in baseline_groups:
        merged[group["group_id"]] = {**dict(group), "synchronization_method": "original_manifest_explicit_fields", "media_validation_status": "valid"}
    baseline_ids = list(merged)
    audio_owners = {group["audio_path"]: (group["image_id"], group["caption"]) for group in merged.values()}

    object_image_ids: set[str] = set()
    candidate_names = {name.lower() for name in candidate_concepts}
    for annotation_source in annotation_sources:
        annotation_payload = annotation_source.payload if annotation_source.payload is not None else _load_metadata(Path(annotation_source.path))
        if not _is_coco_objects(annotation_payload):
            continue
        categories = {str(item["id"]): str(item["name"]).lower() for item in annotation_payload["categories"]}
        object_image_ids.update(str(item["image_id"]) for item in annotation_payload["annotations"] if categories.get(str(item["category_id"])) in candidate_names)

    per_source: list[dict] = []
    for source in sources:
        if not source.usable:
            per_source.append({"path": source.path, "skipped": source.reason})
            continue
        payload = source.payload if source.payload is not None else _load_metadata(Path(source.path))
        payload, metadata_records_seen, metadata_records_considered = _bounded_sync_payload(
            payload,
            source.schema,
            candidate_concepts=candidate_concepts,
            object_image_ids=object_image_ids,
            max_records=max_metadata_records,
        )
        added = 0
        try:
            normalized = normalize_manifest(
                payload,
                source.schema,
                image_roots=image_roots,
                audio_roots=audio_roots,
                source_checksum=source.checksum or "",
                min_complete_groups=1,
                max_missing_fraction=1.0,
            )
        except SynchronizationError as exc:
            per_source.append(
                {
                    "path": source.path,
                    "checksum": source.checksum,
                    "n_groups": 0,
                    "note": f"no groups resolved: {str(exc).splitlines()[0]}",
                }
            )
            continue
        for group in normalized.groups:
            if not group.get("image_id") or not group.get("audio_path"):
                raise DatasetCoverageError(f"rejected incomplete synchronized join from {source.path}: {group}")
            owner = audio_owners.get(group["audio_path"])
            identity = (group["image_id"], group["caption"])
            if owner is not None and owner != identity:
                raise DatasetCoverageError(
                    f"conflicting join: audio {group['audio_path']} maps to both {owner} and {identity}"
                )
            audio_owners[group["audio_path"]] = identity
            if max_groups is not None and len(merged) >= max_groups:
                break
            if group["group_id"] not in merged:
                merged[group["group_id"]] = {
                    **group,
                    "source_file": source.path,
                    "source_metadata_checksum": source.checksum,
                    "synchronization_method": "explicit_metadata_fields",
                    "caption_id": group.get("caption_id") or "caption_sha256:" + hashlib.sha256(group["caption"].encode()).hexdigest()[:16],
                    "audio_record_id": group.get("audio_record_id") or Path(group["audio_path"]).stem,
                    "identifier_derivation": {"caption_id": "caption_text_sha256_when_absent", "audio_record_id": "validated_unique_audio_path_stem_when_absent"},
                    "media_validation_status": "valid",
                    "split_provenance": {"source_split": group.get("source_split"), "assignment": "pending_deterministic_pilot_split"},
                }
                added += 1
        per_source.append(
            {
                "path": source.path,
                "checksum": source.checksum,
                "n_records": source.n_records,
                "n_metadata_records_seen": metadata_records_seen,
                "n_metadata_records_considered": metadata_records_considered,
                "n_groups_resolved": len(normalized.groups),
                "n_groups_added": added,
                "n_missing_image_files": normalized.audit["n_missing_image_files"],
                "n_missing_audio_files": normalized.audit["n_missing_audio_files"],
            }
        )

    if not merged:
        raise DatasetCoverageError(
            "no synchronized image/caption/audio group could be built from any "
            "local metadata file. Sources examined:\n"
            + json.dumps(per_source, indent=2)
        )
    categories: dict[str, set[str]] = {}
    for source in annotation_sources:
        payload = source.payload if source.payload is not None else _load_metadata(Path(source.path))
        if not _is_coco_objects(payload):
            continue
        names = {str(item["id"]): str(item["name"]).lower() for item in payload["categories"]}
        for annotation in payload["annotations"]:
            name = names.get(str(annotation["category_id"]))
            if name:
                categories.setdefault(str(annotation["image_id"]), set()).add(name)
    for group in merged.values():
        labels = sorted(categories.get(str(group["image_id"]), set()))
        group["concept_annotations"] = labels
        group["annotation_source"] = "coco_object_annotation" if labels else "caption_normalized"
        group.setdefault("synchronized_group_id", group["group_id"])

    groups = [merged[key] for key in sorted(merged)]
    return ExpansionResult(
        groups=groups,
        sources=list(sources),
        per_source=per_source,
        baseline_group_ids=baseline_ids,
    )


# --------------------------------------------------------------- ranking


def _split_plan(
    image_ids: Sequence[str],
    by_image: Mapping[str, Sequence[Mapping]],
    *,
    concept: str,
    groups_per_concept: int,
    max_groups_per_image: int,
    seed: str,
) -> dict:
    """Mirror :func:`~jlens.mmpilot.manifest.build_subset`'s selection exactly.

    Ranking has to predict what the split will actually contain, so it applies
    the same stable ordering and the same halving rather than estimating.
    """
    ordered = sorted(image_ids, key=lambda image_id: _stable_rank(image_id, f"{seed}|{concept}"))
    chosen = ordered[:groups_per_concept]
    n_train = min(max(1, groups_per_concept - 2), max(1, len(chosen) - 1))
    train_images, test_images = chosen[:n_train], chosen[n_train:]

    def count(images: Sequence[str]) -> int:
        return sum(len(by_image[image_id][:max_groups_per_image]) for image_id in images)

    return {
        "n_selected_images": len(chosen),
        "n_train_images": len(train_images),
        "n_test_images": len(test_images),
        "n_train_positives": count(train_images),
        "n_test_positives": count(test_images),
        "n_groups_selected": count(chosen),
    }


def validated_filename_identifier(path: str, expected_ids: Sequence[str]) -> str:
    """Extract an ID only when metadata validates exactly one filename match."""
    stem = Path(path).stem
    matches = sorted({str(value) for value in expected_ids if stem == str(value) or stem.endswith("_" + str(value))})
    if len(matches) != 1:
        raise DatasetCoverageError(
            f"filename-derived identifier is ambiguous for {path!r}: matches={matches}; explicit metadata is required"
        )
    return matches[0]


def _normalized_words(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    out = set(words)
    out.update(word[:-1] for word in words if len(word) > 3 and word.endswith("s") and not word.endswith("ss"))
    return out


def _group_matches(group: Mapping, concept: str, keywords: Sequence[str]) -> tuple[bool, str]:
    official = {str(label).lower() for label in group.get("concept_annotations", [])}
    if official:
        return concept.lower() in official or bool(official & {word.lower() for word in keywords}), "coco_object_annotation"
    words = _normalized_words(str(group.get("caption", "")))
    exact = {word.lower() for word in keywords}
    hit = bool(words & exact) or concept.lower() in words
    return hit, "caption_normalized" if hit else "none"


def rank_concepts(
    groups: Sequence[Mapping],
    concepts: Mapping[str, Sequence[str]],
    *,
    requirements: ConceptRequirements | None = None,
    groups_per_concept: int = 6,
    max_groups_per_image: int = 2,
    seed: str = "spokencoco-pilot",
) -> list[dict]:
    """Score every candidate concept against what the split will really yield.

    Ordering is feasibility first, then distinct images, then synchronized
    groups, then speakers, then name — deterministic, and biased toward the
    quantity that actually limits this pilot (independent images).

    Each row carries ``unmet``: the named requirements it failed, so a
    DATASET NO-GO can say exactly what was short.
    """
    requirements = requirements or ConceptRequirements()
    by_image: dict[str, list[dict]] = {}
    for group in groups:
        by_image.setdefault(group["image_id"], []).append(dict(group))
    for image_groups in by_image.values():
        image_groups.sort(key=lambda g: g["group_id"])

    matched_concepts: dict[str, set[str]] = {}
    for image_id, image_groups in by_image.items():
        matched_concepts[image_id] = {
            concept
            for concept, keywords in concepts.items()
            if any(_group_matches(g, concept, keywords)[0] for g in image_groups)
        }
    negative_images = [
        image_id for image_id, matched in matched_concepts.items() if not matched
    ]
    n_negative_groups = sum(
        len(by_image[image_id][:max_groups_per_image]) for image_id in negative_images
    )

    rows: list[dict] = []
    for concept in sorted(concepts):
        # Only images that mention this concept *alone* are usable: an image
        # naming two candidate concepts is dropped by build_subset, so counting
        # it here would promise groups the split will not deliver.
        pure = [
            image_id
            for image_id, matched in matched_concepts.items()
            if matched == {concept}
        ]
        concept_groups = [g for image_id in pure for g in by_image[image_id]]
        speakers = {g["speaker"] for g in concept_groups if g.get("speaker")}
        plan = _split_plan(
            pure,
            by_image,
            concept=concept,
            groups_per_concept=groups_per_concept,
            max_groups_per_image=max_groups_per_image,
            seed=seed,
        )
        unmet = []
        if len(pure) < requirements.min_distinct_images:
            unmet.append(f"distinct_images {len(pure)} < {requirements.min_distinct_images}")
        if plan["n_groups_selected"] < requirements.min_groups:
            unmet.append(
                f"synchronized_groups {plan['n_groups_selected']} < {requirements.min_groups}"
            )
        if plan["n_train_positives"] < requirements.min_train_positives:
            unmet.append(
                f"train_positives {plan['n_train_positives']} < {requirements.min_train_positives}"
            )
        if plan["n_test_positives"] < requirements.min_test_positives:
            unmet.append(
                f"test_positives {plan['n_test_positives']} < {requirements.min_test_positives}"
            )
        if n_negative_groups < requirements.min_negatives:
            unmet.append(f"negatives {n_negative_groups} < {requirements.min_negatives}")
        rows.append(
            {
                "concept": concept,
                "n_distinct_images": len(pure),
                "n_groups": len(concept_groups),
                "n_speakers": len(speakers),
                "annotation_source": "coco_object_annotation" if any(g.get("concept_annotations") for g in concept_groups) else "caption_normalized",
                "eligible_train_positives": plan["n_train_positives"],
                "eligible_held_out_positives": plan["n_test_positives"],
                "matched_negatives": n_negative_groups,
                "split_feasible": not any(item.startswith(("distinct_images", "synchronized_groups", "train_positives", "test_positives")) for item in unmet),
                "n_negative_groups": n_negative_groups,
                "n_negative_images": len(negative_images),
                **plan,
                "feasible": not unmet,
                "unmet": unmet,
            }
        )
    rows.sort(
        key=lambda row: (
            not row["feasible"],
            -row["n_distinct_images"],
            -row["n_groups"],
            -row["n_speakers"],
            row["concept"],
        )
    )
    return rows


def format_ranking_table(rows: Sequence[Mapping]) -> str:
    """A fixed-width ranked table, printed before any concept is selected."""
    header = (
        f"{'concept':12s} {'images':>6s} {'groups':>6s} {'spk':>4s} "
        f"{'train+':>6s} {'test+':>5s} {'neg':>4s} {'source':>22s} {'split':>5s} {'ok':>2s} rejection"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{row['concept']:12s} {row['n_distinct_images']:6d} "
            f"{row['n_groups_selected']:6d} {row['n_speakers']:4d} "
            f"{row['n_train_positives']:6d} {row['n_test_positives']:5d} "
            f"{row['n_negative_groups']:4d} {row['annotation_source']:>22s} "
            f"{'yes' if row['split_feasible'] else 'no':>5s} {'ok' if row['feasible'] else 'NO':>2s} "
            f"{'; '.join(row['unmet'])}"
        )
    return "\n".join(lines)


def select_concepts(
    rows: Sequence[Mapping],
    *,
    n_concepts: int = 2,
    requirements: ConceptRequirements | None = None,
    total_synchronized_records: int | None = None,
    coverage_cause: str = "insufficient valid synchronized records after metadata join and media validation",
) -> list[str]:
    """The top ``n_concepts`` feasible concepts, or a DATASET NO-GO.

    Raises:
        DatasetCoverageError: If fewer than ``n_concepts`` concepts clear the
            requirements. The message carries the ranked table and says plainly
            that the fix is more local data, not smaller thresholds — lowering
            them here would change what the pilot's GO means without saying so.
    """
    requirements = requirements or ConceptRequirements()
    feasible = [row["concept"] for row in rows if row["feasible"]]
    if len(feasible) < n_concepts:
        raise DatasetCoverageError(
            f"DATASET NO-GO: {len(feasible)} concept(s) meet the pilot's coverage "
            f"requirements, {n_concepts} needed.\n\n"
            f"Total synchronized records found: {total_synchronized_records if total_synchronized_records is not None else 'see table'}\n"
            f"Coverage cause: {coverage_cause}\n"
            "Smallest additional coverage required is shown by each '<' shortfall in the table; at minimum two concepts must each reach 6 images/groups, 4 train positives and 2 held-out positives, with 6 negatives.\n\n"

            f"{format_ranking_table(rows)}\n\n"
            f"Requirements applied: {requirements.to_dict()}\n\n"
            "The local Drive dataset does not hold enough synchronized "
            "image/caption/audio groups for these concepts. Add more local "
            "SpokenCOCO records (images and their recordings both present on "
            "disk) and re-run.\n"
            "These thresholds are NOT lowered automatically: a smaller split "
            "would change what a GO verdict means. To validate plumbing only, "
            "set TINY_SMOKE=True, whose result is explicitly not scientific "
            "evidence and never feeds the research verdict."
        )
    return feasible[:n_concepts]
