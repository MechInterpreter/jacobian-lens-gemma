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

from jlens.mmpilot.concepts import AMBIGUITY_SCORE as _AMBIGUITY_SCORE
from jlens.mmpilot.evidence import (
    EvidenceConfig,
    EvidenceIndex,
    build_evidence_index,
    config_for_concepts,
    group_evidence,
    is_valid_positive,
)
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
from jlens.mmpilot.selection import PILOT_PROFILE, SubsetProfile
from jlens.mmpilot.store import UnitStore, payload_checksum

#: Metadata formats inspected locally. Discovery is bounded by depth, file count and bytes.
SUPPORTED_SUFFIXES = frozenset({".json", ".jsonl", ".csv", ".tsv"})
DEFAULT_SOURCE_PATTERNS = tuple(f"*{suffix}" for suffix in sorted(SUPPORTED_SUFFIXES))
DISCOVERY_SCHEMA_VERSION = "jlens.mmpilot.metadata_discovery.v2"
DERIVATION_SCHEMA_VERSION = "jlens.mmpilot.expanded_manifest.v3"
COCO_IMAGE_KEY_VERSION = "split-aware-file-name-v1"

_COCO_SPLIT_RE = re.compile(
    r"(?P<split>train|val|test)(?P<year>20\d{2})", re.IGNORECASE
)
_COCO_ID_RE = re.compile(r"(?:^|_)(?P<id>\d{1,12})(?:\.[^.]+)?$")


#: Directories never searched — the run's own outputs must not feed back in.
EXCLUDED_DIR_NAMES = frozenset({"units", "__pycache__", ".ipynb_checkpoints"})
#: Large media-only trees are pruned before traversal. Metadata belongs beside
#: these directories or under ``annotations/``, never inside media folders.
MEDIA_TREE_DIR_NAMES = frozenset(
    {"wavs", "train2014", "val2014", "test2014", "train2017", "val2017", "images"}
)



class DatasetCoverageError(RuntimeError):
    """The local dataset cannot support the pilot at its stated thresholds."""


def canonical_coco_image_key(
    image_id: object,
    *,
    image_ref: object = "",
    split: object = "",
) -> str:
    """Normalize COCO integer ids and filenames to one split-aware key.

    Official SpokenCOCO metadata usually identifies an image through a name
    such as ``COCO_train2014_000000419532.jpg`` whereas COCO annotations use
    integer id ``419532``. Comparing those raw strings gives zero overlap.
    """
    references = [str(image_ref or ""), str(image_id or ""), str(split or "")]
    split_name = ""
    for reference in references:
        match = _COCO_SPLIT_RE.search(reference.replace("\\", "/"))
        if match:
            split_name = f"{match.group('split').lower()}{match.group('year')}"
            break

    raw_id = str(image_id or "").strip()
    id_match = _COCO_ID_RE.search(raw_id)
    if not id_match:
        id_match = _COCO_ID_RE.search(Path(str(image_ref or "")).name)
    normalized_id = str(int(id_match.group("id"))) if id_match else raw_id
    if not normalized_id:
        return ""
    return (
        f"{split_name}:{normalized_id}"
        if split_name and id_match
        else normalized_id
    )


def _annotation_image_key(
    image_id: object, *, image_ref: object, source_path: object
) -> str:
    """Key an annotation without inventing COCO semantics for generic ids."""
    raw_id = str(image_id or "").strip()
    split_hint = source_path if image_ref or raw_id.isdigit() else ""
    return canonical_coco_image_key(
        image_id,
        image_ref=image_ref,
        split=split_hint,
    )


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


class ExpandedManifestIncompatible(RuntimeError):
    """A stored expanded manifest was not derived from these inputs.

    Raised rather than silently rebuilt, because the two failures look
    identical from a notebook's point of view and cost very different amounts:
    a genuine miss is twenty-five minutes of honest work, and a fingerprint
    typo is twenty-five minutes of work whose result is then discarded.
    """

    def __init__(self, message: str, *, record: Mapping | None = None) -> None:
        super().__init__(message)
        self.record = dict(record or {})


def expanded_manifest_compatibility(
    stored: Mapping | None,
    *,
    original_checksum: str,
    expected_sources: Mapping[str, str],
    conversion: Mapping,
    expected_group_count: int | None = None,
    expected_lexicon_hash: str | None = None,
) -> dict:
    """Clause-by-clause: was ``stored`` derived from exactly these inputs?

    Every clause is reported with its own flag and its two values, so a miss
    says *which* input moved instead of only that something did.

    ``expected_group_count`` and ``expected_lexicon_hash`` are optional
    because they are assertions about a *particular* cache a study expects to
    find, not properties of the format. A study that knows it is loading the
    125,198-group join states the number and gets a refusal if it is handed a
    48-group smoke manifest.
    """
    stored = dict(stored or {})
    recorded_conversion = dict(stored.get("conversion") or {})
    clauses = [
        {
            "clause": "schema_version",
            "expected": DERIVATION_SCHEMA_VERSION,
            "found": stored.get("schema_version"),
        },
        {
            "clause": "original_manifest_checksum",
            "expected": str(original_checksum),
            "found": stored.get("original_manifest_checksum"),
        },
        {
            "clause": "source_metadata_checksums",
            "expected": dict(expected_sources),
            "found": stored.get("source_metadata_checksums"),
        },
        {
            "clause": "conversion_hash",
            "expected": payload_checksum(dict(conversion)),
            "found": stored.get("conversion_hash"),
        },
    ]
    if expected_group_count is not None:
        clauses.append(
            {
                "clause": "n_groups",
                "expected": int(expected_group_count),
                "found": (
                    int(stored["n_groups"]) if "n_groups" in stored else None
                ),
            }
        )
    if expected_lexicon_hash is not None:
        clauses.append(
            {
                "clause": "evidence_lexicon_hash",
                "expected": str(expected_lexicon_hash),
                "found": recorded_conversion.get("evidence_lexicon_hash"),
            }
        )
    for clause in clauses:
        clause["passed"] = clause["found"] == clause["expected"]
    failed = [clause["clause"] for clause in clauses if not clause["passed"]]
    return {
        "schema": "jlens.mmpilot.expanded_manifest_compatibility.v1",
        "derivation_schema_version": DERIVATION_SCHEMA_VERSION,
        "clauses": clauses,
        "failed_clauses": failed,
        "compatible": not failed,
        "n_groups": stored.get("n_groups"),
    }


def load_expanded_manifest(
    path: str | os.PathLike[str],
    *,
    original_checksum: str,
    expected_sources: Mapping[str, str],
    conversion: Mapping,
    expected_group_count: int | None = None,
    expected_lexicon_hash: str | None = None,
) -> tuple[dict, dict]:
    """Read a compatible expanded manifest **without building one**.

    This is the half of :func:`persist_expanded_manifest` that a resuming study
    actually needs, separated out because the combined function cannot be used
    for a cache hit: it takes an :class:`ExpansionResult`, so by the time it can
    say "compatible, reuse it" the caller has already paid for
    :func:`build_expanded_manifest` and the entire evidence join. The
    compatibility test never depended on the rebuild — only on the source
    checksums, which are cheap — so it is available here on its own.

    The manifest is opened read-only and nothing is written.

    Returns:
        ``(payload, record)``. ``payload`` is the stored document, groups
        included; ``record`` is the clause table.

    Raises:
        ExpandedManifestIncompatible: If the file is missing, unreadable, or
            fails any clause.
    """
    target = Path(path)
    if not target.is_file():
        raise ExpandedManifestIncompatible(
            f"no expanded manifest at {target}; there is nothing to load and "
            "this loader does not build one",
            record={"path": str(target), "present": False, "compatible": False},
        )
    try:
        stored = json.loads(target.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
        raise ExpandedManifestIncompatible(
            f"{target} is not readable JSON: {exc}",
            record={"path": str(target), "present": True, "readable": False},
        ) from exc
    if not isinstance(stored, Mapping):
        raise ExpandedManifestIncompatible(
            f"{target} does not hold a JSON object",
            record={"path": str(target), "present": True, "readable": False},
        )
    record = expanded_manifest_compatibility(
        stored,
        original_checksum=original_checksum,
        expected_sources=expected_sources,
        conversion=conversion,
        expected_group_count=expected_group_count,
        expected_lexicon_hash=expected_lexicon_hash,
    )
    record["path"] = str(target)
    record["present"] = True
    record["readable"] = True
    record["manifest_file_checksum"] = manifest_checksum(str(target))
    if not record["compatible"]:
        detail = "\n  - ".join(
            f"{clause['clause']}: expected {clause['expected']!r}, "
            f"found {clause['found']!r}"
            for clause in record["clauses"]
            if not clause["passed"]
        )
        raise ExpandedManifestIncompatible(
            f"{target} was derived from different inputs:\n  - {detail}\n"
            "Refusing to reuse it. Point the study at the cache it belongs to, "
            "or rebuild deliberately.",
            record=record,
        )
    if not isinstance(stored.get("groups"), list):
        raise ExpandedManifestIncompatible(
            f"{target} passed every provenance clause but carries no 'groups' "
            "list, so there is nothing to load",
            record=record,
        )
    return dict(stored), record


def persist_expanded_manifest(
    path: str | os.PathLike[str],
    result: ExpansionResult,
    *,
    original_checksum: str,
    conversion: Mapping,
) -> tuple[dict, str]:
    """Atomically save, or reuse only when source checksums/config match.

    Note that reuse here is a *late* check: the caller has already built
    ``result``. Use :func:`load_expanded_manifest` when the point is to avoid
    the build.
    """
    path = Path(path)
    expected_sources = {source.path: source.checksum for source in result.sources}
    if path.is_file():
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            stored = {}
        record = expanded_manifest_compatibility(
            stored,
            original_checksum=original_checksum,
            expected_sources=expected_sources,
            conversion=conversion,
        )
        if record["compatible"]:
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
        image_ref = record.get(schema.image_field, "")
        source_split = (
            record.get(schema.split_field, "") if schema.split_field else ""
        )
        image_key = canonical_coco_image_key(
            image_id or Path(str(image_ref)).stem,
            image_ref=image_ref,
            split=source_split,
        )
        nested = record.get(schema.nested_key, []) if schema.nested_key else [record]
        captions = [str(item.get(schema.caption_field, "")) for item in nested if isinstance(item, Mapping)]
        is_candidate = image_key in object_image_ids or any(
            _normalized_words(caption) & keywords for caption in captions
        )
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


def coco_object_labels(annotation_sources: Sequence[MetadataSource]) -> dict[str, set[str]]:
    """``{image_id: {category_name, ...}}`` from official COCO instance files.

    This is the *visual* half of the evidence rule. It says what is in the
    picture, and nothing at all about what the caption says.
    """
    labels: dict[str, set[str]] = {}
    for source in annotation_sources:
        payload = (
            source.payload
            if source.payload is not None
            else _load_metadata(Path(source.path))
        )
        if not _is_coco_objects(payload):
            continue
        names = {
            str(item["id"]): str(item["name"]).lower()
            for item in payload["categories"]
        }
        source_split = Path(source.path).stem
        image_files = {
            str(item["id"]): str(item.get("file_name", ""))
            for item in payload.get("images", [])
            if isinstance(item, Mapping) and "id" in item
        }
        for annotation in payload["annotations"]:
            name = names.get(str(annotation["category_id"]))
            if name:
                image_id = annotation["image_id"]
                key = _annotation_image_key(
                    image_id,
                    image_ref=image_files.get(str(image_id), ""),
                    source_path=source_split,
                )
                labels.setdefault(key, set()).add(name)
    return labels


def attach_concept_annotations(
    groups, annotation_sources: Sequence[MetadataSource]
) -> None:
    """Attach COCO object labels to ``groups`` in place.

    ``annotation_source`` records whether an image actually had an object
    annotation. ``"none"`` is a real state: without it, the group carries no
    visual evidence and cannot be a valid synchronized positive for anything.
    """
    labels = coco_object_labels(annotation_sources)
    matched = 0
    unmatched: list[dict] = []
    for group in groups:
        key = canonical_coco_image_key(
            group.get("image_id"),
            image_ref=group.get("image_path", ""),
            split=group.get("source_split", ""),
        )
        found = sorted(labels.get(key, set()))
        group["concept_annotations"] = found
        group["annotation_source"] = "coco_object_annotation" if found else "none"
        group["coco_image_key"] = key
        group.setdefault("synchronized_group_id", group["group_id"])
        if found:
            matched += 1
        elif len(unmatched) < 5:
            unmatched.append(
                {
                    "raw_image_id": group.get("image_id"),
                    "image_path": group.get("image_path"),
                    "source_split": group.get("source_split"),
                    "normalized_key": key,
                }
            )
    if annotation_sources and labels and matched == 0:
        raise DatasetCoverageError(
            "COCO annotation join failure: instance files were discovered and "
            "parsed, but zero synchronized groups matched their split-aware "
            "image keys. This is not a missing-annotation-file error.\n"
            f"  synchronized key examples: {json.dumps(unmatched, sort_keys=True)}\n"
            f"  annotation key examples: {json.dumps(sorted(labels)[:5])}"
        )


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
        categories = {
            str(item["id"]): str(item["name"]).lower()
            for item in annotation_payload["categories"]
        }
        image_files = {
            str(item["id"]): str(item.get("file_name", ""))
            for item in annotation_payload.get("images", [])
            if isinstance(item, Mapping) and "id" in item
        }
        source_split = Path(annotation_source.path).stem
        object_image_ids.update(
            _annotation_image_key(
                item["image_id"],
                image_ref=image_files.get(str(item["image_id"]), ""),
                source_path=source_split,
            )
            for item in annotation_payload["annotations"]
            if categories.get(str(item["category_id"])) in candidate_names
        )

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
    attach_concept_annotations(merged.values(), annotation_sources)

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
    evidence_config: EvidenceConfig,
    profile: SubsetProfile | None = None,
    valid_group_ids: set[str] | None = None,
) -> dict:
    """Mirror :func:`~jlens.mmpilot.manifest.build_subset`'s selection exactly.

    Ranking has to predict what the split will actually contain, so it applies
    the same stable ordering and the same split rule rather than estimating.

    The ``profile`` argument is not decoration. Under a profile with explicit
    per-split image counts the subset builder splits 8/8, while this function's
    pilot rule splits ``groups_per_concept - 2`` / 2 — so without it the
    ranking would report every concept as short of held-out positives and
    reject the whole candidate set for a shortfall that does not exist.
    """
    profile = profile or PILOT_PROFILE
    ordered = sorted(image_ids, key=lambda image_id: _stable_rank(image_id, f"{seed}|{concept}"))
    if profile.explicit_image_counts:
        max_groups_per_image = profile.max_groups_per_image
        n_train = int(profile.n_train_positive_images)
        n_test = int(profile.n_test_positive_images)
        source_train = [
            image_id
            for image_id in ordered
            if any(
                str(g.get("source_split", "")).lower().startswith("train")
                for g in by_image[image_id]
            )
        ]
        source_test = [
            image_id
            for image_id in ordered
            if image_id not in set(source_train)
            and any(
                str(g.get("source_split", "")).lower().startswith(("val", "test"))
                for g in by_image[image_id]
            )
        ]
        if len(source_train) >= n_train and len(source_test) >= n_test:
            chosen = source_train[:n_train] + source_test[:n_test]
        else:
            chosen = ordered[: n_train + n_test]
        n_train = min(n_train, max(0, len(chosen) - 1)) if len(chosen) <= n_train else n_train
    else:
        chosen = ordered[:groups_per_concept]
        n_train = min(max(1, groups_per_concept - 2), max(1, len(chosen) - 1))
    train_images, test_images = chosen[:n_train], chosen[n_train:]

    def count(images: Sequence[str]) -> int:
        return sum(
            len(
                [
                    group
                    for group in by_image[image_id]
                    if (
                        group["group_id"] in valid_group_ids
                        if valid_group_ids is not None
                        else is_valid_positive(group, concept, evidence_config)
                    )
                ][:max_groups_per_image]
            )
            for image_id in images
        )

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


def _group_matches(
    group: Mapping,
    concept: str,
    keywords: Sequence[str],
    config: EvidenceConfig | None = None,
) -> tuple[bool, str]:
    """Whether ``group`` is a valid synchronized positive for ``concept``.

    Both kinds of evidence are required. This used to return on the COCO
    annotation alone, which is what selected images of a cat whose caption
    never says "cat" — see :mod:`jlens.mmpilot.evidence`.
    """
    config = config or config_for_concepts({concept: tuple(keywords)})
    record = group_evidence(group, concept, config)
    if record["is_valid_synchronized_positive"]:
        return True, "coco_object_annotation_and_caption_lexicon"
    return False, record["rejection_reason"]


#: Weights of the ranking score's components. Fixed constants, documented in
#: :func:`concept_score`, not tuned against any result. They sum to
#: :data:`MAX_CONCEPT_SCORE`.
SCORE_WEIGHTS: dict[str, float] = {
    "images": 3.0,
    "groups": 2.0,
    "split": 1.5,
    "negatives": 1.0,
    "speakers": 1.0,
    "precision": 2.0,
    "ambiguity": 1.5,
    "independence": 1.0,
}
MAX_CONCEPT_SCORE = sum(SCORE_WEIGHTS.values())

#: Saturation points. Beyond these, more of the same buys nothing — the pilot
#: needs *enough* independent images, not the most frequent category.
IMAGE_SATURATION = 12
GROUP_SATURATION = 12
NEGATIVE_SATURATION = 24
SPEAKER_SATURATION = 3
HEADROOM_SATURATION = 4

#: How much each lexical-ambiguity status is trusted. Mirrors
#: :data:`jlens.mmpilot.concepts.AMBIGUITY_SCORE`, duplicated as a fallback so
#: a config carrying a status this module has not seen still scores something
#: conservative rather than raising.
AMBIGUITY_SCORE = dict(_AMBIGUITY_SCORE)


def _saturating(value: float, cap: float) -> float:
    return min(max(float(value), 0.0) / float(cap), 1.0) if cap else 0.0


def concept_score(row: Mapping) -> dict:
    """The deterministic ranking score for one coverage row, and its parts.

    Feasible concepts are ordered by this score, **not** by raw frequency. The
    most frequent COCO category is ``person``, which is annotated on most images
    and therefore discriminates nothing; ranking by count alone would put it
    first. Each component is a saturating ratio in ``[0, 1]``, multiplied by a
    fixed weight:

    ``images`` (3.0)
        Distinct images that are a valid positive for this concept **alone**,
        saturating at :data:`IMAGE_SATURATION`. Independent images are what the
        statistics actually need, so this is the heaviest term.
    ``groups`` (2.0)
        Synchronized caption-confirmed groups the split will really select,
        saturating at :data:`GROUP_SATURATION`.
    ``split`` (1.5)
        Headroom above the train and held-out minimums, half each — a concept
        that clears them with room to spare survives a lost media file.
    ``negatives`` (1.0)
        Clean matched negatives available, saturating at
        :data:`NEGATIVE_SATURATION`.
    ``speakers`` (1.0)
        Distinct speakers among the concept's groups, saturating at
        :data:`SPEAKER_SATURATION`. Speaker metadata is often absent, in which
        case every concept scores zero here and the term drops out.
    ``precision`` (2.0)
        Share of this concept's caption mentions that the COCO annotation also
        backs. A term that fires on captions whose images do not contain the
        object is imprecise, and this is the only precision signal available
        without labelling anything by hand.
    ``ambiguity`` (1.5)
        From the lexical specification: ``clean`` 1.0 down to ``ambiguous``
        0.4. See :mod:`jlens.mmpilot.concepts`.
    ``independence`` (1.0)
        ``1 - max co-occurrence fraction``: how much this concept depends on
        one other category being in the same pictures. A concept whose images
        always also contain ``person`` cannot be isolated from it.

    Every input is an integer count or an exactly-representable ratio, and the
    result is rounded to six decimals, so the ordering is reproducible across
    machines.
    """
    parts = {
        "images": _saturating(row.get("n_distinct_images", 0), IMAGE_SATURATION),
        "groups": _saturating(row.get("n_groups_selected", 0), GROUP_SATURATION),
        "split": 0.5
        * (
            _saturating(row.get("train_headroom", 0), HEADROOM_SATURATION)
            + _saturating(row.get("test_headroom", 0), HEADROOM_SATURATION)
        ),
        "negatives": _saturating(row.get("n_negative_groups", 0), NEGATIVE_SATURATION),
        "speakers": _saturating(row.get("n_speakers", 0), SPEAKER_SATURATION),
        "precision": min(max(float(row.get("lexical_precision", 0.0)), 0.0), 1.0),
        "ambiguity": AMBIGUITY_SCORE.get(row.get("lexical_ambiguity", "clean"), 0.4),
        "independence": 1.0
        - min(max(float(row.get("max_cooccurrence_fraction", 0.0)), 0.0), 1.0),
    }
    total = sum(SCORE_WEIGHTS[name] * value for name, value in parts.items())
    return {
        "score": round(total, 6),
        "components": {name: round(value, 6) for name, value in sorted(parts.items())},
        "weights": dict(sorted(SCORE_WEIGHTS.items())),
        "max_score": MAX_CONCEPT_SCORE,
    }


def cooccurrence_statistics(
    image_ids: Sequence[str],
    by_image: Mapping[str, Sequence[Mapping]],
    concept: str,
    *,
    top_n: int = 5,
) -> dict:
    """Which other COCO categories share this concept's images, and how often.

    Read off the object annotations already attached to each group. A concept
    whose every image also carries one ubiquitous category (``person``, in
    COCO) cannot be separated from it by a contrast over those images, so the
    ranking penalises it.
    """
    counts: dict[str, int] = {}
    total = 0
    for image_id in image_ids:
        groups = by_image.get(image_id) or []
        labels = {
            str(label).casefold()
            for group in groups
            for label in (group.get("concept_annotations") or [])
        }
        labels.discard(concept.casefold())
        if not groups:
            continue
        total += 1
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
    if not total:
        return {
            "n_images": 0,
            "top": [],
            "max_fraction": 0.0,
            "dominant_category": None,
        }
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    top = [
        {"category": name, "n_images": count, "fraction": round(count / total, 6)}
        for name, count in ranked[:top_n]
    ]
    return {
        "n_images": total,
        "top": top,
        "max_fraction": top[0]["fraction"] if top else 0.0,
        "dominant_category": top[0]["category"] if top else None,
    }


def rank_concepts(
    groups: Sequence[Mapping],
    concepts: Mapping[str, Sequence[str]],
    *,
    requirements: ConceptRequirements | None = None,
    groups_per_concept: int = 6,
    max_groups_per_image: int = 2,
    seed: str = "spokencoco-pilot",
    evidence_config: EvidenceConfig | None = None,
    profile: SubsetProfile | None = None,
    evidence_index: EvidenceIndex | None = None,
) -> list[dict]:
    """Score every candidate concept against what the split will really yield.

    Counting is done over **valid synchronized positives only**: a group is a
    positive for a concept when its image carries the COCO object annotation
    *and* its written caption carries the concept or an approved synonym. Every
    row also reports the two components separately (``n_annotated_images``,
    ``n_caption_evidence_groups``) so a concept short on written evidence is
    visible as such rather than as a bare shortfall.

    Ordering is **feasibility first**, then the deterministic
    :func:`concept_score`, then distinct images, groups and name as tiebreaks.
    Ranking by raw frequency would put COCO's most common category first
    regardless of whether it can discriminate anything; the score weighs
    coverage against speaker diversity, split headroom, clean negatives,
    lexical precision, ambiguity and co-occurrence independence instead.

    The scientific minimums in ``requirements`` are applied *before* the score
    and are never relaxed by it: a high-scoring concept that misses a threshold
    is still infeasible.

    Each row carries ``unmet``: the named requirements it failed, so a
    DATASET NO-GO can say exactly what was short.
    """
    requirements = requirements or ConceptRequirements()
    config = evidence_config or config_for_concepts(concepts)
    index = (
        evidence_index.restrict(tuple(concepts))
        if evidence_index is not None
        else build_evidence_index(groups, tuple(concepts), config)
    )
    by_image = index.by_image
    matched_concepts = index.valid_concepts_by_image
    evidence_concepts = index.evidence_concepts_by_image
    # A negative must carry *neither* kind of evidence for *any* concept, so a
    # visual-only cat picture is excluded from the negatives as well as from
    # the positives rather than quietly becoming a contrast example.
    negative_images = [
        image_id for image_id, evidence in evidence_concepts.items() if not evidence
    ]
    n_negative_groups = sum(
        len(by_image[image_id][:max_groups_per_image]) for image_id in negative_images
    )

    rows: list[dict] = []
    for concept in sorted(concepts):
        # Only images that are a valid positive for this concept *alone* are
        # usable: an image matching two candidate concepts is dropped by
        # build_subset, so counting it here would promise groups the split will
        # not deliver.
        pure = [
            image_id
            for image_id, matched in matched_concepts.items()
            if matched == {concept}
        ]
        concept_groups = [g for image_id in pure for g in by_image[image_id]]
        speakers = {g["speaker"] for g in concept_groups if g.get("speaker")}
        annotated_images = index.annotated_images_by_concept[concept]
        caption_group_ids = index.caption_group_ids_by_concept[concept]
        valid_group_ids = index.valid_group_ids_by_concept[concept]
        plan = _split_plan(
            pure,
            by_image,
            concept=concept,
            groups_per_concept=groups_per_concept,
            max_groups_per_image=max_groups_per_image,
            seed=seed,
            evidence_config=config,
            profile=profile,
            valid_group_ids=valid_group_ids,
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
        # When a concept is short, say which half of the evidence rule was the
        # binding constraint. That is the whole diagnosis this repair exists for.
        if unmet and len(annotated_images) > len(pure) and not caption_group_ids:
            evidence_gap = "no written-caption evidence for any annotated image"
        elif unmet and len(annotated_images) > len(pure):
            evidence_gap = (
                f"{len(annotated_images) - len(pure)} annotated image(s) lack "
                "written-caption evidence"
            )
        elif unmet and caption_group_ids and not annotated_images:
            evidence_gap = "caption evidence without any COCO object annotation"
        else:
            evidence_gap = ""
        # Lexical precision: of the groups whose caption fires this concept's
        # terms, how many does the COCO annotation actually back? A term that
        # matches captions of images without the object is imprecise, and this
        # is the only precision signal available without hand labelling.
        lexical_precision = (
            round(len(valid_group_ids) / len(caption_group_ids), 6)
            if caption_group_ids
            else 0.0
        )
        cooccurrence = cooccurrence_statistics(pure, by_image, concept)
        row = {
            "concept": concept,
            "n_distinct_images": len(pure),
            "n_groups": len(concept_groups),
            "n_speakers": len(speakers),
            "n_annotated_images": len(annotated_images),
            "n_caption_evidence_groups": len(caption_group_ids),
            "n_valid_synchronized_groups": len(valid_group_ids),
            "annotation_source": "coco_object_annotation_and_caption_lexicon",
            "evidence_rule": "visual_annotation_AND_caption_lexicon",
            "evidence_lexicon_hash": config.lexicon_hash,
            "lexical_ambiguity": config.ambiguity_of(concept),
            "lexical_terms": list(config.terms_for(concept)),
            "lexical_exclusions": list(config.exclusions_for(concept)),
            "lexical_precision": lexical_precision,
            "eligible_train_positives": plan["n_train_positives"],
            "eligible_held_out_positives": plan["n_test_positives"],
            "matched_negatives": n_negative_groups,
            "train_headroom": plan["n_train_positives"] - requirements.min_train_positives,
            "test_headroom": plan["n_test_positives"] - requirements.min_test_positives,
            "split_feasible": not any(item.startswith(("distinct_images", "synchronized_groups", "train_positives", "test_positives")) for item in unmet),
            "n_negative_groups": n_negative_groups,
            "n_negative_images": len(negative_images),
            "cooccurrence": cooccurrence,
            "max_cooccurrence_fraction": cooccurrence["max_fraction"],
            "dominant_cooccurring_category": cooccurrence["dominant_category"],
            **plan,
            "feasible": not unmet,
            "unmet": unmet,
            "evidence_gap": evidence_gap,
            "rejection_reason": "; ".join([*unmet, evidence_gap] if evidence_gap else unmet),
        }
        row["ranking"] = concept_score(row)
        row["score"] = row["ranking"]["score"]
        rows.append(row)
    # Feasibility first — the scientific minimums are a gate, not a term in the
    # score — then the documented score, then coverage, then the name, so the
    # order is total and reproducible.
    rows.sort(
        key=lambda row: (
            not row["feasible"],
            -row["score"],
            -row["n_distinct_images"],
            -row["n_groups_selected"],
            row["concept"],
        )
    )
    for position, row in enumerate(rows, start=1):
        row["rank"] = position
    return rows


def format_ranking_table(rows: Sequence[Mapping], *, limit: int | None = None) -> str:
    """The complete ranked coverage table, printed before any concept is chosen.

    Columns: ``images`` and ``groups`` count only what satisfies **both** halves
    of the evidence rule, while ``annot`` and ``capt`` show the two halves
    separately, so ``annot >> images`` is visible as "the pictures exist but the
    captions do not name them". ``prec`` is lexical precision, ``coocc`` the
    largest single co-occurring-category fraction, ``amb`` the lexical ambiguity
    status, and ``score`` the deterministic ranking score from
    :func:`concept_score`.

    Rejected concepts are included with their reasons — that is the point of
    printing the whole table rather than the survivors.
    """
    header = (
        f"{'#':>3s} {'concept':16s} {'score':>6s} {'images':>6s} {'groups':>6s} "
        f"{'annot':>6s} {'capt':>5s} {'spk':>4s} {'train+':>6s} {'test+':>5s} "
        f"{'neg':>4s} {'prec':>5s} {'coocc':>6s} {'amb':<20s} {'split':>5s} {'ok':>2s} rejection"
    )
    lines = [header, "-" * len(header)]
    shown = rows if limit is None else rows[:limit]
    for row in shown:
        lines.append(
            f"{row.get('rank', 0):3d} {row['concept'][:16]:16s} "
            f"{row.get('score', 0.0):6.3f} {row['n_distinct_images']:6d} "
            f"{row['n_groups_selected']:6d} {row.get('n_annotated_images', 0):6d} "
            f"{row.get('n_caption_evidence_groups', 0):5d} {row['n_speakers']:4d} "
            f"{row['n_train_positives']:6d} {row['n_test_positives']:5d} "
            f"{row['n_negative_groups']:4d} {row.get('lexical_precision', 0.0):5.2f} "
            f"{row.get('max_cooccurrence_fraction', 0.0):6.2f} "
            f"{row.get('lexical_ambiguity', 'clean')[:20]:<20s} "
            f"{'yes' if row['split_feasible'] else 'no':>5s} "
            f"{'ok' if row['feasible'] else 'NO':>2s} "
            f"{row.get('rejection_reason') or '; '.join(row['unmet'])}"
        )
    if limit is not None and len(rows) > limit:
        lines.append(f"... and {len(rows) - limit} more rows (all infeasible)")
    return "\n".join(lines)


def select_concepts(
    rows: Sequence[Mapping],
    *,
    n_concepts: int = 2,
    max_concepts: int | None = None,
    requirements: ConceptRequirements | None = None,
    total_synchronized_records: int | None = None,
    coverage_cause: str = "insufficient valid synchronized records after metadata join and media validation",
) -> list[str]:
    """Feasible concepts, or a DATASET NO-GO.

    ``rows`` must already be ordered by :func:`rank_concepts`, so "the feasible
    ones, in order" is the documented deterministic ranking and not an accident
    of dictionary iteration.

    Args:
        n_concepts: The **minimum** number of feasible concepts required. Fewer
            than this is a DATASET NO-GO.
        max_concepts: How many to return when more are feasible. Defaults to
            ``n_concepts``. The pilot screens every discovered COCO category and
            takes the top two feasible ones.

    Raises:
        DatasetCoverageError: If fewer than ``n_concepts`` concepts clear the
            requirements. The message carries the ranked table and says plainly
            that the fix is more local data, not smaller thresholds — lowering
            them here would change what the pilot's GO means without saying so.
    """
    requirements = requirements or ConceptRequirements()
    limit = n_concepts if max_concepts is None else max(max_concepts, n_concepts)
    feasible = [row["concept"] for row in rows if row["feasible"]]
    if len(feasible) < n_concepts:
        raise DatasetCoverageError(
            f"DATASET NO-GO: {len(feasible)} concept(s) meet the pilot's coverage "
            f"requirements, {n_concepts} needed.\n\n"
            f"Total synchronized records found: {total_synchronized_records if total_synchronized_records is not None else 'see table'}\n"
            f"Coverage cause: {coverage_cause}\n"
            "Smallest additional coverage required is shown by each '<' shortfall in the table; at minimum two concepts must each reach 6 images/groups, 4 train positives and 2 held-out positives, with 6 negatives.\n\n"

            f"{format_ranking_table(rows, limit=25)}\n\n"
            f"Requirements applied: {requirements.to_dict()}\n\n"
            "A positive requires BOTH the COCO object annotation AND the "
            "concept (or an approved synonym) in the written caption. Where "
            "'annot' exceeds 'images' in the table, the images exist but their "
            "captions do not name the concept — those groups carry visual but "
            "not written evidence and are not valid synchronized positives.\n"
            "The local Drive dataset does not hold enough synchronized "
            "image/caption/audio groups for these concepts. Add more local "
            "SpokenCOCO records (images and their recordings both present on "
            "disk) and re-run.\n"
            "These thresholds are NOT lowered automatically: a smaller split "
            "would change what a GO verdict means. To validate plumbing only, "
            "set TINY_SMOKE=True, whose result is explicitly not scientific "
            "evidence and never feeds the research verdict."
        )
    return feasible[:limit]
