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

import errno
import hashlib
import json
import os
import re
import stat as stat_module
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from jlens.mmpilot.evidence import (
    CONCEPT_LEXICON,
    EvidenceConfig,
    config_for_concepts,
    group_evidence,
    has_any_evidence,
    is_valid_positive,
    negative_evidence,
)

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


class MediaRootError(SynchronizationError):
    """The configured media roots cannot resolve the manifest's paths.

    A subclass of :class:`SynchronizationError` because it is one specific,
    common cause of that failure and callers already handle the base class.
    """


class MediaIOError(RuntimeError):
    """The filesystem failed on a media path that is not known to be absent.

    Deliberately *not* a :class:`SynchronizationError`: a mount that stopped
    answering is an environment problem, not evidence about the dataset. It
    must never be recorded as a missing file, because that would quietly shrink
    the pilot's subset and change what the experiment measured.
    """


#: Errnos treated as "the mount hiccuped, ask again". ``EIO`` is what a Colab
#: Drive mount raises when its FUSE layer times out mid-session; ``ESTALE`` is
#: the classic network-filesystem handle expiry. None of these mean absent.
TRANSIENT_ERRNOS = frozenset(
    {
        errno.EIO,
        errno.ESTALE,
        errno.EAGAIN,
        errno.EBUSY,
        errno.EINTR,
        errno.ETIMEDOUT,
        errno.ENETDOWN,
        errno.ENETUNREACH,
    }
)

#: Errnos that mean the path genuinely is not there.
ABSENT_ERRNOS = frozenset({errno.ENOENT, errno.ENOTDIR, errno.ENAMETOOLONG})

DEFAULT_PROBE_ATTEMPTS = 4
DEFAULT_PROBE_DELAY = 0.05
MAX_PROBE_DELAY = 2.0


def _stat(path: Path) -> os.stat_result:
    """Indirection so tests can inject filesystem failures deterministically."""
    return os.stat(path)


def _sleep(seconds: float) -> None:
    """Indirection so tests can assert the backoff without waiting for it."""
    time.sleep(seconds)


def probe_path(
    path: Path,
    *,
    root: Path | str | None = None,
    attempts: int = DEFAULT_PROBE_ATTEMPTS,
    initial_delay: float = DEFAULT_PROBE_DELAY,
    journal: list | None = None,
) -> os.stat_result | None:
    """``os.stat`` with bounded retries, or ``None`` when the path is absent.

    Google Drive mounts in Colab intermittently fail a ``stat`` with
    ``OSError: [Errno 5] Input/output error`` on a path that exists and reads
    fine a moment later. Treating that as "file missing" would silently drop
    real samples, so transient errnos are retried with exponential backoff and
    anything still failing becomes a :class:`MediaIOError`.

    Args:
        attempts: Total tries, including the first. Default 4 — roughly
            0.35 s of waiting in the worst case, which covers the observed
            hiccups without stalling an audit over thousands of paths.
        journal: If given, one dict is appended per *retried* failure, so the
            audit can report how flaky the mount was.

    Returns:
        The ``stat_result`` for an existing path, or ``None`` if it is absent.

    Raises:
        MediaIOError: On a permission error (never retried — waiting cannot
            fix it), on an unrecognised ``OSError`` (refusing to guess that it
            means absent), or when the retries for a transient errno run out.
    """
    observed: list[dict] = []
    delay = initial_delay
    for attempt in range(1, max(1, attempts) + 1):
        try:
            return _stat(path)
        except FileNotFoundError:
            return None
        except NotADirectoryError:
            return None
        except PermissionError as exc:
            raise MediaIOError(
                _io_message(
                    path,
                    root,
                    attempts=attempt,
                    observed=[{"errno": exc.errno, "error": str(exc)}],
                    reason="permission denied",
                    advice=(
                        "This is not a missing file. Check the file's permissions, "
                        "and that Drive is mounted with access to this folder."
                    ),
                )
            ) from exc
        except OSError as exc:
            if exc.errno in ABSENT_ERRNOS:
                return None
            entry = {
                "attempt": attempt,
                "path": str(path),
                "root": str(root) if root else None,
                "errno": exc.errno,
                "errno_name": errno.errorcode.get(exc.errno, "unknown"),
                "error": str(exc),
            }
            observed.append(entry)
            if exc.errno not in TRANSIENT_ERRNOS:
                raise MediaIOError(
                    _io_message(
                        path,
                        root,
                        attempts=attempt,
                        observed=observed,
                        reason=f"unrecognised filesystem error {entry['errno_name']}",
                        advice=(
                            "Refusing to record this path as missing on the strength "
                            "of an error that is not known to mean absent."
                        ),
                    )
                ) from exc
            if attempt >= attempts:
                raise MediaIOError(
                    _io_message(
                        path,
                        root,
                        attempts=attempt,
                        observed=observed,
                        reason=f"transient {entry['errno_name']} did not clear",
                        advice=(
                            "Remount Google Drive and re-run: in Colab, "
                            "drive.flush_and_unmount() then drive.mount('/content/drive', "
                            "force_remount=True). The run directory resumes, so "
                            "completed work is not repeated."
                        ),
                    )
                ) from exc
            if journal is not None:
                journal.append(entry)
            _sleep(min(delay, MAX_PROBE_DELAY))
            delay = min(delay * 2, MAX_PROBE_DELAY)
    return None  # pragma: no cover - loop either returns or raises


def _io_message(path, root, *, attempts: int, observed: Sequence[Mapping], reason: str,
                advice: str) -> str:
    seen = sorted({str(entry.get("errno_name") or entry.get("errno")) for entry in observed})
    return (
        f"filesystem probe failed for {path}\n"
        f"  configured root: {root or '<none>'}\n"
        f"  attempts:        {attempts}\n"
        f"  errno observed:  {', '.join(seen) or 'unknown'}\n"
        f"  reason:          {reason}\n"
        f"{advice}"
    )


def safe_is_file(path: Path, **kwargs) -> bool:
    """Whether ``path`` is a regular file, retrying transient mount failures."""
    result = probe_path(path, **kwargs)
    return result is not None and stat_module.S_ISREG(result.st_mode)


def safe_is_dir(path: Path, **kwargs) -> bool:
    """Whether ``path`` is a directory, retrying transient mount failures."""
    result = probe_path(path, **kwargs)
    return result is not None and stat_module.S_ISDIR(result.st_mode)


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


@dataclass(frozen=True)
class MediaResolution:
    """Where one manifest-relative media path was found, and under which root.

    Attributes:
        relative: The path as the manifest wrote it.
        resolved: Absolute path on disk, or ``None`` if nothing matched.
        root: The root that produced ``resolved``.
        mode: ``"absolute"``, ``"exact"`` (root + the manifest's relative path),
            or ``"basename"`` (the fallback for manifests carrying a prefix the
            layout does not have).
        candidates: Every distinct file that matched, in root priority order.
        ambiguous: Two or more roots matched with *different* file sizes, so
            which one the manifest meant cannot be decided.
    """

    relative: str
    resolved: str | None
    root: str | None
    mode: str | None
    candidates: tuple[str, ...] = ()
    ambiguous: bool = False


def _unique_existing(
    probes: Sequence[tuple[Path, Path | None]], journal: list | None
) -> list[tuple[Path, os.stat_result]]:
    """Distinct existing files with their stat results, in root priority order.

    De-duplication happens on the normalized absolute path *before* any retry
    bookkeeping is consulted, so a path that needed three attempts still counts
    exactly once — a retry must never turn one file into two candidates and
    trip the ambiguity check.
    """
    seen: set[str] = set()
    out: list[tuple[Path, os.stat_result]] = []
    for probe, root in probes:
        key = os.path.normcase(os.path.abspath(str(probe)))
        if key in seen:
            continue
        result = probe_path(probe, root=root, journal=journal)
        if result is None or not stat_module.S_ISREG(result.st_mode):
            continue
        seen.add(key)
        out.append((probe, result))
    return out


def _resolve_one(
    relative: str, roots: Sequence[Path], *, journal: list | None = None
) -> MediaResolution:
    """Resolve one relative media path against ``roots``, in priority order.

    Exact ``root / relative`` joins are tried first across *every* root, then
    the basename fallback. Trying every root rather than stopping at the first
    hit is what makes ambiguity detectable: when two roots hold different files
    for the same manifest path, neither can be assumed correct.

    Two roots holding the *same-sized* file (a download cache mirroring the
    dataset, say) is not ambiguity — the first root by configured priority wins
    and both are recorded in ``candidates``.
    """
    candidate = Path(relative)
    if candidate.is_absolute():
        exists = safe_is_file(candidate, journal=journal)
        return MediaResolution(
            relative=relative,
            resolved=str(candidate) if exists else None,
            root=None,
            mode="absolute" if exists else None,
            candidates=(str(candidate),) if exists else (),
        )

    matches = _unique_existing([(root / candidate, root) for root in roots], journal)
    mode = "exact"
    if not matches:
        matches = _unique_existing(
            [(root / candidate.name, root) for root in roots], journal
        )
        mode = "basename" if matches else None
    if not matches:
        return MediaResolution(relative, None, None, None)

    # Sizes come from the stat results the probe already fetched, so ambiguity
    # detection costs no extra filesystem round trips on a flaky mount.
    if len({result.st_size for _, result in matches}) > 1:
        return MediaResolution(
            relative=relative,
            resolved=None,
            root=None,
            mode=mode,
            candidates=tuple(str(probe) for probe, _ in matches),
            ambiguous=True,
        )

    winner = matches[0][0]
    owning_root = next(
        (
            str(root)
            for root in roots
            if os.path.normcase(os.path.abspath(str(winner))).startswith(
                os.path.normcase(os.path.abspath(str(root)))
            )
        ),
        None,
    )
    return MediaResolution(
        relative=relative,
        resolved=str(winner),
        root=owning_root,
        mode=mode,
        candidates=tuple(str(probe) for probe, _ in matches),
    )


def _resolve_media(relative: str, roots: Sequence[Path]) -> str | None:
    """Backward-compatible thin wrapper returning just the resolved path."""
    return _resolve_one(relative, roots).resolved


def resolve_media_roots(
    *,
    media_roots: Sequence[str | os.PathLike[str]] | None = None,
    image_roots: Sequence[str | os.PathLike[str]] | None = None,
    audio_roots: Sequence[str | os.PathLike[str]] | None = None,
    require_existing: bool = True,
) -> dict[str, list[Path]]:
    """Normalize root configuration into ``{"image": [...], "audio": [...]}``.

    Images and audio may live under different sibling roots — SpokenCOCO ships
    COCO images under ``coco/`` and recordings under ``SpokenCOCO/`` — so the
    two roles are resolved independently. Passing only ``media_roots`` keeps
    the older single-root behaviour: both roles use that list.

    Non-existent roots are dropped (a download cache that was never created is
    not an error), but a role left with no existing root at all is.

    Raises:
        MediaRootError: If a role ends up with no usable root.
    """
    if media_roots is None and image_roots is None and audio_roots is None:
        raise MediaRootError(
            "no media roots given; pass media_roots, or image_roots/audio_roots"
        )
    fallback = list(media_roots or [])
    per_role = {
        "image": list(image_roots if image_roots is not None else fallback),
        "audio": list(audio_roots if audio_roots is not None else fallback),
    }
    out: dict[str, list[Path]] = {}
    for role, configured in per_role.items():
        paths = [Path(root) for root in configured]
        usable = (
            [path for path in paths if safe_is_dir(path, root=path)]
            if require_existing
            else paths
        )
        if not usable:
            raise MediaRootError(
                f"no existing directory among the {role} roots "
                f"{[str(path) for path in paths]}; check the configured paths"
            )
        out[role] = usable
    return out


def _role_fields(schema: ManifestSchema) -> dict[str, str]:
    return {"image": schema.image_field, "audio": schema.audio_field}


def audit_media_roots(
    payload,
    schema: ManifestSchema,
    roots: Mapping[str, Sequence[Path]],
    *,
    expected_roots: Mapping[str, str | os.PathLike[str]] | None = None,
    n_samples: int = 8,
) -> dict:
    """Probe representative manifest paths and report which root resolves each.

    Runs *before* normalization so a root misconfiguration is diagnosed against
    a handful of paths with the evidence printed, rather than showing up as
    "96 of 96 media references did not resolve".

    Args:
        expected_roots: ``{"image": path, "audio": path}`` — the roots the
            configuration intends for each role. Supplying them enables the
            swapped-roots check.

    Raises:
        MediaRootError: If a role resolves nothing, if any sampled path is
            ambiguous, or if the image and audio roots appear to be swapped.
        MediaIOError: If the mount kept failing on a path that is not known to
            be absent — reported with the path, root, retry count and errno
            rather than as a bare pathlib traceback.
    """
    _, records = (
        _find_records(payload)
        if schema.records_key is None
        else (schema.records_key, [r for r in payload[schema.records_key] if isinstance(r, Mapping)])
    )
    rows = _flatten(records, schema)
    report: dict = {
        "roots": {role: [str(path) for path in paths] for role, paths in roots.items()},
        "samples": {},
        "resolution_by_root": {},
        "unresolved": {},
        "ambiguous": {},
        "n_sampled": min(n_samples, len(rows)),
    }
    problems: list[str] = []
    winners: dict[str, str | None] = {}

    journal: list[dict] = []
    for role, field_name in _role_fields(schema).items():
        resolutions: list[MediaResolution] = []
        for row in rows[:n_samples]:
            relative = str(row.get(field_name, "") or "")
            if not relative:
                continue
            try:
                resolutions.append(_resolve_one(relative, roots[role], journal=journal))
            except MediaIOError as exc:
                raise MediaIOError(
                    f"the {role} media probe failed while auditing "
                    f"{relative!r}.\n{exc}"
                ) from exc
        counts: dict[str, int] = {}
        for resolution in resolutions:
            if resolution.root:
                counts[resolution.root] = counts.get(resolution.root, 0) + 1
        report["samples"][role] = [
            {
                "relative": r.relative,
                "resolved": r.resolved,
                "root": r.root,
                "mode": r.mode,
                "candidates": list(r.candidates),
            }
            for r in resolutions
        ]
        report["resolution_by_root"][role] = counts
        report["unresolved"][role] = [
            r.relative for r in resolutions if r.resolved is None and not r.ambiguous
        ]
        report["ambiguous"][role] = [
            {"relative": r.relative, "candidates": list(r.candidates)}
            for r in resolutions
            if r.ambiguous
        ]
        winners[role] = max(counts, key=lambda key: (counts[key], key)) if counts else None

        if resolutions and not counts and not report["ambiguous"][role]:
            problems.append(
                f"no configured {role} root resolves any sampled {role} path. "
                f"Tried {[str(path) for path in roots[role]]}; examples: "
                f"{[r.relative for r in resolutions[:3]]}"
            )
        if report["ambiguous"][role]:
            problems.append(
                f"{len(report['ambiguous'][role])} sampled {role} path(s) resolve "
                f"under several roots to files of different sizes, e.g. "
                f"{report['ambiguous'][role][0]}"
            )

    if expected_roots and winners.get("image") and winners.get("audio"):
        def same(a, b) -> bool:
            return os.path.normcase(os.path.abspath(str(a))) == os.path.normcase(
                os.path.abspath(str(b))
            )

        expected_image = expected_roots.get("image")
        expected_audio = expected_roots.get("audio")
        if (
            expected_image
            and expected_audio
            and not same(expected_image, expected_audio)
            and same(winners["image"], expected_audio)
            and same(winners["audio"], expected_image)
        ):
            problems.append(
                f"the image and audio roots look swapped: image paths resolve "
                f"under {expected_audio} and audio paths under {expected_image}. "
                "Exchange IMAGE_MEDIA_ROOT and AUDIO_MEDIA_ROOT."
            )

    report["winning_root"] = winners
    report["transient_io_retries"] = len(journal)
    report["transient_io_examples"] = journal[:5]
    report["ok"] = not problems
    report["problems"] = problems
    if problems:
        raise MediaRootError(
            "media-root audit failed:\n  - "
            + "\n  - ".join(problems)
            + f"\n\nSampled resolutions: {json.dumps(report['samples'], indent=2)}"
        )
    return report


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
    source_checksum: str,
    media_roots: Sequence[str | os.PathLike[str]] | None = None,
    image_roots: Sequence[str | os.PathLike[str]] | None = None,
    audio_roots: Sequence[str | os.PathLike[str]] | None = None,
    max_missing_examples: int = 20,
    min_complete_groups: int = 24,
    max_missing_fraction: float = 0.5,
) -> NormalizedManifest:
    """Build synchronized groups with resolved absolute media paths.

    Args:
        payload: Parsed manifest (read-only).
        schema: Mapping from :func:`inspect_manifest`.
        media_roots: Roots applied to *both* modalities, in priority order.
            Enough for a layout that keeps images and audio under one tree.
        image_roots / audio_roots: Per-role roots, for layouts that do not —
            SpokenCOCO puts COCO images under ``coco/`` and recordings under
            ``SpokenCOCO/``. Either one falls back to ``media_roots`` when not
            given, so the older single-root call still works.
        source_checksum: ``sha256:`` of the original manifest file.
        max_missing_examples: How many missing-file examples to keep in the audit.
        min_complete_groups: Refuse below this many fully synchronized groups.
        max_missing_fraction: Refuse when more than this fraction of rows lose
            a media file — that means the roots are wrong, not that the data
            is slightly incomplete.

    Raises:
        MediaRootError: If a role has no usable root, or if a media path is
            ambiguous between roots holding different files.
        SynchronizationError: When the audit shows synchronization cannot be
            trusted. Nothing downstream runs on a guessed correspondence.
    """
    roots = resolve_media_roots(
        media_roots=media_roots, image_roots=image_roots, audio_roots=audio_roots
    )
    records_key, records = _find_records(payload) if schema.records_key is None else (
        schema.records_key,
        [r for r in payload[schema.records_key] if isinstance(r, Mapping)],
    )
    del records_key
    rows = _flatten(records, schema)

    groups: list[dict] = []
    missing_image: list[str] = []
    missing_audio: list[str] = []
    ambiguous: list[dict] = []
    roots_used: dict[str, dict[str, int]] = {"image": {}, "audio": {}}
    io_journal: list[dict] = []
    seen: dict[str, int] = {}
    duplicates: list[str] = []
    for index, row in enumerate(rows):
        raw_image = str(row.get(schema.image_field, "") or "")
        raw_audio = str(row.get(schema.audio_field, "") or "")
        caption = str(row.get(schema.caption_field, "") or "").strip()
        image = (
            _resolve_one(raw_image, roots["image"], journal=io_journal)
            if raw_image
            else None
        )
        audio = (
            _resolve_one(raw_audio, roots["audio"], journal=io_journal)
            if raw_audio
            else None
        )
        image_path = image.resolved if image else None
        audio_path = audio.resolved if audio else None
        for role, resolution in (("image", image), ("audio", audio)):
            if resolution is None:
                continue
            if resolution.ambiguous:
                ambiguous.append(
                    {
                        "role": role,
                        "relative": resolution.relative,
                        "candidates": list(resolution.candidates),
                    }
                )
            elif resolution.root:
                roots_used[role][resolution.root] = (
                    roots_used[role].get(resolution.root, 0) + 1
                )
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
        "media_roots": {role: [str(path) for path in paths] for role, paths in roots.items()},
        "resolved_by_root": roots_used,
        "n_ambiguous_media": len(ambiguous),
        "ambiguous_media_examples": ambiguous[:max_missing_examples],
        "n_transient_io_retries": len(io_journal),
        "transient_io_examples": io_journal[:max_missing_examples],
    }

    if ambiguous:
        raise MediaRootError(
            f"{len(ambiguous)} media path(s) resolve under several roots to files "
            f"of different sizes, so which one the manifest meant cannot be "
            f"decided. Remove or reorder the overlapping root. Examples: "
            f"{json.dumps(ambiguous[:3], indent=2)}"
        )

    # Root-level failures are diagnosed first: "the paths are wrong" is a much
    # more useful message than "too few groups", and it is the usual cause.
    #
    # The fraction is evaluated **per role**. One modality losing every file
    # while the other resolves cleanly is the signature of sibling roots, and
    # it would hide behind a combined ratio: 72 missing images out of 144 total
    # references is exactly half, under any sensible global threshold, yet the
    # run is completely broken.
    failed_roles = []
    for role, missing in (("image", missing_image), ("audio", missing_audio)):
        denominator = max(1, len(rows))
        if len(missing) / denominator > max_missing_fraction:
            failed_roles.append((role, len(missing), denominator))
    if failed_roles:
        detail = ", ".join(
            f"{count} of {total} {role} references" for role, count, total in failed_roles
        )
        raise SynchronizationError(
            f"{detail} did not resolve.\n"
            f"  image roots: {[str(p) for p in roots['image']]}\n"
            f"  audio roots: {[str(p) for p in roots['audio']]}\n"
            "Images and audio may live under different sibling roots (SpokenCOCO "
            "keeps images under coco/ and recordings under SpokenCOCO/) — pass "
            "image_roots and audio_roots separately, and run audit_media_roots "
            f"first to see which root resolves what. Audit: {json.dumps(audit, indent=2)}"
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
#:
#: One lexicon, defined in :mod:`jlens.mmpilot.evidence` and aliased here. Two
#: copies would let selection and the audit drift apart, which is precisely the
#: class of bug this module is being repaired for.
DEFAULT_CONCEPTS: dict[str, tuple[str, ...]] = dict(CONCEPT_LEXICON)


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
    evidence_config: EvidenceConfig | None = None,
) -> dict:
    """Deterministically select a small, image-disjoint pilot subset.

    A positive requires **both** kinds of evidence — the COCO object annotation
    *and* the concept (or an approved synonym) in the written caption. The
    earlier rule short-circuited on the annotation and never read the caption,
    which selected images of a cat whose caption never says "cat" and then
    marked the text arm wrong for not seeing one. See
    :mod:`jlens.mmpilot.evidence`.

    Matched negatives are stricter still: an image is only a negative when it
    carries *neither* kind of evidence for *any* screened concept.

    Each image is assigned to at most one concept: an image that is a valid
    positive for two selected concepts is dropped rather than counted twice.
    Every group belonging to one image lands in the same split, so the split is
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

    config = evidence_config or config_for_concepts(concepts)

    by_image: dict[str, list[dict]] = {}
    for group in groups:
        by_image.setdefault(group["image_id"], []).append(dict(group))

    # Two different questions, deliberately kept apart:
    #   image_concepts  — what this image is a *valid positive* for.
    #   image_evidence  — what it carries *any* evidence for, which is what
    #                     disqualifies it from being a matched negative.
    image_concepts: dict[str, set[str]] = {}
    image_evidence: dict[str, set[str]] = {}
    for image_id, image_groups in by_image.items():
        image_concepts[image_id] = {
            concept
            for concept in concepts
            if any(is_valid_positive(g, concept, config) for g in image_groups)
        }
        image_evidence[image_id] = {
            concept
            for concept in concepts
            if any(has_any_evidence(g, concept, config) for g in image_groups)
        }

    selected: dict[str, dict] = {}
    used_images: set[str] = set()
    for concept in sorted(concepts):
        pure = sorted(
            (image_id for image_id, matched in image_concepts.items()
             if matched == {concept} and image_id not in used_images),
            key=lambda image_id: _stable_rank(image_id, f"{seed}|{concept}"),
        )
        source_train = [image_id for image_id in pure if any(str(g.get("source_split", "")).lower().startswith("train") for g in by_image[image_id])]
        source_test = [image_id for image_id in pure if any(str(g.get("source_split", "")).lower().startswith(("val", "test")) for g in by_image[image_id])]
        if groups_per_concept >= 6 and len(source_train) >= 4 and len(source_test) >= 2:
            chosen = source_train[:4] + source_test[:2]
            n_train = 4
            split_method = "source_provided_train_validation"
        else:
            chosen = pure[:groups_per_concept]
            n_train = min(max(1, groups_per_concept - 2), max(1, len(chosen) - 1))
            split_method = "stable_id_seeded_4_to_2"
        used_images.update(chosen)
        selected[concept] = {
            "train_images": chosen[:n_train],
            "test_images": chosen[n_train:],
            "n_available_images": len(pure),
            "split_method": split_method,
        }

    negatives = sorted(
        (image_id for image_id, evidence in image_evidence.items()
         if not evidence and image_id not in used_images),
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
                # Every selected row carries the evidence that justified it, so
                # a reviewer never has to take the label on trust.
                justification = (
                    group_evidence(group, concept, config)
                    if concept is not None
                    else negative_evidence(group, sorted(concepts), config)
                )
                out.append({**group, "concept": concept, "split": split,
                            "is_positive": concept is not None,
                            "evidence": justification,
                            "split_provenance": {
                                **(group.get("split_provenance") or {}),
                                "source_split": group.get("source_split"),
                                "assignment": split,
                                "assignment_rule": "all groups of one image share a split",
                            }})
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
                "images that are a VALID SYNCHRONIZED POSITIVE for exactly one "
                "selected concept — COCO object annotation AND an approved "
                "caption term, both required; ordered by "
                "sha256(seed|concept|image_id); first half train"
            ),
            "negative_rule": (
                "images carrying neither visual nor caption evidence for any "
                "screened concept"
            ),
            "concept_keywords": {k: list(v) for k, v in concepts.items()},
            "evidence_config": config.to_dict(),
            "evidence_lexicon_hash": config.lexicon_hash,
            "evidence_config_fingerprint": config.fingerprint,
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
