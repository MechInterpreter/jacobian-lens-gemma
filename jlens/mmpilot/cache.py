# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""A fingerprint-addressed cache for the expensive CPU metadata audit.

Discovering the COCO category universe, joining SpokenCOCO metadata against
media that is actually on disk, auditing every group against every category and
ranking the result takes a long time on a Colab CPU runtime — and it produces
the same answer every time the inputs are the same. Re-deriving it once per
session is pure waste, and the sessions where it matters are exactly the ones
where the runtime is about to be spent on an L4.

So the derived artifacts are published into Drive under a directory named for a
**fingerprint of everything they were derived from**. A later session computes
the same fingerprint from the same sources and either finds the directory (a
hit, load and go) or does not (a miss, build and publish).

What the fingerprint covers
---------------------------

Checksums of the SpokenCOCO source metadata, checksums of the COCO instance
annotation files, the original manifest's checksum, the evidence
normalization/version identifier, the discovered category universe, the lexical
specification hash, the visual-plus-caption evidence rule, the media-root layout
configuration, the scientific thresholds, and the split seed and split algorithm
version. Change any one of them and the artifacts land in a different directory:
there is nothing to invalidate, and nothing to overwrite.

The rules that keep this honest
-------------------------------

- Source manifests and annotation files are read and checksummed, never written.
- No credentials are stored, and no user-specific absolute Drive path is: media
  is recorded relative to its configured root, so the same cache is valid from
  a differently-mounted Drive.
- Artifacts are built into a **local** staging directory under ``/content``
  first, then published to a *new* fingerprint directory. ``_SUCCESS.json`` is
  written last and carries a checksum for every published file.
- A directory without a valid ``_SUCCESS.json`` is incomplete — an interrupted
  publish — and is ignored and rebuilt, never half-read.
- A directory whose stored fingerprint disagrees with the requested one is
  refused loudly. Reuse is never inferred from a filename.
- Publication is a handful of file writes, not thousands: Drive is slow per
  file, so per-record artifacts are one gzipped JSONL stream.

This is a cache, not a workflow engine. Two methods do the work:
:meth:`DerivedCache.state` says what is there, and :meth:`DerivedCache.publish`
puts something there.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from jlens.mmpilot.store import payload_checksum

CACHE_SCHEMA_VERSION = "jlens.mmpilot.derived_cache.v1"
SPLIT_ALGORITHM_VERSION = "jlens.mmpilot.build_subset.v2_visual_and_caption"

#: Where published fingerprint directories live. One level below it is a
#: directory per fingerprint; nothing else is written there.
DEFAULT_CACHE_ROOT = (
    "/content/drive/MyDrive/datasets/cstf_spokencoco_derived/jlens_mmpilot_v1"
)

#: Local staging root. Artifacts are assembled here and only copied to Drive
#: once every one of them exists.
DEFAULT_STAGING_ROOT = "/content/jlens_mmpilot_cache_staging"

#: The marker that makes a directory readable. Written last, always.
SUCCESS_NAME = "_SUCCESS.json"

#: The derived artifacts, in the order they are built. Small JSON documents
#: plus one gzipped stream for the per-record index — the minimum needed to
#: skip the audit, not a general-purpose dataset format.
ARTIFACT_NAMES = (
    "metadata.json",
    "concept_coverage.json",
    "concept_evidence_index.jsonl.gz",
    "rejected_evidence_counts.json",
    "selected_concepts.json",
    "pilot_subset.json",
    "split_provenance.json",
)

#: Cache states, printed verbatim so a session's log says what happened.
STATE_HIT = "hit"
STATE_MISS = "miss"
STATE_INCOMPLETE = "incomplete"
STATE_INCOMPATIBLE = "incompatible"


class CacheError(RuntimeError):
    """Base class for every refusal this module makes."""


class IncompatibleCacheError(CacheError):
    """A published directory was derived under a different fingerprint."""


class CorruptCacheError(CacheError):
    """A published artifact failed its checksum or is missing."""


# ------------------------------------------------------------------- paths


def file_sha256(path: str | os.PathLike[str]) -> str:
    """``sha256:`` of a file, streamed so a large annotation file is cheap."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def relative_to_roots(
    path: str | os.PathLike[str], roots: Sequence[str | os.PathLike[str]]
) -> str:
    """``path`` relative to the first configured root that contains it.

    Falls back to the basename when no root matches. Absolute Drive paths are
    user-specific — ``/content/drive/MyDrive/...`` differs between accounts and
    mounts — so nothing user-specific is ever written into a cached artifact.
    """
    target = os.path.normcase(os.path.abspath(str(path)))
    best: str | None = None
    for root in roots:
        prefix = os.path.normcase(os.path.abspath(str(root)))
        if target.startswith(prefix.rstrip("\\/") + os.sep) or target == prefix:
            candidate = os.path.relpath(str(path), str(root)).replace(os.sep, "/")
            if best is None or len(candidate) < len(best):
                best = candidate
    return best if best is not None else Path(str(path)).name


def resolve_relative(relative: str, roots: Sequence[str | os.PathLike[str]]) -> str | None:
    """The first existing ``root / relative``, or ``None``.

    The inverse of :func:`relative_to_roots`, used on a cache hit to turn the
    stored relative paths back into paths on *this* session's mount.
    """
    for root in roots:
        candidate = Path(root) / relative
        if candidate.is_file():
            return str(candidate)
    return None


def _relative_checksums(
    paths: Mapping[str, str], roots: Sequence[str | os.PathLike[str]]
) -> dict[str, str]:
    """Re-key ``{absolute path: checksum}`` by a root-relative path."""
    return {
        relative_to_roots(path, roots): checksum
        for path, checksum in sorted(paths.items())
    }


# ------------------------------------------------------------- fingerprint


@dataclass(frozen=True)
class CacheFingerprint:
    """Everything the derived artifacts were derived from.

    Every field participates in :attr:`digest`. Two sessions agree on a cache
    directory exactly when they agree on all of this — which is the point: a
    changed lexicon, a re-exported annotation file, a different threshold or a
    different split seed all produce a different directory rather than a
    silently mixed one.

    Attributes:
        spokencoco_metadata_checksums: ``{root-relative path: sha256}`` for the
            SpokenCOCO annotation files the join read.
        coco_annotation_checksums: the same for ``instances_*.json``.
        original_manifest_checksum: the hand-made manifest's ``sha256:``.
        evidence_version: evidence schema plus normalization identifier.
        category_universe: the discovered COCO category names, sorted.
        lexical_spec_hash: hash of every category's lexical specification.
        evidence_rule: the visual-plus-caption rule, as a string.
        media_layout: role -> root-relative layout description. Never absolute
            Drive paths.
        thresholds: the scientific minimums, as a dict.
        split_seed / split_algorithm_version: what the split will be.
    """

    spokencoco_metadata_checksums: Mapping[str, str]
    coco_annotation_checksums: Mapping[str, str]
    original_manifest_checksum: str
    evidence_version: str
    category_universe: tuple[str, ...]
    lexical_spec_hash: str
    evidence_rule: str
    media_layout: Mapping[str, object]
    thresholds: Mapping[str, object]
    split_seed: str
    split_algorithm_version: str = SPLIT_ALGORITHM_VERSION
    schema_version: str = CACHE_SCHEMA_VERSION
    extra: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "spokencoco_metadata_checksums": dict(
                sorted(self.spokencoco_metadata_checksums.items())
            ),
            "coco_annotation_checksums": dict(
                sorted(self.coco_annotation_checksums.items())
            ),
            "original_manifest_checksum": self.original_manifest_checksum,
            "evidence_version": self.evidence_version,
            "category_universe": list(self.category_universe),
            "lexical_spec_hash": self.lexical_spec_hash,
            "evidence_rule": self.evidence_rule,
            "media_layout": dict(sorted(self.media_layout.items())),
            "thresholds": dict(sorted(self.thresholds.items())),
            "split_seed": self.split_seed,
            "split_algorithm_version": self.split_algorithm_version,
            "extra": dict(sorted(self.extra.items())),
        }

    @property
    def digest(self) -> str:
        return payload_checksum(self.to_dict())

    @property
    def short(self) -> str:
        """The directory name: a readable prefix of the digest."""
        return self.digest.split(":", 1)[1][:20]

    def differences(self, other: Mapping) -> list[str]:
        """Field-by-field differences against a stored fingerprint payload."""
        mine, theirs = self.to_dict(), dict(other)
        out: list[str] = []
        for key in sorted(set(mine) | set(theirs)):
            a, b = mine.get(key, "<absent>"), theirs.get(key, "<absent>")
            if json.dumps(a, sort_keys=True, default=str) != json.dumps(
                b, sort_keys=True, default=str
            ):
                out.append(f"{key}: stored={b!r} requested={a!r}")
        return out


def build_fingerprint(
    *,
    spokencoco_sources: Mapping[str, str],
    coco_annotation_sources: Mapping[str, str],
    original_manifest_checksum: str,
    evidence_config,
    universe,
    media_roots: Mapping[str, Sequence[str | os.PathLike[str]]],
    thresholds: Mapping[str, object],
    split_seed: str,
    extra: Mapping[str, object] | None = None,
) -> CacheFingerprint:
    """Assemble a :class:`CacheFingerprint` from the objects a session has.

    Media roots are reduced to a *layout* — how many roots each role has and
    what each root's final path component is — so two people with the same
    dataset arranged the same way under different Drive accounts share a cache,
    while a genuinely different arrangement does not.
    """
    roots = {
        role: [Path(str(root)).name for root in paths]
        for role, paths in sorted(media_roots.items())
    }
    all_roots = [root for paths in media_roots.values() for root in paths]
    return CacheFingerprint(
        spokencoco_metadata_checksums=_relative_checksums(spokencoco_sources, all_roots),
        coco_annotation_checksums=_relative_checksums(coco_annotation_sources, all_roots),
        original_manifest_checksum=str(original_manifest_checksum),
        evidence_version=(
            f"{evidence_config.to_dict()['schema_version']}|"
            f"{evidence_config.normalization}|{evidence_config.matching}"
        ),
        category_universe=tuple(universe.categories),
        lexical_spec_hash=universe.lexical_hash,
        evidence_rule=(
            "visual_annotation_AND_caption_lexicon"
            f"|visual={evidence_config.require_visual_evidence}"
            f"|caption={evidence_config.require_caption_evidence}"
            f"|lexicon={evidence_config.lexicon_hash}"
        ),
        media_layout={"roles": roots, "resolution": "per_role_ordered_roots"},
        thresholds=dict(thresholds),
        split_seed=str(split_seed),
        extra=dict(extra or {}),
    )


# ------------------------------------------------------------------ writes


def write_json(path: str | os.PathLike[str], payload) -> Path:
    """Write one JSON artifact into the staging directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return path


def write_jsonl_gz(path: str | os.PathLike[str], rows: Sequence[Mapping]) -> Path:
    """Write per-record rows as one gzipped JSONL stream.

    One file rather than one file per record: Drive charges per write, and a
    few thousand tiny files there is slower than the audit it is meant to skip.
    ``mtime=0`` keeps the bytes reproducible for the same rows.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.GzipFile(path, "wb", mtime=0) as handle:
        for row in rows:
            handle.write(
                (json.dumps(row, sort_keys=True, default=str) + "\n").encode("utf-8")
            )
    return path


def read_jsonl_gz(path: str | os.PathLike[str]) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


# ------------------------------------------------------------- the cache


class DerivedCache:
    """One fingerprint's worth of derived audit artifacts in Drive.

    Args:
        root: The cache root; a directory per fingerprint is created under it.
        fingerprint: What this session's artifacts would be derived from.
        staging_root: Local directory to assemble artifacts in before
            publishing. Defaults to :data:`DEFAULT_STAGING_ROOT`.
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        fingerprint: CacheFingerprint,
        *,
        staging_root: str | os.PathLike[str] | None = None,
    ) -> None:
        self.root = Path(root)
        self.fingerprint = fingerprint
        self.staging_root = Path(staging_root or DEFAULT_STAGING_ROOT)

    # ------------------------------------------------------------ locations

    @property
    def directory(self) -> Path:
        return self.root / self.fingerprint.short

    @property
    def success_path(self) -> Path:
        return self.directory / SUCCESS_NAME

    @property
    def staging_directory(self) -> Path:
        return self.staging_root / self.fingerprint.short

    # ---------------------------------------------------------------- state

    def _stored_success(self) -> dict | None:
        if not self.success_path.is_file():
            return None
        try:
            return json.loads(self.success_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            return None

    def state(self) -> str:
        """One of ``hit``, ``miss``, ``incomplete``, ``incompatible``.

        ``incomplete`` means the directory exists but has no readable success
        marker — an interrupted publish. It is rebuilt, never partly read.
        ``incompatible`` means the marker is there and describes a *different*
        fingerprint, which can only happen on a digest-prefix collision or a
        hand-edited directory; either way, reuse is refused rather than guessed.
        """
        if not self.directory.is_dir():
            return STATE_MISS
        success = self._stored_success()
        if success is None:
            return STATE_INCOMPLETE
        if success.get("fingerprint_digest") != self.fingerprint.digest:
            return STATE_INCOMPATIBLE
        if success.get("schema_version") != CACHE_SCHEMA_VERSION:
            return STATE_INCOMPATIBLE
        missing = [
            name
            for name in success.get("artifacts", {})
            if not (self.directory / name).is_file()
        ]
        return STATE_INCOMPLETE if missing else STATE_HIT

    def describe(self) -> str:
        """A single line naming the state and the directory, for the log."""
        state = self.state()
        prose = {
            STATE_HIT: "reusing the published derived audit",
            STATE_MISS: "no cache for this fingerprint — building it",
            STATE_INCOMPLETE: "cache directory has no valid success marker — rebuilding",
            STATE_INCOMPATIBLE: "cache directory holds a DIFFERENT fingerprint — refusing to reuse it",
        }[state]
        return f"cache {state.upper()}: {prose}\n  {self.directory}"

    # ----------------------------------------------------------------- load

    def load(self) -> dict:
        """Every artifact, after validating the fingerprint and each checksum.

        Raises:
            IncompatibleCacheError: The stored fingerprint disagrees.
            CorruptCacheError: A file is missing, or its bytes do not match the
                checksum ``_SUCCESS.json`` recorded for it. Either way the
                artifacts are not what was published and are not used.
        """
        success = self._stored_success()
        if success is None:
            raise CorruptCacheError(
                f"{self.directory} has no readable {SUCCESS_NAME}; it is an "
                "incomplete publication and cannot be reused."
            )
        if success.get("fingerprint_digest") != self.fingerprint.digest:
            diffs = "\n  ".join(
                self.fingerprint.differences(success.get("fingerprint", {}))
            )
            raise IncompatibleCacheError(
                f"{self.directory} was derived under a different fingerprint; "
                f"refusing to reuse it.\n  {diffs}"
            )
        payload: dict = {}
        for name, expected in sorted(success.get("artifacts", {}).items()):
            path = self.directory / name
            if not path.is_file():
                raise CorruptCacheError(f"{path} is listed in {SUCCESS_NAME} but absent")
            actual = file_sha256(path)
            if actual != expected.get("checksum"):
                raise CorruptCacheError(
                    f"{path} failed its checksum: stored={expected.get('checksum')} "
                    f"actual={actual}. The artifact is not what was published."
                )
            payload[name] = (
                read_jsonl_gz(path)
                if name.endswith(".jsonl.gz")
                else json.loads(path.read_text(encoding="utf-8"))
            )
        payload["_success"] = success
        return payload

    # -------------------------------------------------------------- publish

    def stage(self, artifacts: Mapping[str, object]) -> Path:
        """Write ``artifacts`` into the local staging directory.

        Args:
            artifacts: ``{filename: payload}``. ``.jsonl.gz`` names take a
                sequence of rows; everything else takes a JSON-serialisable
                document.
        """
        staging = self.staging_directory
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True, exist_ok=True)
        for name, payload in artifacts.items():
            if name.endswith(".jsonl.gz"):
                write_jsonl_gz(staging / name, payload)
            else:
                write_json(staging / name, payload)
        return staging

    def publish(self, artifacts: Mapping[str, object] | None = None) -> dict:
        """Copy the staged artifacts into a new fingerprint directory.

        ``_SUCCESS.json`` is written last and lists a checksum and a size for
        every published file, so a reader can tell a complete publication from
        an interrupted one and a correct file from a truncated one.

        Raises:
            IncompatibleCacheError: If a *different* fingerprint already owns
                this directory. An existing, compatible directory is left alone
                and reported as already published.
            CacheError: If a required artifact was never staged.
        """
        staging = self.stage(artifacts) if artifacts is not None else self.staging_directory
        missing = [name for name in ARTIFACT_NAMES if not (staging / name).is_file()]
        if missing:
            raise CacheError(
                f"refusing to publish an incomplete cache: {missing} were never "
                f"staged under {staging}"
            )
        state = self.state()
        if state == STATE_INCOMPATIBLE:
            raise IncompatibleCacheError(
                f"{self.directory} already holds artifacts under a different "
                "fingerprint; refusing to overwrite them."
            )
        if state == STATE_HIT:
            return {"published": False, "reason": "already published", **(self._stored_success() or {})}

        self.directory.mkdir(parents=True, exist_ok=True)
        # The marker goes last, so an interruption anywhere above leaves a
        # directory that reads as INCOMPLETE rather than as usable.
        self.success_path.unlink(missing_ok=True)
        entries: dict[str, dict] = {}
        for name in ARTIFACT_NAMES:
            source = staging / name
            shutil.copyfile(source, self.directory / name)
            entries[name] = {
                "checksum": file_sha256(self.directory / name),
                "size_bytes": (self.directory / name).stat().st_size,
            }
        success = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "fingerprint_digest": self.fingerprint.digest,
            "fingerprint": self.fingerprint.to_dict(),
            "artifacts": entries,
            "source_metadata_mutated": False,
            "media_redownloaded": False,
            "audio_transcribed": False,
            "credentials_stored": False,
            "paths_are_relative_to_configured_roots": True,
        }
        write_json(self.success_path, success)
        return {"published": True, **success}

    # ---------------------------------------------------------- maintenance

    def probe_media(
        self,
        rows: Sequence[Mapping],
        roots: Mapping[str, Sequence[str | os.PathLike[str]]],
        *,
        n_probes: int = 6,
    ) -> dict:
        """Check that a handful of cached media references still resolve.

        A cache hit skips the full media validation, so this is the cheap
        replacement for it: enough probes to catch an unmounted Drive or a
        moved dataset, few enough that a hit stays fast. It is deliberately
        **not** a revalidation — a hit asserts that the source checksums match,
        and those already describe the media the audit accepted.
        """
        checked: list[dict] = []
        for row in list(rows)[:n_probes]:
            entry = {"group_id": row.get("group_id")}
            for role, key in (("image", "image_relpath"), ("audio", "audio_relpath")):
                relative = row.get(key)
                if not relative:
                    continue
                entry[role] = {
                    "relative": relative,
                    "resolved": resolve_relative(relative, roots.get(role, ())),
                }
            checked.append(entry)
        unresolved = [
            entry
            for entry in checked
            for role in ("image", "audio")
            if role in entry and entry[role]["resolved"] is None
        ]
        return {
            "n_probed": len(checked),
            "n_unresolved": len(unresolved),
            "ok": not unresolved,
            "probes": checked,
        }


# ------------------------------------------------------- relative payloads


def to_relative_rows(
    rows: Sequence[Mapping], roots: Mapping[str, Sequence[str | os.PathLike[str]]]
) -> list[dict]:
    """Copies of ``rows`` with media recorded relative to the configured roots.

    The absolute ``image_path`` / ``audio_path`` are dropped: they name one
    person's Drive mount, and writing them into a shared cache would make the
    cache wrong for everybody else.
    """
    out: list[dict] = []
    for row in rows:
        entry = {k: v for k, v in dict(row).items() if k not in ("image_path", "audio_path")}
        if row.get("image_path"):
            entry["image_relpath"] = relative_to_roots(row["image_path"], roots.get("image", ()))
        if row.get("audio_path"):
            entry["audio_relpath"] = relative_to_roots(row["audio_path"], roots.get("audio", ()))
        out.append(entry)
    return out


def to_absolute_rows(
    rows: Sequence[Mapping], roots: Mapping[str, Sequence[str | os.PathLike[str]]]
) -> list[dict]:
    """The inverse: re-resolve relative media against *this* session's roots.

    A row whose media no longer resolves keeps a ``None`` path rather than
    being dropped, so the caller can report the gap instead of quietly running
    on a smaller subset than the cache described.
    """
    out: list[dict] = []
    for row in rows:
        entry = dict(row)
        if row.get("image_relpath"):
            entry["image_path"] = resolve_relative(row["image_relpath"], roots.get("image", ()))
        if row.get("audio_relpath"):
            entry["audio_path"] = resolve_relative(row["audio_relpath"], roots.get("audio", ()))
        out.append(entry)
    return out


def subset_to_relative(
    subset: Mapping, roots: Mapping[str, Sequence[str | os.PathLike[str]]]
) -> dict:
    """A pilot subset with every split row's media made relative."""
    return {
        **{k: v for k, v in dict(subset).items() if k != "splits"},
        "splits": {
            split: to_relative_rows(rows, roots)
            for split, rows in subset["splits"].items()
        },
        "paths_are_relative": True,
    }


def subset_to_absolute(
    payload: Mapping, roots: Mapping[str, Sequence[str | os.PathLike[str]]]
) -> dict:
    """The inverse of :func:`subset_to_relative`, against this session's roots."""
    return {
        **{k: v for k, v in dict(payload).items() if k not in ("splits", "paths_are_relative")},
        "splits": {
            split: to_absolute_rows(rows, roots)
            for split, rows in payload["splits"].items()
        },
    }


__all__ = [
    "ARTIFACT_NAMES",
    "CACHE_SCHEMA_VERSION",
    "DEFAULT_CACHE_ROOT",
    "DEFAULT_STAGING_ROOT",
    "SPLIT_ALGORITHM_VERSION",
    "STATE_HIT",
    "STATE_INCOMPATIBLE",
    "STATE_INCOMPLETE",
    "STATE_MISS",
    "SUCCESS_NAME",
    "CacheError",
    "CacheFingerprint",
    "CorruptCacheError",
    "DerivedCache",
    "IncompatibleCacheError",
    "build_fingerprint",
    "file_sha256",
    "read_jsonl_gz",
    "relative_to_roots",
    "resolve_relative",
    "subset_to_absolute",
    "subset_to_relative",
    "to_absolute_rows",
    "to_relative_rows",
    "write_json",
    "write_jsonl_gz",
]
