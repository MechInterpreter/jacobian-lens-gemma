# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Resumable, Drive-backed preprocessing for the L32 convergence-resolution study.

Why this module exists
======================

Section 8a of the resolution notebook used to do two things that are fine on a
local disk and ruinous on a Google Drive mount:

1. it walked **every file** of every completed run twice — once to take a
   name/size/mtime tree digest, then again to read identities out of the unit
   payloads; and
2. it held the entire result in Python memory, so interrupting the runtime
   threw the whole scan away.

A real L4 session spent more than four hours there with no progress output and
nothing durable to resume from. This module replaces that with preprocessing
that is:

**minimal**
    Only the artifacts that can *carry an identity* are enumerated, and only
    the smallest set of those that is **provably complete** is read. The proof
    is written down (:func:`prove_completeness`), and when it cannot be
    established the harvest escalates to a checkpointed fallback over the
    remaining unit families rather than silently narrowing the exclusion set.

**checkpointed**
    Work is done in bounded batches — at most 25 files or 30 seconds, whichever
    comes first — and every batch is committed as one atomically written,
    checksummed shard before the cursor advances. Interrupting the runtime
    loses at most the single in-flight batch. It never restarts from file zero
    while a valid checkpoint exists.

**process-independent**
    Everything a later session needs lives on Drive under a deterministic
    preparation-cache directory keyed by a *pre-model* fingerprint. A fresh
    Python process — in particular the GPU session that runs Stage A — loads
    and verifies the cache without reading a single source unit.

**cheap to verify**
    A completed cache is immutable: recomputing it from the same inputs must
    reproduce the same exclusion digest, and a disagreement is a refusal.

What replaced the whole-tree digest, and why it is stronger
==========================================================

The old proof digested *every* file in a completed run by name, size and mtime,
before and after. This module instead:

* enumerates exactly the identity-bearing families (:data:`FAMILY_PATTERNS`)
  and records each file's name, size and mtime — the same three facts the old
  scan recorded, for exactly the files that can influence this study;
* additionally records a **sha256 of every byte** of those files, taken during
  the single read that harvests them, which the old scan never did — an edit
  that preserves size and mtime is invisible to the old digest and impossible
  to hide from this one;
* re-enumerates those same families afterwards, so a *new* identity-bearing
  file is caught even though tens of thousands of irrelevant files are not
  re-walked;
* refuses, structurally, to open any write path under a protected run prefix
  (:func:`assert_write_allowed`), which is what actually prevents this study
  from modifying a completed run — a digest can only notice afterwards.

For files that cannot contribute an identity, neither scheme protects the
science; the write refusal does.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from jlens.mmpilot.l32_resolution import (
    EXCLUSION_HARVEST_VERSION,
    IDENTITY_FAMILIES,
    ExclusionSet,
    absorb_identities,
)
from jlens.mmpilot.store import canonical_json, payload_checksum

# ------------------------------------------------------------------ versions

#: The preparation protocol. Bound into the preparation fingerprint, so a
#: change gives every preparation a new cache directory instead of resuming one
#: built under different rules.
PREPARATION_VERSION = "mmpilot.l32_resolution_preparation.v1"

#: The Drive namespace preparation caches live in. Never a run directory: a
#: preparation is not a scientific result and must not look like one.
PREP_CACHE_NAMESPACE = "jlens_l32_resolution_prep_v1"

#: How sources are chosen and how completeness is proven.
SOURCE_STRATEGY_VERSION = "mmpilot.l32_resolution_source_strategy.v1"

INVENTORY_SCHEMA = "jlens.mmpilot.l32_resolution_source_inventory.v1"
HARVEST_STATE_SCHEMA = "jlens.mmpilot.l32_resolution_harvest_state.v1"
SHARD_SCHEMA = "jlens.mmpilot.l32_resolution_harvest_shard.v1"
COMPLETENESS_SCHEMA = "jlens.mmpilot.l32_resolution_completeness_proof.v1"
SELECTION_SCHEMA = "jlens.mmpilot.l32_resolution_prepared_selection.v1"
COMPLETE_SCHEMA = "jlens.mmpilot.l32_resolution_preparation_complete.v1"

# --------------------------------------------------------------- work sizing

#: Files per bounded work unit. The brief's number, kept here so the notebook
#: does not carry a magic constant.
DEFAULT_BATCH_FILES = 25

#: Seconds after which an in-flight batch is committed even if it is short. A
#: Drive stall must not turn a 25-file batch into a 25-minute one.
DEFAULT_CHECKPOINT_SECONDS = 30.0

#: How often progress must be printed. No loop in this module may be quieter.
DEFAULT_PROGRESS_SECONDS = 30.0

#: Bounded retries around the atomic rename. See :func:`atomic_write_bytes`.
RENAME_ATTEMPTS = 5
RENAME_BACKOFF_SECONDS = 0.05

#: Hex characters of the preparation digest that name the cache directory.
#: See :func:`preparation_cache_dir` for why this is a prefix and not the whole
#: digest — the full value is still compared before anything is reused.
CACHE_DIR_DIGEST_CHARS = 32


class PreparationRefused(RuntimeError):
    """A precondition of the preprocessing stage does not hold."""


class PreparationIncompatible(PreparationRefused):
    """Persisted preprocessing state was produced under different inputs."""


class CorruptShard(PreparationRefused):
    """A checkpoint shard failed its own checksum and could not be trusted."""


class CompletenessNotProven(PreparationRefused):
    """The exclusion set could not be shown to cover the completed population."""


# ================================================================== families


#: Every artifact family that can carry a media identity, and how to find it
#: inside a completed run. Nothing outside this mapping is ever read, and
#: nothing inside it is ever written.
FAMILY_PATTERNS: dict[str, tuple[str, ...]] = {
    "population_manifest": ("population_manifest.json",),
    "run_documents": (
        "fingerprint.json",
        "split_provenance.json",
        "run_manifest.json",
    ),
    "activation": ("units/activation/*.json",),
    "capability": ("units/capability/*.json",),
    "jspace": ("units/jspace/*.json",),
    "direction": ("units/direction/*.json",),
    "intervention": ("units/intervention/*.json",),
    "readout": ("convergence/readout_units/*.json",),
}

#: The families tried first. ``population_manifest`` is a bulk enumeration when
#: a run wrote one; ``activation`` is the smallest per-unit family that the
#: pipeline's own invariants make a superset of every other unit family.
MINIMAL_FAMILIES: tuple[str, ...] = (
    "population_manifest",
    "run_documents",
    "activation",
)

#: Read only when completeness cannot be proven from the minimal set. This is
#: the resumable fallback, not a default.
FALLBACK_FAMILIES: tuple[str, ...] = (
    "capability",
    "jspace",
    "direction",
    "intervention",
    "readout",
)

#: Enumeration order. Used to order the inventory deterministically.
ALL_FAMILIES: tuple[str, ...] = (*MINIMAL_FAMILIES, *FALLBACK_FAMILIES)

#: Why the fallback families cannot hold an identity the minimal set missed,
#: *when* the minimal set's recovered counts match the completed run's own
#: recorded population. Recorded in every completeness proof, because the
#: reader must be able to check the reasoning and not just the conclusion.
SKIP_JUSTIFICATION = (
    "jlens.mmpilot.pipeline.stage_activations iterates every group of both "
    "splits (keeping concept=None negatives) and writes one activation unit "
    "per (group, modality, layer); stage_capability iterates the same subset "
    "but is capped by max_capability_groups_per_concept; stage_codes and "
    "stage_directions consume activation records; stage_causal draws its "
    "targets from subset['splits']['test'] and keys them by the same "
    "sample_id(group_id, modality); the native readout units are built from "
    "the activation population. Every one of those families is therefore a "
    "SUBSET of the activation population in group_id, image_id and sample_id. "
    "That invariant is not taken on trust here: the proof compares the "
    "recovered distinct group and image counts against the completed run's own "
    "split_provenance.json, and any shortfall escalates to the checkpointed "
    "fallback scan over the skipped families."
)


# =============================================================== small helpers


def utc_now() -> str:
    """An ISO-8601 UTC stamp. One place, so every artifact agrees."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def assert_write_allowed(
    path: str | os.PathLike[str], *, protected_prefixes: Sequence[str] = ()
) -> str:
    """Refuse a write whose path passes through a protected run namespace.

    This is the structural half of the read-only guarantee. A tree digest can
    only tell you afterwards that a completed run changed; this refuses the
    write in the first place, which is the property the study actually needs.

    Raises:
        PreparationRefused: When any path component starts with a protected
            prefix.
    """
    target = Path(path)
    offending = sorted(
        {
            prefix
            for prefix in protected_prefixes
            for part in target.parts
            if part.startswith(prefix)
        }
    )
    if offending:
        raise PreparationRefused(
            f"refusing to write {target}: it lies inside a completed run "
            f"namespace ({offending}). Completed runs are evidence. "
            "Preparation artifacts belong in the derived preparation cache and "
            "scientific outputs in this study's own run directory."
        )
    return str(target)


def atomic_write_bytes(
    path: str | os.PathLike[str],
    data: bytes,
    *,
    protected_prefixes: Sequence[str] = (),
) -> Path:
    """Write ``data`` to ``path`` atomically, or refuse a protected location.

    ``os.replace`` is the whole atomicity story: a reader either sees the
    previous file or the complete new one, never a half-written one. The
    temporary name carries the pid so two processes cannot collide on it.

    The rename is retried a few times. A Drive mount and a Windows filesystem
    can both fail a rename transiently for reasons that have nothing to do with
    this program — a scanner holding the destination open, a mount hiccup — and
    losing a whole checkpoint to one of those would defeat the point. The retry
    is bounded and re-raises rather than pretending the write happened.
    """
    target = Path(assert_write_allowed(path, protected_prefixes=protected_prefixes))
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.name}.tmp.{os.getpid()}")
    tmp.write_bytes(data)
    for attempt in range(RENAME_ATTEMPTS):
        try:
            os.replace(tmp, target)
            return target
        except OSError:
            if attempt == RENAME_ATTEMPTS - 1:
                tmp.unlink(missing_ok=True)
                raise
            time.sleep(RENAME_BACKOFF_SECONDS * (attempt + 1))
    return target  # pragma: no cover - the loop either returns or raises


def atomic_write_json(
    path: str | os.PathLike[str],
    payload: object,
    *,
    protected_prefixes: Sequence[str] = (),
) -> Path:
    """:func:`atomic_write_bytes` for an indented JSON document."""
    return atomic_write_bytes(
        path,
        (
            json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n"
        ).encode("utf-8"),
        protected_prefixes=protected_prefixes,
    )


def atomic_write_text(
    path: str | os.PathLike[str],
    text: str,
    *,
    protected_prefixes: Sequence[str] = (),
) -> Path:
    """:func:`atomic_write_bytes` for a UTF-8 text document."""
    return atomic_write_bytes(
        path, text.encode("utf-8"), protected_prefixes=protected_prefixes
    )


def _read_json(path: Path) -> object | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None


def _relpath(root: Path, path: Path) -> str:
    return str(path.relative_to(root)).replace("\\", "/")


# ======================================================= progress reporting


@dataclass
class ProgressReporter:
    """Rate-limited progress, so no long loop is ever silent.

    ``interval`` is a *maximum* quiet period, not a minimum: the first and last
    ticks of a phase always print. ``clock`` is injectable so the test suite can
    prove the reporter speaks during a long scan without sleeping through one.
    """

    interval: float = DEFAULT_PROGRESS_SECONDS
    printer: Callable[[str], None] = print
    clock: Callable[[], float] = time.monotonic
    prefix: str = "[prep]"
    lines: list[str] = field(default_factory=list)
    _last: float | None = field(default=None, init=False, repr=False)
    _start: float | None = field(default=None, init=False, repr=False)

    # ---------------------------------------------------------------- phases

    def begin(self, message: str) -> None:
        """Start (or restart) the clock and print an unconditional banner."""
        self._start = self.clock()
        self._last = None
        self.emit(message)

    def emit(self, message: str) -> None:
        """Print unconditionally and remember the line for the artifact."""
        line = f"{self.prefix} {message}"
        self.lines.append(line)
        self.printer(line)
        self._last = self.clock()

    @property
    def elapsed(self) -> float:
        return 0.0 if self._start is None else max(0.0, self.clock() - self._start)

    def restart_clock(self) -> None:
        """Measure throughput from *now*, so reused work does not flatter it."""
        self._start = self.clock()
        self._last = self._start

    # ------------------------------------------------------------------ tick

    def note(self, message: str, *, force: bool = False) -> bool:
        """A rate-limited line with no progress arithmetic in it."""
        now = self.clock()
        if not force and self._last is not None and now - self._last < self.interval:
            return False
        self.emit(message)
        return True

    def tick(
        self,
        *,
        done: int,
        total: int,
        computed: int | None = None,
        force: bool = False,
        **fields: object,
    ) -> bool:
        """Maybe print one progress line. Returns whether it printed.

        The estimate is measured throughput, not a guess: seconds elapsed in
        *this* session divided by the files this session actually read. Files
        reused from a previous session took no time here and would make the
        remaining estimate a fiction if they were counted.
        """
        now = self.clock()
        if not force and self._last is not None and now - self._last < self.interval:
            return False
        elapsed = self.elapsed
        basis = done if computed is None else computed
        rate = basis / elapsed if elapsed > 0 and basis > 0 else 0.0
        remaining = (total - done) / rate if rate > 0 else None
        detail = "  ".join(f"{name}={value}" for name, value in fields.items())
        self.emit(
            f"{done}/{total} files  {detail}  "
            f"elapsed {elapsed:.0f}s  "
            + (
                f"eta {remaining:.0f}s"
                if remaining is not None
                else "eta unknown (no throughput measured yet)"
            )
        )
        return True


def _null_progress() -> ProgressReporter:
    return ProgressReporter(printer=lambda _line: None)


# ==================================================== preparation fingerprint


#: Everything upstream of the model that can change which media are excluded or
#: which are selected. A change in any of them must produce a different cache
#: directory — reusing a preparation across a change here would silently mix two
#: populations.
PREPARATION_FINGERPRINT_FIELDS: tuple[str, ...] = (
    "preparation_version",
    "harvest_version",
    "source_strategy_version",
    "identity_families",
    "fallback_rule",
    "completed_run_basenames",
    "completed_run_fingerprints",
    "completed_summary_checksums",
    "cached_expanded_manifest_checksum",
    "cache_schema_version",
    "evidence_lexicon_hash",
    "frozen_selected_concepts",
    "frozen_focal_concepts",
    "sample_size_rule_version",
    "sample_size_plan_digest",
    "selection_algorithm_version",
    "selection_seed",
    "selection_profile_version",
    "n_train_positive_images",
    "n_test_positive_images",
    "n_train_negative_images",
    "n_test_negative_images",
)


def preparation_fingerprint(**fields: object) -> dict:
    """Bind every pre-model input to a digest, refusing a missing one.

    Raises:
        PreparationRefused: If a declared field is missing or an undeclared one
            is supplied. Defaulting a forgotten field to ``None`` is how two
            different preparations come to share a cache.
    """
    missing = [name for name in PREPARATION_FINGERPRINT_FIELDS if name not in fields]
    unknown = sorted(set(fields) - set(PREPARATION_FINGERPRINT_FIELDS))
    if missing or unknown:
        raise PreparationRefused(
            "the preparation fingerprint is incomplete or over-specified: "
            f"missing {missing}, unknown {unknown}"
        )
    payload = {name: fields[name] for name in PREPARATION_FINGERPRINT_FIELDS}
    return {**payload, "preparation_digest": payload_checksum(payload)}


def default_fingerprint_constants() -> dict:
    """The fields this module owns, so a caller cannot mistype one."""
    return {
        "preparation_version": PREPARATION_VERSION,
        "harvest_version": EXCLUSION_HARVEST_VERSION,
        "source_strategy_version": SOURCE_STRATEGY_VERSION,
        "identity_families": list(IDENTITY_FAMILIES),
        "fallback_rule": (
            "minimal sources first; escalate to a checkpointed scan of "
            f"{list(FALLBACK_FAMILIES)} when the recovered group/image counts "
            "do not reach the completed run's own recorded population"
        ),
    }


def preparation_cache_dir(
    root: str | os.PathLike[str], fingerprint: Mapping
) -> Path:
    """``<root>/<PREP_CACHE_NAMESPACE>/prep_<preparation digest prefix>``.

    Deterministic, and deliberately **not** keyed by a timestamped run
    directory: preprocessing is a function of its inputs, and a study that had
    to name a run before it could reuse a scan would rescan once per session.

    The directory name carries a :data:`CACHE_DIR_DIGEST_CHARS`-hex prefix
    rather than the whole digest. 128 bits is not a collision risk, and the
    whole digest still has to match: it is stored in
    ``preparation_fingerprint.json`` and compared in full before any cached
    work is reused. Shortening it keeps the deepest artifact path inside the
    260-character limit Windows still enforces by default, which a 64-character
    directory plus a shard filename does not.
    """
    digest = str(fingerprint["preparation_digest"]).split(":", 1)[-1]
    return Path(root) / PREP_CACHE_NAMESPACE / f"prep_{digest[:CACHE_DIR_DIGEST_CHARS]}"


def completed_run_identity(run_dir: str | os.PathLike[str]) -> dict:
    """What a completed run is, for fingerprinting purposes.

    Its basename, the digest its own ``fingerprint.json`` records, and a
    checksum of each summary/report document. Never its absolute path: a
    remounted Drive is not a different experiment.
    """
    root = Path(run_dir)
    record: dict = {
        "run": root.name,
        "exists": root.is_dir(),
        "fingerprint_digest": None,
        "summary_checksums": {},
    }
    payload = _read_json(root / "fingerprint.json")
    if isinstance(payload, Mapping):
        stored = dict(payload)
        stored.pop("written_utc", None)
        record["fingerprint_digest"] = str(
            stored.get("fingerprint_digest") or payload_checksum(stored)
        )
    for name in sorted(
        path.name
        for path in root.glob("*report.json")
        if path.is_file()
    ) + sorted(
        path.name
        for path in root.glob("*summary.json")
        if path.is_file()
    ):
        try:
            record["summary_checksums"][name] = "sha256:" + hashlib.sha256(
                (root / name).read_bytes()
            ).hexdigest()
        except OSError as error:  # pragma: no cover - reported, not swallowed
            record["summary_checksums"][name] = f"unreadable: {error}"
    return record


# ================================================================ source plan


def _inspect_population_manifest(path: Path) -> dict | None:
    """What a bulk population manifest actually enumerates, if it exists."""
    payload = _read_json(path)
    if not isinstance(payload, Mapping):
        return None
    units = payload.get("units")
    if not isinstance(units, Sequence) or isinstance(units, (str, bytes)):
        return None
    rows = [row for row in units if isinstance(row, Mapping)]
    return {
        "n_units": len(rows),
        "n_group_ids": len({str(row["group_id"]) for row in rows if row.get("group_id")}),
        "n_image_ids": len({str(row["image_id"]) for row in rows if row.get("image_id")}),
        "has_audio_paths": any(row.get("audio_path") for row in rows),
        "has_captions": any(row.get("caption") for row in rows),
    }


def _expected_population(run_dir: Path) -> dict:
    """The completed run's own record of how many groups and images it used.

    This is the anchor the completeness proof needs. Without it "we recovered
    412 group ids" is a number with nothing to check it against, and the proof
    would only be able to say that the harvest agreed with itself.
    """
    provenance = _read_json(run_dir / "split_provenance.json")
    if isinstance(provenance, Mapping) and provenance.get("n_groups") is not None:
        return {
            "source": "split_provenance.json",
            "n_groups": int(provenance["n_groups"]),
            "n_distinct_images": (
                int(provenance["n_distinct_images"])
                if provenance.get("n_distinct_images") is not None
                else None
            ),
        }
    manifest = _inspect_population_manifest(run_dir / "population_manifest.json")
    if manifest is not None and manifest["n_group_ids"]:
        return {
            "source": "population_manifest.json:units",
            "n_groups": manifest["n_group_ids"],
            "n_distinct_images": manifest["n_image_ids"] or None,
        }
    return {"source": None, "n_groups": None, "n_distinct_images": None}


def plan_sources(run_dirs: Sequence[str | os.PathLike[str]]) -> dict:
    """Choose the minimal identity-bearing artifacts for each completed run.

    Preference order, per the study's brief:

    1. a bulk population manifest that enumerates every used group;
    2. any other bulk document that carries complete identities;
    3. the activation unit family, whose superset property over the other unit
       families is stated in :data:`SKIP_JUSTIFICATION` **and checked** against
       the run's own recorded population;
    4. a checkpointed fallback scan of the remaining families, only when
       completeness cannot be established.

    Capability units are never treated as a population source: they are capped
    by ``max_capability_groups_per_concept`` and cover the positives only.
    """
    runs: list[dict] = []
    for run_dir in run_dirs:
        root = Path(run_dir)
        expected = _expected_population(root)
        bulk = _inspect_population_manifest(root / "population_manifest.json")
        families = ["population_manifest", "run_documents"]
        strategy = "bulk_population_manifest"
        if (
            bulk is None
            or expected["n_groups"] is None
            or bulk["n_group_ids"] < expected["n_groups"]
        ):
            families.append("activation")
            strategy = "activation_units_with_recorded_population_anchor"
        runs.append(
            {
                "run": root.name,
                "run_dir": str(root),
                "exists": root.is_dir(),
                "expected": expected,
                "bulk_population_manifest": bulk,
                "strategy": strategy,
                # ``minimal_families`` never changes; escalation fills in
                # ``fallback_families``. Keeping them apart is what lets a
                # resumed session recompute the minimal stage's inventory
                # without the fallback's entries shifting every cursor in it.
                "minimal_families": families,
                "fallback_families": [],
            }
        )
    return {
        "schema": "jlens.mmpilot.l32_resolution_source_plan.v1",
        "strategy_version": SOURCE_STRATEGY_VERSION,
        "runs": runs,
        "skip_justification": SKIP_JUSTIFICATION,
        "capability_units_are_not_a_population_source": (
            "stage_capability stops at max_capability_groups_per_concept and "
            "scores positives only, so capability units cannot be assumed to "
            "enumerate the population"
        ),
    }


# =========================================================== source inventory


def build_source_inventory(
    run_dirs: Sequence[str | os.PathLike[str]],
    families_by_run: Mapping[str, Sequence[str]],
    *,
    progress: ProgressReporter | None = None,
    label: str = "inventory",
) -> dict:
    """Enumerate exactly the chosen artifacts, once, in a deterministic order.

    One enumeration serves three jobs — source integrity, identity harvesting
    and the after-the-fact immutability check — which is the point: the old
    section 8a walked the whole tree twice and read nothing useful the second
    time.
    """
    reporter = progress or _null_progress()
    reporter.begin(f"{label}: enumerating identity-bearing artifacts")
    entries: list[dict] = []
    per_run: list[dict] = []
    for run_dir in run_dirs:
        root = Path(run_dir)
        chosen = list(families_by_run.get(root.name, ()))
        counts: dict[str, int] = {}
        for family in sorted(chosen, key=ALL_FAMILIES.index):
            found: list[Path] = []
            for pattern in FAMILY_PATTERNS[family]:
                found.extend(path for path in root.glob(pattern) if path.is_file())
            for path in sorted(set(found)):
                try:
                    stat = path.stat()
                except OSError:
                    continue
                entries.append(
                    {
                        "run": root.name,
                        "run_dir": str(root),
                        "family": family,
                        "relpath": _relpath(root, path),
                        "size": int(stat.st_size),
                        "mtime": int(stat.st_mtime),
                    }
                )
            counts[family] = len(set(found))
            reporter.note(
                f"{label}: enumerated {root.name}/{family} "
                f"({counts[family]} file(s), {len(entries)} so far)"
            )
        per_run.append(
            {
                "run": root.name,
                "run_dir": str(root),
                "exists": root.is_dir(),
                "families": counts,
                "n_files": sum(counts.values()),
            }
        )

    entries.sort(key=lambda row: (row["run"], ALL_FAMILIES.index(row["family"]), row["relpath"]))
    inventory = {
        "schema": INVENTORY_SCHEMA,
        "strategy_version": SOURCE_STRATEGY_VERSION,
        "label": label,
        "runs": per_run,
        "n_files": len(entries),
        "entries": entries,
    }
    inventory["inventory_digest"] = payload_checksum(
        [
            [row["run"], row["family"], row["relpath"], row["size"], row["mtime"]]
            for row in entries
        ]
    )
    reporter.emit(
        f"{label}: {len(entries)} file(s) across "
        f"{len([r for r in per_run if r['n_files']])} run(s)  "
        f"digest {inventory['inventory_digest'][:23]}..."
    )
    return inventory


def verify_sources_unchanged(
    run_dirs: Sequence[str | os.PathLike[str]],
    inventory: Mapping,
    families_by_run: Mapping[str, Sequence[str]],
    *,
    rehash: bool = False,
    file_checksums: Mapping[str, str] | None = None,
) -> dict:
    """Re-enumerate the harvested families and prove none of them moved.

    Cheap by construction: it walks only the families that were read, so a new
    unrelated file elsewhere in a completed run does not cost a rescan of tens
    of thousands of irrelevant files — while a new *identity-bearing* file, or
    a changed size or mtime on one, is caught.

    Args:
        rehash: Also re-read and re-hash every enumerated file and compare
            against ``file_checksums`` (the digests taken during the harvest).
            Off by default because it doubles the Drive reads the whole module
            exists to avoid; the harvest-time content digests are what a later
            reader verifies against.
    """
    after = build_source_inventory(
        run_dirs, families_by_run, progress=_null_progress(), label="verify"
    )
    before_rows = {
        (row["run"], row["relpath"]): row for row in inventory.get("entries", [])
    }
    after_rows = {(row["run"], row["relpath"]): row for row in after["entries"]}
    appeared = sorted(f"{run}/{rel}" for run, rel in set(after_rows) - set(before_rows))
    vanished = sorted(f"{run}/{rel}" for run, rel in set(before_rows) - set(after_rows))
    modified = sorted(
        f"{run}/{rel}"
        for run, rel in set(before_rows) & set(after_rows)
        if (before_rows[(run, rel)]["size"], before_rows[(run, rel)]["mtime"])
        != (after_rows[(run, rel)]["size"], after_rows[(run, rel)]["mtime"])
    )

    rehashed: list[str] = []
    if rehash and file_checksums is not None:
        for (run, rel), row in sorted(after_rows.items()):
            expected = file_checksums.get(f"{run}/{rel}")
            if expected is None:
                continue
            try:
                actual = "sha256:" + hashlib.sha256(
                    (Path(row["run_dir"]) / rel).read_bytes()
                ).hexdigest()
            except OSError:
                rehashed.append(f"{run}/{rel} (unreadable)")
                continue
            if actual != expected:
                rehashed.append(f"{run}/{rel}")

    return {
        "schema": "jlens.mmpilot.l32_resolution_source_immutability.v1",
        "method": (
            "re-enumeration of the harvested identity-bearing families "
            "(name+size+mtime), against sha256 content digests taken during the "
            "single harvest read"
        ),
        "families_verified": {
            run: sorted(families_by_run.get(run, ())) for run in sorted(families_by_run)
        },
        "inventory_digest_before": inventory.get("inventory_digest"),
        "inventory_digest_after": after["inventory_digest"],
        "n_files_before": inventory.get("n_files"),
        "n_files_after": after["n_files"],
        "appeared": appeared,
        "vanished": vanished,
        "modified": modified,
        "content_rehash_performed": bool(rehash and file_checksums is not None),
        "content_mismatches": rehashed,
        "unchanged": not (appeared or vanished or modified or rehashed),
    }


def assert_sources_unchanged(record: Mapping) -> dict:
    """Refuse when :func:`verify_sources_unchanged` found a difference."""
    if not record.get("unchanged"):
        raise PreparationRefused(
            "a completed run's identity-bearing artifacts changed while this "
            "study ran, which must never happen:\n"
            + json.dumps(
                {
                    key: record.get(key)
                    for key in (
                        "appeared",
                        "vanished",
                        "modified",
                        "content_mismatches",
                    )
                },
                indent=2,
            )
        )
    return dict(record)


# ================================================ checkpointed shard harvest


def _shard_path(stage_dir: Path, index: int) -> Path:
    return stage_dir / "shards" / f"shard_{index:05d}.json.gz"


def _write_shard(
    stage_dir: Path, body: dict, *, protected_prefixes: Sequence[str]
) -> dict:
    record = {**body, "shard_checksum": payload_checksum(body)}
    atomic_write_bytes(
        _shard_path(stage_dir, int(body["shard_index"])),
        gzip.compress(canonical_json(record).encode("utf-8"), mtime=0),
        protected_prefixes=protected_prefixes,
    )
    return record


def read_shard(path: Path) -> dict:
    """Load and verify one shard.

    Raises:
        CorruptShard: On a torn gzip stream, unparseable JSON, a missing
            checksum or a checksum that does not match the body. A shard that
            cannot prove it is intact is never treated as data.
    """
    try:
        record = json.loads(gzip.decompress(path.read_bytes()).decode("utf-8"))
    except (OSError, gzip.BadGzipFile, EOFError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CorruptShard(f"{path} is unreadable: {error}") from error
    if not isinstance(record, dict) or "shard_checksum" not in record:
        raise CorruptShard(f"{path} carries no shard checksum")
    body = {key: value for key, value in record.items() if key != "shard_checksum"}
    if payload_checksum(body) != record["shard_checksum"]:
        raise CorruptShard(
            f"{path} failed its own checksum; the shard is torn and is not data"
        )
    return record


def _quarantine(stage_dir: Path, path: Path, *, protected_prefixes: Sequence[str]) -> str:
    destination = stage_dir / "quarantine" / path.name
    assert_write_allowed(destination, protected_prefixes=protected_prefixes)
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(path, destination)
    return str(destination)


def _empty_state(stage: str, inventory: Mapping, fingerprint: Mapping) -> dict:
    return {
        "schema": HARVEST_STATE_SCHEMA,
        "stage": stage,
        "preparation_digest": fingerprint["preparation_digest"],
        "inventory_digest": inventory["inventory_digest"],
        "n_files_total": int(inventory["n_files"]),
        "cursor": 0,
        "complete": int(inventory["n_files"]) == 0,
        "shards": [],
        "quarantined": [],
        "updated_utc": utc_now(),
    }


def load_harvest_state(
    stage_dir: Path,
    inventory: Mapping,
    fingerprint: Mapping,
    *,
    stage: str,
    on_corrupt: str = "quarantine",
    protected_prefixes: Sequence[str] = (),
) -> dict:
    """Recover the last durable checkpoint, or start one.

    Three things can be found on disk and each is handled explicitly:

    * **no state** — a fresh preparation;
    * **state from different inputs** — a refusal, never a silent restart;
    * **state whose shards do not all verify** — the first bad shard and every
      shard after it are quarantined and the cursor rewinds to that shard's
      start. Earlier valid checkpoints are kept, so a torn shard costs one
      batch and never the whole scan.

    A shard written just before an interruption killed the state update is
    *adopted* when it verifies and starts exactly at the cursor: it is a
    durable, atomically written checkpoint, and repeating it would be work this
    module promised not to repeat.
    """
    path = stage_dir / "harvest_state.json"
    stored = _read_json(path)
    if not isinstance(stored, Mapping):
        return _empty_state(stage, inventory, fingerprint)

    mismatches = [
        f"{key}: stored={stored.get(key)!r} requested={value!r}"
        for key, value in (
            ("schema", HARVEST_STATE_SCHEMA),
            ("stage", stage),
            ("preparation_digest", fingerprint["preparation_digest"]),
            ("inventory_digest", inventory["inventory_digest"]),
        )
        if stored.get(key) != value
    ]
    if mismatches:
        raise PreparationIncompatible(
            f"{path} was written under different preprocessing inputs; refusing "
            "to resume it.\n  " + "\n  ".join(mismatches) + "\n"
            "A completed run's identity-bearing artifacts, or the preparation "
            "fingerprint, changed. Preprocessing under the new inputs belongs "
            "in a new preparation cache directory."
        )

    state = {
        **_empty_state(stage, inventory, fingerprint),
        "quarantined": list(stored.get("quarantined") or []),
    }
    cursor = 0
    for entry in stored.get("shards") or []:
        index = int(entry.get("index", -1))
        shard_file = _shard_path(stage_dir, index)
        try:
            record = read_shard(shard_file)
            if (
                record.get("shard_checksum") != entry.get("checksum")
                or int(record["cursor_start"]) != cursor
            ):
                raise CorruptShard(
                    f"{shard_file} does not continue the checkpoint chain "
                    f"(expected cursor_start={cursor})"
                )
        except CorruptShard as error:
            if on_corrupt == "refuse":
                raise
            if shard_file.is_file():
                state["quarantined"].append(
                    {
                        "shard": _quarantine(
                            stage_dir, shard_file, protected_prefixes=protected_prefixes
                        ),
                        "reason": str(error),
                        "recomputed_from_cursor": cursor,
                    }
                )
            else:
                state["quarantined"].append(
                    {"shard": str(shard_file), "reason": str(error), "missing": True}
                )
            break
        state["shards"].append(dict(entry))
        cursor = int(record["cursor_end"])

    # Quarantine every shard file at or beyond the surviving chain that we are
    # not about to adopt, so a rewritten batch can never collide with a stale
    # file of the same name.
    while True:
        index = len(state["shards"])
        shard_file = _shard_path(stage_dir, index)
        if not shard_file.is_file():
            break
        try:
            record = read_shard(shard_file)
        except CorruptShard as error:
            if on_corrupt == "refuse":
                raise
            state["quarantined"].append(
                {
                    "shard": _quarantine(
                        stage_dir, shard_file, protected_prefixes=protected_prefixes
                    ),
                    "reason": str(error),
                    "recomputed_from_cursor": cursor,
                }
            )
            break
        if int(record["cursor_start"]) != cursor:
            state["quarantined"].append(
                {
                    "shard": _quarantine(
                        stage_dir, shard_file, protected_prefixes=protected_prefixes
                    ),
                    "reason": (
                        f"orphan shard starts at {record['cursor_start']}, not at "
                        f"the resume cursor {cursor}"
                    ),
                    "recomputed_from_cursor": cursor,
                }
            )
            break
        state["shards"].append(
            {
                "index": index,
                "cursor_start": int(record["cursor_start"]),
                "cursor_end": int(record["cursor_end"]),
                "checksum": record["shard_checksum"],
                "n_files": len(record.get("files") or []),
                "adopted_orphan": True,
            }
        )
        cursor = int(record["cursor_end"])

    state["cursor"] = cursor
    state["complete"] = cursor >= int(inventory["n_files"])
    return state


def _harvest_file(entry: Mapping) -> tuple[dict, ExclusionSet | None, str | None]:
    """Read one file once: hash its bytes, parse it, pull its identities out."""
    path = Path(entry["run_dir"]) / entry["relpath"]
    try:
        raw = path.read_bytes()
    except OSError as error:
        return dict(entry), None, f"unreadable: {error}"
    checksum = "sha256:" + hashlib.sha256(raw).hexdigest()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        return {**entry, "checksum": checksum}, None, f"unparseable: {error}"
    found = ExclusionSet()
    n_identities = absorb_identities(found, payload)
    return (
        {**entry, "checksum": checksum, "n_identities": int(n_identities)},
        found,
        None,
    )


def _identity_payload(found: Mapping[str, ExclusionSet]) -> dict:
    return {
        run: {
            name: sorted(getattr(exclusion, name))
            for name in (
                "image_ids",
                "group_ids",
                "sample_ids",
                "audio_paths",
                "captions",
                "media_checksums",
            )
        }
        for run, exclusion in sorted(found.items())
    }


def harvest_checkpointed(
    stage_dir: str | os.PathLike[str],
    inventory: Mapping,
    fingerprint: Mapping,
    *,
    stage: str,
    batch_files: int = DEFAULT_BATCH_FILES,
    checkpoint_seconds: float = DEFAULT_CHECKPOINT_SECONDS,
    progress: ProgressReporter | None = None,
    protected_prefixes: Sequence[str] = (),
    on_corrupt: str = "quarantine",
    clock: Callable[[], float] = time.monotonic,
) -> dict:
    """Harvest the inventory in bounded, atomically committed batches.

    A batch ends at ``batch_files`` files **or** ``checkpoint_seconds``,
    whichever comes first, and the shard is on disk before the cursor moves.
    Stopping the runtime at any moment therefore loses at most the one batch
    that was in flight, and never a completed one.
    """
    root = Path(stage_dir)
    assert_write_allowed(root, protected_prefixes=protected_prefixes)
    root.mkdir(parents=True, exist_ok=True)
    reporter = progress or _null_progress()

    state = load_harvest_state(
        root,
        inventory,
        fingerprint,
        stage=stage,
        on_corrupt=on_corrupt,
        protected_prefixes=protected_prefixes,
    )
    entries = list(inventory["entries"])
    total = len(entries)
    reused_files = sum(int(shard["n_files"]) for shard in state["shards"])

    if state["shards"] or state["quarantined"]:
        reporter.emit("=" * 68)
        reporter.emit(f"RESUMING PREPROCESSING — stage {stage!r}")
        reporter.emit(f"  completed shards        {len(state['shards'])}")
        reporter.emit(f"  completed files         {state['cursor']}/{total}")
        reporter.emit(f"  remaining files         {max(0, total - state['cursor'])}")
        reporter.emit(
            f"  last durable checkpoint {_shard_path(root, len(state['shards']) - 1).name}"
            if state["shards"]
            else "  last durable checkpoint none"
        )
        reporter.emit(f"  reused from Drive       {reused_files} file(s)")
        reporter.emit(f"  quarantined shards      {len(state['quarantined'])}")
        reporter.emit("=" * 68)
    else:
        reporter.emit(f"starting preprocessing stage {stage!r}: {total} file(s) to read")

    computed_files = 0
    # Identity counts printed alongside progress are this session's, and are
    # labelled as such: seeding them from the committed shards would mean
    # reading every shard back just to make a progress line look bigger.
    session_ids: dict[str, set[str]] = {name: set() for name in IDENTITY_FAMILIES}
    reporter.restart_clock()
    while state["cursor"] < total:
        batch_start_cursor = state["cursor"]
        batch_started = clock()
        by_run: dict[str, ExclusionSet] = {}
        files: list[dict] = []
        unreadable: list[dict] = []
        cursor = batch_start_cursor
        while cursor < total:
            entry = entries[cursor]
            record, found, error = _harvest_file(entry)
            if error is not None:
                unreadable.append({**record, "error": error})
            else:
                target = by_run.setdefault(str(entry["run"]), ExclusionSet())
                if found is not None:
                    for name in (
                        "image_ids",
                        "group_ids",
                        "sample_ids",
                        "audio_paths",
                        "captions",
                        "media_checksums",
                    ):
                        getattr(target, name).update(getattr(found, name))
                        if name in session_ids:
                            session_ids[name].update(getattr(found, name))
                files.append(record)
            cursor += 1
            computed_files += 1
            reporter.tick(
                done=cursor,
                total=total,
                computed=computed_files,
                run=entry["run"],
                family=entry["family"],
                shard=len(state["shards"]),
                new_ids=" ".join(
                    f"{name.split('_')[0][:3]}={len(values)}"
                    for name, values in session_ids.items()
                ),
                work="computed",
            )
            if (
                cursor - batch_start_cursor >= batch_files
                or clock() - batch_started >= checkpoint_seconds
            ):
                break

        index = len(state["shards"])
        record = _write_shard(
            root,
            {
                "schema": SHARD_SCHEMA,
                "stage": stage,
                "shard_index": index,
                "preparation_digest": fingerprint["preparation_digest"],
                "inventory_digest": inventory["inventory_digest"],
                "cursor_start": batch_start_cursor,
                "cursor_end": cursor,
                "files": files,
                "unreadable": unreadable,
                "identities": _identity_payload(by_run),
                "written_utc": utc_now(),
            },
            protected_prefixes=protected_prefixes,
        )
        state["shards"].append(
            {
                "index": index,
                "cursor_start": batch_start_cursor,
                "cursor_end": cursor,
                "checksum": record["shard_checksum"],
                "n_files": len(files),
            }
        )
        state["cursor"] = cursor
        state["complete"] = cursor >= total
        state["updated_utc"] = utc_now()
        atomic_write_json(
            root / "harvest_state.json",
            state,
            protected_prefixes=protected_prefixes,
        )

    state["batch_files"] = int(batch_files)
    state["checkpoint_seconds"] = float(checkpoint_seconds)
    state["files_computed_this_session"] = computed_files
    state["files_reused_from_drive"] = reused_files
    state["complete"] = state["cursor"] >= total
    state["updated_utc"] = utc_now()
    atomic_write_json(
        root / "harvest_state.json", state, protected_prefixes=protected_prefixes
    )
    reporter.emit(
        f"stage {stage!r} complete: {state['cursor']}/{total} file(s), "
        f"{len(state['shards'])} shard(s), "
        f"{computed_files} computed this session, {reused_files} reused from Drive"
    )
    return state


def reduce_shards(stage_dirs: Sequence[str | os.PathLike[str]]) -> dict:
    """Merge every committed shard into one deduplicated identity set.

    Reduction is a pure function of the shards, so it is redone on every load
    rather than cached: a cached reduction is one more thing that can disagree
    with the checkpoints it claims to summarize.
    """
    by_run: dict[str, ExclusionSet] = {}
    file_checksums: dict[str, str] = {}
    unreadable: list[dict] = []
    per_family: dict[str, int] = {}
    n_shards = 0
    n_files = 0
    for stage_dir in stage_dirs:
        root = Path(stage_dir)
        state = _read_json(root / "harvest_state.json")
        if not isinstance(state, Mapping):
            continue
        for entry in state.get("shards") or []:
            record = read_shard(_shard_path(root, int(entry["index"])))
            n_shards += 1
            for row in record.get("files") or []:
                n_files += 1
                per_family[row["family"]] = per_family.get(row["family"], 0) + 1
                if row.get("checksum"):
                    file_checksums[f"{row['run']}/{row['relpath']}"] = row["checksum"]
            for row in record.get("unreadable") or []:
                unreadable.append(row)
                # A file that could be read but not parsed still had its bytes
                # hashed; binding it keeps the content digest honest about what
                # this study actually looked at.
                if row.get("checksum"):
                    file_checksums[f"{row['run']}/{row['relpath']}"] = row["checksum"]
            for run, identities in (record.get("identities") or {}).items():
                target = by_run.setdefault(str(run), ExclusionSet())
                for name, values in identities.items():
                    getattr(target, name).update(str(value) for value in values)

    return {
        "by_run": by_run,
        "file_checksums": file_checksums,
        "unreadable": unreadable,
        "n_shards": n_shards,
        "n_files": n_files,
        "per_family": dict(sorted(per_family.items())),
        "content_digest": payload_checksum(dict(sorted(file_checksums.items()))),
    }


def exclusion_from_reduction(
    reduction: Mapping, run_dirs: Sequence[str | os.PathLike[str]]
) -> ExclusionSet:
    """Assemble the module-level :class:`ExclusionSet` from reduced shards.

    The result is byte-comparable with what
    :func:`jlens.mmpilot.l32_resolution.harvest_excluded_identities` produces
    over the same identities: the digest is taken over the identity sets and
    the run *basenames*, never over how they were read.
    """
    exclusion = ExclusionSet()
    by_run: Mapping[str, ExclusionSet] = reduction["by_run"]
    for run_dir in run_dirs:
        root = Path(run_dir)
        if not root.is_dir():
            exclusion.sources.append(
                {
                    "run_dir": str(root),
                    "exists": False,
                    "n_files_read": 0,
                    "n_identities": 0,
                    "unreadable": [],
                }
            )
            continue
        exclusion.run_dirs.append(str(root))
        found = by_run.get(root.name, ExclusionSet())
        for name in (
            "image_ids",
            "group_ids",
            "sample_ids",
            "audio_paths",
            "captions",
            "media_checksums",
        ):
            getattr(exclusion, name).update(getattr(found, name))
        exclusion.sources.append(
            {
                "run_dir": str(root),
                "exists": True,
                "n_files_read": sum(
                    1
                    for key in reduction["file_checksums"]
                    if key.startswith(f"{root.name}/")
                ),
                "n_identities": found.n_identities,
                "unreadable": [
                    row
                    for row in reduction["unreadable"]
                    if str(row.get("run")) == root.name
                ],
            }
        )
    return exclusion


# ========================================================= completeness proof


def prove_completeness(
    plan: Mapping,
    reduction: Mapping,
    *,
    fallback_used: bool,
) -> dict:
    """Compare what was recovered against the completed runs' own records.

    Completeness is *proven*, not assumed: each run's recovered distinct group
    and image counts are checked against the population that run wrote down for
    itself. A run whose anchor is missing is never called complete — an
    unanchored count agrees only with itself.
    """
    by_run: Mapping[str, ExclusionSet] = reduction["by_run"]
    runs: list[dict] = []
    for entry in plan["runs"]:
        used = families_used(entry)
        skipped = [name for name in ALL_FAMILIES if name not in used]
        found = by_run.get(entry["run"], ExclusionSet())
        expected = entry["expected"]
        n_groups = len(found.group_ids)
        n_images = len(found.image_ids)
        shortfall: dict[str, int] = {}
        if expected["n_groups"] is not None and n_groups < expected["n_groups"]:
            shortfall["group_ids"] = expected["n_groups"] - n_groups
        if (
            expected["n_distinct_images"] is not None
            and n_images < expected["n_distinct_images"]
        ):
            shortfall["image_ids"] = expected["n_distinct_images"] - n_images
        anchored = expected["n_groups"] is not None
        runs.append(
            {
                "run": entry["run"],
                "strategy": entry["strategy"],
                "expected_group_count": expected["n_groups"],
                "expected_image_count": expected["n_distinct_images"],
                "expected_from": expected["source"],
                "group_ids_recovered": n_groups,
                "image_ids_recovered": n_images,
                "audio_paths_recovered": len(found.audio_paths),
                "captions_recovered": len(found.captions),
                "source_artifacts": sorted(
                    pattern
                    for family in used
                    for pattern in FAMILY_PATTERNS[family]
                ),
                "families_used": list(used),
                "families_skipped": list(skipped),
                "n_files_read": sum(
                    1
                    for key in reduction["file_checksums"]
                    if key.startswith(f"{entry['run']}/")
                ),
                "anchored": anchored,
                "complete": bool(anchored and not shortfall),
                "shortfall": shortfall,
            }
        )

    missing = [
        row
        for row in reduction["unreadable"]
        if row.get("error")
    ]
    return {
        "schema": COMPLETENESS_SCHEMA,
        "strategy_version": SOURCE_STRATEGY_VERSION,
        "complete": all(row["complete"] for row in runs) and bool(runs),
        "fallback_required": bool(fallback_used),
        "runs": runs,
        "unit_families_skipped": sorted(
            {family for row in runs for family in row["families_skipped"]}
        ),
        "why_skipped_families_cannot_add_identities": SKIP_JUSTIFICATION,
        "capability_units_are_not_a_population_source": plan[
            "capability_units_are_not_a_population_source"
        ],
        "missing_or_invalid_units": missing[:50],
        "n_missing_or_invalid_units": len(missing),
        "content_digest": reduction["content_digest"],
        "n_shards": reduction["n_shards"],
        "n_files_read": reduction["n_files"],
        "files_by_family": reduction["per_family"],
    }


def assert_complete(proof: Mapping) -> dict:
    """Refuse an exclusion set that was never shown to cover the population."""
    if not proof.get("complete"):
        shortfalls = {
            row["run"]: {
                "expected_group_count": row["expected_group_count"],
                "group_ids_recovered": row["group_ids_recovered"],
                "shortfall": row["shortfall"],
                "anchored": row["anchored"],
            }
            for row in proof.get("runs", [])
            if not row["complete"]
        }
        raise CompletenessNotProven(
            "the exclusion set could not be proven to cover every media unit "
            "the completed runs spent, even after the fallback scan. "
            "Independence would then be an assumption rather than a verified "
            "property, and a silently narrowed exclusion set is exactly the "
            "failure this study exists to avoid.\n"
            + json.dumps(shortfalls, indent=2)
        )
    return dict(proof)


# ============================================================== orchestration


def families_used(plan_row: Mapping) -> list[str]:
    """The families a run was actually harvested from, minimal plus fallback."""
    return [
        *plan_row.get("minimal_families", ()),
        *plan_row.get("fallback_families", ()),
    ]


def _cached_plan(
    cache_dir: Path,
    run_dirs: Sequence[str | os.PathLike[str]],
    *,
    protected_prefixes: Sequence[str],
) -> dict:
    """The source plan, decided once and then reused from the cache.

    Re-deciding it costs a read of every completed run's ``split_provenance``
    and ``population_manifest`` on each resume. Small, but the point of this
    module is that a resumed session touches the completed runs as little as it
    possibly can — and with a complete cache, not at all.
    """
    path = cache_dir / "source_plan.json"
    stored = _read_json(path)
    if (
        isinstance(stored, Mapping)
        and stored.get("schema") == "jlens.mmpilot.l32_resolution_source_plan.v1"
    ):
        return json.loads(json.dumps(stored))
    plan = plan_sources(run_dirs)
    atomic_write_json(path, plan, protected_prefixes=protected_prefixes)
    return plan


def _reload_completed_preparation(
    root: Path,
    run_dirs: Sequence[str | os.PathLike[str]],
    fingerprint: Mapping,
    reporter: ProgressReporter,
) -> dict | None:
    """Rebuild the preparation record from Drive without reading any source unit.

    This is the finalized-cache path, and it deliberately does **no** source
    I/O at all — not even the enumeration a resumed, unfinished preparation
    does. The completed runs are checked separately and explicitly by
    :func:`verify_sources_unchanged` at the end of the study, which is where an
    immutability claim belongs; making the reload path re-enumerate would put a
    Drive traversal back into the session this whole module exists to keep
    fast.

    Returns ``None`` when the cache is not in a state that can be reloaded, so
    the caller falls through to the ordinary resumable path rather than
    guessing.
    """
    marker = preparation_is_complete(root)
    if marker is None or str(marker.get("preparation_digest")) != str(
        fingerprint["preparation_digest"]
    ):
        return None
    plan = _read_json(root / "source_plan.json")
    proof = _read_json(root / "completeness_proof.json")
    if not isinstance(plan, Mapping) or not isinstance(proof, Mapping):
        return None
    exclusion = load_exclusion_set(root)
    if exclusion is None:
        return None
    stage_dirs = [
        path
        for path in (root / "harvest_minimal", root / "harvest_fallback")
        if (path / "harvest_state.json").is_file()
    ]
    reduction = reduce_shards(stage_dirs)
    if exclusion.digest != exclusion_from_reduction(reduction, run_dirs).digest:
        raise PreparationIncompatible(
            f"{root / 'exclusion_set.json'} disagrees with the checkpoint shards "
            "it was reduced from. A completed preparation cache is immutable "
            "and this one is not internally consistent; refusing to reuse it."
        )
    inventories = [
        dict(stored)
        for stored in (
            _read_json(path / "source_inventory.json") for path in stage_dirs
        )
        if isinstance(stored, Mapping)
    ]
    reporter.emit("=" * 68)
    reporter.emit("PREPROCESSING REUSED FROM DRIVE — no source unit was read")
    reporter.emit(f"  preparation digest  {fingerprint['preparation_digest']}")
    reporter.emit(f"  exclusion digest    {exclusion.digest}")
    reporter.emit(f"  shards verified     {reduction['n_shards']}")
    reporter.emit(f"  files represented   {reduction['n_files']}")
    reporter.emit("=" * 68)
    return {
        "cache_dir": str(root),
        "plan": json.loads(json.dumps(plan)),
        "families_by_run": {
            row["run"]: families_used(row) for row in plan["runs"]
        },
        "inventories": inventories,
        "inventory": inventories[0] if inventories else None,
        "fallback_inventory": inventories[1] if len(inventories) > 1 else None,
        "minimal_state": _read_json(root / "harvest_minimal" / "harvest_state.json"),
        "fallback_state": _read_json(root / "harvest_fallback" / "harvest_state.json"),
        "reduction": reduction,
        "exclusion": exclusion,
        "completeness": json.loads(json.dumps(proof)),
        "files_computed_this_session": 0,
        "files_reused_from_drive": reduction["n_files"],
        "reused_complete_cache": True,
    }


def run_exclusion_preparation(
    cache_dir: str | os.PathLike[str],
    run_dirs: Sequence[str | os.PathLike[str]],
    *,
    fingerprint: Mapping,
    batch_files: int = DEFAULT_BATCH_FILES,
    checkpoint_seconds: float = DEFAULT_CHECKPOINT_SECONDS,
    progress: ProgressReporter | None = None,
    protected_prefixes: Sequence[str] = (),
    on_corrupt: str = "quarantine",
    allow_fallback: bool = True,
) -> dict:
    """Inventory, harvest, reduce and prove — resumable at every step.

    Returns a record carrying the :class:`ExclusionSet`, the completeness
    proof, both stage states and whether anything was read at all this session.
    A cache that is already complete and compatible reads **zero** source units.
    """
    root = Path(cache_dir)
    assert_write_allowed(root, protected_prefixes=protected_prefixes)
    root.mkdir(parents=True, exist_ok=True)
    reporter = progress or _null_progress()

    reused = _reload_completed_preparation(root, run_dirs, fingerprint, reporter)
    if reused is not None:
        return reused

    atomic_write_json(
        root / "preparation_fingerprint.json",
        dict(fingerprint),
        protected_prefixes=protected_prefixes,
    )

    plan = _cached_plan(root, run_dirs, protected_prefixes=protected_prefixes)
    minimal_families = {row["run"]: row["minimal_families"] for row in plan["runs"]}
    minimal_dir = root / "harvest_minimal"

    inventory = _cached_inventory(
        minimal_dir,
        run_dirs,
        minimal_families,
        label="minimal",
        progress=reporter,
        protected_prefixes=protected_prefixes,
    )
    minimal_state = harvest_checkpointed(
        minimal_dir,
        inventory,
        fingerprint,
        stage="minimal",
        batch_files=batch_files,
        checkpoint_seconds=checkpoint_seconds,
        progress=reporter,
        protected_prefixes=protected_prefixes,
        on_corrupt=on_corrupt,
    )

    stage_dirs = [minimal_dir]
    reduction = reduce_shards(stage_dirs)
    proof = prove_completeness(plan, reduction, fallback_used=False)

    fallback_state = None
    fallback_inventory = None
    if not proof["complete"] and allow_fallback:
        reporter.emit(
            "completeness NOT established from the minimal sources; escalating "
            f"to the checkpointed fallback scan over {list(FALLBACK_FAMILIES)}"
        )
        fallback_families = {
            row["run"]: [
                family
                for family in FALLBACK_FAMILIES
                if family not in row["minimal_families"]
            ]
            for row in plan["runs"]
        }
        fallback_dir = root / "harvest_fallback"
        fallback_inventory = _cached_inventory(
            fallback_dir,
            run_dirs,
            fallback_families,
            label="fallback",
            progress=reporter,
            protected_prefixes=protected_prefixes,
        )
        fallback_state = harvest_checkpointed(
            fallback_dir,
            fallback_inventory,
            fingerprint,
            stage="fallback",
            batch_files=batch_files,
            checkpoint_seconds=checkpoint_seconds,
            progress=reporter,
            protected_prefixes=protected_prefixes,
            on_corrupt=on_corrupt,
        )
        stage_dirs.append(fallback_dir)
        for row in plan["runs"]:
            row["fallback_families"] = list(fallback_families[row["run"]])
        reduction = reduce_shards(stage_dirs)
        proof = prove_completeness(plan, reduction, fallback_used=True)

    exclusion = exclusion_from_reduction(reduction, run_dirs)
    inventories = [inventory] + ([fallback_inventory] if fallback_inventory else [])
    families_by_run = {row["run"]: families_used(row) for row in plan["runs"]}

    atomic_write_json(
        root / "source_plan.json", plan, protected_prefixes=protected_prefixes
    )
    atomic_write_json(
        root / "completeness_proof.json", proof, protected_prefixes=protected_prefixes
    )
    atomic_write_json(
        root / "exclusion_set.json",
        exclusion.to_dict(),
        protected_prefixes=protected_prefixes,
    )
    atomic_write_json(
        root / "exclusion_set_checksum.json",
        {
            "schema": "jlens.mmpilot.l32_resolution_exclusion_checksum.v1",
            "exclusion_digest": exclusion.digest,
            "payload_checksum": payload_checksum(exclusion.to_dict()),
            "content_digest": reduction["content_digest"],
            "harvest_version": EXCLUSION_HARVEST_VERSION,
        },
        protected_prefixes=protected_prefixes,
    )

    return {
        "cache_dir": str(root),
        "plan": plan,
        "families_by_run": families_by_run,
        "inventories": inventories,
        "inventory": inventory,
        "fallback_inventory": fallback_inventory,
        "minimal_state": minimal_state,
        "fallback_state": fallback_state,
        "reduction": reduction,
        "exclusion": exclusion,
        "completeness": proof,
        "files_computed_this_session": int(
            minimal_state.get("files_computed_this_session", 0)
        )
        + int((fallback_state or {}).get("files_computed_this_session", 0)),
        "files_reused_from_drive": int(
            minimal_state.get("files_reused_from_drive", 0)
        )
        + int((fallback_state or {}).get("files_reused_from_drive", 0)),
        "reused_complete_cache": False,
    }


def _cached_inventory(
    stage_dir: Path,
    run_dirs: Sequence[str | os.PathLike[str]],
    families_by_run: Mapping[str, Sequence[str]],
    *,
    label: str,
    progress: ProgressReporter,
    protected_prefixes: Sequence[str],
) -> dict:
    """The stage's inventory, re-enumerated each session and compared.

    Enumeration is deliberately *not* cached. It is a glob and a stat over only
    the identity-bearing families — cheap next to reading them — and it is the
    single place a resumed session can notice that a completed run gained,
    lost or altered an artifact that contributes identities. A cached
    enumeration would make the harvest state's inventory check unreachable, and
    a new activation unit would then be silently left out of the exclusion set.

    Raises:
        PreparationIncompatible: When the enumeration disagrees with the one
            this preparation's checkpoints were taken against.
    """
    path = stage_dir / "source_inventory.json"
    inventory = build_source_inventory(
        run_dirs, families_by_run, progress=progress, label=label
    )
    stored = _read_json(path)
    if isinstance(stored, Mapping) and stored.get("schema") == INVENTORY_SCHEMA:
        if stored.get("inventory_digest") != inventory["inventory_digest"]:
            before = {
                (row["run"], row["relpath"]): row for row in stored.get("entries", [])
            }
            after = {
                (row["run"], row["relpath"]): row for row in inventory["entries"]
            }
            raise PreparationIncompatible(
                f"the {label} source inventory changed since this preparation's "
                "checkpoints were taken, so resuming it would harvest a "
                "different set of artifacts than the shards on disk were built "
                "from.\n"
                f"  inventory_digest stored={stored.get('inventory_digest')} "
                f"recomputed={inventory['inventory_digest']}\n"
                f"  appeared: "
                f"{sorted(f'{r}/{p}' for r, p in set(after) - set(before))[:10]}\n"
                f"  vanished: "
                f"{sorted(f'{r}/{p}' for r, p in set(before) - set(after))[:10]}\n"
                "A completed run's identity-bearing artifacts are supposed to be "
                "immutable. Preprocessing under the changed inputs belongs in a "
                "new preparation cache directory."
            )
        progress.note(
            f"{label} inventory unchanged: {inventory['n_files']} file(s)  "
            f"digest {str(inventory['inventory_digest'])[:23]}..."
        )
        return inventory
    atomic_write_json(path, inventory, protected_prefixes=protected_prefixes)
    return inventory


# ================================================= the prepared selection half


def save_prepared_selection(
    cache_dir: str | os.PathLike[str],
    payload: Mapping,
    *,
    protected_prefixes: Sequence[str] = (),
) -> dict:
    """Persist everything downstream of the exclusion set, atomically.

    The point is the sentence "no required scientific variable survives only in
    Python memory": a fresh process rebuilds the pool, the ranking, the frozen
    concept feasibility and the selected population from here and from the
    cached expanded manifest, and checks each against a digest rather than
    trusting the file.
    """
    record = {
        "schema": SELECTION_SCHEMA,
        "written_utc": utc_now(),
        **dict(payload),
    }
    atomic_write_json(
        Path(cache_dir) / "prepared_selection.json",
        record,
        protected_prefixes=protected_prefixes,
    )
    return record


def load_prepared_selection(cache_dir: str | os.PathLike[str]) -> dict | None:
    """The persisted selection, or ``None`` when preprocessing has not got there."""
    stored = _read_json(Path(cache_dir) / "prepared_selection.json")
    if isinstance(stored, Mapping) and stored.get("schema") == SELECTION_SCHEMA:
        return dict(stored)
    return None


def load_exclusion_set(cache_dir: str | os.PathLike[str]) -> ExclusionSet | None:
    """Rehydrate the persisted exclusion set, or ``None``.

    Refuses a payload whose stored digest disagrees with the digest of the
    identities it actually contains: a corrupted or hand-edited exclusion file
    must not be able to shrink the exclusion set quietly.
    """
    stored = _read_json(Path(cache_dir) / "exclusion_set.json")
    if not isinstance(stored, Mapping):
        return None
    exclusion = ExclusionSet()
    for name in (
        "image_ids",
        "group_ids",
        "sample_ids",
        "audio_paths",
        "captions",
        "media_checksums",
    ):
        getattr(exclusion, name).update(str(value) for value in stored.get(name) or [])
    exclusion.run_dirs = list(stored.get("run_dirs") or [])
    exclusion.sources = [dict(entry) for entry in stored.get("sources") or []]
    if stored.get("exclusion_digest") != exclusion.digest:
        raise PreparationRefused(
            f"{Path(cache_dir) / 'exclusion_set.json'} does not match its own "
            f"digest (stored {stored.get('exclusion_digest')}, recomputed "
            f"{exclusion.digest}). A preparation cache that cannot verify its "
            "own exclusion set is not reused."
        )
    return exclusion


def finalize_preparation(
    cache_dir: str | os.PathLike[str],
    payload: Mapping,
    *,
    protected_prefixes: Sequence[str] = (),
) -> dict:
    """Mark a preparation complete, and refuse to disagree with itself.

    A completed cache is immutable. If the same inputs are preprocessed again
    and produce a different exclusion or population digest, that is a bug in
    something and it is reported as a refusal rather than overwritten.
    """
    path = Path(cache_dir) / "preparation_complete.json"
    record = {
        "schema": COMPLETE_SCHEMA,
        "preparation_version": PREPARATION_VERSION,
        "written_utc": utc_now(),
        **dict(payload),
    }
    stored = _read_json(path)
    if isinstance(stored, Mapping):
        differing = {
            key: (stored.get(key), record.get(key))
            for key in (
                "preparation_digest",
                "exclusion_digest",
                "population_digest",
                "pool_digest",
                "ranking_digest",
                "frozen_concept_feasibility_digest",
            )
            if key in record and stored.get(key) != record.get(key)
        }
        if differing:
            raise PreparationIncompatible(
                "a completed preparation cache is immutable, and recomputing it "
                "from the same inputs produced different digests:\n"
                + json.dumps(
                    {k: {"stored": a, "recomputed": b} for k, (a, b) in differing.items()},
                    indent=2,
                )
                + "\nRefusing to overwrite it. Preprocessing is supposed to be a "
                "function of its inputs; if it is not, the study cannot rely on "
                "the population either."
            )
        return dict(stored)
    atomic_write_json(path, record, protected_prefixes=protected_prefixes)
    return record


def preparation_is_complete(cache_dir: str | os.PathLike[str]) -> dict | None:
    """The completion marker, or ``None`` if preprocessing is unfinished."""
    stored = _read_json(Path(cache_dir) / "preparation_complete.json")
    if isinstance(stored, Mapping) and stored.get("schema") == COMPLETE_SCHEMA:
        return dict(stored)
    return None


# ===================================================================== report


def render_preprocessing_report(record: Mapping) -> str:
    """``preprocessing_report.md`` — what was read, what was reused, what was proven."""
    proof = record.get("completeness") or {}
    lines = [
        "# L32 convergence resolution — preprocessing report",
        "",
        f"- preparation version: `{PREPARATION_VERSION}`",
        f"- preparation digest: `{record.get('preparation_digest')}`",
        f"- cache directory: `{record.get('cache_dir')}`",
        f"- exclusion digest: `{record.get('exclusion_digest')}`",
        f"- content digest: `{proof.get('content_digest')}`",
        f"- files read: {proof.get('n_files_read')} across "
        f"{proof.get('n_shards')} checkpoint shard(s)",
        f"- computed this session: {record.get('files_computed_this_session')}",
        f"- reused from Drive: {record.get('files_reused_from_drive')}",
        "",
        "## Checkpoint and resume semantics",
        "",
        f"- bounded work unit: at most {record.get('batch_files', DEFAULT_BATCH_FILES)} "
        f"files or {record.get('checkpoint_seconds', DEFAULT_CHECKPOINT_SECONDS)} "
        "seconds, whichever comes first",
        "- each unit is written as one atomically replaced, checksummed gzip shard "
        "**before** the cursor advances",
        "- stopping at any moment loses at most the single in-flight unit and never "
        "restarts from file zero while a valid checkpoint exists",
        "- a torn shard is quarantined and only its own batch is recomputed",
        "",
        "## Source artifacts and the completeness proof",
        "",
        "| run | strategy | expected groups | groups recovered | images recovered | complete |",
        "|---|---|---|---|---|---|",
    ]
    for row in proof.get("runs", []):
        lines.append(
            f"| `{row['run']}` | {row['strategy']} | {row['expected_group_count']} "
            f"| {row['group_ids_recovered']} | {row['image_ids_recovered']} "
            f"| {row['complete']} |"
        )
    lines += [
        "",
        f"- fallback scan required: {proof.get('fallback_required')}",
        f"- unit families skipped: {proof.get('unit_families_skipped')}",
        "",
        "Why the skipped families cannot hold an identity the harvested ones missed:",
        "",
        f"> {proof.get('why_skipped_families_cannot_add_identities')}",
        "",
        f"- missing or invalid units: {proof.get('n_missing_or_invalid_units')}",
        "",
        "## Files read, by family",
        "",
    ]
    for family, count in (proof.get("files_by_family") or {}).items():
        lines.append(f"- `{family}`: {count}")
    lines += [
        "",
        "## Immutability",
        "",
        "Completed runs are never written to. Every write path in this module "
        "passes `assert_write_allowed`, which refuses a path through a protected "
        "run prefix outright, and the harvested families are re-enumerated "
        "afterwards and compared by name, size and mtime against sha256 content "
        "digests taken during the single harvest read.",
        "",
        "## Scope",
        "",
        "This report describes preprocessing only. It contains no model output "
        "and is not evidence about Gemma, about layer 32, or about convergence.",
        "",
    ]
    return "\n".join(lines)
