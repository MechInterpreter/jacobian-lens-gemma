# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Atomic per-unit results and fingerprint-gated resume.

The pilot runs on a Colab runtime that can disconnect at any moment, so every
stage writes one small JSON file per unit of work (one sample, one modality,
one layer, one intervention condition) as soon as that unit finishes. A rerun
reloads the units it can verify and recomputes only the rest.

Two rules keep resume honest:

1. **Checksummed units.** Each file stores the payload next to a checksum of
   its canonical JSON. A torn or edited file fails the check and is treated as
   missing rather than trusted.
2. **A fingerprint gate.** The run directory records what the results were
   produced *from* — model revision, processor revision, layers, lens
   checksum, dataset manifest checksum, split, intervention config. A rerun
   with a different fingerprint is refused, not silently mixed.

There is no workflow engine here: a stage asks :meth:`UnitStore.has` before
computing and calls :meth:`UnitStore.save` after. That is the whole protocol.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

#: Stages that write per-unit artifacts. Ordered as the notebook runs them.
STAGES = (
    "capability",
    "activation",
    "jspace",
    "reconstruction_control",
    "direction",
    "intervention",
    "metric",
)

_SAFE_KEY = re.compile(r"[^A-Za-z0-9._-]+")


class IncompatibleStateError(RuntimeError):
    """Raised when a run directory was produced under a different fingerprint."""


def canonical_json(payload: Any) -> str:
    """Deterministic JSON for hashing: sorted keys, no incidental whitespace."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def payload_checksum(payload: Any) -> str:
    """``sha256:<hex>`` over :func:`canonical_json` of ``payload``."""
    return "sha256:" + hashlib.sha256(canonical_json(payload).encode()).hexdigest()


def safe_key(*parts: object) -> str:
    """Filesystem-safe unit key built from ``parts`` joined with ``__``.

    Characters outside ``[A-Za-z0-9._-]`` are replaced with ``-``. Long keys
    are truncated and disambiguated with a hash of the full key, so two
    distinct units can never collide on disk.
    """
    raw = "__".join(str(part) for part in parts)
    cleaned = _SAFE_KEY.sub("-", raw).strip("-") or "unit"
    if len(cleaned) <= 120:
        return cleaned
    digest = hashlib.sha256(raw.encode()).hexdigest()[:12]
    return f"{cleaned[:107]}-{digest}"


@dataclass(frozen=True)
class RunFingerprint:
    """What a run's artifacts were produced from.

    Any change here invalidates every stored unit. ``extra`` carries anything
    else a caller wants to bind results to (it participates in the digest).
    """

    mode: str
    model_repo_id: str
    model_revision: str
    processor_revision: str
    layers: tuple[int, ...]
    lens_checksum: str
    manifest_checksum: str
    split_id: str
    intervention_config: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["layers"] = list(self.layers)
        return payload

    @property
    def digest(self) -> str:
        return payload_checksum(self.to_dict())

    def differences(self, other: RunFingerprint | dict) -> list[str]:
        """Human-readable field-by-field differences (empty when identical)."""
        theirs = other.to_dict() if isinstance(other, RunFingerprint) else dict(other)
        mine = self.to_dict()
        out: list[str] = []
        for key in sorted(set(mine) | set(theirs)):
            a, b = mine.get(key, "<absent>"), theirs.get(key, "<absent>")
            if canonical_json(a) != canonical_json(b):
                out.append(f"{key}: stored={b!r} requested={a!r}")
        return out


class UnitStore:
    """Per-unit JSON artifacts under ``root``, gated by a :class:`RunFingerprint`.

    Args:
        root: Run directory (created if missing). On Colab this lives on Drive.
        fingerprint: What this run's results are bound to.

    Attributes:
        status: ``"starting"`` or ``"resuming"``, set by :meth:`open`.
        invalid_units: Unit paths that failed their checksum and were dropped.
    """

    def __init__(self, root: str | os.PathLike[str], fingerprint: RunFingerprint) -> None:
        self.root = Path(root)
        self.fingerprint = fingerprint
        self.status: str | None = None
        self.invalid_units: list[str] = []

    # ------------------------------------------------------------- lifecycle

    @property
    def fingerprint_path(self) -> Path:
        return self.root / "fingerprint.json"

    def open(self) -> str:
        """Create or validate the run directory.

        Returns ``"starting"`` for a fresh directory and ``"resuming"`` for one
        whose stored fingerprint matches.

        Raises:
            IncompatibleStateError: If a stored fingerprint disagrees. The
                message lists every differing field; the caller is expected to
                pick a new run directory rather than mix artifacts.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.fingerprint_path.is_file():
            self._write_json(self.fingerprint_path, self.fingerprint.to_dict())
            self.status = "starting"
            return self.status
        stored = json.loads(self.fingerprint_path.read_text(encoding="utf-8"))
        stored.pop("written_utc", None)
        if payload_checksum(stored) != self.fingerprint.digest:
            diffs = "\n  ".join(self.fingerprint.differences(stored))
            raise IncompatibleStateError(
                f"{self.root} holds artifacts from a different configuration; "
                f"refusing to reuse them.\n  {diffs}\n"
                "Point the run at a new directory (or delete this one yourself)."
            )
        self.status = "resuming"
        return self.status

    # ---------------------------------------------------------------- units

    def stage_dir(self, stage: str) -> Path:
        if stage not in STAGES:
            raise ValueError(f"unknown stage {stage!r}; known stages are {STAGES}")
        return self.root / "units" / stage

    def unit_path(self, stage: str, key: str) -> Path:
        return self.stage_dir(stage) / f"{key}.json"

    def save(self, stage: str, key: str, payload: dict) -> Path:
        """Atomically write one unit. Returns the path written."""
        path = self.unit_path(stage, key)
        self._write_json(
            path,
            {
                "schema": "jlens.mmpilot.unit.v1",
                "stage": stage,
                "key": key,
                "fingerprint_digest": self.fingerprint.digest,
                "unit_checksum": payload_checksum(payload),
                "payload": payload,
            },
        )
        return path

    def load(self, stage: str, key: str) -> dict | None:
        """Return a stored unit's payload, or ``None`` if it is absent, torn,
        checksum-invalid, or bound to a different fingerprint."""
        path = self.unit_path(stage, key)
        if not path.is_file():
            return None
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            payload = record["payload"]
            valid = (
                record.get("unit_checksum") == payload_checksum(payload)
                and record.get("fingerprint_digest") == self.fingerprint.digest
            )
        except (json.JSONDecodeError, KeyError, TypeError, UnicodeDecodeError):
            valid, payload = False, None
        if not valid:
            self.invalid_units.append(str(path))
            return None
        return payload

    def has(self, stage: str, key: str) -> bool:
        """Whether a checksum-valid unit already exists (so it can be skipped)."""
        return self.load(stage, key) is not None

    def keys(self, stage: str) -> list[str]:
        """Sorted keys with a file present (validity not checked)."""
        directory = self.stage_dir(stage)
        if not directory.is_dir():
            return []
        return sorted(path.stem for path in directory.glob("*.json"))

    def load_all(self, stage: str) -> dict[str, dict]:
        """Every valid payload in ``stage``, keyed by unit key."""
        out: dict[str, dict] = {}
        for key in self.keys(stage):
            payload = self.load(stage, key)
            if payload is not None:
                out[key] = payload
        return out

    # --------------------------------------------------------------- report

    def status_report(self, stages: Sequence[str] = STAGES) -> dict:
        """Counts per stage plus the resume status, for the notebook's final cell."""
        return {
            "run_dir": str(self.root),
            "status": self.status or "unopened",
            "fingerprint_digest": self.fingerprint.digest,
            "completed_units": {stage: len(self.load_all(stage)) for stage in stages},
            "invalid_units": list(self.invalid_units),
        }

    # -------------------------------------------------------------- helpers

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
        tmp.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        os.replace(tmp, path)
