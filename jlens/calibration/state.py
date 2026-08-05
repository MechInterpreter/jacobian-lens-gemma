# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Per-unit calibration state, atomic persistence, and fingerprint-gated resume.

A 1,000-prompt fit is 20-30 hours of L4 time (``jlens.calibration.plan``), so
every stage of this study will be interrupted. Two rules keep the resume honest,
and they are the same two the multimodal pilot uses:

1. **Checksummed units.** Each file stores its payload beside a checksum of the
   payload's canonical JSON. A torn or edited file fails the check and is
   treated as missing rather than trusted.
2. **A fingerprint gate.** The run directory records what its results were
   produced *from*. A rerun under a different fingerprint is refused with a
   field-by-field diff, never silently mixed.

Only pure serialisation helpers are borrowed from ``jlens.mmpilot.store``
(canonical JSON, checksums, filesystem-safe keys, the resume error type). The
stage vocabulary is defined here rather than widening the multimodal package's,
following the precedent set by :mod:`jlens.native_readout`. **No SpokenCOCO,
image, audio or multimodal data is reachable through this import.**

What the fingerprint binds, and why each entry is there:

* model and tokenizer revision — a different checkpoint is a different `J_l`;
* the capture plan digest — layers, target, ``dim_batch``, ``max_seq_len``,
  ``skip_first``; changing any of them changes the estimator;
* the corpus manifest checksum — including all three split checksums, so a
  reshuffled split cannot reuse a stored unit;
* the gate digest — editing a threshold invalidates results rather than
  rescoring them, which is what makes the gate predeclared in practice;
* the plateau-rule digest — same reason, for the scale decision;
* the scale schedule — a run that was going to stop at 1k is not the same run
  as one that will continue to 10k, because the nested prefix must match.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from jlens.mmpilot.store import (
    IncompatibleStateError,
    canonical_json,
    payload_checksum,
    safe_key,
)

__all__ = [
    "CALIBRATION_STAGES",
    "CalibrationFingerprint",
    "CalibrationStore",
    "IncompatibleStateError",
    "canonical_json",
    "payload_checksum",
    "safe_key",
]

#: Stages that write per-unit artifacts, ordered as the notebook runs them.
CALIBRATION_STAGES = (
    "corpus",
    "fit_diagnostics",
    "scale_snapshot",
    "validation",
    "scale_comparison",
    "confirmation",
    "publication",
)

SCHEMA = "jlens.calibration.unit.v1"


@dataclass(frozen=True)
class CalibrationFingerprint:
    """What a calibration run's artifacts were produced from.

    Any change invalidates every stored unit. ``extra`` carries anything else a
    caller wants results bound to and participates in the digest.
    """

    mode: str
    protocol_version: str
    model_repo_id: str
    model_revision: str
    tokenizer_revision: str
    capture_plan_digest: str
    corpus_manifest_checksum: str
    gate_digest: str
    plateau_rule_digest: str
    scale_points: tuple[int, ...]
    artifact_format_version: str
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["scale_points"] = [int(point) for point in self.scale_points]
        return payload

    @property
    def digest(self) -> str:
        return payload_checksum(self.to_dict())

    def differences(self, other: CalibrationFingerprint | dict) -> list[str]:
        """Field-by-field differences (empty when identical)."""
        theirs = (
            other.to_dict() if isinstance(other, CalibrationFingerprint) else dict(other)
        )
        mine = self.to_dict()
        out: list[str] = []
        for key in sorted(set(mine) | set(theirs)):
            a, b = mine.get(key, "<absent>"), theirs.get(key, "<absent>")
            if canonical_json(a) != canonical_json(b):
                out.append(f"{key}: stored={b!r} requested={a!r}")
        return out


class CalibrationStore:
    """Per-unit JSON artifacts under ``root``, gated by a fingerprint.

    Args:
        root: Run directory, created if missing. On Colab this lives on Drive.
        fingerprint: What this run's results are bound to.
    """

    def __init__(
        self, root: str | os.PathLike[str], fingerprint: CalibrationFingerprint
    ) -> None:
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

        Returns ``"starting"`` for a fresh directory, ``"resuming"`` for one
        whose stored fingerprint matches.

        Raises:
            IncompatibleStateError: If a stored fingerprint disagrees. The
                message lists every differing field; the caller is expected to
                choose a new run directory rather than mix artifacts.
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
                f"{self.root} holds calibration artifacts from a different "
                f"configuration; refusing to reuse them.\n  {diffs}\n"
                "Point the run at a new directory (or delete this one yourself)."
            )
        self.status = "resuming"
        return self.status

    # ---------------------------------------------------------------- units

    def stage_dir(self, stage: str) -> Path:
        if stage not in CALIBRATION_STAGES:
            raise ValueError(
                f"unknown calibration stage {stage!r}; known stages are "
                f"{CALIBRATION_STAGES}"
            )
        return self.root / "units" / stage

    def unit_path(self, stage: str, key: str) -> Path:
        return self.stage_dir(stage) / f"{key}.json"

    def save(self, stage: str, key: str, payload: dict) -> Path:
        """Atomically write one unit. Returns the path written."""
        path = self.unit_path(stage, key)
        self._write_json(
            path,
            {
                "schema": SCHEMA,
                "stage": stage,
                "key": key,
                "fingerprint_digest": self.fingerprint.digest,
                "unit_checksum": payload_checksum(payload),
                "payload": payload,
            },
        )
        return path

    def load(self, stage: str, key: str) -> dict | None:
        """A stored unit's payload, or ``None`` if absent, torn,
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
        """Whether a checksum-valid unit exists, so it can be skipped."""
        return self.load(stage, key) is not None

    def keys(self, stage: str) -> list[str]:
        directory = self.stage_dir(stage)
        if not directory.is_dir():
            return []
        return sorted(path.stem for path in directory.glob("*.json"))

    def load_all(self, stage: str) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for key in self.keys(stage):
            payload = self.load(stage, key)
            if payload is not None:
                out[key] = payload
        return out

    # -------------------------------------------------------------- paths

    @property
    def checkpoint_path(self) -> Path:
        """Where :func:`jlens.fitting.fit` writes its resumable accumulator."""
        return self.root / "checkpoints" / "jacobian_sum.pt"

    def snapshot_path(self, scale: int) -> Path:
        return self.root / "artifacts" / f"lens.scale{int(scale)}.pt"

    def published_path(self, layer: int, scale: int) -> Path:
        return (
            self.root
            / "artifacts"
            / "published"
            / f"lens.layer{int(layer)}.scale{int(scale)}.validated.pt"
        )

    # --------------------------------------------------------------- report

    def status_report(self, stages: Sequence[str] = CALIBRATION_STAGES) -> dict:
        return {
            "run_dir": str(self.root),
            "status": self.status or "unopened",
            "fingerprint_digest": self.fingerprint.digest,
            "completed_units": {stage: len(self.load_all(stage)) for stage in stages},
            "invalid_units": list(self.invalid_units),
            "checkpoint_present": self.checkpoint_path.is_file(),
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
