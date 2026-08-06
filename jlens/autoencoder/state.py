# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Stage state, resume identity, and safe interruption.

A Colab runtime is a machine that can be taken away mid-sentence. This module is
what makes that survivable: every expensive stage records *where it is* and *what
it was doing it against*, so a reconnected session can tell the difference
between "continue" and "you changed the experiment, start over".

Three pieces:

**Identity.** :class:`StageIdentity` is the answer to "may this checkpoint be
resumed here?". It splits into a **semantic** part — model revision, lens
checksum, source layer, pursuit settings, split policy, dataset identity,
architecture dimensions — and a **non-semantic** part (epoch counts, batch
sizes, logging, cadence). A semantic mismatch is fatal and
:data:`OVERRIDE_FORBIDDEN_FIELDS` keeps it that way: no flag reopens it, because
resuming across one of those changes produces a run whose artifacts describe a
configuration that never actually produced them. Non-semantic drift is refused
by default and can be waived explicitly.

**State.** :class:`StageState` owns ``RUN_DIR/state/<stage>/`` — one status of
:data:`STAGE_STATUSES`, a progress document, and an atomic completion marker
carrying the config and artifact hashes. Completed stages are skipped on rerun
unless forced or unless their artifacts no longer validate.

**Interruption.** :class:`InterruptGuard` installs SIGINT/SIGTERM handlers that
do nothing but set a flag. Serializing torch state inside a signal handler is
how you get a half-written checkpoint written by a re-entered handler; the
handler flips a bit and the training loop checkpoints at the next safe boundary,
where it already knows the epoch, the batch, and the RNG are consistent.

None of this survives ``SIGKILL`` or a hard VM teardown — nothing running inside
the VM can. What limits the loss there is checkpoint *cadence*, which is why
every stage exposes one.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import signal
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import torch

from jlens.autoencoder.errors import AutoencoderError

STATE_SCHEMA = "jlens.autoencoder.state.v1"
CHECKPOINT_SCHEMA_VERSION = 2

#: The stages that own a state directory, in execution order.
STAGES = (
    "dataset",
    "reconstructor",
    "adapter_warm",
    "adapter_preference",
    "evaluation",
)

#: Every status a stage can report. ``incompatible`` is distinct from ``failed``:
#: the work is intact, it simply does not belong to the current configuration.
STAGE_STATUSES = (
    "not_started",
    "in_progress",
    "interrupted",
    "complete",
    "incompatible",
    "failed",
)

#: Why a checkpoint was written. Recorded so a resume can say what it is
#: resuming *from* rather than only where.
CHECKPOINT_REASONS = (
    "periodic",
    "epoch_complete",
    "keyboard_interrupt",
    "stage_complete",
)

#: Identity fields that no override may waive. These are the inputs whose change
#: makes previously computed work describe a different experiment.
OVERRIDE_FORBIDDEN_FIELDS = (
    "model_repo_id",
    "model_revision",
    "lens_sha256",
    "lens_run_dir_name",
    "source_layer",
    "pursuit",
    "split_policy",
    "dataset_identity",
    "architecture",
)


class StageInterrupted(Exception):
    """Raised at a safe boundary after a stop was requested.

    Carries the checkpoint (or progress) path that was written, so the caller can
    print an exact resume instruction instead of a generic apology.
    """

    def __init__(self, message: str, *, stage: str, checkpoint_path: str | None = None):
        super().__init__(message)
        self.stage = stage
        self.checkpoint_path = checkpoint_path


class IncompatibleState(AutoencoderError):
    """Existing state does not belong to the current configuration."""


# ----------------------------------------------------------------- atomic I/O


def _tmp_name(path: str) -> str:
    """A temp name that is recognisably incomplete.

    The ``.tmp.`` infix is what :func:`iter_valid_files` filters on, so a crash
    between write and rename leaves a file no reader will ever mistake for a
    finished one.
    """
    return f"{path}.tmp.{os.getpid()}"


def atomic_write_bytes(path: str, payload: bytes) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp_path = _tmp_name(path)
    with open(tmp_path, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)
    return path


def atomic_write_json(path: str, payload: Any) -> str:
    """Write pretty JSON atomically, with a stable key order.

    ``sort_keys`` is not cosmetic: shard and manifest files are hashed, and a
    dict whose iteration order depends on insertion would make two runs that
    computed identical values disagree on their checksums.
    """
    body = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=str)
    return atomic_write_bytes(path, (body + "\n").encode("utf-8"))


def atomic_torch_save(payload: dict, path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp_path = _tmp_name(path)
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)
    return path


def read_json_if_valid(path: str) -> dict | None:
    """Parse ``path``, or return ``None`` if it is missing or malformed.

    A truncated shard is not an error to propagate — it is work that has to be
    redone, and the caller decides that. Reporting is the caller's job too;
    this returns a fact, not a policy.
    """
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def iter_valid_files(directory: str, *, suffix: str) -> list[str]:
    """Sorted paths in ``directory`` ending in ``suffix``, skipping temporaries.

    Files containing ``.tmp.`` are in-flight writes from this or a dead process
    and are never candidates.
    """
    if not os.path.isdir(directory):
        return []
    names = [
        name
        for name in sorted(os.listdir(directory))
        if name.endswith(suffix) and ".tmp." not in name
    ]
    return [os.path.join(directory, name) for name in names]


def sha256_of_json(payload: Any) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


def sha256_of_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------ RNG state


def capture_rng_state() -> dict:
    """Every RNG stream a stage can consume, in one payload.

    Python's ``random``, NumPy, torch CPU, and *all* CUDA devices. Missing any
    one of them means a resumed run diverges from the uninterrupted one in a way
    that only shows up as "the numbers moved a bit", which is the hardest kind of
    difference to notice and the worst kind to explain.
    """
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
        "cuda": None,
        "numpy": None,
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except Exception:  # noqa: BLE001 - NumPy is optional for RNG fidelity
        state["numpy"] = None
    return state


def restore_rng_state(state: dict | None) -> dict:
    """Restore what :func:`capture_rng_state` captured; report what was applied.

    CUDA state is skipped when the resuming machine has no (or fewer) CUDA
    devices rather than raising: a checkpoint written on an L4 is still a valid
    thing to inspect on a CPU box, and the report says the stream was not
    restored so nobody claims bit-exact continuation that did not happen.
    """
    applied = {"python": False, "numpy": False, "torch_cpu": False, "cuda": False}
    if not state:
        return applied
    python_state = state.get("python")
    if python_state is not None:
        random.setstate(
            tuple(python_state) if isinstance(python_state, list) else python_state
        )
        applied["python"] = True
    cpu_state = state.get("torch_cpu")
    if cpu_state is not None:
        torch.set_rng_state(
            cpu_state.to(torch.uint8) if torch.is_tensor(cpu_state) else cpu_state
        )
        applied["torch_cpu"] = True
    numpy_state = state.get("numpy")
    if numpy_state is not None:
        try:
            import numpy as np

            np.random.set_state(
                tuple(numpy_state) if isinstance(numpy_state, list) else numpy_state
            )
            applied["numpy"] = True
        except Exception:  # noqa: BLE001 - a missing NumPy is not a resume failure
            applied["numpy"] = False
    cuda_state = state.get("cuda")
    if cuda_state is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(cuda_state)
            applied["cuda"] = True
        except Exception:  # noqa: BLE001 - device-count mismatch is reported, not fatal
            applied["cuda"] = False
    return applied


def generator_state(generator: torch.Generator | None):
    return None if generator is None else generator.get_state()


def restore_generator(generator: torch.Generator | None, state) -> bool:
    """Put a ``torch.Generator`` back where it was. Returns whether it moved.

    Sampler orders come from dedicated generators rather than the global stream,
    so restoring these is what makes a resumed epoch see the same shuffle the
    uninterrupted run would have seen.
    """
    if generator is None or state is None:
        return False
    generator.set_state(state.to(torch.uint8) if torch.is_tensor(state) else state)
    return True


# ------------------------------------------------------------------- identity


def _architecture_identity(config) -> dict:
    return {
        "d_model": int(config.model.expect_d_model),
        "n_layers": int(config.model.expect_n_layers),
        "vocab_size": int(config.model.expect_vocab_size),
        "reconstructor_hidden_dim": int(config.reconstructor.hidden_dim),
        "reconstructor_n_layers": int(config.reconstructor.n_layers),
        "reconstructor_n_heads": int(config.reconstructor.n_heads),
        "reconstructor_max_phrase_tokens": int(config.reconstructor.max_phrase_tokens),
        "adapter_n_memory_tokens": int(config.adapter.n_memory_tokens),
        "adapter_hidden_dim": int(config.adapter.hidden_dim),
    }


def _dataset_identity(config) -> dict:
    """What makes two dataset builds *the same dataset*.

    Deliberately excludes anything that only affects how the build is executed
    (``capture_batch_size``, ``benchmark_batches``): those change the schedule,
    not the records, so a run that changes them may still reuse its shards.
    """
    return {
        "mode": config.dataset.mode,
        "corpus": config.dataset.corpus,
        "source_layer": int(config.dataset.source_layer),
        "n_phrases": int(config.dataset.n_phrases),
        "occurrences_per_phrase": int(config.dataset.occurrences_per_phrase),
        "min_phrase_tokens": int(config.dataset.min_phrase_tokens),
        "max_phrase_tokens": int(config.dataset.max_phrase_tokens),
        "min_context_tokens": int(config.dataset.min_context_tokens),
        "max_context_tokens": int(config.dataset.max_context_tokens),
        "max_documents": int(config.dataset.max_documents),
        "min_document_chars": int(config.dataset.min_document_chars),
        "seed": int(config.dataset.seed),
    }


def _split_policy(config) -> dict:
    return {
        "split_salt": config.dataset.split_salt,
        "val_fraction": float(config.dataset.val_fraction),
        "heldout_fraction": float(config.dataset.heldout_fraction),
    }


@dataclass(frozen=True)
class StageIdentity:
    """Everything a resume must agree on, split by whether it may be waived."""

    schema: str
    stage: str
    run_id: str
    run_dir: str
    config_fingerprint: str
    model_repo_id: str
    model_revision: str | None
    lens_sha256: str | None
    lens_run_dir_name: str
    source_layer: int
    pursuit: dict
    split_policy: dict
    dataset_identity: dict
    architecture: dict
    dataset_manifest_sha256: str | None = None
    reconstructor_sha256: str | None = None
    adapter_warm_sha256: str | None = None
    stage_config: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
            "stage": self.stage,
            "run_id": self.run_id,
            "run_dir": self.run_dir,
            "config_fingerprint": self.config_fingerprint,
            "model_repo_id": self.model_repo_id,
            "model_revision": self.model_revision,
            "lens_sha256": self.lens_sha256,
            "lens_run_dir_name": self.lens_run_dir_name,
            "source_layer": self.source_layer,
            "pursuit": dict(self.pursuit),
            "split_policy": dict(self.split_policy),
            "dataset_identity": dict(self.dataset_identity),
            "architecture": dict(self.architecture),
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "reconstructor_sha256": self.reconstructor_sha256,
            "adapter_warm_sha256": self.adapter_warm_sha256,
            "stage_config": dict(self.stage_config),
            "semantic_fingerprint": self.semantic_fingerprint(),
        }

    def semantic_fingerprint(self) -> str:
        """Hash of the fields no override may waive, plus the upstream artifacts.

        Upstream identity (dataset manifest, reconstructor, warm-start adapter)
        belongs here because a preference checkpoint trained against a different
        reconstructor is not a checkpoint of this experiment, however identical
        the YAML looks.
        """
        payload = {
            field_name: getattr(self, field_name) for field_name in OVERRIDE_FORBIDDEN_FIELDS
        }
        payload["dataset_manifest_sha256"] = self.dataset_manifest_sha256
        payload["reconstructor_sha256"] = self.reconstructor_sha256
        payload["adapter_warm_sha256"] = self.adapter_warm_sha256
        return sha256_of_json(payload)


def normalize_run_dir(path: str) -> str:
    """Absolute, symlink-resolved, case-normalized run directory.

    A Drive mount reached as ``/content/drive/MyDrive/x`` and as ``x`` from a
    working directory is the same run; a checkpoint should not refuse to resume
    over a spelling difference. Normalizing also means the recorded identity is
    comparable across the Colab and local paths of the same folder.
    """
    return os.path.normcase(os.path.realpath(os.path.abspath(str(path))))


def stage_identity(
    config,
    *,
    stage: str,
    run_dir: str,
    run_id: str,
    lens_sha256: str | None = None,
    dataset_manifest_sha256: str | None = None,
    reconstructor_sha256: str | None = None,
    adapter_warm_sha256: str | None = None,
    stage_config: dict | None = None,
) -> StageIdentity:
    """Build the identity block for one stage of one run."""
    if stage not in STAGES:
        raise AutoencoderError(f"unknown stage {stage!r}; known stages are {list(STAGES)}")
    return StageIdentity(
        schema=STATE_SCHEMA,
        stage=stage,
        run_id=str(run_id),
        run_dir=normalize_run_dir(run_dir),
        config_fingerprint=config.fingerprint(),
        model_repo_id=config.model.repo_id,
        model_revision=config.model.revision,
        lens_sha256=lens_sha256 or config.lens.expect_file_sha256,
        lens_run_dir_name=config.lens.run_dir_name,
        source_layer=int(config.dataset.source_layer),
        pursuit={
            "k": int(config.pursuit.k),
            "normalize_atoms": bool(config.pursuit.normalize_atoms),
            "refine_steps": int(config.pursuit.refine_steps),
            "tol_relative_residual": float(config.pursuit.tol_relative_residual),
            "atoms_dtype": config.pursuit.atoms_dtype,
        },
        split_policy=_split_policy(config),
        dataset_identity=_dataset_identity(config),
        architecture=_architecture_identity(config),
        dataset_manifest_sha256=dataset_manifest_sha256,
        reconstructor_sha256=reconstructor_sha256,
        adapter_warm_sha256=adapter_warm_sha256,
        stage_config=dict(stage_config or {}),
    )


@dataclass
class CompatibilityReport:
    """Whether stored state may be resumed, and precisely why not."""

    compatible: bool
    semantic_mismatches: list[str] = field(default_factory=list)
    nonsemantic_mismatches: list[str] = field(default_factory=list)
    overridden: bool = False
    missing_identity: bool = False

    @property
    def mismatches(self) -> list[str]:
        return [*self.semantic_mismatches, *self.nonsemantic_mismatches]

    def to_dict(self) -> dict:
        return {
            "compatible": self.compatible,
            "semantic_mismatches": list(self.semantic_mismatches),
            "nonsemantic_mismatches": list(self.nonsemantic_mismatches),
            "overridden": self.overridden,
            "missing_identity": self.missing_identity,
        }

    def describe(self) -> str:
        if self.compatible:
            if self.overridden and self.nonsemantic_mismatches:
                return (
                    "compatible under --allow-config-drift; non-semantic changes: "
                    + ", ".join(self.nonsemantic_mismatches)
                )
            return "compatible"
        if self.missing_identity:
            return "stored state carries no identity block and cannot be verified"
        parts = []
        if self.semantic_mismatches:
            parts.append(
                "semantic changes (never resumable): " + ", ".join(self.semantic_mismatches)
            )
        if self.nonsemantic_mismatches:
            parts.append(
                "non-semantic changes (resumable with --allow-config-drift): "
                + ", ".join(self.nonsemantic_mismatches)
            )
        return "; ".join(parts)


def check_compatible(
    stored: dict | None,
    current: StageIdentity,
    *,
    allow_nonsemantic_drift: bool = False,
) -> CompatibilityReport:
    """Compare a stored identity block against the current one.

    Semantic mismatches are fatal regardless of ``allow_nonsemantic_drift`` —
    that flag exists for cadence and epoch-count edits, and
    :data:`OVERRIDE_FORBIDDEN_FIELDS` is checked here rather than trusted to the
    caller so no call site can widen it by accident.
    """
    if not stored:
        return CompatibilityReport(compatible=False, missing_identity=True)
    semantic: list[str] = []
    nonsemantic: list[str] = []
    current_dict = current.to_dict()

    if stored.get("stage") != current.stage:
        semantic.append(f"stage {stored.get('stage')!r} != {current.stage!r}")
    for name in OVERRIDE_FORBIDDEN_FIELDS:
        if stored.get(name) != current_dict.get(name):
            semantic.append(f"{name}: {stored.get(name)!r} -> {current_dict.get(name)!r}")
    for name in ("dataset_manifest_sha256", "reconstructor_sha256", "adapter_warm_sha256"):
        expected = current_dict.get(name)
        found = stored.get(name)
        # ``None`` on the current side means "this stage does not depend on it".
        if expected is not None and found != expected:
            semantic.append(f"{name}: {found!r} -> {expected!r}")
    if stored.get("config_fingerprint") != current.config_fingerprint and not semantic:
        nonsemantic.append("config_fingerprint changed (non-semantic fields only)")
    if stored.get("run_id") not in (None, current.run_id):
        nonsemantic.append(f"run_id: {stored.get('run_id')!r} -> {current.run_id!r}")
    if stored.get("run_dir") not in (None, current.run_dir):
        nonsemantic.append(f"run_dir: {stored.get('run_dir')!r} -> {current.run_dir!r}")

    compatible = not semantic and (not nonsemantic or allow_nonsemantic_drift)
    return CompatibilityReport(
        compatible=compatible,
        semantic_mismatches=semantic,
        nonsemantic_mismatches=nonsemantic,
        overridden=bool(allow_nonsemantic_drift and nonsemantic and not semantic),
    )


# ---------------------------------------------------------------- stage state


def state_root(run_dir: str) -> str:
    return os.path.join(run_dir, "state")


def stage_dir(run_dir: str, stage: str) -> str:
    return os.path.join(state_root(run_dir), stage)


def checkpoints_dir(run_dir: str) -> str:
    return os.path.join(run_dir, "checkpoints")


def evaluation_shards_dir(run_dir: str) -> str:
    return os.path.join(run_dir, "evaluation_shards")


def dataset_shards_dir(run_dir: str) -> str:
    return os.path.join(run_dir, "dataset", "shards")


class StageState:
    """The on-disk state of one stage of one run.

    Layout under ``RUN_DIR/state/<stage>/``::

        state.json     status + identity + last checkpoint
        progress.json  stage-specific progress (units done / expected / failed)
        complete.json  atomic completion marker with artifact hashes

    ``complete.json`` is written last and only after the stage's artifacts
    exist, so its presence is a claim the artifacts back up rather than a claim
    about how far a loop got.
    """

    def __init__(self, run_dir: str, stage: str):
        if stage not in STAGES:
            raise AutoencoderError(f"unknown stage {stage!r}; known stages are {list(STAGES)}")
        self.run_dir = os.path.abspath(run_dir)
        self.stage = stage
        self.dir = stage_dir(self.run_dir, stage)
        self.state_path = os.path.join(self.dir, "state.json")
        self.progress_path = os.path.join(self.dir, "progress.json")
        self.complete_path = os.path.join(self.dir, "complete.json")

    # -- reading

    def read(self) -> dict:
        return read_json_if_valid(self.state_path) or {}

    def read_progress(self) -> dict:
        return read_json_if_valid(self.progress_path) or {}

    def read_marker(self) -> dict:
        return read_json_if_valid(self.complete_path) or {}

    @property
    def status(self) -> str:
        """The stage's status, with ``complete`` backed by the marker only.

        ``state.json`` is updated first and ``complete.json`` last, so a process
        killed between the two leaves a state file claiming completion with no
        marker behind it. That is an *interrupted* stage, not a complete one —
        reporting it as complete would skip work that never finished.
        """
        marker_complete = self.read_marker().get("stage") == self.stage
        status = self.read().get("status")
        if status not in STAGE_STATUSES:
            status = "not_started"
        if marker_complete:
            return "complete"
        return "interrupted" if status == "complete" else status

    def identity(self) -> dict | None:
        stored = self.read().get("identity")
        return stored if isinstance(stored, dict) else None

    # -- writing

    def write(
        self,
        *,
        status: str,
        identity: StageIdentity | None = None,
        checkpoint_path: str | None = None,
        reason: str | None = None,
        detail: dict | None = None,
    ) -> dict:
        if status not in STAGE_STATUSES:
            raise AutoencoderError(f"unknown status {status!r}; known are {list(STAGE_STATUSES)}")
        if reason is not None and reason not in CHECKPOINT_REASONS:
            raise AutoencoderError(
                f"unknown checkpoint reason {reason!r}; known are {list(CHECKPOINT_REASONS)}"
            )
        payload = dict(self.read())
        payload.update(
            {
                "schema": STATE_SCHEMA,
                "stage": self.stage,
                "status": status,
                "updated_utc": utcnow(),
            }
        )
        if identity is not None:
            payload["identity"] = identity.to_dict()
        if checkpoint_path is not None:
            payload["last_checkpoint"] = os.path.abspath(checkpoint_path)
            payload["last_checkpoint_utc"] = utcnow()
        if reason is not None:
            payload["last_checkpoint_reason"] = reason
        if detail:
            payload["detail"] = dict(detail)
        atomic_write_json(self.state_path, payload)
        return payload

    def write_progress(self, progress: dict) -> dict:
        payload = {
            "schema": STATE_SCHEMA,
            "stage": self.stage,
            "updated_utc": utcnow(),
            **dict(progress),
        }
        atomic_write_json(self.progress_path, payload)
        return payload

    def mark_complete(
        self, *, identity: StageIdentity, artifacts: dict, detail: dict | None = None
    ) -> dict:
        """Write the completion marker. Call only once the artifacts are on disk.

        ``artifacts`` maps a label to a path; each is hashed here so a later run
        can tell "complete" from "complete, and the files still are what they
        were".
        """
        hashed = {}
        for label, path in dict(artifacts).items():
            if path and os.path.isfile(path):
                hashed[label] = {"path": os.path.abspath(path), "sha256": sha256_of_file(path)}
            else:
                hashed[label] = {"path": None if not path else os.path.abspath(path), "sha256": None}
        marker = {
            "schema": STATE_SCHEMA,
            "stage": self.stage,
            "completed_utc": utcnow(),
            "identity": identity.to_dict(),
            "artifacts": hashed,
            "detail": dict(detail or {}),
        }
        self.write(status="complete", identity=identity, reason="stage_complete")
        atomic_write_json(self.complete_path, marker)
        return marker

    def clear_completion(self) -> None:
        """Drop the completion marker so the stage runs again.

        Used by ``--force``. Only the marker goes: checkpoints and shards stay,
        because "run this stage again" and "throw away what it computed" are
        different requests, and the second one is the user's to make explicitly.
        """
        if os.path.isfile(self.complete_path):
            os.replace(self.complete_path, self.complete_path + ".superseded")

    # -- decisions

    def validate_artifacts(self) -> tuple[bool, list[str]]:
        """Do the recorded artifacts still exist with the recorded hashes?"""
        marker = self.read_marker()
        if not marker:
            return False, ["no completion marker"]
        problems: list[str] = []
        for label, entry in (marker.get("artifacts") or {}).items():
            path = entry.get("path")
            expected = entry.get("sha256")
            if not path or not os.path.isfile(path):
                problems.append(f"{label}: missing at {path}")
                continue
            if expected and sha256_of_file(path) != expected:
                problems.append(f"{label}: checksum changed at {path}")
        return (not problems), problems

    def plan(
        self,
        identity: StageIdentity,
        *,
        resume: bool = True,
        force: bool = False,
        allow_nonsemantic_drift: bool = False,
    ) -> StagePlan:
        """What rerunning this stage will do, decided before any work starts.

        The default is the safe one: reuse compatible state, skip a completed
        stage whose artifacts still validate, and refuse rather than overwrite
        anything it cannot verify.
        """
        compatibility = check_compatible(
            self.identity(), identity, allow_nonsemantic_drift=allow_nonsemantic_drift
        )
        status = self.status
        if status == "complete" and not force:
            valid, problems = self.validate_artifacts()
            if not valid:
                # The marker exists (that is what ``complete`` means here) but
                # its artifacts are gone or changed. Resume rather than restart:
                # whatever intermediate work is on disk is still valid, and only
                # the missing artifacts have to be produced again.
                return StagePlan(
                    "resume" if resume else "restart",
                    status="failed",
                    compatibility=compatibility,
                    message=(
                        "completion marker present but its artifacts do not validate "
                        f"({'; '.join(problems)}); recomputing what is missing"
                    ),
                )
            if not compatibility.compatible and not compatibility.missing_identity:
                return StagePlan(
                    "incompatible",
                    status="incompatible",
                    compatibility=compatibility,
                    message=(
                        "a completed stage exists for a different configuration: "
                        + compatibility.describe()
                    ),
                )
            return StagePlan(
                "skip",
                status=status,
                compatibility=compatibility,
                message="already complete; skipping",
            )
        if force:
            return StagePlan(
                "restart",
                status=status,
                compatibility=compatibility,
                message="--force requested; restarting this stage from the beginning",
            )
        if not resume:
            return StagePlan(
                "restart",
                status=status,
                compatibility=compatibility,
                message="--no-resume requested; starting a new stage run",
            )
        if status in ("not_started",) or compatibility.missing_identity:
            return StagePlan(
                "restart",
                status=status,
                compatibility=compatibility,
                message="starting new stage",
            )
        if not compatibility.compatible:
            return StagePlan(
                "incompatible",
                status="incompatible",
                compatibility=compatibility,
                message="incompatible checkpoint; refusing to continue: "
                + compatibility.describe(),
            )
        return StagePlan(
            "resume",
            status=status,
            compatibility=compatibility,
            message="resuming from stored state",
        )


@dataclass
class StagePlan:
    """The decision :meth:`StageState.plan` reached, with its reasoning."""

    action: str  # restart | resume | skip | incompatible
    status: str
    compatibility: CompatibilityReport
    message: str

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "status": self.status,
            "message": self.message,
            "compatibility": self.compatibility.to_dict(),
        }

    def require_runnable(self) -> None:
        if self.action == "incompatible":
            raise IncompatibleState(self.message)


# --------------------------------------------------------------- interruption


class InterruptGuard:
    """Turn SIGINT/SIGTERM into a flag the loop checks at a safe boundary.

    Used as a context manager. Inside the block, :meth:`should_stop` reports
    whether a stop was requested; :meth:`check` raises :class:`StageInterrupted`
    so a loop can bail out at a point where its state is consistent. A second
    signal restores the default handler and re-raises, so an operator who really
    means it is never trapped by a stage that refuses to die.

    ``KeyboardInterrupt`` raised *between* checks is caught by the caller's own
    ``except`` — the guard's job is the signal, not the exception. Handlers are
    only installed on the main thread; elsewhere (a notebook worker thread,
    a test) the guard degrades to flag-only operation instead of raising, which
    is why :attr:`installed` exists.
    """

    def __init__(self, stage: str, *, signals: Sequence[int] | None = None):
        self.stage = stage
        self._requested: str | None = None
        self._previous: dict[int, Any] = {}
        self.installed = False
        if signals is None:
            candidates = [getattr(signal, "SIGINT", None), getattr(signal, "SIGTERM", None)]
            signals = [s for s in candidates if s is not None]
        self._signals = list(signals)

    # -- context manager

    def __enter__(self) -> InterruptGuard:
        for signum in self._signals:
            try:
                self._previous[signum] = signal.getsignal(signum)
                signal.signal(signum, self._handle)
                self.installed = True
            except (ValueError, OSError, RuntimeError):
                # Not the main thread, or a platform without this signal.
                self._previous.pop(signum, None)
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        for signum, handler in self._previous.items():
            try:
                signal.signal(signum, handler)
            except (ValueError, OSError, RuntimeError):
                pass
        self._previous.clear()
        return False

    # -- signal side

    def _handle(self, signum, _frame) -> None:
        """Set a flag. Nothing else.

        No torch save, no file write, no logging call that could take a lock the
        interrupted code already holds — a handler that serializes state is a
        handler that can corrupt the very checkpoint it is trying to write.
        """
        if self._requested is not None:
            # Second signal: give the process back to the operator.
            previous = self._previous.get(signum, signal.SIG_DFL)
            try:
                signal.signal(signum, previous)
            except (ValueError, OSError, RuntimeError):
                pass
            raise KeyboardInterrupt(f"second signal {signum} during {self.stage}")
        self._requested = (
            "keyboard_interrupt"
            if signum == getattr(signal, "SIGINT", None)
            else f"signal_{signum}"
        )

    # -- loop side

    def request_stop(self, reason: str = "keyboard_interrupt") -> None:
        """Request a stop from ordinary code (a caught ``KeyboardInterrupt``)."""
        if self._requested is None:
            self._requested = reason

    def should_stop(self) -> bool:
        return self._requested is not None

    @property
    def reason(self) -> str | None:
        return self._requested

    def check(self, *, checkpoint_path: str | None = None) -> None:
        """Raise :class:`StageInterrupted` if a stop was requested."""
        if self._requested is not None:
            raise StageInterrupted(
                f"{self.stage}: stop requested ({self._requested})",
                stage=self.stage,
                checkpoint_path=checkpoint_path,
            )


def report_interruption(
    stage: str,
    *,
    checkpoint_path: str | None,
    resume_command: str,
    stream=None,
) -> None:
    """Print the stage, where its state went, and how to continue.

    Deliberately loud and deliberately last: the operator sees this at the
    moment they interrupted, and the next thing they need is a command they can
    paste, not a stack trace.
    """
    out = stream if stream is not None else sys.stdout
    lines = [
        "",
        "=" * 72,
        f"INTERRUPTED: stage {stage!r} stopped at a safe boundary.",
        f"  state saved to: {checkpoint_path or '(no checkpoint was due yet)'}",
        "  no completion marker was written; the stage is resumable.",
        "",
        "  resume with:",
        f"    {resume_command}",
        "=" * 72,
        "",
    ]
    print("\n".join(lines), file=out, flush=True)
    for handler in getattr(__import__("logging").getLogger(), "handlers", []):
        try:
            handler.flush()
        except Exception:  # noqa: BLE001 - flushing is best-effort on the way out
            pass


def describe_run_status(run_dir: str, stages: Iterable[str] = STAGES) -> list[dict]:
    """One status row per stage, for the notebook's run-status cell."""
    rows: list[dict] = []
    for stage in stages:
        state = StageState(run_dir, stage)
        record = state.read()
        progress = state.read_progress()
        marker = state.read_marker()
        valid, problems = (True, []) if not marker else state.validate_artifacts()
        rows.append(
            {
                "stage": stage,
                "status": state.status,
                "last_checkpoint": record.get("last_checkpoint"),
                "last_checkpoint_utc": record.get("last_checkpoint_utc"),
                "last_checkpoint_reason": record.get("last_checkpoint_reason"),
                "updated_utc": record.get("updated_utc"),
                "completed_utc": marker.get("completed_utc"),
                "completed_units": progress.get("completed_units"),
                "expected_units": progress.get("expected_units"),
                "failed_units": progress.get("failed_units"),
                "artifacts_valid": valid,
                "artifact_problems": problems,
                "identity": record.get("identity"),
            }
        )
    return rows
