# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Checkpoints with identity, and resume that cannot silently resume the wrong
thing.

Every checkpoint carries:

* a ``kind`` (``reconstructor`` / ``adapter``) that loading verifies,
* a **sha256 of its own parameters**, so "which reconstructor was the adapter
  trained against?" is answerable by equality rather than by filename,
* the resolved config fingerprint it was produced under,
* the metrics at the time of writing,
* optimizer and RNG state when the write is a resume point.

Writes are atomic (temp file + ``os.replace``), so an interrupted Colab runtime
can never leave a half-written checkpoint that loads as a valid one.

:func:`save_training_checkpoint` extends this to *resume points*: the same
identity block plus optimizer, scheduler, AMP scaler, every RNG stream, the
epoch/batch/step position, and the sampler order — everything a stage needs to
continue on the trajectory it was on rather than on a new one that happens to
start at the same weights. See :mod:`jlens.autoencoder.state` for the stage
state machine those checkpoints live inside.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any

import torch
from torch import nn

from jlens.autoencoder.errors import AutoencoderError
from jlens.autoencoder.state import (
    CHECKPOINT_REASONS,
    CHECKPOINT_SCHEMA_VERSION,
    StageIdentity,
    atomic_torch_save,
    capture_rng_state,
    generator_state,
    iter_valid_files,
    restore_generator,
    restore_rng_state,
    utcnow,
)

CHECKPOINT_SCHEMA = "jlens.autoencoder.checkpoint.v1"

#: Resume points carry more than weights, so they declare a schema of their own.
#: A v1 reader that met one would silently ignore the optimizer and RNG blocks
#: and resume onto a fresh trajectory, which is exactly the failure this whole
#: module exists to prevent.
TRAINING_CHECKPOINT_SCHEMA = "jlens.autoencoder.training_checkpoint.v1"

CHECKPOINT_KINDS = ("reconstructor", "adapter", "adapter_reference")

#: Training stages that may own a resume point. Checked on load, so a
#: warm-start checkpoint can never be restored as preference state.
TRAINING_STAGES = ("reconstructor", "adapter_warm", "adapter_preference")


def _tensor_digest(digest: hashlib._Hash, key: str, tensor: torch.Tensor) -> None:
    tensor = tensor.detach().to("cpu").contiguous()
    digest.update(key.encode())
    if tensor.is_floating_point():
        payload = tensor.to(torch.float32).numpy().astype("<f4", copy=False)
    else:
        payload = tensor.to(torch.int64).numpy().astype("<i8", copy=False)
    digest.update(payload.tobytes())


def state_dict_payload_sha256(state: dict) -> str:
    """Fingerprint a plain ``{name: tensor}`` mapping.

    Separate from :func:`state_dict_sha256` because a *loaded* checkpoint has to
    be validated before there is a module to load it into: a truncated or
    bit-rotted tensor block should be rejected on read, not discovered later as
    a model that trains strangely.
    """
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        if not torch.is_tensor(value):
            raise AutoencoderError(f"state entry {key!r} is not a tensor")
        _tensor_digest(digest, key, value)
    return "sha256:" + digest.hexdigest()


def state_dict_sha256(module: nn.Module) -> str:
    """Deterministic fingerprint of a module's parameters and buffers.

    Hashed as little-endian float32/int64 bytes in sorted key order on CPU, so
    the value depends only on the numbers — not on device, storage dtype, or
    parameter ordering in memory.
    """
    return state_dict_payload_sha256(module.state_dict())


def parameter_counts(module: nn.Module) -> dict:
    """Trainable / frozen parameter counts for one module."""
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in module.parameters() if not p.requires_grad)
    return {
        "trainable_parameters": int(trainable),
        "frozen_parameters": int(frozen),
        "total_parameters": int(trainable + frozen),
    }


def _atomic_torch_save(payload: dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def save_checkpoint(
    path: str,
    module: nn.Module,
    *,
    kind: str,
    config: dict,
    metrics: dict | None = None,
    extra: dict | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int | None = None,
    include_rng: bool = True,
) -> dict:
    """Write ``module`` plus its identity block; return the metadata written.

    The parameter counts describe the module **as saved**. A module frozen on
    the way out of training therefore reports ``trainable_parameters: 0``, which
    is accurate — the count from before freezing belongs in ``extra`` (the
    reconstructor stage records it under ``training_summary``).

    Passing ``optimizer``/``epoch`` makes the checkpoint a **resume point**: the
    optimizer state and CPU/CUDA RNG state are stored alongside the weights so
    an interrupted run continues from the same trajectory rather than from a
    fresh one that merely starts at the same weights.
    """
    if kind not in CHECKPOINT_KINDS:
        raise ValueError(f"checkpoint kind {kind!r} not in {CHECKPOINT_KINDS}")
    metadata = {
        "schema": CHECKPOINT_SCHEMA,
        "kind": kind,
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "module_class": type(module).__name__,
        "state_dict_sha256": state_dict_sha256(module),
        "config": dict(config),
        "config_sha256": "sha256:"
        + hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":"), default=str).encode()
        ).hexdigest(),
        "metrics": dict(metrics or {}),
        "extra": dict(extra or {}),
        "epoch": epoch,
        **parameter_counts(module),
    }
    payload: dict[str, Any] = {
        "metadata": metadata,
        "state_dict": {k: v.detach().cpu() for k, v in module.state_dict().items()},
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if include_rng:
        payload["rng_state"] = {
            "cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
    _atomic_torch_save(payload, path)
    sidecar = os.path.splitext(path)[0] + ".json"
    tmp_sidecar = f"{sidecar}.tmp.{os.getpid()}"
    with open(tmp_sidecar, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp_sidecar, sidecar)
    return metadata


def load_checkpoint(path: str, *, expect_kind: str | None = None) -> dict:
    """Load a checkpoint written by :func:`save_checkpoint`.

    ``weights_only=False`` is required because the payload carries optimizer and
    RNG state; the file is one this pipeline wrote into its own run directory.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "metadata" not in payload:
        raise ValueError(f"{path} is not a jlens.autoencoder checkpoint")
    metadata = payload["metadata"]
    if metadata.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError(f"{path}: unexpected schema {metadata.get('schema')!r}")
    if expect_kind is not None and metadata.get("kind") != expect_kind:
        raise ValueError(
            f"{path}: expected a {expect_kind!r} checkpoint, found {metadata.get('kind')!r}"
        )
    return payload


def restore_module(module: nn.Module, payload: dict, *, strict: bool = True) -> dict:
    """Load weights into ``module`` and verify the stored fingerprint.

    The fingerprint check is the point: a state dict that loads without error can
    still be the wrong one (same shapes, different training). Equality of the
    recomputed hash proves it is the exact tensor set that was written.
    """
    module.load_state_dict(payload["state_dict"], strict=strict)
    recomputed = state_dict_sha256(module)
    expected = payload["metadata"].get("state_dict_sha256")
    if expected is not None and recomputed != expected:
        raise ValueError(
            f"restored parameters hash to {recomputed} but the checkpoint records "
            f"{expected}; the load did not reproduce the saved module"
        )
    return dict(payload["metadata"])


def restore_rng(payload: dict) -> bool:
    """Restore CPU/CUDA RNG state from a resume point. Returns whether it did."""
    state = payload.get("rng_state")
    if not state:
        return False
    cpu_state = state.get("cpu")
    if cpu_state is not None:
        torch.set_rng_state(cpu_state.to(torch.uint8) if torch.is_tensor(cpu_state) else cpu_state)
    cuda_state = state.get("cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)
    return True


def find_resume_checkpoint(directory: str, *, kind: str) -> str | None:
    """Newest ``<kind>_epoch*.pt`` resume point in ``directory``, or ``None``.

    Returns a path only when the file loads and reports the expected kind, so a
    truncated or foreign file is skipped rather than crashing a resume.

    Name-prefixed on purpose: ``adapter_preference_epoch*.pt`` does not start
    with ``adapter_epoch``, so warm-start discovery cannot pick up a preference
    checkpoint. :func:`find_training_checkpoint` makes that separation explicit
    by also checking the recorded stage.
    """
    if not os.path.isdir(directory):
        return None
    candidates = sorted(
        name
        for name in os.listdir(directory)
        if name.startswith(f"{kind}_epoch") and name.endswith(".pt") and ".tmp." not in name
    )
    for name in reversed(candidates):
        path = os.path.join(directory, name)
        try:
            load_checkpoint(path, expect_kind=kind)
        except Exception:  # noqa: BLE001 - a broken candidate is simply skipped
            continue
        return path
    return None


# ------------------------------------------------------- training resume points


def save_training_checkpoint(
    path: str,
    module: nn.Module,
    *,
    kind: str,
    stage: str,
    identity: StageIdentity,
    reason: str,
    config: dict,
    epoch: int,
    batch_index: int,
    global_step: int,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    scaler: Any = None,
    sampler_order: list[int] | None = None,
    generators: dict[str, torch.Generator] | None = None,
    metrics: dict | None = None,
    history: list | None = None,
    extra: dict | None = None,
) -> dict:
    """Write a full resume point atomically; return the metadata written.

    The position is stored as ``(epoch, batch_index, global_step)`` *plus* the
    epoch's ``sampler_order``. Storing the order rather than only the generator
    seed is what makes mid-epoch resume exact: the shuffle for the interrupted
    epoch was drawn before the interruption, and re-deriving it from a generator
    that has since advanced would silently reorder the remaining batches.
    """
    if kind not in CHECKPOINT_KINDS:
        raise AutoencoderError(f"checkpoint kind {kind!r} not in {CHECKPOINT_KINDS}")
    if stage not in TRAINING_STAGES:
        raise AutoencoderError(f"training stage {stage!r} not in {TRAINING_STAGES}")
    if reason not in CHECKPOINT_REASONS:
        raise AutoencoderError(f"checkpoint reason {reason!r} not in {CHECKPOINT_REASONS}")
    state = {k: v.detach().cpu() for k, v in module.state_dict().items()}
    metadata = {
        "schema": TRAINING_CHECKPOINT_SCHEMA,
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "kind": kind,
        "stage": stage,
        "reason": reason,
        "written_utc": utcnow(),
        "module_class": type(module).__name__,
        "state_dict_sha256": state_dict_payload_sha256(state),
        "identity": identity.to_dict(),
        "config": dict(config),
        "config_sha256": "sha256:"
        + hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":"), default=str).encode()
        ).hexdigest(),
        "epoch": int(epoch),
        "batch_index": int(batch_index),
        "global_step": int(global_step),
        "metrics": dict(metrics or {}),
        "history": list(history or []),
        "extra": dict(extra or {}),
        **parameter_counts(module),
    }
    payload: dict[str, Any] = {
        "metadata": metadata,
        "state_dict": state,
        "rng_state": capture_rng_state(),
        "sampler_order": list(sampler_order) if sampler_order is not None else None,
        "generator_states": {
            name: generator_state(gen) for name, gen in dict(generators or {}).items()
        },
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        payload["scheduler_state"] = scheduler.state_dict()
    if scaler is not None and hasattr(scaler, "state_dict"):
        payload["scaler_state"] = scaler.state_dict()
    atomic_torch_save(payload, path)
    sidecar = os.path.splitext(path)[0] + ".json"
    tmp_sidecar = f"{sidecar}.tmp.{os.getpid()}"
    with open(tmp_sidecar, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp_sidecar, sidecar)
    return metadata


def load_training_checkpoint(
    path: str, *, expect_kind: str | None = None, expect_stage: str | None = None
) -> dict:
    """Load and validate a resume point.

    Validation is by recomputed checksum, not by "it deserialized": a file that
    unpickles cleanly can still have the wrong weights in it, and the whole point
    of resuming is that you cannot tell by looking at the loss.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "metadata" not in payload:
        raise AutoencoderError(f"{path} is not a jlens.autoencoder training checkpoint")
    metadata = payload["metadata"]
    if metadata.get("schema") != TRAINING_CHECKPOINT_SCHEMA:
        raise AutoencoderError(f"{path}: unexpected schema {metadata.get('schema')!r}")
    if metadata.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise AutoencoderError(
            f"{path}: checkpoint schema version {metadata.get('schema_version')!r} != "
            f"{CHECKPOINT_SCHEMA_VERSION}"
        )
    if expect_kind is not None and metadata.get("kind") != expect_kind:
        raise AutoencoderError(
            f"{path}: expected a {expect_kind!r} checkpoint, found {metadata.get('kind')!r}"
        )
    if expect_stage is not None and metadata.get("stage") != expect_stage:
        raise AutoencoderError(
            f"{path}: expected stage {expect_stage!r}, found {metadata.get('stage')!r}. "
            f"Refusing to resume one training stage from another's checkpoint."
        )
    recorded = metadata.get("state_dict_sha256")
    observed = state_dict_payload_sha256(payload["state_dict"])
    if recorded is not None and recorded != observed:
        raise AutoencoderError(
            f"{path}: parameter checksum {observed} does not match the recorded "
            f"{recorded}; the file is corrupt or truncated"
        )
    return payload


def find_training_checkpoint(
    directory: str, *, kind: str, stage: str, prefix: str
) -> tuple[str | None, list[dict]]:
    """Newest valid ``<prefix>*.pt`` resume point for ``stage``, plus a report.

    Returns ``(path, rejected)`` where ``rejected`` lists every candidate that
    was skipped and why. Newest-first by name, so an interrupted-then-corrupted
    latest checkpoint falls back to the previous valid one instead of failing
    the resume outright — with the reason visible rather than swallowed.
    """
    rejected: list[dict] = []
    candidates = [
        path
        for path in iter_valid_files(directory, suffix=".pt")
        if os.path.basename(path).startswith(prefix)
    ]
    for path in reversed(candidates):
        try:
            load_training_checkpoint(path, expect_kind=kind, expect_stage=stage)
        except Exception as exc:  # noqa: BLE001 - a broken candidate is reported, not fatal
            rejected.append({"path": path, "reason": str(exc)})
            continue
        return path, rejected
    return None, rejected


def restore_training_state(
    payload: dict,
    module: nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    scaler: Any = None,
    generators: dict[str, torch.Generator] | None = None,
    strict: bool = True,
) -> dict:
    """Put a resume point back into live objects; report what was restored."""
    metadata = restore_module(module, payload, strict=strict)
    report = {
        "epoch": int(metadata.get("epoch") or 0),
        "batch_index": int(metadata.get("batch_index") or 0),
        "global_step": int(metadata.get("global_step") or 0),
        "reason": metadata.get("reason"),
        "written_utc": metadata.get("written_utc"),
        "state_dict_sha256": metadata.get("state_dict_sha256"),
        "optimizer_restored": False,
        "scheduler_restored": False,
        "scaler_restored": False,
        "generators_restored": [],
    }
    if optimizer is not None and payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
        report["optimizer_restored"] = True
    if scheduler is not None and payload.get("scheduler_state") is not None:
        scheduler.load_state_dict(payload["scheduler_state"])
        report["scheduler_restored"] = True
    if scaler is not None and payload.get("scaler_state") is not None:
        scaler.load_state_dict(payload["scaler_state"])
        report["scaler_restored"] = True
    for name, generator in dict(generators or {}).items():
        if restore_generator(generator, (payload.get("generator_states") or {}).get(name)):
            report["generators_restored"].append(name)
    report["rng_restored"] = restore_rng_state(payload.get("rng_state"))
    report["sampler_order"] = payload.get("sampler_order")
    report["history"] = list(metadata.get("history") or [])
    return report
