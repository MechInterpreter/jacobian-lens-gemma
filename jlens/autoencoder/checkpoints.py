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
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any

import torch
from torch import nn

CHECKPOINT_SCHEMA = "jlens.autoencoder.checkpoint.v1"

CHECKPOINT_KINDS = ("reconstructor", "adapter", "adapter_reference")


def state_dict_sha256(module: nn.Module) -> str:
    """Deterministic fingerprint of a module's parameters and buffers.

    Hashed as little-endian float32/int64 bytes in sorted key order on CPU, so
    the value depends only on the numbers — not on device, storage dtype, or
    parameter ordering in memory.
    """
    digest = hashlib.sha256()
    state = module.state_dict()
    for key in sorted(state):
        tensor = state[key].detach().to("cpu").contiguous()
        digest.update(key.encode())
        if tensor.is_floating_point():
            payload = tensor.to(torch.float32).numpy().astype("<f4", copy=False)
        else:
            payload = tensor.to(torch.int64).numpy().astype("<i8", copy=False)
        digest.update(payload.tobytes())
    return "sha256:" + digest.hexdigest()


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
    """
    if not os.path.isdir(directory):
        return None
    candidates = sorted(
        name
        for name in os.listdir(directory)
        if name.startswith(f"{kind}_epoch") and name.endswith(".pt")
    )
    for name in reversed(candidates):
        path = os.path.join(directory, name)
        try:
            load_checkpoint(path, expect_kind=kind)
        except Exception:  # noqa: BLE001 - a broken candidate is simply skipped
            continue
        return path
    return None
