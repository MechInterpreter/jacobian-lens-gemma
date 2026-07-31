# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Shared CLI plumbing for the four stage scripts.

Argument parsing, run-directory layout, config overrides, and metadata writing
live here so the four scripts differ only in what they *do*. A stage that
writes its metadata differently from its neighbours is a stage whose artifacts
cannot be joined later.

Run layout::

    <output-dir>/
      dataset/          records.jsonl, tensors.pt, manifest.json
      reconstructor.pt  + reconstructor.json (sidecar metadata)
      adapter.pt        + adapter.json
      adapter_epoch*.pt resume points
      artifacts/        gate.json, evaluation.json, gonogo.json, benchmark.json
      summary.md
      <stage>_metadata.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone

from jlens.autoencoder.config import AutoencoderConfig, load_autoencoder_config
from jlens.autoencoder.errors import AutoencoderError
from jlens.autoencoder.pipeline import build_stack
from jlens.metadata import environment_manifest, execution_record

DEFAULT_OUTPUT_DIR = "runs/jspace_language_autoencoder"

logger = logging.getLogger("jspace_language_autoencoder")


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", required=True, help="Path to the YAML config")
    parser.add_argument(
        "--output-dir",
        default=os.environ.get("JLANG_OUTPUT_DIR", DEFAULT_OUTPUT_DIR),
        help="Run directory (created; never cleared)",
    )
    parser.add_argument(
        "--runs-root",
        default=os.environ.get("JLENS_RUNS_ROOT", "runs"),
        help="Directory holding the pilot run whose lens.pt is consumed",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use the deterministic CPU mock stack — no checkpoint, no downloads",
    )
    parser.add_argument("--allow-model-load", action="store_true")
    parser.add_argument("--device-map", default=os.environ.get("JLENS_DEVICE_MAP"))
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smallest end-to-end pass (implies --mock unless --allow-model-load)",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def configure_logging(level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, str(level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )
    return logger


#: Overrides applied by ``--smoke``. Small enough to finish on a CPU in seconds,
#: large enough that every split is non-empty and every code path is exercised.
SMOKE_OVERRIDES: dict = {
    "dataset": {
        "mode": "smoke",
        "corpus": "mock",
        "n_phrases": 32,
        "occurrences_per_phrase": 2,
        "min_context_tokens": 3,
        "max_context_tokens": 48,
        "max_documents": 96,
        "min_document_chars": 40,
    },
    "reconstructor": {
        "hidden_dim": 32,
        "n_heads": 4,
        "n_layers": 1,
        "epochs": 5,
        "batch_size": 4,
        "n_distractors": 3,
    },
    "adapter": {
        "n_memory_tokens": 2,
        "hidden_dim": 32,
        "epochs": 1,
        "batch_size": 2,
        "beam_width": 3,
        "max_new_tokens": 6,
    },
    "preference": {
        "epochs": 1,
        "batch_size": 2,
        "n_unrelated_cones": 3,
        "max_pairs_per_example": 2,
    },
    "evaluation": {
        "beam_width": 3,
        "max_new_tokens": 6,
        "n_unrelated_cones": 3,
        "n_distractors": 3,
    },
}


def _deep_merge(base: dict, overrides: dict) -> dict:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_config(args: argparse.Namespace) -> AutoencoderConfig:
    """Load the config and apply ``--smoke`` overrides.

    Overrides are applied to the *config dict* and revalidated, so a smoke run
    is a real, fully-validated configuration — not a set of special cases
    scattered through the stage code.
    """
    config = load_autoencoder_config(args.config)
    if getattr(args, "smoke", False):
        config = AutoencoderConfig.from_dict(_deep_merge(config.to_dict(), SMOKE_OVERRIDES))
    return config


def use_mock(args: argparse.Namespace) -> bool:
    """Whether this invocation runs on the mock stack.

    ``--smoke`` implies ``--mock`` unless the caller explicitly asked to load the
    real model, so "smoke test" can never accidentally start a 16 GB download.
    """
    if getattr(args, "mock", False):
        return True
    return bool(getattr(args, "smoke", False)) and not getattr(args, "allow_model_load", False)


def prepare_run_dir(args: argparse.Namespace) -> str:
    path = os.path.abspath(args.output_dir)
    os.makedirs(os.path.join(path, "artifacts"), exist_ok=True)
    return path


def build_stack_from_args(config: AutoencoderConfig, args: argparse.Namespace):
    return build_stack(
        config,
        mock=use_mock(args),
        allow_model_load=bool(getattr(args, "allow_model_load", False)),
        device_map=getattr(args, "device_map", None),
        runs_root=getattr(args, "runs_root", "runs"),
    )


def write_json(path: str, payload: dict) -> str:
    """Atomic pretty JSON write; parents created."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp_path, path)
    return path


def write_text(path: str, text: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        handle.write(text)
    os.replace(tmp_path, path)
    return path


def stage_metadata(
    stage: str,
    config: AutoencoderConfig,
    args: argparse.Namespace,
    stack,
    *,
    payload: dict | None = None,
) -> dict:
    """The provenance block every stage writes.

    Distinguishes the configured model-load policy from what actually happened,
    the same way :func:`jlens.metadata.execution_record` does for the other
    experiments on this repository — a run that says ``allow_model_load: false``
    while having loaded Gemma is exactly the ambiguity that record exists to
    remove.
    """
    mock = bool(stack.is_mock)
    configured = bool(config.model.allow_model_load)
    resolved = configured or bool(getattr(args, "allow_model_load", False))
    override = "cli:--allow-model-load" if resolved != configured else None
    return {
        "schema": "jlens.autoencoder.stage.v1",
        "stage": stage,
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "argv": list(sys.argv),
        "config": config.to_dict(),
        "config_fingerprint": config.fingerprint(),
        "execution": execution_record(
            configured_allow_model_load=configured,
            resolved_allow_model_load=resolved,
            model_loaded=not mock,
            override_source=override,
        ),
        "environment": environment_manifest(),
        "stack_provenance": stack.provenance,
        **dict(payload or {}),
    }


def require_file(path: str, *, what: str) -> str:
    if not os.path.isfile(path):
        raise AutoencoderError(
            f"{what} not found at {path}; run the preceding stage first"
        )
    return path
