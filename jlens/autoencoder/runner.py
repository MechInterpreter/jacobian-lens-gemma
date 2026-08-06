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
                        shards/     per-chunk build shards (resume)
      state/            <stage>/state.json, progress.json, complete.json
      checkpoints/      <stage> resume points (epoch / periodic / interrupt)
      evaluation_shards/ baseline/, robustness/, confabulation/
      reconstructor.pt  + reconstructor.json (sidecar metadata)
      adapter.pt        + adapter.json
      adapter_epoch*.pt warm-start resume points (legacy location, still written)
      artifacts/        gate.json, evaluation.json, gonogo.json, benchmark.json
      summary.md
      <stage>_metadata.json

Every stage takes the same resume flags (:func:`add_common_args`) and defaults to
the safe behaviour: resume compatible state, skip a completed stage whose
artifacts validate, refuse incompatible state, never overwrite silently.
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
from jlens.autoencoder.pipeline import build_stack, file_sha256
from jlens.autoencoder.state import (
    STAGES,
    InterruptGuard,
    StageInterrupted,
    StageState,
    checkpoints_dir,
    describe_run_status,
    report_interruption,
    stage_identity,
)
from jlens.metadata import environment_manifest, execution_record

DEFAULT_OUTPUT_DIR = "runs/jspace_language_autoencoder"

logger = logging.getLogger("jspace_language_autoencoder")


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", required=True, help="Path to the YAML config")
    parser.add_argument(
        "--output-dir",
        "--run-dir",
        dest="output_dir",
        default=os.environ.get("JLANG_OUTPUT_DIR", DEFAULT_OUTPUT_DIR),
        help="Run directory (created; never cleared). --run-dir is an alias.",
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
    return add_resume_args(parser)


def add_resume_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """The resume/interruption flags every stage shares.

    ``--resume`` is the default and is offered explicitly anyway, so a script
    invocation in a notebook or a log can state the intent rather than rely on
    the reader knowing what the default is.
    """
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="Continue from compatible stored state (default)",
    )
    group.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Ignore stored progress and run the stage from the beginning",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun even if the stage is marked complete (the marker is set aside, "
        "not deleted, and existing artifacts are only replaced on success)",
    )
    parser.add_argument(
        "--checkpoint-every-steps",
        type=int,
        default=int(os.environ.get("JLANG_CHECKPOINT_EVERY_STEPS", "0")),
        help="Periodic checkpoint cadence in optimizer steps (0 = epoch boundaries only)",
    )
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="Print the run's stage status and exit without doing any work",
    )
    parser.add_argument(
        "--validate-state",
        action="store_true",
        help="Validate stored checkpoints, shards, and artifacts, then exit",
    )
    parser.add_argument(
        "--allow-config-drift",
        action="store_true",
        help="Resume across NON-SEMANTIC config changes only. Never permits a change "
        "of model revision, lens checksum, layer, pursuit settings, phrase split, "
        "dataset identity, or architecture dimensions.",
    )
    parser.add_argument(
        "--run-id",
        default=os.environ.get("JLANG_RUN_ID"),
        help="Run identity recorded in state and checkpoints (defaults to the run "
        "directory's basename)",
    )
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
    os.makedirs(checkpoints_dir(path), exist_ok=True)
    for stage in STAGES:
        os.makedirs(os.path.join(path, "state", stage), exist_ok=True)
    return path


def resolve_run_id(args: argparse.Namespace, run_dir: str) -> str:
    """The run's identity. Defaults to the run directory's own name.

    Stable across reconnects (unlike a timestamp), which is what a resume needs:
    the run id is part of the identity a checkpoint is checked against.
    """
    return str(getattr(args, "run_id", None) or os.path.basename(os.path.normpath(run_dir)))


def lens_checksum(config: AutoencoderConfig, args: argparse.Namespace) -> str | None:
    """The lens file's actual sha256 when it is reachable, else the expected one.

    Resolved from disk where possible so a run against a *substituted* lens file
    is refused on resume even when the config still names the expected hash.
    """
    try:
        path = os.path.join(
            getattr(args, "runs_root", "runs"),
            config.lens.run_dir_name,
            config.lens.artifact_relpath,
        )
        if os.path.isfile(path):
            return file_sha256(path)
    except OSError:
        pass
    return config.lens.expect_file_sha256


def build_identity(
    config: AutoencoderConfig,
    args: argparse.Namespace,
    *,
    stage: str,
    run_dir: str,
    dataset_manifest_sha256: str | None = None,
    reconstructor_sha256: str | None = None,
    adapter_warm_sha256: str | None = None,
    stage_config: dict | None = None,
):
    return stage_identity(
        config,
        stage=stage,
        run_dir=run_dir,
        run_id=resolve_run_id(args, run_dir),
        lens_sha256=lens_checksum(config, args),
        dataset_manifest_sha256=dataset_manifest_sha256,
        reconstructor_sha256=reconstructor_sha256,
        adapter_warm_sha256=adapter_warm_sha256,
        stage_config=dict(stage_config or {}),
    )


def announce_plan(stage: str, plan, *, log=None) -> None:
    """Print exactly one of the four sentences a stage cell is allowed to print."""
    text = {
        "restart": f"[{stage}] starting new stage",
        "resume": f"[{stage}] resuming: {plan.message}",
        "skip": f"[{stage}] already complete; skipping",
        "incompatible": f"[{stage}] incompatible checkpoint; refusing to continue",
    }[plan.action]
    print(text, flush=True)
    if plan.action == "incompatible":
        print(f"  reason: {plan.compatibility.describe()}", flush=True)
    elif plan.message and plan.action != "resume":
        print(f"  {plan.message}", flush=True)
    if log is not None:
        log.info("%s: %s (%s)", stage, plan.action, plan.message)


def resume_command(stage_script: str, args: argparse.Namespace) -> str:
    """The exact command that continues this stage."""
    parts = [
        "python",
        "-u",
        f"scripts/{stage_script}.py",
        "--config",
        str(args.config),
        "--output-dir",
        str(args.output_dir),
        "--resume",
    ]
    for flag, value in (
        ("--runs-root", getattr(args, "runs_root", None)),
        ("--device-map", getattr(args, "device_map", None)),
    ):
        if value:
            parts += [flag, str(value)]
    for flag in ("smoke", "mock", "allow_model_load"):
        if getattr(args, flag, False):
            parts.append("--" + flag.replace("_", "-"))
    return " ".join(parts)


def print_run_status(run_dir: str) -> list[dict]:
    """Human-readable status of every stage in ``run_dir``."""
    rows = describe_run_status(run_dir)
    print(f"run directory: {run_dir}")
    print(f"{'stage':<20} {'status':<13} {'done/expected':<16} {'last checkpoint'}")
    print("-" * 96)
    for row in rows:
        done = row.get("completed_units")
        expected = row.get("expected_units")
        progress = "-" if done is None and expected is None else f"{done}/{expected}"
        checkpoint = row.get("last_checkpoint") or row.get("completed_utc") or "-"
        print(f"{row['stage']:<20} {row['status']:<13} {progress:<16} {os.path.basename(str(checkpoint))}")
        if row["status"] == "complete" and not row["artifacts_valid"]:
            for problem in row["artifact_problems"]:
                print(f"    ! {problem}")
    return rows


def handle_status_flags(args: argparse.Namespace, run_dir: str) -> int | None:
    """Serve ``--status-only``; returns an exit code when the script should stop."""
    if getattr(args, "status_only", False):
        print_run_status(run_dir)
        return 0
    return None


def run_stage(
    stage: str,
    *,
    args: argparse.Namespace,
    run_dir: str,
    identity,
    script: str,
    body,
    log=None,
) -> int:
    """Plan, guard, execute, and record one stage.

    The whole interruption contract lives here so all five stages behave the
    same: decide before doing anything, install the signal guard, run, and on a
    stop print the stage, the saved state, and a pasteable resume command —
    without writing a completion marker.
    """
    state = StageState(run_dir, stage)
    plan = state.plan(
        identity,
        resume=bool(getattr(args, "resume", True)),
        force=bool(getattr(args, "force", False)),
        allow_nonsemantic_drift=bool(getattr(args, "allow_config_drift", False)),
    )
    announce_plan(stage, plan, log=log)
    if plan.action == "incompatible":
        raise AutoencoderError(plan.message)
    if plan.action == "skip":
        return 0
    if plan.action == "restart" and getattr(args, "force", False):
        state.clear_completion()
    state.write(status="in_progress", identity=identity, detail=plan.to_dict())
    with InterruptGuard(stage) as guard:
        try:
            return int(body(guard=guard, plan=plan, state=state) or 0)
        except (StageInterrupted, KeyboardInterrupt) as exc:
            checkpoint_path = getattr(exc, "checkpoint_path", None) or state.read().get(
                "last_checkpoint"
            )
            state.write(
                status="interrupted",
                identity=identity,
                checkpoint_path=checkpoint_path,
                reason="keyboard_interrupt",
            )
            report_interruption(
                stage,
                checkpoint_path=checkpoint_path,
                resume_command=resume_command(script, args),
            )
            return 130
        except Exception:
            state.write(status="failed", identity=identity, detail={"error": "see traceback"})
            raise


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
