# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Calibration orchestration: nested scale snapshots and per-layer diagnostics.

**This module does not implement the estimator.** It calls the unmodified
upstream :func:`jlens.fitting.fit` and watches it through its ``on_prompt``
hook. If upstream's Jacobian is wrong, this is wrong identically — which is the
property that makes the study a reproduction rather than a re-derivation.

The one idea here is the **scale snapshot**. Because ``J_l`` is a running mean
over a deterministically ordered prompt list, the accumulator's state after
100 prompts *is* the 100-prompt lens. So the three scale points are taken as
snapshots of one accumulator rather than three separate fits:

    fit(prompts[:1_000], on_prompt=snapshot_scales(...))
       -> writes lens.scale100.pt  when n_done hits 100
       -> writes lens.scale250.pt  when n_done hits 250
       -> writes lens.scale1000.pt when n_done hits 1_000

The 100, 250 and 1k lenses therefore cost exactly as much as the 1k lens alone,
and each is bit-identical to what a standalone fit on the same prefix would
produce. A test asserts that equality rather than assuming it.

Diagnostics are recorded **per layer**, because that is where they are
informative: ``mean_relative_change`` of each layer's running mean should fall
like ``1/n`` once the estimator has settled, and a layer whose per-prompt
Jacobian norm is heavy-tailed will show it here long before the gate does.

Note on vocabulary: ``fit_loss`` does not appear anywhere in this module.
There is no loss. See :mod:`jlens.calibration`.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from jlens.calibration.corpus import CorpusRecord, nested_subset
from jlens.calibration.plan import CapturePlan
from jlens.calibration.state import CalibrationStore
from jlens.fitting import fit
from jlens.lens import JacobianLens
from jlens.metadata import file_sha256
from jlens.mmpilot.store import payload_checksum

__all__ = [
    "CalibrationResult",
    "ScaleSnapshot",
    "filter_records_by_tokens",
    "layer_diagnostics",
    "run_calibration",
    "snapshot_scales",
]


class ScaleNotReachedError(RuntimeError):
    """Fitting ended before a configured scale point was reached.

    Raised rather than relabelled. A snapshot written at 973 prompts and named
    ``scale1000`` would silently corrupt every comparison that follows.
    """


@dataclass(frozen=True)
class ScaleSnapshot:
    """One nested scale point, written from the shared accumulator."""

    scale: int
    n_prompts: int
    path: str
    checksum: str
    layers: tuple[int, ...]
    d_model: int
    written_seconds: float

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["layers"] = list(self.layers)
        return payload


@dataclass
class CalibrationResult:
    """What one calibration run produced."""

    snapshots: dict[int, ScaleSnapshot] = field(default_factory=dict)
    diagnostics: list[dict] = field(default_factory=list)
    n_done: int = 0
    n_skipped: int = 0
    elapsed_seconds: float = 0.0
    checkpoint_path: str | None = None

    def lens_for_scale(self, scale: int) -> JacobianLens:
        """Load the lens for one scale point from disk."""
        snapshot = self.snapshots[int(scale)]
        return JacobianLens.load(snapshot.path)

    def to_dict(self) -> dict:
        return {
            "snapshots": {
                str(scale): snapshot.to_dict()
                for scale, snapshot in sorted(self.snapshots.items())
            },
            "n_done": self.n_done,
            "n_skipped": self.n_skipped,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "checkpoint_path": self.checkpoint_path,
            "n_diagnostic_rows": len(self.diagnostics),
            "objective": "not_applicable_estimator_is_a_sample_mean",
        }


# ------------------------------------------------------------- pre-filtering


def filter_records_by_tokens(
    records: Sequence[CorpusRecord],
    *,
    token_count: Callable[[str], int],
    skip_first: int,
    max_seq_len: int,
) -> tuple[list[CorpusRecord], list[dict]]:
    """Drop records too short to contribute a single valid source position.

    Upstream ``fit`` skips such prompts at fit time, which would make ``n_done``
    drift below the prompt index and turn a "1,000-prompt lens" into a lens
    fitted on some unknown number below 1,000. Filtering first — with the
    tokenizer only, no forward pass — makes the scale points exact.

    Args:
        token_count: Tokenizes and returns a length. The tokenizer is the only
            model component involved; this is cheap and runs on CPU.
    """
    minimum = skip_first + 2  # need at least one position in [skip_first, len-1)
    kept: list[CorpusRecord] = []
    dropped: list[dict] = []
    for record in records:
        length = min(int(token_count(record.text)), int(max_seq_len))
        if length >= minimum:
            kept.append(record)
        else:
            dropped.append(
                {
                    "record_id": record.record_id,
                    "token_length": length,
                    "minimum_required": minimum,
                }
            )
    return kept, dropped


# ------------------------------------------------------------------ snapshots


def _atomic_torch_save(lens: JacobianLens, path: Path) -> None:
    """``JacobianLens.save`` via a temp file, so a crash never leaves a torn
    snapshot. Upstream's ``fit`` checkpoint is already atomic; its lens
    serializer is not."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    lens.save(str(tmp))
    os.replace(tmp, path)


def layer_diagnostics(
    per_prompt: dict[int, torch.Tensor],
    jacobian_sum: dict[int, torch.Tensor],
    n_done: int,
) -> dict[str, dict]:
    """Per-layer convergence and heavy-tail diagnostics for one prompt.

    ``mean_relative_change`` is the relative movement of the running mean caused
    by this prompt. It should decay like ``1/n`` once the estimator has settled;
    a layer where it does not is the layer to look at, and it is the closest
    thing to a "convergence" signal a sample-mean estimator has.
    """
    out: dict[str, dict] = {}
    for layer, contribution in sorted(per_prompt.items()):
        norm = float(contribution.norm())
        mean_new = jacobian_sum[layer] / n_done
        if n_done > 1:
            mean_prev = (jacobian_sum[layer] - contribution) / (n_done - 1)
            prev_norm = float(mean_prev.norm())
            relative = (
                float((mean_new - mean_prev).norm()) / prev_norm
                if prev_norm > 0
                else float("nan")
            )
        else:
            relative = float("nan")
        out[str(layer)] = {
            "prompt_jacobian_norm": round(norm, 6),
            "running_mean_norm": round(float(mean_new.norm()), 6),
            "mean_relative_change": (
                round(relative, 9) if relative == relative else None
            ),
            "finite": bool(torch.isfinite(contribution).all()),
        }
    return out


def snapshot_scales(
    *,
    scale_points: Sequence[int],
    path_for_scale: Callable[[int], Path],
    layers: Sequence[int],
    d_model: int,
    result: CalibrationResult,
    diagnostics_every: int = 25,
    on_snapshot: Callable[[ScaleSnapshot], None] | None = None,
    on_diagnostics: Callable[[list[dict]], None] | None = None,
) -> Callable[[dict], None]:
    """Build the ``on_prompt`` observer that writes nested scale snapshots.

    The observer writes a snapshot the moment ``n_done`` reaches a scale point.
    It never writes one twice, and it never writes one whose ``n_prompts``
    disagrees with its name.
    """
    wanted = sorted({int(point) for point in scale_points})
    started = time.perf_counter()

    def observe(payload: dict) -> None:
        if payload["status"] != "ok":
            result.n_skipped += 1
            return
        n_done = int(payload["n_done"])
        result.n_done = n_done

        row = {
            "prompt_index": int(payload["prompt_index"]),
            "n_done": n_done,
            "seq_len": int(payload["seq_len"]),
            "n_valid_positions": int(payload["n_valid_positions"]),
            "elapsed_seconds": round(float(payload["elapsed_seconds"]), 3),
            "checkpoint_written": bool(payload["checkpoint_written"]),
            "layers": layer_diagnostics(
                payload["per_prompt_jacobians"], payload["jacobian_sum"], n_done
            ),
        }
        result.diagnostics.append(row)

        if n_done in wanted:
            jacobian_sum = payload["jacobian_sum"]
            lens = JacobianLens(
                jacobians={
                    layer: jacobian_sum[layer] / n_done for layer in sorted(layers)
                },
                n_prompts=n_done,
                d_model=int(d_model),
            )
            path = Path(path_for_scale(n_done))
            _atomic_torch_save(lens, path)
            snapshot = ScaleSnapshot(
                scale=n_done,
                n_prompts=n_done,
                path=str(path),
                checksum=file_sha256(str(path)),
                layers=tuple(sorted(layers)),
                d_model=int(d_model),
                written_seconds=round(time.perf_counter() - started, 2),
            )
            result.snapshots[n_done] = snapshot
            if on_snapshot is not None:
                on_snapshot(snapshot)

        if on_diagnostics is not None and n_done % max(1, diagnostics_every) == 0:
            on_diagnostics(result.diagnostics)

    return observe


# ------------------------------------------------------------------ the run


def run_calibration(
    model,
    records: Sequence[CorpusRecord],
    *,
    plan: CapturePlan,
    scale_points: Sequence[int],
    store: CalibrationStore,
    checkpoint_every: int = 25,
    diagnostics_every: int = 25,
) -> CalibrationResult:
    """Fit ``J_l`` at every layer in ``plan``, snapshotting the nested scales.

    ``records`` must already be in nested scale order and long enough for the
    largest scale point; :func:`filter_records_by_tokens` should have run first
    so that no prompt is skipped and ``n_done`` tracks the prompt index exactly.

    Resume is upstream's: the accumulator checkpoint under ``store`` is written
    atomically and reloaded automatically, and upstream refuses a checkpoint
    fitted with different ``source_layers`` / ``target_layer`` / ``skip_first``.
    Snapshots already on disk are recovered without refitting.

    Raises:
        ScaleNotReachedError: If fitting ends before the largest scale point.
    """
    wanted = sorted({int(point) for point in scale_points})
    largest = max(wanted)
    prompts = [record.text for record in nested_subset(records, largest)]

    result = CalibrationResult(checkpoint_path=str(store.checkpoint_path))

    # Recover snapshots a previous session already wrote.
    for scale in wanted:
        path = store.snapshot_path(scale)
        if path.is_file():
            existing = JacobianLens.load(str(path))
            if existing.n_prompts == scale:
                result.snapshots[scale] = ScaleSnapshot(
                    scale=scale,
                    n_prompts=existing.n_prompts,
                    path=str(path),
                    checksum=file_sha256(str(path)),
                    layers=tuple(existing.source_layers),
                    d_model=existing.d_model,
                    written_seconds=0.0,
                )

    if largest in result.snapshots:
        result.n_done = largest
        return result

    def record_snapshot(snapshot: ScaleSnapshot) -> None:
        store.save("scale_snapshot", f"scale{snapshot.scale}", snapshot.to_dict())

    def record_diagnostics(rows: list[dict]) -> None:
        store.save(
            "fit_diagnostics",
            "rolling",
            {
                "n_rows": len(rows),
                "layers": list(plan.layers),
                "rows": rows[-diagnostics_every * 4 :],
                "objective": "not_applicable_estimator_is_a_sample_mean",
            },
        )

    observer = snapshot_scales(
        scale_points=wanted,
        path_for_scale=store.snapshot_path,
        layers=plan.layers,
        d_model=plan.d_model,
        result=result,
        diagnostics_every=diagnostics_every,
        on_snapshot=record_snapshot,
        on_diagnostics=record_diagnostics,
    )

    store.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    fit(
        model,
        prompts,
        source_layers=list(plan.layers),
        target_layer=plan.target_layer,
        dim_batch=plan.dim_batch,
        max_seq_len=plan.max_seq_len,
        skip_first=plan.skip_first,
        checkpoint_path=str(store.checkpoint_path),
        checkpoint_every=checkpoint_every,
        resume=True,
        on_prompt=observer,
    )
    result.elapsed_seconds = time.perf_counter() - started

    missing = [scale for scale in wanted if scale not in result.snapshots]
    if missing:
        raise ScaleNotReachedError(
            f"fitting finished with n_done={result.n_done} but scale point(s) "
            f"{missing} were never reached ({result.n_skipped} prompt(s) skipped). "
            "Run filter_records_by_tokens before fitting, or supply more records. "
            "Refusing to name a snapshot after a scale it does not have."
        )

    store.save(
        "fit_diagnostics",
        "summary",
        {
            **result.to_dict(),
            "capture_plan": plan.to_dict(),
            "diagnostics_checksum": payload_checksum(result.diagnostics),
        },
    )
    return result
