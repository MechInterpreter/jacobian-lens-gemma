# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Per-prompt Jacobian diagnostics for lens fitting.

The layer-21 investigation (docs/jspace_run_report.md §4) found the fitted
``J_21`` to be a 15–50× Frobenius-norm outlier relative to every other
fitted layer, with three candidate explanations: an unstable corpus-averaged
fit, dominance by a few heavy-tailed prompts, or a reproducible property of
the layer. Distinguishing them requires *per-prompt* statistics that
:func:`jlens.fitting.fit` never recorded. This module provides them through
``fit(..., on_prompt=recorder.on_prompt)``:

- per prompt: Frobenius norm, max |entry|, finiteness (checked with the
  bounded-memory :func:`jlens.pursuit.validate_finite`), sequence stats,
  wall time, GPU memory;
- running accumulation: Frobenius norm of the running-mean Jacobian, Welford
  mean/variance of the per-prompt norms, each prompt's norm contribution
  relative to the accumulated sum, its cosine alignment with the running
  mean so far, and the relative change it causes in the running mean;
- post-hoc: a dominance/stabilization summary (top prompts by norm, norm
  shares, tail behaviour of the running mean).

The recorder persists every row to a JSONL file *as it happens* (append +
flush), so diagnostics survive a crash and line up with ``fit``'s own
checkpoint/resume: on resume it reloads its rows and refuses indices that
would rewind or duplicate. Raw prompt text is never stored — only the
16-hex prompt hash (:func:`jlens.metadata.prompt_hashes` convention).

Interpretation boundary: these are descriptive statistics of one estimator
on one corpus. A large per-prompt norm marks an influential prompt, not a
"bad" prompt; stabilization of the running mean is evidence about estimator
variance, not about model internals by itself.
"""

from __future__ import annotations

import json
import math
import os
import statistics

import torch

from jlens.metadata import prompt_hashes
from jlens.pursuit import validate_finite

DIAGNOSTIC_ROW_SCHEMA = "jlens.fit_diagnostics.row.v1"
DIAGNOSTIC_SUMMARY_SCHEMA = "jlens.fit_diagnostics.summary.v1"


def _gpu_memory() -> dict:
    if not torch.cuda.is_available():
        return {"allocated_gb": None, "reserved_gb": None}
    return {
        "allocated_gb": round(torch.cuda.memory_allocated() / 2**30, 3),
        "reserved_gb": round(torch.cuda.memory_reserved() / 2**30, 3),
    }


class PromptDiagnosticsRecorder:
    """Observer for :func:`jlens.fitting.fit` recording per-prompt Jacobian
    diagnostics to a JSONL file.

    Args:
        path: JSONL file to append rows to (created if missing). On
            construction, existing rows are reloaded so a resumed ``fit``
            continues the same file; a row whose ``prompt_index`` does not
            advance past the last recorded one raises (guards against
            accidentally rerunning into the same file with ``resume=False``).
        expected_prompt_hashes: Optional full list of prompt hashes in fit
            order (e.g. the pilot run's recorded ``prompt_hashes``); each
            incoming prompt is verified against it by index, pinning the
            corpus identity prompt-by-prompt.
    """

    def __init__(
        self,
        path: str,
        *,
        expected_prompt_hashes: list[str] | None = None,
    ) -> None:
        self.path = path
        self.expected_prompt_hashes = expected_prompt_hashes
        self.rows: list[dict] = []
        # Welford state over per-prompt Frobenius norms (max over layers,
        # which for a single-layer fit is just that layer's norm).
        self._welford_n = 0
        self._welford_mean = 0.0
        self._welford_m2 = 0.0
        if os.path.exists(path):
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    if row.get("schema") != DIAGNOSTIC_ROW_SCHEMA:
                        raise ValueError(
                            f"{path}: unexpected row schema {row.get('schema')!r}"
                        )
                    self.rows.append(row)
                    if row["status"] == "ok":
                        self._welford_update(row["frobenius_norm_max_over_layers"])

    # ------------------------------------------------------------- internals

    def _welford_update(self, value: float) -> None:
        self._welford_n += 1
        delta = value - self._welford_mean
        self._welford_mean += delta / self._welford_n
        self._welford_m2 += delta * (value - self._welford_mean)

    @property
    def _welford_variance(self) -> float | None:
        if self._welford_n < 2:
            return None
        return self._welford_m2 / (self._welford_n - 1)

    def _last_index(self) -> int:
        return self.rows[-1]["prompt_index"] if self.rows else -1

    def _append(self, row: dict) -> None:
        self.rows.append(row)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()

    # ------------------------------------------------------------------ hook

    def on_prompt(self, event: dict) -> None:
        """``fit(..., on_prompt=...)`` callback. See the module docstring for
        what is recorded."""
        index = event["prompt_index"]
        if index <= self._last_index():
            raise ValueError(
                f"{self.path}: prompt_index {index} does not advance past the "
                f"last recorded index {self._last_index()}; refusing to mix "
                "runs in one diagnostics file (use a fresh file, or resume "
                "the fit from its checkpoint)"
            )
        prompt_hash = prompt_hashes([event["prompt"]])[0]
        if self.expected_prompt_hashes is not None:
            expected = self.expected_prompt_hashes[index]
            if prompt_hash != expected:
                raise ValueError(
                    f"prompt {index} hash {prompt_hash} != expected {expected}; "
                    "the fitting corpus does not match the reference run"
                )

        row: dict = {
            "schema": DIAGNOSTIC_ROW_SCHEMA,
            "prompt_index": index,
            "prompt_hash": prompt_hash,
            "status": event["status"],
            "elapsed_seconds": round(event["elapsed_seconds"], 3),
            "n_done": event["n_done"],
            "checkpoint_path": event.get("checkpoint_path"),
            "checkpoint_written": event.get("checkpoint_written", False),
            "gpu_memory": _gpu_memory(),
        }
        if event["status"] != "ok":
            row["reason"] = event.get("reason")
            self._append(row)
            return

        row["seq_len"] = event["seq_len"]
        row["n_valid_positions"] = event["n_valid_positions"]

        per_prompt = event["per_prompt_jacobians"]
        jacobian_sum = event["jacobian_sum"]
        n_done = event["n_done"]  # includes this prompt
        per_layer: dict[str, dict] = {}
        norms: list[float] = []
        for layer in sorted(per_prompt):
            J = per_prompt[layer].float()
            S = jacobian_sum[layer].float()
            norm = float(J.norm())
            norms.append(norm)
            mean_after = S / n_done
            sum_norm = float(S.norm())
            entry: dict = {
                "frobenius_norm": norm,
                "max_abs": float(J.abs().max()),
                "all_finite": validate_finite(J),
                "running_mean_frobenius_norm": float(mean_after.norm()),
                "contribution_fraction_of_sum_norm": (
                    norm / sum_norm if sum_norm > 0 else None
                ),
            }
            if n_done > 1:
                mean_before = (S - J) / (n_done - 1)
                before_norm = float(mean_before.norm())
                denominator = float(mean_before.norm()) * float(J.norm())
                entry["alignment_cosine_with_running_mean"] = (
                    float((J * mean_before).sum()) / denominator
                    if denominator > 0
                    else None
                )
                entry["running_mean_relative_change"] = (
                    float((mean_after - mean_before).norm()) / before_norm
                    if before_norm > 0
                    else None
                )
            else:
                entry["alignment_cosine_with_running_mean"] = None
                entry["running_mean_relative_change"] = None
            per_layer[str(layer)] = entry

        row["per_layer"] = per_layer
        row["frobenius_norm_max_over_layers"] = max(norms)
        self._welford_update(max(norms))
        row["running_norm_mean"] = self._welford_mean
        row["running_norm_variance"] = self._welford_variance
        self._append(row)

    # --------------------------------------------------------------- outputs

    def summary(self, *, top_n: int = 5, tail_window: int = 10) -> dict:
        """Dominance and stabilization summary over the recorded rows.

        Reports per-prompt norm statistics, the ``top_n`` largest-norm
        prompts with their share of the total norm mass (sum of per-prompt
        Frobenius norms — a linear proxy for influence on the running sum),
        and how the running-mean norm and its per-prompt relative change
        behave over the final ``tail_window`` prompts. Descriptive only.
        """
        ok_rows = [r for r in self.rows if r["status"] == "ok"]
        skipped = [r for r in self.rows if r["status"] != "ok"]
        if not ok_rows:
            return {
                "schema": DIAGNOSTIC_SUMMARY_SCHEMA,
                "n_prompts_ok": 0,
                "n_prompts_skipped": len(skipped),
            }
        norms = [r["frobenius_norm_max_over_layers"] for r in ok_rows]
        total = sum(norms)
        ranked = sorted(ok_rows, key=lambda r: (-r["frobenius_norm_max_over_layers"], r["prompt_index"]))
        top = [
            {
                "prompt_index": r["prompt_index"],
                "prompt_hash": r["prompt_hash"],
                "frobenius_norm": r["frobenius_norm_max_over_layers"],
                "share_of_total_norm_mass": (
                    r["frobenius_norm_max_over_layers"] / total if total > 0 else None
                ),
            }
            for r in ranked[:top_n]
        ]
        running_means = [
            layer_entry["running_mean_frobenius_norm"]
            for r in ok_rows
            for layer_entry in [next(iter(r["per_layer"].values()))]
        ]
        tail = ok_rows[-tail_window:]
        tail_changes = [
            entry["running_mean_relative_change"]
            for r in tail
            for entry in r["per_layer"].values()
            if entry["running_mean_relative_change"] is not None
        ]
        half = len(running_means) // 2
        return {
            "schema": DIAGNOSTIC_SUMMARY_SCHEMA,
            "n_prompts_ok": len(ok_rows),
            "n_prompts_skipped": len(skipped),
            "all_finite": all(
                entry["all_finite"]
                for r in ok_rows
                for entry in r["per_layer"].values()
            ),
            "per_prompt_norm": {
                "min": min(norms),
                "median": statistics.median(norms),
                "max": max(norms),
                "mean": self._welford_mean,
                "variance": self._welford_variance,
                "std": (
                    math.sqrt(self._welford_variance)
                    if self._welford_variance is not None
                    else None
                ),
                "max_over_median": (
                    max(norms) / statistics.median(norms)
                    if statistics.median(norms) > 0
                    else None
                ),
            },
            "top_prompts_by_norm": top,
            "top1_share_of_total_norm_mass": top[0]["share_of_total_norm_mass"],
            "top_n_share_of_total_norm_mass": (
                sum(t["share_of_total_norm_mass"] for t in top)
                if total > 0
                else None
            ),
            "running_mean_frobenius_norm": {
                "first": running_means[0],
                "at_half": running_means[half] if half < len(running_means) else None,
                "final": running_means[-1],
                "relative_change_over_tail_window": (
                    abs(running_means[-1] - running_means[-min(tail_window, len(running_means))])
                    / running_means[-min(tail_window, len(running_means))]
                    if running_means[-min(tail_window, len(running_means))] > 0
                    else None
                ),
            },
            "max_running_mean_relative_change_in_tail": (
                max(tail_changes) if tail_changes else None
            ),
            "tail_window": tail_window,
            "note": (
                "Descriptive statistics of the corpus-averaged Jacobian "
                "estimator; norm shares flag influential prompts, they do "
                "not by themselves establish instability or dominance."
            ),
        }

    def write_csv(self, path: str) -> None:
        """Flatten rows (one line per prompt x layer) to CSV for inspection."""
        import csv

        fieldnames = [
            "prompt_index", "prompt_hash", "status", "layer", "seq_len",
            "n_valid_positions", "elapsed_seconds", "frobenius_norm",
            "max_abs", "all_finite", "running_mean_frobenius_norm",
            "contribution_fraction_of_sum_norm",
            "alignment_cosine_with_running_mean",
            "running_mean_relative_change", "running_norm_mean",
            "running_norm_variance", "n_done", "checkpoint_written",
            "gpu_allocated_gb", "gpu_reserved_gb", "reason",
        ]
        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            for row in self.rows:
                base = {
                    "prompt_index": row["prompt_index"],
                    "prompt_hash": row["prompt_hash"],
                    "status": row["status"],
                    "seq_len": row.get("seq_len"),
                    "n_valid_positions": row.get("n_valid_positions"),
                    "elapsed_seconds": row["elapsed_seconds"],
                    "running_norm_mean": row.get("running_norm_mean"),
                    "running_norm_variance": row.get("running_norm_variance"),
                    "n_done": row["n_done"],
                    "checkpoint_written": row.get("checkpoint_written"),
                    "gpu_allocated_gb": row["gpu_memory"]["allocated_gb"],
                    "gpu_reserved_gb": row["gpu_memory"]["reserved_gb"],
                    "reason": row.get("reason"),
                }
                if row["status"] != "ok":
                    writer.writerow({**base, "layer": None})
                    continue
                for layer, entry in sorted(row["per_layer"].items()):
                    writer.writerow({**base, "layer": layer, **entry})

    def write_summary(self, path: str, **kwargs) -> None:
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(self.summary(**kwargs), handle, indent=1, sort_keys=True)
            handle.write("\n")
