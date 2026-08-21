"""Clean-loading-first selection for J-lens/R-lens causal studies.

The selector in this module is deliberately blind to intervention outcomes.
It consumes only clean activation/lens measurements and chooses both the
instrument and contiguous layer band before a causal unit can be evaluated.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence

from jlens.lens import JacobianLens
from jlens.mmpilot.store import payload_checksum

LOADING_FIRST_VERSION = "mmpilot.loading_first_instrument_selection.v1"


class LoadingFirstRefused(RuntimeError):
    """The clean measurements cannot license a causal instrument."""


def combine_disjoint_layer_lenses(
    lenses: Sequence[JacobianLens],
    *,
    expected_layers: Sequence[int],
) -> JacobianLens:
    """Join identically fitted lens shards that cover disjoint layer ranges.

    This is deliberately *not* :meth:`JacobianLens.merge`: ``merge`` averages
    lenses fitted on disjoint prompt populations but the same layers.  Here the
    prompt population and estimator must be identical and the layer sets must
    be disjoint.  The result only concatenates their per-layer matrices so a
    paper-style clamp can span the complete contiguous range.
    """

    supplied = tuple(lenses)
    wanted = tuple(map(int, expected_layers))
    if not supplied:
        raise LoadingFirstRefused("at least one lens shard is required")
    if not wanted or tuple(range(wanted[0], wanted[-1] + 1)) != wanted:
        raise LoadingFirstRefused(
            f"expected_layers must be one sorted contiguous band, got {wanted}"
        )
    first = supplied[0]
    jacobians = {}
    for lens in supplied:
        if lens.d_model != first.d_model or lens.n_prompts != first.n_prompts:
            raise LoadingFirstRefused(
                "layer shards disagree on d_model or fitted-prompt count"
            )
        overlap = sorted(set(jacobians) & set(lens.source_layers))
        if overlap:
            raise LoadingFirstRefused(
                f"layer shards overlap at {overlap}; concatenation is ambiguous"
            )
        jacobians.update(lens.jacobians)
    if tuple(sorted(jacobians)) != wanted:
        missing = sorted(set(wanted) - set(jacobians))
        extra = sorted(set(jacobians) - set(wanted))
        raise LoadingFirstRefused(
            f"layer shards do not exactly cover the declared band; "
            f"missing={missing}, extra={extra}"
        )
    return JacobianLens(
        jacobians=jacobians,
        n_prompts=first.n_prompts,
        d_model=first.d_model,
    )


def select_loading_instrument(
    rows_by_instrument: Mapping[str, Sequence[Mapping]],
    *,
    tasks: Sequence[str],
    layers: Sequence[int],
    position_class: str = "final_prompt_token",
    min_source_cosine: float = 0.0,
    min_source_advantage: float = 0.0,
) -> dict:
    """Select an instrument and contiguous band from clean loading only.

    A layer passes for an instrument only when every required task has a row
    at the frozen position class and the task-median source cosine and source
    advantage exceed their frozen bars. This follows the paper's aggregation
    convention without allowing a missing task to disappear. The instrument's
    admissible band is its longest
    contiguous passing run (deeper run wins a length tie).  Instruments rank
    by band length, then the weakest task/layer source advantage, then the
    weakest source cosine, and finally name.  No causal field is accepted.
    """

    required_tasks = tuple(map(str, tasks))
    candidate_layers = tuple(sorted(set(map(int, layers))))
    if not required_tasks or not candidate_layers:
        raise LoadingFirstRefused("tasks and layers must both be nonempty")

    ranking = []
    for instrument, supplied in sorted(rows_by_instrument.items()):
        rows = [dict(row) for row in supplied]
        if any(bool(row.get("causal_result_consulted")) for row in rows):
            raise LoadingFirstRefused(
                f"instrument {instrument!r} contains a causal-contaminated row"
            )
        by_cell: dict[tuple[str, int], list[dict]] = {}
        for row in rows:
            if str(row.get("position_class")) != position_class:
                continue
            task = str(row.get("sample_id"))
            layer = int(row.get("layer"))
            if task in required_tasks and layer in candidate_layers:
                by_cell.setdefault((task, layer), []).append(row)

        layer_rows = []
        passing = []
        for layer in candidate_layers:
            task_rows = []
            complete = True
            for task in required_tasks:
                cell = by_cell.get((task, layer), [])
                if not cell:
                    complete = False
                    continue
                source_cosine = float(
                    statistics.median(float(row["source_cosine"]) for row in cell)
                )
                source_advantage = float(
                    statistics.median(
                        float(row["source_advantage"]) for row in cell
                    )
                )
                if not math.isfinite(source_cosine) or not math.isfinite(
                    source_advantage
                ):
                    raise LoadingFirstRefused(
                        f"instrument {instrument!r} has a non-finite clean row"
                    )
                task_rows.append(
                    {
                        "task": task,
                        "source_cosine": source_cosine,
                        "source_advantage": source_advantage,
                    }
                )
            median_source_cosine = (
                float(statistics.median(row["source_cosine"] for row in task_rows))
                if task_rows
                else None
            )
            median_source_advantage = (
                float(
                    statistics.median(
                        row["source_advantage"] for row in task_rows
                    )
                )
                if task_rows
                else None
            )
            passed = (
                complete
                and median_source_cosine is not None
                and median_source_advantage is not None
                and median_source_cosine > float(min_source_cosine)
                and median_source_advantage > float(min_source_advantage)
            )
            if passed:
                passing.append(layer)
            layer_rows.append(
                {
                    "layer": layer,
                    "complete": complete,
                    "passed": passed,
                    "median_source_cosine": median_source_cosine,
                    "median_source_advantage": median_source_advantage,
                    "tasks": task_rows,
                }
            )

        runs: list[list[int]] = []
        for layer in passing:
            if not runs or layer != runs[-1][-1] + 1:
                runs.append([layer])
            else:
                runs[-1].append(layer)
        selected_band = max(runs, key=lambda run: (len(run), run[-1]), default=[])
        selected_cells = [
            layer_row
            for layer_row in layer_rows
            if layer_row["layer"] in selected_band
        ]
        weakest_advantage = min(
            (row["median_source_advantage"] for row in selected_cells),
            default=float("-inf"),
        )
        weakest_cosine = min(
            (row["median_source_cosine"] for row in selected_cells),
            default=float("-inf"),
        )
        ranking.append(
            {
                "instrument": str(instrument),
                "passing_layers": passing,
                "contiguous_runs": runs,
                "selected_band": selected_band,
                "band_length": len(selected_band),
                "weakest_source_advantage": weakest_advantage,
                "weakest_source_cosine": weakest_cosine,
                "layer_evidence": layer_rows,
            }
        )

    ranking.sort(
        key=lambda row: (
            -int(row["band_length"]),
            -float(row["weakest_source_advantage"]),
            -float(row["weakest_source_cosine"]),
            str(row["instrument"]),
        )
    )
    selected = ranking[0] if ranking and ranking[0]["band_length"] else None
    payload = {
        "version": LOADING_FIRST_VERSION,
        "position_class": position_class,
        "tasks": list(required_tasks),
        "candidate_layers": list(candidate_layers),
        "min_source_cosine": float(min_source_cosine),
        "min_source_advantage": float(min_source_advantage),
        "causal_result_consulted": False,
        "ranking": ranking,
        "verdict": (
            "LOADING_FIRST_INSTRUMENT_GO"
            if selected is not None
            else "LOADING_FIRST_INSTRUMENT_NO_GO"
        ),
        "selected_instrument": (
            selected["instrument"] if selected is not None else None
        ),
        "selected_band": selected["selected_band"] if selected is not None else [],
    }
    return {**payload, "selection_digest": payload_checksum(payload)}
