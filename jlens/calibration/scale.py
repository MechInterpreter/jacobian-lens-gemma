# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Comparing nested scale points, and the predeclared plateau rule.

The question this module answers is narrow: **does more calibration data buy
anything at layers earlier than 38, and where does it stop buying?**

Two rules decide it, and both are fixed here before any number exists:

* :data:`PLATEAU_RULE` — when, if ever, extending past 10,000 prompts is
  justified. It is a conjunction of four clauses, three of which must show that
  an *earlier* layer is still genuinely climbing, and one of which forbids
  trading a late layer away to get there.
* :func:`select_scale` — which scale point the study reports. Parsimony:
  the smallest scale that already reaches the largest scale's eligible set.

Both rule digests are bound into the run fingerprint, so editing either
invalidates stored results rather than re-deciding on them.

A note on what a plateau means here. The estimator is a sample mean
(:mod:`jlens.calibration`), so the sequence of lenses converges to a fixed
population Jacobian, and every metric derived from it converges too. A plateau
is therefore the *expected* outcome, not a disappointment — and the upstream
README says so outright: *"Quality saturates quickly (§9.3); ~100 prompts is
usable."* The rule is written to stop, because the primary source predicts
stopping is correct.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

from jlens.mmpilot.store import payload_checksum

__all__ = [
    "PLATEAU_RULE",
    "PLATEAU_REACHED",
    "PLATEAU_STILL_IMPROVING",
    "PlateauRule",
    "compare_scales",
    "evaluate_plateau",
    "select_scale",
]

PLATEAU_REACHED = "PLATEAU_REACHED"
PLATEAU_STILL_IMPROVING = "STILL_IMPROVING_EXTENSION_JUSTIFIED"

#: Metrics carried into the comparison table for every (layer, scale).
COMPARED_METRICS = (
    "mean_reciprocal_rank",
    "median_midrank",
    "median_optimistic_rank",
    "median_pessimistic_rank",
    "top10_inclusion",
    "unique_top1_agreement",
    "argmax_top1_agreement",
    "tied_at_max_rate",
    "mean_top10_overlap",
    "mean_margin_over_best_non_target",
)


@dataclass(frozen=True)
class PlateauRule:
    """When extending past the largest planned scale point is justified.

    All four clauses must hold. Attributes mirror the frozen config.

    Attributes:
        earlier_layers: The layers the rule is about. Layer 38 is already
            established, so improvement there is not a reason to spend another
            week of L4 time; the question is whether anything *earlier* is
            still climbing.
        min_mrr_improvement / min_median_midrank_relative_drop: Both must be
            met by the same layer. One metric alone moves too easily.
        min_continuation_fraction: The last step's improvement must be at least
            this fraction of the previous step's. A layer whose gains are
            already decaying is plateauing, however positive the last delta is.
        require_no_degeneracy_increase: A layer that improves its MRR while its
            tied-at-maximum rate rises is not improving; it is concentrating
            mass in a tie block.
        require_no_eligible_layer_lost: Extending must not be bought by losing
            a layer that already passed.
    """

    version: str = "conservative-earlier-layer-improvement-v1"
    earlier_layers: tuple[int, ...] = (8, 14, 20, 26, 32)
    min_mrr_improvement: float = 0.05
    min_median_midrank_relative_drop: float = 0.20
    min_continuation_fraction: float = 0.50
    require_no_degeneracy_increase: bool = True
    require_no_eligible_layer_lost: bool = True
    declared_before_results: bool = True

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["earlier_layers"] = list(self.earlier_layers)
        return payload

    @property
    def digest(self) -> str:
        return payload_checksum(self.to_dict())

    def text(self) -> str:
        return f"""\
PREDECLARED PLATEAU RULE — {self.version}
Digest: {self.digest}
Fixed before any scale-comparison number exists. Not revisable afterwards.

Extending past the largest planned scale point is justified ONLY IF ALL hold:

  1. Some layer in {list(self.earlier_layers)} that FAILED the gate at the
     second-largest scale improves at the largest scale by
       MRR increase >= {self.min_mrr_improvement:.2f}   AND
       median midrank relative drop >= {self.min_median_midrank_relative_drop:.0%}
     — both, on the development set.
  2. That layer's tied-at-maximum rate does not increase over the same step.
  3. That layer's last improvement is >= {self.min_continuation_fraction:.0%} of its previous
     improvement (still climbing, not decaying into a plateau).
  4. No layer that passed at the second-largest scale fails at the largest.

Otherwise the verdict is {PLATEAU_REACHED} and the study stops.

Even when the rule fires, nothing runs automatically: RUN_OPTIONAL_LARGE_SCALE
and CONFIRM_OPTIONAL_LARGE_SCALE_BUDGET are separate manual switches, and the
optional scales cost 25-50x the 1,000-prompt point.
"""


#: The single rule this workflow runs under.
PLATEAU_RULE = PlateauRule()


# ------------------------------------------------------------- comparison


def _metrics_of(verdict: Mapping) -> dict:
    """The J-lens metric row from one layer verdict, plus its control margins."""
    jlens = verdict["metrics"]["j_lens"]
    row = {name: jlens.get(name) for name in COMPARED_METRICS}
    row["margin_over_permuted"] = (
        jlens["mean_reciprocal_rank"]
        - verdict["metrics"]["permuted"]["mean_reciprocal_rank"]
    )
    row["margin_over_random"] = (
        jlens["mean_reciprocal_rank"]
        - verdict["metrics"]["random"]["mean_reciprocal_rank"]
    )
    row["margin_over_wrong_layer"] = (
        jlens["mean_reciprocal_rank"]
        - verdict["metrics"]["wrong_layer"]["mean_reciprocal_rank"]
    )
    row["logit_lens_mrr"] = verdict["metrics"]["logit_lens"]["mean_reciprocal_rank"]
    return row


def compare_scales(
    validation_by_scale: Mapping[int, Mapping[int, Mapping]],
    *,
    layers: Sequence[int],
) -> dict:
    """Assemble the per-layer, per-scale comparison table and its deltas.

    Args:
        validation_by_scale: ``{scale: {layer: verdict}}`` as produced by
            :func:`jlens.calibration.gate.evaluate_calibration_layers`.
        layers: The layer grid, in order.

    Returns:
        A payload with a ``rows`` table (one row per layer per scale), a
        ``deltas`` table (one row per layer per consecutive scale pair), and
        ``eligible_by_scale``.
    """
    scales = sorted(int(scale) for scale in validation_by_scale)
    if not scales:
        raise ValueError("compare_scales needs at least one scale point")
    ordered_layers = [int(layer) for layer in layers]

    rows: list[dict] = []
    for scale in scales:
        for layer in ordered_layers:
            verdict = validation_by_scale[scale][layer]
            rows.append(
                {
                    "scale": scale,
                    "layer": layer,
                    "passed": bool(verdict["passed"]),
                    "status": verdict["status"],
                    "failed_checks": list(verdict["failed_checks"]),
                    "n_distinct_target_tokens": verdict["target_diversity"][
                        "n_distinct_target_tokens"
                    ],
                    "max_target_token_share": verdict["target_diversity"][
                        "max_target_token_share"
                    ],
                    "fold_mrr": verdict["fold_mrr"]["j_lens"],
                    **_metrics_of(verdict),
                }
            )

    deltas: list[dict] = []
    for smaller, larger in zip(scales, scales[1:], strict=False):
        for layer in ordered_layers:
            before = validation_by_scale[smaller][layer]
            after = validation_by_scale[larger][layer]
            before_metrics, after_metrics = _metrics_of(before), _metrics_of(after)
            midrank_before = float(before_metrics["median_midrank"])
            midrank_after = float(after_metrics["median_midrank"])
            relative_drop = (
                (midrank_before - midrank_after) / midrank_before
                if midrank_before > 0
                else 0.0
            )
            deltas.append(
                {
                    "layer": layer,
                    "from_scale": smaller,
                    "to_scale": larger,
                    "delta_mrr": float(after_metrics["mean_reciprocal_rank"])
                    - float(before_metrics["mean_reciprocal_rank"]),
                    "median_midrank_relative_drop": relative_drop,
                    "delta_tied_at_max_rate": float(after_metrics["tied_at_max_rate"])
                    - float(before_metrics["tied_at_max_rate"]),
                    "passed_before": bool(before["passed"]),
                    "passed_after": bool(after["passed"]),
                }
            )

    eligible_by_scale = {
        scale: sorted(
            layer
            for layer in ordered_layers
            if validation_by_scale[scale][layer]["passed"]
        )
        for scale in scales
    }

    payload = {
        "scales": scales,
        "layers": ordered_layers,
        "rows": rows,
        "deltas": deltas,
        "eligible_by_scale": {str(k): v for k, v in eligible_by_scale.items()},
        "compared_metrics": list(COMPARED_METRICS),
        "nested": True,
    }
    payload["comparison_checksum"] = payload_checksum(payload)
    return payload


# ----------------------------------------------------------------- plateau


def _delta(comparison: Mapping, layer: int, from_scale: int, to_scale: int) -> dict | None:
    for row in comparison["deltas"]:
        if (
            row["layer"] == layer
            and row["from_scale"] == from_scale
            and row["to_scale"] == to_scale
        ):
            return row
    return None


def evaluate_plateau(
    comparison: Mapping, *, rule: PlateauRule = PLATEAU_RULE
) -> dict:
    """Apply the predeclared plateau rule to a scale comparison.

    With fewer than three scale points, clause 3 (the continuation check) has no
    previous step to compare against and cannot be satisfied; the verdict is
    ``PLATEAU_REACHED`` with that stated, rather than the clause being skipped.
    """
    scales = list(comparison["scales"])
    if len(scales) < 2:
        return {
            "rule": rule.to_dict(),
            "rule_digest": rule.digest,
            "verdict": PLATEAU_REACHED,
            "extension_justified": False,
            "reason": "fewer than two scale points; no step exists to evaluate",
            "clauses": [],
            "candidates": [],
        }

    largest, previous = scales[-1], scales[-2]
    earlier_previous = scales[-3] if len(scales) >= 3 else None

    candidates: list[dict] = []
    for layer in rule.earlier_layers:
        if layer not in comparison["layers"]:
            continue
        last = _delta(comparison, layer, previous, largest)
        if last is None or last["passed_before"]:
            continue  # clause 1 is about layers that FAILED at the previous scale

        improved = (
            last["delta_mrr"] >= rule.min_mrr_improvement
            and last["median_midrank_relative_drop"]
            >= rule.min_median_midrank_relative_drop
        )
        no_degeneracy = (
            last["delta_tied_at_max_rate"] <= 0.0
            if rule.require_no_degeneracy_increase
            else True
        )
        if earlier_previous is None:
            still_climbing = False
            continuation_detail = (
                "no earlier step available; the continuation clause cannot be met"
            )
        else:
            prior = _delta(comparison, layer, earlier_previous, previous)
            prior_delta = float(prior["delta_mrr"]) if prior else 0.0
            still_climbing = (
                prior_delta <= 0.0
                or last["delta_mrr"] >= rule.min_continuation_fraction * prior_delta
            )
            continuation_detail = (
                f"last dMRR {last['delta_mrr']:+.4f} vs previous {prior_delta:+.4f} "
                f"(need >= {rule.min_continuation_fraction:.0%} of previous)"
            )

        candidates.append(
            {
                "layer": layer,
                "improved": bool(improved),
                "no_degeneracy_increase": bool(no_degeneracy),
                "still_climbing": bool(still_climbing),
                "qualifies": bool(improved and no_degeneracy and still_climbing),
                "delta_mrr": last["delta_mrr"],
                "median_midrank_relative_drop": last["median_midrank_relative_drop"],
                "delta_tied_at_max_rate": last["delta_tied_at_max_rate"],
                "continuation_detail": continuation_detail,
            }
        )

    lost = [
        row["layer"]
        for row in comparison["deltas"]
        if row["from_scale"] == previous
        and row["to_scale"] == largest
        and row["passed_before"]
        and not row["passed_after"]
    ]
    qualifying = [row["layer"] for row in candidates if row["qualifies"]]

    clauses = [
        {
            "clause": "earlier_layer_materially_improves",
            "passed": bool(qualifying),
            "detail": (
                f"qualifying layers {qualifying}"
                if qualifying
                else f"no layer in {list(rule.earlier_layers)} that failed at "
                f"{previous} improved materially by {largest}"
            ),
        },
        {
            "clause": "no_degeneracy_increase",
            "passed": all(row["no_degeneracy_increase"] for row in candidates)
            if candidates
            else False,
            "detail": ", ".join(
                f"L{row['layer']} dTiedAtMax={row['delta_tied_at_max_rate']:+.4f}"
                for row in candidates
            )
            or "no candidate layers",
        },
        {
            "clause": "still_climbing",
            "passed": any(row["still_climbing"] for row in candidates),
            "detail": ", ".join(
                f"L{row['layer']}: {row['continuation_detail']}" for row in candidates
            )
            or "no candidate layers",
        },
        {
            "clause": "no_eligible_layer_lost",
            "passed": not lost if rule.require_no_eligible_layer_lost else True,
            "detail": f"layers lost between {previous} and {largest}: {lost}",
        },
    ]

    justified = all(clause["passed"] for clause in clauses)
    payload = {
        "rule": rule.to_dict(),
        "rule_digest": rule.digest,
        "from_scale": previous,
        "to_scale": largest,
        "verdict": PLATEAU_STILL_IMPROVING if justified else PLATEAU_REACHED,
        "extension_justified": bool(justified),
        "clauses": clauses,
        "failed_clauses": [c["clause"] for c in clauses if not c["passed"]],
        "candidates": candidates,
        "eligible_layers_lost": lost,
        "runs_automatically": False,
        "note": (
            "Even when justified, the optional extension requires "
            "RUN_OPTIONAL_LARGE_SCALE and CONFIRM_OPTIONAL_LARGE_SCALE_BUDGET "
            "to be set by hand."
        ),
    }
    payload["plateau_checksum"] = payload_checksum(payload)
    return payload


# ---------------------------------------------------------------- selection


def select_scale(comparison: Mapping) -> dict:
    """Choose the scale point the study reports.

    Parsimony: the **smallest** scale whose eligible-layer set equals the
    largest computed scale's set. If no smaller scale agrees, the largest is
    chosen. Fixed in advance so that "which scale we report" cannot become a
    free parameter selected after seeing the table.
    """
    scales = list(comparison["scales"])
    eligible = {int(k): list(v) for k, v in comparison["eligible_by_scale"].items()}
    largest = scales[-1]
    target = eligible[largest]

    chosen = largest
    reason = (
        f"no smaller scale reproduces the eligible set at {largest}; "
        f"the largest computed scale is reported"
    )
    for scale in scales:
        if eligible[scale] == target:
            chosen = scale
            reason = (
                f"smallest scale whose eligible set {target} equals the set at "
                f"{largest}"
            )
            break

    payload = {
        "selected_scale": int(chosen),
        "selection_rule": (
            "smallest scale whose development eligible-layer set equals the "
            "largest computed scale's set; otherwise the largest"
        ),
        "reason": reason,
        "eligible_by_scale": {str(k): v for k, v in eligible.items()},
        "eligible_at_selected_scale": eligible[chosen],
        "declared_before_results": True,
        "confirmation_not_consulted": True,
    }
    payload["selection_checksum"] = payload_checksum(payload)
    return payload
