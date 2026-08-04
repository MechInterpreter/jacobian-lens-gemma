# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Stage B: which layers are allowed to carry a causal claim, and why.

An earlier layer does not become testable by being written into a layer list.
Before any causal result may rest on it, the frozen text-calibrated lens has to
demonstrate that it reads out *something* at that layer — on text alone, on
prompts it has never seen, against controls, under a criterion fixed before the
numbers exist. That is all this module does. It fits nothing; every entry point
consumes a frozen :class:`~jlens.lens.JacobianLens` and refuses to modify it.

Why a new gate, and why it is not a weaker one
----------------------------------------------

The v2 recalibration validated layer 38 and failed 20, 26 and 32. Layer 32 is
the interesting failure: median target rank 1 and MRR about 0.71, but **zero**
unique exact top-1 agreements, and a top-10 overlap (0.113) that lost to the
wrong-layer control (0.150).

Those two facts are the same fact. A rank of 1 computed as "how many tokens
score *strictly* above the target" is 1 whenever nothing beats the target —
including when four thousand tokens tie with it at the maximum. In that state
``argmax`` returns whichever index the kernel visits first, so "exact top-1
agreement" measures the tie-break rule, not the lens; and a top-10 slice cut out
of a large tie block is an arbitrary ten of the tied tokens, which is why its
overlap with the model's real top-10 can fall below a wrong-layer control's.

So the gate is rebuilt around ties rather than relaxed around them:

* Every rank is reported under **three** explicit conventions — optimistic,
  pessimistic and midrank — and the *midrank* is what the criterion uses.
  Midrank is the only one of the three that is invariant to how ties are
  broken, so it is the only one that can carry a threshold.
* The **tied-at-maximum rate** becomes a blocking degeneracy check. A lens whose
  argmax is a tie on most prompts is not reading anything out, whatever its
  optimistic rank says, and this clause fails it directly.
* Unique exact top-1 agreement is still computed and still printed. It is
  demoted from *the* criterion to a reported statistic, because under ties it is
  a property of the tie-break, not of the lens.
* The **wrong-layer control keeps a blocking margin** — moved from top-10
  overlap, where a tie block makes it meaningless, onto MRR, where it is the
  same concern stated on a metric that survives ties.

Three of those four clauses are strictly harder than the v2 conjunction. The one
that is softer — dropping the unique-top-1 floor — is softer because the
quantity it thresholded is not well defined in the presence of ties.

What "predeclared" can and cannot mean here
-------------------------------------------

Layer 32's v2 numbers were known when this gate was written; pretending
otherwise would be worse than saying so. What is genuinely fixed in advance is
everything that decides the outcome: the metrics, the tie convention, the
thresholds, the controls, the fold structure, and the held-out prompt set, all
frozen before a single new number is computed and all bound into
:attr:`LayerValidityGate.digest`, which participates in the resume fingerprint.
Editing any of them invalidates stored results rather than rescoring them.

This gate cannot be satisfied by tuning after the fact, and nobody — including
the author — knows whether layer 32 passes it. That is the property that makes
it evidence.
"""

from __future__ import annotations

import hashlib
import math
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass

import torch

from jlens.mmlocalize.layers import LOCALIZATION_LAYERS
from jlens.mmpilot.store import payload_checksum

#: Protocol tag on every artifact. Distinct from the v2 tag and from the
#: layer-32 confirmatory tag: these results must never be mistaken for either.
VALIDITY_PROTOCOL = "mmlocalize-multilayer-tie-aware-native-readout-v1"

#: Rank conventions computed for every (prompt, variant, layer) row.
#:
#: ``optimistic``  strictly-better tokens + 1. What the v2 run reported. Best
#:                 case for the lens: a target buried in a tie block still
#:                 scores 1.
#: ``pessimistic`` strictly-better + tied-with-target. Worst case.
#: ``midrank``     strictly-better + (tied-with-target + 1) / 2. The average
#:                 rank over all tie-break orders, and the only convention here
#:                 that does not change when the tie-break does. **The criterion
#:                 uses this one.**
RANK_CONVENTIONS = ("optimistic", "pessimistic", "midrank")

#: The convention every threshold in :data:`LOCALIZATION_VALIDITY_GATE` is
#: stated against, written into every artifact.
CRITERION_RANK_CONVENTION = "midrank"

#: Randomised or mismatched controls. Same constructions as the v2 run, from the
#: same seed, so a control number here means what a control number meant there.
CONTROL_VARIANTS = ("permuted", "random", "wrong_layer")

#: Reported for context, never blocking. The ordinary logit lens is neither
#: randomised nor wrong-layer, so it cannot answer "would any matrix do this?".
DIAGNOSTIC_VARIANTS = ("logit_lens",)

READOUT_VARIANTS = ("j_lens", *CONTROL_VARIANTS, *DIAGNOSTIC_VARIANTS)

#: Controls that answer "is this better than noise?". The wrong-layer control
#: answers a different question — "is this about *this* layer?" — and carries
#: its own, separate margin.
NOISE_CONTROLS = ("permuted", "random")

#: Held-out prompts per layer. Four times the v2 sample, matching the layer-32
#: confirmatory run so the two are directly comparable.
N_VALIDATION_PROMPTS = 32

#: Deterministic folds for the stability clause. Prompt ``i`` joins fold
#: ``i % N_STABILITY_FOLDS`` — a fixed partition, not a resample, so the clause
#: cannot be re-rolled until it passes.
N_STABILITY_FOLDS = 4

#: Seed for the randomised control lenses. Matches the v2 run's control seed.
CONTROL_SEED = 1234

#: Seed choosing which eligible passages become the held-out set.
VALIDATION_PROMPT_SEED = 20260804

LAYER_ELIGIBLE = "ELIGIBLE"
LAYER_INELIGIBLE = "INELIGIBLE"

#: Selection is stratified on the model's final-layer argmax before any lens
#: readout is computed.  This prevents a deterministic sample with too few
#: target tokens from making the non-degeneracy gate impossible by construction.
VALIDATION_PROMPT_SELECTION_PROTOCOL = "target-token-stratified-stable-rank-v1"


class InsufficientTargetDiversityError(RuntimeError):
    """The independent prompt pool cannot satisfy the declared diversity floor."""


def select_target_diverse_prompts(
    pool: Sequence[str],
    *,
    n_prompts: int,
    min_distinct_target_tokens: int,
    excluded: Mapping[str, str],
    seed: int,
    target_token_for_prompt,
) -> tuple[list[str], dict]:
    """Choose a deterministic held-out set that can pass the diversity clause.

    ``target_token_for_prompt`` may run the frozen model's ordinary output path,
    but it must not inspect a J-lens or any candidate layer.  We first scan the
    independent pool for final-output target IDs, reserve one prompt for each of
    the required distinct IDs, and fill the remaining positions in the same
    stable hash order.  The threshold is not relaxed when the pool is short.
    """
    if min_distinct_target_tokens > n_prompts:
        raise ValueError(
            "min_distinct_target_tokens cannot exceed the held-out prompt count"
        )

    unique: dict[str, str] = {}
    for prompt in pool:
        sha = hashlib.sha256(str(prompt).encode()).hexdigest()
        if sha not in excluded:
            unique.setdefault(sha, str(prompt))
    ordered = sorted(
        unique.items(),
        key=lambda item: (
            hashlib.sha256(f"{seed}|{item[0]}".encode()).hexdigest(),
            item[0],
        ),
    )
    if len(ordered) < n_prompts:
        raise InsufficientTargetDiversityError(
            f"only {len(ordered)} independent prompts remain after exclusions; "
            f"the protocol requires exactly {n_prompts}"
        )

    candidates: list[dict] = []
    for order, (sha, prompt) in enumerate(ordered):
        try:
            target_token_id = int(target_token_for_prompt(prompt))
        except Exception as exc:
            raise RuntimeError(
                f"failed to discover the final-output target for prompt {sha}"
            ) from exc
        candidates.append(
            {
                "prompt": prompt,
                "prompt_sha256": sha,
                "target_token_id": target_token_id,
                "stable_order": order,
            }
        )

    first_by_target: dict[int, dict] = {}
    for row in candidates:
        first_by_target.setdefault(row["target_token_id"], row)
    if len(first_by_target) < min_distinct_target_tokens:
        raise InsufficientTargetDiversityError(
            f"the {len(candidates)}-prompt independent pool exposes only "
            f"{len(first_by_target)} distinct final-output target token(s); "
            f"the fixed gate requires at least {min_distinct_target_tokens}. "
            "Refusing before any J-lens readout is scored."
        )

    reserved = list(first_by_target.values())[:min_distinct_target_tokens]
    selected_shas = {row["prompt_sha256"] for row in reserved}
    selected = list(reserved)
    for row in candidates:
        if len(selected) >= n_prompts:
            break
        if row["prompt_sha256"] not in selected_shas:
            selected.append(row)
            selected_shas.add(row["prompt_sha256"])
    selected.sort(key=lambda row: (row["stable_order"], row["prompt_sha256"]))

    selected_targets = {row["target_token_id"] for row in selected}
    if len(selected_targets) < min_distinct_target_tokens:  # defensive invariant
        raise AssertionError("target-diverse selection lost its reserved target tokens")
    prompts = [row["prompt"] for row in selected]
    manifest = {
        "protocol": VALIDATION_PROMPT_SELECTION_PROTOCOL,
        "seed": int(seed),
        "pool_size": len(pool),
        "n_unique_unexcluded": len(candidates),
        "n_excluded": len(pool) - len(candidates),
        "n_prompts": len(prompts),
        "min_distinct_target_tokens": int(min_distinct_target_tokens),
        "n_available_distinct_target_tokens": len(first_by_target),
        "n_selected_distinct_target_tokens": len(selected_targets),
        "prompts": [
            {
                "prompt_sha256": row["prompt_sha256"],
                "target_token_id": row["target_token_id"],
                "stable_order": row["stable_order"],
            }
            for row in selected
        ],
    }
    manifest["selection_checksum"] = payload_checksum(manifest)
    return prompts, manifest


class LayerNotEligibleError(RuntimeError):
    """A causal or representational claim was requested for a failed layer.

    Raised, never downgraded. A layer that could not be shown to read anything
    out is a layer at which an intervention's effect has no interpretation; its
    diagnostic numbers are kept, and its intervention stage is skipped.
    """


# ------------------------------------------------------------------ the gate


@dataclass(frozen=True)
class LayerValidityGate:
    """The pass rule for one layer, fixed before any number exists.

    Every threshold is stated against the **midrank** convention
    (:data:`CRITERION_RANK_CONVENTION`).

    Attributes:
        n_prompts: Exact held-out prompts required. A short run is not a weaker
            pass; it is not a result.
        max_tied_at_max_rate: Ceiling on the fraction of prompts whose J-lens
            readout has a non-unique argmax. This is the clause layer 32's v2
            result would have had to answer: a readout that ties at the maximum
            on most prompts is reporting the tie-break rule.
        min_noise_control_mrr_ratio / min_noise_control_mrr_margin: The J-lens
            MRR must clear the permuted and random controls both
            multiplicatively and additively. Ratio alone lets two tiny numbers
            pass; margin alone lets two large ones.
        min_wrong_layer_mrr_margin: Additive margin over the wrong-layer
            control. Separate from the noise controls because it answers a
            different question — whether the readout is about *this* layer.
        max_median_midrank / min_top_k_inclusion / top_k: The rank criterion.
            Inclusion uses the **pessimistic** rank, so a tie block cannot
            manufacture membership.
        min_fold_mrr_fraction: Every fold's J-lens MRR must reach this fraction
            of the overall MRR, and must beat every control on its own fold. One
            lucky eighth of the prompts cannot carry a layer.
        min_distinct_target_tokens: Degeneracy floor. If every prompt's target
            is the same token, agreement measures the token.
    """

    n_prompts: int = N_VALIDATION_PROMPTS
    rank_convention: str = CRITERION_RANK_CONVENTION
    max_tied_at_max_rate: float = 0.50
    min_noise_control_mrr_ratio: float = 1.5
    min_noise_control_mrr_margin: float = 0.10
    min_wrong_layer_mrr_margin: float = 0.15
    max_median_midrank: float = 5.0
    min_top_k_inclusion: float = 0.50
    top_k: int = 10
    n_folds: int = N_STABILITY_FOLDS
    min_fold_mrr_fraction: float = 0.50
    min_distinct_target_tokens: int = 8
    controls: tuple[str, ...] = CONTROL_VARIANTS
    noise_controls: tuple[str, ...] = NOISE_CONTROLS
    version: str = VALIDITY_PROTOCOL

    def to_dict(self) -> dict:
        payload = asdict(self)
        for key in ("controls", "noise_controls"):
            payload[key] = list(payload[key])
        return payload

    @property
    def digest(self) -> str:
        """Checksum of the gate, bound into the resume fingerprint."""
        return payload_checksum(self.to_dict())


@dataclass(frozen=True)
class LegacyValidityGate:
    """The v2 / layer-32-confirmatory rule, applied only so it can be reported.

    Kept verbatim so the report can put the two side by side and show what
    changed. It never decides eligibility here — its top-1 clause is the one
    that ties make ill defined.
    """

    min_top1_agreement: float = 0.75
    control_top1_margin: float = 0.25
    #: The v2 conjunction also required the J-lens's mean top-10 overlap to beat
    #: every control's. That is the clause layer 32 failed, at 0.113 against the
    #: wrong-layer control's 0.150.
    require_top10_overlap_beats_controls: bool = True
    name: str = "v2_conjunction_and_layer32_confirmatory"

    def to_dict(self) -> dict:
        return asdict(self)


#: The single gate this workflow runs under.
LOCALIZATION_VALIDITY_GATE = LayerValidityGate()

#: The old rule, reported beside it.
LEGACY_VALIDITY_GATE = LegacyValidityGate()


def gate_text(
    gate: LayerValidityGate = LOCALIZATION_VALIDITY_GATE,
    legacy: LegacyValidityGate = LEGACY_VALIDITY_GATE,
) -> str:
    """The criterion block, printed before any result-producing cell runs."""
    return f"""\
PREDECLARED LAYER-VALIDITY GATE — {gate.version}
Fixed before any new number is computed. Not revisable after seeing results.
Criterion digest: {gate.digest}

Every threshold below is stated against the MIDRANK convention:
    midrank = (tokens scoring strictly above the target)
              + (tokens tied with the target + 1) / 2
Midrank is the mean rank over all tie-break orders, and is the only rank here
that does not change when the tie-break changes. Optimistic and pessimistic
ranks are computed and reported too, but nothing is decided on them.

A layer is ELIGIBLE only if ALL of the following hold, over exactly
{gate.n_prompts} held-out text prompts the lens never saw:

  1. coverage and non-degeneracy
       every variant scored on all {gate.n_prompts} prompts, every metric
       finite, at least {gate.min_distinct_target_tokens} distinct target tokens,
       and the J-lens's tied-at-maximum rate at most {gate.max_tied_at_max_rate:.2f}.
  2. beats the noise controls
       J-lens MRR >= {gate.min_noise_control_mrr_ratio:.1f}x AND
       >= +{gate.min_noise_control_mrr_margin:.2f} over BOTH the row-permuted and the
       norm-matched-random control.
  3. beats the wrong-layer control
       J-lens MRR >= wrong-layer MRR + {gate.min_wrong_layer_mrr_margin:.2f}.
  4. rank and top-k
       median midrank <= {gate.max_median_midrank:.1f} AND top-{gate.top_k} inclusion
       >= {gate.min_top_k_inclusion:.2f}, inclusion measured at the PESSIMISTIC rank so a
       tie block cannot manufacture membership.
  5. stability across held-out subsets
       over {gate.n_folds} fixed folds (prompt i -> fold i mod {gate.n_folds}), every fold's
       J-lens MRR beats every control on that same fold, and no fold falls below
       {gate.min_fold_mrr_fraction:.2f}x the overall MRR.

Reported, never blocking: unique exact top-1 agreement, argmax top-1 agreement,
tied-at-maximum rate, mean top-{gate.top_k} overlap, margin over the strongest
non-target token, and the ordinary logit lens as a diagnostic.

WHY THIS DIFFERS FROM THE OLD GATE ({legacy.name})
The old rule required unique exact top-1 agreement >= {legacy.min_top1_agreement:.2f} with every
control at least {legacy.control_top1_margin:.2f} below, plus mean top-10 overlap beating every
control. Layer 32 failed it with median rank 1 and MRR ~0.71 but zero unique
top-1 agreements — the same fact twice, because both quantities were being read
off a tie block at the maximum, where argmax reports the tie-break rule and a
top-10 slice is an arbitrary ten of the tied tokens.

The new gate addresses ties rather than lowering the bar. It ADDS a blocking
tied-at-maximum ceiling, ADDS a blocking wrong-layer MRR margin (the wrong-layer
concern that failed layer 32, restated on a metric ties do not corrupt), and
ADDS a blocking fold-stability clause. It DROPS only the unique-top-1 floor,
because under ties that quantity is a property of the tie-break and not of the
lens. Both gates are computed for every layer and both are printed.

An INELIGIBLE layer keeps its diagnostic numbers and is skipped causally.
"""


# --------------------------------------------------------------- scoring rows


def tie_aware_row(
    *,
    sample_index: int,
    prompt_sha: str,
    layer: int,
    variant: str,
    variant_logits: torch.Tensor,
    actual_logits: torch.Tensor,
    top_k: int = 10,
) -> dict:
    """Score one (prompt, variant, layer) triple, reporting ties explicitly.

    ``actual_logits`` are the model's real final-layer logits at the last prompt
    position; ``variant_logits`` are what the variant reads out from the
    layer-``layer`` residual at that same position. The target is the model's
    own argmax — the lens is judged against the model, not against a label.

    Returns every rank convention in :data:`RANK_CONVENTIONS` plus the tie
    counts they are derived from, so a reader can recompute any of them and see
    exactly how large the tie block was.
    """
    actual = actual_logits.detach().float().flatten().cpu()
    scores = variant_logits.detach().float().flatten().cpu()
    if scores.shape != actual.shape:
        raise ValueError(
            f"variant logits {tuple(scores.shape)} do not match actual logits "
            f"{tuple(actual.shape)}"
        )

    target = int(actual.argmax())
    target_score = scores[target]
    max_score = scores.max()

    n_strictly_above = int((scores > target_score).sum())
    n_tied_with_target = int((scores == target_score).sum())
    n_tied_at_max = int((scores == max_score).sum())

    optimistic = float(n_strictly_above + 1)
    pessimistic = float(n_strictly_above + n_tied_with_target)
    midrank = float(n_strictly_above + (n_tied_with_target + 1) / 2.0)

    # The best score achieved by anything other than the target. Non-positive
    # margin means the target is tied for the lead at best.
    without_target = scores.clone()
    without_target[target] = float("-inf")
    margin = float(target_score - without_target.max())

    actual_top = set(actual.topk(top_k).indices.tolist())
    variant_top = set(scores.topk(top_k).indices.tolist())
    return {
        "sample": int(sample_index),
        "prompt_sha256": prompt_sha,
        "layer": int(layer),
        "variant": variant,
        "target_token_id": target,
        "predicted_token_id": int(scores.argmax()),
        # The old metric: argmax equality under an arbitrary tie-break.
        "argmax_top1_agreement": bool(int(scores.argmax()) == target),
        # The honest one: the target is the sole maximum.
        "unique_top1_agreement": bool(n_strictly_above == 0 and n_tied_with_target == 1),
        "tied_at_max": bool(n_tied_at_max > 1),
        "n_tied_at_max": n_tied_at_max,
        "n_strictly_above_target": n_strictly_above,
        "n_tied_with_target": n_tied_with_target,
        "rank_optimistic": optimistic,
        "rank_pessimistic": pessimistic,
        "rank_midrank": midrank,
        "reciprocal_rank": 1.0 / midrank,
        # Pessimistic on purpose: a tie block must not manufacture membership.
        f"in_top{top_k}": bool(pessimistic <= top_k),
        "margin_over_best_non_target": margin,
        f"top{top_k}_overlap": len(actual_top & variant_top) / float(top_k),
        "rank_convention_used_for_mrr": CRITERION_RANK_CONVENTION,
    }


def summarize_variant(rows: Sequence[Mapping], *, top_k: int = 10) -> dict:
    """Aggregate one variant's rows at one layer into reported metrics."""
    if not rows:
        raise ValueError("cannot summarize an empty row set")
    mean = statistics.fmean
    return {
        "n_prompts": len(rows),
        "mean_reciprocal_rank": mean(row["reciprocal_rank"] for row in rows),
        "median_midrank": float(statistics.median(row["rank_midrank"] for row in rows)),
        "median_optimistic_rank": float(
            statistics.median(row["rank_optimistic"] for row in rows)
        ),
        "median_pessimistic_rank": float(
            statistics.median(row["rank_pessimistic"] for row in rows)
        ),
        "unique_top1_agreement": mean(
            1.0 if row["unique_top1_agreement"] else 0.0 for row in rows
        ),
        "argmax_top1_agreement": mean(
            1.0 if row["argmax_top1_agreement"] else 0.0 for row in rows
        ),
        "tied_at_max_rate": mean(1.0 if row["tied_at_max"] else 0.0 for row in rows),
        "mean_n_tied_at_max": mean(float(row["n_tied_at_max"]) for row in rows),
        f"top{top_k}_inclusion": mean(
            1.0 if row[f"in_top{top_k}"] else 0.0 for row in rows
        ),
        f"mean_top{top_k}_overlap": mean(row[f"top{top_k}_overlap"] for row in rows),
        "mean_margin_over_best_non_target": mean(
            row["margin_over_best_non_target"] for row in rows
        ),
    }


def summarize_layer(
    rows: Iterable[Mapping],
    *,
    layer: int,
    variants: Sequence[str] = READOUT_VARIANTS,
    top_k: int = 10,
) -> dict[str, dict]:
    """Per-variant metrics for one layer."""
    selected = [row for row in rows if int(row["layer"]) == int(layer)]
    metrics: dict[str, dict] = {}
    for variant in variants:
        variant_rows = [row for row in selected if row["variant"] == variant]
        if not variant_rows:
            raise ValueError(f"no rows for variant {variant!r} at layer {layer}")
        metrics[variant] = summarize_variant(variant_rows, top_k=top_k)
    return metrics


def fold_of(sample_index: int, *, n_folds: int = N_STABILITY_FOLDS) -> int:
    """Which stability fold a prompt belongs to. A fixed partition, not a draw."""
    return int(sample_index) % int(n_folds)


def fold_mrr(
    rows: Sequence[Mapping], *, variant: str, n_folds: int = N_STABILITY_FOLDS
) -> dict[int, float]:
    """``{fold: mean reciprocal rank}`` for one variant at one layer."""
    buckets: dict[int, list[float]] = {index: [] for index in range(n_folds)}
    for row in rows:
        if row["variant"] != variant:
            continue
        buckets[fold_of(row["sample"], n_folds=n_folds)].append(
            float(row["reciprocal_rank"])
        )
    return {
        fold: (statistics.fmean(values) if values else float("nan"))
        for fold, values in buckets.items()
    }


# ------------------------------------------------------------------- verdicts


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def evaluate_legacy_gate(
    metrics: Mapping[str, Mapping],
    *,
    legacy: LegacyValidityGate = LEGACY_VALIDITY_GATE,
    controls: Sequence[str] = CONTROL_VARIANTS,
    top_k: int = 10,
) -> dict:
    """Apply the old rule, for reporting only.

    Uses ``unique_top1_agreement`` as the old rule's top-1 quantity. The v2 run
    computed it with ``argmax``, which under ties is the same number only when
    the argmax is unique — so both are reported, and the difference between them
    is precisely the effect the new gate exists to handle.
    """
    jlens = metrics["j_lens"]
    control_metrics = {name: metrics[name] for name in controls}
    top1 = jlens["unique_top1_agreement"]
    overlap_key = f"mean_top{top_k}_overlap"

    checks = [
        {
            "check": "legacy_top1_floor",
            "passed": top1 >= legacy.min_top1_agreement,
            "detail": (
                f"unique top-1 {top1:.4f} vs floor {legacy.min_top1_agreement:.2f} "
                f"(argmax top-1 was {jlens['argmax_top1_agreement']:.4f}; they "
                f"differ exactly when the argmax is a tie)"
            ),
        },
        {
            "check": "legacy_control_top1_margin",
            "passed": all(
                control["unique_top1_agreement"] <= top1 - legacy.control_top1_margin
                for control in control_metrics.values()
            ),
            "detail": ", ".join(
                f"{name}={control['unique_top1_agreement']:.4f}"
                for name, control in control_metrics.items()
            ),
        },
    ]
    if legacy.require_top10_overlap_beats_controls:
        checks.append(
            {
                "check": "legacy_top10_overlap_beats_controls",
                "passed": all(
                    jlens[overlap_key] > control[overlap_key]
                    for control in control_metrics.values()
                ),
                "detail": (
                    f"J-lens {jlens[overlap_key]:.3f} vs "
                    + ", ".join(
                        f"{name}={control[overlap_key]:.3f}"
                        for name, control in control_metrics.items()
                    )
                    + " — this is the clause layer 32 failed in v2"
                ),
            }
        )
    return {
        "gate": legacy.to_dict(),
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
        "failed_checks": [c["check"] for c in checks if not c["passed"]],
        "is_binding": False,
        "reading": (
            "reported so the two gates can be compared; it does not decide "
            "eligibility here"
        ),
    }


def evaluate_layer_validity(
    rows: Sequence[Mapping],
    *,
    layer: int,
    gate: LayerValidityGate = LOCALIZATION_VALIDITY_GATE,
    legacy: LegacyValidityGate = LEGACY_VALIDITY_GATE,
) -> dict:
    """Apply the predeclared gate to one layer and return its eligibility.

    Every clause reports its own pass flag and its own numbers, so a failure
    names the clause that failed rather than the layer.
    """
    layer_rows = [row for row in rows if int(row["layer"]) == int(layer)]
    metrics = summarize_layer(layer_rows, layer=layer, top_k=gate.top_k)
    jlens = metrics["j_lens"]
    noise = {name: metrics[name] for name in gate.noise_controls}
    wrong_layer = metrics["wrong_layer"]

    target_tokens = {
        row["target_token_id"] for row in layer_rows if row["variant"] == "j_lens"
    }
    all_finite = all(
        _finite(value)
        for variant in metrics.values()
        for key, value in variant.items()
        if key != "n_prompts"
    )
    full_coverage = all(
        variant["n_prompts"] == gate.n_prompts for variant in metrics.values()
    )
    inclusion_key = f"top{gate.top_k}_inclusion"

    folds = {
        variant: fold_mrr(layer_rows, variant=variant, n_folds=gate.n_folds)
        for variant in ("j_lens", *gate.controls)
    }
    fold_beats_controls = {
        fold: all(
            folds["j_lens"][fold] > folds[control][fold] for control in gate.controls
        )
        for fold in range(gate.n_folds)
    }
    overall_mrr = jlens["mean_reciprocal_rank"]
    fold_floor = gate.min_fold_mrr_fraction * overall_mrr
    fold_above_floor = {
        fold: bool(value >= fold_floor) for fold, value in folds["j_lens"].items()
    }

    checks: list[dict] = [
        {
            "check": "coverage_and_nondegeneracy",
            "passed": bool(
                all_finite
                and full_coverage
                and len(target_tokens) >= gate.min_distinct_target_tokens
                and jlens["tied_at_max_rate"] <= gate.max_tied_at_max_rate
            ),
            "detail": (
                f"finite={all_finite}, every variant on {gate.n_prompts} "
                f"prompts={full_coverage}, distinct targets={len(target_tokens)} "
                f"(need >= {gate.min_distinct_target_tokens}), tied-at-max rate "
                f"{jlens['tied_at_max_rate']:.3f} (ceiling "
                f"{gate.max_tied_at_max_rate:.2f}; mean tie block "
                f"{jlens['mean_n_tied_at_max']:.1f} tokens)"
            ),
        },
        {
            "check": "mrr_beats_noise_controls",
            "passed": all(
                overall_mrr >= gate.min_noise_control_mrr_ratio * control["mean_reciprocal_rank"]
                and overall_mrr
                >= control["mean_reciprocal_rank"] + gate.min_noise_control_mrr_margin
                for control in noise.values()
            ),
            "detail": (
                f"J-lens MRR {overall_mrr:.5f} vs "
                + ", ".join(
                    f"{name}={control['mean_reciprocal_rank']:.5f}"
                    for name, control in noise.items()
                )
                + f" (need >= {gate.min_noise_control_mrr_ratio:.1f}x and "
                f">= +{gate.min_noise_control_mrr_margin:.2f})"
            ),
        },
        {
            "check": "mrr_beats_wrong_layer_by_margin",
            "passed": bool(
                overall_mrr
                >= wrong_layer["mean_reciprocal_rank"] + gate.min_wrong_layer_mrr_margin
            ),
            "detail": (
                f"J-lens MRR {overall_mrr:.5f} vs wrong-layer "
                f"{wrong_layer['mean_reciprocal_rank']:.5f}; need "
                f"+{gate.min_wrong_layer_mrr_margin:.2f}"
            ),
        },
        {
            "check": "median_rank_and_top_k",
            "passed": bool(
                jlens["median_midrank"] <= gate.max_median_midrank
                and jlens[inclusion_key] >= gate.min_top_k_inclusion
            ),
            "detail": (
                f"median midrank {jlens['median_midrank']:.2f} (ceiling "
                f"{gate.max_median_midrank:.1f}), top-{gate.top_k} inclusion "
                f"{jlens[inclusion_key]:.3f} (floor {gate.min_top_k_inclusion:.2f}); "
                f"optimistic median was {jlens['median_optimistic_rank']:.2f} and "
                f"pessimistic {jlens['median_pessimistic_rank']:.2f}"
            ),
        },
        {
            "check": "stable_across_heldout_subsets",
            "passed": bool(
                all(fold_beats_controls.values()) and all(fold_above_floor.values())
            ),
            "detail": (
                f"{sum(fold_beats_controls.values())}/{gate.n_folds} folds beat every "
                f"control, {sum(fold_above_floor.values())}/{gate.n_folds} folds at or "
                f"above {fold_floor:.4f} "
                f"({gate.min_fold_mrr_fraction:.2f}x overall MRR); per-fold J-lens MRR "
                + ", ".join(
                    f"f{fold}={value:.4f}" for fold, value in sorted(folds["j_lens"].items())
                )
            ),
        },
    ]

    eligible = all(check["passed"] for check in checks)
    legacy_result = evaluate_legacy_gate(metrics, legacy=legacy, top_k=gate.top_k)
    return {
        "protocol": VALIDITY_PROTOCOL,
        "layer": int(layer),
        "status": LAYER_ELIGIBLE if eligible else LAYER_INELIGIBLE,
        "eligible": eligible,
        "rank_convention": gate.rank_convention,
        "rank_conventions_reported": list(RANK_CONVENTIONS),
        "gate": gate.to_dict(),
        "gate_digest": gate.digest,
        "checks": checks,
        "failed_checks": [c["check"] for c in checks if not c["passed"]],
        "metrics": metrics,
        "fold_mrr": {name: dict(sorted(value.items())) for name, value in folds.items()},
        "fold_beats_all_controls": dict(sorted(fold_beats_controls.items())),
        "n_distinct_target_tokens": len(target_tokens),
        "legacy_gate": legacy_result,
        "gates_agree": bool(eligible == legacy_result["passed"]),
        "diagnostic_variants": list(DIAGNOSTIC_VARIANTS),
        "secondary_metrics_are_non_blocking": True,
    }


def evaluate_all_layers(
    rows: Sequence[Mapping],
    *,
    layers: Sequence[int],
    gate: LayerValidityGate = LOCALIZATION_VALIDITY_GATE,
    legacy: LegacyValidityGate = LEGACY_VALIDITY_GATE,
) -> dict[int, dict]:
    """One eligibility verdict per layer, keyed by physical layer."""
    return {
        int(layer): evaluate_layer_validity(rows, layer=layer, gate=gate, legacy=legacy)
        for layer in layers
    }


def eligible_layers(results: Mapping[int, Mapping]) -> list[int]:
    """The layers that earned a causal claim, shallow to deep."""
    return sorted(int(layer) for layer, result in results.items() if result["eligible"])


def assert_causally_eligible(layer: int, results: Mapping[int, Mapping]) -> None:
    """Refuse to run an intervention stage at a layer that failed Stage B.

    Raises:
        LayerNotEligibleError: If ``layer`` was not evaluated, or was evaluated
            and failed. The message names the clauses that failed.
    """
    result = results.get(int(layer))
    if result is None:
        raise LayerNotEligibleError(
            f"layer {layer} has no lens-validity result; a causal claim cannot "
            "rest on a layer whose readout was never tested"
        )
    if not result["eligible"]:
        raise LayerNotEligibleError(
            f"layer {layer} is {LAYER_INELIGIBLE} under {result['gate_digest']}: "
            f"failed {result['failed_checks']}. Its diagnostic results are kept; "
            "its intervention stage is skipped. A layer that could not be shown "
            "to read anything out is a layer at which an intervention's effect "
            "has no interpretation."
        )


# ------------------------------------------------------------ recalibration


#: A bounded text-only recalibration, if and only if the frozen v2 artifact
#: turns out to be genuinely underpowered. Predetermined here so that choosing
#: it later cannot also mean choosing its size, its layers, or its split.
RECALIBRATION_PLAN = {
    "protocol": "text_jlens_early_layer_recalibration_v3_bounded",
    "n_fitting_prompts": 256,
    "n_heldout_prompts": N_VALIDATION_PROMPTS,
    "layers": list(LOCALIZATION_LAYERS),
    "modality": "text-only",
    "multimodal_examples_used": False,
    "cross_modal_alignment": False,
    "modality_specific_lens": False,
    "calibration_uses_cat_or_toilet_targets": False,
    "heldout_independent_of_fitting": True,
    "frozen_before_multimodal_evaluation": True,
    "runner": "notebooks/gemma_4_e4b_text_jlens_recalibration_colab.ipynb",
    "new_artifact_dir": (
        "runs/text_jlens_early_layer_recalibration_v3/artifacts/lens.validated.pt"
    ),
    "never_overwrite": (
        "runs/text_jlens_early_layer_recalibration_v2/artifacts/lens.validated.pt"
    ),
}


class RecalibrationRefused(RuntimeError):
    """A recalibration was requested that would destroy or bias evidence."""


def check_recalibration_target(
    new_lens_path: str, *, plan: Mapping = RECALIBRATION_PLAN
) -> dict:
    """Refuse a recalibration that would overwrite v2 or reuse its path.

    The v2 artifact is the evidence base for the completed robustness run. A
    recalibration that wrote over it would make that run unreproducible, so the
    check is a refusal rather than a warning.

    Raises:
        RecalibrationRefused: If ``new_lens_path`` is the v2 artifact.
    """
    protected = str(plan["never_overwrite"]).replace("\\", "/")
    candidate = str(new_lens_path).replace("\\", "/")
    if candidate.endswith(protected) or protected.endswith(candidate):
        raise RecalibrationRefused(
            f"{new_lens_path} is the frozen v2 artifact. It is the evidence base "
            "for the completed robustness run and is never overwritten; a "
            "recalibration publishes a new path and a new checksum."
        )
    return {
        "new_lens_path": str(new_lens_path),
        "protected_path": protected,
        "overwrites_v2": False,
        "plan": dict(plan),
    }


def format_recalibration_plan(plan: Mapping = RECALIBRATION_PLAN) -> str:
    """The block printed when a recalibration is contemplated. Nothing runs."""
    return "\n".join(
        [
            "BOUNDED TEXT-ONLY RECALIBRATION — plan only, nothing is fitted here",
            f"  protocol            {plan['protocol']}",
            f"  fitting prompts     {plan['n_fitting_prompts']} (text only)",
            f"  held-out prompts    {plan['n_heldout_prompts']} (independent of fitting)",
            f"  layers              {plan['layers']}",
            "  no multimodal examples, no cross-modal alignment, no modality-specific",
            "  lens, and no cat/toilet target example takes part in the calibration.",
            f"  new artifact        {plan['new_artifact_dir']}",
            f"  never overwritten   {plan['never_overwrite']}",
            "",
            "  This notebook does not fit a lens. Recalibration is a separate,",
            f"  explicitly chosen run of {plan['runner']}, frozen and checksummed",
            "  before any multimodal evaluation may point at it.",
        ]
    )


__all__ = [
    "CONTROL_SEED",
    "CONTROL_VARIANTS",
    "CRITERION_RANK_CONVENTION",
    "DIAGNOSTIC_VARIANTS",
    "LAYER_ELIGIBLE",
    "LAYER_INELIGIBLE",
    "LEGACY_VALIDITY_GATE",
    "LOCALIZATION_VALIDITY_GATE",
    "NOISE_CONTROLS",
    "N_STABILITY_FOLDS",
    "N_VALIDATION_PROMPTS",
    "RANK_CONVENTIONS",
    "READOUT_VARIANTS",
    "RECALIBRATION_PLAN",
    "VALIDATION_PROMPT_SEED",
    "VALIDITY_PROTOCOL",
    "LayerNotEligibleError",
    "LayerValidityGate",
    "LegacyValidityGate",
    "RecalibrationRefused",
    "assert_causally_eligible",
    "check_recalibration_target",
    "eligible_layers",
    "evaluate_all_layers",
    "evaluate_layer_validity",
    "evaluate_legacy_gate",
    "fold_mrr",
    "fold_of",
    "format_recalibration_plan",
    "gate_text",
    "summarize_layer",
    "summarize_variant",
    "tie_aware_row",
]
