# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Source-derived concept directions and the causal-transfer test.

A concept direction is estimated **entirely from one modality's training
examples**: the mean nonnegative J-code of positive source-training samples
minus the mean code of matched negatives, rectified, optionally truncated to
its strongest coordinates, and mapped back through the frozen dictionary
(``v = V delta``). No target-modality activation, and no test example of any
modality, is used to build it.

The test then applies that direction at the final prompt token of a *held-out
target-modality* example: subtracted from a positive (the concept should get
harder to read out) and added to a matched negative (it should get easier).
The controls — zero coefficient, norm-matched random, an unrelated concept's
direction, and the raw residual positive-minus-negative difference — are run
through exactly the same code path, so a "transfer" that any of them also
produces is visible as non-specific.

This is directional addition and subtraction on a residual stream. It is not
erasure, not projection ablation, and nothing here should be described as such.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from jlens.interventions import isotropic_random_direction, residual_intervention
from jlens.mmpilot.backend import BuiltInputs, PilotBackend
from jlens.mmpilot.capability import prediction_and_margin, score_candidate_sequences
from jlens.mmpilot.jspace import (
    CONVENTIONS,
    reconstruct_direction,
    tensor_checksum,
)
from jlens.mmpilot.store import safe_key

#: Coefficient sweep. Alpha multiplies the direction *after* it has been
#: scaled to the mean norm of the target modality's clean activations, so
#: alpha=1.0 is an edit of the same magnitude as the activation itself.
DEFAULT_ALPHAS = (0.0, 0.25, 0.5, 1.0)

#: Control conditions, all run through the same intervention path.
CONTROL_KINDS = (
    "source_concept",
    "zero",
    "random_norm_matched",
    "unrelated_concept",
    "raw_residual_difference",
)


def mean_code(codes: Sequence[Mapping[int, float]]) -> dict[int, float]:
    """Coefficient-wise mean over sparse codes (absent atoms count as zero)."""
    if not codes:
        return {}
    totals: dict[int, float] = {}
    for code in codes:
        for atom, coefficient in code.items():
            totals[int(atom)] = totals.get(int(atom), 0.0) + float(coefficient)
    return {atom: total / len(codes) for atom, total in sorted(totals.items())}


def estimate_concept_direction(
    positive_codes: Sequence[Mapping[int, float]],
    negative_codes: Sequence[Mapping[int, float]],
    dictionary,
    *,
    concept: str,
    source_modality: str,
    layer: int,
    positive_ids: Sequence[str],
    negative_ids: Sequence[str],
    lens_checksum: str,
    top_k: int | None = None,
) -> dict:
    """``delta = ReLU(mean_positive - mean_negative)``; ``v = V delta``.

    Args:
        top_k: Keep only the ``top_k`` largest positive coordinates of
            ``delta``. ``None`` keeps all of them.

    Returns a record carrying the sparse coefficients, the support, the
    reconstructed unit direction, its pre-normalization norm, and checksums.

    Raises:
        ValueError: If the rectified difference is empty — that means the
            positives and negatives are not separable in J-space at all, and a
            zero direction must not be silently used.
    """
    positive_mean = mean_code(positive_codes)
    negative_mean = mean_code(negative_codes)
    delta = {
        atom: positive_mean.get(atom, 0.0) - negative_mean.get(atom, 0.0)
        for atom in sorted(set(positive_mean) | set(negative_mean))
    }
    delta = {atom: value for atom, value in delta.items() if value > 0}
    if not delta:
        raise ValueError(
            f"ReLU(mean_positive - mean_negative) is empty for concept {concept!r} "
            f"in {source_modality}; no direction can be estimated"
        )
    if top_k is not None:
        kept = sorted(delta.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
        delta = dict(sorted(kept))

    vector = reconstruct_direction(delta, dictionary)
    raw_norm = float(vector.norm())
    if raw_norm == 0.0:
        raise ValueError(f"reconstructed direction for {concept!r} has zero norm")
    unit = (vector / raw_norm).cpu()
    return {
        "kind": "source_concept",
        "concept": concept,
        "source_modality": source_modality,
        "layer": layer,
        "source_positive_ids": list(positive_ids),
        "source_negative_ids": list(negative_ids),
        "support": [int(atom) for atom in delta],
        "coefficients": [float(value) for value in delta.values()],
        "top_k": top_k,
        "reconstructed_norm": raw_norm,
        "unit_direction": [float(x) for x in unit.tolist()],
        "direction_checksum": tensor_checksum(unit),
        "lens_checksum": lens_checksum,
        "normalization": CONVENTIONS["direction_normalization"],
        "uses_target_modality_data": False,
    }


def raw_residual_direction(
    positive_activations: Sequence[Sequence[float]],
    negative_activations: Sequence[Sequence[float]],
    *,
    concept: str,
    source_modality: str,
    layer: int,
) -> dict:
    """Control: mean positive minus mean negative in *raw* residual space.

    Same source examples, same normalization, no lens. If this matches the
    J-space direction's effect, the J-space decomposition added nothing.
    """
    positives = torch.tensor(positive_activations, dtype=torch.float32)
    negatives = torch.tensor(negative_activations, dtype=torch.float32)
    vector = positives.mean(dim=0) - negatives.mean(dim=0)
    raw_norm = float(vector.norm())
    if raw_norm == 0.0:
        raise ValueError("raw residual difference direction has zero norm")
    unit = vector / raw_norm
    return {
        "kind": "raw_residual_difference",
        "concept": concept,
        "source_modality": source_modality,
        "layer": layer,
        "reconstructed_norm": raw_norm,
        "unit_direction": [float(x) for x in unit.tolist()],
        "direction_checksum": tensor_checksum(unit),
        "normalization": CONVENTIONS["direction_normalization"],
        "uses_target_modality_data": False,
    }


def unit_direction_tensor(record: Mapping) -> torch.Tensor:
    return torch.tensor(record["unit_direction"], dtype=torch.float32)


def condition_key(
    *,
    concept: str,
    source_modality: str,
    target_modality: str,
    layer: int,
    control_kind: str,
    alpha: float,
    sample_id: str,
) -> str:
    """Stable per-condition unit key (also the resume key)."""
    return safe_key(
        concept,
        f"{source_modality}-to-{target_modality}",
        f"L{layer}",
        control_kind,
        f"a{alpha:g}",
        sample_id,
    )


def run_condition(
    backend: PilotBackend,
    inputs: BuiltInputs,
    *,
    layer: int,
    unit_direction: torch.Tensor,
    alpha: float,
    reference_norm: float,
    sign: int,
    candidate_ids: Mapping[str, Sequence[int]],
    target_concept: str,
    clean_scores: Mapping[str, Mapping],
) -> dict:
    """One intervened forward pass and its comparison with the clean run.

    Args:
        sign: ``-1`` for a positive target example (subtract the direction),
            ``+1`` for a matched negative (add it).
        reference_norm: Mean L2 norm of the target modality's clean
            activations at this layer; the unit direction is scaled by it so
            ``alpha`` is dimensionless.
        clean_scores: The unedited candidate scores for the same input.

    ``signed_target_effect`` folds the sign in: it is positive when the edit
    moved the target concept in the intended direction, whichever half of the
    positive/negative pair this is.
    """
    delta = unit_direction.to(torch.float32) * reference_norm
    with residual_intervention(
        backend.blocks,
        layer,
        position=inputs.final_prompt_position,
        delta=delta,
        multiplier=float(sign) * float(alpha),
    ) as stats:
        scores = score_candidate_sequences(backend, inputs, candidate_ids)
    verdict = prediction_and_margin(scores, target_concept)
    clean_verdict = prediction_and_margin(clean_scores, target_concept)
    target_change = verdict["target_score"] - clean_verdict["target_score"]
    unrelated = {
        candidate: scores[candidate]["sum_logprob"] - clean_scores[candidate]["sum_logprob"]
        for candidate in scores
        if candidate != target_concept
    }
    norm_before = float(stats["activation_norm"] or 0.0)
    norm_after = float(stats["edited_norm"] or 0.0)
    return {
        "alpha": float(alpha),
        "sign": int(sign),
        "layer": layer,
        "delta_norm": float(stats["delta_norm"] or 0.0) * abs(alpha),
        "reference_norm": float(reference_norm),
        "activation_norm_before": norm_before,
        "activation_norm_after": norm_after,
        "activation_norm_ratio": norm_after / norm_before if norm_before else 0.0,
        "resolved_position": stats["resolved_position"],
        "clean_target_score": clean_verdict["target_score"],
        "target_score": verdict["target_score"],
        "target_score_change": target_change,
        "signed_target_effect": float(sign) * target_change,
        "clean_target_margin": clean_verdict["target_margin"],
        "target_margin": verdict["target_margin"],
        "target_margin_change": verdict["target_margin"] - clean_verdict["target_margin"],
        "signed_margin_effect": float(sign)
        * (verdict["target_margin"] - clean_verdict["target_margin"]),
        "unrelated_candidate_changes": unrelated,
        "max_abs_unrelated_change": max((abs(v) for v in unrelated.values()), default=0.0),
        "clean_prediction": clean_verdict["prediction"],
        "prediction": verdict["prediction"],
        "prediction_changed": verdict["prediction"] != clean_verdict["prediction"],
        "candidate_scores": scores,
    }


def random_control_direction(
    d_model: int, *, seed: int, matched_to: str
) -> dict:
    """Norm-matched isotropic random control (unit norm; scaled like the rest)."""
    delta, info = isotropic_random_direction(
        d_model, match_norm=1.0, seed=seed, matched_target_condition_id=matched_to
    )
    return {
        "kind": "random_norm_matched",
        "seed": int(seed),
        "matched_to": matched_to,
        "reconstructed_norm": 1.0,
        "unit_direction": [float(x) for x in delta.tolist()],
        "direction_checksum": tensor_checksum(delta),
        "delta_info": info.__dict__ if hasattr(info, "__dict__") else {},
        "normalization": CONVENTIONS["direction_normalization"],
        "uses_target_modality_data": False,
    }


def summarize_interventions(records: Sequence[Mapping]) -> dict:
    """Aggregate intervention units into the numbers the rubric reads.

    Groups by (concept, source->target, layer, control kind, alpha) and
    reports the mean signed effect, how often the sign was as intended, the
    mean absolute unrelated-candidate movement, and the activation-norm ratio.
    Off-diagonal (cross-modality) conditions are flagged, since those are the
    only ones that answer the research question.
    """
    cells: dict[tuple, list[Mapping]] = {}
    for record in records:
        key = (
            record["concept"],
            record["source_modality"],
            record["target_modality"],
            record["layer"],
            record["control_kind"],
            float(record["alpha"]),
        )
        cells.setdefault(key, []).append(record)

    rows = []
    for key, group in sorted(cells.items(), key=lambda item: [str(x) for x in item[0]]):
        concept, source, target, layer, control, alpha = key
        effects = [float(r["signed_target_effect"]) for r in group]
        margins = [float(r["signed_margin_effect"]) for r in group]
        rows.append(
            {
                "concept": concept,
                "source_modality": source,
                "target_modality": target,
                "pair": f"{source}->{target}",
                "off_diagonal": source != target,
                "layer": layer,
                "control_kind": control,
                "alpha": alpha,
                "n": len(group),
                "mean_signed_target_effect": sum(effects) / len(effects),
                "mean_signed_margin_effect": sum(margins) / len(margins),
                "fraction_expected_sign": sum(1 for e in effects if e > 0) / len(effects),
                "mean_abs_unrelated_change": sum(
                    float(r["max_abs_unrelated_change"]) for r in group
                )
                / len(group),
                "mean_activation_norm_ratio": sum(
                    float(r["activation_norm_ratio"]) for r in group
                )
                / len(group),
                "n_prediction_changes": sum(1 for r in group if r["prediction_changed"]),
            }
        )
    return {"n_records": len(records), "rows": rows, "control_kinds": list(CONTROL_KINDS)}
