# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Paper-first replication and source-loading localization.

This module is deliberately small.  It does not implement another intervention:
all causal edits still go through :mod:`jlens.mmpilot.coordinate_swap`.  Its job
is to keep the order of evidence honest:

1. reproduce the paper's text-only task with the exact alpha=1 coordinate swap;
2. measure whether the clean residual actually loads on the source J-lens row;
3. choose a contiguous layer band and a prompt-position rule from that clean
   loading measurement only;
4. freeze that choice before a fresh multimodal population is opened.

The functions here are pure apart from :func:`capture_source_loading`, which
performs one no-gradient model forward pass.  They are therefore suitable for
unit tests and for a resumable notebook orchestrator.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import torch

from jlens.hooks import ActivationRecorder
from jlens.mmpilot.coordinate_swap import coordinate_swap_band, read_coordinates
from jlens.mmpilot.store import payload_checksum

PROTOCOL_VERSION = "mmpilot.paper_first_workspace_replication.v2"
TEXT_OUTPUT_ENDPOINT_VERSION = "mmpilot.unrestricted_greedy_complete_answer.v1"
TEXT_MAX_NEW_TOKENS = 2
LOADING_VERSION = "mmpilot.clean_source_loading.v1"
LOCALIZATION_VERSION = "mmpilot.loading_only_localization.v1"
CONFIRMATION_VERSION = "mmpilot.fresh_multimodal_confirmation.v1"


class WorkspaceReplicationRefused(RuntimeError):
    """The requested claim is not licensed by the recorded evidence."""


@dataclass(frozen=True)
class TextReplicationTask:
    """One predeclared text-only use case from the paper's task family."""

    task_id: str
    family: str
    prompt: str
    source: str
    target: str
    clean_answer: str
    swapped_answer: str
    implicit_intermediate: bool

    def to_dict(self) -> dict:
        return asdict(self)


def anthropic_text_tasks() -> tuple[TextReplicationTask, ...]:
    """Frozen text tasks, defined before any Gemma result is read.

    ``spider_to_ant_legs`` is the downstream-recomputation task.  The country
    rows are the paper's flexible-function family: one concept replacement is
    read through several downstream questions.  They are useful replication
    targets but do not by themselves establish recomputation.
    """

    return (
        TextReplicationTask(
            "spider_to_ant_legs",
            "implicit_two_hop",
            "The number of legs on the animal that spins webs is",
            "spider",
            "ant",
            "8",
            "6",
            True,
        ),
        TextReplicationTask(
            "france_to_china_capital",
            "flexible_function",
            "The capital of France is",
            "France",
            "China",
            "Paris",
            "Beijing",
            False,
        ),
        TextReplicationTask(
            "france_to_china_language",
            "flexible_function",
            "The main language of France is",
            "France",
            "China",
            "French",
            "Chinese",
            False,
        ),
        TextReplicationTask(
            "france_to_china_continent",
            "flexible_function",
            "France is located in",
            "France",
            "China",
            "Europe",
            "Asia",
            False,
        ),
        TextReplicationTask(
            "china_to_france_capital",
            "flexible_function",
            "The capital of China is",
            "China",
            "France",
            "Beijing",
            "Paris",
            False,
        ),
        TextReplicationTask(
            "china_to_france_language",
            "flexible_function",
            "The main language of China is",
            "China",
            "France",
            "Chinese",
            "French",
            False,
        ),
        TextReplicationTask(
            "china_to_france_continent",
            "flexible_function",
            "China is located in",
            "China",
            "France",
            "Asia",
            "Europe",
            False,
        ),
    )


def text_task_digest(tasks: Sequence[TextReplicationTask] | None = None) -> str:
    selected = anthropic_text_tasks() if tasks is None else tuple(tasks)
    return payload_checksum(
        {"version": PROTOCOL_VERSION, "tasks": [task.to_dict() for task in selected]}
    )


@torch.no_grad()
def unrestricted_greedy_completion(
    backend,
    inputs,
    *,
    answer: str,
    max_new_tokens: int = TEXT_MAX_NEW_TOKENS,
) -> dict:
    """Generate a complete answer without candidates or teacher forcing.

    Anthropic's tokenizer represents the paper's digit answers as one next
    token. Gemma 4 represents the same continuation as two tokens (a whitespace
    token followed by the digit), so a one-row global-argmax endpoint is not
    defined for this model. This endpoint preserves the literal prompt and
    answer while greedily generating the complete token sequence. Every token
    is selected from the full vocabulary; no answer token is appended.
    """

    from jlens.mmpilot.capability import _extend_tensors
    from jlens.mmpilot.full_vocabulary import (
        greedy_matches,
        normalize_generated_text,
        token_decoder,
    )

    budget = int(max_new_tokens)
    if budget < 1:
        raise WorkspaceReplicationRefused("max_new_tokens must be positive")
    decoder = token_decoder(backend)
    tensors = dict(inputs.tensors)
    input_ids = tensors.get("input_ids")
    if not torch.is_tensor(input_ids) or input_ids.ndim != 2:
        raise WorkspaceReplicationRefused(
            "unrestricted complete-answer generation requires rank-two input_ids"
        )
    if int(input_ids.shape[1]) != int(inputs.prompt_len):
        raise WorkspaceReplicationRefused(
            "generation must begin at the untouched prompt boundary"
        )

    generated: list[int] = []
    for _ in range(budget):
        step_tensors = (
            tensors
            if not generated
            else _extend_tensors(tensors, int(inputs.prompt_len), generated)
        )
        logits = backend.forward_logits(step_tensors)
        step = logits[0, -1].float()
        if not bool(torch.isfinite(step).all()):
            raise WorkspaceReplicationRefused(
                "non-finite logits during unrestricted generation"
            )
        generated.append(int(step.argmax()))

    text = "".join(decoder(token_id) for token_id in generated)
    return {
        "endpoint_version": TEXT_OUTPUT_ENDPOINT_VERSION,
        "endpoint": "unrestricted_greedy_complete_answer",
        "teacher_forcing_used": False,
        "candidate_list_supplied": False,
        "temperature": 0.0,
        "do_sample": False,
        "max_new_tokens": budget,
        "n_forward_passes": budget,
        "generated_token_ids": generated,
        "generated_text": text,
        "normalized_generated_text": normalize_generated_text(text),
        "answer": str(answer),
        "normalized_answer": normalize_generated_text(str(answer)),
        "answer_match": bool(greedy_matches(text, str(answer))),
    }


@torch.no_grad()
def unrestricted_greedy_swap_trial(
    backend,
    inputs,
    *,
    bases: Mapping,
    alpha: float,
    answer: str,
    max_new_tokens: int = TEXT_MAX_NEW_TOKENS,
    position_rule: str = "all_prompt_positions",
) -> dict:
    """Run the paper's swap and freely generate the complete answer.

    Hooks remain active for every greedy decoding step but patch only the
    pre-existing prompt positions. Thus the generated answer is observed, not
    supplied, while the intervention is identical on each recomputed forward.
    """

    with coordinate_swap_band(
        backend.blocks,
        bases,
        alpha=float(alpha),
        prompt_len=int(inputs.prompt_len),
        position_rule=str(position_rule),
        evidence_span=getattr(inputs, "modality_token_range", None),
        record_coordinates=False,
    ) as stats:
        generated = unrestricted_greedy_completion(
            backend,
            inputs,
            answer=str(answer),
            max_new_tokens=int(max_new_tokens),
        )

    positions = {
        str(layer): list(stats[layer].get("positions") or [])
        for layer in sorted(stats)
    }
    expected = list(range(int(inputs.prompt_len)))
    return {
        **generated,
        "alpha": float(alpha),
        "alpha_role": "exact_exchange" if float(alpha) == 1.0 else "nonexact",
        "position_rule": str(position_rule),
        "layers_patched": sorted(int(layer) for layer in stats),
        "positions_patched": positions,
        "all_prompt_positions_patched": all(
            layer_positions == expected for layer_positions in positions.values()
        )
        if str(position_rule) == "all_prompt_positions"
        else None,
        "hook_forward_passes_by_layer": {
            str(layer): int(stats[layer].get("n_forward_passes") or 0)
            for layer in sorted(stats)
        },
    }


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    left = a.detach().to(torch.float64).flatten()
    right = b.detach().to(torch.float64).flatten()
    denominator = float(left.norm() * right.norm())
    if denominator == 0.0:
        return 0.0
    value = float(left.dot(right)) / denominator
    if not math.isfinite(value):
        raise WorkspaceReplicationRefused("a clean source-loading cosine is non-finite")
    return value


@torch.no_grad()
def capture_source_loading(
    backend,
    inputs,
    *,
    vectors_by_layer: Mapping[int, Mapping[str, torch.Tensor]],
    source: str,
    target: str,
    unrelated: Sequence[str] = (),
    sample_id: str,
    modality: str,
) -> list[dict]:
    """Measure clean J-lens loading at every original prompt position.

    This is an observation-only forward pass.  No hook returns a replacement,
    no candidate token is appended, and no causal outcome is available to the
    caller while localization is chosen.
    """

    layers = tuple(sorted(int(layer) for layer in vectors_by_layer))
    if not layers:
        raise WorkspaceReplicationRefused("source loading needs at least one layer")
    required = {source, target, *map(str, unrelated)}
    missing = {
        layer: sorted(required - set(vectors_by_layer[layer]))
        for layer in layers
        if required - set(vectors_by_layer[layer])
    }
    if missing:
        raise WorkspaceReplicationRefused(f"missing lens vectors: {missing}")

    with ActivationRecorder(backend.blocks, at=layers) as recorder:
        backend.forward_logits(inputs.tensors)
    evidence_span = inputs.modality_token_range
    evidence_positions = (
        set(range(int(evidence_span[0]), int(evidence_span[1])))
        if evidence_span is not None
        else set()
    )
    rows: list[dict] = []
    for layer in layers:
        activation = recorder.activations[layer].detach()[0, : inputs.prompt_len]
        source_vector = vectors_by_layer[layer][source]
        target_vector = vectors_by_layer[layer][target]
        V = torch.stack((source_vector, target_vector), dim=1)
        coordinates = read_coordinates(activation, V)
        for position in range(inputs.prompt_len):
            h = activation[position]
            unrelated_cosines = {
                name: _cosine(h, vectors_by_layer[layer][name])
                for name in unrelated
            }
            source_cosine = _cosine(h, source_vector)
            target_cosine = _cosine(h, target_vector)
            control = max([target_cosine, *unrelated_cosines.values()])
            rows.append(
                {
                    "version": LOADING_VERSION,
                    "sample_id": str(sample_id),
                    "modality": str(modality),
                    "layer": int(layer),
                    "position": int(position),
                    "position_class": (
                        "final_prompt_token"
                        if position == inputs.final_prompt_position
                        else "evidence"
                        if position in evidence_positions
                        else "non_evidence"
                    ),
                    "source": str(source),
                    "target": str(target),
                    "source_cosine": source_cosine,
                    "target_cosine": target_cosine,
                    "unrelated_cosines": unrelated_cosines,
                    "source_advantage": source_cosine - control,
                    "source_coordinate": float(coordinates[position, 0]),
                    "target_coordinate": float(coordinates[position, 1]),
                    "prompt_len": int(inputs.prompt_len),
                    "evidence_span": list(evidence_span) if evidence_span else None,
                    "causal_result_consulted": False,
                }
            )
    return rows


def _median(values: Sequence[float]) -> float:
    if not values:
        raise WorkspaceReplicationRefused("cannot summarize an empty loading cell")
    return float(statistics.median(map(float, values)))


def summarize_loading(rows: Sequence[Mapping]) -> dict:
    """Aggregate loading without discarding the per-position measurements."""

    if not rows:
        raise WorkspaceReplicationRefused("no clean source-loading rows were supplied")
    grouped: dict[tuple[str, int, str], list[Mapping]] = {}
    for row in rows:
        grouped.setdefault(
            (str(row["modality"]), int(row["layer"]), str(row["position_class"])),
            [],
        ).append(row)
    cells = []
    for (modality, layer, position_class), cell in sorted(grouped.items()):
        cells.append(
            {
                "modality": modality,
                "layer": layer,
                "position_class": position_class,
                "n": len(cell),
                "n_samples": len({str(row["sample_id"]) for row in cell}),
                "median_source_cosine": _median(
                    [float(row["source_cosine"]) for row in cell]
                ),
                "median_target_cosine": _median(
                    [float(row["target_cosine"]) for row in cell]
                ),
                "median_source_advantage": _median(
                    [float(row["source_advantage"]) for row in cell]
                ),
            }
        )
    payload = {
        "version": LOADING_VERSION,
        "n_rows": len(rows),
        "n_samples": len({str(row["sample_id"]) for row in rows}),
        "cells": cells,
        "causal_result_consulted": False,
    }
    return {**payload, "report_checksum": payload_checksum(payload)}


def _runs(layers: Sequence[int]) -> list[tuple[int, ...]]:
    runs: list[list[int]] = []
    for layer in sorted(set(map(int, layers))):
        if not runs or layer != runs[-1][-1] + 1:
            runs.append([layer])
        else:
            runs[-1].append(layer)
    return [tuple(run) for run in runs]


def freeze_loading_localization(
    rows: Sequence[Mapping],
    *,
    required_modalities: Sequence[str],
    candidate_layers: Sequence[int],
    min_source_advantage: float = 0.0,
    evidence_position_margin: float = 0.0,
) -> dict:
    """Choose a band and position rule using clean loading and nothing else.

    A layer is admissible only when its median source advantage is above the
    frozen margin in *every* required modality.  The longest contiguous run is
    used; ties choose the deeper run deterministically.  For image/audio, an
    evidence-only rule is chosen only when evidence positions beat non-evidence
    positions by the frozen margin in every modality that has an evidence span.
    Otherwise the literal paper rule (all prompt positions) remains in force.
    """

    required = tuple(map(str, required_modalities))
    layers = tuple(sorted(set(map(int, candidate_layers))))
    if not required or not layers:
        raise WorkspaceReplicationRefused("localization needs modalities and layers")
    by_cell: dict[tuple[str, int], list[Mapping]] = {}
    for row in rows:
        if int(row["layer"]) in layers and str(row["modality"]) in required:
            by_cell.setdefault((str(row["modality"]), int(row["layer"])), []).append(row)

    layer_evidence = []
    eligible = []
    for layer in layers:
        medians = {}
        complete = True
        for modality in required:
            cell = by_cell.get((modality, layer), [])
            if not cell:
                complete = False
                medians[modality] = None
            else:
                medians[modality] = _median(
                    [float(row["source_advantage"]) for row in cell]
                )
        passed = complete and all(
            float(medians[modality]) > float(min_source_advantage)
            for modality in required
        )
        if passed:
            eligible.append(layer)
        layer_evidence.append(
            {"layer": layer, "median_source_advantage": medians, "passed": passed}
        )

    runs = _runs(eligible)
    selected = max(runs, key=lambda run: (len(run), run[-1]), default=())

    position_evidence = []
    position_rule_by_modality = {modality: "all_prompt_positions" for modality in required}
    for modality in required:
        evidence_values = [
            float(row["source_advantage"])
            for row in rows
            if str(row["modality"]) == modality
            and int(row["layer"]) in selected
            and str(row["position_class"]) == "evidence"
        ]
        non_evidence_values = [
            float(row["source_advantage"])
            for row in rows
            if str(row["modality"]) == modality
            and int(row["layer"]) in selected
            and str(row["position_class"]) in {"non_evidence", "final_prompt_token"}
        ]
        if not evidence_values:
            continue
        evidence_median = _median(evidence_values)
        non_evidence_median = _median(non_evidence_values)
        passed = evidence_median >= non_evidence_median + float(evidence_position_margin)
        if selected and passed:
            position_rule_by_modality[modality] = "evidence_span_only"
        position_evidence.append(
            {
                "modality": modality,
                "median_evidence_advantage": evidence_median,
                "median_non_evidence_advantage": non_evidence_median,
                "passed": passed,
            }
        )
    unique_rules = sorted(set(position_rule_by_modality.values()))
    position_rule = unique_rules[0] if len(unique_rules) == 1 else "modality_specific"
    verdict = "LOADING_LOCALIZATION_GO" if selected else "LOADING_LOCALIZATION_NO_GO"
    payload = {
        "version": LOCALIZATION_VERSION,
        "verdict": verdict,
        "candidate_layers": list(layers),
        "required_modalities": list(required),
        "min_source_advantage": float(min_source_advantage),
        "evidence_position_margin": float(evidence_position_margin),
        "layer_evidence": layer_evidence,
        "eligible_layers": eligible,
        "contiguous_runs": [list(run) for run in runs],
        "selected_band": list(selected),
        "position_rule": position_rule,
        "position_rule_by_modality": position_rule_by_modality,
        "position_evidence": position_evidence,
        "causal_result_consulted": False,
        "selection_depended_on_causal_outcome": False,
    }
    return {**payload, "design_digest": payload_checksum(payload)}


def select_pair_from_loading(
    rows: Sequence[Mapping],
    *,
    candidate_pairs: Sequence[Sequence[str]],
    required_modalities: Sequence[str],
) -> dict:
    """Select a concept pair from clean source loading, never causal response.

    The score is the weakest modality's median source advantage, pooled across
    layers and positions.  This rewards a source representation that is
    already visible in every channel instead of a pair that happened to react
    strongly to an intervention.
    """

    pairs = [tuple(map(str, pair)) for pair in candidate_pairs]
    if any(len(pair) != 2 or pair[0] == pair[1] for pair in pairs):
        raise WorkspaceReplicationRefused(f"invalid candidate pairs: {pairs}")
    modalities = tuple(map(str, required_modalities))
    ranking = []
    for source, target in pairs:
        per_modality = {}
        for modality in modalities:
            values = [
                float(row["source_advantage"])
                for row in rows
                if str(row.get("source")) == source
                and str(row.get("target")) == target
                and str(row.get("modality")) == modality
            ]
            per_modality[modality] = _median(values) if values else None
        complete = all(value is not None for value in per_modality.values())
        score = (
            min(float(value) for value in per_modality.values())
            if complete
            else float("-inf")
        )
        ranking.append(
            {
                "source": source,
                "target": target,
                "per_modality_median_source_advantage": per_modality,
                "weakest_modality_score": score,
                "complete": complete,
            }
        )
    ranking.sort(
        key=lambda row: (
            -float(row["weakest_modality_score"]),
            str(row["source"]),
            str(row["target"]),
        )
    )
    if not ranking or not ranking[0]["complete"]:
        raise WorkspaceReplicationRefused(
            "no candidate pair has loading measurements in every required modality"
        )
    payload = {
        "version": LOCALIZATION_VERSION,
        "selection_rule": "maximize the weakest modality's median clean source advantage",
        "required_modalities": list(modalities),
        "ranking": ranking,
        "selected_pair": [ranking[0]["source"], ranking[0]["target"]],
        "causal_result_consulted": False,
        "selection_depended_on_causal_outcome": False,
    }
    return {**payload, "selection_digest": payload_checksum(payload)}


def text_replication_verdict(rows: Sequence[Mapping]) -> dict:
    """Gate later stages on the paper task, not a multimodal hope."""

    tasks = {task.task_id: task for task in anthropic_text_tasks()}
    by_task = {str(row["task_id"]): dict(row) for row in rows}
    missing = sorted(set(tasks) - set(by_task))
    clean = not missing and all(bool(by_task[name].get("clean_correct")) for name in tasks)
    implicit = bool(
        by_task.get("spider_to_ant_legs", {}).get(
            "exact_alpha1_swapped_answer_generated",
            by_task.get("spider_to_ant_legs", {}).get("exact_alpha1_target_top1"),
        )
    )
    flexible_rows = [
        by_task[name] for name, task in tasks.items() if task.family == "flexible_function"
    ]
    flexible_rate = (
        sum(
            bool(
                row.get(
                    "exact_alpha1_swapped_answer_generated",
                    row.get("exact_alpha1_target_top1"),
                )
            )
            for row in flexible_rows
        )
        / len(flexible_rows)
        if flexible_rows
        else 0.0
    )
    controls = not missing and all(
        not bool(
            row.get(
                "random_swapped_answer_generated", row.get("random_target_top1")
            )
        )
        and not bool(
            row.get(
                "unrelated_swapped_answer_generated",
                row.get("unrelated_target_top1"),
            )
        )
        for row in by_task.values()
    )
    passed = clean and implicit and flexible_rate >= 0.5 and controls
    payload = {
        "version": PROTOCOL_VERSION,
        "verdict": "TEXT_PAPER_REPLICATION_GO" if passed else "TEXT_PAPER_REPLICATION_NO_GO",
        "all_clean_answers_correct": clean,
        "output_endpoint": "unrestricted_greedy_complete_answer",
        "implicit_two_hop_swapped_answer_rate": 1.0 if implicit else 0.0,
        "flexible_function_swapped_answer_rate": flexible_rate,
        "matched_controls_pass": controls,
        "missing_tasks": missing,
        "multimodal_stage_licensed": passed,
    }
    return {**payload, "report_checksum": payload_checksum(payload)}


def freeze_confirmation_design(
    *,
    text_verdict: Mapping,
    localization: Mapping,
    pair: Sequence[str],
    alpha: float = 1.0,
    sensitivity_alpha: float | None = 0.75,
    prompt_protocol: str,
    development_population_digest: str,
) -> dict:
    """Freeze the confirmatory design before any fresh media are opened."""

    if text_verdict.get("verdict") != "TEXT_PAPER_REPLICATION_GO":
        raise WorkspaceReplicationRefused(
            "the text-only paper replication did not pass; multimodal confirmation is blocked"
        )
    if localization.get("verdict") != "LOADING_LOCALIZATION_GO":
        raise WorkspaceReplicationRefused(
            "clean source loading did not license a layer band"
        )
    names = tuple(map(str, pair))
    if len(names) != 2 or names[0] == names[1]:
        raise WorkspaceReplicationRefused(f"confirmation needs two concepts, got {names}")
    payload = {
        "version": CONFIRMATION_VERSION,
        "pair": list(names),
        "primary_alpha": float(alpha),
        "primary_alpha_role": "exact_exchange" if float(alpha) == 1.0 else "nonexact",
        "sensitivity_alpha": (
            float(sensitivity_alpha) if sensitivity_alpha is not None else None
        ),
        "sensitivity_alpha_role": "interpolation_not_primary",
        "layer_band": list(localization["selected_band"]),
        "position_rule": str(localization["position_rule"]),
        "position_rule_by_modality": dict(
            localization.get("position_rule_by_modality")
            or {"text": str(localization["position_rule"])}
        ),
        "prompt_protocol": str(prompt_protocol),
        "development_population_digest": str(development_population_digest),
        "fresh_population_required": True,
        "development_images_forbidden": True,
        "teacher_forcing_used": False,
        "candidate_list_supplied": False,
        "selection_depended_on_causal_outcome": False,
    }
    return {**payload, "design_digest": payload_checksum(payload)}


def assert_fresh_population(
    confirmation_groups: Sequence[Mapping],
    *,
    forbidden_image_ids: Sequence[str],
    forbidden_group_ids: Sequence[str] = (),
) -> dict:
    """Prove that confirmation media were absent from development and prior runs."""

    images = [str(row["image_id"]) for row in confirmation_groups]
    groups = [str(row["group_id"]) for row in confirmation_groups]
    image_overlap = sorted(set(images) & set(map(str, forbidden_image_ids)))
    group_overlap = sorted(set(groups) & set(map(str, forbidden_group_ids)))
    if image_overlap or group_overlap:
        raise WorkspaceReplicationRefused(
            f"confirmation population is not fresh: images={image_overlap}, groups={group_overlap}"
        )
    payload = {
        "version": CONFIRMATION_VERSION,
        "n_groups": len(groups),
        "n_distinct_groups": len(set(groups)),
        "n_distinct_images": len(set(images)),
        "image_overlap": image_overlap,
        "group_overlap": group_overlap,
        "fresh": True,
    }
    return {**payload, "population_digest": payload_checksum(payload)}


def paired_binary_superiority(
    treatment: Sequence[bool], control: Sequence[bool]
) -> dict:
    """Exact one-sided paired sign test for two binary causal conditions."""

    if len(treatment) != len(control) or not treatment:
        raise WorkspaceReplicationRefused(
            "paired binary superiority needs two equal, non-empty sequences"
        )
    wins = sum(bool(a) and not bool(b) for a, b in zip(treatment, control, strict=True))
    losses = sum(not bool(a) and bool(b) for a, b in zip(treatment, control, strict=True))
    discordant = wins + losses
    pvalue = (
        sum(math.comb(discordant, k) for k in range(wins, discordant + 1))
        / (2**discordant)
        if discordant
        else 1.0
    )
    return {
        "n": len(treatment),
        "treatment_rate": sum(map(bool, treatment)) / len(treatment),
        "control_rate": sum(map(bool, control)) / len(control),
        "wins": wins,
        "losses": losses,
        "ties": len(treatment) - discordant,
        "one_sided_exact_p": float(pvalue),
    }


def holm_adjust(pvalues: Mapping[str, float]) -> dict[str, float]:
    """Holm step-down family-wise correction, returned in original keys."""

    ordered = sorted((float(value), str(name)) for name, value in pvalues.items())
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for index, (value, name) in enumerate(ordered):
        running = max(running, min(1.0, value * (total - index)))
        adjusted[name] = running
    return {str(name): adjusted[str(name)] for name in pvalues}


__all__ = [
    "CONFIRMATION_VERSION",
    "LOADING_VERSION",
    "LOCALIZATION_VERSION",
    "PROTOCOL_VERSION",
    "TEXT_MAX_NEW_TOKENS",
    "TEXT_OUTPUT_ENDPOINT_VERSION",
    "TextReplicationTask",
    "WorkspaceReplicationRefused",
    "anthropic_text_tasks",
    "assert_fresh_population",
    "capture_source_loading",
    "freeze_confirmation_design",
    "freeze_loading_localization",
    "holm_adjust",
    "paired_binary_superiority",
    "select_pair_from_loading",
    "summarize_loading",
    "text_replication_verdict",
    "text_task_digest",
    "unrestricted_greedy_completion",
    "unrestricted_greedy_swap_trial",
]
