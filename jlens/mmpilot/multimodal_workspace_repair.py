# SPDX-License-Identifier: Apache-2.0
"""Gated repair study for multimodal J/R-space causal transfer.

The completed L21 trimodal run is immutable evidence.  It revealed a protocol
failure (text and image exhausted a two-token endpoint on ``"The animal"``)
and, independently, a causal null.  This module defines the prospective repair:

1. one answer-neutral assistant-prefill protocol for every modality;
2. clean capability before any causal spending;
3. layer/position/instrument selection from clean source loading only;
4. exact alpha=1 coordinate exchange against paired controls;
5. a separately frozen, media-disjoint confirmation population.

The functions here are deliberately report builders rather than model code.
Model passes remain in the notebook and are atomically stored one unit at a
time; these functions make the scientific gates deterministic and testable.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence

from jlens.mmpilot.store import payload_checksum
from jlens.mmpilot.workspace_replication import (
    WorkspaceReplicationRefused,
    freeze_loading_localization,
    holm_adjust,
    paired_binary_superiority,
    summarize_loading,
)

REPAIR_PROTOCOL_VERSION = "mmpilot.multimodal_workspace_repair.v2"
CAPABILITY_VERSION = "mmpilot.multimodal_prefill_capability.v2"
TOMOGRAPHY_VERSION = "mmpilot.multimodal_loading_tomography.v2"
DEVELOPMENT_VERSION = "mmpilot.multimodal_swap_development.v1"
CONFIRMATION_VERSION = "mmpilot.multimodal_swap_fresh_confirmation.v1"
DESIGN_VERSION = "mmpilot.multimodal_workspace_confirmation_design.v1"
POSITION_DIAGNOSTIC_VERSION = "mmpilot.multimodal_position_diagnostic.v1"
POSITION_DESIGN_VERSION = "mmpilot.multimodal_position_selected_design.v1"

POSITION_DIAGNOSTIC_STRATEGIES = {
    "all_prompt_positions": {
        "text": "all_prompt_positions",
        "image": "all_prompt_positions",
        "spoken_audio": "all_prompt_positions",
    },
    "final_prompt_token_only": {
        "text": "final_prompt_token_only",
        "image": "final_prompt_token_only",
        "spoken_audio": "final_prompt_token_only",
    },
    "modality_evidence_only": {
        # Text inputs do not expose a modality-token span.  Their final prompt
        # token is the answer-neutral location supported by the loading audit.
        "text": "final_prompt_token_only",
        "image": "evidence_span_only",
        "spoken_audio": "evidence_span_only",
    },
}


def _median(values: Sequence[float]) -> float:
    if not values:
        raise WorkspaceReplicationRefused("cannot summarize an empty cell")
    return float(statistics.median(map(float, values)))


def multimodal_capability_report(
    rows: Sequence[Mapping],
    *,
    concepts: Sequence[str],
    modalities: Sequence[str] = ("text", "image", "spoken_audio"),
    min_property_rate: float = 0.75,
) -> dict:
    """Gate concept pairs using unrestricted clean property completions.

    Identity rows are reported but do not substitute for the downstream
    property endpoint.  A concept is eligible only if its property answer rate
    reaches the frozen bar in every modality.
    """

    concepts = tuple(map(str, concepts))
    modalities = tuple(map(str, modalities))
    if not concepts or not modalities:
        raise WorkspaceReplicationRefused(
            "capability reporting requires concepts and modalities"
        )
    cells = []
    eligible = []
    for concept in concepts:
        concept_pass = True
        for prompt_kind in ("identity", "property"):
            for modality in modalities:
                cell = [
                    row
                    for row in rows
                    if str(row.get("concept")) == concept
                    and str(row.get("modality")) == modality
                    and str(row.get("prompt_kind")) == prompt_kind
                ]
                if not cell:
                    raise WorkspaceReplicationRefused(
                        f"missing capability cell {concept}/{modality}/{prompt_kind}"
                    )
                successes = sum(bool(row.get("clean_correct")) for row in cell)
                rate = successes / len(cell)
                passed = (
                    rate >= float(min_property_rate)
                    if prompt_kind == "property"
                    else None
                )
                if prompt_kind == "property" and not passed:
                    concept_pass = False
                cells.append(
                    {
                        "concept": concept,
                        "modality": modality,
                        "prompt_kind": prompt_kind,
                        "n": len(cell),
                        "successes": successes,
                        "rate": rate,
                        "property_gate_passed": passed,
                    }
                )
        if concept_pass:
            eligible.append(concept)
    payload = {
        "version": CAPABILITY_VERSION,
        "min_property_rate": float(min_property_rate),
        "modalities": list(modalities),
        "concepts": list(concepts),
        "eligible_concepts": eligible,
        "cells": cells,
        "teacher_forcing_used": False,
        "candidate_list_supplied": False,
        "causal_outcomes_opened": False,
    }
    return {**payload, "report_checksum": payload_checksum(payload)}


def _bidirectional_score(
    rows: Sequence[Mapping],
    *,
    pair: Sequence[str],
    modalities: Sequence[str],
    position_class: str | None = None,
) -> tuple[float | None, dict]:
    left, right = map(str, pair)
    per_direction_modality = {}
    values = []
    for source, target in ((left, right), (right, left)):
        direction = f"{source}->{target}"
        per_direction_modality[direction] = {}
        for modality in modalities:
            cell = [
                float(row["source_advantage"])
                for row in rows
                if str(row.get("source")) == source
                and str(row.get("target")) == target
                and str(row.get("modality")) == modality
                and (
                    position_class is None
                    or str(row.get("position_class")) == str(position_class)
                )
            ]
            value = _median(cell) if cell else None
            per_direction_modality[direction][str(modality)] = value
            if value is not None:
                values.append(value)
    complete = len(values) == 2 * len(tuple(modalities))
    return (min(values) if complete else None), per_direction_modality


def select_loading_tomography(
    instrument_rows: Mapping[str, Sequence[Mapping]],
    *,
    instrument_layers: Mapping[str, Sequence[int]],
    candidate_pairs: Sequence[Sequence[str]],
    capability: Mapping,
    modalities: Sequence[str] = ("text", "image", "spoken_audio"),
    min_source_advantage: float = 0.0,
    min_source_cosine: float = 0.0,
    evidence_position_margin: float = 0.0,
    primary_loading_position_class: str = "final_prompt_token",
    intervention_position_rule: str = "all_prompt_positions",
) -> dict:
    """Select one instrument/band from clean loading, never swap outcomes.

    Both directions of a pair must be capability-admissible and source-loaded.
    Ranking first maximizes contiguous band length, then the weakest
    direction×modality source advantage.  This prevents a strong text cell from
    hiding an absent image or audio coordinate.
    """

    modalities = tuple(map(str, modalities))
    eligible_concepts = set(map(str, capability.get("eligible_concepts") or ()))
    ranking = []
    for instrument, raw_rows in sorted(instrument_rows.items()):
        rows = list(raw_rows)
        layers = tuple(sorted(set(map(int, instrument_layers.get(instrument, ())))))
        if not rows or not layers:
            continue
        for pair in candidate_pairs:
            left, right = map(str, pair)
            capable = {left, right}.issubset(eligible_concepts)
            pair_rows = [
                row
                for row in rows
                if (str(row.get("source")), str(row.get("target")))
                in {(left, right), (right, left)}
            ]
            weakest, direction_scores = _bidirectional_score(
                pair_rows,
                pair=(left, right),
                modalities=modalities,
                position_class=primary_loading_position_class,
            )
            source_cosine_evidence = []
            cosine_eligible_layers = []
            for layer in layers:
                cell_medians = {}
                complete = True
                for source, target in ((left, right), (right, left)):
                    direction = f"{source}->{target}"
                    cell_medians[direction] = {}
                    for modality in modalities:
                        values = [
                            float(row["source_cosine"])
                            for row in pair_rows
                            if int(row.get("layer")) == layer
                            and str(row.get("source")) == source
                            and str(row.get("target")) == target
                            and str(row.get("modality")) == modality
                            and str(row.get("position_class"))
                            == str(primary_loading_position_class)
                        ]
                        value = _median(values) if values else None
                        cell_medians[direction][modality] = value
                        if value is None:
                            complete = False
                passed = bool(
                    complete
                    and all(
                        float(value) > float(min_source_cosine)
                        for direction in cell_medians.values()
                        for value in direction.values()
                    )
                )
                if passed:
                    cosine_eligible_layers.append(layer)
                source_cosine_evidence.append(
                    {
                        "layer": layer,
                        "direction_modality_medians": cell_medians,
                        "passed": passed,
                    }
                )
            localization = freeze_loading_localization(
                pair_rows,
                required_modalities=modalities,
                candidate_layers=(cosine_eligible_layers or layers),
                min_source_advantage=min_source_advantage,
                evidence_position_margin=evidence_position_margin,
                primary_position_class=primary_loading_position_class,
                intervention_position_rule=intervention_position_rule,
            )
            if not cosine_eligible_layers:
                localization = {
                    **localization,
                    "verdict": "LOADING_LOCALIZATION_NO_GO",
                    "eligible_layers": [],
                    "contiguous_runs": [],
                    "selected_band": [],
                    "position_rule_by_modality": {
                        modality: str(intervention_position_rule)
                        for modality in modalities
                    },
                }
            selected_band = list(localization["selected_band"])
            eligible = bool(capable and selected_band and weakest is not None)
            ranking.append(
                {
                    "instrument": str(instrument),
                    "pair": [left, right],
                    "capability_admissible": capable,
                    "weakest_direction_modality_advantage": weakest,
                    "direction_modality_advantages": direction_scores,
                    "min_source_cosine": float(min_source_cosine),
                    "source_cosine_evidence": source_cosine_evidence,
                    "source_cosine_eligible_layers": cosine_eligible_layers,
                    "loading_report": summarize_loading(pair_rows),
                    "localization": localization,
                    "selected_band": selected_band,
                    "band_length": len(selected_band),
                    "eligible": eligible,
                    "causal_result_consulted": False,
                }
            )
    ranking.sort(
        key=lambda row: (
            not row["eligible"],
            -int(row["band_length"]),
            -float(row["weakest_direction_modality_advantage"] or -1e30),
            str(row["instrument"]),
            tuple(row["pair"]),
        )
    )
    selected = next((row for row in ranking if row["eligible"]), None)
    payload = {
        "version": TOMOGRAPHY_VERSION,
        "verdict": (
            "MULTIMODAL_LOADING_TOMOGRAPHY_GO"
            if selected
            else "MULTIMODAL_LOADING_TOMOGRAPHY_NO_GO"
        ),
        "modalities": list(modalities),
        "min_source_advantage": float(min_source_advantage),
        "min_source_cosine": float(min_source_cosine),
        "evidence_position_margin": float(evidence_position_margin),
        "primary_loading_position_class": str(primary_loading_position_class),
        "intervention_position_rule": str(intervention_position_rule),
        "ranking": ranking,
        "selected_instrument": selected["instrument"] if selected else None,
        "selected_pair": selected["pair"] if selected else None,
        "selected_band": selected["selected_band"] if selected else [],
        "position_rule_by_modality": (
            selected["localization"]["position_rule_by_modality"]
            if selected
            else {}
        ),
        "selection_depended_on_causal_outcome": False,
        "causal_outcomes_opened": False,
    }
    return {**payload, "selection_digest": payload_checksum(payload)}


def causal_swap_report(
    rows: Sequence[Mapping],
    *,
    stage: str,
    modalities: Sequence[str] = ("text", "image", "spoken_audio"),
    min_clean_rate: float = 0.75,
    min_primary_rate: float = 0.25,
    min_primary_successes: int = 4,
    familywise_alpha: float = 0.05,
) -> dict:
    """Score exact alpha=1 swaps against paired zero/random/unrelated controls."""

    if stage not in {"development", "confirmation"}:
        raise ValueError("stage must be 'development' or 'confirmation'")
    rows = list(rows)
    modalities = tuple(map(str, modalities))
    if not rows:
        raise WorkspaceReplicationRefused("causal reporting needs trial rows")
    directions = sorted(
        {
            str(row.get("direction") or "")
            for row in rows
            if str(row.get("direction") or "")
        }
    )
    if len(directions) != 2:
        raise WorkspaceReplicationRefused(
            "causal reporting requires exactly two explicitly recorded directions"
        )
    cells = []
    global_raw_p = {}
    for direction in directions:
        for modality in modalities:
            cell = [
                row
                for row in rows
                if str(row.get("direction")) == direction
                and str(row.get("modality")) == modality
            ]
            if not cell:
                raise WorkspaceReplicationRefused(
                    f"missing causal cell {direction}/{modality}"
                )
            clean_successes = sum(bool(row.get("clean_correct")) for row in cell)
            primary = [
                bool(row["conditions"]["exact_alpha1"]["success"])
                for row in cell
            ]
            controls = {
                name: [bool(row["conditions"][name]["success"]) for row in cell]
                for name in ("zero", "random_alpha1", "unrelated_alpha1")
            }
            primary_successes = sum(primary)
            control_stats = {
                name: paired_binary_superiority(primary, values)
                for name, values in controls.items()
            }
            adjusted = holm_adjust(
                {
                    name: record["one_sided_exact_p"]
                    for name, record in control_stats.items()
                }
            )
            clean_pass = clean_successes / len(cell) >= float(min_clean_rate)
            effect_pass = (
                primary_successes >= int(min_primary_successes)
                and primary_successes / len(cell) >= float(min_primary_rate)
            )
            controls_pass = all(
                control_stats[name]["treatment_rate"]
                > control_stats[name]["control_rate"]
                and adjusted[name] <= float(familywise_alpha)
                for name in controls
            )
            integrity_pass = all(
                bool(row["conditions"][name].get("integrity_passed", True))
                for row in cell
                for name in (
                    "exact_alpha1",
                    "zero",
                    "random_alpha1",
                    "unrelated_alpha1",
                )
            )
            passed = clean_pass and effect_pass and controls_pass and integrity_pass
            for name, record in control_stats.items():
                global_raw_p[f"{direction}|{modality}|{name}"] = record[
                    "one_sided_exact_p"
                ]
            cells.append(
                {
                    "direction": direction,
                    "modality": modality,
                    "n": len(cell),
                    "clean_successes": clean_successes,
                    "clean_rate": clean_successes / len(cell),
                    "primary_successes": primary_successes,
                    "primary_rate": primary_successes / len(cell),
                    "controls": control_stats,
                    "holm_adjusted_p": adjusted,
                    "clean_pass": clean_pass,
                    "effect_pass": effect_pass,
                    "controls_pass": controls_pass,
                    "integrity_pass": integrity_pass,
                    "passed": passed,
                }
            )
    if stage == "confirmation":
        global_adjusted = holm_adjust(global_raw_p)
        for cell in cells:
            prefix = f"{cell['direction']}|{cell['modality']}|"
            cell["holm_adjusted_p"] = {
                name: global_adjusted[prefix + name]
                for name in cell["controls"]
            }
            cell["controls_pass"] = all(
                cell["controls"][name]["treatment_rate"]
                > cell["controls"][name]["control_rate"]
                and cell["holm_adjusted_p"][name]
                <= float(familywise_alpha)
                for name in cell["controls"]
            )
            cell["passed"] = bool(
                cell["clean_pass"]
                and cell["effect_pass"]
                and cell["controls_pass"]
                and cell["integrity_pass"]
            )
    else:
        global_adjusted = None
    cell_passes = [bool(cell["passed"]) for cell in cells]
    go = all(cell_passes)
    version = DEVELOPMENT_VERSION if stage == "development" else CONFIRMATION_VERSION
    payload = {
        "version": version,
        "stage": stage,
        "directions": directions,
        "both_directions_required": True,
        "multiplicity_scope": (
            "all_direction_modality_control_tests"
            if stage == "confirmation"
            else "within_direction_modality_development_gate"
        ),
        "global_holm_adjusted_p": global_adjusted,
        "verdict": (
            f"MULTIMODAL_SWAP_{stage.upper()}_GO"
            if go
            else f"MULTIMODAL_SWAP_{stage.upper()}_NO_GO"
        ),
        "cells": cells,
        "thresholds": {
            "min_clean_rate": float(min_clean_rate),
            "min_primary_rate": float(min_primary_rate),
            "min_primary_successes": int(min_primary_successes),
            "familywise_alpha": float(familywise_alpha),
        },
        "teacher_forcing_used": False,
        "candidate_list_supplied": False,
    }
    return {**payload, "report_checksum": payload_checksum(payload)}


def position_diagnostic_report(
    reports: Mapping[str, Mapping],
    *,
    original_development_report_checksum: str,
) -> dict:
    """Rank predeclared position strategies on opened development media.

    This is explicitly exploratory: causal outcomes choose the strategy, so a
    selected strategy can license only a separately frozen, untouched
    confirmation.  It can never amend the original all-position NO_GO.
    """

    expected = set(POSITION_DIAGNOSTIC_STRATEGIES)
    if set(map(str, reports)) != expected:
        raise WorkspaceReplicationRefused(
            "position diagnostics require exactly the predeclared strategies "
            f"{sorted(expected)}"
        )
    ranking = []
    shared = None
    for strategy in sorted(expected):
        report = dict(reports[strategy])
        if report.get("stage") != "development":
            raise WorkspaceReplicationRefused(
                f"position strategy {strategy!r} is not a development report"
            )
        identity = {
            "instrument": report.get("instrument"),
            "pair": list(report.get("pair") or ()),
            "layer_band": list(report.get("layer_band") or ()),
        }
        if shared is None:
            shared = identity
        elif identity != shared:
            raise WorkspaceReplicationRefused(
                "position strategies changed the instrument, pair, or layer band"
            )
        expected_rules = POSITION_DIAGNOSTIC_STRATEGIES[strategy]
        if dict(report.get("position_rule_by_modality") or {}) != expected_rules:
            raise WorkspaceReplicationRefused(
                f"position strategy {strategy!r} does not match its frozen rules"
            )
        cells = list(report.get("cells") or ())
        if len(cells) != 6:
            raise WorkspaceReplicationRefused(
                f"position strategy {strategy!r} must report six cells"
            )
        primary_successes = sum(int(cell.get("primary_successes") or 0) for cell in cells)
        control_successes = sum(
            int(round(float(control.get("control_rate") or 0.0) * int(cell["n"])))
            for cell in cells
            for control in dict(cell.get("controls") or {}).values()
        )
        ranking.append(
            {
                "strategy": strategy,
                "verdict": report.get("verdict"),
                "passed_cells": sum(bool(cell.get("passed")) for cell in cells),
                "primary_successes": primary_successes,
                "control_successes_across_three_arms": control_successes,
                "all_integrity_passed": all(
                    bool(cell.get("integrity_pass")) for cell in cells
                ),
                "report_checksum": report.get("report_checksum"),
                "position_rule_by_modality": dict(
                    report.get("position_rule_by_modality") or {}
                ),
            }
        )
    ranking.sort(
        key=lambda row: (
            row["verdict"] != "MULTIMODAL_SWAP_DEVELOPMENT_GO",
            -int(row["passed_cells"]),
            -int(row["primary_successes"]),
            int(row["control_successes_across_three_arms"]),
            str(row["strategy"]),
        )
    )
    selected = next(
        (
            row
            for row in ranking
            if row["verdict"] == "MULTIMODAL_SWAP_DEVELOPMENT_GO"
        ),
        None,
    )
    payload = {
        "version": POSITION_DIAGNOSTIC_VERSION,
        "verdict": (
            "MULTIMODAL_POSITION_DIAGNOSTIC_GO"
            if selected
            else "MULTIMODAL_POSITION_DIAGNOSTIC_NO_GO"
        ),
        **(shared or {}),
        "strategies_predeclared_before_new_trials": sorted(expected),
        "ranking": ranking,
        "selected_strategy": selected["strategy"] if selected else None,
        "selected_position_rule_by_modality": (
            selected["position_rule_by_modality"] if selected else None
        ),
        "original_development_report_checksum": str(
            original_development_report_checksum
        ),
        "original_all_position_no_go_remains_unchanged": True,
        "causal_outcomes_selected_position_strategy": True,
        "development_only": True,
        "fresh_confirmation_required": True,
        "teacher_forcing_used": False,
        "candidate_list_supplied": False,
    }
    return {**payload, "report_checksum": payload_checksum(payload)}


def freeze_position_selected_confirmation_design(
    *,
    capability: Mapping,
    tomography: Mapping,
    position_diagnostic: Mapping,
    development_population_digest: str,
    confirmation_population_digest: str,
    forbidden_development_image_ids: Sequence[str],
    forbidden_prior_image_ids: Sequence[str],
) -> dict:
    """Freeze a development-selected position strategy for fresh confirmation."""

    if position_diagnostic.get("verdict") != "MULTIMODAL_POSITION_DIAGNOSTIC_GO":
        raise WorkspaceReplicationRefused("position diagnostic did not pass")
    selected = str(position_diagnostic.get("selected_strategy") or "")
    if selected not in POSITION_DIAGNOSTIC_STRATEGIES:
        raise WorkspaceReplicationRefused("position diagnostic selected no known strategy")
    payload = {
        "version": POSITION_DESIGN_VERSION,
        "instrument": position_diagnostic.get("instrument"),
        "pair": list(position_diagnostic.get("pair") or ()),
        "layer_band": list(position_diagnostic.get("layer_band") or ()),
        "position_strategy": selected,
        "position_rule_by_modality": dict(POSITION_DIAGNOSTIC_STRATEGIES[selected]),
        "primary_alpha": 1.0,
        "primary_alpha_role": "exact_coordinate_exchange",
        "input_protocol": "mmpilot.multimodal_assistant_prefill_completion.v2",
        "output_endpoint": "unrestricted_greedy_complete_answer",
        "max_new_tokens": 4,
        "capability_report_checksum": capability.get("report_checksum"),
        "tomography_selection_digest": tomography.get("selection_digest"),
        "position_diagnostic_report_checksum": position_diagnostic.get(
            "report_checksum"
        ),
        "development_population_digest": str(development_population_digest),
        "confirmation_population_digest": str(confirmation_population_digest),
        "forbidden_development_image_ids": sorted(
            set(map(str, forbidden_development_image_ids))
        ),
        "forbidden_prior_image_ids": sorted(set(map(str, forbidden_prior_image_ids))),
        "teacher_forcing_used": False,
        "candidate_list_supplied": False,
        "causal_outcomes_selected_position_strategy": True,
        "fresh_confirmation_required": True,
        "original_all_position_no_go_remains_unchanged": True,
    }
    return {**payload, "design_digest": payload_checksum(payload)}


def freeze_repair_confirmation_design(
    *,
    capability: Mapping,
    tomography: Mapping,
    development: Mapping,
    development_population_digest: str,
    confirmation_population_digest: str,
    forbidden_development_image_ids: Sequence[str],
    forbidden_prior_image_ids: Sequence[str],
) -> dict:
    """Freeze the successful development design before confirmation outcomes."""

    if tomography.get("verdict") != "MULTIMODAL_LOADING_TOMOGRAPHY_GO":
        raise WorkspaceReplicationRefused("loading tomography did not pass")
    if development.get("verdict") != "MULTIMODAL_SWAP_DEVELOPMENT_GO":
        raise WorkspaceReplicationRefused("causal development did not pass")
    expected = {
        "instrument": tomography.get("selected_instrument"),
        "pair": list(tomography.get("selected_pair") or ()),
        "layer_band": list(tomography.get("selected_band") or ()),
        "position_rule_by_modality": dict(
            tomography.get("position_rule_by_modality") or {}
        ),
    }
    for key, value in expected.items():
        if development.get(key) != value:
            raise WorkspaceReplicationRefused(
                f"development {key} changed after clean loading selection"
            )
    if not expected["layer_band"] or len(expected["pair"]) != 2:
        raise WorkspaceReplicationRefused("the selected confirmation design is empty")
    payload = {
        "version": DESIGN_VERSION,
        **expected,
        "primary_alpha": 1.0,
        "primary_alpha_role": "exact_coordinate_exchange",
        "input_protocol": "mmpilot.multimodal_assistant_prefill_completion.v1",
        "output_endpoint": "unrestricted_greedy_complete_answer",
        "max_new_tokens": 4,
        "capability_report_checksum": capability.get("report_checksum"),
        "tomography_selection_digest": tomography.get("selection_digest"),
        "development_report_checksum": development.get("report_checksum"),
        "development_population_digest": str(development_population_digest),
        "confirmation_population_digest": str(confirmation_population_digest),
        "forbidden_development_image_ids": sorted(
            set(map(str, forbidden_development_image_ids))
        ),
        "forbidden_prior_image_ids": sorted(set(map(str, forbidden_prior_image_ids))),
        "teacher_forcing_used": False,
        "candidate_list_supplied": False,
        "selection_depended_on_causal_outcome": False,
        "causal_development_used_only_as_go_no_go_gate": True,
        "fresh_confirmation_required": True,
    }
    return {**payload, "design_digest": payload_checksum(payload)}


__all__ = [
    "CAPABILITY_VERSION",
    "CONFIRMATION_VERSION",
    "DEVELOPMENT_VERSION",
    "DESIGN_VERSION",
    "POSITION_DESIGN_VERSION",
    "POSITION_DIAGNOSTIC_STRATEGIES",
    "POSITION_DIAGNOSTIC_VERSION",
    "REPAIR_PROTOCOL_VERSION",
    "TOMOGRAPHY_VERSION",
    "causal_swap_report",
    "freeze_repair_confirmation_design",
    "freeze_position_selected_confirmation_design",
    "multimodal_capability_report",
    "position_diagnostic_report",
    "select_loading_tomography",
]
