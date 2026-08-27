"""Frozen scoring for the no-refit multimodal leg-count generalization study.

The study keeps the confirmed bird-source method fixed and varies only the
target identity.  Cat, ant, and spider imply three distinct downstream answers
(4, 6, and 8), so a passing result cannot be explained by bird-coordinate
removal alone.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from jlens.mmpilot.multimodal_lens import holm_adjust, paired_binary_one_sided_p
from jlens.mmpilot.store import payload_checksum

VERSION = "mmpilot.multimodal_leg_count_generalization.v1"
SOURCE = "bird"
TARGET_ANSWERS = {"cat": "4", "ant": "6", "spider": "8"}
NUMBER_WORDS = {"2": "two", "4": "four", "6": "six", "8": "eight"}
CALIBRATION_TARGET = "cat"
NOVEL_TARGETS = ("ant", "spider")
MODALITIES = ("text", "image", "spoken_audio")
CONDITIONS = ("exact", "zero", "random", "unrelated")


class LegCountGeneralizationRefused(RuntimeError):
    """The frozen study is incomplete or internally inconsistent."""


def leg_count_answer_matches(observed: str, expected: str) -> bool:
    """Match an unrestricted number answer without constraining generation.

    The model is never shown these alternatives.  They are a predeclared
    output-side equivalence between a digit and its ordinary English spelling.
    """

    normalized = " ".join(str(observed).strip().casefold().split())
    expected_digit = str(expected).strip()
    allowed = {expected_digit, NUMBER_WORDS[expected_digit]}
    return normalized in allowed


def frozen_design() -> dict:
    body = {
        "version": VERSION,
        "source": SOURCE,
        "target_answers": dict(TARGET_ANSWERS),
        "calibration_target": CALIBRATION_TARGET,
        "novel_targets": list(NOVEL_TARGETS),
        "modalities": list(MODALITIES),
        "conditions": list(CONDITIONS),
        "layers": list(range(16, 41)),
        "alpha": 1.0,
        "positions": "every_original_prompt_position",
        "lens_refitted": False,
        "development": {
            "candidates": 9,
            "recruited": 6,
            "minimum_exact_rate": 0.50,
            "minimum_control_margin": 0.25,
            "minimum_answer_leverage_rate": 0.75,
        },
        "confirmation": {
            "candidates": 22,
            "recruited": 12,
            "minimum_exact_rate": 0.75,
            "minimum_control_margin": 0.25,
            "minimum_answer_leverage_rate": 0.75,
            "familywise_alpha": 0.05,
        },
        "confirmation_rule": (
            "bird-to-cat calibration and at least one Holm-corrected novel "
            "target must pass every modality"
        ),
    }
    return {**body, "design_digest": payload_checksum(body)}


def _cell(rows: Sequence[Mapping], *, target: str, modality: str) -> dict:
    selected = [
        row
        for row in rows
        if row.get("target") == target and row.get("modality") == modality
    ]
    by_condition = {
        condition: [row for row in selected if row.get("condition") == condition]
        for condition in CONDITIONS
    }
    unit_sets = {
        condition: {str(row.get("group_id")) for row in condition_rows}
        for condition, condition_rows in by_condition.items()
    }
    if any(len(unit_ids) != len(by_condition[name]) for name, unit_ids in unit_sets.items()):
        raise LegCountGeneralizationRefused(
            f"duplicate rows in {target}/{modality} trial cell"
        )
    if len({frozenset(unit_ids) for unit_ids in unit_sets.values()}) != 1:
        raise LegCountGeneralizationRefused(
            f"conditions in {target}/{modality} are not paired"
        )
    summaries = {}
    for condition, condition_rows in by_condition.items():
        successes = sum(bool(row.get("success")) for row in condition_rows)
        summaries[condition] = {
            "n": len(condition_rows),
            "successes": successes,
            "rate": successes / len(condition_rows) if condition_rows else 0.0,
            "integrity_pass": bool(condition_rows)
            and all(bool(row.get("integrity_pass")) for row in condition_rows),
        }
    return {
        "target": target,
        "answer": TARGET_ANSWERS[target],
        "modality": modality,
        "conditions": summaries,
    }


def _leverage_cell(rows: Sequence[Mapping], *, target: str, modality: str) -> dict:
    selected = [
        row
        for row in rows
        if row.get("target") == target and row.get("modality") == modality
    ]
    units = [str(row.get("group_id")) for row in selected]
    if len(units) != len(set(units)):
        raise LegCountGeneralizationRefused(
            f"duplicate answer-leverage rows in {target}/{modality}"
        )
    successes = sum(bool(row.get("success")) for row in selected)
    return {
        "target": target,
        "answer": TARGET_ANSWERS[target],
        "modality": modality,
        "n": len(selected),
        "successes": successes,
        "rate": successes / len(selected) if selected else 0.0,
        "integrity_pass": bool(selected)
        and all(bool(row.get("integrity_pass")) for row in selected),
    }


def development_report(
    leverage_rows: Sequence[Mapping],
    trial_rows: Sequence[Mapping],
    *,
    expected_n: int = 6,
) -> dict:
    """Select novel targets on development data under the frozen method."""

    design = frozen_design()
    leverage = [
        _leverage_cell(leverage_rows, target=target, modality=modality)
        for target in TARGET_ANSWERS
        for modality in MODALITIES
    ]
    cells = [
        _cell(trial_rows, target=target, modality=modality)
        for target in TARGET_ANSWERS
        for modality in MODALITIES
    ]
    passing_targets = []
    target_results = []
    for target in TARGET_ANSWERS:
        target_leverage = [row for row in leverage if row["target"] == target]
        target_cells = [row for row in cells if row["target"] == target]
        leverage_pass = bool(target_leverage) and all(
            row["n"] == expected_n
            and row["rate"] >= design["development"]["minimum_answer_leverage_rate"]
            and row["integrity_pass"]
            for row in target_leverage
        )
        effect_pass = bool(target_cells) and all(
            cell["conditions"]["exact"]["n"] == expected_n
            and cell["conditions"]["exact"]["rate"]
            >= design["development"]["minimum_exact_rate"]
            and all(
                control["n"] == expected_n
                and cell["conditions"]["exact"]["rate"] - control["rate"]
                >= design["development"]["minimum_control_margin"]
                for control in (
                    cell["conditions"]["zero"],
                    cell["conditions"]["random"],
                    cell["conditions"]["unrelated"],
                )
            )
            and all(row["integrity_pass"] for row in cell["conditions"].values())
            for cell in target_cells
        )
        # Direct answer-coordinate insertion measures a different mechanism
        # from identity-driven downstream recomputation.  It is retained as a
        # diagnostic but cannot veto an exact identity exchange that passes
        # its own controls.
        passed = effect_pass
        if passed:
            passing_targets.append(target)
        target_results.append(
            {
                "target": target,
                "answer": TARGET_ANSWERS[target],
                "answer_leverage_passed": leverage_pass,
                "answer_leverage_is_diagnostic_only": True,
                "effect_passed": effect_pass,
                "passed": passed,
            }
        )
    calibration_passed = CALIBRATION_TARGET in passing_targets
    selected_novel_targets = [
        target for target in NOVEL_TARGETS if target in passing_targets
    ]
    passed = calibration_passed and bool(selected_novel_targets)
    body = {
        "version": VERSION,
        "stage": "development",
        "verdict": (
            "LEG_COUNT_GENERALIZATION_DEVELOPMENT_GO"
            if passed
            else "LEG_COUNT_GENERALIZATION_DEVELOPMENT_NO_GO"
        ),
        "design_digest": design["design_digest"],
        "calibration_passed": calibration_passed,
        "selected_novel_targets": selected_novel_targets,
        "target_results": target_results,
        "answer_leverage_cells": leverage,
        "effect_cells": cells,
        "fresh_confirmation_opened": False,
        "fitting_performed": False,
        "backward_passes": 0,
    }
    return {**body, "report_checksum": payload_checksum(body)}


def confirmation_report(
    leverage_rows: Sequence[Mapping],
    trial_rows: Sequence[Mapping],
    *,
    frozen_targets: Sequence[str],
    expected_n: int = 12,
) -> dict:
    """Score one untouched confirmation population with familywise control."""

    design = frozen_design()
    targets = tuple(map(str, frozen_targets))
    if CALIBRATION_TARGET not in targets or not set(targets).issubset(TARGET_ANSWERS):
        raise LegCountGeneralizationRefused(
            "confirmation targets must include cat and only predeclared targets"
        )
    if not set(targets).intersection(NOVEL_TARGETS):
        raise LegCountGeneralizationRefused("confirmation has no novel target")
    leverage = [
        _leverage_cell(leverage_rows, target=target, modality=modality)
        for target in targets
        for modality in MODALITIES
    ]
    cells = [
        _cell(trial_rows, target=target, modality=modality)
        for target in targets
        for modality in MODALITIES
    ]
    comparisons = []
    for cell in cells:
        exact_rows = sorted(
            [
                row
                for row in trial_rows
                if row.get("target") == cell["target"]
                and row.get("modality") == cell["modality"]
                and row.get("condition") == "exact"
            ],
            key=lambda row: str(row["group_id"]),
        )
        for control_name in ("zero", "random", "unrelated"):
            control_rows = sorted(
                [
                    row
                    for row in trial_rows
                    if row.get("target") == cell["target"]
                    and row.get("modality") == cell["modality"]
                    and row.get("condition") == control_name
                ],
                key=lambda row: str(row["group_id"]),
            )
            if [row["group_id"] for row in exact_rows] != [
                row["group_id"] for row in control_rows
            ]:
                raise LegCountGeneralizationRefused("paired row order differs")
            comparisons.append(
                {
                    "target": cell["target"],
                    "modality": cell["modality"],
                    "control": control_name,
                    **paired_binary_one_sided_p(
                        [bool(row.get("success")) for row in exact_rows],
                        [bool(row.get("success")) for row in control_rows],
                    ),
                }
            )
    adjusted = holm_adjust(comparisons)
    adjusted_index = {
        (row["target"], row["modality"], row["control"]): row
        for row in adjusted
    }
    target_results = []
    passing_targets = []
    for target in targets:
        target_leverage = [row for row in leverage if row["target"] == target]
        target_cells = [row for row in cells if row["target"] == target]
        leverage_pass = all(
            row["n"] == expected_n
            and row["rate"] >= design["confirmation"]["minimum_answer_leverage_rate"]
            and row["integrity_pass"]
            for row in target_leverage
        )
        effect_pass = all(
            cell["conditions"]["exact"]["n"] == expected_n
            and cell["conditions"]["exact"]["rate"]
            >= design["confirmation"]["minimum_exact_rate"]
            and all(
                cell["conditions"]["exact"]["rate"]
                - cell["conditions"][control]["rate"]
                >= design["confirmation"]["minimum_control_margin"]
                and adjusted_index[(target, cell["modality"], control)][
                    "holm_adjusted_p"
                ]
                <= design["confirmation"]["familywise_alpha"]
                for control in ("zero", "random", "unrelated")
            )
            and all(row["integrity_pass"] for row in cell["conditions"].values())
            for cell in target_cells
        )
        passed = leverage_pass and effect_pass
        if passed:
            passing_targets.append(target)
        target_results.append(
            {
                "target": target,
                "answer": TARGET_ANSWERS[target],
                "answer_leverage_passed": leverage_pass,
                "effect_passed": effect_pass,
                "passed": passed,
            }
        )
    calibration_passed = CALIBRATION_TARGET in passing_targets
    novel_passing = [target for target in NOVEL_TARGETS if target in passing_targets]
    passed = calibration_passed and bool(novel_passing)
    body = {
        "version": VERSION,
        "stage": "fresh_confirmation",
        "verdict": (
            "FRESH_MULTIMODAL_LEG_COUNT_GENERALIZATION_GO"
            if passed
            else "FRESH_MULTIMODAL_LEG_COUNT_GENERALIZATION_NO_GO"
        ),
        "design_digest": design["design_digest"],
        "frozen_targets": list(targets),
        "calibration_passed": calibration_passed,
        "novel_passing_targets": novel_passing,
        "target_results": target_results,
        "answer_leverage_cells": leverage,
        "effect_cells": cells,
        "paired_comparisons": adjusted,
        "familywise_alpha": design["confirmation"]["familywise_alpha"],
        "fitting_performed": False,
        "backward_passes": 0,
    }
    return {**body, "report_checksum": payload_checksum(body)}


__all__ = [
    "CALIBRATION_TARGET",
    "CONDITIONS",
    "MODALITIES",
    "NUMBER_WORDS",
    "NOVEL_TARGETS",
    "SOURCE",
    "TARGET_ANSWERS",
    "VERSION",
    "LegCountGeneralizationRefused",
    "confirmation_report",
    "development_report",
    "frozen_design",
    "leg_count_answer_matches",
]
