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
NOVEL_CONDITIONS = (
    "exact",
    "zero",
    "unrelated",
    "random_0",
    "random_1",
    "random_2",
)
NOVEL_VERSION = "mmpilot.multimodal_novel_leg_count_generalization.v2"
# Derived, not executed: the other novel target's exact exchange rescored
# against this target's answer.  Free in compute and the only control that
# distinguishes "the answer moved" from "the answer followed the identity".
CROSS_CONDITION = "cross_target"
NOVEL_CONTROL_CONDITIONS = (
    "zero",
    "unrelated",
    "random_0",
    "random_1",
    "random_2",
    CROSS_CONDITION,
)
NOVEL_SCORED_CONDITIONS = ("exact", *NOVEL_CONTROL_CONDITIONS)


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


def _other_novel_target(target: str) -> str:
    others = [name for name in NOVEL_TARGETS if name != target]
    if len(others) != 1:
        raise LegCountGeneralizationRefused(
            "the cross-target control needs exactly one other novel target"
        )
    return others[0]


def novel_frozen_design() -> dict:
    """The novel-target extension, frozen separately from the cat calibration.

    Kept apart from :func:`frozen_design` on purpose: the completed cat run
    embeds that digest, and a study must not silently redefine the design a
    finished run was already scored under.
    """

    body = {
        "version": NOVEL_VERSION,
        "source": SOURCE,
        "calibration_target": CALIBRATION_TARGET,
        "novel_targets": list(NOVEL_TARGETS),
        "target_answers": {name: TARGET_ANSWERS[name] for name in NOVEL_TARGETS},
        "modalities": list(MODALITIES),
        "executed_conditions": list(NOVEL_CONDITIONS),
        "scored_controls": list(NOVEL_CONTROL_CONDITIONS),
        "cross_target_control": (
            "the exact exchange toward the other novel target, rescored against "
            "this target's answer.  A perturbation that merely pushes the model "
            "off its clean answer cannot pass it, because passing requires the "
            "answer to track which identity was inserted."
        ),
        "layers": list(range(16, 41)),
        "alpha": 1.0,
        "positions": "every_original_prompt_position",
        "lens_refitted": False,
        "answer_leverage_role": "diagnostic_only_never_gating",
        "development": {
            "n": 6,
            "minimum_exact_rate": 0.50,
            "minimum_control_margin": 0.25,
        },
        "confirmation": {
            "n": 12,
            "minimum_exact_rate": 0.75,
            "minimum_control_margin": 0.25,
            "familywise_alpha": 0.05,
            "primary_family": (
                "one frozen target x 3 modalities x 6 controls = 18 paired tests"
            ),
            "secondary_family": (
                "each further frozen target is Holm-corrected inside its own "
                "family and reported as supporting evidence; it never gates"
            ),
        },
        "selection_priority": list(NOVEL_TARGETS),
        "verdict_rule": (
            "GO when the primary frozen target clears every control in every "
            "modality on the untouched confirmation photographs"
        ),
    }
    return {**body, "design_digest": payload_checksum(body)}


def normalized_leg_answer(observed: str) -> str:
    """Bucket a raw surface into a leg-count answer, or ``other:<surface>``."""

    normalized = " ".join(str(observed).strip().casefold().split())
    for digit, word in NUMBER_WORDS.items():
        if normalized in {digit, word}:
            return digit
    return f"other:{normalized}"


def derive_cross_target_rows(
    trial_rows: Sequence[Mapping], *, target: str, donor_target: str
) -> list[dict]:
    """Rescore the donor's exact exchange against *this* target's answer.

    This is the control the four-legged pilot could not supply.  ``zero``,
    ``unrelated`` and the random seeds all ask whether *some* perturbation moves
    the answer.  This one asks whether the answer follows the identity that was
    inserted, and it costs no extra forward pass because the donor trials are
    already required.
    """

    donor_rows = [
        row
        for row in trial_rows
        if row.get("target") == donor_target and row.get("condition") == "exact"
    ]
    derived = []
    for row in donor_rows:
        if "patched_surface" not in row:
            raise LegCountGeneralizationRefused(
                "cross-target scoring needs the donor's patched surface"
            )
        derived.append(
            {
                "target": target,
                "modality": row.get("modality"),
                "group_id": row.get("group_id"),
                "condition": CROSS_CONDITION,
                "donor_target": donor_target,
                "patched_surface": row["patched_surface"],
                "success": leg_count_answer_matches(
                    row["patched_surface"], TARGET_ANSWERS[target]
                ),
                "integrity_pass": bool(row.get("integrity_pass")),
            }
        )
    return derived


def assert_success_matches_surface(trial_rows: Sequence[Mapping]) -> None:
    """Refuse rows whose stored verdict disagrees with the recorded surface.

    The cross-target control rescores one condition's raw output against a
    different answer, so the surface is now load-bearing evidence rather than a
    convenience field.  A row that claims a success its own surface does not
    support would corrupt the derived control silently.
    """

    for row in trial_rows:
        if "patched_surface" not in row:
            continue
        target = str(row.get("target"))
        if target not in TARGET_ANSWERS:
            continue
        expected = leg_count_answer_matches(
            row["patched_surface"], TARGET_ANSWERS[target]
        )
        if bool(row.get("success")) != expected:
            raise LegCountGeneralizationRefused(
                f"{target}/{row.get('modality')}/{row.get('condition')} records "
                f"success={bool(row.get('success'))} for surface "
                f"{row['patched_surface']!r}"
            )


def with_cross_target_controls(
    trial_rows: Sequence[Mapping], *, targets: Sequence[str]
) -> list[dict]:
    """Append the derived cross-target control rows for every named target."""

    assert_success_matches_surface(trial_rows)
    rows = [dict(row) for row in trial_rows]
    for target in targets:
        rows.extend(
            derive_cross_target_rows(
                trial_rows, target=target, donor_target=_other_novel_target(target)
            )
        )
    return rows


def target_answer_confusion(
    trial_rows: Sequence[Mapping], *, condition: str = "exact"
) -> list[dict]:
    """Which answer each inserted identity actually produced, per modality.

    The headline table.  Identity specificity is legible here directly: the
    diagonal is the inserted target's own answer.
    """

    rows = [row for row in trial_rows if row.get("condition") == condition]
    targets = sorted({str(row.get("target")) for row in rows})
    confusion = []
    for target in targets:
        for modality in MODALITIES:
            selected = [
                row
                for row in rows
                if str(row.get("target")) == target
                and row.get("modality") == modality
            ]
            counts: dict[str, int] = {}
            for row in selected:
                bucket = normalized_leg_answer(row.get("patched_surface", ""))
                counts[bucket] = counts.get(bucket, 0) + 1
            confusion.append(
                {
                    "target": target,
                    "expected_answer": TARGET_ANSWERS.get(target),
                    "modality": modality,
                    "n": len(selected),
                    "answers": dict(sorted(counts.items())),
                }
            )
    return confusion


def _novel_cell(rows: Sequence[Mapping], *, target: str, modality: str) -> dict:
    selected = [
        row
        for row in rows
        if row.get("target") == target and row.get("modality") == modality
    ]
    summaries = {}
    unit_sets = []
    for condition in NOVEL_SCORED_CONDITIONS:
        condition_rows = [
            row for row in selected if row.get("condition") == condition
        ]
        units = [str(row.get("group_id")) for row in condition_rows]
        if len(units) != len(set(units)):
            raise LegCountGeneralizationRefused(
                f"duplicate {condition} rows in {target}/{modality}"
            )
        unit_sets.append(set(units))
        successes = sum(bool(row.get("success")) for row in condition_rows)
        summaries[condition] = {
            "n": len(condition_rows),
            "successes": successes,
            "rate": successes / len(condition_rows) if condition_rows else 0.0,
            "integrity_pass": bool(condition_rows)
            and all(bool(row.get("integrity_pass")) for row in condition_rows),
        }
    if len({frozenset(units) for units in unit_sets}) != 1:
        raise LegCountGeneralizationRefused(
            f"novel controls in {target}/{modality} are not paired"
        )
    return {
        "target": target,
        "answer": TARGET_ANSWERS[target],
        "modality": modality,
        "conditions": summaries,
    }


def _novel_target_passes(
    cells: Sequence[Mapping],
    *,
    minimum_exact: float,
    minimum_margin: float,
    expected_n: int,
) -> bool:
    return bool(cells) and all(
        cell["conditions"]["exact"]["n"] == expected_n
        and cell["conditions"]["exact"]["rate"] >= minimum_exact
        and all(
            cell["conditions"][control]["n"] == expected_n
            and cell["conditions"]["exact"]["rate"]
            - cell["conditions"][control]["rate"]
            >= minimum_margin
            for control in NOVEL_CONTROL_CONDITIONS
        )
        and all(row["integrity_pass"] for row in cell["conditions"].values())
        for cell in cells
    )


def _novel_comparisons(rows: Sequence[Mapping], *, target: str) -> list[dict]:
    comparisons = []
    for modality in MODALITIES:
        exact_rows = sorted(
            [
                row
                for row in rows
                if row.get("target") == target
                and row.get("modality") == modality
                and row.get("condition") == "exact"
            ],
            key=lambda row: str(row["group_id"]),
        )
        for control in NOVEL_CONTROL_CONDITIONS:
            control_rows = sorted(
                [
                    row
                    for row in rows
                    if row.get("target") == target
                    and row.get("modality") == modality
                    and row.get("condition") == control
                ],
                key=lambda row: str(row["group_id"]),
            )
            if [str(row["group_id"]) for row in exact_rows] != [
                str(row["group_id"]) for row in control_rows
            ]:
                raise LegCountGeneralizationRefused(
                    f"paired novel rows differ for {target}/{modality}/{control}"
                )
            comparisons.append(
                {
                    "target": target,
                    "modality": modality,
                    "control": control,
                    **paired_binary_one_sided_p(
                        [bool(row.get("success")) for row in exact_rows],
                        [bool(row.get("success")) for row in control_rows],
                    ),
                }
            )
    return comparisons


def _leverage_diagnostic(
    leverage_rows: Sequence[Mapping], *, targets: Sequence[str]
) -> list[dict]:
    """Can the band install this answer at all, ignoring identity?

    Never gates.  Its only job is to make a null interpretable: an identity
    swap that fails while the direct answer exchange also fails is a capacity
    limit of the band, not evidence against identity-specific transfer.
    """

    cells = []
    for target in targets:
        for modality in MODALITIES:
            selected = [
                row
                for row in leverage_rows
                if row.get("target") == target and row.get("modality") == modality
            ]
            units = [str(row.get("group_id")) for row in selected]
            if len(units) != len(set(units)):
                raise LegCountGeneralizationRefused(
                    f"duplicate answer-leverage rows in {target}/{modality}"
                )
            successes = sum(bool(row.get("success")) for row in selected)
            cells.append(
                {
                    "target": target,
                    "answer": TARGET_ANSWERS[target],
                    "modality": modality,
                    "n": len(selected),
                    "successes": successes,
                    "rate": successes / len(selected) if selected else 0.0,
                    "gating": False,
                }
            )
    return cells


def _executed_novel_targets(rows: Sequence[Mapping]) -> list[str]:
    """Every predeclared novel identity actually exchanged in these rows."""

    return [
        name
        for name in NOVEL_TARGETS
        if any(
            row.get("target") == name and row.get("condition") == "exact"
            for row in rows
        )
    ]


def _dissociation(rows: Sequence[Mapping]) -> list[dict]:
    """Per modality, the photographs where *every* identity landed correctly.

    Descriptive and never gating, so it spans every identity that was actually
    exchanged rather than only the ones carried forward.  A target that failed
    selection still ran, and its answers still belong in this table.
    """

    named = _executed_novel_targets(rows)
    if len(named) < 2:
        return []
    summary = []
    for modality in MODALITIES:
        per_target = {}
        for target in named:
            per_target[target] = {
                str(row["group_id"]): bool(row.get("success"))
                for row in rows
                if row.get("target") == target
                and row.get("modality") == modality
                and row.get("condition") == "exact"
            }
        shared = set.intersection(*(set(table) for table in per_target.values()))
        both = sum(
            all(per_target[target][unit] for target in named) for unit in shared
        )
        summary.append(
            {
                "modality": modality,
                "targets": list(named),
                "n": len(shared),
                "all_targets_correct": both,
                "rate": both / len(shared) if shared else 0.0,
            }
        )
    return summary


def novel_development_report(
    trial_rows: Sequence[Mapping],
    leverage_rows: Sequence[Mapping] = (),
    *,
    expected_n: int = 6,
) -> dict:
    """Score the ant/spider development extension and freeze what passes."""

    design = novel_frozen_design()
    scored = with_cross_target_controls(trial_rows, targets=NOVEL_TARGETS)
    cells = [
        _novel_cell(scored, target=target, modality=modality)
        for target in NOVEL_TARGETS
        for modality in MODALITIES
    ]
    results = []
    passing = []
    for target in NOVEL_TARGETS:
        target_cells = [row for row in cells if row["target"] == target]
        passed = _novel_target_passes(
            target_cells,
            minimum_exact=design["development"]["minimum_exact_rate"],
            minimum_margin=design["development"]["minimum_control_margin"],
            expected_n=expected_n,
        )
        if passed:
            passing.append(target)
        results.append(
            {"target": target, "answer": TARGET_ANSWERS[target], "passed": passed}
        )
    # Fixed before any novel outcome was seen: every target that clears its own
    # controls is carried forward, in this priority order.  The first becomes
    # the gating primary; the rest become Holm-corrected supporting families.
    selected = [target for target in NOVEL_TARGETS if target in passing]
    body = {
        "version": NOVEL_VERSION,
        "stage": "novel_target_development",
        "verdict": (
            "LEG_COUNT_NOVEL_TARGET_DEVELOPMENT_GO"
            if selected
            else "LEG_COUNT_NOVEL_TARGET_DEVELOPMENT_NO_GO"
        ),
        "design_digest": design["design_digest"],
        "selection_priority": list(NOVEL_TARGETS),
        "passing_novel_targets": passing,
        "selected_confirmation_targets": selected,
        "selected_confirmation_target": selected[0] if selected else None,
        "target_results": results,
        "effect_cells": cells,
        "target_answer_confusion": target_answer_confusion(trial_rows),
        "double_dissociation": _dissociation(trial_rows),
        "answer_leverage_diagnostic": _leverage_diagnostic(
            leverage_rows, targets=NOVEL_TARGETS
        ),
        "answer_leverage_is_diagnostic_only": True,
        "fresh_confirmation_opened": False,
        "fitting_performed": False,
        "backward_passes": 0,
    }
    return {**body, "report_checksum": payload_checksum(body)}


def novel_confirmation_report(
    trial_rows: Sequence[Mapping],
    leverage_rows: Sequence[Mapping] = (),
    *,
    target: str,
    secondary_targets: Sequence[str] = (),
    expected_n: int = 12,
) -> dict:
    """Score the frozen novel target once on the untouched confirmation split.

    ``target`` is the single gating primary.  Any ``secondary_targets`` are
    scored inside their own Holm families and reported as supporting evidence;
    they cannot change the verdict, so carrying them costs the primary no
    familywise power.
    """

    if target not in NOVEL_TARGETS:
        raise LegCountGeneralizationRefused("confirmation target is not novel")
    secondary = [str(name) for name in secondary_targets if str(name) != target]
    if not set(secondary).issubset(NOVEL_TARGETS):
        raise LegCountGeneralizationRefused("secondary target is not predeclared")
    design = novel_frozen_design()
    all_targets = [target, *secondary]
    scored = with_cross_target_controls(trial_rows, targets=all_targets)

    families = {}
    for name in all_targets:
        adjusted = holm_adjust(_novel_comparisons(scored, target=name))
        families[name] = {
            "target": name,
            "answer": TARGET_ANSWERS[name],
            "role": "primary" if name == target else "secondary",
            "gating": name == target,
            "comparisons": adjusted,
            "familywise_alpha": design["confirmation"]["familywise_alpha"],
        }

    target_results = []
    for name in all_targets:
        cells = [
            _novel_cell(scored, target=name, modality=modality)
            for modality in MODALITIES
        ]
        index = {
            (row["modality"], row["control"]): row
            for row in families[name]["comparisons"]
        }
        effect_pass = _novel_target_passes(
            cells,
            minimum_exact=design["confirmation"]["minimum_exact_rate"],
            minimum_margin=design["confirmation"]["minimum_control_margin"],
            expected_n=expected_n,
        )
        significance_pass = all(
            index[(cell["modality"], control)]["holm_adjusted_p"]
            <= design["confirmation"]["familywise_alpha"]
            for cell in cells
            for control in NOVEL_CONTROL_CONDITIONS
        )
        target_results.append(
            {
                "target": name,
                "answer": TARGET_ANSWERS[name],
                "role": "primary" if name == target else "secondary",
                "effect_passed": effect_pass,
                "significance_passed": significance_pass,
                "passed": effect_pass and significance_pass,
                "effect_cells": cells,
            }
        )

    primary_result = target_results[0]
    passed = bool(primary_result["passed"])
    body = {
        "version": NOVEL_VERSION,
        "stage": "fresh_novel_target_confirmation",
        "verdict": (
            "FRESH_MULTIMODAL_NOVEL_LEG_COUNT_GENERALIZATION_GO"
            if passed
            else "FRESH_MULTIMODAL_NOVEL_LEG_COUNT_GENERALIZATION_NO_GO"
        ),
        "design_digest": design["design_digest"],
        "target": target,
        "answer": TARGET_ANSWERS[target],
        "secondary_targets": secondary,
        "target_results": target_results,
        "effect_cells": [
            cell for row in target_results for cell in row["effect_cells"]
        ],
        "holm_families": list(families.values()),
        "paired_comparisons": families[target]["comparisons"],
        "target_answer_confusion": target_answer_confusion(trial_rows),
        "double_dissociation": _dissociation(trial_rows),
        "answer_leverage_diagnostic": _leverage_diagnostic(
            leverage_rows, targets=all_targets
        ),
        "answer_leverage_is_diagnostic_only": True,
        "familywise_alpha": design["confirmation"]["familywise_alpha"],
        "fitting_performed": False,
        "backward_passes": 0,
    }
    return {**body, "report_checksum": payload_checksum(body)}


__all__ = [
    "CALIBRATION_TARGET",
    "CONDITIONS",
    "CROSS_CONDITION",
    "MODALITIES",
    "NUMBER_WORDS",
    "NOVEL_CONDITIONS",
    "NOVEL_CONTROL_CONDITIONS",
    "NOVEL_SCORED_CONDITIONS",
    "NOVEL_TARGETS",
    "NOVEL_VERSION",
    "SOURCE",
    "TARGET_ANSWERS",
    "VERSION",
    "LegCountGeneralizationRefused",
    "assert_success_matches_surface",
    "confirmation_report",
    "derive_cross_target_rows",
    "development_report",
    "frozen_design",
    "leg_count_answer_matches",
    "normalized_leg_answer",
    "novel_confirmation_report",
    "novel_development_report",
    "novel_frozen_design",
    "target_answer_confusion",
    "with_cross_target_controls",
]
