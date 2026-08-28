"""No-refit downstream-property generalization for the confirmed bird->cat swap.

This extension changes the *question*, not the causal instrument.  Candidate
properties are selected using clean-answer capability before any coordinate-
exchange outcome is opened.  Development and fresh confirmation then use the
checksum-pinned pooled multimodal J-lens, exact alpha-one exchange, and the
same negative controls as the confirmed leg-count study.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from jlens.mmpilot.multimodal_lens import holm_adjust, paired_binary_one_sided_p
from jlens.mmpilot.store import payload_checksum

VERSION = "mmpilot.bird_cat_downstream_property_generalization.v3"
SOURCE = "bird"
TARGET = "cat"
MODALITIES = ("text", "image", "spoken_audio")
CONDITIONS = ("exact", "zero", "unrelated", "random_0", "random_1", "random_2")
CONTROL_CONDITIONS = CONDITIONS[1:]
TWO_MODALITY_PRIMARY = ("image", "spoken_audio")
TWO_MODALITY_SECONDARY = ("text",)

# Ordered before capability is measured.  Selection uses clean capability only,
# never an intervention outcome.  Both properties are nonvisual facts with a
# short, unrestricted answer and a different answer for bird and cat.
PROPERTY_SPECS = {
    "taxonomic_class": {
        "question": (
            "What biological class does the animal in the evidence belong to? "
            "Answer with exactly one word."
        ),
        "source_answer": "bird",
        "target_answer": "mammal",
        "source_aliases": ("bird", "avian", "aves"),
        "target_aliases": ("mammal", "mammalia", "mammalian"),
        "rationale": "taxonomy is identity-dependent and is not visible in a still image",
    },
    "young_name": {
        "question": (
            "What is the usual one-word name for a young member of the animal "
            "species in the evidence? Answer with exactly one word."
        ),
        "source_answer": "chick",
        "target_answer": "kitten",
        "source_aliases": (
            "chick",
            "chicks",
            "fledgling",
            "fledglings",
            "hatchling",
            "hatchlings",
        ),
        "target_aliases": ("kitten", "kittens"),
        "rationale": "offspring name is identity-dependent and absent from the evidence",
    },
}
PROPERTY_PRIORITY = tuple(PROPERTY_SPECS)


class BirdCatPropertyRefused(RuntimeError):
    """The frozen property-generalization protocol is incomplete."""


def frozen_design() -> dict:
    body = {
        "version": VERSION,
        "source": SOURCE,
        "target": TARGET,
        "property_priority": list(PROPERTY_PRIORITY),
        "properties": PROPERTY_SPECS,
        "modalities": list(MODALITIES),
        "conditions": list(CONDITIONS),
        "layers": list(range(16, 41)),
        "alpha": 1.0,
        "positions": "every_original_prompt_position",
        "output_endpoint": "unrestricted_greedy_complete_answer",
        "max_new_tokens": 4,
        "model_dtype": "torch.float32",
        "teacher_forcing_used": False,
        "candidate_list_supplied": False,
        "selection_uses": "clean capability only; no intervention outcome",
        "development": {
            "n": 6,
            "minimum_clean_rate": 0.75,
            "minimum_exact_rate": 0.50,
            "minimum_control_margin": 0.25,
        },
        "confirmation": {
            "candidates": 22,
            "n": 12,
            "minimum_clean_rate": 0.75,
            "minimum_exact_rate": 0.75,
            "minimum_control_margin": 0.25,
            "familywise_alpha": 0.05,
        },
        "lens_refitted": False,
    }
    return {**body, "design_digest": payload_checksum(body)}


def property_prompt(property_name: str, modality: str, caption: str = "") -> str:
    if property_name not in PROPERTY_SPECS:
        raise BirdCatPropertyRefused(f"unknown property {property_name!r}")
    if modality not in MODALITIES:
        raise BirdCatPropertyRefused(f"unknown modality {modality!r}")
    question = PROPERTY_SPECS[property_name]["question"]
    if modality == "text":
        return f"Evidence: {caption}\n{question}\nAnswer:"
    if modality == "image":
        return f"Use the attached image as the only evidence. {question}\nAnswer:"
    return f"Use the spoken recording as the only evidence. {question}\nAnswer:"


def answer_matches(observed: str, property_name: str, role: str) -> bool:
    if property_name not in PROPERTY_SPECS or role not in {"source", "target"}:
        raise BirdCatPropertyRefused("unknown property or answer role")
    normalized = re.sub(r"[^a-z]+", " ", str(observed).casefold()).strip()
    words = set(normalized.split())
    aliases = PROPERTY_SPECS[property_name][f"{role}_aliases"]
    return any(alias.casefold() in words for alias in aliases)


def capability_report(rows: Sequence[Mapping], *, expected_n: int = 6) -> dict:
    design = frozen_design()
    cells = []
    for property_name in PROPERTY_PRIORITY:
        for modality in MODALITIES:
            selected = [
                row
                for row in rows
                if row.get("property") == property_name
                and row.get("modality") == modality
            ]
            groups = [str(row.get("group_id")) for row in selected]
            if len(groups) != len(set(groups)):
                raise BirdCatPropertyRefused("duplicate capability rows")
            successes = sum(bool(row.get("pass")) for row in selected)
            cells.append(
                {
                    "property": property_name,
                    "modality": modality,
                    "n": len(selected),
                    "successes": successes,
                    "rate": successes / len(selected) if selected else 0.0,
                }
            )
    candidates = []
    for priority, property_name in enumerate(PROPERTY_PRIORITY):
        own = [row for row in cells if row["property"] == property_name]
        complete = all(row["n"] == expected_n for row in own)
        minimum = min((row["rate"] for row in own), default=0.0)
        mean = sum(row["rate"] for row in own) / len(own) if own else 0.0
        passed = complete and minimum >= design["development"]["minimum_clean_rate"]
        candidates.append(
            {
                "property": property_name,
                "passed": passed,
                "minimum_rate": minimum,
                "mean_rate": mean,
                "priority": priority,
            }
        )
    passing = [row for row in candidates if row["passed"]]
    passing.sort(key=lambda row: (-row["minimum_rate"], -row["mean_rate"], row["priority"]))
    selected = passing[0]["property"] if passing else None
    body = {
        "version": VERSION,
        "stage": "capability_selection",
        "verdict": "BIRD_CAT_PROPERTY_CAPABILITY_GO" if selected else "BIRD_CAT_PROPERTY_CAPABILITY_NO_GO",
        "design_digest": design["design_digest"],
        "selected_property": selected,
        "selection_read_intervention_outcomes": False,
        "cells": cells,
        "candidates": candidates,
    }
    return {**body, "report_checksum": payload_checksum(body)}


def _effect_cells(rows: Sequence[Mapping], *, property_name: str, expected_n: int) -> list[dict]:
    cells = []
    for modality in MODALITIES:
        selected = [
            row
            for row in rows
            if row.get("property") == property_name and row.get("modality") == modality
        ]
        conditions = {}
        group_sets = []
        for condition in CONDITIONS:
            own = [row for row in selected if row.get("condition") == condition]
            groups = [str(row.get("group_id")) for row in own]
            if len(groups) != len(set(groups)):
                raise BirdCatPropertyRefused("duplicate intervention rows")
            group_sets.append(set(groups))
            successes = sum(bool(row.get("success")) for row in own)
            conditions[condition] = {
                "n": len(own),
                "successes": successes,
                "rate": successes / len(own) if own else 0.0,
                "integrity_pass": bool(own)
                and all(bool(row.get("integrity_pass")) for row in own),
            }
        if not group_sets or any(group_set != group_sets[0] for group_set in group_sets[1:]):
            raise BirdCatPropertyRefused("conditions are not paired")
        if any(row["n"] != expected_n for row in conditions.values()):
            raise BirdCatPropertyRefused("an intervention cell is incomplete")
        cells.append({"property": property_name, "modality": modality, "conditions": conditions})
    return cells


def development_report(
    capability: Mapping, rows: Sequence[Mapping], *, expected_n: int = 6
) -> dict:
    design = frozen_design()
    if capability.get("verdict") != "BIRD_CAT_PROPERTY_CAPABILITY_GO":
        property_name = None
        cells = []
        passed = False
    else:
        property_name = str(capability["selected_property"])
        cells = _effect_cells(rows, property_name=property_name, expected_n=expected_n)
        passed = all(
            cell["conditions"]["exact"]["rate"]
            >= design["development"]["minimum_exact_rate"]
            and all(
                cell["conditions"]["exact"]["rate"]
                - cell["conditions"][control]["rate"]
                >= design["development"]["minimum_control_margin"]
                for control in CONTROL_CONDITIONS
            )
            and all(row["integrity_pass"] for row in cell["conditions"].values())
            for cell in cells
        )
    body = {
        "version": VERSION,
        "stage": "development",
        "verdict": "BIRD_CAT_PROPERTY_DEVELOPMENT_GO" if passed else "BIRD_CAT_PROPERTY_DEVELOPMENT_NO_GO",
        "design_digest": design["design_digest"],
        "selected_property": property_name,
        "capability_report_checksum": capability.get("report_checksum"),
        "effect_cells": cells,
        "fresh_confirmation_opened": False,
        "fitting_performed": False,
        "backward_passes": 0,
    }
    return {**body, "report_checksum": payload_checksum(body)}


def confirmation_report(
    rows: Sequence[Mapping], *, property_name: str, expected_n: int = 12
) -> dict:
    design = frozen_design()
    if property_name not in PROPERTY_SPECS:
        raise BirdCatPropertyRefused("confirmation property was not predeclared")
    cells = _effect_cells(rows, property_name=property_name, expected_n=expected_n)
    comparisons = []
    for modality in MODALITIES:
        exact = sorted(
            [row for row in rows if row.get("property") == property_name and row.get("modality") == modality and row.get("condition") == "exact"],
            key=lambda row: str(row["group_id"]),
        )
        for control in CONTROL_CONDITIONS:
            other = sorted(
                [row for row in rows if row.get("property") == property_name and row.get("modality") == modality and row.get("condition") == control],
                key=lambda row: str(row["group_id"]),
            )
            if [row["group_id"] for row in exact] != [row["group_id"] for row in other]:
                raise BirdCatPropertyRefused("confirmation rows are not paired")
            comparisons.append(
                {
                    "modality": modality,
                    "control": control,
                    **paired_binary_one_sided_p(
                        [bool(row.get("success")) for row in exact],
                        [bool(row.get("success")) for row in other],
                    ),
                }
            )
    adjusted = holm_adjust(comparisons)
    adjusted_index = {(row["modality"], row["control"]): row for row in adjusted}
    passed = all(
        cell["conditions"]["exact"]["rate"]
        >= design["confirmation"]["minimum_exact_rate"]
        and all(
            cell["conditions"]["exact"]["rate"] - cell["conditions"][control]["rate"]
            >= design["confirmation"]["minimum_control_margin"]
            and adjusted_index[(cell["modality"], control)]["holm_adjusted_p"]
            <= design["confirmation"]["familywise_alpha"]
            for control in CONTROL_CONDITIONS
        )
        and all(row["integrity_pass"] for row in cell["conditions"].values())
        for cell in cells
    )
    body = {
        "version": VERSION,
        "stage": "fresh_confirmation",
        "verdict": "FRESH_MULTIMODAL_BIRD_CAT_PROPERTY_GENERALIZATION_GO" if passed else "FRESH_MULTIMODAL_BIRD_CAT_PROPERTY_GENERALIZATION_NO_GO",
        "design_digest": design["design_digest"],
        "frozen_property": property_name,
        "effect_cells": cells,
        "paired_comparisons": adjusted,
        "familywise_alpha": design["confirmation"]["familywise_alpha"],
        "fitting_performed": False,
        "backward_passes": 0,
    }
    return {**body, "report_checksum": payload_checksum(body)}


def two_modality_confirmation_report(
    rows: Sequence[Mapping], *, property_name: str, expected_n: int = 18
) -> dict:
    """Score the prospective image-plus-audio follow-up.

    The primary modalities were selected from the completed development run,
    so this report labels that provenance explicitly. Text remains a frozen
    secondary outcome and cannot make or break the primary verdict.
    """
    if property_name not in PROPERTY_SPECS:
        raise BirdCatPropertyRefused("confirmation property was not predeclared")
    cells = _effect_cells(rows, property_name=property_name, expected_n=expected_n)
    comparisons = []
    for modality in TWO_MODALITY_PRIMARY:
        exact = sorted(
            [
                row
                for row in rows
                if row.get("property") == property_name
                and row.get("modality") == modality
                and row.get("condition") == "exact"
            ],
            key=lambda row: str(row["group_id"]),
        )
        for control in CONTROL_CONDITIONS:
            other = sorted(
                [
                    row
                    for row in rows
                    if row.get("property") == property_name
                    and row.get("modality") == modality
                    and row.get("condition") == control
                ],
                key=lambda row: str(row["group_id"]),
            )
            if [row["group_id"] for row in exact] != [
                row["group_id"] for row in other
            ]:
                raise BirdCatPropertyRefused("confirmation rows are not paired")
            comparisons.append(
                {
                    "modality": modality,
                    "control": control,
                    **paired_binary_one_sided_p(
                        [bool(row.get("success")) for row in exact],
                        [bool(row.get("success")) for row in other],
                    ),
                }
            )
    adjusted = holm_adjust(comparisons)
    adjusted_index = {
        (row["modality"], row["control"]): row for row in adjusted
    }
    cell_index = {cell["modality"]: cell for cell in cells}
    minimum_exact_rate = 0.50
    minimum_control_margin = 0.25
    familywise_alpha = 0.05
    passed = all(
        cell_index[modality]["conditions"]["exact"]["rate"]
        >= minimum_exact_rate
        and all(
            cell_index[modality]["conditions"]["exact"]["rate"]
            - cell_index[modality]["conditions"][control]["rate"]
            >= minimum_control_margin
            and adjusted_index[(modality, control)]["holm_adjusted_p"]
            <= familywise_alpha
            for control in CONTROL_CONDITIONS
        )
        and all(
            row["integrity_pass"]
            for row in cell_index[modality]["conditions"].values()
        )
        for modality in TWO_MODALITY_PRIMARY
    )
    body = {
        "version": "mmpilot.bird_cat_property_two_modality_confirmation.v1",
        "stage": "fresh_two_modality_confirmation",
        "verdict": (
            "FRESH_BIRD_CAT_PROPERTY_TWO_MODALITY_GENERALIZATION_GO"
            if passed
            else "FRESH_BIRD_CAT_PROPERTY_TWO_MODALITY_GENERALIZATION_NO_GO"
        ),
        "selection_provenance": (
            "image and spoken_audio selected from completed fp32 development; "
            "fresh confirmation population remained sealed until design freeze"
        ),
        "frozen_property": property_name,
        "primary_modalities": list(TWO_MODALITY_PRIMARY),
        "secondary_modalities": list(TWO_MODALITY_SECONDARY),
        "minimum_exact_rate": minimum_exact_rate,
        "minimum_control_margin": minimum_control_margin,
        "effect_cells": cells,
        "primary_paired_comparisons": adjusted,
        "familywise_alpha": familywise_alpha,
        "fitting_performed": False,
        "backward_passes": 0,
    }
    return {**body, "report_checksum": payload_checksum(body)}


__all__ = [
    "BirdCatPropertyRefused",
    "CONDITIONS",
    "CONTROL_CONDITIONS",
    "MODALITIES",
    "PROPERTY_PRIORITY",
    "PROPERTY_SPECS",
    "TWO_MODALITY_PRIMARY",
    "TWO_MODALITY_SECONDARY",
    "answer_matches",
    "capability_report",
    "confirmation_report",
    "development_report",
    "frozen_design",
    "property_prompt",
    "two_modality_confirmation_report",
]
