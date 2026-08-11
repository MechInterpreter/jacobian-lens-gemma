# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Paper-style hidden-intermediate coordinate-swap study helpers.

This module deliberately contains no model loading and no notebook state.  It
defines the pre-model population selection, the sampled-layer suffix bands,
and the frozen aggregation/verdict for the real experiment described in
``Verbalizable Representations Form a Global Workspace in Language Models``.

The causal comparison is between two uses of the *same* intervention:

* intermediate arm: exchange the inferred animal coordinates (e.g. bird/cat);
* answer arm: exchange the corresponding answer coordinates (e.g. two/four).

Both use ``h + V(sigma(pinv(V)h) - pinv(V)h)`` at every original prompt
position.  A result is evidence for earlier intermediate computation only when
the intermediate arm first works at a shallower tested band than the answer
arm.  Native direct readout and source-derived steering are not inputs.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

from jlens.mmpilot.evidence import EvidenceConfig, visual_evidence
from jlens.mmpilot.prompt_protocol import (
    CONCEPT_DOMAINS,
    DOMAIN_ANIMAL,
    concept_spec,
    contains_surface,
    leg_count_surfaces,
    normalize,
    resolve_leg_count,
)
from jlens.mmpilot.selection import stable_rank
from jlens.mmpilot.store import payload_checksum

PAPER_REASONING_SWAP_VERSION = "mmpilot.paper_reasoning_coordinate_swap.v1"
POPULATION_VERSION = "mmpilot.hidden_animal_population.v1"
SAMPLED_BAND_VERSION = "mmpilot.confirmed_sampled_suffix_bands.v1"
VERDICT_VERSION = "mmpilot.paper_reasoning_onset_verdict.v1"


class PaperSwapRefused(RuntimeError):
    """The paper-style study cannot be constructed without changing its design."""


@dataclass(frozen=True)
class PaperSwapThresholds:
    """Frozen image-level gates for one layer-band/arm/modality/direction cell."""

    min_images: int = 4
    min_clean_accuracy: float = 0.70
    min_target_flip_rate: float = 0.50
    min_joint_intermediate_rate: float = 0.50
    max_answer_identity_flip_rate: float = 0.25
    min_margin_gain: float = 0.0
    control_margin: float = 0.0

    def __post_init__(self) -> None:
        if self.min_images < 2:
            raise ValueError("min_images must be at least 2")
        for name in (
            "min_clean_accuracy",
            "min_target_flip_rate",
            "min_joint_intermediate_rate",
            "max_answer_identity_flip_rate",
        ):
            value = float(getattr(self, name))
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1], got {value}")

    @property
    def digest(self) -> str:
        return payload_checksum({"version": VERDICT_VERSION, **asdict(self)})


def sampled_suffix_bands(
    layers: Sequence[int], *, validated_layers: Sequence[int]
) -> tuple[tuple[int, ...], ...]:
    """Suffix bands over an explicitly sampled, independently confirmed grid.

    Anthropic reports interventions over layer ranges while evaluating an
    evenly sampled layer grid.  Gemma currently has confirmed lenses only at
    32/35/38/40, so each start depth patches that layer and every later member
    of the *same confirmed grid*.  Missing physical layers are never called
    validated or silently treated as patched.
    """
    grid = tuple(int(layer) for layer in layers)
    if not grid or tuple(sorted(set(grid))) != grid:
        raise PaperSwapRefused(f"sampled layers must be distinct and increasing, got {grid}")
    validated = {int(layer) for layer in validated_layers}
    missing = [layer for layer in grid if layer not in validated]
    if missing:
        raise PaperSwapRefused(
            f"sampled layer(s) {missing} lack independent lens confirmation"
        )
    return tuple(grid[index:] for index in range(len(grid)))


def sampled_band_record(
    layers: Sequence[int], *, validated_layers: Sequence[int]
) -> dict:
    bands = sampled_suffix_bands(layers, validated_layers=validated_layers)
    payload = {
        "version": SAMPLED_BAND_VERSION,
        "sampled_layers": list(map(int, layers)),
        "validated_layers": sorted(map(int, validated_layers)),
        "bands": [list(band) for band in bands],
        "start_layers": [band[0] for band in bands],
        "physical_layers_between_samples_are_unpatched": True,
        "limitation": (
            "sparse confirmed-layer grid; onset is localized only among tested "
            "start depths, not every physical layer"
        ),
    }
    return {**payload, "digest": payload_checksum(payload)}


def _surface_present(text: str, surfaces: Sequence[str]) -> bool:
    haystack = normalize(text)
    return any(contains_surface(haystack, surface) for surface in surfaces)


def hidden_animal_population(
    groups: Sequence[Mapping],
    *,
    concept_names: Sequence[str],
    evidence_config: EvidenceConfig,
    images_per_concept: int,
    seed: str,
    domain_registry: Mapping[str, str] | None = None,
    leg_counts: Mapping[str, Sequence[int]] | None = None,
) -> dict:
    """Select one synchronized group per image with the animal name hidden.

    Selection uses COCO object annotations for ground truth.  The caption (and
    therefore SpokenCOCO transcript) must contain none of the selected animal's
    aliases and neither the numeric nor word form of its leg-count answer.
    This is stricter than pair-specific filtering and makes every selected row
    reusable for any predeclared pair without exposing a target.
    """
    if images_per_concept < 2:
        raise PaperSwapRefused("images_per_concept must be at least 2")
    registry = CONCEPT_DOMAINS if domain_registry is None else domain_registry
    names = tuple(str(name) for name in concept_names)
    if len(set(names)) != len(names):
        raise PaperSwapRefused(f"duplicate concept names: {names}")

    specs = {}
    answer_surfaces: set[str] = set()
    for name in names:
        if registry.get(name) != DOMAIN_ANIMAL:
            raise PaperSwapRefused(f"{name!r} is not registered as an animal")
        count = resolve_leg_count(name, registry=leg_counts)
        specs[name] = concept_spec(name, domain_registry=registry)
        answer_surfaces.update(leg_count_surfaces(count))
        answer_surfaces.add(str(count))

    all_aliases = sorted(
        {surface for spec in specs.values() for surface in spec.surface_forms},
        key=lambda value: (len(value), value),
    )
    eligible: dict[str, list[dict]] = defaultdict(list)
    rejected = defaultdict(int)
    seen_group_ids: set[str] = set()
    for original in groups:
        row = dict(original)
        group_id = str(row.get("group_id") or "")
        image_id = str(row.get("image_id") or "")
        caption = str(row.get("caption") or "").strip()
        if not group_id or group_id in seen_group_ids:
            continue
        seen_group_ids.add(group_id)
        if not (
            image_id
            and caption
            and row.get("image_path")
            and row.get("audio_path")
        ):
            rejected["unsynchronized"] += 1
            continue
        if _surface_present(caption, all_aliases):
            rejected["entity_surface_in_caption_or_transcript"] += 1
            continue
        if _surface_present(caption, tuple(answer_surfaces)):
            rejected["property_answer_in_caption_or_transcript"] += 1
            continue
        for name in names:
            if visual_evidence(row, name, evidence_config)["present"]:
                eligible[name].append(row)

    selected: list[dict] = []
    coverage = {}
    spent_images: set[str] = set()
    for name in names:
        by_image: dict[str, list[dict]] = defaultdict(list)
        for row in eligible[name]:
            by_image[str(row["image_id"])].append(row)
        ranked_images = sorted(
            by_image,
            key=lambda image_id: stable_rank(
                f"{name}|{image_id}", f"{seed}|{POPULATION_VERSION}"
            ),
        )
        chosen = []
        for image_id in ranked_images:
            if image_id in spent_images:
                continue
            siblings = sorted(
                by_image[image_id],
                key=lambda row: stable_rank(
                    str(row["group_id"]), f"{seed}|{name}|group"
                ),
            )
            picked = {**siblings[0], "concept": name}
            chosen.append(picked)
            spent_images.add(image_id)
            if len(chosen) == images_per_concept:
                break
        coverage[name] = {
            "eligible_distinct_images": len(by_image),
            "selected_distinct_images": len(chosen),
            "required": int(images_per_concept),
        }
        if len(chosen) < images_per_concept:
            raise PaperSwapRefused(
                f"{name!r} has only {len(chosen)} disjoint hidden-caption images; "
                f"{images_per_concept} were predeclared"
            )
        selected.extend(chosen)

    image_ids = [str(row["image_id"]) for row in selected]
    group_ids = [str(row["group_id"]) for row in selected]
    if len(set(image_ids)) != len(image_ids) or len(set(group_ids)) != len(group_ids):
        raise PaperSwapRefused("the selected population is not image/group unique")
    payload = {
        "version": POPULATION_VERSION,
        "seed": str(seed),
        "concepts": list(names),
        "images_per_concept": int(images_per_concept),
        "n_groups": len(selected),
        "n_distinct_images": len(set(image_ids)),
        "one_group_per_image": True,
        "caption_is_spokencoco_transcript": True,
        "entity_aliases_hidden": all_aliases,
        "property_answers_hidden": sorted(answer_surfaces),
        "coverage": coverage,
        "rejections": dict(sorted(rejected.items())),
        "groups": selected,
    }
    return {**payload, "population_digest": payload_checksum(payload)}


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def summarize_cells(records: Sequence[Mapping]) -> list[dict]:
    """Aggregate stored per-image conditions without pseudoreplication."""
    grouped: dict[tuple, list[Mapping]] = defaultdict(list)
    for row in records:
        key = (
            int(row["start_layer"]),
            str(row["arm"]),
            str(row["condition"]),
            str(row["modality"]),
            str(row["source"]),
            str(row["target"]),
            str(row["readout"]),
        )
        grouped[key].append(row)
    out = []
    for key, rows in sorted(grouped.items()):
        images = {str(row["image_id"]) for row in rows}
        if len(images) != len(rows):
            raise PaperSwapRefused(
                f"cell {key} has {len(rows)} rows but {len(images)} images"
            )
        out.append(
            {
                "start_layer": key[0],
                "arm": key[1],
                "condition": key[2],
                "modality": key[3],
                "source": key[4],
                "target": key[5],
                "readout": key[6],
                "n_images": len(images),
                "clean_source_accuracy": _mean(
                    [1.0 if row["clean_prediction"] == row["source_answer"] else 0.0 for row in rows]
                ),
                "target_flip_rate": _mean(
                    [1.0 if row["prediction"] == row["target_answer"] else 0.0 for row in rows]
                ),
                "target_flip_image_ids": sorted(
                    str(row["image_id"])
                    for row in rows
                    if row["prediction"] == row["target_answer"]
                ),
                "mean_target_margin_change": _mean(
                    [float(row["target_margin_change"]) for row in rows]
                ),
                "image_ids": sorted(images),
            }
        )
    return out


def _cell_index(cells: Sequence[Mapping]) -> dict[tuple, Mapping]:
    return {
        (
            int(row["start_layer"]), str(row["arm"]), str(row["condition"]),
            str(row["modality"]), str(row["source"]), str(row["target"]),
            str(row["readout"]),
        ): row
        for row in cells
    }


def paper_onset_verdict(
    cells: Sequence[Mapping],
    *,
    bands: Sequence[Sequence[int]],
    directed_pairs: Sequence[Mapping],
    modalities: Sequence[str],
    thresholds: PaperSwapThresholds | None = None,
) -> dict:
    """Compare intermediate-coordinate and answer-coordinate onset depth."""
    thresholds = thresholds or PaperSwapThresholds()
    index = _cell_index(cells)
    controls = ("zero", "random", "unrelated", "position_control")
    starts = [int(band[0]) for band in bands]
    detail = []
    passing: dict[str, list[int]] = {"intermediate": [], "answer": []}

    for arm in ("intermediate", "answer"):
        for start in starts:
            clauses = []
            for pair in directed_pairs:
                source, target = str(pair["source"]), str(pair["target"])
                for modality in modalities:
                    primary = {}
                    for readout in ("identity", "property"):
                        key = (start, arm, "swap", modality, source, target, readout)
                        cell = index.get(key)
                        if cell is None:
                            clauses.append(False)
                            continue
                        primary[readout] = cell
                        passed = (
                            int(cell["n_images"]) >= thresholds.min_images
                            and float(cell["clean_source_accuracy"]) >= thresholds.min_clean_accuracy
                        )
                        drives_target = arm == "intermediate" or readout == "property"
                        if drives_target:
                            passed = passed and (
                                float(cell["mean_target_margin_change"])
                                > thresholds.min_margin_gain
                            )
                        for control in (controls if drives_target else ()):
                            control_cell = index.get(
                                (start, arm, control, modality, source, target, readout)
                            )
                            passed = passed and control_cell is not None and (
                                float(cell["mean_target_margin_change"])
                                > float(control_cell["mean_target_margin_change"])
                                + thresholds.control_margin
                            )
                        clauses.append(bool(passed))
                    if set(primary) != {"identity", "property"}:
                        continue
                    identity, property_cell = primary["identity"], primary["property"]
                    if set(identity["image_ids"]) != set(property_cell["image_ids"]):
                        clauses.append(False)
                        continue
                    if arm == "intermediate":
                        joint = set(identity["target_flip_image_ids"]) & set(
                            property_cell["target_flip_image_ids"]
                        )
                        clauses.append(
                            len(joint) / int(identity["n_images"])
                            >= thresholds.min_joint_intermediate_rate
                        )
                    else:
                        clauses.extend(
                            (
                                float(property_cell["target_flip_rate"])
                                >= thresholds.min_target_flip_rate,
                                float(identity["target_flip_rate"])
                                <= thresholds.max_answer_identity_flip_rate,
                            )
                        )
            band_passed = bool(clauses) and all(clauses)
            detail.append(
                {"arm": arm, "start_layer": start, "passed": band_passed, "n_clauses": len(clauses)}
            )
            if band_passed:
                passing[arm].append(start)

    intermediate_onset = min(passing["intermediate"], default=None)
    answer_onset = min(passing["answer"], default=None)
    if intermediate_onset is not None and answer_onset is not None:
        if intermediate_onset < answer_onset:
            verdict = "PAPER_STYLE_EARLIER_INTERMEDIATE_GO"
        elif intermediate_onset == answer_onset:
            verdict = "PAPER_STYLE_SAME_TESTED_DEPTH"
        else:
            verdict = "PAPER_STYLE_ANSWER_EARLIER"
    else:
        verdict = "PAPER_STYLE_INCONCLUSIVE"
    payload = {
        "version": VERDICT_VERSION,
        "verdict": verdict,
        "intermediate_onset_start_layer": intermediate_onset,
        "answer_onset_start_layer": answer_onset,
        "tested_start_layers": starts,
        "passing_start_layers": passing,
        "cells": detail,
        "thresholds": asdict(thresholds),
        "threshold_digest": thresholds.digest,
        "native_direct_readout_used": False,
        "source_derived_steering_used": False,
        "causal_comparison": "exact intermediate-coordinate swap vs exact answer-coordinate swap",
        "localization_scope": "confirmed sampled start layers only",
    }
    return {**payload, "verdict_digest": payload_checksum(payload)}


__all__ = [
    "PAPER_REASONING_SWAP_VERSION",
    "POPULATION_VERSION",
    "SAMPLED_BAND_VERSION",
    "VERDICT_VERSION",
    "PaperSwapRefused",
    "PaperSwapThresholds",
    "hidden_animal_population",
    "paper_onset_verdict",
    "sampled_band_record",
    "sampled_suffix_bands",
    "summarize_cells",
]
