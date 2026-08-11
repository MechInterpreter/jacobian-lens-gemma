# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Paper-style hidden-intermediate coordinate-swap study helpers.

This module deliberately contains no model loading and no notebook state.  It
defines the pre-model population selection, the historical v1 suffix bands,
the independent-layer v2 design, and the frozen aggregation/verdicts for the
real experiment described in
``Verbalizable Representations Form a Global Workspace in Language Models``.

The causal comparison is between two uses of the *same* intervention:

* intermediate arm: exchange the inferred animal coordinates (e.g. bird/cat);
* answer arm: exchange the corresponding answer coordinates (e.g. two/four).

Both use ``h + V(sigma(pinv(V)h) - pinv(V)h)`` at every original prompt
position.  V2 applies one exchange at one independently confirmed physical
layer per forward pass.  A result is evidence for earlier intermediate
computation only when the intermediate arm first works at a shallower tested
layer than the answer arm.  Native direct readout and source-derived steering
are not inputs.
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

# V2 is intentionally additive.  The completed v1 run remains interpretable
# under its original schema, while new runs bind these versions into a new
# fingerprint and can never resume into the old unit store.
PAPER_REASONING_SWAP_V2_VERSION = "mmpilot.paper_reasoning_coordinate_swap.v2"
INDEPENDENT_LAYER_VERSION = "mmpilot.confirmed_independent_layers.v2"
CAPABILITY_SELECTION_VERSION = "mmpilot.paper_capability_selection.v2"
VERDICT_V2_VERSION = "mmpilot.paper_reasoning_onset_verdict.v2"
DIRECTION_MATCHED_AMENDMENT_VERSION = (
    "mmpilot.paper_reasoning_direction_matched_amendment.v3"
)
ALPHA2_CONFIRMATION_VERSION = (
    "mmpilot.paper_reasoning_alpha2_independent_confirmation.v2"
)


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


@dataclass(frozen=True)
class PaperSwapV2Thresholds:
    """Frozen gates for the independent-layer paper-style replication.

    The final-token-only intervention is deliberately absent: it is a useful
    position diagnostic, but the paper's published success criterion does not
    require an all-position swap to outperform a final-position swap.
    """

    min_images: int = 4
    min_target_flip_rate: float = 0.50
    min_joint_intermediate_rate: float = 0.50
    max_answer_identity_flip_rate: float = 0.25
    min_margin_gain: float = 0.0
    control_margin: float = 0.0
    blocking_controls: tuple[str, ...] = ("zero", "random", "unrelated")

    def __post_init__(self) -> None:
        if self.min_images < 2:
            raise ValueError("min_images must be at least 2")
        for name in (
            "min_target_flip_rate",
            "min_joint_intermediate_rate",
            "max_answer_identity_flip_rate",
        ):
            value = float(getattr(self, name))
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if not self.blocking_controls:
            raise ValueError("at least one blocking control is required")
        if "position_descriptive" in self.blocking_controls:
            raise ValueError("the final-token position diagnostic is non-blocking")

    @property
    def digest(self) -> str:
        return payload_checksum({"version": VERDICT_V2_VERSION, **asdict(self)})


def independent_layer_record(
    layers: Sequence[int], *, validated_layers: Sequence[int]
) -> dict:
    """One exact exchange per independently confirmed physical layer.

    This is the **v2 run's own design record**, and the completed run's
    fingerprint is bound to it, so its payload is frozen exactly as written.

    ``repeated_exchange_forbidden`` records what v2 chose to do; it is not a
    property of the method and it does not forbid a band. The involution
    cancels *one fixed update applied twice*. A contiguous-band clamp is a
    different operation: each physical layer has its own ``W_U @ J_l``, so each
    layer's ``V`` is a different pair of directions and each re-reads ``c`` from
    the activation that actually arrived there. Two band layers cancel only
    insofar as their bases and coordinates agree, which in a real transformer
    they do not.

    What v2 actually lacked was lenses: the confirmed grid was 32/35/38/40, and
    33, 34, 36, 37 and 39 had none, so no contiguous band over that window
    existed to clamp. :mod:`jlens.mmpilot.band_swap` and
    :mod:`jlens.mmpilot.band_lens` are the study that produces them and runs the
    band. Every cell here remains exactly the published two-coordinate equation,
    applied at every original prompt position, once.
    """
    grid = tuple(int(layer) for layer in layers)
    if not grid or tuple(sorted(set(grid))) != grid:
        raise PaperSwapRefused(
            f"sampled layers must be distinct and increasing, got {grid}"
        )
    validated = {int(layer) for layer in validated_layers}
    missing = [layer for layer in grid if layer not in validated]
    if missing:
        raise PaperSwapRefused(
            f"sampled layer(s) {missing} lack independent lens confirmation"
        )
    payload = {
        "version": INDEPENDENT_LAYER_VERSION,
        "sampled_layers": list(grid),
        "validated_layers": sorted(validated),
        "bands": [[layer] for layer in grid],
        "start_layers": list(grid),
        "one_exchange_per_forward_pass": True,
        "repeated_exchange_forbidden": True,
        "physical_layers_between_samples_are_unpatched": True,
        "limitation": (
            "onset is localized among four independently confirmed physical "
            "layers; untested layers are not interpolated"
        ),
    }
    return {**payload, "digest": payload_checksum(payload)}


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


def select_capability_eligible_samples(
    records: Sequence[Mapping],
    *,
    concepts: Sequence[str],
    modalities: Sequence[str],
    max_images_per_cell: int,
    min_images_per_cell: int,
    seed: str,
) -> dict:
    """Freeze the causal population from clean behavior only.

    A group is eligible only when both the hidden identity and hidden property
    questions were answered correctly in *every* required modality before any
    intervention.  Selection is one seeded stable rank per concept, so every
    modality is causally tested on the same synchronized photographs and
    recordings.  No swap result participates.
    """
    if max_images_per_cell < min_images_per_cell:
        raise PaperSwapRefused(
            "max_images_per_cell cannot be smaller than min_images_per_cell"
        )
    by_cell: dict[tuple[str, str], dict[str, dict[str, Mapping]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in records:
        concept = str(row.get("source") or "")
        modality = str(row.get("modality") or "")
        readout = str(row.get("readout") or "")
        group_id = str(row.get("group_id") or "")
        if (
            concept in concepts
            and modality in modalities
            and readout in {"identity", "property"}
            and group_id
        ):
            by_cell[(concept, modality)][group_id][readout] = row

    selected: dict[str, list[str]] = {}
    cells = []
    all_sufficient = True
    for concept in concepts:
        group_sets = [
            set(by_cell[(str(concept), str(modality))]) for modality in modalities
        ]
        common_groups = set.intersection(*group_sets) if group_sets else set()
        eligible = []
        for group_id in common_groups:
            image_ids = set()
            valid = True
            for modality in modalities:
                readouts = by_cell[(str(concept), str(modality))][group_id]
                if set(readouts) != {"identity", "property"} or not all(
                    bool(readouts[name].get("correct"))
                    for name in ("identity", "property")
                ):
                    valid = False
                    break
                image_ids.update(
                    str(readouts[name].get("image_id"))
                    for name in ("identity", "property")
                )
            if not valid:
                continue
            if len(image_ids) != 1:
                raise PaperSwapRefused(
                    f"{group_id} maps to inconsistent images in the clean gate"
                )
            eligible.append((group_id, next(iter(image_ids))))
        eligible.sort(
            key=lambda item: stable_rank(
                f"{concept}|{item[1]}|{item[0]}",
                f"{seed}|{CAPABILITY_SELECTION_VERSION}",
            )
        )
        chosen = eligible[: int(max_images_per_cell)]
        sufficient = len(chosen) >= int(min_images_per_cell)
        all_sufficient = all_sufficient and sufficient
        for modality in modalities:
            key = f"{concept}|{modality}"
            selected[key] = [group_id for group_id, _ in chosen]
            cells.append(
                {
                    "concept": str(concept),
                    "modality": str(modality),
                    "n_eligible": len(eligible),
                    "n_selected": len(chosen),
                    "selected_group_ids": selected[key],
                    "selected_image_ids": [image_id for _, image_id in chosen],
                    "minimum_required": int(min_images_per_cell),
                    "sufficient": sufficient,
                }
            )
    payload = {
        "version": CAPABILITY_SELECTION_VERSION,
        "seed": str(seed),
        "concepts": list(map(str, concepts)),
        "modalities": list(map(str, modalities)),
        "max_images_per_cell": int(max_images_per_cell),
        "min_images_per_cell": int(min_images_per_cell),
        "all_cells_sufficient": all_sufficient,
        "cells": cells,
        "selected_group_ids": selected,
        "same_groups_in_every_modality": True,
        "swap_results_consulted": False,
    }
    return {**payload, "digest": payload_checksum(payload)}


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


def _v2_direction_cell(
    index: Mapping[tuple, Mapping],
    *,
    layer: int,
    arm: str,
    condition: str,
    modality: str,
    source: str,
    target: str,
    thresholds: PaperSwapV2Thresholds,
) -> dict:
    """Evaluate one directed, modality-specific v2 cell with named clauses."""
    primary = {
        readout: index.get(
            (layer, arm, condition, modality, source, target, readout)
        )
        for readout in ("identity", "property")
    }
    clauses: dict[str, bool] = {
        "identity_record_present": primary["identity"] is not None,
        "property_record_present": primary["property"] is not None,
    }
    if None in primary.values():
        return {
            "layer": layer,
            "arm": arm,
            "condition": condition,
            "modality": modality,
            "source": source,
            "target": target,
            "passed": False,
            "clauses": clauses,
            "position_diagnostic": None,
        }

    identity = primary["identity"]
    property_cell = primary["property"]
    same_images = set(identity["image_ids"]) == set(property_cell["image_ids"])
    clauses["same_images_for_both_readouts"] = same_images
    clauses["minimum_independent_images"] = (
        int(identity["n_images"]) >= thresholds.min_images
        and int(property_cell["n_images"]) >= thresholds.min_images
    )

    driven = ("identity", "property") if arm == "intermediate" else ("property",)
    control_comparisons = {}
    for readout in driven:
        cell = primary[readout]
        clauses[f"{readout}_positive_margin_gain"] = (
            float(cell["mean_target_margin_change"]) > thresholds.min_margin_gain
        )
        alpha_suffix = "alpha2" if condition == "swap_alpha2" else "alpha1"
        for control in thresholds.blocking_controls:
            control_condition = (
                control if control == "zero" else f"{control}_{alpha_suffix}"
            )
            control_cell = index.get(
                (
                    layer,
                    arm,
                    control_condition,
                    modality,
                    source,
                    target,
                    readout,
                )
            )
            passed = control_cell is not None and (
                float(cell["mean_target_margin_change"])
                > float(control_cell["mean_target_margin_change"])
                + thresholds.control_margin
            )
            clauses[f"{readout}_beats_{control_condition}"] = passed
            control_comparisons[f"{readout}|{control_condition}"] = {
                "present": control_cell is not None,
                "primary_margin_gain": float(cell["mean_target_margin_change"]),
                "control_margin_gain": (
                    float(control_cell["mean_target_margin_change"])
                    if control_cell is not None
                    else None
                ),
                "passed": passed,
            }

    if arm == "intermediate":
        identity_flips = set(identity["target_flip_image_ids"])
        property_flips = set(property_cell["target_flip_image_ids"])
        denominator = max(1, int(identity["n_images"]))
        joint_rate = len(identity_flips & property_flips) / denominator
        clauses["identity_flip_rate"] = (
            float(identity["target_flip_rate"]) >= thresholds.min_target_flip_rate
        )
        clauses["property_flip_rate"] = (
            float(property_cell["target_flip_rate"])
            >= thresholds.min_target_flip_rate
        )
        clauses["joint_identity_property_flip_rate"] = (
            joint_rate >= thresholds.min_joint_intermediate_rate
        )
    else:
        joint_rate = None
        clauses["property_flip_rate"] = (
            float(property_cell["target_flip_rate"])
            >= thresholds.min_target_flip_rate
        )
        clauses["identity_preserved"] = (
            float(identity["target_flip_rate"])
            <= thresholds.max_answer_identity_flip_rate
        )

    position_rows = {
        readout: index.get(
            (
                layer,
                arm,
                "position_descriptive",
                modality,
                source,
                target,
                readout,
            )
        )
        for readout in ("identity", "property")
    }
    position_diagnostic = {
        "blocking": False,
        "reason": (
            "the paper applies the swap at all positions; a final-token-only "
            "swap is reported to localize the effect, never used as a veto"
        ),
        "rows_present": {name: row is not None for name, row in position_rows.items()},
        "margin_gain_delta_vs_all_positions": {
            name: (
                float(primary[name]["mean_target_margin_change"])
                - float(row["mean_target_margin_change"])
                if row is not None
                else None
            )
            for name, row in position_rows.items()
        },
    }
    return {
        "layer": int(layer),
        "arm": str(arm),
        "condition": str(condition),
        "modality": str(modality),
        "source": str(source),
        "target": str(target),
        "n_images": int(identity["n_images"]),
        "identity_flip_rate": float(identity["target_flip_rate"]),
        "property_flip_rate": float(property_cell["target_flip_rate"]),
        "joint_flip_rate": joint_rate,
        "control_comparisons": control_comparisons,
        "position_diagnostic": position_diagnostic,
        "clauses": clauses,
        "failed_clauses": sorted(name for name, passed in clauses.items() if not passed),
        "passed": bool(clauses) and all(clauses.values()),
    }


def paper_onset_verdict_v2(
    cells: Sequence[Mapping],
    *,
    layers: Sequence[int],
    directed_pairs: Sequence[Mapping],
    modalities: Sequence[str],
    thresholds: PaperSwapV2Thresholds | None = None,
) -> dict:
    """Independent-layer onset comparison with α=1 primary and α=2 sensitivity.

    A layer/arm passes when at least one *same directed source→target pair*
    passes in every modality.  Both directions remain visible in
    ``direction_cells``; a weak reverse direction cannot erase a successful,
    predeclared trial, but different easy directions cannot be stitched
    together across modalities.
    """
    thresholds = thresholds or PaperSwapV2Thresholds()
    index = _cell_index(cells)
    layer_grid = tuple(int(layer) for layer in layers)
    conditions = ("swap_alpha1", "swap_alpha2")
    direction_cells = []
    layer_cells = []
    passing: dict[str, dict[str, list[int]]] = {
        condition: {"intermediate": [], "answer": []}
        for condition in conditions
    }
    pair_names = [
        f"{pair['source']}->{pair['target']}" for pair in directed_pairs
    ]
    passing_by_pair: dict[str, dict[str, dict[str, list[int]]]] = {
        condition: {
            pair_name: {"intermediate": [], "answer": []}
            for pair_name in pair_names
        }
        for condition in conditions
    }
    for condition in conditions:
        for arm in ("intermediate", "answer"):
            for layer in layer_grid:
                rows = []
                for pair in directed_pairs:
                    for modality in modalities:
                        row = _v2_direction_cell(
                            index,
                            layer=layer,
                            arm=arm,
                            condition=condition,
                            modality=str(modality),
                            source=str(pair["source"]),
                            target=str(pair["target"]),
                            thresholds=thresholds,
                        )
                        rows.append(row)
                        direction_cells.append(row)
                modality_pass = {
                    str(modality): any(
                        row["passed"] and row["modality"] == str(modality)
                        for row in rows
                    )
                    for modality in modalities
                }
                pair_pass = {
                    f"{pair['source']}->{pair['target']}": all(
                        any(
                            row["passed"]
                            and row["modality"] == str(modality)
                            and row["source"] == str(pair["source"])
                            and row["target"] == str(pair["target"])
                            for row in rows
                        )
                        for modality in modalities
                    )
                    for pair in directed_pairs
                }
                passed = bool(pair_pass) and any(pair_pass.values())
                layer_cells.append(
                    {
                        "condition": condition,
                        "arm": arm,
                        "layer": layer,
                        "passed": passed,
                        "modality_pass": modality_pass,
                        "same_direction_pair_pass": pair_pass,
                        "n_direction_cells_passing": sum(
                            1 for row in rows if row["passed"]
                        ),
                        "n_direction_cells": len(rows),
                    }
                )
                if passed:
                    passing[condition][arm].append(layer)
                for pair_name, pair_passed in pair_pass.items():
                    if pair_passed:
                        passing_by_pair[condition][pair_name][arm].append(layer)

    def onset(condition: str, arm: str) -> int | None:
        return min(passing[condition][arm], default=None)

    def pair_result(condition: str, pair_name: str) -> dict:
        arms = passing_by_pair[condition][pair_name]
        intermediate = min(arms["intermediate"], default=None)
        answer = min(arms["answer"], default=None)
        if intermediate is not None and answer is not None:
            if intermediate < answer:
                classification = "EARLIER_INTERMEDIATE"
            elif intermediate == answer:
                classification = "SAME_TESTED_LAYER"
            else:
                classification = "ANSWER_EARLIER"
        elif answer is not None:
            classification = "INTERMEDIATE_NULL_WITH_ANSWER_POSITIVE"
        elif intermediate is not None:
            classification = "ANSWER_NULL_WITH_INTERMEDIATE_POSITIVE"
        else:
            classification = "INCONCLUSIVE"
        return {
            "pair": pair_name,
            "intermediate_onset": intermediate,
            "answer_onset": answer,
            "classification": classification,
            "intermediate_passing_layers": list(arms["intermediate"]),
            "answer_passing_layers": list(arms["answer"]),
        }

    matched_pair_results = {
        condition: {
            pair_name: pair_result(condition, pair_name)
            for pair_name in pair_names
        }
        for condition in conditions
    }

    def any_pair(condition: str, classification: str) -> bool:
        return any(
            row["classification"] == classification
            for row in matched_pair_results[condition].values()
        )

    def condition_verdict(condition: str, *, sensitivity: bool) -> str:
        prefix = "PAPER_STYLE_ALPHA2_SENSITIVITY_" if sensitivity else "PAPER_STYLE_"
        if any_pair(condition, "EARLIER_INTERMEDIATE"):
            return prefix + "EARLIER_INTERMEDIATE"
        if any_pair(condition, "SAME_TESTED_LAYER"):
            return prefix + "SAME_TESTED_LAYER"
        if any_pair(condition, "ANSWER_EARLIER"):
            return prefix + "ANSWER_EARLIER"
        if any_pair(condition, "INTERMEDIATE_NULL_WITH_ANSWER_POSITIVE"):
            return prefix + "INTERMEDIATE_NULL_WITH_ANSWER_POSITIVE"
        if any_pair(condition, "ANSWER_NULL_WITH_INTERMEDIATE_POSITIVE"):
            return prefix + "ANSWER_NULL_WITH_INTERMEDIATE_POSITIVE"
        return prefix + "INCONCLUSIVE"

    primary_verdict = condition_verdict("swap_alpha1", sensitivity=False)
    sensitivity_verdict = condition_verdict("swap_alpha2", sensitivity=True)
    verdict = primary_verdict
    if primary_verdict == "PAPER_STYLE_INCONCLUSIVE":
        verdict = sensitivity_verdict
    if verdict == "PAPER_STYLE_EARLIER_INTERMEDIATE":
        verdict = "PAPER_STYLE_EARLIER_INTERMEDIATE_GO"

    primary_intermediate = onset("swap_alpha1", "intermediate")
    primary_answer = onset("swap_alpha1", "answer")
    sensitivity_intermediate = onset("swap_alpha2", "intermediate")
    sensitivity_answer = onset("swap_alpha2", "answer")

    payload = {
        "version": VERDICT_V2_VERSION,
        "verdict": verdict,
        "primary_verdict": primary_verdict,
        "alpha2_sensitivity_verdict": sensitivity_verdict,
        "primary_alpha": 1.0,
        "sensitivity_alpha": 2.0,
        "primary_onsets": {
            "intermediate": primary_intermediate,
            "answer": primary_answer,
        },
        "sensitivity_onsets": {
            "intermediate": sensitivity_intermediate,
            "answer": sensitivity_answer,
        },
        "tested_layers": list(layer_grid),
        "passing_layers": passing,
        "passing_layers_by_pair": passing_by_pair,
        "matched_pair_results": matched_pair_results,
        "cross_arm_direction_matching_required": True,
        "descriptive_unmatched_global_onsets": {
            "swap_alpha1": {
                "intermediate": primary_intermediate,
                "answer": primary_answer,
            },
            "swap_alpha2": {
                "intermediate": sensitivity_intermediate,
                "answer": sensitivity_answer,
            },
        },
        "layer_cells": layer_cells,
        "direction_cells": direction_cells,
        "thresholds": asdict(thresholds),
        "threshold_digest": thresholds.digest,
        "native_direct_readout_used": False,
        "source_derived_steering_used": False,
        "repeated_coordinate_exchange_used": False,
        "position_diagnostic_is_blocking": False,
        "causal_comparison": (
            "independent single-layer exact intermediate-coordinate exchange "
            "versus exact answer-coordinate exchange"
        ),
        "localization_scope": "four independently confirmed physical layers",
    }
    return {**payload, "verdict_digest": payload_checksum(payload)}


def amend_paper_v2_report_direction_matching(
    report: Mapping,
    *,
    expected_report_checksum: str,
) -> dict:
    """Recompute only the cross-arm direction join from a completed v2 report.

    The original report is checksum-verified and remains immutable.  No model,
    lens, media, prompt, or intervention unit is read or recomputed.
    """
    original = dict(report)
    stored_checksum = str(original.pop("report_checksum", ""))
    if stored_checksum != str(expected_report_checksum):
        raise PaperSwapRefused(
            "completed v2 report checksum does not match the pinned checksum"
        )
    if payload_checksum(original) != stored_checksum:
        raise PaperSwapRefused(
            "completed v2 report payload no longer matches its own checksum"
        )
    if original.get("schema") != "mmpilot.paper_reasoning_swap_report.v2":
        raise PaperSwapRefused(
            f"expected the completed v2 report schema, got {original.get('schema')!r}"
        )
    cells = original.get("cells_full")
    if not isinstance(cells, list) or not cells:
        raise PaperSwapRefused("completed v2 report carries no aggregated cells")
    threshold_payload = dict(original.get("thresholds") or {})
    if "blocking_controls" in threshold_payload:
        threshold_payload["blocking_controls"] = tuple(
            threshold_payload["blocking_controls"]
        )
    thresholds = PaperSwapV2Thresholds(**threshold_payload)
    if thresholds.digest != original.get("threshold_digest"):
        raise PaperSwapRefused("the completed v2 threshold digest does not match")
    recomputed = paper_onset_verdict_v2(
        cells,
        layers=original["tested_layers"],
        directed_pairs=original["directed_pairs"],
        modalities=original["capability_selection"]["modalities"],
        thresholds=thresholds,
    )
    amended = {
        **original,
        **recomputed,
        "schema": "mmpilot.paper_reasoning_swap_report.direction_matched.v3",
        "amendment_version": DIRECTION_MATCHED_AMENDMENT_VERSION,
        "original_report_checksum": stored_checksum,
        "original_verdict": original.get("verdict"),
        "original_primary_onsets": original.get("primary_onsets"),
        "original_sensitivity_onsets": original.get("sensitivity_onsets"),
        "source_cells_digest": payload_checksum(cells),
        "scientific_units_recomputed": 0,
        "model_loaded": False,
        "amendment_scope": (
            "reporting only: require the same directed source-to-target pair "
            "for intermediate and answer onset comparisons"
        ),
    }
    amended["report_checksum"] = payload_checksum(amended)
    return amended


def paper_alpha2_confirmation_verdict(
    result: Mapping,
    *,
    source: str = "bird",
    target: str = "cat",
    expected_intermediate_layer: int = 32,
) -> dict:
    """Predeclared verdict for a fresh, direction-matched alpha=2 replication.

    This deliberately does not promote the exploratory v2 population.  It
    consumes a normal :func:`paper_onset_verdict_v2` result from a separately
    fingerprinted population and asks whether the exact same directed pair has
    an alpha=2 intermediate onset at the predeclared early layer and a strictly
    later answer onset.  Alpha=1 remains descriptive dose sensitivity.
    """
    pair_name = f"{source}->{target}"
    matched = dict(result.get("matched_pair_results") or {})
    alpha2 = dict((matched.get("swap_alpha2") or {}).get(pair_name) or {})
    alpha1 = dict((matched.get("swap_alpha1") or {}).get(pair_name) or {})
    intermediate = alpha2.get("intermediate_onset")
    answer = alpha2.get("answer_onset")
    clauses = {
        "direction_is_predeclared": list(
            (result.get("matched_pair_results") or {})
            .get("swap_alpha2", {})
            .keys()
        ) == [pair_name],
        "alpha2_intermediate_onset_replicates": (
            intermediate == int(expected_intermediate_layer)
        ),
        "alpha2_answer_onset_is_later": (
            answer is not None
            and intermediate is not None
            and int(answer) > int(intermediate)
        ),
        "same_direction_used_across_arms": (
            alpha2.get("pair") == pair_name
            and bool(result.get("cross_arm_direction_matching_required"))
        ),
    }
    passed = all(clauses.values())
    payload = {
        "version": ALPHA2_CONFIRMATION_VERSION,
        "verdict": (
            "PAPER_STYLE_ALPHA2_INDEPENDENT_CONFIRMATION_GO"
            if passed
            else "PAPER_STYLE_ALPHA2_INDEPENDENT_CONFIRMATION_NO_GO"
        ),
        "pair": pair_name,
        "alpha2_is_primary": True,
        "alpha2_intermediate_onset": intermediate,
        "alpha2_answer_onset": answer,
        "alpha2_pair_result": alpha2,
        "alpha1_secondary_pair_result": alpha1,
        "expected_intermediate_layer": int(expected_intermediate_layer),
        "clauses": clauses,
        "failed_clauses": sorted(name for name, ok in clauses.items() if not ok),
        "interpretation": (
            "GO independently replicates an alpha=2 intermediate-coordinate "
            "effect at the predeclared early layer before the same directed "
            "pair's answer-coordinate effect. Alpha=2 is an amplified "
            "extrapolation and is labelled as such."
        ),
    }
    return {**payload, "verdict_digest": payload_checksum(payload)}


__all__ = [
    "PAPER_REASONING_SWAP_VERSION",
    "POPULATION_VERSION",
    "SAMPLED_BAND_VERSION",
    "VERDICT_VERSION",
    "CAPABILITY_SELECTION_VERSION",
    "INDEPENDENT_LAYER_VERSION",
    "PAPER_REASONING_SWAP_V2_VERSION",
    "VERDICT_V2_VERSION",
    "DIRECTION_MATCHED_AMENDMENT_VERSION",
    "ALPHA2_CONFIRMATION_VERSION",
    "PaperSwapRefused",
    "PaperSwapThresholds",
    "PaperSwapV2Thresholds",
    "hidden_animal_population",
    "independent_layer_record",
    "paper_onset_verdict",
    "paper_onset_verdict_v2",
    "amend_paper_v2_report_direction_matching",
    "paper_alpha2_confirmation_verdict",
    "sampled_band_record",
    "sampled_suffix_bands",
    "select_capability_eligible_samples",
    "summarize_cells",
]
