# SPDX-License-Identifier: Apache-2.0
"""Prospective multimodal country-fact coordinate-swap benchmark.

The existing SpokenCOCO population cannot answer a country-identity question:
its ontology is COCO objects and its audio is a reading of COCO captions.  This
module defines a separate, deliberately small benchmark in which the evidence
identifies a country and the requested answer is a memorized fact about that
country.  Images, text, and spoken audio therefore differ only in how the
identity is supplied; the answer is never present in the evidence.

Everything here is pure bookkeeping and aggregation.  Dataset download, audio
rendering, model forwards, lens fitting, and interventions live in the Colab
notebook, while the rules that decide what those results mean stay testable.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from jlens.mmpilot.multimodal_lens import holm_adjust, paired_binary_one_sided_p
from jlens.mmpilot.store import payload_checksum

PROTOCOL_VERSION = "mmpilot.country_workspace_generalization.v1"
CAPABILITY_GATE_VERSION = "mmpilot.country_direction_capability.v2"
COUNTRY_COMPLETION_INSTRUCTION = (
    "Use the supplied evidence to identify the country, then use ordinary "
    "factual knowledge to complete the assistant's sentence directly. "
    "Do not restart, explain, or list choices."
)
COUNTRY_IDENTITY_LENS_VALIDATION_VERSION = (
    "mmpilot.country_identity_lens_validation.v1"
)
COUNTRY_IDENTITY_BAND_CALIBRATION_VERSION = (
    "mmpilot.country_identity_band_calibration.v1"
)
DATASET_ID = "tokeron/country-flags-variations"
DATASET_REVISION = "1ea3cce246ab44f0fe8ecb526ad759ea11d28465"
DATASET_CARD = "https://huggingface.co/datasets/tokeron/country-flags-variations"
MODALITIES = ("text", "image", "spoken_audio")
PROPERTIES = ("capital", "continent")
LAYERS = tuple(range(16, 41))
TARGET_LAYER = 41
ALPHA = 1.0
POSITION_RULE = "all_prompt_positions"
FIT_COUNTRIES = (
    "Argentina",
    "Australia",
    "Canada",
    "Germany FRG",
    "India",
    "Italy",
    "Mexico",
    "Norway",
    "Spain",
    "Sweden",
    "USA",
)
EVAL_COUNTRIES = ("France", "China", "Japan", "Egypt")
CONTROL_COUNTRIES = ("Canada", "Italy")
DIRECTIONS = (
    ("France", "China"),
    ("China", "France"),
    ("Japan", "Egypt"),
    ("Egypt", "Japan"),
)
FACTS: dict[str, dict[str, str]] = {
    "France": {"capital": "Paris", "continent": "Europe"},
    "China": {"capital": "Beijing", "continent": "Asia"},
    "Japan": {"capital": "Tokyo", "continent": "Asia"},
    "Egypt": {"capital": "Cairo", "continent": "Africa"},
}
ANSWER_ALIASES: dict[str, tuple[str, ...]] = {
    "Paris": ("Paris",),
    "Beijing": ("Beijing",),
    "Tokyo": ("Tokyo",),
    "Cairo": ("Cairo",),
    "Europe": ("Europe",),
    "Asia": ("Asia",),
    "Africa": ("Africa",),
}
PATH_BANDS = (
    tuple(range(16, 24)),
    tuple(range(20, 28)),
    tuple(range(24, 32)),
    tuple(range(28, 36)),
    tuple(range(33, 41)),
    tuple(range(16, 41)),
)
N_FIT_PER_COUNTRY = 3
N_DEVELOPMENT_PER_COUNTRY = 4
N_CONFIRMATION_PER_COUNTRY = 14
N_LOCALIZATION_PER_COUNTRY = 4
MIN_CAPABILITY_RATE = 0.75
MIN_DIRECT_ANSWER_RATE = 0.75
MIN_SWAP_RATE = 0.75
MIN_CONTROL_MARGIN = 0.25
FAMILYWISE_ALPHA = 0.05
MIN_IDENTITY_LENS_TOP1_RATE = 0.75
MIN_IDENTITY_SWAP_RATE = 0.75


class CountryWorkspaceRefused(RuntimeError):
    """A prospective design or stored result violates the frozen protocol."""


@dataclass(frozen=True)
class CountryMediaRow:
    unit_id: str
    country: str
    source_split: str
    source_index: int
    source_seed: str
    image_path: str
    image_checksum: str
    audio_path: str
    audio_checksum: str
    speech_text: str
    speech_voice: str
    speech_speed: int
    speech_pitch: int
    study_split: str
    ocr_text: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def normalize_surface(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value)).casefold()
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


def answer_matches(generated: str, expected: str) -> bool:
    """Match a complete unrestricted answer before the first control token."""

    observed = normalize_surface(str(generated).split("<", 1)[0])
    aliases = ANSWER_ALIASES.get(str(expected), (str(expected),))
    for alias in aliases:
        wanted = normalize_surface(alias)
        if observed == wanted or observed.startswith(f"{wanted} "):
            return True
    return False


def identity_matches(generated: str, country: str) -> bool:
    observed = normalize_surface(str(generated).split("<", 1)[0])
    wanted = normalize_surface(country.replace(" FRG", ""))
    return bool(observed == wanted or observed.startswith(f"{wanted} "))


def text_evidence(country: str) -> str:
    return f"Country evidence: {country.replace(' FRG', '')}."


def speech_evidence(country: str) -> str:
    return f"The country in this recording is {country.replace(' FRG', '')}."


def assistant_prefill(property_name: str) -> str:
    if property_name == "identity":
        return "The country in the evidence is"
    if property_name == "capital":
        return "The capital of the country in the evidence is"
    if property_name == "continent":
        return "The continent containing the country in the evidence is"
    raise CountryWorkspaceRefused(f"unknown property {property_name!r}")


def fact(country: str, property_name: str) -> str:
    try:
        return FACTS[str(country)][str(property_name)]
    except KeyError as exc:
        raise CountryWorkspaceRefused(
            f"no frozen {property_name!r} fact for {country!r}"
        ) from exc


def benchmark_spec(*, dataset_revision: str) -> dict:
    payload = {
        "version": PROTOCOL_VERSION,
        "dataset_id": DATASET_ID,
        "dataset_revision": str(dataset_revision),
        "dataset_card": DATASET_CARD,
        "dataset_role": (
            "synthetic but image-verified flag variations; no country name or "
            "downstream answer is supplied to the model in the image condition"
        ),
        "audio_role": (
            "deterministic synthetic spoken country labels rendered as 16 kHz "
            "mono float32 WAV; the transcript is provenance only"
        ),
        "fit_countries": list(FIT_COUNTRIES),
        "evaluation_countries": list(EVAL_COUNTRIES),
        "control_countries": list(CONTROL_COUNTRIES),
        "directions": [list(pair) for pair in DIRECTIONS],
        "facts": FACTS,
        "properties": list(PROPERTIES),
        "modalities": list(MODALITIES),
        "layers": list(LAYERS),
        "target_layer": TARGET_LAYER,
        "alpha": ALPHA,
        "alpha_is_exact_exchange": True,
        "position_rule": POSITION_RULE,
        "path_bands": [list(band) for band in PATH_BANDS],
        "fit_per_country": N_FIT_PER_COUNTRY,
        "development_per_country": N_DEVELOPMENT_PER_COUNTRY,
        "confirmation_per_country": N_CONFIRMATION_PER_COUNTRY,
        "localization_per_country": N_LOCALIZATION_PER_COUNTRY,
        "minimum_capability_rate": MIN_CAPABILITY_RATE,
        "minimum_direct_answer_rate": MIN_DIRECT_ANSWER_RATE,
        "minimum_swap_rate": MIN_SWAP_RATE,
        "minimum_control_margin": MIN_CONTROL_MARGIN,
        "familywise_alpha": FAMILYWISE_ALPHA,
        "multiplicity_rule": (
            "within each direction-property-modality cell, compare exact "
            "success to the per-unit maximum of zero, random, and unrelated; "
            "Holm-correct the resulting 24 predeclared cell hypotheses"
        ),
        "selection_rule": (
            "choose one band per property using only clean capability and the "
            "norm-matched direct-answer positive control; exact identity-swap "
            "outcomes are unavailable during path selection"
        ),
        "endpoint": "unrestricted greedy complete answer",
        "teacher_forcing": False,
        "candidate_list": False,
    }
    return {**payload, "protocol_digest": payload_checksum(payload)}


def validate_media_plan(rows: Sequence[Mapping[str, Any]]) -> dict:
    """Prove the prepared fit/development/confirmation population is usable."""

    records = [dict(row) for row in rows]
    problems: list[str] = []
    by_country_split: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in records:
        by_country_split[(str(row.get("country")), str(row.get("study_split")))].append(
            row
        )
        for field in (
            "unit_id",
            "country",
            "image_path",
            "image_checksum",
            "audio_path",
            "audio_checksum",
            "study_split",
        ):
            if not str(row.get(field) or "").strip():
                problems.append(f"row {row.get('unit_id')!r} has no {field}")

    for country in FIT_COUNTRIES:
        observed = len(by_country_split[(country, "fit")])
        if observed != N_FIT_PER_COUNTRY:
            problems.append(
                f"fit country {country!r} has {observed}, not {N_FIT_PER_COUNTRY}"
            )
    for country in EVAL_COUNTRIES:
        for split, expected in (
            ("development", N_DEVELOPMENT_PER_COUNTRY),
            ("confirmation", N_CONFIRMATION_PER_COUNTRY),
        ):
            observed = len(by_country_split[(country, split)])
            if observed != expected:
                problems.append(
                    f"{split} country {country!r} has {observed}, not {expected}"
                )

    unit_ids = [str(row.get("unit_id")) for row in records]
    image_checksums = [str(row.get("image_checksum")) for row in records]
    audio_checksums = [str(row.get("audio_checksum")) for row in records]
    if len(set(unit_ids)) != len(unit_ids):
        problems.append("unit ids are not unique")
    if len(set(image_checksums)) != len(image_checksums):
        problems.append("image checksums are not unique across study units")
    if len(set(audio_checksums)) != len(audio_checksums):
        problems.append("audio checksums are not unique across study units")

    split_images = {
        split: {
            str(row.get("image_checksum"))
            for row in records
            if row.get("study_split") == split
        }
        for split in ("fit", "development", "confirmation")
    }
    overlaps = {
        f"{left}_vs_{right}": sorted(split_images[left] & split_images[right])
        for left, right in (
            ("fit", "development"),
            ("fit", "confirmation"),
            ("development", "confirmation"),
        )
    }
    if any(overlaps.values()):
        problems.append("image content overlaps between study splits")

    # OCR is allowed to see confirmation images because it is an outcome-blind
    # media audit. Any recognized country/fact label is a refusal, not a row
    # silently removed after model results exist.
    forbidden = {
        normalize_surface(value)
        for value in (*EVAL_COUNTRIES, *FIT_COUNTRIES)
    } | {
        normalize_surface(value)
        for mapping in FACTS.values()
        for value in mapping.values()
    }
    ocr_hits = []
    for row in records:
        words = f" {normalize_surface(str(row.get('ocr_text') or ''))} "
        hits = sorted(
            token for token in forbidden if token and f" {token} " in words
        )
        if hits:
            ocr_hits.append({"unit_id": row.get("unit_id"), "hits": hits})
    if ocr_hits:
        problems.append("OCR found a country or downstream answer in image evidence")

    body = {
        "version": PROTOCOL_VERSION,
        "n_rows": len(records),
        "counts": {
            f"{country}:{split}": len(group)
            for (country, split), group in sorted(by_country_split.items())
        },
        "split_image_overlaps": overlaps,
        "ocr_hits": ocr_hits,
        "problems": problems,
        "passed": not problems,
        "population_digest": payload_checksum(records),
    }
    return {**body, "validation_checksum": payload_checksum(body)}


def _capability_cell(rows: Sequence[Mapping], country: str, property_name: str, modality: str) -> dict:
    selected = [
        row
        for row in rows
        if row.get("country") == country
        and row.get("property") == property_name
        and row.get("modality") == modality
    ]
    successes = sum(bool(row.get("success")) for row in selected)
    return {
        "country": country,
        "property": property_name,
        "modality": modality,
        "n": len(selected),
        "successes": successes,
        "rate": successes / len(selected) if selected else 0.0,
        "complete": len(selected) == N_DEVELOPMENT_PER_COUNTRY,
        "passed": len(selected) == N_DEVELOPMENT_PER_COUNTRY
        and successes / len(selected) >= MIN_CAPABILITY_RATE,
    }


def capability_report(rows: Sequence[Mapping]) -> dict:
    selected = [dict(row) for row in rows]
    cells = [
        _capability_cell(selected, country, property_name, modality)
        for country in EVAL_COUNTRIES
        for property_name in ("identity", *PROPERTIES)
        for modality in MODALITIES
    ]
    eligible = []
    for country in EVAL_COUNTRIES:
        country_cells = [cell for cell in cells if cell["country"] == country]
        if country_cells and all(cell["passed"] for cell in country_cells):
            eligible.append(country)
    eligible_directions = [
        f"{source}->{target}"
        for source, target in DIRECTIONS
        if source in eligible
    ]
    eligible_pairs = {
        tuple(sorted(direction.split("->"))) for direction in eligible_directions
    }
    generalization_ready = len(eligible_pairs) >= 2
    body = {
        "version": PROTOCOL_VERSION,
        "capability_gate_version": CAPABILITY_GATE_VERSION,
        "verdict": (
            "COUNTRY_CAPABILITY_GENERALIZATION_GO"
            if generalization_ready
            else "COUNTRY_CAPABILITY_PARTIAL_GO"
            if eligible_directions
            else "COUNTRY_CAPABILITY_NO_GO"
        ),
        "eligible_countries": eligible,
        "eligible_directions": eligible_directions,
        "n_eligible_pairs": len(eligible_pairs),
        "generalization_ready": generalization_ready,
        "gate_rationale": (
            "Clean capability is required for a direction's source evidence. "
            "A target need not be cleanly recognized as input because it is "
            "never supplied as evidence in that direction."
        ),
        "cells": cells,
        "rows": selected,
    }
    return {**body, "report_checksum": payload_checksum(body)}


def _path_id(band: Sequence[int]) -> str:
    return f"L{int(band[0])}-L{int(band[-1])}"


def direct_answer_localization_report(rows: Sequence[Mapping]) -> dict:
    """Select one path per property without reading exact-swap outcomes."""

    selected = [dict(row) for row in rows]
    if any(str(row.get("condition")) != "direct_answer" for row in selected):
        raise CountryWorkspaceRefused(
            "path selection received a non-direct-answer outcome"
        )
    frozen_direction_names = {
        f"{source}->{target}" for source, target in DIRECTIONS
    }
    observed_directions = sorted(
        {str(row.get("direction")) for row in selected}
    )
    if not observed_directions or not set(observed_directions) <= frozen_direction_names:
        raise CountryWorkspaceRefused(
            f"path selection received invalid directions {observed_directions}"
        )
    candidates = []
    for property_name in PROPERTIES:
        for band in PATH_BANDS:
            cells = []
            for source, target in DIRECTIONS:
                if f"{source}->{target}" not in observed_directions:
                    continue
                for modality in MODALITIES:
                    subset = [
                        row
                        for row in selected
                        if row.get("property") == property_name
                        and row.get("target") == target
                        and row.get("modality") == modality
                        and list(map(int, row.get("layers_patched") or ()))
                        == list(band)
                    ]
                    successes = sum(bool(row.get("success")) for row in subset)
                    integrity = bool(subset) and all(
                        bool(row.get("integrity_pass")) for row in subset
                    )
                    cells.append(
                        {
                            "target": target,
                            "modality": modality,
                            "n": len(subset),
                            "successes": successes,
                            "rate": successes / len(subset) if subset else 0.0,
                            "integrity_pass": integrity,
                        }
                    )
            minimum = min((cell["rate"] for cell in cells), default=0.0)
            complete = all(
                cell["n"] == N_LOCALIZATION_PER_COUNTRY for cell in cells
            )
            passed = bool(cells) and complete and all(
                cell["rate"] >= MIN_DIRECT_ANSWER_RATE
                and cell["integrity_pass"]
                for cell in cells
            )
            candidates.append(
                {
                    "property": property_name,
                    "path_id": _path_id(band),
                    "band": list(band),
                    "minimum_cell_rate": minimum,
                    "mean_rate": (
                        sum(cell["rate"] for cell in cells) / len(cells)
                        if cells
                        else 0.0
                    ),
                    "complete": complete,
                    "passed": passed,
                    "cells": cells,
                }
            )

    chosen: dict[str, dict] = {}
    for property_name in PROPERTIES:
        eligible = [
            row
            for row in candidates
            if row["property"] == property_name and row["passed"]
        ]
        eligible.sort(
            key=lambda row: (
                -float(row["minimum_cell_rate"]),
                -float(row["mean_rate"]),
                len(row["band"]),
                row["band"],
            )
        )
        if eligible:
            chosen[property_name] = {
                key: eligible[0][key]
                for key in ("path_id", "band", "minimum_cell_rate", "mean_rate")
            }
    body = {
        "version": PROTOCOL_VERSION,
        "verdict": (
            "COUNTRY_DIRECT_PATHS_GO"
            if set(chosen) == set(PROPERTIES)
            else "COUNTRY_DIRECT_PATHS_NO_GO"
        ),
        "selection_used_exact_swap_outcomes": False,
        "eligible_directions": observed_directions,
        "n_paths_per_property": len(PATH_BANDS),
        "selected_paths": chosen,
        "candidates": candidates,
        "rows": selected,
    }
    return {**body, "report_checksum": payload_checksum(body)}


def identity_lens_validation_report(
    rows: Sequence[Mapping],
    *,
    expected_n_per_modality: int,
) -> dict:
    """Validate whether a lens identifies the clean source country.

    This diagnostic ranks the ground-truth source against the fixed evaluation
    and unrelated-control country tokens. It never supplies candidates to the
    model's generated output. A layer is eligible only when the source is the
    sole maximum on at least 75% of examples in every modality.
    """

    records = [dict(row) for row in rows]
    cells = []
    eligible_layers = []
    observed_layers = sorted({int(row["layer"]) for row in records})
    for layer in observed_layers:
        layer_cells = []
        for modality in MODALITIES:
            selected = [
                row
                for row in records
                if int(row.get("layer", -1)) == layer
                and row.get("modality") == modality
            ]
            top1 = sum(bool(row.get("source_is_sole_top1")) for row in selected)
            complete = len(selected) == int(expected_n_per_modality)
            finite = bool(selected) and all(bool(row.get("all_finite")) for row in selected)
            passed = (
                complete
                and finite
                and top1 / len(selected) >= MIN_IDENTITY_LENS_TOP1_RATE
            )
            cell = {
                "layer": layer,
                "modality": modality,
                "n": len(selected),
                "sole_top1": top1,
                "top1_rate": top1 / len(selected) if selected else 0.0,
                "complete": complete,
                "all_finite": finite,
                "passed": passed,
            }
            cells.append(cell)
            layer_cells.append(cell)
        if len(layer_cells) == len(MODALITIES) and all(
            cell["passed"] for cell in layer_cells
        ):
            eligible_layers.append(layer)
    body = {
        "version": COUNTRY_IDENTITY_LENS_VALIDATION_VERSION,
        "verdict": (
            "COUNTRY_IDENTITY_LENS_VALIDATION_GO"
            if eligible_layers
            else "COUNTRY_IDENTITY_LENS_VALIDATION_NO_GO"
        ),
        "candidate_universe": [*EVAL_COUNTRIES, *CONTROL_COUNTRIES],
        "generated_output_used_for_selection": False,
        "minimum_top1_rate": MIN_IDENTITY_LENS_TOP1_RATE,
        "eligible_layers": eligible_layers,
        "cells": cells,
        "rows": records,
    }
    return {**body, "report_checksum": payload_checksum(body)}


def identity_band_calibration_report(
    rows: Sequence[Mapping],
    *,
    eligible_directions: Sequence[str],
    expected_n: int,
) -> dict:
    """Choose a band using identity swaps, never downstream-fact outcomes."""

    records = [dict(row) for row in rows]
    allowed = set(map(str, eligible_directions))
    candidates = []
    for band in PATH_BANDS:
        direction_cells = []
        passing_directions = []
        for source, target in DIRECTIONS:
            direction = f"{source}->{target}"
            if direction not in allowed:
                continue
            modality_cells = []
            for modality in MODALITIES:
                subset = [
                    row
                    for row in records
                    if row.get("direction") == direction
                    and row.get("modality") == modality
                    and tuple(map(int, row.get("layers_patched") or ())) == band
                ]
                conditions = {
                    condition: _condition_summary(subset, condition)
                    for condition in ("exact", "zero", "random", "unrelated")
                }
                exact = conditions["exact"]
                margins = {
                    control: exact["rate"] - conditions[control]["rate"]
                    for control in ("zero", "random", "unrelated")
                }
                complete = all(
                    condition["n"] == int(expected_n)
                    for condition in conditions.values()
                )
                passed = (
                    complete
                    and exact["rate"] >= MIN_IDENTITY_SWAP_RATE
                    and exact["integrity_pass"]
                    and all(
                        conditions[name]["integrity_pass"]
                        for name in ("zero", "random", "unrelated")
                    )
                    and min(margins.values()) >= MIN_CONTROL_MARGIN
                )
                cell = {
                    "direction": direction,
                    "modality": modality,
                    "conditions": conditions,
                    "margins": margins,
                    "complete": complete,
                    "passed": passed,
                }
                modality_cells.append(cell)
                direction_cells.append(cell)
            if len(modality_cells) == len(MODALITIES) and all(
                cell["passed"] for cell in modality_cells
            ):
                passing_directions.append(direction)
        unordered_pairs = {
            tuple(sorted(direction.split("->", 1)))
            for direction in passing_directions
        }
        exact_rates = [
            cell["conditions"]["exact"]["rate"] for cell in direction_cells
        ]
        candidates.append(
            {
                "path_id": f"L{band[0]}-{band[-1]}",
                "band": list(band),
                "passing_directions": passing_directions,
                "n_distinct_pairs": len(unordered_pairs),
                "generalized_across_two_pairs": len(unordered_pairs) >= 2,
                "minimum_exact_rate": min(exact_rates, default=0.0),
                "mean_exact_rate": (
                    sum(exact_rates) / len(exact_rates) if exact_rates else 0.0
                ),
                "cells": direction_cells,
            }
        )
    eligible = [
        candidate
        for candidate in candidates
        if candidate["generalized_across_two_pairs"]
    ]
    eligible.sort(
        key=lambda candidate: (
            -float(candidate["minimum_exact_rate"]),
            -float(candidate["mean_exact_rate"]),
            len(candidate["band"]),
            candidate["band"],
        )
    )
    selected = None
    if eligible:
        selected = {
            key: eligible[0][key]
            for key in (
                "path_id",
                "band",
                "passing_directions",
                "n_distinct_pairs",
                "minimum_exact_rate",
                "mean_exact_rate",
            )
        }
    body = {
        "version": COUNTRY_IDENTITY_BAND_CALIBRATION_VERSION,
        "verdict": (
            "COUNTRY_IDENTITY_BAND_CALIBRATION_GO"
            if selected is not None
            else "COUNTRY_IDENTITY_BAND_CALIBRATION_NO_GO"
        ),
        "selection_used_downstream_fact_outcomes": False,
        "eligible_directions": sorted(allowed),
        "minimum_swap_rate": MIN_IDENTITY_SWAP_RATE,
        "minimum_control_margin": MIN_CONTROL_MARGIN,
        "selected": selected,
        "candidates": candidates,
        "rows": records,
    }
    return {**body, "report_checksum": payload_checksum(body)}


def _condition_summary(rows: Sequence[Mapping], condition: str) -> dict:
    selected = [row for row in rows if row.get("condition") == condition]
    successes = sum(bool(row.get("success")) for row in selected)
    return {
        "n": len(selected),
        "successes": successes,
        "rate": successes / len(selected) if selected else 0.0,
        "integrity_pass": bool(selected)
        and all(bool(row.get("integrity_pass")) for row in selected),
    }


def causal_report(
    rows: Sequence[Mapping],
    *,
    stage: str,
    expected_n: int,
    frozen_directions: Sequence[str] | None = None,
) -> dict:
    """Aggregate development or confirmation exact-swap trials."""

    if stage not in {"development", "confirmation"}:
        raise CountryWorkspaceRefused(f"unknown causal stage {stage!r}")
    records = [dict(row) for row in rows]
    allowed = None if frozen_directions is None else set(map(str, frozen_directions))
    cells = []
    comparisons = []
    for source, target in DIRECTIONS:
        direction = f"{source}->{target}"
        if allowed is not None and direction not in allowed:
            continue
        for property_name in PROPERTIES:
            for modality in MODALITIES:
                subset = [
                    row
                    for row in records
                    if row.get("direction") == direction
                    and row.get("property") == property_name
                    and row.get("modality") == modality
                ]
                conditions = {
                    condition: _condition_summary(subset, condition)
                    for condition in ("exact", "zero", "random", "unrelated")
                }
                exact = conditions["exact"]
                margins = {
                    control: exact["rate"] - conditions[control]["rate"]
                    for control in ("zero", "random", "unrelated")
                }
                complete = all(
                    row["n"] == expected_n for row in conditions.values()
                )
                passed = (
                    complete
                    and exact["rate"] >= MIN_SWAP_RATE
                    and exact["integrity_pass"]
                    and all(
                        conditions[name]["integrity_pass"]
                        for name in ("zero", "random", "unrelated")
                    )
                    and min(margins.values()) >= MIN_CONTROL_MARGIN
                )
                cells.append(
                    {
                        "direction": direction,
                        "property": property_name,
                        "modality": modality,
                        "conditions": conditions,
                        "margins": margins,
                        "complete": complete,
                        "passed_before_multiplicity": passed,
                    }
                )
                if stage == "confirmation" and complete:
                    exact_values = [
                        bool(row.get("success"))
                        for row in subset
                        if row.get("condition") == "exact"
                    ]
                    exact_by_unit = {
                        str(row.get("unit_id")): bool(row.get("success"))
                        for row in subset
                        if row.get("condition") == "exact"
                    }
                    controls_by_name = {
                        control: {
                            str(row.get("unit_id")): bool(row.get("success"))
                            for row in subset
                            if row.get("condition") == control
                        }
                        for control in ("zero", "random", "unrelated")
                    }
                    ordered_ids = sorted(exact_by_unit)
                    if ordered_ids and all(
                        set(ordered_ids) == set(control_rows)
                        for control_rows in controls_by_name.values()
                    ):
                        any_control = [
                            any(
                                controls_by_name[control][key]
                                for control in ("zero", "random", "unrelated")
                            )
                            for key in ordered_ids
                        ]
                        test = paired_binary_one_sided_p(
                            [exact_by_unit[key] for key in ordered_ids],
                            any_control,
                        )
                        comparisons.append(
                            {
                                **test,
                                "cell": f"{direction}:{property_name}:{modality}",
                                "control": "any_negative_control",
                                "n_exact": len(exact_values),
                            }
                        )

    adjusted = holm_adjust(comparisons) if comparisons else []
    significant_cells = {
        row["cell"]
        for row in adjusted
        if float(row["holm_adjusted_p"]) <= FAMILYWISE_ALPHA
    }
    for cell in cells:
        name = f"{cell['direction']}:{cell['property']}:{cell['modality']}"
        cell_comparisons = [row for row in adjusted if row["cell"] == name]
        cell["paired_comparisons"] = cell_comparisons
        cell["passed"] = bool(cell["passed_before_multiplicity"]) and (
            stage == "development"
            or (
                len(cell_comparisons) == 1
                and name in significant_cells
                and all(
                    float(row["holm_adjusted_p"]) <= FAMILYWISE_ALPHA
                    for row in cell_comparisons
                )
            )
        )

    passing_direction_properties = []
    for source, target in DIRECTIONS:
        direction = f"{source}->{target}"
        for property_name in PROPERTIES:
            group = [
                cell
                for cell in cells
                if cell["direction"] == direction
                and cell["property"] == property_name
            ]
            if len(group) == len(MODALITIES) and all(cell["passed"] for cell in group):
                passing_direction_properties.append(f"{direction}:{property_name}")

    passing_directions = []
    for source, target in DIRECTIONS:
        direction = f"{source}->{target}"
        if all(
            f"{direction}:{property_name}" in passing_direction_properties
            for property_name in PROPERTIES
        ):
            passing_directions.append(direction)

    unordered_pairs = {
        tuple(sorted(direction.split("->"))) for direction in passing_directions
    }
    bidirectional_pairs = []
    for left, right in sorted(unordered_pairs):
        if f"{left}->{right}" in passing_directions and f"{right}->{left}" in passing_directions:
            bidirectional_pairs.append(f"{left}<->{right}")
    generalized = len(unordered_pairs) >= 2
    bidirectional = bool(bidirectional_pairs)
    full_grid = set(passing_directions) == {
        f"{source}->{target}" for source, target in DIRECTIONS
    }

    prefix = "COUNTRY_DEVELOPMENT" if stage == "development" else "COUNTRY_CONFIRMATION"
    verdict = (
        f"{prefix}_FULL_GRID_GO"
        if full_grid
        else f"{prefix}_GENERALIZED_AND_BIDIRECTIONAL_GO"
        if generalized and bidirectional
        else f"{prefix}_GENERALIZED_GO"
        if generalized
        else f"{prefix}_BIDIRECTIONAL_GO"
        if bidirectional
        else f"{prefix}_PARTIAL_GO"
        if passing_directions
        else f"{prefix}_NO_GO"
    )
    body = {
        "version": PROTOCOL_VERSION,
        "stage": stage,
        "verdict": verdict,
        "passing_direction_properties": passing_direction_properties,
        "passing_directions_both_properties": passing_directions,
        "n_distinct_pairs_both_properties": len(unordered_pairs),
        "bidirectional_pairs_both_properties": bidirectional_pairs,
        "generalized_across_two_pairs": generalized,
        "bidirectional_on_at_least_one_pair": bidirectional,
        "full_predeclared_grid_passed": full_grid,
        "cells": cells,
        "paired_comparisons": adjusted,
        "rows": records,
        "claim_boundary": (
            "A confirmation GO supports only the directions, properties, "
            "modalities, flag-image distribution, and synthetic spoken-country "
            "distribution measured here. No finite benchmark establishes "
            "universal coordinate-swap generalization."
        ),
    }
    return {**body, "report_checksum": payload_checksum(body)}


def france_china_followup_report(
    rows: Sequence[Mapping],
    *,
    stage: str,
    expected_n: int,
) -> dict:
    """Score the prospective France-to-China downstream-fact follow-up."""

    if stage not in {"development", "confirmation"}:
        raise CountryWorkspaceRefused(f"unknown follow-up stage {stage!r}")
    records = [dict(row) for row in rows]
    if any(
        row.get("direction") != "France->China"
        or row.get("property") not in PROPERTIES
        or row.get("modality") not in MODALITIES
        for row in records
    ):
        raise CountryWorkspaceRefused(
            "the France-to-China follow-up contains an unfrozen direction, "
            "property, or modality"
        )
    base = causal_report(
        records,
        stage=stage,
        expected_n=expected_n,
        frozen_directions=("France->China",),
    )
    passed = set(base["passing_direction_properties"])
    required = {f"France->China:{property_name}" for property_name in PROPERTIES}
    all_cells_pass = required <= passed
    prefix = "COUNTRY_FRANCE_CHINA_" + stage.upper()
    body = {
        **{key: value for key, value in base.items() if key != "report_checksum"},
        "version": f"{PROTOCOL_VERSION}.france_china_followup.v1",
        "verdict": f"{prefix}_{'GO' if all_cells_pass else 'NO_GO'}",
        "frozen_direction": "France->China",
        "frozen_properties": list(PROPERTIES),
        "frozen_modalities": list(MODALITIES),
        "frozen_alpha": ALPHA,
        "frozen_band": list(LAYERS),
        "all_properties_every_modality_passed": all_cells_pass,
        "parent_two_pair_verdict_unchanged": True,
        "claim_boundary": (
            "This prospective single-direction follow-up can support only "
            "France-to-China downstream recomputation of capital and continent "
            "in the measured modalities. It does not convert the parent "
            "two-pair country-generalization NO_GO into a GO."
        ),
    }
    return {**body, "report_checksum": payload_checksum(body)}


def freeze_france_china_followup_design(
    *,
    protocol: Mapping,
    media_validation: Mapping,
    capability: Mapping,
    lens_validation: Mapping,
    identity_calibration: Mapping,
    lens_checksum: str,
    development: Mapping,
) -> dict:
    """Freeze fresh confirmation after the prospective development stage."""

    if not bool(media_validation.get("passed")):
        raise CountryWorkspaceRefused("media validation did not pass")
    if lens_validation.get("verdict") != "COUNTRY_IDENTITY_LENS_VALIDATION_GO":
        raise CountryWorkspaceRefused("country identity lens validation did not pass")
    if not str(lens_checksum).startswith("sha256:"):
        raise CountryWorkspaceRefused("the task-matched lens checksum is not pinned")
    full_band = next(
        (
            candidate
            for candidate in identity_calibration.get("candidates", ())
            if tuple(candidate.get("band", ())) == LAYERS
        ),
        None,
    )
    if full_band is None or "France->China" not in full_band.get(
        "passing_directions", ()
    ):
        raise CountryWorkspaceRefused(
            "the stored identity calibration does not establish France-to-China "
            "at the frozen L16-L40 band"
        )
    if development.get("verdict") != "COUNTRY_FRANCE_CHINA_DEVELOPMENT_GO":
        raise CountryWorkspaceRefused(
            "France-to-China downstream development did not pass"
        )
    body = {
        "version": f"{PROTOCOL_VERSION}.france_china_followup_design.v1",
        "protocol_digest": (
            protocol.get("effective_protocol_digest")
            or protocol.get("protocol_digest")
        ),
        "media_population_digest": media_validation.get("population_digest"),
        "capability_report_checksum": capability.get("report_checksum"),
        "lens_validation_report_checksum": lens_validation.get("report_checksum"),
        "identity_calibration_report_checksum": identity_calibration.get(
            "report_checksum"
        ),
        "task_matched_lens_checksum": lens_checksum,
        "development_report_checksum": development.get("report_checksum"),
        "direction": "France->China",
        "properties": list(PROPERTIES),
        "modalities": list(MODALITIES),
        "band": list(LAYERS),
        "alpha": ALPHA,
        "position_rule": POSITION_RULE,
        "confirmation_per_country": N_CONFIRMATION_PER_COUNTRY,
        "selection_used_downstream_fact_outcomes": False,
        "fresh_confirmation_outputs_opened": False,
        "parent_two_pair_verdict_unchanged": True,
    }
    return {**body, "design_checksum": payload_checksum(body)}


def freeze_confirmation_design(
    *,
    protocol: Mapping,
    media_validation: Mapping,
    capability: Mapping,
    localization: Mapping,
    development: Mapping,
    predeclared_band: Sequence[int] | None = None,
) -> dict:
    if not bool(media_validation.get("passed")):
        raise CountryWorkspaceRefused("media validation did not pass")
    if not bool(capability.get("generalization_ready")):
        raise CountryWorkspaceRefused(
            "clean source capability did not cover two independent pairs"
        )
    if predeclared_band is None:
        if localization.get("verdict") != "COUNTRY_DIRECT_PATHS_GO":
            raise CountryWorkspaceRefused("direct-answer path localization did not pass")
        selected_paths = localization.get("selected_paths")
        localization_role = "outcome_blind_path_selector"
    else:
        band = tuple(map(int, predeclared_band))
        if band != LAYERS:
            raise CountryWorkspaceRefused(
                "the diagnostic-gate amendment permits only the predeclared "
                f"full L{LAYERS[0]}-L{LAYERS[-1]} band"
            )
        selected_paths = {
            property_name: {
                "path_id": f"L{band[0]}-{band[-1]}",
                "band": list(band),
                "selection_basis": "predeclared_full_band_before_exact_swap_outputs",
            }
            for property_name in PROPERTIES
        }
        localization_role = "diagnostic_not_a_necessary_gate"
    directions = list(development.get("passing_directions_both_properties") or ())
    if not bool(development.get("generalized_across_two_pairs")):
        raise CountryWorkspaceRefused(
            "development did not pass both properties across two independent pairs"
        )
    body = {
        "version": PROTOCOL_VERSION,
        "protocol_digest": (
            protocol.get("effective_protocol_digest")
            or protocol.get("protocol_digest")
        ),
        "media_population_digest": media_validation.get("population_digest"),
        "capability_report_checksum": capability.get("report_checksum"),
        "localization_report_checksum": localization.get("report_checksum"),
        "development_report_checksum": development.get("report_checksum"),
        "selected_paths": selected_paths,
        "localization_role": localization_role,
        "directions": directions,
        "properties": list(PROPERTIES),
        "modalities": list(MODALITIES),
        "alpha": ALPHA,
        "position_rule": POSITION_RULE,
        "confirmation_per_country": N_CONFIRMATION_PER_COUNTRY,
        "frozen_before_confirmation_outputs": True,
    }
    return {**body, "design_checksum": payload_checksum(body)}


def freeze_identity_calibrated_confirmation_design(
    *,
    protocol: Mapping,
    media_validation: Mapping,
    capability: Mapping,
    lens_validation: Mapping,
    identity_calibration: Mapping,
    development: Mapping,
) -> dict:
    """Freeze downstream confirmation after outcome-blind identity calibration."""

    if not bool(media_validation.get("passed")):
        raise CountryWorkspaceRefused("media validation did not pass")
    if not bool(capability.get("generalization_ready")):
        raise CountryWorkspaceRefused(
            "clean source capability did not cover two independent pairs"
        )
    if lens_validation.get("verdict") != "COUNTRY_IDENTITY_LENS_VALIDATION_GO":
        raise CountryWorkspaceRefused("country identity lens validation did not pass")
    if (
        identity_calibration.get("verdict")
        != "COUNTRY_IDENTITY_BAND_CALIBRATION_GO"
    ):
        raise CountryWorkspaceRefused("country identity band calibration did not pass")
    selected = dict(identity_calibration.get("selected") or {})
    band = tuple(map(int, selected.get("band") or ()))
    if band not in PATH_BANDS:
        raise CountryWorkspaceRefused(
            f"identity calibration selected unregistered band {list(band)}"
        )
    directions = list(development.get("passing_directions_both_properties") or ())
    if not bool(development.get("generalized_across_two_pairs")):
        raise CountryWorkspaceRefused(
            "development did not pass both properties across two independent pairs"
        )
    selected_paths = {
        property_name: {
            "path_id": selected["path_id"],
            "band": list(band),
            "selection_basis": "identity_swap_only_before_downstream_fact_outputs",
        }
        for property_name in PROPERTIES
    }
    body = {
        "version": PROTOCOL_VERSION,
        "protocol_digest": protocol.get("protocol_digest"),
        "media_population_digest": media_validation.get("population_digest"),
        "capability_report_checksum": capability.get("report_checksum"),
        "lens_validation_report_checksum": lens_validation.get("report_checksum"),
        "identity_calibration_report_checksum": identity_calibration.get(
            "report_checksum"
        ),
        "development_report_checksum": development.get("report_checksum"),
        "selected_paths": selected_paths,
        "selection_used_downstream_fact_outcomes": False,
        "directions": directions,
        "properties": list(PROPERTIES),
        "modalities": list(MODALITIES),
        "alpha": ALPHA,
        "position_rule": POSITION_RULE,
        "confirmation_per_country": N_CONFIRMATION_PER_COUNTRY,
        "frozen_before_confirmation_outputs": True,
    }
    return {**body, "design_checksum": payload_checksum(body)}


def assert_finite_rows(rows: Sequence[Mapping]) -> None:
    for index, row in enumerate(rows):
        for key, value in row.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise CountryWorkspaceRefused(
                    f"row {index} field {key!r} is non-finite"
                )


__all__ = [
    "ALPHA",
    "ANSWER_ALIASES",
    "CAPABILITY_GATE_VERSION",
    "COUNTRY_COMPLETION_INSTRUCTION",
    "COUNTRY_IDENTITY_BAND_CALIBRATION_VERSION",
    "COUNTRY_IDENTITY_LENS_VALIDATION_VERSION",
    "CONTROL_COUNTRIES",
    "CountryMediaRow",
    "CountryWorkspaceRefused",
    "DATASET_CARD",
    "DATASET_ID",
    "DATASET_REVISION",
    "DIRECTIONS",
    "EVAL_COUNTRIES",
    "FACTS",
    "FAMILYWISE_ALPHA",
    "FIT_COUNTRIES",
    "LAYERS",
    "MIN_CAPABILITY_RATE",
    "MIN_CONTROL_MARGIN",
    "MIN_DIRECT_ANSWER_RATE",
    "MIN_IDENTITY_LENS_TOP1_RATE",
    "MIN_IDENTITY_SWAP_RATE",
    "MIN_SWAP_RATE",
    "MODALITIES",
    "N_CONFIRMATION_PER_COUNTRY",
    "N_DEVELOPMENT_PER_COUNTRY",
    "N_FIT_PER_COUNTRY",
    "N_LOCALIZATION_PER_COUNTRY",
    "PATH_BANDS",
    "POSITION_RULE",
    "PROPERTIES",
    "PROTOCOL_VERSION",
    "TARGET_LAYER",
    "answer_matches",
    "assistant_prefill",
    "benchmark_spec",
    "capability_report",
    "causal_report",
    "direct_answer_localization_report",
    "fact",
    "freeze_confirmation_design",
    "freeze_france_china_followup_design",
    "freeze_identity_calibrated_confirmation_design",
    "france_china_followup_report",
    "identity_band_calibration_report",
    "identity_lens_validation_report",
    "identity_matches",
    "normalize_surface",
    "speech_evidence",
    "text_evidence",
    "validate_media_plan",
]
