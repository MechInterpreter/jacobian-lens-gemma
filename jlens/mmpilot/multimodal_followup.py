# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Follow-ups to the confirmed pooled multimodal coordinate-exchange result.

The confirmed result this module builds on is narrow and should be stated
narrowly: a *pooled* multimodal J-lens, fitted on interleaved text, image and
spoken-audio examples, supports an exact alpha=1 coordinate exchange over the
contiguous band L16-L40 at every original prompt position, and that exchange
moves the model's free, unrestricted answer from the ``bird`` leg count to the
``cat`` leg count in all three modalities on a fresh population.

Three things that result does **not** establish, and which this module exists
to test or to label honestly:

**A. Where in L16-L40 the effect lives.** The confirmed study patched the whole
band. Nothing in it localizes the effect. :func:`localization_grid` freezes a
band grid and :func:`summarize_localization` scores it, but the only population
available for this is the *already spent* broad development population, so
every localization output carries ``"exploratory"`` and no confirmation
language. :func:`localization_claim_boundary` additionally refuses to emit an
onset claim from a nested band family, because a nested suffix or prefix chain
confounds band position with band length and total perturbation magnitude.

**B. Whether the effect generalizes past one pair and one property.** The four
concepts used so far answer the leg-count question ``bird=2``, ``cat=4``,
``zebra=4``, ``giraffe=4``. So ``bird->cat``, ``bird->zebra`` and
``bird->giraffe`` all test the *same* downstream answer change, 2 -> 4, and
``cat->zebra`` changes no observable leg-count answer at all. Leg count can
therefore test target-concept generalization and cannot test generalization to
a new downstream property. :data:`PROPERTY_FAMILIES` declares candidate
non-leg-count properties, and :func:`audit_property_family` refuses the ones
whose correct surface answer is genuinely contested.

**C. Whether the observed direction asymmetry is real.** ``cat->bird`` *was*
tested in development and produced 0 successes in 24 trials; see
:func:`development_direction_record`. That is a recorded development
observation on a spent population, not an established property of the
representation, and this module never describes it as one.

Every population identity opened by the completed studies — including all 64
photographs opened during confirmation capability screening, not merely the 16
that were recruited — is spent. :func:`exclusion_universe` collects them and
:func:`artifact_exclusion_audit` proves, read-only, what a new population
excluded.

Nothing here fits a lens. The pooled L16-L40 lens is loaded by checksum and
reused; :func:`assert_lens_reused_not_refitted` is the guard.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from jlens.mmpilot.multimodal_lens import (
    MODALITIES,
    _verified_payload,
    holm_adjust,
    paired_binary_one_sided_p,
)
from jlens.mmpilot.store import payload_checksum

FOLLOWUP_VERSION = "mmpilot.multimodal_followup.v1"
LOCALIZATION_VERSION = "mmpilot.multimodal_band_localization_exploratory.v1"
PROPERTY_AUDIT_VERSION = "mmpilot.multimodal_property_audit.v1"
PROPERTY_PROMPT_SCREEN_VERSION = "mmpilot.multimodal_property_prompt_screen.v1"
NEW_PROPERTY_DEVELOPMENT_VERSION = "mmpilot.multimodal_new_property_development.v1"
NEW_PROPERTY_FREEZE_VERSION = "mmpilot.multimodal_new_property_frozen_design.v1"
NEW_PROPERTY_CONFIRMATION_VERSION = "mmpilot.multimodal_new_property_confirmation.v1"
ASYMMETRY_VERSION = "mmpilot.multimodal_asymmetry_replication.v1"
EXCLUSION_AUDIT_VERSION = "mmpilot.multimodal_population_exclusion_audit.v1"

#: The validated contiguous band. Only the **pooled** lens spans it: the broad
#: study fitted the pooled early shard L16-L32 and combined it with the pooled
#: L33-L40 shard. The text-only, image-only and spoken-audio-only lenses cover
#: L33-L40 only, so no four-arm comparison over L16-L40 exists or is claimed.
VALIDATED_BAND: tuple[int, ...] = tuple(range(16, 41))
POOLED_ONLY_BAND_NOTE = (
    "only the pooled arm spans L16-L40; the text, image and spoken-audio arms "
    "cover L33-L40 only, so a four-arm L16-L40 comparison would require "
    "fitting L16-L32 for those three arms and has not been done"
)

#: The three controls. A study that drops one is refused, not downgraded.
CONTROL_CONDITIONS: tuple[str, ...] = ("zero", "random", "unrelated")
REQUIRED_CONDITIONS: tuple[str, ...] = ("exact", *CONTROL_CONDITIONS)

# Prompt development is deliberately isolated from causal development.  These
# alternatives were declared after the first animal-sound capability audit
# failed, so the resulting screen is outcome-informed development evidence.
# It may choose a prompt, but it may not support a causal claim.  A chosen
# prompt must clear the unchanged clean-capability gate again on a fresh
# population before any coordinate exchange runs.
ANIMAL_SOUND_PROMPT_CANDIDATES: tuple[dict, ...] = (
    {
        "prompt_id": "baseline_v1",
        "rationale": "the original capability-audit wording",
        "templates": {
            "text": (
                "Caption: {caption}\nWhat sound does the animal in the evidence "
                "make? Answer with one word.\nAnswer:"
            ),
            "image": (
                "What sound does the animal in the evidence make? Answer with "
                "one word.\nAnswer:"
            ),
            "spoken_audio": (
                "What sound does the animal in the evidence make? Answer with "
                "one word.\nAnswer:"
            ),
        },
    },
    {
        "prompt_id": "identity_explicit_v1",
        "rationale": (
            "clarifies that the evidence identifies an animal and the requested "
            "sound is typical of that kind of animal; it supplies neither the "
            "identity nor the answer"
        ),
        "templates": {
            "text": (
                "Use this caption to identify the animal: {caption}\nWhat sound "
                "does that kind of animal typically make? Reply with only the "
                "sound word.\nAnswer:"
            ),
            "image": (
                "Identify the animal in the image. What sound does that kind of "
                "animal typically make? Reply with only the sound word.\nAnswer:"
            ),
            "spoken_audio": (
                "Use the spoken caption to identify the animal. What sound does "
                "that kind of animal typically make? Reply with only the sound "
                "word.\nAnswer:"
            ),
        },
    },
    {
        "prompt_id": "knowledge_cloze_v1",
        "rationale": (
            "states the same identity-conditioned recall task as a short cloze, "
            "reducing meta-answers about whether a literal sound was supplied"
        ),
        "templates": {
            "text": (
                "Caption: {caption}\nThe typical sound made by this kind of "
                "animal is"
            ),
            "image": "The typical sound made by the animal shown is",
            "spoken_audio": (
                "The typical sound made by the animal described in the spoken "
                "caption is"
            ),
        },
    },
)
PROPERTY_PROMPT_SCREEN_CONCEPTS: tuple[str, ...] = ("cat", "cow")


class MultimodalFollowupRefused(RuntimeError):
    """The requested follow-up would mix, overstate or mislabel evidence."""


# --------------------------------------------------------------- 0. record


#: What the broad development study actually tested at alpha=1, verbatim.
#: Every one of the six directions was run; none is "untested". Successes are
#: out of 24 trials (8 photographs x 3 modalities).
ORIGINAL_DEVELOPMENT_DIRECTIONS: tuple[dict, ...] = (
    {"direction": "bird->cat", "successes": 24, "trials": 24, "tested": True},
    {"direction": "cat->bird", "successes": 0, "trials": 24, "tested": True},
    {"direction": "bird->zebra", "successes": 1, "trials": 24, "tested": True},
    {"direction": "zebra->bird", "successes": 0, "trials": 24, "tested": True},
    {"direction": "bird->giraffe", "successes": 3, "trials": 24, "tested": True},
    {"direction": "giraffe->bird", "successes": 0, "trials": 24, "tested": True},
)

#: The leg-count answer each screened concept has. Three of the four share one
#: answer, which is exactly why leg count cannot carry a new-property claim.
LEG_COUNT_ANSWERS: dict[str, str] = {
    "bird": "2",
    "cat": "4",
    "zebra": "4",
    "giraffe": "4",
}


def development_direction_record() -> dict:
    """The corrected, citable statement of what development measured.

    Used wherever a report or notebook would otherwise be tempted to say that
    the reverse direction was never tried. It was tried and it failed; whether
    that failure replicates is an open prospective question (Experiment C).
    """

    payload = {
        "version": FOLLOWUP_VERSION,
        "alpha": 1.0,
        "trials_per_direction": 24,
        "trial_structure": "8 photographs x 3 modalities",
        "population": "broad development population (spent)",
        "directions": [dict(row) for row in ORIGINAL_DEVELOPMENT_DIRECTIONS],
        "any_direction_untested": False,
        "reverse_direction_tested": True,
        "accurate_statement": (
            "Cat to bird was tested in development and produced 0 successes in "
            "24 trials, but this apparent asymmetry has not been "
            "independently tested on fresh data."
        ),
        "forbidden_statement": "cat->bird was untested",
        "asymmetry_established": False,
        "asymmetry_candidate_explanations": [
            "model capability on the reverse question",
            "prompt behaviour and answer-surface conventions",
            "coordinate quality for the two concept tokens",
            "concept geometry (the two lens directions are not symmetric)",
            "a genuine representational asymmetry",
        ],
        "what_would_settle_it": (
            "a prospective cat->bird study on fresh cat media under the frozen "
            "leg-count protocol (Experiment C); it tests replication of the "
            "observed difference, not its cause"
        ),
    }
    return {**payload, "record_digest": payload_checksum(payload)}


def leg_count_property_limit() -> dict:
    """Why the next experiment cannot keep using leg count."""

    distinct = sorted(set(LEG_COUNT_ANSWERS.values()))
    payload = {
        "version": FOLLOWUP_VERSION,
        "answers": dict(LEG_COUNT_ANSWERS),
        "distinct_answers": distinct,
        "downstream_change_tested_so_far": "2 -> 4",
        "directions_sharing_that_change": [
            "bird->cat",
            "bird->zebra",
            "bird->giraffe",
        ],
        "directions_with_no_observable_change": [
            f"{a}->{b}"
            for a in LEG_COUNT_ANSWERS
            for b in LEG_COUNT_ANSWERS
            if a != b and LEG_COUNT_ANSWERS[a] == LEG_COUNT_ANSWERS[b]
        ],
        "can_test": "target-concept generalization",
        "cannot_test": "generalization to a new downstream property",
        "consequence": (
            "a new-property study needs a property whose correct answer differs "
            "between the source and target concept and is not leg count"
        ),
    }
    return {**payload, "limit_digest": payload_checksum(payload)}


# --------------------------------------------- A. exploratory localization


@dataclass(frozen=True)
class LocalizationBand:
    """One predeclared contiguous sub-band of the validated L16-L40 band."""

    name: str
    start: int
    stop: int  # inclusive
    families: tuple[str, ...]

    @property
    def layers(self) -> tuple[int, ...]:
        return tuple(range(int(self.start), int(self.stop) + 1))

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "start": int(self.start),
            "stop_inclusive": int(self.stop),
            "n_layers": len(self.layers),
            "layers": list(self.layers),
            "families": list(self.families),
        }


#: Suffix bands: every one ends at L40, so they are a nested chain.
_SUFFIX_STARTS = (16, 20, 24, 28, 33, 37)
#: Prefix bands: every one starts at L16, so they are a nested chain too.
_PREFIX_STOPS = (20, 24, 28, 32, 36)
#: A disjoint five-way partition of L16-L40. This is the only family whose
#: members do not contain one another.
_PARTITION = ((16, 20), (21, 25), (26, 30), (31, 35), (36, 40))


def localization_grid() -> dict:
    """Freeze the band grid and the analysis rule before any band is run.

    Three families, deliberately chosen for what they can and cannot show:

    ``suffix``
        nested bands ending at L40. Tells you how late a band can start and
        still carry the effect, *confounded* with how many layers it patches.
    ``prefix``
        nested bands starting at L16. The mirror question, same confound.
    ``partition``
        five disjoint five-layer windows. The only family that can attribute
        the effect to a region without a nesting confound — but it tests
        individual sufficiency of each window, not necessity, and a
        distributed or redundant code can fail every window while the full
        band passes.
    """

    bands: dict[str, LocalizationBand] = {}

    def _add(start: int, stop: int, family: str) -> None:
        name = f"L{start}_L{stop}"
        existing = bands.get(name)
        families = ((*existing.families, family) if existing else (family,))
        bands[name] = LocalizationBand(name, start, stop, tuple(dict.fromkeys(families)))

    for start in _SUFFIX_STARTS:
        _add(start, 40, "suffix")
    for stop in _PREFIX_STOPS:
        _add(16, stop, "prefix")
    for start, stop in _PARTITION:
        _add(start, stop, "partition")

    ordered = [bands[name] for name in sorted(bands, key=lambda key: (bands[key].start, bands[key].stop))]
    for band in ordered:
        if not set(band.layers) <= set(VALIDATED_BAND):
            raise MultimodalFollowupRefused(
                f"band {band.name} leaves the validated L16-L40 band"
            )
    payload = {
        "version": LOCALIZATION_VERSION,
        "validated_band": list(VALIDATED_BAND),
        "bands": [band.to_dict() for band in ordered],
        "n_bands": len(ordered),
        "families": {
            "suffix": [band.name for band in ordered if "suffix" in band.families],
            "prefix": [band.name for band in ordered if "prefix" in band.families],
            "partition": [band.name for band in ordered if "partition" in band.families],
        },
        "analysis_rule": {
            "frozen_before_any_sub_band_outcome": True,
            "conditions": list(REQUIRED_CONDITIONS),
            "alpha": 1.0,
            "position_rule": "every original prompt position",
            "endpoint": "unrestricted full-vocabulary next-token top1",
            "teacher_forcing_used": False,
            "candidate_list_supplied": False,
            "band_carries_effect_if": (
                "exact success rate >= 0.50 in every modality and strictly "
                "greater than the zero, random and unrelated control rates in "
                "every modality, with coordinate-integrity and activation-norm "
                "checks passing in every condition"
            ),
            "min_exact_success_rate": 0.50,
            "max_activation_norm_ratio": 1.25,
            "max_update_to_activation_norm_ratio": 0.50,
            "max_orthogonal_residual_drift": 1e-5,
            "max_coordinate_update_error": 1e-5,
        },
        "identifiability": {
            "suffix": (
                "orders bands by start layer while band length shrinks with the "
                "start layer; a pass/fail boundary in this family cannot be read "
                "as an onset because position and perturbation size move together"
            ),
            "prefix": (
                "mirror of suffix, with the same length confound; can bound a "
                "sufficient early region, cannot exclude a later one"
            ),
            "partition": (
                "disjoint windows; a passing window is individually sufficient "
                "on this spent population. A failing window is not evidence of "
                "non-involvement: the code may be distributed across windows, "
                "or redundant, or need more layers than five to move the answer"
            ),
            "not_identifiable_by_any_family": [
                "necessity of any layer (no complement-ablation arm is run)",
                "an exact onset layer",
                "whether the effect is carried by one mechanism or several",
            ],
        },
        "population": "already spent broad development population",
        "label": "exploratory",
        "is_confirmation": False,
        "claim_language": (
            "exploratory/descriptive only: the population was already used for "
            "development, so nothing here is confirmation and no verdict from "
            "it may be promoted without a fresh prospective study"
        ),
    }
    return {**payload, "grid_digest": payload_checksum(payload)}


def bands_are_nested_chain(bands: Sequence[Mapping]) -> bool:
    """Whether every pair of bands is in a containment relation."""

    layer_sets = [frozenset(map(int, band.get("layers") or ())) for band in bands]
    return all(
        a <= b or b <= a
        for index, a in enumerate(layer_sets)
        for b in layer_sets[index + 1 :]
    )


def localization_budget(
    *,
    grid: Mapping,
    n_photographs: int,
    modalities: Sequence[str] = MODALITIES,
    conditions: Sequence[str] = REQUIRED_CONDITIONS,
) -> dict:
    """Exact new forward-pass count, printed before the model is loaded.

    Experiment A refits nothing and recruits nothing, but it still spends one
    model forward per band x photograph x modality x condition, plus one clean
    forward per photograph x modality that every band reuses.
    """

    assert_controls_complete(conditions)
    n_bands = int(grid["n_bands"])
    cells = int(n_photographs) * len(tuple(modalities))
    patched = n_bands * cells * len(tuple(conditions))
    payload = {
        "version": LOCALIZATION_VERSION,
        "n_bands": n_bands,
        "n_photographs": int(n_photographs),
        "modalities": list(modalities),
        "conditions": list(conditions),
        "clean_forwards": cells,
        "patched_forwards": patched,
        "total_forwards": cells + patched,
        "backward_passes": 0,
        "lens_fits": 0,
        "new_media_opened": 0,
        "resume_unit": "one band x photograph x modality x condition JSON",
        "note": (
            "no new data and no fitting, but every band is a new set of model "
            "forwards; the population is reused and stays spent"
        ),
    }
    return payload


def load_localization_population(
    run_dir: str | Path,
    *,
    expected_report_checksum: str,
    direction: tuple[str, str] = ("bird", "cat"),
) -> dict:
    """Read the spent broad development population, read-only and by checksum.

    Returns the photographs that the development study actually recruited for
    ``direction`` — the same units the confirmed intervention ran on — so that
    localization changes the band and nothing else.
    """

    root = Path(run_dir)
    report = _verified_payload(
        root / "broad_pooled_multimodal_j_workspace_report.json",
        expected_checksum=expected_report_checksum,
        label="broad pooled multimodal J-lens development report",
    )
    wanted = f"{direction[0]}->{direction[1]}"
    rows = [
        row
        for row in report.get("rows") or []
        if str(row.get("direction")) == wanted
        and str(row.get("condition")) == "exact_alpha1"
    ]
    if not rows:
        raise MultimodalFollowupRefused(
            f"the development report records no alpha=1 exact rows for {wanted}"
        )
    identities: dict[str, str] = {}
    for row in rows:
        identities[str(row["group_id"])] = str(row["image_id"])
    method = dict(report.get("method") or {})
    if method.get("layers") != list(VALIDATED_BAND):
        raise MultimodalFollowupRefused(
            "the development report did not patch the validated L16-L40 band"
        )
    if method.get("teacher_forcing_used") or method.get("candidate_list_supplied"):
        raise MultimodalFollowupRefused(
            "the development report used a restricted output endpoint"
        )
    payload = {
        "version": LOCALIZATION_VERSION,
        "run_dir": str(root),
        "report_checksum": expected_report_checksum,
        "direction": list(direction),
        "groups": [
            {"group_id": group_id, "image_id": image_id}
            for group_id, image_id in sorted(identities.items())
        ],
        "n_groups": len(identities),
        "population_status": "spent_development_population",
        "reuse_licence": "descriptive and exploratory analyses only",
        "lens_refitted": False,
    }
    return {**payload, "population_digest": payload_checksum(payload)}


def load_verified_report(
    path: str | Path, *, expected_checksum: str, label: str
) -> dict:
    """Read one completed report by checksum, refusing an edited artifact.

    A thin public name for the shared verifier so a follow-up stage never has
    to reach into another module's private helper to read the evidence it is
    pinned to.
    """

    return _verified_payload(
        Path(path), expected_checksum=str(expected_checksum), label=str(label)
    )


def load_spent_confirmation_population(
    run_dir: str | Path,
    *,
    expected_report_checksum: str,
    expected_candidates: int = 64,
    expected_recruited: int = 16,
) -> dict:
    """Read the completed confirmation study read-only, to spend its media.

    The confirmation report holds one capability row per candidate photograph
    per modality. Every candidate whose row exists was opened — the model was
    run on it and its clean answer was seen — so all of them are spent, not
    only the ones recruited afterwards. This function refuses to return a
    smaller set than the study opened, which is the failure mode it exists to
    prevent.
    """

    root = Path(run_dir)
    report = _verified_payload(
        root / "fresh_multimodal_confirmation_report.json",
        expected_checksum=expected_report_checksum,
        label="fresh multimodal confirmation report",
    )
    capability_rows = list(report.get("capability_rows") or [])
    candidate_images = sorted(
        {str(row["image_id"]) for row in capability_rows if row.get("image_id")}
    )
    recruited_images = sorted(
        {str(row["image_id"]) for row in report.get("rows") or [] if row.get("image_id")}
    )
    problems: list[str] = []
    if len(candidate_images) != int(expected_candidates):
        problems.append(
            f"expected {expected_candidates} opened candidate photographs, "
            f"found {len(candidate_images)}"
        )
    if len(recruited_images) != int(expected_recruited):
        problems.append(
            f"expected {expected_recruited} recruited photographs, found "
            f"{len(recruited_images)}"
        )
    if not set(recruited_images) <= set(candidate_images):
        problems.append("recruited photographs are not a subset of the candidates")
    if report.get("method", {}).get("teacher_forcing_used") or report.get(
        "method", {}
    ).get("candidate_list_supplied"):
        problems.append("the completed confirmation used a restricted endpoint")
    if problems:
        raise MultimodalFollowupRefused(
            "the completed confirmation cannot be read as a spent "
            "population:\n  - " + "\n  - ".join(problems)
        )
    payload = {
        "version": EXCLUSION_AUDIT_VERSION,
        "run_dir": str(root),
        "report_checksum": expected_report_checksum,
        "verdict": report.get("verdict"),
        "candidate_image_ids": candidate_images,
        "n_candidates": len(candidate_images),
        "recruited_image_ids": recruited_images,
        "n_recruited": len(recruited_images),
        "n_capability_rows": len(capability_rows),
        "all_candidates_spent": True,
        "excluding_only_recruits_would_be_wrong": True,
    }
    return {**payload, "spent_digest": payload_checksum(payload)}


def _rate(rows: Sequence[Mapping]) -> float:
    return sum(bool(row.get("success")) for row in rows) / len(rows) if rows else 0.0


def summarize_localization(
    rows: Sequence[Mapping],
    *,
    grid: Mapping,
    modalities: Sequence[str] = MODALITIES,
) -> dict:
    """Score every tested band. Descriptive; produces no confirmation verdict."""

    rule = dict(grid["analysis_rule"])
    assert_controls_complete(rule["conditions"])
    cells: list[dict] = []
    passing: list[str] = []
    for band in grid["bands"]:
        band_rows = [row for row in rows if str(row.get("band")) == band["name"]]
        band_cells = []
        for modality in modalities:
            modality_rows = [
                row for row in band_rows if str(row.get("modality")) == modality
            ]
            by_condition = {
                condition: [
                    row
                    for row in modality_rows
                    if str(row.get("condition")) == condition
                ]
                for condition in rule["conditions"]
            }
            exact = by_condition["exact"]
            cell = {
                "band": band["name"],
                "layers": list(band["layers"]),
                "modality": modality,
                "n": len(exact),
                "exact_successes": sum(bool(row.get("success")) for row in exact),
                "exact_success_rate": _rate(exact),
                "controls": {
                    condition: {
                        "n": len(by_condition[condition]),
                        "successes": sum(
                            bool(row.get("success")) for row in by_condition[condition]
                        ),
                        "success_rate": _rate(by_condition[condition]),
                    }
                    for condition in CONTROL_CONDITIONS
                },
                "integrity_pass": bool(exact)
                and all(
                    bool(row.get("all_prompt_positions_patched"))
                    and list(row.get("layers_patched") or []) == list(band["layers"])
                    and float(row.get("max_orthogonal_residual_drift", 0.0))
                    <= float(rule["max_orthogonal_residual_drift"])
                    and float(row.get("max_coordinate_update_error", 0.0))
                    <= float(rule["max_coordinate_update_error"])
                    for condition_rows in by_condition.values()
                    for row in condition_rows
                ),
                "max_activation_norm_ratio": max(
                    (float(row.get("max_activation_norm_ratio", 1.0)) for row in exact),
                    default=1.0,
                ),
                "max_update_to_activation_norm_ratio": max(
                    (
                        float(row.get("max_update_to_activation_norm_ratio", 0.0))
                        for row in exact
                    ),
                    default=0.0,
                ),
            }
            cell["carries_effect"] = bool(
                cell["n"]
                and cell["exact_success_rate"] >= float(rule["min_exact_success_rate"])
                and all(
                    cell["exact_success_rate"] > control["success_rate"]
                    for control in cell["controls"].values()
                )
                and cell["integrity_pass"]
                and cell["max_activation_norm_ratio"]
                <= float(rule["max_activation_norm_ratio"])
                and cell["max_update_to_activation_norm_ratio"]
                <= float(rule["max_update_to_activation_norm_ratio"])
            )
            band_cells.append(cell)
        cells.extend(band_cells)
        if band_cells and all(cell["carries_effect"] for cell in band_cells):
            passing.append(band["name"])

    by_name = {band["name"]: band for band in grid["bands"]}
    payload = {
        "version": LOCALIZATION_VERSION,
        "grid_digest": grid["grid_digest"],
        "label": "exploratory",
        "is_confirmation": False,
        "population": grid["population"],
        "n_bands_tested": len(grid["bands"]),
        "bands_tested": [band["name"] for band in grid["bands"]],
        "cells": cells,
        "bands_carrying_effect": passing,
        "bands_by_family": {
            family: [name for name in passing if name in names]
            for family, names in grid["families"].items()
        },
        "claim_boundary": localization_claim_boundary(
            [by_name[name] for name in passing], grid=grid
        ),
    }
    payload["verdict"] = (
        "EXPLORATORY_LOCALIZATION_NO_BAND_CARRIES_EFFECT"
        if not passing
        else "EXPLORATORY_LOCALIZATION_DESCRIBED"
    )
    return {**payload, "report_checksum": payload_checksum(payload)}


def localization_claim_boundary(
    passing_bands: Sequence[Mapping], *, grid: Mapping
) -> dict:
    """What may and may not be said about a set of passing bands."""

    nested_only = bands_are_nested_chain(passing_bands) if passing_bands else True
    partition_passing = [
        band["name"]
        for band in passing_bands
        if "partition" in list(band.get("families") or [])
    ]
    payload = {
        "version": LOCALIZATION_VERSION,
        "exploratory": True,
        "is_confirmation": False,
        "passing_bands": [band["name"] for band in passing_bands],
        "passing_bands_are_nested_chain": bool(nested_only),
        "onset_layer_claimed": False,
        "onset_claim": (
            "no onset layer is identified: the passing bands form a nested "
            "chain, in which start layer and band length vary together"
            if nested_only
            else "no onset layer is identified: even with disjoint passing "
            "windows this design measures individual sufficiency on a spent "
            "population, not the layer at which the effect begins"
        ),
        "individually_sufficient_disjoint_windows": partition_passing,
        "necessity_claimed": False,
        "necessity_note": (
            "no band is shown to be necessary; the complement of each band was "
            "never ablated"
        ),
        "population_note": (
            "the population was already used for development, so these rates "
            "are descriptive; they cannot confirm anything and any localization "
            "worth asserting needs a fresh prospective study"
        ),
        "pooled_lens_only": POOLED_ONLY_BAND_NOTE,
        "grid_identifiability": dict(grid["identifiability"]),
    }
    return payload


# ---------------------------------------------- B0. property/prompt audit


@dataclass(frozen=True)
class PropertyAnswer:
    """One concept's answer under one property family, with its verdict.

    ``aliases`` are predeclared surface forms of *one* answer, never a set of
    different answers pretending to be one. ``admissible`` is False whenever
    competent speakers would disagree about the correct surface answer; the
    reason is recorded either way so the audit can be read years later.
    """

    concept: str
    answer: str
    aliases: tuple[str, ...]
    admissible: bool
    reason: str
    #: When True the answer is not declared here. It is resolved from the clean
    #: capability screen by :data:`DOMINANT_ANSWER_RULE`, a rule fixed before
    #: any data is opened. Used where the concept's *correct* answer varies
    #: across subtypes the model may see, but its *stable* answer is the thing
    #: the causal test actually needs.
    empirical_answer_required: bool = False

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["aliases"] = list(self.aliases)
        payload["alias_set_size"] = len(self.aliases)
        return payload


@dataclass(frozen=True)
class PropertyFamily:
    """A candidate downstream property, its prompt, and its answer table."""

    name: str
    question: str
    answers: tuple[PropertyAnswer, ...]
    rationale: str
    #: Whether the property can be answered by looking at the photograph
    #: instead of by consulting the animal's identity. This is the criterion
    #: the first audit was missing. A perceptually available property gives the
    #: model a route around the variable the intervention edits, so a null in
    #: the image modality cannot be told apart from a perceptual override and
    #: the causal test is not interpretable.
    perceptually_available: bool
    perceptual_rationale: str
    max_new_tokens: int = 6
    caption_prefix: str = "Caption: {caption}\n"
    notes: tuple[str, ...] = field(default_factory=tuple)
    disqualified_reason: str = ""

    @property
    def disqualified(self) -> bool:
        """Perceptually available properties cannot carry a causal claim here."""

        return bool(self.perceptually_available or self.disqualified_reason)

    def prompt(self, modality: str, caption: str) -> str:
        """The frozen prompt. Text carries the caption; image/audio do not."""

        if modality not in MODALITIES:
            raise MultimodalFollowupRefused(f"unknown modality {modality!r}")
        if modality == "text":
            return self.caption_prefix.format(caption=caption) + self.question
        return self.question

    def answer_for(self, concept: str) -> PropertyAnswer:
        for row in self.answers:
            if row.concept == concept:
                return row
        raise MultimodalFollowupRefused(
            f"{self.name!r} declares no answer for concept {concept!r}"
        )


#: The rule that fixes an empirical answer, committed before any completion is
#: read. It names a decision procedure, not an answer, so applying it to data
#: later is not a post-hoc choice.
DOMINANT_ANSWER_RULE = {
    "version": PROPERTY_AUDIT_VERSION,
    "rule": (
        "the concept's answer is the single most frequent normalized final "
        "lexical item across the clean capability screen, pooled over "
        "modalities; the concept is admissible only if that same item is also "
        "the most frequent in every individual modality and reaches the "
        "capability threshold in every individual modality"
    ),
    "ties_refused": True,
    "declared_before_data": True,
    "standard": (
        "stability, not taxonomic correctness: the causal test asks whether an "
        "identity edit changes the model's identity-conditioned answer, so the "
        "source answer must be reliably produced, not externally right"
    ),
}


def answer_key(generated: str) -> str:
    """The normalized final lexical item of a completion, or ''.

    The same surface reduction the scorer uses, exposed so the dominant-answer
    rule and the matcher can never drift apart.
    """

    import re

    from jlens.mmpilot.full_vocabulary import normalize_generated_text
    from jlens.mmpilot.workspace_replication import (
        _CONTROL_TOKEN_WORDS,
        _before_first_control_token,
    )

    words = re.findall(
        r"\w+",
        normalize_generated_text(_before_first_control_token(str(generated))),
        flags=re.UNICODE,
    )
    while len(words) > 1 and words[-1] in _CONTROL_TOKEN_WORDS:
        words.pop()
    return words[-1] if words else ""


def resolve_dominant_answer(
    completions_by_modality: Mapping[str, Sequence[str]],
    *,
    threshold: float,
    modalities: Sequence[str] = MODALITIES,
) -> dict:
    """Apply :data:`DOMINANT_ANSWER_RULE` to one concept's clean completions."""

    from collections import Counter

    per_modality: dict[str, Counter] = {}
    pooled: Counter = Counter()
    for modality in modalities:
        keys = [answer_key(value) for value in completions_by_modality.get(modality, ())]
        keys = [key for key in keys if key]
        per_modality[modality] = Counter(keys)
        pooled.update(keys)

    result = {
        "rule": dict(DOMINANT_ANSWER_RULE),
        "threshold": float(threshold),
        "pooled_counts": dict(pooled.most_common(8)),
        "counts_by_modality": {
            modality: dict(counter.most_common(5))
            for modality, counter in per_modality.items()
        },
    }
    if not pooled:
        return {**result, "resolved": False, "reason": "no completions"}
    ranked = pooled.most_common()
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return {**result, "resolved": False, "reason": "pooled tie between top answers"}
    answer = ranked[0][0]

    rates = {}
    for modality in modalities:
        counter = per_modality[modality]
        total = sum(counter.values())
        rates[modality] = (counter.get(answer, 0) / total) if total else 0.0
        if total and counter.most_common(1)[0][0] != answer:
            return {
                **result,
                "resolved": False,
                "answer_candidate": answer,
                "rates_by_modality": rates,
                "reason": f"a different answer dominates in {modality}",
            }
    if any(rate < float(threshold) for rate in rates.values()):
        return {
            **result,
            "resolved": False,
            "answer_candidate": answer,
            "rates_by_modality": rates,
            "reason": "the dominant answer is below threshold in some modality",
        }
    return {
        **result,
        "resolved": True,
        "answer": answer,
        "rates_by_modality": rates,
    }


_BODY_COVERING = PropertyFamily(
    name="body_covering",
    question=(
        "What does the body of the animal in the evidence have on the outside? "
        "Answer with one word.\nAnswer:"
    ),
    rationale=(
        "Body covering is not derivable from leg count, is a property every "
        "COCO animal has, and has a short common-noun answer that an "
        "instruction model produces unprompted. It keeps 'bird' available, "
        "which links the new property to the confirmed phenomenon. It was "
        "tried first and is now disqualified; see perceptual_rationale."
    ),
    perceptually_available=True,
    perceptual_rationale=(
        "The covering is visible in the photograph, so the model can answer by "
        "describing pixels instead of consulting identity. Measured: the image "
        "route scored 0.729/0.667/0.625 for bird/cat/sheep against 0.77-0.94 "
        "for text and spoken audio, and its failures were appearance words "
        "(black, spotted, stripes, iridescent, patches) rather than wrong "
        "coverings. A swap null in the image route could therefore mean the "
        "identity variable does not feed the answer, or that visual evidence "
        "overrode it, and this design cannot separate those."
    ),
    disqualified_reason=(
        "perceptually available: the causal test would not be interpretable in "
        "the image modality"
    ),
    answers=(
        PropertyAnswer(
            "bird", "feathers", ("feathers", "feather"), True,
            "every bird species in COCO photographs is feathered; no competent "
            "speaker answers otherwise",
        ),
        PropertyAnswer(
            "cat", "fur", ("fur",), True,
            "domestic cats are furred; 'fur' is the single ordinary answer",
        ),
        PropertyAnswer(
            "dog", "fur", ("fur",), True,
            "same as cat; 'hair' is used technically but 'fur' is the ordinary "
            "answer and the only one this audit accepts",
        ),
        PropertyAnswer(
            "sheep", "wool", ("wool", "fleece"), True,
            "wool and fleece name the same covering of the same animal and are "
            "declared here as two surfaces of one answer, not two answers",
        ),
        PropertyAnswer(
            "bear", "fur", ("fur",), True, "bears are furred; unambiguous",
        ),
        PropertyAnswer(
            "horse", "", (), False,
            "contested between 'hair', 'fur' and 'coat' with no single ordinary "
            "answer",
        ),
        PropertyAnswer(
            "cow", "", (), False,
            "contested between 'hide', 'hair' and 'fur'",
        ),
        PropertyAnswer(
            "zebra", "", (), False,
            "contested between 'fur' and 'hair', and the striping invites the "
            "model to answer 'stripes' instead of the covering",
        ),
        PropertyAnswer(
            "giraffe", "", (), False,
            "contested between 'fur', 'hair' and 'skin'; patterning invites the "
            "same distractor as zebra",
        ),
        PropertyAnswer(
            "elephant", "", (), False,
            "'skin' and 'hide' compete and elephants also have sparse hair",
        ),
    ),
    notes=(
        "an admissible pair must have different answers, so bird<->cat, "
        "bird<->sheep and cat<->sheep are candidates while cat<->dog is not",
    ),
)

_ANIMAL_SOUND = PropertyFamily(
    name="animal_sound",
    question=(
        "What sound does the animal in the evidence make? "
        "Answer with one word.\nAnswer:"
    ),
    rationale=(
        "Animal sound cannot be read off a still photograph, so every route "
        "must go through the animal's identity to answer it. That is the same "
        "structure that makes leg count work, and it is why this is now the "
        "primary candidate rather than the fallback."
    ),
    perceptually_available=False,
    perceptual_rationale=(
        "A still image carries no sound, so the answer cannot be obtained by "
        "describing the picture. The model must identify the animal and then "
        "recall its call, which is the identity-mediated path the intervention "
        "edits."
    ),
    answers=(
        PropertyAnswer(
            "cat", "meow", ("meow", "meows"), True,
            "the conventional English answer, with no competing surface",
        ),
        PropertyAnswer(
            "dog", "bark", ("bark", "barks", "woof"), True,
            "'bark' and 'woof' are the verb and the imitation of one sound; "
            "declared together as one answer",
        ),
        PropertyAnswer(
            "cow", "moo", ("moo", "moos"), True, "conventional and unrivalled",
        ),
        PropertyAnswer(
            "sheep", "baa", ("baa", "bleat", "bleats"), True,
            "'baa' is the imitation and 'bleat' the verb for the same sound",
        ),
        PropertyAnswer(
            "bird", "", (), True,
            "no single answer is taxonomically correct for COCO birds, which "
            "span gulls, ducks, pigeons and raptors. But the causal test needs "
            "a stable identity-conditioned answer, not a correct one, and the "
            "selection rule admits a group only when its caption contains the "
            "literal word 'bird', so the text and spoken-audio routes see a "
            "generic bird while only the image route shows a species. Whether "
            "one answer is stable in all three routes is therefore an "
            "empirical question, resolved by DOMINANT_ANSWER_RULE and refused "
            "if no single answer dominates in every modality",
            empirical_answer_required=True,
        ),
        PropertyAnswer(
            "horse", "", (), False,
            "'neigh', 'whinny' and 'nicker' name different sounds",
        ),
        PropertyAnswer(
            "zebra", "", (), False, "no conventional English answer",
        ),
        PropertyAnswer(
            "giraffe", "", (), False,
            "giraffes are near-silent; no conventional answer exists",
        ),
        PropertyAnswer(
            "elephant", "", (), False,
            "'trumpet' competes with 'rumble' and is also a musical-instrument "
            "homonym that pollutes the readout",
        ),
        PropertyAnswer(
            "bear", "", (), False, "'growl', 'roar' and 'grunt' all compete",
        ),
    ),
    notes=(
        "no bird entry is admissible, so this family cannot reuse the confirmed "
        "bird source concept at all",
    ),
)

#: Candidate non-leg-count properties. Candidates only: admissibility of a
#: concept here is a claim about English, not about data availability or about
#: whether Gemma 4 can actually answer it. Stage B0 must still check media
#: counts and clean capability in all three modalities before anything freezes.
PROPERTY_FAMILIES: dict[str, PropertyFamily] = {
    _BODY_COVERING.name: _BODY_COVERING,
    _ANIMAL_SOUND.name: _ANIMAL_SOUND,
}


def property_prompt_candidate(prompt_id: str) -> dict:
    """Return one declared animal-sound prompt candidate by stable ID."""

    for row in ANIMAL_SOUND_PROMPT_CANDIDATES:
        if row["prompt_id"] == str(prompt_id):
            return {
                **row,
                "templates": dict(row["templates"]),
            }
    raise MultimodalFollowupRefused(
        f"unknown animal-sound prompt {prompt_id!r}; declared IDs are "
        f"{[row['prompt_id'] for row in ANIMAL_SOUND_PROMPT_CANDIDATES]}"
    )


def property_prompt(prompt_id: str, modality: str, caption: str = "") -> str:
    """Render one declared prompt without exposing another modality's data."""

    if modality not in MODALITIES:
        raise MultimodalFollowupRefused(f"unknown modality {modality!r}")
    candidate = property_prompt_candidate(prompt_id)
    template = str(candidate["templates"][modality])
    if modality != "text" and "{caption}" in template:
        raise MultimodalFollowupRefused(
            f"prompt {prompt_id!r} leaks the caption into {modality}"
        )
    return template.format(caption=str(caption))


def property_prompt_screen_verdict(
    rows: Sequence[Mapping],
    *,
    expected_per_cell: int,
    min_clean_capability_rate: float = 0.75,
    concepts: Sequence[str] = PROPERTY_PROMPT_SCREEN_CONCEPTS,
    modalities: Sequence[str] = MODALITIES,
) -> dict:
    """Select a capability prompt without consulting a causal outcome.

    The screen is deliberately development-only.  It may choose the prompt
    with the best worst-cell capability among candidates that clear the frozen
    threshold everywhere.  Ties use mean capability and then declaration
    order.  The chosen prompt must still pass a fresh capability gate before a
    coordinate exchange is allowed to run.
    """

    if int(expected_per_cell) <= 0:
        raise MultimodalFollowupRefused("expected_per_cell must be positive")
    declared = [row["prompt_id"] for row in ANIMAL_SOUND_PROMPT_CANDIDATES]
    expected_keys = {
        (prompt_id, concept, modality)
        for prompt_id in declared
        for concept in concepts
        for modality in modalities
    }
    grouped: dict[tuple[str, str, str], list[Mapping]] = {
        key: [] for key in expected_keys
    }
    seen_units: set[tuple[str, str, str, str]] = set()
    for row in rows:
        key = (
            str(row.get("prompt_id")),
            str(row.get("concept")),
            str(row.get("modality")),
        )
        if key not in grouped:
            raise MultimodalFollowupRefused(
                f"undeclared prompt-screen cell {key}; the screen cannot widen "
                "after outputs are observed"
            )
        unit = (*key, str(row.get("group_id")))
        if unit in seen_units:
            raise MultimodalFollowupRefused(f"duplicate prompt-screen unit {unit}")
        seen_units.add(unit)
        grouped[key].append(row)

    incomplete = {
        "/".join(key): len(cell)
        for key, cell in grouped.items()
        if len(cell) != int(expected_per_cell)
    }
    if incomplete:
        raise MultimodalFollowupRefused(
            f"prompt screen is incomplete; expected {expected_per_cell} per "
            f"cell, got {incomplete}"
        )

    candidates = []
    for priority, prompt_id in enumerate(declared):
        rates = {
            concept: {
                modality: sum(
                    bool(row.get("pass"))
                    for row in grouped[(prompt_id, concept, modality)]
                ) / int(expected_per_cell)
                for modality in modalities
            }
            for concept in concepts
        }
        flat = [rates[c][m] for c in concepts for m in modalities]
        candidates.append(
            {
                **property_prompt_candidate(prompt_id),
                "rates": rates,
                "minimum_cell_rate": min(flat),
                "mean_cell_rate": sum(flat) / len(flat),
                "passes_every_cell": all(
                    rate >= float(min_clean_capability_rate) for rate in flat
                ),
                "declaration_priority": priority,
            }
        )

    passing = [row for row in candidates if row["passes_every_cell"]]
    selected = (
        sorted(
            passing,
            key=lambda row: (
                -float(row["minimum_cell_rate"]),
                -float(row["mean_cell_rate"]),
                int(row["declaration_priority"]),
            ),
        )[0]
        if passing
        else None
    )
    payload = {
        "version": PROPERTY_PROMPT_SCREEN_VERSION,
        "property_family": "animal_sound",
        "concepts": list(concepts),
        "modalities": list(modalities),
        "expected_per_cell": int(expected_per_cell),
        "min_clean_capability_rate": float(min_clean_capability_rate),
        "candidates": candidates,
        "selection_rule": (
            "among prompts passing the unchanged threshold in every declared "
            "concept-by-modality cell, maximize the minimum cell rate, then "
            "mean cell rate, then use declaration order"
        ),
        "selected_prompt_id": selected["prompt_id"] if selected else None,
        "selected_prompt": selected,
        "verdict": "PROPERTY_PROMPT_SCREEN_GO" if selected else "PROPERTY_PROMPT_SCREEN_NO_GO",
        "outcome_informed_development": True,
        "causal_outcomes_used_for_selection": False,
        "lens_fitted": False,
        "backward_passes": 0,
        "fresh_capability_revalidation_required": True,
        "causal_spending_licensed": False,
        "claim_boundary": (
            "this screen chooses wording on already-spent development media; "
            "it is not a causal result and cannot open confirmation"
        ),
        "rows": [dict(row) for row in rows],
    }
    return {**payload, "report_checksum": payload_checksum(payload)}


def property_answer_matches(generated: str, answer: PropertyAnswer | Mapping) -> bool:
    """Score an unrestricted complete generation against one predeclared answer.

    Delegates the surface rule to the frozen completion matcher already used by
    the text replication study: control tokens are stripped, the answer must be
    the final lexical item, and a negated completion never counts. Any declared
    alias of the answer satisfies it.
    """

    from jlens.mmpilot.workspace_replication import completion_answer_matches

    aliases = (
        answer.aliases
        if isinstance(answer, PropertyAnswer)
        else tuple(answer.get("aliases") or ())
    )
    if not aliases:
        raise MultimodalFollowupRefused(
            "an answer with no declared alias cannot be scored; the concept was "
            "refused by the property audit"
        )
    return any(completion_answer_matches(generated, alias) for alias in aliases)


def audit_property_family(
    family: str | PropertyFamily,
    *,
    available_media: Mapping[str, int] | None = None,
    min_media_per_concept: int = 0,
    clean_capability: Mapping[str, Mapping[str, float]] | None = None,
    min_clean_capability_rate: float = 0.75,
    observed_completions: Mapping[str, Mapping[str, Sequence[str]]] | None = None,
    allow_perceptually_available: bool = False,
) -> dict:
    """Decide which concepts and directions this property may legitimately use.

    Four filters, in order, all of which are recorded:

    0. **Perceptual availability** — a property the model can answer by looking
       at the photograph gives it a route around the variable the intervention
       edits, so a null in the image modality is uninterpretable. Such a family
       is disqualified outright. This is the filter the first audit lacked, and
       its absence is why ``body_covering`` was tried and had to be withdrawn.
    1. **Semantic admissibility** — declared in :data:`PROPERTY_FAMILIES`; a
       concept whose correct surface answer is contested is refused, unless it
       is marked ``empirical_answer_required``, in which case its answer is
       resolved from ``observed_completions`` by :data:`DOMINANT_ANSWER_RULE`.
    2. **Media availability** — ``available_media[concept]`` fresh photograph
       groups must reach ``min_media_per_concept``.
    3. **Clean capability** — the untouched model must answer the property
       question in *every* modality at ``min_clean_capability_rate``.

    A direction survives only if both endpoints survive all four and the two
    answers differ, which is the filter leg count could not provide.
    """

    spec = PROPERTY_FAMILIES[family] if isinstance(family, str) else family
    disqualified = bool(spec.disqualified) and not allow_perceptually_available
    rows: list[dict] = []
    for answer in spec.answers:
        record = answer.to_dict()
        record["empirical_resolution"] = None
        if answer.empirical_answer_required:
            resolution = (
                resolve_dominant_answer(
                    (observed_completions or {}).get(answer.concept, {}),
                    threshold=float(min_clean_capability_rate),
                )
                if observed_completions is not None
                else {"resolved": False, "reason": "no completions supplied yet"}
            )
            record["empirical_resolution"] = resolution
            if resolution.get("resolved"):
                record["answer"] = str(resolution["answer"])
                record["aliases"] = [str(resolution["answer"])]
                record["alias_set_size"] = 1
                record["admissible"] = True
            else:
                record["admissible"] = False
                record["reason"] = (
                    f"{record['reason']}; the rule did not resolve a stable "
                    f"answer ({resolution.get('reason')})"
                )
        media = int((available_media or {}).get(answer.concept, 0))
        record["available_media"] = media
        record["media_sufficient"] = (
            media >= int(min_media_per_concept) if available_media is not None else None
        )
        resolved_empirically = bool(
            answer.empirical_answer_required
            and record["empirical_resolution"]
            and record["empirical_resolution"].get("resolved")
        )
        if resolved_empirically:
            # The resolution's own per-modality rates ARE the capability
            # check. Requiring a separately supplied clean_capability dict
            # here would recreate the chicken-and-egg dependency this
            # resolution exists to break: the declared answer needed to score
            # capability does not exist until the resolution has run.
            capability = dict(record["empirical_resolution"]["rates_by_modality"])
            record["clean_capability"] = capability
            record["capability_by_modality_sufficient"] = all(
                float(capability.get(modality, 0.0)) >= float(min_clean_capability_rate)
                for modality in MODALITIES
            )
        else:
            capability = dict((clean_capability or {}).get(answer.concept, {}))
            record["clean_capability"] = capability
            record["capability_by_modality_sufficient"] = (
                all(
                    float(capability.get(modality, 0.0)) >= float(min_clean_capability_rate)
                    for modality in MODALITIES
                )
                if clean_capability is not None
                else None
            )
        record["usable"] = bool(
            record["admissible"]
            and not disqualified
            and (record["media_sufficient"] is not False)
            and (record["capability_by_modality_sufficient"] is not False)
        )
        rows.append(record)

    usable = [row for row in rows if row["usable"]]
    directions = [
        {
            "direction": f"{source['concept']}->{target['concept']}",
            "source": source["concept"],
            "target": target["concept"],
            "source_answer": source["answer"],
            "target_answer": target["answer"],
            "source_aliases": list(source["aliases"]),
            "target_aliases": list(target["aliases"]),
            "changes_property": True,
        }
        for source in usable
        for target in usable
        if source["concept"] != target["concept"]
        and not (set(source["aliases"]) & set(target["aliases"]))
    ]
    payload = {
        "version": PROPERTY_AUDIT_VERSION,
        "family": spec.name,
        "perceptually_available": bool(spec.perceptually_available),
        "perceptual_rationale": spec.perceptual_rationale,
        "family_disqualified": disqualified,
        "family_disqualified_reason": spec.disqualified_reason,
        "dominant_answer_rule": dict(DOMINANT_ANSWER_RULE),
        "question": spec.question,
        "prompt_by_modality": {
            modality: spec.prompt(modality, "{caption}") for modality in MODALITIES
        },
        "max_new_tokens": int(spec.max_new_tokens),
        "endpoint": "unrestricted complete generation, scored after the fact",
        "teacher_forcing_used": False,
        "candidate_list_supplied": False,
        "answer_normalization": (
            "NFKC, casefold, whitespace collapse, punctuation stripped, control "
            "tokens stripped, answer must be the final lexical item, negated "
            "completions rejected; declared aliases only, no post-hoc synonyms"
        ),
        "rationale": spec.rationale,
        "notes": list(spec.notes),
        "concepts": rows,
        "admissible_concepts": [row["concept"] for row in rows if row["admissible"]],
        "refused_concepts": [
            {"concept": row["concept"], "reason": row["reason"]}
            for row in rows
            if not row["admissible"]
        ],
        "usable_concepts": [row["concept"] for row in usable],
        "candidate_directions": directions,
        "min_media_per_concept": int(min_media_per_concept),
        "min_clean_capability_rate": float(min_clean_capability_rate),
        "leg_count_limit": leg_count_property_limit(),
        "boundary": (
            "admissibility here is a claim about English answer surfaces and "
            "about this checkpoint's measured clean capability; it is not a "
            "claim that any direction will show a causal effect"
        ),
    }
    payload["verdict"] = (
        "PROPERTY_AUDIT_PERCEPTUALLY_AVAILABLE_NO_GO"
        if disqualified
        else "PROPERTY_AUDIT_GO"
        if directions
        else "PROPERTY_AUDIT_NO_GO"
    )
    return {**payload, "audit_digest": payload_checksum(payload)}


def assert_property_pair_changes_answer(
    family: str | PropertyFamily,
    source: str,
    target: str,
    *,
    resolved: Mapping[str, Mapping] | None = None,
) -> dict:
    """Refuse a direction whose two concepts share an answer or an alias.

    ``resolved`` supplies the current audit's per-concept rows (from
    :func:`audit_property_family`'s ``"concepts"`` list) for any concept whose
    answer is decided empirically rather than declared. Without it, a concept
    marked ``empirical_answer_required`` that has not yet been resolved from
    data is refused outright, rather than silently treated as an admissible
    answer of ``""`` with no aliases — which would let an unresolved pair like
    ``bird->cat`` pass this check by accident, since an empty alias set never
    overlaps anything.
    """

    spec = PROPERTY_FAMILIES[family] if isinstance(family, str) else family
    if spec.disqualified:
        raise MultimodalFollowupRefused(
            f"{spec.name}: {spec.disqualified_reason}"
        )

    def _answer_for(concept: str) -> PropertyAnswer:
        declared = spec.answer_for(concept)
        row = (resolved or {}).get(concept)
        if row is None:
            return declared
        return PropertyAnswer(
            concept=concept,
            answer=str(row.get("answer") or ""),
            aliases=tuple(row.get("aliases") or ()),
            admissible=bool(row.get("admissible")),
            reason=str(row.get("reason") or ""),
            empirical_answer_required=bool(
                row.get("empirical_answer_required")
                or declared.empirical_answer_required
            ),
        )

    source_answer = _answer_for(source)
    target_answer = _answer_for(target)
    for answer in (source_answer, target_answer):
        if not answer.admissible or (
            answer.empirical_answer_required and not answer.aliases
        ):
            raise MultimodalFollowupRefused(
                f"{spec.name}: concept {answer.concept!r} was refused by the "
                f"property audit "
                f"({answer.reason or 'no resolved empirical answer'})"
            )
    overlap = sorted(set(source_answer.aliases) & set(target_answer.aliases))
    if overlap or source_answer.answer == target_answer.answer:
        raise MultimodalFollowupRefused(
            f"{spec.name}: {source}->{target} does not change the property "
            f"answer (shared surfaces {overlap or [source_answer.answer]})"
        )
    return {
        "family": spec.name,
        "direction": f"{source}->{target}",
        "source_answer": source_answer.to_dict(),
        "target_answer": target_answer.to_dict(),
        "changes_property": True,
        "is_leg_count": False,
    }


# ---------------------------------------------------- shared gates/budgets


def assert_controls_complete(conditions: Sequence[str]) -> None:
    """Every study here runs exact plus all three controls, or it does not run."""

    missing = [name for name in REQUIRED_CONDITIONS if name not in set(conditions)]
    if missing:
        raise MultimodalFollowupRefused(
            f"the zero, random and unrelated controls are not optional; missing "
            f"{missing}"
        )


def assert_lens_reused_not_refitted(config: Mapping) -> None:
    """Refuse any configuration that would fit or refit a lens here."""

    if config.get("lens_refitted") or config.get("fit_lens") or config.get("refit"):
        raise MultimodalFollowupRefused(
            "these follow-ups reuse the checksum-pinned pooled L16-L40 lens; "
            "no fitting entry point is reachable from them"
        )
    if int(config.get("backward_passes", 0) or 0) > 0:
        raise MultimodalFollowupRefused(
            "a follow-up stage requested backward passes; nothing here fits"
        )
    if not str(config.get("lens_checksum") or "").startswith("sha256:"):
        raise MultimodalFollowupRefused(
            "a follow-up stage must pin the pooled lens by checksum"
        )


def assert_open_endpoint(config: Mapping) -> None:
    """Refuse teacher forcing or a candidate list anywhere in these studies."""

    if config.get("teacher_forcing_used") or config.get("candidate_list_supplied"):
        raise MultimodalFollowupRefused(
            "the endpoint must stay unrestricted: no teacher forcing and no "
            "candidate list"
        )


def _collect_image_ids(node: object) -> set[str]:
    """Recursively pull every ``image_id`` value out of an arbitrary report."""

    found: set[str] = set()
    if isinstance(node, Mapping):
        value = node.get("image_id")
        if isinstance(value, (str, int)) and str(value).strip():
            found.add(str(value))
        for child in node.values():
            found |= _collect_image_ids(child)
    elif isinstance(node, (list, tuple)):
        for item in node:
            found |= _collect_image_ids(item)
    return found


def load_extra_spent_image_ids(report_paths: Sequence[str | Path]) -> dict:
    """Best-effort, unpinned union of every ``image_id`` in arbitrary reports.

    The checksum-pinned loaders above (:func:`load_broad_pooled_development_source`,
    :func:`load_spent_confirmation_population`) are the only trustworthy source
    of *disjointness proof* for a population: an edited or truncated artifact
    fails their checksum and is refused. This function has no such guarantee —
    it exists only to let a run manually widen exclusion across artifacts this
    module does not know how to name and checksum-pin: an abandoned property
    family before trying the declared fallback, or Experiment B's opened media
    before Experiment C runs in the same session.

    Because it is unverified, this function can only ever be used to make a
    population **more** exclusive. It must never be treated as sufficient
    disjointness proof on its own, and a caller that wants a checksum-backed
    guarantee should use the pinned loaders instead.
    """

    ids_by_report: dict[str, list[str]] = {}
    union: set[str] = set()
    for raw_path in report_paths:
        path = Path(raw_path)
        if not path.is_file():
            raise MultimodalFollowupRefused(f"extra spent report not found: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise MultimodalFollowupRefused(
                f"could not read extra spent report: {path}"
            ) from exc
        found = _collect_image_ids(payload)
        ids_by_report[str(path)] = sorted(found)
        union |= found
    payload = {
        "version": EXCLUSION_AUDIT_VERSION,
        "report_paths": [str(path) for path in report_paths],
        "image_ids_by_report": ids_by_report,
        "image_ids": sorted(union),
        "n_image_ids": len(union),
        "checksum_verified": False,
        "boundary": (
            "unpinned best-effort union of image_id fields found in the given "
            "reports; widens exclusion only and never substitutes for a "
            "checksum-pinned population loader"
        ),
    }
    return {**payload, "digest": payload_checksum(payload)}


def exclusion_universe(
    *,
    fit_image_ids: Sequence[str] = (),
    eval_image_ids: Sequence[str] = (),
    prior_causal_image_ids: Sequence[str] = (),
    broad_development_image_ids: Sequence[str] = (),
    confirmation_candidate_image_ids: Sequence[str] = (),
    extra_image_ids: Mapping[str, Sequence[str]] | None = None,
) -> dict:
    """Every photograph identity that a new population must not contain.

    ``confirmation_candidate_image_ids`` is *all 64* candidates opened by the
    successful confirmation study, not the 16 that were recruited from them.
    Opening a photograph for clean capability spends it: the model's answer on
    it has been seen, and re-using it in a later population would let a known
    clean answer leak into a new recruitment.
    """

    sources = {
        "fit": list(map(str, fit_image_ids)),
        "cross_evaluation": list(map(str, eval_image_ids)),
        "prior_causal_screens": list(map(str, prior_causal_image_ids)),
        "broad_development": list(map(str, broad_development_image_ids)),
        "confirmation_candidates_all_opened": list(
            map(str, confirmation_candidate_image_ids)
        ),
        **{
            str(name): list(map(str, values))
            for name, values in (extra_image_ids or {}).items()
        },
    }
    excluded = sorted({value for values in sources.values() for value in values})
    payload = {
        "version": EXCLUSION_AUDIT_VERSION,
        "sources": {name: sorted(set(values)) for name, values in sources.items()},
        "counts_by_source": {name: len(set(values)) for name, values in sources.items()},
        "excluded_image_ids": excluded,
        "n_excluded": len(excluded),
        "spent_definition": (
            "a photograph is spent once the model has been run on it in any "
            "stage, including clean capability screening"
        ),
        "candidates_not_only_recruits": True,
    }
    return {**payload, "exclusion_digest": payload_checksum(payload)}


def artifact_exclusion_audit(
    population: Sequence[Mapping], *, universe: Mapping, label: str
) -> dict:
    """Read-only proof of which identities a population excluded, and that it did."""

    images = [str(row["image_id"]) for row in population]
    groups = [str(row.get("group_id") or "") for row in population]
    excluded = set(map(str, universe["excluded_image_ids"]))
    overlap = sorted(set(images) & excluded)
    payload = {
        "version": EXCLUSION_AUDIT_VERSION,
        "label": str(label),
        "n_population": len(images),
        "n_distinct_images": len(set(images)),
        "n_distinct_groups": len(set(groups)),
        "exclusion_digest": universe["exclusion_digest"],
        "n_excluded_identities": int(universe["n_excluded"]),
        "counts_by_source": dict(universe["counts_by_source"]),
        "overlap_with_excluded": overlap,
        "disjoint": not overlap,
        "read_only": True,
    }
    if overlap:
        raise MultimodalFollowupRefused(
            f"{label} population reuses spent photographs {overlap}"
        )
    return {**payload, "audit_digest": payload_checksum(payload)}


def followup_budget(
    *,
    stage: str,
    n_candidates: int,
    n_recruited: int,
    modalities: Sequence[str] = MODALITIES,
    conditions: Sequence[str] = REQUIRED_CONDITIONS,
    max_new_tokens: int = 1,
    n_directions: int = 1,
) -> dict:
    """Exact forward counts for a capability screen plus an intervention grid."""

    assert_controls_complete(conditions)
    tokens = max(1, int(max_new_tokens))
    capability = int(n_candidates) * len(tuple(modalities)) * tokens
    cells = int(n_recruited) * len(tuple(modalities)) * int(n_directions)
    patched = cells * len(tuple(conditions)) * tokens
    return {
        "version": FOLLOWUP_VERSION,
        "stage": str(stage),
        "n_candidates": int(n_candidates),
        "n_recruited": int(n_recruited),
        "n_directions": int(n_directions),
        "modalities": list(modalities),
        "conditions": list(conditions),
        "max_new_tokens": tokens,
        "capability_forwards": capability,
        # No separate clean pass in the intervention grid: the capability
        # screen already recorded the untouched answer for every recruited
        # photograph, and the zero condition is the paired in-grid baseline.
        "clean_forwards": 0,
        "clean_forward_note": (
            "the capability screen supplies the clean answer and the zero "
            "condition supplies the paired baseline"
        ),
        "n_intervention_cells": cells,
        "patched_forwards": patched,
        "total_forwards": capability + patched,
        "backward_passes": 0,
        "lens_fits": 0,
        "resume_unit": "one photograph x modality x condition JSON",
    }


# ------------------------------------------- B1/B3/C. verdicts and freezing


def _cell_records(
    rows: Sequence[Mapping],
    *,
    modalities: Sequence[str],
    conditions: Sequence[str],
    layers: Sequence[int],
    max_activation_norm_ratio: float,
    max_update_ratio: float,
) -> list[dict]:
    cells: list[dict] = []
    for modality in modalities:
        modality_rows = [row for row in rows if str(row.get("modality")) == modality]
        by_condition = {
            condition: sorted(
                [row for row in modality_rows if str(row.get("condition")) == condition],
                key=lambda row: str(row.get("group_id")),
            )
            for condition in conditions
        }
        exact = by_condition["exact"]
        cell = {
            "modality": modality,
            "n": len(exact),
            "exact_successes": sum(bool(row.get("success")) for row in exact),
            "exact_success_rate": _rate(exact),
            "controls": {},
            "integrity_pass": bool(exact)
            and all(
                bool(row.get("all_prompt_positions_patched"))
                and list(row.get("layers_patched") or []) == list(layers)
                for condition_rows in by_condition.values()
                for row in condition_rows
            ),
            "max_activation_norm_ratio": max(
                (float(row.get("max_activation_norm_ratio", 1.0)) for row in exact),
                default=1.0,
            ),
            "max_update_to_activation_norm_ratio": max(
                (
                    float(row.get("max_update_to_activation_norm_ratio", 0.0))
                    for row in exact
                ),
                default=0.0,
            ),
        }
        cell["activation_norms_sane"] = (
            cell["max_activation_norm_ratio"] <= float(max_activation_norm_ratio)
            and cell["max_update_to_activation_norm_ratio"] <= float(max_update_ratio)
        )
        for control in CONTROL_CONDITIONS:
            control_rows = by_condition[control]
            rate = _rate(control_rows)
            cell["controls"][control] = {
                "n": len(control_rows),
                "successes": sum(bool(row.get("success")) for row in control_rows),
                "success_rate": rate,
                "exact_minus_control": cell["exact_success_rate"] - rate,
            }
        cells.append(cell)
    return cells


def failure_mode(
    cells: Sequence[Mapping],
    *,
    min_success_rate: float,
    min_control_margin: float,
) -> str:
    """Name why a set of cells failed, so a null is never confused with a bug.

    A run where the controls moved the answer too is not a null result about
    the hypothesis; it is a broken instrument, and the two must never share a
    verdict string.
    """

    if not cells or any(not cell["n"] for cell in cells):
        return "no_trials"
    if not all(cell["integrity_pass"] for cell in cells):
        return "coordinate_integrity_failed"
    if not all(cell["activation_norms_sane"] for cell in cells):
        return "activation_norms_out_of_range"
    controls_moved = any(
        control["exact_minus_control"] < float(min_control_margin)
        and control["success_rate"] > 0.0
        for cell in cells
        for control in cell["controls"].values()
    )
    effect_present = all(
        cell["exact_success_rate"] >= float(min_success_rate) for cell in cells
    )
    if controls_moved:
        return "controls_also_moved_the_answer"
    if not effect_present:
        return "no_effect_in_every_modality"
    if not all(
        control["exact_minus_control"] >= float(min_control_margin)
        for cell in cells
        for control in cell["controls"].values()
    ):
        return "control_margin_too_small"
    return "none"


def generation_trial_row(
    trial: Mapping,
    *,
    group: Mapping,
    modality: str,
    condition: str,
    direction: Sequence[str],
    answer: PropertyAnswer | Mapping,
    layers: Sequence[int] = VALIDATED_BAND,
) -> dict:
    """Flatten one complete-generation swap trial into the scored row schema.

    The generation path reports its coordinate and activation diagnostics per
    layer under ``intervention_diagnostics``; the verdict functions read flat
    worst-case fields. This is the only place that mapping happens, so a
    renamed diagnostic breaks one function rather than four notebook cells.
    """

    diagnostics = dict(trial.get("intervention_diagnostics") or {})
    by_layer = list(diagnostics.get("by_layer") or [])

    def _worst(key: str, default: float) -> float:
        values = [
            float(row[key])
            for row in by_layer
            if isinstance(row.get(key), (int, float))
        ]
        return max(values) if values else float(default)

    generated = str(trial.get("generated_text") or "")
    aliases = (
        list(answer.aliases)
        if isinstance(answer, PropertyAnswer)
        else list(answer.get("aliases") or [])
    )
    return {
        "direction": f"{direction[0]}->{direction[1]}",
        "source": str(direction[0]),
        "target": str(direction[1]),
        "group_id": str(group["group_id"]),
        "image_id": str(group["image_id"]),
        "modality": str(modality),
        "condition": str(condition),
        "alpha": float(trial.get("alpha", 0.0)),
        "layers_patched": sorted(int(layer) for layer in trial.get("layers_patched") or layers),
        "all_prompt_positions_patched": bool(
            trial.get("all_prompt_positions_patched")
        ),
        "n_forward_passes": int(trial.get("n_forward_passes") or 0),
        "generated_text": generated,
        "expected_aliases": aliases,
        "expected": aliases[0] if aliases else "",
        "success": property_answer_matches(generated, answer),
        "max_activation_norm_ratio": _worst(
            "max_after_to_before_activation_ratio", 1.0
        ),
        "max_update_to_activation_norm_ratio": _worst(
            "max_update_to_activation_ratio", 0.0
        ),
        "max_orthogonal_residual_drift": float(
            diagnostics.get("max_post_cast_relative_residual_drift") or 0.0
        ),
        "max_coordinate_update_error": float(
            diagnostics.get("max_post_cast_relative_coordinate_error") or 0.0
        ),
        "all_hooks_fired": bool(diagnostics.get("all_hooks_fired")),
        "teacher_forcing_used": False,
        "candidate_list_supplied": False,
    }


def new_property_development_verdict(
    rows: Sequence[Mapping],
    *,
    audit: Mapping,
    layers: Sequence[int] = VALIDATED_BAND,
    capability_go: bool,
    min_success_rate: float = 0.50,
    min_control_margin: float = 0.25,
    max_activation_norm_ratio: float = 1.25,
    max_update_ratio: float = 0.50,
    modalities: Sequence[str] = MODALITIES,
) -> dict:
    """Score Stage B1. Development only; a pass licenses freezing, nothing else."""

    assert_controls_complete(REQUIRED_CONDITIONS)
    directions = sorted({str(row.get("direction")) for row in rows if row.get("direction")})
    per_direction = []
    for direction in directions:
        direction_rows = [row for row in rows if str(row.get("direction")) == direction]
        cells = _cell_records(
            direction_rows,
            modalities=modalities,
            conditions=REQUIRED_CONDITIONS,
            layers=layers,
            max_activation_norm_ratio=max_activation_norm_ratio,
            max_update_ratio=max_update_ratio,
        )
        passed = bool(cells) and all(
            cell["n"]
            and cell["exact_success_rate"] >= float(min_success_rate)
            and all(
                control["exact_minus_control"] >= float(min_control_margin)
                for control in cell["controls"].values()
            )
            and cell["integrity_pass"]
            and cell["activation_norms_sane"]
            for cell in cells
        )
        per_direction.append(
            {
                "direction": direction,
                "cells": cells,
                "passed": passed,
                "failure_mode": failure_mode(
                    cells,
                    min_success_rate=min_success_rate,
                    min_control_margin=min_control_margin,
                ),
            }
        )
    passing = [row["direction"] for row in per_direction if row["passed"]]
    control_failures = [
        row["direction"]
        for row in per_direction
        if row["failure_mode"] == "controls_also_moved_the_answer"
    ]
    payload = {
        "version": NEW_PROPERTY_DEVELOPMENT_VERSION,
        "property_family": audit["family"],
        "audit_digest": audit["audit_digest"],
        "layers": list(layers),
        "alpha": 1.0,
        "position_rule": "every original prompt position",
        "endpoint": "unrestricted complete generation",
        "teacher_forcing_used": False,
        "candidate_list_supplied": False,
        "lens_refitted": False,
        "capability_go": bool(capability_go),
        "thresholds": {
            "min_success_rate": float(min_success_rate),
            "min_control_margin": float(min_control_margin),
            "max_activation_norm_ratio": float(max_activation_norm_ratio),
            "max_update_to_activation_norm_ratio": float(max_update_ratio),
            "predeclared_before_outcomes": True,
        },
        "directions": per_direction,
        "passing_directions": passing,
        "control_failure_directions": control_failures,
        "failure_modes": {
            row["direction"]: row["failure_mode"] for row in per_direction
        },
        "claim_boundary": (
            "development on a fresh-but-now-spent population. A passing "
            "direction may be frozen and repeated; it may not be reported as a "
            "confirmed generalization of the workspace result."
        ),
    }
    payload["verdict"] = (
        "NEW_PROPERTY_DEVELOPMENT_CAPABILITY_NO_GO"
        if not capability_go
        else "NEW_PROPERTY_DEVELOPMENT_GO"
        if passing
        else "NEW_PROPERTY_DEVELOPMENT_CONTROL_FAILURE"
        if control_failures
        else "NEW_PROPERTY_DEVELOPMENT_NO_GO"
    )
    return {**payload, "report_checksum": payload_checksum(payload)}


def freeze_new_property_design(
    *,
    development: Mapping,
    audit: Mapping,
    direction: Sequence[str],
    lens_checksum: str,
    layers: Sequence[int] = VALIDATED_BAND,
    alpha: float = 1.0,
    position_rule: str = "every original prompt position",
    exclusions: Mapping,
    n_candidates: int,
    n_recruited: int,
    min_success_rate: float,
    min_control_margin: float,
    min_clean_capability_rate: float,
    familywise_alpha: float,
    recruitment_rule: str,
    seed: str,
) -> dict:
    """Stage B2. Write this before a single fresh photograph is opened.

    Refuses to freeze anything development did not license, and refuses a pair
    whose two concepts share a property answer.
    """

    if development.get("verdict") != "NEW_PROPERTY_DEVELOPMENT_GO":
        raise MultimodalFollowupRefused(
            "confirmation cannot be frozen: development returned "
            f"{development.get('verdict')!r}"
        )
    name = f"{direction[0]}->{direction[1]}"
    if name not in list(development.get("passing_directions") or []):
        raise MultimodalFollowupRefused(
            f"{name} is not among the directions development licensed"
        )
    pair_record = assert_property_pair_changes_answer(
        str(audit["family"]), str(direction[0]), str(direction[1]),
        resolved={row["concept"]: row for row in audit.get("concepts") or ()},
    )
    spec = PROPERTY_FAMILIES[str(audit["family"])]
    payload = {
        "version": NEW_PROPERTY_FREEZE_VERSION,
        "frozen_before_fresh_population_opened": True,
        "property_family": spec.name,
        "prompt_id": audit.get("prompt_id", "baseline_v1"),
        "prompt": audit.get("question", spec.question),
        "prompt_by_modality": dict(audit["prompt_by_modality"]),
        "prompt_screen_report_checksum": audit.get(
            "prompt_screen_report_checksum"
        ),
        "max_new_tokens": int(spec.max_new_tokens),
        "answer_normalization": audit["answer_normalization"],
        "answer_aliases": {
            str(direction[0]): list(pair_record["source_answer"]["aliases"]),
            str(direction[1]): list(pair_record["target_answer"]["aliases"]),
        },
        "direction": list(direction),
        "pair_changes_property": True,
        "lens_checksum": str(lens_checksum),
        "lens_refitted": False,
        "layers": list(layers),
        "alpha": float(alpha),
        "position_rule": str(position_rule),
        "controls": list(CONTROL_CONDITIONS),
        "conditions": list(REQUIRED_CONDITIONS),
        "endpoint": "unrestricted complete generation",
        "teacher_forcing_used": False,
        "candidate_list_supplied": False,
        "thresholds": {
            "min_success_rate": float(min_success_rate),
            "min_control_margin": float(min_control_margin),
            "min_clean_capability_rate": float(min_clean_capability_rate),
        },
        "statistical_tests": "exact one-sided paired sign test, exact vs each control",
        "familywise_correction": "Holm across 3 modalities x 3 controls",
        "familywise_alpha": float(familywise_alpha),
        "n_candidates": int(n_candidates),
        "n_recruited": int(n_recruited),
        "recruitment_rule": str(recruitment_rule),
        "capability_required_in_every_modality": True,
        "fresh_population_rule": (
            "select candidates before opening any clean answer; exclude every "
            "identity in the exclusion digest below"
        ),
        "exclusion_digest": exclusions["exclusion_digest"],
        "n_excluded_identities": int(exclusions["n_excluded"]),
        "exclusion_counts_by_source": dict(exclusions["counts_by_source"]),
        "development_report_checksum": development["report_checksum"],
        "audit_digest": audit["audit_digest"],
        "seed": str(seed),
        "revision_after_outcomes_forbidden": [
            "thresholds",
            "prompt",
            "pair",
            "answer aliases",
            "recruitment rule",
            "layers",
            "alpha",
            "controls",
        ],
    }
    assert_lens_reused_not_refitted(payload)
    assert_open_endpoint(payload)
    return {**payload, "design_digest": payload_checksum(payload)}


def assert_design_frozen(
    design_path: str | Path, *, expected_digest: str | None = None
) -> dict:
    """Refuse to open confirmation unless a valid frozen design already exists."""

    path = Path(design_path)
    if not path.is_file():
        raise MultimodalFollowupRefused(
            f"no frozen design at {path}; Stage B2 must run and persist before "
            "any fresh confirmation photograph is opened"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded = str(payload.get("design_digest") or "")
    body = {key: value for key, value in payload.items() if key != "design_digest"}
    recomputed = payload_checksum(body)
    if recorded != recomputed:
        raise MultimodalFollowupRefused(
            f"the frozen design at {path} was edited after it was written"
        )
    if expected_digest is not None and recorded != expected_digest:
        raise MultimodalFollowupRefused(
            f"frozen design digest {recorded} does not match the pinned "
            f"{expected_digest}"
        )
    if not payload.get("frozen_before_fresh_population_opened"):
        raise MultimodalFollowupRefused(
            "the design does not record that it was frozen before the fresh "
            "population was opened"
        )
    return payload


def confirmation_verdict(
    rows: Sequence[Mapping],
    *,
    design: Mapping,
    capability_go: bool,
    exclusion_audit: Mapping,
    version: str = NEW_PROPERTY_CONFIRMATION_VERSION,
    modalities: Sequence[str] = MODALITIES,
    max_activation_norm_ratio: float = 1.25,
    max_update_ratio: float = 0.50,
    go_verdict: str = "NEW_PROPERTY_CONFIRMATION_GO",
    no_go_verdict: str = "NEW_PROPERTY_CONFIRMATION_NO_GO",
    capability_no_go_verdict: str = "NEW_PROPERTY_CONFIRMATION_CAPABILITY_NO_GO",
) -> dict:
    """Score a fresh confirmation strictly under an already-frozen design."""

    assert_controls_complete(design["conditions"])
    assert_open_endpoint(design)
    if not capability_go or not rows:
        # Recruitment failed, so no intervention ran and there is nothing to
        # pair. Report the capability verdict rather than falling through to
        # the pairing check, which would raise a misleading "missing control"
        # error for what is really an ordinary, expected outcome.
        payload = {
            "version": version,
            "design_digest": design["design_digest"],
            "direction": list(design["direction"]),
            "property_family": design.get("property_family"),
            "layers": list(design["layers"]),
            "alpha": float(design["alpha"]),
            "lens_refitted": False,
            "teacher_forcing_used": False,
            "candidate_list_supplied": False,
            "exclusion_audit": dict(exclusion_audit),
            "cells": [],
            "paired_comparisons": [],
            "gate": {
                "design_was_frozen_first": bool(
                    design.get("frozen_before_fresh_population_opened")
                ),
                "population_fresh": bool(exclusion_audit.get("disjoint")),
                "capability_in_every_modality": False,
            },
            "rows": [dict(row) for row in rows],
            "design_altered_after_outcomes": False,
            "failure_mode": "no_trials",
            "verdict": capability_no_go_verdict,
        }
        return {**payload, "report_checksum": payload_checksum(payload)}
    cells = _cell_records(
        rows,
        modalities=modalities,
        conditions=REQUIRED_CONDITIONS,
        layers=design["layers"],
        max_activation_norm_ratio=max_activation_norm_ratio,
        max_update_ratio=max_update_ratio,
    )
    comparisons: list[dict] = []
    for cell in cells:
        modality_rows = [
            row for row in rows if str(row.get("modality")) == cell["modality"]
        ]
        by_condition = {
            condition: sorted(
                [row for row in modality_rows if str(row.get("condition")) == condition],
                key=lambda row: str(row.get("group_id")),
            )
            for condition in REQUIRED_CONDITIONS
        }
        exact = [bool(row.get("success")) for row in by_condition["exact"]]
        for control in CONTROL_CONDITIONS:
            control_outcomes = [
                bool(row.get("success")) for row in by_condition[control]
            ]
            if len(control_outcomes) != len(exact) or not exact:
                raise MultimodalFollowupRefused(
                    f"{cell['modality']}/{control} is not paired with the exact "
                    "condition; a missing control is refused, not ignored"
                )
            comparisons.append(
                {
                    "modality": cell["modality"],
                    "control": control,
                    **paired_binary_one_sided_p(exact, control_outcomes),
                }
            )
    adjusted = holm_adjust(comparisons)
    for comparison in adjusted:
        for cell in cells:
            if cell["modality"] == comparison["modality"]:
                cell["controls"][comparison["control"]]["paired_test"] = comparison

    thresholds = dict(design["thresholds"])
    gate = {
        "design_was_frozen_first": bool(
            design.get("frozen_before_fresh_population_opened")
        ),
        "population_fresh": bool(exclusion_audit.get("disjoint")),
        "capability_in_every_modality": bool(capability_go),
        "success_rate_in_every_modality": bool(cells)
        and all(
            cell["exact_success_rate"] >= float(thresholds["min_success_rate"])
            for cell in cells
        ),
        "control_margin_in_every_comparison": bool(cells)
        and all(
            control["exact_minus_control"] >= float(thresholds["min_control_margin"])
            for cell in cells
            for control in cell["controls"].values()
        ),
        "holm_passing_in_every_comparison": bool(adjusted)
        and all(
            float(row["holm_adjusted_p"]) <= float(design["familywise_alpha"])
            for row in adjusted
        ),
        "coordinate_integrity_in_every_modality": bool(cells)
        and all(cell["integrity_pass"] for cell in cells),
        "activation_norms_sane": bool(cells)
        and all(cell["activation_norms_sane"] for cell in cells),
    }
    payload = {
        "version": version,
        "design_digest": design["design_digest"],
        "direction": list(design["direction"]),
        "property_family": design.get("property_family"),
        "layers": list(design["layers"]),
        "alpha": float(design["alpha"]),
        "lens_refitted": False,
        "teacher_forcing_used": False,
        "candidate_list_supplied": False,
        "exclusion_audit": dict(exclusion_audit),
        "cells": cells,
        "paired_comparisons": adjusted,
        "gate": gate,
        "rows": [dict(row) for row in rows],
        "design_altered_after_outcomes": False,
    }
    payload["failure_mode"] = failure_mode(
        cells,
        min_success_rate=float(thresholds["min_success_rate"]),
        min_control_margin=float(thresholds["min_control_margin"]),
    )
    payload["verdict"] = (
        capability_no_go_verdict
        if not capability_go
        else go_verdict
        if all(gate.values())
        else no_go_verdict
    )
    return {**payload, "report_checksum": payload_checksum(payload)}


def asymmetry_replication_design(
    *,
    lens_checksum: str,
    exclusions: Mapping,
    layers: Sequence[int] = VALIDATED_BAND,
    alpha: float = 1.0,
    n_candidates: int = 64,
    n_recruited: int = 16,
    min_success_rate: float = 0.75,
    min_control_margin: float = 0.25,
    min_clean_capability_rate: float = 0.75,
    familywise_alpha: float = 0.05,
    seed: str = "multimodal-asymmetry-replication-v1",
) -> dict:
    """Experiment C. The same protocol as bird->cat, run backwards on fresh cats.

    This is framed as a replication test of a development *difference*, and the
    two possible outcomes are asymmetric in what they license:

    * few or no successes replicate the observed development failure. That is
      consistent with a real asymmetry and says nothing about why one exists;
      capability, prompt behaviour, coordinate quality and concept geometry
      remain live alternatives.
    * a clear effect shows the development failure did not replicate, and the
      apparent asymmetry should not be reported at all.
    """

    payload = {
        "version": ASYMMETRY_VERSION,
        "direction": ["cat", "bird"],
        "reference_direction": ["bird", "cat"],
        "development_record": development_direction_record(),
        "property": "leg_count",
        "prompt": (
            "How many legs does the animal in the evidence typically have? "
            "Answer with one digit.\nAnswer:"
        ),
        "source_answer": LEG_COUNT_ANSWERS["cat"],
        "target_answer": LEG_COUNT_ANSWERS["bird"],
        "endpoint": "unrestricted full-vocabulary next-token top1",
        "teacher_forcing_used": False,
        "candidate_list_supplied": False,
        "lens_checksum": str(lens_checksum),
        "lens_refitted": False,
        "layers": list(layers),
        "alpha": float(alpha),
        "position_rule": "every original prompt position",
        "conditions": list(REQUIRED_CONDITIONS),
        "controls": list(CONTROL_CONDITIONS),
        "n_candidates": int(n_candidates),
        "n_recruited": int(n_recruited),
        "media": "fresh cat photographs with their caption and spoken caption",
        "thresholds": {
            "min_success_rate": float(min_success_rate),
            "min_control_margin": float(min_control_margin),
            "min_clean_capability_rate": float(min_clean_capability_rate),
        },
        "familywise_alpha": float(familywise_alpha),
        "exclusion_digest": exclusions["exclusion_digest"],
        "n_excluded_identities": int(exclusions["n_excluded"]),
        "seed": str(seed),
        "frozen_before_fresh_population_opened": True,
        "interpretation": {
            "failure_means": (
                "the development failure replicated; this supports the "
                "existence of an asymmetry and explains nothing about its cause"
            ),
            "success_means": (
                "the development failure did not replicate; the apparent "
                "asymmetry should not be reported"
            ),
            "cause_identified": False,
        },
        "familywise_correction": "Holm across 3 modalities x 3 controls",
        "statistical_tests": "exact one-sided paired sign test, exact vs each control",
        "recruitment_rule": "clean leg-count capability in all three modalities",
        "revision_after_outcomes_forbidden": [
            "thresholds",
            "prompt",
            "pair",
            "recruitment rule",
        ],
    }
    assert_lens_reused_not_refitted(payload)
    assert_open_endpoint(payload)
    return {**payload, "design_digest": payload_checksum(payload)}


def asymmetry_replication_verdict(
    rows: Sequence[Mapping],
    *,
    design: Mapping,
    capability_go: bool,
    exclusion_audit: Mapping,
    modalities: Sequence[str] = MODALITIES,
) -> dict:
    """Score Experiment C and phrase the outcome as a replication test."""

    report = confirmation_verdict(
        rows,
        design=design,
        capability_go=capability_go,
        exclusion_audit=exclusion_audit,
        version=ASYMMETRY_VERSION,
        modalities=modalities,
        go_verdict="ASYMMETRY_DID_NOT_REPLICATE_REVERSE_EFFECT_FOUND",
        no_go_verdict="ASYMMETRY_REPLICATED_NO_REVERSE_EFFECT",
        capability_no_go_verdict="ASYMMETRY_REPLICATION_CAPABILITY_NO_GO",
    )
    body = {key: value for key, value in report.items() if key != "report_checksum"}
    successes = sum(int(cell["exact_successes"]) for cell in report["cells"])
    trials = sum(int(cell["n"]) for cell in report["cells"])
    body.update(
        {
            "framing": "asymmetry replication test",
            "development_record": development_direction_record(),
            "reverse_successes": successes,
            "reverse_trials": trials,
            "interpretation": dict(design["interpretation"]),
            "cause_of_asymmetry_identified": False,
            "claim_boundary": (
                "a null here replicates a development difference; it does not "
                "show that the representation is inherently asymmetric, and a "
                "positive result would retire the asymmetry claim entirely"
            ),
        }
    )
    return {**body, "report_checksum": payload_checksum(body)}


def stage_map() -> dict:
    """The printed stage map: what runs, on which population, at what cost."""

    return {
        "version": FOLLOWUP_VERSION,
        "stages": [
            {
                "stage": "A",
                "name": "exploratory band localization",
                "population": "spent broad development population",
                "label": "exploratory/descriptive",
                "fits": 0,
                "new_media": 0,
                "confirms": False,
            },
            {
                "stage": "B00",
                "name": "animal-sound prompt screen",
                "population": "already-spent B0 development media",
                "label": "outcome-informed prompt development",
                "fits": 0,
                "confirms": False,
            },
            {
                "stage": "B0",
                "name": "property and prompt audit",
                "population": "fresh candidates, clean model only",
                "label": "audit",
                "fits": 0,
                "confirms": False,
            },
            {
                "stage": "B1",
                "name": "new-property development",
                "population": "fresh development media, disjoint from all prior",
                "label": "development",
                "fits": 0,
                "confirms": False,
            },
            {
                "stage": "B2",
                "name": "freeze",
                "population": "none",
                "label": "artifact",
                "fits": 0,
                "confirms": False,
            },
            {
                "stage": "B3",
                "name": "fresh confirmation",
                "population": "untouched, excludes all 64 confirmation candidates",
                "label": "confirmation",
                "fits": 0,
                "confirms": True,
            },
            {
                "stage": "C",
                "name": "prospective asymmetry replication",
                "population": "fresh cat media",
                "label": "prospective replication",
                "fits": 0,
                "confirms": False,
            },
        ],
        "never": [
            "refit the pooled lens",
            "teacher forcing",
            "a candidate list",
            "dropping a control",
            "changing a threshold after seeing an outcome",
            "reusing a spent photograph in a confirmation population",
        ],
    }


__all__ = [
    "ANIMAL_SOUND_PROMPT_CANDIDATES",
    "ASYMMETRY_VERSION",
    "CONTROL_CONDITIONS",
    "EXCLUSION_AUDIT_VERSION",
    "FOLLOWUP_VERSION",
    "LEG_COUNT_ANSWERS",
    "LOCALIZATION_VERSION",
    "LocalizationBand",
    "MultimodalFollowupRefused",
    "NEW_PROPERTY_CONFIRMATION_VERSION",
    "NEW_PROPERTY_DEVELOPMENT_VERSION",
    "NEW_PROPERTY_FREEZE_VERSION",
    "ORIGINAL_DEVELOPMENT_DIRECTIONS",
    "POOLED_ONLY_BAND_NOTE",
    "PROPERTY_AUDIT_VERSION",
    "PROPERTY_PROMPT_SCREEN_CONCEPTS",
    "PROPERTY_PROMPT_SCREEN_VERSION",
    "PROPERTY_FAMILIES",
    "PropertyAnswer",
    "PropertyFamily",
    "REQUIRED_CONDITIONS",
    "VALIDATED_BAND",
    "artifact_exclusion_audit",
    "assert_controls_complete",
    "assert_design_frozen",
    "assert_lens_reused_not_refitted",
    "assert_open_endpoint",
    "DOMINANT_ANSWER_RULE",
    "answer_key",
    "assert_property_pair_changes_answer",
    "asymmetry_replication_design",
    "asymmetry_replication_verdict",
    "audit_property_family",
    "bands_are_nested_chain",
    "confirmation_verdict",
    "development_direction_record",
    "exclusion_universe",
    "failure_mode",
    "followup_budget",
    "freeze_new_property_design",
    "generation_trial_row",
    "leg_count_property_limit",
    "load_extra_spent_image_ids",
    "load_localization_population",
    "load_spent_confirmation_population",
    "load_verified_report",
    "localization_budget",
    "localization_claim_boundary",
    "localization_grid",
    "new_property_development_verdict",
    "property_prompt",
    "property_prompt_candidate",
    "property_prompt_screen_verdict",
    "resolve_dominant_answer",
    "stage_map",
    "summarize_localization",
]
