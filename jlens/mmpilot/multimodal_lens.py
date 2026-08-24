# SPDX-License-Identifier: Apache-2.0
"""Modality-matched Jacobian-lens fitting and cross evaluation.

The upstream fitter accepts strings and tokenizes them with the language-model
tokenizer.  That is exactly right for a text-only J-lens and exactly wrong for
the experiment in this module: image and spoken-audio examples must pass
through the checkpoint's real processor and modality towers before the
decoder Jacobian is measured.

This module keeps the upstream estimator unchanged.  It only replaces the
``model.encode(prompt)`` boundary with a :class:`~jlens.mmpilot.backend.BuiltInputs`
object produced by the audited multimodal backend.  For every valid decoder
position it estimates the same average ``d h_final / d h_l`` and stores the
same fp32 ``[d_model, d_model]`` matrices.

Four equal-size arms are supported by design: text, image, spoken audio, and a
pooled arm whose photographs are assigned as evenly as possible across the
three modalities.  Each arm has its own fingerprint-bound checkpoint.  A
changed population, order, media checksum, processor protocol, layer grid, or
estimator configuration refuses resume instead of mixing accumulators.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from jlens.fitting import valid_position_mask
from jlens.hooks import ActivationRecorder
from jlens.lens import JacobianLens
from jlens.metadata import file_sha256
from jlens.mmpilot.coordinate_swap import (
    ConceptToken,
    ModelDtypeRealizationPolicy,
    SwapBasis,
    build_swap_basis_from_vectors,
    coordinate_swap_band,
)
from jlens.mmpilot.store import payload_checksum

MODALITIES = ("text", "image", "spoken_audio")
LENS_ARMS = ("text", "image", "spoken_audio", "pooled")
ESTIMATOR_VERSION = "mmpilot.multimodal_average_jacobian.v1"
PLAN_VERSION = "mmpilot.matched_multimodal_lens_plan.v1"
CROSS_EVAL_VERSION = "mmpilot.multimodal_lens_cross_eval.v1"
PRIMARY_POSITION_RULE = "all_prompt_positions"
ANSWER_EQUIVALENCE_VERSION = (
    "mmpilot.open_answer_equivalence.casefold_whitespace.v1"
)
CAUSAL_SOURCE_VERSION = "mmpilot.matched_multimodal_causal_source.v1"
ALPHA_SWEEP_SOURCE_VERSION = "mmpilot.matched_multimodal_alpha_sweep_source.v1"
BROAD_CONFIRMATION_SOURCE_VERSION = (
    "mmpilot.broad_pooled_multimodal_confirmation_source.v1"
)


class MultimodalLensRefused(RuntimeError):
    """The requested run would mix or mislabel scientific inputs."""


def normalize_open_answer_surface(value: str) -> str:
    """Normalize only tokenizer/case aliases declared before causal sampling.

    This deliberately does not remove punctuation, singularize words, or map
    species names onto a parent category.  It licenses only Unicode
    normalization, case folding, and whitespace equivalence, which are
    properties of surface realization rather than semantic relabeling.
    """

    normalized = unicodedata.normalize("NFKC", str(value)).casefold()
    return " ".join(normalized.split())


def open_answer_matches(observed: str, expected: str) -> bool:
    """Whether two one-token answer surfaces are equivalent under v1."""

    return normalize_open_answer_surface(observed) == normalize_open_answer_surface(
        expected
    )


def answer_equivalence_record() -> dict:
    """Machine-readable boundary for the prospective causal follow-up."""

    payload = {
        "version": ANSWER_EQUIVALENCE_VERSION,
        "unicode_normalization": "NFKC",
        "case_sensitive": False,
        "whitespace_rule": "strip_and_collapse",
        "punctuation_removed": False,
        "semantic_aliases": [],
        "plural_or_taxonomy_mapping": False,
    }
    return {**payload, "protocol_digest": payload_checksum(payload)}


def _verified_payload(path: Path, *, expected_checksum: str, label: str) -> dict:
    if not path.is_file():
        raise MultimodalLensRefused(f"missing {label}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MultimodalLensRefused(f"could not read {label}: {path}") from exc
    recorded = str(payload.get("report_checksum") or "")
    body = {key: value for key, value in payload.items() if key != "report_checksum"}
    recomputed = payload_checksum(body)
    if recorded != expected_checksum or recomputed != expected_checksum:
        raise MultimodalLensRefused(
            f"{label} checksum mismatch: recorded={recorded!r}, "
            f"recomputed={recomputed!r}, expected={expected_checksum!r}"
        )
    return payload


def load_completed_causal_source(
    run_dir: str | Path,
    *,
    expected_final_report_checksum: str,
    expected_cross_report_checksum: str,
    expected_causal_report_checksum: str,
    expected_lens_checksums: Mapping[str, str],
) -> dict:
    """Verify the completed four-lens run reused by a fresh causal follow-up.

    The old clean-screen photographs are harvested from the completed causal
    report and become mandatory exclusions.  Lens files are read in place and
    checksum-pinned; they are never refitted or copied into the new run.
    """

    root = Path(run_dir)
    final = _verified_payload(
        root / "matched_multimodal_jlens_report.json",
        expected_checksum=expected_final_report_checksum,
        label="completed matched-multimodal report",
    )
    causal = _verified_payload(
        root / "multimodal_lens_causal_comparison_report.json",
        expected_checksum=expected_causal_report_checksum,
        label="completed causal report",
    )
    cross_path = root / "multimodal_lens_cross_eval_report.json"
    _cross = _verified_payload(
        cross_path,
        expected_checksum=expected_cross_report_checksum,
        label="completed cross-evaluation report",
    )
    if causal.get("verdict") != "CAPABILITY_NO_GO":
        raise MultimodalLensRefused(
            "the pinned source causal report is not the completed "
            "CAPABILITY_NO_GO run"
        )
    if (final.get("causal_comparison") or {}).get("report_checksum") != (
        expected_causal_report_checksum
    ):
        raise MultimodalLensRefused(
            "the final report does not embed the pinned causal report"
        )
    if (final.get("cross_evaluation") or {}).get("report_checksum") != (
        expected_cross_report_checksum
    ):
        raise MultimodalLensRefused(
            "the final report does not embed the pinned cross-evaluation report"
        )

    recorded_checksums = dict(final.get("lens_checksums") or {})
    if set(recorded_checksums) != set(LENS_ARMS):
        raise MultimodalLensRefused(
            f"the source report does not record all four lens arms: "
            f"{sorted(recorded_checksums)}"
        )
    lens_paths: dict[str, str] = {}
    for arm in LENS_ARMS:
        expected = str(expected_lens_checksums.get(arm) or "")
        recorded = str(recorded_checksums.get(arm) or "")
        path = root / "lenses" / f"lens.{arm}.pt"
        observed = file_sha256(str(path)) if path.is_file() else "missing"
        if not expected or recorded != expected or observed != expected:
            raise MultimodalLensRefused(
                f"source lens {arm!r} is not checksum-pinned: "
                f"recorded={recorded!r}, observed={observed!r}, "
                f"expected={expected!r}"
            )
        lens_paths[arm] = str(path)

    clean_screen = list(causal.get("clean_screen") or [])
    excluded_image_ids = sorted(
        {
            str(row.get("image_id"))
            for row in clean_screen
            if str(row.get("image_id") or "").strip()
        }
    )
    if not excluded_image_ids:
        raise MultimodalLensRefused(
            "the completed causal report records no screened image identities"
        )
    payload = {
        "version": CAUSAL_SOURCE_VERSION,
        "run_dir": str(root),
        "source_scientific_fingerprint": final.get("scientific_fingerprint"),
        "final_report_checksum": expected_final_report_checksum,
        "cross_report_checksum": expected_cross_report_checksum,
        "cross_report_path": str(cross_path),
        "causal_report_checksum": expected_causal_report_checksum,
        "lens_checksums": recorded_checksums,
        "lens_paths": lens_paths,
        "excluded_image_ids": excluded_image_ids,
        "n_excluded_images": len(excluded_image_ids),
    }
    return {
        **payload,
        "source_digest": payload_checksum(payload),
        # Added outside the historical source digest so existing checksum-pinned
        # alpha-sweep provenance remains byte-for-byte compatible.
        "fit_plan_digest": (final.get("scientific_config") or {}).get(
            "plan_digest"
        ),
    }


def load_completed_alpha_sweep_source(
    run_dir: str | Path,
    *,
    expected_final_report_checksum: str,
    expected_causal_report_checksum: str,
    expected_scientific_fingerprint: str,
    expected_lens_checksums: Mapping[str, str],
    expected_lens_source_digest: str,
) -> dict:
    """Verify the completed alpha=1 population reused by a dose-response run.

    This is deliberately a *sensitivity* source, not a new confirmation
    population.  The completed run fixed the 16 clean-capable photographs
    before any alpha other than one was measured.  Reusing exactly those
    photographs gives the alpha sweep paired power without silently recruiting
    a population that happens to respond at a preferred strength.
    """

    root = Path(run_dir)
    final = _verified_payload(
        root / "matched_multimodal_jlens_report.json",
        expected_checksum=expected_final_report_checksum,
        label="completed alpha=1 matched-multimodal report",
    )
    causal = _verified_payload(
        root / "multimodal_lens_causal_comparison_report.json",
        expected_checksum=expected_causal_report_checksum,
        label="completed alpha=1 causal report",
    )
    problems: list[str] = []
    if final.get("scientific_fingerprint") != expected_scientific_fingerprint:
        problems.append("scientific fingerprint differs from the pinned alpha=1 run")
    if (final.get("causal_comparison") or {}).get("report_checksum") != (
        expected_causal_report_checksum
    ):
        problems.append("the final report does not embed the pinned causal report")
    if dict(final.get("lens_checksums") or {}) != dict(expected_lens_checksums):
        problems.append("the final report's lens checksums differ from the pinned lenses")
    if causal.get("protocol") != "matched_multimodal_jlens_unrestricted_swap.v3":
        problems.append("the source is not the unrestricted alpha=1 protocol v3")
    if causal.get("verdict") != "MEASURED":
        problems.append("the source causal run did not reach its measured endpoint")
    if float(causal.get("primary_alpha", float("nan"))) != 1.0:
        problems.append("the source causal run is not alpha=1")
    if bool(causal.get("teacher_forcing_used")):
        problems.append("the source used teacher forcing")
    if bool(causal.get("candidate_list_supplied")):
        problems.append("the source supplied a candidate list")
    if list(causal.get("arms_compared") or []) != ["text", "pooled"]:
        problems.append("the source did not compare the text and pooled arms")
    if set(causal.get("controls") or []) != {"random", "unrelated"}:
        problems.append("the source lacks the random and unrelated controls")
    if dict(causal.get("recruited_counts") or {}) != {"bird": 8, "cat": 8}:
        problems.append("the source did not freeze eight clean-capable images per concept")
    if not bool(causal.get("clean_capability_required_in_every_modality_and_endpoint")):
        problems.append("the source did not require clean capability in all six cells")
    equivalence = dict(causal.get("answer_equivalence") or {})
    if equivalence.get("version") != ANSWER_EQUIVALENCE_VERSION:
        problems.append("the source used a different open-answer equivalence rule")
    if equivalence.get("semantic_aliases") or equivalence.get("punctuation_removed"):
        problems.append("the source answer rule included semantic or punctuation aliases")
    fresh = dict(causal.get("fresh_population") or {})
    if int(fresh.get("candidate_count_per_concept", -1)) != 96:
        problems.append("the source did not screen 96 candidates per concept")
    if int(fresh.get("excluded_previous_screen_images", -1)) != 64:
        problems.append("the source did not exclude all 64 prior-screen photographs")
    provenance = dict(causal.get("source_run_provenance") or {})
    if provenance.get("source_digest") != expected_lens_source_digest:
        problems.append("the source does not point to the pinned four-lens run")
    if dict(provenance.get("lens_checksums") or {}) != dict(expected_lens_checksums):
        problems.append("the source provenance records different lens checksums")

    rows = list(causal.get("rows") or [])
    if len(rows) != 96:
        problems.append(f"the source records {len(rows)} causal rows, not 96")
    alpha1_exact_outcomes: list[dict] = []
    for row in rows:
        for arm in ("text", "pooled"):
            exact = dict((row.get("arms") or {}).get(arm, {}).get("exact") or {})
            if "patched_top_token_id" not in exact:
                problems.append(
                    f"a source row lacks the {arm} alpha=1 exact outcome"
                )
                continue
            alpha1_exact_outcomes.append(
                {
                    "source": str(row.get("source")),
                    "target": str(row.get("target")),
                    "group_id": str(row.get("group_id")),
                    "modality": str(row.get("modality")),
                    "prompt_kind": str(row.get("prompt_kind")),
                    "lens_arm": arm,
                    "patched_top_token_id": int(exact["patched_top_token_id"]),
                    "success": bool(exact.get("success")),
                }
            )
    if len(alpha1_exact_outcomes) != 192:
        problems.append(
            f"the source records {len(alpha1_exact_outcomes)} exact alpha=1 arm "
            "outcomes, not 192"
        )
    groups_by_source: dict[str, list[dict]] = {}
    for source in ("bird", "cat"):
        selected = [row for row in rows if row.get("source") == source]
        identities: dict[str, str] = {}
        for row in selected:
            group_id = str(row.get("group_id") or "")
            image_id = str(row.get("image_id") or "")
            if not group_id or not image_id:
                problems.append(f"a {source} row lacks a group_id or image_id")
                continue
            previous = identities.setdefault(group_id, image_id)
            if previous != image_id:
                problems.append(f"group {group_id!r} maps to multiple images")
        if len(identities) != 8:
            problems.append(
                f"the source has {len(identities)} distinct {source} groups, not 8"
            )
        for group_id in identities:
            cells = {
                (str(row.get("modality")), str(row.get("prompt_kind")))
                for row in selected
                if str(row.get("group_id")) == group_id
            }
            expected_cells = {
                (modality, prompt_kind)
                for modality in MODALITIES
                for prompt_kind in ("identity", "property")
            }
            if cells != expected_cells:
                problems.append(
                    f"group {group_id!r} does not contain all six endpoint cells"
                )
        groups_by_source[source] = [
            {"group_id": group_id, "image_id": image_id}
            for group_id, image_id in sorted(identities.items())
        ]
    if problems:
        raise MultimodalLensRefused(
            "the completed alpha=1 run cannot seed the sensitivity sweep:\n  - "
            + "\n  - ".join(problems)
        )

    payload = {
        "version": ALPHA_SWEEP_SOURCE_VERSION,
        "run_dir": str(root),
        "scientific_fingerprint": expected_scientific_fingerprint,
        "final_report_checksum": expected_final_report_checksum,
        "causal_report_checksum": expected_causal_report_checksum,
        "lens_source_digest": expected_lens_source_digest,
        "lens_checksums": dict(expected_lens_checksums),
        "groups_by_source": groups_by_source,
        "alpha1_exact_outcomes": alpha1_exact_outcomes,
        "population_digest": str(
            fresh.get("causal_population_digest")
            or ""
        ),
        "answer_equivalence": dict(causal.get("answer_equivalence") or {}),
    }
    return {**payload, "source_digest": payload_checksum(payload)}


def load_broad_pooled_development_source(
    run_dir: str | Path,
    *,
    expected_report_checksum: str,
    expected_population_digest: str,
    expected_lens_checksum: str,
    expected_direction: tuple[str, str] = ("bird", "cat"),
) -> dict:
    """Pin the sole development winner before fresh confirmation opens data."""

    root = Path(run_dir)
    report = _verified_payload(
        root / "broad_pooled_multimodal_j_workspace_report.json",
        expected_checksum=expected_report_checksum,
        label="broad pooled multimodal J-lens development report",
    )
    population_path = root / "development_population.json"
    if not population_path.is_file():
        raise MultimodalLensRefused(
            f"missing broad development population: {population_path}"
        )
    population_payload = json.loads(population_path.read_text(encoding="utf-8"))
    population = dict(population_payload.get("population") or {})
    observed_population_digest = payload_checksum(population)
    recorded_population_digest = str(
        population_payload.get("population_digest") or ""
    )
    lens_path = root / "lenses" / "lens.pooled.l16_l40.pt"
    observed_lens_checksum = (
        file_sha256(str(lens_path)) if lens_path.is_file() else "missing"
    )
    direction = f"{expected_direction[0]}->{expected_direction[1]}"
    problems: list[str] = []
    if report.get("verdict") != "BROAD_POOLED_J_DEVELOPMENT_ALPHA1_GO":
        problems.append("development did not return the alpha=1 GO verdict")
    if list(report.get("alpha1_primary_passing_directions") or []) != [direction]:
        problems.append("the sole frozen alpha=1 winner is not bird->cat")
    if report.get("alpha2_sensitivity_passing_directions"):
        problems.append("development unexpectedly records an alpha=2 winner")
    if str(report.get("population_digest") or "") != expected_population_digest:
        problems.append("report population digest differs from the pin")
    if recorded_population_digest != expected_population_digest:
        problems.append("population artifact digest differs from the pin")
    if observed_population_digest != expected_population_digest:
        problems.append("population contents no longer match their digest")
    provenance = dict(report.get("lens_provenance") or {})
    if provenance.get("combined_checksum") != expected_lens_checksum:
        problems.append("report combined-lens checksum differs from the pin")
    if observed_lens_checksum != expected_lens_checksum:
        problems.append("combined-lens file checksum differs from the pin")
    method = dict(report.get("method") or {})
    if method.get("layers") != list(range(16, 41)):
        problems.append("development did not patch the contiguous L16-L40 band")
    if method.get("positions") != "every original prompt position":
        problems.append("development used a different position rule")
    if method.get("teacher_forcing_used") or method.get("candidate_list_supplied"):
        problems.append("development used a restricted output endpoint")
    excluded_image_ids = sorted(
        {
            str(row.get("image_id"))
            for rows in population.values()
            for row in rows
            if str(row.get("image_id") or "").strip()
        }
    )
    if not excluded_image_ids:
        problems.append("development population contains no image identities")
    if problems:
        raise MultimodalLensRefused(
            "the broad development run cannot seed confirmation:\n  - "
            + "\n  - ".join(problems)
        )
    payload = {
        "version": BROAD_CONFIRMATION_SOURCE_VERSION,
        "run_dir": str(root),
        "report_checksum": expected_report_checksum,
        "population_digest": expected_population_digest,
        "lens_checksum": expected_lens_checksum,
        "lens_path": str(lens_path),
        "direction": list(expected_direction),
        "alpha": 1.0,
        "layers": list(range(16, 41)),
        "excluded_image_ids": excluded_image_ids,
        "n_excluded_images": len(excluded_image_ids),
    }
    return {**payload, "source_digest": payload_checksum(payload)}


@dataclass(frozen=True)
class FitUnit:
    """One processor input in a frozen fitting order."""

    unit_id: str
    group_id: str
    image_id: str
    modality: str
    caption: str
    image_path: str
    audio_path: str
    prompt: str

    def to_dict(self) -> dict:
        return asdict(self)


def _stable_rank(value: str, seed: str) -> str:
    return hashlib.sha256(f"{seed}|{value}".encode()).hexdigest()


def _surface_contains(text: str, concept: str) -> bool:
    words = "".join(ch.lower() if ch.isalnum() else " " for ch in text).split()
    needle = "".join(ch.lower() if ch.isalnum() else " " for ch in concept).split()
    if not needle:
        return False
    width = len(needle)
    return any(words[i : i + width] == needle for i in range(len(words) - width + 1))


def fitting_prompt(modality: str, caption: str) -> str:
    """A long, candidate-free prompt whose evidence channel is explicit.

    Text receives the written caption.  Image and audio receive no transcript;
    their media is supplied separately to the processor.  The common suffix
    keeps the textual register comparable and makes the text sequence long
    enough for the estimator's frozen attention-sink exclusion.
    """

    suffix = (
        "Examine the evidence carefully and form a concise internal summary of "
        "its main subject, setting, and action. Do not use a candidate list. "
        "Prepare to continue with one descriptive word.\nSummary:"
    )
    if modality == "text":
        return f"Evidence is a written caption.\nCaption: {caption.strip()}\n{suffix}"
    if modality == "image":
        return f"Evidence is the attached image.\n{suffix}"
    if modality == "spoken_audio":
        return f"Evidence is the attached spoken recording.\n{suffix}"
    raise ValueError(f"unknown modality {modality!r}")


def evaluation_prompt(modality: str, caption: str) -> str:
    """Open, candidate-free prompt for full-vocabulary cross evaluation."""

    question = (
        "What is the main subject of the evidence? Answer with one word.\nAnswer:"
    )
    if modality == "text":
        return f"Caption: {caption.strip()}\n{question}"
    if modality in ("image", "spoken_audio"):
        return question
    raise ValueError(f"unknown modality {modality!r}")


def _concept_fields(group: Mapping) -> list[str]:
    values: list[str] = []
    # ExpandedManifest v3 stores audited COCO object labels under
    # ``concept_annotations``.  The shorter names are retained for normalized
    # manifests and MOCK fixtures.
    for key in (
        "concept",
        "concepts",
        "categories",
        "category_names",
        "concept_annotations",
    ):
        value = group.get(key)
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, Sequence):
            values.extend(str(item) for item in value)
    return values


def _eligible_group(group: Mapping, excluded_concepts: Sequence[str]) -> bool:
    required = ("group_id", "image_id", "caption", "image_path", "audio_path")
    if any(not str(group.get(key) or "").strip() for key in required):
        return False
    haystacks = [str(group.get("caption") or ""), *_concept_fields(group)]
    return not any(
        _surface_contains(haystack, concept)
        for concept in excluded_concepts
        for haystack in haystacks
    )


def build_matched_plan(
    groups: Sequence[Mapping],
    *,
    n_fit_groups: int,
    n_eval_groups: int,
    seed: str,
    excluded_eval_concepts: Sequence[str] = (),
) -> dict:
    """Freeze matched, image-disjoint fit and evaluation populations.

    One recording is retained per photograph before ranking.  Every retained
    fit photograph contributes exactly one unit to each unimodal arm and one
    unit to the pooled arm.  The pooled assignment cycles across modalities,
    so it has the same number of fitting examples as every comparator while
    remaining balanced to within one example.  The evaluation photographs are
    the next ranked photographs and never enter any fit arm.
    """

    if n_fit_groups < 1 or n_eval_groups < 1:
        raise ValueError("n_fit_groups and n_eval_groups must be positive")
    by_image: dict[str, list[dict]] = {}
    for raw in groups:
        group = dict(raw)
        if not _eligible_group(group, excluded_eval_concepts):
            continue
        by_image.setdefault(str(group["image_id"]), []).append(group)

    representatives: list[dict] = []
    for siblings in by_image.values():
        representatives.append(
            min(
                siblings,
                key=lambda row: _stable_rank(str(row["group_id"]), f"{seed}|sibling"),
            )
        )
    ordered = sorted(
        representatives,
        key=lambda row: _stable_rank(str(row["image_id"]), f"{seed}|image"),
    )
    needed = n_fit_groups + n_eval_groups
    if len(ordered) < needed:
        raise MultimodalLensRefused(
            f"only {len(ordered)} concept-neutral distinct photographs are "
            f"available, but the frozen plan needs {needed}"
        )
    fit_groups = ordered[:n_fit_groups]
    eval_groups = ordered[n_fit_groups:needed]

    def unit(group: Mapping, modality: str) -> FitUnit:
        group_id = str(group["group_id"])
        return FitUnit(
            unit_id=f"{group_id}:{modality}",
            group_id=group_id,
            image_id=str(group["image_id"]),
            modality=modality,
            caption=str(group["caption"]),
            image_path=str(group["image_path"]),
            audio_path=str(group["audio_path"]),
            prompt=fitting_prompt(modality, str(group["caption"])),
        )

    per_modality = {
        modality: [unit(group, modality) for group in fit_groups]
        for modality in MODALITIES
    }
    # One view per photograph keeps the pooled arm sample-count matched to all
    # three unimodal arms.  Cycling modalities makes every prefix maximally
    # balanced without selecting examples from model results.
    pooled = [
        per_modality[MODALITIES[index % len(MODALITIES)]][index]
        for index in range(n_fit_groups)
    ]
    arms = {**per_modality, "pooled": pooled}
    payload = {
        "version": PLAN_VERSION,
        "seed": seed,
        "excluded_eval_concepts": list(excluded_eval_concepts),
        "n_fit_groups": n_fit_groups,
        "n_eval_groups": n_eval_groups,
        "fit_image_ids": [str(group["image_id"]) for group in fit_groups],
        "eval_image_ids": [str(group["image_id"]) for group in eval_groups],
        "fit_groups": fit_groups,
        "eval_groups": eval_groups,
        "arms": {name: [row.to_dict() for row in rows] for name, rows in arms.items()},
        "pooled_modality_counts": {
            modality: sum(row.modality == modality for row in pooled)
            for modality in MODALITIES
        },
        "one_group_per_image": True,
        "fit_eval_image_overlap": sorted(
            {str(g["image_id"]) for g in fit_groups}
            & {str(g["image_id"]) for g in eval_groups}
        ),
    }
    payload["plan_digest"] = payload_checksum(payload)
    return payload


def plan_units(plan: Mapping, arm: str) -> list[FitUnit]:
    if arm not in LENS_ARMS:
        raise ValueError(f"unknown lens arm {arm!r}")
    return [FitUnit(**row) for row in plan["arms"][arm]]


def select_causal_groups(
    groups: Sequence[Mapping],
    *,
    concepts: Sequence[str],
    n_per_concept: int,
    excluded_image_ids: Sequence[str],
    seed: str,
    forbidden_concepts: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, list[dict]]:
    """Freeze fresh, distinct photographs for the unrestricted causal stage.

    ``forbidden_concepts`` prevents a source example from carrying evidence for
    its intended swap target.  The exclusion is checked against both audited
    object labels and the caption, so it applies to image, text, and spoken
    caption inputs alike.
    """

    excluded = {str(value) for value in excluded_image_ids}
    result: dict[str, list[dict]] = {}
    for concept in concepts:
        forbidden = tuple((forbidden_concepts or {}).get(str(concept), ()))
        by_image: dict[str, list[dict]] = {}
        for raw in groups:
            group = dict(raw)
            image_id = str(group.get("image_id") or "")
            if not image_id or image_id in excluded:
                continue
            if any(
                not str(group.get(key) or "").strip()
                for key in ("group_id", "caption", "image_path", "audio_path")
            ):
                continue
            labelled = any(
                _surface_contains(value, concept) for value in _concept_fields(group)
            )
            mentioned = _surface_contains(str(group.get("caption") or ""), concept)
            if not (labelled and mentioned):
                continue
            if any(
                _surface_contains(value, other)
                for other in forbidden
                for value in (*_concept_fields(group), str(group.get("caption") or ""))
            ):
                continue
            by_image.setdefault(image_id, []).append(group)
        representatives = [
            min(
                siblings,
                key=lambda row: _stable_rank(
                    str(row["group_id"]), f"{seed}|{concept}|sibling"
                ),
            )
            for siblings in by_image.values()
        ]
        ordered = sorted(
            representatives,
            key=lambda row: _stable_rank(
                str(row["image_id"]), f"{seed}|{concept}|image"
            ),
        )
        if len(ordered) < n_per_concept:
            raise MultimodalLensRefused(
                f"concept {concept!r} has only {len(ordered)} fresh, caption-"
                f"supported photographs; {n_per_concept} are required"
            )
        result[str(concept)] = ordered[:n_per_concept]
    overlap = set()
    seen: set[str] = set()
    for rows in result.values():
        for row in rows:
            image_id = str(row["image_id"])
            if image_id in seen:
                overlap.add(image_id)
            seen.add(image_id)
    if overlap:
        raise MultimodalLensRefused(
            f"causal concepts share selected photographs {sorted(overlap)}"
        )
    return result


def _replicate_tensors(tensors: Mapping[str, Any], batch_size: int) -> dict:
    """Repeat one processor example without changing any feature value."""

    out: dict[str, Any] = {}
    for key, value in dict(tensors).items():
        if not torch.is_tensor(value):
            out[key] = value
            continue
        if value.ndim == 0:
            out[key] = value
        elif value.shape[0] == 1:
            out[key] = value.expand(batch_size, *value.shape[1:]).contiguous()
        else:
            raise MultimodalLensRefused(
                f"processor tensor {key!r} has batch dimension {value.shape[0]}, "
                "expected one example before Jacobian replication"
            )
    out.setdefault("use_cache", False)
    # The Jacobian estimator reads recorded block activations and discards the
    # forward's return value entirely, but Gemma4 still materializes
    # lm_head(hidden) as [batch, seq, 262144] and then chains three more
    # same-sized temporaries through the tanh softcap. At dim_batch=8 that is
    # ~1.25 GiB per temporary for a multimodal unit, which OOM'd an L4 on the
    # pooled arm's first image example while the text arm (shorter sequences)
    # completed fine. logits_to_keep=1 computes them for one position instead.
    #
    # This cannot change any fitted value: the logits sit *downstream* of the
    # target layer, so they are not on the path from source layers to target
    # and contribute nothing to torch.autograd.grad(target, sources).
    out.setdefault("logits_to_keep", 1)
    return out


def jacobian_for_built_inputs(
    backend,
    inputs,
    source_layers: Sequence[int],
    *,
    target_layer: int,
    dim_batch: int = 8,
    skip_first: int = 16,
    backward_context: Callable[[], object] | None = None,
) -> tuple[dict[int, torch.Tensor], int, int]:
    """The upstream estimator over real processor tensors.

    Media placeholders and their decoder positions participate exactly as they
    do in an ordinary multimodal forward.  No transcript is accepted here;
    that invariant is enforced by the backend that built ``inputs``.
    """

    source_layers = sorted({int(layer) for layer in source_layers})
    if not source_layers or source_layers[-1] >= int(target_layer):
        raise ValueError("source layers must be nonempty and precede target_layer")
    if int(target_layer) >= int(backend.n_layers):
        raise ValueError("target_layer is outside the decoder")
    seq_len = int(inputs.prompt_len)
    mask = valid_position_mask(seq_len, skip_first=skip_first)
    valid = mask.nonzero(as_tuple=True)[0]
    n_valid = int(valid.numel())
    d_model = int(backend.d_model)
    jacobians = {
        layer: torch.zeros(d_model, d_model, dtype=torch.float32)
        for layer in source_layers
    }
    n_passes = math.ceil(d_model / dim_batch)
    tensors = _replicate_tensors(inputs.tensors, dim_batch)

    with (
        backward_context() if backward_context is not None else nullcontext(),
        ActivationRecorder(
            backend.blocks,
            at=[*source_layers, int(target_layer)],
            start_graph_at=min(source_layers),
        ) as recorder,
        torch.enable_grad(),
    ):
        backend.hf_model(**tensors)
        target = recorder.activations[int(target_layer)]
        sources = [recorder.activations[layer] for layer in source_layers]
        valid_target = valid.to(target.device)
        batch = torch.arange(dim_batch, device=target.device)
        cotangent = torch.zeros_like(target)
        for pass_index, start in enumerate(range(0, d_model, dim_batch)):
            width = min(dim_batch, d_model - start)
            cotangent.zero_()
            cotangent[
                batch[:width, None], valid_target[None, :], start + batch[:width, None]
            ] = 1.0
            grads = torch.autograd.grad(
                outputs=target,
                inputs=sources,
                grad_outputs=cotangent,
                retain_graph=pass_index < n_passes - 1,
            )
            for layer, grad in zip(source_layers, grads, strict=True):
                positions = valid.to(grad.device)
                rows = grad[:width, positions, :].float().mean(dim=1)
                jacobians[layer][start : start + width] = rows.cpu()
            del grads
    return jacobians, seq_len, n_valid


def _atomic_torch_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def fit_arm(
    backend,
    units: Sequence[FitUnit],
    *,
    build_inputs: Callable[[FitUnit], Any],
    source_layers: Sequence[int],
    target_layer: int,
    checkpoint_path: str | Path,
    arm: str,
    scientific_fingerprint: str,
    dim_batch: int = 8,
    skip_first: int = 16,
    checkpoint_every: int = 5,
    backward_context: Callable[[], object] | None = None,
    progress: Callable[[dict], None] | None = None,
) -> JacobianLens:
    """Fit one arm with an atomic, fingerprint-bound accumulator."""

    if arm not in LENS_ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    layers = sorted({int(layer) for layer in source_layers})
    record_digest = payload_checksum([unit.to_dict() for unit in units])
    contract = {
        "estimator_version": ESTIMATOR_VERSION,
        "scientific_fingerprint": scientific_fingerprint,
        "arm": arm,
        "record_digest": record_digest,
        "source_layers": layers,
        "target_layer": int(target_layer),
        "dim_batch": int(dim_batch),
        "skip_first": int(skip_first),
        "d_model": int(backend.d_model),
    }
    contract_digest = payload_checksum(contract)
    checkpoint = Path(checkpoint_path)
    if checkpoint.is_file():
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state.get("contract_digest") != contract_digest:
            raise MultimodalLensRefused(
                f"checkpoint {checkpoint} belongs to a different arm, "
                "population, processor protocol, layer grid, or estimator; "
                "refusing to mix accumulators"
            )
        sums = state["jacobian_sum"]
        n_done = int(state["n_done"])
        next_index = int(state["next_index"])
    else:
        sums = {
            layer: torch.zeros(backend.d_model, backend.d_model, dtype=torch.float32)
            for layer in layers
        }
        n_done = 0
        next_index = 0

    def save() -> None:
        _atomic_torch_save(
            {
                "contract": contract,
                "contract_digest": contract_digest,
                "jacobian_sum": sums,
                "n_done": n_done,
                "next_index": next_index,
            },
            checkpoint,
        )

    for index, unit in enumerate(units):
        if index < next_index:
            continue
        started = time.perf_counter()
        built = build_inputs(unit)
        contribution, seq_len, n_valid = jacobian_for_built_inputs(
            backend,
            built,
            layers,
            target_layer=target_layer,
            dim_batch=dim_batch,
            skip_first=skip_first,
            backward_context=backward_context,
        )
        for layer in layers:
            sums[layer] += contribution[layer]
        n_done += 1
        next_index = index + 1
        wrote = next_index % max(1, checkpoint_every) == 0
        if wrote:
            save()
        if progress is not None:
            progress(
                {
                    "arm": arm,
                    "index": next_index,
                    "total": len(units),
                    "unit_id": unit.unit_id,
                    "modality": unit.modality,
                    "seq_len": seq_len,
                    "n_valid_positions": n_valid,
                    "n_done": n_done,
                    "elapsed_seconds": time.perf_counter() - started,
                    "checkpoint_written": wrote,
                }
            )
    save()
    if n_done != len(units):
        raise MultimodalLensRefused(
            f"arm {arm} accumulated {n_done} units for a {len(units)}-unit plan"
        )
    return JacobianLens(
        jacobians={layer: sums[layer] / n_done for layer in layers},
        n_prompts=n_done,
        d_model=int(backend.d_model),
    )


def combine_layer_shards(
    shards: Sequence[JacobianLens],
    *,
    expected_layers: Sequence[int] | None = None,
) -> JacobianLens:
    """Join lenses fitted on the same examples but on disjoint layer sets.

    This is deliberately different from :meth:`JacobianLens.merge`, which
    averages lenses fitted on disjoint *prompt* subsets.  Here every shard must
    have the same ``n_prompts`` and ``d_model`` and no physical layer may occur
    twice.  The result simply places the already-estimated per-layer matrices
    beside one another.  When ``expected_layers`` is supplied, it must describe
    one exact contiguous band; missing, duplicated, or extra layers are refused.
    """

    if not shards:
        raise MultimodalLensRefused("at least one layer shard is required")
    first = shards[0]
    combined: dict[int, torch.Tensor] = {}
    for index, shard in enumerate(shards):
        if shard.d_model != first.d_model:
            raise MultimodalLensRefused(
                f"layer shard {index} has d_model={shard.d_model}, not "
                f"{first.d_model}"
            )
        if shard.n_prompts != first.n_prompts:
            raise MultimodalLensRefused(
                f"layer shard {index} has n_prompts={shard.n_prompts}, not "
                f"{first.n_prompts}; layer shards must use the same fit set"
            )
        overlap = sorted(set(combined) & set(shard.jacobians))
        if overlap:
            raise MultimodalLensRefused(
                f"layer shards overlap at physical layer(s) {overlap}"
            )
        combined.update(shard.jacobians)

    observed = sorted(combined)
    if expected_layers is not None:
        expected = sorted({int(layer) for layer in expected_layers})
        if expected != list(range(expected[0], expected[-1] + 1)):
            raise MultimodalLensRefused(
                f"expected layer band is not contiguous: {expected}"
            )
        if observed != expected:
            raise MultimodalLensRefused(
                f"combined layer shards cover {observed}, not {expected}"
            )
    return JacobianLens(
        jacobians=combined,
        n_prompts=first.n_prompts,
        d_model=first.d_model,
    )


def paired_binary_one_sided_p(
    primary: Sequence[bool], control: Sequence[bool]
) -> dict:
    """Exact one-sided paired sign test for binary intervention outcomes.

    Only discordant pairs carry information.  Under the null, a discordant
    outcome is equally likely to favour either condition, so the p-value is the
    upper binomial tail for the number favouring ``primary``.  No SciPy runtime
    dependency is required, which keeps the Colab confirmation self-contained.
    """

    left = [bool(value) for value in primary]
    right = [bool(value) for value in control]
    if len(left) != len(right) or not left:
        raise MultimodalLensRefused(
            "paired binary test requires equal non-empty outcome sequences"
        )
    primary_only = sum(a and not b for a, b in zip(left, right, strict=True))
    control_only = sum(b and not a for a, b in zip(left, right, strict=True))
    discordant = primary_only + control_only
    if discordant == 0:
        p_value = 1.0
    else:
        numerator = sum(
            math.comb(discordant, successes)
            for successes in range(primary_only, discordant + 1)
        )
        p_value = numerator / (2**discordant)
    return {
        "n": len(left),
        "primary_only": primary_only,
        "control_only": control_only,
        "discordant": discordant,
        "one_sided_p": float(min(1.0, p_value)),
    }


def holm_adjust(records: Sequence[Mapping], *, p_key: str = "one_sided_p") -> list[dict]:
    """Return records with monotone Holm family-wise adjusted p-values."""

    rows = [dict(record) for record in records]
    if not rows:
        return []
    ordered = sorted(enumerate(rows), key=lambda item: float(item[1][p_key]))
    adjusted_by_index: dict[int, float] = {}
    running = 0.0
    total = len(rows)
    for rank, (original_index, row) in enumerate(ordered):
        candidate = min(1.0, float(row[p_key]) * (total - rank))
        running = max(running, candidate)
        adjusted_by_index[original_index] = running
    return [
        {**row, "holm_adjusted_p": adjusted_by_index[index]}
        for index, row in enumerate(rows)
    ]


def selected_lens_vector(
    lens: JacobianLens, unembedding_weight: torch.Tensor, *, layer: int, token_id: int
) -> torch.Tensor:
    """One row of ``W_U @ J_l`` without materializing the full dictionary."""

    row = unembedding_weight[int(token_id)].detach().float().cpu()
    return row @ lens.jacobians[int(layer)].detach().float().cpu()


def build_swap_bases_for_lens(
    lens: JacobianLens,
    unembedding_weight: torch.Tensor,
    *,
    layers: Sequence[int],
    source: ConceptToken,
    target: ConceptToken,
) -> dict[int, SwapBasis]:
    return {
        int(layer): build_swap_basis_from_vectors(
            selected_lens_vector(
                lens, unembedding_weight, layer=int(layer), token_id=source.token_id
            ),
            selected_lens_vector(
                lens, unembedding_weight, layer=int(layer), token_id=target.token_id
            ),
            layer=int(layer),
            source=source,
            target=target,
        )
        for layer in layers
    }


@torch.no_grad()
def capture_eval_rows(
    backend,
    lenses: Mapping[str, JacobianLens],
    eval_groups: Sequence[Mapping],
    *,
    build_inputs: Callable[[Mapping, str, str], Any],
    layers: Sequence[int],
) -> list[dict]:
    """Full-vocabulary native-answer fidelity for every arm x modality x layer."""

    rows: list[dict] = []
    norm = backend.hf_model.model.language_model.norm
    head = backend.hf_model.lm_head
    for group in eval_groups:
        for modality in MODALITIES:
            prompt = evaluation_prompt(modality, str(group["caption"]))
            built = build_inputs(group, modality, prompt)
            with ActivationRecorder(backend.blocks, at=list(layers)) as recorder:
                clean_logits = backend.forward_logits(built.tensors)[
                    0, built.final_prompt_position
                ].float()
            clean_token = int(clean_logits.argmax())
            for arm, lens in lenses.items():
                for layer in layers:
                    h = recorder.activations[int(layer)][
                        0, built.final_prompt_position
                    ].detach().float().cpu()
                    transported = lens.transport(h, int(layer))
                    device = head.weight.device
                    lens_logits = head(
                        norm(transported.to(device=device, dtype=head.weight.dtype))
                    )[0].float() if transported.ndim == 2 else head(
                        norm(transported.to(device=device, dtype=head.weight.dtype))
                    ).float()
                    target_score = lens_logits[clean_token]
                    optimistic_rank = 1 + int((lens_logits > target_score).sum())
                    pessimistic_rank = int((lens_logits >= target_score).sum())
                    rows.append(
                        {
                            "version": CROSS_EVAL_VERSION,
                            "group_id": str(group["group_id"]),
                            "image_id": str(group["image_id"]),
                            "test_modality": modality,
                            "lens_arm": arm,
                            "layer": int(layer),
                            "clean_token_id": clean_token,
                            "lens_top_token_id": int(lens_logits.argmax()),
                            "top1_agreement": int(lens_logits.argmax()) == clean_token,
                            "optimistic_rank": optimistic_rank,
                            "pessimistic_rank": pessimistic_rank,
                            "midrank": (optimistic_rank + pessimistic_rank) / 2.0,
                            "reciprocal_midrank": 2.0
                            / (optimistic_rank + pessimistic_rank),
                        }
                    )
    return rows


def summarize_cross_eval(rows: Sequence[Mapping], *, shuffle_seed: int = 20260819) -> dict:
    """Summarize the frozen 4x3 matrix and a shuffled-target control."""

    grouped: dict[tuple[str, str, int], list[dict]] = {}
    for raw in rows:
        row = dict(raw)
        grouped.setdefault(
            (str(row["lens_arm"]), str(row["test_modality"]), int(row["layer"])),
            [],
        ).append(row)
    cells: list[dict] = []
    for (arm, modality, layer), cell in sorted(grouped.items()):
        n = len(cell)
        cells.append(
            {
                "lens_arm": arm,
                "test_modality": modality,
                "layer": layer,
                "n": n,
                "top1_agreement": sum(bool(r["top1_agreement"]) for r in cell) / n,
                "mrr": sum(float(r["reciprocal_midrank"]) for r in cell) / n,
                "median_midrank": float(
                    torch.tensor([float(r["midrank"]) for r in cell]).median()
                ),
            }
        )
    # The control permutes target ownership within each modality/layer.  The
    # full per-token logits are deliberately not persisted, so this control is
    # expressed as the top-1 agreement expected under a derangement of the
    # recorded native targets.
    controls: list[dict] = []
    for (arm, modality, layer), cell in sorted(grouped.items()):
        ordered = sorted(cell, key=lambda r: (str(r["image_id"]), str(r["group_id"])))
        targets = [int(r["clean_token_id"]) for r in ordered]
        shuffled = targets[:]
        random.Random(f"{shuffle_seed}|{arm}|{modality}|{layer}").shuffle(shuffled)
        controls.append(
            {
                "lens_arm": arm,
                "test_modality": modality,
                "layer": layer,
                "kind": "shuffled_native_target_top1",
                "top1_agreement": sum(
                    int(r["lens_top_token_id"]) == target
                    for r, target in zip(ordered, shuffled, strict=True)
                )
                / len(ordered),
            }
        )
    payload = {
        "version": CROSS_EVAL_VERSION,
        "n_rows": len(rows),
        "cells": cells,
        "controls": controls,
        "primary_comparison": "pooled_vs_text_on_image_and_spoken_audio",
        "interpretation": (
            "native-answer fidelity measures whether each frozen Jacobian map "
            "transports a modality-induced residual toward the model's own "
            "unrestricted next-token answer; it is not semantic accuracy"
        ),
    }
    payload["report_checksum"] = payload_checksum(payload)
    return payload


#: The exact wording the completed leg-count confirmation asked. It lives here
#: rather than as a closure inside one notebook cell because Stage 3DA replays
#: that run, and a replay under a different prompt is not a replay.
CONFIRMATION_LEG_COUNT_QUESTION = (
    "How many legs does the animal in the evidence typically have? "
    "Answer with one digit.\nAnswer:"
)


def confirmation_leg_count_prompt(modality: str, caption: str) -> str:
    """The confirmed study's prompt: the caption is shown to the text arm only.

    The image and spoken-audio arms see the question alone, so the evidence
    they answer from is the photograph or the recording rather than the words.
    """
    if str(modality) == "text":
        return f"Caption: {caption}\n{CONFIRMATION_LEG_COUNT_QUESTION}"
    return CONFIRMATION_LEG_COUNT_QUESTION


@torch.no_grad()
def unrestricted_swap_trial(
    backend,
    inputs,
    *,
    bases: Mapping[int, SwapBasis],
    alpha: float = 1.0,
    target_token_id: int | None = None,
    source_token_id: int | None = None,
    clean_logits: torch.Tensor | None = None,
    compact_positions: bool = False,
    position_rule: str = PRIMARY_POSITION_RULE,
    realization_policy: ModelDtypeRealizationPolicy | None = None,
) -> dict:
    """One paper-style exchange scored on the unrestricted next-token output.

    ``alpha=1`` is the exact two-coordinate exchange.  Other values use the
    identical intervention path and are labelled interpolation or
    extrapolation.  Optional token ids add graded full-vocabulary diagnostics;
    they never restrict the model's output or supply candidates to it.

    Two fields here are easy to misread and were misread once.
    ``max_coordinate_update_error`` and ``max_orthogonal_residual_drift`` are
    the **float64 pre-cast** solve errors: exact by construction, and no
    evidence at all about what the model's bf16 residual stream received. The
    ``max_post_cast_*`` fields are the ones that answer that, and they are now
    persisted rather than computed and discarded. ``realization_policy`` is
    optional and defaults to ``None`` so that a replication can reproduce a
    completed run's exact uncorrected path; pass
    :data:`jlens.mmpilot.workspace_replication.TEXT_MODEL_DTYPE_REALIZATION`
    for any new measurement.
    """

    if clean_logits is None:
        clean = backend.forward_logits(inputs.tensors)[
            0, inputs.final_prompt_position
        ].float()
    else:
        clean = clean_logits.detach().float()
        if clean.ndim != 1:
            raise MultimodalLensRefused(
                f"clean_logits must be rank one, got shape {tuple(clean.shape)}"
            )
    with coordinate_swap_band(
        backend.blocks,
        bases,
        alpha=float(alpha),
        prompt_len=inputs.prompt_len,
        position_rule=position_rule,
        evidence_span=inputs.modality_token_range,
        record_coordinates=False,
        realization_policy=realization_policy,
    ) as stats:
        patched = backend.forward_logits(inputs.tensors)[
            0, inputs.final_prompt_position
        ].float()
    position_records: dict[str, Any] = {}
    all_prompt_positions_patched = True
    for layer in sorted(stats):
        positions = list(stats[layer].get("positions") or [])
        all_prompt_positions_patched &= positions == list(range(inputs.prompt_len))
        if compact_positions:
            position_records[str(layer)] = {
                "start": positions[0] if positions else None,
                "stop_exclusive": positions[-1] + 1 if positions else None,
                "count": len(positions),
                "contiguous": positions
                == list(range(positions[0], positions[-1] + 1))
                if positions
                else False,
            }
        else:
            position_records[str(layer)] = positions

    norm_ratios: list[float] = []
    update_ratios: list[float] = []
    orthogonal_drifts: list[float] = []
    coordinate_errors: list[float] = []
    post_cast_coordinate_errors: list[float] = []
    post_cast_residual_drifts: list[float] = []
    realizations_converged: list[bool] = []
    exact_before_cast: list[bool] = []
    corrections_applied: list[int] = []
    realization_payload: dict | None = None
    for layer in sorted(stats):
        swap = dict(stats[layer].get("swap") or {})
        before = list(swap.get("activation_norm_before") or [])
        after = list(swap.get("activation_norm_after") or [])
        updates = list(swap.get("update_norm") or [])
        norm_ratios.extend(
            float(post) / max(float(pre), 1e-12)
            for pre, post in zip(before, after, strict=True)
        )
        update_ratios.extend(
            float(update) / max(float(pre), 1e-12)
            for pre, update in zip(before, updates, strict=True)
        )
        orthogonal_drifts.append(float(swap.get("max_orthogonal_residual_drift", 0.0)))
        coordinate_errors.append(float(swap.get("max_coordinate_update_error", 0.0)))
        # What the model's own dtype actually received. Computed by
        # swap_coordinates on every call, policy or not; previously discarded
        # here, which is why no completed artifact can answer the question.
        if "max_post_cast_relative_coordinate_update_error" in swap:
            post_cast_coordinate_errors.append(
                float(swap["max_post_cast_relative_coordinate_update_error"])
            )
        if "max_post_cast_relative_orthogonal_residual_drift" in swap:
            post_cast_residual_drifts.append(
                float(swap["max_post_cast_relative_orthogonal_residual_drift"])
            )
        realizations_converged.append(
            bool(swap.get("model_dtype_realization_converged", True))
        )
        exact_before_cast.append(bool(swap.get("alpha_one_is_exact_exchange")))
        corrections_applied.append(
            int(swap.get("model_dtype_corrections_applied") or 0)
        )
        if realization_payload is None and isinstance(
            swap.get("model_dtype_realization"), Mapping
        ):
            realization_payload = dict(swap["model_dtype_realization"])

    result = {
        "alpha": float(alpha),
        "alpha_role": (
            "exact_exchange"
            if float(alpha) == 1.0
            else "interpolation"
            if 0.0 <= float(alpha) < 1.0
            else "extrapolation"
        ),
        "alpha_is_extrapolation": bool(float(alpha) > 1.0),
        "teacher_forcing_used": False,
        "candidate_list_supplied": False,
        "position_rule": str(position_rule),
        "clean_top_token_id": int(clean.argmax()),
        "patched_top_token_id": int(patched.argmax()),
        "prediction_changed": int(clean.argmax()) != int(patched.argmax()),
        "layers_patched": sorted(int(layer) for layer in stats),
        "positions_patched": position_records,
        "all_prompt_positions_patched": bool(all_prompt_positions_patched),
        "max_activation_norm_ratio": max(norm_ratios, default=1.0),
        "min_activation_norm_ratio": min(norm_ratios, default=1.0),
        "max_update_to_activation_norm_ratio": max(update_ratios, default=0.0),
        "max_orthogonal_residual_drift": max(orthogonal_drifts, default=0.0),
        "max_coordinate_update_error": max(coordinate_errors, default=0.0),
        "coordinate_error_basis": "float64_pre_cast_solve",
        "max_post_cast_relative_coordinate_error": max(
            post_cast_coordinate_errors, default=None
        ),
        "max_post_cast_relative_residual_drift": max(
            post_cast_residual_drifts, default=None
        ),
        "all_layers_are_exact_alpha_one_exchange_before_cast": (
            bool(exact_before_cast) and all(exact_before_cast)
        ),
        "all_model_dtype_realizations_converged": (
            bool(realizations_converged) and all(realizations_converged)
        ),
        "max_model_dtype_corrections_applied": max(corrections_applied, default=0),
        "model_dtype_realization_policy": realization_payload,
        "model_dtype_realization_policy_supplied": realization_payload is not None,
    }
    if target_token_id is not None:
        target = int(target_token_id)
        if not 0 <= target < clean.numel():
            raise MultimodalLensRefused(
                f"target token id {target} is outside vocabulary size {clean.numel()}"
            )
        clean_log_probs = clean.log_softmax(dim=-1)
        patched_log_probs = patched.log_softmax(dim=-1)
        clean_target = float(clean[target])
        patched_target = float(patched[target])
        clean_rank = 1 + int((clean > clean[target]).sum())
        patched_rank = 1 + int((patched > patched[target]).sum())
        clean_probability = float(clean_log_probs[target].exp())
        patched_probability = float(patched_log_probs[target].exp())
        result.update(
            {
                "target_token_id": target,
                "clean_target_logit": clean_target,
                "patched_target_logit": patched_target,
                "target_logit_delta": patched_target - clean_target,
                "clean_target_rank": clean_rank,
                "patched_target_rank": patched_rank,
                "target_rank_improvement": clean_rank - patched_rank,
                "clean_target_probability": clean_probability,
                "patched_target_probability": patched_probability,
                "target_probability_delta": patched_probability - clean_probability,
                "target_is_top1": int(patched.argmax()) == target,
                "kl_clean_to_patched": float(
                    (clean_log_probs.exp() * (clean_log_probs - patched_log_probs)).sum()
                ),
            }
        )
    if source_token_id is not None:
        source = int(source_token_id)
        if not 0 <= source < clean.numel():
            raise MultimodalLensRefused(
                f"source token id {source} is outside vocabulary size {clean.numel()}"
            )
        result.update(
            {
                "source_token_id": source,
                "clean_source_logit": float(clean[source]),
                "patched_source_logit": float(patched[source]),
                "source_logit_delta": float(patched[source] - clean[source]),
            }
        )
    return result


def fit_budget(
    *, n_fit_groups: int, n_layers: int, d_model: int = 2560, dim_batch: int = 8
) -> dict:
    """Exact forward/backward counts printed before model load."""

    per_prompt_backward = math.ceil(d_model / dim_batch)
    per_arm = {
        "text": n_fit_groups,
        "image": n_fit_groups,
        "spoken_audio": n_fit_groups,
        "pooled": n_fit_groups,
    }
    total_prompts = sum(per_arm.values())
    return {
        "n_fit_groups": n_fit_groups,
        "n_layers": n_layers,
        "dim_batch": dim_batch,
        "per_prompt_forward": 1,
        "per_prompt_backward": per_prompt_backward,
        "prompts_by_arm": per_arm,
        "total_prompt_forwards": total_prompts,
        "total_backward_passes": total_prompts * per_prompt_backward,
        "checkpoint_contract": (
            "one atomic accumulator per arm; at most checkpoint_every newly "
            "completed units are recomputed after a disconnect"
        ),
    }


__all__ = [
    "CROSS_EVAL_VERSION",
    "ESTIMATOR_VERSION",
    "FitUnit",
    "LENS_ARMS",
    "MODALITIES",
    "MultimodalLensRefused",
    "build_matched_plan",
    "build_swap_bases_for_lens",
    "capture_eval_rows",
    "combine_layer_shards",
    "evaluation_prompt",
    "fit_arm",
    "fit_budget",
    "fitting_prompt",
    "holm_adjust",
    "jacobian_for_built_inputs",
    "load_completed_alpha_sweep_source",
    "load_broad_pooled_development_source",
    "paired_binary_one_sided_p",
    "plan_units",
    "confirmation_leg_count_prompt",
    "selected_lens_vector",
    "select_causal_groups",
    "summarize_cross_eval",
    "unrestricted_swap_trial",
]
