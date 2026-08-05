# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""The three-modality extension: budget, six directional pairs, five verdicts.

The image-unique robustness study asked whether text <-> image transfer
replicates. This asks the question that one could not: **does the same frozen,
text-calibrated J-space carry spoken audio?** SpokenCOCO's recordings are
*spoken captions* — linguistic audio. Nothing here is evidence about
environmental sound, and no verdict may be phrased as if it were.

Three things are worth stating plainly.

**Text <-> image is demoted to an internal replication.** It is already a
confirmed result; reporting it again as the finding would be reporting the
question that was already answered. The four audio-related directions —
``text->spoken_audio``, ``spoken_audio->text``, ``image->spoken_audio``,
``spoken_audio->image`` — are what this study is for, and a GO explicitly
requires the result not to be text <-> image alone.

**The stages are separated so the cost is confirmable.** Stage A (capability
and representation at all three layers) is cheap and decides whether anything
else is worth spending. Stage B is one layer of causal work. Stage C repeats
the *frozen* Stage-B design at the other two layers. Each has its own budget and
its own confirmation flag, and :func:`estimate_stage_passes` derives every count
from the configuration rather than guessing it.

**Layer 35 is primary because it is the earliest independently confirmed lens,
and for no other reason.** It is not "pre-language", not "pre-convergence", and
not "early" in any sense this study measures. Convergence timing is unresolved
here and every artifact says so.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from jlens.mmpilot.report import FAIL, NOT_EVALUATED, PASS, criterion

#: Bound into the run fingerprint. Distinct from the pilot's, the robustness
#: study's, the localization study's and the audio audit's verdict tags, so no
#: report can be mistaken for another study's.
TRI_MODAL_VERDICT_VERSION = "mmpilot.tri_modal_transfer_verdict.v1"

SPOKEN_AUDIO = "spoken_audio"

#: Every ordered cross-modal direction, in the order the report prints them.
ALL_PAIRS: tuple[str, ...] = (
    "text->image",
    "image->text",
    "text->spoken_audio",
    "spoken_audio->text",
    "image->spoken_audio",
    "spoken_audio->image",
)

#: The four this study exists to answer.
AUDIO_PAIRS: tuple[str, ...] = tuple(pair for pair in ALL_PAIRS if SPOKEN_AUDIO in pair)

#: The two already confirmed. Reported as an internal replication, never as the
#: finding.
REPLICATION_PAIRS: tuple[str, ...] = ("text->image", "image->text")

#: Verdict A.
AUDIO_CAPABILITY_GO = "AUDIO_CAPABILITY_GO"
AUDIO_CAPABILITY_NO_GO = "AUDIO_CAPABILITY_NO_GO"

#: Verdicts B, C, D.
TRANSFER_SUPPORTED = "SUPPORTED"
TRANSFER_WEAK = "WEAK"
TRANSFER_UNSUPPORTED = "UNSUPPORTED"
TRANSFER_NOT_EVALUATED = "NOT_EVALUATED"

#: Verdict E.
THREE_MODALITY_GO = "THREE_MODALITY_GO"
THREE_MODALITY_WEAK_GO = "THREE_MODALITY_WEAK_GO"
THREE_MODALITY_NO_GO = "THREE_MODALITY_NO_GO"

#: Printed on every artifact that mentions audio.
SPOKEN_AUDIO_SCOPE = (
    "SpokenCOCO carries spoken *captions*: a person reading the written caption "
    "aloud. This study is about linguistic audio only. It tests nothing about "
    "environmental sound and no result here is evidence about it."
)

#: Printed on every artifact that mentions layer 35.
LAYER_CHOICE_NOTE = (
    "Layer 35 is primary because it is the earliest layer whose lens passed the "
    "calibration run's untouched confirmation set — not because anything here "
    "shows it precedes answer-language convergence. Convergence timing remains "
    "unresolved unless independently tested."
)


@dataclass(frozen=True)
class TriModalThresholds:
    """The rubric's numbers, stated once and printed in every report."""

    #: Verdict A: concepts that must clear the gate in text, image AND audio.
    min_concepts_all_three_modalities: int = 2
    capability_threshold: float = 0.7
    #: Verdict B: an audio-related pair must strictly beat its shuffled p95.
    min_audio_pairs_beating_shuffled: int = 1
    require_both_audio_directions_representational: bool = False
    #: Verdict C: at least this many audio-related causal cells must pass.
    min_audio_causal_cells: int = 1
    min_fraction_expected_sign: float = 0.75
    control_separation_factor: float = 1.5
    norm_ratio_bounds: tuple[float, float] = (0.5, 2.0)
    #: The design's per-cell target counts.
    required_positive_images_per_cell: int = 8
    required_negative_images_per_cell: int = 8
    #: The *claim* floor, separate from the design: no cell may carry a claim on
    #: fewer than this many distinct photographs/recordings on either side.
    min_distinct_images_supporting_a_claim: int = 2
    version: str = TRI_MODAL_VERDICT_VERSION

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["norm_ratio_bounds"] = list(self.norm_ratio_bounds)
        return payload


# ------------------------------------------------------------- pass budget


@dataclass(frozen=True)
class StageBudget:
    """What one stage will cost, before anything is loaded."""

    stage: str
    description: str
    n_candidates: int
    n_modalities: int
    n_layers: int
    n_causal_cells: int
    n_targets_per_cell: int
    n_conditions_per_target: int
    capability_passes: int
    activation_passes: int
    causal_clean_passes: int
    causal_intervention_passes: int
    total_passes: int
    estimated_units: dict
    estimated_drive_bytes: int

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def estimated_drive_mb(self) -> float:
        return self.estimated_drive_bytes / (1024 * 1024)

    def hours(self, seconds_per_pass: float) -> float:
        return self.total_passes * seconds_per_pass / 3600.0


#: Deliberate over-estimates. A budget that surprises you upward mid-run is
#: worse than one that surprises you downward.
_BYTES_PER_UNIT = {
    "capability": 12_000,
    "activation": lambda d_model: d_model * 20 + 2_000,
    "jspace": 4_000,
    "direction": lambda d_model: d_model * 20 + 2_000,
    "intervention": 4_000,
    "metric": 200_000,
}


def _unit_bytes(name: str, count: int, d_model: int) -> int:
    per = _BYTES_PER_UNIT[name]
    return count * (per(d_model) if callable(per) else per)


def estimate_stage_passes(
    *,
    n_concepts: int,
    n_focal_concepts: int,
    modalities: Sequence[str],
    layers: Sequence[int],
    causal_layers: Sequence[int],
    n_total_groups: int,
    n_capability_groups: int,
    n_targets_per_cell: int,
    alphas: Sequence[float],
    stage: str,
    n_control_kinds_with_alpha: int = 4,
    n_candidate_orders: int = 2,
    d_model: int = 2560,
) -> StageBudget:
    """Count every forward pass one stage implies. No model is touched.

    A "pass" is one teacher-forced forward over prompt plus one candidate
    sequence — the unit the capability gate and every intervention are billed
    in. Scoring six candidates is six passes, which is why widening the concept
    set is not free, and why three modalities cost half again what two did.

    ``stage`` is ``"A"`` (capability, activations, J-space, representation),
    ``"B"`` (causal at ``causal_layers``) or ``"C"`` (the same design repeated
    at the remaining layers). Stage A spends no causal passes; B and C spend no
    capability or activation passes, because Stage A already stored them and the
    unit store reuses them.
    """
    stage = str(stage).upper()
    if stage not in ("A", "B", "C"):
        raise ValueError(f"unknown stage {stage!r}; expected 'A', 'B' or 'C'")

    n_modalities = len(modalities)
    n_candidates = n_concepts
    n_layers = len(layers)
    positive_alphas = [alpha for alpha in alphas if alpha > 0]
    n_conditions = 1 + len(positive_alphas) * n_control_kinds_with_alpha
    # Off-diagonal only: every ordered (source, target) pair with source != target.
    n_cells_per_layer = n_focal_concepts * n_modalities * (n_modalities - 1)
    n_causal_layers = len(causal_layers)

    capability = activation = clean = intervention = 0
    units: dict[str, int] = {}
    if stage == "A":
        capability = n_capability_groups * n_modalities * n_candidate_orders * n_candidates
        activation = n_total_groups * n_modalities
        units = {
            "capability": n_capability_groups * n_modalities,
            "activation": n_total_groups * n_modalities * n_layers,
            "jspace": n_total_groups * n_modalities * n_layers,
            "metric": 2 + n_layers,
        }
        n_cells = 0
    else:
        n_cells = n_cells_per_layer * n_causal_layers
        clean = n_cells * n_targets_per_cell * n_candidates
        intervention = n_cells * n_targets_per_cell * n_conditions * n_candidates
        units = {
            "direction": n_concepts * n_modalities * 2 * n_causal_layers,
            "intervention": n_cells * n_targets_per_cell * n_conditions,
            "metric": 2 * n_causal_layers,
        }

    estimated_bytes = sum(_unit_bytes(name, count, d_model) for name, count in units.items())
    description = {
        "A": "capability, activation capture, J-space codes and the six "
        "representational pairs at every validated layer",
        "B": "the primary causal test at the earliest confirmed layer",
        "C": "the frozen Stage-B design repeated at the remaining layers",
    }[stage]
    return StageBudget(
        stage=stage,
        description=description,
        n_candidates=n_candidates,
        n_modalities=n_modalities,
        n_layers=n_layers if stage == "A" else n_causal_layers,
        n_causal_cells=n_cells,
        n_targets_per_cell=n_targets_per_cell if stage != "A" else 0,
        n_conditions_per_target=n_conditions if stage != "A" else 0,
        capability_passes=capability,
        activation_passes=activation,
        causal_clean_passes=clean,
        causal_intervention_passes=intervention,
        total_passes=capability + activation + clean + intervention,
        estimated_units=units,
        estimated_drive_bytes=estimated_bytes,
    )


def format_stage_budget(budget: StageBudget, *, seconds_per_pass: float = 0.5) -> str:
    """The block the notebook prints before asking for that stage's confirmation."""
    lines = [
        "=" * 72,
        f"STAGE {budget.stage} PASS BUDGET — read this before confirming",
        "=" * 72,
        f"  {budget.description}",
        "",
        f"  candidates scored per question {budget.n_candidates} "
        "(every candidate is a separate forward pass)",
        f"  modalities                     {budget.n_modalities}",
        f"  layers in this stage           {budget.n_layers}",
    ]
    if budget.n_causal_cells:
        lines += [
            f"  off-diagonal causal cells      {budget.n_causal_cells}",
            f"  targets per cell               {budget.n_targets_per_cell} "
            "(distinct images/recordings)",
            f"  conditions per target          {budget.n_conditions_per_target}",
        ]
    lines += [
        "",
        f"  capability passes              {budget.capability_passes:>9,}",
        f"  activation passes              {budget.activation_passes:>9,}",
        f"  causal clean-scoring passes    {budget.causal_clean_passes:>9,}",
        f"  causal intervention passes     {budget.causal_intervention_passes:>9,}",
        f"  TOTAL model forward passes     {budget.total_passes:>9,}",
        "",
        f"  estimated wall clock           ~{budget.hours(seconds_per_pass):.1f} h at "
        f"{seconds_per_pass:.2f} s/pass on one L4",
        f"  estimated Drive footprint      ~{budget.estimated_drive_mb:.0f} MB",
        "  estimated units: "
        + ", ".join(f"{k}={v}" for k, v in sorted(budget.estimated_units.items())),
    ]
    return "\n".join(lines)


def format_total_budget(
    budgets: Sequence[StageBudget], *, seconds_per_pass: float = 0.5
) -> str:
    """All stages side by side, so the whole study's cost is one number too."""
    total = sum(budget.total_passes for budget in budgets)
    drive = sum(budget.estimated_drive_bytes for budget in budgets)
    lines = [
        "=" * 72,
        "WHOLE-STUDY BUDGET (stages are confirmed separately and never run "
        "automatically)",
        "=" * 72,
        "  stage   passes        hours     drive",
    ]
    for budget in budgets:
        lines.append(
            f"  {budget.stage:<6}  {budget.total_passes:>9,}  "
            f"{budget.hours(seconds_per_pass):>7.1f}   "
            f"{budget.estimated_drive_mb:>6.0f} MB"
        )
    lines += [
        f"  TOTAL   {total:>9,}  "
        f"{total * seconds_per_pass / 3600.0:>7.1f}   {drive / (1024 * 1024):>6.0f} MB",
    ]
    return "\n".join(lines)


# ------------------------------------------- architecture and invariance


class ArchitectureMismatch(RuntimeError):
    """A structural assumption the measurement position depends on is false."""


class AudioProtocolMismatch(RuntimeError):
    """The resolved spoken-audio protocol is not the one the study is bound to."""


def assert_audio_protocol(resolved: Any, *, expected_fingerprint: str) -> dict:
    """Hold the probed audio interface to the fingerprint this study is bound to.

    The fingerprint covers the model and processor revisions, the transformers
    version, the call and chat conventions, the sample rate and the placeholder
    token — everything a later run would have to match for its audio inputs to
    mean the same thing. A mismatch is refused rather than warned about,
    because a warning printed into a multi-hour Colab log is not a control.

    Raises:
        AudioProtocolMismatch: When the fingerprints disagree. The message
            prints the observed protocol so the differing field can be found
            without re-running anything.
    """
    observed = resolved.protocol_fingerprint
    if observed != expected_fingerprint:
        raise AudioProtocolMismatch(
            f"resolved audio protocol {observed} != expected {expected_fingerprint}.\n"
            "The expected fingerprint comes from the completed engineering audit. "
            "It covers the model and processor revisions, the transformers "
            "version, the calling convention, the sampling rate and the "
            "placeholder token, so the most common cause is a different "
            "transformers release in this runtime.\n"
            f"observed protocol: {resolved.to_dict()}"
        )
    return {**resolved.to_record(), "matches_expected_fingerprint": True}


def modality_architecture_report(
    backend: Any, inputs_by_modality: Mapping[str, Any], *, layers: Sequence[int]
) -> dict:
    """Where each modality's evidence sits, and where the study reads and edits.

    Checks, per modality, that the final prompt position is the last token, that
    a media span (when the processor exposes one) ends **before** it, and that
    every requested layer exists as a hookable decoder block. For spoken audio
    it additionally requires the placeholder/feature agreement the input was
    built under to be present — a spoken-audio input that never verified its own
    placeholder span is one the model would reject much later, opaquely.

    Raises:
        ArchitectureMismatch: On any violation, naming the modality and the
            assumption.
    """
    problems: list[str] = []
    n_blocks = len(backend.blocks)
    for layer in layers:
        if not 0 <= int(layer) < n_blocks:
            problems.append(
                f"layer {layer} is outside the {n_blocks} decoder blocks this "
                "backend exposes"
            )

    per_modality: dict[str, dict] = {}
    for modality, inputs in sorted(inputs_by_modality.items()):
        span = inputs.modality_token_range
        final_position = inputs.final_prompt_position
        sequence_length = int(inputs.tensors["input_ids"].shape[1])
        entry = {
            "modality": modality,
            "prompt_len": int(inputs.prompt_len),
            "sequence_length": sequence_length,
            "final_prompt_position": final_position,
            "modality_token_span": list(span) if span else None,
            "route": dict(inputs.route),
            "media_checksum": inputs.media_checksum,
        }
        if sequence_length != int(inputs.prompt_len):
            problems.append(
                f"{modality}: input_ids length {sequence_length} != prompt_len "
                f"{inputs.prompt_len}"
            )
        if span:
            entry["final_prompt_position_follows_span"] = final_position >= int(span[1])
            if final_position < int(span[1]):
                problems.append(
                    f"{modality}: the final prompt position {final_position} is "
                    f"inside the media span {list(span)}; the study reads and "
                    "edits a media token, not the prompt"
                )
            entry["n_media_tokens"] = int(span[1]) - int(span[0])
        if modality == SPOKEN_AUDIO:
            # ``audio`` is the record from
            # :func:`jlens.mmpilot.audio.verify_audio_encoding`, which raises on
            # every disagreement rather than reporting one — so its *presence*
            # is the evidence that placeholders and features agreed, that the
            # span was contiguous, and that the count tracked the recording.
            audio = inputs.audio or {}
            native = getattr(backend, "audio_interface", None) is not None
            entry["native_audio_path"] = native
            entry["audio"] = {
                key: audio.get(key)
                for key in (
                    "n_placeholders",
                    "n_expected_from_features",
                    "audio_token_span",
                    "feature_keys",
                    "feature_shapes",
                    "prompt_len",
                    "final_prompt_position",
                )
                if key in audio
            }
            if native and not audio:
                problems.append(
                    "spoken_audio: the backend resolved a native audio interface "
                    "but the input carries no placeholder-verification record"
                )
            if audio:
                placeholders = int(audio.get("n_placeholders", 0))
                expected = int(audio.get("n_expected_from_features", -1))
                entry["placeholder_feature_counts_agree"] = placeholders == expected
                entry["n_audio_placeholders"] = placeholders
                if placeholders != expected:
                    problems.append(
                        f"spoken_audio: {placeholders} placeholder token(s) vs "
                        f"{expected} implied by the feature mask"
                    )
                if placeholders <= 0:
                    problems.append(
                        "spoken_audio: zero audio placeholder tokens — the "
                        "recording never entered the model"
                    )
        per_modality[modality] = entry

    report = {
        "schema": "jlens.mmpilot.tri_modal_architecture.v1",
        "n_decoder_blocks": n_blocks,
        "d_model": int(backend.d_model),
        "layers": [int(layer) for layer in layers],
        "hook_site": "output of the decoder block at each layer (post-block residual)",
        "read_and_edit_position": "final prompt token (prompt_len - 1)",
        "per_modality": per_modality,
        "problems": problems,
        "passed": not problems,
    }
    if problems:
        raise ArchitectureMismatch(
            "architecture checks failed:\n  - " + "\n  - ".join(problems)
        )
    return report


def run_invariance_by_modality(
    backend: Any,
    inputs_by_modality: Mapping[str, Any],
    layers: Sequence[int],
    *,
    tolerance: float = 1e-4,
) -> dict:
    """The capture-no-op and zero-coefficient checks, in **every** modality.

    A gate that passed on text says nothing about whether an image or audio
    forward pass survives the same hook: the media path builds different
    tensors and can take a different branch. Any failure raises, and the caller
    is expected to refuse causal work for that modality rather than record a
    weaker result.
    """
    from jlens.mmpilot.backend import run_invariance_gate

    per_modality = {
        modality: run_invariance_gate(backend, inputs, list(layers), tolerance=tolerance)
        for modality, inputs in sorted(inputs_by_modality.items())
    }
    return {
        "schema": "jlens.mmpilot.tri_modal_invariance.v1",
        "layers": [int(layer) for layer in layers],
        "tolerance": tolerance,
        "per_modality": per_modality,
        "modalities": sorted(per_modality),
        "passed": all(entry["passed"] for entry in per_modality.values()),
    }


# ------------------------------------------------------ representation rows


def representational_rows(report: Mapping, *, pairs: Sequence[str] = ALL_PAIRS) -> list[dict]:
    """One row per requested direction, with everything it is judged on.

    A pair the report does not contain becomes a row with ``evaluated=False``
    rather than being dropped: a missing direction and a failing direction are
    different findings, and a table that silently omits one cannot say which
    happened.
    """
    entries = report.get("pairs") or {}
    out: list[dict] = []
    for pair in pairs:
        entry = entries.get(pair)
        if not entry:
            out.append(
                {
                    "pair": pair,
                    "evaluated": False,
                    "audio_related": SPOKEN_AUDIO in pair,
                    "beats_shuffled": False,
                    "reason": "no retrieval was computed for this direction",
                }
            )
            continue
        retrieval = entry["jspace_retrieval"]
        exclusions = entry.get("exclusions", {})
        jspace_gap = entry["jspace_separation"]["gap"]
        raw_gap = entry["raw_residual_separation"]["gap"]
        top1 = retrieval["top1_accuracy"]
        shuffled = entry["shuffled_control"]["p95_top1_accuracy"]
        out.append(
            {
                "pair": pair,
                "evaluated": True,
                "audio_related": SPOKEN_AUDIO in pair,
                "layer": report.get("layer"),
                "n_sources": entry.get("n_sources"),
                "n_targets": entry.get("n_targets"),
                "n_queries": retrieval["n_queries"],
                "jspace_top1": top1,
                "jspace_mrr": retrieval["mrr"],
                "shuffled_mean": entry["shuffled_control"]["mean_top1_accuracy"],
                "shuffled_p95": shuffled,
                # Accuracy is discrete at 1/n_queries, so a fixed additive
                # margin above p95 can demand an accuracy above 1.0. The stated
                # control is: strictly beat it.
                "beats_shuffled": bool(retrieval["n_queries"]) and top1 > shuffled,
                "jspace_separation_gap": jspace_gap,
                "raw_separation_gap": raw_gap,
                "jspace_separation_beats_raw": jspace_gap > raw_gap,
                "support_overlap_gap": entry["jspace_support_overlap"]["gap"],
                "raw_top1": entry["raw_residual_retrieval"]["top1_accuracy"],
                "eligible_targets": exclusions.get("eligible_targets"),
                "n_excluded_same_group": exclusions.get("n_excluded_same_group"),
                "n_excluded_same_image_different_group": exclusions.get(
                    "n_excluded_same_image_different_group"
                ),
                "n_distinct_source_images": entry.get("n_distinct_source_images"),
                "n_distinct_target_images": entry.get("n_distinct_target_images"),
            }
        )
    return out


def representational_rows_by_layer(
    reports: Mapping[int, Mapping], *, pairs: Sequence[str] = ALL_PAIRS
) -> list[dict]:
    """:func:`representational_rows` for every layer, each row carrying its layer."""
    out: list[dict] = []
    for layer in sorted(int(key) for key in reports):
        for row in representational_rows(reports[layer], pairs=pairs):
            out.append({**row, "layer": int(layer)})
    return out


# ------------------------------------------------------------- causal cells


def _best_row(
    rows: Sequence[Mapping], *, layer: int, concept: str, pair: str, control_kind: str
) -> dict | None:
    """The strongest positive-alpha row for one (layer, concept, pair, control)."""
    candidates = [
        row
        for row in rows
        if int(row["layer"]) == int(layer)
        and row["concept"] == concept
        and row["pair"] == pair
        and row["control_kind"] == control_kind
        and float(row["alpha"]) > 0
    ]
    return (
        max(candidates, key=lambda row: float(row["mean_signed_target_effect"]))
        if candidates
        else None
    )


def _matched(rows: Sequence[Mapping], best: Mapping, control_kind: str) -> dict | None:
    for row in rows:
        if (
            int(row["layer"]) == int(best["layer"])
            and row["concept"] == best["concept"]
            and row["pair"] == best["pair"]
            and float(row["alpha"]) == float(best["alpha"])
            and row["control_kind"] == control_kind
        ):
            return row
    return None


def evaluate_causal_cells(
    interventions: Mapping,
    *,
    layer: int,
    focal_concepts: Sequence[str],
    pairs: Sequence[str] = ALL_PAIRS,
    thresholds: TriModalThresholds,
) -> list[dict]:
    """One verdict per (concept, direction) at one layer, against its own controls.

    ``interventions`` must be the **image-level** aggregate. This study's unit is
    the photograph — and, for the audio arm, the recording that photograph's
    group carries — so a group-level table would let one image's repeated
    captions count more than once.

    Every clause in the list is required. A cell that clears its controls on a
    mean but had the wrong sign on half its photographs has not transferred, and
    neither has one whose distinct images turned out to be fewer than the design
    said.
    """
    rows = list(interventions.get("rows", []))
    low, high = thresholds.norm_ratio_bounds
    out: list[dict] = []
    for concept in focal_concepts:
        for pair in pairs:
            best = _best_row(
                rows, layer=layer, concept=concept, pair=pair, control_kind="source_concept"
            )
            if best is None:
                out.append(
                    {
                        "layer": int(layer),
                        "concept": concept,
                        "pair": pair,
                        "audio_related": SPOKEN_AUDIO in pair,
                        "evaluated": False,
                        "passes": False,
                        "reasons": ["no source-concept row for this cell"],
                    }
                )
                continue
            random_row = _matched(rows, best, "random_norm_matched")
            unrelated_row = _matched(rows, best, "unrelated_concept")
            raw_row = _matched(rows, best, "raw_residual_difference")
            effect = float(best["mean_signed_target_effect"])
            controls = [
                float(row["mean_signed_target_effect"])
                for row in (random_row, unrelated_row)
                if row is not None
            ]
            strongest = max(controls) if controls else 0.0

            reasons: list[str] = []
            if not math.isfinite(effect) or effect <= 0:
                reasons.append(f"effect {effect:+.4f} is not positive")
            if float(best["fraction_expected_sign"]) < thresholds.min_fraction_expected_sign:
                reasons.append(
                    f"expected-sign fraction {best['fraction_expected_sign']:.2f} < "
                    f"{thresholds.min_fraction_expected_sign}"
                )
            if len(controls) < 2:
                reasons.append("a matched random or unrelated-concept control is missing")
            elif not (
                effect > float(random_row["mean_signed_target_effect"])
                and effect > float(unrelated_row["mean_signed_target_effect"])
                and effect >= thresholds.control_separation_factor * max(strongest, 0.0)
            ):
                reasons.append(
                    f"effect {effect:+.4f} does not clear "
                    f"{thresholds.control_separation_factor}x the strongest control "
                    f"({strongest:+.4f})"
                )
            ratio = float(best["mean_activation_norm_ratio"])
            if not low <= ratio <= high:
                reasons.append(f"activation norm ratio {ratio:.3f} outside [{low}, {high}]")
            unrelated_change = float(best["mean_abs_unrelated_change"])
            if unrelated_change >= abs(effect):
                reasons.append(
                    f"unrelated candidates moved {unrelated_change:.4f}, not less than "
                    f"the target's {abs(effect):.4f} — the edit looks global"
                )
            n_positive = int(best.get("n_positive_images", 0))
            n_negative = int(best.get("n_negative_images", 0))
            if n_positive < thresholds.required_positive_images_per_cell:
                reasons.append(
                    f"{n_positive} distinct positive image(s) < "
                    f"{thresholds.required_positive_images_per_cell}"
                )
            if n_negative < thresholds.required_negative_images_per_cell:
                reasons.append(
                    f"{n_negative} distinct negative image(s) < "
                    f"{thresholds.required_negative_images_per_cell}"
                )
            floor = thresholds.min_distinct_images_supporting_a_claim
            beats_raw = raw_row is None or effect >= float(
                raw_row["mean_signed_target_effect"]
            )
            out.append(
                {
                    "layer": int(layer),
                    "concept": concept,
                    "pair": pair,
                    "audio_related": SPOKEN_AUDIO in pair,
                    "evaluated": True,
                    "alpha": float(best["alpha"]),
                    "mean_signed_target_effect": effect,
                    "fraction_expected_sign": float(best["fraction_expected_sign"]),
                    "mean_activation_norm_ratio": ratio,
                    "mean_abs_unrelated_change": unrelated_change,
                    "n_distinct_images": best.get("n_distinct_images"),
                    "n_positive_images": n_positive,
                    "n_negative_images": n_negative,
                    "meets_claim_image_floor": bool(
                        n_positive >= floor and n_negative >= floor
                    ),
                    "random_control": (random_row or {}).get("mean_signed_target_effect"),
                    "unrelated_control": (unrelated_row or {}).get(
                        "mean_signed_target_effect"
                    ),
                    "raw_residual_control": (raw_row or {}).get("mean_signed_target_effect"),
                    # Reported and allowed to downgrade, never to veto on its
                    # own: the raw difference-in-means direction answers "did
                    # the decomposition earn its keep", not "did transfer
                    # happen".
                    "jspace_beats_raw_direction": beats_raw,
                    "passes": not reasons,
                    "reasons": reasons,
                }
            )
    return out


# ---------------------------------------------------------------- verdict A


def audio_capability_verdict(
    capability: Mapping,
    *,
    selected_concepts: Sequence[str],
    modalities: Sequence[str],
    thresholds: TriModalThresholds,
) -> dict:
    """Can the model read each concept out of *all three* channels?

    Nothing about activations means anything for a concept the model cannot
    recognize behaviorally, so this decides whether the expensive stages may run
    at all. A concept is never replaced after its capability result is seen.
    """
    per_concept = capability.get("per_concept", {})
    all_three = sorted(capability.get("retained_concepts", []))
    passing = [concept for concept in selected_concepts if concept in set(all_three)]
    audio_available = SPOKEN_AUDIO in modalities
    enough = len(passing) >= thresholds.min_concepts_all_three_modalities

    per_modality_counts = {
        modality: sorted(
            concept
            for concept in selected_concepts
            if per_concept.get(concept, {}).get(modality, {}).get("passed", False)
        )
        for modality in modalities
    }
    verdict = AUDIO_CAPABILITY_GO if (audio_available and enough) else AUDIO_CAPABILITY_NO_GO
    if not audio_available:
        rationale = (
            "spoken_audio was not an available modality, so no three-modality "
            "capability claim exists. The channel is reported blocked, never faked."
        )
    elif enough:
        rationale = (
            f"{len(passing)} of {len(selected_concepts)} selected concepts cleared "
            f"the {thresholds.capability_threshold:.0%} gate in text, image AND "
            f"spoken audio: {passing}. This is behavioral capability only — it is "
            "not evidence that J-space coordinates transfer."
        )
    else:
        rationale = (
            f"only {len(passing)} concept(s) cleared the gate in all three "
            f"modalities ({passing}); {thresholds.min_concepts_all_three_modalities} "
            "are required. The expensive stages are skipped rather than run on a "
            "capability the model does not have."
        )
    return {
        "schema": "jlens.mmpilot.audio_capability_verdict.v1",
        "verdict": verdict,
        "rationale": rationale,
        "threshold": capability.get("threshold", thresholds.capability_threshold),
        "modalities_evaluated": list(modalities),
        "spoken_audio_available": audio_available,
        "selected_concepts": list(selected_concepts),
        "concepts_passing_all_three": passing,
        "concepts_passing_text_and_image": sorted(
            capability.get("text_image_retained_concepts", [])
        ),
        "concepts_passing_per_modality": per_modality_counts,
        "required": thresholds.min_concepts_all_three_modalities,
        "per_concept": per_concept,
        "scope": SPOKEN_AUDIO_SCOPE,
    }


# ---------------------------------------------------------------- verdict B


def representational_transfer_verdict(
    reports: Mapping[int, Mapping],
    *,
    thresholds: TriModalThresholds,
    primary_layer: int,
) -> dict:
    """Do the six directional pairs beat their shuffled controls?

    The decision clause is about the **audio-related** pairs. Text <-> image is
    computed under exactly the same rule and reported beside them as an internal
    replication, so a reader can see whether the study reproduced the confirmed
    result — but reproducing it is not what this verdict is for.
    """
    rows = representational_rows_by_layer(reports)
    audio_rows = [row for row in rows if row["audio_related"]]
    replication_rows = [row for row in rows if not row["audio_related"]]
    evaluated_audio = [row for row in audio_rows if row["evaluated"]]

    audio_beating = [row for row in evaluated_audio if row["beats_shuffled"]]
    at_primary = [
        row
        for row in evaluated_audio
        if int(row["layer"]) == int(primary_layer) and row["beats_shuffled"]
    ]
    directions_beating = sorted({row["pair"] for row in audio_beating})
    both_directions = {
        "text<->spoken_audio": {"text->spoken_audio", "spoken_audio->text"}
        <= set(directions_beating),
        "image<->spoken_audio": {"image->spoken_audio", "spoken_audio->image"}
        <= set(directions_beating),
    }
    replication_beating = sorted(
        {row["pair"] for row in replication_rows if row["evaluated"] and row["beats_shuffled"]}
    )

    if not evaluated_audio:
        verdict = TRANSFER_NOT_EVALUATED
        rationale = (
            "no audio-related retrieval was computed, so nothing is claimed "
            "about representational transfer to or from spoken audio."
        )
    elif len(directions_beating) >= thresholds.min_audio_pairs_beating_shuffled and (
        any(both_directions.values())
        or not thresholds.require_both_audio_directions_representational
    ):
        verdict = TRANSFER_SUPPORTED if at_primary else TRANSFER_WEAK
        rationale = (
            f"{len(directions_beating)} audio-related direction(s) beat the "
            f"shuffled-label p95 under image-disjoint retrieval: "
            f"{directions_beating}."
            + (
                ""
                if at_primary
                else f" None of them did so at the primary layer {primary_layer}, "
                "so the structure is reported as weak."
            )
        )
    else:
        verdict = TRANSFER_UNSUPPORTED
        rationale = (
            "no audio-related direction beat its shuffled-label control. The "
            "J-space codes carry no cross-modal structure this test can detect "
            "between spoken audio and the other channels."
        )
    return {
        "schema": "jlens.mmpilot.representational_transfer_verdict.v1",
        "verdict": verdict,
        "rationale": rationale,
        "primary_layer": int(primary_layer),
        "layers": sorted(int(key) for key in reports),
        "pairs": list(ALL_PAIRS),
        "audio_pairs": list(AUDIO_PAIRS),
        "replication_pairs": list(REPLICATION_PAIRS),
        "rows": rows,
        "audio_directions_beating_shuffled": directions_beating,
        "audio_directions_beating_shuffled_at_primary_layer": sorted(
            {row["pair"] for row in at_primary}
        ),
        "bidirectional": both_directions,
        "replication_directions_beating_shuffled": replication_beating,
        "text_image_replicates": len(replication_beating) == len(REPLICATION_PAIRS),
        "thresholds": thresholds.to_dict(),
        "reading": (
            "Retrieval and separation are representational evidence only. Causal "
            "transfer is tested separately and is not implied by these numbers."
        ),
        "scope": SPOKEN_AUDIO_SCOPE,
    }


# ------------------------------------------------------------- verdicts C/D


def causal_transfer_verdict(
    interventions: Mapping,
    *,
    layer: int,
    focal_concepts: Sequence[str],
    thresholds: TriModalThresholds,
    name: str,
) -> dict:
    """One layer's causal result, decided on the audio-related cells.

    Text <-> image cells are evaluated identically and reported beside them, and
    they can never carry this verdict on their own: a study that concluded
    "three-modality transfer" from a text/image cell would be reporting the
    replication as the finding.
    """
    cells = evaluate_causal_cells(
        interventions,
        layer=layer,
        focal_concepts=focal_concepts,
        pairs=ALL_PAIRS,
        thresholds=thresholds,
    )
    audio_cells = [cell for cell in cells if cell["audio_related"]]
    passing_audio = [cell for cell in audio_cells if cell["passes"]]
    claim_supported = [cell for cell in passing_audio if cell.get("meets_claim_image_floor")]
    passing_replication = [
        cell for cell in cells if not cell["audio_related"] and cell["passes"]
    ]
    directions = sorted({cell["pair"] for cell in claim_supported})
    bidirectional_audio = sorted(
        {
            cell["concept"]
            for cell in claim_supported
            if {
                other["pair"]
                for other in claim_supported
                if other["concept"] == cell["concept"]
            }
            & {"spoken_audio->text", "spoken_audio->image"}
            and {
                other["pair"]
                for other in claim_supported
                if other["concept"] == cell["concept"]
            }
            & {"text->spoken_audio", "image->spoken_audio"}
        }
    )
    raw_exceptions = [
        {"concept": cell["concept"], "pair": cell["pair"]}
        for cell in claim_supported
        if not cell["jspace_beats_raw_direction"]
    ]

    if not any(cell["evaluated"] for cell in audio_cells):
        verdict = TRANSFER_NOT_EVALUATED
        rationale = (
            f"no audio-related intervention rows exist at layer {layer}; nothing "
            "is claimed about causal transfer there."
        )
    elif len(claim_supported) >= thresholds.min_audio_causal_cells and bidirectional_audio:
        verdict = TRANSFER_SUPPORTED
        rationale = (
            f"{len(claim_supported)} audio-related off-diagonal cell(s) at layer "
            f"{layer} moved the target concept in the expected direction against "
            f"their own matched random and external unrelated controls, in both "
            f"directions for {bidirectional_audio}: {directions}."
        )
    elif claim_supported:
        verdict = TRANSFER_WEAK
        rationale = (
            f"{len(claim_supported)} audio-related cell(s) at layer {layer} passed "
            f"({directions}), but no focal concept transferred in both directions, "
            "so the effect is reported as one-directional."
        )
    else:
        verdict = TRANSFER_UNSUPPORTED
        failures = sorted(
            {reason for cell in audio_cells for reason in cell.get("reasons", [])}
        )
        rationale = (
            f"no audio-related cell at layer {layer} cleared its controls. "
            f"Reasons seen: {failures[:5]}"
        )
    return {
        "schema": "jlens.mmpilot.causal_transfer_verdict.v1",
        "name": name,
        "verdict": verdict,
        "rationale": rationale,
        "layer": int(layer),
        "focal_concepts": list(focal_concepts),
        "cells": cells,
        "audio_cells_passing": [
            {"concept": cell["concept"], "pair": cell["pair"]} for cell in passing_audio
        ],
        "audio_cells_supporting_a_claim": [
            {"concept": cell["concept"], "pair": cell["pair"]} for cell in claim_supported
        ],
        "audio_directions_passing": directions,
        "concepts_transferring_both_audio_directions": bidirectional_audio,
        "replication_cells_passing": [
            {"concept": cell["concept"], "pair": cell["pair"]}
            for cell in passing_replication
        ],
        "raw_direction_exceptions": raw_exceptions,
        "thresholds": thresholds.to_dict(),
        "layer_choice_note": LAYER_CHOICE_NOTE,
        "scope": SPOKEN_AUDIO_SCOPE,
    }


def replication_verdict(
    verdicts: Mapping[int, Mapping],
    *,
    primary: Mapping | None,
    layers: Sequence[int],
) -> dict:
    """Did the frozen Stage-B design reproduce at the remaining layers?

    A layer that was never run is ``NOT_EVALUATED``, never a failure. The design
    is frozen before Stage C runs, and no layer may be selected on its causal
    outcome — this function only reads results, it never chooses a layer.
    """
    evaluated = {
        int(layer): verdicts[int(layer)] for layer in layers if int(layer) in verdicts
    }
    if not evaluated:
        return {
            "schema": "jlens.mmpilot.replication_verdict.v1",
            "verdict": TRANSFER_NOT_EVALUATED,
            "rationale": (
                f"layers {sorted(int(x) for x in layers)} were not run. Stage C is "
                "behind its own confirmation flag and does not start automatically."
            ),
            "layers_requested": sorted(int(x) for x in layers),
            "layers_evaluated": [],
            "per_layer": {},
            "primary_verdict": (primary or {}).get("verdict"),
            "layer_choice_note": LAYER_CHOICE_NOTE,
        }
    supported = sorted(
        layer for layer, entry in evaluated.items() if entry["verdict"] == TRANSFER_SUPPORTED
    )
    weak = sorted(
        layer for layer, entry in evaluated.items() if entry["verdict"] == TRANSFER_WEAK
    )
    if supported:
        verdict = TRANSFER_SUPPORTED
        rationale = f"the frozen design reproduced at layer(s) {supported}."
    elif weak:
        verdict = TRANSFER_WEAK
        rationale = f"layer(s) {weak} reproduced only one-directionally."
    else:
        verdict = TRANSFER_UNSUPPORTED
        rationale = (
            f"the design did not reproduce at layer(s) {sorted(evaluated)}. This is "
            "a result about those layers, not a reason to prefer another one."
        )
    return {
        "schema": "jlens.mmpilot.replication_verdict.v1",
        "verdict": verdict,
        "rationale": rationale,
        "layers_requested": sorted(int(x) for x in layers),
        "layers_evaluated": sorted(evaluated),
        "layers_supported": supported,
        "layers_weak": weak,
        "per_layer": {
            str(layer): entry["verdict"] for layer, entry in sorted(evaluated.items())
        },
        "primary_verdict": (primary or {}).get("verdict"),
        "layer_choice_note": LAYER_CHOICE_NOTE,
    }


# ---------------------------------------------------------------- verdict E


def overall_verdict(
    *,
    capability: Mapping,
    representational: Mapping,
    primary_causal: Mapping | None,
    replication: Mapping,
    invariance: Mapping | None = None,
    blocked_modalities: Sequence[str] = (),
    thresholds: TriModalThresholds | None = None,
) -> dict:
    """The three-modality decision, from the four verdicts above.

    A GO requires every clause: three-modality behavioral capability, J-space
    structure beating the shuffled control for audio-related pairs, at least one
    audio-related off-diagonal causal cell with the expected sign clearing its
    matched random and external unrelated controls, specificity rather than
    global candidate disruption, sane activation norms, at least two distinct
    target images/recordings behind each claimed cell, and — explicitly — a
    result that is not text <-> image replication alone.
    """
    thresholds = thresholds or TriModalThresholds()
    criteria: dict[str, dict] = {}

    criteria["three_modality_capability"] = criterion(
        PASS if capability.get("verdict") == AUDIO_CAPABILITY_GO else FAIL,
        {
            "concepts_passing_all_three": capability.get("concepts_passing_all_three"),
            "required": capability.get("required"),
            "threshold": capability.get("threshold"),
            "blocked_modalities": list(blocked_modalities),
            "reading": (
                "behavioral capability in all three channels; it is not evidence "
                "of J-space transfer and never becomes one"
            ),
        },
    )

    representational_status = {
        TRANSFER_SUPPORTED: PASS,
        TRANSFER_WEAK: PASS,
        TRANSFER_UNSUPPORTED: FAIL,
        TRANSFER_NOT_EVALUATED: NOT_EVALUATED,
    }[representational.get("verdict", TRANSFER_NOT_EVALUATED)]
    criteria["audio_jspace_beats_shuffled"] = criterion(
        representational_status,
        {
            "audio_directions_beating_shuffled": representational.get(
                "audio_directions_beating_shuffled"
            ),
            "bidirectional": representational.get("bidirectional"),
            "text_image_replicates": representational.get("text_image_replicates"),
        },
        skipped_because=(
            "no audio-related retrieval was computed"
            if representational_status == NOT_EVALUATED
            else None
        ),
    )

    causal = primary_causal or {}
    causal_skip = (
        None
        if causal and causal.get("verdict") != TRANSFER_NOT_EVALUATED
        else "the causal stage did not run, or produced no audio-related rows"
    )
    claimed = list(causal.get("audio_cells_supporting_a_claim", []))
    criteria["audio_causal_cell_with_expected_sign"] = criterion(
        NOT_EVALUATED
        if causal_skip
        else (PASS if len(claimed) >= thresholds.min_audio_causal_cells else FAIL),
        {
            "audio_cells_supporting_a_claim": claimed,
            "audio_directions_passing": causal.get("audio_directions_passing"),
            "required": thresholds.min_audio_causal_cells,
            "layer": causal.get("layer"),
        },
        skipped_because=causal_skip,
    )

    cells = list(causal.get("cells", []))
    audio_cells = [cell for cell in cells if cell["audio_related"] and cell["evaluated"]]
    nonspecific = [
        {"concept": cell["concept"], "pair": cell["pair"]}
        for cell in audio_cells
        if cell["mean_abs_unrelated_change"] >= abs(cell["mean_signed_target_effect"])
    ]
    criteria["effects_are_specific"] = criterion(
        NOT_EVALUATED
        if causal_skip
        else (PASS if audio_cells and not nonspecific else FAIL),
        {
            "cells_where_unrelated_candidates_moved_at_least_as_much": nonspecific,
            "reading": (
                "an edit that moves every candidate as much as the target is a "
                "global disruption, not concept transfer"
            ),
        },
        skipped_because=causal_skip,
    )

    low, high = thresholds.norm_ratio_bounds
    insane_norms = [
        {
            "concept": cell["concept"],
            "pair": cell["pair"],
            "ratio": cell["mean_activation_norm_ratio"],
        }
        for cell in audio_cells
        if not low <= float(cell["mean_activation_norm_ratio"]) <= high
    ]
    criteria["activation_norms_sane"] = criterion(
        NOT_EVALUATED if causal_skip else (PASS if not insane_norms else FAIL),
        {"bounds": [low, high], "violations": insane_norms},
        skipped_because=causal_skip,
    )

    thin = [
        {
            "concept": cell["concept"],
            "pair": cell["pair"],
            "n_positive_images": cell["n_positive_images"],
            "n_negative_images": cell["n_negative_images"],
        }
        for cell in audio_cells
        if cell["passes"] and not cell.get("meets_claim_image_floor")
    ]
    criteria["two_distinct_images_support_each_claim"] = criterion(
        NOT_EVALUATED if causal_skip else (PASS if not thin else FAIL),
        {
            "floor": thresholds.min_distinct_images_supporting_a_claim,
            "cells_below_the_floor": thin,
        },
        skipped_because=causal_skip,
    )

    audio_evidence = bool(
        representational.get("audio_directions_beating_shuffled") or claimed
    )
    criteria["not_text_image_replication_alone"] = criterion(
        PASS if audio_evidence else FAIL,
        {
            "audio_representational_directions": representational.get(
                "audio_directions_beating_shuffled"
            ),
            "audio_causal_cells": claimed,
            "text_image_replication": {
                "representational": representational.get(
                    "replication_directions_beating_shuffled"
                ),
                "causal": causal.get("replication_cells_passing"),
            },
            "reading": (
                "the confirmed text/image result is an internal replication here; "
                "it can never be the evidence for a three-modality claim"
            ),
        },
    )

    invariance_passed = bool((invariance or {}).get("passed"))
    criteria["invariance_gate"] = criterion(
        PASS if invariance_passed else (NOT_EVALUATED if invariance is None else FAIL),
        {
            "modalities": sorted((invariance or {}).get("per_modality", {})),
            "reading": (
                "capture must be a no-op and a zero-coefficient edit must "
                "reproduce the clean logits, in every modality that is intervened on"
            ),
        },
        skipped_because="the invariance gate did not run" if invariance is None else None,
    )

    criteria["environmental_audio_not_claimed"] = criterion(
        PASS,
        {
            "reading": SPOKEN_AUDIO_SCOPE,
            "modality_tested": SPOKEN_AUDIO,
        },
    )

    statuses = {name: entry["status"] for name, entry in criteria.items()}
    go_requirements = (
        "three_modality_capability",
        "audio_jspace_beats_shuffled",
        "audio_causal_cell_with_expected_sign",
        "effects_are_specific",
        "activation_norms_sane",
        "two_distinct_images_support_each_claim",
        "not_text_image_replication_alone",
        "invariance_gate",
    )
    all_required = all(statuses.get(name) == PASS for name in go_requirements)

    # A WEAK GO is for causal evidence that is *thin* — untested, or real but
    # one-directional. It is never for causal evidence that actively contradicts
    # the claim. A non-specific edit, an activation norm off its rails, a failed
    # invariance gate, or interventions indistinguishable from random are all
    # NO-GO: each one means an intervention's effect at that layer has no
    # interpretation, which is worse than not having measured it.
    blocking = [
        name
        for name in (
            "three_modality_capability",
            "audio_jspace_beats_shuffled",
            "not_text_image_replication_alone",
            "effects_are_specific",
            "activation_norms_sane",
            "two_distinct_images_support_each_claim",
            "invariance_gate",
        )
        if statuses.get(name) == FAIL
    ]
    causal_verdict = causal.get("verdict", TRANSFER_NOT_EVALUATED)
    causal_is_thin_not_contradictory = causal_verdict in (
        TRANSFER_WEAK,
        TRANSFER_NOT_EVALUATED,
    )
    representation_only = (
        not blocking
        and statuses["three_modality_capability"] == PASS
        and statuses["audio_jspace_beats_shuffled"] == PASS
        and statuses["not_text_image_replication_alone"] == PASS
        and causal_is_thin_not_contradictory
    )

    if all_required and causal_verdict == TRANSFER_WEAK:
        # Every clause passed, but no focal concept transferred in both audio
        # directions. A one-directional effect is a real result and not a full
        # one: "the coordinate carries into audio" and "the coordinate is the
        # same coordinate in both channels" are different claims.
        verdict = THREE_MODALITY_WEAK_GO
        rationale = (
            "Every requirement was met, but the audio-related causal transfer "
            "was one-directional: "
            f"{causal.get('audio_directions_passing')}. No focal concept "
            "transferred in both directions to and from spoken audio, so the "
            "claim is downgraded rather than stated in full."
        )
    elif all_required:
        verdict = THREE_MODALITY_GO
        rationale = (
            "Behavioral capability held in all three channels, J-space structure "
            "beat the shuffled control for audio-related directions, and at least "
            "one audio-related off-diagonal cell moved the target concept in the "
            "expected direction against its own matched random and external "
            "unrelated controls, specifically and on more than one photograph. "
            "The result does not rest on text <-> image replication."
        )
    elif representation_only:
        verdict = THREE_MODALITY_WEAK_GO
        rationale = (
            "Representational transfer to or from spoken audio beat its controls, "
            "but the causal evidence is weak, one-directional, or absent: "
            f"{sorted(name for name in go_requirements if statuses.get(name) != PASS)}. "
            "Coordinates that retrieve across channels are not, by themselves, "
            "coordinates the model uses."
        )
    else:
        failed = sorted(name for name in go_requirements if statuses.get(name) == FAIL)
        verdict = THREE_MODALITY_NO_GO
        rationale = (
            "The three-modality claim is not supported. "
            + (f"Failing: {failed}. " if failed else "Required stages did not run. ")
            + (
                "No audio-related intervention cleared its controls, so the "
                "edits are not distinguishable from the random and unrelated "
                "directions run through the same code path."
                if causal_verdict == TRANSFER_UNSUPPORTED
                else ""
            )
        ).strip()

    return {
        "schema": "jlens.mmpilot.three_modality_verdict.v1",
        "version": TRI_MODAL_VERDICT_VERSION,
        "verdict": verdict,
        "rationale": rationale,
        "criteria": criteria,
        "criteria_status": statuses,
        "go_requirements": list(go_requirements),
        "component_verdicts": {
            "A_audio_capability": capability.get("verdict"),
            "B_representational_transfer": representational.get("verdict"),
            "C_l35_causal_transfer": causal.get("verdict", TRANSFER_NOT_EVALUATED),
            "D_l38_l40_replication": replication.get("verdict"),
        },
        "thresholds": thresholds.to_dict(),
        "layer_choice_note": LAYER_CHOICE_NOTE,
        "scope_limitation": SPOKEN_AUDIO_SCOPE,
        "intervention_limitation": (
            "Interventions add and subtract a direction on the residual stream at "
            "the final prompt token. That is not erasure and not projection "
            "ablation."
        ),
        "alignment_limitation": (
            "No cross-modal alignment, adapter, probe or projection was fitted. "
            "The lenses are frozen, text-calibrated and published elsewhere."
        ),
    }


# ------------------------------------------------------------------ report


def _number(value) -> str:
    return "n/a" if value is None else f"{float(value):+.4f}"


def render_report(
    *,
    run_dir: str,
    capability: Mapping,
    representational: Mapping,
    primary_causal: Mapping | None,
    replication: Mapping,
    overall: Mapping,
    lens_report: Mapping | None,
    audio_protocol: Mapping | None,
    budgets: Sequence[Mapping] = (),
    resume: Mapping | None = None,
    mode: str = "mock",
) -> str:
    """The study report: the five verdicts, then the tables they came from."""
    mode_note = (
        "  \n- **MOCK run: pipeline evidence only, not scientific evidence.**"
        if str(mode).lower() == "mock"
        else ""
    )
    lines = [
        f"# Three-modality J-space transfer — {overall['verdict']}",
        "",
        f"- mode: **{mode}**{mode_note}",
        f"- run directory: `{run_dir}`",
        f"- verdict protocol: `{overall['version']}`",
        "",
        "## Verdicts",
        "",
        "| # | verdict | outcome |",
        "| --- | --- | --- |",
        f"| A | AUDIO_CAPABILITY | {capability.get('verdict')} |",
        f"| B | REPRESENTATIONAL_TRANSFER | {representational.get('verdict')} |",
        f"| C | L35_CAUSAL_TRANSFER | "
        f"{(primary_causal or {}).get('verdict', TRANSFER_NOT_EVALUATED)} |",
        f"| D | L38_L40_REPLICATION | {replication.get('verdict')} |",
        f"| E | OVERALL_THREE_MODALITY_VERDICT | **{overall['verdict']}** |",
        "",
        f"**{overall['verdict']}** — {overall['rationale']}",
        "",
        "## A. Behavioral capability in all three channels",
        "",
        f"{capability.get('rationale')}",
        "",
        f"- concepts passing text + image + spoken audio: "
        f"{capability.get('concepts_passing_all_three')}",
        f"- concepts passing text + image only: "
        f"{capability.get('concepts_passing_text_and_image')}",
        "",
        "## B. Representation — all six directions, image-disjoint",
        "",
        "| layer | direction | audio | queries | J top-1 | J MRR | shuffled p95 | "
        "beats shuffled | J gap | raw gap | support gap | src imgs | tgt imgs | "
        "same-image excl |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | "
        "--- | --- |",
    ]
    for row in representational.get("rows", []):
        if not row.get("evaluated"):
            lines.append(
                f"| {row.get('layer')} | {row['pair']} | "
                f"{'yes' if row['audio_related'] else 'no'} | | | | | not evaluated "
                "| | | | | | |"
            )
            continue
        lines.append(
            f"| {row['layer']} | {row['pair']} | "
            f"{'yes' if row['audio_related'] else 'no'} | {row['n_queries']} | "
            f"{row['jspace_top1']:.3f} | {row['jspace_mrr']:.3f} | "
            f"{row['shuffled_p95']:.3f} | "
            f"{'yes' if row['beats_shuffled'] else 'no'} | "
            f"{row['jspace_separation_gap']:+.4f} | {row['raw_separation_gap']:+.4f} | "
            f"{row['support_overlap_gap']:+.4f} | "
            f"{row.get('n_distinct_source_images', 'n/a')} | "
            f"{row.get('n_distinct_target_images', 'n/a')} | "
            f"{row.get('n_excluded_same_image_different_group')} |"
        )
    lines += ["", representational.get("rationale", ""), ""]

    for label, entry in (
        ("C. Primary causal test", primary_causal),
        ("D. Replication", replication),
    ):
        if label.startswith("D"):
            lines += [
                f"## {label}",
                "",
                f"{replication.get('rationale')}",
                f"- per layer: {replication.get('per_layer')}",
                "",
            ]
            continue
        lines += [f"## {label}", ""]
        if not entry:
            lines += ["Not run. Stage B is behind its own confirmation flag.", ""]
            continue
        lines += [
            f"Layer {entry['layer']}. {entry['rationale']}",
            "",
            "| concept | direction | audio | alpha | +img | -img | effect | sign frac | "
            "random | unrelated | raw | norm | passes |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | "
            "--- |",
        ]
        for cell in entry.get("cells", []):
            if not cell.get("evaluated"):
                lines.append(
                    f"| {cell['concept']} | {cell['pair']} | "
                    f"{'yes' if cell['audio_related'] else 'no'} | | | | | | | | | | "
                    "not evaluated |"
                )
                continue
            lines.append(
                f"| {cell['concept']} | {cell['pair']} | "
                f"{'yes' if cell['audio_related'] else 'no'} | {cell['alpha']:g} | "
                f"{cell['n_positive_images']} | {cell['n_negative_images']} | "
                f"{cell['mean_signed_target_effect']:+.4f} | "
                f"{cell['fraction_expected_sign']:.2f} | "
                f"{_number(cell['random_control'])} | "
                f"{_number(cell['unrelated_control'])} | "
                f"{_number(cell['raw_residual_control'])} | "
                f"{cell['mean_activation_norm_ratio']:.3f} | "
                f"{'yes' if cell['passes'] else 'no'} |"
            )
        failures = [
            f"- **{cell['concept']} {cell['pair']}**: " + "; ".join(cell["reasons"])
            for cell in entry.get("cells", [])
            if not cell["passes"] and cell.get("reasons")
        ]
        if failures:
            lines += ["", "Why a cell did not pass:", "", *failures]
        lines.append("")

    lines += ["## E. Criteria", "", "| criterion | status |", "| --- | --- |"]
    lines += [
        f"| {name} | {status.replace('_', ' ')} |"
        for name, status in overall["criteria_status"].items()
    ]

    if lens_report:
        lines += [
            "",
            "## The lenses",
            "",
            "Frozen, text-calibrated, published and independently confirmed "
            "elsewhere. **Nothing was fitted in this run.**",
            "",
            "| layer | checksum |",
            "| --- | --- |",
        ]
        lines += [
            f"| {layer} | `{value}` |"
            for layer, value in sorted((lens_report.get("checksums") or {}).items())
        ]
        lines.append(f"\n- combined digest: `{lens_report.get('combined_checksum')}`")
    if audio_protocol:
        lines += [
            "",
            "## The spoken-audio protocol",
            "",
            f"- protocol: `{audio_protocol.get('protocol_version')}`",
            f"- fingerprint: `{audio_protocol.get('protocol_fingerprint')}`",
            f"- calling convention: `{audio_protocol.get('call_convention')}`",
            f"- dynamic placeholder count: "
            f"{audio_protocol.get('dynamic_placeholder_count')}",
        ]
    if budgets:
        lines += ["", "## What it cost", "", "| stage | passes |", "| --- | --- |"]
        lines += [
            f"| {budget.get('stage')} | {budget.get('total_passes'):,} |"
            for budget in budgets
        ]
    if resume:
        lines += [
            "",
            "## Resume",
            "",
            f"- state: {resume.get('status')}",
            f"- units: {resume.get('completed_units')}",
        ]
    lines += [
        "",
        "## Scope and limits",
        "",
        f"- {overall['scope_limitation']}",
        f"- {overall['layer_choice_note']}",
        f"- {overall['intervention_limitation']}",
        f"- {overall['alignment_limitation']}",
        "- One photograph contributes one synchronized group, and each group "
        "contributes one recording. Sibling captions were excluded at selection, "
        "not averaged away afterwards.",
        "- Behavioral capability in a channel is not evidence that J-space "
        "coordinates transfer through it; the engineering readiness of the audio "
        "path is not evidence either.",
        "",
    ]
    return "\n".join(lines)


__all__ = [
    "ALL_PAIRS",
    "AUDIO_CAPABILITY_GO",
    "AUDIO_CAPABILITY_NO_GO",
    "AUDIO_PAIRS",
    "LAYER_CHOICE_NOTE",
    "REPLICATION_PAIRS",
    "SPOKEN_AUDIO",
    "SPOKEN_AUDIO_SCOPE",
    "THREE_MODALITY_GO",
    "THREE_MODALITY_NO_GO",
    "THREE_MODALITY_WEAK_GO",
    "TRANSFER_NOT_EVALUATED",
    "TRANSFER_SUPPORTED",
    "TRANSFER_UNSUPPORTED",
    "TRANSFER_WEAK",
    "TRI_MODAL_VERDICT_VERSION",
    "ArchitectureMismatch",
    "AudioProtocolMismatch",
    "StageBudget",
    "TriModalThresholds",
    "assert_audio_protocol",
    "audio_capability_verdict",
    "causal_transfer_verdict",
    "estimate_stage_passes",
    "evaluate_causal_cells",
    "format_stage_budget",
    "format_total_budget",
    "modality_architecture_report",
    "overall_verdict",
    "render_report",
    "replication_verdict",
    "representational_rows",
    "representational_rows_by_layer",
    "representational_transfer_verdict",
    "run_invariance_by_modality",
]
