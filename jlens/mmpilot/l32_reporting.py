# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""How the layer-32 follow-up *says* what it measured. No science lives here.

The first real L32 run produced correct numbers and reported several of them
badly. This module fixes the reporting and nothing else: no threshold, criterion
digest, prompt protocol, concept set, sample, intervention, control, alpha,
fingerprint or verdict is touched, and no stored unit is rewritten. Every
function here either formats something already computed or re-derives a display
table from stored rows with the same frozen functions that produced them —
and :func:`recompute_convergence_view` refuses if that re-derivation disagrees
with what the run recorded.

The four defects this exists for
================================

**A generic criterion.** The audit printed
:data:`jlens.mmpilot.convergence.CRITERION_TEXT`, which is the completed
three-modality study's rule and is titled "L35 / L38 / L40". It is the wrong
document for a single-layer L32 audit, and it names layer 35 as the primary
throughout. :func:`format_l32_criterion` writes the L32 statement instead. The
historical constant is **not** mutated — it is the record of what the earlier
audit ran under, and editing it would rewrite that run's protocol.

**Metric keys that do not exist.** The cell table asked for ``unique_top1_rate``
and ``median_entropy``; :func:`jlens.mmpilot.convergence.summarize_cell` stores
``unique_top1_target_rate`` and ``median_candidate_entropy_nats``. A missing key
read through ``.get`` prints ``None``, which is indistinguishable from a
measurement that came back empty. :func:`convergence_cell_rows` reads the names
that exist and refuses a cell that is missing one.

**Controls read one level too shallow.** ``per_layer[layer][variant]`` is
``None`` because the variants live under ``per_layer[layer]["controls"]``. Worse,
:func:`jlens.mmpilot.convergence.summarize_controls` ``continue``\\ s past a
variant with no rows, so an absent control leaves ``all_controls_passed`` True.
:func:`assert_controls_complete` closes that: **missing is not passing**.

**Two counts that were both right.** Verdict C reported one passing
audio-related cell; verdict E reported two passing off-diagonal cells. Both are
correct — the second cell is ``text->image``, which is cross-modal but not
audio-related, and the audio verdict is deliberately decided on the audio arm
alone. :func:`causal_cell_breakdown` prints the buckets so the arithmetic is
visible instead of looking like a contradiction.

The amendment path
==================

:func:`build_reporting_amendment` reads a completed run **read-only** and
returns a v2 reporting view bound to the original report's checksum, the run
fingerprint, a source-unit digest and this module's version.
:func:`write_reporting_amendment` writes it beside the original under new
names, atomically, and refuses to touch the original.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

from jlens.metadata import file_sha256
from jlens.mmpilot.convergence import (
    CONTROL_VARIANTS,
    CONVERGED,
    CONVERGENCE_CRITERION,
    CONVERGENCE_PROTOCOL,
    MODALITIES,
    NOT_CONVERGED,
    WRONG_LAYER_CONTROL_NOTE,
    ConvergenceCriterion,
)
from jlens.mmpilot.l32_followup import (
    CONVERGENCE_PHRASE,
    L32_FOLLOWUP_PROTOCOL,
    L32_LAYER,
    SELECTED_SCALE,
)
from jlens.mmpilot.store import payload_checksum

#: Bound into the amendment. A change here means the *presentation* changed, and
#: is deliberately separate from every scientific version string so that fixing
#: a printed table can never invalidate a completed run's resume.
L32_REPORTING_VERSION = "mmpilot.l32_followup_reporting.v2"

REPORTING_SCHEMA = "jlens.mmpilot.l32_followup_reporting_amendment.v2"

#: The original artifacts. Never written by anything in this module.
ORIGINAL_REPORT_NAME = "l32_followup_report.json"

#: What the amendment writes, beside the original.
AMENDED_REPORT_NAME = "l32_followup_report_reporting_v2.json"
AMENDED_MARKDOWN_NAME = "l32_followup_report_reporting_v2.md"

#: Where :class:`~jlens.mmpilot.convergence.ConvergenceStore` put the rows.
READOUT_SUBDIR = ("convergence", "readout_units")

#: Cell fields the table prints, in order. Every one exists in
#: :func:`jlens.mmpilot.convergence.summarize_cell`; that is the point.
CELL_FIELDS: tuple[str, ...] = (
    "n",
    "n_with_clean_reference",
    "n_distinct_images",
    "n_distinct_recordings",
    "n_distinct_predictions",
    "clean_agreement_unique",
    "clean_agreement_argmax",
    "target_accuracy_unique",
    "target_accuracy_argmax",
    "unique_top1_target_rate",
    "median_target_rank",
    "median_target_margin",
    "median_candidate_entropy_nats",
    "median_top_two_margin",
    "tied_at_max_rate",
)

#: Keys the first version of the notebook asked for and that have never existed.
#: Kept so a test can prove the table no longer reads one.
RETIRED_CELL_FIELDS: dict[str, str] = {
    "unique_top1_rate": "unique_top1_target_rate",
    "median_entropy": "median_candidate_entropy_nats",
    "decided_by": "failed_converged_clauses / failed_not_converged_clauses",
}


class ReportingAmendmentRefused(RuntimeError):
    """A precondition of the reporting amendment does not hold."""


class ControlRecordsIncomplete(ReportingAmendmentRefused):
    """An expected control variant produced no record. Missing is not passing."""


class ConvergenceViewMismatch(ReportingAmendmentRefused):
    """Re-deriving the display table disagreed with what the run recorded."""


# ------------------------------------------------------------- the criterion


def format_l32_criterion(
    *,
    layer: int = L32_LAYER,
    criterion: ConvergenceCriterion = CONVERGENCE_CRITERION,
    scale: int = SELECTED_SCALE,
) -> str:
    """The predeclared criterion, stated for **this** audit.

    Same thresholds, same digest, same two-sided rule as
    :data:`jlens.mmpilot.convergence.CRITERION_TEXT` — read off the criterion
    object rather than retyped, so the two can never drift. What differs is only
    what is true of *this* run: one layer, no trajectory clause, and no claim
    that layer 35 is being audited.
    """
    required = ", ".join(criterion.required_modalities)
    return f"""\
PREDECLARED CONVERGENCE CRITERION — native direct readout at physical layer {layer}
Fixed before any result-producing cell ran. Not revisable after seeing results.
Criterion digest {criterion.digest}
Thresholds unchanged from the frozen rule; only the scope statement differs.

SCOPE. This audit evaluates exactly one layer: physical layer {layer}, whose
lens the early-layer extension confirmed at scale {scale} on its own untouched
256-prompt confirmation set. It is a SINGLE-LAYER classification. No other layer
is scored, and **no later-layer trajectory clause is applied** — the completed
three-modality study's rule additionally required a later validated layer to be
measurably more converged, and that clause cannot be evaluated, or claimed,
from one point.

THE MEASUREMENT. For each stored clean final-prompt-token residual h at layer
{layer}, compute the model's OWN output head on it —

    logits = lm_head(final_norm(h))            [modules called, not reimplemented]
    logits = softcap * tanh(logits / softcap)  [if the config declares a softcap]

— restricted to the fixed behavioral answer candidates. No lens, no dictionary,
no J-space code, no intervention and no learned probe takes part in this number.

CONVERGED, only if in EVERY one of {required}, over capability-admissible
concepts:
  1. the target is the SOLE maximum among the candidates AND equals the model's
     own clean final answer, on at least
     {criterion.converged_min_clean_agreement:.0%} of samples;
  2. the same against the ground-truth concept, on at least
     {criterion.converged_min_target_accuracy:.0%};
  3. the median midrank of the target is at most
     {criterion.converged_max_median_rank:.1f}.

NOT_CONVERGED, only if in EVERY one of those modalities, scored with the
GENEROUS argmax rule (ties resolved in the layer's favour):
  4. agreement with the model's clean final answer is at most
     {criterion.not_converged_max_clean_agreement:.0%};
  5. agreement with the ground-truth concept is at most
     {criterion.not_converged_max_target_accuracy:.0%};
  6. the median midrank of the target is at least
     {criterion.not_converged_min_median_rank:.1f}.

AMBIGUOUS is everything between the two bars. It is a real outcome, not a
failure to measure: it says the layer is neither clearly reading out the answer
nor clearly failing to. An AMBIGUOUS layer {layer} supports no
pre-convergence claim, and the thresholds are not revisable to move it.

GUARDS, any of which forces an inconclusive timing result:
  - fewer than {criterion.min_samples_per_cell} samples in any modality cell;
  - fewer than {criterion.min_distinct_predictions} distinct predicted
    candidates — a readout that answers the same word every time has failed,
    and a failed readout is not a fact about the representation;
  - any non-finite readout value;
  - a candidate set that is not single-token AND cannot support a labelled
    first-token-only diagnostic;
  - any control variant reaching the primary readout's result. A control that
    produced NO record is not a control that passed.

INTERPRETATION BOUNDARY. A weak native direct readout means the
final-prompt-token residual has not converged onto the model's answer under this
criterion. It is NOT proof that linguistic information is absent: no claim is
made, and none may be derived, about what a nonlinear decoder or a trained probe
could recover from the same activation. Say "{CONVERGENCE_PHRASE}" — never
"pre-linguistic", never "language-free", and never "before language exists".

{WRONG_LAYER_CONTROL_NOTE}
"""


# ------------------------------------------------------------ the cell table


def convergence_cell_rows(
    summary: Mapping, *, layer: int, modalities: Sequence[str] = MODALITIES
) -> list[dict]:
    """One display row per modality, reading the keys that actually exist.

    Raises:
        ReportingAmendmentRefused: If a non-empty cell is missing a field this
            table prints. A ``None`` in a report is indistinguishable from a
            measurement that came back empty, so an absent key stops the
            amendment rather than printing one.
    """
    per_layer = summary.get("per_layer") or {}
    entry = per_layer.get(str(int(layer)), per_layer.get(int(layer)))
    if entry is None:
        raise ReportingAmendmentRefused(
            f"the summary carries no layer {layer} (it has "
            f"{sorted(per_layer)}); this amendment reports one layer and will "
            "not substitute another"
        )
    per_modality = entry.get("per_modality") or {}

    rows: list[dict] = []
    for modality in modalities:
        cell = per_modality.get(modality) or {"n": 0}
        row = {"modality": modality, "layer": int(layer)}
        if int(cell.get("n", 0)) == 0:
            rows.append({**row, "n": 0, "empty": True})
            continue
        missing = [name for name in CELL_FIELDS if name not in cell]
        if missing:
            raise ReportingAmendmentRefused(
                f"the {modality} cell at layer {layer} is missing {missing}. "
                "Printing None for a field the summary never stored hides a "
                "reporting bug behind what looks like a measurement."
            )
        row.update({name: cell[name] for name in CELL_FIELDS})
        row["empty"] = False
        rows.append(row)
    return rows


def _fmt(value: object, spec: str = ".3f") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int,)):
        return str(value)
    if isinstance(value, float):
        return format(value, spec)
    return str(value)


def format_convergence_cells(rows: Sequence[Mapping], *, layer: int) -> str:
    """The per-modality table, with every field named and none of them None."""
    lines = [f"PER-MODALITY NATIVE READOUT AT LAYER {layer}", ""]
    for row in rows:
        if row.get("empty"):
            lines.append(f"  {row['modality']:13s} no rows in this cell")
            continue
        lines += [
            f"  {row['modality']}",
            f"    n                        {row['n']}  "
            f"(with a clean reference: {row['n_with_clean_reference']})",
            f"    distinct images          {row['n_distinct_images']}  "
            f"recordings {row['n_distinct_recordings']}",
            f"    distinct predictions     {row['n_distinct_predictions']}",
            f"    clean agreement          unique {_fmt(row['clean_agreement_unique'])}"
            f"   argmax {_fmt(row['clean_agreement_argmax'])}",
            f"    ground-truth accuracy    unique {_fmt(row['target_accuracy_unique'])}"
            f"   argmax {_fmt(row['target_accuracy_argmax'])}",
            f"    unique top-1 (target)    {_fmt(row['unique_top1_target_rate'])}",
            f"    median target midrank    {_fmt(row['median_target_rank'], '.2f')}",
            f"    median target margin     {_fmt(row['median_target_margin'])}",
            f"    median entropy (nats)    "
            f"{_fmt(row['median_candidate_entropy_nats'])}",
            f"    median top-two margin    {_fmt(row['median_top_two_margin'])}",
            f"    tied-at-max rate         {_fmt(row['tied_at_max_rate'])}",
        ]
    return "\n".join(lines)


def classification_detail(classification: Mapping) -> dict:
    """Which clauses decided the classification, and the pooled evidence.

    ``classify_layer`` records ``failed_converged_clauses`` and
    ``failed_not_converged_clauses``; it has never had a ``decided_by`` field,
    and asking for one printed ``None``. AMBIGUOUS means *both* lists are
    non-empty — the layer cleared neither bar — and naming both is what makes
    the outcome legible.
    """
    failed_converged = list(classification.get("failed_converged_clauses") or [])
    failed_not_converged = list(
        classification.get("failed_not_converged_clauses") or []
    )
    value = classification.get("classification")
    if value == CONVERGED:
        decided_by = "every CONVERGED clause held in every required modality"
    elif value == NOT_CONVERGED:
        decided_by = "every NOT_CONVERGED clause held in every required modality"
    else:
        decided_by = (
            "neither bar was cleared: the CONVERGED side failed "
            f"{failed_converged or ['(none recorded)']} and the NOT_CONVERGED "
            f"side failed {failed_not_converged or ['(none recorded)']}"
        )
    bootstrap = classification.get("pooled_bootstrap") or {}
    return {
        "layer": classification.get("layer"),
        "classification": value,
        "decided_by": decided_by,
        "failed_converged_clauses": failed_converged,
        "failed_not_converged_clauses": failed_not_converged,
        "undersized_cells": list(classification.get("undersized_cells") or []),
        "pooled_clean_agreement_unique": classification.get(
            "pooled_clean_agreement_unique"
        ),
        "pooled_clean_agreement_argmax": classification.get(
            "pooled_clean_agreement_argmax"
        ),
        "pooled_target_accuracy_argmax": classification.get(
            "pooled_target_accuracy_argmax"
        ),
        "pooled_median_target_rank": classification.get("pooled_median_target_rank"),
        "n_distinct_predictions": classification.get("n_distinct_predictions"),
        "bootstrap_point": bootstrap.get("point"),
        "bootstrap_low": bootstrap.get("low"),
        "bootstrap_high": bootstrap.get("high"),
        "bootstrap_independent_units": bootstrap.get("n_units"),
        "bootstrap_n_rows": bootstrap.get("n"),
        "bootstrap_resamples": bootstrap.get("resamples"),
        "criterion_digest": classification.get("criterion_digest"),
    }


def format_classification(detail: Mapping) -> str:
    lines = [
        f"CLASSIFICATION  L{detail['layer']}  {detail['classification']}",
        "",
        f"  decided by: {detail['decided_by']}",
        "",
        f"  pooled clean agreement   unique "
        f"{_fmt(detail['pooled_clean_agreement_unique'])}   argmax "
        f"{_fmt(detail['pooled_clean_agreement_argmax'])}",
        f"  pooled ground-truth acc  argmax "
        f"{_fmt(detail['pooled_target_accuracy_argmax'])}",
        f"  pooled median midrank    "
        f"{_fmt(detail['pooled_median_target_rank'], '.2f')}",
        f"  distinct predictions     {detail['n_distinct_predictions']}",
        f"  bootstrap (image-level)  point {_fmt(detail['bootstrap_point'])}  "
        f"[{_fmt(detail['bootstrap_low'])}, {_fmt(detail['bootstrap_high'])}]  "
        f"over {detail['bootstrap_independent_units']} independent unit(s), "
        f"{detail['bootstrap_n_rows']} row(s), "
        f"{detail['bootstrap_resamples']} resamples",
        f"  undersized cells         {detail['undersized_cells'] or 'none'}",
        f"  criterion digest         {detail['criterion_digest']}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------- the controls


def assert_controls_complete(
    controls: Mapping,
    *,
    layer: int,
    variants: Sequence[str] = CONTROL_VARIANTS,
) -> dict:
    """Every expected control variant produced a record. Missing is not passing.

    :func:`jlens.mmpilot.convergence.summarize_controls` skips a variant with no
    rows, so an absent control leaves ``all_controls_passed`` True. That boolean
    then means "nothing that ran reproduced the primary result", which is not
    the claim a reader takes from it.

    Raises:
        ControlRecordsIncomplete: If the layer has no entry, or any expected
            variant is absent.
    """
    per_layer = controls.get("per_layer") or {}
    entry = per_layer.get(str(int(layer)), per_layer.get(int(layer)))
    if entry is None:
        raise ControlRecordsIncomplete(
            f"no control record exists for layer {layer} (records exist for "
            f"{sorted(per_layer)}). A layer with no controls has not passed its "
            "controls; it has not been controlled."
        )
    recorded = entry.get("controls") or {}
    missing = [name for name in variants if name not in recorded]
    if missing:
        raise ControlRecordsIncomplete(
            f"layer {layer} is missing control record(s) {missing}; recorded: "
            f"{sorted(recorded)}. summarize_controls skips a variant that "
            "produced no rows, so 'all_controls_passed' would be True with the "
            "control never having run. Missing is not passing."
        )
    return {
        "layer": int(layer),
        "expected_variants": list(variants),
        "recorded_variants": sorted(recorded),
        "complete": True,
        "all_controls_passed": bool(controls.get("all_controls_passed")),
        "failed_controls": list(controls.get("failed_controls") or []),
    }


def control_rows(
    controls: Mapping,
    *,
    layer: int,
    variants: Sequence[str] = CONTROL_VARIANTS,
) -> list[dict]:
    """One flat row per control variant, read from the nested structure.

    The bug this replaces indexed ``per_layer[layer][variant]``; the variants
    live under ``per_layer[layer]["controls"][variant]``.
    """
    assert_controls_complete(controls, layer=layer, variants=variants)
    per_layer = controls["per_layer"]
    entry = per_layer.get(str(int(layer)), per_layer.get(int(layer)))
    recorded = entry["controls"]
    rows = []
    for variant in variants:
        record = recorded[variant]
        rows.append(
            {
                "variant": variant,
                "compared_field": record["compared_field"],
                "primary_value": record["primary_value"],
                "control_value": record["control_value"],
                "chance_rate": record["chance_rate"],
                "margin": record.get("margin"),
                "primary_is_informative": record["primary_is_informative"],
                "passed": record["passed"],
                "reason": record["reason"],
                "expectation": record.get("expectation"),
            }
        )
    return rows


def format_controls(rows: Sequence[Mapping], *, controls: Mapping, layer: int) -> str:
    lines = [f"CONVERGENCE CONTROLS AT LAYER {layer}", ""]
    for row in rows:
        lines += [
            f"  {row['variant']}",
            f"    compared on            {row['compared_field']}",
            f"    primary                {_fmt(row['primary_value'])}",
            f"    control                {_fmt(row['control_value'])}",
            f"    chance rate            {_fmt(row['chance_rate'])}"
            + (
                f"   separation margin {_fmt(row['margin'], '.2f')}"
                if row.get("margin") is not None
                else ""
            ),
            f"    primary_is_informative {row['primary_is_informative']}",
            f"    passed                 {row['passed']}",
            f"    reason                 {row['reason']}",
        ]
    lines += [
        "",
        f"  all {len(rows)} expected control(s) produced a record",
        f"  all_controls_passed  {controls.get('all_controls_passed')}",
        f"  failed_controls      {controls.get('failed_controls') or 'none'}",
        f"  {controls.get('pass_rule', '')}",
    ]
    return "\n".join(lines)


# ------------------------------------------------------------- causal buckets


def _is_off_diagonal(pair: object) -> bool:
    parts = str(pair).split("->")
    return len(parts) == 2 and parts[0] != parts[1]


def causal_cell_breakdown(causal: Mapping) -> dict:
    """Separate the passing cells into the buckets each verdict counts.

    Verdict C is decided on the **audio arm**: the text/image replication is
    computed identically and reported beside it, and a study that concluded
    "three-modality transfer" from a text/image cell would be reporting the
    replication as the finding. Verdict E counts every admissible **off-diagonal**
    cell, audio-related or not.

    So "one audio-related cell passed" and "two off-diagonal cells passed" are
    both true of the same table, and the difference is a text<->image cell. This
    function makes that arithmetic explicit instead of leaving two numbers to
    look like a contradiction.
    """
    cells = list(causal.get("cells") or [])
    passing = [
        cell
        for cell in cells
        if cell.get("evaluated")
        and cell.get("passes")
        and cell.get("counted_toward_verdict")
        and _is_off_diagonal(cell.get("pair"))
    ]

    def _entry(cell: Mapping) -> dict:
        return {
            "concept": cell.get("concept"),
            "pair": cell.get("pair"),
            "mean_signed_target_effect": cell.get("mean_signed_target_effect"),
            "random_control": cell.get("random_control"),
            "unrelated_control": cell.get("unrelated_control"),
            "raw_residual_control": cell.get("raw_residual_control"),
            "mean_abs_unrelated_change": cell.get("mean_abs_unrelated_change"),
            "mean_activation_norm_ratio": cell.get("mean_activation_norm_ratio"),
            "fraction_expected_sign": cell.get("fraction_expected_sign"),
            "n_distinct_images": cell.get("n_distinct_images"),
            "n_positive_images": cell.get("n_positive_images"),
            "n_negative_images": cell.get("n_negative_images"),
            "meets_claim_image_floor": cell.get("meets_claim_image_floor"),
        }

    audio = [_entry(cell) for cell in passing if "spoken_audio" in str(cell["pair"])]
    replication = [
        _entry(cell) for cell in passing if "spoken_audio" not in str(cell["pair"])
    ]
    by_concept: dict[str, set] = {}
    for cell in passing:
        by_concept.setdefault(str(cell["concept"]), set()).add(str(cell["pair"]))
    bidirectional = sorted(
        concept
        for concept, pairs in by_concept.items()
        if any(
            f"{b}->{a}" in pairs
            for pair in pairs
            for a, b in [pair.split("->")]
        )
    )
    all_passing = [_entry(cell) for cell in passing]
    return {
        "schema": "jlens.mmpilot.l32_causal_breakdown.v1",
        "n_off_diagonal_passing": len(all_passing),
        "off_diagonal_passing": all_passing,
        "n_audio_related_passing": len(audio),
        "audio_related_passing": audio,
        "n_text_image_passing": len(replication),
        "text_image_passing": replication,
        "concepts_transferring_both_directions": bidirectional,
        "n_distinct_images_max_over_cells": max(
            (
                min(
                    int(entry["n_positive_images"] or 0),
                    int(entry["n_negative_images"] or 0),
                )
                for entry in all_passing
            ),
            default=0,
        ),
        "reconciliation": (
            f"{len(all_passing)} admissible off-diagonal cell(s) passed. "
            f"{len(audio)} of them are audio-related and are what verdict C is "
            f"decided on; {len(replication)} are text<->image, which is the "
            "internal replication and can never carry the three-modality claim "
            "on its own. Verdict E counts all off-diagonal cells, so the two "
            "counts differ by exactly the text<->image cell(s) and neither "
            "number is wrong."
        ),
    }


def format_causal_breakdown(breakdown: Mapping, *, layer: int) -> str:
    def block(title: str, entries: Sequence[Mapping]) -> list[str]:
        lines = [f"  {title} ({len(entries)})"]
        if not entries:
            lines.append("    none")
        for entry in entries:
            lines += [
                f"    {entry['concept']}  {entry['pair']}",
                f"      effect {_fmt(entry['mean_signed_target_effect'], '+.4f')}   "
                f"random {_fmt(entry['random_control'], '+.4f')}   "
                f"unrelated {_fmt(entry['unrelated_control'], '+.4f')}   "
                f"raw {_fmt(entry['raw_residual_control'], '+.4f')}",
                f"      expected-sign fraction "
                f"{_fmt(entry['fraction_expected_sign'])}   "
                f"unrelated movement {_fmt(entry['mean_abs_unrelated_change'])}   "
                f"norm ratio {_fmt(entry['mean_activation_norm_ratio'])}",
                f"      distinct images: positive "
                f"{entry['n_positive_images']}  negative "
                f"{entry['n_negative_images']}  overall "
                f"{entry['n_distinct_images']}  "
                f"claim floor met {entry['meets_claim_image_floor']}",
            ]
        return lines

    lines = [f"PASSING CAUSAL CELLS AT LAYER {layer}", ""]
    lines += block("all admissible off-diagonal", breakdown["off_diagonal_passing"])
    lines.append("")
    lines += block("audio-related (verdict C's arm)", breakdown["audio_related_passing"])
    lines.append("")
    lines += block("text<->image (internal replication)", breakdown["text_image_passing"])
    lines += [
        "",
        "  concepts transferring in both directions: "
        f"{breakdown['concepts_transferring_both_directions'] or 'none'}",
        "  photographs/recordings behind the strongest cell: "
        f"{breakdown['n_distinct_images_max_over_cells']}",
        "",
        f"  {breakdown['reconciliation']}",
    ]
    return "\n".join(lines)


# --------------------------------------------------- reading a completed run


def read_readout_rows(run_dir: str | os.PathLike[str]) -> dict:
    """Every stored native-readout row, checksum-validated, read-only.

    A torn or edited row is reported, never used — the same rule the store
    itself applies.
    """
    directory = Path(run_dir).joinpath(*READOUT_SUBDIR)
    if not directory.is_dir():
        raise ReportingAmendmentRefused(
            f"{directory} does not exist; this run stored no native-readout "
            "rows, so no convergence view can be re-derived from it"
        )
    rows: list[dict] = []
    invalid: list[str] = []
    for path in sorted(directory.glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            payload = record["payload"]
            valid = record.get("unit_checksum") == payload_checksum(payload)
        except (json.JSONDecodeError, KeyError, TypeError, UnicodeDecodeError):
            valid, payload = False, None
        if not valid:
            invalid.append(str(path))
            continue
        rows.append(payload)
    if invalid:
        raise ReportingAmendmentRefused(
            f"{len(invalid)} stored readout row(s) failed their own checksum: "
            f"{invalid[:5]}. An unverifiable row is not a row, and a table "
            "built from a partial set would silently be a different measurement."
        )
    if not rows:
        raise ReportingAmendmentRefused(f"{directory} holds no valid rows")
    return {
        "rows": rows,
        "n_rows": len(rows),
        "directory": str(directory),
        "rows_digest": payload_checksum(
            sorted(
                (str(row.get("variant")), str(row.get("sample_id")), str(row.get("layer")))
                for row in rows
            )
        ),
    }


def source_unit_digest_from_disk(
    run_dir: str | os.PathLike[str],
    *,
    stages: Sequence[str] = ("capability", "intervention", "metric", "activation"),
) -> dict:
    """A digest over the scientific units, read straight off disk.

    Deliberately does **not** open a :class:`~jlens.mmpilot.store.UnitStore`:
    that needs the run's exact ``RunFingerprint`` reconstructed, and the whole
    point of a CPU reporting pass is that it reads a directory without being
    able to write one.
    """
    root = Path(run_dir)
    per_stage: dict[str, dict] = {}
    for stage in stages:
        directory = root / "units" / stage
        checksums: dict[str, str] = {}
        if directory.is_dir():
            for path in sorted(directory.glob("*.json")):
                try:
                    record = json.loads(path.read_text(encoding="utf-8"))
                    payload = record["payload"]
                    checksum = record.get("unit_checksum")
                except (json.JSONDecodeError, KeyError, TypeError, UnicodeDecodeError):
                    continue
                if checksum == payload_checksum(payload):
                    checksums[path.stem] = str(checksum)
        per_stage[stage] = {
            "n_units": len(checksums),
            "units_digest": payload_checksum(sorted(checksums.items())),
        }
    return {
        "schema": "jlens.mmpilot.l32_source_unit_digest.v1",
        "stages": list(stages),
        "per_stage": per_stage,
        "combined_digest": payload_checksum(per_stage),
    }


def recompute_convergence_view(
    run_dir: str | os.PathLike[str],
    *,
    layer: int,
    recorded_classification: Mapping,
    criterion: ConvergenceCriterion = CONVERGENCE_CRITERION,
) -> dict:
    """Re-derive the display tables from the stored rows, and check them.

    The original report kept the classification and the controls but not the
    per-modality summary, so the table has to come from the rows. It is produced
    by the **same** frozen functions, at the **same** criterion digest, over the
    **same** checksum-validated rows — and the resulting classification is
    compared against what the run recorded.

    Raises:
        ConvergenceViewMismatch: If the re-derived classification differs. That
            would mean this is not a reporting pass.
    """
    from jlens.mmpilot.convergence import (
        classify_layer,
        summarize_controls,
        summarize_rows,
    )

    if criterion.digest != CONVERGENCE_CRITERION.digest:
        raise ReportingAmendmentRefused(
            "the criterion offered to the reporting amendment is not the frozen "
            f"one ({criterion.digest} != {CONVERGENCE_CRITERION.digest}). A "
            "reporting pass never moves a threshold."
        )
    read = read_readout_rows(run_dir)
    rows = read["rows"]
    summary = summarize_rows(rows, criterion=criterion, layers=(int(layer),))
    classification = classify_layer(
        summary["per_layer"][str(int(layer))], criterion=criterion
    )
    controls = summarize_controls(rows, layers=(int(layer),))

    recorded = recorded_classification.get("classification")
    if recorded is not None and classification["classification"] != recorded:
        raise ConvergenceViewMismatch(
            f"re-deriving layer {layer} from its stored rows gives "
            f"{classification['classification']!r}, but the run recorded "
            f"{recorded!r}. A reporting amendment may not change a "
            "classification; something other than the presentation differs."
        )
    return {
        "layer": int(layer),
        "n_rows": read["n_rows"],
        "rows_digest": read["rows_digest"],
        "readout_units_dir": read["directory"],
        "summary": summary,
        "classification": classification,
        "controls": controls,
        "criterion_digest": criterion.digest,
        "recomputed_matches_recorded": True,
        "recomputation_note": (
            "the same summarize_rows / classify_layer / summarize_controls the "
            "run used, at the same criterion digest, over the same "
            "checksum-validated rows. Nothing is re-measured; the model is not "
            "loaded."
        ),
    }


# ------------------------------------------------------------ the amendment


def _read_original(run_dir: str | os.PathLike[str]) -> tuple[Path, dict, str]:
    path = Path(run_dir) / ORIGINAL_REPORT_NAME
    if not path.is_file():
        raise ReportingAmendmentRefused(
            f"{path} does not exist. The amendment describes the run you name; "
            "it never discovers one and never creates one."
        )
    checksum = file_sha256(str(path))
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ReportingAmendmentRefused(f"{path} is not readable JSON: {error}") from error
    if report.get("schema") != "jlens.mmpilot.l32_followup_report.v1":
        raise ReportingAmendmentRefused(
            f"{path} declares schema {report.get('schema')!r}; this amendment "
            "reads jlens.mmpilot.l32_followup_report.v1 and will not guess at "
            "another format"
        )
    return path, report, checksum


def build_reporting_amendment(
    run_dir: str | os.PathLike[str],
    *,
    layer: int = L32_LAYER,
    criterion: ConvergenceCriterion = CONVERGENCE_CRITERION,
    expected_report_checksum: str | None = None,
) -> dict:
    """The v2 reporting view of one completed run. Reads only; writes nothing.

    Args:
        expected_report_checksum: Optional pin on the original report's bytes.
            When given it must match, so an amendment can be tied to exactly the
            artifact it was reviewed against.
    """
    root = Path(run_dir)
    report_path, report, report_checksum = _read_original(root)
    if (
        expected_report_checksum
        and expected_report_checksum != report_checksum
    ):
        raise ReportingAmendmentRefused(
            f"{report_path} is {report_checksum}, not the pinned "
            f"{expected_report_checksum}"
        )

    fingerprint_path = root / "fingerprint.json"
    stored_fingerprint = None
    fingerprint_digest = None
    if fingerprint_path.is_file():
        stored_fingerprint = json.loads(fingerprint_path.read_text(encoding="utf-8"))
        stored_fingerprint.pop("written_utc", None)
        fingerprint_digest = payload_checksum(stored_fingerprint)
        recorded = report.get("run_fingerprint_digest")
        if recorded and recorded != fingerprint_digest:
            raise ReportingAmendmentRefused(
                f"{fingerprint_path} digests to {fingerprint_digest}, but the "
                f"report was written under {recorded}. The report and the run "
                "directory disagree about what was run."
            )

    verdicts = report.get("verdicts") or {}
    recorded_classification = verdicts.get("D_native_output_convergence") or {}
    view = recompute_convergence_view(
        root,
        layer=layer,
        recorded_classification=recorded_classification,
        criterion=criterion,
    )
    cells = convergence_cell_rows(view["summary"], layer=layer)
    detail = classification_detail(view["classification"])
    completeness = assert_controls_complete(view["controls"], layer=layer)
    controls = control_rows(view["controls"], layer=layer)
    breakdown = causal_cell_breakdown(verdicts.get("C_causal_transfer") or {})

    unchanged = {
        "A_lens_integrity": (verdicts.get("A_lens_integrity") or {}).get("verdict"),
        "B_representational_transfer": (
            verdicts.get("B_representational_transfer") or {}
        ).get("verdict"),
        "C_causal_transfer": (verdicts.get("C_causal_transfer") or {}).get("verdict"),
        "D_native_output_convergence": recorded_classification.get("classification"),
        "E_pre_convergence_causal_transfer": (
            verdicts.get("E_pre_convergence_causal_transfer") or {}
        ).get("verdict"),
    }

    payload = {
        "schema": REPORTING_SCHEMA,
        "reporting_version": L32_REPORTING_VERSION,
        "followup_protocol": L32_FOLLOWUP_PROTOCOL,
        "convergence_protocol": CONVERGENCE_PROTOCOL,
        "amends": {
            "run_dir": str(root),
            "original_report": ORIGINAL_REPORT_NAME,
            "original_report_checksum": report_checksum,
            "original_report_immutable": True,
            "run_fingerprint_digest": fingerprint_digest
            or report.get("run_fingerprint_digest"),
            "followup_fingerprint_digest": (
                report.get("followup_fingerprint") or {}
            ).get("fingerprint_digest"),
            "source_unit_digest": source_unit_digest_from_disk(root),
            "readout_rows_digest": view["rows_digest"],
        },
        "layer": int(layer),
        "criterion_digest": criterion.digest,
        "criterion_text": format_l32_criterion(layer=layer, criterion=criterion),
        "convergence_cells": cells,
        "classification_detail": detail,
        "controls_completeness": completeness,
        "controls": controls,
        "causal_breakdown": breakdown,
        "recomputation": {
            key: view[key]
            for key in (
                "n_rows",
                "rows_digest",
                "readout_units_dir",
                "recomputed_matches_recorded",
                "recomputation_note",
            )
        },
        "verdicts_unchanged": unchanged,
        "statement": (
            f"This is a reporting repair. Layer {layer} remains "
            f"{unchanged['D_native_output_convergence']}; L{layer} causal "
            f"transfer remains {unchanged['C_causal_transfer']}; "
            "PRE_CONVERGENCE_CAUSAL_TRANSFER remains "
            f"{unchanged['E_pre_convergence_causal_transfer']}. No threshold, "
            "criterion digest, prompt protocol, concept, sample, intervention, "
            "control, alpha, fingerprint or verdict changed, and no stored "
            "scientific unit was rewritten."
        ),
        "changes_no_scientific_verdict": True,
        "convergence_phrase": CONVERGENCE_PHRASE,
    }
    payload["amendment_checksum"] = payload_checksum(payload)
    return payload


def render_amendment_markdown(amendment: Mapping) -> str:
    """The human-readable half of the amendment."""
    amends = amendment["amends"]
    layer = amendment["layer"]
    unchanged = amendment["verdicts_unchanged"]
    lines = [
        f"# Layer {layer} follow-up — reporting amendment v2",
        "",
        "**This changes no scientific result.** It repairs how a completed run "
        "was displayed: a criterion written for another audit's layers, three "
        "metric keys that do not exist, controls read one level too shallow, "
        "and two cell counts that looked contradictory and were not.",
        "",
        "## What it amends",
        "",
        f"- run directory `{amends['run_dir']}`",
        f"- original report `{amends['original_report']}` "
        f"(`{amends['original_report_checksum']}`) — **unchanged**",
        f"- run fingerprint `{amends['run_fingerprint_digest']}`",
        f"- follow-up fingerprint `{amends['followup_fingerprint_digest']}`",
        f"- source-unit digest `{amends['source_unit_digest']['combined_digest']}`",
        f"- readout-row digest `{amends['readout_rows_digest']}`",
        f"- reporting version `{amendment['reporting_version']}`",
        "",
        "## Verdicts, unchanged",
        "",
        "| verdict | value |",
        "|---|---|",
    ]
    for name, value in unchanged.items():
        lines.append(f"| {name} | `{value}` |")
    lines += [
        "",
        amendment["statement"],
        "",
        "## The criterion, stated for this audit",
        "",
        "```",
        amendment["criterion_text"].rstrip(),
        "```",
        "",
        "## Native readout, per modality",
        "",
        "```",
        format_convergence_cells(amendment["convergence_cells"], layer=layer),
        "```",
        "",
        "```",
        format_classification(amendment["classification_detail"]),
        "```",
        "",
        "## Controls",
        "",
        "```",
        format_controls(
            amendment["controls"],
            controls={
                "all_controls_passed": amendment["controls_completeness"][
                    "all_controls_passed"
                ],
                "failed_controls": amendment["controls_completeness"][
                    "failed_controls"
                ],
                "pass_rule": (
                    "a control fails only by reproducing a primary result that "
                    "is itself above chance"
                ),
            },
            layer=layer,
        ),
        "",
        f"  all {len(amendment['controls_completeness']['expected_variants'])} "
        "expected variants produced a record; a missing one would have refused "
        "this amendment rather than counting as a pass.",
        "```",
        "",
        "## Passing causal cells",
        "",
        "```",
        format_causal_breakdown(amendment["causal_breakdown"], layer=layer),
        "```",
        "",
        "## How this was derived",
        "",
        amendment["recomputation"]["recomputation_note"],
        "",
        f"{amendment['recomputation']['n_rows']} checksum-validated readout rows "
        f"from `{amendment['recomputation']['readout_units_dir']}`. The "
        "re-derived classification was required to equal the recorded one before "
        "anything was written.",
    ]
    return "\n".join(lines) + "\n"


def write_reporting_amendment(
    run_dir: str | os.PathLike[str],
    amendment: Mapping,
    *,
    report_name: str = AMENDED_REPORT_NAME,
    markdown_name: str = AMENDED_MARKDOWN_NAME,
) -> dict:
    """Write the amendment beside the original, atomically. Never overwrites it.

    Raises:
        ReportingAmendmentRefused: If either destination is the original report
            or a stored unit.
    """
    root = Path(run_dir)
    if not root.is_dir():
        raise ReportingAmendmentRefused(f"{root} is not an existing run directory")

    # Validate the destinations BEFORE rendering anything. Rendering first would
    # let a bad destination surface as whatever the renderer happened to raise.
    for path in (root / report_name, root / markdown_name):
        if path.name == ORIGINAL_REPORT_NAME:
            raise ReportingAmendmentRefused(
                f"refusing to write {path}: the original report is the evidence "
                "this amendment is about and is never overwritten"
            )
        if "units" in path.parts or "readout_units" in path.parts:
            raise ReportingAmendmentRefused(
                f"refusing to write {path}: stored scientific units are never "
                "rewritten by a reporting pass"
            )

    destinations = {
        root / report_name: json.dumps(
            amendment, indent=2, ensure_ascii=False, default=str
        ),
        root / markdown_name: render_amendment_markdown(amendment),
    }
    staged: list[tuple[Path, Path]] = []
    try:
        for path, text in destinations.items():
            tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
            tmp.write_text(text, encoding="utf-8")
            staged.append((tmp, path))
        for tmp, path in staged:
            os.replace(tmp, path)
    finally:
        for tmp, _ in staged:
            if tmp.exists():  # pragma: no cover - only on a failed write
                tmp.unlink()
    return {
        "report": str(root / report_name),
        "markdown": str(root / markdown_name),
    }


__all__ = [
    "AMENDED_MARKDOWN_NAME",
    "AMENDED_REPORT_NAME",
    "CELL_FIELDS",
    "L32_REPORTING_VERSION",
    "ORIGINAL_REPORT_NAME",
    "REPORTING_SCHEMA",
    "RETIRED_CELL_FIELDS",
    "ControlRecordsIncomplete",
    "ConvergenceViewMismatch",
    "ReportingAmendmentRefused",
    "assert_controls_complete",
    "build_reporting_amendment",
    "causal_cell_breakdown",
    "classification_detail",
    "control_rows",
    "convergence_cell_rows",
    "format_causal_breakdown",
    "format_classification",
    "format_controls",
    "format_convergence_cells",
    "format_l32_criterion",
    "read_readout_rows",
    "recompute_convergence_view",
    "render_amendment_markdown",
    "source_unit_digest_from_disk",
    "write_reporting_amendment",
]
