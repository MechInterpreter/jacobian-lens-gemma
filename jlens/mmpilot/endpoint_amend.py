# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Relabel a completed result's endpoint without recomputing one number.

What was wrong in the completed reports is *what the numbers were called*, not
what they measured. ``two`` really did outrank ``four``; the target concept's
conditional log-probability really did move against its controls. Neither of
those is "the model answered", and neither is comparable to a paper endpoint
that inspects the whole vocabulary.

So this module writes a **versioned amendment beside** the original artifact. It
never opens a completed report for writing, never touches ``units/``, and never
recomputes a scientific quantity: ``scientific_recompute`` is ``0`` in every
amendment and a test asserts it. What an amendment carries is the original
report's path and checksum, the original run fingerprint, the endpoint-audit
digest that decided the classification, a source-unit digest when the units are
reachable, and an explicit before/after table of the exact terminology that
changed.

The corrected labels
====================

``RESTRICTED_CANDIDATE_PREFERENCE_GO``
    A forced-choice preference among the supplied candidates cleared its
    controls. Says nothing about the model's output.

``CONTROLLED_TARGET_LOGPROB_EFFECT``
    A controlled change in the target answer's conditional log-probability.
    Real, causal, and not autonomous output.

``FULL_VOCABULARY_NOT_EVALUATED``
    Carried by every amended result until the corrected experiment has actually
    run. It is not a failure and it is not a pass; it records that the question
    was never asked.

A historical GO is never turned into a new numerical verdict by prose. The
original verdict string is reproduced verbatim in the amendment and the
corrected labels sit *beside* it.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from jlens.mmpilot.full_vocab_study import (
    CONTROLLED_TARGET_LOGPROB_EFFECT,
    FULL_VOCAB_NOT_EVALUATED,
    RESTRICTED_CANDIDATE_PREFERENCE_GO,
)
from jlens.mmpilot.full_vocabulary import ENDPOINT_UNRESTRICTED_NEXT_TOKEN
from jlens.mmpilot.store import payload_checksum

__all__ = [
    "AMENDMENT_DIRNAME",
    "BAND_FOLLOWUP_TERMINOLOGY",
    "CORRECTED_LABELS",
    "ENDPOINT_AMENDMENT_VERSION",
    "THREE_MODALITY_TERMINOLOGY",
    "AmendmentRefused",
    "TerminologyChange",
    "amendment_filename",
    "build_endpoint_amendment",
    "render_amendment_markdown",
    "source_unit_digest_from_disk",
    "verify_amendment_binding",
    "write_endpoint_amendment",
]

ENDPOINT_AMENDMENT_VERSION = "mmpilot.endpoint_semantics_amendment.v1"
AMENDMENT_SCHEMA = "jlens.mmpilot.endpoint_semantics_amendment.v1"

#: Amendments live in the **new audit run directory**, never in the completed
#: run whose report they describe.
AMENDMENT_DIRNAME = "endpoint_amendments"

CORRECTED_LABELS: tuple[str, ...] = (
    RESTRICTED_CANDIDATE_PREFERENCE_GO,
    CONTROLLED_TARGET_LOGPROB_EFFECT,
    FULL_VOCAB_NOT_EVALUATED,
)


class AmendmentRefused(RuntimeError):
    """An amendment cannot be written, or is not bound to what it claims."""


@dataclass(frozen=True)
class TerminologyChange:
    """One exact wording correction, with the reason it was required."""

    field_path: str
    original_wording: str
    corrected_wording: str
    reason: str

    def to_dict(self) -> dict:
        return dict(asdict(self))


#: The L33-L40 follow-up report's corrections. Every row names a field that
#: exists in that report; the numbers behind them are untouched.
BAND_FOLLOWUP_TERMINOLOGY: tuple[TerminologyChange, ...] = (
    TerminologyChange(
        field_path="reasoning_verdict.paper_comparable",
        original_wording="paper_comparable",
        corrected_wording="restricted_candidate_preference",
        reason=(
            "the field reports an argmax over two supplied candidates. "
            "Anthropic's trial definition inspects the complete next-token "
            "distribution and counts success only at global rank 1, so the two "
            "are not comparable and the name asserted that they were"
        ),
    ),
    TerminologyChange(
        field_path="reasoning_verdict.paper_comparable.criterion",
        original_wording="Anthropic's top-1 trial definition at alpha=1",
        corrected_wording=(
            "restricted-candidate preference at alpha=1: the target answer "
            "outranks the other predeclared candidate by teacher-forced "
            "conditional sequence likelihood"
        ),
        reason=(
            "the criterion never consulted any vocabulary row outside the two "
            "supplied answers"
        ),
    ),
    TerminologyChange(
        field_path="reasoning_verdict.primary_endpoint",
        original_wording=(
            "fraction of trials in which the target-appropriate downstream "
            "answer is top-1 of the externally scored candidate set"
        ),
        corrected_wording=(
            "fraction of trials in which the target-appropriate downstream "
            "answer is top-1 of the externally scored candidate set "
            "[restricted-candidate endpoint; the full vocabulary was not "
            "evaluated]"
        ),
        reason=(
            "the wording was already accurate but was read as a global top-1 "
            "rate; the qualifier is now explicit"
        ),
    ),
    TerminologyChange(
        field_path="followup_verdict.verdict",
        original_wording="L33_L40_VALIDATED_BAND_FOLLOWUP_ALPHA2_SENSITIVITY_ONLY",
        corrected_wording=(
            "L33_L40_VALIDATED_BAND_FOLLOWUP_ALPHA2_SENSITIVITY_ONLY "
            f"[{RESTRICTED_CANDIDATE_PREFERENCE_GO} at alpha=2 sensitivity only; "
            f"{FULL_VOCAB_NOT_EVALUATED}]"
        ),
        reason=(
            "the verdict string is immutable and is reproduced verbatim; the "
            "endpoint labels are added beside it, not substituted for it"
        ),
    ),
    TerminologyChange(
        field_path="trial records: target_rank",
        original_wording="target_rank",
        corrected_wording="restricted_candidate_target_rank (out of 2)",
        reason=(
            "`target_rank` was a rank among the two scored candidates, not a "
            "rank in the vocabulary"
        ),
    ),
)

#: The canonical three-modality run's corrections.
THREE_MODALITY_TERMINOLOGY: tuple[TerminologyChange, ...] = (
    TerminologyChange(
        field_path="capability verdict: per_concept.accuracy",
        original_wording="the model can read the concept out of this channel",
        corrected_wording=(
            "in a six-way forced choice whose options are named in the prompt, "
            "the concept's complete-sequence likelihood exceeded the other five"
        ),
        reason=(
            "the gate is multiple-choice evidence. Open recognition was never "
            "measured, and no vocabulary row outside the six was consulted"
        ),
    ),
    TerminologyChange(
        field_path="causal_transfer_verdict.rationale",
        original_wording="moved the target concept in the expected direction",
        corrected_wording=(
            "changed the target concept's conditional log-probability in the "
            f"expected direction [{CONTROLLED_TARGET_LOGPROB_EFFECT}; "
            f"{FULL_VOCAB_NOT_EVALUATED}]"
        ),
        reason=(
            "a controlled likelihood effect is a real causal result and is not "
            "evidence that any output token changed"
        ),
    ),
    TerminologyChange(
        field_path="intervention records: prediction_changed",
        original_wording="prediction_changed",
        corrected_wording="restricted_candidate_preference_changed",
        reason=(
            "the flag records which of the six supplied candidates scored "
            "highest, before and after; it is not a change of output"
        ),
    ),
    TerminologyChange(
        field_path="overall_verdict: THREE_MODALITY_GO",
        original_wording="THREE_MODALITY_GO",
        corrected_wording=(
            f"THREE_MODALITY_GO [{CONTROLLED_TARGET_LOGPROB_EFFECT}, "
            f"candidate-conditioned; {FULL_VOCAB_NOT_EVALUATED}]"
        ),
        reason=(
            "the verdict string is immutable and is reproduced verbatim; the "
            "endpoint labels are added beside it. The existing "
            "candidate-conditioned limitation from the prompt-protocol rule "
            "still applies and is unchanged by this amendment"
        ),
    ),
)


def source_unit_digest_from_disk(
    run_dir: str | os.PathLike[str], *, stages: Sequence[str] = ("intervention",)
) -> dict | None:
    """A digest over exactly which units, and which contents, were on disk.

    Returns ``None`` when the completed run's units are not reachable from this
    machine — a CPU amendment session may legitimately have only the report. The
    amendment records that absence rather than inventing a digest.
    """
    root = Path(run_dir) / "units"
    if not root.exists():
        return None
    per_stage: dict[str, dict] = {}
    for stage in stages:
        stage_dir = root / stage
        if not stage_dir.exists():
            continue
        checksums: dict[str, str] = {}
        for path in sorted(stage_dir.glob("*.json")):
            try:
                stored = json.loads(path.read_text(encoding="utf-8"))
                checksums[path.stem] = str(stored.get("unit_checksum"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                checksums[path.stem] = "unreadable"
        per_stage[stage] = {
            "n_units": len(checksums),
            "units_digest": payload_checksum(sorted(checksums.items())),
        }
    if not per_stage:
        return None
    return {
        "schema": "jlens.mmpilot.source_unit_digest.v1",
        "stages": list(per_stage),
        "per_stage": per_stage,
        "combined_digest": payload_checksum(per_stage),
    }


def amendment_filename(name: str) -> str:
    """The amendment's stem, distinct from any completed artifact's name."""
    return f"{name}.endpoint_amendment_v1"


def build_endpoint_amendment(
    *,
    name: str,
    study: str,
    original_report_path: str | os.PathLike[str],
    original_report_checksum: str,
    original_run_name: str,
    original_run_fingerprint: str,
    original_verdict: str,
    original_endpoint_class: str,
    endpoint_audit_digest: str,
    claim_ledger_digest: str,
    terminology: Sequence[TerminologyChange],
    corrected_labels: Sequence[str],
    source_unit_digest: Mapping | None = None,
    written_utc: str | None = None,
) -> dict:
    """One versioned amendment. Binds everything; recomputes nothing.

    Raises:
        AmendmentRefused: On a missing binding, an undeclared corrected label,
            or an attempt to amend to the unrestricted endpoint class — which
            only the corrected experiment's own run may ever assert.
    """
    missing = [
        field
        for field, value in (
            ("original_report_checksum", original_report_checksum),
            ("original_run_fingerprint", original_run_fingerprint),
            ("endpoint_audit_digest", endpoint_audit_digest),
            ("claim_ledger_digest", claim_ledger_digest),
            ("original_verdict", original_verdict),
        )
        if not str(value or "")
    ]
    if missing:
        raise AmendmentRefused(
            f"the amendment {name!r} is missing its binding(s): {missing}. An "
            "amendment that cannot name the artifact it corrects is not an "
            "amendment"
        )
    if original_endpoint_class == ENDPOINT_UNRESTRICTED_NEXT_TOKEN:
        raise AmendmentRefused(
            f"{name!r} claims the original endpoint was already unrestricted; "
            "there is nothing for this amendment to correct"
        )
    unknown = sorted(set(corrected_labels) - set(CORRECTED_LABELS))
    if unknown:
        raise AmendmentRefused(
            f"{name!r} uses undeclared corrected label(s) {unknown}. The label "
            f"vocabulary is frozen: {list(CORRECTED_LABELS)}"
        )
    if not terminology:
        raise AmendmentRefused(f"{name!r} changes no terminology")

    payload = {
        "schema": AMENDMENT_SCHEMA,
        "amendment_version": ENDPOINT_AMENDMENT_VERSION,
        "name": str(name),
        "study": str(study),
        "written_utc": written_utc
        or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        # --- the binding
        "original_report_path": str(original_report_path),
        "original_report_checksum": str(original_report_checksum),
        "original_run_name": str(original_run_name),
        "original_run_fingerprint": str(original_run_fingerprint),
        "endpoint_audit_digest": str(endpoint_audit_digest),
        "claim_ledger_digest": str(claim_ledger_digest),
        "source_unit_digest": dict(source_unit_digest) if source_unit_digest else None,
        "source_units_reachable": bool(source_unit_digest),
        # --- what changed, exactly
        "original_verdict": str(original_verdict),
        "original_verdict_is_reproduced_verbatim": True,
        "original_endpoint_class": str(original_endpoint_class),
        "corrected_labels": [str(value) for value in corrected_labels],
        "terminology_changes": [row.to_dict() for row in terminology],
        "n_terminology_changes": len(terminology),
        # --- what did not change
        "scientific_numbers_unchanged": True,
        "scientific_recompute": 0,
        "original_report_modified": False,
        "original_units_modified": False,
        "original_run_modified": False,
        "verdict_changed_by_prose": False,
        "full_vocabulary_evaluated": False,
        "full_vocabulary_status": FULL_VOCAB_NOT_EVALUATED,
        "boundary": (
            "This amendment corrects what the completed numbers are called. It "
            "does not re-measure, re-derive, promote, demote or replace any of "
            "them, and it turns no historical GO into a new numerical verdict."
        ),
    }
    return {**payload, "amendment_checksum": payload_checksum(payload)}


def verify_amendment_binding(
    amendment: Mapping,
    *,
    original_report_checksum: str,
    original_run_fingerprint: str,
    endpoint_audit_digest: str,
    source_unit_digest: Mapping | None = None,
) -> dict:
    """Refuse an amendment that describes a different measurement.

    Raises:
        AmendmentRefused: On any binding disagreement, or when the amendment
            does not rehash to its own recorded checksum.
    """
    problems: list[str] = []
    recomputed = payload_checksum(
        {k: v for k, v in dict(amendment).items() if k != "amendment_checksum"}
    )
    if recomputed != str(amendment.get("amendment_checksum")):
        problems.append("the amendment does not match its own checksum")
    for field, expected in (
        ("original_report_checksum", original_report_checksum),
        ("original_run_fingerprint", original_run_fingerprint),
        ("endpoint_audit_digest", endpoint_audit_digest),
    ):
        if str(amendment.get(field)) != str(expected):
            problems.append(
                f"{field} is {amendment.get(field)!r}, expected {expected!r}"
            )
    if source_unit_digest is not None:
        stored = amendment.get("source_unit_digest") or {}
        if str(stored.get("combined_digest")) != str(
            source_unit_digest.get("combined_digest")
        ):
            problems.append(
                "the source-unit digest differs; the units this amendment "
                "describes are not the units on disk"
            )
    if int(amendment.get("scientific_recompute", -1)) != 0:
        problems.append("scientific_recompute is not 0")
    if problems:
        raise AmendmentRefused(
            f"amendment {amendment.get('name')!r} is not bound to what it "
            "claims:\n  - " + "\n  - ".join(problems)
        )
    return {"bound": True, "checked": ["checksum", "fingerprint", "audit", "units"]}


def render_amendment_markdown(amendment: Mapping) -> str:
    """The human-readable half of one amendment."""
    lines = [
        f"# Endpoint amendment: {amendment['name']}",
        "",
        f"`{amendment['amendment_version']}` · checksum "
        f"`{amendment['amendment_checksum']}`",
        "",
        "## What this is",
        "",
        amendment["boundary"],
        "",
        "## Binding",
        "",
        f"- original report: `{amendment['original_report_path']}`",
        f"- original report checksum: `{amendment['original_report_checksum']}`",
        f"- original run: `{amendment['original_run_name']}`",
        f"- original run fingerprint: `{amendment['original_run_fingerprint']}`",
        f"- endpoint audit digest: `{amendment['endpoint_audit_digest']}`",
        f"- claim ledger digest: `{amendment['claim_ledger_digest']}`",
        "- source-unit digest: "
        + (
            f"`{amendment['source_unit_digest']['combined_digest']}`"
            if amendment.get("source_unit_digest")
            else "not reachable from this session (recorded as absent)"
        ),
        "",
        "## The original verdict, verbatim and unchanged",
        "",
        f"```\n{amendment['original_verdict']}\n```",
        "",
        f"Original endpoint class: `{amendment['original_endpoint_class']}`",
        "",
        "## Corrected labels (added beside, not substituted for, the above)",
        "",
    ]
    for label in amendment["corrected_labels"]:
        lines.append(f"- `{label}`")
    lines += [
        "",
        "## Terminology changed",
        "",
        "| field | was | is | why |",
        "| --- | --- | --- | --- |",
    ]
    for row in amendment["terminology_changes"]:
        lines.append(
            "| `{path}` | {was} | {now} | {why} |".format(
                path=row["field_path"],
                was=row["original_wording"].replace("\n", " "),
                now=row["corrected_wording"].replace("\n", " "),
                why=row["reason"].replace("\n", " "),
            )
        )
    lines += [
        "",
        "## What did not change",
        "",
        f"- `scientific_recompute`: **{amendment['scientific_recompute']}**",
        f"- scientific numbers unchanged: **{amendment['scientific_numbers_unchanged']}**",
        f"- original report modified: **{amendment['original_report_modified']}**",
        f"- original units modified: **{amendment['original_units_modified']}**",
        f"- full vocabulary evaluated: **{amendment['full_vocabulary_evaluated']}** "
        f"(`{amendment['full_vocabulary_status']}`)",
        "",
    ]
    return "\n".join(lines).rstrip() + "\n"


def write_endpoint_amendment(
    audit_run_dir: str | os.PathLike[str], amendment: Mapping
) -> dict:
    """Write one amendment into the **new** audit run directory, atomically.

    Both files are staged and then placed, so a disconnect cannot leave a JSON
    amendment without its Markdown. An existing pair with the same binding is
    reused; an existing pair with a *different* binding is refused rather than
    overwritten.

    Raises:
        AmendmentRefused: On a torn pair, or on a binding disagreement with an
            amendment already on disk.
    """
    root = Path(audit_run_dir) / AMENDMENT_DIRNAME
    root.mkdir(parents=True, exist_ok=True)
    stem = amendment_filename(str(amendment["name"]))
    json_path = root / f"{stem}.json"
    md_path = root / f"{stem}.md"

    if json_path.exists() != md_path.exists():
        raise AmendmentRefused(
            f"a torn amendment pair at {root}: exactly one of {json_path.name} "
            f"and {md_path.name} exists. Neither is overwritten"
        )
    if json_path.exists():
        existing = json.loads(json_path.read_text(encoding="utf-8"))
        if str(existing.get("amendment_checksum")) == str(
            amendment["amendment_checksum"]
        ):
            return {
                "status": "reused",
                "json_path": str(json_path),
                "markdown_path": str(md_path),
            }
        raise AmendmentRefused(
            f"{json_path} already holds an amendment with a different checksum "
            f"({existing.get('amendment_checksum')!r} vs "
            f"{amendment['amendment_checksum']!r}). It is not overwritten"
        )

    json_tmp = json_path.with_suffix(".json.tmp")
    md_tmp = md_path.with_suffix(".md.tmp")
    # Explicit LF and a trailing newline, so a file written here is byte-identical
    # to the reviewable copy `scripts/write_endpoint_audit.py` commits.
    json_tmp.write_text(
        json.dumps(dict(amendment), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    md_tmp.write_text(
        render_amendment_markdown(amendment), encoding="utf-8", newline="\n"
    )
    json_tmp.replace(json_path)
    md_tmp.replace(md_path)
    return {
        "status": "written",
        "json_path": str(json_path),
        "markdown_path": str(md_path),
    }
