# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Amend a completed run's report without recomputing the run.

The expensive part of this study — capability scoring, activation capture,
J-space codes, directions, thousands of interventions — is *measurement*. The
part that was wrong was *interpretation*: cells belonging to a concept that
failed the behavioral capability gate were counted as scientific support. The
measurements are unaffected by that error and remain valid exactly as they are.

So this module re-derives the verdicts from the stored units and writes a new,
separately versioned artifact beside the original. It never loads a model, never
writes into ``units/``, and never overwrites the original report.

**Two fingerprints, deliberately.** Binding the reporting rule into the
*generation* fingerprint would make a completed run un-resumable the moment a
report bug was fixed, which is the opposite of what resume is for. So:

* the **raw-generation fingerprint** (:class:`jlens.mmpilot.store.RunFingerprint`)
  is preserved untouched — the completed capability and intervention units stay
  reusable and a rerun still resumes them;
* the **postprocessing rule** is versioned and checksummed on its own
  (:data:`POSTPROCESSING_VERSION` plus the admissibility rule's checksum);
* the amended artifact **binds to both**, plus a digest over the exact source
  units it read. Re-deriving it against changed units is refused, so an amended
  report can never quietly describe a different measurement than the one it was
  computed from.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

from jlens.mmpilot.admissibility import (
    CLAIM_ADMISSIBILITY_RULE_VERSION,
    admissibility_rule_record,
)
from jlens.mmpilot.store import UnitStore, payload_checksum
from jlens.mmpilot.tri_modal import (
    TriModalThresholds,
    causal_transfer_verdict,
    overall_verdict,
    render_report,
    replication_verdict,
)

#: The postprocessing contract. Distinct from the raw-generation fingerprint and
#: from the verdict protocol; bumping it means "the same units, read under a
#: different reporting rule".
POSTPROCESSING_VERSION = "mmpilot.capability_filtered_postprocessing.v2"

#: Written beside — never over — ``native_audio_transfer_report.md``.
AMENDED_REPORT_NAME = "native_audio_transfer_report_capability_filtered_v2.md"
AMENDED_SUMMARY_NAME = "native_audio_transfer_summary_capability_filtered_v2.json"

#: The unit stages an amended report's conclusions actually depend on. Activation
#: and J-space units feed Verdict B, which this amendment does not recompute.
SOURCE_STAGES: tuple[str, ...] = ("capability", "intervention", "metric")


class AmendedReportBindingError(RuntimeError):
    """The stored units are not the ones an amended report was computed from."""


class AmendedReportInputMissing(RuntimeError):
    """A stored metric the amendment needs is absent from the run directory."""


def source_unit_digest(
    store: UnitStore, *, stages: Sequence[str] = SOURCE_STAGES
) -> dict:
    """A checksum over exactly which units, and which contents, were read.

    Per stage: the unit count and a digest over the sorted ``(key, checksum)``
    pairs. Adding, removing, or editing any unit changes the digest, so an
    amended report bound to it refuses to be reused.
    """
    per_stage: dict[str, dict] = {}
    for stage in stages:
        checksums = store.unit_checksums(stage)
        per_stage[stage] = {
            "n_units": len(checksums),
            "units_digest": payload_checksum(sorted(checksums.items())),
        }
    return {
        "schema": "jlens.mmpilot.source_unit_digest.v1",
        "stages": list(stages),
        "per_stage": per_stage,
        "combined_digest": payload_checksum(per_stage),
    }


def _require_metric(store: UnitStore, key: str) -> dict:
    payload = store.load("metric", key)
    if payload is None:
        raise AmendedReportInputMissing(
            f"the run directory {store.root} has no valid metric unit {key!r}. "
            "The amended report is derived from stored units only; it cannot "
            "recompute one, and it will not guess."
        )
    return payload


def rebuild_verdicts(
    store: UnitStore,
    *,
    primary_layer: int,
    replication_layers: Sequence[int],
    focal_concepts: Sequence[str],
    thresholds: TriModalThresholds,
    invariance: Mapping | None = None,
    blocked_modalities: Sequence[str] = (),
    principal_three_modality: bool = True,
) -> dict:
    """Re-derive verdicts C, D and E from stored units. No model is loaded.

    Verdicts A and B are read back unchanged: the capability result is what the
    admissibility rule is *applied from*, and Verdict B's retrieval is pooled
    over concepts and is not recomputed here (the overall verdict states that
    limitation explicitly).
    """
    capability_verdict = _require_metric(store, "audio_capability_verdict")
    representational = _require_metric(store, "representational_transfer_verdict")

    layer_verdicts: dict[int, dict] = {}
    for layer in (int(primary_layer), *(int(x) for x in replication_layers)):
        image_level = store.load("metric", f"interventions_image_level_L{layer}")
        if image_level is None:
            continue
        layer_verdicts[layer] = causal_transfer_verdict(
            image_level,
            layer=layer,
            focal_concepts=focal_concepts,
            thresholds=thresholds,
            name=f"L{layer}_CAUSAL_TRANSFER",
            capability=capability_verdict,
            principal_three_modality=principal_three_modality,
        )
    primary = layer_verdicts.get(int(primary_layer))
    replication = replication_verdict(
        layer_verdicts,
        primary=primary,
        layers=[int(x) for x in replication_layers],
    )
    overall = overall_verdict(
        capability=capability_verdict,
        representational=representational,
        primary_causal=primary,
        replication=replication,
        invariance=invariance if invariance is not None else store.load(
            "metric", "invariance"
        ),
        blocked_modalities=blocked_modalities,
        thresholds=thresholds,
    )
    return {
        "capability": capability_verdict,
        "representational": representational,
        "per_layer_causal": layer_verdicts,
        "primary_causal": primary,
        "replication": replication,
        "overall": overall,
    }


def build_amended_report(
    store: UnitStore,
    *,
    primary_layer: int,
    replication_layers: Sequence[int],
    focal_concepts: Sequence[str],
    thresholds: TriModalThresholds,
    original_report_path: str | Path | None = None,
    lens_report: Mapping | None = None,
    audio_protocol: Mapping | None = None,
    invariance: Mapping | None = None,
    blocked_modalities: Sequence[str] = (),
    mode: str = "native_audio_transfer",
    reason: str = (
        "The original report counted causal cells belonging to a concept that "
        "failed the behavioral spoken-audio capability gate as scientific "
        "support. The measurements are unchanged and remain valid; only what "
        "they are allowed to be evidence for has been corrected."
    ),
    principal_three_modality: bool = True,
) -> dict:
    """Regenerate the report under the admissibility rule and bind it to its source.

    Returns a dict with ``report`` (Markdown), ``summary`` (the JSON payload)
    and ``binding``. Nothing is written; :func:`write_amended_report` does that.
    """
    verdicts = rebuild_verdicts(
        store,
        primary_layer=primary_layer,
        replication_layers=replication_layers,
        focal_concepts=focal_concepts,
        thresholds=thresholds,
        invariance=invariance,
        blocked_modalities=blocked_modalities,
        principal_three_modality=principal_three_modality,
    )
    _, recorded_threshold = (
        verdicts["capability"].get("per_concept"),
        verdicts["capability"].get("threshold"),
    )
    rule = admissibility_rule_record(
        threshold=(
            float(recorded_threshold)
            if recorded_threshold is not None
            else thresholds.capability_threshold
        ),
        principal_three_modality=principal_three_modality,
    )
    binding = {
        "schema": "jlens.mmpilot.amended_report_binding.v1",
        "postprocessing_version": POSTPROCESSING_VERSION,
        "admissibility_rule_version": CLAIM_ADMISSIBILITY_RULE_VERSION,
        "admissibility_rule_checksum": rule["rule_checksum"],
        # The raw-generation contract, preserved. The units were produced under
        # this and are still resumable under this; only the reporting changed.
        "run_fingerprint_digest": store.fingerprint.digest,
        "raw_generation_fingerprint_unchanged": True,
        "source_units": source_unit_digest(store),
        "original_report": (
            str(original_report_path) if original_report_path else None
        ),
        "amended_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_loaded": False,
        "units_written": False,
    }
    report = render_report(
        run_dir=str(store.root),
        capability=verdicts["capability"],
        representational=verdicts["representational"],
        primary_causal=verdicts["primary_causal"],
        replication=verdicts["replication"],
        overall=verdicts["overall"],
        lens_report=lens_report,
        audio_protocol=audio_protocol,
        resume=store.status_report(),
        mode=mode,
    )
    header = "\n".join(
        [
            "> **AMENDED REPORT — capability-filtered, "
            f"`{POSTPROCESSING_VERSION}`.**",
            ">",
            f"> {reason}",
            ">",
            "> No model was loaded and no unit was recomputed. This report is "
            "derived from the completed run's stored units, which remain valid "
            "and resumable under their original generation fingerprint "
            f"`{store.fingerprint.digest}`.",
            ">",
            f"> Source units: `{binding['source_units']['combined_digest']}`. "
            "Re-deriving this report against different units is refused.",
            ">",
            "> The original report is kept unchanged at "
            f"`{binding['original_report']}`.",
            "",
            "",
        ]
    )
    summary = {
        "schema": "jlens.mmpilot.amended_report_summary.v1",
        "binding": binding,
        "admissibility_rule": rule,
        "amendment_reason": reason,
        "primary_layer": int(primary_layer),
        "replication_layers": [int(x) for x in replication_layers],
        "focal_concepts": list(focal_concepts),
        "thresholds": thresholds.to_dict(),
        "verdicts": {
            "A_audio_capability": verdicts["capability"],
            "B_representational_transfer": verdicts["representational"],
            "C_primary_causal": verdicts["primary_causal"],
            "D_replication": verdicts["replication"],
            "E_overall": verdicts["overall"],
        },
        "resume": store.status_report(),
    }
    return {
        "report": header + report,
        "summary": summary,
        "binding": binding,
        "verdicts": verdicts,
    }


def write_amended_report(
    amended: Mapping,
    *,
    run_dir: str | Path,
    report_name: str = AMENDED_REPORT_NAME,
    summary_name: str = AMENDED_SUMMARY_NAME,
) -> dict[str, Path]:
    """Write the amended artifacts beside the original, never over it.

    Raises:
        FileExistsError: If the target path is the original report's. The
            original is evidence of what was concluded before the correction and
            is not overwritten silently.
    """
    root = Path(run_dir)
    if report_name == "native_audio_transfer_report.md":
        raise FileExistsError(
            "refusing to write the amended report over the original "
            "native_audio_transfer_report.md; the original is kept as the record "
            "of what was concluded before the correction"
        )
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / report_name
    summary_path = root / summary_name
    report_path.write_text(amended["report"], encoding="utf-8")
    summary_path.write_text(
        json.dumps(amended["summary"], indent=2, default=str), encoding="utf-8"
    )
    return {"report": report_path, "summary": summary_path}


def verify_amended_binding(summary: Mapping, store: UnitStore) -> dict:
    """Hold an existing amended report to the units it claims to describe.

    Raises:
        AmendedReportBindingError: When the run fingerprint, the postprocessing
            version, the admissibility rule checksum, or any source-unit digest
            differs from what the store holds now. The message names every
            differing field.
    """
    binding = summary.get("binding") or {}
    observed = source_unit_digest(
        store, stages=binding.get("source_units", {}).get("stages", SOURCE_STAGES)
    )
    problems: list[str] = []
    if binding.get("run_fingerprint_digest") != store.fingerprint.digest:
        problems.append(
            f"run fingerprint: bound={binding.get('run_fingerprint_digest')} "
            f"observed={store.fingerprint.digest}"
        )
    if binding.get("postprocessing_version") != POSTPROCESSING_VERSION:
        problems.append(
            f"postprocessing version: bound={binding.get('postprocessing_version')} "
            f"installed={POSTPROCESSING_VERSION}"
        )
    if binding.get("admissibility_rule_version") != CLAIM_ADMISSIBILITY_RULE_VERSION:
        problems.append(
            "admissibility rule version: "
            f"bound={binding.get('admissibility_rule_version')} "
            f"installed={CLAIM_ADMISSIBILITY_RULE_VERSION}"
        )
    bound_units = binding.get("source_units") or {}
    if bound_units.get("combined_digest") != observed["combined_digest"]:
        for stage, entry in observed["per_stage"].items():
            was = (bound_units.get("per_stage") or {}).get(stage, {})
            if was.get("units_digest") != entry["units_digest"]:
                problems.append(
                    f"{stage} units: bound n={was.get('n_units')} "
                    f"digest={was.get('units_digest')} observed "
                    f"n={entry['n_units']} digest={entry['units_digest']}"
                )
    if problems:
        raise AmendedReportBindingError(
            "the amended report does not describe the units in this run "
            "directory; refusing to reuse it.\n  - " + "\n  - ".join(problems) + "\n"
            "Re-derive the amended report from the current units rather than "
            "trusting a stale one."
        )
    return {
        "bound": True,
        "run_fingerprint_digest": store.fingerprint.digest,
        "postprocessing_version": POSTPROCESSING_VERSION,
        "source_units_digest": observed["combined_digest"],
    }


__all__ = [
    "AMENDED_REPORT_NAME",
    "AMENDED_SUMMARY_NAME",
    "POSTPROCESSING_VERSION",
    "SOURCE_STAGES",
    "AmendedReportBindingError",
    "AmendedReportInputMissing",
    "build_amended_report",
    "rebuild_verdicts",
    "source_unit_digest",
    "verify_amended_binding",
    "write_amended_report",
]
