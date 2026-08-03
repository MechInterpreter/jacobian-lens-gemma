# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""The CPU-only image-independence audit of a completed pilot run.

This reads a finished run directory and answers one question: how much of its
verdict survives once a photograph, rather than a synchronized caption/audio
group, is the independent unit.

It never loads Gemma, never loads a processor, never touches media, and never
recomputes an activation or an intervention. Everything it needs is already on
disk as per-unit JSON — which is why it runs on a free CPU runtime in seconds.

Four rules govern it.

**Nothing original is overwritten.** ``report.md``, ``summary.json``, the unit
files, the manifests and the codes are read-only inputs. Every output is a new
versioned artifact, and the originals' checksums are recorded before and after
so the claim is verifiable rather than asserted. See :data:`PROTECTED_NAMES`.

**Identity is resolved, not assumed.** If image identity is ambiguous the audit
stops with ``AUDIT_BLOCKED``. A dependence audit that guessed at identity would
be answering a different question than the one it claims to.

**Incompatible state is refused.** The audit fingerprint binds the original run
fingerprint, the subset, the expanded manifest, the lens, the layer, and the
version of every rule applied. A rerun under a different fingerprint is refused,
not merged.

**The original verdict is not privileged.** The amended verdict is computed from
the corrected numbers by the same rubric. ``GO`` is one of four outcomes, and
the audit has no preference among them.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from jlens.mmpilot.causal import summarize_interventions
from jlens.mmpilot.independence import (
    CAUSAL_AGGREGATION_VERSION,
    IMAGE_IDENTITY_RULE_VERSION,
    ImageIdentityError,
    audit_image_independence,
    divergence_summary,
    recompute_representational,
    resolve_image_identity,
    summarize_interventions_by_image,
)
from jlens.mmpilot.jspace import EXCLUSION_RULE_VERSION, NoEligibleTargetError
from jlens.mmpilot.report import (
    FAIL,
    NOT_EVALUATED,
    PASS,
    criterion,
    decide,
    evaluate_criteria,
)
from jlens.mmpilot.store import (
    IncompatibleStateError,
    RunFingerprint,
    UnitStore,
    canonical_json,
    payload_checksum,
)

#: The four possible amended verdicts. The original GO is never forced.
GO_CONFIRMED = "GO_CONFIRMED_AFTER_IMAGE_DEDUP"
WEAK_GO = "WEAK_GO_AFTER_IMAGE_DEDUP"
NO_GO = "NO_GO_AFTER_IMAGE_DEDUP"
AUDIT_BLOCKED = "AUDIT_BLOCKED"

#: How the amended verdict is decided. Part of the fingerprint.
VERDICT_CONFIG_VERSION = "image_dedup_verdict.v1"

#: Files the audit must never write to. Checked before every write, and their
#: checksums are recorded before and after the run.
PROTECTED_NAMES = (
    "report.md",
    "summary.json",
    "fingerprint.json",
    "derived_manifest.json",
    "expanded_manifest.json",
    "synchronized_evidence_audit.json",
    "synchronized_evidence_manifest.json",
    "concept_ranking.json",
    "split_provenance.json",
)

#: Where each new artifact goes, relative to the run directory.
ARTIFACTS = {
    "audit": "audits/image_independence_audit.json",
    "representational": "metrics/representational_image_disjoint_v1.json",
    "interventions": "metrics/interventions_image_level_v1.json",
    "summary": "summary_image_disjoint_v1.json",
    "report": "report_image_disjoint_v1.md",
}

_TEXT_IMAGE_PAIRS = ("text->image", "image->text")


class AuditInputError(RuntimeError):
    """The run directory does not hold what the audit needs to read."""


def file_checksum(path: str | os.PathLike[str]) -> str:
    """``sha256:`` of a file's bytes, streamed."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


# ------------------------------------------------------------- configuration


@dataclass(frozen=True)
class VerdictConfig:
    """The rubric applied to the corrected numbers.

    Defaults restate the pilot's own thresholds, except
    ``required_behavioral_concepts``: a GO after dedup requires *all* the run's
    concepts to have passed behaviorally, not the pilot's minimum of two.
    """

    required_behavioral_concepts: int = 4
    min_fraction_expected_sign: float = 0.75
    control_separation_factor: float = 1.5
    norm_ratio_bounds: tuple[float, float] = (0.5, 2.0)
    min_distinct_images: int = 2
    n_permutations: int = 50
    seed: int = 20260731
    version: str = VERDICT_CONFIG_VERSION

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["norm_ratio_bounds"] = list(self.norm_ratio_bounds)
        return payload

    def thresholds(self) -> dict:
        return {
            "min_concepts_text_image": self.required_behavioral_concepts,
            "min_fraction_expected_sign": self.min_fraction_expected_sign,
            "control_separation_factor": self.control_separation_factor,
            "norm_ratio_bounds": list(self.norm_ratio_bounds),
        }


@dataclass(frozen=True)
class AuditFingerprint:
    """What the audit's artifacts were produced from.

    Any change invalidates every stored audit artifact. The rule versions are
    in here on purpose: two runs of this audit under different exclusion or
    aggregation rules are two different measurements, and mixing them would
    produce a verdict nobody computed.
    """

    original_run_fingerprint_digest: str
    original_subset_checksum: str
    expanded_manifest_checksum: str
    lens_checksum: str
    selected_layer: int
    image_identity_rule_version: str = IMAGE_IDENTITY_RULE_VERSION
    representational_exclusion_rule_version: str = EXCLUSION_RULE_VERSION
    causal_aggregation_version: str = CAUSAL_AGGREGATION_VERSION
    verdict_config: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def digest(self) -> str:
        return payload_checksum(self.to_dict())

    def differences(self, other: Mapping) -> list[str]:
        theirs, mine = dict(other), self.to_dict()
        out = []
        for key in sorted(set(mine) | set(theirs)):
            a, b = mine.get(key, "<absent>"), theirs.get(key, "<absent>")
            if canonical_json(a) != canonical_json(b):
                out.append(f"{key}: stored={b!r} requested={a!r}")
        return out


# ------------------------------------------------------------ the workspace


class AuditWorkspace:
    """Versioned, atomically written audit artifacts under a run directory.

    Mirrors :class:`~jlens.mmpilot.store.UnitStore`'s two rules — a checksum on
    every payload and a fingerprint gate on the directory — but writes into
    ``audits/`` and ``metrics/`` beside the run rather than into ``units/``.
    """

    def __init__(
        self, run_dir: str | os.PathLike[str], fingerprint: AuditFingerprint
    ) -> None:
        self.run_dir = Path(run_dir)
        self.fingerprint = fingerprint
        self.status: str | None = None
        self.reused: list[str] = []
        self.computed: list[str] = []
        self.invalid: list[str] = []

    @property
    def fingerprint_path(self) -> Path:
        return self.run_dir / "audits" / "audit_fingerprint.json"

    def open(self) -> str:
        """``"starting"`` or ``"resuming"``; refuses an incompatible directory."""
        self.fingerprint_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.fingerprint_path.is_file():
            _atomic_write_text(
                self.fingerprint_path,
                json.dumps(self.fingerprint.to_dict(), indent=2, default=str),
            )
            self.status = "starting"
            return self.status
        stored = json.loads(self.fingerprint_path.read_text(encoding="utf-8"))
        if payload_checksum(stored) != self.fingerprint.digest:
            diffs = "\n  ".join(self.fingerprint.differences(stored))
            raise IncompatibleStateError(
                f"{self.fingerprint_path.parent} holds audit artifacts from a "
                f"different configuration; refusing to reuse or mix them.\n  {diffs}\n"
                "Point the audit at a new directory (or move this one aside)."
            )
        self.status = "resuming"
        return self.status

    def path(self, name: str) -> Path:
        relative = ARTIFACTS[name]
        if Path(relative).name in PROTECTED_NAMES:  # pragma: no cover - guard
            raise AuditInputError(f"refusing to write protected artifact {relative!r}")
        return self.run_dir / relative

    def load(self, name: str) -> dict | None:
        """A stored artifact's payload, or ``None`` if absent, torn, or stale."""
        path = self.path(name)
        if not path.is_file():
            return None
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            payload = record["payload"]
            valid = record.get("checksum") == payload_checksum(payload) and record.get(
                "fingerprint_digest"
            ) == self.fingerprint.digest
        except (json.JSONDecodeError, KeyError, TypeError, UnicodeDecodeError):
            valid, payload = False, None
        if not valid:
            self.invalid.append(str(path))
            return None
        self.reused.append(name)
        return payload

    def save(self, name: str, payload: dict) -> Path:
        path = self.path(name)
        _atomic_write_text(
            path,
            json.dumps(
                {
                    "schema": "jlens.mmpilot.audit_artifact.v1",
                    "artifact": name,
                    "written_utc": datetime.now(timezone.utc).isoformat(),
                    "fingerprint_digest": self.fingerprint.digest,
                    "checksum": payload_checksum(payload),
                    "payload": payload,
                },
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
        )
        self.computed.append(name)
        return path

    def save_markdown(self, name: str, text: str) -> Path:
        path = self.path(name)
        _atomic_write_text(path, text)
        _atomic_write_text(
            path.with_suffix(path.suffix + ".sha256"),
            f"{file_checksum(path)}  {path.name}\n",
        )
        self.computed.append(name)
        return path

    def artifact_checksums(self) -> dict:
        return {
            ARTIFACTS[name]: file_checksum(self.path(name))
            for name in ARTIFACTS
            if self.path(name).is_file()
        }

    def status_report(self) -> dict:
        return {
            "run_dir": str(self.run_dir),
            "status": self.status or "unopened",
            "fingerprint_digest": self.fingerprint.digest,
            "reused": sorted(set(self.reused)),
            "computed": sorted(set(self.computed)),
            "invalid_artifacts": list(self.invalid),
        }


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


# ------------------------------------------------------------- reading a run


@dataclass
class LoadedRun:
    """Everything the audit reads, and where each piece came from."""

    run_dir: Path
    run_fingerprint: RunFingerprint
    units: dict
    summary: dict
    subset: dict | None
    subset_source: str
    subset_checksum: str
    expanded_manifest_checksum: str
    selected_layer: int
    original_checksums: dict


def load_run(
    run_dir: str | os.PathLike[str],
    *,
    subset_path: str | os.PathLike[str] | None = None,
    layer: int | None = None,
) -> LoadedRun:
    """Read a completed run's units, summary and subset. Nothing is written."""
    root = Path(run_dir)
    if not root.is_dir():
        raise AuditInputError(f"{root} is not a directory")
    fingerprint_path = root / "fingerprint.json"
    if not fingerprint_path.is_file():
        raise AuditInputError(
            f"{fingerprint_path} is missing, so the saved units cannot be "
            "verified against what produced them."
        )
    stored = json.loads(fingerprint_path.read_text(encoding="utf-8"))
    stored.pop("written_utc", None)
    run_fingerprint = RunFingerprint(
        mode=stored["mode"],
        model_repo_id=stored["model_repo_id"],
        model_revision=stored["model_revision"],
        processor_revision=stored["processor_revision"],
        layers=tuple(stored["layers"]),
        lens_checksum=stored["lens_checksum"],
        manifest_checksum=stored["manifest_checksum"],
        split_id=stored["split_id"],
        intervention_config=stored.get("intervention_config", {}),
        extra=stored.get("extra", {}),
    )
    store = UnitStore(root, run_fingerprint)
    units = {
        stage: store.load_all(stage)
        for stage in ("capability", "activation", "jspace", "intervention", "metric")
    }
    if not units["intervention"]:
        raise AuditInputError(
            f"{root} holds no checksum-valid intervention units; there is "
            "nothing to re-aggregate."
        )
    if not units["activation"] or not units["jspace"]:
        raise AuditInputError(
            f"{root} holds no checksum-valid activation or J-space units; the "
            "representational tests cannot be recomputed without recomputing "
            "them, which this audit will not do."
        )

    summary_path = root / "summary.json"
    if not summary_path.is_file():
        raise AuditInputError(f"{summary_path} is missing; the original verdict "
                              "and its capability evidence cannot be read.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    subset, subset_source, subset_checksum = _load_subset(root, subset_path)
    expanded = root / "expanded_manifest.json"
    resolved_layer = _resolve_layer(summary, units, layer)
    return LoadedRun(
        run_dir=root,
        run_fingerprint=run_fingerprint,
        units=units,
        summary=summary,
        subset=subset,
        subset_source=subset_source,
        subset_checksum=subset_checksum,
        expanded_manifest_checksum=(
            file_checksum(expanded) if expanded.is_file() else "absent"
        ),
        selected_layer=resolved_layer,
        original_checksums=_protected_checksums(root),
    )


def _protected_checksums(root: Path) -> dict:
    return {
        name: file_checksum(root / name)
        for name in PROTECTED_NAMES
        if (root / name).is_file()
    }


def _load_subset(
    root: Path, subset_path: str | os.PathLike[str] | None
) -> tuple[dict | None, str, str]:
    """The pilot subset, if one was saved beside the run.

    The pilot writes its subset into the derived cache, not the run directory,
    so this is genuinely optional: the saved units already carry ``group_id``,
    ``image_id``, ``concept``, ``split`` and ``modality``, which is everything
    the audit needs. When a subset *is* available it is used as an additional
    identity source and cross-check.
    """
    candidates = [Path(subset_path)] if subset_path else [
        root / "pilot_subset.json",
        root / "subset.json",
    ]
    for candidate in candidates:
        if candidate.is_file():
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            if "splits" in payload:
                return payload, str(candidate), file_checksum(candidate)
            raise AuditInputError(
                f"{candidate} does not look like a pilot subset (no 'splits' key)"
            )
    if subset_path:
        raise AuditInputError(f"{subset_path} does not exist")
    return None, "reconstructed_from_saved_units", "absent"


def _resolve_layer(summary: Mapping, units: Mapping, layer: int | None) -> int:
    if layer is not None:
        return int(layer)
    reported = (summary.get("representational") or {}).get("layer")
    if reported is not None:
        return int(reported)
    causal_layers = (summary.get("config") or {}).get("causal_layers") or []
    if causal_layers:
        return int(causal_layers[-1])
    layers = sorted({int(r["layer"]) for r in units["jspace"].values()})
    if not layers:  # pragma: no cover - guarded by load_run
        raise AuditInputError("no layer could be resolved from the saved units")
    return layers[-1]


# ----------------------------------------------------------------- verdict


def _best_row(rows: Sequence[Mapping], *, control_kind: str) -> dict | None:
    candidates = [
        row
        for row in rows
        if row["control_kind"] == control_kind
        and row["off_diagonal"]
        and row["alpha"] > 0
        and row["pair"] in _TEXT_IMAGE_PAIRS
    ]
    return (
        max(candidates, key=lambda row: row["mean_signed_target_effect"])
        if candidates
        else None
    )


def _matched(rows: Sequence[Mapping], best: Mapping, control_kind: str) -> dict | None:
    for row in rows:
        if (
            row["concept"] == best["concept"]
            and row["source_modality"] == best["source_modality"]
            and row["target_modality"] == best["target_modality"]
            and row["layer"] == best["layer"]
            and row["alpha"] == best["alpha"]
            and row["control_kind"] == control_kind
        ):
            return row
    return None


def replication_report(rows: Sequence[Mapping], config: VerdictConfig) -> dict:
    """Whether the corrected effect holds up per concept and per direction."""
    cells = []
    for row in rows:
        if (
            row["control_kind"] != "source_concept"
            or not row["off_diagonal"]
            or row["alpha"] <= 0
            or row["pair"] not in _TEXT_IMAGE_PAIRS
        ):
            continue
        random_row = _matched(rows, row, "random_norm_matched")
        unrelated_row = _matched(rows, row, "unrelated_concept")
        controls = [
            r["mean_signed_target_effect"]
            for r in (random_row, unrelated_row)
            if r is not None
        ]
        strongest = max(controls) if controls else 0.0
        cells.append(
            {
                "concept": row["concept"],
                "pair": row["pair"],
                "alpha": row["alpha"],
                "n_distinct_images": row["n_distinct_images"],
                "n_groups": row["n_groups"],
                "mean_signed_target_effect": row["mean_signed_target_effect"],
                "fraction_expected_sign": row["fraction_expected_sign"],
                "random_control": (random_row or {}).get("mean_signed_target_effect"),
                "unrelated_control": (unrelated_row or {}).get(
                    "mean_signed_target_effect"
                ),
                "passes": bool(
                    row["mean_signed_target_effect"] > 0
                    and row["fraction_expected_sign"] >= config.min_fraction_expected_sign
                    and len(controls) == 2
                    and row["mean_signed_target_effect"]
                    >= config.control_separation_factor * max(strongest, 0.0)
                    and row["n_distinct_images"] >= config.min_distinct_images
                ),
            }
        )
    concepts = sorted({cell["concept"] for cell in cells})
    return {
        "cells": cells,
        "by_concept": {
            concept: any(
                cell["passes"] for cell in cells if cell["concept"] == concept
            )
            for concept in concepts
        },
        "by_direction": {
            pair: any(cell["passes"] for cell in cells if cell["pair"] == pair)
            for pair in _TEXT_IMAGE_PAIRS
        },
        "replicates_across_concepts": sum(
            1
            for concept in concepts
            if any(cell["passes"] for cell in cells if cell["concept"] == concept)
        )
        >= 2,
        "replicates_across_directions": all(
            any(cell["passes"] for cell in cells if cell["pair"] == pair)
            for pair in _TEXT_IMAGE_PAIRS
        ),
    }


#: Every criterion a confirmed GO requires. All must be PASS.
GO_REQUIRES = (
    "behavioral_capability",
    "representational_structure",
    "causal_transfer_sign",
    "control_specificity",
    "activation_norm_sanity",
    "effect_specificity_not_global",
    "no_train_test_image_leakage",
    "evidence_not_single_image",
    "source_exceeds_random_and_external_unrelated",
    "image_identity_resolved",
)


def amended_verdict(
    *,
    capability: Mapping,
    lens_validation: Mapping | None,
    code_stats: Mapping,
    reconstruction_control: Mapping | None,
    representational: Mapping,
    interventions: Mapping,
    audit: Mapping,
    blocked_modalities: Sequence[str] = (),
    config: VerdictConfig | None = None,
) -> dict:
    """Apply the rubric to the corrected, image-disjoint, image-level numbers.

    The base criteria come from the pilot's own
    :func:`~jlens.mmpilot.report.evaluate_criteria`, fed the corrected
    representational report and the image-level intervention rows, so the
    correction is the only thing that changed. Four dedup-specific criteria are
    added on top.
    """
    config = config or VerdictConfig()
    criteria = dict(
        evaluate_criteria(
            capability=capability,
            lens_validation=lens_validation,
            code_stats=code_stats,
            representational=representational,
            interventions=interventions,
            reconstruction_control=reconstruction_control,
            blocked_modalities=blocked_modalities,
            thresholds=config.thresholds(),
        )
    )

    criteria["image_identity_resolved"] = criterion(
        PASS,
        {
            "rule_version": audit.get("image_identity_rule_version"),
            "n_groups": audit.get("n_groups"),
            "n_distinct_images": audit.get("n_distinct_images"),
        },
    )
    leakage = list(audit.get("hard_failures", []))
    criteria["no_train_test_image_leakage"] = criterion(
        PASS if not leakage else FAIL,
        {
            "train_test_image_overlap": audit.get("train_test_image_overlap", []),
            "sibling_groups_crossing_splits": audit.get(
                "sibling_groups_crossing_splits", []
            ),
            "hard_failures": leakage,
        },
    )

    rows = list(interventions.get("rows", []))
    best = _best_row(rows, control_kind="source_concept")
    causal_skip = (
        "no intervention rows survived image-level aggregation" if not rows else None
    )
    criteria["evidence_not_single_image"] = criterion(
        NOT_EVALUATED
        if causal_skip
        else (
            PASS
            if best and best["n_distinct_images"] >= config.min_distinct_images
            else FAIL
        ),
        {
            "best_row": best,
            "n_distinct_images": (best or {}).get("n_distinct_images"),
            "required": config.min_distinct_images,
            "reading": (
                "an effect measured on one photograph is one observation, "
                "however many captions that photograph was given"
            ),
        },
        skipped_because=causal_skip,
    )

    random_row = _matched(rows, best, "random_norm_matched") if best else None
    unrelated_row = _matched(rows, best, "unrelated_concept") if best else None
    exceeds = bool(
        best
        and random_row is not None
        and unrelated_row is not None
        and best["mean_signed_target_effect"] > random_row["mean_signed_target_effect"]
        and best["mean_signed_target_effect"]
        > unrelated_row["mean_signed_target_effect"]
        and best["mean_signed_target_effect"] > 0
    )
    criteria["source_exceeds_random_and_external_unrelated"] = criterion(
        NOT_EVALUATED if causal_skip else (PASS if exceeds else FAIL),
        {
            "source_effect": (best or {}).get("mean_signed_target_effect"),
            "random_control": (random_row or {}).get("mean_signed_target_effect"),
            "external_unrelated_control": (unrelated_row or {}).get(
                "mean_signed_target_effect"
            ),
            "aggregation": interventions.get("aggregation_version"),
        },
        skipped_because=causal_skip,
    )

    base = decide(criteria)
    replication = replication_report(rows, config)
    statuses = {name: entry["status"] for name, entry in criteria.items()}
    confirmed = all(statuses.get(name) == PASS for name in GO_REQUIRES)
    representation_holds = (
        statuses.get("behavioral_capability") == PASS
        and statuses.get("representational_structure") == PASS
        and statuses.get("no_train_test_image_leakage") == PASS
    )
    if confirmed:
        verdict = GO_CONFIRMED
        rationale = (
            "Every criterion the original GO rested on survives image-level "
            "correction. Retrieval and separation exclude every target sharing "
            "the query's photograph, and the causal effect is computed with the "
            "image as the independent unit."
        )
    elif representation_holds:
        verdict = WEAK_GO
        failed = sorted(
            name for name in GO_REQUIRES if statuses.get(name) not in (PASS,)
        )
        rationale = (
            "Image-disjoint cross-modal structure survives, but the causal "
            "evidence does not clear the rubric once photographs rather than "
            f"caption groups are the unit. Not passing: {failed}."
        )
    else:
        verdict = NO_GO
        failed = sorted(name for name in GO_REQUIRES if statuses.get(name) == FAIL)
        rationale = (
            "The corrected evidence does not support the original "
            f"recommendation. Failing: {failed}."
        )
    return {
        "schema": "jlens.mmpilot.image_disjoint_verdict.v1",
        "verdict": verdict,
        "rationale": rationale,
        "original_rubric_recommendation": base["recommendation"],
        "original_rubric_rationale": base["rationale"],
        "criteria": criteria,
        "criteria_status": statuses,
        "go_requires": list(GO_REQUIRES),
        "replication": replication,
        "verdict_config": config.to_dict(),
        "late_layer_limitation": (
            "The validated layer is late in the decoder. A final-prompt-token "
            "edit there cannot establish that the effect precedes "
            "answer-language convergence, so no claim of transfer before "
            "convergence is made here — corrected or not."
        ),
    }


# ------------------------------------------------------------------ markdown


def _percent(value) -> str:
    return "unknown" if value is None else f"{100.0 * float(value):.1f}%"


def render_report(
    *,
    run_dir: str,
    verdict: Mapping,
    audit: Mapping,
    representational: Mapping,
    interventions: Mapping,
    divergence: Mapping,
    original: Mapping,
    fingerprint: AuditFingerprint,
    resume: Mapping,
    preservation: Mapping,
) -> str:
    """The amended report. The original ``report.md`` is left untouched."""
    lines = [
        f"# Image-independence audit — {verdict['verdict']}",
        "",
        f"- audited run: `{run_dir}`",
        f"- original recommendation: **{original.get('recommendation')}** (unchanged on disk)",
        f"- amended verdict: **{verdict['verdict']}**",
        f"- audit fingerprint: `{fingerprint.digest}`",
        f"- audit state: {resume.get('status')} "
        f"(reused: {resume.get('reused') or 'none'}; computed: {resume.get('computed') or 'none'})",
        "- **No model was loaded.** No activation, code, direction or "
        "intervention was recomputed. This audit reads saved artifacts only.",
        "",
        "## What the dependence actually is",
        "",
        f"- synchronized groups: **{audit.get('n_groups')}**",
        f"- distinct images: **{audit.get('n_distinct_images')}**",
        f"- images entering the run as more than one group: "
        f"**{audit.get('n_images_with_multiple_groups')}**",
        f"- concepts affected: {audit.get('concepts_affected') or 'none'}",
        f"- modality records on repeated images: {audit.get('n_modality_records_affected')}",
        "",
        "SpokenCOCO gives one photograph several written captions and a spoken "
        "reading of each. Where the subset kept more than one, those groups are "
        "one image seen twice — not two observations.",
        "",
        "| split | groups | distinct images | images with >1 group |",
        "| --- | --- | --- | --- |",
    ]
    for split, block in sorted((audit.get("by_split") or {}).items()):
        lines.append(
            f"| {split} | {block['n_groups']} | {block['n_distinct_images']} | "
            f"{block['n_images_with_multiple_groups']} |"
        )
    lines += [
        "",
        f"- train/test image overlap: {audit.get('train_test_image_overlap') or 'none'}",
        f"- sibling groups crossing splits: "
        f"{len(audit.get('sibling_groups_crossing_splits') or [])}",
        f"- hard failures: {audit.get('hard_failures') or 'none'}",
        "",
        "## Corrected representation (image-disjoint)",
        "",
        f"Exclusion rule: `{representational.get('exclusion_rule_version')}` — a "
        "candidate target is dropped whenever it shares the query's `image_id`. "
        "Group-level exclusion still applies and is superseded by it. The rule "
        "is applied identically to retrieval, matched/mismatched separation, "
        "weighted support overlap, the raw-residual baseline and the shuffled "
        "control.",
        "",
        "| pair | queries | eligible targets (min/med/max) | same-group excl. | "
        "extra same-image excl. | J top-1 | J MRR | shuffled p95 | J gap | raw top-1 | raw gap |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for pair, entry in sorted((representational.get("pairs") or {}).items()):
        exclusions = entry["exclusions"]
        eligible = exclusions["eligible_targets"]
        lines.append(
            f"| {pair} | {entry['jspace_retrieval']['n_queries']} | "
            f"{eligible['min']}/{eligible['median']}/{eligible['max']} | "
            f"{exclusions['n_excluded_same_group']} | "
            f"{exclusions['n_excluded_same_image_different_group']} | "
            f"{_percent(entry['jspace_retrieval']['top1_accuracy'])} | "
            f"{entry['jspace_retrieval']['mrr']:.3f} | "
            f"{_percent(entry['shuffled_control']['p95_top1_accuracy'])} | "
            f"{entry['jspace_separation']['gap']:+.4f} | "
            f"{_percent(entry['raw_residual_retrieval']['top1_accuracy'])} | "
            f"{entry['raw_residual_separation']['gap']:+.4f} |"
        )
        lines.append(
            f"| {pair} (support overlap) | | | | | | | | "
            f"{entry['jspace_support_overlap']['gap']:+.4f} | | |"
        )

    lines += [
        "",
        "## Corrected causation (image is the unit)",
        "",
        f"Aggregation: `{interventions.get('aggregation_version')}` — repeated "
        "observations of one photograph are averaged **within** the image, then "
        "the cell statistic is computed over images. No intervention was rerun; "
        f"all {interventions.get('n_records')} saved units were re-aggregated. "
        f"Group-level rows are preserved under `group_level` for provenance.",
        "",
        "| concept | pair | alpha | control | n groups | n images | +img | -img | "
        "effect (image) | effect (group) | delta | sign frac | unrelated | norm ratio |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in interventions.get("rows", []):
        if not row["off_diagonal"] or row["pair"] not in _TEXT_IMAGE_PAIRS:
            continue
        group_effect = (row.get("group_level") or {}).get("mean_signed_target_effect")
        delta = row["divergence_from_group_level"]
        group_cell = "n/a" if group_effect is None else f"{group_effect:+.4f}"
        delta_cell = "n/a" if delta is None else f"{delta:+.4f}"
        lines.append(
            f"| {row['concept']} | {row['pair']} | {row['alpha']:g} | "
            f"{row['control_kind']} | {row['n_groups']} | {row['n_distinct_images']} | "
            f"{row['n_positive_images']} | {row['n_negative_images']} | "
            f"{row['mean_signed_target_effect']:+.4f} | "
            f"{group_cell} | {delta_cell} | "
            f"{row['fraction_expected_sign']:.2f} | "
            f"{row['mean_abs_unrelated_change']:.4f} | "
            f"{row['mean_activation_norm_ratio']:.3f} |"
        )

    replication = verdict["replication"]
    lines += [
        "",
        f"- cells pseudoreplicated at group level: "
        f"{divergence.get('n_rows_pseudoreplicated_at_group_level')} of {divergence.get('n_rows')}",
        f"- max |image-level minus group-level| effect: {divergence.get('max_abs_divergence'):.4f}",
        f"- off-diagonal source cells resting on a single image: "
        f"{divergence.get('off_diagonal_source_rows_on_a_single_image') or 'none'}",
        "",
        "## Replication",
        "",
        f"- by concept: {replication['by_concept']}",
        f"- by direction: {replication['by_direction']}",
        f"- replicates across at least two concepts: {replication['replicates_across_concepts']}",
        f"- replicates across both directions: {replication['replicates_across_directions']}",
        "",
        "## Amended verdict",
        "",
        f"**{verdict['verdict']}** — {verdict['rationale']}",
        "",
        "| criterion | status |",
        "| --- | --- |",
    ]
    lines += [
        f"| {name} | {status.replace('_', ' ')} |"
        for name, status in verdict["criteria_status"].items()
    ]
    lines += [
        "",
        "## Scope and limits",
        "",
        f"- {verdict['late_layer_limitation']}",
        "- Distinct captions from one image remain visible per cell as "
        "descriptive group-level detail. They are never counted as independent "
        "image observations.",
        "- Identical image conditions are never counted more than once as "
        "independent observations.",
        "- Interventions add and subtract a direction on the residual stream. "
        "That is not erasure and not projection ablation.",
        "- This audit re-reads a completed run. It cannot repair a defect in "
        "how that run selected its targets — only report it.",
        "",
        "## Preservation",
        "",
        "The original artifacts were read and left byte-identical:",
        "",
        "| artifact | checksum before | unchanged |",
        "| --- | --- | --- |",
    ]
    lines += [
        f"| {name} | `{checksum}` | {preservation['unchanged'].get(name)} |"
        for name, checksum in sorted(preservation["before"].items())
    ]
    lines += [
        "",
        "## Audit fingerprint",
        "",
        "```json",
        json.dumps(fingerprint.to_dict(), indent=2, default=str),
        "```",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------- the runner


def run_image_independence_audit(
    run_dir: str | os.PathLike[str],
    *,
    subset_path: str | os.PathLike[str] | None = None,
    layer: int | None = None,
    config: VerdictConfig | None = None,
) -> dict:
    """Audit a completed run for same-image dependence. CPU only.

    Returns a dict with the amended verdict, every artifact path, the resume
    status, and the preservation evidence. Raises nothing for a scientific
    failure — a blocked or negative result is a return value. It *does* raise
    :class:`~jlens.mmpilot.store.IncompatibleStateError` for a fingerprint
    mismatch and :class:`AuditInputError` for a run directory it cannot read,
    because those are not results.
    """
    config = config or VerdictConfig()
    run = load_run(run_dir, subset_path=subset_path, layer=layer)
    fingerprint = AuditFingerprint(
        original_run_fingerprint_digest=run.run_fingerprint.digest,
        original_subset_checksum=run.subset_checksum,
        expanded_manifest_checksum=run.expanded_manifest_checksum,
        lens_checksum=run.run_fingerprint.lens_checksum,
        selected_layer=run.selected_layer,
        verdict_config=config.to_dict(),
    )
    workspace = AuditWorkspace(run.run_dir, fingerprint)
    status = workspace.open()

    activations = list(run.units["activation"].values())
    codes = list(run.units["jspace"].values())
    interventions = list(run.units["intervention"].values())
    capability_units = list(run.units["capability"].values())

    modality_records: dict[str, int] = {}
    for record in activations:
        modality_records[str(record["group_id"])] = modality_records.get(
            str(record["group_id"]), 0
        ) + 1

    # ------------------------------------------------------------- identity
    try:
        identity = resolve_image_identity(
            [*activations, *codes, *capability_units, *interventions],
            subset=run.subset,
        )
    except ImageIdentityError as exc:
        blocked = _blocked(run, fingerprint, workspace, str(exc), status)
        return blocked

    concepts = sorted(
        {str(g.concept) for g in identity.groups.values() if g.concept}
    )
    audit = workspace.load("audit")
    if audit is None:
        audit = audit_image_independence(
            identity,
            interventions=interventions,
            modality_records=modality_records,
            concepts=concepts,
        )
        audit["identity"] = identity.to_dict()
        workspace.save("audit", audit)
    if audit.get("hard_failures"):
        return _blocked(
            run,
            fingerprint,
            workspace,
            "image-level hard failure: "
            + "; ".join(f["detail"] for f in audit["hard_failures"]),
            status,
            audit=audit,
        )

    # ---------------------------------------------------- representation
    modalities = list(
        dict.fromkeys(
            record["modality"]
            for record in sorted(activations, key=lambda r: r["sample_id"])
        )
    )
    representational = workspace.load("representational")
    if representational is None:
        try:
            representational = recompute_representational(
                activations,
                codes,
                identity,
                layer=run.selected_layer,
                modalities=modalities,
                n_permutations=config.n_permutations,
                seed=config.seed,
            )
        except NoEligibleTargetError as exc:
            return _blocked(run, fingerprint, workspace, str(exc), status, audit=audit)
        workspace.save("representational", representational)

    # --------------------------------------------------------- causation
    image_level = workspace.load("interventions")
    if image_level is None:
        original_intervention_summary = (run.summary.get("interventions") or None)
        image_level = summarize_interventions_by_image(
            interventions,
            identity,
            group_summary=original_intervention_summary
            or summarize_interventions(interventions),
        )
        workspace.save("interventions", image_level)
    if image_level.get("hard_failures"):
        return _blocked(
            run,
            fingerprint,
            workspace,
            "causal aggregation hard failure: "
            + json.dumps(image_level["hard_failures"], default=str),
            status,
            audit=audit,
        )

    divergence = divergence_summary(image_level)
    verdict = amended_verdict(
        capability=run.summary.get("capability") or {},
        lens_validation=run.summary.get("lens_validation"),
        code_stats=run.summary.get("code_statistics") or {},
        reconstruction_control=run.summary.get("reconstruction_control"),
        representational=representational,
        interventions=image_level,
        audit=audit,
        blocked_modalities=run.summary.get("blocked_modalities") or [],
        config=config,
    )

    preservation = _preservation(run)
    summary = {
        "schema": "jlens.mmpilot.summary_image_disjoint.v1",
        "run_dir": str(run.run_dir),
        "mode": run.summary.get("mode"),
        "scientific_evidence": run.summary.get("scientific_evidence"),
        "audit_fingerprint": fingerprint.to_dict(),
        "audit_fingerprint_digest": fingerprint.digest,
        "selected_layer": run.selected_layer,
        "subset_source": run.subset_source,
        "model_loaded": False,
        "interventions_rerun": False,
        "original_recommendation": run.summary.get("recommendation"),
        "amended_verdict": verdict["verdict"],
        "verdict": verdict,
        "image_independence_audit": audit,
        "representational_image_disjoint": representational,
        "interventions_image_level": image_level,
        "divergence": divergence,
        "preservation": preservation,
        "resume": workspace.status_report(),
    }
    workspace.save("summary", summary)
    markdown = render_report(
        run_dir=str(run.run_dir),
        verdict=verdict,
        audit=audit,
        representational=representational,
        interventions=image_level,
        divergence=divergence,
        original=run.summary,
        fingerprint=fingerprint,
        resume=workspace.status_report(),
        preservation=preservation,
    )
    workspace.save_markdown("report", markdown)

    resume = workspace.status_report()
    return {
        "ok": True,
        "verdict": verdict["verdict"],
        "rationale": verdict["rationale"],
        "original_recommendation": run.summary.get("recommendation"),
        "audit": audit,
        "representational": representational,
        "interventions_image_level": image_level,
        "divergence": divergence,
        "replication": verdict["replication"],
        "summary": summary,
        "report_markdown": markdown,
        "artifacts": {
            name: str(workspace.path(name)) for name in ARTIFACTS
        },
        "artifact_checksums": workspace.artifact_checksums(),
        "resume": resume,
        "status": status,
        "preservation": preservation,
        "model_loaded": False,
    }


def _preservation(run: LoadedRun) -> dict:
    after = _protected_checksums(run.run_dir)
    return {
        "before": run.original_checksums,
        "after": after,
        "unchanged": {
            name: after.get(name) == checksum
            for name, checksum in run.original_checksums.items()
        },
        "all_unchanged": all(
            after.get(name) == checksum
            for name, checksum in run.original_checksums.items()
        ),
        "protected_names": list(PROTECTED_NAMES),
    }


def _blocked(
    run: LoadedRun,
    fingerprint: AuditFingerprint,
    workspace: AuditWorkspace,
    reason: str,
    status: str,
    *,
    audit: Mapping | None = None,
) -> dict:
    """Stop with ``AUDIT_BLOCKED`` and say exactly what stopped it."""
    preservation = _preservation(run)
    summary = {
        "schema": "jlens.mmpilot.summary_image_disjoint.v1",
        "run_dir": str(run.run_dir),
        "audit_fingerprint": fingerprint.to_dict(),
        "audit_fingerprint_digest": fingerprint.digest,
        "selected_layer": run.selected_layer,
        "model_loaded": False,
        "interventions_rerun": False,
        "original_recommendation": run.summary.get("recommendation"),
        "amended_verdict": AUDIT_BLOCKED,
        "blocked_reason": reason,
        "image_independence_audit": dict(audit) if audit else None,
        "preservation": preservation,
        "resume": workspace.status_report(),
    }
    workspace.save("summary", summary)
    markdown = "\n".join(
        [
            f"# Image-independence audit — {AUDIT_BLOCKED}",
            "",
            f"- audited run: `{run.run_dir}`",
            f"- original recommendation: **{run.summary.get('recommendation')}** "
            "(unchanged on disk)",
            "- **No model was loaded.** Nothing was recomputed.",
            "",
            "## Why the audit stopped",
            "",
            reason,
            "",
            "No amended scientific verdict is issued. This is not a finding "
            "against the original result and must not be read as one: it says "
            "the corrected measurement could not be made.",
            "",
        ]
    )
    workspace.save_markdown("report", markdown)
    return {
        "ok": False,
        "verdict": AUDIT_BLOCKED,
        "rationale": reason,
        "original_recommendation": run.summary.get("recommendation"),
        "audit": dict(audit) if audit else None,
        "summary": summary,
        "report_markdown": markdown,
        "artifacts": {name: str(workspace.path(name)) for name in ARTIFACTS},
        "artifact_checksums": workspace.artifact_checksums(),
        "resume": workspace.status_report(),
        "status": status,
        "preservation": preservation,
        "model_loaded": False,
    }


__all__ = [
    "ARTIFACTS",
    "AUDIT_BLOCKED",
    "GO_CONFIRMED",
    "GO_REQUIRES",
    "NO_GO",
    "PROTECTED_NAMES",
    "VERDICT_CONFIG_VERSION",
    "WEAK_GO",
    "AuditFingerprint",
    "AuditInputError",
    "AuditWorkspace",
    "LoadedRun",
    "VerdictConfig",
    "amended_verdict",
    "file_checksum",
    "load_run",
    "render_report",
    "replication_report",
    "run_image_independence_audit",
]
