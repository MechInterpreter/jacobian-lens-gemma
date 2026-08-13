# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""The corrected wrong-layer control for the L32-L40 band, and what it needs.

The defect this module repairs
==============================

The completed band-interior validation
(``runs/mmband/bandlens_real_de9338ec2a6e``) built its wrong-layer control as::

    layer_mapped_lens(lens, distant_layer_mapping(layers))

with ``layers`` equal to the *newly fitted subset* ``(33, 34, 36, 37, 39)``.
:func:`~jlens.controls.distant_layer_mapping` maps each layer to the farthest
**fitted** layer, so over that subset the control became::

    33 -> 39   34 -> 39   36 -> 33   37 -> 33   39 -> 33

Every substituted Jacobian is therefore another nearby, strong, late-workspace
lens, and clearing ``J-lens MRR >= wrong-layer MRR + 0.15`` means beating a
late layer by 0.15 rather than beating a genuinely distant one.

The earlier scale-250 study that admitted L32/L35/L38/L40 ran the same nominal
control over its own broad fitted grid ``(8, 14, 20, 26, 32, 35, 38, 40)``,
where the same function sends every late layer to **L8**. So the identical
``+0.15`` gate carried a materially different difficulty depending only on which
layers happened to be fitted together, and the two validations are not
comparable. L36/L37/L39 failed *only* that clause; L33/L34 additionally failed
fold stability, whose folds are required to beat the same control.

What the correction is, and what it is not
==========================================

The repair is to the **control's layer inventory**, nothing else:

* the wrong-layer mapping is computed once, from a **fixed** control universe
  (:data:`FIXED_CONTROL_UNIVERSE`) that does not depend on which layers a given
  run fitted, and is frozen as :data:`WRONG_LAYER_MAPPING`;
* every threshold in :data:`~jlens.calibration.extension.EXTENSION_CONFIRMATION_GATE`
  is used unchanged — this module never constructs a gate;
* no matrix is refitted. The scale-250 matrices are read from the two existing
  snapshots and proved equivalent by :func:`build_control_universe`;
* the superseded run is opened read-only and its report is never rewritten.

Because the correction was designed *after* seeing the failed report, the failed
run's 256 development and 256 confirmation records are treated as development
history. A new deterministic 256-prompt confirmation population is drawn with
every previously opened record excluded, and the protocol and the selection are
frozen and persisted **before** that population is opened
(:func:`assert_protocol_persisted`).

All nine physical layers 32-40 are scored together on that one population under
one manifest. Old verdicts for 32/35/38/40 are never combined with new verdicts
for the interior — :func:`corrected_band_verdict` refuses a row whose
confirmation manifest checksum is not the common one.

The original ``BAND_INTERIOR_LENS_NO_GO`` remains an immutable historical
result. It is superseded for band-admissibility purposes because its
wrong-layer control depended on the fitted subset; it is not fraudulent and it
is not erased.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from jlens.calibration.corpus import MAX_HAMMING_DISTANCE, CorpusRecord
from jlens.calibration.extension import (
    EXTENSION_CONFIRMATION_GATE,
    FreshEvaluationSplits,
    audit_fresh_split_leakage,
    build_fresh_evaluation_splits,
)
from jlens.calibration.gate import CONTROL_SEED, CalibrationGate
from jlens.calibration.plan import normalized_depth
from jlens.calibration.state import CalibrationStore
from jlens.controls import control_lens, distant_layer_mapping, mapping_provenance
from jlens.metadata import file_sha256
from jlens.mmpilot.band_lens import BAND_INTERIOR_LAYERS, BAND_SCALE, BAND_WINDOW
from jlens.mmpilot.store import payload_checksum

__all__ = [
    "BAND_SCORING_LAYERS",
    "CAPTURE_GEOMETRY_FIELDS",
    "CONTROL_ONLY_LAYERS",
    "CORRECTED_ARTIFACT_SCHEMA",
    "CORRECTED_BAND_GO",
    "CORRECTED_BAND_NO_GO",
    "CORRECTED_CONFIRMATION_PROMPT_SEED",
    "CORRECTED_GATE",
    "CORRECTED_PUBLISHED_STATUS",
    "CORRECTED_REPORT_SCHEMA",
    "CORRECTED_SCALE",
    "CORRECTED_SPLIT_PROTOCOL",
    "CORRECTED_SPLIT_SEED",
    "CORRECTED_STAGES",
    "CORRECTED_TARGET_LAYER",
    "CORRECTED_VALIDATION_DIRNAME",
    "CORRECTION_PROTOCOL_VERSION",
    "FIXED_CONTROL_UNIVERSE",
    "READOUT_IMPLEMENTATION",
    "SUPERSEDED_REPORT_NAME",
    "SUPERSEDED_RUN_NAME",
    "SUPERSEDED_VERDICT",
    "SUPERSEDED_WRONG_LAYER_MAPPING",
    "TARGET_TOKEN_DISCOVERY_PROTOCOL",
    "WRONG_LAYER_MAPPING",
    "ControlUniverse",
    "CorrectedControlRefused",
    "CorrectedControlStore",
    "LensSnapshot",
    "assert_no_opened_records",
    "assert_protocol_persisted",
    "assert_superseded_run_unchanged",
    "band_interior_snapshot_facts",
    "build_control_universe",
    "build_corrected_confirmation_population",
    "corrected_band_verdict",
    "corrected_confirmation_manifest",
    "corrected_control_lenses",
    "corrected_layer_rows",
    "corrected_protocol_record",
    "corrected_readout_budget",
    "corrected_validation_report",
    "evaluate_corrected_layers",
    "extension_snapshot_facts",
    "format_corrected_protocol",
    "format_corrected_readout_budget",
    "format_corrected_verdict",
    "format_wrong_layer_mapping",
    "publish_corrected_layer",
    "read_scale_snapshot",
    "score_corrected_readout_rows",
    "superseded_run_digest",
    "verify_reconstructed_superseded_splits",
    "wrong_layer_mapping_for_universe",
]


class CorrectedControlRefused(RuntimeError):
    """The corrected control cannot be built or trusted as specified."""


# --------------------------------------------------------------- the protocol

#: Bound into every corrected artifact, into the confirmation manifest and into
#: the resume fingerprint. A new version invalidates every stored unit.
CORRECTION_PROTOCOL_VERSION = "mmpilot.band_interior_corrected_control.v1"

CORRECTED_REPORT_SCHEMA = "jlens.mmpilot.band_interior_corrected_validation_report.v1"
CORRECTED_ARTIFACT_SCHEMA = "jlens.mmpilot.band_interior_corrected_lens_artifact.v1"
CORRECTED_PUBLISHED_STATUS = "PUBLISHED_CORRECTED_CONTROL_BAND_LAYER"

#: New directory, never an existing one. Published artifacts of the superseded
#: run — of which there are none, because it published nothing — and of the
#: extension are left exactly where they are.
CORRECTED_VALIDATION_DIRNAME = "corrected_validation_v1"

#: **The fix.** The control's layer inventory, frozen as an ordered tuple and
#: independent of which layers any run happened to fit. The four shallow layers
#: are the extension's descriptive grid: they exist to *define* controls and are
#: never band members.
FIXED_CONTROL_UNIVERSE: tuple[int, ...] = (
    8, 14, 20, 26, 32, 33, 34, 35, 36, 37, 38, 39, 40,
)

#: Every physical layer a 32-40 band clamps. These and only these are scored,
#: and a full band needs all nine.
BAND_SCORING_LAYERS: tuple[int, ...] = tuple(
    range(int(BAND_WINDOW[0]), int(BAND_WINDOW[1]) + 1)
)

#: Universe members that are controls only.
CONTROL_ONLY_LAYERS: tuple[int, ...] = tuple(
    layer for layer in FIXED_CONTROL_UNIVERSE if layer not in set(BAND_SCORING_LAYERS)
)


def wrong_layer_mapping_for_universe(
    universe: Sequence[int] = FIXED_CONTROL_UNIVERSE,
) -> dict[int, int]:
    """``distant_layer_mapping`` over the **fixed universe**, not a fitted subset.

    This is the whole correction in one call. The argument is the frozen
    inventory, so adding or removing newly fitted layers cannot move a single
    arrow in the result.
    """
    return distant_layer_mapping(universe)


#: The explicit mapping, computed once from the fixed universe and printed
#: before any confirmation data is opened. Every band layer 32-40 is transported
#: with the Jacobian fitted at L8 — the same substitution the scale-250 study's
#: late layers faced, which is what makes the two validations comparable.
WRONG_LAYER_MAPPING: dict[int, int] = wrong_layer_mapping_for_universe()

#: What the superseded run used, kept so the two can be printed side by side.
SUPERSEDED_WRONG_LAYER_MAPPING: dict[int, int] = distant_layer_mapping(
    BAND_INTERIOR_LAYERS
)

SUPERSEDED_RUN_NAME = "bandlens_real_de9338ec2a6e"
SUPERSEDED_VERDICT = "BAND_INTERIOR_LENS_NO_GO"
SUPERSEDED_REPORT_NAME = "band_interior_lens_report.json"

#: The scale every band lens shares. Not a new choice and not this module's to
#: make: it is the scale of the lenses already in the band.
CORRECTED_SCALE = int(BAND_SCALE)
CORRECTED_TARGET_LAYER = 41

#: The frozen gate, used unchanged. This module never constructs a gate and
#: never overrides a threshold; ``CORRECTED_GATE is EXTENSION_CONFIRMATION_GATE``.
CORRECTED_GATE: CalibrationGate = EXTENSION_CONFIRMATION_GATE

#: Fresh seeds, so the new population is a genuinely new draw rather than the
#: superseded run's under another name.
CORRECTED_SPLIT_PROTOCOL = "band-corrected-control-confirmation-hash-bucket-v1"
CORRECTED_SPLIT_SEED = 20260901
CORRECTED_CONFIRMATION_PROMPT_SEED = 20260902

TARGET_TOKEN_DISCOVERY_PROTOCOL = (
    "jlens.calibration.gate.ordinary_next_token_argmax: the frozen model's "
    "ordinary output path, argmax at the last prompt position. No parameter "
    "through which a J-lens or a candidate layer could be supplied."
)
READOUT_IMPLEMENTATION = (
    "jlens.mmpilot.band_control.score_corrected_readout_rows.v1: one forward "
    "pass per prompt, ActivationRecorder at every scored layer and the target "
    "layer, jlens.mmlocalize.lens_validity.tie_aware_row for every "
    "(prompt, layer, variant)"
)

CORRECTED_BAND_GO = "BAND_CORRECTED_CONTROL_GO"
CORRECTED_BAND_NO_GO = "BAND_CORRECTED_CONTROL_NO_GO"

#: Stages this correction writes. Kept apart from the band study's own stage
#: vocabulary so a corrected unit can never be mistaken for a superseded one.
CORRECTED_STAGES = (
    "corrected_protocol",
    "corrected_universe",
    "corrected_population",
    "corrected_readout",
    "corrected_development_readout",
    "corrected_confirmation",
    "corrected_development",
    "corrected_publication",
    "corrected_verdict",
)


def format_wrong_layer_mapping(
    mapping: Mapping[int, int] = WRONG_LAYER_MAPPING,
    *,
    superseded: Mapping[int, int] | None = None,
    title: str = "CORRECTED WRONG-LAYER CONTROL",
) -> str:
    """The explicit mapping block, printed before any confirmation is opened."""
    lines = [
        "=" * 78,
        title,
        "=" * 78,
        f"  control universe   {list(FIXED_CONTROL_UNIVERSE)}",
        f"  scored layers      {list(BAND_SCORING_LAYERS)}",
        f"  control-only       {list(CONTROL_ONLY_LAYERS)} (never band members)",
        "",
        f"  {'applied at':>10}  {'J fitted at':>12}  {'distance':>8}",
    ]
    for row in mapping_provenance(dict(mapping)):
        lines.append(
            f"  {row['applied_at_layer']:>10}  {row['jacobian_fitted_at_layer']:>12}  "
            f"{row['layer_distance']:>8}"
        )
    if superseded is not None:
        lines += [
            "",
            "  superseded (set-dependent) mapping, for comparison only:",
            "    " + ", ".join(
                f"{layer}->{source}" for layer, source in sorted(dict(superseded).items())
            ),
            "",
            "  The superseded mapping was computed over the newly fitted subset",
            "  alone, so every substituted Jacobian was another nearby late",
            "  layer. The corrected mapping is computed over the fixed universe",
            "  above and does not move when the fitted subset changes.",
        ]
    return "\n".join(lines)


def superseded_run_digest(run_dir: str | os.PathLike[str]) -> dict:
    """A byte-level digest of every file in the superseded run, sorted.

    Recorded before and after the correction so "unchanged" is a checked claim
    rather than an assurance. Nothing in this module ever writes into that
    directory.
    """
    root = Path(run_dir)
    if not root.is_dir():
        raise CorrectedControlRefused(
            f"the superseded run directory {root} does not exist; its immutability "
            "cannot be demonstrated, so the correction refuses to claim it"
        )
    files = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(str(path)),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]
    payload = {
        "protocol": CORRECTION_PROTOCOL_VERSION,
        "run_dir": str(root),
        "n_files": len(files),
        "files": files,
        "written_by_this_correction": False,
    }
    payload["tree_checksum"] = payload_checksum(payload)
    return payload


def assert_superseded_run_unchanged(before: Mapping, after: Mapping) -> dict:
    """Prove the superseded run is byte-for-byte what it was.

    Raises:
        CorrectedControlRefused: On any added, removed or altered file.
    """
    before_files = {row["path"]: row for row in before.get("files", ())}
    after_files = {row["path"]: row for row in after.get("files", ())}
    added = sorted(set(after_files) - set(before_files))
    removed = sorted(set(before_files) - set(after_files))
    altered = sorted(
        path
        for path in set(before_files) & set(after_files)
        if before_files[path]["sha256"] != after_files[path]["sha256"]
    )
    payload = {
        "protocol": CORRECTION_PROTOCOL_VERSION,
        "run_dir": after.get("run_dir"),
        "tree_checksum_before": before.get("tree_checksum"),
        "tree_checksum_after": after.get("tree_checksum"),
        "identical": not (added or removed or altered),
        "added": added,
        "removed": removed,
        "altered": altered,
        "statement": (
            "the superseded band-interior run is completed evidence; this "
            "correction reads it and never writes to it"
        ),
    }
    payload["immutability_checksum"] = payload_checksum(payload)
    if not payload["identical"]:
        raise CorrectedControlRefused(
            "the superseded run directory changed during the correction "
            f"(added={added}, removed={removed}, altered={altered}). The completed "
            "failed validation is immutable historical evidence; refusing."
        )
    return payload


# ------------------------------------------------------------- the lens sources


@dataclass(frozen=True)
class LensSnapshot:
    """One scale-250 snapshot, and every clause a merge has to agree on.

    Nothing here is defaulted. A field the source artifact did not record stays
    ``None`` and :func:`build_control_universe` refuses the merge, because a
    clause that cannot be demonstrated has not been demonstrated.
    """

    name: str
    path: str
    file_checksum: str
    layers: tuple[int, ...]
    matrix_checksums: dict[int, str]
    scale: int | None = None
    model_repo_id: str | None = None
    model_revision: str | None = None
    target_layer: int | None = None
    hook_site: str | None = None
    residual_convention: str | None = None
    d_model: int | None = None
    corpus_id: str | None = None
    corpus_revision: str | None = None
    fit_prefix_checksum: str | None = None
    estimator: str | None = None
    capture_geometry: dict | None = None
    hook_site_source: str | None = None

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["layers"] = [int(layer) for layer in self.layers]
        payload["matrix_checksums"] = {
            str(int(layer)): str(value)
            for layer, value in sorted(self.matrix_checksums.items())
        }
        return payload

    #: The clauses two snapshots must agree on before their matrices may sit in
    #: one band. Order is the order the refusal reports them in.
    #:
    #: ``capture_geometry`` is the recorded capture plan **without its layer
    #: grid** — target layer, width, dim batch, sequence length, skipped prefix
    #: and stack depth. It is the machine-checkable form of "the same estimator
    #: read the same positions", and it is why ``hook_site`` can be carried
    #: across from the snapshot that published a sidecar: this repository has
    #: one hook convention, and two runs with identical estimator and identical
    #: capture geometry cannot have used different ones. ``hook_site_source``
    #: records how each snapshot's value was established and is reported, never
    #: compared.
    AGREEMENT_CLAUSES = (
        "model_repo_id",
        "model_revision",
        "target_layer",
        "hook_site",
        "residual_convention",
        "d_model",
        "corpus_id",
        "corpus_revision",
        "fit_prefix_checksum",
        "scale",
        "estimator",
        "capture_geometry",
    )


@dataclass(frozen=True)
class ControlUniverse:
    """The merged, frozen inventory the corrected control is computed from."""

    layers: tuple[int, ...]
    mapping: dict[int, int]
    scoring_layers: tuple[int, ...]
    source_of_layer: dict[int, str]
    matrix_checksums: dict[int, str]
    scale: int
    evidence: dict

    @property
    def digest(self) -> str:
        return str(self.evidence["universe_checksum"])

    def source_layers_needed(self) -> tuple[int, ...]:
        """Universe layers a scored readout actually loads: the nine scored
        layers plus every layer their controls substitute in."""
        needed = {int(layer) for layer in self.scoring_layers}
        needed |= {int(self.mapping[layer]) for layer in self.scoring_layers}
        return tuple(sorted(needed))


def build_control_universe(
    snapshots: Sequence[LensSnapshot],
    *,
    universe: Sequence[int] = FIXED_CONTROL_UNIVERSE,
    scoring_layers: Sequence[int] = BAND_SCORING_LAYERS,
    scale: int = CORRECTED_SCALE,
) -> ControlUniverse:
    """Merge the scale-250 snapshots into one control inventory — or refuse.

    Refuses, rather than merging, when:

    * a snapshot declares a scale other than ``scale`` (a scale-100 artifact and
      a scale-250 artifact are different fits and are never one lens);
    * two snapshots disagree on any clause in
      :data:`LensSnapshot.AGREEMENT_CLAUSES`, or a clause is unrecorded;
    * a universe layer has no matrix in any snapshot;
    * two snapshots offer the same layer with different matrix checksums.

    Raises:
        CorrectedControlRefused: Listing every clause that failed.
    """
    wanted = tuple(int(layer) for layer in universe)
    scored = tuple(int(layer) for layer in scoring_layers)
    if not snapshots:
        raise CorrectedControlRefused(
            "no scale-250 snapshots were offered; the corrected control universe "
            f"needs matrices for {list(wanted)}"
        )
    missing_scored = sorted(set(scored) - set(wanted))
    if missing_scored:
        raise CorrectedControlRefused(
            f"scoring layers {missing_scored} are not members of the fixed control "
            f"universe {list(wanted)}; a layer cannot be scored against a control "
            "inventory it does not belong to"
        )

    problems: list[str] = []
    checks: list[dict] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})
        if not ok:
            problems.append(f"{name}: {detail}")

    # --- scale first: a mixed-scale merge is refused before anything else.
    for snapshot in snapshots:
        check(
            f"scale[{snapshot.name}]",
            snapshot.scale is not None and int(snapshot.scale) == int(scale),
            f"snapshot {snapshot.name!r} declares scale {snapshot.scale!r}, "
            f"this band requires {int(scale)}; artifacts fitted at different "
            "scales are not one lens and are never merged",
        )

    # --- every remaining identity clause, pairwise against the first snapshot.
    reference = snapshots[0]
    for clause in LensSnapshot.AGREEMENT_CLAUSES:
        if clause == "scale":
            continue
        values = {snapshot.name: getattr(snapshot, clause) for snapshot in snapshots}
        unrecorded = sorted(name for name, value in values.items() if value in (None, ""))
        if unrecorded:
            check(
                clause,
                False,
                f"unrecorded in {unrecorded}; a clause that cannot be demonstrated "
                "from stored artifacts has not been demonstrated",
            )
            continue
        agree = all(value == getattr(reference, clause) for value in values.values())
        check(clause, agree, f"{values}")

    # --- coverage and per-layer matrix agreement.
    by_layer: dict[int, tuple[str, str]] = {}
    for snapshot in snapshots:
        for layer in snapshot.layers:
            layer = int(layer)
            checksum = snapshot.matrix_checksums.get(layer) or snapshot.matrix_checksums.get(
                str(layer)
            )
            if checksum in (None, ""):
                check(
                    f"matrix_checksum[L{layer}]",
                    False,
                    f"snapshot {snapshot.name!r} lists layer {layer} but records no "
                    "matrix checksum for it",
                )
                continue
            existing = by_layer.get(layer)
            if existing is not None and existing[1] != checksum:
                check(
                    f"matrix_agreement[L{layer}]",
                    False,
                    f"{existing[0]} and {snapshot.name} both offer layer {layer} with "
                    f"different matrices ({existing[1]} vs {checksum}); which matrix a "
                    "band rests on is not decided by iteration order",
                )
                continue
            if existing is None:
                by_layer[layer] = (snapshot.name, str(checksum))

    absent = sorted(layer for layer in wanted if layer not in by_layer)
    check(
        "universe_coverage",
        not absent,
        f"no scale-{int(scale)} matrix for physical layer(s) {absent}. The fixed "
        "control universe is not negotiable: a missing interior matrix means the "
        "corrected control cannot be computed, not that the universe shrinks",
    )

    mapping = wrong_layer_mapping_for_universe(wanted)
    evidence = {
        "protocol": CORRECTION_PROTOCOL_VERSION,
        "scale": int(scale),
        "fixed_control_universe": list(wanted),
        "scoring_layers": list(scored),
        "control_only_layers": [
            layer for layer in wanted if layer not in set(scored)
        ],
        "wrong_layer_mapping": {str(k): int(v) for k, v in sorted(mapping.items())},
        "wrong_layer_mapping_provenance": mapping_provenance(mapping),
        "wrong_layer_mapping_rule": (
            "jlens.controls.distant_layer_mapping over the FIXED control universe; "
            "it is not recomputed from whichever layers a run fitted"
        ),
        "superseded_wrong_layer_mapping": {
            str(k): int(v) for k, v in sorted(SUPERSEDED_WRONG_LAYER_MAPPING.items())
        },
        "snapshots": [snapshot.to_dict() for snapshot in snapshots],
        "source_of_layer": {
            str(layer): by_layer[layer][0] for layer in sorted(by_layer)
        },
        "matrix_checksums": {
            str(layer): by_layer[layer][1] for layer in sorted(by_layer)
        },
        "checks": checks,
        "failed_checks": [row["check"] for row in checks if not row["ok"]],
        "passed": not problems,
        "no_matrix_was_refitted": True,
        "matrices_read_from": "existing scale-250 snapshots, read-only",
        "matrix_checksum_provenance": (
            "each snapshot file was resolved from its run's own scale_snapshot "
            "unit and re-checksummed against the sha256 that unit recorded; the "
            "per-layer checksums below are computed from those verified files"
        ),
    }
    evidence["universe_checksum"] = payload_checksum(evidence)
    if problems:
        raise CorrectedControlRefused(
            "the scale-250 snapshots do not establish one control universe:\n  - "
            + "\n  - ".join(problems)
            + "\nRefusing rather than merging matrices whose equivalence is unproven."
        )
    return ControlUniverse(
        layers=wanted,
        mapping=mapping,
        scoring_layers=scored,
        source_of_layer={layer: by_layer[layer][0] for layer in sorted(by_layer)},
        matrix_checksums={layer: by_layer[layer][1] for layer in sorted(by_layer)},
        scale=int(scale),
        evidence=evidence,
    )


# ------------------------------------------------- reading the two snapshots


#: Capture-plan fields that describe the estimator's geometry rather than which
#: layers it was pointed at. Two runs that agree here read the same positions of
#: the same stack into the same target basis.
CAPTURE_GEOMETRY_FIELDS = (
    "target_layer",
    "d_model",
    "dim_batch",
    "max_seq_len",
    "skip_first",
    "n_layers",
)


def _json_payload(path: Path, *, what: str) -> dict:
    if not path.is_file():
        raise CorrectedControlRefused(f"{what} not found at {path}")
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise CorrectedControlRefused(
            f"{what} at {path} is not readable JSON: {error}"
        ) from error
    if not isinstance(record, dict):
        raise CorrectedControlRefused(f"{what} at {path} does not hold a JSON object")
    payload = record.get("payload") if isinstance(record.get("payload"), dict) else record
    recorded = record.get("unit_checksum")
    if recorded is not None and payload_checksum(payload) != recorded:
        raise CorrectedControlRefused(
            f"{what} at {path} does not match its own unit checksum; a torn unit "
            "is not evidence"
        )
    return dict(payload)


def _capture_geometry(plan: Mapping | None) -> dict | None:
    if not plan:
        return None
    geometry = {
        field: plan.get(field)
        for field in CAPTURE_GEOMETRY_FIELDS
        if plan.get(field) is not None
    }
    if set(geometry) != set(CAPTURE_GEOMETRY_FIELDS):
        return None
    return {field: int(geometry[field]) for field in CAPTURE_GEOMETRY_FIELDS}


def read_scale_snapshot(
    run_dir: str | os.PathLike[str],
    *,
    scale: int = CORRECTED_SCALE,
    stage: str = "scale_snapshot",
) -> tuple[Path, str, tuple[int, ...]]:
    """Resolve a run's scale snapshot from its own unit, and re-checksum it.

    A filename is never assumed and a checksum is never invented: the path and
    the expected checksum come from the run's stored ``scale_snapshot`` unit,
    and the file on disk has to agree.

    Raises:
        CorrectedControlRefused: If the unit or the file is missing, or the
            file's sha256 differs from the recorded one.
    """
    root = Path(run_dir)
    payload = _json_payload(
        root / "units" / stage / f"scale{int(scale)}.json",
        what=f"scale-{int(scale)} snapshot unit in {root}",
    )
    path = Path(str(payload.get("path", "")))
    if not path.is_file():
        path = root / "artifacts" / path.name
    if not path.is_file():
        raise CorrectedControlRefused(
            f"{root} records a scale-{int(scale)} snapshot at "
            f"{payload.get('path')!r}, which is not present"
        )
    actual = file_sha256(str(path))
    if actual != payload.get("checksum"):
        raise CorrectedControlRefused(
            f"the scale-{int(scale)} snapshot at {path} checksums to {actual}, not "
            f"the recorded {payload.get('checksum')!r}"
        )
    return path, actual, tuple(sorted(int(value) for value in payload.get("layers", ())))


def _matrix_checksums(path: Path) -> tuple[dict[int, str], int]:
    """Per-layer checksums and the width, read from the snapshot file itself."""
    from jlens.lens import JacobianLens
    from jlens.mmpilot.jspace import tensor_checksum

    lens = JacobianLens.load(str(path))
    checksums = {
        int(layer): tensor_checksum(matrix)
        for layer, matrix in sorted(lens.jacobians.items())
    }
    return checksums, int(lens.d_model)


def extension_snapshot_facts(
    extension_run_dir: str | os.PathLike[str],
    *,
    scale: int = CORRECTED_SCALE,
    published_layer: int = 32,
    name: str = "extension_scale250_snapshot",
) -> LensSnapshot:
    """The broad scale-250 snapshot, with every clause read from that run.

    The identity clauses come from two places, both the run's own: its report
    (corpus, resolved revision, fit-prefix checksum, capture plan) and the
    published artifact sidecar for ``published_layer``, which is where
    :func:`jlens.calibration.publication.build_artifact` records the hook site
    and the residual convention.
    """
    from jlens.mmpilot.l32_followup import (
        discover_published_l32_lens,
        read_extension_report,
    )
    from jlens.mmpilot.published_lens import read_artifact_sidecar

    run_dir = Path(extension_run_dir)
    _, report = read_extension_report(run_dir)
    discovered = discover_published_l32_lens(
        run_dir, layer=int(published_layer), expected_scale=int(scale)
    )
    _, artifact = read_artifact_sidecar(Path(discovered.lens_path))

    corpus = dict(report.get("corpus") or {})
    plan = dict((report.get("continuation") or {}).get("capture_plan") or {})
    if not plan:
        plan = dict((report.get("budget") or {}).get("anchor") or {})
    path, file_checksum, layers = read_scale_snapshot(run_dir, scale=scale)
    checksums, d_model = _matrix_checksums(path)
    return LensSnapshot(
        name=str(name),
        path=str(path),
        file_checksum=file_checksum,
        layers=layers or tuple(sorted(checksums)),
        matrix_checksums=checksums,
        scale=int(scale),
        model_repo_id=artifact.get("model_repo_id"),
        model_revision=artifact.get("model_revision"),
        target_layer=artifact.get("target_layer") or plan.get("target_layer"),
        hook_site=artifact.get("hook_site"),
        residual_convention=artifact.get("residual_convention"),
        d_model=d_model,
        corpus_id=corpus.get("corpus_id"),
        corpus_revision=corpus.get("revision"),
        fit_prefix_checksum=(corpus.get("fit_prefix_checksums") or {}).get(
            str(int(scale))
        ),
        estimator="jlens.fitting.fit (upstream, unmodified)",
        capture_geometry=_capture_geometry(plan),
        hook_site_source=(
            f"published scale-{int(scale)} artifact sidecar for L{int(published_layer)}"
        ),
    )


def band_interior_snapshot_facts(
    band_run_dir: str | os.PathLike[str],
    *,
    hook_convention_from: LensSnapshot,
    scale: int = CORRECTED_SCALE,
    name: str = "band_interior_scale250_snapshot",
) -> LensSnapshot:
    """The interior scale-250 snapshot of the superseded run, read-only.

    That run published nothing, so it has no artifact sidecar to state a hook
    site. The value is therefore carried across from ``hook_convention_from``
    and recorded as derived — which is sound exactly when ``capture_geometry``
    and ``estimator`` agree, and :func:`build_control_universe` refuses the
    merge if they do not.

    Nothing here writes to ``band_run_dir``.

    Raises:
        CorrectedControlRefused: If the run's report is missing, is not a real
            band-interior report, or does not record the corpus equivalence the
            clauses are read from.
    """
    run_dir = Path(band_run_dir)
    report = _json_payload(
        run_dir / "artifacts" / SUPERSEDED_REPORT_NAME,
        what=f"band-interior report in {run_dir}",
    )
    if report.get("schema") != "jlens.mmpilot.band_interior_lens_report.v1":
        raise CorrectedControlRefused(
            f"{run_dir} declares report schema {report.get('schema')!r}, not the "
            "band-interior lens report"
        )
    if report.get("mode") != "real":
        raise CorrectedControlRefused(
            f"{run_dir} was written in mode {report.get('mode')!r}; a MOCK report "
            "supplies no matrices about Gemma"
        )
    equivalence = dict(report.get("corpus_equivalence") or {})
    if not equivalence:
        raise CorrectedControlRefused(
            f"{run_dir} records no corpus_equivalence block, so the corpus, "
            "revision and fit-prefix its matrices were fitted under cannot be "
            "demonstrated from stored artifacts"
        )
    path, file_checksum, layers = read_scale_snapshot(run_dir, scale=scale)
    checksums, d_model = _matrix_checksums(path)
    return LensSnapshot(
        name=str(name),
        path=str(path),
        file_checksum=file_checksum,
        layers=layers or tuple(sorted(checksums)),
        matrix_checksums=checksums,
        scale=int(equivalence.get("scale", scale)),
        model_repo_id=equivalence.get("model_repo_id"),
        model_revision=equivalence.get("model_revision"),
        target_layer=(equivalence.get("band_capture_plan") or {}).get("target_layer"),
        hook_site=hook_convention_from.hook_site,
        residual_convention=hook_convention_from.residual_convention,
        d_model=d_model,
        corpus_id=equivalence.get("extension_corpus_id"),
        corpus_revision=equivalence.get("extension_corpus_revision"),
        fit_prefix_checksum=equivalence.get("extension_fit_prefix_checksum"),
        estimator=equivalence.get("same_estimator"),
        capture_geometry=_capture_geometry(equivalence.get("band_capture_plan")),
        hook_site_source=(
            "derived: this run published no artifact sidecar, so the value is "
            f"carried from {hook_convention_from.name!r}. Sound because the "
            "estimator and the capture geometry are proved identical, and the "
            "repository has one hook convention "
            "(jlens.hooks.ActivationRecorder over model.layers)"
        ),
    )


# ----------------------------------------------------------------- the control


def corrected_control_lenses(
    lens,
    *,
    scoring_layers: Sequence[int] = BAND_SCORING_LAYERS,
    mapping: Mapping[int, int] = WRONG_LAYER_MAPPING,
    seed: int = CONTROL_SEED,
) -> dict:
    """The three controls, restricted to the layers actually scored.

    ``permuted`` and ``random`` are :func:`jlens.controls.control_lens` at the
    unchanged :data:`jlens.calibration.gate.CONTROL_SEED`. ``wrong_layer`` is
    :func:`jlens.controls.layer_mapped_lens` under ``mapping`` — restricted to
    the scored layers so a 13-layer universe does not have to be cloned four
    times to read out nine layers. ``tests`` prove the restriction equals the
    unrestricted lens on those layers.

    Raises:
        CorrectedControlRefused: If the lens is missing a scored layer or a
            layer the mapping substitutes in. A control silently built from
            fewer layers is the defect this module exists to repair.
    """
    from jlens.lens import JacobianLens

    scored = [int(layer) for layer in scoring_layers]
    fitted = set(int(layer) for layer in lens.source_layers)
    needed = set(scored) | {int(mapping[layer]) for layer in scored if layer in mapping}
    missing_map = sorted(layer for layer in scored if layer not in mapping)
    if missing_map:
        raise CorrectedControlRefused(
            f"the wrong-layer mapping does not cover scored layer(s) {missing_map}"
        )
    absent = sorted(needed - fitted)
    if absent:
        substitutions = {layer: int(mapping[layer]) for layer in scored}
        raise CorrectedControlRefused(
            f"the lens has no matrix for layer(s) {absent}, which the corrected "
            f"control needs (scored {scored}, mapping {substitutions}). Refusing to "
            "build a control over whichever layers happen to be present — that "
            "dependence is the defect this correction repairs."
        )

    scored_only = JacobianLens(
        jacobians={layer: lens.jacobians[layer] for layer in scored},
        n_prompts=lens.n_prompts,
        d_model=lens.d_model,
    )
    wrong = JacobianLens(
        jacobians={
            layer: lens.jacobians[int(mapping[layer])].clone() for layer in scored
        },
        n_prompts=lens.n_prompts,
        d_model=lens.d_model,
    )
    return {
        "permuted": control_lens(scored_only, "permuted", seed=int(seed)),
        "random": control_lens(scored_only, "random", seed=int(seed)),
        "wrong_layer": wrong,
    }


def corrected_protocol_record(
    *,
    universe: ControlUniverse,
    gate: CalibrationGate = CORRECTED_GATE,
    model_repo_id: str,
    model_revision: str,
    processor_revision: str | None = None,
    transformers_version: str | None = None,
    exclusion_sources: Mapping[str, str],
    superseded_run: Mapping | None = None,
    control_seed: int = CONTROL_SEED,
    split_seed: int = CORRECTED_SPLIT_SEED,
    prompt_seed: int = CORRECTED_CONFIRMATION_PROMPT_SEED,
    n_confirmation: int | None = None,
) -> dict:
    """Everything the correction is bound to, frozen before any data is opened.

    A changed field changes the digest, and a changed digest makes
    :class:`CorrectedControlStore` refuse to resume rather than mix units from
    two protocols.
    """
    payload = {
        "schema": "jlens.mmpilot.band_corrected_control_protocol.v1",
        "correction_protocol_version": CORRECTION_PROTOCOL_VERSION,
        "model_repo_id": str(model_repo_id),
        "model_revision": str(model_revision),
        "processor_revision": processor_revision,
        "transformers_version": transformers_version,
        "fixed_control_universe": list(universe.layers),
        "scoring_layers": list(universe.scoring_layers),
        "wrong_layer_mapping": {
            str(k): int(v) for k, v in sorted(universe.mapping.items())
        },
        "wrong_layer_mapping_rule": universe.evidence["wrong_layer_mapping_rule"],
        "lens_checksums": {
            str(k): str(v) for k, v in sorted(universe.matrix_checksums.items())
        },
        "source_snapshot_checksums": {
            row["name"]: row["file_checksum"] for row in universe.evidence["snapshots"]
        },
        "source_of_layer": {
            str(k): str(v) for k, v in sorted(universe.source_of_layer.items())
        },
        "universe_checksum": universe.digest,
        "scale": int(universe.scale),
        "target_layer": int(CORRECTED_TARGET_LAYER),
        "hook_convention": next(
            (
                row["hook_site"]
                for row in universe.evidence["snapshots"]
                if row.get("hook_site")
            ),
            None,
        ),
        "residual_convention": next(
            (
                row["residual_convention"]
                for row in universe.evidence["snapshots"]
                if row.get("residual_convention")
            ),
            None,
        ),
        "fit_prefix_checksum": next(
            (
                row["fit_prefix_checksum"]
                for row in universe.evidence["snapshots"]
                if row.get("fit_prefix_checksum")
            ),
            None,
        ),
        "exclusion_source_checksums": {
            str(k): str(v) for k, v in sorted(dict(exclusion_sources).items())
        },
        "gate_version": gate.version,
        "gate_digest": gate.digest,
        "gate_thresholds": gate.to_dict(),
        "gate_is_the_frozen_one": gate == EXTENSION_CONFIRMATION_GATE,
        "n_confirmation_prompts": int(
            gate.n_prompts if n_confirmation is None else n_confirmation
        ),
        "target_token_discovery_protocol": TARGET_TOKEN_DISCOVERY_PROTOCOL,
        "readout_implementation": READOUT_IMPLEMENTATION,
        "split_protocol": CORRECTED_SPLIT_PROTOCOL,
        "split_seed": int(split_seed),
        "confirmation_prompt_seed": int(prompt_seed),
        "control_seed": int(control_seed),
        "random_control_seed_rule": "jlens.controls.control_lens: seed + layer",
        "permuted_control_seed_rule": "jlens.controls.control_lens: seed + layer",
        "superseded_run": dict(superseded_run or {}),
        "no_lens_was_refitted": True,
        "no_threshold_was_changed": True,
        "confirmation_population_not_inspected_when_frozen": True,
    }
    payload["protocol_digest"] = payload_checksum(payload)
    return payload


def format_corrected_protocol(protocol: Mapping) -> str:
    """The block printed before the new confirmation population is opened."""
    return "\n".join(
        [
            "=" * 78,
            f"CORRECTED CONTROL PROTOCOL — {protocol['correction_protocol_version']}",
            "=" * 78,
            f"  protocol digest    {protocol['protocol_digest']}",
            f"  universe checksum  {protocol['universe_checksum']}",
            f"  gate               {protocol['gate_version']}",
            f"  gate digest        {protocol['gate_digest']}  "
            f"(frozen: {protocol['gate_is_the_frozen_one']})",
            f"  scale / target     {protocol['scale']} -> L{protocol['target_layer']}",
            f"  hook convention    {protocol['hook_convention']}",
            f"  fit prefix         {protocol['fit_prefix_checksum']}",
            f"  split seed         {protocol['split_seed']}  "
            f"prompt seed {protocol['confirmation_prompt_seed']}  "
            f"control seed {protocol['control_seed']}",
            f"  confirmation size  {protocol['n_confirmation_prompts']}",
            "",
            "  no lens was refitted:            "
            f"{protocol['no_lens_was_refitted']}",
            "  no threshold was changed:        "
            f"{protocol['no_threshold_was_changed']}",
            "  population uninspected when frozen: "
            f"{protocol['confirmation_population_not_inspected_when_frozen']}",
        ]
    )


# ------------------------------------------------------------------ the store


class CorrectedControlStore(CalibrationStore):
    """A calibration store that accepts only the corrected stages.

    Subclassed rather than widening :data:`~jlens.calibration.state.CALIBRATION_STAGES`,
    so the completed calibration, extension and band stage vocabularies stay
    exactly as they are and a corrected unit can never land in a superseded
    stage directory.
    """

    def stage_dir(self, stage: str) -> Path:
        if stage not in CORRECTED_STAGES:
            raise ValueError(
                f"unknown corrected stage {stage!r}; known stages are "
                f"{CORRECTED_STAGES}"
            )
        return self.root / "units" / stage

    def status_report(self, stages: Sequence[str] = CORRECTED_STAGES) -> dict:
        return super().status_report(stages)

    @property
    def corrected_dir(self) -> Path:
        return self.root / "artifacts" / CORRECTED_VALIDATION_DIRNAME

    def published_path(self, layer: int, scale: int) -> Path:
        return (
            self.corrected_dir
            / "published"
            / f"lens.corrected.layer{int(layer)}.scale{int(scale)}.validated.pt"
        )


def assert_protocol_persisted(store: CorrectedControlStore) -> dict:
    """Refuse to open the confirmation population before the design is on disk.

    The order matters and is the reason this exists: a protocol frozen *after*
    a population has been looked at is not a predeclaration.

    Raises:
        CorrectedControlRefused: If the protocol or universe unit is absent.
    """
    protocol = store.load("corrected_protocol", "protocol")
    universe = store.load("corrected_universe", "universe")
    missing = [
        name
        for name, value in (("corrected_protocol/protocol", protocol),
                            ("corrected_universe/universe", universe))
        if value is None
    ]
    if missing:
        raise CorrectedControlRefused(
            f"{missing} has not been persisted. The corrected control definition, "
            "layer inventory, selection rule, seeds and thresholds are frozen and "
            "written before a new confirmation population is opened — a protocol "
            "recorded afterwards is not a predeclaration."
        )
    return {
        "protocol_digest": protocol.get("protocol_digest"),
        "universe_checksum": universe.get("universe_checksum"),
        "persisted_before_population_opened": True,
    }


# ------------------------------------------------------- the new population


def build_corrected_confirmation_population(
    pool: Sequence[CorpusRecord],
    *,
    excluded: Mapping[str, Sequence[CorpusRecord]],
    corpus_id: str,
    seed: int = CORRECTED_SPLIT_SEED,
    n_confirmation: int | None = None,
    max_hamming: int = MAX_HAMMING_DISTANCE,
) -> tuple[FreshEvaluationSplits, dict]:
    """One deterministic, untouched confirmation population, plus its audit.

    The construction, the exact-duplicate arithmetic and the banded-SimHash
    near-duplicate rule are the repository's frozen ones
    (:func:`jlens.calibration.extension.build_fresh_evaluation_splits`) — only
    the seed and the protocol label are new. There is exactly one selection: no
    search over seeds, and no development partition, because the corrected
    development diagnostics rescore *already opened* records and are never
    independent confirmation.

    Raises:
        CorrectedControlRefused: If the caller did not offer the superseded
            run's own development and confirmation sets for exclusion. Those two
            sets have been read and their verdicts acted on; they are the whole
            reason a new population exists.
    """
    required = {"superseded_development", "superseded_confirmation"}
    missing = sorted(required - set(excluded))
    if missing:
        raise CorrectedControlRefused(
            f"the exclusion map is missing {missing}. The superseded run's 256 "
            "development and 256 confirmation records have been opened and read; a "
            "layer confirmed on them would be confirmed on development history."
        )
    size = int(CORRECTED_GATE.n_prompts if n_confirmation is None else n_confirmation)
    splits = build_fresh_evaluation_splits(
        pool,
        excluded=excluded,
        corpus_id=corpus_id,
        seed=int(seed),
        n_development=0,
        n_confirmation=size,
        max_hamming=int(max_hamming),
    )
    splits = replace(splits, protocol=CORRECTED_SPLIT_PROTOCOL)
    audit = audit_fresh_split_leakage(splits, excluded=excluded, max_hamming=max_hamming)
    return splits, audit


def verify_reconstructed_superseded_splits(
    splits: FreshEvaluationSplits, *, band_report: Mapping
) -> dict:
    """Prove a re-derived copy of the superseded run's sets is the same draw.

    That run recorded checksums and record ids, not prompt text, so the only way
    to exclude the 512 records it opened is to rebuild them under its own seed
    and prove the rebuild reproduces its recorded checksums. An unproven
    reconstruction would exclude the wrong records and leave the real ones in
    the new confirmation population.

    Raises:
        CorrectedControlRefused: On any disagreement.
    """
    recorded = dict(band_report.get("fresh_splits") or {})
    expected = dict(recorded.get("checksums") or {})
    sizes = dict(recorded.get("sizes") or {})
    rows = []
    for name in ("development", "confirmation"):
        actual = splits.checksum(name)
        rows.append(
            {
                "partition": name,
                "expected_checksum": expected.get(name),
                "actual_checksum": actual,
                "expected_size": sizes.get(name),
                "actual_size": len(splits.get(name)),
                "matches": expected.get(name) == actual,
            }
        )
    payload = {
        "protocol": CORRECTION_PROTOCOL_VERSION,
        "superseded_run": SUPERSEDED_RUN_NAME,
        "superseded_verdict": band_report.get("band_verdict", {}).get("verdict"),
        "superseded_split_protocol": recorded.get("protocol"),
        "superseded_manifest_checksum": recorded.get("manifest_checksum"),
        "partitions": rows,
        "all_match": all(row["matches"] for row in rows),
        "why": (
            "the superseded run's 256 development and 256 confirmation records "
            "have been opened, scored and read; they are development-only data "
            "for this correction, and an exclusion list is trustworthy only if "
            "the reconstruction that produced it is proved to be the same draw"
        ),
    }
    payload["verification_checksum"] = payload_checksum(payload)
    if not payload["all_match"]:
        failed = [row for row in rows if not row["matches"]]
        raise CorrectedControlRefused(
            "rebuilding the superseded run's evaluation sets did not reproduce its "
            f"recorded checksums: {failed}. The corpus stream, the pool, the "
            "exclusions or the split seed differs, so the records this correction "
            "would exclude are not the ones that run actually spent. Refusing."
        )
    return payload


def assert_no_opened_records(
    splits: FreshEvaluationSplits, *, opened_record_ids: Mapping[str, Sequence[str]]
) -> dict:
    """Second, independent guard: no previously opened record id may appear.

    The builder excludes by normalized checksum and SimHash. This checks the
    same fact from the other direction — the record ids the superseded and
    extension runs actually recorded — so a reconstruction that silently drifted
    cannot leave a spent prompt in the new population unnoticed.

    Raises:
        CorrectedControlRefused: Naming every offending record and its source.
    """
    new_ids = set(splits.record_ids("confirmation"))
    hits = []
    for source, ids in dict(opened_record_ids).items():
        for record_id in ids:
            if str(record_id) in new_ids:
                hits.append({"record_id": str(record_id), "opened_by": str(source)})
    payload = {
        "protocol": CORRECTION_PROTOCOL_VERSION,
        "n_confirmation": len(new_ids),
        "checked_sources": {
            str(name): len(list(ids)) for name, ids in dict(opened_record_ids).items()
        },
        "hits": hits,
        "ok": not hits,
    }
    payload["opened_record_audit_checksum"] = payload_checksum(payload)
    if hits:
        raise CorrectedControlRefused(
            f"{len(hits)} previously opened record(s) reached the new confirmation "
            f"population; the first is {hits[0]}. A population containing prompts "
            "an earlier run already scored is exactly as spent as the one it "
            "replaces. Refusing."
        )
    return payload


def corrected_confirmation_manifest(
    splits: FreshEvaluationSplits,
    *,
    protocol: Mapping,
    prompts: Sequence[str],
    selection: Mapping,
    exclusion_digest: str,
    corpus_revision: str,
    leakage_audit: Mapping,
    opened_record_audit: Mapping,
) -> dict:
    """The frozen identity of the population, written before it is scored.

    Carries the selection seed, the source corpus revision, the exclusion-set
    digest, the record ids, the prompt hashes, the target-token discovery
    protocol, the diversity requirements and its own checksum. Everything a
    reader needs to ask "is this the population that was declared?" and get an
    answer that does not depend on trusting the run.
    """
    prompt_hashes = [
        "sha256:" + hashlib.sha256(str(prompt).encode()).hexdigest()
        for prompt in prompts
    ]
    payload = {
        "schema": "jlens.mmpilot.band_corrected_confirmation_manifest.v1",
        "correction_protocol_version": CORRECTION_PROTOCOL_VERSION,
        "protocol_digest": protocol.get("protocol_digest"),
        "split_protocol": splits.protocol,
        "selection_seed": int(splits.seed),
        "confirmation_prompt_seed": int(protocol.get("confirmation_prompt_seed", 0)),
        "source_corpus_id": splits.corpus_id,
        "source_corpus_revision": str(corpus_revision),
        "exclusion_set_digest": str(exclusion_digest),
        "n_records": len(splits.confirmation),
        "record_ids": list(splits.record_ids("confirmation")),
        "confirmation_split_checksum": splits.checksum("confirmation"),
        "n_prompts": len(prompt_hashes),
        "prompt_hashes": prompt_hashes,
        "target_token_discovery_protocol": TARGET_TOKEN_DISCOVERY_PROTOCOL,
        "target_token_selection_checksum": selection.get("selection_checksum"),
        "selected_by_jlens_performance": selection.get(
            "selected_by_jlens_performance", False
        ),
        "diversity": selection.get("diversity"),
        "min_distinct_target_tokens": protocol.get("gate_thresholds", {}).get(
            "min_distinct_target_tokens"
        ),
        "max_target_token_share": protocol.get("gate_thresholds", {}).get(
            "max_target_token_share"
        ),
        "readout_implementation": READOUT_IMPLEMENTATION,
        "leakage_audit_checksum": leakage_audit.get("audit_checksum"),
        "opened_record_audit_checksum": opened_record_audit.get(
            "opened_record_audit_checksum"
        ),
        "one_deterministic_predeclared_selection": True,
        "population_searched_for_a_favourable_outcome": False,
        "is_independent_confirmation": True,
    }
    payload["manifest_checksum"] = payload_checksum(payload)
    return payload


# --------------------------------------------------------------- the readout


def _prompt_sha(prompt: str) -> str:
    return hashlib.sha256(str(prompt).encode()).hexdigest()


def score_corrected_readout_rows(
    model,
    lens,
    prompts: Sequence[str],
    *,
    scoring_layers: Sequence[int] = BAND_SCORING_LAYERS,
    target_layer: int = CORRECTED_TARGET_LAYER,
    max_seq_len: int = 128,
    mapping: Mapping[int, int] = WRONG_LAYER_MAPPING,
    control_seed: int = CONTROL_SEED,
    store: CorrectedControlStore | None = None,
    stage: str = "corrected_readout",
    manifest_checksum: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> tuple[list[dict], dict]:
    """Tie-aware rows for every (prompt, layer, variant), resumable per prompt.

    One forward pass per prompt captures every scored layer and the target
    layer at once; the five readout variants are matrix products on the captured
    residual, so the cost is forward-pass-only. There is no fitting here and no
    ``fit`` import.

    Resume granularity is the (prompt, layer) pair: a stored unit that covers
    only some layers contributes those layers and the rest are recomputed. A
    unit written under a different fingerprint or a torn unit is ignored by
    :meth:`~jlens.calibration.state.CalibrationStore.load` and recomputed, so a
    disconnect costs at most the prompts that were in flight.

    Args:
        manifest_checksum: The population manifest these rows belong to. Stored
            in every unit and required to match on resume, so a confirmation
            unit can never be reused as a development unit or vice versa. The
            prompt hash is checked as well; both have to agree.

    Returns:
        ``(rows, progress_record)``.
    """
    import torch

    from jlens.hooks import ActivationRecorder
    from jlens.mmlocalize.lens_validity import tie_aware_row

    scored = [int(layer) for layer in scoring_layers]
    controls = corrected_control_lenses(
        lens, scoring_layers=scored, mapping=mapping, seed=control_seed
    )
    record_at = sorted({*scored, int(target_layer)})

    rows: list[dict] = []
    n_reused_prompts = n_computed_prompts = n_partial_prompts = 0
    n_reused_layers = n_computed_layers = 0

    for index, prompt in enumerate(prompts):
        sha = _prompt_sha(prompt)
        key = f"prompt{index:05d}"
        stored = store.load(stage, key) if store is not None else None
        cached: dict[int, list[dict]] = {}
        if (
            stored is not None
            and stored.get("prompt_sha") == sha
            and stored.get("manifest_checksum") == manifest_checksum
        ):
            for layer_key, layer_rows in (stored.get("layers") or {}).items():
                if int(layer_key) in set(scored) and layer_rows:
                    cached[int(layer_key)] = list(layer_rows)
        needed = [layer for layer in scored if layer not in cached]
        n_reused_layers += len(cached)
        n_computed_layers += len(needed)
        if not needed:
            n_reused_prompts += 1
            for layer in scored:
                rows.extend(cached[layer])
            continue
        if cached:
            n_partial_prompts += 1
        n_computed_prompts += 1

        ids = model.encode(prompt, max_length=int(max_seq_len))
        with torch.no_grad():
            with ActivationRecorder(model.layers, at=record_at) as recorder:
                model.forward(ids)
                captured = {
                    layer: recorder.activations[layer].detach() for layer in record_at
                }
            actual = model.unembed(captured[int(target_layer)][0, -1:].float())[0]
            for layer in needed:
                hidden = captured[layer][0, -1:].float()
                readouts = {
                    "j_lens": model.unembed(lens.transport(hidden, layer))[0],
                    "logit_lens": model.unembed(hidden)[0],
                }
                for name, control in controls.items():
                    readouts[name] = model.unembed(control.transport(hidden, layer))[0]
                cached[layer] = [
                    tie_aware_row(
                        sample_index=index,
                        prompt_sha=sha,
                        layer=layer,
                        variant=name,
                        variant_logits=logits,
                        actual_logits=actual,
                    )
                    for name, logits in readouts.items()
                ]
        del captured
        if store is not None:
            store.save(
                stage,
                key,
                {
                    "correction_protocol_version": CORRECTION_PROTOCOL_VERSION,
                    "manifest_checksum": manifest_checksum,
                    "prompt_index": index,
                    "prompt_sha": sha,
                    "scoring_layers": scored,
                    "readout_implementation": READOUT_IMPLEMENTATION,
                    "layers": {str(layer): cached[layer] for layer in sorted(cached)},
                },
            )
        for layer in scored:
            rows.extend(cached[layer])
        if progress is not None and (index == 0 or (index + 1) % 25 == 0):
            progress(
                f"  readout {index + 1}/{len(prompts)} prompts  "
                f"reused={n_reused_prompts} computed={n_computed_prompts}"
            )

    record = {
        "protocol": CORRECTION_PROTOCOL_VERSION,
        "stage": stage,
        "manifest_checksum": manifest_checksum,
        "n_prompts": len(list(prompts)),
        "n_prompts_reused": n_reused_prompts,
        "n_prompts_computed": n_computed_prompts,
        "n_prompts_partially_reused": n_partial_prompts,
        "n_layer_results_reused": n_reused_layers,
        "n_layer_results_computed": n_computed_layers,
        "scoring_layers": scored,
        "resume_granularity": "one checksum-valid unit per prompt, merged per layer",
        "forward_passes": n_computed_prompts,
        "backward_passes": 0,
        "fitting_performed": False,
    }
    return rows, record


# ------------------------------------------------------------------ verdicts


def evaluate_corrected_layers(
    rows: Sequence[Mapping],
    *,
    manifest_checksum: str,
    layers: Sequence[int] = BAND_SCORING_LAYERS,
    scale: int = CORRECTED_SCALE,
    stage: str = "confirmation",
    gate: CalibrationGate = CORRECTED_GATE,
    mapping: Mapping[int, int] = WRONG_LAYER_MAPPING,
    universe: Sequence[int] = FIXED_CONTROL_UNIVERSE,
) -> dict[int, dict]:
    """The frozen gate, applied per layer, with the control provenance stamped on.

    :func:`jlens.calibration.gate.evaluate_calibration_layers` is called
    unchanged — no threshold is touched here. What this adds is the record of
    *which* population and *which* wrong-layer substitution produced each
    verdict, so :func:`corrected_band_verdict` can refuse a set of verdicts that
    did not all come from one population, and a reader never has to infer the
    control from the run it was found in.
    """
    from jlens.calibration.gate import evaluate_calibration_layers

    scored = [int(layer) for layer in layers]
    results = evaluate_calibration_layers(
        rows, layers=scored, scale=int(scale), stage=stage, gate=gate
    )
    stamped: dict[int, dict] = {}
    for layer, result in results.items():
        stamped[int(layer)] = {
            **dict(result),
            "correction_protocol_version": CORRECTION_PROTOCOL_VERSION,
            "confirmation_manifest_checksum": str(manifest_checksum),
            "fixed_control_universe": [int(value) for value in universe],
            "wrong_layer_jacobian_fitted_at": int(mapping[int(layer)]),
            "wrong_layer_mapping_rule": (
                "jlens.controls.distant_layer_mapping over the FIXED control "
                "universe"
            ),
            "is_independent_confirmation": stage == "confirmation",
        }
    return stamped


def corrected_layer_rows(
    confirmation: Mapping[int, Mapping],
    *,
    development: Mapping[int, Mapping] | None = None,
    scoring_layers: Sequence[int] = BAND_SCORING_LAYERS,
    matrix_present: Mapping[int, bool] | None = None,
    published_layers: Sequence[int] = (),
) -> list[dict]:
    """One explicit row per layer.

    The ambiguous ``publishable`` field the superseded report carried is gone.
    Four independent facts are reported instead, and a layer that failed
    confirmation is never ``publication_eligible``:

    ``matrix_artifact_exists``  a scale-250 matrix for the layer was found.
    ``confirmation_passed``     the frozen gate passed on the new population.
    ``publication_eligible``    both of the above.
    ``published``               an artifact was actually written for it.
    """
    scored = [int(layer) for layer in scoring_layers]
    present = {int(k): bool(v) for k, v in dict(matrix_present or {}).items()}
    published = {int(layer) for layer in published_layers}
    if published - set(scored):
        raise CorrectedControlRefused(
            f"layer(s) {sorted(published - set(scored))} are reported as published "
            f"but are not scored band layers {scored}"
        )
    rows = []
    for layer in scored:
        result = dict(confirmation.get(layer) or {})
        metrics = (result.get("metrics") or {}).get("j_lens", {})
        wrong = (result.get("metrics") or {}).get("wrong_layer", {})
        has_matrix = present.get(layer, layer in confirmation)
        passed = bool(result.get("passed", False))
        eligible = bool(has_matrix and passed)
        rows.append(
            {
                "layer": layer,
                "normalized_depth": normalized_depth(layer),
                "role": "band_layer",
                "matrix_artifact_exists": bool(has_matrix),
                "confirmation_passed": passed,
                "publication_eligible": eligible,
                "published": bool(layer in published),
                "confirmation_scored": layer in confirmation,
                "confirmation_failed_checks": list(result.get("failed_checks", [])),
                "confirmation_manifest_checksum": result.get(
                    "confirmation_manifest_checksum"
                ),
                "mean_reciprocal_rank": metrics.get("mean_reciprocal_rank"),
                "median_midrank": metrics.get("median_midrank"),
                "top10_inclusion": metrics.get("top10_inclusion"),
                "tied_at_max_rate": metrics.get("tied_at_max_rate"),
                "wrong_layer_mean_reciprocal_rank": wrong.get("mean_reciprocal_rank"),
                "wrong_layer_jacobian_fitted_at": WRONG_LAYER_MAPPING.get(layer),
                "development_passed": (
                    bool((development or {}).get(layer, {}).get("passed", False))
                    if development
                    else None
                ),
                "development_is_not_confirmation": True,
            }
        )
    return rows


def corrected_band_verdict(
    confirmation: Mapping[int, Mapping],
    *,
    confirmation_manifest_checksum: str,
    scale: int = CORRECTED_SCALE,
    development: Mapping[int, Mapping] | None = None,
    scoring_layers: Sequence[int] = BAND_SCORING_LAYERS,
    window: tuple[int, int] = BAND_WINDOW,
    matrix_present: Mapping[int, bool] | None = None,
    published_layers: Sequence[int] = (),
    protocol: Mapping | None = None,
) -> dict:
    """GO or NO-GO for the whole band, from one population and one manifest.

    ``GO`` requires **every** physical layer in ``window`` to have passed the
    frozen gate on the same new confirmation population under the corrected
    control. A partial pass reports the largest contiguous passing sub-band,
    computed from layer geometry alone; no causal outcome exists at this stage
    and none is consulted.

    Raises:
        CorrectedControlRefused: If a scored layer is missing a verdict, or if
            any verdict was produced under a different confirmation manifest.
            Old verdicts for 32/35/38/40 and new verdicts for the interior are
            never combined — that is precisely the comparison the superseded
            run's control made impossible.
    """
    from jlens.mmpilot.band_swap import largest_admissible_band

    scored = [int(layer) for layer in scoring_layers]
    results = {int(layer): dict(row) for layer, row in confirmation.items()}
    missing = sorted(layer for layer in scored if layer not in results)
    if missing:
        raise CorrectedControlRefused(
            f"no corrected confirmation verdict for physical layer(s) {missing}. All "
            f"{len(scored)} band layers are evaluated together on one population; a "
            "band assembled from layers judged on different data is the defect this "
            "correction repairs."
        )
    wrong_manifest = sorted(
        layer
        for layer in scored
        if str(results[layer].get("confirmation_manifest_checksum"))
        != str(confirmation_manifest_checksum)
    )
    if wrong_manifest:
        raise CorrectedControlRefused(
            f"layer(s) {wrong_manifest} carry a different confirmation manifest "
            f"checksum than the common {confirmation_manifest_checksum}. Verdicts "
            "from the earlier runs may not be combined with these: every layer must "
            "be evaluated on the same new population under the same corrected "
            "protocol."
        )

    rows = corrected_layer_rows(
        results,
        development=development,
        scoring_layers=scored,
        matrix_present=matrix_present,
        published_layers=published_layers,
    )
    passed = sorted(row["layer"] for row in rows if row["confirmation_passed"])
    failed = sorted(row["layer"] for row in rows if not row["confirmation_passed"])
    admissible = largest_admissible_band(passed, low=window[0], high=window[1])
    full_band = admissible == (int(window[0]), int(window[1]))

    payload = {
        "schema": "jlens.mmpilot.band_corrected_control_verdict.v1",
        "correction_protocol_version": CORRECTION_PROTOCOL_VERSION,
        "verdict": CORRECTED_BAND_GO if not failed else CORRECTED_BAND_NO_GO,
        "scale": int(scale),
        "protocol_digest": (protocol or {}).get("protocol_digest"),
        "confirmation_manifest_checksum": str(confirmation_manifest_checksum),
        "fixed_control_universe": list(FIXED_CONTROL_UNIVERSE),
        "wrong_layer_mapping": {
            str(k): int(v) for k, v in sorted(WRONG_LAYER_MAPPING.items())
        },
        "band_window": [int(window[0]), int(window[1])],
        "scoring_layers": scored,
        "layers_passing": passed,
        "layers_failing": failed,
        "publication_eligible_layers": sorted(
            row["layer"] for row in rows if row["publication_eligible"]
        ),
        "full_band_available": bool(full_band),
        "largest_admissible_contiguous_band": (
            None if admissible is None else list(admissible)
        ),
        "sub_band_selected_by": (
            "layer geometry only: longest contiguous run of layers that passed the "
            "corrected confirmation, ties to the shallowest start; no causal "
            "outcome is consulted and none exists at this stage"
        ),
        "all_layers_scored_on_one_population": True,
        "old_and_new_verdicts_combined": False,
        "layers": rows,
        "stage3_unblocked": bool(full_band),
        "statement": (
            f"every physical layer L{window[0]}-L{window[1]} passed the frozen gate "
            f"on the new untouched confirmation population at scale {int(scale)} "
            "under the corrected fixed-universe wrong-layer control; the contiguous "
            "band is admissible and the causal stage is unblocked"
            if not failed
            else (
                f"layer(s) {failed} did not pass the frozen gate on the new "
                f"untouched confirmation population at scale {int(scale)} under the "
                "corrected control. Every metric is reported, nothing is published "
                f"for them, the largest contiguous passing sub-band is "
                f"{None if admissible is None else list(admissible)}, and the causal "
                "band swap stays blocked."
            )
        ),
        "all_layer_results_recorded": True,
    }
    return {**payload, "verdict_checksum": payload_checksum(payload)}


def _fixed(value: object, width: int = 7) -> str:
    """Right-aligned three-decimal number, or a dash when the metric is absent."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):>{width}.3f}"
    return f"{'-':>{width}}"


def format_corrected_verdict(verdict: Mapping) -> str:
    """The verdict block the operator sends back."""
    lines = [
        "=" * 78,
        f"CORRECTED BAND VERDICT — {verdict['verdict']}",
        "=" * 78,
        f"  confirmation manifest  {verdict['confirmation_manifest_checksum']}",
        f"  passing layers         {verdict['layers_passing']}",
        f"  failing layers         {verdict['layers_failing']}",
        f"  full band available    {verdict['full_band_available']}",
        f"  largest sub-band       {verdict['largest_admissible_contiguous_band']}",
        f"  stage 3 unblocked      {verdict['stage3_unblocked']}",
        "",
        f"  {'layer':>5} {'matrix':>7} {'confirmed':>10} {'eligible':>9} "
        f"{'published':>10} {'MRR':>7} {'wrongL':>7} {'median':>7} {'top10':>6}",
    ]
    for row in verdict["layers"]:
        lines.append(
            f"  {row['layer']:>5} {str(row['matrix_artifact_exists']):>7} "
            f"{str(row['confirmation_passed']):>10} "
            f"{str(row['publication_eligible']):>9} {str(row['published']):>10} "
            f"{_fixed(row['mean_reciprocal_rank'])} "
            f"{_fixed(row['wrong_layer_mean_reciprocal_rank'])} "
            f"{_fixed(row['median_midrank'])} {_fixed(row['top10_inclusion'], 6)}"
        )
    lines += ["", "  " + str(verdict["statement"])]
    return "\n".join(lines)


# ---------------------------------------------------------------- publication


def publish_corrected_layer(
    *,
    layer: int,
    scale: int,
    lens,
    destination: Path | str,
    confirmation_verdict: Mapping,
    development_verdict: Mapping | None,
    universe: ControlUniverse,
    protocol: Mapping,
    confirmation_manifest: Mapping,
    corrected_dir: Path | str,
    superseded_immutability: Mapping,
    scoring_layers: Sequence[int] = BAND_SCORING_LAYERS,
    protected_run_dirs: Sequence[Path | str] = (),
    gate: CalibrationGate = CORRECTED_GATE,
) -> dict:
    """Write one corrected artifact for a layer that passed the new confirmation.

    Every refusal is explicit and none of them can be argued around:

    * a layer outside ``scoring_layers``;
    * a layer whose confirmation did not pass — publication eligibility is not
      a judgement call;
    * a verdict produced under a different confirmation manifest;
    * a destination outside the new versioned corrected-validation directory;
    * a destination inside any protected run (the parent calibration, the
      extension, the superseded band-interior run);
    * a destination that already exists — existing published artifacts are never
      overwritten.

    ``lens`` is the single-layer :class:`~jlens.lens.JacobianLens` for ``layer``,
    carrying the matrix read from the source snapshot and nothing else.
    """
    destination = Path(destination)
    corrected_root = Path(corrected_dir).resolve()
    resolved = destination.resolve()

    if int(layer) not in {int(value) for value in scoring_layers}:
        raise CorrectedControlRefused(
            f"layer {layer} is not one of the corrected band layers "
            f"{sorted(int(v) for v in scoring_layers)}"
        )
    if not bool(confirmation_verdict.get("passed", False)):
        raise CorrectedControlRefused(
            f"layer {layer} did not pass the corrected confirmation "
            f"(failed checks {list(confirmation_verdict.get('failed_checks', []))}); "
            "a layer that failed confirmation is never publication eligible"
        )
    if str(confirmation_verdict.get("confirmation_manifest_checksum")) != str(
        confirmation_manifest.get("manifest_checksum")
    ):
        raise CorrectedControlRefused(
            f"the layer {layer} verdict was produced under confirmation manifest "
            f"{confirmation_verdict.get('confirmation_manifest_checksum')!r}, not the "
            f"population's {confirmation_manifest.get('manifest_checksum')!r}"
        )
    for protected in protected_run_dirs:
        protected_root = Path(protected).resolve()
        if protected_root == resolved or protected_root in resolved.parents:
            raise CorrectedControlRefused(
                f"{destination} is inside {protected_root}, which is completed-run "
                "evidence and is never written to"
            )
    if corrected_root not in resolved.parents:
        raise CorrectedControlRefused(
            f"{destination} is outside the corrected-validation directory "
            f"{corrected_root}. Corrected artifacts are written only there, and "
            "never over an existing published artifact."
        )
    if destination.exists():
        raise CorrectedControlRefused(
            f"{destination} already exists. A corrected artifact is written once "
            "into a new versioned directory; nothing is overwritten."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    lens.save(str(destination))
    checksum = file_sha256(str(destination))

    artifact = {
        "schema": CORRECTED_ARTIFACT_SCHEMA,
        "correction_protocol_version": CORRECTION_PROTOCOL_VERSION,
        "frozen": True,
        "validated": True,
        "publication_status": CORRECTED_PUBLISHED_STATUS,
        "physical_layer": int(layer),
        "normalized_depth": normalized_depth(int(layer)),
        "scale_point": int(scale),
        "band_window": list(BAND_WINDOW),
        # --- the file
        "lens_path": str(destination),
        "lens_checksum": checksum,
        "matrix_checksum": universe.matrix_checksums.get(int(layer)),
        "source_snapshot": universe.source_of_layer.get(int(layer)),
        "source_snapshot_checksums": protocol.get("source_snapshot_checksums"),
        # --- the corrected control protocol
        "corrected_control_protocol": {
            "version": CORRECTION_PROTOCOL_VERSION,
            "rule": universe.evidence["wrong_layer_mapping_rule"],
            "why": (
                "the superseded control computed distant_layer_mapping over the "
                "newly fitted subset, so its difficulty depended on which layers "
                "were fitted together"
            ),
        },
        "fixed_control_universe": list(universe.layers),
        "wrong_layer_mapping": {
            str(k): int(v) for k, v in sorted(universe.mapping.items())
        },
        "universe_checksum": universe.digest,
        "protocol_digest": protocol.get("protocol_digest"),
        # --- how it was judged
        "gate_version": gate.version,
        "gate_digest": gate.digest,
        "development_verdict": {
            "passed": bool((development_verdict or {}).get("passed", False)),
            "failed_checks": list((development_verdict or {}).get("failed_checks", [])),
            "metrics": (development_verdict or {}).get("metrics"),
            "is_not_independent_confirmation": True,
            "why": (
                "development diagnostics rescore records opened by earlier runs "
                "under the repaired control; they are development, never "
                "confirmation"
            ),
        },
        "confirmation_verdict": {
            "passed": True,
            "failed_checks": list(confirmation_verdict.get("failed_checks", [])),
            "metrics": confirmation_verdict.get("metrics"),
            "target_diversity": confirmation_verdict.get("target_diversity"),
        },
        "confirmation_manifest_checksum": confirmation_manifest.get("manifest_checksum"),
        "confirmation_population_record_ids": confirmation_manifest.get("record_ids"),
        # --- provenance
        "no_lens_was_refitted": True,
        "no_threshold_was_changed": True,
        "superseded_run": protocol.get("superseded_run"),
        "superseded_run_immutability_checksum": superseded_immutability.get(
            "immutability_checksum"
        ),
        "existing_publications_unchanged": True,
        "calibration_modality": "text-only",
        "spokencoco_used": False,
        "multimodal_data_used": False,
    }
    artifact["artifact_checksum"] = payload_checksum(artifact)

    sidecar = destination.with_suffix(".corrected.json")
    temporary = sidecar.with_name(f"{sidecar.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, sidecar)
    return artifact


# -------------------------------------------------------------------- budget


def corrected_readout_budget(
    *,
    n_confirmation: int | None = None,
    n_development: int = 0,
    scoring_layers: Sequence[int] = BAND_SCORING_LAYERS,
    universe_layers: Sequence[int] = FIXED_CONTROL_UNIVERSE,
    d_model: int = 2560,
    seconds_per_prompt_low: float = 1.0,
    seconds_per_prompt_high: float = 3.0,
) -> dict:
    """What the corrected confirmation costs. Forward passes only.

    There is no fitting term because there is no fitting: the scale-250 matrices
    are read from existing snapshots. Each prompt costs **two** forward passes —
    one for target-token discovery through the model's ordinary output path, and
    one for the readout, which then does five ``[d_model, d_model]`` products per
    scored layer on the captured residual.

    The estimate covers the scored work only. Loading the model and rebuilding
    the corpus ordering are additional, and the notebook says so where it
    matters.
    """
    confirmation = int(CORRECTED_GATE.n_prompts if n_confirmation is None else n_confirmation)
    prompts = confirmation + int(n_development)
    n_scored = len(list(scoring_layers))
    matrix_bytes = int(d_model) * int(d_model) * 4
    forwards = 2 * prompts
    payload = {
        "protocol": CORRECTION_PROTOCOL_VERSION,
        "workload": "forward-pass only; no fitting, no backward passes",
        "n_confirmation_prompts": confirmation,
        "n_development_prompts": int(n_development),
        "n_prompts_total": prompts,
        "scored_layers": list(int(layer) for layer in scoring_layers),
        "forward_passes": forwards,
        "forward_passes_target_discovery": prompts,
        "forward_passes_readout": prompts,
        "backward_passes": 0,
        "fitting_performed": False,
        "readout_products_per_prompt": 5 * n_scored,
        "excludes": (
            "model load and the corpus reconstruction that rebuilds the fit "
            "ordering and the previously opened sets"
        ),
        "l4_minutes_low": round(forwards * float(seconds_per_prompt_low) / 60.0, 1),
        "l4_minutes_high": round(forwards * float(seconds_per_prompt_high) / 60.0, 1),
        "storage_bytes": {
            "universe_matrices_resident": len(list(universe_layers)) * matrix_bytes,
            "per_prompt_units": prompts * 48 * 1024,
            "published_max": n_scored * matrix_bytes // 2,
            "reports": 24 * 1024 * 1024,
        },
        "checkpoint_granularity": (
            "one atomically written, checksum-valid unit per prompt, merged per "
            "layer on resume"
        ),
        "extrapolation_not_measurement": True,
    }
    payload["storage_bytes"]["total"] = sum(
        value for key, value in payload["storage_bytes"].items() if key != "total"
    )
    payload["budget_checksum"] = payload_checksum(payload)
    return payload


def format_corrected_readout_budget(budget: Mapping) -> str:
    """The block printed before any model is loaded."""
    storage = budget["storage_bytes"]
    return "\n".join(
        [
            "CORRECTED CONFIRMATION BUDGET — forward passes only, no fitting",
            f"  prompts            {budget['n_prompts_total']} "
            f"({budget['n_confirmation_prompts']} confirmation + "
            f"{budget['n_development_prompts']} development)",
            f"  scored layers      {budget['scored_layers']}",
            f"  forward passes     {budget['forward_passes']} "
            f"({budget['forward_passes_target_discovery']} target discovery + "
            f"{budget['forward_passes_readout']} readout)",
            f"  backward passes    {budget['backward_passes']}  "
            f"(fitting_performed={budget['fitting_performed']})",
            f"  readout products   {budget['readout_products_per_prompt']} per prompt",
            f"  L4 wall time       {budget['l4_minutes_low']:.0f}-"
            f"{budget['l4_minutes_high']:.0f} min for the scored work",
            f"  excludes           {budget['excludes']}",
            f"  Drive              {storage['total'] / 2**20:.0f} MiB",
            f"  checkpoint         {budget['checkpoint_granularity']}",
            "",
            "  EXTRAPOLATION, NOT MEASUREMENT.",
        ]
    )


# --------------------------------------------------------------- the report


def corrected_validation_report(
    *,
    mode: str,
    protocol: Mapping,
    universe: ControlUniverse,
    confirmation_manifest: Mapping,
    development: Mapping[int, Mapping] | None,
    confirmation: Mapping[int, Mapping],
    verdict: Mapping,
    publication: Mapping,
    superseded_immutability: Mapping,
    leakage_audit: Mapping | None = None,
    opened_record_audit: Mapping | None = None,
    readout_record: Mapping | None = None,
    budget: Mapping | None = None,
    resume: Mapping | None = None,
) -> dict:
    """The corrected report, including everything it must say about the first one.

    The provenance block is not decoration. It is the reason a reader can trust
    that the failed result was superseded on a methodological ground rather than
    replaced because someone did not like it.
    """
    payload = {
        "schema": CORRECTED_REPORT_SCHEMA,
        "mode": str(mode),
        "correction_protocol_version": CORRECTION_PROTOCOL_VERSION,
        "protocol": dict(protocol),
        "control_universe": dict(universe.evidence),
        "fixed_control_universe": list(universe.layers),
        "wrong_layer_mapping": {
            str(k): int(v) for k, v in sorted(universe.mapping.items())
        },
        "superseded_wrong_layer_mapping": {
            str(k): int(v) for k, v in sorted(SUPERSEDED_WRONG_LAYER_MAPPING.items())
        },
        "confirmation_manifest": dict(confirmation_manifest),
        "confirmation_manifest_checksum": confirmation_manifest.get("manifest_checksum"),
        "fresh_split_leakage_audit": dict(leakage_audit or {}),
        "opened_record_audit": dict(opened_record_audit or {}),
        "readout": dict(readout_record or {}),
        "development": {str(k): v for k, v in dict(development or {}).items()},
        "development_verdict_is_not_confirmation": True,
        "confirmation": {str(k): v for k, v in dict(confirmation).items()},
        "band_verdict": dict(verdict),
        "publication": dict(publication),
        "budget": dict(budget or {}),
        "resume": dict(resume or {}),
        "superseded_run_immutability": dict(superseded_immutability),
        "provenance": {
            "original_result": SUPERSEDED_VERDICT,
            "original_run": SUPERSEDED_RUN_NAME,
            "original_result_status": (
                "immutable historical result; a superseded set-dependent-control "
                "validation, not a fraudulent or erased one"
            ),
            "why_superseded_for_band_admissibility": (
                "its wrong-layer control was built with "
                "distant_layer_mapping over the newly fitted subset "
                f"{list(BAND_INTERIOR_LAYERS)}, so every substituted Jacobian was "
                "another nearby late-workspace lens and the +0.15 margin was a "
                "different test than the one the scale-250 study's late layers "
                "faced over its broad grid. The gate's difficulty therefore "
                "depended on which layers happened to be fitted together, which "
                "makes the two validations incomparable"
            ),
            "no_frozen_numerical_threshold_was_changed": True,
            "gate_digest": protocol.get("gate_digest"),
            "no_matrix_was_refitted": True,
            "matrices_reused_from": protocol.get("source_snapshot_checksums"),
            "new_confirmation_population_uninspected_when_protocol_frozen": True,
            "previously_opened_records_are_development_only": True,
            "stage3_blocked_unless_all_nine_layers_pass": True,
            "stage3_unblocked": bool(verdict.get("full_band_available", False)),
            "old_and_new_verdicts_combined": False,
        },
        "mock_proves_pipeline_only": str(mode) != "real",
    }
    payload["report_checksum"] = payload_checksum(payload)
    return payload
