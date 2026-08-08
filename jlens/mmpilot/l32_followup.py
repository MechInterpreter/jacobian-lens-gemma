# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""The layer-32 confirmatory follow-up: discovery, comparability, verdicts.

The research-grade early-layer extension confirmed **physical layer 32** at
scale 250 on its own untouched 256-prompt confirmation set
(``EARLY_LAYER_CALIBRATION_GO``). This module is what a study needs in order to
*use* that artifact, and nothing more. It fits nothing, refits nothing, and
tests no adjacent layer.

Four things live here, each because doing it inline in a notebook is how it
would go wrong:

**Discovery, not assumption.** The published filename is never typed into a
configuration. :func:`discover_published_l32_lens` starts from the extension
run's own report, reads the publication block's checksum for layer 32, scans the
run's ``artifacts/published`` directory for the ``.extension.json`` sidecar that
claims that layer and scale, takes the path *from the sidecar*, and requires the
file's own digest to equal both the sidecar's and the report's. A filename that
matches but a checksum that does not is a refusal, and so is a directory
offering two candidates.

**Scale is part of "confirmed".** Layer 32 failed at scale 100 and passed at
scale 250. :func:`l32_expectations` builds the
:class:`~jlens.mmpilot.published_lens.PublishedLensExpectations` that says so,
so every schema, checksum, geometry and calibration-modality clause in
:func:`~jlens.mmpilot.published_lens.validate_published_artifact` runs unchanged
against the right scale. :data:`SELECTED_SCALE` is bound into the fingerprint;
an artifact fitted on any other prompt count is refused.

**Comparability, decided before any pass is spent.** The completed
three-modality study ran under ``gemma-it-chat-balanced-options-v1``, whose
question lists all six candidates. This follow-up runs under an *open* protocol.
Those are different measurements. :func:`prompt_protocol_comparability` reads
the completed run's own artifacts, says whether the protocols match, and
:func:`assert_paired_reference_available` refuses to produce a cross-layer
comparison when they do not and no paired L35 reference was run.

**Verdicts A–E**, with E kept deliberately narrow: it is a statement about layer
32 alone, and it never needs the L35 reference. The reference exists so a
*comparison* can be made, not so a conclusion about layer 32 can be reached.

The intervention this study runs
================================

Source-derived J-space **causal steering** — :mod:`jlens.mmpilot.causal`,
``v = V ReLU(mean_positive_code - mean_negative_code)`` estimated from source
modality training examples only, applied as ``h' = h +- alpha * v_hat * ||h||``
at the final prompt token. This is the method behind the completed L35 result
and it is what replicates here.

It is **not** the Anthropic two-coordinate swap
(:mod:`jlens.mmpilot.coordinate_swap`). That intervention measures its
coefficient from ``h`` via ``c = pinv(V) h`` and requires a *contiguous* band of
confirmed layers, which the confirmed set (32, and separately 35/38/40) does not
provide. Nothing here renames one as the other, and
:data:`INTERVENTION_FAMILY` is written into every artifact so a swap run can
never resume from a steering run's directory.
"""

from __future__ import annotations

import json
import os
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from jlens.metadata import file_sha256
from jlens.mmpilot.convergence import (
    AMBIGUOUS,
    CONVERGED,
    NOT_CONVERGED,
)
from jlens.mmpilot.published_lens import (
    PublishedLensExpectations,
    PublishedLensSpec,
    read_artifact_sidecar,
    validate_published_artifact,
)
from jlens.mmpilot.store import payload_checksum

# ------------------------------------------------------------------ versions

#: Bound into the run fingerprint. A change refuses to resume anything produced
#: under the older rule.
L32_FOLLOWUP_PROTOCOL = "mmpilot.l32_open_prompt_followup.v1"

#: How the published artifact is located. Versioned separately from the study,
#: because a change in *how the file was found* invalidates the finding even
#: when nothing about the experiment moved.
ARTIFACT_DISCOVERY_VERSION = "mmpilot.l32_extension_artifact_discovery.v1"

#: The one word that keeps steering and coordinate swapping apart on disk. No
#: steering run has ever written any other value, and the coordinate-swap module
#: writes its own; a directory carrying one refuses a run of the other.
INTERVENTION_FAMILY = "source_derived_jspace_steering"

#: The extension report schema this module reads.
EXTENSION_REPORT_SCHEMA = "jlens.calibration.early_layer_extension_report.v1"

#: The extension publication sidecar schema.
EXTENSION_ARTIFACT_SCHEMA = "jlens.calibration.early_layer_artifact.v1"

#: The report file the extension run writes.
EXTENSION_REPORT_NAME = "early_layer_extension_report.json"

#: Where the extension store publishes.
PUBLISHED_SUBDIR = ("artifacts", "published")

#: The verdict string the extension report must carry.
EARLY_LAYER_GO = "EARLY_LAYER_CALIBRATION_GO"

#: The publication status the extension sidecar must carry.
PUBLISHED_STATUS = "PUBLISHED_VALIDATED_EARLY_LAYER"

# --------------------------------------------------------------- the numbers

#: The layer this study is about. Not configurable: a "layer 32 follow-up" that
#: quietly ran layer 33 would be undetectable in the report.
L32_LAYER = 32

#: The scale the extension selected and confirmed layer 32 at.
SELECTED_SCALE = 250

#: The size of the untouched confirmation set that cleared it.
CONFIRMATION_SET_SIZE = 256

#: The layer the completed three-modality causal result belongs to.
REFERENCE_LAYER = 35

#: The completed study's prompt protocol. Candidates were in the question.
COMPLETED_STUDY_PROMPT_PROTOCOL = "gemma-it-chat-balanced-options-v1"

#: What this study asks under. Domain-neutral, because the SpokenCOCO concept
#: set is mixed (``toilet`` and ``microwave`` are not animals) and the
#: animal-domain protocol refuses a mixed set outright.
OPEN_PROMPT_PROTOCOL = "mmpilot.open_entity_identification.v1"

#: How the phrase must be written, everywhere, in every artifact.
CONVERGENCE_PHRASE = "before native direct-readout convergence"

#: Phrases that must never appear in a report this module produced.
FORBIDDEN_PHRASES: tuple[str, ...] = (
    "pre-linguistic",
    "prelinguistic",
    "language-free",
    "language free",
    "before language exists",
)

# ------------------------------------------------------------------ verdicts

VERDICT_A = "L32_LENS_INTEGRITY"
VERDICT_B = "L32_REPRESENTATIONAL_TRANSFER"
VERDICT_C = "L32_CAUSAL_TRANSFER"
VERDICT_D = "L32_NATIVE_OUTPUT_CONVERGENCE"
VERDICT_E = "PRE_CONVERGENCE_CAUSAL_TRANSFER"

INTEGRITY_PASSED = "PASSED"
INTEGRITY_REFUSED = "REFUSED"

SUPPORTED = "SUPPORTED"
NOT_SUPPORTED = "NOT_SUPPORTED"
INCONCLUSIVE = "INCONCLUSIVE"
TRANSFER_AT_OR_AFTER_CONVERGENCE = "TRANSFER_AT_OR_AFTER_CONVERGENCE"

#: How many distinct photographs/recordings an effect must appear on.
MIN_DISTINCT_UNITS = 2


class L32FollowupRefused(RuntimeError):
    """A precondition of the follow-up does not hold, so nothing is measured."""


class ArtifactDiscoveryRefused(L32FollowupRefused):
    """The published layer-32 artifact could not be resolved and verified."""

    def __init__(self, message: str, *, problems: Sequence[str] = (), evidence: Mapping | None = None) -> None:
        super().__init__(message)
        self.problems = list(problems)
        self.evidence = dict(evidence or {})


class PairedReferenceRequired(L32FollowupRefused):
    """A cross-layer comparison was asked for that the protocols do not allow."""


# ---------------------------------------------------------------- expectations


def l32_expectations(
    *,
    model_repo_id: str,
    model_revision: str,
    tokenizer_revision: str | None = None,
    d_model: int = 2560,
    hook_site: str = "block_output",
    layer: int = L32_LAYER,
    scale: int = SELECTED_SCALE,
) -> PublishedLensExpectations:
    """What the scale-250 layer-32 artifact is held to.

    The scale-100 module defaults say layer 32 *failed*, which was true of that
    run and is not true of this artifact. So the expectations are stated
    explicitly here: scale 250, layer 32 confirmed, and an empty
    failed-confirmation list — because at scale 250 no layer this study loads is
    on one. Everything else (schema completeness, checksum agreement, frozen and
    validated flags, text-only calibration, geometry) is the same clause set.

    Args:
        layer / scale: The MOCK world's decoder has six blocks and its extension
            stand-in is published at another layer number, exactly as the MOCK
            three-modality run uses layers 2/3/4 for 35/38/40. They default to
            the real values, and the real path never overrides them — the
            notebook asserts that.
    """
    return PublishedLensExpectations(
        model_repo_id=str(model_repo_id),
        model_revision=str(model_revision),
        tokenizer_revision=tokenizer_revision,
        scale_point=int(scale),
        d_model=int(d_model),
        calibration_modality="text-only",
        hook_site=str(hook_site),
        confirmed_layers=(int(layer),),
        failed_confirmation_layers=(),
    )


# ------------------------------------------------------------------ discovery


@dataclass(frozen=True)
class DiscoveredLens:
    """One published artifact, located from metadata rather than named."""

    layer: int
    scale: int
    lens_path: str
    lens_checksum: str
    base_sidecar_path: str
    extension_sidecar_path: str
    report_path: str
    report_checksum: str
    extension_artifact_checksum: str
    discovery_evidence: dict = field(default_factory=dict)

    def spec(self) -> PublishedLensSpec:
        return PublishedLensSpec(
            layer=int(self.layer),
            path=str(self.lens_path),
            expect_sha256=str(self.lens_checksum),
        )

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["discovery_version"] = ARTIFACT_DISCOVERY_VERSION
        return payload


def _read_json(path: Path, *, what: str) -> dict:
    if not path.is_file():
        raise ArtifactDiscoveryRefused(
            f"{what} not found at {path}", problems=[f"{what}_present"]
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ArtifactDiscoveryRefused(
            f"{what} at {path} is not readable JSON: {error}",
            problems=[f"{what}_readable"],
        ) from error
    if not isinstance(payload, dict):
        raise ArtifactDiscoveryRefused(
            f"{what} at {path} does not hold a JSON object",
            problems=[f"{what}_readable"],
        )
    return payload


def read_extension_report(extension_run_dir: str | os.PathLike[str]) -> tuple[Path, dict]:
    """The extension run's own report, checked for the schema this module reads."""
    path = Path(extension_run_dir) / "artifacts" / EXTENSION_REPORT_NAME
    report = _read_json(path, what="extension_report")
    if report.get("schema") != EXTENSION_REPORT_SCHEMA:
        raise ArtifactDiscoveryRefused(
            f"{path} declares schema {report.get('schema')!r}, not "
            f"{EXTENSION_REPORT_SCHEMA!r}; this module has never seen that format "
            "and will not guess at its fields",
            problems=["report_schema"],
        )
    return path, report


def discover_published_l32_lens(
    extension_run_dir: str | os.PathLike[str],
    *,
    layer: int = L32_LAYER,
    expected_scale: int = SELECTED_SCALE,
    require_real_mode: bool = True,
) -> DiscoveredLens:
    """Resolve the published artifact for ``layer`` from the run's own metadata.

    The filename is never assumed. The chain is:

    1. the extension report's ``early_layer_verdict`` is
       ``EARLY_LAYER_CALIBRATION_GO`` and its ``scale_selection`` selected
       ``expected_scale``;
    2. its ``publication`` block lists ``layer`` among ``published_layers`` and
       carries a checksum for it;
    3. exactly one ``*.extension.json`` sidecar in ``artifacts/published``
       claims that layer, that scale, ``validated`` and the published status;
    4. the lens path comes from **that sidecar**, resolved inside the published
       directory so a report copied between mounts still works;
    5. the file's own sha256 equals the sidecar's ``lens_checksum`` *and* the
       report's ``published_checksums`` entry;
    6. the sidecar's ``artifact_checksum`` recomputes over its own body.

    Raises:
        ArtifactDiscoveryRefused: On any broken link, listing every problem
            found rather than the first. A directory offering two candidate
            sidecars for one layer is refused rather than resolved by sorting.
    """
    run_dir = Path(extension_run_dir)
    report_path, report = read_extension_report(run_dir)
    problems: list[str] = []
    evidence: dict = {
        "discovery_version": ARTIFACT_DISCOVERY_VERSION,
        "extension_run_dir": str(run_dir),
        "report_path": str(report_path),
        "requested_layer": int(layer),
        "requested_scale": int(expected_scale),
    }

    mode = report.get("mode")
    evidence["report_mode"] = mode
    if require_real_mode and mode != "real":
        problems.append(
            f"report mode is {mode!r}, not 'real'; a MOCK report proves pipeline "
            "behaviour and publishes nothing about Gemma"
        )

    verdict = (report.get("early_layer_verdict") or {}).get("verdict")
    evidence["early_layer_verdict"] = verdict
    if verdict != EARLY_LAYER_GO:
        problems.append(f"early_layer_verdict is {verdict!r}, not {EARLY_LAYER_GO!r}")

    selection = report.get("scale_selection") or {}
    selected_scale = selection.get("selected_scale")
    evidence["selected_scale"] = selected_scale
    evidence["selection_checksum"] = selection.get("selection_checksum")
    if selected_scale is None or int(selected_scale) != int(expected_scale):
        problems.append(
            f"the run selected scale {selected_scale!r}, this study requires "
            f"{expected_scale}"
        )

    publication = report.get("publication") or {}
    published_layers = [int(value) for value in publication.get("published_layers", [])]
    checksums = {str(k): str(v) for k, v in (publication.get("published_checksums") or {}).items()}
    evidence["published_layers"] = published_layers
    evidence["report_checksum_for_layer"] = checksums.get(str(int(layer)))
    if int(layer) not in published_layers:
        problems.append(
            f"the report publishes layers {published_layers}; layer {layer} is not "
            "among them"
        )
    if str(int(layer)) not in checksums:
        problems.append(f"the report carries no published checksum for layer {layer}")

    resume = report.get("resume") or {}
    evidence["invalid_units"] = list(resume.get("invalid_units", []))
    if evidence["invalid_units"]:
        problems.append(
            f"the extension run recorded {len(evidence['invalid_units'])} invalid "
            "unit(s); its publication rests on a torn store"
        )

    immutability = report.get("parent_immutability_proof") or {}
    evidence["parent_immutable"] = immutability.get("immutable")
    if immutability.get("immutable") is not True:
        problems.append(
            "the extension run did not prove its parent run immutable "
            f"(immutable={immutability.get('immutable')!r})"
        )

    if problems:
        raise ArtifactDiscoveryRefused(
            "the extension report does not support loading a published layer "
            f"{layer} lens:\n  - " + "\n  - ".join(problems),
            problems=problems,
            evidence=evidence,
        )

    published_dir = run_dir.joinpath(*PUBLISHED_SUBDIR)
    if not published_dir.is_dir():
        raise ArtifactDiscoveryRefused(
            f"the report publishes layer {layer} but {published_dir} does not exist",
            problems=["published_dir_present"],
            evidence=evidence,
        )

    candidates: list[tuple[Path, dict]] = []
    for sidecar in sorted(published_dir.glob("*.extension.json")):
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        if (
            payload.get("schema") == EXTENSION_ARTIFACT_SCHEMA
            and int(payload.get("physical_layer", -1)) == int(layer)
            and int(payload.get("scale_point", -1)) == int(expected_scale)
            and payload.get("validated") is True
            and payload.get("publication_status") == PUBLISHED_STATUS
        ):
            candidates.append((sidecar, payload))

    evidence["extension_sidecars_scanned"] = [
        path.name for path in sorted(published_dir.glob("*.extension.json"))
    ]
    evidence["matching_sidecars"] = [path.name for path, _ in candidates]

    if not candidates:
        raise ArtifactDiscoveryRefused(
            f"no published extension sidecar in {published_dir} claims layer "
            f"{layer} at scale {expected_scale} with validated=true and status "
            f"{PUBLISHED_STATUS!r}. Scanned: "
            f"{evidence['extension_sidecars_scanned']}",
            problems=["extension_sidecar_found"],
            evidence=evidence,
        )
    if len(candidates) > 1:
        raise ArtifactDiscoveryRefused(
            f"{len(candidates)} published sidecars in {published_dir} claim layer "
            f"{layer} at scale {expected_scale}: {evidence['matching_sidecars']}. "
            "Which artifact a result rests on is not decided by sort order.",
            problems=["extension_sidecar_unique"],
            evidence=evidence,
        )

    sidecar_path, artifact = candidates[0]

    recomputed = payload_checksum(
        {key: value for key, value in artifact.items() if key != "artifact_checksum"}
    )
    if recomputed != artifact.get("artifact_checksum"):
        raise ArtifactDiscoveryRefused(
            f"{sidecar_path} does not match its own artifact_checksum "
            f"(recomputed {recomputed}, recorded {artifact.get('artifact_checksum')})",
            problems=["extension_sidecar_intact"],
            evidence=evidence,
        )

    # The recorded path names the file; the *directory* is this run's published
    # directory, because a run copied to another Drive mount keeps its bytes and
    # loses its absolute paths.
    recorded_name = Path(str(artifact.get("lens_path", ""))).name
    if not recorded_name:
        raise ArtifactDiscoveryRefused(
            f"{sidecar_path} records no lens_path",
            problems=["extension_sidecar_names_lens"],
            evidence=evidence,
        )
    lens_path = published_dir / recorded_name
    evidence["recorded_lens_path"] = str(artifact.get("lens_path"))
    evidence["resolved_lens_path"] = str(lens_path)
    if not lens_path.is_file():
        raise ArtifactDiscoveryRefused(
            f"{sidecar_path} names {recorded_name}, which is not present in "
            f"{published_dir}",
            problems=["lens_file_exists"],
            evidence=evidence,
        )

    actual = file_sha256(str(lens_path))
    evidence["file_checksum"] = actual
    mismatches: list[str] = []
    if actual != artifact.get("lens_checksum"):
        mismatches.append(
            f"sidecar says {artifact.get('lens_checksum')}, file is {actual}"
        )
    if actual != checksums[str(int(layer))]:
        mismatches.append(
            f"report says {checksums[str(int(layer))]}, file is {actual}"
        )
    if mismatches:
        raise ArtifactDiscoveryRefused(
            f"the layer {layer} lens at {lens_path} is not the published file:\n  - "
            + "\n  - ".join(mismatches),
            problems=["lens_checksum_agreement"],
            evidence=evidence,
        )

    base_sidecar, _ = read_artifact_sidecar(lens_path)

    return DiscoveredLens(
        layer=int(layer),
        scale=int(expected_scale),
        lens_path=str(lens_path),
        lens_checksum=actual,
        base_sidecar_path=str(base_sidecar),
        extension_sidecar_path=str(sidecar_path),
        report_path=str(report_path),
        report_checksum=file_sha256(str(report_path)),
        extension_artifact_checksum=str(artifact.get("artifact_checksum")),
        discovery_evidence=evidence,
    )


def validate_discovered_lens(
    discovered: DiscoveredLens, expectations: PublishedLensExpectations
) -> dict:
    """Run every published-artifact clause against the discovered file.

    Discovery proves the *file is the published one*; this proves the published
    one is *usable by this study*. Both are needed: a correctly-located artifact
    calibrated on multimodal data would pass the first and fail the second.

    Raises:
        PublishedLensRefused: With every failed clause listed.
    """
    spec = discovered.spec()
    sidecar_path, artifact = read_artifact_sidecar(Path(discovered.lens_path))
    record = validate_published_artifact(
        spec,
        expectations,
        artifact=artifact,
        actual_checksum=discovered.lens_checksum,
        sidecar_path=sidecar_path,
    )
    record["discovery"] = discovered.to_dict()
    record["confirmation_set_size_expected"] = CONFIRMATION_SET_SIZE
    return record


# -------------------------------------------------------------- comparability


def prompt_protocol_comparability(
    completed_summary: Mapping,
    *,
    new_protocol: str = OPEN_PROMPT_PROTOCOL,
) -> dict:
    """Did the completed study ask the same question this one does?

    Reads the completed run's own recorded protocol rather than trusting the
    constant in :mod:`jlens.mmpilot.capability`, because the constant describes
    the code as it stands and the artifact describes the run as it happened.

    ``paired_reference_required`` is True whenever the protocols differ. That is
    a statement about *comparison*, not about layer 32: the layer-32 verdicts
    stand on their own measurements either way.
    """
    recorded = (
        _dig(completed_summary, "selection_fingerprint", "capability_protocol")
        or _dig(completed_summary, "fingerprint", "selection_config", "capability_protocol")
        or _dig(completed_summary, "prompt_protocol")
    )
    matches = recorded is not None and str(recorded) == str(new_protocol)
    return {
        "rule_version": L32_FOLLOWUP_PROTOCOL,
        "completed_run_protocol": recorded,
        "completed_run_protocol_known": recorded is not None,
        "new_protocol": str(new_protocol),
        "protocols_match": bool(matches),
        "paired_reference_required": not matches,
        "reusable_without_rerun": bool(matches),
        "statement": (
            "the completed study and this one asked the same question, so its "
            f"layer {REFERENCE_LAYER} results may be reused after checksum and "
            "fingerprint verification"
            if matches
            else (
                f"the completed study asked under {recorded!r} (candidates listed "
                f"in the prompt) and this study asks under {new_protocol!r} "
                "(candidates scored off-prompt). These are different "
                "measurements. The completed numbers are read-only historical "
                f"context; a numerical layer 32 vs layer {REFERENCE_LAYER} "
                "comparison requires a paired reference condition run under this "
                "protocol on identical samples, targets, controls and scoring."
            )
        ),
    }


def _dig(payload: Mapping, *path: str):
    current: object = payload
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def assert_paired_reference_available(
    comparability: Mapping, *, paired_reference_ran: bool
) -> None:
    """Refuse a cross-layer comparison the prompt protocols do not license.

    Raises:
        PairedReferenceRequired: When the protocols differ and no paired
            reference was run under the new one.
    """
    if comparability.get("paired_reference_required") and not paired_reference_ran:
        raise PairedReferenceRequired(
            "refusing to compare layer 32 open-prompt results against the "
            f"completed layer {REFERENCE_LAYER} results: "
            f"{comparability.get('statement')}\n"
            "Set RUN_PAIRED_L35_REFERENCE (and its budget confirmation) to "
            "measure the reference under this protocol, or report the layer 32 "
            "results without a cross-layer claim."
        )


def separate_measured_from_historical(
    *, measured: Mapping, historical: Mapping, comparability: Mapping
) -> dict:
    """Keep the two apart in one object, labelled, never merged into a table."""
    return {
        "rule_version": L32_FOLLOWUP_PROTOCOL,
        "newly_measured": dict(measured),
        "read_only_historical_context": dict(historical),
        "comparability": dict(comparability),
        "may_be_compared_numerically": bool(comparability.get("protocols_match")),
        "note": (
            "Units generated under different prompt protocols are not mixed. "
            "The historical block is context; it is never pooled with, "
            "differenced against, or ranked beside the measured block unless "
            "may_be_compared_numerically is true."
        ),
    }


# ------------------------------------------------------------------- budgets


#: The band to quote when nothing better is available. The lower end is a fast
#: text-only pass on an L4; the upper end is a spoken-audio pass through the
#: audio tower on a shared runtime.
FALLBACK_SECONDS_PER_PASS = (0.4, 1.8)


def derive_seconds_per_pass(
    run_dir: str | os.PathLike[str],
    *,
    stages: Sequence[str] = ("capability", "activation", "intervention"),
    max_gap_seconds: float = 120.0,
    min_samples: int = 20,
) -> dict:
    """Seconds per model pass, measured from a completed run's own unit files.

    There is no explicit timing field in any artifact, so the measurement used
    is the one the completed run actually left behind: each unit file's
    modification time. Consecutive writes within one stage give inter-arrival
    gaps; the median of those gaps is one unit's wall-clock cost.

    Gaps longer than ``max_gap_seconds`` are dropped, because they are runtime
    disconnects and resume boundaries rather than compute. That makes this a
    *lower* bound on total elapsed time and a fair estimate of marginal
    per-pass cost, which is what a budget needs.

    Returns a record with ``available: False`` when the run is not reachable or
    left too few units to measure — the caller then quotes
    :data:`FALLBACK_SECONDS_PER_PASS` as a range and says so. Nothing here
    invents a number.
    """
    root = Path(run_dir)
    record: dict = {
        "source": "unit_file_mtimes",
        "run_dir": str(root),
        "stages": list(stages),
        "max_gap_seconds": float(max_gap_seconds),
        "min_samples": int(min_samples),
        "available": False,
        "caveat": (
            "derived from unit-file modification times, not from an explicit "
            "timing field: gaps above the cutoff are dropped as disconnects, so "
            "this estimates marginal per-unit cost and understates total elapsed "
            "wall clock"
        ),
    }
    if not root.is_dir():
        record["reason"] = f"{root} is not reachable from this runtime"
        record["fallback_range_seconds"] = list(FALLBACK_SECONDS_PER_PASS)
        return record

    gaps: list[float] = []
    per_stage: dict[str, int] = {}
    for stage in stages:
        stage_dir = root / "units" / stage
        if not stage_dir.is_dir():
            per_stage[stage] = 0
            continue
        times = sorted(path.stat().st_mtime for path in stage_dir.glob("*.json"))
        per_stage[stage] = len(times)
        for earlier, later in zip(times, times[1:], strict=False):
            gap = float(later - earlier)
            if 0.0 < gap <= max_gap_seconds:
                gaps.append(gap)
    record["units_per_stage"] = per_stage
    record["n_gaps"] = len(gaps)

    if len(gaps) < min_samples:
        record["reason"] = (
            f"{len(gaps)} usable inter-arrival gap(s) < {min_samples}; too few to "
            "quote a rate"
        )
        record["fallback_range_seconds"] = list(FALLBACK_SECONDS_PER_PASS)
        return record

    ordered = sorted(gaps)
    record.update(
        {
            "available": True,
            "median_seconds_per_unit": statistics.median(ordered),
            "p25_seconds_per_unit": ordered[int(0.25 * (len(ordered) - 1))],
            "p75_seconds_per_unit": ordered[int(0.75 * (len(ordered) - 1))],
        }
    )
    return record


@dataclass(frozen=True)
class FollowupBudget:
    """Pass counts for the two conditions, stated before anything is loaded."""

    l32_only_passes: int
    paired_l35_passes: int
    l32_units: dict = field(default_factory=dict)
    paired_l35_units: dict = field(default_factory=dict)
    estimated_drive_bytes: int = 0

    @property
    def total_passes(self) -> int:
        return int(self.l32_only_passes) + int(self.paired_l35_passes)

    @property
    def estimated_drive_mb(self) -> float:
        return self.estimated_drive_bytes / (1024 * 1024)

    def hours(self, seconds_per_pass: float) -> float:
        return self.total_passes * float(seconds_per_pass) / 3600.0

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["total_passes"] = self.total_passes
        payload["estimated_drive_mb"] = round(self.estimated_drive_mb, 2)
        return payload


def format_budget(
    budget: FollowupBudget, timing: Mapping, *, paired_reference_planned: bool
) -> str:
    """The block printed before the budget-confirmation cell."""
    lines = [
        "=" * 72,
        "PASS BUDGET — printed before any model is loaded",
        "=" * 72,
        f"  L32-only run              {budget.l32_only_passes:>8,} passes",
        f"  paired L35 reference      {budget.paired_l35_passes:>8,} passes"
        + ("" if paired_reference_planned else "   (not planned this run)"),
        f"  TOTAL                     {budget.total_passes:>8,} passes",
        "",
        f"  estimated Drive           {budget.estimated_drive_mb:,.1f} MiB",
        "",
    ]
    if timing.get("available"):
        low = float(timing["p25_seconds_per_unit"])
        mid = float(timing["median_seconds_per_unit"])
        high = float(timing["p75_seconds_per_unit"])
        lines += [
            "  L4 runtime, from the completed native-audio run's own unit-file",
            "  inter-arrival times (NOT an assumed 0.5 s/pass):",
            f"    median   {mid:.2f} s/pass  ->  {budget.hours(mid):.2f} h",
            f"    p25-p75  {low:.2f}-{high:.2f} s/pass  ->  "
            f"{budget.hours(low):.2f}-{budget.hours(high):.2f} h",
            "",
            f"  {timing['caveat']}",
        ]
    else:
        low, high = FALLBACK_SECONDS_PER_PASS
        lines += [
            "  L4 runtime: the completed run's timing could not be measured from",
            f"  this runtime ({timing.get('reason', 'unavailable')}), so a RANGE is",
            "  quoted instead of a point estimate:",
            f"    {low:.1f}-{high:.1f} s/pass  ->  "
            f"{budget.hours(low):.2f}-{budget.hours(high):.2f} h",
            "",
            "  No speed claim is made from an assumed 0.5 s/pass.",
        ]
    return "\n".join(lines)


# ------------------------------------------------------------- the native head


def assert_native_head_agrees(head_audit: Mapping, *, required: bool) -> dict:
    """Refuse to score a readout that is not the model's own.

    :func:`jlens.mmpilot.convergence.audit_native_head` already raises when it
    *runs* the comparison and the gap is too wide. The failure this guards is
    the quieter one: the comparison was never run at all, because ``model=None``
    was passed or the loaded object has no ``unembed``. That leaves
    ``matches_model_unembed`` as ``None``, which is not a pass — and reading it
    with a truthiness test would silently accept it.

    Args:
        required: True on the real path. False in MOCK, where the stub has no
            ``unembed`` to compare against; the record then says the comparison
            did not run instead of implying it succeeded.

    Raises:
        L32FollowupRefused: If ``required`` and the audit did not record an
            explicit ``True``.
    """
    matched = head_audit.get("matches_model_unembed")
    gap = head_audit.get("max_abs_difference_vs_model_unembed")
    record = {
        "required": bool(required),
        "matches_model_unembed": matched,
        "max_abs_difference_vs_model_unembed": gap,
        "comparison_ran": matched is not None,
        "protocol": (head_audit.get("unembed_comparison_protocol") or {}).get(
            "protocol"
        ),
        "passed": (matched is True) if required else True,
    }
    if required and matched is not True:
        raise L32FollowupRefused(
            "the audited native output head has not been shown to agree with the "
            f"model's own unembed (matches_model_unembed={matched!r}, "
            f"max_abs_difference={gap!r}). 'Native' is a fact to be established, "
            "not a label: a comparison that never ran is not a comparison that "
            "passed. Refusing to measure convergence."
        )
    return record


# ---------------------------------------------------- single-layer convergence


def run_single_layer_convergence(
    *,
    population: Mapping,
    head,
    tokenization: Mapping,
    head_audit: Mapping,
    store,
    layer: int,
    confirmation_record: Mapping,
    criterion=None,
    control_seed: int = 20260808,
) -> dict:
    """The native-readout audit for **one** layer, with per-unit resume.

    Why not :func:`jlens.mmpilot.convergence.run_convergence_audit`
    ============================================================

    That function is the completed three-modality study's driver, and its
    verdict is a *cross-layer* rule: ``PRE_CONVERGENCE_TRANSFER_SUPPORTED``
    additionally requires "at least one later validated layer clearly more
    converged", with a bootstrap interval separating it from layer 35. It also
    hard-codes layer 35 as the primary in its trajectory, sensitivity and probe
    steps.

    This study audits **one** layer. Feeding a single point to a trajectory rule
    would be inventing the comparison the rule exists to make. So the pieces are
    called directly and nothing about the completed study's rule is touched:

    * the same :func:`~jlens.mmpilot.convergence.direct_readout_row`;
    * the same three controls, from the same generator, at the same seeds;
    * the same :func:`~jlens.mmpilot.convergence.summarize_rows` and
      :func:`~jlens.mmpilot.convergence.classify_layer`, under the **frozen**
      :class:`~jlens.mmpilot.convergence.ConvergenceCriterion` whose thresholds
      are not touched here;
    * the same :class:`~jlens.mmpilot.convergence.ConvergenceStore` for atomic,
      checksum-validated, fingerprint-gated resume.

    Verdict E is then assembled by :func:`pre_convergence_verdict` from its own
    predeclared single-layer clause list.

    Args:
        confirmation_record: The published artifact's passing validation record.
            Required — layer 32 is on
            :data:`~jlens.mmpilot.convergence.LENS_INVALID_LAYERS` and nothing
            else clears it.
    """
    import torch

    from jlens.mmpilot.convergence import (
        CONVERGENCE_CRITERION,
        CONVERGENCE_PROTOCOL,
        PRIMARY_VARIANT,
        _control_rows,
        assert_lens_valid_layer,
        classify_layer,
        direct_readout_row,
        summarize_controls,
        summarize_rows,
    )

    criterion = criterion or CONVERGENCE_CRITERION
    layer = int(layer)
    assert_lens_valid_layer(
        layer, audited=(layer,), confirmation_record=confirmation_record
    )

    config_hash = payload_checksum(
        {
            "protocol": CONVERGENCE_PROTOCOL,
            "followup_protocol": L32_FOLLOWUP_PROTOCOL,
            "criterion": criterion.to_dict(),
            "tokenization": tokenization["digest"],
            "layers": [layer],
            "readout_mode": tokenization["readout_mode"],
        }
    )
    head_checksum = str(head_audit.get("head_checksum", ""))
    units = [unit for unit in population["units"] if int(unit["layer"]) == layer]

    rows: list[dict] = []
    computed = reused = 0

    def emit(variant: str, payload: dict) -> None:
        nonlocal computed
        store.save(
            store.unit_key(
                variant=variant,
                layer=layer,
                modality=str(payload["modality"]),
                sample_id=str(payload["sample_id"]),
            ),
            payload,
        )
        rows.append(payload)
        computed += 1

    for unit in units:
        key = store.unit_key(
            variant=PRIMARY_VARIANT,
            layer=layer,
            modality=str(unit["modality"]),
            sample_id=str(unit["sample_id"]),
        )
        cached = store.load(key)
        if cached is not None:
            rows.append(cached)
            reused += 1
            continue
        emit(
            PRIMARY_VARIANT,
            direct_readout_row(
                activation=torch.tensor(unit["activation"], dtype=torch.float32),
                head=head,
                tokenization=tokenization,
                concept=unit["concept"],
                clean_prediction=unit["clean_final_prediction"],
                sample_id=unit["sample_id"],
                group_id=unit["group_id"],
                image_id=unit["image_id"],
                recording_id=unit["recording_id"],
                modality=unit["modality"],
                layer=layer,
                split=unit["split"],
                capability_admissible=unit["capability_admissible"],
                activation_checksum=unit["activation_checksum"],
                head_checksum=head_checksum,
                config_hash=config_hash,
            ),
        )

    # Controls are drawn within a modality cell, so a permutation never crosses
    # a boundary the primary metric is reported across.
    cells: dict[str, list[Mapping]] = {}
    for unit in units:
        cells.setdefault(str(unit["modality"]), []).append(unit)
    for index, (modality, cell_units) in enumerate(sorted(cells.items())):
        for variant, kwargs in _control_rows(
            cell_units,
            head=head,
            tokenization=tokenization,
            head_checksum=head_checksum,
            config_hash=config_hash,
            seed=control_seed + 100 * index,
        ):
            key = store.unit_key(
                variant=variant,
                layer=layer,
                modality=modality,
                sample_id=str(kwargs["sample_id"]),
            )
            cached = store.load(key)
            if cached is not None:
                rows.append(cached)
                reused += 1
                continue
            emit(variant, direct_readout_row(**kwargs))

    summary = summarize_rows(rows, criterion=criterion, layers=(layer,))
    # ``per_layer`` is keyed by the layer's *string* form, as
    # :func:`classify_all_layers` also reads it.
    classification = classify_layer(
        summary["per_layer"][str(layer)], criterion=criterion
    )
    controls = summarize_controls(rows, layers=(layer,))
    return {
        "schema": "jlens.mmpilot.l32_single_layer_convergence.v1",
        "protocol": L32_FOLLOWUP_PROTOCOL,
        "convergence_protocol": CONVERGENCE_PROTOCOL,
        "layer": layer,
        "criterion": criterion.to_dict(),
        "criterion_digest": criterion.digest,
        "criterion_thresholds_unchanged": criterion.digest == CONVERGENCE_CRITERION.digest,
        "config_hash": config_hash,
        "summary": summary,
        "classification": classification,
        "controls": controls,
        "units_computed": computed,
        "units_reused": reused,
        "n_rows": len(rows),
        "confirmation_record_layer": int(confirmation_record.get("layer", -1)),
        "cross_layer_rule_used": False,
        "cross_layer_rule_note": (
            "convergence_verdict's trajectory clause needs a later validated "
            "layer and is deliberately not applied to a single-layer audit"
        ),
    }


# ------------------------------------------------------------------ verdict A


def lens_integrity_verdict(
    validation: Mapping,
    *,
    invariance: Mapping | None,
    discovery: Mapping,
    layer: int = L32_LAYER,
    scale: int = SELECTED_SCALE,
) -> dict:
    """A: did the published layer-32 lens survive every integrity clause?

    Args:
        layer / scale: What the artifact must be for. They default to the real
            values; the MOCK world substitutes its own, exactly as it does for
            the published lens itself.
    """
    recorded = validation.get("recorded") or {}
    evidence = discovery.get("discovery_evidence") or {}
    checks: list[dict] = [
        {
            "check": "artifact_discovered_from_metadata",
            "passed": discovery.get("discovery_version") == ARTIFACT_DISCOVERY_VERSION,
            "detail": (
                f"resolved {evidence.get('resolved_lens_path')} from "
                f"{evidence.get('report_path')}"
            ),
        },
        {
            "check": "published_artifact_validated",
            "passed": bool(validation.get("passed")),
            "detail": f"failed clauses {validation.get('failed_checks')}",
        },
        {
            "check": "selected_scale_binding",
            "passed": int(recorded.get("scale_point", -1)) == int(scale),
            "detail": f"scale_point={recorded.get('scale_point')!r} required {scale}",
        },
        {
            "check": "physical_layer",
            "passed": int(validation.get("layer", -1)) == int(layer),
            "detail": f"layer={validation.get('layer')!r} required {layer}",
        },
        {
            "check": "text_only_calibration",
            "passed": recorded.get("calibration_modality") == "text-only",
            "detail": str(recorded.get("calibration_modality")),
        },
    ]
    if invariance is None:
        checks.append(
            {
                "check": "capture_and_zero_coefficient_invariance",
                "passed": False,
                "detail": "no invariance record: a missing gate is a refusal, not a default",
            }
        )
    else:
        checks.append(
            {
                "check": "capture_and_zero_coefficient_invariance",
                "passed": bool(invariance.get("passed")),
                "detail": f"modalities {invariance.get('modalities')}",
            }
        )

    failed = [entry["check"] for entry in checks if not entry["passed"]]
    return {
        "verdict_name": VERDICT_A,
        "verdict": INTEGRITY_PASSED if not failed else INTEGRITY_REFUSED,
        "checks": checks,
        "failed_checks": failed,
        "layer": int(layer),
        "scale": int(scale),
        "protocol": L32_FOLLOWUP_PROTOCOL,
    }


# ------------------------------------------------------------------ verdict E


def _is_off_diagonal(pair: object) -> bool:
    """``"image->text"`` is off-diagonal; ``"text->text"`` is not."""
    parts = str(pair).split("->")
    return len(parts) == 2 and parts[0] != parts[1]


def admissible_supporting_cells(causal: Mapping) -> list[dict]:
    """The off-diagonal cells this verdict is allowed to rest on.

    Read from the causal verdict's own complete cell table, so a cell that was
    measured but is ``CAPABILITY_INELIGIBLE`` stays visible in the table and
    counts for nothing here — which is the whole point of
    :mod:`jlens.mmpilot.admissibility`.
    """
    return [
        cell
        for cell in (causal.get("cells") or [])
        if cell.get("evaluated")
        and cell.get("passes")
        and cell.get("counted_toward_verdict")
        and _is_off_diagonal(cell.get("pair"))
    ]


def _distinct_units(cells: Sequence[Mapping]) -> int:
    """Distinct photographs/recordings the supporting evidence rests on.

    A cell's evidence needs both halves of its pair — a positive it subtracted
    from and a matched negative it added to — so the cell contributes the
    *smaller* of its two distinct-image counts, and the maximum over cells is
    the number of photographs any single supporting cell stands on. Summing
    across cells would count one photograph once per direction it appears in.
    """
    return max(
        (
            min(
                int(cell.get("n_positive_images", 0)),
                int(cell.get("n_negative_images", 0)),
            )
            for cell in cells
        ),
        default=0,
    )


def pre_convergence_verdict(
    *,
    integrity: Mapping,
    causal: Mapping,
    convergence: Mapping,
    controls: Mapping,
    capability: Mapping,
) -> dict:
    """E: is there causal transfer *before native direct-readout convergence*?

    Every clause is predeclared and each is reported with its own pass flag, so
    a reader sees which one decided the verdict rather than only the outcome.

    The layer-32 classification is checked in the order the criterion fixes:
    ``CONVERGED`` returns :data:`TRANSFER_AT_OR_AFTER_CONVERGENCE` first and
    cannot be masked by a later clause; ``AMBIGUOUS`` can only produce
    ``INCONCLUSIVE``.
    """
    classification = convergence.get("classification")
    supporting = admissible_supporting_cells(causal)
    n_units = _distinct_units(supporting)

    def _positive_sign(cell: Mapping) -> bool:
        return float(cell.get("mean_signed_target_effect", 0.0)) > 0.0

    def _beats_controls(cell: Mapping) -> bool:
        effect = float(cell.get("mean_signed_target_effect", 0.0))
        random_control = cell.get("random_control")
        unrelated_control = cell.get("unrelated_control")
        if random_control is None or unrelated_control is None:
            return False
        return effect > float(random_control) and effect > float(unrelated_control)

    def _specific(cell: Mapping) -> bool:
        effect = abs(float(cell.get("mean_signed_target_effect", 0.0)))
        return float(cell.get("mean_abs_unrelated_change", effect)) < effect

    def _norms_sane(cell: Mapping) -> bool:
        return not any(
            "activation norm ratio" in str(reason)
            for reason in cell.get("reasons", [])
        )

    signed = [cell for cell in supporting if _positive_sign(cell)]
    beating = [cell for cell in supporting if _beats_controls(cell)]
    specific = [cell for cell in supporting if _specific(cell)]
    sane = [cell for cell in supporting if _norms_sane(cell)]

    clauses = [
        {
            "clause": "lens_integrity_passed",
            "passed": integrity.get("verdict") == INTEGRITY_PASSED,
            "detail": str(integrity.get("verdict")),
        },
        {
            "clause": "behavioral_capability_valid",
            "passed": capability.get("verdict") == "AUDIO_CAPABILITY_GO",
            "detail": str(capability.get("verdict", "no capability verdict")),
        },
        {
            "clause": "off_diagonal_effect_with_expected_sign",
            "passed": bool(signed),
            "detail": (
                f"{len(signed)} of {len(supporting)} admissible off-diagonal "
                "cell(s) moved the target in the expected direction"
            ),
        },
        {
            "clause": "exceeds_matched_random_and_external_unrelated",
            "passed": bool(beating),
            "detail": (
                f"{len(beating)} cell(s) exceeded BOTH their matched-random and "
                "external-unrelated controls"
            ),
        },
        {
            "clause": "effect_is_specific_not_global_disruption",
            "passed": bool(specific),
            "detail": (
                f"{len(specific)} cell(s) moved unrelated candidates less than "
                "the target"
            ),
        },
        {
            "clause": "activation_norms_sane",
            "passed": len(sane) == len(supporting) and bool(supporting),
            "detail": (
                f"{len(sane)} of {len(supporting)} cell(s) inside the norm-ratio "
                "bounds"
            ),
        },
        {
            "clause": f"evidence_on_at_least_{MIN_DISTINCT_UNITS}_distinct_photographs",
            "passed": n_units >= MIN_DISTINCT_UNITS,
            "detail": (
                f"the strongest supporting cell rests on {n_units} distinct "
                "photograph(s)/recording(s)"
            ),
        },
        {
            "clause": "l32_not_converged_under_frozen_criterion",
            "passed": classification == NOT_CONVERGED,
            "detail": f"classification={classification!r}",
        },
        {
            # ``summarize_controls`` names it ``all_controls_passed``; the key
            # is read rather than renamed, and its absence is a failure rather
            # than a default — a missing control record is not a passing one.
            "clause": "convergence_controls_pass",
            "passed": controls.get(
                "all_controls_passed", controls.get("passed")
            ) is True,
            "detail": str(controls.get("failed_controls", "no control record")),
        },
    ]
    failed = [entry["clause"] for entry in clauses if not entry["passed"]]

    if classification == CONVERGED:
        verdict = TRANSFER_AT_OR_AFTER_CONVERGENCE
        rationale = (
            f"Layer {L32_LAYER} is classified {CONVERGED} under the frozen "
            "native-readout criterion, so any causal effect measured there "
            f"occurs at or after native direct-readout convergence. This branch "
            "is checked first and cannot be masked."
        )
    elif classification == AMBIGUOUS:
        verdict = INCONCLUSIVE
        rationale = (
            f"Layer {L32_LAYER} falls between the two bars ({AMBIGUOUS}). The "
            "criterion admits no pre-convergence conclusion from an ambiguous "
            "layer, and its thresholds are not revisable now that the result is "
            "visible."
        )
    elif not failed:
        verdict = SUPPORTED
        rationale = (
            f"Every predeclared clause holds: an admissible off-diagonal causal "
            f"effect at layer {L32_LAYER} with the expected sign, above its "
            "matched-random and external-unrelated controls, specific rather "
            f"than global, on {n_units} distinct photographs/recordings, while "
            f"layer {L32_LAYER} is {NOT_CONVERGED} under the frozen criterion. "
            f"This is causal transfer {CONVERGENCE_PHRASE}."
        )
    else:
        verdict = NOT_SUPPORTED if supporting else INCONCLUSIVE
        rationale = (
            f"Layer {L32_LAYER} is {NOT_CONVERGED}, but "
            f"{len(failed)} predeclared clause(s) did not hold: {failed}. "
            "No pre-convergence conclusion is forced."
        )

    return {
        "verdict_name": VERDICT_E,
        "verdict": verdict,
        "layer": L32_LAYER,
        "classification": classification,
        "clauses": clauses,
        "failed_clauses": failed,
        "supporting_cells": [
            {
                "concept": cell.get("concept"),
                "pair": cell.get("pair"),
                "mean_signed_target_effect": cell.get("mean_signed_target_effect"),
                "random_control": cell.get("random_control"),
                "unrelated_control": cell.get("unrelated_control"),
                "n_positive_images": cell.get("n_positive_images"),
                "n_negative_images": cell.get("n_negative_images"),
            }
            for cell in supporting
        ],
        "n_admissible_off_diagonal_supporting_cells": len(supporting),
        "n_distinct_units": n_units,
        "min_distinct_units": MIN_DISTINCT_UNITS,
        "rationale": rationale,
        "phrase": CONVERGENCE_PHRASE,
        "intervention_family": INTERVENTION_FAMILY,
        "protocol": L32_FOLLOWUP_PROTOCOL,
    }


# ------------------------------------------------------- next-step recommendation


def adjacent_layer_recommendation(
    *, causal_verdict: str, classification: str, reference_causal: str | None = None
) -> dict:
    """What to do next — conditional on the result, decided in advance.

    Layers 33 and 34 are neither fitted nor tested in this study. Whether they
    *should* be depends on which of four states the result lands in, and the
    mapping is fixed here so it is not chosen after the fact.
    """
    if classification == CONVERGED:
        return {
            "recommendation": "investigate_layers_earlier_than_32",
            "fit_l33_l34": False,
            "rationale": (
                f"Layer {L32_LAYER} has already converged, so layers 33 and 34 "
                "are later still and cannot establish an earlier "
                f"{CONVERGENCE_PHRASE} result. The question moves to layers "
                f"shallower than {L32_LAYER}, which requires a fresh calibration "
                "at those depths."
            ),
        }
    if classification == AMBIGUOUS:
        return {
            "recommendation": "smallest_convergence_resolution_study",
            "fit_l33_l34": False,
            "rationale": (
                f"Layer {L32_LAYER} sits between the two bars. Adjacent layers "
                "cannot resolve an ambiguity about layer 32 itself, so the next "
                "step is a separately fingerprinted, predeclared **independent "
                f"layer-{L32_LAYER} convergence-resolution population** — new "
                "photographs and recordings under this same open protocol, "
                "scored against these same frozen thresholds, with the "
                "classification declared before the data are opened. "
                "AMBIGUOUS is a measured outcome, not a sample-size problem: "
                "an ambiguous layer can stay ambiguous at any n, and nothing "
                "about a larger population makes NOT_CONVERGED more likely to "
                "be the answer. What a fresh population buys is an independent "
                "test of the same question, not a nudge toward a preferred side."
            ),
        }
    if causal_verdict == SUPPORTED and classification == NOT_CONVERGED:
        return {
            "recommendation": "no_adjacent_sweep_required",
            "fit_l33_l34": False,
            "rationale": (
                f"The core claim is established at layer {L32_LAYER}: causal "
                f"transfer {CONVERGENCE_PHRASE}. Layers 33 and 34 would sharpen "
                "localization and are optional; nothing in the claim waits on "
                "them."
            ),
        }
    if causal_verdict != SUPPORTED and reference_causal == SUPPORTED:
        return {
            "recommendation": "fit_l33_l34",
            "fit_l33_l34": True,
            "rationale": (
                f"Layer {L32_LAYER} is noncausal while the matched layer "
                f"{REFERENCE_LAYER} reference under the same protocol is causal, "
                "so the effect appears somewhere between them. Layers 33 and 34 "
                "are the smallest bracket that localizes it, and they must be "
                "fitted and confirmed before they can be tested."
            ),
        }
    return {
        "recommendation": "no_recommendation_the_premises_are_not_met",
        "fit_l33_l34": False,
        "rationale": (
            f"Layer {L32_LAYER} did not show causal transfer and no matched "
            f"layer {REFERENCE_LAYER} reference under this protocol is available "
            "to say whether the effect exists anywhere under an open prompt. "
            "Fitting adjacent layers would be searching, not localizing."
        ),
    }


# ------------------------------------------------------------------ fingerprint


#: Every scientific configuration field. A change in any one refuses the resume.
FINGERPRINT_FIELDS: tuple[str, ...] = (
    "protocol",
    "intervention_family",
    "model_repo_id",
    "model_revision",
    "processor_revision",
    "transformers_version",
    "audio_protocol_version",
    "audio_protocol_fingerprint",
    "lens_path",
    "lens_checksum",
    "physical_layer",
    "d_model",
    "hook_site",
    "residual_convention",
    "final_prompt_token_position",
    "dictionary_orientation",
    "dictionary_normalization",
    "calibration_modality",
    "selected_scale",
    "confirmation_status",
    "confirmation_set_checksum",
    "publication_metadata_checksum",
    "original_manifest_checksum",
    "derived_manifest_checksum",
    "subset_provenance_checksum",
    "split_provenance_checksum",
    "prompt_protocol",
    "prompt_protocol_version",
    "prompt_hash",
    "prompt_protocol_digest",
    "selected_concepts",
    "capability_admissible_concepts",
    "source_sample_ids",
    "target_sample_ids",
    "controls",
    "alphas",
    "direction_normalization",
    "convergence_criterion_version",
    "convergence_criterion_digest",
)


def followup_fingerprint(**fields) -> dict:
    """Bind every scientific configuration field, and refuse a missing one.

    A field absent from ``fields`` is a refusal rather than a ``None``: a
    fingerprint that silently omits the audio protocol would let an audio run
    resume from a text run's directory.

    Raises:
        L32FollowupRefused: If any field in :data:`FINGERPRINT_FIELDS` is
            missing, or an unknown field is supplied.
    """
    missing = [name for name in FINGERPRINT_FIELDS if name not in fields]
    unknown = sorted(set(fields) - set(FINGERPRINT_FIELDS))
    if missing or unknown:
        raise L32FollowupRefused(
            "the follow-up fingerprint is incomplete or over-specified: "
            f"missing {missing}, unknown {unknown}"
        )
    payload = {name: fields[name] for name in FINGERPRINT_FIELDS}
    return {**payload, "fingerprint_digest": payload_checksum(payload)}


#: Words that mark a forbidden phrase as being *ruled out* rather than asserted.
#: The rule is "the report must not claim pre-linguistic", not "the report must
#: not contain the letters" — a criterion that tells a reader which phrases are
#: forbidden has to be able to name them.
#:
#: Deliberately only these two. A bare ``"not "`` was considered and rejected:
#: it is common enough in ordinary prose that it would excuse an affirmative use
#: sitting a sentence away from an unrelated negation, which is exactly the slip
#: this check exists to catch.
PHRASE_NEGATION_MARKERS: tuple[str, ...] = ("never", "forbidden")

#: How far either side of an occurrence a negation marker is looked for.
PHRASE_NEGATION_WINDOW = 400


def assert_report_phrasing(text: str) -> dict:
    """The report says ``before native direct-readout convergence`` and nothing else.

    Every occurrence of a forbidden phrase must sit inside a passage that is
    *ruling it out*. A bare affirmative use is refused; a sentence such as
    "never 'pre-linguistic'" is not, because otherwise the criterion could not
    state its own boundary and the check would be evaded by paraphrase instead
    of enforced.

    Raises:
        L32FollowupRefused: If a forbidden phrase appears with no negation
            marker near it. The claim this study can make is about a readout;
            the forbidden phrases are claims about language, and no measurement
            here supports one.
    """
    lowered = str(text).lower()
    asserted: list[dict] = []
    ruled_out: list[str] = []
    for phrase in FORBIDDEN_PHRASES:
        start = 0
        while (index := lowered.find(phrase, start)) != -1:
            window = lowered[
                max(0, index - PHRASE_NEGATION_WINDOW) : index
                + len(phrase)
                + PHRASE_NEGATION_WINDOW
            ]
            if any(marker in window for marker in PHRASE_NEGATION_MARKERS):
                ruled_out.append(phrase)
            else:
                asserted.append({"phrase": phrase, "context": window.strip()[:200]})
            start = index + len(phrase)
    if asserted:
        raise L32FollowupRefused(
            "the report uses phrasing this study cannot support: "
            f"{[entry['phrase'] for entry in asserted]}. Say "
            f"{CONVERGENCE_PHRASE!r} — a weak native readout is a fact about the "
            "readout, not about whether linguistic information is present. "
            f"First occurrence: {asserted[0]['context']!r}"
        )
    return {
        "checked_phrases": list(FORBIDDEN_PHRASES),
        "negation_markers": list(PHRASE_NEGATION_MARKERS),
        "phrases_named_as_forbidden": sorted(set(ruled_out)),
        "phrases_asserted": [],
        "required_phrase": CONVERGENCE_PHRASE,
        "required_phrase_present": CONVERGENCE_PHRASE in lowered,
        "passed": True,
    }


__all__ = [
    "ARTIFACT_DISCOVERY_VERSION",
    "CONFIRMATION_SET_SIZE",
    "CONVERGENCE_PHRASE",
    "COMPLETED_STUDY_PROMPT_PROTOCOL",
    "EXTENSION_ARTIFACT_SCHEMA",
    "EXTENSION_REPORT_NAME",
    "EXTENSION_REPORT_SCHEMA",
    "FALLBACK_SECONDS_PER_PASS",
    "FINGERPRINT_FIELDS",
    "FORBIDDEN_PHRASES",
    "INTERVENTION_FAMILY",
    "L32_FOLLOWUP_PROTOCOL",
    "L32_LAYER",
    "MIN_DISTINCT_UNITS",
    "OPEN_PROMPT_PROTOCOL",
    "REFERENCE_LAYER",
    "SELECTED_SCALE",
    "ArtifactDiscoveryRefused",
    "DiscoveredLens",
    "FollowupBudget",
    "L32FollowupRefused",
    "PairedReferenceRequired",
    "adjacent_layer_recommendation",
    "admissible_supporting_cells",
    "assert_native_head_agrees",
    "assert_paired_reference_available",
    "assert_report_phrasing",
    "derive_seconds_per_pass",
    "discover_published_l32_lens",
    "followup_fingerprint",
    "format_budget",
    "l32_expectations",
    "lens_integrity_verdict",
    "pre_convergence_verdict",
    "prompt_protocol_comparability",
    "read_extension_report",
    "run_single_layer_convergence",
    "separate_measured_from_historical",
    "validate_discovered_lens",
]
