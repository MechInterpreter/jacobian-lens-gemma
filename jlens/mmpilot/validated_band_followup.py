# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""The prospective causal follow-up over the validated physical band L33-L40.

What this is, stated once and repeated in every artifact it writes:

The corrected independent lens validation (run ``bandcorr_real_eb5b00f135e4``,
report ``artifacts/corrected_validation_v1/band_interior_corrected_validation_report.json``)
scored all nine physical layers 32-40 on one untouched confirmation population
under the corrected fixed-universe wrong-layer control. **L32 failed** the
frozen coverage/nondegeneracy clause — tied-at-max 0.51171875 against a frozen
ceiling of 0.50 — and L33-L40 passed. The largest validated contiguous band is
therefore ``[33, 40]``, and the originally planned full L32-L40 confirmatory
study remains :data:`ORIGINAL_VERDICT`.

This module runs the *same* paper-style two-coordinate contiguous-band swap over
``[33, 40]``. That is a **prospective causal follow-up selected from a completed
text-only lens-validation result**, and it is never the originally planned
L32-L40 confirmatory band. The distinction is not cosmetic: the band was chosen
after seeing a validation outcome, so the follow-up can support a claim about
L33-L40 under this protocol and can support no claim whatever about a band
beginning at L32, nor about any layer earlier than 33.

**Nothing here refits, rescores, reinterprets or writes into the completed
corrected run.** It is read, checksummed before and after, and proved unchanged.
:func:`assert_corrected_run_unmodified` is the proof, and the preflight refuses
before a model can load if it fails.

What is reused rather than reimplemented, because a second implementation of
any of it would be a second thing to keep in step:

* the intervention — :func:`jlens.mmpilot.coordinate_swap.coordinate_swap_band`,
  :func:`~jlens.mmpilot.coordinate_swap.run_swap_condition`,
  :func:`~jlens.mmpilot.coordinate_swap.build_swap_basis_from_vectors`,
  :func:`~jlens.mmpilot.coordinate_swap.random_two_direction_basis`;
* the band algebra — :func:`jlens.mmpilot.band_swap.assert_contiguous`,
  :func:`~jlens.mmpilot.band_swap.build_band`,
  :func:`~jlens.mmpilot.band_swap.predeclare_suffix_bands`;
* the record, the aggregation and the judgement —
  :func:`~jlens.mmpilot.band_swap.band_trial_record`,
  :func:`~jlens.mmpilot.band_swap.summarize_band_cells`,
  :func:`~jlens.mmpilot.band_swap.band_reasoning_verdict`,
  :func:`~jlens.mmpilot.band_swap.band_onset_timing`,
  :class:`~jlens.mmpilot.band_swap.BandSwapThresholds`;
* the population — :func:`jlens.mmpilot.paper_reasoning_swap.hidden_animal_population`,
  :func:`~jlens.mmpilot.paper_reasoning_swap.select_capability_eligible_samples`.

The verdict wrappers here **re-label and nothing else**. No threshold is
loosened, no clause is added or removed, no criterion is restated. They exist
because the follow-up must not be reported under the completed study's names.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from jlens.metadata import file_sha256
from jlens.mmpilot.band_control import (
    CORRECTED_ARTIFACT_SCHEMA,
    CORRECTED_PUBLISHED_STATUS,
    CORRECTED_REPORT_SCHEMA,
    CORRECTED_SCALE,
    CORRECTED_VALIDATION_DIRNAME,
    CORRECTION_PROTOCOL_VERSION,
    FIXED_CONTROL_UNIVERSE,
)
from jlens.mmpilot.band_swap import (
    ALPHA_ROLES,
    BAND_ARMS,
    BAND_CONDITIONS,
    BAND_INTERVENTION_FAMILY,
    PRIMARY_ALPHA,
    SECONDARY_ALPHA,
    assert_contiguous,
    band_key,
    band_onset_timing,
    contiguous_runs,
)
from jlens.mmpilot.coordinate_swap import (
    METHOD_VERSION as COORDINATE_SWAP_METHOD_VERSION,
)
from jlens.mmpilot.coordinate_swap import (
    PRIMARY_POSITION_RULE,
)
from jlens.mmpilot.store import payload_checksum

__all__ = [
    "CORRECTED_REPORT_NAME",
    "EXPECTED_ARTIFACT_CHECKSUMS",
    "EXPECTED_CONFIRMATION_MANIFEST_CHECKSUM",
    "EXPECTED_PROTOCOL_DIGEST",
    "EXPECTED_REPORT_CHECKSUM",
    "EXPECTED_UNIVERSE_CHECKSUM",
    "EXCLUDED_FAILED_LAYER",
    "EXCLUDED_LAYER_REASON",
    "FOLLOWUP_BAND_END",
    "FOLLOWUP_BAND_START",
    "FOLLOWUP_CAPABILITY_NO_GO",
    "FOLLOWUP_DESIGN_VERSION",
    "FOLLOWUP_INTERVENTION_FAMILY",
    "FOLLOWUP_NOT_RUN",
    "FOLLOWUP_PRIMARY_BAND",
    "FOLLOWUP_PROTOCOL_VERSION",
    "FOLLOWUP_REPORT_NAME",
    "FOLLOWUP_REPORT_SCHEMA",
    "FOLLOWUP_STUDY_NAME",
    "FOLLOWUP_SUFFIX_STARTS",
    "FOLLOWUP_TIMING_NAMES",
    "FOLLOWUP_TIMING_VERSION",
    "FOLLOWUP_VERDICT_NAMES",
    "FOLLOWUP_VERDICT_VERSION",
    "ORIGINAL_RUN_NAME",
    "ORIGINAL_VERDICT",
    "REPORTING_BOUNDARY",
    "SELECTION_EVIDENCE_STATEMENT",
    "CorrectedArtifact",
    "FollowupRefused",
    "assert_band_hook_integrity",
    "assert_corrected_run_unmodified",
    "assert_followup_band",
    "assert_no_fitting_entry_point",
    "corrected_run_digest",
    "discover_corrected_band_lenses",
    "followup_design_record",
    "followup_fingerprint",
    "followup_onset_timing",
    "followup_pass_budget",
    "followup_preflight_record",
    "followup_report",
    "followup_verdict",
    "format_followup_boundary",
    "format_followup_pass_budget",
    "format_followup_preflight",
    "format_followup_verdict",
    "read_corrected_validation_report",
    "read_followup_units",
]


class FollowupRefused(RuntimeError):
    """The follow-up cannot be admitted, designed, resumed or scored."""


# ------------------------------------------------------- the frozen selection

#: The completed corrected validation this follow-up is selected from. It is
#: read-only evidence; nothing in this module writes into it.
ORIGINAL_RUN_NAME = "bandcorr_real_eb5b00f135e4"

#: The verdict the originally planned full L32-L40 study carries, permanently.
#: A favourable follow-up result does not change it and may never be reported
#: as having changed it.
ORIGINAL_VERDICT = "BAND_CORRECTED_CONTROL_NO_GO"

CORRECTED_REPORT_NAME = "band_interior_corrected_validation_report.json"

#: The corrected report's own ``report_checksum``, pinned. The preflight
#: recomputes it and refuses on any disagreement, so the selection evidence
#: cannot be swapped for a different report.
EXPECTED_REPORT_CHECKSUM = (
    "sha256:318d2106ca35dd752e1c6fa72e336d85e705473794065a0dfd2f9ebb00ce210b"
)
EXPECTED_PROTOCOL_DIGEST = (
    "sha256:2e9086c170445cfdd24ee206a780776b16322a8d47f37d67f9969d27eedfc92b"
)
EXPECTED_UNIVERSE_CHECKSUM = (
    "sha256:a64c13fa902bf22f2c8319cfe41256c0b66843aa6f68f3ddc554f6a94bfa2452"
)
EXPECTED_CONFIRMATION_MANIFEST_CHECKSUM = (
    "sha256:44e351f2873149f8e406cfc28ff27565d95a769b5be3858d3ea8111973aeb16c"
)

#: The eight corrected publications, by physical layer. Every one is
#: re-checksummed from the file on disk and matched against this table; a
#: missing, extra, duplicated or mismatched artifact is refused rather than
#: worked around.
EXPECTED_ARTIFACT_CHECKSUMS: dict[int, str] = {
    33: "sha256:4b86e311394e50f02c95dcf00e216bf2cc73825407c9c892f4a382f1059e41f3",
    34: "sha256:712f0a94b9b304ba2f7425b2f39970a65385ceb879bd9378ca1a0fdc3b6ab92e",
    35: "sha256:6e752df531dc4ba013dbf8688880ca3124041f8546f8620316af9e3900f3112b",
    36: "sha256:71cd563cb2a93019a16184841b82bbcad506f7d2477e74a1d594237e39c95bfe",
    37: "sha256:b11697465e01e323928b90a3c4daef37ecb7c01da3f6d0ca4e02eae192728ffb",
    38: "sha256:184d41d65377a0dc789aacc1c4a54ea941afa2671f0b0863eaa3dd0a8df54abf",
    39: "sha256:6cf321cc98dc8e13bdb8e34214e21a2edbbd57891f0bdc2f8cb9830ae86615aa",
    40: "sha256:f4ff5c7687b69b53a3e8cf7a687625309a9dca1e3a2c7838daa374c337c23816",
}

#: The layer that failed the frozen gate. It is excluded categorically: not
#: because it is inconvenient, and not by a threshold this study is free to
#: revisit. There is no configuration of this module that admits it.
EXCLUDED_FAILED_LAYER = 32

#: Why L32 is out, in the words of the completed report's own clause names. It
#: passed MRR-vs-noise, MRR-vs-wrong-layer, rank/top-10 and fold stability, and
#: failed only coverage/nondegeneracy. That is still a failure of the frozen
#: gate, and a partially-passing layer is not a validated layer.
EXCLUDED_LAYER_REASON = (
    "L32 failed the frozen coverage/nondegeneracy clause on the corrected "
    "confirmation population: tied-at-max rate 0.51171875 against the frozen "
    "ceiling 0.50. It passed the MRR-vs-noise, MRR-vs-wrong-layer, rank/top-10 "
    "and fold-stability clauses, and that does not make it validated. The "
    "ceiling is frozen, the population is not re-drawn, and no band admitted "
    "here begins at 32."
)

FOLLOWUP_BAND_START = 33
FOLLOWUP_BAND_END = 40

#: The primary contiguous band, and the only one this module will admit.
FOLLOWUP_PRIMARY_BAND: tuple[int, ...] = tuple(
    range(FOLLOWUP_BAND_START, FOLLOWUP_BAND_END + 1)
)

#: The predeclared sparse suffix-band **starts**. The earlier intended study's
#: topology (32/35/38/40), adjusted in exactly one place because the validated
#: primary band starts at 33. These are starts of contiguous ranges ending at
#: :data:`FOLLOWUP_BAND_END`, never a sampled layer grid, and they are frozen
#: before any causal number exists. Nothing is added or removed afterwards.
FOLLOWUP_SUFFIX_STARTS: tuple[int, ...] = (33, 35, 38, 40)

FOLLOWUP_STUDY_NAME = "L33_L40_VALIDATED_BAND_FOLLOWUP"
FOLLOWUP_PROTOCOL_VERSION = "mmpilot.l33_l40_validated_band_followup.v1"
FOLLOWUP_DESIGN_VERSION = "mmpilot.l33_l40_validated_band_followup_design.v1"
FOLLOWUP_VERDICT_VERSION = "mmpilot.l33_l40_validated_band_followup_verdict.v1"
FOLLOWUP_TIMING_VERSION = "mmpilot.l33_l40_validated_band_followup_timing.v1"
FOLLOWUP_REPORT_SCHEMA = "jlens.mmpilot.l33_l40_validated_band_followup_report.v1"
FOLLOWUP_REPORT_NAME = "l33_l40_validated_band_followup_report.json"

#: The unit family. Deliberately distinct from
#: :data:`jlens.mmpilot.band_swap.BAND_INTERVENTION_FAMILY` at the *study*
#: level, while the stored trial records keep that family because the
#: intervention genuinely is that intervention. What separates the two is the
#: run fingerprint and the report schema, so a completed L32-L40 unit can never
#: be read as a follow-up unit and vice versa.
FOLLOWUP_INTERVENTION_FAMILY = "l33_l40_validated_band_coordinate_swap"

SELECTION_EVIDENCE_STATEMENT = (
    "L33-L40 was selected solely from the completed, immutable text-only "
    "corrected lens-validation report. No causal result was consulted, and none "
    "existed for any band when the selection was made."
)

#: Printed by the preflight, written into every report, and asserted by the
#: tests. These are the sentences that keep the follow-up from being read as
#: something it is not.
REPORTING_BOUNDARY: tuple[str, ...] = (
    "L33-L40 was selected AFTER the independent lens-validation result.",
    "This is therefore a PROSPECTIVE CAUSAL FOLLOW-UP, not the originally "
    "planned L32-L40 confirmatory band.",
    f"The originally planned L32-L40 study remains {ORIGINAL_VERDICT}, "
    "permanently; a favourable result here does not change it and is never "
    "reported as confirming it.",
    "L32 is excluded because it failed the frozen gate.",
    "No result here supports a claim about a band beginning at L32.",
    "No claim may be made about any layer earlier than 33.",
    "SpokenCOCO tests linguistic spoken captions, not environmental sound.",
    "The model outputs text; image and audio are input modalities only.",
    "alpha=2 is sensitivity evidence and is not interchangeable with the "
    "alpha=1 primary evidence.",
)


def format_followup_boundary() -> str:
    """The reporting-boundary block, printed before anything else happens."""
    lines = ["=" * 78, f"REPORTING BOUNDARY — {FOLLOWUP_STUDY_NAME}", "=" * 78]
    lines += [f"  * {line}" for line in REPORTING_BOUNDARY]
    return "\n".join(lines)


# --------------------------------------------------- reading the selection

def _read_json(path: Path, *, what: str) -> dict:
    if not path.is_file():
        raise FollowupRefused(f"{what} not found at {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise FollowupRefused(f"{what} at {path} is not readable JSON: {error}") from error
    if not isinstance(payload, dict):
        raise FollowupRefused(f"{what} at {path} does not hold a JSON object")
    return payload


def read_corrected_validation_report(
    corrected_run_dir: str | os.PathLike[str],
    *,
    expected_report_checksum: str = EXPECTED_REPORT_CHECKSUM,
    expected_protocol_digest: str = EXPECTED_PROTOCOL_DIGEST,
    expected_universe_checksum: str = EXPECTED_UNIVERSE_CHECKSUM,
    expected_confirmation_manifest_checksum: str = (
        EXPECTED_CONFIRMATION_MANIFEST_CHECKSUM
    ),
    expected_model_repo_id: str,
    expected_model_revision: str,
    expected_scale: int = CORRECTED_SCALE,
    require_real_mode: bool = True,
) -> tuple[Path, dict]:
    """Read and verify the completed corrected validation report.

    ``corrected_run_dir`` is the **explicitly configured** run directory. There
    is deliberately no globbing and no "latest run" resolution anywhere in this
    module: a study whose selection evidence depends on directory sort order is
    a study whose selection evidence can change without anyone editing it.

    Every clause below is a refusal, not a warning:

    * the file exists at the corrected publication directory's own report name;
    * its ``schema`` is :data:`~jlens.mmpilot.band_control.CORRECTED_REPORT_SCHEMA`;
    * its recorded ``report_checksum`` equals ``expected_report_checksum`` **and**
      recomputes over its own body;
    * ``mode`` is ``"real"``;
    * the protocol digest, universe checksum and confirmation-manifest checksum
      are the pinned ones;
    * the model identity and revision match;
    * the scale is 250, the frozen gate digest is intact, and the protocol
      records that nothing was refitted and no threshold changed.

    Returns:
        ``(report_path, report)``.

    Raises:
        FollowupRefused: On any failed clause, naming every problem found.
    """
    root = Path(corrected_run_dir)
    path = root / "artifacts" / CORRECTED_VALIDATION_DIRNAME / CORRECTED_REPORT_NAME
    report = _read_json(path, what="corrected validation report")
    problems: list[str] = []

    if report.get("schema") != CORRECTED_REPORT_SCHEMA:
        problems.append(
            f"schema is {report.get('schema')!r}, not {CORRECTED_REPORT_SCHEMA!r}"
        )
    recorded = str(report.get("report_checksum"))
    if recorded != str(expected_report_checksum):
        problems.append(
            f"report_checksum is {recorded!r}, not the expected "
            f"{expected_report_checksum!r}"
        )
    recomputed = payload_checksum(
        {k: v for k, v in report.items() if k != "report_checksum"}
    )
    if recomputed != recorded:
        problems.append(
            "the report does not match its own report_checksum (body hashes to "
            f"{recomputed!r}, the file claims {recorded!r}). Either the body was "
            "edited after it was written, or a non-JSON-native value was "
            "stringified by the writer's default=str on the way to disk and no "
            "longer round-trips. Report this rather than editing anything: the "
            "recorded checksum above already matched its pin, so the file is the "
            "one the completed run produced."
        )
    if require_real_mode and report.get("mode") != "real":
        problems.append(
            f"mode is {report.get('mode')!r}; a MOCK report selects no band and "
            "validates no layer"
        )
    if report.get("correction_protocol_version") != CORRECTION_PROTOCOL_VERSION:
        problems.append(
            f"correction_protocol_version is "
            f"{report.get('correction_protocol_version')!r}, not "
            f"{CORRECTION_PROTOCOL_VERSION!r}"
        )

    protocol = dict(report.get("protocol") or {})
    if str(protocol.get("protocol_digest")) != str(expected_protocol_digest):
        problems.append(
            f"protocol_digest is {protocol.get('protocol_digest')!r}, not "
            f"{expected_protocol_digest!r}"
        )
    if str(protocol.get("universe_checksum")) != str(expected_universe_checksum):
        problems.append(
            f"universe_checksum is {protocol.get('universe_checksum')!r}, not "
            f"{expected_universe_checksum!r}"
        )
    if str(report.get("confirmation_manifest_checksum")) != str(
        expected_confirmation_manifest_checksum
    ):
        problems.append(
            "confirmation_manifest_checksum is "
            f"{report.get('confirmation_manifest_checksum')!r}, not "
            f"{expected_confirmation_manifest_checksum!r}"
        )
    if str(protocol.get("model_repo_id")) != str(expected_model_repo_id):
        problems.append(
            f"model_repo_id is {protocol.get('model_repo_id')!r}, not "
            f"{expected_model_repo_id!r}"
        )
    if str(protocol.get("model_revision")) != str(expected_model_revision):
        problems.append(
            f"model_revision is {protocol.get('model_revision')!r}, not "
            f"{expected_model_revision!r}"
        )
    if int(protocol.get("scale", -1)) != int(expected_scale):
        problems.append(
            f"scale is {protocol.get('scale')!r}, not {int(expected_scale)}"
        )
    if not protocol.get("gate_is_the_frozen_one", False):
        problems.append("the protocol does not record the frozen gate")
    if not protocol.get("gate_digest"):
        problems.append("the protocol records no gate digest")
    for flag in ("no_lens_was_refitted", "no_threshold_was_changed"):
        if not protocol.get(flag, False):
            problems.append(f"the protocol does not record {flag}")
    if not protocol.get("hook_convention"):
        problems.append("the protocol records no hook convention")
    if not protocol.get("fit_prefix_checksum"):
        problems.append("the protocol records no fit-prefix checksum")

    if problems:
        raise FollowupRefused(
            f"{path} is not the completed corrected validation this follow-up is "
            "selected from:\n  - " + "\n  - ".join(problems)
        )
    return path, report


def assert_followup_band(
    report: Mapping,
    *,
    expected_passing: Sequence[int] = FOLLOWUP_PRIMARY_BAND,
    excluded_layer: int = EXCLUDED_FAILED_LAYER,
) -> dict:
    """Prove the corrected report admits exactly this follow-up band and no other.

    The clauses are deliberately equalities rather than containments. "L33-L40
    is *among* the passing layers" would admit a report that also passed L32, and
    a follow-up designed against that report would be the original confirmatory
    study wearing a different name.

    Raises:
        FollowupRefused: If ``layers_passing`` is not exactly ``expected_passing``,
            if ``excluded_layer`` is not among ``layers_failing``, if
            ``largest_admissible_contiguous_band`` is not exactly ``[33, 40]``,
            if the report unblocked the original stage 3, or if it records a
            changed threshold or a refitted matrix.
    """
    verdict = dict(report.get("band_verdict") or {})
    provenance = dict(report.get("provenance") or {})
    wanted = [int(layer) for layer in expected_passing]
    problems: list[str] = []

    passing = [int(layer) for layer in verdict.get("layers_passing", [])]
    failing = [int(layer) for layer in verdict.get("layers_failing", [])]
    if passing != wanted:
        problems.append(
            f"layers_passing is {passing}, not exactly {wanted}. Only the exact "
            "validated band admits this follow-up; a different passing set is a "
            "different study and needs its own predeclaration."
        )
    if int(excluded_layer) not in failing:
        problems.append(
            f"layer {excluded_layer} is not among layers_failing {failing}. This "
            "follow-up exists because it failed; a report in which it did not is "
            "not the selection evidence."
        )
    if int(excluded_layer) in passing:
        problems.append(
            f"layer {excluded_layer} is among layers_passing {passing}, which "
            "contradicts the completed result"
        )
    band = verdict.get("largest_admissible_contiguous_band")
    if band is None or [int(v) for v in band] != [wanted[0], wanted[-1]]:
        problems.append(
            f"largest_admissible_contiguous_band is {band!r}, not "
            f"{[wanted[0], wanted[-1]]}"
        )
    if verdict.get("verdict") != ORIGINAL_VERDICT:
        problems.append(
            f"the completed band verdict is {verdict.get('verdict')!r}, not the "
            f"frozen {ORIGINAL_VERDICT!r}"
        )
    if verdict.get("full_band_available", False):
        problems.append(
            "the completed report claims the full L32-L40 band is available; it "
            "is not, and a report that says otherwise is not this one"
        )
    for flag, message in (
        ("stage3_unblocked", "the completed report unblocked its own stage 3"),
        ("old_and_new_verdicts_combined", "old and new verdicts were combined"),
    ):
        if provenance.get(flag, False) or verdict.get(flag, False):
            problems.append(message)
    if not provenance.get("no_frozen_numerical_threshold_was_changed", False):
        problems.append("the report does not record that no threshold was changed")
    if not provenance.get("no_matrix_was_refitted", False):
        problems.append("the report does not record that no matrix was refitted")

    if problems:
        raise FollowupRefused(
            "the corrected validation report does not admit the L33-L40 "
            "follow-up:\n  - " + "\n  - ".join(problems)
        )

    assert_contiguous(wanted, what="follow-up primary band")
    return {
        "layers_passing": passing,
        "layers_failing": failing,
        "followup_band": list(wanted),
        "excluded_failed_layer": int(excluded_layer),
        "excluded_layer_reason": EXCLUDED_LAYER_REASON,
        "original_verdict": ORIGINAL_VERDICT,
        "original_stage3_remained_blocked": True,
        "selection_evidence": SELECTION_EVIDENCE_STATEMENT,
        "no_causal_outcome_consulted": True,
    }


# --------------------------------------------------- the published artifacts


@dataclass(frozen=True)
class CorrectedArtifact:
    """One corrected publication, resolved from the report's own sidecars.

    ``layer_key_in_file`` is the key the Jacobian sits under inside the saved
    lens, which for a single-layer corrected publication is the physical layer
    itself. It is recorded rather than assumed so the reader never has to guess.
    """

    layer: int
    lens_path: str
    lens_checksum: str
    sidecar_path: str
    layer_key_in_file: int
    scale: int
    matrix_checksum: str | None
    universe_checksum: str | None
    protocol_digest: str | None
    confirmation_manifest_checksum: str | None
    gate_digest: str | None
    provenance: str = "corrected_validation_published_artifact"

    def to_dict(self) -> dict:
        return asdict(self)


_GEOMETRY_FIELDS = (
    "universe_checksum",
    "protocol_digest",
    "confirmation_manifest_checksum",
    "gate_digest",
)


def discover_corrected_band_lenses(
    corrected_run_dir: str | os.PathLike[str],
    *,
    report: Mapping,
    layers: Sequence[int] = FOLLOWUP_PRIMARY_BAND,
    expected_checksums: Mapping[int, str] = EXPECTED_ARTIFACT_CHECKSUMS,
    scale: int = CORRECTED_SCALE,
    excluded_layer: int = EXCLUDED_FAILED_LAYER,
) -> tuple[dict[int, CorrectedArtifact], dict]:
    """Resolve each band layer's corrected publication from the report itself.

    No filename is guessed. The chain, per layer:

    1. the report's ``publication`` block lists the layer and carries a
       checksum for it;
    2. exactly one ``*.corrected.json`` sidecar in the report's **own**
       publication directory claims that layer, at that scale, ``validated``,
       with :data:`~jlens.mmpilot.band_control.CORRECTED_PUBLISHED_STATUS`;
    3. the sidecar's ``artifact_checksum`` recomputes over its own body;
    4. the lens path comes from that sidecar, resolved inside the publication
       directory so a report copied between mounts still works;
    5. the file's own sha256 equals the sidecar's ``lens_checksum``, the
       report's published checksum, **and** ``expected_checksums[layer]``;
    6. the sidecar agrees with every other layer's on scale, universe
       checksum, protocol digest, confirmation manifest, gate digest, capture
       geometry and estimator, and records text-only calibration.

    ``excluded_layer`` is refused categorically before anything is read: there
    is no argument, threshold or configuration that puts it in the band.

    Raises:
        FollowupRefused: On any broken link, duplicate, mismatch, absent
            artifact, or mixed provenance.
    """
    wanted = sorted(int(layer) for layer in layers)
    if int(excluded_layer) in wanted:
        raise FollowupRefused(
            f"layer {excluded_layer} was requested for the follow-up band. "
            + EXCLUDED_LAYER_REASON
        )
    if wanted != sorted(int(k) for k in expected_checksums):
        raise FollowupRefused(
            f"the requested band {wanted} is not the band the pinned artifact "
            f"checksums describe {sorted(int(k) for k in expected_checksums)}"
        )
    assert_contiguous(wanted, what="corrected artifact band")

    root = Path(corrected_run_dir)
    published_dir = (
        root / "artifacts" / CORRECTED_VALIDATION_DIRNAME / "published"
    )
    if not published_dir.is_dir():
        raise FollowupRefused(
            f"the corrected run's publication directory {published_dir} does not "
            "exist; the follow-up resolves artifacts from the report's own "
            "publication directory and never from a guessed path"
        )
    publication = dict(report.get("publication") or {})
    published_layers = {int(v) for v in publication.get("published_layers", [])}
    report_checksums = {
        str(k): str(v)
        for k, v in (publication.get("published_checksums") or {}).items()
    }
    if int(excluded_layer) in published_layers:
        raise FollowupRefused(
            f"the corrected report publishes layer {excluded_layer}, which "
            "contradicts the frozen result that it failed"
        )

    sidecars = sorted(published_dir.glob("*.corrected.json"))
    artifacts: dict[int, CorrectedArtifact] = {}
    problems: list[str] = []
    geometry: dict[str, set] = {name: set() for name in _GEOMETRY_FIELDS}
    geometry["capture_geometry"] = set()
    geometry["estimator"] = set()

    for layer in wanted:
        if layer not in published_layers:
            problems.append(
                f"L{layer}: the corrected report does not publish it "
                f"(published: {sorted(published_layers)})"
            )
            continue
        candidates = []
        for sidecar in sidecars:
            payload = _read_json(sidecar, what="corrected artifact sidecar")
            if (
                payload.get("schema") == CORRECTED_ARTIFACT_SCHEMA
                and int(payload.get("physical_layer", -1)) == layer
                and int(payload.get("scale_point", -1)) == int(scale)
                and payload.get("validated") is True
                and payload.get("publication_status") == CORRECTED_PUBLISHED_STATUS
            ):
                candidates.append((sidecar, payload))
        if len(candidates) != 1:
            problems.append(
                f"L{layer}: {len(candidates)} corrected sidecars in "
                f"{published_dir} claim it at scale {scale}; which artifact a "
                "band rests on is not decided by sort order"
            )
            continue
        sidecar_path, artifact = candidates[0]
        recomputed = payload_checksum(
            {k: v for k, v in artifact.items() if k != "artifact_checksum"}
        )
        if recomputed != artifact.get("artifact_checksum"):
            problems.append(
                f"L{layer}: {sidecar_path.name} does not match its own "
                "artifact_checksum"
            )
            continue
        lens_path = published_dir / Path(str(artifact.get("lens_path", ""))).name
        if not lens_path.is_file():
            problems.append(
                f"L{layer}: {sidecar_path.name} names a lens absent from "
                f"{published_dir}"
            )
            continue
        actual = file_sha256(str(lens_path))
        expected = str(expected_checksums[layer])
        if actual != str(artifact.get("lens_checksum")):
            problems.append(
                f"L{layer}: the file hashes to {actual}, the sidecar records "
                f"{artifact.get('lens_checksum')!r}"
            )
            continue
        if actual != report_checksums.get(str(layer)):
            problems.append(
                f"L{layer}: the file hashes to {actual}, the report records "
                f"{report_checksums.get(str(layer))!r}"
            )
            continue
        if actual != expected:
            problems.append(
                f"L{layer}: the file hashes to {actual}, not the pinned {expected}"
            )
            continue
        if artifact.get("calibration_modality") != "text-only":
            problems.append(
                f"L{layer}: calibration_modality is "
                f"{artifact.get('calibration_modality')!r}, not 'text-only'"
            )
            continue
        if artifact.get("multimodal_data_used", False) or artifact.get(
            "spokencoco_used", False
        ):
            problems.append(
                f"L{layer}: the artifact records multimodal data in its "
                "calibration; the selection evidence must be text-only"
            )
            continue
        if not artifact.get("no_lens_was_refitted", False):
            problems.append(f"L{layer}: the artifact does not record no_lens_was_refitted")
            continue
        if not artifact.get("no_threshold_was_changed", False):
            problems.append(
                f"L{layer}: the artifact does not record no_threshold_was_changed"
            )
            continue
        confirmation = dict(artifact.get("confirmation_verdict") or {})
        if not confirmation.get("passed", False):
            problems.append(
                f"L{layer}: the artifact does not record an independent "
                "confirmation pass"
            )
            continue

        for name in _GEOMETRY_FIELDS:
            geometry[name].add(str(artifact.get(name)))
        geometry["capture_geometry"].add(
            payload_checksum(
                {
                    "fixed_control_universe": list(
                        artifact.get("fixed_control_universe") or ()
                    ),
                    "wrong_layer_mapping": dict(
                        artifact.get("wrong_layer_mapping") or {}
                    ),
                    "band_window": list(artifact.get("band_window") or ()),
                }
            )
        )
        geometry["estimator"].add(
            payload_checksum(dict(artifact.get("corrected_control_protocol") or {}))
        )
        artifacts[layer] = CorrectedArtifact(
            layer=layer,
            lens_path=str(lens_path),
            lens_checksum=actual,
            sidecar_path=str(sidecar_path),
            layer_key_in_file=layer,
            scale=int(artifact.get("scale_point", scale)),
            matrix_checksum=artifact.get("matrix_checksum"),
            universe_checksum=artifact.get("universe_checksum"),
            protocol_digest=artifact.get("protocol_digest"),
            confirmation_manifest_checksum=artifact.get(
                "confirmation_manifest_checksum"
            ),
            gate_digest=artifact.get("gate_digest"),
        )

    for name, values in geometry.items():
        if len(values) > 1:
            problems.append(
                f"the band mixes {name}: {sorted(values)}. Layers judged under "
                "different geometry, estimators or populations are not one band."
            )
    scales = sorted({artifact.scale for artifact in artifacts.values()})
    if len(scales) > 1:
        problems.append(
            f"the band mixes scales {scales}; layers fitted at different scales "
            "are not one lens"
        )
    if not problems and sorted(artifacts) != wanted:
        # Unreachable while every skip path above records a problem, and asserted
        # anyway: a band silently short of a layer is the failure mode that would
        # be hardest to notice in a report.
        problems.append(
            f"resolved {sorted(artifacts)} but the band is {wanted}"
        )
    if problems:
        raise FollowupRefused(
            "the corrected publications do not support the L33-L40 follow-up "
            "band:\n  - " + "\n  - ".join(problems)
        )

    evidence = {
        "corrected_run_dir": str(root),
        "published_dir": str(published_dir),
        "layers": wanted,
        "scale": int(scale),
        "excluded_failed_layer": int(excluded_layer),
        "excluded_layer_reason": EXCLUDED_LAYER_REASON,
        "artifacts": {str(k): v.to_dict() for k, v in sorted(artifacts.items())},
        "lens_checksums": {
            str(k): v.lens_checksum for k, v in sorted(artifacts.items())
        },
        "single_scale": sorted({v.scale for v in artifacts.values()}),
        "single_universe_checksum": sorted(geometry["universe_checksum"]),
        "single_protocol_digest": sorted(geometry["protocol_digest"]),
        "single_confirmation_manifest": sorted(
            geometry["confirmation_manifest_checksum"]
        ),
        "single_gate_digest": sorted(geometry["gate_digest"]),
        "text_only_calibration": True,
        "independently_confirmed_on_one_population": True,
        "paths_and_checksums_read_from_the_report": True,
    }
    evidence["discovery_checksum"] = payload_checksum(evidence)
    return artifacts, evidence


# ------------------------------------------------ the completed run is immutable


def corrected_run_digest(corrected_run_dir: str | os.PathLike[str]) -> dict:
    """A checksum over every file in the completed corrected run.

    Taken before the follow-up reads anything and again after the last read, so
    the claim that nothing was modified is a measurement rather than an
    assertion. Paths are recorded relative to the run root, so a digest taken on
    one mount compares against one taken on another.
    """
    root = Path(corrected_run_dir)
    if not root.is_dir():
        raise FollowupRefused(
            f"the completed corrected run {root} is not a directory; it is the "
            "selection evidence and must be present to be proved unchanged"
        )
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            files[str(path.relative_to(root)).replace(os.sep, "/")] = file_sha256(
                str(path)
            )
    payload = {
        "run_dir": str(root),
        "run_name": root.name,
        "n_files": len(files),
        "files": files,
    }
    payload["immutability_checksum"] = payload_checksum(payload)
    return payload


def assert_corrected_run_unmodified(before: Mapping, after: Mapping) -> dict:
    """Prove the completed corrected run is byte-for-byte what it was.

    Raises:
        FollowupRefused: On any added, removed or changed file. The follow-up
            reads that run; it never writes into it, and a difference means
            something did.
    """
    files_before = dict(before.get("files") or {})
    files_after = dict(after.get("files") or {})
    added = sorted(set(files_after) - set(files_before))
    removed = sorted(set(files_before) - set(files_after))
    changed = sorted(
        name
        for name in set(files_before) & set(files_after)
        if files_before[name] != files_after[name]
    )
    identical = not (added or removed or changed)
    if not identical:
        raise FollowupRefused(
            "the completed corrected validation run changed while the follow-up "
            f"read it: added={added} removed={removed} changed={changed}. That "
            "run is immutable evidence and this study only ever reads it."
        )
    return {
        "run_name": before.get("run_name"),
        "identical": True,
        "checksum_before": before.get("immutability_checksum"),
        "checksum_after": after.get("immutability_checksum"),
        "n_files": int(before.get("n_files", 0)),
        "added": added,
        "removed": removed,
        "changed": changed,
        "read_only": True,
    }


# --------------------------------------------------------------- the preflight


def followup_preflight_record(
    *,
    report_path: str | os.PathLike[str],
    report: Mapping,
    admission: Mapping,
    discovery: Mapping,
    immutability: Mapping,
    corrected_run_dir: str | os.PathLike[str],
) -> dict:
    """Everything the CPU preflight established, before any model can load."""
    payload = {
        "schema": "jlens.mmpilot.l33_l40_validated_band_followup_preflight.v1",
        "protocol_version": FOLLOWUP_PROTOCOL_VERSION,
        "study_name": FOLLOWUP_STUDY_NAME,
        "corrected_run_dir": str(corrected_run_dir),
        "corrected_run_name": ORIGINAL_RUN_NAME,
        "corrected_report_path": str(report_path),
        "corrected_report_checksum": report.get("report_checksum"),
        "corrected_protocol_digest": (report.get("protocol") or {}).get(
            "protocol_digest"
        ),
        "corrected_universe_checksum": (report.get("protocol") or {}).get(
            "universe_checksum"
        ),
        "confirmation_manifest_checksum": report.get("confirmation_manifest_checksum"),
        "admission": dict(admission),
        "artifact_discovery": dict(discovery),
        "corrected_run_immutability": dict(immutability),
        "original_l32_l40_verdict": ORIGINAL_VERDICT,
        "original_stage3_remained_blocked": True,
        "followup_band": list(FOLLOWUP_PRIMARY_BAND),
        "selection_source": "text-only corrected lens validation only",
        "no_causal_outcome_selected_the_band": True,
        "no_fitting_will_occur": True,
        "no_threshold_was_changed": True,
        "no_lens_was_refitted": True,
        "reporting_boundary": list(REPORTING_BOUNDARY),
    }
    payload["preflight_checksum"] = payload_checksum(payload)
    return payload


def format_followup_preflight(preflight: Mapping) -> str:
    """The block printed before any model is loaded, and sent back afterwards."""
    discovery = dict(preflight.get("artifact_discovery") or {})
    checksums = dict(discovery.get("lens_checksums") or {})
    lines = [
        "=" * 78,
        f"PREFLIGHT — {preflight['study_name']}",
        "=" * 78,
        f"  corrected run          {preflight['corrected_run_name']}",
        f"  corrected report       {preflight['corrected_report_path']}",
        f"  report checksum        {preflight['corrected_report_checksum']}",
        f"  protocol digest        {preflight['corrected_protocol_digest']}",
        f"  universe checksum      {preflight['corrected_universe_checksum']}",
        f"  confirmation manifest  {preflight['confirmation_manifest_checksum']}",
        f"  run unchanged          "
        f"{(preflight.get('corrected_run_immutability') or {}).get('identical')}",
        "",
        f"  ORIGINAL L32-L40 VERDICT REMAINS       {preflight['original_l32_l40_verdict']}",
        f"  new follow-up band                     L{FOLLOWUP_BAND_START}-L{FOLLOWUP_BAND_END}",
        f"  selection source                       {preflight['selection_source']}",
        "  no causal outcome selected the band    "
        f"{preflight['no_causal_outcome_selected_the_band']}",
        f"  no fitting will occur                  {preflight['no_fitting_will_occur']}",
        "",
        f"  {'layer':>5}  {'scale':>6}  checksum",
    ]
    for layer in sorted(int(key) for key in checksums):
        row = dict((discovery.get("artifacts") or {}).get(str(layer)) or {})
        lines.append(
            f"  {layer:>5}  {row.get('scale', '-'):>6}  {checksums[str(layer)]}"
        )
    lines += [
        "",
        f"  excluded failed layer  L{EXCLUDED_FAILED_LAYER}",
        f"    {EXCLUDED_LAYER_REASON}",
    ]
    return "\n".join(lines)


# ------------------------------------------------------------------ the design


def followup_design_record(
    *,
    primary_band,
    suffix_bands: Sequence,
    admission: Mapping,
    discovery: Mapping,
    position_rule: str = PRIMARY_POSITION_RULE,
    alphas: Sequence[float] = (PRIMARY_ALPHA, SECONDARY_ALPHA),
    conditions: Sequence[str] = BAND_CONDITIONS,
    arms: Sequence[str] = BAND_ARMS,
    suffix_starts: Sequence[int] = FOLLOWUP_SUFFIX_STARTS,
) -> dict:
    """The follow-up design, frozen and checksummed before any model result.

    Raises:
        FollowupRefused: If the primary band is not exactly
            :data:`FOLLOWUP_PRIMARY_BAND`, if the suffix starts are not exactly
            the predeclared ones, if any band is non-contiguous or is not a
            suffix ending at :data:`FOLLOWUP_BAND_END`, or if a band contains
            :data:`EXCLUDED_FAILED_LAYER`.
    """
    unknown = [name for name in conditions if name not in BAND_CONDITIONS]
    if unknown:
        raise FollowupRefused(f"unknown conditions {unknown}")
    unknown_arms = [name for name in arms if name not in BAND_ARMS]
    if unknown_arms:
        raise FollowupRefused(f"unknown arms {unknown_arms}")

    primary_layers = assert_contiguous(
        tuple(primary_band.layers), what="follow-up primary band"
    )
    if primary_layers != FOLLOWUP_PRIMARY_BAND:
        raise FollowupRefused(
            f"the primary band {list(primary_layers)} is not the validated "
            f"{list(FOLLOWUP_PRIMARY_BAND)}"
        )
    starts = tuple(sorted(int(value) for value in suffix_starts))
    if starts != tuple(sorted(FOLLOWUP_SUFFIX_STARTS)):
        raise FollowupRefused(
            f"the suffix starts {list(starts)} are not the predeclared "
            f"{list(sorted(FOLLOWUP_SUFFIX_STARTS))}. The comparison topology is "
            "frozen before any causal result exists and is never adjusted after "
            "one."
        )
    observed_starts = []
    for band in suffix_bands:
        layers = assert_contiguous(tuple(band.layers), what="predeclared suffix band")
        if int(EXCLUDED_FAILED_LAYER) in layers:
            raise FollowupRefused(
                f"band {list(layers)} contains L{EXCLUDED_FAILED_LAYER}. "
                + EXCLUDED_LAYER_REASON
            )
        if layers[-1] != FOLLOWUP_BAND_END:
            raise FollowupRefused(
                f"band {list(layers)} does not end at L{FOLLOWUP_BAND_END}; the "
                "predeclared topology is suffix bands of one common end"
            )
        if set(layers) - set(primary_layers):
            raise FollowupRefused(
                f"band {list(layers)} leaves the validated primary band "
                f"{list(primary_layers)}"
            )
        observed_starts.append(int(layers[0]))
    if tuple(sorted(observed_starts)) != starts:
        raise FollowupRefused(
            f"the built bands start at {sorted(observed_starts)}, not the "
            f"predeclared {list(starts)}"
        )

    payload = {
        "version": FOLLOWUP_DESIGN_VERSION,
        "study_name": FOLLOWUP_STUDY_NAME,
        "protocol_version": FOLLOWUP_PROTOCOL_VERSION,
        "study_family": FOLLOWUP_INTERVENTION_FAMILY,
        "trial_record_family": BAND_INTERVENTION_FAMILY,
        "primary_band": primary_band.to_dict(),
        "suffix_bands": [band.to_dict() for band in suffix_bands],
        "band_keys": [band_key(band) for band in suffix_bands],
        "band_start_layers": sorted(observed_starts),
        "band_end_layer": int(FOLLOWUP_BAND_END),
        "predeclared_suffix_starts": list(starts),
        "sampled_start_list_is_not_the_patched_layers": (
            f"{list(starts)} are band STARTS. The band beginning at "
            f"{starts[0]} patches every physical layer {list(primary_layers)}; "
            "reading the start list as the patched set would report layers as "
            "patched that no hook ever touched."
        ),
        "every_physical_layer_in_each_band_is_patched": True,
        "hooks_installed_simultaneously_across_the_band": True,
        "coordinates_recomputed_per_layer_from_its_own_activation": True,
        "position_rule": str(position_rule),
        "alphas": [float(value) for value in alphas],
        "alpha_roles": dict(ALPHA_ROLES),
        "alpha_swept_per_sample": False,
        "conditions": list(conditions),
        "arms": list(arms),
        "excluded_failed_layer": int(EXCLUDED_FAILED_LAYER),
        "excluded_layer_reason": EXCLUDED_LAYER_REASON,
        "admission": dict(admission),
        "artifact_discovery_checksum": discovery.get("discovery_checksum"),
        "lens_checksums": dict(discovery.get("lens_checksums") or {}),
        "original_l32_l40_verdict": ORIGINAL_VERDICT,
        "is_the_original_l32_l40_confirmatory_band": False,
        "selection_evidence": SELECTION_EVIDENCE_STATEMENT,
        "reporting_boundary": list(REPORTING_BOUNDARY),
    }
    return {**payload, "design_digest": payload_checksum(payload)}


def followup_fingerprint(
    *,
    design: Mapping,
    preflight: Mapping,
    lens_checksums: Mapping[int, str],
    model_repo_id: str,
    model_revision: str,
    processor_revision: str,
    transformers_version: str,
    audio_protocol_fingerprint: str | None,
    prompt_protocol: Sequence[Mapping] | Mapping | None,
    candidate_token_ids: Mapping,
    directed_pairs: Sequence[Mapping],
    population: Mapping,
    exclusion: Mapping,
    thresholds: Mapping,
    seeds: Mapping,
    readout_arms: Sequence[str],
    scoring_rule: str,
    coordinate_swap_method_version: str = COORDINATE_SWAP_METHOD_VERSION,
    verdict_version: str = FOLLOWUP_VERDICT_VERSION,
) -> dict:
    """Everything a follow-up result is bound to. A change refuses a resume.

    The selection evidence is bound in as tightly as the design: the corrected
    report's path and checksum, its protocol and universe digests, the
    confirmation-manifest checksum, and each of the eight artifact checksums. A
    follow-up resumed against a different selection is a different study, and
    the store refuses it rather than mixing units.

    Raises:
        FollowupRefused: If any band layer has no recorded lens checksum.
    """
    checksums = {int(layer): str(value) for layer, value in lens_checksums.items()}
    bands = [list(band["layers"]) for band in design["suffix_bands"]]
    missing = sorted(
        {layer for band in bands for layer in band if int(layer) not in checksums}
    )
    if missing:
        raise FollowupRefused(
            f"no lens checksum recorded for band layer(s) {missing}; every "
            "patched layer must name the validated artifact that defined its "
            "coordinates"
        )
    payload = {
        "study_family": FOLLOWUP_INTERVENTION_FAMILY,
        "protocol_version": FOLLOWUP_PROTOCOL_VERSION,
        "study_name": FOLLOWUP_STUDY_NAME,
        "coordinate_swap_method_version": str(coordinate_swap_method_version),
        "design_digest": design["design_digest"],
        # --- the selection evidence
        "corrected_report_path": preflight.get("corrected_report_path"),
        "corrected_report_checksum": preflight.get("corrected_report_checksum"),
        "corrected_protocol_digest": preflight.get("corrected_protocol_digest"),
        "corrected_universe_checksum": preflight.get("corrected_universe_checksum"),
        "confirmation_manifest_checksum": preflight.get(
            "confirmation_manifest_checksum"
        ),
        "artifact_checksums": {
            str(layer): checksums[layer] for layer in sorted(checksums)
        },
        # --- the design
        "primary_band": list(design["primary_band"]["layers"]),
        "ordered_bands": bands,
        "band_keys": list(design["band_keys"]),
        "predeclared_suffix_starts": list(design["predeclared_suffix_starts"]),
        "position_rule": design["position_rule"],
        "alphas": list(design["alphas"]),
        "alpha_roles": dict(design["alpha_roles"]),
        "conditions": list(design["conditions"]),
        "arms": list(design["arms"]),
        "readout_arms": [str(name) for name in readout_arms],
        # --- the model and the protocol pins
        "model_repo_id": str(model_repo_id),
        "model_revision": str(model_revision),
        "processor_revision": str(processor_revision),
        "transformers_version": str(transformers_version),
        "audio_protocol_fingerprint": audio_protocol_fingerprint,
        "prompt_protocol": (
            [dict(item) for item in prompt_protocol]
            if isinstance(prompt_protocol, (list, tuple))
            else (dict(prompt_protocol) if prompt_protocol else None)
        ),
        "candidate_token_ids": {
            str(readout): {
                str(name): list(ids) for name, ids in sorted(dict(mapping).items())
            }
            for readout, mapping in sorted(dict(candidate_token_ids).items())
        },
        # --- the population
        "directed_pairs": [dict(pair) for pair in directed_pairs],
        "population_digest": population.get("population_digest"),
        "population_selection": {
            key: value
            for key, value in dict(population).items()
            if key not in ("groups", "rejected_examples")
        },
        "exclusion": dict(exclusion),
        # --- how it is judged
        "thresholds": dict(thresholds),
        "seeds": dict(seeds),
        "behavioral_scoring_rule": str(scoring_rule),
        "verdict_version": str(verdict_version),
        "no_fitting_entry_point_is_reachable": True,
    }
    return {**payload, "followup_fingerprint_digest": payload_checksum(payload)}


# ------------------------------------------------------------ hook integrity


def assert_band_hook_integrity(
    result: Mapping,
    *,
    band: Sequence[int],
    prompt_len: int,
    expected_forward_passes: int | None = None,
) -> dict:
    """Every requested hook fired, once per scored pass, at the right positions.

    :func:`jlens.mmpilot.band_swap.band_trial_record` already refuses a trial
    whose hooks fired at the wrong *set* of layers. This adds the three things
    that check cannot see:

    * **count** — every band layer fired the same number of times, one per
      scored forward pass. A layer that fired more or fewer times than its
      neighbours was not part of the same simultaneous clamp.
    * **location** — every layer patched exactly the original prompt positions
      ``[0, prompt_len)``, and every layer patched the same ones.
    * **boundary** — no layer touched a position at or beyond ``prompt_len``,
      which is where the teacher-forced candidate tokens live.

    Raises:
        FollowupRefused: On a missing, extra, mislocated or unevenly-fired hook.
    """
    layers = assert_contiguous(band, what="trial band")
    stats = {int(key): dict(value) for key, value in (result.get("layer_stats") or {}).items()}
    problems: list[str] = []

    missing = [layer for layer in layers if layer not in stats]
    extra = sorted(layer for layer in stats if layer not in set(layers))
    if missing:
        problems.append(f"no hook record for band layer(s) {missing}")
    if extra:
        problems.append(f"hooks fired at layer(s) {extra} outside the band {list(layers)}")

    counts = {
        layer: int(stats[layer].get("n_forward_passes") or 0)
        for layer in layers
        if layer in stats
    }
    if counts and len(set(counts.values())) != 1:
        problems.append(
            f"band layers fired unequal numbers of times {counts}; a simultaneous "
            "clamp fires every hook once per scored forward pass"
        )
    fired = sorted(set(counts.values()))
    if fired and fired[0] == 0:
        problems.append(
            f"band layer(s) {[k for k, v in counts.items() if v == 0]} never fired"
        )
    if expected_forward_passes is not None and fired and fired[0] != int(
        expected_forward_passes
    ):
        problems.append(
            f"each hook fired {fired[0]} time(s), expected exactly "
            f"{int(expected_forward_passes)} — one per scored candidate pass"
        )

    wanted_positions = list(range(int(prompt_len)))
    for layer in layers:
        row = stats.get(layer)
        if not row:
            continue
        positions = [int(value) for value in (row.get("positions") or [])]
        if positions != wanted_positions:
            problems.append(
                f"layer {layer} patched positions {positions[:8]}"
                f"{'...' if len(positions) > 8 else ''} rather than every original "
                f"prompt position [0, {int(prompt_len)})"
            )
        beyond = [p for p in positions if p >= int(prompt_len)]
        if beyond:
            problems.append(
                f"layer {layer} patched position(s) {beyond} at or beyond "
                f"prompt_len {int(prompt_len)}; those are teacher-forced "
                "candidate tokens and are never patched"
            )

    if problems:
        raise FollowupRefused(
            "the band clamp did not fire as requested:\n  - " + "\n  - ".join(problems)
        )
    return {
        "band": list(layers),
        "n_hooks": len(layers),
        "forward_passes_per_hook": fired[0] if fired else 0,
        "n_positions_patched": len(wanted_positions),
        "prompt_len": int(prompt_len),
        "candidate_positions_patched": 0,
        "every_requested_hook_fired": True,
        "every_original_prompt_position_patched": True,
        "no_candidate_position_patched": True,
    }


def assert_no_fitting_entry_point(*modules: object) -> dict:
    """Prove no fitting entry point is reachable from the follow-up's namespace.

    The follow-up reads eight validated matrices and fits nothing. This is the
    machine-checkable version of that sentence: the module namespaces it is
    given must expose no calibration or fitting callable at all.

    Raises:
        FollowupRefused: If any forbidden name is bound.
    """
    forbidden = (
        "run_calibration",
        "fit_jacobian_lens",
        "fit_lens",
        "run_fitting",
        "fit_gemma",
    )
    found: list[str] = []
    for module in modules:
        name = getattr(module, "__name__", repr(module))
        for symbol in forbidden:
            if hasattr(module, symbol):
                found.append(f"{name}.{symbol}")
    if found:
        raise FollowupRefused(
            "a fitting entry point is reachable from the follow-up: "
            f"{found}. This study fits nothing; it reads eight already-validated "
            "matrices."
        )
    return {
        "checked_modules": [getattr(m, "__name__", repr(m)) for m in modules],
        "forbidden_symbols": list(forbidden),
        "no_fitting_entry_point_is_reachable": True,
        "backward_passes": 0,
    }


# ------------------------------------------------------------------- budget


def followup_pass_budget(
    *,
    n_pair_concepts: int,
    n_modalities: int,
    n_readouts: int,
    n_candidates_per_readout: int,
    candidate_images_per_concept: int,
    max_analysis_images_per_cell: int,
    n_bands: int,
    n_arms: int,
    n_conditions: int,
    band_layer_counts: Sequence[int],
    expected_clean: int = 576,
    expected_intervention: int = 10_752,
    seconds_per_pass_low: float = 0.9,
    seconds_per_pass_high: float = 2.2,
) -> dict:
    """The exact pass budget, derived from configuration and never hard-coded.

    A *candidate pass* is one teacher-forced scored forward pass. The clean
    screen scores every candidate image in every modality under both readouts;
    the causal stage scores only the capability-eligible cells, once per band,
    arm, condition and readout. Installing eight hooks costs no more forward
    passes than installing one, so a band trial costs what a single-layer trial
    costs.

    ``expected_clean`` / ``expected_intervention`` are the design budget this
    study is meant to retain. They are checked, not used: the returned numbers
    are always the derived ones, and ``matches_expected_design`` is False when
    they disagree so the notebook can stop and name the factor before a model
    loads.
    """
    for name, value in (
        ("n_pair_concepts", n_pair_concepts),
        ("n_modalities", n_modalities),
        ("n_readouts", n_readouts),
        ("n_candidates_per_readout", n_candidates_per_readout),
        ("candidate_images_per_concept", candidate_images_per_concept),
        ("max_analysis_images_per_cell", max_analysis_images_per_cell),
        ("n_bands", n_bands),
        ("n_arms", n_arms),
        ("n_conditions", n_conditions),
    ):
        if int(value) < 1:
            raise FollowupRefused(f"{name} must be at least 1, got {value}")

    clean = (
        int(n_pair_concepts)
        * int(candidate_images_per_concept)
        * int(n_modalities)
        * int(n_readouts)
        * int(n_candidates_per_readout)
    )
    analysis_cells = (
        int(n_pair_concepts) * int(n_modalities) * int(max_analysis_images_per_cell)
    )
    intervention = (
        analysis_cells
        * int(n_bands)
        * int(n_arms)
        * int(n_conditions)
        * int(n_readouts)
        * int(n_candidates_per_readout)
    )
    total = clean + intervention
    matches = clean == int(expected_clean) and intervention == int(
        expected_intervention
    )
    payload = {
        "protocol_version": FOLLOWUP_PROTOCOL_VERSION,
        "clean_candidate_passes": clean,
        "intervention_candidate_passes": intervention,
        "total": total,
        "factors": {
            "clean": {
                "pair_concepts": int(n_pair_concepts),
                "candidate_images_per_concept": int(candidate_images_per_concept),
                "modalities": int(n_modalities),
                "readouts": int(n_readouts),
                "candidates_per_readout": int(n_candidates_per_readout),
            },
            "intervention": {
                "analysis_cells": analysis_cells,
                "bands": int(n_bands),
                "arms": int(n_arms),
                "conditions": int(n_conditions),
                "readouts": int(n_readouts),
                "candidates_per_readout": int(n_candidates_per_readout),
            },
        },
        "expected_clean_candidate_passes": int(expected_clean),
        "expected_intervention_candidate_passes": int(expected_intervention),
        "expected_total": int(expected_clean) + int(expected_intervention),
        "matches_expected_design": bool(matches),
        "hooks_per_trial": [int(value) for value in band_layer_counts],
        "backward_passes": 0,
        "fitting_performed": False,
        "l4_hours_low": round(total * float(seconds_per_pass_low) / 3600.0, 1),
        "l4_hours_high": round(total * float(seconds_per_pass_high) / 3600.0, 1),
        "checkpoint_granularity": (
            "one atomically written, checksum-valid unit per clean sample and "
            "per intervention trial; a disconnect loses at most the trial being "
            "executed"
        ),
        "extrapolation_not_measurement": True,
    }
    payload["budget_checksum"] = payload_checksum(payload)
    return payload


def format_followup_pass_budget(budget: Mapping) -> str:
    """The block printed before any model is loaded."""
    clean = budget["factors"]["clean"]
    intervention = budget["factors"]["intervention"]
    lines = [
        "=" * 78,
        f"PASS BUDGET — {FOLLOWUP_STUDY_NAME} (forward passes only, no fitting)",
        "=" * 78,
        f"  clean candidate passes        {budget['clean_candidate_passes']:,}",
        "    = "
        + " x ".join(
            f"{value} {name}" for name, value in clean.items()
        ),
        f"  intervention candidate passes {budget['intervention_candidate_passes']:,}",
        "    = "
        + " x ".join(
            f"{value} {name}" for name, value in intervention.items()
        ),
        f"  total                         {budget['total']:,}",
        f"  hooks per trial               {budget['hooks_per_trial']}",
        f"  backward passes               {budget['backward_passes']}  "
        f"(fitting_performed={budget['fitting_performed']})",
        f"  expected design total         {budget['expected_total']:,}  "
        f"(matches: {budget['matches_expected_design']})",
        f"  L4 wall time                  {budget['l4_hours_low']:.1f}-"
        f"{budget['l4_hours_high']:.1f} h; an A100 is usually faster",
        f"  checkpoint                    {budget['checkpoint_granularity']}",
        "",
        "  EXTRAPOLATION, NOT MEASUREMENT.",
    ]
    return "\n".join(lines)


# ------------------------------------------------------------------ verdicts

#: The follow-up's verdict names. The mapping is a **re-label of the existing
#: computation** and nothing else: the clauses, the thresholds, the controls and
#: the endpoint are :func:`jlens.mmpilot.band_swap.band_reasoning_verdict`'s,
#: unchanged. They are renamed so no reader can mistake a follow-up result for
#: the completed study's, in either direction.
FOLLOWUP_VERDICT_NAMES: dict[str, str] = {
    "BAND_SWAP_PAPER_COMPARABLE_GO": f"{FOLLOWUP_STUDY_NAME}_GO",
    "BAND_SWAP_ALPHA2_ONLY": f"{FOLLOWUP_STUDY_NAME}_ALPHA2_SENSITIVITY_ONLY",
    "BAND_SWAP_NO_GO": f"{FOLLOWUP_STUDY_NAME}_NULL",
}

#: Returned when the clean behavioural screen did not produce enough eligible
#: cells. It is not a null causal result — no causal trial was run.
FOLLOWUP_CAPABILITY_NO_GO = f"{FOLLOWUP_STUDY_NAME}_CAPABILITY_NO_GO"

#: Returned when the causal stage did not run in this session at all — a
#: CPU-only preflight, for instance. It is neither a null result nor a
#: capability failure, and it must not be printed as either.
FOLLOWUP_NOT_RUN = f"{FOLLOWUP_STUDY_NAME}_NOT_RUN"


def followup_verdict(
    reasoning: Mapping | None,
    *,
    capability_sufficient: bool,
    capability_selection: Mapping | None = None,
    causal_stage_ran: bool = True,
) -> dict:
    """Re-label the existing reasoning verdict under the follow-up's names.

    No clause is evaluated here. ``reasoning`` is
    :func:`jlens.mmpilot.band_swap.band_reasoning_verdict`'s output, produced
    under its own frozen thresholds, and this function decides only what the
    result is *called* and what it is allowed to be read as saying.

    Three non-result states are kept apart, because collapsing them is how a
    CPU-only session comes to print something that reads like a finding:
    :data:`FOLLOWUP_NOT_RUN` (no causal stage in this session),
    :data:`FOLLOWUP_CAPABILITY_NO_GO` (the screen ran and was insufficient, so
    no causal trial was run) and ``..._NULL`` (trials ran and moved nothing).

    Raises:
        FollowupRefused: If a reasoning verdict is offered under a name this
            study has never seen, or if a reasoning verdict is offered at all
            when the causal stage did not run or the capability screen did not
            pass.
    """
    if not causal_stage_ran:
        if reasoning is not None:
            raise FollowupRefused(
                "a reasoning verdict was produced although the causal stage did "
                "not run; a verdict is never assembled from nothing"
            )
        payload = {
            "version": FOLLOWUP_VERDICT_VERSION,
            "study_name": FOLLOWUP_STUDY_NAME,
            "verdict": FOLLOWUP_NOT_RUN,
            "underlying_verdict": None,
            "causal_stage_ran": False,
            "capability_sufficient": None,
            "is_a_null_causal_result": False,
            "is_a_capability_failure": False,
            "why": (
                "the causal stage did not run in this session, so there is no "
                "causal evidence of any kind here — neither positive, nor null, "
                "nor a capability failure"
            ),
            "original_l32_l40_verdict": ORIGINAL_VERDICT,
            "reporting_boundary": list(REPORTING_BOUNDARY),
        }
        return {**payload, "verdict_digest": payload_checksum(payload)}

    if not capability_sufficient:
        if reasoning is not None:
            raise FollowupRefused(
                "a reasoning verdict was produced although the clean capability "
                "screen was insufficient; the causal stage runs only on "
                "predeclared capability-eligible cells"
            )
        payload = {
            "version": FOLLOWUP_VERDICT_VERSION,
            "study_name": FOLLOWUP_STUDY_NAME,
            "verdict": FOLLOWUP_CAPABILITY_NO_GO,
            "underlying_verdict": None,
            "causal_stage_ran": True,
            "capability_sufficient": False,
            "capability_selection": dict(capability_selection or {}),
            "is_a_null_causal_result": False,
            "is_a_capability_failure": True,
            "why": (
                "the clean behavioural screen did not yield the predeclared "
                "minimum of capability-eligible photographs in every cell, so no "
                "causal trial was run. Concepts and samples are not replaced "
                "based on outcomes."
            ),
            "original_l32_l40_verdict": ORIGINAL_VERDICT,
            "reporting_boundary": list(REPORTING_BOUNDARY),
        }
        return {**payload, "verdict_digest": payload_checksum(payload)}

    if reasoning is None:
        raise FollowupRefused(
            "the capability screen passed but no reasoning verdict was produced"
        )
    underlying = str(reasoning.get("verdict"))
    if underlying not in FOLLOWUP_VERDICT_NAMES:
        raise FollowupRefused(
            f"unknown underlying verdict {underlying!r}; this study re-labels "
            f"{sorted(FOLLOWUP_VERDICT_NAMES)} and invents no others"
        )
    payload = {
        "version": FOLLOWUP_VERDICT_VERSION,
        "study_name": FOLLOWUP_STUDY_NAME,
        "verdict": FOLLOWUP_VERDICT_NAMES[underlying],
        "underlying_verdict": underlying,
        "underlying_verdict_version": reasoning.get("version"),
        "underlying_verdict_digest": reasoning.get("verdict_digest"),
        "relabel_only": True,
        "no_threshold_was_changed": True,
        "threshold_digest": reasoning.get("threshold_digest"),
        "causal_stage_ran": True,
        "capability_sufficient": True,
        "capability_selection": dict(capability_selection or {}),
        "paper_comparable": dict(reasoning.get("paper_comparable") or {}),
        "alpha2_sensitivity": dict(reasoning.get("alpha2_sensitivity") or {}),
        "modality_extension": dict(reasoning.get("modality_extension") or {}),
        "tested_bands": list(reasoning.get("tested_bands") or ()),
        "band": list(FOLLOWUP_PRIMARY_BAND),
        "original_l32_l40_verdict": ORIGINAL_VERDICT,
        "is_the_original_l32_l40_confirmatory_band": False,
        "supports_no_claim_about_a_band_beginning_at_l32": True,
        "supports_no_claim_about_layers_earlier_than_33": True,
        "selection_evidence": SELECTION_EVIDENCE_STATEMENT,
        "reporting_boundary": list(REPORTING_BOUNDARY),
    }
    return {**payload, "verdict_digest": payload_checksum(payload)}


#: Timing verdict names for the follow-up. Same re-label discipline: the
#: classification is :func:`jlens.mmpilot.band_swap.band_onset_timing`'s, made
#: over ``deepest_effective_start`` with cross-arm direction matching, and the
#: native direct-readout convergence gate is not used by either.
FOLLOWUP_TIMING_NAMES: dict[str, str] = {
    "BAND_ONSET_INTERMEDIATE_EARLIER": (
        f"{FOLLOWUP_STUDY_NAME}_INTERMEDIATE_CONSUMED_EARLIER"
    ),
    "BAND_ONSET_SAME_TESTED_START": f"{FOLLOWUP_STUDY_NAME}_SAME_TESTED_START",
    "BAND_ONSET_ANSWER_EARLIER": f"{FOLLOWUP_STUDY_NAME}_ANSWER_CONSUMED_EARLIER",
    "BAND_ONSET_INCONCLUSIVE": f"{FOLLOWUP_STUDY_NAME}_TIMING_INCONCLUSIVE",
}


def followup_onset_timing(
    reasoning: Mapping,
    *,
    bands: Sequence[str],
    directed_pairs: Sequence[Mapping],
    modalities: Sequence[str],
    condition: str = "swap_alpha1",
    modality: str = "text",
) -> dict:
    """Intermediate versus answer over the four frozen suffix starts.

    Delegates the whole computation to
    :func:`jlens.mmpilot.band_swap.band_onset_timing` and then adds what this
    protocol must report separately:

    * **each direction on its own**, before any pooled summary, so a strong
      forward direction cannot stand in for a weak reverse one;
    * an explicit statement that nesting alone licenses no monotonicity claim —
      the observed effective starts and their controls are reported exactly as
      measured;
    * an explicit inconclusive/null verdict when no licensed separation exists.
    """
    inner = band_onset_timing(
        reasoning,
        bands=bands,
        directed_pairs=directed_pairs,
        modalities=modalities,
        condition=condition,
        modality=modality,
    )
    underlying = str(inner["verdict"])
    if underlying not in FOLLOWUP_TIMING_NAMES:
        raise FollowupRefused(
            f"unknown underlying timing verdict {underlying!r}; this study "
            f"re-labels {sorted(FOLLOWUP_TIMING_NAMES)} and invents no others"
        )

    per_direction = []
    for row in inner["pairs"]:
        arms = dict(row["effective_band_starts"])
        deepest = dict(row["deepest_effective_start"])
        separated = (
            deepest.get("intermediate") is not None
            and deepest.get("answer") is not None
            and deepest["intermediate"] != deepest["answer"]
        )
        per_direction.append(
            {
                "pair": row["pair"],
                "intermediate_effective_starts": list(arms.get("intermediate") or ()),
                "answer_effective_starts": list(arms.get("answer") or ()),
                "intermediate_deepest_effective_start": deepest.get("intermediate"),
                "answer_deepest_effective_start": deepest.get("answer"),
                "intermediate_earliest_effective_start": row[
                    "earliest_effective_start"
                ].get("intermediate"),
                "answer_earliest_effective_start": row["earliest_effective_start"].get(
                    "answer"
                ),
                "classification": row["classification"],
                "licensed_separation": bool(separated),
                "reported_before_any_pooled_summary": True,
            }
        )

    any_separation = any(row["licensed_separation"] for row in per_direction)
    verdict_name = (
        FOLLOWUP_TIMING_NAMES[underlying]
        if any_separation
        else FOLLOWUP_TIMING_NAMES["BAND_ONSET_INCONCLUSIVE"]
    )
    payload = {
        "version": FOLLOWUP_TIMING_VERSION,
        "study_name": FOLLOWUP_STUDY_NAME,
        "verdict": verdict_name,
        "underlying_verdict": underlying,
        "underlying_version": inner["version"],
        "relabel_only": True,
        "licensed_separation_exists": bool(any_separation),
        "per_direction": per_direction,
        "pooled_summary": {
            "classifications": sorted({row["classification"] for row in per_direction}),
            "reported_after_each_direction": True,
        },
        "condition": str(condition),
        "modality": str(modality),
        "tested_band_starts": list(inner["tested_band_starts"]),
        "tested_bands": list(inner["tested_bands"]),
        "classified_on": inner["classified_on"],
        "why_not_earliest": inner["why_not_earliest"],
        "monotonicity_not_asserted_from_nesting": (
            "the suffix bands are nested by construction, and nesting alone "
            "licenses no monotonicity claim. The observed effective starts and "
            "their intensity-matched controls are reported exactly as measured."
        ),
        "band_starts_are_not_exact_physical_onsets": True,
        "cross_arm_direction_matching_required": True,
        "native_direct_readout_convergence_gate_used": False,
        "source_derived_steering_used": False,
        "original_l32_l40_verdict": ORIGINAL_VERDICT,
        "reporting_boundary": list(REPORTING_BOUNDARY),
    }
    return {**payload, "timing_digest": payload_checksum(payload)}


# -------------------------------------------------------------------- report


def read_followup_units(run_dir) -> tuple[list[dict], dict]:
    """Re-read a completed follow-up run's intervention units, read-only.

    The CPU report/timing stage opens a finished run without a model, a
    processor or any media. Each unit is validated against **its own** recorded
    checksum, and the bands and directed pairs come from that run's report, so
    the analysis is over the design that was frozen rather than over whatever is
    on disk.

    Deliberately separate from :func:`jlens.mmpilot.band_swap.read_band_units`:
    that function opens ``anthropic_band_swap_report.json``, and a follow-up
    directory must never be readable as a completed L32-L40 band run, nor the
    reverse.

    Raises:
        FollowupRefused: If the directory holds no follow-up report, no units,
            or a report from a different study.
    """
    root = Path(run_dir)
    report_path = root / FOLLOWUP_REPORT_NAME
    report = _read_json(report_path, what="follow-up report")
    if report.get("schema") != FOLLOWUP_REPORT_SCHEMA:
        raise FollowupRefused(
            f"{report_path} declares schema {report.get('schema')!r}, not "
            f"{FOLLOWUP_REPORT_SCHEMA!r}"
        )
    if report.get("study_family") != FOLLOWUP_INTERVENTION_FAMILY:
        raise FollowupRefused(
            f"{report_path} belongs to study family "
            f"{report.get('study_family')!r}, not {FOLLOWUP_INTERVENTION_FAMILY!r}"
        )

    records: list[dict] = []
    invalid: list[str] = []
    for path in sorted((root / "units" / "intervention").glob("*.json")):
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
            payload = stored["payload"]
            valid = stored.get("unit_checksum") == payload_checksum(payload)
        except (json.JSONDecodeError, KeyError, TypeError, UnicodeDecodeError):
            valid, payload = False, None
        if not valid:
            invalid.append(str(path))
            continue
        if payload.get("status") == "complete":
            records.append(payload)
    if not records:
        raise FollowupRefused(f"{root} holds no complete follow-up intervention units")
    context = {
        "run_dir": str(root),
        "report_checksum": report.get("report_checksum"),
        "band_keys": list((report.get("design") or {}).get("band_keys") or ()),
        "directed_pairs": list(report.get("directed_pairs") or ()),
        "thresholds": dict(report.get("thresholds") or {}),
        "capability_sufficient": bool(report.get("capability_sufficient", True)),
        "n_units": len(records),
        "n_invalid_units": len(invalid),
        "invalid_units": invalid,
    }
    return records, context


def followup_report(
    *,
    mode: str,
    preflight: Mapping,
    design: Mapping,
    fingerprint: Mapping | None,
    population: Mapping | None,
    exclusion: Mapping | None,
    capability_selection: Mapping | None,
    media_checksums: Mapping | None = None,
    capability_sufficient: bool,
    directed_pairs: Sequence[Mapping],
    band_keys: Sequence[str],
    thresholds: Mapping,
    cells: Sequence[Mapping],
    reasoning: Mapping | None,
    verdict: Mapping,
    timing: Mapping | None,
    budget: Mapping | None = None,
    hook_integrity: Mapping | None = None,
    fitting_audit: Mapping | None = None,
    immutability: Mapping | None = None,
    resume: Mapping | None = None,
    run_dir: str | None = None,
    reanalysis_of: Mapping | None = None,
) -> dict:
    """The follow-up report, including everything it must say about itself."""
    payload = {
        "schema": FOLLOWUP_REPORT_SCHEMA,
        "mode": str(mode),
        "study_name": FOLLOWUP_STUDY_NAME,
        "protocol_version": FOLLOWUP_PROTOCOL_VERSION,
        "study_family": FOLLOWUP_INTERVENTION_FAMILY,
        "trial_record_family": BAND_INTERVENTION_FAMILY,
        "run_dir": run_dir,
        "preflight": dict(preflight),
        "design": dict(design),
        "fingerprint": dict(fingerprint or {}),
        "population": {
            key: value
            for key, value in dict(population or {}).items()
            if key not in ("groups", "rejected_examples")
        },
        "population_groups": [
            {
                "group_id": row.get("group_id"),
                "image_id": row.get("image_id"),
                "concept": row.get("concept"),
                "image_path": row.get("image_path"),
                "audio_path": row.get("audio_path"),
            }
            for row in (population or {}).get("groups", ())
        ],
        "exclusion": dict(exclusion or {}),
        # One entry per photograph/recording cell, keyed by group and modality:
        # the bytes the processor actually saw, not the path they came from.
        # Recorded rather than bound — see the notebook's clean-screen cell.
        "media_checksums": dict(media_checksums or {}),
        "capability_selection": dict(capability_selection or {}),
        "capability_sufficient": bool(capability_sufficient),
        "directed_pairs": [dict(pair) for pair in directed_pairs],
        "band_keys": list(band_keys),
        "thresholds": dict(thresholds),
        "cells": [dict(row) for row in cells],
        "reasoning_verdict": dict(reasoning) if reasoning else None,
        "followup_verdict": dict(verdict),
        "timing": dict(timing) if timing else None,
        "budget": dict(budget or {}),
        "hook_integrity": dict(hook_integrity or {}),
        "fitting_audit": dict(fitting_audit or {}),
        "corrected_run_immutability": dict(immutability or {}),
        "resume": dict(resume or {}),
        "reanalysis_of": dict(reanalysis_of or {}) or None,
        "method_statement": (
            "one exact two-coordinate exchange at every physical layer of a "
            "contiguous band, all hooks installed simultaneously, coordinates "
            "recomputed at each layer from that layer's own activation and its "
            "own validated J-lens vectors, applied at every original prompt "
            "position and never at a teacher-forced candidate token"
        ),
        # --- the reporting boundary, in the report itself
        "band_selected_after_lens_validation": True,
        "is_a_prospective_causal_followup": True,
        "is_the_original_l32_l40_confirmatory_band": False,
        "original_l32_l40_verdict": ORIGINAL_VERDICT,
        "original_l32_l40_run": ORIGINAL_RUN_NAME,
        "original_stage3_remained_blocked": True,
        "excluded_failed_layer": int(EXCLUDED_FAILED_LAYER),
        "excluded_layer_reason": EXCLUDED_LAYER_REASON,
        "supports_no_claim_about_a_band_beginning_at_l32": True,
        "supports_no_claim_about_layers_earlier_than_33": True,
        "spokencoco_tests_linguistic_spoken_captions_not_environmental_sound": True,
        "model_outputs_text_image_and_audio_are_input_modalities": True,
        "alpha2_is_sensitivity_not_primary_evidence": True,
        "selection_evidence": SELECTION_EVIDENCE_STATEMENT,
        "reporting_boundary": list(REPORTING_BOUNDARY),
        "no_lens_was_refitted": True,
        "no_threshold_was_changed": True,
        "completed_corrected_run_read_or_modified": "read-only",
        "mock_proves_pipeline_only": str(mode) != "real",
        "public_method_reference": (
            "https://transformer-circuits.pub/2026/workspace/index.html"
            "#technical-details-of-j-lens-use-cases"
        ),
    }
    payload["report_checksum"] = payload_checksum(payload)
    return payload


def format_followup_verdict(verdict: Mapping, timing: Mapping | None = None) -> str:
    """The verdict block the operator sends back."""
    lines = [
        "=" * 78,
        f"{verdict['study_name']} — {verdict['verdict']}",
        "=" * 78,
        f"  underlying verdict     {verdict.get('underlying_verdict')}",
        f"  causal stage ran       {verdict.get('causal_stage_ran')}",
        f"  capability sufficient  {verdict.get('capability_sufficient')}",
    ]
    if verdict.get("why"):
        lines.append(f"  why                    {verdict['why']}")
    paper = dict(verdict.get("paper_comparable") or {})
    alpha2 = dict(verdict.get("alpha2_sensitivity") or {})
    if paper:
        lines += [
            f"  alpha=1 primary bands  {paper.get('passing_bands')}",
            f"  alpha=2 sensitivity    {alpha2.get('passing_bands')}  "
            "(sensitivity evidence, not interchangeable with alpha=1)",
        ]
    if timing:
        lines += ["", f"  timing                 {timing['verdict']}"]
        for row in timing["per_direction"]:
            lines.append(
                f"    {row['pair']:<16} intermediate deepest="
                f"{row['intermediate_deepest_effective_start']}  answer deepest="
                f"{row['answer_deepest_effective_start']}  "
                f"licensed={row['licensed_separation']}"
            )
    lines += ["", f"  ORIGINAL L32-L40 REMAINS {ORIGINAL_VERDICT}"]
    lines += [f"  {line}" for line in REPORTING_BOUNDARY[:3]]
    return "\n".join(lines)


def _check_frozen_constants() -> None:
    """Import-time self-check on the frozen band constants.

    Raised rather than asserted so it survives ``python -O``: these are the
    constants that decide which layers a hook is ever installed at, and a typo
    in one of them must stop the import, not the run.
    """
    if contiguous_runs(FOLLOWUP_PRIMARY_BAND) != (
        (FOLLOWUP_BAND_START, FOLLOWUP_BAND_END),
    ):
        raise FollowupRefused(
            f"the frozen primary band {list(FOLLOWUP_PRIMARY_BAND)} is not one "
            "contiguous run"
        )
    if int(EXCLUDED_FAILED_LAYER) in FOLLOWUP_PRIMARY_BAND:
        raise FollowupRefused(
            f"L{EXCLUDED_FAILED_LAYER} is inside the frozen primary band. "
            + EXCLUDED_LAYER_REASON
        )
    if not set(FOLLOWUP_PRIMARY_BAND).issubset(set(FIXED_CONTROL_UNIVERSE)):
        raise FollowupRefused(
            "the frozen primary band leaves the corrected control universe "
            f"{list(FIXED_CONTROL_UNIVERSE)}"
        )
    if sorted(EXPECTED_ARTIFACT_CHECKSUMS) != list(FOLLOWUP_PRIMARY_BAND):
        raise FollowupRefused(
            "the pinned artifact checksums do not cover exactly the frozen band"
        )
    if len(set(EXPECTED_ARTIFACT_CHECKSUMS.values())) != len(
        EXPECTED_ARTIFACT_CHECKSUMS
    ):
        raise FollowupRefused("two band layers carry the same pinned checksum")
    if set(FOLLOWUP_SUFFIX_STARTS) - set(FOLLOWUP_PRIMARY_BAND):
        raise FollowupRefused(
            f"suffix starts {list(FOLLOWUP_SUFFIX_STARTS)} leave the primary band"
        )
    if min(FOLLOWUP_SUFFIX_STARTS) != FOLLOWUP_BAND_START:
        raise FollowupRefused(
            "the shallowest predeclared suffix start is not the primary band start"
        )


_check_frozen_constants()
