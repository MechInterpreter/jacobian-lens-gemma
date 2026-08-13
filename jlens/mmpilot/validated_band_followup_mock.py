# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""MOCK mode for the L33-L40 validated-band causal follow-up.

Two halves, both built on code that already exists and both designed so a green
run demonstrates the pipeline *distinguishing* outcomes rather than agreeing
with everything.

**The preflight half** writes a synthetic corrected-validation run to a
temporary directory — the report, the eight ``*.corrected.json`` sidecars and
the eight lens files — and runs the **real** preflight over it:
:func:`~jlens.mmpilot.validated_band_followup.read_corrected_validation_report`,
:func:`~jlens.mmpilot.validated_band_followup.assert_followup_band`,
:func:`~jlens.mmpilot.validated_band_followup.discover_corrected_band_lenses`,
:func:`~jlens.mmpilot.validated_band_followup.corrected_run_digest` and
:func:`~jlens.mmpilot.validated_band_followup.assert_corrected_run_unmodified`.
:data:`PREFLIGHT_SCENARIOS` names the damage each variant does, and every one of
them must be refused. The fixture writes ``mode="mock"``, so the real preflight
refuses it outright unless ``require_real_mode=False`` is passed explicitly —
which is itself one of the tested refusals.

**The causal half** builds trial records through the real
:func:`jlens.mmpilot.band_swap.band_trial_record` from synthetic
``run_swap_condition``-shaped results, and scores them with the real
:func:`~jlens.mmpilot.band_swap.summarize_band_cells`,
:func:`~jlens.mmpilot.band_swap.band_reasoning_verdict`,
:func:`~jlens.mmpilot.validated_band_followup.followup_verdict` and
:func:`~jlens.mmpilot.validated_band_followup.followup_onset_timing`.
:data:`CAUSAL_SCENARIOS` commissions six cases with bounded, predeclared
verdicts, including the two the pipeline most needs to get right: a primary rate
that its own intensity-matched controls match (``control_failure``, which must
**not** be a GO) and an alpha=2 effect with no alpha=1 effect
(``alpha2_only``, which must be labelled sensitivity-only).

The notebook's own end-to-end MOCK stage-3 path uses
:func:`jlens.mmpilot.band_swap_mock.run_mock_band_trials` against the real
synthetic transformer in :mod:`jlens.mmpilot.coordinate_swap_mock`, unchanged.
That world has a reasoning layer at 5, so its arms are separable by
construction; the scenarios here are fixtures on the *outcome* and exist to pin
what each outcome is allowed to be called.

**A green MOCK run proves pipeline behaviour and nothing else.** It is not
evidence about Gemma 4, about layers 33-40, about any modality, or about the
workspace hypothesis.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from jlens.metadata import file_sha256
from jlens.mmpilot.band_control import (
    CORRECTED_ARTIFACT_SCHEMA,
    CORRECTED_BAND_NO_GO,
    CORRECTED_PUBLISHED_STATUS,
    CORRECTED_REPORT_SCHEMA,
    CORRECTED_SCALE,
    CORRECTED_VALIDATION_DIRNAME,
    CORRECTION_PROTOCOL_VERSION,
    FIXED_CONTROL_UNIVERSE,
    WRONG_LAYER_MAPPING,
)
from jlens.mmpilot.band_swap import (
    BAND_ARMS,
    BAND_CONDITIONS,
    CONDITION_ALPHA,
    band_key,
    band_trial_record,
)
from jlens.mmpilot.store import payload_checksum
from jlens.mmpilot.validated_band_followup import (
    CORRECTED_REPORT_NAME,
    EXCLUDED_FAILED_LAYER,
    FOLLOWUP_BAND_END,
    FOLLOWUP_CAPABILITY_NO_GO,
    FOLLOWUP_PRIMARY_BAND,
    FOLLOWUP_STUDY_NAME,
    FOLLOWUP_SUFFIX_STARTS,
    ORIGINAL_RUN_NAME,
    ORIGINAL_VERDICT,
)

__all__ = [
    "CAUSAL_SCENARIOS",
    "MOCK_DIRECTED_PAIRS",
    "MOCK_MODALITIES",
    "MOCK_N_IMAGES",
    "PREFLIGHT_SCENARIOS",
    "CausalScenario",
    "PreflightScenario",
    "mock_band_keys",
    "mock_corrected_run",
    "mock_followup_records",
    "mock_swap_result",
]

MOCK_MODALITIES: tuple[str, ...] = ("text", "image", "spoken_audio")
MOCK_N_IMAGES = 4
MOCK_PROMPT_LEN = 12

#: Both directions of the one predeclared pair, so the direction-matching
#: requirement is exercised rather than assumed.
MOCK_DIRECTED_PAIRS: tuple[dict, ...] = (
    {"source": "bird", "target": "cat"},
    {"source": "cat", "target": "bird"},
)

_ANSWERS = {"bird": "two", "cat": "four"}


def mock_band_keys(
    starts: Sequence[int] = FOLLOWUP_SUFFIX_STARTS, end: int = FOLLOWUP_BAND_END
) -> tuple[str, ...]:
    """The follow-up's frozen suffix bands as stored band keys."""
    return tuple(
        band_key(tuple(range(int(start), int(end) + 1))) for start in sorted(starts)
    )


# ------------------------------------------------------ the preflight fixture


@dataclass(frozen=True)
class PreflightScenario:
    """One commissioned preflight case and what the pipeline must do with it."""

    key: str
    label: str
    must_be_refused: bool
    note: str


PREFLIGHT_SCENARIOS: dict[str, PreflightScenario] = {
    "admits_followup": PreflightScenario(
        key="admits_followup",
        label="L33-L40 pass, L32 fails; the follow-up band is admitted",
        must_be_refused=False,
        note=(
            "the shape of the completed corrected run. Nothing about this "
            "fixture is evidence for Gemma"
        ),
    ),
    "l32_passes": PreflightScenario(
        key="l32_passes",
        label="a report in which L32 also passed",
        must_be_refused=True,
        note=(
            "that report would admit the originally planned L32-L40 confirmatory "
            "band, which is a different study with a different predeclaration; "
            "this follow-up refuses it rather than quietly widening"
        ),
    ),
    "different_band": PreflightScenario(
        key="different_band",
        label="a report whose passing band is L34-L40",
        must_be_refused=True,
        note="only exactly L33-L40 admits this follow-up",
    ),
    "mock_mode_report": PreflightScenario(
        key="mock_mode_report",
        label="a MOCK-mode corrected report",
        must_be_refused=True,
        note="a MOCK report selects no band and validates no layer",
    ),
    "missing_sidecar": PreflightScenario(
        key="missing_sidecar",
        label="L36's corrected sidecar is absent",
        must_be_refused=True,
        note="a patched layer that cannot name its artifact is not a measurement",
    ),
    "duplicate_sidecar": PreflightScenario(
        key="duplicate_sidecar",
        label="two sidecars claim L37",
        must_be_refused=True,
        note="which artifact a band rests on is not decided by sort order",
    ),
    "checksum_mismatch": PreflightScenario(
        key="checksum_mismatch",
        label="L39's lens file does not hash to its pinned checksum",
        must_be_refused=True,
        note="a file that is not the published file is not the published file",
    ),
    "mixed_scale": PreflightScenario(
        key="mixed_scale",
        label="L38 was published at scale 100",
        must_be_refused=True,
        note="layers fitted at different scales are not one lens",
    ),
    "mixed_fit_prefix": PreflightScenario(
        key="mixed_fit_prefix",
        label="L35 carries a different protocol digest",
        must_be_refused=True,
        note="a different protocol digest means a different fit prefix and gate",
    ),
    "mixed_confirmation_population": PreflightScenario(
        key="mixed_confirmation_population",
        label="L34 was confirmed on a different population",
        must_be_refused=True,
        note=(
            "layers judged on different populations are not one band; that is "
            "the defect the corrected control was built to repair"
        ),
    ),
    "mixed_capture_geometry": PreflightScenario(
        key="mixed_capture_geometry",
        label="L33 records a different control universe and band window",
        must_be_refused=True,
        note="different capture geometry is a different measurement",
    ),
    "mixed_estimator": PreflightScenario(
        key="mixed_estimator",
        label="L40 records a different corrected-control rule",
        must_be_refused=True,
        note="a different estimator is a different measurement",
    ),
    "mixed_hook_convention": PreflightScenario(
        key="mixed_hook_convention",
        label="the protocol records no hook convention",
        must_be_refused=True,
        note="a readout whose hook site is unrecorded cannot be reproduced",
    ),
    "multimodal_calibration": PreflightScenario(
        key="multimodal_calibration",
        label="L34's artifact records multimodal calibration data",
        must_be_refused=True,
        note="the selection evidence must be text-only",
    ),
    "unpublished_layer": PreflightScenario(
        key="unpublished_layer",
        label="the report does not publish L35",
        must_be_refused=True,
        note="an unpublished layer has no artifact to read coordinates from",
    ),
    "l32_published": PreflightScenario(
        key="l32_published",
        label="the report publishes L32 even though it failed",
        must_be_refused=True,
        note="a failed layer is never publication eligible",
    ),
}

_MOCK_MODEL_REPO_ID = "google/gemma-4-E4B-it"
_MOCK_MODEL_REVISION = "mock-revision"
_MOCK_GATE_DIGEST = "sha256:mock-frozen-gate"
_MOCK_UNIVERSE_CHECKSUM = "sha256:mock-corrected-universe"
_MOCK_CONFIRMATION_MANIFEST = "sha256:mock-corrected-confirmation-manifest"
_MOCK_FIT_PREFIX = "sha256:mock-fit-prefix"
_MOCK_HOOK_SITE = "block_output"


def _mock_protocol(scenario: str) -> dict:
    payload = {
        "schema": "jlens.mmpilot.band_corrected_control_protocol.v1",
        "correction_protocol_version": CORRECTION_PROTOCOL_VERSION,
        "model_repo_id": _MOCK_MODEL_REPO_ID,
        "model_revision": _MOCK_MODEL_REVISION,
        "processor_revision": "mock-processor-revision",
        "transformers_version": "5.13.1",
        "fixed_control_universe": list(FIXED_CONTROL_UNIVERSE),
        "wrong_layer_mapping": {
            str(k): int(v) for k, v in sorted(WRONG_LAYER_MAPPING.items())
        },
        "universe_checksum": _MOCK_UNIVERSE_CHECKSUM,
        "scale": int(CORRECTED_SCALE),
        "target_layer": 41,
        "hook_convention": None if scenario == "mixed_hook_convention" else _MOCK_HOOK_SITE,
        "fit_prefix_checksum": _MOCK_FIT_PREFIX,
        "gate_version": "mock-extension-confirmation-gate",
        "gate_digest": _MOCK_GATE_DIGEST,
        "gate_is_the_frozen_one": True,
        "no_lens_was_refitted": True,
        "no_threshold_was_changed": True,
        "confirmation_population_not_inspected_when_frozen": True,
    }
    payload["protocol_digest"] = payload_checksum(payload)
    return payload


def _mock_sidecar(layer: int, *, scenario: str, protocol_digest: str) -> dict:
    artifact = {
        "schema": CORRECTED_ARTIFACT_SCHEMA,
        "correction_protocol_version": CORRECTION_PROTOCOL_VERSION,
        "frozen": True,
        "validated": True,
        "publication_status": CORRECTED_PUBLISHED_STATUS,
        "physical_layer": int(layer),
        "scale_point": (
            100 if (scenario == "mixed_scale" and layer == 38) else int(CORRECTED_SCALE)
        ),
        "band_window": (
            [30, 40]
            if (scenario == "mixed_capture_geometry" and layer == 33)
            else [32, 40]
        ),
        "lens_path": f"lens.corrected.layer{int(layer)}.scale{int(CORRECTED_SCALE)}.validated.pt",
        "lens_checksum": None,  # filled in once the file exists
        "matrix_checksum": f"sha256:mock-matrix-L{int(layer)}",
        "source_snapshot": "mock_scale250_snapshot",
        "corrected_control_protocol": {
            "version": CORRECTION_PROTOCOL_VERSION,
            "rule": (
                "mock-alternative-rule"
                if (scenario == "mixed_estimator" and layer == 40)
                else "distant_layer_mapping over the fixed control universe"
            ),
        },
        "fixed_control_universe": (
            [8, 14, 20, 26, 32, 33, 34, 35, 36, 37, 38, 39]
            if (scenario == "mixed_capture_geometry" and layer == 33)
            else list(FIXED_CONTROL_UNIVERSE)
        ),
        "wrong_layer_mapping": {
            str(k): int(v) for k, v in sorted(WRONG_LAYER_MAPPING.items())
        },
        "universe_checksum": _MOCK_UNIVERSE_CHECKSUM,
        "protocol_digest": (
            "sha256:mock-other-protocol"
            if (scenario == "mixed_fit_prefix" and layer == 35)
            else protocol_digest
        ),
        "gate_version": "mock-extension-confirmation-gate",
        "gate_digest": _MOCK_GATE_DIGEST,
        "confirmation_verdict": {"passed": True, "failed_checks": [], "metrics": {}},
        "confirmation_manifest_checksum": (
            "sha256:mock-other-population"
            if (scenario == "mixed_confirmation_population" and layer == 34)
            else _MOCK_CONFIRMATION_MANIFEST
        ),
        "no_lens_was_refitted": True,
        "no_threshold_was_changed": True,
        "calibration_modality": (
            "multimodal"
            if (scenario == "multimodal_calibration" and layer == 34)
            else "text-only"
        ),
        "spokencoco_used": scenario == "multimodal_calibration" and layer == 34,
        "multimodal_data_used": scenario == "multimodal_calibration" and layer == 34,
    }
    return artifact


def mock_corrected_run(
    root: str | os.PathLike[str], *, scenario: str = "admits_followup"
) -> dict:
    """Write a synthetic corrected-validation run and return its pins.

    The returned dict carries the ``expected_*`` values the caller passes back
    into the **real** preflight, so the fixture never has to weaken a check to be
    readable: the pins are explicit, and every structural clause is the real one.

    Raises:
        KeyError: On an unknown scenario name.
    """
    if scenario not in PREFLIGHT_SCENARIOS:
        raise KeyError(
            f"unknown preflight scenario {scenario!r}; known: "
            f"{sorted(PREFLIGHT_SCENARIOS)}"
        )
    run_dir = Path(root) / ORIGINAL_RUN_NAME
    corrected_dir = run_dir / "artifacts" / CORRECTED_VALIDATION_DIRNAME
    published = corrected_dir / "published"
    published.mkdir(parents=True, exist_ok=True)

    protocol = _mock_protocol(scenario)

    if scenario == "l32_passes":
        passing = [EXCLUDED_FAILED_LAYER, *FOLLOWUP_PRIMARY_BAND]
        failing: list[int] = []
        admissible = [EXCLUDED_FAILED_LAYER, FOLLOWUP_BAND_END]
    elif scenario == "different_band":
        passing = [layer for layer in FOLLOWUP_PRIMARY_BAND if layer != 33]
        failing = [EXCLUDED_FAILED_LAYER, 33]
        admissible = [34, FOLLOWUP_BAND_END]
    else:
        passing = list(FOLLOWUP_PRIMARY_BAND)
        failing = [EXCLUDED_FAILED_LAYER]
        admissible = [FOLLOWUP_PRIMARY_BAND[0], FOLLOWUP_BAND_END]

    published_layers = list(passing)
    if scenario == "unpublished_layer":
        published_layers = [layer for layer in published_layers if layer != 35]
    if scenario == "l32_published":
        published_layers = [EXCLUDED_FAILED_LAYER, *published_layers]

    # --- the lens files and their sidecars
    checksums: dict[int, str] = {}
    for layer in sorted(set(published_layers)):
        artifact = _mock_sidecar(layer, scenario=scenario, protocol_digest=protocol["protocol_digest"])
        lens_path = published / str(artifact["lens_path"])
        body = f"mock-corrected-lens-L{layer}-scale{CORRECTED_SCALE}\n"
        if scenario == "checksum_mismatch" and layer == 39:
            body = "mock-corrected-lens-L39-TAMPERED\n"
        lens_path.write_text(body, encoding="utf-8")
        checksums[layer] = file_sha256(str(lens_path))
        artifact["lens_checksum"] = checksums[layer]
        artifact["artifact_checksum"] = payload_checksum(artifact)
        if scenario == "missing_sidecar" and layer == 36:
            continue
        sidecar = lens_path.with_suffix(".corrected.json")
        sidecar.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
        if scenario == "duplicate_sidecar" and layer == 37:
            twin = published / "lens.corrected.layer37.duplicate.corrected.json"
            twin.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    # The pins always cover exactly the follow-up band, whatever the fixture
    # published. A scenario that withholds a layer must be refused by the
    # discovery clause it is testing, not by an accidental mismatch between the
    # band and the pin table.
    pinned = {
        layer: checksums.get(layer, "sha256:" + "e" * 63 + str(layer % 10))
        for layer in FOLLOWUP_PRIMARY_BAND
    }
    if scenario == "checksum_mismatch":
        pinned[39] = "sha256:" + "0" * 64

    verdict = {
        "schema": "jlens.mmpilot.band_corrected_control_verdict.v1",
        "verdict": "BAND_CORRECTED_CONTROL_GO" if not failing else CORRECTED_BAND_NO_GO,
        "scale": int(CORRECTED_SCALE),
        "confirmation_manifest_checksum": _MOCK_CONFIRMATION_MANIFEST,
        "band_window": [32, 40],
        "layers_passing": sorted(passing),
        "layers_failing": sorted(failing),
        "publication_eligible_layers": sorted(passing),
        "full_band_available": not failing,
        "largest_admissible_contiguous_band": admissible,
        "all_layers_scored_on_one_population": True,
        "old_and_new_verdicts_combined": False,
        "stage3_unblocked": not failing,
        "layers": [],
    }
    verdict["verdict_checksum"] = payload_checksum(verdict)

    report = {
        "schema": CORRECTED_REPORT_SCHEMA,
        "mode": "mock",
        "correction_protocol_version": CORRECTION_PROTOCOL_VERSION,
        "protocol": protocol,
        "confirmation_manifest_checksum": _MOCK_CONFIRMATION_MANIFEST,
        "band_verdict": verdict,
        "publication": {
            "directory": str(published),
            "n_published": len(published_layers),
            "published_layers": sorted(published_layers),
            "published_checksums": {
                str(layer): checksums[layer] for layer in sorted(published_layers)
            },
            "existing_artifacts_overwritten": False,
        },
        "provenance": {
            "original_result": ORIGINAL_VERDICT,
            "original_run": ORIGINAL_RUN_NAME,
            "no_frozen_numerical_threshold_was_changed": True,
            "no_matrix_was_refitted": True,
            "stage3_unblocked": not failing,
            "old_and_new_verdicts_combined": False,
        },
        "mock_proves_pipeline_only": True,
    }
    report["report_checksum"] = payload_checksum(
        {k: v for k, v in report.items() if k != "report_checksum"}
    )
    path = corrected_dir / CORRECTED_REPORT_NAME
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return {
        "scenario": scenario,
        "run_dir": str(run_dir),
        "report_path": str(path),
        "expected_report_checksum": report["report_checksum"],
        "expected_protocol_digest": protocol["protocol_digest"],
        "expected_universe_checksum": _MOCK_UNIVERSE_CHECKSUM,
        "expected_confirmation_manifest_checksum": _MOCK_CONFIRMATION_MANIFEST,
        "expected_model_repo_id": _MOCK_MODEL_REPO_ID,
        "expected_model_revision": _MOCK_MODEL_REVISION,
        "expected_artifact_checksums": pinned,
        "must_be_refused": PREFLIGHT_SCENARIOS[scenario].must_be_refused,
    }


# --------------------------------------------------------- the causal fixture


@dataclass(frozen=True)
class CausalScenario:
    """One commissioned causal case and the bounded verdict it must produce.

    ``wins`` decides, for one cell, whether the target answer becomes top-1. It
    is a fixture on the *outcome*; every record it produces is then built,
    aggregated and judged by the real pipeline.
    """

    key: str
    label: str
    expected_verdict: str
    expected_timing: str | None
    note: str
    capability_sufficient: bool = True
    wins: object = field(default=None)


def _favorable(*, band_start, arm, condition, readout, pair, modality) -> bool:
    del band_start, pair, modality
    if condition not in ("swap_alpha1", "swap_alpha2"):
        return False
    if readout != "property":
        return False
    return arm in ("intermediate", "answer")


def _null(**_kwargs) -> bool:
    return False


def _control_failure(*, band_start, arm, condition, readout, pair, modality) -> bool:
    del band_start, pair, modality
    if readout != "property" or arm != "intermediate":
        return False
    # The swap moves the answer — and so does every intensity-matched control at
    # the same alpha. A pipeline that calls this a GO is broken.
    return condition in (
        "swap_alpha1",
        "swap_alpha2",
        "zero",
        "random_alpha1",
        "random_alpha2",
        "unrelated_alpha1",
        "unrelated_alpha2",
    )


def _alpha2_only(*, band_start, arm, condition, readout, pair, modality) -> bool:
    del band_start, pair, modality
    if readout != "property" or arm != "intermediate":
        return False
    return condition == "swap_alpha2"


def _asymmetric(*, band_start, arm, condition, readout, pair, modality) -> bool:
    del modality
    if condition not in ("swap_alpha1", "swap_alpha2") or readout != "property":
        return False
    if arm == "answer":
        # The answer coordinates stay editable to the last tested band.
        return True
    # The intermediate identity stops being editable past the shallow bands, and
    # only in one direction — so the pooled summary can never stand in for the
    # per-direction rows.
    return pair["source"] == "bird" and int(band_start) <= 35


CAUSAL_SCENARIOS: dict[str, CausalScenario] = {
    "favorable": CausalScenario(
        key="favorable",
        label="the swap moves the downstream answer and no control does",
        expected_verdict=f"{FOLLOWUP_STUDY_NAME}_GO",
        expected_timing=f"{FOLLOWUP_STUDY_NAME}_TIMING_INCONCLUSIVE",
        note=(
            "both arms remain effective at every tested start, so there is no "
            "licensed separation and the timing verdict is inconclusive even "
            "though the causal verdict is favourable. That pairing is the point: "
            "a GO on the endpoint licenses no ordering claim on its own"
        ),
        wins=_favorable,
    ),
    "null": CausalScenario(
        key="null",
        label="nothing moves anywhere",
        expected_verdict=f"{FOLLOWUP_STUDY_NAME}_NULL",
        expected_timing=f"{FOLLOWUP_STUDY_NAME}_TIMING_INCONCLUSIVE",
        note="the honest empty result; no band, no arm, no alpha",
        wins=_null,
    ),
    "control_failure": CausalScenario(
        key="control_failure",
        label="the swap moves the answer and so does every matched control",
        expected_verdict=f"{FOLLOWUP_STUDY_NAME}_NULL",
        expected_timing=f"{FOLLOWUP_STUDY_NAME}_TIMING_INCONCLUSIVE",
        note=(
            "a high primary rate that the intensity-matched controls match is "
            "not evidence of anything, and the pipeline must return NULL rather "
            "than reporting the raw rate as a result"
        ),
        wins=_control_failure,
    ),
    "alpha2_only": CausalScenario(
        key="alpha2_only",
        label="alpha=2 moves the answer, alpha=1 does not",
        expected_verdict=f"{FOLLOWUP_STUDY_NAME}_ALPHA2_SENSITIVITY_ONLY",
        expected_timing=f"{FOLLOWUP_STUDY_NAME}_TIMING_INCONCLUSIVE",
        note=(
            "alpha=2 is compared against alpha=2-matched controls and is "
            "reported as sensitivity evidence; it is never promoted to the "
            "alpha=1 primary result"
        ),
        wins=_alpha2_only,
    ),
    "asymmetric_direction": CausalScenario(
        key="asymmetric_direction",
        label="bird->cat intermediate stops working past L35; the answer arm does not",
        expected_verdict=f"{FOLLOWUP_STUDY_NAME}_GO",
        expected_timing=f"{FOLLOWUP_STUDY_NAME}_INTERMEDIATE_CONSUMED_EARLIER",
        note=(
            "one direction separates and the other does not; each direction is "
            "reported before any pooled summary, and the separation is licensed "
            "only where both arms carry the same source->target pair"
        ),
        wins=_asymmetric,
    ),
    "capability_no_go": CausalScenario(
        key="capability_no_go",
        label="the clean behavioural screen was insufficient",
        expected_verdict=FOLLOWUP_CAPABILITY_NO_GO,
        expected_timing=None,
        note=(
            "no causal trial ran, so this is not a null causal result and is "
            "never reported as one. Samples and concepts are not replaced to "
            "make a cell sufficient"
        ),
        capability_sufficient=False,
        wins=_null,
    ),
}


def mock_swap_result(
    *,
    band: Sequence[int],
    alpha: float,
    prediction: str,
    clean_prediction: str,
    target_answer: str,
    candidates: Sequence[str],
    prompt_len: int = MOCK_PROMPT_LEN,
    n_candidate_passes: int = 2,
) -> dict:
    """A synthetic result in :func:`run_swap_condition`'s exact output shape.

    Every field :func:`jlens.mmpilot.band_swap.band_trial_record` and
    :func:`~jlens.mmpilot.validated_band_followup.assert_band_hook_integrity`
    read is present and internally consistent, including one ``layer_stats``
    entry per band layer recording every original prompt position and no
    candidate position.
    """
    layers = [int(layer) for layer in band]
    scores = {
        str(name): {
            "sum_logprob": 1.0 if str(name) == str(prediction) else -1.0 - index,
            "n_tokens": 1,
        }
        for index, name in enumerate(candidates)
    }
    positions = list(range(int(prompt_len)))
    return {
        "intervention_family": "anthropic_coordinate_swap",
        "alpha": float(alpha),
        "alpha_is_extrapolation": bool(float(alpha) > 1.0),
        "position_rule": "all_prompt_positions",
        "layer_band": layers,
        "layers_patched": layers,
        "positions_patched": positions,
        "n_positions_patched": len(positions),
        "n_candidate_positions_skipped": 1,
        "clean_target_score": scores[str(target_answer)]["sum_logprob"] - 0.5,
        "target_score": scores[str(target_answer)]["sum_logprob"],
        "clean_target_margin": -0.5 if str(prediction) == str(target_answer) else 0.25,
        "target_margin": 0.5 if str(prediction) == str(target_answer) else -0.25,
        "target_margin_change": 1.0 if str(prediction) == str(target_answer) else -0.5,
        "clean_prediction": str(clean_prediction),
        "prediction": str(prediction),
        "prediction_changed": str(prediction) != str(clean_prediction),
        "candidate_scores": scores,
        "layer_stats": {
            str(layer): {
                "layer": int(layer),
                "alpha": float(alpha),
                "position_rule": "all_prompt_positions",
                "prompt_len": int(prompt_len),
                "n_forward_passes": int(n_candidate_passes),
                "positions": positions,
                "n_positions": len(positions),
                "seq_len": int(prompt_len) + 1,
                "n_candidate_positions_skipped": 1,
            }
            for layer in layers
        },
    }


def mock_followup_records(
    scenario: CausalScenario | str,
    *,
    starts: Sequence[int] = FOLLOWUP_SUFFIX_STARTS,
    end: int = FOLLOWUP_BAND_END,
    modalities: Sequence[str] = MOCK_MODALITIES,
    directed_pairs: Sequence[Mapping] = MOCK_DIRECTED_PAIRS,
    conditions: Sequence[str] = BAND_CONDITIONS,
    n_images: int = MOCK_N_IMAGES,
) -> list[dict]:
    """Every stored trial record a MOCK causal scenario produces.

    Built through the real :func:`jlens.mmpilot.band_swap.band_trial_record`, so
    the record shape, the band-key derivation and the "hooks fired at exactly
    the requested band" refusal are the real ones.
    """
    if isinstance(scenario, str):
        scenario = CAUSAL_SCENARIOS[scenario]
    if not scenario.capability_sufficient:
        return []

    records: list[dict] = []
    for start in sorted(int(value) for value in starts):
        band = tuple(range(start, int(end) + 1))
        for pair in directed_pairs:
            source, target = str(pair["source"]), str(pair["target"])
            for modality in modalities:
                for arm in BAND_ARMS:
                    for condition in conditions:
                        for readout in ("identity", "property"):
                            candidates = (
                                ("bird", "cat")
                                if readout == "identity"
                                else ("two", "four")
                            )
                            source_answer = (
                                source if readout == "identity" else _ANSWERS[source]
                            )
                            target_answer = (
                                target if readout == "identity" else _ANSWERS[target]
                            )
                            won = bool(
                                scenario.wins(
                                    band_start=start,
                                    arm=arm,
                                    condition=condition,
                                    readout=readout,
                                    pair=pair,
                                    modality=modality,
                                )
                            )
                            prediction = target_answer if won else source_answer
                            for index in range(int(n_images)):
                                result = mock_swap_result(
                                    band=band,
                                    alpha=CONDITION_ALPHA[condition],
                                    prediction=prediction,
                                    clean_prediction=source_answer,
                                    target_answer=target_answer,
                                    candidates=candidates,
                                )
                                records.append(
                                    band_trial_record(
                                        result,
                                        band=band,
                                        arm=arm,
                                        condition=condition,
                                        modality=str(modality),
                                        source=source,
                                        target=target,
                                        source_answer=source_answer,
                                        target_answer=target_answer,
                                        readout=readout,
                                        group_id=f"mock-group-{source}-{index}",
                                        image_id=f"mock-image-{source}-{index}",
                                        prompt_hash=f"sha256:mock-prompt-{readout}",
                                    )
                                )
    return records
