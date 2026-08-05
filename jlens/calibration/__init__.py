# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Research-grade, text-only, multilayer J-lens calibration.

This package orchestrates a calibration study; it does **not** reimplement the
Jacobian estimator. :mod:`jlens.calibration.fitting` calls the unmodified
upstream :func:`jlens.fitting.fit` and observes it through its ``on_prompt``
hook. If upstream's estimator is wrong, this package is wrong identically —
which is the intended property.

Read :doc:`docs/research_grade_jlens_methodology` before changing anything here.
The one fact that shapes every module: ``J_l = E[dh_final/dh_l]`` is a
**population mean estimated by a running average**, not a learned parameter.
There is no loss, no optimizer, and no convergence criterion. Increasing the
number of prompts reduces estimator variance and does nothing else.

Two consequences are load-bearing:

* **Scale points nest for free.** The accumulator at prompt 100 *is* the
  100-prompt lens; continuing the same accumulator to 250 and then 1,000 gives
  the later snapshots. The three scale points cost exactly as much as the
  largest confirmed prefix alone.
* **A three-way split is still required**, not for the estimator — which cannot
  overfit — but for the *gate*, which is a decision procedure applied to
  eight layers at three scales and would otherwise be selected on its own
  evaluation data.

Text-only by construction: nothing in this package can reach SpokenCOCO, an
image, an audio waveform, a multimodal activation, or a cross-modal mapping.
"""

from __future__ import annotations

from jlens.calibration.baseline import (
    BASELINE_ARTIFACTS,
    BaselineArtifact,
    baseline_manifest,
    format_baseline_manifest,
)
from jlens.calibration.corpus import (
    SPLIT_PROTOCOL,
    CorpusLeakageError,
    CorpusRecord,
    Partitions,
    assign_bucket,
    audit_leakage,
    build_partitions,
    nested_subset,
    normalize_text,
    normalized_sha256,
    record_id_for,
    simhash64,
)
from jlens.calibration.fitting import (
    CalibrationResult,
    ScaleSnapshot,
    run_calibration,
    snapshot_scales,
)
from jlens.calibration.gate import (
    CALIBRATION_GATE,
    CALIBRATION_VALIDITY_PROTOCOL,
    CalibrationGate,
    InsufficientTargetDiversityError,
    audit_target_diversity,
    evaluate_calibration_layer,
    gate_text,
    select_diverse_validation_prompts,
)
from jlens.calibration.plan import (
    CALIBRATION_LAYERS,
    SCALE_POINTS,
    Budget,
    CapturePlan,
    build_capture_plan,
    estimate_budget,
    normalized_depth,
)
from jlens.calibration.publication import (
    ConfirmationLocked,
    ConfirmationVault,
    PublicationRefused,
    build_artifact,
    publish_layer,
)
from jlens.calibration.report import (
    calibration_report_markdown,
    calibration_report_payload,
)
from jlens.calibration.scale import (
    PLATEAU_RULE,
    PlateauRule,
    compare_scales,
    evaluate_plateau,
    select_scale,
)
from jlens.calibration.state import (
    CALIBRATION_STAGES,
    CalibrationFingerprint,
    CalibrationStore,
    IncompatibleStateError,
)

__all__ = [
    "BASELINE_ARTIFACTS",
    "CALIBRATION_GATE",
    "CALIBRATION_LAYERS",
    "CALIBRATION_STAGES",
    "CALIBRATION_VALIDITY_PROTOCOL",
    "PLATEAU_RULE",
    "SCALE_POINTS",
    "SPLIT_PROTOCOL",
    "BaselineArtifact",
    "Budget",
    "CalibrationFingerprint",
    "CalibrationGate",
    "CalibrationResult",
    "CalibrationStore",
    "CapturePlan",
    "ConfirmationLocked",
    "ConfirmationVault",
    "CorpusLeakageError",
    "CorpusRecord",
    "IncompatibleStateError",
    "InsufficientTargetDiversityError",
    "Partitions",
    "PlateauRule",
    "PublicationRefused",
    "ScaleSnapshot",
    "assign_bucket",
    "audit_leakage",
    "audit_target_diversity",
    "baseline_manifest",
    "build_artifact",
    "build_capture_plan",
    "build_partitions",
    "calibration_report_markdown",
    "calibration_report_payload",
    "compare_scales",
    "estimate_budget",
    "evaluate_calibration_layer",
    "evaluate_plateau",
    "format_baseline_manifest",
    "gate_text",
    "nested_subset",
    "normalize_text",
    "normalized_depth",
    "normalized_sha256",
    "publish_layer",
    "record_id_for",
    "run_calibration",
    "select_diverse_validation_prompts",
    "select_scale",
    "simhash64",
    "snapshot_scales",
]
