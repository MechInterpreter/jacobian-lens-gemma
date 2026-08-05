# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""The confirmation guard and the artifact publication guard.

Two refusals live here, and both are refusals rather than warnings because a
warning that is printed into a 20-hour Colab log is not a control.

**The confirmation set is locked.** :class:`ConfirmationVault` holds the final
prompts and will not release them until a scale selection has been recorded
against it. This makes "the confirmation set influenced nothing" a property of
the code rather than a claim in a document: a caller who wants to peek at
confirmation numbers before choosing a scale has to modify this file, and that
modification changes the fingerprint.

**Publication is earned on the confirmation set.** :func:`publish_layer` refuses
a layer that has no confirmation verdict, refuses one whose confirmation verdict
failed, and refuses to write over the frozen v2 artifact. Passing at a scale
point during the study is not publication; it is a development number.

Fitting completing successfully is not a reason to publish anything. There is no
code path from "the fit finished" to "a lens is validated".
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from jlens.calibration.gate import CALIBRATION_GATE, CalibrationGate
from jlens.calibration.plan import normalized_depth
from jlens.lens import JacobianLens
from jlens.metadata import file_sha256
from jlens.mmpilot.store import payload_checksum

__all__ = [
    "PROTECTED_ARTIFACTS",
    "ConfirmationLocked",
    "ConfirmationVault",
    "PublicationRefused",
    "build_artifact",
    "publish_layer",
]

#: Completed-run artifacts that are never overwritten. The v2 lens is the
#: evidence base for the confirmed robustness and localization runs; writing
#: over it would make those runs unreproducible.
PROTECTED_ARTIFACTS = (
    "runs/text_jlens_early_layer_recalibration_v2/artifacts/lens.validated.pt",
)

ARTIFACT_FORMAT_VERSION = "jlens.calibration.artifact.v1"


class ConfirmationLocked(RuntimeError):
    """The untouched confirmation set was requested before scale selection."""


class PublicationRefused(RuntimeError):
    """A lens was offered for publication that has not earned it."""


# ------------------------------------------------------------------- the vault


@dataclass
class ConfirmationVault:
    """Holds the final-confirmation prompts until the scale choice is frozen.

    Attributes:
        records: The confirmation partition, never exposed until unlocked.
        selection: The recorded scale selection, once :meth:`unlock` is called.
    """

    records: tuple = ()
    selection: dict | None = None
    _opened: bool = field(default=False, repr=False)

    @property
    def locked(self) -> bool:
        return self.selection is None

    def unlock(self, selection: Mapping) -> dict:
        """Record the scale selection and release the confirmation set.

        Args:
            selection: The payload from
                :func:`jlens.calibration.scale.select_scale`. It must state
                that confirmation was not consulted; a selection that cannot
                make that claim is refused.

        Raises:
            ConfirmationLocked: If ``selection`` is not a completed selection.
        """
        if not isinstance(selection, Mapping) or "selected_scale" not in selection:
            raise ConfirmationLocked(
                "unlock() needs the payload from select_scale(); the confirmation "
                "set is released only against a recorded scale choice"
            )
        if not selection.get("confirmation_not_consulted", False):
            raise ConfirmationLocked(
                "the scale selection does not assert confirmation_not_consulted; "
                "refusing to release the confirmation set to a decision that may "
                "already have seen it"
            )
        self.selection = dict(selection)
        return self.selection

    def open(self) -> tuple:
        """The confirmation records.

        Raises:
            ConfirmationLocked: If :meth:`unlock` has not been called. The
                message is the whole point of the class.
        """
        if self.locked:
            raise ConfirmationLocked(
                "the final-confirmation set has not been unlocked. It may not "
                "influence corpus selection, scale selection, optimizer settings, "
                "layer selection, stopping decisions, or threshold changes — so it "
                "is released only after select_scale() has recorded a choice made "
                "without it."
            )
        self._opened = True
        return tuple(self.records)

    def status(self) -> dict:
        return {
            "locked": self.locked,
            "opened": bool(self._opened),
            "n_records": len(self.records),
            "selected_scale": (self.selection or {}).get("selected_scale"),
            "selection_checksum": (self.selection or {}).get("selection_checksum"),
        }


# --------------------------------------------------------------- artifacts


def build_artifact(
    *,
    layer: int,
    scale: int,
    lens_path: str,
    lens_checksum: str,
    d_model: int,
    n_fitting_prompts: int,
    load_info: Mapping,
    corpus_manifest: Mapping,
    capture_plan: Mapping,
    validation_verdict: Mapping,
    confirmation_verdict: Mapping,
    fitting_diagnostics: Mapping,
    environment: Mapping,
    gate: CalibrationGate = CALIBRATION_GATE,
    protocol_version: str,
) -> dict:
    """The metadata record published beside one layer's lens.

    Carries everything needed to re-derive, re-verify, or refuse to trust the
    artifact, including the explicit statement that calibration was text-only.
    """
    splits = corpus_manifest.get("splits", {})
    checksums = splits.get("checksums", {})
    payload = {
        "artifact_format_version": ARTIFACT_FORMAT_VERSION,
        "protocol_version": protocol_version,
        "frozen": True,
        "validated": bool(confirmation_verdict["passed"]),
        # --- identity
        "model_repo_id": load_info.get("model_repo_id"),
        "model_revision": load_info.get("model_revision"),
        "tokenizer_repo_id": load_info.get("tokenizer_repo_id"),
        "tokenizer_revision": load_info.get("tokenizer_revision"),
        # --- geometry
        "physical_layer": int(layer),
        "normalized_depth": normalized_depth(
            layer, n_layers=capture_plan.get("n_layers", 42)
        ),
        "d_model": int(d_model),
        "target_layer": capture_plan.get("target_layer"),
        "hook_site": capture_plan.get("hook", "block_output"),
        "residual_convention": (
            "pre-final-norm residual after attention, MLP, per-layer-embedding "
            "re-injection and layer_scalar; the exact input to block l+1"
        ),
        "vector_orientation": (
            "J_l maps a layer-l residual into the final-layer basis; applied as "
            "residual @ J_l.T, i.e. rows of J_l index final-layer dimensions"
        ),
        "normalization_convention": (
            "readout is lm_head(final_norm(J_l @ h)); scored pre-softcap. Gemma's "
            "30*tanh(x/30) cap is strictly monotonic and does not change rankings."
        ),
        "logit_softcap": capture_plan.get("logit_softcap", 30.0),
        # --- calibration
        "calibration_modality": "text-only",
        "spokencoco_used": False,
        "multimodal_data_used": False,
        "cross_modal_alignment": False,
        "modality_specific_lens": False,
        "corpus_id": corpus_manifest.get("corpus_id"),
        "corpus_revision": corpus_manifest.get("revision"),
        "corpus_revision_status": corpus_manifest.get("revision_status"),
        "corpus_manifest_checksum": corpus_manifest.get("corpus_manifest_checksum"),
        "fit_split_checksum": checksums.get("fit"),
        "validation_split_checksum": checksums.get("validation"),
        "confirmation_split_checksum": checksums.get("confirmation"),
        "n_fitting_prompts": int(n_fitting_prompts),
        "scale_point": int(scale),
        "capture_plan": dict(capture_plan),
        # --- how it was judged
        "gate_digest": gate.digest,
        "gate": gate.to_dict(),
        "validation_protocol": validation_verdict.get("protocol"),
        "confirmation_protocol": confirmation_verdict.get("protocol"),
        "validation_metrics": validation_verdict.get("metrics"),
        "confirmation_metrics": confirmation_verdict.get("metrics"),
        "validation_failed_checks": list(validation_verdict.get("failed_checks", [])),
        "confirmation_failed_checks": list(
            confirmation_verdict.get("failed_checks", [])
        ),
        "target_diversity": confirmation_verdict.get("target_diversity"),
        # --- how it was made
        "estimator": "jlens.fitting.fit (upstream, unmodified)",
        "objective": "not_applicable_estimator_is_a_sample_mean",
        "fitting_diagnostics": dict(fitting_diagnostics),
        "environment": dict(environment),
        # --- the file
        "lens_path": str(lens_path),
        "lens_checksum": lens_checksum,
    }
    payload["artifact_checksum"] = payload_checksum(payload)
    return payload


def _refuse_protected(path: str) -> None:
    candidate = str(path).replace("\\", "/")
    for protected in PROTECTED_ARTIFACTS:
        if candidate.endswith(protected) or protected.endswith(candidate):
            raise PublicationRefused(
                f"{path} is a frozen completed-run artifact. It is the evidence base "
                "for the confirmed robustness and localization runs and is never "
                "overwritten; a new calibration publishes a new path and a new "
                "checksum."
            )


def publish_layer(
    *,
    layer: int,
    scale: int,
    lens: JacobianLens,
    destination: Path | str,
    confirmation_verdict: Mapping,
    validation_verdict: Mapping,
    vault: ConfirmationVault,
    **artifact_fields,
) -> dict:
    """Write one passing layer's lens and its metadata record.

    Raises:
        PublicationRefused: If the confirmation set was never opened, if the
            layer's confirmation verdict failed, if the verdict is for a
            different layer or scale, or if the destination is a protected
            completed-run artifact.
    """
    destination = Path(destination)
    _refuse_protected(str(destination))

    if vault.locked or not vault._opened:
        raise PublicationRefused(
            f"layer {layer} cannot be published: the final-confirmation set was "
            "never opened, so no confirmation number exists. A lens is not "
            "published because fitting completed."
        )
    if int(confirmation_verdict.get("layer", -1)) != int(layer):
        raise PublicationRefused(
            f"confirmation verdict is for layer {confirmation_verdict.get('layer')}, "
            f"not layer {layer}"
        )
    if int(confirmation_verdict.get("scale", -1)) != int(scale):
        raise PublicationRefused(
            f"confirmation verdict is for scale {confirmation_verdict.get('scale')}, "
            f"not scale {scale}"
        )
    if confirmation_verdict.get("stage") != "confirmation":
        raise PublicationRefused(
            f"verdict for layer {layer} is a "
            f"{confirmation_verdict.get('stage')!r} result, not a confirmation "
            "result; publication requires the untouched confirmation set"
        )
    if not confirmation_verdict.get("passed", False):
        raise PublicationRefused(
            f"layer {layer} failed the confirmation gate "
            f"({confirmation_verdict.get('failed_checks')}). Its diagnostics are "
            "kept and it is recorded with validated=false; it is never marked "
            "validated."
        )
    if int(layer) not in set(lens.source_layers):
        raise PublicationRefused(
            f"layer {layer} is not in the fitted lens (fitted: {lens.source_layers})"
        )

    single = JacobianLens(
        jacobians={int(layer): lens.jacobians[int(layer)]},
        n_prompts=lens.n_prompts,
        d_model=lens.d_model,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(f"{destination.name}.tmp.{os.getpid()}")
    single.save(str(tmp))
    os.replace(tmp, destination)

    artifact = build_artifact(
        layer=int(layer),
        scale=int(scale),
        lens_path=str(destination),
        lens_checksum=file_sha256(str(destination)),
        d_model=lens.d_model,
        n_fitting_prompts=lens.n_prompts,
        validation_verdict=validation_verdict,
        confirmation_verdict=confirmation_verdict,
        **artifact_fields,
    )
    sidecar = destination.with_suffix(".json")
    tmp_sidecar = sidecar.with_name(f"{sidecar.name}.tmp.{os.getpid()}")
    tmp_sidecar.write_text(
        __import__("json").dumps(artifact, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    os.replace(tmp_sidecar, sidecar)
    return artifact


def record_failed_layer(
    *,
    layer: int,
    scale: int,
    confirmation_verdict: Mapping,
    validation_verdict: Mapping,
) -> dict:
    """The diagnostics record for a layer that did not pass.

    Failed layers keep every number they produced. They are never marked
    validated, and no lens file is written for them.
    """
    return {
        "artifact_format_version": ARTIFACT_FORMAT_VERSION,
        "physical_layer": int(layer),
        "normalized_depth": normalized_depth(layer),
        "scale_point": int(scale),
        "validated": False,
        "frozen": True,
        "published": False,
        "reason": "did not pass the confirmation gate",
        "validation_failed_checks": list(validation_verdict.get("failed_checks", [])),
        "confirmation_failed_checks": list(
            confirmation_verdict.get("failed_checks", [])
        ),
        "validation_metrics": validation_verdict.get("metrics"),
        "confirmation_metrics": confirmation_verdict.get("metrics"),
        "target_diversity": confirmation_verdict.get("target_diversity"),
    }


def publication_summary(
    published: Sequence[Mapping], failed: Sequence[Mapping]
) -> dict:
    """What the run published, and what it refused to."""
    payload = {
        "n_published": len(published),
        "n_failed": len(failed),
        "published_layers": sorted(int(item["physical_layer"]) for item in published),
        "failed_layers": sorted(int(item["physical_layer"]) for item in failed),
        "published_checksums": {
            str(item["physical_layer"]): item["lens_checksum"] for item in published
        },
        "failed_layers_marked_validated": False,
        "protected_artifacts_untouched": list(PROTECTED_ARTIFACTS),
    }
    payload["summary_checksum"] = payload_checksum(payload)
    return payload
