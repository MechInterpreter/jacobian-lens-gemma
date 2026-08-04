# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""The real path: load the frozen lens for layers under test, and preflight.

Three L4 starts died before the robustness study's preflight existed, every one
of them on a call that could have been bound in milliseconds without loading
anything. :func:`localization_preflight` extends that check to this study's
additional call sites and to the two things only this study has — a frozen
target manifest and a set of layers that are deliberately *not* yet validated —
and refuses to let the 16 GB download start until all of them pass.

Why loading the lens needs its own entry point
----------------------------------------------

:func:`jlens.mmpilot.real_backend.load_validated_lens` requires every requested
layer to appear in the manifest's ``native_readout_layers_passing``. That is
exactly right for the robustness study, which used only layer 38 and had no
business touching a layer the manifest had not certified.

It is wrong here, and not because the rule is too strict. This study's Stage B
*is* the thing that decides whether layers 20, 26 and 32 read anything out, so a
loader that demanded prior certification would make the question unaskable.
:func:`load_lens_for_localization` therefore separates two claims the older
function bundles together:

* **fitted** — the lens has a Jacobian at this layer, so a readout is defined.
  Required for every requested layer; a missing Jacobian is a hard failure.
* **certified** — a held-out native readout has already passed there. Recorded,
  reported, and *not* required.

Everything else the older loader checks — pinned checksum, published manifest,
text-only calibration, matching model revision — is checked here identically.
Layers that are fitted but not certified come back as ``layers_under_test``, and
nothing in this package lets one of them carry a causal claim until
:func:`jlens.mmlocalize.lens_validity.assert_causally_eligible` says so.
"""

from __future__ import annotations

import inspect
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jlens.mmlocalize.layers import (
    LOCALIZATION_LAYERS,
    MODEL_N_LAYERS,
    assert_immutable_layer_set,
)
from jlens.mmlocalize.targets import LOCALIZATION_CONCEPTS

#: A stand-in bound to parameters during signature checks. Never called.
_ARG = object()

#: The hook site every artifact in this lineage was produced at.
EXPECTED_HOOK_SITE = "block_output"


class LocalizationPreflightError(RuntimeError):
    """The real path would fail; the download must not start."""


class LocalizationLensError(RuntimeError):
    """The frozen lens cannot support the layers this study asks about."""


# ------------------------------------------------------------- lens loading


@dataclass
class LocalizationLens:
    """A frozen lens plus what its manifest does and does not certify."""

    lens: Any
    path: str
    checksum: str
    fitted_layers: list[int]
    natively_validated_layers: list[int]
    layers_under_test: list[int]
    manifest: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "lens_path": self.path,
            "lens_checksum": self.checksum,
            "fitted_layers": list(self.fitted_layers),
            "natively_validated_layers": list(self.natively_validated_layers),
            "layers_under_test": list(self.layers_under_test),
            "reading": (
                "layers_under_test are fitted but not previously certified. They "
                "carry no causal claim until the Stage B gate passes them."
            ),
        }


def load_lens_for_localization(
    lens_path: str | Path,
    *,
    expect_checksum: str,
    layers: tuple[int, ...] | list[int],
    model_revision: str,
    expect_d_model: int | None = None,
) -> LocalizationLens:
    """Load the published lens for every requested layer, certified or not.

    Raises:
        LocalizationLensError: Missing lens, checksum mismatch against the pin,
            missing or inconsistent manifest, a calibration that is not
            text-only, a model revision other than the one the manifest names,
            a width mismatch, or a requested layer with no fitted Jacobian.
    """
    from jlens.lens import JacobianLens
    from jlens.metadata import file_sha256

    lens_path = Path(lens_path)
    if not lens_path.is_file():
        raise LocalizationLensError(
            f"the frozen validated lens was not found at {lens_path}. This study "
            "does not fit a lens; point LENS_PATH at the published artifact."
        )
    checksum = file_sha256(str(lens_path))
    if checksum != expect_checksum:
        raise LocalizationLensError(
            f"lens checksum {checksum} != pinned {expect_checksum}; refusing to "
            "use a lens other than the validated one"
        )
    manifest_path = lens_path.parent / "validated_lens_manifest.json"
    if not manifest_path.is_file():
        raise LocalizationLensError(
            f"missing {manifest_path}: a lens without its validation manifest is "
            "an unvalidated lens"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    problems: list[str] = []
    if manifest.get("status") != "validated_text_only":
        problems.append(f"manifest status is {manifest.get('status')!r}")
    if manifest.get("lens_checksum") != checksum:
        problems.append("manifest checksum does not match the lens file")
    if manifest.get("model_revision") != model_revision:
        problems.append(
            f"manifest model revision {manifest.get('model_revision')!r} does not "
            f"match the loaded model revision {model_revision!r}"
        )
    protocol = str(manifest.get("prompt_protocol", ""))
    if "text-only" not in protocol:
        problems.append(
            f"manifest prompt_protocol {protocol!r} does not declare a text-only "
            "calibration"
        )
    if any(
        marker in protocol.lower()
        for marker in ("spokencoco", "coco", "image", "audio", "multimodal")
    ):
        problems.append(f"manifest prompt_protocol {protocol!r} references non-text data")
    if problems:
        raise LocalizationLensError("incompatible published lens: " + "; ".join(problems))

    lens = JacobianLens.load(str(lens_path))
    fitted = sorted(int(layer) for layer in lens.jacobians)
    requested = [int(layer) for layer in layers]
    missing = [layer for layer in requested if layer not in fitted]
    if missing:
        raise LocalizationLensError(
            f"the lens has no fitted Jacobian at layer(s) {missing}; it was fitted "
            f"at {fitted}. A readout is undefined at an unfitted layer, so this is "
            "not something Stage B could test — it is a missing artifact."
        )
    if expect_d_model is not None and int(lens.d_model) != int(expect_d_model):
        raise LocalizationLensError(
            f"lens d_model {lens.d_model} != expected {expect_d_model}"
        )

    certified = sorted(
        int(layer) for layer in manifest.get("native_readout_layers_passing", ())
    )
    return LocalizationLens(
        lens=lens,
        path=str(lens_path),
        checksum=checksum,
        fitted_layers=fitted,
        natively_validated_layers=certified,
        layers_under_test=sorted(layer for layer in requested if layer not in certified),
        manifest=manifest,
    )


# ----------------------------------------------------------- call contracts


def _localization_call_contracts() -> list[tuple[str, Any, tuple, dict]]:
    """Every call this notebook's real path adds on top of the robustness one."""
    from jlens.mmlocalize.lens_validity import (
        assert_causally_eligible,
        evaluate_all_layers,
        evaluate_layer_validity,
        tie_aware_row,
    )
    from jlens.mmlocalize.targets import (
        assert_same_targets_across_layers,
        audit_image_exclusions,
        choose_target_policy,
        completed_run_images,
        freeze_targets,
        target_manifest,
        verify_completed_run,
    )
    from jlens.mmlocalize.verdict import (
        estimate_localization_passes,
        localization_verdict,
        paired_layer_comparison,
        render_report,
    )

    return [
        (
            "load_lens_for_localization",
            load_lens_for_localization,
            (_ARG,),
            {
                "expect_checksum": _ARG,
                "layers": _ARG,
                "model_revision": _ARG,
                "expect_d_model": _ARG,
            },
        ),
        (
            "tie_aware_row",
            tie_aware_row,
            (),
            {
                "sample_index": _ARG,
                "prompt_sha": _ARG,
                "layer": _ARG,
                "variant": _ARG,
                "variant_logits": _ARG,
                "actual_logits": _ARG,
            },
        ),
        ("evaluate_layer_validity", evaluate_layer_validity, (_ARG,), {"layer": _ARG}),
        ("evaluate_all_layers", evaluate_all_layers, (_ARG,), {"layers": _ARG}),
        ("assert_causally_eligible", assert_causally_eligible, (_ARG, _ARG), {}),
        (
            "verify_completed_run",
            verify_completed_run,
            (_ARG,),
            {"expect_fingerprint": _ARG},
        ),
        ("completed_run_images", completed_run_images, (_ARG,), {}),
        (
            "choose_target_policy",
            choose_target_policy,
            (),
            {
                "n_available_fresh_images": _ARG,
                "n_available_fresh_negatives": _ARG,
                "concepts": _ARG,
            },
        ),
        (
            "freeze_targets",
            freeze_targets,
            (),
            {
                "policy": _ARG,
                "source_positive_images": _ARG,
                "source_negative_images": _ARG,
                "target_positive_images": _ARG,
                "target_negative_images": _ARG,
                "completed_run_images": _ARG,
                "concepts": _ARG,
            },
        ),
        (
            "audit_image_exclusions",
            audit_image_exclusions,
            (_ARG,),
            {"completed_run": _ARG, "n_available_images": _ARG},
        ),
        (
            "target_manifest",
            target_manifest,
            (_ARG,),
            {"audit": _ARG, "completed_run": _ARG, "layers": _ARG},
        ),
        (
            "assert_same_targets_across_layers",
            assert_same_targets_across_layers,
            (_ARG, _ARG),
            {},
        ),
        (
            "estimate_localization_passes",
            estimate_localization_passes,
            (),
            {
                "n_concepts": _ARG,
                "modalities": _ARG,
                "n_total_groups": _ARG,
                "n_capability_groups": _ARG,
                "n_layers_captured": _ARG,
                "n_eligible_causal_layers": _ARG,
                "n_targets_per_cell": _ARG,
                "alphas": _ARG,
                "n_validation_prompts": _ARG,
                "recalibration_enabled": _ARG,
            },
        ),
        (
            "paired_layer_comparison",
            paired_layer_comparison,
            (_ARG,),
            {"layers": _ARG, "reference_layer": _ARG, "concepts": _ARG},
        ),
        (
            "localization_verdict",
            localization_verdict,
            (),
            {
                "validity": _ARG,
                "representational": _ARG,
                "interventions": _ARG,
                "target_manifest": _ARG,
                "layers": _ARG,
                "concepts": _ARG,
                "thresholds": _ARG,
            },
        ),
        (
            "render_report",
            render_report,
            (),
            {
                "run_dir": _ARG,
                "verdict": _ARG,
                "validity": _ARG,
                "budget": _ARG,
                "resume": _ARG,
                "mode": _ARG,
            },
        ),
    ]


def check_localization_call_contracts() -> list[str]:
    """Bind every localization call against the installed signature.

    Includes the robustness study's contracts, because this notebook drives the
    same stage functions and inherits the same drift risk.
    """
    from jlens.mmpilot.preflight import check_call_contracts

    failures = list(check_call_contracts())
    for name, target, args, kwargs in _localization_call_contracts():
        try:
            inspect.signature(target).bind(*args, **kwargs)
        except TypeError as exc:
            failures.append(f"{name}: {exc}")
    return failures


# ---------------------------------------------------------------- preflight


def localization_preflight(
    *,
    model_repo_id: str,
    model_revision: str,
    lens_path: str | Path,
    lens_expect_checksum: str,
    layers: tuple[int, ...] | list[int],
    expect_d_model: int,
    expect_n_layers: int = MODEL_N_LAYERS,
    hook_site: str = EXPECTED_HOOK_SITE,
    concepts: list[str] | tuple[str, ...] = LOCALIZATION_CONCEPTS,
    target_manifest: dict,
    completed_run: dict,
    fingerprint_fields: dict,
    runs_root: str | Path,
    min_free_bytes: int = 512 * 1024 * 1024,
    token: str | None = None,
    resolve_hub_revision: bool = True,
) -> dict:
    """Validate the whole real path without loading model weights.

    Returns a report ``{checks: [...], passed: True}`` and prints nothing.

    Raises:
        LocalizationPreflightError: Listing **every** failed check, not just the
            first — one Colab round trip should surface all of them.
    """
    from jlens.mmpilot.preflight import _hub_model_info

    checks: list[dict] = []
    failures: list[str] = []

    def record(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})
        if not ok:
            failures.append(f"{name}: {detail}")

    # ---- the layer set is the predetermined one, in the predetermined order
    try:
        assert_immutable_layer_set(layers)
        record(
            "layer_set_is_immutable",
            True,
            f"{list(LOCALIZATION_LAYERS)} (physical), unchanged",
        )
    except Exception as exc:  # noqa: BLE001 - every failure must be listed
        record("layer_set_is_immutable", False, str(exc))

    # ---- lens identity, and every requested layer actually fitted
    lens_path = Path(lens_path)
    if not lens_path.is_file():
        record("lens_file_exists", False, f"{lens_path} does not exist")
    else:
        try:
            loaded = load_lens_for_localization(
                lens_path,
                expect_checksum=lens_expect_checksum,
                layers=layers,
                model_revision=model_revision,
                expect_d_model=expect_d_model,
            )
            record(
                "lens_identity_and_layers_fitted",
                True,
                f"checksum {loaded.checksum}; fitted {loaded.fitted_layers}; "
                f"already certified {loaded.natively_validated_layers}; "
                f"under test {loaded.layers_under_test}",
            )
            record(
                "lens_d_model",
                int(loaded.lens.d_model) == int(expect_d_model),
                f"{loaded.lens.d_model} vs expected {expect_d_model}",
            )
            record(
                "reference_layer_is_already_certified",
                38 in loaded.natively_validated_layers,
                f"certified layers {loaded.natively_validated_layers}; the "
                "reference layer must be one of them or there is no anchor",
            )
        except Exception as exc:  # noqa: BLE001 - report, do not download
            record(
                "lens_identity_and_layers_fitted",
                False,
                f"{type(exc).__name__}: {exc}",
            )

    record(
        "hook_site_convention",
        hook_site == EXPECTED_HOOK_SITE,
        f"{hook_site!r} (the lens was fitted at the post-block residual)",
    )
    record(
        "model_depth_expectation",
        int(expect_n_layers) == MODEL_N_LAYERS,
        f"{expect_n_layers} decoder blocks; normalized depths are computed from this",
    )

    # ---- the revision must exist on the Hub before 16 GB are attempted
    if resolve_hub_revision:
        try:
            info = _hub_model_info(model_repo_id, model_revision, token)
            record(
                "hub_revision_resolves",
                True,
                f"{model_repo_id}@{model_revision} -> sha {getattr(info, 'sha', '?')}",
            )
        except Exception as exc:  # noqa: BLE001 - report, do not download
            record(
                "hub_revision_resolves",
                False,
                f"{model_repo_id}@{model_revision}: {type(exc).__name__}: {exc}",
            )
    else:
        checks.append(
            {
                "check": "hub_revision_resolves",
                "ok": True,
                "detail": "skipped by caller (no network)",
            }
        )

    # ---- installed signatures must bind every call the real path will make
    contract_failures = check_localization_call_contracts()
    record(
        "call_signatures_bind",
        not contract_failures,
        "; ".join(contract_failures)
        if contract_failures
        else (
            f"{len(_localization_call_contracts())} localization call sites plus "
            "the robustness contracts bound against installed signatures"
        ),
    )

    # ---- the completed run must be the one this study conditions on
    record(
        "completed_run_verified",
        bool(completed_run.get("fingerprint_matches_pin")),
        f"{completed_run.get('run_dir')} fingerprint "
        f"{completed_run.get('fingerprint')} verdict {completed_run.get('verdict')!r}",
    )

    # ---- the targets must be frozen, complete and disjoint before any pass
    audit = target_manifest.get("image_exclusion_audit") or {}
    record(
        "target_manifest_frozen",
        bool(target_manifest.get("target_checksum"))
        and bool(target_manifest.get("frozen_before_any_layer_result")),
        f"checksum {target_manifest.get('target_checksum')}, policy "
        f"{target_manifest.get('policy')}",
    )
    record(
        "target_manifest_covers_every_layer",
        [int(x) for x in (target_manifest.get("layers") or [])]
        == [int(x) for x in layers],
        f"manifest layers {target_manifest.get('layers')} vs requested {list(layers)}",
    )
    record(
        "targets_concepts_match",
        list(target_manifest.get("concepts") or []) == list(concepts),
        f"{target_manifest.get('concepts')} vs {list(concepts)}",
    )
    record(
        "source_target_images_disjoint",
        not audit.get("source_target_overlap"),
        f"overlap {audit.get('source_target_overlap')}",
    )
    record(
        "target_policy_satisfied",
        bool(audit.get("fresh_policy_satisfied", False)),
        f"policy {target_manifest.get('policy')}; overlap with the completed run: "
        f"{audit.get('n_overlap_all')} image(s), of which "
        f"{audit.get('n_overlap_causal_targets')} were its causal targets",
    )

    # ---- the fingerprint must be complete before results are bound to it
    required_fields = (
        "model_repo_id",
        "model_revision",
        "processor_revision",
        "lens_checksum",
        "calibration_protocol",
        "layers",
        "validity_gate_digest",
        "manifest_checksum",
        "target_checksum",
        "source_image_ids",
        "target_image_ids",
        "concepts",
        "prompt_protocol",
        "alphas",
        "controls",
        "pursuit_config",
    )
    absent = [
        name
        for name in required_fields
        if fingerprint_fields.get(name) in (None, "", [], {})
    ]
    record(
        "run_fingerprint_complete",
        not absent,
        f"missing {absent}" if absent else f"{len(required_fields)} fields present",
    )

    # ---- Drive must have room for what the run will write
    try:
        usage = shutil.disk_usage(str(Path(runs_root).parent if not Path(runs_root).is_dir() else runs_root))
        record(
            "storage_available",
            usage.free >= min_free_bytes,
            f"{usage.free / 1e9:.1f} GB free at {runs_root}, need "
            f"{min_free_bytes / 1e9:.2f} GB",
        )
    except OSError as exc:
        record("storage_available", False, f"could not stat {runs_root}: {exc}")

    report = {"passed": not failures, "checks": checks}
    if failures:
        raise LocalizationPreflightError(
            "REAL PATH PREFLIGHT: FAIL — the model download must not start.\n  - "
            + "\n  - ".join(failures)
        )
    return report


def format_preflight(report: dict) -> str:
    lines = ["REAL PATH PREFLIGHT: PASS"]
    lines += [
        f"  [{'ok' if check['ok'] else 'FAIL'}] {check['check']}: {check['detail']}"
        for check in report["checks"]
    ]
    return "\n".join(lines)


__all__ = [
    "EXPECTED_HOOK_SITE",
    "LocalizationLens",
    "LocalizationLensError",
    "LocalizationPreflightError",
    "check_localization_call_contracts",
    "format_preflight",
    "load_lens_for_localization",
    "localization_preflight",
]
