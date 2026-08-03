# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Everything that can fail cheaply must fail before the 16 GB download.

Each of the wasted L4 starts followed the same shape: an hour of setup, the
model download, and then a ``TypeError`` from a call whose signature could have
been checked in milliseconds without loading anything. This module runs those
checks — lens on disk and matching its pin, manifest naming this exact model
revision, revision resolvable on the Hub, every real-path call site bindable
against the installed signatures, and the selection inputs complete — and
refuses to let the expensive part start until all of them pass.

Nothing here loads model weights. The lens file is read (it is small), the Hub
is asked for repository metadata only, and every signature check is pure
:mod:`inspect`.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

#: A stand-in bound to parameters during signature checks. Never called.
_ARG = object()


class PreflightError(RuntimeError):
    """The real path would fail; the download must not start."""


def _hub_model_info(repo_id: str, revision: str, token: str | None):
    """Import hook: Hub metadata only, never weights. Patched in tests."""
    from huggingface_hub import HfApi

    return HfApi().model_info(repo_id, revision=revision, token=token)


# ----------------------------------------------------------- call contracts


def _call_contracts() -> list[tuple[str, Any, tuple, dict]]:
    """Every external call the real path makes, with the exact arguments the
    notebook passes. ``inspect.signature(...).bind`` raises on a renamed
    keyword, a removed parameter, or a new required one — which is precisely
    the class of drift that has been reaching the L4 instead of CI.
    """
    from jlens.gemma4 import load_gemma4
    from jlens.mmpilot.backend import (
        GemmaPilotBackend,
        resolve_processor_interface,
        run_invariance_gate,
    )
    from jlens.mmpilot.independence import (
        resolve_image_identity,
        summarize_interventions_by_image,
    )
    from jlens.mmpilot.jspace import validate_lens
    from jlens.mmpilot.pipeline import (
        available_modalities,
        build_dictionaries,
        scientific_fingerprint,
        stage_activations,
        stage_capability,
        stage_causal,
        stage_codes,
        stage_directions,
        stage_representational,
    )
    from jlens.mmpilot.real_backend import build_real_backend, load_validated_lens
    from jlens.mmpilot.robustness import (
        estimate_model_passes,
        render_report,
        robustness_verdict,
    )
    from jlens.mmpilot.store import RunFingerprint, UnitStore

    return [
        (
            "load_gemma4",
            load_gemma4,
            (_ARG,),
            {
                "revision": _ARG,
                "dtype": _ARG,
                "device_map": _ARG,
                "allow_model_load": True,
                "token": _ARG,
            },
        ),
        (
            "build_real_backend",
            build_real_backend,
            (_ARG,),
            {
                "revision": _ARG,
                "token": _ARG,
                "device": _ARG,
                "allow_model_load": True,
            },
        ),
        (
            "GemmaPilotBackend",
            GemmaPilotBackend,
            (_ARG, _ARG, _ARG),
            {"device": _ARG},
        ),
        (
            "resolve_processor_interface",
            resolve_processor_interface,
            (_ARG, _ARG),
            {},
        ),
        ("run_invariance_gate", run_invariance_gate, (_ARG, _ARG, _ARG), {}),
        (
            "load_validated_lens",
            load_validated_lens,
            (_ARG,),
            {"expect_checksum": _ARG, "layers": _ARG, "model_revision": _ARG},
        ),
        (
            "validate_lens",
            validate_lens,
            (_ARG,),
            {
                "lens_path": _ARG,
                "lens_checksum": _ARG,
                "layers": _ARG,
                "model_repo_id": _ARG,
                "model_revision": _ARG,
                "expect_model_repo_id": _ARG,
                "expect_model_revision": _ARG,
                "expect_d_model": _ARG,
                "expect_checksum": _ARG,
            },
        ),
        (
            "RunFingerprint",
            RunFingerprint,
            (),
            {
                "mode": _ARG,
                "model_repo_id": _ARG,
                "model_revision": _ARG,
                "processor_revision": _ARG,
                "layers": _ARG,
                "lens_checksum": _ARG,
                "manifest_checksum": _ARG,
                "split_id": _ARG,
                "intervention_config": _ARG,
                "selection_config": _ARG,
            },
        ),
        ("UnitStore", UnitStore, (_ARG, _ARG), {}),
        ("available_modalities", available_modalities, (_ARG, _ARG), {}),
        (
            "scientific_fingerprint",
            scientific_fingerprint,
            (_ARG,),
            {
                "ranked_concepts": _ARG,
                "selected_concepts": _ARG,
                "focal_concepts": _ARG,
                "unrelated_controls": _ARG,
                "derived_cache_fingerprint": _ARG,
                "split_provenance_checksum": _ARG,
                "n_train_positive_images": _ARG,
                "n_train_negative_images": _ARG,
                "n_test_positive_images": _ARG,
                "n_test_negative_images": _ARG,
                "verdict_version": _ARG,
            },
        ),
        (
            "stage_capability",
            stage_capability,
            (_ARG, _ARG, _ARG, _ARG, _ARG),
            {"modalities": _ARG},
        ),
        (
            "stage_activations",
            stage_activations,
            (_ARG, _ARG, _ARG, _ARG, _ARG),
            {"modalities": _ARG, "retained_concepts": _ARG, "model_revision": _ARG},
        ),
        (
            "build_dictionaries",
            build_dictionaries,
            (_ARG, _ARG, _ARG),
            {"device": _ARG, "dtype": _ARG, "build_chunk_rows": _ARG},
        ),
        (
            "stage_codes",
            stage_codes,
            (_ARG, _ARG, _ARG, _ARG),
            {"lens_checksum": _ARG},
        ),
        (
            "stage_representational",
            stage_representational,
            (_ARG, _ARG, _ARG, _ARG),
            {"layer": _ARG, "modalities": _ARG},
        ),
        (
            "stage_directions",
            stage_directions,
            (_ARG, _ARG, _ARG, _ARG, _ARG),
            {"concepts": _ARG, "modalities": _ARG, "lens_checksum": _ARG},
        ),
        (
            "stage_causal",
            stage_causal,
            (_ARG, _ARG, _ARG, _ARG, _ARG, _ARG, _ARG, _ARG),
            {
                "concepts": _ARG,
                "modalities": _ARG,
                "all_concepts": _ARG,
                "unrelated_controls": _ARG,
            },
        ),
        (
            "resolve_image_identity",
            resolve_image_identity,
            (_ARG,),
            {},
        ),
        (
            "summarize_interventions_by_image",
            summarize_interventions_by_image,
            (_ARG, _ARG),
            {"group_summary": _ARG},
        ),
        (
            "estimate_model_passes",
            estimate_model_passes,
            (),
            {
                "n_concepts": _ARG,
                "n_focal_concepts": _ARG,
                "modalities": _ARG,
                "n_total_groups": _ARG,
                "n_capability_groups": _ARG,
                "n_targets_per_cell": _ARG,
                "alphas": _ARG,
                "off_diagonal_only": _ARG,
            },
        ),
        (
            "robustness_verdict",
            robustness_verdict,
            (),
            {
                "capability": _ARG,
                "representational": _ARG,
                "interventions": _ARG,
                "selected_concepts": _ARG,
                "focal_concepts": _ARG,
                "unrelated_controls": _ARG,
                "blocked_modalities": _ARG,
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
                "budget": _ARG,
                "resume": _ARG,
                "mode": _ARG,
            },
        ),
    ]


def check_call_contracts() -> list[str]:
    """Bind every notebook call against the installed signature.

    Returns the list of failures (empty when everything binds). Signatures are
    read from the same checked-out commit the notebook will import from, so a
    drifted parameter fails here — on CPU, in CI — instead of after the model
    download.
    """
    failures = []
    for name, target, args, kwargs in _call_contracts():
        try:
            inspect.signature(target).bind(*args, **kwargs)
        except TypeError as exc:
            failures.append(f"{name}: {exc}")
    return failures


# --------------------------------------------------------------- the checks


def real_path_preflight(
    *,
    model_repo_id: str,
    model_revision: str,
    lens_path: str | Path,
    lens_expect_checksum: str,
    layers: tuple[int, ...] | list[int],
    expect_d_model: int,
    selected_concepts: list[str],
    focal_concepts: list[str],
    unrelated_controls: dict,
    subset: dict,
    split_provenance_checksum: str,
    token: str | None = None,
    resolve_hub_revision: bool = True,
) -> dict:
    """Validate the whole real path without loading model weights.

    Returns a report ``{checks: [...], passed: True}`` and prints nothing.

    Raises:
        PreflightError: Listing **every** failed check, not just the first —
            one Colab round trip should surface all of them.
    """
    checks: list[dict] = []
    failures: list[str] = []

    def record(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})
        if not ok:
            failures.append(f"{name}: {detail}")

    # ---- the lens and its manifest bind the model choice
    lens_path = Path(lens_path)
    manifest = None
    if not lens_path.is_file():
        record("lens_file_exists", False, f"{lens_path} does not exist")
    else:
        from jlens.metadata import file_sha256

        checksum = file_sha256(str(lens_path))
        record(
            "lens_checksum_matches_pin",
            checksum == lens_expect_checksum,
            f"file {checksum} vs pinned {lens_expect_checksum}",
        )
        manifest_path = lens_path.parent / "validated_lens_manifest.json"
        if not manifest_path.is_file():
            record("lens_manifest_exists", False, f"{manifest_path} is missing")
        else:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            record("lens_manifest_exists", True, str(manifest_path))
            record(
                "lens_manifest_status",
                manifest.get("status") == "validated_text_only",
                f"status={manifest.get('status')!r}",
            )
            record(
                "lens_manifest_checksum_agrees",
                manifest.get("lens_checksum") == checksum,
                f"manifest={manifest.get('lens_checksum')} file={checksum}",
            )
            record(
                "requested_revision_matches_lens_manifest",
                manifest.get("model_revision") == model_revision,
                f"manifest={manifest.get('model_revision')!r} "
                f"requested={model_revision!r} (the manifest records no repo "
                "id; the revision is what binds lens and model)",
            )
            passing = set(manifest.get("native_readout_layers_passing", []))
            record(
                "layers_natively_validated",
                set(int(x) for x in layers) <= passing,
                f"requested {sorted(int(x) for x in layers)}, validated "
                f"{sorted(passing)}",
            )
        try:
            from jlens.lens import JacobianLens

            lens = JacobianLens.load(str(lens_path))
            missing = [x for x in layers if int(x) not in lens.jacobians]
            record(
                "lens_contains_requested_layers",
                not missing,
                f"missing {missing}" if missing else f"layers {sorted(lens.jacobians)}",
            )
            record(
                "lens_d_model",
                lens.d_model == expect_d_model,
                f"lens d_model {lens.d_model} vs expected {expect_d_model}",
            )
        except Exception as exc:  # noqa: BLE001 - every failure must be listed
            record("lens_loads", False, f"{type(exc).__name__}: {exc}")

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

    # ---- installed signatures must bind the calls the notebook will make
    contract_failures = check_call_contracts()
    record(
        "call_signatures_bind",
        not contract_failures,
        "; ".join(contract_failures) if contract_failures else
        f"{len(_call_contracts())} call sites bound against installed signatures",
    )

    # ---- the selection inputs must be complete before anything is spent
    record(
        "six_concepts_selected",
        len(selected_concepts) == 6,
        f"{len(selected_concepts)} selected: {selected_concepts}",
    )
    record(
        "three_focal_concepts",
        len(focal_concepts) == 3
        and focal_concepts == list(selected_concepts[:3]),
        f"focal {focal_concepts} vs first three of {selected_concepts[:3]}",
    )
    record(
        "unrelated_controls_external",
        set(unrelated_controls) == set(focal_concepts)
        and all(c not in focal_concepts for c in unrelated_controls.values()),
        f"{unrelated_controls}",
    )
    splits = subset.get("splits") or {}
    for split in ("train", "test"):
        rows = splits.get(split) or []
        images = {row["image_id"] for row in rows}
        record(
            f"subset_{split}_image_unique",
            bool(rows) and len(images) == len(rows),
            f"{len(rows)} rows over {len(images)} distinct images",
        )
    record(
        "split_provenance_checksum_present",
        bool(split_provenance_checksum)
        and str(split_provenance_checksum).startswith("sha256:"),
        str(split_provenance_checksum),
    )

    report = {"passed": not failures, "checks": checks}
    if failures:
        raise PreflightError(
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
    "PreflightError",
    "check_call_contracts",
    "format_preflight",
    "real_path_preflight",
]
