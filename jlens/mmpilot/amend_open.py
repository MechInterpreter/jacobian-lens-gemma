# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Reopen one **explicitly named** completed run and amend its report — nothing else.

:mod:`jlens.mmpilot.amend` re-derives the capability-filtered verdicts from a
:class:`~jlens.mmpilot.store.UnitStore` that somebody already has. This module is
what produces that store when nobody does: a CPU session that opens the notebook,
sets the amendment switches, and runs the amendment cell with no model stage
having executed in the kernel.

The bug this exists for: the amendment cell began ``if STORE is None``, but
``STORE`` is only bound inside the Stage-A model path. With the model stages off
it was never defined at all, so the documented CPU procedure died on
``NameError: name 'STORE' is not defined``, and the setup cells meanwhile created
a fresh empty ``mmaudio_<timestamp>`` directory that the amendment had no use
for.

So the contract here is deliberately narrow and entirely explicit:

* the run directory is **given**, never discovered — no newest-directory guess,
  no scan, no "there was only one candidate";
* the directory and its ``fingerprint.json`` must already exist, and a missing
  one is refused rather than created;
* the stored fingerprint is reconstructed field by field and must **re-derive its
  own digest**, so a file this module does not fully understand is refused
  instead of being coerced into a :class:`RunFingerprint` that merely looks right;
* that digest must equal the caller's expected pin;
* the store is opened ``resuming``. A ``starting`` store would mean the units are
  not there, which is not something to amend;
* no model, no processor, no lens and no dataset is loaded, and the completed
  run's own artifacts are read-only. The only files this path may write are the
  separately named amended ones.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import MISSING
from dataclasses import fields as dataclass_fields
from pathlib import Path

from jlens.mmpilot.admissibility import admissibility_rule_record
from jlens.mmpilot.amend import (
    AMENDED_REPORT_NAME,
    AMENDED_SUMMARY_NAME,
    AmendedReportInputMissing,
    build_amended_report,
    verify_amended_binding,
    write_amended_report,
)
from jlens.mmpilot.store import RunFingerprint, UnitStore, payload_checksum
from jlens.mmpilot.tri_modal import TriModalThresholds

#: The completed run's own summary, written by section 18. Read-only here.
ORIGINAL_SUMMARY_NAME = "native_audio_transfer_summary.json"

#: The completed run's own report. Never read, never written, never overwritten.
ORIGINAL_REPORT_NAME = "native_audio_transfer_report.md"

#: Keys a fingerprint file may carry that are *about the file* rather than about
#: the run. Removed before the digest is recomputed, exactly as
#: :meth:`jlens.mmpilot.store.UnitStore.open` removes them.
TRANSPORT_METADATA_KEYS: tuple[str, ...] = ("written_utc",)

#: Switches that spend model passes. Amendment-only mode is mutually exclusive
#: with every one of them: the whole point is that nothing is computed.
MODEL_AND_CAUSAL_SWITCHES: tuple[str, ...] = (
    "RUN_MODEL_STAGES",
    "CONFIRM_MODEL_LOAD",
    "CONFIRM_REPRESENTATION_BUDGET",
    "RUN_L35_CAUSAL_STAGE",
    "CONFIRM_L35_CAUSAL_BUDGET",
    "RUN_L38_L40_REPLICATION",
    "CONFIRM_REPLICATION_BUDGET",
)

#: The report-only values section 18b needs and the model sections would
#: otherwise have left in the kernel.
RESTORED_REPORT_FIELDS: tuple[str, ...] = (
    "lens_report",
    "audio_protocol",
    "invariance",
    "blocked_modalities",
)


class AmendmentModeError(RuntimeError):
    """Amendment-only mode was configured in a way that cannot be honored."""


class ExistingRunNotFound(AmendmentModeError):
    """The named run directory, or its fingerprint, is not there.

    Refused rather than created. Amendment describes a completed run; there is
    nothing to describe about a directory this process just made.
    """


class FingerprintMismatch(AmendmentModeError):
    """A stored fingerprint is not the one the caller pinned, or is unreadable."""


class AmendedArtifactsTorn(AmendmentModeError):
    """Exactly one of the two amended artifacts exists.

    Neither reusing nor regenerating is safe: a lone report has no binding to
    check, and a lone summary claims a report that is not there.
    """


# ----------------------------------------------------------- switch exclusion


def assert_amendment_mode_exclusive(
    switches: Mapping[str, object],
    *,
    amendment_switch: str = "RUN_AMENDED_REPORT_ONLY",
    model_switches: Sequence[str] = MODEL_AND_CAUSAL_SWITCHES,
) -> dict:
    """Refuse amendment-only mode combined with any model or causal switch.

    Returns the observed switch positions when the combination is legal. Called
    from the configuration cell *and* again from the amendment cell, so editing
    one switch and re-running one cell cannot leave the two in disagreement.

    Raises:
        AmendmentModeError: If ``amendment_switch`` is truthy and any switch in
            ``model_switches`` is also truthy.
    """
    observed = {name: bool(switches.get(name)) for name in model_switches}
    amending = bool(switches.get(amendment_switch))
    if not amending:
        return {"amendment_only": False, "model_switches": observed}
    conflicting = sorted(name for name, value in observed.items() if value)
    if conflicting:
        raise AmendmentModeError(
            f"{amendment_switch} is True together with {conflicting}. "
            "Amendment-only mode re-derives a completed run's report from its "
            "stored units on CPU: it loads no model and computes no unit, so a "
            "model or causal switch beside it is a contradiction rather than an "
            "extra. Set one or the other to False and run section 2 again."
        )
    return {"amendment_only": True, "model_switches": observed}


# -------------------------------------------------------------- reconstruction


def fingerprint_from_dict(payload: Mapping) -> RunFingerprint:
    """Rebuild a :class:`RunFingerprint` from a stored ``fingerprint.json``.

    Field order is irrelevant — the payload is read by name. ``layers`` comes
    back from JSON as a list and is normalized to the tuple the dataclass
    declares, so the reconstruction digests identically to the original.

    Raises:
        FingerprintMismatch: If a required field is missing, or the payload
            carries a key this dataclass has no field for. An unknown key is a
            fingerprint written by different code; silently dropping it would
            produce a digest that matches by omission.
    """
    known = {field.name for field in dataclass_fields(RunFingerprint)}
    stored = {
        key: value
        for key, value in dict(payload).items()
        if key not in TRANSPORT_METADATA_KEYS
    }
    unknown = sorted(set(stored) - known)
    if unknown:
        raise FingerprintMismatch(
            f"the stored fingerprint carries unknown field(s) {unknown}. This "
            "run was written by different code than the installed "
            "jlens.mmpilot.store.RunFingerprint; refusing to reconstruct it "
            "rather than digest a payload that is missing part of what bound "
            "the run."
        )
    required = [
        field.name
        for field in dataclass_fields(RunFingerprint)
        if field.default is MISSING and field.default_factory is MISSING
    ]
    missing = sorted(name for name in required if name not in stored)
    if missing:
        raise FingerprintMismatch(
            f"the stored fingerprint is missing required field(s) {missing}; "
            "it cannot be reconstructed and nothing is guessed for it."
        )
    kwargs = dict(stored)
    kwargs["layers"] = tuple(int(layer) for layer in stored["layers"])
    for name in ("intervention_config", "extra", "selection_config"):
        if name in kwargs and kwargs[name] is None:
            kwargs[name] = {}
    return RunFingerprint(**kwargs)


def load_run_fingerprint(run_dir: str | os.PathLike[str]) -> tuple[RunFingerprint, dict]:
    """Read and faithfully reconstruct a completed run's fingerprint.

    Returns ``(fingerprint, stored_payload)``. The reconstruction is verified by
    re-deriving the digest from the reconstructed object and comparing it to the
    digest of the stored payload itself: they agree only when every field
    round-tripped, which is the property later checks are entitled to assume.

    Raises:
        ExistingRunNotFound: If the directory or ``fingerprint.json`` is absent.
            Neither is created.
        FingerprintMismatch: If the file is not readable JSON, or the
            reconstruction does not re-derive the stored payload's digest.
    """
    root = Path(run_dir)
    if not root.is_dir():
        raise ExistingRunNotFound(
            f"{root} does not exist (or is not a directory). Amendment-only mode "
            "opens a completed run; it never creates one. Check "
            "AMEND_EXISTING_RUN_DIR, and that Drive is mounted."
        )
    path = root / "fingerprint.json"
    if not path.is_file():
        raise ExistingRunNotFound(
            f"{path} does not exist, so {root} is not a completed run directory "
            "this study wrote. Nothing is written to create one."
        )
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise FingerprintMismatch(
            f"{path} is not readable JSON ({error}); refusing to guess what the "
            "run was bound to."
        ) from error
    if not isinstance(stored, dict):
        raise FingerprintMismatch(
            f"{path} does not hold a JSON object; refusing to reconstruct a "
            "fingerprint from it."
        )
    fingerprint = fingerprint_from_dict(stored)
    on_disk = {
        key: value
        for key, value in stored.items()
        if key not in TRANSPORT_METADATA_KEYS
    }
    if payload_checksum(on_disk) != fingerprint.digest:
        differences = "\n  - ".join(fingerprint.differences(on_disk)) or "(none named)"
        raise FingerprintMismatch(
            f"the fingerprint reconstructed from {path} does not re-derive the "
            "stored payload's digest, so the reconstruction is not faithful and "
            "every later check would be checking the wrong object.\n  - "
            + differences
        )
    return fingerprint, stored


def open_existing_store(
    run_dir: str | os.PathLike[str] | None,
    *,
    expected_fingerprint: str,
) -> UnitStore:
    """Open one explicitly named completed run, or refuse.

    Args:
        run_dir: The completed run directory. Required and explicit — there is
            no default, no search root, and no "most recent" fallback.
        expected_fingerprint: The ``sha256:`` digest the caller expects. Required
            for the same reason: a pin the operator wrote down is what makes
            "this is the run I mean" checkable rather than asserted.

    Returns:
        A :class:`UnitStore` whose :attr:`~UnitStore.status` is ``"resuming"``.

    Raises:
        AmendmentModeError: If either argument is empty.
        ExistingRunNotFound: If the directory or its fingerprint is absent.
        FingerprintMismatch: If the stored digest is not ``expected_fingerprint``.
    """
    if not str(run_dir or "").strip():
        raise AmendmentModeError(
            "amendment-only mode needs AMEND_EXISTING_RUN_DIR set to the "
            "completed run directory. It is never inferred: picking 'the newest "
            "mmaudio_* directory' would silently amend whichever run happened to "
            "sort last."
        )
    if not str(expected_fingerprint or "").strip():
        raise AmendmentModeError(
            "amendment-only mode needs AMEND_EXPECTED_FINGERPRINT set to the "
            "completed run's raw-generation fingerprint. Without it, opening the "
            "wrong directory would be indistinguishable from opening the right "
            "one."
        )
    root = Path(run_dir)
    fingerprint, _stored = load_run_fingerprint(root)
    if fingerprint.digest != str(expected_fingerprint).strip():
        raise FingerprintMismatch(
            f"{root} holds run fingerprint\n  {fingerprint.digest}\n"
            f"but AMEND_EXPECTED_FINGERPRINT is\n  {expected_fingerprint}\n"
            "Refusing to amend a run other than the one that was named. If the "
            "pin is what is stale, verify which run you mean before changing it."
        )
    store = UnitStore(root, fingerprint)
    status = store.open()
    if status != "resuming":
        raise ExistingRunNotFound(
            f"{root} opened as {status!r} rather than 'resuming', which means it "
            "held no fingerprint to resume from. Amendment-only mode describes a "
            "completed run and does not start one."
        )
    return store


# ----------------------------------------------------- report-only metadata


def _dig(payload: Mapping, *path: str):
    current = payload
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _thresholds_from_dict(payload: Mapping) -> TriModalThresholds:
    known = {field.name for field in dataclass_fields(TriModalThresholds)}
    kwargs = {key: value for key, value in payload.items() if key in known}
    if "norm_ratio_bounds" in kwargs and kwargs["norm_ratio_bounds"] is not None:
        low, high = kwargs["norm_ratio_bounds"]
        kwargs["norm_ratio_bounds"] = (float(low), float(high))
    return TriModalThresholds(**kwargs)


def restore_report_metadata(
    store: UnitStore, *, summary_name: str = ORIGINAL_SUMMARY_NAME
) -> dict:
    """Recover section 18b's inputs from the completed run's own summary.

    Section 18b needs the design (which layer was primary, which were the
    replication layers, which concepts were focal, which thresholds applied) and
    four report-only values the model sections would have left in the kernel:
    ``LENS_REPORT``, ``AUDIO_PROTOCOL``, ``INVARIANCE`` and
    ``BLOCKED_MODALITIES``. On a CPU amendment none of those exist in memory.

    They are read back from what the completed run recorded, never inferred from
    the current runtime: the installed transformers version, the lens files
    presently on Drive, and whichever modalities this process could support say
    nothing about the run being described.

    The design comes from the fingerprint the store was opened under — the
    strongest binding available — and the summary is required to agree with it.

    ``INVARIANCE`` in particular has no honest default: ``invariance_gate`` is a
    GO requirement, so substituting ``None`` would silently downgrade the verdict
    and substituting ``{"passed": True}`` would fabricate a gate result. A
    summary without it is refused.

    Raises:
        AmendedReportInputMissing: If the summary is absent, unreadable, or does
            not record a value that has no honest fallback.
        FingerprintMismatch: If the summary describes a different run.
    """
    path = store.root / summary_name
    if not path.is_file():
        raise AmendedReportInputMissing(
            f"{path} does not exist, so this run never wrote a completed report. "
            "Section 18b amends a completed run's interpretation; it cannot "
            "reconstruct the lens validation, the audio protocol or the "
            "invariance gate from stored units, and it will not invent them."
        )
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise AmendedReportInputMissing(
            f"{path} is not readable JSON ({error})."
        ) from error

    recorded = summary.get("fingerprint_digest")
    if recorded != store.fingerprint.digest:
        raise FingerprintMismatch(
            f"{path} records fingerprint\n  {recorded}\nbut the run directory "
            f"holds\n  {store.fingerprint.digest}\n"
            "The summary describes a different measurement than the units do; "
            "refusing to restore report metadata from it."
        )

    intervention = store.fingerprint.intervention_config or {}
    selection = store.fingerprint.selection_config or {}
    missing: list[str] = []

    primary_layer = intervention.get("primary_causal_layer")
    if primary_layer is None:
        missing.append("intervention_config.primary_causal_layer")
    replication_layers = intervention.get("replication_layers")
    if replication_layers is None:
        missing.append("intervention_config.replication_layers")
    focal = selection.get("focal_concepts")
    if focal is None:
        focal = _dig(summary, "selection_fingerprint", "focal_concepts")
    if focal is None:
        missing.append("selection_config.focal_concepts")
    thresholds_payload = _dig(summary, "verdicts", "E_overall", "thresholds")
    if thresholds_payload is None:
        missing.append("verdicts.E_overall.thresholds")
    lens_report = summary.get("lens_validation")
    if lens_report is None:
        missing.append("lens_validation")
    audio_protocol = summary.get("audio_protocol")
    if audio_protocol is None:
        missing.append("audio_protocol")
    invariance = summary.get("invariance")
    if invariance is None:
        missing.append("invariance")
    blocked = _dig(
        summary,
        "verdicts",
        "E_overall",
        "criteria",
        "three_modality_capability",
        "evidence",
        "blocked_modalities",
    )
    if blocked is None:
        missing.append(
            "verdicts.E_overall.criteria.three_modality_capability.evidence."
            "blocked_modalities"
        )
    if missing:
        raise AmendedReportInputMissing(
            f"{path} does not record {sorted(missing)}, and none of those has a "
            "fallback that would still be true of this run. In particular an "
            "absent invariance gate is not a passed one, an absent blocked-"
            "modality list is not an empty one, and neither the lens validation "
            "nor the audio protocol may be re-derived from the current runtime. "
            "Amend from a run whose section 18 completed."
        )

    return {
        "summary_path": path,
        "mode": store.fingerprint.mode,
        "primary_layer": int(primary_layer),
        "replication_layers": tuple(int(layer) for layer in replication_layers),
        "focal_concepts": list(focal),
        "thresholds": _thresholds_from_dict(thresholds_payload),
        "lens_report": lens_report,
        "audio_protocol": audio_protocol,
        "invariance": invariance,
        "blocked_modalities": list(blocked),
        "original_report_path": store.root / ORIGINAL_REPORT_NAME,
        "restored_from": str(path),
        "restored_fields": list(RESTORED_REPORT_FIELDS),
    }


# ------------------------------------------------------------- idempotent write


def _expected_rule_checksum(store: UnitStore, thresholds: TriModalThresholds) -> str:
    """The admissibility rule checksum this installation would bind right now."""
    capability = store.load("metric", "audio_capability_verdict") or {}
    recorded = capability.get("threshold")
    return admissibility_rule_record(
        threshold=(
            float(recorded)
            if recorded is not None
            else thresholds.capability_threshold
        ),
        principal_three_modality=True,
    )["rule_checksum"]


def amend_or_reuse(
    store: UnitStore,
    *,
    primary_layer: int,
    replication_layers: Sequence[int],
    focal_concepts: Sequence[str],
    thresholds: TriModalThresholds,
    lens_report: Mapping | None,
    audio_protocol: Mapping | None,
    invariance: Mapping | None,
    blocked_modalities: Sequence[str],
    mode: str,
    original_report_path: str | Path | None = None,
    report_name: str = AMENDED_REPORT_NAME,
    summary_name: str = AMENDED_SUMMARY_NAME,
    existing: str = "reuse_or_refuse",
    **extra,
) -> dict:
    """Produce the amended artifacts once, and only once.

    ``existing`` selects what happens when both artifacts are already there:

    ``"reuse_or_refuse"``
        The amendment path. The stored summary is held to the reopened store —
        run fingerprint, postprocessing version, admissibility rule version *and*
        checksum, and every source-unit digest. All matching returns
        ``status="reused"`` and **nothing is written**: the artifacts on Drive
        are the evidence, and rewriting them to identical conclusions would still
        change their bytes and their timestamps for no reason. Any difference
        raises rather than overwrites.
    ``"regenerate"``
        The live model path, where section 18 has just rewritten the original
        report from this same run. The amended artifacts are rewritten beside it
        and ``status="written"``.

    Exactly one of the two artifacts present is refused in both modes: that is a
    torn write, and neither reusing nor regenerating over it is safe.

    Returns:
        ``{"status", "paths", "amended", "binding", "verdicts"}``. On reuse
        ``amended`` holds the summary read back from disk and ``verdicts`` is
        taken from it, so callers print what the artifact actually says.
    """
    if existing not in ("reuse_or_refuse", "regenerate"):
        raise ValueError(f"unknown existing policy {existing!r}")
    root = Path(store.root)
    report_path = root / report_name
    summary_path = root / summary_name
    present = {"report": report_path.is_file(), "summary": summary_path.is_file()}
    if sum(present.values()) == 1:
        there, absent = (
            ("report", "summary") if present["report"] else ("summary", "report")
        )
        raise AmendedArtifactsTorn(
            f"{root} holds the amended {there} but not the amended {absent} "
            f"({report_path.name} / {summary_name}). That is a torn write, not a "
            "completed amendment: reusing it would trust a binding with nothing "
            "to bind, and regenerating over it would destroy the half that "
            "survived. Move or delete the orphan yourself, deliberately."
        )

    if all(present.values()) and existing == "reuse_or_refuse":
        try:
            stored_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise AmendedArtifactsTorn(
                f"{summary_path} exists but is not readable JSON ({error}); "
                "refusing to reuse it and refusing to overwrite it."
            ) from error
        bound = verify_amended_binding(
            stored_summary,
            store,
            expected_rule_checksum=_expected_rule_checksum(store, thresholds),
        )
        return {
            "status": "reused",
            "paths": {"report": report_path, "summary": summary_path},
            "amended": {"summary": stored_summary, "binding": stored_summary.get("binding")},
            "binding": stored_summary.get("binding"),
            "verdicts": {
                "capability": _dig(stored_summary, "verdicts", "A_audio_capability"),
                "representational": _dig(
                    stored_summary, "verdicts", "B_representational_transfer"
                ),
                "primary_causal": _dig(stored_summary, "verdicts", "C_primary_causal"),
                "replication": _dig(stored_summary, "verdicts", "D_replication"),
                "overall": _dig(stored_summary, "verdicts", "E_overall"),
            },
            "bound": bound,
        }

    amended = build_amended_report(
        store,
        primary_layer=primary_layer,
        replication_layers=replication_layers,
        focal_concepts=focal_concepts,
        thresholds=thresholds,
        original_report_path=original_report_path or (root / ORIGINAL_REPORT_NAME),
        lens_report=lens_report,
        audio_protocol=audio_protocol,
        invariance=invariance,
        blocked_modalities=blocked_modalities,
        mode=mode,
        **extra,
    )
    paths = write_amended_report(
        amended, run_dir=root, report_name=report_name, summary_name=summary_name
    )
    return {
        "status": "written",
        "paths": paths,
        "amended": amended,
        "binding": amended["binding"],
        "verdicts": amended["verdicts"],
        "bound": verify_amended_binding(amended["summary"], store),
    }


def format_amendment_inputs(inputs: Mapping) -> str:
    """The provenance block the amendment cell prints before it decides anything."""
    thresholds = inputs["thresholds"]
    return "\n".join(
        [
            "restored from the completed run (nothing inferred from this runtime):",
            f"  summary            {inputs['restored_from']}",
            f"  mode               {inputs['mode']}",
            f"  primary layer      L{inputs['primary_layer']}",
            f"  replication layers {list(inputs['replication_layers'])}",
            f"  focal concepts     {list(inputs['focal_concepts'])}",
            f"  capability gate    {thresholds.capability_threshold}",
            f"  lens validation    "
            f"{(inputs['lens_report'] or {}).get('combined_checksum')}",
            f"  audio protocol     "
            f"{(inputs['audio_protocol'] or {}).get('protocol_fingerprint')}",
            f"  invariance passed  {(inputs['invariance'] or {}).get('passed')}",
            f"  blocked modalities {list(inputs['blocked_modalities'])}",
        ]
    )


def amendment_mode_run_dir(
    run_dir: str | os.PathLike[str] | None,
) -> Path:
    """Resolve ``RUN_DIR`` for amendment-only setup without creating anything.

    The setup cell's normal branch derives a timestamped ``RUN_ID`` and calls
    ``mkdir``. In amendment-only mode that is exactly wrong — it litters the runs
    root with an empty directory and points the rest of the notebook at a run
    that holds nothing.
    """
    if not str(run_dir or "").strip():
        raise AmendmentModeError(
            "RUN_AMENDED_REPORT_ONLY is True but AMEND_EXISTING_RUN_DIR is empty. "
            "Amendment-only mode never derives a run id and never creates a run "
            "directory; name the completed one."
        )
    root = Path(run_dir)
    if not root.is_dir():
        raise ExistingRunNotFound(
            f"{root} does not exist. Amendment-only mode opens a completed run "
            "read-only and does not create it."
        )
    return root


def assert_no_model_module_imported(modules: Mapping[str, object] | None = None) -> dict:
    """Check that this path pulled in no model machinery. Cheap, and checkable.

    ``torch`` is imported by :mod:`jlens` itself and is not model machinery; a
    transformers import, a Gemma adapter, or loaded weights would be.
    """
    if modules is None:  # pragma: no cover - trivial default
        import sys

        modules = sys.modules
    forbidden = sorted(
        name
        for name in modules
        if name == "transformers"
        or name.startswith("transformers.")
        or name.startswith("jlens.gemma4")
        or name.startswith("jlens.mmpilot.real_backend")
    )
    return {"clean": not forbidden, "imported": forbidden}


__all__ = [
    "AMENDED_REPORT_NAME",
    "AMENDED_SUMMARY_NAME",
    "MODEL_AND_CAUSAL_SWITCHES",
    "ORIGINAL_REPORT_NAME",
    "ORIGINAL_SUMMARY_NAME",
    "RESTORED_REPORT_FIELDS",
    "TRANSPORT_METADATA_KEYS",
    "AmendedArtifactsTorn",
    "AmendmentModeError",
    "ExistingRunNotFound",
    "FingerprintMismatch",
    "amend_or_reuse",
    "amendment_mode_run_dir",
    "assert_amendment_mode_exclusive",
    "assert_no_model_module_imported",
    "fingerprint_from_dict",
    "format_amendment_inputs",
    "load_run_fingerprint",
    "open_existing_store",
    "restore_report_metadata",
]
