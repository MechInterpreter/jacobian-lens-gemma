# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Loading a *published, independently confirmed* calibration lens — or refusing.

The completed research-grade calibration run publishes one lens file per layer
that earned it, with a metadata sidecar written by
:func:`jlens.calibration.publication.build_artifact`. Three layers earned it at
scale 100 — **35, 38 and 40**. Layer 32 and every earlier tested layer failed
confirmation *at that scale* and were recorded with ``validated=false``.

Scale is part of the fact, and this matters
===========================================

"Layer 32 failed confirmation" is a statement about **scale 100**, not about
layer 32 forever. The research-grade early-layer extension refitted the same
estimator at scale 250 and layer 32 passed its own untouched 256-prompt
confirmation set (run ``rgext_real_c18f03f06e7b``,
``EARLY_LAYER_CALIBRATION_GO``). :data:`CONFIRMED_LAYERS` and
:data:`FAILED_CONFIRMATION_LAYERS` below are therefore **scale-100 defaults**,
not a global registry, and they are the defaults of
:class:`PublishedLensExpectations` fields rather than constants this module
reads directly — every clause in :func:`validate_published_artifact` consults
``expectations``. A study using the scale-250 artifact states its own
expectations (see :mod:`jlens.mmpilot.l32_followup`), and gets the identical
schema, checksum and confirmation clauses applied to them.

What has not changed: nothing here loads a lens whose own sidecar says
``validated=false``, and no layer is ever described as validated on the
strength of a filename.

Why a second loader
===================

:func:`jlens.mmpilot.real_backend.load_validated_lens` reads the *v2*
convention: one ``lens.validated.pt`` beside a shared
``validated_lens_manifest.json``. The published calibration artifacts use a
different one — a per-layer sidecar carrying the confirmation verdict, the
scale point, the corpus split checksums and the geometry conventions. Pointing
the old loader at a new artifact would fail on a missing file; pointing it at a
*renamed* one would silently skip every confirmation field the new schema adds.
So this module reads the new schema explicitly.

What "verify the schema instead of assuming it" means here
==========================================================

Every clause below names the field it read and refuses when that field is
**absent**, rather than defaulting it. A missing ``validated`` key is not a
False; it is an artifact this code has never seen, and treating it as a False
would be a guess dressed as a check. :func:`artifact_schema_report` records the
keys actually present so a refusal can be diagnosed from the report alone.

Nothing here fits, rescales, merges-in, or modifies a lens. The only mutation
this module performs is assembling the separately published single-layer
Jacobians into one frozen :class:`~jlens.lens.JacobianLens` so a caller can ask
it for a layer — and that assembly refuses unless every artifact agrees on
``d_model`` and on the number of fitting prompts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from jlens.metadata import file_sha256
from jlens.mmpilot.store import payload_checksum

#: Bound into the run fingerprint. A change here refuses to resume anything
#: validated under the older rule.
PUBLISHED_LENS_PROTOCOL = "jlens.mmpilot.published_lens.v1"

#: The artifact format this module knows how to read.
EXPECTED_ARTIFACT_FORMAT = "jlens.calibration.artifact.v1"

#: The layers the completed **scale-100** calibration run's untouched
#: confirmation set passed. Stated here so a typo in a notebook cannot quietly
#: widen it. A study at another scale supplies its own tuple.
CONFIRMED_LAYERS: tuple[int, ...] = (35, 38, 40)

#: Layers tested at **scale 100** that failed its confirmation. Named
#: explicitly because the failure is the informative part: layer 32 has
#: scale-100 diagnostics and no scale-100 lens. It is *not* a permanent
#: exclusion — layer 32 passed at scale 250 under the early-layer extension,
#: and a caller reading that artifact says so in its own expectations.
FAILED_CONFIRMATION_LAYERS: tuple[int, ...] = (32,)

#: Fields the sidecar must carry. Absence is a refusal, never a default.
REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "artifact_format_version",
    "artifact_checksum",
    "frozen",
    "validated",
    "model_repo_id",
    "model_revision",
    "tokenizer_revision",
    "physical_layer",
    "d_model",
    "hook_site",
    "residual_convention",
    "vector_orientation",
    "normalization_convention",
    "calibration_modality",
    "scale_point",
    "n_fitting_prompts",
    "confirmation_protocol",
    "confirmation_failed_checks",
    "lens_path",
    "lens_checksum",
)

#: Flags that must all be False. A lens that saw multimodal data during
#: calibration cannot answer "does a *text-calibrated* lens transfer".
REQUIRED_FALSE_FIELDS: tuple[str, ...] = (
    "spokencoco_used",
    "multimodal_data_used",
    "cross_modal_alignment",
    "modality_specific_lens",
)


class PublishedLensRefused(RuntimeError):
    """A lens artifact was offered that this study may not use.

    Attributes:
        problems: Every failed clause, not just the first — one Colab round
            trip should surface all of them.
        report: The validation record, including the observed schema.
    """

    def __init__(self, message: str, *, problems: Sequence[str] = (), report: Mapping | None = None) -> None:
        super().__init__(message)
        self.problems = list(problems)
        self.report = dict(report or {})


@dataclass(frozen=True)
class PublishedLensSpec:
    """One published artifact, as the notebook states it.

    Attributes:
        layer: The physical decoder layer the artifact must be for.
        path: The ``.pt`` file. Its sidecar is the same path with ``.json``.
        expect_sha256: ``sha256:<hex>`` pin of the ``.pt`` bytes.
    """

    layer: int
    path: str
    expect_sha256: str

    def to_dict(self) -> dict:
        return {"layer": int(self.layer), "path": str(self.path), "expect_sha256": str(self.expect_sha256)}

    @property
    def sidecar_path(self) -> Path:
        return Path(self.path).with_suffix(".json")


@dataclass(frozen=True)
class PublishedLensExpectations:
    """What every artifact is held to. Stated once, printed in the report."""

    model_repo_id: str
    model_revision: str
    #: ``None`` means "must equal ``model_revision``" — the calibration run
    #: loaded both from the same pinned commit, so a divergence is a defect.
    tokenizer_revision: str | None = None
    scale_point: int = 100
    d_model: int = 2560
    calibration_modality: str = "text-only"
    hook_site: str = "block_output"
    confirmed_layers: tuple[int, ...] = CONFIRMED_LAYERS
    failed_confirmation_layers: tuple[int, ...] = FAILED_CONFIRMATION_LAYERS
    protocol: str = PUBLISHED_LENS_PROTOCOL

    def resolved_tokenizer_revision(self) -> str:
        return self.tokenizer_revision or self.model_revision

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["confirmed_layers"] = list(self.confirmed_layers)
        payload["failed_confirmation_layers"] = list(self.failed_confirmation_layers)
        payload["tokenizer_revision"] = self.resolved_tokenizer_revision()
        return payload


# ------------------------------------------------------------------ schema


def artifact_schema_report(artifact: Mapping) -> dict:
    """What the sidecar actually contains, before anything is asked of it."""
    present = sorted(str(key) for key in artifact)
    return {
        "observed_keys": present,
        "n_keys": len(present),
        "artifact_format_version": artifact.get("artifact_format_version"),
        "missing_required_fields": [
            field_name for field_name in REQUIRED_ARTIFACT_FIELDS if field_name not in artifact
        ],
        "unexpected_top_level_types": sorted(
            key
            for key, value in artifact.items()
            if not isinstance(value, (str, int, float, bool, list, dict, type(None)))
        ),
    }


def read_artifact_sidecar(lens_path: str | Path) -> tuple[Path, dict]:
    """The metadata record published beside ``lens_path``.

    Raises:
        PublishedLensRefused: If the sidecar is missing or is not a JSON
            object. A lens without its publication record is an unpublished
            lens, whatever its filename says.
    """
    import json

    sidecar = Path(lens_path).with_suffix(".json")
    if not sidecar.is_file():
        raise PublishedLensRefused(
            f"missing {sidecar}: the published calibration artifact writes its "
            "metadata record beside the lens, and a lens without that record "
            "carries no publication or confirmation status at all",
            problems=["sidecar_present"],
        )
    try:
        artifact = json.loads(sidecar.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise PublishedLensRefused(
            f"{sidecar} is not readable JSON: {error}", problems=["sidecar_readable"]
        ) from error
    if not isinstance(artifact, dict):
        raise PublishedLensRefused(
            f"{sidecar} does not hold a JSON object", problems=["sidecar_readable"]
        )
    return sidecar, artifact


# -------------------------------------------------------------- validation


def _check(records: list[dict], problems: list[str], name: str, ok: bool, detail: str) -> None:
    records.append({"check": name, "ok": bool(ok), "detail": detail})
    if not ok:
        problems.append(f"{name}: {detail}")


def validate_published_artifact(
    spec: PublishedLensSpec,
    expectations: PublishedLensExpectations,
    *,
    artifact: Mapping,
    actual_checksum: str,
    sidecar_path: str | Path,
) -> dict:
    """Hold one artifact to every clause, and report all of them.

    Returns the validation record. Raises rather than returning a failed one,
    because a caller that had to remember to inspect a flag is a caller that
    can forget to.

    Raises:
        PublishedLensRefused: On any failed clause. ``problems`` lists them all.
    """
    checks: list[dict] = []
    problems: list[str] = []
    schema = artifact_schema_report(artifact)

    _check(
        checks,
        problems,
        "artifact_schema_complete",
        not schema["missing_required_fields"],
        f"missing fields {schema['missing_required_fields']}"
        if schema["missing_required_fields"]
        else f"all {len(REQUIRED_ARTIFACT_FIELDS)} required fields present",
    )
    _check(
        checks,
        problems,
        "artifact_format_version",
        artifact.get("artifact_format_version") == EXPECTED_ARTIFACT_FORMAT,
        f"{artifact.get('artifact_format_version')!r} != {EXPECTED_ARTIFACT_FORMAT!r}",
    )

    # ---- the file is the pinned one, and the record is about this file
    _check(
        checks,
        problems,
        "lens_checksum_matches_pin",
        actual_checksum == spec.expect_sha256,
        f"file {actual_checksum} vs pinned {spec.expect_sha256}",
    )
    _check(
        checks,
        problems,
        "artifact_describes_this_file",
        artifact.get("lens_checksum") == actual_checksum,
        f"record names {artifact.get('lens_checksum')}, file is {actual_checksum}",
    )
    recorded_name = Path(str(artifact.get("lens_path", ""))).name
    _check(
        checks,
        problems,
        "artifact_filename_agrees",
        recorded_name == Path(spec.path).name,
        f"record was published as {recorded_name!r}, loading {Path(spec.path).name!r}",
    )
    if "artifact_checksum" in artifact:
        recomputed = payload_checksum(
            {key: value for key, value in artifact.items() if key != "artifact_checksum"}
        )
        _check(
            checks,
            problems,
            "artifact_record_is_intact",
            recomputed == artifact.get("artifact_checksum"),
            f"recomputed {recomputed} vs recorded {artifact.get('artifact_checksum')}",
        )

    # ---- publication and confirmation status
    _check(
        checks,
        problems,
        "publication_status",
        artifact.get("validated") is True and artifact.get("published") is not False,
        f"validated={artifact.get('validated')!r}, published={artifact.get('published')!r}"
        + (
            " — this is a failed-layer diagnostics record, not a published lens"
            if artifact.get("published") is False
            else ""
        ),
    )
    _check(
        checks,
        problems,
        "frozen_status",
        artifact.get("frozen") is True,
        f"frozen={artifact.get('frozen')!r}",
    )
    failed_checks = artifact.get("confirmation_failed_checks")
    _check(
        checks,
        problems,
        "confirmation_status",
        bool(artifact.get("confirmation_protocol"))
        and isinstance(failed_checks, list)
        and not failed_checks,
        f"protocol={artifact.get('confirmation_protocol')!r}, "
        f"failed_checks={failed_checks!r}",
    )
    _check(
        checks,
        problems,
        "layer_independently_confirmed",
        int(spec.layer) in set(expectations.confirmed_layers)
        and int(spec.layer) not in set(expectations.failed_confirmation_layers),
        f"layer {spec.layer} vs confirmed {list(expectations.confirmed_layers)}"
        + (
            f"; layer {spec.layer} failed confirmation and is never a validated lens"
            if int(spec.layer) in set(expectations.failed_confirmation_layers)
            else ""
        ),
    )

    # ---- identity
    _check(
        checks,
        problems,
        "model_repo_id",
        artifact.get("model_repo_id") == expectations.model_repo_id,
        f"{artifact.get('model_repo_id')!r} != {expectations.model_repo_id!r}",
    )
    _check(
        checks,
        problems,
        "model_revision",
        artifact.get("model_revision") == expectations.model_revision,
        f"{artifact.get('model_revision')!r} != {expectations.model_revision!r}",
    )
    _check(
        checks,
        problems,
        "tokenizer_revision",
        artifact.get("tokenizer_revision") == expectations.resolved_tokenizer_revision(),
        f"{artifact.get('tokenizer_revision')!r} != "
        f"{expectations.resolved_tokenizer_revision()!r}",
    )

    # ---- what it was fitted on
    _check(
        checks,
        problems,
        "fitted_scale",
        int(artifact.get("scale_point", -1)) == int(expectations.scale_point)
        and int(artifact.get("n_fitting_prompts", -1)) == int(expectations.scale_point),
        f"scale_point={artifact.get('scale_point')!r}, "
        f"n_fitting_prompts={artifact.get('n_fitting_prompts')!r}, "
        f"required {expectations.scale_point}",
    )
    _check(
        checks,
        problems,
        "calibration_modality",
        artifact.get("calibration_modality") == expectations.calibration_modality,
        f"{artifact.get('calibration_modality')!r} != "
        f"{expectations.calibration_modality!r}",
    )
    wrong_flags = {
        name: artifact.get(name)
        for name in REQUIRED_FALSE_FIELDS
        if artifact.get(name) is not False
    }
    _check(
        checks,
        problems,
        "no_multimodal_calibration",
        not wrong_flags,
        f"{wrong_flags}" if wrong_flags else f"all of {list(REQUIRED_FALSE_FIELDS)} are False",
    )

    # ---- geometry
    _check(
        checks,
        problems,
        "physical_layer",
        int(artifact.get("physical_layer", -1)) == int(spec.layer),
        f"record is for layer {artifact.get('physical_layer')!r}, requested {spec.layer}",
    )
    _check(
        checks,
        problems,
        "d_model",
        int(artifact.get("d_model", -1)) == int(expectations.d_model),
        f"{artifact.get('d_model')!r} != {expectations.d_model}",
    )
    _check(
        checks,
        problems,
        "hook_site",
        artifact.get("hook_site") == expectations.hook_site,
        f"{artifact.get('hook_site')!r} != {expectations.hook_site!r}",
    )
    for name in ("residual_convention", "vector_orientation", "normalization_convention"):
        value = artifact.get(name)
        _check(
            checks,
            problems,
            name,
            isinstance(value, str) and bool(value.strip()),
            f"{name}={value!r}",
        )

    record = {
        "protocol": expectations.protocol,
        "layer": int(spec.layer),
        "lens_path": str(spec.path),
        "sidecar_path": str(sidecar_path),
        "lens_checksum": actual_checksum,
        "expected_checksum": spec.expect_sha256,
        "expectations": expectations.to_dict(),
        "schema": schema,
        "checks": checks,
        "failed_checks": [entry["check"] for entry in checks if not entry["ok"]],
        "passed": not problems,
        # Copied verbatim so the run report can print what the artifact claims
        # about itself rather than re-deriving it.
        "recorded": {
            key: artifact.get(key)
            for key in (
                "artifact_format_version",
                "protocol_version",
                "frozen",
                "validated",
                "model_repo_id",
                "model_revision",
                "tokenizer_repo_id",
                "tokenizer_revision",
                "physical_layer",
                "normalized_depth",
                "d_model",
                "target_layer",
                "hook_site",
                "residual_convention",
                "vector_orientation",
                "normalization_convention",
                "logit_softcap",
                "calibration_modality",
                "scale_point",
                "n_fitting_prompts",
                "corpus_id",
                "corpus_revision",
                "confirmation_protocol",
                "confirmation_failed_checks",
                "gate_digest",
            )
            if key in artifact
        },
    }
    if problems:
        raise PublishedLensRefused(
            f"refusing the lens artifact for layer {spec.layer} at {spec.path}:\n  - "
            + "\n  - ".join(problems),
            problems=problems,
            report=record,
        )
    return record


# ------------------------------------------------------------------ loading


@dataclass
class LoadedPublishedLenses:
    """Three separately published single-layer lenses, assembled and pinned."""

    lens: Any
    layers: tuple[int, ...]
    checksums: dict[int, str]
    validations: dict[int, dict]
    combined_checksum: str
    n_prompts: int
    d_model: int
    specs: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "protocol": PUBLISHED_LENS_PROTOCOL,
            "layers": list(self.layers),
            "checksums": {str(layer): value for layer, value in sorted(self.checksums.items())},
            "combined_checksum": self.combined_checksum,
            "n_prompts": self.n_prompts,
            "d_model": self.d_model,
            "specs": list(self.specs),
            "validations": {
                str(layer): value for layer, value in sorted(self.validations.items())
            },
            "frozen": True,
            "fitted_here": False,
        }


def combined_lens_checksum(checksums: Mapping[int, str]) -> str:
    """One digest over the three per-layer pins, for the run fingerprint."""
    return payload_checksum(
        {
            "protocol": PUBLISHED_LENS_PROTOCOL,
            "per_layer": {str(layer): str(value) for layer, value in sorted(checksums.items())},
        }
    )


def load_published_lenses(
    specs: Sequence[PublishedLensSpec],
    expectations: PublishedLensExpectations,
    *,
    lens_loader=None,
) -> LoadedPublishedLenses:
    """Validate every artifact, then assemble the layers into one frozen lens.

    Args:
        lens_loader: ``path -> JacobianLens``. Defaults to
            :meth:`jlens.lens.JacobianLens.load`; the tests substitute a loader
            over tiny fixtures so this path runs on CPU in milliseconds.

    Raises:
        PublishedLensRefused: If any artifact fails a clause, if two artifacts
            disagree on ``d_model`` or on the number of fitting prompts, if a
            loaded file does not actually contain the layer its record names,
            or if the same layer is offered twice.
    """
    from jlens.lens import JacobianLens

    if not specs:
        raise PublishedLensRefused(
            "no published lens artifacts were configured", problems=["specs_present"]
        )
    duplicates = sorted(
        {int(spec.layer) for spec in specs if [s.layer for s in specs].count(spec.layer) > 1}
    )
    if duplicates:
        raise PublishedLensRefused(
            f"layers {duplicates} were offered more than once", problems=["specs_distinct"]
        )
    lens_loader = lens_loader or JacobianLens.load

    validations: dict[int, dict] = {}
    checksums: dict[int, str] = {}
    jacobians: dict[int, Any] = {}
    prompt_counts: dict[int, int] = {}
    widths: dict[int, int] = {}
    for spec in sorted(specs, key=lambda item: int(item.layer)):
        path = Path(spec.path)
        if not path.is_file():
            raise PublishedLensRefused(
                f"the published lens for layer {spec.layer} was not found at {path}. "
                "This study does not fit a lens; point the configuration at the "
                "published artifact.",
                problems=["lens_file_exists"],
            )
        checksum = file_sha256(str(path))
        sidecar_path, artifact = read_artifact_sidecar(path)
        validations[int(spec.layer)] = validate_published_artifact(
            spec,
            expectations,
            artifact=artifact,
            actual_checksum=checksum,
            sidecar_path=sidecar_path,
        )
        checksums[int(spec.layer)] = checksum

        loaded = lens_loader(str(path))
        if int(spec.layer) not in getattr(loaded, "jacobians", {}):
            raise PublishedLensRefused(
                f"{path} does not contain layer {spec.layer} (it holds "
                f"{sorted(getattr(loaded, 'jacobians', {}))}); the record and the "
                "file disagree",
                problems=["file_contains_recorded_layer"],
                report=validations[int(spec.layer)],
            )
        jacobians[int(spec.layer)] = loaded.jacobians[int(spec.layer)]
        prompt_counts[int(spec.layer)] = int(loaded.n_prompts)
        widths[int(spec.layer)] = int(loaded.d_model)

    if len(set(widths.values())) != 1:
        raise PublishedLensRefused(
            f"the artifacts disagree on d_model: {widths}", problems=["d_model_agreement"]
        )
    if len(set(prompt_counts.values())) != 1:
        raise PublishedLensRefused(
            f"the artifacts were fitted on different prompt counts: {prompt_counts}. "
            "Layers fitted at different scales are not one lens.",
            problems=["scale_agreement"],
        )
    d_model = next(iter(widths.values()))
    n_prompts = next(iter(prompt_counts.values()))
    if d_model != int(expectations.d_model):
        raise PublishedLensRefused(
            f"loaded d_model {d_model} != expected {expectations.d_model}",
            problems=["d_model"],
        )
    if n_prompts != int(expectations.scale_point):
        raise PublishedLensRefused(
            f"loaded lenses were fitted on {n_prompts} prompts, not "
            f"{expectations.scale_point}",
            problems=["fitted_scale"],
        )

    assembled = JacobianLens(jacobians=jacobians, n_prompts=n_prompts, d_model=d_model)
    return LoadedPublishedLenses(
        lens=assembled,
        layers=tuple(sorted(jacobians)),
        checksums=checksums,
        validations=validations,
        combined_checksum=combined_lens_checksum(checksums),
        n_prompts=n_prompts,
        d_model=d_model,
        specs=[spec.to_dict() for spec in sorted(specs, key=lambda item: int(item.layer))],
    )


def format_lens_report(loaded: LoadedPublishedLenses) -> str:
    """The block the notebook prints after the artifacts are accepted."""
    lines = [
        "=" * 72,
        "PUBLISHED, INDEPENDENTLY CONFIRMED LENSES — nothing is fitted here",
        "=" * 72,
        f"  protocol          {PUBLISHED_LENS_PROTOCOL}",
        f"  layers            {list(loaded.layers)}",
        f"  fitted on         {loaded.n_prompts} text-only prompts",
        f"  d_model           {loaded.d_model}",
        f"  combined digest   {loaded.combined_checksum}",
        "",
    ]
    for layer in loaded.layers:
        record = loaded.validations[layer]
        recorded = record["recorded"]
        lines += [
            f"  layer {layer}",
            f"    file        {record['lens_path']}",
            f"    checksum    {record['lens_checksum']}",
            f"    confirmed   {recorded.get('confirmation_protocol')} "
            f"(failed checks {recorded.get('confirmation_failed_checks')})",
            f"    calibration {recorded.get('calibration_modality')} at scale "
            f"{recorded.get('scale_point')}",
            f"    hook site   {recorded.get('hook_site')}",
            f"    checks      {len(record['checks'])} passed",
        ]
    # Taken from what these artifacts were actually held to, not from the
    # module defaults: a scale-250 study confirms layers the scale-100 defaults
    # list as failures, and printing the defaults would contradict the lens
    # that was just accepted.
    first = loaded.validations[loaded.layers[0]]["expectations"]
    refused = list(first["failed_confirmation_layers"])
    lines += [
        "",
        f"  Held to scale {first['scale_point']}: confirmed layers "
        f"{list(first['confirmed_layers'])}"
        + (
            f", refused layers {refused} (tested at this scale, failed its "
            "untouched confirmation set — never loaded, never described as "
            "validated)."
            if refused
            else "; no layer is on this study's failed-confirmation list."
        ),
    ]
    return "\n".join(lines)


__all__ = [
    "CONFIRMED_LAYERS",
    "EXPECTED_ARTIFACT_FORMAT",
    "FAILED_CONFIRMATION_LAYERS",
    "PUBLISHED_LENS_PROTOCOL",
    "REQUIRED_ARTIFACT_FIELDS",
    "REQUIRED_FALSE_FIELDS",
    "LoadedPublishedLenses",
    "PublishedLensExpectations",
    "PublishedLensRefused",
    "PublishedLensSpec",
    "artifact_schema_report",
    "combined_lens_checksum",
    "format_lens_report",
    "load_published_lenses",
    "read_artifact_sidecar",
    "validate_published_artifact",
]
