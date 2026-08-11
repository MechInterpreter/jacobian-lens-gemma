# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""The lenses a contiguous band needs: discovery, fitting, confirmation.

A band clamp over physical layers 32-40 patches **every** layer in that range,
so every one of them must have an independently confirmed J-lens fitted on the
same corpus at the same scale. Gemma currently has:

===========================  ==================================================
L32                          scale 250, published artifact of the early-layer
                             extension run, confirmed on that run's fresh
                             256-prompt confirmation set.
L35 / L38 / L40              scale 250 **inside the extension's own scale-250
                             snapshot**, with recorded confirmation verdicts
                             from the same fresh set. They were scored but not
                             published, because the extension's publication
                             targets were L26 and L32. (Their separately
                             published artifacts are scale *100*, from the
                             parent calibration — a different fit, and a band
                             may not mix scales.)
L33 / L34 / L36 / L37 / L39  nothing. This module fits them.
===========================  ==================================================

So the band's missing work is exactly five layers, and this module is the
smallest correction that produces them:

**Stage 1 — fit.** One pass of the unmodified upstream estimator
(:func:`jlens.fitting.fit` through :func:`jlens.calibration.fitting.run_calibration`)
over the *same* deterministic 250-prompt fit ordering the extension used, with
``source_layers = (33, 34, 36, 37, 39)``. Five layers in one pass, because the
estimator already differentiates to every requested source layer from one
forward and one set of backward passes.

This is **not** a continuation of the extension's accumulator and must never be
seeded from it: that checkpoint holds ``jacobian_sum`` for a different layer
grid, and upstream refuses a resume across grids for exactly the right reason.
It is a fresh accumulator over an identical, checksum-verified prompt list —
which makes the resulting ``J_l`` corpus-equivalent to the scale-250 lenses it
will sit beside, and :func:`assert_corpus_equivalence` refuses the run if that
equivalence cannot be established from stored artifacts.

**Stage 2 — confirm.** The same frozen native-readout validity gate the
confirmed layers passed (:data:`jlens.calibration.extension.EXTENSION_GATE` for
development, :data:`~jlens.calibration.extension.EXTENSION_CONFIRMATION_GATE`
for confirmation), on **third-generation** held-out sets: the parent's fit,
development and confirmation records are excluded, *and so are the extension's
own fresh development and confirmation sets*, which have been opened and read
and are therefore spent. A layer publishes only if it passes confirmation, and
a band is available only if every physical layer in it passes.

Nothing here fits, rescales, republishes or overwrites an existing lens. The
parent calibration run and the extension run are opened read-only, and
:func:`publish_band_layer` refuses any destination outside the band run.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from jlens.calibration.corpus import MAX_HAMMING_DISTANCE, CorpusRecord
from jlens.calibration.extension import (
    EXTENSION_CONFIRMATION_GATE,
    EXTENSION_GATE,
    FreshEvaluationSplits,
    audit_fresh_split_leakage,
    build_fresh_evaluation_splits,
)
from jlens.calibration.gate import CalibrationGate
from jlens.calibration.plan import CapturePlan, build_capture_plan, normalized_depth
from jlens.calibration.publication import (
    ConfirmationVault,
    PublicationRefused,
    publish_layer,
)
from jlens.calibration.state import CALIBRATION_STAGES, CalibrationStore
from jlens.metadata import file_sha256
from jlens.mmpilot.band_swap import BandLensRow
from jlens.mmpilot.published_lens import read_artifact_sidecar
from jlens.mmpilot.store import payload_checksum

__all__ = [
    "BAND_CONFIRMATION_PROMPT_SEED",
    "BAND_DEVELOPMENT_PROMPT_SEED",
    "BAND_INTERIOR_LAYERS",
    "BAND_LENS_ARTIFACT_SCHEMA",
    "BAND_LENS_PROTOCOL",
    "BAND_LENS_REPORT_SCHEMA",
    "BAND_ONLY_STAGES",
    "BAND_PUBLISHED_STATUS",
    "BAND_SCALE",
    "BAND_SPLIT_PROTOCOL",
    "BAND_SPLIT_SEED",
    "BAND_STAGES",
    "BAND_TARGET_LAYER",
    "BAND_WINDOW",
    "BandLensRefused",
    "BandLensStore",
    "assert_corpus_equivalence",
    "band_capture_plan",
    "band_fit_budget",
    "band_layer_verdict",
    "band_scale_selection",
    "build_band_evaluation_splits",
    "discover_extension_scale250_lenses",
    "discover_published_band_lenses",
    "format_band_fit_budget",
    "publish_band_layer",
    "verify_reconstructed_extension_splits",
]

#: Bound into every band-lens artifact and into the run fingerprint.
BAND_LENS_PROTOCOL = "mmpilot.band_interior_lens_fit.v1"

BAND_LENS_REPORT_SCHEMA = "jlens.mmpilot.band_interior_lens_report.v1"
BAND_LENS_ARTIFACT_SCHEMA = "jlens.mmpilot.band_interior_lens_artifact.v1"
BAND_PUBLISHED_STATUS = "PUBLISHED_VALIDATED_BAND_INTERIOR_LAYER"

#: The physical layers a 32-40 band needs and Gemma does not have.
BAND_INTERIOR_LAYERS: tuple[int, ...] = (33, 34, 36, 37, 39)

#: The band window this study is about, and the scale every one of its lenses
#: must share. 250 is not a new choice: it is the scale the extension selected
#: and published L32 at, and a band may not mix scales.
BAND_WINDOW: tuple[int, int] = (32, 40)
BAND_SCALE = 250
BAND_TARGET_LAYER = 41

#: Fresh seeds again, so the third-generation sets are a genuinely new draw and
#: not the extension's sets under another name.
BAND_SPLIT_PROTOCOL = "band-interior-fresh-eval-hash-bucket-v1"
BAND_SPLIT_SEED = 20260811
BAND_DEVELOPMENT_PROMPT_SEED = 20260812
BAND_CONFIRMATION_PROMPT_SEED = 20260813

BAND_ONLY_STAGES = (
    "band_preflight",
    "band_splits",
    "band_fit",
    "band_publication",
    "band_verdict",
)
BAND_STAGES = (*CALIBRATION_STAGES, *BAND_ONLY_STAGES)

BAND_LAYER_GO = "BAND_INTERIOR_LENS_GO"
BAND_LAYER_NO_GO = "BAND_INTERIOR_LENS_NO_GO"


class BandLensRefused(RuntimeError):
    """The band's lenses cannot be produced or trusted as specified."""


# ------------------------------------------------------------------ the plan


def band_capture_plan(
    *,
    layers: Sequence[int] = BAND_INTERIOR_LAYERS,
    target_layer: int = BAND_TARGET_LAYER,
    d_model: int = 2560,
    dim_batch: int = 8,
    max_seq_len: int = 128,
    skip_first: int = 16,
    n_layers: int = 42,
) -> CapturePlan:
    """The capture plan for the missing layers — the extension's, minus layers.

    Every geometry field except the layer grid is the extension's, because a
    lens fitted under a different ``skip_first``, ``max_seq_len``,
    ``dim_batch`` or target layer is a different estimator and could not sit in
    one band with the lenses it has to match.
    """
    return build_capture_plan(
        layers=layers,
        target_layer=int(target_layer),
        d_model=int(d_model),
        dim_batch=int(dim_batch),
        max_seq_len=int(max_seq_len),
        skip_first=int(skip_first),
        n_layers=int(n_layers),
    )


def assert_corpus_equivalence(
    *,
    extension_report: Mapping,
    plan: CapturePlan,
    scale: int = BAND_SCALE,
    model_repo_id: str,
    model_revision: str,
) -> dict:
    """Prove the new fit is the same fit, minus its layer grid — or refuse.

    Every clause is read out of the extension run's own report. If any of them
    is absent the run stops here, before a model is loaded, with an explicit
    scientific refusal: a lens fitted on an unestablished corpus cannot be
    placed in a band beside lenses whose corpus is known, and mixing them
    silently is the failure this function exists to prevent.

    Raises:
        BandLensRefused: Listing every clause that failed.
    """
    corpus = dict(extension_report.get("corpus") or {})
    capture = dict((extension_report.get("continuation") or {}).get("capture_plan") or {})
    if not capture:
        capture = dict(
            (extension_report.get("budget") or {}).get("anchor") or {}
        )
    selection = dict(extension_report.get("scale_selection") or {})
    splits = dict(corpus.get("splits") or {})
    checksums = dict(splits.get("checksums") or {})
    fit_prefixes = dict(corpus.get("fit_prefix_checksums") or {})

    problems: list[str] = []
    checks: list[dict] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})
        if not ok:
            problems.append(f"{name}: {detail}")

    check(
        "extension_report_schema",
        extension_report.get("schema") == "jlens.calibration.early_layer_extension_report.v1",
        f"schema={extension_report.get('schema')!r}",
    )
    check(
        "extension_mode_is_real",
        extension_report.get("mode") == "real",
        f"mode={extension_report.get('mode')!r}",
    )
    check(
        "selected_scale",
        selection.get("selected_scale") is not None
        and int(selection["selected_scale"]) == int(scale),
        f"selected_scale={selection.get('selected_scale')!r}, required {scale}",
    )
    check(
        "fit_prefix_checksum_for_this_scale",
        str(int(scale)) in fit_prefixes,
        f"fit_prefix_checksums keys={sorted(fit_prefixes)}",
    )
    check(
        "text_only_corpus",
        corpus.get("modality") == "text-only"
        and corpus.get("spokencoco_used") is False
        and corpus.get("multimodal_data_used") is False,
        f"modality={corpus.get('modality')!r}, "
        f"spokencoco_used={corpus.get('spokencoco_used')!r}, "
        f"multimodal_data_used={corpus.get('multimodal_data_used')!r}",
    )
    for field_name in ("corpus_id", "revision", "min_chars"):
        check(
            f"corpus_{field_name}_recorded",
            corpus.get(field_name) not in (None, ""),
            f"{field_name}={corpus.get(field_name)!r}",
        )
    check(
        "fit_order_protocol",
        corpus.get("fit_order_protocol") == "parent-prefix-pinned-nested-order-v1",
        f"fit_order_protocol={corpus.get('fit_order_protocol')!r}",
    )
    check(
        "split_checksums_recorded",
        {"fit", "validation", "confirmation"} <= set(checksums),
        f"splits.checksums keys={sorted(checksums)}",
    )
    recorded_target = capture.get("target_layer")
    check(
        "target_layer_matches",
        recorded_target is None or int(recorded_target) == int(plan.target_layer),
        f"extension target_layer={recorded_target!r}, band plan={plan.target_layer}",
    )
    check(
        "model_identity",
        str(extension_report.get("protocol_version") or "").startswith(
            "research-grade-early-layer"
        ),
        f"protocol_version={extension_report.get('protocol_version')!r}",
    )

    payload = {
        "protocol": BAND_LENS_PROTOCOL,
        "scale": int(scale),
        "model_repo_id": str(model_repo_id),
        "model_revision": str(model_revision),
        "band_capture_plan": plan.to_dict(),
        "extension_corpus_id": corpus.get("corpus_id"),
        "extension_corpus_revision": corpus.get("revision"),
        "extension_min_chars": corpus.get("min_chars"),
        "extension_fit_prefix_checksum": fit_prefixes.get(str(int(scale))),
        "extension_split_checksums": checksums,
        "extension_fresh_splits_manifest_checksum": (
            dict(extension_report.get("fresh_splits") or {})
        ).get("manifest_checksum"),
        "checks": checks,
        "failed_checks": [row["check"] for row in checks if not row["ok"]],
        "passed": not problems,
        "same_estimator": "jlens.fitting.fit (upstream, unmodified)",
        "difference_from_the_extension_fit": (
            "source_layers only; the corpus, ordering, scale, positions, target "
            "layer and estimator are identical"
        ),
        "seeded_from_the_extension_accumulator": False,
        "why_not_seeded": (
            "the extension accumulator holds jacobian_sum for a different layer "
            "grid; upstream refuses that resume, and appending five new layers to "
            "it would claim 250 prompts for matrices that saw none"
        ),
    }
    payload["equivalence_checksum"] = payload_checksum(payload)
    if problems:
        raise BandLensRefused(
            "the extension run's artifacts do not establish that a new fit would "
            "be corpus-equivalent to the scale-"
            f"{scale} lenses it must sit beside:\n  - " + "\n  - ".join(problems)
            + "\nRefusing rather than silently mixing lenses from different "
            "corpora into one band."
        )
    return payload


# ------------------------------------------------------------------ the store


class BandLensStore(CalibrationStore):
    """A calibration store that also accepts the band stages.

    Subclassed rather than widening :data:`CALIBRATION_STAGES`, so the completed
    calibration and extension stage vocabularies stay exactly as they are.
    """

    def stage_dir(self, stage: str) -> Path:
        if stage not in BAND_STAGES:
            raise ValueError(
                f"unknown band stage {stage!r}; known stages are {BAND_STAGES}"
            )
        return self.root / "units" / stage

    def status_report(self, stages: Sequence[str] = BAND_STAGES) -> dict:
        return super().status_report(stages)

    def snapshot_path(self, scale: int) -> Path:
        return self.root / "artifacts" / f"lens.band.scale{int(scale)}.pt"

    def published_path(self, layer: int, scale: int) -> Path:
        return (
            self.root
            / "artifacts"
            / "published"
            / f"lens.band.layer{int(layer)}.scale{int(scale)}.validated.pt"
        )


# --------------------------------------------------------- the held-out sets


def verify_reconstructed_extension_splits(
    splits: FreshEvaluationSplits, *, extension_report: Mapping
) -> dict:
    """Prove a re-derived copy of the extension's fresh sets is the same draw.

    The extension recorded checksums and record ids, not text, so the only way
    to exclude its development and confirmation sets is to rebuild them under
    the extension's own seed and prove the rebuild reproduces those checksums.

    Raises:
        BandLensRefused: On any disagreement — an unproven reconstruction would
            exclude the wrong records and leave the real ones in the new sets.
    """
    recorded = dict(extension_report.get("fresh_splits") or {})
    expected = dict(recorded.get("checksums") or {})
    rows = []
    for name in ("development", "confirmation"):
        actual = splits.checksum(name)
        rows.append(
            {
                "partition": name,
                "expected_checksum": expected.get(name),
                "actual_checksum": actual,
                "expected_size": (recorded.get("sizes") or {}).get(name),
                "actual_size": len(splits.get(name)),
                "matches": expected.get(name) == actual,
            }
        )
    payload = {
        "protocol": BAND_LENS_PROTOCOL,
        "extension_split_protocol": recorded.get("protocol"),
        "extension_manifest_checksum": recorded.get("manifest_checksum"),
        "partitions": rows,
        "all_match": all(row["matches"] for row in rows),
        "why": (
            "the extension's development and confirmation sets have been opened "
            "and read; they are excluded from the band's sets, and an exclusion "
            "list can only be trusted if the reconstruction that produced it is "
            "proved to be the same draw"
        ),
    }
    payload["verification_checksum"] = payload_checksum(payload)
    if not payload["all_match"]:
        failed = [row for row in rows if not row["matches"]]
        raise BandLensRefused(
            f"rebuilding the extension's fresh evaluation sets did not reproduce "
            f"its recorded checksums: {failed}. The corpus stream, the pool, the "
            "exclusions or the split seed differs, so the records this run would "
            "exclude are not the ones the extension actually spent. Refusing."
        )
    return payload


def build_band_evaluation_splits(
    pool: Sequence[CorpusRecord],
    *,
    excluded: Mapping[str, Sequence[CorpusRecord]],
    corpus_id: str,
    seed: int = BAND_SPLIT_SEED,
    n_development: int = EXTENSION_GATE.n_prompts,
    n_confirmation: int = EXTENSION_CONFIRMATION_GATE.n_prompts,
    max_hamming: int = MAX_HAMMING_DISTANCE,
) -> tuple[FreshEvaluationSplits, dict]:
    """Third-generation development and confirmation sets, plus their audit.

    The construction, the exclusion arithmetic and the independent leakage
    audit are the extension's own, unchanged — only the seed and the protocol
    label are new, and the exclusion map is expected to carry the extension's
    spent sets on top of the parent's.

    Raises:
        BandLensRefused: If the caller did not offer the extension's own
            development and confirmation sets for exclusion. Those two sets are
            the whole reason this generation exists.
    """
    required = {"extension_development", "extension_confirmation"}
    missing = sorted(required - set(excluded))
    if missing:
        raise BandLensRefused(
            f"the exclusion map is missing {missing}. The extension's fresh sets "
            "have been opened and their verdicts read; a band lens confirmed on "
            "them would be confirmed on development history."
        )
    splits = build_fresh_evaluation_splits(
        pool,
        excluded=excluded,
        corpus_id=corpus_id,
        seed=int(seed),
        n_development=int(n_development),
        n_confirmation=int(n_confirmation),
        max_hamming=int(max_hamming),
    )
    splits = replace(splits, protocol=BAND_SPLIT_PROTOCOL)
    audit = audit_fresh_split_leakage(splits, excluded=excluded, max_hamming=max_hamming)
    return splits, audit


def band_scale_selection(
    *,
    scale: int = BAND_SCALE,
    equivalence: Mapping,
) -> dict:
    """The 'selection' the confirmation vault opens against.

    There is no scale to choose here and that is the point: the band's scale is
    fixed by :func:`assert_corpus_equivalence` — it must equal the scale of the
    lenses the new ones will sit beside — so it is decided before any band
    development number exists, and it cannot be re-decided afterwards. The
    payload is shaped like
    :func:`jlens.calibration.extension.select_extension_scale`'s so
    :class:`~jlens.calibration.publication.ConfirmationVault` can gate on it
    unchanged.
    """
    payload = {
        "selected_scale": int(scale),
        "selection_rule_version": BAND_LENS_PROTOCOL,
        "clause_applied": "fixed_by_corpus_equivalence_with_the_confirmed_band_layers",
        "reason": (
            f"a band may not mix scales; the confirmed L32/L35/L38/L40 lenses this "
            f"band must join are scale {int(scale)}, so the missing layers are "
            "fitted at that scale and no other scale is a candidate"
        ),
        "candidate_scales": [int(scale)],
        "declared_before_results": True,
        "confirmation_not_consulted": True,
        "development_not_consulted_for_the_scale": True,
        "equivalence_checksum": equivalence.get("equivalence_checksum"),
    }
    payload["selection_checksum"] = payload_checksum(payload)
    return payload


# ------------------------------------------------------------- the verdict


def band_layer_verdict(
    confirmation: Mapping[int, Mapping],
    *,
    scale: int,
    selection: Mapping,
    development: Mapping[int, Mapping] | None = None,
    interior_layers: Sequence[int] = BAND_INTERIOR_LAYERS,
    already_confirmed_layers: Sequence[int] = (),
    window: tuple[int, int] = BAND_WINDOW,
) -> dict:
    """GO or NO-GO for the band's interior layers, from confirmation alone.

    ``GO`` requires **every** interior layer to have passed the frozen gate on
    the untouched third-generation confirmation set, because a band is only
    available when every physical layer in it is confirmed. A partial pass is
    reported with the largest admissible contiguous sub-band, computed from
    layer geometry alone — no causal outcome exists yet, and none is consulted.
    """
    from jlens.mmpilot.band_swap import largest_admissible_band

    results = {int(layer): dict(row) for layer, row in confirmation.items()}
    interior = tuple(int(layer) for layer in interior_layers)
    passed = sorted(
        layer for layer in interior if results.get(layer, {}).get("passed", False)
    )
    failed = sorted(layer for layer in interior if layer not in set(passed))
    usable = sorted({*passed, *(int(layer) for layer in already_confirmed_layers)})
    admissible = largest_admissible_band(usable, low=window[0], high=window[1])

    layers = []
    for layer in sorted(results):
        row = results[layer]
        metrics = (row.get("metrics") or {}).get("j_lens", {})
        layers.append(
            {
                "layer": layer,
                "normalized_depth": normalized_depth(layer),
                "role": "band_interior" if layer in set(interior) else "reported_only",
                "confirmation_passed": bool(row.get("passed", False)),
                "confirmation_failed_checks": list(row.get("failed_checks", [])),
                "mean_reciprocal_rank": metrics.get("mean_reciprocal_rank"),
                "median_midrank": metrics.get("median_midrank"),
                "top10_inclusion": metrics.get("top10_inclusion"),
                "tied_at_max_rate": metrics.get("tied_at_max_rate"),
                "development_passed": (
                    bool((development or {}).get(layer, {}).get("passed", False))
                    if development
                    else None
                ),
                "publishable": layer in set(interior),
            }
        )

    payload = {
        "schema": "jlens.mmpilot.band_interior_lens_verdict.v1",
        "protocol": BAND_LENS_PROTOCOL,
        "verdict": BAND_LAYER_GO if interior and not failed else BAND_LAYER_NO_GO,
        "scale": int(scale),
        "selection_checksum": selection.get("selection_checksum"),
        "interior_layers": list(interior),
        "interior_layers_passing": passed,
        "interior_layers_failing": failed,
        "already_confirmed_layers": sorted(int(x) for x in already_confirmed_layers),
        "usable_layers_after_this_run": usable,
        "full_band_available": bool(
            admissible == (int(window[0]), int(window[1]))
        ),
        "largest_admissible_contiguous_band": (
            None if admissible is None else list(admissible)
        ),
        "sub_band_selected_by": (
            "layer geometry only: longest contiguous run of confirmed layers, "
            "ties to the shallowest start; no causal outcome is consulted and "
            "none exists at this stage"
        ),
        "layers": layers,
        "statement": (
            "every interior layer passed the frozen gate on the untouched "
            f"third-generation confirmation set at scale {int(scale)}; the "
            f"contiguous band L{window[0]}-L{window[1]} is available"
            if interior and not failed
            else (
                f"layer(s) {failed} did not pass the frozen gate on the untouched "
                f"confirmation set at scale {int(scale)}. Every metric is "
                "reported, nothing is published for them, and the band is "
                "reduced to the largest confirmed contiguous run."
            )
        ),
        "all_layer_results_recorded": True,
    }
    return {**payload, "verdict_checksum": payload_checksum(payload)}


# ----------------------------------------------------------------- publish


def publish_band_layer(
    *,
    layer: int,
    scale: int,
    lens,
    destination: Path | str,
    confirmation_verdict: Mapping,
    development_verdict: Mapping,
    vault: ConfirmationVault,
    splits: FreshEvaluationSplits,
    selection: Mapping,
    equivalence: Mapping,
    band_run_dir: Path | str,
    interior_layers: Sequence[int] = BAND_INTERIOR_LAYERS,
    protected_run_dirs: Sequence[Path | str] = (),
    gate: CalibrationGate = EXTENSION_CONFIRMATION_GATE,
    **artifact_fields,
) -> dict:
    """Publish one interior layer that passed the untouched confirmation gate.

    Every refusal :func:`jlens.calibration.publication.publish_layer` enforces
    applies unchanged — a locked vault, a failed verdict, a development verdict
    offered in its place, a layer/scale mismatch, a protected path. Three are
    added:

    * a layer outside ``interior_layers``, so an existing confirmed lens is
      never republished under a new name;
    * a destination outside the band run;
    * a destination inside any protected run (the parent calibration and the
      early-layer extension), which are read-only evidence.

    Writes the lens, the standard artifact sidecar, and a ``.band.json``
    sidecar carrying the corpus-equivalence proof and the third-generation
    split checksums.
    """
    destination = Path(destination)
    run_dir = Path(band_run_dir).resolve()
    resolved = destination.resolve()

    if int(layer) not in {int(value) for value in interior_layers}:
        raise PublicationRefused(
            f"layer {layer} is not a band-interior publication target "
            f"{sorted(int(v) for v in interior_layers)}. The already-confirmed "
            "band layers keep their own artifacts; this run neither replaces nor "
            "republishes them."
        )
    # The protected check runs first: "you are writing into completed evidence"
    # is the more important thing to say when both refusals apply.
    for protected in protected_run_dirs:
        protected_root = Path(protected).resolve()
        if protected_root == resolved or protected_root in resolved.parents:
            raise PublicationRefused(
                f"{destination} is inside {protected_root}, which is completed-run "
                "evidence and is never written to."
            )
    if run_dir not in resolved.parents:
        raise PublicationRefused(
            f"{destination} is outside the band run directory {run_dir}. Band "
            "artifacts are written only into the band run."
        )

    artifact = publish_layer(
        layer=int(layer),
        scale=int(scale),
        lens=lens,
        destination=destination,
        confirmation_verdict=confirmation_verdict,
        validation_verdict=development_verdict,
        vault=vault,
        gate=gate,
        protocol_version=BAND_LENS_PROTOCOL,
        **artifact_fields,
    )

    band_artifact = {
        "schema": BAND_LENS_ARTIFACT_SCHEMA,
        "protocol": BAND_LENS_PROTOCOL,
        "frozen": True,
        "validated": bool(confirmation_verdict.get("passed", False)),
        "publication_status": BAND_PUBLISHED_STATUS,
        "physical_layer": int(layer),
        "normalized_depth": normalized_depth(int(layer)),
        "scale_point": int(scale),
        "band_window": list(BAND_WINDOW),
        "fitted_for": "a contiguous-band coordinate swap; every band layer is patched",
        # --- what makes it comparable to the lenses it joins
        "corpus_equivalence": dict(equivalence),
        "corpus_equivalence_checksum": equivalence.get("equivalence_checksum"),
        "seeded_from_another_accumulator": False,
        # --- identity and geometry, copied from the base artifact
        "model_repo_id": artifact.get("model_repo_id"),
        "model_revision": artifact.get("model_revision"),
        "tokenizer_repo_id": artifact.get("tokenizer_repo_id"),
        "tokenizer_revision": artifact.get("tokenizer_revision"),
        "target_layer": artifact.get("target_layer"),
        "hook_site": artifact.get("hook_site"),
        "residual_convention": artifact.get("residual_convention"),
        "n_fitting_prompts": artifact.get("n_fitting_prompts"),
        # --- the third-generation sets
        "split_protocol": splits.protocol,
        "development_split_checksum": splits.checksum("development"),
        "confirmation_split_checksum": splits.checksum("confirmation"),
        "fresh_splits_manifest_checksum": splits.manifest()["manifest_checksum"],
        "extension_sets_excluded": True,
        "parent_sets_excluded": True,
        # --- how it was judged
        "gate_version": gate.version,
        "gate_digest": gate.digest,
        "selection_checksum": selection.get("selection_checksum"),
        "development_metrics": development_verdict.get("metrics"),
        "development_failed_checks": list(development_verdict.get("failed_checks", [])),
        "confirmation_metrics": confirmation_verdict.get("metrics"),
        "confirmation_failed_checks": list(confirmation_verdict.get("failed_checks", [])),
        "target_diversity": confirmation_verdict.get("target_diversity"),
        # --- the file
        "lens_path": artifact.get("lens_path"),
        "lens_checksum": artifact.get("lens_checksum"),
        "base_artifact_checksum": artifact.get("artifact_checksum"),
        "calibration_modality": "text-only",
        "spokencoco_used": False,
        "multimodal_data_used": False,
        "existing_publications_unchanged": True,
    }
    band_artifact["artifact_checksum"] = payload_checksum(band_artifact)

    sidecar = destination.with_suffix(".band.json")
    temporary = sidecar.with_name(f"{sidecar.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(band_artifact, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, sidecar)
    return band_artifact


# ----------------------------------------------------------------- discovery


@dataclass(frozen=True)
class BandLensSource:
    """Where one band layer's ``J_l`` is actually read from.

    A band assembles Jacobians from more than one artifact — a published
    single-layer file for L32, the extension's multi-layer scale-250 snapshot
    for L35/L38/L40, and this study's own published files for the interior —
    so every layer records the file, the checksum of that file, and the layer
    key inside it.
    """

    layer: int
    path: str
    checksum: str
    layer_key_in_file: int
    provenance: str

    def to_dict(self) -> dict:
        return asdict(self)


def _read_json(path: Path, *, what: str) -> dict:
    if not path.is_file():
        raise BandLensRefused(f"{what} not found at {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise BandLensRefused(f"{what} at {path} is not readable JSON: {error}") from error
    if not isinstance(payload, dict):
        raise BandLensRefused(f"{what} at {path} does not hold a JSON object")
    return payload


def _unit_payload(store_dir: Path, stage: str, key: str, *, what: str) -> dict:
    record = _read_json(store_dir / "units" / stage / f"{key}.json", what=what)
    payload = record.get("payload") if isinstance(record.get("payload"), dict) else record
    if payload_checksum(payload) != record.get("unit_checksum", payload_checksum(payload)):
        raise BandLensRefused(
            f"{what} in {store_dir} does not match its own unit checksum; a torn "
            "unit is not evidence"
        )
    return dict(payload)


def discover_extension_scale250_lenses(
    extension_run_dir: str | os.PathLike[str],
    *,
    snapshot_layers: Sequence[int] = (35, 38, 40),
    published_layers: Sequence[int] = (32,),
    scale: int = BAND_SCALE,
) -> tuple[list[BandLensRow], dict[int, BandLensSource], dict]:
    """Locate every scale-250 lens the extension run already holds.

    Two chains, both starting from the run's own report:

    ``published_layers``
        Resolved through :func:`jlens.mmpilot.l32_followup.discover_published_l32_lens`,
        which proves the file is the published artifact and that the report
        says so. Nothing is renamed and no filename is assumed.

    ``snapshot_layers``
        Present inside the run's scale-250 **snapshot** with recorded
        confirmation verdicts from the same fresh set, but never published as
        separate artifacts because the extension's publication targets were L26
        and L32. The snapshot's path and checksum come from the run's own
        ``scale_snapshot`` unit and the verdicts from its ``confirmation``
        unit; the file is re-checksummed here and refused on any disagreement.

    Returns:
        ``(rows, sources, evidence)`` — inventory rows for
        :func:`jlens.mmpilot.band_swap.lens_inventory`, the per-layer read
        instructions, and the discovery evidence for the report.
    """
    from jlens.mmpilot.l32_followup import (
        discover_published_l32_lens,
        read_extension_report,
    )

    run_dir = Path(extension_run_dir)
    report_path, report = read_extension_report(run_dir)
    rows: list[BandLensRow] = []
    sources: dict[int, BandLensSource] = {}
    evidence: dict = {
        "extension_run_dir": str(run_dir),
        "report_path": str(report_path),
        "report_checksum": file_sha256(str(report_path)),
        "scale": int(scale),
    }

    corpus = dict(report.get("corpus") or {})
    fresh = dict(report.get("fresh_splits") or {})
    corpus_label = (
        f"{corpus.get('corpus_id')}@{corpus.get('revision')} "
        f"(fit prefix {(corpus.get('fit_prefix_checksums') or {}).get(str(int(scale)))})"
    )
    validation_label = (
        f"{fresh.get('protocol')} confirmation "
        f"{(fresh.get('checksums') or {}).get('confirmation')}"
    )

    for layer in sorted(int(value) for value in published_layers):
        discovered = discover_published_l32_lens(
            run_dir, layer=layer, expected_scale=int(scale)
        )
        _, artifact = read_artifact_sidecar(Path(discovered.lens_path))
        rows.append(
            BandLensRow(
                layer=layer,
                lens_path=str(discovered.lens_path),
                lens_checksum=str(discovered.lens_checksum),
                fit_scale=int(artifact.get("scale_point", scale)),
                fit_corpus=corpus_label,
                validation_set=validation_label,
                confirmation_verdict=(
                    "PASSED (published artifact, "
                    f"failed checks {artifact.get('confirmation_failed_checks')})"
                ),
                usable=True,
                reason="published, independently confirmed scale-250 artifact",
                provenance="early_layer_extension_published_artifact",
            )
        )
        sources[layer] = BandLensSource(
            layer=layer,
            path=str(discovered.lens_path),
            checksum=str(discovered.lens_checksum),
            layer_key_in_file=layer,
            provenance="early_layer_extension_published_artifact",
        )
        evidence[f"published_L{layer}"] = discovered.to_dict()

    wanted = sorted(int(value) for value in snapshot_layers)
    if wanted:
        snapshot = _unit_payload(
            run_dir, "scale_snapshot", f"scale{int(scale)}", what="scale snapshot unit"
        )
        snapshot_path = Path(snapshot.get("path", ""))
        if not snapshot_path.is_file():
            snapshot_path = run_dir / "artifacts" / Path(snapshot.get("path", "")).name
        if not snapshot_path.is_file():
            raise BandLensRefused(
                f"the extension run records a scale-{scale} snapshot at "
                f"{snapshot.get('path')!r}, which is not present in {run_dir}"
            )
        actual = file_sha256(str(snapshot_path))
        if actual != snapshot.get("checksum"):
            raise BandLensRefused(
                f"the scale-{scale} snapshot at {snapshot_path} checksums to "
                f"{actual}, not the recorded {snapshot.get('checksum')!r}"
            )
        snapshot_layers_present = {int(value) for value in snapshot.get("layers", [])}
        confirmation = _unit_payload(
            run_dir, "confirmation", f"scale{int(scale)}", what="confirmation unit"
        )
        by_layer = {int(k): v for k, v in (confirmation.get("by_layer") or {}).items()}
        evidence["snapshot"] = {
            "path": str(snapshot_path),
            "checksum": actual,
            "layers": sorted(snapshot_layers_present),
            "confirmation_unit_layers": sorted(by_layer),
        }
        for layer in wanted:
            verdict = dict(by_layer.get(layer) or {})
            present = layer in snapshot_layers_present
            passed = bool(verdict.get("passed", False))
            usable = bool(present and passed)
            if usable:
                reason = (
                    "fitted at scale 250 in the extension's own snapshot and "
                    "passed that run's untouched confirmation set; not separately "
                    "published because the extension's publication targets were "
                    "L26 and L32"
                )
            elif not present:
                reason = f"layer {layer} is not in the scale-{scale} snapshot"
            else:
                reason = (
                    f"recorded scale-{scale} confirmation did not pass "
                    f"(failed checks {verdict.get('failed_checks')})"
                )
            rows.append(
                BandLensRow(
                    layer=layer,
                    lens_path=str(snapshot_path) if present else None,
                    lens_checksum=actual if present else None,
                    fit_scale=int(scale) if present else None,
                    fit_corpus=corpus_label,
                    validation_set=validation_label,
                    confirmation_verdict=(
                        ("PASSED" if passed else "FAILED")
                        + f" (failed checks {verdict.get('failed_checks')})"
                        if verdict
                        else None
                    ),
                    usable=usable,
                    reason=reason,
                    provenance="early_layer_extension_scale250_snapshot",
                )
            )
            if usable:
                sources[layer] = BandLensSource(
                    layer=layer,
                    path=str(snapshot_path),
                    checksum=actual,
                    layer_key_in_file=layer,
                    provenance="early_layer_extension_scale250_snapshot",
                )
    evidence["rows"] = [row.to_dict() for row in rows]
    evidence["discovery_checksum"] = payload_checksum(evidence)
    return rows, sources, evidence


def discover_published_band_lenses(
    band_run_dir: str | os.PathLike[str],
    *,
    layers: Sequence[int] = BAND_INTERIOR_LAYERS,
    scale: int = BAND_SCALE,
) -> tuple[list[BandLensRow], dict[int, BandLensSource], dict]:
    """Locate the interior lenses this study published, from its own report.

    The chain mirrors the extension's: the band report says the layer passed
    and publishes a checksum for it; exactly one ``*.band.json`` sidecar in the
    band run's published directory claims that layer at that scale with
    ``validated`` and the published status; the lens path comes from that
    sidecar; the file's own sha256 must equal both the sidecar's and the
    report's. A filename is never assumed and a checksum is never invented.
    """
    run_dir = Path(band_run_dir)
    report_path = run_dir / "artifacts" / "band_interior_lens_report.json"
    report = _read_json(report_path, what="band interior lens report")
    if report.get("schema") != BAND_LENS_REPORT_SCHEMA:
        raise BandLensRefused(
            f"{report_path} declares schema {report.get('schema')!r}, not "
            f"{BAND_LENS_REPORT_SCHEMA!r}"
        )
    if report.get("mode") != "real":
        raise BandLensRefused(
            f"{report_path} was written in mode {report.get('mode')!r}; a MOCK "
            "report publishes nothing about Gemma"
        )
    publication = dict(report.get("publication") or {})
    published_layers = {int(value) for value in publication.get("published_layers", [])}
    checksums = {
        str(k): str(v) for k, v in (publication.get("published_checksums") or {}).items()
    }
    published_dir = run_dir / "artifacts" / "published"

    rows: list[BandLensRow] = []
    sources: dict[int, BandLensSource] = {}
    evidence: dict = {
        "band_run_dir": str(run_dir),
        "report_path": str(report_path),
        "report_checksum": file_sha256(str(report_path)),
        "published_layers": sorted(published_layers),
    }
    fresh = dict(report.get("fresh_splits") or {})
    validation_label = (
        f"{fresh.get('protocol')} confirmation "
        f"{(fresh.get('checksums') or {}).get('confirmation')}"
    )
    corpus_label = str(
        (report.get("corpus_equivalence") or {}).get("extension_corpus_id")
    ) + f"@{(report.get('corpus_equivalence') or {}).get('extension_corpus_revision')}"

    for layer in sorted(int(value) for value in layers):
        if layer not in published_layers:
            rows.append(
                BandLensRow(
                    layer=layer,
                    usable=False,
                    reason=(
                        f"the band report does not publish layer {layer} "
                        f"(published: {sorted(published_layers)})"
                    ),
                    provenance="band_interior_run",
                )
            )
            continue
        candidates = []
        for sidecar in sorted(published_dir.glob("*.band.json")):
            payload = _read_json(sidecar, what="band sidecar")
            if (
                payload.get("schema") == BAND_LENS_ARTIFACT_SCHEMA
                and int(payload.get("physical_layer", -1)) == layer
                and int(payload.get("scale_point", -1)) == int(scale)
                and payload.get("validated") is True
                and payload.get("publication_status") == BAND_PUBLISHED_STATUS
            ):
                candidates.append((sidecar, payload))
        if len(candidates) != 1:
            raise BandLensRefused(
                f"{len(candidates)} published band sidecars in {published_dir} claim "
                f"layer {layer} at scale {scale}; which artifact a band rests on is "
                "not decided by sort order"
            )
        sidecar_path, artifact = candidates[0]
        recomputed = payload_checksum(
            {k: v for k, v in artifact.items() if k != "artifact_checksum"}
        )
        if recomputed != artifact.get("artifact_checksum"):
            raise BandLensRefused(
                f"{sidecar_path} does not match its own artifact_checksum"
            )
        lens_path = published_dir / Path(str(artifact.get("lens_path", ""))).name
        if not lens_path.is_file():
            raise BandLensRefused(f"{sidecar_path} names a lens absent from {published_dir}")
        actual = file_sha256(str(lens_path))
        if actual != artifact.get("lens_checksum") or actual != checksums.get(str(layer)):
            raise BandLensRefused(
                f"the layer {layer} lens at {lens_path} is not the published file "
                f"(file {actual}, sidecar {artifact.get('lens_checksum')}, report "
                f"{checksums.get(str(layer))})"
            )
        rows.append(
            BandLensRow(
                layer=layer,
                lens_path=str(lens_path),
                lens_checksum=actual,
                fit_scale=int(artifact.get("scale_point", scale)),
                fit_corpus=corpus_label,
                validation_set=validation_label,
                confirmation_verdict=(
                    "PASSED (band interior, failed checks "
                    f"{artifact.get('confirmation_failed_checks')})"
                ),
                usable=True,
                reason="published by this study after untouched confirmation",
                provenance="band_interior_run_published_artifact",
            )
        )
        sources[layer] = BandLensSource(
            layer=layer,
            path=str(lens_path),
            checksum=actual,
            layer_key_in_file=layer,
            provenance="band_interior_run_published_artifact",
        )
    evidence["rows"] = [row.to_dict() for row in rows]
    evidence["discovery_checksum"] = payload_checksum(evidence)
    return rows, sources, evidence


# -------------------------------------------------------------------- budget

#: The operator's own anchor, carried over from the extension: 100 prompts in
#: 7.1 minutes on one L4, fitting eight source layers with a backward span of
#: 33 blocks (L8 -> L41).
ANCHOR_MINUTES = 7.1
ANCHOR_PROMPTS = 100
ANCHOR_BACKWARD_SPAN = 33

#: Wider than the extension's band on the high side, because this fit also
#: changes the backward span and the span-scaling is itself a model.
BAND_UNCERTAINTY_BAND = (0.9, 2.0)


def band_fit_budget(
    *,
    plan: CapturePlan,
    scale: int = BAND_SCALE,
    n_development: int = EXTENSION_GATE.n_prompts,
    n_confirmation: int = EXTENSION_CONFIRMATION_GATE.n_prompts,
    observed_minutes: float = ANCHOR_MINUTES,
    anchor_prompts: int = ANCHOR_PROMPTS,
    anchor_backward_span: int = ANCHOR_BACKWARD_SPAN,
) -> dict:
    """What Stage 1 and Stage 2 cost, and the two ways of estimating it.

    The anchor fitted layers 8-40 with a backward span of 33 blocks. This fit's
    shallowest layer is 33, so each backward pass traverses far fewer blocks —
    upstream starts the graph at ``min(source_layers)``. Two numbers are
    reported and both are shown, because the span-scaled one is a model and the
    unscaled one is the observation:

    ``span_scaled_minutes``   the anchor scaled by ``backward_span``;
    ``unscaled_minutes``      the anchor applied per prompt with no scaling —
                              the pessimistic planning number.
    """
    per_prompt = float(observed_minutes) / float(anchor_prompts)
    span_ratio = plan.backward_span / float(anchor_backward_span)
    span_scaled = per_prompt * span_ratio * int(scale)
    unscaled = per_prompt * int(scale)
    low, high = BAND_UNCERTAINTY_BAND
    readout_prompts = int(n_development) + int(n_confirmation)
    matrix_bytes = plan.d_model * plan.d_model
    n_layers = len(plan.layers)
    payload = {
        "protocol": BAND_LENS_PROTOCOL,
        "anchor": {
            "source": "the operator's observed scale-100 fit on one NVIDIA L4",
            "n_prompts": int(anchor_prompts),
            "minutes": float(observed_minutes),
            "backward_span": int(anchor_backward_span),
        },
        "plan": {
            "layers": list(plan.layers),
            "backward_span": plan.backward_span,
            "backward_passes_per_prompt": plan.backward_passes_per_prompt,
            "forwards_per_prompt": plan.forwards_per_prompt,
            "all_layers_in_one_pass": True,
        },
        "fit": {
            "n_prompts": int(scale),
            "forward_passes": int(scale) * plan.forwards_per_prompt,
            "backward_passes": int(scale) * plan.backward_passes_per_prompt,
            "span_scaled_minutes": round(span_scaled, 1),
            "span_scaled_hours": round(span_scaled / 60.0, 2),
            "unscaled_minutes": round(unscaled, 1),
            "unscaled_hours": round(unscaled / 60.0, 2),
            "planning_hours_low": round(span_scaled * low / 60.0, 2),
            "planning_hours_high": round(unscaled * high / 60.0, 2),
            "span_ratio": round(span_ratio, 4),
        },
        "readout": {
            "n_prompts": readout_prompts,
            "forward_passes": readout_prompts,
            "minutes_low": round(2 + 4 * readout_prompts / 128.0, 1),
            "minutes_high": round(2 + 12 * readout_prompts / 128.0, 1),
            "why": (
                "development and confirmation are forward-pass workloads; the "
                f"{plan.backward_passes_per_prompt} backward passes per prompt "
                "happen only during fitting"
            ),
        },
        "storage_bytes": {
            "checkpoint": n_layers * matrix_bytes * 4,
            "snapshot": n_layers * matrix_bytes * 2,
            "published_max": matrix_bytes * 2 * n_layers,
            "reports": 20 * 1024 * 1024,
        },
        "resume": (
            "the accumulator is written atomically every checkpoint_every "
            "prompts and reloaded automatically; a disconnect loses at most one "
            "checkpoint interval of fitting"
        ),
        "extrapolation_not_measurement": True,
    }
    payload["storage_bytes"]["total"] = sum(
        value for key, value in payload["storage_bytes"].items() if key != "total"
    )
    payload["budget_checksum"] = payload_checksum(payload)
    return payload


def format_band_fit_budget(budget: Mapping) -> str:
    """The block printed before any fitting switch may be set."""
    anchor, plan, fit, readout = (
        budget["anchor"],
        budget["plan"],
        budget["fit"],
        budget["readout"],
    )
    storage = budget["storage_bytes"]
    return "\n".join(
        [
            "BAND INTERIOR LENS BUDGET — five layers, one pass, one scale",
            f"  anchor           {anchor['n_prompts']} prompts in {anchor['minutes']:.1f} min "
            f"on one L4 at backward span {anchor['backward_span']}",
            f"  this fit         layers {plan['layers']} -> backward span "
            f"{plan['backward_span']} ({fit['span_ratio']:.2f}x the anchor's)",
            f"  per prompt       1 forward + {plan['backward_passes_per_prompt']} backward passes",
            "",
            f"  fitting {fit['n_prompts']} prompts",
            f"    span-scaled    {fit['span_scaled_minutes']:.1f} min "
            f"({fit['span_scaled_hours']:.2f} h)",
            f"    unscaled       {fit['unscaled_minutes']:.1f} min "
            f"({fit['unscaled_hours']:.2f} h)  <- pessimistic planning number",
            f"    planning range {fit['planning_hours_low']:.2f}-{fit['planning_hours_high']:.2f} h",
            "",
            f"  readout          {readout['n_prompts']} held-out prompts, "
            f"{readout['minutes_low']:.0f}-{readout['minutes_high']:.0f} min",
            f"  Drive            {storage['total'] / 2**30:.2f} GiB",
            f"  resume           {budget['resume']}",
            "",
            "  EXTRAPOLATION, NOT MEASUREMENT. The span scaling is a model of the",
            "  cost, not an observation of it; plan with the unscaled row.",
        ]
    )
