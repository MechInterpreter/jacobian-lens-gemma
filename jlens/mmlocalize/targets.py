# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Stage A: the completed run is read-only, and the targets are frozen first.

Two things have to be true before a single layer result exists, or the depth
comparison this package reports is not a comparison.

**The evidence it conditions on must be the evidence it names.** The completed
six-concept robustness run is verified by fingerprint and by artifact checksum
and then only read. Nothing here writes into it, and a fingerprint mismatch
refuses rather than proceeds — a localization conditioned on "some run in that
directory" would inherit a provenance nobody can state.

**The target images must be chosen before any layer is scored.** If layer 32
were evaluated on images picked after layer 38's numbers came back, the
difference between them would be partly a difference in photographs.
:func:`freeze_targets` fixes the set, checksums it, and
:func:`assert_same_targets_across_layers` refuses any later drift.

Two target policies, decided before results and never mixed
-----------------------------------------------------------

:data:`POLICY_FRESH_DISJOINT` is preferred: target photographs that the
completed robustness run never touched. It buys an independent test of the same
question rather than a re-reading of the same images.

:data:`POLICY_REUSED_PAIRED` is the fallback when the derived cache cannot
supply enough fresh validated groups. It is a **paired within-sample layer
comparison** and nothing more: the images already produced the layer-38 result,
so layer 38's reproduction on them is not independent evidence. Every artifact
and every report carries that limitation in words, because a reader who missed
it would over-read the result.

:func:`choose_target_policy` decides between them from availability alone, and
the decision is recorded in the manifest that the run fingerprint binds.

Concepts are conditioned, not sampled
-------------------------------------

The localization concepts are **cat** and **toilet** — fixed, because they are
the two that replicated bidirectionally in the completed run. That makes this a
depth question asked of concepts already known to work. It is deliberately
**not** a new estimate of how prevalent cross-modal transfer is across concepts,
and :data:`CONCEPT_CONDITIONING_LIMITATION` says so in every artifact.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from jlens.mmpilot.store import payload_checksum

#: The two concepts. Fixed before anything runs; never swapped for a third.
LOCALIZATION_CONCEPTS: tuple[str, ...] = ("cat", "toilet")

#: Printed and written into every artifact that reports a localization result.
CONCEPT_CONDITIONING_LIMITATION = (
    "Localization is conditioned on cat and toilet because those two, and only "
    "those two, transferred bidirectionally in the completed robustness run. "
    "Every number here is therefore 'how early does transfer appear for concepts "
    "already known to transfer at layer 38', not 'how many concepts transfer'. "
    "This design cannot estimate concept-general prevalence and does not try."
)

#: Fresh photographs, disjoint from everything the completed run touched.
POLICY_FRESH_DISJOINT = "fresh_image_disjoint_from_completed_run.v1"

#: The completed run's own images, reused as a paired within-sample comparison.
POLICY_REUSED_PAIRED = "reused_completed_run_images_paired_within_sample.v1"

#: Stated in full wherever the fallback policy is used.
REUSED_POLICY_LIMITATION = (
    "FALLBACK POLICY IN USE: the target photographs are the completed robustness "
    "run's own images. This is a paired WITHIN-SAMPLE layer comparison. Layer "
    "38's reproduction on these images is not independent evidence — the layer-38 "
    "result was measured on them — so only the earlier-versus-38 contrasts carry "
    "information, and they carry it about these photographs rather than about "
    "held-out ones."
)

#: Distinct images each role needs, per concept. Half the robustness study's
#: eight: this is a four-layer paired depth comparison on two known-good
#: concepts, not a prevalence estimate, and the passes scale with the layer count.
N_SOURCE_POSITIVE_IMAGES = 4
N_SOURCE_NEGATIVE_IMAGES = 4
N_TARGET_POSITIVE_IMAGES = 4
N_TARGET_NEGATIVE_IMAGES = 4

#: Bound into the run fingerprint.
TARGET_MANIFEST_VERSION = "mmlocalize.target_manifest.v1"


class CompletedRunError(RuntimeError):
    """The completed robustness run is missing, moved, or not the named one."""


class TargetPolicyError(RuntimeError):
    """The frozen target set violates the policy it was frozen under."""


class TargetDriftError(RuntimeError):
    """A layer was about to be scored on images other than the frozen ones."""


# -------------------------------------------------- the completed run, read-only


def verify_completed_run(
    run_dir: str | Path,
    *,
    expect_fingerprint: str,
    expect_verdict: str = "ROBUSTNESS_GO",
) -> dict:
    """Verify the completed run by fingerprint and verdict. Reads only.

    Args:
        expect_fingerprint: The run's immutable ``sha256:`` digest. Compared
            against the digest recomputed from ``fingerprint.json`` **and**
            against the digest the run recorded in its own summary, so a
            hand-edited summary cannot agree with a hand-edited fingerprint.

    Returns:
        A serialisable record of what was observed, including the checksum of
        every top-level artifact found.

    Raises:
        CompletedRunError: On a missing directory, a missing fingerprint, a
            digest mismatch, or a verdict other than ``expect_verdict``.
    """
    root = Path(run_dir)
    if not root.is_dir():
        raise CompletedRunError(
            f"the completed robustness run was not found at {root}. This study "
            "conditions on that run; it is not discovered automatically."
        )

    fingerprint_path = root / "fingerprint.json"
    if not fingerprint_path.is_file():
        raise CompletedRunError(
            f"{fingerprint_path} is missing: a run without its fingerprint cannot "
            "be shown to be the run this study names"
        )
    stored = json.loads(fingerprint_path.read_text(encoding="utf-8"))
    stored.pop("written_utc", None)
    observed = payload_checksum(stored)
    if observed != expect_fingerprint:
        raise CompletedRunError(
            f"{root} has fingerprint {observed}, not the expected "
            f"{expect_fingerprint}. Refusing to condition a localization on a run "
            "other than the one it names."
        )

    summary_path = root / "robustness_summary.json"
    summary: dict = {}
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        recorded = summary.get("fingerprint_digest")
        if recorded and recorded != expect_fingerprint:
            raise CompletedRunError(
                f"{summary_path} records fingerprint {recorded} while "
                f"{fingerprint_path} hashes to {observed}; the run's own artifacts "
                "disagree about what it is"
            )
        verdict = (summary.get("verdict") or {}).get("verdict")
        if verdict != expect_verdict:
            raise CompletedRunError(
                f"{root} reports verdict {verdict!r}, not {expect_verdict!r}. "
                "This study localizes a confirmed result; it does not localize an "
                "unconfirmed one."
            )

    artifacts = {
        path.name: payload_checksum(path.read_text(encoding="utf-8"))
        for path in sorted(root.glob("*.json"))
    }
    return {
        "run_dir": str(root),
        "fingerprint": observed,
        "fingerprint_matches_pin": True,
        "verdict": (summary.get("verdict") or {}).get("verdict"),
        "artifact_checksums": artifacts,
        "read_only": True,
        "reused": (
            "selected concepts, image identities, split provenance and capability "
            "results, read-only"
        ),
    }


def completed_run_images(run_dir: str | Path) -> dict:
    """Every image the completed run touched, split by role. Reads only.

    ``causal_target_images`` is the set the brief requires fresh targets to
    avoid; ``all_images`` is everything the run touched at all, which is what
    :data:`POLICY_FRESH_DISJOINT` actually excludes. Excluding the larger set is
    strictly safer and costs nothing when the cache is large enough.
    """
    root = Path(run_dir)
    causal: set[str] = set()
    every: set[str] = set()

    for stage, sink in (("intervention", causal), ("activation", None), ("capability", None)):
        stage_dir = root / "units" / stage
        if not stage_dir.is_dir():
            continue
        for path in sorted(stage_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8")).get("payload") or {}
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            image_id = payload.get("image_id")
            if image_id in (None, ""):
                continue
            every.add(str(image_id))
            if sink is not None:
                sink.add(str(image_id))

    summary_path = root / "robustness_summary.json"
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            summary = {}
        for row in (summary.get("interventions_image_level") or {}).get("rows", []):
            for image_id in row.get("image_ids", ()):
                causal.add(str(image_id))
                every.add(str(image_id))

    return {
        "run_dir": str(root),
        "causal_target_images": sorted(causal),
        "all_images": sorted(every),
        "n_causal_target_images": len(causal),
        "n_all_images": len(every),
        "source": "per-unit artifacts, read-only",
    }


# ------------------------------------------------------------- policy choice


def choose_target_policy(
    *,
    n_available_fresh_images: Mapping[str, int],
    n_required_per_concept: int = (
        N_SOURCE_POSITIVE_IMAGES + N_TARGET_POSITIVE_IMAGES
    ),
    n_required_negatives: int = (
        N_SOURCE_NEGATIVE_IMAGES + N_TARGET_NEGATIVE_IMAGES
    ),
    n_available_fresh_negatives: int = 0,
    concepts: Sequence[str] = LOCALIZATION_CONCEPTS,
) -> dict:
    """Pick the target policy from availability alone, before any layer runs.

    Returns a decision record naming the policy, the counts it was decided
    from, and — when the fallback is chosen — the limitation that must then
    appear in every artifact. The two policies are never mixed: either every
    target is fresh or every target is reused.
    """
    shortfalls = {
        concept: {
            "available": int(n_available_fresh_images.get(concept, 0)),
            "required": int(n_required_per_concept),
        }
        for concept in concepts
        if int(n_available_fresh_images.get(concept, 0)) < n_required_per_concept
    }
    negatives_short = int(n_available_fresh_negatives) < int(n_required_negatives)
    feasible = not shortfalls and not negatives_short

    return {
        "policy": POLICY_FRESH_DISJOINT if feasible else POLICY_REUSED_PAIRED,
        "fresh_targets_feasible": feasible,
        "per_concept_available": {
            concept: int(n_available_fresh_images.get(concept, 0)) for concept in concepts
        },
        "required_positive_images_per_concept": int(n_required_per_concept),
        "available_negative_images": int(n_available_fresh_negatives),
        "required_negative_images": int(n_required_negatives),
        "shortfalls": shortfalls,
        "negatives_short": negatives_short,
        "limitation": None if feasible else REUSED_POLICY_LIMITATION,
        "policies_are_never_mixed": True,
        "decided_before_any_layer_result": True,
    }


# -------------------------------------------------------------- frozen targets


@dataclass(frozen=True)
class LocalizationTargets:
    """The image set every layer is scored on, frozen before the first result.

    Attributes:
        source_positive_images / source_negative_images: Per concept, the
            training photographs a direction is estimated from.
        target_positive_images / target_negative_images: Per concept, the
            held-out photographs every layer intervenes on. The *same* images at
            every layer — that is what makes the depth contrast paired.
    """

    policy: str
    concepts: tuple[str, ...]
    source_positive_images: dict[str, list[str]]
    source_negative_images: dict[str, list[str]]
    target_positive_images: dict[str, list[str]]
    target_negative_images: dict[str, list[str]]
    excluded_completed_run_images: list[str] = field(default_factory=list)
    limitation: str | None = None

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["concepts"] = list(self.concepts)
        return payload

    @property
    def checksum(self) -> str:
        return payload_checksum(self.to_dict())

    def all_target_images(self) -> list[str]:
        return sorted(
            {
                image
                for mapping in (self.target_positive_images, self.target_negative_images)
                for images in mapping.values()
                for image in images
            }
        )

    def all_source_images(self) -> list[str]:
        return sorted(
            {
                image
                for mapping in (self.source_positive_images, self.source_negative_images)
                for images in mapping.values()
                for image in images
            }
        )


def freeze_targets(
    *,
    policy: str,
    source_positive_images: Mapping[str, Sequence[str]],
    source_negative_images: Mapping[str, Sequence[str]],
    target_positive_images: Mapping[str, Sequence[str]],
    target_negative_images: Mapping[str, Sequence[str]],
    completed_run_images: Sequence[str] = (),
    concepts: Sequence[str] = LOCALIZATION_CONCEPTS,
) -> LocalizationTargets:
    """Validate and freeze the target set. Called once, before any layer runs.

    Enforces, in order: every concept present; the required distinct-image
    counts; source and target images disjoint; positive and negative images
    disjoint; and — under :data:`POLICY_FRESH_DISJOINT` — no overlap at all with
    the completed run.

    Raises:
        TargetPolicyError: On the first violation, naming what was asked for and
            what was supplied. A silently shrunk or overlapping cell reports an
            ``n`` it never had.
    """
    if policy not in (POLICY_FRESH_DISJOINT, POLICY_REUSED_PAIRED):
        raise TargetPolicyError(f"unknown target policy {policy!r}")

    requirements = (
        ("source positives", source_positive_images, N_SOURCE_POSITIVE_IMAGES),
        ("source negatives", source_negative_images, N_SOURCE_NEGATIVE_IMAGES),
        ("target positives", target_positive_images, N_TARGET_POSITIVE_IMAGES),
        ("target negatives", target_negative_images, N_TARGET_NEGATIVE_IMAGES),
    )
    for role, mapping, required in requirements:
        for concept in concepts:
            images = sorted(set(mapping.get(concept, ())))
            if len(images) < required:
                raise TargetPolicyError(
                    f"{concept} / {role}: {len(images)} distinct image(s) but the "
                    f"design states {required}. Refusing to proceed on a smaller "
                    "set than the design states."
                )

    frozen = LocalizationTargets(
        policy=policy,
        concepts=tuple(concepts),
        source_positive_images={
            c: sorted(set(source_positive_images.get(c, ())))[:N_SOURCE_POSITIVE_IMAGES]
            for c in concepts
        },
        source_negative_images={
            c: sorted(set(source_negative_images.get(c, ())))[:N_SOURCE_NEGATIVE_IMAGES]
            for c in concepts
        },
        target_positive_images={
            c: sorted(set(target_positive_images.get(c, ())))[:N_TARGET_POSITIVE_IMAGES]
            for c in concepts
        },
        target_negative_images={
            c: sorted(set(target_negative_images.get(c, ())))[:N_TARGET_NEGATIVE_IMAGES]
            for c in concepts
        },
        excluded_completed_run_images=sorted({str(x) for x in completed_run_images}),
        limitation=None if policy == POLICY_FRESH_DISJOINT else REUSED_POLICY_LIMITATION,
    )

    sources = set(frozen.all_source_images())
    targets = set(frozen.all_target_images())
    overlap = sorted(sources & targets)
    if overlap:
        raise TargetPolicyError(
            f"{len(overlap)} photograph(s) appear as both a source-training image "
            f"and a causal target: {overlap[:8]}. A direction estimated from an "
            "image cannot then be tested on it."
        )
    for concept in concepts:
        shared = sorted(
            set(frozen.target_positive_images[concept])
            & set(frozen.target_negative_images[concept])
        )
        if shared:
            raise TargetPolicyError(
                f"{concept}: {len(shared)} photograph(s) are both a held-out "
                f"positive and a held-out negative: {shared[:8]}"
            )

    if policy == POLICY_FRESH_DISJOINT:
        reused = sorted((sources | targets) & set(frozen.excluded_completed_run_images))
        if reused:
            raise TargetPolicyError(
                f"the fresh-target policy was requested but {len(reused)} "
                f"photograph(s) were already used by the completed robustness run: "
                f"{reused[:8]}. Either exclude them or declare the reused-paired "
                "policy — the two are never mixed."
            )
    return frozen


def audit_image_exclusions(
    targets: LocalizationTargets,
    *,
    completed_run: Mapping,
    n_available_images: int | None = None,
) -> dict:
    """The exact image-exclusion audit written beside the manifest.

    Counts rather than adjectives: how many photographs the completed run
    touched, how many were excluded, how many survived, and the intersection
    (which must be empty under the fresh policy, and is reported either way).
    """
    completed_all = {str(x) for x in completed_run.get("all_images", ())}
    completed_causal = {str(x) for x in completed_run.get("causal_target_images", ())}
    used = set(targets.all_source_images()) | set(targets.all_target_images())

    return {
        "version": TARGET_MANIFEST_VERSION,
        "policy": targets.policy,
        "completed_run_dir": completed_run.get("run_dir"),
        "n_completed_run_images": len(completed_all),
        "n_completed_run_causal_target_images": len(completed_causal),
        "n_images_used_by_this_study": len(used),
        "overlap_with_completed_run_all": sorted(used & completed_all),
        "overlap_with_completed_run_causal_targets": sorted(used & completed_causal),
        "n_overlap_all": len(used & completed_all),
        "n_overlap_causal_targets": len(used & completed_causal),
        "fresh_policy_satisfied": (
            targets.policy != POLICY_FRESH_DISJOINT or not (used & completed_all)
        ),
        "n_available_images_before_exclusion": n_available_images,
        "source_target_overlap": sorted(
            set(targets.all_source_images()) & set(targets.all_target_images())
        ),
        "per_concept": {
            concept: {
                "source_positive": targets.source_positive_images[concept],
                "source_negative": targets.source_negative_images[concept],
                "target_positive": targets.target_positive_images[concept],
                "target_negative": targets.target_negative_images[concept],
            }
            for concept in targets.concepts
        },
        "limitation": targets.limitation,
    }


def target_manifest(
    targets: LocalizationTargets,
    *,
    audit: Mapping,
    completed_run: Mapping,
    layers: Sequence[int],
) -> dict:
    """The fingerprinted localization manifest, frozen before any layer result."""
    payload = {
        "version": TARGET_MANIFEST_VERSION,
        "concepts": list(targets.concepts),
        "concept_conditioning_limitation": CONCEPT_CONDITIONING_LIMITATION,
        "policy": targets.policy,
        "policy_limitation": targets.limitation,
        "layers": [int(layer) for layer in layers],
        "same_targets_at_every_layer": True,
        "targets": targets.to_dict(),
        "target_checksum": targets.checksum,
        "image_exclusion_audit": dict(audit),
        "completed_run": {
            "run_dir": completed_run.get("run_dir"),
            "fingerprint": completed_run.get("fingerprint"),
            "verdict": completed_run.get("verdict"),
            "read_only": True,
        },
        "frozen_before_any_layer_result": True,
    }
    payload["manifest_checksum"] = payload_checksum(payload)
    return payload


def assert_same_targets_across_layers(
    targets: LocalizationTargets, observed_by_layer: Mapping[int, Sequence[str]]
) -> dict:
    """Refuse if any layer was scored on images other than the frozen ones.

    The depth contrast is only paired if the photographs are identical across
    layers. A layer that quietly got a different set would produce a difference
    that is partly about images and would read as a difference about depth.

    Raises:
        TargetDriftError: Naming each layer whose image set diverged.
    """
    expected = set(targets.all_target_images())
    divergences = {}
    for layer, images in sorted(observed_by_layer.items()):
        observed = {str(image) for image in images}
        if observed != expected:
            divergences[int(layer)] = {
                "missing": sorted(expected - observed),
                "unexpected": sorted(observed - expected),
            }
    if divergences:
        raise TargetDriftError(
            "the frozen target set was not used at every layer, so the depth "
            f"comparison is not paired: {divergences}"
        )
    return {
        "paired": True,
        "n_target_images": len(expected),
        "layers_checked": sorted(int(layer) for layer in observed_by_layer),
        "target_checksum": targets.checksum,
    }


def format_targets(targets: LocalizationTargets, audit: Mapping) -> str:
    """The block the notebook prints once the targets are frozen."""
    lines = [
        "=" * 72,
        "FROZEN LOCALIZATION TARGETS — fixed before any layer result exists",
        "=" * 72,
        f"  policy            {targets.policy}",
        f"  concepts          {list(targets.concepts)}",
        f"  target checksum   {targets.checksum}",
        f"  images used       {audit['n_images_used_by_this_study']}",
        f"  completed run     {audit['n_completed_run_images']} image(s) touched, "
        f"{audit['n_completed_run_causal_target_images']} of them causal targets",
        f"  overlap with it   {audit['n_overlap_all']} "
        f"(causal targets: {audit['n_overlap_causal_targets']})",
        f"  source/target overlap {audit['source_target_overlap'] or 'none'}",
        "",
    ]
    for concept in targets.concepts:
        entry = audit["per_concept"][concept]
        lines.append(
            f"  {concept:8s} source +{len(entry['source_positive'])} "
            f"-{len(entry['source_negative'])}  "
            f"target +{len(entry['target_positive'])} "
            f"-{len(entry['target_negative'])} distinct photographs"
        )
    lines += ["", f"  {CONCEPT_CONDITIONING_LIMITATION}"]
    if targets.limitation:
        lines += ["", f"  {targets.limitation}"]
    return "\n".join(lines)


__all__ = [
    "CONCEPT_CONDITIONING_LIMITATION",
    "LOCALIZATION_CONCEPTS",
    "N_SOURCE_NEGATIVE_IMAGES",
    "N_SOURCE_POSITIVE_IMAGES",
    "N_TARGET_NEGATIVE_IMAGES",
    "N_TARGET_POSITIVE_IMAGES",
    "POLICY_FRESH_DISJOINT",
    "POLICY_REUSED_PAIRED",
    "REUSED_POLICY_LIMITATION",
    "TARGET_MANIFEST_VERSION",
    "CompletedRunError",
    "LocalizationTargets",
    "TargetDriftError",
    "TargetPolicyError",
    "assert_same_targets_across_layers",
    "audit_image_exclusions",
    "choose_target_policy",
    "completed_run_images",
    "format_targets",
    "freeze_targets",
    "target_manifest",
    "verify_completed_run",
]
