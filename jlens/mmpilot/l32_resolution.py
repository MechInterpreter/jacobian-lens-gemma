# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""The independent layer-32 convergence-resolution study.

The completed open-prompt follow-up
(``mml32_l32_followup_20260808T182717``) classified physical layer 32
``AMBIGUOUS`` under the frozen native direct-readout criterion. Its own
predeclared next step — :func:`jlens.mmpilot.l32_followup.adjacent_layer_recommendation`,
written before that result was visible — was a *separately fingerprinted,
independently sampled* layer-32 population under the same open protocol and the
same frozen thresholds. This module is that study.

What it is not
==============

It is **not an attempt to reach ``NOT_CONVERGED``**. The thresholds are frozen
and their digest is checked; the concepts are frozen in ranking order and are
never replaced after a result; the sample size is fixed before any model output
is observed; and all three outcomes are reported in the same words. An
``AMBIGUOUS`` independent population is a finding — it says the layer really
does sit between the bars, which is not a sample-size problem and is not fixed
by drawing again.

It is also not a resumption of anything. Every completed run this study reads —
the calibration, the extension, the native-audio transfer, the robustness,
localization, convergence-audit and L32 follow-up runs — is opened read-only and
checksummed before and after. Nothing here writes outside its own
``mml32res_*`` directory.

What "independent" has to mean, concretely
==========================================

A fresh population is not "another random draw". Four identities have to be
disjoint from the completed run, and each of them can fail on its own:

``image_id``
    The photograph. The independent unit of every rate this study reports.

``group_id``
    The synchronized (image, caption, recording) triple. Two groups of one
    photograph are not two observations.

audio path / recording
    The recording actually fed to the audio tower. A caption reused with a
    different speaker is still the same sentence.

caption text
    The written string. Text-modality units of the *same* sentence are the same
    measurement whatever their ids say.

:func:`harvest_excluded_identities` reads all four out of the completed runs'
own artifacts, :func:`resolve_excluded_media` maps the recorded group ids back
through the manifest to recover the recordings and captions that the unit files
do not store, and :func:`audit_population_disjointness` **verifies** the four
sets rather than trusting that filtering worked. A study that cannot build the
required independent population refuses; it never shrinks quietly.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from jlens.mmpilot.convergence import (
    AMBIGUOUS,
    CONTROL_VARIANTS,
    CONVERGED,
    CONVERGENCE_CRITERION,
    CONVERGENCE_PROTOCOL,
    MODALITIES,
    NOT_CONVERGED,
)
from jlens.mmpilot.l32_followup import (
    CONVERGENCE_PHRASE,
    INTERVENTION_FAMILY,
    L32_LAYER,
    OPEN_PROMPT_PROTOCOL,
    REFERENCE_LAYER,
    SELECTED_SCALE,
)
from jlens.mmpilot.store import payload_checksum

# ------------------------------------------------------------------ versions

#: Bound into the run fingerprint. A change refuses to resume anything produced
#: under the older rule.
L32_RESOLUTION_PROTOCOL = "mmpilot.l32_convergence_resolution.v1"

#: Run directories this study creates. Distinct from every completed family
#: (``rgcal_``, ``rgext_``, ``mmaudio_``, ``mmrobust_``, ``mmlocalize_``,
#: ``mmconv_``, ``mml32_``), so a resume can never land in one of them.
RESOLUTION_RUN_PREFIX = "mml32res"

#: How the fresh population is chosen. Versioned separately from the study,
#: because a change in *how the sample was drawn* invalidates the independence
#: claim even when nothing else moved.
POPULATION_SELECTION_VERSION = "mmpilot.l32_resolution_independent_population.v1"

#: How the excluded identities were recovered from the completed runs.
EXCLUSION_HARVEST_VERSION = "mmpilot.l32_resolution_exclusion_harvest.v1"

#: The disjointness proof's own version.
DISJOINTNESS_AUDIT_VERSION = "mmpilot.l32_resolution_disjointness.v1"

#: The predeclared sample-size rule.
SAMPLE_SIZE_RULE_VERSION = "mmpilot.l32_resolution_sample_size.v1"

#: The Stage-A / Stage-B plan and its conditional gate.
STAGE_PLAN_VERSION = "mmpilot.l32_resolution_stage_plan.v2_conditional"

# --------------------------------------------------------------- the numbers

#: The six candidates, frozen in the ranking order the completed studies fixed.
#: Never re-ranked, never substituted, and never reduced on the strength of a
#: capability or convergence result.
SELECTED_CONCEPTS: tuple[str, ...] = (
    "zebra",
    "cat",
    "toilet",
    "giraffe",
    "bird",
    "microwave",
)

#: The three focal concepts. Also frozen: an ineligible focal concept is
#: *excluded from evidentiary verdicts*, never replaced by a fourth.
FOCAL_CONCEPTS: tuple[str, ...] = ("zebra", "cat", "toilet")

#: The completed run this study is independent *of*. Read-only, and named here
#: so the exclusion set has a default rather than a notebook literal.
COMPLETED_FOLLOWUP_RUN = "mml32_l32_followup_20260808T182717"

#: Its fingerprint, as reported by the completed study.
COMPLETED_FOLLOWUP_FINGERPRINT = (
    "sha256:29aa1d1351a0d6a609d8d074265b8c060d4353d7b9d99f22db6df3a6cf75b67e"
)

#: The frozen criterion digest. Checked, never recomputed into agreement.
FROZEN_CRITERION_DIGEST = (
    "sha256:abbb23e1b4a114982f3962cb74b47875d539437b65ba84da7d9f3ed0446b5a1f"
)

# ------------------------------------------------------------------ verdicts

VERDICT_NAME = "L32_INDEPENDENT_CONVERGENCE"

L32_INDEPENDENT_CONVERGED = "L32_INDEPENDENT_CONVERGED"
L32_INDEPENDENT_NOT_CONVERGED = "L32_INDEPENDENT_NOT_CONVERGED"
L32_INDEPENDENT_AMBIGUOUS = "L32_INDEPENDENT_AMBIGUOUS"
REFUSED_INVALID = "REFUSED_INVALID"

#: The four, and only the four.
RESOLUTION_VERDICTS: tuple[str, ...] = (
    L32_INDEPENDENT_NOT_CONVERGED,
    L32_INDEPENDENT_CONVERGED,
    L32_INDEPENDENT_AMBIGUOUS,
    REFUSED_INVALID,
)

#: How a classification maps to a verdict, once validity is established.
_CLASSIFICATION_TO_VERDICT = {
    CONVERGED: L32_INDEPENDENT_CONVERGED,
    NOT_CONVERGED: L32_INDEPENDENT_NOT_CONVERGED,
    AMBIGUOUS: L32_INDEPENDENT_AMBIGUOUS,
}


class ResolutionRefused(RuntimeError):
    """A precondition of the resolution study does not hold."""


class ExclusionEvidenceMissing(ResolutionRefused):
    """The completed run's identities could not be recovered, so independence
    cannot be *verified* — only assumed, which is not the same thing."""


class PopulationNotIndependent(ResolutionRefused):
    """The fresh population shares an identity with the completed run."""


class PseudoreplicationError(ResolutionRefused):
    """A photograph or a recording appears more than once in the population."""


# ============================================================ exclusion set


#: Where per-unit identities live in a completed run. Every one of these is
#: optional — a run that predates a directory simply contributes nothing — but
#: at least one must yield identities or the study refuses.
UNIT_GLOBS: tuple[str, ...] = (
    "units/capability/*.json",
    "units/activation/*.json",
    "units/jspace/*.json",
    "units/direction/*.json",
    "units/intervention/*.json",
    "convergence/readout_units/*.json",
)

#: Top-level documents that may carry identities in bulk.
IDENTITY_DOCUMENTS: tuple[str, ...] = (
    "population_manifest.json",
    "split_provenance.json",
    "fingerprint.json",
    "run_manifest.json",
)

#: Keys that hold an identity, wherever they appear in a unit payload.
_IDENTITY_KEYS = {
    "image_id": "image_ids",
    "group_id": "group_ids",
    "sample_id": "sample_ids",
    "recording_id": "group_ids",
    "audio_path": "audio_paths",
    "caption": "captions",
    "media_checksum": "media_checksums",
}

#: The four identity families disjointness is proven over.
IDENTITY_FAMILIES: tuple[str, ...] = (
    "image_ids",
    "group_ids",
    "audio_paths",
    "captions",
)


@dataclass
class ExclusionSet:
    """Every identity the fresh population must avoid, and where it came from."""

    image_ids: set[str] = field(default_factory=set)
    group_ids: set[str] = field(default_factory=set)
    sample_ids: set[str] = field(default_factory=set)
    audio_paths: set[str] = field(default_factory=set)
    captions: set[str] = field(default_factory=set)
    media_checksums: set[str] = field(default_factory=set)
    sources: list[dict] = field(default_factory=list)
    run_dirs: list[str] = field(default_factory=list)

    @property
    def n_identities(self) -> int:
        return sum(
            len(getattr(self, family)) for family in IDENTITY_FAMILIES
        ) + len(self.sample_ids)

    def counts(self) -> dict[str, int]:
        return {
            "image_ids": len(self.image_ids),
            "group_ids": len(self.group_ids),
            "sample_ids": len(self.sample_ids),
            "audio_paths": len(self.audio_paths),
            "captions": len(self.captions),
            "media_checksums": len(self.media_checksums),
        }

    @property
    def digest(self) -> str:
        """A checksum over the identities themselves, not over their count.

        Bound into the run fingerprint as ``exclusion_run_checksum``: two runs
        that excluded different photographs are different experiments even when
        they excluded the same *number* of them.

        Run directories are digested by **name** and recordings by **recording
        name**, for the reason :func:`selection_digest` gives — a Drive mount is
        not part of an experiment's identity, and a study re-run from a
        remounted Drive must not refuse to resume itself.
        """
        return payload_checksum(
            {
                "harvest_version": EXCLUSION_HARVEST_VERSION,
                "media_identity": "recording_name_not_resolved_path",
                "run_dirs": sorted(Path(name).name for name in self.run_dirs),
                "audio_paths": sorted(
                    Path(str(path)).name for path in self.audio_paths
                ),
                **{
                    name: sorted(getattr(self, name))
                    for name in (
                        "image_ids",
                        "group_ids",
                        "sample_ids",
                        "captions",
                        "media_checksums",
                    )
                },
            }
        )

    def to_dict(self) -> dict:
        return {
            "schema": "jlens.mmpilot.l32_resolution_exclusion_set.v1",
            "harvest_version": EXCLUSION_HARVEST_VERSION,
            "run_dirs": sorted(self.run_dirs),
            "counts": self.counts(),
            "n_identities": self.n_identities,
            "exclusion_digest": self.digest,
            "sources": [dict(entry) for entry in self.sources],
            "image_ids": sorted(self.image_ids),
            "group_ids": sorted(self.group_ids),
            "audio_paths": sorted(self.audio_paths),
            "captions": sorted(self.captions),
            "sample_ids": sorted(self.sample_ids),
            "media_checksums": sorted(self.media_checksums),
        }


def _absorb(target: ExclusionSet, payload: object, *, depth: int = 0) -> int:
    """Pull every identity key out of an arbitrarily nested payload.

    Walking rather than reading fixed paths, because the completed runs do not
    agree on shape: a capability unit stores ``image_id`` at the top level, a
    convergence readout unit stores it inside ``payload``, and a fingerprint
    stores sample ids in a list under a name that has changed twice. A walk
    finds all three and cannot be broken by the next one.
    """
    if depth > 8:
        return 0
    found = 0
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            attribute = _IDENTITY_KEYS.get(str(key))
            if attribute is not None and isinstance(value, (str, int)):
                text = str(value).strip()
                if text and text.lower() not in ("none", "null"):
                    getattr(target, attribute).add(text)
                    found += 1
                continue
            if str(key) in ("source_sample_ids", "target_sample_ids"):
                for item in value if isinstance(value, (list, tuple)) else ():
                    text = str(item).strip()
                    if text:
                        target.sample_ids.add(text)
                        target.group_ids.add(text)
                        found += 1
                continue
            found += _absorb(target, value, depth=depth + 1)
    elif isinstance(payload, (list, tuple)):
        for item in payload:
            found += _absorb(target, item, depth=depth + 1)
    return found


def harvest_excluded_identities(
    run_dirs: Sequence[str | os.PathLike[str]],
    *,
    unit_globs: Sequence[str] = UNIT_GLOBS,
    documents: Sequence[str] = IDENTITY_DOCUMENTS,
    require: bool = True,
) -> ExclusionSet:
    """Read the completed runs' own artifacts for the identities to avoid.

    Every file is opened read-only. Unreadable files are recorded and skipped
    rather than failing the harvest — a torn unit in a completed run is that
    run's problem, and skipping it only makes the exclusion set *smaller*, which
    is the direction that gets caught by
    :func:`audit_population_disjointness` rather than the direction that hides.

    Args:
        require: Refuse when no identity could be recovered from any run. The
            default, because "no overlap found" and "nothing was looked at" are
            indistinguishable in the artifact otherwise.

    Raises:
        ExclusionEvidenceMissing: When ``require`` and nothing was recovered.
    """
    exclusion = ExclusionSet()
    for run_dir in run_dirs:
        root = Path(run_dir)
        entry: dict = {
            "run_dir": str(root),
            "exists": root.is_dir(),
            "n_files_read": 0,
            "n_identities": 0,
            "unreadable": [],
        }
        if root.is_dir():
            exclusion.run_dirs.append(str(root))
            paths: list[Path] = []
            for pattern in unit_globs:
                paths.extend(sorted(root.glob(pattern)))
            for name in documents:
                candidate = root / name
                if candidate.is_file():
                    paths.append(candidate)
            for path in paths:
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError, UnicodeDecodeError) as exc:
                    entry["unreadable"].append({"path": str(path), "error": str(exc)})
                    continue
                entry["n_files_read"] += 1
                entry["n_identities"] += _absorb(exclusion, payload)
        exclusion.sources.append(entry)

    if require and exclusion.n_identities == 0:
        raise ExclusionEvidenceMissing(
            "no image, group, recording or caption identity could be read from "
            f"{[str(d) for d in run_dirs]}. Independence would then be an "
            "assumption rather than a verified property, and this study's whole "
            "claim is that the population is independent. Refusing to select.\n"
            f"Sources examined: {json.dumps(exclusion.sources, indent=2)}"
        )
    return exclusion


def resolve_excluded_media(
    exclusion: ExclusionSet, groups: Sequence[Mapping]
) -> dict:
    """Recover the recordings and captions behind the excluded group ids.

    Unit files store ``group_id`` and ``image_id``; they do not store the audio
    path or the caption text. Those two identities are recoverable from the
    manifest, because ``group_id`` is a hash of exactly
    ``image_id|caption|audio`` — so a group id present in the manifest names one
    recording and one caption unambiguously.

    Mutates ``exclusion`` in place, adding what it resolved, and returns a
    record saying how much of the excluded set it could account for. An
    unresolved group id is reported rather than dropped: it means the completed
    run used media this manifest does not contain, and the ``group_id`` and
    ``image_id`` exclusions still apply to it.
    """
    by_group = {str(group["group_id"]): group for group in groups}
    resolved = 0
    unresolved: list[str] = []
    before = {
        "audio_paths": len(exclusion.audio_paths),
        "captions": len(exclusion.captions),
    }
    for group_id in sorted(exclusion.group_ids):
        group = by_group.get(group_id)
        if group is None:
            unresolved.append(group_id)
            continue
        if group.get("audio_path"):
            exclusion.audio_paths.add(str(group["audio_path"]))
        if group.get("caption"):
            exclusion.captions.add(str(group["caption"]))
        if group.get("image_id"):
            exclusion.image_ids.add(str(group["image_id"]))
        resolved += 1

    # An excluded image is excluded wholesale: every sibling caption of a
    # photograph the completed run used is the same photograph.
    for group in groups:
        if str(group.get("image_id")) in exclusion.image_ids:
            if group.get("audio_path"):
                exclusion.audio_paths.add(str(group["audio_path"]))
            if group.get("caption"):
                exclusion.captions.add(str(group["caption"]))
            exclusion.group_ids.add(str(group["group_id"]))

    return {
        "schema": "jlens.mmpilot.l32_resolution_media_resolution.v1",
        "harvest_version": EXCLUSION_HARVEST_VERSION,
        "n_excluded_group_ids": len(exclusion.group_ids),
        "n_resolved_in_manifest": resolved,
        "n_unresolved": len(unresolved),
        "unresolved_examples": unresolved[:10],
        "audio_paths_added": len(exclusion.audio_paths) - before["audio_paths"],
        "captions_added": len(exclusion.captions) - before["captions"],
        "sibling_expansion_note": (
            "every group sharing an excluded photograph is excluded too, so a "
            "sibling caption of a used image can never enter the fresh "
            "population"
        ),
        "counts": exclusion.counts(),
        "exclusion_digest": exclusion.digest,
    }


def independent_pool(
    groups: Sequence[Mapping], exclusion: ExclusionSet
) -> tuple[list[dict], dict]:
    """The groups the fresh population may be drawn from, plus the exclusion log.

    Filtering happens **before** selection rather than after, so the selection
    rule sees only admissible candidates and its determinism is a property of
    the pool it was given. Filtering afterwards would make the chosen set depend
    on which excluded groups happened to be ranked ahead of it.
    """
    kept: list[dict] = []
    dropped: dict[str, int] = {family: 0 for family in IDENTITY_FAMILIES}
    dropped_examples: list[dict] = []
    for group in groups:
        reasons = []
        if str(group.get("image_id")) in exclusion.image_ids:
            reasons.append("image_ids")
        if str(group.get("group_id")) in exclusion.group_ids:
            reasons.append("group_ids")
        if str(group.get("audio_path")) in exclusion.audio_paths:
            reasons.append("audio_paths")
        if str(group.get("caption")) in exclusion.captions:
            reasons.append("captions")
        if reasons:
            for reason in reasons:
                dropped[reason] += 1
            if len(dropped_examples) < 20:
                dropped_examples.append(
                    {
                        "group_id": str(group.get("group_id")),
                        "image_id": str(group.get("image_id")),
                        "reasons": reasons,
                    }
                )
            continue
        kept.append(dict(group))

    record = {
        "schema": "jlens.mmpilot.l32_resolution_pool.v1",
        "selection_version": POPULATION_SELECTION_VERSION,
        "n_groups_available": len(groups),
        "n_groups_excluded": len(groups) - len(kept),
        "n_groups_in_independent_pool": len(kept),
        "n_distinct_images_in_pool": len({str(g.get("image_id")) for g in kept}),
        "excluded_by_identity": dict(dropped),
        "excluded_examples": dropped_examples,
        "exclusion_digest": exclusion.digest,
        "note": (
            "exclusion is applied to the pool before any selection rule runs, "
            "so the chosen set does not depend on where excluded groups sat in "
            "the ranking"
        ),
    }
    return kept, record


# ======================================================= disjointness proof


def _subset_identities(subset: Mapping) -> dict[str, set[str]]:
    rows = [
        row
        for split in ("train", "test")
        for row in subset["splits"][split]
    ]
    return {
        "image_ids": {str(row["image_id"]) for row in rows},
        "group_ids": {str(row["group_id"]) for row in rows},
        "audio_paths": {str(row["audio_path"]) for row in rows},
        "captions": {str(row["caption"]) for row in rows},
    }


def audit_population_disjointness(
    subset: Mapping, exclusion: ExclusionSet, *, require: bool = True
) -> dict:
    """Prove — not assume — that the fresh population shares nothing.

    Checked *after* selection, over the population that was actually built,
    because the filter and the proof must be able to disagree. A proof derived
    from the same predicate that did the filtering proves only that the
    predicate is consistent with itself.

    Raises:
        PopulationNotIndependent: On any overlap in any of the four families,
            listing every one rather than the first.
    """
    ours = _subset_identities(subset)
    theirs = {family: getattr(exclusion, family) for family in IDENTITY_FAMILIES}
    overlaps = {
        family: sorted(ours[family] & theirs[family]) for family in IDENTITY_FAMILIES
    }
    failed = [family for family, items in overlaps.items() if items]
    record = {
        "schema": "jlens.mmpilot.l32_resolution_disjointness.v1",
        "audit_version": DISJOINTNESS_AUDIT_VERSION,
        "families_checked": list(IDENTITY_FAMILIES),
        "population_counts": {k: len(v) for k, v in sorted(ours.items())},
        "exclusion_counts": {k: len(v) for k, v in sorted(theirs.items())},
        "overlaps": {k: v[:20] for k, v in sorted(overlaps.items())},
        "n_overlaps": {k: len(v) for k, v in sorted(overlaps.items())},
        "failed_families": failed,
        "disjoint": not failed,
        "exclusion_digest": exclusion.digest,
        "population_digest": payload_checksum(
            {name: sorted(values) for name, values in sorted(ours.items())}
        ),
        "note": (
            "verified over the selected population, not inferred from the "
            "filter that produced it"
        ),
    }
    if require and failed:
        detail = "\n  - ".join(
            f"{family}: {len(overlaps[family])} shared, e.g. {overlaps[family][:5]}"
            for family in failed
        )
        raise PopulationNotIndependent(
            "the fresh population is not independent of the completed run:\n  - "
            f"{detail}\n"
            "This is not repairable by dropping the offending units — the "
            "selection would then depend on the completed run's contents. "
            "Refusing."
        )
    return record


def assert_one_unit_per_photograph(subset: Mapping) -> dict:
    """One group per image and one recording per group, over the whole subset.

    ``check_split_leakage`` proves train and test do not share; this proves the
    population has no pseudoreplication *within* a split either, which is the
    failure that inflates an n without adding an observation.

    Raises:
        PseudoreplicationError: If any photograph contributes two groups, or any
            recording is used twice.
    """
    rows = [row for split in ("train", "test") for row in subset["splits"][split]]
    by_image: dict[str, list[str]] = {}
    by_audio: dict[str, list[str]] = {}
    for row in rows:
        by_image.setdefault(str(row["image_id"]), []).append(str(row["group_id"]))
        by_audio.setdefault(str(row["audio_path"]), []).append(str(row["group_id"]))
    repeated_images = {k: v for k, v in by_image.items() if len(v) > 1}
    repeated_audio = {k: v for k, v in by_audio.items() if len(v) > 1}
    record = {
        "schema": "jlens.mmpilot.l32_resolution_pseudoreplication.v1",
        "n_units": len(rows),
        "n_distinct_images": len(by_image),
        "n_distinct_recordings": len(by_audio),
        "n_images_with_multiple_groups": len(repeated_images),
        "n_recordings_used_twice": len(repeated_audio),
        "repeated_image_examples": dict(list(repeated_images.items())[:5]),
        "repeated_recording_examples": dict(list(repeated_audio.items())[:5]),
        "one_group_per_image": not repeated_images,
        "one_recording_per_unit": not repeated_audio,
        "passed": not repeated_images and not repeated_audio,
    }
    if not record["passed"]:
        raise PseudoreplicationError(
            f"{len(repeated_images)} photograph(s) contribute more than one "
            f"group and {len(repeated_audio)} recording(s) are used twice. "
            "The image-level rates this study reports would then be computed "
            f"over {len(rows)} rows standing on {len(by_image)} observations."
        )
    return record


def selection_digest(subset: Mapping) -> str:
    """A checksum over the exact selected sample, for the run fingerprint.

    Media are digested by **recording name**, not by resolved path. An absolute
    path is a property of the mount, not of the sample: the same population read
    from a differently-mounted Drive would otherwise get a different digest and
    refuse to resume itself, which is the same reasoning that makes
    :class:`~jlens.mmpilot.cache.DerivedCache` record media relative to its
    root. The caption and the group id are already mount-independent, and the
    group id is a hash of ``image_id|caption|audio`` — so the recording is
    pinned exactly whatever its path happens to be today.

    :func:`audit_population_disjointness` still compares *full* paths, because
    both of its sides are read within one session from one mount, and there the
    stricter comparison is free.
    """
    ours = _subset_identities(subset)
    ours["audio_paths"] = {
        Path(str(path)).name for path in ours["audio_paths"]
    }
    return payload_checksum(
        {
            "selection_version": POPULATION_SELECTION_VERSION,
            "media_identity": "recording_name_not_resolved_path",
            **{name: sorted(values) for name, values in sorted(ours.items())},
        }
    )


# ==================================================== sample-size justification


def wilson_interval(k: int, n: int, *, confidence: float = 0.95) -> dict:
    """Wilson score interval for a binomial proportion.

    Wilson rather than Wald because every bar this study is measured against
    sits near 0 or 1 (0.50 and 0.90), and a Wald interval at 0.90 with n in the
    tens runs past 1.0 and stops meaning anything. No SciPy: the normal
    quantile is the only thing needed and it is tabulated below.
    """
    n = int(n)
    if n <= 0:
        return {"n": 0, "k": int(k), "point": None, "low": None, "high": None,
                "half_width": None, "confidence": float(confidence)}
    z = _normal_quantile(1.0 - (1.0 - float(confidence)) / 2.0)
    p = float(k) / n
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    spread = (
        z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
    )
    return {
        "n": n,
        "k": int(k),
        "point": p,
        "low": max(0.0, centre - spread),
        "high": min(1.0, centre + spread),
        "half_width": spread,
        "confidence": float(confidence),
        "method": "wilson_score",
    }


#: Two-sided normal quantiles, tabulated rather than approximated. Only the
#: confidences this study can be run at are here; anything else refuses.
_NORMAL_QUANTILES = {
    0.900: 1.2815515655446004,
    0.950: 1.6448536269514722,
    0.975: 1.9599639845400545,
    0.990: 2.3263478740408408,
    0.995: 2.5758293035489004,
}


def _normal_quantile(cumulative: float) -> float:
    key = round(float(cumulative), 3)
    if key not in _NORMAL_QUANTILES:
        raise ResolutionRefused(
            f"no tabulated normal quantile for {cumulative!r}; this module "
            f"supports {sorted(_NORMAL_QUANTILES)} and will not approximate one "
            "into a predeclared interval"
        )
    return _NORMAL_QUANTILES[key]


def binomial_at_most(k: int, n: int, p: float) -> float:
    """Exact ``P(X <= k)`` for ``X ~ Binomial(n, p)``."""
    n, k = int(n), int(k)
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return sum(
        math.comb(n, i) * (p**i) * ((1.0 - p) ** (n - i)) for i in range(k + 1)
    )


def binomial_at_least(k: int, n: int, p: float) -> float:
    """Exact ``P(X >= k)``."""
    return 1.0 - binomial_at_most(int(k) - 1, n, p)


@dataclass(frozen=True)
class SampleSizeRule:
    """The precision target, fixed before any model output is observed.

    Attributes:
        target_half_width: Largest acceptable 95% Wilson half-width at the
            worst-case proportion ``p = 0.5``. Set to half the 0.40 gap between
            the frozen bars, so a point estimate sitting on one bar has an
            interval that does not reach the other.
        min_power: Smallest acceptable probability of landing on the correct
            side of a bar when the truth is clearly on that side.
        not_converged_truth: The "clearly not converged" rate the lower-bar
            power is computed at. 0.40 rather than 0.50 because a design sized
            to detect a truth sitting exactly on the bar needs an infinite n.
        converged_truth: The "clearly converged" rate for the upper bar.
        expected_admissible_focal: How many focal concepts the design is sized
            for. **Two**, not three: the completed study already found `zebra`
            capability-ineligible in spoken audio under the frozen admissibility
            rule, so sizing for three would be sizing for a population this
            study has positive evidence it will not have.
    """

    target_half_width: float = 0.20
    min_power: float = 0.80
    not_converged_truth: float = 0.40
    converged_truth: float = 0.95
    confidence: float = 0.95
    expected_admissible_focal: int = 2
    rule_version: str = SAMPLE_SIZE_RULE_VERSION

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def digest(self) -> str:
        return payload_checksum(self.to_dict())


#: The one rule this study runs under.
SAMPLE_SIZE_RULE = SampleSizeRule()

#: Candidate per-concept image counts, smallest first. The plan takes the first
#: that satisfies the rule; the ladder is fixed here so "the smallest adequate
#: n" is a lookup rather than a choice made after seeing anything.
IMAGE_COUNT_LADDER: tuple[int, ...] = (8, 12, 16, 20, 24, 32)


def cell_precision(n: int, rule: SampleSizeRule = SAMPLE_SIZE_RULE) -> dict:
    """What a cell of ``n`` independent photographs can and cannot resolve."""
    criterion = CONVERGENCE_CRITERION
    worst_case = wilson_interval(
        k=int(round(0.5 * n)), n=n, confidence=rule.confidence
    )
    lower_bar_k = math.floor(criterion.not_converged_max_clean_agreement * n)
    upper_bar_k = math.ceil(criterion.converged_min_clean_agreement * n)
    return {
        "n": int(n),
        "wilson_half_width_at_p_0.5": worst_case["half_width"],
        "meets_half_width_target": (
            worst_case["half_width"] is not None
            and worst_case["half_width"] <= rule.target_half_width
        ),
        "power_to_observe_not_converged": binomial_at_most(
            lower_bar_k, n, rule.not_converged_truth
        ),
        "power_to_observe_converged": binomial_at_least(
            upper_bar_k, n, rule.converged_truth
        ),
        "lower_bar_max_successes": int(lower_bar_k),
        "upper_bar_min_successes": int(upper_bar_k),
        "meets_power_target": (
            binomial_at_most(lower_bar_k, n, rule.not_converged_truth)
            >= rule.min_power
            and binomial_at_least(upper_bar_k, n, rule.converged_truth)
            >= rule.min_power
        ),
    }


@dataclass(frozen=True)
class SamplePlan:
    """The predeclared design, printed before any model is loaded."""

    images_per_concept: int
    n_train_positive_images: int
    n_test_positive_images: int
    n_train_negative_images: int
    n_test_negative_images: int
    n_selected_concepts: int
    n_focal_concepts: int
    rule: SampleSizeRule
    precision_by_admissible: dict
    adequate: bool
    ladder_considered: tuple[int, ...]

    @property
    def total_positive_images(self) -> int:
        return self.images_per_concept * self.n_selected_concepts

    @property
    def total_negative_images(self) -> int:
        return self.n_train_negative_images + self.n_test_negative_images

    @property
    def total_units(self) -> int:
        return self.total_positive_images + self.total_negative_images

    def units_per_modality(self, n_admissible_focal: int) -> int:
        return self.images_per_concept * int(n_admissible_focal)

    def to_dict(self) -> dict:
        payload = {
            "schema": "jlens.mmpilot.l32_resolution_sample_plan.v1",
            "rule_version": SAMPLE_SIZE_RULE_VERSION,
            "images_per_concept": self.images_per_concept,
            "n_train_positive_images": self.n_train_positive_images,
            "n_test_positive_images": self.n_test_positive_images,
            "n_train_negative_images": self.n_train_negative_images,
            "n_test_negative_images": self.n_test_negative_images,
            "n_selected_concepts": self.n_selected_concepts,
            "n_focal_concepts": self.n_focal_concepts,
            "total_positive_images": self.total_positive_images,
            "total_negative_images": self.total_negative_images,
            "total_units_per_modality": self.total_units,
            "modalities": list(MODALITIES),
            "total_units_all_modalities": self.total_units * len(MODALITIES),
            "rule": self.rule.to_dict(),
            "rule_digest": self.rule.digest,
            "precision_by_admissible_focal_count": self.precision_by_admissible,
            "adequate_at_expected_admissible_count": self.adequate,
            "ladder_considered": list(self.ladder_considered),
            "stopping_rule": STOPPING_RULE,
        }
        payload["plan_digest"] = payload_checksum(payload)
        return payload


#: The exact stopping rule, fixed with the sample size and printed with it.
STOPPING_RULE = (
    "Fixed-n, no interim look. Every unit in the predeclared population is "
    "scored before any convergence rate is computed, and the study stops when "
    "the population is complete. There is no rule under which more units are "
    "added after a classification is seen: an AMBIGUOUS layer is a measured "
    "outcome, not an under-powered one, and enlarging the population until the "
    "band is crossed would be sampling to a foregone conclusion. If a stage "
    "cannot complete its population, the verdict is REFUSED_INVALID and no "
    "classification is reported at all."
)


def plan_sample_size(
    *,
    rule: SampleSizeRule = SAMPLE_SIZE_RULE,
    n_selected_concepts: int = len(SELECTED_CONCEPTS),
    n_focal_concepts: int = len(FOCAL_CONCEPTS),
    ladder: Sequence[int] = IMAGE_COUNT_LADDER,
) -> SamplePlan:
    """The smallest per-concept image count the frozen bars can be resolved at.

    The chosen count is the first rung of ``ladder`` whose cell — sized at
    ``rule.expected_admissible_focal`` focal concepts — meets both the
    half-width target and the two power targets. Precision at one, two and
    three admissible focal concepts is reported alongside, so a reader sees what
    the design buys in the case that actually occurs rather than only in the
    case it was sized for.
    """
    chosen = None
    for candidate in ladder:
        precision = cell_precision(
            candidate * rule.expected_admissible_focal, rule=rule
        )
        if precision["meets_half_width_target"] and precision["meets_power_target"]:
            chosen = candidate
            break
    adequate = chosen is not None
    if chosen is None:
        chosen = int(ladder[-1])

    per_admissible = {
        str(count): cell_precision(chosen * count, rule=rule)
        for count in range(1, n_focal_concepts + 1)
    }
    half = chosen // 2
    return SamplePlan(
        images_per_concept=int(chosen),
        n_train_positive_images=int(half),
        n_test_positive_images=int(chosen - half),
        n_train_negative_images=int(half),
        n_test_negative_images=int(chosen - half),
        n_selected_concepts=int(n_selected_concepts),
        n_focal_concepts=int(n_focal_concepts),
        rule=rule,
        precision_by_admissible=per_admissible,
        adequate=bool(adequate),
        ladder_considered=tuple(int(x) for x in ladder),
    )


def format_sample_plan(plan: SamplePlan) -> str:
    """The block printed before any model output is observed."""
    rule = plan.rule
    lines = [
        "=" * 72,
        "PREDECLARED SAMPLE SIZE — fixed before any model output is observed",
        "=" * 72,
        f"  rule version           {SAMPLE_SIZE_RULE_VERSION}",
        f"  rule digest            {rule.digest}",
        "",
        "  The frozen bars are 0.50 (NOT_CONVERGED, generous scoring) and 0.90",
        "  (CONVERGED, unique scoring). The gap is 0.40, so the design targets a",
        f"  95% Wilson half-width of at most {rule.target_half_width:.2f} at the",
        "  worst-case proportion p = 0.5, plus at least",
        f"  {rule.min_power:.0%} power to land below 0.50 when the truth is",
        f"  {rule.not_converged_truth:.2f} and at least {rule.min_power:.0%} power to land",
        f"  at or above 0.90 when the truth is {rule.converged_truth:.2f}.",
        "",
        f"  sized for {rule.expected_admissible_focal} admissible focal concept(s) — "
        "the completed study",
        "  already found `zebra` capability-ineligible in spoken audio, so",
        "  sizing for three would size for a population we have evidence",
        "  against.",
        "",
        f"  distinct images per concept      {plan.images_per_concept}",
        f"    train (source) split           {plan.n_train_positive_images}",
        f"    held-out (target) split        {plan.n_test_positive_images}",
        f"  distinct recordings per concept  {plan.images_per_concept} "
        "(one per photograph)",
        f"  selected concepts                {plan.n_selected_concepts}",
        f"  focal concepts                   {plan.n_focal_concepts}",
        f"  matched-negative images          {plan.total_negative_images}",
        "",
        f"  total units per modality         {plan.total_units}",
        f"  modalities                       {len(MODALITIES)} {list(MODALITIES)}",
        f"  total units, all modalities      {plan.total_units * len(MODALITIES)}",
        "",
        "  RESOLVING POWER, by how many focal concepts survive capability:",
        f"    {'admissible':>10}  {'cell n':>7}  {'95% half-width':>14}  "
        f"{'P(<=0.50|p=.40)':>16}  {'P(>=0.90|p=.95)':>16}",
    ]
    for count, precision in sorted(plan.precision_by_admissible.items()):
        lines.append(
            f"    {count:>10}  {precision['n']:>7}  "
            f"{precision['wilson_half_width_at_p_0.5']:>14.3f}  "
            f"{precision['power_to_observe_not_converged']:>16.3f}  "
            f"{precision['power_to_observe_converged']:>16.3f}"
        )
    lines += [
        "",
        f"  design adequate at the expected admissible count: {plan.adequate}",
        "",
        "  STOPPING RULE",
        f"  {STOPPING_RULE}",
        "=" * 72,
    ]
    return "\n".join(lines)


# ================================================================ stage plan


#: The gate, in one sentence, so it can be quoted rather than paraphrased.
STAGE_B_RULE = (
    "Stage B (causal replication on this same fresh population) runs only when "
    "Stage A classifies layer 32 L32_INDEPENDENT_NOT_CONVERGED with passing "
    "controls. Stage B exists for exactly one purpose: to support the combined "
    f"claim of causal transfer {CONVERGENCE_PHRASE}. That claim requires the "
    "convergence half to be NOT_CONVERGED, so under CONVERGED or AMBIGUOUS the "
    "causal passes cannot contribute to it no matter how they come out."
)

#: Why the gate is an efficiency rule and not a filter on unwelcome results.
STAGE_B_RULE_RATIONALE = (
    "The gate stops Stage B in exactly the cases where the headline result is "
    "*unfavourable* to the pre-convergence hypothesis — CONVERGED kills the "
    "claim outright, AMBIGUOUS leaves it unsupported — and those Stage-A "
    "outcomes are reported in full as the study's primary verdict. Nothing is "
    "suppressed by not spending passes on a measurement that could not change "
    "them. The opposite design (run Stage B regardless) was considered and "
    "rejected only on cost, so the notebook exposes "
    "RUN_STAGE_B_CAUSAL_REPLICATION as an explicit override; an overridden run "
    "is stamped `gate_overridden: true` in every artifact and its causal "
    "numbers are reported as descriptive, never as the predeclared path."
)


def stage_plan() -> dict:
    """The two stages and the conditional rule, fixed before Stage A opens."""
    return {
        "schema": "jlens.mmpilot.l32_resolution_stage_plan.v1",
        "stage_plan_version": STAGE_PLAN_VERSION,
        "mode": "conditional",
        "stages": [
            {
                "stage": "A",
                "name": "independent_convergence",
                "runs": "always",
                "contents": [
                    "behavioral capability under the open prompt on the fresh population",
                    "final-prompt-token residual capture at physical layer 32",
                    "native direct readout lm_head(final_norm(h)) with the declared softcap",
                    "the three convergence controls",
                    "classification under the frozen criterion",
                ],
                "produces": list(RESOLUTION_VERDICTS),
            },
            {
                "stage": "B",
                "name": "causal_replication",
                "runs": "conditionally",
                "condition": STAGE_B_RULE,
                "contents": [
                    "source-derived J-space steering at layer 32 on this same population",
                    "the same open prompt, matched-random, external-unrelated, "
                    "specificity, norm and image-independence controls",
                ],
                "forbidden": [
                    "the candidate-listed protocol the historical run used",
                    "an Anthropic-style two-coordinate swap (a separate study)",
                ],
            },
        ],
        "conditional_rule": STAGE_B_RULE,
        "rationale": STAGE_B_RULE_RATIONALE,
        "efficiency_gate_not_suppression": True,
        "all_stage_a_outcomes_reported": True,
        "intervention_family": INTERVENTION_FAMILY,
    }


def stage_b_decision(
    *,
    verdict: str,
    controls_passed: bool,
    requested: bool,
    budget_confirmed: bool,
) -> dict:
    """Whether Stage B runs, and under which of the two paths.

    ``requested`` without the gate being met is not an error — it is an
    override, and it is recorded as one. The distinction that matters is
    reported on the record, not enforced by refusing to measure.
    """
    gate_met = bool(verdict == L32_INDEPENDENT_NOT_CONVERGED and controls_passed)
    runs = bool(requested and budget_confirmed)
    return {
        "schema": "jlens.mmpilot.l32_resolution_stage_b_decision.v1",
        "stage_plan_version": STAGE_PLAN_VERSION,
        "stage_a_verdict": verdict,
        "controls_passed": bool(controls_passed),
        "gate_met": gate_met,
        "requested": bool(requested),
        "budget_confirmed": bool(budget_confirmed),
        "runs": runs,
        "gate_overridden": bool(runs and not gate_met),
        "rule": STAGE_B_RULE,
        "rationale": STAGE_B_RULE_RATIONALE,
        "statement": (
            "Stage B runs under the predeclared gate"
            if runs and gate_met
            else (
                "Stage B runs as an EXPLICIT OVERRIDE of the predeclared gate; "
                "its causal numbers are descriptive and do not enter the "
                "combined pre-convergence claim"
                if runs
                else (
                    "Stage B does not run. "
                    + (
                        "The gate is met but the run was not requested or its "
                        "budget was not confirmed."
                        if gate_met
                        else f"The gate is not met (Stage A returned {verdict})."
                    )
                )
            )
        ),
    }


# ============================================================== stage gates


#: The switches a human sets by hand in section 2. Nothing else is an input.
RESOLUTION_RAW_SWITCHES: tuple[str, ...] = (
    "RUN_REAL_L32_CONVERGENCE_RESOLUTION",
    "RUN_MODEL_STAGE",
    "CONFIRM_MODEL_LOAD",
    "CONFIRM_STAGE_A_BUDGET",
    "RUN_STAGE_B_CAUSAL_REPLICATION",
    "CONFIRM_STAGE_B_BUDGET",
)

#: The names :func:`derive_resolution_gates` produces, in dependency order.
RESOLUTION_DERIVED_GATES: tuple[str, ...] = (
    "MODEL_STAGE_ENABLED",
    "STAGE_A_ENABLED",
    "STAGE_B_REQUESTED",
)

RESOLUTION_GATE_RULE_VERSION = "mmpilot.l32_resolution_stage_gates.v1"


def derive_resolution_gates(switches: Mapping[str, object]) -> dict[str, bool]:
    """Derive the three gates from the six raw switches. Pure.

    Same rule as :mod:`jlens.mmpilot.stage_gates` and for the same reason: a
    Jupyter kernel keeps whatever was executed last, so a gate computed once and
    read many cells later goes stale the moment section 2 is re-run. Every cell
    that can spend model passes re-derives its gate immediately beforehand.

    Raises:
        MissingStageSwitch: If any raw switch is absent — which means section 2
            was never executed in this kernel. Defaulting it to ``False`` would
            print "not requested" for a switch the operator had just set.
    """
    from jlens.mmpilot.stage_gates import MissingStageSwitch

    missing = [name for name in RESOLUTION_RAW_SWITCHES if name not in switches]
    if missing:
        raise MissingStageSwitch(
            "the resolution stage gates cannot be derived because these raw "
            f"switches are not defined: {missing}. Execute section 2 in this "
            "kernel first."
        )
    values = {name: bool(switches[name]) for name in RESOLUTION_RAW_SWITCHES}
    model_stage = values["RUN_MODEL_STAGE"] and values["CONFIRM_MODEL_LOAD"]
    stage_a = model_stage and values["CONFIRM_STAGE_A_BUDGET"]
    stage_b = (
        stage_a
        and values["RUN_STAGE_B_CAUSAL_REPLICATION"]
        and values["CONFIRM_STAGE_B_BUDGET"]
    )
    return {
        "MODEL_STAGE_ENABLED": bool(model_stage),
        "STAGE_A_ENABLED": bool(stage_a),
        "STAGE_B_REQUESTED": bool(stage_b),
    }


def refresh_resolution_gates(namespace) -> dict[str, bool]:
    """Recompute the gates from ``namespace`` and write them back into it."""
    gates = derive_resolution_gates(namespace)
    namespace.update(gates)
    return gates


def format_resolution_gates(gates: Mapping, *, switches: Mapping) -> str:
    """The block a cell prints to show the gates it just re-derived."""
    lines = [
        "=" * 72,
        f"STAGE GATES (re-derived now — {RESOLUTION_GATE_RULE_VERSION})",
        "=" * 72,
    ]
    lines += [
        f"  {name:38s} {bool(switches[name])}" for name in RESOLUTION_RAW_SWITCHES
    ]
    lines.append("")
    lines += [
        f"  {name:38s} {bool(gates[name])}" for name in RESOLUTION_DERIVED_GATES
    ]
    return "\n".join(lines)


# ==================================================== clean-answer references


def clean_predictions_from_capability(
    records: Iterable[Mapping],
) -> dict[str, str]:
    """``sample_id -> the model's own clean answer``, from capability units.

    The completed follow-up recovered this from zero-alpha *intervention* units,
    because it ran the causal stage first. Stage A of this study deliberately
    does not run interventions, and the capability stage already measured
    exactly the same quantity on exactly the same input: the argmax over the six
    externally-scored candidates for the unedited prompt.

    Raises:
        ResolutionRefused: If two records disagree about one sample. A "clean"
            reference that is not a property of the sample would make every
            agreement rate meaningless.
    """
    out: dict[str, str] = {}
    for record in records:
        sample_id = str(record.get("sample_id", ""))
        prediction = record.get("prediction")
        if not sample_id or prediction is None:
            continue
        prediction = str(prediction)
        existing = out.get(sample_id)
        if existing is not None and existing != prediction:
            raise ResolutionRefused(
                f"two capability units disagree about sample {sample_id}'s clean "
                f"answer ({existing!r} vs {prediction!r}). The clean reference "
                "must be a property of the sample; refusing to average two "
                "readings of it."
            )
        out[sample_id] = prediction
    return out


# ======================================================= controls completeness


def assert_controls_recorded(
    controls: Mapping,
    *,
    layer: int = L32_LAYER,
    required: Sequence[str] = CONTROL_VARIANTS,
) -> dict:
    """A missing control record is a failure, never a pass.

    Completeness and the flat row table come from
    :mod:`jlens.mmpilot.l32_reporting`, which already knows that the variants
    live under ``per_layer[layer]["controls"][variant]`` and that
    ``summarize_controls`` silently omits a variant with no rows — so an absent
    control leaves ``all_controls_passed`` ``True`` with nothing having run.
    Reading that flag alone is the failure this wrapper exists to prevent, and
    it is a refusal here rather than a note.

    ``primary_is_informative`` is reported and deliberately **not** made a
    refusal. It is ``False`` when the primary readout sits at the
    ``1/n_candidates`` chance rate, which is precisely what a genuinely
    non-converged layer looks like: refusing on it would make
    ``NOT_CONVERGED`` unreachable by construction while leaving ``CONVERGED``
    reachable, which is a thumb on the scale in the direction this study most
    has to avoid. Degeneracy is caught instead by the criterion's own guards —
    distinct-prediction count and finiteness — which
    :func:`resolution_verdict` checks as validity clauses.

    Raises:
        ResolutionRefused: If a required variant is absent or reproduced the
            primary readout.
    """
    from jlens.mmpilot.l32_reporting import (
        ControlRecordsIncomplete,
        assert_controls_complete,
        control_rows,
    )

    try:
        completeness = assert_controls_complete(
            controls, layer=int(layer), variants=required
        )
        rows = control_rows(controls, layer=int(layer), variants=required)
    except ControlRecordsIncomplete as exc:
        raise ResolutionRefused(
            f"{exc}\nThe aggregate flag says all_controls_passed="
            f"{(controls or {}).get('all_controls_passed')!r}, which is exactly "
            "the reading this check exists to prevent."
        ) from exc

    failing = [row["variant"] for row in rows if not row["passed"]]
    uninformative = [row["variant"] for row in rows if not row["primary_is_informative"]]
    record = {
        "schema": "jlens.mmpilot.l32_resolution_controls.v1",
        "layer": int(layer),
        "required_variants": list(required),
        "rows": rows,
        "completeness": completeness,
        "missing_or_empty": [],
        "failing": failing,
        "uninformative": uninformative,
        "uninformative_note": (
            "a control whose primary sits at the chance rate had nothing to "
            "reproduce. Reported, not refused: at a genuinely non-converged "
            "layer that is the expected state, and refusing on it would make "
            "NOT_CONVERGED unreachable while leaving CONVERGED reachable"
        ),
        "all_present": True,
        "passed": not failing,
        "reported_all_controls_passed": (controls or {}).get("all_controls_passed"),
    }
    if failing:
        raise ResolutionRefused(
            f"convergence control(s) {failing} reproduced the primary readout at "
            f"layer {layer}. The readout is not measuring what it claims to; "
            "refusing to classify."
        )
    return record


# =================================================================== verdict


#: Clauses that must hold before *any* classification may be reported. A failure
#: here produces ``REFUSED_INVALID`` — not a classification with a caveat.
VALIDITY_CLAUSES: tuple[str, ...] = (
    "lens_integrity_passed",
    "native_head_agrees_with_model_unembed",
    "criterion_digest_unchanged",
    "population_disjoint_from_completed_run",
    "one_unit_per_photograph",
    "sample_size_predeclared",
    "controls_present_and_passing",
    "outputs_finite_and_nondegenerate",
    "at_least_one_admissible_focal_concept",
)


def resolution_verdict(
    *,
    integrity: Mapping,
    convergence: Mapping,
    controls: Mapping,
    disjointness: Mapping,
    pseudoreplication: Mapping,
    sample_plan: Mapping,
    head_agreement: Mapping,
    admissibility: Mapping,
) -> dict:
    """The one primary verdict, and the clause that decided it.

    Validity is established first and separately. A study whose controls did not
    run has not measured ``AMBIGUOUS``; it has measured nothing, and saying so is
    :data:`REFUSED_INVALID` rather than a hedged classification.
    """
    classification_block = convergence.get("classification") or {}
    classification = classification_block.get("classification")
    criterion_digest = convergence.get("criterion_digest")
    admissible = list(admissibility.get("eligible_concepts") or [])
    summary = convergence.get("summary") or {}
    # The layer comes from the measurement, never from the module constant: the
    # MOCK world's decoder has six blocks and stands in for layer 32 at layer 1,
    # so reading ``per_layer["32"]`` would find nothing there and report a
    # degenerate readout for a cell that is perfectly well populated.
    layer = int(convergence.get("layer", L32_LAYER))
    per_layer = (summary.get("per_layer") or {}).get(str(layer)) or {}
    per_modality = per_layer.get("per_modality") or {}

    # Non-finite logits cannot reach here: ``direct_readout_row`` refuses the
    # row outright, so every row that exists is finite by construction. What
    # *can* reach here is a readout that answered the same candidate every time,
    # and a modality cell that produced too few rows to be a rate at all.
    degenerate = [
        modality
        for modality in CONVERGENCE_CRITERION.required_modalities
        if int((per_modality.get(modality) or {}).get("n_distinct_predictions", 0))
        < CONVERGENCE_CRITERION.min_distinct_predictions
    ]
    undersized = [
        modality
        for modality in CONVERGENCE_CRITERION.required_modalities
        if int((per_modality.get(modality) or {}).get("n", 0))
        < CONVERGENCE_CRITERION.min_samples_per_cell
    ]

    clauses = [
        {
            "clause": "lens_integrity_passed",
            "passed": integrity.get("verdict") == "PASSED",
            "detail": str(integrity.get("verdict")),
        },
        {
            "clause": "native_head_agrees_with_model_unembed",
            "passed": bool(head_agreement.get("passed")),
            "detail": (
                f"matches={head_agreement.get('matches_model_unembed')!r} "
                f"comparison_ran={head_agreement.get('comparison_ran')!r}"
            ),
        },
        {
            "clause": "criterion_digest_unchanged",
            "passed": criterion_digest == FROZEN_CRITERION_DIGEST,
            "detail": f"{criterion_digest} vs frozen {FROZEN_CRITERION_DIGEST}",
        },
        {
            "clause": "population_disjoint_from_completed_run",
            "passed": bool(disjointness.get("disjoint")),
            "detail": str(disjointness.get("failed_families")),
        },
        {
            "clause": "one_unit_per_photograph",
            "passed": bool(pseudoreplication.get("passed")),
            "detail": (
                f"{pseudoreplication.get('n_units')} unit(s) on "
                f"{pseudoreplication.get('n_distinct_images')} photograph(s)"
            ),
        },
        {
            "clause": "sample_size_predeclared",
            "passed": bool(sample_plan.get("plan_digest")),
            "detail": str(sample_plan.get("plan_digest")),
        },
        {
            "clause": "controls_present_and_passing",
            "passed": bool(controls.get("passed")),
            "detail": (
                f"missing={controls.get('missing_or_empty')} "
                f"failing={controls.get('failing')}"
            ),
        },
        {
            "clause": "outputs_finite_and_nondegenerate",
            "passed": not degenerate and not undersized,
            "detail": (
                f"single-prediction modalities {degenerate}, undersized cells "
                f"{undersized}; finiteness is enforced upstream by "
                "direct_readout_row, which refuses a non-finite row"
            ),
        },
        {
            "clause": "at_least_one_admissible_focal_concept",
            "passed": bool(admissible),
            "detail": f"admissible {admissible}",
        },
    ]
    failed = [entry["clause"] for entry in clauses if not entry["passed"]]

    if failed:
        verdict = REFUSED_INVALID
        rationale = (
            "No classification is reported. "
            f"{len(failed)} validity clause(s) did not hold: {failed}. A study "
            "whose measurement is not established has not observed AMBIGUOUS; "
            "it has observed nothing, and reporting a classification anyway "
            "would put a number where there is no measurement."
        )
    else:
        verdict = _CLASSIFICATION_TO_VERDICT.get(classification, REFUSED_INVALID)
        if verdict == REFUSED_INVALID:
            rationale = (
                f"the frozen criterion returned {classification!r}, which is not "
                f"one of {sorted(_CLASSIFICATION_TO_VERDICT)}"
            )
        elif verdict == L32_INDEPENDENT_CONVERGED:
            rationale = (
                f"On an independently selected population, physical layer "
                f"{L32_LAYER} meets every CONVERGED clause in every required "
                "modality under the frozen criterion. Any causal effect measured "
                "at this layer therefore occurs at or after native "
                "direct-readout convergence, and the pre-convergence reading of "
                "the completed causal evidence is not available."
            )
        elif verdict == L32_INDEPENDENT_NOT_CONVERGED:
            rationale = (
                f"On an independently selected population, physical layer "
                f"{L32_LAYER} meets every NOT_CONVERGED clause in every required "
                "modality under the frozen criterion, scored generously. This is "
                "the convergence half of a pre-convergence claim and nothing "
                "more: it says the native readout has not converged there, not "
                "that anything causal happens there."
            )
        else:
            rationale = (
                f"On an independently selected population, physical layer "
                f"{L32_LAYER} again falls between the two frozen bars. This is a "
                "measured outcome on a fresh sample, not a shortage of data: an "
                "ambiguous layer can stay ambiguous at any n, and the "
                "predeclared stopping rule forbids enlarging the population "
                "until the band is crossed."
            )

    return {
        "schema": "jlens.mmpilot.l32_resolution_verdict.v1",
        "verdict_name": VERDICT_NAME,
        "verdict": verdict,
        "possible_verdicts": list(RESOLUTION_VERDICTS),
        "classification": classification,
        "criterion_digest": criterion_digest,
        "criterion_thresholds_unchanged": criterion_digest == FROZEN_CRITERION_DIGEST,
        "validity_clauses": clauses,
        "failed_validity_clauses": failed,
        "admissible_focal_concepts": admissible,
        "inadmissible_focal_concepts": sorted(
            set(FOCAL_CONCEPTS) - set(admissible)
        ),
        "concepts_replaced": False,
        "rationale": rationale,
        "layer": layer,
        "physical_layer_under_study": int(L32_LAYER),
        "scale": int(SELECTED_SCALE),
        "protocol": L32_RESOLUTION_PROTOCOL,
        "prompt_protocol": OPEN_PROMPT_PROTOCOL,
        "phrase": CONVERGENCE_PHRASE,
    }


# ================================================================= synthesis


def causal_synthesis(
    *,
    verdict: Mapping,
    completed_causal: Mapping,
    stage_b: Mapping | None = None,
) -> dict:
    """What the independent convergence result does, and does not, license.

    The trap this exists to close: an independent ``NOT_CONVERGED`` plus the
    completed run's causal evidence looks like a combined pre-convergence claim,
    and it is not one. The completed causal measurement was made on a *different*
    population; pairing its effect with this population's convergence would
    attribute a causal result to photographs it was never measured on.
    """
    primary = verdict.get("verdict")
    completed_verdict = str(completed_causal.get("verdict", "UNKNOWN"))
    stage_b_ran = bool((stage_b or {}).get("runs"))
    stage_b_verdict = (stage_b or {}).get("causal_verdict")

    replication_required = [
        "source-derived J-space causal steering at physical layer 32 on THIS "
        "fresh population, under the open protocol "
        f"{OPEN_PROMPT_PROTOCOL!r}",
        "off-diagonal (source modality != target modality) cells only",
        "the same matched-random, external-unrelated, specificity, activation-"
        "norm and image-independence controls",
        "an effect on at least 2 distinct photographs/recordings, with the "
        "expected sign, above both controls",
        "capability admissibility measured under this same open prompt, with "
        "ineligible concepts excluded and never replaced",
    ]

    if primary == L32_INDEPENDENT_NOT_CONVERGED and stage_b_ran:
        combined = (
            "AVAILABLE_PENDING_STAGE_B_VERDICT"
            if stage_b_verdict is None
            else (
                "SUPPORTED"
                if str(stage_b_verdict) == "SUPPORTED"
                else "NOT_SUPPORTED"
            )
        )
        statement = (
            "Stage B measured causal transfer on the same fresh population that "
            f"was classified {primary}, so the convergence half and the causal "
            "half are for once about the same photographs. The combined claim "
            f"is {combined}."
        )
    elif primary == L32_INDEPENDENT_NOT_CONVERGED:
        combined = "NOT_ESTABLISHED_CAUSAL_REPLICATION_MISSING"
        statement = (
            f"Layer {L32_LAYER} is {NOT_CONVERGED} on this independent "
            "population. This does NOT establish causal transfer "
            f"{CONVERGENCE_PHRASE}. The completed causal evidence "
            f"({completed_verdict}) was measured on a different population, and "
            "pairing it with this one's convergence would credit a causal effect "
            "to photographs it was never measured on. What is still required is "
            "listed below, and it is a measurement, not an argument."
        )
    elif primary == L32_INDEPENDENT_CONVERGED:
        combined = "RULED_OUT_AT_LAYER_32"
        statement = (
            f"Layer {L32_LAYER} is {CONVERGED} on this independent population, "
            "so any causal effect there occurs at or after native direct-readout "
            "convergence. The pre-convergence claim cannot be made at this layer "
            "at all, and the question moves to layers shallower than "
            f"{L32_LAYER}, which requires a fresh calibration at those depths."
        )
    elif primary == L32_INDEPENDENT_AMBIGUOUS:
        combined = "NOT_ESTABLISHED_CONVERGENCE_HALF_AMBIGUOUS"
        statement = (
            f"Layer {L32_LAYER} is {AMBIGUOUS} on a second, independent "
            "population. Two independent samples now place it between the bars, "
            "which is evidence that the layer genuinely sits there rather than "
            "that either sample was unlucky. No pre-convergence claim follows "
            "from an ambiguous layer under the frozen criterion, and the "
            "criterion is not revisable now that the result is visible."
        )
    else:
        combined = "NOT_ESTABLISHED_STUDY_INVALID"
        statement = (
            "The independent convergence study returned "
            f"{REFUSED_INVALID}, so it contributes nothing to the combined "
            "claim in either direction."
        )

    return {
        "schema": "jlens.mmpilot.l32_resolution_synthesis.v1",
        "protocol": L32_RESOLUTION_PROTOCOL,
        "independent_convergence_verdict": primary,
        "completed_causal_verdict": completed_verdict,
        "completed_causal_population": completed_causal.get(
            "run_dir", COMPLETED_FOLLOWUP_RUN
        ),
        "completed_causal_fingerprint": completed_causal.get(
            "fingerprint", COMPLETED_FOLLOWUP_FINGERPRINT
        ),
        "populations_are_the_same": bool(stage_b_ran),
        "combined_pre_convergence_claim": combined,
        "statement": statement,
        # Cleared only when a causal replication on THIS population could
        # actually complete the combined claim. A Stage B run under an override
        # at a CONVERGED or AMBIGUOUS layer measured something real and still
        # leaves the claim unreachable, so the requirement stands.
        "additional_causal_replication_required": (
            []
            if (stage_b_ran and primary == L32_INDEPENDENT_NOT_CONVERGED)
            else replication_required
        ),
        "reference_layer": int(REFERENCE_LAYER),
        "phrase": CONVERGENCE_PHRASE,
        "never_claimed": [
            "pre-convergence causal transfer inferred from a NOT_CONVERGED "
            "population alone",
            "any statement about environmental (non-speech) audio",
            "an Anthropic-style two-coordinate swap, which is a separate study",
        ],
    }


# =============================================================== fingerprint


#: Every scientific configuration field. A change in any one refuses the resume.
RESOLUTION_FINGERPRINT_FIELDS: tuple[str, ...] = (
    "protocol",
    "stage_plan_version",
    "conditional_stage_b_rule_digest",
    "intervention_family",
    "model_repo_id",
    "model_revision",
    "processor_revision",
    "transformers_version",
    "torch_version",
    "audio_protocol_version",
    "audio_protocol_fingerprint",
    "lens_path",
    "lens_checksum",
    "lens_confirmation_status",
    "lens_confirmation_set_checksum",
    "publication_metadata_checksum",
    "physical_layer",
    "hook_site",
    "d_model",
    "residual_convention",
    "final_prompt_token_position",
    "dictionary_orientation",
    "dictionary_normalization",
    "calibration_modality",
    "selected_scale",
    "original_manifest_checksum",
    "expanded_manifest_checksum",
    "cache_schema_version",
    "evidence_lexicon_hash",
    "exclusion_run_dirs",
    "exclusion_run_checksum",
    "prompt_protocol",
    "prompt_hash",
    "prompt_protocol_digest",
    "selected_concepts",
    "focal_concepts",
    "capability_admissible_concepts",
    "capability_protocol",
    "admissibility_rule_version",
    "selection_algorithm_version",
    "selection_seed",
    "selection_profile_version",
    "selected_population_digest",
    "sample_size_rule_version",
    "sample_size_plan_digest",
    "convergence_criterion_version",
    "convergence_criterion_digest",
    "control_variants",
    "control_seed",
    "media_io_version",
    "jlens_version",
)


def resolution_fingerprint(**fields) -> dict:
    """Bind every scientific configuration field, and refuse a missing one.

    Raises:
        ResolutionRefused: If any field in
            :data:`RESOLUTION_FINGERPRINT_FIELDS` is missing, or an unknown
            field is supplied. A ``None`` default would let an audio run resume
            from a text run's directory the first time someone forgot a field.
    """
    missing = [name for name in RESOLUTION_FINGERPRINT_FIELDS if name not in fields]
    unknown = sorted(set(fields) - set(RESOLUTION_FINGERPRINT_FIELDS))
    if missing or unknown:
        raise ResolutionRefused(
            "the resolution fingerprint is incomplete or over-specified: "
            f"missing {missing}, unknown {unknown}"
        )
    payload = {name: fields[name] for name in RESOLUTION_FINGERPRINT_FIELDS}
    return {**payload, "fingerprint_digest": payload_checksum(payload)}


def assert_fresh_run_namespace(
    run_dir: str | os.PathLike[str], *, protected_prefixes: Sequence[str]
) -> str:
    """Refuse a run directory inside any completed run's namespace.

    Raises:
        ResolutionRefused: When the path passes through a protected prefix, or
            when its own name does not start with
            :data:`RESOLUTION_RUN_PREFIX`.
    """
    root = Path(run_dir)
    offending = sorted(
        {
            prefix
            for prefix in protected_prefixes
            for part in root.parts
            if part.startswith(prefix)
        }
    )
    if offending:
        raise ResolutionRefused(
            f"{root} is inside a completed run namespace ({offending}). "
            "Completed runs are evidence and are never written into; this study "
            f"creates its own {RESOLUTION_RUN_PREFIX}_* directory."
        )
    if not root.name.startswith(RESOLUTION_RUN_PREFIX):
        raise ResolutionRefused(
            f"{root.name!r} does not start with {RESOLUTION_RUN_PREFIX!r}. The "
            "run family is part of the study's identity: a directory named like "
            "another family's could be resumed into by that family's notebook."
        )
    return str(root)


# ======================================================= read-only guarantee


def run_tree_digest(
    run_dir: str | os.PathLike[str], *, max_files: int = 20000
) -> dict:
    """A digest over every file in a completed run — names, sizes and mtimes.

    Full content checksums over a multi-gigabyte run directory would cost more
    than the study; names, sizes and modification times catch every write this
    study could plausibly make (a new artifact, a rewritten summary, a truncated
    unit) and cost one directory walk.
    """
    root = Path(run_dir)
    entries: list[tuple[str, int, int]] = []
    truncated = False
    if root.is_dir():
        for index, path in enumerate(sorted(root.rglob("*"))):
            if index >= max_files:
                truncated = True
                break
            if not path.is_file():
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            entries.append(
                (
                    str(path.relative_to(root)).replace("\\", "/"),
                    int(stat.st_size),
                    int(stat.st_mtime),
                )
            )
    digest = hashlib.sha256()
    for name, size, mtime in entries:
        digest.update(f"{name}|{size}|{mtime}\n".encode())
    return {
        "run_dir": str(root),
        "exists": root.is_dir(),
        "n_files": len(entries),
        "truncated": truncated,
        "tree_digest": "sha256:" + digest.hexdigest(),
    }


def assert_completed_runs_unchanged(
    before: Sequence[Mapping], after: Sequence[Mapping]
) -> dict:
    """Prove no completed run moved while this study ran.

    Raises:
        ResolutionRefused: On the first difference, including a run that
            appeared or vanished.
    """
    left = {str(entry["run_dir"]): entry for entry in before}
    right = {str(entry["run_dir"]): entry for entry in after}
    changed = [
        {
            "run_dir": name,
            "before": left.get(name, {}).get("tree_digest"),
            "after": right.get(name, {}).get("tree_digest"),
            "n_files_before": left.get(name, {}).get("n_files"),
            "n_files_after": right.get(name, {}).get("n_files"),
        }
        for name in sorted(set(left) | set(right))
        if left.get(name, {}).get("tree_digest")
        != right.get(name, {}).get("tree_digest")
    ]
    if changed:
        raise ResolutionRefused(
            "a completed run changed while this study ran, which must never "
            f"happen: {json.dumps(changed, indent=2)}"
        )
    return {
        "schema": "jlens.mmpilot.l32_resolution_immutability.v1",
        "runs_checked": sorted(set(left) | set(right)),
        "unchanged": True,
        "digests": {name: left[name]["tree_digest"] for name in sorted(left)},
        "method": "name+size+mtime tree digest, taken before and after",
    }


# ==================================================================== report


def build_summary(
    *,
    fingerprint: Mapping,
    verdict: Mapping,
    synthesis: Mapping,
    sample_plan: Mapping,
    disjointness: Mapping,
    pseudoreplication: Mapping,
    pool: Mapping,
    exclusion: Mapping,
    convergence: Mapping,
    controls: Mapping,
    capability: Mapping,
    stage_plan_record: Mapping,
    stage_b: Mapping,
    immutability: Mapping,
    cache: Mapping,
    resume: Mapping,
    mode: str,
) -> dict:
    """``l32_convergence_resolution_summary.json``, assembled in one place."""
    return {
        "schema": "jlens.mmpilot.l32_convergence_resolution_summary.v1",
        "protocol": L32_RESOLUTION_PROTOCOL,
        "mode": str(mode),
        "layer": int(L32_LAYER),
        "selected_scale": int(SELECTED_SCALE),
        "prompt_protocol": OPEN_PROMPT_PROTOCOL,
        "convergence_protocol": CONVERGENCE_PROTOCOL,
        "selected_concepts": list(SELECTED_CONCEPTS),
        "focal_concepts": list(FOCAL_CONCEPTS),
        "concepts_frozen_in_ranking_order": True,
        "concepts_replaced_after_results": False,
        "primary_verdict": verdict.get("verdict"),
        "verdict": dict(verdict),
        "synthesis": dict(synthesis),
        "stage_plan": dict(stage_plan_record),
        "stage_b_decision": dict(stage_b),
        "sample_size_plan": dict(sample_plan),
        "population": {
            "exclusion_set": dict(exclusion),
            "independent_pool": dict(pool),
            "disjointness_audit": dict(disjointness),
            "pseudoreplication_audit": dict(pseudoreplication),
        },
        "convergence": dict(convergence),
        "controls": dict(controls),
        "capability": dict(capability),
        "cached_data_loading": dict(cache),
        "completed_run_immutability": dict(immutability),
        "resume": dict(resume),
        "fingerprint": dict(fingerprint),
        "interpretation_boundary": (
            "A weak native direct readout is a fact about the readout at this "
            "layer. It is never evidence about what a nonlinear decoder or a "
            f"trained probe could recover. Say {CONVERGENCE_PHRASE!r}."
        ),
    }


def render_report(summary: Mapping) -> str:
    """``l32_convergence_resolution_report.md``."""
    verdict = summary.get("verdict") or {}
    synthesis = summary.get("synthesis") or {}
    plan = summary.get("sample_size_plan") or {}
    population = summary.get("population") or {}
    disjointness = population.get("disjointness_audit") or {}
    stage_b = summary.get("stage_b_decision") or {}

    lines = [
        f"# Independent layer-{L32_LAYER} convergence resolution",
        "",
        f"- protocol: `{L32_RESOLUTION_PROTOCOL}`",
        f"- mode: **{summary.get('mode')}**",
        f"- prompt protocol: `{summary.get('prompt_protocol')}`",
        f"- criterion digest: `{verdict.get('criterion_digest')}` "
        f"(unchanged: {verdict.get('criterion_thresholds_unchanged')})",
        f"- fingerprint: `{(summary.get('fingerprint') or {}).get('fingerprint_digest')}`",
        "",
        "## Primary verdict",
        "",
        f"**{verdict.get('verdict')}**",
        "",
        verdict.get("rationale", ""),
        "",
        "### Validity clauses",
        "",
        "| clause | passed | detail |",
        "| --- | --- | --- |",
    ]
    for clause in verdict.get("validity_clauses") or []:
        lines.append(
            f"| `{clause['clause']}` | {clause['passed']} | {clause['detail']} |"
        )

    lines += [
        "",
        "## Independence of the population",
        "",
        f"- families verified: {disjointness.get('families_checked')}",
        f"- disjoint: **{disjointness.get('disjoint')}**",
        f"- overlaps: {disjointness.get('n_overlaps')}",
        f"- exclusion digest: `{disjointness.get('exclusion_digest')}`",
        f"- population digest: `{disjointness.get('population_digest')}`",
        "",
        "## Sample size, predeclared",
        "",
        f"- rule: `{plan.get('rule_version')}` digest `{plan.get('rule_digest')}`",
        f"- distinct images per concept: {plan.get('images_per_concept')}",
        f"- distinct recordings per concept: {plan.get('images_per_concept')}",
        f"- total units per modality: {plan.get('total_units_per_modality')}",
        f"- adequate at the expected admissible count: "
        f"{plan.get('adequate_at_expected_admissible_count')}",
        "",
        f"> {plan.get('stopping_rule')}",
        "",
        "## Concepts",
        "",
        f"- selected (frozen ranking order): {summary.get('selected_concepts')}",
        f"- focal: {summary.get('focal_concepts')}",
        f"- capability-admissible: {verdict.get('admissible_focal_concepts')}",
        f"- capability-ineligible (excluded, **never replaced**): "
        f"{verdict.get('inadmissible_focal_concepts')}",
        "",
        "## Stage B",
        "",
        f"- runs: **{stage_b.get('runs')}**",
        f"- gate met: {stage_b.get('gate_met')}",
        f"- gate overridden: {stage_b.get('gate_overridden')}",
        "",
        f"{stage_b.get('statement', '')}",
        "",
        f"> {stage_b.get('rule', '')}",
        "",
        f"> {stage_b.get('rationale', '')}",
        "",
        "## Synthesis with the completed causal evidence",
        "",
        f"- completed causal verdict (read-only): "
        f"`{synthesis.get('completed_causal_verdict')}`",
        f"- same population as this convergence result: "
        f"{synthesis.get('populations_are_the_same')}",
        f"- combined pre-convergence claim: "
        f"**{synthesis.get('combined_pre_convergence_claim')}**",
        "",
        synthesis.get("statement", ""),
        "",
    ]
    required = synthesis.get("additional_causal_replication_required") or []
    if required:
        lines += ["### Causal replication still required", ""]
        lines += [f"- {item}" for item in required]
        lines.append("")
    lines += [
        "## Interpretation boundary",
        "",
        str(summary.get("interpretation_boundary", "")),
        "",
    ]
    return "\n".join(lines)


__all__ = [
    "COMPLETED_FOLLOWUP_FINGERPRINT",
    "COMPLETED_FOLLOWUP_RUN",
    "DISJOINTNESS_AUDIT_VERSION",
    "EXCLUSION_HARVEST_VERSION",
    "FOCAL_CONCEPTS",
    "FROZEN_CRITERION_DIGEST",
    "IDENTITY_FAMILIES",
    "IMAGE_COUNT_LADDER",
    "L32_INDEPENDENT_AMBIGUOUS",
    "L32_INDEPENDENT_CONVERGED",
    "L32_INDEPENDENT_NOT_CONVERGED",
    "L32_RESOLUTION_PROTOCOL",
    "POPULATION_SELECTION_VERSION",
    "REFUSED_INVALID",
    "RESOLUTION_DERIVED_GATES",
    "RESOLUTION_FINGERPRINT_FIELDS",
    "RESOLUTION_GATE_RULE_VERSION",
    "RESOLUTION_RAW_SWITCHES",
    "RESOLUTION_RUN_PREFIX",
    "RESOLUTION_VERDICTS",
    "SAMPLE_SIZE_RULE",
    "SAMPLE_SIZE_RULE_VERSION",
    "SELECTED_CONCEPTS",
    "STAGE_B_RULE",
    "STAGE_B_RULE_RATIONALE",
    "STAGE_PLAN_VERSION",
    "STOPPING_RULE",
    "VALIDITY_CLAUSES",
    "VERDICT_NAME",
    "ExclusionEvidenceMissing",
    "ExclusionSet",
    "PopulationNotIndependent",
    "PseudoreplicationError",
    "ResolutionRefused",
    "SamplePlan",
    "SampleSizeRule",
    "assert_completed_runs_unchanged",
    "assert_controls_recorded",
    "assert_fresh_run_namespace",
    "assert_one_unit_per_photograph",
    "audit_population_disjointness",
    "binomial_at_least",
    "binomial_at_most",
    "build_summary",
    "causal_synthesis",
    "cell_precision",
    "clean_predictions_from_capability",
    "derive_resolution_gates",
    "format_resolution_gates",
    "format_sample_plan",
    "harvest_excluded_identities",
    "independent_pool",
    "plan_sample_size",
    "refresh_resolution_gates",
    "render_report",
    "resolution_fingerprint",
    "resolution_verdict",
    "resolve_excluded_media",
    "run_tree_digest",
    "selection_digest",
    "stage_b_decision",
    "stage_plan",
    "wilson_interval",
]
