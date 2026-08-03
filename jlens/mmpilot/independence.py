# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Image-level independence: identity, dependence audit, and re-aggregation.

SpokenCOCO gives a COCO photograph roughly five written captions and a spoken
reading of each. The pilot's subset builder keeps ``max_groups_per_image`` of
them — two, by default — so **one image can enter the run as two synchronized
groups**. Those two groups are not two observations of anything. They are one
photograph seen twice, with the caption text differing.

That has two separate consequences, and this module addresses them separately.

*Representation.* A cross-modal query must not be able to retrieve its own
photograph. Excluding the query's exact group does not achieve that when a
sibling group carries the same image, so the admissibility rule lives in
:func:`jlens.mmpilot.jspace.admissible_targets` and excludes on ``image_id``.

*Causation.* Intervention units are stored per synchronized group. Averaging
them flat counts one image twice and reports an ``n`` the design never earned —
pseudoreplication. :func:`summarize_interventions_by_image` averages within
image first and treats the image as the independent unit, keeping the
group-level numbers alongside for provenance rather than discarding them.

Nothing here loads a model, reads media, or recomputes an activation. It reads
saved artifacts and rearranges them, which is why the whole audit runs on a
free CPU runtime.

Identity is resolved, never assumed. If two groups claim one image id but
different image bytes, or one image's bytes appear under two ids, the module
refuses rather than picking a winner: an audit that guesses at identity cannot
establish independence, which is the only thing it exists to do.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from jlens.mmpilot.causal import CONTROL_KINDS
from jlens.mmpilot.expansion import canonical_coco_image_key
from jlens.mmpilot.jspace import (
    EXCLUSION_RULE_VERSION,
    code_map,
    representational_report,
)

#: How an image's identity is decided. Bound into the audit fingerprint: a
#: result computed under one rule must never be merged with another's.
IMAGE_IDENTITY_RULE_VERSION = "coco_image_id_normalized_media_crosschecked.v1"

#: How repeated observations of one image become a single independent unit.
CAUSAL_AGGREGATION_VERSION = "mean_of_within_image_means.v1"

#: Field aliases actually seen in this repository's artifacts. Probed in order;
#: the alias that resolved is recorded, so nothing here assumes a schema.
IMAGE_ID_FIELDS = ("image_id", "imageid", "image_key", "cocoid", "coco_id", "img_id")

#: Where a normalized image path may be found, for cross-checking the id.
IMAGE_REF_FIELDS = ("image_relpath", "image_path", "image_file", "file_name")


class ImageIdentityError(RuntimeError):
    """Image identity could not be resolved reliably, so the audit stops.

    Raised for a missing id, for one group claiming two images, for one image
    id spanning two distinct media checksums, and for one media checksum
    spanning two image ids. Every message names the offending records.
    """


@dataclass(frozen=True)
class GroupIdentity:
    """What one synchronized group is, assembled from whatever recorded it."""

    group_id: str
    image_id: str
    raw_image_id: str
    image_ref: str | None = None
    media_checksum: str | None = None
    concept: str | None = None
    split: str | None = None
    modalities: tuple[str, ...] = ()
    id_field: str | None = None
    sources: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "group_id": self.group_id,
            "image_id": self.image_id,
            "raw_image_id": self.raw_image_id,
            "image_ref": self.image_ref,
            "media_checksum": self.media_checksum,
            "concept": self.concept,
            "split": self.split,
            "modalities": list(self.modalities),
            "id_field": self.id_field,
            "sources": list(self.sources),
        }


@dataclass(frozen=True)
class ImageIdentityMap:
    """Resolved group/image identity plus the checks that justified it."""

    rule_version: str
    groups: Mapping[str, GroupIdentity]
    sample_to_group: Mapping[str, str]
    cross_checks: Mapping = field(default_factory=dict)

    def image_for_group(self, group_id: str) -> str:
        try:
            return self.groups[group_id].image_id
        except KeyError:  # pragma: no cover - guarded by resolve_image_identity
            raise ImageIdentityError(
                f"group {group_id!r} has no resolved image identity"
            ) from None

    def image_for_sample(self, sample_id: str) -> str:
        group_id = self.sample_to_group.get(sample_id)
        if group_id is None:
            # ``sample_id`` is ``f"{group_id}:{modality}"`` by construction, but
            # the map is preferred: a derived id is a guess, a recorded one is
            # evidence.
            group_id = str(sample_id).rsplit(":", 1)[0]
        return self.image_for_group(group_id)

    def groups_of_image(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for group_id, identity in sorted(self.groups.items()):
            out.setdefault(identity.image_id, []).append(group_id)
        return out

    @property
    def n_distinct_images(self) -> int:
        return len({identity.image_id for identity in self.groups.values()})

    def to_dict(self) -> dict:
        return {
            "rule_version": self.rule_version,
            "n_groups": len(self.groups),
            "n_distinct_images": self.n_distinct_images,
            "groups": {k: v.to_dict() for k, v in sorted(self.groups.items())},
            "groups_of_image": self.groups_of_image(),
            "cross_checks": dict(self.cross_checks),
        }


# ------------------------------------------------------------------ identity


def _first_present(record: Mapping, fields: Sequence[str]) -> tuple[str | None, str]:
    """``(value, field_name)`` for the first non-empty alias present."""
    for name in fields:
        value = record.get(name)
        if value not in (None, ""):
            return str(value), name
    return None, ""


def resolve_image_identity(
    records: Iterable[Mapping],
    *,
    subset: Mapping | None = None,
) -> ImageIdentityMap:
    """Map every group to one canonical image, or refuse to continue.

    Args:
        records: Saved unit payloads and/or subset rows, in any mixture. Each
            contributes whatever it happens to carry; nothing is required to
            carry everything.
        subset: An optional pilot subset. Its rows are folded in as another
            record source and used to cross-check split and concept.

    Raises:
        ImageIdentityError: If any group lacks a resolvable image id, if a
            group resolves to two different images, if one image id spans two
            distinct image media checksums, or if one media checksum appears
            under two image ids. An audit of image-level independence cannot
            proceed on a guessed identity.
    """
    rows = list(records)
    if subset is not None:
        for split, split_rows in (subset.get("splits") or {}).items():
            for row in split_rows:
                rows.append({**dict(row), "split": row.get("split") or split})

    raw_ids: dict[str, dict[str, list[str]]] = {}
    canonical: dict[str, dict[str, list[str]]] = {}
    fields_used: dict[str, str] = {}
    refs: dict[str, str] = {}
    checksums: dict[str, dict[str, list[str]]] = {}
    concepts: dict[str, set] = {}
    splits: dict[str, set] = {}
    modalities: dict[str, set] = {}
    sources: dict[str, set] = {}
    sample_to_group: dict[str, set] = {}
    missing: list[str] = []

    for row in rows:
        group_id = row.get("group_id")
        sample_id = row.get("sample_id")
        if not group_id and sample_id:
            group_id = str(sample_id).rsplit(":", 1)[0]
        if not group_id:
            continue
        group_id = str(group_id)
        if sample_id:
            sample_to_group.setdefault(str(sample_id), set()).add(group_id)

        raw_id, id_field = _first_present(row, IMAGE_ID_FIELDS)
        image_ref, _ = _first_present(row, IMAGE_REF_FIELDS)
        if image_ref:
            refs.setdefault(group_id, image_ref)
        if raw_id is None:
            missing.append(group_id)
            continue
        fields_used.setdefault(group_id, id_field)
        key = canonical_coco_image_key(
            raw_id, image_ref=image_ref or "", split=row.get("source_split") or ""
        )
        if not key:
            missing.append(group_id)
            continue
        raw_ids.setdefault(group_id, {}).setdefault(raw_id, []).append(id_field)
        canonical.setdefault(group_id, {}).setdefault(key, []).append(id_field)

        modality = row.get("modality")
        if modality:
            modalities.setdefault(group_id, set()).add(str(modality))
        # Only an image-modality record's media checksum says anything about
        # the photograph. A text record's checksum is about its prompt.
        checksum = row.get("media_checksum")
        if checksum and modality == "image":
            checksums.setdefault(key, {}).setdefault(str(checksum), []).append(group_id)
        if row.get("concept") is not None:
            concepts.setdefault(group_id, set()).add(str(row["concept"]))
        if row.get("split"):
            splits.setdefault(group_id, set()).add(str(row["split"]))
        if row.get("layer") is not None:
            sources.setdefault(group_id, set()).add("activation_or_jspace_unit")
        elif "prediction" in row:
            sources.setdefault(group_id, set()).add("capability_unit")
        else:
            sources.setdefault(group_id, set()).add("subset_row")

    known = sorted(set(raw_ids) | set(missing))
    unresolved = sorted(group for group in missing if group not in canonical)
    if unresolved:
        raise ImageIdentityError(
            f"{len(unresolved)} of {len(known)} synchronized group(s) carry no "
            f"resolvable image identity under any of {list(IMAGE_ID_FIELDS)}: "
            f"{unresolved[:8]}. The audit refuses to continue: without image "
            "identity there is no way to tell an independent observation from a "
            "repeat of the same photograph."
        )

    ambiguous = {
        group: sorted(keys) for group, keys in canonical.items() if len(keys) > 1
    }
    if ambiguous:
        raise ImageIdentityError(
            "image identity is ambiguous — these synchronized group(s) resolve "
            f"to more than one image: {dict(sorted(ambiguous.items())[:5])}. "
            "Refusing to pick one."
        )

    split_ids = {
        image_id: sorted(entries)
        for image_id, entries in checksums.items()
        if len(entries) > 1
    }
    if split_ids:
        raise ImageIdentityError(
            "image identity is ambiguous — these image id(s) span more than one "
            f"distinct image media checksum: {dict(sorted(split_ids.items())[:5])}. "
            "One id naming two different photographs would make the exclusion "
            "rule silently unsound."
        )
    by_checksum: dict[str, set] = {}
    for image_id, entries in checksums.items():
        for checksum in entries:
            by_checksum.setdefault(checksum, set()).add(image_id)
    aliased = {
        checksum: sorted(ids) for checksum, ids in by_checksum.items() if len(ids) > 1
    }
    if aliased:
        raise ImageIdentityError(
            "image identity is ambiguous — the same image bytes appear under "
            f"more than one image id: {dict(sorted(aliased.items())[:5])}. Two "
            "ids for one photograph would let a query retrieve its own image."
        )

    conflicting_samples = {
        sample: sorted(groups)
        for sample, groups in sample_to_group.items()
        if len(groups) > 1
    }
    if conflicting_samples:
        raise ImageIdentityError(
            "sample identity is ambiguous — these sample id(s) belong to more "
            f"than one group: {dict(sorted(conflicting_samples.items())[:5])}."
        )

    groups: dict[str, GroupIdentity] = {}
    for group_id, keys in sorted(canonical.items()):
        image_id = next(iter(keys))
        concept_values = sorted(concepts.get(group_id, set()))
        split_values = sorted(splits.get(group_id, set()))
        groups[group_id] = GroupIdentity(
            group_id=group_id,
            image_id=image_id,
            raw_image_id=sorted(raw_ids[group_id])[0],
            image_ref=refs.get(group_id),
            media_checksum=next(
                (
                    checksum
                    for checksum, owners in checksums.get(image_id, {}).items()
                    if group_id in owners
                ),
                None,
            ),
            concept=concept_values[0] if len(concept_values) == 1 else None,
            split=split_values[0] if len(split_values) == 1 else None,
            modalities=tuple(sorted(modalities.get(group_id, set()))),
            id_field=fields_used.get(group_id),
            sources=tuple(sorted(sources.get(group_id, set()))),
        )

    inconsistent_split = sorted(
        group for group, values in splits.items() if len(values) > 1
    )
    if inconsistent_split:
        raise ImageIdentityError(
            "these synchronized group(s) are recorded in more than one split: "
            f"{inconsistent_split[:8]}. A group in two splits makes every "
            "held-out claim in the run unverifiable."
        )

    return ImageIdentityMap(
        rule_version=IMAGE_IDENTITY_RULE_VERSION,
        groups=groups,
        sample_to_group={
            sample: next(iter(values)) for sample, values in sample_to_group.items()
        },
        cross_checks={
            "n_records_read": len(rows),
            "id_fields_used": sorted(set(fields_used.values())),
            "n_groups_with_image_media_checksum": sum(
                1 for identity in groups.values() if identity.media_checksum
            ),
            "media_checksum_crosscheck": (
                "verified" if checksums else "unavailable: no image-modality "
                "media checksum was saved, so the id was accepted on the "
                "normalized path alone"
            ),
            "subset_supplied": subset is not None,
        },
    )


# --------------------------------------------------------------- subset audit


def _histogram(values: Iterable[int]) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        out[str(value)] = out.get(str(value), 0) + 1
    return dict(sorted(out.items(), key=lambda item: int(item[0])))


def _split_block(
    identities: Sequence[GroupIdentity], modality_records: Mapping[str, int]
) -> dict:
    by_image: dict[str, list[GroupIdentity]] = {}
    for identity in identities:
        by_image.setdefault(identity.image_id, []).append(identity)
    repeated = {
        image_id: members for image_id, members in by_image.items() if len(members) > 1
    }
    return {
        "n_groups": len(identities),
        "n_distinct_images": len(by_image),
        "groups_per_image": {
            image_id: len(members) for image_id, members in sorted(by_image.items())
        },
        "groups_per_image_histogram": _histogram(
            len(members) for members in by_image.values()
        ),
        "n_images_with_multiple_groups": len(repeated),
        "images_with_multiple_groups": [
            {
                "image_id": image_id,
                "n_groups": len(members),
                "group_ids": [m.group_id for m in members],
                "concepts": sorted({m.concept for m in members if m.concept}),
                "n_modality_records": sum(
                    modality_records.get(m.group_id, 0) for m in members
                ),
            }
            for image_id, members in sorted(repeated.items())
        ],
        "n_modality_records": sum(
            modality_records.get(identity.group_id, 0) for identity in identities
        ),
        "n_modality_records_on_repeated_images": sum(
            modality_records.get(m.group_id, 0)
            for members in repeated.values()
            for m in members
        ),
    }


def audit_image_independence(
    identity: ImageIdentityMap,
    *,
    interventions: Sequence[Mapping] = (),
    modality_records: Mapping[str, int] | None = None,
    concepts: Sequence[str] = (),
) -> dict:
    """How much of this run's apparent ``n`` is one photograph counted twice.

    Reports train and test separately, names every image that entered as more
    than one synchronized group, and counts the causal cells' targets at the
    distinct-image level.

    Two conditions are recorded as **hard failures**, because no re-aggregation
    can repair them: an image appearing in both splits, and sibling groups of
    one image landing on opposite sides of the split. Either makes every
    held-out claim in the run untrue.
    """
    modality_records = dict(modality_records or {})
    per_split: dict[str, list[GroupIdentity]] = {}
    for group in identity.groups.values():
        per_split.setdefault(group.split or "unassigned", []).append(group)

    images_by_split = {
        split: {group.image_id for group in members}
        for split, members in per_split.items()
    }
    overlap = sorted(
        images_by_split.get("train", set()) & images_by_split.get("test", set())
    )
    crossing = []
    for image_id, group_ids in identity.groups_of_image().items():
        splits = {identity.groups[g].split for g in group_ids}
        if len({s for s in splits if s}) > 1:
            crossing.append(
                {
                    "image_id": image_id,
                    "group_ids": group_ids,
                    "splits": sorted(s for s in splits if s),
                }
            )

    positives = [g for g in identity.groups.values() if g.concept]
    negatives = [g for g in identity.groups.values() if not g.concept]
    train_positive_images: dict[str, list[str]] = {}
    for group in positives:
        if group.split == "train":
            train_positive_images.setdefault(group.concept, []).append(group.image_id)

    cells: dict[tuple, dict] = {}
    for record in interventions:
        key = (
            record.get("concept"),
            record.get("source_modality"),
            record.get("target_modality"),
            record.get("layer"),
        )
        entry = cells.setdefault(
            key,
            {
                "concept": key[0],
                "source_modality": key[1],
                "target_modality": key[2],
                "pair": f"{key[1]}->{key[2]}",
                "layer": key[3],
                "positive_images": set(),
                "negative_images": set(),
                "positive_groups": set(),
                "negative_groups": set(),
            },
        )
        image_id = identity.image_for_sample(str(record["sample_id"]))
        group_id = str(record.get("group_id") or "")
        if record.get("target_is_positive"):
            entry["positive_images"].add(image_id)
            entry["positive_groups"].add(group_id)
        else:
            entry["negative_images"].add(image_id)
            entry["negative_groups"].add(group_id)

    causal_cells = [
        {
            "concept": entry["concept"],
            "source_modality": entry["source_modality"],
            "target_modality": entry["target_modality"],
            "pair": entry["pair"],
            "layer": entry["layer"],
            "n_positive_groups": len(entry["positive_groups"]),
            "n_negative_groups": len(entry["negative_groups"]),
            "n_positive_images": len(entry["positive_images"]),
            "n_negative_images": len(entry["negative_images"]),
            "positive_images": sorted(entry["positive_images"]),
            "negative_images": sorted(entry["negative_images"]),
            "targets_are_pseudoreplicated": (
                len(entry["positive_groups"]) > len(entry["positive_images"])
                or len(entry["negative_groups"]) > len(entry["negative_images"])
            ),
        }
        for entry in sorted(
            cells.values(),
            key=lambda item: (
                str(item["concept"]),
                item["pair"],
                str(item["layer"]),
            ),
        )
    ]

    hard_failures = []
    if overlap:
        hard_failures.append(
            {
                "kind": "train_test_image_overlap",
                "detail": f"{len(overlap)} image(s) appear in both splits: {overlap[:8]}",
            }
        )
    if crossing:
        hard_failures.append(
            {
                "kind": "sibling_groups_cross_splits",
                "detail": (
                    f"{len(crossing)} image(s) have sibling synchronized groups on "
                    f"opposite sides of the split: {crossing[:5]}"
                ),
            }
        )

    all_repeated = {
        image_id: group_ids
        for image_id, group_ids in identity.groups_of_image().items()
        if len(group_ids) > 1
    }
    return {
        "schema": "jlens.mmpilot.image_independence_audit.v1",
        "image_identity_rule_version": identity.rule_version,
        "independent_unit": "image_id",
        "n_groups": len(identity.groups),
        "n_distinct_images": identity.n_distinct_images,
        "n_images_with_multiple_groups": len(all_repeated),
        "images_with_multiple_groups": {
            image_id: group_ids for image_id, group_ids in sorted(all_repeated.items())
        },
        "concepts_affected": sorted(
            {
                identity.groups[group_id].concept
                for group_ids in all_repeated.values()
                for group_id in group_ids
                if identity.groups[group_id].concept
            }
        ),
        "n_modality_records_affected": sum(
            modality_records.get(group_id, 0)
            for group_ids in all_repeated.values()
            for group_id in group_ids
        ),
        "by_split": {
            split: _split_block(members, modality_records)
            for split, members in sorted(per_split.items())
        },
        "source_training": {
            "positive_images_per_concept": {
                concept: len(set(image_ids))
                for concept, image_ids in sorted(train_positive_images.items())
            },
            "positive_groups_per_concept": {
                concept: len(image_ids)
                for concept, image_ids in sorted(train_positive_images.items())
            },
            "n_negative_images": len(
                {g.image_id for g in negatives if g.split == "train"}
            ),
            "n_negative_groups": sum(1 for g in negatives if g.split == "train"),
        },
        "causal_cells": causal_cells,
        "train_test_image_overlap": overlap,
        "sibling_groups_crossing_splits": crossing,
        "hard_failures": hard_failures,
        "concepts_declared": list(concepts),
        "note": (
            "Repeated caption/audio groups from one photograph are never counted "
            "as independent image observations. Group-level counts are kept for "
            "provenance only."
        ),
    }


# ------------------------------------------- corrected representational tests


def build_representational_samples(
    activations: Sequence[Mapping],
    codes: Sequence[Mapping],
    identity: ImageIdentityMap,
    *,
    layer: int,
) -> list[dict]:
    """Join saved activations and codes at ``layer``, with resolved image ids.

    The saved ``image_id`` is deliberately overwritten by the canonical one:
    two artifacts written under different aliases must not read as two images.
    """
    by_sample = {
        record["sample_id"]: record
        for record in activations
        if int(record["layer"]) == layer
    }
    samples = []
    for code in codes:
        if int(code["layer"]) != layer:
            continue
        activation = by_sample.get(code["sample_id"])
        if activation is None:
            continue
        samples.append(
            {
                "sample_id": code["sample_id"],
                "group_id": code["group_id"],
                "image_id": identity.image_for_sample(code["sample_id"]),
                "concept": code["concept"],
                "modality": code["modality"],
                "split": code.get("split"),
                "code": code_map(code),
                "activation": activation["activation"],
            }
        )
    return samples


def recompute_representational(
    activations: Sequence[Mapping],
    codes: Sequence[Mapping],
    identity: ImageIdentityMap,
    *,
    layer: int,
    modalities: Sequence[str],
    n_permutations: int = 50,
    seed: int = 20260731,
) -> dict:
    """The representational report under image-disjoint retrieval.

    Every test in the report — nearest-neighbour retrieval, matched-versus-
    mismatched separation, weighted support overlap, the raw-residual baseline
    and the shuffled-label control — runs through the same admissibility rule,
    so none of them can be image-disjoint while another is not.

    Raises:
        jlens.mmpilot.jspace.NoEligibleTargetError: If a modality pair keeps no
            evaluable query at all.
    """
    samples = build_representational_samples(
        activations, codes, identity, layer=layer
    )
    report = representational_report(
        samples,
        modalities=modalities,
        n_permutations=n_permutations,
        seed=seed,
        strict=True,
    )
    report["layer"] = layer
    report["image_identity_rule_version"] = identity.rule_version
    report["exclusion_rule_version"] = EXCLUSION_RULE_VERSION
    report["image_disjoint"] = True
    return report


# ------------------------------------------------------ causal re-aggregation


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _image_block(records: Sequence[Mapping]) -> dict:
    """One image's repeated observations, collapsed to a single unit."""
    effects = [float(r["signed_target_effect"]) for r in records]
    margins = [float(r["signed_margin_effect"]) for r in records]
    return {
        "n_records": len(records),
        "group_ids": sorted({str(r.get("group_id") or "") for r in records}),
        "sample_ids": sorted({str(r["sample_id"]) for r in records}),
        "target_is_positive": bool(records[0].get("target_is_positive")),
        "mean_signed_target_effect": _mean(effects),
        "mean_signed_margin_effect": _mean(margins),
        "fraction_expected_sign": _mean([1.0 if e > 0 else 0.0 for e in effects]),
        "mean_abs_unrelated_change": _mean(
            [float(r["max_abs_unrelated_change"]) for r in records]
        ),
        "mean_activation_norm_ratio": _mean(
            [float(r["activation_norm_ratio"]) for r in records]
        ),
        "n_prediction_changes": sum(1 for r in records if r["prediction_changed"]),
    }


def summarize_interventions_by_image(
    records: Sequence[Mapping],
    identity: ImageIdentityMap,
    *,
    group_summary: Mapping | None = None,
) -> dict:
    """Re-aggregate saved intervention units with the image as the unit.

    No intervention is rerun. Every saved unit is mapped to its image, repeated
    observations of one image are averaged **within** the image, and the cell
    statistic is then computed over images. ``n`` therefore counts photographs,
    not captions.

    The group-level row for each cell is preserved verbatim under
    ``group_level`` and the difference is reported under
    ``divergence_from_group_level``, so a reader can see exactly what the
    correction changed rather than being handed a replacement.

    Args:
        group_summary: The original ``summarize_interventions`` output, if it
            was saved. When absent the group-level rows are recomputed from the
            same records, which gives the same numbers.
    """
    from jlens.mmpilot.causal import summarize_interventions

    group_rows = {
        (
            row["concept"],
            row["source_modality"],
            row["target_modality"],
            row["layer"],
            row["control_kind"],
            float(row["alpha"]),
        ): row
        for row in (group_summary or summarize_interventions(records))["rows"]
    }

    cells: dict[tuple, list[Mapping]] = {}
    for record in records:
        key = (
            record["concept"],
            record["source_modality"],
            record["target_modality"],
            record["layer"],
            record["control_kind"],
            float(record["alpha"]),
        )
        cells.setdefault(key, []).append(record)

    role_conflicts: list[dict] = []
    id_disagreements: list[dict] = []
    rows: list[dict] = []
    for key, group in sorted(cells.items(), key=lambda item: [str(x) for x in item[0]]):
        concept, source, target, layer, control, alpha = key
        by_image: dict[str, list[Mapping]] = {}
        for record in group:
            image_id = identity.image_for_sample(str(record["sample_id"]))
            recorded = record.get("image_id")
            if recorded not in (None, "") and str(recorded) != image_id:
                # The unit was written before identities were canonicalized, or
                # under a different alias. Reported, never silently preferred.
                id_disagreements.append(
                    {
                        "sample_id": record["sample_id"],
                        "recorded_image_id": str(recorded),
                        "resolved_image_id": image_id,
                    }
                )
            by_image.setdefault(image_id, []).append(record)

        per_image = {}
        for image_id, members in sorted(by_image.items()):
            roles = {bool(r.get("target_is_positive")) for r in members}
            if len(roles) > 1:
                role_conflicts.append(
                    {
                        "image_id": image_id,
                        "cell": f"{concept}|{source}->{target}|L{layer}|{control}|a{alpha:g}",
                        "sample_ids": sorted(str(r["sample_id"]) for r in members),
                    }
                )
            per_image[image_id] = _image_block(members)

        positives = [b for b in per_image.values() if b["target_is_positive"]]
        negatives = [b for b in per_image.values() if not b["target_is_positive"]]
        blocks = list(per_image.values())
        group_row = group_rows.get(key)
        image_effect = _mean([b["mean_signed_target_effect"] for b in blocks])
        rows.append(
            {
                "concept": concept,
                "source_modality": source,
                "target_modality": target,
                "pair": f"{source}->{target}",
                "off_diagonal": source != target,
                "layer": layer,
                "control_kind": control,
                "alpha": alpha,
                # ``n`` is the independent unit count the rubric reads.
                "n": len(blocks),
                "n_distinct_images": len(blocks),
                "n_groups": len({str(r.get("group_id") or "") for r in group}),
                "n_records": len(group),
                "n_positive_images": len(positives),
                "n_negative_images": len(negatives),
                "n_positive_groups": len(
                    {
                        str(r.get("group_id") or "")
                        for r in group
                        if r.get("target_is_positive")
                    }
                ),
                "n_negative_groups": len(
                    {
                        str(r.get("group_id") or "")
                        for r in group
                        if not r.get("target_is_positive")
                    }
                ),
                "mean_signed_target_effect": image_effect,
                "mean_signed_margin_effect": _mean(
                    [b["mean_signed_margin_effect"] for b in blocks]
                ),
                "fraction_expected_sign": _mean(
                    [b["fraction_expected_sign"] for b in blocks]
                ),
                "mean_abs_unrelated_change": _mean(
                    [b["mean_abs_unrelated_change"] for b in blocks]
                ),
                "mean_activation_norm_ratio": _mean(
                    [b["mean_activation_norm_ratio"] for b in blocks]
                ),
                "n_prediction_changes": sum(
                    1 for b in blocks if b["n_prediction_changes"]
                ),
                "n_prediction_changes_groups": (group_row or {}).get(
                    "n_prediction_changes"
                ),
                "mean_positive_image_effect": _mean(
                    [b["mean_signed_target_effect"] for b in positives]
                ),
                "mean_negative_image_effect": _mean(
                    [b["mean_signed_target_effect"] for b in negatives]
                ),
                "evidence_is_single_image": len(blocks) <= 1,
                "pseudoreplicated_at_group_level": len(group) > len(blocks),
                "per_image": per_image,
                "group_level": dict(group_row) if group_row else None,
                "divergence_from_group_level": (
                    image_effect - float(group_row["mean_signed_target_effect"])
                    if group_row
                    else None
                ),
                "aggregation_version": CAUSAL_AGGREGATION_VERSION,
            }
        )

    hard_failures = []
    if role_conflicts:
        hard_failures.append(
            {
                "kind": "image_is_both_positive_and_negative_target",
                "detail": role_conflicts[:5],
            }
        )
    all_images = {
        identity.image_for_sample(str(r["sample_id"])) for r in records
    }
    return {
        "schema": "jlens.mmpilot.interventions_image_level.v1",
        "aggregation_version": CAUSAL_AGGREGATION_VERSION,
        "image_identity_rule_version": identity.rule_version,
        "independent_unit": "image_id",
        "n_records": len(records),
        "n_distinct_images_overall": len(all_images),
        "n_groups_overall": len({str(r.get("group_id") or "") for r in records}),
        "rows": rows,
        "group_level_rows": list(group_rows.values()),
        "control_kinds": list(CONTROL_KINDS),
        "recorded_image_id_disagreements": id_disagreements[:20],
        "hard_failures": hard_failures,
        "note": (
            "Image targets are never counted twice: identical image conditions "
            "are averaged within the image first. Distinct captions from one "
            "image remain visible per cell under `per_image`, as descriptive "
            "group-level detail only."
        ),
    }


def divergence_summary(image_level: Mapping) -> dict:
    """How far the image-level answer moved from the group-level one."""
    divergences = [
        row["divergence_from_group_level"]
        for row in image_level.get("rows", [])
        if row.get("divergence_from_group_level") is not None
    ]
    off_diagonal = [
        row
        for row in image_level.get("rows", [])
        if row["off_diagonal"] and row["control_kind"] == "source_concept"
    ]
    return {
        "n_rows": len(image_level.get("rows", [])),
        "n_rows_pseudoreplicated_at_group_level": sum(
            1 for row in image_level.get("rows", []) if row["pseudoreplicated_at_group_level"]
        ),
        "max_abs_divergence": max((abs(d) for d in divergences), default=0.0),
        "median_abs_divergence": (
            statistics.median([abs(d) for d in divergences]) if divergences else 0.0
        ),
        "off_diagonal_source_rows_on_a_single_image": [
            {
                "cell": f"{row['concept']}|{row['pair']}|a{row['alpha']:g}",
                "n_distinct_images": row["n_distinct_images"],
            }
            for row in off_diagonal
            if row["evidence_is_single_image"]
        ],
    }


__all__ = [
    "CAUSAL_AGGREGATION_VERSION",
    "IMAGE_IDENTITY_RULE_VERSION",
    "IMAGE_ID_FIELDS",
    "GroupIdentity",
    "ImageIdentityError",
    "ImageIdentityMap",
    "audit_image_independence",
    "build_representational_samples",
    "divergence_summary",
    "recompute_representational",
    "resolve_image_identity",
    "summarize_interventions_by_image",
]
