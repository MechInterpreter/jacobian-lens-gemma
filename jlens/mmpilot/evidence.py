# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""What counts as evidence that a concept is present in a synchronized group.

The first real run of this pilot failed its behavioral gate asymmetrically:
image 8/8 for every concept, text 3/8 to 6/8. The cause was a labelling rule,
not the model. A group was accepted as a ``cat`` positive whenever the COCO
*object annotation* for its image listed ``cat`` — the written caption was
never consulted. SpokenCOCO images routinely carry an object the caption does
not mention, so the text arm was asked "is there a cat?" about a sentence with
no cat in it, and was marked wrong for answering honestly.

That group holds **visual** evidence. It does not hold **written** evidence.
It is therefore not a valid synchronized positive for a transfer experiment,
because the two modalities are not carrying the same claim.

This module states the rule once, in one place:

    A valid synchronized positive requires BOTH
      1. visual evidence  — a COCO object annotation for the concept, and
      2. caption evidence — the concept or an approved synonym present in the
         written caption as a whole word or phrase.

Everything here is CPU-only, deterministic, and inspectable. Matching is a
normalized whole-word regex against an explicit lexicon: no embeddings, no
learned classifier, no language model, no fuzzy similarity, no network call.
``cat`` does not match ``cattle``.

Because SpokenCOCO's recordings are spoken readings of the written captions,
caption evidence also establishes what linguistic content the *recording*
carries. That is a fact about the dataset's metadata. It is **not** evidence
that Gemma recognises speech, and no audio is transcribed here.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from jlens.mmpilot.store import UnitStore, payload_checksum

EVIDENCE_SCHEMA_VERSION = "jlens.mmpilot.synchronized_evidence.v1"
AUDIT_SCHEMA_VERSION = "jlens.mmpilot.evidence_audit.v1"
RANKING_SCHEMA_VERSION = "jlens.mmpilot.evidence_ranking.v1"


class IncompatibleEvidenceStateError(RuntimeError):
    """A stored evidence artifact was produced under a different configuration."""


# --------------------------------------------------------------- the lexicon

#: Canonical concept -> the written terms that count as expressing it.
#:
#: Deliberately small and boring. Every entry is the concept word itself, its
#: plural, or a synonym a careful reader would accept without argument
#: (a *kitten* is a cat; a *puppy* is a dog). Nothing here is a guess about
#: what a caption "probably means".
#:
#: Known limitation, stated rather than hidden: whole-word matching has no
#: part-of-speech sense, so "train for a marathon" matches ``train``. That is
#: why :func:`format_examples` prints matched captions with their spans before
#: any model pass — the lexicon is meant to be *checked*, not trusted.
CONCEPT_LEXICON: dict[str, tuple[str, ...]] = {
    "bus": ("bus", "buses", "busses"),
    "cat": ("cat", "cats", "kitten", "kittens", "kitty", "kitties"),
    "dog": ("dog", "dogs", "puppy", "puppies"),
    "horse": ("horse", "horses"),
    "pizza": ("pizza", "pizzas"),
    "train": ("train", "trains"),
}

#: Canonical concept -> the official COCO object category names that count as
#: visual evidence for it. COCO-80 names, verbatim.
CONCEPT_COCO_CATEGORIES: dict[str, tuple[str, ...]] = {
    "bus": ("bus",),
    "cat": ("cat",),
    "dog": ("dog",),
    "horse": ("horse",),
    "pizza": ("pizza",),
    "train": ("train",),
}

#: Rejection reasons, enumerated so counts are comparable across runs.
REASON_VALID = "valid_synchronized_positive"
REASON_NO_VISUAL = "no_visual_annotation_evidence"
REASON_NO_CAPTION = "no_caption_lexical_evidence"
REASON_NO_EITHER = "no_visual_or_caption_evidence"
REASON_NO_CAPTION_TEXT = "group_has_no_caption_text"
REASON_UNSYNCHRONIZED = "group_is_not_a_synchronized_image_caption_audio_triple"

REJECTION_REASONS = (
    REASON_VALID,
    REASON_NO_VISUAL,
    REASON_NO_CAPTION,
    REASON_NO_EITHER,
    REASON_NO_CAPTION_TEXT,
    REASON_UNSYNCHRONIZED,
)


def normalize_caption(text: object) -> str:
    """Deterministic caption normalization used for every lexical decision.

    NFKC, casefold, non-alphanumerics collapsed to single spaces. Recorded
    alongside every match so a span can be re-checked by hand.
    """
    folded = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return re.sub(r"[^0-9a-z]+", " ", folded).strip()


def _term_pattern(term: str) -> re.Pattern[str] | None:
    """Whole-word / whole-phrase pattern for ``term`` on a normalized caption."""
    normalized = normalize_caption(term)
    if not normalized:
        return None
    return re.compile(rf"(?<![0-9a-z]){re.escape(normalized)}(?![0-9a-z])")


def find_term(normalized_caption: str, term: str) -> tuple[int, int] | None:
    """First whole-word match span of ``term``, or ``None``.

    Substring hits never count: ``cat`` does not match ``cattle`` and ``bus``
    does not match ``business``.
    """
    pattern = _term_pattern(term)
    if pattern is None:
        return None
    match = pattern.search(normalized_caption)
    return (match.start(), match.end()) if match else None


# ---------------------------------------------------------------- the config


@dataclass(frozen=True)
class EvidenceConfig:
    """The evidence rule, as data, so a run can be fingerprinted against it.

    Args:
        lexicon: Concept -> approved written terms.
        coco_categories: Concept -> official COCO object category names.
        require_visual_evidence: A positive must carry a COCO annotation.
        require_caption_evidence: A positive must carry a caption term match.

    Both requirements default to True. Turning either off changes what a
    positive *means*, so it changes the fingerprint and a run directory built
    under the other setting is refused rather than mixed.
    """

    lexicon: Mapping[str, tuple[str, ...]] = field(
        default_factory=lambda: {k: tuple(v) for k, v in CONCEPT_LEXICON.items()}
    )
    coco_categories: Mapping[str, tuple[str, ...]] = field(
        default_factory=lambda: {k: tuple(v) for k, v in CONCEPT_COCO_CATEGORIES.items()}
    )
    require_visual_evidence: bool = True
    require_caption_evidence: bool = True
    normalization: str = "nfkc_casefold_alnum_whole_word"
    matching: str = "explicit_lexicon_whole_word_regex"

    def terms_for(self, concept: str) -> tuple[str, ...]:
        return tuple(self.lexicon.get(concept, (concept,)))

    def categories_for(self, concept: str) -> tuple[str, ...]:
        return tuple(self.coco_categories.get(concept, (concept,)))

    @property
    def concepts(self) -> tuple[str, ...]:
        return tuple(sorted(self.lexicon))

    def to_dict(self) -> dict:
        return {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "lexicon": {k: list(v) for k, v in sorted(self.lexicon.items())},
            "coco_categories": {
                k: list(v) for k, v in sorted(self.coco_categories.items())
            },
            "require_visual_evidence": self.require_visual_evidence,
            "require_caption_evidence": self.require_caption_evidence,
            "normalization": self.normalization,
            "matching": self.matching,
            "audio_transcribed": False,
            "learned_components": [],
        }

    @property
    def lexicon_hash(self) -> str:
        """Hash of the lexicon and categories alone (not the flags)."""
        return payload_checksum(
            {
                "lexicon": {k: list(v) for k, v in sorted(self.lexicon.items())},
                "coco_categories": {
                    k: list(v) for k, v in sorted(self.coco_categories.items())
                },
            }
        )

    @property
    def fingerprint(self) -> str:
        """Hash of the whole rule. Reuse across a different value is refused."""
        return payload_checksum(self.to_dict())


def default_config() -> EvidenceConfig:
    """The pilot's evidence rule: visual AND caption evidence, both required."""
    return EvidenceConfig()


def config_for_concepts(
    concepts: Mapping[str, Sequence[str]] | None = None, **flags
) -> EvidenceConfig:
    """Build a config for a caller-supplied ``{concept: keywords}`` mapping.

    The canonical concept name is always an approved term for itself, and COCO
    categories fall back to the concept name when the concept is not one of the
    six the pilot ships a mapping for.
    """
    if not concepts:
        return EvidenceConfig(**flags)
    lexicon = {
        str(name): tuple(dict.fromkeys([str(name), *(str(w) for w in terms)]))
        for name, terms in concepts.items()
    }
    categories = {
        str(name): CONCEPT_COCO_CATEGORIES.get(str(name), (str(name),))
        for name in concepts
    }
    return EvidenceConfig(lexicon=lexicon, coco_categories=categories, **flags)


def has_any_evidence(group: Mapping, concept: str, config: EvidenceConfig) -> bool:
    """Either kind of evidence. Used to *disqualify* matched negatives.

    A negative that carries visual-only evidence for the concept is still a
    picture of the concept; contrasting against it would blunt the very
    direction the contrast is meant to isolate.
    """
    return (
        visual_evidence(group, concept, config)["present"]
        or caption_evidence(group, concept, config)["present"]
    )


def is_valid_positive(group: Mapping, concept: str, config: EvidenceConfig) -> bool:
    """The one rule: visual annotation evidence AND caption lexical evidence."""
    return bool(group_evidence(group, concept, config)["is_valid_synchronized_positive"])


# ------------------------------------------------------------ per-group facts


def _annotations(group: Mapping) -> list[str]:
    return sorted({str(label).casefold() for label in group.get("concept_annotations", []) or []})


def visual_evidence(group: Mapping, concept: str, config: EvidenceConfig) -> dict:
    """COCO object-annotation evidence for ``concept`` on this group's image."""
    present = _annotations(group)
    wanted = {name.casefold() for name in config.categories_for(concept)}
    matched = sorted(set(present) & wanted)
    return {
        "present": bool(matched),
        "matched_categories": matched,
        "image_annotations": present,
        "source": "coco_object_annotation",
        "annotation_available": bool(present) or bool(group.get("annotation_source")),
    }


def caption_evidence(group: Mapping, concept: str, config: EvidenceConfig) -> dict:
    """Whole-word lexicon evidence for ``concept`` in this group's caption."""
    caption = str(group.get("caption", "") or "")
    normalized = normalize_caption(caption)
    for term in config.terms_for(concept):
        span = find_term(normalized, term)
        if span is not None:
            return {
                "present": True,
                "matched_term": term,
                "match_span": list(span),
                "matched_text": normalized[span[0] : span[1]],
                "normalized_caption": normalized,
                "terms_considered": list(config.terms_for(concept)),
                "source": "caption_lexicon_whole_word",
            }
    return {
        "present": False,
        "matched_term": None,
        "match_span": None,
        "matched_text": None,
        "normalized_caption": normalized,
        "terms_considered": list(config.terms_for(concept)),
        "source": "caption_lexicon_whole_word",
    }


def _is_synchronized(group: Mapping) -> bool:
    return bool(
        group.get("image_id")
        and group.get("image_path")
        and group.get("audio_path")
        and str(group.get("caption", "")).strip()
    )


def group_evidence(group: Mapping, concept: str, config: EvidenceConfig) -> dict:
    """One audit record: every field needed to re-derive the verdict by hand."""
    visual = visual_evidence(group, concept, config)
    caption = caption_evidence(group, concept, config)
    synchronized = _is_synchronized(group)

    if not synchronized:
        reason = (
            REASON_NO_CAPTION_TEXT
            if not str(group.get("caption", "")).strip()
            else REASON_UNSYNCHRONIZED
        )
        valid = False
    elif config.require_visual_evidence and config.require_caption_evidence:
        if visual["present"] and caption["present"]:
            valid, reason = True, REASON_VALID
        elif not visual["present"] and not caption["present"]:
            valid, reason = False, REASON_NO_EITHER
        elif not visual["present"]:
            valid, reason = False, REASON_NO_VISUAL
        else:
            valid, reason = False, REASON_NO_CAPTION
    else:
        needed = []
        if config.require_visual_evidence and not visual["present"]:
            needed.append(REASON_NO_VISUAL)
        if config.require_caption_evidence and not caption["present"]:
            needed.append(REASON_NO_CAPTION)
        has_any = visual["present"] or caption["present"]
        valid = not needed and has_any
        reason = REASON_VALID if valid else (needed[0] if needed else REASON_NO_EITHER)

    split_provenance = dict(group.get("split_provenance") or {})
    split_provenance.setdefault("source_split", group.get("source_split"))

    return {
        "group_id": group.get("group_id"),
        "synchronized_group_id": group.get("synchronized_group_id", group.get("group_id")),
        "image_id": group.get("image_id"),
        "caption_id": group.get("caption_id"),
        "audio_record_id": group.get("audio_record_id"),
        "audio_path": group.get("audio_path"),
        "image_path": group.get("image_path"),
        "speaker": group.get("speaker"),
        "split_provenance": split_provenance,
        "concept": concept,
        "caption": str(group.get("caption", "") or ""),
        "normalized_caption": caption["normalized_caption"],
        "visual_evidence": visual,
        "caption_evidence": caption,
        "matched_term": caption["matched_term"],
        "match_span": caption["match_span"],
        "spoken_evidence": {
            "basis": "caption_is_the_read_script",
            "present": caption["present"],
            "audio_transcribed": False,
            "note": (
                "SpokenCOCO recordings are spoken readings of the written "
                "caption, so caption evidence describes the recording's "
                "linguistic content. It says nothing about whether the model "
                "can hear."
            ),
        },
        "is_valid_synchronized_positive": valid,
        "rejection_reason": None if valid else reason,
        "evidence_rule": "visual_annotation_AND_caption_lexicon",
        "lexicon_hash": config.lexicon_hash,
        "config_fingerprint": config.fingerprint,
    }


def negative_evidence(
    group: Mapping, concepts: Sequence[str], config: EvidenceConfig
) -> dict:
    """Why this group qualifies as a matched negative for every concept.

    A negative must carry neither the visual annotation nor a caption term for
    *any* screened concept — otherwise it is a positive the audit missed, and
    contrasting against it would blunt the direction it is meant to isolate.
    """
    per_concept = {}
    disqualifying = []
    for concept in sorted(concepts):
        visual = visual_evidence(group, concept, config)
        caption = caption_evidence(group, concept, config)
        per_concept[concept] = {
            "visual_evidence_present": visual["present"],
            "caption_evidence_present": caption["present"],
            "matched_categories": visual["matched_categories"],
            "matched_term": caption["matched_term"],
        }
        if visual["present"] or caption["present"]:
            disqualifying.append(concept)
    return {
        "group_id": group.get("group_id"),
        "image_id": group.get("image_id"),
        "caption_id": group.get("caption_id"),
        "audio_path": group.get("audio_path"),
        "speaker": group.get("speaker"),
        "caption": str(group.get("caption", "") or ""),
        "normalized_caption": normalize_caption(group.get("caption")),
        "per_concept": per_concept,
        "qualifies_as_negative": not disqualifying,
        "disqualifying_concepts": disqualifying,
        "qualification_rule": (
            "no COCO object annotation and no approved caption term for any "
            "screened concept"
        ),
        "lexicon_hash": config.lexicon_hash,
    }


# ------------------------------------------------------------------ the audit


@dataclass
class EvidenceAudit:
    """Every candidate positive, with the verdict and the reason behind it."""

    records: list[dict]
    negatives: list[dict]
    config: EvidenceConfig
    concepts: tuple[str, ...]
    n_groups: int = 0
    source_checksums: dict = field(default_factory=dict)

    def valid_records(self, concept: str | None = None) -> list[dict]:
        return [
            record
            for record in self.records
            if record["is_valid_synchronized_positive"]
            and (concept is None or record["concept"] == concept)
        ]

    def rejected_records(self, concept: str | None = None) -> list[dict]:
        return [
            record
            for record in self.records
            if not record["is_valid_synchronized_positive"]
            and (concept is None or record["concept"] == concept)
        ]

    def rejection_counts(self) -> dict[str, int]:
        counts = {reason: 0 for reason in REJECTION_REASONS}
        for record in self.records:
            counts[record["rejection_reason"] or REASON_VALID] += 1
        return counts

    def rejection_counts_by_concept(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for record in self.records:
            per = out.setdefault(
                record["concept"], {reason: 0 for reason in REJECTION_REASONS}
            )
            per[record["rejection_reason"] or REASON_VALID] += 1
        return out

    def valid_group_ids(self, concept: str) -> set[str]:
        return {record["group_id"] for record in self.valid_records(concept)}

    def to_dict(self, *, conversion: Mapping | None = None) -> dict:
        return {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "evidence_rule": (
                "a valid synchronized positive requires BOTH a COCO object "
                "annotation for the concept AND the concept or an approved "
                "synonym in the written caption, matched whole-word"
            ),
            "evidence_config": self.config.to_dict(),
            "lexicon_hash": self.config.lexicon_hash,
            "config_fingerprint": self.config.fingerprint,
            "concepts_screened": list(self.concepts),
            "n_groups_examined": self.n_groups,
            "n_candidate_positives": len(self.records),
            "n_valid_synchronized_positives": len(self.valid_records()),
            "rejection_counts": self.rejection_counts(),
            "rejection_counts_by_concept": self.rejection_counts_by_concept(),
            "source_metadata_checksums": dict(self.source_checksums),
            "conversion": dict(conversion or {}),
            "conversion_hash": payload_checksum(dict(conversion or {})),
            "audio_transcribed": False,
            "original_manifest_mutated": False,
            "records": self.records,
            "negatives": self.negatives,
        }


def audit_groups(
    groups: Sequence[Mapping],
    *,
    config: EvidenceConfig | None = None,
    concepts: Sequence[str] | None = None,
    source_checksums: Mapping | None = None,
) -> EvidenceAudit:
    """Audit every group against every screened concept. CPU only, no model.

    A group becomes a *candidate positive* for a concept when it carries either
    kind of evidence for it; the record then says whether both kinds were
    present. Groups carrying no evidence for any concept are audited as
    candidate negatives instead.
    """
    config = config or default_config()
    screened = tuple(sorted(concepts if concepts is not None else config.concepts))
    records: list[dict] = []
    negatives: list[dict] = []
    for group in sorted(groups, key=lambda g: str(g.get("group_id"))):
        touched = False
        for concept in screened:
            visual = visual_evidence(group, concept, config)
            caption = caption_evidence(group, concept, config)
            if not (visual["present"] or caption["present"]):
                continue
            touched = True
            records.append(group_evidence(group, concept, config))
        if not touched:
            negatives.append(negative_evidence(group, screened, config))
    return EvidenceAudit(
        records=records,
        negatives=negatives,
        config=config,
        concepts=screened,
        n_groups=len(groups),
        source_checksums=dict(source_checksums or {}),
    )


def annotate_groups(
    groups: Sequence[Mapping], audit: EvidenceAudit
) -> list[dict]:
    """Copy ``groups`` with the audit's verdicts attached.

    Adds ``valid_positive_concepts`` (the concepts this group is a valid
    synchronized positive for) and ``evidence`` (the per-concept record). The
    inputs are never mutated: selection reads the copies.
    """
    by_group: dict[str, dict[str, dict]] = {}
    for record in audit.records:
        by_group.setdefault(str(record["group_id"]), {})[record["concept"]] = record
    out = []
    for group in groups:
        per_concept = by_group.get(str(group.get("group_id")), {})
        valid = sorted(
            concept
            for concept, record in per_concept.items()
            if record["is_valid_synchronized_positive"]
        )
        out.append(
            {
                **dict(group),
                "evidence": per_concept,
                "valid_positive_concepts": valid,
                "evidence_lexicon_hash": audit.config.lexicon_hash,
                "evidence_config_fingerprint": audit.config.fingerprint,
            }
        )
    return out


def synchronized_manifest_payload(
    groups: Sequence[Mapping],
    audit: EvidenceAudit,
    *,
    original_checksum: str,
    conversion: Mapping,
) -> dict:
    """The derived synchronized-evidence manifest, as written to the run dir."""
    annotated = annotate_groups(groups, audit)
    return {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "original_manifest_checksum": original_checksum,
        "original_manifest_mutated": False,
        "media_redownloaded": False,
        "audio_transcribed": False,
        "evidence_config": audit.config.to_dict(),
        "lexicon_hash": audit.config.lexicon_hash,
        "config_fingerprint": audit.config.fingerprint,
        "source_metadata_checksums": dict(audit.source_checksums),
        "conversion": dict(conversion),
        "conversion_hash": payload_checksum(dict(conversion)),
        "concepts_screened": list(audit.concepts),
        "rejection_counts": audit.rejection_counts(),
        "n_groups": len(annotated),
        "n_valid_synchronized_positives": len(audit.valid_records()),
        "n_candidate_negatives": len(audit.negatives),
        "groups": annotated,
    }


# ---------------------------------------------------------------- persistence


def _persist(
    path: str | os.PathLike[str],
    payload: Mapping,
    *,
    identity_keys: Sequence[str],
    label: str,
) -> tuple[dict, str]:
    """Atomically write ``payload``, or reuse a byte-compatible existing file.

    Reuse requires every key in ``identity_keys`` to match. A file that
    disagrees is refused with :class:`IncompatibleEvidenceStateError` rather
    than overwritten — an audit silently replaced under a different lexicon is
    exactly the failure this repair exists to prevent.
    """
    path = Path(path)
    payload = dict(payload)
    if path.is_file():
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            stored = {}
        if stored:
            differing = [
                f"{key}: stored={stored.get(key)!r} requested={payload.get(key)!r}"
                for key in identity_keys
                if stored.get(key) != payload.get(key)
            ]
            if differing:
                raise IncompatibleEvidenceStateError(
                    f"{path} holds a {label} produced under a different "
                    "configuration; refusing to reuse or overwrite it.\n  "
                    + "\n  ".join(differing)
                    + "\nPoint the run at a new directory (or delete this file "
                    "yourself)."
                )
            return stored, f"resuming: reused checksum-compatible {label}"
    UnitStore._write_json(path, payload)
    return payload, f"starting: wrote {label} atomically"


def persist_evidence_audit(
    path: str | os.PathLike[str],
    audit: EvidenceAudit,
    *,
    conversion: Mapping,
) -> tuple[dict, str]:
    """Write ``synchronized_evidence_audit.json``; resume only if compatible."""
    return _persist(
        path,
        audit.to_dict(conversion=conversion),
        identity_keys=(
            "schema_version",
            "config_fingerprint",
            "lexicon_hash",
            "conversion_hash",
            "source_metadata_checksums",
            "concepts_screened",
        ),
        label="evidence audit",
    )


def persist_synchronized_manifest(
    path: str | os.PathLike[str],
    groups: Sequence[Mapping],
    audit: EvidenceAudit,
    *,
    original_checksum: str,
    conversion: Mapping,
) -> tuple[dict, str]:
    """Write ``synchronized_evidence_manifest.json``; resume only if compatible."""
    return _persist(
        path,
        synchronized_manifest_payload(
            groups, audit, original_checksum=original_checksum, conversion=conversion
        ),
        identity_keys=(
            "schema_version",
            "original_manifest_checksum",
            "config_fingerprint",
            "lexicon_hash",
            "conversion_hash",
            "source_metadata_checksums",
        ),
        label="synchronized-evidence manifest",
    )


def persist_concept_ranking(
    path: str | os.PathLike[str],
    rows: Sequence[Mapping],
    audit: EvidenceAudit,
    *,
    requirements: Mapping,
    conversion: Mapping,
) -> tuple[dict, str]:
    """Write the concept ranking; resume only if compatible."""
    payload = {
        "schema_version": RANKING_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_fingerprint": audit.config.fingerprint,
        "lexicon_hash": audit.config.lexicon_hash,
        "conversion_hash": payload_checksum(dict(conversion)),
        "conversion": dict(conversion),
        "requirements": dict(requirements),
        "ranked_from": "valid_synchronized_positives_only",
        "rejection_counts": audit.rejection_counts(),
        "rows": [dict(row) for row in rows],
    }
    return _persist(
        path,
        payload,
        identity_keys=(
            "schema_version",
            "config_fingerprint",
            "lexicon_hash",
            "conversion_hash",
            "requirements",
        ),
        label="concept ranking",
    )


def source_checksums(paths: Sequence[str | os.PathLike[str]]) -> dict[str, str]:
    """``sha256:`` digests of the source metadata files the audit read."""
    out: dict[str, str] = {}
    for candidate in paths:
        path = Path(candidate)
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        out[str(path)] = "sha256:" + digest.hexdigest()
    return out


# --------------------------------------------------------- printed inspection


def format_examples(
    audit: EvidenceAudit,
    concepts: Sequence[str],
    *,
    n_positive: int = 3,
    n_rejected: int = 3,
    caption_width: int = 88,
) -> str:
    """Representative accepted and rejected groups, for manual verification.

    Bounded on purpose: enough rows to check the rule by eye, never the whole
    dataset.
    """

    def clip(text: str) -> str:
        text = " ".join(str(text).split())
        return text if len(text) <= caption_width else text[: caption_width - 1] + "…"

    lines: list[str] = []
    for concept in sorted(concepts):
        valid = audit.valid_records(concept)[:n_positive]
        rejected = audit.rejected_records(concept)[:n_rejected]
        lines.append(f"\nconcept {concept!r}")
        lines.append(
            f"  accepted: {len(audit.valid_records(concept))}  "
            f"rejected: {len(audit.rejected_records(concept))}"
        )
        lines.append("  -- accepted (visual AND caption evidence) --")
        if not valid:
            lines.append("     (none)")
        for record in valid:
            lines.append(f"     group {record['group_id']}  image {record['image_id']}")
            lines.append(f"       caption: {clip(record['caption'])}")
            lines.append(
                f"       visual:  {record['visual_evidence']['matched_categories']} "
                f"(of {record['visual_evidence']['image_annotations']})"
            )
            lines.append(
                f"       caption term: {record['matched_term']!r} "
                f"at span {record['match_span']}"
            )
        lines.append("  -- rejected --")
        if not rejected:
            lines.append("     (none)")
        for record in rejected:
            lines.append(f"     group {record['group_id']}  image {record['image_id']}")
            lines.append(f"       caption: {clip(record['caption'])}")
            lines.append(
                f"       visual:  {record['visual_evidence']['matched_categories'] or 'none'}"
            )
            lines.append(f"       caption term: {record['matched_term'] or 'none'}")
            lines.append(f"       rejected because: {record['rejection_reason']}")
    return "\n".join(lines)


def format_rejection_counts(audit: EvidenceAudit) -> str:
    """One line per rejection reason, over all screened concepts."""
    counts = audit.rejection_counts()
    width = max(len(reason) for reason in counts)
    return "\n".join(
        f"  {reason:{width}s}  {counts[reason]:5d}" for reason in REJECTION_REASONS
    )


__all__ = [
    "AUDIT_SCHEMA_VERSION",
    "CONCEPT_COCO_CATEGORIES",
    "CONCEPT_LEXICON",
    "EVIDENCE_SCHEMA_VERSION",
    "RANKING_SCHEMA_VERSION",
    "REASON_NO_CAPTION",
    "REASON_NO_CAPTION_TEXT",
    "REASON_NO_EITHER",
    "REASON_NO_VISUAL",
    "REASON_UNSYNCHRONIZED",
    "REASON_VALID",
    "REJECTION_REASONS",
    "EvidenceAudit",
    "EvidenceConfig",
    "IncompatibleEvidenceStateError",
    "annotate_groups",
    "audit_groups",
    "caption_evidence",
    "config_for_concepts",
    "default_config",
    "find_term",
    "format_examples",
    "format_rejection_counts",
    "group_evidence",
    "has_any_evidence",
    "is_valid_positive",
    "negative_evidence",
    "normalize_caption",
    "persist_concept_ranking",
    "persist_evidence_audit",
    "persist_synchronized_manifest",
    "source_checksums",
    "synchronized_manifest_payload",
    "visual_evidence",
]
