# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Cone records, signatures, and trajectory utilities for J-space
decompositions.

A :func:`gradient pursuit <jlens.pursuit.gradient_pursuit>` run on one
activation identifies the *local cone* (active set of J-lens vectors) that
best approximates that point. This module wraps each such result into a
stable, provenance-complete artifact record (:func:`make_cone_record`,
schema ``jlens.cones.record.v1``) and provides utilities to compare records
across layers and positions.

**A cone signature is deterministic bookkeeping, not a concept claim.**
The signature is a function of the record's *effective* active set (tokens
with strictly positive coefficients): sorted unique token ids plus a
SHA-256 digest. Two records with the same signature selected the same token
set — that does not establish that they realize one universal semantic
concept, and nothing in this module asserts otherwise.

**Aggregation is transparent and separate from decomposition.**
:func:`recurring_signatures` counts exact signature recurrences;
:func:`group_by_signature` groups by exact match;
:func:`group_by_similarity` is a deliberately simple greedy grouping by
weighted active-set similarity with an explicit threshold. No opaque
clustering method is used at this stage — cluster analyses, if any, come
later and build on these records.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Sequence

CONE_RECORD_SCHEMA = "jlens.cones.record.v1"
TRANSITION_SCHEMA = "jlens.cones.transition.v1"


def cone_signature(token_ids: Iterable[int]) -> dict:
    """Deterministic signature of an active set.

    Order- and multiplicity-invariant: the signature is computed from the
    sorted set of unique token ids. Returns ``{"token_ids": [...],
    "digest": "sha256:<16 hex>"}``.
    """
    ids = sorted({int(i) for i in token_ids})
    digest = hashlib.sha256(
        json.dumps(ids, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    return {"token_ids": ids, "digest": f"sha256:{digest}"}


def make_cone_record(
    pursuit_record: dict,
    *,
    decoded_labels: Sequence[str],
    layer: int,
    position: int,
    input_token_id: int | None,
    input_token: str | None,
    prompt_hash: str,
    prompt_slug: str | None,
    prompt_format: str,
    run_provenance: dict,
) -> dict:
    """Wrap one :meth:`jlens.pursuit.PursuitResult.to_records` item into the
    stable cone-record artifact schema.

    Args:
        pursuit_record: One record from ``PursuitResult.to_records()``.
        decoded_labels: Decoded token strings aligned with the record's
            ``token_ids`` (pass what the tokenizer produced; not re-derived
            here so records stay tokenizer-agnostic).
        layer: Source layer of the decomposed activation.
        position: Token position (Python indexing convention of the run).
        input_token_id / input_token: The token *at* the decomposed
            position (``None`` if not applicable).
        prompt_hash: Stable prompt hash (``jlens.metadata.prompt_hashes``
            convention); the full prompt text is never stored.
        prompt_slug: Optional safe identifier (e.g. eval-set slug).
        prompt_format: ``"plain"`` or ``"chat"``.
        run_provenance: Run-level facts: run id/dir, lens artifact
            fingerprint, model revision, commits, settings. Stored verbatim.
    """
    if pursuit_record.get("schema") != "jlens.pursuit.result.v1":
        raise ValueError(
            f"expected a jlens.pursuit.result.v1 record, got "
            f"{pursuit_record.get('schema')!r}"
        )
    token_ids = list(pursuit_record["token_ids"])
    coefficients = list(pursuit_record["coefficients"])
    if len(decoded_labels) != len(token_ids):
        raise ValueError(
            f"decoded_labels has {len(decoded_labels)} entries for "
            f"{len(token_ids)} selected tokens"
        )
    effective = [
        (i, label, c)
        for i, label, c in zip(token_ids, decoded_labels, coefficients, strict=True)
        if c > 0
    ]
    return {
        "schema": CONE_RECORD_SCHEMA,
        "run_provenance": dict(run_provenance),
        "prompt_hash": prompt_hash,
        "prompt_slug": prompt_slug,
        "format": prompt_format,
        "layer": int(layer),
        "position": int(position),
        "input_token_id": None if input_token_id is None else int(input_token_id),
        "input_token": input_token,
        "requested_k": pursuit_record["requested_k"],
        "n_selected": pursuit_record["n_selected"],
        "selected_token_ids": token_ids,
        "selected_labels": list(decoded_labels),
        "coefficients": coefficients,
        "effective_token_ids": [i for i, _, _ in effective],
        "effective_labels": [label for _, label, _ in effective],
        "effective_coefficients": [c for _, _, c in effective],
        "reconstruction": {
            "target_norm": pursuit_record["target_norm"],
            "residual_norm": pursuit_record["residual_norm"],
            "relative_residual": pursuit_record["relative_residual"],
            "explained_fraction": pursuit_record["explained_fraction"],
        },
        "stopping": {
            "stop_reason": pursuit_record["stop_reason"],
            "n_iterations": pursuit_record["n_iterations"],
            "residual_norm_history": pursuit_record["residual_norm_history"],
        },
        "algorithm_settings": pursuit_record["settings"],
        "dictionary_provenance": pursuit_record["dictionary_provenance"],
        "cone_signature": cone_signature(i for i, _, _ in effective),
    }


def save_cone_records(records: Sequence[dict], path: str) -> None:
    """Write cone records as pretty JSON (schema-checked)."""
    for record in records:
        if record.get("schema") != CONE_RECORD_SCHEMA:
            raise ValueError(f"not a cone record: schema={record.get('schema')!r}")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(list(records), handle, indent=2, ensure_ascii=False)


def load_cone_records(path: str) -> list[dict]:
    """Load records written by :func:`save_cone_records` (schema-checked)."""
    with open(path, encoding="utf-8") as handle:
        records = json.load(handle)
    for record in records:
        if record.get("schema") != CONE_RECORD_SCHEMA:
            raise ValueError(
                f"{path}: unexpected record schema {record.get('schema')!r}"
            )
    return records


# ------------------------------------------------------------ comparison


def active_set_overlap(ids_a: Iterable[int], ids_b: Iterable[int]) -> dict:
    """Set overlap between two active sets: intersection size and Jaccard
    index (1.0 for two empty sets — identical, if vacuously)."""
    set_a, set_b = {int(i) for i in ids_a}, {int(i) for i in ids_b}
    union = set_a | set_b
    intersection = set_a & set_b
    return {
        "intersection_size": len(intersection),
        "union_size": len(union),
        "jaccard": (len(intersection) / len(union)) if union else 1.0,
    }


def weighted_active_set_similarity(record_a: dict, record_b: dict) -> float:
    """Cosine similarity of the two records' coefficient vectors on the
    union of their effective token ids (0.0 if either is empty)."""
    coeffs_a = dict(
        zip(record_a["effective_token_ids"], record_a["effective_coefficients"], strict=True)
    )
    coeffs_b = dict(
        zip(record_b["effective_token_ids"], record_b["effective_coefficients"], strict=True)
    )
    union = sorted(set(coeffs_a) | set(coeffs_b))
    if not union:
        return 0.0
    dot = sum(coeffs_a.get(i, 0.0) * coeffs_b.get(i, 0.0) for i in union)
    norm_a = sum(c * c for c in coeffs_a.values()) ** 0.5
    norm_b = sum(c * c for c in coeffs_b.values()) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def coefficient_concentration(coefficients: Sequence[float]) -> dict:
    """Concentration of a nonnegative coefficient vector: Herfindahl index
    (sum of squared shares; 1.0 = one dominant atom, 1/n = uniform) and the
    top-1 share."""
    total = float(sum(coefficients))
    if total <= 0 or not len(coefficients):
        return {"herfindahl": 0.0, "top1_share": 0.0, "n_nonzero": 0}
    shares = [c / total for c in coefficients if c > 0]
    return {
        "herfindahl": sum(s * s for s in shares),
        "top1_share": max(shares),
        "n_nonzero": len(shares),
    }


def cone_trajectory(records: Sequence[dict]) -> list[dict]:
    """Layer-by-layer transitions for records of the *same* prompt/position.

    Records are sorted by layer; consecutive pairs yield one transition
    record (schema ``jlens.cones.transition.v1``) with active-set overlap,
    weighted similarity, reconstruction-quality change, concentration
    change, and concept entry/exit token lists. Raises if the records mix
    prompts or positions — trajectories are per-(prompt, position) by
    definition.
    """
    if not records:
        return []
    keys = {(r["prompt_hash"], r["position"], r["format"]) for r in records}
    if len(keys) > 1:
        raise ValueError(
            f"cone_trajectory needs records from one (prompt, position); got {keys}"
        )
    ordered = sorted(records, key=lambda r: r["layer"])
    transitions = []
    for prev, curr in zip(ordered, ordered[1:], strict=False):
        ids_prev = set(prev["effective_token_ids"])
        ids_curr = set(curr["effective_token_ids"])
        labels_curr = dict(
            zip(curr["effective_token_ids"], curr["effective_labels"], strict=True)
        )
        labels_prev = dict(
            zip(prev["effective_token_ids"], prev["effective_labels"], strict=True)
        )
        transitions.append(
            {
                "schema": TRANSITION_SCHEMA,
                "prompt_hash": prev["prompt_hash"],
                "prompt_slug": prev.get("prompt_slug"),
                "format": prev["format"],
                "position": prev["position"],
                "layer_from": prev["layer"],
                "layer_to": curr["layer"],
                "active_set_overlap": active_set_overlap(ids_prev, ids_curr),
                "weighted_similarity": weighted_active_set_similarity(prev, curr),
                "explained_fraction_from": prev["reconstruction"]["explained_fraction"],
                "explained_fraction_to": curr["reconstruction"]["explained_fraction"],
                "delta_explained_fraction": (
                    curr["reconstruction"]["explained_fraction"]
                    - prev["reconstruction"]["explained_fraction"]
                ),
                "concentration_from": coefficient_concentration(
                    prev["effective_coefficients"]
                ),
                "concentration_to": coefficient_concentration(
                    curr["effective_coefficients"]
                ),
                "entered_token_ids": sorted(ids_curr - ids_prev),
                "entered_labels": [
                    labels_curr[i] for i in sorted(ids_curr - ids_prev)
                ],
                "exited_token_ids": sorted(ids_prev - ids_curr),
                "exited_labels": [
                    labels_prev[i] for i in sorted(ids_prev - ids_curr)
                ],
            }
        )
    return transitions


def recurring_signatures(records: Sequence[dict]) -> list[dict]:
    """Frequency table of exact cone-signature recurrences across records,
    most frequent first (ties: lexicographic digest). Recurrence counts are
    descriptive only — recurrence does not by itself establish a shared
    concept."""
    counts: Counter[str] = Counter()
    examples: dict[str, dict] = {}
    for record in records:
        digest = record["cone_signature"]["digest"]
        counts[digest] += 1
        examples.setdefault(
            digest,
            {
                "token_ids": record["cone_signature"]["token_ids"],
                "labels": record["effective_labels"],
                "first_seen": {
                    "prompt_slug": record.get("prompt_slug"),
                    "layer": record["layer"],
                    "position": record["position"],
                },
            },
        )
    return [
        {"digest": digest, "count": count, **examples[digest]}
        for digest, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    ]


def group_by_signature(records: Sequence[dict]) -> dict[str, list[int]]:
    """Exact grouping: signature digest -> indices into ``records``."""
    groups: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        groups.setdefault(record["cone_signature"]["digest"], []).append(index)
    return groups


def group_by_similarity(
    records: Sequence[dict], *, threshold: float
) -> list[list[int]]:
    """Transparent greedy grouping by weighted active-set similarity.

    Records are visited in input order; each joins the first existing group
    whose *representative* (the group's first record) has weighted
    similarity >= ``threshold``, else starts a new group. Deterministic,
    order-dependent, and intentionally simple — this is a grouping aid for
    inspection, not a clustering result.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    groups: list[list[int]] = []
    for index, record in enumerate(records):
        for group in groups:
            if (
                weighted_active_set_similarity(records[group[0]], record)
                >= threshold
            ):
                group.append(index)
                break
        else:
            groups.append([index])
    return groups
