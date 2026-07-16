# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Similarity-based recurrence and atom-frequency statistics for cone records.

The exact cone signature (:func:`jlens.cones.cone_signature`) is a SHA-256
digest over the sorted positive-coefficient token ids. It is kept for
provenance, but exact set equality is a brittle recurrence criterion: on the
completed Gemma 4 E4B run no ten-token signature ever repeated, which says
nothing about whether *similar* sparse structure recurs. This module provides
transparent, deterministic similarity measures at four explicitly separated
levels:

1. **Exact signatures** — set equality (the legacy criterion, preserved).
2. **Similar supports** — set-based: Jaccard similarity, top-m overlap.
3. **Coefficient-similar cones** — magnitude-aware: weighted Jaccard and
   cosine similarity over sparse nonnegative coefficient maps.
4. **Recurring atoms** — individual-token selection frequencies and
   count-based enrichment, independent of whole-set matching.

Design constraints (all deliberate): no embedding models, no external
services, no model downloads; deterministic ordering and tie-breaking
everywhere; grouping avoids all-pairs comparison via a shared-atom inverted
index (every metric here is exactly 0 for records with no shared atom).

**Similarity groups are not concept clusters.** :func:`similarity_groups`
returns connected components under "similarity >= threshold". A component
proves only that a chain of pairwise-similar records exists — it does not
establish that its members realize one semantic concept, and
:func:`threshold_sensitivity` exists precisely so no single cutoff has to be
trusted.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence

#: Strata keys supported by the grouped statistics below. ``requested_k`` is
#: the pursuit sparsity budget; the rest are capture facts on every record.
DEFAULT_STRATA: tuple[str, ...] = ("layer", "requested_k", "format", "position")


# --------------------------------------------------------------- sparse maps


def coefficient_map(record: Mapping) -> dict[int, float]:
    """The record's effective (strictly positive) coefficients keyed by
    token id."""
    return {
        int(i): float(c)
        for i, c in zip(
            record["effective_token_ids"],
            record["effective_coefficients"],
            strict=True,
        )
    }


def top_m_atoms(record: Mapping, m: int) -> list[int]:
    """The ``m`` largest-coefficient token ids of a record, ordered by
    descending coefficient with ties broken toward the lowest token id."""
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}")
    ranked = sorted(coefficient_map(record).items(), key=lambda kv: (-kv[1], kv[0]))
    return [token_id for token_id, _ in ranked[:m]]


# ------------------------------------------------------------------- metrics


def jaccard_similarity(ids_a: Iterable[int], ids_b: Iterable[int]) -> float:
    """Set Jaccard index. Two empty sets are identical (1.0), matching
    :func:`jlens.cones.active_set_overlap`."""
    set_a, set_b = {int(i) for i in ids_a}, {int(i) for i in ids_b}
    union = set_a | set_b
    return (len(set_a & set_b) / len(union)) if union else 1.0


def weighted_jaccard(
    map_a: Mapping[int, float], map_b: Mapping[int, float]
) -> float:
    """Weighted Jaccard ``sum(min)/sum(max)`` over two nonnegative sparse
    coefficient maps (1.0 for two empty maps, matching
    :func:`jaccard_similarity`; 0.0 if only one is empty)."""
    for coeffs in (map_a, map_b):
        if any(c < 0 for c in coeffs.values()):
            raise ValueError("weighted_jaccard requires nonnegative coefficients")
    if not map_a and not map_b:
        return 1.0
    keys = set(map_a) | set(map_b)
    num = sum(min(map_a.get(k, 0.0), map_b.get(k, 0.0)) for k in keys)
    den = sum(max(map_a.get(k, 0.0), map_b.get(k, 0.0)) for k in keys)
    return num / den if den > 0 else 0.0


def sparse_cosine(map_a: Mapping[int, float], map_b: Mapping[int, float]) -> float:
    """Cosine similarity between two sparse coefficient maps (0.0 if either
    has zero norm)."""
    norm_a = math.sqrt(sum(c * c for c in map_a.values()))
    norm_b = math.sqrt(sum(c * c for c in map_b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    dot = sum(c * map_b.get(k, 0.0) for k, c in map_a.items())
    return dot / (norm_a * norm_b)


def top_m_overlap(record_a: Mapping, record_b: Mapping, m: int) -> float:
    """Fraction of the top-``m`` atoms shared between two records:
    ``|top_m(a) & top_m(b)| / m``. Deterministic through the tie-breaking of
    :func:`top_m_atoms`."""
    top_a, top_b = set(top_m_atoms(record_a, m)), set(top_m_atoms(record_b, m))
    return len(top_a & top_b) / m


def reweighted_map(
    coeffs: Mapping[int, float], atom_weights: Mapping[int, float]
) -> dict[int, float]:
    """Multiply a coefficient map by per-atom weights (e.g.
    :func:`inverse_frequency_weights`) for frequency-adjusted similarity.
    Atoms without a weight keep weight 1.0."""
    return {k: c * float(atom_weights.get(k, 1.0)) for k, c in coeffs.items()}


_METRICS = ("exact", "jaccard", "weighted_jaccard", "cosine", "top_m")


def record_similarity(
    record_a: Mapping,
    record_b: Mapping,
    *,
    metric: str = "weighted_jaccard",
    m: int = 5,
    atom_weights: Mapping[int, float] | None = None,
) -> float:
    """Similarity between two cone records under a named metric.

    ``metric``: one of ``exact`` (1.0 iff the effective sets are equal),
    ``jaccard``, ``weighted_jaccard``, ``cosine``, ``top_m``. When
    ``atom_weights`` is given, the coefficient-based metrics
    (``weighted_jaccard``, ``cosine``) operate on the reweighted maps; the
    raw metrics stay available by simply not passing weights.
    """
    if metric not in _METRICS:
        raise ValueError(f"unknown metric {metric!r}; expected one of {_METRICS}")
    if metric == "exact":
        ids_a = set(record_a["effective_token_ids"])
        ids_b = set(record_b["effective_token_ids"])
        return 1.0 if ids_a == ids_b else 0.0
    if metric == "jaccard":
        return jaccard_similarity(
            record_a["effective_token_ids"], record_b["effective_token_ids"]
        )
    if metric == "top_m":
        return top_m_overlap(record_a, record_b, m)
    map_a, map_b = coefficient_map(record_a), coefficient_map(record_b)
    if atom_weights is not None:
        map_a = reweighted_map(map_a, atom_weights)
        map_b = reweighted_map(map_b, atom_weights)
    if metric == "weighted_jaccard":
        return weighted_jaccard(map_a, map_b)
    return sparse_cosine(map_a, map_b)


# ------------------------------------------------------------------ grouping


def stratum_of(record: Mapping, strata: Sequence[str]) -> tuple:
    """The record's stratum key (values of the requested fields, in order)."""
    return tuple(record[field] for field in strata)


def similarity_groups(
    records: Sequence[Mapping],
    *,
    metric: str = "weighted_jaccard",
    threshold: float,
    strata: Sequence[str] = DEFAULT_STRATA,
    m: int = 5,
    atom_weights: Mapping[int, float] | None = None,
) -> list[dict]:
    """Connected components of "similarity >= threshold" within each stratum.

    Comparison is restricted to records inside one stratum (by default
    layer, k, format, and position — cross-layer comparison must be asked
    for explicitly by passing ``strata`` without ``"layer"`` and labeling
    the result as such). Candidate pairs are generated from a shared-atom
    inverted index, so records with disjoint supports (similarity exactly 0
    under every metric here) are never compared; a full quadratic sweep only
    happens in the degenerate case where all records share an atom.

    Returns one dict per group, sorted by (stratum, -size, first index):
    ``{"stratum": {...}, "record_indices": [...], "size": int}``. Indices
    refer to positions in ``records``. Components are **similarity groups**,
    not validated concept clusters.
    """
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")
    if metric not in _METRICS:
        raise ValueError(f"unknown metric {metric!r}; expected one of {_METRICS}")

    by_stratum: dict[tuple, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_stratum[stratum_of(record, strata)].append(index)

    parent: dict[int, int] = {}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            # Deterministic: lower index becomes the root.
            lo, hi = min(ra, rb), max(ra, rb)
            parent[hi] = lo

    for indices in by_stratum.values():
        for index in indices:
            parent[index] = index
        atom_index: dict[int, list[int]] = defaultdict(list)
        for index in indices:
            for token_id in set(records[index]["effective_token_ids"]):
                atom_index[token_id].append(index)
        seen_pairs: set[tuple[int, int]] = set()
        for token_id in sorted(atom_index):
            bucket = atom_index[token_id]
            for i, a in enumerate(bucket):
                for b in bucket[i + 1 :]:
                    pair = (a, b) if a < b else (b, a)
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    similarity = record_similarity(
                        records[pair[0]],
                        records[pair[1]],
                        metric=metric,
                        m=m,
                        atom_weights=atom_weights,
                    )
                    if similarity >= threshold:
                        union(*pair)

    groups: dict[int, list[int]] = defaultdict(list)
    for index in parent:
        groups[find(index)].append(index)
    strata_fields = list(strata)
    payload = [
        {
            "stratum": dict(
                zip(strata_fields, stratum_of(records[members[0]], strata), strict=True)
            ),
            "record_indices": sorted(members),
            "size": len(members),
        }
        for members in groups.values()
    ]
    payload.sort(
        key=lambda g: (
            tuple(str(v) for v in g["stratum"].values()),
            -g["size"],
            g["record_indices"][0],
        )
    )
    return payload


def threshold_sensitivity(
    records: Sequence[Mapping],
    *,
    metric: str = "weighted_jaccard",
    thresholds: Sequence[float],
    strata: Sequence[str] = DEFAULT_STRATA,
    m: int = 5,
    atom_weights: Mapping[int, float] | None = None,
) -> list[dict]:
    """Group statistics across a threshold grid, so conclusions never rest
    on one arbitrary cutoff.

    Returns, per threshold (ascending): number of groups, number of
    non-singleton groups, records in non-singleton groups, and the largest
    group size.
    """
    rows = []
    for threshold in sorted(thresholds):
        groups = similarity_groups(
            records,
            metric=metric,
            threshold=threshold,
            strata=strata,
            m=m,
            atom_weights=atom_weights,
        )
        non_singleton = [g for g in groups if g["size"] > 1]
        rows.append(
            {
                "metric": metric,
                "threshold": threshold,
                "n_groups": len(groups),
                "n_non_singleton_groups": len(non_singleton),
                "n_records_in_non_singleton_groups": sum(
                    g["size"] for g in non_singleton
                ),
                "max_group_size": max((g["size"] for g in groups), default=0),
            }
        )
    return rows


# ------------------------------------------------------------ atom frequency


def atom_selection_frequencies(
    records: Sequence[Mapping],
    *,
    strata: Sequence[str] = DEFAULT_STRATA,
) -> dict:
    """Selection counts per atom, overall and per stratum.

    Returns ``{"n_records", "overall": Counter, "labels": {id: str},
    "by_stratum": {stratum_tuple: Counter}, "strata": [...],
    "record_counts_by_stratum": {stratum_tuple: int}}``. An atom is counted
    once per record it appears in (selection presence, not coefficient
    mass).
    """
    overall: Counter[int] = Counter()
    by_stratum: dict[tuple, Counter[int]] = defaultdict(Counter)
    record_counts: Counter[tuple] = Counter()
    labels: dict[int, str] = {}
    for record in records:
        key = stratum_of(record, strata)
        record_counts[key] += 1
        for token_id, label in zip(
            record["effective_token_ids"], record["effective_labels"], strict=True
        ):
            token_id = int(token_id)
            overall[token_id] += 1
            by_stratum[key][token_id] += 1
            labels.setdefault(token_id, label)
    return {
        "n_records": len(records),
        "strata": list(strata),
        "overall": overall,
        "labels": labels,
        "by_stratum": dict(by_stratum),
        "record_counts_by_stratum": dict(record_counts),
    }


def inverse_frequency_weights(
    frequencies: Mapping[int, int], n_records: int
) -> dict[int, float]:
    """Smoothed inverse-document-frequency weights over atom selection
    counts: ``log(1 + n_records / (1 + count))``.

    The add-one in the denominator keeps unseen/rare atoms finite and the
    ``1 +`` inside the log keeps every weight strictly positive; both are
    standard IDF smoothing and are the only smoothing applied.
    """
    if n_records < 1:
        raise ValueError(f"n_records must be >= 1, got {n_records}")
    return {
        int(token_id): math.log(1.0 + n_records / (1.0 + count))
        for token_id, count in frequencies.items()
    }


def atom_enrichment(
    frequencies: dict,
    *,
    smoothing: float = 0.5,
) -> list[dict]:
    """Observed-versus-expected enrichment of each atom in each stratum.

    For atom ``v`` in stratum ``s``: observed = selections of ``v`` in
    ``s``; expected = overall selections of ``v`` scaled by the stratum's
    share of records; plus a smoothed log-odds of per-record presence in
    stratum versus out of stratum (Haldane–Anscombe correction, adding
    ``smoothing`` — default 0.5 — to every cell; the only smoothing used).

    Input is the output of :func:`atom_selection_frequencies`. Rows are
    sorted by (stratum, -observed, token_id). Enrichment describes selection
    statistics only — a frequent or enriched atom is not thereby a concept.
    """
    if smoothing <= 0:
        raise ValueError(f"smoothing must be > 0, got {smoothing}")
    n_records = frequencies["n_records"]
    overall = frequencies["overall"]
    rows = []
    for stratum_key in sorted(
        frequencies["by_stratum"], key=lambda key: tuple(str(v) for v in key)
    ):
        counts = frequencies["by_stratum"][stratum_key]
        n_in = frequencies["record_counts_by_stratum"][stratum_key]
        n_out = n_records - n_in
        for token_id in sorted(counts, key=lambda t: (-counts[t], t)):
            observed = counts[token_id]
            expected = overall[token_id] * (n_in / n_records)
            outside = overall[token_id] - observed
            log_odds = math.log(
                (observed + smoothing) / (n_in - observed + smoothing)
            ) - math.log(
                (outside + smoothing) / (max(n_out - outside, 0) + smoothing)
            )
            rows.append(
                {
                    "stratum": dict(
                        zip(frequencies["strata"], stratum_key, strict=True)
                    ),
                    "token_id": token_id,
                    "label": frequencies["labels"].get(token_id, ""),
                    "observed": observed,
                    "expected": expected,
                    "observed_over_expected": (
                        observed / expected if expected > 0 else math.inf
                    ),
                    "log_odds": log_odds,
                    "stratum_records": n_in,
                }
            )
    return rows


def output_token_recurrence(records: Sequence[Mapping]) -> list[dict]:
    """Recurrence of atoms conditioned on being the record's model top-1
    output token.

    For each atom that is ever selected *as* the record's own model top-1
    output token (``run_provenance.model_top1_id``), report how often that
    happens versus how often the atom is selected in records where it is
    not the output token. Sorted by (-as_output_count, token_id).
    """
    as_output: Counter[int] = Counter()
    elsewhere: Counter[int] = Counter()
    labels: dict[int, str] = {}
    for record in records:
        top1 = record.get("run_provenance", {}).get("model_top1_id")
        for token_id, label in zip(
            record["effective_token_ids"], record["effective_labels"], strict=True
        ):
            token_id = int(token_id)
            labels.setdefault(token_id, label)
            if top1 is not None and token_id == int(top1):
                as_output[token_id] += 1
            else:
                elsewhere[token_id] += 1
    return [
        {
            "token_id": token_id,
            "label": labels[token_id],
            "as_output_token": as_output[token_id],
            "as_non_output_atom": elsewhere[token_id],
        }
        for token_id in sorted(as_output, key=lambda t: (-as_output[t], t))
    ]
