# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Corpus records, deterministic splits, and the leakage audit.

Three jobs, in order:

1. **Normalize and identify.** Every record gets a stable ID derived from the
   corpus and its streaming index, plus a sha256 over normalized text. The ID
   is what splits are computed from, so the split is reproducible from record
   identity alone and does not drift if the stream is re-read.
2. **Split.** Assignment is a hash bucket, not a position, so the fit,
   development and confirmation partitions are independent by construction and
   a record's partition never depends on how many records were drawn.
3. **Audit.** Exact duplicates are caught by normalized checksum; near
   duplicates by a banded 64-bit SimHash. A cross-partition hit **raises**.
   Silently dropping the offending record would quietly change the split.

Nested scale subsets are produced here too: within the fit partition, records
are ordered by a second stable hash, so the first 100 are always a subset of
the first 250 and 1,000. That is what makes scale the only variable that changes
between scale points — and, because the estimator is a running mean, what makes
the three scale points cost as much as the largest alone.

Nothing here consults a lens, a layer, or any model output. Selection by J-lens
performance is impossible in this module because no entry point accepts one.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field

from jlens.mmpilot.store import payload_checksum

#: Tag written into every split manifest. Changing the split maths must change
#: this string, so an old manifest can never be mistaken for a new one.
SPLIT_PROTOCOL = "stable-hash-bucket-v1"

#: Default seed for bucket assignment. Frozen in
#: ``configs/research_grade_jlens_calibration_v1.json``.
SPLIT_SEED = 20260805

#: Buckets per partition. Fit is deliberately large: the study needs up to
#: 1,000 fit prompts but only 128 each for development and confirmation.
N_BUCKETS = 100
FIT_BUCKETS = (0, 79)
VALIDATION_BUCKETS = (80, 89)
CONFIRMATION_BUCKETS = (90, 99)

PARTITIONS = ("fit", "validation", "confirmation")

#: SimHash band layout. Four 16-bit bands over a 64-bit signature: two records
#: within Hamming distance 3 must agree exactly on at least one band, so
#: banding is a sound filter (no false negatives) rather than an approximation.
SIMHASH_BITS = 64
SIMHASH_BANDS = 4
SIMHASH_BAND_BITS = SIMHASH_BITS // SIMHASH_BANDS
MAX_HAMMING_DISTANCE = 3

#: Word n-gram width the SimHash is computed over.
SHINGLE_WIDTH = 3

_WHITESPACE = re.compile(r"\s+")


class CorpusLeakageError(RuntimeError):
    """A record (or a near-duplicate of one) appears in two partitions.

    Raised rather than repaired. Dropping the offending record would silently
    change the split that every downstream checksum is bound to.
    """


# --------------------------------------------------------------- normalization


def normalize_text(text: str) -> str:
    """NFKC, lowercased, whitespace-collapsed, stripped.

    This is the form duplicate detection works on. It is deliberately *not*
    the form the model sees: fitting uses the raw record text.
    """
    normalized = unicodedata.normalize("NFKC", str(text))
    return _WHITESPACE.sub(" ", normalized).strip().lower()


def normalized_sha256(text: str) -> str:
    """``sha256:<hex>`` over :func:`normalize_text` of ``text``."""
    return "sha256:" + hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def record_id_for(corpus_id: str, stream_index: int) -> str:
    """Stable record identity: ``{corpus_id}/{stream_index}``.

    The streaming index is part of the identity because HuggingFace streaming
    yields a dataset split in a fixed order for a fixed revision. The corpus
    revision is bound separately, in the run fingerprint.
    """
    return f"{corpus_id}/{int(stream_index)}"


def _shingles(normalized: str) -> list[str]:
    """Word ``SHINGLE_WIDTH``-grams; short texts fall back to the whole string."""
    words = normalized.split()
    if len(words) < SHINGLE_WIDTH:
        return [normalized] if normalized else []
    return [
        " ".join(words[index : index + SHINGLE_WIDTH])
        for index in range(len(words) - SHINGLE_WIDTH + 1)
    ]


def simhash64(text: str) -> int:
    """64-bit SimHash of ``text`` over word 3-grams.

    Deterministic across processes and platforms: the per-shingle hash is
    sha256, not Python's salted :func:`hash`.
    """
    normalized = normalize_text(text)
    shingles = _shingles(normalized)
    if not shingles:
        return 0
    accumulator = [0] * SIMHASH_BITS
    for shingle in shingles:
        digest = hashlib.sha256(shingle.encode("utf-8")).digest()[:8]
        value = int.from_bytes(digest, "big")
        for bit in range(SIMHASH_BITS):
            accumulator[bit] += 1 if (value >> bit) & 1 else -1
    signature = 0
    for bit in range(SIMHASH_BITS):
        if accumulator[bit] > 0:
            signature |= 1 << bit
    return signature


def hamming_distance(left: int, right: int) -> int:
    """Population count of ``left ^ right``."""
    return int(left ^ right).bit_count()


def _bands(signature: int) -> tuple[int, ...]:
    """The four 16-bit bands of ``signature``, tagged by position."""
    mask = (1 << SIMHASH_BAND_BITS) - 1
    return tuple(
        (index << SIMHASH_BAND_BITS) | ((signature >> (index * SIMHASH_BAND_BITS)) & mask)
        for index in range(SIMHASH_BANDS)
    )


# -------------------------------------------------------------------- records


@dataclass(frozen=True)
class CorpusRecord:
    """One calibration text with everything the protocol binds it by.

    Attributes:
        record_id: Stable identity, ``{corpus_id}/{stream_index}``.
        text: Raw text, exactly as the model will see it.
        stream_index: Position in the streamed split.
        normalized_checksum: ``sha256:`` over normalized text.
        simhash: 64-bit near-duplicate signature.
    """

    record_id: str
    text: str
    stream_index: int
    normalized_checksum: str
    simhash: int

    @classmethod
    def build(cls, corpus_id: str, stream_index: int, text: str) -> CorpusRecord:
        return cls(
            record_id=record_id_for(corpus_id, stream_index),
            text=str(text),
            stream_index=int(stream_index),
            normalized_checksum=normalized_sha256(text),
            simhash=simhash64(text),
        )

    def to_dict(self) -> dict:
        """Serializable identity. **The text is deliberately absent** — corpus
        text is never committed; artifacts carry checksums."""
        return {
            "record_id": self.record_id,
            "stream_index": self.stream_index,
            "normalized_checksum": self.normalized_checksum,
            "simhash": f"{self.simhash:016x}",
        }


def build_records(
    corpus_id: str, texts: Iterable[str], *, min_chars: int = 600
) -> list[CorpusRecord]:
    """Records for every text of at least ``min_chars`` stripped characters.

    ``stream_index`` counts **every** text seen, kept or not, so a record's
    identity does not depend on the filter threshold.
    """
    records: list[CorpusRecord] = []
    for stream_index, text in enumerate(texts):
        if len(str(text).strip()) >= min_chars:
            records.append(CorpusRecord.build(corpus_id, stream_index, text))
    return records


def collect_records_for_partition_quotas(
    corpus_id: str,
    texts: Iterable[str],
    *,
    min_chars: int = 600,
    min_fit: int,
    n_validation: int,
    n_confirmation: int,
    max_texts: int,
    seed: int = SPLIT_SEED,
) -> tuple[list[CorpusRecord], Partitions]:
    """Consume a text stream until every deterministic split quota is met.

    Hash bucketing is not guaranteed to put an exact nominal sample count into
    each partition.  This helper therefore counts *unique* normalized records
    as it streams and stops only after fit, validation, and confirmation all
    satisfy their requested sizes.  ``max_texts`` keeps a corrupt or unsuitable
    stream from running forever; quotas are never silently reduced.
    """
    requirements = {
        "fit": int(min_fit),
        "validation": int(n_validation),
        "confirmation": int(n_confirmation),
    }
    if any(value < 0 for value in requirements.values()):
        raise ValueError(f"partition quotas must be non-negative: {requirements}")
    if max_texts <= 0:
        raise ValueError(f"max_texts must be positive, got {max_texts}")

    records: list[CorpusRecord] = []
    unique_checksums: set[str] = set()
    counts = {name: 0 for name in PARTITIONS}
    texts_seen = 0
    for stream_index, text in enumerate(texts):
        if texts_seen >= max_texts:
            break
        texts_seen += 1
        if len(str(text).strip()) < min_chars:
            continue
        record = CorpusRecord.build(corpus_id, stream_index, text)
        records.append(record)
        if record.normalized_checksum in unique_checksums:
            continue
        unique_checksums.add(record.normalized_checksum)
        bucket = assign_bucket(record.record_id, seed=seed)
        name = _partition_for_bucket(
            bucket,
            fit=FIT_BUCKETS,
            validation=VALIDATION_BUCKETS,
            confirmation=CONFIRMATION_BUCKETS,
        )
        if name is not None:
            counts[name] += 1
        if all(counts[name] >= required for name, required in requirements.items()):
            return records, build_partitions(
                records,
                corpus_id=corpus_id,
                seed=seed,
                n_validation=n_validation,
                n_confirmation=n_confirmation,
            )

    raise ValueError(
        "corpus stream ended or reached the bounded collection limit before "
        f"partition quotas were met: observed={counts}, required={requirements}, "
        f"texts_seen={texts_seen}, max_texts={max_texts}. Stream more corpus "
        "records; do not shrink the held-out partitions"
    )


# --------------------------------------------------------------------- splits


def assign_bucket(record_id: str, *, seed: int = SPLIT_SEED, n_buckets: int = N_BUCKETS) -> int:
    """Deterministic bucket in ``[0, n_buckets)`` for ``record_id``."""
    digest = hashlib.sha256(f"{seed}|{record_id}".encode()).hexdigest()
    return int(digest[:8], 16) % int(n_buckets)


def _partition_for_bucket(
    bucket: int,
    *,
    fit: tuple[int, int],
    validation: tuple[int, int],
    confirmation: tuple[int, int],
) -> str | None:
    for name, (low, high) in (
        ("fit", fit),
        ("validation", validation),
        ("confirmation", confirmation),
    ):
        if low <= bucket <= high:
            return name
    return None


def nested_order_key(record_id: str, *, seed: int = SPLIT_SEED) -> str:
    """Sort key giving the fit partition its nested scale ordering.

    A *second* hash, distinct from the bucket hash, so ordering within the fit
    partition is independent of which bucket a record landed in.
    """
    return hashlib.sha256(f"{seed}|nested|{record_id}".encode()).hexdigest()


@dataclass(frozen=True)
class Partitions:
    """Independent fit / development / confirmation record sets.

    ``fit`` is stored in nested scale order: ``fit[:100]`` is the first scale
    point and is a strict prefix of ``fit[:250]`` and ``fit[:1000]``.
    """

    fit: tuple[CorpusRecord, ...]
    validation: tuple[CorpusRecord, ...]
    confirmation: tuple[CorpusRecord, ...]
    corpus_id: str
    split_seed: int
    protocol: str = SPLIT_PROTOCOL
    dropped_exact_duplicates: tuple[str, ...] = field(default_factory=tuple)

    def get(self, name: str) -> tuple[CorpusRecord, ...]:
        if name not in PARTITIONS:
            raise ValueError(f"unknown partition {name!r}; expected {PARTITIONS}")
        return getattr(self, name)

    def checksum(self, name: str) -> str:
        """Checksum over the ordered record identities of one partition."""
        return payload_checksum([record.to_dict() for record in self.get(name)])

    def manifest(self) -> dict:
        payload = {
            "protocol": self.protocol,
            "corpus_id": self.corpus_id,
            "split_seed": int(self.split_seed),
            "sizes": {name: len(self.get(name)) for name in PARTITIONS},
            "checksums": {name: self.checksum(name) for name in PARTITIONS},
            "n_dropped_exact_duplicates": len(self.dropped_exact_duplicates),
            "nested_order": "sha256(seed|nested|record_id), ascending",
        }
        payload["manifest_checksum"] = payload_checksum(payload)
        return payload


def build_partitions(
    records: Sequence[CorpusRecord],
    *,
    corpus_id: str,
    seed: int = SPLIT_SEED,
    n_buckets: int = N_BUCKETS,
    fit_buckets: tuple[int, int] = FIT_BUCKETS,
    validation_buckets: tuple[int, int] = VALIDATION_BUCKETS,
    confirmation_buckets: tuple[int, int] = CONFIRMATION_BUCKETS,
    n_validation: int | None = None,
    n_confirmation: int | None = None,
) -> Partitions:
    """Split ``records`` into three independent partitions.

    Exact duplicates (identical normalized checksum) are collapsed to their
    first occurrence *before* bucketing, so a duplicated text cannot straddle
    two partitions by accident. Which occurrence survives is deterministic:
    the lowest stream index.

    Args:
        records: Corpus records, any order.
        corpus_id: Recorded in the manifest and bound into the fingerprint.
        seed: Bucket-assignment seed.
        n_validation / n_confirmation: Truncate those partitions to exactly
            this many records, in nested order. ``None`` keeps everything.

    Raises:
        ValueError: If the requested development or confirmation size exceeds
            what the corpus supplies. A short partition is not a weaker run;
            it is a different protocol.
    """
    by_checksum: dict[str, CorpusRecord] = {}
    dropped: list[str] = []
    for record in sorted(records, key=lambda item: (item.stream_index, item.record_id)):
        existing = by_checksum.get(record.normalized_checksum)
        if existing is None:
            by_checksum[record.normalized_checksum] = record
        else:
            dropped.append(record.record_id)

    buckets: dict[str, list[CorpusRecord]] = {name: [] for name in PARTITIONS}
    for record in by_checksum.values():
        bucket = assign_bucket(record.record_id, seed=seed, n_buckets=n_buckets)
        name = _partition_for_bucket(
            bucket,
            fit=fit_buckets,
            validation=validation_buckets,
            confirmation=confirmation_buckets,
        )
        if name is not None:
            buckets[name].append(record)

    ordered = {
        name: sorted(
            items, key=lambda record: (nested_order_key(record.record_id, seed=seed),
                                       record.record_id)
        )
        for name, items in buckets.items()
    }

    for name, requested in (("validation", n_validation), ("confirmation", n_confirmation)):
        if requested is not None:
            available = len(ordered[name])
            if available < requested:
                raise ValueError(
                    f"the {name} partition holds {available} records but the protocol "
                    f"requires exactly {requested}; stream more corpus records rather "
                    f"than shrinking the partition"
                )
            ordered[name] = ordered[name][:requested]

    return Partitions(
        fit=tuple(ordered["fit"]),
        validation=tuple(ordered["validation"]),
        confirmation=tuple(ordered["confirmation"]),
        corpus_id=corpus_id,
        split_seed=int(seed),
        dropped_exact_duplicates=tuple(dropped),
    )


def nested_subset(records: Sequence[CorpusRecord], n: int) -> tuple[CorpusRecord, ...]:
    """The first ``n`` records — the nested scale subset.

    Raises:
        ValueError: If fewer than ``n`` records are available. Quietly
            returning a shorter list would make a named scale snapshot that is
            secretly a 4,317k scale point.
    """
    if n < 0:
        raise ValueError(f"scale point must be non-negative, got {n}")
    if len(records) < n:
        raise ValueError(
            f"scale point {n} needs {n} fit records but only {len(records)} are "
            f"available; stream more corpus records"
        )
    return tuple(records[:n])


# ---------------------------------------------------------------------- audit


def _candidate_pairs(
    left: Sequence[CorpusRecord], right: Sequence[CorpusRecord]
) -> Iterator[tuple[CorpusRecord, CorpusRecord]]:
    """Cross-partition pairs sharing at least one SimHash band.

    Banding is exact for this threshold, not heuristic: 3 differing bits
    cannot touch all 4 bands, so any pair within Hamming distance 3 is
    guaranteed to collide on some band and be offered here.
    """
    index: dict[int, list[CorpusRecord]] = {}
    for record in right:
        for band in _bands(record.simhash):
            index.setdefault(band, []).append(record)
    for record in left:
        seen: set[str] = set()
        for band in _bands(record.simhash):
            for other in index.get(band, ()):
                if other.record_id not in seen:
                    seen.add(other.record_id)
                    yield record, other


def audit_leakage(
    partitions: Partitions,
    *,
    max_hamming: int = MAX_HAMMING_DISTANCE,
    scale_points: Sequence[int] = (),
) -> dict:
    """Refuse any exact or near duplicate that crosses a partition boundary.

    Only the fit prefix actually used is audited when ``scale_points`` is
    given, so the audit covers exactly the records the study will consume.

    Raises:
        CorpusLeakageError: On the first offending pair, naming both records.
    """
    largest = max(scale_points) if scale_points else len(partitions.fit)
    fit = partitions.fit[:largest]
    sets = {
        "fit": fit,
        "validation": partitions.validation,
        "confirmation": partitions.confirmation,
    }

    exact_hits: list[dict] = []
    for left_name, right_name in (
        ("fit", "validation"),
        ("fit", "confirmation"),
        ("validation", "confirmation"),
    ):
        right_by_checksum = {
            record.normalized_checksum: record for record in sets[right_name]
        }
        for record in sets[left_name]:
            other = right_by_checksum.get(record.normalized_checksum)
            if other is not None:
                exact_hits.append(
                    {
                        "kind": "exact",
                        "left_partition": left_name,
                        "right_partition": right_name,
                        "left_record_id": record.record_id,
                        "right_record_id": other.record_id,
                        "checksum": record.normalized_checksum,
                    }
                )

    near_hits: list[dict] = []
    n_compared = 0
    for left_name, right_name in (
        ("fit", "validation"),
        ("fit", "confirmation"),
        ("validation", "confirmation"),
    ):
        for record, other in _candidate_pairs(sets[left_name], sets[right_name]):
            n_compared += 1
            distance = hamming_distance(record.simhash, other.simhash)
            if distance <= max_hamming:
                near_hits.append(
                    {
                        "kind": "near",
                        "left_partition": left_name,
                        "right_partition": right_name,
                        "left_record_id": record.record_id,
                        "right_record_id": other.record_id,
                        "hamming_distance": distance,
                    }
                )

    report = {
        "protocol": partitions.protocol,
        "max_hamming_distance": int(max_hamming),
        "audited_fit_records": len(fit),
        "candidate_pairs_compared": n_compared,
        "n_exact_hits": len(exact_hits),
        "n_near_hits": len(near_hits),
        "hits": exact_hits + near_hits,
        "ok": not exact_hits and not near_hits,
    }
    report["audit_checksum"] = payload_checksum(report)
    if not report["ok"]:
        first = report["hits"][0]
        raise CorpusLeakageError(
            f"{len(report['hits'])} record(s) cross a partition boundary; the first is "
            f"a {first['kind']} duplicate between {first['left_partition']} "
            f"({first['left_record_id']}) and {first['right_partition']} "
            f"({first['right_record_id']}). Refusing: dropping it here would change "
            "the split every downstream checksum is bound to."
        )
    return report


def scale_nesting_audit(fit: Sequence[CorpusRecord], scale_points: Sequence[int]) -> dict:
    """Confirm each scale point is a strict prefix of the next.

    Cheap, and it catches the one failure mode that would silently invalidate
    the whole scale comparison: an ordering that is not stable between points.
    """
    points = sorted(int(point) for point in scale_points)
    subsets = {point: nested_subset(fit, point) for point in points}
    nested = True
    details: list[dict] = []
    for smaller, larger in zip(points, points[1:], strict=False):
        prefix_ok = subsets[larger][: len(subsets[smaller])] == subsets[smaller]
        nested = nested and prefix_ok
        details.append(
            {"smaller": smaller, "larger": larger, "is_prefix": bool(prefix_ok)}
        )
    payload = {
        "scale_points": points,
        "nested": bool(nested),
        "pairs": details,
        "checksums": {
            str(point): payload_checksum([r.to_dict() for r in subsets[point]])
            for point in points
        },
    }
    payload["audit_checksum"] = payload_checksum(payload)
    return payload


def corpus_manifest(
    partitions: Partitions,
    *,
    corpus_config: Mapping,
    scale_points: Sequence[int],
) -> dict:
    """The corpus block written into every artifact."""
    payload = {
        "corpus_id": partitions.corpus_id,
        "hf_dataset": corpus_config.get("hf_dataset"),
        "config": corpus_config.get("config"),
        "split": corpus_config.get("split"),
        "revision": corpus_config.get("revision"),
        "revision_status": corpus_config.get("revision_status"),
        "min_chars": corpus_config.get("min_chars"),
        "license": corpus_config.get("license"),
        "modality": "text-only",
        "spokencoco_used": False,
        "multimodal_data_used": False,
        "splits": partitions.manifest(),
        "scale_points": [int(point) for point in scale_points],
    }
    payload["corpus_manifest_checksum"] = payload_checksum(payload)
    return payload
