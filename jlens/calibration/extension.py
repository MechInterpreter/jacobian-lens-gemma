# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""The early-layer extension: continue one accumulator, replace every held-out set.

The completed scale-100 calibration validated layers 35, 38 and 40 and failed at
32 and 26. Layer 32 failed closely enough to make a larger calibration worth
running, and the completed multimodal work has since established controlled
three-modality transfer at 35/38/40 and shown those layers already converged onto
the clean answer. The next scientific step is therefore a *validated earlier*
readout, at layer 32 primarily and layer 26 if it earns it.

**One thing is reused and one thing is destroyed.**

*Reused:* the scale-100 fitting accumulator. ``J_l`` is a running mean over a
deterministically ordered prompt list, and :func:`jlens.fitting.fit` reads no
validation or confirmation result while accumulating it. There is no code path
by which a held-out number could have entered it, so continuing it to 250 and
1,000 prompts is exactly a longer fit — not a fit informed by its own evaluation.
The continuation is bit-identical to a fresh nested fit over the same ordering,
because the additions happen in the same sequence; a test asserts that rather
than assuming it.

*Destroyed:* the old confirmation set. It has been opened, its verdict has been
read, and that verdict is the reason this extension exists. A set whose result
already influenced the decision to run a larger scale cannot be that scale's
untouched endpoint. It is not reused, not relabelled, and not reset — it is
**excluded** from every new split, and the exclusion is counted and checksummed.
The old development set goes with it, for the same reason.

So this module builds: a frozen protocol (:data:`EXTENSION_PROTOCOL`), a gate
that is the calibration gate at a larger sample with a *stricter* diversity floor
(:data:`EXTENSION_GATE`), completely fresh development and confirmation sets
(:func:`build_fresh_evaluation_splits`), a deterministic continuation
(:func:`build_extension_fit_order`, :func:`verify_fit_prefix`,
:func:`seed_extension_checkpoint`), a predeclared scale choice
(:func:`select_extension_scale`), and a publication path that refuses anything
that has not passed fresh confirmation (:func:`publish_early_layer`).

Text-only by construction, exactly as :mod:`jlens.calibration` is: nothing here
can reach SpokenCOCO, an image, an audio waveform, or a cross-modal mapping.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from jlens.calibration.corpus import (
    MAX_HAMMING_DISTANCE,
    CorpusRecord,
    _bands,
    hamming_distance,
)
from jlens.calibration.gate import (
    CALIBRATION_GATE,
    CalibrationGate,
    audit_target_diversity,
)
from jlens.calibration.parent import ParentAccumulator, ParentRun
from jlens.calibration.plan import CapturePlan, normalized_depth
from jlens.calibration.publication import (
    ConfirmationVault,
    PublicationRefused,
    publish_layer,
)
from jlens.calibration.state import CALIBRATION_STAGES, CalibrationStore
from jlens.metadata import file_sha256
from jlens.mmpilot.store import payload_checksum

__all__ = [
    "BASELINE_SCALE",
    "CONFIRMATION_PROMPT_SEED",
    "DEVELOPMENT_PROMPT_SEED",
    "DEVELOPMENT_SCALE_POINTS",
    "EARLY_LAYERS_OF_INTEREST",
    "EARLY_LAYER_GO",
    "EARLY_LAYER_NO_GO",
    "EXTENSION_FIT_ORDER_PROTOCOL",
    "EXTENSION_GATE",
    "EXTENSION_PROTOCOL",
    "EXTENSION_PROTOCOL_VERSION",
    "EXTENSION_SCALE_POINTS",
    "EXTENSION_SELECTION_RULE",
    "EXTENSION_SPLIT_PROTOCOL",
    "EXTENSION_SPLIT_SEED",
    "EXTENSION_STAGES",
    "N_CONFIRMATION_PROMPTS",
    "N_DEVELOPMENT_PROMPTS",
    "PRIMARY_EARLY_LAYER",
    "PUBLISHABLE_LAYERS",
    "SECONDARY_EARLY_LAYER",
    "ContinuationRefused",
    "ExtensionProtocol",
    "ExtensionSelectionRule",
    "ExtensionSplitRefused",
    "ExtensionStore",
    "FreshEvaluationSplits",
    "EXTENSION_CONFIRMATION_GATE",
    "audit_extension_target_diversity",
    "audit_fresh_split_leakage",
    "build_extension_fit_order",
    "build_fresh_evaluation_splits",
    "early_layer_verdict",
    "extension_budget",
    "extension_corpus_manifest",
    "extension_gate_text",
    "format_extension_budget",
    "parent_collection_parameters",
    "publish_early_layer",
    "seed_extension_checkpoint",
    "select_extension_scale",
    "verify_continuation_equals_fresh_fit",
    "verify_fit_prefix",
    "verify_reconstructed_partitions",
]

# ------------------------------------------------------------ frozen identity

EXTENSION_PROTOCOL_VERSION = "research-grade-early-layer-jlens-extension-v1"

#: How the extension's fit list is ordered. Named so that a lens fitted under a
#: different ordering can never be mistaken for one fitted under this.
EXTENSION_FIT_ORDER_PROTOCOL = "parent-prefix-pinned-nested-order-v1"

#: How the fresh evaluation sets are drawn. Distinct from the parent's
#: ``stable-hash-bucket-v1``: a different seed, a different bucket layout, a
#: two-way split, and a mandatory exclusion pass.
EXTENSION_SPLIT_PROTOCOL = "extension-fresh-eval-hash-bucket-v1"

#: The scale points this extension may reach. 100 is the parent's and is
#: **descriptive only** — it cannot be newly confirmed, because the set that
#: would confirm it has been spent.
EXTENSION_SCALE_POINTS = (250, 1_000)
BASELINE_SCALE = 100
MAXIMUM_AUTHORIZED_SCALE = 1_000

#: Scales the **development** set is scored at. The baseline is included so the
#: plateau rule has three points and can say something — with two points its
#: continuation clause has no previous step and the verdict is
#: ``PLATEAU_REACHED`` by construction, which would be a satisfied condition
#: rather than a measured one. Scoring the parent's scale-100 lens on the *fresh*
#: development set is descriptive and read-only; it is not, and cannot become, a
#: new confirmation of scale 100.
DEVELOPMENT_SCALE_POINTS = (BASELINE_SCALE, *EXTENSION_SCALE_POINTS)

PRIMARY_EARLY_LAYER = 32
SECONDARY_EARLY_LAYER = 26
EARLY_LAYERS_OF_INTEREST = (SECONDARY_EARLY_LAYER, PRIMARY_EARLY_LAYER)

#: Layers whose readout this extension may publish. 35, 38 and 40 already have
#: published, confirmed lenses; this extension does not republish or replace
#: them, and :func:`publish_early_layer` refuses to. Layers 8, 14 and 20 are
#: reported continuously but are descriptive unless independently confirmed.
PUBLISHABLE_LAYERS = EARLY_LAYERS_OF_INTEREST
ESTABLISHED_LAYERS = (35, 38, 40)
DESCRIPTIVE_LAYERS = (8, 14, 20)

#: Fresh seeds. Different numbers from the parent's, so a record's partition and
#: a prompt's selection order are both new.
EXTENSION_SPLIT_SEED = 20260807
DEVELOPMENT_PROMPT_SEED = 20260808
CONFIRMATION_PROMPT_SEED = 20260809

N_DEVELOPMENT_PROMPTS = 256
N_CONFIRMATION_PROMPTS = 256

EXTENSION_N_BUCKETS = 100
DEVELOPMENT_BUCKETS = (0, 49)
CONFIRMATION_BUCKETS = (50, 99)

#: Stages the extension writes, on top of the calibration stages. A separate
#: tuple rather than a widening of :data:`jlens.calibration.state.CALIBRATION_STAGES`,
#: so the completed study's stage vocabulary is untouched.
EXTENSION_ONLY_STAGES = (
    "parent_import",
    "fresh_splits",
    "continuation",
    "scale_selection",
    "early_layer_verdict",
)
EXTENSION_STAGES = (*CALIBRATION_STAGES, *EXTENSION_ONLY_STAGES)

EARLY_LAYER_GO = "EARLY_LAYER_CALIBRATION_GO"
EARLY_LAYER_NO_GO = "EARLY_LAYER_CALIBRATION_NO_GO"

EXTENSION_ARTIFACT_SCHEMA = "jlens.calibration.early_layer_artifact.v1"


class ExtensionSplitRefused(RuntimeError):
    """A fresh evaluation set could not be built to the frozen specification.

    Raised rather than satisfied by relaxation. Shrinking a held-out set or
    lowering a diversity floor because the corpus was inconvenient would make
    the extension's endpoint weaker than the endpoint it replaces.
    """


class ContinuationRefused(RuntimeError):
    """The continuation cannot be proved to equal a fresh nested fit."""


# ------------------------------------------------------------------ the gate

#: The validity gate, unchanged in every threshold that decides a layer, at the
#: larger evaluation sample and with a **stricter** target-diversity floor:
#: 32 distinct targets instead of 24. Nothing is loosened. The share ceiling,
#: the tie ceiling, the control ratios and margins, the rank and top-k clauses
#: and the fold rule are byte-identical to
#: :data:`jlens.calibration.gate.CALIBRATION_GATE`; a test asserts field by
#: field that the only differences are ``n_prompts``,
#: ``min_distinct_target_tokens`` and ``version``.
EXTENSION_GATE = CalibrationGate(
    **{
        **asdict(CALIBRATION_GATE),
        "n_prompts": N_DEVELOPMENT_PROMPTS,
        "min_distinct_target_tokens": 32,
        "version": (
            "research-grade-early-layer-extension-tie-aware-native-readout-v1"
        ),
    }
)

#: The same gate at the confirmation sample size. Identical rule, untouched data.
EXTENSION_CONFIRMATION_GATE = CalibrationGate(
    **{**asdict(EXTENSION_GATE), "n_prompts": N_CONFIRMATION_PROMPTS}
)


def extension_gate_text(gate: CalibrationGate = EXTENSION_GATE) -> str:
    """The criterion block, printed before any extension number is computed."""
    base = CALIBRATION_GATE
    return f"""\
PREDECLARED EARLY-LAYER EXTENSION GATE — {gate.version}
Digest: {gate.digest}
Frozen before any extension result exists. Not revisable after seeing one.

This is the completed study's calibration gate, carried over threshold for
threshold, at a larger held-out sample and with a STRICTER diversity floor:

  sample size            {base.n_prompts} -> {gate.n_prompts} prompts (development and confirmation)
  distinct targets       >= {base.min_distinct_target_tokens} -> >= {gate.min_distinct_target_tokens}   (STRICTER)
  max single-target share  <= {gate.max_target_token_share:.0%}   (unchanged)
  tied-at-maximum rate     <= {gate.max_tied_at_max_rate:.2f}   (unchanged)
  noise-control MRR        >= {gate.min_noise_control_mrr_ratio:.1f}x AND >= +{gate.min_noise_control_mrr_margin:.2f}   (unchanged)
  wrong-layer margin       >= +{gate.min_wrong_layer_mrr_margin:.2f} via {gate.wrong_layer_mapping}   (unchanged)
  median midrank           <= {gate.max_median_midrank:.1f}   (unchanged)
  top-{gate.top_k} inclusion         >= {gate.min_top_k_inclusion:.2f} at the PESSIMISTIC rank   (unchanged)
  fold stability           {gate.n_folds} fixed folds, each beating every control and
                           reaching >= {gate.min_fold_mrr_fraction:.2f}x the overall MRR   (unchanged)

NOTHING IS LOOSENED to let layer 32 or layer 26 pass. The one changed decision
threshold is harder, and it is frozen here, before any extension result exists,
because the larger sample makes a 24-token floor easier to clear by accident.

Continuous metrics are reported for EVERY layer at EVERY scale regardless of
pass or fail. Passing on development is not publication: publication requires
this same gate on the fresh, untouched confirmation set, after the scale is
chosen without it.
"""


# --------------------------------------------------------------- the protocol


@dataclass(frozen=True)
class ExtensionProtocol:
    """What this extension is, frozen before any of its results exist.

    Every statement here is load-bearing and every one is checksummed into the
    run fingerprint, so editing one invalidates stored results rather than
    re-deciding on them.
    """

    version: str = EXTENSION_PROTOCOL_VERSION
    motivated_by: str = (
        "the completed scale-100 confirmation result: L35/L38/L40 passed "
        "untouched confirmation, L32 and L26 failed, and L32 failed closely "
        "enough to motivate a larger calibration"
    )
    old_confirmation_status: str = (
        "development history. The old confirmation set has been opened and its "
        "result motivated this extension, so it is never reused, relabelled or "
        "reset; it is excluded from every new split."
    )
    old_development_status: str = (
        "development history. Excluded from every new split for the same reason."
    )
    reused_parent_state: str = (
        "the scale-100 fitting accumulator (jacobian_sum, n_done) and nothing "
        "else. The fitting loop reads no held-out result, so no evaluation "
        "number could have influenced it."
    )
    candidate_scales: tuple[int, ...] = EXTENSION_SCALE_POINTS
    baseline_scale: int = BASELINE_SCALE
    baseline_scale_status: str = (
        "descriptive baseline only. Scale 100 cannot be newly confirmed: the "
        "set that would confirm it has been spent."
    )
    primary_layer: int = PRIMARY_EARLY_LAYER
    secondary_layer: int = SECONDARY_EARLY_LAYER
    descriptive_layers: tuple[int, ...] = DESCRIPTIVE_LAYERS
    descriptive_layer_status: str = (
        "L20 and shallower are reported continuously and are descriptive unless "
        "they independently pass the same gate on the fresh confirmation set."
    )
    established_layers: tuple[int, ...] = ESTABLISHED_LAYERS
    established_layer_status: str = (
        "the existing L35/L38/L40 publications remain unchanged; this extension "
        "neither overwrites nor republishes them."
    )
    thresholds_frozen_before_results: bool = True
    multimodal_data_in_fitting: bool = False
    multimodal_data_in_lens_validation: bool = False
    maximum_authorized_scale: int = MAXIMUM_AUTHORIZED_SCALE
    fit_order_protocol: str = EXTENSION_FIT_ORDER_PROTOCOL
    split_protocol: str = EXTENSION_SPLIT_PROTOCOL
    n_development_prompts: int = N_DEVELOPMENT_PROMPTS
    n_confirmation_prompts: int = N_CONFIRMATION_PROMPTS

    def to_dict(self) -> dict:
        payload = asdict(self)
        for key in (
            "candidate_scales",
            "descriptive_layers",
            "established_layers",
        ):
            payload[key] = list(payload[key])
        return payload

    @property
    def digest(self) -> str:
        return payload_checksum(self.to_dict())

    def text(self) -> str:
        return f"""\
EARLY-LAYER EXTENSION PROTOCOL — {self.version}
Digest: {self.digest}
Frozen before any extension result exists.

  1. WHY THIS EXISTS
     {self.motivated_by}.

  2. THE OLD CONFIRMATION SET IS SPENT
     {self.old_confirmation_status}
     The old development set: {self.old_development_status}

  3. WHAT CROSSES THE BOUNDARY
     {self.reused_parent_state}

  4. SCALES
     candidate extension scales {list(self.candidate_scales)}; maximum
     authorized {self.maximum_authorized_scale:,}. Scale {self.baseline_scale}:
     {self.baseline_scale_status}

  5. LAYERS
     primary earlier layer      L{self.primary_layer}
     secondary / stretch        L{self.secondary_layer}
     descriptive only           {list(self.descriptive_layers)} — {self.descriptive_layer_status}
     already published          {list(self.established_layers)} — {self.established_layer_status}

  6. THRESHOLDS
     frozen before extension results: {self.thresholds_frozen_before_results}.
     The validation gate is the completed study's gate, unchanged except for a
     larger sample and a stricter target-diversity floor.

  7. MODALITY
     multimodal data in fitting:          {self.multimodal_data_in_fitting}
     multimodal data in lens validation:  {self.multimodal_data_in_lens_validation}
     Calibration and validation are text-only, on the pinned WikiText corpus.
"""


EXTENSION_PROTOCOL = ExtensionProtocol()


# ------------------------------------------------------------------ the store


class ExtensionStore(CalibrationStore):
    """A calibration store that also accepts the extension's own stages.

    Subclassed rather than widening :data:`CALIBRATION_STAGES`, so the completed
    study's stage vocabulary — and the tests that pin it — are untouched.
    """

    def stage_dir(self, stage: str) -> Path:
        if stage not in EXTENSION_STAGES:
            raise ValueError(
                f"unknown extension stage {stage!r}; known stages are "
                f"{EXTENSION_STAGES}"
            )
        return self.root / "units" / stage

    def status_report(self, stages: Sequence[str] = EXTENSION_STAGES) -> dict:
        return super().status_report(stages)

    def snapshot_path(self, scale: int) -> Path:
        return self.root / "artifacts" / f"lens.ext.scale{int(scale)}.pt"

    def published_path(self, layer: int, scale: int) -> Path:
        return (
            self.root
            / "artifacts"
            / "published"
            / f"lens.early.layer{int(layer)}.scale{int(scale)}.validated.pt"
        )


# ----------------------------------------------------- fresh evaluation sets


def _extension_bucket(record_id: str, *, seed: int, n_buckets: int) -> int:
    """Bucket under the extension's own seed and tag.

    The ``|ext|`` tag guarantees this is a genuinely different assignment from
    the parent's, not merely the same function at a different seed.
    """
    digest = hashlib.sha256(f"{seed}|ext|{record_id}".encode()).hexdigest()
    return int(digest[:8], 16) % int(n_buckets)


def _extension_order_key(record_id: str, *, seed: int) -> str:
    return hashlib.sha256(f"{seed}|ext-order|{record_id}".encode()).hexdigest()


class _NearDuplicateIndex:
    """Banded SimHash index over a set of records to exclude.

    Uses :mod:`jlens.calibration.corpus`'s own band layout, so the exactness
    argument carries over unchanged: three differing bits cannot touch all four
    16-bit bands, so any pair within Hamming distance 3 is guaranteed to be
    offered for comparison. Sharing the layout is deliberate — a second
    implementation would be a second thing to keep in step.
    """

    def __init__(self, records: Sequence[tuple[str, CorpusRecord]]) -> None:
        self._by_band: dict[int, list[tuple[str, CorpusRecord]]] = {}
        for source, record in records:
            for band in _bands(record.simhash):
                self._by_band.setdefault(band, []).append((source, record))
        self.n_records = len(records)
        self.n_comparisons = 0

    def hit(
        self, record: CorpusRecord, *, max_hamming: int
    ) -> tuple[str, CorpusRecord, int] | None:
        seen: set[str] = set()
        for band in _bands(record.simhash):
            for source, other in self._by_band.get(band, ()):
                if other.record_id in seen:
                    continue
                seen.add(other.record_id)
                self.n_comparisons += 1
                distance = hamming_distance(record.simhash, other.simhash)
                if distance <= max_hamming:
                    return source, other, distance
        return None


@dataclass(frozen=True)
class FreshEvaluationSplits:
    """Independent development and confirmation sets built after every exclusion.

    ``development`` and ``confirmation`` are drawn from disjoint hash buckets
    under a new seed, from a pool that already excludes every old fit,
    development and confirmation record and every new fit record through the
    largest candidate scale.
    """

    development: tuple[CorpusRecord, ...]
    confirmation: tuple[CorpusRecord, ...]
    corpus_id: str
    seed: int
    protocol: str = EXTENSION_SPLIT_PROTOCOL
    n_pool: int = 0
    excluded_exact: dict = field(default_factory=dict)
    excluded_near: dict = field(default_factory=dict)
    excluded_pool_duplicates: int = 0
    n_near_comparisons: int = 0

    def get(self, name: str) -> tuple[CorpusRecord, ...]:
        if name not in ("development", "confirmation"):
            raise ValueError(
                f"unknown extension partition {name!r}; expected "
                "'development' or 'confirmation'"
            )
        return getattr(self, name)

    def checksum(self, name: str) -> str:
        return payload_checksum([record.to_dict() for record in self.get(name)])

    def record_ids(self, name: str) -> list[str]:
        return [record.record_id for record in self.get(name)]

    def manifest(self) -> dict:
        payload = {
            "protocol": self.protocol,
            "corpus_id": self.corpus_id,
            "split_seed": int(self.seed),
            "sizes": {
                name: len(self.get(name)) for name in ("development", "confirmation")
            },
            "checksums": {
                name: self.checksum(name) for name in ("development", "confirmation")
            },
            "record_ids": {
                name: self.record_ids(name) for name in ("development", "confirmation")
            },
            "bucket_rule": (
                f"sha256(seed|ext|record_id) % {EXTENSION_N_BUCKETS}; development "
                f"{list(DEVELOPMENT_BUCKETS)}, confirmation "
                f"{list(CONFIRMATION_BUCKETS)}"
            ),
            "order_rule": "sha256(seed|ext-order|record_id), ascending",
            "pool_size_after_exclusions": int(self.n_pool),
            "excluded_exact": dict(self.excluded_exact),
            "excluded_near": dict(self.excluded_near),
            "excluded_pool_duplicates": int(self.excluded_pool_duplicates),
            "n_near_duplicate_comparisons": int(self.n_near_comparisons),
            "selected_by_jlens_performance": False,
            "old_sets_reused": False,
        }
        payload["manifest_checksum"] = payload_checksum(payload)
        return payload


def build_fresh_evaluation_splits(
    pool: Sequence[CorpusRecord],
    *,
    excluded: Mapping[str, Sequence[CorpusRecord]],
    corpus_id: str,
    seed: int = EXTENSION_SPLIT_SEED,
    n_development: int = N_DEVELOPMENT_PROMPTS,
    n_confirmation: int = N_CONFIRMATION_PROMPTS,
    n_buckets: int = EXTENSION_N_BUCKETS,
    development_buckets: tuple[int, int] = DEVELOPMENT_BUCKETS,
    confirmation_buckets: tuple[int, int] = CONFIRMATION_BUCKETS,
    max_hamming: int = MAX_HAMMING_DISTANCE,
) -> FreshEvaluationSplits:
    """Build the fresh development and confirmation sets.

    Order of operations, and each step is there for a reason:

    1. **Exclude exactly.** Any pool record whose normalized checksum matches an
       excluded record's is dropped and counted by which set it collided with.
    2. **Exclude near-duplicates.** Banded SimHash against the same excluded
       sets, at the study's standard Hamming distance.
    3. **Collapse duplicates inside the pool**, keeping the lowest stream index.
    4. **Bucket** under the extension's own seed into two disjoint ranges.
    5. **Order and truncate** to exactly the requested sizes.

    Args:
        pool: Candidate records. The caller supplies records the parent's
            collection never reached, so the exclusions below are a guard rather
            than the mechanism.
        excluded: ``{name: records}`` — the old fit, old development, old
            confirmation and new fit sets. Named so a refusal can say which set
            a record collided with.

    Raises:
        ExtensionSplitRefused: If either partition cannot be filled exactly. The
            sizes are never reduced to fit the corpus.
    """
    exact_index: dict[str, tuple[str, CorpusRecord]] = {}
    near_records: list[tuple[str, CorpusRecord]] = []
    for name, records in excluded.items():
        for record in records:
            exact_index.setdefault(record.normalized_checksum, (name, record))
            near_records.append((name, record))
    near_index = _NearDuplicateIndex(near_records)

    excluded_exact: dict[str, int] = {name: 0 for name in excluded}
    excluded_near: dict[str, int] = {name: 0 for name in excluded}
    pool_duplicates = 0

    kept: dict[str, CorpusRecord] = {}
    for record in sorted(pool, key=lambda item: (item.stream_index, item.record_id)):
        collision = exact_index.get(record.normalized_checksum)
        if collision is not None:
            excluded_exact[collision[0]] += 1
            continue
        near = near_index.hit(record, max_hamming=max_hamming)
        if near is not None:
            excluded_near[near[0]] += 1
            continue
        if record.normalized_checksum in kept:
            pool_duplicates += 1
            continue
        kept[record.normalized_checksum] = record

    buckets: dict[str, list[CorpusRecord]] = {"development": [], "confirmation": []}
    for record in kept.values():
        bucket = _extension_bucket(record.record_id, seed=seed, n_buckets=n_buckets)
        if development_buckets[0] <= bucket <= development_buckets[1]:
            buckets["development"].append(record)
        elif confirmation_buckets[0] <= bucket <= confirmation_buckets[1]:
            buckets["confirmation"].append(record)

    ordered = {
        name: sorted(
            items,
            key=lambda record: (
                _extension_order_key(record.record_id, seed=seed),
                record.record_id,
            ),
        )
        for name, items in buckets.items()
    }

    for name, requested in (
        ("development", n_development),
        ("confirmation", n_confirmation),
    ):
        available = len(ordered[name])
        if available < requested:
            raise ExtensionSplitRefused(
                f"the fresh {name} partition holds {available} records but the "
                f"frozen protocol requires exactly {requested}. Stream more "
                "corpus records; the held-out sizes are not reduced because "
                "coverage was inconvenient.\n"
                f"  pool after exclusions: {len(kept)}\n"
                f"  excluded as exact duplicates: {excluded_exact}\n"
                f"  excluded as near duplicates:  {excluded_near}"
            )
        ordered[name] = ordered[name][:requested]

    return FreshEvaluationSplits(
        development=tuple(ordered["development"]),
        confirmation=tuple(ordered["confirmation"]),
        corpus_id=str(corpus_id),
        seed=int(seed),
        n_pool=len(kept),
        excluded_exact=excluded_exact,
        excluded_near=excluded_near,
        excluded_pool_duplicates=pool_duplicates,
        n_near_comparisons=near_index.n_comparisons,
    )


def audit_fresh_split_leakage(
    splits: FreshEvaluationSplits,
    *,
    excluded: Mapping[str, Sequence[CorpusRecord]],
    max_hamming: int = MAX_HAMMING_DISTANCE,
) -> dict:
    """Refuse any record shared between the new sets, or with any old set.

    Run *after* construction, on the constructed sets, so it is an independent
    check rather than a restatement of the filter that built them.

    Raises:
        ExtensionSplitRefused: On the first offending pair, naming both records
            and which set the old one came from. Dropping it here would silently
            change the split every downstream checksum is bound to.
    """
    new_sets = {
        "development": splits.development,
        "confirmation": splits.confirmation,
    }
    hits: list[dict] = []
    comparisons = 0

    # new vs new
    confirmation_by_checksum = {
        record.normalized_checksum: record for record in splits.confirmation
    }
    for record in splits.development:
        other = confirmation_by_checksum.get(record.normalized_checksum)
        if other is not None:
            hits.append(
                {
                    "kind": "exact",
                    "left": "development",
                    "right": "confirmation",
                    "left_record_id": record.record_id,
                    "right_record_id": other.record_id,
                }
            )
    cross_index = _NearDuplicateIndex(
        [("confirmation", record) for record in splits.confirmation]
    )
    for record in splits.development:
        near = cross_index.hit(record, max_hamming=max_hamming)
        if near is not None:
            hits.append(
                {
                    "kind": "near",
                    "left": "development",
                    "right": "confirmation",
                    "left_record_id": record.record_id,
                    "right_record_id": near[1].record_id,
                    "hamming_distance": near[2],
                }
            )
    comparisons += cross_index.n_comparisons

    # new vs every old set
    old_index = _NearDuplicateIndex(
        [(name, record) for name, records in excluded.items() for record in records]
    )
    old_by_checksum = {
        record.normalized_checksum: (name, record)
        for name, records in excluded.items()
        for record in records
    }
    for new_name, records in new_sets.items():
        for record in records:
            collision = old_by_checksum.get(record.normalized_checksum)
            if collision is not None:
                hits.append(
                    {
                        "kind": "exact",
                        "left": new_name,
                        "right": collision[0],
                        "left_record_id": record.record_id,
                        "right_record_id": collision[1].record_id,
                    }
                )
            near = old_index.hit(record, max_hamming=max_hamming)
            if near is not None:
                hits.append(
                    {
                        "kind": "near",
                        "left": new_name,
                        "right": near[0],
                        "left_record_id": record.record_id,
                        "right_record_id": near[1].record_id,
                        "hamming_distance": near[2],
                    }
                )
    comparisons += old_index.n_comparisons

    report = {
        "protocol": splits.protocol,
        "max_hamming_distance": int(max_hamming),
        "audited_sets": {name: len(records) for name, records in new_sets.items()},
        "excluded_sets": {name: len(records) for name, records in excluded.items()},
        "candidate_pairs_compared": comparisons,
        "n_exact_hits": sum(1 for hit in hits if hit["kind"] == "exact"),
        "n_near_hits": sum(1 for hit in hits if hit["kind"] == "near"),
        "hits": hits,
        "ok": not hits,
    }
    report["audit_checksum"] = payload_checksum(report)
    if hits:
        first = hits[0]
        raise ExtensionSplitRefused(
            f"{len(hits)} record(s) leak across the extension's splits; the first "
            f"is a {first['kind']} duplicate between {first['left']} "
            f"({first['left_record_id']}) and {first['right']} "
            f"({first['right_record_id']}). Refusing. An old confirmation prompt "
            "in a new set would make the new confirmation set exactly as spent "
            "as the one it replaces."
        )
    return report


def audit_extension_target_diversity(
    target_token_ids: Sequence[int], *, gate: CalibrationGate = EXTENSION_GATE
) -> dict:
    """Diversity statistics under the extension's stricter floor.

    Thin wrapper over the calibration audit so the extension's floor is applied
    rather than the completed study's; the arithmetic is unchanged.
    """
    report = audit_target_diversity(target_token_ids, gate=gate)
    report["gate_version"] = gate.version
    report["gate_digest"] = gate.digest
    report["diversity_checksum"] = payload_checksum(
        {key: value for key, value in report.items() if key != "counts"}
    )
    return report


# ---------------------------------------------------------- the continuation


def build_extension_fit_order(
    parent_fit: Sequence[CorpusRecord],
    *,
    n_needed: int,
    extension_pool: Sequence[CorpusRecord] = (),
    fit_bucket_for: Callable[[CorpusRecord], bool] | None = None,
) -> list[CorpusRecord]:
    """The fit list the extension consumes, prefix-pinned to the parent's.

    The ordering is declared, not discovered:

    * positions ``0 .. len(parent_fit) - 1`` are the parent's own fit partition
      in the parent's own nested order, so **every prefix the parent fitted is a
      prefix of this list**;
    * any further positions come from records the parent's collection never
      reached, in ascending stream index — an ordering that cannot be perturbed
      by streaming further, which is the property the nested-hash order does not
      have once the record set grows.

    That second clause is why the ordering is defined here rather than taken
    from :func:`jlens.calibration.corpus.build_partitions`: re-running the
    parent's hash ordering over a *longer* stream can insert a late record ahead
    of an early one, which would silently break the nesting the continuation
    depends on.

    Raises:
        ContinuationRefused: If fewer than ``n_needed`` records are available.
    """
    ordered = list(parent_fit)
    seen = {record.record_id for record in ordered}
    if len(ordered) < int(n_needed):
        extra = [
            record
            for record in sorted(
                extension_pool, key=lambda item: (item.stream_index, item.record_id)
            )
            if record.record_id not in seen
            and (fit_bucket_for is None or fit_bucket_for(record))
        ]
        ordered.extend(extra)

    if len(ordered) < int(n_needed):
        raise ContinuationRefused(
            f"the extension fit ordering yields {len(ordered)} records but scale "
            f"{n_needed} needs {n_needed}. Stream more of the pinned corpus; the "
            "scale point is not reduced to fit the records on hand, because a "
            f"snapshot named scale{n_needed} that holds fewer prompts corrupts "
            "every comparison downstream."
        )
    return ordered[: int(n_needed)]


def parent_collection_parameters(parent: ParentRun) -> dict:
    """The exact arguments that reproduce the parent's corpus collection.

    Every value is read from the parent's own artifacts and its source is named,
    because a reconstruction driven by a guess is a reconstruction of something
    else. ``max_texts`` is the one value the parent did not record directly; it
    is derived from the collection formula the parent notebook used, and a wrong
    derivation cannot pass unnoticed because
    :func:`verify_reconstructed_partitions` checksums the result.

    Raises:
        ContinuationRefused: If a needed value is absent from the parent run.
    """
    corpus = parent.corpus
    splits = parent.splits
    sizes = parent.split_sizes
    scale_points = [int(point) for point in parent.fingerprint.get("scale_points", [])]

    missing = []
    if corpus.get("min_chars") is None:
        missing.append("corpus.min_chars")
    if splits.get("split_seed") is None:
        missing.append("corpus.splits.split_seed")
    for name in ("validation", "confirmation"):
        if sizes.get(name) is None:
            missing.append(f"corpus.splits.sizes.{name}")
    if not scale_points:
        missing.append("fingerprint.scale_points")
    if missing:
        raise ContinuationRefused(
            f"the parent run at {parent.root} does not record "
            f"{missing}; the collection that produced its splits cannot be "
            "reproduced, so its fit ordering cannot be continued. Recover the "
            "corpus unit from the parent run. Nothing is defaulted."
        )

    baseline = max(int(point) for point in scale_points)
    n_validation = int(sizes["validation"])
    n_confirmation = int(sizes["confirmation"])
    # The parent notebook's own bound: a hard safety limit, not a target count.
    max_texts = max(10_000, 8 * (baseline + n_validation + n_confirmation))
    return {
        "min_chars": int(corpus["min_chars"]),
        "seed": int(splits["split_seed"]),
        "min_fit": baseline,
        "n_validation": n_validation,
        "n_confirmation": n_confirmation,
        "max_texts": int(max_texts),
        "sources": {
            "min_chars": "parent corpus manifest",
            "seed": "parent corpus manifest splits.split_seed",
            "min_fit": "parent fingerprint scale_points (largest)",
            "n_validation": "parent corpus manifest splits.sizes.validation",
            "n_confirmation": "parent corpus manifest splits.sizes.confirmation",
            "max_texts": (
                "derived from the parent notebook's collection bound "
                "max(10000, 8 * (largest_scale + n_validation + n_confirmation)); "
                "verified by checksum, never trusted on its own"
            ),
        },
    }


def verify_reconstructed_partitions(partitions, *, parent: ParentRun) -> dict:
    """Prove a re-streamed collection reproduces the parent's exact splits.

    Compares the size and the ordered-identity checksum of every partition
    against what the parent recorded. This is what makes the parent's fit
    ordering — and therefore its fitted prompts — recoverable at all: the parent
    stored checksums, not texts.

    Raises:
        ContinuationRefused: On any disagreement, naming the partition.
    """
    expected_sizes = parent.split_sizes
    expected_checksums = parent.split_checksums
    rows: list[dict] = []
    for name in ("fit", "validation", "confirmation"):
        records = partitions.get(name)
        actual_checksum = partitions.checksum(name)
        rows.append(
            {
                "partition": name,
                "expected_size": expected_sizes.get(name),
                "actual_size": len(records),
                "expected_checksum": expected_checksums.get(name),
                "actual_checksum": actual_checksum,
                "matches": bool(
                    int(expected_sizes.get(name, -1)) == len(records)
                    and expected_checksums.get(name) == actual_checksum
                ),
            }
        )
    payload = {
        "partitions": rows,
        "all_match": all(row["matches"] for row in rows),
        "why": (
            "the parent stored split checksums rather than corpus text, so the "
            "only way to recover its fit ordering is to re-stream the pinned "
            "corpus under the parent's own collection parameters and prove the "
            "result reproduces those checksums"
        ),
    }
    payload["verification_checksum"] = payload_checksum(payload)
    if not payload["all_match"]:
        failed = [row for row in rows if not row["matches"]]
        raise ContinuationRefused(
            f"re-streaming the pinned corpus did not reproduce the parent's "
            f"splits: {failed}. The corpus revision, the record filter, the "
            "split seed or the collection bound differs from the parent's. "
            "Refusing to continue — the fit ordering this would produce is not "
            "the parent's."
        )
    return payload


def verify_fit_prefix(
    fit_records: Sequence[CorpusRecord],
    *,
    n_parent: int,
    parent_prefix_checksum: str,
) -> dict:
    """Prove the reconstructed prefix is the parent's fitted prompts exactly.

    The parent's scale-nesting audit recorded a checksum over the identities of
    the first ``n_parent`` fit records — record id, stream index, normalized text
    checksum and SimHash. Recomputing it here proves the reconstruction is the
    same prompts in the same order.

    Raises:
        ContinuationRefused: On any disagreement. Skipping the first
            ``n_parent`` prompts without this proof would mean continuing an
            accumulator over a prompt list that is not the one it was built
            from — a lens whose stated prompt count is a fiction.
    """
    prefix = list(fit_records[: int(n_parent)])
    recomputed = payload_checksum([record.to_dict() for record in prefix])
    matches = recomputed == str(parent_prefix_checksum)
    payload = {
        "n_parent_prompts": int(n_parent),
        "parent_prefix_checksum": str(parent_prefix_checksum),
        "reconstructed_prefix_checksum": recomputed,
        "matches": bool(matches),
        "first_record_id": prefix[0].record_id if prefix else None,
        "last_record_id": prefix[-1].record_id if prefix else None,
        "skip_authorized": bool(matches),
    }
    payload["verification_checksum"] = payload_checksum(payload)
    if not matches:
        raise ContinuationRefused(
            f"the reconstructed first {n_parent} fit records do not match the "
            f"parent's fit-prompt manifest (parent {parent_prefix_checksum}, "
            f"reconstructed {recomputed}). Refusing to skip them.\n"
            "The corpus stream, the record filter, the split seed or the fit "
            "ordering differs from the parent's. Continuing anyway would append "
            "150 or 900 prompts to an accumulator built from a different 100 and "
            "call the result a 250- or 1,000-prompt lens."
        )
    return payload


def seed_extension_checkpoint(
    accumulator: ParentAccumulator,
    destination: str | os.PathLike[str],
    *,
    torch_load: Callable | None = None,
) -> dict:
    """Copy the parent accumulator into the extension run, atomically.

    The parent file is opened for reading only and never renamed, truncated or
    written. The copy is checksum-verified against the parent's, and an existing
    copy — the resume case — is verified for compatibility instead of being
    overwritten.

    Raises:
        ContinuationRefused: If an existing extension checkpoint was fitted with
            a different layer grid, target layer or ``skip_first``, or has
            fallen *behind* the parent's prompt count.
    """
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.is_file():
        from jlens.calibration.parent import _load_accumulator  # noqa: PLC0415

        existing = _load_accumulator(target, torch_load=torch_load)
        incompatible = []
        if existing.source_layers != accumulator.source_layers:
            incompatible.append(
                f"source_layers {list(existing.source_layers)} != "
                f"{list(accumulator.source_layers)}"
            )
        if existing.target_layer != accumulator.target_layer:
            incompatible.append(
                f"target_layer {existing.target_layer} != {accumulator.target_layer}"
            )
        if existing.skip_first != accumulator.skip_first:
            incompatible.append(
                f"skip_first {existing.skip_first} != {accumulator.skip_first}"
            )
        if existing.n_done < accumulator.n_done:
            incompatible.append(
                f"n_done {existing.n_done} is behind the parent's "
                f"{accumulator.n_done}"
            )
        if incompatible:
            raise ContinuationRefused(
                f"the extension checkpoint at {target} is not a continuation of "
                f"the parent accumulator: {'; '.join(incompatible)}. Refusing to "
                "resume. Point the extension at a new run directory rather than "
                "mixing two accumulators."
            )
        return {
            "action": "resumed",
            "path": str(target),
            "checksum": file_sha256(str(target)),
            "n_done": existing.n_done,
            "next_idx": existing.next_idx,
            "parent_checkpoint_path": accumulator.path,
            "parent_checkpoint_checksum": accumulator.checksum,
            "parent_n_done": accumulator.n_done,
            "parent_written": False,
        }

    temporary = target.with_name(f"{target.name}.tmp.{os.getpid()}")
    shutil.copyfile(accumulator.path, temporary)
    os.replace(temporary, target)
    copied = file_sha256(str(target))
    if copied != accumulator.checksum:
        raise ContinuationRefused(
            f"the copy of the parent accumulator at {target} checksums to "
            f"{copied}, not the parent's {accumulator.checksum}. Refusing to "
            "continue a checkpoint that is not byte-identical to the one that "
            "was audited."
        )
    return {
        "action": "seeded",
        "path": str(target),
        "checksum": copied,
        "n_done": accumulator.n_done,
        "next_idx": accumulator.next_idx,
        "parent_checkpoint_path": accumulator.path,
        "parent_checkpoint_checksum": accumulator.checksum,
        "parent_n_done": accumulator.n_done,
        "parent_written": False,
    }


def verify_continuation_equals_fresh_fit(
    model,
    records: Sequence[CorpusRecord],
    *,
    plan: CapturePlan,
    scale: int,
    continued_checkpoint: str | os.PathLike[str],
    scratch_dir: str | os.PathLike[str],
) -> dict:
    """Fit ``scale`` prompts from scratch and compare, tensor by tensor.

    The claim being checked is exactness, not closeness: the continuation adds
    prompts ``n_parent .. scale-1`` to a sum that already holds ``0 .. n_parent-1``
    in that order, so the sequence of floating-point additions is identical to a
    fresh fit over the same ordering, and the accumulators must be bit-equal.

    Affordable only where a full refit is cheap — MOCK mode and the test suite.
    In a real run the same guarantee comes from the prefix checksum plus
    upstream's own resume path; this function is how that guarantee is *proved*,
    on a model where proving it costs seconds.

    Raises:
        ContinuationRefused: On any difference.
    """
    import torch  # noqa: PLC0415

    from jlens.fitting import fit  # noqa: PLC0415

    scratch = Path(scratch_dir)
    scratch.mkdir(parents=True, exist_ok=True)
    fresh_path = scratch / f"fresh_scale{int(scale)}.pt"
    if fresh_path.is_file():
        fresh_path.unlink()

    prompts = [record.text for record in records[: int(scale)]]
    fit(
        model,
        prompts,
        source_layers=list(plan.layers),
        target_layer=plan.target_layer,
        dim_batch=plan.dim_batch,
        max_seq_len=plan.max_seq_len,
        skip_first=plan.skip_first,
        checkpoint_path=str(fresh_path),
        checkpoint_every=None,
        resume=False,
    )
    fresh = torch.load(str(fresh_path), map_location="cpu", weights_only=True)
    continued = torch.load(
        str(continued_checkpoint), map_location="cpu", weights_only=True
    )

    differences: list[dict] = []
    if int(fresh["n_done"]) != int(continued["n_done"]):
        differences.append(
            {
                "layer": None,
                "reason": "n_done",
                "fresh": int(fresh["n_done"]),
                "continued": int(continued["n_done"]),
            }
        )
    for layer in sorted(fresh["jacobian_sum"]):
        left = fresh["jacobian_sum"][layer]
        right = continued["jacobian_sum"].get(layer)
        if right is None or not torch.equal(left, right):
            differences.append(
                {
                    "layer": int(layer),
                    "reason": "jacobian_sum",
                    "max_abs_difference": (
                        float((left - right).abs().max()) if right is not None else None
                    ),
                }
            )

    payload = {
        "scale": int(scale),
        "n_done_fresh": int(fresh["n_done"]),
        "n_done_continued": int(continued["n_done"]),
        "layers_compared": sorted(int(layer) for layer in fresh["jacobian_sum"]),
        "bit_identical": not differences,
        "differences": differences,
        "why_exact": (
            "the continuation replays the same additions in the same order, so "
            "the accumulators are bit-equal, not merely close"
        ),
    }
    payload["equivalence_checksum"] = payload_checksum(payload)
    if differences:
        raise ContinuationRefused(
            f"continuing the parent accumulator to {scale} prompts does NOT equal "
            f"a fresh nested fit at {scale}: {differences}. Refusing the "
            "continuation — the resulting lens would not be the lens its prompt "
            "count claims."
        )
    return payload


# --------------------------------------------------------- the corpus record


def extension_corpus_manifest(
    *,
    corpus_config: Mapping,
    corpus_id: str,
    fit_records: Sequence[CorpusRecord],
    splits: FreshEvaluationSplits,
    scale_points: Sequence[int],
    parent: ParentRun | None = None,
    fit_prefix_verification: Mapping | None = None,
) -> dict:
    """The corpus block written into every extension artifact.

    Shaped like :func:`jlens.calibration.corpus.corpus_manifest` — including
    ``splits.checksums`` with ``fit`` / ``validation`` / ``confirmation`` keys —
    so the existing artifact builder consumes it unchanged. ``validation`` is
    this extension's *development* set; the key name is kept because that is
    what the artifact schema calls it.
    """
    fit_checksums = {
        str(int(point)): payload_checksum(
            [record.to_dict() for record in fit_records[: int(point)]]
        )
        for point in sorted({int(point) for point in scale_points})
    }
    splits_manifest = splits.manifest()
    payload = {
        "corpus_id": str(corpus_id),
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
        "scale_points": [int(point) for point in scale_points],
        "fit_order_protocol": EXTENSION_FIT_ORDER_PROTOCOL,
        "fit_prefix_checksums": fit_checksums,
        "fit_prefix_verification": dict(fit_prefix_verification or {}),
        "fresh_splits": splits_manifest,
        "splits": {
            "protocol": splits.protocol,
            "corpus_id": str(corpus_id),
            "split_seed": int(splits.seed),
            "sizes": {
                "fit": len(fit_records),
                "validation": len(splits.development),
                "confirmation": len(splits.confirmation),
            },
            "checksums": {
                "fit": fit_checksums[str(max(int(p) for p in scale_points))],
                "validation": splits.checksum("development"),
                "confirmation": splits.checksum("confirmation"),
            },
            "nested_order": (
                "parent fit prefix pinned, then ascending stream index "
                f"({EXTENSION_FIT_ORDER_PROTOCOL})"
            ),
        },
        "parent_run": (
            {
                "root": parent.root,
                "fingerprint_digest": parent.fingerprint_digest,
                "accumulator_checksum": parent.accumulator.checksum,
                "baseline_scale": parent.accumulator.n_done,
                "old_split_checksums": parent.split_checksums,
                "old_sets_reused": False,
            }
            if parent is not None
            else None
        ),
    }
    payload["corpus_manifest_checksum"] = payload_checksum(payload)
    return payload


# ----------------------------------------------------------- scale selection


@dataclass(frozen=True)
class ExtensionSelectionRule:
    """Which scale the extension takes to confirmation, decided in advance.

    Attributes:
        require_plateau_reached: A smaller scale is only chosen when the
            predeclared plateau rule says improvement has stopped. Without this
            clause "the smaller set already matches" could be chosen while the
            metrics are still climbing.
        on_no_early_layer_development_pass: What to do when neither L32 nor L26
            passes development at any candidate scale. Predeclared as one honest
            confirmation at the largest scale, so a negative result is a
            measured negative rather than an abandoned run.
    """

    version: str = "early-layer-extension-scale-selection-v1"
    candidate_scales: tuple[int, ...] = EXTENSION_SCALE_POINTS
    earlier_layers: tuple[int, ...] = EARLY_LAYERS_OF_INTEREST
    require_plateau_reached: bool = True
    fallback_scale: int = MAXIMUM_AUTHORIZED_SCALE
    on_no_early_layer_development_pass: str = "confirm_once_at_largest_scale"
    declared_before_results: bool = True
    confirmation_may_be_consulted: bool = False
    multimodal_outcomes_may_be_consulted: bool = False

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["candidate_scales"] = list(self.candidate_scales)
        payload["earlier_layers"] = list(self.earlier_layers)
        return payload

    @property
    def digest(self) -> str:
        return payload_checksum(self.to_dict())

    def text(self) -> str:
        return f"""\
PREDECLARED SCALE-SELECTION RULE — {self.version}
Digest: {self.digest}
Fixed before any development number exists.

  0. The development set is scored at {list(DEVELOPMENT_SCALE_POINTS)}. Scale
     {BASELINE_SCALE} is DESCRIPTIVE and cannot be selected; it is scored so the
     plateau rule has three points and can report something measured. With only
     two points its continuation clause has no previous step and the verdict
     would be PLATEAU_REACHED by construction.
  1. Determine the eligible EARLIER-layer set ({list(self.earlier_layers)}) at
     each candidate scale {list(self.candidate_scales)}, using the unchanged gate.
  2. Select the SMALLEST candidate scale whose eligible earlier-layer set equals
     the set at scale {max(self.candidate_scales):,}, provided the predeclared
     plateau rule reports that improvement has stopped
     (required: {self.require_plateau_reached}).
  3. Otherwise select scale {self.fallback_scale:,}.
  4. If NEITHER L{PRIMARY_EARLY_LAYER} NOR L{SECONDARY_EARLY_LAYER} passes
     development at any candidate scale, the rule is
     "{self.on_no_early_layer_development_pass}": one honest confirmation at
     scale {self.fallback_scale:,}, reported whatever it says.
  5. Confirmation results may be consulted: {self.confirmation_may_be_consulted}.
     Multimodal causal outcomes may be consulted: {self.multimodal_outcomes_may_be_consulted}.

The scale is never chosen after seeing a confirmation number. The vault that
holds the confirmation set will not open without the payload this rule produces.
"""


EXTENSION_SELECTION_RULE = ExtensionSelectionRule()


def select_extension_scale(
    comparison: Mapping,
    *,
    plateau: Mapping | None = None,
    rule: ExtensionSelectionRule = EXTENSION_SELECTION_RULE,
    candidate_scales: Sequence[int] | None = None,
) -> dict:
    """Apply the predeclared rule to the development scale comparison.

    Args:
        comparison: The payload from
            :func:`jlens.calibration.scale.compare_scales`, computed on the
            **development** set only. It may — and normally does — also carry
            the descriptive baseline scale; the baseline is reported and is
            never selectable.
        plateau: The payload from
            :func:`jlens.calibration.scale.evaluate_plateau`. When absent, or
            when it reports improvement is still visible, the smaller candidate
            scale cannot be selected.
        candidate_scales: The scales that may be selected. Defaults to the
            rule's own; MOCK passes its small equivalents.

    Returns:
        A selection payload carrying ``confirmation_not_consulted``, which is
        what :class:`jlens.calibration.publication.ConfirmationVault` requires
        before it will release the confirmation set.

    Raises:
        ValueError: If a candidate scale is missing from the comparison. A
            selection made over a subset of the candidates it declared is not
            the predeclared rule.
    """
    scored = sorted(int(scale) for scale in comparison["scales"])
    candidates = sorted(
        {int(scale) for scale in (candidate_scales or rule.candidate_scales)}
    )
    if not candidates:
        raise ValueError("select_extension_scale needs at least one candidate scale")
    missing = [scale for scale in candidates if scale not in scored]
    if missing:
        raise ValueError(
            f"candidate scale(s) {missing} were not scored on the development "
            f"set (scored: {scored}); refusing to select over a subset of the "
            "declared candidates"
        )
    descriptive = [scale for scale in scored if scale not in candidates]

    eligible = {int(k): list(v) for k, v in comparison["eligible_by_scale"].items()}
    earlier = {int(layer) for layer in rule.earlier_layers}
    eligible_earlier = {
        scale: sorted(layer for layer in eligible[scale] if layer in earlier)
        for scale in scored
    }

    largest = candidates[-1]
    target = eligible_earlier[largest]
    plateau_reached = bool(
        plateau is not None and not plateau.get("extension_justified", False)
    )
    any_earlier_passes = any(eligible_earlier[scale] for scale in candidates)

    if not any_earlier_passes:
        chosen = largest
        clause = "no_early_layer_development_pass"
        reason = (
            f"no earlier layer in {sorted(earlier)} passed development at any "
            f"candidate scale {candidates}; the predeclared rule is "
            f"'{rule.on_no_early_layer_development_pass}', so scale {chosen:,} "
            "goes to one honest confirmation and its result is reported whatever "
            "it says"
        )
    else:
        chosen = largest
        clause = "fallback_to_largest"
        reason = (
            f"no smaller candidate scale reproduces the eligible earlier-layer "
            f"set {target} at scale {largest:,} under a reached plateau "
            f"(plateau_reached={plateau_reached}); the largest candidate scale "
            "is taken to confirmation"
        )
        for scale in candidates[:-1]:
            if eligible_earlier[scale] == target and (
                plateau_reached or not rule.require_plateau_reached
            ):
                chosen = scale
                clause = "smallest_scale_matching_largest"
                reason = (
                    f"smallest candidate scale whose eligible earlier-layer set "
                    f"{target} equals the set at {largest:,}, with the plateau "
                    f"rule reporting improvement has stopped "
                    f"(plateau_reached={plateau_reached})"
                )
                break

    payload = {
        "selected_scale": int(chosen),
        "selection_rule": rule.to_dict(),
        "selection_rule_digest": rule.digest,
        "selection_rule_version": rule.version,
        "clause_applied": clause,
        "reason": reason,
        "candidate_scales": candidates,
        "descriptive_scales_not_selectable": descriptive,
        "eligible_by_scale": {str(k): v for k, v in eligible.items()},
        "eligible_earlier_by_scale": {str(k): v for k, v in eligible_earlier.items()},
        "eligible_at_selected_scale": eligible[chosen],
        "eligible_earlier_at_selected_scale": eligible_earlier[chosen],
        "plateau_verdict": (plateau or {}).get("verdict"),
        "plateau_reached": plateau_reached,
        "plateau_clause_informative": bool(len(scored) >= 3),
        "declared_before_results": True,
        "confirmation_not_consulted": True,
        "multimodal_outcomes_not_consulted": True,
        "selected_from_development_only": True,
    }
    payload["selection_checksum"] = payload_checksum(payload)
    return payload


# ---------------------------------------------------------------- the verdict


def early_layer_verdict(
    confirmation: Mapping[int, Mapping],
    *,
    scale: int,
    selection: Mapping,
    development: Mapping[int, Mapping] | None = None,
    protocol: ExtensionProtocol = EXTENSION_PROTOCOL,
) -> dict:
    """GO or NO-GO for a validated earlier readout, from confirmation alone.

    ``GO`` requires at least one of L32 / L26 to have passed the same frozen gate
    on the fresh, untouched confirmation set. Anything else is
    :data:`EARLY_LAYER_NO_GO`, stated plainly: the current estimator did not
    yield a validated earlier readout at the authorized endpoint.
    """
    results = {int(layer): dict(verdict) for layer, verdict in confirmation.items()}
    passed_early = sorted(
        layer
        for layer in (protocol.primary_layer, protocol.secondary_layer)
        if results.get(layer, {}).get("passed", False)
    )
    verdict = EARLY_LAYER_GO if passed_early else EARLY_LAYER_NO_GO

    layers: list[dict] = []
    for layer in sorted(results):
        result = results[layer]
        metrics = (result.get("metrics") or {}).get("j_lens", {})
        layers.append(
            {
                "layer": layer,
                "normalized_depth": normalized_depth(layer),
                "role": (
                    "primary"
                    if layer == protocol.primary_layer
                    else "secondary"
                    if layer == protocol.secondary_layer
                    else "already_published"
                    if layer in protocol.established_layers
                    else "descriptive"
                ),
                "confirmation_passed": bool(result.get("passed", False)),
                "confirmation_failed_checks": list(result.get("failed_checks", [])),
                "mean_reciprocal_rank": metrics.get("mean_reciprocal_rank"),
                "median_midrank": metrics.get("median_midrank"),
                "median_optimistic_rank": metrics.get("median_optimistic_rank"),
                "top10_inclusion": metrics.get("top10_inclusion"),
                "tied_at_max_rate": metrics.get("tied_at_max_rate"),
                "development_passed": bool(
                    (development or {}).get(layer, {}).get("passed", False)
                )
                if development
                else None,
                "publishable": layer in PUBLISHABLE_LAYERS,
            }
        )

    payload = {
        "schema": "jlens.calibration.early_layer_verdict.v1",
        "protocol_version": protocol.version,
        "protocol_digest": protocol.digest,
        "verdict": verdict,
        "scale": int(scale),
        "selected_scale": int(selection.get("selected_scale", scale)),
        "selection_checksum": selection.get("selection_checksum"),
        "primary_layer": protocol.primary_layer,
        "secondary_layer": protocol.secondary_layer,
        "earlier_layers_passing_confirmation": passed_early,
        "layers": layers,
        "statement": (
            f"L{', L'.join(str(layer) for layer in passed_early)} passed the "
            "frozen gate on the fresh untouched confirmation set at scale "
            f"{int(scale):,}."
            if passed_early
            else (
                "The current estimator did not yield a validated earlier readout "
                f"at the authorized endpoint (scale {int(scale):,}). Neither "
                f"L{protocol.primary_layer} nor L{protocol.secondary_layer} "
                "passed the frozen gate on the fresh untouched confirmation set. "
                "Every metric is reported; no earlier lens is published."
            )
        ),
        "existing_publications_unchanged": list(protocol.established_layers),
        "all_layer_results_recorded": True,
    }
    payload["verdict_checksum"] = payload_checksum(payload)
    return payload


# ------------------------------------------------------------- publication


def publish_early_layer(
    *,
    layer: int,
    scale: int,
    lens,
    destination: Path | str,
    confirmation_verdict: Mapping,
    development_verdict: Mapping,
    vault: ConfirmationVault,
    parent: ParentRun,
    parent_audit: Mapping,
    continuation: Mapping,
    splits: FreshEvaluationSplits,
    selection: Mapping,
    extension_run_dir: Path | str,
    gate: CalibrationGate = EXTENSION_CONFIRMATION_GATE,
    protocol: ExtensionProtocol = EXTENSION_PROTOCOL,
    **artifact_fields,
) -> dict:
    """Publish one earlier layer that passed the fresh confirmation gate.

    Every refusal :func:`jlens.calibration.publication.publish_layer` already
    enforces applies unchanged — no confirmation verdict, a failed verdict, a
    development verdict offered in its place, a layer/scale mismatch, a
    protected completed-run path. Three refusals are added here:

    * a layer outside :data:`PUBLISHABLE_LAYERS`, so the confirmed L35/L38/L40
      lenses are neither overwritten nor quietly republished under a new name;
    * a destination outside the extension's own run directory;
    * a destination inside the parent run.

    Writes the lens, the standard artifact sidecar, and a second
    ``.extension.json`` sidecar carrying the parent checksums, the fresh split
    checksums, the continuation record and both sets of metrics.
    """
    destination = Path(destination)
    run_dir = Path(extension_run_dir).resolve()
    resolved = destination.resolve()

    if int(layer) not in {int(item) for item in PUBLISHABLE_LAYERS}:
        raise PublicationRefused(
            f"layer {layer} is not an extension publication target "
            f"{list(PUBLISHABLE_LAYERS)}. L{'/L'.join(str(l) for l in protocol.established_layers)} "
            "already have confirmed published lenses and this extension neither "
            "replaces nor republishes them; shallower layers are descriptive "
            "unless independently confirmed, and this extension does not treat "
            "them as publication targets."
        )
    if run_dir not in resolved.parents:
        raise PublicationRefused(
            f"{destination} is outside the extension run directory {run_dir}. "
            "Extension artifacts are written only into the extension's own run."
        )
    parent_root = Path(parent.root).resolve()
    if parent_root == resolved or parent_root in resolved.parents:
        raise PublicationRefused(
            f"{destination} is inside the parent run {parent.root}. The parent "
            "run is the evidence base for the completed scale-100 study and is "
            "never written to."
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
        protocol_version=protocol.version,
        **artifact_fields,
    )

    extension_artifact = {
        "schema": EXTENSION_ARTIFACT_SCHEMA,
        "protocol_version": protocol.version,
        "protocol_digest": protocol.digest,
        "frozen": True,
        "validated": bool(confirmation_verdict.get("passed", False)),
        "publication_status": "PUBLISHED_VALIDATED_EARLY_LAYER",
        "physical_layer": int(layer),
        "normalized_depth": normalized_depth(int(layer)),
        "scale_point": int(scale),
        # --- what it was continued from
        "parent_run_root": parent.root,
        "parent_run_fingerprint_digest": parent.fingerprint_digest,
        "parent_accumulator_checksum": parent.accumulator.checksum,
        "parent_accumulator_n_done": parent.accumulator.n_done,
        "parent_audit_checksum": parent_audit.get("audit_checksum"),
        "parent_baseline_lens_checksum": parent.baseline_lens_checksum,
        "old_confirmation_set_reused": False,
        "old_development_set_reused": False,
        # --- identity
        "model_repo_id": artifact.get("model_repo_id"),
        "model_revision": artifact.get("model_revision"),
        "tokenizer_repo_id": artifact.get("tokenizer_repo_id"),
        "tokenizer_revision": artifact.get("tokenizer_revision"),
        "corpus_id": artifact.get("corpus_id"),
        "corpus_revision": artifact.get("corpus_revision"),
        "corpus_revision_status": artifact.get("corpus_revision_status"),
        # --- the fit
        "n_fitting_prompts": artifact.get("n_fitting_prompts"),
        "fit_order_protocol": EXTENSION_FIT_ORDER_PROTOCOL,
        "continuation": dict(continuation),
        "estimator": artifact.get("estimator"),
        "objective": "not_applicable_estimator_is_a_sample_mean",
        "target_layer": artifact.get("target_layer"),
        "hook_site": artifact.get("hook_site"),
        "residual_convention": artifact.get("residual_convention"),
        # --- the splits
        "fit_split_checksum": artifact.get("fit_split_checksum"),
        "development_split_checksum": splits.checksum("development"),
        "confirmation_split_checksum": splits.checksum("confirmation"),
        "fresh_splits_manifest_checksum": splits.manifest()["manifest_checksum"],
        "split_protocol": splits.protocol,
        # --- how it was judged
        "gate_version": gate.version,
        "gate_digest": gate.digest,
        "selection_rule_digest": selection.get("selection_rule_digest"),
        "selection_checksum": selection.get("selection_checksum"),
        "development_metrics": development_verdict.get("metrics"),
        "development_failed_checks": list(development_verdict.get("failed_checks", [])),
        "confirmation_metrics": confirmation_verdict.get("metrics"),
        "confirmation_failed_checks": list(
            confirmation_verdict.get("failed_checks", [])
        ),
        "target_diversity": confirmation_verdict.get("target_diversity"),
        # --- the files
        "lens_path": artifact.get("lens_path"),
        "lens_checksum": artifact.get("lens_checksum"),
        "base_artifact_checksum": artifact.get("artifact_checksum"),
        # --- what stays put
        "existing_publications_unchanged": list(protocol.established_layers),
        "parent_run_written": False,
        "calibration_modality": "text-only",
        "spokencoco_used": False,
        "multimodal_data_used": False,
    }
    extension_artifact["artifact_checksum"] = payload_checksum(extension_artifact)

    sidecar = destination.with_suffix(".extension.json")
    temporary = sidecar.with_name(f"{sidecar.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(extension_artifact, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, sidecar)
    return extension_artifact


# ------------------------------------------------------------------- budget


#: The user's own measurement of the parent run: 100 prompts, this exact layer
#: grid and capture plan, on one NVIDIA L4. Everything below is extrapolated
#: from it, linearly in prompts, and the linearity is the assumption to
#: distrust first.
OBSERVED_SCALE100_FIT_MINUTES = 7.1

#: Multiplicative band on the extrapolation. Wider on the high side because the
#: anchor is a single observation on a shared cloud runtime.
EXTENSION_UNCERTAINTY_BAND = (0.9, 1.6)


def extension_budget(
    *,
    plan: CapturePlan,
    scale_points: Sequence[int] = EXTENSION_SCALE_POINTS,
    baseline_scale: int = BASELINE_SCALE,
    n_development: int = N_DEVELOPMENT_PROMPTS,
    n_confirmation: int = N_CONFIRMATION_PROMPTS,
    observed_minutes: float = OBSERVED_SCALE100_FIT_MINUTES,
) -> dict:
    """Incremental cost of the continuation, anchored on the observed run.

    Rows are **incremental**, not cumulative: the parent's first
    ``baseline_scale`` prompts are already in the accumulator and are never
    refitted. That is the whole economy of the continuation, and stating it as
    an incremental table is the honest way to show it.
    """
    points = sorted({int(point) for point in scale_points})
    minutes_per_prompt = float(observed_minutes) / float(baseline_scale)
    low, high = EXTENSION_UNCERTAINTY_BAND

    rows: list[dict] = []
    previous = int(baseline_scale)
    for point in points:
        incremental = point - previous
        central = minutes_per_prompt * incremental
        rows.append(
            {
                "scale": point,
                "incremental_prompts": incremental,
                "prompts_already_in_accumulator": previous,
                "forward_passes": incremental * plan.forwards_per_prompt,
                "backward_passes": incremental * plan.backward_passes_per_prompt,
                "l4_minutes_central": round(central, 1),
                "l4_hours_central": round(central / 60.0, 2),
                "l4_hours_low": round(central * low / 60.0, 2),
                "l4_hours_high": round(central * high / 60.0, 2),
                "cumulative_prompts": point,
            }
        )
        previous = point

    matrix_bytes = plan.d_model * plan.d_model
    n_layers = len(plan.layers)
    checkpoint_bytes = n_layers * matrix_bytes * 4
    snapshot_bytes = n_layers * matrix_bytes * 2
    published_bytes = matrix_bytes * 2 * len(PUBLISHABLE_LAYERS)
    reports_bytes = 20 * 1024 * 1024

    payload = {
        "anchor": {
            "source": "the operator's observed scale-100 fit on one NVIDIA L4",
            "n_prompts": int(baseline_scale),
            "minutes": float(observed_minutes),
            "minutes_per_prompt": round(minutes_per_prompt, 4),
            "layers": list(plan.layers),
            "target_layer": plan.target_layer,
            "backward_passes_per_prompt": plan.backward_passes_per_prompt,
        },
        "extrapolation_uncertainty": {
            "band": list(EXTENSION_UNCERTAINTY_BAND),
            "why": (
                "one observation, one runtime, one prompt-length distribution. "
                "The model is linear in prompts, which the estimator's structure "
                "supports (one forward plus a fixed number of backward passes per "
                "prompt), but a shared cloud L4 is not a fixed-speed device and "
                "sequence lengths vary. Treat the high end as the planning number."
            ),
        },
        "rows": rows,
        "incremental_not_cumulative": True,
        "storage_bytes": {
            "checkpoint": checkpoint_bytes,
            "snapshots": snapshot_bytes * len(points),
            "published_max": published_bytes,
            "reports": reports_bytes,
            "total": checkpoint_bytes
            + snapshot_bytes * len(points)
            + published_bytes
            + reports_bytes,
        },
        "non_fitting_model_minutes": {
            "target_discovery_and_readout_low": round(
                2 + 4 * (n_development + n_confirmation) / 128.0, 1
            ),
            "target_discovery_and_readout_high": round(
                2 + 12 * (n_development + n_confirmation) / 128.0, 1
            ),
            "why": (
                "development and confirmation are forward-pass workloads; the "
                f"{plan.backward_passes_per_prompt} backward passes per prompt "
                "happen only during fitting"
            ),
        },
        "resume": (
            "the accumulator checkpoint is written atomically every "
            "checkpoint_every prompts and reloaded automatically; a disconnect "
            "loses at most one checkpoint interval, never the parent's 100"
        ),
    }
    payload["budget_checksum"] = payload_checksum(payload)
    return payload


def format_extension_budget(budget: Mapping) -> str:
    """The block printed before any budget switch may be set."""
    anchor = budget["anchor"]
    lines = [
        "EXTENSION COMPUTE AND STORAGE BUDGET — incremental, not cumulative",
        f"  anchor            {anchor['n_prompts']} prompts in "
        f"{anchor['minutes']:.1f} min on one L4 (observed on this project's own run)",
        f"  per prompt        {anchor['minutes_per_prompt'] * 60:.1f} s = 1 forward + "
        f"{anchor['backward_passes_per_prompt']} backward passes",
        f"  layers            {anchor['layers']} -> L{anchor['target_layer']}",
        f"  uncertainty       x{budget['extrapolation_uncertainty']['band'][0]}"
        f"-x{budget['extrapolation_uncertainty']['band'][1]} on a single observation",
        "",
        "  The parent's first 100 prompts are ALREADY in the accumulator and are",
        "  never refitted. Rows below are the INCREMENTAL cost of reaching each",
        "  scale from the one before it.",
        "",
        "  scale      incremental   forward   backward     L4 hours (central)   range",
    ]
    for row in budget["rows"]:
        lines.append(
            f"  {row['scale']:>7,}  {row['incremental_prompts']:>11,}  "
            f"{row['forward_passes']:>8,}  {row['backward_passes']:>10,}  "
            f"{row['l4_hours_central']:>18.2f}   "
            f"{row['l4_hours_low']:.2f}-{row['l4_hours_high']:.2f}"
        )
    storage = budget["storage_bytes"]
    non_fitting = budget["non_fitting_model_minutes"]
    lines += [
        "",
        f"  checkpoint        {storage['checkpoint'] / 2**20:.0f} MiB (fp32 running sum)",
        f"  snapshots         {storage['snapshots'] / 2**20:.0f} MiB",
        f"  published (max)   {storage['published_max'] / 2**20:.0f} MiB",
        f"  Drive total       {storage['total'] / 2**30:.2f} GiB",
        f"  non-fitting model work  "
        f"{non_fitting['target_discovery_and_readout_low']:.0f}-"
        f"{non_fitting['target_discovery_and_readout_high']:.0f} min",
        "",
        f"  resume            {budget['resume']}",
        "",
        "  EXTRAPOLATION, NOT MEASUREMENT. The 7.1-minute anchor is one",
        "  observation of one run; the 250- and 1,000-prompt rows are that number",
        "  multiplied out. The high end of the band is the planning number.",
    ]
    return "\n".join(lines)
