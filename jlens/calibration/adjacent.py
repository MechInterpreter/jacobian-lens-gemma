# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""The adjacent-layer interval: physical layers 27–31, fitted fresh at scale 250.

Why this interval and no other
==============================

Three results bracket it, and all three are already on the record:

* **L26** failed the frozen J-lens confirmation gate.
* **L32** passed J-lens confirmation at scale 250 and was then classified
  ``AMBIGUOUS`` *twice*, independently, under the frozen native direct-readout
  convergence criterion.
* **L35** is ``CONVERGED``.

So the open question — is there a layer with a *confirmed* lens whose native
readout has *not yet* converged — lives strictly between 26 and 32. The
candidates are therefore exactly ``(27, 28, 29, 30, 31)``, declared here, before
any of them has been fitted. There is no L33/L34 clause, no widening after a
result, and no replacement layer: :data:`ADJACENT_CANDIDATE_LAYERS` is a frozen
tuple bound into the run fingerprint, and a notebook that fits a different set
gets a different fingerprint and cannot resume this one.

What is reused, and what cannot be
==================================

*Reused:* the pinned checkpoint and revision, the text-only calibration
protocol, the corpus family and its prompt normalization, the target layer, the
Jacobian convention, the final-norm convention, the tie-aware native-readout
validity rule and **every one of its thresholds** — :data:`ADJACENT_GATE` and
:data:`ADJACENT_CONFIRMATION_GATE` *are* the extension's gate objects, not
copies with an edit. The fitting scale is 250 because that is the scale at which
L32 was selected and independently confirmed; no new scale search is run.

*Not reusable:* the accumulator. ``runs/rgext_*`` holds a ``jacobian_sum`` over
source layers ``[8, 14, 20, 26, 32, 35, 38, 40]``. None of 27–31 is in it, and
an accumulator does not acquire a layer it never captured. New Jacobian
accumulation over the five new source layers is required, and
:func:`assert_new_source_layers` refuses any attempt to seed this study's
checkpoint from a parent whose layer grid disagrees.

*Also not reusable:* the extension's confirmation set. It has been opened —
that is how L26's failure and L32's pass are known — so it is development
history for these five candidates. :func:`build_untouched_confirmation` draws a
genuinely untouched set and refuses rather than shrinking.

Text-only by construction, exactly as :mod:`jlens.calibration` is.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from jlens.calibration.corpus import (
    MAX_HAMMING_DISTANCE,
    CorpusRecord,
)
from jlens.calibration.extension import (
    EXTENSION_CONFIRMATION_GATE,
    EXTENSION_GATE,
    ExtensionStore,
    _NearDuplicateIndex,
)
from jlens.calibration.gate import CalibrationGate, audit_target_diversity
from jlens.calibration.state import CALIBRATION_STAGES
from jlens.mmpilot.store import payload_checksum

__all__ = [
    "ADJACENT_CANDIDATE_LAYERS",
    "ADJACENT_CONFIRMATION_GATE",
    "ADJACENT_CONFIRMATION_PROMPT_SEED",
    "ADJACENT_FITTING_SCALE",
    "ADJACENT_GATE",
    "ADJACENT_HOOK_SITE",
    "ADJACENT_LENS_GO",
    "ADJACENT_LENS_NO_GO",
    "ADJACENT_PROTOCOL",
    "ADJACENT_PROTOCOL_VERSION",
    "ADJACENT_RUN_PREFIX",
    "ADJACENT_SELECTION_RULE",
    "ADJACENT_SPLIT_PROTOCOL",
    "ADJACENT_SPLIT_SEED",
    "ADJACENT_STAGES",
    "AMBIGUOUS_UPPER_LAYER",
    "CONVERGED_REFERENCE_LAYER",
    "FAILED_LOWER_LAYER",
    "N_CONFIRMATION_PROMPTS",
    "N_DEVELOPMENT_PROMPTS",
    "AdjacentProtocol",
    "AdjacentSelectionRule",
    "AdjacentStore",
    "ConfirmationNotUntouched",
    "SourceLayerSetRefused",
    "UntouchedConfirmation",
    "UntouchedConfirmationRefused",
    "adjacent_budget",
    "adjacent_corpus_manifest",
    "adjacent_gate_text",
    "adjacent_lens_verdict",
    "adjacent_target_diversity",
    "assert_new_source_layers",
    "audit_untouched_confirmation",
    "build_untouched_confirmation",
    "confirmation_table",
    "format_adjacent_budget",
    "format_confirmation_table",
    "select_earliest_confirmed_layer",
]

# ------------------------------------------------------------ frozen identity

ADJACENT_PROTOCOL_VERSION = "research-grade-l27-l31-adjacent-layer-jlens-v1"

#: The five candidates, and only these five. Frozen before any of them exists as
#: a fitted lens.
ADJACENT_CANDIDATE_LAYERS: tuple[int, ...] = (27, 28, 29, 30, 31)

#: The layer that failed the frozen confirmation gate; the interval's floor.
FAILED_LOWER_LAYER = 26

#: The layer that passed confirmation at scale 250 and was classified AMBIGUOUS
#: twice under the frozen convergence criterion; the interval's ceiling.
AMBIGUOUS_UPPER_LAYER = 32

#: The layer already classified CONVERGED. Reported for orientation, never
#: fitted, re-confirmed or republished here.
CONVERGED_REFERENCE_LAYER = 35

#: The one scale. L32 was selected and independently confirmed at 250, so 250 is
#: what makes a new layer's confirmation comparable to it. No scale search runs.
ADJACENT_FITTING_SCALE = 250

#: The exact hook site, recorded so a lens fitted somewhere else cannot be
#: mistaken for one of these.
ADJACENT_HOOK_SITE = (
    "output of model.language_model.layers[l] — the post-block residual; the "
    "final prompt token where a single position is read"
)

ADJACENT_RUN_PREFIX = "rgadj"

#: How the untouched confirmation set is drawn. A distinct tag and a distinct
#: seed from the extension's, so a record's partition is genuinely new rather
#: than the same function at a different number.
ADJACENT_SPLIT_PROTOCOL = "adjacent-untouched-confirmation-hash-bucket-v1"
ADJACENT_SPLIT_SEED = 20260810
ADJACENT_CONFIRMATION_PROMPT_SEED = 20260811

#: Sizes. Identical to the extension's, because a smaller endpoint for a harder
#: question would be a weaker endpoint.
N_DEVELOPMENT_PROMPTS = 256
N_CONFIRMATION_PROMPTS = 256

ADJACENT_N_BUCKETS = 100
ADJACENT_CONFIRMATION_BUCKETS = (0, 99)

ADJACENT_ONLY_STAGES = (
    "corpus_provenance",
    "untouched_confirmation",
    "adjacent_fit",
    "adjacent_development",
    "adjacent_confirmation",
    "layer_selection",
)
ADJACENT_STAGES = (*CALIBRATION_STAGES, *ADJACENT_ONLY_STAGES)

ADJACENT_LENS_GO = "ADJACENT_LENS_GO"
ADJACENT_LENS_NO_GO = "ADJACENT_LENS_NO_GO"


class UntouchedConfirmationRefused(RuntimeError):
    """An untouched confirmation set of the required size cannot be built.

    Raised rather than satisfied by relaxation. The three relaxations available
    — shrink the set, substitute a different corpus, reopen a spent set — each
    destroy exactly the property the set exists to have.
    """


class ConfirmationNotUntouched(UntouchedConfirmationRefused):
    """A constructed confirmation record collides with something already spent."""


class SourceLayerSetRefused(RuntimeError):
    """A checkpoint or accumulator does not hold this study's source layers."""


# ------------------------------------------------------------------ the gate

#: The validity gate, **unchanged**. This is the extension's own object, bound
#: here by reference: the thresholds that decided L26 and L32 are the thresholds
#: that decide 27–31, down to the digest. A study that adjusted a bar to admit a
#: new layer would not be applying the frozen gate to it.
ADJACENT_GATE: CalibrationGate = EXTENSION_GATE

#: The same rule at the confirmation sample size. Also unchanged.
ADJACENT_CONFIRMATION_GATE: CalibrationGate = EXTENSION_CONFIRMATION_GATE


def adjacent_gate_text(gate: CalibrationGate = ADJACENT_CONFIRMATION_GATE) -> str:
    """The criterion block, printed before any adjacent-layer number exists."""
    return f"""\
FROZEN J-LENS VALIDITY GATE — {gate.version}
Digest: {gate.digest}

This is the SAME OBJECT the early-layer extension applied to L26 and L32. Not a
copy, not a re-derivation, not a re-tuned variant: `ADJACENT_GATE is
EXTENSION_GATE` and `ADJACENT_CONFIRMATION_GATE is EXTENSION_CONFIRMATION_GATE`,
and a test asserts that identity rather than comparing field by field.

  sample size              {gate.n_prompts} prompts
  distinct targets         >= {gate.min_distinct_target_tokens}
  max single-target share  <= {gate.max_target_token_share:.0%}
  tied-at-maximum rate     <= {gate.max_tied_at_max_rate:.2f}
  noise-control MRR        >= {gate.min_noise_control_mrr_ratio:.1f}x AND >= +{gate.min_noise_control_mrr_margin:.2f}
  wrong-layer margin       >= +{gate.min_wrong_layer_mrr_margin:.2f} via {gate.wrong_layer_mapping}
  median midrank           <= {gate.max_median_midrank:.1f}
  top-{gate.top_k} inclusion          >= {gate.min_top_k_inclusion:.2f} at the PESSIMISTIC rank
  fold stability           {gate.n_folds} fixed folds, each beating every control and
                           reaching >= {gate.min_fold_mrr_fraction:.2f}x the overall MRR

NOTHING IS CHANGED to let a layer in 27-31 pass, and nothing is changed after a
result is seen. Continuous metrics are reported for EVERY candidate at EVERY
clause whether it passes or fails.
"""


# --------------------------------------------------------------- the protocol


@dataclass(frozen=True)
class AdjacentProtocol:
    """What this study is, frozen before any of its results exist."""

    version: str = ADJACENT_PROTOCOL_VERSION
    question: str = (
        "does any predeclared layer in 27-31 simultaneously have a rigorously "
        "confirmed text-calibrated J-lens, a clearly NOT_CONVERGED native "
        "direct readout across text, image and spoken audio, and controlled "
        "cross-modal causal transfer"
    )
    candidate_layers: tuple[int, ...] = ADJACENT_CANDIDATE_LAYERS
    candidate_layers_are_closed: bool = True
    interval_floor: int = FAILED_LOWER_LAYER
    interval_floor_status: str = "failed the frozen J-lens confirmation gate"
    interval_ceiling: int = AMBIGUOUS_UPPER_LAYER
    interval_ceiling_status: str = (
        "passed J-lens confirmation at scale 250; classified AMBIGUOUS twice, "
        "independently, under the frozen native direct-readout criterion"
    )
    reference_layer: int = CONVERGED_REFERENCE_LAYER
    reference_layer_status: str = "already CONVERGED"
    fitting_scale: int = ADJACENT_FITTING_SCALE
    fitting_scale_rationale: str = (
        "L32 was selected and independently confirmed at scale 250, so 250 is "
        "the scale at which a new layer's confirmation is comparable to it. No "
        "scale search is run and no other scale is authorized."
    )
    hook_site: str = ADJACENT_HOOK_SITE
    accumulator_reuse: str = (
        "NONE. The completed extension accumulator holds source layers "
        "[8, 14, 20, 26, 32, 35, 38, 40]; none of 27-31 is in it. New Jacobian "
        "accumulation over the five new source layers is required, and seeding "
        "from a parent with a different layer grid is refused."
    )
    development_set: str = (
        "the extension's already-designated development records, reused as "
        "development with their role and hashes recorded"
    )
    confirmation_set: str = (
        "genuinely untouched for these five candidates, disjoint by prompt hash "
        "from every fitting prompt, every development prompt, every previously "
        "opened confirmation prompt and every prior calibration manifest named "
        "as a dependency"
    )
    gate_version: str = ADJACENT_CONFIRMATION_GATE.version
    gate_digest: str = ADJACENT_CONFIRMATION_GATE.digest
    gate_is_the_extension_gate: bool = True
    thresholds_frozen_before_results: bool = True
    layer_selection: str = (
        "the LOWEST physical candidate that passes every frozen confirmation "
        "clause. Never the best-looking failure, and never chosen after the "
        "table is read."
    )
    multimodal_data_in_fitting: bool = False
    multimodal_data_in_lens_validation: bool = False
    no_go_verdict: str = ADJACENT_LENS_NO_GO

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["candidate_layers"] = list(payload["candidate_layers"])
        return payload

    @property
    def digest(self) -> str:
        return payload_checksum(self.to_dict())

    def text(self) -> str:
        return f"""\
ADJACENT-LAYER STUDY PROTOCOL — {self.version}
Digest: {self.digest}
Frozen before any adjacent-layer result exists.

  1. THE QUESTION
     {self.question}?

  2. THE INTERVAL, AND WHY IT IS CLOSED
     L{self.interval_floor}: {self.interval_floor_status}
     L{self.interval_ceiling}: {self.interval_ceiling_status}
     L{self.reference_layer}: {self.reference_layer_status}
     candidates {list(self.candidate_layers)} — closed: {self.candidate_layers_are_closed}
     No L33/L34, no widening after a result, no replacement layer.

  3. SCALE
     {self.fitting_scale} — {self.fitting_scale_rationale}

  4. HOOK SITE
     {self.hook_site}

  5. ACCUMULATOR
     {self.accumulator_reuse}

  6. EVALUATION SETS
     development:  {self.development_set}
     confirmation: {self.confirmation_set}

  7. THE GATE
     {self.gate_version}
     digest {self.gate_digest}
     unchanged from the extension's: {self.gate_is_the_extension_gate}
     frozen before results: {self.thresholds_frozen_before_results}

  8. LAYER SELECTION
     {self.layer_selection}
     If none pass: {self.no_go_verdict}.

  9. MODALITY
     multimodal data in fitting:         {self.multimodal_data_in_fitting}
     multimodal data in lens validation: {self.multimodal_data_in_lens_validation}
"""


ADJACENT_PROTOCOL = AdjacentProtocol()


# --------------------------------------------------------- the selection rule


@dataclass(frozen=True)
class AdjacentSelectionRule:
    """How the reported layer is chosen. Fixed before confirmation is opened."""

    version: str = "adjacent-earliest-fully-confirmed-layer-v1"
    candidates: tuple[int, ...] = ADJACENT_CANDIDATE_LAYERS
    evaluate: str = "every one of the five predeclared candidates"
    choose: str = (
        "the LOWEST physical layer that passes every frozen confirmation clause"
    )
    on_none_passing: str = ADJACENT_LENS_NO_GO
    best_looking_failure_may_be_chosen: bool = False
    complete_table_recorded_even_when_none_pass: bool = True
    declared_before_confirmation_opened: bool = True
    multimodal_outcomes_may_be_consulted: bool = False

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["candidates"] = list(payload["candidates"])
        return payload

    @property
    def digest(self) -> str:
        return payload_checksum(self.to_dict())

    def text(self) -> str:
        return f"""\
LAYER-SELECTION RULE — {self.version}
Digest: {self.digest}
Fixed BEFORE the untouched confirmation set is opened.

  evaluate            {self.evaluate} {list(self.candidates)}
  choose              {self.choose}
  if none pass        {self.on_none_passing}
  best-looking fail   may be chosen: {self.best_looking_failure_may_be_chosen}
  full table recorded {self.complete_table_recorded_even_when_none_pass}
  multimodal results  may be consulted: {self.multimodal_outcomes_may_be_consulted}

"Lowest that passes" is a rule a result cannot bend. "Closest to passing" is
not a rule at all — every table has one.
"""


ADJACENT_SELECTION_RULE = AdjacentSelectionRule()


# ------------------------------------------------------------------ the store


class AdjacentStore(ExtensionStore):
    """A calibration store that accepts this study's own stage vocabulary."""

    def stage_dir(self, stage: str) -> Path:
        if stage not in ADJACENT_STAGES:
            raise ValueError(
                f"unknown adjacent-layer stage {stage!r}; known stages are "
                f"{ADJACENT_STAGES}"
            )
        return self.root / "units" / stage

    def status_report(self, stages: Sequence[str] = ADJACENT_STAGES) -> dict:
        return super().status_report(stages)

    def snapshot_path(self, scale: int) -> Path:
        return self.root / "artifacts" / f"lens.adjacent.scale{int(scale)}.pt"

    def published_path(self, layer: int, scale: int) -> Path:
        return (
            self.root
            / "artifacts"
            / "published"
            / f"lens.adjacent.layer{int(layer)}.scale{int(scale)}.validated.pt"
        )


# ------------------------------------------------- the source-layer guardrail


def assert_new_source_layers(
    *,
    candidate_layers: Sequence[int] = ADJACENT_CANDIDATE_LAYERS,
    parent_source_layers: Sequence[int],
    parent_label: str = "the parent accumulator",
) -> dict:
    """Refuse to treat a parent accumulator as if it held these layers.

    The extension accumulator is a legitimate object and this study reads the
    parent run for its corpus provenance and fit ordering. What it must never do
    is *continue* that accumulator and call the result a lens at 27–31.

    Raises:
        SourceLayerSetRefused: If any candidate is already in the parent's grid,
            which would make "new accumulation" ambiguous, or if the grids are
            equal, which would mean the notebook is pointed at the wrong study.
    """
    candidates = tuple(int(layer) for layer in candidate_layers)
    parent = tuple(int(layer) for layer in parent_source_layers)
    overlap = sorted(set(candidates) & set(parent))
    payload = {
        "candidate_layers": list(candidates),
        "parent_source_layers": list(parent),
        "overlap": overlap,
        "disjoint": not overlap,
        "new_accumulation_required": True,
        "parent_accumulator_may_be_seeded": False,
        "why": (
            "a running mean over source layer l exists only if layer l was "
            "captured; an accumulator does not acquire a layer it never saw, so "
            "the five new source layers require new Jacobian accumulation"
        ),
    }
    payload["source_layer_checksum"] = payload_checksum(payload)
    if overlap:
        raise SourceLayerSetRefused(
            f"{parent_label} already holds source layer(s) {overlap}, which are "
            f"also candidates here. Refusing: with an overlap, 'the accumulator "
            "was newly fitted for these layers' stops being checkable. Point "
            "this study at a parent whose layer grid is disjoint from "
            f"{list(candidates)}."
        )
    return payload


# ------------------------------------------ the untouched confirmation set


def _adjacent_bucket(record_id: str, *, seed: int, n_buckets: int) -> int:
    """Bucket under this study's own seed and ``|adj|`` tag."""
    digest = hashlib.sha256(f"{seed}|adj|{record_id}".encode()).hexdigest()
    return int(digest[:8], 16) % int(n_buckets)


def _adjacent_order_key(record_id: str, *, seed: int) -> str:
    return hashlib.sha256(f"{seed}|adj-order|{record_id}".encode()).hexdigest()


@dataclass(frozen=True)
class UntouchedConfirmation:
    """The confirmation set for L27–L31, after every exclusion.

    ``records`` are drawn from corpus the fitting prompts, the development
    prompts and every previously opened confirmation set never reached, and the
    exclusion of each of those is counted rather than assumed.
    """

    records: tuple[CorpusRecord, ...]
    corpus_id: str
    seed: int
    protocol: str = ADJACENT_SPLIT_PROTOCOL
    n_pool: int = 0
    excluded_exact: dict = field(default_factory=dict)
    excluded_near: dict = field(default_factory=dict)
    excluded_pool_duplicates: int = 0
    n_near_comparisons: int = 0
    development_role: dict = field(default_factory=dict)
    dependency_manifests: tuple[str, ...] = ()

    @property
    def checksum(self) -> str:
        return payload_checksum([record.to_dict() for record in self.records])

    def record_ids(self) -> list[str]:
        return [record.record_id for record in self.records]

    def manifest(self) -> dict:
        payload = {
            "protocol": self.protocol,
            "corpus_id": self.corpus_id,
            "split_seed": int(self.seed),
            "size": len(self.records),
            "checksum": self.checksum,
            "record_ids": self.record_ids(),
            "bucket_rule": (
                f"sha256(seed|adj|record_id) % {ADJACENT_N_BUCKETS}; "
                f"confirmation {list(ADJACENT_CONFIRMATION_BUCKETS)}"
            ),
            "order_rule": "sha256(seed|adj-order|record_id), ascending",
            "pool_size_after_exclusions": int(self.n_pool),
            "excluded_exact": dict(self.excluded_exact),
            "excluded_near": dict(self.excluded_near),
            "excluded_pool_duplicates": int(self.excluded_pool_duplicates),
            "n_near_duplicate_comparisons": int(self.n_near_comparisons),
            "development_role": dict(self.development_role),
            "dependency_manifests": list(self.dependency_manifests),
            "selected_by_jlens_performance": False,
            "previously_opened_sets_reused": False,
            "size_reduced_to_fit_corpus": False,
        }
        payload["manifest_checksum"] = payload_checksum(payload)
        return payload


def build_untouched_confirmation(
    pool: Sequence[CorpusRecord],
    *,
    excluded: Mapping[str, Sequence[CorpusRecord]],
    corpus_id: str,
    seed: int = ADJACENT_SPLIT_SEED,
    n_confirmation: int = N_CONFIRMATION_PROMPTS,
    n_buckets: int = ADJACENT_N_BUCKETS,
    confirmation_buckets: tuple[int, int] = ADJACENT_CONFIRMATION_BUCKETS,
    max_hamming: int = MAX_HAMMING_DISTANCE,
    development_role: Mapping | None = None,
    dependency_manifests: Sequence[str] = (),
) -> UntouchedConfirmation:
    """Draw a confirmation set nothing in this lineage has seen.

    ``excluded`` must name, at minimum, every fitting prompt, every development
    prompt, and every previously opened confirmation prompt. The names are used
    in the refusal message, so a caller that lumps them into one key gets a
    less useful refusal — not a weaker check.

    Raises:
        UntouchedConfirmationRefused: If the required size cannot be filled. The
            size is never reduced, no substitute corpus is reached for, and no
            opened set is recycled: each of those would hand back exactly the
            property that made the extension's confirmation set unusable here.
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

    bucketed = [
        record
        for record in kept.values()
        if confirmation_buckets[0]
        <= _adjacent_bucket(record.record_id, seed=seed, n_buckets=n_buckets)
        <= confirmation_buckets[1]
    ]
    ordered = sorted(
        bucketed,
        key=lambda record: (
            _adjacent_order_key(record.record_id, seed=seed),
            record.record_id,
        ),
    )

    if len(ordered) < int(n_confirmation):
        raise UntouchedConfirmationRefused(
            f"the untouched confirmation partition holds {len(ordered)} records "
            f"but the frozen protocol requires exactly {n_confirmation}. This "
            "study is BLOCKED, not resized.\n"
            f"  pool after exclusions: {len(kept)}\n"
            f"  excluded as exact duplicates: {excluded_exact}\n"
            f"  excluded as near duplicates:  {excluded_near}\n"
            "Stream more of the pinned corpus. The three shortcuts available "
            "here — shrink the set, substitute a corpus, reopen a spent set — "
            "each destroy the untouchedness the set exists to provide."
        )

    return UntouchedConfirmation(
        records=tuple(ordered[: int(n_confirmation)]),
        corpus_id=str(corpus_id),
        seed=int(seed),
        n_pool=len(kept),
        excluded_exact=excluded_exact,
        excluded_near=excluded_near,
        excluded_pool_duplicates=pool_duplicates,
        n_near_comparisons=near_index.n_comparisons,
        development_role=dict(development_role or {}),
        dependency_manifests=tuple(str(name) for name in dependency_manifests),
    )


def audit_untouched_confirmation(
    confirmation: UntouchedConfirmation,
    *,
    excluded: Mapping[str, Sequence[CorpusRecord]],
    max_hamming: int = MAX_HAMMING_DISTANCE,
) -> dict:
    """Prove — on the constructed set — that nothing spent leaked into it.

    Run after construction by a predicate that is not the filter that built it,
    because a proof derived from the filter proves only that the filter agrees
    with itself.

    Raises:
        ConfirmationNotUntouched: On the first offending record, naming the set
            it collided with.
    """
    hits: list[dict] = []
    old_by_checksum = {
        record.normalized_checksum: (name, record)
        for name, records in excluded.items()
        for record in records
    }
    old_index = _NearDuplicateIndex(
        [(name, record) for name, records in excluded.items() for record in records]
    )
    for record in confirmation.records:
        collision = old_by_checksum.get(record.normalized_checksum)
        if collision is not None:
            hits.append(
                {
                    "kind": "exact",
                    "confirmation_record_id": record.record_id,
                    "collides_with_set": collision[0],
                    "collides_with_record_id": collision[1].record_id,
                }
            )
        near = old_index.hit(record, max_hamming=max_hamming)
        if near is not None:
            hits.append(
                {
                    "kind": "near",
                    "confirmation_record_id": record.record_id,
                    "collides_with_set": near[0],
                    "collides_with_record_id": near[1].record_id,
                    "hamming_distance": near[2],
                }
            )

    # Internal duplicates would inflate n without adding evidence.
    seen: dict[str, str] = {}
    for record in confirmation.records:
        previous = seen.get(record.normalized_checksum)
        if previous is not None:
            hits.append(
                {
                    "kind": "internal",
                    "confirmation_record_id": record.record_id,
                    "collides_with_set": "confirmation",
                    "collides_with_record_id": previous,
                }
            )
        seen[record.normalized_checksum] = record.record_id

    report = {
        "protocol": confirmation.protocol,
        "max_hamming_distance": int(max_hamming),
        "n_confirmation_records": len(confirmation.records),
        "excluded_sets": {name: len(records) for name, records in excluded.items()},
        "required_disjoint_from": sorted(excluded),
        "dependency_manifests": list(confirmation.dependency_manifests),
        "candidate_pairs_compared": old_index.n_comparisons,
        "n_exact_hits": sum(1 for hit in hits if hit["kind"] == "exact"),
        "n_near_hits": sum(1 for hit in hits if hit["kind"] == "near"),
        "n_internal_duplicates": sum(1 for hit in hits if hit["kind"] == "internal"),
        "hits": hits,
        "untouched": not hits,
    }
    report["audit_checksum"] = payload_checksum(report)
    if hits:
        first = hits[0]
        raise ConfirmationNotUntouched(
            f"{len(hits)} confirmation record(s) are not untouched; the first is "
            f"a {first['kind']} collision between "
            f"{first['confirmation_record_id']} and "
            f"{first['collides_with_set']} ({first['collides_with_record_id']}). "
            "Refusing. A spent prompt in this set would make it exactly as spent "
            "as the extension set it replaces."
        )
    return report


def adjacent_target_diversity(
    target_token_ids: Sequence[int], *, gate: CalibrationGate = ADJACENT_GATE
) -> dict:
    """Diversity statistics under the unchanged floor."""
    report = audit_target_diversity(target_token_ids, gate=gate)
    report["gate_version"] = gate.version
    report["gate_digest"] = gate.digest
    report["diversity_checksum"] = payload_checksum(
        {key: value for key, value in report.items() if key != "counts"}
    )
    return report


# ------------------------------------------------------- the corpus manifest


def adjacent_corpus_manifest(
    *,
    corpus_config: Mapping,
    corpus_id: str,
    fit_records: Sequence[CorpusRecord],
    development_records: Sequence[CorpusRecord],
    confirmation: UntouchedConfirmation,
    scale: int = ADJACENT_FITTING_SCALE,
    dependency_manifests: Mapping | None = None,
) -> dict:
    """The corpus block written into every adjacent-layer artifact.

    Shaped like :func:`jlens.calibration.corpus.corpus_manifest` — with
    ``splits.checksums`` carrying ``fit`` / ``validation`` / ``confirmation`` —
    so the existing artifact builders consume it unchanged. ``validation`` is
    this study's reused development set; the key name is what the schema calls
    it, and ``development_provenance`` records what it actually is.
    """
    fit_checksum = payload_checksum(
        [record.to_dict() for record in fit_records[: int(scale)]]
    )
    development_checksum = payload_checksum(
        [record.to_dict() for record in development_records]
    )
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
        "scale_points": [int(scale)],
        "splits": {
            "protocol": ADJACENT_SPLIT_PROTOCOL,
            "sizes": {
                "fit": int(scale),
                "validation": len(development_records),
                "confirmation": len(confirmation.records),
            },
            "checksums": {
                "fit": fit_checksum,
                "validation": development_checksum,
                "confirmation": confirmation.checksum,
            },
        },
        "confirmation_manifest": confirmation.manifest(),
        "development_provenance": dict(confirmation.development_role),
        "dependency_manifests": dict(dependency_manifests or {}),
        "fit_checksums_by_scale": {str(int(scale)): fit_checksum},
    }
    payload["corpus_manifest_checksum"] = payload_checksum(payload)
    return payload


# ------------------------------------------------------------ the selection


def confirmation_table(
    confirmation: Mapping[int, Mapping],
    *,
    candidates: Sequence[int] = ADJACENT_CANDIDATE_LAYERS,
    development: Mapping[int, Mapping] | None = None,
) -> list[dict]:
    """One row per candidate, pass or fail, with every reported clause.

    Built for **all** candidates including the ones that failed, because the
    complete table is what makes "lowest that passes" auditable rather than
    assertable.
    """
    rows: list[dict] = []
    for layer in sorted(int(candidate) for candidate in candidates):
        verdict = confirmation.get(layer) or confirmation.get(str(layer)) or {}
        metrics = (verdict.get("metrics") or {}).get("j_lens") or {}
        diversity = verdict.get("target_diversity") or {}
        development_verdict = None
        if development:
            development_verdict = development.get(layer) or development.get(str(layer))
        rows.append(
            {
                "layer": layer,
                "evaluated": bool(verdict),
                "passed": bool(verdict.get("passed")),
                "failed_clauses": list(verdict.get("failed_checks") or []),
                "checks": verdict.get("checks"),
                "mean_reciprocal_rank": metrics.get("mean_reciprocal_rank"),
                "median_midrank": metrics.get("median_midrank"),
                "median_optimistic_rank": metrics.get("median_optimistic_rank"),
                "median_pessimistic_rank": metrics.get("median_pessimistic_rank"),
                "tied_at_max_rate": metrics.get("tied_at_max_rate"),
                "top_k_inclusion": metrics.get("top10_inclusion"),
                "controls": {
                    name: values.get("mean_reciprocal_rank")
                    for name, values in (verdict.get("metrics") or {}).items()
                    if name != "j_lens"
                },
                "fold_mrr": verdict.get("fold_mrr"),
                "fold_beats_all_controls": verdict.get("fold_beats_all_controls"),
                "n_prompts": metrics.get("n_prompts"),
                "prompt_coverage": {
                    name: values.get("n_prompts")
                    for name, values in (verdict.get("metrics") or {}).items()
                },
                "n_distinct_target_tokens": diversity.get("n_distinct_target_tokens"),
                "max_target_token_share": diversity.get("max_target_token_share"),
                "development_passed": (
                    bool(development_verdict.get("passed"))
                    if development_verdict
                    else None
                ),
            }
        )
    return rows


def format_confirmation_table(rows: Sequence[Mapping]) -> str:
    """The table printed for every candidate, whatever the outcome."""
    lines = [
        f"{'layer':>6} {'result':>7} {'MRR':>8} {'midrank':>9} {'opt':>7} "
        f"{'pess':>7} {'tie@max':>8} {'top10':>7} {'distinct':>9}  failed clauses",
    ]

    def _fmt(value, spec: str) -> str:
        if value is None:
            return "-".rjust(int(spec.split(".")[0].strip("<>")) if spec else 6)
        return format(value, spec)

    for row in rows:
        lines.append(
            f"{row['layer']:>6} "
            f"{('PASS' if row['passed'] else 'fail' if row['evaluated'] else 'n/a'):>7} "
            f"{_fmt(row['mean_reciprocal_rank'], '8.4f')} "
            f"{_fmt(row['median_midrank'], '9.2f')} "
            f"{_fmt(row['median_optimistic_rank'], '7.1f')} "
            f"{_fmt(row['median_pessimistic_rank'], '7.1f')} "
            f"{_fmt(row['tied_at_max_rate'], '8.3f')} "
            f"{_fmt(row['top_k_inclusion'], '7.3f')} "
            f"{str(row['n_distinct_target_tokens'] or '-'):>9}  "
            f"{row['failed_clauses'] or ''}"
        )
    return "\n".join(lines)


def select_earliest_confirmed_layer(
    confirmation: Mapping[int, Mapping],
    *,
    candidates: Sequence[int] = ADJACENT_CANDIDATE_LAYERS,
    development: Mapping[int, Mapping] | None = None,
    rule: AdjacentSelectionRule = ADJACENT_SELECTION_RULE,
) -> dict:
    """Apply the predeclared rule: the lowest candidate that passes everything.

    Returns a payload with ``selected_layer`` set to ``None`` when nothing
    passes. That is a first-class outcome and the caller reports
    :data:`ADJACENT_LENS_NO_GO`; there is no branch here that reaches for the
    layer that came closest.
    """
    rows = confirmation_table(confirmation, candidates=candidates, development=development)
    passing = [row["layer"] for row in rows if row["passed"]]
    selected = min(passing) if passing else None
    payload = {
        "schema": "jlens.calibration.adjacent_layer_selection.v1",
        "rule_version": rule.version,
        "rule_digest": rule.digest,
        "candidates": [int(layer) for layer in candidates],
        "evaluated_layers": [row["layer"] for row in rows if row["evaluated"]],
        "passing_layers": passing,
        "selected_layer": selected,
        "selection_basis": "lowest physical layer passing every frozen clause",
        "best_looking_failure_considered": False,
        "table": rows,
        "verdict": ADJACENT_LENS_GO if selected is not None else ADJACENT_LENS_NO_GO,
        "statement": (
            f"L{selected} is the lowest candidate in {list(candidates)} that "
            "passed every frozen confirmation clause on the untouched set."
            if selected is not None
            else (
                f"No candidate in {list(candidates)} passed every frozen "
                "confirmation clause on the untouched set. The complete table is "
                "recorded. No layer is promoted on the strength of being closest."
            )
        ),
    }
    payload["selection_checksum"] = payload_checksum(payload)
    return payload


def adjacent_lens_verdict(
    selection: Mapping,
    *,
    confirmation_manifest: Mapping,
    untouched_audit: Mapping,
    source_layer_record: Mapping,
    gate: CalibrationGate = ADJACENT_CONFIRMATION_GATE,
) -> dict:
    """``ADJACENT_LENS_VALIDITY``: is there a confirmed lens in this interval?

    Validity clauses come before the layer: a selection made against a set that
    was not proved untouched, or against a lens whose accumulation cannot be
    shown to be new, is not a confirmation of anything.
    """
    clauses = [
        {
            "clause": "confirmation_set_proved_untouched",
            "passed": bool(untouched_audit.get("untouched")),
            "detail": (
                f"{untouched_audit.get('n_exact_hits')} exact, "
                f"{untouched_audit.get('n_near_hits')} near, "
                f"{untouched_audit.get('n_internal_duplicates')} internal "
                f"collision(s) against {untouched_audit.get('required_disjoint_from')}"
            ),
        },
        {
            "clause": "confirmation_size_not_reduced",
            "passed": int(confirmation_manifest.get("size", 0))
            == N_CONFIRMATION_PROMPTS,
            "detail": (
                f"{confirmation_manifest.get('size')} of {N_CONFIRMATION_PROMPTS}"
            ),
        },
        {
            "clause": "new_jacobian_accumulation_for_new_source_layers",
            "passed": bool(source_layer_record.get("disjoint")),
            "detail": (
                f"candidates {source_layer_record.get('candidate_layers')} vs "
                f"parent {source_layer_record.get('parent_source_layers')}; "
                f"overlap {source_layer_record.get('overlap')}"
            ),
        },
        {
            "clause": "gate_digest_unchanged",
            "passed": gate.digest == ADJACENT_CONFIRMATION_GATE.digest,
            "detail": f"{gate.digest}",
        },
        {
            "clause": "selection_rule_declared_before_confirmation",
            "passed": bool(ADJACENT_SELECTION_RULE.declared_before_confirmation_opened),
            "detail": ADJACENT_SELECTION_RULE.digest,
        },
        {
            "clause": "all_candidates_evaluated",
            "passed": sorted(selection.get("evaluated_layers") or [])
            == sorted(int(layer) for layer in ADJACENT_CANDIDATE_LAYERS),
            "detail": (
                f"evaluated {selection.get('evaluated_layers')} of "
                f"{list(ADJACENT_CANDIDATE_LAYERS)}"
            ),
        },
    ]
    failed = [clause["clause"] for clause in clauses if not clause["passed"]]
    selected = selection.get("selected_layer")
    if failed:
        verdict = ADJACENT_LENS_NO_GO
        rationale = (
            f"{len(failed)} validity clause(s) did not hold: {failed}. No layer "
            "is reported as confirmed, because the confirmation itself is not "
            "established."
        )
        selected = None
    elif selected is None:
        verdict = ADJACENT_LENS_NO_GO
        rationale = str(selection.get("statement"))
    else:
        verdict = ADJACENT_LENS_GO
        rationale = (
            f"Physical layer {selected} passed every clause of the frozen "
            f"gate ({gate.version}) on a confirmation set proved untouched for "
            "these five candidates. It is the lowest candidate to do so."
        )
    payload = {
        "schema": "jlens.calibration.adjacent_lens_verdict.v1",
        "verdict_name": "ADJACENT_LENS_VALIDITY",
        "verdict": verdict,
        "selected_layer": selected,
        "candidates": list(ADJACENT_CANDIDATE_LAYERS),
        "passing_layers": list(selection.get("passing_layers") or []),
        "validity_clauses": clauses,
        "failed_validity_clauses": failed,
        "gate_version": gate.version,
        "gate_digest": gate.digest,
        "selection_rule_digest": ADJACENT_SELECTION_RULE.digest,
        "protocol_digest": ADJACENT_PROTOCOL.digest,
        "table": selection.get("table"),
        "rationale": rationale,
    }
    payload["verdict_checksum"] = payload_checksum(payload)
    return payload


# ---------------------------------------------------------------- the budget

#: The operator's own measurement: 100 prompts over an eight-layer grid in 7.1
#: minutes on one L4. The cost driver is the backward span, so a five-layer grid
#: whose shallowest layer is 27 is cheaper per prompt than the eight-layer grid
#: whose shallowest was 8 — this scales by *measured* span, never by layer count.
OBSERVED_SCALE100_MINUTES = 7.1
OBSERVED_SCALE100_PROMPTS = 100
OBSERVED_BACKWARD_SPAN = 41 - 8
ADJACENT_UNCERTAINTY_BAND = (0.9, 1.6)


def adjacent_budget(
    *,
    scale: int = ADJACENT_FITTING_SCALE,
    layers: Sequence[int] = ADJACENT_CANDIDATE_LAYERS,
    target_layer: int = 41,
    band: tuple[float, float] = ADJACENT_UNCERTAINTY_BAND,
) -> dict:
    """Fitting cost for the five new source layers at one scale."""
    span = int(target_layer) - min(int(layer) for layer in layers)
    scaling = span / OBSERVED_BACKWARD_SPAN
    per_prompt = (OBSERVED_SCALE100_MINUTES * 60.0 / OBSERVED_SCALE100_PROMPTS) * scaling
    central = per_prompt * int(scale)
    payload = {
        "schema": "jlens.calibration.adjacent_budget.v1",
        "scale": int(scale),
        "layers": [int(layer) for layer in layers],
        "target_layer": int(target_layer),
        "backward_span_blocks": span,
        "observed_anchor": {
            "minutes": OBSERVED_SCALE100_MINUTES,
            "n_prompts": OBSERVED_SCALE100_PROMPTS,
            "backward_span_blocks": OBSERVED_BACKWARD_SPAN,
            "device": "one NVIDIA L4",
        },
        "seconds_per_prompt": round(per_prompt, 3),
        "fit_hours_central": round(central / 3600.0, 3),
        "fit_hours_low": round(central * band[0] / 3600.0, 3),
        "fit_hours_high": round(central * band[1] / 3600.0, 3),
        "uncertainty_band": list(band),
        "uncertainty_note": (
            "one observation, one runtime, one prompt-length distribution. The "
            "high end is the planning number."
        ),
        "n_forward_passes": int(scale),
        "n_backward_passes": int(scale) * span,
    }
    payload["budget_checksum"] = payload_checksum(payload)
    return payload


def format_adjacent_budget(budget: Mapping) -> str:
    """The block printed before any fitting switch may be set."""
    return f"""\
ADJACENT-LAYER FITTING BUDGET — scale {budget['scale']}, layers {budget['layers']}

  backward span        {budget['backward_span_blocks']} blocks (L{min(budget['layers'])} -> L{budget['target_layer']})
  forward passes       {budget['n_forward_passes']:,}
  backward passes      {budget['n_backward_passes']:,}
  seconds per prompt   {budget['seconds_per_prompt']:.2f}

  L4 fitting time      {budget['fit_hours_low']:.2f} - {budget['fit_hours_high']:.2f} h
                       (central {budget['fit_hours_central']:.2f} h)

  anchor               {budget['observed_anchor']['n_prompts']} prompts in \
{budget['observed_anchor']['minutes']:.1f} min on {budget['observed_anchor']['device']},
                       backward span {budget['observed_anchor']['backward_span_blocks']} blocks
  {budget['uncertainty_note']}

This is ONE bounded work unit's worth of planning, not a promise. The fit
checkpoints every bounded batch, so an interrupted session loses at most that
batch — never the hours before it.
"""
