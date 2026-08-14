# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Which question did each completed number actually answer?

A repository-wide audit of endpoint semantics. Every active scientific module,
notebook and canonical completed report is classified into exactly one of the
six classes in :data:`~jlens.mmpilot.full_vocabulary.ENDPOINT_CLASSES`, and the
classification is **traced to the function that computes the field** rather than
read off the report's prose. Prose is what is being audited; it cannot also be
the evidence.

Two artifacts come out of it:

``endpoint_semantics_audit.json`` / ``.md``
    The classification, per module, per computing function, with the exact
    fields each one produces.

the claim ledger
    One row per active headline claim, carrying the run it came from, the
    report checksum it is pinned to, the function that computed it, the
    candidate universe, whether tokens were appended, whether the global
    vocabulary was ever consulted, whether anything was generated, the
    interpretation the mathematics justifies, the stronger wording that is
    therefore prohibited, whether full-vocabulary revalidation is required, and
    whether the claim survives unchanged, survives with narrower wording, or is
    unsupported.

The registry is hand-traced and then **machine-verified**: :func:`verify_registry`
imports every named module and resolves every named function, so a row can never
describe a function that no longer exists. That is the whole defence against a
ledger drifting away from the code it claims to describe.

The blocking clause
===================

:func:`scan_active_sources` greps the *active* package for the four prohibited
descriptions of a restricted-candidate argmax — "full-vocabulary", "global
top-1", "the model output", "paper-comparable" — and
:func:`endpoint_audit_record` fails the audit when one appears without an
explicit endpoint qualifier on the same line or in the surrounding context.
Completed reports and historical notebooks are *not* rewritten; they are listed
as ``requires_amendment`` and answered by a versioned amendment artifact and by
the superseded-terminology index.
"""

from __future__ import annotations

import importlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from jlens.mmpilot.full_vocabulary import (
    ENDPOINT_CLASSES,
    ENDPOINT_CONDITIONAL_LOGPROB,
    ENDPOINT_ENGINEERING,
    ENDPOINT_GENERATION,
    ENDPOINT_REPRESENTATIONAL,
    ENDPOINT_RESTRICTED_CANDIDATE,
    ENDPOINT_UNRESTRICTED_NEXT_TOKEN,
)
from jlens.mmpilot.store import payload_checksum

__all__ = [
    "AUDITED_ENDPOINTS",
    "AUDIT_JSON_NAME",
    "AUDIT_MARKDOWN_NAME",
    "ENDPOINT_AUDIT_VERSION",
    "LEDGER_JSON_NAME",
    "PROHIBITED_PATTERNS",
    "SCANNED_ENDPOINT_CLASSES",
    "SURVIVES_NARROWER",
    "SURVIVES_UNCHANGED",
    "UNSUPPORTED",
    "AuditedEndpoint",
    "EndpointAuditFailed",
    "claim_ledger",
    "endpoint_audit_digest",
    "endpoint_audit_files",
    "endpoint_audit_record",
    "format_endpoint_audit_markdown",
    "scan_active_sources",
    "scanned_modules",
    "verify_registry",
]

ENDPOINT_AUDIT_VERSION = "mmpilot.endpoint_semantics_audit.v1"

AUDIT_JSON_NAME = "endpoint_semantics_audit.json"
AUDIT_MARKDOWN_NAME = "endpoint_semantics_audit.md"
LEDGER_JSON_NAME = "endpoint_claim_ledger.json"

#: Survival classes for a claim under the corrected endpoint vocabulary.
SURVIVES_UNCHANGED = "survives_unchanged"
SURVIVES_NARROWER = "survives_with_narrower_wording"
UNSUPPORTED = "unsupported"

SURVIVAL_CLASSES: tuple[str, ...] = (
    SURVIVES_UNCHANGED,
    SURVIVES_NARROWER,
    UNSUPPORTED,
)


class EndpointAuditFailed(RuntimeError):
    """An active report describes a restricted endpoint as an unrestricted one."""


# ------------------------------------------------------------------- the registry


@dataclass(frozen=True)
class AuditedEndpoint:
    """One traced endpoint: a claim, and the function that actually computes it."""

    claim: str
    study: str
    source_run: str | None
    report_checksum_pin: str | None
    module: str
    function: str
    report_fields: tuple[str, ...]
    endpoint_class: str
    candidate_universe: str
    tokens_appended: bool
    global_vocabulary_considered: bool
    generation_occurred: bool
    justified_interpretation: str
    prohibited_wording: tuple[str, ...]
    requires_full_vocabulary_revalidation: bool
    survival: str
    notes: str = ""
    aliases: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.endpoint_class not in ENDPOINT_CLASSES:
            raise ValueError(
                f"{self.claim!r}: unknown endpoint class {self.endpoint_class!r}"
            )
        if self.survival not in SURVIVAL_CLASSES:
            raise ValueError(f"{self.claim!r}: unknown survival {self.survival!r}")
        if self.endpoint_class == ENDPOINT_UNRESTRICTED_NEXT_TOKEN and not (
            self.global_vocabulary_considered
        ):
            raise ValueError(
                f"{self.claim!r} is classified unrestricted but records that the "
                "global vocabulary was never considered"
            )
        if self.endpoint_class == ENDPOINT_RESTRICTED_CANDIDATE and (
            self.global_vocabulary_considered
        ):
            raise ValueError(
                f"{self.claim!r} is classified restricted-candidate but records "
                "that the global vocabulary was considered"
            )

    def to_dict(self) -> dict:
        return {key: _plain(value) for key, value in asdict(self).items()}


def _plain(value):
    if isinstance(value, tuple):
        return list(value)
    return value


_NO_STRONGER = (
    "the model answered",
    "the model output",
    "global top-1",
    "full-vocabulary",
    "paper-comparable top-1",
)

#: The hand-traced registry. Every row was written by reading the computing
#: function, not the report it feeds. :func:`verify_registry` proves the
#: functions exist; nothing here proves a function is *correct*, only what
#: question it asks.
AUDITED_ENDPOINTS: tuple[AuditedEndpoint, ...] = (
    # ----------------------------------------------------- audio engineering
    AuditedEndpoint(
        claim="spoken audio reaches the model through its native pathway",
        study="native spoken-audio feasibility / audit",
        source_run=None,
        report_checksum_pin=None,
        module="jlens.mmpilot.audio_audit",
        function="run_audio_audit",
        report_fields=(
            "placeholder_span",
            "features_present",
            "transcript_leakage",
            "checks",
        ),
        endpoint_class=ENDPOINT_ENGINEERING,
        candidate_universe="not applicable",
        tokens_appended=False,
        global_vocabulary_considered=False,
        generation_occurred=False,
        justified_interpretation=(
            "the processor expands a recording into placeholder tokens the "
            "model consumes, and no transcript is visible to the model"
        ),
        prohibited_wording=("the model understood the speech",),
        requires_full_vocabulary_revalidation=False,
        survival=SURVIVES_UNCHANGED,
        notes="An input-path property. Unaffected by any endpoint question.",
    ),
    AuditedEndpoint(
        claim="a candidate scoring path is valid for the audio condition",
        study="native spoken-audio feasibility / audit",
        source_run=None,
        report_checksum_pin=None,
        module="jlens.mmpilot.audio_audit",
        function="check_scoring_validity",
        report_fields=("scoring_checks", "per_token_logprobs_sum_to_total"),
        endpoint_class=ENDPOINT_ENGINEERING,
        candidate_universe="predeclared audit candidates",
        tokens_appended=True,
        global_vocabulary_considered=False,
        generation_occurred=False,
        justified_interpretation=(
            "the teacher-forced sum equals its own per-token terms, so the "
            "scorer is internally consistent"
        ),
        prohibited_wording=("the model answered correctly",),
        requires_full_vocabulary_revalidation=False,
        survival=SURVIVES_UNCHANGED,
        notes="Validates the arithmetic of a scorer, not a scientific outcome.",
    ),
    # ------------------------------------------------------- capability gate
    AuditedEndpoint(
        claim="the model can read each concept out of each channel",
        study="behavioral capability gate (all multimodal studies)",
        source_run="mmaudio_native_audio_transfer_20260806T144822",
        report_checksum_pin="CANONICAL_AUDIO_REPORT_CHECKSUM",
        module="jlens.mmpilot.capability",
        function="prediction_and_margin",
        report_fields=(
            "prediction",
            "correct",
            "target_margin",
            "per_concept.accuracy",
            "retained_concepts",
        ),
        endpoint_class=ENDPOINT_RESTRICTED_CANDIDATE,
        candidate_universe="six concepts, all named in the prompt",
        tokens_appended=True,
        global_vocabulary_considered=False,
        generation_occurred=False,
        justified_interpretation=(
            "in a six-way forced choice whose options are listed in the prompt, "
            "the target concept's complete-sequence likelihood exceeded the "
            "other five"
        ),
        prohibited_wording=(
            "the model identified the concept",
            "the model answered",
            *_NO_STRONGER,
        ),
        requires_full_vocabulary_revalidation=False,
        survival=SURVIVES_NARROWER,
        notes=(
            "Narrower multiple-choice evidence. It gates admissibility and was "
            "never claimed as output; the wording 'can read out' is tightened to "
            "'can select in a six-way forced choice'."
        ),
    ),
    AuditedEndpoint(
        claim="capability accuracy per (concept, modality) clears 70%",
        study="behavioral capability gate",
        source_run="mmaudio_native_audio_transfer_20260806T144822",
        report_checksum_pin="CANONICAL_AUDIO_REPORT_CHECKSUM",
        module="jlens.mmpilot.capability",
        function="capability_summary",
        report_fields=("per_concept", "retained_concepts", "threshold"),
        endpoint_class=ENDPOINT_RESTRICTED_CANDIDATE,
        candidate_universe="six concepts, all named in the prompt",
        tokens_appended=True,
        global_vocabulary_considered=False,
        generation_occurred=False,
        justified_interpretation=(
            "forced-choice accuracy over 8 samples per cell against a frozen "
            "threshold"
        ),
        prohibited_wording=_NO_STRONGER,
        requires_full_vocabulary_revalidation=False,
        survival=SURVIVES_NARROWER,
    ),
    # -------------------------------------------- J-lens calibration and gate
    AuditedEndpoint(
        claim="the J-lens reproduces the model's own next token",
        study="research-grade J-lens calibration",
        source_run=None,
        report_checksum_pin=None,
        module="jlens.calibration.gate",
        function="ordinary_next_token_argmax",
        report_fields=("target_token_id",),
        endpoint_class=ENDPOINT_UNRESTRICTED_NEXT_TOKEN,
        candidate_universe="the entire vocabulary",
        tokens_appended=False,
        global_vocabulary_considered=True,
        generation_occurred=False,
        justified_interpretation=(
            "argmax over the model's complete next-token logits at the final "
            "prompt position — the same endpoint the corrected causal study uses"
        ),
        prohibited_wording=(),
        requires_full_vocabulary_revalidation=False,
        survival=SURVIVES_UNCHANGED,
        notes=(
            "Unaffected. The lens work always measured the unrestricted "
            "endpoint; only the behavioral/causal work did not."
        ),
    ),
    AuditedEndpoint(
        claim="lens rank of the model's true next token beats its controls",
        study="research-grade J-lens calibration / confirmation",
        source_run=None,
        report_checksum_pin=None,
        module="jlens.calibration.gate",
        function="evaluate_calibration_layer",
        report_fields=("mrr", "top1_agreement", "median_rank", "eligible"),
        endpoint_class=ENDPOINT_UNRESTRICTED_NEXT_TOKEN,
        candidate_universe="the entire vocabulary",
        tokens_appended=False,
        global_vocabulary_considered=True,
        generation_occurred=False,
        justified_interpretation=(
            "full-vocabulary rank statistics of the model's own argmax token "
            "under the lens, against noise and wrong-layer controls"
        ),
        prohibited_wording=(),
        requires_full_vocabulary_revalidation=False,
        survival=SURVIVES_UNCHANGED,
    ),
    AuditedEndpoint(
        claim="the published lens confirms on held-out prompts",
        study="native direct readout / confirmatory validation",
        source_run=None,
        report_checksum_pin=None,
        module="jlens.native_readout",
        function="evaluate_confirmatory",
        report_fields=("top1_agreement", "mrr", "median_rank", "verdict"),
        endpoint_class=ENDPOINT_UNRESTRICTED_NEXT_TOKEN,
        candidate_universe="the entire vocabulary",
        tokens_appended=False,
        global_vocabulary_considered=True,
        generation_occurred=False,
        justified_interpretation=(
            "agreement and rank of the model's actual final-layer top-1 token "
            "under the lens, over the full vocabulary"
        ),
        prohibited_wording=(),
        requires_full_vocabulary_revalidation=False,
        survival=SURVIVES_UNCHANGED,
    ),
    # ----------------------------------------------- representational J-space
    AuditedEndpoint(
        claim="J-space codes retrieve the matching cross-modal example",
        study="representational retrieval and shuffled controls",
        source_run="mmaudio_native_audio_transfer_20260806T144822",
        report_checksum_pin="CANONICAL_AUDIO_REPORT_CHECKSUM",
        module="jlens.mmpilot.jspace",
        function="retrieval_metrics",
        report_fields=("top1_accuracy", "mrr", "n_queries"),
        endpoint_class=ENDPOINT_REPRESENTATIONAL,
        candidate_universe="the retrieval gallery of held-out examples",
        tokens_appended=False,
        global_vocabulary_considered=False,
        generation_occurred=False,
        justified_interpretation=(
            "activations of one modality are nearer their same-concept "
            "counterparts in another modality than a shuffled assignment is"
        ),
        prohibited_wording=(
            "the model output the concept",
            *_NO_STRONGER,
        ),
        requires_full_vocabulary_revalidation=False,
        survival=SURVIVES_UNCHANGED,
        notes=(
            "Representational evidence. It never was an output claim, so the "
            "endpoint correction does not touch it."
        ),
    ),
    AuditedEndpoint(
        claim="retrieval beats its shuffled-label control",
        study="representational retrieval and shuffled controls",
        source_run="mmaudio_native_audio_transfer_20260806T144822",
        report_checksum_pin="CANONICAL_AUDIO_REPORT_CHECKSUM",
        module="jlens.mmpilot.jspace",
        function="shuffled_label_control",
        report_fields=("p95_top1_accuracy", "mean_top1_accuracy"),
        endpoint_class=ENDPOINT_REPRESENTATIONAL,
        candidate_universe="the retrieval gallery under permuted labels",
        tokens_appended=False,
        global_vocabulary_considered=False,
        generation_occurred=False,
        justified_interpretation=(
            "the retrieval statistic exceeds the 95th percentile of the same "
            "statistic under permuted labels"
        ),
        prohibited_wording=_NO_STRONGER,
        requires_full_vocabulary_revalidation=False,
        survival=SURVIVES_UNCHANGED,
    ),
    # ---------------------------------------------------- the original pilot
    AuditedEndpoint(
        claim="the multimodal pilot's causal cells moved the target",
        study="original multimodal J-space pilot",
        source_run=None,
        report_checksum_pin=None,
        module="jlens.mmpilot.causal",
        function="run_condition",
        report_fields=(
            "signed_target_effect",
            "target_score_change",
            "target_margin_change",
            "prediction_changed",
        ),
        endpoint_class=ENDPOINT_CONDITIONAL_LOGPROB,
        candidate_universe="the predeclared concept candidates",
        tokens_appended=True,
        global_vocabulary_considered=False,
        generation_occurred=False,
        justified_interpretation=(
            "the target answer's teacher-forced conditional log-probability "
            "moved in the intended direction relative to the clean run"
        ),
        prohibited_wording=(
            "the intervention changed the model's answer",
            *_NO_STRONGER,
        ),
        requires_full_vocabulary_revalidation=True,
        survival=SURVIVES_NARROWER,
        notes=(
            "`prediction_changed` is a restricted-candidate flip, not an output "
            "change. Genuine conditional-likelihood effect; not autonomous output."
        ),
    ),
    AuditedEndpoint(
        claim="the pilot's go/no-go criteria",
        study="original multimodal J-space pilot",
        source_run=None,
        report_checksum_pin=None,
        module="jlens.mmpilot.report",
        function="evaluate_criteria",
        report_fields=("criteria", "decision"),
        endpoint_class=ENDPOINT_CONDITIONAL_LOGPROB,
        candidate_universe="the predeclared concept candidates",
        tokens_appended=True,
        global_vocabulary_considered=False,
        generation_occurred=False,
        justified_interpretation=(
            "aggregate conditional-log-probability effects and their controls "
            "cleared frozen thresholds"
        ),
        prohibited_wording=_NO_STRONGER,
        requires_full_vocabulary_revalidation=True,
        survival=SURVIVES_NARROWER,
    ),
    # ------------------------------------------ robustness and localization
    AuditedEndpoint(
        claim="text<->image transfer replicates on image-unique data",
        study="robustness study",
        source_run=None,
        report_checksum_pin=None,
        module="jlens.mmpilot.robustness",
        function="evaluate_causal_cells",
        report_fields=("mean_signed_target_effect", "passes", "reasons"),
        endpoint_class=ENDPOINT_CONDITIONAL_LOGPROB,
        candidate_universe="the predeclared concept candidates",
        tokens_appended=True,
        global_vocabulary_considered=False,
        generation_occurred=False,
        justified_interpretation=(
            "controlled conditional-log-probability effects replicate on a "
            "population with one group per image"
        ),
        prohibited_wording=_NO_STRONGER,
        requires_full_vocabulary_revalidation=True,
        survival=SURVIVES_NARROWER,
    ),
    AuditedEndpoint(
        claim="the causal effect is localized to particular layers",
        study="layer localization study",
        source_run=None,
        report_checksum_pin=None,
        module="jlens.mmlocalize.verdict",
        function="localization_verdict",
        report_fields=("per_layer", "verdict", "paired_comparison"),
        endpoint_class=ENDPOINT_CONDITIONAL_LOGPROB,
        candidate_universe="the predeclared concept candidates",
        tokens_appended=True,
        global_vocabulary_considered=False,
        generation_occurred=False,
        justified_interpretation=(
            "the size of the conditional-log-probability effect differs "
            "between layers on paired inputs"
        ),
        prohibited_wording=_NO_STRONGER,
        requires_full_vocabulary_revalidation=True,
        survival=SURVIVES_NARROWER,
    ),
    # ------------------------------------------- native spoken-audio transfer
    AuditedEndpoint(
        claim="THREE_MODALITY_GO — cross-modal causal transfer including speech",
        study="native spoken-audio transfer (canonical three-modality run)",
        source_run="mmaudio_native_audio_transfer_20260806T144822",
        report_checksum_pin="CANONICAL_AUDIO_REPORT_CHECKSUM",
        module="jlens.mmpilot.tri_modal",
        function="causal_transfer_verdict",
        report_fields=(
            "verdict",
            "rationale",
            "audio_cells_supporting_a_claim",
            "concepts_transferring_both_audio_directions",
        ),
        endpoint_class=ENDPOINT_CONDITIONAL_LOGPROB,
        candidate_universe="six concepts, all named in the prompt",
        tokens_appended=True,
        global_vocabulary_considered=False,
        generation_occurred=False,
        justified_interpretation=(
            "a source-derived J-space direction applied at one layer changed the "
            "target concept's conditional log-probability in the intended "
            "direction, against matched random, unrelated, zero and raw-residual "
            "controls, in both directions for at least one admissible concept"
        ),
        prohibited_wording=(
            "the model said the swapped concept",
            "the model's answer changed",
            *_NO_STRONGER,
        ),
        requires_full_vocabulary_revalidation=True,
        survival=SURVIVES_NARROWER,
        notes=(
            "A genuine controlled conditional-likelihood effect. It is not "
            "evidence that any output token changed: the full vocabulary was "
            "never consulted. Corrected label: CONTROLLED_TARGET_LOGPROB_EFFECT "
            "plus FULL_VOCABULARY_NOT_EVALUATED."
        ),
        aliases=("L35_CAUSAL_TRANSFER", "THREE_MODALITY_GO"),
    ),
    AuditedEndpoint(
        claim="the capability-filtered v2 amendment's verdicts",
        study="native spoken-audio transfer, amended",
        source_run="mmaudio_native_audio_transfer_20260806T144822",
        report_checksum_pin="CANONICAL_AUDIO_AMENDED_SUMMARY_CHECKSUM",
        module="jlens.mmpilot.amend",
        function="rebuild_verdicts",
        report_fields=("primary_causal", "replication", "overall"),
        endpoint_class=ENDPOINT_CONDITIONAL_LOGPROB,
        candidate_universe="six concepts, all named in the prompt",
        tokens_appended=True,
        global_vocabulary_considered=False,
        generation_occurred=False,
        justified_interpretation=(
            "the same conditional-log-probability effects, re-read under the "
            "capability-admissibility rule; no measurement was recomputed"
        ),
        prohibited_wording=_NO_STRONGER,
        requires_full_vocabulary_revalidation=True,
        survival=SURVIVES_NARROWER,
    ),
    # ------------------------------------------- single-layer reasoning swaps
    AuditedEndpoint(
        claim="hidden-intermediate onset is localized among confirmed layers",
        study="paper-style single-layer reasoning swap (v2)",
        source_run=None,
        report_checksum_pin=None,
        module="jlens.mmpilot.paper_reasoning_swap",
        function="paper_onset_verdict_v2",
        report_fields=("classification", "per_direction", "verdict"),
        endpoint_class=ENDPOINT_RESTRICTED_CANDIDATE,
        candidate_universe="two answers (two/four) or two identities (bird/cat)",
        tokens_appended=True,
        global_vocabulary_considered=False,
        generation_occurred=False,
        justified_interpretation=(
            "after the exchange the target answer outranked the one other "
            "supplied candidate more often than every matched control did"
        ),
        prohibited_wording=(
            "the model output the target answer",
            "paper-comparable top-1",
            *_NO_STRONGER,
        ),
        requires_full_vocabulary_revalidation=True,
        survival=SURVIVES_NARROWER,
    ),
    # -------------------------------------------------- the L33-L40 follow-up
    AuditedEndpoint(
        claim="target answer becomes top-1 after the band swap",
        study="L33-L40 validated-band follow-up",
        source_run="band3340_real_2a72bda9b4ba",
        report_checksum_pin="BAND_FOLLOWUP_REPORT_CHECKSUM",
        module="jlens.mmpilot.band_swap",
        function="band_trial_record",
        report_fields=(
            "prediction",
            "target_rank",
            "n_candidates_scored",
            "candidate_ranking",
        ),
        endpoint_class=ENDPOINT_RESTRICTED_CANDIDATE,
        candidate_universe="exactly two answers per readout",
        tokens_appended=True,
        global_vocabulary_considered=False,
        generation_occurred=False,
        justified_interpretation=(
            "`target_rank` is a rank among two supplied candidates; "
            "`prediction` is which of the two scored higher"
        ),
        prohibited_wording=(
            "the target token became the model's top output",
            *_NO_STRONGER,
        ),
        requires_full_vocabulary_revalidation=True,
        survival=SURVIVES_NARROWER,
        notes=(
            "The completed report measured whether `two` outranked `four`, not "
            "whether `two` was the global next-token argmax."
        ),
    ),
    AuditedEndpoint(
        claim="L33_L40_VALIDATED_BAND_FOLLOWUP_ALPHA2_SENSITIVITY_ONLY",
        study="L33-L40 validated-band follow-up",
        source_run="band3340_real_2a72bda9b4ba",
        report_checksum_pin="BAND_FOLLOWUP_REPORT_CHECKSUM",
        module="jlens.mmpilot.band_swap",
        function="band_reasoning_verdict",
        report_fields=(
            "verdict",
            "restricted_candidate_preference",
            "paper_comparable",
            "alpha2_sensitivity",
            "modality_extension",
        ),
        endpoint_class=ENDPOINT_RESTRICTED_CANDIDATE,
        candidate_universe="exactly two answers per readout",
        tokens_appended=True,
        global_vocabulary_considered=False,
        generation_occurred=False,
        justified_interpretation=(
            "at alpha=2 only, the swap made the target answer the preferred one "
            "of two supplied candidates more often than every alpha=2-matched "
            "control; alpha=1 did not"
        ),
        prohibited_wording=(
            "paper-comparable",
            "Anthropic's top-1 endpoint",
            *_NO_STRONGER,
        ),
        requires_full_vocabulary_revalidation=True,
        survival=SURVIVES_NARROWER,
        notes=(
            "Corrected label: RESTRICTED_CANDIDATE_PREFERENCE_GO at alpha=2 "
            "sensitivity only, plus FULL_VOCABULARY_NOT_EVALUATED. The numbers "
            "are unchanged; the `paper_comparable` field name is superseded by "
            "`restricted_candidate_preference`."
        ),
        aliases=("paper_comparable",),
    ),
    AuditedEndpoint(
        claim="every physical band layer was patched on every trial",
        study="L33-L40 validated-band follow-up",
        source_run="band3340_real_2a72bda9b4ba",
        report_checksum_pin="BAND_FOLLOWUP_REPORT_CHECKSUM",
        module="jlens.mmpilot.validated_band_followup",
        function="assert_band_hook_integrity",
        report_fields=("layers_patched", "n_positions_patched", "hook_integrity"),
        endpoint_class=ENDPOINT_ENGINEERING,
        candidate_universe="not applicable",
        tokens_appended=False,
        global_vocabulary_considered=False,
        generation_occurred=False,
        justified_interpretation=(
            "the hooks fired at exactly the requested layers and at exactly the "
            "prompt positions the rule names"
        ),
        prohibited_wording=(),
        requires_full_vocabulary_revalidation=False,
        survival=SURVIVES_UNCHANGED,
    ),
    AuditedEndpoint(
        claim="L33-L40 is the admissible band; L32 is excluded",
        study="band corrected control",
        source_run="bandcorr_real_eb5b00f135e4",
        report_checksum_pin="CORRECTED_VALIDATION_REPORT_CHECKSUM",
        module="jlens.mmpilot.band_control",
        function="corrected_band_verdict",
        report_fields=("per_layer", "verdict", "admissible_layers"),
        endpoint_class=ENDPOINT_UNRESTRICTED_NEXT_TOKEN,
        candidate_universe="the entire vocabulary",
        tokens_appended=False,
        global_vocabulary_considered=True,
        generation_occurred=False,
        justified_interpretation=(
            "lens readout quality against noise, wrong-layer and coverage "
            "clauses, all computed over full-vocabulary rank statistics"
        ),
        prohibited_wording=(),
        requires_full_vocabulary_revalidation=False,
        survival=SURVIVES_UNCHANGED,
        notes=(
            "Lens validation, not behaviour. Unaffected by the endpoint "
            "correction, and it is why the L33-L40 lenses may still be reused."
        ),
    ),
    # ------------------------------------------------ convergence and readout
    AuditedEndpoint(
        claim="the layer has/has not converged to its output",
        study="output convergence audit / L32 resolution",
        source_run=None,
        report_checksum_pin=None,
        module="jlens.mmpilot.convergence",
        function="direct_readout_row",
        report_fields=("target_rank", "tie_aware_ranks", "top1", "entropy"),
        endpoint_class=ENDPOINT_UNRESTRICTED_NEXT_TOKEN,
        candidate_universe="the entire vocabulary",
        tokens_appended=False,
        global_vocabulary_considered=True,
        generation_occurred=False,
        justified_interpretation=(
            "full-vocabulary logits produced by the model's own output head "
            "applied to an intermediate residual"
        ),
        prohibited_wording=(),
        requires_full_vocabulary_revalidation=False,
        survival=SURVIVES_UNCHANGED,
        notes=(
            "This study already did what the causal studies did not: it read "
            "the whole distribution."
        ),
    ),
    AuditedEndpoint(
        claim="convergence classification per layer",
        study="output convergence audit",
        source_run=None,
        report_checksum_pin=None,
        module="jlens.mmpilot.convergence",
        function="classify_layer",
        report_fields=("classification", "rates", "controls"),
        endpoint_class=ENDPOINT_UNRESTRICTED_NEXT_TOKEN,
        candidate_universe="the entire vocabulary",
        tokens_appended=False,
        global_vocabulary_considered=True,
        generation_occurred=False,
        justified_interpretation=(
            "rates of full-vocabulary top-1 agreement and rank, against "
            "shuffled/permuted controls"
        ),
        prohibited_wording=(),
        requires_full_vocabulary_revalidation=False,
        survival=SURVIVES_UNCHANGED,
    ),
    # ------------------------------------------------- the corrected endpoint
    AuditedEndpoint(
        claim="the target token is the global next-token argmax",
        study="full-vocabulary causal validation (this study)",
        source_run=None,
        report_checksum_pin=None,
        module="jlens.mmpilot.full_vocabulary",
        function="score_unrestricted_next_token",
        report_fields=(
            "global_argmax_token_id",
            "target_rank",
            "target_is_unique_global_top1",
            "top_tokens",
        ),
        endpoint_class=ENDPOINT_UNRESTRICTED_NEXT_TOKEN,
        candidate_universe="the entire vocabulary; no candidate list is supplied",
        tokens_appended=False,
        global_vocabulary_considered=True,
        generation_occurred=False,
        justified_interpretation=(
            "the complete next-token distribution at the final prompt position "
            "of the untouched prompt, with the intervention applied during that "
            "same forward pass"
        ),
        prohibited_wording=(),
        requires_full_vocabulary_revalidation=False,
        survival=SURVIVES_UNCHANGED,
        notes="Not yet run against Gemma. Implementation only.",
    ),
    AuditedEndpoint(
        claim="the model greedily writes the target answer",
        study="full-vocabulary causal validation (this study)",
        source_run=None,
        report_checksum_pin=None,
        module="jlens.mmpilot.full_vocabulary",
        function="greedy_generate",
        report_fields=(
            "generated_text",
            "generated_token_ids",
            "exact_answer_match",
        ),
        endpoint_class=ENDPOINT_GENERATION,
        candidate_universe="not applicable",
        tokens_appended=False,
        global_vocabulary_considered=True,
        generation_occurred=True,
        justified_interpretation=(
            "a deterministic temperature-0 continuation of the prompt under the "
            "same intervention"
        ),
        prohibited_wording=(
            "greedy text alone establishes the causal claim",
        ),
        requires_full_vocabulary_revalidation=False,
        survival=SURVIVES_UNCHANGED,
        notes="Secondary demonstration. No verdict may rest on it.",
    ),
)


def verify_registry(
    endpoints: Sequence[AuditedEndpoint] = AUDITED_ENDPOINTS,
) -> list[dict]:
    """Resolve every registry row's module and function against the live code.

    A row naming a function that no longer exists is a ledger that has drifted
    from the code, which is exactly the failure this audit exists to prevent.

    Raises:
        EndpointAuditFailed: On the first unresolvable module or function.
    """
    resolved: list[dict] = []
    for row in endpoints:
        try:
            module = importlib.import_module(row.module)
        except ImportError as error:  # pragma: no cover - defensive
            raise EndpointAuditFailed(
                f"claim {row.claim!r} names module {row.module!r}, which does "
                f"not import: {error}"
            ) from error
        target = getattr(module, row.function, None)
        if target is None:
            raise EndpointAuditFailed(
                f"claim {row.claim!r} names {row.module}.{row.function}, which "
                "does not exist. The claim ledger must trace to a real "
                "computing function"
            )
        resolved.append(
            {
                "claim": row.claim,
                "module": row.module,
                "function": row.function,
                "qualified_name": f"{row.module}.{row.function}",
                "callable": callable(target),
                "doc_first_line": (target.__doc__ or "").strip().splitlines()[:1],
            }
        )
    return resolved


# ----------------------------------------------------------------- the scanner

#: The four descriptions a restricted-candidate argmax may never carry.
#:
#: ``model_output`` is deliberately narrow. "The model outputs text; image and
#: audio are input modalities only" is a true statement about modalities, not a
#: claim about an endpoint, and a scanner that cannot tell those apart produces
#: noise instead of findings.
PROHIBITED_PATTERNS: tuple[tuple[str, str], ...] = (
    ("full_vocabulary", r"full[-\s]vocabular\w*"),
    ("global_top1", r"global\s+top-?1"),
    (
        "model_output",
        r"the model'?s\s+(output|answer)\b"
        r"|the model\s+(answered|said|wrote|emitted|produced the)\b"
        r"|\bmodel output token\b",
    ),
    ("paper_comparable", r"paper[-\s]comparable"),
)

#: Wording that makes a hit explicitly qualified rather than an overclaim.
QUALIFIER_PATTERNS: tuple[str, ...] = (
    r"restricted[-\s]candidate",
    r"candidate[-\s]set",
    r"candidate universe",
    r"predeclared,? externally scored",
    r"supplied candidates",
    r"is_paper_comparable",
    r"is_not_the_model_output",
    r"NOT? evaluated",
    r"not\s+(a\s+)?(the\s+)?(global|full[-\s]vocabular)",
    r"full_vocabulary_evaluated",
    r"forced[-\s]choice",
    r"deprecated",
    r"superseded",
    r"prohibited",
    r"must not",
    r"never",
    r"unrestricted",
    r"ENDPOINT_",
)

#: Study wrappers that re-label or re-report a restricted-candidate endpoint
#: without themselves computing one. They are in scope for the scan even though
#: the registry classifies their audited function as engineering.
ADDITIONAL_SCANNED_MODULES: tuple[str, ...] = (
    "jlens.mmpilot.validated_band_followup",
    "jlens.mmpilot.l32_followup",
    "jlens.mmpilot.l32_reporting",
    "jlens.mmpilot.preconvergence",
)

#: Endpoint classes whose modules the blocking scan applies to. A module that
#: only ever computes a genuine full-vocabulary statistic (calibration, native
#: readout, the convergence audit) is *entitled* to the phrase
#: "full-vocabulary", and scanning it would produce noise, not findings.
SCANNED_ENDPOINT_CLASSES: tuple[str, ...] = (
    ENDPOINT_RESTRICTED_CANDIDATE,
    ENDPOINT_CONDITIONAL_LOGPROB,
)

_QUALIFIER = re.compile("|".join(QUALIFIER_PATTERNS), re.IGNORECASE)


def scanned_modules(
    endpoints: Sequence[AuditedEndpoint] = AUDITED_ENDPOINTS,
) -> tuple[str, ...]:
    """Exactly the modules whose claims the blocking clause is about.

    Derived from the registry rather than listed by hand: a module that
    computes a restricted-candidate or conditional-log-probability endpoint is
    the kind of module that can call one of those a full-vocabulary result, and
    those are the modules the audit refuses to let do so.
    """
    names = {
        row.module
        for row in endpoints
        if row.endpoint_class in SCANNED_ENDPOINT_CLASSES
    }
    names.update(ADDITIONAL_SCANNED_MODULES)
    return tuple(sorted(names))


def scan_active_sources(
    root: str | Path,
    *,
    modules: Iterable[str] | None = None,
    context_lines: int = 3,
) -> dict:
    """Grep the restricted-endpoint modules for prohibited descriptions.

    A hit is an ``overclaim`` only when neither its own line nor the
    ``context_lines`` around it carries an explicit endpoint qualifier. That
    two-sided rule is deliberate: a sentence that says "this is *not* the
    model's output" contains the prohibited phrase and is exactly the wording
    the audit wants to see.
    """
    base = Path(root)
    names = tuple(modules) if modules is not None else scanned_modules()
    findings: list[dict] = []
    n_files = 0
    missing: list[str] = []
    for name in names:
        path = base / Path(*name.split("."))
        path = path.with_suffix(".py")
        if not path.exists():
            missing.append(name)
            continue
        posix = path.relative_to(base).as_posix()
        n_files += 1
        lines = path.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            for label, pattern in PROHIBITED_PATTERNS:
                if not re.search(pattern, line, re.IGNORECASE):
                    continue
                low = max(0, index - context_lines)
                high = min(len(lines), index + context_lines + 1)
                context = "\n".join(lines[low:high])
                qualified = bool(_QUALIFIER.search(context))
                findings.append(
                    {
                        "path": posix,
                        "line": index + 1,
                        "pattern": label,
                        "text": line.strip(),
                        "qualified": qualified,
                        "classification": "qualified" if qualified else "overclaim",
                    }
                )
    overclaims = [row for row in findings if row["classification"] == "overclaim"]
    return {
        "scanned_modules": list(names),
        "modules_not_found": missing,
        "scan_rule": (
            "only modules the registry classifies as restricted-candidate or "
            "conditional-log-probability endpoints, plus their study wrappers"
        ),
        "n_files_scanned": n_files,
        "n_hits": len(findings),
        "n_overclaims": len(overclaims),
        "overclaims": overclaims,
        "qualified_hits": [
            row for row in findings if row["classification"] == "qualified"
        ],
        "passed": not overclaims,
    }


# ------------------------------------------------------------ the claim ledger


def claim_ledger(
    endpoints: Sequence[AuditedEndpoint] = AUDITED_ENDPOINTS,
    *,
    report_checksums: Mapping[str, str] | None = None,
) -> dict:
    """The machine-readable ledger: one row per active headline claim.

    Args:
        report_checksums: Pin name -> checksum, from the study's required
            configuration. A row whose pin is configured carries the checksum;
            a row whose pin is empty carries ``None`` and
            ``source_report_checksum_configured = False``, which the study's
            preflight refuses to proceed on. A checksum is never taken from the
            run being verified.
    """
    pins = dict(report_checksums or {})
    rows = []
    for row in endpoints:
        pin = row.report_checksum_pin
        checksum = pins.get(pin) if pin else None
        rows.append(
            {
                "claim": row.claim,
                "study": row.study,
                "aliases": list(row.aliases),
                "source_run": row.source_run,
                "source_report_checksum_pin": pin,
                "source_report_checksum": checksum or None,
                "source_report_checksum_configured": bool(checksum),
                "computing_function": f"{row.module}.{row.function}",
                "report_fields": list(row.report_fields),
                "endpoint_class": row.endpoint_class,
                "candidate_universe": row.candidate_universe,
                "tokens_appended": row.tokens_appended,
                "global_vocabulary_considered": row.global_vocabulary_considered,
                "generation_occurred": row.generation_occurred,
                "justified_interpretation": row.justified_interpretation,
                "prohibited_wording": list(row.prohibited_wording),
                "requires_full_vocabulary_revalidation": (
                    row.requires_full_vocabulary_revalidation
                ),
                "survival": row.survival,
                "notes": row.notes,
            }
        )
    payload = {
        "schema": "jlens.mmpilot.endpoint_claim_ledger.v1",
        "audit_version": ENDPOINT_AUDIT_VERSION,
        "n_claims": len(rows),
        "by_endpoint_class": {
            name: sum(1 for row in rows if row["endpoint_class"] == name)
            for name in ENDPOINT_CLASSES
        },
        "by_survival": {
            name: sum(1 for row in rows if row["survival"] == name)
            for name in SURVIVAL_CLASSES
        },
        "requiring_revalidation": [
            row["claim"]
            for row in rows
            if row["requires_full_vocabulary_revalidation"]
        ],
        "rows": rows,
    }
    return {**payload, "ledger_digest": payload_checksum(payload)}


def endpoint_audit_record(
    *,
    repo_root: str | Path,
    endpoints: Sequence[AuditedEndpoint] = AUDITED_ENDPOINTS,
    report_checksums: Mapping[str, str] | None = None,
    scan: Mapping | None = None,
) -> dict:
    """The complete Stage-0 audit. CPU only; imports no model.

    ``passed`` is False when the source scan found an unqualified prohibited
    description in the active package. The caller stops there — the audit is a
    gate on the rest of the study, not a report to skim.
    """
    resolved = verify_registry(endpoints)
    scan_result = dict(
        scan if scan is not None else scan_active_sources(repo_root)
    )
    ledger = claim_ledger(endpoints, report_checksums=report_checksums)
    payload = {
        "schema": "jlens.mmpilot.endpoint_semantics_audit.v1",
        "audit_version": ENDPOINT_AUDIT_VERSION,
        "endpoint_classes": list(ENDPOINT_CLASSES),
        "n_endpoints_audited": len(endpoints),
        "endpoints": [row.to_dict() for row in endpoints],
        "resolved_functions": resolved,
        "source_scan": scan_result,
        "claim_ledger_digest": ledger["ledger_digest"],
        "claim_ledger": ledger,
        "passed": bool(scan_result.get("passed")),
        "endpoint_trace_rule": (
            "every row was classified by reading the function that computes the "
            "field; no endpoint was inferred from report prose"
        ),
        "completed_reports_untouched": True,
        "scientific_recompute": 0,
    }
    return {**payload, "audit_digest": payload_checksum(payload)}


def endpoint_audit_files(record: Mapping) -> dict[str, str]:
    """The three artifacts, by filename, as exact text.

    One renderer for every caller — the script, the notebook and the test — so
    the committed files and the ones a notebook run writes are byte-identical.
    Write them with ``newline="\\n"``: a platform-translated newline would make
    the same audit differ between machines.
    """
    return {
        AUDIT_JSON_NAME: json.dumps(dict(record), indent=2, sort_keys=True) + "\n",
        AUDIT_MARKDOWN_NAME: format_endpoint_audit_markdown(record),
        LEDGER_JSON_NAME: json.dumps(
            dict(record["claim_ledger"]), indent=2, sort_keys=True
        )
        + "\n",
    }


def endpoint_audit_digest(record: Mapping) -> str:
    """The digest a downstream fingerprint binds to."""
    return str(record["audit_digest"])


def format_endpoint_audit_markdown(record: Mapping) -> str:
    """``endpoint_semantics_audit.md`` — the human-readable half."""
    ledger = dict(record["claim_ledger"])
    scan = dict(record["source_scan"])
    lines = [
        "# Endpoint-semantics audit",
        "",
        f"`{record['audit_version']}` · audit digest `{record['audit_digest']}`",
        "",
        "Every row below was classified by reading the function that computes",
        "the report field, not the report's prose. No completed report, unit or",
        "run directory was modified; `scientific_recompute = 0`.",
        "",
        "## Endpoint classes",
        "",
        "| class | claims |",
        "| --- | ---: |",
    ]
    for name, count in ledger["by_endpoint_class"].items():
        lines.append(f"| `{name}` | {count} |")
    lines += [
        "",
        "## Survival",
        "",
        "| survival | claims |",
        "| --- | ---: |",
    ]
    for name, count in ledger["by_survival"].items():
        lines.append(f"| `{name}` | {count} |")
    lines += [
        "",
        "## Claim ledger",
        "",
        "| claim | run | computing function | endpoint | candidates | appended | "
        "global vocab | generated | revalidate | survival |",
        "| --- | --- | --- | --- | --- | :-: | :-: | :-: | :-: | --- |",
    ]
    for row in ledger["rows"]:
        lines.append(
            "| {claim} | {run} | `{fn}` | `{endpoint}` | {universe} | {appended} | "
            "{globalv} | {generated} | {revalidate} | `{survival}` |".format(
                claim=row["claim"],
                run=row["source_run"] or "—",
                fn=row["computing_function"],
                endpoint=row["endpoint_class"],
                universe=row["candidate_universe"],
                appended="yes" if row["tokens_appended"] else "no",
                globalv="yes" if row["global_vocabulary_considered"] else "no",
                generated="yes" if row["generation_occurred"] else "no",
                revalidate="yes"
                if row["requires_full_vocabulary_revalidation"]
                else "no",
                survival=row["survival"],
            )
        )
    lines += [
        "",
        "## Prohibited wording, per claim",
        "",
    ]
    for row in ledger["rows"]:
        if not row["prohibited_wording"]:
            continue
        lines.append(f"**{row['claim']}**")
        lines.append("")
        lines.append(f"- justified: {row['justified_interpretation']}")
        for phrase in row["prohibited_wording"]:
            lines.append(f"- prohibited: *{phrase}*")
        if row["notes"]:
            lines.append(f"- note: {row['notes']}")
        lines.append("")
    lines += [
        "## Active-source scan",
        "",
        f"- files scanned: {scan.get('n_files_scanned')}",
        f"- prohibited-phrase hits: {scan.get('n_hits')}",
        f"- unqualified overclaims: {scan.get('n_overclaims')}",
        f"- passed: **{scan.get('passed')}**",
        "",
    ]
    for row in scan.get("overclaims", []):
        lines.append(f"- `{row['path']}:{row['line']}` ({row['pattern']}) {row['text']}")
    return "\n".join(lines).rstrip() + "\n"
