# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""How far the clean residual has already converged onto the model's own answer.

The completed three-modality run established controlled causal transfer at layer
35 by editing the residual at the **final prompt token** and measuring the
behavioral answer. It never established *when* in the stack the answer stops
being open. :mod:`jlens.mmpilot.tri_modal` says so in as many words — "layer 35
is primary because it is the earliest independently confirmed lens, and for no
other reason ... convergence timing is unresolved here and every artifact says
so". This module is the independent test that sentence points at.

The question, stated operationally
----------------------------------

At each independently validated layer (35, 38, 40), take the *stored* clean
final-prompt-token residual and push it through the model's **own frozen output
head** — the final normalization module and the unembedding, called as modules,
not reimplemented — then restrict the resulting logits to the six behavioral
answer candidates. Ask how often that readout already agrees with the answer the
model actually gave, and how sharply.

Nothing is fitted. Nothing is learned. No probe decides anything. The readout is
the model's own output pathway applied one layer early, which is the only
readout that can be called *native*.

Why this is independent of the J-space result
---------------------------------------------

The causal result is about a **direction estimated in J-space from one
modality** moving a behavioral answer in another. This measurement never touches
the lens, the dictionary, the J-space codes, or any intervention. It reads the
same stored activations through a different, model-owned map. The two can
therefore disagree, and a disagreement is the informative outcome: causal
purchase at a layer whose native readout has *not* converged is evidence that
the manipulated representation precedes the answer-language commitment.

The interpretation boundary, which is not negotiable
----------------------------------------------------

A weak direct readout means the representation has not converged **under the
predeclared native criterion**. It does not mean the answer is absent, and it is
not evidence that no nonlinear decoder or trained probe could recover it. Every
artifact this module writes carries :data:`INTERPRETATION_BOUNDARY`, and the
verdict strings say "before native direct-readout convergence", never
"pre-linguistic" and never "language-free".

The criterion is two-sided on purpose
-------------------------------------

There is an obvious way to cheat this audit: set the "converged" bar high, watch
layer 35 fail it, and call the failure evidence of pre-convergence. A single
threshold makes "not yet converged" the default state, and the default state is
exactly the one the interesting verdict needs.

So :class:`ConvergenceCriterion` declares **two** bars with a deliberate gap
between them. A layer is ``CONVERGED`` only above the upper bar and
``NOT_CONVERGED`` only below the lower one; everything between is ``AMBIGUOUS``
and yields :data:`INCONCLUSIVE_CONVERGENCE_TIMING`. The claim-supporting state
has to be *earned* against a bar of its own, and the two bars are measured with
deliberately mismatched strictness: convergence is scored with the strict
unique-top-1 rule (a tie is not convergence) and non-convergence with the
generous argmax rule (a layer must look weak even when ties are resolved in its
favour).

See :data:`CRITERION_TEXT` for the rule in words. It is a module constant so the
notebook can print it before a single result exists, and so editing it changes
the audit fingerprint and forces recomputation rather than silently rescoring
stored rows under a rule they were not produced by.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from jlens.mmpilot.admissibility import (
    CLAIM_ADMISSIBILITY_RULE_VERSION,
    concept_admissibility,
)
from jlens.mmpilot.store import (
    IncompatibleStateError,
    canonical_json,
    payload_checksum,
    safe_key,
)

# --------------------------------------------------------------------- protocol

#: Bound into every artifact and into the audit fingerprint. Distinct from the
#: transfer study's tag, the localization study's and the confirmatory
#: validation's, so no reader can mistake a convergence number for a causal one.
CONVERGENCE_PROTOCOL = "mmpilot.output_convergence_audit.v1"

#: The layers this audit may speak about. Exactly the layers whose lenses passed
#: the scale-100 calibration run's untouched confirmation set.
AUDITED_LAYERS: tuple[int, ...] = (35, 38, 40)

#: The layer the completed causal result belongs to.
PRIMARY_LAYER = 35

#: Layers whose lens failed confirmation. Never interpreted, never audited here.
LENS_INVALID_LAYERS: tuple[int, ...] = (32,)

#: Modality names, byte-identical to the completed run's.
MODALITIES: tuple[str, ...] = ("text", "image", "spoken_audio")

#: Printed on every artifact, and repeated in the report's conclusion.
INTERPRETATION_BOUNDARY = (
    "A weak native direct readout means the final-prompt-token residual has not "
    "converged onto the model's answer under the predeclared criterion in this "
    "module. It is NOT proof that linguistic information is absent: no claim is "
    "made, and none may be derived, about what a nonlinear decoder or a trained "
    "probe could recover from the same activation. Say 'before native "
    "direct-readout convergence', never 'pre-linguistic' and never "
    "'language-free'."
)

#: Why the literal wrong-layer control the brief asks for does not exist here.
WRONG_LAYER_CONTROL_NOTE = (
    "Gemma 4 has exactly one final normalization module and one unembedding, "
    "shared by every layer; there is no layer-specific readout component to "
    "misapply, so a literal wrong-layer readout control is not technically "
    "meaningful. The permuted-activation control below is its substitute: it "
    "keeps the head and shuffles which sample's residual it is applied to, "
    "which is the same question ('is this readout about this activation?') in "
    "the form this architecture admits."
)

#: Readout modes. Never mixed, and the mode is stamped on every row.
READOUT_SINGLE_TOKEN = "single_token_complete"
READOUT_FIRST_TOKEN = "first_token_only_diagnostic"

#: Rank conventions, matching :mod:`jlens.mmlocalize.lens_validity` so a rank
#: here and a rank there mean the same thing.
RANK_CONVENTIONS = ("optimistic", "pessimistic", "midrank")
CRITERION_RANK_CONVENTION = "midrank"

#: Control variants. None of them is a learned probe.
CONTROL_VARIANTS: tuple[str, ...] = (
    "shuffled_target_labels",
    "permuted_candidate_tokens",
    "permuted_activations",
)

#: The measurement under test.
PRIMARY_VARIANT = "native_readout"

#: Layer convergence classes.
CONVERGED = "CONVERGED"
NOT_CONVERGED = "NOT_CONVERGED"
AMBIGUOUS = "AMBIGUOUS"

#: The three possible verdicts.
PRE_CONVERGENCE_TRANSFER_SUPPORTED = "PRE_CONVERGENCE_TRANSFER_SUPPORTED"
TRANSFER_AT_OR_AFTER_CONVERGENCE = "TRANSFER_AT_OR_AFTER_CONVERGENCE"
INCONCLUSIVE_CONVERGENCE_TIMING = "INCONCLUSIVE_CONVERGENCE_TIMING"

#: Files in the completed run that this audit checksums before and after and
#: must never write. The audit refuses to finish if any of them moved.
PROTECTED_RUN_FILES: tuple[str, ...] = (
    "fingerprint.json",
    "native_audio_transfer_report.md",
    "native_audio_transfer_summary.json",
    "native_audio_transfer_report_capability_filtered_v2.md",
    "native_audio_transfer_summary_capability_filtered_v2.json",
    "run_manifest.json",
)


class ConvergenceRefused(RuntimeError):
    """A precondition of the audit does not hold, so nothing is measured."""


class CompletedRunModified(ConvergenceRefused):
    """A protected file in the completed run changed during the audit."""


class CandidateTokenizationError(ConvergenceRefused):
    """The candidate set cannot support a valid direct-readout comparison.

    Raised rather than downgraded, because every downgrade available here is a
    silent one: scoring first tokens while calling it a sequence score, or
    letting two candidates that share a first token compete for the same logit.
    """


class LensInvalidLayerError(ConvergenceRefused):
    """A layer whose lens failed confirmation was asked to be interpreted."""


# -------------------------------------------------------------------- criterion


@dataclass(frozen=True)
class ConvergenceCriterion:
    """The two-sided pass rule, fixed before any real result exists.

    The upper bar (``converged_*``) decides ``CONVERGED``; the lower bar
    (``not_converged_*``) decides ``NOT_CONVERGED``; the gap between them is
    ``AMBIGUOUS`` and can only produce :data:`INCONCLUSIVE_CONVERGENCE_TIMING`.

    Attributes:
        converged_min_clean_agreement: Fraction of samples whose native readout
            makes the model's own clean answer the **sole** maximum among the
            six candidates. This is the primary metric: "converged onto the
            final candidate-answer representation" is, operationally, "already
            names the answer the model went on to give". 0.90 is chosen because
            at 90% the answer is effectively determined at that layer — nine
            samples in ten need nothing further from the remaining blocks.
        converged_min_target_accuracy: The same, against the ground-truth
            concept rather than the model's answer. Reported and required
            together, because a readout that agrees with the model only where
            the model is wrong has not converged on an *answer*, it has
            converged on an error mode.
        converged_max_median_rank: Median midrank of the target among the six
            candidates. 1.0 means the target is typically first outright.
        not_converged_max_clean_agreement: Ceiling for the claim-supporting
            state, scored with the *generous* argmax rule. 0.50 is three times
            the six-candidate chance rate (0.167) and still means the readout
            disagrees with the model's own answer on at least half the samples.
            A representation that cannot name the model's answer more than half
            the time has not converged onto it under any reading.
        not_converged_max_target_accuracy: The same against ground truth.
        not_converged_min_median_rank: Median midrank must be at least 2 — the
            target is typically not even first.
        min_later_layer_separation: How much higher a later validated layer's
            clean agreement must be than layer 35's before the trajectory counts
            as showing movement rather than noise. Absolute, on a rate in
            [0, 1].
        require_disjoint_bootstrap_intervals: Additionally require the later
            layer's bootstrap CI to sit entirely above layer 35's. A fixed gap
            alone can be produced by a small sample; disjoint image-level
            intervals cannot as easily.
        max_non_monotonic_drop: How far clean agreement may fall from one
            audited layer to a later one before the trajectory is called
            non-monotonic and the audit refuses to read a direction into it.
        min_samples_per_cell: Floor on samples in a (layer, modality) cell. A
            rate over two samples is not a rate.
        min_distinct_predictions: Floor on how many distinct candidates the
            readout predicts at layer 35. A readout that answers the same word
            for every sample has failed, and a failed readout must not be read
            as a fact about the representation — this is the check that keeps
            the claim from resting on a broken measurement.
        required_modalities: Every one of these must satisfy the bar in its own
            right. A layer that has converged in text but not in audio is not a
            converged layer.
    """

    converged_min_clean_agreement: float = 0.90
    converged_min_target_accuracy: float = 0.90
    converged_max_median_rank: float = 1.0
    not_converged_max_clean_agreement: float = 0.50
    not_converged_max_target_accuracy: float = 0.50
    not_converged_min_median_rank: float = 2.0
    min_later_layer_separation: float = 0.20
    require_disjoint_bootstrap_intervals: bool = True
    max_non_monotonic_drop: float = 0.10
    min_samples_per_cell: int = 4
    min_distinct_predictions: int = 2
    required_modalities: tuple[str, ...] = MODALITIES
    bootstrap_resamples: int = 2000
    bootstrap_seed: int = 20260806
    bootstrap_confidence: float = 0.95

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["required_modalities"] = list(self.required_modalities)
        return payload

    @property
    def digest(self) -> str:
        return payload_checksum(self.to_dict())


#: The single criterion the audit runs under.
CONVERGENCE_CRITERION = ConvergenceCriterion()

#: Fixed alternative thresholds, reported for sensitivity and never able to
#: change the primary rule. They exist so a reader can see how far the verdict
#: is from the boundary without the boundary being moved to reach a conclusion.
SENSITIVITY_VARIANTS: tuple[tuple[str, ConvergenceCriterion], ...] = (
    (
        "lenient_converged_bar",
        ConvergenceCriterion(
            converged_min_clean_agreement=0.80,
            converged_min_target_accuracy=0.80,
            converged_max_median_rank=1.5,
        ),
    ),
    (
        "strict_not_converged_bar",
        ConvergenceCriterion(
            not_converged_max_clean_agreement=0.35,
            not_converged_max_target_accuracy=0.35,
            not_converged_min_median_rank=3.0,
        ),
    ),
    (
        "narrow_ambiguous_band",
        ConvergenceCriterion(
            converged_min_clean_agreement=0.75,
            converged_min_target_accuracy=0.75,
            not_converged_max_clean_agreement=0.60,
            not_converged_max_target_accuracy=0.60,
            not_converged_min_median_rank=1.5,
        ),
    ),
)


CRITERION_TEXT = f"""\
PREDECLARED CONVERGENCE CRITERION — native direct readout at L35 / L38 / L40
Fixed before any result-producing cell runs. Not revisable after seeing results.

The measurement. For each stored clean final-prompt-token residual h at an
audited layer, compute the model's OWN output head on it —

    logits = lm_head(final_norm(h))            [modules called, not reimplemented]
    logits = softcap * tanh(logits / softcap)  [if the config declares a softcap]

— and restrict those logits to the six fixed behavioral answer candidates. No
lens, no dictionary, no J-space code, no intervention and no learned probe takes
part in this number.

A layer is classified CONVERGED only if, in EVERY one of
{", ".join(CONVERGENCE_CRITERION.required_modalities)}, over
capability-admissible concepts:
  1. the target is the SOLE maximum among the six candidates, and equals the
     model's own clean final answer, on at least
     {CONVERGENCE_CRITERION.converged_min_clean_agreement:.0%} of samples;
  2. the same holds against the ground-truth concept on at least
     {CONVERGENCE_CRITERION.converged_min_target_accuracy:.0%} of samples;
  3. the median midrank of the target among the six candidates is at most
     {CONVERGENCE_CRITERION.converged_max_median_rank:.1f}.

A layer is classified NOT_CONVERGED only if, in EVERY one of those modalities,
scored with the GENEROUS argmax rule (ties resolved in the layer's favour):
  4. agreement with the model's clean final answer is at most
     {CONVERGENCE_CRITERION.not_converged_max_clean_agreement:.0%}
     (six-candidate chance is 16.7%);
  5. agreement with the ground-truth concept is at most
     {CONVERGENCE_CRITERION.not_converged_max_target_accuracy:.0%};
  6. the median midrank of the target is at least
     {CONVERGENCE_CRITERION.not_converged_min_median_rank:.1f}.

Anything between the two bars is AMBIGUOUS. An AMBIGUOUS layer 35 yields
{INCONCLUSIVE_CONVERGENCE_TIMING} and nothing else.

Guards, all of which force {INCONCLUSIVE_CONVERGENCE_TIMING} when they trip:
  - fewer than {CONVERGENCE_CRITERION.min_samples_per_cell} samples in any
    (layer, modality) cell;
  - fewer than {CONVERGENCE_CRITERION.min_distinct_predictions} distinct
    predicted candidates at layer {PRIMARY_LAYER} — a readout that answers the
    same word every time has failed, and a failed readout is not a fact about
    the representation;
  - any non-finite readout value;
  - a candidate set that is not single-token AND cannot support a labelled
    first-token-only diagnostic;
  - clean agreement falling by more than
    {CONVERGENCE_CRITERION.max_non_monotonic_drop:.2f} from an audited layer to
    a later one (the trajectory has no direction to read);
  - any control variant reaching the primary readout's clean agreement.

{PRE_CONVERGENCE_TRANSFER_SUPPORTED} additionally requires ALL of:
  - layer {PRIMARY_LAYER} classified NOT_CONVERGED;
  - the completed run's capability-filtered causal verdict at layer
    {PRIMARY_LAYER} still SUPPORTED, read from its artifacts and not recomputed;
  - at least one later validated layer clearly more converged: clean agreement
    higher by at least
    {CONVERGENCE_CRITERION.min_later_layer_separation:.2f} AND its image-level
    bootstrap interval entirely above layer {PRIMARY_LAYER}'s;
  - every requirement above satisfied on capability-admissible concepts only.

{TRANSFER_AT_OR_AFTER_CONVERGENCE} is returned whenever layer {PRIMARY_LAYER}
is classified CONVERGED. That branch is checked FIRST, so it can never be
masked by a later clause.

Verdict is exactly one of:
  {PRE_CONVERGENCE_TRANSFER_SUPPORTED}
  {TRANSFER_AT_OR_AFTER_CONVERGENCE}
  {INCONCLUSIVE_CONVERGENCE_TIMING}

{INTERPRETATION_BOUNDARY}
"""


# ------------------------------------------------------------ lens-validity gate


def assert_lens_valid_layer(layer: int, *, audited: Sequence[int] = AUDITED_LAYERS) -> None:
    """Refuse to interpret a layer whose lens is not independently validated.

    Raises:
        LensInvalidLayerError: For layer 32 (which failed the untouched
            confirmation set) and for any layer outside the audited set.
    """
    if int(layer) in tuple(int(x) for x in LENS_INVALID_LAYERS):
        raise LensInvalidLayerError(
            f"layer {layer} failed the calibration run's untouched confirmation "
            "set; it is not a validated lens layer and this audit refuses to "
            "produce a causal or convergence interpretation for it"
        )
    if int(layer) not in tuple(int(x) for x in audited):
        raise LensInvalidLayerError(
            f"layer {layer} is not one of the independently validated layers "
            f"{list(audited)}; refusing to interpret it"
        )


# ---------------------------------------------------------- candidate tokenization


def resolve_candidate_tokens(candidate_token_ids: Mapping[str, Sequence[int]]) -> dict:
    """Decide whether the candidate set admits a valid direct readout, and how.

    A single hidden state yields exactly one next-token distribution. If every
    candidate is one token, that distribution *is* a complete score for each
    candidate and the readout is comparable, token for token, with the run's own
    teacher-forced scoring. If any candidate is longer, it is not — so this
    returns the explicitly labelled
    :data:`READOUT_FIRST_TOKEN` mode instead of pretending otherwise.

    Raises:
        CandidateTokenizationError: If the candidate set is empty, a candidate
            encodes to zero tokens, or two candidates share a first token. The
            last case is fatal even for the first-token diagnostic: two
            candidates competing for one logit cannot be ranked against each
            other at all.
    """
    if not candidate_token_ids:
        raise CandidateTokenizationError("the candidate set is empty")

    candidates = sorted(candidate_token_ids)
    lengths: dict[str, int] = {}
    first_tokens: dict[str, int] = {}
    for candidate in candidates:
        ids = [int(i) for i in candidate_token_ids[candidate]]
        if not ids:
            raise CandidateTokenizationError(
                f"candidate {candidate!r} encodes to zero tokens"
            )
        lengths[candidate] = len(ids)
        first_tokens[candidate] = ids[0]

    collisions: dict[int, list[str]] = {}
    for candidate, token in first_tokens.items():
        collisions.setdefault(token, []).append(candidate)
    shared = {
        token: names for token, names in collisions.items() if len(names) > 1
    }
    if shared:
        detail = "; ".join(
            f"token {token} is the first token of {sorted(names)}"
            for token, names in sorted(shared.items())
        )
        raise CandidateTokenizationError(
            "two or more candidates share a first token, so a single hidden "
            f"state cannot rank them against each other: {detail}. No "
            "direct-readout comparison is valid for this candidate set."
        )

    all_single = all(length == 1 for length in lengths.values())
    mode = READOUT_SINGLE_TOKEN if all_single else READOUT_FIRST_TOKEN
    return {
        "schema": "jlens.mmpilot.candidate_tokenization.v1",
        "candidates": candidates,
        "n_candidates": len(candidates),
        "token_ids": {c: [int(i) for i in candidate_token_ids[c]] for c in candidates},
        "token_lengths": lengths,
        "readout_token_ids": {c: first_tokens[c] for c in candidates},
        "all_candidates_single_token": all_single,
        "readout_mode": mode,
        "scoring_note": (
            "every candidate is one token, so the readout scores the complete "
            "candidate and is directly comparable with the run's teacher-forced "
            "sequence score"
            if all_single
            else (
                "at least one candidate is multi-token. This is a FIRST-TOKEN-ONLY "
                "convergence diagnostic: a single hidden state does not yield a "
                "complete teacher-forced sequence score, and no row here is a "
                "sequence score. The run's own clean predictions remain complete "
                "sequence scores and are used only as the reference answer, never "
                "mixed into a readout total."
            )
        ),
        "multi_token_candidates": sorted(
            c for c, length in lengths.items() if length > 1
        ),
        "digest": payload_checksum(
            {
                "token_ids": {
                    c: [int(i) for i in candidate_token_ids[c]] for c in candidates
                },
                "mode": mode,
            }
        ),
    }


# ------------------------------------------------------------------ native head


@dataclass
class NativeHead:
    """The model's own frozen output pathway, held as modules and called as-is.

    ``final_norm`` and ``lm_head`` are the live modules from the loaded
    checkpoint. They are never reimplemented here: Gemma's RMSNorm applies its
    weight with an implementation-specific offset convention, and a
    hand-rolled ``x / rms * w`` is exactly the kind of "standard logit lens
    formula" that is wrong on this architecture. :func:`audit_native_head`
    records *which* convention the live module actually implements rather than
    assuming one.

    Attributes:
        softcap: ``final_logit_softcapping`` from the text config, or ``None``.
            Strictly monotonic, so it cannot change a ranking — but it does
            change logit values, entropies and margins, so the audited primary
            path applies it, because that is what the model's output head does.
    """

    final_norm: Any
    lm_head: Any
    softcap: float | None = None
    d_model: int = 0
    vocab_size: int = 0

    @torch.no_grad()
    def logits(self, activation: torch.Tensor) -> torch.Tensor:
        """Full-vocabulary logits for one residual, on the model's own path."""
        weight = self.lm_head.weight
        hidden = activation.to(dtype=weight.dtype, device=weight.device)
        if hidden.ndim == 1:
            hidden = hidden.unsqueeze(0)
        out = self.lm_head(self.final_norm(hidden)).float().squeeze(0)
        if self.softcap is not None:
            out = float(self.softcap) * torch.tanh(out / float(self.softcap))
        return out

    @torch.no_grad()
    def candidate_logits(
        self, activation: torch.Tensor, token_ids: Sequence[int]
    ) -> torch.Tensor:
        """Logits restricted to ``token_ids``, in the given order."""
        full = self.logits(activation)
        index = torch.tensor([int(i) for i in token_ids], dtype=torch.long)
        return full.index_select(0, index)


def head_from_model(model: Any) -> NativeHead:
    """Build a :class:`NativeHead` from a loaded :class:`~jlens.hf.HFLensModel`.

    Reads the same private attributes :meth:`jlens.hf.HFLensModel.unembed` uses,
    so the audited path and the model's own readout are the same two modules by
    construction rather than by resemblance.
    """
    final_norm = model._final_norm
    lm_head = model._lm_head
    return NativeHead(
        final_norm=final_norm,
        lm_head=lm_head,
        softcap=getattr(model, "_logit_softcap", None),
        d_model=int(lm_head.weight.shape[1]),
        vocab_size=int(lm_head.weight.shape[0]),
    )


def _module_checksum(module: Any) -> str:
    """``sha256:`` over a module's float32 parameter and buffer bytes."""
    digest = hashlib.sha256()
    named = list(getattr(module, "named_parameters", lambda: [])()) + list(
        getattr(module, "named_buffers", lambda: [])()
    )
    for name, tensor in sorted(named, key=lambda item: item[0]):
        digest.update(name.encode())
        digest.update(
            tensor.detach().to(torch.float32).cpu().contiguous().numpy().tobytes()
        )
    return "sha256:" + digest.hexdigest()


@torch.no_grad()
def audit_native_head(head: NativeHead, *, model: Any = None, probes: int = 4) -> dict:
    """Record what the frozen output head actually is, rather than assuming it.

    Three things are established empirically:

    1. **The normalization convention.** A probe vector is pushed through the
       live ``final_norm`` and the result is compared against both candidate
       RMSNorm conventions — ``x_hat * w`` and ``x_hat * (1 + w)``. Whichever
       matches is *recorded*; neither is used to compute anything. If the module
       is not an RMSNorm at all (a ``LayerNorm``, as in the CPU mock), that is
       recorded too and nothing fails.
    2. **The softcap.** Present or absent, and its value.
    3. **Agreement with the model's own ``unembed``.** When ``model`` is given,
       the audited path is compared against
       :meth:`jlens.hf.HFLensModel.unembed` on the same probes. They must agree
       to within float tolerance, which is what makes "native" a fact rather
       than a claim.

    Raises:
        ConvergenceRefused: If the audited path and the model's own ``unembed``
            disagree. That would mean the audit is reading the model through a
            head the model does not use.
    """
    generator = torch.Generator().manual_seed(11)
    probe = torch.randn(probes, head.d_model, generator=generator)

    weight = getattr(head.final_norm, "weight", None)
    convention = "unknown"
    max_gap: dict[str, float] = {}
    if weight is not None:
        w = weight.detach().float().cpu()
        x = probe.float()
        eps = float(
            getattr(head.final_norm, "eps", None)
            or getattr(head.final_norm, "variance_epsilon", None)
            or 1e-6
        )
        x_hat = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
        observed = (
            head.final_norm(probe.to(w.dtype)).detach().float().cpu()
        )
        for name, predicted in (
            ("rmsnorm_weight", x_hat * w),
            ("rmsnorm_one_plus_weight", x_hat * (1.0 + w)),
        ):
            max_gap[name] = float((observed - predicted).abs().max())
        convention = min(max_gap, key=max_gap.get)
        if max_gap[convention] > 1e-2:
            convention = "not_rmsnorm"

    report = {
        "schema": "jlens.mmpilot.native_head_audit.v1",
        "final_norm_class": type(head.final_norm).__name__,
        "lm_head_class": type(head.lm_head).__name__,
        "d_model": int(head.d_model),
        "vocab_size": int(head.vocab_size),
        "has_norm_weight": weight is not None,
        "norm_weight_convention": convention,
        "norm_convention_residuals": max_gap,
        "final_logit_softcapping": (
            None if head.softcap is None else float(head.softcap)
        ),
        "softcap_applied": head.softcap is not None,
        "softcap_is_monotonic": True,
        "softcap_note": (
            "cap * tanh(x / cap) is strictly monotonic, so it cannot change a "
            "ranking; it does change logit values, margins and entropies, and "
            "the audited path applies it because the model's output head does"
        ),
        "readout_expression": "lm_head(final_norm(h))" + (
            " then cap * tanh(./cap)" if head.softcap is not None else ""
        ),
        "modules_called_not_reimplemented": True,
        "final_norm_checksum": _module_checksum(head.final_norm),
        "lm_head_checksum": _module_checksum(head.lm_head),
        "matches_model_unembed": None,
        "max_abs_difference_vs_model_unembed": None,
    }
    report["head_checksum"] = payload_checksum(
        {
            "final_norm": report["final_norm_checksum"],
            "lm_head": report["lm_head_checksum"],
            "softcap": report["final_logit_softcapping"],
        }
    )

    if model is not None and hasattr(model, "unembed"):
        theirs = model.unembed(probe).detach().float().cpu()
        ours = torch.stack([head.logits(row) for row in probe]).cpu()
        gap = float((ours - theirs).abs().max())
        report["matches_model_unembed"] = bool(gap <= 1e-2)
        report["max_abs_difference_vs_model_unembed"] = gap
        if not report["matches_model_unembed"]:
            raise ConvergenceRefused(
                "the audited output head disagrees with the model's own "
                f"unembed by {gap:.4g}; this audit would be reading the model "
                "through a head it does not use. Refusing to measure."
            )
    return report


# --------------------------------------------------------------- scoring one row


def _entropy_and_margins(logits: torch.Tensor) -> dict:
    """Restricted-distribution statistics over the fixed candidates."""
    scores = logits.detach().float().flatten()
    log_probs = torch.log_softmax(scores, dim=0)
    probs = log_probs.exp()
    entropy = float(-(probs * log_probs).sum())
    ordered = torch.sort(scores, descending=True).values
    top_two = float(ordered[0] - ordered[1]) if scores.numel() > 1 else float("inf")
    return {
        "candidate_log_probs": [float(x) for x in log_probs.tolist()],
        "candidate_entropy_nats": entropy,
        "candidate_entropy_normalized": (
            entropy / math.log(scores.numel()) if scores.numel() > 1 else 0.0
        ),
        "top_two_margin": top_two,
    }


def tie_aware_ranks(scores: torch.Tensor, target_index: int) -> dict:
    """Optimistic / pessimistic / midrank of ``target_index``, plus tie counts.

    Same convention as :func:`jlens.mmlocalize.lens_validity.tie_aware_row`, so
    a median rank here is the same statistic as a median rank there.
    """
    flat = scores.detach().float().flatten()
    target_score = flat[int(target_index)]
    strictly_above = int((flat > target_score).sum())
    tied = int((flat == target_score).sum())
    return {
        "rank_optimistic": float(strictly_above + 1),
        "rank_pessimistic": float(strictly_above + tied),
        "rank_midrank": float(strictly_above + (tied + 1) / 2.0),
        "n_strictly_above_target": strictly_above,
        "n_tied_with_target": tied,
        "n_tied_at_max": int((flat == flat.max()).sum()),
    }


def direct_readout_row(
    *,
    activation: torch.Tensor,
    head: NativeHead,
    tokenization: Mapping,
    concept: str,
    clean_prediction: str | None,
    sample_id: str,
    group_id: str,
    image_id: str,
    recording_id: str | None,
    modality: str,
    layer: int,
    split: str,
    capability_admissible: bool,
    activation_checksum: str,
    head_checksum: str,
    config_hash: str,
    variant: str = PRIMARY_VARIANT,
    token_assignment: Mapping[str, int] | None = None,
    label_override: str | None = None,
) -> dict:
    """Score one (sample, modality, layer) activation against the fixed candidates.

    Args:
        clean_prediction: The model's own clean final candidate prediction for
            this sample, as the completed run recorded it (a complete
            teacher-forced sequence score). Used only as the reference answer;
            it is never combined with a readout score.
        token_assignment: Which token id supplies each candidate's logit.
            Defaults to the tokenization's own assignment; the permuted-candidate
            control passes a shuffled one, so each candidate is scored by another
            candidate's token while the candidate names stay put.
        label_override: The label to score against, replacing ``concept``. Used
            by the shuffled-label control. For the primary variant the scored
            target *is* the ground-truth concept.

    Raises:
        ConvergenceRefused: If the readout produces a non-finite value.
    """
    candidates = list(tokenization["candidates"])
    assignment = dict(
        token_assignment
        if token_assignment is not None
        else tokenization["readout_token_ids"]
    )
    readout_ids = [int(assignment[c]) for c in candidates]
    scores = head.candidate_logits(activation, readout_ids)
    if not bool(torch.isfinite(scores).all()):
        raise ConvergenceRefused(
            f"non-finite direct-readout logits for {sample_id} at layer {layer} "
            f"({modality}); refusing to record an uninterpretable row"
        )

    target = str(label_override if label_override is not None else concept)
    if target not in candidates:
        raise ConvergenceRefused(
            f"target {target!r} is not among the fixed candidates {candidates}"
        )
    target_index = candidates.index(target)
    ranks = tie_aware_ranks(scores, target_index)
    prediction = candidates[int(scores.argmax())]

    without_target = scores.clone()
    without_target[target_index] = float("-inf")
    margin = float(scores[target_index] - without_target.max())

    unique_top1 = bool(
        ranks["n_strictly_above_target"] == 0 and ranks["n_tied_with_target"] == 1
    )
    return {
        "schema": "jlens.mmpilot.direct_readout_row.v1",
        "protocol": CONVERGENCE_PROTOCOL,
        "variant": variant,
        "sample_id": sample_id,
        "group_id": group_id,
        "image_id": image_id,
        # SpokenCOCO carries one recording per synchronized group, so for the
        # audio arm the group *is* the recording. Recorded explicitly rather
        # than left to be inferred.
        "recording_id": recording_id if recording_id is not None else group_id,
        "concept": concept,
        "scored_target": target,
        "capability_admissible": bool(capability_admissible),
        "modality": modality,
        "layer": int(layer),
        "split": split,
        "readout_mode": tokenization["readout_mode"],
        "candidates": candidates,
        "candidate_token_ids": {
            c: list(tokenization["token_ids"][c]) for c in candidates
        },
        "readout_token_ids": readout_ids,
        "candidate_logits": [float(x) for x in scores.tolist()],
        **_entropy_and_margins(scores),
        "direct_readout_prediction": prediction,
        "target_rank": ranks["rank_midrank"],
        **ranks,
        "rank_convention_used": CRITERION_RANK_CONVENTION,
        "target_margin": margin,
        "unique_top1_target": unique_top1,
        "argmax_target": bool(prediction == target),
        "clean_final_prediction": clean_prediction,
        "agrees_with_clean_final_prediction_argmax": (
            None if clean_prediction is None else bool(prediction == clean_prediction)
        ),
        "agrees_with_clean_final_prediction_unique": (
            None
            if clean_prediction is None
            else bool(
                prediction == clean_prediction
                and ranks["n_tied_at_max"] == 1
            )
        ),
        # Measured against the *scored* target, which for the primary variant is
        # the ground-truth concept and for the shuffled-label control is the
        # permuted label. Scoring these against ``concept`` regardless would
        # leave the label control unable to move the metric it exists to move.
        "agrees_with_ground_truth_argmax": bool(prediction == target),
        "agrees_with_ground_truth_unique": bool(
            prediction == target and ranks["n_tied_at_max"] == 1
        ),
        "activation_checksum": activation_checksum,
        "head_checksum": head_checksum,
        "config_hash": config_hash,
        "interpretation_boundary": INTERPRETATION_BOUNDARY,
    }


# --------------------------------------------------------------------- controls


def shuffled_label_assignment(
    rows: Sequence[Mapping], *, seed: int
) -> dict[str, str]:
    """Permute which concept each sample is scored against.

    Target accuracy must collapse toward chance. Agreement with the model's
    clean prediction is label-free and is expected to be *unchanged* — that is
    the point of running this control rather than assuming it: it shows which
    reported number the labels can and cannot move.
    """
    keys = sorted({str(row["sample_id"]) for row in rows})
    labels = [
        str(row["concept"])
        for row in sorted(rows, key=lambda r: str(r["sample_id"]))
    ]
    rng = random.Random(seed)
    shuffled = list(labels)
    rng.shuffle(shuffled)
    return dict(zip(keys, shuffled, strict=True))


def permuted_token_assignment(
    tokenization: Mapping, *, seed: int
) -> dict[str, int]:
    """Give every candidate a different candidate's token.

    The candidate *names* stay where they are and only the token supplying each
    one's logit moves, so the readout is asked which token is most likely and
    then told the wrong word for it. Both the agreement and the accuracy metric
    must collapse. A derangement, so no candidate accidentally keeps its own
    token.
    """
    names = list(tokenization["candidates"])
    tokens = [int(tokenization["readout_token_ids"][name]) for name in names]
    if len(names) < 2:
        return dict(zip(names, tokens, strict=True))
    rng = random.Random(seed)
    for _ in range(64):
        shuffled = list(tokens)
        rng.shuffle(shuffled)
        if all(a != b for a, b in zip(tokens, shuffled, strict=True)):
            return dict(zip(names, shuffled, strict=True))
    return dict(zip(names, tokens[1:] + tokens[:1], strict=True))


def permuted_activation_assignment(
    rows: Sequence[Mapping], *, seed: int
) -> dict[str, str]:
    """Map each sample to a different sample's activation.

    The substitute for the wrong-layer control the architecture does not admit
    (:data:`WRONG_LAYER_CONTROL_NOTE`). Holds the head fixed and asks whether
    the readout is about *this* activation or about a prior the head carries
    regardless of input.
    """
    keys = sorted({str(row["sample_id"]) for row in rows})
    if len(keys) < 2:
        return {key: key for key in keys}
    rng = random.Random(seed)
    for _ in range(64):
        shuffled = list(keys)
        rng.shuffle(shuffled)
        if all(a != b for a, b in zip(keys, shuffled, strict=True)):
            return dict(zip(keys, shuffled, strict=True))
    return dict(zip(keys, keys[1:] + keys[:1], strict=True))


# ------------------------------------------------------------------ aggregation


def _rate(rows: Sequence[Mapping], field_name: str) -> float | None:
    values = [row[field_name] for row in rows if row.get(field_name) is not None]
    if not values:
        return None
    return statistics.fmean(1.0 if bool(v) else 0.0 for v in values)


def summarize_cell(rows: Sequence[Mapping]) -> dict:
    """Every reported statistic for one group of rows.

    ``clean_agreement_unique`` is the strict primary metric (sole maximum, and
    it is the model's own answer). ``clean_agreement_argmax`` is the generous
    one. The criterion reads the strict one upward and the generous one
    downward, so neither bar can be cleared by a tie-breaking convention.
    """
    rows = list(rows)
    if not rows:
        return {"n": 0}
    ranks = [float(row["rank_midrank"]) for row in rows]
    margins = [float(row["target_margin"]) for row in rows]
    entropies = [float(row["candidate_entropy_nats"]) for row in rows]
    top_two = [float(row["top_two_margin"]) for row in rows]
    predictions = [str(row["direct_readout_prediction"]) for row in rows]
    n_with_clean = sum(
        1 for row in rows if row.get("clean_final_prediction") is not None
    )
    return {
        "n": len(rows),
        "n_with_clean_reference": n_with_clean,
        "n_distinct_images": len({str(row["image_id"]) for row in rows}),
        "n_distinct_recordings": len({str(row["recording_id"]) for row in rows}),
        "n_distinct_predictions": len(set(predictions)),
        "clean_agreement_unique": _rate(rows, "agrees_with_clean_final_prediction_unique"),
        "clean_agreement_argmax": _rate(rows, "agrees_with_clean_final_prediction_argmax"),
        "target_accuracy_unique": _rate(rows, "agrees_with_ground_truth_unique"),
        "target_accuracy_argmax": _rate(rows, "agrees_with_ground_truth_argmax"),
        "unique_top1_target_rate": _rate(rows, "unique_top1_target"),
        "median_target_rank": float(statistics.median(ranks)),
        "mean_target_rank": statistics.fmean(ranks),
        "median_target_margin": float(statistics.median(margins)),
        "mean_target_margin": statistics.fmean(margins),
        "median_candidate_entropy_nats": float(statistics.median(entropies)),
        "mean_candidate_entropy_nats": statistics.fmean(entropies),
        "median_top_two_margin": float(statistics.median(top_two)),
        "tied_at_max_rate": statistics.fmean(
            1.0 if int(row["n_tied_at_max"]) > 1 else 0.0 for row in rows
        ),
        "readout_mode": rows[0]["readout_mode"],
    }


def bootstrap_rate(
    rows: Sequence[Mapping],
    field_name: str,
    *,
    unit_field: str = "image_id",
    seed: int,
    resamples: int,
    confidence: float = 0.95,
) -> dict:
    """Image- (or recording-) level bootstrap CI for a boolean rate.

    The photograph is the independent unit — the completed run established that
    and excluded image-level pseudoreplication on the strength of it — so the
    resampling unit is ``unit_field`` and every row belonging to a drawn unit
    comes along with it. Fully deterministic in ``seed``: the same rows and seed
    give the same interval on any machine.
    """
    usable = [row for row in rows if row.get(field_name) is not None]
    if not usable:
        return {"point": None, "low": None, "high": None, "n_units": 0, "n": 0}

    by_unit: dict[str, list[float]] = {}
    for row in usable:
        by_unit.setdefault(str(row[unit_field]), []).append(
            1.0 if bool(row[field_name]) else 0.0
        )
    units = sorted(by_unit)
    point = statistics.fmean(value for values in by_unit.values() for value in values)
    if len(units) < 2 or resamples <= 0:
        return {
            "point": point,
            "low": point,
            "high": point,
            "n_units": len(units),
            "n": len(usable),
            "resamples": 0,
            "note": "fewer than two independent units; no interval is estimated",
        }

    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(int(resamples)):
        pooled: list[float] = []
        for _ in units:
            pooled.extend(by_unit[units[rng.randrange(len(units))]])
        draws.append(statistics.fmean(pooled))
    draws.sort()
    alpha = (1.0 - float(confidence)) / 2.0
    low = draws[max(0, min(len(draws) - 1, int(math.floor(alpha * len(draws)))))]
    high = draws[max(0, min(len(draws) - 1, int(math.ceil((1 - alpha) * len(draws))) - 1))]
    return {
        "point": point,
        "low": low,
        "high": high,
        "n_units": len(units),
        "n": len(usable),
        "resamples": int(resamples),
        "unit_field": unit_field,
        "confidence": float(confidence),
        "seed": int(seed),
    }


def derived_seed(base: int, *parts: object) -> int:
    """A stable seed from ``base`` and any identifying parts.

    Derived from a checksum rather than from lengths or Python's salted
    ``hash()``: two modality names of equal length must not silently share a
    resampling seed, and the same inputs must give the same seed on every
    machine and in every process.
    """
    digest = payload_checksum([int(base), *[str(part) for part in parts]])
    return int(digest.split(":")[1][:8], 16)


def summarize_rows(
    rows: Sequence[Mapping],
    *,
    criterion: ConvergenceCriterion = CONVERGENCE_CRITERION,
    layers: Sequence[int] = AUDITED_LAYERS,
    variant: str = PRIMARY_VARIANT,
) -> dict:
    """Per-layer, per-modality and per-concept summaries over the primary rows.

    Per-concept results come first and pooled principal results are derived from
    them, never the other way round: pooling two concepts before looking at
    either is how a single strong concept comes to stand for a claim about both.
    """
    selected = [
        row
        for row in rows
        if row["variant"] == variant and bool(row["capability_admissible"])
    ]
    inadmissible = [
        row
        for row in rows
        if row["variant"] == variant and not bool(row["capability_admissible"])
    ]

    per_layer: dict[str, dict] = {}
    for layer in layers:
        layer_rows = [row for row in selected if int(row["layer"]) == int(layer)]
        per_modality: dict[str, dict] = {}
        for modality in MODALITIES:
            modality_rows = [row for row in layer_rows if row["modality"] == modality]
            per_concept = {
                concept: summarize_cell(
                    [row for row in modality_rows if row["concept"] == concept]
                )
                for concept in sorted({row["concept"] for row in modality_rows})
            }
            summary = summarize_cell(modality_rows)
            summary["per_concept"] = per_concept
            summary["bootstrap_clean_agreement_unique"] = bootstrap_rate(
                modality_rows,
                "agrees_with_clean_final_prediction_unique",
                seed=derived_seed(criterion.bootstrap_seed, "clean", layer, modality),
                resamples=criterion.bootstrap_resamples,
                confidence=criterion.bootstrap_confidence,
            )
            summary["bootstrap_target_accuracy_argmax"] = bootstrap_rate(
                modality_rows,
                "agrees_with_ground_truth_argmax",
                seed=derived_seed(criterion.bootstrap_seed, "accuracy", layer, modality),
                resamples=criterion.bootstrap_resamples,
                confidence=criterion.bootstrap_confidence,
            )
            per_modality[modality] = summary

        pooled = summarize_cell(layer_rows)
        pooled["per_concept"] = {
            concept: summarize_cell(
                [row for row in layer_rows if row["concept"] == concept]
            )
            for concept in sorted({row["concept"] for row in layer_rows})
        }
        pooled["bootstrap_clean_agreement_unique"] = bootstrap_rate(
            layer_rows,
            "agrees_with_clean_final_prediction_unique",
            seed=derived_seed(criterion.bootstrap_seed, "pooled", layer),
            resamples=criterion.bootstrap_resamples,
            confidence=criterion.bootstrap_confidence,
        )
        per_layer[str(int(layer))] = {
            "layer": int(layer),
            "per_modality": per_modality,
            "pooled_principal": pooled,
            "pooled_note": (
                "pooled over capability-admissible concepts only, and reported "
                "after the per-concept rows it is derived from"
            ),
        }

    return {
        "schema": "jlens.mmpilot.convergence_summary.v1",
        "protocol": CONVERGENCE_PROTOCOL,
        "variant": variant,
        "per_layer": per_layer,
        "descriptive_inadmissible": {
            "n_rows": len(inadmissible),
            "concepts": sorted({row["concept"] for row in inadmissible}),
            "per_layer": {
                str(int(layer)): {
                    concept: summarize_cell(
                        [
                            row
                            for row in inadmissible
                            if int(row["layer"]) == int(layer)
                            and row["concept"] == concept
                        ]
                    )
                    for concept in sorted({row["concept"] for row in inadmissible})
                }
                for layer in layers
            },
            "note": (
                "descriptive only. These concepts failed the behavioral "
                "capability gate in at least one required modality and are "
                "excluded from every principal number and from the verdict."
            ),
        },
    }


# ------------------------------------------------------------ layer classification


def classify_layer(
    layer_summary: Mapping,
    *,
    criterion: ConvergenceCriterion = CONVERGENCE_CRITERION,
) -> dict:
    """Apply the two-sided bar to one layer, and say which clause decided it."""
    per_modality = layer_summary["per_modality"]
    converged_checks: list[dict] = []
    not_converged_checks: list[dict] = []
    undersized: list[str] = []

    for modality in criterion.required_modalities:
        cell = per_modality.get(modality) or {"n": 0}
        n = int(cell.get("n", 0))
        if n < criterion.min_samples_per_cell:
            undersized.append(
                f"{modality}: {n} sample(s) < {criterion.min_samples_per_cell}"
            )
            converged_checks.append(
                {"modality": modality, "clause": "cell_size", "passed": False,
                 "detail": f"{n} samples"}
            )
            not_converged_checks.append(
                {"modality": modality, "clause": "cell_size", "passed": False,
                 "detail": f"{n} samples"}
            )
            continue

        clean_unique = cell.get("clean_agreement_unique")
        clean_argmax = cell.get("clean_agreement_argmax")
        target_unique = cell.get("target_accuracy_unique")
        target_argmax = cell.get("target_accuracy_argmax")
        median_rank = cell.get("median_target_rank")

        converged_checks += [
            {
                "modality": modality,
                "clause": "clean_agreement_unique_floor",
                "passed": clean_unique is not None
                and clean_unique >= criterion.converged_min_clean_agreement,
                "detail": f"{clean_unique} vs >= {criterion.converged_min_clean_agreement}",
            },
            {
                "modality": modality,
                "clause": "target_accuracy_unique_floor",
                "passed": target_unique is not None
                and target_unique >= criterion.converged_min_target_accuracy,
                "detail": f"{target_unique} vs >= {criterion.converged_min_target_accuracy}",
            },
            {
                "modality": modality,
                "clause": "median_rank_ceiling",
                "passed": median_rank is not None
                and median_rank <= criterion.converged_max_median_rank,
                "detail": f"{median_rank} vs <= {criterion.converged_max_median_rank}",
            },
        ]
        not_converged_checks += [
            {
                "modality": modality,
                "clause": "clean_agreement_argmax_ceiling",
                "passed": clean_argmax is not None
                and clean_argmax <= criterion.not_converged_max_clean_agreement,
                "detail": f"{clean_argmax} vs <= {criterion.not_converged_max_clean_agreement}",
            },
            {
                "modality": modality,
                "clause": "target_accuracy_argmax_ceiling",
                "passed": target_argmax is not None
                and target_argmax <= criterion.not_converged_max_target_accuracy,
                "detail": f"{target_argmax} vs <= {criterion.not_converged_max_target_accuracy}",
            },
            {
                "modality": modality,
                "clause": "median_rank_floor",
                "passed": median_rank is not None
                and median_rank >= criterion.not_converged_min_median_rank,
                "detail": f"{median_rank} vs >= {criterion.not_converged_min_median_rank}",
            },
        ]

    is_converged = bool(converged_checks) and all(
        check["passed"] for check in converged_checks
    )
    is_not_converged = bool(not_converged_checks) and all(
        check["passed"] for check in not_converged_checks
    )
    if is_converged and is_not_converged:  # pragma: no cover - impossible by the gap
        raise ConvergenceRefused(
            "the criterion's two bars overlap; the configured thresholds leave "
            "no ambiguous band and cannot classify a layer"
        )
    if is_converged:
        classification = CONVERGED
    elif is_not_converged:
        classification = NOT_CONVERGED
    else:
        classification = AMBIGUOUS

    pooled = layer_summary["pooled_principal"]
    return {
        "layer": int(layer_summary["layer"]),
        "classification": classification,
        "converged_checks": converged_checks,
        "not_converged_checks": not_converged_checks,
        "failed_converged_clauses": [
            f"{c['modality']}.{c['clause']}" for c in converged_checks if not c["passed"]
        ],
        "failed_not_converged_clauses": [
            f"{c['modality']}.{c['clause']}"
            for c in not_converged_checks
            if not c["passed"]
        ],
        "undersized_cells": undersized,
        "pooled_clean_agreement_unique": pooled.get("clean_agreement_unique"),
        "pooled_clean_agreement_argmax": pooled.get("clean_agreement_argmax"),
        "pooled_target_accuracy_argmax": pooled.get("target_accuracy_argmax"),
        "pooled_median_target_rank": pooled.get("median_target_rank"),
        "pooled_bootstrap": pooled.get("bootstrap_clean_agreement_unique"),
        "n_distinct_predictions": pooled.get("n_distinct_predictions"),
        "criterion_digest": criterion.digest,
    }


def classify_all_layers(
    summary: Mapping,
    *,
    criterion: ConvergenceCriterion = CONVERGENCE_CRITERION,
    layers: Sequence[int] = AUDITED_LAYERS,
) -> dict[int, dict]:
    for layer in layers:
        assert_lens_valid_layer(int(layer), audited=layers)
    return {
        int(layer): classify_layer(
            summary["per_layer"][str(int(layer))], criterion=criterion
        )
        for layer in layers
    }


def trajectory_report(
    classifications: Mapping[int, Mapping],
    *,
    criterion: ConvergenceCriterion = CONVERGENCE_CRITERION,
    primary_layer: int = PRIMARY_LAYER,
) -> dict:
    """Is the layer-to-layer trajectory readable, and is a later layer clearly ahead?"""
    layers = sorted(int(layer) for layer in classifications)
    agreements = {
        layer: classifications[layer].get("pooled_clean_agreement_unique")
        for layer in layers
    }
    drops: list[dict] = []
    for earlier, later in zip(layers, layers[1:], strict=False):
        a, b = agreements.get(earlier), agreements.get(later)
        if a is None or b is None:
            continue
        if b < a - criterion.max_non_monotonic_drop:
            drops.append(
                {"from_layer": earlier, "to_layer": later, "drop": float(a - b)}
            )

    primary = classifications.get(int(primary_layer), {})
    primary_point = primary.get("pooled_clean_agreement_unique")
    primary_ci = primary.get("pooled_bootstrap") or {}
    more_converged: list[dict] = []
    for layer in layers:
        if layer <= int(primary_layer):
            continue
        point = agreements.get(layer)
        ci = classifications[layer].get("pooled_bootstrap") or {}
        if point is None or primary_point is None:
            continue
        gap = float(point - primary_point)
        intervals_disjoint = (
            ci.get("low") is not None
            and primary_ci.get("high") is not None
            and float(ci["low"]) > float(primary_ci["high"])
        )
        clearly = gap >= criterion.min_later_layer_separation and (
            intervals_disjoint or not criterion.require_disjoint_bootstrap_intervals
        )
        more_converged.append(
            {
                "layer": layer,
                "clean_agreement_unique": point,
                "gap_over_primary": gap,
                "bootstrap_low": ci.get("low"),
                "primary_bootstrap_high": primary_ci.get("high"),
                "intervals_disjoint": bool(intervals_disjoint),
                "clearly_more_converged": bool(clearly),
            }
        )

    return {
        "layers": layers,
        "clean_agreement_unique_by_layer": {
            str(layer): agreements[layer] for layer in layers
        },
        "monotone_within_tolerance": not drops,
        "non_monotonic_drops": drops,
        "max_non_monotonic_drop": criterion.max_non_monotonic_drop,
        "later_layers": more_converged,
        "any_later_layer_clearly_more_converged": any(
            entry["clearly_more_converged"] for entry in more_converged
        ),
    }


# --------------------------------------------------------------- control summary


#: How far below the primary readout a control must sit before it counts as
#: beaten, and how far above chance the primary must sit before a control has
#: anything to reproduce.
CONTROL_SEPARATION_MARGIN = 0.15


def summarize_controls(
    rows: Sequence[Mapping],
    *,
    layers: Sequence[int] = AUDITED_LAYERS,
    variants: Sequence[str] = CONTROL_VARIANTS,
    margin: float = CONTROL_SEPARATION_MARGIN,
) -> dict:
    """Per-layer control metrics next to the primary readout's, with a pass flag.

    A control fails when it **reproduces** the primary readout's result. That
    question is only meaningful where the primary result is itself above chance:
    at a layer whose readout sits at the ``1/n_candidates`` floor there is no
    result for a control to reproduce, and requiring the control to fall *below*
    chance would be requiring it to do something no permutation can. Such a cell
    is recorded as ``primary_is_informative: false`` rather than being scored as
    a pass on the merits — the distinction is visible in the artifact instead of
    being buried in a boolean.

    Each control is compared on the metric it actually perturbs: the label
    shuffle on target accuracy (it does not touch the model's clean answer, so
    it cannot move agreement), the other two on agreement with that answer.
    """
    out: dict[str, dict] = {}
    failures: list[str] = []
    for layer in layers:
        primary_rows = [
            row
            for row in rows
            if row["variant"] == PRIMARY_VARIANT
            and int(row["layer"]) == int(layer)
            and bool(row["capability_admissible"])
        ]
        if not primary_rows:
            continue
        primary = summarize_cell(primary_rows)
        n_candidates = max(len(primary_rows[0]["candidates"]), 1)
        chance = 1.0 / n_candidates
        entry: dict[str, Any] = {
            "primary": primary,
            "chance_rate": chance,
            "controls": {},
        }
        for variant in variants:
            control_rows = [
                row
                for row in rows
                if row["variant"] == variant
                and int(row["layer"]) == int(layer)
                and bool(row["capability_admissible"])
            ]
            if not control_rows:
                continue
            control = summarize_cell(control_rows)
            if variant == "shuffled_target_labels":
                compared_field = "target_accuracy_argmax"
                expectation = (
                    "target accuracy collapses toward the 1/n_candidates chance "
                    "rate; agreement with the model's clean answer is label-free, "
                    "so this control cannot move it and is not scored on it"
                )
            else:
                compared_field = "clean_agreement_argmax"
                expectation = "agreement with the model's clean answer collapses"
            primary_value = primary.get(compared_field)
            control_value = control.get(compared_field)
            informative = (
                primary_value is not None and primary_value > chance + margin
            )
            if primary_value is None or control_value is None:
                passed, reason = True, "metric unavailable in this cell"
            elif not informative:
                passed = True
                reason = (
                    f"primary {compared_field} {primary_value:.3f} is at or near "
                    f"the {chance:.3f} chance rate, so no result rests on this "
                    "cell and the control has nothing to reproduce"
                )
            else:
                passed = control_value <= primary_value - margin
                reason = (
                    f"control {control_value:.3f} vs primary {primary_value:.3f} "
                    f"minus margin {margin:.2f}"
                )
            entry["controls"][variant] = {
                "metrics": control,
                "compared_field": compared_field,
                "primary_value": primary_value,
                "control_value": control_value,
                "chance_rate": chance,
                "primary_is_informative": bool(informative),
                "margin": float(margin),
                "expectation": expectation,
                "reason": reason,
                "passed": bool(passed),
            }
            if not passed:
                failures.append(f"L{layer}:{variant}")
        out[str(int(layer))] = entry
    return {
        "schema": "jlens.mmpilot.convergence_controls.v2",
        "per_layer": out,
        "margin": float(margin),
        "all_controls_passed": not failures,
        "failed_controls": failures,
        "pass_rule": (
            "a control fails only by reproducing a primary result that is itself "
            "above chance; a cell whose primary readout sits at chance is "
            "recorded as non-informative rather than scored"
        ),
        "wrong_layer_control_note": WRONG_LAYER_CONTROL_NOTE,
        "no_learned_probe_is_primary": True,
    }


# ------------------------------------------------------- secondary linear probe


def image_disjoint_folds(
    rows: Sequence[Mapping], *, n_folds: int = 4
) -> dict[str, int]:
    """Assign each image id to a fold, so no photograph spans train and test.

    A fixed partition of the sorted image ids, not a draw: the same images give
    the same folds every time.
    """
    images = sorted({str(row["image_id"]) for row in rows})
    return {image: index % int(n_folds) for index, image in enumerate(images)}


def secondary_linear_probe(
    rows: Sequence[Mapping],
    activations: Mapping[str, Sequence[float]],
    *,
    n_folds: int = 4,
    ridge: float = 1.0,
) -> dict:
    """A ridge readout of the concept from the residual, cross-validated by image.

    **Secondary and non-determining.** It answers a different question from the
    audit's: not "has the representation converged onto the model's answer" but
    "is the concept linearly decodable at all". A probe that succeeds where the
    native readout fails is exactly the outcome
    :data:`INTERPRETATION_BOUNDARY` warns about, and it changes no verdict.

    Closed-form least squares on one-hot targets — no iterations, no learning
    rate, no early stopping, so the number is a deterministic function of the
    inputs.
    """
    usable = [row for row in rows if str(row["sample_id"]) in activations]
    concepts = sorted({str(row["concept"]) for row in usable})
    folds = image_disjoint_folds(usable, n_folds=n_folds)
    if len(concepts) < 2 or len(usable) < 2 * n_folds:
        return {
            "schema": "jlens.mmpilot.convergence_probe.v1",
            "ran": False,
            "reason": "too few concepts or samples for image-disjoint folds",
            "is_secondary_diagnostic": True,
            "determines_verdict": False,
        }

    per_fold: list[dict] = []
    for fold in range(int(n_folds)):
        train = [r for r in usable if folds[str(r["image_id"])] != fold]
        test = [r for r in usable if folds[str(r["image_id"])] == fold]
        if not train or not test:
            continue
        x_train = torch.tensor(
            [list(activations[str(r["sample_id"])]) for r in train], dtype=torch.float64
        )
        y_train = torch.zeros(len(train), len(concepts), dtype=torch.float64)
        for i, row in enumerate(train):
            y_train[i, concepts.index(str(row["concept"]))] = 1.0
        gram = x_train.T @ x_train + float(ridge) * torch.eye(
            x_train.shape[1], dtype=torch.float64
        )
        weights = torch.linalg.solve(gram, x_train.T @ y_train)
        x_test = torch.tensor(
            [list(activations[str(r["sample_id"])]) for r in test], dtype=torch.float64
        )
        predicted = (x_test @ weights).argmax(dim=1)
        correct = sum(
            1
            for i, row in enumerate(test)
            if concepts[int(predicted[i])] == str(row["concept"])
        )
        per_fold.append(
            {
                "fold": fold,
                "n_train": len(train),
                "n_test": len(test),
                "accuracy": correct / len(test),
            }
        )
    return {
        "schema": "jlens.mmpilot.convergence_probe.v1",
        "ran": True,
        "concepts": concepts,
        "n_folds": int(n_folds),
        "ridge": float(ridge),
        "fold_accuracy": per_fold,
        "mean_accuracy": (
            statistics.fmean(entry["accuracy"] for entry in per_fold)
            if per_fold
            else None
        ),
        "chance": 1.0 / len(concepts),
        "image_disjoint": True,
        "is_secondary_diagnostic": True,
        "determines_verdict": False,
        "caveat": (
            "A probe that decodes the concept where the native readout does not "
            "is consistent with the audit's finding, not contrary to it: the "
            "audit measures convergence onto the model's own output, not "
            "decodability. " + INTERPRETATION_BOUNDARY
        ),
    }


# --------------------------------------------------------------- frozen evidence


def read_frozen_causal_evidence(summary: Mapping, *, layers: Sequence[int] = AUDITED_LAYERS) -> dict:
    """Extract the completed run's causal verdicts. Read, never recomputed.

    ``summary`` is the capability-filtered amended summary
    (``native_audio_transfer_summary_capability_filtered_v2.json``) or the run's
    own ``native_audio_transfer_summary.json``; both carry ``verdicts`` in the
    same shape. Layer 35's verdict comes from ``C_primary_causal`` and the
    replication layers from ``D_replication.per_layer``.
    """
    verdicts = dict(summary.get("verdicts") or {})
    primary = dict(verdicts.get("C_primary_causal") or {})
    replication = dict(verdicts.get("D_replication") or {})
    per_layer_replication = dict(replication.get("per_layer") or {})

    per_layer: dict[int, dict] = {}
    for layer in layers:
        layer = int(layer)
        if primary and int(primary.get("layer", -1)) == layer:
            source, record = "C_primary_causal", primary
        elif str(layer) in per_layer_replication:
            source, record = "D_replication.per_layer", dict(
                per_layer_replication[str(layer)]
            )
        elif layer in per_layer_replication:
            source, record = "D_replication.per_layer", dict(
                per_layer_replication[layer]
            )
        else:
            per_layer[layer] = {
                "layer": layer,
                "source": None,
                "verdict": "NOT_EVALUATED",
                "supported": False,
                "note": "no causal verdict recorded for this layer",
            }
            continue

        cells = list(record.get("cells") or [])
        supporting = list(record.get("audio_cells_supporting_a_claim") or [])
        admissible_cells = [c for c in cells if c.get("capability_admissible")]
        per_layer[layer] = {
            "layer": layer,
            "source": source,
            "verdict": record.get("verdict"),
            "supported": record.get("verdict") == "SUPPORTED",
            "rationale": record.get("rationale"),
            "concepts_supporting": sorted(
                {str(entry["concept"]) for entry in supporting}
            ),
            "audio_pairs_supporting": sorted(
                {str(entry["pair"]) for entry in supporting}
            ),
            "n_supporting_cells": len(supporting),
            "audio_cells_measured_but_inadmissible": list(
                record.get("audio_cells_measured_but_inadmissible") or []
            ),
            "control_gaps": [
                {
                    "concept": cell.get("concept"),
                    "pair": cell.get("pair"),
                    "effect": cell.get("mean_signed_target_effect"),
                    "random_control": cell.get("random_control"),
                    "unrelated_control": cell.get("unrelated_control"),
                    "raw_residual_control": cell.get("raw_residual_control"),
                    "jspace_beats_raw_direction": cell.get(
                        "jspace_beats_raw_direction"
                    ),
                }
                for cell in admissible_cells
                if cell.get("counted_toward_verdict")
            ],
            "activation_norm_ratios": sorted(
                {
                    round(float(cell["mean_activation_norm_ratio"]), 4)
                    for cell in admissible_cells
                    if cell.get("mean_activation_norm_ratio") is not None
                }
            ),
            "n_distinct_target_images": max(
                (
                    int(cell.get("n_distinct_images") or 0)
                    for cell in admissible_cells
                ),
                default=0,
            ),
            "n_distinct_positive_images": max(
                (int(cell.get("n_positive_images") or 0) for cell in admissible_cells),
                default=0,
            ),
            "n_distinct_negative_images": max(
                (int(cell.get("n_negative_images") or 0) for cell in admissible_cells),
                default=0,
            ),
        }
    return {
        "schema": "jlens.mmpilot.frozen_causal_evidence.v1",
        "read_only": True,
        "recomputed": False,
        "overall_verdict": (verdicts.get("E_overall") or {}).get("verdict"),
        "capability_verdict": (verdicts.get("A_audio_capability") or {}).get("verdict"),
        "admissibility_rule_version": CLAIM_ADMISSIBILITY_RULE_VERSION,
        "per_layer": per_layer,
    }


def layer_convergence_table(
    classifications: Mapping[int, Mapping],
    causal: Mapping,
    *,
    summary: Mapping,
    layers: Sequence[int] = AUDITED_LAYERS,
) -> list[dict]:
    """One row per layer: convergence beside frozen causal support.

    Layers and concepts are kept apart. Nothing here averages layer 35 with
    layer 40, and nothing averages ``cat`` with ``toilet`` before both are
    printed in their own right.
    """
    rows: list[dict] = []
    for layer in layers:
        layer = int(layer)
        classification = dict(classifications.get(layer) or {})
        evidence = dict((causal.get("per_layer") or {}).get(layer) or {})
        layer_summary = dict((summary.get("per_layer") or {}).get(str(layer)) or {})
        per_modality = layer_summary.get("per_modality") or {}
        pooled = layer_summary.get("pooled_principal") or {}
        rows.append(
            {
                "layer": layer,
                "lens_validity_gate_passed": layer not in LENS_INVALID_LAYERS
                and layer in tuple(int(x) for x in AUDITED_LAYERS),
                "convergence_classification": classification.get("classification"),
                "clean_agreement_unique": classification.get(
                    "pooled_clean_agreement_unique"
                ),
                "clean_agreement_argmax": classification.get(
                    "pooled_clean_agreement_argmax"
                ),
                "target_accuracy_argmax": classification.get(
                    "pooled_target_accuracy_argmax"
                ),
                "median_target_rank": classification.get("pooled_median_target_rank"),
                "median_candidate_entropy_nats": pooled.get(
                    "median_candidate_entropy_nats"
                ),
                "bootstrap_low": (classification.get("pooled_bootstrap") or {}).get("low"),
                "bootstrap_high": (classification.get("pooled_bootstrap") or {}).get(
                    "high"
                ),
                "n_samples": pooled.get("n"),
                "readout_mode": pooled.get("readout_mode"),
                **{
                    f"clean_agreement_unique_{modality}": (
                        per_modality.get(modality) or {}
                    ).get("clean_agreement_unique")
                    for modality in MODALITIES
                },
                **{
                    f"n_{modality}": (per_modality.get(modality) or {}).get("n")
                    for modality in MODALITIES
                },
                "causal_transfer_verdict": evidence.get("verdict"),
                "causal_transfer_supported": bool(evidence.get("supported")),
                "causal_concepts_supporting": ";".join(
                    evidence.get("concepts_supporting") or []
                ),
                "causal_audio_pairs_supporting": ";".join(
                    evidence.get("audio_pairs_supporting") or []
                ),
                "n_supporting_cells": evidence.get("n_supporting_cells"),
                "control_gaps": json.dumps(
                    evidence.get("control_gaps") or [], sort_keys=True
                ),
                "activation_norm_ratios": ";".join(
                    str(x) for x in (evidence.get("activation_norm_ratios") or [])
                ),
                "n_distinct_target_images": evidence.get("n_distinct_target_images"),
                "n_distinct_positive_images": evidence.get(
                    "n_distinct_positive_images"
                ),
                "n_distinct_negative_images": evidence.get(
                    "n_distinct_negative_images"
                ),
                "n_distinct_recordings": pooled.get("n_distinct_recordings"),
            }
        )
    return rows


def write_layer_table_csv(rows: Sequence[Mapping], path: str | os.PathLike[str]) -> Path:
    """Write the layer table as CSV, atomically, with a stable column order."""
    import csv
    import io

    target = Path(path)
    fieldnames = list(rows[0]) if rows else ["layer"]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key) for key in fieldnames})
    _atomic_write_text(target, buffer.getvalue())
    return target


# ---------------------------------------------------------------------- verdict


def convergence_verdict(
    *,
    classifications: Mapping[int, Mapping],
    trajectory: Mapping,
    causal: Mapping,
    controls: Mapping,
    tokenization: Mapping,
    integrity: Mapping,
    criterion: ConvergenceCriterion = CONVERGENCE_CRITERION,
    primary_layer: int = PRIMARY_LAYER,
) -> dict:
    """The conservative three-way verdict.

    Clause order matters and is fixed: :data:`TRANSFER_AT_OR_AFTER_CONVERGENCE`
    is decided first, so the unfavourable outcome can never be masked by a later
    condition failing.
    """
    primary = dict(classifications.get(int(primary_layer)) or {})
    primary_class = primary.get("classification")
    primary_causal = dict((causal.get("per_layer") or {}).get(int(primary_layer)) or {})

    checks: list[dict] = []

    def record(name: str, passed: bool, detail: str) -> bool:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})
        return bool(passed)

    integrity_ok = record(
        "integrity",
        bool(integrity.get("passed")),
        f"completed-run integrity: {integrity.get('summary', 'not reported')}",
    )
    controls_ok = record(
        "controls",
        bool(controls.get("all_controls_passed")),
        f"failed controls: {controls.get('failed_controls') or 'none'}",
    )
    readout_ok = record(
        "readout_is_interpretable",
        tokenization.get("readout_mode")
        in (READOUT_SINGLE_TOKEN, READOUT_FIRST_TOKEN),
        f"readout mode {tokenization.get('readout_mode')!r}",
    )
    distinct = int(primary.get("n_distinct_predictions") or 0)
    not_degenerate = record(
        "readout_not_degenerate",
        distinct >= criterion.min_distinct_predictions,
        f"layer {primary_layer} predicted {distinct} distinct candidate(s); "
        f"need >= {criterion.min_distinct_predictions}. A readout that answers "
        "the same word every time has failed, and a failed readout is not "
        "evidence about the representation.",
    )
    monotone_ok = record(
        "trajectory_monotone_within_tolerance",
        bool(trajectory.get("monotone_within_tolerance")),
        f"non-monotonic drops: {trajectory.get('non_monotonic_drops') or 'none'}",
    )
    causal_ok = record(
        "primary_layer_causal_transfer_still_supported",
        bool(primary_causal.get("supported")),
        f"L{primary_layer} capability-filtered causal verdict "
        f"{primary_causal.get('verdict')!r} (read, not recomputed)",
    )
    later_ok = record(
        "later_validated_layer_clearly_more_converged",
        bool(trajectory.get("any_later_layer_clearly_more_converged")),
        f"later layers: {trajectory.get('later_layers')}",
    )
    admissible_only = record(
        "principal_evidence_is_capability_admissible_only",
        True,
        "every principal number is computed over capability-admissible "
        "concepts; inadmissible concepts are reported descriptively and enter "
        "no verdict clause",
    )

    if primary_class == CONVERGED:
        verdict = TRANSFER_AT_OR_AFTER_CONVERGENCE
        rationale = (
            f"Layer {primary_layer} satisfies the predeclared convergence "
            "criterion in every required modality: its native direct readout "
            "already names the model's own clean answer as the sole maximum on "
            f"{primary.get('pooled_clean_agreement_unique')} of "
            "capability-admissible samples. The established causal transfer at "
            "that layer therefore acts on a representation that has already "
            "converged onto the answer under this criterion, and no "
            "pre-convergence claim is available."
        )
    elif primary_class == NOT_CONVERGED and all(
        [
            integrity_ok,
            controls_ok,
            readout_ok,
            not_degenerate,
            monotone_ok,
            causal_ok,
            later_ok,
            admissible_only,
        ]
    ):
        verdict = PRE_CONVERGENCE_TRANSFER_SUPPORTED
        rationale = (
            f"Layer {primary_layer} is classified NOT_CONVERGED under the "
            "predeclared native direct-readout criterion in every required "
            "modality, its capability-filtered causal-transfer result is "
            "unchanged, at least one later validated layer is clearly more "
            "converged with a disjoint image-level bootstrap interval, and "
            "every control and integrity check holds. The controlled "
            "cross-modal transfer at layer "
            f"{primary_layer} therefore occurs BEFORE native direct-readout "
            "convergence onto the final candidate answer. This is a statement "
            "about convergence under this criterion and about nothing else."
        )
    else:
        verdict = INCONCLUSIVE_CONVERGENCE_TIMING
        failed = [check["check"] for check in checks if not check["passed"]]
        if primary_class == AMBIGUOUS:
            reason = (
                f"layer {primary_layer} falls in the criterion's ambiguous band: "
                "it neither clears the converged bar nor sits below the "
                "not-converged bar. The gap between the two bars exists "
                "precisely so that this case is reported as unresolved rather "
                "than as a weak version of either conclusion."
            )
        elif failed:
            reason = f"required checks did not hold: {failed}"
        else:  # pragma: no cover - defensive
            reason = "the diagnostic did not produce an interpretable result"
        rationale = (
            f"No convergence-timing conclusion is supported: {reason} "
            "The continuous trajectory is reported in full regardless."
        )

    return {
        "schema": "jlens.mmpilot.convergence_verdict.v1",
        "protocol": CONVERGENCE_PROTOCOL,
        "verdict": verdict,
        "rationale": rationale,
        "primary_layer": int(primary_layer),
        "primary_layer_classification": primary_class,
        "layer_classifications": {
            str(layer): classifications[layer]["classification"]
            for layer in sorted(classifications)
        },
        "checks": checks,
        "failed_checks": [check["check"] for check in checks if not check["passed"]],
        "criterion": criterion.to_dict(),
        "criterion_digest": criterion.digest,
        "criterion_text": CRITERION_TEXT,
        "trajectory": dict(trajectory),
        "readout_mode": tokenization.get("readout_mode"),
        "all_candidates_single_token": tokenization.get("all_candidates_single_token"),
        "interpretation_boundary": INTERPRETATION_BOUNDARY,
        "failure_of_direct_readout_is_not_proof": (
            "Failure of the native direct readout is NOT proof that linguistic "
            "information is absent from the representation. It establishes only "
            "that the representation has not converged under the predeclared "
            "criterion stated in this artifact."
        ),
        "wrong_layer_control_note": WRONG_LAYER_CONTROL_NOTE,
    }


def sensitivity_report(
    summary: Mapping,
    *,
    variants: Sequence[tuple[str, ConvergenceCriterion]] = SENSITIVITY_VARIANTS,
    layers: Sequence[int] = AUDITED_LAYERS,
    primary_layer: int = PRIMARY_LAYER,
) -> dict:
    """How each layer would be classified under fixed alternative thresholds.

    Reported beside the primary rule and never in place of it. The alternatives
    are fixed in :data:`SENSITIVITY_VARIANTS` before results exist, for the same
    reason the primary rule is.
    """
    rows: list[dict] = []
    for name, criterion in variants:
        classifications = classify_all_layers(summary, criterion=criterion, layers=layers)
        rows.append(
            {
                "variant": name,
                "criterion_digest": criterion.digest,
                "converged_min_clean_agreement": criterion.converged_min_clean_agreement,
                "not_converged_max_clean_agreement": (
                    criterion.not_converged_max_clean_agreement
                ),
                "classifications": {
                    str(layer): classifications[layer]["classification"]
                    for layer in sorted(classifications)
                },
                "primary_layer_classification": classifications[int(primary_layer)][
                    "classification"
                ],
            }
        )
    return {
        "schema": "jlens.mmpilot.convergence_sensitivity.v1",
        "primary_rule_unchanged": True,
        "variants": rows,
    }


# --------------------------------------------------------------------- integrity


def file_checksum(path: str | os.PathLike[str]) -> str | None:
    """``sha256:`` over a file's bytes, or ``None`` when it is absent."""
    target = Path(path)
    if not target.is_file():
        return None
    return "sha256:" + hashlib.sha256(target.read_bytes()).hexdigest()


def protected_file_checksums(
    run_dir: str | os.PathLike[str], *, names: Sequence[str] = PROTECTED_RUN_FILES
) -> dict[str, str | None]:
    """Checksums of the completed run's protected files, before or after."""
    root = Path(run_dir)
    return {name: file_checksum(root / name) for name in names}


def assert_run_unchanged(
    before: Mapping[str, str | None], after: Mapping[str, str | None]
) -> dict:
    """Prove no protected file moved during the audit.

    Raises:
        CompletedRunModified: On the first difference. A file that appeared is a
            difference too: this audit writes nothing into the completed run.
    """
    changed = [
        {"file": name, "before": before.get(name), "after": after.get(name)}
        for name in sorted(set(before) | set(after))
        if before.get(name) != after.get(name)
    ]
    if changed:
        raise CompletedRunModified(
            "the completed run changed during the audit, which must never "
            f"happen: {changed}"
        )
    return {
        "schema": "jlens.mmpilot.completed_run_immutability.v1",
        "checked_files": sorted(set(before) | set(after)),
        "unchanged": True,
        "checksums": dict(sorted(before.items())),
    }


def verify_completed_run(
    *,
    run_dir: str | os.PathLike[str],
    fingerprint_payload: Mapping,
    expected_fingerprint_digest: str,
    expected_model_repo_id: str,
    expected_model_revision: str,
    expected_processor_revision: str,
    expected_audio_protocol_version: str,
    expected_audio_protocol_fingerprint: str | None,
    expected_lens_checksums: Mapping[int, str],
    expected_combined_lens_checksum: str,
    summary: Mapping,
    layers: Sequence[int] = AUDITED_LAYERS,
) -> dict:
    """Check every Stage-1 precondition. Incomplete state is refused, not patched.

    Everything compared here is read from the completed run's own artifacts; no
    value is inferred and none is defaulted. A missing field fails the check it
    belongs to rather than being treated as agreement.

    Raises:
        ConvergenceRefused: On the first mismatch, naming the field.
    """
    checks: list[dict] = []

    def require(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(ok), "detail": detail})
        if not ok:
            raise ConvergenceRefused(f"{name} failed: {detail}")

    root = Path(run_dir)
    require("run_directory_exists", root.is_dir(), f"{root} is not a directory")

    stored = dict(fingerprint_payload)
    stored.pop("written_utc", None)
    require(
        "run_fingerprint_digest",
        payload_checksum(stored) == expected_fingerprint_digest,
        f"stored fingerprint digest {payload_checksum(stored)} != expected "
        f"{expected_fingerprint_digest}",
    )
    require(
        "model_repo_id",
        stored.get("model_repo_id") == expected_model_repo_id,
        f"run records {stored.get('model_repo_id')!r}, expected {expected_model_repo_id!r}",
    )
    require(
        "model_revision",
        stored.get("model_revision") == expected_model_revision,
        f"run records {stored.get('model_revision')!r}, expected {expected_model_revision!r}",
    )
    require(
        "processor_revision",
        stored.get("processor_revision") == expected_processor_revision,
        f"run records {stored.get('processor_revision')!r}, expected "
        f"{expected_processor_revision!r}",
    )

    extra = dict(stored.get("extra") or {})
    require(
        "audio_protocol_version",
        extra.get("audio_protocol_version") == expected_audio_protocol_version,
        f"run records {extra.get('audio_protocol_version')!r}, expected "
        f"{expected_audio_protocol_version!r}",
    )
    if expected_audio_protocol_fingerprint is not None:
        require(
            "audio_protocol_fingerprint",
            extra.get("audio_protocol_fingerprint")
            == expected_audio_protocol_fingerprint,
            f"run records {extra.get('audio_protocol_fingerprint')!r}, expected "
            f"{expected_audio_protocol_fingerprint!r}",
        )

    require(
        "combined_lens_checksum",
        stored.get("lens_checksum") == expected_combined_lens_checksum,
        f"run records {stored.get('lens_checksum')!r}, expected "
        f"{expected_combined_lens_checksum!r}",
    )
    per_layer = {
        int(layer): str(value)
        for layer, value in (extra.get("per_layer_lens_checksums") or {}).items()
    }
    for layer in layers:
        assert_lens_valid_layer(int(layer), audited=layers)
        require(
            f"lens_checksum_L{int(layer)}",
            per_layer.get(int(layer)) == str(expected_lens_checksums[int(layer)]),
            f"run records {per_layer.get(int(layer))!r} for layer {int(layer)}, "
            f"expected {expected_lens_checksums[int(layer)]!r}",
        )
    require(
        "audited_layers_are_run_layers",
        set(int(x) for x in layers).issubset(set(int(x) for x in stored.get("layers") or [])),
        f"run layers {stored.get('layers')} do not cover audited layers {list(layers)}",
    )
    for invalid in LENS_INVALID_LAYERS:
        require(
            f"lens_invalid_layer_{invalid}_not_audited",
            int(invalid) not in set(int(x) for x in layers),
            f"layer {invalid} failed confirmation and may never be audited here",
        )

    lens_validation = dict(summary.get("lens_validation") or {})
    require(
        "lens_confirmation_status_recorded",
        bool(lens_validation),
        "the completed run's summary carries no lens_validation record",
    )
    verdicts = dict(summary.get("verdicts") or {})
    require(
        "capability_filtered_verdicts_present",
        bool(verdicts.get("C_primary_causal")) and bool(verdicts.get("E_overall")),
        "the summary carries no C_primary_causal / E_overall verdicts",
    )
    overall = dict(verdicts.get("E_overall") or {})
    require(
        "overall_verdict_recorded",
        bool(overall.get("verdict")),
        "the overall three-modality verdict is missing",
    )

    return {
        "schema": "jlens.mmpilot.convergence_integrity.v1",
        "passed": True,
        "summary": f"{len(checks)} precondition checks passed",
        "run_dir": str(root),
        "run_fingerprint_digest": expected_fingerprint_digest,
        "model_repo_id": expected_model_repo_id,
        "model_revision": expected_model_revision,
        "processor_revision": expected_processor_revision,
        "audio_protocol_version": expected_audio_protocol_version,
        "lens_checksums": {str(k): str(v) for k, v in sorted(expected_lens_checksums.items())},
        "combined_lens_checksum": expected_combined_lens_checksum,
        "completed_overall_verdict": overall.get("verdict"),
        "checks": checks,
    }


# ---------------------------------------------------------------- the population


def build_population(
    *,
    activations: Sequence[Mapping],
    clean_predictions: Mapping[str, str],
    capability: Mapping,
    focal_concepts: Sequence[str],
    layers: Sequence[int] = AUDITED_LAYERS,
) -> dict:
    """The frozen evaluation population, taken as-is from the completed run.

    ``activations`` are the completed run's stored ``activation`` units — the
    clean final-prompt-token residuals. Nothing is reselected: every unit whose
    layer is audited, whose modality is one of the three, and whose concept is
    in the predeclared focal set is included, and each one carries the
    admissibility label the completed run's own rule assigns it.

    ``clean_predictions`` maps ``sample_id`` to the model's clean final
    candidate prediction, recovered from the completed run's zero-alpha
    intervention units. A sample without one keeps ``None`` and is excluded from
    the agreement metric alone, not from the population.
    """
    roster = concept_admissibility(list(focal_concepts), capability=capability)
    admissible = set(roster["eligible_concepts"])
    audited = {int(x) for x in layers}

    units: list[dict] = []
    for record in activations:
        layer = int(record["layer"])
        modality = str(record["modality"])
        concept = record.get("concept")
        if layer not in audited or modality not in MODALITIES:
            continue
        if concept is None or concept not in set(focal_concepts):
            continue
        sample_id = str(record["sample_id"])
        units.append(
            {
                "sample_id": sample_id,
                "group_id": str(record["group_id"]),
                "image_id": str(record["image_id"]),
                "recording_id": str(record["group_id"]),
                "concept": str(concept),
                "modality": modality,
                "layer": layer,
                "split": str(record.get("split", "test")),
                "capability_admissible": str(concept) in admissible,
                "activation": list(record["activation"]),
                "activation_checksum": str(record["activation_checksum"]),
                "clean_final_prediction": clean_predictions.get(sample_id),
            }
        )
    units.sort(key=lambda u: (u["layer"], u["modality"], u["sample_id"]))
    return {
        "schema": "jlens.mmpilot.convergence_population.v1",
        "units": units,
        "n_units": len(units),
        "focal_concepts": list(focal_concepts),
        "admissible_concepts": sorted(admissible),
        "inadmissible_concepts": sorted(set(focal_concepts) - admissible),
        "admissibility": roster,
        "layers": sorted(audited),
        "modalities": list(MODALITIES),
        "n_with_clean_reference": sum(
            1 for unit in units if unit["clean_final_prediction"] is not None
        ),
        "reselection_performed": False,
        "note": (
            "the concepts, examples, modalities, candidate set and split are the "
            "completed run's. Nothing here reselects any of them."
        ),
    }


def clean_predictions_from_interventions(
    interventions: Iterable[Mapping],
) -> dict[str, str]:
    """Recover the model's clean final answer per sample from stored units.

    Every intervention unit records ``clean_prediction`` — the unedited
    teacher-forced argmax over the six candidates for that exact input — so the
    reference answer is read out of the completed run rather than recomputed
    with a fresh forward pass.

    Raises:
        ConvergenceRefused: If two units disagree about one sample's clean
            prediction. That would mean the "clean" reference is not a property
            of the sample, and no agreement metric built on it would mean
            anything.
    """
    out: dict[str, str] = {}
    for record in interventions:
        sample_id = record.get("sample_id")
        prediction = record.get("clean_prediction")
        if sample_id is None or prediction is None:
            continue
        sample_id, prediction = str(sample_id), str(prediction)
        if sample_id in out and out[sample_id] != prediction:
            raise ConvergenceRefused(
                f"stored units disagree about the clean final prediction for "
                f"{sample_id}: {out[sample_id]!r} and {prediction!r}"
            )
        out[sample_id] = prediction
    return out


# ------------------------------------------------------------------------- store


@dataclass(frozen=True)
class ConvergenceFingerprint:
    """What an audit's stored rows were produced from.

    Binds the completed run, the model revision, the candidate set, the audited
    layers, the readout convention, the criterion and the code/config version.
    Any change refuses the resume rather than mixing rows produced under two
    different rules.
    """

    protocol: str
    completed_run_fingerprint_digest: str
    completed_run_dir: str
    model_repo_id: str
    model_revision: str
    processor_revision: str
    layers: tuple[int, ...]
    candidate_digest: str
    readout_mode: str
    head_checksum: str
    criterion_digest: str
    code_version: str
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["layers"] = list(self.layers)
        return payload

    @property
    def digest(self) -> str:
        return payload_checksum(self.to_dict())

    def differences(self, other: ConvergenceFingerprint | Mapping) -> list[str]:
        theirs = (
            other.to_dict() if isinstance(other, ConvergenceFingerprint) else dict(other)
        )
        mine = self.to_dict()
        out: list[str] = []
        for key in sorted(set(mine) | set(theirs)):
            a, b = mine.get(key, "<absent>"), theirs.get(key, "<absent>")
            if canonical_json(a) != canonical_json(b):
                out.append(f"{key}: stored={b!r} requested={a!r}")
        return out


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


class ConvergenceStore:
    """Atomic, checksum-validated per-unit rows under ``root``.

    Same two rules as every other store in this repository, for the same reason:
    a Colab runtime can vanish mid-run. Each unit carries a checksum of its own
    payload, and the directory records what its rows came from. A torn file is
    treated as missing; a fingerprint mismatch is refused.

    ``root`` is a **new** audit directory outside the completed run. Nothing
    here can write into the run being audited.
    """

    SCHEMA = "jlens.mmpilot.convergence.unit.v1"

    def __init__(
        self, root: str | os.PathLike[str], fingerprint: ConvergenceFingerprint
    ) -> None:
        self.root = Path(root)
        self.fingerprint = fingerprint
        self.status: str | None = None
        self.invalid_units: list[str] = []

    @property
    def fingerprint_path(self) -> Path:
        return self.root / "fingerprint.json"

    @property
    def results_dir(self) -> Path:
        return self.root / "readout_units"

    def open(self) -> str:
        """Create or validate the audit directory.

        Raises:
            IncompatibleStateError: If the stored fingerprint disagrees. The
                message lists every differing field.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.fingerprint_path.is_file():
            _atomic_write_text(
                self.fingerprint_path,
                json.dumps(self.fingerprint.to_dict(), indent=2, default=str),
            )
            self.status = "starting"
            return self.status
        stored = json.loads(self.fingerprint_path.read_text(encoding="utf-8"))
        stored.pop("written_utc", None)
        if payload_checksum(stored) != self.fingerprint.digest:
            diffs = "\n  ".join(self.fingerprint.differences(stored))
            raise IncompatibleStateError(
                f"{self.root} holds convergence rows from a different "
                f"configuration; refusing to reuse them.\n  {diffs}\n"
                "Point the audit at a new directory (or delete this one yourself)."
            )
        self.status = "resuming"
        return self.status

    def unit_key(self, *, variant: str, layer: int, modality: str, sample_id: str) -> str:
        return safe_key(variant, f"L{int(layer)}", modality, sample_id)

    def unit_path(self, key: str) -> Path:
        return self.results_dir / f"{key}.json"

    def save(self, key: str, payload: Mapping) -> Path:
        path = self.unit_path(key)
        _atomic_write_text(
            path,
            json.dumps(
                {
                    "schema": self.SCHEMA,
                    "key": key,
                    "fingerprint_digest": self.fingerprint.digest,
                    "unit_checksum": payload_checksum(payload),
                    "payload": dict(payload),
                },
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
        )
        return path

    def load(self, key: str) -> dict | None:
        path = self.unit_path(key)
        if not path.is_file():
            return None
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            payload = record["payload"]
            valid = (
                record.get("unit_checksum") == payload_checksum(payload)
                and record.get("fingerprint_digest") == self.fingerprint.digest
                and record.get("key") == key
            )
        except (json.JSONDecodeError, KeyError, TypeError, UnicodeDecodeError):
            valid, payload = False, None
        if not valid:
            self.invalid_units.append(str(path))
            return None
        return payload

    def has(self, key: str) -> bool:
        return self.load(key) is not None

    def load_all(self) -> list[dict]:
        if not self.results_dir.is_dir():
            return []
        rows: list[dict] = []
        for path in sorted(self.results_dir.glob("*.json")):
            payload = self.load(path.stem)
            if payload is not None:
                rows.append(payload)
        return rows

    def write_artifact(self, name: str, payload: Mapping | Sequence | str) -> Path:
        path = self.root / name
        if isinstance(payload, str):
            _atomic_write_text(path, payload)
        else:
            _atomic_write_text(
                path, json.dumps(payload, indent=2, ensure_ascii=False, default=str)
            )
        return path

    def write_jsonl(self, name: str, rows: Sequence[Mapping]) -> Path:
        path = self.root / name
        _atomic_write_text(
            path,
            "".join(
                json.dumps(row, sort_keys=True, ensure_ascii=False, default=str) + "\n"
                for row in rows
            ),
        )
        return path

    def status_report(self) -> dict:
        present = (
            len(list(self.results_dir.glob("*.json"))) if self.results_dir.is_dir() else 0
        )
        return {
            "audit_dir": str(self.root),
            "status": self.status or "unopened",
            "fingerprint_digest": self.fingerprint.digest,
            "stored_units": present,
            "invalid_units": list(self.invalid_units),
        }


# ------------------------------------------------------------------- figures


def _svg(width: int, height: int, body: str, title: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="Helvetica,Arial,sans-serif">'
        f"<title>{title}</title>"
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>'
        f"{body}</svg>\n"
    )


def figure_convergence_versus_layer(
    table: Sequence[Mapping], *, criterion: ConvergenceCriterion = CONVERGENCE_CRITERION
) -> str:
    """Clean agreement against layer, with both criterion bars drawn.

    Deterministic SVG rather than a plotting dependency: the repository has
    none, and a figure whose bytes are a pure function of the numbers is one a
    test can pin.
    """
    width, height, pad = 640, 360, 60
    plot_w, plot_h = width - 2 * pad, height - 2 * pad
    layers = [int(row["layer"]) for row in table]
    if not layers:
        return _svg(width, height, "", "convergence versus layer")
    lo, hi = min(layers), max(layers)
    span = max(hi - lo, 1)

    def x_of(layer: int) -> float:
        return pad + plot_w * (layer - lo) / span

    def y_of(value: float) -> float:
        return pad + plot_h * (1.0 - max(0.0, min(1.0, value)))

    parts = [
        f'<rect x="{pad}" y="{pad}" width="{plot_w}" height="{plot_h}" '
        'fill="none" stroke="#cccccc"/>'
    ]
    for value, colour, label in (
        (criterion.converged_min_clean_agreement, "#2a7f3f", "converged bar"),
        (criterion.not_converged_max_clean_agreement, "#b03030", "not-converged bar"),
    ):
        y = y_of(value)
        parts.append(
            f'<line x1="{pad}" y1="{y:.1f}" x2="{pad + plot_w}" y2="{y:.1f}" '
            f'stroke="{colour}" stroke-dasharray="5,4"/>'
        )
        parts.append(
            f'<text x="{pad + plot_w - 4:.1f}" y="{y - 5:.1f}" font-size="11" '
            f'fill="{colour}" text-anchor="end">{label} {value:.2f}</text>'
        )
    points = [
        (x_of(int(row["layer"])), y_of(float(row["clean_agreement_unique"] or 0.0)))
        for row in table
        if row.get("clean_agreement_unique") is not None
    ]
    if len(points) > 1:
        path = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        parts.append(
            f'<polyline points="{path}" fill="none" stroke="#1f4e9c" stroke-width="2"/>'
        )
    for row in table:
        value = row.get("clean_agreement_unique")
        if value is None:
            continue
        x, y = x_of(int(row["layer"])), y_of(float(value))
        low, high = row.get("bootstrap_low"), row.get("bootstrap_high")
        if low is not None and high is not None:
            parts.append(
                f'<line x1="{x:.1f}" y1="{y_of(float(low)):.1f}" x2="{x:.1f}" '
                f'y2="{y_of(float(high)):.1f}" stroke="#1f4e9c" stroke-width="1"/>'
            )
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#1f4e9c"/>')
        parts.append(
            f'<text x="{x:.1f}" y="{pad + plot_h + 18:.1f}" font-size="12" '
            f'text-anchor="middle">L{int(row["layer"])}</text>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{y - 10:.1f}" font-size="11" '
            f'text-anchor="middle" fill="#333333">{float(value):.2f}</text>'
        )
    parts.append(
        f'<text x="{pad}" y="{pad - 22}" font-size="14" font-weight="bold">'
        "Native direct-readout convergence versus layer</text>"
    )
    parts.append(
        f'<text x="{pad}" y="{pad - 6}" font-size="11" fill="#555555">'
        "agreement with the model&#39;s own clean final answer (sole maximum), "
        "capability-admissible concepts, image-level 95% CI</text>"
    )
    return _svg(width, height, "".join(parts), "convergence versus layer")


def figure_causal_versus_convergence(table: Sequence[Mapping]) -> str:
    """Causal support beside convergence classification, one row per layer."""
    width = 640
    height = 120 + 48 * max(len(table), 1)
    parts = [
        '<text x="30" y="40" font-size="14" font-weight="bold">'
        "Causal support versus convergence status</text>",
        '<text x="30" y="60" font-size="11" fill="#555555">'
        "causal verdicts are read from the completed run, never recomputed</text>",
    ]
    colours = {
        CONVERGED: "#2a7f3f",
        NOT_CONVERGED: "#b03030",
        AMBIGUOUS: "#a08020",
    }
    for index, row in enumerate(table):
        y = 96 + 48 * index
        classification = str(row.get("convergence_classification"))
        supported = bool(row.get("causal_transfer_supported"))
        parts.append(
            f'<text x="30" y="{y}" font-size="13" font-weight="bold">'
            f'L{int(row["layer"])}</text>'
        )
        parts.append(
            f'<rect x="80" y="{y - 14}" width="160" height="20" rx="4" '
            f'fill="{colours.get(classification, "#888888")}" opacity="0.18"/>'
        )
        parts.append(
            f'<text x="88" y="{y}" font-size="12" '
            f'fill="{colours.get(classification, "#333333")}">{classification}</text>'
        )
        parts.append(
            f'<rect x="260" y="{y - 14}" width="200" height="20" rx="4" '
            f'fill="{"#2a7f3f" if supported else "#888888"}" opacity="0.18"/>'
        )
        parts.append(
            f'<text x="268" y="{y}" font-size="12">causal '
            f'{row.get("causal_transfer_verdict")}</text>'
        )
        parts.append(
            f'<text x="480" y="{y}" font-size="11" fill="#555555">'
            f'{row.get("causal_concepts_supporting") or "-"}</text>'
        )
    return _svg(width, height, "".join(parts), "causal versus convergence")


def figure_per_modality_trajectories(table: Sequence[Mapping]) -> str:
    """One line per modality, so a layer that converged in text only is visible."""
    width, height, pad = 640, 360, 60
    plot_w, plot_h = width - 2 * pad, height - 2 * pad
    layers = [int(row["layer"]) for row in table]
    if not layers:
        return _svg(width, height, "", "per-modality trajectories")
    lo, hi = min(layers), max(layers)
    span = max(hi - lo, 1)
    colours = {"text": "#1f4e9c", "image": "#2a7f3f", "spoken_audio": "#b03030"}
    parts = [
        f'<rect x="{pad}" y="{pad}" width="{plot_w}" height="{plot_h}" '
        'fill="none" stroke="#cccccc"/>',
        f'<text x="{pad}" y="{pad - 22}" font-size="14" font-weight="bold">'
        "Per-modality convergence trajectories</text>",
    ]
    for index, modality in enumerate(MODALITIES):
        points = []
        for row in table:
            value = row.get(f"clean_agreement_unique_{modality}")
            if value is None:
                continue
            x = pad + plot_w * (int(row["layer"]) - lo) / span
            y = pad + plot_h * (1.0 - max(0.0, min(1.0, float(value))))
            points.append((x, y))
        if points:
            path = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
            parts.append(
                f'<polyline points="{path}" fill="none" '
                f'stroke="{colours[modality]}" stroke-width="2"/>'
            )
            for x, y in points:
                parts.append(
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" '
                    f'fill="{colours[modality]}"/>'
                )
        parts.append(
            f'<text x="{pad + 8}" y="{pad + 16 + 16 * index}" font-size="12" '
            f'fill="{colours[modality]}">{modality}</text>'
        )
    for row in table:
        x = pad + plot_w * (int(row["layer"]) - lo) / span
        parts.append(
            f'<text x="{x:.1f}" y="{pad + plot_h + 18:.1f}" font-size="12" '
            f'text-anchor="middle">L{int(row["layer"])}</text>'
        )
    return _svg(width, height, "".join(parts), "per-modality trajectories")


# -------------------------------------------------------------------- reporting


def _fmt(value: Any, spec: str = ".3f") -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return format(float(value), spec)
    return str(value)


def convergence_report_markdown(
    *,
    verdict: Mapping,
    summary: Mapping,
    table: Sequence[Mapping],
    controls: Mapping,
    sensitivity: Mapping,
    integrity: Mapping,
    head_audit: Mapping,
    tokenization: Mapping,
    population: Mapping,
    causal: Mapping,
    probe: Mapping | None = None,
) -> str:
    """The audit report, in the order a reader needs it."""
    lines = [
        "# Output-convergence timing audit — L35 / L38 / L40",
        "",
        f"**Verdict: {verdict['verdict']}**",
        "",
        verdict["rationale"],
        "",
        "> " + INTERPRETATION_BOUNDARY.replace("\n", " "),
        "",
        "## What was measured",
        "",
        "The completed three-modality run is **read only**. No capability gate, "
        "activation extraction, J-space pursuit, direction estimation or causal "
        "intervention was rerun. The single new measurement is the model's own "
        "frozen output head applied to the run's **stored** clean "
        "final-prompt-token residuals.",
        "",
        f"- readout: `{head_audit.get('readout_expression')}`",
        f"- final norm: `{head_audit.get('final_norm_class')}` "
        f"(weight convention observed: `{head_audit.get('norm_weight_convention')}`)",
        f"- softcap: `{head_audit.get('final_logit_softcapping')}` "
        f"(applied: {_fmt(head_audit.get('softcap_applied'))})",
        f"- agrees with the model's own `unembed`: "
        f"{_fmt(head_audit.get('matches_model_unembed'))} "
        f"(max abs difference {_fmt(head_audit.get('max_abs_difference_vs_model_unembed'), '.2e')})",
        f"- readout mode: `{tokenization.get('readout_mode')}`",
        f"- every candidate single-token: "
        f"{_fmt(tokenization.get('all_candidates_single_token'))}",
        "",
        tokenization.get("scoring_note", ""),
        "",
        "## Provenance and integrity",
        "",
        f"- completed run: `{integrity.get('run_dir')}`",
        f"- run fingerprint: `{integrity.get('run_fingerprint_digest')}`",
        f"- model: `{integrity.get('model_repo_id')}` @ "
        f"`{integrity.get('model_revision')}`",
        f"- processor revision: `{integrity.get('processor_revision')}`",
        f"- audio protocol: `{integrity.get('audio_protocol_version')}`",
        f"- lens checksums: `{integrity.get('lens_checksums')}`",
        f"- completed run's own verdict: "
        f"`{integrity.get('completed_overall_verdict')}` (unchanged)",
        "",
        "## Evaluation population",
        "",
        f"- units: {population.get('n_units')} "
        f"(with a recorded clean reference answer: "
        f"{population.get('n_with_clean_reference')})",
        f"- capability-admissible concepts: "
        f"{population.get('admissible_concepts')}",
        f"- excluded as capability-ineligible: "
        f"{population.get('inadmissible_concepts')} — descriptive only, in no "
        "principal number and in no verdict clause",
        f"- modalities: {population.get('modalities')}",
        "",
        "## Per-concept results, before any pooling",
        "",
    ]

    per_layer = summary.get("per_layer") or {}
    for layer_key in sorted(per_layer, key=int):
        entry = per_layer[layer_key]
        lines += [f"### Layer {entry['layer']}", ""]
        lines += [
            "| modality | concept | n | clean agreement (unique) | "
            "target accuracy (argmax) | median rank | median margin | entropy (nats) |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for modality in MODALITIES:
            cell = (entry["per_modality"] or {}).get(modality) or {}
            for concept, stats in sorted((cell.get("per_concept") or {}).items()):
                lines.append(
                    f"| {modality} | {concept} | {stats.get('n')} | "
                    f"{_fmt(stats.get('clean_agreement_unique'))} | "
                    f"{_fmt(stats.get('target_accuracy_argmax'))} | "
                    f"{_fmt(stats.get('median_target_rank'), '.1f')} | "
                    f"{_fmt(stats.get('median_target_margin'))} | "
                    f"{_fmt(stats.get('median_candidate_entropy_nats'))} |"
                )
            if cell:
                boot = cell.get("bootstrap_clean_agreement_unique") or {}
                lines.append(
                    f"| {modality} | **pooled** | {cell.get('n')} | "
                    f"{_fmt(cell.get('clean_agreement_unique'))} "
                    f"[{_fmt(boot.get('low'))}, {_fmt(boot.get('high'))}] | "
                    f"{_fmt(cell.get('target_accuracy_argmax'))} | "
                    f"{_fmt(cell.get('median_target_rank'), '.1f')} | "
                    f"{_fmt(cell.get('median_target_margin'))} | "
                    f"{_fmt(cell.get('median_candidate_entropy_nats'))} |"
                )
        lines.append("")

    lines += [
        "## Layer table — convergence beside frozen causal evidence",
        "",
        "| layer | lens gate | convergence | clean agreement | median rank | "
        "causal verdict | concepts | audio pairs | distinct target images |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in table:
        lines.append(
            f"| L{row['layer']} | "
            f"{'PASS' if row['lens_validity_gate_passed'] else 'FAIL'} | "
            f"{row['convergence_classification']} | "
            f"{_fmt(row['clean_agreement_unique'])} | "
            f"{_fmt(row['median_target_rank'], '.1f')} | "
            f"{row['causal_transfer_verdict']} | "
            f"{row['causal_concepts_supporting'] or '-'} | "
            f"{row['causal_audio_pairs_supporting'] or '-'} | "
            f"{row['n_distinct_target_images']} |"
        )

    lines += [
        "",
        "## Controls",
        "",
        f"All controls passed: {_fmt(controls.get('all_controls_passed'))}"
        + (
            f" (failures: {controls.get('failed_controls')})"
            if controls.get("failed_controls")
            else ""
        ),
        "",
        "| layer | control | compared on | primary | control | passed |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for layer_key in sorted(controls.get("per_layer") or {}, key=int):
        entry = (controls["per_layer"])[layer_key]
        for name, control in sorted((entry.get("controls") or {}).items()):
            lines.append(
                f"| L{layer_key} | `{name}` | {control['compared_field']} | "
                f"{_fmt(control['primary_value'])} | "
                f"{_fmt(control['control_value'])} | "
                f"{'yes' if control['passed'] else 'NO'} |"
            )
    lines += ["", "> " + WRONG_LAYER_CONTROL_NOTE, ""]

    if probe is not None:
        lines += [
            "## Secondary linear probe (does not determine the verdict)",
            "",
            f"- ran: {_fmt(probe.get('ran'))}",
            f"- image-disjoint cross-validated accuracy: "
            f"{_fmt(probe.get('mean_accuracy'))} (chance {_fmt(probe.get('chance'))})",
            "",
            "> " + str(probe.get("caveat", "")),
            "",
        ]

    lines += [
        "## Sensitivity at fixed alternative thresholds",
        "",
        "The primary rule is unchanged by anything in this section.",
        "",
        "| variant | converged bar | not-converged bar | L35 | L38 | L40 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for entry in sensitivity.get("variants") or []:
        classes = entry["classifications"]
        lines.append(
            f"| `{entry['variant']}` | "
            f"{_fmt(entry['converged_min_clean_agreement'], '.2f')} | "
            f"{_fmt(entry['not_converged_max_clean_agreement'], '.2f')} | "
            + " | ".join(classes.get(str(layer), "-") for layer in AUDITED_LAYERS)
            + " |"
        )

    lines += [
        "",
        "## Predeclared criterion",
        "",
        "```",
        str(verdict["criterion_text"]).rstrip(),
        "```",
        "",
        "## Verdict checks",
        "",
    ]
    for check in verdict["checks"]:
        lines.append(
            f"- **{'PASS' if check['passed'] else 'FAIL'}** `{check['check']}` — "
            f"{check['detail']}"
        )
    lines += [
        "",
        "## What this does and does not establish",
        "",
        "- It establishes how far the clean final-prompt-token residual has "
        "converged onto the model's own answer at each validated layer, under "
        "one predeclared, non-learned, model-owned readout.",
        "- It does **not** establish the absence of linguistic information. "
        + verdict["failure_of_direct_readout_is_not_proof"],
        "- It does not revisit, recompute or amend the completed causal result. "
        f"That run's verdict remains `{causal.get('overall_verdict')}`.",
        "",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------- driver


def _control_rows(
    units: Sequence[Mapping],
    *,
    head: NativeHead,
    tokenization: Mapping,
    head_checksum: str,
    config_hash: str,
    seed: int,
) -> Iterable[tuple[str, dict]]:
    """Yield ``(variant, kwargs)`` for every control row of one layer/modality cell.

    The controls share the primary readout's head, population and code path.
    Only one thing changes in each, so a control that reaches the primary
    result names exactly which assumption failed.
    """
    labels = shuffled_label_assignment(units, seed=seed)
    assignment = permuted_token_assignment(tokenization, seed=seed + 1)
    partner = permuted_activation_assignment(units, seed=seed + 2)
    by_id = {str(unit["sample_id"]): unit for unit in units}

    for unit in units:
        sample_id = str(unit["sample_id"])
        base = {
            "head": head,
            "tokenization": tokenization,
            "concept": unit["concept"],
            "clean_prediction": unit["clean_final_prediction"],
            "sample_id": sample_id,
            "group_id": unit["group_id"],
            "image_id": unit["image_id"],
            "recording_id": unit["recording_id"],
            "modality": unit["modality"],
            "layer": unit["layer"],
            "split": unit["split"],
            "capability_admissible": unit["capability_admissible"],
            "activation_checksum": unit["activation_checksum"],
            "head_checksum": head_checksum,
            "config_hash": config_hash,
        }
        activation = torch.tensor(unit["activation"], dtype=torch.float32)
        yield (
            "shuffled_target_labels",
            {
                **base,
                "activation": activation,
                "variant": "shuffled_target_labels",
                "label_override": labels[sample_id],
            },
        )
        yield (
            "permuted_candidate_tokens",
            {
                **base,
                "activation": activation,
                "variant": "permuted_candidate_tokens",
                "token_assignment": assignment,
            },
        )
        other = by_id[partner[sample_id]]
        yield (
            "permuted_activations",
            {
                **base,
                "activation": torch.tensor(other["activation"], dtype=torch.float32),
                "variant": "permuted_activations",
                "activation_checksum": str(other["activation_checksum"]),
            },
        )


def run_convergence_audit(
    *,
    population: Mapping,
    head: NativeHead,
    tokenization: Mapping,
    head_audit: Mapping,
    integrity: Mapping,
    completed_summary: Mapping,
    store: ConvergenceStore,
    criterion: ConvergenceCriterion = CONVERGENCE_CRITERION,
    layers: Sequence[int] = AUDITED_LAYERS,
    run_probe: bool = False,
    write_figures: bool = True,
    control_seed: int = 20260806,
) -> dict:
    """Score the population, apply the criterion, and write every artifact.

    Resume is per unit: a row already stored under this fingerprint is reused
    and not recomputed, so a disconnected Colab session picks up where it left
    off. Nothing is written into the completed run — ``store.root`` is a new
    audit directory.
    """
    for layer in layers:
        assert_lens_valid_layer(int(layer), audited=layers)

    config_hash = payload_checksum(
        {
            "protocol": CONVERGENCE_PROTOCOL,
            "criterion": criterion.to_dict(),
            "tokenization": tokenization["digest"],
            "layers": [int(x) for x in layers],
            "readout_mode": tokenization["readout_mode"],
        }
    )
    head_checksum = str(head_audit.get("head_checksum", ""))
    units = list(population["units"])

    computed = reused = 0
    rows: list[dict] = []

    def store_row(variant: str, payload: dict) -> None:
        nonlocal computed
        key = store.unit_key(
            variant=variant,
            layer=int(payload["layer"]),
            modality=str(payload["modality"]),
            sample_id=str(payload["sample_id"]),
        )
        store.save(key, payload)
        rows.append(payload)
        computed += 1

    for unit in units:
        key = store.unit_key(
            variant=PRIMARY_VARIANT,
            layer=int(unit["layer"]),
            modality=str(unit["modality"]),
            sample_id=str(unit["sample_id"]),
        )
        cached = store.load(key)
        if cached is not None:
            rows.append(cached)
            reused += 1
            continue
        store_row(
            PRIMARY_VARIANT,
            direct_readout_row(
                activation=torch.tensor(unit["activation"], dtype=torch.float32),
                head=head,
                tokenization=tokenization,
                concept=unit["concept"],
                clean_prediction=unit["clean_final_prediction"],
                sample_id=unit["sample_id"],
                group_id=unit["group_id"],
                image_id=unit["image_id"],
                recording_id=unit["recording_id"],
                modality=unit["modality"],
                layer=unit["layer"],
                split=unit["split"],
                capability_admissible=unit["capability_admissible"],
                activation_checksum=unit["activation_checksum"],
                head_checksum=head_checksum,
                config_hash=config_hash,
            ),
        )

    # Controls are drawn within a (layer, modality) cell so a permutation never
    # crosses a boundary the primary metric is reported across.
    cells: dict[tuple[int, str], list[Mapping]] = {}
    for unit in units:
        cells.setdefault((int(unit["layer"]), str(unit["modality"])), []).append(unit)
    for (layer, modality), cell_units in sorted(cells.items()):
        seed = derived_seed(control_seed, "control", layer, modality)
        for variant, kwargs in _control_rows(
            cell_units,
            head=head,
            tokenization=tokenization,
            head_checksum=head_checksum,
            config_hash=config_hash,
            seed=seed,
        ):
            key = store.unit_key(
                variant=variant,
                layer=layer,
                modality=modality,
                sample_id=str(kwargs["sample_id"]),
            )
            cached = store.load(key)
            if cached is not None:
                rows.append(cached)
                reused += 1
                continue
            store_row(variant, direct_readout_row(**kwargs))

    summary = summarize_rows(rows, criterion=criterion, layers=layers)
    classifications = classify_all_layers(summary, criterion=criterion, layers=layers)
    trajectory = trajectory_report(classifications, criterion=criterion)
    controls = summarize_controls(rows, layers=layers)
    causal = read_frozen_causal_evidence(completed_summary, layers=layers)
    table = layer_convergence_table(classifications, causal, summary=summary, layers=layers)
    verdict = convergence_verdict(
        classifications=classifications,
        trajectory=trajectory,
        causal=causal,
        controls=controls,
        tokenization=tokenization,
        integrity=integrity,
        criterion=criterion,
    )
    sensitivity = sensitivity_report(summary, layers=layers)

    probe = None
    if run_probe:
        primary_units = [u for u in units if int(u["layer"]) == PRIMARY_LAYER]
        probe = secondary_linear_probe(
            [
                row
                for row in rows
                if row["variant"] == PRIMARY_VARIANT
                and int(row["layer"]) == PRIMARY_LAYER
                and row["capability_admissible"]
            ],
            {str(u["sample_id"]): u["activation"] for u in primary_units},
        )

    provenance = {
        "schema": "jlens.mmpilot.convergence_provenance.v1",
        "protocol": CONVERGENCE_PROTOCOL,
        "audit_fingerprint": store.fingerprint.to_dict(),
        "audit_fingerprint_digest": store.fingerprint.digest,
        "integrity": dict(integrity),
        "native_head_audit": dict(head_audit),
        "candidate_tokenization": dict(tokenization),
        "population": {
            key: value for key, value in population.items() if key != "units"
        },
        "criterion": criterion.to_dict(),
        "criterion_digest": criterion.digest,
        "config_hash": config_hash,
        "completed_run_read_only": True,
        "causal_evidence_recomputed": False,
        "model_forwards_executed": 0,
        "interpretation_boundary": INTERPRETATION_BOUNDARY,
    }
    summary_payload = {
        "schema": "jlens.mmpilot.output_convergence_summary.v1",
        "protocol": CONVERGENCE_PROTOCOL,
        "verdict": verdict,
        "layer_classifications": {str(k): v for k, v in sorted(classifications.items())},
        "trajectory": trajectory,
        "measurements": summary,
        "controls": controls,
        "sensitivity": sensitivity,
        "frozen_causal_evidence": causal,
        "layer_table": list(table),
        "secondary_probe": probe,
        "provenance": provenance,
        "resume": store.status_report(),
        "units_computed": computed,
        "units_reused": reused,
    }

    report = convergence_report_markdown(
        verdict=verdict,
        summary=summary,
        table=table,
        controls=controls,
        sensitivity=sensitivity,
        integrity=integrity,
        head_audit=head_audit,
        tokenization=tokenization,
        population=population,
        causal=causal,
        probe=probe,
    )

    written: list[str] = []
    written.append(str(store.write_artifact("output_convergence_report.md", report)))
    written.append(
        str(store.write_artifact("output_convergence_summary.json", summary_payload))
    )
    written.append(str(store.write_artifact("provenance.json", provenance)))
    written.append(
        str(store.write_jsonl("per_sample_direct_readout.jsonl", sorted(
            rows,
            key=lambda r: (r["variant"], int(r["layer"]), r["modality"], r["sample_id"]),
        )))
    )
    written.append(
        str(write_layer_table_csv(table, store.root / "layer_convergence_table.csv"))
    )
    if write_figures:
        for name, svg in (
            ("figure_convergence_versus_layer.svg", figure_convergence_versus_layer(table, criterion=criterion)),
            ("figure_causal_versus_convergence.svg", figure_causal_versus_convergence(table)),
            ("figure_per_modality_trajectories.svg", figure_per_modality_trajectories(table)),
        ):
            written.append(str(store.write_artifact(name, svg)))

    checksums = {
        "schema": "jlens.mmpilot.convergence_checksums.v1",
        "audit_artifacts": {
            Path(path).name: file_checksum(path) for path in sorted(written)
        },
        "completed_run_protected_files": dict(
            (integrity.get("immutability") or {}).get("checksums") or {}
        ),
        "audit_fingerprint_digest": store.fingerprint.digest,
        "criterion_digest": criterion.digest,
    }
    written.append(str(store.write_artifact("checksums.json", checksums)))

    return {
        "verdict": verdict,
        "criterion": criterion,
        "summary": summary_payload,
        "report": report,
        "table": table,
        "rows": rows,
        "classifications": classifications,
        "trajectory": trajectory,
        "controls": controls,
        "sensitivity": sensitivity,
        "causal": causal,
        "probe": probe,
        "provenance": provenance,
        "checksums": checksums,
        "artifacts": written,
        "units_computed": computed,
        "units_reused": reused,
    }


__all__ = [
    "AMBIGUOUS",
    "AUDITED_LAYERS",
    "CONTROL_VARIANTS",
    "CONVERGED",
    "CONVERGENCE_CRITERION",
    "CONVERGENCE_PROTOCOL",
    "CRITERION_TEXT",
    "INCONCLUSIVE_CONVERGENCE_TIMING",
    "INTERPRETATION_BOUNDARY",
    "LENS_INVALID_LAYERS",
    "MODALITIES",
    "NOT_CONVERGED",
    "PRE_CONVERGENCE_TRANSFER_SUPPORTED",
    "PRIMARY_LAYER",
    "PRIMARY_VARIANT",
    "PROTECTED_RUN_FILES",
    "READOUT_FIRST_TOKEN",
    "READOUT_SINGLE_TOKEN",
    "SENSITIVITY_VARIANTS",
    "TRANSFER_AT_OR_AFTER_CONVERGENCE",
    "WRONG_LAYER_CONTROL_NOTE",
    "CandidateTokenizationError",
    "CompletedRunModified",
    "ConvergenceCriterion",
    "ConvergenceFingerprint",
    "ConvergenceRefused",
    "ConvergenceStore",
    "IncompatibleStateError",
    "LensInvalidLayerError",
    "NativeHead",
    "assert_lens_valid_layer",
    "assert_run_unchanged",
    "audit_native_head",
    "bootstrap_rate",
    "build_population",
    "classify_all_layers",
    "classify_layer",
    "clean_predictions_from_interventions",
    "derived_seed",
    "convergence_report_markdown",
    "convergence_verdict",
    "direct_readout_row",
    "figure_causal_versus_convergence",
    "figure_convergence_versus_layer",
    "figure_per_modality_trajectories",
    "file_checksum",
    "head_from_model",
    "image_disjoint_folds",
    "layer_convergence_table",
    "permuted_activation_assignment",
    "permuted_token_assignment",
    "protected_file_checksums",
    "read_frozen_causal_evidence",
    "resolve_candidate_tokens",
    "run_convergence_audit",
    "secondary_linear_probe",
    "sensitivity_report",
    "shuffled_label_assignment",
    "summarize_cell",
    "summarize_controls",
    "summarize_rows",
    "tie_aware_ranks",
    "trajectory_report",
    "verify_completed_run",
    "write_layer_table_csv",
]
