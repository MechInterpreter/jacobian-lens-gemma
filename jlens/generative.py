# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Generative J-cone steering: can weighted reconstructions of active
J-space generators make the model's own decoder verbalize a latent concept?

This module is the isolated backend for the go/no-go experiment on branch
``experiment/generative-jlens-validation``. It reuses the repository's
established conventions without changing them:

- **Site**: the exact ``block_output`` site the lens reads and
  :mod:`jlens.interventions` edits — the tensor returned by
  ``model.language_model.layers[l]`` (post attention, MLP, PLE, and
  ``layer_scalar``).
- **Vector convention**: steering vectors are *weighted reconstructions*
  ``q_C = sum_{i in C} a_i v_i`` over the **raw** J-space dictionary atoms
  (rows of ``W_U @ J_l``), exactly the pursuit's recorded coefficient
  convention (:mod:`jlens.pursuit`). They are not averages, and a selected
  subset ``C`` is a *candidate semantic group*, not a proven binding.
- **Arithmetic**: float32 math cast back to the block dtype, identical to
  :func:`jlens.interventions.residual_intervention`; a zero delta therefore
  reproduces baseline logits exactly.

What is new here, and only here:

1. :class:`SteeringSchedule` — prompt-only injection, constant reinjection
   at every generated-token position, or exponentially decaying reinjection.
2. :func:`steering_injection` — a multi-position variant of the
   single-position intervention hook, driven by a schedule.
3. :func:`greedy_decode` — manual **uncached** autoregressive decoding
   (every step is a full forward pass, so the injection semantics are
   unambiguous). Zero-steering decoding must match ordinary greedy
   generation; that equivalence is a test, not an assumption.
4. :func:`target_logprob` — teacher-forced scoring of a multi-token target
   under steering. With causal attention this single forward pass is exactly
   the sum of per-step decode log-probabilities under the same schedule.
5. Condition builders for the control battery (zero / unrelated cone /
   random matched-norm / coordinate-shuffled / sign-reversed / raw
   activation / activation difference / coefficient-mass subcones / manual
   subcones), plus norm-relative strength scaling.

Anything not implemented raises :class:`GenerativeError` rather than
returning placeholder values.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from jlens.interventions import (
    _assert_frozen,
    _finite_or_raise,
    isotropic_random_direction,
)

GENERATIVE_RECORD_SCHEMA = "jlens.generative.record.v1"

#: Steering-vector conditions every benchmark example supports. ``none`` runs
#: without any hook; ``zero`` injects an exact zero (parity control);
#: ``natural_scale`` is the full cone injected at **its own** norm with no
#: rescaling, so the run records what the cone's intrinsic perturbation
#: magnitude actually does (and what ratio it corresponds to);
#: ``natural_unrelated_cone`` / ``natural_random_matched_norm`` /
#: ``natural_shuffled`` / ``natural_sign_reversed`` / ``natural_mass_subcone``
#: are their ratio-scaled counterparts' *directions*, rescaled to the same
#: injected delta norm as ``natural_scale`` (not to any ratio of the receiving
#: activation) — see :data:`NATURAL_SCALE_MATCHED_CONDITIONS`. Ratio-scaled
#: controls alone cannot show whether a low-strength gain is specific to the
#: correct J-cone's *direction*, because they are injected at a different norm
#: than the natural cone; matching the norm isolates direction from magnitude.
#: ``wrong_layer`` / ``wrong_position`` reuse the correct full-cone vector at
#: an incorrect site (the vector builder returns the same delta; the runner
#: moves the site).
VECTOR_CONDITIONS = (
    "none",
    "zero",
    "full_cone",
    "natural_scale",
    "mass_subcone",
    "manual_subcone",
    "unrelated_cone",
    "random_matched_norm",
    "shuffled",
    "sign_reversed",
    "wrong_layer",
    "wrong_position",
    "raw_activation",
    "activation_diff",
    "natural_unrelated_cone",
    "natural_random_matched_norm",
    "natural_shuffled",
    "natural_sign_reversed",
    "natural_mass_subcone",
)

#: Conditions whose injected delta is rescaled to match ``natural_scale``'s own
#: norm (the correct cone's unscaled magnitude), rather than to a ratio of the
#: receiving activation. Order matches the base conditions they mirror.
NATURAL_SCALE_MATCHED_CONDITIONS = (
    "natural_unrelated_cone",
    "natural_random_matched_norm",
    "natural_shuffled",
    "natural_sign_reversed",
    "natural_mass_subcone",
)


def condition_scaling_mode(condition: str) -> str:
    """How a condition's injected delta norm was determined.

    One of ``"none"`` (``none``/``zero``: no scaling question applies),
    ``"natural_unscaled"`` (``natural_scale``: the cone at its own norm — the
    reference other natural-scale conditions match), ``"natural_matched"``
    (:data:`NATURAL_SCALE_MATCHED_CONDITIONS`: rescaled to that same norm), or
    ``"ratio_scaled"`` (every other condition: rescaled to ``ratio *
    receiving_activation_norm``).

    A pure function of the condition name, used both to tag records at
    construction time and to group :func:`summarize_by_condition` output, so
    the two can never disagree about which bucket a condition belongs to.
    """
    if condition in ("none", "zero"):
        return "none"
    if condition == "natural_scale":
        return "natural_unscaled"
    if condition in NATURAL_SCALE_MATCHED_CONDITIONS:
        return "natural_matched"
    return "ratio_scaled"


SCHEDULE_KINDS = ("prompt_only", "constant", "decaying")

#: How a receiver prompt is turned into token ids (see the receiver section
#: below for the full rationale).
#:
#: - ``chat``: one user message through the tokenizer's own chat template with
#:   ``add_generation_prompt=True`` — exactly one BOS, one user turn, the
#:   end-of-turn marker, and the model-generation prefix.
#: - ``legacy_raw``: the previous behaviour (raw text through the tokenizer,
#:   BOS ensured), retained only so earlier runs can be reproduced.
RECEIVER_FORMATS = ("chat", "legacy_raw")

DEFAULT_RECEIVER_FORMAT = "chat"

#: Substrings a **default** receiver prompt must not contain. Each one primed
#: the instruction-tuned model to echo the prompt's own vocabulary back
#: ("Internal Concept" was the observed output), which is indistinguishable
#: from a steering effect in the recorded metrics. Matched case-insensitively
#: as substrings, so inflections ("labels", "valued") are caught too.
FORBIDDEN_PROMPT_TERMS = (
    "internal",
    "representation",
    "concept",
    "label",
    "value",
)

#: Chat-control token surfaces excluded from a contextual target. Looked up
#: through the tokenizer, so a tokenizer without them is fine.
CHAT_CONTROL_TOKENS = (
    "<bos>",
    "<eos>",
    "<pad>",
    "<start_of_turn>",
    "<end_of_turn>",
)


class GenerativeError(ValueError):
    """Invalid request or unimplemented feature in the generative backend."""


# ------------------------------------------------------------- schedules


@dataclass(frozen=True)
class SteeringSchedule:
    """When and how strongly the steering vector is injected.

    Weights are a pure function of absolute token position, so one-shot
    teacher-forced scoring and step-by-step uncached decoding see identical
    injections at every position (causal attention makes them equivalent).

    Attributes:
        kind: ``prompt_only`` (anchor position only), ``constant`` (anchor
            plus every generated-token position, weight 1), or ``decaying``
            (anchor weight 1; generated position at offset ``o`` from the
            prompt-final position gets ``decay ** o``).
        decay: Decay base for ``decaying`` (ignored otherwise).
    """

    kind: str = "prompt_only"
    decay: float = 0.5

    def __post_init__(self) -> None:
        if self.kind not in SCHEDULE_KINDS:
            raise GenerativeError(
                f"schedule kind {self.kind!r} not in {SCHEDULE_KINDS}"
            )
        if self.kind == "decaying" and not (0.0 < self.decay < 1.0):
            raise GenerativeError(
                f"decay must be in (0, 1) for decaying schedules, got {self.decay}"
            )

    def weights(
        self, *, anchor: int, prompt_len: int, seq_len: int
    ) -> dict[int, float]:
        """Map absolute positions to injection weights for one forward pass.

        ``anchor`` is where the offset-0 injection lands (the prompt-final
        position by default; the wrong-position control moves it). Generated
        tokens occupy positions ``prompt_len .. seq_len - 1`` and are only
        reinjected under ``constant`` / ``decaying``.
        """
        if not (0 <= anchor < seq_len):
            raise GenerativeError(
                f"anchor {anchor} out of range for sequence length {seq_len}"
            )
        if not (0 < prompt_len <= seq_len):
            raise GenerativeError(
                f"prompt_len {prompt_len} invalid for sequence length {seq_len}"
            )
        weights = {anchor: 1.0}
        if self.kind == "prompt_only":
            return weights
        for position in range(prompt_len, seq_len):
            offset = position - (prompt_len - 1)
            weight = 1.0 if self.kind == "constant" else self.decay**offset
            # The anchor's weight stays 1.0 even if a generated position
            # coincides with it (cannot happen with the default anchor).
            weights.setdefault(position, weight)
        return weights

    def describe(self) -> dict:
        return {"kind": self.kind, "decay": self.decay if self.kind == "decaying" else None}


# ------------------------------------------------------------------ hook


@contextmanager
def steering_injection(
    blocks: Sequence[nn.Module],
    layer: int,
    *,
    delta: torch.Tensor,
    schedule: SteeringSchedule,
    prompt_len: int,
    anchor: int | None = None,
    batch_row: int = 0,
    require_frozen: bool = True,
):
    """Context manager: schedule-weighted additive injection of ``delta`` at
    ``blocks[layer]``'s output on every forward pass inside the block.

    For each forward pass with sequence length ``S`` the hook computes
    ``schedule.weights(anchor, prompt_len, S)`` and applies::

        h[batch_row, pos] += weight[pos] * delta        (float32 math)

    cast back to the output dtype. Tuple block outputs have only element 0
    edited. The hook handle is removed on normal exit and on any exception,
    including exceptions the hook itself raises during ``model(...)``.

    Yields a stats dict: ``anchor_activation_norm`` (pre-edit, first
    forward), ``delta_norm``, ``measured_ratio`` (delta norm over anchor
    activation norm), ``n_forward_passes``, ``resolved_anchor``.
    """
    if not isinstance(layer, int) or not (0 <= layer < len(blocks)):
        raise GenerativeError(f"layer {layer} out of range for {len(blocks)} blocks")
    if delta.ndim != 1:
        raise GenerativeError(
            f"delta must be a 1-D [d_model] vector, got shape {tuple(delta.shape)}"
        )
    if not bool(torch.isfinite(delta).all()):
        raise GenerativeError("steering delta contains NaN/Inf")
    if prompt_len < 1:
        raise GenerativeError(f"prompt_len must be >= 1, got {prompt_len}")
    resolved_anchor = prompt_len - 1 if anchor is None else anchor
    if require_frozen:
        _assert_frozen(blocks[layer])

    stats: dict = {
        "delta_norm": float(delta.float().norm()),
        "anchor_activation_norm": None,
        "measured_ratio": None,
        "resolved_anchor": int(resolved_anchor),
        "n_forward_passes": 0,
    }

    def hook(module: nn.Module, inputs, output):
        is_tensor = torch.is_tensor(output)
        hidden = output if is_tensor else output[0]
        if hidden.ndim != 3:
            raise GenerativeError(
                f"expected [batch, seq, d_model] block output, got "
                f"{tuple(hidden.shape)}"
            )
        batch, seq_len, _ = hidden.shape
        if not (0 <= batch_row < batch):
            raise GenerativeError(
                f"batch_row {batch_row} out of range for batch size {batch}"
            )
        weights = schedule.weights(
            anchor=resolved_anchor, prompt_len=prompt_len, seq_len=seq_len
        )
        stats["n_forward_passes"] += 1
        d = delta.to(device=hidden.device, dtype=torch.float32)
        new_hidden = hidden.clone()
        for position, weight in weights.items():
            original = hidden[batch_row, position].float()
            if stats["anchor_activation_norm"] is None and position == resolved_anchor:
                norm = float(original.norm())
                stats["anchor_activation_norm"] = norm
                stats["measured_ratio"] = (
                    stats["delta_norm"] / norm if norm > 0 else None
                )
            edited = original + weight * d
            if not bool(torch.isfinite(edited).all()):
                raise GenerativeError(
                    "steered activation contains NaN/Inf; refusing to continue"
                )
            new_hidden[batch_row, position] = edited.to(hidden.dtype)
        if is_tensor:
            return new_hidden
        return (new_hidden, *tuple(output)[1:])

    handle = blocks[layer].register_forward_hook(hook)
    try:
        yield stats
    finally:
        handle.remove()


@dataclass(frozen=True)
class SteeringSpec:
    """Everything the decode/scoring loops need to steer one condition.

    ``delta=None`` means no hook at all (the ``none`` condition); a zero
    tensor is the hooked parity control.
    """

    layer: int
    delta: torch.Tensor | None
    schedule: SteeringSchedule
    anchor: int | None = None

    def context(self, blocks: Sequence[nn.Module], *, prompt_len: int):
        if self.delta is None:
            return nullcontext({})
        return steering_injection(
            blocks,
            self.layer,
            delta=self.delta,
            schedule=self.schedule,
            prompt_len=prompt_len,
            anchor=self.anchor,
        )


# ------------------------------------------------------- decode & scoring


def _next_token_logprobs(
    model, input_ids: torch.Tensor, *, n_last: int | None = None
) -> torch.Tensor:
    """Next-token log-probabilities through the model's native output pathway
    (softcap included when present).

    Goes through :meth:`~jlens.protocol.LensModel.logits_from_ids` — the
    model's own head — so these log-probabilities are exactly the ones
    ``generate()`` decodes from. Steering hooks live on the residual blocks
    and still fire inside this call.

    Args:
        model: A :class:`~jlens.hf.HFLensModel` (or a compatible mock).
        input_ids: ``[1, seq]`` token ids.
        n_last: Score only the final ``n_last`` positions, returning
            ``[n_last, vocab]``. ``None`` returns ``[seq, vocab]``.

    Always pass ``n_last`` when only the trailing positions are needed.
    ``generate()`` runs the LM head with ``logits_to_keep=1``, i.e. on a
    single-row hidden-state slice; a full-sequence head is a different GEMM
    shape and in reduced precision accumulates differently, so the two paths
    disagree by well over a decode-equivalence tolerance even though both are
    "correct". Matching the shape makes them bit-identical in float32.

    This must **not** be rewritten as ``unembed(forward(ids).last_hidden_state)``:
    HuggingFace text models apply the final norm before returning
    ``last_hidden_state``, so that form applies the final norm twice. A trained
    RMSNorm gain makes the norm non-idempotent, which shifts the logits enough
    to change the argmax — it silently breaks both decoding and target scoring.
    """
    logits = model.logits_from_ids(input_ids, n_last=n_last)[0].float()
    return torch.log_softmax(logits, dim=-1)


@dataclass
class DecodeResult:
    """One greedy decode: generated ids, text, chosen-token log-probs, and
    degeneration bookkeeping."""

    token_ids: list[int]
    text: str
    chosen_logprobs: list[float]
    stop_reason: str
    n_steps: int
    max_bigram_repeats: int

    def to_dict(self) -> dict:
        return {
            "generated_token_ids": self.token_ids,
            "generated_text": self.text,
            "chosen_logprobs": self.chosen_logprobs,
            "stop_reason": self.stop_reason,
            "n_steps": self.n_steps,
            "max_bigram_repeats": self.max_bigram_repeats,
        }


def _max_bigram_repeats(ids: Sequence[int]) -> int:
    """Highest repeat count of any bigram — a cheap degeneration signal."""
    counts: dict[tuple[int, int], int] = {}
    for a, b in zip(ids, ids[1:], strict=False):
        counts[(a, b)] = counts.get((a, b), 0) + 1
    return max(counts.values(), default=0)


def greedy_decode(
    model,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    steering: SteeringSpec | None = None,
    eos_token_ids: Sequence[int] = (),
) -> DecodeResult:
    """Manual **uncached** greedy decoding with optional steering.

    Every step reruns the full forward pass (``use_cache=False`` through
    ``model.forward``), so schedule weights apply identically at every
    position on every step; there is no KV-cache state to reconcile with the
    injection. Deliberately simple and correct; add caching only after the
    zero-steering path is proven equivalent (see tests).

    Args:
        model: A :class:`~jlens.hf.HFLensModel` (or the mock used in tests).
        input_ids: ``[1, prompt_len]`` prompt token ids.
        max_new_tokens: Decode budget (must be >= 1).
        steering: Optional :class:`SteeringSpec`; ``None`` decodes unsteered.
        eos_token_ids: Stop when the argmax token is any of these.
    """
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise GenerativeError(
            f"input_ids must be [1, prompt_len], got {tuple(input_ids.shape)}"
        )
    if max_new_tokens < 1:
        raise GenerativeError(f"max_new_tokens must be >= 1, got {max_new_tokens}")
    prompt_len = input_ids.shape[1]
    eos = {int(t) for t in eos_token_ids}
    blocks = model.layers

    generated: list[int] = []
    logprobs: list[float] = []
    stop_reason = "max_new_tokens"
    context = (
        steering.context(blocks, prompt_len=prompt_len)
        if steering is not None
        else nullcontext({})
    )
    with torch.no_grad(), context:
        ids = input_ids
        for _ in range(max_new_tokens):
            # n_last=1 matches generate()'s logits_to_keep=1 exactly.
            log_p = _next_token_logprobs(model, ids, n_last=1)[-1]
            next_id = int(log_p.argmax())
            generated.append(next_id)
            logprobs.append(float(log_p[next_id]))
            if next_id in eos:
                stop_reason = "eos"
                break
            ids = torch.cat(
                [ids, torch.tensor([[next_id]], dtype=ids.dtype, device=ids.device)],
                dim=1,
            )
    text = model.tokenizer.decode(generated, skip_special_tokens=True)
    return DecodeResult(
        token_ids=generated,
        text=text,
        chosen_logprobs=logprobs,
        stop_reason=stop_reason,
        n_steps=len(generated),
        max_bigram_repeats=_max_bigram_repeats(generated),
    )


def target_logprob(
    model,
    input_ids: torch.Tensor,
    target_ids: Sequence[int],
    *,
    steering: SteeringSpec | None = None,
) -> dict:
    """Teacher-forced log-probability of a multi-token target under steering.

    Runs one forward pass over ``[prompt + target]`` and reads the next-token
    log-probability of each target token at its causal position. Because
    schedule weights depend only on absolute position and attention is
    causal, this equals the sum of per-step log-probabilities an uncached
    steered decode would assign along the forced path.

    Returns a dict with ``per_token_logprobs``, ``total_logprob``,
    ``first_token_rank`` (0 = argmax), and ``first_token_logprob``.
    """
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise GenerativeError(
            f"input_ids must be [1, prompt_len], got {tuple(input_ids.shape)}"
        )
    targets = [int(t) for t in target_ids]
    if not targets:
        raise GenerativeError("target_ids is empty")
    prompt_len = input_ids.shape[1]
    full = torch.cat(
        [
            input_ids,
            torch.tensor([targets], dtype=input_ids.dtype, device=input_ids.device),
        ],
        dim=1,
    )
    context = (
        steering.context(model.layers, prompt_len=prompt_len)
        if steering is not None
        else nullcontext({})
    )
    # Scored positions are the prompt-final one plus one per target token, i.e.
    # absolute rows prompt_len-1 .. prompt_len+len(targets)-2. Those are the
    # trailing len(targets)+1 rows of the [prompt + target] sequence, so ask
    # the head for exactly those: same reason as decoding (match generate()'s
    # sliced-head GEMM instead of running the full-sequence head). Row r of the
    # returned slice is absolute position prompt_len-1+r.
    n_last = len(targets) + 1
    with torch.no_grad(), context:
        log_p = _next_token_logprobs(model, full, n_last=n_last)
    per_token = [float(log_p[j, token]) for j, token in enumerate(targets)]
    first_row = log_p[0]
    first_rank = int((first_row > first_row[targets[0]]).sum())
    return {
        "target_token_ids": targets,
        "per_token_logprobs": per_token,
        "total_logprob": float(sum(per_token)),
        "first_token_rank": first_rank,
        "first_token_logprob": per_token[0],
    }


def first_token_distribution(
    model,
    input_ids: torch.Tensor,
    *,
    steering: SteeringSpec | None = None,
) -> torch.Tensor:
    """Next-token log-probabilities at the prompt-final position (``[vocab]``,
    float32) — the distribution KL divergences are computed against."""
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise GenerativeError(
            f"input_ids must be [1, prompt_len], got {tuple(input_ids.shape)}"
        )
    prompt_len = input_ids.shape[1]
    context = (
        steering.context(model.layers, prompt_len=prompt_len)
        if steering is not None
        else nullcontext({})
    )
    with torch.no_grad(), context:
        # Only the prompt-final row is needed; n_last=1 matches generate().
        return _next_token_logprobs(model, input_ids, n_last=1)[-1]


#: Default number of leading tokens the decode-equivalence comparison treats as
#: decision-relevant. Greedy decoding depends only on the argmax; teacher-forced
#: scoring reads targets that are, in practice, well inside the head of the
#: distribution. Tokens outside the top-k cannot change a decode.
DEFAULT_AGREEMENT_TOP_K = 10


def compare_next_token_distributions(
    log_p_a: torch.Tensor,
    log_p_b: torch.Tensor,
    *,
    top_k: int = DEFAULT_AGREEMENT_TOP_K,
) -> dict:
    """Compare two next-token log-probability distributions over one position.

    Built for asking "are these two decoding paths the same computation?" in
    reduced precision, where a naive max-over-vocabulary difference is the wrong
    question. ``log_softmax`` turns a fixed absolute logit perturbation into a
    log-probability difference of similar size *everywhere*, including tokens
    with probability ~1e-13 that no decode or score can ever be sensitive to. On
    a 262k-token vocabulary the max is therefore reported by the deepest tail,
    not by anything that matters, and it sits at the dtype's quantization floor:
    for bfloat16 a single ULP near ``|logit| = 30`` (Gemma's softcap bound) is
    already 0.0625.

    So this reports both decision-relevant and whole-distribution measures:

    - ``argmax_agrees`` / ``top_k_sets_agree``: does the ranking that decoding
      actually consumes match?
    - ``max_abs_logprob_diff_topk``: worst log-prob gap over the *union* of both
      sides' top-``k`` — the tokens that can affect a decision.
    - ``total_variation``: ``0.5 * sum |p_a - p_b|``, a single bounded
      whole-vocabulary number that cannot be dominated by one tail outlier.
    - ``max_abs_logprob_diff_full_vocab`` / ``mean_abs...``: recorded as
      diagnostics, deliberately *not* the pass/fail criterion.
    """
    if log_p_a.shape != log_p_b.shape or log_p_a.ndim != 1:
        raise GenerativeError(
            f"expected two 1-D distributions of equal shape, got "
            f"{tuple(log_p_a.shape)} vs {tuple(log_p_b.shape)}"
        )
    if top_k < 1:
        raise GenerativeError(f"top_k must be >= 1, got {top_k}")
    a = log_p_a.float()
    b = log_p_b.float()
    k = min(int(top_k), a.shape[0])
    top_a = torch.topk(a, k).indices
    top_b = torch.topk(b, k).indices
    union = torch.unique(torch.cat([top_a, top_b]))
    diff = (a - b).abs()
    return {
        "top_k": k,
        "argmax_a": int(a.argmax()),
        "argmax_b": int(b.argmax()),
        "argmax_agrees": int(a.argmax()) == int(b.argmax()),
        "top_k_sets_agree": set(top_a.tolist()) == set(top_b.tolist()),
        "max_abs_logprob_diff_topk": float(diff[union].max()),
        "total_variation": 0.5 * float((a.exp() - b.exp()).abs().sum()),
        "max_abs_logprob_diff_full_vocab": float(diff.max()),
        "mean_abs_logprob_diff_full_vocab": float(diff.mean()),
    }


def kl_from_baseline(log_p_steered: torch.Tensor, log_p_baseline: torch.Tensor) -> float:
    """``KL(P_steered || P_baseline)`` over the full vocabulary (float32)."""
    if log_p_steered.shape != log_p_baseline.shape:
        raise GenerativeError(
            f"distribution shapes differ: {tuple(log_p_steered.shape)} vs "
            f"{tuple(log_p_baseline.shape)}"
        )
    p = log_p_steered.float().exp()
    return float((p * (log_p_steered.float() - log_p_baseline.float())).sum())


# ------------------------------------------------------ vector conditions


def coefficient_mass_indices(
    coefficients: Sequence[float], threshold: float
) -> list[int]:
    """Indices of the smallest generator subset whose coefficients cover at
    least ``threshold`` of the total positive coefficient mass.

    Generators are taken in descending coefficient order (ties break to the
    earlier index, deterministically). Returned indices are positions into
    ``coefficients``, sorted ascending.
    """
    if not (0.0 < threshold <= 1.0):
        raise GenerativeError(f"threshold must be in (0, 1], got {threshold}")
    positive = [(c, i) for i, c in enumerate(coefficients) if c > 0]
    if not positive:
        raise GenerativeError("no strictly positive coefficients")
    total = sum(c for c, _ in positive)
    picked: list[int] = []
    mass = 0.0
    for c, i in sorted(positive, key=lambda pair: (-pair[0], pair[1])):
        picked.append(i)
        mass += c
        if mass >= threshold * total - 1e-12:
            break
    return sorted(picked)


def weighted_reconstruction(
    atoms: torch.Tensor,
    token_ids: Sequence[int],
    coefficients: Sequence[float],
    *,
    subset: Sequence[int] | None = None,
) -> torch.Tensor:
    """The weighted reconstruction ``q_C = sum_{i in C} a_i v_i`` over raw
    dictionary atoms (float32, on ``atoms.device``).

    ``subset`` selects positions into ``token_ids``/``coefficients``;
    ``None`` uses every generator (the full active cone's reconstruction).
    """
    ids = [int(t) for t in token_ids]
    coeffs = [float(c) for c in coefficients]
    if len(ids) != len(coeffs):
        raise GenerativeError(
            f"{len(ids)} token ids vs {len(coeffs)} coefficients"
        )
    if not ids:
        raise GenerativeError("empty generator set")
    if subset is not None:
        bad = [i for i in subset if not (0 <= int(i) < len(ids))]
        if bad:
            raise GenerativeError(f"subset indices {bad} out of range")
        ids = [ids[int(i)] for i in subset]
        coeffs = [coeffs[int(i)] for i in subset]
        if not ids:
            raise GenerativeError("subset selects no generators")
    if any(c < 0 for c in coeffs):
        raise GenerativeError("coefficients must be nonnegative")
    index = torch.tensor(ids, dtype=torch.long, device=atoms.device)
    weight = torch.tensor(coeffs, dtype=torch.float32, device=atoms.device)
    q = torch.einsum("m,md->d", weight, atoms[index].float())
    _finite_or_raise(q, "weighted reconstruction")
    return q


def shuffled_coordinates(delta: torch.Tensor, *, seed: int) -> torch.Tensor:
    """Deterministic coordinate permutation of ``delta`` (CPU generator, then
    back to the input device). Preserves the norm and value multiset while
    destroying the direction."""
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    perm = torch.randperm(delta.shape[0], generator=generator)
    return delta.detach().float().cpu()[perm].to(delta.device)


def scale_to_ratio(
    delta: torch.Tensor, *, activation_norm: float, ratio: float
) -> tuple[torch.Tensor, dict]:
    """Rescale ``delta`` so ``||delta|| = ratio * activation_norm``.

    Returns ``(scaled, info)`` where ``info`` records the raw norm, the
    requested ratio, and the applied scale factor. A zero ``delta`` cannot
    be rescaled and raises (the zero condition never goes through here).
    """
    if not math.isfinite(ratio) or ratio <= 0:
        raise GenerativeError(f"ratio must be finite and > 0, got {ratio}")
    if not math.isfinite(activation_norm) or activation_norm <= 0:
        raise GenerativeError(
            f"activation_norm must be finite and > 0, got {activation_norm}"
        )
    raw_norm = float(delta.float().norm())
    if raw_norm == 0.0:
        raise GenerativeError("cannot norm-scale a zero vector")
    scale = ratio * activation_norm / raw_norm
    scaled = delta.float() * scale
    return scaled, {
        "raw_delta_norm": raw_norm,
        "requested_ratio": float(ratio),
        "scale_factor": float(scale),
        "scaled_delta_norm": float(scaled.norm()),
    }


def scale_to_norm(
    delta: torch.Tensor, *, target_norm: float
) -> tuple[torch.Tensor, dict]:
    """Rescale ``delta`` so ``||delta|| = target_norm`` exactly.

    Unlike :func:`scale_to_ratio`, ``target_norm`` is an **absolute** norm, not
    a ratio against a receiving activation. This is what the natural-scale
    matched controls (:data:`NATURAL_SCALE_MATCHED_CONDITIONS`) use: their
    target is another vector's *observed* norm (the example's own unscaled
    full J-cone, from the ``natural_scale`` condition), not a fraction of the
    residual at the injection site.

    Returns ``(scaled, info)`` where ``info`` records the raw norm before
    matching, the target, the applied scale factor, and the achieved norm — the
    same shape of provenance :func:`scale_to_ratio` records, so both scaling
    paths are equally auditable. A zero ``delta`` cannot be rescaled and raises.
    """
    if not math.isfinite(target_norm) or target_norm <= 0:
        raise GenerativeError(
            f"target_norm must be finite and > 0, got {target_norm}"
        )
    raw_norm = float(delta.float().norm())
    if raw_norm == 0.0:
        raise GenerativeError("cannot norm-scale a zero vector")
    scale = target_norm / raw_norm
    scaled = delta.float() * scale
    return scaled, {
        "raw_delta_norm": raw_norm,
        "target_norm": float(target_norm),
        "scale_factor": float(scale),
        "scaled_delta_norm": float(scaled.norm()),
    }


@dataclass(frozen=True)
class ConditionVector:
    """A built steering vector plus its provenance (JSON-safe metadata)."""

    condition: str
    delta: torch.Tensor | None
    meta: dict = field(default_factory=dict)


def build_condition_vector(
    condition: str,
    *,
    atoms: torch.Tensor | None = None,
    token_ids: Sequence[int] | None = None,
    coefficients: Sequence[float] | None = None,
    mass_threshold: float | None = None,
    manual_indices: Sequence[int] | None = None,
    unrelated_token_ids: Sequence[int] | None = None,
    unrelated_coefficients: Sequence[float] | None = None,
    raw_activation: torch.Tensor | None = None,
    control_activation: torch.Tensor | None = None,
    seed: int | None = None,
    d_model: int | None = None,
    match_norm: float | None = None,
) -> ConditionVector:
    """Build the unscaled steering vector for one named condition.

    Site-manipulation conditions (``wrong_layer`` / ``wrong_position``) reuse
    the correct full-cone vector — this builder returns that vector; moving
    the injection site is the runner's job. ``none`` returns no delta (no
    hook); ``zero`` returns an explicit zero vector (hooked parity control).
    Any missing ingredient raises loudly.
    """
    if condition not in VECTOR_CONDITIONS:
        raise GenerativeError(
            f"condition {condition!r} not in {VECTOR_CONDITIONS}"
        )

    def need(value, what: str):
        if value is None:
            raise GenerativeError(f"condition {condition!r} requires {what}")
        return value

    if condition == "none":
        return ConditionVector(condition, None, {"hooked": False})

    if condition == "zero":
        dim = need(d_model, "d_model")
        return ConditionVector(
            condition, torch.zeros(int(dim), dtype=torch.float32), {"hooked": True}
        )

    if condition in (
        "full_cone",
        "natural_scale",
        "wrong_layer",
        "wrong_position",
        "sign_reversed",
        "shuffled",
        "natural_sign_reversed",
        "natural_shuffled",
    ):
        q = weighted_reconstruction(
            need(atoms, "atoms"),
            need(token_ids, "token_ids"),
            need(coefficients, "coefficients"),
        )
        meta = {
            "generator_token_ids": [int(t) for t in token_ids],
            "generator_coefficients": [float(c) for c in coefficients],
            "n_generators": len(list(token_ids)),
        }
        if condition in ("sign_reversed", "natural_sign_reversed"):
            return ConditionVector(condition, -q, meta)
        if condition in ("shuffled", "natural_shuffled"):
            s = need(seed, "seed")
            return ConditionVector(
                condition, shuffled_coordinates(q, seed=s), {**meta, "seed": int(s)}
            )
        return ConditionVector(condition, q, meta)

    if condition in ("mass_subcone", "natural_mass_subcone"):
        threshold = need(mass_threshold, "mass_threshold")
        coeffs = list(need(coefficients, "coefficients"))
        subset = coefficient_mass_indices(coeffs, threshold)
        q = weighted_reconstruction(
            need(atoms, "atoms"), need(token_ids, "token_ids"), coeffs, subset=subset
        )
        return ConditionVector(
            condition,
            q,
            {
                "mass_threshold": float(threshold),
                "subset_indices": subset,
                "generator_token_ids": [int(list(token_ids)[i]) for i in subset],
                "generator_coefficients": [coeffs[i] for i in subset],
                "n_generators": len(subset),
            },
        )

    if condition == "manual_subcone":
        subset = [int(i) for i in need(manual_indices, "manual_indices")]
        q = weighted_reconstruction(
            need(atoms, "atoms"),
            need(token_ids, "token_ids"),
            need(coefficients, "coefficients"),
            subset=subset,
        )
        return ConditionVector(
            condition,
            q,
            {
                "subset_indices": subset,
                "generator_token_ids": [int(list(token_ids)[i]) for i in subset],
                "generator_coefficients": [float(list(coefficients)[i]) for i in subset],
                "n_generators": len(subset),
            },
        )

    if condition in ("unrelated_cone", "natural_unrelated_cone"):
        q = weighted_reconstruction(
            need(atoms, "atoms"),
            need(unrelated_token_ids, "unrelated_token_ids"),
            need(unrelated_coefficients, "unrelated_coefficients"),
        )
        return ConditionVector(
            condition,
            q,
            {
                "generator_token_ids": [int(t) for t in unrelated_token_ids],
                "generator_coefficients": [float(c) for c in unrelated_coefficients],
                "n_generators": len(list(unrelated_token_ids)),
            },
        )

    if condition in ("random_matched_norm", "natural_random_matched_norm"):
        dim = need(d_model, "d_model")
        norm = need(match_norm, "match_norm")
        s = need(seed, "seed")
        delta, info = isotropic_random_direction(
            int(dim), match_norm=float(norm), seed=int(s)
        )
        return ConditionVector(
            condition, delta, {"seed": int(s), "matched_norm": info.delta_norm}
        )

    if condition == "raw_activation":
        h = need(raw_activation, "raw_activation").detach().float()
        _finite_or_raise(h, "raw activation")
        return ConditionVector(condition, h, {"source": "raw_activation"})

    if condition == "activation_diff":
        h = need(raw_activation, "raw_activation").detach().float()
        c = need(control_activation, "control_activation").detach().float()
        if h.shape != c.shape:
            raise GenerativeError(
                f"activation shapes differ: {tuple(h.shape)} vs {tuple(c.shape)}"
            )
        diff = h - c
        _finite_or_raise(diff, "activation difference")
        if float(diff.norm()) == 0.0:
            raise GenerativeError("source and control activations are identical")
        return ConditionVector(condition, diff, {"source": "source_minus_control"})

    raise GenerativeError(f"condition {condition!r} is not implemented")


# ----------------------------------------------------------------- records


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Deterministic ``sha256:`` fingerprint of a vector's float32 contents.

    Hashed as little-endian float32 bytes on CPU, so the value depends only on
    the numbers — not on the device, the storage dtype, or whether the tensor
    is a view. Two records carrying the same fingerprint were built from the
    same numbers; two carrying different ones were not. That is the property
    that makes "did this Mandela record actually get Mandela's activation?"
    answerable from ``records.jsonl`` alone, without rerunning anything.
    """
    import hashlib

    flat = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    payload = flat.numpy().astype("<f4", copy=False).tobytes()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def vector_identity(tensor: torch.Tensor | None) -> tuple[float | None, str | None]:
    """``(norm, sha256)`` for a vector, or ``(None, None)`` for ``None``.

    Never fabricates a value: a condition with no vector (``none``) records
    both fields as null rather than a placeholder zero.
    """
    if tensor is None:
        return None, None
    return float(tensor.detach().float().norm()), tensor_sha256(tensor)


#: Which example's J-cone a condition's steering vector is built from.
#:
#: - ``"self"``: the scored example's own active generators.
#: - ``"donor"``: the unrelated-cone donor's generators — the *only* conditions
#:   in which a record's injected vector legitimately comes from a different
#:   example. Any other condition whose recorded ``cone_source_example_id``
#:   differs from its ``example_id`` is a bug, not a control.
#: - ``"none"``: no cone is involved (``none``/``zero``, the isotropic random
#:   directions, and the raw-activation transplants, which use the example's
#:   own activation rather than a reconstruction).
CONE_SOURCE_SELF_CONDITIONS = (
    "full_cone",
    "natural_scale",
    "mass_subcone",
    "natural_mass_subcone",
    "manual_subcone",
    "shuffled",
    "natural_shuffled",
    "sign_reversed",
    "natural_sign_reversed",
    "wrong_layer",
    "wrong_position",
)

CONE_SOURCE_DONOR_CONDITIONS = (
    "unrelated_cone",
    "natural_unrelated_cone",
)


def cone_source_role(condition: str) -> str:
    """``"self"``, ``"donor"``, or ``"none"`` — whose cone ``condition`` uses.

    A pure function of the condition name, so the runner's per-record
    ``cone_source_example_id`` and any downstream contamination check derive
    the expectation from the same place and cannot disagree.
    """
    if condition not in VECTOR_CONDITIONS:
        raise GenerativeError(f"condition {condition!r} not in {VECTOR_CONDITIONS}")
    if condition in CONE_SOURCE_SELF_CONDITIONS:
        return "self"
    if condition in CONE_SOURCE_DONOR_CONDITIONS:
        return "donor"
    return "none"


def expected_cone_source_example_id(
    condition: str, *, example_id: str, donor_example_id: str | None
) -> str | None:
    """The ``cone_source_example_id`` a correct run must record.

    ``None`` when the condition uses no cone. Raises if a donor-sourced
    condition has no donor, which would mean the run built an "unrelated" cone
    out of nothing.
    """
    role = cone_source_role(condition)
    if role == "self":
        return example_id
    if role == "donor":
        if donor_example_id is None:
            raise GenerativeError(
                f"condition {condition!r} needs a donor example but none was "
                f"recorded for {example_id!r}"
            )
        return donor_example_id
    return None


def make_generative_record(
    *,
    run_id: str,
    example_id: str,
    condition: str,
    source_layer: int,
    injection_layer: int,
    source_position: int,
    injection_anchor: int,
    schedule: SteeringSchedule,
    neutral_prompt_id: str,
    strength_ratio: float | None,
    vector_meta: dict,
    hook_stats: dict,
    scoring: dict,
    decode: DecodeResult | None,
    delta_vs_zero: float | None,
    delta_vs_unrelated: float | None,
    kl_divergence: float | None,
    target_phrase: str,
    target_token_strings: Sequence[str] | None = None,
    target_recovered_exact: bool | None,
    target_recovered_substring: bool | None,
    seed: int | None,
    provenance: dict,
    receiver_prompt_id: str | None = None,
    receiver_format: str = DEFAULT_RECEIVER_FORMAT,
    receiver_prompt_len: int | None = None,
    anchor_token_id: int | None = None,
    anchor_token_string: str | None = None,
    source_example_id: str | None = None,
    cone_source_example_id: str | None = None,
    donor_example_id: str | None = None,
    source_prompt: str | None = None,
    source_activation_norm: float | None = None,
    source_activation_sha256: str | None = None,
    cone_norm: float | None = None,
    cone_sha256: str | None = None,
    injected_delta_sha256: str | None = None,
) -> dict:
    """Assemble one JSON-safe result record (schema
    ``jlens.generative.record.v1``). No activation tensors are stored.

    ``receiver_prompt_id`` / ``receiver_format`` / ``receiver_prompt_len`` and
    the anchor token identify *how the prompt was presented to the model*,
    which the first generative runs did not record and which turned out to be
    the thing that invalidated them. ``generated_token_ids`` is always present
    (``None`` when the record was not decoded) so a reader never has to infer
    whether decoding happened from a missing key.

    The identity block written into ``provenance`` answers "which example did
    each artifact in this record actually come from?" without rerunning
    anything:

    - ``source_example_id`` — whose source activation was captured (always the
      scored example; recorded explicitly so a mix-up would be *visible*
      rather than merely absent).
    - ``cone_source_example_id`` — whose J-cone the injected vector was built
      from. Equal to ``example_id`` for every condition except the
      unrelated-cone controls, where it must equal ``donor_example_id``; see
      :func:`expected_cone_source_example_id`.
    - ``donor_example_id`` — the run's unrelated-cone donor for this example,
      recorded on **every** record so a reader always knows which example
      *could* have contributed a cone here, not just the ones where it did.
    - ``source_prompt`` — the prompt the activation was captured from, so the
      record is interpretable without the manifest.
    - ``source_activation_norm`` / ``source_activation_sha256`` — the captured
      activation the pursuit ran on.
    - ``cone_norm`` / ``cone_sha256`` — the **unscaled** built steering vector
      (the weighted J-cone reconstruction for cone-based conditions; the raw
      unit direction for ``random_matched_norm``, which is why the role is
      still reported separately by :func:`cone_source_role`).
    - ``injected_delta_sha256`` — the vector that actually entered the model
      after scaling, so the injected quantity is identified and not just the
      thing it was derived from.

    Together the fingerprints make cross-example artifact reuse checkable by
    equality rather than by inference from decoded text.

    Every field defaults to ``None`` and is written as ``None`` when the caller
    does not supply it. Nothing here is invented or back-filled.
    """
    record = {
        "schema": GENERATIVE_RECORD_SCHEMA,
        "run_id": run_id,
        "example_id": example_id,
        "vector_condition": condition,
        "source_layer": int(source_layer),
        "injection_layer": int(injection_layer),
        "source_position": int(source_position),
        "injection_anchor": int(injection_anchor),
        "steering_schedule": schedule.describe(),
        "neutral_prompt_id": neutral_prompt_id,
        # Same value as neutral_prompt_id, under the name the receiver-format
        # work uses; both are written so old readers keep working.
        "receiver_prompt_id": (
            neutral_prompt_id if receiver_prompt_id is None else receiver_prompt_id
        ),
        "receiver_format": receiver_format,
        "receiver_prompt_len": receiver_prompt_len,
        "anchor_token_id": anchor_token_id,
        "anchor_token_string": anchor_token_string,
        "requested_ratio": strength_ratio,
        "measured_ratio": hook_stats.get("measured_ratio"),
        "delta_norm": hook_stats.get("delta_norm"),
        "receiving_activation_norm": hook_stats.get("anchor_activation_norm"),
        "vector_meta": vector_meta,
        "target_phrase": target_phrase,
        # Segmentation is recorded alongside the ids (which arrive via
        # `scoring`) so a reader can see how the phrase was split without
        # needing the tokenizer.
        "target_token_strings": (
            None if target_token_strings is None else list(target_token_strings)
        ),
        "delta_logprob_vs_zero": delta_vs_zero,
        "delta_logprob_vs_unrelated": delta_vs_unrelated,
        "kl_divergence_from_baseline": kl_divergence,
        "target_recovered_exact": target_recovered_exact,
        "target_recovered_substring": target_recovered_substring,
        # Always present; DecodeResult.to_dict() overwrites it when decoded.
        "generated_token_ids": None,
        "random_seed": seed,
        "provenance": {
            **dict(provenance),
            # Identity block: which example every artifact in this record came
            # from. Written last so it cannot be shadowed by a caller-supplied
            # provenance key of the same name.
            "example_id": example_id,
            "source_example_id": source_example_id,
            "cone_source_example_id": cone_source_example_id,
            "donor_example_id": donor_example_id,
            "source_prompt": source_prompt,
            "target_phrase": target_phrase,
            "source_layer": int(source_layer),
            "source_activation_norm": source_activation_norm,
            "source_activation_sha256": source_activation_sha256,
            "cone_norm": cone_norm,
            "cone_sha256": cone_sha256,
            "injected_delta_sha256": injected_delta_sha256,
        },
    }
    record.update(scoring)
    if decode is not None:
        record.update(decode.to_dict())
    return record


# ---------------------------------------------------------------- benchmark


REQUIRED_EXAMPLE_KEYS = (
    "example_id",
    "category",
    "source_prompt",
    "target_phrase",
    "extraction_position",
)

BENCHMARK_CATEGORIES = (
    "split_word",
    "compound_word",
    "named_entity",
    "noun_phrase",
)


def load_benchmark(path: str) -> dict:
    """Load and validate the generative benchmark manifest.

    Layout: ``{"version": int, "dev": [example...], "heldout": [example...]}``
    where each example has :data:`REQUIRED_EXAMPLE_KEYS`, optionally
    ``control_prompt``, ``target_token_ids`` (resolved at run time against
    the pinned tokenizer when null), ``source_layers``, and
    ``manual_subcone`` (a per-layer map of generator indices). Example IDs
    must be unique across both splits.
    """
    import json

    with open(path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    for split in ("dev", "heldout"):
        if split not in manifest or not isinstance(manifest[split], list):
            raise GenerativeError(f"{path}: missing example list {split!r}")
    seen: set[str] = set()
    for split in ("dev", "heldout"):
        for example in manifest[split]:
            missing = [k for k in REQUIRED_EXAMPLE_KEYS if k not in example]
            if missing:
                raise GenerativeError(
                    f"{path}: example {example.get('example_id')!r} missing "
                    f"keys {missing}"
                )
            if example["category"] not in BENCHMARK_CATEGORIES:
                raise GenerativeError(
                    f"{path}: example {example['example_id']!r} category "
                    f"{example['category']!r} not in {BENCHMARK_CATEGORIES}"
                )
            if example["example_id"] in seen:
                raise GenerativeError(
                    f"{path}: duplicate example_id {example['example_id']!r}"
                )
            seen.add(example["example_id"])
            if not str(example["target_phrase"]).strip():
                raise GenerativeError(
                    f"{path}: example {example['example_id']!r} has an empty "
                    f"target phrase"
                )
    return manifest


#: Inclusive bounds on the token count of a main-benchmark target phrase.
#:
#: The lower bound is the whole point of the experiment: a single-token target
#: cannot test multi-token target scoring, and — because every steering schedule
#: injects identically at the prompt-final position and differs only at
#: *generated* positions — prompt_only / constant / decaying necessarily produce
#: identical target log-probabilities for it. Such an example silently
#: contributes three duplicate rows and no schedule signal. (This is exactly
#: what ``dev-split-photosynthesis`` did: it tokenized to the single id 93036.)
#:
#: The upper bound keeps teacher-forced totals comparable across examples; a
#: very long target's summed log-probability is dominated by its length.
MIN_TARGET_TOKENS = 2
MAX_TARGET_TOKENS = 6


def token_strings(tokenizer, token_ids: Sequence[int]) -> list[str]:
    """Per-token surface strings for ``token_ids``, decoded one id at a time.

    Recorded next to the ids so a reader can see *how* a phrase was segmented
    without needing the tokenizer. Decoded individually on purpose: joint
    decoding hides the boundaries, which are the thing under inspection.
    """
    return [tokenizer.decode([int(t)]) for t in token_ids]


def validate_target_tokens(
    examples: Sequence[dict],
    tokenizer,
    *,
    resolve,
    min_tokens: int = MIN_TARGET_TOKENS,
    max_tokens: int = MAX_TARGET_TOKENS,
    use_manifest_ids: bool = True,
) -> dict[str, dict]:
    """Resolve every example's target against the **pinned** tokenizer and
    require ``min_tokens <= n <= max_tokens``.

    Call this after the model/tokenizer is loaded — the token count is a
    property of the actual checkpoint's vocabulary, so it cannot be decided
    when the manifest is written.

    Args:
        examples: Benchmark examples (``example_id``, ``target_phrase``, and an
            optional pre-resolved ``target_token_ids``).
        tokenizer: The pinned tokenizer.
        resolve: ``(tokenizer, phrase) -> list[int]`` continuation tokenizer.
        min_tokens: Inclusive lower bound (see :data:`MIN_TARGET_TOKENS`).
        max_tokens: Inclusive upper bound.
        use_manifest_ids: Honour a manifest's pre-resolved
            ``target_token_ids``. Must be ``False`` whenever ``resolve``
            derives ids **contextually** (chat mode): a standalone id list
            written into the manifest is not the segmentation the model sees
            after the formatted prompt, so silently preferring it would score
            the wrong tokens.

    Returns:
        ``{example_id: {"target_token_ids": [...], "target_token_strings":
        [...], "n_target_tokens": int}}``.

    Raises:
        GenerativeError: If any example violates the bound. The message lists
            every offender with its id, phrase, token ids, and token strings, so
            one run surfaces all of them rather than one per attempt.
    """
    if not 1 <= min_tokens <= max_tokens:
        raise GenerativeError(
            f"invalid bounds: min_tokens={min_tokens}, max_tokens={max_tokens}"
        )
    resolved: dict[str, dict] = {}
    problems: list[str] = []
    for example in examples:
        example_id = example["example_id"]
        phrase = example["target_phrase"]
        manifest_ids = example.get("target_token_ids") if use_manifest_ids else None
        ids = [int(t) for t in (manifest_ids or resolve(tokenizer, phrase))]
        strings = token_strings(tokenizer, ids)
        resolved[example_id] = {
            "target_token_ids": ids,
            "target_token_strings": strings,
            "n_target_tokens": len(ids),
        }
        if not (min_tokens <= len(ids) <= max_tokens):
            problems.append(
                f"  {example_id}: target_phrase={phrase!r} tokenizes to "
                f"{len(ids)} token(s) (need {min_tokens}-{max_tokens}); "
                f"target_token_ids={ids}; token_strings={strings}"
            )
    if problems:
        raise GenerativeError(
            f"{len(problems)} benchmark target(s) violate the "
            f"{min_tokens}-{max_tokens} token requirement under the pinned "
            f"tokenizer:\n" + "\n".join(problems) + "\n"
            "A single-token target cannot test multi-token scoring, and makes "
            "every steering schedule produce identical target "
            "log-probabilities. Revise the benchmark concept."
        )
    return resolved


def select_split_examples(
    manifest: dict, split: str, *, limit: int | None = None
) -> tuple[list[dict], set[str]]:
    """Examples to run for ``split``, plus the ids present only as
    unrelated-cone donors.

    The unrelated-cone control borrows another example's active cone, so at
    least two examples must be present even when ``limit`` asks for one. The
    extra example is always drawn **from the same split**: a development run
    that borrowed a held-out example's cone would leak held-out information
    into calibration, which is the whole thing the split exists to prevent.

    Returns:
        ``(examples, donor_only_ids)``. ``examples`` is the run list in order;
        ids in ``donor_only_ids`` are scored for nobody — they exist purely to
        donate a cone and must be skipped when emitting records.

    Raises:
        GenerativeError: If ``split`` is unknown, or has fewer than two
            examples (no same-split donor exists, and crossing splits is not an
            option).
    """
    if split not in manifest or not isinstance(manifest[split], list):
        raise GenerativeError(f"manifest has no example list for split {split!r}")
    available = manifest[split]
    if len(available) < 2:
        raise GenerativeError(
            f"split {split!r} has {len(available)} example(s); the "
            f"unrelated-cone control needs a second example from the *same* "
            f"split (borrowing across splits would leak the other split)"
        )
    selected = available if limit is None else available[: max(int(limit), 1)]
    donor_only: set[str] = set()
    if len(selected) < 2:
        donor = available[len(selected)]
        selected = [*selected, donor]
        donor_only.add(donor["example_id"])
    return list(selected), donor_only


# ---------------------------------------------------------------- receiver
#
# The receiver is the neutral prompt the steering vector is injected into, plus
# the exact way that prompt is turned into token ids. Both halves were wrong in
# the first generative runs and the two failures compounded:
#
# 1. **No chat formatting.** The pinned checkpoint is ``google/gemma-4-E4B-it``,
#    an *instruction-tuned* model. Prompts were tokenized with the raw
#    ``model.encode()`` path, so the model never saw ``<start_of_turn>user ...
#    <end_of_turn>`` or the ``<start_of_turn>model`` generation prefix it was
#    tuned to answer after. Its continuation of a bare instruction fragment is
#    not the same distribution as its answer to a user turn.
# 2. **Lexically priming prompts.** The original prompts contained the literal
#    words "internal concept", "internal representation", and "Label:", so the
#    highest-probability continuation of the *prompt itself* was a paraphrase of
#    the prompt ("Internal Concept"). Any such output says nothing about the
#    injected vector — it is a property of the wording.
#
# So the receiver now has an explicit **format** (:data:`RECEIVER_FORMATS`) and
# a set of default prompts that are target-independent *and* free of the
# priming vocabulary (:data:`FORBIDDEN_PROMPT_TERMS`). The old prompts are kept
# only as clearly named legacy diagnostics, and the raw encoding only as
# ``legacy_raw``, so previous runs stay reproducible without being the default.

#: Default receiver prompts: target-independent, free of
#: :data:`FORBIDDEN_PROMPT_TERMS`, and phrased as an instruction a chat model
#: answers rather than a fragment it continues. Checked at import time.
NEUTRAL_PROMPTS: dict[str, str] = {
    "noun-phrase-only": (
        "Reply with only a specific one- to four-word noun phrase."
    ),
    "what-is-described": (
        "What is being described? Reply with only its specific name."
    ),
    "name-the-entity": (
        "Answer with only the specific name of the entity, object, event, or "
        "phenomenon."
    ),
}

#: The original prompts, retained **only** as diagnostics for reproducing the
#: confounded runs. Every one of them contains at least one
#: :data:`FORBIDDEN_PROMPT_TERMS` entry — that is the point: they are what the
#: "Internal Concept" outputs were a response to. Never use them as a default,
#: and never in smoke mode.
LEGACY_DIAGNOSTIC_PROMPTS: dict[str, str] = {
    "legacy-label-colon": (
        "Return only the shortest noun or noun phrase naming the represented "
        "internal concept.\nLabel:"
    ),
    "legacy-answer-four-words": (
        "Name the represented internal concept using no more than four "
        "words.\nAnswer:"
    ),
    "legacy-shortest-label-is": (
        "The shortest label for the internal representation is:"
    ),
}


def prompt_safety_violations(text: str) -> list[str]:
    """Which :data:`FORBIDDEN_PROMPT_TERMS` appear in ``text`` (lowercased
    substring match), in declaration order. Empty means clean."""
    lowered = str(text).lower()
    return [term for term in FORBIDDEN_PROMPT_TERMS if term in lowered]


def assert_clean_prompt(text: str, *, prompt_id: str = "<prompt>") -> None:
    """Raise unless ``text`` is free of the priming vocabulary.

    A prompt containing "concept"/"label"/... makes its own restatement the
    model's most likely continuation, so a decode that echoes it is evidence
    about the wording, not about the injected vector.
    """
    if not str(text).strip():
        raise GenerativeError(f"receiver prompt {prompt_id!r} is empty")
    violations = prompt_safety_violations(text)
    if violations:
        raise GenerativeError(
            f"receiver prompt {prompt_id!r} contains priming term(s) "
            f"{violations}: {text!r}. Default prompts must be "
            f"target-independent and free of {list(FORBIDDEN_PROMPT_TERMS)}; "
            f"use LEGACY_DIAGNOSTIC_PROMPTS explicitly if reproducing a "
            f"confounded run."
        )


# Import-time guard: a default prompt can never silently regain priming
# vocabulary. Legacy diagnostics are deliberately not checked.
for _prompt_id, _prompt_text in NEUTRAL_PROMPTS.items():
    assert_clean_prompt(_prompt_text, prompt_id=_prompt_id)
del _prompt_id, _prompt_text


def is_clean_prompt_id(prompt_id: str) -> bool:
    """Whether ``prompt_id`` names a default (non-legacy, clean) prompt."""
    return prompt_id in NEUTRAL_PROMPTS


def resolve_neutral_prompt(prompt_id: str) -> str:
    """Prompt text for ``prompt_id``, from the defaults or the legacy
    diagnostics. Unknown ids raise rather than silently falling back."""
    if prompt_id in NEUTRAL_PROMPTS:
        return NEUTRAL_PROMPTS[prompt_id]
    if prompt_id in LEGACY_DIAGNOSTIC_PROMPTS:
        return LEGACY_DIAGNOSTIC_PROMPTS[prompt_id]
    raise GenerativeError(
        f"unknown receiver prompt id {prompt_id!r}; known ids are "
        f"{sorted(NEUTRAL_PROMPTS)} (default) and "
        f"{sorted(LEGACY_DIAGNOSTIC_PROMPTS)} (legacy diagnostics)"
    )


def receiver_format_from_config(config: dict) -> str:
    """The validated ``receiver.format`` of a generative config, defaulting to
    :data:`DEFAULT_RECEIVER_FORMAT`."""
    section = config.get("receiver") or {}
    fmt = section.get("format", DEFAULT_RECEIVER_FORMAT)
    if fmt not in RECEIVER_FORMATS:
        raise GenerativeError(
            f"receiver.format {fmt!r} not in {RECEIVER_FORMATS}"
        )
    return str(fmt)


def _tokenize_ids(
    tokenizer, text: str, *, add_special_tokens: bool, max_length: int
) -> list[int]:
    """Token ids of ``text`` as a plain list (no tensors, no device)."""
    encoded = tokenizer(
        text,
        return_tensors=None,
        truncation=True,
        max_length=int(max_length),
        add_special_tokens=add_special_tokens,
    )
    ids = encoded.input_ids if hasattr(encoded, "input_ids") else encoded["input_ids"]
    if ids and isinstance(ids[0], (list, tuple)):  # some tokenizers batch
        ids = ids[0]
    return [int(t) for t in ids]


def _ensure_single_bos(tokenizer, ids: Sequence[int], *, max_length: int) -> list[int]:
    """Exactly one leading BOS, matching :meth:`Gemma4LensModel.encode`'s
    deterministic prepend. A tokenizer without a BOS is left alone."""
    out = [int(t) for t in ids]
    bos = getattr(tokenizer, "bos_token_id", None)
    if bos is None:
        return out
    if not out or out[0] != int(bos):
        out = [int(bos), *out]
    return out[: int(max_length)]


def chat_control_token_ids(tokenizer) -> set[int]:
    """Token ids that are chat/control machinery rather than answer content.

    Union of the tokenizer's declared special ids and any of
    :data:`CHAT_CONTROL_TOKENS` it knows — notably the assistant end-of-turn
    marker, which a contextual target must never include.
    """
    control: set[int] = set()
    for attribute in ("bos_token_id", "eos_token_id", "pad_token_id"):
        value = getattr(tokenizer, attribute, None)
        if isinstance(value, int):
            control.add(int(value))
    special = getattr(tokenizer, "all_special_ids", None)
    if isinstance(special, (list, tuple, set)):
        control.update(int(t) for t in special if isinstance(t, int))
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    unk = getattr(tokenizer, "unk_token_id", None)
    if callable(convert):
        for token in CHAT_CONTROL_TOKENS:
            try:
                value = convert(token)
            except Exception:  # noqa: BLE001 - tokenizer-specific failures
                continue
            if isinstance(value, int) and value >= 0 and value != unk:
                control.add(int(value))
    return control


def render_receiver_prompt(
    tokenizer, text: str, *, receiver_format: str = DEFAULT_RECEIVER_FORMAT
) -> str:
    """The exact string that gets tokenized for one receiver prompt.

    ``chat`` renders a single user message through the tokenizer's own chat
    template with ``add_generation_prompt=True``; ``legacy_raw`` returns the
    text unchanged. Source prompts are **not** affected — only the receiver.
    """
    if receiver_format not in RECEIVER_FORMATS:
        raise GenerativeError(
            f"receiver format {receiver_format!r} not in {RECEIVER_FORMATS}"
        )
    if receiver_format == "legacy_raw":
        return text
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_chat_template):
        raise GenerativeError(
            "receiver format 'chat' requires a tokenizer with "
            "apply_chat_template; pass receiver.format: legacy_raw to opt out"
        )
    return apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
    )


def _check_chat_structure(
    tokenizer, text: str, rendered: str, ids: Sequence[int]
) -> dict:
    """Verify the chat rendering has exactly the shape steering assumes.

    Required: exactly one BOS (at position 0), the user content present exactly
    once, and a non-empty model-generation prefix appended after the
    end-of-turn marker. When the tokenizer's template uses Gemma's turn markers
    the counts are checked exactly (two ``<start_of_turn>``, one
    ``<end_of_turn>``); a template without them is still checked structurally.
    """
    problems: list[str] = []
    bos = getattr(tokenizer, "bos_token_id", None)
    n_bos = None
    if bos is not None:
        n_bos = sum(1 for t in ids if int(t) == int(bos))
        if n_bos != 1:
            problems.append(f"expected exactly 1 BOS token, found {n_bos}")
        elif int(ids[0]) != int(bos):
            problems.append("BOS is present but not at position 0")
    base = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=False,
    )
    generation_prefix = ""
    if not rendered.startswith(base):
        problems.append(
            "add_generation_prompt=True rendering is not an extension of the "
            "add_generation_prompt=False rendering"
        )
    else:
        generation_prefix = rendered[len(base) :]
        if not generation_prefix.strip():
            problems.append(
                "chat template appended no model-generation prefix "
                "(add_generation_prompt had no effect)"
            )
    n_content = rendered.count(text)
    if n_content != 1:
        problems.append(
            f"user content appears {n_content} time(s) in the rendered prompt, "
            f"expected exactly 1"
        )
    markers_checked = any(marker in rendered for marker in ("<start_of_turn>", "<end_of_turn>"))
    n_start = rendered.count("<start_of_turn>")
    n_end = rendered.count("<end_of_turn>")
    if markers_checked:
        if n_start != 2:
            problems.append(
                f"expected 2 <start_of_turn> markers (user turn + model "
                f"generation prefix), found {n_start}"
            )
        if n_end != 1:
            problems.append(
                f"expected 1 <end_of_turn> marker (end of the user turn), "
                f"found {n_end}"
            )
    if problems:
        raise GenerativeError(
            "chat receiver rendering is malformed:\n  "
            + "\n  ".join(problems)
            + f"\nrendered prompt: {rendered!r}"
        )
    return {
        "n_bos_tokens": n_bos,
        "generation_prefix": generation_prefix,
        "n_start_of_turn_markers": n_start,
        "n_end_of_turn_markers": n_end,
        "turn_markers_checked": markers_checked,
        "n_user_content_occurrences": n_content,
    }


def resolve_steering_anchor(
    token_ids: Sequence[int], tokenizer=None, *, anchor: int | None = None
) -> dict:
    """Resolve and describe the injection anchor for a formatted prompt.

    The default anchor is the **final prompt position** — the one whose
    next-token distribution is the first answer token — computed from the
    *formatted* prompt length, so chat control tokens are counted. Site
    controls pass ``anchor`` explicitly (``wrong_position``); negative indices
    count from the end.

    Returns ``prompt_len``, ``anchor_index``, ``anchor_token_id``,
    ``anchor_token_string``, and whether the anchor is the final prompt
    position.
    """
    ids = [int(t) for t in token_ids]
    prompt_len = len(ids)
    if prompt_len < 1:
        raise GenerativeError("cannot anchor steering in an empty prompt")
    index = prompt_len - 1 if anchor is None else int(anchor)
    if index < 0:
        index = prompt_len + index
    if not (0 <= index < prompt_len):
        raise GenerativeError(
            f"steering anchor {anchor} out of range for formatted prompt "
            f"length {prompt_len}"
        )
    token_id = ids[index]
    token_string = None
    if tokenizer is not None:
        token_string = tokenizer.decode([token_id])
    return {
        "prompt_len": prompt_len,
        "anchor_index": index,
        "anchor_token_id": token_id,
        "anchor_token_string": token_string,
        "is_final_prompt_position": index == prompt_len - 1,
        "predicts_first_answer_token": index == prompt_len - 1,
    }


@dataclass(frozen=True)
class ReceiverPrompt:
    """One receiver prompt, encoded and fully described.

    Everything the run needs to defend the prompting decision later: the raw
    prompt, the rendering that was actually tokenized, its ids and per-token
    strings, the decode of those ids, and the resolved steering anchor.
    """

    prompt_id: str
    receiver_format: str
    raw_prompt: str
    rendered_prompt: str
    token_ids: tuple[int, ...]
    token_strings: tuple[str, ...]
    decoded_prompt: str
    steering_anchor: dict
    structure: dict

    @property
    def prompt_len(self) -> int:
        return len(self.token_ids)

    def input_ids(self, *, device=None, dtype=torch.long) -> torch.Tensor:
        """``[1, prompt_len]`` ids for the decode / scoring entry points."""
        return torch.tensor([list(self.token_ids)], dtype=dtype, device=device)

    def to_debug_dict(self) -> dict:
        return {
            "prompt_id": self.prompt_id,
            "receiver_format": self.receiver_format,
            "raw_prompt": self.raw_prompt,
            "rendered_prompt": self.rendered_prompt,
            "prompt_token_ids": list(self.token_ids),
            "prompt_tokens": list(self.token_strings),
            "decoded_prompt": self.decoded_prompt,
            "prompt_len": self.prompt_len,
            "steering_anchor": dict(self.steering_anchor),
            "structure": dict(self.structure),
            "prompt_safety_violations": prompt_safety_violations(self.raw_prompt),
            "is_default_prompt": is_clean_prompt_id(self.prompt_id),
        }


def encode_receiver_prompt(
    tokenizer,
    text: str,
    *,
    prompt_id: str = "<prompt>",
    receiver_format: str = DEFAULT_RECEIVER_FORMAT,
    max_length: int = 512,
    anchor: int | None = None,
) -> ReceiverPrompt:
    """**The** receiver encoding path. Every receiver prompt goes through here.

    Centralized on purpose: the confounded runs encoded receiver prompts in one
    place (the runner's sweep), the parity gate in another, and the target ids
    in a third, so a formatting change could reach one and miss the others.
    """
    if receiver_format not in RECEIVER_FORMATS:
        raise GenerativeError(
            f"receiver format {receiver_format!r} not in {RECEIVER_FORMATS}"
        )
    rendered = render_receiver_prompt(
        tokenizer, text, receiver_format=receiver_format
    )
    # chat: the template already emits BOS as literal text, so specials must not
    # be added again. legacy_raw: exactly the old model.encode() behaviour.
    ids = _tokenize_ids(
        tokenizer,
        rendered,
        add_special_tokens=receiver_format == "legacy_raw",
        max_length=max_length,
    )
    ids = _ensure_single_bos(tokenizer, ids, max_length=max_length)
    if not ids:
        raise GenerativeError(f"receiver prompt {prompt_id!r} tokenized to nothing")
    if receiver_format == "chat":
        structure = _check_chat_structure(tokenizer, text, rendered, ids)
    else:
        structure = {
            "n_bos_tokens": sum(
                1
                for t in ids
                if getattr(tokenizer, "bos_token_id", None) is not None
                and int(t) == int(tokenizer.bos_token_id)
            ),
            "generation_prefix": None,
            "turn_markers_checked": False,
        }
    return ReceiverPrompt(
        prompt_id=prompt_id,
        receiver_format=receiver_format,
        raw_prompt=text,
        rendered_prompt=rendered,
        token_ids=tuple(ids),
        token_strings=tuple(token_strings(tokenizer, ids)),
        decoded_prompt=tokenizer.decode(ids),
        steering_anchor=resolve_steering_anchor(ids, tokenizer, anchor=anchor),
        structure=structure,
    )


def contextual_target_token_ids(
    tokenizer,
    receiver: ReceiverPrompt,
    target_phrase: str,
    *,
    max_length: int = 512,
) -> dict:
    """Target token ids as the assistant continuation of the **formatted**
    prompt.

    In ``chat`` mode the target is whatever ``rendered_prompt + target_phrase``
    tokenizes to beyond the prompt's own ids — the segmentation the model
    actually sees after ``<start_of_turn>model``, which is not in general the
    same as tokenizing the phrase on its own. BOS, chat-control tokens, and the
    assistant end-of-turn marker are removed
    (:func:`chat_control_token_ids`), so scoring covers answer content only.

    In ``legacy_raw`` mode the old behaviour is preserved exactly: the phrase
    is tokenized standalone with no special tokens and nothing is filtered.

    Raises:
        GenerativeError: If appending the phrase re-tokenizes the prompt itself
            (the prompt ids are then not a prefix of the joint ids, so no
            alignment between the two is safe), or if nothing survives
            filtering.
    """
    phrase = str(target_phrase)
    if not phrase:
        raise GenerativeError("target_phrase is empty")
    if receiver.receiver_format == "legacy_raw":
        ids = _tokenize_ids(
            tokenizer, phrase, add_special_tokens=False, max_length=max_length
        )
        if not ids:
            raise GenerativeError(f"target phrase {phrase!r} tokenized to nothing")
        return {
            "target_token_ids": ids,
            "target_token_strings": token_strings(tokenizer, ids),
            "n_target_tokens": len(ids),
            "decoded_target": tokenizer.decode(ids),
            "raw_continuation_token_ids": list(ids),
            "excluded_token_ids": [],
            "derivation": "legacy_raw_standalone_phrase",
        }

    prompt_ids = list(receiver.token_ids)
    full_ids = _ensure_single_bos(
        tokenizer,
        _tokenize_ids(
            tokenizer,
            receiver.rendered_prompt + phrase,
            add_special_tokens=False,
            max_length=max_length,
        ),
        max_length=max_length,
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        shared = 0
        for a, b in zip(full_ids, prompt_ids, strict=False):
            if a != b:
                break
            shared += 1
        raise GenerativeError(
            f"appending target phrase {phrase!r} re-tokenized the formatted "
            f"prompt (prompt ids are not a prefix of the joint ids; they agree "
            f"on {shared}/{len(prompt_ids)} tokens), so the continuation cannot "
            f"be sliced off safely. Adjust the target phrase's leading "
            f"whitespace, or run with receiver.format: legacy_raw."
        )
    continuation = full_ids[len(prompt_ids) :]
    if not continuation:
        raise GenerativeError(
            f"target phrase {phrase!r} added no tokens after the formatted "
            f"prompt"
        )
    control = chat_control_token_ids(tokenizer)
    kept = [t for t in continuation if t not in control]
    excluded = [t for t in continuation if t in control]
    if not kept:
        raise GenerativeError(
            f"target phrase {phrase!r} produced only control tokens "
            f"{excluded} after the formatted prompt"
        )
    return {
        "target_token_ids": kept,
        "target_token_strings": token_strings(tokenizer, kept),
        "n_target_tokens": len(kept),
        "decoded_target": tokenizer.decode(kept),
        "raw_continuation_token_ids": list(continuation),
        "excluded_token_ids": excluded,
        "derivation": "chat_assistant_continuation",
    }


def contextual_target_resolver(
    receiver: ReceiverPrompt, *, max_length: int = 512
):
    """A ``(tokenizer, phrase) -> list[int]`` resolver bound to one formatted
    receiver prompt, for :func:`validate_target_tokens`."""

    def resolve(tokenizer, phrase: str) -> list[int]:
        return contextual_target_token_ids(
            tokenizer, receiver, phrase, max_length=max_length
        )["target_token_ids"]

    return resolve


def receiver_prompt_debug(
    receiver: ReceiverPrompt, targets: dict[str, dict] | None = None
) -> dict:
    """Run-level debug entry for one receiver prompt: the rendering, its ids
    and tokens, the anchor, and every example's contextual target."""
    entry: dict[str, Any] = receiver.to_debug_dict()
    entry["targets"] = {
        example_id: dict(info) for example_id, info in (targets or {}).items()
    }
    return entry


# ------------------------------------------------------------- aggregation


def _mean(values: Sequence[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


def summarize_by_condition(records: Sequence[dict]) -> list[dict]:
    """Per-condition aggregate of scored records: mean/min/max target
    log-probability improvement over zero, specificity over the unrelated
    cone, recovery rates, and mean KL. Purely descriptive.

    Every row is tagged with :func:`condition_scaling_mode`, so a caller can
    filter to (or exclude) natural-scale-matched controls without
    reconstructing that grouping from the condition name itself. Rows are
    never merged across conditions — ``natural_unrelated_cone`` and
    ``unrelated_cone`` are always separate rows, even though one is a rescaled
    copy of the other's direction — so ratio-scaled and natural-scale-matched
    results can never be silently averaged together.
    """
    groups: dict[str, list[dict]] = {}
    for record in records:
        groups.setdefault(record["vector_condition"], []).append(record)
    out = []
    for condition in sorted(groups):
        rows = groups[condition]
        deltas = [
            r["delta_logprob_vs_zero"]
            for r in rows
            if r.get("delta_logprob_vs_zero") is not None
        ]
        decoded = [r for r in rows if r.get("target_recovered_exact") is not None]
        exact = [r for r in decoded if r.get("target_recovered_exact")]
        substring = [r for r in decoded if r.get("target_recovered_substring")]
        out.append(
            {
                "vector_condition": condition,
                "scaling_mode": condition_scaling_mode(condition),
                "n_records": len(rows),
                "mean_delta_vs_zero": _mean(deltas),
                "min_delta_vs_zero": min(deltas) if deltas else None,
                "max_delta_vs_zero": max(deltas) if deltas else None,
                "mean_specificity_vs_unrelated": _mean(
                    [r.get("delta_logprob_vs_unrelated") for r in rows]
                ),
                "mean_kl_from_baseline": _mean(
                    [r.get("kl_divergence_from_baseline") for r in rows]
                ),
                "n_decoded": len(decoded),
                "exact_recovery_rate": len(exact) / len(decoded) if decoded else None,
                "substring_recovery_rate": (
                    len(substring) / len(decoded) if decoded else None
                ),
            }
        )
    return out


def _matches(
    record: dict, *, source_layer: int, ratio: float, schedule_kind: str
) -> bool:
    return (
        record.get("source_layer") == source_layer
        and record.get("requested_ratio") == ratio
        and record.get("steering_schedule", {}).get("kind") == schedule_kind
    )


def choose_calibration(records: Sequence[dict], *, condition: str = "full_cone") -> dict:
    """Pick the (source_layer, ratio, schedule kind) with the best mean
    target-log-probability improvement over zero for ``condition`` — the dev
    split's calibration step. Fails loudly with no scored candidates."""
    candidates: dict[tuple, list[float]] = {}
    for record in records:
        if record.get("vector_condition") != condition:
            continue
        delta = record.get("delta_logprob_vs_zero")
        if delta is None:
            continue
        key = (
            record["source_layer"],
            record["requested_ratio"],
            record["steering_schedule"]["kind"],
        )
        candidates.setdefault(key, []).append(delta)
    if not candidates:
        raise GenerativeError(f"no scored {condition!r} records to calibrate on")
    scored = {key: _mean(values) for key, values in candidates.items()}
    best = max(sorted(scored), key=lambda key: scored[key])
    return {
        "condition": condition,
        "source_layer": best[0],
        "ratio": best[1],
        "schedule_kind": best[2],
        "mean_delta_vs_zero": scored[best],
        "n_records": len(candidates[best]),
    }


#: Controls the correct vector must beat, per example, for a "go" verdict.
GONOGO_CONTROLS = (
    "random_matched_norm",
    "shuffled",
    "sign_reversed",
    "unrelated_cone",
)


def per_example_verdicts(
    records: Sequence[dict],
    *,
    source_layer: int,
    ratio: float,
    schedule_kind: str,
    correct_condition: str = "full_cone",
) -> list[dict]:
    """Per-example go/no-go criteria at one calibrated operating point.

    For each example: does ``correct_condition`` beat zero and every control
    in :data:`GONOGO_CONTROLS` on target log-probability, on more than one
    neutral prompt; was the target recovered (exact or substring) by
    decoding; and how does it compare to ``raw_activation`` transplantation?
    """
    by_example: dict[str, list[dict]] = {}
    for record in records:
        at_point = _matches(
            record, source_layer=source_layer, ratio=ratio, schedule_kind=schedule_kind
        )
        if at_point or record.get("vector_condition") in ("none", "zero"):
            by_example.setdefault(record["example_id"], []).append(record)
    verdicts = []
    for example_id in sorted(by_example):
        rows = by_example[example_id]

        def rows_for(condition: str, rows: list[dict] = rows) -> list[dict]:
            return [r for r in rows if r["vector_condition"] == condition]

        correct = rows_for(correct_condition)
        if not correct:
            continue
        beats_zero_prompts = [
            r["neutral_prompt_id"]
            for r in correct
            if (r.get("delta_logprob_vs_zero") or 0) > 0
        ]
        correct_mean = _mean([r.get("total_logprob") for r in correct])
        beats_controls: dict[str, bool | None] = {}
        for control in GONOGO_CONTROLS:
            totals = [
                r["total_logprob"]
                for r in rows_for(control)
                if r.get("total_logprob") is not None
            ]
            if not totals or correct_mean is None:
                beats_controls[control] = None
            else:
                beats_controls[control] = correct_mean > _mean(totals)
        recovered = [
            r
            for r in correct
            if r.get("target_recovered_exact") or r.get("target_recovered_substring")
        ]
        raw_mean = _mean([r.get("total_logprob") for r in rows_for("raw_activation")])
        verdicts.append(
            {
                "example_id": example_id,
                "correct_condition": correct_condition,
                "n_neutral_prompts_scored": len(correct),
                "n_prompts_beating_zero": len(beats_zero_prompts),
                "beats_zero_on_majority": len(beats_zero_prompts) * 2 > len(correct),
                "survives_multiple_prompts": len(beats_zero_prompts) >= 2,
                "beats_controls": beats_controls,
                "beats_all_controls": all(v is True for v in beats_controls.values()),
                "recovered_any": bool(recovered),
                "mean_total_logprob_correct": correct_mean,
                "mean_total_logprob_raw_activation": raw_mean,
                "jspace_vs_raw_activation": (
                    None
                    if correct_mean is None or raw_mean is None
                    else correct_mean - raw_mean
                ),
            }
        )
    return verdicts


def gonogo_report(verdicts: Sequence[dict]) -> dict:
    """Aggregate per-example verdicts into the go/no-go summary. ``go`` is
    True when a clear majority of examples pass the joint criterion (beats
    zero on a majority of prompts, beats every control, and survives more
    than one neutral prompt)."""
    if not verdicts:
        raise GenerativeError("no verdicts to aggregate")
    passing = [
        v
        for v in verdicts
        if v["beats_zero_on_majority"]
        and v["beats_all_controls"]
        and v["survives_multiple_prompts"]
    ]
    recovered = [v for v in verdicts if v["recovered_any"]]
    competitive = [
        v
        for v in verdicts
        if v["jspace_vs_raw_activation"] is not None
        and v["jspace_vs_raw_activation"] >= 0
    ]
    n = len(verdicts)
    return {
        "n_examples": n,
        "n_passing_joint_criterion": len(passing),
        "passing_fraction": len(passing) / n,
        "n_recovered_any": len(recovered),
        "recovery_fraction": len(recovered) / n,
        "n_jspace_competitive_with_raw": len(competitive),
        "go": len(passing) * 2 > n,
        "criteria": (
            "majority of examples: correct vector beats zero on a majority of "
            "neutral prompts, beats random/shuffled/sign-reversed/unrelated "
            "controls, and survives >= 2 neutral prompts"
        ),
    }


#: Natural-scale-matched controls the correct cone must beat, injected at
#: literally the same delta norm, for a natural-scale gain to be attributable
#: to the correct *direction* rather than to the injection magnitude alone.
#: Mirrors :data:`GONOGO_CONTROLS` (``mass_subcone`` is excluded there too — a
#: cone subset is a "how much is needed" ablation, not a specificity control).
NATURAL_SCALE_GONOGO_CONTROLS = (
    "natural_random_matched_norm",
    "natural_shuffled",
    "natural_sign_reversed",
    "natural_unrelated_cone",
)


def natural_scale_verdicts(
    records: Sequence[dict],
    *,
    correct_condition: str = "natural_scale",
    controls: Sequence[str] = NATURAL_SCALE_GONOGO_CONTROLS,
) -> list[dict]:
    """Per-example, per-schedule go/no-go among conditions matched to the
    correct cone's own (unscaled) injected norm.

    :func:`per_example_verdicts` answers this at one calibrated *ratio* —
    which cannot show whether a natural-scale gain is specific to the correct
    cone's direction, because :data:`GONOGO_CONTROLS` are injected at a
    different norm than the natural cone, and the natural cone's own effect
    varies by schedule (``constant``/``decaying`` reinject at every generated
    position; ``prompt_only`` does not). An apparent low-strength gain could
    otherwise be an artifact of the schedule or the injection magnitude rather
    than of the direction. This compares every condition at literally the same
    delta norm — the example's own cone — across every schedule
    ``natural_scale`` ran under, which is what isolates direction from those
    confounds.

    There is no ratio to calibrate on here (the injected norm is fixed by the
    example's own cone, not chosen), so the comparison is per (example,
    source_layer, schedule) rather than at one operating point.

    The zero baseline is looked up by ``(example_id, source_layer)`` alone,
    **not** by schedule: a zero delta is schedule-invariant (any schedule
    weight times zero is still zero), and the runner emits exactly one zero
    record per (example, layer) — under whichever schedule happens to be
    first in ``steering.schedules`` — rather than one per schedule. Keying the
    lookup by schedule as well would silently drop that baseline for every
    other schedule, producing ``mean_total_logprob_zero: null`` and a false
    ``beats_zero: False`` for them (this happened in a real run: constant and
    decaying both had a genuine positive ``delta_logprob_vs_zero``, but only
    ``prompt_only`` found its zero record and passed).
    """
    by_key: dict[tuple[str, int, str], dict[str, list[dict]]] = {}
    zero_by_example_layer: dict[tuple[str, int], list[dict]] = {}
    wanted = {correct_condition, "zero", *controls}
    for record in records:
        condition = record.get("vector_condition")
        if condition not in wanted:
            continue
        if condition == "zero":
            zero_by_example_layer.setdefault(
                (record["example_id"], record["source_layer"]), []
            ).append(record)
            continue
        key = (
            record["example_id"],
            record["source_layer"],
            record["steering_schedule"]["kind"],
        )
        by_key.setdefault(key, {}).setdefault(condition, []).append(record)

    verdicts = []
    for example_id, source_layer, schedule_kind in sorted(by_key):
        rows = by_key[(example_id, source_layer, schedule_kind)]
        correct = rows.get(correct_condition, [])
        if not correct:
            continue
        correct_mean = _mean([r.get("total_logprob") for r in correct])
        zero_rows = zero_by_example_layer.get((example_id, source_layer), [])
        zero_mean = _mean([r.get("total_logprob") for r in zero_rows])
        beats_zero = (
            correct_mean is not None
            and zero_mean is not None
            and correct_mean > zero_mean
        )
        beats_controls: dict[str, bool | None] = {}
        for control in controls:
            totals = [
                r["total_logprob"]
                for r in rows.get(control, [])
                if r.get("total_logprob") is not None
            ]
            if not totals or correct_mean is None:
                beats_controls[control] = None
            else:
                beats_controls[control] = correct_mean > _mean(totals)
        n_available = sum(1 for v in beats_controls.values() if v is not None)
        verdicts.append(
            {
                "example_id": example_id,
                "source_layer": source_layer,
                "schedule_kind": schedule_kind,
                "correct_condition": correct_condition,
                "mean_total_logprob_correct": correct_mean,
                "mean_total_logprob_zero": zero_mean,
                "beats_zero": beats_zero,
                "beats_controls": beats_controls,
                "n_controls_available": n_available,
                "beats_all_controls": (
                    n_available > 0
                    and all(v is True for v in beats_controls.values())
                ),
            }
        )
    return verdicts


def natural_scale_gonogo_report(verdicts: Sequence[dict]) -> dict:
    """Aggregate :func:`natural_scale_verdicts` the way :func:`gonogo_report`
    aggregates :func:`per_example_verdicts`: ``go`` is True when a clear
    majority of (example, layer, schedule) points beat zero and every
    natural-scale-matched control."""
    if not verdicts:
        raise GenerativeError("no natural-scale verdicts to aggregate")
    passing = [v for v in verdicts if v["beats_zero"] and v["beats_all_controls"]]
    n = len(verdicts)
    return {
        "n_points": n,
        "n_passing": len(passing),
        "passing_fraction": len(passing) / n,
        "go": len(passing) * 2 > n,
        "criteria": (
            "majority of (example, layer, schedule) points at the correct "
            "cone's own natural norm: beats zero and every natural-scale-"
            f"matched control ({', '.join(NATURAL_SCALE_GONOGO_CONTROLS)})"
        ),
    }
