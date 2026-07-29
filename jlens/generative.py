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
#: ``wrong_layer`` / ``wrong_position`` reuse the correct full-cone vector at
#: an incorrect site (the vector builder returns the same delta; the runner
#: moves the site).
VECTOR_CONDITIONS = (
    "none",
    "zero",
    "full_cone",
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
)

SCHEDULE_KINDS = ("prompt_only", "constant", "decaying")


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


def _next_token_logprobs(model, input_ids: torch.Tensor) -> torch.Tensor:
    """Full-sequence next-token log-probabilities ``[seq, vocab]`` through the
    model's native output pathway (softcap included when present)."""
    hidden = model.forward(input_ids).last_hidden_state
    logits = model.unembed(hidden)[0].float()
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
            log_p = _next_token_logprobs(model, ids)[-1]
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
    with torch.no_grad(), context:
        log_p = _next_token_logprobs(model, full)
    per_token = [
        float(log_p[prompt_len - 1 + j, token]) for j, token in enumerate(targets)
    ]
    first_row = log_p[prompt_len - 1]
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
        return _next_token_logprobs(model, input_ids)[-1]


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

    if condition in ("full_cone", "wrong_layer", "wrong_position", "sign_reversed", "shuffled"):
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
        if condition == "sign_reversed":
            return ConditionVector(condition, -q, meta)
        if condition == "shuffled":
            s = need(seed, "seed")
            return ConditionVector(
                condition, shuffled_coordinates(q, seed=s), {**meta, "seed": int(s)}
            )
        return ConditionVector(condition, q, meta)

    if condition == "mass_subcone":
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

    if condition == "unrelated_cone":
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

    if condition == "random_matched_norm":
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
    target_recovered_exact: bool | None,
    target_recovered_substring: bool | None,
    seed: int | None,
    provenance: dict,
) -> dict:
    """Assemble one JSON-safe result record (schema
    ``jlens.generative.record.v1``). No activation tensors are stored."""
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
        "requested_ratio": strength_ratio,
        "measured_ratio": hook_stats.get("measured_ratio"),
        "delta_norm": hook_stats.get("delta_norm"),
        "receiving_activation_norm": hook_stats.get("anchor_activation_norm"),
        "vector_meta": vector_meta,
        "target_phrase": target_phrase,
        "delta_logprob_vs_zero": delta_vs_zero,
        "delta_logprob_vs_unrelated": delta_vs_unrelated,
        "kl_divergence_from_baseline": kl_divergence,
        "target_recovered_exact": target_recovered_exact,
        "target_recovered_substring": target_recovered_substring,
        "random_seed": seed,
        "provenance": dict(provenance),
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


#: Neutral verbalization prompts. None mentions any target concept; each ends
#: at the point where the concept label should be produced.
NEUTRAL_PROMPTS: dict[str, str] = {
    "label-colon": (
        "Return only the shortest noun or noun phrase naming the represented "
        "internal concept.\nLabel:"
    ),
    "answer-four-words": (
        "Name the represented internal concept using no more than four "
        "words.\nAnswer:"
    ),
    "shortest-label-is": "The shortest label for the internal representation is:",
}


# ------------------------------------------------------------- aggregation


def _mean(values: Sequence[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


def summarize_by_condition(records: Sequence[dict]) -> list[dict]:
    """Per-condition aggregate of scored records: mean/min/max target
    log-probability improvement over zero, specificity over the unrelated
    cone, recovery rates, and mean KL. Purely descriptive."""
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
