# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Residual-stream interventions at block_output sites.

Minimal, correct infrastructure for the causal steering smoke study on
Gemma 4 E4B. The intervention site is exactly the site the lens reads and
the pursuit decomposes (see :mod:`jlens.gemma4`): the tensor returned by
``model.language_model.layers[l]`` — the residual stream after attention,
MLP, PLE re-injection, and the block's ``layer_scalar``. Some HF blocks
return tuples ``(hidden, ...)``; only element 0 is edited and every other
element is passed through untouched.

Semantics (documented in ``docs/causal_smoke_run.md``)::

    h_intervened[row, pos] = h_original[row, pos] + multiplier * delta

with the arithmetic done in float32 and the result cast back to the block
output's dtype and device. ``multiplier = 0`` therefore reproduces the
original values exactly (bf16 -> fp32 -> bf16 round-trips are exact and the
added term is exactly zero), which is one of the three baseline-parity
checks; ``delta=None`` is the identical-copy writeback check (clone and
write back, no arithmetic).

Deltas come from the *existing* pursuit convention, audited rather than
assumed: coefficients in a cone record refer to the **raw** dictionary atoms
(rows of ``W_U @ J_l``; ``normalize_atoms`` only rescales selection
correlations — see :mod:`jlens.pursuit`), so an atom's contribution to the
reconstruction is ``c_v * atoms[v]`` and the full-cone delta is the recorded
reconstruction ``sum_v c_v * atoms[v]``. The isotropic random control is a
deterministic CPU-generated Gaussian direction rescaled to exactly the norm
of its matched target delta.

Safety rails: model parameters must be frozen; positions are validated
against the actual sequence length; nonfinite deltas and nonfinite edited
outputs raise; hooks are removed on success *and* on every exception path;
no raw activation tensor is stored by default (only scalar norms).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import nn

INTERVENTION_RECORD_SCHEMA = "jlens.interventions.record.v1"

TARGET_KINDS = (
    "output_atom_contribution",
    "top_non_output_atom_contribution",
    "full_cone_reconstruction",
    "isotropic_random_direction",
)


class InterventionError(ValueError):
    """Invalid intervention request or corrupted state."""


# ------------------------------------------------------------ condition IDs


def condition_id(
    *,
    example_id: str,
    layer: int,
    position: int,
    target_kind: str,
    multiplier: float,
    atom_token_id: int | None = None,
    norm_preserving: bool = False,
    variant: str | None = None,
) -> str:
    """Deterministic condition identifier.

    A pure function of the condition's defining coordinates — the same
    condition always maps to the same ID across runs and machines, which is
    what makes checkpoint/resume and record merging safe.
    """
    if target_kind not in TARGET_KINDS:
        raise InterventionError(
            f"target_kind {target_kind!r} not in {TARGET_KINDS}"
        )
    payload = json.dumps(
        {
            "example_id": example_id,
            "layer": int(layer),
            "position": int(position),
            "target_kind": target_kind,
            "multiplier": float(multiplier),
            "atom_token_id": None if atom_token_id is None else int(atom_token_id),
            "norm_preserving": bool(norm_preserving),
            "variant": variant,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return "cond_" + hashlib.sha256(payload.encode()).hexdigest()[:16]


def _derived_seed(condition: str) -> int:
    """Deterministic 31-bit seed from a condition ID."""
    return int(hashlib.sha256(condition.encode()).hexdigest()[:8], 16) % (2**31)


# ---------------------------------------------------------------- deltas


@dataclass(frozen=True)
class DeltaInfo:
    """What a delta is and where it came from (JSON-safe via ``__dict__``)."""

    target_kind: str
    atom_token_id: int | None
    atom_label: str | None
    coefficient: float | None
    delta_norm: float
    seed: int | None = None
    matched_target_condition_id: str | None = None


def _finite_or_raise(tensor: torch.Tensor, what: str) -> None:
    if not bool(torch.isfinite(tensor).all()):
        raise InterventionError(f"{what} contains NaN/Inf")


def cone_delta(
    atoms: torch.Tensor,
    cone_record: dict,
    *,
    target_kind: str,
    model_top1_id: int | None,
) -> tuple[torch.Tensor, DeltaInfo]:
    """Build a targeted delta from a completed k=10 cone record.

    Args:
        atoms: The layer's J-space dictionary atoms ``[n_atoms, d_model]``
            (raw, unnormalized — the pursuit coefficient convention).
        cone_record: A ``jlens.cones.record.v1`` dict (effective sets used).
        target_kind: ``output_atom_contribution`` /
            ``top_non_output_atom_contribution`` / ``full_cone_reconstruction``.
        model_top1_id: The model's measured top-1 token id at this
            example/position (defines which atom is "the output atom").

    Returns:
        ``(delta, info)`` with ``delta`` float32 on ``atoms.device``.
    """
    ids = [int(t) for t in cone_record["effective_token_ids"]]
    labels = list(cone_record["effective_labels"])
    coeffs = [float(c) for c in cone_record["effective_coefficients"]]
    if not ids:
        raise InterventionError("cone record has an empty effective set")

    if target_kind == "full_cone_reconstruction":
        index_tensor = torch.tensor(ids, dtype=torch.long, device=atoms.device)
        coeff_tensor = torch.tensor(coeffs, dtype=torch.float32, device=atoms.device)
        delta = torch.einsum("m,md->d", coeff_tensor, atoms[index_tensor].float())
        _finite_or_raise(delta, "full-cone delta")
        return delta, DeltaInfo(
            target_kind=target_kind,
            atom_token_id=None,
            atom_label=None,
            coefficient=None,
            delta_norm=float(delta.norm()),
        )

    if target_kind == "output_atom_contribution":
        if model_top1_id is None or int(model_top1_id) not in ids:
            raise InterventionError(
                f"model top-1 id {model_top1_id} is not in the cone's effective "
                f"set; no output-atom intervention is available here"
            )
        pick = ids.index(int(model_top1_id))
    elif target_kind == "top_non_output_atom_contribution":
        candidates = [
            (coeff, index)
            for index, (token, coeff) in enumerate(zip(ids, coeffs, strict=True))
            if model_top1_id is None or token != int(model_top1_id)
        ]
        if not candidates:
            raise InterventionError("no non-output atom in the effective set")
        # Highest coefficient; ties break to the earliest-selected atom.
        pick = max(candidates, key=lambda pair: (pair[0], -pair[1]))[1]
    else:
        raise InterventionError(f"unsupported target_kind {target_kind!r}")

    delta = coeffs[pick] * atoms[ids[pick]].float()
    _finite_or_raise(delta, f"{target_kind} delta")
    return delta, DeltaInfo(
        target_kind=target_kind,
        atom_token_id=ids[pick],
        atom_label=labels[pick],
        coefficient=coeffs[pick],
        delta_norm=float(delta.norm()),
    )


def isotropic_random_direction(
    d_model: int,
    *,
    match_norm: float,
    seed: int,
    device: torch.device | str = "cpu",
    matched_target_condition_id: str | None = None,
) -> tuple[torch.Tensor, DeltaInfo]:
    """Deterministic isotropic Gaussian direction with **exactly** the given
    norm. Generated on CPU with a fixed-seed generator (bitwise reproducible
    across machines), then moved to ``device``."""
    if not math.isfinite(match_norm) or match_norm < 0:
        raise InterventionError(f"match_norm must be finite and >= 0, got {match_norm}")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    direction = torch.randn(d_model, generator=generator, dtype=torch.float32)
    norm = direction.norm()
    if float(norm) == 0.0:
        raise InterventionError("degenerate zero random direction")
    delta = (direction / norm * match_norm).to(device)
    _finite_or_raise(delta, "random-control delta")
    return delta, DeltaInfo(
        target_kind="isotropic_random_direction",
        atom_token_id=None,
        atom_label=None,
        coefficient=None,
        delta_norm=float(delta.norm()),
        seed=int(seed),
        matched_target_condition_id=matched_target_condition_id,
    )


# ------------------------------------------------------------------ hooks


def _assert_frozen(module: nn.Module) -> None:
    for name, parameter in module.named_parameters():
        if parameter.requires_grad:
            raise InterventionError(
                f"model parameter {name} requires grad; freeze the model before "
                f"intervening"
            )


@contextmanager
def residual_intervention(
    blocks: Sequence[nn.Module],
    layer: int,
    *,
    position: int,
    delta: torch.Tensor | None,
    multiplier: float = 1.0,
    batch_row: int = 0,
    norm_preserving: bool = False,
    require_frozen: bool = True,
):
    """Context manager: edit one (row, position) of ``blocks[layer]``'s output.

    Inside the ``with`` block, the next forward pass(es) see::

        h[batch_row, position] += multiplier * delta        (float32 math)

    cast back to the block output's dtype/device. Tuple outputs have only
    element 0 edited; all other elements pass through untouched.

    Args:
        blocks: The decoder blocks (``model.language_model.layers``).
        layer: Block index to intervene at.
        position: Token position; negative indices resolve against the actual
            sequence length. Out-of-range positions raise at forward time.
        delta: ``[d_model]`` float direction, or ``None`` for the
            identical-copy writeback parity mode (clone, no arithmetic).
        multiplier: Signed scale; 0 is the exact-no-op parity mode.
        batch_row: Which batch row to edit (validated at forward time).
        norm_preserving: If True, the edited vector is rescaled to the
            original vector's L2 norm after the addition.
        require_frozen: Verify no parameter requires grad (default on).

    The hook records scalar bookkeeping in the yielded dict:
    ``activation_norm`` (pre-edit), ``edited_norm`` (post-edit),
    ``delta_norm``, ``resolved_position``, ``n_forward_passes``. No raw
    activation tensor is retained.

    The hook handle is removed on normal exit **and** on any exception,
    including exceptions raised by the hook itself during ``model(...)``.
    """
    if not isinstance(layer, int) or not (0 <= layer < len(blocks)):
        raise InterventionError(
            f"layer {layer} out of range for {len(blocks)} blocks"
        )
    if not math.isfinite(multiplier):
        raise InterventionError(f"multiplier must be finite, got {multiplier}")
    if delta is not None:
        if delta.ndim != 1:
            raise InterventionError(
                f"delta must be a 1-D [d_model] vector, got shape {tuple(delta.shape)}"
            )
        _finite_or_raise(delta, "delta")
    if require_frozen:
        _assert_frozen(blocks[layer])

    stats: dict = {
        "delta_norm": None if delta is None else float(delta.float().norm()),
        "activation_norm": None,
        "edited_norm": None,
        "resolved_position": None,
        "n_forward_passes": 0,
        "norm_preserving": bool(norm_preserving),
        "multiplier": float(multiplier),
    }

    def hook(module: nn.Module, inputs, output):
        is_tensor = torch.is_tensor(output)
        hidden = output if is_tensor else output[0]
        if hidden.ndim != 3:
            raise InterventionError(
                f"expected [batch, seq, d_model] block output, got "
                f"{tuple(hidden.shape)}"
            )
        batch, seq_len, _ = hidden.shape
        if not (0 <= batch_row < batch):
            raise InterventionError(
                f"batch_row {batch_row} out of range for batch size {batch}"
            )
        resolved = position if position >= 0 else seq_len + position
        if not (0 <= resolved < seq_len):
            raise InterventionError(
                f"position {position} out of range for sequence length {seq_len}"
            )
        original = hidden[batch_row, resolved].float()
        stats["activation_norm"] = float(original.norm())
        stats["resolved_position"] = int(resolved)
        stats["n_forward_passes"] += 1

        new_hidden = hidden.clone()
        if delta is None:
            edited = original
        else:
            edited = original + multiplier * delta.to(
                device=original.device, dtype=torch.float32
            )
            if norm_preserving:
                edited_norm = edited.norm()
                if float(edited_norm) > 0:
                    edited = edited * (original.norm() / edited_norm)
        if not bool(torch.isfinite(edited).all()):
            raise InterventionError(
                "intervened activation contains NaN/Inf; refusing to continue"
            )
        stats["edited_norm"] = float(edited.norm())
        new_hidden[batch_row, resolved] = edited.to(hidden.dtype)
        if is_tensor:
            return new_hidden
        return (new_hidden, *tuple(output)[1:])

    handle = blocks[layer].register_forward_hook(hook)
    try:
        yield stats
    finally:
        handle.remove()


# ---------------------------------------------------------------- metrics


def _rank_of(logits: torch.Tensor, token_id: int) -> int:
    """0-based rank of ``token_id`` (0 = argmax); deterministic on ties via
    strict greater-than counting."""
    value = logits[token_id]
    return int((logits > value).sum())


def logit_metrics(
    logits_before: torch.Tensor,
    logits_after: torch.Tensor,
    *,
    target_token_id: int,
    top_k: int = 10,
    decode=None,
) -> dict:
    """Before/after readout metrics for one intervention, all in float32.

    ``decode(token_id) -> str`` optionally labels the top-k lists. KL is
    ``KL(P_after || P_before)`` over the full softmax.
    """
    before = logits_before.detach().float().flatten()
    after = logits_after.detach().float().flatten()
    if before.shape != after.shape:
        raise InterventionError(
            f"logit shapes differ: {tuple(before.shape)} vs {tuple(after.shape)}"
        )
    _finite_or_raise(before, "logits_before")
    _finite_or_raise(after, "logits_after")

    log_p_before = torch.log_softmax(before, dim=-1)
    log_p_after = torch.log_softmax(after, dim=-1)
    p_after = log_p_after.exp()
    kl = float((p_after * (log_p_after - log_p_before)).sum())

    def topk(logits: torch.Tensor, log_p: torch.Tensor) -> list[dict]:
        values, indices = logits.topk(top_k)
        return [
            {
                "token_id": int(i),
                "token": decode(int(i)) if decode else None,
                "logit": float(v),
                "prob": float(log_p[int(i)].exp()),
            }
            for v, i in zip(values, indices, strict=True)
        ]

    top_before = topk(before, log_p_before)
    top_after = topk(after, log_p_after)
    ids_before = {t["token_id"] for t in top_before}
    ids_after = {t["token_id"] for t in top_after}
    target = int(target_token_id)
    return {
        "target_token_id": target,
        "target_logit_before": float(before[target]),
        "target_logit_after": float(after[target]),
        "target_logit_delta": float(after[target] - before[target]),
        "target_rank_before": _rank_of(before, target),
        "target_rank_after": _rank_of(after, target),
        "target_prob_before": float(log_p_before[target].exp()),
        "target_prob_after": float(log_p_after[target].exp()),
        "top1_before": top_before[0],
        "top1_after": top_after[0],
        "top10_before": top_before,
        "top10_after": top_after,
        "top10_overlap": len(ids_before & ids_after) / top_k,
        "kl_divergence_after_vs_before": kl,
    }


def parity_report(
    logits_a: torch.Tensor, logits_b: torch.Tensor, *, top_k: int = 10
) -> dict:
    """Comparison used by the baseline-parity gate (unhooked vs multiplier-0
    vs identical-copy writeback)."""
    a = logits_a.detach().float().flatten()
    b = logits_b.detach().float().flatten()
    if a.shape != b.shape:
        raise InterventionError(
            f"logit shapes differ: {tuple(a.shape)} vs {tuple(b.shape)}"
        )
    diff = (a - b).abs()
    top_a = set(a.topk(top_k).indices.tolist())
    top_b = set(b.topk(top_k).indices.tolist())
    return {
        "max_abs_logit_diff": float(diff.max()),
        "mean_abs_logit_diff": float(diff.mean()),
        "top1_identical": int(a.argmax()) == int(b.argmax()),
        "top10_overlap": len(top_a & top_b) / top_k,
    }


# ------------------------------------------------- append-safe record store


def make_intervention_record(
    *,
    condition: str,
    example_id: str,
    layer: int,
    position: int,
    delta_info: DeltaInfo,
    multiplier: float,
    stats: dict,
    metrics: dict | None,
    status: str = "measured",
    control_family: str | None = None,
    completion_before: str | None = None,
    completion_after: str | None = None,
    provenance: dict | None = None,
) -> dict:
    """Assemble one JSON-safe intervention record (schema
    ``jlens.interventions.record.v1``)."""
    activation_norm = stats.get("activation_norm")
    delta_norm = delta_info.delta_norm
    record = {
        "schema": INTERVENTION_RECORD_SCHEMA,
        "condition_id": condition,
        "example_id": example_id,
        "layer": int(layer),
        "position": int(position),
        "target_kind": delta_info.target_kind,
        "atom_token_id": delta_info.atom_token_id,
        "atom_label": delta_info.atom_label,
        "atom_coefficient": delta_info.coefficient,
        "multiplier": float(multiplier),
        "status": status,
        "norm_preserving": bool(stats.get("norm_preserving", False)),
        "delta_norm": delta_norm,
        "activation_norm": activation_norm,
        "delta_to_activation_ratio": (
            delta_norm / activation_norm
            if activation_norm not in (None, 0)
            else None
        ),
        "resolved_position": stats.get("resolved_position"),
        "control_family": control_family,
        "matched_target_condition_id": delta_info.matched_target_condition_id,
        "random_seed": delta_info.seed,
        "completion_before": completion_before,
        "completion_after": completion_after,
        "provenance": provenance or {},
    }
    if metrics:
        record.update(metrics)
    return record


def append_record(path: str, record: dict) -> None:
    """Append one record as a JSONL line (flushed + fsynced so an interrupted
    run keeps every completed condition)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_records(path: str) -> list[dict]:
    """Load a JSONL record file (missing file -> empty list); malformed lines
    raise rather than being silently dropped."""
    if not os.path.isfile(path):
        return []
    records = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise InterventionError(
                    f"{path}:{line_number}: malformed JSONL line"
                ) from exc
            if record.get("schema") != INTERVENTION_RECORD_SCHEMA:
                raise InterventionError(
                    f"{path}:{line_number}: unexpected schema "
                    f"{record.get('schema')!r}"
                )
            records.append(record)
    return records


def completed_condition_ids(path: str) -> set[str]:
    """Condition IDs already recorded — the resume skip-list."""
    return {record["condition_id"] for record in load_records(path)}


def assert_run_resumable(run_dir: str) -> None:
    """Refuse to resume into a run that already wrote its final metadata
    (completed runs are immutable; start a fresh run directory instead)."""
    final = os.path.join(run_dir, "run_metadata.json")
    if os.path.isfile(final):
        raise InterventionError(
            f"{run_dir} already contains run_metadata.json (completed run); "
            f"refusing to resume — start a new run directory"
        )
