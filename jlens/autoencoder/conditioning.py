# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""How the adapter's memory enters frozen Gemma.

The pilot uses **soft-prefix memory**: a forward hook on the input-embedding
module overwrites the rows of its ``[batch, seq, d_model]`` output that
correspond to the prompt's memory slots. Everything downstream — every block,
the final norm, the LM head, the logit softcap — runs untouched, in the
library's own order, so gradients reach the adapter through the entire frozen
stack and no part of the model is bypassed.

The mechanism sits behind :class:`ConditioningBackend` so a native-layer
variant can replace it without touching the adapter, the trainers, or the
evaluator. See :data:`NATIVE_LAYER_MEMORY_TODO`.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from typing import Protocol, runtime_checkable

import torch
from torch import nn

from jlens.autoencoder.errors import AutoencoderError

#: What a native-layer memory backend would have to change, recorded so the
#: option stays open and cheap:
#:
#: 1. Replace the ``embed_tokens`` hook with a hook on ``model.layers[L]``'s
#:    output, writing (not adding) the memory rows at the same span — the
#:    mechanics of :func:`jlens.generative.steering_injection`, minus the
#:    schedule, plus write-instead-of-add semantics.
#: 2. Recalibrate :func:`measure_memory_scale` against the *layer-L residual*
#:    RMS instead of the embedding RMS.
#: 3. Nothing else: the prompt construction, adapter, beam search, preference
#:    loop, and evaluation take the backend as an argument.
NATIVE_LAYER_MEMORY_TODO = (
    "native-layer memory: hook blocks[L].output instead of embed_tokens, write "
    "the memory rows at the same span, and recalibrate the memory scale against "
    "the layer-L residual RMS"
)


@runtime_checkable
class ConditioningBackend(Protocol):
    """Anything that can make frozen Gemma read a continuous memory."""

    name: str

    def conditioned(
        self, model, *, memory: torch.Tensor, span: tuple[int, int]
    ):  # pragma: no cover - protocol
        """Context manager installing ``memory`` at token positions ``span``."""
        ...


def assert_gemma_frozen(model, *, where: str = "model") -> dict:
    """Fail unless every parameter of the wrapped model has ``requires_grad=False``.

    Returns a small report (parameter count, dtype set) so a run records that
    the check ran and what it saw, rather than only that it did not raise.
    """
    hf_model = getattr(model, "_hf_model", model)
    trainable = [n for n, p in hf_model.named_parameters() if p.requires_grad]
    if trainable:
        raise AutoencoderError(
            f"{where}: {len(trainable)} parameter(s) require grad (first: "
            f"{trainable[0]}); Gemma must stay frozen for this experiment"
        )
    parameters = list(hf_model.parameters())
    return {
        "frozen": True,
        "n_parameters": sum(p.numel() for p in parameters),
        "n_tensors": len(parameters),
        "dtypes": sorted({str(p.dtype) for p in parameters}),
    }


def assert_no_frozen_parameters_in_optimizer(
    optimizer, *, frozen_modules: Sequence[nn.Module | object], trainable: nn.Module
) -> dict:
    """Prove the optimizer holds *only* ``trainable``'s parameters.

    Membership is checked by tensor identity (``id``), not by name: a name-based
    check passes for a module that was re-registered under a different
    attribute, which is exactly how a frozen model sneaks into an optimizer.
    """
    allowed = {id(p) for p in trainable.parameters()}
    forbidden: set[int] = set()
    for module in frozen_modules:
        target = getattr(module, "_hf_model", module)
        if isinstance(target, nn.Module):
            forbidden |= {id(p) for p in target.parameters()}
    n_params = 0
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            n_params += 1
            if id(parameter) in forbidden:
                raise AutoencoderError(
                    "optimizer contains a parameter belonging to a frozen module; "
                    "only the trainable module's parameters may be optimized"
                )
            if id(parameter) not in allowed:
                raise AutoencoderError(
                    "optimizer contains a parameter that is not part of the "
                    "trainable module"
                )
    if n_params == 0:
        raise AutoencoderError("optimizer holds no parameters")
    return {
        "n_optimizer_tensors": n_params,
        "n_trainable_tensors": len(allowed),
        "n_frozen_tensors_checked": len(forbidden),
    }


@torch.no_grad()
def measure_memory_scale(model, input_ids: torch.Tensor) -> dict:
    """Measure the RMS of the embedding module's output on real token ids.

    The adapter's memory is written into this stream, so its magnitude has to be
    calibrated against what actually lives there. Gemma folds a ``sqrt(d_model)``
    scale into the embedding module itself, which is why this is measured at the
    module's *output* rather than computed from the weight.
    """
    embed = _embedding_module(model)
    captured: list[torch.Tensor] = []

    def hook(_module, _inputs, output):
        captured.append(output.detach().float())

    handle = embed.register_forward_hook(hook)
    try:
        embed(input_ids.to(_embedding_device(model)))
    finally:
        handle.remove()
    if not captured:
        raise AutoencoderError("embedding hook captured nothing")
    values = captured[0]
    return {
        "embedding_rms": float(values.pow(2).mean().sqrt()),
        "embedding_mean_norm": float(values.norm(dim=-1).mean()),
        "d_model": int(values.shape[-1]),
        "n_positions_measured": int(values.shape[0] * values.shape[1]),
    }


def _embedding_module(model) -> nn.Module:
    embed = getattr(model, "_embed_tokens", None)
    if embed is None:
        raise AutoencoderError(
            "model has no _embed_tokens; a Gemma4LensModel / HFLensModel is required"
        )
    return embed


def _embedding_device(model) -> torch.device:
    return _embedding_module(model).weight.device


class SoftPrefixConditioner:
    """Write the memory into the input-embedding stream at a fixed token span.

    The filler token's own embedding is **overwritten**, never added to, so the
    memory is the sole content of those positions and the filler id cannot leak
    semantics.
    """

    name = "soft_prefix_embedding"

    def __init__(self, *, require_frozen: bool = True) -> None:
        self.require_frozen = require_frozen

    @contextmanager
    def conditioned(self, model, *, memory: torch.Tensor, span: tuple[int, int]):
        """Install ``memory`` for every forward pass inside the ``with`` block.

        Args:
            model: A :class:`~jlens.hf.HFLensModel` (or compatible).
            memory: ``[M, d_model]`` (broadcast over the batch) or
                ``[batch, M, d_model]``.
            span: ``(start, end)`` token positions; ``end - start`` must equal
                ``M``.

        Yields a stats dict recording how many forward passes were conditioned
        and the memory's measured RMS — enough to tell a silently-inert hook
        (zero passes) from a working one.
        """
        start, end = int(span[0]), int(span[1])
        if start < 0 or end <= start:
            raise AutoencoderError(f"invalid memory span {span!r}")
        if memory.ndim == 2:
            memory = memory.unsqueeze(0)
        if memory.ndim != 3:
            raise AutoencoderError(
                f"memory must be [M, d] or [batch, M, d], got {tuple(memory.shape)}"
            )
        if memory.shape[1] != end - start:
            raise AutoencoderError(
                f"memory has {memory.shape[1]} slots but the span covers "
                f"{end - start} positions"
            )
        if not bool(torch.isfinite(memory).all()):
            raise AutoencoderError("memory contains NaN/Inf")
        if self.require_frozen:
            assert_gemma_frozen(model, where="conditioning")

        embed = _embedding_module(model)
        stats: dict = {
            "backend": self.name,
            "memory_span": [start, end],
            "n_memory_tokens": end - start,
            "memory_rms": float(memory.detach().float().pow(2).mean().sqrt()),
            "n_conditioned_forward_passes": 0,
        }

        def hook(_module: nn.Module, _inputs, output: torch.Tensor) -> torch.Tensor:
            if output.ndim != 3:
                raise AutoencoderError(
                    f"expected [batch, seq, d_model] embedding output, got "
                    f"{tuple(output.shape)}"
                )
            batch, seq_len, d_model = output.shape
            if seq_len < end:
                raise AutoencoderError(
                    f"forward pass of length {seq_len} does not reach the memory "
                    f"span [{start}, {end}); the prompt was truncated or the wrong "
                    f"ids were passed"
                )
            if memory.shape[-1] != d_model:
                raise AutoencoderError(
                    f"memory width {memory.shape[-1]} != d_model {d_model}"
                )
            rows = memory
            if rows.shape[0] == 1 and batch > 1:
                rows = rows.expand(batch, -1, -1)
            elif rows.shape[0] != batch:
                raise AutoencoderError(
                    f"memory batch {rows.shape[0]} does not match forward batch {batch}"
                )
            edited = output.clone()
            edited[:, start:end, :] = rows.to(device=output.device, dtype=output.dtype)
            stats["n_conditioned_forward_passes"] += 1
            return edited

        handle = embed.register_forward_hook(hook)
        try:
            yield stats
        finally:
            handle.remove()


@contextmanager
def no_conditioning():
    """A conditioning context that installs nothing — the zero-memory baseline's
    'not even a hook' variant, kept explicit so a caller never has to branch on
    ``None``."""
    yield {
        "backend": "none",
        "memory_span": None,
        "n_memory_tokens": 0,
        "memory_rms": None,
        "n_conditioned_forward_passes": 0,
    }
