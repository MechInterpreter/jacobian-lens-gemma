# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Refuse an fp32 model load cleanly, before it starts, if the GPU is too small.

The cat->dog animal-sound study runs the model in float32 rather than bf16 —
deliberately, because the corrected instrument's own diagnostics
(:mod:`jlens.mmpilot.multimodal_instrument`) showed the bf16 cast alone
accounts for the entire post-cast coordinate error on the confirmed leg-count
replay (the float64 pre-cast solve error was ~4e-12; all measured error was
introduced by the cast to bf16). fp32 has a 24-bit mantissa against bf16's
7-bit one, so the same exchange should realize far more cleanly.

fp32 weights are roughly twice the size of the bf16 checkpoint, though, and
this repository does not silently fall back to bf16 when fp32 does not fit:
that would substitute a different, undeclared instrument for the one the
study's own scientific config records. This module estimates whether an fp32
load will fit **before** any weights are loaded, and raises a clean, specific
refusal instead.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

__all__ = [
    "FP32_PREFLIGHT_VERSION",
    "GEMMA4_E4B_APPROX_PARAM_COUNT",
    "GEMMA4_E4B_BF16_CHECKPOINT_BYTES_REFERENCE",
    "InsufficientGPUMemoryError",
    "Fp32MemoryEstimate",
    "estimate_fp32_inference_memory",
    "assert_fp32_memory_sufficient",
    "preflight_fp32_or_refuse",
]

FP32_PREFLIGHT_VERSION = "mmpilot.fp32_inference_memory_preflight.v1"

#: ``jlens.gemma4.load_gemma4`` refuses its bf16 download without
#: ``allow_model_load=True`` and names it "~16 GB" in that refusal message —
#: this is the only documented size reference for this checkpoint in the
#: repository, so the fp32 estimate is derived from it rather than restated as
#: an independent guess. 16 GiB, per that comment.
GEMMA4_E4B_BF16_CHECKPOINT_BYTES_REFERENCE = 16 * 1024**3

_BYTES_PER_PARAM_BF16 = 2
_BYTES_PER_PARAM_FP32 = 4

#: Implied by the bf16 checkpoint size at 2 bytes/parameter. This is an
#: approximation stated for exactly what it is; a caller may override
#: ``n_params`` in every function below if a truer count becomes available
#: (e.g. read from the loaded model's own ``.numel()`` sum on a prior run).
GEMMA4_E4B_APPROX_PARAM_COUNT = (
    GEMMA4_E4B_BF16_CHECKPOINT_BYTES_REFERENCE // _BYTES_PER_PARAM_BF16
)


class InsufficientGPUMemoryError(RuntimeError):
    """The estimated fp32 load would not fit; refused before touching the GPU."""


@dataclass(frozen=True)
class Fp32MemoryEstimate:
    """Every component of the estimate, so the number is auditable, not magic.

    Attributes:
        n_params: Parameter count the weight estimate is based on.
        weight_bytes: ``n_params * 4`` — the fp32 weight tensors themselves.
        workspace_bytes: Activations, KV-cache for the few generated tokens,
            and framework scratch space, estimated as ``workspace_fraction *
            weight_bytes``. Generous for batch-size-one inference with no
            gradients and no optimizer state, since those are the only two
            things this estimate must NOT need room for.
        cuda_context_bytes: Fixed overhead for the CUDA context itself, before
            any tensor is allocated.
        safety_margin: Multiplies the raw total; the preflight requires free
            memory to exceed ``total_bytes * safety_margin``, not merely
            ``total_bytes``, so a load that would fit by a hair is refused
            rather than risking a mid-run OOM.
    """

    n_params: int
    weight_bytes: int
    workspace_bytes: int
    cuda_context_bytes: int
    safety_margin: float

    @property
    def raw_total_bytes(self) -> int:
        return self.weight_bytes + self.workspace_bytes + self.cuda_context_bytes

    @property
    def required_bytes(self) -> int:
        return int(self.raw_total_bytes * float(self.safety_margin))

    @property
    def raw_total_gib(self) -> float:
        return self.raw_total_bytes / 1024**3

    @property
    def required_gib(self) -> float:
        return self.required_bytes / 1024**3

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["version"] = FP32_PREFLIGHT_VERSION
        payload["raw_total_bytes"] = self.raw_total_bytes
        payload["required_bytes"] = self.required_bytes
        payload["raw_total_gib"] = self.raw_total_gib
        payload["required_gib"] = self.required_gib
        return payload


def estimate_fp32_inference_memory(
    *,
    n_params: int = GEMMA4_E4B_APPROX_PARAM_COUNT,
    workspace_fraction: float = 0.30,
    cuda_context_bytes: int = 1 * 1024**3,
    safety_margin: float = 1.15,
) -> Fp32MemoryEstimate:
    """Build the estimate. Pure computation — no GPU is touched.

    Args:
        n_params: See :data:`GEMMA4_E4B_APPROX_PARAM_COUNT`.
        workspace_fraction: Activation/KV-cache/scratch budget as a fraction of
            weight size. 0.30 is generous for batch=1, no-gradient inference;
            widen it (never shrink it to force a pass) if a real run is
            observed to need more.
        cuda_context_bytes: Fixed CUDA context overhead, 1 GiB by default.
        safety_margin: See :class:`Fp32MemoryEstimate`.
    """
    if n_params <= 0:
        raise ValueError("n_params must be positive")
    if not 0.0 <= workspace_fraction:
        raise ValueError("workspace_fraction must be non-negative")
    if safety_margin < 1.0:
        raise ValueError(
            "safety_margin must be >= 1.0; a margin below 1 would accept a "
            "load estimated to already exceed available memory"
        )
    weight_bytes = int(n_params) * _BYTES_PER_PARAM_FP32
    workspace_bytes = int(weight_bytes * float(workspace_fraction))
    return Fp32MemoryEstimate(
        n_params=int(n_params),
        weight_bytes=weight_bytes,
        workspace_bytes=workspace_bytes,
        cuda_context_bytes=int(cuda_context_bytes),
        safety_margin=float(safety_margin),
    )


def assert_fp32_memory_sufficient(
    *,
    free_bytes: int,
    device_name: str = "",
    total_bytes: int | None = None,
    estimate: Fp32MemoryEstimate | None = None,
    **estimate_kwargs,
) -> dict:
    """Refuse, with a specific message, if ``free_bytes`` cannot cover the
    estimate's required (margin-inflated) total. Pure — no GPU access, so this
    is the function unit tests exercise directly with a synthetic
    ``free_bytes``.

    Args:
        free_bytes: Bytes actually free on the target device right now.
        device_name: For the error message only.
        total_bytes: The device's total memory, recorded for context.
        estimate: A prebuilt :class:`Fp32MemoryEstimate`, or ``None`` to build
            one from ``estimate_kwargs`` via
            :func:`estimate_fp32_inference_memory`.

    Raises:
        InsufficientGPUMemoryError: If ``free_bytes`` is short of the
            required, margin-inflated total. Names the shortfall in GiB.

    Returns:
        A payload recording the estimate and the pass, for the printed
        preflight block and for the run's persisted scientific config.
    """
    if estimate is None:
        estimate = estimate_fp32_inference_memory(**estimate_kwargs)
    required = estimate.required_bytes
    sufficient = int(free_bytes) >= required
    payload = {
        **estimate.to_dict(),
        "device_name": str(device_name),
        "free_bytes": int(free_bytes),
        "free_gib": int(free_bytes) / 1024**3,
        "total_bytes": None if total_bytes is None else int(total_bytes),
        "total_gib": None if total_bytes is None else int(total_bytes) / 1024**3,
        "sufficient": sufficient,
        "no_bf16_fallback": True,
    }
    if not sufficient:
        shortfall_gib = (required - int(free_bytes)) / 1024**3
        raise InsufficientGPUMemoryError(
            f"fp32 load refused before touching the GPU: "
            f"{payload['free_gib']:.1f} GiB free on "
            f"{device_name or 'this device'}, "
            f"{payload['required_gib']:.1f} GiB required "
            f"({payload['raw_total_gib']:.1f} GiB estimate x "
            f"{estimate.safety_margin:.2f} safety margin) -- short by "
            f"{shortfall_gib:.1f} GiB. This is not falling back to bf16: "
            "choose a larger GPU (an 80 GB A100) or lower the workspace "
            "estimate only if you have measured that it is actually "
            "over-conservative."
        )
    return payload


def preflight_fp32_or_refuse(
    *, device_index: int = 0, **estimate_kwargs
) -> dict:
    """The notebook-facing entry point: query the real GPU and refuse cleanly.

    Raises:
        InsufficientGPUMemoryError: See :func:`assert_fp32_memory_sufficient`.
        RuntimeError: If no CUDA device is visible at all.
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "fp32 preflight requires a CUDA GPU; none is visible. This is not "
            "a signal to fall back to CPU or to bf16 -- reconnect to a GPU "
            "runtime."
        )
    free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
    device_name = torch.cuda.get_device_properties(device_index).name
    return assert_fp32_memory_sufficient(
        free_bytes=free_bytes,
        total_bytes=total_bytes,
        device_name=device_name,
        **estimate_kwargs,
    )
