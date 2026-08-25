# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Tests for the fp32-load memory preflight. No GPU required.

:func:`assert_fp32_memory_sufficient` is pure — it takes ``free_bytes``
directly rather than querying a device — so every case here runs on CPU-only
CI and still exercises the real refusal path a GPU session would hit.
"""

from __future__ import annotations

import pytest

from jlens.mmpilot.fp32_preflight import (
    GEMMA4_E4B_APPROX_PARAM_COUNT,
    GEMMA4_E4B_BF16_CHECKPOINT_BYTES_REFERENCE,
    Fp32MemoryEstimate,
    InsufficientGPUMemoryError,
    assert_fp32_memory_sufficient,
    estimate_fp32_inference_memory,
)


def test_the_reference_checkpoint_size_is_16_gib() -> None:
    assert GEMMA4_E4B_BF16_CHECKPOINT_BYTES_REFERENCE == 16 * 1024**3


def test_param_count_is_derived_from_the_bf16_reference_at_2_bytes() -> None:
    assert (
        GEMMA4_E4B_APPROX_PARAM_COUNT * 2
        == GEMMA4_E4B_BF16_CHECKPOINT_BYTES_REFERENCE
    )


def test_fp32_weight_bytes_are_exactly_four_times_param_count() -> None:
    estimate = estimate_fp32_inference_memory(
        n_params=1_000_000_000, workspace_fraction=0.0, cuda_context_bytes=0,
    )
    assert estimate.weight_bytes == 4_000_000_000
    assert estimate.workspace_bytes == 0
    assert estimate.raw_total_bytes == 4_000_000_000


def test_workspace_fraction_scales_off_weight_bytes() -> None:
    estimate = estimate_fp32_inference_memory(
        n_params=1_000_000_000, workspace_fraction=0.5, cuda_context_bytes=0,
    )
    assert estimate.workspace_bytes == 2_000_000_000
    assert estimate.raw_total_bytes == 6_000_000_000


def test_the_default_estimate_uses_the_gemma4_e4b_param_count() -> None:
    estimate = estimate_fp32_inference_memory()
    assert estimate.n_params == GEMMA4_E4B_APPROX_PARAM_COUNT
    # fp32 weights should be roughly 2x the ~16 GiB bf16 checkpoint
    assert estimate.weight_bytes == pytest.approx(32 * 1024**3, rel=0.01)


@pytest.mark.parametrize("n_params", [0, -5])
def test_nonpositive_param_count_is_rejected(n_params: int) -> None:
    with pytest.raises(ValueError):
        estimate_fp32_inference_memory(n_params=n_params)


def test_negative_workspace_fraction_is_rejected() -> None:
    with pytest.raises(ValueError):
        estimate_fp32_inference_memory(n_params=1000, workspace_fraction=-0.1)


def test_a_safety_margin_below_one_is_rejected() -> None:
    with pytest.raises(ValueError):
        estimate_fp32_inference_memory(n_params=1000, safety_margin=0.9)


# -------------------------------------------------------------- the refusal


def test_ample_free_memory_passes_and_reports_the_estimate() -> None:
    result = assert_fp32_memory_sufficient(
        free_bytes=200 * 1024**3, device_name="A100-80GB",
        total_bytes=80 * 1024**3,
    )
    assert result["sufficient"] is True
    assert result["no_bf16_fallback"] is True
    assert result["device_name"] == "A100-80GB"
    assert result["required_bytes"] < result["free_bytes"]


def test_insufficient_memory_raises_with_the_shortfall_named() -> None:
    tiny = estimate_fp32_inference_memory(
        n_params=1_000_000_000, workspace_fraction=0.1, cuda_context_bytes=0,
        safety_margin=1.0,
    )
    # required = 4.4 GB; give it far less
    with pytest.raises(InsufficientGPUMemoryError) as excinfo:
        assert_fp32_memory_sufficient(
            free_bytes=1 * 1024**3, device_name="Tesla T4", estimate=tiny,
        )
    message = str(excinfo.value)
    assert "T4" in message
    assert "GiB" in message
    assert "short by" in message
    assert "bf16" in message  # states explicitly that this is not a fallback


def test_the_gate_is_at_required_not_raw_total() -> None:
    """The safety margin must actually gate, not just annotate the estimate."""
    estimate = estimate_fp32_inference_memory(
        n_params=1_000_000_000, workspace_fraction=0.0, cuda_context_bytes=0,
        safety_margin=2.0,
    )
    assert estimate.raw_total_bytes == 4_000_000_000
    assert estimate.required_bytes == 8_000_000_000
    # exactly the raw total is NOT enough once margin > 1
    with pytest.raises(InsufficientGPUMemoryError):
        assert_fp32_memory_sufficient(
            free_bytes=estimate.raw_total_bytes, estimate=estimate,
        )
    # but the required (margin-inflated) amount is
    result = assert_fp32_memory_sufficient(
        free_bytes=estimate.required_bytes, estimate=estimate,
    )
    assert result["sufficient"] is True


def test_a_realistic_40gb_a100_fails_the_default_gemma4_e4b_estimate() -> None:
    """~32 GiB fp32 weights * 1.3 workspace * 1.15 margin exceeds 40 GB.

    This is a real, checkable consequence of the default estimate, not an
    assertion about the actual model -- it documents why an 80 GB card is the
    practical recommendation for this study.
    """
    with pytest.raises(InsufficientGPUMemoryError):
        assert_fp32_memory_sufficient(
            free_bytes=40 * 1024**3, device_name="A100-40GB",
            total_bytes=40 * 1024**3,
        )


def test_an_80gb_a100_passes_the_default_gemma4_e4b_estimate() -> None:
    result = assert_fp32_memory_sufficient(
        free_bytes=79 * 1024**3, device_name="A100-80GB",
        total_bytes=80 * 1024**3,
    )
    assert result["sufficient"] is True


def test_estimate_object_round_trips_through_to_dict() -> None:
    estimate = Fp32MemoryEstimate(
        n_params=10, weight_bytes=40, workspace_bytes=8,
        cuda_context_bytes=2, safety_margin=1.1,
    )
    payload = estimate.to_dict()
    assert payload["raw_total_bytes"] == 50
    assert payload["required_bytes"] == 55
    assert payload["version"]


# --------------------------------------------------------- the GPU-facing entry


def test_preflight_refuses_with_no_cuda_visible(monkeypatch) -> None:
    import torch

    from jlens.mmpilot import fp32_preflight

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="none is visible"):
        fp32_preflight.preflight_fp32_or_refuse()


def test_preflight_queries_mem_get_info_and_gates_on_it(monkeypatch) -> None:
    import torch

    from jlens.mmpilot import fp32_preflight

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda index: (5 * 1024**3, 80 * 1024**3)
    )

    class _Props:
        name = "A100-80GB"

    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda index: _Props()
    )
    with pytest.raises(InsufficientGPUMemoryError, match="A100-80GB"):
        fp32_preflight.preflight_fp32_or_refuse()
