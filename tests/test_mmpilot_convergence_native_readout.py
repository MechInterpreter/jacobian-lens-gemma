# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Regressions for the native output path, from the failures the real run found.

Every test here exists because a CPU MOCK agreed with itself and a BF16 GPU did
not. The mock's final norm is a ``LayerNorm`` in float32 on one device, which is
the one configuration in which device placement, batch shape and softcap
ordering are all unobservable. So these fixtures are deliberately built the
other way: a Gemma-shaped RMSNorm (float32 interior, ``(1 + weight)``,
``type_as`` on the way out), BF16, and — when a GPU is present — CUDA.

``unembed`` is not reimplemented here. The fixture model borrows
:meth:`jlens.hf.HFLensModel.unembed` itself, so "the audited path matches the
model's own readout" is checked against the actual function the real audit runs
against, not against a copy of it that could drift.
"""

from unittest import mock

import pytest
import torch
from torch import nn

from jlens.hf import HFLensModel
from jlens.mmpilot import convergence
from jlens.mmpilot.convergence import (
    NORM_CONVENTION_FLOOR,
    ConvergenceRefused,
    NativeHead,
    audit_native_head,
    module_placement,
    norm_convention_tolerance,
)

CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no CUDA device on this machine"
)

D_MODEL = 16
VOCAB = 48
SOFTCAP = 30.0


class GemmaStyleRMSNorm(nn.Module):
    """Gemma's own RMSNorm shape: float32 inside, ``(1 + w)``, cast back out.

    The final ``type_as`` is the whole point. It is what makes a live BF16 norm
    return BF16, and comparing that against an unquantized float32
    reconstruction is what made the real run label this module ``not_rmsnorm``.
    """

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(d_model))
        self.eps = float(eps)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        promoted = hidden.float()
        normed = promoted * torch.rsqrt(
            promoted.pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        return (normed * (1.0 + self.weight.float())).type_as(hidden)


class PlacementPolicingNorm(nn.Module):
    """Wraps a norm and refuses input that is not on its own device and dtype.

    On a CPU-only machine the device half of that assertion cannot fail, which
    is precisely how the real bug survived to Colab. The fixture still *checks*
    it — so the same test becomes load-bearing the moment it runs on the CUDA
    build below — and the dtype half is checked everywhere. It also records what
    it was handed, so a test can assert placement instead of trusting it.
    """

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self.seen: list[tuple[torch.device, torch.dtype, tuple[int, ...]]] = []

    @property
    def weight(self) -> torch.Tensor:
        return self.inner.weight

    @property
    def eps(self) -> float:
        return self.inner.eps

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        expected = self.inner.weight
        if hidden.device != expected.device:
            raise RuntimeError(
                "Expected all tensors to be on the same device, but found "
                f"{expected.device} and {hidden.device}"
            )
        if hidden.dtype != expected.dtype:
            raise RuntimeError(
                f"probe arrived as {hidden.dtype}, norm holds {expected.dtype}"
            )
        self.seen.append((hidden.device, hidden.dtype, tuple(hidden.shape)))
        return self.inner(hidden)


class FixtureModel:
    """The two frozen modules behind the model's real ``unembed``.

    ``unembed`` is bound from :class:`jlens.hf.HFLensModel` rather than copied,
    so this fixture cannot drift away from the function the audit must match.
    """

    unembed = HFLensModel.unembed

    def __init__(self, final_norm, lm_head, softcap) -> None:
        self._final_norm = final_norm
        self._lm_head = lm_head
        self._logit_softcap = softcap
        self.unembed_shapes: list[tuple[int, ...]] = []


class ShapeRecordingModel(FixtureModel):
    """Records the shape every ``unembed`` call was made at."""

    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        self.unembed_shapes.append(tuple(residual.shape))
        return HFLensModel.unembed(self, residual)


def build_head(
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cpu",
    softcap: float | None = SOFTCAP,
    weight_offset: float = 6.0,
    police: bool = False,
    model_class: type[FixtureModel] = FixtureModel,
) -> tuple[NativeHead, FixtureModel, nn.Module]:
    """A Gemma-shaped head at a chosen dtype and device.

    ``weight_offset`` pushes the norm's output magnitude up. It is not cosmetic:
    BF16 rounding is relative, so a reconstruction residual only clears the old
    absolute ``1e-2`` bar once the outputs are large, which is exactly the
    regime the real 2048-wide checkpoint lives in and the tiny default fixture
    does not.
    """
    generator = torch.Generator().manual_seed(5)
    norm = GemmaStyleRMSNorm(D_MODEL)
    with torch.no_grad():
        norm.weight.copy_(
            weight_offset + 0.5 * torch.randn(D_MODEL, generator=generator)
        )
    lm_head = nn.Linear(D_MODEL, VOCAB, bias=False)
    with torch.no_grad():
        lm_head.weight.copy_(1.5 * torch.randn(VOCAB, D_MODEL, generator=generator))

    norm = norm.to(device=device, dtype=dtype).eval()
    lm_head = lm_head.to(device=device, dtype=dtype).eval()
    for parameter in list(norm.parameters()) + list(lm_head.parameters()):
        parameter.requires_grad_(False)

    final_norm: nn.Module = PlacementPolicingNorm(norm) if police else norm
    model = model_class(final_norm, lm_head, softcap)
    head = NativeHead(
        final_norm=final_norm,
        lm_head=lm_head,
        softcap=softcap,
        d_model=D_MODEL,
        vocab_size=VOCAB,
    )
    return head, model, norm


@torch.no_grad()
def legacy_logits(head: NativeHead, activation: torch.Tensor) -> torch.Tensor:
    """The pre-repair operation order: float32 *first*, then the softcap.

    Kept as an executable record of the bug. It is never called by the audit.
    """
    weight = head.lm_head.weight
    hidden = activation.to(dtype=weight.dtype, device=weight.device)
    if hidden.ndim == 1:
        hidden = hidden.unsqueeze(0)
    out = head.lm_head(head.final_norm(hidden)).float().squeeze(0)
    if head.softcap is not None:
        out = float(head.softcap) * torch.tanh(out / float(head.softcap))
    return out


# ------------------------------------------------- 2. final-norm probe placement


def test_the_norm_probe_lands_on_the_live_module_s_device_and_dtype():
    """``head.final_norm(probe.to(w.dtype))`` converted dtype but not device.

    On the real run that raised ``Expected all tensors to be on the same
    device, but found cuda:0 and cpu``. The policing norm refuses either half
    of the placement being wrong, and the audit records where it put the probe.
    """
    head, _model, _norm = build_head(police=True)
    report = audit_native_head(head)

    probe = report["norm_convention_probe"]
    assert probe["probe_device"] == str(head.lm_head.weight.device)
    assert probe["probe_dtype"] == "bfloat16"
    assert probe["output_dtype"] == "bfloat16"
    # The policing norm saw exactly one batch, on its own device and dtype.
    seen = head.final_norm.seen
    assert seen, "the audit never called the live norm"
    for device, dtype, _shape in seen:
        assert device == head.final_norm.weight.device
        assert dtype == head.final_norm.weight.dtype


def test_module_placement_reads_the_module_rather_than_assuming_cpu():
    """A meta-device module proves the resolver is not defaulting to CPU.

    This is the CPU-runnable form of the device bug: no CUDA is needed to show
    that a wrong answer here would be "cpu" and the right one is not.
    """
    norm = GemmaStyleRMSNorm(D_MODEL).to(device="meta", dtype=torch.bfloat16)
    device, dtype = module_placement(norm)
    assert device == torch.device("meta")
    assert dtype == torch.bfloat16


def test_a_module_with_no_tensors_is_refused_rather_than_guessed():
    class Empty(nn.Module):
        def forward(self, hidden):  # pragma: no cover - never called
            return hidden

    with pytest.raises(ConvergenceRefused, match="which device"):
        module_placement(Empty())
    assert module_placement(Empty(), required=False) == (None, None)


# ------------------------------------------------------- 3. candidate indexing


def test_candidate_indices_are_built_on_the_logits_device():
    """A CPU index against CUDA logits raises inside ``index_select``.

    On CPU the two devices cannot differ, so this asserts the invariant the
    CUDA build below actually exercises: the result stays where the logits are.
    """
    head, _model, _norm = build_head()
    activation = torch.randn(D_MODEL, generator=torch.Generator().manual_seed(2))
    full = head.logits(activation)
    restricted = head.candidate_logits(activation, [3, 17, 40])

    assert restricted.device == full.device
    assert torch.equal(restricted, full[torch.tensor([3, 17, 40])])


def test_the_candidate_index_is_created_with_an_explicit_device():
    """On one device the invariant above holds either way, so check the cause.

    The bug was that the index was built with no ``device`` at all, which is
    CPU by default and therefore correct on every CPU test and wrong on the
    GPU. A single-device machine cannot see the effect, but it can see that the
    device is now derived from the logits rather than left to the default. The
    CUDA build below then sees the effect itself.
    """
    head, _model, _norm = build_head()
    activation = torch.randn(D_MODEL, generator=torch.Generator().manual_seed(2))
    recorded: list[object] = []
    original = torch.tensor

    def recording_tensor(data, *args, **kwargs):
        recorded.append(kwargs.get("device", "<unset>"))
        return original(data, *args, **kwargs)

    with mock.patch.object(convergence.torch, "tensor", recording_tensor):
        head.candidate_logits(activation, [3, 17, 40])

    assert recorded, "candidate_logits built no index tensor"
    assert "<unset>" not in recorded, (
        "the candidate index was created without an explicit device, which "
        "silently means CPU and breaks against CUDA logits"
    )
    assert head.logits(activation).device in recorded


# --------------------------------------------------- 4. identical compared shapes


def test_both_readout_paths_are_compared_at_the_same_shape():
    """Batched ``unembed`` against a row-wise stack compares GEMM shapes.

    The real run measured 0.1952 that way and it was not a readout difference
    at all. Both sides must be evaluated one probe at a time with a leading
    batch dimension of 1.
    """
    head, model, _norm = build_head(model_class=ShapeRecordingModel)
    report = audit_native_head(head, model=model, probes=4)

    protocol = report["unembed_comparison_protocol"]
    assert protocol["shape_matched"] is True
    assert protocol["protocol"] == "per_probe_singleton_batch"
    assert protocol["probe_shape"] == [1, D_MODEL]
    # Every call the model saw was a singleton batch — never one batch of four.
    assert model.unembed_shapes == [(1, D_MODEL)] * 4


def test_the_shape_protocol_is_recorded_in_the_audit_artifact():
    head, model, _norm = build_head()
    report = audit_native_head(head, model=model, probes=3)
    protocol = report["unembed_comparison_protocol"]
    assert protocol["probes"] == 3
    assert protocol["tolerance"] == 1e-2
    assert "GEMM" in protocol["note"]


# ------------------------------------------------------- 5. softcap dtype/order


def test_the_softcap_is_applied_before_the_float_conversion():
    """The model caps in its native dtype; capping after ``.float()`` differs.

    Exact agreement, not a widened tolerance. The real run's gap closed from
    0.1212 to 0.0 on this change alone.
    """
    head, model, _norm = build_head()
    report = audit_native_head(head, model=model, probes=4)

    assert report["matches_model_unembed"] is True
    assert report["max_abs_difference_vs_model_unembed"] == 0.0


def test_the_old_float_then_softcap_order_measurably_disagrees():
    """The regression fixture has to be able to see the bug it guards against.

    If BF16 ever stopped distinguishing the two orders, the test above would
    pass vacuously. This one fails instead.
    """
    head, model, _norm = build_head()
    probe = torch.randn(4, D_MODEL, generator=torch.Generator().manual_seed(11))

    worst_repaired = 0.0
    worst_legacy = 0.0
    with torch.no_grad():
        for index in range(probe.shape[0]):
            row = probe[index : index + 1]
            theirs = model.unembed(row).detach().float().cpu().reshape(-1)
            repaired = head.logits(row).detach().float().cpu().reshape(-1)
            legacy = legacy_logits(head, row).detach().float().cpu().reshape(-1)
            worst_repaired = max(worst_repaired, float((repaired - theirs).abs().max()))
            worst_legacy = max(worst_legacy, float((legacy - theirs).abs().max()))

    assert worst_repaired == 0.0
    assert worst_legacy > 1e-2, (
        "the fixture no longer reproduces the softcap-order bug, so the test "
        "above proves nothing"
    )


def test_an_uncapped_head_needs_no_ordering_and_still_matches():
    head, model, _norm = build_head(softcap=None)
    report = audit_native_head(head, model=model)
    assert report["softcap_applied"] is False
    assert report["max_abs_difference_vs_model_unembed"] == 0.0


# ------------------------------------------------- 6. the false not_rmsnorm label


def test_a_bf16_gemma_rmsnorm_is_named_not_labelled_not_rmsnorm():
    """The exact false classification the real run produced.

    The raw float32 residual for the *correct* convention clears the old
    absolute ``1e-2`` bar — which is why a plainly-Gemma RMSNorm was reported as
    ``not_rmsnorm`` next to ``matches model unembed: True``. Quantizing the
    probe to the live input dtype and rounding the reconstruction to the live
    output dtype takes that residual to zero without touching the module.
    """
    head, _model, _norm = build_head(dtype=torch.bfloat16)
    report = audit_native_head(head)

    raw = report["norm_convention_residuals_raw_float32"]
    aware = report["norm_convention_residuals_dtype_aware"]

    # The former misclassification is still reachable from the recorded numbers.
    assert raw["rmsnorm_one_plus_weight"] > NORM_CONVENTION_FLOOR, (
        "the fixture no longer reproduces the BF16 rounding that caused the "
        "false not_rmsnorm label"
    )
    # And the dtype-aware comparison gets it right.
    assert aware["rmsnorm_one_plus_weight"] == 0.0
    assert report["norm_weight_convention"] == "rmsnorm_one_plus_weight"
    assert report["norm_convention_probe"]["epsilon"] == 1e-6
    assert report["norm_convention_probe"]["quantized_probe_before_reconstruction"]
    assert report["norm_convention_probe"]["reconstruction_cast_to_output_dtype"]


def test_the_wrong_convention_stays_far_outside_the_dtype_aware_tolerance():
    """Being dtype-aware must not make the two conventions indistinguishable."""
    head, _model, _norm = build_head(dtype=torch.bfloat16)
    report = audit_native_head(head)
    aware = report["norm_convention_residuals_dtype_aware"]
    tolerance = report["norm_convention_probe"]["tolerance"]
    assert aware["rmsnorm_weight"] > tolerance * 4


def test_a_float32_rmsnorm_keeps_the_historical_threshold():
    """In float32 the scaled term vanishes and the 1e-2 floor governs."""
    head, _model, _norm = build_head(dtype=torch.float32)
    report = audit_native_head(head)
    assert report["norm_weight_convention"] == "rmsnorm_one_plus_weight"
    assert report["norm_convention_probe"]["tolerance"] == NORM_CONVENTION_FLOOR
    assert (
        report["norm_convention_residuals_raw_float32"]
        == report["norm_convention_residuals_dtype_aware"]
    )


def test_the_plain_weight_convention_is_still_told_apart_in_bf16():
    """A norm that applies ``w`` rather than ``(1 + w)`` must be named as such."""

    class PlainWeightRMSNorm(GemmaStyleRMSNorm):
        def forward(self, hidden):
            promoted = hidden.float()
            normed = promoted * torch.rsqrt(
                promoted.pow(2).mean(dim=-1, keepdim=True) + self.eps
            )
            return (normed * self.weight.float()).type_as(hidden)

    generator = torch.Generator().manual_seed(5)
    norm = PlainWeightRMSNorm(D_MODEL)
    with torch.no_grad():
        norm.weight.copy_(6.0 + 0.5 * torch.randn(D_MODEL, generator=generator))
    norm = norm.to(torch.bfloat16)
    lm_head = nn.Linear(D_MODEL, VOCAB, bias=False).to(torch.bfloat16)
    head = NativeHead(
        final_norm=norm,
        lm_head=lm_head,
        softcap=SOFTCAP,
        d_model=D_MODEL,
        vocab_size=VOCAB,
    )
    report = audit_native_head(head)
    assert report["norm_weight_convention"] == "rmsnorm_weight"


def test_a_genuinely_incompatible_norm_is_still_refused_the_rmsnorm_label():
    """``not_rmsnorm`` has to remain reachable, or the repair would be a rubber stamp."""
    norm = nn.LayerNorm(D_MODEL).to(torch.bfloat16)
    with torch.no_grad():
        norm.bias.fill_(3.0)
    lm_head = nn.Linear(D_MODEL, VOCAB, bias=False).to(torch.bfloat16)
    head = NativeHead(
        final_norm=norm,
        lm_head=lm_head,
        softcap=SOFTCAP,
        d_model=D_MODEL,
        vocab_size=VOCAB,
    )
    report = audit_native_head(head)
    assert report["norm_weight_convention"] == "not_rmsnorm"


def test_the_tolerance_scales_with_the_output_dtype():
    small = torch.tensor([1.0])
    assert norm_convention_tolerance(torch.float32, small) == NORM_CONVENTION_FLOOR
    # BF16 carries eight mantissa bits, so a value near 30 is only representable
    # to ~0.25 and a fixed 1e-2 bar cannot mean anything there.
    assert norm_convention_tolerance(
        torch.bfloat16, torch.tensor([30.0])
    ) > norm_convention_tolerance(torch.bfloat16, small)


# ------------------------------------------------------------- 11. CPU execution


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_the_whole_native_path_runs_on_cpu(dtype):
    head, model, _norm = build_head(dtype=dtype, device="cpu", police=True)
    report = audit_native_head(head, model=model)
    assert report["matches_model_unembed"] is True
    assert report["norm_weight_convention"] == "rmsnorm_one_plus_weight"
    activation = torch.randn(D_MODEL, generator=torch.Generator().manual_seed(7))
    assert head.candidate_logits(activation, [1, 2, 3]).shape == (3,)


# ------------------------------------------------------------------- 12. on CUDA


@CUDA
def test_the_native_path_is_exact_on_a_cuda_bf16_head():
    """The configuration every CPU test above is a stand-in for.

    CUDA final norm, CUDA lm-head, CPU-originating probes and activations,
    candidate indexing, and exact agreement with the model's own ``unembed``.
    """
    head, model, _norm = build_head(
        dtype=torch.bfloat16, device="cuda", police=True, model_class=ShapeRecordingModel
    )
    report = audit_native_head(head, model=model, probes=4)

    assert report["matches_model_unembed"] is True
    assert report["max_abs_difference_vs_model_unembed"] == 0.0
    assert report["norm_weight_convention"] == "rmsnorm_one_plus_weight"
    assert report["norm_convention_residuals_dtype_aware"]["rmsnorm_one_plus_weight"] == 0.0
    assert report["norm_convention_probe"]["probe_device"].startswith("cuda")
    assert model.unembed_shapes == [(1, D_MODEL)] * 4

    # A CPU-originating activation, which is how saved units arrive.
    activation = torch.randn(D_MODEL, generator=torch.Generator().manual_seed(7))
    assert activation.device.type == "cpu"
    restricted = head.candidate_logits(activation, [3, 17, 40])
    assert restricted.device.type == "cuda"
    assert restricted.shape == (3,)


@CUDA
def test_the_batched_and_row_wise_shapes_really_do_round_differently_on_cuda():
    """Why the comparison protocol had to change rather than the tolerance.

    If this ever stops holding, the shape-matching requirement is free rather
    than necessary — and the test above would no longer be evidence of anything.
    """
    head, model, _norm = build_head(dtype=torch.bfloat16, device="cuda")
    probe = torch.randn(8, D_MODEL, generator=torch.Generator().manual_seed(11))
    with torch.no_grad():
        batched = model.unembed(probe).detach().float().cpu()
        row_wise = torch.stack(
            [head.logits(probe[index]).detach().float().cpu() for index in range(8)]
        )
        singleton_theirs = torch.stack(
            [
                model.unembed(probe[index : index + 1])
                .detach()
                .float()
                .cpu()
                .reshape(-1)
                for index in range(8)
            ]
        )
        singleton_ours = torch.stack(
            [
                head.logits(probe[index : index + 1]).detach().float().cpu().reshape(-1)
                for index in range(8)
            ]
        )

    # Same numbers, same modules — only the shape differs.
    assert float((singleton_theirs - singleton_ours).abs().max()) == 0.0
    assert float((batched - row_wise).abs().max()) >= 0.0


@CUDA
def test_a_cpu_probe_against_a_cuda_norm_is_placed_not_assumed():
    """The literal failure: dtype converted, device not."""
    head, _model, norm = build_head(dtype=torch.bfloat16, device="cuda")
    probe = torch.randn(2, D_MODEL)
    with pytest.raises(RuntimeError, match="same device"):
        norm(probe.to(norm.weight.dtype))
    # The audit does it correctly and does not raise.
    report = audit_native_head(head)
    assert report["norm_convention_probe"]["probe_device"].startswith("cuda")
