# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Tests for jlens.interventions: hook semantics, delta construction,
determinism, record store. CPU only, no model, no network."""

import math

import pytest
import torch
from torch import nn

from jlens.interventions import (
    InterventionError,
    _derived_seed,
    append_record,
    assert_run_resumable,
    completed_condition_ids,
    condition_id,
    cone_delta,
    isotropic_random_direction,
    load_records,
    logit_metrics,
    make_intervention_record,
    parity_report,
    residual_intervention,
)

D = 8


class TensorBlock(nn.Module):
    """Block returning a bare tensor (passes input through a frozen linear)."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(D, D)
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, hidden):
        return hidden + self.linear(hidden)


class TupleBlock(TensorBlock):
    """Block returning (hidden, extra) like some HF decoder layers."""

    def forward(self, hidden):
        return super().forward(hidden), {"cache": "untouched"}


def run_blocks(blocks, hidden):
    for block in blocks:
        out = block(hidden)
        hidden = out if torch.is_tensor(out) else out[0]
    return hidden


@pytest.fixture
def blocks():
    torch.manual_seed(0)
    return nn.ModuleList([TensorBlock(), TupleBlock(), TensorBlock()])


@pytest.fixture
def hidden():
    torch.manual_seed(1)
    return torch.randn(2, 5, D)


def test_exact_position_editing_tensor_output(blocks, hidden):
    delta = torch.arange(D, dtype=torch.float32)
    base_block0 = blocks[0](hidden)
    with residual_intervention(blocks, 0, position=2, delta=delta, multiplier=1.5,
                               batch_row=1):
        edited_block0_downstream = run_blocks(blocks, hidden)
        # Re-run block 0 alone inside the hook to inspect its output.
        edited = blocks[0](hidden)
    expected = base_block0.clone()
    expected[1, 2] = (base_block0[1, 2].float() + 1.5 * delta).to(base_block0.dtype)
    assert torch.equal(edited, expected)
    # Only the (1, 2) slot differs.
    mask = torch.ones_like(base_block0, dtype=torch.bool)
    mask[1, 2] = False
    assert torch.equal(edited[mask], base_block0[mask])
    assert edited_block0_downstream.shape == hidden.shape


def test_negative_position_resolves_against_seq_len(blocks, hidden):
    delta = torch.ones(D)
    base = blocks[0](hidden)
    with residual_intervention(blocks, 0, position=-1, delta=delta) as stats:
        edited = blocks[0](hidden)
    assert stats["resolved_position"] == hidden.shape[1] - 1
    assert not torch.equal(edited[0, -1], base[0, -1])
    assert torch.equal(edited[0, :-1], base[0, :-1])


def test_tuple_output_preservation(blocks, hidden):
    delta = torch.ones(D)
    base_hidden, base_extra = blocks[1](hidden)
    with residual_intervention(blocks, 1, position=0, delta=delta):
        out = blocks[1](hidden)
    assert isinstance(out, tuple) and len(out) == 2
    assert out[1] is base_extra or out[1] == base_extra  # untouched element
    assert not torch.equal(out[0][0, 0], base_hidden[0, 0])


def test_multiplier_zero_is_exact_noop(blocks, hidden):
    base = blocks[0](hidden)
    delta = torch.randn(D)
    with residual_intervention(blocks, 0, position=1, delta=delta, multiplier=0.0):
        edited = blocks[0](hidden)
    assert torch.equal(edited, base)


def test_multiplier_zero_exact_on_bfloat16(hidden):
    torch.manual_seed(3)
    block = TensorBlock().to(torch.bfloat16)
    blocks = nn.ModuleList([block])
    bf16_hidden = hidden.to(torch.bfloat16)
    base = block(bf16_hidden)
    with residual_intervention(blocks, 0, position=1, delta=torch.randn(D),
                               multiplier=0.0):
        edited = block(bf16_hidden)
    assert edited.dtype == torch.bfloat16
    assert torch.equal(edited, base)


def test_writeback_mode_is_exact_noop(blocks, hidden):
    base = blocks[0](hidden)
    with residual_intervention(blocks, 0, position=3, delta=None):
        edited = blocks[0](hidden)
    assert torch.equal(edited, base)


def test_norm_preserving_rescales_to_original_norm(blocks, hidden):
    delta = 5.0 * torch.ones(D)
    base = blocks[0](hidden)
    with residual_intervention(blocks, 0, position=2, delta=delta,
                               norm_preserving=True) as stats:
        edited = blocks[0](hidden)
    original_norm = float(base[0, 2].float().norm())
    assert stats["norm_preserving"] is True
    assert math.isclose(float(edited[0, 2].float().norm()), original_norm,
                        rel_tol=1e-5)


def test_invalid_positions_rejected(blocks, hidden):
    delta = torch.ones(D)
    with residual_intervention(blocks, 0, position=99, delta=delta):
        with pytest.raises(InterventionError, match="position 99 out of range"):
            blocks[0](hidden)
    with residual_intervention(blocks, 0, position=-6, delta=delta):
        with pytest.raises(InterventionError, match="out of range"):
            blocks[0](hidden)
    with pytest.raises(InterventionError, match="layer 7 out of range"):
        with residual_intervention(blocks, 7, position=0, delta=delta):
            pass


def test_invalid_batch_row_rejected(blocks, hidden):
    with residual_intervention(blocks, 0, position=0, delta=torch.ones(D),
                               batch_row=5):
        with pytest.raises(InterventionError, match="batch_row 5 out of range"):
            blocks[0](hidden)


def test_nonfinite_delta_rejected(blocks):
    bad = torch.ones(D)
    bad[3] = float("nan")
    with pytest.raises(InterventionError, match="delta contains NaN/Inf"):
        with residual_intervention(blocks, 0, position=0, delta=bad):
            pass
    with pytest.raises(InterventionError, match="multiplier must be finite"):
        with residual_intervention(blocks, 0, position=0, delta=torch.ones(D),
                                   multiplier=float("inf")):
            pass


def test_nonfinite_output_rejected(blocks, hidden):
    # Finite delta whose scaled addition overflows float32 -> Inf output.
    delta = torch.full((D,), 3.0e38)
    with residual_intervention(blocks, 0, position=0, delta=delta, multiplier=2.0):
        with pytest.raises(InterventionError, match="NaN/Inf"):
            blocks[0](hidden)


def test_hooks_removed_on_success_and_exception(blocks, hidden):
    delta = torch.ones(D)
    assert len(blocks[0]._forward_hooks) == 0
    with residual_intervention(blocks, 0, position=0, delta=delta):
        assert len(blocks[0]._forward_hooks) == 1
        blocks[0](hidden)
    assert len(blocks[0]._forward_hooks) == 0

    # Exception raised BY the hook during forward still cleans up.
    with pytest.raises(InterventionError):
        with residual_intervention(blocks, 0, position=99, delta=delta):
            blocks[0](hidden)
    assert len(blocks[0]._forward_hooks) == 0

    # Exception raised by the body cleans up too.
    with pytest.raises(RuntimeError, match="body"):
        with residual_intervention(blocks, 0, position=0, delta=delta):
            raise RuntimeError("body")
    assert len(blocks[0]._forward_hooks) == 0


def test_frozen_parameter_check(blocks):
    blocks[0].linear.weight.requires_grad_(True)
    with pytest.raises(InterventionError, match="requires grad"):
        with residual_intervention(blocks, 0, position=0, delta=torch.ones(D)):
            pass
    blocks[0].linear.weight.requires_grad_(False)


# ------------------------------------------------------------------ deltas


def _cone_record():
    return {
        "effective_token_ids": [5, 2, 9],
        "effective_labels": [" out", " semantic", " other"],
        "effective_coefficients": [4.0, 7.0, 7.0],
    }


def test_cone_delta_full_cone_matches_reconstruction():
    torch.manual_seed(2)
    atoms = torch.randn(12, D)
    delta, info = cone_delta(atoms, _cone_record(),
                             target_kind="full_cone_reconstruction",
                             model_top1_id=5)
    expected = 4.0 * atoms[5] + 7.0 * atoms[2] + 7.0 * atoms[9]
    assert torch.allclose(delta, expected)
    assert info.atom_token_id is None
    assert math.isclose(info.delta_norm, float(expected.norm()), rel_tol=1e-6)


def test_cone_delta_output_atom():
    torch.manual_seed(2)
    atoms = torch.randn(12, D)
    delta, info = cone_delta(atoms, _cone_record(),
                             target_kind="output_atom_contribution",
                             model_top1_id=5)
    assert torch.allclose(delta, 4.0 * atoms[5])
    assert info.atom_token_id == 5 and info.coefficient == 4.0

    with pytest.raises(InterventionError, match="not in the cone"):
        cone_delta(atoms, _cone_record(),
                   target_kind="output_atom_contribution", model_top1_id=999)


def test_cone_delta_top_non_output_atom_tie_breaks_to_earliest():
    torch.manual_seed(2)
    atoms = torch.randn(12, D)
    # Coefficients 7.0 tie between token 2 (earlier) and token 9.
    delta, info = cone_delta(atoms, _cone_record(),
                             target_kind="top_non_output_atom_contribution",
                             model_top1_id=5)
    assert info.atom_token_id == 2 and info.atom_label == " semantic"
    assert torch.allclose(delta, 7.0 * atoms[2])


def test_random_direction_deterministic_and_exactly_norm_matched():
    a, info_a = isotropic_random_direction(D, match_norm=12.5, seed=1234)
    b, _ = isotropic_random_direction(D, match_norm=12.5, seed=1234)
    c, _ = isotropic_random_direction(D, match_norm=12.5, seed=1235)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
    assert math.isclose(float(a.norm()), 12.5, rel_tol=1e-6)
    assert info_a.seed == 1234
    with pytest.raises(InterventionError):
        isotropic_random_direction(D, match_norm=float("nan"), seed=1)


# --------------------------------------------------------------- metrics


def test_logit_metrics_directions_and_kl():
    before = torch.tensor([2.0, 1.0, 0.0, -1.0])
    after = torch.tensor([1.0, 3.0, 0.0, -1.0])
    metrics = logit_metrics(before, after, target_token_id=1, top_k=2)
    assert metrics["target_rank_before"] == 1
    assert metrics["target_rank_after"] == 0
    assert metrics["target_logit_delta"] == pytest.approx(2.0)
    assert metrics["top1_before"]["token_id"] == 0
    assert metrics["top1_after"]["token_id"] == 1
    assert metrics["kl_divergence_after_vs_before"] > 0
    identical = logit_metrics(before, before, target_token_id=1, top_k=2)
    assert identical["kl_divergence_after_vs_before"] == pytest.approx(0.0, abs=1e-9)
    assert identical["top10_overlap"] == 1.0


def test_parity_report_identical_and_perturbed():
    logits = torch.randn(50)
    report = parity_report(logits, logits)
    assert report["max_abs_logit_diff"] == 0.0
    assert report["top1_identical"] and report["top10_overlap"] == 1.0
    report2 = parity_report(logits, logits + 0.01)
    assert report2["max_abs_logit_diff"] == pytest.approx(0.01, rel=1e-4)


# ------------------------------------------------------------ IDs + store


def test_condition_ids_deterministic_and_distinct():
    kwargs = dict(example_id="text:x:0011223344556677", layer=35, position=-1,
                  target_kind="output_atom_contribution", multiplier=1.0,
                  atom_token_id=42)
    assert condition_id(**kwargs) == condition_id(**kwargs)
    assert condition_id(**kwargs) != condition_id(**{**kwargs, "multiplier": -1.0})
    assert condition_id(**kwargs) != condition_id(**{**kwargs, "layer": 38})
    assert condition_id(**kwargs) != condition_id(**{**kwargs, "variant": "matched:x"})
    assert condition_id(**kwargs).startswith("cond_")
    with pytest.raises(InterventionError, match="target_kind"):
        condition_id(**{**kwargs, "target_kind": "made_up"})
    seed = _derived_seed(condition_id(**kwargs))
    assert seed == _derived_seed(condition_id(**kwargs))
    assert 0 <= seed < 2**31


def test_append_safe_resumption(tmp_path):
    path = str(tmp_path / "records.jsonl")
    assert completed_condition_ids(path) == set()

    from jlens.interventions import DeltaInfo

    def record(cid):
        return make_intervention_record(
            condition=cid, example_id="text:x:0011223344556677", layer=35,
            position=-1,
            delta_info=DeltaInfo(
                target_kind="full_cone_reconstruction", atom_token_id=None,
                atom_label=None, coefficient=None, delta_norm=1.0),
            multiplier=1.0,
            stats={"activation_norm": 10.0, "resolved_position": 8,
                   "norm_preserving": False},
            metrics=None,
        )

    append_record(path, record("cond_a"))
    append_record(path, record("cond_b"))
    assert completed_condition_ids(path) == {"cond_a", "cond_b"}
    records = load_records(path)
    assert len(records) == 2
    assert records[0]["delta_to_activation_ratio"] == pytest.approx(0.1)

    with open(path, "a", encoding="utf-8") as handle:
        handle.write("{not json\n")
    with pytest.raises(InterventionError, match="malformed"):
        load_records(path)


def test_completed_run_refusal(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    assert_run_resumable(str(run_dir))  # no metadata yet: fine
    (run_dir / "run_metadata.json").write_text("{}", encoding="utf-8")
    with pytest.raises(InterventionError, match="refusing to resume"):
        assert_run_resumable(str(run_dir))
