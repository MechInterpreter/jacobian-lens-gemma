# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Adapter tests against the Gemma4-shaped mock: no network, no transformers."""

import pytest
import torch

from jlens.fitting import fit
from jlens.gemma4 import (
    GEMMA4_LAYOUT,
    Gemma4LensModel,
    load_gemma4,
    resolve_revision,
    verify_architecture,
)
from jlens.hf import _find_layout

from .mock_gemma4 import MockGemma4ForConditionalGeneration, MockTokenizer


def _wrap(auto_bos: bool = False, **model_kwargs) -> Gemma4LensModel:
    return Gemma4LensModel(
        MockGemma4ForConditionalGeneration(**model_kwargs), MockTokenizer(auto_bos)
    )


def test_layout_autodetection_skips_towers():
    """Upstream auto-detection must land on model.language_model, not be
    confused by the vision/audio tower siblings."""
    found = _find_layout(MockGemma4ForConditionalGeneration())
    assert found.path == "model.language_model"
    assert found == GEMMA4_LAYOUT


def test_wrap_basics():
    model = _wrap()
    assert model.n_layers == 6
    assert model.d_model == 8
    assert model.layout.path == "model.language_model"
    assert all(not p.requires_grad for p in model._hf_model.parameters())
    assert model._logit_softcap == 30.0


@pytest.mark.parametrize("auto_bos", [False, True])
def test_encode_always_starts_with_bos(auto_bos):
    model = _wrap(auto_bos=auto_bos)
    ids = model.encode("hello world")
    assert int(ids[0, 0]) == MockTokenizer.bos_token_id
    # No double BOS when the tokenizer already prepends one.
    assert int(ids[0, 1]) != MockTokenizer.bos_token_id
    assert model.bos_prepended_by_tokenizer is auto_bos


def test_encode_respects_max_length():
    model = _wrap()
    ids = model.encode("x" * 100, max_length=10)
    assert ids.shape[1] == 10
    assert int(ids[0, 0]) == MockTokenizer.bos_token_id


def test_encode_can_disable_bos():
    model = _wrap()
    model.ensure_bos = False
    ids = model.encode("hello")
    assert int(ids[0, 0]) != MockTokenizer.bos_token_id


def test_unembed_pair_softcap_relationship():
    model = _wrap()
    residual = torch.randn(3, 8) * 50  # large enough that the cap matters
    pre, capped = model.unembed_pair(residual)
    assert pre.shape == capped.shape == (3, 32)
    torch.testing.assert_close(capped, 30.0 * torch.tanh(pre / 30.0))
    assert not torch.allclose(pre, capped)  # the cap actually bites here
    # unembed() is Gemma's actual output pathway == the capped branch.
    torch.testing.assert_close(model.unembed(residual), capped)
    # The cap is monotonic: rankings agree exactly.
    assert torch.equal(pre.argsort(-1), capped.argsort(-1))


def test_unembed_pair_no_softcap_is_identity():
    model = _wrap()
    model._logit_softcap = None
    residual = torch.randn(2, 8)
    pre, capped = model.unembed_pair(residual)
    torch.testing.assert_close(pre, capped)


def test_verify_architecture_happy_path():
    model = _wrap(auto_bos=False)
    report = verify_architecture(
        model, expect_n_layers=6, expect_d_model=8, expect_vocab_size=32
    )
    assert report.model_class == "MockGemma4ForConditionalGeneration"
    assert report.layout_path == "model.language_model"
    assert report.dense and report.params_frozen and report.tied_unembedding
    assert report.vocab_size == 32
    assert report.final_logit_softcapping == 30.0
    assert report.layer_scalars == [1.0] * 6 and report.layer_scalars_all_unit
    assert report.bos_token_id == 2
    assert report.bos_prepended_by_tokenizer is False
    assert report.encode_starts_with_bos is True
    assert report.num_kv_shared_layers == 2
    assert report.n_full_attention_layers == 1
    assert report.warnings == []
    assert report.to_dict()["d_model"] == 8


def test_verify_architecture_records_nonunit_layer_scalar():
    """Adjustment 4: a non-unit layer_scalar is recorded, never fatal."""
    model = _wrap()
    with torch.no_grad():
        model.layers[3].layer_scalar.fill_(1.5)
    report = verify_architecture(model)
    assert report.layer_scalars[3] == pytest.approx(1.5)
    assert not report.layer_scalars_all_unit
    assert any("layer_scalar" in w for w in report.warnings)


def test_verify_architecture_untied_is_warning_not_error():
    hf = MockGemma4ForConditionalGeneration()
    hf.lm_head.weight = torch.nn.Parameter(hf.lm_head.weight.detach().clone())
    report = verify_architecture(Gemma4LensModel(hf, MockTokenizer()))
    assert not report.tied_unembedding
    assert any("tied" in w for w in report.warnings)


def test_verify_architecture_moe_raises():
    model = _wrap()
    model._hf_model.config.text_config.enable_moe_block = True
    with pytest.raises(ValueError, match="MoE"):
        verify_architecture(model)


def test_verify_architecture_expectation_mismatch_raises():
    model = _wrap()
    with pytest.raises(ValueError, match="hidden_size=16"):
        verify_architecture(model, expect_d_model=16)


def test_verify_architecture_trainable_param_raises():
    model = _wrap()
    next(model._hf_model.parameters()).requires_grad_(True)
    with pytest.raises(ValueError, match="require grad"):
        verify_architecture(model)


def test_fit_and_apply_through_adapter(tmp_path):
    """End-to-end: fit on the Gemma4-shaped mock, save/load, apply, and check
    both readout paths. Also pins Jacobian orientation on this architecture:
    with last block (h + W h) * s, J at the penultimate site is s*(I + W)."""
    model = _wrap()
    with torch.no_grad():
        model.layers[5].layer_scalar.fill_(1.25)
    prompts = ["abcdefghij klmnopqrst " * 3, "the quick brown fox jumps " * 2]
    lens = fit(model, prompts, source_layers=[1, 3, 4], dim_batch=4, max_seq_len=48)
    assert lens.source_layers == [1, 3, 4]
    for J in lens.jacobians.values():
        assert J.shape == (8, 8) and torch.isfinite(J).all()

    expected_J4 = 1.25 * (
        torch.eye(8) + model.layers[5].linear.weight.detach().float()
    )
    torch.testing.assert_close(lens.jacobians[4], expected_J4, rtol=0, atol=1e-5)

    path = tmp_path / "lens.pt"
    lens.save(str(path))
    from jlens.lens import JacobianLens

    reloaded = JacobianLens.load(str(path))
    lens_logits, model_logits, input_ids = reloaded.apply(
        model, "hello world this is a longer prompt " * 2, layers=[1, 4]
    )
    assert set(lens_logits) == {1, 4}
    assert model_logits.shape[-1] == 32
    assert all(torch.isfinite(v).all() for v in lens_logits.values())

    # Dual readout on the final residual: capped branch equals apply()'s
    # unembed path up to fp32 casting.
    residual = torch.randn(2, 8)
    pre, capped = model.unembed_pair(reloaded.transport(residual, 4))
    assert pre.shape == capped.shape == (2, 32)


def test_apply_dual_matches_single_path_apply(tmp_path):
    """apply_dual's capped branch must equal what plain apply() (Gemma's real
    output pathway) produces, and its pre branch must differ when the cap
    bites; the softcap must be restored afterwards."""
    from jlens.gemma4 import apply_dual

    model = _wrap()
    prompts = ["abcdefghij klmnopqrst " * 3, "the quick brown fox jumps " * 2]
    lens = fit(model, prompts, source_layers=[1, 4], dim_batch=4, max_seq_len=48)

    prompt = "hello world this is a longer prompt " * 2
    lens_dual, model_dual, ids_dual = apply_dual(lens, model, prompt, layers=[1, 4])
    assert model._logit_softcap == 30.0  # restored
    lens_single, model_single, ids_single = lens.apply(model, prompt, layers=[1, 4])
    assert torch.equal(ids_dual, ids_single)
    torch.testing.assert_close(model_dual["capped"], model_single)
    for layer in (1, 4):
        torch.testing.assert_close(lens_dual[layer]["capped"], lens_single[layer])
        # Monotone cap: identical rankings pre/capped.
        assert torch.equal(
            lens_dual[layer]["pre"].argsort(-1), lens_dual[layer]["capped"].argsort(-1)
        )


def test_load_gemma4_requires_explicit_flag():
    with pytest.raises(RuntimeError, match="allow_model_load"):
        load_gemma4(allow_model_load=False)


def test_resolve_revision_validates_sha(monkeypatch):
    import huggingface_hub

    class FakeApi:
        def __init__(self, token=None):
            pass

        def model_info(self, repo_id, revision=None):
            from types import SimpleNamespace

            return SimpleNamespace(sha="a" * 40)

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    assert resolve_revision("org/model") == "a" * 40
    # A pinned full SHA that resolves elsewhere must refuse.
    with pytest.raises(RuntimeError, match="different"):
        resolve_revision("org/model", "b" * 40)


def test_resolve_revision_missing_sha_raises(monkeypatch):
    import huggingface_hub

    class FakeApi:
        def __init__(self, token=None):
            pass

        def model_info(self, repo_id, revision=None):
            from types import SimpleNamespace

            return SimpleNamespace(sha=None)

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    with pytest.raises(RuntimeError, match="immutable"):
        resolve_revision("org/model")
