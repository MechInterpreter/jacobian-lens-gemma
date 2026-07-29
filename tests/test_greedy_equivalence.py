# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the greedy-equivalence root cause.

The real-Gemma smoke run failed because manual uncached greedy decoding
disagreed with ``generate()`` on the very first token even with
``do_sample=False, num_beams=1, use_cache=False`` and identical inputs. The
cause was **not** sampling, masks, position ids, token indexing, or logits
processors: it was a **double final norm**.

HuggingFace text models apply the final norm themselves before returning
``last_hidden_state`` (``Gemma4TextModel.forward``), and the causal-LM wrapper
feeds that tensor straight into ``lm_head`` with no further norm. The decode
path instead computed ``unembed(forward(ids).last_hidden_state)``, and
``unembed`` is *defined* to apply the final norm (it exists to read
residual-stream activations captured from block hooks). So the final norm was
applied twice.

A freshly initialized Gemma RMSNorm has ``weight=0`` => gain ``(1+w)=1``, and a
unit-gain norm is idempotent — which is exactly why this never showed up in
tests. Any trained checkpoint has a non-unit gain, making the norm
non-idempotent, shifting the logits and changing the argmax.

These tests pin all of that down at two fidelities: the shared Gemma4-shaped
mock, and a genuinely tiny ``transformers`` Gemma 4 text model, which exercises
real library semantics including ``generate()``.
"""

from __future__ import annotations

import torch

from jlens.generative import greedy_decode

from .mock_gemma4 import MockGemma4ForConditionalGeneration, MockTokenizer

# --------------------------------------------------------------- mock fidelity


def _mock_model():
    from jlens.gemma4 import Gemma4LensModel

    return Gemma4LensModel(MockGemma4ForConditionalGeneration(), MockTokenizer())


def test_mock_final_norm_is_not_idempotent():
    """Guards the guard.

    If the mock's final norm were idempotent (as a default-initialized
    LayerNorm/RMSNorm is), applying it twice would be a no-op and every
    double-norm test below would pass vacuously against a broken
    implementation. The mock deliberately randomizes the norm's affine
    parameters; assert that this actually bites.
    """
    model = _mock_model()
    norm = model._final_norm
    hidden = torch.randn(1, 5, model.d_model)
    once = norm(hidden)
    twice = norm(once)
    assert float((once - twice).abs().max()) > 1e-3


def test_mock_text_model_returns_post_norm_last_hidden_state():
    """The mock mirrors HF: the text model applies the final norm itself, so
    ``last_hidden_state`` is already normed and must not be re-normed."""
    model = _mock_model()
    ids = model.encode("post norm probe")
    with torch.no_grad():
        last_hidden = model.forward(ids).last_hidden_state
        block_output = None

        def hook(module, inputs, output):
            nonlocal block_output
            block_output = output

        handle = model.layers[model.n_layers - 1].register_forward_hook(hook)
        try:
            model.forward(ids)
        finally:
            handle.remove()
        expected = model._final_norm(block_output)
    torch.testing.assert_close(last_hidden, expected)
    # And it is genuinely different from the raw residual stream.
    assert float((last_hidden - block_output).abs().max()) > 1e-4


def test_logits_from_ids_matches_models_own_head_and_not_double_norm():
    """``logits_from_ids`` is the model's own pathway; the old
    ``unembed(forward(...).last_hidden_state)`` form is measurably different."""
    model = _mock_model()
    ids = model.encode("head pathway probe")
    with torch.no_grad():
        via_helper = model.logits_from_ids(ids)
        via_model = model._hf_model(input_ids=ids, use_cache=False).logits
        double_normed = model.unembed(model.forward(ids).last_hidden_state)
    torch.testing.assert_close(via_helper, via_model)
    assert float((via_helper - double_normed).abs().max()) > 1e-3


def test_greedy_decode_uses_the_models_own_head():
    """``greedy_decode`` must follow the model's real output pathway, so its
    per-step choices reproduce a reference loop driven by that same head."""
    model = _mock_model()
    ids = model.encode("greedy head probe")
    result = greedy_decode(model, ids, max_new_tokens=6)

    reference: list[int] = []
    seq = ids
    for _ in range(6):
        with torch.no_grad():
            logits = model.logits_from_ids(seq)[0, -1].float()
        next_id = int(logits.argmax())
        reference.append(next_id)
        seq = torch.cat([seq, torch.tensor([[next_id]])], dim=1)
    assert result.token_ids == reference


# Note: there is deliberately no mock-level test that double-norming changes
# the decoded *tokens*. The mock's 32-entry tied vocabulary collapses to a
# single attractor token from every probe prompt, so its argmax is insensitive
# to the readout even though its logits are not (see the test above). Token- and
# argmax-level divergence is asserted on the real Gemma 4 model below, where it
# is meaningful.


# ------------------------------------------------- real transformers semantics


class _StubTokenizer:
    """Only what ``greedy_decode`` touches."""

    bos_token_id = None
    eos_token_id = None

    def decode(self, ids, **_kw) -> str:
        return " ".join(str(int(i)) for i in ids)


def _tiny_real_gemma4():
    """A genuinely tiny ``transformers`` Gemma 4 text model on CPU.

    Small enough to build in-test, but a real ``Gemma4ForCausalLM``: real
    masks, position ids, PLE, KV sharing, softcapping, and a real
    ``generate()``. Returns ``None`` if this transformers build has no Gemma 4.
    """
    try:
        from transformers import Gemma4ForCausalLM
        from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    except Exception:  # pragma: no cover - depends on the installed build
        return None

    torch.manual_seed(0)
    config = Gemma4TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        hidden_size_per_layer_input=8,
        num_kv_shared_layers=2,
        sliding_window=16,
        # Each attention type must appear before the KV-shared tail that reuses
        # it, so alternate rather than grouping.
        layer_types=[
            "full_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
        ],
        final_logit_softcapping=30.0,
    )
    model = Gemma4ForCausalLM(config).eval()
    with torch.no_grad():
        # Trained checkpoints have a non-unit RMSNorm gain. Without this the
        # final norm is idempotent and double-norming is undetectable.
        model.model.norm.weight.normal_(mean=0.0, std=0.3)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _tiny_lens_model():
    from jlens.hf import HFLensModel, Layout

    hf_model = _tiny_real_gemma4()
    if hf_model is None:
        return None
    return HFLensModel(
        hf_model, _StubTokenizer(), layout=Layout("model"), force_bos=False
    )


def test_real_gemma4_text_model_applies_final_norm_itself():
    """The library fact this whole class of bug rests on."""
    model = _tiny_lens_model()
    if model is None:
        import pytest

        pytest.skip("this transformers build has no Gemma 4")
    ids = torch.tensor([[2, 5, 9, 14, 21]])
    captured = {}

    def hook(module, inputs, output):
        captured["block"] = output

    handle = model.layers[model.n_layers - 1].register_forward_hook(hook)
    try:
        with torch.no_grad():
            last_hidden = model.forward(ids).last_hidden_state
    finally:
        handle.remove()
    with torch.no_grad():
        expected = model._final_norm(captured["block"])
    torch.testing.assert_close(last_hidden, expected)


def test_real_gemma4_logits_from_ids_matches_generate_first_step():
    """``logits_from_ids`` reproduces ``generate()``'s own first-step logits,
    while the double-norm form does not — and picks a different token."""
    model = _tiny_lens_model()
    if model is None:
        import pytest

        pytest.skip("this transformers build has no Gemma 4")
    ids = torch.tensor([[2, 5, 9, 14, 21]])
    with torch.no_grad():
        generated = model._hf_model.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=4,
            do_sample=False,
            num_beams=1,
            use_cache=False,
            return_dict_in_generate=True,
            output_logits=True,
            output_scores=True,
        )
        correct = model.logits_from_ids(ids)[0, -1].float()
        double_normed = model.unembed(model.forward(ids).last_hidden_state)[
            0, -1
        ].float()

    generate_raw = generated.logits[0][0].float()
    # No logits processors are active for pure greedy decoding, so generate's
    # raw logits and processed scores coincide; that rules processors out as a
    # cause of the original mismatch.
    torch.testing.assert_close(generate_raw, generated.scores[0][0].float())

    torch.testing.assert_close(correct, generate_raw, rtol=0, atol=1e-5)
    assert float((double_normed - generate_raw).abs().max()) > 1e-3
    assert int(double_normed.argmax()) != int(generate_raw.argmax())


def test_real_gemma4_greedy_decode_matches_generate():
    """End-to-end: the manual uncached decoder and a fully-determinized
    ``generate()`` produce identical tokens on a real Gemma 4 model. This is
    the gate's assertion, exercised against real library semantics."""
    model = _tiny_lens_model()
    if model is None:
        import pytest

        pytest.skip("this transformers build has no Gemma 4")
    ids = torch.tensor([[2, 5, 9, 14, 21]])
    n_new = 6
    manual = greedy_decode(model, ids, max_new_tokens=n_new)
    with torch.no_grad():
        generated = model._hf_model.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=n_new,
            do_sample=False,
            num_beams=1,
            use_cache=False,
        )
    assert manual.token_ids == [int(t) for t in generated[0, ids.shape[1] :]]
