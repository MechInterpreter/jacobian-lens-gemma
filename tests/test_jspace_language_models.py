# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Prompting, memory conditioning, the adapter, the reconstructor, and the
frozen-Gemma decode interface.

These are the tests that make the leakage and freezing claims in
``docs/jspace_language_autoencoder.md`` checkable rather than asserted.
"""

import pytest
import torch

from jlens.autoencoder.adapter import ConeAdapter
from jlens.autoencoder.checkpoints import (
    load_checkpoint,
    restore_module,
    save_checkpoint,
    state_dict_sha256,
)
from jlens.autoencoder.conditioning import (
    SoftPrefixConditioner,
    assert_gemma_frozen,
    assert_no_frozen_parameters_in_optimizer,
    measure_memory_scale,
)
from jlens.autoencoder.config import AdapterConfig, ReconstructorConfig
from jlens.autoencoder.errors import AutoencoderError
from jlens.autoencoder.mocks import build_mock_stack
from jlens.autoencoder.prompting import (
    PRIMING_RISK_INSTRUCTIONS,
    VERBALIZER_INSTRUCTIONS,
    build_verbalizer_prompt,
    phrase_target_ids,
    resolve_end_of_turn_id,
    resolve_instruction,
)
from jlens.autoencoder.reconstructor import PhraseEmbedder, PhraseReconstructor
from jlens.autoencoder.verbalizer import (
    assert_no_gemma_gradients,
    beam_search,
    greedy_phrase,
    sequence_logprobs,
)
from jlens.generative import prompt_safety_violations

ADAPTER_CONFIG = AdapterConfig(n_memory_tokens=2, hidden_dim=16, beam_width=3, max_new_tokens=4)
RECONSTRUCTOR_CONFIG = ReconstructorConfig(hidden_dim=16, n_heads=4, n_layers=1, dropout=0.0)


@pytest.fixture(scope="module")
def stack():
    return build_mock_stack()


@pytest.fixture(scope="module")
def prompt(stack):
    return build_verbalizer_prompt(stack.tokenizer, n_memory_tokens=2)


@pytest.fixture(scope="module")
def adapter(stack):
    torch.manual_seed(0)
    return ConeAdapter(d_model=stack.d_model, config=ADAPTER_CONFIG, target_rms=5.0).eval()


# ------------------------------------------------------------------ prompting


def test_every_default_instruction_is_priming_free():
    for prompt_id, text in VERBALIZER_INSTRUCTIONS.items():
        assert prompt_safety_violations(text) == [], prompt_id


def test_the_briefs_literal_wording_is_available_but_not_a_default():
    text = PRIMING_RISK_INSTRUCTIONS["verbalizer-brief-literal"]
    assert resolve_instruction("verbalizer-brief-literal") == text
    # It is opt-in precisely because it contains the vocabulary that confounded
    # this repository's earlier generative decodes.
    assert prompt_safety_violations(text)
    assert "verbalizer-brief-literal" not in VERBALIZER_INSTRUCTIONS


def test_unknown_prompt_id_raises_rather_than_defaulting():
    with pytest.raises(AutoencoderError, match="unknown verbalizer prompt id"):
        resolve_instruction("verbalizer-typo")


def test_prompt_has_one_bos_two_turn_markers_and_an_exact_memory_span(stack, prompt):
    ids = list(prompt.token_ids)
    assert ids.count(stack.tokenizer.bos_token_id) == 1
    assert ids[0] == stack.tokenizer.bos_token_id
    assert prompt.structure["n_start_of_turn_markers"] == 2
    assert prompt.structure["n_end_of_turn_markers"] == 1
    assert prompt.memory_end - prompt.memory_start == 2
    assert ids[prompt.memory_start : prompt.memory_end] == [prompt.filler_token_id] * 2
    # The generation prefix must follow the memory, or the answer never attends it.
    assert "<start_of_turn>model" in prompt.right_text


def test_memory_span_scales_with_the_number_of_slots(stack):
    for slots in (1, 4, 8):
        built = build_verbalizer_prompt(stack.tokenizer, n_memory_tokens=slots)
        assert built.n_memory_tokens == slots
        assert built.prompt_len == len(built.token_ids)


def test_end_of_turn_resolution_records_the_expected_id(stack):
    resolved = resolve_end_of_turn_id(stack.tokenizer, expected=106)
    assert resolved["end_of_turn_id"] == 5  # the mock's id
    assert resolved["matches_expected"] is False  # recorded, not silently followed
    assert resolved["end_of_turn_id"] in resolved["stop_token_ids"]


def test_phrase_targets_are_contextual_and_end_with_end_of_turn(stack, prompt):
    resolved = phrase_target_ids(
        stack.tokenizer, prompt, "Great Barrier Reef", end_of_turn_id=5
    )
    assert resolved["n_phrase_tokens"] == 3
    assert resolved["target_token_ids"] == [*resolved["phrase_token_ids"], 5]
    assert resolved["decoded_phrase"] == "Great Barrier Reef"


def test_empty_phrase_is_rejected(stack, prompt):
    with pytest.raises(AutoencoderError, match="phrase is empty"):
        phrase_target_ids(stack.tokenizer, prompt, "   ", end_of_turn_id=5)


# --------------------------------------------------------------- conditioning


def test_gemma_is_frozen_and_the_check_reports_what_it_saw(stack):
    report = assert_gemma_frozen(stack.model)
    assert report["frozen"] is True
    assert report["n_parameters"] > 0


def test_frozen_check_fires_when_a_parameter_is_unfrozen(stack):
    parameter = next(stack.model._hf_model.parameters())
    parameter.requires_grad_(True)
    try:
        with pytest.raises(AutoencoderError, match="must stay frozen"):
            assert_gemma_frozen(stack.model)
    finally:
        parameter.requires_grad_(False)


def test_conditioning_changes_the_logits_only_through_the_memory(stack, prompt):
    conditioner = SoftPrefixConditioner()
    ids = prompt.input_ids(batch=1)
    with torch.no_grad():
        baseline = stack.model.logits_from_ids(ids, n_last=1)
        memory = torch.randn(2, stack.d_model) * 5.0
        with conditioner.conditioned(
            stack.model, memory=memory, span=(prompt.memory_start, prompt.memory_end)
        ) as stats:
            steered = stack.model.logits_from_ids(ids, n_last=1)
        # The hook must be gone the moment the block exits.
        after = stack.model.logits_from_ids(ids, n_last=1)
    assert stats["n_conditioned_forward_passes"] == 1
    assert not torch.allclose(baseline, steered)
    assert torch.allclose(baseline, after)


def test_zero_memory_is_not_a_no_op_but_is_deterministic(stack, prompt):
    """Writing zeros *replaces* the filler embedding rather than adding to them,
    so zero memory is a real condition, not "no hook"."""
    conditioner = SoftPrefixConditioner()
    ids = prompt.input_ids(batch=1)
    zeros = torch.zeros(2, stack.d_model)
    with torch.no_grad():
        with conditioner.conditioned(
            stack.model, memory=zeros, span=(prompt.memory_start, prompt.memory_end)
        ):
            first = stack.model.logits_from_ids(ids, n_last=1)
        with conditioner.conditioned(
            stack.model, memory=zeros, span=(prompt.memory_start, prompt.memory_end)
        ):
            second = stack.model.logits_from_ids(ids, n_last=1)
    assert torch.equal(first, second)


def test_conditioning_broadcasts_one_memory_over_a_beam_batch(stack, prompt):
    conditioner = SoftPrefixConditioner()
    ids = prompt.input_ids(batch=4)
    memory = torch.randn(1, 2, stack.d_model)
    with torch.no_grad(), conditioner.conditioned(
        stack.model, memory=memory, span=(prompt.memory_start, prompt.memory_end)
    ):
        logits = stack.model.logits_from_ids(ids, n_last=1)
    assert logits.shape[0] == 4
    assert torch.allclose(logits[0], logits[3])


def test_conditioning_rejects_a_mismatched_span(stack, prompt):
    conditioner = SoftPrefixConditioner()
    with pytest.raises(AutoencoderError, match="span covers"):
        with conditioner.conditioned(
            stack.model, memory=torch.zeros(3, stack.d_model), span=(1, 3)
        ):
            pass


def test_conditioning_rejects_non_finite_memory(stack, prompt):
    conditioner = SoftPrefixConditioner()
    memory = torch.zeros(2, stack.d_model)
    memory[0, 0] = float("nan")
    with pytest.raises(AutoencoderError, match="NaN/Inf"):
        with conditioner.conditioned(
            stack.model, memory=memory, span=(prompt.memory_start, prompt.memory_end)
        ):
            pass


def test_measured_memory_scale_matches_the_embedding_stream(stack, prompt):
    scale = measure_memory_scale(stack.model, prompt.input_ids(batch=1))
    assert scale["d_model"] == stack.d_model
    assert scale["embedding_rms"] > 0


# -------------------------------------------------------------------- adapter


def test_adapter_takes_q_and_nothing_else():
    import inspect

    parameters = list(inspect.signature(ConeAdapter.forward).parameters)
    assert parameters == ["self", "q"], parameters


def test_adapter_output_shape_and_calibrated_rms(stack, adapter):
    q = torch.randn(5, stack.d_model)
    memory = adapter(q)
    assert memory.shape == (5, ADAPTER_CONFIG.n_memory_tokens, stack.d_model)
    rms = memory.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.full_like(rms, 5.0), atol=1e-4)


def test_adapter_is_permutation_equivariant_so_nothing_positional_leaks(stack, adapter):
    q = torch.randn(4, stack.d_model)
    order = torch.tensor([3, 1, 0, 2])
    assert torch.allclose(adapter(q)[order], adapter(q[order]), atol=1e-6)


def test_adapter_ignores_the_norm_of_q(stack, adapter):
    """``q``'s norm is activation magnitude, not meaning; the input LayerNorm
    removes it."""
    q = torch.randn(1, stack.d_model)
    assert torch.allclose(adapter(q), adapter(q * 37.0), atol=1e-4)


def test_adapter_rejects_non_finite_q(stack, adapter):
    q = torch.zeros(1, stack.d_model)
    q[0, 0] = float("inf")
    with pytest.raises(AutoencoderError, match="NaN/Inf"):
        adapter(q)


def test_optimizer_guard_rejects_frozen_parameters(stack, adapter):
    good = torch.optim.AdamW(adapter.parameters(), lr=1e-3)
    report = assert_no_frozen_parameters_in_optimizer(
        good, frozen_modules=[stack.model], trainable=adapter
    )
    assert report["n_optimizer_tensors"] > 0
    bad = torch.optim.AdamW(
        [*adapter.parameters(), *stack.model._hf_model.parameters()], lr=1e-3
    )
    with pytest.raises(AutoencoderError, match="frozen module"):
        assert_no_frozen_parameters_in_optimizer(
            bad, frozen_modules=[stack.model], trainable=adapter
        )


# -------------------------------------------------------------- reconstructor


def test_reconstructor_output_is_unit_norm_and_matches_q_width(stack):
    torch.manual_seed(0)
    model = PhraseReconstructor(d_model=stack.d_model, config=RECONSTRUCTOR_CONFIG).eval()
    embedder = PhraseEmbedder(stack.model, max_phrase_tokens=8)
    embeddings, mask = embedder.batch([[10, 11, 12], [13, 14]])
    with torch.no_grad():
        output = model(embeddings, mask, 14)
    assert output.shape == (2, stack.d_model)
    assert torch.allclose(output.norm(dim=-1), torch.ones(2), atol=1e-5)


def test_reconstructor_ignores_padding(stack):
    torch.manual_seed(0)
    model = PhraseReconstructor(d_model=stack.d_model, config=RECONSTRUCTOR_CONFIG).eval()
    embedder = PhraseEmbedder(stack.model, max_phrase_tokens=8)
    short_alone, mask_alone = embedder.batch([[13, 14]])
    padded, mask_padded = embedder.batch([[13, 14], [10, 11, 12, 15]])
    with torch.no_grad():
        alone = model(short_alone, mask_alone, 14)
        together = model(padded, mask_padded, 14)[:1]
    assert torch.allclose(alone, together, atol=1e-5)


def test_reconstructor_rejects_phrases_longer_than_the_window(stack):
    embedder = PhraseEmbedder(stack.model, max_phrase_tokens=4)
    with pytest.raises(AutoencoderError, match="max_phrase_tokens"):
        embedder.batch([[10, 11, 12, 13, 14]])


# ----------------------------------------------------------------- verbalizer


def test_sequence_logprobs_are_differentiable_into_the_adapter_only(stack, prompt, adapter):
    trainable = ConeAdapter(
        d_model=stack.d_model, config=ADAPTER_CONFIG, target_rms=5.0
    )
    q = torch.randn(2, stack.d_model)
    memory = trainable(q)
    scored = sequence_logprobs(
        stack.model,
        prompt,
        memory,
        [[10, 11], [12, 13, 14]],
        conditioner=SoftPrefixConditioner(),
        pad_token_id=0,
    )
    assert scored["total"].shape == (2,)
    assert torch.equal(scored["n_tokens"], torch.tensor([2, 3]))
    scored["total"].sum().backward()
    assert any(p.grad is not None for p in trainable.parameters())
    assert_no_gemma_gradients(stack.model)


def test_padding_cannot_change_a_sequences_score(stack, prompt, adapter):
    q = torch.randn(1, stack.d_model)
    with torch.no_grad():
        memory = adapter(q)
        alone = sequence_logprobs(
            stack.model, prompt, memory, [[10, 11]],
            conditioner=SoftPrefixConditioner(), pad_token_id=0,
        )["total"]
        padded = sequence_logprobs(
            stack.model, prompt, memory, [[10, 11], [12, 13, 14, 15]],
            conditioner=SoftPrefixConditioner(), pad_token_id=0,
        )["total"][:1]
    assert torch.allclose(alone, padded, atol=1e-5)


def test_beam_search_returns_sorted_finished_candidates(stack, prompt, adapter):
    q = torch.randn(1, stack.d_model)
    with torch.no_grad():
        memory = adapter(q)
    candidates = beam_search(
        stack.model,
        prompt,
        memory,
        conditioner=SoftPrefixConditioner(),
        beam_width=3,
        max_new_tokens=4,
        stop_token_ids=stack.tokenizer.all_special_ids[:1] + [5],
        pad_token_id=0,
    )
    assert len(candidates) == 3
    scores = [c.logprob for c in candidates]
    assert scores == sorted(scores, reverse=True)
    for candidate in candidates:
        assert candidate.n_tokens <= 4
        assert 5 not in candidate.token_ids  # stop tokens are trimmed off


def test_greedy_is_beam_width_one(stack, prompt, adapter):
    q = torch.randn(1, stack.d_model)
    with torch.no_grad():
        memory = adapter(q)
    kwargs = dict(
        conditioner=SoftPrefixConditioner(),
        max_new_tokens=4,
        stop_token_ids=[5],
        pad_token_id=0,
    )
    greedy = greedy_phrase(stack.model, prompt, memory, **kwargs)
    beam = beam_search(stack.model, prompt, memory, beam_width=1, **kwargs)[0]
    assert greedy.token_ids == beam.token_ids


def test_beam_search_is_deterministic(stack, prompt, adapter):
    q = torch.randn(1, stack.d_model)
    with torch.no_grad():
        memory = adapter(q)
    kwargs = dict(
        conditioner=SoftPrefixConditioner(),
        beam_width=3,
        max_new_tokens=4,
        stop_token_ids=[5],
        pad_token_id=0,
    )
    first = beam_search(stack.model, prompt, memory, **kwargs)
    second = beam_search(stack.model, prompt, memory, **kwargs)
    assert [c.token_ids for c in first] == [c.token_ids for c in second]


def test_different_memories_score_generations_differently(stack, prompt, adapter):
    """The memory must reach the beam scores. Asserting *different tokens* would
    be a claim about the mock's semantics, which it does not have; asserting
    different scores is the property the real experiment depends on."""
    kwargs = dict(
        conditioner=SoftPrefixConditioner(),
        beam_width=2,
        max_new_tokens=4,
        stop_token_ids=[5],
        pad_token_id=0,
    )
    torch.manual_seed(0)
    a = beam_search(stack.model, prompt, torch.randn(2, stack.d_model) * 8, **kwargs)
    b = beam_search(stack.model, prompt, torch.randn(2, stack.d_model) * 8, **kwargs)
    assert [c.logprob for c in a] != [c.logprob for c in b]


# ---------------------------------------------------------------- checkpoints


def test_checkpoint_round_trip_verifies_the_parameter_hash(tmp_path, stack):
    torch.manual_seed(0)
    model = PhraseReconstructor(d_model=stack.d_model, config=RECONSTRUCTOR_CONFIG)
    path = str(tmp_path / "reconstructor.pt")
    metadata = save_checkpoint(path, model, kind="reconstructor", config={"a": 1})
    assert metadata["state_dict_sha256"] == state_dict_sha256(model)
    restored = PhraseReconstructor(d_model=stack.d_model, config=RECONSTRUCTOR_CONFIG)
    assert state_dict_sha256(restored) != metadata["state_dict_sha256"]
    payload = load_checkpoint(path, expect_kind="reconstructor")
    restore_module(restored, payload)
    assert state_dict_sha256(restored) == metadata["state_dict_sha256"]


def test_checkpoint_kind_is_enforced(tmp_path, stack):
    model = PhraseReconstructor(d_model=stack.d_model, config=RECONSTRUCTOR_CONFIG)
    path = str(tmp_path / "reconstructor.pt")
    save_checkpoint(path, model, kind="reconstructor", config={})
    with pytest.raises(ValueError, match="expected a 'adapter' checkpoint"):
        load_checkpoint(path, expect_kind="adapter")
