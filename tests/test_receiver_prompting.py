# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Receiver prompting for the generative J-cone experiment.

Covers the fix for the two compounding defects that invalidated the first
generative runs on the instruction-tuned ``google/gemma-4-E4B-it`` checkpoint:

1. receiver prompts were tokenized raw, so the model never saw the chat turn
   structure it was tuned to answer; and
2. the prompts themselves contained "internal concept" / "Label:", making a
   restatement of the prompt ("Internal Concept") the most likely continuation
   of the *wording* — indistinguishable, in the recorded metrics, from a
   steering effect.

Everything here runs on the CPU mock: no weights are downloaded.
"""

from __future__ import annotations

import pytest
import torch

from jlens.gemma4 import Gemma4LensModel
from jlens.generative import (
    DEFAULT_RECEIVER_FORMAT,
    FORBIDDEN_PROMPT_TERMS,
    LEGACY_DIAGNOSTIC_PROMPTS,
    NEUTRAL_PROMPTS,
    RECEIVER_FORMATS,
    GenerativeError,
    SteeringSchedule,
    SteeringSpec,
    assert_clean_prompt,
    chat_control_token_ids,
    contextual_target_resolver,
    contextual_target_token_ids,
    encode_receiver_prompt,
    greedy_decode,
    is_clean_prompt_id,
    make_generative_record,
    prompt_safety_violations,
    receiver_format_from_config,
    receiver_prompt_debug,
    render_receiver_prompt,
    resolve_neutral_prompt,
    resolve_steering_anchor,
    steering_injection,
    validate_target_tokens,
)

from .mock_gemma4 import (
    MOCK_SPECIAL_TOKENS,
    MockGemma4ForConditionalGeneration,
    MockTokenizer,
)

D_MODEL = 8


def _model(**kw) -> Gemma4LensModel:
    return Gemma4LensModel(MockGemma4ForConditionalGeneration(**kw), MockTokenizer())


def _receiver(prompt_id="noun-phrase-only", receiver_format="chat", tokenizer=None):
    return encode_receiver_prompt(
        tokenizer or MockTokenizer(),
        resolve_neutral_prompt(prompt_id),
        prompt_id=prompt_id,
        receiver_format=receiver_format,
    )


# ------------------------------------------------------- clean prompt safety


def test_default_prompts_are_free_of_priming_vocabulary():
    """The exact defect: a prompt containing "concept"/"label" makes its own
    paraphrase the model's likeliest continuation."""
    assert set(NEUTRAL_PROMPTS) == {
        "noun-phrase-only",
        "what-is-described",
        "name-the-entity",
    }
    for prompt_id, text in NEUTRAL_PROMPTS.items():
        assert prompt_safety_violations(text) == [], prompt_id
        assert_clean_prompt(text, prompt_id=prompt_id)
        assert is_clean_prompt_id(prompt_id)
        # Target-independent: no benchmark concept may appear in a prompt.
        for concept in ("eclipse", "mandela", "everest", "wheelbarrow"):
            assert concept not in text.lower()


def test_default_prompt_texts_are_exactly_the_specified_wording():
    assert NEUTRAL_PROMPTS["noun-phrase-only"] == (
        "Reply with only a specific one- to four-word noun phrase."
    )
    assert NEUTRAL_PROMPTS["what-is-described"] == (
        "What is being described? Reply with only its specific name."
    )
    assert NEUTRAL_PROMPTS["name-the-entity"] == (
        "Answer with only the specific name of the entity, object, event, or "
        "phenomenon."
    )


def test_legacy_prompts_are_kept_but_flagged_as_confounded():
    """The old prompts stay available for reproducing the confounded runs, are
    clearly named as legacy, and each really does carry priming vocabulary —
    so the clean/legacy distinction is a fact about the text, not a label."""
    assert set(LEGACY_DIAGNOSTIC_PROMPTS) == {
        "legacy-label-colon",
        "legacy-answer-four-words",
        "legacy-shortest-label-is",
    }
    for prompt_id, text in LEGACY_DIAGNOSTIC_PROMPTS.items():
        assert prompt_id.startswith("legacy-")
        assert prompt_safety_violations(text), prompt_id
        assert not is_clean_prompt_id(prompt_id)
        with pytest.raises(GenerativeError, match="priming term"):
            assert_clean_prompt(text, prompt_id=prompt_id)
    # The old ids are gone: a stale config cannot silently resolve to them.
    for retired in ("label-colon", "answer-four-words", "shortest-label-is"):
        with pytest.raises(GenerativeError, match="unknown receiver prompt id"):
            resolve_neutral_prompt(retired)


def test_assert_clean_prompt_catches_every_forbidden_term():
    for term in FORBIDDEN_PROMPT_TERMS:
        with pytest.raises(GenerativeError, match="priming term"):
            assert_clean_prompt(f"Name the {term} shown.", prompt_id="probe")
    # Case-insensitive and inflection-tolerant (substring match).
    with pytest.raises(GenerativeError):
        assert_clean_prompt("Give the LABELS.", prompt_id="probe")
    with pytest.raises(GenerativeError):
        assert_clean_prompt("", prompt_id="probe")


def test_shipped_configs_use_clean_prompts_and_chat_format():
    pytest.importorskip("yaml")
    from jlens.metadata import load_generative_config

    for path in (
        "configs/gemma_generative_validation.yaml",
        "configs/gemma_generative_dev_calibration.yaml",
    ):
        config = load_generative_config(path)
        assert receiver_format_from_config(config) == "chat", path
        assert config["neutral_prompts"] == [
            "noun-phrase-only",
            "what-is-described",
            "name-the-entity",
        ], path
        for prompt_id in config["neutral_prompts"]:
            assert_clean_prompt(resolve_neutral_prompt(prompt_id), prompt_id=prompt_id)


def test_config_validation_rejects_unknown_and_legacy_only_prompts():
    pytest.importorskip("yaml")
    import copy

    from jlens.metadata import load_generative_config, validate_generative_config

    config = load_generative_config("configs/gemma_generative_validation.yaml")

    bad = copy.deepcopy(config)
    bad["neutral_prompts"] = ["label-colon"]
    with pytest.raises(ValueError, match="unknown prompt id"):
        validate_generative_config(bad)

    bad = copy.deepcopy(config)
    bad["neutral_prompts"] = ["legacy-label-colon"]
    with pytest.raises(ValueError, match="at least one default"):
        validate_generative_config(bad)

    bad = copy.deepcopy(config)
    bad["receiver"] = {"format": "raw"}
    with pytest.raises(ValueError, match="receiver.format"):
        validate_generative_config(bad)

    # Legacy diagnostics remain usable *alongside* a default prompt.
    ok = copy.deepcopy(config)
    ok["neutral_prompts"] = ["noun-phrase-only", "legacy-label-colon"]
    validate_generative_config(ok)


# ------------------------------------------------------ chat vs legacy encoding


def test_chat_encoding_has_one_bos_one_user_turn_and_generation_prefix():
    receiver = _receiver()
    tokenizer = MockTokenizer()
    rendered = receiver.rendered_prompt

    assert receiver.receiver_format == "chat" == DEFAULT_RECEIVER_FORMAT
    assert rendered.count("<bos>") == 1
    assert rendered.count("<start_of_turn>") == 2  # user turn + model prefix
    assert rendered.count("<end_of_turn>") == 1
    assert rendered.startswith("<bos><start_of_turn>user")
    assert rendered.endswith("<start_of_turn>model\n")
    assert rendered.count(receiver.raw_prompt) == 1

    ids = list(receiver.token_ids)
    assert ids.count(tokenizer.bos_token_id) == 1
    assert ids[0] == tokenizer.bos_token_id
    assert receiver.structure["n_bos_tokens"] == 1
    assert receiver.structure["generation_prefix"] == "<start_of_turn>model\n"
    assert receiver.structure["turn_markers_checked"] is True
    # The recorded decode is the decode of the recorded ids (the mock's
    # character mapping is lossy, so this is not the rendering itself), and the
    # turn structure survives tokenization as single control tokens.
    assert receiver.decoded_prompt == tokenizer.decode(ids)
    assert receiver.decoded_prompt.count("<start_of_turn>") == 2
    assert receiver.decoded_prompt.count("<end_of_turn>") == 1
    prefix_ids = tokenizer(
        "<start_of_turn>model\n", return_tensors=None, add_special_tokens=False
    ).input_ids
    assert ids[-len(prefix_ids) :] == prefix_ids
    assert len(receiver.token_strings) == len(ids) == receiver.prompt_len


def test_legacy_raw_encoding_matches_the_previous_model_encode_path():
    """``legacy_raw`` must be bit-identical to what the confounded runs did —
    ``Gemma4LensModel.encode`` on the raw prompt text — or those runs are not
    reproducible."""
    model = _model()
    text = resolve_neutral_prompt("legacy-label-colon")
    legacy = encode_receiver_prompt(
        model.tokenizer,
        text,
        prompt_id="legacy-label-colon",
        receiver_format="legacy_raw",
    )
    assert legacy.rendered_prompt == text
    assert list(legacy.token_ids) == [int(t) for t in model.encode(text)[0]]
    assert legacy.structure["turn_markers_checked"] is False
    assert legacy.structure["generation_prefix"] is None


def test_chat_and_legacy_encodings_genuinely_differ():
    text = NEUTRAL_PROMPTS["what-is-described"]
    tokenizer = MockTokenizer()
    chat = encode_receiver_prompt(tokenizer, text, receiver_format="chat")
    raw = encode_receiver_prompt(tokenizer, text, receiver_format="legacy_raw")
    assert chat.token_ids != raw.token_ids
    assert chat.prompt_len > raw.prompt_len
    assert MOCK_SPECIAL_TOKENS["<start_of_turn>"] in chat.token_ids
    assert MOCK_SPECIAL_TOKENS["<start_of_turn>"] not in raw.token_ids


def test_unknown_receiver_format_is_rejected_everywhere():
    tokenizer = MockTokenizer()
    assert RECEIVER_FORMATS == ("chat", "legacy_raw")
    with pytest.raises(GenerativeError, match="receiver format"):
        render_receiver_prompt(tokenizer, "probe", receiver_format="raw")
    with pytest.raises(GenerativeError, match="receiver format"):
        encode_receiver_prompt(tokenizer, "probe", receiver_format="chatml")
    with pytest.raises(GenerativeError, match="receiver.format"):
        receiver_format_from_config({"receiver": {"format": "instruct"}})
    assert receiver_format_from_config({}) == "chat"
    assert receiver_format_from_config({"receiver": {"format": "legacy_raw"}}) == (
        "legacy_raw"
    )


def test_chat_format_requires_a_chat_template():
    class NoTemplate(MockTokenizer):
        apply_chat_template = None

    with pytest.raises(GenerativeError, match="apply_chat_template"):
        encode_receiver_prompt(NoTemplate(), "probe", receiver_format="chat")


def test_malformed_chat_template_is_rejected():
    class DoubleBos(MockTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            rendered = MockTokenizer.apply_chat_template(self, messages, **kwargs)
            return "<bos>" + rendered

    with pytest.raises(GenerativeError, match="exactly 1 BOS"):
        encode_receiver_prompt(DoubleBos(), "probe", receiver_format="chat")

    class NoGenerationPrompt(MockTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            kwargs["add_generation_prompt"] = False
            return MockTokenizer.apply_chat_template(self, messages, **kwargs)

    with pytest.raises(GenerativeError, match="model-generation prefix"):
        encode_receiver_prompt(
            NoGenerationPrompt(), "probe", receiver_format="chat"
        )


# ------------------------------------------------- contextual target extraction


def test_contextual_target_is_the_continuation_of_the_formatted_prompt():
    tokenizer = MockTokenizer()
    receiver = _receiver(tokenizer=tokenizer)
    info = contextual_target_token_ids(tokenizer, receiver, " owl")

    assert info["derivation"] == "chat_assistant_continuation"
    joint = tokenizer(
        receiver.rendered_prompt + " owl",
        return_tensors=None,
        add_special_tokens=False,
    ).input_ids
    assert info["raw_continuation_token_ids"] == joint[receiver.prompt_len :]
    assert info["target_token_ids"] == info["raw_continuation_token_ids"]
    assert info["n_target_tokens"] == len(info["target_token_ids"])
    assert info["target_token_strings"] == [
        tokenizer.decode([t]) for t in info["target_token_ids"]
    ]
    assert info["decoded_target"] == tokenizer.decode(info["target_token_ids"])


def test_contextual_target_excludes_bos_and_chat_control_tokens():
    """A continuation that carries chat machinery (BOS, turn markers, the
    assistant end-of-turn) must be reduced to answer content only."""
    tokenizer = MockTokenizer()
    receiver = _receiver(tokenizer=tokenizer)
    control = chat_control_token_ids(tokenizer)
    assert {
        MOCK_SPECIAL_TOKENS["<bos>"],
        MOCK_SPECIAL_TOKENS["<start_of_turn>"],
        MOCK_SPECIAL_TOKENS["<end_of_turn>"],
    } <= control

    info = contextual_target_token_ids(tokenizer, receiver, " owl<end_of_turn>")
    assert MOCK_SPECIAL_TOKENS["<end_of_turn>"] in info["raw_continuation_token_ids"]
    assert info["excluded_token_ids"] == [MOCK_SPECIAL_TOKENS["<end_of_turn>"]]
    assert not (set(info["target_token_ids"]) & control)
    assert info["target_token_ids"] == contextual_target_token_ids(
        tokenizer, receiver, " owl"
    )["target_token_ids"]

    with pytest.raises(GenerativeError, match="only control tokens"):
        contextual_target_token_ids(tokenizer, receiver, "<end_of_turn>")


def test_legacy_raw_target_derivation_is_the_old_standalone_behaviour():
    tokenizer = MockTokenizer()
    receiver = _receiver(prompt_id="legacy-label-colon", receiver_format="legacy_raw",
                         tokenizer=tokenizer)
    info = contextual_target_token_ids(tokenizer, receiver, " owl")
    assert info["derivation"] == "legacy_raw_standalone_phrase"
    assert info["excluded_token_ids"] == []
    assert info["target_token_ids"] == [
        int(t)
        for t in tokenizer(
            " owl", return_tensors=None, add_special_tokens=False
        ).input_ids
    ]


def test_contextual_targets_go_through_the_2_to_6_token_rule():
    tokenizer = MockTokenizer()
    receiver = _receiver(tokenizer=tokenizer)
    examples = [
        {"example_id": "ok", "target_phrase": " owl"},          # 4 mock tokens
        {"example_id": "long", "target_phrase": " wheelbarrow"},  # 12 tokens
    ]
    resolve = contextual_target_resolver(receiver)
    with pytest.raises(GenerativeError, match="token requirement"):
        validate_target_tokens(
            examples, tokenizer, resolve=resolve, use_manifest_ids=False
        )
    resolved = validate_target_tokens(
        examples[:1], tokenizer, resolve=resolve, use_manifest_ids=False
    )
    assert resolved["ok"]["n_target_tokens"] == 4
    assert resolved["ok"]["target_token_ids"] == resolve(tokenizer, " owl")


def test_manifest_target_ids_never_override_contextual_derivation():
    """A pre-resolved id list in the manifest is a *standalone* segmentation, so
    honouring it in chat mode would score tokens the model never produces."""
    tokenizer = MockTokenizer()
    receiver = _receiver(tokenizer=tokenizer)
    examples = [
        {"example_id": "ex", "target_phrase": " owl", "target_token_ids": [7, 8, 9]}
    ]
    resolve = contextual_target_resolver(receiver)
    contextual = validate_target_tokens(
        examples, tokenizer, resolve=resolve, use_manifest_ids=False
    )
    assert contextual["ex"]["target_token_ids"] == resolve(tokenizer, " owl")
    assert contextual["ex"]["target_token_ids"] != [7, 8, 9]
    # The old opt-in behaviour still exists for legacy reproduction.
    honoured = validate_target_tokens(examples, tokenizer, resolve=resolve)
    assert honoured["ex"]["target_token_ids"] == [7, 8, 9]


def test_retokenized_prompt_boundary_fails_loudly():
    """If appending the phrase re-tokenizes the prompt, no slice is safe — the
    run must abort rather than silently score a misaligned target."""

    class MergingTokenizer(MockTokenizer):
        def _tokenize(self, text: str) -> list[int]:
            ids = MockTokenizer._tokenize(self, text)
            # Any prompt that has something appended loses its last token, so
            # the prompt ids stop being a prefix of the joint ids.
            if text.endswith("owl"):
                del ids[-5]
            return ids

    tokenizer = MergingTokenizer()
    receiver = _receiver(tokenizer=tokenizer)
    with pytest.raises(GenerativeError, match="re-tokenized the formatted"):
        contextual_target_token_ids(tokenizer, receiver, " owl")


# --------------------------------------------------------------- hook anchoring


def test_anchor_is_the_final_formatted_prompt_position():
    tokenizer = MockTokenizer()
    receiver = _receiver(tokenizer=tokenizer)
    anchor = receiver.steering_anchor

    assert anchor["prompt_len"] == len(receiver.token_ids)
    assert anchor["anchor_index"] == anchor["prompt_len"] - 1
    assert anchor["anchor_token_id"] == receiver.token_ids[-1]
    assert anchor["anchor_token_string"] == tokenizer.decode(
        [receiver.token_ids[-1]]
    )
    assert anchor["is_final_prompt_position"] is True
    assert anchor["predicts_first_answer_token"] is True
    # The chat prompt is longer than the raw one, so the anchor really moved:
    # a raw-prompt anchor index would land mid-prompt here.
    raw = encode_receiver_prompt(
        tokenizer, receiver.raw_prompt, receiver_format="legacy_raw"
    )
    assert anchor["anchor_index"] > raw.prompt_len - 1


def test_explicit_and_negative_anchors_resolve_against_the_formatted_length():
    tokenizer = MockTokenizer()
    receiver = _receiver(tokenizer=tokenizer)
    n = receiver.prompt_len
    wrong = resolve_steering_anchor(receiver.token_ids, tokenizer, anchor=1)
    assert wrong["anchor_index"] == 1
    assert wrong["is_final_prompt_position"] is False
    assert wrong["anchor_token_id"] == receiver.token_ids[1]
    assert resolve_steering_anchor(
        receiver.token_ids, tokenizer, anchor=-1
    ) == receiver.steering_anchor
    with pytest.raises(GenerativeError, match="out of range"):
        resolve_steering_anchor(receiver.token_ids, tokenizer, anchor=n)
    with pytest.raises(GenerativeError, match="empty prompt"):
        resolve_steering_anchor([], tokenizer)


def test_hook_injects_at_the_resolved_chat_anchor_only():
    """End-to-end anchoring: the position the resolver names is the position the
    hook edits, for a chat-formatted prompt."""
    model = _model()
    receiver = _receiver(tokenizer=model.tokenizer)
    ids = receiver.input_ids(device=model.input_device)
    anchor = receiver.steering_anchor
    with torch.no_grad():
        baseline = model.forward(ids).last_hidden_state
    with steering_injection(
        model.layers,
        model.n_layers - 1,  # last block: no downstream mixing in the mock
        delta=torch.full((D_MODEL,), 3.0),
        schedule=SteeringSchedule("prompt_only"),
        prompt_len=anchor["prompt_len"],
    ) as stats:
        with torch.no_grad():
            steered = model.forward(ids).last_hidden_state
    changed = (
        ((steered - baseline).abs().sum(dim=-1)[0] > 0).nonzero().flatten().tolist()
    )
    assert changed == [anchor["anchor_index"]]
    assert stats["resolved_anchor"] == anchor["anchor_index"]


# ------------------------------------------------- generated-token serialization


def test_greedy_decode_returns_only_new_tokens_and_decodes_only_those():
    model = _model()
    receiver = _receiver(tokenizer=model.tokenizer)
    ids = receiver.input_ids(device=model.input_device)
    result = greedy_decode(model, ids, max_new_tokens=4)

    assert 1 <= len(result.token_ids) <= 4
    assert result.n_steps == len(result.token_ids)
    assert len(result.chosen_logprobs) == len(result.token_ids)
    prompt_ids = list(receiver.token_ids)
    assert result.token_ids != prompt_ids[-len(result.token_ids) :]
    assert result.text == model.tokenizer.decode(
        result.token_ids, skip_special_tokens=True
    )
    # No prompt echo: the recorded text is not the prompt's decode.
    assert receiver.decoded_prompt not in result.text
    assert result.to_dict()["generated_token_ids"] == result.token_ids


def test_record_always_carries_generated_ids_and_receiver_provenance():
    model = _model()
    receiver = _receiver(tokenizer=model.tokenizer)
    ids = receiver.input_ids(device=model.input_device)
    decode = greedy_decode(model, ids, max_new_tokens=3)
    anchor = receiver.steering_anchor

    def record(decode_result):
        return make_generative_record(
            run_id="run",
            example_id="ex",
            condition="full_cone",
            source_layer=1,
            injection_layer=1,
            source_position=-1,
            injection_anchor=anchor["anchor_index"],
            schedule=SteeringSchedule("prompt_only"),
            neutral_prompt_id=receiver.prompt_id,
            receiver_prompt_id=receiver.prompt_id,
            receiver_format=receiver.receiver_format,
            receiver_prompt_len=anchor["prompt_len"],
            anchor_token_id=anchor["anchor_token_id"],
            anchor_token_string=anchor["anchor_token_string"],
            strength_ratio=0.1,
            vector_meta={},
            hook_stats={},
            scoring={"total_logprob": -1.0},
            decode=decode_result,
            delta_vs_zero=None,
            delta_vs_unrelated=None,
            kl_divergence=None,
            target_phrase=" owl",
            target_recovered_exact=None,
            target_recovered_substring=None,
            seed=None,
            provenance={},
        )

    decoded = record(decode)
    assert decoded["generated_token_ids"] == decode.token_ids
    assert decoded["generated_text"] == decode.text
    assert decoded["receiver_prompt_id"] == "noun-phrase-only"
    assert decoded["receiver_format"] == "chat"
    assert decoded["receiver_prompt_len"] == anchor["prompt_len"]
    assert decoded["anchor_token_id"] == anchor["anchor_token_id"]
    assert decoded["anchor_token_string"] == anchor["anchor_token_string"]

    # Not decoded: the key is present and null, never absent.
    scored_only = record(None)
    assert "generated_token_ids" in scored_only
    assert scored_only["generated_token_ids"] is None

    import json

    json.dumps(decoded)
    json.dumps(scored_only)


def test_prompt_debug_entry_documents_prompt_anchor_and_targets():
    tokenizer = MockTokenizer()
    receiver = _receiver(tokenizer=tokenizer)
    targets = {
        "ex": contextual_target_token_ids(tokenizer, receiver, " owl"),
    }
    entry = receiver_prompt_debug(receiver, targets)

    for key in (
        "prompt_id",
        "receiver_format",
        "raw_prompt",
        "rendered_prompt",
        "prompt_token_ids",
        "prompt_tokens",
        "decoded_prompt",
        "prompt_len",
        "steering_anchor",
        "targets",
    ):
        assert key in entry, key
    assert entry["prompt_safety_violations"] == []
    assert entry["targets"]["ex"]["target_token_ids"]
    assert entry["targets"]["ex"]["decoded_target"]

    import json

    json.dumps(entry)


# ---------------------------------------------- parity and controls, chat prompt


def test_none_zero_parity_holds_for_a_chat_formatted_prompt():
    """The parity control must still be exact once the prompt is chat-formatted:
    an unhooked forward and a zero-delta hooked forward are the same logits."""
    model = _model()
    receiver = _receiver(tokenizer=model.tokenizer)
    ids = receiver.input_ids(device=model.input_device)
    with torch.no_grad():
        baseline = model.logits_from_ids(ids, n_last=1)[0, -1]
    spec = SteeringSpec(
        layer=2,
        delta=torch.zeros(D_MODEL),
        schedule=SteeringSchedule("constant"),
    )
    with spec.context(model.layers, prompt_len=receiver.prompt_len):
        with torch.no_grad():
            hooked = model.logits_from_ids(ids, n_last=1)[0, -1]
    assert torch.equal(baseline, hooked)


def test_nonzero_delta_at_the_chat_anchor_changes_the_first_answer_token():
    """Sanity check that the anchor is causally connected to what the model
    predicts next: a large delta at the resolved anchor must move the
    next-token distribution."""
    model = _model()
    receiver = _receiver(tokenizer=model.tokenizer)
    ids = receiver.input_ids(device=model.input_device)
    with torch.no_grad():
        baseline = model.logits_from_ids(ids, n_last=1)[0, -1]
    spec = SteeringSpec(
        layer=2,
        delta=torch.full((D_MODEL,), 5.0),
        schedule=SteeringSchedule("prompt_only"),
    )
    with spec.context(model.layers, prompt_len=receiver.prompt_len):
        with torch.no_grad():
            steered = model.logits_from_ids(ids, n_last=1)[0, -1]
    assert not torch.equal(baseline, steered)
