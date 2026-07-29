# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""A tiny CPU-only mock with the exact module layout of
``Gemma4ForConditionalGeneration``: text decoder at ``model.language_model``
(with ``layers`` / ``norm`` / ``embed_tokens``), sibling ``vision_tower`` /
``audio_tower`` modules (so layout auto-detection must skip ``model``), a
tied top-level ``lm_head``, per-block ``layer_scalar`` buffers, and a
``final_logit_softcapping`` text config. No transformers dependency."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn


class MockGemma4TextConfig:
    def __init__(self, n_layers: int, d_model: int, vocab: int) -> None:
        self.model_type = "gemma4_text"
        self.num_hidden_layers = n_layers
        self.hidden_size = d_model
        self.vocab_size = vocab
        self.final_logit_softcapping = 30.0
        self.enable_moe_block = False
        self.hidden_size_per_layer_input = 4
        self.num_kv_shared_layers = 2
        self.sliding_window = 8
        self.layer_types = ["sliding_attention"] * (n_layers - 1) + ["full_attention"]


class MockGemma4Config:
    def __init__(self, text_config: MockGemma4TextConfig) -> None:
        self.model_type = "gemma4"
        self.text_config = text_config

    def get_text_config(self) -> MockGemma4TextConfig:
        return self.text_config


class MockBlock(nn.Module):
    """Residual block ``(h + 0.1 * linear(h)) * layer_scalar`` returning a
    plain tensor, like ``Gemma4TextDecoderLayer``."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, d_model, bias=False)
        with torch.no_grad():
            self.linear.weight.mul_(0.1)
        self.register_buffer("layer_scalar", torch.ones(1))

    def forward(self, hidden: torch.Tensor, **kwargs) -> torch.Tensor:
        return (hidden + self.linear(hidden)) * self.layer_scalar


class MockTextModel(nn.Module):
    """Mirror of ``Gemma4TextModel``, including the detail that HuggingFace
    text models apply the **final norm themselves** before returning
    ``last_hidden_state`` (``modeling_gemma4.Gemma4TextModel.forward``). So
    ``last_hidden_state`` here is post-norm, exactly as for the real model —
    a mock that skipped this would hide double-norm bugs in callers.

    The norm's affine parameters are deliberately randomized away from the
    identity: a unit-gain RMSNorm/LayerNorm is idempotent, so with default
    initialization applying the final norm twice is a no-op and the bug
    becomes invisible. A trained checkpoint never has identity gain.
    """

    def __init__(self, n_layers: int, d_model: int, vocab: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, d_model)
        self.layers = nn.ModuleList(MockBlock(d_model) for _ in range(n_layers))
        self.norm = nn.LayerNorm(d_model)
        with torch.no_grad():
            self.norm.weight.normal_(mean=1.0, std=0.3)
            self.norm.bias.normal_(mean=0.0, std=0.1)

    def forward(self, input_ids: torch.Tensor | None = None, use_cache: bool = False):
        hidden = self.embed_tokens(input_ids)
        for block in self.layers:
            hidden = block(hidden)
        return SimpleNamespace(last_hidden_state=self.norm(hidden))


class MockGemma4Model(nn.Module):
    """Mirror of ``Gemma4Model``: language_model + tower siblings, no
    ``layers``/``norm``/``embed_tokens`` of its own."""

    def __init__(self, n_layers: int, d_model: int, vocab: int) -> None:
        super().__init__()
        self.language_model = MockTextModel(n_layers, d_model, vocab)
        self.vision_tower = nn.Linear(3, 3)
        self.audio_tower = nn.Linear(3, 3)


class MockGemma4ForConditionalGeneration(nn.Module):
    def __init__(self, n_layers: int = 6, d_model: int = 8, vocab: int = 32) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.model = MockGemma4Model(n_layers, d_model, vocab)
        self.lm_head = nn.Linear(d_model, vocab, bias=False)
        self.lm_head.weight = self.model.language_model.embed_tokens.weight  # tied
        self.config = MockGemma4Config(MockGemma4TextConfig(n_layers, d_model, vocab))

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool = False,
        logits_to_keep: int | torch.Tensor = 0,
        **_kwargs,
    ):
        """The model's own output pathway, mirroring
        ``Gemma4ForConditionalGeneration.forward``: the text model's
        (already-normed) ``last_hidden_state`` goes straight into ``lm_head``
        with **no** further norm, then the final-logit softcap.

        ``logits_to_keep`` is accepted and applied with the same slice-before-
        the-head semantics as the real model, so that the mock exercises the
        path ``generate()`` actually drives (``GenerationMixin`` sets it to 1)
        and ``HFLensModel.supports_logits_to_keep`` detects it.
        """
        hidden = self.model.language_model(
            input_ids=input_ids, use_cache=use_cache
        ).last_hidden_state
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden[:, slice_indices, :])
        cap = self.config.get_text_config().final_logit_softcapping
        if cap is not None:
            logits = logits / cap
            logits = torch.tanh(logits)
            logits = logits * cap
        return SimpleNamespace(logits=logits)


class MockTokenizer:
    """Byte-ish tokenizer. ``auto_bos`` controls whether it prepends BOS on
    its own, so tests can exercise both branches of the adapter's explicit
    BOS handling."""

    bos_token_id = 2
    eos_token_id = None

    def __init__(self, auto_bos: bool = False) -> None:
        self.auto_bos = auto_bos

    def __call__(
        self,
        text: str,
        *,
        return_tensors: str = "pt",
        truncation: bool = True,
        max_length: int = 128,
        add_special_tokens: bool = True,
    ):
        ids = [3 + (b % 29) for b in text.encode()]
        if self.auto_bos and add_special_tokens:
            ids = [self.bos_token_id] + ids
        ids = ids[:max_length]
        if return_tensors == "pt":
            return SimpleNamespace(input_ids=torch.tensor([ids]))
        return SimpleNamespace(input_ids=ids)

    def decode(self, ids, **_kw) -> str:
        return "".join(chr(93 + int(i)) for i in ids)

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        continue_final_message: bool = False,
    ) -> str:
        parts = [f"<{m['role']}> {m['content']}" for m in messages]
        if add_generation_prompt:
            parts.append("<assistant>")
        return "\n".join(parts)
