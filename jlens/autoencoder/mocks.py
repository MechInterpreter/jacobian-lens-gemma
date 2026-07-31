# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""A deterministic CPU stack that stands in for the whole frozen pipeline.

Smoke mode has to exercise the *real* code paths — chat formatting, memory
splicing, pursuit, beam search, preference pairs, the GO/NO-GO report — without
a 16 GB checkpoint. This module supplies the four frozen inputs those paths
need:

* :class:`MockWordTokenizer` — a **word-level** tokenizer with Gemma's special
  surfaces and chat template. Word-level on purpose: the experiment's whole
  dataset contract is "2-6 Gemma tokens per phrase", and a character-level mock
  would make every phrase-length constraint vacuous.
* :class:`MockGemma4ForConditionalGeneration` — the exact module layout the
  adapter hooks into (``model.language_model.{layers,norm,embed_tokens}`` plus a
  tied top-level ``lm_head``), including the detail that the text model applies
  the final norm itself before returning ``last_hidden_state`` and that the
  embedding module carries Gemma's ``sqrt(d_model)`` scale.
* a fitted :class:`~jlens.lens.JacobianLens` and its J-space dictionary.
* :data:`MOCK_PHRASES` and :func:`mock_corpus` — a tiny synthetic corpus in
  which every phrase genuinely recurs, so mining and occurrence collection have
  something real to find.

Everything is seeded; two calls to :func:`build_mock_stack` on any machine
produce bit-identical weights, ids, and corpus text.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from types import SimpleNamespace

import torch
from torch import nn

from jlens.autoencoder.errors import AutoencoderError
from jlens.lens import JacobianLens

#: Special-token surfaces, mirroring the real Gemma tokenizer. Ids are fixed so
#: a stored mock artifact stays readable.
MOCK_SPECIAL_TOKENS: dict[str, int] = {
    "<pad>": 0,
    "<eos>": 1,
    "<bos>": 2,
    "<unk>": 3,
    "<start_of_turn>": 4,
    "<end_of_turn>": 5,
}

#: Where ordinary word ids start.
_WORD_BASE = len(MOCK_SPECIAL_TOKENS)

_WORD_RE = re.compile(r"[A-Za-z0-9']+|[^\sA-Za-z0-9']")

#: Multi-word phrases the synthetic corpus is built around. Each is 2-4 words,
#: i.e. 2-4 mock tokens, inside the experiment's 2-6 token window.
MOCK_PHRASES: tuple[str, ...] = (
    "Great Barrier Reef",
    "black hole",
    "quantum entanglement",
    "Nelson Mandela",
    "solar eclipse",
    "Pacific Ocean",
    "Mount Everest",
    "Amazon rainforest",
    "Roman Empire",
    "Industrial Revolution",
    "Silk Road",
    "Renaissance painter",
    "Arctic tundra",
    "coral polyp",
    "steam engine",
    "printing press",
    "Sahara desert",
    "Danube river",
    "Byzantine mosaic",
    "glacial moraine",
    "monsoon season",
    "tidal estuary",
    "granite outcrop",
    "harbour seal",
    "orbital period",
    "magnetic field",
    "neural pathway",
    "volcanic caldera",
    "limestone cavern",
    "migratory heron",
    "alpine meadow",
    "copper alloy",
    "desert oasis",
    "cypress grove",
    "marble column",
    "wheat harvest",
)

_CONTEXT_TEMPLATES: tuple[str, ...] = (
    "The survey team documented the region around the {phrase} in detail .",
    "Historians have long argued about the significance of the {phrase} .",
    "In the following chapter the author turns to the {phrase} .",
    "Local records from that decade rarely mention the {phrase} .",
    "A later expedition returned with measurements of the {phrase} .",
    "The museum catalogue lists several studies of the {phrase} .",
)

#: Generic words a mined phrase may not consist of (the real miner uses a much
#: larger list; this one only has to make the mock's filter observable).
MOCK_FUNCTION_WORDS = frozenset(
    {"the", "of", "and", "a", "an", "in", "on", "for", "to", "with", "that", "this"}
)


def _vocabulary() -> list[str]:
    """Deterministic word list: every word the corpus, phrases, templates, and
    the prompting/evaluation constants can produce, sorted."""
    from jlens.autoencoder.prompting import (
        VERBALIZER_INSTRUCTIONS,
        VERBALIZER_MEMORY_SENTINEL,
    )

    words: set[str] = set()
    sources = [
        *MOCK_PHRASES,
        *_CONTEXT_TEMPLATES,
        *VERBALIZER_INSTRUCTIONS.values(),
        VERBALIZER_MEMORY_SENTINEL,
        "photosynthesis user model assistant memory phrase name",
    ]
    for text in sources:
        for match in _WORD_RE.findall(text.replace("{phrase}", " ")):
            words.add(match)
    return sorted(words)


class MockWordTokenizer:
    """Word-level tokenizer with Gemma's special surfaces and chat template.

    Unknown words map to ``<unk>``; :meth:`decode` joins word tokens with single
    spaces, which is exactly how :data:`MOCK_PHRASES` are written, so a decoded
    phrase compares equal to its source text.
    """

    def __init__(self, words: list[str] | None = None) -> None:
        self._words = list(words) if words is not None else _vocabulary()
        self.word_to_id = {w: _WORD_BASE + i for i, w in enumerate(self._words)}
        self.id_to_word = {i: w for w, i in self.word_to_id.items()}
        self.bos_token_id = MOCK_SPECIAL_TOKENS["<bos>"]
        self.eos_token_id = MOCK_SPECIAL_TOKENS["<eos>"]
        self.pad_token_id = MOCK_SPECIAL_TOKENS["<pad>"]
        self.unk_token_id = MOCK_SPECIAL_TOKENS["<unk>"]
        self.all_special_ids = sorted(MOCK_SPECIAL_TOKENS.values())

    @property
    def vocab_size(self) -> int:
        return _WORD_BASE + len(self._words)

    def convert_tokens_to_ids(self, token: str) -> int:
        if token in MOCK_SPECIAL_TOKENS:
            return MOCK_SPECIAL_TOKENS[token]
        return self.word_to_id.get(token, self.unk_token_id)

    def _tokenize(self, text: str) -> list[int]:
        ids: list[int] = []
        index = 0
        while index < len(text):
            for surface, token_id in MOCK_SPECIAL_TOKENS.items():
                if text.startswith(surface, index):
                    ids.append(token_id)
                    index += len(surface)
                    break
            else:
                match = _WORD_RE.match(text, index)
                if match is None:  # whitespace
                    index += 1
                    continue
                ids.append(self.word_to_id.get(match.group(0), self.unk_token_id))
                index = match.end()
        return ids

    def __call__(
        self,
        text: str,
        *,
        return_tensors: str | None = "pt",
        truncation: bool = True,
        max_length: int = 512,
        add_special_tokens: bool = True,
    ):
        ids = self._tokenize(text)[: int(max_length)]
        if return_tensors == "pt":
            return SimpleNamespace(input_ids=torch.tensor([ids], dtype=torch.long))
        return SimpleNamespace(input_ids=ids)

    def decode(self, ids, *, skip_special_tokens: bool = False, **_kw) -> str:
        by_id = {v: k for k, v in MOCK_SPECIAL_TOKENS.items()}
        parts: list[str] = []
        for token_id in ids:
            token_id = int(token_id)
            if token_id in by_id:
                if not skip_special_tokens:
                    parts.append(by_id[token_id])
                continue
            parts.append(self.id_to_word.get(token_id, "<unk>"))
        return " ".join(parts)

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        continue_final_message: bool = False,
    ) -> str:
        parts = ["<bos>"]
        last = len(messages) - 1
        for index, message in enumerate(messages):
            opened = f"<start_of_turn>{message['role']}\n{message['content']}"
            if continue_final_message and index == last:
                parts.append(opened)
            else:
                parts.append(f"{opened}<end_of_turn>\n")
        if add_generation_prompt:
            parts.append("<start_of_turn>model\n")
        return "".join(parts)


class _ScaledEmbedding(nn.Embedding):
    """Gemma's embedding scale lives *inside* the embedding module, so a hook on
    the module output sees already-scaled vectors. The mock reproduces that, or
    the adapter's memory-scale calibration would be tuned against the wrong
    magnitude here and the wrong one there."""

    def __init__(self, vocab: int, d_model: int) -> None:
        super().__init__(vocab, d_model)
        self.embed_scale = float(math.sqrt(d_model))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return super().forward(input_ids) * self.embed_scale


class _MockTextConfig:
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


class _MockConfig:
    def __init__(self, text_config: _MockTextConfig) -> None:
        self.model_type = "gemma4"
        self.text_config = text_config

    def get_text_config(self) -> _MockTextConfig:
        return self.text_config


class _MockBlock(nn.Module):
    """``(h + 0.1 * W h) * layer_scalar`` with a causal mixing term, so a
    position's activation actually depends on the ones before it — a mock whose
    blocks were position-independent could not exercise "capture the activation
    immediately before the phrase begins"."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, d_model, bias=False)
        self.mix = nn.Linear(d_model, d_model, bias=False)
        with torch.no_grad():
            self.linear.weight.mul_(0.1)
            self.mix.weight.mul_(0.05)
        self.register_buffer("layer_scalar", torch.ones(1))

    def forward(self, hidden: torch.Tensor, **_kwargs) -> torch.Tensor:
        # Causal prefix mean: position t sees positions <= t only.
        cumulative = hidden.cumsum(dim=1)
        counts = torch.arange(
            1, hidden.shape[1] + 1, device=hidden.device, dtype=hidden.dtype
        ).view(1, -1, 1)
        context = self.mix(cumulative / counts)
        return (hidden + self.linear(hidden) + context) * self.layer_scalar


class _MockTextModel(nn.Module):
    def __init__(self, n_layers: int, d_model: int, vocab: int) -> None:
        super().__init__()
        self.embed_tokens = _ScaledEmbedding(vocab, d_model)
        self.layers = nn.ModuleList(_MockBlock(d_model) for _ in range(n_layers))
        self.norm = nn.LayerNorm(d_model)
        with torch.no_grad():
            # Never identity gain: a unit-gain norm is idempotent and would hide
            # the double-norm bug the repo's HF notes warn about.
            self.norm.weight.normal_(mean=1.0, std=0.3)
            self.norm.bias.normal_(mean=0.0, std=0.1)

    def forward(self, input_ids: torch.Tensor | None = None, use_cache: bool = False):
        hidden = self.embed_tokens(input_ids)
        for block in self.layers:
            hidden = block(hidden)
        return SimpleNamespace(last_hidden_state=self.norm(hidden))


class _MockModel(nn.Module):
    def __init__(self, n_layers: int, d_model: int, vocab: int) -> None:
        super().__init__()
        self.language_model = _MockTextModel(n_layers, d_model, vocab)
        self.vision_tower = nn.Linear(3, 3)
        self.audio_tower = nn.Linear(3, 3)


class MockGemma4ForConditionalGeneration(nn.Module):
    """Layout-compatible stand-in for the real multimodal checkpoint."""

    def __init__(self, n_layers: int = 16, d_model: int = 32, vocab: int = 64) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.model = _MockModel(n_layers, d_model, vocab)
        self.lm_head = nn.Linear(d_model, vocab, bias=False)
        self.lm_head.weight = self.model.language_model.embed_tokens.weight  # tied
        self.config = _MockConfig(_MockTextConfig(n_layers, d_model, vocab))

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool = False,
        logits_to_keep: int | torch.Tensor = 0,
        **_kwargs,
    ):
        hidden = self.model.language_model(
            input_ids=input_ids, use_cache=use_cache
        ).last_hidden_state
        slice_indices = (
            slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        )
        logits = self.lm_head(hidden[:, slice_indices, :])
        cap = self.config.get_text_config().final_logit_softcapping
        if cap is not None:
            logits = cap * torch.tanh(logits / cap)
        return SimpleNamespace(logits=logits)


def mock_corpus(*, n_documents: int = 96, seed: int = 0) -> list[str]:
    """Deterministic synthetic documents in which every :data:`MOCK_PHRASES`
    entry recurs several times in differing contexts."""
    generator = torch.Generator().manual_seed(int(seed))
    documents: list[str] = []
    for index in range(int(n_documents)):
        sentences: list[str] = []
        for step in range(4):
            phrase = MOCK_PHRASES[(index * 4 + step) % len(MOCK_PHRASES)]
            template_index = int(
                torch.randint(len(_CONTEXT_TEMPLATES), (1,), generator=generator)
            )
            sentences.append(_CONTEXT_TEMPLATES[template_index].format(phrase=phrase))
        documents.append(" ".join(sentences))
    return documents


@dataclass
class MockStack:
    """Everything the pipeline treats as frozen, in mock form."""

    model: object  # Gemma4LensModel
    tokenizer: MockWordTokenizer
    lens: JacobianLens
    source_layer: int
    documents: list[str]

    @property
    def d_model(self) -> int:
        return int(self.model.d_model)


def build_mock_stack(
    *,
    source_layer: int = 14,
    n_layers: int = 16,
    d_model: int = 32,
    n_documents: int = 96,
    seed: int = 0,
) -> MockStack:
    """Construct the frozen mock stack.

    ``source_layer`` defaults to 14 so smoke mode runs the same layer index the
    real pilot does; ``n_layers`` must therefore exceed it.
    """
    if not 0 <= source_layer < n_layers:
        raise AutoencoderError(
            f"source_layer {source_layer} out of range for a {n_layers}-layer mock"
        )
    from jlens.gemma4 import Gemma4LensModel

    tokenizer = MockWordTokenizer()
    hf_model = MockGemma4ForConditionalGeneration(
        n_layers=n_layers, d_model=d_model, vocab=tokenizer.vocab_size
    )
    model = Gemma4LensModel(hf_model, tokenizer)
    generator = torch.Generator().manual_seed(int(seed) + 7)
    jacobian = torch.randn(d_model, d_model, generator=generator) / math.sqrt(d_model)
    lens = JacobianLens(
        jacobians={source_layer: jacobian}, n_prompts=8, d_model=d_model
    )
    return MockStack(
        model=model,
        tokenizer=tokenizer,
        lens=lens,
        source_layer=source_layer,
        documents=mock_corpus(n_documents=n_documents, seed=seed),
    )
