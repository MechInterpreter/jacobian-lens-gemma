# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Deterministic MOCK mode. No Gemma, no Hub, no Drive, no corpus download.

MOCK mode proves **pipeline behaviour and nothing else**. It cannot tell you
whether a research-grade lens exists, and a green MOCK run is not evidence about
Gemma. It is built in two halves, deliberately:

**Half one — the estimator really runs.** :class:`MockCalibrationModel` is a
tiny differentiable stack implementing :class:`jlens.protocol.LensModel`, so
:func:`jlens.fitting.fit` executes for real against it: real forward pass, real
``torch.autograd.grad``, real multilayer capture in one pass, real per-layer
matrices, real running mean, real atomic checkpoint, real resume. Nothing about
the fitting path is simulated. It is small enough to run on CPU in seconds.

**Half two — the verdict logic meets known cases.** The gate, the scale
comparison and the plateau rule need layers whose *statistics* are known in
advance, which a tiny random model will not supply. So validation rows are
generated from synthetic logit vectors with controlled rank and tie structure —
and then scored by the **real**
:func:`jlens.mmlocalize.lens_validity.tie_aware_row`. The rows are synthetic;
the scoring, the gate, the folds and the plateau arithmetic are not.

The four archetypes exist so that a passing MOCK run demonstrates the pipeline
distinguishing cases, rather than saying yes to everything:

============  ==========================================================
``L8``        **degenerate at every scale** — a large tie block at the
              maximum, so it fails clause 1 however good its optimistic
              rank looks.
``L14``       **improves with scale and eventually passes** — the case
              the scale study exists to detect.
``L32``       **passes on optimistic rank, fails tie-aware validation** —
              the layer-32 shape: optimistic rank 1, midrank deep inside
              a tie block. This is the archetype the whole tie-aware gate
              was built for.
``L38``       **passes at every scale** — the established anchor.
============  ==========================================================

The remaining grid layers get plain behaviours (20 fails cleanly, 26 improves
but never quite passes, 35 and 40 pass) so the table is not four rows wide.

Do not tune these archetypes to make a real run look better. They are fixtures.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn

from jlens.mmlocalize.lens_validity import tie_aware_row

__all__ = [
    "ARCHETYPES",
    "MOCK_D_MODEL",
    "MOCK_LAYERS",
    "MOCK_N_LAYERS",
    "MOCK_SCALE_POINTS",
    "MOCK_TARGET_LAYER",
    "MOCK_VOCAB",
    "LayerArchetype",
    "MockCalibrationModel",
    "mock_corpus_texts",
    "mock_validation_rows",
]

#: The mock stack is the **real depth** (42 blocks, target 41) at a tiny width,
#: so MOCK exercises the actual layer grid rather than a stand-in for it — the
#: fitted layers and the archetype layers are the same numbers, which is what
#: lets the MOCK run reach publication. Width 32 keeps it a few seconds on CPU:
#: cost is ``ceil(d_model / dim_batch)`` backward passes, so 32/8 = 4 per prompt
#: against Gemma's 320.
MOCK_N_LAYERS = 42
MOCK_TARGET_LAYER = 41
MOCK_D_MODEL = 32
MOCK_VOCAB = 512

#: The frozen grid, unchanged.
MOCK_LAYERS = (8, 14, 20, 26, 32, 35, 38, 40)

#: Small synthetic equivalents of 100 / 250 / 1k. Nesting behaviour is what is
#: under test, not the absolute numbers.
MOCK_SCALE_POINTS = (8, 16, 32)


# ------------------------------------------------------------- the tiny model


class _MockBlock(nn.Module):
    """``h + 0.1 * W h`` returning a plain tensor, like a decoder layer."""

    def __init__(self, d_model: int, seed: int) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        weight = torch.randn(d_model, d_model, generator=generator) * 0.1
        self.linear = nn.Linear(d_model, d_model, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(weight)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + 0.1 * self.linear(hidden)


class _MockTokenizer:
    """Just enough tokenizer for the protocol and for token-length filtering."""

    def decode(self, token_ids) -> str:
        return " ".join(str(int(t)) for t in token_ids)

    def token_count(self, text: str) -> int:
        return len(str(text).split())


class MockCalibrationModel(nn.Module):
    """A tiny, frozen, deterministic :class:`~jlens.protocol.LensModel`.

    Real autograd, real hooks, real multilayer capture. Every parameter has
    ``requires_grad=False``, exactly as the architecture audit demands of Gemma,
    so :class:`~jlens.hooks.ActivationRecorder`'s ``start_graph_at`` roots the
    graph at the shallowest source layer just as it does on the real model.

    The final norm is given a **non-identity gain** on purpose: a unit-gain
    RMSNorm is idempotent, which would hide a double-normalization bug in any
    fixture standing in for a trained checkpoint.
    """

    def __init__(
        self,
        *,
        n_layers: int = MOCK_N_LAYERS,
        d_model: int = MOCK_D_MODEL,
        vocab_size: int = MOCK_VOCAB,
        seed: int = 20260805,
    ) -> None:
        super().__init__()
        self.n_layers = int(n_layers)
        self.d_model = int(d_model)
        self.vocab_size = int(vocab_size)
        generator = torch.Generator().manual_seed(seed)

        self.embed = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [_MockBlock(d_model, seed + index) for index in range(n_layers)]
        )
        self.norm_gain = nn.Parameter(
            1.0 + 0.25 * torch.randn(d_model, generator=generator), requires_grad=False
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        with torch.no_grad():
            self.embed.weight.copy_(
                torch.randn(vocab_size, d_model, generator=generator) * 0.05
            )
            self.lm_head.weight.copy_(self.embed.weight)  # tied, like Gemma 4
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

        self.tokenizer = _MockTokenizer()

    @property
    def layers(self) -> nn.ModuleList:
        return self.blocks

    def encode(self, text: str, *, max_length: int = 128) -> torch.Tensor:
        """Deterministic hash tokenization — no vocabulary file, no download."""
        words = str(text).split()[: int(max_length)]
        ids = [
            int(hashlib.sha256(word.encode("utf-8")).hexdigest()[:8], 16)
            % self.vocab_size
            for word in words
        ] or [0]
        return torch.tensor([ids], dtype=torch.long)

    def forward(self, input_ids: torch.Tensor):
        hidden = self.embed(input_ids)
        for block in self.blocks:
            hidden = block(hidden)
        return hidden

    def _final_norm(self, residual: torch.Tensor) -> torch.Tensor:
        variance = residual.float().pow(2).mean(dim=-1, keepdim=True)
        return residual * torch.rsqrt(variance + 1e-6) * self.norm_gain

    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self._final_norm(residual))


def mock_corpus_texts(n: int, *, seed: int = 20260805, words: int = 40) -> list[str]:
    """Deterministic synthetic corpus records. No network, no dataset.

    Long enough that every record clears the ``skip_first + 2`` token floor, and
    varied enough that the SimHash near-duplicate detector sees genuinely
    distinct documents.
    """
    vocabulary = [f"tok{index}" for index in range(997)]
    texts: list[str] = []
    for index in range(int(n)):
        digest = hashlib.sha256(f"{seed}|{index}".encode()).digest()
        picks = [
            vocabulary[(digest[position % len(digest)] * (position + 7) + index) % len(vocabulary)]
            for position in range(int(words))
        ]
        texts.append(f"record {index} " + " ".join(picks) + ".")
    return texts


# ------------------------------------------------------------- the archetypes


@dataclass(frozen=True)
class LayerArchetype:
    """A known statistical case the pipeline must handle.

    The knob is the **exact optimistic rank** the target is placed at, per
    scale-point index, rather than a score offset. Placing the target between
    two order statistics of the random field makes the resulting rank exact and
    the fixture legible: ``rank=(60, 12, 1)`` reads as "60th, then 12th, then
    first", and the gate's arithmetic follows from it without tuning.

    Attributes:
        rank: Optimistic rank per scale index. ``1`` is a unique argmax.
        tie_block: Tokens tied at the maximum per scale index, the target
            among them. ``1`` means no tie. A block of ``t`` gives optimistic
            rank 1 and midrank ``(t + 1) / 2`` — the layer-32 shape.
    """

    layer: int
    label: str
    rank: tuple[int, ...]
    tie_block: tuple[int, ...] = (1, 1, 1)
    expected: str = ""


#: Behaviour per grid layer, indexed by scale-point position (small, mid, large).
ARCHETYPES: dict[int, LayerArchetype] = {
    8: LayerArchetype(
        layer=8,
        label="degenerate at every scale",
        rank=(1, 1, 1),
        tie_block=(64, 64, 64),
        expected="optimistic rank 1, midrank 32.5, tied-at-max rate 1.0; "
        "fails clause 1 at every scale",
    ),
    14: LayerArchetype(
        layer=14,
        label="early layer that improves with scale and eventually passes",
        rank=(60, 12, 1),
        expected="fails at the two smaller scales, passes at the largest",
    ),
    20: LayerArchetype(
        layer=20,
        label="fails cleanly at every scale",
        rank=(200, 200, 190),
        expected="never passes; no tie pathology, simply a poor readout",
    ),
    26: LayerArchetype(
        layer=26,
        label="improves but never reaches the bar",
        rank=(150, 60, 20),
        expected="monotone improvement, still ineligible at the largest scale",
    ),
    32: LayerArchetype(
        layer=32,
        label="optimistic rank 1, fails tie-aware validation",
        rank=(1, 1, 1),
        tie_block=(24, 24, 24),
        expected="median optimistic rank 1.0, median midrank 12.5; the shape the "
        "tie-aware gate exists for",
    ),
    35: LayerArchetype(
        layer=35,
        label="late layer, passes at every scale",
        rank=(1, 1, 1),
        expected="passes everywhere",
    ),
    38: LayerArchetype(
        layer=38,
        label="the established anchor, passes at every scale",
        rank=(1, 1, 1),
        expected="passes everywhere",
    ),
    40: LayerArchetype(
        layer=40,
        label="deepest layer, passes at every scale",
        rank=(1, 1, 1),
        expected="passes everywhere",
    ),
}

#: Fixed optimistic rank for each control, at every layer and scale. Deep enough
#: that a real readout clears the gate's ratio and margin clauses, and identical
#: across layers so a control number never depends on which layer it sits under.
CONTROL_RANK = {
    "permuted": 250,
    "random": 300,
    "wrong_layer": 200,
    "logit_lens": 220,
}


def _place_at_rank(scores: torch.Tensor, target: int, rank: int) -> None:
    """Set ``scores[target]`` so exactly ``rank - 1`` other tokens beat it."""
    others = torch.cat([scores[:target], scores[target + 1 :]])
    ordered = torch.sort(others, descending=True).values
    rank = max(1, min(int(rank), int(ordered.numel())))
    if rank == 1:
        scores[target] = float(ordered[0]) + 1.0
    else:
        scores[target] = (float(ordered[rank - 2]) + float(ordered[rank - 1])) / 2.0


def _scores(
    *,
    layer: int,
    scale_index: int,
    prompt_index: int,
    variant: str,
    target: int,
    vocab: int,
    n_distinct_targets: int,
) -> torch.Tensor:
    """A synthetic logit vector with controlled rank and tie structure."""
    seed = int(
        hashlib.sha256(
            f"{layer}|{scale_index}|{prompt_index}|{variant}".encode()
        ).hexdigest()[:8],
        16,
    )
    generator = torch.Generator().manual_seed(seed)
    scores = torch.randn(vocab, generator=generator)

    archetype = ARCHETYPES[layer]
    if variant == "j_lens":
        rank = archetype.rank[scale_index]
        tie_block = archetype.tie_block[scale_index]
    else:
        rank = CONTROL_RANK[variant]
        tie_block = 1

    _place_at_rank(scores, target, rank)

    if tie_block > 1:
        # Tie partners come from a band that never collides with the target
        # range, so tie structure and target diversity stay independent knobs.
        offset = int(n_distinct_targets) + 1
        peak = float(scores.max()) + 1.0
        partners = {
            offset + (prompt_index * 7 + step * 13) % (vocab - offset)
            for step in range(tie_block - 1)
        }
        partners.discard(target)
        for index in list(partners)[: tie_block - 1]:
            scores[index] = peak
        scores[target] = peak
    return scores


def mock_validation_rows(
    *,
    layers: Sequence[int] = MOCK_LAYERS,
    scale: int,
    scale_index: int,
    n_prompts: int,
    n_distinct_targets: int = 40,
    vocab: int = MOCK_VOCAB,
    variants: Sequence[str] = ("j_lens", "permuted", "random", "wrong_layer", "logit_lens"),
) -> list[dict]:
    """Rows for one scale point, scored by the **real** tie-aware scorer.

    ``n_distinct_targets`` controls target diversity directly: prompt ``i``
    targets token ``i % n_distinct_targets``, so 40 distinct targets over 128
    prompts gives a maximum share of ~2.5% — comfortably inside the gate's 25%
    ceiling, and 40 >= the 24-token floor.
    """
    rows: list[dict] = []
    for layer in layers:
        for prompt_index in range(int(n_prompts)):
            target = prompt_index % int(n_distinct_targets)
            actual = torch.full((int(vocab),), -10.0)
            actual[target] = 10.0
            prompt_sha = hashlib.sha256(
                f"mock|{scale}|{prompt_index}".encode()
            ).hexdigest()
            for variant in variants:
                rows.append(
                    tie_aware_row(
                        sample_index=prompt_index,
                        prompt_sha=prompt_sha,
                        layer=int(layer),
                        variant=variant,
                        variant_logits=_scores(
                            layer=int(layer),
                            scale_index=int(scale_index),
                            prompt_index=prompt_index,
                            variant=variant,
                            target=target,
                            vocab=int(vocab),
                            n_distinct_targets=int(n_distinct_targets),
                        ),
                        actual_logits=actual,
                    )
                )
    return rows
