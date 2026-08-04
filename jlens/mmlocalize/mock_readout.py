# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""A tiny nonlinear text model whose J-lens is its own true Jacobian.

Stage B is the gate that decides which layers may carry a causal claim, so a
MOCK run that skipped it would leave the most consequential code in the study
unexecuted. This module supplies the smallest model that lets the *same* Stage B
lines run: :class:`MockReadoutModel` exposes exactly the surface the notebook
touches — ``encode``, ``forward``, ``unembed``, ``layers``, ``n_layers`` — so the
MOCK path and the L4 path execute one implementation, not two.

The construction is deliberately not a convenient fiction.

Each block is a fixed rotation composed with a saturating perturbation, and
:func:`mock_localization_lens` builds ``J_l`` as the **true first-order
Jacobian** of the remaining stack — the product of the later blocks'
linearizations, averaged over reference inputs the way real fitting averages
over a prompt set. That is the same object the real lens estimates, so the
mock's J-space is the mock model's actual J-space rather than something
arranged to look like one.

**All four layers pass the Stage B gate in MOCK, and that is expected.** The
mock's lens is its own model's exact Jacobian over a 96-token vocabulary in 32
dimensions; transport is nearly exact at every depth, so every layer reads out.
Real early layers fail for reasons this model does not have — a 262k-token
vocabulary whose tail is dense with near-ties, bf16 rounding, and twenty blocks
of genuine nonlinearity between the layer and the output.

That is a limitation of the MOCK, not a defect to be tuned away. Adjusting
:data:`MOCK_BLOCK_SCALE` or :data:`MOCK_HIDDEN_SCALE` until some layer fails
would be manufacturing a result, so it is not done; the "ineligible layer is
skipped causally" branch is exercised directly in the test suite, by
constructing a failing validity record, which tests the behaviour rather than
hoping a mock world happens to produce it.

**What this can and cannot show.** A MOCK verdict is evidence that the pipeline
executes, resumes, and applies its rubric to whatever it is given. It is not
evidence about Gemma, and the mock's eligibility pattern is not a prediction
about which layers will pass on the L4.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import torch
from torch import nn

from jlens.lens import JacobianLens
from jlens.mmlocalize.layers import LOCALIZATION_LAYERS, MODEL_N_LAYERS

#: Small enough to be instant, wide enough for a 262k-token vocabulary's
#: qualitative behaviour (a maximum with a margin, not a plateau).
MOCK_READOUT_D_MODEL = 32
MOCK_READOUT_VOCAB = 96
MOCK_READOUT_SEED = 20260804

#: Per-block saturation. Larger means more nonlinearity and therefore a faster
#: loss of first-order fidelity across that block. A property of the mock world.
MOCK_BLOCK_SCALE = 0.85

#: Hidden-state magnitude. Large enough that ``tanh`` is genuinely curved at a
#: typical coordinate; at unit norm over 32 dimensions it would not be, and the
#: model would be linear in all but name.
MOCK_HIDDEN_SCALE = 3.0

MOCK_LENS_CHECKSUM = "sha256:mock-localization-jacobian-lens"


def _seeded(shape: tuple[int, ...], key: str) -> torch.Tensor:
    digest = hashlib.sha256(key.encode()).digest()
    generator = torch.Generator().manual_seed(int.from_bytes(digest[:8], "big") % (2**63))
    return torch.randn(*shape, generator=generator)


def _rotation(d_model: int, key: str) -> torch.Tensor:
    """A fixed orthogonal matrix. Orthogonal so norms stay O(1) across 42 blocks."""
    q, _ = torch.linalg.qr(_seeded((d_model, d_model), key))
    return q


class _MockReadoutBlock(nn.Module):
    """``h -> R @ ((1 - s) * h + s * tanh(h))``.

    Mostly a rotation, with a saturating perturbation of weight ``s``. The
    perturbation is what makes a first-order transport imperfect, and keeping it
    small is what puts the model in the regime real transport lives in: nearly
    linear over a few blocks, visibly not over twenty.
    """

    def __init__(self, rotation: torch.Tensor, scale: float) -> None:
        super().__init__()
        self.register_buffer("rotation", rotation)
        self.scale = float(scale)

    def forward(self, hidden: torch.Tensor, **_kwargs) -> torch.Tensor:
        perturbed = (1.0 - self.scale) * hidden + self.scale * torch.tanh(hidden)
        return perturbed @ self.rotation.T


class _MockTokenizer:
    """The two methods the Stage B cell calls on a tokenizer."""

    def __init__(self, vocab: int) -> None:
        self.vocab = vocab

    def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=True):
        content = "\n".join(str(message.get("content", "")) for message in messages)
        return f"<start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>model\n"

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict:
        return {"input_ids": _encode_ids(text, self.vocab)}


def _encode_ids(text: str, vocab: int, *, max_length: int = 16) -> list[int]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [digest[i] % vocab for i in range(min(max_length, len(digest)))]


class MockReadoutModel(nn.Module):
    """A 42-block nonlinear text model with the ``Gemma4LensModel`` surface.

    The depth is the real model's, so the MOCK path evaluates the *same physical
    layers* — 20, 26, 32, 38 — as the L4 path. A mock that renumbered its layers
    would leave the physical/normalized mapping untested exactly where it
    matters.
    """

    def __init__(
        self,
        *,
        d_model: int = MOCK_READOUT_D_MODEL,
        vocab: int = MOCK_READOUT_VOCAB,
        n_layers: int = MODEL_N_LAYERS,
        block_scale: float = MOCK_BLOCK_SCALE,
        hidden_scale: float = MOCK_HIDDEN_SCALE,
        seed: int = MOCK_READOUT_SEED,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab = vocab
        self.n_layers = n_layers
        self.block_scale = block_scale
        self.tokenizer = _MockTokenizer(vocab)
        self.layers = nn.ModuleList(
            [
                _MockReadoutBlock(_rotation(d_model, f"{seed}|rot|{index}"), block_scale)
                for index in range(n_layers)
            ]
        )
        self.hidden_scale = float(hidden_scale)
        embedding = _seeded((vocab, d_model), f"{seed}|embed")
        unembedding = _seeded((vocab, d_model), f"{seed}|unembed")
        self.register_buffer(
            "embedding",
            hidden_scale * embedding / embedding.norm(dim=-1, keepdim=True),
        )
        self.register_buffer(
            "unembedding", unembedding / unembedding.norm(dim=-1, keepdim=True)
        )
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    # ------------------------------------------------- the notebook's surface

    def encode(self, prompt: str, max_length: int = 16) -> torch.Tensor:
        return torch.tensor(
            [_encode_ids(prompt, self.vocab, max_length=max_length)], dtype=torch.long
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding[input_ids]
        for block in self.layers:
            hidden = block(hidden)
        return hidden

    def unembed(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.unembedding @ hidden.reshape(-1).float()

    # --------------------------------------------------- the true Jacobian

    @torch.no_grad()
    def reference_trajectory(self, reference_prompt: str = "reference") -> list[torch.Tensor]:
        """Hidden state entering each block, for the reference input.

        The lens is linearized here — one fixed operating point, the same choice
        the real fitting procedure makes when it averages over a prompt set.
        """
        hidden = self.embedding[self.encode(reference_prompt)][0, -1]
        trajectory = [hidden]
        for block in self.layers:
            hidden = block(hidden.unsqueeze(0)).squeeze(0)
            trajectory.append(hidden)
        return trajectory

    @torch.no_grad()
    def _jacobians_for(
        self, reference_prompt: str, wanted: set[int]
    ) -> dict[int, torch.Tensor]:
        """``{layer: d(final residual)/d(residual at layer)}`` for one input.

        "Residual at layer ``l``" is the **output** of block ``l`` — the site
        :class:`~jlens.hooks.ActivationRecorder` captures and the one the whole
        lineage is fitted at. The map to the final residual is therefore the
        product of the linearizations of blocks ``l+1 .. n-1``, which is why the
        accumulator is recorded *before* block ``l`` itself is folded in.
        """
        trajectory = self.reference_trajectory(reference_prompt)
        accumulated = torch.eye(self.d_model)
        jacobians: dict[int, torch.Tensor] = {}
        for index in range(self.n_layers - 1, -1, -1):
            if index in wanted:
                jacobians[index] = accumulated.clone()
            block = self.layers[index]
            entering = trajectory[index]
            derivative = (1.0 - block.scale) + block.scale * (
                1.0 - torch.tanh(entering) ** 2
            )
            accumulated = accumulated @ (block.rotation * derivative.unsqueeze(0))
        return jacobians

    @torch.no_grad()
    def true_jacobians(
        self,
        layers: Sequence[int] = LOCALIZATION_LAYERS,
        *,
        n_reference_prompts: int = 8,
    ) -> dict[int, torch.Tensor]:
        """The lens: each layer's Jacobian averaged over reference inputs.

        Averaging rather than linearizing at one point is what the real fitting
        procedure does, and it matters for the same reason: a single operating
        point gives a lens that is exact there and unrepresentative elsewhere.
        """
        wanted = {int(layer) for layer in layers}
        per_prompt = [
            self._jacobians_for(f"reference-{index}", wanted)
            for index in range(n_reference_prompts)
        ]
        return {
            layer: torch.stack([entry[layer] for entry in per_prompt]).mean(dim=0)
            for layer in sorted(wanted)
        }


def mock_localization_lens(
    model: MockReadoutModel | None = None,
    *,
    layers: Sequence[int] = LOCALIZATION_LAYERS,
) -> tuple[JacobianLens, MockReadoutModel]:
    """``(lens, model)`` where the lens is the model's own true Jacobian.

    Returns the model as well so callers cannot accidentally pair the lens with
    a differently seeded model — which would make every layer look wrong-layer.
    """
    model = model or MockReadoutModel()
    return (
        JacobianLens(
            jacobians=model.true_jacobians(layers),
            n_prompts=256,
            d_model=model.d_model,
        ),
        model,
    )


def mock_validation_prompts(n_prompts: int, *, offset: int = 0) -> list[str]:
    """Deterministic stand-ins for the held-out WikiText passages."""
    return [
        f"<start_of_turn>user\nContinue this passage.\n\nmock passage {index}"
        f"<end_of_turn>\n<start_of_turn>model\n"
        for index in range(offset, offset + n_prompts)
    ]


__all__ = [
    "MOCK_BLOCK_SCALE",
    "MOCK_LENS_CHECKSUM",
    "MOCK_READOUT_D_MODEL",
    "MOCK_READOUT_SEED",
    "MOCK_READOUT_VOCAB",
    "MockReadoutModel",
    "mock_localization_lens",
    "mock_validation_prompts",
]
