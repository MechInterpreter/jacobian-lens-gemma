# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""A synthetic world built to exercise the coordinate swap — and only that.

This is a *mechanism* mock, not a data mock. :mod:`jlens.mmpilot.mock` already
provides a synthetic SpokenCOCO and a synthetic Gemma for the completed pilot;
nothing here replaces it. What that world cannot do is make the *coordinate*
story legible: its readout is a dot product with a unit-norm row, so source and
target coordinates and source and target dot products coincide, and the one
thing this method turns on — that ``pinv(V) h`` is not ``V^T h`` when the lens
vectors are nonorthogonal — would be invisible.

So this world is built backwards from the algebra:

* An orthonormal frame ``Q`` of the residual space is fixed once. Every named
  quantity is a stated combination of its columns, so every expected value
  below is derived, not measured.
* The source and target lens vectors are **deliberately nonorthogonal**
  (cosine :data:`LENS_COSINE`) and **deliberately not unit norm**
  (:data:`SOURCE_NORM`, :data:`TARGET_NORM`). A dot-product read of the
  coordinates gets a visibly wrong answer here, which is the point.
* The identity readout rows are the **rows of** ``pinv(V)``, so the logit of
  ``bird`` is exactly the lens coordinate ``c_source``. Every prediction in the
  MOCK notebook is therefore a closed-form consequence of the swap.
* The property readout (``two`` / ``four`` legs) lives in directions
  orthogonal to ``span(V)``, and is *computed* at
  :data:`REASONING_LAYER` from whatever identity coordinates arrive there.
  That is what makes downstream recomputation testable rather than assumed:
  the property is not stored in the evidence, it is derived from the identity.

The layer stack, and why it is shaped this way::

    layer 0   inject evidence at the evidence-token positions only
    layers 1-3   carry (h <- CARRY_GAIN * h + drift)   <-- the primary band
    layer 4   broadcast: every position reads the mean of the evidence positions
    layer 5   reasoning: identity coordinates -> legs directions
    layers 6-7   carry                                  <-- the control band

Before layer 4 the identity lives *only* at the evidence positions, so
``all_prompt_positions`` and ``evidence_span_only`` both reach it and
``non_evidence_prompt_positions`` does not — a position control that is null by
construction rather than by luck. After layer 5 the legs answer has already
been written, so a swap over layers 6-7 changes the identity and leaves the
property alone — which is the whole reason identity replacement and downstream
recomputation are separate claims.

The bands are odd-length on purpose. Exact exchange is an involution, and these
carry blocks commute with it (they scale ``h`` and add a vector orthogonal to
``span(V)``), so an even-length band here very nearly cancels. That is a true
property of the method under a synthetic model whose blocks nearly commute with
the exchange, and :func:`band_parity_diagnostic` measures it rather than hiding
it. A real transformer's blocks do not commute with the exchange, so the real
band must be chosen from measured activations — not from this number.

**A passing MOCK run proves the implementation computes what it says it
computes. It is not evidence about Gemma, about any modality, and about no
scientific claim whatsoever.**
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from types import SimpleNamespace

import torch
from torch import nn

from jlens.mmpilot.backend import BuiltInputs, text_hash
from jlens.mmpilot.coordinate_swap import (
    ConceptToken,
    StabilityPolicy,
    build_swap_basis,
    read_coordinates,
)

MOCK_MODE = "mock"

MOCK_D_MODEL = 16
MOCK_VOCAB = 32
MOCK_N_LAYERS = 8

#: Reserved token ids. The atom index *is* the token id, exactly as in a real
#: ``W_U @ J_l`` dictionary.
PAD_ID, BOS_ID, EVIDENCE_ID, ANSWER_SUFFIX_ID = 0, 1, 2, 3
BIRD_ID, CAT_ID, TWO_ID, FOUR_ID, DOG_ID, CAR_ID = 8, 9, 10, 11, 12, 13
#: A concept whose lens vector is an exact multiple of ``bird``'s, so the
#: rank-deficiency refusal has something real to refuse.
PARALLEL_ID = 14
TEXT_TOKEN_BASE = 16

TOKEN_TEXT = {
    BIRD_ID: " bird",
    CAT_ID: " cat",
    TWO_ID: " two",
    FOUR_ID: " four",
    DOG_ID: " dog",
    CAR_ID: " car",
    PARALLEL_ID: " parallel",
}

#: Sequence layout. Three evidence placeholder tokens after BOS, then the
#: question. Mirrors how a processor expands an image or an audio clip into a
#: contiguous run of placeholder tokens.
EVIDENCE_SPAN: tuple[int, int] = (1, 4)

#: Geometry of the lens pair. Nonorthogonal and non-unit on purpose.
LENS_COSINE = 0.45
SOURCE_NORM = 0.8
TARGET_NORM = 1.3

#: Layer roles.
INJECT_LAYER = 0
BROADCAST_LAYER = 4
REASONING_LAYER = 5

#: The primary band: contiguous, before the broadcast, odd length.
PRIMARY_BAND: tuple[int, ...] = (1, 2, 3)
#: The layer-band control: contiguous, odd length, and starting at the output
#: of :data:`REASONING_LAYER` — so the property has already been computed
#: before the first patch lands. Identity moves, the property does not, which
#: is the separation the two claims rest on.
POST_REASONING_BAND: tuple[int, ...] = (5, 6, 7)
#: What the MOCK pretends passed an untouched confirmation gate. The real
#: study reads this from :data:`jlens.mmpilot.published_lens.CONFIRMED_LAYERS`.
MOCK_VALIDATED_LAYERS: tuple[int, ...] = (1, 2, 3, 5, 6, 7)

CARRY_GAIN = 1.2
EVIDENCE_AMPLITUDE = 1.0
MODALITY_AMPLITUDE = 0.35
NUISANCE_AMPLITUDE = 0.2
BROADCAST_GAIN = 1.0
REASONING_GAIN = 1.0
DRIFT_AMPLITUDE = 0.05

MOCK_MODALITIES = ("text", "image", "spoken_audio")

#: Pins the MOCK reports in place of a real model/processor revision, so a MOCK
#: artifact can never be confused with a real one.
MOCK_MODEL_REVISION = "mock-no-model"
MOCK_PROCESSOR_REVISION = "mock-no-processor"
MOCK_LENS_CHECKSUM = "sha256:mock-synthetic-lens-not-a-real-artifact"


def _frame(d_model: int = MOCK_D_MODEL) -> torch.Tensor:
    """A fixed orthonormal frame ``Q`` (``[d_model, d_model]``, columns).

    From a seeded Gaussian by QR, then sign-fixed on the diagonal of ``R`` so
    the result is bit-identical on every machine and every LAPACK build.
    """
    generator = torch.Generator().manual_seed(20260807)
    Q, R = torch.linalg.qr(torch.randn(d_model, d_model, generator=generator, dtype=torch.float64))
    return (Q * torch.sign(torch.diagonal(R))).contiguous()


class SwapMockWorld:
    """The generative story: one frame, and everything named in terms of it."""

    def __init__(self, d_model: int = MOCK_D_MODEL) -> None:
        self.d_model = d_model
        self.Q = _frame(d_model)
        sine = (1.0 - LENS_COSINE**2) ** 0.5
        # span{q0, q1} is the lens plane. Nonorthogonal, non-unit, rank 2.
        self.v_source = SOURCE_NORM * self.Q[:, 0]
        self.v_target = TARGET_NORM * (LENS_COSINE * self.Q[:, 0] + sine * self.Q[:, 1])
        # The property lives outside the lens plane, so the swap cannot touch
        # it directly and any change in it had to be recomputed.
        self.u_two = self.Q[:, 2]
        self.u_four = self.Q[:, 3]
        # An unrelated, well-conditioned pair for the unrelated-swap control.
        self.v_dog = 0.9 * self.Q[:, 8]
        self.v_car = 1.1 * (0.3 * self.Q[:, 8] + (1 - 0.09) ** 0.5 * self.Q[:, 9])
        self.modality_vectors = {
            name: self.Q[:, 4 + index] for index, name in enumerate(MOCK_MODALITIES)
        }
        self.drift = DRIFT_AMPLITUDE * self.Q[:, 7]

    # ------------------------------------------------------------- dictionary

    @property
    def V(self) -> torch.Tensor:
        """``[d_model, 2]`` — the primary pair, columns, float64."""
        return torch.stack([self.v_source, self.v_target], dim=1)

    def atoms(self) -> torch.Tensor:
        """The synthetic ``W_U @ J_l`` dictionary: ``[MOCK_VOCAB, d_model]``.

        Rows, unnormalized, no final-norm weight folded in — the repository's
        convention, restated in a place small enough to check by eye.
        """
        generator = torch.Generator().manual_seed(4242)
        atoms = 0.15 * torch.randn(
            MOCK_VOCAB, self.d_model, generator=generator, dtype=torch.float64
        )
        atoms[BIRD_ID] = self.v_source
        atoms[CAT_ID] = self.v_target
        atoms[TWO_ID] = self.u_two
        atoms[FOUR_ID] = self.u_four
        atoms[DOG_ID] = self.v_dog
        atoms[CAR_ID] = self.v_car
        atoms[PARALLEL_ID] = 2.0 * self.v_source  # exactly parallel: rank 1
        return atoms

    def atoms_by_layer(self, layers: Sequence[int]) -> dict[int, torch.Tensor]:
        """The same frozen dictionary at every layer.

        A real study has a different ``J_l`` per layer; the MOCK does not
        pretend otherwise, and reusing one dictionary keeps every expected value
        in the notebook closed-form. What it *does* exercise is that a basis is
        built and diagnosed per layer, and that each layer's hook reads its own
        activation.
        """
        atoms = self.atoms()
        return {int(layer): atoms.clone() for layer in layers}

    # --------------------------------------------------------------- evidence

    def nuisance(self, key: str) -> torch.Tensor:
        seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") % (2**31 - 1)
        generator = torch.Generator().manual_seed(seed)
        raw = torch.randn(self.d_model, generator=generator, dtype=torch.float64)
        # Project out the lens plane and the property directions so a sample's
        # nuisance can never masquerade as concept or property content.
        for direction in (self.Q[:, 0], self.Q[:, 1], self.u_two, self.u_four):
            raw = raw - direction * raw.dot(direction)
        return raw / raw.norm()

    def evidence(self, concept: str, modality: str, *, nuisance_key: str = "") -> torch.Tensor:
        """Concept content that is **identical across modalities** by design."""
        concept_vector = {"bird": self.v_source, "cat": self.v_target}[concept]
        return (
            EVIDENCE_AMPLITUDE * concept_vector
            + MODALITY_AMPLITUDE * self.modality_vectors[modality]
            + NUISANCE_AMPLITUDE * self.nuisance(f"{concept}|{modality}|{nuisance_key}")
        )

    # ----------------------------------------------------------------- readout

    def duals(self) -> torch.Tensor:
        """``pinv(V)`` — ``[2, d_model]``. Rows are the identity readout."""
        V = self.V
        return torch.linalg.solve(V.T @ V, V.T)

    def coordinates(self, h: torch.Tensor) -> torch.Tensor:
        return read_coordinates(h, self.V)


# ------------------------------------------------------------------ the model


class _SwapMockBlock(nn.Module):
    """One decoder block. Its role is fixed by its index, never learned."""

    def __init__(self, index: int, world: SwapMockWorld) -> None:
        super().__init__()
        self.index = index
        self.world = world
        self.register_buffer("V", world.V.to(torch.float32), persistent=False)
        self.register_buffer("u_two", world.u_two.to(torch.float32), persistent=False)
        self.register_buffer("u_four", world.u_four.to(torch.float32), persistent=False)
        self.register_buffer("drift", world.drift.to(torch.float32), persistent=False)

    def forward(self, hidden: torch.Tensor, evidence=None, evidence_span=None):
        if self.index == INJECT_LAYER:
            out = hidden.clone()
            if evidence is not None:
                start, end = evidence_span
                out[:, start:end] = out[:, start:end] + evidence.reshape(
                    hidden.shape[0], 1, -1
                )
            return out
        if self.index == BROADCAST_LAYER:
            start, end = evidence_span
            pooled = hidden[:, start:end].mean(dim=1, keepdim=True)
            return hidden + BROADCAST_GAIN * pooled
        if self.index == REASONING_LAYER:
            V = self.V
            coordinates = torch.linalg.solve(
                V.T @ V, (hidden @ V).transpose(-1, -2)
            ).transpose(-1, -2)
            legs = torch.relu(coordinates[..., :1]) * self.u_two + torch.relu(
                coordinates[..., 1:2]
            ) * self.u_four
            return hidden + REASONING_GAIN * legs
        return CARRY_GAIN * hidden + self.drift


class _SwapMockTextModel(nn.Module):
    def __init__(self, world: SwapMockWorld, n_layers: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(MOCK_VOCAB, world.d_model)
        with torch.no_grad():
            generator = torch.Generator().manual_seed(7)
            self.embed_tokens.weight.copy_(
                0.02 * torch.randn(MOCK_VOCAB, world.d_model, generator=generator)
            )
        self.layers = nn.ModuleList(_SwapMockBlock(i, world) for i in range(n_layers))
        self.norm = nn.Identity()

    def forward(self, input_ids, evidence=None, evidence_span=None, **_kw):
        hidden = self.embed_tokens(input_ids)
        for block in self.layers:
            hidden = block(hidden, evidence=evidence, evidence_span=evidence_span)
        return SimpleNamespace(last_hidden_state=self.norm(hidden))


class _SwapMockInner(nn.Module):
    def __init__(self, world: SwapMockWorld, n_layers: int) -> None:
        super().__init__()
        self.language_model = _SwapMockTextModel(world, n_layers)


class SwapMockModel(nn.Module):
    """Gemma 4's module layout with an analytically clean, closed-form readout.

    ``lm_head`` row ``bird`` is row 0 of ``pinv(V)``, so ``logit(bird)`` *is*
    the lens coordinate ``c_source``; likewise ``cat``. Rows ``two`` and
    ``four`` invert the property directions, so their logits are exactly the
    coefficients the reasoning layer wrote. Every other row is zero, so the
    candidates only ever compete with each other.
    """

    def __init__(self, world: SwapMockWorld, *, n_layers: int = MOCK_N_LAYERS) -> None:
        super().__init__()
        self.world = world
        self.model = _SwapMockInner(world, n_layers)
        self.lm_head = nn.Linear(world.d_model, MOCK_VOCAB, bias=False)
        duals = world.duals().to(torch.float32)
        with torch.no_grad():
            self.lm_head.weight.zero_()
            self.lm_head.weight[BIRD_ID] = duals[0]
            self.lm_head.weight[CAT_ID] = duals[1]
            self.lm_head.weight[TWO_ID] = world.u_two.to(torch.float32)
            self.lm_head.weight[FOUR_ID] = world.u_four.to(torch.float32)
        self.config = SimpleNamespace(
            get_text_config=lambda: SimpleNamespace(
                hidden_size=world.d_model,
                num_hidden_layers=n_layers,
                vocab_size=MOCK_VOCAB,
            ),
            image_token_id=EVIDENCE_ID,
            audio_token_id=EVIDENCE_ID,
        )
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, input_ids=None, evidence=None, evidence_span=None, **_kwargs):
        hidden = self.model.language_model(
            input_ids=input_ids, evidence=evidence, evidence_span=evidence_span
        )
        return SimpleNamespace(logits=self.lm_head(hidden.last_hidden_state))


#: The two questions. Byte-identical evidence, byte-identical swap; only the
#: question changes, which is the flexible-generalization design.
IDENTITY_QUESTION = "Question: what do you see or hear: bird, cat? Answer:"
PROPERTY_QUESTION = "Question: how many legs does it have: two, four? Answer:"

IDENTITY_CANDIDATES = ("bird", "cat")
PROPERTY_CANDIDATES = ("two", "four")


class SwapMockBackend:
    """A :class:`~jlens.mmpilot.backend.PilotBackend` over :class:`SwapMockModel`.

    Evidence occupies a contiguous placeholder span reported as
    ``modality_token_range``, so the position rules that need an evidence span
    have a real one to resolve against instead of a fabricated index.
    """

    def __init__(self, world: SwapMockWorld | None = None, *, n_layers: int = MOCK_N_LAYERS) -> None:
        self.world = world or SwapMockWorld()
        self.hf_model = SwapMockModel(self.world, n_layers=n_layers)
        self.d_model = self.world.d_model
        self.n_layers = n_layers
        self.answer_tokens = {
            "bird": BIRD_ID, "cat": CAT_ID, "two": TWO_ID, "four": FOUR_ID,
        }

    @property
    def blocks(self):
        return self.hf_model.model.language_model.layers

    def supports(self, modality: str) -> bool:
        if modality in MOCK_MODALITIES:
            return True
        raise ValueError(f"unknown modality {modality!r}")

    def unembedding_weight(self) -> torch.Tensor:
        return self.hf_model.lm_head.weight

    def encode_candidate(self, text: str) -> list[int]:
        stripped = text.strip()
        if stripped in self.answer_tokens:
            # Two tokens on purpose: complete-sequence scoring must be exercised,
            # and the appended position must be provably never patched.
            return [self.answer_tokens[stripped], ANSWER_SUFFIX_ID]
        return self._tokenize(stripped)

    def encode_token(self, text: str) -> list[int]:
        """The plain tokenizer, without the MOCK's deliberate answer suffix.

        Concept resolution asks "which vocabulary row is this concept's lens
        vector", which is a tokenizer question; candidate scoring asks "what
        complete continuation should be teacher-forced", which is not. On a
        real backend both go through the same tokenizer and
        ``encode_candidate`` answers either. Here they must differ, because the
        MOCK appends :data:`ANSWER_SUFFIX_ID` to every candidate on purpose —
        and a single-token *concept* whose *candidate* is two tokens is exactly
        the situation the real study will be in.
        """
        stripped = text.strip()
        if stripped in self.answer_tokens:
            return [self.answer_tokens[stripped]]
        return self._tokenize(stripped)

    def _tokenize(self, text: str) -> list[int]:
        ids = []
        for word in text.split():
            digest = hashlib.sha256(word.encode()).digest()
            ids.append(TEXT_TOKEN_BASE + digest[0] % (MOCK_VOCAB - TEXT_TOKEN_BASE))
        return ids or [PAD_ID]

    def build_inputs(
        self,
        *,
        prompt: str,
        modality: str,
        concept: str = "bird",
        nuisance_key: str = "",
        image=None,
        audio=None,
        sampling_rate: int | None = None,
        media_path: str | None = None,
    ) -> BuiltInputs:
        """Build one input. ``concept`` selects which evidence arrives.

        The three modalities differ only in the modality offset added to the
        *same* concept vector — the MOCK's claim that a concept is shared
        across channels is true by construction, which is precisely why a MOCK
        pass says nothing about whether it is true in Gemma.
        """
        if not self.supports(modality):  # pragma: no cover - defensive
            raise ValueError(f"unsupported modality {modality!r}")
        start, end = EVIDENCE_SPAN
        ids = [BOS_ID, *([EVIDENCE_ID] * (end - start)), *self._tokenize(prompt)]
        evidence = self.world.evidence(concept, modality, nuisance_key=nuisance_key)
        return BuiltInputs(
            tensors={
                "input_ids": torch.tensor([ids], dtype=torch.long),
                "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
                "evidence": evidence.to(torch.float32).reshape(1, 1, -1),
                "evidence_span": EVIDENCE_SPAN,
            },
            prompt_len=len(ids),
            modality=modality,
            prompt_hash=text_hash(f"{prompt}|{modality}|{concept}"),
            media_checksum=None,
            route={"route": "mock_coordinate_swap"},
            modality_token_range=[start, end],
        )

    @torch.no_grad()
    def forward_logits(self, tensors: Mapping) -> torch.Tensor:
        payload = {k: v for k, v in dict(tensors).items() if k != "use_cache"}
        return self.hf_model(**payload).logits.float()


# ------------------------------------------------------------------ helpers


def mock_concept_tokens(backend: SwapMockBackend) -> dict[str, ConceptToken]:
    """Resolve every MOCK concept through the backend's own tokenizer.

    Uses the same single-token resolution the real path uses (and the same
    refusal), against :meth:`SwapMockBackend.encode_token`.
    """
    from jlens.mmpilot.coordinate_swap import resolve_concept_token

    return {
        name: resolve_concept_token(backend.encode_token, name)
        for name in ("bird", "cat", "two", "four", "dog", "car")
    }


def mock_bases(
    world: SwapMockWorld,
    *,
    layers: Sequence[int],
    source: ConceptToken,
    target: ConceptToken,
    policy: StabilityPolicy | None = None,
) -> dict[int, object]:
    """One :class:`~jlens.mmpilot.coordinate_swap.SwapBasis` per layer."""
    atoms = world.atoms_by_layer(layers)
    kwargs = {} if policy is None else {"policy": policy}
    return {
        int(layer): build_swap_basis(
            atoms[layer], layer=int(layer), source=source, target=target, **kwargs
        )
        for layer in layers
    }


def mock_lens_checksums(layers: Sequence[int]) -> dict[int, str]:
    """Per-layer lens checksums, visibly labelled as synthetic."""
    return {int(layer): f"{MOCK_LENS_CHECKSUM}:L{int(layer)}" for layer in layers}


def band_parity_diagnostic(
    backend: SwapMockBackend,
    inputs: BuiltInputs,
    *,
    source: ConceptToken,
    target: ConceptToken,
    max_band_length: int = 4,
    start_layer: int = 1,
) -> list[dict]:
    """Measure the involution directly: identity outcome vs band length.

    Exact exchange is its own inverse, and this world's carry blocks nearly
    commute with it, so an even-length band nearly cancels. Reported as a
    number rather than described, because it is the one MOCK result with a
    direct bearing on how a real band is chosen.
    """
    from jlens.mmpilot.capability import (
        prediction_and_margin,
        score_candidate_sequences,
    )
    from jlens.mmpilot.coordinate_swap import coordinate_swap_band

    candidate_ids = {
        name: backend.encode_candidate(f" {name}") for name in IDENTITY_CANDIDATES
    }
    clean = score_candidate_sequences(backend, inputs, candidate_ids)
    rows = []
    for length in range(1, max_band_length + 1):
        layers = tuple(range(start_layer, start_layer + length))
        bases = mock_bases(backend.world, layers=layers, source=source, target=target)
        with coordinate_swap_band(
            backend.blocks,
            bases,
            alpha=1.0,
            prompt_len=inputs.prompt_len,
            position_rule="all_prompt_positions",
            record_coordinates=False,
        ):
            scores = score_candidate_sequences(backend, inputs, candidate_ids)
        verdict = prediction_and_margin(scores, target.concept)
        rows.append(
            {
                "band": list(layers),
                "band_length": length,
                "parity": "odd" if length % 2 else "even",
                "clean_prediction": prediction_and_margin(clean, target.concept)["prediction"],
                "prediction": verdict["prediction"],
                "swapped_to_target": verdict["prediction"] == target.concept,
            }
        )
    return rows
