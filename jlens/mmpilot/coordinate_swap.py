# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Exact two-coordinate J-lens patching — the paper's coordinate swap.

This module implements *one* intervention, the one defined in
https://transformer-circuits.pub/2026/workspace/index.html ("Technical details
of J-lens use cases"). Given the layer-``l`` J-lens vectors of a source token
and a target token::

    V = [v_source  v_target]            # [d_model, 2], columns
    c = pinv(V) @ h                     # [2], the measured lens coordinates
    h_patched = h + alpha * V @ (sigma(c) - c)

where ``sigma`` exchanges the two entries of ``c``. At ``alpha = 1`` this is a
*measured* exchange of the two coordinates, and the component of ``h``
orthogonal to ``span{v_source, v_target}`` is unchanged exactly.

What this is **not**
===================

Three different things are easy to confuse, and this repository now contains
all three. They are not interchangeable and must never share an artifact.

1. **Source-derived J-space steering** — :mod:`jlens.mmpilot.causal`, the
   method behind the completed three-modality causal result. A concept
   direction is estimated from *source-modality training examples* as
   ``v = V ReLU(mean_positive_code - mean_negative_code)``, normalized, and
   added or subtracted: ``h' = h +- alpha * v_concept``. It is a valid causal
   steering experiment. It is not a coordinate swap, and describing it as one
   would misstate what was measured.

2. **Single-vector J-lens steering** — ``h' = h + alpha * (v_target -
   v_source)``. This uses the lens vectors but assumes unit coefficients: it
   adds a fixed multiple of a fixed direction regardless of how much source
   content ``h`` actually carries.

3. **Exact two-coordinate patching** — this module. The update direction is
   also parallel to ``v_source - v_target`` (see :func:`swap_coordinates` for
   the algebra), but its *coefficient is measured from* ``h``:

       V @ (sigma(c) - c) = (c_target - c_source) * (v_source - v_target)

   An activation with no source content is left almost alone; an activation
   saturated with it is fully rewritten. That measured coefficient is the
   whole difference between (2) and (3), and it is why the coordinates must
   be read with a pseudoinverse rather than with dot products: ``v_source``
   and ``v_target`` are rows of ``W_U @ J_l`` and are in general neither
   orthogonal nor unit norm, so ``v_source . h`` is not ``c_source``.

Conventions this module assumes, all verified against the repository
=====================================================================

- **Vectors.** A J-lens vector is a *row* of ``W_U @ J_l`` —
  ``JSpaceDictionary.atoms[token_id]``, shape ``[d_model]``
  (:mod:`jlens.pursuit`). Token id indexes the row directly, so the dictionary
  atom index *is* the vocabulary token id.
- **Normalization.** Raw, unnormalized. ``JSpaceDictionary`` stores
  ``atom_norms`` separately and only ever uses them to rescale *selection*
  correlations; coefficients always refer to raw atoms. Nothing is normalized
  here either — normalizing ``V``'s columns would change ``c`` and therefore
  change what "exchange the coordinates" means.
- **Final norm weights.** Not folded in. The pilot builds its dictionaries
  with ``final_norm_weight=None`` (:func:`jlens.mmpilot.jspace.build_dictionary`)
  and records ``final_norm_weight_folded: False``. This module records the same
  flag and refuses a basis built under the other convention unless told
  explicitly.
- **Hook site.** The output of ``model.language_model.layers[l]`` — the same
  tensor the lens reads, the pursuit decomposes, and
  :func:`jlens.interventions.residual_intervention` edits.

Numerical policy
================

The 2x2 solve is done in **float64** regardless of the model's dtype. ``c`` is
obtained from the normal equations ``(V^T V) c = V^T h`` rather than from a
full pseudoinverse: for a rank-2 ``V`` these agree exactly in exact
arithmetic, and the 2x2 Gram matrix is the only object whose conditioning
matters. Both the rank and the condition number of ``V`` are measured from its
singular values *before* any solve, and a rank-deficient or ill-conditioned
pair is **refused** rather than regularized (see :class:`StabilityPolicy`).
At the hook boundary the patched activation is cast back to the block output's
own dtype and device, so the model never sees float64.

``alpha`` semantics
===================

- ``alpha = 0`` — exact no-op. The update is computed in full (so the whole
  code path is exercised) and multiplied by exactly zero.
- ``alpha = 1`` — the full, measured coordinate exchange. Applying it twice
  returns the original activation.
- ``0 < alpha < 1`` — interpolation between the two coordinate assignments.
- ``alpha > 1`` — **amplified extrapolation, not an exchange.** The
  coordinates land at ``c + alpha (sigma(c) - c)``, which overshoots
  ``sigma(c)``. The paper's "double strength" swap is this case. It must be
  labelled as extrapolation in every artifact, never as "a stronger swap".

The prompt this is measured through
===================================

The completed steering study asked *"which one of these is present: bird, cat,
giraffe, microwave, toilet, zebra?"* — a valid question, and one that puts every
candidate, including what would be a swap target, into the model's own input.
The primary coordinate-swap study may not be asked that way: identity
replacement means the model produces a concept it was never shown. So the
primary protocol is an **open prompt** built by
:mod:`jlens.mmpilot.prompt_protocol`, with the candidate answers supplied only
to the external teacher-forced scorer, and
:func:`assert_open_prompt_protocol` refuses any other configuration. The
candidate-listed prompt remains available as a labelled comparison condition.

An involution, and what that does and does not cost a layer band
===============================================================

Exact exchange is an involution: applying **one fixed update** twice cancels.
That is a statement about one update, and it does not forbid a band. Across a
layer band the swap is *recomputed from each layer's own activation*, against
that layer's own ``W_U @ J_l``, so two consecutive band layers cancel only to
the extent that their bases *and* their coordinates agree. In a real
transformer they do not; in a synthetic model whose blocks commute with the
exchange they do, and an even-length band is then close to a no-op. The MOCK
notebook measures that explicitly. It is a property of the method under a
commuting stack, not a bug, and the real band must be chosen with it in mind.

What a band does require is a confirmed lens at **every** physical layer inside
it — see :func:`build_layer_band`, and :mod:`jlens.mmpilot.band_swap` for the
contiguous-band study built on it.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass, field

import torch
from torch import nn

from jlens.mmpilot.store import payload_checksum

#: Bound into every artifact. A change here refuses to resume anything written
#: under the older algebra.
METHOD_VERSION = "jlens.mmpilot.coordinate_swap.v1"

#: Optional model-dtype realization used by new studies. The historical v1
#: path remains the default so completed artifacts keep their original meaning.
MODEL_DTYPE_REALIZATION_VERSION = (
    "jlens.mmpilot.coordinate_swap.model_dtype_realization.v1"
)

#: The intervention family. The completed multimodal result carries
#: :data:`STEERING_FAMILY`; nothing may carry both.
INTERVENTION_FAMILY = "anthropic_coordinate_swap"

#: What the completed three-modality causal result used
#: (:mod:`jlens.mmpilot.causal`). Named here only so a resume can refuse it.
STEERING_FAMILY = "source_derived_jspace_steering"

#: Exactly what the columns of ``V`` are, recorded in every spec.
VECTOR_CONVENTION = (
    "column j of V is row token_id_j of W_U @ J_l "
    "(jlens.pursuit.JSpaceDictionary.atoms[token_id]); shape [d_model, 2]"
)

#: Exactly what was and was not done to them.
NORMALIZATION_CONVENTION = (
    "raw atoms, unnormalized, final-norm weight not folded "
    "(final_norm_weight_folded=False)"
)

#: How ``c`` is obtained.
SOLVE_POLICY = (
    "c solves the float64 normal equations (V^T V) c = V^T h via "
    "torch.linalg.solve; rank and condition number are measured from "
    "torch.linalg.svdvals(V) before the solve and a failing pair is refused, "
    "never regularized"
)


@dataclass(frozen=True)
class ModelDtypeRealizationPolicy:
    """Make intended coordinates true in the dtype the model consumes.

    A float64 coordinate exchange can move after it is rounded to bf16. This
    policy performs bounded corrections in ``span(V)`` and re-audits the cast
    tensor after every correction. It never changes alpha or adds a component
    outside the two-coordinate subspace.
    """

    max_corrections: int = 8
    relative_coordinate_tolerance: float = 0.02
    relative_residual_tolerance: float = 0.02
    minimum_scale: float = 1.0

    def __post_init__(self) -> None:
        if int(self.max_corrections) < 0:
            raise ValueError("max_corrections must be non-negative")
        for name in (
            "relative_coordinate_tolerance",
            "relative_residual_tolerance",
            "minimum_scale",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive, got {value}")

    def to_dict(self) -> dict:
        return {
            "version": MODEL_DTYPE_REALIZATION_VERSION,
            "max_corrections": int(self.max_corrections),
            "relative_coordinate_tolerance": float(
                self.relative_coordinate_tolerance
            ),
            "relative_residual_tolerance": float(self.relative_residual_tolerance),
            "minimum_scale": float(self.minimum_scale),
        }

#: Prompt protocols the **primary** coordinate-swap study may run under. The
#: swap target must be absent from everything the model can see, which a prompt
#: that enumerates every candidate cannot promise. The legacy candidate-listed
#: protocol is admissible only as a labelled comparison condition, never as the
#: primary claim. See :mod:`jlens.mmpilot.prompt_protocol`.
PRIMARY_PROMPT_PROTOCOLS: tuple[str, ...] = (
    "mmpilot.open_animal_identification.v1",
    "mmpilot.open_entity_identification.v1",
    "mmpilot.open_animal_legs.v1",
    "mmpilot.hidden_animal_legs.v1",
)

#: The rule that keeps teacher-forced candidate tokens out of every patch.
PROMPT_BOUNDARY_RULE = (
    "positions p are patched only when 0 <= p < prompt_len; positions "
    ">= prompt_len are teacher-forced candidate-completion tokens appended by "
    "jlens.mmpilot.capability.score_candidate_sequences and are never patched"
)

#: Position-selection rules. ``all_prompt_positions`` is the primary,
#: paper-faithful protocol ("at every token position"); it includes multimodal
#: evidence positions. ``final_prompt_token_only`` exists **only** as a
#: labelled comparison to the completed pilot's protocol and is never the
#: primary method. The two ``evidence`` rules are position controls.
POSITION_RULES = (
    "all_prompt_positions",
    "final_prompt_token_only",
    "evidence_span_only",
    "non_evidence_prompt_positions",
)

#: The primary rule, named once so a notebook typo cannot quietly demote it.
PRIMARY_POSITION_RULE = "all_prompt_positions"

#: Control conditions for the planned study. Every one of them is a
#: *configuration*, run through the same code path as the real condition.
CONTROL_KINDS = (
    # The condition under test.
    "coordinate_swap",
    # alpha = 0 through the full path.
    "zero",
    # Two random directions, norms matched to v_source / v_target, same algebra.
    "random_two_direction_norm_matched",
    # A source/target pair unrelated to the evidence.
    "unrelated_pair_swap",
    # The same pair with the roles exchanged (see reverse_basis).
    "reverse_swap",
    # The completed pilot's baselines, unchanged, for comparison only.
    "raw_residual_difference",
    "source_derived_jspace_steering",
    # Inserting the downstream answer's lens vector directly.
    "direct_answer_vector",
    # Same swap, positions that carry no evidence.
    "position_control",
    # Same swap, a different layer band.
    "layer_band_control",
)


class CoordinateSwapError(ValueError):
    """Invalid coordinate-swap request, basis, or stored state."""


class MultiTokenConceptError(CoordinateSwapError):
    """A concept does not resolve to exactly one token under this tokenizer."""


class RankDeficientPairError(CoordinateSwapError):
    """``V``'s two columns are (numerically) parallel — there is no 2-D span."""


class IllConditionedPairError(CoordinateSwapError):
    """``V`` has rank 2 but the coordinates cannot be read reliably."""


class LayerBandError(CoordinateSwapError):
    """A requested band is non-contiguous or contains an unvalidated layer."""


def _finite_or_raise(tensor: torch.Tensor, what: str) -> None:
    if not bool(torch.isfinite(tensor).all()):
        raise CoordinateSwapError(f"{what} contains NaN/Inf")


def _assert_frozen(module: nn.Module) -> None:
    for name, parameter in module.named_parameters():
        if parameter.requires_grad:
            raise CoordinateSwapError(
                f"model parameter {name} requires grad; freeze the model before "
                f"intervening"
            )


def tensor_checksum(tensor: torch.Tensor) -> str:
    """``sha256:`` over the float64 bytes — exact and order-stable.

    float64 rather than :func:`jlens.mmpilot.jspace.tensor_checksum`'s float32
    because a basis is stored in float64 here and a checksum must not depend on
    a lossy cast.
    """
    contiguous = tensor.detach().to(torch.float64).cpu().contiguous()
    return "sha256:" + hashlib.sha256(contiguous.numpy().tobytes()).hexdigest()


# --------------------------------------------------------------- token resolution


@dataclass(frozen=True)
class ConceptToken:
    """A concept and the single vocabulary token it resolved to.

    Attributes:
        concept: The concept as the experiment names it (``"bird"``).
        token_id: The vocabulary id, which is also the dictionary atom index.
        token_text: The exact string that was encoded (``" bird"``).
        variant: The variant template that produced it.
    """

    concept: str
    token_id: int
    token_text: str
    variant: str

    def to_dict(self) -> dict:
        return asdict(self)


#: Tried in order. The leading-space form is first because after ``Answer:``
#: the model continues with ``" bird"`` — the same reasoning as
#: :func:`jlens.mmpilot.capability.candidate_token_ids`.
DEFAULT_TOKEN_VARIANTS: tuple[str, ...] = (" {}", "{}")


def resolve_concept_token(
    encode: Callable[[str], Sequence[int]],
    concept: str,
    *,
    variants: Sequence[str] = DEFAULT_TOKEN_VARIANTS,
) -> ConceptToken:
    """Resolve ``concept`` to exactly one token, or refuse.

    Args:
        encode: ``str -> [token ids]`` with no special tokens — for the pilot
            backends this is ``backend.encode_candidate``.
        concept: The concept name.
        variants: Format strings tried in order. The **first** one that encodes
            to exactly one token wins.

    Raises:
        MultiTokenConceptError: If no variant is single-token. The message
            lists every variant with its token count and ids. Nothing is
            truncated: using only a multi-token concept's first token would
            silently intervene on a *different* concept ("straw" for
            "strawberry"), so it is refused instead.

    Multi-token concepts are a real limitation of this version, not an
    oversight. Extending the method to them means choosing what ``V`` should
    be for a several-token concept, which is a methodological decision the
    paper does not make for us.
    """
    attempts: list[dict] = []
    for variant in variants:
        text = variant.format(concept)
        ids = [int(i) for i in encode(text)]
        attempts.append({"variant": variant, "text": text, "n_tokens": len(ids), "token_ids": ids})
        if len(ids) == 1:
            return ConceptToken(
                concept=concept, token_id=ids[0], token_text=text, variant=variant
            )
    detail = "; ".join(
        f"{a['text']!r} -> {a['n_tokens']} tokens {a['token_ids']}" for a in attempts
    )
    raise MultiTokenConceptError(
        f"concept {concept!r} does not resolve to a single token under this "
        f"tokenizer ({detail}). This version of the coordinate swap requires "
        f"exactly one token per concept; it will not use a prefix of a "
        f"multi-token concept, because that intervenes on a different concept."
    )


#: Matches :data:`jlens.mmpilot.convergence.READOUT_FIRST_TOKEN`: the same
#: honest label for the same situation, so a report reader who knows one
#: recognizes the other.
ANSWER_READOUT_FIRST_TOKEN = "first_token_only_diagnostic"


def resolve_answer_readout_token(
    encode: Callable[[str], Sequence[int]],
    answer: str,
    *,
    variants: Sequence[str] = DEFAULT_TOKEN_VARIANTS,
) -> ConceptToken:
    """Resolve ``answer`` to the token a readout direction is built from.

    This is **not** :func:`resolve_concept_token` and must never replace it for
    a swap concept. That function's single-token requirement exists because a
    multi-token concept has no well-defined two-coordinate subspace — using a
    prefix would swap a *different* concept ("straw" for "strawberry").

    A downstream **answer** surface (the direct-answer positive control's
    target, e.g. ``"meow"``) is not a swap concept: no subspace is built from
    it, and the greedy generation this control is scored against must itself
    emit the answer's first token before any later token matters. Falling back
    to that first token when the surface is not single-token is the same
    honest choice :func:`jlens.mmpilot.convergence.candidate_tokenization`
    already makes for readout scoring, labelled the same way, and it is
    recorded on the returned token via ``variant`` starting with ``"first:"``
    so every downstream artifact can tell an exact match from a first-token
    fallback.

    Args:
        encode: Same contract as :func:`resolve_concept_token`.
        answer: The answer surface, e.g. ``"meow"``.
        variants: Tried in order; the first variant that encodes to at least
            one token wins, taking only its leading token if there is more
            than one.

    Raises:
        CoordinateSwapError: If no variant encodes to at least one token.
    """
    for variant in variants:
        text = variant.format(answer)
        ids = [int(i) for i in encode(text)]
        if not ids:
            continue
        if len(ids) == 1:
            return ConceptToken(
                concept=answer, token_id=ids[0], token_text=text, variant=variant
            )
        return ConceptToken(
            concept=answer,
            token_id=ids[0],
            token_text=text,
            variant=f"first:{variant}",
        )
    raise CoordinateSwapError(
        f"answer {answer!r} encodes to zero tokens under every variant "
        f"{list(variants)}; no readout direction can be built from it"
    )


# ------------------------------------------------------------ stability policy


@dataclass(frozen=True)
class StabilityPolicy:
    """Documented, configurable limits on the pair ``V``.

    Attributes:
        max_condition_number: Refuse when ``s_max / s_min`` exceeds this. The
            default of 1e4 keeps the float64 solve's relative error many orders
            of magnitude below the float32 activations it is applied to, while
            still admitting genuinely correlated concept pairs.
        rank_tolerance: Relative singular-value floor. A pair is rank-deficient
            when ``s_min <= rank_tolerance * s_max``, which is the standard
            numerical rank test and the same criterion a pseudoinverse would
            use internally — except that here it refuses instead of silently
            dropping to a rank-1 solution.
        solve_dtype: The dtype the 2x2 solve runs in. float64 always; exposed
            so the choice is recorded rather than implicit.
    """

    max_condition_number: float = 1.0e4
    rank_tolerance: float = 1.0e-8
    solve_dtype: str = "float64"

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def torch_solve_dtype(self) -> torch.dtype:
        if self.solve_dtype != "float64":
            raise CoordinateSwapError(
                f"solve_dtype must be 'float64', got {self.solve_dtype!r}; the "
                "2x2 solve is not run in a lower precision"
            )
        return torch.float64


DEFAULT_STABILITY = StabilityPolicy()


# ------------------------------------------------------------------- the basis


def assert_vector_orientation(V: torch.Tensor, *, d_model: int) -> None:
    """Refuse a transposed or mis-shaped ``V``.

    ``V`` must be ``[d_model, 2]`` — lens vectors in *columns*. A ``[2,
    d_model]`` matrix is the single most likely orientation mistake, it makes
    the shapes of ``V @ (sigma(c) - c)`` fail only sometimes, and for
    ``d_model == 2`` it would not fail at all. Checked explicitly.
    """
    if V.ndim != 2:
        raise CoordinateSwapError(f"V must be 2-D [d_model, 2], got {tuple(V.shape)}")
    rows, cols = V.shape
    if (rows, cols) == (2, d_model) and d_model != 2:
        raise CoordinateSwapError(
            f"V is {tuple(V.shape)}, which is [2, d_model]: the lens vectors are "
            f"in rows, not columns. Pass V.T — column j must be the j-th lens "
            f"vector ({VECTOR_CONVENTION})."
        )
    if (rows, cols) != (d_model, 2):
        raise CoordinateSwapError(
            f"V must be [{d_model}, 2], got {tuple(V.shape)}"
        )


@dataclass(frozen=True)
class SwapBasis:
    """One layer's ``V = [v_source, v_target]`` plus its stability diagnostics.

    Attributes:
        layer: The decoder block whose output this basis reads and patches.
        source / target: The resolved concept tokens.
        V: ``[d_model, 2]`` float64 on CPU. Columns, raw atoms.
        diagnostics: Norms, cosine, singular values, condition number, rank,
            and the checksum of ``V``.
        final_norm_weight_folded: Recorded from the dictionary it came from.
    """

    layer: int
    source: ConceptToken
    target: ConceptToken
    V: torch.Tensor
    diagnostics: dict
    final_norm_weight_folded: bool = False
    kind: str = "coordinate_swap"

    @property
    def d_model(self) -> int:
        return int(self.V.shape[0])

    def to_dict(self) -> dict:
        """JSON-safe description. The vectors themselves are *not* included —
        only their checksum — so an artifact stays small and a basis can never
        be reconstructed from a record and mistaken for the lens."""
        return {
            "layer": int(self.layer),
            "kind": self.kind,
            "source": self.source.to_dict(),
            "target": self.target.to_dict(),
            "vector_convention": VECTOR_CONVENTION,
            "normalization": NORMALIZATION_CONVENTION,
            "final_norm_weight_folded": bool(self.final_norm_weight_folded),
            "diagnostics": dict(self.diagnostics),
        }


def _column_pair(
    v_source: torch.Tensor, v_target: torch.Tensor, *, dtype: torch.dtype
) -> torch.Tensor:
    for name, vector in (("v_source", v_source), ("v_target", v_target)):
        if vector.ndim != 1:
            raise CoordinateSwapError(
                f"{name} must be a 1-D [d_model] lens vector (a row of W_U @ J_l), "
                f"got shape {tuple(vector.shape)}"
            )
    if v_source.shape != v_target.shape:
        raise CoordinateSwapError(
            f"v_source is {tuple(v_source.shape)} but v_target is "
            f"{tuple(v_target.shape)}"
        )
    return torch.stack(
        [v_source.detach().to(dtype).cpu(), v_target.detach().to(dtype).cpu()], dim=1
    )


def basis_diagnostics(V: torch.Tensor, *, policy: StabilityPolicy = DEFAULT_STABILITY) -> dict:
    """Rank, condition number, norms and cosine of a ``[d_model, 2]`` pair."""
    singular = torch.linalg.svdvals(V.to(policy.torch_solve_dtype))
    s_max, s_min = float(singular[0]), float(singular[1])
    condition = math.inf if s_min == 0.0 else s_max / s_min
    norms = V.norm(dim=0)
    denominator = float(norms[0]) * float(norms[1])
    cosine = float(V[:, 0].dot(V[:, 1])) / denominator if denominator > 0 else 0.0
    rank = int(sum(1 for value in (s_max, s_min) if value > policy.rank_tolerance * s_max))
    return {
        "d_model": int(V.shape[0]),
        "source_norm": float(norms[0]),
        "target_norm": float(norms[1]),
        "cosine": cosine,
        "singular_values": [s_max, s_min],
        "condition_number": condition,
        "numerical_rank": rank,
        "rank_tolerance": policy.rank_tolerance,
        "max_condition_number": policy.max_condition_number,
        "solve_policy": SOLVE_POLICY,
        "solve_dtype": policy.solve_dtype,
        "V_checksum": tensor_checksum(V),
    }


def assert_stable(diagnostics: Mapping, *, policy: StabilityPolicy = DEFAULT_STABILITY) -> None:
    """Refuse a pair that cannot carry a reliable two-coordinate read.

    Raises:
        RankDeficientPairError: ``V`` is numerically rank 1 — the two lens
            vectors are parallel, ``span(V)`` is a line, and "exchange the two
            coordinates" has no meaning. Regularizing would invent a second
            direction that the lens does not have.
        IllConditionedPairError: rank 2 but ``s_max / s_min`` exceeds the
            policy. The coordinates exist but small activation differences move
            them arbitrarily far, so the "measured" coefficient would be noise.
    """
    if int(diagnostics["numerical_rank"]) < 2:
        raise RankDeficientPairError(
            f"the source and target lens vectors are numerically parallel "
            f"(singular values {diagnostics['singular_values']}, cosine "
            f"{diagnostics['cosine']:.6f}); span(V) is one-dimensional and there "
            f"are no two coordinates to exchange. Refusing rather than "
            f"regularizing."
        )
    condition = float(diagnostics["condition_number"])
    limit = float(diagnostics["max_condition_number"])
    if condition > limit:
        raise IllConditionedPairError(
            f"condition number {condition:.4g} exceeds the configured limit "
            f"{limit:.4g} (cosine {diagnostics['cosine']:.6f}); the coordinate "
            f"read would be dominated by numerical noise. Choose a less "
            f"correlated concept pair, or raise "
            f"StabilityPolicy.max_condition_number deliberately and record it."
        )


def build_swap_basis(
    atoms: torch.Tensor,
    *,
    layer: int,
    source: ConceptToken,
    target: ConceptToken,
    policy: StabilityPolicy = DEFAULT_STABILITY,
    final_norm_weight_folded: bool = False,
    kind: str = "coordinate_swap",
) -> SwapBasis:
    """``V = [atoms[source], atoms[target]]``, diagnosed and stability-gated.

    Args:
        atoms: ``[n_atoms, d_model]`` dictionary atoms — rows of ``W_U @ J_l``.
            Pass ``JSpaceDictionary.atoms`` directly; the atom index *is* the
            vocabulary token id.
        layer: The layer the dictionary belongs to. Checked against the
            dictionary when one is passed to :func:`build_swap_bases`.
        final_norm_weight_folded: Recorded, and required to be ``False`` unless
            the caller states otherwise — the pilot's dictionaries are built
            with ``final_norm_weight=None``.

    Raises:
        CoordinateSwapError: On a mis-shaped ``atoms``, an out-of-range token
            id, a non-finite vector, or ``source.token_id == target.token_id``.
        RankDeficientPairError / IllConditionedPairError: See
            :func:`assert_stable`.
    """
    if atoms.ndim != 2:
        raise CoordinateSwapError(
            f"atoms must be [n_atoms, d_model] (rows of W_U @ J_l), got "
            f"{tuple(atoms.shape)}"
        )
    n_atoms, d_model = atoms.shape
    for token in (source, target):
        if not 0 <= token.token_id < n_atoms:
            raise CoordinateSwapError(
                f"token id {token.token_id} for {token.concept!r} is out of range "
                f"for a {n_atoms}-atom dictionary"
            )
    if source.token_id == target.token_id:
        raise CoordinateSwapError(
            f"source and target resolve to the same token id {source.token_id} "
            f"({source.concept!r} / {target.concept!r}); there is nothing to exchange"
        )
    return build_swap_basis_from_vectors(
        atoms[source.token_id],
        atoms[target.token_id],
        layer=layer,
        source=source,
        target=target,
        policy=policy,
        final_norm_weight_folded=final_norm_weight_folded,
        kind=kind,
    )


def build_swap_basis_from_vectors(
    source_vector: torch.Tensor,
    target_vector: torch.Tensor,
    *,
    layer: int,
    source: ConceptToken,
    target: ConceptToken,
    policy: StabilityPolicy = DEFAULT_STABILITY,
    final_norm_weight_folded: bool = False,
    kind: str = "coordinate_swap",
) -> SwapBasis:
    """Build ``V`` from two already-selected rows of ``W_U @ J_l``.

    This is algebraically identical to :func:`build_swap_basis`, but avoids
    materializing the full ``[vocab, d_model]`` dictionary when an experiment
    uses only a fixed handful of token vectors.
    """
    if source.token_id == target.token_id:
        raise CoordinateSwapError(
            f"source and target resolve to the same token id {source.token_id} "
            f"({source.concept!r} / {target.concept!r}); there is nothing to exchange"
        )
    if source_vector.ndim != 1 or target_vector.ndim != 1:
        raise CoordinateSwapError(
            "selected lens vectors must both be 1-D; got "
            f"{tuple(source_vector.shape)} and {tuple(target_vector.shape)}"
        )
    if source_vector.shape != target_vector.shape:
        raise CoordinateSwapError(
            "selected lens vectors have different widths: "
            f"{tuple(source_vector.shape)} and {tuple(target_vector.shape)}"
        )
    V = _column_pair(source_vector, target_vector, dtype=policy.torch_solve_dtype)
    _finite_or_raise(V, "swap basis V")
    assert_vector_orientation(V, d_model=int(source_vector.shape[0]))
    diagnostics = basis_diagnostics(V, policy=policy)
    assert_stable(diagnostics, policy=policy)
    return SwapBasis(
        layer=int(layer),
        source=source,
        target=target,
        V=V,
        diagnostics=diagnostics,
        final_norm_weight_folded=bool(final_norm_weight_folded),
        kind=kind,
    )


def reverse_basis(basis: SwapBasis) -> SwapBasis:
    """The same pair with the source and target roles exchanged.

    **The reverse swap is the same operator.** Exchange is symmetric in its two
    arguments::

        V  = [v_s, v_t],  c  = (c_s, c_t),  V (sigma(c) - c)  = (c_t - c_s)(v_s - v_t)
        V' = [v_t, v_s],  c' = (c_t, c_s),  V'(sigma(c') - c') = (c_t - c_s)(v_s - v_t)

    so ``h_patched`` is identical for any ``alpha``. What reverses is the
    *bookkeeping*: which coordinate is reported as "source" and which as
    "target". This is a genuine discriminator between an exchange and a
    direction steer — ``h +- alpha * v`` is antisymmetric under the same
    relabelling, an exchange is symmetric — and it is why ``reverse_swap`` is a
    control over the **evidence** (apply the bird/cat pair to cat evidence and
    check the identification moves toward bird), never a control over the
    operator.
    """
    return SwapBasis(
        layer=basis.layer,
        source=basis.target,
        target=basis.source,
        V=basis.V.flip(1).contiguous(),
        diagnostics=basis_diagnostics(basis.V.flip(1).contiguous()),
        final_norm_weight_folded=basis.final_norm_weight_folded,
        kind="reverse_swap",
    )


def random_two_direction_basis(
    basis: SwapBasis, *, seed: int, policy: StabilityPolicy = DEFAULT_STABILITY
) -> SwapBasis:
    """Norm-matched random control: two Gaussian directions, same algebra.

    Each column is a deterministic CPU-generated Gaussian rescaled to *exactly*
    the norm of the column it replaces, so the control differs from the real
    condition in direction only. The generated pair goes through the same
    stability gate — a random pair that happened to be ill-conditioned would be
    refused for the same reason a real one would.
    """
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    columns = []
    for index in range(2):
        raw = torch.randn(basis.d_model, generator=generator, dtype=torch.float64)
        norm = raw.norm()
        if float(norm) == 0.0:  # pragma: no cover - probability zero
            raise CoordinateSwapError("degenerate zero random direction")
        columns.append(raw / norm * basis.V[:, index].norm())
    V = torch.stack(columns, dim=1)
    diagnostics = basis_diagnostics(V, policy=policy)
    assert_stable(diagnostics, policy=policy)
    diagnostics["random_seed"] = int(seed)
    diagnostics["matched_to_V_checksum"] = basis.diagnostics["V_checksum"]
    return SwapBasis(
        layer=basis.layer,
        source=basis.source,
        target=basis.target,
        V=V,
        diagnostics=diagnostics,
        final_norm_weight_folded=basis.final_norm_weight_folded,
        kind="random_two_direction_norm_matched",
    )


def build_swap_bases(
    atoms_by_layer: Mapping[int, torch.Tensor],
    *,
    layers: Sequence[int],
    source: ConceptToken,
    target: ConceptToken,
    policy: StabilityPolicy = DEFAULT_STABILITY,
    final_norm_weight_folded: bool = False,
) -> dict[int, SwapBasis]:
    """One basis per layer in ``layers``. Every layer must have a dictionary."""
    missing = [layer for layer in layers if layer not in atoms_by_layer]
    if missing:
        raise CoordinateSwapError(
            f"no dictionary atoms for layers {missing}; a coordinate swap needs "
            f"that layer's own W_U @ J_l"
        )
    return {
        int(layer): build_swap_basis(
            atoms_by_layer[layer],
            layer=int(layer),
            source=source,
            target=target,
            policy=policy,
            final_norm_weight_folded=final_norm_weight_folded,
        )
        for layer in layers
    }


# --------------------------------------------------------------- the algebra


def read_coordinates(
    h: torch.Tensor, V: torch.Tensor, *, policy: StabilityPolicy = DEFAULT_STABILITY
) -> torch.Tensor:
    """``c = pinv(V) h`` for one or many activations.

    Args:
        h: ``[d_model]`` or ``[n, d_model]``.
        V: ``[d_model, 2]``.

    Returns:
        ``[2]`` or ``[n, 2]`` float64 coordinates.

    Solved from the normal equations, not from independent dot products: the
    columns of ``V`` are generally nonorthogonal, so ``v_source . h`` is not
    ``c_source`` and using it would exchange the wrong quantities.
    """
    dtype = policy.torch_solve_dtype
    single = h.ndim == 1
    H = (h.reshape(1, -1) if single else h).detach().to(dtype)
    Vd = V.to(device=H.device, dtype=dtype)
    if H.ndim != 2 or H.shape[1] != Vd.shape[0]:
        raise CoordinateSwapError(
            f"activation shape {tuple(h.shape)} is not compatible with a "
            f"[{Vd.shape[0]}, 2] basis"
        )
    gram = Vd.T @ Vd                      # [2, 2]
    rhs = Vd.T @ H.T                      # [2, n]
    c = torch.linalg.solve(gram, rhs).T   # [n, 2]
    return c[0] if single else c


def orthogonal_residual(
    h: torch.Tensor, V: torch.Tensor, *, policy: StabilityPolicy = DEFAULT_STABILITY
) -> torch.Tensor:
    """``h - V pinv(V) h`` — the component the swap must leave alone."""
    dtype = policy.torch_solve_dtype
    single = h.ndim == 1
    H = (h.reshape(1, -1) if single else h).detach().to(dtype)
    c = read_coordinates(H, V, policy=policy)
    Vd = V.to(device=H.device, dtype=dtype)
    residual = H - c @ Vd.T
    return residual[0] if single else residual


def swap_coordinates(
    h: torch.Tensor,
    V: torch.Tensor,
    *,
    alpha: float = 1.0,
    policy: StabilityPolicy = DEFAULT_STABILITY,
    realization_policy: ModelDtypeRealizationPolicy | None = None,
) -> tuple[torch.Tensor, dict]:
    """``h_patched = h + alpha * V (sigma(c) - c)``, with diagnostics.

    Args:
        h: ``[d_model]`` or ``[n, d_model]``, any dtype/device.
        V: ``[d_model, 2]``.
        alpha: See the module docstring. ``0`` is an exact no-op; ``1`` is the
            exchange; ``>1`` is extrapolation.

    Returns:
        ``(h_patched, record)``. ``h_patched`` has ``h``'s original dtype and
        device. ``record`` carries the pre/post coordinates, the update norm,
        the orthogonal-residual norm before and after, and ``V``'s checksum.

    The whole update is computed even when ``alpha == 0`` — the multiplication
    by exactly zero is what makes it a no-op, so the identical code path is
    exercised in the parity condition and in the real one.
    """
    if not math.isfinite(alpha):
        raise CoordinateSwapError(f"alpha must be finite, got {alpha}")
    dtype = policy.torch_solve_dtype
    single = h.ndim == 1
    original_dtype, original_device = h.dtype, h.device
    H = (h.reshape(1, -1) if single else h).detach().to(dtype)
    Vd = V.to(device=H.device, dtype=dtype)
    assert_vector_orientation(Vd, d_model=int(H.shape[1]))

    c = read_coordinates(H, Vd, policy=policy)          # [n, 2]
    swapped = c.flip(-1)                                # sigma(c)
    update = (swapped - c) @ Vd.T                       # [n, d_model]
    _finite_or_raise(update, "coordinate-swap update")
    patched = H + float(alpha) * update

    residual_before = H - c @ Vd.T
    coordinates_after = read_coordinates(patched, Vd, policy=policy)
    expected_coordinates_after = c + float(alpha) * (swapped - c)
    residual_after = patched - coordinates_after @ Vd.T

    # The model consumes the update in the block output's original dtype, not
    # in float64.  Audit that actual value as well as the ideal solve above.
    # This is especially important for bf16 models: an exact exchange before
    # the cast is not evidence that the value written back to the residual
    # stream still realizes the intended coordinates.
    expected_scale = expected_coordinates_after.abs().amax(dim=-1).clamp_min(
        1.0 if realization_policy is None else float(realization_policy.minimum_scale)
    )
    residual_scale = residual_before.norm(dim=-1).clamp_min(
        1.0 if realization_policy is None else float(realization_policy.minimum_scale)
    )
    candidate = patched
    realization_history = []
    max_corrections = (
        0 if realization_policy is None else int(realization_policy.max_corrections)
    )
    for correction_index in range(max_corrections + 1):
        patched_model_dtype = candidate.to(
            device=original_device, dtype=original_dtype
        )
        patched_after_cast = patched_model_dtype.detach().to(
            device=H.device, dtype=dtype
        )
        coordinates_after_cast = read_coordinates(
            patched_after_cast, Vd, policy=policy
        )
        residual_after_cast = patched_after_cast - coordinates_after_cast @ Vd.T
        post_cast_coordinate_error = (
            coordinates_after_cast - expected_coordinates_after
        ).abs().amax(dim=-1)
        post_cast_residual_drift = (
            residual_after_cast - residual_before
        ).norm(dim=-1)
        relative_coordinate_error = post_cast_coordinate_error / expected_scale
        relative_residual_drift = post_cast_residual_drift / residual_scale
        coordinate_max = float(relative_coordinate_error.max())
        residual_max = float(relative_residual_drift.max())
        realization_history.append(
            {
                "correction_index": correction_index,
                "max_relative_coordinate_error": coordinate_max,
                "max_relative_residual_drift": residual_max,
            }
        )
        if realization_policy is None or (
            coordinate_max
            <= float(realization_policy.relative_coordinate_tolerance)
            and residual_max
            <= float(realization_policy.relative_residual_tolerance)
        ):
            break
        coordinate_correction = (
            expected_coordinates_after - coordinates_after_cast
        ) @ Vd.T
        candidate = patched_after_cast + coordinate_correction

    realization_converged = bool(
        realization_policy is None
        or (
            float(relative_coordinate_error.max())
            <= float(realization_policy.relative_coordinate_tolerance)
            and float(relative_residual_drift.max())
            <= float(realization_policy.relative_residual_tolerance)
        )
    )

    record = {
        "method_version": METHOD_VERSION,
        "alpha": float(alpha),
        "alpha_is_extrapolation": bool(alpha > 1.0),
        "n_positions": int(H.shape[0]),
        "coordinates_before": [[float(x) for x in row] for row in c],
        "coordinates_after": [[float(x) for x in row] for row in coordinates_after],
        "update_norm": [float(x) for x in update.norm(dim=-1) * abs(float(alpha))],
        "activation_norm_before": [float(x) for x in H.norm(dim=-1)],
        "activation_norm_after": [float(x) for x in patched.norm(dim=-1)],
        "max_coordinate_update_error": float(
            (coordinates_after - expected_coordinates_after).abs().max()
        ),
        "alpha_one_is_exact_exchange": bool(float(alpha) == 1.0),
        "orthogonal_residual_norm_before": [float(x) for x in residual_before.norm(dim=-1)],
        "orthogonal_residual_norm_after": [float(x) for x in residual_after.norm(dim=-1)],
        "max_orthogonal_residual_drift": float(
            (residual_after - residual_before).norm(dim=-1).max()
        ),
        "model_dtype": str(original_dtype),
        "max_post_cast_coordinate_update_error": float(
            post_cast_coordinate_error.max()
        ),
        "max_post_cast_relative_coordinate_update_error": float(
            relative_coordinate_error.max()
        ),
        "max_post_cast_orthogonal_residual_drift": float(
            post_cast_residual_drift.max()
        ),
        "max_post_cast_relative_orthogonal_residual_drift": float(
            relative_residual_drift.max()
        ),
        "model_dtype_realization": (
            None if realization_policy is None else realization_policy.to_dict()
        ),
        "model_dtype_realization_converged": realization_converged,
        "model_dtype_corrections_applied": len(realization_history) - 1,
        "model_dtype_realization_history": realization_history,
        "V_checksum": tensor_checksum(Vd.cpu()),
        "solve_policy": SOLVE_POLICY,
    }
    out = patched_model_dtype
    return (out[0] if single else out), record


# ------------------------------------------------------- positions and bands


def resolve_positions(
    rule: str,
    *,
    prompt_len: int,
    seq_len: int,
    evidence_span: Sequence[int] | None = None,
) -> list[int]:
    """Which token positions a rule selects, with the boundary enforced.

    Args:
        rule: One of :data:`POSITION_RULES`.
        prompt_len: Length of the *original prompt/evidence* sequence, before
            any teacher-forced candidate tokens were appended.
        seq_len: The actual sequence length the hook is seeing. When it exceeds
            ``prompt_len`` the extra positions are candidate-completion tokens.
        evidence_span: ``[start, end)`` of the modality's token span
            (``BuiltInputs.modality_token_range``), when one is known.

    Raises:
        CoordinateSwapError: On an unknown rule, ``seq_len < prompt_len`` (the
            boundary would be meaningless), or a span-based rule without a span.

    :data:`PROMPT_BOUNDARY_RULE` is enforced for every rule, so a
    teacher-forced candidate token can never be patched by any configuration.
    """
    if rule not in POSITION_RULES:
        raise CoordinateSwapError(f"unknown position rule {rule!r}; known: {POSITION_RULES}")
    if prompt_len < 1:
        raise CoordinateSwapError(f"prompt_len must be >= 1, got {prompt_len}")
    if seq_len < prompt_len:
        raise CoordinateSwapError(
            f"sequence length {seq_len} is shorter than prompt_len {prompt_len}; "
            f"the prompt/candidate boundary cannot be resolved"
        )
    prompt_positions = list(range(prompt_len))
    if rule == "all_prompt_positions":
        selected = prompt_positions
    elif rule == "final_prompt_token_only":
        selected = [prompt_len - 1]
    else:
        if evidence_span is None:
            raise CoordinateSwapError(
                f"position rule {rule!r} needs an evidence span, but this input "
                f"reports none (modality_token_range is None). Refusing to guess "
                f"which positions carry the evidence."
            )
        start, end = int(evidence_span[0]), int(evidence_span[1])
        if not 0 <= start < end <= prompt_len:
            raise CoordinateSwapError(
                f"evidence span [{start}, {end}) is not inside the prompt "
                f"[0, {prompt_len})"
            )
        span = set(range(start, end))
        selected = (
            sorted(span)
            if rule == "evidence_span_only"
            else [p for p in prompt_positions if p not in span]
        )
    if not selected:
        raise CoordinateSwapError(
            f"position rule {rule!r} selected no positions (prompt_len="
            f"{prompt_len}, evidence_span={evidence_span})"
        )
    return selected


@dataclass(frozen=True)
class LayerBand:
    """A contiguous, explicitly validated band of decoder layers."""

    start: int
    end: int
    layers: tuple[int, ...]
    validated_layers: tuple[int, ...]

    def to_dict(self) -> dict:
        return {
            "start": int(self.start),
            "end": int(self.end),
            "layers": list(self.layers),
            "validated_layers": list(self.validated_layers),
            "contiguous": True,
        }


def build_layer_band(
    start: int, end: int, *, validated_layers: Sequence[int], n_layers: int | None = None
) -> LayerBand:
    """A contiguous band ``[start, end]`` (inclusive), gated on validation.

    Raises:
        LayerBandError: If ``end < start``, if the band leaves the model, or if
            any layer in it is not in ``validated_layers``. The gate exists
            because a lens that never passed the calibration run's untouched
            confirmation set has no established relationship to the model's
            computation, so a swap read through it would measure nothing.
            ``validated_layers`` should come from a recorded artifact —
            :data:`jlens.mmpilot.published_lens.CONFIRMED_LAYERS` for the
            published lenses — never from a notebook literal.
    """
    if end < start:
        raise LayerBandError(f"empty band: end {end} < start {start}")
    layers = tuple(range(int(start), int(end) + 1))
    if n_layers is not None:
        out_of_range = [layer for layer in layers if not 0 <= layer < n_layers]
        if out_of_range:
            raise LayerBandError(
                f"band layers {out_of_range} are out of range for a {n_layers}-layer model"
            )
    allowed = set(int(layer) for layer in validated_layers)
    unvalidated = [layer for layer in layers if layer not in allowed]
    if unvalidated:
        raise LayerBandError(
            f"band [{start}, {end}] includes layers {unvalidated}, which did not "
            f"pass the recorded untouched confirmation gate (validated: "
            f"{sorted(allowed)}). Refusing: an unconfirmed lens cannot define "
            f"the coordinates being exchanged."
        )
    return LayerBand(
        start=int(start),
        end=int(end),
        layers=layers,
        validated_layers=tuple(sorted(allowed)),
    )


# ------------------------------------------------------------------- the hook


@contextmanager
def coordinate_swap_layer(
    blocks: Sequence[nn.Module],
    basis: SwapBasis,
    *,
    alpha: float,
    prompt_len: int,
    position_rule: str = PRIMARY_POSITION_RULE,
    evidence_span: Sequence[int] | None = None,
    batch_row: int = 0,
    policy: StabilityPolicy = DEFAULT_STABILITY,
    realization_policy: ModelDtypeRealizationPolicy | None = None,
    require_frozen: bool = True,
    record_coordinates: bool = True,
):
    """Patch one layer's block output at every selected position.

    The coordinates are recomputed from *this layer's current activation* on
    every forward pass — nothing is precomputed and replayed. Tuple block
    outputs have only element 0 edited; every other element passes through.
    The patched values are cast back to the block output's own dtype and
    device, so the model never sees float64.

    Yields a stats dict that the forward pass fills in.
    """
    layer = int(basis.layer)
    if not 0 <= layer < len(blocks):
        raise CoordinateSwapError(f"layer {layer} out of range for {len(blocks)} blocks")
    if require_frozen:
        _assert_frozen(blocks[layer])

    stats: dict = {
        "layer": layer,
        "alpha": float(alpha),
        "position_rule": position_rule,
        "prompt_len": int(prompt_len),
        "prompt_boundary_rule": PROMPT_BOUNDARY_RULE,
        "basis": basis.to_dict(),
        "n_forward_passes": 0,
        "positions": None,
        "n_positions": None,
        "seq_len": None,
        "n_candidate_positions_skipped": None,
        "swap": None,
        "swap_history": [],
    }

    def hook(module: nn.Module, inputs, output):
        is_tensor = torch.is_tensor(output)
        hidden = output if is_tensor else output[0]
        if hidden.ndim != 3:
            raise CoordinateSwapError(
                f"expected [batch, seq, d_model] block output, got {tuple(hidden.shape)}"
            )
        batch, seq_len, d_model = hidden.shape
        if not 0 <= batch_row < batch:
            raise CoordinateSwapError(
                f"batch_row {batch_row} out of range for batch size {batch}"
            )
        if d_model != basis.d_model:
            raise CoordinateSwapError(
                f"layer {layer} block output has d_model {d_model} but the basis "
                f"is [{basis.d_model}, 2]"
            )
        positions = resolve_positions(
            position_rule,
            prompt_len=int(prompt_len),
            seq_len=int(seq_len),
            evidence_span=evidence_span,
        )
        selected = hidden[batch_row, positions]
        patched, record = swap_coordinates(
            selected,
            basis.V,
            alpha=alpha,
            policy=policy,
            realization_policy=realization_policy,
        )
        _finite_or_raise(patched, f"patched activation at layer {layer}")
        new_hidden = hidden.clone()
        new_hidden[batch_row, positions] = patched.to(hidden.dtype)

        stats["n_forward_passes"] += 1
        stats["seq_len"] = int(seq_len)
        stats["positions"] = list(positions)
        stats["n_positions"] = len(positions)
        stats["n_candidate_positions_skipped"] = int(seq_len) - int(prompt_len)
        if not record_coordinates:
            record = {
                key: value
                for key, value in record.items()
                if key not in ("coordinates_before", "coordinates_after")
            }
        stats["swap"] = record
        stats["swap_history"].append(record)
        if is_tensor:
            return new_hidden
        return (new_hidden, *tuple(output)[1:])

    handle = blocks[layer].register_forward_hook(hook)
    try:
        yield stats
    finally:
        handle.remove()


@contextmanager
def coordinate_swap_band(
    blocks: Sequence[nn.Module],
    bases: Mapping[int, SwapBasis],
    *,
    alpha: float,
    prompt_len: int,
    position_rule: str = PRIMARY_POSITION_RULE,
    evidence_span: Sequence[int] | None = None,
    batch_row: int = 0,
    policy: StabilityPolicy = DEFAULT_STABILITY,
    realization_policy: ModelDtypeRealizationPolicy | None = None,
    require_frozen: bool = True,
    record_coordinates: bool = True,
):
    """The paper's protocol: the same swap at every layer of a band.

    One hook per layer in ``bases``, each recomputing its own coordinates from
    its own activation. Every hook is removed on normal exit and on any
    exception, including one raised inside a hook during the forward pass.

    Yields ``{layer: stats}``.
    """
    if not bases:
        raise CoordinateSwapError("a coordinate-swap band needs at least one layer")
    stats: dict[int, dict] = {}
    with ExitStack() as stack:
        for layer in sorted(bases):
            stats[int(layer)] = stack.enter_context(
                coordinate_swap_layer(
                    blocks,
                    bases[layer],
                    alpha=alpha,
                    prompt_len=prompt_len,
                    position_rule=position_rule,
                    evidence_span=evidence_span,
                    batch_row=batch_row,
                    policy=policy,
                    realization_policy=realization_policy,
                    require_frozen=require_frozen,
                    record_coordinates=record_coordinates,
                )
            )
        yield stats


# --------------------------------------------------- the direct-answer control


def direct_answer_vector(
    atoms: torch.Tensor, *, answer_token_id: int, scale: float = 1.0
) -> torch.Tensor:
    """The lens vector of a downstream **answer** token, scaled.

    The control that decides whether a downstream property change may be called
    *recomputation*. If inserting the answer's own lens vector at the same
    layer (or an earlier one) moves the answer just as well as swapping the
    intermediate entity does, then the swap's downstream effect is evidence of
    a shortcut, not of the model re-deriving the property from a changed
    identity. Applied through the existing single-direction path
    (:func:`jlens.interventions.residual_intervention`) precisely because it is
    *not* a coordinate swap and must not be recorded as one.
    """
    if atoms.ndim != 2:
        raise CoordinateSwapError(
            f"atoms must be [n_atoms, d_model], got {tuple(atoms.shape)}"
        )
    if not 0 <= answer_token_id < atoms.shape[0]:
        raise CoordinateSwapError(
            f"answer token id {answer_token_id} out of range for "
            f"{atoms.shape[0]} atoms"
        )
    vector = atoms[int(answer_token_id)].detach().to(torch.float32)
    norm = float(vector.norm())
    if norm == 0.0:
        raise CoordinateSwapError(f"answer atom {answer_token_id} has zero norm")
    return vector / norm * float(scale)


# ----------------------------------------------------------- one condition


def run_swap_condition(
    backend,
    inputs,
    *,
    bases: Mapping[int, SwapBasis],
    alpha: float,
    candidate_ids: Mapping[str, Sequence[int]],
    target_concept: str,
    clean_scores: Mapping[str, Mapping],
    position_rule: str = PRIMARY_POSITION_RULE,
    evidence_span: Sequence[int] | None = None,
    policy: StabilityPolicy = DEFAULT_STABILITY,
    record_coordinates: bool = False,
) -> dict:
    """One banded coordinate swap, scored against the clean run.

    Deliberately the same shape as :func:`jlens.mmpilot.causal.run_condition`
    so the two families can be reported side by side — and deliberately a
    *separate* function, so a record from one can never be produced by the
    other. Candidate scoring goes through
    :func:`jlens.mmpilot.capability.score_candidate_sequences`, unchanged, so
    clean and patched runs are measured identically; the swap hooks stay
    installed across every candidate's forward pass and skip the appended
    candidate tokens by :data:`PROMPT_BOUNDARY_RULE`.
    """
    from jlens.mmpilot.capability import (
        prediction_and_margin,
        score_candidate_sequences,
    )

    span = evidence_span
    if span is None:
        span = getattr(inputs, "modality_token_range", None)
    with coordinate_swap_band(
        backend.blocks,
        bases,
        alpha=alpha,
        prompt_len=inputs.prompt_len,
        position_rule=position_rule,
        evidence_span=span,
        policy=policy,
        record_coordinates=record_coordinates,
    ) as stats:
        scores = score_candidate_sequences(backend, inputs, candidate_ids)
    verdict = prediction_and_margin(scores, target_concept)
    clean_verdict = prediction_and_margin(clean_scores, target_concept)
    unrelated = {
        candidate: scores[candidate]["sum_logprob"] - clean_scores[candidate]["sum_logprob"]
        for candidate in scores
        if candidate != target_concept
    }
    patched = sorted(int(layer) for layer, row in stats.items() if row["n_forward_passes"])
    return {
        "intervention_family": INTERVENTION_FAMILY,
        "method_version": METHOD_VERSION,
        "alpha": float(alpha),
        "alpha_is_extrapolation": bool(alpha > 1.0),
        "position_rule": position_rule,
        "layer_band": sorted(int(layer) for layer in bases),
        "layers_patched": patched,
        "positions_patched": stats[patched[0]]["positions"] if patched else [],
        "n_positions_patched": stats[patched[0]]["n_positions"] if patched else 0,
        "n_candidate_positions_skipped": (
            stats[patched[0]]["n_candidate_positions_skipped"] if patched else None
        ),
        "clean_target_score": clean_verdict["target_score"],
        "target_score": verdict["target_score"],
        "target_score_change": verdict["target_score"] - clean_verdict["target_score"],
        "clean_target_margin": clean_verdict["target_margin"],
        "target_margin": verdict["target_margin"],
        "target_margin_change": verdict["target_margin"] - clean_verdict["target_margin"],
        "unrelated_candidate_changes": unrelated,
        "max_abs_unrelated_change": max((abs(v) for v in unrelated.values()), default=0.0),
        "clean_prediction": clean_verdict["prediction"],
        "prediction": verdict["prediction"],
        "prediction_changed": verdict["prediction"] != clean_verdict["prediction"],
        "candidate_scores": scores,
        "layer_stats": {str(layer): stats[layer] for layer in sorted(stats)},
    }


# ------------------------------------------------------------------- the spec


@dataclass(frozen=True)
class CoordinateSwapSpec:
    """The versioned, fully-stated description of one coordinate-swap condition.

    Everything a later reader needs in order to know *what was done*, and
    everything a resume must match. Constructed with
    :func:`build_spec`, which fills the convention fields from this module's
    constants so a notebook cannot restate them differently.
    """

    source_concept: str
    target_concept: str
    source_token: str
    target_token: str
    source_token_id: int
    target_token_id: int
    layer_band: tuple[int, ...]
    alpha: float
    position_rule: str
    control_kind: str
    lens_checksums: dict[str, str]
    model_revision: str
    processor_revision: str
    intervention_family: str = INTERVENTION_FAMILY
    method_version: str = METHOD_VERSION
    vector_convention: str = VECTOR_CONVENTION
    normalization: str = NORMALIZATION_CONVENTION
    solve_policy: str = SOLVE_POLICY
    prompt_boundary_rule: str = PROMPT_BOUNDARY_RULE
    stability: dict = field(default_factory=lambda: DEFAULT_STABILITY.to_dict())
    audio_protocol_fingerprint: str | None = None
    #: The prompt protocol the behavioral readout was asked under, and the
    #: digest of its full fingerprint. ``None`` on a spec built before a prompt
    #: was bound — which the primary study refuses (see
    #: :func:`assert_open_prompt_protocol`).
    prompt_protocol_version: str | None = None
    prompt_protocol_digest: str | None = None

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["layer_band"] = list(self.layer_band)
        return payload

    @property
    def digest(self) -> str:
        return payload_checksum(self.to_dict())


def build_spec(
    *,
    source: ConceptToken,
    target: ConceptToken,
    layer_band: LayerBand | Sequence[int],
    alpha: float,
    position_rule: str,
    control_kind: str,
    lens_checksums: Mapping[int, str],
    model_revision: str,
    processor_revision: str,
    policy: StabilityPolicy = DEFAULT_STABILITY,
    audio_protocol_fingerprint: str | None = None,
    prompt_protocol: Mapping | None = None,
) -> CoordinateSwapSpec:
    """Assemble a spec, validating the parts that have a fixed vocabulary.

    Args:
        prompt_protocol: The record from
            :func:`jlens.mmpilot.prompt_protocol.prompt_protocol_fingerprint`.
            Its version and digest are carried into the spec so that *what was
            asked* is bound to *what was patched*. Optional here and **required**
            by :func:`assert_open_prompt_protocol` for the primary study.
    """
    if position_rule not in POSITION_RULES:
        raise CoordinateSwapError(
            f"unknown position rule {position_rule!r}; known: {POSITION_RULES}"
        )
    if control_kind not in CONTROL_KINDS:
        raise CoordinateSwapError(
            f"unknown control kind {control_kind!r}; known: {CONTROL_KINDS}"
        )
    layers = tuple(
        int(layer)
        for layer in (layer_band.layers if isinstance(layer_band, LayerBand) else layer_band)
    )
    if not layers:
        raise CoordinateSwapError("layer_band is empty")
    missing = [layer for layer in layers if layer not in lens_checksums]
    if missing:
        raise CoordinateSwapError(
            f"no lens checksum recorded for band layers {missing}; every layer "
            f"that is patched must name the lens its coordinates came from"
        )
    return CoordinateSwapSpec(
        source_concept=source.concept,
        target_concept=target.concept,
        source_token=source.token_text,
        target_token=target.token_text,
        source_token_id=int(source.token_id),
        target_token_id=int(target.token_id),
        layer_band=layers,
        alpha=float(alpha),
        position_rule=position_rule,
        control_kind=control_kind,
        lens_checksums={str(layer): str(lens_checksums[layer]) for layer in layers},
        model_revision=str(model_revision),
        processor_revision=str(processor_revision),
        stability=policy.to_dict(),
        audio_protocol_fingerprint=audio_protocol_fingerprint,
        prompt_protocol_version=(
            None if prompt_protocol is None else str(prompt_protocol["prompt_protocol_version"])
        ),
        prompt_protocol_digest=(
            None if prompt_protocol is None else str(prompt_protocol["prompt_protocol_digest"])
        ),
    )


# ------------------------------------------------------------- fingerprinting


def coordinate_swap_fingerprint(
    spec: CoordinateSwapSpec,
    *,
    alphas: Sequence[float],
    controls: Sequence[str],
    control_config: Mapping | None = None,
    prompt_protocol: Mapping | None = None,
    pair_selection: Mapping | None = None,
) -> dict:
    """The ``intervention_config`` a coordinate-swap run binds its artifacts to.

    Passed to :class:`jlens.mmpilot.store.RunFingerprint`. Because it names an
    ``intervention_family`` that no direction-steering run ever wrote, a run
    directory from the completed pilot has a different digest and
    :meth:`~jlens.mmpilot.store.UnitStore.open` refuses it — but the digest
    check reports a diff, not a diagnosis, so
    :func:`assert_coordinate_swap_artifacts` states the reason explicitly.
    """
    unknown = [kind for kind in controls if kind not in CONTROL_KINDS]
    if unknown:
        raise CoordinateSwapError(
            f"unknown control kinds {unknown}; known: {CONTROL_KINDS}"
        )
    if any(not math.isfinite(float(a)) for a in alphas):
        raise CoordinateSwapError(f"alphas must all be finite, got {list(alphas)}")
    return {
        "intervention_family": INTERVENTION_FAMILY,
        "coordinate_swap_method_version": METHOD_VERSION,
        "source_token_id": int(spec.source_token_id),
        "target_token_id": int(spec.target_token_id),
        "source_token": spec.source_token,
        "target_token": spec.target_token,
        "vector_convention": spec.vector_convention,
        "normalization": spec.normalization,
        "solve_policy": spec.solve_policy,
        "condition_number_threshold": float(spec.stability["max_condition_number"]),
        "rank_tolerance": float(spec.stability["rank_tolerance"]),
        "solve_dtype": spec.stability["solve_dtype"],
        "alphas": [float(a) for a in alphas],
        "position_rule": spec.position_rule,
        "prompt_length_boundary_rule": spec.prompt_boundary_rule,
        "layer_band": list(spec.layer_band),
        "lens_checksums_by_layer": dict(spec.lens_checksums),
        "model_revision": spec.model_revision,
        "processor_revision": spec.processor_revision,
        "audio_protocol_fingerprint": spec.audio_protocol_fingerprint,
        "controls": sorted(controls),
        "control_config": dict(control_config or {}),
        # What the model was asked, bound beside what was patched. A run that
        # scored a different candidate set, or asked under a different protocol,
        # is not the same measurement and must not resume from this directory.
        "prompt_protocol": dict(prompt_protocol) if prompt_protocol else None,
        "prompt_protocol_version": spec.prompt_protocol_version,
        "prompt_protocol_digest": spec.prompt_protocol_digest,
        "property_pair_selection": dict(pair_selection) if pair_selection else None,
        "property_pair_selection_version": (
            None
            if pair_selection is None
            else pair_selection.get("pair_selection_version")
        ),
        "ordered_directed_pairs": (
            []
            if pair_selection is None
            else list(pair_selection.get("ordered_directed_pairs") or ())
        ),
        "property_contrast_rule": (
            None
            if pair_selection is None
            else pair_selection.get("property_contrast_rule")
        ),
        "property_pair_selection_checksum": (
            None
            if pair_selection is None
            else pair_selection.get("pair_selection_checksum")
        ),
        "spec_digest": spec.digest,
    }


def assert_open_prompt_protocol(
    prompt_protocol: Mapping | None,
    *,
    allowed: Sequence[str] = PRIMARY_PROMPT_PROTOCOLS,
) -> dict:
    """Refuse a primary coordinate-swap study asked under a candidate-listed prompt.

    The paper's identity-replacement and downstream-property experiments turn on
    the swap **target** being absent from everything the model can see. A prompt
    that enumerates every candidate introduces the target itself, so the strongest
    thing such a run could support is a candidate-conditioned claim — which is
    what the completed steering study already established and is not what this
    method is for.

    Args:
        prompt_protocol: The record from
            :func:`jlens.mmpilot.prompt_protocol.prompt_protocol_fingerprint`.
        allowed: The admissible protocol identifiers.

    Returns:
        The record, unchanged, when it is admissible.

    Raises:
        CoordinateSwapError: If no prompt protocol is bound, if it is not one of
            ``allowed``, if its candidates were rendered into the prompt, or if
            its registered leakage audit did not pass. Every one of those is a
            refusal rather than a warning: there is no configuration in which the
            primary study proceeds without them.
    """
    if not prompt_protocol:
        raise CoordinateSwapError(
            "the primary coordinate-swap study requires a bound prompt protocol. "
            "Build the prompt with jlens.mmpilot.prompt_protocol and pass its "
            "prompt_protocol_fingerprint; an unrecorded prompt cannot be shown "
            "to have kept the swap target out of the model's input."
        )
    version = prompt_protocol.get("prompt_protocol_version")
    if version not in tuple(allowed):
        raise CoordinateSwapError(
            f"prompt protocol {version!r} is not admissible for the primary "
            f"coordinate-swap study; admissible: {tuple(allowed)}. A "
            "candidate-listed prompt names the swap target, so a swap run under "
            "it could support only a candidate-conditioned claim. Run it as a "
            "labelled comparison condition if you want it, never as the primary."
        )
    if prompt_protocol.get("candidates_in_prompt"):
        raise CoordinateSwapError(
            f"prompt protocol {version!r} recorded candidates_in_prompt=True; the "
            "candidates must be external to the model's prompt"
        )
    if not prompt_protocol.get("leakage_audit_passed"):
        raise CoordinateSwapError(
            f"the registered leakage audit did not pass for prompt protocol "
            f"{version!r}; refusing to run the primary study on a prompt whose "
            "target-absence could not be established"
        )
    return dict(prompt_protocol)


def assert_coordinate_swap_artifacts(stored: Mapping | None) -> None:
    """Refuse to treat another family's artifacts as coordinate-swap results.

    Raises:
        CoordinateSwapError: If ``stored`` (a run directory's
            ``fingerprint.json``, or its ``intervention_config``) was not
            written by a coordinate-swap run. The completed direction-steering
            runs measured a different quantity; reusing one of their
            intervention units here would silently relabel it.
    """
    if not stored:
        raise CoordinateSwapError(
            "no stored fingerprint to check; a coordinate-swap run must be "
            "opened against its own fingerprint"
        )
    config = dict(stored.get("intervention_config") or stored)
    family = config.get("intervention_family")
    if family == INTERVENTION_FAMILY:
        version = config.get("coordinate_swap_method_version")
        if version != METHOD_VERSION:
            raise CoordinateSwapError(
                f"stored artifacts use coordinate-swap method version {version!r}, "
                f"this code is {METHOD_VERSION!r}; refusing to mix algebra versions"
            )
        return
    if family is None:
        raise CoordinateSwapError(
            "stored artifacts name no intervention family. They predate the "
            "coordinate swap and were produced by source-derived J-space "
            "steering (h' = h +- alpha v_concept), which measures a different "
            "quantity. Refusing to resume a coordinate-swap run from them — "
            "point the run at a new directory."
        )
    raise CoordinateSwapError(
        f"stored artifacts belong to intervention family {family!r}, not "
        f"{INTERVENTION_FAMILY!r}. Refusing to reuse them."
    )
