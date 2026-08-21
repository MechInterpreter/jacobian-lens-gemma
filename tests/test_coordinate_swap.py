# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""The coordinate swap's algebra, refusals, hook semantics, and boundaries.

Every test here is deterministic and CPU-only. The point of the file is that
the *refusals* are tested as hard as the successes: a rank-deficient pair, a
transposed basis, a multi-token concept, a candidate-completion position, an
unvalidated layer and a steering run's artifacts must each be rejected with a
reason, because each of them would otherwise produce a plausible-looking number
that means something other than what it would be reported as.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from jlens.metadata import config_fingerprint
from jlens.mmpilot import causal
from jlens.mmpilot.capability import prediction_and_margin, score_candidate_sequences
from jlens.mmpilot.coordinate_swap import (
    CONTROL_KINDS,
    INTERVENTION_FAMILY,
    METHOD_VERSION,
    POSITION_RULES,
    PRIMARY_POSITION_RULE,
    ConceptToken,
    CoordinateSwapError,
    IllConditionedPairError,
    LayerBandError,
    ModelDtypeRealizationPolicy,
    MultiTokenConceptError,
    RankDeficientPairError,
    StabilityPolicy,
    assert_coordinate_swap_artifacts,
    assert_vector_orientation,
    basis_diagnostics,
    build_layer_band,
    build_spec,
    build_swap_bases,
    build_swap_basis,
    build_swap_basis_from_vectors,
    coordinate_swap_band,
    coordinate_swap_fingerprint,
    coordinate_swap_layer,
    direct_answer_vector,
    orthogonal_residual,
    random_two_direction_basis,
    read_coordinates,
    resolve_concept_token,
    resolve_positions,
    reverse_basis,
    run_swap_condition,
    swap_coordinates,
)
from jlens.mmpilot.coordinate_swap_mock import (
    BIRD_ID,
    CAT_ID,
    FOUR_ID,
    IDENTITY_CANDIDATES,
    IDENTITY_QUESTION,
    MOCK_VALIDATED_LAYERS,
    POST_REASONING_BAND,
    PRIMARY_BAND,
    PROPERTY_CANDIDATES,
    PROPERTY_QUESTION,
    REASONING_LAYER,
    TWO_ID,
    SwapMockBackend,
    SwapMockWorld,
    band_parity_diagnostic,
    mock_bases,
    mock_concept_tokens,
    mock_lens_checksums,
)
from jlens.mmpilot.store import IncompatibleStateError, RunFingerprint, UnitStore

REPO_ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------ fixtures


@pytest.fixture(scope="module")
def backend():
    return SwapMockBackend()


@pytest.fixture(scope="module")
def tokens(backend):
    return mock_concept_tokens(backend)


@pytest.fixture(scope="module")
def bases(backend, tokens):
    return mock_bases(
        backend.world, layers=PRIMARY_BAND, source=tokens["bird"], target=tokens["cat"]
    )


@pytest.fixture(scope="module")
def V(backend):
    return backend.world.V


def _activation(seed: int = 3, d_model: int = 16) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(d_model, generator=generator, dtype=torch.float32)


def _identity_setup(backend, *, concept="bird", modality="image"):
    inputs = backend.build_inputs(
        prompt=IDENTITY_QUESTION, modality=modality, concept=concept
    )
    candidate_ids = {name: backend.encode_candidate(f" {name}") for name in IDENTITY_CANDIDATES}
    return inputs, candidate_ids, score_candidate_sequences(backend, inputs, candidate_ids)


def _property_setup(backend, *, concept="bird", modality="image"):
    inputs = backend.build_inputs(
        prompt=PROPERTY_QUESTION, modality=modality, concept=concept
    )
    candidate_ids = {name: backend.encode_candidate(f" {name}") for name in PROPERTY_CANDIDATES}
    return inputs, candidate_ids, score_candidate_sequences(backend, inputs, candidate_ids)


# ------------------------------------------------------------- 1. the algebra


def test_alpha_zero_is_an_exact_no_op(V):
    """Requirement 1. Bit-exact, not merely close — the full update is computed
    and multiplied by exactly zero, so the parity condition runs the same code
    as the real one."""
    h = _activation()
    patched, record = swap_coordinates(h, V, alpha=0.0)
    assert torch.equal(patched, h)
    assert patched.dtype == h.dtype
    assert record["alpha"] == 0.0
    assert record["alpha_is_extrapolation"] is False


def test_double_swap_at_alpha_one_recovers_the_activation(V):
    """Requirement 2. Exchange is an involution."""
    h = _activation(seed=11).double()
    once, _ = swap_coordinates(h, V, alpha=1.0)
    twice, _ = swap_coordinates(once, V, alpha=1.0)
    assert torch.allclose(twice, h, atol=1e-12, rtol=0)
    assert not torch.allclose(once, h, atol=1e-6)


def test_coordinates_are_exchanged_at_alpha_one(V):
    """Requirement 3."""
    h = _activation(seed=5).double()
    before = read_coordinates(h, V)
    patched, record = swap_coordinates(h, V, alpha=1.0)
    after = read_coordinates(patched, V)
    assert torch.allclose(after, before.flip(0), atol=1e-12, rtol=0)
    assert record["coordinates_after"][0] == pytest.approx(
        list(reversed(record["coordinates_before"][0])), abs=1e-12
    )
    assert record["alpha_one_is_exact_exchange"] is True
    assert record["max_coordinate_update_error"] < 1e-12


def test_component_orthogonal_to_span_is_unchanged(V):
    """Requirement 4. The defining property of the method."""
    h = _activation(seed=17).double()
    before = orthogonal_residual(h, V)
    for alpha in (0.0, 0.37, 1.0, 2.0):
        patched, record = swap_coordinates(h, V, alpha=alpha)
        after = orthogonal_residual(patched, V)
        assert torch.allclose(after, before, atol=1e-12, rtol=0)
        assert record["max_orthogonal_residual_drift"] < 1e-12
        # And the update really does live in span(V).
        update = (patched - h).double()
        assert torch.allclose(orthogonal_residual(update, V), torch.zeros_like(update), atol=1e-12)


def test_nonorthogonal_vectors_need_the_pseudoinverse(V, backend):
    """Requirement 5. Dot products are *not* the coordinates here, and the
    swap must use the coordinates."""
    world = backend.world
    assert abs(float(V[:, 0].dot(V[:, 1]))) > 0.1  # genuinely nonorthogonal
    # An activation that is exactly 2 * v_source has coordinates (2, 0) ...
    h = (2.0 * world.v_source).clone()
    coordinates = read_coordinates(h, V)
    assert coordinates.tolist() == pytest.approx([2.0, 0.0], abs=1e-12)
    # ... but its dot products with the two columns are both nonzero.
    dots = (V.T @ h).tolist()
    assert dots[1] != pytest.approx(0.0, abs=1e-6)
    # Exchanging the coordinates therefore lands exactly on 2 * v_target.
    patched, _ = swap_coordinates(h, V, alpha=1.0)
    assert torch.allclose(patched, 2.0 * world.v_target, atol=1e-12)
    # A dot-product "swap" would land somewhere else entirely.
    naive = h + (dots[0] - dots[1]) * (V[:, 1] - V[:, 0])
    assert not torch.allclose(naive, patched, atol=1e-3)


def test_update_is_the_measured_coefficient_not_a_fixed_direction(V):
    """The one-line difference from single-vector J-lens steering: the update
    is parallel to v_source - v_target but its size is read off h."""
    small = 0.01 * _activation(seed=2).double()
    large = 100.0 * _activation(seed=2).double()
    small_update = swap_coordinates(small, V, alpha=1.0)[0] - small
    large_update = swap_coordinates(large, V, alpha=1.0)[0] - large
    assert float(large_update.norm()) == pytest.approx(
        10000.0 * float(small_update.norm()), rel=1e-9
    )
    direction = (V[:, 0] - V[:, 1]) / (V[:, 0] - V[:, 1]).norm()
    unit = small_update / small_update.norm()
    assert abs(abs(float(unit.dot(direction))) - 1.0) < 1e-12


def test_alpha_semantics_interpolate_and_extrapolate(V):
    h = _activation(seed=8).double()
    before = read_coordinates(h, V)
    half, _ = swap_coordinates(h, V, alpha=0.5)
    assert torch.allclose(
        read_coordinates(half, V), 0.5 * (before + before.flip(0)), atol=1e-12
    )
    double, record = swap_coordinates(h, V, alpha=2.0)
    assert torch.allclose(
        read_coordinates(double, V), 2.0 * before.flip(0) - before, atol=1e-12
    )
    assert record["alpha_is_extrapolation"] is True
    assert record["alpha_one_is_exact_exchange"] is False
    assert record["max_coordinate_update_error"] < 1e-12


def test_swap_preserves_dtype_and_batches(V):
    for dtype in (torch.float32, torch.bfloat16, torch.float64):
        h = torch.randn(5, 16, dtype=torch.float32).to(dtype)
        patched, record = swap_coordinates(h, V, alpha=1.0)
        assert patched.dtype == dtype and patched.shape == h.shape
        assert record["n_positions"] == 5
        assert len(record["coordinates_before"]) == 5
        assert record["model_dtype"] == str(dtype)
        assert math.isfinite(record["max_post_cast_coordinate_update_error"])
        assert math.isfinite(
            record["max_post_cast_relative_coordinate_update_error"]
        )


def test_quantization_aware_realization_corrects_the_cast_tensor() -> None:
    generator = torch.Generator().manual_seed(7)
    basis = torch.randn(128, 2, generator=generator, dtype=torch.float64) * 0.2
    activation = torch.randn(12, 128, generator=generator).to(torch.bfloat16)
    _, naive = swap_coordinates(activation, basis, alpha=1.0)
    policy = ModelDtypeRealizationPolicy(
        max_corrections=20,
        relative_coordinate_tolerance=0.001,
        relative_residual_tolerance=0.02,
    )
    patched, corrected = swap_coordinates(
        activation,
        basis,
        alpha=1.0,
        realization_policy=policy,
    )
    assert naive["max_post_cast_relative_coordinate_update_error"] > 0.001
    assert corrected["model_dtype_realization_converged"] is True
    assert corrected["max_post_cast_relative_coordinate_update_error"] <= 0.001
    assert corrected["model_dtype_corrections_applied"] > 0
    assert patched.dtype == activation.dtype


def test_hook_retains_one_post_cast_audit_per_forward(backend, bases):
    inputs, _, _ = _identity_setup(backend)
    layer = PRIMARY_BAND[-1]
    with coordinate_swap_layer(
        backend.blocks, bases[layer], alpha=1.0, prompt_len=inputs.prompt_len
    ) as stats:
        backend.forward_logits(inputs.tensors)
        backend.forward_logits(inputs.tensors)
    assert stats["n_forward_passes"] == 2
    assert len(stats["swap_history"]) == 2
    assert all(
        row["alpha_one_is_exact_exchange"] for row in stats["swap_history"]
    )
    assert all(
        math.isfinite(row["max_post_cast_relative_coordinate_update_error"])
        for row in stats["swap_history"]
    )


def test_alpha_must_be_finite(V):
    with pytest.raises(CoordinateSwapError, match="alpha must be finite"):
        swap_coordinates(_activation(), V, alpha=float("nan"))


# ---------------------------------------------------------------- 2. refusals


def test_rank_deficient_pair_is_refused(backend, tokens):
    """Requirement 6. ``parallel``'s atom is exactly 2 * bird's."""
    atoms = backend.world.atoms()
    parallel = ConceptToken(
        concept="parallel", token_id=14, token_text=" parallel", variant=" {}"
    )
    with pytest.raises(RankDeficientPairError, match="numerically parallel"):
        build_swap_basis(atoms, layer=1, source=tokens["bird"], target=parallel)


def test_ill_conditioned_pair_is_refused_and_the_limit_is_configurable():
    """Requirement 7. Refused by default; admitted only by an explicit,
    recorded policy change."""
    d_model = 16
    q = torch.eye(d_model, dtype=torch.float64)
    cosine = 1.0 - 1e-9
    atoms = torch.zeros(20, d_model, dtype=torch.float64)
    atoms[0] = q[0]
    atoms[1] = cosine * q[0] + (1 - cosine**2) ** 0.5 * q[1]
    source = ConceptToken("a", 0, " a", " {}")
    target = ConceptToken("b", 1, " b", " {}")
    with pytest.raises(IllConditionedPairError, match="condition number"):
        build_swap_basis(atoms, layer=1, source=source, target=target)
    permissive = StabilityPolicy(max_condition_number=1e9)
    basis = build_swap_basis(atoms, layer=1, source=source, target=target, policy=permissive)
    assert basis.diagnostics["condition_number"] > 1e4
    assert basis.diagnostics["max_condition_number"] == 1e9
    assert basis.diagnostics["numerical_rank"] == 2


def test_identical_tokens_are_refused(backend, tokens):
    with pytest.raises(CoordinateSwapError, match="nothing to exchange"):
        build_swap_basis(
            backend.world.atoms(), layer=1, source=tokens["bird"], target=tokens["bird"]
        )


def test_out_of_range_token_id_is_refused(backend, tokens):
    bad = ConceptToken("nope", 9999, " nope", " {}")
    with pytest.raises(CoordinateSwapError, match="out of range"):
        build_swap_basis(backend.world.atoms(), layer=1, source=tokens["bird"], target=bad)


def test_vector_orientation_mistakes_are_detected(V):
    """Requirement 8. A [2, d_model] basis is the likely mistake, and it is
    named as such rather than failing later on a shape error."""
    assert_vector_orientation(V, d_model=16)
    with pytest.raises(CoordinateSwapError, match=r"lens vectors are\s+in rows"):
        assert_vector_orientation(V.T, d_model=16)
    with pytest.raises(CoordinateSwapError, match="lens vectors are"):
        swap_coordinates(_activation(), V.T.contiguous(), alpha=1.0)
    with pytest.raises(CoordinateSwapError, match=r"must be \[16, 2\]"):
        assert_vector_orientation(torch.zeros(16, 3), d_model=16)
    with pytest.raises(CoordinateSwapError, match="2-D"):
        assert_vector_orientation(torch.zeros(16), d_model=16)


def test_row_vectors_are_refused_by_the_basis_builder(backend, tokens):
    with pytest.raises(CoordinateSwapError, match=r"\[n_atoms, d_model\]"):
        build_swap_basis(
            backend.world.atoms()[0], layer=1, source=tokens["bird"], target=tokens["cat"]
        )


def test_multi_token_concepts_are_refused_not_truncated(backend):
    """Neither variant of ``strawberry`` is one token in the MOCK vocabulary."""

    def encode(text: str) -> list[int]:
        return [1, 2, 3] if "strawberry" in text else [7]

    with pytest.raises(MultiTokenConceptError) as excinfo:
        resolve_concept_token(encode, "strawberry")
    message = str(excinfo.value)
    assert "does not resolve to a single token" in message
    assert "[1, 2, 3]" in message  # every attempt is reported
    assert "prefix" in message
    assert resolve_concept_token(encode, "bird").token_id == 7


def test_concept_resolution_prefers_the_leading_space_variant(backend):
    token = resolve_concept_token(backend.encode_token, "bird")
    assert token.token_id == BIRD_ID
    assert token.token_text == " bird"
    assert token.variant == " {}"


# ------------------------------------------------------ 3. source/target roles


def test_reversing_source_and_target_reverses_the_coordinate_bookkeeping(V, backend, tokens):
    """Requirement 9.

    An exchange is symmetric in its two arguments: the reversed basis produces
    the *same* patched activation, and what reverses is which coordinate is
    reported as source and which as target. That symmetry is itself the
    discriminator — a direction steer ``h +- alpha v`` is antisymmetric under
    the same relabelling, so this test would fail for one.
    """
    h = _activation(seed=23).double()
    forward = mock_bases(
        backend.world, layers=(1,), source=tokens["bird"], target=tokens["cat"]
    )[1]
    reversed_basis = reverse_basis(forward)
    assert reversed_basis.source.concept == "cat"
    assert reversed_basis.target.concept == "bird"
    assert reversed_basis.kind == "reverse_swap"

    patched_forward, record_forward = swap_coordinates(h, forward.V, alpha=1.0)
    patched_reverse, record_reverse = swap_coordinates(h, reversed_basis.V, alpha=1.0)
    assert torch.allclose(patched_forward, patched_reverse, atol=1e-12, rtol=0)
    assert record_reverse["coordinates_before"][0] == pytest.approx(
        list(reversed(record_forward["coordinates_before"][0])), abs=1e-12
    )
    assert record_reverse["coordinates_after"][0] == pytest.approx(
        list(reversed(record_forward["coordinates_after"][0])), abs=1e-12
    )
    # A single-vector steer would not survive the same relabelling.
    steer_forward = h + 1.0 * (forward.V[:, 1] - forward.V[:, 0])
    steer_reverse = h + 1.0 * (reversed_basis.V[:, 1] - reversed_basis.V[:, 0])
    assert not torch.allclose(steer_forward, steer_reverse, atol=1e-6)


def test_selected_vector_basis_matches_full_dictionary_basis(backend, tokens):
    atoms = backend.world.atoms()
    full = build_swap_basis(
        atoms, layer=1, source=tokens["bird"], target=tokens["cat"]
    )
    selected = build_swap_basis_from_vectors(
        atoms[tokens["bird"].token_id],
        atoms[tokens["cat"].token_id],
        layer=1,
        source=tokens["bird"],
        target=tokens["cat"],
    )
    assert torch.equal(selected.V, full.V)
    assert selected.diagnostics == full.diagnostics


def test_reverse_swap_moves_target_evidence_back_to_the_source(backend, tokens, bases):
    """The scientific reverse control: same pair, evidence carrying the target."""
    inputs, candidate_ids, clean = _identity_setup(backend, concept="cat", modality="spoken_audio")
    assert prediction_and_margin(clean, "cat")["prediction"] == "cat"
    reversed_bases = {layer: reverse_basis(basis) for layer, basis in bases.items()}
    result = run_swap_condition(
        backend,
        inputs,
        bases=reversed_bases,
        alpha=1.0,
        candidate_ids=candidate_ids,
        target_concept="bird",
        clean_scores=clean,
    )
    assert result["prediction"] == "bird"
    assert result["prediction_changed"] is True


# ---------------------------------------------------- 4. positions and bands


def test_position_rules_never_reach_candidate_completion_tokens():
    for rule in POSITION_RULES:
        positions = resolve_positions(
            rule, prompt_len=10, seq_len=13, evidence_span=[2, 5]
        )
        assert positions == sorted(set(positions))
        assert max(positions) < 10
        assert min(positions) >= 0
    assert resolve_positions("all_prompt_positions", prompt_len=4, seq_len=6) == [0, 1, 2, 3]
    assert resolve_positions("final_prompt_token_only", prompt_len=4, seq_len=6) == [3]
    assert resolve_positions(
        "evidence_span_only", prompt_len=6, seq_len=6, evidence_span=[1, 4]
    ) == [1, 2, 3]
    assert resolve_positions(
        "non_evidence_prompt_positions", prompt_len=6, seq_len=6, evidence_span=[1, 4]
    ) == [0, 4, 5]


def test_position_rules_refuse_rather_than_guess():
    with pytest.raises(CoordinateSwapError, match="unknown position rule"):
        resolve_positions("everything", prompt_len=4, seq_len=4)
    with pytest.raises(CoordinateSwapError, match="needs an evidence span"):
        resolve_positions("evidence_span_only", prompt_len=4, seq_len=4)
    with pytest.raises(CoordinateSwapError, match="shorter than prompt_len"):
        resolve_positions("all_prompt_positions", prompt_len=8, seq_len=4)
    with pytest.raises(CoordinateSwapError, match="not inside the prompt"):
        resolve_positions(
            "evidence_span_only", prompt_len=4, seq_len=4, evidence_span=[2, 9]
        )
    with pytest.raises(CoordinateSwapError, match="selected no positions"):
        resolve_positions(
            "non_evidence_prompt_positions", prompt_len=3, seq_len=3, evidence_span=[0, 3]
        )


def test_prompt_positions_are_patched_and_candidate_positions_are_not(backend, bases):
    """Requirement 10, measured on activations rather than inferred from stats.

    The candidate token appended by teacher-forced scoring occupies position
    ``prompt_len``; the hook must leave it byte-identical while every prompt
    position moves.
    """
    from jlens.hooks import ActivationRecorder
    from jlens.mmpilot.capability import _extend_tensors

    inputs, candidate_ids, _ = _identity_setup(backend)
    layer = PRIMARY_BAND[-1]
    extended = _extend_tensors(inputs.tensors, inputs.prompt_len, candidate_ids["cat"])
    seq_len = extended["input_ids"].shape[1]
    assert seq_len > inputs.prompt_len

    with ActivationRecorder(backend.blocks, at=[layer]) as recorder:
        backend.forward_logits(extended)
        clean = recorder.activations[layer].detach().clone()
    with coordinate_swap_layer(
        backend.blocks,
        bases[layer],
        alpha=1.0,
        prompt_len=inputs.prompt_len,
        position_rule=PRIMARY_POSITION_RULE,
    ) as stats:
        with ActivationRecorder(backend.blocks, at=[layer]) as recorder:
            backend.forward_logits(extended)
            patched = recorder.activations[layer].detach().clone()

    assert stats["seq_len"] == seq_len
    assert stats["positions"] == list(range(inputs.prompt_len))
    assert stats["n_candidate_positions_skipped"] == seq_len - inputs.prompt_len
    for position in range(inputs.prompt_len):
        assert not torch.allclose(
            clean[0, position], patched[0, position], atol=1e-6
        ), f"prompt position {position} was not patched"
    for position in range(inputs.prompt_len, seq_len):
        assert torch.equal(clean[0, position], patched[0, position]), (
            f"candidate-completion position {position} was modified"
        )


def test_every_band_layer_is_patched_and_others_are_untouched(backend, bases):
    """Requirements 11 and 12, in one forward pass.

    "Patched" is asserted on each layer's own recorded update, not on whether
    that layer's output ends up differing from the clean run. In this synthetic
    world the carry blocks nearly commute with the exchange, so an
    even-numbered patch lands almost back on the clean state — the involution
    :func:`~jlens.mmpilot.coordinate_swap_mock.band_parity_diagnostic`
    measures. A layer whose output coincided with the clean one is still a
    layer that was patched, and the update norms say so.
    """
    from jlens.hooks import ActivationRecorder

    inputs, _, _ = _identity_setup(backend)
    all_layers = list(range(backend.n_layers))
    with ActivationRecorder(backend.blocks, at=all_layers) as recorder:
        clean_logits = backend.forward_logits(inputs.tensors).clone()
        clean = {i: recorder.activations[i].detach().clone() for i in all_layers}
    with coordinate_swap_band(
        backend.blocks,
        bases,
        alpha=1.0,
        prompt_len=inputs.prompt_len,
        position_rule=PRIMARY_POSITION_RULE,
    ) as stats:
        with ActivationRecorder(backend.blocks, at=all_layers) as recorder:
            patched_logits = backend.forward_logits(inputs.tensors).clone()
            patched = {i: recorder.activations[i].detach().clone() for i in all_layers}

    assert sorted(stats) == sorted(PRIMARY_BAND)
    for layer in PRIMARY_BAND:
        assert stats[layer]["n_forward_passes"] == 1
        assert stats[layer]["n_positions"] == inputs.prompt_len
        assert stats[layer]["positions"] == list(range(inputs.prompt_len))
        update_norms = stats[layer]["swap"]["update_norm"]
        assert len(update_norms) == inputs.prompt_len
        assert min(update_norms) > 0.0, f"layer {layer} applied a zero update"
    # Requirement 12: layers before the band are byte-identical. Layers after it
    # necessarily differ — that is the intervention propagating, not a leak.
    for layer in range(min(PRIMARY_BAND)):
        assert torch.equal(clean[layer], patched[layer])
    assert not torch.allclose(clean[max(PRIMARY_BAND)], patched[max(PRIMARY_BAND)], atol=1e-6)
    assert not torch.allclose(clean_logits, patched_logits, atol=1e-6)


def test_a_single_layer_patch_touches_exactly_that_layer(backend, bases):
    """Requirement 12 again, without the band's downstream propagation in the
    way: only the hooked layer's output moves, and every earlier one is
    byte-identical."""
    from jlens.hooks import ActivationRecorder

    inputs, _, _ = _identity_setup(backend)
    layer = PRIMARY_BAND[-1]
    all_layers = list(range(backend.n_layers))
    with ActivationRecorder(backend.blocks, at=all_layers) as recorder:
        backend.forward_logits(inputs.tensors)
        clean = {i: recorder.activations[i].detach().clone() for i in all_layers}
    with coordinate_swap_layer(
        backend.blocks, bases[layer], alpha=1.0, prompt_len=inputs.prompt_len
    ):
        with ActivationRecorder(backend.blocks, at=all_layers) as recorder:
            backend.forward_logits(inputs.tensors)
            patched = {i: recorder.activations[i].detach().clone() for i in all_layers}
    for earlier in range(layer):
        assert torch.equal(clean[earlier], patched[earlier]), earlier
    assert not torch.allclose(clean[layer], patched[layer], atol=1e-6)


def test_each_layer_recomputes_its_own_coordinates(backend, bases):
    """Not one update computed once and replayed: the carry blocks rescale the
    residual, so each layer's pre-swap coordinates must differ."""
    inputs, _, _ = _identity_setup(backend)
    with coordinate_swap_band(
        backend.blocks,
        bases,
        alpha=1.0,
        prompt_len=inputs.prompt_len,
        position_rule=PRIMARY_POSITION_RULE,
    ) as stats:
        backend.forward_logits(inputs.tensors)
    seen = [
        tuple(stats[layer]["swap"]["coordinates_before"][1]) for layer in sorted(PRIMARY_BAND)
    ]
    assert len(set(seen)) == len(seen), f"layers reused one update: {seen}"


def test_hooks_are_removed_on_every_path(backend, bases):
    inputs, _, _ = _identity_setup(backend)
    before = [len(block._forward_hooks) for block in backend.blocks]
    with pytest.raises(RuntimeError, match="boom"):
        with coordinate_swap_band(
            backend.blocks, bases, alpha=1.0, prompt_len=inputs.prompt_len
        ):
            raise RuntimeError("boom")
    assert [len(block._forward_hooks) for block in backend.blocks] == before
    # And when the hook itself raises during the forward pass.
    with pytest.raises(CoordinateSwapError):
        with coordinate_swap_band(
            backend.blocks, bases, alpha=1.0, prompt_len=inputs.prompt_len + 500
        ):
            backend.forward_logits(inputs.tensors)
    assert [len(block._forward_hooks) for block in backend.blocks] == before


def test_hook_refuses_an_unfrozen_block(backend, bases):
    block = backend.blocks[PRIMARY_BAND[0]]
    block.register_parameter("scratch", torch.nn.Parameter(torch.zeros(1)))
    try:
        with pytest.raises(CoordinateSwapError, match="requires grad"):
            with coordinate_swap_layer(backend.blocks, bases[PRIMARY_BAND[0]], alpha=1.0, prompt_len=4):
                pass
    finally:
        del block._parameters["scratch"]


def test_layer_band_must_be_contiguous_and_validated():
    band = build_layer_band(35, 40, validated_layers=(35, 36, 37, 38, 39, 40), n_layers=42)
    assert band.layers == (35, 36, 37, 38, 39, 40)
    with pytest.raises(LayerBandError, match="untouched confirmation gate"):
        build_layer_band(35, 40, validated_layers=(35, 38, 40))
    with pytest.raises(LayerBandError, match="empty band"):
        build_layer_band(40, 35, validated_layers=(35, 40))
    with pytest.raises(LayerBandError, match="out of range"):
        build_layer_band(40, 45, validated_layers=tuple(range(50)), n_layers=42)


def test_layer_band_gate_matches_the_published_lens_record():
    """The published calibration run confirmed 35, 38 and 40 — not a band.
    A contiguous band over them is therefore refused until an earlier-layer
    calibration widens the validated set."""
    from jlens.mmpilot.published_lens import (
        CONFIRMED_LAYERS,
        FAILED_CONFIRMATION_LAYERS,
    )

    assert CONFIRMED_LAYERS == (35, 38, 40)
    assert 32 in FAILED_CONFIRMATION_LAYERS
    with pytest.raises(LayerBandError, match=r"\[36, 37\]"):
        build_layer_band(35, 38, validated_layers=CONFIRMED_LAYERS, n_layers=42)
    assert build_layer_band(35, 35, validated_layers=CONFIRMED_LAYERS).layers == (35,)


def test_band_needs_a_dictionary_for_every_layer(backend, tokens):
    atoms = backend.world.atoms_by_layer((1, 2))
    with pytest.raises(CoordinateSwapError, match=r"no dictionary atoms for layers \[3\]"):
        build_swap_bases(atoms, layers=(1, 2, 3), source=tokens["bird"], target=tokens["cat"])


# ----------------------------------------------------- 5. invariance at the model


def test_capture_hook_and_zero_intervention_preserve_the_model(backend):
    """Requirements 13 and 14, through the repository's own gate."""
    from jlens.mmpilot.backend import run_invariance_gate

    inputs, _, _ = _identity_setup(backend)
    report = run_invariance_gate(backend, inputs, list(PRIMARY_BAND))
    assert report["passed"] is True
    assert report["capture_noop"]["passed"] is True


def test_zero_alpha_swap_preserves_complete_candidate_scores(backend, bases):
    """Requirement 14 for the *coordinate-swap* path specifically: the hooks
    are installed, the solve runs, and every candidate's complete sequence
    score is unchanged."""
    inputs, candidate_ids, clean = _identity_setup(backend)
    result = run_swap_condition(
        backend,
        inputs,
        bases=bases,
        alpha=0.0,
        candidate_ids=candidate_ids,
        target_concept="cat",
        clean_scores=clean,
    )
    for name, score in result["candidate_scores"].items():
        assert score["sum_logprob"] == pytest.approx(clean[name]["sum_logprob"], abs=1e-9)
        assert score["token_ids"] == clean[name]["token_ids"]
        assert score["n_tokens"] == len(candidate_ids[name])
    assert result["prediction_changed"] is False
    assert result["layers_patched"] == sorted(PRIMARY_BAND)


# ------------------------------------------------ 6. what the MOCK world shows


def test_mock_identity_swaps_in_every_evidence_modality(backend, bases):
    for modality in ("text", "image", "spoken_audio"):
        inputs, candidate_ids, clean = _identity_setup(backend, modality=modality)
        assert prediction_and_margin(clean, "cat")["prediction"] == "bird"
        result = run_swap_condition(
            backend,
            inputs,
            bases=bases,
            alpha=1.0,
            candidate_ids=candidate_ids,
            target_concept="cat",
            clean_scores=clean,
        )
        assert result["prediction"] == "cat", modality
        assert result["intervention_family"] == INTERVENTION_FAMILY


def test_mock_downstream_property_recomputes_only_before_the_reasoning_layer(backend, tokens):
    """Identity replacement and downstream recomputation are separate claims,
    and the MOCK separates them: a band before the reasoning layer moves both,
    a band starting at its output moves only the identity."""
    before_band = mock_bases(
        backend.world, layers=PRIMARY_BAND, source=tokens["bird"], target=tokens["cat"]
    )
    after_band = mock_bases(
        backend.world, layers=POST_REASONING_BAND, source=tokens["bird"], target=tokens["cat"]
    )
    assert max(PRIMARY_BAND) < REASONING_LAYER <= min(POST_REASONING_BAND)

    identity_inputs, identity_ids, identity_clean = _identity_setup(backend)
    property_inputs, property_ids, property_clean = _property_setup(backend)
    assert prediction_and_margin(property_clean, "four")["prediction"] == "two"

    def run(inputs, ids, clean, band, target):
        return run_swap_condition(
            backend, inputs, bases=band, alpha=1.0, candidate_ids=ids,
            target_concept=target, clean_scores=clean,
        )["prediction"]

    assert run(identity_inputs, identity_ids, identity_clean, before_band, "cat") == "cat"
    assert run(property_inputs, property_ids, property_clean, before_band, "four") == "four"
    assert run(identity_inputs, identity_ids, identity_clean, after_band, "cat") == "cat"
    assert run(property_inputs, property_ids, property_clean, after_band, "four") == "two"


def test_mock_controls_do_not_reproduce_the_swap(backend, tokens, bases):
    inputs, candidate_ids, clean = _identity_setup(backend)

    def predict(band, alpha=1.0):
        return run_swap_condition(
            backend, inputs, bases=band, alpha=alpha, candidate_ids=candidate_ids,
            target_concept="cat", clean_scores=clean,
        )["prediction"]

    assert predict(bases, alpha=1.0) == "cat"
    assert predict(bases, alpha=0.0) == "bird"
    random_band = {
        layer: random_two_direction_basis(basis, seed=1000 + layer)
        for layer, basis in bases.items()
    }
    assert predict(random_band) == "bird"
    unrelated = mock_bases(
        backend.world, layers=PRIMARY_BAND, source=tokens["dog"], target=tokens["car"]
    )
    assert predict(unrelated) == "bird"


def test_random_control_matches_the_norms_it_replaces(bases):
    basis = bases[PRIMARY_BAND[0]]
    control = random_two_direction_basis(basis, seed=99)
    assert control.kind == "random_two_direction_norm_matched"
    assert control.diagnostics["source_norm"] == pytest.approx(
        basis.diagnostics["source_norm"], rel=1e-12
    )
    assert control.diagnostics["target_norm"] == pytest.approx(
        basis.diagnostics["target_norm"], rel=1e-12
    )
    assert control.diagnostics["random_seed"] == 99
    assert control.diagnostics["matched_to_V_checksum"] == basis.diagnostics["V_checksum"]
    again = random_two_direction_basis(basis, seed=99)
    assert torch.equal(control.V, again.V)


def test_position_rules_localize_the_effect_in_the_mock(backend, bases):
    """Before the MOCK's broadcast layer the identity lives only at the
    evidence positions, so ``non_evidence_prompt_positions`` is a null control
    by construction — and ``final_prompt_token_only``, the completed pilot's
    rule, is *not* equivalent to the paper's all-positions protocol."""
    inputs, candidate_ids, clean = _identity_setup(backend)
    outcomes = {}
    for rule in POSITION_RULES:
        outcomes[rule] = run_swap_condition(
            backend, inputs, bases=bases, alpha=1.0, candidate_ids=candidate_ids,
            target_concept="cat", clean_scores=clean, position_rule=rule,
        )["prediction"]
    assert outcomes["all_prompt_positions"] == "cat"
    assert outcomes["evidence_span_only"] == "cat"
    assert outcomes["non_evidence_prompt_positions"] == "bird"
    assert outcomes["final_prompt_token_only"] == "bird"


def test_direct_answer_vector_control_changes_the_property_without_the_identity(backend, tokens):
    """The control that stops a downstream change being called recomputation.

    Inserting the ``four`` lens vector directly moves the property answer while
    leaving the identity alone — so a property change on its own is not
    evidence that the model re-derived anything.
    """
    from jlens.interventions import residual_intervention

    atoms = backend.world.atoms()
    delta = direct_answer_vector(atoms, answer_token_id=FOUR_ID, scale=8.0)
    property_inputs, property_ids, property_clean = _property_setup(backend)
    identity_inputs, identity_ids, identity_clean = _identity_setup(backend)
    # The final *prompt* token, resolved explicitly: a negative index would
    # resolve against the teacher-forced sequence, which is longer.
    with residual_intervention(
        backend.blocks,
        PRIMARY_BAND[-1],
        position=property_inputs.final_prompt_position,
        delta=delta,
        multiplier=1.0,
    ):
        property_scores = score_candidate_sequences(backend, property_inputs, property_ids)
    with residual_intervention(
        backend.blocks,
        PRIMARY_BAND[-1],
        position=identity_inputs.final_prompt_position,
        delta=delta,
        multiplier=1.0,
    ):
        identity_scores = score_candidate_sequences(backend, identity_inputs, identity_ids)
    assert prediction_and_margin(property_clean, "four")["prediction"] == "two"
    assert prediction_and_margin(property_scores, "four")["prediction"] == "four"
    assert prediction_and_margin(identity_scores, "cat")["prediction"] == "bird"
    assert prediction_and_margin(identity_clean, "cat")["prediction"] == "bird"


def test_direct_answer_vector_refuses_bad_inputs(backend):
    atoms = backend.world.atoms()
    with pytest.raises(CoordinateSwapError, match="out of range"):
        direct_answer_vector(atoms, answer_token_id=9999)
    with pytest.raises(CoordinateSwapError, match=r"\[n_atoms, d_model\]"):
        direct_answer_vector(atoms[0], answer_token_id=FOUR_ID)


def test_band_parity_diagnostic_reports_the_involution(backend, tokens):
    inputs, _, _ = _identity_setup(backend)
    rows = band_parity_diagnostic(
        backend, inputs, source=tokens["bird"], target=tokens["cat"], max_band_length=4
    )
    assert [row["swapped_to_target"] for row in rows] == [True, False, True, False]
    assert all(row["clean_prediction"] == "bird" for row in rows)


def test_mock_world_geometry_is_what_it_claims(backend):
    world = backend.world
    diagnostics = basis_diagnostics(world.V)
    assert diagnostics["cosine"] == pytest.approx(0.45, abs=1e-12)
    assert diagnostics["source_norm"] == pytest.approx(0.8, abs=1e-12)
    assert diagnostics["target_norm"] == pytest.approx(1.3, abs=1e-12)
    assert diagnostics["numerical_rank"] == 2
    assert diagnostics["condition_number"] < 10
    # The property directions really are outside the lens plane.
    for direction in (world.u_two, world.u_four):
        assert torch.allclose(
            orthogonal_residual(direction, world.V), direction, atol=1e-12
        )
    atoms = world.atoms()
    assert torch.equal(atoms[BIRD_ID], world.v_source)
    assert torch.equal(atoms[CAT_ID], world.v_target)


def test_mock_frame_is_deterministic():
    assert torch.equal(SwapMockWorld().Q, SwapMockWorld().Q)
    assert torch.allclose(
        SwapMockWorld().Q.T @ SwapMockWorld().Q, torch.eye(16, dtype=torch.float64), atol=1e-12
    )


# ----------------------------------------------- 7. specs and fingerprinting


def _spec(**overrides):
    backend = SwapMockBackend()
    tokens = mock_concept_tokens(backend)
    defaults = dict(
        source=tokens["bird"],
        target=tokens["cat"],
        layer_band=build_layer_band(
            1, 3, validated_layers=MOCK_VALIDATED_LAYERS, n_layers=8
        ),
        alpha=1.0,
        position_rule=PRIMARY_POSITION_RULE,
        control_kind="coordinate_swap",
        lens_checksums=mock_lens_checksums(PRIMARY_BAND),
        model_revision="rev-a",
        processor_revision="proc-a",
    )
    defaults.update(overrides)
    return build_spec(**defaults)


def test_spec_records_every_convention():
    spec = _spec()
    payload = spec.to_dict()
    assert payload["intervention_family"] == INTERVENTION_FAMILY
    assert payload["method_version"] == METHOD_VERSION
    assert payload["source_token_id"] == BIRD_ID
    assert payload["target_token_id"] == CAT_ID
    assert payload["source_token"] == " bird"
    assert payload["layer_band"] == [1, 2, 3]
    assert payload["position_rule"] == PRIMARY_POSITION_RULE
    assert "W_U @ J_l" in payload["vector_convention"]
    assert "final_norm_weight_folded=False" in payload["normalization"]
    assert "float64" in payload["solve_policy"]
    assert payload["stability"]["max_condition_number"] == 1e4
    assert set(payload["lens_checksums"]) == {"1", "2", "3"}
    assert json.loads(json.dumps(payload)) == payload


def test_spec_refuses_unknown_vocabulary_and_missing_checksums():
    with pytest.raises(CoordinateSwapError, match="unknown position rule"):
        _spec(position_rule="somewhere")
    with pytest.raises(CoordinateSwapError, match="unknown control kind"):
        _spec(control_kind="vibes")
    with pytest.raises(CoordinateSwapError, match="no lens checksum recorded"):
        _spec(lens_checksums={1: "sha256:x"})
    with pytest.raises(CoordinateSwapError, match="layer_band is empty"):
        _spec(layer_band=())


def test_fingerprint_carries_every_required_field():
    fingerprint = coordinate_swap_fingerprint(
        _spec(), alphas=(0.0, 0.5, 1.0, 2.0), controls=CONTROL_KINDS
    )
    for key in (
        "intervention_family",
        "coordinate_swap_method_version",
        "source_token_id",
        "target_token_id",
        "vector_convention",
        "normalization",
        "solve_policy",
        "condition_number_threshold",
        "rank_tolerance",
        "alphas",
        "position_rule",
        "prompt_length_boundary_rule",
        "layer_band",
        "lens_checksums_by_layer",
        "model_revision",
        "processor_revision",
        "audio_protocol_fingerprint",
        "controls",
        "control_config",
        "spec_digest",
    ):
        assert key in fingerprint, key
    assert fingerprint["intervention_family"] == INTERVENTION_FAMILY
    assert fingerprint["controls"] == sorted(CONTROL_KINDS)
    with pytest.raises(CoordinateSwapError, match="unknown control kinds"):
        coordinate_swap_fingerprint(_spec(), alphas=(1.0,), controls=("vibes",))
    with pytest.raises(CoordinateSwapError, match="alphas must all be finite"):
        coordinate_swap_fingerprint(_spec(), alphas=(float("inf"),), controls=())


def test_audio_protocol_fingerprint_travels_with_the_spec():
    spec = _spec(audio_protocol_fingerprint="sha256:audio-protocol")
    assert coordinate_swap_fingerprint(spec, alphas=(1.0,), controls=())[
        "audio_protocol_fingerprint"
    ] == "sha256:audio-protocol"


# ---------------------------------------------------- 8. resume and isolation


def _store(tmp_path, fingerprint_extra=None, *, spec=None, alphas=(0.0, 1.0)):
    intervention = coordinate_swap_fingerprint(
        spec or _spec(), alphas=alphas, controls=("coordinate_swap", "zero")
    )
    if fingerprint_extra:
        intervention.update(fingerprint_extra)
    fingerprint = RunFingerprint(
        mode="coordinate_swap",
        model_repo_id="google/gemma-4-E4B-it",
        model_revision="rev-a",
        processor_revision="proc-a",
        layers=(1, 2, 3),
        lens_checksum="sha256:lens",
        manifest_checksum="sha256:manifest",
        split_id="split-1",
        intervention_config=intervention,
    )
    return UnitStore(tmp_path / "run", fingerprint)


def test_a_coordinate_swap_run_cannot_resume_from_a_steering_run(tmp_path):
    """Requirement 17, both ways: the digest gate refuses, and the family check
    says *why*."""
    steering = RunFingerprint(
        mode="pilot",
        model_repo_id="google/gemma-4-E4B-it",
        model_revision="rev-a",
        processor_revision="proc-a",
        layers=(1, 2, 3),
        lens_checksum="sha256:lens",
        manifest_checksum="sha256:manifest",
        split_id="split-1",
        intervention_config={
            "alphas": list(causal.DEFAULT_ALPHAS),
            "control_kinds": list(causal.CONTROL_KINDS),
        },
    )
    UnitStore(tmp_path / "run", steering).open()
    with pytest.raises(IncompatibleStateError, match="different configuration"):
        _store(tmp_path).open()
    stored = json.loads((tmp_path / "run" / "fingerprint.json").read_text(encoding="utf-8"))
    with pytest.raises(CoordinateSwapError, match="source-derived J-space steering"):
        assert_coordinate_swap_artifacts(stored)


def test_coordinate_swap_artifacts_are_recognized_and_version_gated():
    good = coordinate_swap_fingerprint(_spec(), alphas=(1.0,), controls=())
    assert_coordinate_swap_artifacts({"intervention_config": good})
    assert_coordinate_swap_artifacts(good)
    with pytest.raises(CoordinateSwapError, match="method version"):
        assert_coordinate_swap_artifacts({**good, "coordinate_swap_method_version": "v0"})
    with pytest.raises(CoordinateSwapError, match="belong to intervention family"):
        assert_coordinate_swap_artifacts({"intervention_family": "something_else"})
    with pytest.raises(CoordinateSwapError, match="no stored fingerprint"):
        assert_coordinate_swap_artifacts(None)


@pytest.mark.parametrize(
    "label,overrides,extra",
    [
        ("source token", {"source": ConceptToken("dog", 12, " dog", " {}")}, None),
        ("target token", {"target": ConceptToken("car", 13, " car", " {}")}, None),
        ("layer band", {"layer_band": build_layer_band(1, 2, validated_layers=MOCK_VALIDATED_LAYERS)}, None),
        ("position rule", {"position_rule": "final_prompt_token_only"}, None),
        ("model revision", {"model_revision": "rev-b"}, None),
        ("processor revision", {"processor_revision": "proc-b"}, None),
        ("lens checksum", {"lens_checksums": {1: "x", 2: "y", 3: "z"}}, None),
        ("method version", None, {"coordinate_swap_method_version": "jlens.mmpilot.coordinate_swap.v99"}),
        ("condition threshold", None, {"condition_number_threshold": 1e9}),
        ("controls", None, {"controls": ["coordinate_swap"]}),
    ],
)
def test_changing_any_bound_field_refuses_resume(tmp_path, label, overrides, extra):
    """Requirement 18."""
    _store(tmp_path).open()
    changed = _store(
        tmp_path,
        fingerprint_extra=extra,
        spec=_spec(**overrides) if overrides else None,
    )
    with pytest.raises(IncompatibleStateError, match="different configuration"):
        changed.open()


def test_changing_alpha_refuses_resume(tmp_path):
    _store(tmp_path, alphas=(0.0, 1.0)).open()
    with pytest.raises(IncompatibleStateError):
        _store(tmp_path, alphas=(0.0, 1.0, 2.0)).open()


def test_an_unchanged_configuration_resumes(tmp_path):
    first = _store(tmp_path)
    assert first.open() == "starting"
    first.save("intervention", "unit-1", {"prediction": "cat"})
    second = _store(tmp_path)
    assert second.open() == "resuming"
    assert second.load("intervention", "unit-1") == {"prediction": "cat"}


# ----------------------------------------- 9. nothing existing was disturbed


def test_existing_steering_direction_estimation_is_unchanged():
    """Requirement 15. A characterization test on
    :mod:`jlens.mmpilot.causal`: rectified difference in mean codes, mapped
    through the dictionary, unit-normalized. Nothing in this patch touches it.
    """
    assert causal.CONTROL_KINDS == (
        "source_concept",
        "zero",
        "random_norm_matched",
        "unrelated_concept",
        "raw_residual_difference",
    )
    assert causal.DEFAULT_ALPHAS == (0.0, 0.25, 0.5, 1.0)

    world = SwapMockWorld()

    class _Dictionary:
        atoms = world.atoms().to(torch.float32)
        d_model = world.d_model
        device = torch.device("cpu")

    record = causal.estimate_concept_direction(
        [{BIRD_ID: 2.0, TWO_ID: 0.5}],
        [{BIRD_ID: 0.5, TWO_ID: 0.5}],
        _Dictionary(),
        concept="bird",
        source_modality="text",
        layer=1,
        positive_ids=["p"],
        negative_ids=["n"],
        lens_checksum="sha256:lens",
    )
    assert record["kind"] == "source_concept"
    assert record["support"] == [BIRD_ID]
    assert record["coefficients"] == [pytest.approx(1.5)]
    assert record["uses_target_modality_data"] is False
    direction = torch.tensor(record["unit_direction"], dtype=torch.float64)
    assert float(direction.norm()) == pytest.approx(1.0, abs=1e-6)
    expected = world.v_source / world.v_source.norm()
    assert torch.allclose(direction, expected.to(torch.float64), atol=1e-6)
    # And it is emphatically not a coordinate swap: no target token is involved.
    assert "target_token_id" not in record



def test_completed_run_fingerprints_still_recompute():
    """Requirement 16. The tracked completed run's stored config fingerprint
    must still be reproducible from its stored config with today's code."""
    checked = 0
    for metadata_path in sorted((REPO_ROOT / "runs").glob("*/run_metadata.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert config_fingerprint(metadata["config"]) == metadata["config_fingerprint"], (
            f"{metadata_path} no longer reproduces its own fingerprint"
        )
        checked += 1
    assert checked >= 1


def test_run_fingerprint_digest_algorithm_is_pinned():
    """Requirement 16, forward-looking: a pinned digest so a future change to
    :class:`~jlens.mmpilot.store.RunFingerprint` cannot silently invalidate
    every completed run directory."""
    fingerprint = RunFingerprint(
        mode="pilot",
        model_repo_id="google/gemma-4-E4B-it",
        model_revision="rev",
        processor_revision="rev",
        layers=(35, 38, 40),
        lens_checksum="sha256:abc",
        manifest_checksum="sha256:def",
        split_id="split-1",
        intervention_config={
            "alphas": [0.0, 0.25, 0.5, 1.0],
            "control_kinds": ["source_concept", "zero"],
        },
    )
    assert fingerprint.digest == (
        "sha256:363517a07f013052c20114f0607cdbac6306de8f53c20a877bedec919bba8e0d"
    )
    assert "selection_config" not in fingerprint.to_dict()


def test_the_protected_calibration_surface_is_untouched():
    """The active 1,000-prompt calibration owns these modules. This patch adds
    a sibling; if it ever starts importing or shadowing one of them, this
    fails."""
    import jlens.mmpilot.coordinate_swap as module
    import jlens.mmpilot.coordinate_swap_mock as mock_module

    source = Path(module.__file__).read_text(encoding="utf-8")
    mock_source = Path(mock_module.__file__).read_text(encoding="utf-8")
    for forbidden in ("jlens.calibration", "jlens.autoencoder", "rgext_real"):
        assert forbidden not in source, forbidden
        assert forbidden not in mock_source, forbidden
