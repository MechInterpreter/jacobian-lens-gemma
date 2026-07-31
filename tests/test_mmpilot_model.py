# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Model-facing mechanics: candidate scoring, the two invariance checks, lens
compatibility, sparse pursuit, direction estimation, intervention sign, and
control behaviour. All CPU, all against the deterministic mock."""

import math

import pytest
import torch

from jlens.mmpilot import mock as K
from jlens.mmpilot.backend import (
    InvarianceError,
    ModalityUnsupportedError,
    check_capture_noop,
    check_zero_intervention,
    contiguous_token_range,
    run_invariance_gate,
)
from jlens.mmpilot.capability import (
    _extend_tensors,
    build_prompt,
    build_question,
    candidate_token_ids,
    capability_summary,
    prediction_and_margin,
    score_candidate_sequences,
)
from jlens.mmpilot.causal import (
    estimate_concept_direction,
    random_control_direction,
    raw_residual_direction,
    run_condition,
    unit_direction_tensor,
)
from jlens.mmpilot.jspace import (
    LensIncompatibleError,
    build_dictionary,
    capture_final_prompt_activations,
    code_map,
    jspace_code,
    validate_lens,
)
from jlens.pursuit import PursuitSettings

CONCEPTS = sorted(K.MOCK_CONCEPTS)
LAYER = K.MOCK_LAYERS[-1]


@pytest.fixture(scope="module")
def backend():
    return K.MockPilotBackend()


@pytest.fixture(scope="module")
def question():
    return build_question(CONCEPTS)


@pytest.fixture(scope="module")
def candidates(backend):
    return candidate_token_ids(backend, CONCEPTS)


def text_inputs(backend, question, concept):
    caption = f"a photo of a {concept} in a room"
    return backend.build_inputs(
        prompt=build_prompt(question, modality="text", caption=caption), modality="text"
    )


@pytest.fixture(scope="module")
def dictionary(backend):
    return build_dictionary(
        K.mock_lens(),
        LAYER,
        backend.unembedding_weight(),
        dtype=torch.float32,
        build_chunk_rows=None,
    )


# ------------------------------------------------------------------ scoring


def test_candidates_are_multi_token_so_first_token_scoring_would_be_wrong(candidates):
    assert all(len(ids) > 1 for ids in candidates.values())


def test_complete_sequence_score_equals_the_sum_of_its_token_logprobs(
    backend, question, candidates
):
    inputs = text_inputs(backend, question, "dog")
    scores = score_candidate_sequences(backend, inputs, candidates)
    ids = candidates["dog"]
    tensors = dict(inputs.tensors)
    tensors["input_ids"] = torch.cat(
        [tensors["input_ids"], torch.tensor([ids], dtype=torch.long)], dim=1
    )
    tensors["attention_mask"] = torch.ones(1, tensors["input_ids"].shape[1], dtype=torch.long)
    log_probs = torch.log_softmax(backend.forward_logits(tensors)[0], dim=-1)
    expected = sum(
        float(log_probs[inputs.prompt_len - 1 + offset, token])
        for offset, token in enumerate(ids)
    )
    assert scores["dog"]["sum_logprob"] == pytest.approx(expected, abs=1e-5)
    assert scores["dog"]["mean_logprob"] == pytest.approx(expected / len(ids), abs=1e-5)


def test_scoring_does_not_disturb_the_media_tensors(backend, question, candidates):
    world = backend.world
    evidence = world.evidence(
        concepts_present=["cat"], modality="image", nuisance_key="probe"
    )
    inputs = backend.build_inputs(
        prompt=build_prompt(question, modality="image"), modality="image", image=evidence
    )
    before = inputs.tensors["evidence"].clone()
    score_candidate_sequences(backend, inputs, candidates)
    assert torch.equal(inputs.tensors["evidence"], before)


def test_candidate_extension_never_extends_gemma_audio_feature_mask():
    prompt_len = 4
    audio_mask = torch.tensor([[True, True, False, False]])
    tensors = {
        "input_ids": torch.tensor([[7, 8, 9, 10]]),
        "attention_mask": torch.ones(1, prompt_len, dtype=torch.long),
        "mm_token_type_ids": torch.tensor([[1, 1, 0, 0]]),
        "position_ids": torch.arange(prompt_len).unsqueeze(0),
        "input_features": torch.randn(1, prompt_len, 3),
        # Deliberately the same width as the prompt: shape is not semantics.
        "input_features_mask": audio_mask,
    }

    extended = _extend_tensors(tensors, prompt_len, [11, 12])

    assert extended["input_ids"].tolist() == [[7, 8, 9, 10, 11, 12]]
    assert extended["attention_mask"].tolist() == [[1, 1, 1, 1, 1, 1]]
    assert extended["mm_token_type_ids"].tolist() == [[1, 1, 0, 0, 0, 0]]
    assert extended["position_ids"].tolist() == [[0, 1, 2, 3, 4, 5]]
    assert extended["input_features"] is tensors["input_features"]
    assert extended["input_features_mask"] is audio_mask
    assert extended["input_features_mask"].shape == (1, prompt_len)


def test_prediction_and_margin_are_consistent(backend, question, candidates):
    inputs = text_inputs(backend, question, "pizza")
    scores = score_candidate_sequences(backend, inputs, candidates)
    verdict = prediction_and_margin(scores, "pizza")
    assert verdict["prediction"] == "pizza"
    assert verdict["correct"]
    assert verdict["target_margin"] > 0


def test_capability_summary_requires_every_available_modality():
    records = [
        {"concept": "dog", "modality": "text", "correct": True, "target_margin": 1.0},
        {"concept": "dog", "modality": "image", "correct": False, "target_margin": -1.0},
    ]
    summary = capability_summary(records, threshold=0.7, modalities=["text", "image"])
    assert summary["retained_concepts"] == []
    assert summary["per_concept"]["dog"]["text"]["passed"]
    assert not summary["per_concept"]["dog"]["image"]["passed"]


def test_blocked_modality_raises_rather_than_substituting(question):
    backend = K.MockPilotBackend(supports_audio=False)
    assert not backend.supports("spoken_audio")
    with pytest.raises(ModalityUnsupportedError):
        backend.build_inputs(
            prompt=build_prompt(question, modality="spoken_audio"),
            modality="spoken_audio",
            audio=torch.zeros(K.MOCK_D_MODEL),
        )


# -------------------------------------------------------- invariance checks


def test_capture_hook_does_not_change_logits(backend, question):
    inputs = text_inputs(backend, question, "bus")
    report = check_capture_noop(backend, inputs, K.MOCK_LAYERS)
    assert report["passed"]
    assert report["max_abs_logit_diff"] == pytest.approx(0.0, abs=1e-6)


def test_zero_coefficient_intervention_reproduces_the_clean_run(backend, question):
    inputs = text_inputs(backend, question, "bus")
    report = check_zero_intervention(backend, inputs, LAYER)
    assert report["passed"]
    assert report["max_abs_logit_diff"] == pytest.approx(0.0, abs=1e-6)
    assert report["resolved_position"] == inputs.prompt_len - 1


class _DriftingBackend:
    """Wraps the mock so the second forward pass of each check disagrees with
    the first — i.e. exactly the situation the gate exists to catch."""

    def __init__(self, inner):
        self._inner = inner
        self.d_model = inner.d_model
        self.n_layers = inner.n_layers
        self.calls = 0

    @property
    def blocks(self):
        return self._inner.blocks

    def forward_logits(self, tensors):
        self.calls += 1
        return self._inner.forward_logits(tensors) + (1.0 if self.calls % 2 == 0 else 0.0)


def test_the_gate_raises_when_the_forward_pass_is_perturbed(backend, question):
    inputs = text_inputs(backend, question, "bus")
    drifting = _DriftingBackend(backend)
    with pytest.raises(InvarianceError, match="capture hook changed logits"):
        run_invariance_gate(drifting, inputs, [LAYER])
    drifting.calls = 0
    with pytest.raises(InvarianceError, match="zero-coefficient"):
        check_zero_intervention(drifting, inputs, LAYER)


def test_contiguous_token_range_refuses_to_guess():
    ids = torch.tensor([[5, 9, 9, 9, 5]])
    assert contiguous_token_range(ids, 9) == [1, 4]
    assert contiguous_token_range(torch.tensor([[9, 5, 9]]), 9) is None
    assert contiguous_token_range(ids, None) is None


# ------------------------------------------------------------------- lens


def test_validate_lens_accepts_a_matching_artifact():
    report = validate_lens(
        K.mock_lens(),
        lens_path="mock",
        lens_checksum="sha256:lens",
        layers=K.MOCK_LAYERS,
        model_repo_id="mock",
        model_revision="rev-a",
        expect_model_repo_id="mock",
        expect_model_revision="rev-a",
        expect_d_model=K.MOCK_D_MODEL,
        expect_checksum="sha256:lens",
    )
    assert report["frozen"] is True
    assert report["calibration_modality"] == "text"
    assert report["conventions"]["final_norm_weight_folded"] is False


@pytest.mark.parametrize(
    "overrides",
    [
        {"expect_checksum": "sha256:other"},
        {"expect_model_revision": "rev-b"},
        {"expect_d_model": 999},
        {"layers": (2, 41)},
        {"expect_model_repo_id": "someone-else"},
    ],
)
def test_validate_lens_refuses_every_kind_of_mismatch(overrides):
    kwargs = dict(
        lens_path="mock",
        lens_checksum="sha256:lens",
        layers=K.MOCK_LAYERS,
        model_repo_id="mock",
        model_revision="rev-a",
        expect_model_repo_id="mock",
        expect_model_revision="rev-a",
        expect_d_model=K.MOCK_D_MODEL,
        expect_checksum="sha256:lens",
    )
    kwargs.update(overrides)
    with pytest.raises(LensIncompatibleError):
        validate_lens(K.mock_lens(), **kwargs)


# ---------------------------------------------------------------- pursuit


def test_pursuit_recovers_a_planted_atom(dictionary):
    atom = dictionary.atoms[K._ANSWER_BASE].float()
    code = jspace_code(
        atom * 2.0,
        dictionary,
        PursuitSettings(k=4, correlation_chunk_size=None),
        activation_checksum="sha256:x",
        lens_checksum="sha256:lens",
    )
    assert code["support"][0] == K._ANSWER_BASE
    assert code["explained_fraction"] > 0.99
    assert all(coefficient >= 0 for coefficient in code["coefficients"])
    assert code["convergence_status"] in ("max_atoms", "residual_tol")


def test_codes_are_nonnegative_and_reconstruct_real_activations(backend, question, dictionary):
    inputs = text_inputs(backend, question, "cat")
    activation = capture_final_prompt_activations(backend, inputs, [LAYER])[LAYER]
    code = jspace_code(
        activation,
        dictionary,
        PursuitSettings(k=8, correlation_chunk_size=None),
        activation_checksum="sha256:x",
        lens_checksum="sha256:lens",
    )
    assert all(value > 0 for value in code_map(code).values())
    assert code["explained_fraction"] > 0.5
    assert code["reconstruction_error"] < code["target_norm"]


def test_capture_refuses_when_the_position_cannot_be_resolved(backend, question):
    inputs = text_inputs(backend, question, "cat")
    inputs.prompt_len += 3
    with pytest.raises(RuntimeError, match="cannot be resolved"):
        capture_final_prompt_activations(backend, inputs, [LAYER])


# -------------------------------------------------------------- directions


def _codes_for(backend, dictionary, concept, modality, n=3):
    codes = []
    question = build_question(CONCEPTS)
    for index in range(n):
        if modality == "text":
            caption = (
                f"a photo of a {concept} number {index}"
                if concept
                else f"a photo of a table number {index}"
            )
            inputs = backend.build_inputs(
                prompt=build_prompt(question, modality="text", caption=caption),
                modality="text",
            )
        else:
            evidence = backend.world.evidence(
                concepts_present=[concept] if concept else [],
                modality=modality,
                nuisance_key=f"{concept}|{modality}|{index}",
            )
            inputs = backend.build_inputs(
                prompt=build_prompt(question, modality=modality),
                modality=modality,
                image=evidence,
                audio=evidence,
            )
        activation = capture_final_prompt_activations(backend, inputs, [LAYER])[LAYER]
        codes.append(
            (
                activation,
                jspace_code(
                    activation,
                    dictionary,
                    PursuitSettings(k=8, correlation_chunk_size=None),
                    activation_checksum="sha256:x",
                    lens_checksum="sha256:lens",
                ),
            )
        )
    return codes


def test_direction_is_estimated_from_source_examples_only(backend, dictionary):
    positives = _codes_for(backend, dictionary, "dog", "text")
    negatives = _codes_for(backend, dictionary, None, "text")
    record = estimate_concept_direction(
        [code_map(code) for _, code in positives],
        [code_map(code) for _, code in negatives],
        dictionary,
        concept="dog",
        source_modality="text",
        layer=LAYER,
        positive_ids=["p0", "p1", "p2"],
        negative_ids=["n0", "n1", "n2"],
        lens_checksum="sha256:lens",
        top_k=4,
    )
    assert record["uses_target_modality_data"] is False
    assert record["source_modality"] == "text"
    assert all(value > 0 for value in record["coefficients"])
    assert len(record["support"]) <= 4
    direction = unit_direction_tensor(record)
    assert float(direction.norm()) == pytest.approx(1.0, abs=1e-5)
    # The known planted direction is the concept vector.
    concept_vector = backend.world.concept_vector("dog")
    assert float(direction.dot(concept_vector)) > 0.7


def test_direction_estimation_refuses_an_empty_rectified_difference(dictionary):
    with pytest.raises(ValueError, match="empty"):
        estimate_concept_direction(
            [{0: 1.0}],
            [{0: 2.0}],
            dictionary,
            concept="dog",
            source_modality="text",
            layer=LAYER,
            positive_ids=[],
            negative_ids=[],
            lens_checksum="sha256:lens",
        )


def test_raw_residual_direction_is_a_unit_vector(backend, dictionary):
    positives = _codes_for(backend, dictionary, "dog", "text")
    negatives = _codes_for(backend, dictionary, None, "text")
    record = raw_residual_direction(
        [activation.tolist() for activation, _ in positives],
        [activation.tolist() for activation, _ in negatives],
        concept="dog",
        source_modality="text",
        layer=LAYER,
    )
    assert float(unit_direction_tensor(record).norm()) == pytest.approx(1.0, abs=1e-5)


# ----------------------------------------------------------- interventions


def _transfer_setup(backend, dictionary, concept="dog"):
    """A text-derived direction and an image-condition held-out target."""
    positives = _codes_for(backend, dictionary, concept, "text")
    negatives = _codes_for(backend, dictionary, None, "text")
    direction = estimate_concept_direction(
        [code_map(code) for _, code in positives],
        [code_map(code) for _, code in negatives],
        dictionary,
        concept=concept,
        source_modality="text",
        layer=LAYER,
        positive_ids=[],
        negative_ids=[],
        lens_checksum="sha256:lens",
        top_k=4,
    )
    evidence = backend.world.evidence(
        concepts_present=[concept], modality="image", nuisance_key="heldout-target"
    )
    inputs = backend.build_inputs(
        prompt=build_prompt(build_question(CONCEPTS), modality="image"),
        modality="image",
        image=evidence,
    )
    candidates = candidate_token_ids(backend, CONCEPTS)
    clean = score_candidate_sequences(backend, inputs, candidates)
    reference = float(
        capture_final_prompt_activations(backend, inputs, [LAYER])[LAYER].norm()
    )
    return direction, inputs, candidates, clean, reference


def test_subtracting_a_text_direction_lowers_the_concept_in_an_image_example(
    backend, dictionary
):
    direction, inputs, candidates, clean, reference = _transfer_setup(backend, dictionary)
    record = run_condition(
        backend,
        inputs,
        layer=LAYER,
        unit_direction=unit_direction_tensor(direction),
        alpha=1.0,
        reference_norm=reference,
        sign=-1,
        candidate_ids=candidates,
        target_concept="dog",
        clean_scores=clean,
    )
    assert record["target_score_change"] < 0
    assert record["signed_target_effect"] > 0
    assert record["signed_margin_effect"] > 0


def test_adding_a_text_direction_raises_the_concept_in_a_negative_image_example(
    backend, dictionary
):
    direction, _, candidates, _, _ = _transfer_setup(backend, dictionary)
    evidence = backend.world.evidence(
        concepts_present=[], modality="image", nuisance_key="heldout-negative"
    )
    inputs = backend.build_inputs(
        prompt=build_prompt(build_question(CONCEPTS), modality="image"),
        modality="image",
        image=evidence,
    )
    clean = score_candidate_sequences(backend, inputs, candidates)
    reference = float(
        capture_final_prompt_activations(backend, inputs, [LAYER])[LAYER].norm()
    )
    record = run_condition(
        backend,
        inputs,
        layer=LAYER,
        unit_direction=unit_direction_tensor(direction),
        alpha=1.0,
        reference_norm=reference,
        sign=+1,
        candidate_ids=candidates,
        target_concept="dog",
        clean_scores=clean,
    )
    assert record["target_score_change"] > 0
    assert record["signed_target_effect"] > 0


def test_zero_alpha_is_an_exact_no_op(backend, dictionary):
    direction, inputs, candidates, clean, reference = _transfer_setup(backend, dictionary)
    record = run_condition(
        backend,
        inputs,
        layer=LAYER,
        unit_direction=unit_direction_tensor(direction),
        alpha=0.0,
        reference_norm=reference,
        sign=-1,
        candidate_ids=candidates,
        target_concept="dog",
        clean_scores=clean,
    )
    assert record["target_score_change"] == pytest.approx(0.0, abs=1e-6)
    assert record["max_abs_unrelated_change"] == pytest.approx(0.0, abs=1e-6)
    assert record["activation_norm_ratio"] == pytest.approx(1.0, abs=1e-6)
    assert not record["prediction_changed"]


def test_effect_grows_with_alpha(backend, dictionary):
    direction, inputs, candidates, clean, reference = _transfer_setup(backend, dictionary)
    effects = [
        run_condition(
            backend,
            inputs,
            layer=LAYER,
            unit_direction=unit_direction_tensor(direction),
            alpha=alpha,
            reference_norm=reference,
            sign=-1,
            candidate_ids=candidates,
            target_concept="dog",
            clean_scores=clean,
        )["signed_target_effect"]
        for alpha in (0.25, 0.5, 1.0)
    ]
    assert effects == sorted(effects)


def test_random_and_unrelated_controls_are_weaker_than_the_concept_direction(
    backend, dictionary
):
    direction, inputs, candidates, clean, reference = _transfer_setup(backend, dictionary)
    kwargs = dict(
        layer=LAYER,
        alpha=1.0,
        reference_norm=reference,
        sign=-1,
        candidate_ids=candidates,
        target_concept="dog",
        clean_scores=clean,
    )
    real = run_condition(
        backend, inputs, unit_direction=unit_direction_tensor(direction), **kwargs
    )["signed_target_effect"]
    unrelated = estimate_concept_direction(
        [code_map(code) for _, code in _codes_for(backend, dictionary, "bus", "text")],
        [code_map(code) for _, code in _codes_for(backend, dictionary, None, "text")],
        dictionary,
        concept="bus",
        source_modality="text",
        layer=LAYER,
        positive_ids=[],
        negative_ids=[],
        lens_checksum="sha256:lens",
        top_k=4,
    )
    unrelated_effect = run_condition(
        backend, inputs, unit_direction=unit_direction_tensor(unrelated), **kwargs
    )["signed_target_effect"]
    randoms = [
        run_condition(
            backend,
            inputs,
            unit_direction=unit_direction_tensor(
                random_control_direction(backend.d_model, seed=seed, matched_to="t")
            ),
            **kwargs,
        )["signed_target_effect"]
        for seed in range(6)
    ]
    assert real > 1.5 * max(unrelated_effect, 0.0)
    assert real > 1.5 * max(max(randoms), 0.0)
    assert abs(sum(randoms) / len(randoms)) < real / 2


def test_random_control_direction_is_unit_norm_and_seed_reproducible(backend):
    first = random_control_direction(backend.d_model, seed=5, matched_to="x")
    second = random_control_direction(backend.d_model, seed=5, matched_to="x")
    assert first["unit_direction"] == second["unit_direction"]
    assert math.isclose(
        float(unit_direction_tensor(first).norm()), 1.0, abs_tol=1e-5
    )
