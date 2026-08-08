# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""The open-prompt protocol, its leakage audit, and the legacy bytes it protects.

Two halves. The first pins what must **not** change: the completed
candidate-listed question, its protocol string, and a legacy run fingerprint's
digest. The second exercises the new protocols — that the open prompt names no
candidate, that the scorer still sees complete candidate sequences, that the
transcript never crosses into the backend, and that every refusal fires.
"""

from __future__ import annotations

import pytest
import torch

from jlens.mmpilot.capability import (
    DEFAULT_QUESTION,
    PROMPT_PROTOCOL_VERSION,
    build_ordered_questions,
    build_prompt,
    build_question,
    prediction_and_margin,
    score_candidate_sequences,
)
from jlens.mmpilot.coordinate_swap import (
    PRIMARY_PROMPT_PROTOCOLS,
    CoordinateSwapError,
    assert_open_prompt_protocol,
    build_spec,
    coordinate_swap_band,
    coordinate_swap_fingerprint,
)
from jlens.mmpilot.coordinate_swap_mock import (
    MOCK_MODEL_REVISION,
    MOCK_PROCESSOR_REVISION,
    PRIMARY_BAND,
    SwapMockBackend,
    mock_bases,
    mock_concept_tokens,
    mock_lens_checksums,
)
from jlens.mmpilot.prompt_protocol import (
    ANIMAL_CONCEPT_SELECTION_VERSION,
    ANIMAL_LEG_COUNTS,
    ANIMAL_LEGS_QUESTION,
    AUDIT_VERSION,
    CANDIDATE_LISTED_IDENTIFICATION,
    COCO_ANIMAL_CATEGORIES,
    CONCEPT_DOMAINS,
    DOMAIN_ANIMAL,
    DOMAIN_ENTITY,
    HIDDEN_ANIMAL_LEGS,
    LEGACY_CAPABILITY_PROMPT_PROTOCOL,
    OPEN_ANIMAL_IDENTIFICATION,
    OPEN_ANIMAL_IDENTIFICATION_QUESTION,
    OPEN_ANIMAL_LEGS,
    OPEN_ENTITY_IDENTIFICATION,
    OPEN_ENTITY_IDENTIFICATION_QUESTION,
    PROPERTY_LEG_COUNT,
    RETIRED_PROTOCOLS,
    BuiltPrompt,
    ConceptSpec,
    Evidence,
    PromptLeakageError,
    PromptProtocolError,
    PropertyAnswerError,
    TaskDomainError,
    assert_task_domain,
    audit_prompt_leakage,
    backend_input_kwargs,
    build_backend_inputs,
    build_protocol_prompt,
    claim_admissibility_rule_record,
    concept_spec,
    domain_registry_checksum,
    normalize,
    prompt_protocol_fingerprint,
    property_registry_checksum,
    protocol_claim_admissibility,
    resolve_leg_count,
    select_animal_concepts,
)
from jlens.mmpilot.store import IncompatibleStateError, RunFingerprint, UnitStore

LEGACY_CONCEPTS = ["bird", "cat", "giraffe", "microwave", "toilet", "zebra"]

BIRD = concept_spec("bird")
CAT = concept_spec("cat")


# ------------------------------------------------- 1. the legacy bytes, pinned


def test_the_legacy_question_is_byte_for_byte_what_it_always_was():
    """The completed study's question, spelled out. If this test fails, a
    completed run's prompt hash moved and its resume is broken."""
    assert DEFAULT_QUESTION == (
        "Question: which one of these is present: {options}? "
        "Answer with exactly one word.\nAnswer:"
    )
    assert build_question(LEGACY_CONCEPTS) == (
        "Question: which one of these is present: bird, cat, giraffe, microwave, "
        "toilet, zebra? Answer with exactly one word.\nAnswer:"
    )
    assert build_prompt(build_question(LEGACY_CONCEPTS), modality="text", caption="a bird") == (
        "Caption: a bird\n"
        "Question: which one of these is present: bird, cat, giraffe, microwave, "
        "toilet, zebra? Answer with exactly one word.\nAnswer:"
    )
    # Image and spoken audio carry the question alone — the media is the evidence.
    for modality in ("image", "spoken_audio"):
        assert build_prompt(
            build_question(LEGACY_CONCEPTS), modality=modality
        ) == build_question(LEGACY_CONCEPTS)


def test_the_legacy_protocol_string_and_ordering_rule_are_unchanged():
    assert PROMPT_PROTOCOL_VERSION == "gemma-it-chat-balanced-options-v1"
    assert LEGACY_CAPABILITY_PROMPT_PROTOCOL == PROMPT_PROTOCOL_VERSION
    ordered = build_ordered_questions(LEGACY_CONCEPTS)
    assert len(ordered) == 2
    assert ordered[0] == build_question(LEGACY_CONCEPTS)
    assert "zebra, toilet, microwave, giraffe, cat, bird" in ordered[1]


def test_a_legacy_run_fingerprint_keeps_its_digest():
    """A completed run's digest, pinned. New protocol fields live in the
    intervention/selection configs of *new* runs and cannot reach this one."""
    fingerprint = RunFingerprint(
        mode="pilot",
        model_repo_id="google/gemma-4-e4b-it",
        model_revision="rev-a",
        processor_revision="rev-b",
        layers=(35, 38, 40),
        lens_checksum="sha256:lens",
        manifest_checksum="sha256:manifest",
        split_id="split-1",
        selection_config={"capability_protocol": PROMPT_PROTOCOL_VERSION},
    )
    assert fingerprint.digest == (
        "sha256:90528c763d6a155f5b9799e0592341670c1b4a331063655f28aca21fc620d922"
    )


def test_the_protocol_module_can_rebuild_the_legacy_prompt_exactly():
    built = build_protocol_prompt(
        protocol=CANDIDATE_LISTED_IDENTIFICATION,
        evidence=Evidence(modality="text", text="a bird"),
        external_candidates=LEGACY_CONCEPTS,
        legacy_candidate_list=LEGACY_CONCEPTS,
    )
    assert built.model_visible_prompt == build_prompt(
        build_question(LEGACY_CONCEPTS), modality="text", caption="a bird"
    )
    assert built.candidate_visibility["candidates_in_prompt"] is True
    # Legacy leakage is *recorded*, never pretended away.
    assert built.leakage["passed"] is True
    assert "instruction_candidate_leakage" in built.leakage["recorded"]
    assert "candidate_enumeration_detected" in built.leakage["recorded"]


def test_the_legacy_protocol_supports_only_a_candidate_conditioned_claim():
    decision = protocol_claim_admissibility(
        protocol=CANDIDATE_LISTED_IDENTIFICATION,
        leakage=audit_prompt_leakage(
            protocol=CANDIDATE_LISTED_IDENTIFICATION,
            modality="text",
            instruction=build_question(LEGACY_CONCEPTS),
            visible_evidence_text="a bird",
            external_candidates=LEGACY_CONCEPTS,
        ),
        mode="real",
    )
    assert decision["admissible"] is True
    assert decision["granted_claim"] == "candidate_conditioned_identification"
    assert "open cross-modal identification" in decision["excluded_claims"]


# ------------------------------------------------ 2. the open prompt, built


def test_the_open_identification_prompt_contains_no_candidate_name():
    for modality, evidence in (
        ("text", Evidence(modality="text", text="A small bird perched on a branch.")),
        ("image", Evidence(modality="image", media="<pixels>")),
        (
            "spoken_audio",
            Evidence(
                modality="spoken_audio",
                media=[0.0, 0.1],
                transcript="a small bird perched on a branch",
                sampling_rate=16000,
            ),
        ),
    ):
        built = build_protocol_prompt(
            protocol=OPEN_ANIMAL_IDENTIFICATION,
            evidence=evidence,
            external_candidates=("bird", "cat", "giraffe", "zebra"),
            source=BIRD,
            target=CAT,
        )
        assert built.question == OPEN_ANIMAL_IDENTIFICATION_QUESTION
        assert built.modality == modality
        for candidate in ("cat", "giraffe", "zebra"):
            assert candidate not in normalize(built.model_visible_prompt).split()
        assert "," not in built.question
        assert built.candidates_are_external is True
        if modality != "text":
            assert built.model_visible_prompt == OPEN_ANIMAL_IDENTIFICATION_QUESTION


def test_text_evidence_may_carry_the_source_and_the_fact_is_recorded():
    built = build_protocol_prompt(
        protocol=OPEN_ANIMAL_IDENTIFICATION,
        evidence=Evidence(modality="text", text="A small bird perched on a branch."),
        external_candidates=("bird", "cat"),
        source=BIRD,
        target=CAT,
    )
    assert built.leakage["passed"] is True
    assert built.leakage["source_in_visible_evidence"] is True
    finding = built.leakage["findings"]["source_in_visible_evidence"]
    assert finding["status"] == "recorded"
    assert finding["matches"] == [{"surface": "bird", "scope": "visible_evidence"}]


def test_image_evidence_carries_neither_source_nor_target_text():
    built = build_protocol_prompt(
        protocol=OPEN_ANIMAL_IDENTIFICATION,
        evidence=Evidence(
            modality="image", media="<pixels>", media_reference="/coco/bird_000001.jpg"
        ),
        external_candidates=("bird", "cat"),
        source=BIRD,
        target=CAT,
    )
    assert built.leakage["source_in_visible_evidence"] is False
    assert built.leakage["findings"]["target_in_visible_evidence"]["status"] == "clean"
    assert built.leakage["findings"]["semantic_filename_exposure"]["status"] == "clean"


def test_the_target_may_never_appear_in_model_visible_text():
    with pytest.raises(PromptLeakageError, match="target_in_visible_evidence"):
        build_protocol_prompt(
            protocol=OPEN_ANIMAL_IDENTIFICATION,
            evidence=Evidence(modality="text", text="A bird sitting beside a cat."),
            external_candidates=("bird", "cat"),
            source=BIRD,
            target=CAT,
        )


def test_the_target_may_never_appear_in_the_offline_audio_transcript():
    with pytest.raises(PromptLeakageError, match="target_in_audio_transcript"):
        build_protocol_prompt(
            protocol=OPEN_ANIMAL_IDENTIFICATION,
            evidence=Evidence(
                modality="spoken_audio",
                media=[0.0],
                transcript="a bird sitting beside a cat",
                sampling_rate=16000,
            ),
            external_candidates=("bird", "cat"),
            source=BIRD,
            target=CAT,
        )


def test_a_missing_transcript_is_unauditable_not_clean():
    record = audit_prompt_leakage(
        protocol=OPEN_ANIMAL_IDENTIFICATION,
        modality="spoken_audio",
        instruction=OPEN_ANIMAL_IDENTIFICATION_QUESTION,
        transcript=None,
        source=BIRD,
        target=CAT,
        external_candidates=("bird", "cat"),
    )
    assert record["passed"] is False
    assert record["unauditable"] == ["target_in_audio_transcript"]
    with pytest.raises(PromptLeakageError, match="unchecked transcript is not a clean one"):
        build_protocol_prompt(
            protocol=OPEN_ANIMAL_IDENTIFICATION,
            evidence=Evidence(modality="spoken_audio", media=[0.0], sampling_rate=16000),
            external_candidates=("bird", "cat"),
            source=BIRD,
            target=CAT,
        )


def test_an_open_question_that_names_a_candidate_is_refused_before_any_audit():
    with pytest.raises(PromptProtocolError, match="names the external candidate"):
        build_protocol_prompt(
            protocol=OPEN_ANIMAL_IDENTIFICATION,
            evidence=Evidence(modality="image", media="<pixels>"),
            external_candidates=("bird", "cat"),
            question="Is this a bird? Answer:",
            source=BIRD,
            target=CAT,
        )


def test_a_rendered_option_list_is_detected_even_with_unregistered_items():
    record = audit_prompt_leakage(
        protocol=OPEN_ANIMAL_IDENTIFICATION,
        modality="image",
        instruction="Which one of these is present: aardvark, wombat, quokka?",
        external_candidates=("bird", "cat"),
        source=BIRD,
        target=CAT,
    )
    assert record["findings"]["candidate_enumeration_detected"]["status"] == "violation"
    assert record["passed"] is False


def test_a_semantic_filename_in_the_prompt_is_refused_under_every_protocol():
    for protocol, extra in (
        (CANDIDATE_LISTED_IDENTIFICATION, {}),
        (OPEN_ANIMAL_IDENTIFICATION, {}),
    ):
        record = audit_prompt_leakage(
            protocol=protocol,
            modality="image",
            instruction="File: bird_000001.jpg\nWhat is present?",
            external_candidates=("bird", "cat"),
            media_reference="/coco/train/bird_000001.jpg",
            **extra,
        )
        assert record["findings"]["semantic_filename_exposure"]["status"] == "violation"
        assert record["passed"] is False


# --------------------------------------------- 3. downstream property answers


def _property_prompt(**overrides):
    kwargs = {
        "protocol": OPEN_ANIMAL_LEGS,
        "evidence": Evidence(modality="image", media="<pixels>"),
        "external_candidates": ("two", "four"),
        "source": BIRD,
        "target": CAT,
        "property_answers": {"source": ("two", "2"), "target": ("four", "4")},
    }
    kwargs.update(overrides)
    return build_protocol_prompt(**kwargs)


def test_the_downstream_property_question_names_no_entity_and_no_number():
    built = _property_prompt()
    assert built.question == ANIMAL_LEGS_QUESTION
    assert built.model_visible_prompt == ANIMAL_LEGS_QUESTION
    tokens = normalize(built.model_visible_prompt).split()
    for forbidden in ("bird", "cat", "two", "four", "2", "4"):
        assert forbidden not in tokens


def test_the_target_property_answer_may_not_appear_in_the_visible_prompt():
    with pytest.raises(PromptLeakageError, match="property_answer_in_prompt"):
        _property_prompt(
            evidence=Evidence(modality="text", text="An animal with four legs."),
        )


def test_the_source_property_answer_is_permitted_and_recorded():
    built = _property_prompt(
        evidence=Evidence(modality="text", text="An animal standing on two legs."),
    )
    assert built.leakage["passed"] is True
    assert built.leakage["source_property_answer_present"] is True
    assert built.leakage["findings"]["property_answer_in_prompt"]["recorded_matches"] == [
        {"surface": "two", "scope": "visible_evidence"}
    ]


def test_property_answers_are_derived_from_the_registry_not_guessed():
    """Omitting them is fine; the registry supplies them. Passing an empty
    mapping is not, because that would make the audit pass by omission."""
    built = build_protocol_prompt(
        protocol=OPEN_ANIMAL_LEGS,
        evidence=Evidence(modality="image", media="<pixels>"),
        external_candidates=("two", "four"),
        source=BIRD,
        target=CAT,
    )
    assert built.property_answers == {"source": ["two", "2"], "target": ["four", "4"]}
    assert built.property_schema == PROPERTY_LEG_COUNT
    assert built.property_registry_checksum == property_registry_checksum()
    with pytest.raises(PromptProtocolError, match="empty property_answers"):
        build_protocol_prompt(
            protocol=OPEN_ANIMAL_LEGS,
            evidence=Evidence(modality="image", media="<pixels>"),
            external_candidates=("two", "four"),
            source=BIRD,
            target=CAT,
            property_answers={},
        )


# ------------------------------------------------- 4. the hidden intermediate


def _hidden(**overrides):
    kwargs = {
        "protocol": HIDDEN_ANIMAL_LEGS,
        "evidence": Evidence(
            modality="text", text="The animal in the recording is the one that spins webs."
        ),
        "external_candidates": ("six", "eight"),
        "source": concept_spec("spider"),
        "target": concept_spec("ant"),
    }
    kwargs.update(overrides)
    return build_protocol_prompt(**kwargs)


def test_hidden_intermediate_accepts_a_description_that_names_neither_entity():
    built = _hidden()
    assert built.leakage["passed"] is True
    assert built.protocol_version == HIDDEN_ANIMAL_LEGS


def test_hidden_intermediate_refuses_the_source_label_in_visible_text():
    with pytest.raises(PromptLeakageError, match="source_in_visible_evidence"):
        _hidden(evidence=Evidence(modality="text", text="A spider on a web."))


def test_hidden_intermediate_refuses_the_target_label_in_visible_text():
    with pytest.raises(PromptLeakageError, match="target_in_visible_evidence"):
        _hidden(evidence=Evidence(modality="text", text="Not an ant, the web spinner."))


def test_hidden_intermediate_refuses_either_label_in_the_offline_transcript():
    for text, category in (
        ("the animal that spins webs is a spider", "source_in_audio_transcript"),
        ("it is not an ant at all", "target_in_audio_transcript"),
    ):
        with pytest.raises(PromptLeakageError, match=category):
            _hidden(
                evidence=Evidence(
                    modality="spoken_audio",
                    media=[0.0],
                    transcript=text,
                    sampling_rate=16000,
                )
            )


def test_hidden_intermediate_refuses_a_property_answer_that_reveals_the_entity():
    with pytest.raises(PromptLeakageError, match="property_answer_in_prompt"):
        _hidden(
            evidence=Evidence(modality="text", text="The eight-legged web spinner."),
        )


def test_an_open_identification_prompt_is_not_hidden_intermediate():
    """Same evidence, two protocols: permitted-and-recorded under one, refused
    under the other. The distinction is the whole point of having both."""
    evidence = Evidence(modality="text", text="A small bird on a branch.")
    permitted = build_protocol_prompt(
        protocol=OPEN_ANIMAL_IDENTIFICATION,
        evidence=evidence,
        external_candidates=("bird", "cat"),
        source=BIRD,
        target=CAT,
    )
    assert permitted.leakage["passed"] is True
    with pytest.raises(PromptLeakageError, match="source_in_visible_evidence"):
        build_protocol_prompt(
            protocol=HIDDEN_ANIMAL_LEGS,
            evidence=evidence,
            external_candidates=("two", "four"),
            source=BIRD,
            target=CAT,
        )


# ---------------------------------------------- 5. normalization and aliases


@pytest.mark.parametrize(
    "evidence_text",
    [
        "A BIRD and a CAT.",                      # case folding
        "A bird and a ｃａｔ.",       # fullwidth, folded by NFKC
        "A bird and a cát.",                # combining accent, stripped
        "A bird, and a (cat)!",                   # punctuation
        "A bird and  two   cats.",                # registered plural, extra space
    ],
)
def test_the_audit_normalizes_unicode_case_punctuation_and_uses_aliases(evidence_text):
    with pytest.raises(PromptLeakageError, match="target_in_visible_evidence"):
        build_protocol_prompt(
            protocol=OPEN_ANIMAL_IDENTIFICATION,
            evidence=Evidence(modality="text", text=evidence_text),
            external_candidates=("bird", "cat"),
            source=BIRD,
            target=CAT,
        )


def test_matching_is_whole_token_so_substrings_do_not_trigger():
    built = build_protocol_prompt(
        protocol=OPEN_ANIMAL_IDENTIFICATION,
        evidence=Evidence(
            modality="text", text="A bird beside a concatenation of catamarans."
        ),
        external_candidates=("bird", "cat"),
        source=BIRD,
        target=ConceptSpec("cat", (), DOMAIN_ANIMAL),  # no plural alias registered
    )
    assert built.leakage["passed"] is True


def test_an_unregistered_alias_is_a_stated_limit_not_a_silent_pass():
    clean = build_protocol_prompt(
        protocol=OPEN_ANIMAL_IDENTIFICATION,
        evidence=Evidence(modality="text", text="A bird beside a feline."),
        external_candidates=("bird", "cat"),
        source=BIRD,
        target=CAT,
    )
    assert clean.leakage["passed"] is True
    assert any("not a registered alias" in limit for limit in clean.leakage["limits"])
    # Register it and the same evidence is refused.
    with pytest.raises(PromptLeakageError, match="target_in_visible_evidence"):
        build_protocol_prompt(
            protocol=OPEN_ANIMAL_IDENTIFICATION,
            evidence=Evidence(modality="text", text="A bird beside a feline."),
            external_candidates=("bird", "cat"),
            source=BIRD,
            target=ConceptSpec("cat", ("cats", "feline"), DOMAIN_ANIMAL),
        )


# --------------------------------------------- 6. the external scoring edge


@pytest.fixture(scope="module")
def backend():
    return SwapMockBackend()


@pytest.fixture(scope="module")
def open_prompt(backend):
    return build_protocol_prompt(
        protocol=OPEN_ANIMAL_IDENTIFICATION,
        evidence=Evidence(modality="image", media="<pixels>"),
        external_candidates=("bird", "cat"),
        source=BIRD,
        target=CAT,
        encode_candidate=backend.encode_candidate,
        encode_prompt=backend.encode_token,
    )


def test_candidate_sequences_are_complete_and_multi_token(open_prompt):
    ids = open_prompt.external_candidate_token_ids
    assert set(ids) == {"bird", "cat"}
    # The MOCK appends an answer suffix on purpose, so every candidate is a
    # genuine multi-token sequence and prefix-scoring cannot masquerade as this.
    assert all(len(sequence) > 1 for sequence in ids.values())
    assert len({tuple(sequence) for sequence in ids.values()}) == 2


def test_the_transcript_never_reaches_the_backend(backend):
    transcript = "a small bird perched on a branch"
    built = build_protocol_prompt(
        protocol=OPEN_ANIMAL_IDENTIFICATION,
        evidence=Evidence(
            modality="spoken_audio",
            media=[0.0, 0.25],
            transcript=transcript,
            sampling_rate=16000,
            media_reference="/spokencoco/wav/000001.wav",
        ),
        external_candidates=("bird", "cat"),
        source=BIRD,
        target=CAT,
    )
    kwargs = backend_input_kwargs(built, transcript=transcript)
    assert set(kwargs) == {"prompt", "modality", "audio", "sampling_rate", "media_path"}
    assert kwargs["prompt"] == OPEN_ANIMAL_IDENTIFICATION_QUESTION
    for value in kwargs.values():
        if isinstance(value, str):
            assert transcript not in value
    assert built.transcript_hash is not None
    assert transcript not in str(built.to_dict())

    seen: list[dict] = []

    class _Spy:
        def build_inputs(self, **call_kwargs):
            seen.append(call_kwargs)
            return call_kwargs

    build_backend_inputs(_Spy(), built, transcript=transcript)
    assert transcript not in str(seen)


def test_a_transcript_smuggled_into_the_prompt_is_refused(backend):
    transcript = "a small animal perched on a branch"
    built = build_protocol_prompt(
        protocol=OPEN_ANIMAL_IDENTIFICATION,
        evidence=Evidence(modality="image", media="<pixels>"),
        external_candidates=("bird", "cat"),
        question=f"{transcript}\nWhat is present? Answer:",
        source=BIRD,
        target=CAT,
    )
    with pytest.raises(PromptProtocolError, match="reached the backend arguments"):
        backend_input_kwargs(built, transcript=transcript)


def test_the_scorer_sees_complete_sequences_and_is_order_invariant(backend, open_prompt):
    inputs = backend.build_inputs(
        prompt=open_prompt.model_visible_prompt, modality="image", concept="bird"
    )
    forward = dict(open_prompt.external_candidate_token_ids)
    reverse = dict(reversed(list(forward.items())))
    scored = score_candidate_sequences(backend, inputs, forward)
    rescored = score_candidate_sequences(backend, inputs, reverse)
    for name, row in scored.items():
        assert row["n_tokens"] == len(forward[name]) > 1
        assert row["sum_logprob"] == pytest.approx(rescored[name]["sum_logprob"], abs=1e-9)
    assert prediction_and_margin(scored, "bird")["prediction"] == "bird"


def test_the_prompt_hash_does_not_depend_on_the_candidate_order(backend):
    def _built(candidates):
        return build_protocol_prompt(
            protocol=OPEN_ANIMAL_IDENTIFICATION,
            evidence=Evidence(modality="image", media="<pixels>"),
            external_candidates=candidates,
            source=BIRD,
            target=CAT,
            encode_candidate=backend.encode_candidate,
        )

    forward = _built(("bird", "cat", "giraffe"))
    reversed_order = _built(("giraffe", "cat", "bird"))
    assert forward.prompt_hash == reversed_order.prompt_hash
    assert forward.question_hash == reversed_order.question_hash
    assert forward.external_candidates != reversed_order.external_candidates

    def _digest(built):
        return prompt_protocol_fingerprint(
            built, model_revision="rev-a", processor_revision="rev-b"
        )["prompt_protocol_digest"]

    # Order does not change the fingerprint either — the candidate set does.
    assert _digest(forward) == _digest(reversed_order)


def test_the_fingerprint_changes_when_the_candidate_set_changes(backend):
    def _digest(candidates):
        built = build_protocol_prompt(
            protocol=OPEN_ANIMAL_IDENTIFICATION,
            evidence=Evidence(modality="image", media="<pixels>"),
            external_candidates=candidates,
            source=BIRD,
            target=CAT,
            encode_candidate=backend.encode_candidate,
        )
        return prompt_protocol_fingerprint(
            built, model_revision="rev-a", processor_revision="rev-b"
        )

    two = _digest(("bird", "cat"))
    three = _digest(("bird", "cat", "dog"))
    assert two["prompt_protocol_digest"] != three["prompt_protocol_digest"]
    assert two["prompt_hash"] == three["prompt_hash"]
    assert two["external_candidate_token_ids"] != three["external_candidate_token_ids"]


def test_the_fingerprint_carries_every_required_field(backend, open_prompt):
    payload = prompt_protocol_fingerprint(
        open_prompt, model_revision="rev-a", processor_revision="rev-b"
    )
    for key in (
        "prompt_protocol_version",
        "question_template",
        "question_hash",
        "candidate_visibility_rule",
        "leakage_audit_version",
        "source_concept",
        "target_concept",
        "registered_aliases_checksum",
        "external_candidates",
        "external_candidate_token_ids",
        "candidate_scoring_version",
        "prompt_boundary_rule",
        "modality",
        "model_revision",
        "processor_revision",
        "audio_protocol_fingerprint",
        "prompt_protocol_digest",
    ):
        assert key in payload, key
    assert payload["leakage_audit_version"] == AUDIT_VERSION
    assert payload["prompt_protocol_version"] == OPEN_ANIMAL_IDENTIFICATION


def test_the_audio_protocol_fingerprint_is_bound_only_for_spoken_audio():
    built = build_protocol_prompt(
        protocol=OPEN_ANIMAL_IDENTIFICATION,
        evidence=Evidence(
            modality="spoken_audio",
            media=[0.0],
            transcript="a small bird on a branch",
            sampling_rate=16000,
        ),
        external_candidates=("bird", "cat"),
        source=BIRD,
        target=CAT,
    )
    payload = prompt_protocol_fingerprint(
        built,
        model_revision="rev-a",
        processor_revision="rev-b",
        audio_protocol_fingerprint="sha256:audio",
    )
    assert payload["audio_protocol_fingerprint"] == "sha256:audio"


# --------------------------------- 7. the intervention boundary, still holding


def test_candidate_completion_positions_are_never_patched(backend, open_prompt):
    """The external candidates are appended *after* ``prompt_len``, and no
    position rule can reach them — checked against the hook's own record."""
    tokens = mock_concept_tokens(backend)
    bases = mock_bases(
        backend.world, layers=PRIMARY_BAND, source=tokens["bird"], target=tokens["cat"]
    )
    inputs = backend.build_inputs(
        prompt=open_prompt.model_visible_prompt, modality="image", concept="bird"
    )
    candidate_ids = dict(open_prompt.external_candidate_token_ids)
    with coordinate_swap_band(
        backend.blocks,
        bases,
        alpha=1.0,
        prompt_len=inputs.prompt_len,
        position_rule="all_prompt_positions",
        record_coordinates=False,
    ) as stats:
        score_candidate_sequences(backend, inputs, candidate_ids)
    for layer, row in stats.items():
        assert max(row["positions"]) == inputs.prompt_len - 1, layer
        assert row["n_candidate_positions_skipped"] == len(candidate_ids["cat"])
        assert row["seq_len"] > inputs.prompt_len


def test_the_patched_run_leaves_candidate_positions_bit_identical(backend, open_prompt):
    tokens = mock_concept_tokens(backend)
    bases = mock_bases(
        backend.world, layers=PRIMARY_BAND, source=tokens["bird"], target=tokens["cat"]
    )
    inputs = backend.build_inputs(
        prompt=open_prompt.model_visible_prompt, modality="image", concept="bird"
    )
    ids = open_prompt.external_candidate_token_ids["cat"]
    tensors = dict(inputs.tensors)
    tensors["input_ids"] = torch.cat(
        [tensors["input_ids"], torch.tensor([list(ids)], dtype=torch.long)], dim=1
    )
    tensors["attention_mask"] = torch.ones(1, tensors["input_ids"].shape[1], dtype=torch.long)

    captured: dict[str, torch.Tensor] = {}

    def _record(_module, _inputs, output):
        captured["hidden"] = (output if torch.is_tensor(output) else output[0]).clone()

    handle = backend.blocks[max(PRIMARY_BAND)].register_forward_hook(_record)
    try:
        backend.forward_logits(tensors)
        clean_tail = captured["hidden"][0, inputs.prompt_len :].clone()
        with coordinate_swap_band(
            backend.blocks,
            bases,
            alpha=1.0,
            prompt_len=inputs.prompt_len,
            position_rule="all_prompt_positions",
            record_coordinates=False,
        ):
            backend.forward_logits(tensors)
        patched_tail = captured["hidden"][0, inputs.prompt_len :]
    finally:
        handle.remove()
    assert torch.equal(clean_tail, patched_tail)


# ------------------------------------- 8. the coordinate-swap protocol gate


def _swap_spec(prompt_protocol=None):
    tokens = mock_concept_tokens(SwapMockBackend())
    return build_spec(
        source=tokens["bird"],
        target=tokens["cat"],
        layer_band=PRIMARY_BAND,
        alpha=1.0,
        position_rule="all_prompt_positions",
        control_kind="coordinate_swap",
        lens_checksums=mock_lens_checksums(PRIMARY_BAND),
        model_revision=MOCK_MODEL_REVISION,
        processor_revision=MOCK_PROCESSOR_REVISION,
        prompt_protocol=prompt_protocol,
    )


def test_the_primary_swap_study_refuses_a_candidate_listed_prompt(backend):
    legacy = build_protocol_prompt(
        protocol=CANDIDATE_LISTED_IDENTIFICATION,
        evidence=Evidence(modality="text", text="a bird"),
        external_candidates=LEGACY_CONCEPTS,
        legacy_candidate_list=LEGACY_CONCEPTS,
        encode_candidate=backend.encode_candidate,
    )
    payload = prompt_protocol_fingerprint(
        legacy, model_revision=MOCK_MODEL_REVISION, processor_revision=MOCK_PROCESSOR_REVISION
    )
    with pytest.raises(CoordinateSwapError, match="not admissible for the primary"):
        assert_open_prompt_protocol(payload)
    with pytest.raises(CoordinateSwapError, match="requires a bound prompt protocol"):
        assert_open_prompt_protocol(None)


def test_the_primary_swap_study_accepts_an_open_prompt(backend, open_prompt):
    payload = prompt_protocol_fingerprint(
        open_prompt,
        model_revision=MOCK_MODEL_REVISION,
        processor_revision=MOCK_PROCESSOR_REVISION,
    )
    assert assert_open_prompt_protocol(payload)["prompt_protocol_version"] in (
        PRIMARY_PROMPT_PROTOCOLS
    )
    spec = _swap_spec(payload)
    assert spec.prompt_protocol_version == OPEN_ANIMAL_IDENTIFICATION
    assert spec.prompt_protocol_digest == payload["prompt_protocol_digest"]
    fingerprint = coordinate_swap_fingerprint(
        spec, alphas=(0.0, 1.0), controls=("coordinate_swap", "zero"),
        prompt_protocol=payload,
    )
    assert fingerprint["prompt_protocol_version"] == OPEN_ANIMAL_IDENTIFICATION
    assert fingerprint["prompt_protocol"]["external_candidates"] == ["bird", "cat"]


def test_changing_the_candidate_set_changes_the_swap_run_fingerprint(backend):
    def _config(candidates):
        built = build_protocol_prompt(
            protocol=OPEN_ANIMAL_IDENTIFICATION,
            evidence=Evidence(modality="image", media="<pixels>"),
            external_candidates=candidates,
            source=BIRD,
            target=CAT,
            encode_candidate=backend.encode_candidate,
        )
        payload = prompt_protocol_fingerprint(
            built,
            model_revision=MOCK_MODEL_REVISION,
            processor_revision=MOCK_PROCESSOR_REVISION,
        )
        return RunFingerprint(
            mode="coordinate_swap",
            model_repo_id="mock",
            model_revision=MOCK_MODEL_REVISION,
            processor_revision=MOCK_PROCESSOR_REVISION,
            layers=tuple(PRIMARY_BAND),
            lens_checksum="sha256:mock",
            manifest_checksum="sha256:mock",
            split_id="mock",
            intervention_config=coordinate_swap_fingerprint(
                _swap_spec(payload),
                alphas=(1.0,),
                controls=("coordinate_swap",),
                prompt_protocol=payload,
            ),
        ).digest

    assert _config(("bird", "cat")) != _config(("bird", "cat", "dog"))
    assert _config(("bird", "cat")) == _config(("cat", "bird"))


def _swap_run_fingerprint(backend, candidates, protocol=OPEN_ANIMAL_IDENTIFICATION, **overrides):
    built = build_protocol_prompt(
        protocol=protocol,
        evidence=Evidence(modality="image", media="<pixels>"),
        external_candidates=candidates,
        source=BIRD,
        target=CAT,
        encode_candidate=backend.encode_candidate,
        **overrides,
    )
    payload = prompt_protocol_fingerprint(
        built,
        model_revision=MOCK_MODEL_REVISION,
        processor_revision=MOCK_PROCESSOR_REVISION,
    )
    return RunFingerprint(
        mode="coordinate_swap",
        model_repo_id="mock",
        model_revision=MOCK_MODEL_REVISION,
        processor_revision=MOCK_PROCESSOR_REVISION,
        layers=tuple(PRIMARY_BAND),
        lens_checksum="sha256:mock",
        manifest_checksum="sha256:mock",
        split_id="mock",
        intervention_config=coordinate_swap_fingerprint(
            _swap_spec(payload),
            alphas=(1.0,),
            controls=("coordinate_swap",),
            prompt_protocol=payload,
        ),
    )


def test_a_changed_prompt_protocol_refuses_an_incompatible_resume(backend, tmp_path):
    """The digest gate is what makes these fields binding rather than decorative."""
    original = _swap_run_fingerprint(backend, ("bird", "cat"))
    store = UnitStore(tmp_path / "run", original)
    assert store.open() == "starting"
    store.save("metric", "probe", {"value": 1})
    assert UnitStore(tmp_path / "run", original).open() == "resuming"

    for label, changed in (
        ("candidate set", _swap_run_fingerprint(backend, ("bird", "cat", "dog"))),
        (
            "question template",
            _swap_run_fingerprint(
                backend,
                ("bird", "cat"),
                question="Which animal does the evidence show? Answer:",
            ),
        ),
    ):
        with pytest.raises(IncompatibleStateError, match="different configuration"):
            UnitStore(tmp_path / "run", changed).open()
        assert changed.digest != original.digest, label

    # And reordering the same set still resumes: it is the same measurement.
    assert (
        UnitStore(tmp_path / "run", _swap_run_fingerprint(backend, ("cat", "bird"))).open()
        == "resuming"
    )


# ------------------------------------------------- 9. claim admissibility


def _animal_domain(protocol=OPEN_ANIMAL_IDENTIFICATION, **overrides):
    """The record assert_task_domain produces for an all-animal cast."""
    kwargs = {"concepts": (BIRD, CAT), "external_candidates": ("bird", "cat")}
    kwargs.update(overrides)
    return assert_task_domain(protocol, **kwargs)


def _open_leakage(protocol=OPEN_ANIMAL_IDENTIFICATION, **overrides):
    kwargs = {
        "protocol": protocol,
        "modality": "image",
        "instruction": OPEN_ANIMAL_IDENTIFICATION_QUESTION,
        "source": BIRD,
        "target": CAT,
        "external_candidates": ("bird", "cat"),
    }
    kwargs.update(overrides)
    return audit_prompt_leakage(**kwargs)


def test_no_mock_result_supports_any_claim():
    for protocol in (OPEN_ANIMAL_IDENTIFICATION, HIDDEN_ANIMAL_LEGS):
        decision = protocol_claim_admissibility(
            protocol=protocol,
            leakage=_open_leakage(protocol, instruction=ANIMAL_LEGS_QUESTION),
            mode="mock",
            identity_replacement_passed=True,
            direct_answer_control_passed=True,
            direct_answer_onset_control_passed=True,
        )
        assert decision["admissible"] is False
        assert decision["granted_claim"] is None
        assert any("no mock" in reason.lower() for reason in decision["reasons"])


def test_open_identification_needs_the_target_checks_to_pass():
    good = protocol_claim_admissibility(
        protocol=OPEN_ANIMAL_IDENTIFICATION,
        leakage=_open_leakage(),
        task_domain=_animal_domain(),
        mode="real"
    )
    assert good["granted_claim"] == "open_cross_modal_animal_identification"

    leaked = _open_leakage(
        modality="text", visible_evidence_text="a bird beside a cat"
    )
    bad = protocol_claim_admissibility(
        protocol=OPEN_ANIMAL_IDENTIFICATION,
        leakage=leaked,
        task_domain=_animal_domain(),
        mode="real"
    )
    assert bad["admissible"] is False
    assert any("leakage audit refused" in reason for reason in bad["reasons"])


def test_downstream_recomputation_needs_identity_replacement_and_the_controls():
    leakage = _open_leakage(OPEN_ANIMAL_LEGS, instruction=ANIMAL_LEGS_QUESTION)
    assert (
        protocol_claim_admissibility(
            protocol=OPEN_ANIMAL_LEGS,
            leakage=leakage,
            task_domain=_animal_domain(OPEN_ANIMAL_LEGS, external_candidates=()),
            mode="real",
            identity_replacement_passed=True,
            direct_answer_control_passed=True,
        )["granted_claim"]
        == "animal_leg_count_recomputation"
    )
    for kwargs in (
        {"identity_replacement_passed": False, "direct_answer_control_passed": True},
        {"identity_replacement_passed": True, "direct_answer_control_passed": False},
        {},
    ):
        decision = protocol_claim_admissibility(
            protocol=OPEN_ANIMAL_LEGS,
            leakage=leakage,
            task_domain=_animal_domain(OPEN_ANIMAL_LEGS, external_candidates=()),
            mode="real",
            **kwargs
        )
        assert decision["admissible"] is False


def test_multi_hop_reasoning_needs_both_names_absent_and_the_onset_control():
    clean = _open_leakage(HIDDEN_ANIMAL_LEGS, instruction=ANIMAL_LEGS_QUESTION)
    assert (
        protocol_claim_admissibility(
            protocol=HIDDEN_ANIMAL_LEGS,
            leakage=clean,
            task_domain=_animal_domain(HIDDEN_ANIMAL_LEGS, external_candidates=()),
            mode="real",
            direct_answer_onset_control_passed=True,
        )["granted_claim"]
        == "hidden_animal_multi_hop_reasoning"
    )
    assert (
        protocol_claim_admissibility(
            protocol=HIDDEN_ANIMAL_LEGS,
            leakage=clean,
            task_domain=_animal_domain(HIDDEN_ANIMAL_LEGS, external_candidates=()),
            mode="real",
            direct_answer_onset_control_passed=False,
        )["admissible"]
        is False
    )
    named = _open_leakage(
        HIDDEN_ANIMAL_LEGS,
        modality="text",
        instruction=ANIMAL_LEGS_QUESTION,
        visible_evidence_text="a bird",
    )
    decision = protocol_claim_admissibility(
        protocol=HIDDEN_ANIMAL_LEGS,
        leakage=named,
        task_domain=_animal_domain(HIDDEN_ANIMAL_LEGS, external_candidates=()),
        mode="real",
        direct_answer_onset_control_passed=True,
    )
    assert decision["admissible"] is False
    assert any("source_in_visible_evidence" in reason for reason in decision["reasons"])


def test_a_claim_is_never_upgraded_by_a_result():
    """There is no argument that raises a protocol above its ceiling."""
    decision = protocol_claim_admissibility(
        protocol=OPEN_ANIMAL_IDENTIFICATION,
        leakage=_open_leakage(),
        task_domain=_animal_domain(),
        mode="real"
    )
    assert decision["maximum_claim"] == "open_cross_modal_animal_identification"
    assert "multi-hop reasoning" in decision["excluded_claims"]
    assert "never raised after a result is seen" in decision["automatic_upgrade_forbidden"]


def test_the_claim_rule_is_checksummed_for_artifacts_to_bind():
    record = claim_admissibility_rule_record()
    assert record["rule_checksum"].startswith("sha256:")
    assert record["rule_checksum"] == claim_admissibility_rule_record()["rule_checksum"]
    assert any("No MOCK result" in statement for statement in record["statements"])


# ---------------------------------------------------------- 10. determinism


def test_building_the_same_prompt_twice_is_identical():
    def _build():
        return build_protocol_prompt(
            protocol=OPEN_ANIMAL_IDENTIFICATION,
            evidence=Evidence(modality="text", text="A small bird on a branch."),
            external_candidates=("bird", "cat"),
            source=BIRD,
            target=CAT,
        )

    first, second = _build(), _build()
    assert isinstance(first, BuiltPrompt)
    assert first.to_dict() == second.to_dict()


# --------------------------------------------- 11. the task domain, enforced

MIXED_SIX = ("bird", "cat", "giraffe", "microwave", "toilet", "zebra")


@pytest.mark.parametrize("non_animal", ["toilet", "microwave"])
def test_animal_identification_refuses_a_non_animal_candidate(non_animal):
    """The defect this whole domain gate exists for: the pilot's six-concept set
    contains two non-animals, and the animal question cannot screen them."""
    with pytest.raises(TaskDomainError, match=non_animal):
        build_protocol_prompt(
            protocol=OPEN_ANIMAL_IDENTIFICATION,
            evidence=Evidence(modality="image", media="<pixels>"),
            external_candidates=("bird", "cat", non_animal),
            source=BIRD,
            target=CAT,
        )


def test_animal_identification_refuses_the_whole_mixed_six_concept_set():
    with pytest.raises(TaskDomainError) as error:
        build_protocol_prompt(
            protocol=OPEN_ANIMAL_IDENTIFICATION,
            evidence=Evidence(modality="image", media="<pixels>"),
            external_candidates=MIXED_SIX,
            source=BIRD,
            target=CAT,
        )
    message = str(error.value)
    assert "toilet is domain 'furniture'" in message
    assert "microwave is domain 'appliance'" in message
    assert OPEN_ENTITY_IDENTIFICATION in message


def test_animal_identification_refuses_a_source_or_target_of_another_domain():
    for role in ("source", "target"):
        kwargs = {"source": BIRD, "target": CAT, role: concept_spec("toilet")}
        with pytest.raises(TaskDomainError, match="toilet"):
            build_protocol_prompt(
                protocol=OPEN_ANIMAL_IDENTIFICATION,
                evidence=Evidence(modality="image", media="<pixels>"),
                external_candidates=("bird", "cat"),
                **kwargs,
            )


def test_an_unspecified_domain_is_refused_rather_than_assumed():
    with pytest.raises(TaskDomainError, match="no registered domain"):
        build_protocol_prompt(
            protocol=OPEN_ANIMAL_IDENTIFICATION,
            evidence=Evidence(modality="image", media="<pixels>"),
            external_candidates=("bird", "cat", "wombat"),
            source=BIRD,
            target=CAT,
        )
    # Declaring it is what makes it admissible — and it has to be declared.
    built = build_protocol_prompt(
        protocol=OPEN_ANIMAL_IDENTIFICATION,
        evidence=Evidence(modality="image", media="<pixels>"),
        external_candidates=("bird", "cat", "wombat"),
        source=BIRD,
        target=CAT,
        domain_registry={**CONCEPT_DOMAINS, "wombat": DOMAIN_ANIMAL},
    )
    assert built.task_domain["resolved"]["wombat"] == DOMAIN_ANIMAL


def test_the_domain_neutral_protocol_accepts_the_mixed_six_concept_set():
    built = build_protocol_prompt(
        protocol=OPEN_ENTITY_IDENTIFICATION,
        evidence=Evidence(modality="image", media="<pixels>"),
        external_candidates=MIXED_SIX,
        source=BIRD,
        target=CAT,
    )
    assert built.question == OPEN_ENTITY_IDENTIFICATION_QUESTION
    assert built.model_visible_prompt == OPEN_ENTITY_IDENTIFICATION_QUESTION
    assert "animal" not in normalize(built.model_visible_prompt).split()
    assert built.leakage["passed"] is True
    assert built.task_domain["task_domain"] == DOMAIN_ENTITY
    assert built.task_domain["domain_restricted"] is False
    # Mixed categories are recorded rather than refused.
    assert built.task_domain["observed_domains"] == ["animal", "appliance", "furniture"]


def test_the_domain_neutral_protocol_supports_no_property_or_multi_hop_claim():
    decision = protocol_claim_admissibility(
        protocol=OPEN_ENTITY_IDENTIFICATION,
        leakage=_open_leakage(
            OPEN_ENTITY_IDENTIFICATION,
            instruction=OPEN_ENTITY_IDENTIFICATION_QUESTION,
        ),
        mode="real",
    )
    assert decision["granted_claim"] == "open_cross_modal_entity_identification"
    assert "downstream property recomputation" in decision["excluded_claims"]
    assert "multi-hop reasoning" in decision["excluded_claims"]
    assert "animal-specific identification" in decision["excluded_claims"]


def test_a_retired_protocol_name_names_its_replacement():
    for old, new in RETIRED_PROTOCOLS.items():
        with pytest.raises(PromptProtocolError, match=new):
            build_protocol_prompt(
                protocol=old,
                evidence=Evidence(modality="image", media="<pixels>"),
                external_candidates=("bird", "cat"),
                source=BIRD,
                target=CAT,
            )


# ------------------------------------------ 12. the legs protocols, animal-only


@pytest.mark.parametrize("protocol", [OPEN_ANIMAL_LEGS, HIDDEN_ANIMAL_LEGS])
def test_the_legs_protocols_refuse_a_non_animal_concept(protocol):
    with pytest.raises(TaskDomainError, match="toilet"):
        build_protocol_prompt(
            protocol=protocol,
            evidence=Evidence(modality="image", media="<pixels>"),
            external_candidates=("two", "four"),
            source=BIRD,
            target=concept_spec("toilet"),
        )


@pytest.mark.parametrize("protocol", [OPEN_ANIMAL_LEGS, HIDDEN_ANIMAL_LEGS])
def test_the_legs_protocols_refuse_a_missing_leg_count(protocol):
    """An animal with no registered count has no ground truth to score."""
    with pytest.raises(PropertyAnswerError, match="no registered leg count"):
        build_protocol_prompt(
            protocol=protocol,
            evidence=Evidence(modality="image", media="<pixels>"),
            external_candidates=("two", "four"),
            source=BIRD,
            target=concept_spec("dolphin", domain=DOMAIN_ANIMAL),
        )


@pytest.mark.parametrize("protocol", [OPEN_ANIMAL_LEGS, HIDDEN_ANIMAL_LEGS])
def test_the_legs_protocols_refuse_an_ambiguous_leg_count(protocol):
    with pytest.raises(PropertyAnswerError, match="ambiguous"):
        build_protocol_prompt(
            protocol=protocol,
            evidence=Evidence(modality="image", media="<pixels>"),
            external_candidates=("two", "four"),
            source=BIRD,
            target=CAT,
            leg_counts={"bird": (2,), "cat": (2, 4)},
        )


def test_the_legs_protocols_refuse_an_unregistered_answer_candidate():
    with pytest.raises(PropertyAnswerError, match="not registered answers"):
        build_protocol_prompt(
            protocol=OPEN_ANIMAL_LEGS,
            evidence=Evidence(modality="image", media="<pixels>"),
            external_candidates=("two", "seven"),
            source=BIRD,
            target=CAT,
        )


def test_resolve_leg_count_is_the_single_ground_truth():
    assert resolve_leg_count("bird") == 2
    assert resolve_leg_count("cat") == 4
    assert resolve_leg_count("spider") == 8
    assert resolve_leg_count("ant") == 6
    assert set(ANIMAL_LEG_COUNTS) >= set(COCO_ANIMAL_CATEGORIES)
    for name, counts in ANIMAL_LEG_COUNTS.items():
        assert len(set(counts)) == 1, name


# --------------------------------- 13. domain and property in the fingerprint


def _animal_fingerprint(backend, **overrides):
    kwargs = {
        "protocol": OPEN_ANIMAL_IDENTIFICATION,
        "evidence": Evidence(modality="image", media="<pixels>"),
        "external_candidates": ("bird", "cat"),
        "source": BIRD,
        "target": CAT,
        "encode_candidate": backend.encode_candidate,
    }
    kwargs.update(overrides)
    return prompt_protocol_fingerprint(
        build_protocol_prompt(**kwargs),
        model_revision=MOCK_MODEL_REVISION,
        processor_revision=MOCK_PROCESSOR_REVISION,
    )


def test_the_task_domain_is_in_the_fingerprint(backend):
    animal = _animal_fingerprint(backend)
    entity = _animal_fingerprint(backend, protocol=OPEN_ENTITY_IDENTIFICATION)
    assert animal["task_domain"] == DOMAIN_ANIMAL
    assert animal["task_domain_restricted"] is True
    assert entity["task_domain"] == DOMAIN_ENTITY
    assert entity["task_domain_restricted"] is False
    assert animal["prompt_protocol_digest"] != entity["prompt_protocol_digest"]
    assert animal["domain_registry_checksum"] == domain_registry_checksum()


def test_a_changed_domain_registry_changes_the_fingerprint(backend):
    baseline = _animal_fingerprint(backend)
    widened = _animal_fingerprint(
        backend, domain_registry={**CONCEPT_DOMAINS, "wombat": DOMAIN_ANIMAL}
    )
    assert baseline["prompt_hash"] == widened["prompt_hash"]
    assert baseline["domain_registry_checksum"] != widened["domain_registry_checksum"]
    assert baseline["prompt_protocol_digest"] != widened["prompt_protocol_digest"]


def test_the_property_schema_and_its_registry_are_in_the_fingerprint(backend):
    def _legs(**overrides):
        kwargs = {
            "protocol": OPEN_ANIMAL_LEGS,
            "evidence": Evidence(modality="image", media="<pixels>"),
            "external_candidates": ("two", "four"),
            "source": BIRD,
            "target": CAT,
            "encode_candidate": backend.encode_candidate,
        }
        kwargs.update(overrides)
        return prompt_protocol_fingerprint(
            build_protocol_prompt(**kwargs),
            model_revision=MOCK_MODEL_REVISION,
            processor_revision=MOCK_PROCESSOR_REVISION,
        )

    baseline = _legs()
    assert baseline["property_schema"] == PROPERTY_LEG_COUNT
    assert baseline["property_registry_checksum"] == property_registry_checksum()

    # A corrected registry is a different ground truth even when nothing about
    # the prompt moves.
    corrected = _legs(leg_counts={**ANIMAL_LEG_COUNTS, "bear": (2,)})
    assert corrected["prompt_hash"] == baseline["prompt_hash"]
    assert corrected["question_hash"] == baseline["question_hash"]
    assert (
        corrected["property_registry_checksum"]
        != baseline["property_registry_checksum"]
    )
    assert corrected["prompt_protocol_digest"] != baseline["prompt_protocol_digest"]

    # And an identification protocol carries no property schema at all.
    assert _animal_fingerprint(backend)["property_schema"] is None


def test_a_domain_restricted_claim_needs_its_task_domain_record():
    leakage = _open_leakage()
    without = protocol_claim_admissibility(
        protocol=OPEN_ANIMAL_IDENTIFICATION, leakage=leakage, mode="real"
    )
    assert without["admissible"] is False
    assert any("no task-domain record" in reason for reason in without["reasons"])

    contaminated = protocol_claim_admissibility(
        protocol=OPEN_ANIMAL_IDENTIFICATION,
        leakage=leakage,
        task_domain={
            "task_domain": DOMAIN_ANIMAL,
            "resolved": {"bird": DOMAIN_ANIMAL, "toilet": "furniture"},
        },
        mode="real",
    )
    assert contaminated["admissible"] is False
    assert any("toilet" in reason for reason in contaminated["reasons"])


# ------------------------------------- 14. the predeclared animal concept set


def _row(concept, *, feasible=True, unmet=()):
    return {"concept": concept, "feasible": feasible, "unmet": list(unmet)}


def test_the_animal_concept_set_comes_from_the_ranking_and_the_evidence_audit():
    ranked = [
        _row("bird"),
        _row("toilet"),
        _row("cat"),
        _row("microwave"),
        _row("giraffe", feasible=False, unmet=["distinct_images 3 < 8"]),
        _row("zebra"),
        _row("sheep"),
    ]
    selection = select_animal_concepts(ranked, n_focal=2)
    # Ranking order preserved; non-animals and infeasible concepts dropped.
    assert selection["animal_concepts"] == ["bird", "cat", "zebra", "sheep"]
    assert selection["focal"] == ["bird", "cat"]
    assert selection["non_focal"] == ["zebra", "sheep"]
    assert selection["leg_counts"] == {"bird": 2, "cat": 4, "zebra": 4, "sheep": 4}
    assert selection["selection_version"] == ANIMAL_CONCEPT_SELECTION_VERSION

    dropped = {row["concept"]: row for row in selection["excluded"]}
    assert dropped["toilet"]["stage"] == "domain"
    assert "furniture" in dropped["toilet"]["reason"]
    assert dropped["microwave"]["stage"] == "domain"
    assert dropped["giraffe"]["stage"] == "evidence_audit"
    assert "distinct_images" in dropped["giraffe"]["reason"]


def test_coverage_is_never_assumed():
    """The likely candidates are not built in: if the local data does not carry
    enough animals, this refuses instead of filling the gap."""
    with pytest.raises(PromptProtocolError, match="Coverage is"):
        select_animal_concepts([_row("bird"), _row("toilet"), _row("microwave")])
    with pytest.raises(PromptProtocolError, match="Coverage is"):
        select_animal_concepts(
            [_row("bird"), _row("cat", feasible=False, unmet=["negatives 0 < 8"])]
        )


def test_an_animal_without_a_leg_count_is_dropped_only_when_the_property_matters():
    ranked = [_row("bird"), _row("cat"), _row("dolphin"), _row("zebra")]
    registry = {**CONCEPT_DOMAINS, "dolphin": DOMAIN_ANIMAL}
    with_property = select_animal_concepts(ranked, domain_registry=registry)
    assert with_property["animal_concepts"] == ["bird", "cat", "zebra"]
    assert [row["stage"] for row in with_property["excluded"]] == ["property"]

    without_property = select_animal_concepts(
        ranked, domain_registry=registry, require_leg_counts=False
    )
    assert without_property["animal_concepts"] == ["bird", "cat", "dolphin", "zebra"]
    assert without_property["property_schema"] is None


def test_the_concept_set_cannot_be_chosen_from_post_model_rows():
    ranked = [
        {"concept": "bird", "feasible": True, "accuracy": 0.875},
        {"concept": "cat", "feasible": True},
        {"concept": "zebra", "feasible": True},
    ]
    with pytest.raises(PromptProtocolError, match="post-model field"):
        select_animal_concepts(ranked)


def test_the_selection_is_deterministic_and_checksummed():
    ranked = [_row("bird"), _row("cat"), _row("zebra")]
    first = select_animal_concepts(ranked)
    second = select_animal_concepts(list(reversed(ranked)))
    assert first == select_animal_concepts(ranked)
    # Ranking order is the selection: a different input order is a different set.
    assert second["focal"] == ["zebra", "cat"]
    assert first["selection_checksum"] != second["selection_checksum"]
