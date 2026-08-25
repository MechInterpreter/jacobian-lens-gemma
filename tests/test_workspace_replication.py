from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from jlens.mmpilot.coordinate_swap import ConceptToken, build_swap_basis_from_vectors
from jlens.mmpilot.workspace_replication import (
    MULTIMODAL_COMPLETION_INSTRUCTION,
    MULTIMODAL_INPUT_PROTOCOL_VERSION,
    MULTIMODAL_MAX_NEW_TOKENS,
    TEXT_COMPLETION_INSTRUCTION,
    TEXT_DIAGNOSTIC_CONDITIONS,
    TEXT_INPUT_PROTOCOL_VERSION,
    TEXT_MAX_NEW_TOKENS,
    WorkspaceReplicationRefused,
    _cosine,
    anthropic_text_tasks,
    assert_fresh_population,
    build_assistant_prefill_completion_inputs,
    build_multimodal_assistant_prefill_inputs,
    capture_source_loading,
    completion_answer_matches,
    completion_identity_matches,
    freeze_confirmation_design,
    freeze_loading_localization,
    holm_adjust,
    paired_binary_superiority,
    select_pair_from_loading,
    semantic_answer_concept,
    summarize_loading,
    text_capability_verdict,
    text_diagnostic_bands,
    text_replication_verdict,
    text_swap_diagnostic_report,
    text_task_digest,
    unrestricted_greedy_completion,
    unrestricted_greedy_direct_answer_trial,
    unrestricted_greedy_swap_trial,
)


class _Backend:
    def __init__(self) -> None:
        self.blocks = nn.ModuleList([nn.Identity(), nn.Identity(), nn.Identity()])

    def forward_logits(self, tensors):
        hidden = tensors["hidden"]
        for block in self.blocks:
            hidden = block(hidden)
        return hidden


def _inputs() -> SimpleNamespace:
    return SimpleNamespace(
        tensors={
            "hidden": torch.tensor(
                [[[2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [1.0, 1.0, 0.0]]]
            )
        },
        prompt_len=3,
        final_prompt_position=2,
        modality_token_range=[0, 2],
    )


def test_text_tasks_are_frozen_and_include_the_paper_two_hop_case() -> None:
    tasks = anthropic_text_tasks()
    assert tasks[0].task_id == "spider_to_ant_legs"
    assert tasks[0].prompt == "The number of legs on the animal that spins webs is"
    assert tasks[0].clean_answer == "8"
    assert tasks[0].swapped_answer == "6"
    assert tasks[0].implicit_intermediate is True
    assert len({task.task_id for task in tasks}) == len(tasks) == 7
    assert text_task_digest() == text_task_digest(tasks)


def test_text_diagnostic_design_is_singletons_plus_suffixes_only() -> None:
    bands = text_diagnostic_bands(range(33, 41))
    assert len(bands) == 15
    assert len(set(bands)) == len(bands)
    assert {(layer,) for layer in range(33, 41)} <= set(bands)
    assert tuple(range(33, 41)) in bands
    assert tuple(range(39, 41)) in bands
    assert (33, 34) not in bands
    assert semantic_answer_concept("6") == "six"
    assert semantic_answer_concept("Beijing") == "Beijing"


class _DeviceRecordingTensor:
    def __init__(self, values) -> None:
        self.values = torch.tensor(values)
        self.to_calls = []

    def detach(self):
        return self

    def to(self, *args, **kwargs):
        self.to_calls.append((args, kwargs))
        return self.values.to(*args, **kwargs)


def test_cosine_normalizes_activation_and_lens_vector_to_cpu() -> None:
    activation = _DeviceRecordingTensor([1.0, 0.0])
    lens_vector = _DeviceRecordingTensor([1.0, 0.0])
    assert _cosine(activation, lens_vector) == pytest.approx(1.0)
    for operand in (activation, lens_vector):
        assert operand.to_calls == [
            ((), {"device": "cpu", "dtype": torch.float64})
        ]


class _PrefillProcessor:
    def __init__(self) -> None:
        self.chat_calls = []

    def __call__(self, **kwargs):
        raise AssertionError("assistant-prefill input must use the chat template")

    def apply_chat_template(self, messages, **kwargs):
        self.chat_calls.append((messages, dict(kwargs)))
        return {
            "input_ids": torch.tensor([[7, 8, 9]], dtype=torch.long),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
        }


def test_assistant_prefill_uses_continue_final_message() -> None:
    processor = _PrefillProcessor()
    backend = SimpleNamespace(processor=processor, device=torch.device("cpu"))
    inputs = build_assistant_prefill_completion_inputs(
        backend, "The capital of France is"
    )
    messages, kwargs = processor.chat_calls[0]
    assert messages == [
        {
            "role": "user",
            "content": [{"type": "text", "text": TEXT_COMPLETION_INSTRUCTION}],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "The capital of France is"}
            ],
        },
    ]
    assert kwargs == {
        "continue_final_message": True,
        "tokenize": True,
        "return_dict": True,
        "return_tensors": "pt",
    }
    assert inputs.prompt_len == 3
    assert inputs.final_prompt_position == 2
    assert inputs.route == {
        "route": "assistant_prefill_completion",
        "chat_template_used": True,
        "continue_final_message": True,
        "input_protocol": TEXT_INPUT_PROTOCOL_VERSION,
    }


def test_multimodal_prefill_text_keeps_evidence_in_user_turn() -> None:
    processor = _PrefillProcessor()
    backend = SimpleNamespace(
        processor=processor, device=torch.device("cpu"), interface={}
    )
    inputs = build_multimodal_assistant_prefill_inputs(
        backend,
        modality="text",
        caption="A bird is perched on a branch.",
        assistant_prefill="The number of legs on the animal in the evidence is",
    )
    messages, kwargs = processor.chat_calls[0]
    assert messages[0]["role"] == "user"
    assert "A bird is perched" in messages[0]["content"][0]["text"]
    assert MULTIMODAL_COMPLETION_INSTRUCTION in messages[0]["content"][0]["text"]
    assert messages[1] == {
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "The number of legs on the animal in the evidence is",
            }
        ],
    }
    assert kwargs["continue_final_message"] is True
    assert inputs.route["input_protocol"] == MULTIMODAL_INPUT_PROTOCOL_VERSION
    assert inputs.route["candidate_list_supplied"] is False
    assert inputs.route["teacher_forcing_used"] is False
    assert MULTIMODAL_MAX_NEW_TOKENS == 4


def test_multimodal_prefill_image_preserves_media_token_span() -> None:
    class ImageProcessor(_PrefillProcessor):
        def apply_chat_template(self, messages, **kwargs):
            self.chat_calls.append((messages, dict(kwargs)))
            return {
                "input_ids": torch.tensor([[7, 99, 99, 8]], dtype=torch.long),
                "attention_mask": torch.ones((1, 4), dtype=torch.long),
            }

    processor = ImageProcessor()
    backend = SimpleNamespace(
        processor=processor,
        device=torch.device("cpu"),
        interface={"image_token_id": 99},
    )
    image = object()
    inputs = build_multimodal_assistant_prefill_inputs(
        backend,
        modality="image",
        image=image,
        assistant_prefill="The animal in the evidence is",
    )
    messages, _ = processor.chat_calls[0]
    assert messages[0]["content"][0] == {"type": "image", "image": image}
    assert inputs.modality_token_range == [1, 3]
    assert inputs.modality == "image"


@pytest.mark.parametrize(
    ("generated", "answer"),
    (
        (" eight.", "8"),
        (" 8", "8"),
        (" Western Europe", "Europe"),
        (" Mandarin Chinese", "Chinese"),
        (" East Asia", "Asia"),
        (" Paris,", "Paris"),
        ("a bird", "bird"),
    ),
)
def test_semantic_head_match_accepts_fixed_natural_answer_forms(
    generated: str, answer: str
) -> None:
    assert completion_answer_matches(generated, answer) is True


@pytest.mark.parametrize(
    ("generated", "answer"),
    (
        ("not Europe", "Europe"),
        ("Europe or Asia", "Europe"),
        ("Chinese Mandarin", "Chinese"),
        ("eighteen", "8"),
        ("", "Paris"),
    ),
)
def test_semantic_head_match_rejects_negation_and_nonfinal_mentions(
    generated: str, answer: str
) -> None:
    assert completion_answer_matches(generated, answer) is False


def test_identity_match_accepts_a_lexical_concept_before_a_modifier() -> None:
    assert completion_identity_matches(" a bird in flight<turn|>", "bird") is True
    assert completion_identity_matches(" two zebras grazing", "zebra") is True
    assert completion_identity_matches(" a cat is not visible", "cat") is False
    assert completion_identity_matches(" a hawk", "bird") is False


class _GenerationBackend:
    def __init__(self) -> None:
        self.blocks = nn.ModuleList([nn.Identity()])

    def decode_token(self, token_id: int) -> str:
        return {1: " ", 2: "8"}.get(int(token_id), "x")

    def forward_logits(self, tensors):
        input_ids = tensors["input_ids"]
        seq_len = int(input_ids.shape[1])
        hidden = torch.zeros((1, seq_len, 3), dtype=torch.float32)
        for block in self.blocks:
            hidden = block(hidden)
        logits = torch.zeros((1, seq_len, 4), dtype=torch.float32)
        logits[0, -1, 1 if seq_len == 1 else 2] = 10.0
        return logits


class _ControlTerminatingBackend(_GenerationBackend):
    def decode_token(self, token_id: int) -> str:
        return {1: " four", 2: "<turn|>", 3: "garbage"}.get(int(token_id), "x")

    def forward_logits(self, tensors):
        seq_len = int(tensors["input_ids"].shape[1])
        hidden = torch.zeros((1, seq_len, 3), dtype=torch.float32)
        for block in self.blocks:
            hidden = block(hidden)
        logits = torch.zeros((1, seq_len, 4), dtype=torch.float32)
        selected = {1: 1, 2: 2}.get(seq_len, 3)
        logits[0, -1, selected] = 10.0
        return logits


def _generation_inputs() -> SimpleNamespace:
    return SimpleNamespace(
        tensors={
            "input_ids": torch.tensor([[0]], dtype=torch.long),
            "attention_mask": torch.ones((1, 1), dtype=torch.long),
        },
        prompt_len=1,
        final_prompt_position=0,
        modality_token_range=None,
    )


def test_complete_answer_endpoint_handles_gemma_style_two_token_digit() -> None:
    result = unrestricted_greedy_completion(
        _GenerationBackend(),
        _generation_inputs(),
        answer="8",
        max_new_tokens=TEXT_MAX_NEW_TOKENS,
    )
    assert result["generated_token_ids"] == [1, 2]
    assert result["generated_text"] == " 8"
    assert result["answer_match"] is True
    assert result["teacher_forcing_used"] is False
    assert result["candidate_list_supplied"] is False


def test_full_vocabulary_trace_uses_only_self_generated_prefixes() -> None:
    result = unrestricted_greedy_completion(
        _GenerationBackend(),
        _generation_inputs(),
        answer="8",
        diagnostic_token_ids={"swapped_answer_head": 2},
    )
    assert result["full_vocabulary_diagnostic_is_teacher_forced"] is False
    trace = result["full_vocabulary_diagnostic_trace"]
    assert [row["selected_token_id"] for row in trace] == [1, 2]
    assert trace[0]["tokens"]["swapped_answer_head"]["rank"] > 1
    assert trace[1]["tokens"]["swapped_answer_head"]["is_global_top1"] is True


def test_complete_answer_stops_at_gemma_turn_control_token() -> None:
    result = unrestricted_greedy_completion(
        _ControlTerminatingBackend(),
        _generation_inputs(),
        answer="4",
        max_new_tokens=4,
    )
    assert result["generated_token_ids"] == [1, 2]
    assert result["n_forward_passes"] == 2
    assert result["terminated_early"] is True
    assert result["stop_reason"] == "gemma_control_token"
    assert result["answer_match"] is True


def test_complete_answer_swap_keeps_hooks_active_for_both_decode_steps() -> None:
    source = ConceptToken("spider", 10, " spider", " {}")
    target = ConceptToken("ant", 11, " ant", " {}")
    basis = build_swap_basis_from_vectors(
        torch.tensor([1.0, 0.0, 0.0]),
        torch.tensor([0.0, 1.0, 0.0]),
        layer=0,
        source=source,
        target=target,
    )
    result = unrestricted_greedy_swap_trial(
        _ControlTerminatingBackend(),
        _generation_inputs(),
        bases={0: basis},
        alpha=1.0,
        answer="4",
        max_new_tokens=4,
    )
    assert result["answer_match"] is True
    assert result["hook_forward_passes_by_layer"] == {"0": 2}
    assert result["all_prompt_positions_patched"] is True
    diagnostics = result["intervention_diagnostics"]
    assert diagnostics["all_hooks_fired"] is True
    assert diagnostics["expected_forward_passes"] == 2
    assert diagnostics["all_finite"] is True
    assert diagnostics["post_cast_audit_passed"] is True
    assert diagnostics["post_cast_residual_audit_passed"] is True


def test_norm_matched_direct_answer_control_uses_same_hook_budget() -> None:
    source = ConceptToken("spider", 10, " spider", " {}")
    target = ConceptToken("ant", 11, " ant", " {}")
    basis = build_swap_basis_from_vectors(
        torch.tensor([1.0, 0.0, 0.0]),
        torch.tensor([0.0, 1.0, 0.0]),
        layer=0,
        source=source,
        target=target,
    )
    result = unrestricted_greedy_direct_answer_trial(
        _GenerationBackend(),
        _generation_inputs(),
        bases={0: basis},
        answer_vectors={0: torch.tensor([0.0, 0.0, 2.0])},
        answer="8",
        diagnostic_token_ids={"swapped_answer_head": 2},
    )
    assert result["hook_forward_passes_by_layer"] == {"0": 2}
    assert result["all_prompt_positions_patched"] is True
    diagnostic = result["intervention_diagnostics"]
    assert diagnostic["all_hooks_fired"] is True
    assert diagnostic["all_finite"] is True
    assert diagnostic["by_layer"]["0"]["max_relative_norm_match_error"] < 1e-6
    assert diagnostic["max_relative_cumulative_band_displacement_match_error"] < 1e-6


class _TwoLayerCancellationBackend(_GenerationBackend):
    def __init__(self) -> None:
        self.blocks = nn.ModuleList([nn.Identity(), nn.Identity()])

    def forward_logits(self, tensors):
        seq_len = int(tensors["input_ids"].shape[1])
        hidden = torch.zeros((1, seq_len, 3), dtype=torch.float32)
        hidden[..., 0] = 2.0
        for block in self.blocks:
            hidden = block(hidden)
        logits = torch.zeros((1, seq_len, 4), dtype=torch.float32)
        logits[0, -1, 1] = 10.0
        return logits


def test_direct_answer_matches_vector_sum_across_the_band_not_each_layer() -> None:
    """Coherent control steps are downscaled when exact steps partly cancel."""
    first = ConceptToken("first", 10, " first", " {}")
    second = ConceptToken("second", 11, " second", " {}")
    third = ConceptToken("third", 12, " third", " {}")
    bases = {
        0: build_swap_basis_from_vectors(
            torch.tensor([1.0, 0.0, 0.0]),
            torch.tensor([0.0, 1.0, 0.0]),
            layer=0,
            source=first,
            target=second,
        ),
        1: build_swap_basis_from_vectors(
            torch.tensor([0.0, 1.0, 0.0]),
            torch.tensor([0.0, 0.0, 1.0]),
            layer=1,
            source=second,
            target=third,
        ),
    }
    backend = _TwoLayerCancellationBackend()
    direct = unrestricted_greedy_direct_answer_trial(
        backend,
        _generation_inputs(),
        bases=bases,
        answer_vectors={
            0: torch.tensor([0.0, 0.0, 1.0]),
            1: torch.tensor([0.0, 0.0, 1.0]),
        },
        answer="8",
        max_new_tokens=1,
    )["intervention_diagnostics"]
    exact = unrestricted_greedy_swap_trial(
        backend,
        _generation_inputs(),
        bases=bases,
        alpha=1.0,
        answer="8",
        max_new_tokens=1,
    )["intervention_diagnostics"]

    expected = 8.0**0.5
    assert direct["max_exact_exchange_cumulative_band_displacement_norm"] == pytest.approx(
        expected
    )
    assert direct["max_direct_answer_cumulative_band_displacement_norm"] == pytest.approx(
        expected
    )
    assert direct["max_cumulative_band_displacement_scale"] == pytest.approx(0.5)
    assert direct["max_relative_cumulative_band_displacement_match_error"] < 1e-6
    assert exact["max_cumulative_band_displacement_norm"] == pytest.approx(expected)


def test_capture_loading_is_observation_only_and_marks_evidence_positions() -> None:
    vectors = {
        layer: {
            "bird": torch.tensor([1.0, 0.0, 0.0]),
            "cat": torch.tensor([0.0, 1.0, 0.0]),
            "zebra": torch.tensor([0.0, 0.0, 1.0]),
        }
        for layer in (0, 1, 2)
    }
    rows = capture_source_loading(
        _Backend(),
        _inputs(),
        vectors_by_layer=vectors,
        source="bird",
        target="cat",
        unrelated=("zebra",),
        sample_id="g1:image",
        modality="image",
    )
    assert len(rows) == 9
    assert {row["position_class"] for row in rows} == {
        "evidence",
        "final_prompt_token",
    }
    assert all(row["causal_result_consulted"] is False for row in rows)
    assert all(row["source_advantage"] > 0 for row in rows if row["position"] < 2)
    summary = summarize_loading(rows)
    assert summary["n_samples"] == 1
    assert summary["causal_result_consulted"] is False


def _loading_rows(*, layers=(33, 34, 35), evidence_better=True):
    rows = []
    for modality in ("image", "spoken_audio"):
        for layer in layers:
            for position_class, value in (
                ("evidence", 0.30 if evidence_better else 0.10),
                ("non_evidence", 0.10 if evidence_better else 0.30),
            ):
                rows.append(
                    {
                        "sample_id": f"{modality}-{layer}",
                        "modality": modality,
                        "layer": layer,
                        "position_class": position_class,
                        "source_advantage": value,
                    }
                )
    return rows


def test_localization_uses_loading_only_and_selects_contiguous_band() -> None:
    rows = _loading_rows(layers=(33, 34, 36, 37))
    result = freeze_loading_localization(
        rows,
        required_modalities=("image", "spoken_audio"),
        candidate_layers=(33, 34, 35, 36, 37),
    )
    assert result["verdict"] == "LOADING_LOCALIZATION_GO"
    assert result["selected_band"] == [36, 37]
    assert result["position_rule"] == "evidence_span_only"
    assert set(result["position_rule_by_modality"].values()) == {"evidence_span_only"}
    assert result["selection_depended_on_causal_outcome"] is False


def test_localization_falls_back_to_all_positions_when_evidence_does_not_win() -> None:
    result = freeze_loading_localization(
        _loading_rows(evidence_better=False),
        required_modalities=("image", "spoken_audio"),
        candidate_layers=(33, 34, 35),
    )
    assert result["selected_band"] == [33, 34, 35]
    assert result["position_rule"] == "all_prompt_positions"


def test_localization_can_gate_on_final_token_without_changing_swap_positions() -> None:
    rows = []
    for modality in ("text", "image", "spoken_audio"):
        for layer in (33, 34):
            for index in range(20):
                rows.append(
                    {
                        "sample_id": f"{modality}-{layer}-e-{index}",
                        "modality": modality,
                        "layer": layer,
                        "position_class": "evidence",
                        "source_advantage": -0.2,
                    }
                )
            rows.append(
                {
                    "sample_id": f"{modality}-{layer}-final",
                    "modality": modality,
                    "layer": layer,
                    "position_class": "final_prompt_token",
                    "source_advantage": 0.1,
                }
            )
    result = freeze_loading_localization(
        rows,
        required_modalities=("text", "image", "spoken_audio"),
        candidate_layers=(33, 34),
        primary_position_class="final_prompt_token",
        intervention_position_rule="all_prompt_positions",
    )
    assert result["selected_band"] == [33, 34]
    assert result["primary_loading_position_class"] == "final_prompt_token"
    assert result["intervention_position_rule"] == "all_prompt_positions"
    assert set(result["position_rule_by_modality"].values()) == {
        "all_prompt_positions"
    }


def test_pair_selection_uses_weakest_clean_modality_only() -> None:
    rows = []
    for source, target, values in (
        ("bird", "cat", {"text": 0.3, "image": 0.2, "spoken_audio": 0.1}),
        ("zebra", "giraffe", {"text": 0.2, "image": 0.2, "spoken_audio": 0.2}),
    ):
        for modality, value in values.items():
            rows.append(
                {
                    "source": source,
                    "target": target,
                    "modality": modality,
                    "source_advantage": value,
                }
            )
    result = select_pair_from_loading(
        rows,
        candidate_pairs=(("bird", "cat"), ("zebra", "giraffe")),
        required_modalities=("text", "image", "spoken_audio"),
    )
    assert result["selected_pair"] == ["zebra", "giraffe"]
    assert result["causal_result_consulted"] is False


def test_text_replication_gate_requires_two_hop_flexible_and_controls() -> None:
    rows = []
    for task in anthropic_text_tasks():
        rows.append(
            {
                "task_id": task.task_id,
                "clean_correct": True,
                "exact_alpha1_swapped_answer_generated": True,
                "random_swapped_answer_generated": False,
                "unrelated_swapped_answer_generated": False,
            }
        )
    assert text_replication_verdict(rows)["verdict"] == "TEXT_PAPER_REPLICATION_GO"
    rows[0]["exact_alpha1_swapped_answer_generated"] = False
    assert text_replication_verdict(rows)["verdict"] == "TEXT_PAPER_REPLICATION_NO_GO"


def test_text_replication_accepts_paper_alpha2_primary_with_matched_controls() -> None:
    rows = []
    for task in anthropic_text_tasks():
        rows.append(
            {
                "task_id": task.task_id,
                "clean_correct": True,
                "exact_primary_swapped_answer_generated": True,
                "random_primary_swapped_answer_generated": False,
                "unrelated_primary_swapped_answer_generated": False,
            }
        )
    result = text_replication_verdict(rows, primary_alpha=2.0)
    assert result["verdict"] == "TEXT_PAPER_REPLICATION_GO"
    assert result["primary_alpha"] == 2.0
    assert result["primary_alpha_role"] == "double_strength_coordinate_exchange"


def test_text_capability_gate_precedes_and_licenses_causal_spending() -> None:
    rows = [
        {"task_id": task.task_id, "clean_correct": True}
        for task in anthropic_text_tasks()
    ]
    passed = text_capability_verdict(rows)
    assert passed["verdict"] == "TEXT_PAPER_CAPABILITY_GO"
    assert passed["causal_spending_licensed"] is True
    assert passed["interventions_run"] is False

    rows[0]["clean_correct"] = False
    failed = text_capability_verdict(rows)
    assert failed["verdict"] == "TEXT_PAPER_CAPABILITY_NO_GO"
    assert failed["causal_spending_licensed"] is False


def _diagnostic_fixture(*, random_success: bool = False):
    layers = tuple(range(33, 41))
    clean_rows = []
    records = []
    audit = {
        "all_hooks_fired": True,
        "all_finite": True,
        "all_layers_are_exact_alpha_one_exchange_before_cast": True,
        "post_cast_audit_passed": True,
    }
    for task in anthropic_text_tasks():
        clean_rows.append(
            {
                "task_id": task.task_id,
                "clean": {
                    "full_vocabulary_diagnostic_trace": [
                        {
                            "tokens": {
                                "swapped_answer_head": {"logprob": -8.0}
                            }
                        }
                    ]
                },
            }
        )
        for band in text_diagnostic_bands(layers):
            for condition in TEXT_DIAGNOSTIC_CONDITIONS:
                success = condition in {
                    "exact_alpha1",
                    "direct_answer_norm_matched",
                }
                if condition == "random_alpha1" and random_success:
                    success = True
                result = {
                    "answer_match": success,
                    "full_vocabulary_diagnostic_trace": [
                        {
                            "tokens": {
                                "swapped_answer_head": {"logprob": -2.0}
                            }
                        }
                    ],
                }
                if condition == "direct_answer_norm_matched":
                    result["intervention_diagnostics"] = {
                        "all_hooks_fired": True,
                        "all_finite": True,
                        "by_layer": {
                            str(layer): {"max_relative_norm_match_error": 0.0}
                            for layer in band
                        },
                    }
                else:
                    result["intervention_diagnostics"] = audit
                records.append(
                    {
                        "task_id": task.task_id,
                        "band": list(band),
                        "condition": condition,
                        "result": result,
                    }
                )
    return layers, clean_rows, records


def test_text_diagnostic_selects_only_an_audited_specific_development_band() -> None:
    layers, clean_rows, records = _diagnostic_fixture()
    report = text_swap_diagnostic_report(
        records, clean_rows=clean_rows, layers=layers
    )
    assert report["verdict"] == "TEXT_DIAGNOSTIC_ALPHA1_CANDIDATE_FOUND"
    assert report["selected_band_for_fresh_confirmation"] == [40]
    assert report["fresh_confirmation_required"] is True
    assert report["multimodal_stage_licensed"] is False
    assert all(row["coordinate_audits_pass"] for row in report["bands"])
    assert all(row["matched_controls_pass"] for row in report["bands"])


def test_text_diagnostic_can_report_only_a_preselected_clean_loading_band() -> None:
    layers, clean_rows, records = _diagnostic_fixture()
    selected = (tuple(layers),)
    selected_records = [
        row for row in records if tuple(row["band"]) == tuple(layers)
    ]
    report = text_swap_diagnostic_report(
        selected_records,
        clean_rows=clean_rows,
        layers=layers,
        bands=selected,
    )
    assert report["verdict"] == "TEXT_DIAGNOSTIC_ALPHA1_CANDIDATE_FOUND"
    assert report["tested_bands"] == [list(layers)]
    assert report["missing_units"] == []
    assert len(report["bands"]) == 1


def test_text_diagnostic_refuses_a_candidate_reproduced_by_random_control() -> None:
    layers, clean_rows, records = _diagnostic_fixture(random_success=True)
    report = text_swap_diagnostic_report(
        records, clean_rows=clean_rows, layers=layers
    )
    assert report["verdict"] == "TEXT_DIAGNOSTIC_NO_ALPHA1_CANDIDATE"
    assert report["selected_band_for_fresh_confirmation"] is None


def test_one_broken_band_does_not_poison_an_independent_valid_candidate() -> None:
    layers, clean_rows, records = _diagnostic_fixture()
    broken = next(
        row
        for row in records
        if row["condition"] == "unrelated_alpha1"
    )
    broken["result"].pop("intervention_diagnostics")
    report = text_swap_diagnostic_report(
        records, clean_rows=clean_rows, layers=layers
    )
    assert report["verdict"] == "TEXT_DIAGNOSTIC_ALPHA1_CANDIDATE_FOUND"
    assert report["selected_band_for_fresh_confirmation"] == [40]
    assert report["bands_with_integrity_failures"] == [[33]]


def test_every_broken_band_is_an_engineering_no_go() -> None:
    layers, clean_rows, records = _diagnostic_fixture()
    for row in records:
        if row["condition"] == "unrelated_alpha1":
            row["result"].pop("intervention_diagnostics")
    report = text_swap_diagnostic_report(
        records, clean_rows=clean_rows, layers=layers
    )
    assert report["verdict"] == "TEXT_DIAGNOSTIC_ENGINEERING_NO_GO"
    assert report["selected_band_for_fresh_confirmation"] is None


def test_confirmation_design_is_frozen_only_after_both_gates() -> None:
    text = {"verdict": "TEXT_PAPER_REPLICATION_GO"}
    localization = {
        "verdict": "LOADING_LOCALIZATION_GO",
        "selected_band": [35, 36, 37],
        "position_rule": "evidence_span_only",
    }
    design = freeze_confirmation_design(
        text_verdict=text,
        localization=localization,
        pair=("bird", "cat"),
        prompt_protocol="implicit_animal_property.v1",
        development_population_digest="sha256:dev",
    )
    assert design["primary_alpha"] == 1.0
    assert design["sensitivity_alpha"] == 0.75
    assert design["fresh_population_required"] is True
    assert design["teacher_forcing_used"] is False
    with pytest.raises(WorkspaceReplicationRefused, match="text-only"):
        freeze_confirmation_design(
            text_verdict={"verdict": "TEXT_PAPER_REPLICATION_NO_GO"},
            localization=localization,
            pair=("bird", "cat"),
            prompt_protocol="implicit_animal_property.v1",
            development_population_digest="sha256:dev",
        )


def test_confirmation_design_records_paper_double_strength_primary() -> None:
    design = freeze_confirmation_design(
        text_verdict={"verdict": "TEXT_PAPER_REPLICATION_GO"},
        localization={
            "verdict": "LOADING_LOCALIZATION_GO",
            "selected_band": [27, 28, 29],
            "position_rule": "all_prompt_positions",
        },
        pair=("bird", "cat"),
        alpha=2.0,
        sensitivity_alpha=1.0,
        prompt_protocol="implicit_animal_property.v1",
        development_population_digest="sha256:dev",
    )
    assert design["primary_alpha"] == 2.0
    assert design["primary_alpha_role"] == "double_strength_coordinate_exchange"
    assert design["sensitivity_alpha"] == 1.0
    assert design["sensitivity_alpha_role"] == "exact_coordinate_exchange_sensitivity"


def test_prospective_loading_followup_preserves_no_go_but_freezes_band() -> None:
    localization = {
        "verdict": "LOADING_LOCALIZATION_NO_GO",
        "selected_band": [],
        "eligible_layers": [],
        "position_rule": "all_prompt_positions",
        "position_rule_by_modality": {
            "text": "all_prompt_positions",
            "image": "all_prompt_positions",
            "spoken_audio": "all_prompt_positions",
        },
    }
    followup = {
        "selected_band": [21],
        "causal_result_consulted": False,
        "multimodal_causal_outcomes_opened": False,
        "previous_loading_no_go_remains_unchanged": True,
    }
    design = freeze_confirmation_design(
        text_verdict={"verdict": "L21_TEXT_CONFIRMATION_GO"},
        localization=localization,
        pair=("bird", "giraffe"),
        prompt_protocol="implicit_animal_property.v1",
        development_population_digest="sha256:dev",
        prospective_loading_followup=followup,
    )
    assert design["layer_band"] == [21]
    assert design["loading_gate_overridden"] is True
    assert design["prospective_loading_followup"][
        "previous_loading_no_go_remains_unchanged"
    ] is True

    followup["multimodal_causal_outcomes_opened"] = True
    with pytest.raises(WorkspaceReplicationRefused, match="outcomes were opened"):
        freeze_confirmation_design(
            text_verdict={"verdict": "L21_TEXT_CONFIRMATION_GO"},
            localization=localization,
            pair=("bird", "giraffe"),
            prompt_protocol="implicit_animal_property.v1",
            development_population_digest="sha256:dev",
            prospective_loading_followup=followup,
        )


def test_confirmation_uses_the_audited_text_band_and_multimodal_loading() -> None:
    diagnostic = {
        "version": "diagnostic.v2",
        "verdict": "TEXT_DIAGNOSTIC_ALPHA1_CANDIDATE_FOUND",
        "selected_band_for_fresh_confirmation": [35, 36],
        "report_checksum": "sha256:text",
    }
    localization = {
        "verdict": "LOADING_LOCALIZATION_GO",
        "selected_band": [36, 37, 38],
        "eligible_layers": [33, 34, 35, 36, 37, 38, 39, 40],
        "position_rule": "modality_specific",
        "position_rule_by_modality": {
            "text": "all_prompt_positions",
            "image": "evidence_span_only",
            "spoken_audio": "evidence_span_only",
        },
    }
    design = freeze_confirmation_design(
        text_diagnostic=diagnostic,
        localization=localization,
        pair=("bird", "cat"),
        prompt_protocol="implicit_animal_property.v1",
        development_population_digest="sha256:dev",
    )
    assert design["layer_band"] == [35, 36]
    assert design["text_causal_evidence"]["report_checksum"] == "sha256:text"

    localization["eligible_layers"] = [36, 37, 38]
    with pytest.raises(WorkspaceReplicationRefused, match="not cleanly source-loaded"):
        freeze_confirmation_design(
            text_diagnostic=diagnostic,
            localization=localization,
            pair=("bird", "cat"),
            prompt_protocol="implicit_animal_property.v1",
            development_population_digest="sha256:dev",
        )


def test_fresh_population_refuses_any_development_overlap() -> None:
    groups = [
        {"group_id": "g1", "image_id": "i1"},
        {"group_id": "g2", "image_id": "i2"},
    ]
    proof = assert_fresh_population(groups, forbidden_image_ids=("old",))
    assert proof["fresh"] is True
    with pytest.raises(WorkspaceReplicationRefused, match="not fresh"):
        assert_fresh_population(groups, forbidden_image_ids=("i2",))


def test_paired_binary_test_and_holm_are_exact_and_monotone() -> None:
    result = paired_binary_superiority([True] * 6, [False] * 6)
    assert result["wins"] == 6
    assert result["losses"] == 0
    assert result["one_sided_exact_p"] == pytest.approx(1 / 64)
    adjusted = holm_adjust({"a": 0.01, "b": 0.02, "c": 0.50})
    assert adjusted == {"a": pytest.approx(0.03), "b": pytest.approx(0.04), "c": 0.5}
