from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from jlens.mmpilot.coordinate_swap import ConceptToken, build_swap_basis_from_vectors
from jlens.mmpilot.workspace_replication import (
    TEXT_MAX_NEW_TOKENS,
    WorkspaceReplicationRefused,
    anthropic_text_tasks,
    assert_fresh_population,
    capture_source_loading,
    freeze_confirmation_design,
    freeze_loading_localization,
    holm_adjust,
    paired_binary_superiority,
    select_pair_from_loading,
    summarize_loading,
    text_replication_verdict,
    text_task_digest,
    unrestricted_greedy_completion,
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
        _GenerationBackend(),
        _generation_inputs(),
        bases={0: basis},
        alpha=1.0,
        answer="8",
    )
    assert result["answer_match"] is True
    assert result["hook_forward_passes_by_layer"] == {"0": 2}
    assert result["all_prompt_positions_patched"] is True


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
