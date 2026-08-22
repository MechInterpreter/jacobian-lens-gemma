from __future__ import annotations

import json
from pathlib import Path

import pytest

from jlens.metadata import file_sha256
from jlens.mmpilot.l21_confirmation import (
    DISCOVERY_FINGERPRINT,
    L21MultimodalThresholds,
    L21TextThresholds,
    assert_disjoint_from_discovery,
    discover_l21_run,
    l21_multimodal_confirmation_verdict,
    l21_text_confirmation_verdict,
    probe_swap_tasks,
    task_level_loading_admission,
)
from jlens.mmpilot.workspace_replication import (
    anthropic_text_tasks_expanded_v2,
    swapped_answer_diagnostic_tokens,
)

ROOT = Path(__file__).resolve().parent.parent


def test_probe_swap_is_a_fresh_90_item_population() -> None:
    tasks = probe_swap_tasks(ROOT / "data" / "experiments" / "probe-swap.json")
    assert len(tasks) == 90
    assert len({task.task_id for task in tasks}) == 90
    assert all(task.implicit_intermediate for task in tasks)
    result = assert_disjoint_from_discovery(tasks, anthropic_text_tasks_expanded_v2())
    assert result["disjoint"] is True
    assert result["task_id_overlap"] == []
    assert result["prompt_overlap"] == []


def test_missing_answer_head_only_disables_optional_diagnostic() -> None:
    assert swapped_answer_diagnostic_tokens(
        "Spanish", {}, allow_missing_head=True
    ) == {}
    with pytest.raises(KeyError, match="Spanish"):
        swapped_answer_diagnostic_tokens("Spanish", {})


def test_discovers_only_the_pinned_l21_run(tmp_path: Path) -> None:
    run = tmp_path / "mmworkspace" / "mmworkspace_real_l21"
    metric = run / "units" / "metric"
    metric.mkdir(parents=True)
    selection = {
        "selected_band": [21],
        "selected_instrument": "matched_text_r",
        "selection_digest": "sha256:selection",
    }
    (metric / "loading_first_selection.json").write_text(
        json.dumps(
            {
                "fingerprint_digest": DISCOVERY_FINGERPRINT,
                "payload": selection,
            }
        ),
        encoding="utf-8",
    )
    lenses = run / "r_lenses"
    lenses.mkdir()
    records = {}
    for arm in ("text", "pooled"):
        path = lenses / f"lens.{arm}.pt"
        path.write_bytes(arm.encode())
        records[arm] = {"path": str(path), "checksum": file_sha256(str(path))}
    (run / "r_lens_inventory.json").write_text(
        json.dumps({"lenses": records}), encoding="utf-8"
    )
    result = discover_l21_run(tmp_path)
    assert result["run_dir"] == str(run)
    assert result["selected_band"] == [21]
    assert result["artifacts"]["text"]["checksum"] == records["text"]["checksum"]


def test_task_level_loading_is_not_a_median_gate() -> None:
    rows = [
        {
            "sample_id": "a",
            "layer": 21,
            "position_class": "final_prompt_token",
            "source_cosine": 0.002,
            "source_advantage": 0.001,
        },
        {
            "sample_id": "b",
            "layer": 21,
            "position_class": "final_prompt_token",
            "source_cosine": 0.003,
            "source_advantage": -0.001,
        },
    ]
    result = task_level_loading_admission(rows, task_ids=("a", "b"))
    assert result["eligible_task_ids"] == ["a"]
    assert result["ineligible_task_ids"] == ["b"]
    assert result["causal_outcome_consulted"] is False


def test_text_confirmation_passes_six_clean_exact_wins() -> None:
    source = probe_swap_tasks(ROOT / "data" / "experiments" / "probe-swap.json")
    tasks = source[:30]
    rows = []
    for index, task in enumerate(tasks):
        rows.append(
            {
                "task_id": task.task_id,
                "exact_swapped_answer_generated": index < 6,
                "zero_swapped_answer_generated": False,
                "random_swapped_answer_generated": False,
                "unrelated_swapped_answer_generated": False,
                "integrity_passed": True,
            }
        )
    result = l21_text_confirmation_verdict(
        rows,
        eligible_tasks=tasks,
        thresholds=L21TextThresholds(min_success_categories=1),
    )
    assert result["verdict"] == "L21_TEXT_CONFIRMATION_GO"
    assert result["n_exact_successes"] == 6
    assert all(row["passed"] for row in result["paired_controls"].values())


def test_text_confirmation_refuses_missing_or_control_tied_result() -> None:
    tasks = probe_swap_tasks(ROOT / "data" / "experiments" / "probe-swap.json")[:30]
    rows = [
        {
            "task_id": task.task_id,
            "exact_swapped_answer_generated": index < 10,
            "zero_swapped_answer_generated": index < 10,
            "random_swapped_answer_generated": False,
            "unrelated_swapped_answer_generated": False,
            "integrity_passed": True,
        }
        for index, task in enumerate(tasks)
    ]
    result = l21_text_confirmation_verdict(
        rows,
        eligible_tasks=tasks,
        thresholds=L21TextThresholds(min_success_categories=1),
    )
    assert result["verdict"] == "L21_TEXT_CONFIRMATION_NO_GO"
    assert result["paired_controls"]["zero"]["passed"] is False


def test_multimodal_confirmation_pools_directions_but_requires_each_modality() -> None:
    rows = []
    for modality in ("text", "image", "spoken_audio"):
        for index in range(16):
            rows.append(
                {
                    "prompt_kind": "property",
                    "modality": modality,
                    "clean_correct": True,
                    "primary_success": index < 4,
                    "zero_success": False,
                    "random_success": False,
                    "unrelated_success": False,
                    "integrity_passed": True,
                }
            )
    result = l21_multimodal_confirmation_verdict(
        rows, thresholds=L21MultimodalThresholds()
    )
    assert result["verdict"] == "L21_TRIMODAL_DOWNSTREAM_RECOMPUTATION_GO"
    assert all(row["passed"] for row in result["modalities"])

    rows[32]["primary_success"] = False
    rows[33]["primary_success"] = False
    result = l21_multimodal_confirmation_verdict(rows)
    assert result["verdict"] == "L21_TRIMODAL_DOWNSTREAM_RECOMPUTATION_NO_GO"


def test_discovery_refuses_ambiguity(tmp_path: Path) -> None:
    with pytest.raises(Exception, match="exactly one"):
        discover_l21_run(tmp_path)
