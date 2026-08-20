from __future__ import annotations

from pathlib import Path

import pytest
import torch

from jlens.lens import JacobianLens
from jlens.metadata import file_sha256
from jlens.mmpilot.coordinate_swap import resolve_concept_token
from jlens.mmpilot.mock import MockPilotBackend, MockWorld
from jlens.mmpilot.multimodal_lens import (
    MultimodalLensRefused,
    answer_equivalence_record,
    build_matched_plan,
    build_swap_bases_for_lens,
    fit_arm,
    fit_budget,
    jacobian_for_built_inputs,
    load_completed_alpha_sweep_source,
    load_completed_causal_source,
    normalize_open_answer_surface,
    open_answer_matches,
    plan_units,
    select_causal_groups,
    unrestricted_swap_trial,
)
from jlens.mmpilot.store import payload_checksum


def groups(n: int = 12) -> list[dict]:
    rows = []
    for index in range(n):
        concept = "bird" if index >= n - 4 else None
        caption = (
            f"A bird standing in a field number {index}"
            if concept
            else f"A neutral landscape scene number {index} with clouds"
        )
        rows.append(
            {
                "group_id": f"g{index}",
                "image_id": f"i{index}",
                "caption": caption,
                "image_path": f"/images/{index}.jpg",
                "audio_path": f"/audio/{index}.wav",
                "concept": concept,
            }
        )
    return rows


def mock_inputs(backend, unit):
    evidence = backend.world.evidence(
        concepts_present=(), modality=unit.modality, nuisance_key=unit.unit_id
    )
    kwargs = {"prompt": unit.prompt, "modality": unit.modality}
    if unit.modality == "image":
        kwargs["image"] = evidence
    elif unit.modality == "spoken_audio":
        kwargs["audio"] = evidence
    return backend.build_inputs(**kwargs)


def test_matched_plan_is_balanced_disjoint_and_concept_neutral():
    plan = build_matched_plan(
        groups(), n_fit_groups=5, n_eval_groups=2, seed="fixed", excluded_eval_concepts=("bird",)
    )
    assert len(plan_units(plan, "text")) == 5
    assert len(plan_units(plan, "image")) == 5
    assert len(plan_units(plan, "spoken_audio")) == 5
    assert len(plan_units(plan, "pooled")) == 5
    assert plan["fit_eval_image_overlap"] == []
    assert {row["modality"] for row in plan["arms"]["pooled"]} == {
        "text",
        "image",
        "spoken_audio",
    }
    assert sorted(plan["pooled_modality_counts"].values()) == [1, 2, 2]
    assert all("bird" not in row["caption"].lower() for row in plan["fit_groups"])


def test_multimodal_jacobian_uses_processor_tensors():
    backend = MockPilotBackend(MockWorld(), n_layers=4)
    plan = build_matched_plan(
        groups(), n_fit_groups=2, n_eval_groups=1, seed="j", excluded_eval_concepts=("bird",)
    )
    unit = plan_units(plan, "image")[0]
    built = mock_inputs(backend, unit)
    matrices, seq_len, n_valid = jacobian_for_built_inputs(
        backend, built, (1, 2), target_layer=3, dim_batch=4, skip_first=2
    )
    assert seq_len == built.prompt_len
    assert n_valid > 0
    assert set(matrices) == {1, 2}
    assert all(matrix.shape == (backend.d_model, backend.d_model) for matrix in matrices.values())
    assert all(torch.isfinite(matrix).all() for matrix in matrices.values())


def test_fit_arm_resumes_and_refuses_changed_fingerprint(tmp_path: Path):
    backend = MockPilotBackend(MockWorld(), n_layers=4)
    plan = build_matched_plan(
        groups(), n_fit_groups=2, n_eval_groups=1, seed="resume", excluded_eval_concepts=("bird",)
    )
    units = plan_units(plan, "text")
    checkpoint = tmp_path / "text.pt"
    first = fit_arm(
        backend,
        units,
        build_inputs=lambda unit: mock_inputs(backend, unit),
        source_layers=(1, 2),
        target_layer=3,
        checkpoint_path=checkpoint,
        arm="text",
        scientific_fingerprint="sha256:one",
        dim_batch=4,
        skip_first=2,
        checkpoint_every=1,
    )
    second = fit_arm(
        backend,
        units,
        build_inputs=lambda unit: mock_inputs(backend, unit),
        source_layers=(1, 2),
        target_layer=3,
        checkpoint_path=checkpoint,
        arm="text",
        scientific_fingerprint="sha256:one",
        dim_batch=4,
        skip_first=2,
        checkpoint_every=1,
    )
    assert first.n_prompts == second.n_prompts == 2
    assert torch.equal(first.jacobians[1], second.jacobians[1])
    with pytest.raises(MultimodalLensRefused, match="refusing to mix"):
        fit_arm(
            backend,
            units,
            build_inputs=lambda unit: mock_inputs(backend, unit),
            source_layers=(1, 2),
            target_layer=3,
            checkpoint_path=checkpoint,
            arm="text",
            scientific_fingerprint="sha256:changed",
            dim_batch=4,
            skip_first=2,
        )


def test_causal_population_is_fresh_and_label_supported():
    selected = select_causal_groups(
        groups(), concepts=("bird",), n_per_concept=2, excluded_image_ids=("i8",), seed="c"
    )
    assert len(selected["bird"]) == 2
    assert all(row["image_id"] != "i8" for row in selected["bird"])


def test_causal_population_reads_expanded_manifest_annotations():
    rows = []
    for index in range(4):
        rows.append(
            {
                "group_id": f"expanded_g{index}",
                "image_id": f"expanded_i{index}",
                "caption": f"A bird flying over water number {index}",
                "image_path": f"/images/{index}.jpg",
                "audio_path": f"/audio/{index}.wav",
                "concept_annotations": ["bird", "water"],
            }
        )
    selected = select_causal_groups(
        rows,
        concepts=("bird",),
        n_per_concept=3,
        excluded_image_ids=(),
        seed="expanded-manifest-v3",
    )
    assert len(selected["bird"]) == 3


def test_exact_swap_trial_is_unrestricted():
    backend = MockPilotBackend(MockWorld(), n_layers=4)
    identity = torch.eye(backend.d_model)
    lens = JacobianLens(
        jacobians={2: identity}, n_prompts=1, d_model=backend.d_model
    )
    source = resolve_concept_token(backend.encode_token, "bird")
    target = resolve_concept_token(backend.encode_token, "cat")
    bases = build_swap_bases_for_lens(
        lens,
        backend.unembedding_weight(),
        layers=(2,),
        source=source,
        target=target,
    )
    evidence = backend.world.evidence(
        concepts_present=("bird",), modality="text", nuisance_key="bird"
    )
    inputs = backend.build_inputs(
        prompt="What animal is present? Answer with its name. Answer:",
        modality="image",
        image=evidence,
    )
    row = unrestricted_swap_trial(backend, inputs, bases=bases, alpha=1.0)
    assert row["teacher_forcing_used"] is False
    assert row["candidate_list_supplied"] is False
    assert row["layers_patched"] == [2]
    assert row["positions_patched"]["2"] == list(range(inputs.prompt_len))


def test_swap_trial_records_graded_full_vocabulary_diagnostics() -> None:
    backend = MockPilotBackend(MockWorld(), n_layers=4)
    lens = JacobianLens(
        jacobians={2: torch.eye(backend.d_model)},
        n_prompts=1,
        d_model=backend.d_model,
    )
    source = resolve_concept_token(backend.encode_token, "bird")
    target = resolve_concept_token(backend.encode_token, "cat")
    bases = build_swap_bases_for_lens(
        lens,
        backend.unembedding_weight(),
        layers=(2,),
        source=source,
        target=target,
    )
    evidence = backend.world.evidence(
        concepts_present=("bird",), modality="text", nuisance_key="graded"
    )
    inputs = backend.build_inputs(
        prompt="What animal is present? Answer:", modality="image", image=evidence
    )
    clean = backend.forward_logits(inputs.tensors)[
        0, inputs.final_prompt_position
    ].float()
    row = unrestricted_swap_trial(
        backend,
        inputs,
        bases=bases,
        alpha=2.0,
        target_token_id=target.token_id,
        source_token_id=source.token_id,
        clean_logits=clean,
        compact_positions=True,
    )
    assert row["alpha_role"] == "extrapolation"
    assert row["alpha_is_extrapolation"] is True
    assert row["all_prompt_positions_patched"] is True
    assert row["positions_patched"]["2"] == {
        "start": 0,
        "stop_exclusive": inputs.prompt_len,
        "count": inputs.prompt_len,
        "contiguous": True,
    }
    for key in (
        "target_logit_delta",
        "target_rank_improvement",
        "target_probability_delta",
        "source_logit_delta",
        "kl_clean_to_patched",
        "max_activation_norm_ratio",
        "max_update_to_activation_norm_ratio",
    ):
        assert torch.isfinite(torch.tensor(row[key]))


def test_budget_counts_all_four_arms():
    budget = fit_budget(n_fit_groups=250, n_layers=8, d_model=2560, dim_batch=8)
    assert budget["prompts_by_arm"] == {
        "text": 250,
        "image": 250,
        "spoken_audio": 250,
        "pooled": 250,
    }
    assert budget["total_prompt_forwards"] == 1000
    assert budget["total_backward_passes"] == 320_000


def test_open_answer_equivalence_is_only_case_and_whitespace() -> None:
    assert normalize_open_answer_surface("  Cat\n") == "cat"
    assert open_answer_matches(" bird", "BIRD")
    assert open_answer_matches("  2 ", "2")
    assert not open_answer_matches("birds", "bird")
    assert not open_answer_matches("crow", "bird")
    assert not open_answer_matches("cat.", "cat")
    record = answer_equivalence_record()
    assert record["semantic_aliases"] == []
    assert record["punctuation_removed"] is False
    assert record["protocol_digest"].startswith("sha256:")


def _write_report(path: Path, body: dict) -> tuple[dict, str]:
    checksum = payload_checksum(body)
    report = {**body, "report_checksum": checksum}
    path.write_text(__import__("json").dumps(report), encoding="utf-8")
    return report, checksum


def test_completed_causal_source_is_pinned_and_excludes_screened_images(
    tmp_path: Path,
) -> None:
    lens_dir = tmp_path / "lenses"
    lens_dir.mkdir()
    checksums = {}
    for arm in ("text", "image", "spoken_audio", "pooled"):
        lens = JacobianLens(
            jacobians={2: torch.eye(4)}, n_prompts=99, d_model=4
        )
        path = lens_dir / f"lens.{arm}.pt"
        lens.save(str(path))
        checksums[arm] = file_sha256(str(path))

    cross, cross_checksum = _write_report(
        tmp_path / "multimodal_lens_cross_eval_report.json",
        {"version": "cross", "cells": []},
    )
    causal, causal_checksum = _write_report(
        tmp_path / "multimodal_lens_causal_comparison_report.json",
        {
            "protocol": "old",
            "verdict": "CAPABILITY_NO_GO",
            "clean_screen": [
                {"image_id": "spent-2"},
                {"image_id": "spent-1"},
                {"image_id": "spent-1"},
            ],
        },
    )
    final, final_checksum = _write_report(
        tmp_path / "matched_multimodal_jlens_report.json",
        {
            "schema": "final",
            "scientific_fingerprint": "sha256:source",
            "lens_checksums": checksums,
            "cross_evaluation": cross,
            "causal_comparison": causal,
        },
    )
    assert final["report_checksum"] == final_checksum
    source = load_completed_causal_source(
        tmp_path,
        expected_final_report_checksum=final_checksum,
        expected_cross_report_checksum=cross_checksum,
        expected_causal_report_checksum=causal_checksum,
        expected_lens_checksums=checksums,
    )
    assert source["excluded_image_ids"] == ["spent-1", "spent-2"]
    assert source["n_excluded_images"] == 2
    assert source["cross_report_path"].endswith(
        "multimodal_lens_cross_eval_report.json"
    )
    assert source["source_digest"].startswith("sha256:")


def test_completed_causal_source_refuses_a_changed_lens(tmp_path: Path) -> None:
    with pytest.raises(MultimodalLensRefused, match="missing completed"):
        load_completed_causal_source(
            tmp_path,
            expected_final_report_checksum="sha256:missing",
            expected_cross_report_checksum="sha256:missing",
            expected_causal_report_checksum="sha256:missing",
            expected_lens_checksums={},
        )


def test_completed_alpha_sweep_source_freezes_the_measured_population(
    tmp_path: Path,
) -> None:
    checksums = {
        arm: f"sha256:{arm}" for arm in ("text", "image", "spoken_audio", "pooled")
    }
    rows = []
    for source, target in (("bird", "cat"), ("cat", "bird")):
        for index in range(8):
            for modality in ("text", "image", "spoken_audio"):
                for prompt_kind in ("identity", "property"):
                    rows.append(
                        {
                            "source": source,
                            "target": target,
                            "group_id": f"{source}-g{index}",
                            "image_id": f"{source}-i{index}",
                            "modality": modality,
                            "prompt_kind": prompt_kind,
                            "arms": {
                                arm: {
                                    "exact": {
                                        "patched_top_token_id": index,
                                        "success": False,
                                    }
                                }
                                for arm in ("text", "pooled")
                            },
                        }
                    )
    causal, causal_checksum = _write_report(
        tmp_path / "multimodal_lens_causal_comparison_report.json",
        {
            "protocol": "matched_multimodal_jlens_unrestricted_swap.v3",
            "verdict": "MEASURED",
            "primary_alpha": 1.0,
            "teacher_forcing_used": False,
            "candidate_list_supplied": False,
            "arms_compared": ["text", "pooled"],
            "controls": ["random", "unrelated"],
            "recruited_counts": {"bird": 8, "cat": 8},
            "clean_capability_required_in_every_modality_and_endpoint": True,
            "source_run_provenance": {
                "source_digest": "sha256:lenses",
                "lens_checksums": checksums,
            },
            "fresh_population": {
                "candidate_count_per_concept": 96,
                "excluded_previous_screen_images": 64,
                "causal_population_digest": "sha256:population",
            },
            "answer_equivalence": answer_equivalence_record(),
            "rows": rows,
        },
    )
    final, final_checksum = _write_report(
        tmp_path / "matched_multimodal_jlens_report.json",
        {
            "scientific_fingerprint": "sha256:alpha-one",
            "lens_checksums": checksums,
            "causal_comparison": causal,
        },
    )
    assert final["causal_comparison"]["report_checksum"] == causal_checksum
    source = load_completed_alpha_sweep_source(
        tmp_path,
        expected_final_report_checksum=final_checksum,
        expected_causal_report_checksum=causal_checksum,
        expected_scientific_fingerprint="sha256:alpha-one",
        expected_lens_checksums=checksums,
        expected_lens_source_digest="sha256:lenses",
    )
    assert {key: len(value) for key, value in source["groups_by_source"].items()} == {
        "bird": 8,
        "cat": 8,
    }
    assert source["population_digest"] == "sha256:population"
    assert len(source["alpha1_exact_outcomes"]) == 192
    assert source["source_digest"].startswith("sha256:")
