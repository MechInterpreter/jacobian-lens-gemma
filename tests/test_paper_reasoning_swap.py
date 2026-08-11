from __future__ import annotations

import pytest

from jlens.mmpilot.evidence import EvidenceConfig
from jlens.mmpilot.paper_reasoning_swap import (
    PaperSwapRefused,
    PaperSwapThresholds,
    PaperSwapV2Thresholds,
    amend_paper_v2_report_direction_matching,
    hidden_animal_population,
    independent_layer_record,
    paper_alpha2_confirmation_verdict,
    paper_onset_verdict,
    paper_onset_verdict_v2,
    sampled_suffix_bands,
    select_capability_eligible_samples,
    summarize_cells,
)
from jlens.mmpilot.store import payload_checksum


def test_sampled_suffix_bands_are_ranges_on_the_confirmed_grid():
    assert sampled_suffix_bands(
        (32, 35, 38, 40), validated_layers=(32, 35, 38, 40)
    ) == ((32, 35, 38, 40), (35, 38, 40), (38, 40), (40,))
    with pytest.raises(PaperSwapRefused, match="lack independent"):
        sampled_suffix_bands((32, 35), validated_layers=(32,))
    with pytest.raises(PaperSwapRefused, match="increasing"):
        sampled_suffix_bands((35, 32), validated_layers=(32, 35))


def test_v2_layers_are_independent_single_exchanges():
    record = independent_layer_record(
        (32, 35, 38, 40), validated_layers=(32, 35, 38, 40)
    )
    assert record["bands"] == [[32], [35], [38], [40]]
    assert record["one_exchange_per_forward_pass"] is True
    assert record["repeated_exchange_forbidden"] is True
    with pytest.raises(PaperSwapRefused, match="lack independent"):
        independent_layer_record((32, 35), validated_layers=(32,))


def _groups():
    rows = []
    for concept in ("bird", "cat", "zebra"):
        for index in range(4):
            rows.append(
                {
                    "group_id": f"{concept}-{index}",
                    "image_id": f"{concept}-image-{index}",
                    "image_path": f"/{concept}-{index}.jpg",
                    "audio_path": f"/{concept}-{index}.wav",
                    "caption": f"A creature near a tree number sample {index}",
                    "concept_annotations": [concept],
                    "annotation_source": "mock-coco",
                }
            )
    # This otherwise-valid row leaks both the entity and an answer.
    rows.append(
        {
            "group_id": "leaky",
            "image_id": "leaky-image",
            "image_path": "/leaky.jpg",
            "audio_path": "/leaky.wav",
            "caption": "A bird standing on two legs",
            "concept_annotations": ["bird"],
            "annotation_source": "mock-coco",
        }
    )
    return rows


def test_hidden_population_uses_annotations_and_hides_entity_and_answer():
    config = EvidenceConfig(
        lexicon={"bird": ("bird",), "cat": ("cat",), "zebra": ("zebra",)},
        coco_categories={"bird": ("bird",), "cat": ("cat",), "zebra": ("zebra",)},
        require_visual_evidence=True,
        require_caption_evidence=False,
    )
    result = hidden_animal_population(
        _groups(),
        concept_names=("bird", "cat", "zebra"),
        evidence_config=config,
        images_per_concept=3,
        seed="paper-test",
    )
    assert result["n_groups"] == result["n_distinct_images"] == 9
    assert result["one_group_per_image"] is True
    assert result["rejections"]["entity_surface_in_caption_or_transcript"] == 1
    assert all("bird" not in row["caption"].lower() for row in result["groups"])
    again = hidden_animal_population(
        list(reversed(_groups())),
        concept_names=("bird", "cat", "zebra"),
        evidence_config=config,
        images_per_concept=3,
        seed="paper-test",
    )
    assert [row["group_id"] for row in result["groups"]] == [
        row["group_id"] for row in again["groups"]
    ]


def _raw_records():
    rows = []
    starts = (32, 35)
    for start in starts:
        for arm in ("intermediate", "answer"):
            readouts = ("identity", "property")
            for condition in ("swap", "zero", "random", "unrelated", "position_control"):
                for readout in readouts:
                    for image in range(4):
                        target_prediction = (
                            condition == "swap"
                            and ((arm == "intermediate" and start == 32) or start == 35)
                            and not (arm == "answer" and readout == "identity")
                        )
                        rows.append(
                            {
                                "start_layer": start,
                                "arm": arm,
                                "condition": condition,
                                "modality": "text",
                                "source": "bird",
                                "target": "cat",
                                "readout": readout,
                                "image_id": f"bird-{image}",
                                "source_answer": "bird" if readout == "identity" else "two",
                                "target_answer": "cat" if readout == "identity" else "four",
                                "clean_prediction": "bird" if readout == "identity" else "two",
                                "prediction": (
                                    ("cat" if readout == "identity" else "four")
                                    if target_prediction
                                    else ("bird" if readout == "identity" else "two")
                                ),
                                "target_margin_change": 2.0 if target_prediction else 0.0,
                            }
                        )
    return rows


def test_verdict_compares_exact_intermediate_and_answer_onsets():
    cells = summarize_cells(_raw_records())
    verdict = paper_onset_verdict(
        cells,
        bands=((32, 35), (35,)),
        directed_pairs=({"source": "bird", "target": "cat"},),
        modalities=("text",),
        thresholds=PaperSwapThresholds(control_margin=0.1),
    )
    assert verdict["verdict"] == "PAPER_STYLE_EARLIER_INTERMEDIATE_GO"
    assert verdict["intermediate_onset_start_layer"] == 32
    assert verdict["answer_onset_start_layer"] == 35
    assert verdict["native_direct_readout_used"] is False
    assert verdict["source_derived_steering_used"] is False


def test_intermediate_identity_and_property_must_flip_on_the_same_images():
    rows = _raw_records()
    for row in rows:
        if (
            row["start_layer"] == 32
            and row["arm"] == "intermediate"
            and row["condition"] == "swap"
        ):
            image = int(str(row["image_id"]).rsplit("-", 1)[1])
            should_flip = image < 2 if row["readout"] == "identity" else image >= 2
            row["prediction"] = row["target_answer"] if should_flip else row["source_answer"]
            row["target_margin_change"] = 2.0 if should_flip else 0.0
    verdict = paper_onset_verdict(
        summarize_cells(rows),
        bands=((32, 35), (35,)),
        directed_pairs=({"source": "bird", "target": "cat"},),
        modalities=("text",),
        thresholds=PaperSwapThresholds(control_margin=0.1),
    )
    assert verdict["intermediate_onset_start_layer"] == 35
    assert verdict["verdict"] == "PAPER_STYLE_SAME_TESTED_DEPTH"


def test_pseudoreplication_is_refused():
    rows = _raw_records()
    rows[1]["image_id"] = rows[0]["image_id"]
    with pytest.raises(PaperSwapRefused, match="rows but"):
        summarize_cells(rows)


def _clean_records(*, n: int = 6, fail_last_property: bool = False):
    rows = []
    for concept in ("bird", "cat"):
        for modality in ("text", "image", "spoken_audio"):
            for image in range(n):
                for readout in ("identity", "property"):
                    rows.append(
                        {
                            "group_id": f"{concept}-{image}",
                            "image_id": f"{concept}-image-{image}",
                            "source": concept,
                            "modality": modality,
                            "readout": readout,
                            "correct": not (
                                fail_last_property
                                and image == n - 1
                                and readout == "property"
                            ),
                        }
                    )
    return rows


def test_capability_selection_is_deterministic_and_uses_both_readouts():
    first = select_capability_eligible_samples(
        _clean_records(fail_last_property=True),
        concepts=("bird", "cat"),
        modalities=("text", "image", "spoken_audio"),
        max_images_per_cell=4,
        min_images_per_cell=4,
        seed="frozen",
    )
    second = select_capability_eligible_samples(
        list(reversed(_clean_records(fail_last_property=True))),
        concepts=("bird", "cat"),
        modalities=("text", "image", "spoken_audio"),
        max_images_per_cell=4,
        min_images_per_cell=4,
        seed="frozen",
    )
    assert first == second
    assert first["all_cells_sufficient"] is True
    assert first["same_groups_in_every_modality"] is True
    assert first["swap_results_consulted"] is False
    assert all(row["n_eligible"] == 5 for row in first["cells"])
    assert all(row["n_selected"] == 4 for row in first["cells"])
    for concept in ("bird", "cat"):
        chosen = {
            tuple(first["selected_group_ids"][f"{concept}|{modality}"])
            for modality in ("text", "image", "spoken_audio")
        }
        assert len(chosen) == 1


def _v2_raw_records():
    rows = []
    conditions = (
        "swap_alpha1",
        "swap_alpha2",
        "zero",
        "random_alpha1",
        "random_alpha2",
        "unrelated_alpha1",
        "unrelated_alpha2",
        "position_descriptive",
    )
    for layer in (32, 35):
        for arm in ("intermediate", "answer"):
            for condition in conditions:
                for readout in ("identity", "property"):
                    for image in range(4):
                        primary = condition in {
                            "swap_alpha1",
                            "swap_alpha2",
                            "position_descriptive",
                        }
                        target_prediction = primary and (
                            (arm == "intermediate" and layer in {32, 35})
                            or (
                                arm == "answer"
                                and layer == 35
                                and readout == "property"
                            )
                        )
                        rows.append(
                            {
                                "start_layer": layer,
                                "arm": arm,
                                "condition": condition,
                                "modality": "text",
                                "source": "bird",
                                "target": "cat",
                                "readout": readout,
                                "image_id": f"bird-{image}",
                                "source_answer": (
                                    "bird" if readout == "identity" else "two"
                                ),
                                "target_answer": (
                                    "cat" if readout == "identity" else "four"
                                ),
                                "clean_prediction": (
                                    "bird" if readout == "identity" else "two"
                                ),
                                "prediction": (
                                    ("cat" if readout == "identity" else "four")
                                    if target_prediction
                                    else (
                                        "bird" if readout == "identity" else "two"
                                    )
                                ),
                                "target_margin_change": (
                                    2.0 if target_prediction else 0.0
                                ),
                            }
                        )
    return rows


def test_v2_verdict_uses_independent_layers_and_nonblocking_position_diagnostic():
    verdict = paper_onset_verdict_v2(
        summarize_cells(_v2_raw_records()),
        layers=(32, 35),
        directed_pairs=({"source": "bird", "target": "cat"},),
        modalities=("text",),
        thresholds=PaperSwapV2Thresholds(),
    )
    assert verdict["verdict"] == "PAPER_STYLE_EARLIER_INTERMEDIATE_GO"
    assert verdict["primary_verdict"] == "PAPER_STYLE_EARLIER_INTERMEDIATE"
    assert verdict["primary_onsets"] == {"intermediate": 32, "answer": 35}
    assert verdict["matched_pair_results"]["swap_alpha1"]["bird->cat"] == {
        "pair": "bird->cat",
        "intermediate_onset": 32,
        "answer_onset": 35,
        "classification": "EARLIER_INTERMEDIATE",
        "intermediate_passing_layers": [32, 35],
        "answer_passing_layers": [35],
    }
    assert verdict["cross_arm_direction_matching_required"] is True
    assert verdict["repeated_coordinate_exchange_used"] is False
    assert verdict["position_diagnostic_is_blocking"] is False
    l32_intermediate = next(
        row
        for row in verdict["direction_cells"]
        if row["condition"] == "swap_alpha1"
        and row["arm"] == "intermediate"
        and row["layer"] == 32
    )
    assert l32_intermediate["position_diagnostic"]["blocking"] is False
    assert all(l32_intermediate["clauses"].values())


def test_v2_alpha2_controls_are_intensity_matched():
    rows = _v2_raw_records()
    for row in rows:
        if (
            row["start_layer"] == 32
            and row["arm"] == "intermediate"
            and row["condition"] == "random_alpha2"
        ):
            row["target_margin_change"] = 3.0
    verdict = paper_onset_verdict_v2(
        summarize_cells(rows),
        layers=(32, 35),
        directed_pairs=({"source": "bird", "target": "cat"},),
        modalities=("text",),
    )
    alpha2_l32 = next(
        row
        for row in verdict["direction_cells"]
        if row["condition"] == "swap_alpha2"
        and row["arm"] == "intermediate"
        and row["layer"] == 32
    )
    assert alpha2_l32["passed"] is False
    assert "identity_beats_random_alpha2" in alpha2_l32["failed_clauses"]


def test_v2_reports_alpha2_matched_sensitivity_beside_primary_null():
    rows = _v2_raw_records()
    for row in rows:
        if row["condition"] == "swap_alpha1" and row["arm"] == "intermediate":
            row["prediction"] = row["source_answer"]
            row["target_margin_change"] = 0.0
    verdict = paper_onset_verdict_v2(
        summarize_cells(rows),
        layers=(32, 35),
        directed_pairs=({"source": "bird", "target": "cat"},),
        modalities=("text",),
    )
    assert verdict["verdict"] == (
        "PAPER_STYLE_INTERMEDIATE_NULL_WITH_ANSWER_POSITIVE"
    )
    assert verdict["primary_verdict"] == (
        "PAPER_STYLE_INTERMEDIATE_NULL_WITH_ANSWER_POSITIVE"
    )
    assert verdict["alpha2_sensitivity_verdict"] == (
        "PAPER_STYLE_ALPHA2_SENSITIVITY_EARLIER_INTERMEDIATE"
    )


def test_alpha2_confirmation_requires_same_direction_early_then_late():
    result = paper_onset_verdict_v2(
        summarize_cells(_v2_raw_records()),
        layers=(32, 35),
        directed_pairs=({"source": "bird", "target": "cat"},),
        modalities=("text",),
    )
    confirmation = paper_alpha2_confirmation_verdict(result)
    assert confirmation["verdict"] == (
        "PAPER_STYLE_ALPHA2_INDEPENDENT_CONFIRMATION_GO"
    )
    assert confirmation["alpha2_is_primary"] is True
    assert confirmation["alpha2_intermediate_onset"] == 32
    assert confirmation["alpha2_answer_onset"] == 35
    assert all(confirmation["clauses"].values())


def test_alpha2_confirmation_refuses_opposite_direction_stitching():
    result = paper_onset_verdict_v2(
        summarize_cells(_v2_raw_records()),
        layers=(32, 35),
        directed_pairs=({"source": "bird", "target": "cat"},),
        modalities=("text",),
    )
    result["matched_pair_results"]["swap_alpha2"] = {
        "cat->bird": result["matched_pair_results"]["swap_alpha2"].pop(
            "bird->cat"
        )
    }
    confirmation = paper_alpha2_confirmation_verdict(result)
    assert confirmation["verdict"] == (
        "PAPER_STYLE_ALPHA2_INDEPENDENT_CONFIRMATION_NO_GO"
    )
    assert "direction_is_predeclared" in confirmation["failed_clauses"]


def test_v2_never_compares_intermediate_and_answer_from_opposite_directions():
    bird_rows = _v2_raw_records()
    for row in bird_rows:
        if row["arm"] == "answer":
            row["prediction"] = row["source_answer"]
            row["target_margin_change"] = 0.0

    cat_rows = []
    for source_row in _v2_raw_records():
        row = dict(source_row)
        row["source"], row["target"] = "cat", "bird"
        row["image_id"] = str(row["image_id"]).replace("bird-", "cat-")
        row["source_answer"] = (
            "cat" if row["readout"] == "identity" else "four"
        )
        row["target_answer"] = (
            "bird" if row["readout"] == "identity" else "two"
        )
        row["clean_prediction"] = row["source_answer"]
        should_flip = (
            row["arm"] == "answer"
            and row["start_layer"] == 35
            and row["readout"] == "property"
            and row["condition"]
            in {"swap_alpha1", "swap_alpha2", "position_descriptive"}
        )
        row["prediction"] = (
            row["target_answer"] if should_flip else row["source_answer"]
        )
        row["target_margin_change"] = 2.0 if should_flip else 0.0
        cat_rows.append(row)

    verdict = paper_onset_verdict_v2(
        summarize_cells([*bird_rows, *cat_rows]),
        layers=(32, 35),
        directed_pairs=(
            {"source": "bird", "target": "cat"},
            {"source": "cat", "target": "bird"},
        ),
        modalities=("text",),
    )
    assert verdict["descriptive_unmatched_global_onsets"]["swap_alpha1"] == {
        "intermediate": 32,
        "answer": 35,
    }
    assert verdict["matched_pair_results"]["swap_alpha1"]["bird->cat"][
        "answer_onset"
    ] is None
    assert verdict["matched_pair_results"]["swap_alpha1"]["cat->bird"][
        "intermediate_onset"
    ] is None
    assert verdict["verdict"] != "PAPER_STYLE_EARLIER_INTERMEDIATE_GO"


def test_v2_cannot_stitch_different_directions_across_modalities():
    rows = []
    for source, target, source_identity, target_identity in (
        ("bird", "cat", "bird", "cat"),
        ("cat", "bird", "cat", "bird"),
    ):
        for modality in ("text", "image"):
            direction_should_pass = (source, modality) in {
                ("bird", "text"),
                ("cat", "image"),
            }
            for condition in (
                "swap_alpha1",
                "swap_alpha2",
                "zero",
                "random_alpha1",
                "random_alpha2",
                "unrelated_alpha1",
                "unrelated_alpha2",
                "position_descriptive",
            ):
                for readout in ("identity", "property"):
                    for image in range(4):
                        target_prediction = (
                            direction_should_pass
                            and condition
                            in {"swap_alpha1", "swap_alpha2", "position_descriptive"}
                        )
                        source_answer = (
                            source_identity
                            if readout == "identity"
                            else ("two" if source == "bird" else "four")
                        )
                        target_answer = (
                            target_identity
                            if readout == "identity"
                            else ("four" if target == "cat" else "two")
                        )
                        rows.append(
                            {
                                "start_layer": 32,
                                "arm": "intermediate",
                                "condition": condition,
                                "modality": modality,
                                "source": source,
                                "target": target,
                                "readout": readout,
                                "image_id": f"{source}-{image}",
                                "source_answer": source_answer,
                                "target_answer": target_answer,
                                "clean_prediction": source_answer,
                                "prediction": (
                                    target_answer if target_prediction else source_answer
                                ),
                                "target_margin_change": (
                                    2.0 if target_prediction else 0.0
                                ),
                            }
                        )
    verdict = paper_onset_verdict_v2(
        summarize_cells(rows),
        layers=(32,),
        directed_pairs=(
            {"source": "bird", "target": "cat"},
            {"source": "cat", "target": "bird"},
        ),
        modalities=("text", "image"),
    )
    row = next(
        item
        for item in verdict["layer_cells"]
        if item["condition"] == "swap_alpha1"
        and item["arm"] == "intermediate"
    )
    assert row["modality_pass"] == {"text": True, "image": True}
    assert row["same_direction_pair_pass"] == {
        "bird->cat": False,
        "cat->bird": False,
    }
    assert row["passed"] is False


def test_direction_matched_amendment_is_reporting_only_and_checksum_bound():
    cells = summarize_cells(_v2_raw_records())
    result = paper_onset_verdict_v2(
        cells,
        layers=(32, 35),
        directed_pairs=({"source": "bird", "target": "cat"},),
        modalities=("text",),
    )
    report = {
        **result,
        "schema": "mmpilot.paper_reasoning_swap_report.v2",
        "cells_full": cells,
        "directed_pairs": [{"source": "bird", "target": "cat"}],
        "capability_selection": {"modalities": ["text"]},
    }
    report["report_checksum"] = payload_checksum(report)
    amended = amend_paper_v2_report_direction_matching(
        report,
        expected_report_checksum=report["report_checksum"],
    )
    assert amended["scientific_units_recomputed"] == 0
    assert amended["model_loaded"] is False
    assert amended["original_report_checksum"] == report["report_checksum"]
    assert amended["matched_pair_results"]["swap_alpha1"]["bird->cat"][
        "classification"
    ] == "EARLIER_INTERMEDIATE"
    assert amended["primary_verdict"] == "PAPER_STYLE_EARLIER_INTERMEDIATE"
    assert amended["report_checksum"] == payload_checksum(
        {key: value for key, value in amended.items() if key != "report_checksum"}
    )

    tampered = dict(report)
    tampered["verdict"] = "TAMPERED"
    with pytest.raises(PaperSwapRefused, match="payload no longer matches"):
        amend_paper_v2_report_direction_matching(
            tampered,
            expected_report_checksum=report["report_checksum"],
        )
