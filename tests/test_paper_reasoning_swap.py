from __future__ import annotations

import pytest

from jlens.mmpilot.evidence import EvidenceConfig
from jlens.mmpilot.paper_reasoning_swap import (
    PaperSwapRefused,
    PaperSwapThresholds,
    PaperSwapV2Thresholds,
    hidden_animal_population,
    independent_layer_record,
    paper_onset_verdict,
    paper_onset_verdict_v2,
    sampled_suffix_bands,
    select_capability_eligible_samples,
    summarize_cells,
)


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
    assert verdict["primary_onsets"] == {"intermediate": 32, "answer": 35}
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
