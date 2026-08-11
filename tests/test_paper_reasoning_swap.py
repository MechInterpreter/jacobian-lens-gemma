from __future__ import annotations

import pytest

from jlens.mmpilot.evidence import EvidenceConfig
from jlens.mmpilot.paper_reasoning_swap import (
    PaperSwapRefused,
    PaperSwapThresholds,
    hidden_animal_population,
    paper_onset_verdict,
    sampled_suffix_bands,
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
