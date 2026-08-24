"""Guards for the outcome-informed animal-sound prompt screen."""

from __future__ import annotations

import pytest

from jlens.mmpilot.multimodal_followup import (
    ANIMAL_SOUND_PROMPT_CANDIDATES,
    MultimodalFollowupRefused,
    property_prompt,
    property_prompt_screen_verdict,
)


def _rows(rates: dict[str, dict[str, dict[str, float]]], n: int = 8) -> list[dict]:
    rows = []
    for prompt in ANIMAL_SOUND_PROMPT_CANDIDATES:
        prompt_id = prompt["prompt_id"]
        for concept in ("cat", "cow"):
            for modality in ("text", "image", "spoken_audio"):
                successes = round(rates[prompt_id][concept][modality] * n)
                for index in range(n):
                    rows.append(
                        {
                            "prompt_id": prompt_id,
                            "concept": concept,
                            "modality": modality,
                            "group_id": f"{concept}-{index}",
                            "pass": index < successes,
                            "generated": "meow" if concept == "cat" else "moo",
                        }
                    )
    return rows


def _uniform(value: float) -> dict:
    return {
        row["prompt_id"]: {
            concept: {modality: value for modality in ("text", "image", "spoken_audio")}
            for concept in ("cat", "cow")
        }
        for row in ANIMAL_SOUND_PROMPT_CANDIDATES
    }


def test_prompt_variants_never_put_caption_in_image_or_audio() -> None:
    marker = "SECRET_TRANSCRIPT_MARKER"
    for candidate in ANIMAL_SOUND_PROMPT_CANDIDATES:
        prompt_id = candidate["prompt_id"]
        assert marker in property_prompt(prompt_id, "text", marker)
        assert marker not in property_prompt(prompt_id, "image", marker)
        assert marker not in property_prompt(prompt_id, "spoken_audio", marker)


def test_screen_selects_best_worst_cell_among_prompts_that_pass() -> None:
    rates = _uniform(0.75)
    rates["baseline_v1"]["cat"]["text"] = 0.5
    rates["identity_explicit_v1"] = {
        concept: {modality: 0.875 for modality in ("text", "image", "spoken_audio")}
        for concept in ("cat", "cow")
    }
    rates["knowledge_cloze_v1"] = {
        concept: {modality: 0.75 for modality in ("text", "image", "spoken_audio")}
        for concept in ("cat", "cow")
    }
    report = property_prompt_screen_verdict(_rows(rates), expected_per_cell=8)
    assert report["verdict"] == "PROPERTY_PROMPT_SCREEN_GO"
    assert report["selected_prompt_id"] == "identity_explicit_v1"
    assert report["causal_spending_licensed"] is False
    assert report["fresh_capability_revalidation_required"] is True


def test_screen_no_go_does_not_select_or_license_anything() -> None:
    rates = _uniform(0.5)
    report = property_prompt_screen_verdict(_rows(rates), expected_per_cell=8)
    assert report["verdict"] == "PROPERTY_PROMPT_SCREEN_NO_GO"
    assert report["selected_prompt_id"] is None
    assert report["causal_spending_licensed"] is False


def test_screen_refuses_incomplete_or_undeclared_cells() -> None:
    rows = _rows(_uniform(0.75))
    with pytest.raises(MultimodalFollowupRefused, match="incomplete"):
        property_prompt_screen_verdict(rows[:-1], expected_per_cell=8)
    rows[0]["prompt_id"] = "post_hoc_prompt"
    with pytest.raises(MultimodalFollowupRefused, match="undeclared"):
        property_prompt_screen_verdict(rows, expected_per_cell=8)


def test_screen_tie_break_is_declaration_order() -> None:
    report = property_prompt_screen_verdict(
        _rows(_uniform(0.75)), expected_per_cell=8
    )
    assert report["selected_prompt_id"] == "baseline_v1"

