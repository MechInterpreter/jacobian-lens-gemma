# SPDX-License-Identifier: Apache-2.0
"""Frozen scientific rules for the country-workspace benchmark."""

from __future__ import annotations

import pytest

from jlens.mmpilot.country_workspace import (
    DATASET_REVISION,
    DIRECTIONS,
    EVAL_COUNTRIES,
    FACTS,
    FIT_COUNTRIES,
    LAYERS,
    MODALITIES,
    N_CONFIRMATION_PER_COUNTRY,
    N_DEVELOPMENT_PER_COUNTRY,
    N_FIT_PER_COUNTRY,
    PATH_BANDS,
    PROPERTIES,
    CountryWorkspaceRefused,
    answer_matches,
    benchmark_spec,
    capability_report,
    causal_report,
    direct_answer_localization_report,
    freeze_confirmation_design,
    validate_media_plan,
)


def _media_rows() -> list[dict]:
    rows = []
    counter = 0
    for country in FIT_COUNTRIES:
        for _ in range(N_FIT_PER_COUNTRY):
            counter += 1
            rows.append(_media_row(counter, country, "fit"))
    for country in EVAL_COUNTRIES:
        for _ in range(N_DEVELOPMENT_PER_COUNTRY):
            counter += 1
            rows.append(_media_row(counter, country, "development"))
        for _ in range(N_CONFIRMATION_PER_COUNTRY):
            counter += 1
            rows.append(_media_row(counter, country, "confirmation"))
    return rows


def _media_row(index: int, country: str, split: str) -> dict:
    return {
        "unit_id": f"unit-{index}",
        "country": country,
        "source_split": "train",
        "source_index": index,
        "source_seed": str(index),
        "image_path": f"/images/{index}.png",
        "image_checksum": f"sha256:image-{index}",
        "audio_path": f"/audio/{index}.wav",
        "audio_checksum": f"sha256:audio-{index}",
        "speech_text": f"The country is {country}",
        "speech_voice": "en",
        "speech_speed": 140 + index,
        "speech_pitch": 40,
        "study_split": split,
        "ocr_text": "",
    }


def _capability_rows(success: bool = True) -> list[dict]:
    return [
        {
            "country": country,
            "property": property_name,
            "modality": modality,
            "success": success,
        }
        for country in EVAL_COUNTRIES
        for property_name in ("identity", *PROPERTIES)
        for modality in MODALITIES
        for _ in range(N_DEVELOPMENT_PER_COUNTRY)
    ]


def _localization_rows(*, passing_band: tuple[int, ...]) -> list[dict]:
    rows = []
    for property_name in PROPERTIES:
        for source, target in DIRECTIONS:
            for modality in MODALITIES:
                for band in PATH_BANDS:
                    for index in range(N_DEVELOPMENT_PER_COUNTRY):
                        rows.append(
                            {
                                "unit_id": f"{source}-{index}",
                                "source": source,
                                "target": target,
                                "direction": f"{source}->{target}",
                                "property": property_name,
                                "modality": modality,
                                "condition": "direct_answer",
                                "layers_patched": list(band),
                                "success": band == passing_band,
                                "integrity_pass": True,
                            }
                        )
    return rows


def _causal_rows(n: int, passing: set[str]) -> list[dict]:
    rows = []
    for source, target in DIRECTIONS:
        direction = f"{source}->{target}"
        for property_name in PROPERTIES:
            for modality in MODALITIES:
                for index in range(n):
                    for condition in ("exact", "zero", "random", "unrelated"):
                        rows.append(
                            {
                                "unit_id": f"{source}-{index}",
                                "direction": direction,
                                "property": property_name,
                                "modality": modality,
                                "condition": condition,
                                "success": condition == "exact" and direction in passing,
                                "integrity_pass": True,
                            }
                        )
    return rows


def test_dataset_counts_support_the_frozen_design() -> None:
    spec = benchmark_spec(dataset_revision="pinned")
    assert spec["fit_per_country"] == 3
    assert DATASET_REVISION == "1ea3cce246ab44f0fe8ecb526ad759ea11d28465"
    assert spec["development_per_country"] == 4
    assert spec["confirmation_per_country"] == 14
    assert spec["teacher_forcing"] is False
    assert spec["candidate_list"] is False
    assert len(DIRECTIONS) == 4
    assert len(PATH_BANDS) == 6


def test_facts_change_in_every_direction() -> None:
    for source, target in DIRECTIONS:
        for property_name in PROPERTIES:
            assert FACTS[source][property_name] != FACTS[target][property_name]


def test_answer_matcher_accepts_punctuation_but_not_explanatory_lead_in() -> None:
    assert answer_matches(" Paris.<turn|>", "Paris")
    assert answer_matches("Western Europe", "Europe") is False
    assert answer_matches("The answer is Paris", "Paris") is False


def test_media_plan_proves_disjoint_unique_population() -> None:
    report = validate_media_plan(_media_rows())
    assert report["passed"] is True
    assert not any(report["split_image_overlaps"].values())


def test_media_plan_refuses_ocr_leakage_and_duplicate_audio() -> None:
    rows = _media_rows()
    rows[0]["ocr_text"] = "Paris"
    rows[1]["audio_checksum"] = rows[2]["audio_checksum"]
    report = validate_media_plan(rows)
    assert report["passed"] is False
    assert report["ocr_hits"]
    assert "audio checksums are not unique across study units" in report["problems"]


def test_capability_requires_every_country_property_modality_cell() -> None:
    report = capability_report(_capability_rows())
    assert report["verdict"] == "COUNTRY_CAPABILITY_GENERALIZATION_GO"
    assert report["generalization_ready"] is True
    rows = _capability_rows()
    for row in rows:
        if row["country"] == "Japan" and row["property"] == "capital" and row["modality"] == "image":
            row["success"] = False
    amended = capability_report(rows)
    assert amended["verdict"] == "COUNTRY_CAPABILITY_GENERALIZATION_GO"
    assert "Japan->Egypt" not in amended["eligible_directions"]
    assert "Egypt->Japan" in amended["eligible_directions"]


def test_capability_failure_blocks_only_directions_with_that_source() -> None:
    rows = _capability_rows()
    for row in rows:
        if row["country"] == "Egypt" and row["modality"] == "image":
            row["success"] = False
    report = capability_report(rows)
    assert report["eligible_countries"] == ["France", "China", "Japan"]
    assert report["eligible_directions"] == [
        "France->China",
        "China->France",
        "Japan->Egypt",
    ]
    assert report["generalization_ready"] is True


def test_path_selection_reads_only_direct_answer_outcomes() -> None:
    passing = PATH_BANDS[2]
    report = direct_answer_localization_report(
        _localization_rows(passing_band=passing)
    )
    assert report["verdict"] == "COUNTRY_DIRECT_PATHS_GO"
    assert report["selection_used_exact_swap_outcomes"] is False
    assert all(
        choice["band"] == list(passing)
        for choice in report["selected_paths"].values()
    )
    bad = _localization_rows(passing_band=passing)
    bad[0]["condition"] = "exact"
    with pytest.raises(CountryWorkspaceRefused, match="non-direct-answer"):
        direct_answer_localization_report(bad)


def test_path_selection_accepts_only_capability_eligible_directions() -> None:
    passing = PATH_BANDS[1]
    eligible = {"France->China", "China->France", "Japan->Egypt"}
    rows = [
        row
        for row in _localization_rows(passing_band=passing)
        if row["direction"] in eligible
    ]
    report = direct_answer_localization_report(rows)
    assert report["verdict"] == "COUNTRY_DIRECT_PATHS_GO"
    assert set(report["eligible_directions"]) == eligible


def test_development_verdict_distinguishes_generalization_and_bidirectionality() -> None:
    passing = {"France->China", "Japan->Egypt"}
    report = causal_report(
        _causal_rows(N_DEVELOPMENT_PER_COUNTRY, passing),
        stage="development",
        expected_n=N_DEVELOPMENT_PER_COUNTRY,
    )
    assert report["verdict"] == "COUNTRY_DEVELOPMENT_GENERALIZED_GO"
    assert report["generalized_across_two_pairs"] is True
    assert report["bidirectional_on_at_least_one_pair"] is False


def test_confirmation_full_grid_passes_holm_against_worst_control() -> None:
    all_directions = {f"{source}->{target}" for source, target in DIRECTIONS}
    report = causal_report(
        _causal_rows(N_CONFIRMATION_PER_COUNTRY, all_directions),
        stage="confirmation",
        expected_n=N_CONFIRMATION_PER_COUNTRY,
        frozen_directions=sorted(all_directions),
    )
    assert report["verdict"] == "COUNTRY_CONFIRMATION_FULL_GRID_GO"
    assert len(report["paired_comparisons"]) == 24
    assert {row["control"] for row in report["paired_comparisons"]} == {
        "any_negative_control"
    }
    assert max(row["holm_adjusted_p"] for row in report["paired_comparisons"]) < 0.05


def test_confirmation_design_is_frozen_only_after_development_passes() -> None:
    media = validate_media_plan(_media_rows())
    capability = capability_report(_capability_rows())
    localization = direct_answer_localization_report(
        _localization_rows(passing_band=PATH_BANDS[0])
    )
    development = causal_report(
        _causal_rows(
            N_DEVELOPMENT_PER_COUNTRY,
            {"France->China", "Japan->Egypt"},
        ),
        stage="development",
        expected_n=N_DEVELOPMENT_PER_COUNTRY,
    )
    design = freeze_confirmation_design(
        protocol=benchmark_spec(dataset_revision="pinned"),
        media_validation=media,
        capability=capability,
        localization=localization,
        development=development,
    )
    assert design["directions"] == ["France->China", "Japan->Egypt"]
    assert design["frozen_before_confirmation_outputs"] is True
    development["passing_directions_both_properties"] = []
    development["generalized_across_two_pairs"] = False
    with pytest.raises(CountryWorkspaceRefused, match="two independent pairs"):
        freeze_confirmation_design(
            protocol=benchmark_spec(dataset_revision="pinned"),
            media_validation=media,
            capability=capability,
            localization=localization,
            development=development,
        )


def test_full_band_amendment_does_not_require_direct_answer_gate() -> None:
    media = validate_media_plan(_media_rows())
    capability = capability_report(_capability_rows())
    localization = direct_answer_localization_report(
        _localization_rows(passing_band=())
    )
    assert localization["verdict"] == "COUNTRY_DIRECT_PATHS_NO_GO"
    development = causal_report(
        _causal_rows(
            N_DEVELOPMENT_PER_COUNTRY,
            {"France->China", "Japan->Egypt"},
        ),
        stage="development",
        expected_n=N_DEVELOPMENT_PER_COUNTRY,
    )
    design = freeze_confirmation_design(
        protocol=benchmark_spec(dataset_revision="pinned"),
        media_validation=media,
        capability=capability,
        localization=localization,
        development=development,
        predeclared_band=LAYERS,
    )
    assert design["localization_role"] == "diagnostic_not_a_necessary_gate"
    assert all(
        choice["band"] == list(LAYERS)
        for choice in design["selected_paths"].values()
    )
