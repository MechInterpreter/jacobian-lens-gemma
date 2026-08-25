from __future__ import annotations

import copy

import pytest

from jlens.mmpilot.catdog_localization import (
    CATDOG_CONTROL_BAND,
    CATDOG_PATH_BANDS,
    CatDogLocalizationRefused,
    applied_position_rule,
    frozen_grid_record,
    summarize_path_localization,
    verify_inconclusive_source_report,
)
from jlens.mmpilot.store import payload_checksum

REVISION = "model-revision"
LENS = "sha256:lens"


def _source_report() -> dict:
    groups = [f"g_{index}" for index in range(8)]
    body = {
        "verdict": "NEW_PROPERTY_DEVELOPMENT_NO_GO",
        "scientific_config": {
            "model_revision": REVISION,
            "model_dtype": "float32",
            "lens_checksum": LENS,
            "direction": ["cat", "dog"],
            "alpha": 1.0,
            "layers": list(range(16, 41)),
            "positions": "every original prompt position",
            "prompt_id": "identity_explicit_v1",
            "outcome_informed_stage_design": False,
            "is_confirmation": False,
        },
        "directions": [
            {
                "direction": "cat->dog",
                "instrument_state": "INCONCLUSIVE",
                "direct_answer_positive_control": {
                    "passed": False,
                    "by_modality": {
                        modality: {"n": 8, "successes": 0}
                        for modality in ("text", "image", "spoken_audio")
                    },
                },
            }
        ],
        "recruitment": {
            "complete": True,
            "selected": {
                "cat": [{"group_id": group} for group in groups],
                "dog": [],
            },
        },
        "rows": [
            {
                "condition": "direct_answer",
                "group_id": group,
                "modality": modality,
            }
            for group in groups
            for modality in ("text", "image", "spoken_audio")
        ],
    }
    return {**body, "report_checksum": payload_checksum(body)}


def test_source_report_is_checksum_and_protocol_pinned() -> None:
    report = _source_report()
    result = verify_inconclusive_source_report(
        report,
        expected_checksum=report["report_checksum"],
        expected_model_revision=REVISION,
        expected_lens_checksum=LENS,
    )
    assert result["verified"] is True
    assert result["group_ids"] == [f"g_{index}" for index in range(8)]


def test_source_report_refuses_a_changed_result() -> None:
    report = _source_report()
    changed = copy.deepcopy(report)
    changed["directions"][0]["direct_answer_positive_control"]["by_modality"][
        "image"
    ]["successes"] = 1
    changed["report_checksum"] = payload_checksum(
        {key: value for key, value in changed.items() if key != "report_checksum"}
    )
    with pytest.raises(CatDogLocalizationRefused, match="frozen 0/8"):
        verify_inconclusive_source_report(
            changed,
            expected_checksum=changed["report_checksum"],
            expected_model_revision=REVISION,
            expected_lens_checksum=LENS,
        )


def _rows(*, winning_band=tuple(range(20, 28)), winning_rule="all_prompt_positions"):
    grid = frozen_grid_record()
    rows = []
    for band in grid["bands"]:
        for rule in grid["position_rules"]:
            for modality in ("text", "image", "spoken_audio"):
                # A cell the policy leaves undefined is never run, so it
                # never produces a row either.
                if applied_position_rule(rule, modality) is None:
                    continue
                for index in range(8):
                    rows.append(
                        {
                            "condition": "direct_answer",
                            "group_id": f"g_{index}",
                            "modality": modality,
                            "layers_patched": band,
                            "position_rule": rule,
                            "success": (
                                tuple(band) == tuple(winning_band)
                                and rule == winning_rule
                                and index < 6
                            ),
                            "all_hooks_fired": True,
                            "all_finite": True,
                            "all_model_dtype_realizations_converged": True,
                            "max_relative_cumulative_band_displacement_match_error": 1e-6,
                            "teacher_forcing_used": False,
                            "candidate_list_supplied": False,
                        }
                    )
    return rows


def test_localization_selects_from_direct_answer_only() -> None:
    grid = frozen_grid_record()
    report = summarize_path_localization(
        _rows(),
        source_report_checksum="sha256:source",
        grid=grid,
        expected_group_ids=[f"g_{index}" for index in range(8)],
    )
    assert report["verdict"] == "CATDOG_DIRECT_ANSWER_PATH_LOCALIZATION_GO"
    assert report["selected_path"] == {
        "band": list(range(20, 28)),
        "position_rule": "all_prompt_positions",
        "minimum_modality_rate": 0.75,
        "pooled_successes": 18,
        "pooled_n": 24,
    }
    assert report["control_band_reproduces_source_null"] is True
    assert report["selection_uses_exact_exchange_outcomes"] is False
    assert report["can_establish_catdog_causal_transfer"] is False


def test_localization_refuses_exact_exchange_rows() -> None:
    rows = _rows()
    rows[0] = {**rows[0], "condition": "exact"}
    with pytest.raises(CatDogLocalizationRefused, match="direct-answer rows only"):
        summarize_path_localization(
            rows,
            source_report_checksum="sha256:source",
            grid=frozen_grid_record(),
            expected_group_ids=[f"g_{index}" for index in range(8)],
        )


def test_incomplete_or_integrity_failed_cells_do_not_pass() -> None:
    rows = _rows()
    for row in rows:
        if row["success"]:
            row["all_model_dtype_realizations_converged"] = False
    report = summarize_path_localization(
        rows,
        source_report_checksum="sha256:source",
        grid=frozen_grid_record(),
        expected_group_ids=[f"g_{index}" for index in range(8)],
    )
    assert report["verdict"] == "CATDOG_DIRECT_ANSWER_PATH_LOCALIZATION_NO_GO"
    assert report["selected_path"] is None


# ---------------------------------------- the sliding grid and its exclusions


def test_the_sliding_windows_cover_every_layer_in_the_validated_band() -> None:
    covered: set[int] = set()
    for band in CATDOG_PATH_BANDS:
        covered.update(band)
    assert covered == set(range(16, 41))


def test_every_window_is_eight_layers_except_the_full_band_control() -> None:
    for band in CATDOG_PATH_BANDS:
        if tuple(band) == CATDOG_CONTROL_BAND:
            assert len(band) == 25
        else:
            assert len(band) == 8


def test_modality_evidence_is_undefined_for_text_and_never_falls_back() -> None:
    """The fallback silently re-ran all_prompt_positions under another name."""
    assert applied_position_rule("modality_evidence", "text") is None
    assert applied_position_rule("modality_evidence", "image") == "evidence_span_only"
    assert (
        applied_position_rule("modality_evidence", "spoken_audio")
        == "evidence_span_only"
    )
    for modality in ("text", "image", "spoken_audio"):
        assert (
            applied_position_rule("all_prompt_positions", modality)
            == "all_prompt_positions"
        )


def test_a_path_with_an_undefined_modality_is_never_selectable() -> None:
    grid = frozen_grid_record()
    for path in grid["paths"]:
        if path["position_rule"] == "modality_evidence":
            assert path["selectable"] is False
            assert path["applicable_modalities"] == ["image", "spoken_audio"]
        else:
            assert path["selectable"] is True
    assert grid["n_paths"] == len(CATDOG_PATH_BANDS) * 3
    assert grid["n_selectable_paths"] == len(CATDOG_PATH_BANDS) * 2


def test_modality_evidence_cannot_win_even_at_perfect_defined_rates() -> None:
    report = summarize_path_localization(
        _rows(winning_band=tuple(range(24, 32)), winning_rule="modality_evidence"),
        source_report_checksum="sha256:source",
        grid=frozen_grid_record(),
        expected_group_ids=[f"g_{index}" for index in range(8)],
    )
    assert report["verdict"] == "CATDOG_DIRECT_ANSWER_PATH_LOCALIZATION_NO_GO"
    assert report["selected_path"] is None
    # but the defined cells are still reported, with the substitution visible
    candidate = next(
        row for row in report["candidates"]
        if row["band"] == list(range(24, 32))
        and row["position_rule"] == "modality_evidence"
    )
    assert candidate["selectable"] is False
    cells = {cell["modality"]: cell for cell in candidate["cells"]}
    assert cells["text"]["applicable"] is False
    assert cells["text"]["applied_position_rule"] is None
    assert cells["image"]["applied_position_rule"] == "evidence_span_only"
    assert cells["image"]["success_rate"] == 0.75


def test_a_control_band_that_disagrees_with_the_source_blocks_selection() -> None:
    """The full band must reproduce the source's 0/8 or nothing is trusted."""
    rows = _rows()
    for row in rows:
        if tuple(row["layers_patched"]) == CATDOG_CONTROL_BAND and (
            row["position_rule"] == "all_prompt_positions"
        ):
            row["success"] = True
    report = summarize_path_localization(
        rows,
        source_report_checksum="sha256:source",
        grid=frozen_grid_record(),
        expected_group_ids=[f"g_{index}" for index in range(8)],
    )
    assert report["verdict"] == "CATDOG_DIRECT_ANSWER_PATH_LOCALIZATION_CONTROL_DISAGREES"
    assert report["control_band_reproduces_source_null"] is False
    assert report["selected_path"] is None


def test_the_report_discloses_its_own_multiplicity() -> None:
    report = summarize_path_localization(
        _rows(),
        source_report_checksum="sha256:source",
        grid=frozen_grid_record(),
        expected_group_ids=[f"g_{index}" for index in range(8)],
    )
    assert report["n_paths_searched"] == len(CATDOG_PATH_BANDS) * 3
    assert report["n_selectable_paths"] == len(CATDOG_PATH_BANDS) * 2
    assert "n_paths_searched" in report["multiplicity_disclosure"]
    assert report["scientific_grade"] == "instrument_development_only"
