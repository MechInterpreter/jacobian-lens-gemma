# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the no-refit country causal-site diagnostic."""

from __future__ import annotations

import pytest
import torch

from jlens.mmpilot.backend import BuiltInputs
from jlens.mmpilot.country_activation_patch import (
    CountryActivationPatchRefused,
    activation_patch_band,
    causal_site_screen_report,
    localized_development_report,
    patch_position,
    restricted_swap_report,
    single_position_inputs,
    state_validated_selection,
)


def _inputs(
    ids: list[int], *, modality: str = "text", span: list[int] | None = None
) -> BuiltInputs:
    return BuiltInputs(
        tensors={"input_ids": torch.tensor([ids])},
        prompt_len=len(ids),
        modality=modality,
        modality_token_range=span,
    )


def test_patch_positions_are_semantic_and_fail_closed() -> None:
    text = _inputs([10, 20, 30, 40])
    assert patch_position(text, "final_prompt_token") == 3
    assert patch_position(text, "evidence_endpoint", country_token_id=20) == 1

    image = _inputs([1, 2, 3, 4, 5], modality="image", span=[1, 4])
    assert patch_position(image, "evidence_endpoint") == 3

    with pytest.raises(CountryActivationPatchRefused, match="occurs 2 times"):
        patch_position(_inputs([20, 1, 20]), "evidence_endpoint", country_token_id=20)
    with pytest.raises(CountryActivationPatchRefused, match="unknown patch site"):
        patch_position(text, "guessed_position")


def test_single_position_view_changes_only_the_declared_span() -> None:
    inputs = _inputs([1, 2, 3, 4], modality="image", span=[1, 3])
    narrowed = single_position_inputs(inputs, 2)
    assert narrowed.modality_token_range == [2, 3]
    assert narrowed.prompt_len == inputs.prompt_len
    assert narrowed.tensors is inputs.tensors


def test_activation_patch_replaces_one_position_across_contiguous_band() -> None:
    blocks = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])
    hidden = torch.zeros((1, 3, 2), dtype=torch.float32)
    donors = {
        0: torch.tensor([2.0, 3.0]),
        1: torch.tensor([4.0, 5.0]),
    }
    with activation_patch_band(
        blocks, donors, source_position=1, prompt_len=3
    ) as stats:
        output = hidden
        for block in blocks:
            output = block(output)
    assert output[0, 1].tolist() == [4.0, 5.0]
    assert output[0, 0].tolist() == [0.0, 0.0]
    assert all(row["n_forward_passes"] == 1 for row in stats.values())
    assert all(row["all_finite"] for row in stats.values())
    assert all(not block._forward_hooks for block in blocks)


def _screen_rows(*, failing: tuple[str, str, str] | None = None) -> list[dict]:
    rows = []
    for band in ((16, 17), (20, 21, 22)):
        for site in ("evidence_endpoint", "final_prompt_token"):
            for property_name in ("capital", "continent"):
                for modality in ("text", "image", "spoken_audio"):
                    for condition in (
                        "target_state",
                        "self_state",
                        "unrelated_state",
                        "direct_answer",
                    ):
                        success = condition in ("target_state", "direct_answer")
                        if failing == (site, property_name, modality):
                            success = False
                        rows.append(
                            {
                                "layers_patched": list(band),
                                "site": site,
                                "property": property_name,
                                "modality": modality,
                                "condition": condition,
                                "success": success,
                                "integrity_pass": True,
                            }
                        )
    return rows


def test_screen_selects_shortest_control_validated_path() -> None:
    report = causal_site_screen_report(
        _screen_rows(),
        bands=((16, 17), (20, 21, 22)),
        expected_n=1,
        properties=("capital", "continent"),
        modalities=("text", "image", "spoken_audio"),
    )
    assert report["verdict"] == "COUNTRY_CAUSAL_SITE_SCREEN_GO"
    assert report["selected"] == {
        "path_id": "L16-17:evidence_endpoint",
        "band": [16, 17],
        "site": "evidence_endpoint",
    }
    assert report["selection_used_coordinate_swap_outcomes"] is False
    assert report["selection_used_fresh_confirmation"] is False


def test_screen_refuses_a_path_when_positive_control_fails() -> None:
    rows = _screen_rows()
    for row in rows:
        if row["condition"] == "target_state":
            row["success"] = False
    report = causal_site_screen_report(
        rows,
        bands=((16, 17), (20, 21, 22)),
        expected_n=1,
        properties=("capital", "continent"),
        modalities=("text", "image", "spoken_audio"),
    )
    assert report["verdict"] == "COUNTRY_CAUSAL_SITE_SCREEN_NO_GO"
    assert report["selected"] is None


def test_state_selection_can_preserve_original_direct_answer_no_go() -> None:
    rows = _screen_rows()
    for row in rows:
        if row["condition"] == "direct_answer":
            row["success"] = False
    screen = causal_site_screen_report(
        rows,
        bands=((16, 17), (20, 21, 22)),
        expected_n=1,
        properties=("capital", "continent"),
        modalities=("text", "image", "spoken_audio"),
    )
    assert screen["verdict"] == "COUNTRY_CAUSAL_SITE_SCREEN_NO_GO"
    selection = state_validated_selection(screen)
    assert selection["verdict"] == "COUNTRY_STATE_VALIDATED_PATH_GO"
    assert selection["selected"]["band"] == [16, 17]
    assert selection["selection_used_coordinate_swap_outcomes"] is False
    assert selection["selection_used_direct_answer_outcomes"] is False
    assert selection["source_screen_verdict_unchanged"] == screen["verdict"]


def _restricted_rows(exact_success: bool) -> list[dict]:
    return [
        {
            "property": property_name,
            "modality": modality,
            "condition": condition,
            "success": exact_success and condition == "exact",
            "integrity_pass": True,
        }
        for property_name in ("capital", "continent")
        for modality in ("text", "image", "spoken_audio")
        for _ in range(3)
        for condition in ("exact", "zero", "random", "unrelated")
    ]


@pytest.mark.parametrize(
    ("exact_success", "verdict"),
    [
        (True, "COUNTRY_RESTRICTED_SWAP_DEVELOPMENT_GO"),
        (False, "COUNTRY_RESTRICTED_SWAP_DEVELOPMENT_NO_GO"),
    ],
)
def test_restricted_swap_report_requires_every_modality_and_property(
    exact_success: bool, verdict: str
) -> None:
    report = restricted_swap_report(
        _restricted_rows(exact_success),
        expected_n=3,
        properties=("capital", "continent"),
        modalities=("text", "image", "spoken_audio"),
        band=(20, 21),
        site="evidence_endpoint",
    )
    assert report["verdict"] == verdict
    assert report["fresh_confirmation_opened"] is False


def test_localized_report_keeps_full_state_and_jlens_claims_separate() -> None:
    state_rows = [
        {
            "property": property_name,
            "modality": modality,
            "condition": condition,
            "success": condition == "target_state",
            "integrity_pass": True,
        }
        for property_name in ("capital", "continent")
        for modality in ("text", "image", "spoken_audio")
        for _ in range(3)
        for condition in ("target_state", "self_state", "unrelated_state")
    ]
    report = localized_development_report(
        state_rows,
        _restricted_rows(False),
        expected_n=3,
        properties=("capital", "continent"),
        modalities=("text", "image", "spoken_audio"),
        selection={"band": [24, 25], "site": "final_prompt_token"},
    )
    assert report["verdict"] == "COUNTRY_LOCALIZED_DEVELOPMENT_STATE_ONLY_GO"
    assert report["full_state_arm"]["passed"] is True
    assert (
        report["j_lens_coordinate_arm"]["verdict"]
        == "COUNTRY_RESTRICTED_SWAP_DEVELOPMENT_NO_GO"
    )
    assert report["fresh_confirmation_opened"] is False
    assert report["fitting_performed"] is False
