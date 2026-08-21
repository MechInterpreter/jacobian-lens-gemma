import pytest
import torch

from jlens.lens import JacobianLens
from jlens.mmpilot.loading_first import (
    LoadingFirstRefused,
    combine_disjoint_layer_lenses,
    select_loading_instrument,
)


def _rows(instrument, advantages, *, contaminated=False):
    return [
        {
            "instrument": instrument,
            "sample_id": task,
            "layer": layer,
            "position_class": "final_prompt_token",
            "source_cosine": advantage + 0.2,
            "source_advantage": advantage,
            "causal_result_consulted": contaminated,
        }
        for task in ("a", "b")
        for layer, advantage in advantages.items()
    ]


def test_selects_longest_clean_band_before_score():
    result = select_loading_instrument(
        {
            "j": _rows("j", {33: 0.4, 34: -0.1, 35: 0.8}),
            "r": _rows("r", {33: 0.1, 34: 0.2, 35: 0.3}),
        },
        tasks=("a", "b"),
        layers=(33, 34, 35),
    )
    assert result["verdict"] == "LOADING_FIRST_INSTRUMENT_GO"
    assert result["selected_instrument"] == "r"
    assert result["selected_band"] == [33, 34, 35]
    assert result["causal_result_consulted"] is False


def test_no_positive_complete_band_is_no_go():
    result = select_loading_instrument(
        {"j": _rows("j", {33: -0.1})}, tasks=("a", "b"), layers=(33,)
    )
    assert result["verdict"] == "LOADING_FIRST_INSTRUMENT_NO_GO"
    assert result["selected_instrument"] is None


def test_refuses_causal_contamination():
    with pytest.raises(LoadingFirstRefused, match="causal-contaminated"):
        select_loading_instrument(
            {"j": _rows("j", {33: 0.1}, contaminated=True)},
            tasks=("a", "b"),
            layers=(33,),
        )


def _lens(layers, *, n_prompts=99, d_model=3):
    return JacobianLens(
        {layer: torch.eye(d_model) * layer for layer in layers},
        n_prompts=n_prompts,
        d_model=d_model,
    )


def test_combines_disjoint_identically_fitted_shards_into_one_band():
    joined = combine_disjoint_layer_lenses(
        (_lens((27, 28)), _lens((29, 30))),
        expected_layers=(27, 28, 29, 30),
    )
    assert joined.source_layers == [27, 28, 29, 30]
    assert joined.n_prompts == 99
    assert torch.equal(joined.jacobians[30], torch.eye(3) * 30)


def test_combined_band_refuses_overlap_gap_or_fit_count_mismatch():
    with pytest.raises(LoadingFirstRefused, match="overlap"):
        combine_disjoint_layer_lenses(
            (_lens((27, 28)), _lens((28, 29))),
            expected_layers=(27, 28, 29),
        )
    with pytest.raises(LoadingFirstRefused, match="exactly cover"):
        combine_disjoint_layer_lenses(
            (_lens((27, 28)), _lens((30,))),
            expected_layers=(27, 28, 29, 30),
        )
    with pytest.raises(LoadingFirstRefused, match="fitted-prompt count"):
        combine_disjoint_layer_lenses(
            (_lens((27, 28)), _lens((29, 30), n_prompts=98)),
            expected_layers=(27, 28, 29, 30),
        )
