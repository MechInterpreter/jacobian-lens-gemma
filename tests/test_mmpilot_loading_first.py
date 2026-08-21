import pytest

from jlens.mmpilot.loading_first import LoadingFirstRefused, select_loading_instrument


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
