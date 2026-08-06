# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""The three-modality rubric: budgets, six pairs, and the five verdicts.

Every test here is about a decision the study is allowed to make. The ones that
matter most are the refusals: a GO that could be earned from text <-> image
alone, from a single photograph, from a global disruption, or from a control
that beats the effect, would be a GO about something other than spoken audio.
"""

import pytest

from jlens.mmpilot.report import FAIL, NOT_EVALUATED, PASS
from jlens.mmpilot.tri_modal import (
    ALL_PAIRS,
    AUDIO_CAPABILITY_GO,
    AUDIO_CAPABILITY_NO_GO,
    AUDIO_PAIRS,
    REPLICATION_PAIRS,
    THREE_MODALITY_GO,
    THREE_MODALITY_NO_GO,
    THREE_MODALITY_WEAK_GO,
    TRANSFER_NOT_EVALUATED,
    TRANSFER_SUPPORTED,
    TRANSFER_UNSUPPORTED,
    TRANSFER_WEAK,
    TriModalThresholds,
    audio_capability_verdict,
    causal_transfer_verdict,
    estimate_stage_passes,
    evaluate_causal_cells,
    format_stage_budget,
    format_total_budget,
    overall_verdict,
    render_report,
    replication_verdict,
    representational_rows,
    representational_transfer_verdict,
)

MODALITIES = ("text", "image", "spoken_audio")
THRESHOLDS = TriModalThresholds(
    required_positive_images_per_cell=2, required_negative_images_per_cell=2
)


# ------------------------------------------------------------------- pairs


def test_the_six_directions_are_exactly_the_ordered_cross_modal_pairs():
    assert len(ALL_PAIRS) == 6
    assert set(ALL_PAIRS) == {
        f"{a}->{b}" for a in MODALITIES for b in MODALITIES if a != b
    }
    assert set(AUDIO_PAIRS) == {p for p in ALL_PAIRS if "spoken_audio" in p}
    assert len(AUDIO_PAIRS) == 4
    assert set(REPLICATION_PAIRS) == {"text->image", "image->text"}
    assert not set(AUDIO_PAIRS) & set(REPLICATION_PAIRS)


# ------------------------------------------------------------------ budget


def _budget_kwargs(**overrides):
    kwargs = dict(
        n_concepts=6,
        n_focal_concepts=3,
        modalities=MODALITIES,
        layers=(35, 38, 40),
        n_total_groups=112,
        n_capability_groups=48,
        n_targets_per_cell=16,
        alphas=(0.0, 0.25, 0.5, 1.0),
        d_model=2560,
    )
    kwargs.update(overrides)
    return kwargs


def test_stage_a_spends_capability_and_activation_and_no_causal_passes():
    budget = estimate_stage_passes(stage="A", causal_layers=(), **_budget_kwargs())
    assert budget.capability_passes == 48 * 3 * 2 * 6
    assert budget.activation_passes == 112 * 3
    assert budget.causal_clean_passes == 0
    assert budget.causal_intervention_passes == 0
    assert budget.n_causal_cells == 0
    assert budget.estimated_units["activation"] == 112 * 3 * 3


def test_stage_b_is_off_diagonal_only_and_reuses_stage_a_s_units():
    budget = estimate_stage_passes(
        stage="B", causal_layers=(35,), **_budget_kwargs()
    )
    # 3 focal x 3 source modalities x 2 target modalities = 18 cells, one layer.
    assert budget.n_causal_cells == 18
    assert budget.n_conditions_per_target == 1 + 3 * 4
    assert budget.causal_clean_passes == 18 * 16 * 6
    assert budget.causal_intervention_passes == 18 * 16 * 13 * 6
    assert budget.capability_passes == 0
    assert budget.activation_passes == 0


def test_stage_c_is_exactly_stage_b_repeated_at_the_remaining_layers():
    one = estimate_stage_passes(stage="B", causal_layers=(35,), **_budget_kwargs())
    two = estimate_stage_passes(
        stage="C", causal_layers=(38, 40), **_budget_kwargs()
    )
    assert two.n_causal_cells == 2 * one.n_causal_cells
    assert two.total_passes == 2 * one.total_passes


def test_the_budget_blocks_print_the_numbers_a_confirmation_is_made_against():
    budgets = [
        estimate_stage_passes(stage="A", causal_layers=(), **_budget_kwargs()),
        estimate_stage_passes(stage="B", causal_layers=(35,), **_budget_kwargs()),
        estimate_stage_passes(stage="C", causal_layers=(38, 40), **_budget_kwargs()),
    ]
    text = "\n".join(format_stage_budget(b) for b in budgets)
    assert "TOTAL model forward passes" in text
    assert "estimated wall clock" in text
    assert "estimated Drive footprint" in text
    assert "estimated units:" in text
    total = format_total_budget(budgets)
    assert "WHOLE-STUDY BUDGET" in total
    assert f"{sum(b.total_passes for b in budgets):,}" in total


def test_an_unknown_stage_is_refused():
    with pytest.raises(ValueError, match="unknown stage"):
        estimate_stage_passes(stage="Z", causal_layers=(), **_budget_kwargs())


# -------------------------------------------------------------- verdict A


def _capability(passing_all_three, *, selected, audio_only_failures=()):
    per_concept = {}
    for concept in selected:
        entry = {}
        for modality in MODALITIES:
            passed = concept in passing_all_three or (
                modality != "spoken_audio" and concept in audio_only_failures
            )
            # Counts and ``passed`` agree on purpose: the admissibility rule
            # re-derives accuracy from the counts rather than trusting a flag,
            # and 5/8 = 62.5% is the observed spoken-audio failure it must catch.
            n_correct = 8 if passed else 5
            entry[modality] = {
                "n": 8,
                "n_correct": n_correct,
                "accuracy": n_correct / 8,
                "passed": passed,
                "median_target_margin": 1.0,
                "min_target_margin": 0.5,
            }
        per_concept[concept] = entry
    return {
        "threshold": 0.7,
        "modalities_evaluated": list(MODALITIES),
        "per_concept": per_concept,
        "retained_concepts": sorted(passing_all_three),
        "text_image_retained_concepts": sorted(
            set(passing_all_three) | set(audio_only_failures)
        ),
        "n_records": 8 * len(selected),
    }


def test_two_concepts_readable_in_all_three_channels_is_a_capability_go():
    selected = ["bus", "cat", "clock"]
    verdict = audio_capability_verdict(
        _capability({"bus", "cat"}, selected=selected),
        selected_concepts=selected,
        modalities=MODALITIES,
        thresholds=THRESHOLDS,
    )
    assert verdict["verdict"] == AUDIO_CAPABILITY_GO
    assert verdict["concepts_passing_all_three"] == ["bus", "cat"]
    assert "not evidence that J-space coordinates transfer" in verdict["rationale"]


def test_one_concept_in_all_three_is_a_capability_no_go():
    selected = ["bus", "cat", "clock"]
    verdict = audio_capability_verdict(
        _capability({"bus"}, selected=selected, audio_only_failures={"cat", "clock"}),
        selected_concepts=selected,
        modalities=MODALITIES,
        thresholds=THRESHOLDS,
    )
    assert verdict["verdict"] == AUDIO_CAPABILITY_NO_GO
    assert verdict["concepts_passing_text_and_image"] == ["bus", "cat", "clock"]
    assert "expensive stages are skipped" in verdict["rationale"]


def test_a_blocked_audio_channel_is_never_a_capability_go():
    selected = ["bus", "cat"]
    verdict = audio_capability_verdict(
        _capability({"bus", "cat"}, selected=selected),
        selected_concepts=selected,
        modalities=("text", "image"),
        thresholds=THRESHOLDS,
    )
    assert verdict["verdict"] == AUDIO_CAPABILITY_NO_GO
    assert verdict["spoken_audio_available"] is False


# -------------------------------------------------------------- verdict B


def _pair_entry(*, top1, shuffled_p95, jspace_gap=0.4, raw_gap=0.1, queries=12):
    return {
        "n_sources": queries,
        "n_targets": queries,
        "n_distinct_source_images": queries,
        "n_distinct_target_images": queries,
        "exclusions": {
            "eligible_targets": {"min": 8, "median": 10, "max": 11},
            "n_excluded_same_group": queries,
            "n_excluded_same_image_different_group": 0,
        },
        "jspace_retrieval": {"n_queries": queries, "top1_accuracy": top1, "mrr": top1},
        "jspace_separation": {"gap": jspace_gap},
        "jspace_support_overlap": {"gap": 0.2},
        "raw_residual_retrieval": {"top1_accuracy": 0.3},
        "raw_residual_separation": {"gap": raw_gap},
        "shuffled_control": {
            "mean_top1_accuracy": shuffled_p95 / 2,
            "p95_top1_accuracy": shuffled_p95,
        },
    }


def _representational(layer, *, audio_top1, replication_top1=0.9, shuffled=0.3):
    return {
        "layer": layer,
        "pairs": {
            pair: _pair_entry(
                top1=audio_top1 if "spoken_audio" in pair else replication_top1,
                shuffled_p95=shuffled,
            )
            for pair in ALL_PAIRS
        },
    }


def test_all_six_directions_are_reported_with_everything_they_are_judged_on():
    rows = representational_rows(_representational(35, audio_top1=0.8))
    assert [row["pair"] for row in rows] == list(ALL_PAIRS)
    for row in rows:
        for key in (
            "jspace_top1",
            "jspace_mrr",
            "shuffled_p95",
            "jspace_separation_gap",
            "raw_separation_gap",
            "support_overlap_gap",
            "raw_top1",
            "n_distinct_source_images",
            "n_distinct_target_images",
            "n_excluded_same_group",
            "n_excluded_same_image_different_group",
        ):
            assert key in row, key


def test_a_missing_direction_is_reported_as_missing_not_as_failing():
    report = _representational(35, audio_top1=0.8)
    del report["pairs"]["spoken_audio->image"]
    rows = {row["pair"]: row for row in representational_rows(report)}
    assert rows["spoken_audio->image"]["evaluated"] is False
    assert rows["spoken_audio->image"]["beats_shuffled"] is False
    assert "no retrieval was computed" in rows["spoken_audio->image"]["reason"]


def test_audio_directions_beating_the_shuffled_p95_support_representation():
    verdict = representational_transfer_verdict(
        {35: _representational(35, audio_top1=0.8)},
        thresholds=THRESHOLDS,
        primary_layer=35,
    )
    assert verdict["verdict"] == TRANSFER_SUPPORTED
    assert sorted(verdict["audio_directions_beating_shuffled"]) == sorted(AUDIO_PAIRS)
    assert verdict["bidirectional"]["text<->spoken_audio"] is True
    assert verdict["text_image_replicates"] is True


def test_ties_with_the_shuffled_control_do_not_count_as_beating_it():
    """The gate is strictly greater: accuracy is discrete at 1/n_queries."""
    verdict = representational_transfer_verdict(
        {35: _representational(35, audio_top1=0.3, shuffled=0.3)},
        thresholds=THRESHOLDS,
        primary_layer=35,
    )
    assert verdict["verdict"] == TRANSFER_UNSUPPORTED
    assert verdict["audio_directions_beating_shuffled"] == []


def test_text_image_replicating_alone_never_supports_representational_transfer():
    verdict = representational_transfer_verdict(
        {35: _representational(35, audio_top1=0.1, replication_top1=0.95)},
        thresholds=THRESHOLDS,
        primary_layer=35,
    )
    assert verdict["verdict"] == TRANSFER_UNSUPPORTED
    assert verdict["text_image_replicates"] is True
    assert verdict["audio_directions_beating_shuffled"] == []


def test_audio_structure_only_away_from_the_primary_layer_is_weak():
    verdict = representational_transfer_verdict(
        {
            35: _representational(35, audio_top1=0.2),
            38: _representational(38, audio_top1=0.8),
        },
        thresholds=THRESHOLDS,
        primary_layer=35,
    )
    assert verdict["verdict"] == TRANSFER_WEAK
    assert verdict["audio_directions_beating_shuffled_at_primary_layer"] == []


# ------------------------------------------------------------ verdicts C/D


#: Every focal concept in these fixtures clears the gate in all three channels,
#: so the capability filter is a no-op here and each test still measures the
#: control clause it is named for. The filter itself is tested in
#: tests/test_mmpilot_admissibility.py.
_ALL_PASS_CAPABILITY = _capability(
    {"bus", "cat", "clock"}, selected=["bus", "cat", "clock"]
)



def _row(
    *,
    concept,
    pair,
    control_kind,
    alpha,
    effect,
    layer=35,
    sign_fraction=1.0,
    unrelated_change=0.01,
    norm_ratio=1.0,
    n_positive=4,
    n_negative=4,
):
    source, target = pair.split("->")
    return {
        "concept": concept,
        "source_modality": source,
        "target_modality": target,
        "pair": pair,
        "off_diagonal": True,
        "layer": layer,
        "control_kind": control_kind,
        "alpha": alpha,
        "n": n_positive + n_negative,
        "n_distinct_images": n_positive + n_negative,
        "n_positive_images": n_positive,
        "n_negative_images": n_negative,
        "mean_signed_target_effect": effect,
        "mean_signed_margin_effect": effect,
        "fraction_expected_sign": sign_fraction,
        "mean_abs_unrelated_change": unrelated_change,
        "mean_activation_norm_ratio": norm_ratio,
        "n_prediction_changes": 0,
    }


def _interventions(pairs, *, concepts=("bus", "cat"), layer=35, effect=0.5, **row_kwargs):
    rows = []
    for concept in concepts:
        for pair in pairs:
            rows.append(
                _row(
                    concept=concept,
                    pair=pair,
                    control_kind="source_concept",
                    alpha=0.5,
                    effect=effect,
                    layer=layer,
                    **row_kwargs,
                )
            )
            for control, value in (
                ("random_norm_matched", 0.01),
                ("unrelated_concept", 0.02),
                ("raw_residual_difference", 0.05),
            ):
                rows.append(
                    _row(
                        concept=concept,
                        pair=pair,
                        control_kind=control,
                        alpha=0.5,
                        effect=value,
                        layer=layer,
                        **row_kwargs,
                    )
                )
    return {"rows": rows}


def test_a_cell_passes_only_when_it_clears_its_own_matched_controls():
    cells = evaluate_causal_cells(
        _interventions(AUDIO_PAIRS),
        layer=35,
        focal_concepts=["bus", "cat"],
        pairs=AUDIO_PAIRS,
        thresholds=THRESHOLDS,
        capability=_ALL_PASS_CAPABILITY,
    )
    assert len(cells) == 2 * len(AUDIO_PAIRS)
    assert all(cell["passes"] for cell in cells)
    assert all(cell["audio_related"] for cell in cells)


def test_an_effect_that_does_not_clear_the_controls_fails_the_cell():
    interventions = _interventions(AUDIO_PAIRS, effect=0.015)
    cells = evaluate_causal_cells(
        interventions,
        layer=35,
        focal_concepts=["bus"],
        pairs=AUDIO_PAIRS,
        thresholds=THRESHOLDS,
        capability=_ALL_PASS_CAPABILITY,
    )
    assert not any(cell["passes"] for cell in cells)
    assert any("does not clear" in reason for cell in cells for reason in cell["reasons"])


def test_a_global_edit_fails_the_cell_however_large_the_effect():
    cells = evaluate_causal_cells(
        _interventions(AUDIO_PAIRS, effect=0.5, unrelated_change=0.6),
        layer=35,
        focal_concepts=["bus"],
        pairs=AUDIO_PAIRS,
        thresholds=THRESHOLDS,
        capability=_ALL_PASS_CAPABILITY,
    )
    assert not any(cell["passes"] for cell in cells)
    assert any("looks global" in reason for cell in cells for reason in cell["reasons"])


def test_an_insane_activation_norm_fails_the_cell():
    cells = evaluate_causal_cells(
        _interventions(AUDIO_PAIRS, norm_ratio=6.0),
        layer=35,
        focal_concepts=["bus"],
        pairs=AUDIO_PAIRS,
        thresholds=THRESHOLDS,
        capability=_ALL_PASS_CAPABILITY,
    )
    assert any(
        "activation norm ratio" in reason for cell in cells for reason in cell["reasons"]
    )


def test_a_cell_carried_by_one_photograph_fails_the_distinct_image_floor():
    cells = evaluate_causal_cells(
        _interventions(AUDIO_PAIRS, n_positive=1, n_negative=1),
        layer=35,
        focal_concepts=["bus"],
        pairs=AUDIO_PAIRS,
        thresholds=THRESHOLDS,
        capability=_ALL_PASS_CAPABILITY,
    )
    assert not any(cell["passes"] for cell in cells)
    assert any(
        "distinct positive image" in reason for cell in cells for reason in cell["reasons"]
    )
    assert all(cell["meets_claim_image_floor"] is False for cell in cells)


def test_the_wrong_sign_on_half_the_photographs_fails_the_cell():
    cells = evaluate_causal_cells(
        _interventions(AUDIO_PAIRS, sign_fraction=0.5),
        layer=35,
        focal_concepts=["bus"],
        pairs=AUDIO_PAIRS,
        thresholds=THRESHOLDS,
        capability=_ALL_PASS_CAPABILITY,
    )
    assert not any(cell["passes"] for cell in cells)
    assert any(
        "expected-sign fraction" in reason for cell in cells for reason in cell["reasons"]
    )


def test_cells_are_scoped_to_their_layer():
    rows = _interventions(AUDIO_PAIRS, layer=38)
    cells = evaluate_causal_cells(
        rows,
        layer=35,
        focal_concepts=["bus"],
        pairs=AUDIO_PAIRS,
        thresholds=THRESHOLDS,
        capability=_ALL_PASS_CAPABILITY,
    )
    assert all(cell["evaluated"] is False for cell in cells)
    assert all("no source-concept row" in cell["reasons"][0] for cell in cells)


def test_a_causal_verdict_needs_audio_cells_not_text_image_ones():
    verdict = causal_transfer_verdict(
        _interventions(REPLICATION_PAIRS),
        layer=35,
        focal_concepts=["bus", "cat"],
        thresholds=THRESHOLDS,
        name="L35_CAUSAL_TRANSFER",
        capability=_ALL_PASS_CAPABILITY,
    )
    assert verdict["verdict"] == TRANSFER_NOT_EVALUATED
    assert verdict["audio_cells_supporting_a_claim"] == []
    assert len(verdict["replication_cells_passing"]) == 2 * len(REPLICATION_PAIRS)


def test_bidirectional_audio_transfer_is_supported():
    verdict = causal_transfer_verdict(
        _interventions(AUDIO_PAIRS),
        layer=35,
        focal_concepts=["bus", "cat"],
        thresholds=THRESHOLDS,
        name="L35_CAUSAL_TRANSFER",
        capability=_ALL_PASS_CAPABILITY,
    )
    assert verdict["verdict"] == TRANSFER_SUPPORTED
    assert verdict["concepts_transferring_both_audio_directions"] == ["bus", "cat"]
    assert "earliest layer whose lens passed" in verdict["layer_choice_note"]


def test_one_directional_audio_transfer_is_weak():
    verdict = causal_transfer_verdict(
        _interventions(("text->spoken_audio", "image->spoken_audio")),
        layer=35,
        focal_concepts=["bus"],
        thresholds=THRESHOLDS,
        name="L35_CAUSAL_TRANSFER",
        capability=_ALL_PASS_CAPABILITY,
    )
    assert verdict["verdict"] == TRANSFER_WEAK
    assert verdict["concepts_transferring_both_audio_directions"] == []


def test_replication_is_not_evaluated_until_stage_c_runs():
    verdict = replication_verdict({35: {"verdict": TRANSFER_SUPPORTED}}, primary=None, layers=(38, 40))
    assert verdict["verdict"] == TRANSFER_NOT_EVALUATED
    assert verdict["layers_evaluated"] == []
    assert "does not start automatically" in verdict["rationale"]


def test_replication_reports_each_layer_without_choosing_one():
    verdict = replication_verdict(
        {
            35: {"verdict": TRANSFER_SUPPORTED},
            38: {"verdict": TRANSFER_SUPPORTED},
            40: {"verdict": TRANSFER_UNSUPPORTED},
        },
        primary={"verdict": TRANSFER_SUPPORTED},
        layers=(38, 40),
    )
    assert verdict["verdict"] == TRANSFER_SUPPORTED
    assert verdict["per_layer"] == {"38": TRANSFER_SUPPORTED, "40": TRANSFER_UNSUPPORTED}
    assert verdict["layers_evaluated"] == [38, 40]


# -------------------------------------------------------------- verdict E


def _overall(
    *,
    capability_passing=("bus", "cat"),
    audio_top1=0.8,
    causal_pairs=AUDIO_PAIRS,
    invariance=None,
    **cell_kwargs,
):
    selected = ["bus", "cat", "clock"]
    capability = audio_capability_verdict(
        _capability(set(capability_passing), selected=selected),
        selected_concepts=selected,
        modalities=MODALITIES,
        thresholds=THRESHOLDS,
    )
    representational = representational_transfer_verdict(
        {35: _representational(35, audio_top1=audio_top1)},
        thresholds=THRESHOLDS,
        primary_layer=35,
    )
    causal = (
        causal_transfer_verdict(
            _interventions(causal_pairs, **cell_kwargs),
            layer=35,
            focal_concepts=["bus", "cat"],
            thresholds=THRESHOLDS,
            name="L35_CAUSAL_TRANSFER",
            capability=capability,
        )
        if causal_pairs
        else None
    )
    replication = replication_verdict(
        {35: causal} if causal else {}, primary=causal, layers=(38, 40)
    )
    overall = overall_verdict(
        capability=capability,
        representational=representational,
        primary_causal=causal,
        replication=replication,
        invariance=invariance
        if invariance is not None
        else {"passed": True, "per_modality": dict.fromkeys(MODALITIES, {})},
        thresholds=THRESHOLDS,
    )
    return capability, representational, causal, replication, overall


def test_a_complete_three_modality_result_is_a_go():
    _c, _r, _causal, _rep, overall = _overall()
    assert overall["verdict"] == THREE_MODALITY_GO
    assert set(overall["criteria_status"].values()) == {PASS}
    assert overall["component_verdicts"]["A_audio_capability"] == AUDIO_CAPABILITY_GO


def test_representation_without_causality_is_a_weak_go():
    _c, _r, _causal, _rep, overall = _overall(causal_pairs=())
    assert overall["verdict"] == THREE_MODALITY_WEAK_GO
    assert overall["criteria_status"]["audio_causal_cell_with_expected_sign"] == (
        NOT_EVALUATED
    )
    assert "not, by themselves, coordinates the model uses" in overall["rationale"]


def test_a_capability_failure_is_a_no_go():
    _c, _r, _causal, _rep, overall = _overall(capability_passing=("bus",))
    assert overall["verdict"] == THREE_MODALITY_NO_GO
    assert overall["criteria_status"]["three_modality_capability"] == FAIL


def test_a_result_resting_on_text_image_alone_is_a_no_go():
    _c, _r, _causal, _rep, overall = _overall(
        audio_top1=0.1, causal_pairs=REPLICATION_PAIRS
    )
    assert overall["verdict"] == THREE_MODALITY_NO_GO
    assert overall["criteria_status"]["not_text_image_replication_alone"] == FAIL


def test_a_nonspecific_edit_is_a_no_go():
    _c, _r, _causal, _rep, overall = _overall(unrelated_change=0.6)
    assert overall["verdict"] == THREE_MODALITY_NO_GO
    assert overall["criteria_status"]["effects_are_specific"] == FAIL


def test_insane_norms_are_a_no_go():
    _c, _r, _causal, _rep, overall = _overall(norm_ratio=6.0)
    assert overall["verdict"] == THREE_MODALITY_NO_GO
    assert overall["criteria_status"]["activation_norms_sane"] == FAIL


def test_a_failed_invariance_gate_is_a_no_go():
    _c, _r, _causal, _rep, overall = _overall(
        invariance={"passed": False, "per_modality": {}}
    )
    assert overall["verdict"] == THREE_MODALITY_NO_GO
    assert overall["criteria_status"]["invariance_gate"] == FAIL


def test_environmental_audio_is_never_claimed():
    _c, _r, _causal, _rep, overall = _overall()
    assert overall["criteria_status"]["environmental_audio_not_claimed"] == PASS
    assert "spoken *captions*" in overall["scope_limitation"]
    assert "environmental sound" in overall["scope_limitation"]
    assert "not erasure" in overall["intervention_limitation"]
    assert "No cross-modal alignment" in overall["alignment_limitation"]


def test_layer_35_is_never_described_as_pre_convergence():
    _c, _r, _causal, _rep, overall = _overall()
    note = overall["layer_choice_note"]
    assert "earliest layer whose lens passed" in note
    assert "remains unresolved" in note
    for forbidden in ("pre-language", "pre-convergence semantics"):
        assert forbidden not in note.replace("precedes answer-language", "")


# ------------------------------------------------------------------ report


def test_the_report_prints_the_five_verdicts_and_the_tables_behind_them():
    capability, representational, causal, replication, overall = _overall()
    text = render_report(
        run_dir="/tmp/run",
        capability=capability,
        representational=representational,
        primary_causal=causal,
        replication=replication,
        overall=overall,
        lens_report={
            "checksums": {35: "sha256:aaa", 38: "sha256:bbb", 40: "sha256:ccc"},
            "combined_checksum": "sha256:ddd",
        },
        audio_protocol={
            "protocol_version": "jlens.mmpilot.native_spoken_audio.v1",
            "protocol_fingerprint": "sha256:eee",
            "call_convention": "chat_template_audio_content_block",
            "dynamic_placeholder_count": True,
        },
        budgets=[{"stage": "A", "total_passes": 1}],
        resume={"status": "starting", "completed_units": {}},
        mode="native_audio_transfer",
    )
    for label in (
        "AUDIO_CAPABILITY",
        "REPRESENTATIONAL_TRANSFER",
        "L35_CAUSAL_TRANSFER",
        "L38_L40_REPLICATION",
        "OVERALL_THREE_MODALITY_VERDICT",
    ):
        assert label in text
    for pair in ALL_PAIRS:
        assert pair in text
    assert "jlens.mmpilot.native_spoken_audio.v1" in text
    assert "sha256:ddd" in text
    assert "Nothing was fitted in this run" in text
    assert "MOCK run" not in text


def test_a_mock_report_says_so_in_its_second_line():
    capability, representational, causal, replication, overall = _overall()
    text = render_report(
        run_dir="/tmp/run",
        capability=capability,
        representational=representational,
        primary_causal=causal,
        replication=replication,
        overall=overall,
        lens_report=None,
        audio_protocol=None,
        mode="mock",
    )
    assert "MOCK run: pipeline evidence only, not scientific evidence." in text


def test_interventions_that_resemble_random_are_a_no_go_not_a_weak_go():
    """The distinction WEAK GO turns on: thin evidence versus contrary evidence."""
    _c, _r, causal, _rep, overall = _overall(effect=0.015)
    assert causal["verdict"] == TRANSFER_UNSUPPORTED
    assert overall["verdict"] == THREE_MODALITY_NO_GO
    assert "not distinguishable from the random" in overall["rationale"]


def test_one_directional_causality_is_a_weak_go_not_a_no_go():
    _c, _r, causal, _rep, overall = _overall(
        causal_pairs=("text->spoken_audio", "image->spoken_audio")
    )
    assert causal["verdict"] == TRANSFER_WEAK
    assert overall["verdict"] == THREE_MODALITY_WEAK_GO


# ---------------------------------------------- architecture and invariance


def _mock_backend_and_inputs():
    import tempfile
    from pathlib import Path

    from jlens.mmpilot.capability import build_question
    from jlens.mmpilot.mock import MockPilotBackend, MockWorld, build_mock_dataset
    from jlens.mmpilot.pipeline import build_condition_inputs

    root = Path(tempfile.mkdtemp(prefix="trimodal-"))
    world = MockWorld({"bus": ("bus",), "cat": ("cat",)})
    built = build_mock_dataset(
        root, world=world, images_per_concept=2, negative_images=2, captions_per_image=1
    )
    backend = MockPilotBackend(world, supports_audio=True)
    group = {
        "group_id": "g0",
        "image_id": "img0000",
        "caption": "a photo showing a cat",
        "image_path": str(Path(built["image_root"]) / "images" / "img0000.jpg"),
        "audio_path": str(Path(built["audio_root"]) / "audio" / "img0000_0.wav"),
        "concept": "cat",
        "split": "train",
    }
    media = {
        "load_image": __import__(
            "jlens.mmpilot.mock", fromlist=["load_mock_media"]
        ).load_mock_media,
        "load_audio": lambda path: (
            __import__("jlens.mmpilot.mock", fromlist=["load_mock_media"]).load_mock_media(
                path
            ),
            16000,
        ),
    }
    question = build_question(["bus", "cat"])
    inputs = {
        modality: build_condition_inputs(backend, group, modality, question, media)
        for modality in MODALITIES
    }
    return backend, inputs


def test_the_architecture_report_covers_every_modality_and_layer():
    from jlens.mmpilot.tri_modal import modality_architecture_report

    backend, inputs = _mock_backend_and_inputs()
    report = modality_architecture_report(backend, inputs, layers=(2, 3, 4))
    assert report["passed"] is True
    assert sorted(report["per_modality"]) == sorted(MODALITIES)
    assert report["layers"] == [2, 3, 4]
    assert "post-block residual" in report["hook_site"]
    assert "final prompt token" in report["read_and_edit_position"]
    for entry in report["per_modality"].values():
        assert entry["final_prompt_position"] == entry["prompt_len"] - 1


def test_a_layer_outside_the_decoder_is_refused():
    from jlens.mmpilot.tri_modal import (
        ArchitectureMismatch,
        modality_architecture_report,
    )

    backend, inputs = _mock_backend_and_inputs()
    with pytest.raises(ArchitectureMismatch, match="outside the"):
        modality_architecture_report(backend, inputs, layers=(2, 99))


def test_a_final_position_inside_the_media_span_is_refused():
    from jlens.mmpilot.tri_modal import (
        ArchitectureMismatch,
        modality_architecture_report,
    )

    backend, inputs = _mock_backend_and_inputs()
    inputs["image"].modality_token_range = [0, inputs["image"].prompt_len]
    with pytest.raises(ArchitectureMismatch, match="inside the media span"):
        modality_architecture_report(backend, inputs, layers=(2,))


def test_zero_audio_placeholders_are_refused():
    """Features without placeholders is the state that blocked the pilot."""
    from jlens.mmpilot.tri_modal import (
        ArchitectureMismatch,
        modality_architecture_report,
    )

    backend, inputs = _mock_backend_and_inputs()
    inputs["spoken_audio"].audio = {
        "n_placeholders": 0,
        "n_expected_from_features": 0,
        "audio_token_span": [0, 0],
    }
    with pytest.raises(ArchitectureMismatch, match="never entered the model"):
        modality_architecture_report(backend, inputs, layers=(2,))


def test_a_placeholder_feature_disagreement_is_refused():
    from jlens.mmpilot.tri_modal import (
        ArchitectureMismatch,
        modality_architecture_report,
    )

    backend, inputs = _mock_backend_and_inputs()
    inputs["spoken_audio"].audio = {
        "n_placeholders": 12,
        "n_expected_from_features": 25,
        "audio_token_span": [5, 17],
    }
    with pytest.raises(ArchitectureMismatch, match="implied by the feature mask"):
        modality_architecture_report(backend, inputs, layers=(2,))


def test_the_invariance_gate_runs_in_every_modality():
    from jlens.mmpilot.tri_modal import run_invariance_by_modality

    backend, inputs = _mock_backend_and_inputs()
    report = run_invariance_by_modality(backend, inputs, (2, 3, 4))
    assert report["passed"] is True
    assert report["modalities"] == sorted(MODALITIES)
    for entry in report["per_modality"].values():
        assert entry["capture_noop"]["passed"] is True
        assert [check["passed"] for check in entry["zero_intervention"]] == [True] * 3


def test_the_audio_protocol_fingerprint_is_a_refusal_not_a_warning():
    from jlens.mmpilot.audio import ResolvedAudioInterface
    from jlens.mmpilot.tri_modal import AudioProtocolMismatch, assert_audio_protocol

    resolved = ResolvedAudioInterface(
        protocol_version="jlens.mmpilot.native_spoken_audio.v1",
        call_convention="chat_template_audio_content_block",
        audio_kwarg="audio",
        content_block_schema={"type": "audio", "audio": "<ndarray>"},
        chat_template_convention="user_content_blocks",
        sampling_rate=16_000,
        waveform_dtype="float32",
        waveform_ndim=1,
        audio_token="<|audio|>",
        audio_token_id=5,
        boa_token=None,
        eoa_token=None,
        feature_keys=("input_features", "input_features_mask"),
        processor_class="Gemma4Processor",
        feature_extractor_class="Gemma4AudioFeatureExtractor",
        audio_tower_present=True,
        model_repo_id="google/gemma-4-E4B-it",
        model_revision="fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd",
        processor_revision="fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd",
        transformers_version="5.10.2",
        dynamic_placeholder_count=True,
    )
    record = assert_audio_protocol(
        resolved, expected_fingerprint=resolved.protocol_fingerprint
    )
    assert record["matches_expected_fingerprint"] is True
    with pytest.raises(AudioProtocolMismatch, match="transformers"):
        assert_audio_protocol(resolved, expected_fingerprint="sha256:" + "0" * 64)
