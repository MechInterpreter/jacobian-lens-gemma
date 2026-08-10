# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""The L27-L31 adjacent-layer lens protocol: the interval, the gate, the set.

Four properties carry the scientific weight, and each has a test that fails if
it stops holding:

1. the candidate interval is **closed** and is exactly ``(27, 28, 29, 30, 31)``;
2. the gate is the extension's own object — not an equal copy that could drift;
3. the confirmation set is proved untouched, and is **refused** rather than
   shrunk when the corpus cannot fill it;
4. the selected layer is the **lowest** passer, never the best-looking one.
"""

import pytest

from jlens.calibration.adjacent import (
    ADJACENT_CANDIDATE_LAYERS,
    ADJACENT_CONFIRMATION_GATE,
    ADJACENT_FITTING_SCALE,
    ADJACENT_GATE,
    ADJACENT_LENS_GO,
    ADJACENT_LENS_NO_GO,
    ADJACENT_PROTOCOL,
    ADJACENT_SELECTION_RULE,
    AMBIGUOUS_UPPER_LAYER,
    FAILED_LOWER_LAYER,
    N_CONFIRMATION_PROMPTS,
    AdjacentStore,
    ConfirmationNotUntouched,
    SourceLayerSetRefused,
    UntouchedConfirmationRefused,
    adjacent_budget,
    adjacent_gate_text,
    adjacent_lens_verdict,
    assert_new_source_layers,
    audit_untouched_confirmation,
    build_untouched_confirmation,
    confirmation_table,
    format_adjacent_budget,
    format_confirmation_table,
    select_earliest_confirmed_layer,
)
from jlens.calibration.adjacent_mock import ADJACENT_SCENARIOS, mock_adjacent_rows
from jlens.calibration.corpus import build_records
from jlens.calibration.extension import (
    EXTENSION_CONFIRMATION_GATE,
    EXTENSION_GATE,
)
from jlens.calibration.gate import evaluate_calibration_layers
from jlens.calibration.state import CalibrationFingerprint

# --------------------------------------------------------------- the interval


def test_the_candidate_interval_is_exactly_the_five_predeclared_layers():
    assert ADJACENT_CANDIDATE_LAYERS == (27, 28, 29, 30, 31)


def test_the_interval_is_bracketed_by_the_two_results_that_close_it():
    assert FAILED_LOWER_LAYER == 26
    assert AMBIGUOUS_UPPER_LAYER == 32
    assert min(ADJACENT_CANDIDATE_LAYERS) == FAILED_LOWER_LAYER + 1
    assert max(ADJACENT_CANDIDATE_LAYERS) == AMBIGUOUS_UPPER_LAYER - 1


def test_no_layer_outside_the_interval_can_be_a_candidate():
    for outside in (26, 32, 33, 34, 35):
        assert outside not in ADJACENT_CANDIDATE_LAYERS


def test_the_protocol_declares_the_interval_closed_and_the_scale_fixed():
    payload = ADJACENT_PROTOCOL.to_dict()
    assert payload["candidate_layers_are_closed"] is True
    assert payload["candidate_layers"] == list(ADJACENT_CANDIDATE_LAYERS)
    assert payload["fitting_scale"] == ADJACENT_FITTING_SCALE == 250
    assert payload["multimodal_data_in_fitting"] is False
    assert payload["multimodal_data_in_lens_validation"] is False
    assert "output of model.language_model.layers[l]" in payload["hook_site"]


def test_the_protocol_digest_moves_when_the_protocol_does():
    from dataclasses import replace

    assert ADJACENT_PROTOCOL.digest != replace(
        ADJACENT_PROTOCOL, candidate_layers=(27, 28)
    ).digest


# ------------------------------------------------------------------ the gate


def test_the_gate_is_the_extension_gate_object_not_a_copy():
    # Identity, not equality: an equal copy drifts on the next edit; this cannot.
    assert ADJACENT_GATE is EXTENSION_GATE
    assert ADJACENT_CONFIRMATION_GATE is EXTENSION_CONFIRMATION_GATE


def test_no_threshold_differs_from_the_extension_gate():
    from dataclasses import asdict

    assert asdict(ADJACENT_CONFIRMATION_GATE) == asdict(EXTENSION_CONFIRMATION_GATE)


def test_the_gate_text_says_the_gate_is_unchanged():
    text = adjacent_gate_text()
    assert ADJACENT_CONFIRMATION_GATE.digest in text
    assert "NOTHING IS CHANGED" in text


# --------------------------------------------------- the source-layer guardrail


def test_the_extension_accumulator_is_disjoint_from_the_candidates():
    record = assert_new_source_layers(
        parent_source_layers=(8, 14, 20, 26, 32, 35, 38, 40)
    )
    assert record["disjoint"] is True
    assert record["overlap"] == []
    assert record["parent_accumulator_may_be_seeded"] is False
    assert record["new_accumulation_required"] is True


def test_an_overlapping_parent_grid_is_refused():
    with pytest.raises(SourceLayerSetRefused, match="29"):
        assert_new_source_layers(parent_source_layers=(8, 29, 40))


# ------------------------------------------------ the untouched confirmation


def _corpus(n: int, *, offset: int = 0, prefix: str = "doc"):
    texts = [
        f"{prefix} {index + offset} " + " ".join(
            f"w{(index + offset) * 37 + step}" for step in range(120)
        )
        for index in range(n)
    ]
    return build_records("test/corpus", texts, min_chars=100)


def test_the_untouched_set_excludes_every_named_spent_set():
    pool = _corpus(1200)
    spent = {
        "parent_fit": pool[:100],
        "parent_development": pool[100:200],
        "parent_confirmation_opened": pool[200:300],
        "adjacent_fit": pool[300:400],
        "extension_development_reused_here": pool[400:500],
        "extension_confirmation_opened": pool[500:600],
    }
    confirmation = build_untouched_confirmation(
        pool, excluded=spent, corpus_id="test/corpus"
    )
    assert len(confirmation.records) == N_CONFIRMATION_PROMPTS
    spent_ids = {
        record.record_id for records in spent.values() for record in records
    }
    assert not spent_ids & set(confirmation.record_ids())
    report = audit_untouched_confirmation(confirmation, excluded=spent)
    assert report["untouched"] is True
    assert report["n_exact_hits"] == 0
    assert sorted(report["required_disjoint_from"]) == sorted(spent)


def test_a_short_corpus_blocks_the_study_rather_than_shrinking_the_set():
    pool = _corpus(60)
    with pytest.raises(UntouchedConfirmationRefused) as error:
        build_untouched_confirmation(pool, excluded={}, corpus_id="test/corpus")
    message = str(error.value)
    assert "BLOCKED, not resized" in message
    assert "reopen a spent set" in message


def test_a_planted_spent_record_is_caught_by_the_independent_audit():
    pool = _corpus(1200)
    spent = {"extension_confirmation_opened": pool[:200]}
    confirmation = build_untouched_confirmation(
        pool, excluded=spent, corpus_id="test/corpus"
    )
    from dataclasses import replace

    contaminated = replace(
        confirmation,
        records=(*confirmation.records[:-1], spent["extension_confirmation_opened"][0]),
    )
    with pytest.raises(ConfirmationNotUntouched, match="not untouched"):
        audit_untouched_confirmation(contaminated, excluded=spent)


def test_an_internal_duplicate_is_caught_too():
    pool = _corpus(1200)
    confirmation = build_untouched_confirmation(
        pool, excluded={}, corpus_id="test/corpus"
    )
    from dataclasses import replace

    duplicated = replace(
        confirmation,
        records=(*confirmation.records[:-1], confirmation.records[0]),
    )
    with pytest.raises(ConfirmationNotUntouched):
        audit_untouched_confirmation(duplicated, excluded={})


def test_the_confirmation_manifest_records_that_nothing_was_reduced_or_reused():
    pool = _corpus(1200)
    manifest = build_untouched_confirmation(
        pool, excluded={}, corpus_id="test/corpus",
        development_role={"reused": True},
        dependency_manifests=("parent", "extension"),
    ).manifest()
    assert manifest["size"] == N_CONFIRMATION_PROMPTS
    assert manifest["size_reduced_to_fit_corpus"] is False
    assert manifest["previously_opened_sets_reused"] is False
    assert manifest["selected_by_jlens_performance"] is False
    assert manifest["dependency_manifests"] == ["parent", "extension"]


def test_the_bucket_tag_differs_from_the_extension_so_the_partition_is_new():
    from jlens.calibration.adjacent import _adjacent_bucket
    from jlens.calibration.extension import _extension_bucket

    differing = sum(
        _adjacent_bucket(f"r{i}", seed=1, n_buckets=100)
        != _extension_bucket(f"r{i}", seed=1, n_buckets=100)
        for i in range(200)
    )
    assert differing > 150


# ------------------------------------------------------------ the selection


@pytest.fixture(scope="module")
def scenario_results():
    return {
        key: evaluate_calibration_layers(
            mock_adjacent_rows(key, stage="confirmation", n_prompts=256),
            layers=list(ADJACENT_CANDIDATE_LAYERS),
            scale=ADJACENT_FITTING_SCALE,
            stage="confirmation",
            gate=ADJACENT_CONFIRMATION_GATE,
        )
        for key in ADJACENT_SCENARIOS
    }


@pytest.mark.parametrize("key", sorted(ADJACENT_SCENARIOS))
def test_every_commissioned_scenario_reaches_its_expected_layer(key, scenario_results):
    scenario = ADJACENT_SCENARIOS[key]
    selection = select_earliest_confirmed_layer(scenario_results[key])
    assert selection["selected_layer"] == scenario.expected_selected_layer
    assert selection["verdict"] == scenario.expected_verdict


def test_the_rule_takes_the_lowest_passer_not_the_best_looking_one(scenario_results):
    # L31 is deliberately given the better margin in this scenario.
    selection = select_earliest_confirmed_layer(scenario_results["earliest_wins"])
    assert selection["passing_layers"] == [29, 31]
    assert selection["selected_layer"] == 29
    assert selection["best_looking_failure_considered"] is False


def test_the_complete_table_is_recorded_even_when_nothing_passes(scenario_results):
    selection = select_earliest_confirmed_layer(scenario_results["none_pass"])
    assert selection["selected_layer"] is None
    assert selection["verdict"] == ADJACENT_LENS_NO_GO
    assert [row["layer"] for row in selection["table"]] == list(
        ADJACENT_CANDIDATE_LAYERS
    )
    assert all(row["evaluated"] for row in selection["table"])
    assert all(row["failed_clauses"] for row in selection["table"])


def test_development_passing_does_not_confirm_anything():
    development = evaluate_calibration_layers(
        mock_adjacent_rows("development_only", stage="development", n_prompts=256),
        layers=list(ADJACENT_CANDIDATE_LAYERS),
        scale=ADJACENT_FITTING_SCALE,
        stage="validation",
        gate=ADJACENT_GATE,
    )
    confirmation = evaluate_calibration_layers(
        mock_adjacent_rows("development_only", stage="confirmation", n_prompts=256),
        layers=list(ADJACENT_CANDIDATE_LAYERS),
        scale=ADJACENT_FITTING_SCALE,
        stage="confirmation",
        gate=ADJACENT_CONFIRMATION_GATE,
    )
    assert all(entry["passed"] for entry in development.values())
    selection = select_earliest_confirmed_layer(confirmation, development=development)
    assert selection["selected_layer"] is None


def test_the_table_reports_every_required_column(scenario_results):
    rows = confirmation_table(scenario_results["earliest_wins"])
    for row in rows:
        for key in (
            "mean_reciprocal_rank",
            "median_midrank",
            "median_optimistic_rank",
            "median_pessimistic_rank",
            "tied_at_max_rate",
            "top_k_inclusion",
            "controls",
            "fold_mrr",
            "fold_beats_all_controls",
            "prompt_coverage",
            "n_distinct_target_tokens",
        ):
            assert key in row, key
        assert set(row["controls"]) >= {"permuted", "random", "wrong_layer"}
    assert "failed clauses" in format_confirmation_table(rows)


def test_the_selection_rule_is_declared_before_confirmation():
    payload = ADJACENT_SELECTION_RULE.to_dict()
    assert payload["declared_before_confirmation_opened"] is True
    assert payload["best_looking_failure_may_be_chosen"] is False
    assert payload["multimodal_outcomes_may_be_consulted"] is False
    assert payload["on_none_passing"] == ADJACENT_LENS_NO_GO


# ------------------------------------------------------------- the verdict


def _verdict_inputs(selection):
    return {
        "confirmation_manifest": {"size": N_CONFIRMATION_PROMPTS},
        "untouched_audit": {
            "untouched": True,
            "n_exact_hits": 0,
            "n_near_hits": 0,
            "n_internal_duplicates": 0,
            "required_disjoint_from": ["parent_fit"],
        },
        "source_layer_record": {
            "disjoint": True,
            "candidate_layers": list(ADJACENT_CANDIDATE_LAYERS),
            "parent_source_layers": [8, 32],
            "overlap": [],
        },
    }


def test_a_passing_layer_yields_go(scenario_results):
    selection = select_earliest_confirmed_layer(scenario_results["earliest_wins"])
    verdict = adjacent_lens_verdict(selection, **_verdict_inputs(selection))
    assert verdict["verdict"] == ADJACENT_LENS_GO
    assert verdict["selected_layer"] == 29
    assert verdict["failed_validity_clauses"] == []


def test_a_confirmation_set_that_was_not_proved_untouched_refuses(scenario_results):
    selection = select_earliest_confirmed_layer(scenario_results["earliest_wins"])
    inputs = _verdict_inputs(selection)
    inputs["untouched_audit"] = {**inputs["untouched_audit"], "untouched": False}
    verdict = adjacent_lens_verdict(selection, **inputs)
    assert verdict["verdict"] == ADJACENT_LENS_NO_GO
    assert verdict["selected_layer"] is None
    assert "confirmation_set_proved_untouched" in verdict["failed_validity_clauses"]


def test_a_shrunken_confirmation_set_refuses(scenario_results):
    selection = select_earliest_confirmed_layer(scenario_results["earliest_wins"])
    inputs = _verdict_inputs(selection)
    inputs["confirmation_manifest"] = {"size": 128}
    verdict = adjacent_lens_verdict(selection, **inputs)
    assert verdict["verdict"] == ADJACENT_LENS_NO_GO
    assert "confirmation_size_not_reduced" in verdict["failed_validity_clauses"]


def test_a_seeded_parent_accumulator_refuses(scenario_results):
    selection = select_earliest_confirmed_layer(scenario_results["earliest_wins"])
    inputs = _verdict_inputs(selection)
    inputs["source_layer_record"] = {
        **inputs["source_layer_record"],
        "disjoint": False,
        "overlap": [29],
    }
    verdict = adjacent_lens_verdict(selection, **inputs)
    assert verdict["verdict"] == ADJACENT_LENS_NO_GO
    assert (
        "new_jacobian_accumulation_for_new_source_layers"
        in verdict["failed_validity_clauses"]
    )


def test_a_partially_evaluated_candidate_set_refuses(scenario_results):
    selection = select_earliest_confirmed_layer(
        {29: scenario_results["earliest_wins"][29]}, candidates=(29,)
    )
    verdict = adjacent_lens_verdict(selection, **_verdict_inputs(selection))
    assert "all_candidates_evaluated" in verdict["failed_validity_clauses"]


# --------------------------------------------------------------- store, budget


def test_the_store_refuses_a_stage_it_does_not_know(tmp_path):
    fingerprint = CalibrationFingerprint(
        mode="mock",
        protocol_version="x",
        model_repo_id="m",
        model_revision="r",
        tokenizer_revision="r",
        capture_plan_digest="d",
        corpus_manifest_checksum="c",
        gate_digest="g",
        plateau_rule_digest="p",
        scale_points=(250,),
        artifact_format_version="v",
    )
    store = AdjacentStore(tmp_path / "lens", fingerprint)
    store.open()
    store.save("adjacent_fit", "record", {"ok": True})
    assert store.load("adjacent_fit", "record") == {"ok": True}
    with pytest.raises(ValueError, match="unknown adjacent-layer stage"):
        store.stage_dir("continuation")
    assert "adjacent" in store.snapshot_path(250).name


def test_the_budget_scales_by_measured_backward_span_not_layer_count():
    shallow = adjacent_budget(layers=(8,))
    ours = adjacent_budget()
    assert ours["backward_span_blocks"] == 41 - 27
    assert ours["seconds_per_prompt"] < shallow["seconds_per_prompt"]
    assert ours["n_forward_passes"] == 250
    assert ours["n_backward_passes"] == 250 * (41 - 27)
    text = format_adjacent_budget(ours)
    assert "backward span" in text
    assert "checkpoints every bounded batch" in text
