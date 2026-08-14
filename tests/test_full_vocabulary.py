"""The unrestricted next-token endpoint and the corrected causal rerun.

Numbered against the requirements the correction was commissioned under. The
theme running through all of them: the global argmax is a property of the whole
vocabulary and of the untouched prompt, the completed runs are immutable, and a
restricted-candidate preference can never become a full-vocabulary GO.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
import torch

from jlens.mmpilot.coordinate_swap_mock import PROPERTY_QUESTION, SwapMockBackend
from jlens.mmpilot.endpoint_amend import (
    BAND_FOLLOWUP_TERMINOLOGY,
    CORRECTED_LABELS,
    THREE_MODALITY_TERMINOLOGY,
    AmendmentRefused,
    build_endpoint_amendment,
    verify_amendment_binding,
    write_endpoint_amendment,
)
from jlens.mmpilot.endpoint_audit import (
    AUDITED_ENDPOINTS,
    endpoint_audit_record,
    scan_active_sources,
    verify_registry,
)
from jlens.mmpilot.full_vocab_mock import (
    CAUSAL_SCENARIOS,
    MOCK_ANSWERS,
    PROVENANCE_SCENARIOS,
    mock_band_followup_report,
    mock_full_vocab_records,
    mock_population_reuse,
    mock_scored,
)
from jlens.mmpilot.full_vocab_study import (
    CANONICAL_AUDIO_REPORT_CHECKSUM_PIN,
    FULL_VOCAB_REASONING_ALPHA1_GO,
    FULL_VOCAB_REASONING_ALPHA2_ONLY,
    FULL_VOCAB_REASONING_NO_GO,
    PASS_CAP,
    REQUIRED_PINS,
    VERDICT_NAMES,
    FullVocabRefused,
    FullVocabThresholds,
    assert_prompt_reconstruction,
    conditional_logprob_verdict,
    cross_modal_conjunction,
    family_a_trials,
    full_vocab_design_record,
    full_vocab_fingerprint,
    full_vocab_pass_budget,
    read_band_followup_report,
    read_canonical_audio_provenance,
    read_historical_prompt_hashes,
    require_pin,
    resolve_study_tokens,
    reuse_completed_population,
    summarize_full_vocab_cells,
    unrestricted_reasoning_verdict,
)
from jlens.mmpilot.full_vocabulary import (
    ENDPOINT_RESTRICTED_CANDIDATE,
    ENDPOINT_UNRESTRICTED_NEXT_TOKEN,
    MultiTokenAnswerError,
    UnrestrictedScoringRefused,
    answer_token_table,
    greedy_generate,
    normalize_generated_text,
    resolve_answer_token,
    restricted_candidate_top1,
    score_unrestricted_next_token,
    tie_aware_ranks,
    unrestricted_trial_record,
)
from jlens.mmpilot.store import payload_checksum
from jlens.mmpilot.validated_band_followup import (
    EXCLUDED_FAILED_LAYER,
    FOLLOWUP_REPORT_NAME,
)

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def backend() -> SwapMockBackend:
    return SwapMockBackend()


@pytest.fixture(scope="module")
def inputs(backend):
    return backend.build_inputs(
        prompt=PROPERTY_QUESTION, modality="text", concept="bird", nuisance_key="t"
    )


@pytest.fixture(scope="module")
def design() -> dict:
    return full_vocab_design_record()


# --------------------------------------------------------------------------
# 1. Full-vocabulary argmax is independent of supplied candidate names
# --------------------------------------------------------------------------


def test_global_argmax_does_not_depend_on_the_named_targets(backend, inputs):
    table = answer_token_table(backend, ["bird", "cat", "two", "four"])
    ids = table["token_ids"]

    none_named = score_unrestricted_next_token(backend, inputs, top_k=5)
    property_named = score_unrestricted_next_token(
        backend, inputs, target_token_ids={"target": ids["two"], "source": ids["four"]}
    )
    identity_named = score_unrestricted_next_token(
        backend, inputs, target_token_ids={"target": ids["cat"], "source": ids["bird"]}
    )
    reversed_named = score_unrestricted_next_token(
        backend, inputs, target_token_ids={"target": ids["four"], "source": ids["two"]}
    )
    argmaxes = {
        row["global_argmax_token_id"]
        for row in (none_named, property_named, identity_named, reversed_named)
    }
    assert len(argmaxes) == 1
    assert none_named["candidate_list_supplied"] is False


def test_the_scorer_takes_no_candidate_list_parameter():
    source = ast.parse(
        (ROOT / "jlens" / "mmpilot" / "full_vocabulary.py").read_text(encoding="utf-8")
    )
    fn = next(
        node
        for node in ast.walk(source)
        if isinstance(node, ast.FunctionDef)
        and node.name == "score_unrestricted_next_token"
    )
    names = {arg.arg for arg in fn.args.args} | {
        arg.arg for arg in fn.args.kwonlyargs
    }
    assert "candidate_ids" not in names
    assert "candidates" not in names
    assert "candidate_token_ids" not in names


# --------------------------------------------------------------------------
# 2/3. No candidate token is appended; scored length equals prompt length
# --------------------------------------------------------------------------


def test_no_token_is_appended_and_lengths_match(backend, inputs):
    scored = score_unrestricted_next_token(backend, inputs, top_k=3)
    assert scored["n_candidate_positions_appended"] == 0
    assert scored["scored_input_length"] == inputs.prompt_len
    assert scored["prompt_len"] == inputs.prompt_len
    assert scored["final_prompt_position"] == inputs.prompt_len - 1


def test_an_extended_input_is_refused(backend, inputs):
    from jlens.mmpilot.capability import _extend_tensors

    extended = _extend_tensors(inputs.tensors, inputs.prompt_len, [MOCK_ANSWERS["two"]])
    stretched = type(inputs)(
        tensors=extended,
        prompt_len=inputs.prompt_len,
        modality=inputs.modality,
        prompt_hash=inputs.prompt_hash,
        modality_token_range=inputs.modality_token_range,
    )
    with pytest.raises(UnrestrictedScoringRefused, match="appended"):
        score_unrestricted_next_token(backend, stretched)


# --------------------------------------------------------------------------
# 4. The scorer's argmax matches raw logits.argmax()
# --------------------------------------------------------------------------


def test_argmax_matches_raw_logits_argmax(backend, inputs):
    scored = score_unrestricted_next_token(backend, inputs, top_k=3)
    raw = backend.forward_logits(inputs.tensors)
    expected = int(raw[0, inputs.final_prompt_position].argmax())
    assert scored["global_argmax_token_id"] == expected
    assert scored["raw_logits_argmax_token_id"] == expected
    assert scored["argmax_agrees_with_raw_logits"] is True


def test_a_disagreeing_argmax_is_refused(backend, inputs, monkeypatch):
    real = torch.argmax
    calls = {"n": 0}

    def flipped(tensor, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:  # the log-prob argmax only
            return real(-tensor, *args, **kwargs)
        return real(tensor, *args, **kwargs)

    monkeypatch.setattr(torch, "argmax", flipped)
    with pytest.raises(UnrestrictedScoringRefused, match="disagrees"):
        score_unrestricted_next_token(backend, inputs)


def test_non_finite_logits_are_refused(backend, inputs, monkeypatch):
    real = backend.forward_logits

    def poisoned(tensors):
        logits = real(tensors).clone()
        logits[0, -1, 3] = float("nan")
        return logits

    monkeypatch.setattr(backend, "forward_logits", poisoned)
    with pytest.raises(UnrestrictedScoringRefused, match="non-finite"):
        score_unrestricted_next_token(backend, inputs)


def test_an_unexpected_vocabulary_size_is_refused(backend, inputs):
    with pytest.raises(UnrestrictedScoringRefused, match="vocabulary size"):
        score_unrestricted_next_token(backend, inputs, expected_vocab_size=262144)


def test_a_missing_target_id_is_refused(backend, inputs):
    with pytest.raises(UnrestrictedScoringRefused, match="no token id"):
        score_unrestricted_next_token(
            backend, inputs, target_token_ids={"target": None}
        )


# --------------------------------------------------------------------------
# 5. Target full-vocabulary rank is correct, including ties
# --------------------------------------------------------------------------


def test_rank_is_correct_without_ties():
    scores = torch.tensor([0.0, 3.0, 1.0, 2.0])
    assert tie_aware_ranks(scores, 1)["rank"] == 1
    assert tie_aware_ranks(scores, 3)["rank"] == 2
    assert tie_aware_ranks(scores, 2)["rank"] == 3
    assert tie_aware_ranks(scores, 0)["rank"] == 4
    top = tie_aware_ranks(scores, 1)
    assert top["is_unique_maximum"] is True
    assert top["is_tied_maximum"] is True
    assert top["rank_pessimistic"] == 1
    assert top["rank_midrank"] == 1.0


def test_rank_is_tie_aware():
    scores = torch.tensor([3.0, 3.0, 3.0, 1.0])
    row = tie_aware_ranks(scores, 0)
    assert row["rank_optimistic"] == 1
    assert row["rank_pessimistic"] == 3
    assert row["rank_midrank"] == 2.0
    assert row["n_tied_including_self"] == 3
    assert row["is_unique_maximum"] is False
    assert row["is_tied_maximum"] is True
    assert tie_aware_ranks(scores, 3)["rank"] == 4


def test_a_rank_outside_the_vocabulary_is_refused():
    with pytest.raises(UnrestrictedScoringRefused, match="outside the vocabulary"):
        tie_aware_ranks(torch.zeros(4), 9)


def test_ranks_agree_with_the_scorer(backend, inputs):
    ids = answer_token_table(backend, ["two", "four"])["token_ids"]
    scored = score_unrestricted_next_token(
        backend, inputs, target_token_ids={"target": ids["two"], "source": ids["four"]}
    )
    raw = backend.forward_logits(inputs.tensors)[0, inputs.final_prompt_position]
    log_probs = torch.log_softmax(raw.float(), dim=-1)
    expected = int((log_probs > log_probs[ids["two"]]).sum()) + 1
    assert scored["named_tokens"]["target"]["rank"] == expected


# --------------------------------------------------------------------------
# 6/7. Restricted winner may differ from the global argmax, and never GOes
# --------------------------------------------------------------------------


def test_restricted_winner_can_differ_from_the_global_argmax():
    scored = mock_scored(
        target_token_id=MOCK_ANSWERS["four"],
        source_token_id=MOCK_ANSWERS["two"],
        target_is_global_argmax=False,
    )
    restricted = restricted_candidate_top1(
        {"four": {"sum_logprob": -0.1}, "two": {"sum_logprob": -2.0}}, "four"
    )
    record = unrestricted_trial_record(
        scored,
        trial_kind="intervention",
        condition="swap_alpha1",
        arm="intermediate",
        band=[33, 34],
        alpha=1.0,
        modality="text",
        readout="property",
        source_answer="two",
        target_answer="four",
        source_token_id=MOCK_ANSWERS["two"],
        target_token_id=MOCK_ANSWERS["four"],
        group_id="g",
        restricted=restricted,
    )
    assert record["target_is_restricted_candidate_top1"] is True
    assert record["target_is_unique_global_top1"] is False
    assert record["restricted_candidate_diagnostic"]["endpoint"] == (
        ENDPOINT_RESTRICTED_CANDIDATE
    )
    assert record["restricted_candidate_diagnostic"]["is_not_the_model_output"] is True


def test_restricted_only_world_is_a_no_go(design):
    scenario = CAUSAL_SCENARIOS["restricted_only"]
    records = mock_full_vocab_records(
        scenario,
        bands=design["bands"],
        conditions=design["conditions"],
        arms=design["arms"],
    )
    assert any(row["target_is_restricted_candidate_top1"] for row in records)
    assert not any(row["target_is_unique_global_top1"] for row in records)
    cells = summarize_full_vocab_cells(records)
    verdict = unrestricted_reasoning_verdict(
        cells,
        bands=design["band_keys"],
        modalities=design["modalities"],
        directed_pairs=design["directed_pairs"],
    )
    assert verdict["verdict"] == FULL_VOCAB_REASONING_NO_GO
    assert verdict["restricted_candidate_order_alone_is_never_a_go"] is True


def test_a_positive_margin_without_rank_1_is_not_a_success():
    scored = mock_scored(
        target_token_id=MOCK_ANSWERS["four"],
        source_token_id=MOCK_ANSWERS["two"],
        target_is_global_argmax=False,
        target_logprob=-0.20,
        source_logprob=-4.00,
    )
    record = unrestricted_trial_record(
        scored,
        trial_kind="intervention",
        condition="swap_alpha1",
        modality="text",
        readout="property",
        source_answer="two",
        target_answer="four",
        source_token_id=MOCK_ANSWERS["two"],
        target_token_id=MOCK_ANSWERS["four"],
        group_id="g",
    )
    assert record["target_minus_source_logprob"] > 0
    assert record["target_is_unique_global_top1"] is False
    assert record["target_rank"] > 1


# --------------------------------------------------------------------------
# 8/9. Single-token requirement; multi-token answers refused
# --------------------------------------------------------------------------


def test_single_token_answers_resolve(backend):
    row = resolve_answer_token(backend, "two")
    assert row["single_token"] is True
    assert row["n_tokens"] == 1
    assert row["surface"] == " two"


def test_a_multi_token_answer_is_refused_for_paper_comparison(backend):
    with pytest.raises(MultiTokenAnswerError, match="unsupported for this answer"):
        resolve_answer_token(backend, "not a real answer at all")


def test_required_multi_token_answers_refuse_the_study(backend):
    with pytest.raises(MultiTokenAnswerError, match="refused before any model"):
        answer_token_table(
            backend, ["two", "a multi word phrase"], required=["a multi word phrase"]
        )


def test_unsupported_answers_are_reported_not_truncated(backend):
    table = answer_token_table(
        backend, ["two", "four", "a multi word phrase"], required=["two", "four"]
    )
    assert table["all_single_token"] is False
    unsupported = table["unsupported"]["a multi word phrase"]
    assert unsupported["endpoint_supported"] is False
    assert unsupported["n_tokens"] > 1
    assert "a multi word phrase" not in table["token_ids"]


def test_resolve_study_tokens_separates_the_families(backend):
    tokens = resolve_study_tokens(
        backend, three_modality_concepts=("bird", "cat", "a multi word phrase")
    )
    assert tokens["family_a_supported"] is True
    assert tokens["first_token_truncation_used"] is False
    assert tokens["outcome_dependent_replacement_used"] is False
    assert "a multi word phrase" in tokens["family_b_unsupported_for_unrestricted_endpoint"]


def test_ambiguous_tokenization_is_refused(backend, monkeypatch):
    monkeypatch.setattr(backend, "encode_token", lambda text: [11])
    with pytest.raises(UnrestrictedScoringRefused, match="ambiguous"):
        answer_token_table(backend, ["two", "four"])


# --------------------------------------------------------------------------
# 10/11. Clean and intervened use the same scorer; one pass each
# --------------------------------------------------------------------------


def test_clean_and_intervened_use_the_same_unrestricted_scorer():
    source = (ROOT / "scripts" / "_build_full_vocabulary_notebook.py").read_text(
        encoding="utf-8"
    )
    assert source.count("score_unrestricted_next_token(") >= 2
    # There is exactly one scoring function named in the causal cell, and it is
    # never `prediction_and_margin`.
    assert "prediction_and_margin" not in source


def test_one_scoring_forward_pass_per_trial(backend, inputs, monkeypatch):
    calls = {"n": 0}
    real = backend.forward_logits

    def counted(tensors):
        calls["n"] += 1
        return real(tensors)

    monkeypatch.setattr(backend, "forward_logits", counted)
    score_unrestricted_next_token(backend, inputs, top_k=3)
    assert calls["n"] == 1


def test_the_budget_charges_one_pass_per_trial():
    budget = full_vocab_pass_budget(
        a_cells=24, a_bands=4, a_arms=2, a_conditions=7, a_readouts=2
    )
    assert budget["passes_per_trial"] == 1
    assert budget["family_a"]["intervention_and_control_passes"] == 24 * 4 * 2 * 7 * 2
    assert budget["family_a"]["clean_unrestricted_passes"] == 24 * 2


# --------------------------------------------------------------------------
# 12. Greedy generation is secondary and deterministic
# --------------------------------------------------------------------------


def test_greedy_generation_is_deterministic_and_secondary(backend, inputs):
    first = greedy_generate(backend, inputs, max_new_tokens=3, answer="two")
    second = greedy_generate(backend, inputs, max_new_tokens=3, answer="two")
    assert first["generated_token_ids"] == second["generated_token_ids"]
    assert first["temperature"] == 0.0
    assert first["do_sample"] is False
    assert first["deterministic"] is True
    assert first["is_secondary_demonstration"] is True
    assert first["n_forward_passes"] == 3


def test_the_greedy_match_rule_is_frozen_and_normalizes():
    assert normalize_generated_text("  Two. ") == "two"
    assert normalize_generated_text("TWO") == normalize_generated_text("two")
    from jlens.mmpilot.full_vocabulary import greedy_matches

    assert greedy_matches(" two", "two") is True
    assert greedy_matches(" two legs", "two") is True
    assert greedy_matches(" twofold", "two") is False


def test_no_verdict_field_reads_greedy_output(design):
    verdict = unrestricted_reasoning_verdict(
        [],
        bands=design["band_keys"],
        modalities=design["modalities"],
        directed_pairs=design["directed_pairs"],
    )
    assert verdict["greedy_text_alone_is_never_a_go"] is True
    assert "generated_text" not in json.dumps(verdict)


# --------------------------------------------------------------------------
# 13/14. Completed runs stay byte-identical; amendments recompute nothing
# --------------------------------------------------------------------------


def _amendment(audit, **overrides):
    payload = {
        "name": "l33_l40_validated_band_followup_report",
        "study": "L33-L40 validated-band follow-up",
        "original_report_path": "runs/mmband33/band3340_real_2a72bda9b4ba/report.json",
        "original_report_checksum": "sha256:" + "f" * 64,
        "original_run_name": "band3340_real_2a72bda9b4ba",
        "original_run_fingerprint": "sha256:" + "2" * 64,
        "original_verdict": "L33_L40_VALIDATED_BAND_FOLLOWUP_ALPHA2_SENSITIVITY_ONLY",
        "original_endpoint_class": ENDPOINT_RESTRICTED_CANDIDATE,
        "endpoint_audit_digest": audit["audit_digest"],
        "claim_ledger_digest": audit["claim_ledger_digest"],
        "terminology": BAND_FOLLOWUP_TERMINOLOGY,
        "corrected_labels": [
            "RESTRICTED_CANDIDATE_PREFERENCE_GO",
            "FULL_VOCABULARY_NOT_EVALUATED",
        ],
        "written_utc": "2026-08-13T00:00:00+00:00",
    }
    payload.update(overrides)
    return build_endpoint_amendment(**payload)


@pytest.fixture(scope="module")
def audit() -> dict:
    return endpoint_audit_record(repo_root=ROOT)


def test_an_amendment_recomputes_nothing(audit):
    amendment = _amendment(audit)
    assert amendment["scientific_recompute"] == 0
    assert amendment["scientific_numbers_unchanged"] is True
    assert amendment["original_report_modified"] is False
    assert amendment["original_units_modified"] is False
    assert amendment["verdict_changed_by_prose"] is False
    assert amendment["full_vocabulary_evaluated"] is False
    assert (
        amendment["original_verdict"]
        == "L33_L40_VALIDATED_BAND_FOLLOWUP_ALPHA2_SENSITIVITY_ONLY"
    )
    assert amendment["original_verdict_is_reproduced_verbatim"] is True


def test_an_amendment_writes_beside_and_never_over(tmp_path, audit):
    amendment = _amendment(audit)
    first = write_endpoint_amendment(tmp_path, amendment)
    assert first["status"] == "written"
    assert write_endpoint_amendment(tmp_path, amendment)["status"] == "reused"

    different = _amendment(audit, name="l33_l40_validated_band_followup_report",
                           original_report_path="somewhere/else.json")
    with pytest.raises(AmendmentRefused, match="not overwritten"):
        write_endpoint_amendment(tmp_path, different)


def test_amendment_binding_refuses_a_different_measurement(audit):
    amendment = _amendment(audit)
    with pytest.raises(AmendmentRefused, match="original_run_fingerprint"):
        verify_amendment_binding(
            amendment,
            original_report_checksum=amendment["original_report_checksum"],
            original_run_fingerprint="sha256:" + "9" * 64,
            endpoint_audit_digest=audit["audit_digest"],
        )


def test_amendment_labels_are_frozen(audit):
    with pytest.raises(AmendmentRefused, match="undeclared corrected label"):
        _amendment(audit, corrected_labels=["FULL_VOCAB_REASONING_ALPHA1_GO"])
    assert set(CORRECTED_LABELS) == {
        "RESTRICTED_CANDIDATE_PREFERENCE_GO",
        "CONTROLLED_TARGET_LOGPROB_EFFECT",
        "FULL_VOCABULARY_NOT_EVALUATED",
    }


def test_an_already_unrestricted_endpoint_cannot_be_amended(audit):
    with pytest.raises(AmendmentRefused, match="already unrestricted"):
        _amendment(audit, original_endpoint_class=ENDPOINT_UNRESTRICTED_NEXT_TOKEN)


def test_both_required_amendments_exist():
    assert BAND_FOLLOWUP_TERMINOLOGY
    assert THREE_MODALITY_TERMINOLOGY
    band = {row.field_path for row in BAND_FOLLOWUP_TERMINOLOGY}
    assert any("paper_comparable" in path for path in band)
    audio = " ".join(row.corrected_wording for row in THREE_MODALITY_TERMINOLOGY)
    assert "CONTROLLED_TARGET_LOGPROB_EFFECT" in audio
    assert "FULL_VOCABULARY_NOT_EVALUATED" in audio


def test_no_module_writes_into_a_completed_run():
    for name in ("endpoint_amend", "full_vocab_study", "full_vocabulary"):
        source = (ROOT / "jlens" / "mmpilot" / f"{name}.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in (
                "write_text",
                "write_bytes",
            ):
                # Only the amendment writer writes, and only into its argument.
                assert name == "endpoint_amend"


# --------------------------------------------------------------------------
# 15/16. Exact populations reused; missing provenance refuses
# --------------------------------------------------------------------------


def test_the_population_is_reused_not_redrawn():
    report = mock_band_followup_report(fingerprint_pin="sha256:" + "b" * 64)
    reuse = reuse_completed_population(report)
    assert reuse["reused_verbatim"] is True
    assert reuse["redrawn"] is False
    assert reuse["enlarged"] is False
    assert reuse["capability_reselected"] is False
    assert reuse["is_an_independent_replication"] is False
    assert reuse["group_ids"] == [
        row["group_id"] for row in report["population_groups"]
    ]


def test_no_population_selection_function_is_reachable():
    """The names may be *mentioned* in prose; none of them may be *called*."""
    tree = ast.parse(
        (ROOT / "jlens" / "mmpilot" / "full_vocab_study.py").read_text(encoding="utf-8")
    )
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    } | {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    for name in (
        "hidden_animal_population",
        "select_capability_eligible_samples",
        "stable_rank",
        "visual_evidence",
    ):
        assert name not in called, name
        assert name not in imported, name


@pytest.mark.parametrize("key", sorted(PROVENANCE_SCENARIOS))
def test_provenance_scenarios_behave_as_predeclared(tmp_path, key):
    scenario = PROVENANCE_SCENARIOS[key]
    report = mock_band_followup_report(fingerprint_pin="sha256:" + "b" * 64)
    pin = report["report_checksum"]
    mutated = scenario.mutate(dict(report))
    run_dir = tmp_path / key
    run_dir.mkdir()
    (run_dir / FOLLOWUP_REPORT_NAME).write_text(json.dumps(mutated), encoding="utf-8")
    try:
        _, read = read_band_followup_report(
            run_dir,
            expected_report_checksum=pin,
            expected_fingerprint="sha256:" + "b" * 64,
        )
        reuse_completed_population(read)
        refused = False
    except FullVocabRefused:
        refused = True
    assert refused is scenario.expect_refusal, scenario.because


def test_missing_historical_prompts_refuse_rather_than_approximate(tmp_path):
    with pytest.raises(FullVocabRefused, match="refused rather than approximated"):
        read_historical_prompt_hashes(tmp_path)


def test_a_rebuilt_prompt_must_hash_to_the_historical_one():
    historical = {"prompt_hashes": {"g1|text|property": "abc123"}}
    ok = assert_prompt_reconstruction(
        group_id="g1",
        modality="text",
        readout="property",
        rebuilt_prompt_hash="abc123",
        historical=historical,
    )
    assert ok["matches_completed_run"] is True
    with pytest.raises(FullVocabRefused, match="different prompt"):
        assert_prompt_reconstruction(
            group_id="g1",
            modality="text",
            readout="property",
            rebuilt_prompt_hash="deadbeef",
            historical=historical,
        )
    with pytest.raises(FullVocabRefused, match="refused rather than approximated"):
        assert_prompt_reconstruction(
            group_id="g9",
            modality="text",
            readout="property",
            rebuilt_prompt_hash="abc123",
            historical=historical,
        )


def test_an_empty_pin_refuses_and_names_itself():
    assert CANONICAL_AUDIO_REPORT_CHECKSUM_PIN == ""
    with pytest.raises(FullVocabRefused, match="CANONICAL_AUDIO_REPORT_CHECKSUM"):
        require_pin("CANONICAL_AUDIO_REPORT_CHECKSUM")
    with pytest.raises(FullVocabRefused, match="vouches for itself"):
        require_pin("CANONICAL_AUDIO_AMENDED_SUMMARY_CHECKSUM")
    # Family B therefore refuses before any model spending.
    with pytest.raises(FullVocabRefused, match="CANONICAL_AUDIO_REPORT_CHECKSUM"):
        read_canonical_audio_provenance("/nonexistent")


def test_family_a_pins_are_set():
    assert REQUIRED_PINS["BAND_FOLLOWUP_REPORT_CHECKSUM"]["value"].startswith("sha256:")
    assert REQUIRED_PINS["BAND_FOLLOWUP_FINGERPRINT"]["value"].startswith("sha256:")
    assert require_pin("BAND_FOLLOWUP_REPORT_CHECKSUM").endswith("cdcc1ce263cb")


# --------------------------------------------------------------------------
# 17/18. L32 excluded; hook integrity exact
# --------------------------------------------------------------------------


def test_l32_is_excluded_categorically(design):
    assert EXCLUDED_FAILED_LAYER == 32
    assert all(32 not in band for band in design["bands"])
    assert design["excluded_layer"] == 32
    # The topology guard fires first for any start set that is not the
    # completed study's.
    with pytest.raises(FullVocabRefused, match="not the completed follow-up"):
        full_vocab_design_record(suffix_starts=(32, 35, 38, 40))


def test_the_l32_band_guard_is_not_dead_code(monkeypatch):
    """If the frozen topology ever changed, the second guard still refuses L32."""
    import jlens.mmpilot.full_vocab_study as module

    monkeypatch.setattr(module, "FOLLOWUP_SUFFIX_STARTS", (32, 35, 38, 40))
    with pytest.raises(FullVocabRefused, match="excluded layer"):
        module.full_vocab_design_record(suffix_starts=(32, 35, 38, 40))


def test_band_hook_integrity_is_the_completed_studys_function():
    from jlens.mmpilot.validated_band_followup import assert_band_hook_integrity

    source = (ROOT / "scripts" / "_build_full_vocabulary_notebook.py").read_text(
        encoding="utf-8"
    )
    assert "assert_band_hook_integrity" in source
    assert callable(assert_band_hook_integrity)


# --------------------------------------------------------------------------
# 19/20. No teacher-forced position is the primary endpoint; no leakage
# --------------------------------------------------------------------------


def test_the_primary_endpoint_never_scores_a_teacher_forced_position(backend, inputs):
    ids = answer_token_table(backend, ["two", "four"])["token_ids"]
    scored = score_unrestricted_next_token(
        backend,
        inputs,
        target_token_ids={"target": ids["four"], "source": ids["two"]},
        top_k=3,
    )
    assert scored["endpoint"] == ENDPOINT_UNRESTRICTED_NEXT_TOKEN
    assert scored["n_candidate_positions_appended"] == 0
    record = unrestricted_trial_record(
        scored,
        trial_kind="clean",
        condition="clean",
        modality="text",
        readout="property",
        source_answer="two",
        target_answer="four",
        source_token_id=ids["two"],
        target_token_id=ids["four"],
        group_id="g",
    )
    assert record["n_candidate_positions_appended"] == 0
    assert record["scored_input_length"] == record["prompt_len"]


def test_transcript_leakage_remains_impossible():
    source = (ROOT / "scripts" / "_build_full_vocabulary_notebook.py").read_text(
        encoding="utf-8"
    )
    assert "build_backend_inputs(BACKEND, built, transcript=offline)" in source
    assert 'offline = group["caption"] if modality != "text" else None' in source


# --------------------------------------------------------------------------
# 21/22. Alphas matched separately; verdicts cannot overwrite each other
# --------------------------------------------------------------------------


def test_alpha1_and_alpha2_controls_are_matched_separately(design):
    records = mock_full_vocab_records(
        CAUSAL_SCENARIOS["alpha2_only"],
        bands=design["bands"],
        conditions=design["conditions"],
        arms=design["arms"],
    )
    cells = summarize_full_vocab_cells(records)
    verdict = unrestricted_reasoning_verdict(
        cells,
        bands=design["band_keys"],
        modalities=design["modalities"],
        directed_pairs=design["directed_pairs"],
    )
    assert verdict["verdict"] == FULL_VOCAB_REASONING_ALPHA2_ONLY
    assert verdict["alpha2_is_never_primary_evidence"] is True
    alpha2 = [row for row in verdict["per_cell"] if row["condition"] == "swap_alpha2"]
    for row in alpha2:
        if not row.get("evaluated"):
            continue
        assert row["controls"]["random"]["condition"] == "random_alpha2"
        assert row["controls"]["unrelated"]["condition"] == "unrelated_alpha2"
    alpha1 = [
        row
        for row in verdict["per_cell"]
        if row["condition"] == "swap_alpha1" and row.get("evaluated")
    ]
    for row in alpha1:
        assert row["controls"]["random"]["condition"] == "random_alpha1"


def test_the_two_verdict_families_are_separate(design):
    records = mock_full_vocab_records(
        CAUSAL_SCENARIOS["full_vocab_go"],
        bands=design["bands"],
        conditions=design["conditions"],
        arms=design["arms"],
    )
    unrestricted = unrestricted_reasoning_verdict(
        summarize_full_vocab_cells(records),
        bands=design["band_keys"],
        modalities=design["modalities"],
        directed_pairs=design["directed_pairs"],
    )
    conditional = conditional_logprob_verdict(None, evaluated=False)
    assert unrestricted["endpoint"] == ENDPOINT_UNRESTRICTED_NEXT_TOKEN
    assert conditional["endpoint"] != ENDPOINT_UNRESTRICTED_NEXT_TOKEN
    assert conditional["cannot_overwrite_the_full_vocabulary_verdict"] is True
    assert conditional["is_not_an_unrestricted_output_verdict"] is True
    conjunction = cross_modal_conjunction(unrestricted, conditional)
    assert conjunction["unrestricted_verdict"] == unrestricted["verdict"]
    assert conjunction["conditional_logprob_verdict"] == conditional["verdict"]
    assert conjunction[
        "the_two_verdicts_are_separate_and_neither_substitutes_for_the_other"
    ]


def test_every_verdict_name_is_declared():
    assert set(VERDICT_NAMES) == {
        "FULL_VOCAB_REASONING_ALPHA1_GO",
        "FULL_VOCAB_REASONING_ALPHA2_ONLY",
        "FULL_VOCAB_REASONING_NO_GO",
        "FULL_VOCAB_THREE_MODALITY_CAUSAL_GO",
        "CONDITIONAL_LOGPROB_ONLY",
        "CAPABILITY_NO_GO",
        "NOT_EVALUATED",
        "INCONCLUSIVE",
    }


def test_one_modality_never_stands_in_for_three(design):
    records = [
        row
        for row in mock_full_vocab_records(
            CAUSAL_SCENARIOS["full_vocab_go"],
            bands=design["bands"],
            conditions=design["conditions"],
            arms=design["arms"],
            modalities=("text",),
        )
    ]
    unrestricted = unrestricted_reasoning_verdict(
        summarize_full_vocab_cells(records),
        bands=design["band_keys"],
        modalities=design["modalities"],
        directed_pairs=design["directed_pairs"],
    )
    conjunction = cross_modal_conjunction(
        unrestricted, conditional_logprob_verdict(None, evaluated=False)
    )
    assert unrestricted["verdict"] == FULL_VOCAB_REASONING_ALPHA1_GO
    assert conjunction["conjunction_complete"] is False
    assert conjunction["verdict"] != "FULL_VOCAB_THREE_MODALITY_CAUSAL_GO"


def test_the_direct_answer_arm_alone_is_never_a_go(design):
    records = mock_full_vocab_records(
        CAUSAL_SCENARIOS["direct_answer_only"],
        bands=design["bands"],
        conditions=design["conditions"],
        arms=design["arms"],
    )
    verdict = unrestricted_reasoning_verdict(
        summarize_full_vocab_cells(records),
        bands=design["band_keys"],
        modalities=design["modalities"],
        directed_pairs=design["directed_pairs"],
    )
    assert verdict["verdict"] == FULL_VOCAB_REASONING_NO_GO
    assert verdict["direct_answer_arm"]["role"] == "positive control"
    assert verdict["direct_answer_arm"]["alpha1_passing_bands"]
    assert verdict["direct_answer_arm_alone_is_never_a_go"] is True


def test_every_direction_is_reported_before_any_pooled_summary(design):
    verdict = unrestricted_reasoning_verdict(
        summarize_full_vocab_cells(
            mock_full_vocab_records(
                CAUSAL_SCENARIOS["asymmetric_direction"],
                bands=design["bands"],
                conditions=design["conditions"],
                arms=design["arms"],
            )
        ),
        bands=design["band_keys"],
        modalities=design["modalities"],
        directed_pairs=design["directed_pairs"],
    )
    per_direction = {row["pair"]: row for row in verdict["per_direction"]}
    assert per_direction["bird->cat"]["alpha1_passing_bands"]
    assert per_direction["cat->bird"]["alpha1_passing_bands"] == []
    assert verdict["pooled_summary_follows_per_direction_reporting"] is True


# --------------------------------------------------------------------------
# 23/24. Resume trial by trial; a changed configuration refuses stale units
# --------------------------------------------------------------------------


def test_the_trial_plan_has_one_unique_key_per_trial(design):
    trials = family_a_trials(mock_population_reuse(), design)
    keys = [row["key"] for row in trials]
    assert len(set(keys)) == len(keys)
    clean = [row for row in trials if row["kind"] == "clean"]
    intervention = [row for row in trials if row["kind"] == "intervention"]
    assert len(clean) == 24 * 2
    assert len(intervention) == 24 * 4 * 2 * 7 * 2
    # Clean trials come first, so a resume never scores an edit whose clean
    # comparison unit is missing.
    assert all(row["kind"] == "clean" for row in trials[: len(clean)])


def _fingerprint(design, **overrides):
    payload = {
        "design": design,
        "endpoint_audit_digest": "sha256:audit",
        "band_followup_report_checksum": "sha256:report",
        "band_followup_fingerprint": "sha256:fp",
        "canonical_audio_provenance": None,
        "population_reuse": mock_population_reuse(),
        "lens_checksums": {layer: f"sha256:l{layer}" for layer in range(33, 41)},
        "media_checksums": {},
        "model_repo_id": "google/gemma-4-E4B-it",
        "model_revision": "main",
        "processor_revision": "main",
        "transformers_version": "5.13.1",
        "audio_protocol_fingerprint": "sha256:audio",
        "prompt_protocol": None,
        "tokens": resolve_study_tokens(SwapMockBackend()),
        "coordinate_swap_method_version": "jlens.mmpilot.coordinate_swap.v1",
        "thresholds": FullVocabThresholds().to_dict(),
        "seeds": {"a": 1},
        "output_head_convention": "model_forward_logits",
    }
    payload.update(overrides)
    return full_vocab_fingerprint(**payload)


def test_a_relevant_configuration_change_refuses_stale_units(design):
    base = _fingerprint(design)
    same = _fingerprint(design)
    assert base["full_vocab_fingerprint_digest"] == same["full_vocab_fingerprint_digest"]

    for field, value in (
        ("endpoint_audit_digest", "sha256:different-audit"),
        ("band_followup_report_checksum", "sha256:different-report"),
        ("output_head_convention", "hand_rolled_unembedding"),
        ("model_revision", "some-other-revision"),
        ("audio_protocol_fingerprint", "sha256:other-audio"),
    ):
        changed = _fingerprint(design, **{field: value})
        assert (
            changed["full_vocab_fingerprint_digest"]
            != base["full_vocab_fingerprint_digest"]
        ), field

    thresholds = FullVocabThresholds(min_unique_global_top1_rate=0.9)
    changed = _fingerprint(
        full_vocab_design_record(thresholds=thresholds),
        thresholds=thresholds.to_dict(),
    )
    assert (
        changed["full_vocab_fingerprint_digest"] != base["full_vocab_fingerprint_digest"]
    )


def test_a_missing_endpoint_audit_digest_refuses(design):
    with pytest.raises(FullVocabRefused, match="endpoint-audit digest"):
        _fingerprint(design, endpoint_audit_digest="")


def test_a_missing_lens_checksum_refuses(design):
    with pytest.raises(FullVocabRefused, match="no lens checksum"):
        _fingerprint(design, lens_checksums={33: "sha256:l33"})


# --------------------------------------------------------------------------
# 25/26. CPU stages load no model; no fitting entry point is reachable
# --------------------------------------------------------------------------


def test_no_study_module_imports_a_model_loader():
    for name in (
        "full_vocabulary",
        "full_vocab_study",
        "full_vocab_mock",
        "endpoint_audit",
        "endpoint_amend",
    ):
        source = (ROOT / "jlens" / "mmpilot" / f"{name}.py").read_text(encoding="utf-8")
        assert "import transformers" not in source
        assert "from transformers" not in source
        assert "real_backend" not in source
        assert "AutoProcessor" not in source


def test_no_fitting_entry_point_is_reachable():
    for name in ("full_vocabulary", "full_vocab_study", "endpoint_amend"):
        source = (ROOT / "jlens" / "mmpilot" / f"{name}.py").read_text(encoding="utf-8")
        for forbidden in ("fit_lens", "jlens.fitting", "accumulate_jacobian", "backward("):
            assert forbidden not in source, (name, forbidden)


def test_the_budget_records_zero_backward_passes():
    budget = full_vocab_pass_budget(
        a_cells=24, a_bands=4, a_arms=2, a_conditions=7, a_readouts=2
    )
    assert budget["backward_passes"] == 0
    assert budget["fitting_performed"] is False
    assert budget["jacobian_accumulation"] is False


# --------------------------------------------------------------------------
# 27. The derived budget is correct and below the cap
# --------------------------------------------------------------------------


def test_the_derived_budget_is_correct_and_under_the_cap(design):
    trials = family_a_trials(mock_population_reuse(), design)
    clean = [row for row in trials if row["kind"] == "clean"]
    greedy = [
        row
        for row in trials
        if row["modality"] == "text"
        and row["readout"] == "property"
        and row["condition"] in ("clean", "swap_alpha1", "zero")
        and (row["band"] is None or row["band"] == list(range(33, 41)))
        and row["arm"] in (None, "intermediate")
    ]
    budget = full_vocab_pass_budget(
        a_cells=len(clean) // len(design["readouts"]),
        a_bands=len(design["bands"]),
        a_arms=len(design["arms"]),
        a_conditions=len(design["conditions"]),
        a_readouts=len(design["readouts"]),
        greedy_trials=len(greedy),
        greedy_max_new_tokens=4,
    )
    assert budget["total"] == 48 + 2688 + len(greedy) * 4
    assert budget["total"] < PASS_CAP
    assert budget["within_cap"] is True
    assert budget["cap"] == 5000


def test_an_over_cap_budget_stops_and_names_the_factor():
    budget = full_vocab_pass_budget(
        a_cells=24, a_bands=4, a_arms=2, a_conditions=7, a_readouts=2,
        b_claim_supporting_cells=20, b_samples_per_cell=20, b_layers=5,
        b_conditions=5, b_clean_inputs=400,
    )
    assert budget["within_cap"] is False
    assert budget["excess"] > 0
    assert budget["excess_driven_by"] in budget["contributions"]
    assert budget["smallest_lossless_reduction"]
    assert budget["no_condition_is_silently_dropped"] is True


# --------------------------------------------------------------------------
# 28. Every MOCK scenario returns its predeclared verdict
# --------------------------------------------------------------------------


@pytest.mark.parametrize("key", sorted(CAUSAL_SCENARIOS))
def test_mock_scenarios_return_their_predeclared_verdicts(key, design):
    scenario = CAUSAL_SCENARIOS[key]
    records = mock_full_vocab_records(
        scenario,
        bands=design["bands"],
        conditions=design["conditions"],
        arms=design["arms"],
    )
    cells = summarize_full_vocab_cells(records)
    verdict = unrestricted_reasoning_verdict(
        cells,
        bands=design["band_keys"],
        modalities=design["modalities"],
        directed_pairs=design["directed_pairs"],
        capability_sufficient=scenario.capability_sufficient,
        causal_stage_ran=scenario.causal_stage_ran,
    )
    assert verdict["verdict"] == scenario.expected_verdict, scenario.note


def test_the_scenario_set_covers_every_commissioned_case():
    assert set(CAUSAL_SCENARIOS) == {
        "full_vocab_go",
        "alpha2_only",
        "restricted_only",
        "direct_answer_only",
        "asymmetric_direction",
        "control_failure",
        "capability_no_go",
        "null",
    }
    assert "no_population" in PROVENANCE_SCENARIOS


# --------------------------------------------------------------------------
# 30. The claim ledger traces every active headline claim to a function
# --------------------------------------------------------------------------


def test_every_ledger_row_resolves_to_a_real_function():
    resolved = verify_registry()
    assert len(resolved) == len(AUDITED_ENDPOINTS)
    assert all(row["callable"] for row in resolved)


def test_the_ledger_covers_every_commissioned_study():
    studies = " ".join(row.study for row in AUDITED_ENDPOINTS).lower()
    for required in (
        "audio",
        "capability",
        "calibration",
        "retrieval",
        "pilot",
        "robustness",
        "localization",
        "spoken-audio transfer",
        "reasoning swap",
        "l33-l40",
        "convergence",
    ):
        assert required in studies, required


def test_the_ledger_marks_the_restricted_claims_for_revalidation():
    ledger = endpoint_audit_record(repo_root=ROOT)["claim_ledger"]
    rows = {row["claim"]: row for row in ledger["rows"]}
    band = rows["L33_L40_VALIDATED_BAND_FOLLOWUP_ALPHA2_SENSITIVITY_ONLY"]
    assert band["endpoint_class"] == ENDPOINT_RESTRICTED_CANDIDATE
    assert band["global_vocabulary_considered"] is False
    assert band["tokens_appended"] is True
    assert band["requires_full_vocabulary_revalidation"] is True
    assert band["source_run"] == "band3340_real_2a72bda9b4ba"
    assert band["source_report_checksum_configured"] is False  # no pins passed

    lens = rows["the J-lens reproduces the model's own next token"]
    assert lens["endpoint_class"] == ENDPOINT_UNRESTRICTED_NEXT_TOKEN
    assert lens["requires_full_vocabulary_revalidation"] is False


def test_the_ledger_carries_configured_checksums_when_pins_are_set():
    ledger = endpoint_audit_record(
        repo_root=ROOT,
        report_checksums={
            name: row["value"] for name, row in REQUIRED_PINS.items() if row["value"]
        },
    )["claim_ledger"]
    rows = {row["claim"]: row for row in ledger["rows"]}
    band = rows["L33_L40_VALIDATED_BAND_FOLLOWUP_ALPHA2_SENSITIVITY_ONLY"]
    assert band["source_report_checksum_configured"] is True
    assert band["source_report_checksum"].endswith("cdcc1ce263cb")
    # The canonical run's report checksum pin is empty, so its rows stay unpinned.
    audio = rows["THREE_MODALITY_GO — cross-modal causal transfer including speech"]
    assert audio["source_report_checksum_configured"] is False


def test_the_active_source_scan_finds_no_unqualified_overclaim():
    scan = scan_active_sources(ROOT)
    assert scan["passed"] is True, scan["overclaims"]
    assert scan["n_files_scanned"] >= 10
    assert scan["modules_not_found"] == []


def test_the_audit_record_binds_and_passes():
    record = endpoint_audit_record(repo_root=ROOT)
    assert record["passed"] is True
    assert record["scientific_recompute"] == 0
    assert record["completed_reports_untouched"] is True
    assert record["audit_digest"] == payload_checksum(
        {k: v for k, v in record.items() if k != "audit_digest"}
    )
