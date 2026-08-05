# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Deterministic CPU tests for the research-grade calibration package.

Everything here runs on the tiny mock model in
:mod:`jlens.calibration.mock`. No Gemma, no Hub, no Drive, no corpus download —
and one test enforces that by making those imports raise.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

from jlens.calibration.baseline import BASELINE_ARTIFACTS, baseline_manifest
from jlens.calibration.corpus import (
    CorpusLeakageError,
    CorpusRecord,
    audit_leakage,
    build_partitions,
    build_records,
    corpus_manifest,
    hamming_distance,
    nested_subset,
    normalize_text,
    normalized_sha256,
    record_id_for,
    scale_nesting_audit,
    simhash64,
)
from jlens.calibration.fitting import (
    ScaleNotReachedError,
    filter_records_by_tokens,
    run_calibration,
)
from jlens.calibration.gate import (
    CALIBRATION_GATE,
    CalibrationGate,
    InsufficientTargetDiversityError,
    audit_target_diversity,
    eligible_layers,
    evaluate_calibration_layers,
    gate_text,
    select_diverse_validation_prompts,
)
from jlens.calibration.mock import (
    MOCK_LAYERS,
    MOCK_SCALE_POINTS,
    MockCalibrationModel,
    mock_corpus_texts,
    mock_validation_rows,
)
from jlens.calibration.plan import (
    CALIBRATION_LAYERS,
    OPTIONAL_SCALE_POINTS,
    SCALE_POINTS,
    build_capture_plan,
    estimate_budget,
    format_budget,
    normalized_depth,
)
from jlens.calibration.publication import (
    ConfirmationLocked,
    ConfirmationVault,
    PublicationRefused,
    publication_summary,
    publish_layer,
    record_failed_layer,
)
from jlens.calibration.report import (
    calibration_report_markdown,
    calibration_report_payload,
)
from jlens.calibration.scale import (
    PLATEAU_REACHED,
    PLATEAU_RULE,
    PLATEAU_STILL_IMPROVING,
    compare_scales,
    evaluate_plateau,
    select_scale,
)
from jlens.calibration.state import (
    CALIBRATION_STAGES,
    CalibrationFingerprint,
    CalibrationStore,
    IncompatibleStateError,
)
from jlens.fitting import fit
from jlens.lens import JacobianLens

REPO_ROOT = Path(__file__).resolve().parent.parent

MOCK_FIT_LAYERS = [2, 4, 6, 8]
MOCK_TARGET_LAYER = 11
MOCK_SKIP_FIRST = 4
MOCK_SEQ_LEN = 48


# ------------------------------------------------------------------ fixtures


@pytest.fixture(scope="module")
def model():
    # A 12-block stand-in keeps these tests fast; the notebook's MOCK run uses
    # the module default (42 blocks, the real depth).
    return MockCalibrationModel(n_layers=12)


@pytest.fixture(scope="module")
def records():
    return build_records("mock/train", mock_corpus_texts(400), min_chars=100)


@pytest.fixture(scope="module")
def partitions(records):
    return build_partitions(
        records, corpus_id="mock/train", n_validation=32, n_confirmation=32
    )


@pytest.fixture(scope="module")
def plan():
    return build_capture_plan(
        layers=MOCK_FIT_LAYERS,
        target_layer=MOCK_TARGET_LAYER,
        d_model=32,
        dim_batch=8,
        max_seq_len=MOCK_SEQ_LEN,
        skip_first=MOCK_SKIP_FIRST,
        n_layers=12,
    )


def _fingerprint(plan, **overrides):
    fields = {
        "mode": "mock",
        "protocol_version": "research-grade-multilayer-text-jlens-calibration-v1",
        "model_repo_id": "mock",
        "model_revision": "0" * 40,
        "tokenizer_revision": "0" * 40,
        "capture_plan_digest": plan.digest,
        "corpus_manifest_checksum": "sha256:corpus",
        "gate_digest": CALIBRATION_GATE.digest,
        "plateau_rule_digest": PLATEAU_RULE.digest,
        "scale_points": (8, 16, 32),
        "artifact_format_version": "jlens.calibration.artifact.v1",
    }
    fields.update(overrides)
    return CalibrationFingerprint(**fields)


@pytest.fixture(scope="module")
def validation_by_scale():
    return {
        scale: evaluate_calibration_layers(
            mock_validation_rows(scale=scale, scale_index=index, n_prompts=128),
            layers=MOCK_LAYERS,
            scale=scale,
        )
        for index, scale in enumerate(MOCK_SCALE_POINTS)
    }


@pytest.fixture(scope="module")
def comparison(validation_by_scale):
    return compare_scales(validation_by_scale, layers=MOCK_LAYERS)


# ------------------------------------------------------- corpus normalization


def test_normalization_is_nfkc_lowercase_and_whitespace_collapsed():
    assert normalize_text("  The\tQuick\n\nBrown  ") == "the quick brown"
    # NFKC folds the compatibility ligature and the fullwidth digit.
    assert normalize_text("ﬁle １") == "file 1"


def test_normalized_checksum_ignores_only_normalization_differences():
    assert normalized_sha256("Hello   World") == normalized_sha256("hello world")
    assert normalized_sha256("hello world") != normalized_sha256("hello worlds")


def test_record_ids_are_stable_and_carry_the_corpus():
    assert record_id_for("wikitext-103-raw-v1/train", 42) == (
        "wikitext-103-raw-v1/train/42"
    )
    record = CorpusRecord.build("c/train", 7, "some text here")
    assert record.record_id == "c/train/7"
    assert record.normalized_checksum.startswith("sha256:")
    assert "text" not in record.to_dict()  # corpus text is never serialized


def test_stream_index_counts_every_text_not_just_kept_ones():
    kept = build_records("c/train", ["short", "x" * 200, "tiny"], min_chars=100)
    assert [record.stream_index for record in kept] == [1]


def test_simhash_is_deterministic_and_near_duplicates_are_close():
    base = " ".join(f"word{i}" for i in range(200))
    tweaked = base + " and one more clause at the end"
    unrelated = " ".join(f"other{i}" for i in range(200))
    assert simhash64(base) == simhash64(base)
    assert hamming_distance(simhash64(base), simhash64(tweaked)) <= 8
    assert hamming_distance(simhash64(base), simhash64(unrelated)) > 8


# -------------------------------------------------------------------- splits


def test_partitions_are_disjoint_and_deterministic(records):
    first = build_partitions(records, corpus_id="mock/train")
    second = build_partitions(records, corpus_id="mock/train")
    assert [r.record_id for r in first.fit] == [r.record_id for r in second.fit]
    ids = {
        name: {record.record_id for record in first.get(name)}
        for name in ("fit", "validation", "confirmation")
    }
    assert not ids["fit"] & ids["validation"]
    assert not ids["fit"] & ids["confirmation"]
    assert not ids["validation"] & ids["confirmation"]


def test_partition_membership_does_not_depend_on_how_many_records_were_drawn(records):
    full = build_partitions(records, corpus_id="mock/train")
    half = build_partitions(records[:200], corpus_id="mock/train")
    full_validation = {r.record_id for r in full.validation}
    for record in half.validation:
        assert record.record_id in full_validation


def test_exact_duplicates_are_collapsed_before_bucketing():
    texts = ["alpha " * 60, "alpha " * 60, "beta " * 60]
    records = build_records("c/train", texts, min_chars=10)
    partitions = build_partitions(records, corpus_id="c/train")
    assert len(partitions.dropped_exact_duplicates) == 1
    everything = [
        record.record_id
        for name in ("fit", "validation", "confirmation")
        for record in partitions.get(name)
    ]
    assert len(everything) == len(set(everything)) == 2


def test_short_partition_is_refused_rather_than_shrunk(records):
    with pytest.raises(ValueError, match="requires exactly"):
        build_partitions(records, corpus_id="mock/train", n_validation=10_000)


def test_partition_checksums_change_with_content(records, partitions):
    other = build_partitions(records, corpus_id="mock/train", seed=99)
    assert partitions.checksum("fit") != other.checksum("fit")
    assert partitions.manifest()["manifest_checksum"]


# ------------------------------------------------------------------- leakage


def test_clean_split_passes_the_leakage_audit(partitions):
    report = audit_leakage(partitions, scale_points=[8, 16, 32])
    assert report["ok"] and report["n_exact_hits"] == 0 and report["n_near_hits"] == 0


def test_exact_duplicate_across_partitions_is_refused(partitions):
    poisoned = type(partitions)(
        fit=partitions.fit,
        validation=(*partitions.validation, partitions.fit[0]),
        confirmation=partitions.confirmation,
        corpus_id=partitions.corpus_id,
        split_seed=partitions.split_seed,
    )
    with pytest.raises(CorpusLeakageError, match="exact"):
        audit_leakage(poisoned, scale_points=[8])


def test_near_duplicate_across_partitions_is_refused(partitions):
    original = partitions.fit[0]
    twin = CorpusRecord.build("mock/train", 99_999, original.text + " x")
    poisoned = type(partitions)(
        fit=partitions.fit,
        validation=(*partitions.validation, twin),
        confirmation=partitions.confirmation,
        corpus_id=partitions.corpus_id,
        split_seed=partitions.split_seed,
    )
    with pytest.raises(CorpusLeakageError, match="near|exact"):
        audit_leakage(poisoned, scale_points=[8])


# -------------------------------------------------------------- nested scales


def test_scale_points_are_exact_prefixes(partitions):
    audit = scale_nesting_audit(partitions.fit, [8, 16, 32])
    assert audit["nested"] is True
    assert all(pair["is_prefix"] for pair in audit["pairs"])
    assert nested_subset(partitions.fit, 8) == partitions.fit[:8]


def test_scale_point_larger_than_the_corpus_is_refused(partitions):
    with pytest.raises(ValueError, match="needs"):
        nested_subset(partitions.fit, 10_000_000)


# --------------------------------------------------------- target diversity


def _target_for(prompt: str) -> int:
    return int(prompt.split()[1]) % 40


def test_diversity_audit_reports_distinct_count_and_dominance():
    report = audit_target_diversity(list(range(40)) * 3)
    assert report["passed"] and report["n_distinct_target_tokens"] == 40
    dominated = audit_target_diversity([1] * 100 + list(range(2, 40)))
    assert dominated["distinct_ok"] and not dominated["share_ok"]
    assert not dominated["passed"]


def test_selection_refuses_rather_than_lowering_the_distinct_floor():
    pool = [f"record {i} " + "word " * 50 for i in range(64)]
    with pytest.raises(InsufficientTargetDiversityError, match="distinct"):
        select_diverse_validation_prompts(
            pool,
            n_prompts=32,
            target_token_for_prompt=lambda prompt: 7,  # every prompt: same target
        )


def test_selection_refuses_a_dominated_set():
    """Enough distinct targets to clear the count floor, but one dominates.

    This is the case the added clause exists for: a distinct-count floor alone
    would pass a sample that is half one common token.
    """
    pool = [f"record {i} " + "word " * 50 for i in range(400)]

    def dominated(prompt: str) -> int:
        index = int(prompt.split()[1])
        return 0 if index % 2 == 0 else (index % 60) + 1

    with pytest.raises(InsufficientTargetDiversityError, match="dominated"):
        select_diverse_validation_prompts(
            pool, n_prompts=128, target_token_for_prompt=dominated
        )


def test_selection_is_deterministic_and_records_its_choices():
    pool = [f"record {i} " + "word " * 50 for i in range(400)]
    first, manifest = select_diverse_validation_prompts(
        pool, n_prompts=128, target_token_for_prompt=_target_for
    )
    second, _ = select_diverse_validation_prompts(
        pool, n_prompts=128, target_token_for_prompt=_target_for
    )
    assert first == second
    assert len(first) == 128
    assert manifest["selected_by_jlens_performance"] is False
    assert manifest["diversity"]["passed"]
    assert {"prompt_sha256", "target_token_id", "stable_order"} <= set(
        manifest["prompts"][0]
    )


def test_selection_signature_cannot_accept_a_lens():
    import inspect

    parameters = set(inspect.signature(select_diverse_validation_prompts).parameters)
    assert not {"lens", "jacobian_lens", "layer", "layers"} & parameters


# ---------------------------------------------------- multilayer capture/fit


def test_one_forward_pass_captures_every_configured_layer(model, plan, partitions):
    forwards = {"count": 0}
    original = model.forward

    def counting_forward(input_ids):
        forwards["count"] += 1
        return original(input_ids)

    model.forward = counting_forward
    try:
        lens = fit(
            model,
            [record.text for record in partitions.fit[:3]],
            source_layers=list(plan.layers),
            target_layer=plan.target_layer,
            dim_batch=plan.dim_batch,
            max_seq_len=plan.max_seq_len,
            skip_first=plan.skip_first,
            checkpoint_path=None,
        )
    finally:
        model.forward = original
    assert forwards["count"] == 3  # one forward per prompt, not one per layer
    assert lens.source_layers == sorted(plan.layers)
    assert plan.layers_per_forward == tuple(sorted(plan.layers))


def test_each_layer_gets_its_own_matrix_with_no_sharing(model, plan, partitions):
    lens = fit(
        model,
        [record.text for record in partitions.fit[:3]],
        source_layers=list(plan.layers),
        target_layer=plan.target_layer,
        dim_batch=plan.dim_batch,
        max_seq_len=plan.max_seq_len,
        skip_first=plan.skip_first,
        checkpoint_path=None,
    )
    matrices = [lens.jacobians[layer] for layer in plan.layers]
    for index, matrix in enumerate(matrices):
        assert matrix.shape == (plan.d_model, plan.d_model)
        for other in matrices[index + 1 :]:
            assert matrix.data_ptr() != other.data_ptr()  # not the same storage
            assert not torch.equal(matrix, other)  # and not equal by value


def test_filter_by_tokens_drops_only_prompts_with_no_valid_position(model):
    records = build_records("c/train", ["a b c", "word " * 40], min_chars=1)
    kept, dropped = filter_records_by_tokens(
        records, token_count=model.tokenizer.token_count, skip_first=4, max_seq_len=48
    )
    assert len(kept) == 1 and len(dropped) == 1
    assert dropped[0]["minimum_required"] == 6


# --------------------------------------------------- scale snapshots + resume


@pytest.fixture(scope="module")
def calibrated(model, plan, partitions, tmp_path_factory):
    root = tmp_path_factory.mktemp("calib") / "run"
    store = CalibrationStore(root, _fingerprint(plan))
    store.open()
    kept, _ = filter_records_by_tokens(
        partitions.fit,
        token_count=model.tokenizer.token_count,
        skip_first=plan.skip_first,
        max_seq_len=plan.max_seq_len,
    )
    result = run_calibration(
        model,
        kept,
        plan=plan,
        scale_points=[8, 16, 32],
        store=store,
        checkpoint_every=4,
        diagnostics_every=4,
    )
    return store, result, kept


def test_nested_snapshot_equals_a_standalone_fit_exactly(model, plan, calibrated):
    _, result, kept = calibrated
    standalone = fit(
        model,
        [record.text for record in kept[:8]],
        source_layers=list(plan.layers),
        target_layer=plan.target_layer,
        dim_batch=plan.dim_batch,
        max_seq_len=plan.max_seq_len,
        skip_first=plan.skip_first,
        checkpoint_path=None,
    )
    snapshot = JacobianLens.load(result.snapshots[8].path)
    assert snapshot.n_prompts == 8
    for layer in plan.layers:
        assert torch.equal(
            snapshot.jacobians[layer], standalone.jacobians[layer].half().float()
        )


def test_every_scale_point_is_snapshotted_with_the_right_prompt_count(calibrated):
    _, result, _ = calibrated
    assert sorted(result.snapshots) == [8, 16, 32]
    for scale, snapshot in result.snapshots.items():
        assert snapshot.n_prompts == scale
        assert JacobianLens.load(snapshot.path).n_prompts == scale
        assert snapshot.checksum.startswith("sha256:")


def test_per_layer_diagnostics_are_recorded(calibrated, plan):
    _, result, _ = calibrated
    assert len(result.diagnostics) == 32
    layers = result.diagnostics[-1]["layers"]
    assert sorted(int(key) for key in layers) == sorted(plan.layers)
    for entry in layers.values():
        assert entry["finite"] is True
        assert entry["prompt_jacobian_norm"] >= 0.0
    # The running mean settles: later prompts move it less than early ones.
    early = result.diagnostics[2]["layers"][str(plan.layers[0])]["mean_relative_change"]
    late = result.diagnostics[-1]["layers"][str(plan.layers[0])]["mean_relative_change"]
    assert late < early


def test_scale_larger_than_the_corpus_fails_before_any_fitting(model, plan, tmp_path):
    store = CalibrationStore(tmp_path / "short", _fingerprint(plan))
    store.open()
    records = build_records("c/train", mock_corpus_texts(12), min_chars=100)
    with pytest.raises(ValueError, match="needs 999 fit records"):
        run_calibration(
            model, records, plan=plan, scale_points=[8, 999], store=store
        )
    assert not store.checkpoint_path.exists()


def test_scale_not_reached_is_refused_rather_than_relabelled(model, plan, tmp_path):
    """Skipped prompts must not silently rename a short fit after a scale.

    Four records are long enough in characters to survive ``build_records`` but
    too short in tokens for a single valid source position, so upstream ``fit``
    skips them and ``n_done`` stops at 8 — which is exactly the drift
    :func:`filter_records_by_tokens` exists to prevent.
    """
    store = CalibrationStore(tmp_path / "short", _fingerprint(plan))
    store.open()
    texts = mock_corpus_texts(8) + ["wwwwwwwwwwwwwwwwwwwwwwwwww " * 4] * 4
    records = build_records("c/train", texts, min_chars=100)
    assert len(records) == 12
    with pytest.raises(ScaleNotReachedError, match="never reached"):
        run_calibration(
            model, records, plan=plan, scale_points=[12], store=store
        )


def test_resume_reuses_snapshots_without_refitting(model, plan, calibrated):
    store, result, kept = calibrated
    again = CalibrationStore(store.root, _fingerprint(plan))
    assert again.open() == "resuming"
    resumed = run_calibration(
        model, kept, plan=plan, scale_points=[8, 16, 32], store=again
    )
    assert sorted(resumed.snapshots) == [8, 16, 32]
    assert resumed.diagnostics == []  # nothing was refitted
    for scale in (8, 16, 32):
        assert resumed.snapshots[scale].checksum == result.snapshots[scale].checksum


def test_checkpoint_is_written_atomically(calibrated):
    store, _, _ = calibrated
    assert store.checkpoint_path.is_file()
    leftovers = list(store.checkpoint_path.parent.glob("*.tmp.*"))
    assert leftovers == []


# ------------------------------------------------------------ state refusal


def test_incompatible_fingerprint_is_refused_with_a_diff(plan, tmp_path):
    root = tmp_path / "run"
    CalibrationStore(root, _fingerprint(plan)).open()
    other = CalibrationStore(root, _fingerprint(plan, model_revision="1" * 40))
    with pytest.raises(IncompatibleStateError) as excinfo:
        other.open()
    assert "model_revision" in str(excinfo.value)


def test_a_changed_gate_invalidates_stored_results(plan, tmp_path):
    root = tmp_path / "run"
    CalibrationStore(root, _fingerprint(plan)).open()
    laxer = CalibrationGate(max_median_midrank=999.0)
    assert laxer.digest != CALIBRATION_GATE.digest
    with pytest.raises(IncompatibleStateError, match="gate_digest"):
        CalibrationStore(root, _fingerprint(plan, gate_digest=laxer.digest)).open()


def test_tampered_unit_is_treated_as_missing(plan, tmp_path):
    store = CalibrationStore(tmp_path / "run", _fingerprint(plan))
    store.open()
    path = store.save("validation", "layer8", {"value": 1})
    assert store.has("validation", "layer8")
    record = json.loads(path.read_text(encoding="utf-8"))
    record["payload"]["value"] = 2
    path.write_text(json.dumps(record), encoding="utf-8")
    assert store.load("validation", "layer8") is None
    assert store.invalid_units


def test_unknown_stage_is_rejected(plan, tmp_path):
    store = CalibrationStore(tmp_path / "run", _fingerprint(plan))
    with pytest.raises(ValueError, match="unknown calibration stage"):
        store.stage_dir("intervention")
    assert "intervention" not in CALIBRATION_STAGES


# ---------------------------------------------------------- the gate itself


def test_every_archetype_behaves_as_declared(validation_by_scale):
    small, mid, large = MOCK_SCALE_POINTS
    passing = {
        scale: set(eligible_layers(validation_by_scale[scale]))
        for scale in MOCK_SCALE_POINTS
    }
    # a late layer that passes at every scale
    for scale in MOCK_SCALE_POINTS:
        assert {35, 38, 40} <= passing[scale]
    # an early layer that improves with scale and eventually passes
    assert 14 not in passing[small] and 14 not in passing[mid]
    assert 14 in passing[large]
    # a layer that remains degenerate
    assert all(14 != 8 and 8 not in passing[scale] for scale in MOCK_SCALE_POINTS)


def test_degenerate_layer_fails_on_the_tie_clause(validation_by_scale):
    verdict = validation_by_scale[MOCK_SCALE_POINTS[-1]][8]
    assert not verdict["passed"]
    assert "coverage_and_nondegeneracy" in verdict["failed_checks"]
    assert verdict["metrics"]["j_lens"]["tied_at_max_rate"] == 1.0


def test_optimistically_good_layer_fails_tie_aware_validation(validation_by_scale):
    """The layer-32 shape: rank 1 under the old convention, deep under midrank."""
    verdict = validation_by_scale[MOCK_SCALE_POINTS[-1]][32]
    metrics = verdict["metrics"]["j_lens"]
    assert metrics["median_optimistic_rank"] == 1.0
    assert metrics["median_midrank"] == pytest.approx(12.5)
    assert not verdict["passed"]
    assert "median_rank_and_top_k" in verdict["failed_checks"]


def test_controls_are_scored_and_reported_for_every_layer(validation_by_scale):
    verdict = validation_by_scale[MOCK_SCALE_POINTS[-1]][38]
    for variant in ("permuted", "random", "wrong_layer", "logit_lens"):
        assert verdict["metrics"][variant]["n_prompts"] == 128
    jlens = verdict["metrics"]["j_lens"]["mean_reciprocal_rank"]
    for control in ("permuted", "random", "wrong_layer"):
        assert jlens > verdict["metrics"][control]["mean_reciprocal_rank"]
    assert verdict["diagnostic_variants"] == ["logit_lens"]


def test_gate_reports_all_three_rank_conventions_and_decides_on_midrank(
    validation_by_scale,
):
    verdict = validation_by_scale[MOCK_SCALE_POINTS[-1]][38]
    assert verdict["rank_convention"] == "midrank"
    assert verdict["rank_conventions_reported"] == [
        "optimistic",
        "pessimistic",
        "midrank",
    ]


def test_legacy_gate_is_reported_but_never_binding(validation_by_scale):
    verdict = validation_by_scale[MOCK_SCALE_POINTS[-1]][38]
    assert verdict["legacy_gate"]["is_binding"] is False
    assert "passed" in verdict["legacy_gate"]


def test_target_share_clause_is_the_gates_own_addition():
    assert CALIBRATION_GATE.max_target_token_share == 0.25
    rows = mock_validation_rows(
        layers=[38], scale=1, scale_index=2, n_prompts=128, n_distinct_targets=2
    )
    verdict = evaluate_calibration_layers(rows, layers=[38], scale=1)[38]
    assert not verdict["passed"]
    assert "coverage_and_nondegeneracy" in verdict["failed_checks"]
    assert not verdict["target_diversity"]["share_ok"]


def test_gate_digest_changes_when_any_threshold_changes():
    assert CalibrationGate(min_distinct_target_tokens=1).digest != CALIBRATION_GATE.digest
    assert CALIBRATION_GATE.digest == CalibrationGate().digest


def test_gate_text_states_the_criterion_and_its_digest():
    text = gate_text()
    assert CALIBRATION_GATE.digest in text
    assert "MIDRANK" in text
    assert "PASSING AT A SCALE POINT IS NOT PUBLICATION" in text


def test_confirmation_gate_is_the_same_rule_at_the_confirmation_size():
    confirmation = CALIBRATION_GATE.for_confirmation()
    assert confirmation.n_prompts == 128
    for field in ("max_median_midrank", "min_top_k_inclusion", "max_tied_at_max_rate"):
        assert getattr(confirmation, field) == getattr(CALIBRATION_GATE, field)


# --------------------------------------------------- scale comparison/plateau


def test_comparison_covers_every_layer_at_every_scale(comparison):
    assert len(comparison["rows"]) == len(MOCK_LAYERS) * len(MOCK_SCALE_POINTS)
    assert len(comparison["deltas"]) == len(MOCK_LAYERS) * (len(MOCK_SCALE_POINTS) - 1)
    assert comparison["scales"] == sorted(MOCK_SCALE_POINTS)


def test_comparison_records_margins_over_every_control(comparison):
    row = next(r for r in comparison["rows"] if r["layer"] == 38)
    for key in (
        "margin_over_permuted",
        "margin_over_random",
        "margin_over_wrong_layer",
        "logit_lens_mrr",
    ):
        assert key in row


def test_plateau_rule_fires_when_an_earlier_layer_is_still_climbing(comparison):
    verdict = evaluate_plateau(comparison)
    assert verdict["verdict"] == PLATEAU_STILL_IMPROVING
    assert verdict["extension_justified"] is True
    assert 14 in [row["layer"] for row in verdict["candidates"] if row["qualifies"]]
    assert verdict["runs_automatically"] is False


def test_plateau_rule_stops_when_nothing_earlier_improves(validation_by_scale):
    flat = {
        scale: validation_by_scale[MOCK_SCALE_POINTS[0]] for scale in MOCK_SCALE_POINTS
    }
    verdict = evaluate_plateau(compare_scales(flat, layers=MOCK_LAYERS))
    assert verdict["verdict"] == PLATEAU_REACHED
    assert verdict["extension_justified"] is False
    assert "earlier_layer_materially_improves" in verdict["failed_clauses"]


def test_plateau_rule_refuses_to_extend_when_an_eligible_layer_is_lost(
    validation_by_scale,
):
    small, mid, large = MOCK_SCALE_POINTS
    degraded = dict(validation_by_scale[large])
    degraded[38] = {**degraded[38], "passed": False, "status": "INELIGIBLE"}
    verdict = evaluate_plateau(
        compare_scales(
            {
                small: validation_by_scale[small],
                mid: validation_by_scale[mid],
                large: degraded,
            },
            layers=MOCK_LAYERS,
        )
    )
    assert "no_eligible_layer_lost" in verdict["failed_clauses"]
    assert verdict["extension_justified"] is False


def test_plateau_rule_digest_is_stable_and_declared(comparison):
    verdict = evaluate_plateau(comparison)
    assert verdict["rule_digest"] == PLATEAU_RULE.digest
    assert PLATEAU_RULE.declared_before_results is True
    assert PLATEAU_RULE.digest in PLATEAU_RULE.text()


def test_scale_selection_is_parsimonious(comparison):
    selection = select_scale(comparison)
    assert selection["selected_scale"] == MOCK_SCALE_POINTS[-1]
    assert selection["confirmation_not_consulted"] is True


def test_scale_selection_prefers_the_smallest_agreeing_scale(validation_by_scale):
    small, mid, large = MOCK_SCALE_POINTS
    identical = {scale: validation_by_scale[large] for scale in MOCK_SCALE_POINTS}
    selection = select_scale(compare_scales(identical, layers=MOCK_LAYERS))
    assert selection["selected_scale"] == small


# --------------------------------------------- confirmation and publication


def test_confirmation_set_is_locked_until_a_scale_is_selected(partitions):
    vault = ConfirmationVault(records=partitions.confirmation)
    assert vault.locked
    with pytest.raises(ConfirmationLocked, match="not been unlocked"):
        vault.open()


def test_confirmation_unlock_requires_a_real_selection(partitions, comparison):
    vault = ConfirmationVault(records=partitions.confirmation)
    with pytest.raises(ConfirmationLocked):
        vault.unlock({"nothing": True})
    with pytest.raises(ConfirmationLocked, match="confirmation_not_consulted"):
        vault.unlock({"selected_scale": 32, "confirmation_not_consulted": False})
    vault.unlock(select_scale(comparison))
    assert not vault.locked
    assert len(vault.open()) == len(partitions.confirmation)


def _confirmation_verdict(layer, scale, *, passed=True):
    rows = mock_validation_rows(
        layers=[layer],
        scale=scale,
        scale_index=2 if passed else 0,
        n_prompts=128,
    )
    return evaluate_calibration_layers(
        rows, layers=[layer], scale=scale, stage="confirmation"
    )[layer]


def _publish_kwargs(plan, partitions):
    return {
        "load_info": {
            "model_repo_id": "mock",
            "model_revision": "0" * 40,
            "tokenizer_repo_id": "mock",
            "tokenizer_revision": "0" * 40,
        },
        "corpus_manifest": corpus_manifest(
            partitions, corpus_config={"hf_dataset": "mock"}, scale_points=[32]
        ),
        "capture_plan": plan.to_dict(),
        "fitting_diagnostics": {"n_done": 32},
        "environment": {"torch": torch.__version__},
        "protocol_version": "research-grade-multilayer-text-jlens-calibration-v1",
    }


def test_publication_is_refused_before_confirmation_is_opened(
    plan, partitions, calibrated, tmp_path
):
    _, result, _ = calibrated
    lens = JacobianLens.load(result.snapshots[32].path)
    vault = ConfirmationVault(records=partitions.confirmation)
    with pytest.raises(PublicationRefused, match="never opened"):
        publish_layer(
            layer=plan.layers[0],
            scale=32,
            lens=lens,
            destination=tmp_path / "lens.pt",
            confirmation_verdict=_confirmation_verdict(38, 32),
            validation_verdict=_confirmation_verdict(38, 32),
            vault=vault,
            **_publish_kwargs(plan, partitions),
        )


def test_publication_is_refused_for_a_failed_layer(
    plan, partitions, comparison, calibrated, tmp_path
):
    _, result, _ = calibrated
    lens = JacobianLens.load(result.snapshots[32].path)
    vault = ConfirmationVault(records=partitions.confirmation)
    vault.unlock(select_scale(comparison))
    vault.open()
    failed = _confirmation_verdict(20, 32, passed=False)
    failed = {**failed, "layer": plan.layers[0]}
    with pytest.raises(PublicationRefused, match="failed the confirmation gate"):
        publish_layer(
            layer=plan.layers[0],
            scale=32,
            lens=lens,
            destination=tmp_path / "lens.pt",
            confirmation_verdict=failed,
            validation_verdict=failed,
            vault=vault,
            **_publish_kwargs(plan, partitions),
        )
    assert not (tmp_path / "lens.pt").exists()


def test_publication_is_refused_for_a_development_verdict(
    plan, partitions, comparison, calibrated, tmp_path
):
    _, result, _ = calibrated
    lens = JacobianLens.load(result.snapshots[32].path)
    vault = ConfirmationVault(records=partitions.confirmation)
    vault.unlock(select_scale(comparison))
    vault.open()
    development = evaluate_calibration_layers(
        mock_validation_rows(layers=[38], scale=32, scale_index=2, n_prompts=128),
        layers=[38],
        scale=32,
        stage="validation",
    )[38]
    with pytest.raises(PublicationRefused, match="not a confirmation result"):
        publish_layer(
            layer=plan.layers[0],
            scale=32,
            lens=lens,
            destination=tmp_path / "lens.pt",
            confirmation_verdict={**development, "layer": plan.layers[0]},
            validation_verdict=development,
            vault=vault,
            **_publish_kwargs(plan, partitions),
        )


def test_publication_refuses_to_overwrite_the_frozen_v2_artifact(
    plan, partitions, comparison, calibrated, tmp_path
):
    _, result, _ = calibrated
    lens = JacobianLens.load(result.snapshots[32].path)
    vault = ConfirmationVault(records=partitions.confirmation)
    vault.unlock(select_scale(comparison))
    vault.open()
    with pytest.raises(PublicationRefused, match="frozen completed-run artifact"):
        publish_layer(
            layer=plan.layers[0],
            scale=32,
            lens=lens,
            destination=(
                "runs/text_jlens_early_layer_recalibration_v2/artifacts/"
                "lens.validated.pt"
            ),
            confirmation_verdict=_confirmation_verdict(38, 32),
            validation_verdict=_confirmation_verdict(38, 32),
            vault=vault,
            **_publish_kwargs(plan, partitions),
        )


def test_a_passing_layer_publishes_one_single_layer_artifact(
    plan, partitions, comparison, calibrated, tmp_path
):
    _, result, _ = calibrated
    lens = JacobianLens.load(result.snapshots[32].path)
    layer = plan.layers[0]
    vault = ConfirmationVault(records=partitions.confirmation)
    vault.unlock(select_scale(comparison))
    vault.open()
    verdict = {**_confirmation_verdict(38, 32), "layer": layer}
    destination = tmp_path / "published" / "lens.pt"
    artifact = publish_layer(
        layer=layer,
        scale=32,
        lens=lens,
        destination=destination,
        confirmation_verdict=verdict,
        validation_verdict=verdict,
        vault=vault,
        **_publish_kwargs(plan, partitions),
    )
    published = JacobianLens.load(str(destination))
    assert published.source_layers == [layer]  # one artifact per passing layer
    assert artifact["validated"] is True and artifact["frozen"] is True
    assert artifact["calibration_modality"] == "text-only"
    assert artifact["spokencoco_used"] is False
    assert artifact["objective"] == "not_applicable_estimator_is_a_sample_mean"
    assert artifact["lens_checksum"].startswith("sha256:")
    assert artifact["artifact_checksum"].startswith("sha256:")
    assert destination.with_suffix(".json").is_file()
    assert list(destination.parent.glob("*.tmp.*")) == []


def test_published_artifact_carries_every_required_metadata_field(
    plan, partitions, comparison, calibrated, tmp_path
):
    _, result, _ = calibrated
    lens = JacobianLens.load(result.snapshots[32].path)
    layer = plan.layers[0]
    vault = ConfirmationVault(records=partitions.confirmation)
    vault.unlock(select_scale(comparison))
    vault.open()
    verdict = {**_confirmation_verdict(38, 32), "layer": layer}
    artifact = publish_layer(
        layer=layer,
        scale=32,
        lens=lens,
        destination=tmp_path / "lens.pt",
        confirmation_verdict=verdict,
        validation_verdict=verdict,
        vault=vault,
        **_publish_kwargs(plan, partitions),
    )
    required = json.loads(
        (REPO_ROOT / "configs" / "research_grade_jlens_calibration_v1.json").read_text(
            encoding="utf-8"
        )
    )["publication"]["required_metadata"]
    aliases = {
        "python_version": "environment",
        "torch_version": "environment",
        "transformers_version": "environment",
        "upstream_commit": "environment",
        "local_commit": "environment",
    }
    for field in required:
        key = aliases.get(field, field)
        assert key in artifact, f"published artifact is missing {field!r}"


def test_failed_layers_keep_diagnostics_and_are_never_marked_validated():
    verdict = _confirmation_verdict(20, 32, passed=False)
    record = record_failed_layer(
        layer=20, scale=32, confirmation_verdict=verdict, validation_verdict=verdict
    )
    assert record["validated"] is False and record["published"] is False
    assert record["confirmation_metrics"] is not None
    summary = publication_summary([], [record])
    assert summary["failed_layers"] == [20]
    assert summary["failed_layers_marked_validated"] is False


# ------------------------------------------------------------------- budget


def test_two_week_scale_schedule_stops_at_the_paper_endpoint():
    assert SCALE_POINTS == (100, 250, 1_000)
    assert OPTIONAL_SCALE_POINTS == ()


def test_budget_rows_are_cumulative_and_carry_an_uncertainty_range():
    plan = build_capture_plan()
    budget = estimate_budget(plan, scale_points=SCALE_POINTS)
    hours = [row["fitting_hours_central"] for row in budget.per_scale]
    assert hours == sorted(hours)
    # nested, so 250 is 2.5x 100 rather than 100 + 250
    assert budget.row(250)["fitting_hours_central"] == pytest.approx(
        2.5 * budget.row(100)["fitting_hours_central"], rel=1e-3
    )
    for row in budget.per_scale:
        assert row["fitting_hours_low"] < row["fitting_hours_central"]
        assert row["fitting_hours_high"] > row["fitting_hours_central"]
        assert row["cumulative"] is True
    assert budget.to_dict()["cumulative_not_additive"] is True


def test_budget_counts_the_backward_passes_the_estimator_actually_runs():
    plan = build_capture_plan()
    assert plan.backward_passes_per_prompt == 320  # ceil(2560 / 8)
    assert plan.forwards_per_prompt == 1
    assert plan.backward_span == 41 - 8
    budget = estimate_budget(plan, scale_points=[1_000])
    assert budget.row(1_000)["backward_passes"] == 320_000
    assert budget.row(1_000)["forward_passes"] == 1_000


def test_budget_storage_matches_the_matrices_actually_written():
    plan = build_capture_plan()
    budget = estimate_budget(plan, scale_points=SCALE_POINTS)
    assert budget.checkpoint_bytes == 8 * 2560 * 2560 * 4
    assert budget.snapshot_bytes_total == 3 * 8 * 2560 * 2560 * 2
    assert budget.drive_bytes_total < 2**30  # under a gibibyte


def test_budget_text_names_its_single_measurement():
    plan = build_capture_plan()
    text = format_budget(estimate_budget(plan), plan)
    assert "docs/pilot_report.md" in text
    assert "NESTED" in text
    assert "cumulative" in text


def test_layer_grid_and_depths_are_the_frozen_ones():
    assert CALIBRATION_LAYERS == (8, 14, 20, 26, 32, 35, 38, 40)
    assert [normalized_depth(l) for l in CALIBRATION_LAYERS] == [
        19, 33, 48, 62, 76, 83, 90, 95
    ]


def test_capture_plan_refuses_an_impossible_grid():
    with pytest.raises(ValueError, match="shallower than target_layer"):
        build_capture_plan(layers=[8, 41], target_layer=41)
    with pytest.raises(ValueError, match="out of range"):
        build_capture_plan(layers=[8, 99], target_layer=41)
    with pytest.raises(ValueError, match="no valid positions"):
        build_capture_plan(max_seq_len=16, skip_first=16)


# ---------------------------------------------------------------- baselines


def test_baseline_manifest_documents_without_granting_access():
    manifest = baseline_manifest()
    assert manifest["calibration_may_read_multimodal_results"] is False
    assert manifest["readable_by_calibration"] == ["text_jlens_v2_lens"]
    for artifact in manifest["artifacts"]:
        assert artifact["immutable"] is True
        assert artifact["may_be_used_to_fit"] is False
        assert artifact["may_be_used_to_validate"] is False


def test_unknown_baseline_checksums_are_flagged_not_invented():
    for artifact in BASELINE_ARTIFACTS:
        checksum = artifact.checksum
        assert checksum.startswith("sha256:") or checksum == (
            "REQUIRES_VERIFICATION_IN_COLAB"
        )
        if "..." in checksum:
            assert "REQUIRES_VERIFICATION_IN_COLAB" in checksum


def test_baseline_paths_are_never_opened(tmp_path, monkeypatch):
    opened: list[str] = []
    real_open = Path.open

    def tracking_open(self, *args, **kwargs):
        opened.append(str(self))
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", tracking_open)
    baseline_manifest()
    assert not any("MyDrive" in path for path in opened)


# ------------------------------------------------------------------- report


def test_report_states_what_it_does_not_establish(comparison, partitions, plan):
    plan_dict = plan.to_dict()
    payload = calibration_report_payload(
        fingerprint_digest="sha256:abc",
        protocol_version="v1",
        capture_plan=plan_dict,
        budget=estimate_budget(build_capture_plan()).to_dict(),
        corpus_manifest=corpus_manifest(
            partitions, corpus_config={"hf_dataset": "mock"}, scale_points=[32]
        ),
        leakage_audit=audit_leakage(partitions, scale_points=[8]),
        nesting_audit=scale_nesting_audit(partitions.fit, [8, 16, 32]),
        diversity=audit_target_diversity(list(range(40)) * 3),
        comparison=comparison,
        plateau=evaluate_plateau(comparison),
        selection=select_scale(comparison),
        confirmation=None,
        publication=None,
        vault_status=ConfirmationVault(records=partitions.confirmation).status(),
        resume_status={"status": "starting", "completed_units": {}},
        baseline=baseline_manifest(),
        mode="mock",
    )
    markdown = calibration_report_markdown(
        payload, gate_text=gate_text(), plateau_text=PLATEAU_RULE.text()
    )
    assert "MOCK RUN" in markdown
    assert "proves pipeline behaviour only" in markdown
    assert "not** claimed here" in markdown
    assert "no fitting objective and no optimizer" in markdown
    # not-run confirmation must not read as "nothing passed"
    assert "not** the same as 'no layer" in markdown
    assert payload["report_checksum"].startswith("sha256:")


# ------------------------------------------------------ MOCK isolation


def test_mock_mode_touches_no_model_hub_or_dataset():
    """Import and run the mock pipeline with the network libraries poisoned."""
    script = textwrap.dedent(
        """
        import sys, types

        class Blocked(ImportError):
            pass

        class Blocker:
            FORBIDDEN = {
                "transformers", "datasets", "huggingface_hub",
                "google.colab", "safetensors",
            }
            def find_module(self, name, path=None):
                return self if name.split(".")[0] in self.FORBIDDEN else None
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in self.FORBIDDEN:
                    raise Blocked(f"MOCK mode imported {name}")
                return None
            def load_module(self, name):
                raise Blocked(f"MOCK mode imported {name}")

        sys.meta_path.insert(0, Blocker())

        from jlens.calibration.mock import (
            MockCalibrationModel, mock_corpus_texts, mock_validation_rows,
            MOCK_LAYERS, MOCK_SCALE_POINTS,
        )
        from jlens.calibration.corpus import build_records, build_partitions, audit_leakage
        from jlens.calibration.gate import evaluate_calibration_layers, eligible_layers
        from jlens.calibration.scale import compare_scales, evaluate_plateau, select_scale

        model = MockCalibrationModel()
        records = build_records("mock/train", mock_corpus_texts(500), min_chars=100)
        parts = build_partitions(records, corpus_id="mock/train",
                                 n_validation=32, n_confirmation=32)
        assert audit_leakage(parts, scale_points=[8])["ok"]
        by_scale = {
            s: evaluate_calibration_layers(
                mock_validation_rows(scale=s, scale_index=i, n_prompts=128),
                layers=MOCK_LAYERS, scale=s)
            for i, s in enumerate(MOCK_SCALE_POINTS)
        }
        cmp = compare_scales(by_scale, layers=MOCK_LAYERS)
        evaluate_plateau(cmp)
        select_scale(cmp)

        # Nothing forbidden may even have been imported transitively.
        leaked = sorted(m for m in sys.modules if m.split(".")[0] in Blocker.FORBIDDEN)
        assert not leaked, f"MOCK mode pulled in {leaked}"
        print("OK", sorted(eligible_layers(by_scale[MOCK_SCALE_POINTS[-1]])))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    assert "OK [14, 35, 38, 40]" in result.stdout


def test_calibration_package_never_imports_multimodal_data_modules():
    package = REPO_ROOT / "jlens" / "calibration"
    forbidden = (
        "spokencoco",
        "mmpilot.audio",
        "mmpilot.concepts",
        "mmpilot.causal",
        "mmpilot.selection",
        "autoencoder",
        "explorer_export",
        "reconstruction",
    )
    for path in package.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        imports = "\n".join(
            line for line in source.splitlines() if line.strip().startswith(("import ", "from "))
        )
        for name in forbidden:
            assert name not in imports, f"{path.name} imports {name}"
