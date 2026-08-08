# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""The layer-32 reporting repair, against a real-shaped completed run.

The fixture is not a stub: it writes a run directory in the shape the first real
L32 run produced — 48 checksum-validated readout rows per modality, no
undersized cells, an AMBIGUOUS classification, a WEAK causal verdict whose
passing cells are ``cat text->spoken_audio`` and ``cat text->image``, and an
INCONCLUSIVE verdict E. Every assertion below is about how that run is
*displayed*; none of them may move a number.
"""

import json
from pathlib import Path

import pytest

from jlens.mmpilot.convergence import (
    CONTROL_VARIANTS,
    CONVERGENCE_CRITERION,
    CONVERGENCE_PROTOCOL,
    CRITERION_TEXT,
    ConvergenceFingerprint,
    ConvergenceStore,
    audit_native_head,
    resolve_candidate_tokens,
)
from jlens.mmpilot.convergence_mock import (
    MockWorldSpec,
    build_mock_head,
    mock_activation,
    mock_candidate_token_ids,
)
from jlens.mmpilot.jspace import tensor_checksum
from jlens.mmpilot.l32_followup import (
    CONVERGENCE_PHRASE,
    L32_FOLLOWUP_PROTOCOL,
    L32FollowupRefused,
    adjacent_layer_recommendation,
    assert_report_phrasing,
    run_single_layer_convergence,
)
from jlens.mmpilot.l32_reporting import (
    AMENDED_MARKDOWN_NAME,
    AMENDED_REPORT_NAME,
    CELL_FIELDS,
    L32_REPORTING_VERSION,
    ORIGINAL_REPORT_NAME,
    REPORTING_SCHEMA,
    RETIRED_CELL_FIELDS,
    ControlRecordsIncomplete,
    ConvergenceViewMismatch,
    ReportingAmendmentRefused,
    assert_controls_complete,
    build_reporting_amendment,
    causal_cell_breakdown,
    classification_detail,
    control_rows,
    convergence_cell_rows,
    format_causal_breakdown,
    format_classification,
    format_controls,
    format_convergence_cells,
    format_l32_criterion,
    read_readout_rows,
    recompute_convergence_view,
    render_amendment_markdown,
    source_unit_digest_from_disk,
    write_reporting_amendment,
)
from jlens.mmpilot.store import payload_checksum

LAYER = 32
CONCEPTS = ("bird", "cat", "toilet")
CANDIDATES = ("bird", "cat", "giraffe", "microwave", "toilet", "zebra")
MODALITIES = ("text", "image", "spoken_audio")
IMAGES_PER_CONCEPT = 16

#: The strength that lands the synthetic layer between the two frozen bars, the
#: way the real layer 32 landed. Not tuned per assertion — one value, fixed here.
AMBIGUOUS_STRENGTH = 1.5


# --------------------------------------------------------------- the fixture


def _passing_cell(concept, pair, **overrides):
    cell = {
        "layer": LAYER,
        "concept": concept,
        "pair": pair,
        "audio_related": "spoken_audio" in pair,
        "evaluated": True,
        "execution_status": "measured",
        "passes": True,
        "counted_toward_verdict": True,
        "capability_admissible": True,
        "capability_rejection_reason": None,
        "alpha": 1.0,
        "mean_signed_target_effect": 0.31,
        "fraction_expected_sign": 0.875,
        "mean_activation_norm_ratio": 1.04,
        "mean_abs_unrelated_change": 0.06,
        "n_distinct_images": 16,
        "n_positive_images": 8,
        "n_negative_images": 8,
        "meets_claim_image_floor": True,
        "random_control": 0.02,
        "unrelated_control": 0.03,
        "raw_residual_control": 0.11,
        "jspace_beats_raw_direction": True,
        "reasons": [],
    }
    cell.update(overrides)
    return cell


def _failing_cell(concept, pair):
    return _passing_cell(
        concept,
        pair,
        passes=False,
        mean_signed_target_effect=0.01,
        reasons=["effect +0.0100 does not clear 1.5x the strongest control"],
    )


def build_real_shaped_run(root: Path) -> dict:
    """A completed L32 follow-up run, in the shape the real one produced."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    spec = MockWorldSpec(
        mode="ambiguous",
        layers=(LAYER,),
        concepts=CONCEPTS,
        candidates=CANDIDATES,
        images_per_concept=IMAGES_PER_CONCEPT,
        layer_strength={LAYER: AMBIGUOUS_STRENGTH},
    )
    head, directions = build_mock_head(spec)
    tokenization = resolve_candidate_tokens(mock_candidate_token_ids(spec))

    units = []
    for concept in CONCEPTS:
        for index in range(IMAGES_PER_CONCEPT):
            for modality in MODALITIES:
                activation = mock_activation(
                    spec=spec,
                    concept=concept,
                    directions=directions,
                    modality=modality,
                    layer=LAYER,
                    image_index=index,
                )
                units.append(
                    {
                        "sample_id": f"grp-{concept}-{index:02d}::{modality}",
                        "group_id": f"grp-{concept}-{index:02d}",
                        "image_id": f"img-{concept}-{index:02d}",
                        "recording_id": f"grp-{concept}-{index:02d}",
                        "concept": concept,
                        "modality": modality,
                        "layer": LAYER,
                        "split": "test" if index % 2 == 0 else "train",
                        "capability_admissible": True,
                        "activation": [float(x) for x in activation.tolist()],
                        "activation_checksum": tensor_checksum(activation),
                        "clean_final_prediction": concept,
                    }
                )

    head_audit = audit_native_head(head, model=None, probes=2)
    fingerprint = ConvergenceFingerprint(
        protocol=CONVERGENCE_PROTOCOL,
        completed_run_fingerprint_digest="sha256:fixture-run",
        completed_run_dir=str(root),
        model_repo_id="mock/gemma-like",
        model_revision="mock",
        processor_revision="mock",
        layers=(LAYER,),
        candidate_digest=tokenization["digest"],
        readout_mode=tokenization["readout_mode"],
        head_checksum=str(head_audit.get("head_checksum", "")),
        criterion_digest=CONVERGENCE_CRITERION.digest,
        code_version=L32_FOLLOWUP_PROTOCOL,
    )
    store = ConvergenceStore(root / "convergence", fingerprint)
    store.open()
    convergence = run_single_layer_convergence(
        population={"units": units},
        head=head,
        tokenization=tokenization,
        head_audit=head_audit,
        store=store,
        layer=LAYER,
        confirmation_record={"layer": LAYER, "passed": True, "failed_checks": []},
    )
    classification = convergence["classification"]
    assert classification["classification"] == "AMBIGUOUS", (
        "the fixture must reproduce the real run's AMBIGUOUS layer; got "
        f"{classification['classification']}"
    )
    assert classification["undersized_cells"] == []

    # A few scientific units, so the source-unit digest has something to bind.
    for stage, keys in (
        ("capability", ("cap-a", "cap-b")),
        ("intervention", ("int-a", "int-b", "int-c")),
        ("metric", ("l32_causal_verdict",)),
    ):
        directory = root / "units" / stage
        directory.mkdir(parents=True, exist_ok=True)
        for key in keys:
            payload = {"stage": stage, "key": key, "value": 1}
            (directory / f"{key}.json").write_text(
                json.dumps(
                    {
                        "schema": "jlens.mmpilot.unit.v1",
                        "stage": stage,
                        "key": key,
                        "fingerprint_digest": "sha256:fixture-run",
                        "unit_checksum": payload_checksum(payload),
                        "payload": payload,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

    run_fingerprint = {
        "mode": "l32_followup",
        "model_repo_id": "google/gemma-4-E4B-it",
        "model_revision": "f" * 40,
        "processor_revision": "f" * 40,
        "layers": [LAYER],
        "lens_checksum": "sha256:fixture-lens",
        "manifest_checksum": "sha256:fixture-manifest",
        "split_id": "spokencoco-l32-followup-v1",
        "intervention_config": {
            "intervention_family": "source_derived_jspace_steering"
        },
        "extra": {},
    }
    (root / "fingerprint.json").write_text(
        json.dumps(run_fingerprint, indent=2), encoding="utf-8"
    )

    causal_cells = [
        _passing_cell("cat", "text->spoken_audio"),
        _passing_cell("cat", "text->image"),
        _failing_cell("cat", "spoken_audio->text"),
        _failing_cell("bird", "image->spoken_audio"),
        _failing_cell("toilet", "spoken_audio->image"),
    ]
    report = {
        "schema": "jlens.mmpilot.l32_followup_report.v1",
        "protocol": L32_FOLLOWUP_PROTOCOL,
        "mode": "l32_followup",
        "run_dir": str(root),
        "intervention_family": "source_derived_jspace_steering",
        "run_fingerprint_digest": payload_checksum(run_fingerprint),
        "followup_fingerprint": {"fingerprint_digest": "sha256:fixture-followup"},
        "verdicts": {
            "A_lens_integrity": {"verdict": "PASSED"},
            "B_representational_transfer": {"verdict": "SUPPORTED"},
            "C_causal_transfer": {"verdict": "WEAK", "cells": causal_cells},
            "D_native_output_convergence": classification,
            "E_pre_convergence_causal_transfer": {"verdict": "INCONCLUSIVE"},
        },
        "convergence": {
            key: value for key, value in convergence.items() if key != "summary"
        },
    }
    (root / ORIGINAL_REPORT_NAME).write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    return {
        "root": root,
        "report": report,
        "convergence": convergence,
        "classification": classification,
    }


@pytest.fixture(scope="module")
def real_shaped(tmp_path_factory):
    return build_real_shaped_run(tmp_path_factory.mktemp("l32real") / "run")


@pytest.fixture(scope="module")
def amendment(real_shaped):
    return build_reporting_amendment(real_shaped["root"], layer=LAYER)


# ------------------------------------------------------ the L32 criterion prose


def test_the_l32_criterion_never_claims_l35_is_the_audited_layer():
    text = format_l32_criterion(layer=LAYER)
    assert "L35 / L38 / L40" not in text
    assert "physical layer 32" in text
    for stale in ("layer 35", "L35", "layer 38", "L38", "layer 40", "L40"):
        assert stale not in text, f"{stale!r} appears in the L32 criterion"


def test_the_l32_criterion_states_the_single_layer_scope():
    text = format_l32_criterion(layer=LAYER)
    assert "SINGLE-LAYER classification" in text
    assert "no later-layer trajectory clause is applied" in text
    assert "cannot be evaluated, or claimed, from one point" in text.replace(
        "\n", " "
    ).replace("  ", " ")


def test_the_l32_criterion_carries_the_unchanged_frozen_thresholds():
    text = format_l32_criterion(layer=LAYER)
    criterion = CONVERGENCE_CRITERION
    assert criterion.digest in text
    assert f"{criterion.converged_min_clean_agreement:.0%}" in text
    assert f"{criterion.not_converged_max_clean_agreement:.0%}" in text
    assert f"{criterion.converged_max_median_rank:.1f}" in text
    assert f"{criterion.not_converged_min_median_rank:.1f}" in text


def test_the_l32_criterion_defines_all_three_classes():
    text = format_l32_criterion(layer=LAYER)
    assert "CONVERGED, only if" in text
    assert "NOT_CONVERGED, only if" in text
    assert "AMBIGUOUS is everything between the two bars" in text
    assert "AMBIGUOUS is a real outcome" not in text  # phrased as the criterion does
    assert "supports no" in text


def test_the_l32_criterion_states_the_interpretation_boundary_correctly():
    text = format_l32_criterion(layer=LAYER)
    assert CONVERGENCE_PHRASE in text
    for forbidden in ("pre-linguistic", "language-free", "before language exists"):
        index = text.lower().find(forbidden)
        assert index != -1, f"{forbidden!r} should be named as forbidden"
        assert "never" in text.lower()[max(0, index - 200) : index + 50]


def test_the_historical_criterion_constant_is_not_mutated():
    """The earlier audit's protocol is a record, not a template."""
    assert "L35 / L38 / L40" in CRITERION_TEXT
    assert format_l32_criterion(layer=LAYER) != CRITERION_TEXT


# ------------------------------------------------------------- the cell table


def test_the_cell_table_reads_keys_that_exist(amendment):
    rows = amendment["convergence_cells"]
    assert [row["modality"] for row in rows] == list(MODALITIES)
    for row in rows:
        assert row["n"] == 48, row
        for field in CELL_FIELDS:
            assert row[field] is not None, f"{row['modality']}.{field} is None"


def test_the_retired_keys_are_gone_and_named(amendment):
    rows = amendment["convergence_cells"]
    for retired in RETIRED_CELL_FIELDS:
        assert all(retired not in row for row in rows)
    assert RETIRED_CELL_FIELDS["unique_top1_rate"] == "unique_top1_target_rate"
    assert (
        RETIRED_CELL_FIELDS["median_entropy"] == "median_candidate_entropy_nats"
    )


def test_the_specific_fields_the_run_printed_as_none_now_have_values(amendment):
    for row in amendment["convergence_cells"]:
        assert isinstance(row["unique_top1_target_rate"], float)
        assert isinstance(row["median_candidate_entropy_nats"], float)
        assert isinstance(row["n_with_clean_reference"], int)
        assert isinstance(row["target_accuracy_argmax"], float)


def test_a_cell_missing_a_printed_field_refuses(real_shaped):
    summary = real_shaped["convergence"]["summary"]
    broken = json.loads(json.dumps(summary))
    del broken["per_layer"][str(LAYER)]["per_modality"]["text"][
        "median_candidate_entropy_nats"
    ]
    with pytest.raises(ReportingAmendmentRefused, match="missing"):
        convergence_cell_rows(broken, layer=LAYER)


def test_a_summary_for_another_layer_is_refused(real_shaped):
    with pytest.raises(ReportingAmendmentRefused, match="carries no layer 35"):
        convergence_cell_rows(real_shaped["convergence"]["summary"], layer=35)


def test_the_formatted_table_prints_no_none(amendment):
    text = format_convergence_cells(amendment["convergence_cells"], layer=LAYER)
    assert "None" not in text
    assert "unique top-1 (target)" in text
    assert "median entropy (nats)" in text


# ------------------------------------------------------- classification detail


def test_the_classification_names_the_clauses_instead_of_a_missing_field(amendment):
    detail = amendment["classification_detail"]
    assert detail["classification"] == "AMBIGUOUS"
    assert "neither bar was cleared" in detail["decided_by"]
    assert detail["failed_converged_clauses"]
    assert detail["failed_not_converged_clauses"]
    assert detail["undersized_cells"] == []


def test_the_bootstrap_interval_and_unit_count_are_reported(amendment):
    detail = amendment["classification_detail"]
    assert detail["bootstrap_point"] is not None
    assert detail["bootstrap_low"] is not None
    assert detail["bootstrap_high"] is not None
    assert detail["bootstrap_independent_units"] >= 2
    text = format_classification(detail)
    assert "independent unit(s)" in text
    assert "None" not in text


def test_a_converged_or_not_converged_layer_says_which_side_decided_it():
    assert "every CONVERGED clause held" in classification_detail(
        {"classification": "CONVERGED"}
    )["decided_by"]
    assert "every NOT_CONVERGED clause held" in classification_detail(
        {"classification": "NOT_CONVERGED"}
    )["decided_by"]


# ------------------------------------------------------------------ controls


def test_all_three_control_variants_are_read_from_the_nested_structure(amendment):
    rows = amendment["controls"]
    assert [row["variant"] for row in rows] == list(CONTROL_VARIANTS)
    for row in rows:
        assert row["compared_field"] in (
            "target_accuracy_argmax",
            "clean_agreement_argmax",
        )
        assert row["primary_value"] is not None
        assert row["control_value"] is not None
        assert row["chance_rate"] is not None
        assert isinstance(row["primary_is_informative"], bool)
        assert isinstance(row["passed"], bool)
        assert row["reason"]


def test_the_formatted_controls_print_every_required_field(amendment):
    text = format_controls(
        amendment["controls"],
        controls={"all_controls_passed": True, "failed_controls": []},
        layer=LAYER,
    )
    for variant in CONTROL_VARIANTS:
        assert variant in text
    for label in (
        "compared on",
        "primary  ",
        "control  ",
        "chance rate",
        "primary_is_informative",
        "passed",
        "reason",
    ):
        assert label in text
    assert "None" not in text


def test_a_missing_control_refuses_rather_than_passing(real_shaped):
    """summarize_controls skips a variant with no rows; that is not a pass."""
    controls = json.loads(json.dumps(real_shaped["convergence"]["controls"]))
    assert controls["all_controls_passed"] is True
    del controls["per_layer"][str(LAYER)]["controls"]["permuted_activations"]
    # The stale flag still says everything passed. The check must not believe it.
    assert controls["all_controls_passed"] is True
    with pytest.raises(ControlRecordsIncomplete, match="Missing is not passing"):
        assert_controls_complete(controls, layer=LAYER)
    with pytest.raises(ControlRecordsIncomplete):
        control_rows(controls, layer=LAYER)


def test_a_layer_with_no_control_record_at_all_refuses(real_shaped):
    controls = json.loads(json.dumps(real_shaped["convergence"]["controls"]))
    controls["per_layer"] = {}
    with pytest.raises(ControlRecordsIncomplete, match="has not been controlled"):
        assert_controls_complete(controls, layer=LAYER)


def test_an_amendment_over_incomplete_controls_refuses(tmp_path, real_shaped):
    """The refusal reaches the amendment, not just the helper."""
    root = tmp_path / "torn"
    root.mkdir()
    for item in Path(real_shaped["root"]).rglob("*"):
        target = root / item.relative_to(real_shaped["root"])
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(item.read_bytes())
    # Remove every stored row of one control variant.
    removed = 0
    for path in (root / "convergence" / "readout_units").glob(
        "permuted_activations*"
    ):
        path.unlink()
        removed += 1
    assert removed > 0
    with pytest.raises(ControlRecordsIncomplete):
        build_reporting_amendment(root, layer=LAYER)


# ------------------------------------------------------------ causal buckets


def test_the_one_audio_cell_versus_two_off_diagonal_cells_is_explained(amendment):
    breakdown = amendment["causal_breakdown"]
    assert breakdown["n_off_diagonal_passing"] == 2
    assert breakdown["n_audio_related_passing"] == 1
    assert breakdown["n_text_image_passing"] == 1
    assert breakdown["audio_related_passing"][0]["pair"] == "text->spoken_audio"
    assert breakdown["text_image_passing"][0]["pair"] == "text->image"
    reconciliation = breakdown["reconciliation"]
    assert "2 admissible off-diagonal cell(s) passed" in reconciliation
    assert "1 of them are audio-related" in reconciliation
    assert "neither number is wrong" in reconciliation


def test_the_causal_breakdown_reports_distinct_media_counts(amendment):
    breakdown = amendment["causal_breakdown"]
    assert breakdown["n_distinct_images_max_over_cells"] == 8
    for entry in breakdown["off_diagonal_passing"]:
        assert entry["n_positive_images"] == 8
        assert entry["n_negative_images"] == 8
        assert entry["n_distinct_images"] == 16


def test_bidirectional_concepts_are_reported():
    breakdown = causal_cell_breakdown(
        {
            "cells": [
                _passing_cell("cat", "text->spoken_audio"),
                _passing_cell("cat", "spoken_audio->text"),
                _passing_cell("bird", "text->image"),
            ]
        }
    )
    assert breakdown["concepts_transferring_both_directions"] == ["cat"]


def test_inadmissible_and_failing_cells_support_nothing():
    breakdown = causal_cell_breakdown(
        {
            "cells": [
                _passing_cell("cat", "text->image", counted_toward_verdict=False),
                _failing_cell("bird", "text->spoken_audio"),
                _passing_cell("cat", "text->text"),
            ]
        }
    )
    assert breakdown["n_off_diagonal_passing"] == 0


def test_the_formatted_breakdown_separates_the_buckets(amendment):
    text = format_causal_breakdown(amendment["causal_breakdown"], layer=LAYER)
    assert "all admissible off-diagonal (2)" in text
    assert "audio-related (verdict C's arm) (1)" in text
    assert "text<->image (internal replication) (1)" in text
    assert "None" not in text


# ------------------------------------------------ the undersized-cells claim


def test_no_undersized_cells_claim_when_none_exist():
    record = adjacent_layer_recommendation(
        causal_verdict="WEAK", classification="AMBIGUOUS"
    )
    assert record["recommendation"] == "smallest_convergence_resolution_study"
    assert "undersized" not in record["rationale"]
    assert "independent" in record["rationale"]
    assert "separately fingerprinted" in record["rationale"]
    assert "predeclared" in record["rationale"]


def test_the_recommendation_does_not_promise_not_converged():
    rationale = adjacent_layer_recommendation(
        causal_verdict="WEAK", classification="AMBIGUOUS"
    )["rationale"]
    assert "can stay ambiguous at any n" in rationale
    assert "nothing about a larger population makes NOT_CONVERGED more likely" in (
        rationale
    )


# ------------------------------------------------------------- the amendment


def test_the_amendment_binds_source_checksum_fingerprint_and_version(
    amendment, real_shaped
):
    amends = amendment["amends"]
    original = Path(real_shaped["root"]) / ORIGINAL_REPORT_NAME
    from jlens.metadata import file_sha256

    assert amends["original_report_checksum"] == file_sha256(str(original))
    assert amends["original_report_immutable"] is True
    assert amends["run_fingerprint_digest"] == real_shaped["report"][
        "run_fingerprint_digest"
    ]
    assert amends["followup_fingerprint_digest"] == "sha256:fixture-followup"
    assert amends["source_unit_digest"]["combined_digest"].startswith("sha256:")
    assert amends["readout_rows_digest"].startswith("sha256:")
    assert amendment["reporting_version"] == L32_REPORTING_VERSION
    assert amendment["schema"] == REPORTING_SCHEMA
    assert amendment["amendment_checksum"].startswith("sha256:")


def test_a_pinned_checksum_that_does_not_match_refuses(real_shaped):
    with pytest.raises(ReportingAmendmentRefused, match="not the pinned"):
        build_reporting_amendment(
            real_shaped["root"], layer=LAYER, expected_report_checksum="sha256:nope"
        )


def test_a_run_directory_without_the_original_report_refuses(tmp_path):
    with pytest.raises(ReportingAmendmentRefused, match="never discovers one"):
        build_reporting_amendment(tmp_path / "nothing-here", layer=LAYER)


def test_an_unknown_report_schema_is_refused(tmp_path, real_shaped):
    root = tmp_path / "schema"
    root.mkdir()
    report = dict(real_shaped["report"])
    report["schema"] = "some.other.format.v9"
    (root / ORIGINAL_REPORT_NAME).write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ReportingAmendmentRefused, match="will not guess"):
        build_reporting_amendment(root, layer=LAYER)


def test_a_fingerprint_that_disagrees_with_the_report_refuses(tmp_path, real_shaped):
    root = tmp_path / "fpmismatch"
    root.mkdir()
    (root / ORIGINAL_REPORT_NAME).write_text(
        json.dumps(real_shaped["report"], default=str), encoding="utf-8"
    )
    (root / "fingerprint.json").write_text(
        json.dumps({"mode": "something-else"}), encoding="utf-8"
    )
    with pytest.raises(ReportingAmendmentRefused, match="disagree about what was run"):
        build_reporting_amendment(root, layer=LAYER)


def test_the_verdicts_are_carried_through_unchanged(amendment, real_shaped):
    unchanged = amendment["verdicts_unchanged"]
    assert unchanged == {
        "A_lens_integrity": "PASSED",
        "B_representational_transfer": "SUPPORTED",
        "C_causal_transfer": "WEAK",
        "D_native_output_convergence": "AMBIGUOUS",
        "E_pre_convergence_causal_transfer": "INCONCLUSIVE",
    }
    assert amendment["changes_no_scientific_verdict"] is True
    statement = amendment["statement"]
    assert "remains AMBIGUOUS" in statement
    assert "remains WEAK" in statement
    assert "remains INCONCLUSIVE" in statement
    assert "reporting repair" in statement


def test_the_criterion_digest_is_unchanged(amendment):
    assert amendment["criterion_digest"] == CONVERGENCE_CRITERION.digest


def test_a_non_frozen_criterion_is_refused(real_shaped):
    from dataclasses import replace

    loosened = replace(CONVERGENCE_CRITERION, converged_min_clean_agreement=0.5)
    with pytest.raises(ReportingAmendmentRefused, match="never moves a threshold"):
        recompute_convergence_view(
            real_shaped["root"],
            layer=LAYER,
            recorded_classification={},
            criterion=loosened,
        )


def test_a_recomputation_that_disagrees_with_the_run_refuses(real_shaped):
    with pytest.raises(ConvergenceViewMismatch, match="may not change a classification"):
        recompute_convergence_view(
            real_shaped["root"],
            layer=LAYER,
            recorded_classification={"classification": "CONVERGED"},
        )


def test_the_recomputation_matches_and_says_so(amendment):
    recomputation = amendment["recomputation"]
    assert recomputation["recomputed_matches_recorded"] is True
    assert recomputation["n_rows"] == 48 * 3 * 4  # primary + three controls
    assert "the model is not loaded" in recomputation["recomputation_note"]


# --------------------------------------------------- reading is read-only


def _tree_state(root):
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(Path(root).rglob("*"))
        if path.is_file()
    }


def test_building_the_amendment_writes_nothing(real_shaped):
    before = _tree_state(real_shaped["root"])
    build_reporting_amendment(real_shaped["root"], layer=LAYER)
    assert _tree_state(real_shaped["root"]) == before


def test_writing_the_amendment_leaves_the_original_and_units_byte_identical(
    tmp_path, real_shaped
):
    root = tmp_path / "write"
    root.mkdir()
    for item in Path(real_shaped["root"]).rglob("*"):
        target = root / item.relative_to(real_shaped["root"])
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(item.read_bytes())

    before = _tree_state(root)
    built = build_reporting_amendment(root, layer=LAYER)
    written = write_reporting_amendment(root, built)

    after = _tree_state(root)
    new_files = set(after) - set(before)
    assert new_files == {AMENDED_REPORT_NAME, AMENDED_MARKDOWN_NAME}
    for name, payload in before.items():
        assert after[name] == payload, f"{name} was modified"
    assert Path(written["report"]).is_file()
    assert Path(written["markdown"]).is_file()


def test_the_amendment_refuses_to_write_over_the_original(tmp_path, real_shaped):
    root = tmp_path / "overwrite"
    root.mkdir()
    (root / ORIGINAL_REPORT_NAME).write_text(
        json.dumps(real_shaped["report"], default=str), encoding="utf-8"
    )
    with pytest.raises(ReportingAmendmentRefused, match="never overwritten"):
        write_reporting_amendment(
            root, {"schema": REPORTING_SCHEMA}, report_name=ORIGINAL_REPORT_NAME
        )


def test_a_torn_readout_row_refuses(tmp_path, real_shaped):
    root = tmp_path / "tornrow"
    root.mkdir()
    for item in Path(real_shaped["root"]).rglob("*"):
        target = root / item.relative_to(real_shaped["root"])
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(item.read_bytes())
    victim = next((root / "convergence" / "readout_units").glob("*.json"))
    record = json.loads(victim.read_text(encoding="utf-8"))
    record["payload"]["target_margin"] = 999.0
    victim.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ReportingAmendmentRefused, match="failed their own checksum"):
        read_readout_rows(root)


def test_the_source_unit_digest_moves_when_a_unit_moves(tmp_path, real_shaped):
    root = tmp_path / "digest"
    root.mkdir()
    for item in Path(real_shaped["root"]).rglob("*"):
        target = root / item.relative_to(real_shaped["root"])
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(item.read_bytes())
    before = source_unit_digest_from_disk(root)["combined_digest"]
    (root / "units" / "capability" / "cap-a.json").unlink()
    assert source_unit_digest_from_disk(root)["combined_digest"] != before


# ---------------------------------------------------------------- the markdown


def test_the_markdown_states_every_required_conclusion(amendment):
    text = render_amendment_markdown(amendment)
    assert "changes no scientific result" in text
    assert "| D_native_output_convergence | `AMBIGUOUS` |" in text
    assert "| C_causal_transfer | `WEAK` |" in text
    assert "| E_pre_convergence_causal_transfer | `INCONCLUSIVE` |" in text
    assert "**unchanged**" in text
    assert "physical layer 32" in text
    assert "L35 / L38 / L40" not in text


def test_the_markdown_carries_the_binding(amendment):
    text = render_amendment_markdown(amendment)
    assert amendment["amends"]["original_report_checksum"] in text
    assert amendment["reporting_version"] in text
    assert (
        amendment["amends"]["source_unit_digest"]["combined_digest"] in text
    )


# ---------------------------------------------------------------- phrasing


def test_the_amendment_passes_the_phrasing_rule(amendment):
    record = assert_report_phrasing(json.dumps(amendment, default=str))
    assert record["passed"] is True
    assert record["required_phrase_present"] is True
    assert "pre-linguistic" in record["phrases_named_as_forbidden"]


def test_a_bare_affirmative_use_is_still_refused():
    with pytest.raises(L32FollowupRefused, match="cannot support"):
        assert_report_phrasing(
            "Layer 32 shows that the representation is pre-linguistic and the "
            "evidence is overwhelming across every modality we examined here."
        )


def test_naming_a_phrase_as_forbidden_is_allowed():
    record = assert_report_phrasing(
        f'Say "{CONVERGENCE_PHRASE}" — never "pre-linguistic".'
    )
    assert record["passed"] is True
    assert record["phrases_asserted"] == []
