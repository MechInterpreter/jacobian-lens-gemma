# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Complete MOCK execution: every stage, resume, and the GO/NO-GO report.

These assertions are about the *pipeline*. The mock's GO says nothing about
Gemma, and the report itself says so — which is one of the things checked here.
"""

import json

import pytest

from jlens.mmpilot import mock as K
from jlens.mmpilot.store import IncompatibleStateError, RunFingerprint, UnitStore


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    root = tmp_path_factory.mktemp("mockpilot")
    return K.run_mock_pilot(root / "data", root / "run"), root


def test_every_stage_produced_units(run):
    result, _ = run
    assert result["status"] == "starting"
    counts = result["store"].status_report()["completed_units"]
    for stage in ("capability", "activation", "jspace", "direction", "intervention"):
        assert counts[stage] > 0, stage


def test_all_three_modalities_ran_and_none_was_faked(run):
    result, _ = run
    assert result["available_modalities"] == ["text", "image", "spoken_audio"]
    assert result["blocked_modalities"] == []
    modalities = {
        record["modality"] for record in result["outcomes"]["activation"].records
    }
    assert modalities == {"text", "image", "spoken_audio"}


def test_invariance_gate_passed_before_any_intervention(run):
    result, _ = run
    assert result["invariance"]["passed"]
    assert result["invariance"]["capture_noop"]["passed"]
    assert all(entry["passed"] for entry in result["invariance"]["zero_intervention"])


def test_capability_gate_retained_the_planted_concepts(run):
    result, _ = run
    assert sorted(result["capability"]["retained_concepts"]) == sorted(K.MOCK_CONCEPTS)
    for per_modality in result["capability"]["per_concept"].values():
        for entry in per_modality.values():
            assert entry["accuracy"] >= 0.7
            assert entry["median_target_margin"] > 0


def test_jspace_finds_the_planted_cross_modal_structure(run):
    result, _ = run
    pairs = result["representational"]["pairs"]
    assert "text->image" in pairs and "image->text" in pairs
    for key in ("text->image", "image->text"):
        entry = pairs[key]
        assert entry["jspace_retrieval"]["top1_accuracy"] > entry["shuffled_control"][
            "p95_top1_accuracy"
        ]
        assert entry["jspace_separation"]["gap"] > 0


def test_retrieval_never_scores_a_group_against_its_own_twin(run):
    result, _ = run
    # Each concept contributes 3 test images x 2 captions; a query must not be
    # able to retrieve any group sharing its image.
    for entry in result["representational"]["pairs"].values():
        assert entry["jspace_retrieval"]["n_queries"] > 0
        assert entry["jspace_retrieval"]["chance_top1"] < 0.5


def test_causal_transfer_has_the_expected_sign_off_diagonal(run):
    result, _ = run
    rows = [
        row
        for row in result["interventions"]["rows"]
        if row["control_kind"] == "source_concept"
        and row["off_diagonal"]
        and row["alpha"] == 1.0
    ]
    assert rows
    for row in rows:
        assert row["mean_signed_target_effect"] > 0
        assert row["fraction_expected_sign"] == 1.0


def test_zero_control_is_an_exact_no_op_everywhere(run):
    result, _ = run
    zeros = [
        row for row in result["interventions"]["rows"] if row["control_kind"] == "zero"
    ]
    assert zeros
    for row in zeros:
        assert row["mean_signed_target_effect"] == pytest.approx(0.0, abs=1e-6)
        assert row["mean_activation_norm_ratio"] == pytest.approx(1.0, abs=1e-6)


def test_controls_are_weaker_than_the_source_derived_direction(run):
    result, _ = run
    by_key = {
        (row["pair"], row["control_kind"]): row
        for row in result["interventions"]["rows"]
        if row["alpha"] == 1.0
    }
    real = by_key[("text->image", "source_concept")]["mean_signed_target_effect"]
    for kind in ("random_norm_matched", "unrelated_concept"):
        control = by_key[("text->image", kind)]["mean_signed_target_effect"]
        assert real > 1.5 * max(control, 0.0)


def test_report_reaches_a_decision_and_labels_the_mock(run):
    result, root = run
    summary = result["summary"]
    assert summary["recommendation"] in ("GO", "WEAK GO", "NO-GO")
    assert summary["scientific_evidence"] is False
    assert "MOCK run: pipeline evidence only" in result["markdown"]
    assert (root / "run" / "report.md").is_file()
    assert (root / "run" / "summary.json").is_file()
    written = json.loads((root / "run" / "summary.json").read_text(encoding="utf-8"))
    assert written["recommendation"] == summary["recommendation"]


def test_the_report_answers_all_seven_questions(run):
    result, _ = run
    for number in range(1, 8):
        assert f"{number}. **" in result["markdown"]
    assert "not erasure" in result["markdown"]
    assert "environmental audio" in result["markdown"]


def test_mock_world_signal_is_strong_enough_to_reach_go(run):
    """The synthetic world plants a real shared direction, so a correct
    pipeline must find it. A NO-GO here means the pipeline broke, not that the
    data was ambiguous."""
    result, _ = run
    assert result["summary"]["recommendation"] == "GO"


def test_second_run_resumes_and_recomputes_nothing(run, tmp_path_factory):
    _, root = run
    again = K.run_mock_pilot(root / "data", root / "run")
    assert again["status"] == "resuming"
    for stage in ("capability", "activation", "jspace", "direction", "intervention"):
        outcome = again["outcomes"][stage]
        assert outcome.computed == 0, (stage, outcome.computed)
        assert outcome.reused > 0, stage
    assert again["summary"]["recommendation"] == "GO"


def test_a_changed_fingerprint_refuses_the_existing_run_directory(run):
    _, root = run
    stored = json.loads((root / "run" / "fingerprint.json").read_text(encoding="utf-8"))
    stored.pop("written_utc", None)
    stored["layers"] = tuple(stored["layers"])
    stored["model_revision"] = "a-different-revision"
    with pytest.raises(IncompatibleStateError, match="model_revision"):
        UnitStore(root / "run", RunFingerprint(**stored)).open()


def test_blocked_audio_completes_the_text_image_pilot(tmp_path):
    result = K.run_mock_pilot(
        tmp_path / "data", tmp_path / "run", supports_audio=False, n_permutations=10
    )
    assert result["available_modalities"] == ["text", "image"]
    assert result["blocked_modalities"] == ["spoken_audio"]
    assert not result["summary"]["criteria"]["spoken_audio_available"]["passed"]
    assert "spoken_audio" in result["summary"]["blocked_modalities"]
    # The text-image pilot still finishes and still reaches a decision.
    assert "text->image" in result["representational"]["pairs"]
    assert result["summary"]["recommendation"] in ("GO", "WEAK GO", "NO-GO")
    assert all(
        row["target_modality"] != "spoken_audio"
        for row in result["interventions"]["rows"]
    )


def test_derived_manifest_is_written_and_the_original_is_untouched(run):
    result, root = run
    derived = json.loads(
        (root / "run" / "derived_manifest.json").read_text(encoding="utf-8")
    )
    assert derived["schema_version"] == "jlens.mmpilot.manifest.v1"
    assert derived["conversion"]["original_manifest_mutated"] is False
    assert derived["source_checksum"].startswith("sha256:")
    original = json.loads(
        (root / "data" / "spokencoco_manifest.json").read_text(encoding="utf-8")
    )
    assert "captions" in original["data"][0]  # unchanged shape
