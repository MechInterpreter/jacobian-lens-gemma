# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Stage A: the completed run is read-only, and the targets are frozen first.

Two failure modes these tests exist to make impossible. A localization
conditioned on "some run in that directory" would inherit a provenance nobody
can state, so the fingerprint check refuses rather than warns. And a layer
scored on photographs chosen after another layer's numbers came back would
produce a depth difference that is partly an image difference, so the frozen set
is checksummed and drift is refused.
"""

import json

import pytest

from jlens.mmlocalize.targets import (
    CONCEPT_CONDITIONING_LIMITATION,
    LOCALIZATION_CONCEPTS,
    N_SOURCE_POSITIVE_IMAGES,
    N_TARGET_POSITIVE_IMAGES,
    POLICY_FRESH_DISJOINT,
    POLICY_REUSED_PAIRED,
    REUSED_POLICY_LIMITATION,
    CompletedRunError,
    TargetDriftError,
    TargetPolicyError,
    assert_same_targets_across_layers,
    audit_image_exclusions,
    choose_target_policy,
    completed_run_images,
    format_targets,
    freeze_targets,
    target_manifest,
    verify_completed_run,
)
from jlens.mmpilot.store import payload_checksum

FINGERPRINT_PAYLOAD = {
    "mode": "robustness",
    "model_repo_id": "google/gemma-4-E4B-it",
    "layers": [38],
}
FINGERPRINT = payload_checksum(FINGERPRINT_PAYLOAD)


def _completed_run(tmp_path, *, verdict="ROBUSTNESS_GO", payload=None):
    root = tmp_path / "mmrobust_robustness_20260804T154417"
    root.mkdir(parents=True, exist_ok=True)
    (root / "fingerprint.json").write_text(
        json.dumps(payload or FINGERPRINT_PAYLOAD), encoding="utf-8"
    )
    (root / "robustness_summary.json").write_text(
        json.dumps(
            {
                "fingerprint_digest": FINGERPRINT,
                "verdict": {"verdict": verdict, "selected_concepts": ["cat", "toilet"]},
            }
        ),
        encoding="utf-8",
    )
    return root


def _write_units(root, stage, records):
    directory = root / "units" / stage
    directory.mkdir(parents=True, exist_ok=True)
    for index, record in enumerate(records):
        (directory / f"unit{index:03d}.json").write_text(
            json.dumps({"payload": record}), encoding="utf-8"
        )


# ------------------------------------------- the completed run, verified


def test_a_matching_fingerprint_is_accepted_and_only_read(tmp_path):
    root = _completed_run(tmp_path)
    before = {p.name: p.read_bytes() for p in root.rglob("*") if p.is_file()}

    record = verify_completed_run(root, expect_fingerprint=FINGERPRINT)

    assert record["fingerprint"] == FINGERPRINT
    assert record["fingerprint_matches_pin"] is True
    assert record["verdict"] == "ROBUSTNESS_GO"
    assert record["read_only"] is True
    assert record["artifact_checksums"]
    after = {p.name: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert after == before, "verification must not modify the completed run"


def test_a_different_run_is_refused(tmp_path):
    root = _completed_run(tmp_path, payload={"mode": "something-else"})
    with pytest.raises(CompletedRunError, match="Refusing to condition"):
        verify_completed_run(root, expect_fingerprint=FINGERPRINT)


def test_a_missing_run_is_refused(tmp_path):
    with pytest.raises(CompletedRunError, match="not found"):
        verify_completed_run(tmp_path / "absent", expect_fingerprint=FINGERPRINT)


def test_a_run_without_a_fingerprint_is_refused(tmp_path):
    root = tmp_path / "no_fingerprint"
    root.mkdir()
    with pytest.raises(CompletedRunError, match="missing"):
        verify_completed_run(root, expect_fingerprint=FINGERPRINT)


def test_a_run_whose_artifacts_disagree_about_itself_is_refused(tmp_path):
    root = _completed_run(tmp_path)
    (root / "robustness_summary.json").write_text(
        json.dumps(
            {"fingerprint_digest": "sha256:something-else", "verdict": {"verdict": "ROBUSTNESS_GO"}}
        ),
        encoding="utf-8",
    )
    with pytest.raises(CompletedRunError, match="disagree"):
        verify_completed_run(root, expect_fingerprint=FINGERPRINT)


def test_an_unconfirmed_run_is_not_localized(tmp_path):
    root = _completed_run(tmp_path, verdict="ROBUSTNESS_NO_GO")
    with pytest.raises(CompletedRunError, match="does not localize an unconfirmed"):
        verify_completed_run(root, expect_fingerprint=FINGERPRINT)


def test_the_images_the_completed_run_touched_are_collected_by_role(tmp_path):
    root = _completed_run(tmp_path)
    _write_units(root, "intervention", [{"image_id": "imgA"}, {"image_id": "imgB"}])
    _write_units(root, "activation", [{"image_id": "imgC"}, {"image_id": "imgA"}])

    images = completed_run_images(root)

    assert images["causal_target_images"] == ["imgA", "imgB"]
    assert images["all_images"] == ["imgA", "imgB", "imgC"]
    assert images["n_causal_target_images"] == 2
    assert images["n_all_images"] == 3


def test_torn_unit_files_do_not_crash_the_scan(tmp_path):
    root = _completed_run(tmp_path)
    _write_units(root, "intervention", [{"image_id": "imgA"}])
    (root / "units" / "intervention" / "torn.json").write_text("{not json", encoding="utf-8")
    assert completed_run_images(root)["all_images"] == ["imgA"]


# ------------------------------------------------------- the policy choice


def test_ample_fresh_images_choose_the_fresh_policy():
    decision = choose_target_policy(
        n_available_fresh_images={"cat": 40, "toilet": 40},
        n_available_fresh_negatives=40,
    )
    assert decision["policy"] == POLICY_FRESH_DISJOINT
    assert decision["fresh_targets_feasible"] is True
    assert decision["limitation"] is None
    assert decision["decided_before_any_layer_result"] is True


def test_too_few_fresh_images_fall_back_and_carry_the_limitation():
    decision = choose_target_policy(
        n_available_fresh_images={"cat": 3, "toilet": 40},
        n_available_fresh_negatives=40,
    )
    assert decision["policy"] == POLICY_REUSED_PAIRED
    assert decision["shortfalls"]["cat"]["available"] == 3
    assert decision["limitation"] == REUSED_POLICY_LIMITATION
    assert "WITHIN-SAMPLE" in decision["limitation"]


def test_too_few_fresh_negatives_also_fall_back():
    decision = choose_target_policy(
        n_available_fresh_images={"cat": 40, "toilet": 40},
        n_available_fresh_negatives=1,
    )
    assert decision["policy"] == POLICY_REUSED_PAIRED
    assert decision["negatives_short"] is True


def test_the_policies_are_never_mixed():
    decision = choose_target_policy(
        n_available_fresh_images={"cat": 40, "toilet": 40},
        n_available_fresh_negatives=40,
    )
    assert decision["policies_are_never_mixed"] is True


# ------------------------------------------------------- freezing targets


def _images(prefix, n, start=0):
    return [f"{prefix}{index:03d}" for index in range(start, start + n)]


def _valid_kwargs(**overrides):
    kwargs = {
        "policy": POLICY_FRESH_DISJOINT,
        "source_positive_images": {
            "cat": _images("cat_src", 4),
            "toilet": _images("toi_src", 4),
        },
        "source_negative_images": {
            "cat": _images("neg_src", 4),
            "toilet": _images("neg_src", 4),
        },
        "target_positive_images": {
            "cat": _images("cat_tgt", 4),
            "toilet": _images("toi_tgt", 4),
        },
        "target_negative_images": {
            "cat": _images("neg_tgt", 4),
            "toilet": _images("neg_tgt", 4),
        },
        "completed_run_images": ["old001", "old002"],
        "concepts": LOCALIZATION_CONCEPTS,
    }
    kwargs.update(overrides)
    return kwargs


def test_a_valid_target_set_freezes_with_a_checksum():
    targets = freeze_targets(**_valid_kwargs())
    assert targets.policy == POLICY_FRESH_DISJOINT
    assert targets.concepts == LOCALIZATION_CONCEPTS
    assert targets.checksum.startswith("sha256:")
    assert targets.limitation is None
    assert len(targets.all_target_images()) == 12   # 4+4 positives, 4 shared negatives
    assert not set(targets.all_source_images()) & set(targets.all_target_images())


def test_the_checksum_changes_when_a_photograph_changes():
    first = freeze_targets(**_valid_kwargs()).checksum
    second = freeze_targets(
        **_valid_kwargs(
            target_positive_images={
                "cat": _images("cat_tgt", 4, start=10),
                "toilet": _images("toi_tgt", 4),
            }
        )
    ).checksum
    assert first != second


def test_a_short_cell_refuses_rather_than_shrinking():
    with pytest.raises(TargetPolicyError, match="design states"):
        freeze_targets(
            **_valid_kwargs(
                target_positive_images={
                    "cat": _images("cat_tgt", 2),
                    "toilet": _images("toi_tgt", 4),
                }
            )
        )


def test_a_source_image_reused_as_a_target_is_refused():
    with pytest.raises(TargetPolicyError, match="both a source-training image"):
        freeze_targets(
            **_valid_kwargs(
                target_positive_images={
                    "cat": _images("cat_src", 4),      # the training photographs
                    "toilet": _images("toi_tgt", 4),
                }
            )
        )


def test_an_image_that_is_both_a_positive_and_a_negative_target_is_refused():
    with pytest.raises(TargetPolicyError, match="both a held-out"):
        freeze_targets(
            **_valid_kwargs(
                target_negative_images={
                    "cat": _images("cat_tgt", 4),
                    "toilet": _images("neg_tgt", 4),
                }
            )
        )


def test_the_fresh_policy_refuses_any_overlap_with_the_completed_run():
    with pytest.raises(TargetPolicyError, match="already used by the completed"):
        freeze_targets(
            **_valid_kwargs(
                completed_run_images=["cat_tgt000", "other"],
            )
        )


def test_the_fallback_policy_permits_reuse_and_says_so():
    targets = freeze_targets(
        **_valid_kwargs(
            policy=POLICY_REUSED_PAIRED,
            completed_run_images=["cat_tgt000"],
        )
    )
    assert targets.policy == POLICY_REUSED_PAIRED
    assert targets.limitation == REUSED_POLICY_LIMITATION


def test_an_unknown_policy_is_refused():
    with pytest.raises(TargetPolicyError, match="unknown target policy"):
        freeze_targets(**_valid_kwargs(policy="whatever_looks_convenient"))


# ------------------------------------------------------- the audit + manifest


def test_the_audit_reports_counts_and_an_empty_intersection():
    targets = freeze_targets(**_valid_kwargs())
    audit = audit_image_exclusions(
        targets,
        completed_run={
            "run_dir": "/runs/completed",
            "all_images": ["old001", "old002"],
            "causal_target_images": ["old001"],
        },
        n_available_images=500,
    )
    assert audit["n_completed_run_images"] == 2
    assert audit["n_completed_run_causal_target_images"] == 1
    assert audit["n_overlap_all"] == 0
    assert audit["n_overlap_causal_targets"] == 0
    assert audit["fresh_policy_satisfied"] is True
    assert audit["source_target_overlap"] == []
    assert audit["n_available_images_before_exclusion"] == 500
    for concept in LOCALIZATION_CONCEPTS:
        assert len(audit["per_concept"][concept]["target_positive"]) == 4


def test_the_manifest_binds_the_layers_the_targets_and_the_completed_run():
    targets = freeze_targets(**_valid_kwargs())
    audit = audit_image_exclusions(
        targets, completed_run={"run_dir": "/runs/completed", "all_images": []}
    )
    manifest = target_manifest(
        targets,
        audit=audit,
        completed_run={"run_dir": "/runs/completed", "fingerprint": FINGERPRINT},
        layers=(20, 26, 32, 38),
    )
    assert manifest["layers"] == [20, 26, 32, 38]
    assert manifest["same_targets_at_every_layer"] is True
    assert manifest["frozen_before_any_layer_result"] is True
    assert manifest["target_checksum"] == targets.checksum
    assert manifest["manifest_checksum"].startswith("sha256:")
    assert manifest["completed_run"]["read_only"] is True
    assert manifest["concept_conditioning_limitation"] == CONCEPT_CONDITIONING_LIMITATION


# ----------------------------------------------- the same images every layer


def test_identical_images_at_every_layer_are_paired():
    targets = freeze_targets(**_valid_kwargs())
    expected = targets.all_target_images()
    record = assert_same_targets_across_layers(
        targets, {layer: expected for layer in (20, 26, 32, 38)}
    )
    assert record["paired"] is True
    assert record["n_target_images"] == len(expected)
    assert record["layers_checked"] == [20, 26, 32, 38]


def test_a_layer_scored_on_different_photographs_is_refused():
    targets = freeze_targets(**_valid_kwargs())
    expected = targets.all_target_images()
    with pytest.raises(TargetDriftError, match="not paired"):
        assert_same_targets_across_layers(
            targets,
            {38: expected, 32: expected[:-1]},
        )


def test_an_extra_photograph_at_one_layer_is_also_refused():
    targets = freeze_targets(**_valid_kwargs())
    expected = targets.all_target_images()
    with pytest.raises(TargetDriftError):
        assert_same_targets_across_layers(
            targets, {38: expected, 20: [*expected, "smuggled_in"]}
        )


# ----------------------------------------------------------- the printout


def test_the_printed_block_states_the_conditioning_limitation():
    targets = freeze_targets(**_valid_kwargs())
    audit = audit_image_exclusions(
        targets, completed_run={"run_dir": "/runs/c", "all_images": []}
    )
    text = format_targets(targets, audit)
    assert "FROZEN LOCALIZATION TARGETS" in text
    assert "before any layer result exists" in text
    assert "concept-general prevalence" in text
    assert targets.checksum in text


def test_the_fallback_printout_shows_the_within_sample_warning():
    targets = freeze_targets(**_valid_kwargs(policy=POLICY_REUSED_PAIRED))
    audit = audit_image_exclusions(
        targets, completed_run={"run_dir": "/runs/c", "all_images": []}
    )
    assert "WITHIN-SAMPLE" in format_targets(targets, audit)


def test_the_design_counts_are_the_commissioned_ones():
    assert LOCALIZATION_CONCEPTS == ("cat", "toilet")
    assert N_SOURCE_POSITIVE_IMAGES == 4
    assert N_TARGET_POSITIVE_IMAGES == 4
