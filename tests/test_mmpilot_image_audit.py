# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""The CPU-only audit runner: resume, atomicity, preservation, and the verdict.

The end-to-end fixture is a complete MOCK pilot run, so the audit is exercised
against artifacts a real run actually produces rather than against a fixture
shaped to suit it. Its GO says nothing about Gemma, and neither does the
amended verdict computed from it — what is under test is the arithmetic and the
refusals.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from jlens.mmpilot import mock as K
from jlens.mmpilot.image_audit import (
    ARTIFACTS,
    AUDIT_BLOCKED,
    GO_CONFIRMED,
    GO_REQUIRES,
    NO_GO,
    PROTECTED_NAMES,
    WEAK_GO,
    AuditFingerprint,
    AuditInputError,
    AuditWorkspace,
    VerdictConfig,
    amended_verdict,
    file_checksum,
    load_run,
    replication_report,
    run_image_independence_audit,
)
from jlens.mmpilot.store import IncompatibleStateError


@pytest.fixture(scope="module")
def mock_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("audit")
    K.run_mock_pilot(root / "data", root / "run")
    return root / "run"


@pytest.fixture(scope="module")
def audited(mock_run):
    return run_image_independence_audit(mock_run)


# ------------------------------------------------------------ the whole run


def test_the_audit_runs_end_to_end_on_cpu(audited):
    assert audited["ok"]
    assert audited["model_loaded"] is False
    assert audited["verdict"] in (GO_CONFIRMED, WEAK_GO, NO_GO)


def test_the_audit_imports_no_model_loader(mock_run):
    """Checked in a fresh interpreter on purpose.

    ``sys.modules`` in this session is contaminated by every other test that
    ran before it, so an in-process assertion would pass or fail on test
    ordering rather than on what the audit imports.
    """
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, json;"
            "from jlens.mmpilot.image_audit import run_image_independence_audit;"
            "run_image_independence_audit(sys.argv[1]);"
            "print(json.dumps(sorted(n for n in sys.modules "
            "if n.startswith(('transformers', 'jlens.gemma4')))))",
            str(mock_run),
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parent.parent,
    )
    assert probe.returncode == 0, probe.stderr[-2000:]
    assert json.loads(probe.stdout.strip().splitlines()[-1]) == []


def test_the_audit_finds_the_planted_same_image_dependence(audited):
    """The mock keeps two synchronized groups per image, exactly as the pilot
    subset builder does."""
    audit = audited["audit"]
    assert audit["n_groups"] == 2 * audit["n_distinct_images"]
    assert audit["n_images_with_multiple_groups"] == audit["n_distinct_images"]
    assert audit["independent_unit"] == "image_id"
    assert audit["hard_failures"] == []
    for block in audit["by_split"].values():
        assert block["groups_per_image_histogram"] == {"2": block["n_distinct_images"]}


def test_the_mock_split_is_image_disjoint(audited):
    audit = audited["audit"]
    assert audit["train_test_image_overlap"] == []
    assert audit["sibling_groups_crossing_splits"] == []


def test_corrected_retrieval_excluded_same_image_targets(audited):
    for entry in audited["representational"]["pairs"].values():
        exclusions = entry["exclusions"]
        assert exclusions["n_excluded_same_group"] > 0
        assert exclusions["n_excluded_same_image_different_group"] > 0
        assert exclusions["n_sources_without_eligible_target"] == 0
        assert entry["jspace_retrieval"]["n_queries"] > 0


def test_every_intervention_unit_was_reaggregated_and_none_rerun(audited, mock_run):
    stored = len(list((mock_run / "units" / "intervention").glob("*.json")))
    image_level = audited["interventions_image_level"]

    assert image_level["n_records"] == stored
    assert image_level["independent_unit"] == "image_id"
    assert audited["summary"]["interventions_rerun"] is False
    for row in image_level["rows"]:
        assert row["n_distinct_images"] <= row["n_groups"]
        assert row["group_level"] is not None, "provenance is kept, not replaced"


def test_the_amended_verdict_is_one_of_the_four_and_lists_its_criteria(audited):
    verdict = audited["summary"]["verdict"]
    assert verdict["verdict"] in (GO_CONFIRMED, WEAK_GO, NO_GO, AUDIT_BLOCKED)
    assert set(GO_REQUIRES) <= set(verdict["criteria_status"])
    assert verdict["late_layer_limitation"]
    assert "replication" in verdict


# ------------------------------------------------------------- preservation


def test_the_original_report_and_summary_are_left_byte_identical(mock_run):
    before = {
        name: file_checksum(mock_run / name)
        for name in ("report.md", "summary.json")
    }
    result = run_image_independence_audit(mock_run)
    after = {
        name: file_checksum(mock_run / name)
        for name in ("report.md", "summary.json")
    }

    assert after == before
    assert result["preservation"]["all_unchanged"] is True
    assert result["preservation"]["unchanged"]["report.md"] is True
    assert result["preservation"]["unchanged"]["summary.json"] is True


def test_no_unit_file_is_touched(mock_run):
    units = sorted((mock_run / "units").rglob("*.json"))
    before = {path: file_checksum(path) for path in units}
    run_image_independence_audit(mock_run)

    assert {path: file_checksum(path) for path in units} == before


def test_the_audit_writes_only_new_versioned_artifacts(mock_run, audited):
    for relative in ARTIFACTS.values():
        assert (mock_run / relative).is_file(), relative
        assert relative not in PROTECTED_NAMES
    assert (mock_run / "report_image_disjoint_v1.md.sha256").is_file()
    assert (mock_run / "audits" / "audit_fingerprint.json").is_file()


# ------------------------------------------------------ atomicity and resume


def test_saving_is_atomic_and_leaves_no_temporary_files(mock_run, audited):
    assert not list(mock_run.rglob("*.tmp.*")), "a torn write was left behind"
    payload = json.loads(
        (mock_run / ARTIFACTS["audit"]).read_text(encoding="utf-8")
    )
    assert payload["checksum"].startswith("sha256:")
    assert payload["fingerprint_digest"].startswith("sha256:")
    assert payload["schema"] == "jlens.mmpilot.audit_artifact.v1"


def test_a_compatible_rerun_resumes_and_reuses_the_expensive_artifacts(mock_run):
    first = run_image_independence_audit(mock_run)
    second = run_image_independence_audit(mock_run)

    assert second["status"] == "resuming"
    assert set(second["resume"]["reused"]) >= {
        "audit",
        "representational",
        "interventions",
    }
    assert second["verdict"] == first["verdict"]
    assert second["resume"]["invalid_artifacts"] == []


def test_a_checksum_invalid_artifact_is_recomputed_rather_than_trusted(mock_run):
    run_image_independence_audit(mock_run)
    path = mock_run / ARTIFACTS["audit"]
    record = json.loads(path.read_text(encoding="utf-8"))
    record["payload"]["n_distinct_images"] = 999_999
    path.write_text(json.dumps(record), encoding="utf-8")

    result = run_image_independence_audit(mock_run)
    assert str(path) in result["resume"]["invalid_artifacts"]
    assert "audit" in result["resume"]["computed"]
    assert result["audit"]["n_distinct_images"] != 999_999


def test_an_incompatible_audit_fingerprint_is_refused_not_mixed(mock_run):
    run_image_independence_audit(mock_run)

    with pytest.raises(IncompatibleStateError, match="refusing to reuse or mix"):
        # A different verdict configuration is a different measurement.
        run_image_independence_audit(
            mock_run, config=VerdictConfig(min_distinct_images=99)
        )


def test_the_fingerprint_binds_every_rule_version(mock_run):
    run = load_run(mock_run)
    fingerprint = AuditFingerprint(
        original_run_fingerprint_digest=run.run_fingerprint.digest,
        original_subset_checksum=run.subset_checksum,
        expanded_manifest_checksum=run.expanded_manifest_checksum,
        lens_checksum=run.run_fingerprint.lens_checksum,
        selected_layer=run.selected_layer,
        verdict_config=VerdictConfig().to_dict(),
    )
    payload = fingerprint.to_dict()

    for field in (
        "original_run_fingerprint_digest",
        "original_subset_checksum",
        "expanded_manifest_checksum",
        "lens_checksum",
        "selected_layer",
        "image_identity_rule_version",
        "representational_exclusion_rule_version",
        "causal_aggregation_version",
        "verdict_config",
    ):
        assert field in payload, field
    assert fingerprint.digest.startswith("sha256:")


def test_a_layer_change_produces_a_different_audit_fingerprint(mock_run):
    run = load_run(mock_run)
    base = AuditFingerprint(
        original_run_fingerprint_digest=run.run_fingerprint.digest,
        original_subset_checksum=run.subset_checksum,
        expanded_manifest_checksum=run.expanded_manifest_checksum,
        lens_checksum=run.run_fingerprint.lens_checksum,
        selected_layer=run.selected_layer,
    )
    other = AuditFingerprint(**{**base.to_dict(), "selected_layer": 999})
    assert base.digest != other.digest
    assert any("selected_layer" in line for line in base.differences(other.to_dict()))


# --------------------------------------------------------------- refusals


def test_a_directory_that_is_not_a_run_is_refused(tmp_path):
    with pytest.raises(AuditInputError, match="fingerprint.json"):
        load_run(tmp_path)


def test_a_missing_directory_is_refused(tmp_path):
    with pytest.raises(AuditInputError, match="not a directory"):
        load_run(tmp_path / "nowhere")


def test_a_subset_path_that_is_not_a_subset_is_refused(mock_run, tmp_path):
    bogus = tmp_path / "bogus_subset.json"
    bogus.write_text(json.dumps({"not": "a subset"}), encoding="utf-8")

    with pytest.raises(AuditInputError, match="does not look like a pilot subset"):
        load_run(mock_run, subset_path=bogus)


def test_unresolvable_image_identity_blocks_the_audit(tmp_path, mock_run):
    """An audit that guessed at identity would answer a different question."""
    import shutil

    broken = tmp_path / "broken_run"
    shutil.copytree(mock_run, broken)
    if (broken / "audits").is_dir():
        shutil.rmtree(broken / "audits")
    # Every unit stays checksum-valid; only the image identity is gone. The
    # audit must stop rather than fall back on the group id.
    for path in (broken / "units").rglob("*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        if "image_id" not in record["payload"]:
            continue
        record["payload"]["image_id"] = ""
        record["unit_checksum"] = _checksum(record["payload"])
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    result = run_image_independence_audit(broken)
    assert result["ok"] is False
    assert result["verdict"] == AUDIT_BLOCKED
    assert "no resolvable image identity" in result["rationale"]
    assert result["preservation"]["all_unchanged"] is True


def test_ambiguous_image_identity_blocks_the_audit(tmp_path, mock_run):
    import shutil

    ambiguous = tmp_path / "ambiguous_run"
    shutil.copytree(mock_run, ambiguous)
    workspace_dir = ambiguous / "audits"
    if workspace_dir.is_dir():
        shutil.rmtree(workspace_dir)

    # One group claiming two different photographs.
    paths = sorted((ambiguous / "units" / "activation").glob("*.json"))
    first = json.loads(paths[0].read_text(encoding="utf-8"))
    second = json.loads(paths[1].read_text(encoding="utf-8"))
    second["payload"]["group_id"] = first["payload"]["group_id"]
    second["payload"]["sample_id"] = first["payload"]["sample_id"]
    second["payload"]["image_id"] = "definitely-a-different-image"
    second["unit_checksum"] = _checksum(second["payload"])
    paths[1].write_text(json.dumps(second, indent=2), encoding="utf-8")

    result = run_image_independence_audit(ambiguous)
    assert result["ok"] is False
    assert result["verdict"] == AUDIT_BLOCKED
    assert "ambiguous" in result["rationale"]
    assert "AUDIT_BLOCKED" in result["report_markdown"]
    assert "not a finding against the original result" in result["report_markdown"]


def _checksum(payload):
    from jlens.mmpilot.store import payload_checksum

    return payload_checksum(payload)


def test_train_test_image_leakage_blocks_the_audit(tmp_path, mock_run):
    import shutil

    leaky = tmp_path / "leaky_run"
    shutil.copytree(mock_run, leaky)
    if (leaky / "audits").is_dir():
        shutil.rmtree(leaky / "audits")

    # Move one sibling group of a *training* image into the test split. Every
    # unit mentioning that group is patched consistently, so identity stays
    # resolvable and the leakage — not an inconsistency — is what stops the
    # audit. This is the shape the failure would really take: one photograph
    # with a caption on each side of the split.
    units = {
        path: json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((leaky / "units").rglob("*.json"))
    }
    by_group = {}
    for record in units.values():
        payload = record["payload"]
        if payload.get("split") == "train" and payload.get("image_id"):
            by_group.setdefault(payload["image_id"], set()).add(payload["group_id"])
    shared_image = next(
        image_id for image_id, groups in sorted(by_group.items()) if len(groups) > 1
    )
    leaked_group = sorted(by_group[shared_image])[0]
    for path, record in units.items():
        if record["payload"].get("group_id") == leaked_group and record[
            "payload"
        ].get("split"):
            record["payload"]["split"] = "test"
            record["unit_checksum"] = _checksum(record["payload"])
            path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    result = run_image_independence_audit(leaky)
    assert result["ok"] is False
    assert result["verdict"] == AUDIT_BLOCKED
    assert "image" in result["rationale"]
    assert result["audit"]["hard_failures"]


# ------------------------------------------------------------- the verdict


def _row(**overrides):
    row = {
        "concept": "cat",
        "source_modality": "text",
        "target_modality": "image",
        "pair": "text->image",
        "off_diagonal": True,
        "layer": 38,
        "control_kind": "source_concept",
        "alpha": 0.5,
        "n": 2,
        "n_distinct_images": 2,
        "n_groups": 2,
        "mean_signed_target_effect": 10.0,
        "fraction_expected_sign": 1.0,
        "mean_abs_unrelated_change": 0.1,
        "mean_activation_norm_ratio": 1.0,
        "n_prediction_changes": 1,
    }
    row.update(overrides)
    return row


def _verdict_inputs(rows, **overrides):
    payload = {
        "capability": {
            "text_image_retained_concepts": ["cat", "dog", "bus", "pizza"],
            "per_concept": {},
        },
        "lens_validation": {
            "lens_checksum": "sha256:x",
            "native_readout_validation": {
                "status": "validated_text_only",
                "native_readout_layers_passing": [38],
            },
        },
        "code_stats": {"n": 10, "median_explained_fraction": 0.07},
        "reconstruction_control": {"n_records": 4},
        "representational": {
            "pairs": {
                pair: {
                    "jspace_retrieval": {"top1_accuracy": 1.0, "n_queries": 8},
                    "shuffled_control": {"p95_top1_accuracy": 0.5},
                    "raw_residual_retrieval": {"top1_accuracy": 0.5},
                    "jspace_separation": {"gap": 0.5},
                    "raw_residual_separation": {"gap": 0.1},
                }
                for pair in ("text->image", "image->text")
            }
        },
        "interventions": {"rows": rows, "aggregation_version": "v"},
        "audit": {
            "hard_failures": [],
            "n_groups": 4,
            "n_distinct_images": 2,
            "image_identity_rule_version": "v",
            "train_test_image_overlap": [],
            "sibling_groups_crossing_splits": [],
        },
    }
    payload.update(overrides)
    return payload


def _controls(effect_random=1.0, effect_unrelated=1.0):
    return [
        _row(control_kind="random_norm_matched", mean_signed_target_effect=effect_random),
        _row(control_kind="unrelated_concept", mean_signed_target_effect=effect_unrelated),
    ]


def test_a_clean_corrected_result_confirms_the_go():
    verdict = amended_verdict(**_verdict_inputs([_row(), *_controls()]))
    assert verdict["verdict"] == GO_CONFIRMED
    assert all(
        verdict["criteria_status"][name] == "PASS" for name in GO_REQUIRES
    )


def test_evidence_resting_on_one_photograph_cannot_confirm_a_go():
    rows = [_row(n_distinct_images=1, n=1), *_controls()]
    verdict = amended_verdict(**_verdict_inputs(rows))

    assert verdict["verdict"] == WEAK_GO
    assert verdict["criteria_status"]["evidence_not_single_image"] == "FAIL"


def test_an_effect_not_exceeding_the_external_unrelated_control_is_not_a_go():
    rows = [_row(mean_signed_target_effect=1.0), *_controls(effect_unrelated=2.0)]
    verdict = amended_verdict(**_verdict_inputs(rows))

    assert verdict["verdict"] == WEAK_GO
    assert (
        verdict["criteria_status"]["source_exceeds_random_and_external_unrelated"]
        == "FAIL"
    )


def test_image_leakage_forces_a_no_go_even_with_a_perfect_effect():
    inputs = _verdict_inputs([_row(), *_controls()])
    inputs["audit"] = {
        **inputs["audit"],
        "hard_failures": [{"kind": "train_test_image_overlap", "detail": "x"}],
        "train_test_image_overlap": ["100"],
    }
    verdict = amended_verdict(**inputs)

    assert verdict["verdict"] == NO_GO
    assert verdict["criteria_status"]["no_train_test_image_leakage"] == "FAIL"


def test_fewer_than_all_concepts_passing_behaviorally_is_not_a_confirmed_go():
    inputs = _verdict_inputs([_row(), *_controls()])
    inputs["capability"] = {
        "text_image_retained_concepts": ["cat", "dog"],
        "per_concept": {},
    }
    verdict = amended_verdict(**inputs)

    assert verdict["verdict"] == NO_GO
    assert verdict["criteria_status"]["behavioral_capability"] == "FAIL"


def test_the_original_go_is_never_forced():
    """A run whose corrected representation fails gets NO_GO, whatever the
    original report said."""
    inputs = _verdict_inputs([_row(), *_controls()])
    for entry in inputs["representational"]["pairs"].values():
        entry["shuffled_control"]["p95_top1_accuracy"] = 1.0
    verdict = amended_verdict(**inputs)

    assert verdict["verdict"] == NO_GO
    assert verdict["criteria_status"]["representational_structure"] == "FAIL"


def test_replication_is_reported_per_concept_and_per_direction():
    rows = []
    for concept in ("cat", "giraffe"):
        for pair in ("text->image", "image->text"):
            source, target = pair.split("->")
            common = {
                "concept": concept,
                "pair": pair,
                "source_modality": source,
                "target_modality": target,
            }
            # Only cat/text->image clears the controls.
            effect = 10.0 if (concept, pair) == ("cat", "text->image") else 0.5
            rows.append(_row(mean_signed_target_effect=effect, **common))
            rows.append(
                _row(
                    control_kind="random_norm_matched",
                    mean_signed_target_effect=1.0,
                    **common,
                )
            )
            rows.append(
                _row(
                    control_kind="unrelated_concept",
                    mean_signed_target_effect=1.0,
                    **common,
                )
            )

    replication = replication_report(rows, VerdictConfig())
    assert replication["by_concept"] == {"cat": True, "giraffe": False}
    assert replication["by_direction"] == {"text->image": True, "image->text": False}
    assert replication["replicates_across_concepts"] is False
    assert replication["replicates_across_directions"] is False


# --------------------------------------------------------------- workspace


def test_the_workspace_refuses_to_write_a_protected_name(tmp_path):
    fingerprint = AuditFingerprint(
        original_run_fingerprint_digest="sha256:a",
        original_subset_checksum="absent",
        expanded_manifest_checksum="absent",
        lens_checksum="sha256:b",
        selected_layer=38,
    )
    workspace = AuditWorkspace(tmp_path, fingerprint)
    workspace.open()

    for relative in ARTIFACTS.values():
        assert relative.rsplit("/", 1)[-1] not in PROTECTED_NAMES, relative
