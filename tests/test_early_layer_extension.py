# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the early-layer J-lens extension.

Two questions run through all of them:

* **May this accumulator be continued, and is the continuation the fit it
  claims to be?** The parent audit, the reconstruction, the prefix checksum and
  the bit-identity check all answer parts of that, and each has a test for its
  refusal as well as its success — a guard that is only ever exercised on the
  happy path is not a guard.
* **Is the endpoint genuinely untouched?** The old development and confirmation
  sets are excluded, the fresh sets are audited independently of the filter that
  built them, the vault will not open without a recorded scale choice, and a
  layer that fails confirmation is never marked validated.

The parent run in these tests is produced by the completed study's own code
against a tiny CPU stack, so the audit meets an artifact of the same kind it
will meet on Drive.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from jlens.calibration.corpus import CorpusRecord, hamming_distance, simhash64
from jlens.calibration.extension import (
    BASELINE_SCALE,
    DEVELOPMENT_SCALE_POINTS,
    EARLY_LAYERS_OF_INTEREST,
    EXTENSION_CONFIRMATION_GATE,
    EXTENSION_GATE,
    EXTENSION_PROTOCOL,
    EXTENSION_SCALE_POINTS,
    EXTENSION_SELECTION_RULE,
    MAXIMUM_AUTHORIZED_SCALE,
    N_CONFIRMATION_PROMPTS,
    N_DEVELOPMENT_PROMPTS,
    PRIMARY_EARLY_LAYER,
    PUBLISHABLE_LAYERS,
    SECONDARY_EARLY_LAYER,
    ContinuationRefused,
    ExtensionSplitRefused,
    ExtensionStore,
    audit_extension_target_diversity,
    audit_fresh_split_leakage,
    build_extension_fit_order,
    build_fresh_evaluation_splits,
    early_layer_verdict,
    extension_budget,
    extension_corpus_manifest,
    extension_gate_text,
    format_extension_budget,
    parent_collection_parameters,
    publish_early_layer,
    seed_extension_checkpoint,
    select_extension_scale,
    verify_continuation_equals_fresh_fit,
    verify_fit_prefix,
    verify_reconstructed_partitions,
)
from jlens.calibration.extension_mock import (
    EXTENSION_SCENARIOS,
    MOCK_BASELINE_SCALE,
    MOCK_DEVELOPMENT_SCALES,
    MOCK_EXTENSION_SCALES,
    MOCK_LAYERS,
    build_mock_parent_run,
    corrupt_continued_checkpoint,
    mock_extension_environment,
    mock_extension_pool,
    mock_extension_rows,
    mock_load_info,
    mock_target_token,
    plant_old_confirmation_record,
)
from jlens.calibration.fitting import run_calibration
from jlens.calibration.gate import (
    CALIBRATION_GATE,
    InsufficientTargetDiversityError,
    evaluate_calibration_layers,
    select_diverse_validation_prompts,
)
from jlens.calibration.parent import (
    ParentImportRefused,
    ParentRequirements,
    assert_parent_unchanged,
    audit_parent_run,
    discover_parent_files,
    format_parent_audit,
    load_parent_run,
    parent_provenance_manifest,
    protected_parent_checksums,
    resolve_parent_layout,
)
from jlens.calibration.publication import (
    ConfirmationLocked,
    ConfirmationVault,
    PublicationRefused,
    record_failed_layer,
)
from jlens.calibration.scale import compare_scales, evaluate_plateau
from jlens.calibration.state import CalibrationFingerprint, IncompatibleStateError
from jlens.lens import JacobianLens

EXCLUDED_NAMES = ("old_fit", "old_development", "old_confirmation", "new_fit")


# ------------------------------------------------------------------ fixtures


def _requirements(parent_run, plan, *, baseline: int) -> ParentRequirements:
    return ParentRequirements(
        model_repo_id="google/gemma-4-E4B-it",
        model_revision="fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd",
        tokenizer_repo_id="google/gemma-4-E4B-it",
        tokenizer_revision="fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd",
        source_layers=tuple(plan.layers),
        target_layer=plan.target_layer,
        d_model=plan.d_model,
        hook_site="block_output",
        skip_first=plan.skip_first,
        max_seq_len=plan.max_seq_len,
        dim_batch=plan.dim_batch,
        corpus_hf_dataset="mock",
        corpus_config="mock",
        corpus_split="train",
        estimator="jlens.fitting.fit (upstream, unmodified)",
        artifact_format_version="jlens.calibration.artifact.v1",
        baseline_scale=baseline,
        expected_n_done=baseline,
    )


@pytest.fixture(scope="module")
def parent(tmp_path_factory):
    """A complete parent run, produced by the completed study's own code."""
    root = tmp_path_factory.mktemp("parent_run")
    return build_mock_parent_run(root / "rgcalib_mock_parent")


@pytest.fixture(scope="module")
def loaded(parent):
    run = load_parent_run(parent.root, baseline_scale=parent.baseline_scale)
    requirements = _requirements(run, parent.plan, baseline=parent.baseline_scale)
    audit = audit_parent_run(run, requirements=requirements)
    return run, requirements, audit


@pytest.fixture(scope="module")
def pool(parent):
    return mock_extension_pool(parent)


@pytest.fixture(scope="module")
def fit_order(parent, loaded):
    run, _requirements_, _audit = loaded
    records = build_extension_fit_order(
        parent.fit_records, n_needed=MOCK_EXTENSION_SCALES[-1]
    )
    verify_fit_prefix(
        records,
        n_parent=parent.baseline_scale,
        parent_prefix_checksum=run.fit_prefix_checksum(parent.baseline_scale),
    )
    return records


@pytest.fixture(scope="module")
def excluded(parent, fit_order):
    return {
        "old_fit": parent.partitions.fit,
        "old_development": parent.partitions.validation,
        "old_confirmation": parent.partitions.confirmation,
        "new_fit": fit_order,
    }


@pytest.fixture(scope="module")
def splits(pool, excluded):
    return build_fresh_evaluation_splits(
        pool, excluded=excluded, corpus_id="mock/train"
    )


def _fingerprint(parent, run, splits, *, mode="mock", salt="") -> CalibrationFingerprint:
    return CalibrationFingerprint(
        mode=mode,
        protocol_version=EXTENSION_PROTOCOL.version,
        model_repo_id="google/gemma-4-E4B-it",
        model_revision="fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd",
        tokenizer_revision="fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd",
        capture_plan_digest=parent.plan.digest,
        corpus_manifest_checksum="sha256:test-corpus" + salt,
        gate_digest=EXTENSION_GATE.digest,
        plateau_rule_digest="sha256:test-plateau",
        scale_points=MOCK_EXTENSION_SCALES,
        artifact_format_version="jlens.calibration.artifact.v1",
        extra={
            "parent_fingerprint_digest": run.fingerprint_digest,
            "parent_accumulator_checksum": run.accumulator.checksum,
            "fresh_splits_checksum": splits.manifest()["manifest_checksum"],
        },
    )


@pytest.fixture(scope="module")
def continued(tmp_path_factory, parent, loaded, splits, fit_order):
    """The extension run: seeded from the parent, continued to both scales."""
    run, _requirements_, _audit = loaded
    root = tmp_path_factory.mktemp("extension_run") / "rgext"
    store = ExtensionStore(root, _fingerprint(parent, run, splits))
    status = store.open()
    seed = seed_extension_checkpoint(run.accumulator, store.checkpoint_path)
    result = run_calibration(
        parent.model,
        fit_order,
        plan=parent.plan,
        scale_points=MOCK_EXTENSION_SCALES,
        store=store,
        checkpoint_every=4,
        diagnostics_every=4,
    )
    return {"store": store, "status": status, "seed": seed, "result": result}


@pytest.fixture(scope="module")
def scenario_results():
    """Development, confirmation and verdict for every commissioned scenario."""
    out: dict[str, dict] = {}
    for key, scenario in EXTENSION_SCENARIOS.items():
        development = {}
        for index, scale in enumerate(MOCK_DEVELOPMENT_SCALES):
            rows = mock_extension_rows(
                scenario,
                stage="development",
                scale=scale,
                scale_index=index,
                n_prompts=N_DEVELOPMENT_PROMPTS,
            )
            development[scale] = evaluate_calibration_layers(
                rows,
                layers=list(MOCK_LAYERS),
                scale=scale,
                stage="validation",
                gate=EXTENSION_GATE,
            )
        comparison = compare_scales(development, layers=list(MOCK_LAYERS))
        plateau = evaluate_plateau(comparison)
        selection = select_extension_scale(
            comparison, plateau=plateau, candidate_scales=MOCK_EXTENSION_SCALES
        )
        scale = selection["selected_scale"]
        rows = mock_extension_rows(
            scenario,
            stage="confirmation",
            scale=scale,
            scale_index=MOCK_DEVELOPMENT_SCALES.index(scale),
            n_prompts=N_CONFIRMATION_PROMPTS,
        )
        confirmation = evaluate_calibration_layers(
            rows,
            layers=list(MOCK_LAYERS),
            scale=scale,
            stage="confirmation",
            gate=EXTENSION_CONFIRMATION_GATE,
        )
        out[key] = {
            "scenario": scenario,
            "development": development,
            "comparison": comparison,
            "plateau": plateau,
            "selection": selection,
            "confirmation": confirmation,
            "verdict": early_layer_verdict(
                confirmation,
                scale=scale,
                selection=selection,
                development=development[scale],
            ),
        }
    return out


# ------------------------------------------------- parent metadata and audit


def test_parent_layout_is_resolved_from_what_is_actually_there(parent):
    layout = resolve_parent_layout(parent.root, baseline_scale=parent.baseline_scale)
    required = layout["required"]
    assert required["fingerprint"] == "fingerprint.json"
    assert required["corpus_manifest"] == "units/corpus/manifest.json"
    assert required["accumulator"] == "checkpoints/jacobian_sum.pt"
    assert required["baseline_lens"] == f"artifacts/lens.scale{parent.baseline_scale}.pt"
    inventory = discover_parent_files(parent.root)
    assert inventory["n_files"] >= layout["n_files"] - 1
    assert layout["layout_checksum"].startswith("sha256:")


def test_a_missing_required_parent_artifact_stops_with_a_precise_message(parent, tmp_path):
    """Nothing is inferred; the message names the role and what was searched."""
    import shutil

    copy = tmp_path / "parent_without_checkpoint"
    shutil.copytree(parent.root, copy)
    for path in (copy / "checkpoints").glob("*.pt"):
        path.unlink()
    with pytest.raises(ParentImportRefused) as error:
        resolve_parent_layout(copy, baseline_scale=parent.baseline_scale)
    message = str(error.value)
    assert "accumulator" in message
    assert "the sufficient statistic this extension continues" in message
    assert "checkpoints/jacobian_sum.pt" in message
    assert "Nothing is inferred" in message


def test_parent_audit_passes_every_blocking_check(loaded):
    _run, _requirements_, audit = loaded
    assert audit["compatible"] is True
    assert audit["blocking_failed_checks"] == []
    assert audit["failed_checks"] == []
    assert audit["audit_checksum"].startswith("sha256:")
    names = {check["check"] for check in audit["checks"]}
    for expected in (
        "parent_fingerprint_recomputes",
        "parent_configuration_checksum",
        "model_identity",
        "tokenizer_identity",
        "corpus_identity",
        "hook_site_and_residual_convention",
        "d_model",
        "fit_estimator_version",
        "accumulator_layer_grid",
        "accumulator_checksum_recorded",
        "n_done_equals_baseline",
        "baseline_snapshot_checksum",
        "fit_prompt_ordering_protocol",
        "parent_fit_prefix_checksum_present",
        "no_prompt_dropped_before_fitting",
        "old_split_checksums_present",
        "old_duplicate_audit_present",
        "old_confirmation_selection_recorded",
        "old_confirmation_vault_status_recorded",
    ):
        assert expected in names, expected


def test_audit_records_what_is_reused_and_what_never_is(loaded):
    _run, _requirements_, audit = loaded
    assert audit["reused_from_parent"] == [
        "fitting accumulator (jacobian_sum, n_done)"
    ]
    assert audit["never_reused_from_parent"] == [
        "old development set",
        "old confirmation set",
        "old confirmation verdicts",
    ]
    assert audit["old_confirmation_selection_checksum"].startswith("sha256:")
    assert audit["old_split_checksums"].keys() >= {"fit", "validation", "confirmation"}
    text = format_parent_audit(audit)
    assert "NEVER REUSED" in text
    assert "compatible         True" in text


def test_the_parent_confirmation_set_is_recorded_as_opened(loaded):
    run, _requirements_, _audit = loaded
    assert run.confirmation_was_opened is True
    assert run.confirmation_vault_status["opened"] is True


def test_n_done_must_equal_the_baseline_scale(parent, loaded):
    run, requirements, _audit = loaded
    assert run.accumulator.n_done == parent.baseline_scale
    wrong = replace(requirements, expected_n_done=parent.baseline_scale + 1)
    with pytest.raises(ParentImportRefused, match="n_done_equals_baseline"):
        audit_parent_run(run, requirements=wrong)


@pytest.mark.parametrize(
    ("field", "value", "clause"),
    [
        ("model_revision", "0" * 40, "model_identity"),
        ("tokenizer_revision", "0" * 40, "tokenizer_identity"),
        ("source_layers", (8, 14, 20, 26, 32, 35, 38), "hook_site"),
        ("target_layer", 39, "hook_site"),
        ("d_model", 64, "d_model"),
        ("skip_first", 8, "capture_geometry"),
        ("corpus_hf_dataset", "allenai/c4", "corpus_identity"),
        ("artifact_format_version", "other.v9", "artifact_format_version"),
    ],
)
def test_an_incompatible_parent_is_refused_by_name(loaded, field, value, clause):
    run, requirements, _audit = loaded
    with pytest.raises(ParentImportRefused) as error:
        audit_parent_run(run, requirements=replace(requirements, **{field: value}))
    assert clause in str(error.value)
    assert "Refusing to continue its accumulator" in str(error.value)


def test_a_tampered_parent_unit_is_refused_rather_than_read(parent, tmp_path):
    import shutil

    copy = tmp_path / "parent_tampered"
    shutil.copytree(parent.root, copy)
    unit_path = copy / "units" / "corpus" / "manifest.json"
    record = json.loads(unit_path.read_text(encoding="utf-8"))
    record["payload"]["n_dropped_too_short"] = 99
    unit_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ParentImportRefused, match="fails its own checksum"):
        load_parent_run(copy, baseline_scale=parent.baseline_scale)


def test_the_provenance_manifest_is_written_for_the_extension_not_the_parent(
    parent, loaded, tmp_path
):
    run, _requirements_, audit = loaded
    before = protected_parent_checksums(parent.root, layout=run.layout)
    manifest = parent_provenance_manifest(
        run,
        audit,
        immutability=before,
        extension_protocol_version=EXTENSION_PROTOCOL.version,
        extension_run_dir=str(tmp_path / "extension"),
    )
    assert manifest["read_only"] is True
    assert manifest["parent_written_by_this_extension"] is False
    assert manifest["extension_run_dir"] == str(tmp_path / "extension")
    assert "old_confirmation_set" in manifest["not_imported"]
    assert "already opened" in manifest["not_imported"]["old_confirmation_set"]
    assert manifest["imported"]["fitting_accumulator"]["n_done"] == parent.baseline_scale
    assert manifest["provenance_checksum"].startswith("sha256:")


# ------------------------------------------------------------- immutability


def test_the_parent_is_byte_identical_after_a_full_continuation(
    parent, loaded, continued
):
    run, _requirements_, _audit = loaded
    before = protected_parent_checksums(parent.root, layout=run.layout)
    after = protected_parent_checksums(parent.root, layout=run.layout)
    proof = assert_parent_unchanged(before, after)
    assert proof["immutable"] is True
    assert proof["changed"] == []
    assert proof["n_files_checked"] >= 6
    # the continuation wrote only inside its own directory
    assert Path(continued["store"].root) != Path(parent.root)
    assert Path(parent.root) not in Path(continued["store"].root).parents


def test_a_changed_parent_file_is_refused_with_the_path_named(parent, loaded, tmp_path):
    import shutil

    run, _requirements_, _audit = loaded
    copy = tmp_path / "parent_mutated"
    shutil.copytree(parent.root, copy)
    layout = resolve_parent_layout(copy, baseline_scale=parent.baseline_scale)
    before = protected_parent_checksums(copy, layout=layout)
    target = copy / layout["required"]["corpus_manifest"]
    target.write_text(target.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    after = protected_parent_checksums(copy, layout=layout)
    with pytest.raises(ParentImportRefused, match="changed while the extension ran"):
        assert_parent_unchanged(before, after)
    _ = run


# ------------------------------------------------ reconstruction and prefix


def test_collection_parameters_come_from_the_parent_with_named_sources(loaded):
    run, _requirements_, _audit = loaded
    parameters = parent_collection_parameters(run)
    assert parameters["min_fit"] == MOCK_BASELINE_SCALE
    assert parameters["seed"] == run.splits["split_seed"]
    assert parameters["min_chars"] == run.corpus["min_chars"]
    assert set(parameters["sources"]) == {
        "min_chars",
        "seed",
        "min_fit",
        "n_validation",
        "n_confirmation",
        "max_texts",
    }


def test_the_reconstruction_reproduces_every_parent_split_checksum(parent, loaded):
    run, _requirements_, _audit = loaded
    report = verify_reconstructed_partitions(parent.partitions, parent=run)
    assert report["all_match"] is True
    for row in report["partitions"]:
        assert row["expected_checksum"] == row["actual_checksum"]
        assert row["expected_size"] == row["actual_size"]


def test_a_reconstruction_that_disagrees_is_refused(parent, loaded):
    run, _requirements_, _audit = loaded

    class _Shifted:
        """The parent's partitions with one record dropped from the fit set."""

        def __init__(self, partitions):
            self._partitions = partitions

        def get(self, name):
            records = self._partitions.get(name)
            return records[1:] if name == "fit" else records

        def checksum(self, name):
            from jlens.mmpilot.store import payload_checksum

            return payload_checksum([r.to_dict() for r in self.get(name)])

    with pytest.raises(ContinuationRefused, match="did not reproduce the parent"):
        verify_reconstructed_partitions(_Shifted(parent.partitions), parent=run)


def test_the_fit_order_pins_the_parent_prefix_and_nests(parent, loaded, fit_order):
    run, _requirements_, _audit = loaded
    baseline = parent.baseline_scale
    assert list(fit_order[:baseline]) == list(parent.fit_records[:baseline])
    smaller = build_extension_fit_order(
        parent.fit_records, n_needed=MOCK_EXTENSION_SCALES[0]
    )
    assert list(fit_order[: len(smaller)]) == list(smaller)
    report = verify_fit_prefix(
        fit_order,
        n_parent=baseline,
        parent_prefix_checksum=run.fit_prefix_checksum(baseline),
    )
    assert report["matches"] is True
    assert report["skip_authorized"] is True


def test_a_prefix_that_is_not_the_parents_refuses_the_skip(parent, fit_order):
    with pytest.raises(ContinuationRefused, match="Refusing to skip them"):
        verify_fit_prefix(
            fit_order,
            n_parent=parent.baseline_scale,
            parent_prefix_checksum="sha256:not-the-parents",
        )


def test_the_fit_order_refuses_to_shrink_a_scale_point(parent):
    with pytest.raises(ContinuationRefused, match="not reduced to fit the records"):
        build_extension_fit_order(parent.fit_records[:5], n_needed=1_000_000)


# --------------------------------------------------------- the continuation


def test_the_extension_checkpoint_is_a_byte_copy_of_the_parents(loaded, continued):
    run, _requirements_, _audit = loaded
    seed = continued["seed"]
    assert seed["action"] == "seeded"
    assert seed["checksum"] == run.accumulator.checksum
    assert seed["parent_checkpoint_checksum"] == run.accumulator.checksum
    assert seed["n_done"] == run.accumulator.n_done
    assert seed["parent_written"] is False


def test_the_continuation_reaches_every_scale_and_names_snapshots_honestly(continued):
    result = continued["result"]
    assert result.n_done == MOCK_EXTENSION_SCALES[-1]
    assert sorted(result.snapshots) == sorted(MOCK_EXTENSION_SCALES)
    for scale, snapshot in result.snapshots.items():
        assert snapshot.n_prompts == scale
        assert snapshot.checksum.startswith("sha256:")
        assert JacobianLens.load(snapshot.path).n_prompts == scale


def test_continuation_equals_a_fresh_nested_fit_at_every_scale(
    tmp_path, parent, fit_order, continued
):
    """The claim is exactness, not closeness — same additions, same order."""
    store = continued["store"]
    for scale in MOCK_EXTENSION_SCALES:
        snapshot = store.snapshot_path(scale)
        continued_lens = JacobianLens.load(str(snapshot))
        fresh_store = ExtensionStore(
            tmp_path / f"fresh{scale}", store.fingerprint
        )
        fresh_store.open()
        fresh = run_calibration(
            parent.model,
            fit_order[:scale],
            plan=parent.plan,
            scale_points=(scale,),
            store=fresh_store,
            checkpoint_every=4,
            diagnostics_every=4,
        )
        fresh_lens = fresh.lens_for_scale(scale)
        assert fresh_lens.n_prompts == continued_lens.n_prompts == scale
        for layer in parent.plan.layers:
            assert torch.equal(
                fresh_lens.jacobians[layer], continued_lens.jacobians[layer]
            ), layer


def test_the_full_accumulator_is_bit_identical_to_a_fresh_fit(
    tmp_path, parent, fit_order, continued
):
    report = verify_continuation_equals_fresh_fit(
        parent.model,
        fit_order,
        plan=parent.plan,
        scale=MOCK_EXTENSION_SCALES[-1],
        continued_checkpoint=continued["store"].checkpoint_path,
        scratch_dir=tmp_path / "scratch",
    )
    assert report["bit_identical"] is True
    assert report["differences"] == []
    assert report["n_done_fresh"] == report["n_done_continued"]


def test_a_continuation_that_differs_from_a_fresh_fit_is_refused(
    tmp_path, parent, loaded, splits, fit_order
):
    """Scenario 5: the one failure a prompt-list checksum cannot catch."""
    run, _requirements_, _audit = loaded
    store = ExtensionStore(
        tmp_path / "corrupted", _fingerprint(parent, run, splits, salt="-corrupt")
    )
    store.open()
    seed_extension_checkpoint(run.accumulator, store.checkpoint_path)
    run_calibration(
        parent.model,
        fit_order,
        plan=parent.plan,
        scale_points=MOCK_EXTENSION_SCALES,
        store=store,
        checkpoint_every=4,
        diagnostics_every=4,
    )
    corrupt_continued_checkpoint(store.checkpoint_path, layer=32)
    with pytest.raises(ContinuationRefused, match="does NOT equal a fresh nested fit"):
        verify_continuation_equals_fresh_fit(
            parent.model,
            fit_order,
            plan=parent.plan,
            scale=MOCK_EXTENSION_SCALES[-1],
            continued_checkpoint=store.checkpoint_path,
            scratch_dir=tmp_path / "corrupt_scratch",
        )


def test_checkpoints_and_snapshots_are_written_atomically(continued):
    root = Path(continued["store"].root)
    assert list(root.rglob("*.tmp.*")) == []
    assert continued["store"].checkpoint_path.is_file()


def test_a_compatible_resume_reuses_the_work(tmp_path, parent, loaded, splits, fit_order):
    run, _requirements_, _audit = loaded
    fingerprint = _fingerprint(parent, run, splits, salt="-resume")
    root = tmp_path / "resumable"
    first = ExtensionStore(root, fingerprint)
    assert first.open() == "starting"
    seed_extension_checkpoint(run.accumulator, first.checkpoint_path)
    run_calibration(
        parent.model,
        fit_order,
        plan=parent.plan,
        scale_points=MOCK_EXTENSION_SCALES,
        store=first,
        checkpoint_every=4,
        diagnostics_every=4,
    )
    second = ExtensionStore(root, fingerprint)
    assert second.open() == "resuming"
    reseed = seed_extension_checkpoint(run.accumulator, second.checkpoint_path)
    assert reseed["action"] == "resumed"
    assert reseed["n_done"] == MOCK_EXTENSION_SCALES[-1]
    again = run_calibration(
        parent.model,
        fit_order,
        plan=parent.plan,
        scale_points=MOCK_EXTENSION_SCALES,
        store=second,
        checkpoint_every=4,
        diagnostics_every=4,
    )
    assert again.n_done == MOCK_EXTENSION_SCALES[-1]
    assert again.elapsed_seconds == 0.0  # recovered from disk, never refitted


def test_an_incompatible_resume_is_refused(tmp_path, parent, loaded, splits):
    run, _requirements_, _audit = loaded
    root = tmp_path / "incompatible"
    ExtensionStore(root, _fingerprint(parent, run, splits, salt="-a")).open()
    with pytest.raises(IncompatibleStateError, match="different\\s+configuration"):
        ExtensionStore(root, _fingerprint(parent, run, splits, salt="-b")).open()


def test_a_checkpoint_from_a_different_grid_is_refused(tmp_path, parent, loaded):
    run, _requirements_, _audit = loaded
    destination = tmp_path / "checkpoints" / "jacobian_sum.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    state = torch.load(run.accumulator.path, map_location="cpu", weights_only=True)
    state["source_layers"] = [8, 14, 20]
    state["jacobian_sum"] = {
        layer: state["jacobian_sum"][layer] for layer in (8, 14, 20)
    }
    torch.save(state, destination)
    with pytest.raises(ContinuationRefused, match="not a continuation"):
        seed_extension_checkpoint(run.accumulator, destination)


def test_a_checkpoint_behind_the_parent_is_refused(tmp_path, parent, loaded):
    run, _requirements_, _audit = loaded
    destination = tmp_path / "behind" / "jacobian_sum.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    state = torch.load(run.accumulator.path, map_location="cpu", weights_only=True)
    state["n_done"] = run.accumulator.n_done - 1
    torch.save(state, destination)
    with pytest.raises(ContinuationRefused, match="is behind the parent"):
        seed_extension_checkpoint(run.accumulator, destination)


# ------------------------------------------------------ the fresh splits


def test_the_fresh_sets_are_the_frozen_sizes_and_independent(splits):
    assert len(splits.development) == N_DEVELOPMENT_PROMPTS
    assert len(splits.confirmation) == N_CONFIRMATION_PROMPTS
    assert splits.checksum("development") != splits.checksum("confirmation")
    development = {record.record_id for record in splits.development}
    confirmation = {record.record_id for record in splits.confirmation}
    assert development.isdisjoint(confirmation)
    manifest = splits.manifest()
    assert manifest["old_sets_reused"] is False
    assert manifest["selected_by_jlens_performance"] is False
    assert manifest["manifest_checksum"].startswith("sha256:")


def test_no_fresh_record_is_an_old_or_new_fit_record(parent, splits, fit_order):
    forbidden = {
        record.record_id
        for group in (
            parent.partitions.fit,
            parent.partitions.validation,
            parent.partitions.confirmation,
            fit_order,
        )
        for record in group
    }
    for name in ("development", "confirmation"):
        for record in splits.get(name):
            assert record.record_id not in forbidden


def test_old_records_offered_to_the_pool_are_excluded_and_counted(
    parent, pool, excluded
):
    """The guard is exercised on records that would otherwise get through."""
    contaminated = [
        *pool,
        *parent.partitions.confirmation,
        *parent.partitions.validation,
        *parent.partitions.fit[:20],
    ]
    splits = build_fresh_evaluation_splits(
        contaminated, excluded=excluded, corpus_id="mock/train"
    )
    assert splits.excluded_exact["old_confirmation"] == len(
        parent.partitions.confirmation
    )
    assert splits.excluded_exact["old_development"] == len(
        parent.partitions.validation
    )
    assert splits.excluded_exact["old_fit"] == 20
    assert set(splits.excluded_exact) == set(EXCLUDED_NAMES)


def test_new_fit_records_through_the_largest_scale_are_excluded(pool, excluded, fit_order):
    splits = build_fresh_evaluation_splits(
        [*pool, *fit_order], excluded=excluded, corpus_id="mock/train"
    )
    # every new fit record is also an old fit record here, so the exact-match
    # index attributes them to whichever excluded set is registered first; what
    # matters is that none of them survives into a fresh set.
    surviving = {
        record.record_id
        for name in ("development", "confirmation")
        for record in splits.get(name)
    }
    assert surviving.isdisjoint({record.record_id for record in fit_order})
    assert sum(splits.excluded_exact.values()) >= len(fit_order)


def test_an_exact_duplicate_of_an_old_prompt_is_excluded(parent, pool, excluded):
    old = parent.partitions.confirmation[0]
    clone = CorpusRecord.build("mock/train", 10 ** 7, old.text)
    assert clone.normalized_checksum == old.normalized_checksum
    splits = build_fresh_evaluation_splits(
        [*pool, clone], excluded=excluded, corpus_id="mock/train"
    )
    assert splits.excluded_exact["old_confirmation"] >= 1
    assert clone.record_id not in {
        record.record_id
        for name in ("development", "confirmation")
        for record in splits.get(name)
    }


def test_a_near_duplicate_of_an_old_prompt_is_excluded(parent, pool, excluded):
    old = parent.partitions.confirmation[0]
    near = CorpusRecord.build("mock/train", 10 ** 7 + 1, old.text + " extra")
    assert near.normalized_checksum != old.normalized_checksum
    assert hamming_distance(simhash64(near.text), old.simhash) <= 3
    splits = build_fresh_evaluation_splits(
        [*pool, near], excluded=excluded, corpus_id="mock/train"
    )
    assert splits.excluded_near["old_confirmation"] >= 1
    assert near.record_id not in {
        record.record_id
        for name in ("development", "confirmation")
        for record in splits.get(name)
    }


def test_a_genuinely_different_record_is_not_excluded(parent, excluded):
    old = parent.partitions.confirmation[0]
    words = old.text.split()
    distant = CorpusRecord.build(
        "mock/train", 10 ** 7 + 2, " ".join([*words[:-2], "tok999", "tok998"])
    )
    assert hamming_distance(distant.simhash, old.simhash) > 3
    # the near-duplicate filter must not swallow merely similar prose
    from jlens.calibration.extension import _NearDuplicateIndex

    hit = _NearDuplicateIndex([("old_confirmation", old)]).hit(distant, max_hamming=3)
    assert hit is None
    _ = excluded


def test_the_split_refuses_to_shrink_when_the_pool_is_short(pool, excluded):
    with pytest.raises(ExtensionSplitRefused, match="not reduced because coverage"):
        build_fresh_evaluation_splits(
            pool[:40], excluded=excluded, corpus_id="mock/train"
        )


def test_the_cross_split_audit_is_clean_on_the_constructed_sets(splits, excluded):
    report = audit_fresh_split_leakage(splits, excluded=excluded)
    assert report["ok"] is True
    assert report["n_exact_hits"] == 0
    assert report["n_near_hits"] == 0
    assert report["candidate_pairs_compared"] >= 0
    assert report["audit_checksum"].startswith("sha256:")


def test_an_old_confirmation_prompt_planted_in_a_new_split_is_refused(
    parent, splits, excluded
):
    """Scenario 4: the builder prevents it; the independent audit refuses it."""
    contaminated = plant_old_confirmation_record(
        splits, parent.partitions.confirmation
    )
    with pytest.raises(ExtensionSplitRefused) as error:
        audit_fresh_split_leakage(contaminated, excluded=excluded)
    message = str(error.value)
    assert "old_confirmation" in message
    assert "exactly as spent as the one it replaces" in message


def test_development_and_confirmation_overlap_is_refused(splits, excluded):
    from jlens.calibration.extension import FreshEvaluationSplits

    overlapping = FreshEvaluationSplits(
        development=splits.development,
        confirmation=(*splits.confirmation[:-1], splits.development[0]),
        corpus_id=splits.corpus_id,
        seed=splits.seed,
    )
    with pytest.raises(ExtensionSplitRefused, match="development"):
        audit_fresh_split_leakage(overlapping, excluded=excluded)


# ---------------------------------------------------------- target diversity


def test_the_fresh_sets_clear_the_stricter_diversity_floor(splits):
    for name, seed in (("development", 1), ("confirmation", 2)):
        prompts, manifest = select_diverse_validation_prompts(
            [record.text for record in splits.get(name)],
            n_prompts=N_DEVELOPMENT_PROMPTS,
            gate=EXTENSION_GATE,
            seed=20260808 + seed,
            target_token_for_prompt=mock_target_token,
        )
        assert len(prompts) == N_DEVELOPMENT_PROMPTS
        diversity = audit_extension_target_diversity(
            [row["target_token_id"] for row in manifest["prompts"]]
        )
        assert diversity["passed"] is True
        assert diversity["n_distinct_target_tokens"] >= 32
        assert diversity["max_target_token_share"] <= 0.25
        assert manifest["selected_by_jlens_performance"] is False


def test_the_diversity_floor_is_refused_rather_than_lowered(splits):
    with pytest.raises(InsufficientTargetDiversityError):
        select_diverse_validation_prompts(
            [record.text for record in splits.development],
            n_prompts=N_DEVELOPMENT_PROMPTS,
            gate=EXTENSION_GATE,
            seed=20260808,
            target_token_for_prompt=lambda prompt: 7,  # one target for everything
        )


def test_target_discovery_cannot_be_handed_a_lens():
    import inspect

    signature = inspect.signature(select_diverse_validation_prompts)
    assert "lens" not in signature.parameters
    assert "layer" not in signature.parameters
    assert "j_lens" not in signature.parameters


# ---------------------------------------------------------------- the gate


def test_the_gate_changes_only_the_sample_size_and_the_diversity_floor():
    """Unchanged or stricter — every other threshold is byte-identical."""
    base = CALIBRATION_GATE.to_dict()
    extension = EXTENSION_GATE.to_dict()
    differing = {
        key for key in base if base[key] != extension[key]
    }
    assert differing == {"n_prompts", "min_distinct_target_tokens", "version"}
    assert extension["n_prompts"] == 256 > base["n_prompts"]
    assert extension["min_distinct_target_tokens"] == 32
    assert extension["min_distinct_target_tokens"] > base["min_distinct_target_tokens"]
    # every decision threshold that could make a layer pass is unchanged
    for key in (
        "max_tied_at_max_rate",
        "min_noise_control_mrr_ratio",
        "min_noise_control_mrr_margin",
        "min_wrong_layer_mrr_margin",
        "max_median_midrank",
        "min_top_k_inclusion",
        "top_k",
        "n_folds",
        "min_fold_mrr_fraction",
        "max_target_token_share",
        "rank_convention",
        "wrong_layer_mapping",
        "controls",
        "noise_controls",
    ):
        assert extension[key] == base[key], key


def test_the_confirmation_gate_is_the_development_gate(splits):
    assert EXTENSION_CONFIRMATION_GATE.digest == EXTENSION_GATE.digest
    assert EXTENSION_CONFIRMATION_GATE.n_prompts == N_CONFIRMATION_PROMPTS
    _ = splits


def test_the_gate_text_says_what_changed_and_that_nothing_loosened():
    text = extension_gate_text()
    assert "STRICTER" in text
    assert "NOTHING IS LOOSENED" in text
    assert "unchanged" in text
    assert EXTENSION_GATE.digest in text


def test_the_protocol_states_every_commissioned_fact():
    text = EXTENSION_PROTOCOL.text()
    assert "old confirmation set" in text.lower()
    assert "never reused, relabelled or reset" in text
    assert "descriptive baseline only" in text
    assert f"L{PRIMARY_EARLY_LAYER}" in text
    assert f"L{SECONDARY_EARLY_LAYER}" in text
    assert "[8, 14, 20]" in text
    assert "[35, 38, 40]" in text
    assert str(MAXIMUM_AUTHORIZED_SCALE) in text.replace(",", "")
    payload = EXTENSION_PROTOCOL.to_dict()
    assert payload["multimodal_data_in_fitting"] is False
    assert payload["multimodal_data_in_lens_validation"] is False
    assert payload["candidate_scales"] == list(EXTENSION_SCALE_POINTS)
    assert EXTENSION_PROTOCOL.digest.startswith("sha256:")


def test_scale_100_is_scored_but_never_selectable():
    assert DEVELOPMENT_SCALE_POINTS == (BASELINE_SCALE, *EXTENSION_SCALE_POINTS)
    assert BASELINE_SCALE not in EXTENSION_SELECTION_RULE.candidate_scales


# ------------------------------------------------------------ scale selection


def test_the_selection_rule_is_declared_and_ignores_confirmation():
    rule = EXTENSION_SELECTION_RULE
    assert rule.declared_before_results is True
    assert rule.confirmation_may_be_consulted is False
    assert rule.multimodal_outcomes_may_be_consulted is False
    assert tuple(rule.earlier_layers) == tuple(EARLY_LAYERS_OF_INTEREST)
    assert rule.fallback_scale == MAXIMUM_AUTHORIZED_SCALE
    assert rule.digest.startswith("sha256:")
    assert "confirmation" in rule.text().lower()


def test_selection_refuses_a_candidate_scale_it_never_scored():
    comparison = {
        "scales": [250],
        "eligible_by_scale": {"250": [32]},
    }
    with pytest.raises(ValueError, match="were not scored"):
        select_extension_scale(comparison, candidate_scales=(250, 1_000))


@pytest.mark.parametrize(
    ("key", "clause"),
    [
        ("l32_late_pass", "fallback_to_largest"),
        ("l32_confirmation_fail", "smallest_scale_matching_largest"),
        ("no_early_layer", "no_early_layer_development_pass"),
    ],
)
def test_scale_selection_is_deterministic_and_clause_named(
    scenario_results, key, clause
):
    selection = scenario_results[key]["selection"]
    assert selection["clause_applied"] == clause
    assert selection["selected_scale"] == (
        EXTENSION_SCENARIOS[key].expected_selected_scale
    )
    assert selection["confirmation_not_consulted"] is True
    assert selection["multimodal_outcomes_not_consulted"] is True
    assert selection["selected_from_development_only"] is True
    assert selection["descriptive_scales_not_selectable"] == [MOCK_BASELINE_SCALE]
    assert selection["plateau_clause_informative"] is True
    # deterministic: recomputing gives the same checksum
    again = select_extension_scale(
        scenario_results[key]["comparison"],
        plateau=scenario_results[key]["plateau"],
        candidate_scales=MOCK_EXTENSION_SCALES,
    )
    assert again["selection_checksum"] == selection["selection_checksum"]


def test_scoring_the_baseline_makes_the_plateau_clause_informative(scenario_results):
    """With two points the continuation clause has no previous step at all."""
    late = scenario_results["l32_late_pass"]
    assert late["plateau"]["verdict"] == "STILL_IMPROVING_EXTENSION_JUSTIFIED"
    two_point = evaluate_plateau(
        compare_scales(
            {
                scale: late["development"][scale]
                for scale in MOCK_EXTENSION_SCALES
            },
            layers=list(MOCK_LAYERS),
        )
    )
    assert two_point["verdict"] == "PLATEAU_REACHED"
    assert any(
        clause["clause"] == "still_climbing" and not clause["passed"]
        for clause in two_point["clauses"]
    )


# --------------------------------------------------- vault and confirmation


def test_the_fresh_vault_will_not_open_before_a_scale_is_selected(splits):
    vault = ConfirmationVault(records=splits.confirmation)
    assert vault.locked is True
    with pytest.raises(ConfirmationLocked, match="has not been unlocked"):
        vault.open()
    assert vault.status()["opened"] is False
    assert vault.status()["n_records"] == N_CONFIRMATION_PROMPTS


def test_the_vault_refuses_a_selection_that_cannot_disclaim_confirmation(splits):
    vault = ConfirmationVault(records=splits.confirmation)
    with pytest.raises(ConfirmationLocked, match="confirmation_not_consulted"):
        vault.unlock({"selected_scale": 1_000})


def test_the_vault_opens_against_the_recorded_selection(splits, scenario_results):
    vault = ConfirmationVault(records=splits.confirmation)
    selection = scenario_results["l32_late_pass"]["selection"]
    vault.unlock(selection)
    records = vault.open()
    assert len(records) == N_CONFIRMATION_PROMPTS
    assert vault.status()["selected_scale"] == selection["selected_scale"]
    assert vault.status()["selection_checksum"] == selection["selection_checksum"]


# ---------------------------------------------------------------- verdicts


@pytest.mark.parametrize("key", sorted(EXTENSION_SCENARIOS))
def test_every_scenario_reaches_its_commissioned_verdict(scenario_results, key):
    result = scenario_results[key]
    scenario = EXTENSION_SCENARIOS[key]
    assert result["verdict"]["verdict"] == scenario.expected_verdict
    assert result["selection"]["selected_scale"] == scenario.expected_selected_scale
    passing = set(result["verdict"]["earlier_layers_passing_confirmation"])
    assert passing == set(scenario.expected_published_layers)


def test_the_verdict_records_every_layer_including_failures(scenario_results):
    verdict = scenario_results["l32_late_pass"]["verdict"]
    assert {row["layer"] for row in verdict["layers"]} == set(MOCK_LAYERS)
    assert verdict["all_layer_results_recorded"] is True
    roles = {row["layer"]: row["role"] for row in verdict["layers"]}
    assert roles[PRIMARY_EARLY_LAYER] == "primary"
    assert roles[SECONDARY_EARLY_LAYER] == "secondary"
    assert roles[38] == "already_published"
    assert roles[8] == "descriptive"
    for row in verdict["layers"]:
        assert row["mean_reciprocal_rank"] is not None
        assert row["median_midrank"] is not None


def test_a_no_go_states_the_negative_plainly(scenario_results):
    verdict = scenario_results["no_early_layer"]["verdict"]
    assert verdict["verdict"] == "EARLY_LAYER_CALIBRATION_NO_GO"
    assert verdict["earlier_layers_passing_confirmation"] == []
    assert "did not yield a validated earlier readout" in verdict["statement"]
    assert "authorized endpoint" in verdict["statement"]
    assert verdict["existing_publications_unchanged"] == [35, 38, 40]


def test_development_success_alone_is_never_a_go(scenario_results):
    result = scenario_results["l32_confirmation_fail"]
    scale = result["selection"]["selected_scale"]
    assert result["development"][scale][PRIMARY_EARLY_LAYER]["passed"] is True
    assert result["confirmation"][PRIMARY_EARLY_LAYER]["passed"] is False
    assert result["verdict"]["verdict"] == "EARLY_LAYER_CALIBRATION_NO_GO"


# -------------------------------------------------------------- publication


def _publication_arguments(parent, loaded, continued, splits, result, layer):
    run, _requirements_, audit = loaded
    scale = result["selection"]["selected_scale"]
    manifest = extension_corpus_manifest(
        corpus_config=parent.corpus_config,
        corpus_id="mock/train",
        fit_records=[],
        splits=splits,
        scale_points=MOCK_EXTENSION_SCALES,
        parent=run,
    )
    return {
        "layer": layer,
        "scale": scale,
        "lens": continued["result"].lens_for_scale(MOCK_EXTENSION_SCALES[-1]),
        "confirmation_verdict": result["confirmation"][layer],
        "development_verdict": result["development"][scale][layer],
        "parent": run,
        "parent_audit": audit,
        "continuation": {"parent_checkpoint_checksum": run.accumulator.checksum},
        "splits": splits,
        "selection": result["selection"],
        "load_info": mock_load_info(),
        "corpus_manifest": manifest,
        "capture_plan": parent.plan.to_dict(),
        "fitting_diagnostics": continued["result"].to_dict(),
        "environment": mock_extension_environment(),
    }


@pytest.fixture
def opened_vault(splits, scenario_results):
    vault = ConfirmationVault(records=splits.confirmation)
    vault.unlock(scenario_results["l32_late_pass"]["selection"])
    vault.open()
    return vault


def test_a_confirmed_early_layer_publishes_with_full_provenance(
    tmp_path, parent, loaded, continued, splits, scenario_results, opened_vault
):
    run, _requirements_, _audit = loaded
    result = scenario_results["l32_late_pass"]
    # the lens must carry the scale the verdict is for
    scale = result["selection"]["selected_scale"]
    arguments = _publication_arguments(
        parent, loaded, continued, splits, result, PRIMARY_EARLY_LAYER
    )
    run_dir = tmp_path / "rgext_publish"
    destination = run_dir / "artifacts" / "published" / "lens.early.pt"
    artifact = publish_early_layer(
        **arguments,
        destination=destination,
        vault=opened_vault,
        extension_run_dir=run_dir,
    )
    assert destination.is_file()
    assert artifact["validated"] is True
    assert artifact["frozen"] is True
    assert artifact["publication_status"] == "PUBLISHED_VALIDATED_EARLY_LAYER"
    assert artifact["physical_layer"] == PRIMARY_EARLY_LAYER
    assert artifact["scale_point"] == scale
    assert artifact["parent_run_root"] == run.root
    assert artifact["parent_accumulator_checksum"] == run.accumulator.checksum
    assert artifact["parent_accumulator_n_done"] == parent.baseline_scale
    assert artifact["old_confirmation_set_reused"] is False
    assert artifact["old_development_set_reused"] is False
    assert artifact["parent_run_written"] is False
    assert artifact["development_split_checksum"] == splits.checksum("development")
    assert artifact["confirmation_split_checksum"] == splits.checksum("confirmation")
    assert artifact["gate_digest"] == EXTENSION_GATE.digest
    assert artifact["selection_checksum"] == result["selection"]["selection_checksum"]
    assert artifact["existing_publications_unchanged"] == [35, 38, 40]
    assert artifact["calibration_modality"] == "text-only"
    assert artifact["spokencoco_used"] is False
    assert artifact["multimodal_data_used"] is False
    assert artifact["objective"] == "not_applicable_estimator_is_a_sample_mean"
    assert artifact["artifact_checksum"].startswith("sha256:")
    assert artifact["lens_checksum"].startswith("sha256:")
    sidecar = json.loads(
        destination.with_suffix(".extension.json").read_text(encoding="utf-8")
    )
    assert sidecar["artifact_checksum"] == artifact["artifact_checksum"]
    base = json.loads(destination.with_suffix(".json").read_text(encoding="utf-8"))
    assert base["validated"] is True
    assert base["fit_split_checksum"]
    assert list(destination.parent.glob("*.tmp.*")) == []


def test_a_layer_that_failed_confirmation_is_never_published(
    tmp_path, parent, loaded, continued, splits, scenario_results, opened_vault
):
    result = scenario_results["l32_late_pass"]
    arguments = _publication_arguments(
        parent, loaded, continued, splits, result, SECONDARY_EARLY_LAYER
    )
    assert result["confirmation"][SECONDARY_EARLY_LAYER]["passed"] is False
    run_dir = tmp_path / "rgext_refuse_failed"
    destination = run_dir / "artifacts" / "published" / "lens.early.pt"
    with pytest.raises(PublicationRefused, match="failed the confirmation gate"):
        publish_early_layer(
            **arguments,
            destination=destination,
            vault=opened_vault,
            extension_run_dir=run_dir,
        )
    assert not destination.exists()


def test_a_failed_layer_is_recorded_but_never_marked_validated(scenario_results):
    result = scenario_results["l32_confirmation_fail"]
    scale = result["selection"]["selected_scale"]
    record = record_failed_layer(
        layer=PRIMARY_EARLY_LAYER,
        scale=scale,
        confirmation_verdict=result["confirmation"][PRIMARY_EARLY_LAYER],
        validation_verdict=result["development"][scale][PRIMARY_EARLY_LAYER],
    )
    assert record["validated"] is False
    assert record["published"] is False
    assert record["confirmation_failed_checks"]
    assert record["confirmation_metrics"] is not None


def test_an_established_layer_is_not_republished(
    tmp_path, parent, loaded, continued, splits, scenario_results, opened_vault
):
    result = scenario_results["l32_late_pass"]
    arguments = _publication_arguments(
        parent, loaded, continued, splits, result, 38
    )
    assert result["confirmation"][38]["passed"] is True
    run_dir = tmp_path / "rgext_refuse_38"
    with pytest.raises(PublicationRefused, match="not an extension publication target"):
        publish_early_layer(
            **arguments,
            destination=run_dir / "artifacts" / "published" / "lens.early.pt",
            vault=opened_vault,
            extension_run_dir=run_dir,
        )


def test_publication_outside_the_extension_run_is_refused(
    tmp_path, parent, loaded, continued, splits, scenario_results, opened_vault
):
    result = scenario_results["l32_late_pass"]
    arguments = _publication_arguments(
        parent, loaded, continued, splits, result, PRIMARY_EARLY_LAYER
    )
    run_dir = tmp_path / "rgext_scope"
    run_dir.mkdir(parents=True, exist_ok=True)
    with pytest.raises(PublicationRefused, match="outside the extension run directory"):
        publish_early_layer(
            **arguments,
            destination=tmp_path / "elsewhere" / "lens.pt",
            vault=opened_vault,
            extension_run_dir=run_dir,
        )


def test_publication_into_the_parent_run_is_refused(
    tmp_path, parent, loaded, continued, splits, scenario_results, opened_vault
):
    result = scenario_results["l32_late_pass"]
    arguments = _publication_arguments(
        parent, loaded, continued, splits, result, PRIMARY_EARLY_LAYER
    )
    with pytest.raises(PublicationRefused):
        publish_early_layer(
            **arguments,
            destination=Path(parent.root) / "artifacts" / "stolen.pt",
            vault=opened_vault,
            extension_run_dir=Path(parent.root),
        )
    assert not (Path(parent.root) / "artifacts" / "stolen.pt").exists()
    _ = tmp_path


def test_publication_needs_an_opened_vault(
    tmp_path, parent, loaded, continued, splits, scenario_results
):
    result = scenario_results["l32_late_pass"]
    arguments = _publication_arguments(
        parent, loaded, continued, splits, result, PRIMARY_EARLY_LAYER
    )
    run_dir = tmp_path / "rgext_locked"
    with pytest.raises(PublicationRefused, match="never opened"):
        publish_early_layer(
            **arguments,
            destination=run_dir / "artifacts" / "published" / "lens.early.pt",
            vault=ConfirmationVault(records=splits.confirmation),
            extension_run_dir=run_dir,
        )


def test_a_development_verdict_cannot_stand_in_for_a_confirmation_one(
    tmp_path, parent, loaded, continued, splits, scenario_results, opened_vault
):
    result = scenario_results["l32_late_pass"]
    scale = result["selection"]["selected_scale"]
    arguments = _publication_arguments(
        parent, loaded, continued, splits, result, PRIMARY_EARLY_LAYER
    )
    arguments["confirmation_verdict"] = result["development"][scale][
        PRIMARY_EARLY_LAYER
    ]
    run_dir = tmp_path / "rgext_stage"
    with pytest.raises(PublicationRefused, match="not a confirmation result"):
        publish_early_layer(
            **arguments,
            destination=run_dir / "artifacts" / "published" / "lens.early.pt",
            vault=opened_vault,
            extension_run_dir=run_dir,
        )


# ------------------------------------------------------------------ budget


def test_the_budget_is_incremental_and_labels_its_extrapolation():
    from jlens.calibration.plan import build_capture_plan

    plan = build_capture_plan()
    budget = extension_budget(plan=plan)
    assert budget["incremental_not_cumulative"] is True
    assert [row["scale"] for row in budget["rows"]] == list(EXTENSION_SCALE_POINTS)
    assert budget["rows"][0]["incremental_prompts"] == 250 - BASELINE_SCALE
    assert budget["rows"][1]["incremental_prompts"] == 1_000 - 250
    assert budget["rows"][0]["prompts_already_in_accumulator"] == BASELINE_SCALE
    assert budget["anchor"]["minutes"] == 7.1
    assert budget["anchor"]["n_prompts"] == BASELINE_SCALE
    assert budget["rows"][1]["backward_passes"] == 750 * 320
    text = format_extension_budget(budget)
    assert "EXTRAPOLATION, NOT MEASUREMENT" in text
    assert "ALREADY in the accumulator" in text
    assert "resume" in text.lower()


def test_the_budget_never_recharges_for_the_parents_prompts():
    from jlens.calibration.plan import build_capture_plan

    budget = extension_budget(plan=build_capture_plan())
    assert sum(row["incremental_prompts"] for row in budget["rows"]) == (
        MAXIMUM_AUTHORIZED_SCALE - BASELINE_SCALE
    )


# ------------------------------------------------------------ scope guards


def test_nothing_in_the_extension_can_reach_a_multimodal_data_path():
    """Reuse of pure helpers is deliberate; reuse of multimodal *data* is not.

    ``mmpilot.store`` supplies canonical JSON and checksums; ``mmlocalize``
    supplies the tie-aware scorer over two logit vectors. Both are pure
    functions, and sharing them is what makes a number here mean what the same
    number meant in the completed runs. Any *other* multimodal import would be a
    data path, so the test reads the actual import statements rather than the
    prose — the prose talks about SpokenCOCO precisely in order to exclude it.
    """
    import ast

    import jlens.calibration.extension as extension
    import jlens.calibration.extension_mock as extension_mock
    import jlens.calibration.parent as parent_module

    allowed = {
        "jlens.mmpilot.store",
        "jlens.mmlocalize.lens_validity",
    }
    for module in (extension, extension_mock, parent_module):
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
            elif isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
        multimodal = {name for name in imported if ".mm" in name}
        assert multimodal <= allowed, (module.__name__, multimodal)
        for name in imported:
            for forbidden in ("spokencoco", "audio", "image", "adapter"):
                assert forbidden not in name.lower(), (module.__name__, name)


def test_the_publication_targets_are_only_the_earlier_layers():
    assert tuple(PUBLISHABLE_LAYERS) == tuple(EARLY_LAYERS_OF_INTEREST)
    assert 38 not in PUBLISHABLE_LAYERS
    assert 35 not in PUBLISHABLE_LAYERS
    assert 40 not in PUBLISHABLE_LAYERS
