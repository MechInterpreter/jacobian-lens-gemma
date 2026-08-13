"""The corrected wrong-layer control for the L32-L40 band.

The defect under repair: the completed band-interior validation built its
wrong-layer control with ``distant_layer_mapping`` over the *newly fitted
subset* ``(33, 34, 36, 37, 39)``, so the substituted Jacobian was always another
nearby late lens, while the earlier scale-250 study ran the same nominal control
over its broad grid and compared every late layer against L8. The same ``+0.15``
margin was therefore a different test in the two runs.

Every numbered requirement of the correction has a test here.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from jlens.calibration.corpus import build_records
from jlens.calibration.extension import (
    EXTENSION_CONFIRMATION_GATE,
    EXTENSION_GATE,
    ExtensionSplitRefused,
)
from jlens.calibration.gate import CALIBRATION_GATE, CONTROL_SEED
from jlens.calibration.mock import MockCalibrationModel, mock_corpus_texts
from jlens.calibration.state import CalibrationFingerprint
from jlens.controls import distant_layer_mapping, layer_mapped_lens
from jlens.lens import JacobianLens
from jlens.mmpilot.band_control import (
    BAND_SCORING_LAYERS,
    CONTROL_ONLY_LAYERS,
    CORRECTED_BAND_GO,
    CORRECTED_BAND_NO_GO,
    CORRECTED_GATE,
    CORRECTED_SCALE,
    CORRECTED_SPLIT_PROTOCOL,
    CORRECTION_PROTOCOL_VERSION,
    FIXED_CONTROL_UNIVERSE,
    SUPERSEDED_RUN_NAME,
    SUPERSEDED_VERDICT,
    SUPERSEDED_WRONG_LAYER_MAPPING,
    WRONG_LAYER_MAPPING,
    ControlUniverse,
    CorrectedControlRefused,
    CorrectedControlStore,
    _extension_capture_geometry,
    assert_no_opened_records,
    assert_protocol_persisted,
    assert_superseded_run_unchanged,
    build_control_universe,
    build_corrected_confirmation_population,
    corrected_band_verdict,
    corrected_confirmation_manifest,
    corrected_control_lenses,
    corrected_layer_rows,
    corrected_protocol_record,
    corrected_readout_budget,
    corrected_validation_report,
    evaluate_corrected_layers,
    format_corrected_verdict,
    format_wrong_layer_mapping,
    publish_corrected_layer,
    score_corrected_readout_rows,
    superseded_run_digest,
    wrong_layer_mapping_for_universe,
)
from jlens.mmpilot.band_control_mock import (
    CORRECTED_MOCK_MANIFEST_CHECKSUM,
    CORRECTED_MOCK_SCENARIOS,
    mock_control_universe,
    mock_corrected_confirmation,
    mock_lens_snapshots,
)
from jlens.mmpilot.band_lens import BAND_INTERIOR_LAYERS
from jlens.mmpilot.store import payload_checksum

ROOT = Path(__file__).resolve().parent.parent

#: What the superseded run's control actually did, from its own report.
SUPERSEDED_MAPPING_AS_RUN = {33: 39, 34: 39, 36: 33, 37: 33, 39: 33}

#: What the scale-250 study that admitted L32/L35/L38/L40 did, over its grid.
SCALE250_STUDY_GRID = (8, 14, 20, 26, 32, 35, 38, 40)


# ------------------------------------------------------------------ fixtures


@pytest.fixture(scope="module")
def universe() -> ControlUniverse:
    return mock_control_universe()


@pytest.fixture(scope="module")
def protocol(universe: ControlUniverse) -> dict:
    return corrected_protocol_record(
        universe=universe,
        model_repo_id="google/gemma-4-E4B-it",
        model_revision="mock-revision",
        exclusion_sources={"superseded_report": "sha256:mock-superseded"},
    )


@pytest.fixture(scope="module")
def passing_confirmation() -> dict:
    return dict(mock_corrected_confirmation("all_nine_pass"))


def _fingerprint(digest_seed: str = "mock") -> CalibrationFingerprint:
    return CalibrationFingerprint(
        mode="mock",
        protocol_version=CORRECTION_PROTOCOL_VERSION,
        model_repo_id="mock",
        model_revision=digest_seed,
        tokenizer_revision="mock",
        capture_plan_digest="sha256:mock-plan",
        corpus_manifest_checksum="sha256:mock-corpus",
        gate_digest=CORRECTED_GATE.digest,
        plateau_rule_digest="not_applicable_no_scale_selection",
        scale_points=(CORRECTED_SCALE,),
        artifact_format_version="jlens.calibration.artifact.v1",
    )


def _pool(n: int = 900, *, offset: int = 0) -> list:
    """A corpus pool wide enough for a 256-record confirmation population."""
    texts = mock_corpus_texts(n + offset, words=80)[offset:]
    return build_records("mock/train", texts, min_chars=100)


def _small_lens(model: MockCalibrationModel, *, seed: int = 20260901) -> JacobianLens:
    generator = torch.Generator().manual_seed(seed)
    return JacobianLens(
        jacobians={
            layer: torch.randn(model.d_model, model.d_model, generator=generator) * 0.05
            for layer in FIXED_CONTROL_UNIVERSE
        },
        n_prompts=CORRECTED_SCALE,
        d_model=model.d_model,
    )


def _superseded_run(root: Path) -> Path:
    """A stand-in for ``bandlens_real_de9338ec2a6e``, in the real layout."""
    run = root / SUPERSEDED_RUN_NAME
    (run / "artifacts").mkdir(parents=True)
    (run / "units" / "scale_snapshot").mkdir(parents=True)
    lens = JacobianLens(
        jacobians={
            layer: torch.eye(4) * (layer + 1) for layer in BAND_INTERIOR_LAYERS
        },
        n_prompts=CORRECTED_SCALE,
        d_model=4,
    )
    snapshot = run / "artifacts" / f"lens.band.scale{CORRECTED_SCALE}.pt"
    lens.save(str(snapshot))
    from jlens.metadata import file_sha256

    payload = {
        "path": str(snapshot),
        "checksum": file_sha256(str(snapshot)),
        "layers": list(BAND_INTERIOR_LAYERS),
    }
    (run / "units" / "scale_snapshot" / f"scale{CORRECTED_SCALE}.json").write_text(
        json.dumps(
            {
                "schema": "jlens.calibration.unit.v1",
                "stage": "scale_snapshot",
                "key": f"scale{CORRECTED_SCALE}",
                "unit_checksum": payload_checksum(payload),
                "payload": payload,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (run / "artifacts" / "band_interior_lens_report.json").write_text(
        json.dumps(
            {
                "schema": "jlens.mmpilot.band_interior_lens_report.v1",
                "mode": "real",
                "corpus_equivalence": {
                    "scale": CORRECTED_SCALE,
                    "model_repo_id": "google/gemma-4-E4B-it",
                    "model_revision": "mock-revision",
                    "band_capture_plan": {
                        "target_layer": 41,
                        "d_model": 4,
                        "dim_batch": 8,
                        "max_seq_len": 128,
                        "skip_first": 16,
                        "n_layers": 42,
                    },
                    "extension_corpus_id": "mock/train",
                    "extension_corpus_revision": "mock-corpus-revision",
                    "extension_fit_prefix_checksum": "sha256:mock-fit-prefix",
                    "same_estimator": "jlens.fitting.fit (upstream, unmodified)",
                },
                "band_verdict": {"verdict": SUPERSEDED_VERDICT},
                "fresh_splits": {
                    "checksums": {"development": "x", "confirmation": "y"},
                    "sizes": {"development": 256, "confirmation": 256},
                    "record_ids": {"development": [], "confirmation": []},
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return run


# ------------------------------------------- 1-3: the mapping is set-independent


def test_the_mapping_is_computed_over_the_fixed_broad_universe():
    """Requirement 1. The argument is the frozen inventory, not the fitted set."""
    assert wrong_layer_mapping_for_universe() == distant_layer_mapping(
        FIXED_CONTROL_UNIVERSE
    )
    assert FIXED_CONTROL_UNIVERSE == (8, 14, 20, 26, 32, 33, 34, 35, 36, 37, 38, 39, 40)
    assert WRONG_LAYER_MAPPING == wrong_layer_mapping_for_universe()
    # The defect, stated as the thing that must no longer happen.
    assert WRONG_LAYER_MAPPING != distant_layer_mapping(BAND_INTERIOR_LAYERS)


def test_the_explicit_mapping_is_pinned():
    """Requirement 2. Every band layer is compared against L8, as in the
    scale-250 study whose verdicts this correction has to be comparable to."""
    assert WRONG_LAYER_MAPPING == {
        8: 40, 14: 40, 20: 40, 26: 8,
        32: 8, 33: 8, 34: 8, 35: 8, 36: 8, 37: 8, 38: 8, 39: 8, 40: 8,
    }
    assert SUPERSEDED_WRONG_LAYER_MAPPING == SUPERSEDED_MAPPING_AS_RUN
    broad = distant_layer_mapping(SCALE250_STUDY_GRID)
    for layer in (32, 35, 38, 40):
        assert broad[layer] == WRONG_LAYER_MAPPING[layer] == 8


def test_changing_the_newly_fitted_layers_cannot_move_the_mapping():
    """Requirement 3."""
    baseline = wrong_layer_mapping_for_universe()
    for fitted in ((33, 34), (36, 37, 39), BAND_INTERIOR_LAYERS, (33,), ()):
        # However the "newly fitted" set is described, the control universe —
        # and therefore the mapping — is the same object.
        assert wrong_layer_mapping_for_universe() == baseline, fitted
    # And it is genuinely sensitive to the universe, so the test above is not
    # passing because the function ignores its argument.
    assert wrong_layer_mapping_for_universe(SCALE250_STUDY_GRID) != baseline


def test_scoring_targets_are_exactly_the_nine_physical_band_layers():
    """Requirement 4."""
    assert BAND_SCORING_LAYERS == (32, 33, 34, 35, 36, 37, 38, 39, 40)
    assert CONTROL_ONLY_LAYERS == (8, 14, 20, 26)
    assert set(CONTROL_ONLY_LAYERS) & set(BAND_SCORING_LAYERS) == set()
    assert set(BAND_SCORING_LAYERS) | set(CONTROL_ONLY_LAYERS) == set(
        FIXED_CONTROL_UNIVERSE
    )


def test_the_restricted_wrong_layer_control_equals_the_unrestricted_one():
    """The control is built over the whole universe and then read out on the
    nine scored layers; restricting it for memory must not change a number."""
    model = MockCalibrationModel()
    lens = _small_lens(model)
    unrestricted = layer_mapped_lens(lens, WRONG_LAYER_MAPPING)
    restricted = corrected_control_lenses(lens)["wrong_layer"]
    assert restricted.source_layers == list(BAND_SCORING_LAYERS)
    for layer in BAND_SCORING_LAYERS:
        assert torch.equal(unrestricted.jacobians[layer], restricted.jacobians[layer])


def test_a_control_cannot_be_built_from_whichever_layers_are_present():
    """The defect's mechanism, refused directly: a lens without L8 cannot
    quietly fall back to a nearer substitute."""
    model = MockCalibrationModel()
    full = _small_lens(model)
    without_l8 = JacobianLens(
        jacobians={
            layer: matrix
            for layer, matrix in full.jacobians.items()
            if layer != 8
        },
        n_prompts=full.n_prompts,
        d_model=full.d_model,
    )
    with pytest.raises(CorrectedControlRefused, match="no matrix for layer"):
        corrected_control_lenses(without_l8)


# ------------------------------------------ 5, 19: one population, nine layers


def test_all_nine_layers_must_carry_the_same_confirmation_manifest(
    passing_confirmation,
):
    """Requirement 5. Old verdicts for 32/35/38/40 may not be pasted beside new
    verdicts for the interior."""
    mixed = dict(passing_confirmation)
    mixed[35] = {
        **mixed[35],
        "confirmation_manifest_checksum": "sha256:the-earlier-runs-manifest",
    }
    with pytest.raises(CorrectedControlRefused, match="different confirmation manifest"):
        corrected_band_verdict(
            mixed, confirmation_manifest_checksum=CORRECTED_MOCK_MANIFEST_CHECKSUM
        )


def test_a_missing_layer_verdict_is_refused(passing_confirmation):
    """Requirement 5's other half: nine verdicts or none."""
    partial = {
        layer: row for layer, row in passing_confirmation.items() if layer != 37
    }
    with pytest.raises(CorrectedControlRefused, match=r"physical layer\(s\) \[37\]"):
        corrected_band_verdict(
            partial, confirmation_manifest_checksum=CORRECTED_MOCK_MANIFEST_CHECKSUM
        )


def test_a_full_band_requires_every_physical_layer_32_to_40(passing_confirmation):
    """Requirement 19."""
    verdict = corrected_band_verdict(
        passing_confirmation,
        confirmation_manifest_checksum=CORRECTED_MOCK_MANIFEST_CHECKSUM,
    )
    assert verdict["verdict"] == CORRECTED_BAND_GO
    assert verdict["full_band_available"] is True
    assert verdict["largest_admissible_contiguous_band"] == [32, 40]
    assert verdict["layers_passing"] == list(range(32, 41))
    assert verdict["stage3_unblocked"] is True

    for dropped in (32, 36, 40):
        failing = dict(passing_confirmation)
        failing[dropped] = {**failing[dropped], "passed": False}
        reduced = corrected_band_verdict(
            failing, confirmation_manifest_checksum=CORRECTED_MOCK_MANIFEST_CHECKSUM
        )
        assert reduced["verdict"] == CORRECTED_BAND_NO_GO, dropped
        assert reduced["full_band_available"] is False, dropped
        assert reduced["stage3_unblocked"] is False, dropped
        assert reduced["largest_admissible_contiguous_band"] != [32, 40], dropped


def test_the_sub_band_comes_from_geometry_and_never_from_a_causal_outcome(
    passing_confirmation,
):
    failing = dict(passing_confirmation)
    failing[36] = {**failing[36], "passed": False}
    verdict = corrected_band_verdict(
        failing, confirmation_manifest_checksum=CORRECTED_MOCK_MANIFEST_CHECKSUM
    )
    # 32-35 and 37-40 both have four layers; the frozen rule takes the
    # shallowest start.
    assert verdict["largest_admissible_contiguous_band"] == [32, 35]
    assert "no causal outcome is consulted" in verdict["sub_band_selected_by"]
    serialized = json.dumps(verdict)
    for causal in ("target_top1", "swap_alpha", "reasoning_verdict"):
        assert causal not in serialized


# --------------------------------------------- 6-8: the matrices must be one fit


def test_the_merge_proves_scale_corpus_and_estimator_equivalence(universe):
    """Requirement 6."""
    evidence = universe.evidence
    assert evidence["passed"] is True
    assert evidence["failed_checks"] == []
    checked = {row["check"] for row in evidence["checks"]}
    for clause in (
        "model_repo_id", "model_revision", "target_layer", "d_model",
        "corpus_id", "corpus_revision", "fit_prefix_checksum", "estimator",
        "capture_geometry", "universe_coverage",
    ):
        assert clause in checked, clause
    assert any(row["check"].startswith("scale[") for row in evidence["checks"])
    assert evidence["no_matrix_was_refitted"] is True
    assert set(universe.matrix_checksums) == set(FIXED_CONTROL_UNIVERSE)

    for clause, bad in (
        ("model_revision", "another-revision"),
        ("corpus_id", "some/other-corpus"),
        ("corpus_revision", "another-corpus-revision"),
        ("fit_prefix_checksum", "sha256:a-different-250-prompt-prefix"),
        ("estimator", "a locally patched estimator"),
        ("target_layer", 40),
    ):
        first, second = mock_lens_snapshots()
        with pytest.raises(CorrectedControlRefused, match=clause):
            build_control_universe([first, replace(second, **{clause: bad})])


def test_an_unrecorded_clause_is_refused_rather_than_assumed():
    first, second = mock_lens_snapshots()
    with pytest.raises(CorrectedControlRefused, match="has not been demonstrated"):
        build_control_universe([first, replace(second, fit_prefix_checksum=None)])


def test_legacy_extension_geometry_comes_from_published_sidecar():
    """The real extension report predates its sidecar's complete capture plan."""
    geometry = {
        "target_layer": 41,
        "d_model": 2560,
        "dim_batch": 8,
        "max_seq_len": 128,
        "skip_first": 16,
        "n_layers": 42,
    }
    report = {"budget": {"anchor": {"target_layer": 41}}}
    artifact = {"capture_plan": {**geometry, "layers": [8, 14, 20, 26, 32]}}

    assert _extension_capture_geometry(report, artifact) == geometry


def test_complete_extension_geometry_sources_must_agree():
    geometry = {
        "target_layer": 41,
        "d_model": 2560,
        "dim_batch": 8,
        "max_seq_len": 128,
        "skip_first": 16,
        "n_layers": 42,
    }
    report = {"continuation": {"capture_plan": geometry}}
    artifact = {"capture_plan": {**geometry, "max_seq_len": 256}}

    with pytest.raises(CorrectedControlRefused, match="records disagree"):
        _extension_capture_geometry(report, artifact)


def test_missing_extension_geometry_is_not_defaulted():
    assert _extension_capture_geometry({}, {}) is None


def test_mixed_scale_artifacts_are_refused():
    """Requirement 7. A scale-100 artifact and a scale-250 artifact are
    different fits, and a band may not mix scales."""
    first, second = mock_lens_snapshots()
    with pytest.raises(CorrectedControlRefused, match="are not one lens"):
        build_control_universe([first, replace(second, scale=100)])
    with pytest.raises(CorrectedControlRefused, match="are not one lens"):
        build_control_universe([replace(first, scale=100), second])


def test_a_missing_interior_matrix_is_refused():
    """Requirement 8. The universe does not shrink to whatever is on hand."""
    extension, interior = mock_lens_snapshots()
    with pytest.raises(CorrectedControlRefused, match="universe_coverage"):
        build_control_universe([extension])
    short = replace(
        interior,
        layers=(33, 34, 36, 37),
        matrix_checksums={
            layer: f"sha256:mock-matrix-L{layer}" for layer in (33, 34, 36, 37)
        },
    )
    with pytest.raises(CorrectedControlRefused, match=r"layer\(s\) \[39\]"):
        build_control_universe([extension, short])


def test_two_snapshots_offering_one_layer_differently_are_refused():
    extension, interior = mock_lens_snapshots()
    clashing = replace(
        interior,
        layers=(*interior.layers, 35),
        matrix_checksums={**interior.matrix_checksums, 35: "sha256:a-different-L35"},
    )
    with pytest.raises(CorrectedControlRefused, match="not decided by iteration order"):
        build_control_universe([extension, clashing])


# ------------------------------------- 9: the superseded run stays byte-identical


def test_the_failed_run_directory_is_byte_for_byte_unchanged(tmp_path):
    """Requirement 9."""
    from jlens.mmpilot.band_control import band_interior_snapshot_facts

    run = _superseded_run(tmp_path)
    before = superseded_run_digest(run)
    extension, _ = mock_lens_snapshots()
    facts = band_interior_snapshot_facts(
        run, hook_convention_from=extension, scale=CORRECTED_SCALE
    )
    assert sorted(facts.matrix_checksums) == sorted(BAND_INTERIOR_LAYERS)
    proof = assert_superseded_run_unchanged(before, superseded_run_digest(run))
    assert proof["identical"] is True
    assert proof["added"] == proof["removed"] == proof["altered"] == []
    assert before["written_by_this_correction"] is False


def test_a_changed_file_in_the_superseded_run_is_refused(tmp_path):
    run = _superseded_run(tmp_path)
    before = superseded_run_digest(run)
    report = run / "artifacts" / "band_interior_lens_report.json"
    report.write_text(report.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(CorrectedControlRefused, match="changed during the correction"):
        assert_superseded_run_unchanged(before, superseded_run_digest(run))


def test_publication_refuses_to_write_into_the_superseded_run(tmp_path, universe, protocol):
    run = _superseded_run(tmp_path)
    corrected = tmp_path / "bandcorr" / "artifacts" / "corrected_validation_v1"
    lens = JacobianLens(jacobians={32: torch.eye(4)}, n_prompts=250, d_model=4)
    manifest = {"manifest_checksum": "sha256:manifest"}
    verdict = {
        "passed": True,
        "failed_checks": [],
        "confirmation_manifest_checksum": "sha256:manifest",
    }
    with pytest.raises(CorrectedControlRefused, match="completed-run evidence"):
        publish_corrected_layer(
            layer=32, scale=CORRECTED_SCALE, lens=lens,
            destination=run / "artifacts" / "published" / "lens.pt",
            confirmation_verdict=verdict, development_verdict=None,
            universe=universe, protocol=protocol, confirmation_manifest=manifest,
            corrected_dir=corrected, superseded_immutability={},
            protected_run_dirs=(run,),
        )
    assert not (run / "artifacts" / "published").exists()


def test_a_report_from_a_mock_run_supplies_no_matrices(tmp_path):
    from jlens.mmpilot.band_control import band_interior_snapshot_facts

    run = _superseded_run(tmp_path)
    report = run / "artifacts" / "band_interior_lens_report.json"
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["mode"] = "mock"
    report.write_text(json.dumps(payload), encoding="utf-8")
    extension, _ = mock_lens_snapshots()
    with pytest.raises(CorrectedControlRefused, match="MOCK report"):
        band_interior_snapshot_facts(run, hook_convention_from=extension)


# ------------------------------------ 10-12: the new confirmation population


def _population(pool, *, excluded=None, seed=None):
    base = {"superseded_development": (), "superseded_confirmation": ()}
    kwargs = {} if seed is None else {"seed": seed}
    return build_corrected_confirmation_population(
        pool,
        excluded={**base, **(excluded or {})},
        corpus_id="mock/train",
        n_confirmation=64,
        **kwargs,
    )


def test_the_population_refuses_to_build_without_the_spent_sets():
    pool = _pool()
    with pytest.raises(CorrectedControlRefused, match="development history"):
        build_corrected_confirmation_population(
            pool, excluded={}, corpus_id="mock/train", n_confirmation=64
        )
    with pytest.raises(CorrectedControlRefused, match="superseded_confirmation"):
        build_corrected_confirmation_population(
            pool,
            excluded={"superseded_development": ()},
            corpus_id="mock/train",
            n_confirmation=64,
        )


def test_previously_opened_records_cannot_enter_the_new_population():
    """Requirement 10."""
    pool = _pool()
    spent = pool[:200]
    splits, audit = _population(
        pool,
        excluded={
            "superseded_development": spent[:100],
            "superseded_confirmation": spent[100:],
        },
    )
    assert splits.protocol == CORRECTED_SPLIT_PROTOCOL
    assert len(splits.confirmation) == 64
    assert splits.development == ()
    chosen = {record.record_id for record in splits.confirmation}
    assert chosen & {record.record_id for record in spent} == set()
    assert audit["ok"] is True
    assert audit["n_exact_hits"] == audit["n_near_hits"] == 0

    # And the independent record-id guard catches what a drifted reconstruction
    # would leave behind.
    proof = assert_no_opened_records(
        splits,
        opened_record_ids={"superseded_confirmation": [r.record_id for r in spent]},
    )
    assert proof["ok"] is True
    with pytest.raises(CorrectedControlRefused, match="previously opened record"):
        assert_no_opened_records(
            splits,
            opened_record_ids={
                "superseded_confirmation": [splits.confirmation[3].record_id]
            },
        )


def test_exact_and_near_duplicate_exclusions_are_enforced():
    """Requirement 11, under the repository's own frozen SimHash rule."""
    pool = _pool()
    spent = pool[:120]
    exact_twin = replace(
        pool[400],
        record_id="mock/train/900000",
        stream_index=900_000,
        text=spent[0].text,
        normalized_checksum=spent[0].normalized_checksum,
        simhash=spent[0].simhash,
    )
    near_twin = replace(
        pool[401],
        record_id="mock/train/900001",
        stream_index=900_001,
        text=spent[1].text,
        normalized_checksum="sha256:not-the-same-bytes",
        simhash=spent[1].simhash ^ 0b111,  # Hamming distance 3
    )
    splits, audit = _population(
        [*pool, exact_twin, near_twin],
        excluded={"superseded_confirmation": spent},
    )
    chosen = {record.record_id for record in splits.confirmation}
    assert exact_twin.record_id not in chosen
    assert near_twin.record_id not in chosen
    assert splits.excluded_exact["superseded_confirmation"] >= 1
    assert splits.excluded_near["superseded_confirmation"] >= 1
    assert audit["ok"] is True


def test_selection_is_deterministic_under_input_permutation():
    """Requirement 12. One predeclared selection, and it does not depend on the
    order the pool happened to arrive in."""
    pool = _pool()
    spent = pool[:120]
    first, _ = _population(pool, excluded={"superseded_confirmation": spent})
    shuffled = list(reversed(pool))
    second, _ = _population(shuffled, excluded={"superseded_confirmation": spent})
    assert first.record_ids("confirmation") == second.record_ids("confirmation")
    assert first.checksum("confirmation") == second.checksum("confirmation")

    # A different seed is a different draw, so the determinism above is not
    # the function ignoring its inputs.
    other, _ = _population(pool, excluded={"superseded_confirmation": spent}, seed=1)
    assert other.record_ids("confirmation") != first.record_ids("confirmation")


def test_a_population_that_cannot_be_filled_is_refused_not_shrunk():
    with pytest.raises(ExtensionSplitRefused, match="not reduced because"):
        build_corrected_confirmation_population(
            _pool(80),
            excluded={"superseded_development": (), "superseded_confirmation": ()},
            corpus_id="mock/train",
            n_confirmation=256,
        )


# ------------------------------------------------ 13: predeclaration is ordered


def test_the_population_is_opened_only_after_the_protocol_is_persisted(
    tmp_path, universe, protocol
):
    """Requirement 13."""
    store = CorrectedControlStore(tmp_path / "bandcorr", _fingerprint())
    store.open()
    with pytest.raises(CorrectedControlRefused, match="not a predeclaration"):
        assert_protocol_persisted(store)

    store.save("corrected_universe", "universe", universe.evidence)
    with pytest.raises(CorrectedControlRefused, match="corrected_protocol/protocol"):
        assert_protocol_persisted(store)

    store.save("corrected_protocol", "protocol", protocol)
    proof = assert_protocol_persisted(store)
    assert proof["persisted_before_population_opened"] is True
    assert proof["protocol_digest"] == protocol["protocol_digest"]
    assert proof["universe_checksum"] == universe.digest


def test_the_protocol_fingerprints_everything_a_resume_must_not_mix(
    universe, protocol
):
    for field in (
        "correction_protocol_version", "model_repo_id", "model_revision",
        "transformers_version", "fixed_control_universe", "wrong_layer_mapping",
        "lens_checksums", "source_snapshot_checksums", "universe_checksum",
        "scale", "target_layer", "hook_convention", "fit_prefix_checksum",
        "exclusion_source_checksums", "gate_digest", "gate_thresholds",
        "target_token_discovery_protocol", "readout_implementation",
        "split_seed", "confirmation_prompt_seed", "control_seed",
        "superseded_run", "protocol_digest",
    ):
        assert field in protocol, field
    assert protocol["control_seed"] == CONTROL_SEED
    assert protocol["no_lens_was_refitted"] is True
    assert protocol["no_threshold_was_changed"] is True
    assert protocol["confirmation_population_not_inspected_when_frozen"] is True

    # A changed field is a changed digest, so a store bound to it refuses.
    changed = corrected_protocol_record(
        universe=universe,
        model_repo_id="google/gemma-4-E4B-it",
        model_revision="a-different-revision",
        exclusion_sources={"superseded_report": "sha256:mock-superseded"},
    )
    assert changed["protocol_digest"] != protocol["protocol_digest"]


def test_a_changed_fingerprint_refuses_to_resume(tmp_path, universe):
    from jlens.calibration.state import IncompatibleStateError

    root = tmp_path / "bandcorr"
    CorrectedControlStore(root, _fingerprint("first")).open()
    with pytest.raises(IncompatibleStateError):
        CorrectedControlStore(root, _fingerprint("second")).open()


def test_the_store_accepts_only_corrected_stages(tmp_path):
    store = CorrectedControlStore(tmp_path / "bandcorr", _fingerprint())
    store.open()
    with pytest.raises(ValueError, match="unknown corrected stage"):
        store.stage_dir("band_verdict")
    assert "corrected_validation_v1" in str(store.corrected_dir)
    assert str(store.published_path(32, CORRECTED_SCALE)).endswith(
        "lens.corrected.layer32.scale250.validated.pt"
    )


# --------------------------------------- 14, 16: the readout, resumable, no fit


def test_interrupted_scoring_resumes_at_fine_granularity(tmp_path):
    """Requirement 14. A disconnect costs the prompts in flight, not all 256."""
    model = MockCalibrationModel()
    lens = _small_lens(model)
    prompts = mock_corpus_texts(6)
    store = CorrectedControlStore(tmp_path / "bandcorr", _fingerprint())
    store.open()

    rows, first = score_corrected_readout_rows(
        model, lens, prompts, max_seq_len=48, store=store
    )
    assert first["n_prompts_computed"] == 6
    assert first["n_prompts_reused"] == 0
    assert len(rows) == 6 * len(BAND_SCORING_LAYERS) * 5

    resumed, second = score_corrected_readout_rows(
        model, lens, prompts, max_seq_len=48, store=store
    )
    assert second["n_prompts_computed"] == 0
    assert second["n_prompts_reused"] == 6
    assert resumed == rows

    # One layer removed from one prompt's unit: exactly that pair recomputes.
    path = store.unit_path("corrected_readout", "prompt00002")
    unit = json.loads(path.read_text(encoding="utf-8"))
    del unit["payload"]["layers"]["36"]
    unit["unit_checksum"] = payload_checksum(unit["payload"])
    path.write_text(json.dumps(unit), encoding="utf-8")
    partial, third = score_corrected_readout_rows(
        model, lens, prompts, max_seq_len=48, store=store
    )
    assert third["n_prompts_computed"] == 1
    assert third["n_prompts_partially_reused"] == 1
    assert third["n_layer_results_computed"] == 1
    assert third["n_layer_results_reused"] == 6 * len(BAND_SCORING_LAYERS) - 1
    assert partial == rows

    # A torn unit is recomputed rather than trusted.
    path.write_text("{not json", encoding="utf-8")
    _, fourth = score_corrected_readout_rows(
        model, lens, prompts, max_seq_len=48, store=store
    )
    assert fourth["n_prompts_computed"] == 1


def test_units_from_another_population_are_never_reused(tmp_path):
    """A changed manifest refuses resume rather than mixing units: a
    confirmation unit must not be reused as a development one, or vice versa."""
    model = MockCalibrationModel()
    lens = _small_lens(model)
    prompts = mock_corpus_texts(4)
    store = CorrectedControlStore(tmp_path / "bandcorr", _fingerprint())
    store.open()

    _, first = score_corrected_readout_rows(
        model, lens, prompts, max_seq_len=48, store=store,
        manifest_checksum="sha256:confirmation-population",
    )
    assert first["n_prompts_computed"] == 4
    assert first["manifest_checksum"] == "sha256:confirmation-population"

    _, same = score_corrected_readout_rows(
        model, lens, prompts, max_seq_len=48, store=store,
        manifest_checksum="sha256:confirmation-population",
    )
    assert same["n_prompts_reused"] == 4

    _, other = score_corrected_readout_rows(
        model, lens, prompts, max_seq_len=48, store=store,
        manifest_checksum="development:sha256:the-opened-records",
    )
    assert other["n_prompts_computed"] == 4
    assert other["n_prompts_reused"] == 0

    # Different prompts under the same manifest are recomputed too.
    _, moved = score_corrected_readout_rows(
        model, lens, mock_corpus_texts(8)[4:], max_seq_len=48, store=store,
        manifest_checksum="development:sha256:the-opened-records",
    )
    assert moved["n_prompts_computed"] == 4


def test_the_readout_is_forward_pass_only_and_calls_no_fitting_function(
    tmp_path, monkeypatch
):
    """Requirement 16."""
    import jlens.calibration.fitting as calibration_fitting
    import jlens.fitting as upstream_fitting

    def _explode(*args, **kwargs):  # pragma: no cover - the point is not calling it
        raise AssertionError("the corrected control must never fit a lens")

    monkeypatch.setattr(upstream_fitting, "fit", _explode)
    monkeypatch.setattr(calibration_fitting, "run_calibration", _explode)

    model = MockCalibrationModel()
    lens = _small_lens(model)
    store = CorrectedControlStore(tmp_path / "bandcorr", _fingerprint())
    store.open()
    rows, record = score_corrected_readout_rows(
        model, lens, mock_corpus_texts(4), max_seq_len=48, store=store
    )
    assert record["fitting_performed"] is False
    assert record["backward_passes"] == 0
    assert record["forward_passes"] == 4
    assert rows

    budget = corrected_readout_budget()
    assert budget["backward_passes"] == 0
    assert budget["fitting_performed"] is False
    assert "forward-pass only" in budget["workload"]

    # The module names the estimator as provenance and never imports or calls
    # it: no fitting entry point is reachable from the corrected control.
    source = (ROOT / "jlens" / "mmpilot" / "band_control.py").read_text(encoding="utf-8")
    for forbidden in (
        "from jlens.fitting import",
        "import jlens.fitting",
        "from jlens.calibration.fitting import",
        "import jlens.calibration.fitting",
        "run_calibration(",
        "fit(",
    ):
        assert forbidden not in source, forbidden
    assert 'estimator="jlens.fitting.fit (upstream, unmodified)"' in source


def test_every_scored_variant_is_present_in_the_rows(tmp_path):
    model = MockCalibrationModel()
    store = CorrectedControlStore(tmp_path / "bandcorr", _fingerprint())
    store.open()
    rows, _ = score_corrected_readout_rows(
        model, _small_lens(model), mock_corpus_texts(2), max_seq_len=48, store=store
    )
    assert {row["variant"] for row in rows} == {
        "j_lens", "permuted", "random", "wrong_layer", "logit_lens"
    }
    assert {row["layer"] for row in rows} == set(BAND_SCORING_LAYERS)


# ----------------------------------------------- 15: the gate is untouched


def test_thresholds_are_digest_identical_to_the_frozen_gate():
    """Requirement 15."""
    assert CORRECTED_GATE is EXTENSION_CONFIRMATION_GATE
    assert CORRECTED_GATE.digest == EXTENSION_CONFIRMATION_GATE.digest
    assert CORRECTED_GATE.digest == EXTENSION_GATE.digest
    # The digest the completed failed run recorded, unchanged by this repair.
    assert CORRECTED_GATE.digest == (
        "sha256:8c3e9121ec3235682534ef17f7af4070db1d957c918fa2b292db0f33dc253818"
    )
    assert CORRECTED_GATE.min_wrong_layer_mrr_margin == 0.15
    assert CORRECTED_GATE.wrong_layer_mapping == "distant_layer_mapping"
    for field in (
        "max_tied_at_max_rate", "min_noise_control_mrr_ratio",
        "min_noise_control_mrr_margin", "min_wrong_layer_mrr_margin",
        "max_median_midrank", "min_top_k_inclusion", "top_k", "n_folds",
        "min_fold_mrr_fraction", "max_target_token_share",
    ):
        assert getattr(CORRECTED_GATE, field) == getattr(CALIBRATION_GATE, field), field


def test_the_verdicts_are_produced_by_the_frozen_gate(passing_confirmation):
    for layer, row in passing_confirmation.items():
        assert row["gate_digest"] == CORRECTED_GATE.digest, layer
        assert row["gate"]["min_wrong_layer_mrr_margin"] == 0.15, layer
        assert row["wrong_layer_jacobian_fitted_at"] == WRONG_LAYER_MAPPING[layer]
        assert row["fixed_control_universe"] == list(FIXED_CONTROL_UNIVERSE)


def test_development_rescoring_is_never_labelled_confirmation():
    development = mock_corrected_confirmation("all_nine_pass", stage="development")
    for row in development.values():
        assert row["is_independent_confirmation"] is False
        assert row["stage"] == "validation"
    confirmation = mock_corrected_confirmation("all_nine_pass")
    for row in confirmation.values():
        assert row["is_independent_confirmation"] is True


# ----------------------------- 17: a failed layer is never publication eligible


def test_a_layer_failing_confirmation_is_not_publication_eligible():
    """Requirement 17, and the repair of the ambiguous ``publishable`` flag."""
    confirmation = mock_corrected_confirmation("one_interior_fails")
    rows = {row["layer"]: row for row in corrected_layer_rows(confirmation)}
    failed = rows[36]
    assert failed["confirmation_passed"] is False
    assert failed["matrix_artifact_exists"] is True
    assert failed["publication_eligible"] is False
    assert failed["published"] is False
    assert "publishable" not in failed
    assert rows[37]["publication_eligible"] is True

    # A layer with no matrix is not eligible either, however it scored.
    absent = corrected_layer_rows(
        confirmation, matrix_present={layer: layer != 34 for layer in BAND_SCORING_LAYERS}
    )
    row = next(row for row in absent if row["layer"] == 34)
    assert row["confirmation_passed"] is True
    assert row["matrix_artifact_exists"] is False
    assert row["publication_eligible"] is False


def test_publication_refuses_a_failed_layer(tmp_path, universe, protocol):
    confirmation = mock_corrected_confirmation("one_interior_fails")
    corrected = tmp_path / "artifacts" / "corrected_validation_v1"
    lens = JacobianLens(jacobians={36: torch.eye(4)}, n_prompts=250, d_model=4)
    manifest = {"manifest_checksum": CORRECTED_MOCK_MANIFEST_CHECKSUM}
    with pytest.raises(CorrectedControlRefused, match="never publication eligible"):
        publish_corrected_layer(
            layer=36, scale=CORRECTED_SCALE, lens=lens,
            destination=corrected / "published" / "lens.corrected.layer36.pt",
            confirmation_verdict=confirmation[36], development_verdict=None,
            universe=universe, protocol=protocol, confirmation_manifest=manifest,
            corrected_dir=corrected, superseded_immutability={},
        )
    assert not (corrected / "published").exists()


def test_publication_writes_a_new_versioned_artifact_and_never_overwrites(
    tmp_path, universe, protocol
):
    confirmation = mock_corrected_confirmation("all_nine_pass")
    corrected = tmp_path / "artifacts" / "corrected_validation_v1"
    destination = corrected / "published" / "lens.corrected.layer37.scale250.pt"
    manifest = {
        "manifest_checksum": CORRECTED_MOCK_MANIFEST_CHECKSUM,
        "record_ids": ["mock/train/1"],
    }
    lens = JacobianLens(jacobians={37: torch.eye(4)}, n_prompts=250, d_model=4)
    artifact = publish_corrected_layer(
        layer=37, scale=CORRECTED_SCALE, lens=lens, destination=destination,
        confirmation_verdict=confirmation[37],
        development_verdict={"passed": True, "failed_checks": []},
        universe=universe, protocol=protocol, confirmation_manifest=manifest,
        corrected_dir=corrected,
        superseded_immutability={"immutability_checksum": "sha256:immutable"},
    )
    assert destination.is_file()
    sidecar = destination.with_suffix(".corrected.json")
    assert sidecar.is_file()
    stored = json.loads(sidecar.read_text(encoding="utf-8"))
    assert stored["artifact_checksum"] == artifact["artifact_checksum"]
    assert stored["publication_status"] == "PUBLISHED_CORRECTED_CONTROL_BAND_LAYER"
    assert stored["fixed_control_universe"] == list(FIXED_CONTROL_UNIVERSE)
    assert stored["wrong_layer_mapping"]["37"] == 8
    assert stored["no_lens_was_refitted"] is True
    assert stored["no_threshold_was_changed"] is True
    assert stored["development_verdict"]["is_not_independent_confirmation"] is True

    with pytest.raises(CorrectedControlRefused, match="nothing is overwritten"):
        publish_corrected_layer(
            layer=37, scale=CORRECTED_SCALE, lens=lens, destination=destination,
            confirmation_verdict=confirmation[37], development_verdict=None,
            universe=universe, protocol=protocol, confirmation_manifest=manifest,
            corrected_dir=corrected, superseded_immutability={},
        )
    with pytest.raises(CorrectedControlRefused, match="outside the corrected"):
        publish_corrected_layer(
            layer=38, scale=CORRECTED_SCALE, lens=lens,
            destination=tmp_path / "elsewhere" / "lens.pt",
            confirmation_verdict=confirmation[38], development_verdict=None,
            universe=universe, protocol=protocol, confirmation_manifest=manifest,
            corrected_dir=corrected, superseded_immutability={},
        )
    with pytest.raises(CorrectedControlRefused, match="not one of the corrected"):
        publish_corrected_layer(
            layer=26, scale=CORRECTED_SCALE, lens=lens,
            destination=corrected / "published" / "lens.corrected.layer26.pt",
            confirmation_verdict={"passed": True, "failed_checks": []},
            development_verdict=None, universe=universe, protocol=protocol,
            confirmation_manifest=manifest, corrected_dir=corrected,
            superseded_immutability={},
        )


def test_publication_refuses_a_verdict_from_another_population(
    tmp_path, universe, protocol
):
    confirmation = mock_corrected_confirmation("all_nine_pass")
    corrected = tmp_path / "artifacts" / "corrected_validation_v1"
    lens = JacobianLens(jacobians={39: torch.eye(4)}, n_prompts=250, d_model=4)
    with pytest.raises(CorrectedControlRefused, match="not the population's"):
        publish_corrected_layer(
            layer=39, scale=CORRECTED_SCALE, lens=lens,
            destination=corrected / "published" / "lens.corrected.layer39.pt",
            confirmation_verdict=confirmation[39], development_verdict=None,
            universe=universe, protocol=protocol,
            confirmation_manifest={"manifest_checksum": "sha256:another-population"},
            corrected_dir=corrected, superseded_immutability={},
        )


# -------------------------------------------------------- 20: the MOCK cases


@pytest.mark.parametrize("key", sorted(CORRECTED_MOCK_SCENARIOS))
def test_mock_covers_every_commissioned_case(key):
    """Requirement 20."""
    scenario = CORRECTED_MOCK_SCENARIOS[key]
    confirmation = mock_corrected_confirmation(key)
    verdict = corrected_band_verdict(
        confirmation,
        confirmation_manifest_checksum=CORRECTED_MOCK_MANIFEST_CHECKSUM,
        development=mock_corrected_confirmation(key, stage="development"),
    )
    assert verdict["verdict"] == scenario.expected_verdict
    assert verdict["publication_eligible_layers"] == sorted(
        scenario.expected_published_layers
    )
    assert set(verdict["layers_passing"]) | set(verdict["layers_failing"]) == set(
        BAND_SCORING_LAYERS
    )


def test_mock_all_nine_pass_gives_a_full_band():
    verdict = corrected_band_verdict(
        mock_corrected_confirmation("all_nine_pass"),
        confirmation_manifest_checksum=CORRECTED_MOCK_MANIFEST_CHECKSUM,
    )
    assert verdict["verdict"] == CORRECTED_BAND_GO
    assert verdict["full_band_available"] is True


def test_mock_one_interior_failure_gives_no_full_band():
    verdict = corrected_band_verdict(
        mock_corrected_confirmation("one_interior_fails"),
        confirmation_manifest_checksum=CORRECTED_MOCK_MANIFEST_CHECKSUM,
    )
    assert verdict["layers_failing"] == [36]
    assert verdict["full_band_available"] is False
    assert verdict["largest_admissible_contiguous_band"] == [32, 35]


def test_mock_a_previously_confirmed_layer_failing_gives_no_full_band():
    """The case the whole correction exists for: an old confirmation verdict
    does not carry a layer through a new population."""
    verdict = corrected_band_verdict(
        mock_corrected_confirmation("previously_confirmed_fails"),
        confirmation_manifest_checksum=CORRECTED_MOCK_MANIFEST_CHECKSUM,
    )
    assert verdict["layers_failing"] == [35]
    assert 35 in (32, 35, 38, 40)  # a layer the earlier run confirmed
    assert verdict["full_band_available"] is False
    assert verdict["largest_admissible_contiguous_band"] == [36, 40]
    assert verdict["stage3_unblocked"] is False


# ------------------------------------------------- the manifest and the report


def test_the_confirmation_manifest_freezes_the_population_identity():
    pool = _pool()
    splits, leakage = _population(pool, excluded={"superseded_confirmation": pool[:80]})
    protocol = corrected_protocol_record(
        universe=mock_control_universe(),
        model_repo_id="mock", model_revision="mock",
        exclusion_sources={"superseded_report": "sha256:x"},
    )
    prompts = [record.text for record in splits.confirmation]
    manifest = corrected_confirmation_manifest(
        splits,
        protocol=protocol,
        prompts=prompts,
        selection={"selection_checksum": "sha256:selection", "diversity": {}},
        exclusion_digest="sha256:exclusions",
        corpus_revision="mock-corpus-revision",
        leakage_audit=leakage,
        opened_record_audit={"opened_record_audit_checksum": "sha256:opened"},
    )
    for field in (
        "selection_seed", "source_corpus_revision", "exclusion_set_digest",
        "record_ids", "prompt_hashes", "target_token_discovery_protocol",
        "min_distinct_target_tokens", "max_target_token_share",
        "manifest_checksum", "readout_implementation",
    ):
        assert field in manifest, field
    assert manifest["n_records"] == len(splits.confirmation)
    assert len(manifest["prompt_hashes"]) == len(prompts)
    assert manifest["one_deterministic_predeclared_selection"] is True
    assert manifest["population_searched_for_a_favourable_outcome"] is False
    assert manifest["min_distinct_target_tokens"] == 32
    recomputed = payload_checksum(
        {k: v for k, v in manifest.items() if k != "manifest_checksum"}
    )
    assert recomputed == manifest["manifest_checksum"]


def test_the_corrected_report_states_the_full_provenance(universe, protocol):
    confirmation = mock_corrected_confirmation("all_nine_pass")
    verdict = corrected_band_verdict(
        confirmation,
        confirmation_manifest_checksum=CORRECTED_MOCK_MANIFEST_CHECKSUM,
        protocol=protocol,
    )
    report = corrected_validation_report(
        mode="real",
        protocol=protocol,
        universe=universe,
        confirmation_manifest={"manifest_checksum": CORRECTED_MOCK_MANIFEST_CHECKSUM},
        development=mock_corrected_confirmation("all_nine_pass", stage="development"),
        confirmation=confirmation,
        verdict=verdict,
        publication={"n_published": 9},
        superseded_immutability={"identical": True},
    )
    provenance = report["provenance"]
    assert provenance["original_result"] == SUPERSEDED_VERDICT
    assert provenance["original_run"] == SUPERSEDED_RUN_NAME
    assert "immutable historical result" in provenance["original_result_status"]
    assert "superseded set-dependent-control validation" in (
        provenance["original_result_status"]
    )
    assert "depended on which layers happened to be fitted together" in (
        provenance["why_superseded_for_band_admissibility"]
    )
    assert provenance["no_frozen_numerical_threshold_was_changed"] is True
    assert provenance["no_matrix_was_refitted"] is True
    assert provenance[
        "new_confirmation_population_uninspected_when_protocol_frozen"
    ] is True
    assert provenance["previously_opened_records_are_development_only"] is True
    assert provenance["stage3_blocked_unless_all_nine_layers_pass"] is True
    assert provenance["old_and_new_verdicts_combined"] is False
    # It calls the first result superseded. The only place "fraudulent" or
    # "erased" may appear is in the sentence denying both.
    text = json.dumps(report).lower()
    for word in ("fraudulent", "erased"):
        assert text.count(word) == 1, word
    assert "not a fraudulent or erased one" in text
    for word in ("falsified", "fabricat", "misconduct", "dishonest"):
        assert word not in text, word
    assert report["superseded_wrong_layer_mapping"] == {
        str(k): v for k, v in sorted(SUPERSEDED_MAPPING_AS_RUN.items())
    }


def test_the_printed_blocks_name_both_mappings():
    block = format_wrong_layer_mapping(superseded=SUPERSEDED_WRONG_LAYER_MAPPING)
    assert "33->39" in block and "39->33" in block
    assert "does not move when the fitted subset changes" in block
    assert str(list(FIXED_CONTROL_UNIVERSE)) in block

    verdict = corrected_band_verdict(
        mock_corrected_confirmation("one_interior_fails"),
        confirmation_manifest_checksum=CORRECTED_MOCK_MANIFEST_CHECKSUM,
    )
    table = format_corrected_verdict(verdict)
    assert "eligible" in table and "published" in table
    assert CORRECTED_BAND_NO_GO in table


def test_evaluate_corrected_layers_uses_the_frozen_gate_unchanged():
    from jlens.calibration.extension_mock import mock_extension_rows

    rows = mock_extension_rows(
        CORRECTED_MOCK_SCENARIOS["all_nine_pass"],
        stage="confirmation", scale=CORRECTED_SCALE,
        n_prompts=CORRECTED_GATE.n_prompts, layers=BAND_SCORING_LAYERS,
    )
    stamped = evaluate_corrected_layers(
        rows, manifest_checksum="sha256:one-population"
    )
    from jlens.calibration.gate import evaluate_calibration_layers

    plain = evaluate_calibration_layers(
        rows, layers=list(BAND_SCORING_LAYERS), scale=CORRECTED_SCALE,
        stage="confirmation", gate=CORRECTED_GATE,
    )
    for layer in BAND_SCORING_LAYERS:
        assert stamped[layer]["passed"] == plain[layer]["passed"]
        assert stamped[layer]["checks"] == plain[layer]["checks"]
        assert stamped[layer]["metrics"] == plain[layer]["metrics"]
        assert stamped[layer]["confirmation_manifest_checksum"] == (
            "sha256:one-population"
        )
