# SPDX-License-Identifier: Apache-2.0
"""Deterministic CPU tests for the layer-32 confirmatory native readout.

The behaviours under test are the ones that decide whether a verdict means
anything: that the held-out prompts are genuinely new, that the frozen lens is
the one we think it is, that top-10 overlap cannot veto a primary-metric pass
(and cannot rescue a failing one), that a control beating the J-lens blocks the
pass, and that a resume never mixes results from a different configuration.
"""

import json

import pytest
import torch

from jlens.lens import JacobianLens
from jlens.metadata import file_sha256
from jlens.native_readout import (
    CONFIRMATORY_CRITERION,
    CONFIRMATORY_LAYER,
    CONFIRMATORY_PROMPT_SEED,
    CONFIRMATORY_PROTOCOL,
    CONTROL_SEED,
    CONTROL_VARIANTS,
    N_CONFIRMATORY_PROMPTS,
    VERDICT_NO_GO,
    VERDICT_VALIDATED,
    ConfirmatoryCriterion,
    ConfirmatoryFingerprint,
    ConfirmatoryStore,
    IncompatibleStateError,
    PromptOverlapError,
    build_readout_variants,
    confirmatory_report_markdown,
    evaluate_confirmatory,
    excluded_prompt_hashes,
    native_readout_row,
    prompt_sha256,
    select_confirmatory_prompts,
    summarize_variant,
    verify_saved_lens,
)

FITTED_LAYERS = [20, 26, 32, 38]
D_MODEL = 8


# ------------------------------------------------------------------- fixtures


def _lens(d_model=D_MODEL, layers=tuple(FITTED_LAYERS)):
    generator = torch.Generator().manual_seed(0)
    return JacobianLens(
        jacobians={
            layer: torch.randn(d_model, d_model, generator=generator) for layer in layers
        },
        n_prompts=32,
        d_model=d_model,
    )


@pytest.fixture
def saved_lens(tmp_path):
    """A frozen lens on disk with the manifest the v2 run would have written."""
    path = tmp_path / "lens.validated.pt"
    lens = _lens()
    lens.save(str(path))
    checksum = file_sha256(str(path))
    manifest = {
        "status": "validated_text_only",
        "lens_path": str(path),
        "lens_checksum": checksum,
        "model_revision": "fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd",
        "source_layers": FITTED_LAYERS,
        "n_prompts": 32,
        "prompt_protocol": "gemma-it-chat-text-only-v2-early-layer-sweep",
        "native_readout_layers_passing": [38],
        "native_validation_path": str(tmp_path / "native_readout_validation.json"),
    }
    return lens, path, checksum, manifest


def _verify(saved_lens, **overrides):
    lens, path, checksum, manifest = saved_lens
    kwargs = {
        "lens_path": path,
        "expected_checksum": checksum,
        "manifest": manifest,
        "expected_model_repo_id": "google/gemma-4-E4B-it",
        "expected_model_revision": "fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd",
        "expected_d_model": D_MODEL,
        "expected_hook_site": "block_output",
        "layer": CONFIRMATORY_LAYER,
    }
    kwargs.update(overrides)
    return verify_saved_lens(lens, **kwargs)


def _variant_rows(variant, *, agree, rank, overlap, n=N_CONFIRMATORY_PROMPTS, layer=CONFIRMATORY_LAYER):
    """Synthetic rows: the first ``agree`` prompts are top-1 hits (rank 1),
    the rest sit at ``rank``. Targets cycle so the run is nondegenerate."""
    rows = []
    for index in range(n):
        hit = index < agree
        actual_rank = 1 if hit else rank
        rows.append(
            {
                "sample": index,
                "prompt_sha256": f"{index:064d}",
                "layer": layer,
                "variant": variant,
                "target_token_id": 100 + (index % 7),
                "predicted_token_id": 100 + (index % 7) if hit else 999,
                "top1_agreement": hit,
                "target_rank": actual_rank,
                "reciprocal_rank": 1.0 / actual_rank,
                "top10_overlap": overlap,
            }
        )
    return rows


def _passing_rows(**overrides):
    """A row set that clears every clause, shaped like the real layer-32 case:
    the J-lens wins every primary metric while *losing* top-10 overlap to the
    wrong-layer control."""
    spec = {
        "j_lens": {"agree": 28, "rank": 4, "overlap": 0.113},
        "permuted": {"agree": 0, "rank": 900, "overlap": 0.010},
        "random": {"agree": 0, "rank": 1200, "overlap": 0.005},
        "wrong_layer": {"agree": 6, "rank": 40, "overlap": 0.150},
        "logit_lens": {"agree": 3, "rank": 60, "overlap": 0.220},
    }
    for name, changes in overrides.items():
        spec[name] = {**spec[name], **changes}
    rows = []
    for name, values in spec.items():
        rows.extend(_variant_rows(name, **values))
    return rows


# --------------------------------------------------------------------- prompts


def test_exactly_thirty_two_new_prompts_are_selected():
    excluded = {prompt_sha256(f"v2-{i}"): "v2_fitting" for i in range(40)}
    pool = [f"v2-{i}" for i in range(40)] + [f"new-{i}" for i in range(120)]

    prompts, manifest = select_confirmatory_prompts(pool, excluded=excluded)

    assert len(prompts) == N_CONFIRMATORY_PROMPTS == 32
    assert manifest["n_prompts"] == 32
    assert manifest["n_excluded"] == 40
    assert manifest["seed"] == CONFIRMATORY_PROMPT_SEED
    assert manifest["protocol"] == CONFIRMATORY_PROTOCOL
    assert len(manifest["prompts"]) == 32
    assert all(prompt.startswith("new-") for prompt in prompts)
    assert {row["pool_index"] for row in manifest["excluded"]} == set(range(40))


def test_prompt_selection_is_deterministic_under_the_fixed_seed():
    excluded = {prompt_sha256("fit"): "v2_fitting"}
    pool = ["fit"] + [f"new-{i}" for i in range(120)]

    first, _ = select_confirmatory_prompts(pool, excluded=excluded)
    second, _ = select_confirmatory_prompts(pool, excluded=excluded)
    other_seed, _ = select_confirmatory_prompts(pool, excluded=excluded, seed=7)

    assert first == second
    assert first != other_seed, "the documented seed must actually govern selection"


def test_selection_refuses_when_fitting_prompts_would_have_to_be_reused():
    pool = [f"fit-{i}" for i in range(32)] + [f"new-{i}" for i in range(10)]
    excluded = {prompt_sha256(f"fit-{i}"): "v2_fitting" for i in range(32)}

    with pytest.raises(PromptOverlapError, match="only 10 of 42"):
        select_confirmatory_prompts(pool, excluded=excluded)


def test_selection_refuses_when_original_validation_prompts_would_be_reused():
    pool = [f"heldout-{i}" for i in range(8)] + [f"new-{i}" for i in range(20)]
    excluded = {prompt_sha256(f"heldout-{i}"): "v2_validation" for i in range(8)}

    with pytest.raises(PromptOverlapError, match="v2 fitting/validation"):
        select_confirmatory_prompts(pool, excluded=excluded)


def test_truncated_manifest_hashes_still_block_an_overlapping_prompt():
    """``jlens.metadata.prompt_hashes`` writes 16-character digests; a manifest
    in that form must not let an already-seen prompt through."""
    pool = [f"new-{i}" for i in range(60)]
    excluded = {prompt_sha256(prompt)[:16]: "v2_fitting" for prompt in pool[:20]}

    prompts, manifest = select_confirmatory_prompts(pool, excluded=excluded)

    assert manifest["n_excluded"] == 20
    assert set(prompts).isdisjoint(pool[:20])


def test_duplicate_pool_entries_cannot_pad_the_heldout_set():
    pool = ["same"] * 100
    with pytest.raises(PromptOverlapError, match="duplicate"):
        select_confirmatory_prompts(pool, excluded={prompt_sha256("other"): "v2_fitting"})


def test_excluded_hashes_cover_both_v2_prompt_roles():
    excluded = excluded_prompt_hashes({"fit_hashes": ["a", "b"], "heldout_hashes": ["c"]})
    assert excluded == {"a": "v2_fitting", "b": "v2_fitting", "c": "v2_validation"}


def test_metadata_without_recorded_prompts_is_refused():
    with pytest.raises(ValueError, match="cannot prove"):
        excluded_prompt_hashes({"source": "wikitext"})


# ------------------------------------------------------------ lens verification


def test_verification_accepts_the_pinned_artifact(saved_lens):
    record = _verify(saved_lens)
    assert record["layer_under_test"] == 32
    assert record["fitted_source_layers"] == FITTED_LAYERS
    assert record["d_model"] == D_MODEL
    assert record["hook_site"] == "block_output"
    assert all(check["passed"] for check in record["checks"])
    assert {check["check"] for check in record["checks"]} >= {
        "lens_checksum",
        "model_revision",
        "hidden_dimension",
        "hook_site",
        "layer_is_fitted",
        "calibration_modality_is_text",
        "original_native_validation_recorded",
    }


def test_a_wrong_checksum_stops_the_run(saved_lens):
    with pytest.raises(ValueError, match="lens_checksum failed"):
        _verify(saved_lens, expected_checksum="sha256:" + "0" * 64)


def test_a_wrong_model_revision_stops_the_run(saved_lens):
    with pytest.raises(ValueError, match="model_revision failed"):
        _verify(saved_lens, expected_model_revision="deadbeef" * 5)


def test_a_manifest_disagreeing_with_the_file_stops_the_run(saved_lens):
    lens, path, checksum, manifest = saved_lens
    with pytest.raises(ValueError, match="manifest_checksum_agrees failed"):
        _verify(saved_lens, manifest={**manifest, "lens_checksum": "sha256:" + "1" * 64})


def test_a_non_text_calibration_stops_the_run(saved_lens):
    lens, path, checksum, manifest = saved_lens
    multimodal = {**manifest, "prompt_protocol": "spokencoco-multimodal-v1"}
    with pytest.raises(ValueError, match="calibration_modality_is_text failed"):
        _verify(saved_lens, manifest=multimodal)


def test_a_lens_without_layer_32_stops_the_run(tmp_path):
    path = tmp_path / "lens.pt"
    lens = _lens(layers=(20, 26, 38))
    lens.save(str(path))
    checksum = file_sha256(str(path))
    manifest = {
        "status": "validated_text_only",
        "lens_checksum": checksum,
        "model_revision": "rev",
        "source_layers": [20, 26, 38],
        "prompt_protocol": "gemma-it-chat-text-only-v2",
        "native_validation_path": "x.json",
    }
    with pytest.raises(ValueError, match="layer_is_fitted failed"):
        verify_saved_lens(
            lens,
            lens_path=path,
            expected_checksum=checksum,
            manifest=manifest,
            expected_model_repo_id="google/gemma-4-E4B-it",
            expected_model_revision="rev",
            expected_d_model=D_MODEL,
            expected_hook_site="block_output",
        )


def test_a_missing_original_validation_manifest_stops_the_run(saved_lens):
    lens, path, checksum, manifest = saved_lens
    stripped = {k: v for k, v in manifest.items() if k != "native_validation_path"}
    with pytest.raises(ValueError, match="original_native_validation_recorded failed"):
        _verify(saved_lens, manifest=stripped)


def test_verification_never_writes_to_the_lens_file(saved_lens):
    lens, path, checksum, _ = saved_lens
    before = path.read_bytes()
    _verify(saved_lens)
    assert path.read_bytes() == before
    assert file_sha256(str(path)) == checksum


# ---------------------------------------------------------------------- scoring


def test_a_row_scores_the_variant_against_the_model_own_prediction():
    actual = torch.tensor([0.0, 5.0, 1.0, 2.0])
    agreeing = torch.tensor([0.0, 9.0, 1.0, 2.0])
    row = native_readout_row(
        sample_index=3,
        prompt_sha="abc",
        layer=32,
        variant="j_lens",
        variant_logits=agreeing,
        actual_logits=actual,
        top_k=2,
    )
    assert row["target_token_id"] == 1
    assert row["top1_agreement"] is True
    assert row["target_rank"] == 1
    assert row["reciprocal_rank"] == 1.0
    assert row["layer"] == 32 and row["sample"] == 3


def test_a_disagreeing_variant_gets_a_worse_rank_and_no_agreement():
    actual = torch.tensor([0.0, 5.0, 1.0, 2.0])
    row = native_readout_row(
        sample_index=0,
        prompt_sha="abc",
        layer=32,
        variant="random",
        variant_logits=torch.tensor([9.0, 0.0, 8.0, 7.0]),
        actual_logits=actual,
        top_k=2,
    )
    assert row["top1_agreement"] is False
    assert row["target_rank"] == 4
    assert row["reciprocal_rank"] == pytest.approx(0.25)


def test_mismatched_vocabularies_are_refused():
    with pytest.raises(ValueError, match="do not match"):
        native_readout_row(
            sample_index=0,
            prompt_sha="a",
            layer=32,
            variant="j_lens",
            variant_logits=torch.zeros(3),
            actual_logits=torch.zeros(4),
        )


def test_summary_reports_every_primary_and_the_secondary_metric():
    metrics = summarize_variant(_variant_rows("j_lens", agree=24, rank=5, overlap=0.2, n=32))
    assert metrics["n_prompts"] == 32
    assert metrics["top1_agreement"] == pytest.approx(0.75)
    assert metrics["median_target_rank"] == pytest.approx(1.0)
    assert metrics["mean_top10_overlap"] == pytest.approx(0.2)


def test_controls_are_built_from_the_frozen_lens_without_modifying_it():
    lens = _lens()
    before = {layer: J.clone() for layer, J in lens.jacobians.items()}
    variants = build_readout_variants(lens, seed=CONTROL_SEED)

    assert set(variants) == {"j_lens", *CONTROL_VARIANTS}
    assert variants["j_lens"] is lens
    for layer, J in lens.jacobians.items():
        assert torch.equal(J, before[layer]), "the frozen lens was mutated"
    for name in CONTROL_VARIANTS:
        assert not torch.equal(variants[name].jacobians[32], lens.jacobians[32])


# --------------------------------------------------------------------- verdict


def test_a_clean_run_is_validated_for_multimodal_followup():
    verdict = evaluate_confirmatory(_passing_rows())
    assert verdict["verdict"] == VERDICT_VALIDATED
    assert verdict["passed"] is True
    assert verdict["failed_checks"] == []
    assert verdict["layer"] == 32
    assert verdict["criterion_digest"] == CONFIRMATORY_CRITERION.digest


def test_losing_top10_overlap_to_a_control_does_not_veto_a_primary_pass():
    """The exact v2 layer-32 shape: overlap 0.113 against a wrong-layer 0.150.
    Overlap is reported, and it blocks nothing."""
    verdict = evaluate_confirmatory(_passing_rows())

    assert verdict["metrics"]["j_lens"]["mean_top10_overlap"] < (
        verdict["metrics"]["wrong_layer"]["mean_top10_overlap"]
    )
    assert verdict["verdict"] == VERDICT_VALIDATED
    assert not any("top10" in check["check"] for check in verdict["checks"])
    assert all(check["primary"] for check in verdict["checks"])
    assert verdict["secondary_metrics"] == ["mean_top10_overlap"]
    assert verdict["secondary_metrics_are_non_blocking"] is True


def test_winning_top10_overlap_cannot_rescue_a_failed_primary_metric():
    rows = _passing_rows(
        j_lens={"agree": 4, "rank": 30, "overlap": 0.95},
        wrong_layer={"agree": 0, "rank": 40, "overlap": 0.01},
    )
    verdict = evaluate_confirmatory(rows)
    assert verdict["verdict"] == VERDICT_NO_GO
    assert "top1_agreement_floor" in verdict["failed_checks"]


def test_a_control_matching_the_mrr_blocks_the_pass():
    rows = _passing_rows(permuted={"agree": 28, "rank": 4, "overlap": 0.01})
    verdict = evaluate_confirmatory(rows)
    assert verdict["verdict"] == VERDICT_NO_GO
    assert "mrr_exceeds_all_controls" in verdict["failed_checks"]


def test_a_control_matching_the_median_rank_blocks_the_pass():
    rows = _passing_rows(random={"agree": 17, "rank": 1200, "overlap": 0.005})
    verdict = evaluate_confirmatory(rows)
    assert verdict["verdict"] == VERDICT_NO_GO
    assert "median_rank_better_than_all_controls" in verdict["failed_checks"]


def test_a_control_with_comparable_top1_agreement_blocks_the_pass():
    rows = _passing_rows(wrong_layer={"agree": 26, "rank": 40, "overlap": 0.15})
    verdict = evaluate_confirmatory(rows)
    assert verdict["verdict"] == VERDICT_NO_GO
    assert "no_control_reaches_comparable_top1" in verdict["failed_checks"]


def test_the_diagnostic_logit_lens_cannot_block_a_pass():
    """A logit lens that reads out better than the J-lens on every metric is a
    fact about the residual stream, not a randomised control."""
    rows = _passing_rows(logit_lens={"agree": 32, "rank": 1, "overlap": 0.99})
    verdict = evaluate_confirmatory(rows)
    assert verdict["verdict"] == VERDICT_VALIDATED
    assert verdict["diagnostic_variants"] == ["logit_lens"]
    assert "logit_lens" in verdict["metrics"]
    assert not any("logit" in check["check"] for check in verdict["checks"])


def test_a_short_run_is_not_a_pass():
    rows = [
        row
        for row in _passing_rows()
        if row["sample"] < 8  # the v2 sample size
    ]
    verdict = evaluate_confirmatory(rows)
    assert verdict["verdict"] == VERDICT_NO_GO
    assert "prompt_count" in verdict["failed_checks"]


def test_a_degenerate_single_target_run_is_not_a_pass():
    rows = _passing_rows()
    for row in rows:
        row["target_token_id"] = 100
    verdict = evaluate_confirmatory(rows)
    assert verdict["verdict"] == VERDICT_NO_GO
    assert "finite_and_nondegenerate" in verdict["failed_checks"]


def test_a_non_finite_metric_is_not_a_pass():
    rows = _passing_rows()
    for row in rows:
        if row["variant"] == "j_lens":
            row["reciprocal_rank"] = float("nan")
    verdict = evaluate_confirmatory(rows)
    assert verdict["verdict"] == VERDICT_NO_GO
    assert "finite_and_nondegenerate" in verdict["failed_checks"]


def test_only_layer_32_rows_are_scored():
    rows = _passing_rows()
    noise = _variant_rows("j_lens", agree=0, rank=9999, overlap=0.0, layer=38)
    verdict = evaluate_confirmatory(rows + noise)
    assert verdict["metrics"]["j_lens"]["n_prompts"] == 32
    assert verdict["verdict"] == VERDICT_VALIDATED


def test_the_verdict_is_always_one_of_the_two_declared_strings():
    for rows in (_passing_rows(), _passing_rows(j_lens={"agree": 0, "rank": 900, "overlap": 0.0})):
        assert evaluate_confirmatory(rows)["verdict"] in {VERDICT_VALIDATED, VERDICT_NO_GO}


def test_the_report_states_the_verdict_the_criterion_and_the_overlap_caveat(saved_lens):
    verdict = evaluate_confirmatory(_passing_rows())
    _, _, _, manifest = saved_lens
    report = confirmatory_report_markdown(
        verdict,
        prompt_manifest={"n_prompts": 32, "seed": CONFIRMATORY_PROMPT_SEED, "n_excluded": 40},
        lens_record=_verify(saved_lens),
    )
    assert VERDICT_VALIDATED in report
    assert "secondary metric" in report
    assert "cannot veto a primary-metric pass" in report
    assert "No lens was fitted" in report
    assert "| `wrong_layer` | control |" in report


def test_a_failing_report_recommends_keeping_only_layer_38(saved_lens):
    verdict = evaluate_confirmatory(_passing_rows(j_lens={"agree": 2, "rank": 80, "overlap": 0.1}))
    report = confirmatory_report_markdown(
        verdict,
        prompt_manifest={"n_prompts": 32, "seed": CONFIRMATORY_PROMPT_SEED, "n_excluded": 40},
        lens_record=_verify(saved_lens),
    )
    assert VERDICT_NO_GO in report
    assert "keep only" in report.lower() and "layer 38" in report


# ----------------------------------------------------------------------- resume


def _fingerprint(**overrides):
    base = {
        "protocol": CONFIRMATORY_PROTOCOL,
        "lens_checksum": "sha256:" + "a" * 64,
        "model_repo_id": "google/gemma-4-E4B-it",
        "model_revision": "fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd",
        "prompt_protocol": "gemma-it-chat-text-only-v2-early-layer-sweep",
        "prompt_seed": CONFIRMATORY_PROMPT_SEED,
        "prompt_hashes": tuple(f"{i:064d}" for i in range(32)),
        "layer": 32,
        "controls": CONTROL_VARIANTS,
        "control_seed": CONTROL_SEED,
        "criterion_digest": CONFIRMATORY_CRITERION.digest,
    }
    base.update(overrides)
    return ConfirmatoryFingerprint(**base)


def test_a_fresh_directory_starts_and_a_matching_one_resumes(tmp_path):
    root = tmp_path / "layer32_confirmatory_validation"
    assert ConfirmatoryStore(root, _fingerprint()).open() == "starting"
    assert ConfirmatoryStore(root, _fingerprint()).open() == "resuming"


def test_a_completed_prompt_is_reused_and_not_recomputed(tmp_path):
    store = ConfirmatoryStore(tmp_path / "run", _fingerprint())
    store.open()
    sha = f"{0:064d}"
    payload = {"sample": 0, "prompt_sha256": sha, "rows": _variant_rows("j_lens", agree=1, rank=1, overlap=0.1, n=1)}
    store.save_result(0, sha, payload)

    resumed = ConfirmatoryStore(tmp_path / "run", _fingerprint())
    assert resumed.open() == "resuming"
    assert resumed.has_result(0, sha)
    assert resumed.load_result(0, sha) == payload
    assert resumed.load_result(1, f"{1:064d}") is None
    assert resumed.status_report()["stored_prompt_results"] == 1


def test_results_are_written_atomically_and_leave_no_partial_files(tmp_path):
    store = ConfirmatoryStore(tmp_path / "run", _fingerprint())
    store.open()
    sha = f"{7:064d}"
    path = store.save_result(7, sha, {"rows": []})

    assert path.is_file()
    assert list(path.parent.glob("*.tmp*")) == []
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["schema"] == ConfirmatoryStore.SCHEMA
    assert record["fingerprint_digest"] == store.fingerprint.digest
    assert record["prompt_sha256"] == sha


def test_aggregate_artifacts_are_written_atomically(tmp_path):
    store = ConfirmatoryStore(tmp_path / "run", _fingerprint())
    store.open()
    store.write_artifact("verdict.json", {"verdict": VERDICT_VALIDATED})
    store.write_artifact("report.md", "# report\n")

    assert json.loads((store.root / "verdict.json").read_text(encoding="utf-8"))["verdict"] == VERDICT_VALIDATED
    assert (store.root / "report.md").read_text(encoding="utf-8") == "# report\n"
    assert list(store.root.glob("*.tmp*")) == []


def test_a_torn_result_is_treated_as_missing_rather_than_trusted(tmp_path):
    store = ConfirmatoryStore(tmp_path / "run", _fingerprint())
    store.open()
    sha = f"{2:064d}"
    path = store.save_result(2, sha, {"rows": [{"variant": "j_lens"}]})

    record = json.loads(path.read_text(encoding="utf-8"))
    record["payload"]["rows"] = [{"variant": "tampered"}]
    path.write_text(json.dumps(record), encoding="utf-8")

    assert store.load_result(2, sha) is None
    assert str(path) in store.invalid_units


@pytest.mark.parametrize(
    ("field", "value", "needle"),
    [
        ("lens_checksum", "sha256:" + "b" * 64, "lens_checksum"),
        ("model_revision", "0" * 40, "model_revision"),
        ("prompt_protocol", "some-other-protocol", "prompt_protocol"),
        ("prompt_seed", 999, "prompt_seed"),
        ("prompt_hashes", tuple(f"{i:064d}" for i in range(1, 33)), "held-out set changed"),
        ("layer", 38, "layer"),
        ("controls", ("permuted", "random"), "controls"),
        ("control_seed", 4321, "control_seed"),
        ("criterion_digest", "sha256:" + "c" * 64, "criterion_digest"),
    ],
)
def test_an_incompatible_resume_is_refused_and_names_the_field(tmp_path, field, value, needle):
    root = tmp_path / "run"
    ConfirmatoryStore(root, _fingerprint()).open()

    with pytest.raises(IncompatibleStateError) as excinfo:
        ConfirmatoryStore(root, _fingerprint(**{field: value})).open()

    assert needle in str(excinfo.value)
    assert "refusing to reuse" in str(excinfo.value)


def test_a_weakened_criterion_invalidates_stored_results(tmp_path):
    """Editing the criterion after the fact must not silently rescore old rows."""
    root = tmp_path / "run"
    ConfirmatoryStore(root, _fingerprint()).open()
    weakened = ConfirmatoryCriterion(min_top1_agreement=0.10, control_top1_margin=0.0)

    assert weakened.digest != CONFIRMATORY_CRITERION.digest
    with pytest.raises(IncompatibleStateError, match="criterion_digest"):
        ConfirmatoryStore(root, _fingerprint(criterion_digest=weakened.digest)).open()


def test_results_from_a_different_fingerprint_are_not_loaded(tmp_path):
    root = tmp_path / "run"
    store = ConfirmatoryStore(root, _fingerprint())
    store.open()
    sha = f"{0:064d}"
    store.save_result(0, sha, {"rows": []})

    # Same directory, different fingerprint: the gate refuses before any read.
    with pytest.raises(IncompatibleStateError):
        ConfirmatoryStore(root, _fingerprint(layer=38)).open()


# -------------------------------------------------------------------- end to end


def _mock_run(root, prompts, *, layer):
    """The notebook's scoring loop, verbatim in shape, against the CPU mock."""
    from jlens.gemma4 import Gemma4LensModel
    from jlens.hooks import ActivationRecorder

    from .mock_gemma4 import MockGemma4ForConditionalGeneration, MockTokenizer

    model = Gemma4LensModel(MockGemma4ForConditionalGeneration(), MockTokenizer())
    lens = _lens(d_model=model.d_model, layers=(1, 2, layer, 4))
    variants = build_readout_variants(lens, seed=CONTROL_SEED)
    hashes = [prompt_sha256(prompt) for prompt in prompts]

    fingerprint = _fingerprint(prompt_hashes=tuple(hashes), layer=layer)
    store = ConfirmatoryStore(root, fingerprint)
    status = store.open()

    final_layer = model.n_layers - 1
    rows, reused, computed = [], 0, 0
    for index, prompt in enumerate(prompts):
        sha = hashes[index]
        cached = store.load_result(index, sha)
        if cached is not None:
            rows.extend(cached["rows"])
            reused += 1
            continue
        ids = model.encode(prompt, max_length=32)
        with torch.no_grad():
            with ActivationRecorder(model.layers, at=[layer, final_layer]) as recorder:
                model.forward(ids)
            actual = model.unembed(recorder.activations[final_layer][0, -1].float())
            residual = recorder.activations[layer][0, -1].float()
            scored = {"logit_lens": model.unembed(residual)}
            for name, variant in variants.items():
                scored[name] = model.unembed(variant.transport(residual, layer))
        prompt_rows = [
            native_readout_row(
                sample_index=index,
                prompt_sha=sha,
                layer=layer,
                variant=name,
                variant_logits=scored[name],
                actual_logits=actual,
            )
            for name in ("j_lens", *CONTROL_VARIANTS, "logit_lens")
        ]
        store.save_result(index, sha, {"sample": index, "prompt_sha256": sha, "rows": prompt_rows})
        rows.extend(prompt_rows)
        computed += 1
    return store, status, rows, reused, computed


def test_the_full_scoring_loop_runs_and_resumes_on_the_cpu_mock(tmp_path):
    """Exercises the notebook's real path — forward pass, transport, unembed,
    scoring, atomic save, verdict — with no Gemma and no Drive."""
    prompts = [f"held-out passage number {i}" for i in range(N_CONFIRMATORY_PROMPTS)]
    root = tmp_path / "layer32_confirmatory_validation"

    store, status, rows, reused, computed = _mock_run(root, prompts, layer=3)
    assert status == "starting"
    assert (reused, computed) == (0, 32)
    assert len(rows) == 32 * 5
    assert {row["variant"] for row in rows} == {"j_lens", *CONTROL_VARIANTS, "logit_lens"}

    verdict = evaluate_confirmatory(rows, layer=3)
    assert verdict["verdict"] in {VERDICT_VALIDATED, VERDICT_NO_GO}
    assert verdict["metrics"]["j_lens"]["n_prompts"] == 32
    assert all(
        0.0 <= verdict["metrics"][name]["top1_agreement"] <= 1.0
        for name in ("j_lens", *CONTROL_VARIANTS, "logit_lens")
    )

    # A rerun reuses every checksum-valid result and recomputes nothing.
    _, resumed_status, resumed_rows, reused, computed = _mock_run(root, prompts, layer=3)
    assert resumed_status == "resuming"
    assert (reused, computed) == (32, 0)
    assert resumed_rows == rows

    report = confirmatory_report_markdown(
        verdict,
        prompt_manifest={"n_prompts": 32, "seed": CONFIRMATORY_PROMPT_SEED, "n_excluded": 40},
        lens_record={
            "lens_path": "lens.validated.pt",
            "lens_checksum": "sha256:" + "a" * 64,
            "model_revision": "rev",
            "fitted_source_layers": [1, 2, 3, 4],
        },
    )
    assert verdict["verdict"] in report
