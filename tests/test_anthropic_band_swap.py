"""The contiguous-band coordinate swap: admissibility, clamp, scoring, resume.

Numbered against the requirements the band study was specified with. Where a
requirement was already covered for a single layer or an arbitrary layer set,
the test here is the *band* version of it — the clamp is what is new, so what is
tested is that every physical layer of a contiguous band is patched, in one
forward pass, from its own recomputed coordinates.
"""

from __future__ import annotations

import json
import shutil

import pytest
import torch

from jlens.calibration.publication import PublicationRefused
from jlens.mmpilot.band_lens import (
    BAND_INTERIOR_LAYERS,
    BAND_SCALE,
    BandLensRefused,
    band_capture_plan,
    band_fit_budget,
    band_layer_verdict,
    band_scale_selection,
    publish_band_layer,
)
from jlens.mmpilot.band_swap import (
    BAND_CONDITIONS,
    BAND_INTERVENTION_FAMILY,
    CONDITION_ALPHA,
    PARTIAL_MOVEMENT,
    BandDesignRefused,
    BandLensRow,
    BandSwapThresholds,
    assert_contiguous,
    band_design_record,
    band_key,
    band_onset_timing,
    band_reasoning_verdict,
    band_swap_fingerprint,
    band_trial_record,
    bootstrap_interval,
    build_band,
    contiguous_runs,
    controls_for_condition,
    format_lens_inventory,
    largest_admissible_band,
    lens_inventory,
    predeclare_suffix_bands,
    summarize_band_cells,
)
from jlens.mmpilot.band_swap_mock import (
    MOCK_BAND_END,
    MOCK_BAND_STARTS,
    MOCK_BAND_USABLE_LAYERS,
    mock_band_grid,
    run_mock_band_trials,
)
from jlens.mmpilot.coordinate_swap import (
    METHOD_VERSION,
    LayerBandError,
    coordinate_swap_band,
    run_swap_condition,
)
from jlens.mmpilot.coordinate_swap_mock import (
    IDENTITY_QUESTION,
    MOCK_VALIDATED_LAYERS,
    PROPERTY_QUESTION,
    SwapMockBackend,
    mock_bases,
    mock_concept_tokens,
)
from jlens.mmpilot.store import RunFingerprint, UnitStore, payload_checksum, safe_key

FULL_BAND = tuple(range(MOCK_BAND_STARTS[0], MOCK_BAND_END + 1))


@pytest.fixture(scope="module")
def backend():
    return SwapMockBackend()


@pytest.fixture(scope="module")
def tokens(backend):
    return mock_concept_tokens(backend)


@pytest.fixture(scope="module")
def band_bases(backend, tokens):
    return mock_bases(
        backend.world, layers=FULL_BAND, source=tokens["bird"], target=tokens["cat"]
    )


def _identity_inputs(backend, *, modality="image"):
    return backend.build_inputs(
        prompt=IDENTITY_QUESTION, modality=modality, concept="bird"
    )


def _property_inputs(backend, *, modality="image"):
    return backend.build_inputs(
        prompt=PROPERTY_QUESTION, modality=modality, concept="bird"
    )


def _candidate_ids(backend, names):
    return {name: backend.encode_candidate(f" {name}") for name in names}


# ---------------------------------------------------------------- 1, 2, 3


def test_band_hooks_fire_at_every_physical_band_layer(backend, band_bases):
    """Requirement 1. Six hooks, one forward pass, no layer skipped."""
    inputs = _identity_inputs(backend)
    with coordinate_swap_band(
        backend.blocks, band_bases, alpha=1.0, prompt_len=inputs.prompt_len
    ) as stats:
        backend.forward_logits(inputs.tensors)
    assert tuple(sorted(stats)) == FULL_BAND
    for layer in FULL_BAND:
        assert stats[layer]["n_forward_passes"] == 1, layer
        assert min(stats[layer]["swap"]["update_norm"]) > 0.0, layer


def test_each_band_layer_recomputes_its_own_coordinates(backend, band_bases):
    """Requirement 2. Not one update computed once and replayed."""
    inputs = _identity_inputs(backend)
    with coordinate_swap_band(
        backend.blocks, band_bases, alpha=1.0, prompt_len=inputs.prompt_len
    ) as stats:
        backend.forward_logits(inputs.tensors)
    seen = [
        tuple(stats[layer]["swap"]["coordinates_before"][1]) for layer in FULL_BAND
    ]
    assert len(set(seen)) == len(seen), seen


def test_every_original_prompt_position_is_patched_at_every_band_layer(
    backend, band_bases
):
    """Requirement 3."""
    inputs = _identity_inputs(backend)
    with coordinate_swap_band(
        backend.blocks, band_bases, alpha=1.0, prompt_len=inputs.prompt_len
    ) as stats:
        backend.forward_logits(inputs.tensors)
    for layer in FULL_BAND:
        assert stats[layer]["positions"] == list(range(inputs.prompt_len))
        assert stats[layer]["n_positions"] == inputs.prompt_len


# ------------------------------------------------------------------- 4, 5


def test_candidate_completion_positions_are_bit_identical_under_a_band(
    backend, band_bases
):
    """Requirement 4, measured on activations at every band layer."""
    from jlens.hooks import ActivationRecorder
    from jlens.mmpilot.capability import _extend_tensors

    inputs = _identity_inputs(backend)
    candidate_ids = _candidate_ids(backend, ("bird", "cat"))
    extended = _extend_tensors(inputs.tensors, inputs.prompt_len, candidate_ids["cat"])
    seq_len = extended["input_ids"].shape[1]
    assert seq_len > inputs.prompt_len

    layers = list(FULL_BAND)
    with ActivationRecorder(backend.blocks, at=layers) as recorder:
        backend.forward_logits(extended)
        clean = {i: recorder.activations[i].detach().clone() for i in layers}
    with coordinate_swap_band(
        backend.blocks, band_bases, alpha=1.0, prompt_len=inputs.prompt_len
    ) as stats:
        with ActivationRecorder(backend.blocks, at=layers) as recorder:
            backend.forward_logits(extended)
            patched = {i: recorder.activations[i].detach().clone() for i in layers}

    first = min(layers)
    for position in range(inputs.prompt_len, seq_len):
        assert torch.equal(clean[first][0, position], patched[first][0, position]), (
            f"candidate position {position} was modified at the first band layer"
        )
    for layer in layers:
        assert stats[layer]["n_candidate_positions_skipped"] == seq_len - inputs.prompt_len


def test_alpha_zero_over_a_band_is_a_bit_exact_no_op(backend, band_bases):
    """Requirement 5. Every hook fires; the scores are bit-identical."""
    from jlens.mmpilot.capability import score_candidate_sequences

    inputs = _identity_inputs(backend)
    candidate_ids = _candidate_ids(backend, ("bird", "cat"))
    clean = score_candidate_sequences(backend, inputs, candidate_ids)
    with coordinate_swap_band(
        backend.blocks, band_bases, alpha=0.0, prompt_len=inputs.prompt_len
    ) as stats:
        patched = score_candidate_sequences(backend, inputs, candidate_ids)
    assert all(stats[layer]["n_forward_passes"] > 0 for layer in FULL_BAND)
    for name in clean:
        assert patched[name]["sum_logprob"] == clean[name]["sum_logprob"], name


# ------------------------------------------------------------------- 6, 7


def test_alpha_one_is_the_exact_exchange_at_every_band_layer(backend, band_bases):
    """Requirement 6, read out of each layer's own record."""
    inputs = _identity_inputs(backend)
    with coordinate_swap_band(
        backend.blocks, band_bases, alpha=1.0, prompt_len=inputs.prompt_len
    ) as stats:
        backend.forward_logits(inputs.tensors)
    for layer in FULL_BAND:
        record = stats[layer]["swap"]
        assert record["alpha"] == 1.0
        assert record["alpha_is_extrapolation"] is False
        for before, after in zip(
            record["coordinates_before"], record["coordinates_after"], strict=True
        ):
            assert after[0] == pytest.approx(before[1], abs=1e-9)
            assert after[1] == pytest.approx(before[0], abs=1e-9)


def test_alpha_two_is_the_declared_extrapolation_at_every_band_layer(
    backend, band_bases
):
    """Requirement 7. c + 2(sigma(c) - c) overshoots sigma(c), and says so."""
    inputs = _identity_inputs(backend)
    with coordinate_swap_band(
        backend.blocks, band_bases, alpha=2.0, prompt_len=inputs.prompt_len
    ) as stats:
        backend.forward_logits(inputs.tensors)
    for layer in FULL_BAND:
        record = stats[layer]["swap"]
        assert record["alpha"] == 2.0
        assert record["alpha_is_extrapolation"] is True
        for before, after in zip(
            record["coordinates_before"], record["coordinates_after"], strict=True
        ):
            assert after[0] == pytest.approx(2 * before[1] - before[0], abs=1e-9)
            assert after[1] == pytest.approx(2 * before[0] - before[1], abs=1e-9)


# ---------------------------------------------------------------------- 8


def _design(usable=(32, 33, 34, 35, 36, 37, 38, 39, 40), starts=(32, 35, 38, 40)):
    inventory = lens_inventory(
        [
            BandLensRow(
                layer=layer,
                lens_path=f"/lens/L{layer}.pt",
                lens_checksum=f"sha256:{layer:064d}",
                fit_scale=BAND_SCALE,
                fit_corpus="wikitext@rev",
                validation_set="third-generation",
                confirmation_verdict="PASSED",
                usable=True,
                provenance="test",
            )
            for layer in usable
        ],
        layer_range=(32, 40),
        required_scale=BAND_SCALE,
    )
    bands = predeclare_suffix_bands(
        starts=starts, end=40, usable_layers=inventory["usable_layers"], n_layers=42
    )
    return inventory, band_design_record(
        inventory=inventory,
        primary_band=build_band(32, 40, usable_layers=inventory["usable_layers"]),
        suffix_bands=bands,
    )


def _fingerprint(design, checksums, **overrides):
    payload = {
        "design": design,
        "lens_checksums": checksums,
        "model_repo_id": "google/gemma-4-E4B-it",
        "model_revision": "rev",
        "processor_revision": "prev",
        "transformers_version": "5.13.1",
        "audio_protocol_fingerprint": "sha256:audio",
        "prompt_protocol": [{"prompt_protocol_version": "mmpilot.hidden_animal_legs.v1"}],
        "directed_pairs": [{"source": "bird", "target": "cat"}],
        "sample_identities": {"population_digest": "sha256:pop"},
        "thresholds": BandSwapThresholds().to_dict(),
        "coordinate_swap_method_version": METHOD_VERSION,
        "scoring_rule": "top-1 of the externally scored candidates",
    }
    payload.update(overrides)
    return band_swap_fingerprint(**payload)


def test_band_order_and_every_lens_checksum_move_the_fingerprint():
    """Requirement 8."""
    _, design = _design()
    checksums = {layer: f"sha256:{layer:064d}" for layer in range(32, 41)}
    base = _fingerprint(design, checksums)

    _, other_bands = _design(starts=(32, 36, 40))
    assert _fingerprint(other_bands, checksums)["band_fingerprint_digest"] != base[
        "band_fingerprint_digest"
    ]

    for layer in range(32, 41):
        moved = dict(checksums)
        moved[layer] = "sha256:" + "f" * 64
        assert _fingerprint(design, moved)["band_fingerprint_digest"] != base[
            "band_fingerprint_digest"
        ], f"layer {layer}'s lens checksum does not reach the fingerprint"

    assert _fingerprint(design, checksums)["band_fingerprint_digest"] == base[
        "band_fingerprint_digest"
    ]
    with pytest.raises(BandDesignRefused, match="no lens checksum recorded"):
        _fingerprint(design, {layer: "sha256:x" for layer in range(32, 40)})


# ------------------------------------------------------------------ 9, 10


def test_removing_an_interior_lens_refuses_the_band():
    """Requirement 9. A gap is a refusal, not a narrower band chosen quietly."""
    usable = tuple(layer for layer in range(32, 41) if layer != 36)
    with pytest.raises(LayerBandError, match=r"\[36\]"):
        build_band(32, 40, usable_layers=usable)
    with pytest.raises(LayerBandError, match=r"\[36\]"):
        predeclare_suffix_bands(starts=(32, 35), end=40, usable_layers=usable)
    # The honest fallback is reported, and it is chosen from geometry alone:
    # longest run, ties to the shallowest start (32-35 and 37-40 are both four
    # layers, so the shallower one wins rather than the one nearer the answer).
    assert contiguous_runs(usable) == ((32, 35), (37, 40))
    assert largest_admissible_band(usable, low=32, high=40) == (32, 35)
    longer = tuple(layer for layer in range(32, 41) if layer != 34)
    assert largest_admissible_band(longer, low=32, high=40) == (35, 40)


def test_the_sampled_grid_is_rejected_as_non_contiguous():
    """Requirement 10. [32, 35, 38, 40] is four layers, not the range 32-40."""
    with pytest.raises(BandDesignRefused, match=r"\[33, 34, 36, 37, 39\]"):
        assert_contiguous((32, 35, 38, 40))
    with pytest.raises(BandDesignRefused, match="not contiguous"):
        band_key((32, 35, 38, 40))
    inventory = lens_inventory(
        [
            BandLensRow(
                layer=layer,
                lens_path=f"/L{layer}.pt",
                lens_checksum="sha256:x",
                fit_scale=BAND_SCALE,
                usable=True,
            )
            for layer in (32, 35, 38, 40)
        ],
        layer_range=(32, 40),
        required_scale=BAND_SCALE,
    )
    assert inventory["usable_layers"] == [32, 35, 38, 40]
    assert inventory["contiguous_usable_runs"] == [[32, 32], [35, 35], [38, 38], [40, 40]]
    with pytest.raises(LayerBandError):
        build_band(32, 40, usable_layers=inventory["usable_layers"])
    assert "no artifact was discovered" in format_lens_inventory(inventory)


def test_a_lens_at_another_scale_cannot_enter_the_band():
    """A band may not mix scales, and the table says why a layer is unusable."""
    inventory = lens_inventory(
        [
            BandLensRow(
                layer=35,
                lens_path="/L35.pt",
                lens_checksum="sha256:x",
                fit_scale=100,
                usable=True,
            )
        ],
        layer_range=(32, 40),
        required_scale=BAND_SCALE,
    )
    row = next(row for row in inventory["rows"] if row["layer"] == 35)
    assert row["usable"] is False
    assert "scale 250" in row["reason"]


# --------------------------------------------------------------------- 11


def test_interrupted_fitting_resumes_from_the_last_atomic_checkpoint(tmp_path):
    """Requirement 11. The resumed accumulator equals an uninterrupted one."""
    from jlens.calibration.corpus import build_records
    from jlens.calibration.fitting import filter_records_by_tokens, run_calibration
    from jlens.calibration.mock import MockCalibrationModel, mock_corpus_texts
    from jlens.calibration.state import CalibrationFingerprint
    from jlens.mmpilot.band_lens import BandLensStore

    model = MockCalibrationModel()
    plan = band_capture_plan(
        layers=BAND_INTERIOR_LAYERS,
        d_model=model.d_model,
        n_layers=model.n_layers,
        max_seq_len=48,
        skip_first=4,
    )
    records, _ = filter_records_by_tokens(
        build_records("mock/train", mock_corpus_texts(40), min_chars=100),
        token_count=model.tokenizer.token_count,
        skip_first=4,
        max_seq_len=48,
    )

    def store_at(root):
        fingerprint = CalibrationFingerprint(
            mode="mock",
            protocol_version="band-test",
            model_repo_id="mock",
            model_revision="mock",
            tokenizer_revision="mock",
            capture_plan_digest=plan.digest,
            corpus_manifest_checksum="sha256:corpus",
            gate_digest="sha256:gate",
            plateau_rule_digest="n/a",
            scale_points=(12,),
            artifact_format_version="jlens.calibration.artifact.v1",
        )
        store = BandLensStore(root, fingerprint)
        store.open()
        return store

    whole = store_at(tmp_path / "whole")
    reference = run_calibration(
        model, records, plan=plan, scale_points=(12,), store=whole,
        checkpoint_every=4, diagnostics_every=4,
    )

    # An interruption: the session dies after eight prompts, with the atomic
    # checkpoint holding what it had.
    interrupted = store_at(tmp_path / "interrupted")
    run_calibration(
        model, records[:8], plan=plan, scale_points=(8,), store=interrupted,
        checkpoint_every=4, diagnostics_every=4,
    )
    state = torch.load(
        str(interrupted.checkpoint_path), map_location="cpu", weights_only=True
    )
    assert int(state["n_done"]) == 8
    assert sorted(state["jacobian_sum"]) == sorted(BAND_INTERIOR_LAYERS)

    resumed = run_calibration(
        model, records, plan=plan, scale_points=(12,), store=interrupted,
        checkpoint_every=4, diagnostics_every=4,
    )
    assert resumed.n_done == 12
    left = reference.lens_for_scale(12)
    right = resumed.lens_for_scale(12)
    for layer in BAND_INTERIOR_LAYERS:
        assert torch.equal(left.jacobians[layer], right.jacobians[layer]), layer


def test_a_changed_scientific_configuration_refuses_the_fit_directory(tmp_path):
    """A different layer grid is a different fit, and lands elsewhere."""
    from jlens.calibration.state import CalibrationFingerprint
    from jlens.mmpilot.band_lens import BandLensStore
    from jlens.mmpilot.store import IncompatibleStateError

    def fingerprint(digest):
        return CalibrationFingerprint(
            mode="mock",
            protocol_version="band-test",
            model_repo_id="mock",
            model_revision="mock",
            tokenizer_revision="mock",
            capture_plan_digest=digest,
            corpus_manifest_checksum="sha256:corpus",
            gate_digest="sha256:gate",
            plateau_rule_digest="n/a",
            scale_points=(12,),
            artifact_format_version="jlens.calibration.artifact.v1",
        )

    BandLensStore(tmp_path / "run", fingerprint("sha256:a")).open()
    with pytest.raises(IncompatibleStateError):
        BandLensStore(tmp_path / "run", fingerprint("sha256:b")).open()


# --------------------------------------------------------------------- 12


def test_interrupted_interventions_reuse_completed_units(tmp_path, backend, tokens):
    """Requirement 12. A second pass recomputes only what is missing."""
    fingerprint = RunFingerprint(
        mode="anthropic_contiguous_band_coordinate_swap",
        model_repo_id="mock",
        model_revision="mock",
        processor_revision="mock",
        layers=FULL_BAND,
        lens_checksum="sha256:band",
        manifest_checksum="sha256:manifest",
        split_id="band-test",
        intervention_config={"family": BAND_INTERVENTION_FAMILY},
    )
    store = UnitStore(tmp_path / "swap", fingerprint)
    store.open()
    inputs = _property_inputs(backend)
    candidate_ids = _candidate_ids(backend, ("two", "four"))
    from jlens.mmpilot.capability import score_candidate_sequences

    clean = score_candidate_sequences(backend, inputs, candidate_ids)

    def sweep(bands):
        computed = reused = 0
        for band in bands:
            bases = mock_bases(
                backend.world, layers=band, source=tokens["bird"], target=tokens["cat"]
            )
            key = safe_key("band-swap", "g0", "image", band_key(band), "intermediate")
            if store.has("intervention", key):
                reused += 1
                continue
            result = run_swap_condition(
                backend, inputs, bases=bases, alpha=1.0,
                candidate_ids=candidate_ids, target_concept="four",
                clean_scores=clean,
            )
            store.save("intervention", key, band_trial_record(
                result, band=band, arm="intermediate", condition="swap_alpha1",
                modality="image", source="bird", target="cat",
                source_answer="two", target_answer="four", readout="property",
                group_id="g0", image_id="i0",
            ))
            computed += 1
        return computed, reused

    grid = mock_band_grid()
    assert sweep(grid[:2]) == (2, 0)
    assert sweep(grid) == (len(grid) - 2, 2)
    assert sweep(grid) == (0, len(grid))

    # A torn unit is treated as missing rather than trusted.
    path = store.unit_path(
        "intervention",
        safe_key("band-swap", "g0", "image", band_key(grid[0]), "intermediate"),
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["payload"]["target_top1"] = "tampered"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert sweep(grid) == (1, len(grid) - 1)


# --------------------------------------------------------------------- 13


def test_alpha2_is_compared_against_alpha2_matched_controls():
    """Requirement 13."""
    assert controls_for_condition("swap_alpha1") == (
        "zero", "random_alpha1", "unrelated_alpha1"
    )
    assert controls_for_condition("swap_alpha2") == (
        "zero", "random_alpha2", "unrelated_alpha2"
    )
    for condition in ("random_alpha2", "unrelated_alpha2", "swap_alpha2"):
        assert CONDITION_ALPHA[condition] == 2.0
    for condition in ("random_alpha1", "unrelated_alpha1", "swap_alpha1"):
        assert CONDITION_ALPHA[condition] == 1.0
    assert CONDITION_ALPHA["zero"] == 0.0
    assert set(BAND_CONDITIONS) == set(CONDITION_ALPHA)
    with pytest.raises(BandDesignRefused):
        controls_for_condition("zero")


def _cell(**overrides):
    payload = {
        "band_key": "32-40",
        "band_start": 32,
        "arm": "intermediate",
        "condition": "swap_alpha1",
        "alpha": 1.0,
        "modality": "text",
        "source": "bird",
        "target": "cat",
        "readout": "property",
        "n_images": 8,
        "image_ids": [f"i{index}" for index in range(8)],
        "target_top1_rate": 1.0,
        "target_top1_image_ids": [f"i{index}" for index in range(8)],
        "target_top1_bootstrap": {"n": 8, "mean": 1.0, "lower": 1.0, "upper": 1.0},
        "mean_target_rank": 1.0,
        "mean_target_logprob": -1.0,
        "mean_target_margin": 1.0,
        "mean_target_margin_change": 1.0,
        "partial_movement_rate": 0.0,
        "clean_source_top1_rate": 1.0,
    }
    payload.update(overrides)
    return payload


def _cells_for(bands, *, arm_rates, condition="swap_alpha1", identity_rate=0.0):
    cells = []
    for band in bands:
        for arm, rate in arm_rates.items():
            value = rate[band] if isinstance(rate, dict) else rate
            cells.append(_cell(
                band_key=band, band_start=int(band.split("-")[0]), arm=arm,
                condition=condition, target_top1_rate=value,
                mean_target_margin_change=1.0 if value else 0.0,
            ))
            cells.append(_cell(
                band_key=band, band_start=int(band.split("-")[0]), arm=arm,
                condition=condition, readout="identity",
                target_top1_rate=identity_rate,
            ))
            for control in ("zero", "random_alpha1", "unrelated_alpha1"):
                for readout in ("property", "identity"):
                    cells.append(_cell(
                        band_key=band, band_start=int(band.split("-")[0]), arm=arm,
                        condition=control, readout=readout,
                        alpha=CONDITION_ALPHA[control], target_top1_rate=0.0,
                        mean_target_margin_change=0.0,
                    ))
    return cells


# ----------------------------------------------------------------- 14, 15


def test_identity_success_alone_cannot_produce_a_reasoning_go():
    """Requirement 14. The endpoint is the downstream answer, not the identity."""
    bands = ["32-40"]
    cells = _cells_for(
        bands, arm_rates={"intermediate": 0.0, "answer": 0.0}, identity_rate=1.0
    )
    verdict = band_reasoning_verdict(
        cells,
        bands=bands,
        directed_pairs=[{"source": "bird", "target": "cat"}],
        modalities=("text",),
    )
    assert verdict["verdict"] == "BAND_SWAP_NO_GO"
    assert verdict["paper_comparable"]["passed"] is False
    assert verdict["identity_success_alone_is_never_a_reasoning_go"] is True
    cell = next(
        row
        for row in verdict["direction_cells"]
        if row["arm"] == "intermediate" and row["condition"] == "swap_alpha1"
    )
    assert cell["identity_diagnostic"]["identity_top1_rate"] == 1.0
    assert "identity" not in " ".join(
        name for name in cell["clauses"] if name.startswith("target")
    )
    assert cell["passed"] is False


def test_positive_margin_without_top1_is_partial_movement_not_success():
    """Requirement 15."""
    records = []
    for index in range(4):
        records.append({
            "band_key": "32-40", "band": [32, 40], "band_start": 32,
            "arm": "intermediate", "condition": "swap_alpha1", "alpha": 1.0,
            "modality": "text", "source": "bird", "target": "cat",
            "source_answer": "two", "target_answer": "four", "readout": "property",
            "image_id": f"i{index}", "prediction": "two", "clean_prediction": "two",
            "target_rank": 2, "target_score": -2.0, "target_margin": -0.5,
            "target_margin_change": 0.75,
        })
    cell = summarize_band_cells(records)[0]
    assert cell["target_top1_rate"] == 0.0
    assert cell["partial_movement_rate"] == 1.0
    assert cell["partial_movement_label"] == PARTIAL_MOVEMENT
    assert cell["mean_target_margin_change"] == pytest.approx(0.75)

    verdict = band_reasoning_verdict(
        [cell, *_cells_for(["32-40"], arm_rates={"answer": 0.0})],
        bands=["32-40"],
        directed_pairs=[{"source": "bird", "target": "cat"}],
        modalities=("text",),
    )
    assert verdict["verdict"] == "BAND_SWAP_NO_GO"
    assert verdict["margin_only_counts_as"] == PARTIAL_MOVEMENT


def test_pseudoreplication_is_refused():
    """Two trials on one photograph are not two independent observations."""
    row = {
        "band_key": "32-40", "band": [32, 40], "band_start": 32,
        "arm": "intermediate", "condition": "swap_alpha1", "alpha": 1.0,
        "modality": "text", "source": "bird", "target": "cat",
        "source_answer": "two", "target_answer": "four", "readout": "property",
        "image_id": "i0", "prediction": "four", "clean_prediction": "two",
        "target_rank": 1, "target_score": -1.0, "target_margin": 1.0,
        "target_margin_change": 1.0,
    }
    with pytest.raises(BandDesignRefused, match="pseudoreplicate"):
        summarize_band_cells([row, dict(row)])


def test_the_bootstrap_is_deterministic_and_image_level():
    """An interval must be reproducible from the stored cell, on any machine.

    The values are pinned rather than merely re-derived: the whole reason the
    resampler is a seeded LCG instead of numpy is that a reader with the cells
    and this code gets the same interval back, so a change to the resampler has
    to be a visible decision.
    """
    observations = [1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    left = bootstrap_interval(observations, label="a")
    assert left == bootstrap_interval(observations, label="a")
    assert left["n"] == len(observations), "the unit of resampling is the image"
    assert left["mean"] == pytest.approx(0.5)
    assert (left["lower"], left["upper"]) == (0.125, 0.875)
    assert left["iterations"] == 2000
    assert bootstrap_interval([], label="a")["n"] == 0
    assert bootstrap_interval([], label="a")["lower"] is None


# --------------------------------------------------------------------- 16


def test_mock_band_shows_the_intermediate_consumed_before_the_answer():
    """Requirement 16, measured on the synthetic stack rather than asserted.

    The MOCK's reasoning layer writes the legs answer from whichever identity
    coordinates reach it, so a band that begins after it can still exchange the
    *answer* coordinates but can no longer change the answer by exchanging the
    *identity*. Both arms run over the identical bands with identical controls.
    """
    backend = SwapMockBackend()
    records = run_mock_band_trials(backend, modalities=("text",), n_images=4)
    cells = summarize_band_cells(records)
    bands = [band_key(band) for band in mock_band_grid()]
    pairs = [{"source": "bird", "target": "cat"}]
    verdict = band_reasoning_verdict(
        cells, bands=bands, directed_pairs=pairs, modalities=("text",)
    )
    timing = band_onset_timing(
        verdict, bands=bands, directed_pairs=pairs, modalities=("text",)
    )
    row = timing["pairs"][0]
    assert row["classification"] == "INTERMEDIATE_CONSUMED_BEFORE_THE_ANSWER"
    assert timing["verdict"] == "BAND_ONSET_INTERMEDIATE_EARLIER"
    assert (
        row["deepest_effective_start"]["intermediate"]
        < row["deepest_effective_start"]["answer"]
    )
    # Both statistics are reported, and the classification says which it used.
    assert set(row["earliest_effective_start"]) == {"intermediate", "answer"}
    assert timing["classified_on"] == "deepest_effective_start"
    assert timing["band_starts_are_not_exact_physical_onsets"] is True
    assert timing["native_direct_readout_convergence_gate_used"] is False


def test_the_mock_band_window_needs_every_interior_layer():
    """The mock's own confirmed set has a hole at layer 4, and a band over it
    is refused for exactly the reason the real 32-40 band is."""
    assert 4 not in MOCK_VALIDATED_LAYERS
    with pytest.raises(LayerBandError, match=r"\[4\]"):
        build_band(2, 7, usable_layers=MOCK_VALIDATED_LAYERS)
    assert build_band(2, 7, usable_layers=MOCK_BAND_USABLE_LAYERS).layers == FULL_BAND


def test_band_trial_record_refuses_a_trial_that_did_not_clamp_the_whole_band():
    """The stored record is what the aggregation trusts, so it checks."""
    result = {
        "alpha": 1.0, "alpha_is_extrapolation": False,
        "layers_patched": [32, 33],
        "candidate_scores": {
            "two": {"sum_logprob": -2.0}, "four": {"sum_logprob": -1.0},
        },
        "prediction": "four", "clean_prediction": "two",
        "target_score": -1.0, "clean_target_score": -3.0,
        "target_margin": 1.0, "target_margin_change": 2.0,
        "n_positions_patched": 5, "n_candidate_positions_skipped": 1,
    }
    with pytest.raises(BandDesignRefused, match="hooks fired at"):
        band_trial_record(
            result, band=(32, 33, 34), arm="intermediate", condition="swap_alpha1",
            modality="text", source="bird", target="cat", source_answer="two",
            target_answer="four", readout="property", group_id="g", image_id="i",
        )
    record = band_trial_record(
        result, band=(32, 33), arm="intermediate", condition="swap_alpha1",
        modality="text", source="bird", target="cat", source_answer="two",
        target_answer="four", readout="property", group_id="g", image_id="i",
    )
    assert record["target_rank"] == 1
    assert record["band_key"] == "32-33"
    assert record["intervention_family"] == BAND_INTERVENTION_FAMILY


# --------------------------------------------------------------------- 17


def test_completed_run_directories_are_never_written_to(tmp_path):
    """Requirement 17. Publication refuses a destination inside a completed run,
    and a protected directory is byte-for-byte unchanged after the attempt."""
    completed = tmp_path / "rgext_real_c18f03f06e7b"
    (completed / "artifacts" / "published").mkdir(parents=True)
    (completed / "artifacts" / "published" / "lens.pt").write_bytes(b"frozen")
    (completed / "artifacts" / "early_layer_extension_report.json").write_text(
        json.dumps({"schema": "jlens.calibration.early_layer_extension_report.v1"}),
        encoding="utf-8",
    )
    before = {
        str(path.relative_to(completed)): path.read_bytes()
        for path in sorted(completed.rglob("*"))
        if path.is_file()
    }

    band_run = tmp_path / "bandlens_real_deadbeef"
    band_run.mkdir()
    kwargs = {
        "layer": 33,
        "scale": BAND_SCALE,
        "lens": None,
        "confirmation_verdict": {"passed": True, "layer": 33, "scale": BAND_SCALE},
        "development_verdict": {},
        "vault": None,
        "splits": None,
        "selection": {},
        "equivalence": {},
        "band_run_dir": band_run,
        "protected_run_dirs": (completed,),
    }
    with pytest.raises(PublicationRefused, match="completed-run evidence"):
        publish_band_layer(
            destination=completed / "artifacts" / "published" / "stolen.pt", **kwargs
        )
    with pytest.raises(PublicationRefused, match="outside the band run"):
        publish_band_layer(destination=tmp_path / "elsewhere.pt", **kwargs)
    with pytest.raises(PublicationRefused, match="not a band-interior publication"):
        publish_band_layer(
            **{**kwargs, "layer": 35},
            destination=band_run / "artifacts" / "published" / "L35.pt",
        )
    after = {
        str(path.relative_to(completed)): path.read_bytes()
        for path in sorted(completed.rglob("*"))
        if path.is_file()
    }
    assert after == before

    # And a band run misconfigured to live inside the completed run is refused
    # even though the destination is inside its own run directory.
    nested = completed / "bandlens_real_nested"
    nested.mkdir()
    with pytest.raises(PublicationRefused, match="completed-run evidence"):
        publish_band_layer(
            **{**kwargs, "band_run_dir": nested},
            destination=nested / "artifacts" / "published" / "L33.pt",
        )


def test_the_band_study_never_reads_a_single_layer_run_as_a_band_unit(tmp_path):
    """The families are disjoint, so a v2 unit can never be aggregated here."""
    single_layer = {
        "band_key": "32-32", "band": [32], "band_start": 32, "arm": "intermediate",
        "condition": "swap_alpha1", "alpha": 1.0, "modality": "text",
        "source": "bird", "target": "cat", "source_answer": "two",
        "target_answer": "four", "readout": "property", "image_id": "i0",
        "prediction": "four", "clean_prediction": "two", "target_rank": 1,
        "target_score": -1.0, "target_margin": 1.0, "target_margin_change": 1.0,
    }
    cell = summarize_band_cells([single_layer])[0]
    # A one-layer band is a legitimate band key; what distinguishes the studies
    # is the recorded family, which the v2 units do not carry.
    assert cell["band_key"] == "32-32"
    assert "intervention_family" not in cell
    fingerprint = RunFingerprint(
        mode="anthropic_contiguous_band_coordinate_swap",
        model_repo_id="m", model_revision="r", processor_revision="p",
        layers=(32, 33), lens_checksum="sha256:a", manifest_checksum="sha256:b",
        split_id="s", intervention_config={"family": BAND_INTERVENTION_FAMILY},
    )
    other = RunFingerprint(
        mode="paper_reasoning_coordinate_swap",
        model_repo_id="m", model_revision="r", processor_revision="p",
        layers=(32, 33), lens_checksum="sha256:a", manifest_checksum="sha256:b",
        split_id="s",
        intervention_config={"family": "anthropic_independent_single_layer_coordinate_swap"},
    )
    assert fingerprint.digest != other.digest


# ------------------------------------------------------ design and budget


def test_the_design_record_states_what_a_band_is_and_binds_it():
    _, design = _design()
    assert design["primary_band"]["layers"] == list(range(32, 41))
    assert design["band_keys"] == ["32-40", "35-40", "38-40", "40-40"]
    assert design["every_physical_layer_in_each_band_is_patched"] is True
    assert design["hooks_installed_simultaneously_across_the_band"] is True
    assert design["coordinates_recomputed_per_layer_from_its_own_activation"] is True
    assert design["alpha_swept_per_sample"] is False
    assert design["position_rule"] == "all_prompt_positions"
    assert design["alphas"] == [1.0, 2.0]
    assert "involution" in design["involution_does_not_forbid_a_band"]
    assert design["single_layer_v2_run_is_untouched"] is True


def test_the_band_fit_is_a_fresh_accumulator_not_a_seeded_continuation():
    plan = band_capture_plan()
    assert plan.layers == BAND_INTERIOR_LAYERS
    assert plan.target_layer == 41
    budget = band_fit_budget(plan=plan)
    assert budget["fit"]["n_prompts"] == BAND_SCALE
    assert budget["fit"]["unscaled_minutes"] > budget["fit"]["span_scaled_minutes"]
    assert budget["plan"]["all_layers_in_one_pass"] is True


def test_corpus_equivalence_refuses_an_unestablished_corpus():
    from jlens.mmpilot.band_lens import assert_corpus_equivalence

    with pytest.raises(BandLensRefused, match="corpus-equivalent"):
        assert_corpus_equivalence(
            extension_report={"schema": "something.else"},
            plan=band_capture_plan(),
            model_repo_id="google/gemma-4-E4B-it",
            model_revision="rev",
        )


def test_the_band_scale_is_fixed_before_any_development_number():
    selection = band_scale_selection(equivalence={"equivalence_checksum": "sha256:e"})
    assert selection["selected_scale"] == BAND_SCALE
    assert selection["confirmation_not_consulted"] is True
    assert selection["candidate_scales"] == [BAND_SCALE]


def test_a_partial_interior_pass_reports_the_largest_admissible_sub_band():
    confirmation = {
        layer: {"passed": layer != 36, "layer": layer, "metrics": {}}
        for layer in BAND_INTERIOR_LAYERS
    }
    verdict = band_layer_verdict(
        confirmation,
        scale=BAND_SCALE,
        selection={"selection_checksum": "sha256:s"},
        already_confirmed_layers=(32, 35, 38, 40),
    )
    assert verdict["verdict"] == "BAND_INTERIOR_LENS_NO_GO"
    assert verdict["interior_layers_failing"] == [36]
    assert verdict["full_band_available"] is False
    assert verdict["largest_admissible_contiguous_band"] == [32, 35]
    assert "no causal outcome is consulted" in verdict["sub_band_selected_by"]

    passing = {
        layer: {"passed": True, "layer": layer, "metrics": {}}
        for layer in BAND_INTERIOR_LAYERS
    }
    full = band_layer_verdict(
        passing,
        scale=BAND_SCALE,
        selection={"selection_checksum": "sha256:s"},
        already_confirmed_layers=(32, 35, 38, 40),
    )
    assert full["verdict"] == "BAND_INTERIOR_LENS_GO"
    assert full["full_band_available"] is True
    assert full["largest_admissible_contiguous_band"] == [32, 40]


def _unit(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": "jlens.calibration.unit.v1",
                "unit_checksum": payload_checksum(payload),
                "payload": payload,
            }
        ),
        encoding="utf-8",
    )


def test_the_snapshot_backed_band_layers_are_discovered_not_assumed(tmp_path):
    """L35/L38/L40 have scale-250 lenses inside the extension's own snapshot.

    They were scored on that run's fresh confirmation set and not published,
    because its publication targets were L26 and L32. Discovery reads the path
    and the checksum out of the run's units, re-checksums the file, and refuses
    a layer whose recorded verdict did not pass.
    """
    from jlens.mmpilot.band_lens import discover_extension_scale250_lenses

    run = tmp_path / "rgext_real_c18f03f06e7b"
    snapshot = run / "artifacts" / "lens.ext.scale250.pt"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(b"not a real lens, but a real checksum")
    from jlens.metadata import file_sha256

    checksum = file_sha256(str(snapshot))
    (run / "artifacts" / "early_layer_extension_report.json").write_text(
        json.dumps(
            {
                "schema": "jlens.calibration.early_layer_extension_report.v1",
                "mode": "real",
                "corpus": {
                    "corpus_id": "wikitext-103-raw-v1/train",
                    "revision": "b0860",
                    "fit_prefix_checksums": {"250": "sha256:prefix"},
                },
                "fresh_splits": {
                    "protocol": "extension-fresh-eval-hash-bucket-v1",
                    "checksums": {"confirmation": "sha256:conf"},
                },
            }
        ),
        encoding="utf-8",
    )
    _unit(
        run / "units" / "scale_snapshot" / "scale250.json",
        {"scale": 250, "path": str(snapshot), "checksum": checksum,
         "layers": [8, 14, 20, 26, 32, 35, 38, 40]},
    )
    _unit(
        run / "units" / "confirmation" / "scale250.json",
        {
            "scale": 250,
            "by_layer": {
                "35": {"passed": True, "failed_checks": []},
                "38": {"passed": True, "failed_checks": []},
                "40": {"passed": False, "failed_checks": ["median_midrank"]},
            },
        },
    )

    rows, sources, evidence = discover_extension_scale250_lenses(
        run, snapshot_layers=(35, 38, 40, 41), published_layers=(), scale=250
    )
    by_layer = {row.layer: row for row in rows}
    assert by_layer[35].usable and by_layer[35].lens_checksum == checksum
    assert by_layer[38].usable
    assert by_layer[40].usable is False
    assert "did not pass" in by_layer[40].reason
    assert by_layer[41].usable is False
    assert "not in the scale-250 snapshot" in by_layer[41].reason
    assert set(sources) == {35, 38}
    assert sources[35].layer_key_in_file == 35
    assert evidence["snapshot"]["checksum"] == checksum

    snapshot.write_bytes(b"tampered")
    with pytest.raises(BandLensRefused, match="checksums to"):
        discover_extension_scale250_lenses(
            run, snapshot_layers=(35,), published_layers=(), scale=250
        )


def test_published_band_lenses_are_resolved_through_the_band_report(tmp_path):
    from jlens.metadata import file_sha256
    from jlens.mmpilot.band_lens import (
        BAND_LENS_ARTIFACT_SCHEMA,
        BAND_LENS_REPORT_SCHEMA,
        BAND_PUBLISHED_STATUS,
        discover_published_band_lenses,
    )

    run = tmp_path / "bandlens_real_abc123"
    published = run / "artifacts" / "published"
    published.mkdir(parents=True)
    lens = published / "lens.band.layer33.scale250.validated.pt"
    lens.write_bytes(b"layer 33")
    checksum = file_sha256(str(lens))
    artifact = {
        "schema": BAND_LENS_ARTIFACT_SCHEMA,
        "physical_layer": 33,
        "scale_point": 250,
        "validated": True,
        "publication_status": BAND_PUBLISHED_STATUS,
        "lens_path": str(lens),
        "lens_checksum": checksum,
        "confirmation_failed_checks": [],
    }
    artifact["artifact_checksum"] = payload_checksum(artifact)
    (published / "lens.band.layer33.scale250.validated.band.json").write_text(
        json.dumps(artifact), encoding="utf-8"
    )
    (run / "artifacts" / "band_interior_lens_report.json").write_text(
        json.dumps(
            {
                "schema": BAND_LENS_REPORT_SCHEMA,
                "mode": "real",
                "publication": {
                    "published_layers": [33],
                    "published_checksums": {"33": checksum},
                },
                "fresh_splits": {
                    "protocol": "band-interior-fresh-eval-hash-bucket-v1",
                    "checksums": {"confirmation": "sha256:band-conf"},
                },
                "corpus_equivalence": {
                    "extension_corpus_id": "wikitext-103-raw-v1/train",
                    "extension_corpus_revision": "b0860",
                },
            }
        ),
        encoding="utf-8",
    )

    rows, sources, _ = discover_published_band_lenses(run, layers=(33, 34), scale=250)
    by_layer = {row.layer: row for row in rows}
    assert by_layer[33].usable and by_layer[33].lens_checksum == checksum
    assert by_layer[34].usable is False
    assert "does not publish layer 34" in by_layer[34].reason
    assert sources[33].path == str(lens)

    lens.write_bytes(b"tampered")
    with pytest.raises(BandLensRefused, match="is not the published file"):
        discover_published_band_lenses(run, layers=(33,), scale=250)


def test_a_mock_band_report_is_never_read_as_a_published_source(tmp_path):
    from jlens.mmpilot.band_lens import (
        BAND_LENS_REPORT_SCHEMA,
        discover_published_band_lenses,
    )

    run = tmp_path / "bandlens_mock_abc123"
    (run / "artifacts").mkdir(parents=True)
    (run / "artifacts" / "band_interior_lens_report.json").write_text(
        json.dumps({"schema": BAND_LENS_REPORT_SCHEMA, "mode": "mock"}),
        encoding="utf-8",
    )
    with pytest.raises(BandLensRefused, match="MOCK report publishes nothing"):
        discover_published_band_lenses(run)


def test_the_band_split_generation_must_exclude_the_extension_sets():
    from jlens.mmpilot.band_lens import build_band_evaluation_splits

    with pytest.raises(BandLensRefused, match="extension_development"):
        build_band_evaluation_splits(
            [], excluded={"old_fit": ()}, corpus_id="wikitext"
        )


def test_a_shared_completed_directory_survives_a_full_analysis_pass(tmp_path):
    """Reading the historical artifacts must not touch them."""
    completed = tmp_path / "completed"
    completed.mkdir()
    (completed / "report.json").write_text(json.dumps({"a": 1}), encoding="utf-8")
    before = payload_checksum(
        {
            str(path.relative_to(completed)): path.read_bytes().decode()
            for path in sorted(completed.rglob("*"))
            if path.is_file()
        }
    )
    backend = SwapMockBackend()
    records = run_mock_band_trials(backend, modalities=("text",), n_images=2)
    summarize_band_cells(records)
    shutil.rmtree(tmp_path / "unused", ignore_errors=True)
    after = payload_checksum(
        {
            str(path.relative_to(completed)): path.read_bytes().decode()
            for path in sorted(completed.rglob("*"))
            if path.is_file()
        }
    )
    assert after == before
