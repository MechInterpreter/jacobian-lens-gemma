# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Tests for the generative J-cone steering backend (jlens.generative)."""

from __future__ import annotations

import json

import pytest
import torch

from jlens.gemma4 import Gemma4LensModel
from jlens.generative import (
    NEUTRAL_PROMPTS,
    VECTOR_CONDITIONS,
    GenerativeError,
    SteeringSchedule,
    SteeringSpec,
    build_condition_vector,
    coefficient_mass_indices,
    first_token_distribution,
    greedy_decode,
    kl_from_baseline,
    load_benchmark,
    make_generative_record,
    scale_to_ratio,
    shuffled_coordinates,
    steering_injection,
    target_logprob,
    weighted_reconstruction,
)

from .mock_gemma4 import MockGemma4ForConditionalGeneration, MockTokenizer

D_MODEL = 8


def _model(**kw) -> Gemma4LensModel:
    return Gemma4LensModel(
        MockGemma4ForConditionalGeneration(**kw), MockTokenizer()
    )


# ---------------------------------------------------------------- schedules


def test_schedule_prompt_only_injects_only_at_anchor():
    schedule = SteeringSchedule("prompt_only")
    assert schedule.weights(anchor=4, prompt_len=5, seq_len=9) == {4: 1.0}


def test_schedule_constant_covers_anchor_and_generated_positions():
    schedule = SteeringSchedule("constant")
    weights = schedule.weights(anchor=4, prompt_len=5, seq_len=8)
    assert weights == {4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0}


def test_schedule_decaying_weights():
    schedule = SteeringSchedule("decaying", decay=0.5)
    weights = schedule.weights(anchor=4, prompt_len=5, seq_len=8)
    assert weights[4] == 1.0
    assert weights[5] == pytest.approx(0.5)
    assert weights[6] == pytest.approx(0.25)
    assert weights[7] == pytest.approx(0.125)


def test_schedule_wrong_position_anchor_still_reinjects_generated():
    weights = SteeringSchedule("constant").weights(anchor=1, prompt_len=5, seq_len=7)
    assert weights == {1: 1.0, 5: 1.0, 6: 1.0}


def test_schedule_rejects_bad_kind_and_decay():
    with pytest.raises(GenerativeError):
        SteeringSchedule("linear")
    with pytest.raises(GenerativeError):
        SteeringSchedule("decaying", decay=1.5)
    with pytest.raises(GenerativeError):
        SteeringSchedule("prompt_only").weights(anchor=9, prompt_len=5, seq_len=6)


# --------------------------------------------------------------------- hook


def test_zero_delta_injection_reproduces_baseline_exactly():
    model = _model()
    ids = model.encode("steering parity probe")
    with torch.no_grad():
        baseline = model.unembed(model.forward(ids).last_hidden_state)
    with steering_injection(
        model.layers,
        2,
        delta=torch.zeros(D_MODEL),
        schedule=SteeringSchedule("constant"),
        prompt_len=ids.shape[1],
    ):
        with torch.no_grad():
            hooked = model.unembed(model.forward(ids).last_hidden_state)
    assert torch.equal(baseline, hooked)


def test_injection_changes_only_scheduled_positions():
    model = _model()
    ids = model.encode("position scope probe")
    prompt_len = ids.shape[1]
    delta = torch.full((D_MODEL,), 3.0)
    with torch.no_grad():
        baseline = model.forward(ids).last_hidden_state
    with steering_injection(
        model.layers,
        model.n_layers - 1,  # last block: no downstream mixing in the mock
        delta=delta,
        schedule=SteeringSchedule("prompt_only"),
        prompt_len=prompt_len,
    ):
        with torch.no_grad():
            steered = model.forward(ids).last_hidden_state
    diff = (steered - baseline).abs().sum(dim=-1)[0]
    changed = (diff > 0).nonzero().flatten().tolist()
    assert changed == [prompt_len - 1]


def test_hook_removed_after_exception():
    model = _model()
    ids = model.encode("hi")
    n_hooks_before = len(model.layers[1]._forward_hooks)
    with pytest.raises(RuntimeError):
        with steering_injection(
            model.layers,
            1,
            delta=torch.zeros(D_MODEL),
            schedule=SteeringSchedule("prompt_only"),
            prompt_len=ids.shape[1],
        ):
            raise RuntimeError("boom")
    assert len(model.layers[1]._forward_hooks) == n_hooks_before
    with torch.no_grad():
        model.forward(ids)  # still healthy


def test_hook_stats_record_ratio_and_forward_count():
    model = _model()
    ids = model.encode("stats probe")
    delta = torch.randn(D_MODEL)
    with steering_injection(
        model.layers,
        2,
        delta=delta,
        schedule=SteeringSchedule("prompt_only"),
        prompt_len=ids.shape[1],
    ) as stats:
        with torch.no_grad():
            model.forward(ids)
            model.forward(ids)
    assert stats["n_forward_passes"] == 2
    assert stats["anchor_activation_norm"] > 0
    assert stats["measured_ratio"] == pytest.approx(
        stats["delta_norm"] / stats["anchor_activation_norm"]
    )


def test_injection_validates_layer_and_delta():
    model = _model()
    with pytest.raises(GenerativeError):
        with steering_injection(
            model.layers,
            99,
            delta=torch.zeros(D_MODEL),
            schedule=SteeringSchedule("prompt_only"),
            prompt_len=3,
        ):
            pass
    with pytest.raises(GenerativeError):
        with steering_injection(
            model.layers,
            1,
            delta=torch.full((D_MODEL,), float("nan")),
            schedule=SteeringSchedule("prompt_only"),
            prompt_len=3,
        ):
            pass


# ------------------------------------------------------------------- decode


def _reference_greedy(model, ids: torch.Tensor, steps: int) -> list[int]:
    """Independent uncached greedy loop (no jlens.generative code paths)."""
    out = []
    for _ in range(steps):
        with torch.no_grad():
            hidden = model.forward(ids).last_hidden_state
            logits = model.unembed(hidden)[0, -1].float()
        next_id = int(torch.log_softmax(logits, dim=-1).argmax())
        out.append(next_id)
        ids = torch.cat([ids, torch.tensor([[next_id]])], dim=1)
    return out


def test_unsteered_decode_matches_reference_greedy():
    model = _model()
    ids = model.encode("decode equivalence")
    result = greedy_decode(model, ids, max_new_tokens=6)
    assert result.token_ids == _reference_greedy(model, ids, 6)
    assert result.n_steps == 6
    assert result.stop_reason == "max_new_tokens"


def test_zero_steering_decode_matches_unsteered():
    model = _model()
    ids = model.encode("zero steering decode")
    spec = SteeringSpec(
        layer=2,
        delta=torch.zeros(D_MODEL),
        schedule=SteeringSchedule("constant"),
    )
    steered = greedy_decode(model, ids, max_new_tokens=5, steering=spec)
    plain = greedy_decode(model, ids, max_new_tokens=5)
    assert steered.token_ids == plain.token_ids
    assert steered.chosen_logprobs == pytest.approx(plain.chosen_logprobs)


def test_decode_stops_at_eos():
    model = _model()
    ids = model.encode("eos stop")
    unforced = greedy_decode(model, ids, max_new_tokens=4)
    eos = unforced.token_ids[0]
    result = greedy_decode(model, ids, max_new_tokens=4, eos_token_ids=[eos])
    assert result.stop_reason == "eos"
    assert result.token_ids == [eos]


def test_decode_rejects_bad_inputs():
    model = _model()
    ids = model.encode("x")
    with pytest.raises(GenerativeError):
        greedy_decode(model, ids, max_new_tokens=0)
    with pytest.raises(GenerativeError):
        greedy_decode(model, ids[0], max_new_tokens=2)


# ------------------------------------------------------------------ scoring


def test_target_logprob_matches_stepwise_forced_decoding():
    model = _model()
    ids = model.encode("teacher forcing")
    targets = [5, 9, 11]
    spec = SteeringSpec(
        layer=2, delta=torch.randn(D_MODEL) * 0.3, schedule=SteeringSchedule("constant")
    )
    scored = target_logprob(model, ids, targets, steering=spec)

    # Step-by-step: force each target token, score it from the previous
    # sequence under the same schedule-weighted injection.
    stepwise = []
    seq = ids
    for token in targets:
        with steering_injection(
            model.layers,
            spec.layer,
            delta=spec.delta,
            schedule=spec.schedule,
            prompt_len=ids.shape[1],
        ):
            with torch.no_grad():
                hidden = model.forward(seq).last_hidden_state
                logits = model.unembed(hidden)[0, -1].float()
        stepwise.append(float(torch.log_softmax(logits, dim=-1)[token]))
        seq = torch.cat([seq, torch.tensor([[token]])], dim=1)
    assert scored["per_token_logprobs"] == pytest.approx(stepwise, abs=1e-5)
    assert scored["total_logprob"] == pytest.approx(sum(stepwise), abs=1e-4)


def test_target_logprob_first_token_rank():
    model = _model()
    ids = model.encode("rank probe")
    log_p = first_token_distribution(model, ids)
    best = int(log_p.argmax())
    scored = target_logprob(model, ids, [best])
    assert scored["first_token_rank"] == 0
    assert scored["first_token_logprob"] == pytest.approx(float(log_p[best]))


def test_target_logprob_rejects_empty_target():
    model = _model()
    with pytest.raises(GenerativeError):
        target_logprob(model, model.encode("x"), [])


def test_kl_from_baseline_zero_for_identical_and_positive_otherwise():
    model = _model()
    ids = model.encode("kl probe")
    base = first_token_distribution(model, ids)
    assert kl_from_baseline(base, base) == pytest.approx(0.0, abs=1e-9)
    spec = SteeringSpec(
        layer=2, delta=torch.randn(D_MODEL), schedule=SteeringSchedule("prompt_only")
    )
    steered = first_token_distribution(model, ids, steering=spec)
    assert kl_from_baseline(steered, base) > 0


# --------------------------------------------------------------- conditions


def test_coefficient_mass_indices_thresholds():
    coeffs = [5.0, 3.0, 2.0]
    assert coefficient_mass_indices(coeffs, 0.5) == [0]
    assert coefficient_mass_indices(coeffs, 0.6) == [0, 1]
    assert coefficient_mass_indices(coeffs, 0.8) == [0, 1]
    assert coefficient_mass_indices(coeffs, 1.0) == [0, 1, 2]
    # Order in the input does not matter; indices refer to input positions.
    assert coefficient_mass_indices([2.0, 5.0, 3.0], 0.6) == [1, 2]


def test_coefficient_mass_indices_rejects_bad_inputs():
    with pytest.raises(GenerativeError):
        coefficient_mass_indices([1.0], 0.0)
    with pytest.raises(GenerativeError):
        coefficient_mass_indices([0.0, 0.0], 0.5)


def test_weighted_reconstruction_is_coefficient_weighted_sum():
    atoms = torch.eye(4)
    q = weighted_reconstruction(atoms, [1, 3], [2.0, 0.5])
    assert torch.allclose(q, torch.tensor([0.0, 2.0, 0.0, 0.5]))
    sub = weighted_reconstruction(atoms, [1, 3], [2.0, 0.5], subset=[1])
    assert torch.allclose(sub, torch.tensor([0.0, 0.0, 0.0, 0.5]))


def test_weighted_reconstruction_rejects_bad_inputs():
    atoms = torch.eye(4)
    with pytest.raises(GenerativeError):
        weighted_reconstruction(atoms, [1], [2.0, 3.0])
    with pytest.raises(GenerativeError):
        weighted_reconstruction(atoms, [1], [-1.0])
    with pytest.raises(GenerativeError):
        weighted_reconstruction(atoms, [1, 2], [1.0, 1.0], subset=[5])
    with pytest.raises(GenerativeError):
        weighted_reconstruction(atoms, [], [])


def test_shuffled_coordinates_preserve_norm_and_multiset():
    delta = torch.arange(16, dtype=torch.float32)
    shuffled = shuffled_coordinates(delta, seed=7)
    assert not torch.equal(shuffled, delta)
    assert float(shuffled.norm()) == pytest.approx(float(delta.norm()))
    assert sorted(shuffled.tolist()) == sorted(delta.tolist())
    assert torch.equal(shuffled, shuffled_coordinates(delta, seed=7))


def test_scale_to_ratio():
    delta = torch.tensor([3.0, 4.0])  # norm 5
    scaled, info = scale_to_ratio(delta, activation_norm=100.0, ratio=0.1)
    assert float(scaled.norm()) == pytest.approx(10.0)
    assert info["requested_ratio"] == 0.1
    with pytest.raises(GenerativeError):
        scale_to_ratio(torch.zeros(4), activation_norm=1.0, ratio=0.1)
    with pytest.raises(GenerativeError):
        scale_to_ratio(delta, activation_norm=0.0, ratio=0.1)
    with pytest.raises(GenerativeError):
        scale_to_ratio(delta, activation_norm=1.0, ratio=-1.0)


def test_build_condition_vector_core_conditions():
    atoms = torch.eye(6)
    ids = [1, 2, 4]
    coeffs = [3.0, 2.0, 1.0]
    full = build_condition_vector(
        "full_cone", atoms=atoms, token_ids=ids, coefficients=coeffs
    )
    expected = torch.zeros(6)
    expected[1], expected[2], expected[4] = 3.0, 2.0, 1.0
    assert torch.allclose(full.delta, expected)

    reversed_ = build_condition_vector(
        "sign_reversed", atoms=atoms, token_ids=ids, coefficients=coeffs
    )
    assert torch.allclose(reversed_.delta, -expected)

    mass = build_condition_vector(
        "mass_subcone",
        atoms=atoms,
        token_ids=ids,
        coefficients=coeffs,
        mass_threshold=0.8,
    )
    assert mass.meta["subset_indices"] == [0, 1]
    assert torch.allclose(mass.delta, torch.tensor([0, 3.0, 2.0, 0, 0, 0]))

    manual = build_condition_vector(
        "manual_subcone",
        atoms=atoms,
        token_ids=ids,
        coefficients=coeffs,
        manual_indices=[2],
    )
    assert torch.allclose(manual.delta, torch.tensor([0, 0, 0, 0, 1.0, 0]))

    zero = build_condition_vector("zero", d_model=6)
    assert torch.equal(zero.delta, torch.zeros(6))
    assert build_condition_vector("none").delta is None

    random = build_condition_vector(
        "random_matched_norm", d_model=6, match_norm=2.5, seed=11
    )
    assert float(random.delta.norm()) == pytest.approx(2.5)

    shuffled = build_condition_vector(
        "shuffled", atoms=atoms, token_ids=ids, coefficients=coeffs, seed=3
    )
    assert float(shuffled.delta.norm()) == pytest.approx(float(expected.norm()))

    unrelated = build_condition_vector(
        "unrelated_cone",
        atoms=atoms,
        unrelated_token_ids=[0, 5],
        unrelated_coefficients=[1.0, 1.0],
    )
    assert torch.allclose(unrelated.delta, torch.tensor([1.0, 0, 0, 0, 0, 1.0]))

    h = torch.randn(6)
    c = torch.randn(6)
    raw = build_condition_vector("raw_activation", raw_activation=h)
    assert torch.allclose(raw.delta, h)
    diff = build_condition_vector(
        "activation_diff", raw_activation=h, control_activation=c
    )
    assert torch.allclose(diff.delta, h - c)

    # wrong_layer / wrong_position reuse the correct full-cone vector.
    for site_condition in ("wrong_layer", "wrong_position"):
        built = build_condition_vector(
            site_condition, atoms=atoms, token_ids=ids, coefficients=coeffs
        )
        assert torch.allclose(built.delta, expected)


def test_build_condition_vector_fails_loudly():
    with pytest.raises(GenerativeError):
        build_condition_vector("nonexistent_condition")
    with pytest.raises(GenerativeError):
        build_condition_vector("full_cone")  # missing everything
    with pytest.raises(GenerativeError):
        build_condition_vector("zero")  # missing d_model
    with pytest.raises(GenerativeError):
        build_condition_vector(
            "activation_diff",
            raw_activation=torch.randn(4),
            control_activation=torch.randn(5),
        )
    same = torch.randn(4)
    with pytest.raises(GenerativeError):
        build_condition_vector(
            "activation_diff", raw_activation=same, control_activation=same
        )


def test_every_declared_condition_is_buildable_or_rejects_missing_inputs():
    # No condition silently returns a placeholder: each either builds from
    # complete ingredients (covered above) or raises on incomplete ones.
    for condition in VECTOR_CONDITIONS:
        if condition == "none":
            continue
        with pytest.raises(GenerativeError):
            build_condition_vector(condition)


# ---------------------------------------------------------------- benchmark


def _write_manifest(tmp_path, manifest) -> str:
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return str(path)


def _example(example_id="ex1", **overrides):
    example = {
        "example_id": example_id,
        "category": "compound_word",
        "source_prompt": "A house for birds is called a",
        "target_phrase": " birdhouse",
        "extraction_position": -1,
    }
    example.update(overrides)
    return example


def test_load_benchmark_accepts_valid_manifest(tmp_path):
    path = _write_manifest(
        tmp_path,
        {"version": 1, "dev": [_example()], "heldout": [_example("ex2")]},
    )
    manifest = load_benchmark(path)
    assert manifest["dev"][0]["example_id"] == "ex1"


def test_load_benchmark_rejects_bad_manifests(tmp_path):
    with pytest.raises(GenerativeError):
        load_benchmark(_write_manifest(tmp_path, {"version": 1, "dev": []}))
    with pytest.raises(GenerativeError):
        load_benchmark(
            _write_manifest(
                tmp_path,
                {"dev": [_example()], "heldout": [_example()]},  # duplicate id
            )
        )
    bad_category = _example(category="verb_phrase")
    with pytest.raises(GenerativeError):
        load_benchmark(
            _write_manifest(tmp_path, {"dev": [bad_category], "heldout": []})
        )
    missing_key = _example()
    del missing_key["target_phrase"]
    with pytest.raises(GenerativeError):
        load_benchmark(
            _write_manifest(tmp_path, {"dev": [missing_key], "heldout": []})
        )


def test_neutral_prompts_contain_no_concept_clues():
    for prompt in NEUTRAL_PROMPTS.values():
        assert "photosynthesis" not in prompt.lower()
        assert prompt.strip()


# ------------------------------------------------------------------ records


def test_make_generative_record_is_json_safe():
    schedule = SteeringSchedule("decaying", decay=0.7)
    record = make_generative_record(
        run_id="run1",
        example_id="ex1",
        condition="full_cone",
        source_layer=21,
        injection_layer=21,
        source_position=-1,
        injection_anchor=11,
        schedule=schedule,
        neutral_prompt_id="label-colon",
        strength_ratio=0.5,
        vector_meta={"n_generators": 3},
        hook_stats={
            "measured_ratio": 0.49,
            "delta_norm": 2.0,
            "anchor_activation_norm": 4.1,
        },
        scoring={"total_logprob": -3.2, "per_token_logprobs": [-1.6, -1.6]},
        decode=None,
        delta_vs_zero=1.5,
        delta_vs_unrelated=0.9,
        kl_divergence=0.02,
        target_phrase=" birdhouse",
        target_recovered_exact=False,
        target_recovered_substring=True,
        seed=1234,
        provenance={"commit": "abc"},
    )
    assert record["schema"] == "jlens.generative.record.v1"
    assert record["steering_schedule"] == {"kind": "decaying", "decay": 0.7}
    assert record["total_logprob"] == -3.2
    json.dumps(record)  # must serialize


# ------------------------------------------------------- shipped artifacts


def test_shipped_benchmark_manifest_is_valid():
    manifest = load_benchmark("configs/generative_benchmark.json")
    assert len(manifest["dev"]) >= 6
    assert len(manifest["heldout"]) >= 6
    categories = {e["category"] for e in manifest["dev"] + manifest["heldout"]}
    assert categories == {
        "split_word",
        "compound_word",
        "named_entity",
        "noun_phrase",
    }
    for example in manifest["dev"] + manifest["heldout"]:
        # Multi-token concepts: at least two words or a long single word.
        phrase = example["target_phrase"].strip()
        assert len(phrase.split()) >= 2 or len(phrase) >= 9
        assert example["target_phrase"].startswith(" ")


def test_shipped_generative_config_is_valid():
    pytest.importorskip("yaml")
    from jlens.metadata import load_generative_config

    config = load_generative_config("configs/gemma_generative_validation.yaml")
    assert config["mode"] == "generative_validation"
    assert set(config["steering"]["layers"]) <= set(
        config["lens"]["expect_source_layers"]
    )
    assert config["model"]["allow_model_load"] is False


def test_generative_config_validation_fails_loudly():
    pytest.importorskip("yaml")
    from jlens.metadata import load_generative_config, validate_generative_config

    config = load_generative_config("configs/gemma_generative_validation.yaml")
    import copy

    bad = copy.deepcopy(config)
    bad["steering"]["conditions"].remove("zero")
    with pytest.raises(ValueError, match="zero"):
        validate_generative_config(bad)

    bad = copy.deepcopy(config)
    bad["steering"]["layers"] = [9]
    with pytest.raises(ValueError, match="steering.layers"):
        validate_generative_config(bad)

    bad = copy.deepcopy(config)
    bad["benchmark"]["split"] = "test"
    with pytest.raises(ValueError, match="split"):
        validate_generative_config(bad)

    bad = copy.deepcopy(config)
    bad["steering"]["schedules"].append({"kind": "linear"})
    with pytest.raises(ValueError, match="schedules"):
        validate_generative_config(bad)


# -------------------------------------------------------------- aggregation


def _record(
    example_id="ex1",
    condition="full_cone",
    layer=21,
    ratio=0.5,
    schedule="prompt_only",
    prompt_id="label-colon",
    total=-4.0,
    delta_zero=None,
    exact=None,
    substring=None,
):
    return {
        "vector_condition": condition,
        "example_id": example_id,
        "source_layer": layer,
        "requested_ratio": ratio,
        "steering_schedule": {"kind": schedule, "decay": None},
        "neutral_prompt_id": prompt_id,
        "total_logprob": total,
        "delta_logprob_vs_zero": delta_zero,
        "delta_logprob_vs_unrelated": None,
        "kl_divergence_from_baseline": 0.1,
        "target_recovered_exact": exact,
        "target_recovered_substring": substring,
    }


def test_summarize_by_condition():
    from jlens.generative import summarize_by_condition

    records = [
        _record(delta_zero=2.0, exact=True, substring=True),
        _record(delta_zero=1.0, exact=False, substring=False),
        _record(condition="zero", total=-6.0),
    ]
    summary = {s["vector_condition"]: s for s in summarize_by_condition(records)}
    assert summary["full_cone"]["mean_delta_vs_zero"] == pytest.approx(1.5)
    assert summary["full_cone"]["exact_recovery_rate"] == pytest.approx(0.5)
    assert summary["zero"]["n_records"] == 1
    assert summary["zero"]["mean_delta_vs_zero"] is None


def test_choose_calibration_picks_best_operating_point():
    from jlens.generative import GenerativeError, choose_calibration

    records = [
        _record(layer=14, ratio=0.5, delta_zero=0.5),
        _record(layer=21, ratio=1.0, delta_zero=3.0),
        _record(layer=21, ratio=1.0, delta_zero=1.0, prompt_id="other"),
        _record(layer=28, ratio=2.0, delta_zero=-1.0),
    ]
    best = choose_calibration(records)
    assert (best["source_layer"], best["ratio"]) == (21, 1.0)
    assert best["mean_delta_vs_zero"] == pytest.approx(2.0)
    with pytest.raises(GenerativeError):
        choose_calibration([], condition="full_cone")


def test_per_example_verdicts_and_gonogo():
    from jlens.generative import gonogo_report, per_example_verdicts

    def battery(example_id, strong: bool):
        correct_total = -2.0 if strong else -9.0
        rows = [
            _record(
                example_id,
                "full_cone",
                total=correct_total,
                delta_zero=3.0 if strong else -1.0,
                prompt_id=p,
                exact=strong,
                substring=strong,
            )
            for p in ("label-colon", "answer-four-words")
        ]
        for control in (
            "random_matched_norm",
            "shuffled",
            "sign_reversed",
            "unrelated_cone",
            "raw_activation",
        ):
            rows.append(_record(example_id, control, total=-5.0))
        rows.append(_record(example_id, "zero", total=-5.0, ratio=None))
        return rows

    records = battery("ex-strong", True) + battery("ex-weak", False)
    verdicts = per_example_verdicts(
        records, source_layer=21, ratio=0.5, schedule_kind="prompt_only"
    )
    by_id = {v["example_id"]: v for v in verdicts}
    assert by_id["ex-strong"]["beats_all_controls"] is True
    assert by_id["ex-strong"]["survives_multiple_prompts"] is True
    assert by_id["ex-strong"]["recovered_any"] is True
    assert by_id["ex-strong"]["jspace_vs_raw_activation"] == pytest.approx(3.0)
    assert by_id["ex-weak"]["beats_all_controls"] is False
    assert by_id["ex-weak"]["recovered_any"] is False

    report = gonogo_report(verdicts)
    assert report["n_examples"] == 2
    assert report["n_passing_joint_criterion"] == 1
    assert report["go"] is False  # 1 of 2 is not a clear majority

    solo = gonogo_report([by_id["ex-strong"]])
    assert solo["go"] is True
