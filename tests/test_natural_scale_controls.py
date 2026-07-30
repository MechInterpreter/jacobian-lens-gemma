# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Matched natural-scale controls (jlens.generative).

A real GPU smoke run found that the correct J-cone's natural (unscaled)
injection beat zero at low strength, but the existing random/shuffled/
sign-reversed/unrelated controls were only ever evaluated at ratio 1.0 — a
different injected norm than the natural cone, and only under the
``prompt_only`` schedule. That leaves the experiment unable to say whether the
low-strength gain is specific to the correct cone's *direction*, since the
controls were never tested at the same magnitude or across the same
schedules.

This module tests the fix: five new conditions
(``natural_unrelated_cone`` / ``natural_random_matched_norm`` /
``natural_shuffled`` / ``natural_sign_reversed`` / ``natural_mass_subcone``)
that build the exact same *direction* as their ratio-scaled counterpart, then
rescale it to match ``natural_scale``'s own observed norm via
:func:`jlens.generative.scale_to_norm` — plus the aggregation
(:func:`condition_scaling_mode`, :func:`summarize_by_condition`) and go/no-go
(:func:`natural_scale_verdicts`, :func:`natural_scale_gonogo_report`) that keep
this comparison legible and never confuse it with the ratio-scaled battery.

End-to-end wiring (the runner actually matching every control's delta norm to
``natural_scale``'s, across every schedule) is covered by
``test_generative_runner.py``.
"""

from __future__ import annotations

import pytest
import torch

from jlens.generative import (
    GONOGO_CONTROLS,
    NATURAL_SCALE_GONOGO_CONTROLS,
    NATURAL_SCALE_MATCHED_CONDITIONS,
    VECTOR_CONDITIONS,
    GenerativeError,
    build_condition_vector,
    condition_scaling_mode,
    natural_scale_gonogo_report,
    natural_scale_verdicts,
    scale_to_norm,
    summarize_by_condition,
)

ATOMS = torch.eye(6)
IDS = [1, 2, 4]
COEFFS = [3.0, 2.0, 1.0]


def _full_cone_delta() -> torch.Tensor:
    return build_condition_vector(
        "full_cone", atoms=ATOMS, token_ids=IDS, coefficients=COEFFS
    ).delta


# --------------------------------------------------------- condition registry


def test_natural_conditions_are_registered():
    for condition in (
        "natural_unrelated_cone",
        "natural_random_matched_norm",
        "natural_shuffled",
        "natural_sign_reversed",
        "natural_mass_subcone",
    ):
        assert condition in VECTOR_CONDITIONS
        assert condition in NATURAL_SCALE_MATCHED_CONDITIONS
    # natural_scale itself is the reference the matched set is matched *to* —
    # it is not itself one of the matched controls.
    assert "natural_scale" not in NATURAL_SCALE_MATCHED_CONDITIONS
    assert len(NATURAL_SCALE_MATCHED_CONDITIONS) == 5


def test_natural_scale_gonogo_controls_mirror_gonogo_controls():
    """Same four specificity controls as the ratio-scaled go/no-go criterion
    (GONOGO_CONTROLS), natural_-prefixed; mass_subcone is excluded from both
    for the same reason — it's an ablation of the correct cone, not a
    direction control."""
    assert set(NATURAL_SCALE_GONOGO_CONTROLS) == {
        f"natural_{c}" for c in GONOGO_CONTROLS
    }
    assert "natural_mass_subcone" not in NATURAL_SCALE_GONOGO_CONTROLS


# ------------------------------------------------- building the raw direction


def test_natural_sign_reversed_matches_sign_reversed_direction():
    """Same semantics as its ratio-scaled counterpart: negate the full-cone
    reconstruction. Norm is therefore identical *before any scaling step* —
    negation cannot change a vector's norm."""
    base = build_condition_vector(
        "sign_reversed", atoms=ATOMS, token_ids=IDS, coefficients=COEFFS
    )
    natural = build_condition_vector(
        "natural_sign_reversed", atoms=ATOMS, token_ids=IDS, coefficients=COEFFS
    )
    assert torch.equal(natural.delta, base.delta)
    assert float(natural.delta.norm()) == pytest.approx(
        float(_full_cone_delta().norm())
    )


def test_natural_shuffled_matches_shuffled_semantics():
    """Same seeded permutation as its ratio-scaled counterpart, and — like any
    coordinate permutation — norm-preserving before any explicit scaling."""
    base = build_condition_vector(
        "shuffled", atoms=ATOMS, token_ids=IDS, coefficients=COEFFS, seed=7
    )
    natural = build_condition_vector(
        "natural_shuffled", atoms=ATOMS, token_ids=IDS, coefficients=COEFFS, seed=7
    )
    assert torch.equal(natural.delta, base.delta)
    assert float(natural.delta.norm()) == pytest.approx(
        float(_full_cone_delta().norm())
    )
    assert sorted(natural.delta.tolist()) == sorted(_full_cone_delta().tolist())


def test_natural_unrelated_cone_matches_unrelated_cone_direction():
    base = build_condition_vector(
        "unrelated_cone",
        atoms=ATOMS,
        unrelated_token_ids=[0, 5],
        unrelated_coefficients=[1.0, 1.0],
    )
    natural = build_condition_vector(
        "natural_unrelated_cone",
        atoms=ATOMS,
        unrelated_token_ids=[0, 5],
        unrelated_coefficients=[1.0, 1.0],
    )
    assert torch.equal(natural.delta, base.delta)
    assert natural.meta["generator_token_ids"] == base.meta["generator_token_ids"]


def test_natural_mass_subcone_matches_mass_subcone_direction():
    base = build_condition_vector(
        "mass_subcone",
        atoms=ATOMS,
        token_ids=IDS,
        coefficients=COEFFS,
        mass_threshold=0.8,
    )
    natural = build_condition_vector(
        "natural_mass_subcone",
        atoms=ATOMS,
        token_ids=IDS,
        coefficients=COEFFS,
        mass_threshold=0.8,
    )
    assert torch.equal(natural.delta, base.delta)
    assert natural.meta["subset_indices"] == base.meta["subset_indices"]
    # A strict subset of the full cone, so (before scaling) its norm is
    # smaller than the reference — exactly the case scale_to_norm exists for.
    assert float(natural.delta.norm()) < float(_full_cone_delta().norm())


def test_natural_random_matched_norm_uses_unit_norm_raw_direction():
    """Same convention as the ratio-scaled path: build with match_norm=1.0 (a
    unit vector), and let the final scaling step set the real magnitude."""
    built = build_condition_vector(
        "natural_random_matched_norm", d_model=6, match_norm=1.0, seed=11
    )
    assert float(built.delta.norm()) == pytest.approx(1.0)


def test_natural_conditions_reject_missing_inputs():
    for condition in NATURAL_SCALE_MATCHED_CONDITIONS:
        with pytest.raises(GenerativeError):
            build_condition_vector(condition)


# --------------------------------------------------------------- scale_to_norm


def test_scale_to_norm_matches_target_exactly():
    delta = torch.tensor([3.0, 4.0])  # norm 5
    scaled, info = scale_to_norm(delta, target_norm=7.154324531555176)
    assert float(scaled.norm()) == pytest.approx(7.154324531555176, rel=1e-6)
    assert info["raw_delta_norm"] == pytest.approx(5.0)
    assert info["target_norm"] == pytest.approx(7.154324531555176)
    assert info["scale_factor"] == pytest.approx(7.154324531555176 / 5.0)
    assert info["scaled_delta_norm"] == pytest.approx(7.154324531555176, rel=1e-6)


def test_scale_to_norm_rejects_bad_inputs():
    delta = torch.tensor([3.0, 4.0])
    with pytest.raises(GenerativeError):
        scale_to_norm(torch.zeros(4), target_norm=1.0)
    with pytest.raises(GenerativeError):
        scale_to_norm(delta, target_norm=0.0)
    with pytest.raises(GenerativeError):
        scale_to_norm(delta, target_norm=-1.0)
    with pytest.raises(GenerativeError):
        scale_to_norm(delta, target_norm=float("inf"))


def test_scale_to_norm_matches_sign_reversal_and_shuffle_which_need_no_rescale():
    """sign_reversed / shuffled already sit at the reference norm by
    construction (negation and permutation both preserve norm exactly), so
    scale_to_norm applied to them is a near-identity — scale_factor ~= 1."""
    reference = float(_full_cone_delta().norm())
    for condition in ("natural_sign_reversed", "natural_shuffled"):
        kwargs = {"atoms": ATOMS, "token_ids": IDS, "coefficients": COEFFS}
        if condition == "natural_shuffled":
            kwargs["seed"] = 3
        raw = build_condition_vector(condition, **kwargs).delta
        scaled, info = scale_to_norm(raw, target_norm=reference)
        assert info["scale_factor"] == pytest.approx(1.0, rel=1e-5)
        assert float(scaled.norm()) == pytest.approx(reference, rel=1e-5)


def test_scale_to_norm_matches_unrelated_and_mass_subcone_which_do_need_rescale():
    """unrelated_cone and mass_subcone generally do *not* sit at the reference
    norm (different generators / a strict subset), so this is where
    scale_to_norm's rescaling actually does work."""
    reference = float(_full_cone_delta().norm())

    unrelated_raw = build_condition_vector(
        "natural_unrelated_cone",
        atoms=ATOMS,
        unrelated_token_ids=[0, 5],
        unrelated_coefficients=[1.0, 1.0],
    ).delta
    assert float(unrelated_raw.norm()) != pytest.approx(reference)
    scaled, info = scale_to_norm(unrelated_raw, target_norm=reference)
    assert float(scaled.norm()) == pytest.approx(reference, rel=1e-5)
    assert info["scale_factor"] != pytest.approx(1.0)

    mass_raw = build_condition_vector(
        "natural_mass_subcone",
        atoms=ATOMS,
        token_ids=IDS,
        coefficients=COEFFS,
        mass_threshold=0.6,
    ).delta
    scaled, info = scale_to_norm(mass_raw, target_norm=reference)
    assert float(scaled.norm()) == pytest.approx(reference, rel=1e-5)


# ------------------------------------------------------------ condition_scaling_mode


def test_condition_scaling_mode_buckets():
    assert condition_scaling_mode("none") == "none"
    assert condition_scaling_mode("zero") == "none"
    assert condition_scaling_mode("natural_scale") == "natural_unscaled"
    for condition in NATURAL_SCALE_MATCHED_CONDITIONS:
        assert condition_scaling_mode(condition) == "natural_matched"
    for condition in (
        "full_cone",
        "mass_subcone",
        "unrelated_cone",
        "random_matched_norm",
        "shuffled",
        "sign_reversed",
        "wrong_layer",
        "wrong_position",
        "raw_activation",
        "activation_diff",
        "manual_subcone",
    ):
        assert condition_scaling_mode(condition) == "ratio_scaled"


def test_condition_scaling_mode_covers_every_declared_condition():
    for condition in VECTOR_CONDITIONS:
        assert condition_scaling_mode(condition) in (
            "none",
            "natural_unscaled",
            "natural_matched",
            "ratio_scaled",
        )


# --------------------------------------------------------------- aggregation


def _record(condition, *, delta_zero=None, total=-4.0, layer=21, schedule="prompt_only"):
    return {
        "vector_condition": condition,
        "example_id": "ex1",
        "source_layer": layer,
        "requested_ratio": None if condition_scaling_mode(condition) != "ratio_scaled" else 0.5,
        "steering_schedule": {"kind": schedule, "decay": None},
        "neutral_prompt_id": "label-colon",
        "total_logprob": total,
        "delta_logprob_vs_zero": delta_zero,
        "delta_logprob_vs_unrelated": None,
        "kl_divergence_from_baseline": 0.1,
        "target_recovered_exact": None,
        "target_recovered_substring": None,
    }


def test_summarize_by_condition_tags_scaling_mode_and_keeps_conditions_separate():
    records = [
        _record("full_cone", delta_zero=1.0),
        _record("unrelated_cone", delta_zero=0.2),
        _record("natural_scale", delta_zero=2.0),
        _record("natural_unrelated_cone", delta_zero=0.3),
    ]
    summary = {row["vector_condition"]: row for row in summarize_by_condition(records)}
    assert summary["full_cone"]["scaling_mode"] == "ratio_scaled"
    assert summary["unrelated_cone"]["scaling_mode"] == "ratio_scaled"
    assert summary["natural_scale"]["scaling_mode"] == "natural_unscaled"
    assert summary["natural_unrelated_cone"]["scaling_mode"] == "natural_matched"
    # Never merged: unrelated_cone and natural_unrelated_cone are separate rows
    # with their own record counts and means, even though one is a rescaled
    # copy of the other's direction.
    assert summary["unrelated_cone"]["n_records"] == 1
    assert summary["natural_unrelated_cone"]["n_records"] == 1
    assert summary["unrelated_cone"]["mean_delta_vs_zero"] == pytest.approx(0.2)
    assert summary["natural_unrelated_cone"]["mean_delta_vs_zero"] == pytest.approx(0.3)


# ------------------------------------------------- natural_scale_verdicts/gonogo


def _natural_battery(example_id, layer, schedule, *, strong: bool):
    correct_total = -2.0 if strong else -9.0
    rows = [_record("natural_scale", total=correct_total, layer=layer, schedule=schedule)]
    rows[0]["example_id"] = example_id
    for control in NATURAL_SCALE_GONOGO_CONTROLS:
        r = _record(control, total=-5.0, layer=layer, schedule=schedule)
        r["example_id"] = example_id
        rows.append(r)
    zero = _record("zero", total=-5.0, layer=layer, schedule=schedule)
    zero["example_id"] = example_id
    rows.append(zero)
    return rows


def test_natural_scale_verdicts_and_gonogo_report():
    records = []
    for schedule in ("prompt_only", "constant", "decaying"):
        records += _natural_battery("ex-strong", 21, schedule, strong=True)
        records += _natural_battery("ex-weak", 21, schedule, strong=False)

    verdicts = natural_scale_verdicts(records)
    assert len(verdicts) == 6  # 2 examples x 3 schedules

    by_key = {(v["example_id"], v["schedule_kind"]): v for v in verdicts}
    strong = by_key[("ex-strong", "prompt_only")]
    assert strong["beats_zero"] is True
    assert strong["beats_all_controls"] is True
    assert strong["n_controls_available"] == len(NATURAL_SCALE_GONOGO_CONTROLS)

    weak = by_key[("ex-weak", "prompt_only")]
    assert weak["beats_zero"] is False
    assert weak["beats_all_controls"] is False

    report = natural_scale_gonogo_report(verdicts)
    assert report["n_points"] == 6
    assert report["n_passing"] == 3  # only the "strong" example's 3 schedules
    assert report["go"] is False  # exactly half is not a clear majority

    solo = natural_scale_gonogo_report(
        [v for v in verdicts if v["example_id"] == "ex-strong"]
    )
    assert solo["go"] is True


def test_natural_scale_verdicts_missing_conditions_yields_no_verdict():
    """If the run never scored natural_scale for some (example, layer,
    schedule) — e.g. the config didn't include the matched conditions — there
    is nothing to compare, and that point is simply absent rather than
    fabricated as a failure."""
    records = [_record("full_cone", delta_zero=1.0)]  # no natural_scale at all
    assert natural_scale_verdicts(records) == []
    with pytest.raises(GenerativeError, match="no natural-scale verdicts"):
        natural_scale_gonogo_report([])


def test_natural_scale_verdicts_partial_controls_still_reports_available_ones():
    """A control missing from the records (e.g. that condition wasn't
    configured) leaves beats_controls[control] = None rather than crashing or
    silently counting as a pass — and a missing control means
    beats_all_controls cannot be True, even though the one control that *did*
    run was beaten. Mirrors per_example_verdicts's existing semantics: "beats
    all controls" is a claim about every control in the criterion, not just the
    ones that happened to be present."""
    records = [
        _record("natural_scale", total=-2.0),
        _record("natural_unrelated_cone", total=-5.0),
        _record("zero", total=-5.0),
        # natural_random_matched_norm / natural_shuffled / natural_sign_reversed
        # are absent.
    ]
    verdicts = natural_scale_verdicts(records)
    assert len(verdicts) == 1
    v = verdicts[0]
    assert v["beats_controls"]["natural_unrelated_cone"] is True
    assert v["beats_controls"]["natural_random_matched_norm"] is None
    assert v["n_controls_available"] == 1
    assert v["beats_all_controls"] is False
