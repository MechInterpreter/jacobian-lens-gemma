# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""The matched-random reconstruction control, and the criterion it feeds.

The pilot used to gate the frozen lens on an absolute number: explain 50% of a
held-out activation or be declared meaningless. That threshold has no basis.
Anthropic's J-space work reports a median of roughly 6-7% of a concept vector's
variance in its top-k J-space component, and excess over a same-size random
control that never exceeds about 10% — a 50% gate would fail the published
result. The interesting claim is that this small component carries
disproportionate causal content, which reconstruction cannot settle either way.

So these tests pin down the replacement: low absolute reconstruction passes
when it reliably beats matched random directions, and high absolute
reconstruction fails when random directions do just as well.
"""

import math

import pytest
import torch

from jlens.mmpilot import reconstruction as R
from jlens.mmpilot.report import (
    DEFAULT_THRESHOLDS,
    FAIL,
    NOT_EVALUATED,
    PASS,
    evaluate_criteria,
    gonogo_report,
)
from jlens.pursuit import JSpaceDictionary, PursuitSettings

# Wide enough that a handful of atoms cannot explain an activation by
# accident, so 'low absolute reconstruction' in these tests is really low.
D_MODEL = 512
SETTINGS = PursuitSettings(k=8, refine_steps=1, tol_relative_residual=0.0)
CONFIG = R.ReconstructionControlConfig(n_draws=5, k_schedule=(1, 2, 4, 8))


def _unit(seed: int, d_model: int = D_MODEL) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    vector = torch.randn(d_model, generator=generator, dtype=torch.float32)
    return vector / vector.norm()


def target(seed: int = 11) -> torch.Tensor:
    return 4.0 * _unit(seed)


def aligned_dictionary(alignment: float, *, n_atoms: int = 128, seed: int = 5):
    """Atoms carrying ``alignment`` of the target direction plus noise.

    ``alignment`` tunes how much of the target the dictionary can explain, so a
    test can ask for a *low* absolute explained fraction that is still reliably
    above what unaligned random directions of the same norm achieve.
    """
    direction = _unit(11)
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(n_atoms, D_MODEL, generator=generator, dtype=torch.float32)
    noise = noise / noise.norm(dim=-1, keepdim=True)
    atoms = alignment * direction.unsqueeze(0) + math.sqrt(1 - alignment**2) * noise
    return JSpaceDictionary(atoms, layer=7, provenance={"kind": "test"})


def unaligned_dictionary(*, n_atoms: int = 128, seed: int = 9):
    """A dictionary that is statistically nothing but the control itself."""
    generator = torch.Generator().manual_seed(seed)
    atoms = torch.randn(n_atoms, D_MODEL, generator=generator, dtype=torch.float32)
    return JSpaceDictionary(
        atoms / atoms.norm(dim=-1, keepdim=True), layer=7, provenance={"kind": "test"}
    )


def record_for(dictionary, *, sample_id="s1", config=CONFIG):
    return R.reconstruction_control_record(
        target(),
        dictionary,
        SETTINGS,
        config=config,
        sample_id=sample_id,
        layer=7,
        modality="text",
        split="test",
        activation_checksum="sha256:activation",
        lens_checksum="sha256:lens",
    )


# ---------------------------------------------------------- the control itself


def test_random_controls_are_size_and_norm_matched():
    norms = [3.0, 1.0, 7.5]
    control = R.matched_random_dictionary(
        norms, D_MODEL, layer=7, seed_parts=("s1", 0)
    )
    assert control.n_atoms == len(norms)
    assert control.d_model == D_MODEL
    for atom, expected in zip(control.atoms, norms, strict=True):
        assert float(atom.norm()) == pytest.approx(expected, rel=1e-5)
    assert control.provenance["kind"] == "matched_random_control"


def test_controls_preserve_dtype_and_device_of_the_lens_dictionary():
    dictionary = aligned_dictionary(0.3)
    control = R.matched_random_dictionary(
        [1.0, 2.0],
        dictionary.d_model,
        layer=dictionary.layer,
        seed_parts=("s1", 0),
        device=dictionary.device,
        dtype=dictionary.atoms.dtype,
    )
    assert control.atoms.dtype == dictionary.atoms.dtype
    assert control.atoms.device == dictionary.device
    assert control.layer == dictionary.layer


def test_both_controls_are_run_and_only_the_pool_matched_one_gates():
    """A k-atom control is beaten by ANY dictionary with a larger selection
    pool, noise included, so it cannot be the gate. Both are computed; the
    pool-matched one gives random directions the lens's own selection freedom."""
    dictionary = aligned_dictionary(0.3, n_atoms=128)
    record = record_for(dictionary)
    assert record["criterion_control"] == "pool_matched"
    controls = record["controls"]
    assert controls["support_matched"]["n_control_atoms"] == record["n_support_atoms"]
    assert controls["pool_matched"]["n_control_atoms"] == dictionary.n_atoms
    assert controls["pool_matched"]["k"] == SETTINGS.k
    # The gate reads the pool-matched numbers.
    assert record["random_median_explained_fraction"] == (
        controls["pool_matched"]["median_explained_fraction"]
    )
    # The support-matched control is systematically easier to beat.
    assert (
        controls["support_matched"]["median_explained_fraction"]
        < controls["pool_matched"]["median_explained_fraction"]
    )
    assert CONFIG.to_dict()["criterion_control"] == "pool_matched"
    assert "cannot gate anything" in (
        CONFIG.to_dict()["support_matched_control_is_reported_not_gated"]
    )


def test_a_capped_control_pool_is_disclosed_as_a_bias_toward_the_lens():
    small = R.ReconstructionControlConfig(n_draws=2, k_schedule=(1, 2), max_control_pool_atoms=32)
    record = record_for(aligned_dictionary(0.3, n_atoms=512), config=small)
    assert record["control_pool_capped"]
    assert record["lens_pool_size"] == 512
    assert record["control_pool_size"] == 32
    assert record["pool_selection_bias_factor"] > 1.0
    assert "favours the lens" in record["pool_bias_direction"]

    uncapped = record_for(aligned_dictionary(0.3, n_atoms=128))
    assert not uncapped["control_pool_capped"]
    assert uncapped["pool_selection_bias_factor"] == 1.0


def test_the_default_control_pool_equals_the_lens_pool():
    """The only matched comparison: the control searches what the lens searches."""
    assert R.ReconstructionControlConfig().max_control_pool_atoms is None
    for n_atoms in (128, 512):
        record = record_for(aligned_dictionary(0.3, n_atoms=n_atoms))
        assert record["control_pool_size"] == n_atoms
        assert record["pool_matched_exactly"]
        assert record["criterion_status"] == R.STATUS_EVALUATED
        assert record["controls"]["pool_matched"]["n_control_atoms"] == n_atoms


def test_a_capped_pool_is_not_evaluated_rather_than_believed():
    """The Part 5 repair: a control searching a sixteenth of the lens's pool
    understates random performance, so it cannot be counted as evidence."""
    capped = R.ReconstructionControlConfig(
        n_draws=2, k_schedule=(1, 2), max_control_pool_atoms=32
    )
    record = record_for(aligned_dictionary(0.9, n_atoms=512), config=capped)
    # The lens beats this control easily — and that still is not a result.
    assert record["above_random_bound"]
    assert record["criterion_status"] == R.STATUS_NOT_EVALUATED
    assert not record["pool_matched_exactly"]
    assert "cannot establish" in record["criterion_status_reason"]

    summary = R.summarize_reconstruction_controls([record], config=capped, primary_layer=7)
    assert summary["by_layer"]["7"]["criterion_status"] == R.STATUS_NOT_EVALUATED
    assert not summary["by_layer"]["7"]["above_random"]
    assert summary["layers_above_random"] == []
    assert summary["layers_not_evaluated"] == [7]
    assert summary["criterion_evaluable"] is False


def test_a_capped_pool_reports_not_evaluated_not_a_failed_lens():
    """It must not read as a failure either: the comparison was never made."""
    capped = R.ReconstructionControlConfig(
        n_draws=2, k_schedule=(1, 2), max_control_pool_atoms=32
    )
    records = [record_for(aligned_dictionary(0.9, n_atoms=512), config=capped)]
    _, criteria = _criteria(records, retained=("cat", "dog"))
    entry = criteria["lens_sanity_above_random"]
    assert entry["status"] == NOT_EVALUATED
    assert "smaller candidate pool" in entry["not_evaluated_reason"]
    assert "not a finding that it does not" in entry["not_evaluated_reason"]


def test_disabling_the_pool_match_is_conditional_never_a_clean_pass():
    """The escape hatch is explicit, fingerprinted, and still not a PASS."""
    lenient = R.ReconstructionControlConfig(
        n_draws=2,
        k_schedule=(1, 2),
        max_control_pool_atoms=32,
        require_pool_match=False,
    )
    record = record_for(aligned_dictionary(0.9, n_atoms=512), config=lenient)
    assert record["criterion_status"] == R.STATUS_CONDITIONAL
    summary = R.summarize_reconstruction_controls([record], config=lenient, primary_layer=7)
    assert summary["by_layer"]["7"]["criterion_status"] == R.STATUS_CONDITIONAL
    assert not summary["by_layer"]["7"]["above_random"]
    assert summary["criterion_evaluable"] is False
    # And the setting changes the fingerprint, so a run cannot be quietly
    # relabelled after the fact.
    assert lenient.fingerprint != R.ReconstructionControlConfig(
        n_draws=2, k_schedule=(1, 2), max_control_pool_atoms=32
    ).fingerprint


def test_the_pool_ladder_reports_stability_without_gating_on_it():
    ladder = R.ReconstructionControlConfig(
        n_draws=2, k_schedule=(1, 2), max_control_pool_atoms=32, pool_ladder=(16, 64)
    )
    record = record_for(aligned_dictionary(0.9, n_atoms=512), config=ladder)
    sizes = [rung["n_control_atoms"] for rung in record["pool_ladder"]]
    assert sizes == [16, 64]
    assert all("jlens_excess" in rung for rung in record["pool_ladder"])
    # Stability is informative; it does not turn a mismatched pool into a pass.
    assert record["criterion_status"] == R.STATUS_NOT_EVALUATED


def test_draws_and_results_are_deterministic():
    first = record_for(aligned_dictionary(0.3))
    second = record_for(aligned_dictionary(0.3))
    assert first["random_explained_fractions"] == second["random_explained_fractions"]
    assert first["jlens_explained_fraction"] == second["jlens_explained_fraction"]
    assert first["control_config_hash"] == second["control_config_hash"]
    # A different sample id draws different controls.
    other = record_for(aligned_dictionary(0.3), sample_id="s2")
    assert other["random_explained_fractions"] != first["random_explained_fractions"]


def test_a_record_carries_everything_needed_to_re_derive_it():
    record = record_for(aligned_dictionary(0.3))
    for field in (
        "layer",
        "sample_id",
        "modality",
        "split",
        "k",
        "n_support_atoms",
        "jlens_explained_fraction",
        "random_explained_fractions",
        "random_mean_explained_fraction",
        "random_median_explained_fraction",
        "random_stdev_explained_fraction",
        "random_upper_bound_explained_fraction",
        "excess_explained_fraction",
        "excess_over_random_bound",
        "occupancy",
        "seed",
        "control_config_hash",
        "activation_checksum",
        "lens_checksum",
    ):
        assert field in record, field
    assert record["occupancy"]["per_k"], "marginal improvements are recorded"


def test_an_early_stopping_pursuit_is_refused_rather_than_approximated():
    with pytest.raises(R.ControlConfigurationError, match="tol_relative_residual"):
        R.reconstruction_control_record(
            target(),
            aligned_dictionary(0.3),
            PursuitSettings(k=8, tol_relative_residual=0.1),
            config=CONFIG,
            sample_id="s1",
            layer=7,
            modality="text",
            split="test",
            activation_checksum="sha256:a",
            lens_checksum="sha256:l",
        )


# ------------------------------------------------------------------- occupancy


def test_occupancy_uses_marginal_improvement_against_the_random_bound():
    jlens = {1: 0.30, 2: 0.45, 4: 0.50, 8: 0.51}
    random_bound = {1: 0.05, 2: 0.12, 4: 0.20, 8: 0.30}
    #     marginal jlens:  0.30  0.15  0.05  0.01
    #     marginal bound:  0.05  0.07  0.08  0.10
    # so the lens stops buying more than random after k=2.
    estimate = R.occupancy(jlens, random_bound, (1, 2, 4, 8))
    assert estimate["estimated_occupancy"] == 2
    assert [row["above"] for row in estimate["per_k"]] == [True, True, False, False]


def test_occupancy_is_zero_when_the_lens_never_beats_the_control():
    estimate = R.occupancy(
        {1: 0.01, 2: 0.02}, {1: 0.10, 2: 0.10}, (1, 2)
    )
    assert estimate["estimated_occupancy"] == 0


def test_occupancy_never_claims_to_replicate_the_published_measure():
    estimate = R.occupancy({1: 0.5}, {1: 0.1}, (1,))
    assert estimate["is_exact_replication_of_published_occupancy"] is False
    assert "short k schedule" in estimate["approximation"]


def test_the_k_curve_is_exact_not_interpolated():
    """The prefix of a k_max run equals a run configured at that k, so the
    schedule costs one pursuit rather than one per k."""
    dictionary = aligned_dictionary(0.4)
    record = record_for(dictionary)
    for k in (1, 2, 4, 8):
        rerun = R.reconstruction_control_record(
            target(),
            dictionary,
            PursuitSettings(k=k, refine_steps=1, tol_relative_residual=0.0),
            config=R.ReconstructionControlConfig(n_draws=1, k_schedule=(k,)),
            sample_id="s1",
            layer=7,
            modality="text",
            split="test",
            activation_checksum="sha256:a",
            lens_checksum="sha256:l",
        )
        assert record["jlens_curve"][k] == pytest.approx(
            rerun["jlens_explained_fraction"], abs=1e-5
        )


# --------------------------------------------------- the criterion, both ways


def _criteria(records, *, primary_layer=7, retained=("cat", "dog")):
    summary = R.summarize_reconstruction_controls(
        records, config=CONFIG, primary_layer=primary_layer
    )
    return summary, evaluate_criteria(
        capability={"text_image_retained_concepts": list(retained)},
        lens_validation={"lens_checksum": "sha256:lens"},
        code_stats={"n": len(records), "median_explained_fraction": 0.02},
        representational={},
        interventions={},
        reconstruction_control=summary,
    )


def test_low_absolute_reconstruction_passes_when_it_beats_matched_random():
    """The case the old 50% gate got wrong.

    Absolute reconstruction here sits in the published ballpark — well under
    20% — and the old rubric would have declared the lens meaningless. It
    reliably beats matched random directions, so it passes.
    """
    records = [
        record_for(aligned_dictionary(0.05, seed=100 + i), sample_id=f"s{i}")
        for i in range(6)
    ]
    summary, criteria = _criteria(records)
    layer = summary["by_layer"]["7"]
    assert layer["median_explained_fraction"] < 0.2, "absolute reconstruction is low"
    assert layer["median_excess_explained_fraction"] > 0
    assert layer["median_excess_over_random_bound"] > 0
    assert layer["above_random"]
    entry = criteria["lens_sanity_above_random"]
    assert entry["status"] == PASS
    assert entry["evidence"]["absolute_median_explained_fraction"] == 0.02


def test_reconstruction_indistinguishable_from_random_fails():
    """A dictionary of unaligned directions is statistically the control, so
    however much or little it explains, it must not pass."""
    records = [
        record_for(unaligned_dictionary(seed=200 + i), sample_id=f"s{i}")
        for i in range(6)
    ]
    summary, criteria = _criteria(records)
    layer = summary["by_layer"]["7"]
    assert not layer["above_random"]
    assert criteria["lens_sanity_above_random"]["status"] == FAIL


def test_high_absolute_reconstruction_fails_when_random_does_as_well():
    """Absolute level is not evidence: a big unaligned dictionary reconstructs
    a lot and must still fail, however healthy the number looks."""
    records = [
        record_for(unaligned_dictionary(n_atoms=4096, seed=300 + i), sample_id=f"s{i}")
        for i in range(6)
    ]
    summary, criteria = _criteria(records)
    layer = summary["by_layer"]["7"]
    assert layer["median_excess_over_random_bound"] <= 0
    assert criteria["lens_sanity_above_random"]["status"] == FAIL

    # And the crossover, stated outright: this FAILING dictionary explains MORE
    # of the activation in absolute terms than the PASSING one above does.
    passing = R.summarize_reconstruction_controls(
        [
            record_for(aligned_dictionary(0.05, seed=100 + i), sample_id=f"s{i}")
            for i in range(6)
        ],
        config=CONFIG,
        primary_layer=7,
    )["by_layer"]["7"]
    assert layer["median_explained_fraction"] > passing["median_explained_fraction"]
    assert passing["above_random"] and not layer["above_random"]


def test_zero_control_results_are_not_evaluated():
    summary, criteria = _criteria([])
    assert summary["n_records"] == 0
    entry = criteria["lens_sanity_above_random"]
    assert entry["status"] == NOT_EVALUATED
    reason = entry["not_evaluated_reason"]
    assert "no J-space codes and no matched-random control results" in reason
    assert "not a finding that it does not" in reason


def test_degenerate_pursuit_output_cannot_pass():
    records = [
        record_for(aligned_dictionary(0.22, seed=100 + i), sample_id=f"s{i}")
        for i in range(6)
    ]
    records[0] = {**records[0], "nondegenerate": False}
    summary, criteria = _criteria(records)
    assert not summary["by_layer"]["7"]["above_random"]
    assert criteria["lens_sanity_above_random"]["status"] == FAIL


def test_a_second_layer_can_carry_the_criterion():
    """Reproducibility at one of the two selected layers is enough."""
    good = [
        {**record_for(aligned_dictionary(0.3, seed=400 + i), sample_id=f"a{i}"), "layer": 9}
        for i in range(4)
    ]
    bad = [
        {**record_for(unaligned_dictionary(seed=500 + i), sample_id=f"b{i}"), "layer": 7}
        for i in range(4)
    ]
    summary, criteria = _criteria(good + bad, primary_layer=7)
    assert summary["layers_above_random"] == [9]
    assert criteria["lens_sanity_above_random"]["status"] == PASS


# ------------------------------------------------------------------ the report


def _report(records, code_stats=None):
    summary = R.summarize_reconstruction_controls(
        records, config=CONFIG, primary_layer=7
    )
    return gonogo_report(
        mode="pilot",
        run_dir="/runs/x",
        capability={"text_image_retained_concepts": ["cat", "dog"]},
        lens_validation={"lens_checksum": "sha256:lens"},
        code_stats=code_stats or {"n": len(records), "median_explained_fraction": 0.015},
        representational={},
        interventions={},
        invariance=None,
        reconstruction_control=summary,
    )


def test_the_report_separates_absolute_from_excess_from_occupancy():
    records = [
        record_for(aligned_dictionary(0.22, seed=100 + i), sample_id=f"s{i}")
        for i in range(6)
    ]
    markdown, summary = _report(records)
    sentence = next(line for line in markdown.splitlines() if line.startswith("2. **"))
    assert "of total activation variance" in sentence
    assert "Low absolute reconstruction is expected for a sparse workspace" in sentence
    assert "median excess over matched random controls" in sentence
    assert "Estimated occupancy per layer" in sentence
    assert "says nothing on its own about causal usefulness" in sentence
    assert summary["reconstruction_control"]["n_records"] == 6


def test_the_report_never_makes_the_three_forbidden_claims():
    for records in (
        [],
        [record_for(unaligned_dictionary(seed=600 + i), sample_id=f"s{i}") for i in range(4)],
        [record_for(aligned_dictionary(0.3, seed=700 + i), sample_id=f"s{i}") for i in range(4)],
    ):
        markdown, _ = _report(records)
        lowered = markdown.lower()
        assert "coordinates mean nothing" not in lowered
        assert "anthropic expected 50" not in lowered
        assert "50% reconstruction" not in lowered
        assert "min_median_explained_fraction" not in markdown


def test_a_skipped_run_reports_not_evaluated_rather_than_a_failed_lens():
    markdown, summary = _report([], code_stats={"n": 0, "median_explained_fraction": None})
    sentence = next(line for line in markdown.splitlines() if line.startswith("2. **"))
    assert "NOT EVALUATED" in sentence
    assert "Nothing here says the lens reconstructed poorly" in sentence
    assert summary["criteria_status"]["lens_sanity_above_random"] == NOT_EVALUATED


def test_the_report_states_the_control_limitation():
    records = [record_for(aligned_dictionary(0.3), sample_id="s0")]
    markdown, _ = _report(records)
    assert "candidate pool size" in markdown
    assert "capped pool reports **NOT EVALUATED**" in markdown
    assert "necessary condition, not a strong one" in markdown
    assert "no absolute reconstruction threshold" in markdown.lower()


def test_no_absolute_reconstruction_threshold_survives_anywhere():
    assert "min_median_explained_fraction" not in DEFAULT_THRESHOLDS
    assert DEFAULT_THRESHOLDS["min_layers_above_random"] == 1
