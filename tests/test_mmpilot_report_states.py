# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""PASS / FAIL / NOT_EVALUATED, and the claim the report must never make.

The first real run stopped at the behavioral gate, so no J-space code was ever
computed. The report nonetheless said the frozen lens had not reconstructed the
captured activations well enough — a verdict on a measurement that never
happened. A boolean cannot tell "measured and bad" from "never measured", so
the rubric carries three states and the skipped ones say why.
"""

import json

import pytest

from jlens.mmpilot.report import (
    FAIL,
    NOT_EVALUATED,
    PASS,
    code_statistics,
    decide,
    evaluate_criteria,
    gonogo_report,
)

#: Every criterion that is downstream of the behavioral gate and must report
#: NOT_EVALUATED rather than FAIL when its stage never ran.
SKIPPABLE = (
    "lens_sanity_above_random",
    "representational_structure",
    "jspace_beats_raw_residual",
    "causal_transfer_sign",
    "control_specificity",
    "jspace_direction_beats_raw_direction",
    "activation_norm_sanity",
    "effect_specificity_not_global",
    "effect_precedes_output_convergence",
)


def stopped_at_the_gate(retained=()):
    """Exactly the shape the real run produced: gate failed, nothing after it."""
    return {
        "capability": {
            "text_image_retained_concepts": list(retained),
            "per_concept": {
                "cat": {"text": {"n_correct": 6, "n": 8}, "image": {"n_correct": 8, "n": 8}},
                "bus": {"text": {"n_correct": 5, "n": 8}, "image": {"n_correct": 8, "n": 8}},
                "dog": {"text": {"n_correct": 3, "n": 8}, "image": {"n_correct": 8, "n": 8}},
                "train": {"text": {"n_correct": 5, "n": 8}, "image": {"n_correct": 6, "n": 8}},
            },
        },
        "lens_validation": {"lens_checksum": "sha256:frozen"},
        "code_stats": code_statistics([]),
        "representational": {},
        "interventions": {},
        "blocked_modalities": ["spoken_audio"],
    }


def matched_random_control(*, above_random=True, layer=38):
    """A reconstruction-control summary in the shape the rubric reads.

    The absolute numbers are deliberately small — that is what a sparse
    workspace looks like — and the verdict turns on the excess over matched
    random controls, not on the absolute level.
    """
    excess = 0.06 if above_random else -0.01
    return {
        "schema": "jlens.mmpilot.reconstruction_control.v1",
        "n_records": 8,
        "primary_layer": layer,
        "control_config_hash": "sha256:control",
        "config": {"interpretation": "pool-matched control"},
        "layers_above_random": [layer] if above_random else [],
        "by_layer": {
            str(layer): {
                "layer": layer,
                "n_samples": 8,
                "median_explained_fraction": 0.065,
                "median_excess_explained_fraction": excess,
                "median_excess_over_random_bound": excess - 0.01,
                "above_random": above_random,
                "median_estimated_occupancy": 4 if above_random else 0,
                "all_finite": True,
                "all_nondegenerate": True,
            }
        },
    }


def complete_run(*, reconstruction=0.8, above_random=True):
    retrieval = {
        "jspace_retrieval": {"top1_accuracy": 1.0},
        "shuffled_control": {"p95_top1_accuracy": 0.5},
        "raw_residual_retrieval": {"top1_accuracy": 0.6},
        "jspace_separation": {"gap": 0.9},
        "raw_residual_separation": {"gap": 0.2},
    }
    base = {
        "concept": "cat", "source_modality": "text", "target_modality": "image",
        "pair": "text->image", "off_diagonal": True, "layer": 38, "alpha": 0.25,
        "mean_abs_unrelated_change": 2.0, "mean_activation_norm_ratio": 1.03,
        "fraction_expected_sign": 1.0,
    }
    rows = [
        {**base, "control_kind": "source_concept", "mean_signed_target_effect": 20.0},
        {**base, "control_kind": "random_norm_matched", "mean_signed_target_effect": 1.0},
        {**base, "control_kind": "unrelated_concept", "mean_signed_target_effect": 2.0},
        {**base, "control_kind": "raw_residual_difference", "mean_signed_target_effect": 3.0},
    ]
    return {
        "capability": {"text_image_retained_concepts": ["cat", "dog"]},
        "lens_validation": {"lens_checksum": "sha256:frozen"},
        "code_stats": code_statistics(
            [
                {"modality": "text", "split": "test", "explained_fraction": reconstruction,
                 "n_active": 4, "convergence_status": "ok"},
                {"modality": "image", "split": "test", "explained_fraction": reconstruction,
                 "n_active": 4, "convergence_status": "ok"},
            ]
        ),
        "representational": {"pairs": {p: retrieval for p in ("text->image", "image->text")}},
        "interventions": {"rows": rows},
        "reconstruction_control": matched_random_control(above_random=above_random),
        "blocked_modalities": [],
    }


# ------------------------------------------------------- the three-state rubric


def test_zero_codes_is_not_evaluated_not_a_reconstruction_failure():
    criteria = evaluate_criteria(**stopped_at_the_gate())
    entry = criteria["lens_sanity_above_random"]
    assert entry["status"] == NOT_EVALUATED
    assert entry["passed"] is False
    assert entry["evaluated"] is False
    assert entry["evidence"]["n_codes"] == 0
    assert entry["evidence"]["n_control_records"] == 0
    reason = entry["not_evaluated_reason"]
    assert "no J-space codes and no matched-random control results" in reason
    assert "behavioral capability gate failed" in reason
    assert "not a finding that it does not" in reason


def test_every_skipped_downstream_criterion_reports_not_evaluated():
    criteria = evaluate_criteria(**stopped_at_the_gate())
    for name in SKIPPABLE:
        assert criteria[name]["status"] == NOT_EVALUATED, name
        assert criteria[name]["not_evaluated_reason"], name


def test_the_behavioral_gate_and_audio_block_are_still_really_evaluated():
    """Two things the run *did* observe. Neither may hide behind NOT_EVALUATED."""
    criteria = evaluate_criteria(**stopped_at_the_gate())
    assert criteria["behavioral_capability"]["status"] == FAIL
    assert criteria["spoken_audio_available"]["status"] == FAIL
    assert criteria["spoken_audio_available"]["evidence"]["blocked_modalities"] == [
        "spoken_audio"
    ]


def test_a_complete_run_evaluates_everything():
    criteria = evaluate_criteria(**complete_run())
    assert all(entry["status"] != NOT_EVALUATED for entry in criteria.values())
    assert criteria["lens_sanity_above_random"]["status"] == PASS
    assert criteria["behavioral_capability"]["status"] == PASS


def test_published_native_validation_supersedes_posthoc_reconstruction_gate():
    inputs = complete_run()
    inputs["lens_validation"]["native_readout_validation"] = {
        "status": "validated_text_only",
        "native_readout_layers_passing": [38],
        "native_validation_path": "drive/native_readout_validation.json",
    }
    inputs["reconstruction_control"] = {
        "schema": "jlens.mmpilot.native_validation_reference.v1",
        "n_records": 0,
    }
    criteria = evaluate_criteria(**inputs)
    entry = criteria["lens_sanity_above_random"]
    assert entry["status"] == PASS
    assert entry["evidence"]["validation_method"] == "heldout_native_readout"
    assert entry["evidence"]["native_readout_layers_passing"] == [38]


def test_a_real_lens_shortfall_is_still_a_fail():
    """The repair must not turn a genuine failure into a shrug.

    A lens indistinguishable from matched random directions fails, even though
    its absolute explained fraction is identical to the passing case's.
    """
    criteria = evaluate_criteria(**complete_run(above_random=False))
    entry = criteria["lens_sanity_above_random"]
    assert entry["status"] == FAIL
    assert "not_evaluated_reason" not in entry
    assert entry["evidence"]["layers_above_random"] == []
    assert entry["evidence"]["absolute_median_explained_fraction"] == pytest.approx(0.8)


def test_absolute_reconstruction_alone_decides_nothing():
    """Same absolute number, opposite verdicts — that is the whole point."""
    passing = evaluate_criteria(**complete_run(above_random=True))
    failing = evaluate_criteria(**complete_run(above_random=False))
    for criteria in (passing, failing):
        entry = criteria["lens_sanity_above_random"]["evidence"]
        assert entry["by_layer"]["38"]["median_explained_fraction"] == 0.065
    assert passing["lens_sanity_above_random"]["status"] == PASS
    assert failing["lens_sanity_above_random"]["status"] == FAIL


def test_interventions_that_ran_but_found_nothing_still_fail():
    """No off-diagonal effect is a null result, not an unexecuted stage."""
    inputs = complete_run()
    inputs["interventions"] = {
        "rows": [
            {
                "concept": "cat", "source_modality": "text", "target_modality": "text",
                "pair": "text->text", "off_diagonal": False, "layer": 38, "alpha": 0.25,
                "control_kind": "source_concept", "mean_signed_target_effect": 0.0,
                "fraction_expected_sign": 0.0, "mean_abs_unrelated_change": 0.0,
                "mean_activation_norm_ratio": 1.0,
            }
        ]
    }
    criteria = evaluate_criteria(**inputs)
    assert criteria["causal_transfer_sign"]["status"] == FAIL
    assert criteria["control_specificity"]["status"] == FAIL


# ------------------------------------------------------------------ the verdict


def test_the_rationale_blames_only_what_actually_failed():
    criteria = evaluate_criteria(**stopped_at_the_gate())
    decision = decide(criteria)
    assert decision["recommendation"] == "NO-GO"
    rationale = decision["rationale"]
    assert "behavioral gate" in rationale
    # The claims the old report made and must never make again.
    assert "did not reconstruct" not in rationale
    assert "coordinates mean nothing" not in rationale
    assert "lens" not in rationale.split("Not evaluated")[0]
    assert "Not evaluated (skipped, not failed)" in rationale
    for name in SKIPPABLE:
        assert name in rationale, name


def test_the_next_experiment_points_at_the_evidence_audit():
    decision = decide(evaluate_criteria(**stopped_at_the_gate()))
    nxt = decision["next_experiment"]
    assert "written caption" in nxt
    assert "COCO object annotation" in nxt
    assert "CPU" in nxt


def test_a_run_that_stopped_after_a_passing_gate_reports_absence_not_failure():
    inputs = stopped_at_the_gate(retained=("cat", "dog"))
    criteria = evaluate_criteria(**inputs)
    assert criteria["behavioral_capability"]["status"] == PASS
    decision = decide(criteria)
    assert decision["recommendation"] == "NO-GO"
    assert "did not reach the stages" in decision["rationale"]
    assert "nothing here is evidence for or against" in decision["rationale"]
    assert "resume" in decision["next_experiment"].lower()


def test_a_complete_healthy_run_still_reaches_go():
    decision = decide(evaluate_criteria(**complete_run()))
    assert decision["recommendation"] == "GO"
    assert "Not evaluated" not in decision["rationale"]


# ------------------------------------------------------------------- the report


def _report(inputs, **kwargs):
    return gonogo_report(
        mode="pilot",
        run_dir="/runs/x",
        invariance=None,
        **{k: v for k, v in inputs.items()},
        **kwargs,
    )


def test_the_markdown_says_not_evaluated_and_explains_what_that_means():
    markdown, summary = _report(stopped_at_the_gate())
    assert "| lens_sanity_above_random | NOT EVALUATED |" in markdown
    assert "`NOT EVALUATED` means the stage never ran" in markdown
    assert "## Not evaluated" in markdown
    # Question 2 must not read as a reconstruction failure.
    question_two = next(
        line for line in markdown.splitlines() if line.startswith("2. **")
    )
    assert "NOT EVALUATED" in question_two
    assert "nothing here says the lens reconstructed poorly" in question_two.lower()


def test_the_summary_exposes_the_states_machine_readably():
    _, summary = _report(stopped_at_the_gate())
    assert summary["schema"] == "jlens.mmpilot.gonogo.v2"
    assert summary["criteria_status"]["lens_sanity_above_random"] == NOT_EVALUATED
    assert summary["criteria_status"]["behavioral_capability"] == FAIL
    assert set(summary["not_evaluated"]) == set(SKIPPABLE)
    assert all(reason for reason in summary["not_evaluated"].values())
    # Round-trips, so the artifact on disk carries the same three states.
    assert json.loads(json.dumps(summary, default=str))["not_evaluated"]


def test_the_report_states_the_evidence_rule():
    markdown, _ = _report(stopped_at_the_gate())
    assert "COCO object annotation" in markdown
    assert "approved synonym" in markdown
    assert "not a valid synchronized positive" in markdown


def test_a_complete_run_reports_no_not_evaluated_section():
    markdown, summary = _report(complete_run())
    assert summary["not_evaluated"] == {}
    assert "## Not evaluated" not in markdown
    # No criterion row carries the state, only the legend above the table.
    rows = [line for line in markdown.splitlines() if line.startswith("| ")]
    assert rows and all("NOT EVALUATED" not in row for row in rows)
