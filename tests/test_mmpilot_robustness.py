# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""The robustness rubric, the pass budget, and complete fingerprint coverage.

The rubric's central property is that it cannot be satisfied by one lucky cell,
so most of these tests build a design where exactly one cell is strong and
check that the verdict says so.
"""

import pytest

from jlens.mmpilot.capability import PROMPT_PROTOCOL_VERSION, build_ordered_questions
from jlens.mmpilot.pipeline import (
    SELECTION_FINGERPRINT_VERSION,
    PilotConfig,
    scientific_fingerprint,
)
from jlens.mmpilot.robustness import (
    ROBUSTNESS_GO,
    ROBUSTNESS_NO_GO,
    ROBUSTNESS_WEAK_GO,
    RobustnessThresholds,
    estimate_model_passes,
    evaluate_causal_cells,
    format_budget,
    render_report,
    robustness_verdict,
)
from jlens.mmpilot.store import IncompatibleStateError, RunFingerprint, UnitStore

CONCEPTS = ["zebra", "cat", "toilet", "giraffe", "bird", "clock"]
FOCAL = CONCEPTS[:3]
CONTROLS = {"zebra": "giraffe", "cat": "bird", "toilet": "clock"}
PAIRS = ("text->image", "image->text")


def row(concept, pair, control_kind, effect, **overrides):
    source, target = pair.split("->")
    payload = {
        "concept": concept,
        "pair": pair,
        "source_modality": source,
        "target_modality": target,
        "off_diagonal": True,
        "layer": 38,
        "control_kind": control_kind,
        "alpha": 0.5,
        "n": 16,
        "n_distinct_images": 16,
        "n_groups": 16,
        "n_positive_images": 8,
        "n_negative_images": 8,
        "mean_signed_target_effect": effect,
        "mean_signed_margin_effect": effect,
        "fraction_expected_sign": 1.0,
        "mean_abs_unrelated_change": 0.1,
        "mean_activation_norm_ratio": 1.0,
        "n_prediction_changes": 4,
    }
    payload.update(overrides)
    return payload


def cell_rows(concept, pair, *, effect=10.0, **overrides):
    """A source-concept row plus its three matched controls."""
    return [
        row(concept, pair, "source_concept", effect, **overrides),
        row(concept, pair, "random_norm_matched", 1.0),
        row(concept, pair, "unrelated_concept", 1.0),
        row(concept, pair, "raw_residual_difference", 2.0),
    ]


def interventions(rows):
    return {"rows": rows, "aggregation_version": "mean_of_within_image_means.v1"}


def capability(retained=None):
    return {
        "threshold": 0.7,
        "text_image_retained_concepts": list(
            CONCEPTS if retained is None else retained
        ),
        "per_concept": {},
    }


def representational(*, jspace_top1=1.0, shuffled=0.4, jspace_gap=0.5, raw_gap=0.1):
    return {
        "pairs": {
            pair: {
                "jspace_retrieval": {
                    "top1_accuracy": jspace_top1, "mrr": jspace_top1, "n_queries": 48
                },
                "shuffled_control": {"p95_top1_accuracy": shuffled},
                "raw_residual_retrieval": {"top1_accuracy": 0.5},
                "jspace_separation": {"gap": jspace_gap},
                "raw_residual_separation": {"gap": raw_gap},
                "jspace_support_overlap": {"gap": 0.3},
                "exclusions": {
                    "eligible_targets": {"min": 40, "median": 44, "max": 46},
                    "n_excluded_same_image_different_group": 0,
                },
            }
            for pair in PAIRS
        }
    }


def verdict(rows, **overrides):
    payload = {
        "capability": capability(),
        "representational": representational(),
        "interventions": interventions(rows),
        "selected_concepts": CONCEPTS,
        "focal_concepts": FOCAL,
        "unrelated_controls": CONTROLS,
        "thresholds": RobustnessThresholds(),
    }
    payload.update(overrides)
    return robustness_verdict(**payload)


def all_cells(effect=10.0, concepts=FOCAL, pairs=PAIRS):
    return [r for c in concepts for p in pairs for r in cell_rows(c, p, effect=effect)]


# --------------------------------------------------------------- the rubric


def test_full_bidirectional_replication_is_a_go():
    result = verdict(all_cells())

    assert result["verdict"] == ROBUSTNESS_GO
    assert result["concepts_transferring_both_directions"] == sorted(FOCAL)
    assert all(
        result["criteria_status"][name] == "PASS"
        for name in result["go_requirements"]
    )


def test_the_strongest_cell_alone_is_never_enough():
    """The pilot's rubric took the maximum off-diagonal effect. With three
    focal concepts that would report the luckiest cell as the result."""
    rows = cell_rows("zebra", "text->image", effect=100.0)
    for concept in FOCAL:
        for pair in PAIRS:
            if (concept, pair) == ("zebra", "text->image"):
                continue
            rows += cell_rows(concept, pair, effect=0.5)

    result = verdict(rows)
    assert result["verdict"] == ROBUSTNESS_NO_GO
    assert result["concepts_transferring_both_directions"] == []
    assert result["criteria_status"]["bidirectional_causal_replication"] == "FAIL"


def test_two_of_three_concepts_transferring_both_ways_is_enough():
    rows = all_cells(concepts=["zebra", "cat"])
    rows += [r for p in PAIRS for r in cell_rows("toilet", p, effect=0.2)]

    result = verdict(rows)
    assert result["verdict"] == ROBUSTNESS_GO
    assert result["concepts_transferring_both_directions"] == ["cat", "zebra"]


def test_one_direction_only_is_a_weak_go():
    rows = [r for c in FOCAL for r in cell_rows(c, "text->image", effect=10.0)]
    rows += [r for c in FOCAL for r in cell_rows(c, "image->text", effect=0.2)]

    result = verdict(rows)
    assert result["verdict"] == ROBUSTNESS_WEAK_GO
    assert result["concepts_transferring_both_directions"] == []
    assert sorted(result["concepts_transferring_either_direction"]) == sorted(FOCAL)


def test_a_raw_direction_matching_the_jspace_one_downgrades_but_does_not_veto():
    """Reported, not blocking: the raw difference-in-means direction answers
    'did the decomposition earn its keep', not 'did transfer happen'."""
    rows = []
    for concept in FOCAL:
        for pair in PAIRS:
            rows += [
                row(concept, pair, "source_concept", 10.0),
                row(concept, pair, "random_norm_matched", 1.0),
                row(concept, pair, "unrelated_concept", 1.0),
                row(concept, pair, "raw_residual_difference", 20.0),
            ]

    result = verdict(rows)
    assert result["verdict"] == ROBUSTNESS_WEAK_GO
    assert result["raw_direction_exceptions"]
    assert result["criteria_status"]["bidirectional_causal_replication"] == "PASS"
    assert "downgraded" in result["rationale"]


def test_failing_six_way_capability_is_a_no_go():
    result = verdict(all_cells(), capability=capability(retained=CONCEPTS[:5]))

    assert result["verdict"] == ROBUSTNESS_NO_GO
    assert result["criteria_status"]["six_way_capability"] == "FAIL"
    evidence = result["criteria"]["six_way_capability"]["evidence"]
    assert evidence["concepts_failing_the_gate"] == ["clock"]
    assert "never replaced" in evidence["reading"]


def test_representation_at_shuffled_levels_is_a_no_go():
    result = verdict(all_cells(), representational=representational(shuffled=1.0))

    assert result["verdict"] == ROBUSTNESS_NO_GO
    assert result["criteria_status"]["representation_beats_shuffled"] == "FAIL"


def test_raw_separation_beating_jspace_separation_fails_its_criterion():
    result = verdict(all_cells(), representational=representational(jspace_gap=0.1, raw_gap=0.5))

    assert result["criteria_status"]["jspace_separation_beats_raw"] == "FAIL"
    assert result["verdict"] == ROBUSTNESS_NO_GO


def test_a_cell_short_of_its_distinct_images_cannot_pass():
    rows = all_cells()
    for entry in rows:
        if entry["concept"] == "zebra" and entry["pair"] == "text->image":
            entry["n_positive_images"] = 3

    result = verdict(rows)
    assert result["criteria_status"]["distinct_images_per_cell"] == "FAIL"
    cells = {(c["concept"], c["pair"]): c for c in result["causal_cells"]}
    assert not cells[("zebra", "text->image")]["passes"]
    assert any(
        "3 distinct positive image(s) < 8" in reason
        for reason in cells[("zebra", "text->image")]["reasons"]
    )


def test_an_effect_not_clearing_its_own_controls_fails_that_cell():
    rows = []
    for concept in FOCAL:
        for pair in PAIRS:
            rows += [
                row(concept, pair, "source_concept", 1.2),
                row(concept, pair, "random_norm_matched", 1.0),
                row(concept, pair, "unrelated_concept", 1.0),
            ]
    result = verdict(rows)

    assert result["verdict"] == ROBUSTNESS_NO_GO
    assert all(not cell["passes"] for cell in result["causal_cells"])


def test_a_globally_disruptive_edit_fails_its_cell():
    rows = all_cells()
    for entry in rows:
        if entry["control_kind"] == "source_concept":
            entry["mean_abs_unrelated_change"] = 50.0

    result = verdict(rows)
    assert result["verdict"] == ROBUSTNESS_NO_GO
    assert any(
        "the edit looks global" in reason
        for cell in result["causal_cells"]
        for reason in cell["reasons"]
    )


def test_an_unstable_activation_norm_fails_its_cell():
    rows = all_cells()
    for entry in rows:
        if entry["control_kind"] == "source_concept":
            entry["mean_activation_norm_ratio"] = 5.0

    result = verdict(rows)
    assert any(
        "activation norm ratio" in reason
        for cell in result["causal_cells"]
        for reason in cell["reasons"]
    )


def test_a_low_expected_sign_fraction_fails_its_cell():
    rows = all_cells()
    for entry in rows:
        if entry["control_kind"] == "source_concept":
            entry["fraction_expected_sign"] = 0.5

    result = verdict(rows)
    assert any(
        "expected-sign fraction 0.50 < 0.75" in reason
        for cell in result["causal_cells"]
        for reason in cell["reasons"]
    )


def test_an_unrelated_control_inside_the_focal_set_fails():
    result = verdict(all_cells(), unrelated_controls={"zebra": "cat"})

    assert result["criteria_status"]["external_unrelated_control_is_external"] == "FAIL"
    assert "direct contrast" in (
        result["criteria"]["external_unrelated_control_is_external"]["evidence"]["reading"]
    )


def test_the_late_layer_and_scope_limits_are_always_stated():
    result = verdict(all_cells())

    assert "cannot establish" in result["late_layer_limitation"]
    assert "pre-convergence" in result["late_layer_limitation"]
    assert "Spoken audio is excluded by design" in result["scope_limitation"]
    assert result["criteria_status"]["spoken_audio_excluded_by_design"] == "PASS"


def test_evaluate_causal_cells_reports_every_focal_direction():
    cells = evaluate_causal_cells(
        interventions(all_cells()), focal_concepts=FOCAL, thresholds=RobustnessThresholds()
    )
    assert len(cells) == len(FOCAL) * len(PAIRS)
    assert {(c["concept"], c["pair"]) for c in cells} == {
        (c, p) for c in FOCAL for p in PAIRS
    }


def test_a_missing_cell_is_reported_as_unevaluated_not_as_a_pass():
    cells = evaluate_causal_cells(
        interventions(cell_rows("zebra", "text->image")),
        focal_concepts=FOCAL,
        thresholds=RobustnessThresholds(),
    )
    missing = [c for c in cells if not c["evaluated"]]
    assert len(missing) == 5
    assert all(not c["passes"] for c in missing)


def test_the_report_renders_the_replication_tables():
    result = verdict(all_cells())
    report = render_report(
        run_dir="/runs/x", verdict=result, budget=None, resume=None, mode="robustness"
    )

    assert f"# Six-concept robustness study — {ROBUSTNESS_GO}" in report
    assert "Causal replication (image is the unit)" in report
    assert "strongest cell decides nothing on its own" in report
    assert "not erasure" in report
    for concept in FOCAL:
        assert concept in report


def test_a_mock_report_is_labelled_as_pipeline_evidence_only():
    report = render_report(
        run_dir="/runs/x", verdict=verdict(all_cells()), budget=None, resume=None, mode="mock"
    )
    assert "MOCK run: pipeline evidence only, not scientific evidence." in report


# ---------------------------------------------------------- the pass budget


def test_the_budget_counts_every_forward_pass_the_design_implies():
    budget = estimate_model_passes(
        n_concepts=6,
        n_focal_concepts=3,
        modalities=("text", "image"),
        n_total_groups=112,
        n_capability_groups=48,
        n_targets_per_cell=16,
        alphas=(0.0, 0.25, 0.5),
    )
    # 48 groups x 2 modalities x 2 option orders x 6 candidates.
    assert budget.capability_passes == 1152
    assert budget.activation_passes == 224
    # 3 focal x 2 source modalities x 1 target modality (off-diagonal only).
    assert budget.n_causal_cells == 6
    # zero control + 2 positive alphas x 4 control kinds.
    assert budget.n_conditions_per_target == 9
    assert budget.causal_clean_passes == 6 * 16 * 6
    assert budget.causal_intervention_passes == 6 * 16 * 9 * 6
    assert budget.total_passes == 1152 + 224 + 576 + 5184
    assert budget.estimated_drive_mb > 0


def test_skipping_same_modality_cells_halves_the_causal_cost():
    common = dict(
        n_concepts=6, n_focal_concepts=3, modalities=("text", "image"),
        n_total_groups=112, n_capability_groups=48, n_targets_per_cell=16,
        alphas=(0.0, 0.25, 0.5),
    )
    off_diagonal = estimate_model_passes(**common, off_diagonal_only=True)
    everything = estimate_model_passes(**common, off_diagonal_only=False)

    assert off_diagonal.n_causal_cells * 2 == everything.n_causal_cells
    assert off_diagonal.total_passes < everything.total_passes


def test_widening_the_concept_set_costs_more_because_candidates_are_passes():
    four = estimate_model_passes(
        n_concepts=4, n_focal_concepts=3, modalities=("text", "image"),
        n_total_groups=112, n_capability_groups=48, n_targets_per_cell=16,
        alphas=(0.0, 0.25, 0.5),
    )
    six = estimate_model_passes(
        n_concepts=6, n_focal_concepts=3, modalities=("text", "image"),
        n_total_groups=112, n_capability_groups=48, n_targets_per_cell=16,
        alphas=(0.0, 0.25, 0.5),
    )
    assert six.capability_passes / four.capability_passes == 6 / 4
    assert six.n_candidates == 6


def test_the_budget_block_names_the_numbers_a_user_must_read():
    budget = estimate_model_passes(
        n_concepts=6, n_focal_concepts=3, modalities=("text", "image"),
        n_total_groups=112, n_capability_groups=48, n_targets_per_cell=16,
        alphas=(0.0, 0.25, 0.5),
    )
    text = format_budget(budget)
    for expected in (
        "TOTAL model forward passes",
        "estimated wall clock",
        "estimated Drive footprint",
        "capability passes",
        "causal intervention passes",
    ):
        assert expected in text, expected


def test_six_way_candidate_scoring_uses_complete_sequences_in_both_orders():
    questions = build_ordered_questions(CONCEPTS)
    assert len(questions) == 2, "canonical and reversed option orders"
    for concept in CONCEPTS:
        assert concept in questions[0]
    assert questions[0] != questions[1]


# ------------------------------------------------------- fingerprint cover


def robustness_config():
    return PilotConfig(
        mode="robustness",
        layers=(38,),
        causal_layers=(38,),
        modalities=("text", "image"),
        concepts=tuple(CONCEPTS),
        causal_concepts=tuple(FOCAL),
        alphas=(0.0, 0.25, 0.5),
        n_target_examples=8,
        subset_profile="image_unique",
        image_unique_targets=True,
        min_source_positive_images=8,
        min_source_negative_images=8,
        off_diagonal_causal_only=True,
    )


def build_fingerprint(**overrides):
    payload = dict(
        ranked_concepts=CONCEPTS,
        selected_concepts=CONCEPTS,
        focal_concepts=FOCAL,
        unrelated_controls=CONTROLS,
        derived_cache_fingerprint="sha256:cache",
        split_provenance_checksum="sha256:split",
        n_train_positive_images=8,
        n_train_negative_images=8,
        n_test_positive_images=8,
        n_test_negative_images=8,
        verdict_version="mmpilot.robustness_verdict.v1",
    )
    payload.update(overrides)
    return scientific_fingerprint(robustness_config(), **payload)


#: Every field the commission required the fingerprint to bind.
REQUIRED_FIELDS = (
    "ranked_concepts",
    "selected_concepts",
    "focal_concepts",
    "capability_protocol",
    "candidate_ordering_protocol",
    "n_candidates_scored",
    "capability_threshold",
    "derived_cache_fingerprint",
    "split_provenance_checksum",
    "independent_unit",
    "image_identity_rule_version",
    "max_groups_per_image",
    "representative_group_selection_version",
    "source_positive_selection_version",
    "source_negative_selection_version",
    "causal_target_selection_version",
    "n_distinct_train_positive_images",
    "n_distinct_train_negative_images",
    "n_distinct_heldout_positive_images",
    "n_distinct_heldout_negative_images",
    "unrelated_control_selection_version",
    "unrelated_controls",
    "intervention_controls",
    "alphas",
    "direction_top_k",
    "direction_normalization",
    "n_permutations",
    "verdict_version",
)


def test_the_fingerprint_binds_every_required_field():
    payload = build_fingerprint()
    missing = [field for field in REQUIRED_FIELDS if field not in payload]

    assert missing == [], missing
    assert payload["version"] == SELECTION_FINGERPRINT_VERSION
    assert payload["independent_unit"] == "image_id"
    assert payload["capability_protocol"] == PROMPT_PROTOCOL_VERSION
    assert payload["n_candidates_scored"] == 6
    assert payload["max_groups_per_image"] == 1


def test_the_ranking_order_is_bound_not_just_the_concept_set():
    """The order decides the focal concepts, so a reordering is a different
    experiment even when the set is identical."""
    reordered = list(reversed(CONCEPTS))
    assert build_fingerprint()["ranked_concepts"] != build_fingerprint(
        ranked_concepts=reordered
    )["ranked_concepts"]


def run_fingerprint(selection):
    return RunFingerprint(
        mode="robustness",
        model_repo_id="google/gemma-3n-e4b-it",
        model_revision="rev",
        processor_revision="rev",
        layers=(38,),
        lens_checksum="sha256:lens",
        manifest_checksum="sha256:manifest",
        split_id="seed",
        selection_config=selection,
    )


@pytest.mark.parametrize("field", REQUIRED_FIELDS)
def test_every_fingerprint_field_refuses_an_incompatible_resume(tmp_path, field):
    """Changing any one of them must stop the run, not merge two experiments."""
    baseline = build_fingerprint()
    store = UnitStore(tmp_path / field.replace("/", "_"), run_fingerprint(baseline))
    assert store.open() == "starting"

    mutated = dict(baseline)
    value = mutated[field]
    if isinstance(value, bool):
        mutated[field] = not value
    elif isinstance(value, (int, float)):
        mutated[field] = value + 1
    elif isinstance(value, list):
        mutated[field] = [*value, "extra"]
    elif isinstance(value, dict):
        mutated[field] = {**value, "extra": "value"}
    else:
        mutated[field] = f"{value}-changed"

    with pytest.raises(IncompatibleStateError, match="different configuration"):
        UnitStore(store.root, run_fingerprint(mutated)).open()


def test_a_four_concept_pilot_directory_cannot_be_resumed_as_six(tmp_path):
    """Candidate scoring changes from four-way to six-way, so capability and
    intervention artifacts are not comparable across the two studies."""
    four = build_fingerprint(
        selected_concepts=CONCEPTS[:4], ranked_concepts=CONCEPTS[:4], focal_concepts=CONCEPTS[:2]
    )
    store = UnitStore(tmp_path / "run", run_fingerprint(four))
    assert store.open() == "starting"

    with pytest.raises(IncompatibleStateError):
        UnitStore(store.root, run_fingerprint(build_fingerprint())).open()


def test_a_run_directory_without_a_selection_config_keeps_its_old_digest(tmp_path):
    """The completed pilot's directories must stay resumable.

    An empty ``selection_config`` is omitted from the digest entirely, so a
    fingerprint written before the field existed still matches.
    """
    legacy = RunFingerprint(
        mode="pilot",
        model_repo_id="google/gemma-3n-e4b-it",
        model_revision="rev",
        processor_revision="rev",
        layers=(35, 38),
        lens_checksum="sha256:lens",
        manifest_checksum="sha256:manifest",
        split_id="seed",
    )
    assert "selection_config" not in legacy.to_dict()
    store = UnitStore(tmp_path / "legacy", legacy)
    assert store.open() == "starting"
    assert UnitStore(store.root, legacy).open() == "resuming"


def test_adding_a_selection_config_to_a_legacy_directory_refuses(tmp_path):
    legacy = RunFingerprint(
        mode="pilot", model_repo_id="r", model_revision="v", processor_revision="v",
        layers=(38,), lens_checksum="sha256:l", manifest_checksum="sha256:m", split_id="s",
    )
    store = UnitStore(tmp_path / "legacy", legacy)
    store.open()

    with pytest.raises(IncompatibleStateError):
        UnitStore(store.root, run_fingerprint(build_fingerprint())).open()
