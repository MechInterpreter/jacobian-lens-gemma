# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""The independent layer-32 convergence-resolution study's own rules.

Every refusal is exercised, because a refusal that is never made to fire is a
comment. The tests are grouped by the thing they protect: the exclusion set, the
disjointness proof, the frozen thresholds, the predeclared sample size, the
stage plan, the fingerprint, and the read-only guarantee.
"""

import json

import pytest

from jlens.mmpilot.convergence import (
    AMBIGUOUS,
    CONVERGED,
    CONVERGENCE_CRITERION,
    NOT_CONVERGED,
)
from jlens.mmpilot.l32_resolution import (
    FOCAL_CONCEPTS,
    FROZEN_CRITERION_DIGEST,
    IDENTITY_FAMILIES,
    L32_INDEPENDENT_AMBIGUOUS,
    L32_INDEPENDENT_CONVERGED,
    L32_INDEPENDENT_NOT_CONVERGED,
    REFUSED_INVALID,
    RESOLUTION_FINGERPRINT_FIELDS,
    RESOLUTION_RAW_SWITCHES,
    RESOLUTION_RUN_PREFIX,
    SAMPLE_SIZE_RULE,
    SELECTED_CONCEPTS,
    ExclusionEvidenceMissing,
    ExclusionSet,
    PopulationNotIndependent,
    PseudoreplicationError,
    ResolutionRefused,
    assert_completed_runs_unchanged,
    assert_controls_recorded,
    assert_fresh_run_namespace,
    assert_one_unit_per_photograph,
    audit_population_disjointness,
    binomial_at_least,
    binomial_at_most,
    causal_synthesis,
    cell_precision,
    clean_predictions_from_capability,
    derive_resolution_gates,
    harvest_excluded_identities,
    independent_pool,
    plan_sample_size,
    render_report,
    resolution_fingerprint,
    resolution_verdict,
    resolve_excluded_media,
    run_tree_digest,
    selection_digest,
    stage_b_decision,
    stage_plan,
    wilson_interval,
)

# ------------------------------------------------------------------ fixtures


def _group(index, concept="cat"):
    return {
        "group_id": f"g_{index:04d}",
        "image_id": f"img_{index // 2:04d}",
        "caption": f"a photo of a {concept} number {index}",
        "audio_path": f"wavs/train/{index}.wav",
        "image_path": f"train2014/{index}.jpg",
        "concept": concept,
        "split": "train" if index % 2 else "test",
    }


@pytest.fixture
def groups():
    return [_group(index) for index in range(20)]


def _subset(rows):
    train = [row for row in rows if row["split"] == "train"]
    test = [row for row in rows if row["split"] != "train"]
    return {"splits": {"train": train, "test": test}}


def _completed_run(tmp_path, rows):
    root = tmp_path / "mml32_l32_followup_20260808T182717"
    units = root / "units" / "activation"
    units.mkdir(parents=True)
    for row in rows:
        (units / f"{row['group_id']}.json").write_text(
            json.dumps(
                {
                    "schema": "jlens.mmpilot.unit.v1",
                    "payload": {
                        "sample_id": f"{row['group_id']}::text",
                        "group_id": row["group_id"],
                        "image_id": row["image_id"],
                    },
                }
            ),
            encoding="utf-8",
        )
    return root


# ================================================================= exclusions


def test_the_harvest_reads_ids_out_of_nested_unit_payloads(tmp_path, groups):
    run_dir = _completed_run(tmp_path, groups[:6])
    exclusion = harvest_excluded_identities([run_dir])
    assert exclusion.group_ids == {row["group_id"] for row in groups[:6]}
    assert exclusion.image_ids == {row["image_id"] for row in groups[:6]}
    assert exclusion.sample_ids


def test_the_harvest_also_reads_a_fingerprints_sample_id_lists(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    (root / "fingerprint.json").write_text(
        json.dumps(
            {"source_sample_ids": ["g_a"], "target_sample_ids": ["g_b"]}
        ),
        encoding="utf-8",
    )
    exclusion = harvest_excluded_identities([root])
    assert exclusion.group_ids == {"g_a", "g_b"}


def test_a_harvest_that_found_nothing_refuses(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ExclusionEvidenceMissing, match="Independence would then be"):
        harvest_excluded_identities([empty])


def test_an_unreadable_unit_is_recorded_and_skipped(tmp_path, groups):
    run_dir = _completed_run(tmp_path, groups[:4])
    (run_dir / "units" / "activation" / "torn.json").write_text(
        "{not json", encoding="utf-8"
    )
    exclusion = harvest_excluded_identities([run_dir])
    assert exclusion.sources[0]["unreadable"]
    assert exclusion.group_ids


def test_resolving_media_recovers_recordings_and_captions(tmp_path, groups):
    run_dir = _completed_run(tmp_path, groups[:4])
    exclusion = harvest_excluded_identities([run_dir])
    assert not exclusion.audio_paths
    record = resolve_excluded_media(exclusion, groups)
    assert record["n_resolved_in_manifest"] == 4
    assert exclusion.audio_paths
    assert exclusion.captions


def test_excluding_a_photograph_excludes_its_sibling_captions(tmp_path, groups):
    # groups 0 and 1 share img_0000; the completed run used only group 0.
    run_dir = _completed_run(tmp_path, groups[:1])
    exclusion = harvest_excluded_identities([run_dir])
    resolve_excluded_media(exclusion, groups)
    assert groups[1]["group_id"] in exclusion.group_ids
    assert groups[1]["audio_path"] in exclusion.audio_paths


def test_the_pool_excludes_on_every_identity_family(tmp_path, groups):
    exclusion = ExclusionSet(
        image_ids={"img_0000"},
        group_ids={"g_0004"},
        audio_paths={"wavs/train/6.wav"},
        captions={groups[8]["caption"]},
    )
    pool, record = independent_pool(groups, exclusion)
    kept = {row["group_id"] for row in pool}
    assert "g_0000" not in kept and "g_0001" not in kept  # by image
    assert "g_0004" not in kept  # by group
    assert "g_0006" not in kept  # by recording
    assert "g_0008" not in kept  # by caption
    assert record["excluded_by_identity"]["image_ids"] == 2


def test_the_exclusion_digest_depends_on_which_ids_not_how_many():
    left = ExclusionSet(image_ids={"a", "b"})
    right = ExclusionSet(image_ids={"a", "c"})
    assert left.digest != right.digest


# =========================================================== disjointness


def test_a_disjoint_population_passes_and_records_its_digest(groups):
    subset = _subset(groups[10:])
    exclusion = ExclusionSet(
        image_ids={row["image_id"] for row in groups[:8]},
        group_ids={row["group_id"] for row in groups[:8]},
        audio_paths={row["audio_path"] for row in groups[:8]},
        captions={row["caption"] for row in groups[:8]},
    )
    record = audit_population_disjointness(subset, exclusion)
    assert record["disjoint"] is True
    assert record["families_checked"] == list(IDENTITY_FAMILIES)
    assert record["population_digest"].startswith("sha256:")


@pytest.mark.parametrize("family", IDENTITY_FAMILIES)
def test_an_overlap_in_any_single_family_refuses(groups, family):
    subset = _subset(groups[10:])
    field = {
        "image_ids": "image_id",
        "group_ids": "group_id",
        "audio_paths": "audio_path",
        "captions": "caption",
    }[family]
    exclusion = ExclusionSet(**{family: {groups[10][field]}})
    with pytest.raises(PopulationNotIndependent, match=family):
        audit_population_disjointness(subset, exclusion)


def test_the_disjointness_record_can_be_taken_without_raising(groups):
    subset = _subset(groups[10:])
    exclusion = ExclusionSet(image_ids={groups[10]["image_id"]})
    record = audit_population_disjointness(subset, exclusion, require=False)
    assert record["disjoint"] is False
    assert record["failed_families"] == ["image_ids"]


def test_two_groups_of_one_photograph_are_pseudoreplication(groups):
    with pytest.raises(PseudoreplicationError, match="photograph"):
        assert_one_unit_per_photograph(_subset(groups))


def test_one_group_per_photograph_passes(groups):
    unique = [row for index, row in enumerate(groups) if index % 2 == 0]
    record = assert_one_unit_per_photograph(_subset(unique))
    assert record["passed"] is True
    assert record["n_units"] == record["n_distinct_images"]
    assert record["n_units"] == record["n_distinct_recordings"]


def test_a_reused_recording_is_refused(groups):
    unique = [dict(row) for index, row in enumerate(groups) if index % 2 == 0]
    unique[1]["audio_path"] = unique[0]["audio_path"]
    with pytest.raises(PseudoreplicationError, match="recording"):
        assert_one_unit_per_photograph(_subset(unique))


def test_the_selection_digest_is_order_independent(groups):
    unique = [row for index, row in enumerate(groups) if index % 2 == 0]
    forward = selection_digest(_subset(unique))
    backward = selection_digest(_subset(list(reversed(unique))))
    assert forward == backward


# ============================================================ frozen numbers


def test_the_frozen_criterion_digest_is_the_one_this_module_pins():
    assert CONVERGENCE_CRITERION.digest == FROZEN_CRITERION_DIGEST


def test_the_frozen_thresholds_are_exactly_the_predeclared_ones():
    assert CONVERGENCE_CRITERION.converged_min_clean_agreement == 0.90
    assert CONVERGENCE_CRITERION.converged_min_target_accuracy == 0.90
    assert CONVERGENCE_CRITERION.converged_max_median_rank == 1.0
    assert CONVERGENCE_CRITERION.not_converged_max_clean_agreement == 0.50
    assert CONVERGENCE_CRITERION.not_converged_max_target_accuracy == 0.50
    assert CONVERGENCE_CRITERION.not_converged_min_median_rank == 2.0


def test_the_concept_sets_are_frozen_in_ranking_order():
    assert SELECTED_CONCEPTS == (
        "zebra", "cat", "toilet", "giraffe", "bird", "microwave",
    )
    assert FOCAL_CONCEPTS == ("zebra", "cat", "toilet")
    assert list(FOCAL_CONCEPTS) == list(SELECTED_CONCEPTS[:3])


# ========================================================== sample size


def test_wilson_beats_wald_at_the_upper_bar():
    interval = wilson_interval(k=18, n=20)
    assert interval["high"] <= 1.0
    assert 0.0 <= interval["low"] <= interval["point"] <= interval["high"]


def test_the_exact_binomial_tails_agree_with_each_other():
    assert binomial_at_most(20, 20, 0.4) == pytest.approx(1.0)
    assert binomial_at_least(0, 20, 0.4) == pytest.approx(1.0)
    assert binomial_at_most(9, 20, 0.5) + binomial_at_least(
        10, 20, 0.5
    ) == pytest.approx(1.0)


def test_the_plan_meets_its_own_targets_at_the_expected_admissible_count():
    plan = plan_sample_size()
    assert plan.adequate is True
    precision = cell_precision(
        plan.images_per_concept * SAMPLE_SIZE_RULE.expected_admissible_focal
    )
    assert precision["wilson_half_width_at_p_0.5"] <= SAMPLE_SIZE_RULE.target_half_width
    assert precision["power_to_observe_not_converged"] >= SAMPLE_SIZE_RULE.min_power
    assert precision["power_to_observe_converged"] >= SAMPLE_SIZE_RULE.min_power


def test_the_plan_takes_the_smallest_adequate_rung():
    plan = plan_sample_size()
    smaller = [
        rung for rung in plan.ladder_considered if rung < plan.images_per_concept
    ]
    for rung in smaller:
        precision = cell_precision(
            rung * SAMPLE_SIZE_RULE.expected_admissible_focal
        )
        assert not (
            precision["meets_half_width_target"] and precision["meets_power_target"]
        ), f"rung {rung} was adequate and should have been chosen"


def test_the_plan_reports_precision_at_every_admissible_count():
    plan = plan_sample_size()
    assert sorted(plan.precision_by_admissible) == ["1", "2", "3"]
    assert plan.to_dict()["plan_digest"].startswith("sha256:")


def test_the_stopping_rule_forbids_adding_units_after_a_look():
    plan = plan_sample_size().to_dict()
    assert "no interim look" in plan["stopping_rule"]
    assert "sampling to a foregone conclusion" in plan["stopping_rule"]


# ============================================================== stage plan


def test_the_stage_plan_is_conditional_and_says_why():
    plan = stage_plan()
    assert plan["mode"] == "conditional"
    assert plan["efficiency_gate_not_suppression"] is True
    assert plan["all_stage_a_outcomes_reported"] is True
    assert len(plan["stages"]) == 2


@pytest.mark.parametrize(
    ("verdict", "gate_met"),
    [
        (L32_INDEPENDENT_NOT_CONVERGED, True),
        (L32_INDEPENDENT_CONVERGED, False),
        (L32_INDEPENDENT_AMBIGUOUS, False),
        (REFUSED_INVALID, False),
    ],
)
def test_the_stage_b_gate_opens_only_for_not_converged(verdict, gate_met):
    decision = stage_b_decision(
        verdict=verdict, controls_passed=True, requested=True, budget_confirmed=True
    )
    assert decision["gate_met"] is gate_met


def test_failing_controls_close_the_stage_b_gate():
    decision = stage_b_decision(
        verdict=L32_INDEPENDENT_NOT_CONVERGED,
        controls_passed=False,
        requested=True,
        budget_confirmed=True,
    )
    assert decision["gate_met"] is False


def test_running_stage_b_against_the_gate_is_stamped_as_an_override():
    decision = stage_b_decision(
        verdict=L32_INDEPENDENT_AMBIGUOUS,
        controls_passed=True,
        requested=True,
        budget_confirmed=True,
    )
    assert decision["runs"] is True
    assert decision["gate_overridden"] is True
    assert "OVERRIDE" in decision["statement"]


def test_an_unconfirmed_budget_does_not_run_stage_b():
    decision = stage_b_decision(
        verdict=L32_INDEPENDENT_NOT_CONVERGED,
        controls_passed=True,
        requested=True,
        budget_confirmed=False,
    )
    assert decision["runs"] is False


def test_the_gates_are_derived_from_the_raw_switches():
    on = dict.fromkeys(RESOLUTION_RAW_SWITCHES, True)
    assert derive_resolution_gates(on)["STAGE_B_REQUESTED"] is True
    off = {**on, "CONFIRM_MODEL_LOAD": False}
    assert derive_resolution_gates(off)["MODEL_STAGE_ENABLED"] is False
    assert derive_resolution_gates(off)["STAGE_B_REQUESTED"] is False


def test_a_missing_switch_refuses_rather_than_defaulting_to_false():
    from jlens.mmpilot.stage_gates import MissingStageSwitch

    partial = dict.fromkeys(RESOLUTION_RAW_SWITCHES[:-1], True)
    with pytest.raises(MissingStageSwitch, match="CONFIRM_STAGE_B_BUDGET"):
        derive_resolution_gates(partial)


# ================================================================= controls


def _controls(layer=32, *, variants=("shuffled_target_labels",
                                     "permuted_candidate_tokens",
                                     "permuted_activations"), passed=True,
              informative=True):
    return {
        "per_layer": {
            str(layer): {
                "controls": {
                    name: {
                        "compared_field": "clean_agreement_argmax",
                        "primary_value": 0.9,
                        "control_value": 0.2,
                        "chance_rate": 0.167,
                        "margin": 0.15,
                        "primary_is_informative": informative,
                        "passed": passed,
                        "reason": "test",
                        "expectation": "test",
                    }
                    for name in variants
                }
            }
        },
        "all_controls_passed": passed,
        "failed_controls": [] if passed else ["L32:permuted_activations"],
    }


def test_complete_passing_controls_are_accepted():
    record = assert_controls_recorded(_controls())
    assert record["passed"] is True
    assert record["missing_or_empty"] == []


def test_a_missing_control_variant_refuses_even_when_the_flag_says_passed():
    controls = _controls(variants=("shuffled_target_labels",))
    assert controls["all_controls_passed"] is True
    with pytest.raises(ResolutionRefused, match="permuted_activations"):
        assert_controls_recorded(controls)


def test_a_control_that_reproduced_the_primary_refuses():
    with pytest.raises(ResolutionRefused, match="reproduced the primary"):
        assert_controls_recorded(_controls(passed=False))


def test_an_uninformative_primary_is_reported_but_never_refused():
    # A genuinely non-converged layer sits at chance, so its controls have
    # nothing to reproduce. Refusing on that would make NOT_CONVERGED
    # unreachable while leaving CONVERGED reachable.
    record = assert_controls_recorded(_controls(informative=False))
    assert record["passed"] is True
    assert len(record["uninformative"]) == 3


def test_a_layer_with_no_control_record_at_all_refuses():
    with pytest.raises(ResolutionRefused):
        assert_controls_recorded({"per_layer": {}, "all_controls_passed": True})


# ============================================================ clean answers


def test_clean_predictions_come_out_of_capability_units():
    records = [
        {"sample_id": "a::text", "prediction": "cat"},
        {"sample_id": "b::image", "prediction": "zebra"},
    ]
    assert clean_predictions_from_capability(records) == {
        "a::text": "cat",
        "b::image": "zebra",
    }


def test_two_capability_units_disagreeing_about_one_sample_refuses():
    records = [
        {"sample_id": "a::text", "prediction": "cat"},
        {"sample_id": "a::text", "prediction": "dog"},
    ]
    with pytest.raises(ResolutionRefused, match="disagree"):
        clean_predictions_from_capability(records)


# ================================================================= verdicts


def _convergence(classification, *, layer=32, n=24, distinct=3):
    cell = {
        "n": n,
        "n_distinct_predictions": distinct,
        "n_distinct_images": n,
    }
    return {
        "layer": layer,
        "criterion_digest": FROZEN_CRITERION_DIGEST,
        "classification": {"classification": classification},
        "summary": {
            "per_layer": {
                str(layer): {
                    "per_modality": {
                        modality: dict(cell)
                        for modality in CONVERGENCE_CRITERION.required_modalities
                    }
                }
            }
        },
    }


def _verdict_inputs(classification, **overrides):
    payload = {
        "integrity": {"verdict": "PASSED"},
        "convergence": _convergence(classification),
        "controls": {"passed": True, "missing_or_empty": [], "failing": []},
        "disjointness": {"disjoint": True, "failed_families": []},
        "pseudoreplication": {"passed": True, "n_units": 24, "n_distinct_images": 24},
        "sample_plan": {"plan_digest": "sha256:plan"},
        "head_agreement": {"passed": True, "comparison_ran": True,
                           "matches_model_unembed": True},
        "admissibility": {"eligible_concepts": ["cat", "toilet"]},
    }
    payload.update(overrides)
    return payload


@pytest.mark.parametrize(
    ("classification", "expected"),
    [
        (NOT_CONVERGED, L32_INDEPENDENT_NOT_CONVERGED),
        (CONVERGED, L32_INDEPENDENT_CONVERGED),
        (AMBIGUOUS, L32_INDEPENDENT_AMBIGUOUS),
    ],
)
def test_every_classification_reaches_its_own_verdict(classification, expected):
    verdict = resolution_verdict(**_verdict_inputs(classification))
    assert verdict["verdict"] == expected
    assert verdict["failed_validity_clauses"] == []


def test_all_three_outcomes_are_reported_in_the_same_shape():
    shapes = {
        classification: set(
            resolution_verdict(**_verdict_inputs(classification))
        )
        for classification in (NOT_CONVERGED, CONVERGED, AMBIGUOUS)
    }
    assert len({frozenset(keys) for keys in shapes.values()}) == 1


@pytest.mark.parametrize(
    ("override", "clause"),
    [
        ({"integrity": {"verdict": "REFUSED"}}, "lens_integrity_passed"),
        (
            {"head_agreement": {"passed": False, "comparison_ran": False}},
            "native_head_agrees_with_model_unembed",
        ),
        (
            {"disjointness": {"disjoint": False, "failed_families": ["image_ids"]}},
            "population_disjoint_from_completed_run",
        ),
        (
            {"pseudoreplication": {"passed": False}},
            "one_unit_per_photograph",
        ),
        ({"sample_plan": {}}, "sample_size_predeclared"),
        (
            {"controls": {"passed": False, "missing_or_empty": ["x"], "failing": []}},
            "controls_present_and_passing",
        ),
        (
            {"admissibility": {"eligible_concepts": []}},
            "at_least_one_admissible_focal_concept",
        ),
    ],
)
def test_a_failed_validity_clause_produces_refused_invalid(override, clause):
    verdict = resolution_verdict(**_verdict_inputs(NOT_CONVERGED, **override))
    assert verdict["verdict"] == REFUSED_INVALID
    assert clause in verdict["failed_validity_clauses"]


def test_a_changed_criterion_digest_produces_refused_invalid():
    inputs = _verdict_inputs(NOT_CONVERGED)
    inputs["convergence"]["criterion_digest"] = "sha256:something-else"
    verdict = resolution_verdict(**inputs)
    assert verdict["verdict"] == REFUSED_INVALID
    assert "criterion_digest_unchanged" in verdict["failed_validity_clauses"]


def test_a_degenerate_readout_produces_refused_invalid():
    inputs = _verdict_inputs(NOT_CONVERGED)
    inputs["convergence"] = _convergence(NOT_CONVERGED, distinct=1)
    verdict = resolution_verdict(**inputs)
    assert verdict["verdict"] == REFUSED_INVALID
    assert "outputs_finite_and_nondegenerate" in verdict["failed_validity_clauses"]


def test_an_undersized_cell_produces_refused_invalid():
    inputs = _verdict_inputs(NOT_CONVERGED)
    inputs["convergence"] = _convergence(NOT_CONVERGED, n=2)
    verdict = resolution_verdict(**inputs)
    assert verdict["verdict"] == REFUSED_INVALID


def test_the_verdict_reads_the_layer_from_the_measurement_not_a_constant():
    # The MOCK world stands in for layer 32 at layer 1. A verdict that looked
    # up per_layer["32"] would find nothing and call a healthy cell degenerate.
    inputs = _verdict_inputs(NOT_CONVERGED)
    inputs["convergence"] = _convergence(NOT_CONVERGED, layer=1)
    verdict = resolution_verdict(**inputs)
    assert verdict["verdict"] == L32_INDEPENDENT_NOT_CONVERGED
    assert verdict["layer"] == 1


def test_an_excluded_focal_concept_is_never_replaced():
    verdict = resolution_verdict(**_verdict_inputs(AMBIGUOUS))
    assert verdict["inadmissible_focal_concepts"] == ["zebra"]
    assert verdict["concepts_replaced"] is False
    assert set(verdict["admissible_focal_concepts"]) < set(FOCAL_CONCEPTS)


# ================================================================ synthesis


def test_not_converged_alone_does_not_license_a_pre_convergence_claim():
    synthesis = causal_synthesis(
        verdict={"verdict": L32_INDEPENDENT_NOT_CONVERGED},
        completed_causal={"verdict": "WEAK"},
        stage_b={"runs": False},
    )
    assert synthesis["combined_pre_convergence_claim"] == (
        "NOT_ESTABLISHED_CAUSAL_REPLICATION_MISSING"
    )
    assert synthesis["additional_causal_replication_required"]
    assert "different population" in synthesis["statement"]


def test_a_stage_b_run_on_the_same_population_can_complete_the_claim():
    synthesis = causal_synthesis(
        verdict={"verdict": L32_INDEPENDENT_NOT_CONVERGED},
        completed_causal={"verdict": "WEAK"},
        stage_b={"runs": True, "causal_verdict": "SUPPORTED"},
    )
    assert synthesis["combined_pre_convergence_claim"] == "SUPPORTED"
    assert synthesis["additional_causal_replication_required"] == []


def test_an_overridden_stage_b_at_an_ambiguous_layer_still_needs_replication():
    synthesis = causal_synthesis(
        verdict={"verdict": L32_INDEPENDENT_AMBIGUOUS},
        completed_causal={"verdict": "WEAK"},
        stage_b={"runs": True, "causal_verdict": "SUPPORTED"},
    )
    assert synthesis["combined_pre_convergence_claim"] == (
        "NOT_ESTABLISHED_CONVERGENCE_HALF_AMBIGUOUS"
    )
    assert synthesis["additional_causal_replication_required"]


def test_a_converged_layer_rules_the_claim_out():
    synthesis = causal_synthesis(
        verdict={"verdict": L32_INDEPENDENT_CONVERGED},
        completed_causal={"verdict": "WEAK"},
        stage_b={"runs": False},
    )
    assert synthesis["combined_pre_convergence_claim"] == "RULED_OUT_AT_LAYER_32"


def test_the_synthesis_never_claims_environmental_audio_or_a_coordinate_swap():
    synthesis = causal_synthesis(
        verdict={"verdict": L32_INDEPENDENT_AMBIGUOUS},
        completed_causal={"verdict": "WEAK"},
        stage_b={"runs": False},
    )
    joined = " ".join(synthesis["never_claimed"]).lower()
    assert "environmental" in joined
    assert "two-coordinate swap" in joined


# ============================================================== fingerprint


def _fingerprint_fields():
    return {name: f"value-{name}" for name in RESOLUTION_FINGERPRINT_FIELDS}


def test_the_fingerprint_binds_every_declared_field():
    record = resolution_fingerprint(**_fingerprint_fields())
    assert record["fingerprint_digest"].startswith("sha256:")
    for name in RESOLUTION_FINGERPRINT_FIELDS:
        assert name in record


def test_a_missing_fingerprint_field_refuses():
    fields = _fingerprint_fields()
    fields.pop("exclusion_run_checksum")
    with pytest.raises(ResolutionRefused, match="exclusion_run_checksum"):
        resolution_fingerprint(**fields)


def test_an_unknown_fingerprint_field_refuses():
    with pytest.raises(ResolutionRefused, match="unknown"):
        resolution_fingerprint(**_fingerprint_fields(), extra="x")


@pytest.mark.parametrize("field", RESOLUTION_FINGERPRINT_FIELDS)
def test_changing_any_scientific_field_changes_the_fingerprint(field):
    base = resolution_fingerprint(**_fingerprint_fields())
    changed = resolution_fingerprint(**{**_fingerprint_fields(), field: "moved"})
    assert changed["fingerprint_digest"] != base["fingerprint_digest"]


def test_the_exclusion_set_and_the_population_are_both_in_the_fingerprint():
    assert "exclusion_run_checksum" in RESOLUTION_FINGERPRINT_FIELDS
    assert "selected_population_digest" in RESOLUTION_FINGERPRINT_FIELDS
    assert "sample_size_plan_digest" in RESOLUTION_FINGERPRINT_FIELDS
    assert "convergence_criterion_digest" in RESOLUTION_FINGERPRINT_FIELDS


# ========================================================= run namespace


def test_a_run_directory_inside_a_completed_namespace_is_refused(tmp_path):
    bad = tmp_path / "mmaudio_native_audio_transfer_20260806T144822" / "sub"
    with pytest.raises(ResolutionRefused, match="completed run namespace"):
        assert_fresh_run_namespace(bad, protected_prefixes=("mmaudio_",))


def test_a_run_directory_outside_the_family_prefix_is_refused(tmp_path):
    with pytest.raises(ResolutionRefused, match=RESOLUTION_RUN_PREFIX):
        assert_fresh_run_namespace(tmp_path / "something_else",
                                   protected_prefixes=("mmaudio_",))


def test_the_studys_own_directory_is_accepted(tmp_path):
    good = tmp_path / f"{RESOLUTION_RUN_PREFIX}_real_20260809T000000"
    assert assert_fresh_run_namespace(good, protected_prefixes=("mmaudio_",))


# ====================================================== read-only guarantee


def test_an_unchanged_completed_run_passes_the_immutability_proof(tmp_path, groups):
    run_dir = _completed_run(tmp_path, groups[:4])
    before = [run_tree_digest(run_dir)]
    after = [run_tree_digest(run_dir)]
    record = assert_completed_runs_unchanged(before, after)
    assert record["unchanged"] is True


def test_a_file_appearing_in_a_completed_run_is_refused(tmp_path, groups):
    run_dir = _completed_run(tmp_path, groups[:4])
    before = [run_tree_digest(run_dir)]
    (run_dir / "intruder.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ResolutionRefused, match="changed while this study ran"):
        assert_completed_runs_unchanged(before, [run_tree_digest(run_dir)])


# ================================================================== report


def test_the_report_states_the_verdict_and_the_replication_requirement():
    from jlens.mmpilot.l32_resolution import build_summary

    verdict = resolution_verdict(**_verdict_inputs(NOT_CONVERGED))
    synthesis = causal_synthesis(
        verdict=verdict,
        completed_causal={"verdict": "WEAK"},
        stage_b={"runs": False},
    )
    summary = build_summary(
        fingerprint=resolution_fingerprint(**_fingerprint_fields()),
        verdict=verdict,
        synthesis=synthesis,
        sample_plan=plan_sample_size().to_dict(),
        disjointness={"disjoint": True, "families_checked": list(IDENTITY_FAMILIES)},
        pseudoreplication={"passed": True},
        pool={},
        exclusion={},
        convergence={},
        controls={},
        capability={},
        stage_plan_record=stage_plan(),
        stage_b={"runs": False, "gate_met": True, "gate_overridden": False},
        immutability={"unchanged": True},
        cache={},
        resume={},
        mode="real",
    )
    text = render_report(summary)
    assert L32_INDEPENDENT_NOT_CONVERGED in text
    assert "Causal replication still required" in text
    assert "never replaced" in text


def test_the_report_survives_the_phrasing_rule():
    from jlens.mmpilot.l32_followup import assert_report_phrasing
    from jlens.mmpilot.l32_resolution import build_summary

    verdict = resolution_verdict(**_verdict_inputs(AMBIGUOUS))
    summary = build_summary(
        fingerprint=resolution_fingerprint(**_fingerprint_fields()),
        verdict=verdict,
        synthesis=causal_synthesis(
            verdict=verdict,
            completed_causal={"verdict": "WEAK"},
            stage_b={"runs": False},
        ),
        sample_plan=plan_sample_size().to_dict(),
        disjointness={"disjoint": True, "families_checked": list(IDENTITY_FAMILIES)},
        pseudoreplication={"passed": True},
        pool={},
        exclusion={},
        convergence={},
        controls={},
        capability={},
        stage_plan_record=stage_plan(),
        stage_b={"runs": False, "gate_met": False, "gate_overridden": False},
        immutability={"unchanged": True},
        cache={},
        resume={},
        mode="real",
    )
    assert assert_report_phrasing(render_report(summary))["passed"] is True
