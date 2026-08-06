# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""The capability gate on scientific claims, and what it must never do.

These tests are written against the situation that produced the rule. The real
native-audio Stage A returned ``zebra`` at 8/8 text, 8/8 image and **5/8 spoken
audio** against a 70% threshold, and Stage B then counted ``zebra``'s
``spoken_audio -> text`` and ``spoken_audio -> image`` cells as scientific
support for the three-modality claim.

Two things are being pinned here at once, and they pull in opposite directions:

* an ineligible concept's cells must **never** become evidence — not a
  supporting cell, not a bidirectional pair, not a GO, not a WEAK GO;
* and they must **never disappear**. They were predeclared, they were measured,
  and deleting them would be a different kind of dishonesty than counting them.

The third thing, which has no test that can prove it but has one that can catch
its footprint: an excluded concept is not replaced. ``eligible_concepts`` is
always a subset of the fixed set it was asked about.
"""

import pytest

from jlens.mmpilot.admissibility import (
    CAPABILITY_INELIGIBLE,
    CLAIM_ADMISSIBILITY_RULE_VERSION,
    CapabilityTableMissing,
    admissibility_rule_record,
    annotate_causal_cells,
    claim_admissibility,
    concept_admissibility,
)
from jlens.mmpilot.tri_modal import (
    ALL_PAIRS,
    AUDIO_PAIRS,
    THREE_MODALITY_GO,
    THREE_MODALITY_NO_GO,
    THREE_MODALITY_WEAK_GO,
    TRANSFER_SUPPORTED,
    TRANSFER_UNSUPPORTED,
    TriModalThresholds,
    audio_capability_verdict,
    causal_transfer_verdict,
    estimate_stage_passes,
    evaluate_causal_cells,
    format_focal_capability_gate,
    overall_verdict,
    render_report,
    replication_verdict,
    representational_transfer_verdict,
)

MODALITIES = ("text", "image", "spoken_audio")
THRESHOLDS = TriModalThresholds(
    required_positive_images_per_cell=2, required_negative_images_per_cell=2
)

#: The observed Stage A capability table, transcribed. ``zebra`` is the concept
#: whose audio-related cells were wrongly counted; ``giraffe`` at 6/8 is the
#: boundary case a 70% threshold must accept.
OBSERVED_COUNTS = {
    "bird": {"text": 8, "image": 8, "spoken_audio": 8},
    "cat": {"text": 8, "image": 7, "spoken_audio": 8},
    "giraffe": {"text": 8, "image": 8, "spoken_audio": 6},
    "microwave": {"text": 8, "image": 8, "spoken_audio": 8},
    "toilet": {"text": 8, "image": 8, "spoken_audio": 7},
    "zebra": {"text": 8, "image": 8, "spoken_audio": 5},
}


def _capability(counts=None, *, threshold=0.7, n=8):
    counts = OBSERVED_COUNTS if counts is None else counts
    per_concept = {
        concept: {
            modality: {
                "n": n,
                "n_correct": n_correct,
                "accuracy": n_correct / n,
                "passed": n_correct / n >= threshold,
            }
            for modality, n_correct in per_modality.items()
        }
        for concept, per_modality in counts.items()
    }
    retained = sorted(
        concept
        for concept, per_modality in per_concept.items()
        if all(entry["passed"] for entry in per_modality.values())
    )
    return {
        "threshold": threshold,
        "modalities_evaluated": list(MODALITIES),
        "per_concept": per_concept,
        "retained_concepts": retained,
        "text_image_retained_concepts": sorted(
            concept
            for concept, per_modality in per_concept.items()
            if per_modality["text"]["passed"] and per_modality["image"]["passed"]
        ),
        "n_records": n * len(per_concept),
    }


CAPABILITY = _capability()


# ------------------------------------------------------- 1, 2, 3: the rule


def test_a_concept_failing_spoken_audio_at_five_of_eight_is_ineligible():
    decision = claim_admissibility(
        concept="zebra",
        source_modality="spoken_audio",
        target_modality="text",
        capability=CAPABILITY,
    )
    assert decision["admissible"] is False
    assert decision["label"] == CAPABILITY_INELIGIBLE
    assert decision["failing_modalities"] == ["spoken_audio"]
    assert decision["observed"]["spoken_audio"]["n_correct"] == 5
    assert decision["observed"]["spoken_audio"]["accuracy"] == pytest.approx(0.625)
    assert "62.5% < 70.0%" in decision["rejection_reason"]
    # Its text and image capability was fine, and the decision says so rather
    # than implying the concept was unreadable everywhere.
    assert decision["observed"]["text"]["passed"] is True
    assert decision["observed"]["image"]["passed"] is True


def test_exactly_six_of_eight_clears_a_seventy_percent_threshold():
    decision = claim_admissibility(
        concept="giraffe",
        source_modality="spoken_audio",
        target_modality="image",
        capability=CAPABILITY,
    )
    assert decision["observed"]["spoken_audio"]["accuracy"] == pytest.approx(0.75)
    assert decision["admissible"] is True
    assert decision["rejection_reason"] is None
    # 5/8 is the nearest failing value below it; the boundary is not a rounding
    # accident in either direction.
    assert (
        claim_admissibility(
            concept="zebra",
            source_modality="spoken_audio",
            target_modality="image",
            capability=CAPABILITY,
        )["admissible"]
        is False
    )


def test_a_concept_passing_all_three_modalities_is_admissible_in_every_direction():
    for pair in ALL_PAIRS:
        source, target = pair.split("->")
        decision = claim_admissibility(
            concept="bird",
            source_modality=source,
            target_modality=target,
            capability=CAPABILITY,
        )
        assert decision["admissible"] is True, pair
        assert decision["required_modalities"] == list(MODALITIES)
        assert decision["rule_version"] == CLAIM_ADMISSIBILITY_RULE_VERSION


def test_there_is_no_admissible_by_default_path():
    with pytest.raises(CapabilityTableMissing):
        claim_admissibility(
            concept="zebra",
            source_modality="text",
            target_modality="image",
            capability=None,
        )


# --------------------------------------------------- 4: pair-specific fields


def test_pair_specific_mode_requires_only_the_modalities_the_cell_spans():
    principal = claim_admissibility(
        concept="zebra",
        source_modality="text",
        target_modality="image",
        capability=CAPABILITY,
        principal_three_modality=True,
    )
    pair_specific = claim_admissibility(
        concept="zebra",
        source_modality="text",
        target_modality="image",
        capability=CAPABILITY,
        principal_three_modality=False,
    )
    assert principal["required_modalities"] == list(MODALITIES)
    assert principal["admissible"] is False
    assert pair_specific["required_modalities"] == ["text", "image"]
    assert pair_specific["pair"] == "text->image"
    assert pair_specific["source_modality"] == "text"
    assert pair_specific["target_modality"] == "image"
    # zebra can read text and image, so a text->image claim about it survives
    # the weaker rule — and the principal three-modality verdict still refuses
    # it, which is the whole point of using the stronger rule there.
    assert pair_specific["admissible"] is True
    assert pair_specific["rejection_reason"] is None

    audio_pair = claim_admissibility(
        concept="zebra",
        source_modality="spoken_audio",
        target_modality="text",
        capability=CAPABILITY,
        principal_three_modality=False,
    )
    assert audio_pair["required_modalities"] == ["spoken_audio", "text"]
    assert audio_pair["admissible"] is False
    assert audio_pair["failing_modalities"] == ["spoken_audio"]


# ------------------------------------------------------- causal-cell fixture


def _row(
    *,
    concept,
    pair,
    control_kind,
    effect,
    layer=35,
    alpha=0.5,
    n_positive=4,
    n_negative=4,
):
    source, target = pair.split("->")
    return {
        "concept": concept,
        "source_modality": source,
        "target_modality": target,
        "pair": pair,
        "off_diagonal": True,
        "layer": layer,
        "control_kind": control_kind,
        "alpha": alpha,
        "n": n_positive + n_negative,
        "n_distinct_images": n_positive + n_negative,
        "n_positive_images": n_positive,
        "n_negative_images": n_negative,
        "mean_signed_target_effect": effect,
        "mean_signed_margin_effect": effect,
        "fraction_expected_sign": 1.0,
        "mean_abs_unrelated_change": 0.01,
        "mean_activation_norm_ratio": 1.0,
        "n_prediction_changes": 0,
    }


def _interventions(concept_pairs, *, layer=35, effect=0.5):
    """``{concept: [pair, ...]}`` -> an image-level table where each cell passes."""
    rows = []
    for concept, pairs in concept_pairs.items():
        for pair in pairs:
            rows.append(
                _row(
                    concept=concept,
                    pair=pair,
                    control_kind="source_concept",
                    effect=effect,
                    layer=layer,
                )
            )
            for control, value in (
                ("random_norm_matched", 0.01),
                ("unrelated_concept", 0.02),
                ("raw_residual_difference", 0.05),
            ):
                rows.append(
                    _row(
                        concept=concept,
                        pair=pair,
                        control_kind=control,
                        effect=value,
                        layer=layer,
                    )
                )
    return {"rows": rows}


#: The observed Stage B shape: zebra passing two audio directions, cat passing
#: one, toilet passing all four.
OBSERVED_CELLS = {
    "zebra": ["spoken_audio->text", "spoken_audio->image"],
    "cat": ["spoken_audio->image"],
    "toilet": list(AUDIO_PAIRS),
}
FOCAL = ["zebra", "cat", "toilet"]


def _verdict(**overrides):
    kwargs = dict(
        layer=35,
        focal_concepts=FOCAL,
        thresholds=THRESHOLDS,
        name="L35_CAUSAL_TRANSFER",
        capability=CAPABILITY,
    )
    kwargs.update(overrides)
    interventions = kwargs.pop("interventions", _interventions(OBSERVED_CELLS))
    return causal_transfer_verdict(interventions, **kwargs)


# ------------------------------------- 5, 6, 7: kept, but never made evidence


def test_a_failed_capability_cell_stays_in_the_raw_table():
    cells = evaluate_causal_cells(
        _interventions(OBSERVED_CELLS),
        layer=35,
        focal_concepts=FOCAL,
        thresholds=THRESHOLDS,
        capability=CAPABILITY,
    )
    # Every (focal concept x direction) is present: nothing is dropped for
    # being inadmissible.
    assert len(cells) == len(FOCAL) * len(ALL_PAIRS)
    zebra = [c for c in cells if c["concept"] == "zebra" and c["evaluated"]]
    assert {c["pair"] for c in zebra} == set(OBSERVED_CELLS["zebra"])
    # The measurement is intact and still says the cell cleared its controls...
    assert all(c["passes"] for c in zebra)
    assert all(c["mean_signed_target_effect"] == 0.5 for c in zebra)
    # ...and it is labelled, with the arithmetic, and counted for nothing.
    assert all(c["capability_admissible"] is False for c in zebra)
    assert all(c["capability_label"] == CAPABILITY_INELIGIBLE for c in zebra)
    assert all(c["counted_toward_verdict"] is False for c in zebra)
    assert all("5/8 = 62.5%" in c["capability_rejection_reason"] for c in zebra)


def test_a_failed_capability_cell_never_enters_a_supporting_cell_list():
    verdict = _verdict()
    supporting = {(c["concept"], c["pair"]) for c in verdict["audio_cells_supporting_a_claim"]}
    assert not any(concept == "zebra" for concept, _ in supporting)
    assert supporting == {
        ("cat", "spoken_audio->image"),
        ("toilet", "text->spoken_audio"),
        ("toilet", "spoken_audio->text"),
        ("toilet", "image->spoken_audio"),
        ("toilet", "spoken_audio->image"),
    }
    # And it is reported, loudly, in its own list rather than being silent.
    inadmissible = {
        (c["concept"], c["pair"])
        for c in verdict["audio_cells_measured_but_inadmissible"]
    }
    assert inadmissible == {
        ("zebra", "spoken_audio->text"),
        ("zebra", "spoken_audio->image"),
    }
    # The unfiltered measured list still contains zebra.
    measured = {(c["concept"], c["pair"]) for c in verdict["audio_cells_passing"]}
    assert ("zebra", "spoken_audio->text") in measured


def test_a_failed_capability_concept_never_creates_bidirectional_support():
    # zebra alone would look bidirectional-ish; give it all four directions and
    # it still must not appear.
    verdict = _verdict(
        interventions=_interventions(
            {"zebra": list(AUDIO_PAIRS), "cat": ["spoken_audio->image"]}
        ),
        focal_concepts=["zebra", "cat"],
    )
    assert verdict["concepts_transferring_both_audio_directions"] == []
    assert verdict["audio_cells_supporting_a_claim"] == [
        {"concept": "cat", "pair": "spoken_audio->image"}
    ]
    assert verdict["verdict"] != TRANSFER_SUPPORTED


def test_replication_cells_are_capability_filtered_too():
    verdict = _verdict(
        interventions=_interventions(
            {"zebra": ["text->image", "image->text"], "toilet": list(AUDIO_PAIRS)}
        )
    )
    assert verdict["replication_cells_passing"] == []
    assert {
        (c["concept"], c["pair"])
        for c in verdict["replication_cells_measured_but_inadmissible"]
    } == {("zebra", "text->image"), ("zebra", "image->text")}


# ------------------------------------------------ 8, 9: GO and WEAK GO


def _overall(interventions, *, focal=FOCAL, capability=None):
    capability = capability or CAPABILITY
    capability_verdict = audio_capability_verdict(
        capability,
        selected_concepts=sorted(capability["per_concept"]),
        modalities=MODALITIES,
        thresholds=THRESHOLDS,
    )
    representational = representational_transfer_verdict(
        {35: _representational()},
        thresholds=THRESHOLDS,
        primary_layer=35,
        capability=capability,
        pooled_concepts=sorted(capability["per_concept"]),
    )
    causal = causal_transfer_verdict(
        interventions,
        layer=35,
        focal_concepts=focal,
        thresholds=THRESHOLDS,
        name="L35_CAUSAL_TRANSFER",
        capability=capability,
    )
    replication = replication_verdict({35: causal}, primary=causal, layers=(38, 40))
    overall = overall_verdict(
        capability=capability_verdict,
        representational=representational,
        primary_causal=causal,
        replication=replication,
        invariance={"passed": True, "per_modality": dict.fromkeys(MODALITIES, {})},
        thresholds=THRESHOLDS,
    )
    return capability_verdict, representational, causal, replication, overall


def _representational(layer=35, audio_top1=0.9):
    def entry(pair):
        top1 = audio_top1 if "spoken_audio" in pair else 0.9
        return {
            "n_sources": 8,
            "n_targets": 8,
            "jspace_retrieval": {"top1_accuracy": top1, "mrr": top1, "n_queries": 8},
            "raw_residual_retrieval": {"top1_accuracy": 0.2},
            "jspace_separation": {"gap": 0.4},
            "raw_residual_separation": {"gap": 0.1},
            "jspace_support_overlap": {"gap": 0.3},
            "shuffled_control": {"mean_top1_accuracy": 0.1, "p95_top1_accuracy": 0.3},
            "exclusions": {
                "eligible_targets": 8,
                "n_excluded_same_group": 1,
                "n_excluded_same_image_different_group": 0,
            },
            "n_distinct_source_images": 8,
            "n_distinct_target_images": 8,
        }

    return {"layer": layer, "pairs": {pair: entry(pair) for pair in ALL_PAIRS}}


def test_a_failed_capability_cell_cannot_produce_a_go_on_its_own():
    # zebra passes all four audio directions and nothing else does. Under the
    # old rule this was a GO; it must now be a NO-GO, and the measured effects
    # must still be visible in the artifact.
    _, _, causal, _, overall = _overall(
        _interventions({"zebra": list(AUDIO_PAIRS)}), focal=["zebra"]
    )
    assert causal["verdict"] == TRANSFER_UNSUPPORTED
    assert overall["verdict"] == THREE_MODALITY_NO_GO
    assert overall["verdict"] != THREE_MODALITY_WEAK_GO
    assert overall["criteria_status"]["audio_causal_cell_with_expected_sign"] == "FAIL"
    assert overall["measured_but_inadmissible_audio_cells"]
    assert "capability gate" in causal["rationale"]


def test_a_failed_capability_cell_cannot_flip_a_criterion_the_other_way_either():
    # An inadmissible cell with an insane activation norm and a global edit is
    # not evidence *against* the claim any more than it is evidence for it.
    interventions = _interventions(
        {"toilet": list(AUDIO_PAIRS), "zebra": ["spoken_audio->text"]}
    )
    for row in interventions["rows"]:
        if row["concept"] == "zebra":
            row["mean_activation_norm_ratio"] = 9.0
            row["mean_abs_unrelated_change"] = 5.0
    _, _, causal, _, overall = _overall(interventions, focal=["toilet", "zebra"])
    assert overall["criteria_status"]["activation_norms_sane"] == "PASS"
    assert overall["criteria_status"]["effects_are_specific"] == "PASS"
    assert overall["verdict"] == THREE_MODALITY_GO
    # Still recorded, in full, as a diagnostic.
    assert any(
        row["concept"] == "zebra"
        for row in overall["measured_but_inadmissible_audio_cells"]
    )


def test_the_toilet_shaped_evidence_still_supports_the_corrected_verdict():
    # The corrected reading of the observed run: zebra excluded, cat one
    # direction, toilet all four. Toilet alone carries both directions for
    # text/spoken_audio and image/spoken_audio.
    _, _, causal, _, overall = _overall(_interventions(OBSERVED_CELLS))
    assert causal["verdict"] == TRANSFER_SUPPORTED
    assert causal["concepts_transferring_both_audio_directions"] == ["toilet"]
    assert causal["capability_admissibility"]["excluded_concept_names"] == ["zebra"]
    assert overall["criteria_status"][
        "only_capability_admissible_evidence_counted"
    ] == "PASS"
    assert overall["verdict"] in (THREE_MODALITY_GO, THREE_MODALITY_WEAK_GO)


def test_the_go_criterion_set_includes_the_admissibility_invariant():
    _, _, _, _, overall = _overall(_interventions(OBSERVED_CELLS))
    assert "only_capability_admissible_evidence_counted" in overall["go_requirements"]


# ------------------------------------------------- 10: no post-hoc replacement


def test_an_ineligible_concept_is_excluded_and_never_replaced():
    roster = concept_admissibility(FOCAL, capability=CAPABILITY)
    assert roster["fixed_concepts"] == FOCAL
    assert roster["eligible_concepts"] == ["cat", "toilet"]
    assert roster["excluded_concept_names"] == ["zebra"]
    # The eligible set is a subset of the fixed set. A substitute concept —
    # "bird" and "microwave" both pass everything — cannot appear.
    assert set(roster["eligible_concepts"]) <= set(FOCAL)
    assert "bird" not in roster["eligible_concepts"]
    assert "microwave" not in roster["eligible_concepts"]
    assert "not replaced" in roster["no_post_hoc_replacement"]


def test_the_verdict_never_grows_a_concept_it_was_not_given():
    verdict = _verdict()
    concepts_in_cells = {cell["concept"] for cell in verdict["cells"]}
    assert concepts_in_cells == set(FOCAL)
    assert set(verdict["capability_admissibility"]["fixed_concepts"]) == set(FOCAL)


# --------------------------------------------- 14: the Stage C focal budget


def _stage_budget(n_focal, layers):
    return estimate_stage_passes(
        stage="C",
        n_concepts=6,
        n_focal_concepts=n_focal,
        modalities=MODALITIES,
        layers=(35, 38, 40),
        causal_layers=layers,
        n_total_groups=112,
        n_capability_groups=48,
        n_targets_per_cell=16,
        alphas=(0.0, 0.25, 0.5, 1.0),
        d_model=2560,
    )


def test_stage_c_prints_eligible_excluded_and_both_budgets():
    roster = concept_admissibility(FOCAL, capability=CAPABILITY)
    maximum = _stage_budget(len(FOCAL), (38, 40))
    gated = _stage_budget(len(roster["eligible_concepts"]), (38, 40))
    text = format_focal_capability_gate(
        roster, max_budget=maximum, gated_budget=gated, stage="C"
    )
    assert "fixed focal concepts (predeclared)  ['zebra', 'cat', 'toilet']" in text
    assert "capability-eligible                 ['cat', 'toilet']" in text
    assert "CAPABILITY_INELIGIBLE)    ['zebra']" in text
    assert "5/8 = 62.5%" in text
    assert f"{maximum.total_passes:,}" in text
    assert f"{gated.total_passes:,}" in text
    assert "not replaced" in text
    assert "resumable" in text
    # Two of three focal concepts survive, so the gated budget is two thirds.
    assert gated.total_passes * 3 == maximum.total_passes * 2


def test_a_concept_not_executed_in_stage_c_is_recorded_as_such_not_as_missing():
    # Stage C spends nothing on zebra, so its cells have no rows. They must
    # still appear, and must say *why* they are empty.
    verdict = _verdict(
        layer=38,
        interventions=_interventions(
            {"cat": ["spoken_audio->image"], "toilet": list(AUDIO_PAIRS)}, layer=38
        ),
        executed_concepts=["cat", "toilet"],
    )
    zebra = [c for c in verdict["cells"] if c["concept"] == "zebra"]
    assert len(zebra) == len(ALL_PAIRS)
    assert all(c["evaluated"] is False for c in zebra)
    assert all(
        c["execution_status"] == "not_executed_capability_ineligible" for c in zebra
    )
    assert all("no model passes were spent" in c["reasons"][0] for c in zebra)
    assert verdict["focal_concepts"] == FOCAL
    assert verdict["focal_concepts_executed"] == ["cat", "toilet"]


# ---------------------------------------------------------------- the report


def test_the_report_separates_measured_diagnostics_from_admissible_evidence():
    capability_verdict, representational, causal, replication, overall = _overall(
        _interventions(OBSERVED_CELLS)
    )
    text = render_report(
        run_dir="/tmp/run",
        capability=capability_verdict,
        representational=representational,
        primary_causal=causal,
        replication=replication,
        overall=overall,
        lens_report=None,
        audio_protocol=None,
        mode="native_audio_transfer",
    )
    assert "Measured diagnostic effect vs admissible scientific evidence" in text
    assert "CAPABILITY_INELIGIBLE" in text
    assert "zebra" in text
    assert "5/8 = 62.5%" in text
    assert "| admissible | counted |" in text
    assert "not replaced by another concept" in text
    assert CLAIM_ADMISSIBILITY_RULE_VERSION in text


def test_the_rule_record_is_checksummed_and_states_the_policy():
    record = admissibility_rule_record(threshold=0.7)
    assert record["rule_version"] == CLAIM_ADMISSIBILITY_RULE_VERSION
    assert record["required_modalities"] == list(MODALITIES)
    assert record["rule_checksum"].startswith("sha256:")
    assert "cannot be replaced post hoc" in " ".join(record["failed_concept_policy"])
    # The checksum tracks the rule, not the call.
    assert record["rule_checksum"] == admissibility_rule_record(threshold=0.7)[
        "rule_checksum"
    ]
    assert record["rule_checksum"] != admissibility_rule_record(threshold=0.8)[
        "rule_checksum"
    ]


def test_annotating_cells_never_removes_or_rewrites_a_measurement():
    cells = [
        {
            "concept": "zebra",
            "pair": "spoken_audio->text",
            "passes": True,
            "mean_signed_target_effect": 0.42,
        }
    ]
    annotated = annotate_causal_cells(cells, capability=CAPABILITY)
    assert len(annotated) == 1
    assert annotated[0]["mean_signed_target_effect"] == 0.42
    assert annotated[0]["passes"] is True
    assert annotated[0]["counted_toward_verdict"] is False
    assert annotated[0]["capability_decision_pair_specific"]["required_modalities"] == [
        "spoken_audio",
        "text",
    ]
