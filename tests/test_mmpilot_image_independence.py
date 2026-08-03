# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Same-image dependence: exclusion, identity refusals, and re-aggregation.

Every fixture here is hand-built and deterministic. The point of the module
under test is that one photograph entering a run as two synchronized groups is
one observation, so the fixtures plant exactly that and check what each layer
of the pipeline does with it.
"""

import pytest

from jlens.mmpilot.independence import (
    CAUSAL_AGGREGATION_VERSION,
    ImageIdentityError,
    audit_image_independence,
    divergence_summary,
    recompute_representational,
    resolve_image_identity,
    summarize_interventions_by_image,
)
from jlens.mmpilot.jspace import (
    NoEligibleTargetError,
    admissible_targets,
    representational_report,
)


def sample(sample_id, group_id, image_id, concept, modality, code, activation):
    return {
        "sample_id": sample_id,
        "group_id": group_id,
        "image_id": image_id,
        "concept": concept,
        "modality": modality,
        "code": code,
        "activation": activation,
    }


# ------------------------------------------------------------ the exclusion


def test_a_target_from_the_query_s_own_group_is_excluded():
    query = {"group_id": "g1", "image_id": "i1"}
    targets = [{"group_id": "g1", "image_id": "i1"}, {"group_id": "g2", "image_id": "i2"}]
    admissible, counts = admissible_targets(query, targets)

    assert [t["group_id"] for t in admissible] == ["g2"]
    assert counts["n_excluded_same_group"] == 1
    assert counts["n_excluded_same_image_different_group"] == 0


def test_a_target_sharing_the_image_under_a_different_group_is_excluded():
    """The defect this module exists for: the sibling caption's group.

    A group-only rule admits this target, and the query then retrieves its own
    photograph through the dataset's own pairing.
    """
    query = {"group_id": "g1", "image_id": "i1"}
    sibling = {"group_id": "g1b", "image_id": "i1"}
    admissible, counts = admissible_targets(query, [sibling])

    assert admissible == []
    assert counts["n_excluded_same_group"] == 0
    assert counts["n_excluded_same_image_different_group"] == 1
    assert counts["n_eligible"] == 0


def test_a_different_image_stays_eligible():
    query = {"group_id": "g1", "image_id": "i1"}
    targets = [
        {"group_id": "g2", "image_id": "i2"},
        {"group_id": "g3", "image_id": "i3"},
    ]
    admissible, counts = admissible_targets(query, targets)

    assert len(admissible) == 2
    assert counts["n_eligible"] == 2
    assert counts["n_excluded_same_group"] == 0
    assert counts["n_excluded_same_image_different_group"] == 0


def _two_image_world(image_of_text, image_of_target):
    """One text query and one image target, with the images under our control."""
    return [
        sample("t1:text", "gt", image_of_text, "cat", "text", {1: 1.0}, [1.0, 0.0]),
        sample("t2:image", "gi", image_of_target, "cat", "image", {1: 1.0}, [1.0, 0.0]),
    ]


def test_a_pair_with_no_eligible_target_fails_loudly_in_strict_mode():
    """A retrieval accuracy over zero queries is 0.0, which reads exactly like
    a measured failure. Strict mode refuses to report it as one."""
    samples = _two_image_world("shared", "shared")

    with pytest.raises(NoEligibleTargetError, match="image-level exclusion rule"):
        representational_report(
            samples, modalities=("text", "image"), n_permutations=2, strict=True
        )

    lenient = representational_report(
        samples, modalities=("text", "image"), n_permutations=2
    )
    assert lenient["pairs"]["text->image"]["jspace_retrieval"]["n_queries"] == 0


def test_every_representational_test_uses_the_same_exclusion():
    """Retrieval, separation, support overlap, the raw baseline and the
    shuffled control must not disagree about what a query may see."""
    samples = [
        sample("a:text", "g1", "i1", "cat", "text", {1: 1.0}, [1.0, 0.0]),
        sample("b:image", "g1b", "i1", "cat", "image", {1: 1.0}, [1.0, 0.0]),
        sample("c:image", "g2", "i2", "dog", "image", {2: 1.0}, [0.0, 1.0]),
    ]
    report = representational_report(
        samples, modalities=("text", "image"), n_permutations=4
    )
    entry = report["pairs"]["text->image"]

    # The only eligible target for the cat query is the dog image, so both
    # retrieval and separation must see exactly one pair, and it is mismatched.
    assert entry["jspace_retrieval"]["n_queries"] == 1
    assert entry["jspace_retrieval"]["top1_accuracy"] == 0.0
    assert entry["jspace_separation"]["n_matched_pairs"] == 0
    assert entry["jspace_separation"]["n_mismatched_pairs"] == 1
    assert entry["jspace_support_overlap"]["n_mismatched_pairs"] == 1
    assert entry["raw_residual_separation"]["n_mismatched_pairs"] == 1
    assert entry["shuffled_control"]["n_permutations"] == 4
    assert entry["exclusions"]["n_excluded_same_image_different_group"] == 1


def test_the_exclusion_report_carries_the_eligible_target_distribution():
    samples = [
        sample("a:text", "g1", "i1", "cat", "text", {1: 1.0}, [1.0, 0.0]),
        sample("b:text", "g2", "i2", "dog", "text", {2: 1.0}, [0.0, 1.0]),
        sample("c:image", "g1b", "i1", "cat", "image", {1: 1.0}, [1.0, 0.0]),
        sample("d:image", "g3", "i3", "dog", "image", {2: 1.0}, [0.0, 1.0]),
    ]
    report = representational_report(
        samples, modalities=("text", "image"), n_permutations=2
    )
    exclusions = report["pairs"]["text->image"]["exclusions"]

    assert exclusions["n_sources"] == 2
    assert exclusions["n_excluded_same_image_different_group"] == 1
    # The cat query keeps one target, the dog query keeps both.
    assert exclusions["eligible_targets"]["min"] == 1
    assert exclusions["eligible_targets"]["max"] == 2
    assert exclusions["eligible_targets"]["histogram"] == {"1": 1, "2": 1}
    assert report["exclusion_rule_version"]


# --------------------------------------------------------------- identity


def unit(group_id, image_id, *, concept="cat", split="train", modality="text", **extra):
    return {
        "sample_id": f"{group_id}:{modality}",
        "group_id": group_id,
        "image_id": image_id,
        "concept": concept,
        "split": split,
        "modality": modality,
        **extra,
    }


def test_distinct_image_counting_sees_through_repeated_groups():
    identity = resolve_image_identity(
        [
            unit("g1", "100"),
            unit("g1b", "100"),
            unit("g2", "200"),
        ]
    )
    assert identity.n_distinct_images == 2
    assert len(identity.groups) == 3
    assert identity.groups_of_image() == {"100": ["g1", "g1b"], "200": ["g2"]}
    assert identity.image_for_sample("g1b:text") == "100"


def test_unresolved_image_identity_is_refused():
    with pytest.raises(ImageIdentityError, match="no resolvable image identity"):
        resolve_image_identity(
            [{"sample_id": "g1:text", "group_id": "g1", "concept": "cat"}]
        )


def test_ambiguous_image_identity_is_refused_when_one_group_claims_two_images():
    with pytest.raises(ImageIdentityError, match="resolve to more than one image"):
        resolve_image_identity([unit("g1", "100"), unit("g1", "200", modality="image")])


def test_ambiguous_image_identity_is_refused_when_one_id_has_two_media_checksums():
    """One id naming two different photographs makes the exclusion unsound."""
    with pytest.raises(ImageIdentityError, match="distinct image media checksum"):
        resolve_image_identity(
            [
                unit("g1", "100", modality="image", media_checksum="sha256:aaa"),
                unit("g1b", "100", modality="image", media_checksum="sha256:bbb"),
            ]
        )


def test_ambiguous_image_identity_is_refused_when_one_photo_has_two_ids():
    with pytest.raises(ImageIdentityError, match="more than one image id"):
        resolve_image_identity(
            [
                unit("g1", "100", modality="image", media_checksum="sha256:aaa"),
                unit("g2", "200", modality="image", media_checksum="sha256:aaa"),
            ]
        )


def test_a_group_recorded_in_two_splits_is_refused():
    with pytest.raises(ImageIdentityError, match="more than one split"):
        resolve_image_identity(
            [
                unit("g1", "100", split="train"),
                unit("g1", "100", split="test", modality="image"),
            ]
        )


def test_identity_is_resolved_from_whichever_alias_the_artifact_used():
    """Field names are probed against the artifacts, never assumed."""
    identity = resolve_image_identity(
        [
            {"sample_id": "g1:text", "group_id": "g1", "cocoid": 419532, "split": "train"},
            {"sample_id": "g2:text", "group_id": "g2", "image_id": "419533", "split": "train"},
        ]
    )
    assert identity.image_for_group("g1") == "419532"
    assert identity.image_for_group("g2") == "419533"
    assert "cocoid" in identity.cross_checks["id_fields_used"]


def test_a_coco_filename_and_its_integer_id_resolve_to_one_image():
    """``COCO_train2014_000000419532.jpg`` and ``419532`` are one photograph."""
    identity = resolve_image_identity(
        [
            unit("g1", "COCO_train2014_000000419532.jpg"),
            unit("g2", "419532", source_split="train2014"),
        ]
    )
    assert identity.n_distinct_images == 1


# ------------------------------------------------------------ subset audit


def test_train_test_image_leakage_is_a_hard_failure():
    identity = resolve_image_identity(
        [unit("g1", "100", split="train"), unit("g2", "100", split="test")]
    )
    audit = audit_image_independence(identity)

    assert audit["train_test_image_overlap"] == ["100"]
    kinds = {failure["kind"] for failure in audit["hard_failures"]}
    assert "train_test_image_overlap" in kinds


def test_sibling_groups_crossing_splits_are_a_hard_failure():
    identity = resolve_image_identity(
        [
            unit("g1", "100", split="train"),
            unit("g1b", "100", split="test"),
            unit("g2", "200", split="train"),
        ]
    )
    audit = audit_image_independence(identity)

    crossing = audit["sibling_groups_crossing_splits"]
    assert len(crossing) == 1
    assert crossing[0]["group_ids"] == ["g1", "g1b"]
    assert sorted(crossing[0]["splits"]) == ["test", "train"]
    kinds = {failure["kind"] for failure in audit["hard_failures"]}
    assert "sibling_groups_cross_splits" in kinds


def test_a_clean_split_reports_the_dependence_without_failing():
    identity = resolve_image_identity(
        [
            unit("g1", "100", split="train"),
            unit("g1b", "100", split="train"),
            unit("g2", "200", split="test"),
            unit("g3", "300", concept=None, split="train"),
        ]
    )
    audit = audit_image_independence(
        identity, modality_records={"g1": 2, "g1b": 2, "g2": 2, "g3": 2}
    )

    assert audit["hard_failures"] == []
    assert audit["n_groups"] == 4
    assert audit["n_distinct_images"] == 3
    assert audit["n_images_with_multiple_groups"] == 1
    assert audit["concepts_affected"] == ["cat"]
    assert audit["n_modality_records_affected"] == 4
    train = audit["by_split"]["train"]
    assert train["n_groups"] == 3
    assert train["n_distinct_images"] == 2
    assert train["groups_per_image_histogram"] == {"1": 1, "2": 1}
    assert audit["source_training"]["positive_images_per_concept"] == {"cat": 1}
    assert audit["source_training"]["positive_groups_per_concept"] == {"cat": 2}
    assert audit["source_training"]["n_negative_images"] == 1
    assert audit["independent_unit"] == "image_id"


def test_causal_cells_report_positive_and_negative_image_counts_separately():
    identity = resolve_image_identity(
        [
            unit("p1", "100", split="test", modality="image"),
            unit("p2", "100", split="test", modality="image"),
            unit("n1", "300", concept=None, split="test", modality="image"),
            unit("n2", "400", concept=None, split="test", modality="image"),
        ]
    )
    interventions = [
        {
            "sample_id": f"{group}:image",
            "group_id": group,
            "concept": "cat",
            "source_modality": "text",
            "target_modality": "image",
            "layer": 38,
            "target_is_positive": positive,
        }
        for group, positive in (("p1", True), ("p2", True), ("n1", False), ("n2", False))
    ]
    audit = audit_image_independence(identity, interventions=interventions)
    cell = audit["causal_cells"][0]

    assert cell["n_positive_groups"] == 2
    assert cell["n_positive_images"] == 1, "two captions of one photograph"
    assert cell["n_negative_groups"] == 2
    assert cell["n_negative_images"] == 2
    assert cell["targets_are_pseudoreplicated"] is True


# ---------------------------------------------------- causal re-aggregation


def intervention(sample_id, group_id, *, effect, positive=True, alpha=0.5, **extra):
    return {
        "sample_id": sample_id,
        "group_id": group_id,
        "concept": "cat",
        "source_modality": "text",
        "target_modality": "image",
        "control_kind": "source_concept",
        "layer": 38,
        "alpha": alpha,
        "signed_target_effect": effect,
        "signed_margin_effect": effect,
        "max_abs_unrelated_change": 0.1,
        "activation_norm_ratio": 1.0,
        "prediction_changed": effect > 0,
        "target_is_positive": positive,
        **extra,
    }


@pytest.fixture
def duplicated_image_run():
    """Three groups, two of which are the same photograph.

    Group-level: mean of (10, 10, 1) = 7.0 over n=3.
    Image-level: mean of (mean(10, 10), 1) = mean(10, 1) = 5.5 over n=2.
    """
    identity = resolve_image_identity(
        [
            unit("g1", "100", split="test", modality="image"),
            unit("g1b", "100", split="test", modality="image"),
            unit("g2", "200", split="test", modality="image"),
        ]
    )
    records = [
        intervention("g1:image", "g1", effect=10.0),
        intervention("g1b:image", "g1b", effect=10.0),
        intervention("g2:image", "g2", effect=1.0),
    ]
    return identity, records


def test_repeated_observations_are_averaged_within_image_before_aggregation(
    duplicated_image_run,
):
    identity, records = duplicated_image_run
    summary = summarize_interventions_by_image(records, identity)
    row = summary["rows"][0]

    assert row["n_records"] == 3
    assert row["n_groups"] == 3
    assert row["n_distinct_images"] == 2
    assert row["n"] == 2, "n counts photographs, not captions"
    assert row["mean_signed_target_effect"] == pytest.approx(5.5)
    assert row["per_image"]["100"]["n_records"] == 2
    assert row["per_image"]["100"]["mean_signed_target_effect"] == pytest.approx(10.0)
    assert row["aggregation_version"] == CAUSAL_AGGREGATION_VERSION


def test_the_group_level_result_is_preserved_and_the_divergence_reported(
    duplicated_image_run,
):
    identity, records = duplicated_image_run
    summary = summarize_interventions_by_image(records, identity)
    row = summary["rows"][0]

    assert row["group_level"]["n"] == 3
    assert row["group_level"]["mean_signed_target_effect"] == pytest.approx(7.0)
    assert row["divergence_from_group_level"] == pytest.approx(-1.5)
    assert row["pseudoreplicated_at_group_level"] is True

    divergence = divergence_summary(summary)
    assert divergence["n_rows_pseudoreplicated_at_group_level"] == 1
    assert divergence["max_abs_divergence"] == pytest.approx(1.5)


def test_group_and_image_aggregation_agree_when_no_image_repeats():
    identity = resolve_image_identity(
        [
            unit("g1", "100", split="test", modality="image"),
            unit("g2", "200", split="test", modality="image"),
        ]
    )
    records = [
        intervention("g1:image", "g1", effect=10.0),
        intervention("g2:image", "g2", effect=1.0),
    ]
    row = summarize_interventions_by_image(records, identity)["rows"][0]

    assert row["divergence_from_group_level"] == pytest.approx(0.0)
    assert row["pseudoreplicated_at_group_level"] is False
    assert row["n_distinct_images"] == row["n_groups"] == 2


def test_positive_and_negative_target_images_are_counted_separately():
    identity = resolve_image_identity(
        [
            unit("p1", "100", split="test", modality="image"),
            unit("p1b", "100", split="test", modality="image"),
            unit("n1", "300", concept=None, split="test", modality="image"),
        ]
    )
    records = [
        intervention("p1:image", "p1", effect=2.0, positive=True),
        intervention("p1b:image", "p1b", effect=4.0, positive=True),
        intervention("n1:image", "n1", effect=1.0, positive=False),
    ]
    row = summarize_interventions_by_image(records, identity)["rows"][0]

    assert row["n_positive_images"] == 1
    assert row["n_positive_groups"] == 2
    assert row["n_negative_images"] == 1
    assert row["n_negative_groups"] == 1
    assert row["mean_positive_image_effect"] == pytest.approx(3.0)
    assert row["mean_negative_image_effect"] == pytest.approx(1.0)


def test_expected_sign_fraction_is_averaged_within_image_first():
    """Two captions of one photograph disagreeing is half of one image, not
    one of two observations."""
    identity = resolve_image_identity(
        [
            unit("g1", "100", split="test", modality="image"),
            unit("g1b", "100", split="test", modality="image"),
            unit("g2", "200", split="test", modality="image"),
        ]
    )
    records = [
        intervention("g1:image", "g1", effect=1.0),
        intervention("g1b:image", "g1b", effect=-1.0),
        intervention("g2:image", "g2", effect=1.0),
    ]
    row = summarize_interventions_by_image(records, identity)["rows"][0]

    assert row["per_image"]["100"]["fraction_expected_sign"] == pytest.approx(0.5)
    assert row["fraction_expected_sign"] == pytest.approx(0.75)
    assert row["group_level"]["fraction_expected_sign"] == pytest.approx(2 / 3)


def test_a_cell_resting_on_one_photograph_is_flagged():
    identity = resolve_image_identity(
        [
            unit("g1", "100", split="test", modality="image"),
            unit("g1b", "100", split="test", modality="image"),
        ]
    )
    records = [
        intervention("g1:image", "g1", effect=5.0),
        intervention("g1b:image", "g1b", effect=5.0),
    ]
    summary = summarize_interventions_by_image(records, identity)
    row = summary["rows"][0]

    assert row["evidence_is_single_image"] is True
    assert row["n_distinct_images"] == 1
    assert divergence_summary(summary)[
        "off_diagonal_source_rows_on_a_single_image"
    ] == [{"cell": "cat|text->image|a0.5", "n_distinct_images": 1}]


def test_one_image_serving_as_both_a_positive_and_a_negative_is_a_hard_failure():
    identity = resolve_image_identity(
        [
            unit("g1", "100", split="test", modality="image"),
            unit("g1b", "100", split="test", modality="image"),
        ]
    )
    records = [
        intervention("g1:image", "g1", effect=1.0, positive=True),
        intervention("g1b:image", "g1b", effect=1.0, positive=False),
    ]
    summary = summarize_interventions_by_image(records, identity)

    kinds = {failure["kind"] for failure in summary["hard_failures"]}
    assert "image_is_both_positive_and_negative_target" in kinds


def test_a_recorded_image_id_that_disagrees_with_the_resolved_one_is_reported():
    """Units written before identities were canonicalized are reported, never
    silently preferred over the resolved identity."""
    identity = resolve_image_identity(
        [
            unit("g1", "COCO_train2014_000000000100.jpg", split="test", modality="image"),
            unit("g2", "200", split="test", modality="image"),
        ]
    )
    records = [
        intervention("g1:image", "g1", effect=1.0, image_id="legacy-alias"),
        intervention("g2:image", "g2", effect=1.0),
    ]
    summary = summarize_interventions_by_image(records, identity)

    assert summary["recorded_image_id_disagreements"] == [
        {
            "sample_id": "g1:image",
            "recorded_image_id": "legacy-alias",
            # The COCO filename normalizes to a split-aware key.
            "resolved_image_id": "train2014:100",
        }
    ]


# ------------------------------------------ corrected representational path


def test_recompute_representational_overwrites_saved_ids_with_resolved_ones():
    """Two artifacts written under different aliases must not read as two
    photographs, so the canonical identity wins over whatever was saved."""
    records = [
        unit("g1", "COCO_train2014_000000000100.jpg", modality="text", split="test"),
        unit("g1b", "100", modality="image", split="test", source_split="train2014"),
        unit("g2", "200", concept="dog", modality="image", split="test"),
    ]
    identity = resolve_image_identity(records)
    activations = [
        {**record, "layer": 38, "activation": [1.0, 0.0]} for record in records
    ]
    codes = [
        {**record, "layer": 38, "support": [1], "coefficients": [1.0]}
        for record in records
    ]
    report = recompute_representational(
        activations,
        codes,
        identity,
        layer=38,
        modalities=("text", "image"),
        n_permutations=2,
    )
    entry = report["pairs"]["text->image"]

    assert report["image_disjoint"] is True
    assert entry["exclusions"]["n_excluded_same_image_different_group"] == 1
    assert entry["jspace_retrieval"]["n_queries"] == 1
