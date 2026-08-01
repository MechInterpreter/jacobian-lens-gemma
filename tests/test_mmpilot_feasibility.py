# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Feasibility filtering, deterministic ranking, and the split they imply.

The pilot must not rank by raw frequency. COCO's most frequent category is
``person``, annotated on most images, which discriminates nothing and empties
the negative pool. These tests pin down what the ranking actually rewards, that
the scientific minimums are a gate rather than a term in the score, and that the
split the ranking predicts is the split that gets built.
"""

import pytest

from jlens.mmpilot import concepts as C
from jlens.mmpilot import evidence as E
from jlens.mmpilot import expansion as X
from jlens.mmpilot import manifest as M

CONFIG = E.config_from_specs(
    {
        name: C.lexical_spec(name)
        for name in ("zebra", "giraffe", "dog", "hot dog", "remote", "train", "person")
    }
)


def group(
    image_id,
    caption,
    *,
    annotations=(),
    index=0,
    speaker="spk-a",
    split="train",
):
    return {
        "group_id": f"g_{image_id}_{index}",
        "synchronized_group_id": f"g_{image_id}_{index}",
        "image_id": image_id,
        "caption": caption,
        "image_path": f"/root/coco/train2014/{image_id}.jpg",
        "audio_path": f"/root/SpokenCOCO/wavs/{image_id}_{index}.wav",
        "speaker": speaker,
        "source_split": split,
        "concept_annotations": sorted(annotations),
        "annotation_source": "coco_object_annotation" if annotations else "none",
    }


def world(
    concept,
    *,
    n_images=6,
    captions_per_image=2,
    also_annotate=(),
    name_in_caption=True,
    start=0,
    speakers=("spk-a", "spk-b", "spk-c"),
):
    """``n_images`` images of ``concept``, each with ``captions_per_image``."""
    out = []
    for offset in range(n_images):
        image_id = f"{concept.replace(' ', '_')}{start + offset:03d}"
        for index in range(captions_per_image):
            # Captions are unique per (image, caption): SpokenCOCO's are, and a
            # repeated caption would make the split look like it leaked.
            subject = concept if name_in_caption else "small object"
            caption = (
                f"a photo of a {subject} beside window {image_id} number {index}"
            )
            out.append(
                group(
                    image_id,
                    caption,
                    annotations=(concept, *also_annotate),
                    index=index,
                    speaker=speakers[(offset + index) % len(speakers)],
                    split="train" if offset < 4 else "val",
                )
            )
    return out


def negatives(n=12):
    return [
        group(f"neg{i:03d}", f"a plain wooden table beside window neg{i:03d}")
        for i in range(n)
    ]


def rank(groups, concepts=None, **kwargs):
    concepts = concepts or CONFIG.lexicon
    return X.rank_concepts(
        groups, dict(concepts), evidence_config=CONFIG, groups_per_concept=6, **kwargs
    )


# ------------------------------------------------------- lexical matching


def test_whole_word_matching_never_fires_on_a_substring():
    caption = "a herd of cattle grazing near a business park"
    for concept, terms in (("cat", ("cat",)), ("bus", ("bus",))):
        config = E.config_for_concepts({concept: terms})
        assert not E.caption_evidence({"caption": caption}, concept, config)["present"]


def test_a_multi_word_term_is_matched_as_a_phrase_not_as_its_words():
    """COCO's 'remote' is a remote control; the bare word is an adjective."""
    assert not E.caption_evidence(
        {"caption": "a remote beach at sunset"}, "remote", CONFIG
    )["present"]
    hit = E.caption_evidence(
        {"caption": "a remote control on the couch"}, "remote", CONFIG
    )
    assert hit["present"] and hit["matched_term"] == "remote control"


def test_an_exclusion_phrase_voids_a_match_inside_it_and_says_which():
    voided = E.caption_evidence({"caption": "two hot dogs on a plate"}, "dog", CONFIG)
    assert not voided["present"]
    assert voided["voided_matches"][0]["excluded_by"] == "hot dogs"
    # And 'hot dog' itself still matches, so neither category loses its term.
    assert E.caption_evidence({"caption": "two hot dogs on a plate"}, "hot dog", CONFIG)[
        "present"
    ]


def test_an_exclusion_does_not_swallow_a_genuine_mention_elsewhere():
    """The phrase is excluded, not the term: a real dog in the same caption
    still counts, whichever side of the phrase it falls on."""
    for caption in ("a dog begging beside two hot dogs", "two hot dogs and a dog"):
        hit = E.caption_evidence({"caption": caption}, "dog", CONFIG)
        assert hit["present"], caption
        assert hit["matched_term"] == "dog"
        # The span found is the standalone one, never the one inside the phrase.
        start, end = hit["match_span"]
        assert hit["normalized_caption"][start:end] == "dog"
        assert not hit["normalized_caption"][max(0, start - 4) : start].endswith("hot ")
    # When the only candidate span IS inside the phrase, the void is recorded.
    voided = E.caption_evidence({"caption": "two hot dogs"}, "dog", CONFIG)
    assert not voided["present"]
    assert voided["voided_matches"][0]["excluded_by"] == "hot dogs"


def test_the_ambiguity_status_travels_with_the_evidence_record():
    record = E.group_evidence(
        group("i0", "a train at the platform", annotations=("train",)), "train", CONFIG
    )
    assert record["lexical_ambiguity"] == "ambiguous"
    assert record["is_valid_synchronized_positive"]


# --------------------------------------------- visual AND caption evidence


def test_a_positive_still_requires_both_halves_after_discovery():
    """Concept discovery must not weaken the evidence rule."""
    visual_only = group("i0", "a small creature on a sofa", annotations=("zebra",))
    caption_only = group("i1", "a zebra in the grass", annotations=())
    both = group("i2", "a zebra in the grass", annotations=("zebra",))
    assert not E.is_valid_positive(visual_only, "zebra", CONFIG)
    assert not E.is_valid_positive(caption_only, "zebra", CONFIG)
    assert E.is_valid_positive(both, "zebra", CONFIG)

    audit = E.audit_groups([visual_only, caption_only, both], config=CONFIG)
    counts = audit.rejection_counts()
    assert counts[E.REASON_NO_CAPTION] == 1
    assert counts[E.REASON_NO_VISUAL] == 1
    assert counts[E.REASON_VALID] == 1


# ------------------------------------------------------------ feasibility


def test_every_commissioned_metric_appears_on_every_row():
    rows = rank(world("zebra") + negatives())
    for row in rows:
        for key in (
            "n_annotated_images",
            "n_groups_selected",
            "n_valid_synchronized_groups",
            "n_speakers",
            "n_train_positives",
            "n_test_positives",
            "n_negative_groups",
            "split_feasible",
            "cooccurrence",
            "max_cooccurrence_fraction",
            "rejection_reason",
            "lexical_ambiguity",
            "lexical_precision",
            "score",
            "rank",
        ):
            assert key in row, (row["concept"], key)


def test_a_concept_short_of_any_minimum_is_infeasible_with_a_named_reason():
    rows = {r["concept"]: r for r in rank(world("zebra", n_images=3) + negatives())}
    assert not rows["zebra"]["feasible"]
    assert any(item.startswith("distinct_images 3 < 6") for item in rows["zebra"]["unmet"])
    assert "distinct_images" in rows["zebra"]["rejection_reason"]


def test_too_few_negatives_makes_every_concept_infeasible():
    rows = rank(world("zebra") + negatives(2))
    assert not any(row["feasible"] for row in rows)
    assert all(any("negatives" in item for item in row["unmet"]) for row in rows)


def test_the_thresholds_are_not_lowered_when_nothing_qualifies():
    rows = rank(world("zebra", n_images=3) + negatives())
    with pytest.raises(X.DatasetCoverageError, match="DATASET NO-GO"):
        X.select_concepts(rows, n_concepts=2)
    # The applied requirements are the stated ones, printed in the refusal.
    defaults = X.ConceptRequirements()
    assert (defaults.min_distinct_images, defaults.min_groups) == (6, 6)
    assert (defaults.min_train_positives, defaults.min_test_positives) == (4, 2)
    assert defaults.min_negatives == 6


def test_a_visual_only_concept_is_rejected_and_the_gap_is_named():
    rows = {
        r["concept"]: r
        for r in rank(world("zebra", name_in_caption=False) + negatives())
    }
    assert not rows["zebra"]["feasible"]
    assert rows["zebra"]["n_annotated_images"] == 6
    assert rows["zebra"]["n_distinct_images"] == 0
    assert "written-caption evidence" in rows["zebra"]["evidence_gap"]


# --------------------------------------------------------------- ranking


def test_ranking_is_not_raw_frequency():
    """A ubiquitous co-occurring category must not win on count alone."""
    groups = (
        world("zebra", n_images=8, also_annotate=("person",))
        + world("giraffe", n_images=8, start=100)
        + negatives(16)
    )
    rows = rank(groups)
    ordered = [row["concept"] for row in rows if row["feasible"]]
    # Both are feasible and equally covered; giraffe wins because zebra's images
    # are all also 'person' images, so zebra depends on one dominant neighbour.
    assert ordered[:2] == ["giraffe", "zebra"]
    by_name = {row["concept"]: row for row in rows}
    assert by_name["zebra"]["dominant_cooccurring_category"] == "person"
    assert by_name["zebra"]["max_cooccurrence_fraction"] == 1.0
    assert by_name["giraffe"]["max_cooccurrence_fraction"] == 0.0
    assert by_name["giraffe"]["score"] > by_name["zebra"]["score"]


def test_an_unambiguous_concept_outranks_an_equally_covered_ambiguous_one():
    groups = world("zebra") + world("train", start=100) + negatives(16)
    rows = {row["concept"]: row for row in rank(groups)}
    assert rows["zebra"]["lexical_ambiguity"] == "clean"
    assert rows["train"]["lexical_ambiguity"] == "ambiguous"
    assert rows["zebra"]["score"] > rows["train"]["score"]


def test_speaker_diversity_and_split_headroom_raise_the_score():
    one_speaker = world("zebra", speakers=("spk-a",)) + negatives(16)
    many = world("zebra") + negatives(16)
    assert (
        rank(many)[0]["ranking"]["components"]["speakers"]
        > rank(one_speaker)[0]["ranking"]["components"]["speakers"]
    )
    small = rank(world("zebra") + negatives(16))[0]
    large = rank(world("zebra", n_images=12) + negatives(16))[0]
    assert large["ranking"]["components"]["images"] > small["ranking"]["components"]["images"]


def test_lexical_precision_penalises_terms_the_annotation_does_not_back():
    """Half the 'zebra' captions describe images with no zebra annotation."""
    precise = world("zebra") + negatives(16)
    imprecise = precise + [
        group(f"loose{i}", "a zebra painted on a wall", annotations=(), index=0)
        for i in range(6)
    ]
    assert rank(precise)[0]["lexical_precision"] == 1.0
    row = next(r for r in rank(imprecise) if r["concept"] == "zebra")
    assert row["lexical_precision"] < 1.0


def test_the_ranking_is_deterministic_and_independent_of_input_order():
    groups = world("zebra") + world("giraffe", start=100) + negatives(16)
    first = [row["concept"] for row in rank(groups)]
    again = [row["concept"] for row in rank(list(reversed(groups)))]
    assert first == again
    assert [row["rank"] for row in rank(groups)] == list(range(1, len(first) + 1))


def test_infeasible_concepts_always_sort_below_feasible_ones():
    groups = world("zebra") + world("giraffe", n_images=2, start=100) + negatives(16)
    rows = rank(groups)
    feasible = [index for index, row in enumerate(rows) if row["feasible"]]
    infeasible = [index for index, row in enumerate(rows) if not row["feasible"]]
    assert not feasible or not infeasible or max(feasible) < min(infeasible)


def test_the_table_prints_every_row_including_the_rejections():
    rows = rank(world("zebra") + world("giraffe", n_images=2, start=100) + negatives(16))
    table = X.format_ranking_table(rows)
    for row in rows:
        assert row["concept"] in table
    assert " NO " in table and " ok " in table
    assert "score" in table and "coocc" in table and "amb" in table


def test_selection_takes_the_top_two_feasible_in_ranking_order():
    groups = (
        world("zebra", n_images=8, also_annotate=("person",))
        + world("giraffe", n_images=8, start=100)
        + world("dog", n_images=6, start=200)
        + negatives(24)
    )
    rows = rank(groups)
    assert X.select_concepts(rows, n_concepts=2, max_concepts=2) == [
        row["concept"] for row in rows if row["feasible"]
    ][:2]


# ----------------------------------------------------------------- splits


def test_the_split_is_image_and_group_disjoint_on_discovered_concepts():
    groups = world("zebra") + world("giraffe", start=100) + negatives(16)
    subset = M.build_subset(
        groups,
        {name: CONFIG.terms_for(name) for name in ("zebra", "giraffe")},
        groups_per_concept=6,
        negatives_per_concept=6,
        evidence_config=CONFIG,
    )
    leakage = M.check_split_leakage(subset)
    assert leakage["ok"]
    assert not leakage["image_overlap"]
    assert not leakage["group_overlap"]
    assert not leakage["audio_overlap"]
    # Not concept-disjoint: transfer needs the same concept on both sides.
    assert sorted(leakage["shared_concepts_expected"]) == ["giraffe", "zebra"]


def test_the_ranking_predicts_the_split_it_actually_builds():
    groups = world("zebra") + world("giraffe", start=100) + negatives(16)
    row = next(r for r in rank(groups) if r["concept"] == "zebra")
    subset = M.build_subset(
        groups,
        {name: CONFIG.terms_for(name) for name in ("zebra", "giraffe")},
        groups_per_concept=6,
        negatives_per_concept=6,
        evidence_config=CONFIG,
    )
    built_train = [
        g for g in subset["splits"]["train"] if g["concept"] == "zebra"
    ]
    built_test = [g for g in subset["splits"]["test"] if g["concept"] == "zebra"]
    assert row["n_train_positives"] == len(built_train)
    assert row["n_test_positives"] == len(built_test)
