# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Where the candidate concepts come from, and what words count as mentions.

The pilot screened six concepts chosen by hand — bus, cat, dog, horse, pizza,
train — before anyone looked at the data. These tests pin down the replacement:
the universe is read from the local COCO ``instances_*.json`` files, every
category in them is a candidate, and each one carries an explicit lexical
specification whose accept/reject decisions are recorded rather than assumed.
"""

import json

import pytest

from jlens.mmpilot import concepts as C
from jlens.mmpilot import evidence as E
from jlens.mmpilot import expansion as X

# A universe deliberately DISJOINT from the old six-concept seed list, so a
# test that passes cannot be passing because the seeds leaked back in.
FIXTURE_CATEGORIES = [
    {"id": 1, "name": "zebra", "supercategory": "animal"},
    {"id": 2, "name": "giraffe", "supercategory": "animal"},
    {"id": 3, "name": "wine glass", "supercategory": "kitchen"},
    {"id": 4, "name": "remote", "supercategory": "electronic"},
    {"id": 5, "name": "orange", "supercategory": "food"},
    {"id": 6, "name": "person", "supercategory": "person"},
    {"id": 7, "name": "knife", "supercategory": "kitchen"},
]


def write_instances(path, categories=None, annotations=None, name="instances_train2014.json"):
    path.mkdir(parents=True, exist_ok=True)
    target = path / name
    target.write_text(
        json.dumps(
            {
                "categories": categories if categories is not None else FIXTURE_CATEGORIES,
                "annotations": annotations
                or [{"id": 1, "image_id": "img0", "category_id": 1}],
                "images": [{"id": "img0", "file_name": "img0.jpg"}],
            }
        ),
        encoding="utf-8",
    )
    return target


@pytest.fixture
def universe(tmp_path):
    instances = write_instances(tmp_path / "annotations")
    sources = X.discover_metadata_sources([instances.parent], max_files=8, max_depth=1)
    return C.discover_category_universe(
        [s for s in sources if s.source_kind == "coco_object_annotation"]
    )


# ------------------------------------------------------------------ discovery


def test_the_universe_is_read_from_the_local_instances_file(universe):
    assert universe.categories == (
        "giraffe", "knife", "orange", "person", "remote", "wine glass", "zebra",
    )
    assert universe.category_ids["zebra"] == (1,)
    assert universe.supercategories["wine glass"] == "kitchen"


def test_discovery_records_the_source_files_and_their_checksums(universe):
    (source,) = universe.sources
    assert source["filename"] == "instances_train2014.json"
    assert source["checksum"].startswith("sha256:")
    assert source["is_instance_file"]
    assert source["n_categories"] == len(FIXTURE_CATEGORIES)


def test_the_candidate_universe_does_not_depend_on_the_old_six_item_list(universe):
    """The seed list must be neither the universe nor a floor under it."""
    seeds = set(E.CONCEPT_LEXICON)
    assert seeds == {"bus", "cat", "dog", "horse", "pizza", "train"}
    assert not seeds & set(universe.categories)
    assert not seeds & set(universe.eligible)
    assert not seeds & set(universe.lexicon())


def test_val_instance_files_are_read_too_and_merged(tmp_path):
    write_instances(tmp_path / "ann", name="instances_train2014.json")
    write_instances(
        tmp_path / "ann",
        categories=[{"id": 20, "name": "kite"}],
        name="instances_val2014.json",
    )
    sources = X.discover_metadata_sources([tmp_path / "ann"], max_files=8, max_depth=1)
    found = C.discover_category_universe(
        [s for s in sources if s.source_kind == "coco_object_annotation"]
    )
    assert "kite" in found.categories
    assert "zebra" in found.categories
    assert {s["filename"] for s in found.sources} == {
        "instances_train2014.json",
        "instances_val2014.json",
    }


def test_a_caption_file_is_never_mistaken_for_an_object_annotation(tmp_path):
    """captions_*.json has no categories; using it would substitute caption
    text for the visual half of the evidence rule."""
    assert not C.is_instance_annotation_path("annotations/captions_train2014.json")
    assert C.is_instance_annotation_path("/x/annotations/instances_val2017.json")


def test_no_local_annotation_is_refused_rather_than_guessed():
    with pytest.raises(C.CategoryDiscoveryError, match="candidate concept universe is empty"):
        C.discover_category_universe([])


# --------------------------------------------------------------- morphology


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("zebra", ("zebras",)),
        ("bus", ("buses", "busses")),
        ("knife", ("knives",)),
        ("wine glass", ("wine glasses",)),
        ("teddy bear", ("teddy bears",)),
        ("sheep", ()),
        ("skis", ()),
        ("scissors", ()),
    ],
)
def test_plurals_inflect_the_head_word_only(word, expected):
    assert C.safe_plural(word) == expected


# ---------------------------------------------------------- lexical policy


def test_an_unreviewed_category_gets_its_name_and_a_plural_and_says_so():
    spec = C.lexical_spec("giraffe")
    assert spec.terms == ("giraffe", "giraffes")
    assert spec.derivation == "default_morphology"
    assert spec.ambiguity == "clean"
    assert spec.eligible


def test_ubiquitous_and_colour_dominated_categories_are_excluded():
    for name in ("person", "orange"):
        spec = C.lexical_spec(name)
        assert spec.ambiguity == "excluded"
        assert not spec.eligible
        assert spec.terms == ()
        assert spec.note
        # The reason is recorded per rejected form, not just as a flag.
        assert spec.rationale[name]
    assert "co-occurring" in C.lexical_spec("person").note
    assert "colour" in C.lexical_spec("orange").note


def test_an_alias_only_category_rejects_its_own_bare_name_with_a_reason():
    spec = C.lexical_spec("remote")
    assert spec.ambiguity == "alias_only"
    assert "remote" not in spec.terms
    assert "remote control" in spec.terms
    assert "remote" in spec.rejected
    assert "adjective" in spec.rationale["remote"]
    assert spec.eligible


def test_a_collision_is_resolved_by_an_exclusion_phrase_not_by_dropping_a_term():
    spec = C.lexical_spec("dog")
    assert spec.ambiguity == "resolved_by_exclusion"
    assert "dog" in spec.terms and "puppy" in spec.terms
    assert spec.exclusions == ("hot dog", "hot dogs")
    assert "hot dog" in spec.rationale


def test_an_irreducibly_ambiguous_category_is_kept_but_flagged():
    train = C.lexical_spec("train")
    assert train.ambiguity == "ambiguous"
    assert "train" in train.terms
    assert "verb" in train.note
    assert C.AMBIGUITY_SCORE["ambiguous"] < C.AMBIGUITY_SCORE["clean"]


def test_every_conservative_category_named_in_the_brief_has_an_explicit_policy():
    for name in ("train", "orange", "mouse", "tie", "bear", "remote", "person", "hot dog"):
        spec = C.lexical_spec(name)
        assert spec.derivation == "curated", name
        assert spec.ambiguity != "clean", name
        assert spec.note, name
        assert spec.rationale, name


def test_the_universe_hash_and_lexical_hash_track_what_they_name(universe):
    assert universe.universe_hash.startswith("sha256:")
    assert universe.lexical_hash.startswith("sha256:")
    assert universe.universe_hash != universe.lexical_hash
    other = C.CategoryUniverse(
        categories=("zebra",),
        category_ids={"zebra": (1,)},
        supercategories={},
        sources=(),
        specs={"zebra": C.lexical_spec("zebra")},
    )
    assert other.universe_hash != universe.universe_hash


def test_the_universe_serializes_its_exclusions_with_reasons(universe):
    payload = universe.to_dict()
    assert payload["universe_version"] == C.CONCEPT_UNIVERSE_VERSION
    excluded = {entry["category"] for entry in payload["excluded"]}
    assert excluded == {"orange", "person"}
    for entry in payload["excluded"]:
        assert entry["reason"]
        assert entry["rationale"]


# ------------------------------------------------- the config it produces


def test_an_evidence_config_built_from_specs_drops_excluded_categories(universe):
    config = E.config_from_specs(universe.specs)
    assert "person" not in config.concepts
    assert "orange" not in config.concepts
    assert "zebra" in config.concepts
    assert config.terms_for("remote") == ("remote control", "remote controls",
                                          "tv remote", "tv remotes")
    assert config.ambiguity_of("remote") == "alias_only"
    assert config.ambiguity_of("zebra") == "clean"


def test_a_config_with_no_usable_term_is_refused():
    with pytest.raises(ValueError, match="no eligible category"):
        E.config_from_specs({"orange": C.lexical_spec("orange")})
