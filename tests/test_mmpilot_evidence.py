# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""The synchronized-evidence rule: lexicon matching, the audit, and persistence.

These tests exist because the first real run's behavioral gate failed for a
labelling reason, not a modelling one. A group was accepted as a ``cat``
positive on the strength of its COCO object annotation while its caption never
said "cat", so the text arm was scored wrong for answering a question the
caption could not support. Everything here pins the corrected rule down:
a valid synchronized positive needs the annotation *and* the caption term.
"""

import json
from pathlib import Path

import pytest

from jlens.mmpilot import evidence as V
from jlens.mmpilot import expansion as E
from jlens.mmpilot import manifest as M
from jlens.mmpilot import mock as K


def group(
    group_id="g1",
    *,
    caption="a cat on a sofa",
    annotations=("cat",),
    image_id=None,
    speaker="spk-a",
    source_split="train2014",
):
    return {
        "group_id": group_id,
        "synchronized_group_id": group_id,
        "image_id": image_id or f"i-{group_id}",
        "caption_id": f"c-{group_id}",
        "audio_record_id": f"a-{group_id}",
        "caption": caption,
        "image_path": f"/images/{group_id}.jpg",
        "audio_path": f"/wavs/{group_id}.wav",
        "speaker": speaker,
        "source_split": source_split,
        "concept_annotations": list(annotations),
        "annotation_source": "coco_object_annotation" if annotations else "none",
    }


CONFIG = V.default_config()


# ------------------------------------------------------------ lexical matching


@pytest.mark.parametrize(
    "caption,expected_term",
    [
        ("a cat on a sofa", "cat"),
        ("two cats sleeping", "cats"),
        ("a tiny kitten in a box", "kitten"),
        ("three kittens", "kittens"),
        ("A CAT, loudly.", "cat"),
        ("the cat's bowl", "cat"),
    ],
)
def test_singular_plural_and_approved_synonyms_match(caption, expected_term):
    record = V.caption_evidence(group(caption=caption), "cat", CONFIG)
    assert record["present"]
    assert record["matched_term"] == expected_term


@pytest.mark.parametrize(
    "caption",
    [
        "a herd of cattle grazing",       # the reported substring trap
        "concatenate the strings",
        "a catalog of business ideas",
        "scatter the seeds",
        "she is a catalyst",
    ],
)
def test_substring_false_positives_are_rejected(caption):
    assert not V.caption_evidence(group(caption=caption), "cat", CONFIG)["present"]


@pytest.mark.parametrize(
    "caption,concept",
    [
        ("a busy business district", "bus"),
        ("the doggerel verse", "dog"),
        ("horseradish on the plate", "horse"),
        ("a pizzeria on the corner", "pizza"),
        ("the training montage", "train"),
    ],
)
def test_word_boundaries_hold_for_every_pilot_concept(caption, concept):
    assert not V.caption_evidence(group(caption=caption), concept, CONFIG)["present"]


def test_matches_record_a_reproducible_span_on_the_normalized_caption():
    record = V.caption_evidence(group(caption="A  fluffy CAT!"), "cat", CONFIG)
    start, end = record["match_span"]
    assert record["normalized_caption"] == "a fluffy cat"
    assert record["normalized_caption"][start:end] == "cat"
    assert record["matched_text"] == "cat"


def test_normalization_is_deterministic_and_documented():
    assert V.normalize_caption("A  Café —  two cats!") == "a caf two cats"
    assert V.normalize_caption(None) == ""
    assert V.find_term("a cat", "cat") == (2, 5)
    assert V.find_term("a cattle", "cat") is None


# --------------------------------------------------------------- the two halves


def test_visual_only_positive_is_rejected():
    """The exact group that broke the first run: the picture has a cat, the
    caption does not say so."""
    record = V.group_evidence(
        group(caption="a fluffy animal asleep on a red sofa"), "cat", CONFIG
    )
    assert record["visual_evidence"]["present"]
    assert not record["caption_evidence"]["present"]
    assert not record["is_valid_synchronized_positive"]
    assert record["rejection_reason"] == V.REASON_NO_CAPTION


def test_caption_only_positive_is_rejected():
    record = V.group_evidence(
        group(caption="a cat on a sofa", annotations=()), "cat", CONFIG
    )
    assert not record["visual_evidence"]["present"]
    assert record["caption_evidence"]["present"]
    assert not record["is_valid_synchronized_positive"]
    assert record["rejection_reason"] == V.REASON_NO_VISUAL


def test_synchronized_visual_plus_caption_positive_is_accepted():
    record = V.group_evidence(group(caption="a cat on a sofa"), "cat", CONFIG)
    assert record["is_valid_synchronized_positive"]
    assert record["rejection_reason"] is None
    assert record["matched_term"] == "cat"
    assert record["visual_evidence"]["matched_categories"] == ["cat"]
    assert record["evidence_rule"] == "visual_annotation_AND_caption_lexicon"


def test_neither_kind_of_evidence_is_its_own_reason():
    record = V.group_evidence(
        group(caption="an empty street", annotations=()), "cat", CONFIG
    )
    assert record["rejection_reason"] == V.REASON_NO_EITHER


def test_spoken_evidence_is_metadata_never_a_claim_about_hearing():
    record = V.group_evidence(group(), "cat", CONFIG)
    spoken = record["spoken_evidence"]
    assert spoken["basis"] == "caption_is_the_read_script"
    assert spoken["audio_transcribed"] is False
    assert "says nothing about whether the model can hear" in spoken["note"]


def test_every_audit_record_carries_the_identifiers_needed_to_trace_it():
    record = V.group_evidence(group(), "cat", CONFIG)
    for field in (
        "group_id",
        "image_id",
        "caption_id",
        "audio_record_id",
        "audio_path",
        "speaker",
        "split_provenance",
        "concept",
        "caption",
        "normalized_caption",
        "visual_evidence",
        "caption_evidence",
        "matched_term",
        "match_span",
        "is_valid_synchronized_positive",
        "rejection_reason",
        "lexicon_hash",
    ):
        assert field in record, field
    assert record["split_provenance"]["source_split"] == "train2014"


# ------------------------------------------------------------------ negatives


def test_negatives_exclude_both_visual_and_caption_evidence():
    clean = V.negative_evidence(
        group(caption="an empty wooden table", annotations=()),
        ["cat", "dog"],
        CONFIG,
    )
    assert clean["qualifies_as_negative"]
    assert clean["disqualifying_concepts"] == []
    assert clean["qualification_rule"]

    visual_only = V.negative_evidence(
        group(caption="a fluffy animal", annotations=("cat",)), ["cat", "dog"], CONFIG
    )
    assert not visual_only["qualifies_as_negative"]
    assert visual_only["disqualifying_concepts"] == ["cat"]

    caption_only = V.negative_evidence(
        group(caption="a cat", annotations=()), ["cat", "dog"], CONFIG
    )
    assert not caption_only["qualifies_as_negative"]
    assert caption_only["disqualifying_concepts"] == ["cat"]


def test_build_subset_never_uses_a_visual_only_group_as_a_negative():
    groups = [
        *(group(f"pos{i}", caption="a cat on a sofa", image_id=f"p{i}") for i in range(6)),
        # visual evidence only: neither a positive nor a legitimate negative
        *(group(f"vis{i}", caption="a fluffy animal", image_id=f"v{i}") for i in range(4)),
        *(
            group(f"neg{i}", caption="an empty table", annotations=(), image_id=f"n{i}")
            for i in range(6)
        ),
    ]
    subset = M.build_subset(groups, {"cat": V.CONCEPT_LEXICON["cat"]}, groups_per_concept=6)
    rows = subset["splits"]["train"] + subset["splits"]["test"]
    assert {row["group_id"] for row in rows if row["is_positive"]} == {
        f"pos{i}" for i in range(6)
    }
    negatives = {row["group_id"] for row in rows if not row["is_positive"]}
    assert negatives == {f"neg{i}" for i in range(6)}
    assert not any(name.startswith("vis") for name in negatives)


def test_every_selected_row_records_why_it_qualified():
    groups = [
        *(group(f"pos{i}", caption="a cat on a sofa", image_id=f"p{i}") for i in range(6)),
        *(
            group(f"neg{i}", caption="an empty table", annotations=(), image_id=f"n{i}")
            for i in range(6)
        ),
    ]
    subset = M.build_subset(groups, {"cat": V.CONCEPT_LEXICON["cat"]}, groups_per_concept=6)
    for row in subset["splits"]["train"] + subset["splits"]["test"]:
        if row["is_positive"]:
            assert row["evidence"]["is_valid_synchronized_positive"]
            assert row["evidence"]["matched_term"]
        else:
            assert row["evidence"]["qualifies_as_negative"]
        assert row["split_provenance"]["assignment"] == row["split"]
    assert subset["provenance"]["evidence_lexicon_hash"].startswith("sha256:")


# ------------------------------------------------------------------- the audit


@pytest.fixture
def audit_world():
    groups = [
        *(group(f"cat{i}", caption="a cat on a sofa", image_id=f"c{i}") for i in range(6)),
        *(group(f"vis{i}", caption="a fluffy animal", image_id=f"v{i}") for i in range(3)),
        *(
            group(f"txt{i}", caption="a cat somewhere", annotations=(), image_id=f"t{i}")
            for i in range(2)
        ),
        *(
            group(f"neg{i}", caption="an empty table", annotations=(), image_id=f"n{i}")
            for i in range(6)
        ),
    ]
    return groups, V.audit_groups(groups, config=CONFIG, concepts=["cat"])


def test_the_audit_counts_every_rejection_reason(audit_world):
    _, audit = audit_world
    counts = audit.rejection_counts()
    assert counts[V.REASON_VALID] == 6
    assert counts[V.REASON_NO_CAPTION] == 3
    assert counts[V.REASON_NO_VISUAL] == 2
    assert len(audit.negatives) == 6
    assert all(record["qualifies_as_negative"] for record in audit.negatives)


def test_ranking_counts_only_valid_synchronized_positives(audit_world):
    groups, _ = audit_world
    rows = E.rank_concepts(
        groups,
        {"cat": V.CONCEPT_LEXICON["cat"]},
        groups_per_concept=6,
        evidence_config=CONFIG,
    )
    row = rows[0]
    assert row["concept"] == "cat"
    assert row["n_distinct_images"] == 6, "the 3 visual-only images must not count"
    assert row["n_valid_synchronized_groups"] == 6
    assert row["n_annotated_images"] == 9, "visual evidence is still reported"
    assert row["n_caption_evidence_groups"] == 8
    assert row["n_negative_groups"] == 6
    assert row["evidence_rule"] == "visual_annotation_AND_caption_lexicon"


def test_a_visual_only_dataset_is_a_dataset_no_go_that_names_the_gap():
    groups = [
        group(f"vis{i}", caption="a fluffy animal", image_id=f"v{i}") for i in range(12)
    ]
    rows = E.rank_concepts(
        groups, {"cat": V.CONCEPT_LEXICON["cat"]}, groups_per_concept=6, evidence_config=CONFIG
    )
    assert not rows[0]["feasible"]
    assert "written-caption evidence" in rows[0]["rejection_reason"]
    with pytest.raises(E.DatasetCoverageError, match="written caption"):
        E.select_concepts(rows, n_concepts=2)


def test_screening_four_concepts_still_requires_only_two():
    groups = []
    for concept in ("cat", "dog"):
        for i in range(6):
            groups.append(
                group(
                    f"{concept}{i}",
                    caption=f"a {concept} on a sofa",
                    annotations=(concept,),
                    image_id=f"{concept}-{i}",
                )
            )
    groups += [
        group(f"neg{i}", caption="an empty table", annotations=(), image_id=f"n{i}")
        for i in range(6)
    ]
    rows = E.rank_concepts(groups, V.CONCEPT_LEXICON, groups_per_concept=6)
    chosen = E.select_concepts(rows, n_concepts=2, max_concepts=4)
    assert sorted(chosen) == ["cat", "dog"]
    with pytest.raises(E.DatasetCoverageError):
        E.select_concepts(rows, n_concepts=3, max_concepts=4)


def test_splits_are_image_and_group_disjoint_and_keep_an_image_together():
    groups = []
    for i in range(6):
        for caption_index in range(2):
            groups.append(
                group(
                    f"cat{i}-{caption_index}",
                    caption=f"a cat number {caption_index} on sofa {i}",
                    image_id=f"c{i}",
                )
            )
    groups += [
        group(f"neg{i}", caption=f"an empty table {i}", annotations=(), image_id=f"n{i}")
        for i in range(6)
    ]
    subset = M.build_subset(
        groups, {"cat": V.CONCEPT_LEXICON["cat"]}, groups_per_concept=6
    )
    report = M.check_split_leakage(subset)
    assert report["ok"], report
    assert report["image_overlap"] == []
    assert report["group_overlap"] == []
    assert report["audio_overlap"] == []
    assert report["caption_overlap"] == []
    split_of: dict[str, set] = {}
    for split in ("train", "test"):
        for row in subset["splits"][split]:
            split_of.setdefault(row["image_id"], set()).add(split)
    assert all(len(splits) == 1 for splits in split_of.values())


# ----------------------------------------------------------------- persistence


@pytest.fixture
def persisted(tmp_path, audit_world):
    groups, audit = audit_world
    return tmp_path, groups, audit


def test_audit_and_manifest_are_written_atomically(persisted):
    tmp_path, groups, audit = persisted
    path = tmp_path / "synchronized_evidence_audit.json"
    payload, status = V.persist_evidence_audit(path, audit, conversion={"seed": 1})
    assert "wrote" in status
    assert path.is_file()
    assert not list(tmp_path.glob("*.tmp.*")), "no temp file may survive"
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["schema_version"] == V.AUDIT_SCHEMA_VERSION
    assert stored["audio_transcribed"] is False
    assert stored["original_manifest_mutated"] is False
    assert stored["lexicon_hash"] == audit.config.lexicon_hash
    assert stored["config_fingerprint"] == audit.config.fingerprint
    assert stored["rejection_counts"][V.REASON_NO_CAPTION] == 3
    assert payload["n_valid_synchronized_positives"] == 6

    manifest_path = tmp_path / "synchronized_evidence_manifest.json"
    manifest, _ = V.persist_synchronized_manifest(
        manifest_path,
        groups,
        audit,
        original_checksum="sha256:original",
        conversion={"seed": 1},
    )
    assert manifest["schema_version"] == V.EVIDENCE_SCHEMA_VERSION
    assert manifest["original_manifest_checksum"] == "sha256:original"
    assert len(manifest["groups"]) == len(groups)
    valid = [g for g in manifest["groups"] if g["valid_positive_concepts"]]
    assert len(valid) == 6


def test_compatible_reuse_resumes_and_incompatible_reuse_is_refused(persisted):
    tmp_path, _, audit = persisted
    path = tmp_path / "audit.json"
    _, first = V.persist_evidence_audit(path, audit, conversion={"seed": 1})
    _, second = V.persist_evidence_audit(path, audit, conversion={"seed": 1})
    assert "wrote" in first and "reused" in second

    # A different conversion configuration is a different audit.
    with pytest.raises(V.IncompatibleEvidenceStateError, match="conversion_hash"):
        V.persist_evidence_audit(path, audit, conversion={"seed": 2})

    # So is a different lexicon.
    narrowed = V.audit_groups(
        [group()], config=V.config_for_concepts({"cat": ("cat",)}), concepts=["cat"]
    )
    with pytest.raises(V.IncompatibleEvidenceStateError, match="lexicon_hash"):
        V.persist_evidence_audit(path, narrowed, conversion={"seed": 1})


def test_a_refused_reuse_leaves_the_stored_artifact_untouched(persisted):
    tmp_path, _, audit = persisted
    path = tmp_path / "audit.json"
    V.persist_evidence_audit(path, audit, conversion={"seed": 1})
    before = path.read_bytes()
    with pytest.raises(V.IncompatibleEvidenceStateError):
        V.persist_evidence_audit(path, audit, conversion={"seed": 2})
    assert path.read_bytes() == before


def test_the_ranking_artifact_records_what_it_was_ranked_from(persisted, audit_world):
    tmp_path, groups, audit = persisted
    rows = E.rank_concepts(
        groups, {"cat": V.CONCEPT_LEXICON["cat"]}, groups_per_concept=6, evidence_config=CONFIG
    )
    payload, _ = V.persist_concept_ranking(
        tmp_path / "concept_ranking.json",
        rows,
        audit,
        requirements=E.ConceptRequirements().to_dict(),
        conversion={"seed": 1},
    )
    assert payload["ranked_from"] == "valid_synchronized_positives_only"
    assert payload["requirements"]["min_distinct_images"] == 6
    assert payload["lexicon_hash"] == audit.config.lexicon_hash
    assert payload["rows"][0]["concept"] == "cat"


def test_source_checksums_cover_every_file_the_audit_read(tmp_path):
    first = tmp_path / "a.json"
    first.write_text('{"x": 1}', encoding="utf-8")
    digests = V.source_checksums([first, tmp_path / "missing.json"])
    assert list(digests) == [str(first)]
    assert digests[str(first)].startswith("sha256:")
    first.write_text('{"x": 2}', encoding="utf-8")
    assert V.source_checksums([first]) != digests


# ------------------------------------------------------------ immutability


def test_the_audit_never_mutates_the_groups_it_was_given(audit_world):
    groups, audit = audit_world
    before = json.dumps(groups, sort_keys=True, default=str)
    V.annotate_groups(groups, audit)
    V.synchronized_manifest_payload(
        groups, audit, original_checksum="sha256:x", conversion={}
    )
    assert json.dumps(groups, sort_keys=True, default=str) == before


def test_the_original_manifest_is_never_written(tmp_path):
    built = K.build_mock_dataset(tmp_path / "data", layout="sibling", manifest_records=8)
    manifest_path = Path(built["manifest_path"])
    before = manifest_path.read_bytes()
    payload = json.loads(before.decode("utf-8"))
    schema = M.inspect_manifest(payload)
    base = tmp_path / "data"
    normalized = M.normalize_manifest(
        payload,
        schema,
        image_roots=[base / "coco", base / "SpokenCOCO", base],
        audio_roots=[base / "coco", base / "SpokenCOCO", base],
        source_checksum=M.manifest_checksum(manifest_path),
        min_complete_groups=1,
    )
    groups = K.attach_object_annotations(normalized.groups, built)
    audit = V.audit_groups(groups, config=V.config_for_concepts(K.MOCK_CONCEPTS))
    V.persist_evidence_audit(
        tmp_path / "run" / "audit.json", audit, conversion={"seed": 1}
    )
    V.persist_synchronized_manifest(
        tmp_path / "run" / "manifest.json",
        groups,
        audit,
        original_checksum=M.manifest_checksum(manifest_path),
        conversion={"seed": 1},
    )
    assert manifest_path.read_bytes() == before
    assert Path(built["instances_path"]).is_file()


# ---------------------------------------------------------- printed inspection


def test_printed_examples_show_both_verdicts_without_dumping_everything(audit_world):
    _, audit = audit_world
    text = V.format_examples(audit, ["cat"], n_positive=2, n_rejected=2)
    assert "accepted (visual AND caption evidence)" in text
    assert "rejected because:" in text
    assert text.count("caption:") <= 4
    reasons = {
        line.split("rejected because:")[1].strip()
        for line in text.splitlines()
        if "rejected because:" in line
    }
    assert reasons and reasons <= set(V.REJECTION_REASONS)
    full = V.format_examples(audit, ["cat"], n_positive=1, n_rejected=5)
    assert V.REASON_NO_CAPTION in full and V.REASON_NO_VISUAL in full
    counts = V.format_rejection_counts(audit)
    assert V.REASON_VALID in counts and V.REASON_NO_VISUAL in counts


def test_the_mock_world_can_plant_the_failure_mode(tmp_path):
    """Without this the MOCK run could never exercise the rejection path."""
    built = K.build_mock_dataset(
        tmp_path / "data", layout="sibling", manifest_records=8, visual_only_images=2
    )
    assert built["visual_only_image_ids"]
    for image_id in built["visual_only_image_ids"]:
        assert built["object_annotations"][image_id], "the picture holds the concept"
    instances = json.loads(Path(built["instances_path"]).read_text(encoding="utf-8"))
    annotated = {str(item["image_id"]) for item in instances["annotations"]}
    assert set(built["visual_only_image_ids"]) <= annotated
