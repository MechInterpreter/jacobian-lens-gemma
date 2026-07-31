# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Dataset mining, capture, splitting, and the leakage guards."""

from collections import Counter

import pytest
import torch

from jlens.autoencoder.config import DatasetConfig, PursuitConfig
from jlens.autoencoder.dataset import (
    FUNCTION_WORDS,
    JSpaceLanguageDataset,
    assert_no_split_leakage,
    assign_splits,
    benchmark_build,
    build_dataset,
    context_token_ids,
    mine_phrase_occurrences,
    normalize_phrase,
    save_dataset,
    split_position,
)
from jlens.autoencoder.errors import AutoencoderError
from jlens.autoencoder.mocks import MOCK_PHRASES, build_mock_stack, mock_corpus
from jlens.autoencoder.prompting import build_verbalizer_prompt, phrase_target_ids
from jlens.pursuit import JSpaceDictionary

SMOKE_DATASET = DatasetConfig(
    mode="smoke",
    corpus="mock",
    source_layer=14,
    n_phrases=32,
    occurrences_per_phrase=2,
    min_context_tokens=3,
    max_context_tokens=48,
    max_documents=96,
    min_document_chars=40,
)
SMOKE_PURSUIT = PursuitConfig(k=10, correlation_chunk_size=None, build_chunk_rows=None)


@pytest.fixture(scope="module")
def stack():
    return build_mock_stack()


@pytest.fixture(scope="module")
def phrase_ids(stack):
    prompt = build_verbalizer_prompt(stack.tokenizer, n_memory_tokens=2)

    def resolve(phrase: str) -> list[int]:
        return phrase_target_ids(
            stack.tokenizer,
            prompt,
            phrase,
            end_of_turn_id=5,
            include_end_of_turn=False,
        )["phrase_token_ids"]

    return resolve


@pytest.fixture(scope="module")
def dictionary(stack):
    return JSpaceDictionary.from_lens(
        stack.lens, stack.source_layer, stack.model._lm_head.weight, device="cpu"
    )


@pytest.fixture(scope="module")
def built(stack, dictionary, phrase_ids):
    return build_dataset(
        stack.model,
        dictionary,
        stack.documents,
        dataset_config=SMOKE_DATASET,
        pursuit_config=SMOKE_PURSUIT,
        phrase_token_ids=phrase_ids,
    )


def test_normalize_phrase_is_case_and_whitespace_insensitive():
    assert normalize_phrase("  Great   Barrier  Reef ") == normalize_phrase("great barrier reef")


def test_split_position_is_deterministic_and_salt_sensitive():
    a = split_position("Great Barrier Reef", salt="s1")
    assert a == split_position("great barrier reef", salt="s1")
    assert a != split_position("Great Barrier Reef", salt="s2")
    assert 0.0 <= a < 1.0


def test_assign_splits_is_order_independent():
    phrases = list(MOCK_PHRASES)
    forward = assign_splits(phrases, salt="s", val_fraction=0.2, heldout_fraction=0.2)
    backward = assign_splits(
        list(reversed(phrases)), salt="s", val_fraction=0.2, heldout_fraction=0.2
    )
    assert forward == backward


def test_assign_splits_never_leaves_a_split_empty():
    """The regression this replaced hash *bucketing* for: a 32-phrase build was
    observed with zero held-out phrases, which silently removes the only split
    the conclusions may rest on."""
    for size in range(3, len(MOCK_PHRASES) + 1):
        assignment = assign_splits(
            MOCK_PHRASES[:size], salt="s", val_fraction=0.2, heldout_fraction=0.2
        )
        counts = Counter(assignment.values())
        assert set(counts) == {"train", "val", "heldout"}, (size, counts)


def test_assign_splits_keeps_every_occurrence_of_a_phrase_together():
    assignment = assign_splits(
        [*MOCK_PHRASES, *MOCK_PHRASES], salt="s", val_fraction=0.2, heldout_fraction=0.2
    )
    assert len(assignment) == len(MOCK_PHRASES)


def test_mining_rejects_function_word_phrases(stack, phrase_ids):
    documents = ["The of and the of and. The of and the of and."] * 8
    with pytest.raises(AutoencoderError, match="no phrase survived"):
        mine_phrase_occurrences(
            documents, config=SMOKE_DATASET, phrase_token_ids=phrase_ids
        )
    assert "the" in FUNCTION_WORDS


def test_mining_drops_overlapping_phrases(stack, phrase_ids):
    """"Great Barrier" and "Great Barrier Reef" must not both be mined: they
    would otherwise land in different splits and leak."""
    occurrences, stats = mine_phrase_occurrences(
        mock_corpus(), config=SMOKE_DATASET, phrase_token_ids=phrase_ids
    )
    mined = {normalize_phrase(o.phrase) for o in occurrences}
    for a in mined:
        for b in mined:
            if a != b:
                assert a not in b, (a, b)
    assert stats["rejected_overlapping_phrase"] > 0


def test_context_tokens_keep_the_tail_not_the_head(stack):
    tokenizer = stack.tokenizer
    text = " ".join(f"word{i % 20}" for i in range(200))
    ids = context_token_ids(
        tokenizer, text, max_context_tokens=12, min_context_tokens=3
    )
    assert len(ids) == 12
    assert ids[0] == tokenizer.bos_token_id
    full = tokenizer(text, return_tensors=None, add_special_tokens=False).input_ids
    assert ids[1:] == [int(t) for t in full[-11:]]


def test_context_shorter_than_the_minimum_is_rejected(stack):
    with pytest.raises(AutoencoderError, match="min_context_tokens"):
        context_token_ids(stack.tokenizer, "one", max_context_tokens=48, min_context_tokens=10)


def test_build_produces_aligned_records_tensors_and_hashes(built):
    assert len(built) == built.activations.shape[0] == built.cones.shape[0]
    for index, record in enumerate(built.records):
        assert record["record_index"] == index
        assert record["source_layer"] == 14
        assert 2 <= record["n_phrase_tokens"] <= 6
        assert record["cone_sha256"].startswith("sha256:")
        assert record["source_activation_sha256"].startswith("sha256:")
        assert record["cone_norm"] == pytest.approx(float(built.cones[index].norm()), rel=1e-5)
        assert 1 <= record["n_active_atoms"] <= SMOKE_PURSUIT.k


def test_build_records_carry_no_tensors(built):
    """Records are JSON-safe provenance only; tensors live in the cache file."""
    import json

    json.dumps(built.records)


def test_dataset_splits_and_prototypes(built):
    dataset = JSpaceLanguageDataset.from_build(built)
    dataset.assert_usable()
    train = set(dataset.phrases_for_split("train"))
    heldout = set(dataset.phrases_for_split("heldout"))
    assert train and heldout and not (train & heldout)
    phrases, prototypes, dispersion = dataset.prototypes("train")
    assert prototypes.shape == (len(phrases), dataset.d_model)
    assert torch.allclose(prototypes.norm(dim=-1), torch.ones(len(phrases)), atol=1e-5)
    assert dispersion["n_phrases"] == len(phrases)


def test_prototypes_use_only_the_requested_split(built):
    dataset = JSpaceLanguageDataset.from_build(built)
    train_phrases, _, _ = dataset.prototypes("train")
    heldout_phrases = set(dataset.phrases_for_split("heldout"))
    assert not set(train_phrases) & heldout_phrases


def test_leakage_check_passes_on_a_clean_build(built):
    dataset = JSpaceLanguageDataset.from_build(built)
    report = assert_no_split_leakage(
        dataset,
        salt=SMOKE_DATASET.split_salt,
        val_fraction=SMOKE_DATASET.val_fraction,
        heldout_fraction=SMOKE_DATASET.heldout_fraction,
    )
    assert report["clean"] is True
    assert report["violations"] == []


def test_leakage_check_catches_a_tampered_split(built):
    dataset = JSpaceLanguageDataset.from_build(built)
    heldout_index = dataset.indices_for_split("heldout")[0]
    dataset.records[heldout_index] = {**dataset.records[heldout_index], "split": "train"}
    with pytest.raises(AutoencoderError, match="split leakage"):
        assert_no_split_leakage(
            dataset,
            salt=SMOKE_DATASET.split_salt,
            val_fraction=SMOKE_DATASET.val_fraction,
            heldout_fraction=SMOKE_DATASET.heldout_fraction,
        )


def test_save_and_load_round_trip(tmp_path, built):
    directory = str(tmp_path / "dataset")
    manifest = save_dataset(directory, built, manifest_extra={"note": "test"})
    assert manifest["n_records"] == len(built)
    assert manifest["records_sha256"].startswith("sha256:")
    loaded = JSpaceLanguageDataset.load(directory)
    assert len(loaded) == len(built)
    assert torch.equal(loaded.cones, built.cones)
    assert loaded.phrase_token_ids == built.phrase_token_ids
    assert loaded.manifest["note"] == "test"


def test_benchmark_reports_measurements_and_a_labelled_projection(
    stack, dictionary, phrase_ids
):
    report = benchmark_build(
        stack.model,
        dictionary,
        stack.documents,
        dataset_config=SMOKE_DATASET,
        pursuit_config=SMOKE_PURSUIT,
        phrase_token_ids=phrase_ids,
        n_phrases=4,
    )
    assert report["measured"]["n_occurrences"] > 0
    assert report["measured"]["seconds_per_occurrence"] > 0
    assert report["projection"]["planned_occurrences"] == (
        SMOKE_DATASET.n_phrases * SMOKE_DATASET.occurrences_per_phrase
    )
    assert "linear" in report["projection"]["basis"]
