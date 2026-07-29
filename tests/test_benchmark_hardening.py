# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Methodological guards on the generative benchmark.

Two failure modes the corrected GPU smoke run exposed, now pinned:

1. ``dev-split-photosynthesis`` tokenized to the single id 93036. A single-token
   target tests no multi-token scoring, and — because every steering schedule
   injects identically at the prompt-final position and differs only at
   *generated* positions — forces prompt_only / constant / decaying to identical
   target log-probabilities.
2. The dev smoke borrowed ``held-split-metamorphosis`` as its unrelated-cone
   control, leaking a held-out vector into a development run.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from jlens.generative import (
    MAX_TARGET_TOKENS,
    MIN_TARGET_TOKENS,
    GenerativeError,
    load_benchmark,
    select_split_examples,
    token_strings,
    validate_target_tokens,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / "configs" / "generative_benchmark.json"


class CharTokenizer:
    """One token per character, so token counts are predictable in tests."""

    def __call__(self, text, *, add_special_tokens=True, return_tensors=None):
        return SimpleNamespace(input_ids=[ord(c) for c in text])

    def decode(self, ids, **_kw) -> str:
        return "".join(chr(int(i)) for i in ids)


def _resolve(tokenizer, phrase):
    return list(tokenizer(phrase, add_special_tokens=False).input_ids)


def _example(example_id, phrase, **kw):
    return {
        "example_id": example_id,
        "category": "noun_phrase",
        "source_prompt": "prompt",
        "target_phrase": phrase,
        "extraction_position": -1,
        **kw,
    }


# ------------------------------------------------------ target token validation


def test_single_token_target_is_rejected_with_full_detail():
    """The photosynthesis case. The error must name the example, the phrase,
    the ids, and the token strings, so the fix needs no second run."""
    examples = [_example("dev-single", "x")]  # one character => one token
    with pytest.raises(GenerativeError) as excinfo:
        validate_target_tokens(examples, CharTokenizer(), resolve=_resolve)
    message = str(excinfo.value)
    assert "dev-single" in message
    assert repr("x") in message
    assert str(ord("x")) in message
    assert "1 token(s)" in message
    assert f"need {MIN_TARGET_TOKENS}-{MAX_TARGET_TOKENS}" in message


def test_overlong_target_is_rejected():
    examples = [_example("dev-long", "abcdefghij")]
    with pytest.raises(GenerativeError, match="10 token"):
        validate_target_tokens(examples, CharTokenizer(), resolve=_resolve)


def test_every_offender_is_reported_in_one_pass():
    """Listing all violations at once matters: each retry costs a model load."""
    examples = [
        _example("ok-one", "abc"),
        _example("bad-short", "z"),
        _example("bad-long", "abcdefghijk"),
    ]
    with pytest.raises(GenerativeError) as excinfo:
        validate_target_tokens(examples, CharTokenizer(), resolve=_resolve)
    message = str(excinfo.value)
    assert "bad-short" in message and "bad-long" in message
    assert "ok-one" not in message
    assert message.startswith("2 benchmark target(s)")


def test_valid_targets_return_ids_and_token_strings():
    resolved = validate_target_tokens(
        [_example("dev-ok", "abc")], CharTokenizer(), resolve=_resolve
    )
    assert resolved["dev-ok"]["target_token_ids"] == [ord("a"), ord("b"), ord("c")]
    assert resolved["dev-ok"]["target_token_strings"] == ["a", "b", "c"]
    assert resolved["dev-ok"]["n_target_tokens"] == 3


def test_prevalidated_ids_are_checked_not_trusted():
    """A manifest-supplied target_token_ids must face the same requirement."""
    examples = [_example("dev-pinned", "abc", target_token_ids=[7])]
    with pytest.raises(GenerativeError, match="dev-pinned"):
        validate_target_tokens(examples, CharTokenizer(), resolve=_resolve)


def test_token_strings_decodes_each_id_separately():
    """Joint decoding would hide the segmentation, which is the thing recorded."""
    assert token_strings(CharTokenizer(), [104, 105]) == ["h", "i"]


def test_validate_rejects_incoherent_bounds():
    with pytest.raises(GenerativeError):
        validate_target_tokens(
            [_example("e", "ab")],
            CharTokenizer(),
            resolve=_resolve,
            min_tokens=5,
            max_tokens=2,
        )


# -------------------------------------------------- same-split donor selection


def _two_split_manifest():
    return {
        "dev": [
            _example("dev-1", "ab"),
            _example("dev-2", "cd"),
            _example("dev-3", "ef"),
        ],
        "heldout": [_example("held-1", "gh"), _example("held-2", "ij")],
    }


@pytest.mark.parametrize("limit", [None, 1, 2, 3, 99])
@pytest.mark.parametrize("split", ["dev", "heldout"])
def test_no_cross_split_unrelated_control_selection(split, limit):
    """The core guarantee: whatever the limit, every selected example — including
    the one that exists only to donate an unrelated cone — comes from the
    requested split."""
    manifest = _two_split_manifest()
    selected, donor_only = select_split_examples(manifest, split, limit=limit)
    own_ids = {e["example_id"] for e in manifest[split]}
    other = "heldout" if split == "dev" else "dev"
    other_ids = {e["example_id"] for e in manifest[other]}

    selected_ids = {e["example_id"] for e in selected}
    assert selected_ids <= own_ids
    assert not (selected_ids & other_ids)
    assert donor_only <= selected_ids
    # A donor must always exist, or the unrelated-cone control cannot be built.
    assert len(selected) >= 2


def test_limit_one_adds_a_same_split_donor_marked_donor_only():
    manifest = _two_split_manifest()
    selected, donor_only = select_split_examples(manifest, "dev", limit=1)
    assert [e["example_id"] for e in selected] == ["dev-1", "dev-2"]
    # dev-2 donates a cone but is never scored.
    assert donor_only == {"dev-2"}


def test_full_run_has_no_donor_only_examples():
    manifest = _two_split_manifest()
    selected, donor_only = select_split_examples(manifest, "dev", limit=None)
    assert len(selected) == 3
    assert donor_only == set()


def test_split_with_one_example_fails_rather_than_borrowing():
    """Silently reaching into the other split is what broke the smoke run."""
    manifest = {
        "dev": [_example("dev-1", "ab")],
        "heldout": [_example("held-1", "cd")],
    }
    with pytest.raises(GenerativeError, match="same"):
        select_split_examples(manifest, "dev", limit=None)


def test_unknown_split_fails():
    with pytest.raises(GenerativeError, match="no example list"):
        select_split_examples(_two_split_manifest(), "nonexistent")


# -------------------------------------------------------- the shipped manifest


def test_shipped_manifest_loads_and_has_both_splits_populated():
    manifest = load_benchmark(str(MANIFEST))
    assert len(manifest["dev"]) >= 2
    assert len(manifest["heldout"]) >= 2


def test_shipped_manifest_leads_each_split_with_multi_word_targets():
    """--smoke runs the *first* example of a split (with the second as donor),
    so both must be multi-word: a phrase containing a space cannot collapse to a
    single token, which guarantees the smoke run exercises multi-token scoring
    even before the tokenizer is available to check exact counts."""
    manifest = load_benchmark(str(MANIFEST))
    for split in ("dev", "heldout"):
        for position, example in enumerate(manifest[split][:2]):
            phrase = example["target_phrase"].strip()
            assert " " in phrase, (
                f"{split}[{position}] {example['example_id']} target {phrase!r} "
                f"is a single word; the first two examples of a split must be "
                f"multi-word so --smoke cannot land on a single-token target"
            )


def test_shipped_manifest_dropped_the_known_single_token_concepts():
    """Regression: the concepts that were single tokens (or near-certain to be)
    under the pinned Gemma tokenizer must not come back."""
    manifest = load_benchmark(str(MANIFEST))
    banned = {
        "photosynthesis",  # measured: single id 93036
        "serendipity",
        "metamorphosis",
        "cryptocurrency",
        "birdhouse",
        "toothbrush",
        "lighthouse",
        "waterfall",
    }
    for split in ("dev", "heldout"):
        for example in manifest[split]:
            phrase = example["target_phrase"].strip().lower()
            assert phrase not in banned, (
                f"{example['example_id']} reintroduces {phrase!r}, which does "
                f"not reliably tokenize to >= 2 tokens"
            )


def test_shipped_manifest_ids_are_split_prefixed():
    """Prefixes make a cross-split leak visible on inspection of any record."""
    manifest = load_benchmark(str(MANIFEST))
    for example in manifest["dev"]:
        assert example["example_id"].startswith("dev-")
    for example in manifest["heldout"]:
        assert example["example_id"].startswith("held-")


def test_shipped_manifest_smoke_selection_stays_within_dev():
    manifest = load_benchmark(str(MANIFEST))
    selected, _ = select_split_examples(manifest, "dev", limit=1)
    assert all(e["example_id"].startswith("dev-") for e in selected)


def test_shipped_manifest_targets_are_short_enough_to_fit_the_upper_bound():
    """The exact counts need the pinned Gemma tokenizer (checked at run time and
    by scripts/validate_benchmark_targets.py). What is checkable offline: no
    target has more words than the upper bound allows tokens."""
    manifest = load_benchmark(str(MANIFEST))
    for split in ("dev", "heldout"):
        for example in manifest[split]:
            words = example["target_phrase"].split()
            assert 1 <= len(words) <= MAX_TARGET_TOKENS, example["example_id"]


def test_config_declares_the_target_token_bounds():
    yaml = pytest.importorskip("yaml")
    with open(
        REPO_ROOT / "configs" / "gemma_generative_validation.yaml", encoding="utf-8"
    ) as handle:
        config = yaml.safe_load(handle)
    bounds = config["benchmark"]["target_token_bounds"]
    assert bounds["min"] == MIN_TARGET_TOKENS
    assert bounds["max"] == MAX_TARGET_TOKENS


def test_config_sweeps_low_ratios_and_includes_natural_scale():
    yaml = pytest.importorskip("yaml")
    with open(
        REPO_ROOT / "configs" / "gemma_generative_validation.yaml", encoding="utf-8"
    ) as handle:
        config = yaml.safe_load(handle)
    ratios = config["steering"]["strength_ratios"]
    # The informative region is low, around the natural cone scale.
    assert {0.01, 0.03, 0.05, 0.1, 0.25} <= set(ratios)
    assert min(ratios) <= 0.01
    # Stronger ratios are retained only as stress tests.
    assert max(ratios) >= 1.0
    assert "natural_scale" in config["steering"]["conditions"]
    assert "natural_scale" in config["decode"]["generate_conditions"]


def test_manifest_json_is_valid_utf8_and_newline_terminated():
    raw = MANIFEST.read_text(encoding="utf-8")
    json.loads(raw)
    assert raw.endswith("\n")
