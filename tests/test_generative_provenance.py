# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""End-to-end example-identity checks for the generative runner.

These exist because of a specific, unfalsifiable-from-the-artifacts worry: in
the lost run ``generative_20260730T164407987257_0c5fc4c2f7a5``, several
``dev-entity-mandela`` records decoded "Black Hole" — the surface form of a
*different* benchmark target (``held-phrase-black-hole``). Decoded text alone
cannot distinguish "the wrong example's activation/cone was used here" from
"the receiver prompt carries no example information, so the model said the same
generic thing for every example".

So the run must record *which example each artifact came from*, and these tests
must fail if any of it is ever wrong. Every example here is given a source
activation that is unique and recognizable by construction (a per-example
one-hot signature), so a swap cannot hide behind numerical coincidence in the
8-dimensional mock: the fingerprint of example A's activation appearing on
example B's record is proof, not evidence.

Covered end to end: ``--limit-examples``, multiple source layers, multiple
receiver prompts, the full condition battery, donor selection, and record
serialization.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

from jlens.generative import (
    CONE_SOURCE_DONOR_CONDITIONS,
    GenerativeError,
    cone_source_role,
    expected_cone_source_example_id,
    tensor_sha256,
    vector_identity,
)

from .test_generative_runner import (  # noqa: F401 - fixture import
    _import_runner,
    _mock_load_gemma4,
    experiment,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------- unit level


def test_tensor_sha256_depends_only_on_the_numbers():
    a = torch.tensor([1.0, 2.0, 3.0])
    # A view, a non-contiguous slice, and a different storage dtype must all
    # fingerprint identically — otherwise the hash tracks tensor plumbing
    # rather than content and comparing two records proves nothing.
    b = torch.tensor([1.0, 9.0, 2.0, 9.0, 3.0, 9.0])[::2]
    c = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    assert tensor_sha256(a) == tensor_sha256(b) == tensor_sha256(c)
    assert tensor_sha256(a) != tensor_sha256(torch.tensor([1.0, 2.0, 3.001]))


def test_vector_identity_never_invents_a_value():
    assert vector_identity(None) == (None, None)
    norm, digest = vector_identity(torch.tensor([3.0, 4.0]))
    assert norm == pytest.approx(5.0)
    assert digest.startswith("sha256:")


def test_cone_source_role_partitions_every_condition():
    from jlens.generative import VECTOR_CONDITIONS

    roles = {c: cone_source_role(c) for c in VECTOR_CONDITIONS}
    assert set(roles.values()) <= {"self", "donor", "none"}
    # Exactly the unrelated-cone controls may legitimately use another
    # example's cone. If a condition is ever added to that set by accident,
    # this is the test that says so.
    assert {c for c, r in roles.items() if r == "donor"} == set(
        CONE_SOURCE_DONOR_CONDITIONS
    )
    assert roles["full_cone"] == "self"
    assert roles["natural_scale"] == "self"
    assert roles["zero"] == "none"
    with pytest.raises(GenerativeError):
        cone_source_role("not_a_condition")


def test_expected_cone_source_requires_a_donor_for_donor_conditions():
    assert (
        expected_cone_source_example_id(
            "full_cone", example_id="a", donor_example_id="b"
        )
        == "a"
    )
    assert (
        expected_cone_source_example_id(
            "unrelated_cone", example_id="a", donor_example_id="b"
        )
        == "b"
    )
    assert (
        expected_cone_source_example_id("zero", example_id="a", donor_example_id="b")
        is None
    )
    with pytest.raises(GenerativeError, match="needs a donor"):
        expected_cone_source_example_id(
            "unrelated_cone", example_id="a", donor_example_id=None
        )


def test_pursuit_never_selects_one_atom_twice():
    """The runner maps token id -> coefficient through a dict.

    ``dict(zip(token_ids, coefficients))`` silently collapses duplicates, so a
    pursuit that could select the same atom twice would misassign coefficients
    to generators. The pursuit's ``used`` mask is what prevents that; this
    pins the guarantee the runner depends on.
    """
    from jlens.pursuit import JSpaceDictionary, PursuitSettings, gradient_pursuit

    torch.manual_seed(3)
    dictionary = JSpaceDictionary(torch.randn(24, 8), layer=1)
    targets = torch.randn(5, 8)
    result = gradient_pursuit(
        targets, dictionary, PursuitSettings(k=8, correlation_chunk_size=None)
    )
    for item in range(targets.shape[0]):
        record = result.to_records()[item]
        assert len(record["token_ids"]) == len(set(record["token_ids"]))
        assert len(result.active_token_ids(item)) == len(
            set(result.active_token_ids(item))
        )


# ----------------------------------------------------------- end-to-end run


#: Per-example signature scale. Distinct magnitudes as well as distinct
#: directions, so both the fingerprint and the recorded norm identify the
#: example on their own.
_SIGNATURE_SCALE = {"mock-a": 3.0, "mock-b": 7.0}


def _signature_activation(example_id: str, layer: int, d_model: int = 8):
    """A per-(example, layer) activation no other example can produce.

    One-hot at a slot chosen by the example, scaled by the example, offset by
    the layer. Two different examples cannot collide, and the same example at
    two layers cannot either, so any record carrying the wrong fingerprint is
    unambiguously a mix-up.
    """
    order = sorted(_SIGNATURE_SCALE)
    vector = torch.zeros(d_model, dtype=torch.float32)
    vector[order.index(example_id)] = _SIGNATURE_SCALE[example_id]
    vector[len(order) + (layer % (d_model - len(order)))] = 1.0
    return vector


@pytest.fixture()
def signature_run(experiment, monkeypatch, tmp_path):  # noqa: F811 - pytest fixture
    """Run the real runner on the mock with identifiable source activations."""
    runner = _import_runner()
    monkeypatch.setattr(runner, "load_gemma4", _mock_load_gemma4)

    prompt_to_example = {}
    manifest = json.loads(
        Path(experiment["config"]["benchmark"]["manifest_path"]).read_text(
            encoding="utf-8"
        )
    )
    for split in ("dev", "heldout"):
        for example in manifest[split]:
            prompt_to_example[example["source_prompt"]] = example["example_id"]
            if example.get("control_prompt"):
                prompt_to_example[example["control_prompt"]] = (
                    example["example_id"] + "/control"
                )

    def fake_capture(model, prompt, layer, position):
        example_id = prompt_to_example[prompt]
        if example_id.endswith("/control"):
            # The control prompt must stay distinguishable from the source.
            return -_signature_activation(example_id.split("/")[0], layer)
        return _signature_activation(example_id, layer)

    monkeypatch.setattr(runner, "capture_activation", fake_capture)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_generative_validation.py",
            "--config",
            experiment["config_path"],
            "--allow-model-load",
            "--runs-root",
            experiment["runs_root"],
            "--limit-examples",
            "2",
        ],
    )
    runner.main()

    run_dir = next(
        p
        for p in Path(experiment["runs_root"]).iterdir()
        if p.name.startswith("generative_")
    )
    records = [
        json.loads(line)
        for line in (run_dir / "artifacts" / "records.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert records
    return {"run_dir": run_dir, "records": records, "manifest": manifest}


def test_every_record_carries_the_full_identity_block(signature_run):
    required = (
        "example_id",
        "source_example_id",
        "cone_source_example_id",
        "donor_example_id",
        "source_prompt",
        "target_phrase",
        "source_layer",
        "source_activation_norm",
        "source_activation_sha256",
        "cone_norm",
        "cone_sha256",
        "injected_delta_sha256",
    )
    for record in signature_run["records"]:
        provenance = record["provenance"]
        missing = [key for key in required if key not in provenance]
        assert not missing, (record["vector_condition"], missing)


def test_source_activation_provenance_identifies_the_right_example(signature_run):
    """The recorded activation fingerprint is the example's own, at its own layer.

    This is the direct test of the Mandela/Black Hole worry: if any condition
    ever built a record from another example's captured activation, the hash
    on that record would be the other example's.
    """
    by_example = {
        example["example_id"]: example
        for example in signature_run["manifest"]["dev"]
    }
    for record in signature_run["records"]:
        provenance = record["provenance"]
        example_id = record["example_id"]
        layer = record["source_layer"]
        expected = _signature_activation(example_id, layer)
        expected_norm, expected_sha = vector_identity(expected)

        assert provenance["source_example_id"] == example_id
        assert provenance["source_activation_sha256"] == expected_sha
        assert provenance["source_activation_norm"] == pytest.approx(expected_norm)
        assert provenance["source_prompt"] == by_example[example_id]["source_prompt"]
        assert provenance["target_phrase"] == by_example[example_id]["target_phrase"]
        assert record["target_phrase"] == by_example[example_id]["target_phrase"]

        # And explicitly: never the *other* example's activation.
        for other_id in by_example:
            if other_id == example_id:
                continue
            _, other_sha = vector_identity(_signature_activation(other_id, layer))
            assert provenance["source_activation_sha256"] != other_sha


def test_cone_provenance_matches_the_condition_role(signature_run):
    """Only the unrelated-cone controls may carry another example's cone."""
    donor_conditions = set(CONE_SOURCE_DONOR_CONDITIONS)
    seen_donor = False
    for record in signature_run["records"]:
        provenance = record["provenance"]
        condition = record["vector_condition"]
        expected = expected_cone_source_example_id(
            condition,
            example_id=record["example_id"],
            donor_example_id=provenance["donor_example_id"],
        )
        assert provenance["cone_source_example_id"] == expected, condition
        if condition in donor_conditions:
            seen_donor = True
            assert provenance["cone_source_example_id"] != record["example_id"]
            assert provenance["cone_source_example_id"] == provenance[
                "donor_example_id"
            ]
        else:
            assert provenance["cone_source_example_id"] in (
                None,
                record["example_id"],
            ), condition
    assert seen_donor, "no unrelated-cone record was emitted; the check was vacuous"


def test_donor_is_the_other_same_split_example_and_never_the_example_itself(
    signature_run,
):
    heldout_ids = {
        example["example_id"] for example in signature_run["manifest"]["heldout"]
    }
    dev_ids = {example["example_id"] for example in signature_run["manifest"]["dev"]}
    for record in signature_run["records"]:
        donor = record["provenance"]["donor_example_id"]
        assert donor is not None
        assert donor != record["example_id"]
        assert donor in dev_ids
        assert donor not in heldout_ids


def test_unrelated_cone_really_injects_the_donor_cone(signature_run):
    """The donor-sourced record's cone fingerprint is the donor's own cone.

    Asserted by equality against the donor's ``full_cone`` record at the same
    source layer, so this checks the actual vector rather than a label the
    runner wrote next to it.
    """
    full_by_key = {
        (r["example_id"], r["source_layer"]): r["provenance"]["cone_sha256"]
        for r in signature_run["records"]
        if r["vector_condition"] == "full_cone"
    }
    assert full_by_key
    checked = 0
    for record in signature_run["records"]:
        if record["vector_condition"] not in CONE_SOURCE_DONOR_CONDITIONS:
            continue
        donor = record["provenance"]["donor_example_id"]
        key = (donor, record["source_layer"])
        if key not in full_by_key:
            continue
        assert record["provenance"]["cone_sha256"] == full_by_key[key]
        assert record["provenance"]["cone_sha256"] != full_by_key[
            (record["example_id"], record["source_layer"])
        ]
        checked += 1
    assert checked, "no donor-sourced record could be compared to the donor's cone"


def test_identity_is_stable_across_layers_prompts_and_conditions(signature_run):
    """One fingerprint per (example, layer) — never per prompt or condition.

    The activation is captured before the receiver-prompt and condition loops,
    so every record sharing (example, layer) must share its fingerprint. A
    per-prompt or per-condition difference would mean the loops re-capture (or
    mutate) it, which is how stale-loop-variable bugs show up.
    """
    groups: dict[tuple[str, int], set[str]] = {}
    prompts: dict[tuple[str, int], set[str]] = {}
    for record in signature_run["records"]:
        key = (record["example_id"], record["source_layer"])
        groups.setdefault(key, set()).add(
            record["provenance"]["source_activation_sha256"]
        )
        prompts.setdefault(key, set()).add(record["neutral_prompt_id"])
    assert groups
    for key, digests in groups.items():
        assert len(digests) == 1, key
    # The grouping is only meaningful if several prompts and layers really ran.
    assert len({layer for _, layer in groups}) >= 2
    assert all(len(ids) >= 2 for ids in prompts.values())

    # Fingerprints are unique across (example, layer) pairs, so "all records in
    # a group agree" is not satisfiable by everything being identical.
    all_digests = {next(iter(d)) for d in groups.values()}
    assert len(all_digests) == len(groups)


def test_injected_delta_fingerprint_is_present_exactly_when_a_vector_was_injected(
    signature_run,
):
    for record in signature_run["records"]:
        provenance = record["provenance"]
        if record["vector_condition"] == "none":
            assert provenance["injected_delta_sha256"] is None
            assert record["delta_norm"] is None
        else:
            assert provenance["injected_delta_sha256"] is not None


def test_pursuit_log_records_source_prompt_and_activation_identity(signature_run):
    pursuits = json.loads(
        (signature_run["run_dir"] / "artifacts" / "pursuits.json").read_text(
            encoding="utf-8"
        )
    )
    assert pursuits
    for entry in pursuits:
        expected = _signature_activation(entry["example_id"], entry["layer"])
        _, expected_sha = vector_identity(expected)
        assert entry["source_activation_sha256"] == expected_sha
        assert entry["source_prompt"]


def test_baseline_generation_is_a_property_of_the_prompt_not_the_example(
    signature_run,
):
    """The receiver prompt carries no example information — so the unsteered
    decode is the same string for every example.

    This is the alternative explanation for the observed repeated "Black Hole"
    decodes, and it is a structural fact about the design rather than a
    contingent one: ``none`` injects nothing, and the prompt is identical
    across examples, so the same prompt must produce the same text no matter
    which example the record is filed under. A future change that made the
    receiver prompt example-dependent would break this test — and would also
    invalidate the whole comparison, which is why it is worth pinning.
    """
    by_prompt: dict[str, set[str]] = {}
    for record in signature_run["records"]:
        if record["vector_condition"] != "none":
            continue
        if record.get("generated_text") is None:
            continue
        by_prompt.setdefault(record["neutral_prompt_id"], set()).add(
            record["generated_text"]
        )
    assert by_prompt, "no undecoded-baseline record to compare"
    for prompt_id, texts in by_prompt.items():
        assert len(texts) == 1, (prompt_id, texts)
