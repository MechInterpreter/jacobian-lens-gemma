# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""The physical/normalized mapping and the immutability of the layer list.

Confusing a physical layer with a normalized one is a silent error: both are
small integers, both are called "layer", and the wrong one produces a plausible
number rather than a crash. These tests pin the four labels the study reports.
"""

import pytest

from jlens.mmlocalize.layers import (
    LAYER_SET_VERSION,
    LOCALIZATION_LAYERS,
    MODEL_N_LAYERS,
    REFERENCE_LAYER,
    ImmutableLayerSetError,
    assert_immutable_layer_set,
    describe_layers,
    earlier_layers,
    layer_manifest,
    layer_ref,
    normalized_depth,
)

#: The mapping stated in the study's design, which the report and every
#: artifact must reproduce exactly.
EXPECTED_NORMALIZED = {20: 48, 26: 62, 32: 76, 38: 90}


def test_the_model_depth_is_the_gemma_4_e4b_one():
    assert MODEL_N_LAYERS == 42


@pytest.mark.parametrize(("physical", "expected"), sorted(EXPECTED_NORMALIZED.items()))
def test_physical_maps_to_the_stated_normalized_depth(physical, expected):
    assert normalized_depth(physical) == expected
    assert layer_ref(physical).normalized == expected
    assert layer_ref(physical).physical == physical


def test_normalized_depth_follows_the_documented_rule():
    for physical in range(MODEL_N_LAYERS + 1):
        assert normalized_depth(physical) == round(100.0 * physical / MODEL_N_LAYERS)


def test_normalized_depth_rejects_layers_outside_the_model():
    with pytest.raises(ValueError):
        normalized_depth(43)
    with pytest.raises(ValueError):
        normalized_depth(-1)
    with pytest.raises(ValueError):
        normalized_depth(20, n_model_layers=0)


def test_a_normalized_number_is_never_accepted_as_a_physical_one():
    """48, 62, 76 and 90 are valid physical indices in a 42-layer model only by
    accident of being integers — three of them are not, and the one that is
    (none here) must still map somewhere else."""
    for normalized in EXPECTED_NORMALIZED.values():
        if normalized > MODEL_N_LAYERS:
            with pytest.raises(ValueError):
                normalized_depth(normalized)
        else:  # pragma: no cover - defensive; none of the four are <= 42
            assert normalized_depth(normalized) != normalized


def test_the_layer_set_is_the_predetermined_four_in_depth_order():
    assert LOCALIZATION_LAYERS == (20, 26, 32, 38)
    assert list(LOCALIZATION_LAYERS) == sorted(LOCALIZATION_LAYERS)
    assert REFERENCE_LAYER == 38
    assert REFERENCE_LAYER == max(LOCALIZATION_LAYERS)


def test_earlier_layers_excludes_the_reference():
    assert earlier_layers() == [20, 26, 32]
    assert REFERENCE_LAYER not in earlier_layers()


def test_the_immutable_layer_set_accepts_only_itself():
    assert assert_immutable_layer_set([20, 26, 32, 38]) == LOCALIZATION_LAYERS
    assert assert_immutable_layer_set((20, 26, 32, 38)) == LOCALIZATION_LAYERS


@pytest.mark.parametrize(
    "requested",
    [
        (20, 26, 32),            # a layer dropped after it failed
        (20, 26, 32, 38, 14),    # a layer added after seeing results
        (38, 32, 26, 20),        # reordered, so "earliest" changes meaning
        (32,),                   # the "just set LAYERS=(32,)" shortcut
        (),
    ],
)
def test_changing_the_layer_set_is_refused(requested):
    with pytest.raises(ImmutableLayerSetError) as error:
        assert_immutable_layer_set(requested)
    assert "fixed" in str(error.value)


def test_the_refusal_names_what_was_added_and_removed():
    with pytest.raises(ImmutableLayerSetError) as error:
        assert_immutable_layer_set((20, 26, 32, 14))
    message = str(error.value)
    assert "added [14]" in message
    assert "removed [38]" in message


def test_roles_distinguish_the_reference_from_the_layers_under_test():
    assert "positive reference" in layer_ref(38).role
    for physical in (20, 26, 32):
        assert "eligibility must be earned" in layer_ref(physical).role
    assert "outside the predetermined layer set" in layer_ref(14).role


def test_the_manifest_carries_both_numbering_systems():
    manifest = layer_manifest()
    assert manifest["version"] == LAYER_SET_VERSION
    assert manifest["physical_layers"] == [20, 26, 32, 38]
    assert manifest["reference_layer"] == 38
    assert manifest["earlier_layers"] == [20, 26, 32]
    assert manifest["n_model_layers"] == 42
    for entry in manifest["layers"]:
        assert entry["normalized"] == EXPECTED_NORMALIZED[entry["physical"]]
    assert "round(100 * physical / n_model_layers)" in manifest["mapping_rule"]


def test_the_printed_description_shows_both_numbers_for_every_layer():
    text = describe_layers()
    for physical, normalized in EXPECTED_NORMALIZED.items():
        assert f"L{physical}" in text
        assert f"~{normalized}" in text
    assert "never interchanged" in text
