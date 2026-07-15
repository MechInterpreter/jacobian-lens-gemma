# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Negative-control tests (CPU, tiny models)."""

import pytest
import torch

from jlens.controls import (
    adjacent_layer_mapping,
    control_lens,
    distant_layer_mapping,
    layer_mapped_lens,
    mapping_provenance,
    permute_rows,
    ranks_of_targets,
    scale_matched_random,
    shuffled_layer_mapping,
    topk_overlap,
    wrong_layer_lens,
    wrong_layer_mapping,
)
from jlens.fitting import fit
from jlens.lens import JacobianLens

from .tiny import TinyDecoder


@pytest.fixture(scope="module")
def fitted():
    model = TinyDecoder(n_layers=4, d_model=8)
    prompts = ["abcdefghij klmnop " * 3, "zyxwvutsrq ponmlk " * 3]
    lens = fit(model, prompts, source_layers=[0, 1, 2], dim_batch=4, max_seq_len=48)
    return model, lens


def test_permute_rows_preserves_entries_exactly():
    torch.manual_seed(0)
    J = torch.randn(8, 8)
    P = permute_rows(J, seed=3)
    assert P.shape == J.shape
    assert not torch.equal(P, J)
    assert torch.equal(P.norm(), J.norm())  # exact: same entries
    # Same multiset of rows, different arrangement.
    sorted_rows = lambda M: M[torch.argsort(M[:, 0])]  # noqa: E731
    torch.testing.assert_close(sorted_rows(P), sorted_rows(J))
    torch.testing.assert_close(permute_rows(J, seed=3), P)  # deterministic


def test_scale_matched_random_matches_scale():
    torch.manual_seed(0)
    J = torch.randn(8, 8) * 5
    R = scale_matched_random(J, seed=3)
    assert R.shape == J.shape
    torch.testing.assert_close(R.norm(), J.float().norm())
    torch.testing.assert_close(scale_matched_random(J, seed=3), R)
    assert not torch.allclose(scale_matched_random(J, seed=4), R)


def test_control_lens_kinds(fitted):
    _, lens = fitted
    for kind in ("permuted", "random"):
        control = control_lens(lens, kind, seed=0)
        assert control.source_layers == lens.source_layers
        assert control.d_model == lens.d_model
        for layer in lens.source_layers:
            assert not torch.allclose(
                control.jacobians[layer], lens.jacobians[layer]
            ), f"{kind} control equals fitted J at layer {layer}"
    with pytest.raises(ValueError, match="unknown control kind"):
        control_lens(lens, "identity")


def test_control_lens_layers_get_distinct_seeds(fitted):
    _, lens = fitted
    control = control_lens(lens, "random", seed=0)
    layers = lens.source_layers
    assert not torch.allclose(control.jacobians[layers[0]], control.jacobians[layers[1]])


def test_wrong_layer_lens_is_a_cyclic_shift(fitted):
    _, lens = fitted
    shifted = wrong_layer_lens(lens)
    layers = lens.source_layers
    for i, layer in enumerate(layers):
        expected = lens.jacobians[layers[(i + 1) % len(layers)]]
        torch.testing.assert_close(shifted.jacobians[layer], expected)


def test_wrong_layer_lens_needs_two_layers(fitted):
    _, lens = fitted
    single = JacobianLens(
        jacobians={0: lens.jacobians[0]}, n_prompts=lens.n_prompts, d_model=8
    )
    with pytest.raises(ValueError, match=">= 2"):
        wrong_layer_lens(single)


def test_topk_overlap_bounds():
    logits = torch.randn(5, 32)
    assert topk_overlap(logits, logits, k=10) == pytest.approx(1.0)
    disjoint_a = torch.zeros(1, 32)
    disjoint_b = torch.zeros(1, 32)
    disjoint_a[0, :10] = torch.arange(10, 0, -1).float()
    disjoint_b[0, 20:30] = torch.arange(10, 0, -1).float()
    assert topk_overlap(disjoint_a, disjoint_b, k=10) == pytest.approx(0.0)


def test_ranks_of_targets_matches_naive():
    torch.manual_seed(0)
    logits = torch.randn(6, 32)
    targets = torch.randint(0, 32, (6,))
    ranks = ranks_of_targets(logits, targets)
    for i in range(6):
        naive = int((logits[i] > logits[i, targets[i]]).sum())
        assert int(ranks[i]) == naive
    assert int(ranks_of_targets(logits[:1], logits[0].argmax().unsqueeze(0))) == 0


PILOT_LAYERS = [3, 7, 14, 21, 28, 35, 38]


def test_adjacent_layer_mapping_picks_nearest_other_layer():
    mapping = adjacent_layer_mapping(PILOT_LAYERS)
    assert mapping == {3: 7, 7: 3, 14: 21, 21: 28, 28: 35, 35: 38, 38: 35}
    # Ties (equidistant neighbours) break toward the deeper layer.
    assert adjacent_layer_mapping([0, 2, 4]) == {0: 2, 2: 4, 4: 2}
    with pytest.raises(ValueError, match=">= 2"):
        adjacent_layer_mapping([5])


def test_distant_layer_mapping_picks_farthest_layer():
    mapping = distant_layer_mapping(PILOT_LAYERS)
    assert mapping == {3: 38, 7: 38, 14: 38, 21: 3, 28: 3, 35: 3, 38: 3}
    with pytest.raises(ValueError, match=">= 2"):
        distant_layer_mapping([5])


def test_shuffled_layer_mapping_is_a_deterministic_derangement():
    mapping = shuffled_layer_mapping(PILOT_LAYERS, seed=7)
    assert sorted(mapping) == PILOT_LAYERS
    assert sorted(mapping.values()) == PILOT_LAYERS  # a permutation
    assert all(source != layer for layer, source in mapping.items())
    assert shuffled_layer_mapping(PILOT_LAYERS, seed=7) == mapping
    assert shuffled_layer_mapping(PILOT_LAYERS, seed=8) != mapping
    # Smallest possible case must terminate (the only derangement is a swap).
    assert shuffled_layer_mapping([1, 2], seed=0) == {1: 2, 2: 1}


def test_wrong_layer_mapping_documents_the_cyclic_control(fitted):
    """The explicit mapping must reproduce wrong_layer_lens exactly, so the
    historical control's provenance is recorded truthfully."""
    _, lens = fitted
    mapping = wrong_layer_mapping(lens.source_layers)
    via_mapping = layer_mapped_lens(lens, mapping)
    via_legacy = wrong_layer_lens(lens)
    for layer in lens.source_layers:
        torch.testing.assert_close(
            via_mapping.jacobians[layer], via_legacy.jacobians[layer]
        )


def test_layer_mapped_lens_uses_exactly_the_claimed_matrices(fitted):
    """Every layer-mapping control must hold, bit-for-bit, the fitted J of
    the layer its mapping claims — the core provenance guarantee."""
    _, lens = fitted
    for make in (adjacent_layer_mapping, distant_layer_mapping):
        mapping = make(lens.source_layers)
        control = layer_mapped_lens(lens, mapping)
        for layer, source in mapping.items():
            torch.testing.assert_close(
                control.jacobians[layer], lens.jacobians[source]
            )
    mapping = shuffled_layer_mapping(lens.source_layers, seed=3)
    control = layer_mapped_lens(lens, mapping)
    for layer, source in mapping.items():
        torch.testing.assert_close(
            control.jacobians[layer], lens.jacobians[source]
        )


def test_layer_mapped_lens_rejects_bad_mappings(fitted):
    _, lens = fitted
    with pytest.raises(ValueError, match="non-fitted"):
        layer_mapped_lens(lens, {0: 99})
    with pytest.raises(ValueError, match="cover exactly"):
        layer_mapped_lens(lens, {lens.source_layers[0]: lens.source_layers[1]})


def test_mapping_provenance_records_distances():
    rows = mapping_provenance({35: 38, 38: 3})
    assert rows == [
        {"applied_at_layer": 35, "jacobian_fitted_at_layer": 38, "layer_distance": 3},
        {"applied_at_layer": 38, "jacobian_fitted_at_layer": 3, "layer_distance": 35},
    ]


def test_controls_change_the_readout(fitted):
    """Applying a control transport must actually change lens logits, and the
    fitted J-lens must beat the permuted control at matching the model's own
    output from a mid-layer residual (meaningful-transport criterion)."""
    model, lens = fitted
    prompt = "abcdefghij klmnop qrstu " * 2
    lens_logits, model_logits, _ = lens.apply(model, prompt, layers=[1])
    permuted_logits, _, _ = control_lens(lens, "permuted", seed=0).apply(
        model, prompt, layers=[1]
    )
    assert not torch.allclose(lens_logits[1], permuted_logits[1])

    k = 5
    fitted_overlap = topk_overlap(lens_logits[1], model_logits, k)
    permuted_overlap = topk_overlap(permuted_logits[1], model_logits, k)
    assert fitted_overlap > permuted_overlap
