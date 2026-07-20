# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Tests for the bounded-memory paths in jlens.pursuit: chunked finite
validation, chunked norm computation, chunked dictionary construction, and
the execution-provenance record in jlens.metadata (CPU, synthetic)."""

import pytest
import torch

from jlens.lens import JacobianLens
from jlens.metadata import execution_record
from jlens.pursuit import (
    JSpaceDictionary,
    PursuitSettings,
    _chunked_row_norms,
    gradient_pursuit,
    validate_finite,
)


def lens_with_layer(d: int = 8, layer: int = 0, seed: int = 0) -> JacobianLens:
    g = torch.Generator().manual_seed(seed)
    return JacobianLens(
        jacobians={layer: torch.randn(d, d, generator=g)},
        n_prompts=1,
        d_model=d,
    )


# ------------------------------------------------------- validate_finite


def test_validate_finite_clean_tensor_all_paths():
    t = torch.randn(100, 7)
    assert validate_finite(t, chunk_rows=None)
    assert validate_finite(t, chunk_rows=8)
    assert validate_finite(t, chunk_rows=1)
    assert validate_finite(t, chunk_rows=10_000)  # chunk larger than tensor


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("row", [0, 50, 99])
def test_validate_finite_catches_nan_inf_in_any_chunk(bad_value, row):
    t = torch.randn(100, 7)
    t[row, 3] = bad_value
    assert not validate_finite(t, chunk_rows=8)
    assert not validate_finite(t, chunk_rows=None)
    # Chunked and unchunked agree.
    assert validate_finite(t, chunk_rows=8) == validate_finite(t, chunk_rows=None)


def test_validate_finite_zero_dim():
    assert validate_finite(torch.tensor(1.0))
    assert not validate_finite(torch.tensor(float("nan")))


def test_dictionary_init_rejects_nonfinite_via_chunked_check():
    atoms = torch.randn(64, 8)
    atoms[63, 0] = float("inf")  # last chunk with default chunking
    with pytest.raises(ValueError, match="NaN/Inf"):
        JSpaceDictionary(atoms, layer=0)


# ---------------------------------------------------- _chunked_row_norms


def test_chunked_row_norms_match_unchunked_fp16():
    atoms = torch.randn(100, 16).half()
    chunked = _chunked_row_norms(atoms, chunk_rows=7)
    reference = atoms.float().norm(dim=-1)
    assert torch.equal(chunked, reference)


def test_chunked_row_norms_float32_short_circuits():
    atoms = torch.randn(10, 4)
    assert torch.equal(_chunked_row_norms(atoms, chunk_rows=3), atoms.norm(dim=-1))


# --------------------------------------------------- chunked construction


def test_from_lens_chunked_build_matches_one_shot():
    lens = lens_with_layer(d=8)
    W = torch.randn(50, 8, generator=torch.Generator().manual_seed(1))
    one_shot = JSpaceDictionary.from_lens(lens, 0, W)
    for chunk in (1, 7, 50, 1000):
        chunked = JSpaceDictionary.from_lens(lens, 0, W, build_chunk_rows=chunk)
        torch.testing.assert_close(chunked.atoms, one_shot.atoms)
        torch.testing.assert_close(chunked.atom_norms, one_shot.atom_norms)
        assert chunked.provenance == one_shot.provenance


def test_from_lens_chunked_build_deterministic():
    lens = lens_with_layer(d=8)
    W = torch.randn(50, 8, generator=torch.Generator().manual_seed(2)).to(torch.bfloat16)
    a = JSpaceDictionary.from_lens(lens, 0, W, build_chunk_rows=9)
    b = JSpaceDictionary.from_lens(lens, 0, W, build_chunk_rows=9)
    assert torch.equal(a.atoms, b.atoms)


def test_from_lens_chunked_build_bf16_source_and_norm_folding():
    lens = lens_with_layer(d=8)
    W = torch.randn(50, 8, generator=torch.Generator().manual_seed(3)).to(torch.bfloat16)
    w = torch.rand(8) + 0.5
    one_shot = JSpaceDictionary.from_lens(lens, 0, W, final_norm_weight=w)
    chunked = JSpaceDictionary.from_lens(
        lens, 0, W, final_norm_weight=w, build_chunk_rows=13
    )
    torch.testing.assert_close(chunked.atoms, one_shot.atoms)
    assert chunked.provenance["final_norm_weight_folded"] is True


def test_from_lens_chunked_build_lower_precision_storage():
    lens = lens_with_layer(d=8)
    W = torch.randn(50, 8, generator=torch.Generator().manual_seed(4))
    one_shot = JSpaceDictionary.from_lens(lens, 0, W, dtype=torch.float16)
    chunked = JSpaceDictionary.from_lens(
        lens, 0, W, dtype=torch.float16, build_chunk_rows=8
    )
    assert chunked.atoms.dtype == torch.float16
    torch.testing.assert_close(chunked.atoms, one_shot.atoms)


def test_from_lens_rejects_bad_chunk_size():
    lens = lens_with_layer(d=8)
    W = torch.randn(50, 8)
    with pytest.raises(ValueError, match="build_chunk_rows"):
        JSpaceDictionary.from_lens(lens, 0, W, build_chunk_rows=0)


def test_pursuit_identical_on_chunked_and_one_shot_dictionaries():
    lens = lens_with_layer(d=16, seed=5)
    W = torch.randn(200, 16, generator=torch.Generator().manual_seed(6))
    targets = torch.randn(4, 16, generator=torch.Generator().manual_seed(7))
    settings = PursuitSettings(k=5)
    result_a = gradient_pursuit(
        targets, JSpaceDictionary.from_lens(lens, 0, W), settings
    )
    result_b = gradient_pursuit(
        targets,
        JSpaceDictionary.from_lens(lens, 0, W, build_chunk_rows=17),
        settings,
    )
    assert torch.equal(result_a.token_ids, result_b.token_ids)
    torch.testing.assert_close(result_a.coefficients, result_b.coefficients)
    torch.testing.assert_close(
        result_a.explained_fraction, result_b.explained_fraction
    )


# ------------------------------------------------------ execution record


def test_execution_record_consistent():
    record = execution_record(
        configured_allow_model_load=False,
        resolved_allow_model_load=True,
        model_loaded=True,
        override_source="notebook",
    )
    assert record["schema"] == "jlens.metadata.execution.v1"
    assert record["configured_allow_model_load"] is False
    assert record["resolved_allow_model_load"] is True
    assert record["model_loaded"] is True
    assert record["override_source"] == "notebook"


def test_execution_record_rejects_load_without_permission():
    with pytest.raises(ValueError, match="inconsistent"):
        execution_record(
            configured_allow_model_load=True,
            resolved_allow_model_load=False,
            model_loaded=True,
        )


def test_execution_record_requires_override_source_for_divergence():
    with pytest.raises(ValueError, match="override_source"):
        execution_record(
            configured_allow_model_load=False,
            resolved_allow_model_load=True,
            model_loaded=False,
        )


def test_execution_record_no_load_path():
    record = execution_record(
        configured_allow_model_load=False,
        resolved_allow_model_load=False,
        model_loaded=False,
    )
    assert record["model_loaded"] is False
    assert record["override_source"] is None
