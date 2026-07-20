# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Mathematical tests for jlens.pursuit (CPU, synthetic dictionaries).

These tests pin the algorithm's *synthetic* behaviour exactly (recovery,
nonnegativity, determinism, monotonicity, serialization). Real-model
decompositions are approximate by nature — the paper reports the J-space
component captures no more than ~10% of activation variance — so nothing
here should be read as a claim about real-activation reconstruction
quality; that distinction is deliberate.
"""

import json

import pytest
import torch

from jlens.lens import JacobianLens
from jlens.pursuit import (
    PAD_INDEX,
    JSpaceDictionary,
    PursuitSettings,
    gradient_pursuit,
    load_records,
    topk_correlation_baseline,
)


def orthonormal_dictionary(d: int = 16, seed: int = 0) -> JSpaceDictionary:
    torch.manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(d, d, dtype=torch.float64))
    return JSpaceDictionary(Q.float(), layer=0, provenance={"kind": "orthonormal"})


def correlated_dictionary(
    n_atoms: int = 64, d: int = 16, seed: int = 0
) -> JSpaceDictionary:
    """Unit-norm atoms sharing a common component (pairwise correlated)."""
    torch.manual_seed(seed)
    common = torch.randn(d)
    atoms = torch.randn(n_atoms, d) + 1.5 * common
    atoms = atoms / atoms.norm(dim=-1, keepdim=True)
    return JSpaceDictionary(atoms, layer=0, provenance={"kind": "correlated"})


# ---------------------------------------------------------------- recovery


def test_exact_recovery_orthonormal_k_gt_1():
    """A known sparse nonnegative combination in an orthonormal dictionary
    is recovered exactly (atoms, coefficients, and near-zero residual)."""
    dictionary = orthonormal_dictionary()
    true_ids = [3, 7, 11]
    true_coeffs = [2.0, 1.0, 0.5]
    target = sum(
        c * dictionary.atoms[i] for i, c in zip(true_ids, true_coeffs, strict=True)
    )
    result = gradient_pursuit(target, dictionary, PursuitSettings(k=3))
    assert result.stop_reasons == ["residual_tol"] or result.stop_reasons == [
        "max_atoms"
    ]
    assert sorted(result.token_ids[0, : int(result.n_selected[0])].tolist()) == true_ids
    recovered = {
        int(i): float(c)
        for i, c in zip(result.token_ids[0], result.coefficients[0], strict=True)
        if int(i) != PAD_INDEX
    }
    for i, c in zip(true_ids, true_coeffs, strict=True):
        assert recovered[i] == pytest.approx(c, abs=1e-4)
    assert float(result.residual_norm[0]) < 1e-4
    assert float(result.explained_fraction[0]) == pytest.approx(1.0, abs=1e-6)


def test_exact_recovery_k_equals_1():
    dictionary = orthonormal_dictionary()
    target = 3.0 * dictionary.atoms[5]
    result = gradient_pursuit(target, dictionary, PursuitSettings(k=1))
    assert int(result.n_selected[0]) == 1
    assert int(result.token_ids[0, 0]) == 5
    assert float(result.coefficients[0, 0]) == pytest.approx(3.0, abs=1e-5)
    assert float(result.residual_norm[0]) < 1e-5


def test_approximate_recovery_correlated_atoms():
    """With correlated atoms and additive noise, recovery is approximate:
    most of the energy is explained and error decreases with k."""
    dictionary = correlated_dictionary()
    torch.manual_seed(1)
    true_ids = [10, 20, 30]
    target = (
        2.0 * dictionary.atoms[10]
        + 1.5 * dictionary.atoms[20]
        + 1.0 * dictionary.atoms[30]
        + 0.05 * torch.randn(16)
    )
    result_k5 = gradient_pursuit(target, dictionary, PursuitSettings(k=5))
    assert float(result_k5.explained_fraction[0]) > 0.9
    assert float(result_k5.relative_residual[0]) < 0.35
    result_k1 = gradient_pursuit(target, dictionary, PursuitSettings(k=1))
    assert float(result_k5.residual_norm[0]) <= float(result_k1.residual_norm[0]) + 1e-6
    # At least one true atom is found even under correlation (exact synthetic
    # recovery is NOT expected here — that is the point of this test).
    selected = set(result_k5.token_ids[0, : int(result_k5.n_selected[0])].tolist())
    assert selected & set(true_ids)


# ----------------------------------------------------------- constraints


def test_coefficients_are_nonnegative():
    dictionary = correlated_dictionary(seed=2)
    torch.manual_seed(3)
    targets = torch.randn(8, 16)  # arbitrary signs
    result = gradient_pursuit(targets, dictionary, PursuitSettings(k=6))
    assert bool((result.coefficients >= 0).all())
    # Reconstruction really is the claimed nonnegative combination.
    for b in range(8):
        n = int(result.n_selected[b])
        recon = torch.zeros(16)
        for i, c in zip(result.token_ids[b, :n], result.coefficients[b, :n], strict=True):
            recon += float(c) * dictionary.atoms[int(i)]
        torch.testing.assert_close(recon, result.reconstruction[b], atol=1e-4, rtol=1e-4)


def test_stops_when_no_atom_positively_correlated():
    """Under c >= 0, a target anti-aligned with every atom admits no useful
    atom; the pursuit must stop empty rather than force a selection."""
    d = 8
    atoms = torch.eye(d)
    dictionary = JSpaceDictionary(atoms, layer=0)
    target = -torch.ones(d)
    result = gradient_pursuit(target, dictionary, PursuitSettings(k=4))
    assert result.stop_reasons == ["no_positive_correlation"]
    assert int(result.n_selected[0]) == 0
    torch.testing.assert_close(result.residual[0], target)
    assert float(result.explained_fraction[0]) == pytest.approx(0.0)


# ------------------------------------------------- determinism / ties


def test_deterministic_across_runs():
    dictionary = correlated_dictionary(seed=4)
    torch.manual_seed(5)
    targets = torch.randn(4, 16)
    a = gradient_pursuit(targets, dictionary, PursuitSettings(k=5))
    b = gradient_pursuit(targets, dictionary, PursuitSettings(k=5))
    torch.testing.assert_close(a.token_ids, b.token_ids)
    torch.testing.assert_close(a.coefficients, b.coefficients)
    assert a.stop_reasons == b.stop_reasons
    assert json.dumps(a.to_records(), sort_keys=True) == json.dumps(
        b.to_records(), sort_keys=True
    )


def test_tie_break_prefers_lowest_token_id():
    """Two identical atoms tie exactly; the lower index must win."""
    d = 8
    atoms = torch.eye(d)
    atoms = torch.cat([atoms, atoms[3:4]], dim=0)  # atom 8 duplicates atom 3
    dictionary = JSpaceDictionary(atoms, layer=0)
    target = atoms[3].clone()
    result = gradient_pursuit(target, dictionary, PursuitSettings(k=1))
    assert int(result.token_ids[0, 0]) == 3  # not 8


def test_duplicate_atoms_never_selected_twice():
    d = 8
    atoms = torch.eye(d)
    atoms = torch.cat([atoms, atoms[3:4] * 0.999999], dim=0)  # near-duplicate
    dictionary = JSpaceDictionary(atoms, layer=0)
    target = atoms[3] + 0.3 * atoms[5]
    result = gradient_pursuit(target, dictionary, PursuitSettings(k=4))
    n = int(result.n_selected[0])
    chosen = result.token_ids[0, :n].tolist()
    assert len(chosen) == len(set(chosen))
    history = result.residual_norm_history[0]
    assert bool((history[1:] <= history[:-1] + 1e-6).all())


# ------------------------------------------------------------ residuals


def test_residual_norm_history_is_monotone_nonincreasing():
    """The backtracking update never accepts a worsening step, so the
    recorded residual norms must be non-increasing for every item."""
    dictionary = correlated_dictionary(n_atoms=48, seed=6)
    torch.manual_seed(7)
    targets = torch.randn(6, 16) * torch.tensor([0.1, 1.0, 10.0, 1.0, 1.0, 1.0]).unsqueeze(-1)
    result = gradient_pursuit(targets, dictionary, PursuitSettings(k=8))
    history = result.residual_norm_history
    assert bool((history[:, 1:] <= history[:, :-1] + 1e-5).all())
    # residual + reconstruction == target exactly (up to float32 arithmetic)
    torch.testing.assert_close(
        result.reconstruction + result.residual, targets, atol=1e-5, rtol=1e-5
    )


# ------------------------------------------------ shapes / orientation


def test_shape_validation():
    dictionary = orthonormal_dictionary(d=16)
    with pytest.raises(ValueError, match="targets must be"):
        gradient_pursuit(torch.randn(4, 8), dictionary, PursuitSettings(k=2))
    with pytest.raises(ValueError, match="targets must be"):
        gradient_pursuit(torch.randn(2, 3, 16), dictionary, PursuitSettings(k=2))
    single = gradient_pursuit(torch.randn(16), dictionary, PursuitSettings(k=2))
    assert single.token_ids.shape == (1, 2)


def test_settings_validation():
    dictionary = orthonormal_dictionary(d=16)
    with pytest.raises(ValueError, match="k must be >= 1"):
        gradient_pursuit(torch.randn(16), dictionary, PursuitSettings(k=0))
    with pytest.raises(ValueError, match="exceeds dictionary size"):
        gradient_pursuit(torch.randn(16), dictionary, PursuitSettings(k=17))


def test_dictionary_orientation_matches_lens_readout():
    """Atom v must be the direction whose inner product with h equals token
    v's J-lens logit: <(W_U J)_v, h> == (W_U (J h))_v."""
    torch.manual_seed(8)
    d, vocab = 6, 12
    J = torch.randn(d, d)
    W_U = torch.randn(vocab, d)
    lens = JacobianLens(jacobians={2: J}, n_prompts=1, d_model=d)
    dictionary = JSpaceDictionary.from_lens(lens, 2, W_U)
    torch.testing.assert_close(dictionary.atoms, W_U @ J)
    h = torch.randn(d)
    lens_logits = W_U @ lens.transport(h, 2)
    dict_logits = dictionary.atoms @ h
    torch.testing.assert_close(dict_logits, lens_logits, atol=1e-5, rtol=1e-5)
    assert dictionary.provenance["final_norm_weight_folded"] is False


def test_dictionary_final_norm_weight_folding():
    torch.manual_seed(9)
    d, vocab = 6, 12
    J = torch.randn(d, d)
    W_U = torch.randn(vocab, d)
    w = torch.rand(d) + 0.5
    lens = JacobianLens(jacobians={0: J}, n_prompts=1, d_model=d)
    folded = JSpaceDictionary.from_lens(lens, 0, W_U, final_norm_weight=w)
    torch.testing.assert_close(folded.atoms, (W_U * w) @ J)
    assert folded.provenance["final_norm_weight_folded"] is True
    with pytest.raises(ValueError, match="final_norm_weight"):
        JSpaceDictionary.from_lens(lens, 0, W_U, final_norm_weight=torch.ones(3))


def test_dictionary_from_lens_validates_inputs():
    lens = JacobianLens(jacobians={1: torch.eye(4)}, n_prompts=1, d_model=4)
    with pytest.raises(ValueError, match="not in fitted layers"):
        JSpaceDictionary.from_lens(lens, 3, torch.randn(8, 4))
    with pytest.raises(ValueError, match="unembedding_weight"):
        JSpaceDictionary.from_lens(lens, 1, torch.randn(8, 5))
    # Building a dictionary must not mutate the lens matrices.
    before = lens.jacobians[1].clone()
    JSpaceDictionary.from_lens(lens, 1, torch.randn(8, 4))
    torch.testing.assert_close(lens.jacobians[1], before)


# --------------------------------------------------- degenerate inputs


def test_zero_target():
    dictionary = orthonormal_dictionary()
    result = gradient_pursuit(torch.zeros(16), dictionary, PursuitSettings(k=3))
    assert result.stop_reasons == ["zero_target"]
    assert int(result.n_selected[0]) == 0
    assert float(result.relative_residual[0]) == 0.0
    assert float(result.explained_fraction[0]) == 0.0
    assert torch.isfinite(result.coefficients).all()


def test_nan_inf_rejected():
    dictionary = orthonormal_dictionary()
    bad = torch.zeros(16)
    bad[0] = float("nan")
    with pytest.raises(ValueError, match="NaN/Inf"):
        gradient_pursuit(bad, dictionary, PursuitSettings(k=2))
    bad[0] = float("inf")
    with pytest.raises(ValueError, match="NaN/Inf"):
        gradient_pursuit(bad, dictionary, PursuitSettings(k=2))
    with pytest.raises(ValueError, match="NaN/Inf"):
        JSpaceDictionary(torch.full((4, 4), float("nan")), layer=0)


def test_mixed_batch_stop_reasons():
    d = 8
    dictionary = JSpaceDictionary(torch.eye(d), layer=0)
    targets = torch.stack([torch.zeros(d), dictionary.atoms[2] * 2.0, -torch.ones(d)])
    result = gradient_pursuit(targets, dictionary, PursuitSettings(k=2))
    assert result.stop_reasons[0] == "zero_target"
    assert result.stop_reasons[1] in ("residual_tol", "max_atoms")
    assert result.stop_reasons[2] == "no_positive_correlation"
    assert int(result.token_ids[1, 0]) == 2


# ------------------------------------------------------- serialization


def test_records_round_trip(tmp_path):
    dictionary = correlated_dictionary(seed=10)
    torch.manual_seed(11)
    targets = torch.randn(3, 16)
    result = gradient_pursuit(targets, dictionary, PursuitSettings(k=4))
    path = str(tmp_path / "records.json")
    result.save_records(path)
    loaded = load_records(path)
    assert loaded == result.to_records()
    assert all(r["schema"] == "jlens.pursuit.result.v1" for r in loaded)
    assert all(r["requested_k"] == 4 for r in loaded)
    # Settings and dictionary provenance travel with every record.
    assert loaded[0]["settings"]["normalize_atoms"] is True
    assert loaded[0]["dictionary_provenance"]["kind"] == "correlated"


def test_load_records_rejects_unknown_schema(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([{"schema": "something.else"}]), encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected record schema"):
        load_records(str(path))


# ------------------------------------------------------------ baselines


def test_pursuit_beats_random_atom_baseline():
    """Random atoms fitted with the same nonnegative update explain far
    less than pursuit-selected atoms (fixed seeds; methodological sanity
    check, not a general theorem)."""
    dictionary = correlated_dictionary(n_atoms=64, seed=12)
    torch.manual_seed(13)
    target = (
        2.0 * dictionary.atoms[7] + 1.0 * dictionary.atoms[21] + 0.05 * torch.randn(16)
    )
    pursuit = gradient_pursuit(target, dictionary, PursuitSettings(k=4))

    generator = torch.Generator().manual_seed(99)
    random_ids = torch.randperm(64, generator=generator)[:4]
    atoms_S = dictionary.atoms[random_ids]
    coeff = torch.zeros(4)
    residual = target.clone()
    for _ in range(50):
        grad = atoms_S @ residual
        direction = grad @ atoms_S
        denom = (direction * direction).sum().clamp_min(1e-30)
        alpha = (residual * direction).sum() / denom
        new_coeff = (coeff + alpha * grad).clamp_min(0.0)
        new_residual = target - new_coeff @ atoms_S
        if float(new_residual.norm()) <= float(residual.norm()):
            coeff, residual = new_coeff, new_residual
    assert float(pursuit.residual_norm[0]) < float(residual.norm())


def test_pursuit_no_worse_than_topk_similarity_baseline():
    """On this synthetic case pursuit's residual-driven selection matches or
    beats one-shot top-k-by-similarity selection (fixed seed; documents that
    the two are different procedures — top-token similarity is not sparse
    decomposition)."""
    dictionary = correlated_dictionary(n_atoms=64, seed=14)
    torch.manual_seed(15)
    target = 2.0 * dictionary.atoms[3] + 1.0 * dictionary.atoms[40]
    settings = PursuitSettings(k=4)
    pursuit = gradient_pursuit(target, dictionary, settings)
    baseline = topk_correlation_baseline(target, dictionary, settings)
    assert baseline.stop_reasons == ["baseline_topk_correlation"]
    assert float(pursuit.residual_norm[0]) <= float(baseline.residual_norm[0]) + 1e-5


def test_chunked_correlations_match_unchunked():
    dictionary = correlated_dictionary(n_atoms=64, seed=16)
    torch.manual_seed(17)
    targets = torch.randn(3, 16)
    full = gradient_pursuit(targets, dictionary, PursuitSettings(k=5))
    chunked = gradient_pursuit(
        targets, dictionary, PursuitSettings(k=5, correlation_chunk_size=7)
    )
    torch.testing.assert_close(full.token_ids, chunked.token_ids)
    torch.testing.assert_close(full.coefficients, chunked.coefficients)


def test_fp16_atom_storage_computes_in_float32():
    torch.manual_seed(18)
    atoms = torch.randn(32, 16)
    dictionary = JSpaceDictionary(atoms.half(), layer=0)
    result = gradient_pursuit(torch.randn(2, 16), dictionary, PursuitSettings(k=3))
    assert result.coefficients.dtype == torch.float32
    assert result.reconstruction.dtype == torch.float32
    assert torch.isfinite(result.reconstruction).all()
