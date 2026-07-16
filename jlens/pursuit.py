# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Sparse nonnegative J-space decomposition by gradient pursuit.

Implements the decomposition step of the paper's J-space analysis
(https://transformer-circuits.pub/2026/workspace/index.html, Methods,
"The J-Space"), model-agnostically:

**Dictionary (exact paper convention).** The J-lens vectors at layer ``l``
are the rows of ``W_U @ J_l`` — one vector per vocabulary token, living in
the *source-layer residual space*. ``J_l`` is the fitted lens matrix
(:class:`jlens.lens.JacobianLens` convention: rows = target dims, transport
``J @ h``) and ``W_U`` the unembedding matrix ``[vocab, d_model]``. The
paper's formula does not include the final norm's elementwise weight; Gemma
readouts actually multiply by it, so :meth:`JSpaceDictionary.from_lens`
exposes ``final_norm_weight`` as an explicit opt-in (off by default to
follow the paper's stated convention; the choice is recorded in the
dictionary's provenance either way).

**Objective.** Given an activation ``h`` at layer ``l``, solve::

    minimize  || h - sum_{v in S} c_v * d_v ||_2^2
    subject to  c_v >= 0,  |S| <= k

The reconstruction ``sum c_v d_v`` is the activation's *J-space component*,
``h`` minus it is the *non-J-space component*, and the coefficients are its
local J-space coordinates. Geometrically the feasible set for a given
active set ``S`` is a cone; the J-space is the union of these cones. The
active set returned for one activation identifies the *local* cone that
best approximates it — recurring active sets across many activations are a
separate, later aggregation question (see :mod:`jlens.cones`), not
something a single pursuit answers.

**Algorithm.** The paper names gradient pursuit (Blumensath & Davies 2008,
"Gradient Pursuits", IEEE TSP 56(6)) and fixes only ``k`` (their runs use
10, 16, and 25; "typically no more than 25"). The variant here adapts the
standard algorithm to the nonnegativity constraint:

1. Selection: the atom with the largest correlation with the current
   residual (correlations divided by atom norms when
   ``normalize_atoms=True``); ties break deterministically toward the
   lowest token id. If no unselected atom has positive correlation, the
   pursuit stops — under ``c >= 0`` no new atom can reduce the error.
2. Coefficient update: a gradient step on the active-set coefficients
   (direction ``D_S^T r``) with exact line search, then projection onto
   ``c >= 0``. If the projected step does not reduce the residual, the step
   is halved (backtracking) up to ``max_backtracks`` times; a step is only
   ever accepted if it does not increase the residual, so residual norms
   are non-increasing across iterations by construction.
3. Stop on: ``k`` atoms selected, no positive correlation, or relative
   residual at or below ``tol_relative_residual``.

Everything the paper leaves open — atom normalization for selection, the
number of inner refinement steps, tolerances — is exposed on
:class:`PursuitSettings` and recorded in every result; none of it is
presented as canonical.

Scope guards: inputs are validated (finite, shape-compatible), Gemma
parameters and the fitted lens are never modified (the dictionary is a
derived copy), and top-token *ranking* is not conflated with decomposition
— the rank-based readout lives in :mod:`jlens.controls` /
:mod:`jlens.evaluation`; :func:`topk_correlation_baseline` exists here
precisely so the two can be compared explicitly.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import torch

from jlens.lens import JacobianLens

#: Sentinel for unused slots in fixed-width index tensors.
PAD_INDEX = -1

#: Default row-chunk size for whole-dictionary validation/norm passes.
#: 16384 rows of a [262144, 2560] float32 dictionary bound the transient
#: allocations of one pass to ~160 MB (float32 chunk view is free; the
#: isfinite bool mask is 16384*2560 = 40 MB) instead of the ~671 MB bool
#: mask a whole-tensor ``torch.isfinite(atoms)`` materializes — the exact
#: allocation that OOMed dictionary construction on an L4.
VALIDATION_CHUNK_ROWS = 16384


def validate_finite(tensor: torch.Tensor, *, chunk_rows: int | None = VALIDATION_CHUNK_ROWS) -> bool:
    """``torch.isfinite(tensor).all()`` with bounded peak memory.

    Checks the tensor in row chunks so the boolean mask temporary is at most
    ``chunk_rows`` rows instead of the full tensor; each chunk's mask is
    released before the next is allocated. Mathematically identical to the
    whole-tensor check (finiteness is elementwise); ``chunk_rows=None``
    falls back to the single-pass check.
    """
    if chunk_rows is None or tensor.ndim == 0 or tensor.shape[0] <= chunk_rows:
        return bool(torch.isfinite(tensor).all())
    for start in range(0, tensor.shape[0], chunk_rows):
        finite = torch.isfinite(tensor[start : start + chunk_rows]).all()
        if not bool(finite):
            return False
        del finite
    return True


def _chunked_row_norms(tensor: torch.Tensor, *, chunk_rows: int | None = VALIDATION_CHUNK_ROWS) -> torch.Tensor:
    """Float32 L2 norms per row without materializing a full float32 copy of
    a lower-precision tensor (only ``chunk_rows`` rows are upcast at a
    time). Identical per-row arithmetic to ``tensor.float().norm(dim=-1)``."""
    if tensor.dtype == torch.float32 or chunk_rows is None or tensor.shape[0] <= chunk_rows:
        return tensor.float().norm(dim=-1)
    norms = torch.empty(tensor.shape[0], dtype=torch.float32, device=tensor.device)
    for start in range(0, tensor.shape[0], chunk_rows):
        chunk = tensor[start : start + chunk_rows].float()
        norms[start : start + chunk.shape[0]] = chunk.norm(dim=-1)
        del chunk
    return norms


@dataclass(frozen=True)
class PursuitSettings:
    """Configuration for :func:`gradient_pursuit`.

    Attributes:
        k: Maximum number of atoms (the paper's sparsity parameter; the
            paper uses 10/16/25 and "typically no more than 25").
        normalize_atoms: Divide selection correlations by atom L2 norms so
            selection is scale-invariant. The paper does not state this
            choice; coefficients always refer to the *raw* atoms either way.
        refine_steps: Extra gradient/backtrack coefficient passes after each
            atom is added (0 = single update per iteration).
        tol_relative_residual: Stop when ``||r|| / ||h||`` falls to or below
            this value.
        max_backtracks: Step halvings before an update is rejected.
        correlation_chunk_size: If set, compute the ``[batch, vocab]``
            correlation matrix in chunks of this many atoms to bound peak
            memory on large vocabularies.
    """

    k: int = 25
    normalize_atoms: bool = True
    refine_steps: int = 2
    tol_relative_residual: float = 0.0
    max_backtracks: int = 20
    correlation_chunk_size: int | None = None

    def validate(self, n_atoms: int) -> None:
        if self.k < 1:
            raise ValueError(f"k must be >= 1, got {self.k}")
        if self.k > n_atoms:
            raise ValueError(f"k={self.k} exceeds dictionary size {n_atoms}")
        if self.refine_steps < 0 or self.max_backtracks < 0:
            raise ValueError("refine_steps and max_backtracks must be >= 0")
        if not 0.0 <= self.tol_relative_residual < 1.0:
            raise ValueError("tol_relative_residual must be in [0, 1)")


class JSpaceDictionary:
    """The J-lens vectors at one layer: rows of ``W_U @ J_l``.

    Attributes:
        atoms: ``[n_atoms, d_model]`` — atom ``v`` is token ``v``'s J-lens
            vector in source residual space.
        atom_norms: ``[n_atoms]`` L2 norms of the atoms (float32).
        layer: Source layer the dictionary belongs to.
        provenance: Serializable record of how the atoms were built.
    """

    def __init__(
        self,
        atoms: torch.Tensor,
        *,
        layer: int,
        provenance: dict | None = None,
    ) -> None:
        if atoms.ndim != 2:
            raise ValueError(f"atoms must be [n_atoms, d_model], got {tuple(atoms.shape)}")
        if not validate_finite(atoms):
            raise ValueError("dictionary atoms contain NaN/Inf")
        self.atoms = atoms
        self.atom_norms = _chunked_row_norms(atoms)
        self.layer = layer
        self.provenance = provenance or {}

    @property
    def n_atoms(self) -> int:
        return self.atoms.shape[0]

    @property
    def d_model(self) -> int:
        return self.atoms.shape[1]

    @property
    def device(self) -> torch.device:
        return self.atoms.device

    @classmethod
    def from_lens(
        cls,
        lens: JacobianLens,
        layer: int,
        unembedding_weight: torch.Tensor,
        *,
        final_norm_weight: torch.Tensor | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        build_chunk_rows: int | None = None,
    ) -> JSpaceDictionary:
        """Build the layer-``layer`` dictionary from a fitted lens.

        Args:
            lens: The fitted (frozen) lens; its matrices are read, never
                modified.
            layer: Which fitted layer's ``J_l`` to use.
            unembedding_weight: ``W_U`` as ``[vocab, d_model]`` (for Gemma,
                the tied ``lm_head.weight``). Read-only.
            final_norm_weight: If given, an *effective* elementwise
                multiplier folded into ``W_U`` (columns scaled), i.e. atoms
                become rows of ``(W_U * w) @ J_l``. The paper's convention
                omits this; for Gemma's RMSNorm the effective multiplier is
                ``1 + norm.weight`` and must be passed already in that form.
                Off (``None``) by default.
            device: Where to build/store the atoms (defaults to
                ``unembedding_weight.device``).
            dtype: Storage dtype for the atoms (float32 default; correlations
                are always computed in float32).
            build_chunk_rows: If set, build the product in row chunks of this
                size: only ``build_chunk_rows`` rows of ``W_U`` are upcast to
                float32 at a time and each chunk's product is written
                straight into the preallocated output. This avoids the two
                transient full-size tensors of the one-shot path (the float32
                copy of a lower-precision ``W_U`` and, for non-float32
                ``dtype``, the full float32 product held next to its
                converted copy) — for Gemma 4 E4B each is ~2.7 GB. ``None``
                (default) keeps the single ``W @ J`` matmul.

        Peak-memory notes (Gemma 4 E4B, vocab 262144, d_model 2560): float32
        atoms are 2.7 GB and are the *output*, so they always exist; the
        avoidable costs are the float32 ``W_U`` copy (2.7 GB, skipped by
        ``build_chunk_rows`` when the model weight is bf16), the finiteness
        mask (671 MB unchunked, ~40 MB chunked — see
        :func:`validate_finite`), a float32 copy for norms when atoms are
        stored in a lower precision (chunked in ``__init__``), and the
        ``[batch, vocab]`` correlation matrix during pursuit (bounded by
        ``PursuitSettings.correlation_chunk_size``). Callers decomposing
        several layers should drop each layer's dictionary (``del``) before
        building the next; nothing here retains it.
        """
        if layer not in lens.jacobians:
            raise ValueError(
                f"layer {layer} not in fitted layers {lens.source_layers}"
            )
        if unembedding_weight.ndim != 2 or unembedding_weight.shape[1] != lens.d_model:
            raise ValueError(
                f"unembedding_weight must be [vocab, {lens.d_model}], got "
                f"{tuple(unembedding_weight.shape)}"
            )
        device = torch.device(device) if device is not None else unembedding_weight.device
        w = None
        if final_norm_weight is not None:
            w = final_norm_weight.detach().to(device=device, dtype=torch.float32)
            if w.shape != (lens.d_model,):
                raise ValueError(
                    f"final_norm_weight must be [{lens.d_model}], got {tuple(w.shape)}"
                )
        J = lens.jacobians[layer].to(device=device, dtype=torch.float32)
        source = unembedding_weight.detach()
        if build_chunk_rows is None:
            W = source.to(device=device, dtype=torch.float32)
            if w is not None:
                W = W * w  # scale columns: rows of (W_U * w) @ J
            atoms = (W @ J).to(dtype)
        else:
            if build_chunk_rows < 1:
                raise ValueError(
                    f"build_chunk_rows must be >= 1, got {build_chunk_rows}"
                )
            n_atoms = source.shape[0]
            atoms = torch.empty(n_atoms, lens.d_model, dtype=dtype, device=device)
            for start in range(0, n_atoms, build_chunk_rows):
                chunk = source[start : start + build_chunk_rows].to(
                    device=device, dtype=torch.float32
                )
                if w is not None:
                    chunk = chunk * w
                atoms[start : start + chunk.shape[0]] = (chunk @ J).to(dtype)
                del chunk
        return cls(
            atoms,
            layer=layer,
            provenance={
                "convention": "rows of W_U @ J_l (paper Methods, 'The J-Space')",
                "layer": layer,
                "n_atoms": atoms.shape[0],
                "d_model": atoms.shape[1],
                "final_norm_weight_folded": final_norm_weight is not None,
                "storage_dtype": str(dtype),
                "lens_n_prompts": lens.n_prompts,
            },
        )


@dataclass
class PursuitResult:
    """Batched result of :func:`gradient_pursuit`.

    All tensors share the batch dimension ``B``. Fixed-width slots beyond
    ``n_selected[b]`` hold :data:`PAD_INDEX` / ``0.0``.

    Attributes:
        token_ids: ``[B, k]`` selected atom (token) indices in selection
            order.
        coefficients: ``[B, k]`` nonnegative coefficients over the *raw*
            atoms, aligned with ``token_ids``.
        n_selected: ``[B]`` number of atoms actually selected.
        n_iterations: ``[B]`` pursuit iterations executed.
        reconstruction: ``[B, d]`` the J-space component ``D_S c``.
        residual: ``[B, d]`` ``h - reconstruction`` (non-J-space component).
        target_norm / residual_norm: ``[B]`` L2 norms.
        relative_residual: ``[B]`` ``||r|| / ||h||`` (0 for zero targets).
        explained_fraction: ``[B]`` ``1 - ||r||^2 / ||h||^2`` (0 for zero
            targets).
        stop_reasons: length-``B`` list of
            ``"max_atoms" | "no_positive_correlation" | "residual_tol" |
            "zero_target"`` (or ``"baseline_topk_correlation"`` from the
            baseline).
        residual_norm_history: ``[B, k+1]`` residual norm before iteration
            1 and after each executed iteration (padded with the final value).
        settings: The :class:`PursuitSettings` used.
        dictionary_provenance: Copied from the dictionary.
    """

    token_ids: torch.Tensor
    coefficients: torch.Tensor
    n_selected: torch.Tensor
    n_iterations: torch.Tensor
    reconstruction: torch.Tensor
    residual: torch.Tensor
    target_norm: torch.Tensor
    residual_norm: torch.Tensor
    relative_residual: torch.Tensor
    explained_fraction: torch.Tensor
    stop_reasons: list[str]
    residual_norm_history: torch.Tensor
    settings: PursuitSettings
    dictionary_provenance: dict = field(default_factory=dict)

    def active_token_ids(self, item: int) -> list[int]:
        """Selected token ids with strictly positive coefficients for one
        batch item (the *effective* active set; selection order preserved)."""
        n = int(self.n_selected[item])
        ids = self.token_ids[item, :n].tolist()
        coeffs = self.coefficients[item, :n].tolist()
        return [int(i) for i, c in zip(ids, coeffs, strict=True) if c > 0]

    def to_records(self) -> list[dict]:
        """One JSON-safe dict per batch item (stable schema; see
        :mod:`jlens.cones` for the provenance-wrapped artifact record)."""
        records = []
        for b in range(self.token_ids.shape[0]):
            n = int(self.n_selected[b])
            records.append(
                {
                    "schema": "jlens.pursuit.result.v1",
                    "requested_k": self.settings.k,
                    "n_selected": n,
                    "n_iterations": int(self.n_iterations[b]),
                    "token_ids": [int(i) for i in self.token_ids[b, :n]],
                    "coefficients": [float(c) for c in self.coefficients[b, :n]],
                    "target_norm": float(self.target_norm[b]),
                    "residual_norm": float(self.residual_norm[b]),
                    "relative_residual": float(self.relative_residual[b]),
                    "explained_fraction": float(self.explained_fraction[b]),
                    "stop_reason": self.stop_reasons[b],
                    "residual_norm_history": [
                        float(x)
                        for x in self.residual_norm_history[
                            b, : int(self.n_iterations[b]) + 1
                        ]
                    ],
                    "settings": asdict(self.settings),
                    "dictionary_provenance": self.dictionary_provenance,
                }
            )
        return records

    def save_records(self, path: str) -> None:
        """Write :meth:`to_records` as pretty JSON."""
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_records(), handle, indent=2, ensure_ascii=False)


def load_records(path: str) -> list[dict]:
    """Load records written by :meth:`PursuitResult.save_records`, with a
    schema check."""
    with open(path, encoding="utf-8") as handle:
        records = json.load(handle)
    for record in records:
        if record.get("schema") != "jlens.pursuit.result.v1":
            raise ValueError(
                f"{path}: unexpected record schema {record.get('schema')!r}"
            )
    return records


def _validate_targets(
    targets: torch.Tensor, dictionary: JSpaceDictionary
) -> torch.Tensor:
    if targets.ndim == 1:
        targets = targets.unsqueeze(0)
    if targets.ndim != 2 or targets.shape[-1] != dictionary.d_model:
        raise ValueError(
            f"targets must be [batch, {dictionary.d_model}] or "
            f"[{dictionary.d_model}], got {tuple(targets.shape)}"
        )
    if not torch.isfinite(targets).all():
        raise ValueError("targets contain NaN/Inf")
    return targets.to(device=dictionary.device, dtype=torch.float32)


def _correlations(
    residual: torch.Tensor,
    dictionary: JSpaceDictionary,
    settings: PursuitSettings,
) -> torch.Tensor:
    """``[B, n_atoms]`` correlations ``<d_v, r>`` (optionally norm-divided),
    computed in float32, chunked over atoms if configured."""
    chunk = settings.correlation_chunk_size
    if chunk is None or chunk >= dictionary.n_atoms:
        g = residual @ dictionary.atoms.float().T
    else:
        parts = []
        for start in range(0, dictionary.n_atoms, chunk):
            block = dictionary.atoms[start : start + chunk].float()
            parts.append(residual @ block.T)
        g = torch.cat(parts, dim=-1)
    if settings.normalize_atoms:
        g = g / dictionary.atom_norms.clamp_min(1e-30)
    return g


def _first_argmax(values: torch.Tensor) -> torch.Tensor:
    """Deterministic argmax over the last dim: on ties, the lowest index
    wins on every backend (plain ``argmax`` tie order is not guaranteed)."""
    n = values.shape[-1]
    top = values.max(dim=-1, keepdim=True).values
    arange = torch.arange(n, device=values.device).expand_as(values)
    candidates = torch.where(values == top, arange, n)
    return candidates.min(dim=-1).values


def gradient_pursuit(
    targets: torch.Tensor,
    dictionary: JSpaceDictionary,
    settings: PursuitSettings | None = None,
) -> PursuitResult:
    """Sparse nonnegative decomposition of ``targets`` against ``dictionary``.

    Args:
        targets: ``[batch, d_model]`` (or a single ``[d_model]`` vector) of
            activations to decompose. Rejected if NaN/Inf; computation is
            float32 on the dictionary's device.
        dictionary: The layer's J-lens vectors.
        settings: Algorithm configuration (defaults to
            :class:`PursuitSettings`, i.e. the paper's k=25 with this
            module's documented defaults for everything the paper leaves
            open).

    Returns:
        A :class:`PursuitResult`; see the module docstring for the exact
        objective and algorithm.
    """
    settings = settings or PursuitSettings()
    settings.validate(dictionary.n_atoms)
    h = _validate_targets(targets, dictionary)
    B, d = h.shape
    k = settings.k
    device = h.device

    token_ids = torch.full((B, k), PAD_INDEX, dtype=torch.long, device=device)
    coefficients = torch.zeros(B, k, dtype=torch.float32, device=device)
    n_selected = torch.zeros(B, dtype=torch.long, device=device)
    n_iterations = torch.zeros(B, dtype=torch.long, device=device)
    stop_codes = ["max_atoms"] * B

    target_norm = h.norm(dim=-1)
    residual = h.clone()
    residual_norm = residual.norm(dim=-1)
    history = torch.zeros(B, k + 1, dtype=torch.float32, device=device)
    history[:, 0] = residual_norm

    active = target_norm > 0
    for b in torch.nonzero(~active).flatten().tolist():
        stop_codes[b] = "zero_target"
    used = torch.zeros(B, dictionary.n_atoms, dtype=torch.bool, device=device)

    def masked_residual_update(
        atoms_S: torch.Tensor,  # [B, m, d] raw atoms of the active set
        coeff_S: torch.Tensor,  # [B, m]
        rows: torch.Tensor,  # bool [B] — items to update
        n_steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run up to ``n_steps`` gradient/backtrack passes on ``coeff_S`` for
        the selected rows; returns updated (coeff_S, residual_rows_norm)."""
        nonlocal residual
        for _ in range(n_steps):
            r = residual  # [B, d]
            grad = torch.einsum("bmd,bd->bm", atoms_S, r)  # D_S^T r
            direction_img = torch.einsum("bm,bmd->bd", grad, atoms_S)
            denom = (direction_img * direction_img).sum(-1)
            numer = (r * direction_img).sum(-1)
            step_rows = rows & (denom > 0)
            if not bool(step_rows.any()):
                break
            alpha = torch.where(denom > 0, numer / denom.clamp_min(1e-30), 0.0)
            # Backtracking acceptance: never increase the residual norm.
            pending = step_rows.clone()
            accepted_coeff = coeff_S.clone()
            for _try in range(settings.max_backtracks + 1):
                if not bool(pending.any()):
                    break
                cand_coeff = (coeff_S + alpha.unsqueeze(-1) * grad).clamp_min(0.0)
                cand_recon = torch.einsum("bm,bmd->bd", cand_coeff, atoms_S)
                cand_res = h - cand_recon
                cand_norm = cand_res.norm(dim=-1)
                improved = pending & (cand_norm <= residual.norm(dim=-1) + 0.0)
                strictly = pending & (cand_norm < residual.norm(dim=-1) - 1e-12)
                take = improved
                accepted_coeff = torch.where(
                    take.unsqueeze(-1), cand_coeff, accepted_coeff
                )
                residual = torch.where(take.unsqueeze(-1), cand_res, residual)
                pending = pending & ~strictly & ~take
                alpha = torch.where(pending, alpha * 0.5, alpha)
            coeff_S = accepted_coeff
        return coeff_S, residual.norm(dim=-1)

    iteration = 0
    while iteration < k and bool(active.any()):
        g = _correlations(residual, dictionary, settings)
        g = g.masked_fill(used, float("-inf"))
        best_idx = _first_argmax(g)
        best_val = g.gather(-1, best_idx.unsqueeze(-1)).squeeze(-1)

        newly_stopped = active & (best_val <= 0)
        for b in torch.nonzero(newly_stopped).flatten().tolist():
            stop_codes[b] = "no_positive_correlation"
        active = active & ~newly_stopped
        if not bool(active.any()):
            break

        # Record the new atom for active rows.
        slot = n_selected.clamp_max(k - 1)
        token_ids[active, slot[active]] = best_idx[active]
        used[active, best_idx[active]] = True
        n_selected = n_selected + active.long()
        n_iterations = n_iterations + active.long()

        # Gather the active-set atoms (fixed width m = iteration+1; padded
        # slots use atom 0 with coefficient locked at 0 via the mask below).
        m = iteration + 1
        gather_ids = token_ids[:, :m].clamp_min(0)
        atoms_S = dictionary.atoms[gather_ids].float()  # [B, m, d]
        valid_slots = token_ids[:, :m] != PAD_INDEX
        atoms_S = atoms_S * valid_slots.unsqueeze(-1)

        coeff_S = coefficients[:, :m]
        coeff_S, residual_norm = masked_residual_update(
            atoms_S, coeff_S, active, 1 + settings.refine_steps
        )
        coefficients[:, :m] = coeff_S * valid_slots

        history[:, iteration + 1] = residual_norm
        # Tolerance stop.
        rel = residual_norm / target_norm.clamp_min(1e-30)
        tol_stop = active & (rel <= settings.tol_relative_residual)
        for b in torch.nonzero(tol_stop).flatten().tolist():
            stop_codes[b] = "residual_tol"
        active = active & ~tol_stop
        iteration += 1

    # Freeze history padding at each item's final residual norm.
    final_norm = residual.norm(dim=-1)
    for b in range(B):
        history[b, int(n_iterations[b]) + 1 :] = final_norm[b]

    reconstruction = h - residual
    tn = target_norm.clamp_min(1e-30)
    relative = torch.where(target_norm > 0, final_norm / tn, torch.zeros_like(tn))
    explained = torch.where(
        target_norm > 0,
        1.0 - (final_norm**2) / (tn**2),
        torch.zeros_like(tn),
    )
    return PursuitResult(
        token_ids=token_ids,
        coefficients=coefficients,
        n_selected=n_selected,
        n_iterations=n_iterations,
        reconstruction=reconstruction,
        residual=residual,
        target_norm=target_norm,
        residual_norm=final_norm,
        relative_residual=relative,
        explained_fraction=explained.clamp(max=1.0),
        stop_reasons=stop_codes,
        residual_norm_history=history,
        settings=settings,
        dictionary_provenance=dict(dictionary.provenance),
    )


def topk_correlation_baseline(
    targets: torch.Tensor,
    dictionary: JSpaceDictionary,
    settings: PursuitSettings | None = None,
) -> PursuitResult:
    """Baseline: select the ``k`` atoms most correlated with the *target*
    (one shot, no residual feedback), then fit nonnegative coefficients with
    the same update rule as the pursuit.

    Exists so notebooks and tests can compare gradient pursuit against naive
    top-k similarity explicitly — top-token ranking is **not** sparse
    decomposition, and this function is the honest version of that
    comparison.
    """
    settings = settings or PursuitSettings()
    settings.validate(dictionary.n_atoms)
    h = _validate_targets(targets, dictionary)
    g = _correlations(h, dictionary, settings)
    top = g.topk(settings.k, dim=-1).indices  # [B, k]

    # Reuse the pursuit machinery on the restricted dictionary per item by
    # running gradient updates with the fixed active set.
    B = h.shape[0]
    atoms_S = dictionary.atoms[top].float()  # [B, k, d]
    coeff = torch.zeros(B, settings.k, dtype=torch.float32, device=h.device)
    residual = h.clone()
    for _ in range(1 + settings.refine_steps + settings.k):
        grad = torch.einsum("bmd,bd->bm", atoms_S, residual)
        direction_img = torch.einsum("bm,bmd->bd", grad, atoms_S)
        denom = (direction_img * direction_img).sum(-1).clamp_min(1e-30)
        alpha = ((residual * direction_img).sum(-1) / denom).unsqueeze(-1)
        new_coeff = (coeff + alpha * grad).clamp_min(0.0)
        new_residual = h - torch.einsum("bm,bmd->bd", new_coeff, atoms_S)
        better = new_residual.norm(dim=-1) <= residual.norm(dim=-1)
        coeff = torch.where(better.unsqueeze(-1), new_coeff, coeff)
        residual = torch.where(better.unsqueeze(-1), new_residual, residual)

    target_norm = h.norm(dim=-1)
    final_norm = residual.norm(dim=-1)
    tn = target_norm.clamp_min(1e-30)
    n_sel = torch.full((B,), settings.k, dtype=torch.long, device=h.device)
    history = torch.zeros(B, settings.k + 1, dtype=torch.float32, device=h.device)
    history[:, 0] = target_norm
    history[:, 1:] = final_norm.unsqueeze(-1)
    return PursuitResult(
        token_ids=top,
        coefficients=coeff,
        n_selected=n_sel,
        n_iterations=n_sel.clone(),
        reconstruction=h - residual,
        residual=residual,
        target_norm=target_norm,
        residual_norm=final_norm,
        relative_residual=torch.where(
            target_norm > 0, final_norm / tn, torch.zeros_like(tn)
        ),
        explained_fraction=torch.where(
            target_norm > 0, 1.0 - (final_norm**2) / (tn**2), torch.zeros_like(tn)
        ),
        stop_reasons=["baseline_topk_correlation"] * B,
        residual_norm_history=history,
        settings=settings,
        dictionary_provenance={
            **dictionary.provenance,
            "baseline": "topk_correlation (not the paper's pursuit)",
        },
    )
