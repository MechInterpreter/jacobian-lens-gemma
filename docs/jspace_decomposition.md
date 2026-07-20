# Sparse J-space decomposition by gradient pursuit

Methods documentation for the `jspace-gradient-pursuit` branch: what is
implemented, exactly which parts come from the paper, and which choices the
paper leaves open. Source of truth for the method:
[Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/index.html)
(Anthropic, 2026), Methods, "The J-Space". Implementation:
[`jlens/pursuit.py`](../jlens/pursuit.py) (decomposition),
[`jlens/cones.py`](../jlens/cones.py) (cone records and trajectories),
[`jlens/ignition.py`](../jlens/ignition.py) (candidate-ignition
diagnostics, explicitly labeled as such).

## What the paper specifies (implemented exactly)

- **J-lens vectors.** "The rows of `W_U J_ℓ`" are the J-lens vectors at
  layer ℓ — one direction in *source-layer residual space* per vocabulary
  token, over the **full vocabulary** (262 144 for Gemma 4 E4B). In this
  fork `J_ℓ` is the fitted matrix from the frozen pilot lens
  (`JacobianLens` convention: rows = target dims, transport `J @ h`) and
  `W_U` is Gemma's tied `lm_head.weight`. Built by
  `JSpaceDictionary.from_lens`.
- **J-space.** The set of sparse nonnegative combinations of J-lens
  vectors; "geometrically, for a given k, the J-space corresponds to a
  union of k-dimensional cones, one for each possible set of k J-lens
  vectors."
- **Decomposition objective.** For an activation `h_ℓ`, solve by gradient
  pursuit for a sparse nonnegative combination of `k` J-lens vectors that
  best reconstructs `h_ℓ`:

  ```
  minimize   || h_ℓ − Σ_{v∈S} c_v · d_v ||₂²
  subject to  c_v ≥ 0,  |S| ≤ k,   d_v = (W_U J_ℓ) row v
  ```

  The reconstruction is the activation's **J-space component**, the
  difference its **non-J-space component**, and the coefficients its
  "local J-space coordinates". The reconstruction space is the source-layer
  residual space (d = 2560).
- **Sparsity `k`.** The paper uses k = 25 (J-space component
  identification), k = 16 (concept vectors), k = 10 (ablations), and states
  k is "typically no more than 25". These are the supported values used as
  defaults in the Colab workflow.
- **Expectation calibration.** The paper reports the J-space component
  "typically accounts for only a small fraction of total activation
  variance (varying by layer, but never more than 10%)". High residuals on
  real activations are expected, not a bug signal.

## What the paper leaves open (configuration, documented — not canonical)

The paper names **gradient pursuit** and defers algorithm internals; the
canonical reference for the algorithm family is Blumensath & Davies (2008),
"Gradient Pursuits", IEEE Trans. Signal Processing 56(6). Choices this
implementation makes, all exposed on `PursuitSettings` / dictionary
construction and recorded in every output record:

| Choice | Default | Rationale |
|---|---|---|
| Atom selection | max correlation of residual with atoms, ties → lowest token id | standard pursuit selection; deterministic tie-break |
| Selection normalization (`normalize_atoms`) | on (divide by atom L2 norms) | scale-invariant selection; paper silent; coefficients always refer to raw atoms |
| Nonnegativity handling | gradient step on active set + exact line search, projection to `c ≥ 0`, backtracking; step accepted only if the residual norm does not increase | keeps the cheap gradient-pursuit update while guaranteeing monotone residuals |
| Inner refinement (`refine_steps`) | 2 extra passes per iteration | cheap; improves the active-set fit toward the NNLS optimum |
| Stopping | `k` atoms, or no positive correlation, or relative residual ≤ `tol_relative_residual` (default 0) | nonneg pursuit cannot use an atom with non-positive correlation |
| Final-norm weight folding (`final_norm_weight`) | **off** — atoms are literally rows of `W_U J_ℓ` | the paper's formula omits the norm weight; Gemma's actual readout multiplies by RMSNorm's `1 + weight`, so folding is available as an explicit opt-in and recorded in provenance |

The correlation step over the full vocabulary can be chunked
(`correlation_chunk_size`) to bound memory; results are identical (tested).

## Conceptual boundaries (deliberate)

- **Gradient pursuit is per-activation.** One pursuit returns the active
  set (≤ k J-lens vectors) and coefficients that locally approximate one
  activation — it identifies the *local cone* containing/approximating that
  point. It is **not** clustering and says nothing global by itself.
- **Cone aggregation is a separate stage.** Recurring active sets across
  examples/positions/layers are counted and grouped transparently in
  `jlens/cones.py` (exact-signature grouping and threshold-based similarity
  grouping); no opaque clustering method is used at this stage.
- **A cone signature is not a concept claim.** The signature is a
  deterministic function of the nonzero active set (sorted token ids +
  SHA-256). Two identical signatures mean the same token set was selected —
  not that they constitute one universal semantic concept.
- **Candidate ignition ≠ ignition.** `jlens/ignition.py` computes
  layer-transition diagnostics (reconstruction-quality jumps, active-set
  stability, coefficient concentration, output alignment) that are
  *candidate* signals only. Alternative explanations that must be excluded
  before any stronger claim: lens quality varying by layer, tokenizer
  granularity effects, fitting-corpus (WikiText) bias, and ordinary
  late-layer vocabulary alignment (the logit lens also converges to the
  output at layers 35–38 — see [`pilot_report.md`](pilot_report.md)).

## Invariants enforced by tests

Synthetic (exact) behaviour — distinct from approximate real-model
decomposition, which no test claims to validate:

- exact recovery of known sparse nonnegative combinations in orthonormal
  dictionaries (k = 1 and k > 1); approximate recovery with correlated
  atoms;
- nonnegative coefficients always; empty result (with recorded stop
  reason) when no atom is positively correlated;
- deterministic results and lowest-token-id tie-breaking; no atom selected
  twice even with duplicate atoms;
- monotone non-increasing residual-norm history;
- shape/orientation pinned against the lens transport
  (`⟨d_v, h⟩ = (W_U J h)_v`);
- zero-vector and NaN/Inf handling; fp16 atom storage with float32
  compute; chunked = unchunked;
- JSON round-trip of result records (`jlens.pursuit.result.v1` schema);
- explicit baseline comparisons (top-k similarity, random atoms) showing
  pursuit is a different, better procedure than top-token ranking on the
  synthetic cases.
