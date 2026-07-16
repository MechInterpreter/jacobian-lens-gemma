# Recurrence and stability methodology for J-space cone records

Methods documentation for the similarity-based recurrence, atom-frequency,
and stability analyses on branch `jspace-pursuit-analysis`. Implementation:
[`jlens/similarity.py`](../jlens/similarity.py) (metrics, grouping,
frequency statistics), [`jlens/jspace_analysis.py`](../jlens/jspace_analysis.py)
(run loading, integrity checking, aggregation),
[`scripts/analyze_jspace.py`](../scripts/analyze_jspace.py) (deterministic
CLI). Results on the completed run:
[`jspace_run_report.md`](jspace_run_report.md).

## Why exact signatures are not enough

The cone signature ([`jlens/cones.py`](../jlens/cones.py)) is a SHA-256
digest over the sorted positive-coefficient token ids of one decomposition.
It is exact, deterministic, and ideal for provenance — and, as a recurrence
detector, maximally brittle: two cones that share 9 of 10 atoms hash to
unrelated digests. On the completed Gemma 4 E4B run **no signature ever
repeated** (380 records per k, 380 unique digests, for k = 10, 16, and 25),
which by itself distinguishes nothing between "no recurring structure" and
"recurring structure with single-atom jitter". The exact signature is kept
unchanged; recurrence is now measured at four explicitly separated levels.

## Four levels of recurrence (kept separate by design)

| Level | Question | Function |
|---|---|---|
| Exact signatures | did the *identical* atom set recur? | `cone_signature`, `record_similarity(metric="exact")` |
| Similar supports | do atom *sets* overlap? | `jaccard_similarity`, `top_m_overlap` |
| Coefficient-similar cones | do atom sets overlap *with similar weights*? | `weighted_jaccard`, `sparse_cosine` |
| Recurring atoms | which individual atoms recur, regardless of set matching? | `atom_selection_frequencies`, `atom_enrichment`, `output_token_recurrence` |

A statement at one level is never promoted to another: two cones with
Jaccard 0.8 have similar supports; they are not thereby "the same concept",
and a globally frequent atom is not thereby a "universal concept".

## Metric definitions

All metrics operate on the *effective* active set (strictly positive
coefficients) and are pure functions with deterministic tie-breaking.

- **Jaccard** over token-id sets; two empty sets score 1.0 (consistent with
  `jlens.cones.active_set_overlap`).
- **Weighted Jaccard** `Σ min(a_v, b_v) / Σ max(a_v, b_v)` over nonnegative
  sparse coefficient maps. Sensitive to both support and magnitude;
  negative coefficients are rejected.
- **Sparse cosine** over the same maps (scale-invariant; 0.0 when either
  map has zero norm). Note the trajectory artifacts'
  `weighted_similarity` field is exactly this cosine.
- **Top-m overlap** `|top_m(a) ∩ top_m(b)| / m`, where `top_m` orders by
  descending coefficient with ties broken toward the lowest token id.
- **Frequency-adjusted variants**: coefficient maps multiplied by
  inverse-selection-frequency weights (below) before weighted Jaccard or
  cosine. Raw metrics are always reported alongside adjusted ones.

## Grouping: similarity groups, not clusters

`similarity_groups` computes connected components of "similarity ≥
threshold" **within one stratum** — by default (layer, k, format,
position); cross-layer comparison requires explicitly passing different
strata and labeling the result as such. Candidate pairs come from a
shared-atom inverted index: every metric above is exactly 0 for records
with disjoint supports, so those pairs are provably below any positive
threshold and are never scored. Union-find roots and output ordering are
fully deterministic (lowest index wins; groups sorted by stratum, then
size, then first index).

Two deliberate framing rules:

1. **Components are similarity groups.** A component proves a chain of
   pairwise-similar records exists; transitive chaining can join records
   whose direct similarity is below threshold. Nothing here validates a
   component as one semantic concept.
2. **No single threshold is trusted.** `threshold_sensitivity` reports
   group counts/sizes over a threshold grid (default 0.2–0.9), raw and
   frequency-adjusted, so every claim can be checked for threshold
   robustness.

Decoded-string similarity (multilingual, subword, or tokenizer-family
grouping — e.g. treating `' Canberra'`, `'Can'`, `'Canberra'` as one item)
is **not** folded into any metric; it is a separate exploratory analysis to
be labeled as such wherever used.

## Atom-frequency statistics

- **Selection frequency**: an atom counts once per record it appears in
  (presence, not coefficient mass), overall and per stratum.
- **Inverse-frequency weights**: `log(1 + N / (1 + count))` with `N` the
  record count — standard IDF smoothing, the only smoothing applied there.
- **Enrichment**: per (atom, stratum), observed count vs. expected count
  (overall count × stratum share of records) plus a log-odds of per-record
  presence in-stratum vs. out-of-stratum with Haldane–Anscombe correction
  (add 0.5 to every cell; documented, configurable).
- **Output-token conditioning**: for each atom, how often it is selected
  *as* the record's own model top-1 token vs. selected elsewhere.

## Determinism and non-mutation guarantees

The analysis never writes into a run directory (tested: byte-identical
tree hash after a full pipeline pass), performs no model download and no
GPU work, and produces byte-identical outputs across repeated invocations
(sorted iteration everywhere, fixed float formatting `%.8g`, sorted JSON
keys). Tests: [`tests/test_similarity.py`](../tests/test_similarity.py),
[`tests/test_jspace_analysis.py`](../tests/test_jspace_analysis.py).

## Usage

```bash
python scripts/analyze_jspace.py \
    --run-dir runs/jspace_20260716T170808536780_e4118850fb70 \
    --lens runs/pilot_20260715T200437612150_311fd108c23a/artifacts/lens.pt
# outputs under reports/<run_id>/ (integrity_summary.json,
# metrics_by_layer_k_format.csv, k_marginal_gains.csv,
# cross_k_stability.csv, transition_metrics.csv,
# candidate_ignition_summary.csv, atom_frequencies.csv,
# atom_enrichment.csv, similarity_groups.json,
# evaluation_control_summary.csv, evaluation_control_collisions.json,
# analysis_summary.json)
```

`--lens` is optional and adds CPU-only statistics of the fitted `J_l`
matrices to `analysis_summary.json`. The J-space atoms themselves (rows of
`W_U @ J_l`) additionally require Gemma's unembedding matrix and therefore
**cannot** be recomputed without loading the model; every atom-level number
in the reports comes from the recorded run artifacts.
