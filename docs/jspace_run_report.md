# Gemma 4 E4B J-space gradient-pursuit run — analysis report

**Run:** [`runs/jspace_20260716T170808536780_e4118850fb70/`](../runs/jspace_20260716T170808536780_e4118850fb70/)
**Config:** [`configs/gemma_jspace_pursuit.yaml`](../configs/gemma_jspace_pursuit.yaml)
**Config fingerprint:** `sha256:e4118850fb70842b2fea162642f0265cb6c82dc0abfb357aca8eb455eb7f0253`
**Frozen pilot lens:** `sha256:7229c7562d1d55420b70abb13f481934649c4b01417bd851e97cedb47c96f474`
(from [`runs/pilot_20260715T200437612150_311fd108c23a/`](../runs/pilot_20260715T200437612150_311fd108c23a/))
**Model:** `google/gemma-4-E4B-it` @ `fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd`
**Implementation commit at run time:** `6442fff2c67bfb0e1c081cef8c58350820db5e4a`
**Run written:** 2026-07-16T17:09:59Z (NVIDIA A100-SXM4-40GB)

Every number in this report is recomputed deterministically from the run's
artifacts by [`scripts/analyze_jspace.py`](../scripts/analyze_jspace.py);
derived tables live under
[`reports/jspace_20260716T170808536780_e4118850fb70/`](../reports/jspace_20260716T170808536780_e4118850fb70/).
Methodology for the recurrence/stability analyses:
[`jspace_similarity_analysis.md`](jspace_similarity_analysis.md).
Interpretation boundaries of the decomposition itself:
[`jspace_decomposition.md`](jspace_decomposition.md).

## 1. Inventory and integrity

The run is complete and internally consistent
([`integrity_summary.json`](../reports/jspace_20260716T170808536780_e4118850fb70/integrity_summary.json)):

- 15/15 (layer, k) units for layers {14, 21, 28, 35, 38} × k ∈ {10, 16, 25};
  76 cone records each (38 held-out prompts — 30 plain, 8 chat — at
  positions −2 and −1), 1 140 decompositions total.
- Per k: 304 trajectory transitions, 304 candidate-ignition records, and a
  380-row recurring-signature table whose counts sum to the 380 cone
  records.
- Zero integrity issues: no truncated/malformed JSON, no duplicate or
  missing records, prompt hashes and per-record `model_top1_id` agree with
  `capture_meta`, all coefficients strictly positive and finite, no
  duplicate token ids within a cone, every residual history is monotone
  non-increasing, `n_selected`/`stop_reason` are mutually consistent, and
  every recomputed cone-signature digest matches the stored one.
- One provenance **note** (not an artifact error): the recorded config says
  `model.allow_model_load: false` while `load_info` shows Gemma was loaded.
  The config block stores the *static* notebook YAML; the resolved
  execution-time value was never written down. The completed artifact is
  left untouched; future runs should write
  `jlens.metadata.execution_record(...)` (added on this branch), which
  separates configured value, resolved value, override source, and whether
  a load actually happened — and refuses the inconsistent combination.

## 2. k = 10 vs 16 vs 25

Mean explained fraction (EF) of the J-space component, all 76 records per
cell ([`metrics_by_layer_k_format.csv`](../reports/jspace_20260716T170808536780_e4118850fb70/metrics_by_layer_k_format.csv)):

| layer | k=10 | k=16 | k=25 |
|---|---|---|---|
| 14 | 0.00170 | 0.00177 | 0.00179 |
| 21 | 0.000023 | 0.000024 | 0.000024 |
| 28 | 0.00730 | 0.00776 | 0.00806 |
| 35 | 0.01679 | 0.01746 | 0.01782 |
| 38 | 0.01514 | 0.01559 | 0.01582 |

This verifies the earlier review's approximate figures (0.17 % / 0.0023 % /
0.73 % / 1.68 % / 1.51 % at k=10) directly from the artifacts, and is
consistent with the paper's calibration that the J-space component is a
small fraction of activation variance.

Marginal gains ([`k_marginal_gains.csv`](../reports/jspace_20260716T170808536780_e4118850fb70/k_marginal_gains.csv)):
k10→k25 improves EF by only **4.5–10.4 % relative** (largest at layer 28),
and k16→k25 by 1.4–2.1 %; every record improves (the objective is monotone
in k), but the added atoms are small tail coefficients — the largest
coefficient's share falls (e.g. layer 38: 0.318 → 0.238 mean top-1 share
from k=10 to k=25) while output-token *inclusion* barely moves at late
layers (0.842 → 0.895 at layer 38). At k=25 the dictionary begins to run
out of positively correlated atoms at the weak layers (18/76 records at
layer 14 and 5/76 at layer 21 stop early with `no_positive_correlation`;
all k=10/k=16 pursuits run to `max_atoms`).

**Conclusion.** k = 10 already captures essentially all of the
reconstruction and all of the qualitative structure; k = 16/25 add small
tail mass, dilute concentration, and pull in additional low-weight
multilingual/generic atoms (§6). Nothing in this run justifies the 2.5×
cost of k = 25 for this kind of analysis; k = 10 (optionally k = 16 as a
robustness check) is the recommended default.

## 3. Cross-k active-set stability

Matched per activation across k
([`cross_k_stability.csv`](../reports/jspace_20260716T170808536780_e4118850fb70/cross_k_stability.csv)):

- **Supports are effectively nested.** 97.7–99.7 % of k=10 atoms survive in
  the k=16 support; 94.1–99.2 % survive in k=25 (retention is highest at
  layers 35/38). Jaccard sits at its nesting ceiling (measured 0.61–0.62 vs
  the perfect-nesting bound 10/16 = 0.625; 0.38–0.40 vs 10/25 = 0.40).
- Coefficient structure is preserved: sparse cosine 0.80–0.96, shared-atom
  Spearman rank correlation 0.61–0.89 (highest at layer 35).
- The model's top-1 output token, once selected at k=10, persists at k=16
  and k=25 in **100 %** of cases; the largest-coefficient atom always
  persists and remains the largest in 60–96 % of records (highest late).

So larger k neither destabilizes nor re-derives the small-k solution — it
appends tail atoms to it. This is what makes k = 10 safe as the default.

## 4. The layer-21 collapse

Verified: mean k=10 EF at layer 21 is 2.3 × 10⁻⁵ — ~70× below layer 14 and
~700× below layer 35. The collapse is total and uniform: coefficient sums
average **0.1** at layer 21 versus 83–146 elsewhere, the first pursuit step
removes only 5 × 10⁻⁶ of the residual norm (vs 4 × 10⁻⁴ at layer 14 and
~5 × 10⁻³ at layers 35/38), and neither k, format, category, nor position
changes the picture (chat EF at 21 is even marginally *higher* than plain:
2.8 × 10⁻⁵ vs 2.1 × 10⁻⁵). Concentration (Herfindahl 0.127, top-1 share
0.205) is normal — the *shape* of the solution is ordinary; its *magnitude*
is nearly zero, meaning the layer-21 atoms are almost orthogonal to the
held-out activations (best-atom cosine ≈ 3 × 10⁻³).

CPU inspection of the frozen lens (no model load;
`analysis_summary.json → lens_matrix_statistics`) shows **J_21 is a massive
scale outlier**:

| layer | ‖J‖_F | σ₁ | participation-ratio rank | median row norm |
|---|---|---|---|---|
| 3 | 8.5 | 5.9 | 121 | 0.12 |
| 7 | 12.3 | 5.4 | 117 | 0.20 |
| 14 | 3.5 | 1.5 | 275 | 0.05 |
| **21** | **188.7** | **65.6** | 247 | **3.25** |
| 28 | 5.0 | 3.2 | 312 | 0.05 |
| 35 | 12.8 | 4.1 | 2053 | 0.23 |
| 38 | 18.1 | 4.5 | 2203 | 0.34 |

Yet the layer-21 *rank* evaluation is not broken: the fitted J-lens at 21
achieves median rank 6 693 overall (1 824 plain-only) with hit@10 = 0.08 —
far better than every control at that layer (§9) — and the layer-21 cones
are sometimes semantically apt in *direction* (the gold prompt selects
`' gold'`, `'Gold'`, `' Golds'`; the Madrid-language prompt selects
`' Gaelic'`) while their coefficients are ~0.05.

Competing explanations, ranked by this run's evidence:

1. **Poor / unstable corpus-averaged Jacobian at layer 21** (strongest
   support): the 15–50× Frobenius-norm outlier is a classic signature of a
   least-squares average over widely varying per-prompt Jacobians; its
   dominant directions retain enough signal to rank tokens but do not lie
   near held-out activations, so nonnegative projections onto them are
   nearly zero.
2. **Ordinary local-to-global architectural transition** (plausible,
   consistent): trajectories show *complete* active-set replacement across
   14→21 and 21→28 (Jaccard 0.04 and 0.007), and the mid-stack is where
   the pilot report already found weak readout fidelity. This would explain
   why the Jacobian is hard to average there, but is not independently
   established here.
3. **Atom-normalization interaction** (partial): selection divides
   correlations by atom norms (scale-invariant), and coefficients on raw
   atoms scale as 1/‖atom‖ — this fully explains the tiny *coefficient
   sums* given huge atoms, but not the tiny *explained fraction*, which is
   scale-invariant.
4. **Ruled out by the data**: WikiText/plain corpus bias (chat is not
   worse at 21), evaluation-position mismatch (−1 ≈ −2), k sensitivity
   (identical at 10/16/25), and residual-scaling artifacts (target norms at
   21 are ordinary: mean 92.7).

Atom-level diagnostics (norm distribution of `W_U @ J_21` rows) require
Gemma's unembedding matrix and **cannot be computed without loading the
model**; that limitation is explicit. Layer 21 should be treated as a real
measurement of *this lens artifact* — most plausibly a fitting pathology at
that layer, not established as a structural property of Gemma — and remains
formally unresolved pending a refit diagnostic (§11).

## 5. Plain vs chat output alignment

Output-token inclusion (model's top-1 in the active set), k=10:

| layer | plain −2 | plain −1 | chat −2 | chat −1 |
|---|---|---|---|---|
| 35 | 0.967 | 0.867 | 0.500 | 0.125 |
| 38 | 0.967 | 1.000 | 0.625 | 0.000 |

The same ordering holds at k=16/25 and in the 8 matched task families that
exist in both formats (e.g. at layer 38 pos −1: matched plain 1.000 vs
chat 0.000), so it is not a prompt-content artifact. Decomposed further:

- **Position −1 is mostly a tokenization artifact.** At chat position −1
  (the newline after `<start_of_turn>model`), the model's top-1 is the
  *no-leading-space* variant (`'Can'`, `'Late'`, `'Three'`, …), a different
  token id than the `' Canberra'`-style variant. The cones there actually
  contain the right content: e.g. `chat-factual-canberra` pos −1, layer 38
  has `' Canberra'` as its **largest** coefficient (27.5) while inclusion
  scores 0 because the top-1 id is `'Can'`. The rank evaluation confirms
  near-misses rather than absence: median J-lens rank for chat at 38 is
  0.5 (pos −2) and 2.5 (pos −1) versus 0 for plain.
- **Chat cones do not select template tokens late.** Control/format tokens
  (`<turn|>`, `<eos>`, `'model'`, newlines) take **25 % of chat coefficient
  mass at layer 21** and appear in 50 % of chat cones at 14 — but **0 %**
  at layers 35/38. The template dominates mid-stack chat cones, not late
  ones.
- **A genuine but modest EF gap remains**: chat EF at 35/38 is 0.010–0.015
  vs plain 0.016–0.019, and chat's 35→38 stability is somewhat lower
  (Jaccard 0.23 vs 0.33). Chat sequences are also longer (21–29 tokens vs
  6–19) and the lens was fitted on plain WikiText only; these two factors
  cannot be separated with this run's data.

**Conclusion.** The dramatic part of the gap is (a) a token-identity
artifact at position −1 and (b) mid-stack template dominance; what remains
at late layers is a real but moderate format-distribution shift, plausibly
(not provably) inherited from the plain-text fitting corpus. "Chat
formatting causes broken decompositions" is **not** supported.

## 6. Recurrence: exact signatures vs similarity

Exact ten-token signatures **never repeat** — 380 unique digests per k.
That criterion was too strict to detect anything. Similarity-based
recurrence (within layer/k/format/position strata;
[`similarity_groups.json`](../reports/jspace_20260716T170808536780_e4118850fb70/similarity_groups.json))
finds real repeated structure, robust across a threshold grid rather than
at one cutoff (k=10, weighted Jaccard):

| threshold | non-singleton groups | records covered | largest |
|---|---|---|---|
| 0.2 | 24 | 98 | 13 |
| 0.4 | 14 | 54 | 8 |
| 0.5 | 10 | 26 | 4 |
| 0.7 | 1 | 2 | 2 |

With inverse-frequency reweighting the counts shrink but persist through
threshold 0.6 (e.g. 13 groups covering 41 records at 0.4), so recurrence is
not purely an artifact of globally frequent atoms. The groups at 0.5 are
interpretable in two distinct ways:

- **Format-driven**: chat records at layers 14/21 group across *unrelated*
  tasks (canberra/antonym/plural at 14; four different chat tasks share a
  `' concisely'`/`<turn|>` cone at 21) — template structure, not content.
- **Frame-driven**: at layers 35/38 the plain antonym prompts
  (`tall-short`, `north-south`, `fast-slow`) share a cone led by `' to'`
  (the "X is to …" completion frame), and `syntactic-plural-child` /
  `syntactic-past-go` / `multihop-eiffel-capital` share an `' is'`-led
  cone. These are recurring semantic/syntactic structures despite zero
  exact-signature repeats.

## 7. Atom frequency and nuisance directions

([`atom_frequencies.csv`](../reports/jspace_20260716T170808536780_e4118850fb70/atom_frequencies.csv),
[`atom_enrichment.csv`](../reports/jspace_20260716T170808536780_e4118850fb70/atom_enrichment.csv)).
1 974 distinct atoms cover the 3 793 k=10 selections. The most frequent are
generic, multilingual, punctuation, and template directions selected across
unrelated tasks — nuisance candidates, not concepts: `' coinciding'` (35
selections), `' zarówno'` (33, Polish), `' is'` (31), `' eternally'` (23),
`<turn|>` (23, chat-specific), `'…?'` (21), `'¹.'` (20), plus CJK/Indic
fragments. Enrichment localizes them: `<turn|>` is chat/layer-21-specific,
`' coinciding'` concentrates at layers 35/38, `'Trivia'`/`'UNKNOWN'` at
layer 14 chat. Output-token-conditioned counts separate the copula-like
atoms cleanly: `' is'` is selected 26× *as* the record's own predicted
token and only 5× otherwise — frequent because the probe set predicts
`' is'` often, not because it is a free-floating direction.

Frequent atoms do inflate raw similarity (the frequency-adjusted grouping
in §6 shrinks accordingly), which is why every similarity table in
[`reports/`](../reports/jspace_20260716T170808536780_e4118850fb70/) carries
raw and adjusted variants side by side.

## 8. Candidate ignition — NOT validated ignition

Per-transition signals, kept separate
([`transition_metrics.csv`](../reports/jspace_20260716T170808536780_e4118850fb70/transition_metrics.csv);
no composite score is computed anywhere):

| transition (k=10) | mean Jaccard | mean coefficient cosine | mean ΔEF |
|---|---|---|---|
| 14→21 | 0.041 | 0.107 | −0.0017 |
| 21→28 | 0.007 | 0.018 | +0.0073 |
| 28→35 | 0.017 | 0.039 | +0.0095 |
| **35→38** | **0.306** | **0.747** | −0.0017 |

The 35→38 stabilization is robust
([`candidate_ignition_summary.csv`](../reports/jspace_20260716T170808536780_e4118850fb70/candidate_ignition_summary.csv)):
it holds at every k (Jaccard 0.29–0.31; cosine rises with k, 0.75→0.81),
in both formats (plain 0.33 / chat 0.23 — both an order of magnitude above
any earlier transition), at both positions (0.32 / 0.30), and in all six
task categories (0.27–0.35). It **survives frequency adjustment**
(weighted Jaccard 0.294 raw → 0.268 adjusted) and **is not driven solely by
output-token inclusion** (0.214 with each record's model top-1 atom removed
— identical in plain and chat under that removal). Note ΔEF across 35→38 is
slightly *negative*: the coordinates stabilize without reconstruction
improving.

Honest interpretation: **late-layer sparse-coordinate consolidation.** By
layers 35–38 every reasonable readout (including the plain logit lens, §9)
converges to the output distribution, so stable sparse coordinates across
that transition are consistent with ordinary late-layer logit alignment
plus the lens simply being good at both layers. The mid-stack instability
(21→28 Jaccard 0.007) could equally reflect layer-dependent lens quality
(§4) as representation change. None of the paper's ignition evidence
(interventions, ambiguity sweeps, cross-model replication) exists here;
"candidate ignition diagnostics — NOT validated ignition" remains the only
defensible label.

## 9. Evaluation controls

Recomputed from the per-example records
([`evaluation_control_summary.csv`](../reports/jspace_20260716T170808536780_e4118850fb70/evaluation_control_summary.csv);
median rank / hit@10 over both formats and positions, n = 76 per cell):

| layer | jlens | logit lens | adjacent | shuffled | distant | permuted | random |
|---|---|---|---|---|---|---|---|
| 14 | 14 326 / .07 | 94 214 / 0 | 16 504 / .07 | 143 613 / 0 | 143 613 / 0 | 120 866 / 0 | 72 604 / 0 |
| 21 | 6 693 / .08 | 127 119 / 0 | 136 414 / .01 | 146 836 / 0 | 155 690 / 0 | 163 635 / 0 | 201 680 / 0 |
| 28 | 26 / .39 | 2 061 / .22 | 104 / .34 | 148 474 / 0 | 148 474 / 0 | 161 330 / 0 | 79 804 / 0 |
| 35 | 0 / .86 | 0 / .80 | 0 / .87 | 50 187 / 0 | 80 156 / 0 | 84 523 / 0 | 50 882 / 0 |
| 38 | 0 / .92 | 0 / .86 | 0 / .91 | **0 / .91** | 66 285 / 0 | 105 557 / 0 | 90 014 / 0 |

- The destructive controls (permuted, random) fail everywhere, as they
  should. The fitted lens beats the logit lens at every layer, decisively
  in the mid-stack.
- **Mapping collisions are exactly as defined, not accidental**
  ([`evaluation_control_collisions.json`](../reports/jspace_20260716T170808536780_e4118850fb70/evaluation_control_collisions.json)):
  distant and shuffled both borrow J₃₈ at layer 14 and J₃ at layer 28, and
  their per-example ranks are identical there in 38/38 cases — and never
  spuriously identical where the mappings differ. The striking
  `shuffled_layer` score at layer 38 is the same structure: the shuffle
  assigned J₃₅ to layer 38, which is *also* the adjacent control there, so
  "shuffled ≈ fitted at 38" is a statement about **correlated neighboring
  Jacobians**, not a failed control.
- Adjacent-layer Jacobians track the fitted one closely at layers 28–38
  (and even at 14), but *not* across 21 (J₂₈ applied at 21: median 136 414)
  — one more sign that layer 21's fitted matrix is sui generis (§4).
- The ambiguous legacy "wrong layer" label is not used anywhere in the new
  reports; all layer-mapped controls are named by their mapping.

## 10. Memory hardening and provenance (workflow changes on this branch)

The pilot-hardware OOM happened on an L4 in
`torch.isfinite(atoms).all()` over the [262 144 × 2 560] float32 dictionary
— a single ~671 MB boolean temporary on top of the 2.7 GB atoms and a
2.7 GB transient float32 copy of `W_U`. Changes
(all in [`jlens/pursuit.py`](../jlens/pursuit.py), results bit-identical,
tested in [`tests/test_memory_hardening.py`](../tests/test_memory_hardening.py)):

1. `validate_finite(...)`: chunked finiteness check (default 16 384 rows ≈
   40 MB transient instead of 671 MB), each chunk's mask released before
   the next; used by `JSpaceDictionary.__init__`.
2. `_chunked_row_norms(...)`: atom norms without a full float32 copy when
   atoms are stored in half precision (no-op for float32 storage).
3. `JSpaceDictionary.from_lens(..., build_chunk_rows=N)`: optional chunked
   construction that upcasts only N rows of `W_U` at a time and writes each
   chunk's product straight into the preallocated output — avoiding both
   the full float32 `W_U` copy and, for non-float32 storage, the transient
   full-precision product. Off by default (the one-shot matmul is
   unchanged); equality and determinism are tested.
4. Peak-memory accounting documented on `from_lens` (atoms 2.7 GB are the
   output and always exist; correlation memory is already bounded by
   `correlation_chunk_size=65536`, ~80 MB per pass at batch 76; callers
   should `del` each layer's dictionary before building the next —
   nothing in the module retains it).
5. Dictionary storage stays **float32** (unchanged, per the recorded run
   convention).

With the ~3.4 GB of avoidable transients gone, the remaining steady state
(bf16 model + one float32 dictionary + bounded temporaries) fits an L4/T4
(24/16 GB) comfortably; **a lower-memory rerun without an A100 is
realistic**, with fp16 atom storage (`dtype=torch.float16`, compute already
float32) available as a further 1.35 GB saving if ever needed.

Provenance: `jlens.metadata.execution_record(...)` now records
configured-vs-resolved `allow_model_load` and whether a load happened,
rejecting the inconsistent combination; the completed run's contradiction
is documented (§1) and left as-is.

## 11. Conclusions and open questions

Supported by the artifacts:

- k = 10 is sufficient; k = 16/25 buy 4–10 % relative EF via small tail
  coefficients, at lower concentration and higher cost. k=10 supports are
  ~95–99 % preserved inside larger-k supports.
- Layer 35 is the strongest sparse layer by EF (0.0168 at k=10); layer 38
  is close (0.0151) with the strongest output alignment (inclusion 0.84 at
  k=10). Layer 21 is a real, uniform collapse of *this lens*, best
  explained by an outlier fitted Jacobian (‖J₂₁‖_F = 188.7 vs 3.5–18.1
  elsewhere); whether Gemma's layer 21 truly has a weak J-space component
  is **unresolved**.
- The plain/chat gap is real but heterogeneous: a position-−1 tokenization
  artifact plus mid-stack template dominance plus a modest residual EF gap
  attributable to format shift and/or the plain-only fitting corpus.
- Exact-signature recurrence was too strict (zero repeats anywhere);
  similarity-based recurrence finds threshold-robust repeated structure,
  including template cones (chat, mid-stack) and completion-frame cones
  (antonym "X is to", copula `' is'`, late layers).
- Globally frequent atoms are generic/multilingual/punctuation/template
  directions; late-layer output alignment and the 35→38 stabilization both
  survive downweighting or removing them.
- The 35→38 stabilization is robust across k, format, position, and
  category, but "late-layer sparse-coordinate consolidation" is the
  strongest defensible claim; it is consistent with ordinary late-layer
  logit alignment and is **not** validated ignition.
- A lower-memory rerun is feasible after the chunking changes.

Open questions (require new computation, not attempted here): whether
J₂₁'s blow-up is a fitting instability (per-prompt Jacobian variance /
regularized or larger-corpus refit at layer 21 would answer it); whether a
chat-formatted fitting corpus closes the residual chat EF gap; and whether
finer layer sampling between 28 and 38 localizes where sparse coordinates
begin to stabilize.

**Recommended next experiment** (not executed): refit the lens at layer 21
only, on the same 100-prompt corpus, logging per-prompt Jacobian norms and
a condition/variance diagnostic — plus a small chat-formatted fitting set —
then rerun this decomposition on layers {21, 31, 35, 38} at k = 10. This
directly discriminates explanation 1 from explanation 2 in §4 and tests
the corpus hypothesis in §5, and now fits on an L4.

## Appendix: qualitative examples (k = 10, not cherry-picked for success)

- **Strong factual** — `factual-canberra` pos −1, layer 38 (top-1
  `' Canberra'`, EF 0.0147): `' Canberra'`:27.1, `' Sydney'`:11.3,
  `' capital'`:10.0, `' coinciding'`:7.7, `' not'`:7.3 — answer, plausible
  alternative, and topic, with a nuisance atom in the tail.
- **Antonym/directional** — `antonym-tall-short` pos −1, layer 35 (top-1
  `' short'`, EF 0.0108): `' short'`:24.5, `'…?'`:18.4, `' ছোট'`:15.9
  (Bengali "small"), `' tall'`:14.2, `' diminutive'`:13.6, `'矮'`:12.5 —
  both poles of the axis plus cross-lingual synonyms.
- **Counting** — `counting-triangle-sides` pos −1, layer 38 (top-1 is the
  whitespace token, EF 0.0160): `' '`:32.4, `' three'`:16.0,
  `' always'`:13.0 — the formatting token the model actually predicts
  dominates, with the numeric answer second.
- **Chat** — `chat-factual-canberra` pos −1, layer 38 (top-1 `'Can'`, EF
  0.0113): `' Canberra'`:27.5, `' Sydney'`:6.5, `'capital'`:6.5 — correct
  content, scored as a miss by strict token identity (§5).
- **Weak/incoherent** — `multihop-madrid-language` pos −1, layer 21 (top-1
  `' Spanish'`, EF 0.00001): `'ಾಗಿದೆ'`, `' setIs'`, `' ہوا۔'`,
  `' Gaelic'`, `' Staats'`, all with coefficients ≈ 0.04 — the layer-21
  collapse in one record (a faint language-related direction, no usable
  magnitude). Mid-stack incoherence also appears at stronger layers:
  `factual-canberra` at layer 28 is led by `' Baltimore'`:49.2.
