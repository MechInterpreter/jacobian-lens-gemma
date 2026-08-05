# Frozen protocol — research-grade multilayer text-only J-lens calibration v1

Protocol tag: `research-grade-multilayer-text-jlens-calibration-v1`
Machine-readable twin: [`configs/research_grade_jlens_calibration_v1.json`](../configs/research_grade_jlens_calibration_v1.json)
Methodology and provenance: [`docs/research_grade_jlens_methodology.md`](research_grade_jlens_methodology.md)

Everything in this document is fixed **before** any number exists. The config
file is checksummed into the run fingerprint; editing any field invalidates
stored results rather than rescoring them.

---

## 1. What this run is and is not

**Is:** a text-only estimate of `J_l` at eight physical decoder layers of a
frozen Gemma 4 E4B, at three nested corpus scales, evaluated against a
predeclared tie-aware gate, with an untouched confirmation set.

**Is not:** a multimodal experiment, a J-space decomposition, an intervention
study, or evidence about layer 38 that supersedes the completed runs. It reads
no SpokenCOCO caption, image or audio; it reads no multimodal activation; it
consults no downstream multimodal result when choosing anything.

---

## 2. Model, sites, and identity

| Field | Value |
|---|---|
| Model | `google/gemma-4-E4B-it` |
| Revision (immutable) | `fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd` |
| Tokenizer | same repo, same revision |
| Architecture asserted | 42 layers · `d_model` 2560 · vocab 262144 · dense · tied unembedding |
| Physical layer grid | **8, 14, 20, 26, 32, 35, 38, 40** |
| Normalized depth (`round(100·l/42)`) | 19, 33, 48, 62, 76, 83, 90, 95 |
| Target layer | 41 (pre-final-norm residual) |
| Source & target site | `block_output` — after attention, MLP, per-layer-embedding re-injection and `layer_scalar` |
| Hook | `jlens.hooks.ActivationRecorder` on `model.language_model.layers[l]` |
| Readout | `lm_head(final_norm(J_l @ h))`, scored **pre-softcap** |
| Softcap | `30·tanh(x/30)` — strictly monotonic, so rankings are identical either way |
| Position rule | `skip_first = 16`; valid source positions `16 ≤ t < seq_len − 1` |
| Sequence length | 128 tokens |
| `dim_batch` | 8 |

Grid rationale: 20, 26, 32 and 38 are carried over unchanged so that results are
directly comparable to the v2 recalibration and the completed localization run.
8 and 14 extend the study genuinely earlier. 35 and 40 bracket the established
layer 38 so that a pass there cannot be read as a lucky isolated point.

---

## 3. Corpus

**Primary: WikiText-103-raw-v1** (`Salesforce/wikitext`, config
`wikitext-103-raw-v1`, split `train`), streamed, records of at least 600
characters.

Chosen because it is named and versioned (unlike the paper's unnamed
"pretraining-like corpus"), permissively licensed (CC BY-SA 3.0), deterministically
ordered under HuggingFace streaming so record IDs are stable, prose-like rather
than code or dialogue, streamable without a bulk download, and **already the
corpus of this project's completed 100-prompt pilot** — so the 1k result is
comparable to it on every axis except scale.

Record identity is `wikitext-103-raw-v1/train/{stream_index}` paired with a
sha256 of the normalized text. The dataset revision recorded in the config is
marked **`REQUIRES_VERIFICATION_IN_COLAB`** — it must be resolved and written
down on first real contact with the Hub, not asserted from here.

A C4 fallback is configured and **disabled**. Selecting it changes `corpus_id`,
changes the fingerprint, and invalidates every stored unit. There is no silent
switch.

SpokenCOCO is not a calibration corpus and no code path can reach it.

---

## 4. Splits

Assignment is by stable hash, not by position, so it is reproducible from the
record ID alone and cannot drift with streaming order changes:

```
bucket = int(sha256(f"{20260805}|{record_id}").hexdigest()[:8], 16) % 100
```

| Partition | Buckets | Size drawn | Role |
|---|---|---|---|
| Fit | 0–79 | up to 10,000 (50,000 if extended) | estimates `J_l` |
| Development / validation | 80–89 | **128** | every scale-point decision |
| Final confirmation | 90–99 | **128** | touched once, at the end |

**Nested scale ordering.** Within the fit partition, prompts are ordered by
`sha256(f"{split_seed}|nested|{record_id}")`. The first 1,000 are the 1k point,
the first 5,000 the 5k point. Nesting is therefore exact and scale is the only
variable that changes between points.

**The confirmation set influences nothing.** It does not participate in corpus
selection, scale selection, threshold setting, layer selection, or any stopping
decision. In code this is enforced by an object that refuses to hand it over
until the scale choice has been recorded and frozen.

**Duplicate control.** Exact duplicates are caught by a sha256 over normalized
text (NFKC → lowercase → collapse whitespace → strip). Near-duplicates are
caught by a 64-bit SimHash over word 3-grams, banded into four 16-bit bands, with
a Hamming threshold of 3. The leakage audit runs across every pair of partitions
and **refuses** on any hit; it does not drop the offending record silently.

---

## 5. Target-token diversity

The 32-prompt v2 validation accidentally contained only seven distinct model
output targets. For this workflow, over 128 validation prompts:

- **at least 24 distinct target token IDs**, and
- **no single target token may account for more than 25% of prompts**.

Targets are the frozen model's own final-layer argmax at the last prompt
position. No lens and no candidate layer is consulted, so selection cannot be
contaminated by the thing under test. Selection is deterministic and stratified:
one prompt is reserved per required distinct target before the remainder is
filled in stable hash order.

If the pool cannot meet the floor, the run **raises
`InsufficientTargetDiversityError` and stops**. The threshold is never lowered
to make a run proceed. Selected prompt IDs and target IDs are recorded in the
manifest with a checksum.

---

## 6. Validation gate

Tag `research-grade-calibration-tie-aware-native-readout-v1`. Identical in
structure and meaning to the localization Stage-B gate — same scoring code, same
rank conventions — re-parameterized for 128 prompts and given one additional
blocking clause.

Every threshold is stated against the **midrank** convention
(`strictly_above + (tied_with_target + 1)/2`), the only rank convention that is
invariant to how ties are broken. Optimistic and pessimistic ranks are computed
and reported but decide nothing.

A layer passes only if **all** of:

1. **Coverage and non-degeneracy** — every variant scored on all 128 prompts,
   every metric finite, ≥ 24 distinct targets, **no target above 25% share**,
   and J-lens tied-at-maximum rate ≤ 0.50.
2. **Beats the noise controls** — J-lens MRR ≥ 1.5× *and* ≥ +0.10 over both the
   row-permuted and the norm-matched-random control.
3. **Beats the wrong-layer control** — J-lens MRR ≥ wrong-layer MRR + 0.15,
   using `distant_layer_mapping` (not the deprecated cyclic control).
4. **Rank and top-k** — median midrank ≤ 5.0 and top-10 inclusion ≥ 0.50, with
   inclusion measured at the **pessimistic** rank so a tie block cannot
   manufacture membership.
5. **Fold stability** — over 4 fixed folds (`prompt i → fold i mod 4`), every
   fold's J-lens MRR beats every control on that same fold, and no fold falls
   below 0.50× the overall MRR.

Reported, never blocking: unique and argmax top-1 agreement, tie-block sizes,
mean top-10 overlap, margin over the strongest non-target token, and the ordinary
logit lens as a diagnostic. The v2/legacy conjunction is computed and printed
beside the new gate for every layer.

Clause 1's target-share sub-clause is the only substantive addition over the
localization gate, and it exists because a 128-prompt sample dominated by `the`
would satisfy a distinct-count floor while measuring almost nothing.

---

## 7. Scale study and the plateau rule

Scale points **1,000 / 5,000 / 10,000**, exactly nested, evaluated at each point
on the same 128 development prompts with the same controls.

Reported per layer per scale: fitting-loss surrogate (`mean_rel_change` of the
running mean — there is no loss; see §1 of the methodology), held-out MRR,
median midrank, optimistic and pessimistic medians, top-10 inclusion, unique
top-1, tied-at-max rate, per-fold MRR, target diversity, margins over each
control, logit-lens diagnostic, finiteness, and convergence status.

### Plateau rule — `conservative-earlier-layer-improvement-v1`

Extension past 10,000 is justified **only if all four hold**:

1. Some layer in **{8, 14, 20, 26, 32}** that is INELIGIBLE at 5k improves
   between 5k and 10k by **ΔMRR ≥ 0.05 and a ≥ 20% relative drop in median
   midrank** — both, on the development set.
2. That layer's tied-at-max rate does **not** increase from 5k to 10k.
3. That layer's 5k→10k improvement is **at least 50%** of its 1k→5k improvement
   (i.e. it is still climbing, not decaying into a plateau).
4. **No** layer eligible at 5k becomes ineligible at 10k.

Otherwise the verdict is `PLATEAU_REACHED` and the study stops at 10k. The rule
is not revisable after seeing results; its text is checksummed into the
fingerprint. Even when it fires, the extension does **not** run automatically —
`RUN_OPTIONAL_LARGE_SCALE` and `CONFIRM_OPTIONAL_LARGE_SCALE_BUDGET` are separate
manual switches.

### Scale selection

Choose the **smallest** scale point whose development eligible-layer set equals
the largest computed scale's eligible set; if no two agree, choose the largest
computed scale. Parsimony is the tie-break, and the rule is fixed here so that
"which scale we report" cannot become a free parameter chosen after the fact.

---

## 8. Confirmation and publication

Only after the scale is chosen and recorded may the confirmation set be opened.
A layer is published **only** if it passes the same gate, unchanged, on the
128 untouched confirmation prompts.

One artifact per passing layer, each carrying: model and tokenizer revisions,
physical layer and normalized depth, `d_model`, hook site and residual
convention, vector orientation, normalization convention, calibration modality
(`text-only`), corpus ID and revision, all three split checksums, number of
fitting prompts, scale point, validation and confirmation protocol tags, gate
digest, torch/transformers/python versions, upstream and local commits, fitting
diagnostics, validation and confirmation metrics, artifact checksum, and
`frozen: true`.

Failed layers keep every diagnostic number and are written with
`validated: false`. They are never marked validated, and the publication guard
raises rather than warning.

`runs/text_jlens_early_layer_recalibration_v2/artifacts/lens.validated.pt` is
never overwritten; the guard refuses that path.

---

## 9. Compute and storage budget

### 9.1 The measurement everything rests on

This project's completed pilot (`docs/pilot_report.md`) fitted **100 WikiText
prompts × 128 tokens × 7 source layers (shallowest = 3) in 9,665.4 s on one
NVIDIA L4** — **96.7 s per prompt** — at `dim_batch = 8`, peak probe memory
16.84 GB of 24 GB.

Cost structure (see methodology §5): one forward plus
`ceil(2560/8) = 320` backward passes per prompt. Each backward runs from layer
41 down to the **shallowest** source layer and extracts all source layers at
once. So:

- adding layers to the grid is nearly free;
- **lowering** the shallowest layer is the expensive change;
- the 320 backward passes per prompt are irreducible for an exact dense Jacobian.

Our shallowest layer is 8, not 3, so the backward span is 33 blocks rather than
38. Scaling the measurement by span and allowing for eight layers of slice-copy
instead of seven:

**83.9 s per prompt, uncertainty 79.7–104.9 s** (`jlens.calibration.plan`, which
is the source of truth for every number below). The range is wide on purpose:
the span-proportionality assumption is an inference from the cost model, not a
second measurement, and the pilot never recorded full-fit peak memory.

### 9.2 Fitting time — the uncomfortable part

| Scale | Central | Range | 12-hour Colab sessions |
|---|---|---|---|
| 1,000 | **23.3 h** | 22.2–29.1 h | **2–3** |
| 5,000 | **116.6 h** | 110.8–145.7 h | **10–13** |
| 10,000 | **233.2 h** | 221.5–291.5 h | **20–25** |
| *(optional 25,000)* | *582.9 h* | *553.8–728.6 h* | *49–61* |
| *(optional 50,000)* | *1,165.8 h* | *1,107.5–1,457.2 h* | *98–122* |

Because the scale points are nested, these are **cumulative, not additive**:
reaching 10,000 costs 233 h in total and yields all three lenses. Reaching only
1,000 costs 23.3 h.

**This must be said plainly: 10,000 prompts is roughly ten days of continuous L4
time, and the optional extensions are one to two months.** Neither fits inside a
two-week research phase alongside the multimodal work. The 1,000-prompt point
does fit, at two or three sessions.

The mitigation that exists and is real: `JacobianLens.merge()` combines lenses
fitted on disjoint prompt slices as an `n_prompts`-weighted mean, so the fit
parallelizes perfectly across simultaneous runtimes. *N* concurrent L4s divide
the wall time by *N* exactly. Whether that is available is a resourcing
question, not a technical one.

### 9.3 Everything that is not fitting

| Work | Cost |
|---|---|
| Corpus streaming, 10k records @ min 600 chars | 25–60 MB transferred, 2–6 min |
| Corpus streaming, 50k records | 150–300 MB, 10–25 min |
| Split assignment, dedup, leakage audit | CPU-only, < 1 min |
| Target-token discovery (128 + 128 prompts, 1 forward each) | ~1–2 min |
| Validation at one scale point (128 prompts × 8 layers × 5 variants) | 5–15 min |
| Confirmation (same, once) | 5–15 min |
| Architecture audit + cost probe | ~2 min |
| **Total non-fitting model work, whole study** | **≲ 1 h** |

Validation is a forward-pass workload; the 320 backward passes only ever happen
during fitting. This is why the scale study's cost is entirely the fit.

### 9.4 Memory and storage

| Resource | Value |
|---|---|
| GPU VRAM | 16.84 GB measured at `seq_len=48`; the pilot completed at `seq_len=128` on a 24 GB L4, but **full-fit peak was never recorded** — this run instruments it |
| CPU RAM | `jacobian_sum` 200 MiB + per-prompt Jacobians 200 MiB + torch ≈ 1.5–2 GB |
| Checkpoint (fp32 sum, 8 layers) | **200 MiB** (`8 × 2560² × 4 B`) |
| Scale snapshots (fp16, 8 layers × 3) | **300 MiB** |
| Published artifacts (fp16, ≤ 8 single-layer) | **≤ 100 MiB** |
| Reports, diagnostics, per-unit JSON | ~20 MiB |
| **Total Drive** | **0.61 GiB** |
| Model download | **~16 GB per session** unless the HF cache lives on Drive — at 20+ sessions this is the second-largest hidden cost in the study |
| Temporary local disk | model cache 16 GB if not on Drive |

### 9.5 Interruption behaviour

- The fit checkpoints every 25 prompts, so a disconnect loses **at most ~35
  minutes** of work.
- Checkpoint writes are atomic (temp file + `os.replace`), so a crash mid-write
  never produces a torn checkpoint.
- Scale snapshots, once written, are final and never rewritten.
- Validation, confirmation and publication write one checksummed JSON per unit
  and resume individually.
- A resume under a changed fingerprint is **refused**, listing every differing
  field.

### 9.6 What is reused and what is not

- **Reused across scale points: 100% of the fitting work.** Nested snapshots come
  from one accumulator.
- **Not reused: validation, controls and confirmation**, which re-run per scale
  point — and cost minutes, so this is not worth optimizing.
- **All eight layers come from a single forward/backward set per prompt.** There
  is no per-layer pass.
- **Scaling by layer count** affects only slice copies and checkpoint size, not
  the number of backward passes.

### 9.7 Budget confirmation switches

No model work happens until the matching switch is set by hand:
`CONFIRM_1K_BUDGET`, `CONFIRM_5K_BUDGET`, `CONFIRM_10K_BUDGET`,
`CONFIRM_OPTIONAL_LARGE_SCALE_BUDGET` — all False in the committed notebook,
alongside `RUN_REAL_CALIBRATION`, `RUN_MODEL_STAGES`, `RUN_OPTIONAL_LARGE_SCALE`,
`RUN_FINAL_CONFIRMATION` and `PUBLISH_VALIDATED_LENSES`.

---

## 10. Recommendation for the first real run

Run **1,000 only**. It is 2–3 sessions, it is the paper's own production scale,
and it carries nearly all of the scientific information (methodology §6). Decide
5k and 10k after seeing the 1k table and the plateau evidence between the
pilot's 100 and the new 1,000 — with real numbers rather than an estimate
derived from one measurement at a different depth.
