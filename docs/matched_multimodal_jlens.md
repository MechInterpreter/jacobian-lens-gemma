# Matched-distribution multimodal J-Lens study

## Question

The earlier studies used an average Jacobian estimated on text-only inputs.
This study directly tests whether that fitting distribution matters. It fits
four otherwise identical lenses on synchronized SpokenCOCO examples:

1. written captions;
2. photographs;
3. native spoken captions;
4. an equal-size pooled arm assigning the same photographs evenly across the
   three modalities.

All four arms contain 99 processor examples. The pooled arm contains 33 text,
33 image, and 33 audio examples, so a pooled-versus-text difference cannot be
explained merely by giving the pooled lens three times as much fitting data.

Image and audio examples are processed by Gemma 4's pinned native processor.
The estimator is still the ordinary sample-mean Jacobian
`E[d h_final / d h_layer]`; no adapter, probe, classifier, or alignment map is
trained.

## Run order

Use
[`multimodal_jspace_matched_jlens_colab.ipynb`](../notebooks/multimodal_jspace_matched_jlens_colab.ipynb).

For the first A100 session, set:

```python
RUN_REAL_MATCHED_JLENS = True
RUN_STAGE0_FREEZE_PLAN = True
RUN_STAGE1_FIT_LENSES = True
RUN_STAGE2_CROSS_EVALUATE = True
RUN_STAGE3_CAUSAL_COMPARE = False
RUN_STAGE4_WRITE_REPORT = False

CONFIRM_MODEL_LOAD = True
CONFIRM_FIT_BUDGET = True
CONFIRM_CAUSAL_BUDGET = False
REPORT_RUN_DIR = None
```

Run end-to-end. The four fitting accumulators are atomically saved every ten
processor examples. Cross-evaluation saves one unit per photograph. A Colab
disconnect therefore resumes rather than restarting the experiment.

Inspect `multimodal_lens_cross_eval_report.json` before spending the causal
budget. Stage 3 is a separate prospective comparison. To run it, reopen the
same notebook with the same scientific constants and set:

```python
RUN_REAL_MATCHED_JLENS = True
RUN_STAGE0_FREEZE_PLAN = True
RUN_STAGE1_FIT_LENSES = False
RUN_STAGE2_CROSS_EVALUATE = False
RUN_STAGE3_CAUSAL_COMPARE = True
RUN_STAGE4_WRITE_REPORT = True

CONFIRM_MODEL_LOAD = True
CONFIRM_FIT_BUDGET = False
CONFIRM_CAUSAL_BUDGET = True
REPORT_RUN_DIR = None
```

The original causal stage smart-saved unrestricted clean answers for 32
candidate photographs per concept. It was designed to intervene only after
eight photographs per concept answered both identity and leg-count questions
correctly in text, image, and spoken audio. No answer token was appended and no
candidate list was shown to the model.

## Prospective causal follow-up after the completed capability no-go

The first causal screen used exact token-ID equality.  It therefore refused
case and tokenizer aliases even when they decoded to the same answer surface.
The completed run remains immutable and retains its `CAPABILITY_NO_GO`
verdict.  The follow-up declares, before selecting new media, a narrower
surface-equivalence rule: NFKC normalization, case folding, and collapsed
whitespace only.  It does not strip punctuation, singularize, or map bird
species onto the parent category.

The follow-up imports the four completed lens files read-only from
`mmjlens4_real_1d3b1afbd019`, verifies every report and tensor checksum, and
excludes all 64 photographs opened by the completed clean screen.  It selects
96 fresh candidates per concept and writes to a new fingerprinted
`mmjlens5causal_*` run.  No fitting or cross-evaluation is repeated.

Use the same notebook with:

```python
RUN_REAL_MATCHED_JLENS = True
RUN_STAGE0_FREEZE_PLAN = True
RUN_STAGE1_FIT_LENSES = False
RUN_STAGE2_CROSS_EVALUATE = False
RUN_STAGE3_CAUSAL_COMPARE = True
RUN_STAGE4_WRITE_REPORT = True

CONFIRM_MODEL_LOAD = True
CONFIRM_FIT_BUDGET = False
CONFIRM_CAUSAL_BUDGET = True
REPORT_RUN_DIR = None
```

The clean screen and every intervention are stored as fingerprint-bound units.
A disconnect resumes completed work; it never reuses the superseded exact-ID
screen and never refits a lens.

## Paired alpha dose-response

The completed alpha=1 run measured zero target-answer flips for both the text
and pooled lenses.  A separate exploratory stage tests whether that endpoint
conceals a graded causal effect or merely needs a stronger intervention. After
the coarse 0.5/1/2/4 sweep, the notebook now samples
`0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1`. This grid brackets the
strongest stable signal near alpha=0.5 and excludes the already destructive
alpha>=2 regime. It was chosen after inspecting the coarse result, so it is
explicitly exploratory; any favorable alpha must be confirmed on a fresh
population. It
reuses the same 16 clean-capable photographs without reselecting them, verifies
the completed run and all four lens checksums, and smart-saves one
`arm x alpha x condition` trial at a time.

This uses Anthropic's coordinate-swap equation exactly:

```text
V = [v_source, v_target]
c = pinv(V) h
h_patched = h + alpha * V (swap(c) - c)
```

The raw J-lens vectors are used without normalization; coordinates are
recomputed at every physical layer L33-L40; every original prompt position is
patched; and the orthogonal residual is unchanged. Alpha=1 is the exact
exchange. Anthropic's alpha=2 double-strength condition was measured in the
coarse sweep but is not repeated in this stable-range refinement. The
multimodal Gemma/SpokenCOCO task is an extension of the paper's protocol, not an
exact replication of Anthropic's model and data.

Run the same notebook on an L4 with:

```python
RUN_REAL_MATCHED_JLENS = True
RUN_STAGE0_FREEZE_PLAN = True
RUN_STAGE1_FIT_LENSES = False
RUN_STAGE2_CROSS_EVALUATE = False
RUN_STAGE3_CAUSAL_COMPARE = False
RUN_STAGE3B_ALPHA_SWEEP = True
RUN_STAGE4_WRITE_REPORT = True

CONFIRM_MODEL_LOAD = True
CONFIRM_FIT_BUDGET = False
CONFIRM_CAUSAL_BUDGET = False
CONFIRM_ALPHA_SWEEP_BUDGET = True
REPORT_RUN_DIR = None
```

The output is `multimodal_lens_alpha_sweep_report.json` in a new fingerprinted
`mmjlens6alpha_*` directory.  It reports unrestricted top-1 success plus target
logit, target rank, target probability, source-logit suppression, KL, activation
norm, and alpha-matched random/unrelated controls. Alpha=1 remains the primary
result regardless of which sensitivity value is largest, and the report records
that the refined grid was selected after outcomes were observed. A frozen
identity-only specificity rule ranks one common alpha across both lens arms;
the downstream leg-answer endpoint is excluded from alpha selection. The
reported winner remains exploratory until repeated on fresh media.

## Interpretation

The 4-by-3 cross-evaluation matrix is the primary result. It distinguishes:

- a genuinely modality-specific fitting effect;
- a pooled lens that transfers across input channels;
- no material fitting-distribution effect.

A better pooled readout would diagnose a limitation of the text-only fitting
distribution. It would not by itself prove causal downstream recomputation.
That stronger claim depends on the separately reported unrestricted causal
endpoint.
