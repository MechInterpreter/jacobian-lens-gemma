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

## Interpretation

The 4-by-3 cross-evaluation matrix is the primary result. It distinguishes:

- a genuinely modality-specific fitting effect;
- a pooled lens that transfers across input channels;
- no material fitting-distribution effect.

A better pooled readout would diagnose a limitation of the text-only fitting
distribution. It would not by itself prove causal downstream recomputation.
That stronger claim depends on the separately reported unrestricted causal
endpoint.
