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

The causal stage first smart-saves unrestricted clean answers for 32 candidate
photographs per concept. It intervenes only after eight photographs per
concept answer both identity and leg-count questions correctly in text, image,
and spoken audio. It then compares the text and pooled lenses using exact
alpha-one exchange against matched random and unrelated-coordinate controls.
No answer token is appended and no candidate list is shown to the model.

## Interpretation

The 4-by-3 cross-evaluation matrix is the primary result. It distinguishes:

- a genuinely modality-specific fitting effect;
- a pooled lens that transfers across input channels;
- no material fitting-distribution effect.

A better pooled readout would diagnose a limitation of the text-only fitting
distribution. It would not by itself prove causal downstream recomputation.
That stronger claim depends on the separately reported unrestricted causal
endpoint.
