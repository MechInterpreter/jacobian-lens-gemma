# Gemma 4 E4B Jacobian Lens — Smoke Validation Report

**Run:** [`runs/smoke_20260715T172315460316_fb2eefcd91cd/`](../runs/smoke_20260715T172315460316_fb2eefcd91cd/)
**Config:** [`configs/gemma_text_smoke.yaml`](../configs/gemma_text_smoke.yaml)
**Config fingerprint:** `sha256:fb2eefcd91cd30231618e5eddac00221001edac3a4ae34991f5a0bb77c49781c`
**Local repository commit at run time:** `d62814fdf93b8eb36463c86a70ca711891d5c939`
**Upstream Jacobian Lens commit:** `581d398613e5602a5af361e1c34d3a92ea82ba8e` ([anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens))
**Run written:** 2026-07-15T17:31:16Z

This report documents the **smoke** stage of the Gemma 4 E4B Jacobian Lens
pilot — the second of three planned stages (micro-smoke → **smoke** → pilot).
All figures below are drawn directly from the artifacts in the referenced run
directory; no results from other runs or other models are included.

## Motivation

The upstream [Jacobian Lens](https://github.com/anthropics/jacobian-lens)
reference implementation and its companion paper, [Verbalizable
Representations Form a Global Workspace in Language
Models](https://transformer-circuits.pub/2026/workspace/index.html)
(Anthropic, 2026), fit and evaluate the method on Anthropic's own Claude
models. This fork adapts the same reference implementation to an open-weights
model, `google/gemma-4-E4B-it`, to determine whether the method's residual
stream Jacobian estimator, fitting pipeline, and readout convention transfer
to a different architecture without modifying the upstream `jlens` core.

## Experiment objective

Per the project's `configs/gemma_text_smoke.yaml`, the stated goal of the
smoke stage is narrow: **verify correctness and feasibility end-to-end** on
real Gemma 4 E4B weights — not to produce a research-quality lens. The
config's own comment is explicit: *"NOT a research-quality lens — do not
interpret smoke readouts as the method's capability."* This report treats
that constraint as binding throughout.

## Gemma 4 adaptation overview

The adaptation is a narrow layer on top of the unmodified upstream package
(`jlens/protocol.py`, `hf.py`, `hooks.py`, `fitting.py`, `lens.py` are
untouched):

- [`jlens/gemma4.py`](../jlens/gemma4.py) — resolves the Hugging Face model
  revision to an immutable commit SHA before loading, gates real model
  loading behind an explicit flag, wraps the model as a
  `Gemma4LensModel` (explicit BOS handling, dual pre-softcap/softcapped
  readout), and verifies architecture assumptions before fitting.
- [`jlens/controls.py`](../jlens/controls.py) — negative controls
  (row-permuted fitted Jacobian, scale-matched random matrix, wrong-layer
  application) used in this run's evaluation.
- [`jlens/metadata.py`](../jlens/metadata.py) — config validation,
  fingerprinting, and the `write_metadata` mechanism that produced every
  JSON artifact referenced in this report.
- [`notebooks/archive/legacy_prototypes/gemma_4_e4b_text_jlens.ipynb`](../notebooks/archive/legacy_prototypes/gemma_4_e4b_text_jlens.ipynb)
  — the notebook actually executed to produce this run, including a Google
  Drive persistence section that wrote this run's artifacts to a
  run-scoped directory.

## Experimental setup

Values below are taken from `runs/smoke_20260715T172315460316_fb2eefcd91cd/run_metadata.json` (`load_info`, `architecture_report`, `environment`).

| Field | Value |
|---|---|
| Model repo | `google/gemma-4-E4B-it` |
| Resolved model revision | `fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd` |
| Tokenizer repo / revision | `google/gemma-4-E4B-it` / `fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd` |
| Tokenizer class | `GemmaTokenizer` |
| Model class | `Gemma4ForConditionalGeneration` |
| Text decoder layout | `model.language_model` |
| Layers / hidden size / vocab | 42 / 2560 / 262144 |
| Dense / MoE | dense (`enable_moe_block: false`) |
| Tied unembedding | true |
| Final logit softcap | 30.0 |
| Sliding window / full-attention layers | 512 / 7 of 42 |
| KV-shared layers | 18 |
| Parameters frozen | true |
| Dtype / device | `torch.bfloat16` / `cuda:0` |
| Transformers / PyTorch | 5.12.1 / 2.11.0+cu128 |
| Python | 3.12.13 |
| Platform | Linux 6.6.122+ (Colab), CUDA available |
| GPU | **NVIDIA L4** |

Two architecture facts were recorded but did not block the run, per the
project's protocol ("record, don't hard-fail"):

- `bos_prepended_by_tokenizer`: **false** — the raw tokenizer does not
  prepend BOS on its own; `encode_starts_with_bos`: **true** — the adapter's
  explicit BOS-prepending in `Gemma4LensModel.encode` corrected this, as
  designed.
- `layer_scalars_all_unit`: **false** — per-block `layer_scalar` values range
  from **0.0610** to **0.8867** across the 42 blocks (recorded in full in
  `architecture_report.layer_scalars`), not the identity value of 1.0. The
  architecture verification logged this as a warning and continued, since the
  scalar is part of the block output the lens reads by design.

## Fitting configuration

From `run_metadata.json["config"]` (`sites`, `positions`, `fitting`).

| Field | Value |
|---|---|
| Source layers | 7, 14, 21, 28, 35 (of 42) |
| Target layer | 41 (pre-final-norm residual) |
| Source / target residual site | `block_output` (output of `model.language_model.layers[l]`) |
| Position mask | `skip_first=16`; valid source positions `16 ≤ t < seq_len − 1` |
| Target rule | cotangents at every valid position; causality zeroes `t' < t` |
| Prompt source | `configs/prompts/fit_prompts.json` (plain text, 8 prompts) |
| Prompts fitted | **8** (`n_prompts_fitted`) |
| Max sequence length | 96 tokens |
| `dim_batch` | 8 |
| Seed | 42 |
| Checkpoint cadence | every prompt (`checkpoint_every: 1`) |
| Checkpoint path | `runs/smoke_20260715T172315460316_fb2eefcd91cd/checkpoints/ckpt.pt` |
| Lens artifact | `runs/smoke_20260715T172315460316_fb2eefcd91cd/artifacts/lens.pt` |

Prompt hashes (sha256, first 16 hex chars, from `fit_metadata.json`):
`31c8123d256dbae1`, `4583448a1ad0dbf6`, `2aa9a10f75f44f16`, `ba80a5e2324ced62`,
`e58e92a5f1929b00`, `aa48b66070684753`, `cb6c4cd554b8652a`, `2fbeac629d1248ee`.
The fitting corpus was plain pretraining-style text only — no chat-templated
prompt was included in fitting, consistent with `fit_prompts.json`'s stated
format (`"format": "plain_text"`).

## Evaluation methodology

From `configs/prompts/eval_prompts.json` and `eval_metadata.json`.

- 4 plain-text prompts (`multihop-currency`, `capital-france`, `antonym`,
  `counting`), each evaluated at two token positions (`-2`, `-1`) — 8
  plain-format readout rows.
- 1 chat-templated prompt (`chat-multihop-currency`, the same underlying
  question rendered through the tokenizer's chat template), evaluated at the
  same two positions — 2 chat-format readout rows.
- Plain-text and chat-templated prompts were evaluated **separately**; none
  of the evaluation prompts were part of the fitting corpus.
- For every readout, both **pre-softcap** logits (the paper's `W_U
  norm(J h)` convention) and **softcapped** logits (`30·tanh(x/30)`, Gemma's
  actual output pathway) were recorded; the two share identical top-k
  rankings because the cap is monotonic.
- `top_k = 10` for overlap metrics; `control_seed = 1234`.

**Negative controls** were computed on **one** evaluation prompt
(`multihop-currency`) at the same two positions, per layer:

- **Primary — row-permuted fitted Jacobian**: the fitted `J_l` with its rows
  (target dimensions) randomly permuted, same entries, transport destroyed.
- **Scale-matched random**: an i.i.d. Gaussian matrix matched to `J_l`'s
  Frobenius norm.
- **Wrong-layer**: `J_l` swapped for the Jacobian fitted at the next source
  layer in the cyclic order `7→14→21→28→35→7` (e.g. layer 7's residual is
  transported with the layer-14 Jacobian).
- **Logit lens** (identity transport, `use_jacobian=False`) as the standard
  baseline.

Metric definitions, exactly as computed by
[`jlens/controls.py`](../jlens/controls.py) in the notebook's controls cell:
`overlap` is the top-10 overlap with the model's real output logits,
**averaged over both evaluated positions**; `rank_of_model_top1` is the rank
of the model's actual top-1 token under each variant's logits, taken **only
at the final position** (`-1`). See [Interpretation
Boundaries](#interpretation-boundaries) for what this single-prompt scope
does and does not support.

## Runtime

From `run_metadata.json["probe"]` and `["fit_runtime_seconds"]`.

| Stage | Value |
|---|---|
| Pre-fit memory/runtime probe (1 prompt, `dim_batch=8`, `max_seq_len=48`, 5 layers) | 32.22 s |
| Full fit (8 prompts, `max_seq_len=96`, 5 layers) | **361.5 s** (≈ 6.0 minutes) |
| Local CPU test suite (79 tests), run inside the same Colab session | 7.65 s |

The probe and the full fit ran at different `max_seq_len` values (48 vs 96)
and different prompt counts (1 vs 8); the probe's runtime is not a
per-prompt estimate for the full fit and is reported only as the
feasibility/memory check it was designed to be.

## Memory usage

| Measurement | Value |
|---|---|
| Peak CUDA memory during the pre-fit probe (`dim_batch=8`, `max_seq_len=48`) | **16.61 GB** |
| Peak CUDA memory during the full 8-prompt fit (`max_seq_len=96`) | **not recorded** |

The notebook's fitting cell does not call `torch.cuda.reset_peak_memory_stats`
/ `torch.cuda.max_memory_allocated` around the full `fit()` call — only
`jlens.gemma4.probe_fit_cost` (used for the pre-fit probe) measures peak
memory. The 16.61 GB figure is therefore the probe's measurement, not the
full fit's. See Limitations.

## Validation

- **Local CPU test suite**: 79/79 tests passed inside the Colab session that
  produced this run (cell output, `notebooks/archive/legacy_prototypes/gemma_4_e4b_text_jlens.ipynb`),
  before any model loading — layout auto-detection, Jacobian orientation
  (analytic + finite-difference), checkpoint/resume, controls, config
  validation, and the mock fit→apply pipeline.
- **Architecture verification** (`jlens.gemma4.verify_architecture`)
  confirmed: dense routing, `params_frozen=True`, `n_layers=42`,
  `d_model=2560`, `vocab_size=262144` all matched the config's expectations
  (`expect_n_layers`, `expect_d_model`, `expect_vocab_size`); tied unembedding
  confirmed true.
- **Immutable revision resolution**: `google/gemma-4-E4B-it` resolved to
  commit `fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd` before any weights were
  downloaded (notebook cell output: *"google/gemma-4-E4B-it pinned to
  fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd"*), and that exact SHA was used
  for both the model and tokenizer load.
- **Finite Jacobians**: `probe["all_finite"] = true`; the probe's Jacobian
  shapes were `[2560, 2560]` for each of the 5 fitted layers, matching
  `d_model`.
- **Lens round-trip**: the fitting cell asserts
  `JacobianLens.load(lens_path).source_layers == LENS.source_layers`
  immediately after saving; this assertion passed (no error recorded in the
  fit cell's output).
- **Model-load gating**: `allow_model_load` was `false` in the shipped
  `configs/gemma_text_smoke.yaml`; the notebook's user-set
  `JLENS_ALLOW_GEMMA=1` environment variable was the explicit override that
  permitted loading in this run.

## Quantitative results

Control table, computed on the `multihop-currency` prompt at positions
`[-2, -1]` (from `run_metadata.json["control_rows"]` /
`runs/smoke_20260715T172315460316_fb2eefcd91cd/summary.md`):

| Layer | J-lens (overlap / rank) | logit-lens | permuted | random | wrong-layer |
|---:|---|---|---|---|---|
| 7  | 0.00 / 75422  | 0.00 / 54024  | 0.00 / 28972  | 0.00 / 4089   | 0.00 / 93144  |
| 14 | 0.00 / 187397 | 0.00 / 70932  | 0.00 / 119251 | 0.00 / 25935  | 0.05 / 12499  |
| 21 | 0.00 / 44974  | 0.00 / 112694 | 0.00 / 132069 | 0.00 / 240035 | 0.05 / 3785   |
| 28 | 0.15 / 138    | 0.00 / 85662  | 0.00 / 256023 | 0.00 / 132046 | 0.10 / 111995 |
| 35 | 0.20 / 677    | 0.20 / 1344   | 0.00 / 131061 | 0.00 / 15859  | 0.00 / 203601 |

Observed pattern on this prompt, ranking all five variants by
`rank_of_model_top1` at each layer (lower rank = closer to the model's actual
top-1 token; full ordering below):

| Layer | Rank order, best to worst |
|---:|---|
| 7  | random (4089) < permuted (28972) < logit-lens (54024) < **J-lens (75422)** < wrong-layer (93144) |
| 14 | wrong-layer (12499) < random (25935) < logit-lens (70932) < permuted (119251) < **J-lens (187397)** |
| 21 | wrong-layer (3785) < **J-lens (44974)** < logit-lens (112694) < permuted (132069) < random (240035) |
| 28 | **J-lens (138)** < logit-lens (85662) < wrong-layer (111995) < random (132046) < permuted (256023) |
| 35 | **J-lens (677)** < logit-lens (1344) < random (15859) < permuted (131061) < wrong-layer (203601) |

At layers 7 and 14, J-lens has the *worst or second-worst* rank of all five
variants — every top-10 overlap is 0.00 at these two layers for every
variant, so none of the five recovers the model's actual top-10 output there,
and the rank ordering in that all-zero-overlap regime is not a signal of
comparative fidelity. At layer 21, J-lens ranks second only to wrong-layer
(both still 0.00 overlap). Only at layers 28 and 35 — where J-lens overlap
first becomes non-zero (0.15, then 0.20) — does J-lens have the best rank of
all five variants, by a wide margin: 138 and 677 respectively, versus
five-to-six-digit ranks for every control at those same two layers. At layer
35, J-lens and logit-lens reach the same overlap (0.20 each) but J-lens has
the better rank (677 vs 1344).

## Qualitative observations

Selected examples from `eval_readouts` in
`runs/smoke_20260715T172315460316_fb2eefcd91cd/artifacts/eval_metadata.json`.
Top-5 tokens are the literal decoded strings from that file.

**`multihop-currency`, position `-1`** (model's actual top-1 next token:
`" the"`):

| Layer | J-lens top-5 | Logit-lens top-5 |
|---:|---|---|
| 7  | " messages", " books", " chapters", "aring", " texts" | "ago", "稀", " Cann", "ර", " Ris" |
| 14 | "egna", " Andorra", "bock", "bik", "sik" | "Select", "select", "Green", "别", "side" |
| 21 | " country", " countries", " continents", " continent", " nations" | "ders", "ativas", "ду", " komp", "共" |
| 28 | " called", " mostly", " used", " primarily", " known" | " currency", " forex", "鈔", " 단위", " 무엇" |
| 35 | " Euros", " Euro", " euro", " euros", " currency" | " Euros", " euro", " euros", " Euro", "欧元" |

**`capital-france`, position `-2`** (model's actual top-1: `" Paris"`):

| Layer | J-lens top-5 |
|---:|---|
| 7  | "lots", " lots", " Anyway", "っていう", " 거라고" |
| 21 | " cities", " city", " France", " cityName", "🏙" |
| 35 | " Paris", " Parisian", "Paris", "巴黎", " París" |

**`chat-multihop-currency`, position `-2`** (chat-templated; model's actual
top-1: `" Euro"`):

| Layer | J-lens top-5 |
|---:|---|
| 21 | " arXiv", " pertanyaan", "<|channel>", " akong", " Gemma" |
| 35 | " Euro", " Euros", " euro", "Euro", "欧元" |

Across all recorded examples, the pattern is consistent: J-lens top-5 tokens
at layer 7 read as generic high-frequency words unrelated to the prompt;
mid-layer (14–21) readouts vary — sometimes topical (e.g. "country" /
"continents" for the plain `multihop-currency` prompt at layer 21) and
sometimes not (e.g. the chat-templated variant of the same question at layer
21 produces unrelated tokens); layer-35 readouts converge closely with the
model's own actual output across every recorded example. The logit lens is
close to unreadable (non-English, seemingly unrelated tokens) at layers 7–21
in every recorded example, and becomes topical only at layer 28 or 35.

## Limitations

- **Smoke-scale corpus.** 8 fitting prompts is far below the paper's ~1000
  sequences; per the project's own config comment, smoke-mode readouts
  verify plumbing, not lens quality.
- **Controls computed on one prompt.** The control table above reflects a
  single evaluation prompt at two positions, with `rank_of_model_top1`
  reported for only the final position; it is a feasibility diagnostic, not
  a statistically powered comparison.
- **Full-fit peak memory not measured.** Only the pre-fit probe's peak
  memory (16.61 GB, at a shorter sequence length and single prompt) was
  captured; the actual peak during the 8-prompt, 96-token fit is unknown.
- **`layer_scalar` is non-unit.** Per-block scalars range from 0.061 to
  0.887 (not 1.0); the lens reads the scaled residual as-is, and no
  correction or renormalization for this is applied.
- **Chat vs. plain-text distribution mismatch.** The fitting corpus was
  plain text only; the one chat-templated evaluation prompt shows visibly
  less coherent mid-layer J-lens readouts than the plain-text prompts in
  this run (see Qualitative observations) — consistent with, but not proof
  of, a distribution-mismatch effect.
- **Architecture differences from the paper's models.** Gemma 4 E4B uses
  sliding-window attention (window 512, 7 of 42 layers full-attention) and
  KV-sharing across its last 18 layers; the paper's Jacobian estimator was
  characterized on the models it was originally applied to, not this
  attention pattern.
- **Single run, single seed.** No repeated runs or seed variation were
  performed in this smoke stage; run-to-run variance is unknown.

## Future work

- Run the **pilot** stage (`configs/gemma_text_pilot.yaml`: ~100 WikiText-103
  sequences, 7 layers spanning the full depth) — the first stage whose lens
  quality is intended to be interpretable, per the project's staging design.
- Instrument peak-memory measurement around the full `fit()` call, not only
  the pre-fit probe, so pilot-stage memory requirements are measured rather
  than estimated.
- Extend the control table beyond a single prompt to get a corpus-level
  overlap/rank distribution.
- Evaluate a larger, matched set of chat-templated prompts to characterize
  the plain-text/chat distribution-mismatch observation quantitatively.
- Extension points documented but not implemented in this pass: sparse
  J-space decomposition, gradient pursuit, k-cone discovery/visualization,
  activation steering, multimodal inputs (see README § Non-goals /
  Extension points).

## Interpretation boundaries

**Measured, in this run's artifacts:**

- The smoke fit completed end-to-end on real `google/gemma-4-E4B-it` weights
  at the pinned revision `fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd`, producing
  finite `[2560, 2560]` Jacobians at all 5 fitted layers, in 361.5 s.
  (`run_metadata.json["probe"]["all_finite"]`, `["fit_runtime_seconds"]`)
- On the single `multihop-currency` control prompt, J-lens has the best
  (lowest) `rank_of_model_top1` of all five variants at layers 28 and 35 —
  the two layers where its top-10 overlap first becomes non-zero — by a wide
  margin. At layers 7 and 14, J-lens has the *worst or second-worst* rank of
  the five variants, all of them at 0.00 overlap. At layer 21, J-lens ranks
  second only to wrong-layer. (`run_metadata.json["control_rows"]`; see
  Quantitative results for the full per-layer ordering.)
- Across the recorded `eval_readouts`, J-lens top-5 tokens at layer 35 match
  or closely resemble the model's actual output more often than at layer 7,
  for every prompt where this was recorded.
  (`eval_metadata.json["eval_readouts"]`)

**Not measured — do not treat as established by this run:**

- Whether the method is "meaningfully interpretable" on Gemma 4 E4B in
  general. This smoke run evaluated controls on **one** prompt and full
  readouts on **five** prompt/position combinations (four plain, one chat) —
  far too few to support a general claim about the method's reliability on
  this model.
- Whether the pattern seen at layers 28 and 35 (J-lens rank far better than
  every control) would replicate at pilot scale, on different prompts, or
  with a different random seed. This run used one seed (42) and one prompt
  set.
- Why J-lens has the worst or second-worst rank of the five variants at
  layers 7 and 14 (all five variants are at 0.00 overlap there). This run's
  artifacts record the numbers but do not establish a cause.
- Whether the plain-text/chat coherence gap observed for one chat prompt
  reflects a general property of the method versus an idiosyncrasy of that
  specific prompt or its chat template rendering.
- Any claim about full-fit GPU memory requirements — only the smaller-scale
  probe's memory was measured (see Limitations).
- Any comparison to the original paper's quantitative results — the paper's
  models, corpus size, and evaluation protocol differ from this smoke run in
  ways not controlled for here.
