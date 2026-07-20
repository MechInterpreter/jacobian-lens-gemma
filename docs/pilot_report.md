# Gemma 4 E4B Jacobian Lens — Pilot Report

**Run:** [`runs/pilot_20260715T200437612150_311fd108c23a/`](../runs/pilot_20260715T200437612150_311fd108c23a/)
**Config:** [`configs/gemma_text_pilot.yaml`](../configs/gemma_text_pilot.yaml)
**Config fingerprint:** `sha256:311fd108c23a1bd8869c8bbb3e19013660a41fcbdb249710707d9edd65f4dc22`
**Local repository commit at run time:** `541b0b343a7497991ab57a518d805b1cce1da68c`
**Upstream Jacobian Lens commit:** `581d398613e5602a5af361e1c34d3a92ea82ba8e` ([anthropics/jacobian-lens](https://github.com/anthropics/jacobian-lens))
**Run written:** 2026-07-15T22:47:47Z

This report documents the **pilot** stage — the third and final planned stage
(micro-smoke → smoke → **pilot**) — and compares it with the completed
[smoke stage](smoke_report.md)
(run `smoke_20260715T172315460316_fb2eefcd91cd`). All pilot figures are drawn
directly from the artifacts in the referenced run directory
(`run_metadata.json`, `fit_metadata.json`, `eval_metadata.json`,
`summary.md`); all smoke figures are those recorded in
[`docs/smoke_report.md`](smoke_report.md). No results from other runs or
other models are included. The pilot run is treated as the **authoritative
experimental record** for the fitted 100-prompt lens.

## Experimental setup

| Field | Pilot | Smoke |
|---|---|---|
| Model repo | `google/gemma-4-E4B-it` | same |
| Resolved model revision | `fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd` | same |
| Tokenizer revision | `fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd` | same |
| Local repository commit | `541b0b343a7497991ab57a518d805b1cce1da68c` | `d62814fdf93b8eb36463c86a70ca711891d5c939` |
| Model class / layout | `Gemma4ForConditionalGeneration`, text decoder at `model.language_model` | same |
| Layers / hidden size / vocab | 42 / 2560 / 262144 | same |
| Dense / tied unembedding / softcap | dense, tied, 30.0 | same |
| Dtype / device | `torch.bfloat16` / `cuda:0` (NVIDIA L4) | same |
| Transformers / PyTorch / Python | 5.12.1 / 2.11.0+cu128 / 3.12.13 | same |

Both runs recorded the same two non-fatal architecture facts: the raw
tokenizer does not prepend BOS on its own (`bos_prepended_by_tokenizer:
false`; the adapter's explicit BOS handling corrects this,
`encode_starts_with_bos: true`), and per-block `layer_scalar` values range
0.0610–0.8867 rather than 1.0 (the scalar is part of the block output the
lens reads by design).

## Fitting configuration

| Field | Pilot | Smoke |
|---|---|---|
| Prompt source | WikiText-103 (`fitting.prompt_source: wikitext`), streamed | `configs/prompts/fit_prompts.json` (plain text) |
| Prompts fitted | **100** (`n_prompts_fitted`) | 8 |
| Max sequence length | **128** tokens | 96 tokens |
| Source layers | **3, 7, 14, 21, 28, 35, 38** | 7, 14, 21, 28, 35 |
| Target layer | 41 (pre-final-norm residual) | 41 |
| Source / target site | `block_output` | same |
| Position mask | `skip_first=16`; valid source positions `16 ≤ t < seq_len − 1` | same |
| `dim_batch` / seed | 8 / 42 | 8 / 42 |
| Checkpoint cadence | every 5 prompts | every prompt |

The pilot used **layers 3, 7, 14, 21, 28, 35, and 38** — seven source layers
spanning the full depth of the 42-block stack, versus the smoke stage's five.
All 100 per-prompt SHA-256 hashes are recorded in
`fit_metadata.json["prompt_hashes"]`. The fitting corpus was plain
pretraining-style text only; no chat-templated prompt was fitted on. For
scale reference, the paper's lenses use ~1000 sequences of 128 tokens and
report quality saturating quickly (~100 usable); the pilot sits at that
lower usable bound.

## Runtime and memory

| Measurement | Pilot | Smoke |
|---|---|---|
| Pre-fit probe (1 prompt, `dim_batch=8`, `max_seq_len=48`) | 36.38 s | 32.22 s |
| Probe peak CUDA memory | **16.84 GB** | 16.61 GB |
| Full fit runtime | **9665.4 s (≈ 2 h 41 m)**, 100 prompts × 128 tokens × 7 layers | 361.5 s, 8 prompts × 96 tokens × 5 layers |
| Full-fit peak CUDA memory | **not recorded** (probe-only instrumentation, same gap as smoke) | not recorded |

## Validation

- **Finite Jacobians:** `probe["all_finite"] = true`; all seven per-layer
  Jacobians have shape `[2560, 2560]`, matching `d_model`.
- **Architecture verification** passed: dense routing, frozen parameters,
  42 layers / 2560 width / 262144 vocab as expected, tied unembedding.
- **Immutable revision:** the model and tokenizer were both loaded at the
  pinned commit `fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd`.
- The local CPU test suite passed in the same session before model loading
  (per the notebook's gating cell, as in the smoke run).

## Evaluation methodology

Identical to the smoke stage (same notebook code, same
`configs/prompts/eval_prompts.json`): 4 plain-text prompts
(`multihop-currency`, `capital-france`, `antonym`, `counting`) and 1
chat-templated prompt (`chat-multihop-currency`), each read out at token
positions `−2` and `−1`; `top_k = 10`; `control_seed = 1234`; pre-softcap
and softcapped logits both recorded (identical rankings — the cap is
monotonic). Plain-text and chat prompts evaluated separately; none of the
evaluation prompts were in the fitting corpus.

The control table below was computed on **one** evaluation prompt
(`multihop-currency`) at those two positions: `overlap` is top-10 overlap
with the model's real output averaged over both positions;
`rank_of_model_top1` is the rank of the model's actual top-1 token under
each variant's logits at the final position only. This single-prompt scope
is a known limitation carried over from the smoke stage (see Limitations).

## Complete layer-wise control table (pilot)

From `run_metadata.json["control_rows"]`. Format: top-10 overlap /
rank of the model's top-1 token (rank 0 = argmax; vocabulary 262144).

| Layer | J-lens | logit-lens | permuted | random | wrong-layer |
|---:|---|---|---|---|---|
| 3  | 0.00 / 23422 | 0.00 / 25671  | 0.00 / 93091  | 0.00 / 198331 | 0.00 / 1762   |
| 7  | 0.00 / 824   | 0.00 / 54024  | 0.00 / 53437  | 0.00 / 4080   | 0.00 / 7947   |
| 14 | 0.05 / 35572 | 0.00 / 70932  | 0.00 / 204397 | 0.00 / 25885  | 0.05 / 32569  |
| 21 | 0.05 / 18842 | 0.00 / 112694 | 0.00 / 209056 | 0.00 / 240277 | 0.15 / 1131   |
| 28 | 0.10 / 5205  | 0.00 / 85662  | 0.00 / 170479 | 0.00 / 131744 | 0.10 / 78113  |
| 35 | 0.20 / 510   | 0.20 / 1344   | 0.00 / 132405 | 0.00 / 15936  | 0.20 / 175    |
| 38 | **0.20 / 12** | 0.20 / 54    | 0.00 / 64015  | 0.00 / 22715  | 0.00 / 260183 |

Key observations, stated precisely:

- **Layer 38 produced the strongest reported next-token rank among the
  fitted layers**: J-lens rank 12, versus 510 (L35), 5205 (L28), 18842
  (L21), 35572 (L14), 824 (L7), 23422 (L3).
- **Layers 28, 35, and 38 strongly beat the row-permuted and random
  controls**: J-lens ranks 5205 / 510 / 12 versus permuted ranks 170479 /
  132405 / 64015 and random ranks 131744 / 15936 / 22715 at those same
  layers — three to five orders of magnitude separation at layer 38.
- **The existing wrong-layer control is not consistently destructive and
  requires clarification.** At layer 35 the wrong-layer variant (rank 175)
  actually *beats* the true J-lens (rank 510); at layer 21 it is far better
  than the J-lens (1131 vs 18842); at layer 38 it is maximally destructive
  (260183, near the bottom of the 262144-token vocabulary). See the audit
  below — this behaviour is an artifact of the control's cyclic
  construction, not evidence about transport meaningfulness per se.

### Audit of the wrong-layer control

`jlens.controls.wrong_layer_lens` reassigns the fitted Jacobians
**cyclically over the fitted layer list**: the residual at fitted layer
`layers[i]` is transported with the Jacobian fitted at
`layers[(i+1) % n]`. For the pilot's layer list [3, 7, 14, 21, 28, 35, 38]
this concretely means:

| Applied at layer | Jacobian actually used | Layer distance | Character |
|---:|---:|---:|---|
| 3  | J₇  | 4  | near-adjacent |
| 7  | J₁₄ | 7  | moderate |
| 14 | J₂₁ | 7  | moderate |
| 21 | J₂₈ | 7  | moderate |
| 28 | J₃₅ | 7  | moderate |
| 35 | J₃₈ | 3  | **adjacent (both late)** |
| 38 | J₃  | 35 | **maximally distant (cyclic wraparound)** |

The single "wrong_layer" column therefore mixes qualitatively different
substitutions: at layer 35 it applies the Jacobian of the *nearby late
layer 38* (whose transport is empirically similar — hence rank 175, not
destructive at all), while at layer 38 it applies the Jacobian of *layer 3*
(hence rank 260183, maximally destructive). In the smoke run the same
control at layer 35 used J₇ (wraparound, distance 28) and was destructive
(rank 203601) — so **the same control name at the same layer measured
different things in the two runs**, because the fitted layer sets differ.
This ambiguity motivates the explicitly named adjacent-layer,
distant-layer, and deterministic shuffled-layer controls introduced for
future evaluations in `jlens/controls.py` (see
[README § Improved controls](../README.md)); the completed smoke and pilot
artifacts are left untouched.

## Smoke-versus-pilot comparison

Same-layer J-lens results on the shared control prompt
(`multihop-currency`), smoke → pilot:

| Layer | Smoke J-lens (overlap / rank) | Pilot J-lens (overlap / rank) | Change |
|---:|---|---|---|
| 7  | 0.00 / 75422  | 0.00 / 824   | rank improved ~90× |
| 14 | 0.00 / 187397 | 0.05 / 35572 | overlap 0 → 0.05, rank improved ~5× |
| 21 | 0.00 / 44974  | 0.05 / 18842 | overlap 0 → 0.05, rank improved ~2.4× |
| 28 | 0.15 / 138    | 0.10 / 5205  | **worse** (rank 138 → 5205) |
| 35 | 0.20 / 677    | 0.20 / 510   | comparable, slightly better |

Moving from 8 to 100 fitting prompts improved the J-lens's
rank-of-model-top-1 at four of the five shared layers, most dramatically at
layer 7; layer 28 is the exception, degrading from rank 138 to 5205 on this
one prompt. With a single control prompt and a single seed per stage,
none of these per-layer changes is statistically characterized —
prompt-level variance alone could plausibly produce the layer-28 reversal.
The pilot additionally fitted layers 3 and 38; layer 38 (rank 12) is the
best-performing layer in either run and did not exist in the smoke lens.

The qualitative readouts tell the same story: at layer 35 both runs
converge on the model's actual output (`" Euros"`, `" Paris"`, `" right"`),
while mid layers (14–21) become somewhat more topical in the pilot (e.g.
`capital-france` position −2 at layer 21: smoke `" cities"/" city"/
" France"`, pilot `" metropolis"/" city"/" Paris"`).

## Qualitative layer progression (pilot)

Decoded J-lens top-5 tokens from `eval_metadata.json["eval_readouts"]`
(pre-softcap logits; literal strings from the artifact).

**`capital-france`, position −1** (model's actual top-1: `" Paris"`):

| Layer | J-lens top-5 |
|---:|---|
| 3  | `" anymore"`, `" gonna"`, `" guy"`, `" guys"`, `" nowadays"` |
| 7  | `" eight"`, `" Eight"`, `" recognised"`, `" whilst"`, `" learnt"` |
| 14 | `" cityName"`, `" city"`, `"ponym"`, `" name"`, `"cityName"` |
| 21 | `"的名字"`, `" name"`, `" NAME"`, `"newName"`, `"ListName"` |
| 28 | `" Alexandria"`, `" Columbus"`, `" Florence"`, `" Brighton"`, `" Victoria"` |
| 35 | `" Paris"`, `" Parisian"`, `"Paris"`, `"巴黎"`, `" París"` |
| 38 | `" Paris"`, `" Parisian"`, `"Paris"`, `"巴黎"`, `"パリ"` |

The progression — generic filler (L3–7) → task-schema tokens like
`" cityName"`/`" name"` (L14–21) → the right *category* with wrong
*instances* (`" Alexandria"`, `" Florence"` at L28) → the correct answer
(L35–38) — is representative of the plain-text prompts in this run.

**`multihop-currency`, position −1** (model's actual top-1: `" the"`; the
prompt's eventual answer is the currency):

| Layer | J-lens top-5 |
|---:|---|
| 14 | `" countries"`, `" country"`, `" nation"`, `" nations"`, `"两国"` |
| 21 | `" country"`, `" countries"`, `" Italy"`, `" France"`, `" continent"` |
| 28 | `" Spanish"`, `" Columbus"`, `" called"`, `" Colombian"`, `" Louisiana"` |
| 35 | `" Euros"`, `" euro"`, `" Euro"`, `" euros"`, `"欧元"` |
| 38 | `" Euro"`, `" Euros"`, `" euros"`, `" euro"`, `"欧元"` |

Layer 21 surfaces the unverbalized intermediate (`" Italy"`) of the
two-hop question; layers 35–38 read the eventual answer even though the
model's literal next token at this position is `" the"`.

**`counting`, position −1** (model's actual top-1: `" "`; the correct
count is five):

| Layer | J-lens top-5 |
|---:|---|
| 14 | `" numberOf"`, `" number"`, `"numberOf"`, `" Anzahl"`, `"NumberOf"` |
| 21 | `" numberOf"`, `" countable"`, `" counting"`, `" number"`, `" counted"` |
| 28 | `":"`, `" seven"`, `" five"`, `" six"`, `" three"` |
| 35 | `":"`, `" "`, `" five"`, `" seven"`, `" counted"` |
| 38 | `" "`, `":"`, `" five"`, `" counted"`, `" count"` |

At layer 28 several candidate counts appear with the correct one (`" five"`)
among them but not dominant.

**`chat-multihop-currency` (chat format), position −1** (model's actual
top-1: `"Euro"`):

| Layer | J-lens top-5 |
|---:|---|
| 14 | `"Trivia"`, `"UNKNOWN"`, `" Trivia"`, `" trivia"`, `"Incorrect"` |
| 21 | `"<turn|>"`, `"<eos>"`, `` "```" ``, `" الاجابه"`, `"nesium"` |
| 28 | `"Spanish"`, `"Brazilian"`, `"Colombia"`, `"Portuguese"`, `"Colomb"` |
| 35 | `" Euros"`, `" Euro"`, `" euro"`, `" euros"`, `"欧元"` |
| 38 | `" Euro"`, `"Euro"`, `" euro"`, `"欧元"`, `" Euros"` |

Mid-layer chat readouts surface template/control tokens (`"<turn|>"`,
`"<eos>"`) that plain-text prompts never show — consistent with (but not
proof of) the plain-text-only fitting corpus mismatching the chat
distribution, as first observed in the smoke run.

The logit-lens baseline remains near-unreadable below layer 28 in every
pilot example and converges with the J-lens at layers 35–38, as in the
smoke run.

## Interpretation boundaries

**Measured, in this run's artifacts:**

- The pilot fit completed end-to-end on 100 WikiText prompts at 7 layers in
  9665.4 s on an NVIDIA L4, producing finite `[2560, 2560]` Jacobians.
- On the single `multihop-currency` control prompt, the J-lens beats the
  permuted and random controls by large margins at layers 28, 35, and 38,
  and reaches rank 12 for the model's top-1 token at layer 38.
- The wrong-layer control's non-monotone behaviour across layers is fully
  explained by its cyclic construction (audit table above).

**Not measured — do not treat as established by this run:**

- **One pilot run does not establish ignition, a global workspace, or
  generalization across seeds and datasets.** The paper's workspace and
  ignition claims rest on evidence (interventions, capacity studies,
  cross-model replication) that this project has not produced for Gemma.
  Nothing in this run distinguishes "late layers linearly align with the
  output vocabulary" from any stronger workspace-style claim.
- Whether the layer-28 smoke→pilot regression is real or prompt noise
  (single prompt, single seed per stage).
- Lens quality on any distribution other than these 5 evaluation prompts
  (10 readout rows); the control table rests on 1 prompt.
- Any cross-model comparison with the paper's results (different models,
  corpus sizes, and evaluation protocols).

## Limitations of the current evaluation set

- **Five prompts, one control prompt.** Aggregate statistics (median rank,
  hit rates) are meaningless at n=1–5; no per-category breakdown is
  possible; a single anomalous prompt dominates conclusions (see the
  layer-28 reversal).
- **Position convention conflates roles.** Positions −2/−1 mean different
  things across prompts (inside a word vs. before the answer).
- **One chat prompt.** The plain-vs-chat comparison rests on a single
  example.
- **Controls under-specified.** The wrong-layer control mixes adjacent and
  maximally-distant substitutions under one name (audit above), and control
  provenance (which J was used where) was not recorded in the output
  metadata.
- **No aggregate metrics.** Only per-layer single-prompt overlap/rank were
  recorded; no median rank, MRR, or hit rates across examples.

These limitations directly motivate the Phase-2 evaluation improvements
(named controls with recorded provenance, a categorized held-out evaluation
set, and aggregate statistics) in this branch.

## Implications for the sparse-decomposition phase

- **Layer selection.** J-lens fidelity is strongest at layers 28–38 and
  essentially absent at 3–7 on the evaluated prompts. Sparse J-space
  decompositions are therefore run at layers **14, 21, 28, 35, 38** first;
  decompositions at layers 3–7 would be built on transport with no
  demonstrated readout fidelity.
- **The lens is frozen.** The decomposition phase must consume
  `runs/pilot_20260715T200437612150_311fd108c23a/artifacts/lens.pt`
  as-is (fingerprint-verified) — no refitting — so that decomposition
  results remain attributable to this documented lens.
- **Expectation calibration.** The paper reports the J-space component
  accounts for a small fraction of activation variance ("never more than
  10%"); high reconstruction error on real activations is expected and is
  not by itself evidence of failure.
- **Confound to keep visible.** At layers 35–38 the J-lens and logit lens
  converge; any "sharp late-layer transition" in decomposition quality must
  be weighed against ordinary late-layer vocabulary alignment before being
  read as workspace-like structure.
