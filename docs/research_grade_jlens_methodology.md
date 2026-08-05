# Research-grade J-lens calibration: what Anthropic actually did, and what we change

This document exists so that every element of tomorrow's calibration can be
traced to one of three origins: **reproduced** from an Anthropic primary
source, **adapted** because Gemma 4 differs from the models the paper used, or
**new** — invented by this project and carrying no authority from the paper.

Nothing here is taken from a summary, a blog post, or memory. Where a detail is
unknown, it is written down as unknown.

## Primary sources consulted

| Source | What it settles |
|---|---|
| Gurnee, Sofroniew, Pearce, Ameisen, Kauvar, Tarng, Olah & Batson, *Verbalizable Representations Form a Global Workspace in Language Models*, Transformer Circuits Thread, 6 July 2026 — <https://transformer-circuits.pub/2026/workspace/index.html> | The definition of `J_l`, the corpus scale, the role of the lens, the logit-lens and tuned-lens comparisons |
| `anthropics/jacobian-lens` — <https://github.com/anthropics/jacobian-lens> (Apache-2.0) | The reference implementation: estimator, position masking, accumulation, checkpointing, serialization |
| `jlens/fitting.py` module docstring, upstream — <https://github.com/anthropics/jacobian-lens/blob/main/jlens/fitting.py> | The exact estimator and its relation to a strict per-position Jacobian |
| Upstream `README.md`, corpus-size paragraph | The 1000×128 calibration corpus and the saturation claim |

**This repository vendors the upstream package.** `jlens/fitting.py`,
`jlens/lens.py`, `jlens/hooks.py`, `jlens/hf.py`, `jlens/protocol.py`,
`jlens/examples.py`, `jlens/vis.py`, `jlens/_logging.py` and `jlens/__init__.py`
carry `Copyright 2026 Anthropic PBC` headers and are unmodified upstream code at
commit `581d398613e5602a5af361e1c34d3a92ea82ba8e`
(`jlens.metadata.UPSTREAM_COMMIT`). Every other module carries a
`Gemma 4 adaptation — new in this fork` header. **The methodology is therefore
not being re-implemented from a paper description; it is being called.** That is
the strongest form of reproduction available, and it is why the classifications
below can be exact rather than inferred.

---

## 1. The single most important finding

> **The J-lens has no fitting objective, no optimizer, no training schedule, no
> stopping rule, and no train/test separation — because `J_l` is not learned.**

`J_l = E[∂h_final/∂h_l]` is a **population mean of a Jacobian**, and
`jlens.fitting.fit` is a **plug-in estimator** of it: a running sum of
per-prompt Jacobians divided by the number of prompts
(`jlens/fitting.py:391-435`). There is no loss function anywhere in the upstream
package. Nothing is minimized. Nothing converges in the optimization sense.

The Day-1 brief asks this workflow to freeze an "optimizer configuration", a
"fitting objective" and a "stopping rule". Those fields cannot be filled in
against the primary source, because the quantities do not exist. Inventing them
would mean building something other than a J-lens and calling it one — which the
brief itself forbids. The protocol therefore records them as
`not_applicable_estimator_is_a_sample_mean` and states why, rather than
supplying invented values.

Three consequences follow, and they shape the entire study:

1. **"More prompts" reduces estimator *variance*, nothing else.** The estimator
   is unbiased for its own target regardless of scale. Increasing 100 → 1,000
   shrinks the Monte-Carlo error of `Ĵ_l` by roughly `√10 ≈ 3.16×`. It cannot
   fix a lens that is wrong in expectation.
2. **Scale points are exactly nested for free.** Because the estimator is a
   running mean over a deterministically ordered prompt list, the sufficient
   statistic (`jacobian_sum`, `n_done`) at prompt 100 *is* the 100-prompt lens,
   and the same accumulator continued to 250 and 1,000 gives the later lenses.
   The three lenses cost exactly as much as the 1k lens alone. This is
   implemented as snapshots of the accumulator, not as three separate fits.
3. **The primary source predicts the study will find a plateau immediately.**
   The upstream README states verbatim: *"The paper's lenses use 1000 sequences
   of 128 tokens from a pretraining-like corpus. Quality saturates quickly
   (§9.3); ~100 prompts is usable."* Our first scale point tests that stated
   usable threshold, and the 1,000-prompt endpoint matches the paper's full
   production scale. **If earlier layers fail at 1k, the primary source gives us no
   reason to expect larger runs to rescue them.** That is a real answer to the
   scientific question, not a disappointing one — see §6.

---

## 2. Methodology table

`R` = reproduced from a primary source · `A` = adapted for Gemma 4 · `N` = new
in this project (no authority from the paper) · `U` = unknown, not stated in any
primary source we could reach.

### 2.1 The lens itself

| Item | Value | Class | Evidence |
|---|---|---|---|
| Jacobian definition | `J_l = E[∂h_final/∂h_l]` | R | Paper; README |
| Readout | `lens_l(h) = unembed(J_l @ h)` | R | README; `jlens/lens.py:135-216` |
| `unembed` | final norm → `lm_head` | R | Paper ("standard layer normalization before multiplication by `W_U`") |
| Estimator | one-hot cotangent at **every valid target position at once**; gradient at source `p` is `Σ_{p'≥p} ∂h_final[p']/∂h_l[p]`; **mean over source positions** | R | `jlens/fitting.py:11-17` docstring: *"This is the reduction used in the paper."* |
| Alternative per-position estimator | explicitly *not* used | R | Same docstring: gives "a slightly different `J_l`; both work as a lens" |
| Accumulation | running mean over prompts | R | `jlens/fitting.py:391-435` |
| Layer-specific parameters | one `[d_model, d_model]` matrix **per layer**, never shared | R | `jacobians: dict[int, Tensor]`, `jlens/lens.py:22-44` |
| Target layer | final block (`n_layers - 1`) | R | `_check_layer_indices`, `jlens/fitting.py:79` |
| Position mask | `skip_first=16`, final position excluded | R | `SKIP_FIRST_N_POSITIONS = 16`, `jlens/fitting.py:42-72` |
| Reason for `skip_first` | attention-sink positions have atypical residual statistics | R | `jlens/fitting.py:40-41` |
| Sequence length | 128 tokens | R | `max_seq_len: int = 128`; README "1000 sequences of 128 tokens" |
| Corpus scale | 1000 sequences | R | README |
| Corpus identity | "pretraining-like" / "generic web-text" — **no dataset named** | U | Paper and README both decline to name it |
| Saturation | "Quality saturates quickly (§9.3); ~100 prompts is usable" | R | README |
| Precision | fp32 accumulate, fp16 serialize | R | `jlens/fitting.py:149`, `jlens/lens.py:52-55` |
| `dim_batch` | 8 (memory knob; total backward FLOPs unchanged) | R | `jlens/fitting.py:106,251` |
| Sharding | disjoint prompt slices + `JacobianLens.merge` (`n_prompts`-weighted mean) | R | README; `jlens/lens.py:105-133` |
| Checkpointing | atomic temp-file + `os.replace`, every prompt by default | R | `_atomic_save`, `jlens/fitting.py:216-221` |
| Resume refusal | mismatched `source_layers` / `target_layer` / `skip_first` raises | R | `jlens/fitting.py:297-313` |
| **Fitting objective** | **none — sample mean** | R | absence of any loss in upstream |
| **Optimizer / schedule / batching of an optimization** | **none** | R | same |
| **Train/val/test split for fitting** | **none in upstream** | R | `fit()` takes one prompt list |
| Prompt construction / formatting | raw text; chat template available but not used for fitting | R/U | `resolve_prompt` exists for *visualisation* examples; the paper's calibration corpus formatting is not stated |

### 2.2 Gemma 4 adaptations

| Item | Value | Class | Evidence |
|---|---|---|---|
| Model | `google/gemma-4-E4B-it` @ `fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd` | A | `jlens/gemma4.py:58`; pinned by this project |
| Architecture | 42 blocks, `d_model` 2560, vocab 262144, dense, tied unembedding | A | `verify_architecture`, `docs/pilot_report.md` |
| Layout | text decoder at `model.language_model` | A | `GEMMA4_LAYOUT`, `jlens/gemma4.py:65` |
| Source site `h_l` | block output — **after** attention, MLP, per-layer-embedding re-injection and `layer_scalar`; i.e. the exact input to block `l+1` | A | `jlens/gemma4.py:27-36` |
| Target site | block 41 output, the **pre-final-norm** residual | A | same |
| Softcap | `30·tanh(x/30)`, strictly monotonic → **token rankings identical** pre/post cap | A | `jlens/gemma4.py:36-40`; `jlens/controls.py:24-27` |
| Readout convention | pre-softcap `W_U·norm(J_l h)` is the paper's convention; both reported | A | `unembed_pair`, `jlens/gemma4.py:156-177` |
| BOS | prepended explicitly in `encode`, not via tokenizer attributes | A | `Gemma4LensModel.encode`; the raw tokenizer does **not** prepend BOS |
| `layer_scalar` | 0.0610–0.8867, **not** 1.0; part of the block output the lens reads by design | A | `docs/pilot_report.md` |
| Corpus | WikiText-103-raw-v1 (`Salesforce/wikitext`), streamed, `min_chars=600` | A | `load_wikitext_prompts`, `jlens/examples.py:42-60` — **this fork's instantiation of "pretraining-like"; the paper names no corpus** |
| dtype / device | bf16 on CUDA | A | `docs/pilot_report.md` |

### 2.3 New in this project — no authority from the paper

| Item | Class | Note |
|---|---|---|
| Tie-aware layer-validity gate (midrank, tied-at-max ceiling, fold stability) | N | The paper does **not** gate layers for publication. Ours exists because we make per-layer causal claims the paper does not. |
| Permuted-row and norm-matched-random control lenses | N | The paper's comparisons are the logit lens, the tuned lens, and matched-norm *intervention* controls — not control *lenses*. |
| Wrong-layer control | N | Ours. |
| Three-way fit / validation / confirmation split | N | Meaningless for an unbiased sample mean in the paper's framing; we need it because our **gate** is a decision procedure that can overfit. |
| Scale study with a plateau rule | N | The paper reports saturation qualitatively; the nested-scale comparison is ours. |
| Target-token diversity floors | N | Ours, to repair the 7-distinct-target defect in the 32-prompt v2 validation. |
| Per-layer publication guard and frozen artifacts | N | Ours. |
| "Research-grade" as a threshold | N | Not a term the paper uses. |

### 2.4 Deliberately out of scope today

Dictionary / J-space construction, gradient pursuit, cone decomposition and
their validation metrics are **not** part of calibration and are not touched.
They live in `jlens/pursuit.py`, `jlens/cones.py` and `jlens/jspace_analysis.py`
and consume a frozen lens. Today's workflow produces lenses; it does not
decompose them.

---

## 3. Where we do **not** match Anthropic, stated plainly

1. **The corpus is not the paper's corpus.** The paper says "pretraining-like"
   and names nothing. We use WikiText-103. Any difference in lens quality
   between our result and theirs is confounded with this and cannot be
   attributed to depth, scale, or model.
2. **The model is not their model.** Gemma 4 E4B is not a Claude model. The
   paper's layer-depth findings are about their models' geometry.
3. **`layer_scalar` is not 1.0 in Gemma 4.** The block output the lens reads is
   scaled per layer by 0.061–0.887. This is inside the object being
   differentiated, so the estimator remains correct, but the residual-stream
   normalization convention differs from any unit-scalar model.
4. **We publish per-layer validated artifacts; the paper publishes a lens.**
   Our gate is an addition, not a reproduction.
5. **`§9.3` is cited by the upstream README but our fetch of the paper's HTML
   could not confirm a section with that number.** We cite the README's claim as
   the README's claim, and flag the paper-side location as **requiring
   verification**. Do not repeat "§9.3" as if it were read directly.
6. **The paper's own guidance supports staging before the 1k endpoint.**
   See §1.3 and §6.

---

## 4. Reuse audit of this repository

Everything below already exists and is already tested. The calibration package
adds orchestration, not physics.

| Component | Location | Reuse | Change required | Scientific consequence |
|---|---|---|---|---|
| Jacobian estimator | `jlens/fitting.py:100-213` | **As-is, untouched** | none | The estimator stays byte-identical to upstream; our numbers are comparable to the paper's by construction |
| Multi-layer capture in one pass | `jlens.fitting.jacobian_for_prompt` | **As-is** | none | All 8 layers already come from **one** forward and one set of backward passes — `torch.autograd.grad(inputs=source_activations)` takes every source layer at once (`fitting.py:187-192`). Nothing to build. |
| Running-mean accumulation + resume | `jlens.fitting.fit` | **As-is** | none | Already atomic, already refuses incompatible `source_layers`/`target_layer`/`skip_first` |
| Scale snapshots | `fit(on_prompt=...)` hook | **Reuse, new callback** | new observer | `on_prompt` exposes `jacobian_sum` and `n_done` (`fitting.py:412-428`), so 100/250/1k snapshots cost **zero** extra compute |
| Lens container / save / load / merge | `jlens/lens.py` | **As-is** | none | fp16 serialization, `merge()` for sharded runs |
| Model loading, revision pinning | `jlens.gemma4.load_gemma4`, `resolve_revision` | **As-is** | none | Immutable SHA resolved before download; `allow_model_load` gate preserved |
| Architecture verification | `jlens.gemma4.verify_architecture` | **As-is** | none | Hard-fails on MoE/width/depth/trainable params |
| Residual hook | `jlens/hooks.py` `ActivationRecorder` | **As-is** | none | Upstream; `start_graph_at` keeps the graph only from the shallowest source layer |
| Tie-aware scoring row | `jlens.mmlocalize.lens_validity.tie_aware_row` | **As-is** | none | Optimistic/pessimistic/midrank + tie counts, already the project's standard |
| Tie-aware gate | `lens_validity.LayerValidityGate`, `evaluate_layer_validity` | **Reuse, re-parameterized** | new instance with calibration thresholds + its own protocol tag | Same code path, same meaning of every number, different `n_prompts`. The gate's `digest` binds it to the fingerprint. |
| Legacy v2 gate | `lens_validity.LegacyValidityGate` | **Reuse for reporting** | none | Both gates printed per layer, as in localization |
| Control lenses | `jlens/controls.py` `permute_rows`, `scale_matched_random`, `layer_mapped_lens`, `distant_layer_mapping` | **As-is** | use `distant_layer_mapping`, not the deprecated cyclic `wrong_layer_lens` | `wrong_layer_lens` is documented as superseded (`controls.py:88-97`); the cyclic shift conflates near and far substitutions |
| Logit-lens diagnostic | `JacobianLens.apply(use_jacobian=False)` | **As-is** | none | Upstream; diagnostic only, never blocking |
| Target-diverse selection | `lens_validity.select_target_diverse_prompts` | **Reuse, re-parameterized** | none | Already refuses rather than lowering the floor (`InsufficientTargetDiversityError`) |
| Atomic unit store + fingerprint resume | `jlens/mmpilot/store.py` | **Primitives only** | new store; `STAGES` is an mmpilot-specific tuple | We reuse `canonical_json`, `payload_checksum`, `safe_key`, `IncompatibleStateError` and the `RunFingerprint` pattern, and define calibration stages separately rather than widening mmpilot's vocabulary. Precedent: `jlens/native_readout.py` does exactly this. |
| Config validation / fingerprint / env manifest | `jlens/metadata.py` | **Reuse** | `environment_manifest`, `file_sha256`, `write_metadata`, `local_git_commit` as-is | Records torch/transformers/upstream/local commit with every artifact |
| Cost probe | `jlens.gemma4.probe_fit_cost` | **As-is** | none | Measures one prompt's wall time and peak CUDA memory before committing to a full run |
| Colab bootstrap + builder-generated notebook | `scripts/_build_localization_notebook.py`, `tests/_mmlocalize_notebook_runner.py` | **Pattern reuse** | new builder + runner | Keeps the committed notebook output-free and byte-reproducible from source |
| MOCK model | `tests/mock_gemma4.py` | **As-is for architecture tests** | new synthetic world for scale behaviour | The existing mock has the exact Gemma 4 module layout; the scale study needs a *statistical* mock instead (§ Stage 9) |

### Explicitly not imported

Per the brief, and confirmed absent from the calibration package: language
autoencoders (`jlens/autoencoder/`, which lives on another branch), phrase
reconstruction, activation-to-language decoding, multimodal lens fitting
(`jlens/mmpilot/`), cross-modal mappings, and the explorer/dashboard
(`jlens/explorer_export.py`, `explorer/`). The calibration package imports from
`jlens.fitting`, `jlens.lens`, `jlens.controls`, `jlens.gemma4`,
`jlens.metadata`, and the two *pure-function* helpers noted above. No SpokenCOCO
data, image, audio, or multimodal activation is reachable from any code path.

---

## 5. What "capture all layers in one pass" actually costs

Worth stating precisely, because it drives the budget and it is the one place
where intuition is wrong.

`jacobian_for_prompt` runs **one** forward pass, then
`ceil(d_model / dim_batch)` backward passes against the retained graph. For
Gemma 4 E4B at `dim_batch=8` that is `ceil(2560/8) = 320` backward passes **per
prompt**. Each backward pass differentiates from the target layer down to
`min(source_layers)` and extracts gradients for *all* source layers
simultaneously.

Therefore:

- **Adding a layer to the grid is nearly free**, as long as it is not shallower
  than the current shallowest — it adds one `[2560, 2560]` slice-copy per
  backward pass, not another backward pass.
- **Lowering the shallowest layer is the expensive change**, because it extends
  every one of the 320 backward passes through more blocks.
- **The 320 backward passes per prompt are irreducible** for an exact dense
  Jacobian. This is the dominant cost and no configuration removes it.

Measured on this project's own hardware (`docs/pilot_report.md`): 100 WikiText
prompts × 128 tokens × 7 source layers (shallowest = 3) took **9,665.4 s on one
NVIDIA L4** — **96.7 s/prompt** — with peak probe memory 16.84 GB of the L4's
24 GB at `dim_batch=8`, `max_seq_len=48`.

That measurement is the anchor for every number in
`docs/research_grade_jlens_calibration_protocol.md`, and it is what makes the
honest budget uncomfortable.

---

## 6. The scientific question, restated honestly

The brief asks whether earlier-layer failure was caused by inadequate fitting.
Given §1, the sharpest available statement is:

> The v2 lens was fitted on 32 prompts. The paper's lenses use 1,000. If
> layers 20/26/32 fail at 1,000 prompts under the same tie-aware gate they
> failed under at 32, then **inadequate calibration scale is excluded as the
> explanation**, because the lens is now at the primary source's own production
> scale on a corpus of the paper's stated type. The remaining explanations are
> that the earlier-layer residual geometry of Gemma 4 E4B genuinely does not
> support linear transport to the output basis, or that something other than
> scale (corpus type, position rule, site convention) is responsible.
>
> Conversely, if a layer that failed at 32 prompts **passes** at 1,000, the v2
> result is explained as an underpowered estimate and nothing about Gemma's
> depth was ever established by it.

**Both outcomes are informative.** The 100 and 250 checkpoints test the primary
source's saturation claim economically; the 1k endpoint matches the paper. No
larger scale is authorized in the two-week protocol.

---

## 7. Implementation report

Written after the code existed, and it records what the implementation is, not
what it was hoped to be.

- `jlens/calibration/` contains 9 modules and no framework: corpus handling,
  planning/budget, state, fitting orchestration, scale comparison, publication,
  mock, report, baseline manifest.
- The estimator is **not reimplemented**. `jlens.calibration.fitting` calls
  `jlens.fitting.fit` and observes it through `on_prompt`. If upstream's
  estimator is wrong, ours is wrong identically — which is the intended
  property.
- Scale snapshots are taken from the accumulator at the nested boundaries. A
  test asserts the 1k snapshot equals a standalone 1k fit exactly.
- MOCK mode constructs a synthetic world with four known layer archetypes and
  never imports `torch.cuda`, `transformers`, `datasets`, or `huggingface_hub`.
- The confirmation split is guarded by an object that refuses to expose it until
  scale and settings are recorded as chosen, and publication refuses on any
  layer that did not pass the gate *on the confirmation set*.

See `docs/research_grade_jlens_calibration_protocol.md` for the frozen protocol
and the budget.
