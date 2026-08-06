# J-space language autoencoder — design note and feasibility study

Branch: `experiment/jspace-language-autoencoder`
· Package: [`jlens/autoencoder/`](../jlens/autoencoder)
· Config: [`configs/jspace_language_autoencoder.yaml`](../configs/jspace_language_autoencoder.yaml)
· Scripts: [`build_jspace_language_dataset.py`](../scripts/build_jspace_language_dataset.py),
[`train_phrase_reconstructor.py`](../scripts/train_phrase_reconstructor.py),
[`train_cone_adapter.py`](../scripts/train_cone_adapter.py),
[`evaluate_jspace_language.py`](../scripts/evaluate_jspace_language.py)
· Notebook: [`notebooks/jspace_language_autoencoder_colab.ipynb`](../notebooks/jspace_language_autoencoder_colab.ipynb)
· Tests: `tests/test_jspace_language_dataset.py`, `tests/test_jspace_language_models.py`,
`tests/test_jspace_language_eval.py`, `tests/test_jspace_language_e2e.py`,
`tests/test_jspace_language_notebook.py`

> ## Status: implemented, smoke-validated on deterministic mocks. **No real
> Gemma pilot has been run.** Every number below that is not explicitly
> labelled "measured on mocks" is a *prediction*, not a result. The GO/NO-GO
> verdict is produced by code, not asserted here.

This experiment is **separate from and does not touch** the completed
generative-validation experiment (`jlens/generative.py`,
`scripts/run_generative_validation.py`, `configs/gemma_generative_validation.yaml`,
`notebooks/generative_jlens_colab.ipynb`). Nothing in those files is modified.

## The cycle under test

```
h_14  ──pursuit(k=10)──►  q  ──adapter──►  memory  ──frozen Gemma──►  phrase
                          ▲                                             │
                          └────────  frozen reconstructor  ◄────────────┘
                                          q_hat
```

The claim being tested is *not* "the adapter can learn to name things". A
supervised adapter trained on (q, phrase) pairs will name things whether or
not `q` carries the meaning, because the training signal alone is enough to
memorize the training phrases. The claim is the **geometric round trip**: for
*concept-disjoint held-out phrases*, a phrase Gemma emits from `q` alone maps
back — through a reconstructor that never saw those phrases — to a vector
that agrees with `q` and disagrees with unrelated cones. Every gate, control,
and baseline in this document exists to separate those two possibilities.

## Phase 1 — what the existing code already fixes

Inspected before any code was written; the autoencoder reuses all of it
unchanged.

| Concern | Existing mechanism | What the autoencoder does with it |
|---|---|---|
| Activation capture | [`jlens.hooks.ActivationRecorder`](../jlens/hooks.py) — forward hook on `model.language_model.layers[l]`, output is the post-attention/MLP/PLE/`layer_scalar` residual (the `block_output` site) | Same hook, `at=[14]`, last context position |
| Lens loading | [`JacobianLens.load`](../jlens/lens.py) (`weights_only=True`) + [`metadata.file_sha256`](../jlens/metadata.py) | Same loader; the pilot `lens.pt` sha256 is a hard gate |
| J-space dictionary | [`JSpaceDictionary.from_lens`](../jlens/pursuit.py) — rows of `W_U @ J_l`, chunked build, float32 | Same, layer 14 only, `build_chunk_rows` set |
| Gradient pursuit | [`gradient_pursuit`](../jlens/pursuit.py), nonnegative, deterministic tie-break | Same, `k=10`, `normalize_atoms=True`, `refine_steps=2` |
| Cone reconstruction | [`weighted_reconstruction`](../jlens/generative.py) — `q = Σ a_i v_i` over **raw** atoms | Imported and reused; this defines `q` |
| Chat formatting | [`render_receiver_prompt`](../jlens/generative.py) / `_check_chat_structure` — one user turn, `add_generation_prompt=True`, exactly one BOS, structural assertions | Reused verbatim for the verbalizer prompt |
| Generation | [`greedy_decode`](../jlens/generative.py) — uncached, `logits_from_ids(..., n_last=1)` | Beam search here follows the same rule (see "Caching") |
| Double-norm trap | `HFLensModel.logits_from_ids` vs `forward().last_hidden_state` | Never call `unembed(last_hidden_state)`; all logits go through `logits_from_ids` |
| Provenance | [`tensor_sha256`](../jlens/generative.py), [`write_metadata`](../jlens/metadata.py), [`append_record`](../jlens/interventions.py) | Same three helpers for every dataset record and checkpoint |

## Exact tensor shapes (Gemma 4 E4B, `d_model = 2560`, `vocab = 262144`)

| Symbol | Shape | Dtype | Space |
|---|---|---|---|
| `h_14` source activation | `[2560]` | float32 (cast from bf16 block output) | layer-14 residual |
| dictionary `D_14` | `[262144, 2560]` | float32 (2.7 GB) | rows of `W_U @ J_14` |
| pursuit `token_ids` / `coefficients` | `[B, 10]` | long / float32 | atom indices, `a_i ≥ 0` |
| **`q`** cone vector | `[2560]` | float32 | layer-14 residual space |
| `q_unit = q / ‖q‖` | `[2560]` | float32 | the reconstructor's target space |
| phrase token ids | `[T]`, `2 ≤ T ≤ 6` | long | Gemma vocabulary |
| frozen phrase embeddings | `[T, 2560]` | float32 (cast from bf16) | `embed_tokens` output |
| reconstructor hidden | `[T, 512]` | float32 | learned |
| **`q_hat`** | `[2560]`, unit norm | float32 | same space as `q` |
| adapter memory | `[M, 2560]`, `M = 4` | float32 → cast to block dtype | `embed_tokens` output space |
| verbalizer prompt ids | `[1, P + M]` | long | `P ≈ 30` chat tokens |
| beam batch during search | `[W, P + M + t]`, `W = 8` | long | — |

`q` and `q_hat` live in the **same** space by construction: `q` is a
nonnegative combination of rows of `W_U J_14`, which are vectors in the
layer-14 residual space, and the reconstructor's output head emits `2560`
dimensions. Nothing is projected, padded, or truncated between them. The only
transformation applied to either is L2 normalization, recorded per record.

## Where the adapter memory enters Gemma

**Chosen mechanism for the pilot: soft-prefix memory in the input-embedding
stream**, implemented as
[`SoftPrefixConditioner`](../jlens/autoencoder/conditioning.py).

Construction of the prompt (`jlens/autoencoder/prompting.py`):

1. Render the **constant** instruction through the tokenizer's own chat
   template with `add_generation_prompt=True`, with a literal sentinel
   `@@JMEM@@` at the start of the user content.
2. Split the rendered string on the sentinel, tokenize the two halves with
   `add_special_tokens=False`, and ensure exactly one leading BOS.
3. Splice `M` copies of a filler token id between the halves:
   `ids = left ++ [filler]*M ++ right`. The memory span is
   `[len(left), len(left) + M)`.
4. `_check_chat_structure`-equivalent assertions run on the joint rendering
   (one BOS at position 0, two `<start_of_turn>`, one `<end_of_turn>`, a
   non-empty generation prefix).

At forward time a hook on `model._embed_tokens` overwrites rows
`[len(left), len(left)+M)` of its `[batch, seq, 2560]` output with the
adapter's memory, broadcast across the beam batch. The filler token's own
embedding is *discarded*, never added to; its id matters only for the record's
hash. The hook is registered by a context manager and removed on any exit path,
exactly like `jlens.generative.steering_injection`.

Why the embedding site rather than a native layer-14 memory:

- Gradients reach the adapter through the **whole** frozen stack, which is
  what "Gemma reads the memory" has to mean; nothing is bypassed.
- The scale question is answerable: Gemma's `embed_tokens` output has a known
  empirical RMS (the checkpoint applies `sqrt(d_model)` scaling inside the
  embedding module), and the adapter's final `RMSNorm`-style gain is
  calibrated against a measured token-embedding RMS at run time, recorded in
  the checkpoint metadata. Injecting at layer 14 would require picking a
  magnitude relative to a mid-stack residual whose scale we would have to
  measure per-position anyway.
- It composes with `logits_from_ids`, so the LM head, final norm, and Gemma's
  logit softcap all run exactly once, by the library, in the library's order.

`ConditioningBackend` is a `Protocol` (`prompt_ids`, `memory_span`,
`conditioned(...)` context manager). A native-layer variant — injecting the
memory as an additive residual edit at layer 14 through the existing
`steering_injection` hook — can be added later behind the same interface
without touching the adapter, the trainer, or the evaluator.
`jlens/autoencoder/conditioning.py` contains a `NATIVE_LAYER_MEMORY_TODO`
note stating exactly what would change.

## Caching and autoregressive generation

Beam search (`jlens/autoencoder/verbalizer.py`) is **uncached**, matching the
convention `jlens.generative.greedy_decode` established and the repo's
greedy-equivalence test:

- Every step reruns the full forward pass with `use_cache=False`.
- Logits come from `model.logits_from_ids(ids, n_last=1)`, so the LM head runs
  the same one-row GEMM `generate()` uses (`logits_to_keep=1`). Full-sequence
  heads accumulate differently in bf16 and would put the beam scores off the
  path `generate()` would take.
- All `W` beams are kept at **equal length** and evaluated in one batched
  forward `[W, P+M+t]`. A beam that emits `<end_of_turn>` (id 106) or EOS is
  marked finished, its score frozen, and it is padded — never extended — so
  the trailing pad positions can never change a finished beam's score.
- Cost per example: `L` forward passes of batch `W` (`L ≤ 8`, `W = 8`), i.e.
  ≤ 64 sequence-positions-worth of prefill each. This is the dominant cost of
  the whole experiment and is why the pilot uses beams only where they are
  needed (preference training and evaluation), not for the warm start.

Teacher forcing (warm start) needs **no** decoding: one forward over
`[1, P + M + T]` scores every phrase token at its causal position, which
(causal attention, position-independent memory) equals the sum of per-step
decode log-probabilities under the same memory.

The KV cache is deliberately not used. It is a pure speed optimization here
and it interacts with the embedding hook (the hook fires only for positions
actually embedded in a given call); adding it would require a fresh
equivalence proof against the uncached path, which is not worth the pilot's
time budget.

## How concept leakage is prevented

Leakage is the failure mode that would make a positive result meaningless, so
it is enforced by construction and by assertion, not by convention.

1. **Adapter input is `q` and nothing else.** `ConeAdapter.forward(q)` takes a
   single `[B, 2560]` tensor. It has no other arguments — there is no channel
   through which a token id, phrase string, atom index, example id, or source
   prompt could arrive. `tests/test_jspace_language_models.py` asserts the
   signature and asserts that permuting the batch of `q` permutes the memory
   identically (so nothing positional leaks).
2. **The instruction is constant.** One string,
   `jlens.autoencoder.prompting.VERBALIZER_INSTRUCTION`, identical for every
   concept, asserted byte-identical across every record in a run and screened
   by `jlens.generative.assert_clean_prompt` against the priming vocabulary
   (`internal`/`representation`/`concept`/`label`/`value`) that confounded the
   earlier generative runs.
3. **Splits are by phrase identity.** `assign_splits` ranks phrases by
   `sha256(salt | normalized phrase)` and cuts at the split quantiles, so every
   occurrence of a phrase lands in the same split and the assignment depends
   only on the *set* of phrases, never on corpus or mining order. A held-out
   phrase's *string*, its *token ids*, and its *cones* are absent from both the
   reconstructor's and the adapter's training data.
   `assert_no_split_leakage` re-derives the assignment from the phrase text of
   every stored record and fails on any crossing, and additionally fails if
   any normalized held-out phrase string appears in a training phrase's text.
   (Plain hash *bucketing* was the first implementation and was replaced: at
   feasibility-study sizes the phrase set is the top-N by corpus frequency, not
   a random sample, and a 32-phrase smoke build was observed with an empty
   held-out split — silently removing the only split the conclusions may rest
   on. The miner additionally drops phrases that are substrings of an already
   accepted one, so "Great Barrier" and "Great Barrier Reef" can never land in
   different splits.)
4. **The reconstructor is frozen before the adapter is trained.** Its
   checkpoint carries a sha256 of its own weights; the adapter trainer records
   that hash and refuses to run if the reconstructor's parameters have
   `requires_grad=True`. The adapter's optimizer is constructed over
   `adapter.parameters()` only, and an assertion walks
   `optimizer.param_groups` to prove no reconstructor or Gemma parameter is in
   it.
5. **Gemma stays frozen.** `HFLensModel.__init__` already sets
   `requires_grad_(False)` on every parameter; `assert_gemma_frozen` re-checks
   before each training loop and after it, and `verify_architecture` hard-fails
   on any trainable parameter. Gemma parameters cannot appear in any optimizer
   state (asserted by id-set membership, not by name matching).
6. **Metadata never enters a tensor.** Every dataset record is split into a
   `payload` (tensors: `q`, `h`, phrase ids) and a `provenance` block
   (strings/hashes). The training dataloaders read from `payload` only; a test
   feeds a dataset whose provenance strings are the *answers* and asserts the
   trained-model outputs are unchanged.

## Reward and abstention (Phase 5/6)

For a candidate phrase `c` generated from cone `q`:

```
recon(c)      = max(0, cos(q, q_hat(c)))                     # reconstruction
margin(c)     = recon(c) - max_{q' ∈ unrelated} cos(q', q_hat(c))
brevity(c)    = -λ_len * max(0, len(c) - target_len)
dup(c)        = -λ_dup * (occurrences of normalized c in the beam - 1)
R(c)          = w_r*recon + w_m*margin + brevity + dup
```

`alpha* = argmin_{α ≥ 0} ‖q - α q_hat‖²  =  max(0, ⟨q, q_hat⟩) / ‖q_hat‖²`, and
the scale-fitted explained fraction is `1 - ‖q - α* q_hat‖²/‖q‖² = max(0, cos)²`
— reported alongside raw cosine everywhere. A candidate is **accepted** only if
`recon ≥ accept_recon_min` and `margin ≥ accept_margin_min`; otherwise the
example **abstains** and contributes to the abstention rate rather than to a
forced top-1.

Preference optimization is offline and pairwise (DPO-style against a frozen
copy of the warm-start adapter):

```
loss = -log σ( β * [ (logπ(c+) - logπ_ref(c+)) - (logπ(c-) - logπ_ref(c-)) ] )
```

over pairs from the same beam with `R(c+) - R(c-) ≥ reward_gap`. Log-probs are
length-normalized so the pair loss cannot be won by length alone (length is
priced explicitly in the reward instead). A REINFORCE refinement exists behind
`preference.policy_gradient.enabled` and is **off** by default.

## Baselines (Phase 7)

Implemented in [`baselines.py`](../jlens/autoencoder/baselines.py); all eight
run through the identical prompt, beam width, and scorer, so only the memory
differs:

| id | memory |
|---|---|
| `zero_memory` | adapter output replaced by zeros |
| `shuffled_q` | `shuffled_coordinates(q)` (norm and value multiset preserved) |
| `unrelated_q` | another held-out phrase's `q` |
| `sign_reversed_q` | `-q` |
| `naive_token_average` | mean of the dictionary atoms of the phrase's **own** token ids (an upper bound on what constituent-token J-vectors alone could say) |
| `jlens_token_clues` | no adapter: the pursuit's selected atom token strings, ranked by coefficient — the ordinary J-lens readout |
| `adapter_raw_beam` | trained adapter, beam order, **no** reranking |
| `adapter_reranked` | trained adapter + frozen-reconstructor reranking |

`naive_token_average` needs the phrase's tokens and is therefore an *oracle*
baseline: it is reported as a reference ceiling for "constituent tokens alone",
never as a member of the adapter's own pipeline.

## Go / no-go criteria

Encoded in `jlens.autoencoder.evaluation.gonogo_report`, all on
concept-disjoint held-out phrases:

| # | Criterion | Threshold |
|---|---|---|
| 1 | reconstructor correct-vs-distractor AUROC | `≥ 0.80` |
| 2 | reconstructor correct-phrase top-5 retrieval | `≥ 50%` |
| 3 | adapter beam contains correct phrase vs zero-memory | `≥ zero + 0.10` absolute |
| 4 | reranking improves top-1 exact recovery | `> raw beam` |
| 5 | accepted-output precision vs unfiltered | `≥ unfiltered + 0.10` |
| 6 | unrelated/shuffled/zero cones rejected or materially lower | acceptance `≤ 0.5×` correct-cone acceptance |
| 7 | no leakage detected | zero violations |

A failure is attributed to exactly one of: `phrase_reconstructor`,
`cone_adapter`, `decoding_interface`, `cone_information_loss`,
`insufficient_data`, `prompt_dependence`, using the diagnostic table in
`attribute_failure` (e.g. gate 1 failing with high *train*-phrase retrieval is
`insufficient_data`; gate 1 failing with low train retrieval too is
`cone_information_loss`). The report always states the verdict; there is no
code path that hides a NO-GO.

## Known scientific risks

1. **`q` may be dominated by the output token.** The pursuit's first atom is
   usually the model's own top-1 next token. If `q` is essentially "the next
   token's J-vector", the reconstructor can succeed by learning
   token-identity, and the round trip would be tautological rather than
   semantic. The `naive_token_average` baseline is the direct test: if it
   matches `adapter_reranked`, the cycle carries no more than constituent
   tokens. This risk is *not* resolved by any gate above.
2. **Prototype collapse.** Averaging unit `q` over occurrences may wash out
   context-specific structure, leaving a target the reconstructor can hit from
   phrase frequency alone. Reported as per-phrase within/between prototype
   dispersion; a between/within ratio near 1 invalidates gate 1.
3. **Soft-prefix expressivity.** `M = 4` continuous vectors in embedding space
   can encode far more than 2560 float32 dimensions' worth of *phrase* through
   sheer optimization pressure, i.e. the adapter could learn a lookup that
   works on training phrases and transfers by surface similarity. Held-out
   concept disjointness is the guard; the zero-memory baseline is the floor.
4. **Reconstructor as a language model.** A phrase encoder over frozen Gemma
   embeddings can score "plausible English noun phrase" rather than "matches
   this cone". The specificity margin against unrelated cones and the
   confabulation-attractor probe (`black hole`, `photosynthesis`,
   `quantum entanglement`, `Great Barrier Reef`) are the tests; a high
   `recon` with near-zero `margin` means the reconstructor learned English.
5. **Single layer, single k.** Layer 14 and `k = 10` are fixed by the brief. A
   NO-GO under these settings does not generalize to other layers or
   sparsities and must not be reported as one.
6. **WikiText domain.** Phrases mined from WikiText-103 skew encyclopedic;
   `q` for a mid-sentence entity mention may be more "next-word prediction"
   than "concept". Cross-domain transfer is untested.

---

# Implementation report

Branch `experiment/jspace-language-autoencoder`, based on `cd61969`
(`experiment/generative-jlens-validation`). Pushed to
`origin/experiment/jspace-language-autoencoder`.

## What was implemented

The whole cycle, behind four CLI stages and one notebook:

| stage | script | produces |
|---|---|---|
| 2 | `build_jspace_language_dataset.py` | `dataset/{records.jsonl,tensors.pt,manifest.json}`, `artifacts/{benchmark,leakage}.json` |
| 3 | `train_phrase_reconstructor.py` | `reconstructor.pt` + sidecar, `artifacts/reconstructor_{metrics,gate}.json` |
| 4-5 | `train_cone_adapter.py` | `adapter_warm.pt`, `adapter.pt`, `adapter_epoch*.pt` resume points, `artifacts/adapter_training.json` |
| 6-8 | `evaluate_jspace_language.py` | `artifacts/{evaluation,gonogo,prompt_robustness}.json`, `summary.md` |

Conditioning is soft-prefix memory in the input-embedding stream, behind the
`ConditioningBackend` protocol. Generation is uncached batched beam search
(width 8, at most 8 tokens, stopping on `<end_of_turn>` or EOS). Preference
training is offline pairwise DPO against a frozen copy of the warm-start
adapter; REINFORCE exists and is off by default.

## Trainable and frozen parameter counts

At the shipped pilot config (`d_model = 2560`):

| module | trainable | frozen |
|---|---|---|
| `ConeAdapter` (M=4, hidden 1024) | **14,183,424** | 0 |
| `PhraseReconstructor` (hidden 512, 2 layers, 8 heads) | **6,869,504** | 0 |
| **total trainable** | **21,052,928** | — |
| Gemma 4 E4B | 0 | all (`requires_grad=False`, asserted) |
| fitted lens `J_14`, dictionary `W_U J_14` | 0 | frozen inputs, never written |

The reconstructor's checkpoint reports `trainable_parameters: 0` because it is
frozen *before* being saved; the pre-freeze count is recorded separately under
`extra.training_summary`.

## Data-generation procedure

1. Stream WikiText-103 documents of at least 400 characters.
2. Per sentence, propose 2-4 word spans: capitalized runs (not sentence-initial)
   and content-word bigrams/trigrams whose first and last words are not function
   words.
3. Rank candidates by corpus frequency (ties alphabetical, so the order is
   deterministic), then filter: function-word-only, contextual token count
   outside 2-6, fewer occurrences than required, and **substring overlap with an
   already-accepted phrase**.
4. For each accepted phrase, take the first *N* occurrences; the context is the
   document text up to the phrase's first character.
5. Tokenize the context keeping the **tail** (`max_context_tokens`, one leading
   BOS), and drop occurrences with less than `min_context_tokens` of preceding
   context.
6. Capture the layer-14 `block_output` residual at the last context position —
   the position whose next-token distribution is the phrase's first token.
7. Run `gradient_pursuit` (`k=10`, nonnegative, `normalize_atoms`,
   `refine_steps=2`) against rows of `W_U J_14`; `q = Σ a_i v_i`.
8. Assign splits by hash rank over the mined phrase set; write records, tensors,
   and a manifest with file-level sha256s.

Phrase token ids are always the **contextual** segmentation — what appending the
phrase adds beyond the constant prompt's tail — so the dataset, the
reconstructor, and the adapter cannot disagree about how a phrase is split.

## Leakage safeguards (as implemented)

1. `ConeAdapter.forward(self, q)` — asserted by signature inspection in
   `tests/test_jspace_language_models.py` and again in the end-to-end test.
2. One constant instruction, screened by `assert_clean_prompt`; the end-to-end
   test asserts the rendered prompt and memory span are byte-identical across
   all four stages.
3. Splits by phrase identity, re-derived and re-checked by
   `assert_no_split_leakage` (recomputation **and** substring containment); the
   miner drops overlapping phrases up front.
4. The reconstructor is frozen before the adapter is trained; the adapter's
   checkpoint stores the reconstructor's parameter sha256 and the evaluator
   refuses a mismatched pair.
5. `assert_gemma_frozen` before and after every loop,
   `assert_no_frozen_parameters_in_optimizer` by tensor identity, and
   `assert_no_gemma_gradients` after every backward pass.
6. Records separate tensors from provenance strings; training reads only
   `cones`, `phrase_token_ids`, and split membership.

## Reconstructor gate results on the deterministic mock

The mock has no semantics, so these numbers validate the *plumbing*, not the
hypothesis. Smoke build: 32 phrases / 62 occurrences, splits 20/6/6 phrases.

| split | AUROC | top-1 | top-5 | explained fraction | specificity margin |
|---|---|---|---|---|---|
| train | 0.770 | — | 1.000 | 0.124 | +0.008 |
| val | 0.503 | 0.167 | 0.667 | 0.051 | −0.209 |
| heldout | 0.497 | 0.167 | 0.667 | 0.045 | −0.181 |

Gate verdict **NO-GO** (AUROC 0.497 below the 0.80 threshold), and the final
report attributes it to `insufficient_data` — the intended behaviour on a
semantics-free 32-phrase mock, and a demonstration that the gate and the
attribution logic fire, not a claim about J-space.

## Tests and validation

`128 passed` across six new files:

- `test_jspace_language_config.py` (13) — schema, unknown-key rejection, the
  brief's fixed parameters, smoke overrides, fingerprints.
- `test_jspace_language_dataset.py` (16) — mining, overlap filter, tail-keeping
  context, split determinism and non-emptiness, prototypes, leakage detection,
  persistence, benchmark shape.
- `test_jspace_language_models.py` (33) — prompt structure and memory span,
  conditioning (logits change, hook removal, batch broadcast, span mismatch,
  NaN rejection), adapter signature / permutation / scale-invariance,
  reconstructor unit-norm and padding invariance, differentiability into the
  adapter only, padding-independence of scores, beam determinism, checkpoint
  hashing.
- `test_jspace_language_eval.py` (35) — scale fit, AUROC ties, rank ties, reward
  decomposition, preference pairs, baselines, and seven GO/NO-GO cases including
  three that force a NO-GO.
- `test_jspace_language_e2e.py` (16) — the four scripts in order on CPU.
- `test_jspace_language_notebook.py` (15) — nine sections, no stored outputs,
  every cell parses, token / lens / deletion invariants.

`ruff check` is clean on every file added here. Seven pre-existing ruff findings
in `jlens/jspace_analysis.py`, `scripts/analyze_jspace.py`, and
`tests/test_explorer_export.py` are untouched and out of scope. The full suite
reports `566 passed, 3 failed`; all three failures reproduce on the base commit
in this working tree (two notebooks carry uncommitted stored outputs, and
`test_explorer_export` needs a run directory that is not present) and none of
them involve this experiment.

## Commits

| hash | phase |
|---|---|
| `882a112` | scaffold + design note |
| `b4ea8cc` | dataset builder |
| `acee105` | reconstructor + gate |
| `45fd022` | adapter + verbalizer + preference |
| `4b31ba4` | inference + baselines + evaluation |
| `cedfc12` | Colab runner + end-to-end smoke test |

## Exact Colab pilot instructions

1. Open `notebooks/jspace_language_autoencoder_colab.ipynb` on an **L4** runtime.
2. Run section 1 in order (runtime facts, clone and HEAD assertion, API check, HF
   token, Drive mount, configuration, **lens verification**, test suite, stage
   runner).
3. Upload the pilot lens to
   `MyDrive/jacobian-lens-gemma/runs/pilot_20260715T200437612150_311fd108c23a/artifacts/lens.pt`
   (91,753,066 bytes, sha256 `7229c756...c96f474`). Section 1g aborts on a
   mismatch.
4. Set `SMOKE = False` in cell 1f and rerun it.
5. Run sections 2 through 7 in order, stopping at section 4 if the gate fails.

Equivalently, from a shell in the checkout (`DRIVE` pointing at
`MyDrive/jacobian-lens-gemma`):

```bash
python -u scripts/build_jspace_language_dataset.py --config configs/jspace_language_autoencoder.yaml --output-dir "$DRIVE/jlang_runs/jlang_pilot" --allow-model-load --device-map cuda --runs-root "$DRIVE/runs" --benchmark
```

```bash
python -u scripts/train_phrase_reconstructor.py --config configs/jspace_language_autoencoder.yaml --output-dir "$DRIVE/jlang_runs/jlang_pilot" --allow-model-load --device-map cuda --runs-root "$DRIVE/runs"
```

```bash
python -u scripts/train_cone_adapter.py --config configs/jspace_language_autoencoder.yaml --output-dir "$DRIVE/jlang_runs/jlang_pilot" --allow-model-load --device-map cuda --runs-root "$DRIVE/runs"
```

```bash
python -u scripts/evaluate_jspace_language.py --config configs/jspace_language_autoencoder.yaml --output-dir "$DRIVE/jlang_runs/jlang_pilot" --allow-model-load --device-map cuda --runs-root "$DRIVE/runs"
```

Exit codes: `0` GO, `2` aborted precondition, `3` reconstructor gate failed,
`4` NO-GO, `130` interrupted at a safe boundary (rerun to continue).

## Interruption and resume

A Colab runtime can be taken away mid-sentence. Every expensive stage is
therefore resumable from the same Drive-backed run directory: stop it, reconnect
later, rerun the setup cells, and rerun the stage. Nothing that finished is
recomputed and nothing incomplete is mistaken for finished.

### Resume support by stage

| stage | granularity | resume point | scientifically identical to an uninterrupted run? | test |
|---|---|---|---|---|
| dataset construction | one pursuit chunk (`dataset.capture_batch_size` occurrences) | `dataset/shards/shard_*.pt` + checksum sidecar | yes — byte-identical `records.jsonl`, bitwise-identical tensors | `test_dataset_resume_skips_completed_records_and_matches_clean_build` |
| reconstructor training | **batch** (every epoch, plus every `--checkpoint-every-steps`) | `checkpoints/reconstructor_*.pt` | yes — identical parameter sha256 and per-epoch history | `test_reconstructor_resume_matches_uninterrupted_training` |
| adapter warm start | **batch** (as above) | `checkpoints/adapter_warm_*.pt`, plus the legacy `adapter_epoch*.pt` | yes — identical parameter sha256 and history | `test_adapter_warm_start_resume_is_deterministic` |
| preference training | **batch** (as above) | `checkpoints/adapter_preference_*.pt` | yes — identical parameter sha256 and history | `test_preference_resume_matches_uninterrupted_training` |
| evaluation | one result shard: `(record, baseline)`, `(paraphrase, record)`, the attractor probe | `evaluation_shards/{baseline,robustness,confabulation}/*.json` | yes for every result; `resources` records the cost actually incurred | `test_evaluation_reuses_shards_and_aggregates_identically` |

Each row's "yes" is the assertion the named test makes, not a design intention.

### What a checkpoint contains

Training resume points (`jlens.autoencoder.checkpoint.v2` via
[`save_training_checkpoint`](../jlens/autoencoder/checkpoints.py)) carry:

- module parameters, with a sha256 verified on load;
- optimizer state, and scheduler / AMP `GradScaler` state when present;
- Python, NumPy, torch CPU, and all CUDA RNG states;
- the sampler generator's state **and** the interrupted epoch's shuffle order;
- the interrupted epoch's running loss sums, so its reported mean is the mean
  the uninterrupted run would have reported rather than a mean over the tail;
- epoch, batch index, and global step;
- the full stage identity block (below), the timestamp, and the reason:
  `periodic`, `epoch_complete`, `keyboard_interrupt`, or `stage_complete`.

### Configuration compatibility

Resume is refused unless the stored identity matches the current one. Two
classes, and the distinction is not negotiable:

**Semantic — never resumable, no flag overrides.** Model repo and revision, lens
checksum and run directory, source layer, pursuit settings, phrase-split policy
(salt and fractions), dataset identity, architecture dimensions, and the
upstream artifact hashes a stage depends on (dataset manifest, reconstructor,
warm-start adapter). Changing one of these produces a different experiment;
resuming across it would produce artifacts describing a configuration that never
made them.

**Non-semantic — refused by default, waivable with `--allow-config-drift`.**
Epoch counts, batch sizes, checkpoint cadence, run id, run directory path.

A refusal names the fields that moved and writes nothing.

### Stage state layout

```
RUN_DIR/
  state/
    dataset/            state.json  progress.json  complete.json
    reconstructor/
    adapter_warm/
    adapter_preference/
    evaluation/
  checkpoints/          <stage>_{epoch,step,interrupt}_*.pt + .json sidecars
                        beam_cache/   preference beams, keyed and bounded
  dataset/
    shards/             per-chunk build shards
    records.jsonl  tensors.pt  manifest.json
  evaluation_shards/    baseline/  robustness/  confabulation/
  artifacts/            evaluation.json  gonogo.json  gates  leakage
  summary.md
```

Each stage reports one of `not_started`, `in_progress`, `interrupted`,
`complete`, `incompatible`, `failed`. `complete` is backed by `complete.json`
alone: `state.json` is written before the marker, so a process killed between
the two reports `interrupted`, which is what it is.

A completed stage is skipped on rerun unless `--force` is given or its recorded
artifacts fail their checksum check. Nothing is ever silently overwritten.

### Safe stop

1. In the notebook: `stop_stage("<stage>")` (cell 1m). From a shell:
   `kill -INT <pid>` or `kill -TERM <pid>`.
2. The signal handler only sets a flag — it does no serialization, because a
   handler that writes torch state can corrupt the checkpoint it is writing.
3. The loop notices at its next safe boundary, finishes or discards the unit it
   is on, writes a checkpoint atomically, prints the stage, the checkpoint path,
   and an exact resume command, and exits `130`.
4. No completion marker is written, so the stage stays resumable.

Colab's stop button interrupts the *monitoring* cell, not the stage subprocess;
the stage keeps running until `stop_stage()` asks it to stop.

### Resume

```bash
python -u scripts/train_cone_adapter.py --config configs/jspace_language_autoencoder.yaml --run-dir "$DRIVE/jlang_runs/jlang_pilot" --resume --allow-model-load --device-map cuda --runs-root "$DRIVE/runs"
```

`--resume` is the default. Before resuming, `--status-only` prints every stage's
status, progress, and last checkpoint, and `--validate-state` checks every
checkpoint and shard on disk without doing any work. In the notebook these are
cells 1j and 1k.

### Restarting one stage

```bash
python -u scripts/evaluate_jspace_language.py --config configs/jspace_language_autoencoder.yaml --run-dir "$DRIVE/jlang_runs/jlang_pilot" --force
```

`--force` restarts the stage the script owns. `train_cone_adapter.py` owns two,
so it also takes `--force-stage adapter_warm|adapter_preference` to restart
exactly one — useful when preference training needs redoing and the warm start,
the expensive half, does not. In the notebook: `restart_stage("<stage>")`.

A restart sets the completion marker aside (renamed `complete.json.superseded`)
and recomputes. It does **not** delete checkpoints or shards; they are simply not
reused. Deleting them is a separate, manual decision.

### What survives, and what does not

**A clean stop or a runtime disconnect** costs at most the unit in flight: the
current dataset shard, the optimizer steps since the last checkpoint, or the
evaluation shard being generated.

**A hard kill — `SIGKILL`, an OOM kill, Colab reclaiming the VM — is not
handled, and nothing running inside the VM can handle it.** What bounds the
damage is cadence: work since the last checkpoint or shard is lost. Set
`--checkpoint-every-steps` (notebook: `CHECKPOINT_EVERY_STEPS`) to decide how
much that can be. Every write is a temp file followed by `os.replace`, so a hard
kill can lose recent work but cannot leave a half-written file that loads as a
valid one — a `.pt` without its checksum sidecar is treated as absent.

Two residual exposures are worth stating plainly:

- **A single long unit is not interruptible partway.** One pursuit chunk, one
  optimizer step, or one record's beam generation runs to completion before the
  stop flag is checked. On the pilot config that is seconds, not minutes.
- **Preference beams are cached only while the adapter is unchanged.** An
  interruption during a batch's generation is not repaid on resume; once the
  adapter takes a step the cache legitimately misses, because beams from a
  different policy are not reusable.

### Checkpoint cleanup

Nothing here deletes checkpoints; a run directory grows by one resume point per
epoch per training stage, plus any periodic ones. After a stage completes, its
`checkpoints/<stage>_*` files are only needed to re-derive an interrupted
trajectory and can be removed by hand once `reconstructor.pt` / `adapter.pt`
are archived. The beam cache under `checkpoints/beam_cache/` is bounded
(`--beam-cache-capacity`, default 512 entries, oldest dropped first) and is safe
to delete at any time — it is a cache, and a miss only costs regeneration.
Dataset shards and evaluation shards are what make those stages resumable; keep
them until the run's artifacts are archived.

## Expected runtime and storage

Measured on the CPU mock (32 phrases): 0.0039 s per occurrence end to end,
1,280 bytes per occurrence, 444 model forward calls for a 12-record held-out
evaluation. These do **not** transfer to an L4 running E4B — which is why
`--benchmark` exists and why the pilot must be sized from its output rather
than from anything in this document.

What *is* known analytically, and is the main cost risk:

| stage | model forward calls (pilot config) |
|---|---|
| dataset capture | 1,800 (one per occurrence, ~128 tokens) |
| adapter warm start | 4,500 (batch 8, ~40 tokens) |
| **preference training** | **about 57,600 beam steps** (4 epochs x 1,800 examples x 8 steps, batch 8) plus ~3,600 scoring passes |
| held-out evaluation | 8 baselines x held-out records x 8 steps |

Storage: the dataset holds two float32 `d_model` vectors per occurrence, about
21 KB each, so roughly 40 MB at 1,800 occurrences; checkpoints are about 57 MB
(adapter) and 28 MB (reconstructor), plus one resume point per epoch.

Preference training dominates and could plausibly run for several hours on an
L4. If the first benchmark says so, the knobs — in order of least damage to the
claim — are `preference.epochs`, `adapter.beam_width` during training only, and
subsampling the training split for the preference phase. Reducing
`evaluation.beam_width` or the held-out set instead would weaken the result and
should not be the first choice.

## Unresolved scientific risks

The six risks in the design note above are unchanged and none is resolved by any
gate implemented here; output-token dominance of `q` (risk 1) is the most
serious, and `naive_token_average` is the only instrument pointed at it.
Additionally:

- **Preference training has never run against real cones.** Its stability, the
  reward scale, and whether a gap of 0.02 produces usable pairs are all untested
  outside the mock.
- **The memory scale is calibrated, not validated.** `memory_rms_scale = 1.0`
  puts the memory at real-token magnitude; whether frozen Gemma *attends* to it
  at that magnitude is an empirical question the first warm start answers.
- **Soft-prefix versus native-layer memory.** The pilot result is about the
  embedding-stream interface. A NO-GO there does not rule out a layer-14 memory,
  which is why the backend sits behind an interface.
