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
