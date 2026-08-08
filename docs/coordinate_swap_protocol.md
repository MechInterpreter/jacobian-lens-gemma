# Anthropic-style J-lens coordinate swap — method, audit, and protocol

**Status: implemented and MOCK-validated. No real run exists, and none can be
designed yet.** This document specifies the intervention, records the
repository audit it was built from, and states what it may and may not be used
to claim.

![Two J-space interventions](assets/intervention_methods.png)

The figure's editable source is [`assets/intervention_methods.svg`](assets/intervention_methods.svg);
re-render it with `python scripts/render_schematic.py`. It replaces an earlier
proposal graphic that carried the equation
`h' = h + alpha (V δ_target − V δ_source)`. That equation describes **neither**
method below: it is direction steering with a lens-vector difference, and no
run in this repository has ever computed it.

Primary source: <https://transformer-circuits.pub/2026/workspace/index.html>,
"Technical details of J-lens use cases".

---

## 1. Three things that are not the same

| | Method | Formula | Where |
|---|---|---|---|
| **A** | Source-derived J-space steering | `h' = h ± α·v_concept`, `v_concept = V·ReLU(mean_pos_code − mean_neg_code)` | `jlens/mmpilot/causal.py` — **the completed three-modality result** |
| | Single-vector J-lens steering | `h' = h + α·(v_target − v_source)` | not implemented; named here only to be excluded |
| **B** | Exact two-coordinate patching | `V = [v_source v_target]`, `c = pinv(V)h`, `h_patched = h + α·V(σ(c) − c)` | `jlens/mmpilot/coordinate_swap.py` — **planned, no result yet** |

A is a valid causal steering experiment and is what the completed run measured.
It must not be called a coordinate swap. Its terminology —
*source-derived positive-minus-negative J-space directions* — stays exactly as
it is.

### Why B is not the single-vector steer

The algebra collapses to

```
V(σ(c) − c) = (c_target − c_source) · (v_source − v_target)
```

so the update *direction* is parallel to `v_source − v_target`. The difference
is the coefficient: B measures `c_target − c_source` from the activation, while
the single-vector steer assumes 1. An activation carrying no source content is
barely moved by B; one saturated with it is rewritten. Because `v_source` and
`v_target` are rows of `W_U J_l` — in general neither orthogonal nor unit norm —
that coefficient cannot be recovered from dot products, which is why the
coordinates are read with a pseudoinverse and never with `V^T h`.

### What B preserves

`h = V c + r` with `r ⟂ span(V)`, so `h_patched = V σ(c) + r`: the component
orthogonal to `span{v_source, v_target}` is unchanged **exactly**, not
approximately. `tests/test_coordinate_swap.py` asserts a drift below `1e-12` at
every `α`.

---

## 2. Required audit, resolved from code

Every convention below was read out of the repository, not assumed.

| # | Question | Answer | Evidence |
|---|---|---|---|
| 1 | How is the layer-specific frozen J-lens stored? | `JacobianLens` — `{layer: Tensor[d_model, d_model]}`, fp16 on disk, float32 in memory, loaded `weights_only=True`. The published calibration artifacts are one `.pt` per layer with a JSON sidecar carrying the confirmation verdict. | `jlens/lens.py:33-103`, `jlens/mmpilot/published_lens.py:53-99` |
| 2 | Orientation and normalization of each token-indexed lens vector? | Token `v`'s lens vector is **row `v`** of `W_U @ J_l`, shape `[d_model]`, in *source-layer* residual space. **Raw and unnormalized.** `atom_norms` exists but is only ever used to rescale *selection* correlations; coefficients always refer to raw atoms. | `jlens/pursuit.py:158-167`, `PursuitSettings.normalize_atoms` docstring |
| 3 | How do token ids map to dictionary atoms? | Identity. `atoms` is built as `W_U @ J_l` with `W_U = [vocab, d_model]`, so atom index **is** the vocabulary token id — `dictionary.atoms[token_id]`. | `jlens/pursuit.py:266-287` |
| 4 | Are `JSpaceDictionary` vectors rows or columns? | **Rows.** `atoms` is `[n_atoms, d_model]`. The swap therefore has to *transpose them into columns* to build `V = [d_model, 2]`, which is the single most likely orientation mistake and is checked explicitly. | `jlens/pursuit.py:176-181`; `assert_vector_orientation` |
| 5 | Are final-norm weights folded in? | **No.** The pilot builds dictionaries with `final_norm_weight=None` and records `final_norm_weight_folded: False`. The paper's stated convention omits it; `from_lens` exposes it as an explicit opt-in. | `jlens/mmpilot/jspace.py:47-57, 220-243` |
| 6 | How do existing residual hooks handle shape, wrappers, positions, dtype, device? | Block output may be a tensor or a tuple; only element 0 is edited and every other element passes through. Shape must be `[batch, seq, d_model]` or it raises. Negative positions resolve against the actual sequence length. Arithmetic in float32, cast back to the block's own dtype and device. Handles removed on success *and* on every exception path. Parameters must be frozen. | `jlens/interventions.py:314-362`, `jlens/hooks.py:46-56` |
| 7 | How does teacher-forced scoring separate prompt from candidate tokens? | `BuiltInputs.prompt_len` is the original prompt length. `_extend_tensors` appends candidate ids to `input_ids` and to **named decoder-sequence fields only** (`attention_mask`, `token_type_ids`, `mm_token_type_ids`, `position_ids`); media tensors are indexed by pixels or audio frames and pass through untouched. Candidate token `i` is scored at logits position `prompt_len - 1 + i`, and a length mismatch raises rather than being scored. | `jlens/mmpilot/capability.py:83-174` |
| 8 | What of the existing intervention implementation is safely reusable? | Reused unchanged: `ActivationRecorder`; `residual_intervention` (for the direct-answer-vector control, which *is* a single-direction edit); `score_candidate_sequences` / `prediction_and_margin`; `run_invariance_gate`; `isotropic_random_direction`; `UnitStore` / `RunFingerprint`; `payload_checksum`. Not reused: `residual_intervention`'s single-position hook — the swap needs many positions per layer and a per-layer recomputed coordinate read, so it has its own hook built on the same safety pattern. | — |
| 9 | Every methodological difference between the three interventions | §1 above, plus: A derives its direction from *training examples* and never reads the test activation; the single-vector steer reads neither; B reads **only** the current activation and never any training data. A is antisymmetric under swapping the two concepts, B is symmetric (see §5). A was applied at the final prompt token of one layer; B's faithful protocol is all prompt positions across a band. | — |

---

## 3. Numerical-stability policy

- `V` is assembled in **float64** on CPU and kept there.
- Rank and condition number come from `torch.linalg.svdvals(V)` **before** any
  solve. Numerical rank uses the standard relative test
  `s_min > rank_tolerance · s_max` (default `1e-8`).
- A **rank-deficient** pair (`rank < 2`) is refused: `span(V)` is a line and
  there are no two coordinates to exchange. Regularizing would invent a second
  direction the lens does not have.
- An **ill-conditioned** pair (`cond > max_condition_number`, default `1e4`) is
  refused: the coordinates exist but are dominated by numerical noise. The limit
  is configurable through `StabilityPolicy` and is recorded in the fingerprint,
  so raising it is a visible, bound decision rather than a silent one.
- `c` solves the float64 normal equations `(V^T V) c = V^T h` via
  `torch.linalg.solve`. For rank-2 `V` this equals `pinv(V) h`; the 2×2 Gram
  matrix is the only object whose conditioning matters, and it is exactly what
  the gate measures.
- At the hook boundary the patched activation is cast back to the block
  output's own dtype and device. **The model never sees float64.**

---

## 4. `alpha` semantics

| `alpha` | Meaning |
|---|---|
| `0` | Exact no-op. The full update is computed and multiplied by exactly zero, so the parity condition exercises the identical code path. Bit-exact, asserted with `torch.equal`. |
| `1` | The full measured exchange. Applying it twice returns the original activation. |
| `0 < α < 1` | Interpolation between the two coordinate assignments. |
| `α > 1` | **Amplified extrapolation, not an exchange.** Coordinates land at `c + α(σ(c) − c)`, overshooting `σ(c)`. The paper's "double strength" swap is this case. Every artifact carries `alpha_is_extrapolation`, and it must never be reported as "a stronger swap". |

---

## 5. Token positions and the layer band

**Position rules.** The primary, paper-faithful rule is
`all_prompt_positions` — the paper applies the swap "at every token position
across a band of intermediate layers". For a multimodal prompt this includes
every original prompt/evidence position: image placeholder tokens, audio
placeholder tokens, caption tokens, and question tokens.

| Rule | Use |
|---|---|
| `all_prompt_positions` | **Primary.** Every position `0 ≤ p < prompt_len`. |
| `evidence_span_only` | Position control — only the modality's placeholder span. |
| `non_evidence_prompt_positions` | Position control — the prompt minus that span. |
| `final_prompt_token_only` | **Labelled comparison to the completed pilot only.** Never the primary method. |

**The prompt boundary is enforced for every rule.** Positions `≥ prompt_len` are
teacher-forced candidate-completion tokens appended by
`score_candidate_sequences`, and no configuration can reach them. The hook
records `n_candidate_positions_skipped` on every forward pass, and
`tests/test_coordinate_swap.py` asserts byte-identity of those positions'
activations.

**Layer band.** Contiguous, explicitly configured, and gated: `build_layer_band`
refuses any layer that is not in the recorded validated set. Coordinates are
**recomputed from each layer's own current activation** — one update is never
computed once and replayed, and a test asserts the per-layer pre-swap
coordinates differ.

**Today there is no admissible band.** The completed research-grade calibration
confirmed layers **35, 38 and 40** on its untouched confirmation set — three
isolated layers. Layer 32 and every earlier tested layer **failed**. A
contiguous band over 35..38 is therefore refused today, which is exactly why the
MOCK notebook's switch raises instead of running. Whether an admissible band
exists is what the running earlier-layer calibration decides.

### The exchange is an involution, and a band has to survive that

Two swaps of the same pair cancel exactly. Across a band the swap is recomputed
per layer, so it cancels only to the extent that consecutive layers' coordinates
agree. In the synthetic world the carry blocks nearly commute with the exchange,
and an even-length band is measurably close to a no-op —
`band_parity_diagnostic` reports it rather than hiding it. A real transformer's
blocks do not commute with the exchange, but the real band must still be checked
against this rather than assumed safe.

### Source/target reversal

Exchange is symmetric in its two arguments: `V' = [v_target, v_source]` produces
the **same** `h_patched` for any `α`. What reverses is the bookkeeping — which
coordinate is reported as source. That symmetry is itself a discriminator, since
`h ± α v` is *antisymmetric* under the same relabelling. Consequently
`reverse_swap` is a control over the **evidence** (apply the bird/cat pair to
cat evidence and check the identification moves toward bird), never a control
over the operator.

---

## 5b. The prompt: candidates external to the model

**The primary study requires an open prompt.** The completed steering study
asked *"which one of these is present: bird, cat, giraffe, microwave, toilet,
zebra?"*, which puts what would be the swap **target** into the model's own
input. Identity replacement means the model produces a concept it was never
shown, so a candidate-listed prompt cannot carry the primary claim — under it,
the strongest result available is the candidate-conditioned one the completed
study already has.

`assert_open_prompt_protocol()` enforces it, and refuses unless a prompt
protocol is bound, it is one of `mmpilot.open_animal_identification.v1`,
`mmpilot.open_entity_identification.v1`, `mmpilot.open_animal_legs.v1` or
`mmpilot.hidden_animal_legs.v1`, its candidates were external, and its
registered leakage audit passed. The candidate-listed prompt remains available
as a **labelled comparison condition**, never as the primary. Full
specification: [`prompt_protocol.md`](prompt_protocol.md).

**The planned study is animal-only.** Both of its questions presume an animal,
so every source, target and externally scored identity must carry the
predeclared `animal` domain — `toilet` and `microwave`, which are in the
pilot's six-concept set, are refused — and the leg counts come from a registry
(`bird` 2, `cat` 4) rather than a guess. Its concept set is chosen before any
model result by `select_animal_concepts`, from the existing deterministic
ranking and evidence audit. **General object identification is a separate
protocol**: `mmpilot.open_entity_identification.v1` asks a domain-neutral
question, takes a mixed category set, and supports no legs or multi-hop claim.

For the planned bird → cat example:

| | identity condition | downstream condition |
|---|---|---|
| evidence | bird image, bird caption, or spoken bird caption | *the same evidence* |
| swap | bird → cat | *the identical swap* |
| visible question | `What animal is present in the evidence? Answer with the animal name.` | `How many legs does the animal typically have? Answer with a number.` |
| externally scored | `bird`, `cat`, predeclared controls | `two`, `four`, predeclared controls |

The model never sees a rendered list such as `bird, cat, giraffe, ...`. The swap
stays restricted to original prompt and evidence positions; the teacher-forced
identity and property-answer tokens remain outside the intervention boundary by
`PROMPT_BOUNDARY_RULE`, for every position rule.

The `hidden_animal_legs` protocol — neither entity label nor any registered
alias in any model-visible text, and for spoken audio neither in the offline
transcript — is a **later, stronger stage**. It is not designed yet.

---

## 6. Controls for the future study

| Control | What it rules out |
|---|---|
| `zero` | Harness perturbation. `α = 0` through the full path. |
| `random_two_direction_norm_matched` | Any two directions of the right size would do it. Norms matched exactly; same algebra; same stability gate. |
| `unrelated_pair_swap` | Any coordinate exchange would do it. |
| `reverse_swap` | Direction-specificity, applied to target-carrying evidence. |
| `position_control` | The effect came from positions that carry no evidence. |
| `layer_band_control` | The band's depth did not matter. |
| `raw_residual_difference` | The lens added nothing over a difference in means. *(Existing baseline, method A.)* |
| `source_derived_jspace_steering` | The swap did no better than the completed pilot's steering. *(Existing baseline, method A.)* |
| `direct_answer_vector` | **Essential.** See below. |

**The direct-answer-vector control decides whether "recomputation" is sayable at
all.** If inserting the downstream answer's own lens vector at the same depth —
or an earlier one — moves the answer just as well as swapping the intermediate
entity does, then the downstream change is not evidence that the model
re-derived anything. It is applied through
`jlens.interventions.residual_intervention`, precisely because it is a
single-direction edit and must never be recorded as a coordinate swap. The MOCK
demonstrates it moving the property answer *without* moving the identity, which
is why it has to be reported beside every downstream result.

---

## 7. Fingerprinting and artifact isolation

`coordinate_swap_fingerprint` produces the `intervention_config` that a run
binds every artifact to. It carries: `intervention_family`, coordinate-swap
method version, source and target token ids and strings, vector orientation and
normalization, pseudoinverse/solve policy, condition-number threshold and rank
tolerance, solve dtype, the alpha set, the position rule, the prompt-length
boundary rule, the layer band, lens checksums by layer, model and processor
revisions, the audio protocol fingerprint when `spoken_audio` is enabled, the
full control configuration, and — new — the whole `prompt_protocol` record plus
its version and digest, so *what was asked* is bound beside *what was patched*.
Candidate **order** does not move that digest; the candidate **set** does.

Two independent gates keep the families apart:

1. `RunFingerprint.digest` changes, so `UnitStore.open()` refuses a directory
   written under any other configuration — including every existing
   direction-steering run, none of which names an intervention family.
2. `assert_coordinate_swap_artifacts` turns that digest mismatch into a
   *diagnosis*: it names the steering family explicitly and says why its units
   cannot be relabelled.

**No artifact from a direction-steering run may ever be reused as a
coordinate-swap artifact**, and no completed run is read, rewritten, or
reinterpreted by any code in this patch.

---

## 8. What is claimed

**Claimed:** the implementation computes `h + α V(σ(c) − c)` with
`c = pinv(V) h`; `α = 0` is bit-exact; two swaps cancel; the coordinates are
exchanged; the orthogonal component does not move; the hooks patch every
requested band layer at every requested prompt position and never a
teacher-forced candidate token; the documented refusals fire; storage is atomic
and resume is fingerprint-gated.

**Not claimed, by anything in this repository, under method B:**

- open cross-modal identification;
- cross-modal identity replacement;
- downstream property recomputation rather than a shortcut;
- flexible generalization;
- multi-hop reasoning recomputed consistently;
- any generalization across evidence modalities.

A passing MOCK run is evidence about the implementation and about nothing else.
The synthetic world was built so the algebra comes out right; its shared
cross-modal concept vector is stipulated, not measured.

Behavioral outputs remain **text**. `text`, `image` and `spoken_audio` are
*evidence* modalities. `spoken_audio` means **spoken captions**, not
environmental sound. **Identity replacement and downstream recomputation are
separate claims** and are reported separately.
