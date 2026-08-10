# The L27–L31 transition study

`notebooks/research_grade_l27_l31_preconvergence_study_colab.ipynb`, generated
by `scripts/_build_l27_l31_preconvergence_study_notebook.py`.

## The question, and why the interval is closed

Three results already on the record bracket one interval:

| layer | status |
|---|---|
| L26 | failed the frozen J-lens confirmation gate |
| L32 | passed J-lens confirmation at scale 250; classified `AMBIGUOUS` **twice**, independently, under the frozen native direct-readout criterion |
| L35 | `CONVERGED` |

So the open question lives strictly between 26 and 32:

> Does any predeclared layer in **27, 28, 29, 30, 31** simultaneously have
> (1) a rigorously confirmed text-calibrated J-lens, (2) a clearly
> `NOT_CONVERGED` native direct readout across text, image and spoken audio,
> and (3) controlled cross-modal causal transfer?

`jlens.calibration.adjacent.ADJACENT_CANDIDATE_LAYERS` is a frozen tuple bound
into the run fingerprint. There is no L33/L34 clause, no widening after a table
is read, and no replacement layer. A notebook that fits a different set gets a
different fingerprint and cannot resume this one.

## What is reused, and what cannot be

**Reused.** The pinned checkpoint `google/gemma-4-E4B-it` at revision
`fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd`, transformers `5.13.1`, the
text-only calibration protocol, the corpus family and prompt normalization, the
target layer (41), the Jacobian and final-norm conventions, the tie-aware
native-readout validity rule, the open prompt protocol
`mmpilot.open_entity_identification.v1`, the completed studies' intervention
family and controls, and the checkpointed SpokenCOCO preparation cache.

The validity gate is reused by **identity**, not by copy:

```python
ADJACENT_GATE is EXTENSION_GATE                      # True
ADJACENT_CONFIRMATION_GATE is EXTENSION_CONFIRMATION_GATE  # True
```

The pre-download cell asserts both, so there is no copy that could drift.

**Not reusable — the accumulator.** `runs/rgext_*` holds a `jacobian_sum` over
source layers `[8, 14, 20, 26, 32, 35, 38, 40]`. None of 27–31 is in it, and an
accumulator does not acquire a layer it never captured. New Jacobian
accumulation is required, and `assert_new_source_layers` refuses any parent
whose grid overlaps the candidates. The parent run is still read — for its
corpus provenance and its fit ordering — and is checksummed before and after.

**Not reusable — the confirmation set.** The extension's 256-record
confirmation set has been opened, and its verdict is the reason this study
exists. For these five candidates it is development history. A new set is drawn
under a new seed and a new `|adj|` bucket tag, excluding by exact normalized
checksum and by banded SimHash at Hamming distance ≤ 3:

* every fitting prompt (the parent's fit partition, and this study's fit list);
* every development prompt (the parent's validation set, and the extension's);
* every previously opened confirmation prompt (the parent's, and the
  extension's);
* every record in the prior calibration manifests named as dependencies.

`audit_untouched_confirmation` then re-proves it on the *constructed* set by a
different predicate. **If 256 untouched records cannot be built, the study is
blocked** — the size is never reduced, no substitute corpus is reached for, and
no opened set is recycled.

**Reused with its role recorded — development.** The extension's development
partition was designated as development, used as development, and is used as
development again. Its size, checksum and record ids go into the manifest.

## The scale

250, because that is the scale at which L32 was selected and independently
confirmed. **No scale search runs.** A layer confirmed at any other scale would
not be comparable to the result this study is positioned against.

## The hook site

```
output of model.language_model.layers[l]     # post-block residual
final prompt token, where a single position is read
target layer 41
```

## The selection rule, fixed before confirmation opens

Evaluate all five candidates. Select the **lowest** physical layer that passes
every frozen confirmation clause. If none pass, return `ADJACENT_LENS_NO_GO` and
record the complete table anyway. "Closest to passing" is not a rule — every
table has one. The MOCK scenario `earliest_wins` deliberately gives L31 a better
margin than L29 so that an implementation which ranked the passers instead of
ordering them would be caught.

## The stages

| stage | runtime | model | contents |
|---|---|---|---|
| 0 | free CPU | no | corpus provenance, untouched confirmation, SpokenCOCO exclusion harvest, independent population — all checkpointed |
| 1 | L4 | yes | fit L27–L31 at scale 250 in one resumable accumulator |
| 2 | L4 | yes | frozen gate on every candidate; select the earliest passer or stop |
| 3 | L4 | yes | capability and the native direct readout in three modalities |
| 4 | L4/A100 | yes | conditional cross-modal causal transfer with the full control set |

Stage 0 needs a tokenizer (for the token-length filter) but never the weights,
so it stays a free-CPU session.

### Stage 4's gate is an efficiency gate, not a filter

Stage 4 runs only when **all** of: a candidate passed untouched confirmation;
that same layer is `NOT_CONVERGED` in every required modality; every convergence
control passed; capability is sufficient. Under `CONVERGED` the principal claim
is dead at this layer whatever the causal passes show; under `AMBIGUOUS` it is
unsupported whatever they show. The gate therefore withholds passes exactly
where the headline result is *unfavourable* to the hypothesis, and those
Stage-3 outcomes are the study's primary reported verdicts.

`RUN_STAGE_4_CAUSAL_TRANSFER` remains an explicit override. An overridden run is
stamped `gate_overridden: true` and `DESCRIPTIVE_ONLY` in every artifact, and its
numbers never support the principal claim.

## The three populations this study must avoid

* `mmaudio_native_audio_transfer_20260806T144822`
* `mml32_l32_followup_20260808T182717`
* the completed `mml32res_*` convergence-resolution run — **pinned by hand**

The third is not a constant and is never discovered. Several `mml32res_*`
directories can exist on a Drive (a preprocessing session, an abandoned attempt,
the completed study); only the operator knows which is which, and picking "the
newest" would treat a spent population as available.
`assert_completed_population_pins` refuses an unset, wrongly-prefixed, or
un-excluded pin.

Disjointness is proved over `image_id`, `group_id`, audio/recording path,
caption identity, and media checksum where available.

## The prompt

`mmpilot.open_entity_identification.v1`. The model-visible prompt names no
candidate; the six candidates exist only in the external scorer. The
candidate-visibility audit is run and persisted **separately for text, image and
spoken_audio**, because the surfaces differ by modality. A leaked candidate in
any modality is a hard refusal.

## Verdicts

```
ADJACENT_LENS_VALIDITY
EARLIEST_CONFIRMED_LAYER
NATIVE_OUTPUT_CONVERGENCE
THREE_MODALITY_CAUSAL_TRANSFER
PRECONVERGENCE_CAUSAL_TRANSFER
```

Terminal outcomes, all first-class and all reported in the same words:

```
PRECONVERGENCE_CAUSAL_TRANSFER_SUPPORTED
ADJACENT_LENS_NO_GO
CONVERGED_BEFORE_CAUSAL_TEST
AMBIGUOUS_CONVERGENCE
CAUSAL_TRANSFER_NOT_SUPPORTED
REFUSED_INVALID
```

The principal success verdict requires **one and the same physical layer** to
have an untouched confirmed J-lens, a `NOT_CONVERGED` native direct readout, and
controlled cross-modal causal transfer — all three **on the same independent
multimodal population**. `assert_same_population` makes that mechanical:
convergence measured on one population is never paired with a causal effect
measured on another.

## Required causal controls

```
matched_random_direction
external_unrelated_concept
shuffled_permuted_control
zero_intervention
activation_norm_sanity
target_specificity_global_disruption
image_level_independent_aggregation
```

A missing control record is a failure, never a pass.

## Not in scope

* An **Anthropic two-coordinate swap**. `jlens/mmpilot/coordinate_swap.py`
  implements it; it needs a *contiguous* confirmed layer band, which today's
  confirmed set does not provide. The intervention here is additive J-space
  residual steering — the same family the open-prompt L32 follow-up used, reused
  unchanged for comparability — and nothing in this study is described as a swap.
* Environmental (non-speech) audio.
* Republishing or modifying any completed run or published lens.

## Resumability

| stage | unit | what an interruption costs |
|---|---|---|
| 0 | one shard of ≤25 files or ≤30 s | that batch |
| 1 | the accumulator, written every 25 prompts | that batch |
| 2 | one stored verdict per stage | that stage |
| 3 | one behavioural unit / one readout unit | that unit |
| 4 | one intervention unit | that unit |

Every fingerprint binds the model revision, the layer set, the corpus manifest,
the prompt protocol, the target layer, `skip_first`, the dtype and the fitting
scale. A changed configuration **refuses** the resume rather than mixing
checkpoints. `tests/test_preconvergence_notebook.py` interrupts the notebook at
each of the five stages and asserts the resumed run reaches the same conclusion
without recomputing anything durable.

## Engineering: why the real branch is executed in CI

Three L4 starts have died on defects that only exist on a real-only branch.
Two mechanisms address that here:

* `check_preconvergence_call_contracts()` binds **65** real-branch call sites
  against the installed signatures, in milliseconds, before the ~16 GB
  download. The notebook's pre-download cell runs it alongside the shared
  `check_call_contracts()` and refuses the load on any failure.
* `tests/test_preconvergence_real_path.py` **executes the notebook's real
  model-loading cell verbatim**, monkeypatching only the model loader and the
  processor factory. Everything else is production code:
  `build_real_backend`'s unpacking, the parameter freezing, the architecture
  audit, the real `resolve_audio_interface` probe, the real
  `GemmaPilotBackend` constructor, every bundle attribute the cell reads, and
  the real `assert_audio_protocol` validation. A second test resolves every
  `from jlens... import name` in every cell.

The two symbols that do not exist — `jlens.mmpilot.preflight.preflight` and
`jlens.mmpilot.real_backend.load_real_bundle` — are asserted absent from both
the module and the notebook.
