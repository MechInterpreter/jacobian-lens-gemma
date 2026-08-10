# Output-convergence timing: does the L35 transfer precede the answer?

A follow-up to the completed three-modality causal-transfer run
`mmaudio_native_audio_transfer_20260806T144822` (`THREE_MODALITY_GO`). It reruns
nothing from that study and modifies nothing in it.

- Code: [`jlens/mmpilot/convergence.py`](../jlens/mmpilot/convergence.py),
  [`jlens/mmpilot/convergence_mock.py`](../jlens/mmpilot/convergence_mock.py)
- Notebook:
  [`notebooks/archive/completed_studies/multimodal_jspace_output_convergence_audit_colab.ipynb`](../notebooks/archive/completed_studies/multimodal_jspace_output_convergence_audit_colab.ipynb)
- Generator: `scripts/_build_convergence_notebook.py`
- Tests: `tests/test_mmpilot_convergence.py`,
  `tests/test_mmpilot_convergence_notebook.py`

## The gap this closes

The completed run edited the residual at the final prompt token of layer 35 and
measured the behavioral answer. That establishes controlled cross-modal causal
transfer. It does not establish that layer 35 *precedes* the answer, and
[`jlens/mmpilot/tri_modal.py`](../jlens/mmpilot/tri_modal.py) says so in its own
docstring — layer 35 is primary because it is the earliest independently
confirmed lens, "not 'pre-language', not 'pre-convergence' ... convergence timing
is unresolved here and every artifact says so."

This audit is that independent test, and only that.

## Exact operational definition of answer convergence

For a stored clean final-prompt-token residual `h` at an audited layer:

```
logits = lm_head(final_norm(h))            # the live modules, called
logits = softcap * tanh(logits / softcap)  # if the config declares a softcap
```

restricted to the six fixed behavioral answer candidates. From that restricted
distribution the audit records, per sample:

| quantity | meaning |
| --- | --- |
| `clean_agreement_unique` | the target is the **sole** maximum among the six *and* equals the model's own clean final answer — the primary metric |
| `clean_agreement_argmax` | the same under argmax, with ties resolved in the layer's favour |
| `target_accuracy_*` | the same two, against the scored label |
| `target_rank` | midrank among the six, tie-aware (`optimistic` / `pessimistic` / `midrank`, matching `jlens/mmlocalize/lens_validity.py`) |
| `target_margin` | target logit minus the best non-target logit |
| `candidate_entropy_nats` | entropy of the restricted softmax, and its `/ ln(6)` normalization |
| `top_two_margin` | gap between the best and second-best candidate |

"Converged" is then a property of a `(layer, modality)` cell, not of a sample.

## Why this is independent of the J-space causal result

The causal result is about a direction **estimated in J-space from one
modality** moving a behavioral answer in another. This measurement never touches
the lens, the dictionary, the J-space codes, the concept directions, or any
intervention. It reads the same stored activations through a different,
model-owned map — the model's own output head.

The two can therefore disagree, and disagreement is the informative outcome.
Nothing in the causal pipeline is recomputed: the causal verdicts are **read**
from the completed run's capability-filtered summary
(`native_audio_transfer_summary_capability_filtered_v2.json`).

## Exact Gemma normalization and unembedding convention

The readout is `lm_head(final_norm(h))`, then `30 * tanh(x / 30)` where the text
config declares `final_logit_softcapping`. The softcap is strictly monotonic, so
it cannot change a ranking; it does change logit values, margins and entropies,
and the audited path applies it **because the model's output head does**.

Two things are established rather than assumed, by `audit_native_head`:

1. **The normalization convention.** Gemma's RMSNorm applies `x_hat * (1 + w)`,
   not `x_hat * w`. A hand-rolled "standard logit lens" gets this silently
   wrong. The audit pushes probes through the live module, compares the output
   against both conventions, and records which one matches — it never
   reimplements the norm. A `LayerNorm` (the CPU mock) is recorded as
   `not_rmsnorm` without failing.
2. **Agreement with the model's own `unembed`.** The audited path is compared
   against `jlens.hf.HFLensModel.unembed` on the same probes. A disagreement
   raises `ConvergenceRefused` and stops the audit: reading the model through a
   head the model does not use is not a small numerical difference.

The stored activations are **block outputs** (`model.language_model.layers[l]`,
pre-final-norm), which is exactly the site `final_norm` expects. This is not the
`last_hidden_state` trap — that tensor is already normalized and must never be
passed to `unembed`.

## Candidate tokenization

`resolve_candidate_tokens` decides the mode from the completed run's recorded
`candidate_token_ids` and refuses rather than downgrading silently:

- **all single-token** → `single_token_complete`. The next-token distribution
  *is* a complete score for each candidate, directly comparable with the run's
  teacher-forced sequence scores.
- **any multi-token** → `first_token_only_diagnostic`, labelled as such on every
  row and in every artifact. A single hidden state does not yield a complete
  teacher-forced sequence score, so no row here is one. The run's own clean
  predictions remain complete sequence scores and are used **only** as the
  reference answer — never mixed into a readout total.
- **two candidates sharing a first token** → `CandidateTokenizationError`. Two
  candidates competing for one logit cannot be ranked against each other at all,
  so no comparison is valid.

Whether the real six candidates are all single-token is a fact about the
tokenizer that the notebook prints at run time from the completed run's stored
`candidate_token_ids`. It has **not** been resolved here: this work did not run
the real audit, and guessing it would be exactly the kind of assumption the
check exists to prevent.

## The criterion is two-sided

There is an obvious way to cheat this audit: set the "converged" bar high, watch
layer 35 fail it, and call the failure evidence of pre-convergence. A single
threshold makes "not yet converged" the default state, and the default state is
the one the interesting verdict needs.

So `ConvergenceCriterion` declares two bars with a gap between them.

**CONVERGED** — in *every* one of text, image and spoken audio, over
capability-admissible concepts:

- `clean_agreement_unique >= 0.90` — at 90% the answer is effectively determined
  at that layer;
- `target_accuracy_unique >= 0.90` — a readout that agrees with the model only
  where the model is wrong has converged on an error mode, not an answer;
- `median midrank <= 1.0`.

**NOT_CONVERGED** — in every one of those modalities, scored with the *generous*
argmax rule:

- `clean_agreement_argmax <= 0.50` — three times the six-candidate chance rate
  (0.167), and still a disagreement with the model's own answer on at least half
  the samples;
- `target_accuracy_argmax <= 0.50`;
- `median midrank >= 2.0`.

The strictness is deliberately mismatched: convergence is scored strictly (a tie
is not convergence) and non-convergence generously (a layer must look weak even
when ties break its way). Anything between the bars is `AMBIGUOUS`.

Fixed alternative thresholds are reported for sensitivity
(`SENSITIVITY_VARIANTS`) and cannot change the primary rule.

## Verdicts

`TRANSFER_AT_OR_AFTER_CONVERGENCE` is checked **first**, so the unfavourable
outcome can never be masked by a later clause failing.

| verdict | when |
| --- | --- |
| `TRANSFER_AT_OR_AFTER_CONVERGENCE` | layer 35 is `CONVERGED` |
| `PRE_CONVERGENCE_TRANSFER_SUPPORTED` | layer 35 is `NOT_CONVERGED`, **and** its capability-filtered causal verdict is still `SUPPORTED`, **and** a later validated layer is clearly more converged (≥ 0.20 higher with a disjoint image-level bootstrap interval), **and** the trajectory is monotone within tolerance, **and** the readout is non-degenerate, **and** every control and integrity check holds |
| `INCONCLUSIVE_CONVERGENCE_TIMING` | anything else |

The guards that force `INCONCLUSIVE` are as important as the criterion:

- fewer than 4 samples in a `(layer, modality)` cell;
- fewer than 2 distinct predicted candidates at layer 35 — **a readout that
  answers the same word every time has failed, and a failed readout is not a
  fact about the representation**;
- a non-monotonic trajectory (no direction to read);
- a control that reproduces the primary result;
- a candidate set that cannot support a valid comparison.

The MOCK world's `flat_weak` mode exists specifically to prove the second and
third of these: every layer weak, so layer 35 *is* `NOT_CONVERGED` — and the
verdict still refuses, because nothing shows the answer arriving later.

## Controls

| control | what changes | scored on |
| --- | --- | --- |
| `shuffled_target_labels` | the label each sample is scored against | target accuracy (it cannot move agreement with the model's answer, and the artifact says so) |
| `permuted_candidate_tokens` | which token supplies each candidate's logit (a derangement) | agreement with the model's clean answer |
| `permuted_activations` | which sample's residual the head is applied to | agreement with the model's clean answer |

A control **fails by reproducing** a primary result that is itself above chance.
Where the primary readout sits at the `1/n_candidates` floor, no result rests on
that cell and the control has nothing to reproduce — requiring it to fall *below*
chance would be requiring something no permutation can do. Such cells are
recorded as `primary_is_informative: false` rather than scored as passes on the
merits.

**There is no literal wrong-layer control.** Gemma has exactly one final
normalization module and one unembedding, shared by every layer, so there is no
layer-specific readout component to misapply. `WRONG_LAYER_CONTROL_NOTE` states
this in every artifact, and the permuted-activation control is its substitute.

An optional ridge probe with image-disjoint folds is available as a **secondary
diagnostic** (`RUN_SECONDARY_PROBE`). It carries `determines_verdict: False` and
no verdict clause reads it. A probe that decodes the concept where the native
readout does not is consistent with the audit's finding, not contrary to it —
the audit measures convergence onto the model's own output, not decodability.

## Statistical unit

The photograph, as the completed run established. `bootstrap_rate` resamples
distinct `image_id`s with replacement and carries every row of a drawn unit
along, so repeated captions of one image cannot inflate `n`. Deterministic in a
fixed seed. SpokenCOCO carries one recording per synchronized group, so
`recording_id` is recorded explicitly and equals the group.

## Capability filtering

Every principal number is computed over capability-admissible concepts only, via
`jlens.mmpilot.admissibility.concept_admissibility` — the completed run's single
authority for that decision. `zebra` failed the spoken-audio gate and is
retained as explicitly labelled descriptive data
(`descriptive_inadmissible`), entering no principal number and no verdict clause.
Per-concept results are reported **before** any pooled result.

## The interpretation boundary

> A weak native direct readout means the final-prompt-token residual has not
> converged onto the model's answer under the predeclared criterion. It is **not**
> proof that linguistic information is absent: no claim is made, and none may be
> derived, about what a nonlinear decoder or a trained probe could recover from
> the same activation.

Permitted phrasing: "before native direct-readout convergence", "before obvious
output-answer convergence under the predeclared criterion". Never
"pre-linguistic". Never "language-free". `INTERPRETATION_BOUNDARY` is stamped on
every row, every summary and every verdict.

## Safety, provenance and immutability

- The completed run is opened by name — never discovered, never created.
- Stage 1 verifies the run fingerprint digest, model repo and revision,
  processor revision, audio-protocol version and fingerprint, all three lens
  checksums plus the combined one, and the presence of the capability-filtered
  verdicts. The first mismatch raises `ConvergenceRefused`.
- Layer 32 raises `LensInvalidLayerError` anywhere it is asked to be
  interpreted.
- The protected files are checksummed before and after; `assert_run_unchanged`
  raises `CompletedRunModified` on any difference, including a file that
  appeared.
- The audit writes only to a new directory outside the run, and refuses an audit
  directory nested inside it.
- Resume is per unit, checksum-validated. The audit fingerprint binds the
  completed run, model revision, candidate digest, layers, readout mode, head
  checksum, criterion digest and code version; an incompatible resume is refused
  with the differing field named.

## Artifacts

Written to `<AUDIT_ROOT>/<AUDIT_NAME>/`:

```
output_convergence_report.md
output_convergence_summary.json
per_sample_direct_readout.jsonl
layer_convergence_table.csv
provenance.json
checksums.json
figure_convergence_versus_layer.svg
figure_causal_versus_convergence.svg
figure_per_modality_trajectories.svg
fingerprint.json
readout_units/*.json
```

Figures are emitted as deterministic hand-written SVG. The repository has no
plotting dependency, and a figure whose bytes are a pure function of the numbers
is one a test can pin.

## Running it

```bash
python scripts/_build_convergence_notebook.py
```

Open the notebook in Colab. The committed defaults run the MOCK world and spend
nothing. For the real audit, set in section 2:

```
RUN_REAL_CONVERGENCE_AUDIT      = True
CONFIRM_MODEL_LOAD              = True
EXPECTED_RUN_FINGERPRINT_DIGEST = <from the run's run_manifest.json>
EXPECTED_PROCESSOR_REVISION     = <from the run's fingerprint.json>
```

The last two are deliberately empty. Section 6b prints what the run records so
you can compare, and says outright that **printing is not verifying** — a value
copied out of the run and pasted back proves nothing. Paste them from your own
record of the run.

### Cost

The audit executes **zero model forward passes**. Gemma is loaded once
(~16 GB download) purely to obtain the final norm and the unembedding, which are
then called on stored activations. Runtime after the load is a few minutes on
CPU or GPU; a T4 is sufficient, and the load is the whole cost. Drive footprint
is a few MB — one small JSON per scored unit plus the aggregate artifacts.

## MOCK mode

`jlens.mmpilot.convergence_mock` builds a synthetic completed run with the same
`UnitStore` layout, artifact names and verdict shapes, plus a tiny output head
using Gemma's `(1 + weight)` RMSNorm convention. Signal strength per layer is
the only knob, so every verdict branch is reachable:

| mode | verdict |
| --- | --- |
| `pre_convergence` | `PRE_CONVERGENCE_TRANSFER_SUPPORTED` |
| `converged_early` | `TRANSFER_AT_OR_AFTER_CONVERGENCE` |
| `ambiguous` | `INCONCLUSIVE_CONVERGENCE_TIMING` |
| `flat_weak` | `INCONCLUSIVE_CONVERGENCE_TIMING` |
| `degenerate` | `INCONCLUSIVE_CONVERGENCE_TIMING` |
| `pre_convergence` with `causal_supported=False` | `INCONCLUSIVE_CONVERGENCE_TIMING` |

**A passing MOCK run is plumbing evidence.** The trajectory is a knob, so a
`PRE_CONVERGENCE_TRANSFER_SUPPORTED` there says only that the code detects a
separation put in on purpose. It is never evidence about Gemma.
