# Endpoint-semantics audit

`mmpilot.endpoint_semantics_audit.v1` · audit digest `sha256:a27a836b4486af7fc30ef4fa16c85529dd93331fdb79ea20fed2a2334fe9cc92`

Every row below was classified by reading the function that computes
the report field, not the report's prose. No completed report, unit or
run directory was modified; `scientific_recompute = 0`.

## Endpoint classes

| class | claims |
| --- | ---: |
| `unrestricted_full_vocabulary_next_token` | 7 |
| `restricted_candidate_rank` | 5 |
| `conditional_token_or_sequence_logprob` | 6 |
| `free_or_greedy_generation` | 1 |
| `activation_or_jspace_representational` | 2 |
| `engineering_or_input_path_validation` | 3 |

## Survival

| survival | claims |
| --- | ---: |
| `survives_unchanged` | 13 |
| `survives_with_narrower_wording` | 11 |
| `unsupported` | 0 |

## Claim ledger

| claim | run | computing function | endpoint | candidates | appended | global vocab | generated | revalidate | survival |
| --- | --- | --- | --- | --- | :-: | :-: | :-: | :-: | --- |
| spoken audio reaches the model through its native pathway | — | `jlens.mmpilot.audio_audit.run_audio_audit` | `engineering_or_input_path_validation` | not applicable | no | no | no | no | `survives_unchanged` |
| a candidate scoring path is valid for the audio condition | — | `jlens.mmpilot.audio_audit.check_scoring_validity` | `engineering_or_input_path_validation` | predeclared audit candidates | yes | no | no | no | `survives_unchanged` |
| the model can read each concept out of each channel | mmaudio_native_audio_transfer_20260806T144822 | `jlens.mmpilot.capability.prediction_and_margin` | `restricted_candidate_rank` | six concepts, all named in the prompt | yes | no | no | no | `survives_with_narrower_wording` |
| capability accuracy per (concept, modality) clears 70% | mmaudio_native_audio_transfer_20260806T144822 | `jlens.mmpilot.capability.capability_summary` | `restricted_candidate_rank` | six concepts, all named in the prompt | yes | no | no | no | `survives_with_narrower_wording` |
| the J-lens reproduces the model's own next token | — | `jlens.calibration.gate.ordinary_next_token_argmax` | `unrestricted_full_vocabulary_next_token` | the entire vocabulary | no | yes | no | no | `survives_unchanged` |
| lens rank of the model's true next token beats its controls | — | `jlens.calibration.gate.evaluate_calibration_layer` | `unrestricted_full_vocabulary_next_token` | the entire vocabulary | no | yes | no | no | `survives_unchanged` |
| the published lens confirms on held-out prompts | — | `jlens.native_readout.evaluate_confirmatory` | `unrestricted_full_vocabulary_next_token` | the entire vocabulary | no | yes | no | no | `survives_unchanged` |
| J-space codes retrieve the matching cross-modal example | mmaudio_native_audio_transfer_20260806T144822 | `jlens.mmpilot.jspace.retrieval_metrics` | `activation_or_jspace_representational` | the retrieval gallery of held-out examples | no | no | no | no | `survives_unchanged` |
| retrieval beats its shuffled-label control | mmaudio_native_audio_transfer_20260806T144822 | `jlens.mmpilot.jspace.shuffled_label_control` | `activation_or_jspace_representational` | the retrieval gallery under permuted labels | no | no | no | no | `survives_unchanged` |
| the multimodal pilot's causal cells moved the target | — | `jlens.mmpilot.causal.run_condition` | `conditional_token_or_sequence_logprob` | the predeclared concept candidates | yes | no | no | yes | `survives_with_narrower_wording` |
| the pilot's go/no-go criteria | — | `jlens.mmpilot.report.evaluate_criteria` | `conditional_token_or_sequence_logprob` | the predeclared concept candidates | yes | no | no | yes | `survives_with_narrower_wording` |
| text<->image transfer replicates on image-unique data | — | `jlens.mmpilot.robustness.evaluate_causal_cells` | `conditional_token_or_sequence_logprob` | the predeclared concept candidates | yes | no | no | yes | `survives_with_narrower_wording` |
| the causal effect is localized to particular layers | — | `jlens.mmlocalize.verdict.localization_verdict` | `conditional_token_or_sequence_logprob` | the predeclared concept candidates | yes | no | no | yes | `survives_with_narrower_wording` |
| THREE_MODALITY_GO — cross-modal causal transfer including speech | mmaudio_native_audio_transfer_20260806T144822 | `jlens.mmpilot.tri_modal.causal_transfer_verdict` | `conditional_token_or_sequence_logprob` | six concepts, all named in the prompt | yes | no | no | yes | `survives_with_narrower_wording` |
| the capability-filtered v2 amendment's verdicts | mmaudio_native_audio_transfer_20260806T144822 | `jlens.mmpilot.amend.rebuild_verdicts` | `conditional_token_or_sequence_logprob` | six concepts, all named in the prompt | yes | no | no | yes | `survives_with_narrower_wording` |
| hidden-intermediate onset is localized among confirmed layers | — | `jlens.mmpilot.paper_reasoning_swap.paper_onset_verdict_v2` | `restricted_candidate_rank` | two answers (two/four) or two identities (bird/cat) | yes | no | no | yes | `survives_with_narrower_wording` |
| target answer becomes top-1 after the band swap | band3340_real_2a72bda9b4ba | `jlens.mmpilot.band_swap.band_trial_record` | `restricted_candidate_rank` | exactly two answers per readout | yes | no | no | yes | `survives_with_narrower_wording` |
| L33_L40_VALIDATED_BAND_FOLLOWUP_ALPHA2_SENSITIVITY_ONLY | band3340_real_2a72bda9b4ba | `jlens.mmpilot.band_swap.band_reasoning_verdict` | `restricted_candidate_rank` | exactly two answers per readout | yes | no | no | yes | `survives_with_narrower_wording` |
| every physical band layer was patched on every trial | band3340_real_2a72bda9b4ba | `jlens.mmpilot.validated_band_followup.assert_band_hook_integrity` | `engineering_or_input_path_validation` | not applicable | no | no | no | no | `survives_unchanged` |
| L33-L40 is the admissible band; L32 is excluded | bandcorr_real_eb5b00f135e4 | `jlens.mmpilot.band_control.corrected_band_verdict` | `unrestricted_full_vocabulary_next_token` | the entire vocabulary | no | yes | no | no | `survives_unchanged` |
| the layer has/has not converged to its output | — | `jlens.mmpilot.convergence.direct_readout_row` | `unrestricted_full_vocabulary_next_token` | the entire vocabulary | no | yes | no | no | `survives_unchanged` |
| convergence classification per layer | — | `jlens.mmpilot.convergence.classify_layer` | `unrestricted_full_vocabulary_next_token` | the entire vocabulary | no | yes | no | no | `survives_unchanged` |
| the target token is the global next-token argmax | — | `jlens.mmpilot.full_vocabulary.score_unrestricted_next_token` | `unrestricted_full_vocabulary_next_token` | the entire vocabulary; no candidate list is supplied | no | yes | no | no | `survives_unchanged` |
| the model greedily writes the target answer | — | `jlens.mmpilot.full_vocabulary.greedy_generate` | `free_or_greedy_generation` | not applicable | no | yes | yes | no | `survives_unchanged` |

## Prohibited wording, per claim

**spoken audio reaches the model through its native pathway**

- justified: the processor expands a recording into placeholder tokens the model consumes, and no transcript is visible to the model
- prohibited: *the model understood the speech*
- note: An input-path property. Unaffected by any endpoint question.

**a candidate scoring path is valid for the audio condition**

- justified: the teacher-forced sum equals its own per-token terms, so the scorer is internally consistent
- prohibited: *the model answered correctly*
- note: Validates the arithmetic of a scorer, not a scientific outcome.

**the model can read each concept out of each channel**

- justified: in a six-way forced choice whose options are listed in the prompt, the target concept's complete-sequence likelihood exceeded the other five
- prohibited: *the model identified the concept*
- prohibited: *the model answered*
- prohibited: *the model answered*
- prohibited: *the model output*
- prohibited: *global top-1*
- prohibited: *full-vocabulary*
- prohibited: *paper-comparable top-1*
- note: Narrower multiple-choice evidence. It gates admissibility and was never claimed as output; the wording 'can read out' is tightened to 'can select in a six-way forced choice'.

**capability accuracy per (concept, modality) clears 70%**

- justified: forced-choice accuracy over 8 samples per cell against a frozen threshold
- prohibited: *the model answered*
- prohibited: *the model output*
- prohibited: *global top-1*
- prohibited: *full-vocabulary*
- prohibited: *paper-comparable top-1*

**J-space codes retrieve the matching cross-modal example**

- justified: activations of one modality are nearer their same-concept counterparts in another modality than a shuffled assignment is
- prohibited: *the model output the concept*
- prohibited: *the model answered*
- prohibited: *the model output*
- prohibited: *global top-1*
- prohibited: *full-vocabulary*
- prohibited: *paper-comparable top-1*
- note: Representational evidence. It never was an output claim, so the endpoint correction does not touch it.

**retrieval beats its shuffled-label control**

- justified: the retrieval statistic exceeds the 95th percentile of the same statistic under permuted labels
- prohibited: *the model answered*
- prohibited: *the model output*
- prohibited: *global top-1*
- prohibited: *full-vocabulary*
- prohibited: *paper-comparable top-1*

**the multimodal pilot's causal cells moved the target**

- justified: the target answer's teacher-forced conditional log-probability moved in the intended direction relative to the clean run
- prohibited: *the intervention changed the model's answer*
- prohibited: *the model answered*
- prohibited: *the model output*
- prohibited: *global top-1*
- prohibited: *full-vocabulary*
- prohibited: *paper-comparable top-1*
- note: `prediction_changed` is a restricted-candidate flip, not an output change. Genuine conditional-likelihood effect; not autonomous output.

**the pilot's go/no-go criteria**

- justified: aggregate conditional-log-probability effects and their controls cleared frozen thresholds
- prohibited: *the model answered*
- prohibited: *the model output*
- prohibited: *global top-1*
- prohibited: *full-vocabulary*
- prohibited: *paper-comparable top-1*

**text<->image transfer replicates on image-unique data**

- justified: controlled conditional-log-probability effects replicate on a population with one group per image
- prohibited: *the model answered*
- prohibited: *the model output*
- prohibited: *global top-1*
- prohibited: *full-vocabulary*
- prohibited: *paper-comparable top-1*

**the causal effect is localized to particular layers**

- justified: the size of the conditional-log-probability effect differs between layers on paired inputs
- prohibited: *the model answered*
- prohibited: *the model output*
- prohibited: *global top-1*
- prohibited: *full-vocabulary*
- prohibited: *paper-comparable top-1*

**THREE_MODALITY_GO — cross-modal causal transfer including speech**

- justified: a source-derived J-space direction applied at one layer changed the target concept's conditional log-probability in the intended direction, against matched random, unrelated, zero and raw-residual controls, in both directions for at least one admissible concept
- prohibited: *the model said the swapped concept*
- prohibited: *the model's answer changed*
- prohibited: *the model answered*
- prohibited: *the model output*
- prohibited: *global top-1*
- prohibited: *full-vocabulary*
- prohibited: *paper-comparable top-1*
- note: A genuine controlled conditional-likelihood effect. It is not evidence that any output token changed: the full vocabulary was never consulted. Corrected label: CONTROLLED_TARGET_LOGPROB_EFFECT plus FULL_VOCABULARY_NOT_EVALUATED.

**the capability-filtered v2 amendment's verdicts**

- justified: the same conditional-log-probability effects, re-read under the capability-admissibility rule; no measurement was recomputed
- prohibited: *the model answered*
- prohibited: *the model output*
- prohibited: *global top-1*
- prohibited: *full-vocabulary*
- prohibited: *paper-comparable top-1*

**hidden-intermediate onset is localized among confirmed layers**

- justified: after the exchange the target answer outranked the one other supplied candidate more often than every matched control did
- prohibited: *the model output the target answer*
- prohibited: *paper-comparable top-1*
- prohibited: *the model answered*
- prohibited: *the model output*
- prohibited: *global top-1*
- prohibited: *full-vocabulary*
- prohibited: *paper-comparable top-1*

**target answer becomes top-1 after the band swap**

- justified: `target_rank` is a rank among two supplied candidates; `prediction` is which of the two scored higher
- prohibited: *the target token became the model's top output*
- prohibited: *the model answered*
- prohibited: *the model output*
- prohibited: *global top-1*
- prohibited: *full-vocabulary*
- prohibited: *paper-comparable top-1*
- note: The completed report measured whether `two` outranked `four`, not whether `two` was the global next-token argmax.

**L33_L40_VALIDATED_BAND_FOLLOWUP_ALPHA2_SENSITIVITY_ONLY**

- justified: at alpha=2 only, the swap made the target answer the preferred one of two supplied candidates more often than every alpha=2-matched control; alpha=1 did not
- prohibited: *paper-comparable*
- prohibited: *Anthropic's top-1 endpoint*
- prohibited: *the model answered*
- prohibited: *the model output*
- prohibited: *global top-1*
- prohibited: *full-vocabulary*
- prohibited: *paper-comparable top-1*
- note: Corrected label: RESTRICTED_CANDIDATE_PREFERENCE_GO at alpha=2 sensitivity only, plus FULL_VOCABULARY_NOT_EVALUATED. The numbers are unchanged; the `paper_comparable` field name is superseded by `restricted_candidate_preference`.

**the model greedily writes the target answer**

- justified: a deterministic temperature-0 continuation of the prompt under the same intervention
- prohibited: *greedy text alone establishes the causal claim*
- note: Secondary demonstration. No verdict may rest on it.

## Active-source scan

- files scanned: 13
- prohibited-phrase hits: 6
- unqualified overclaims: 0
- passed: **True**
