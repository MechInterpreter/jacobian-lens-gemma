# Endpoint amendment: native_audio_transfer_three_modality_verdict

`mmpilot.endpoint_semantics_amendment.v1` · checksum `sha256:179c29dd106789f140639e22ae149253c568f9cda37cef37fe7d4d172cbbf64c`

## What this is

This amendment corrects what the completed numbers are called. It does not re-measure, re-derive, promote, demote or replace any of them, and it turns no historical GO into a new numerical verdict.

## Binding

- original report: `runs/mmaudio/mmaudio_native_audio_transfer_20260806T144822/native_audio_transfer_summary.json`
- original report checksum: `sha256:c868999e3d59fba5a44fda9ed5f4815c8f6085432ec902552180f70896665920`
- original run: `mmaudio_native_audio_transfer_20260806T144822`
- original run fingerprint: `sha256:c868999e3d59fba5a44fda9ed5f4815c8f6085432ec902552180f70896665920`
- endpoint audit digest: `sha256:a27a836b4486af7fc30ef4fa16c85529dd93331fdb79ea20fed2a2334fe9cc92`
- claim ledger digest: `sha256:677e31780452a2a43670987e9c2eb68bdddfdd4769a3fffcf7d03da07a093d4e`
- source-unit digest: not reachable from this session (recorded as absent)

## The original verdict, verbatim and unchanged

```
THREE_MODALITY_GO
```

Original endpoint class: `conditional_token_or_sequence_logprob`

## Corrected labels (added beside, not substituted for, the above)

- `CONTROLLED_TARGET_LOGPROB_EFFECT`
- `FULL_VOCABULARY_NOT_EVALUATED`

## Terminology changed

| field | was | is | why |
| --- | --- | --- | --- |
| `capability verdict: per_concept.accuracy` | the model can read the concept out of this channel | in a six-way forced choice whose options are named in the prompt, the concept's complete-sequence likelihood exceeded the other five | the gate is multiple-choice evidence. Open recognition was never measured, and no vocabulary row outside the six was consulted |
| `causal_transfer_verdict.rationale` | moved the target concept in the expected direction | changed the target concept's conditional log-probability in the expected direction [CONTROLLED_TARGET_LOGPROB_EFFECT; FULL_VOCABULARY_NOT_EVALUATED] | a controlled likelihood effect is a real causal result and is not evidence that any output token changed |
| `intervention records: prediction_changed` | prediction_changed | restricted_candidate_preference_changed | the flag records which of the six supplied candidates scored highest, before and after; it is not a change of output |
| `overall_verdict: THREE_MODALITY_GO` | THREE_MODALITY_GO | THREE_MODALITY_GO [CONTROLLED_TARGET_LOGPROB_EFFECT, candidate-conditioned; FULL_VOCABULARY_NOT_EVALUATED] | the verdict string is immutable and is reproduced verbatim; the endpoint labels are added beside it. The existing candidate-conditioned limitation from the prompt-protocol rule still applies and is unchanged by this amendment |

## What did not change

- `scientific_recompute`: **0**
- scientific numbers unchanged: **True**
- original report modified: **False**
- original units modified: **False**
- full vocabulary evaluated: **False** (`FULL_VOCABULARY_NOT_EVALUATED`)
