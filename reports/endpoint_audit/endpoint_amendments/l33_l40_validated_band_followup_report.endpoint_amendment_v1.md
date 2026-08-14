# Endpoint amendment: l33_l40_validated_band_followup_report

`mmpilot.endpoint_semantics_amendment.v1` · checksum `sha256:c8e597ba36ce64ece1a91f2afc56214e40740d145768665cf5bf9543cb52875e`

## What this is

This amendment corrects what the completed numbers are called. It does not re-measure, re-derive, promote, demote or replace any of them, and it turns no historical GO into a new numerical verdict.

## Binding

- original report: `runs/mmband33/band3340_real_2a72bda9b4ba/l33_l40_validated_band_followup_report.json`
- original report checksum: `sha256:f808ac89236c640269698d18c999412e0164533349b69a4d9960cdcc1ce263cb`
- original run: `band3340_real_2a72bda9b4ba`
- original run fingerprint: `sha256:2a72bda9b4bad352d93e387ba7d1dd109b3e7f7d6a14093638e4c3a7ee1c412e`
- endpoint audit digest: `sha256:a27a836b4486af7fc30ef4fa16c85529dd93331fdb79ea20fed2a2334fe9cc92`
- claim ledger digest: `sha256:677e31780452a2a43670987e9c2eb68bdddfdd4769a3fffcf7d03da07a093d4e`
- source-unit digest: not reachable from this session (recorded as absent)

## The original verdict, verbatim and unchanged

```
L33_L40_VALIDATED_BAND_FOLLOWUP_ALPHA2_SENSITIVITY_ONLY
```

Original endpoint class: `restricted_candidate_rank`

## Corrected labels (added beside, not substituted for, the above)

- `RESTRICTED_CANDIDATE_PREFERENCE_GO`
- `FULL_VOCABULARY_NOT_EVALUATED`

## Terminology changed

| field | was | is | why |
| --- | --- | --- | --- |
| `reasoning_verdict.paper_comparable` | paper_comparable | restricted_candidate_preference | the field reports an argmax over two supplied candidates. Anthropic's trial definition inspects the complete next-token distribution and counts success only at global rank 1, so the two are not comparable and the name asserted that they were |
| `reasoning_verdict.paper_comparable.criterion` | Anthropic's top-1 trial definition at alpha=1 | restricted-candidate preference at alpha=1: the target answer outranks the other predeclared candidate by teacher-forced conditional sequence likelihood | the criterion never consulted any vocabulary row outside the two supplied answers |
| `reasoning_verdict.primary_endpoint` | fraction of trials in which the target-appropriate downstream answer is top-1 of the externally scored candidate set | fraction of trials in which the target-appropriate downstream answer is top-1 of the externally scored candidate set [restricted-candidate endpoint; the full vocabulary was not evaluated] | the wording was already accurate but was read as a global top-1 rate; the qualifier is now explicit |
| `followup_verdict.verdict` | L33_L40_VALIDATED_BAND_FOLLOWUP_ALPHA2_SENSITIVITY_ONLY | L33_L40_VALIDATED_BAND_FOLLOWUP_ALPHA2_SENSITIVITY_ONLY [RESTRICTED_CANDIDATE_PREFERENCE_GO at alpha=2 sensitivity only; FULL_VOCABULARY_NOT_EVALUATED] | the verdict string is immutable and is reproduced verbatim; the endpoint labels are added beside it, not substituted for it |
| `trial records: target_rank` | target_rank | restricted_candidate_target_rank (out of 2) | `target_rank` was a rank among the two scored candidates, not a rank in the vocabulary |

## What did not change

- `scientific_recompute`: **0**
- scientific numbers unchanged: **True**
- original report modified: **False**
- original units modified: **False**
- full vocabulary evaluated: **False** (`FULL_VOCABULARY_NOT_EVALUATED`)
