# Capability admissibility in the three-modality study

## What went wrong

The native spoken-audio Stage A and Stage B completed under run fingerprint

```
sha256:c868999e3d59fba5a44fda9ed5f4815c8f6085432ec902552180f70896665920
```

Stage A measured behavioral capability at 8 samples per (concept, modality),
against a 70% threshold:

| concept | text | image | spoken_audio | all three? |
| --- | --- | --- | --- | --- |
| bird | 8/8 | 8/8 | 8/8 | yes |
| cat | 8/8 | 7/8 | 8/8 | yes |
| giraffe | 8/8 | 8/8 | 6/8 (75%) | yes |
| microwave | 8/8 | 8/8 | 8/8 | yes |
| toilet | 8/8 | 8/8 | 7/8 | yes |
| **zebra** | 8/8 | 8/8 | **5/8 = 62.5%** | **no** |

Stage B then measured 3,744 intervention units over 32 distinct images, one
synchronized group per image, 0/234 pseudoreplicated rows — and its report
counted these as scientific support:

```
zebra  spoken_audio -> text
zebra  spoken_audio -> image
cat    spoken_audio -> image
toilet text          -> spoken_audio
toilet spoken_audio -> text
toilet image         -> spoken_audio
toilet spoken_audio -> image
```

**The root cause is a missing gate, not a bad measurement.** `Stage A` decided
which concepts the model can read out of each channel, and `Stage B`'s verdict
never consulted that decision. `causal_transfer_verdict` took a set of focal
concepts and an intervention table and had no access to the capability result at
all, so the two audio-related `zebra` cells — real measurements, correctly
computed, clearing their own matched random and external unrelated controls —
were treated as evidence that a *concept* had transferred into spoken audio.
They cannot be: the premise that the model recognizes `zebra` from speech is the
thing Stage A tested and rejected.

**The raw measurements remain valid.** Nothing about the capability result
changes what the intervention units measured. They are not recomputed, not
deleted, and not reinterpreted. What changes is what they are allowed to be
evidence *for*.

## The rule

One function, versioned, in `jlens/mmpilot/admissibility.py`:

```
CLAIM_ADMISSIBILITY_RULE_VERSION = "mmpilot.claim_admissibility.v1"
```

> A causal cell may support a scientific claim only when the concept passed the
> behavioral capability gate in every modality involved in that claim. For the
> principal three-modality verdict the required set is the stronger and simpler
> one: **text, image and spoken audio, all three**, for every cell regardless of
> which pair it is.

`claim_admissibility(concept, source_modality, target_modality, capability,
threshold, principal_three_modality)` returns a serializable decision carrying
`admissible`, the concept, both modalities, the required modalities, the observed
counts and accuracies, the threshold, the rejection reason and the rule version.
Accuracy is re-derived from `n_correct / n` rather than trusting a stored flag,
so `6/8 = 75%` clears a 70% threshold and `5/8 = 62.5%` does not, and the
boundary is not a float accident.

A concept that fails is `CAPABILITY_INELIGIBLE`:

- it **remains** in the raw tables as the predeclared diagnostic it is;
- it is labelled, with the arithmetic that rejected it;
- it cannot count toward supporting-cell totals;
- it cannot count toward bidirectional transfer;
- it cannot satisfy a GO or a WEAK GO criterion — in **either** direction: an
  inadmissible cell with an insane activation norm does not fail the study any
  more than an inadmissible cell with a clean effect passes it;
- it cannot be replaced post hoc by another concept. Substituting a concept
  chosen once the capability results were visible would make the selection
  depend on the outcome.

Every causal cell now carries `capability_admissible`,
`capability_rejection_reason`, `counted_toward_verdict`, the full
`capability_decision`, and `capability_decision_pair_specific` (the same question
asked of only the cell's own two modalities — reported as a diagnostic, never
used as the gate). `counted_toward_verdict` is the only flag any verdict reads.

`overall_verdict` additionally carries a hard invariant,
`only_capability_admissible_evidence_counted`, in its GO requirements: if a
future change ever let an ineligible cell into a supporting list, the GO is
refused rather than earned.

### Where the rule does not reach, and why that is stated

Verdict B's retrieval is **pooled over all selected concepts**. It cannot be
split per concept without recomputing retrieval, so a capability-ineligible
concept's activations are part of that pool. Verdict B is therefore reported
as-measured, its `capability_pool_disclosure` names the ineligible concepts in
the pool, and the overall verdict's `admissibility_limitation` says so in the
artifact. A three-modality claim never rests on Verdict B alone.

## The corrected Stage B interpretation

Under the rule, the eligible supporting cells implied by the observed Stage B
output are:

```
cat    spoken_audio -> image
toilet text          -> spoken_audio
toilet spoken_audio -> text
toilet image         -> spoken_audio
toilet spoken_audio -> image
```

`toilet` alone still provides all four audio-related directions, and therefore
both directions for `text <-> spoken_audio` and for `image <-> spoken_audio`, so
`L35_CAUSAL_TRANSFER` **may** remain `SUPPORTED` provided every existing control
and norm criterion still passes on the admissible cells.

**That verdict is not claimed here.** It is not hard-coded anywhere, and it is
recomputed from the saved real units by the amendment below. Until that has run
against the real run directory, the corrected verdict is unverified.

## The amended report

`jlens/mmpilot/amend.py` re-derives verdicts C, D and E from the stored units.
It loads no model, writes no unit, and does not touch the original report.

```
POSTPROCESSING_VERSION = "mmpilot.capability_filtered_postprocessing.v2"
native_audio_transfer_report_capability_filtered_v2.md
native_audio_transfer_summary_capability_filtered_v2.json
```

**Two fingerprints, deliberately separate.** Binding the reporting rule into the
run's generation fingerprint would refuse a completed run merely because a report
bug was fixed — the opposite of what resume is for. So:

- the **raw-generation fingerprint is preserved untouched**. Every completed
  capability and intervention unit stays reusable and a rerun still resumes it;
- the **postprocessing rule is versioned and checksummed separately**
  (`POSTPROCESSING_VERSION` plus the admissibility rule's own `rule_checksum`);
- the amended artifact **binds to both**, plus `source_unit_digest` — per stage,
  the unit count and a digest over the sorted `(key, unit_checksum)` pairs.
  `verify_amended_binding` refuses the artifact if the run fingerprint, the
  postprocessing version, the rule version, or any stage's unit digest differs.

Adding, removing, or editing one unit changes the digest and refuses the reuse.

## Stage C

The fixed focal design is unchanged, because `focal_concepts` is bound into
`scientific_fingerprint` — narrowing it would make the completed Stage A and
Stage B units unresumable, which is exactly the trade the amendment exists to
avoid.

What changes is **execution**: `stage_causal(concepts=...)` is a call parameter,
not a fingerprinted one, so Stage C spends model passes only on
capability-eligible focal concepts. An excluded concept's cells appear in the
Stage C table with `execution_status = "not_executed_capability_ineligible"`,
which is a different fact from "measured and inadmissible" and is recorded as
such. The verdict is still handed the full fixed focal set, so an excluded
concept leaves a visible hole rather than vanishing from the design.

Stage C prints, before spending anything: the fixed focal concepts, the
capability-eligible ones, the excluded ones with their reasons, the maximum
design budget, the actual capability-gated budget, and the no-post-hoc-
replacement statement.

## The stale stage gate

Observed: section 2 was edited so `RUN_L35_CAUSAL_STAGE` and
`CONFIRM_L35_CAUSAL_BUDGET` were `True` and that cell was executed — and section
15 still printed `skipped: STAGE_B_REQUESTED is False`, because the derived gates
had been computed in the budget cell and left behind in the kernel.

`jlens/mmpilot/stage_gates.py` derives the four gates from the eight raw
switches, and `refresh_stage_gates(globals())` writes them back into the
namespace. Every cell that can spend model passes calls it immediately before
deciding, so the only way to get a stale gate is to not run the cell. A missing
raw switch is refused rather than defaulted to `False`. No repair cell is needed,
and a test refuses any cell that assigns a derived gate by hand.

## Reruns that load nothing

To regenerate the amended report against the completed run, on a CPU runtime:

- `RUN_REAL_AUDIO_TRANSFER = True` (real run directory and real lens pins)
- `RUN_MODEL_STAGES = False`

Sections 1–7 are CPU-only; section 18b (`build_amended_report`) reads the stored
units directly. It prints the binding, writes both `_capability_filtered_v2`
artifacts, and leaves `native_audio_transfer_report.md` untouched.

To run Stage C afterwards, only once the amended Stage B verdict has been read:
`RUN_REAL_AUDIO_TRANSFER`, `RUN_MODEL_STAGES`, `CONFIRM_MODEL_LOAD`,
`CONFIRM_REPRESENTATION_BUDGET`, `RUN_L35_CAUSAL_STAGE`,
`CONFIRM_L35_CAUSAL_BUDGET`, `RUN_L38_L40_REPLICATION` and
`CONFIRM_REPLICATION_BUDGET` all `True`. Stage A and Stage B resume from their
completed units; Stage C spends passes only on the eligible focal concepts.
