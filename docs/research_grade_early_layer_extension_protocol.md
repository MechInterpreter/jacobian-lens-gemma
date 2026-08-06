# Early-layer J-lens extension — protocol

**Status:** frozen, not yet run. `research-grade-early-layer-jlens-extension-v1`.
**Configuration:** [`configs/research_grade_early_layer_jlens_extension_v1.json`](../configs/research_grade_early_layer_jlens_extension_v1.json).
**Notebook:** [`notebooks/research_grade_early_layer_jlens_extension_colab.ipynb`](../notebooks/research_grade_early_layer_jlens_extension_colab.ipynb).

This document states what the extension is and, more importantly, what it is
not allowed to do. It extends
[`research_grade_jlens_calibration_protocol.md`](research_grade_jlens_calibration_protocol.md);
that protocol and its configuration are **unmodified**.

---

## 1. Why this exists

The completed scale-100 calibration (`rgcalib_real_7e3736b4de8f`) validated
physical layers **35, 38 and 40** on an untouched confirmation set, and **failed
at 32 and 26**. Layer 32 failed closely enough to make more calibration data
worth spending. The completed multimodal experiment subsequently established
controlled three-modality causal transfer at 35/38/40, and the native-output
convergence audit found those layers already converged onto the clean answer.

The next scientific step is therefore a **validated earlier readout**:

- **primary target:** physical layer 32,
- **secondary / stretch:** physical layer 26,
- **descriptive only:** layers 20, 14 and 8 — reported continuously at every
  scale, and descriptive unless they independently pass the same gate on the
  fresh confirmation set,
- **untouched:** layers 35, 38 and 40 keep their existing published lenses. This
  extension neither overwrites, replaces nor republishes them.

---

## 2. The evidentiary boundary

This is the load-bearing section. Exactly one thing crosses from the completed
run, and exactly one thing is destroyed by it.

### The fitting accumulator is reusable

`J_l = E[∂h_final/∂h_l]` is a population mean estimated by a running average over
a deterministically ordered prompt list. `jlens.fitting.fit` reads **no**
validation or confirmation result while accumulating it — there is no parameter,
no callback and no branch through which a held-out number could reach the
accumulator. Continuing `{jacobian_sum, n_done}` from 100 prompts to 250 and
1,000 is therefore a **longer fit**, not a fit informed by its own evaluation.

Two consequences follow, and both are checked rather than assumed:

- the continuation is **bit-identical** to a fresh nested fit over the same
  ordering, because the additions happen in the same sequence
  (`verify_continuation_equals_fresh_fit`, proved in MOCK and in the test suite);
- the parent's first 100 prompts are never refitted, which is the whole economy
  of the extension.

### The old confirmation set is spent

The scale-100 confirmation set has been **opened**, its verdict has been
**read**, and that verdict is why this extension exists. A set whose result
already influenced the decision to run a larger scale cannot be that scale's
untouched endpoint.

It is therefore **not reused, not relabelled and not reset**. It is *excluded*
from every new split — by exact normalized checksum and by banded SimHash at
Hamming distance ≤ 3 — and the exclusion is counted and checksummed. The old
development set goes with it, for the same reason.

**Scale 100 is descriptive only.** It is scored on the *fresh* development set so
that the plateau rule has three points, but it can never be newly confirmed: the
set that would confirm it has been spent.

### The parent run is read-only

Every resolved parent file is checksummed before the extension runs and again
afterwards, and the two are compared byte for byte
(`protected_parent_checksums` / `assert_parent_unchanged`). Nothing in this
package opens a parent file for writing; the proof exists to make that auditable
rather than asserted.

---

## 3. Stage 1 — parent-run audit

Filenames and schema are **resolved, not assumed**. `discover_parent_files`
walks the directory; `resolve_parent_layout` matches the roles the extension
needs against what is actually there, with fallback glob patterns; a missing
required role stops the run with the role named, why it is needed, the paths
searched, and the files that do exist.

### Required parent artifacts

| role | canonical path | why |
|---|---|---|
| `fingerprint` | `fingerprint.json` | what the parent's results were produced from |
| `corpus_manifest` | `units/corpus/manifest.json` | corpus identity, split checksums, fit-prefix nesting audit |
| `fit_summary` | `units/fit_diagnostics/summary.json` | `n_done`, capture plan, snapshot table |
| `baseline_snapshot_unit` | `units/scale_snapshot/scale100.json` | the scale-100 snapshot's recorded checksum |
| `accumulator` | `checkpoints/jacobian_sum.pt` | the sufficient statistic being continued |
| `baseline_lens` | `artifacts/lens.scale100.pt` | the descriptive baseline lens |

Recorded when present, never blocking: `units/validation/scale100.json`,
`units/confirmation/scale100.json`, `units/publication/scale100.json`,
`units/scale_comparison/comparison.json`,
`artifacts/calibration_report.json`, `units/fit_diagnostics/rolling.json`.

Every unit is verified against **its own checksum** and against the run's
**fingerprint digest** before its contents are believed. A unit that fails
either is refused, not skipped.

### Blocking checks

`parent_fingerprint_recomputes`, `parent_configuration_checksum`,
`model_identity`, `tokenizer_identity`, `corpus_identity`,
`corpus_revision_recorded`, `text_only_corpus`,
`hook_site_and_residual_convention`, `d_model`, `capture_geometry`,
`fit_estimator_version`, `artifact_format_version`, `accumulator_format`,
`accumulator_layer_grid`, `accumulator_checksum_recorded`,
`n_done_equals_baseline`, `resume_cursor_matches_n_done`,
`baseline_snapshot_checksum`, `fit_prompt_ordering_protocol`,
`parent_fit_prefix_checksum_present`, `no_prompt_dropped_before_fitting`,
`old_split_checksums_present`, `old_duplicate_audit_present`,
`old_confirmation_selection_recorded`.

Non-blocking, recorded: `old_confirmation_vault_status_recorded`.

### The parent fit-prompt manifest

The parent stored split **checksums**, not corpus text. Its fit-prompt manifest
is `units/corpus/manifest.json → scale_nesting_audit.checksums["100"]`: a
checksum over the identities (`record_id`, `stream_index`, normalized-text
sha256, SimHash) of the first 100 fit records in nested order.

That checksum authenticates the parent's *fitted* prompts only because the parent
recorded `n_dropped_too_short == 0`, so its split prefix and its fitted prefix
are the same 100 records. **If that count is non-zero the extension refuses**:
the fitted prompt identity would not be recoverable from stored artifacts, and
nothing is inferred.

A read-only provenance manifest is written into the **extension's** directory
(`artifacts/parent_provenance.json`), never the parent's.

---

## 4. Stage 3 — the continuation rule

`parent-prefix-pinned-nested-order-v1`:

1. Reproduce the parent's collection exactly. Every parameter is read from the
   parent's artifacts with its source named (`parent_collection_parameters`):
   `min_chars` and `split_seed` from the corpus manifest, `min_fit` from the
   fingerprint's largest scale point, the held-out sizes from the recorded split
   sizes, and `max_texts` derived from the parent notebook's own collection
   bound. A wrong derivation cannot pass unnoticed, because —
2. `verify_reconstructed_partitions` requires the re-streamed collection to
   reproduce **every** parent split checksum and size. Any disagreement refuses.
3. Positions `0 … 99` of the extension's fit list are the parent's fit partition
   in the parent's nested order.
4. Any further positions come from records the parent's collection never
   reached, in **ascending stream index**.
5. `verify_fit_prefix` recomputes the checksum over the first 100 records and
   requires exact agreement with the parent's fit-prompt manifest. **The skip is
   authorized only after that agreement.**
6. The parent checkpoint is copied byte-for-byte into the extension run,
   checksum-verified, and upstream `fit` skips the first 100 via its own
   `next_idx`. Snapshots are written at 250 and 1,000.

**Why not simply re-run the parent's nested-hash ordering over a longer stream?**
Because `build_partitions` sorts by a hash of the record id and then truncates:
over a *larger* record set a late-streamed record can sort ahead of an early one,
silently breaking the prefix nesting the continuation depends on. The
pinned-prefix ordering is stable under stream extension; the pure hash ordering
is not. This is stated in the config as
`why_not_the_nested_hash_order_over_the_longer_stream`.

**Resume.** Atomic checkpoints, automatic reload. A checkpoint with different
`source_layers`, `target_layer` or `skip_first`, or one that has fallen behind
the parent's `n_done`, is **refused** (`ContinuationRefused`). A run directory
whose stored fingerprint disagrees is refused with a field-by-field diff
(`IncompatibleStateError`). Every extension artifact records the parent
checkpoint checksum.

**The estimator is unchanged.** No optimizer is introduced and no different
object is fitted.

---

## 5. Stage 4 — the fresh evaluation sets

`extension-fresh-eval-hash-bucket-v1`.

- **Pool:** records the parent's collection never reached — stream index beyond
  the parent's stop point. The exclusions below are therefore a guard, not the
  mechanism, and the tests exercise them on records that would otherwise get
  through.
- **New seeds:** split seed `20260807`, development prompt seed `20260808`,
  confirmation prompt seed `20260809`. The bucket hash is tagged `|ext|` so it is
  a genuinely different assignment, not the same function at a different seed.
- **Buckets:** 100 buckets; development `[0, 49]`, confirmation `[50, 99]`.
- **Sizes:** 256 development, 256 confirmation. **Never reduced.** If the corpus
  cannot fill them the run stops and asks for more corpus.
- **Exclusions, before bucketing:** every old fit record, every old development
  record, every old confirmation record, every new fit record through scale
  1,000 — by exact normalized checksum and by banded SimHash (Hamming ≤ 3),
  counted per excluded set.
- **Cross-split leakage audit:** run *independently*, on the constructed sets,
  against each other and against every excluded set. A hit raises
  (`ExtensionSplitRefused`). This is why the audit is a separate function from
  the builder: a guard only ever exercised by the thing that makes it
  unnecessary is not a guard.
- **Recorded:** record ids, stream indices, normalized checksums, SimHashes,
  prompt sha256s, target token ids, bucket rule, order rule, exclusion counts and
  every split checksum.

**Target discovery** is the frozen model's own final-layer argmax at the last
prompt position, via `ordinary_next_token_argmax`.
`select_diverse_validation_prompts` has **no parameter** through which a J-lens
or a candidate layer could be supplied — "confirmation selection must not consult
any J-lens result" is a property of the signature.

---

## 6. Stage 5 — the gate

`research-grade-early-layer-extension-tie-aware-native-readout-v1`, derived from
the completed study's `research-grade-calibration-tie-aware-native-readout-v1`.

| threshold | parent | extension |
|---|---|---|
| held-out prompts | 128 | **256** |
| distinct target tokens | ≥ 24 | **≥ 32 (stricter)** |
| max single-target share | ≤ 25% | ≤ 25% (unchanged) |
| tied-at-maximum rate | ≤ 0.50 | ≤ 0.50 (unchanged) |
| noise-control MRR | ≥ 1.5× and ≥ +0.10 | unchanged |
| wrong-layer margin | ≥ +0.15, `distant_layer_mapping` | unchanged |
| median midrank | ≤ 5.0 | unchanged |
| top-10 inclusion (pessimistic rank) | ≥ 0.50 | unchanged |
| fold stability | 4 folds, each beating every control and ≥ 0.50× overall MRR | unchanged |
| rank convention | midrank | unchanged |

**Nothing is loosened.** The one changed decision threshold is *harder*, and it
is frozen before any extension result exists, because a larger sample makes a
24-token floor easier to clear by accident. A test asserts field by field that
the only differences from the parent gate are `n_prompts`,
`min_distinct_target_tokens` and `version`.

Continuous metrics — MRR, median midrank, optimistic and pessimistic rank,
top-10 inclusion, tied-at-maximum rate, fold MRRs, control margins, the logit-lens
diagnostic, target diversity — are reported for **every layer at every scale**,
pass or fail.

---

## 7. Stage 6 — scale selection

`early-layer-extension-scale-selection-v1`, fixed before any development number
exists, digest bound into the run fingerprint.

0. The development set is scored at **100, 250 and 1,000**. Scale 100 is
   descriptive and **not selectable**; it is scored so the plateau rule has three
   points. (With only two points the plateau rule's continuation clause has no
   previous step and the verdict is `PLATEAU_REACHED` by construction — a
   satisfied condition rather than a measured one.)
1. Determine the eligible **earlier-layer** set (`{26, 32}`) at scale 250 and
   scale 1,000, using the unchanged gate.
2. Select the **smallest** candidate scale whose eligible earlier-layer set
   equals the scale-1,000 set, provided the predeclared plateau rule reports that
   improvement has stopped.
3. Otherwise select scale 1,000.
4. If **neither L32 nor L26** passes development at any candidate scale, still
   select scale 1,000 for one honest confirmation and report the result whatever
   it says.
5. Record `confirmation_not_consulted` and
   `multimodal_outcomes_not_consulted`.

The selection is written atomically **before** the confirmation vault is opened,
and `ConfirmationVault.unlock` refuses any payload that cannot assert
`confirmation_not_consulted`. Multimodal causal outcomes are not consulted.
Nothing is published from development results.

---

## 8. Stage 7 — untouched confirmation

The fresh 256-record confirmation set is held in a `ConfirmationVault`. It may be
constructed and sealed before scale selection, but it is released only against a
written selection payload, **exactly once**, and only the selected scale is
evaluated. The same frozen gate applies. Every layer's result is recorded,
including every failure. No layer is retried at another scale afterwards and no
threshold changes afterwards.

---

## 9. Stage 8 — publication

`publish_early_layer` inherits every refusal `publish_layer` already enforces —
no confirmation verdict, a failed verdict, a development verdict offered in its
place, a layer/scale mismatch, a protected completed-run path — and adds three:

- a layer outside the publication targets `{26, 32}`, so the confirmed
  L35/L38/L40 lenses are neither overwritten nor quietly republished;
- a destination outside the extension's own run directory;
- a destination inside the parent run.

Each published artifact carries: the parent run root, fingerprint digest,
accumulator checksum and `n_done`, the parent audit checksum; model, tokenizer
and corpus revisions; the fit prompt count, fit-order protocol and continuation
record; the new fit / development / confirmation split checksums and the fresh
splits manifest checksum; layer, normalized depth, hook site, residual
convention, target layer and estimator; development metrics and confirmation
metrics; gate version and digest, selection-rule digest and selection checksum;
the lens checksum, the base artifact checksum and its own artifact checksum;
`frozen`, `validated` and `publication_status`.

A layer that fails confirmation keeps every number it produced and is **never**
marked validated; no lens file is written for it.

If neither earlier layer passes at the authorized endpoint the run produces
**`EARLY_LAYER_CALIBRATION_NO_GO`** and states that the current estimator did not
yield a validated earlier readout at the authorized endpoint.

---

## 10. Budget

Anchor: the operator's own observation — the scale-100 fit took **≈ 7.1 minutes
on one L4** with this exact layer grid and capture plan.

**This is extrapolation, not measurement.** One observation, one runtime, one
prompt-length distribution. The estimator's structure supports linearity in
prompts (one forward plus a fixed `ceil(2560/8) = 320` backward passes per
prompt), but a shared cloud L4 is not a fixed-speed device. The band applied is
×0.9 – ×1.6, and **the high end is the planning number**.

Rows are **incremental, not cumulative** — the parent's 100 prompts are already
in the accumulator:

| scale | incremental prompts | forward | backward | L4 hours (central) | range |
|---|---|---|---|---|---|
| 250 | 150 | 150 | 48,000 | ≈ 0.18 | 0.16 – 0.28 |
| 1,000 | 750 | 750 | 240,000 | ≈ 0.89 | 0.80 – 1.42 |

Total from the parent's 100: **900 incremental prompts, ≈ 1.1 L4-hours central
(1.0 – 1.7 h)**. Drive ≈ **0.43 GiB** (200 MiB fp32 checkpoint, 200 MiB of fp16
snapshots, ≤ 25 MiB published, 20 MiB reports). Development and confirmation are
forward-pass workloads — roughly 18–50 minutes of model time for the whole study
— because the 320 backward passes per prompt happen only during fitting.

Resume: the accumulator checkpoint is written atomically every 25 prompts and
reloaded automatically. A disconnect loses at most one checkpoint interval and
never the parent's 100 prompts.

---

## 11. MOCK mode

Deterministic, CPU-only, no Gemma, no Hub, no Drive, no corpus download. Built in
two halves:

- **The parent is real.** `build_mock_parent_run` produces a parent by running
  the completed study's own code — the real store, the real `run_calibration`,
  the real upstream `fit` — against a tiny frozen CPU stack at the real depth
  (42 blocks) and the real layer grid. The audit and the continuation therefore
  meet an artifact of the same kind they will meet on Drive.
- **The verdicts meet known cases.** Rows are synthesised with controlled rank
  and tie structure and scored by the **real** `tie_aware_row`; the gate, folds,
  comparison, plateau rule and selection rule are the real ones.

| scenario | what it proves |
|---|---|
| `l32_late_pass` | L32 fails at the smaller scale, passes development at the largest, passes fresh confirmation → GO, published |
| `l32_confirmation_fail` | L32 passes development at both scales (so the *smaller* scale is selected) and fails the untouched confirmation → NO-GO, nothing published, never marked validated |
| `no_early_layer` | L26 and L32 both fail → the predeclared rule still takes the largest scale to one honest confirmation → NO-GO |
| `plant_old_confirmation_record` | an old confirmation prompt planted in a new split is refused by the independent cross-split audit |
| `corrupt_continued_checkpoint` | a continuation that does not equal a fresh nested fit is refused |

**MOCK success proves pipeline behaviour only.** It is not evidence about
Gemma 4, about layer 32, about layer 26, or about whether a validated earlier
readout exists.

---

## 12. Scope boundaries

Not done, and not reachable from this code: SpokenCOCO or any image or audio
data in calibration or lens validation; modality-specific lenses; cross-modal
adapters; a changed estimator; a lowered gate; reuse of old confirmation
evidence; a rerun of the completed multimodal experiment; any modification of the
parent calibration run; any scale beyond 1,000.

The only multimodal imports are pure helpers already sanctioned by the completed
study: `jlens.mmpilot.store` (canonical JSON and checksums) and
`jlens.mmlocalize.lens_validity` (the tie-aware scorer over two logit vectors).
Sharing them is what makes a number here mean what the same number meant in the
completed runs; a test reads the actual import statements to enforce it.

---

## 13. What a passing result would and would not establish

A pass at layer 32 or 26 would be a statement about **text-only native readout**
under a stronger calibration and a stricter diversity floor, on data nothing has
seen. It would **not** be a statement about causal transfer, about J-space, or
about any modality other than text. Earlier-layer causal transfer would still
require the multimodal causal experiment actually being run against an
independently confirmed lens — which this extension does not do and does not
authorize.
