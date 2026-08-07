# Multimodal J-space transfer pilot (SpokenCOCO × Gemma 4 E4B IT)

A minimal Colab pilot that answers one question and then stops: does a fixed,
previously fitted, **text-calibrated** J-lens expose J-space coordinates that
identify the same concept across written text, images, and spoken captions —
and can a concept direction estimated in one modality causally move that concept
in another, with no cross-modal alignment learned anywhere?

Notebook: [`notebooks/multimodal_jspace_spokencoco_pilot_colab.ipynb`](../notebooks/multimodal_jspace_spokencoco_pilot_colab.ipynb).
Code: [`jlens/mmpilot/`](../jlens/mmpilot).

This is a go/no-go probe, not a framework. If it says GO, the larger project is
worth building; if it says NO-GO, the money is saved.

## Running it

Open the notebook in Colab and run **section 0 first** — three cells that clone
or update this branch into `/content/jacobian-lens-gemma`, verify the checked-out
branch and print the commit, install the package with `pip install -e`, change
into the checkout, and confirm that `import jlens` resolves there. Nothing from
the repository is imported before that, and Drive is not needed for any of it.
The clone cell is idempotent: it clones when absent and otherwise fetches,
checks out, and hard-resets to origin, so a reconnected runtime picks up where
it left off. A private repo needs a `GITHUB_TOKEN` Colab secret; it is passed
per invocation as a header override and never written into `.git/config`.

Then run sections 1–17 in order. `RUN_REAL_PILOT` is `False`, so the first pass
exercises every cell against the synthetic world in about a minute.

The workflow is **CPU-first**. Sections 2–6 mount Drive, inspect the source
paths and metadata, discover the candidate concepts from the local COCO
annotations, compute the derived-data fingerprint, load a compatible cache or
build and publish one, and print the ranked coverage table and the selection.
With `RUN_MODEL_STAGES = False` — the committed default — the notebook stops
there. None of it needs an L4, and no model is loaded.

Only when the audit reports at least `MIN_CONCEPTS_REQUIRED` feasible concepts,
and the printed captions really do state their concepts, is it worth switching
to an L4 and setting `RUN_REAL_PILOT = True` and `RUN_MODEL_STAGES = True` by
hand. Nothing flips either of them for you.

The first real CPU pass publishes the derived cache; later sessions with the
same sources hit it and skip straight to the split.

### Media roots

Images and audio do **not** share a root. The dataset in Drive is laid out as

```
cstf_spokencoco/
├── coco/            train2014/ val2014/
├── SpokenCOCO/      wavs/
└── cstf_dataset_marker.json
```

while the manifest addresses images as `train2014/COCO_train2014_….jpg` and
recordings as `wavs/train/….wav`. An image path therefore resolves only under
`coco/` and an audio path only under `SpokenCOCO/`, which is why a single
dataset root resolved nothing at all.

Section 1 configures four roots and section 2 offers all of the ones that exist,
in this priority order, to both modalities:

```python
SPOKENCOCO_BASE_ROOT = ".../datasets/cstf_spokencoco"
IMAGE_MEDIA_ROOT     = ".../datasets/cstf_spokencoco/coco"
AUDIO_MEDIA_ROOT     = ".../datasets/cstf_spokencoco/SpokenCOCO"
DOWNLOAD_CACHE       = ".../datasets/cstf_spokencoco_download_cache"
```

`normalize_manifest` takes `image_roots` and `audio_roots` separately, and each
role is resolved against its own list. Passing only `media_roots` keeps the old
behaviour for layouts that do keep everything under one tree, so both
arrangements work.

Section 6 runs `audit_media_roots` **before** normalization: it probes a handful
of representative manifest paths and prints which root resolved each and how
(`exact` join vs `basename` fallback). It fails with the sampled evidence
attached when nothing resolves for a modality, when a path matches several roots
holding files of *different sizes* (a byte-identical download-cache mirror is
not a conflict — the higher-priority root simply wins), or when the image and
audio roots look exchanged. Normalization's own failure message reports the two
modalities separately, because one modality losing every file while the other
resolves cleanly is the signature of sibling roots and would otherwise hide
behind a combined ratio.

### Flaky Drive mounts

A Colab Drive mount intermittently fails a `stat` with
`OSError: [Errno 5] Input/output error` on a path that exists and reads fine
moments later. Every media probe therefore goes through `probe_path`, which
retries transient errnos (`EIO`, `ESTALE`, `EAGAIN`, `EBUSY`, `EINTR`,
`ETIMEDOUT`, and the two network ones) with bounded exponential backoff —
4 attempts, about 0.35 s in the worst case.

The distinction the code is careful about: **a transient failure is never
recorded as a missing file.** Doing so would quietly shrink the pilot's subset
and change what the experiment measured, without anything in the output saying
so. `FileNotFoundError`, `ENOENT`, and `ENOTDIR` mean absent; everything else
either retries or raises.

When retries run out, `MediaIOError` names the path, the configured root, the
attempt count, the errno, and the fix — remount Drive
(`drive.flush_and_unmount()`, then `drive.mount(..., force_remount=True)`) and
re-run, which resumes rather than repeating completed work. A permission error
is not retried, since waiting cannot fix it, and an unrecognised `OSError` is
refused rather than guessed to mean absent. Retries that *did* clear are counted
in the audit (`n_transient_io_retries`) and the notebook prints them, so a mount
that is merely degraded is visible before it becomes fatal.

## What SpokenCOCO can answer

SpokenCOCO carries COCO images, written captions, and spoken readings of those
captions. So the pilot tests transfer among **visual**, **written-linguistic**,
and **spoken-linguistic** evidence, under the names `text`, `image`, and
`spoken_audio`. It says nothing about environmental audio — barking, sirens,
instruments — and no artifact it writes may be described that way. The pair that
matters is text ↔ image; speech is a supplementary condition.

## What is deliberately not here

No phrase generation, no latent-to-language decoding, no reconstruction, no
natural-language autoencoding, no contrastive alignment, no cross-modal adapter,
no learned modality mapping, and no supervised probe as the primary method. No
probe is fitted at all: a linear diagnostic was permitted only if nearly free,
and it would not have changed any decision in the rubric.

Interventions add and subtract a direction on the residual stream. That is not
erasure and not projection ablation, and the report says so in the artifact
itself.

## Design

| Stage | What it does | Why it is shaped this way |
| --- | --- | --- |
| Model audit | Loads the model + processor, resolves the interface by inspecting `__call__` and the component sub-processors | Checkpoint naming is not evidence of audio support |
| Invariance gate | A capture hook must not change logits; a coefficient-zero edit must reproduce them | A "small effect" is meaningless if the harness itself perturbs the model |
| Manifest audit | Discovers the field mapping by name *and* by whether values resolve to real files; refuses ambiguity | The real manifest's schema is not known in advance and must not be guessed |
| Subset | Image-disjoint, group-disjoint, sample-disjoint; **not** concept-disjoint | The same concept must appear in source-train and in distinct target-test examples |
| Capability gate | Same question and candidates in every modality; complete candidate sequences scored by teacher-forced conditional log likelihood | First-token scoring would compare prefixes, not answers |
| Lens | Checksum-pinned, validated against repo/revision/width/layers, then read-only | No random basis, no SAE dictionary, no freshly fitted cross-modal lens |
| Codes | `h ≈ V s`, `s ≥ 0`, via the repository's existing nonnegative gradient pursuit | Reuses the validated settings rather than inventing a solver |
| Representational tests | Retrieval, matched-vs-mismatched similarity, weighted support overlap, raw-residual baseline, shuffled-label control | Retrieval alone is not causal evidence and is labelled as such |
| Directions | `ReLU(mean positive code − mean negative code)`, top-k, `v = V δ` | Estimated from source-modality *training* examples only |
| Causal transfer | Subtract from a held-out positive, add to a matched held-out negative, across the source × target matrix | The sign is derivable, so `signed_target_effect > 0` means "moved as intended" |

### Conventions, fixed once

- Hook site: output of `model.language_model.layers[l]` — the post-block residual.
- Position: the final prompt token (`prompt_len - 1`).
- Dictionary: rows of `W_U @ J_l`, final-norm weight **not** folded in.
- Direction normalization: unit L2, then scaled by the mean L2 norm of the
  target modality's clean activations at that layer. `alpha` multiplies that, so
  `alpha = 1.0` is an edit the size of the activation itself.

Retrieval never scores a query against a group from its own image. The caption,
its recording, and the image share content by construction; matching them would
measure the dataset's pairing, not the model.

## Local metadata expansion and coverage gate

The committed 48-record manifest remains immutable. After Drive is mounted,
section 4 performs a bounded breadth-first search (depth 3, at most 40 candidate
files, at most 512 MiB per file) under the configured SpokenCOCO and COCO roots.
It considers JSON, JSONL, CSV, and TSV metadata only; it does not recursively
stat media trees and never downloads anything. Every candidate is printed with
size, checksum, format, top-level schema, likely role fields, record count, and
an accept/reject reason.

Synchronized sources must expose deterministic image, caption, and audio fields.
The converter rejects conflicting audio-to-caption joins and missing explicit
identifiers; filename-derived identifiers are allowed only when a known metadata
ID set validates exactly one match.

## The candidate concepts are discovered, not chosen

The pilot used to screen six concepts written down by hand - bus, cat, dog,
horse, pizza, train. That is arbitrary relative to the dataset: it is not
derived from the data, cannot be checked against it, and decides in advance the
question the audit exists to ask.

SpokenCOCO has no concept ontology of its own; it inherits its visual semantics
from MS COCO, whose `instances_train*.json` / `instances_val*.json` files define
the object categories the images are actually annotated with. Section 4 reads
those files on the machine the notebook is running on, records their paths and
checksums, and treats **every** category in them as a candidate
(`jlens/mmpilot/concepts.py`). `captions_*.json` is deliberately not matched: it
carries no categories, and using it would substitute caption text for the visual
half of the evidence rule.

Each discovered category gets an explicit **lexical specification**: the accepted
written forms, the forms considered and rejected, the reason in both cases, an
ambiguity status, and any phrases that void a match. Plurals inflect the head
word only (`wine glass` -> `wine glasses`), with an irregular table for the cases
regular rules get wrong; there is no `-fe -> -ves` rule, because it is right for
`knife` and wrong for `giraffe` and nothing in the spelling distinguishes them.

The ambiguity policy has five statuses, and the ranking reads them:

| status | meaning | examples |
| --- | --- | --- |
| `clean` | no non-object sense common in captions | `zebra`, `pizza` |
| `resolved_by_exclusion` | a real collision, removed exactly by a phrase exclusion | `dog` excludes `hot dog`; `car` excludes `train car` |
| `alias_only` | the bare name is unusable; only an unambiguous phrase counts | `remote` -> `remote control`; `tie` -> `necktie`; `mouse` -> `computer mouse` |
| `ambiguous` | a non-object sense cannot be excised, so the category is flagged and penalised | `train` (the verb), `bear` (beyond `teddy bear`), `bicycle` (`bike` is also a motorbike) |
| `excluded` | no defensible form, or useless as a contrast | `orange` (the colour dominates), `person` (annotated on most images, so it discriminates nothing and empties the negative pool) |

A category absent from the curated table is still a candidate - it gets its name
plus a safe plural and is marked `derivation="default_morphology"`, so an
unreviewed category is visible as such rather than mistaken for a reviewed one.

## Feasibility, then a deterministic ranking

Every discovered category is measured against what the split will really yield:
distinct annotated images, synchronized image-caption-audio groups,
caption-confirmed positive groups, distinct speakers, feasible source-training
and held-out positives, available matched negatives, split feasibility,
category co-occurrence statistics, lexical ambiguity status, and a rejection
reason when infeasible.

The scientific minimums are a **gate applied first** and are never relaxed by
the score: 6 distinct images, 6 synchronized caption-confirmed groups, 4
source-training positives, 2 held-out positives, 6 matched negatives, and an
image- and synchronized-group-disjoint split. Fewer than two feasible concepts
is a DATASET NO-GO with the complete ranked table attached.

Among feasible concepts the order is a documented deterministic score
(`jlens.mmpilot.expansion.concept_score`), **not** raw frequency - ranking by
count alone would put COCO's most common category first regardless of whether it
can discriminate anything. Each component is a saturating ratio in `[0, 1]`
times a fixed weight, summed and rounded to six decimals:

| component | weight | what it rewards |
| --- | --- | --- |
| `images` | 3.0 | distinct valid-positive images, saturating at 12 |
| `groups` | 2.0 | synchronized groups the split will select, saturating at 12 |
| `split` | 1.5 | headroom above the train and held-out minimums |
| `negatives` | 1.0 | clean matched negatives, saturating at 24 |
| `speakers` | 1.0 | distinct speakers, saturating at 3 |
| `precision` | 2.0 | share of caption mentions the annotation also backs |
| `ambiguity` | 1.5 | the lexical status above (`clean` 1.0 down to `ambiguous` 0.4) |
| `independence` | 1.0 | `1 - max co-occurrence fraction` |

Ties break on distinct images, then selected groups, then the name, so the
ordering is total and reproducible across machines. The complete table -
rejected concepts and reasons included - is printed before anything is selected,
and the top two feasible concepts are taken with `GROUPS_PER_CONCEPT=6`.

The selected concepts are **shared** between source-training and target-test
examples. The split is image-disjoint and synchronized-group-disjoint, and
deliberately not concept-disjoint: transfer is measured on the same concept seen
through a different channel.

## The persistent derived-data cache

The CPU audit is expensive and deterministic in its inputs, so its results are
published into Drive under a directory named for a fingerprint of those inputs
(`jlens/mmpilot/cache.py`), by default under
`/content/drive/MyDrive/datasets/cstf_spokencoco_derived/jlens_mmpilot_v1`.

The fingerprint covers the SpokenCOCO source metadata checksums, the COCO
instance annotation checksums, the original manifest checksum, the
evidence-normalization/version identifier, the discovered category universe, the
lexical specification hash, the visual-plus-caption evidence rule, the
media-root layout, the scientific thresholds, and the split seed and split
algorithm version. Change any one and the artifacts land in a different
directory: there is nothing to invalidate and nothing to overwrite.

A fingerprint directory holds `metadata.json`, `concept_coverage.json`,
`concept_evidence_index.jsonl.gz`, `rejected_evidence_counts.json`,
`selected_concepts.json`, `pilot_subset.json`, `split_provenance.json` and
`_SUCCESS.json`.

The rules that make reuse safe:

- Source manifests and annotation files are read and checksummed, never written.
- No credentials, and no user-specific absolute Drive path: media is recorded
  relative to its configured root, so the cache is valid from a differently
  mounted Drive.
- Artifacts are built into a local staging directory under `/content` first and
  copied to a **new** fingerprint directory only once all of them exist.
- `_SUCCESS.json` is written **last** and carries a checksum and size for every
  published file.
- A directory without a valid success marker is *incomplete* - an interrupted
  publish - and is ignored and rebuilt, never half-read. A checksum mismatch on
  any file refuses the whole load.
- A directory whose stored fingerprint disagrees is *incompatible* and is
  refused loudly. Compatibility is never inferred from a directory name, and a
  timestamped run directory is never promoted on the strength of its contents
  looking right.
- Per-record data is one gzipped JSONL stream rather than thousands of tiny
  files, because Drive charges per write.

The notebook prints the state verbatim - `cache HIT`, `cache MISS`,
`cache INCOMPLETE`, `cache INCOMPATIBLE`. On a hit the coverage, selected
concepts, subset and split load directly and only a few representative
media-existence probes run; the join, the media validation and the audit are
skipped entirely.

## What counts as a positive

A valid synchronized positive requires **both** kinds of evidence:

1. **visual** - a COCO object annotation for the concept on the group's image;
2. **caption** - the concept, or an approved synonym from the explicit lexicon
   in `jlens/mmpilot/evidence.py`, present in the written caption as a whole
   word or phrase.

The first real run used the annotation alone and never read the caption. COCO
images routinely contain an object the caption does not mention, so groups were
selected as `cat` positives whose captions never said "cat"; the image arm
answered correctly and the text arm was scored wrong for answering honestly.
That is the asymmetry the behavioral gate reported (image 8/8, text 3-6/8), and
a group like it is not a valid test of transfer, because the two modalities are
not carrying the same claim.

Matching is normalized whole-word regex against the lexicon - `cat` does not
match `cattle`, `bus` does not match `business`. No embeddings, no learned
classifier, no language model, no fuzzy similarity, no external service. Every
match records the normalized caption and the matched span. SpokenCOCO recordings
are spoken readings of the written captions, so caption evidence describes the
recording's linguistic content; **no audio is transcribed**, and none of this is
evidence that the model can hear.

Matched negatives are stricter still: a negative must carry *neither* kind of
evidence for *any* screened concept, so a visual-only picture of a cat is
excluded from the negatives as well as from the positives.

The CPU-only audit (notebook sections 4-6, `jlens/mmpilot/evidence.py`) writes
`synchronized_evidence_audit.json`, `synchronized_evidence_manifest.json`,
`concept_ranking.json` and `split_provenance.json` into the run directory,
atomically, each carrying the lexicon hash, the configuration fingerprint and
the source checksums, and each refusing reuse under a different configuration.
It needs no GPU and no HuggingFace token.

The expanded manifest is atomically written under the persistent run directory
and reused only when the original checksum, every discovered-source checksum,
and conversion hash match. The final path is printed as `original` or
`expanded_derived`. Ranking preserves the scientific gate: two concepts, six
distinct image/groups each, four source-training positives, two held-out
positives, and six negatives. Source train/validation splits are preferred when
they satisfy those counts; otherwise stable IDs and the saved seed produce a
4/2 image-disjoint split. Up to four concepts are screened and at least two must
qualify; if fewer do, section 6 raises `DATASET NO-GO` before model execution
and reports exact shortfalls, including whether the binding constraint was
missing written-caption evidence on otherwise annotated images. It does not
lower `GROUPS_PER_CONCEPT=6`.

`TINY_SMOKE=False` by default. Setting it explicitly uses two concepts and two
groups only to validate real-media plumbing; its report is marked non-scientific
and cannot contribute to the research GO/WEAK-GO/NO-GO verdict.

## Assumptions this pilot makes explicit

**Gemma audio support is not assumed.** The processor is probed at run time for
an audio keyword argument and an audio component. If either is missing,
`spoken_audio` is marked blocked, the text-image pilot completes, and the report
records audio as a NO-GO with the observed interface attached. Speech is never
transcribed as a substitute.

> **Superseded, and the correction matters.** This pilot's real run blocked
> `spoken_audio` because the processor *"produced audio features but zero audio
> placeholder tokens"*. That observation is reproducible, and the diagnosis
> stopped one layer short: the cause was the **calling convention**, not the
> checkpoint. `Gemma4Processor` only expands audio tokens that are already in
> the text, so a bare `processor(text=..., audio=...)` call computes features
> and scatters them into nothing, silently. The supported native path is the
> chat-template audio content block. See
> [native_spoken_audio.md](native_spoken_audio.md) — the protocol is now
> implemented and probed in `jlens/mmpilot/audio.py`, and
> `notebooks/gemma4_native_spoken_audio_feasibility_colab.ipynb` audits it.
>
> **This changes nothing about the completed text-and-image results.** Audio was
> genuinely absent from them, and the probe-based support check is opt-in
> (`build_real_backend(..., resolve_audio=True)`, default False), so no existing
> run's behavior or fingerprint moves. That native audio is *technically usable*
> is engineering evidence only; it is not evidence that spoken-audio J-space
> transfer works, and that study has not been run.

**The lens artifact is assumed to exist, not to be creatable.** The pilot expects
the completed run directory `pilot_20260715T200437612150_311fd108c23a` in Drive
with `artifacts/lens.pt`, checksum
`sha256:7229c756…c96f474`, fitted at model revision
`fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd` over layers `[3, 7, 14, 21, 28, 35, 38]`.
Layers 35 and 38 are used. A missing or mismatched artifact is a hard NO-GO with
a named cause; nothing is substituted and no lens is fitted here.

## Cost of one real run

With the committed defaults — up to 4 concepts, 6 images per concept, 2 captions
per image, 3 modalities, 2 layers for activations, 1 layer for interventions,
1 concept and 2 held-out positives + 2 matched negatives in the transfer matrix:

| Stage | Model forward passes |
| --- | --- |
| Capability gate | ~380 (4 concepts × 8 groups × 3 modalities × 4 candidates) |
| Activation capture | ~220 (one pass yields both layers) |
| Interventions | ~2,000 (9 modality pairs × 4 targets × 13 conditions × 4 candidates, plus clean) |
| **Total** | **≈ 2,600** |

On one L4: roughly 30–45 minutes of compute after a 5–10 minute model load. If
audio is blocked the matrix drops to 2×2 and interventions fall to ~900 passes.
Drive usage is well under 100 MB — activations dominate at ~30 KB per unit, and
there are a few hundred of them.

Peak GPU memory is the reason dictionaries are built one layer at a time in fp16
with chunked construction: a float32 262144 × 2560 atom matrix is 2.7 GB, which
does not fit beside a bf16 Gemma 4 E4B on a 24 GB card twice over.

## Resume

Every stage writes one checksummed JSON unit per piece of work. A rerun reloads
what it can verify and recomputes the rest, printing `starting` or `resuming`
and per-stage computed/reused counts. The run directory is bound to a
fingerprint over model revision, processor revision, layers, lens checksum,
manifest checksum, split, and intervention config; a mismatch is refused with a
field-by-field diff rather than silently mixed.

## MOCK mode

`RUN_REAL_PILOT = False` runs the identical cells against a synthetic world with
a **known planted shared concept direction** — each concept has one vector that
enters the model the same way whether the evidence arrived as caption text,
"image" bytes, or "audio" bytes, on top of a modality offset and per-sample
nuisance. The mock model has Gemma 4's module layout, an identity final norm,
and unit-norm unembedding rows, so the expected sign of every intervention is
derivable rather than empirical.

The mock is not tuned to make J-space beat the raw difference-in-means baseline;
in a world this simple the raw estimator is very good and often wins, and the
report says so. **A MOCK GO proves the pipeline runs and nothing else.** Every
artifact it writes carries `mode: "mock"` and `scientific_evidence: false`.

## The decision rubric

Evaluated in code (`jlens/mmpilot/report.py`) so the recommendation cannot drift
with how the numbers are read.

Each criterion is `PASS`, `FAIL`, or `NOT_EVALUATED`. The third state exists
because a boolean cannot distinguish "measured and bad" from "never
measured": the first real run stopped at the behavioral gate, and the report
nonetheless declared the lens reconstruction inadequate — a verdict on a
measurement that never happened. A skipped stage is now reported as skipped,
with the precondition that stopped it named, and the recommendation's
rationale never blames it.

- **GO** — ≥2 concepts pass the behavioral gate for both text and image; the
  lens clears the matched-random reconstruction control; cross-modal J-space
  retrieval beats the shuffled
  control on both text→image and image→text; at least one off-diagonal
  text-image cell has the expected sign in ≥75% of its samples; that effect is
  ≥1.5× the random and unrelated-concept controls; activation norms stay in
  [0.5, 2.0]×; and the target moved more than the other candidates did.
- **WEAK GO** — representational transfer is clear but causal transfer is weak,
  inconsistent, or not separated from controls. The report names the smallest
  next experiment: hold the winning concept and layer fixed and run only that
  cell with more held-out targets and a denser alpha sweep.
- **NO-GO** — capability fails, no compatible lens exists, reconstruction is
  indistinguishable from matched random directions, or cross-modal structure is
  indistinguishable from shuffled labels.

## The lens criterion is above-random, not absolute

There is **no absolute reconstruction threshold**. An earlier version of this
rubric required the frozen lens to explain 50% of a held-out activation's
variance. That number has no basis. Anthropic's J-space work reports that the
top-k J-space component of a concept vector carries a **median of roughly 6-7%**
of its variance, and that at median occupancy the variance explained in excess
of a same-size random-direction control never exceeds about 10%. The finding is
that this *small* component carries disproportionate causal and reportable
content - under a 50% gate the published result itself would read as a failed
lens.

Absolute explained fraction is therefore a descriptive statistic. The criterion
(`jlens/mmpilot/reconstruction.py`) is: for held-out **text** activations at
each evaluated layer, decompose against the frozen dictionary, then against
deterministic random dictionaries matched in candidate-pool size, sparsity `k`,
atom norms, hidden width, dtype, device and pursuit settings. The lens passes
when the median excess over the control is positive and clears the control's
upper quantile at the primary layer or reproducibly at one of the two selected
layers, with finite, checksum-valid, non-degenerate pursuit output.

Two controls are computed. The **support-matched** one - exactly `k` random
atoms, norm-matched one for one - is reported for scale and **cannot gate
anything**: any dictionary with a selection pool larger than `k` beats it,
including pure noise. The **pool-matched** one gives the random directions the
same candidate pool and the same `k` as the lens, so the comparison isolates
atom direction, and it does fail for an unaligned dictionary.

### A capped control pool cannot produce a PASS

The pool-matched control was previously capped at 16384 candidate atoms while
the lens searched its full dictionary - roughly 262k atoms for Gemma. That is
not a matched comparison. The greedy maximum correlation with a target grows
with the number of candidates searched, so a control restricted to a sixteenth
of the lens's pool *understates* what random directions can do, in the direction
that flatters the lens. Disclosing the bias in each record was not enough: the
summary still read that comparison as evidence and could turn a 16x
search-space advantage into a scientific PASS.

The default is now an **equal-opportunity** comparison: `max_control_pool_atoms`
defaults to `None`, meaning the control pool has exactly as many candidate atoms
as the lens dictionary. Atoms are generated in chunks of 16384 directly in the
lens's dtype and device, so a full-size control never needs a transient float32
copy of itself and peak memory is one control dictionary the same size as the
lens - affordable on one L4 with Gemma E4B resident, and freed before the next
draw is built.

Where an equal pool genuinely is not affordable, the cap may still be set. When
it binds, the record's `criterion_status` becomes
`not_evaluated_pool_mismatch`, the layer is excluded from
`layers_above_random`, and the report renders the criterion **NOT EVALUATED** -
neither a pass nor a failure, because the comparison was never made. Setting
`require_pool_match=False` is the explicit, fingerprinted way to accept a
mismatched comparison; it yields `conditional_pool_mismatch` and still never
reads as a clean pass. An optional `pool_ladder` runs the same comparison at
increasing pool sizes and reports whether the lens's margin survives more search
freedom; that is informative, and it does not upgrade the verdict.

There is still no absolute explained-variance threshold anywhere.

A short `k` schedule (1, 2, 4, 8, 16, 25) also yields an **occupancy estimate**:
the largest `k` at which the lens's marginal reconstruction gain still exceeds
the control's bound on the same marginal. It is inspired by the published
occupancy measure and is explicitly **not** a replication of it - short
schedule, a handful of activations, two layers, capped control pool. The `k`
curve itself is exact: the pursuit is greedy and records
`residual_norm_history`, so the prefix of a `k_max` run equals a run configured
at that `k`.

Absolute reconstruction, excess over random, occupancy and causal usefulness are
four separate readings. None implies another; in particular a small-variance
J-space component may still move behaviour, which is what the intervention
criteria test.

The raw-residual difference direction is measured and reported but is **not**
allowed to veto specificity: it answers "did the decomposition earn its keep",
not "did transfer happen".

## What would justify the larger project

One concrete result: a direction estimated only from **text** training examples,
subtracted from a **held-out image** example, lowers that concept's
complete-sequence score with the expected sign on at least three of four target
samples, by at least 1.5× the norm-matched random and unrelated-concept controls
at the same alpha, while the activation norm stays within a factor of two and
the other answer candidates move less than the target does — and the mirrored
image→text cell shows the same pattern. Anything less than that is a WEAK GO at
best, and the follow-up should be the single cell that resolves it, not a
framework.

## Same-image dependence, and what the independent unit is

SpokenCOCO gives one COCO photograph roughly five written captions and a spoken
reading of each. `build_subset` keeps `max_groups_per_image` of them — two, by
default — so **one image can enter a run as two synchronized groups**. Those two
groups are not two observations. They are one photograph with the caption text
differing, which is why the capability output showed image margins arriving in
identical pairs while the text margins moved.

The consequence splits in two, and the two halves are handled differently.

**Representation.** `jlens.mmpilot.jspace.admissible_targets` drops any target
sharing the query's `image_id`, and that rule supersedes the group-level one.
Excluding only the exact synchronized group is not sufficient: a caption could
otherwise reach its own photograph through a sibling caption's group, which
measures the dataset's pairing rather than the model. The rule is applied
identically to retrieval, matched/mismatched separation, weighted support
overlap, the raw-residual baseline and the shuffled-label control — a report in
which some of those were image-disjoint and others were not would not mean
anything. The rule is versioned (`EXCLUSION_RULE_VERSION`) and the exclusion
accounting, including the eligible-target distribution, is reported per pair. A
modality pair that keeps no evaluable query raises rather than reporting the
`0.0` an empty denominator produces, which is indistinguishable from a measured
failure.

**Causation.** Intervention units are stored per synchronized group, so
averaging them flat counts one image twice and reports an `n` the design never
earned. `jlens.mmpilot.independence.summarize_interventions_by_image` averages
within image first and then computes the cell statistic over images: `n` counts
photographs. The group-level row is preserved verbatim under `group_level` and
the difference is reported, so the correction is visible rather than
substituted. Distinct captions of one image stay available per cell under
`per_image` as descriptive detail; identical image conditions never count more
than once as independent observations.

Note the asymmetry in what each half can fix. The representational rule is a
genuine repair — future runs measure the right thing. The causal aggregation is
an honest recount of evidence a completed run already produced; it cannot undo
`_select_targets` having deduplicated on `group_id` rather than `image_id` in
that run, only report where a cell rests on fewer photographs than it appears
to.

Image identity is resolved, never assumed: `resolve_image_identity` probes the
aliases the artifacts actually use, normalizes COCO filenames and integer ids to
one key, and cross-checks against the saved image media checksums. One id
spanning two photographs, one photograph under two ids, one group claiming two
images, or a group recorded in two splits each stop the audit. An audit that
guessed at identity could not establish independence, which is the only thing it
exists to do.

An image appearing in both splits, or sibling groups of one image landing on
opposite sides of the split, are **hard failures**: no re-aggregation repairs
them, because they make every held-out claim in the run untrue.

`notebooks/mmpilot_image_independence_audit_colab.ipynb` runs the whole audit on
a free CPU runtime against a completed run directory. It loads no model, no
processor and no media, needs no Hugging Face token, writes only new versioned
artifacts, and records the originals' checksums before and after. Its verdict is
one of `GO_CONFIRMED_AFTER_IMAGE_DEDUP`, `WEAK_GO_AFTER_IMAGE_DEDUP`,
`NO_GO_AFTER_IMAGE_DEDUP` or `AUDIT_BLOCKED`; the original recommendation is not
privileged, and a `GO` requires every concept to have passed behaviorally, both
directions to beat the shuffled control after image exclusion, the source effect
to exceed both the random and the external unrelated control, and the evidence
not to rest on a single duplicated photograph.

## The bounded six-concept robustness study

The corrected pilot returned `GO_CONFIRMED_AFTER_IMAGE_DEDUP`. The smallest
follow-up that result earns is: **does it replicate?** Not a bigger framework —
six concepts instead of four, eight distinct photographs per cell instead of
two, three focal concepts instead of two, one layer, two modalities, one frozen
lens, off-diagonal cells only.

### The photograph is the unit from selection onward

The audit corrected a completed run's arithmetic after the passes had been
spent. That is the right repair for evidence already collected and the wrong
place to fix the problem: a run that selects two captions of one photograph has
bought one observation at twice the price, and no downstream averaging returns
the observation it never made. So `jlens.mmpilot.selection` moves the rule
before execution, under an explicitly versioned `SubsetProfile`:

- `PILOT_PROFILE` reproduces the completed four-concept run byte for byte. It
  has to — that run's artifacts are on disk and must stay resumable and
  re-derivable.
- `IMAGE_UNIQUE_PROFILE` takes **one synchronized group per image**, chosen by a
  seeded stable rank over the content-derived group id. Ranking on the id rather
  than on position is what makes a re-derived subset the *same* subset: manifest
  ordering is an accident of how the file was written and must not decide which
  caption represents a photograph. The sibling captions it excludes are recorded
  on the row with the reason, never dropped silently.

Source-training positives and matched negatives are drawn from distinct,
mutually disjoint images; causal targets are deduplicated on `image_id` before
examples are chosen and held disjoint from the source-training images. A set
short of the stated count **refuses** rather than shrinking:
`InsufficientDistinctImagesError` names what was asked for, what was found, and
how many repeats were discarded. A silently shrunk cell reports an `n` it never
had.

One latent hazard was closed on the way: `build_subset` took prefixes of the
source-train and source-val image lists, and an image whose captions span both
would land on both sides of the split. The new profile draws the test pool from
what train did not claim, and a disjointness guard now fires under *any* profile
— it only triggers where a leak would have occurred, which the notebook already
rejected downstream.

`_split_plan`, which predicts what `build_subset` will yield so the ranking can
screen feasibility, takes the profile too. Without it the ranking applied the
pilot's `n-2`/`2` split to a design that splits 8/8 and rejected every candidate
for a shortfall that did not exist.

### What the fingerprint now binds

The pilot's fingerprint bound the model, lens, layers, manifest and alpha sweep.
It did not bind which concepts were chosen, in what order, how many distinct
images each cell got, how one group per photograph was picked, how targets were
deduplicated, or how the unrelated control was assigned — so two runs could
differ in all of that and still be treated as the same run.
`scientific_fingerprint` binds all of it (36 fields), and
`RunFingerprint.selection_config` carries it. Ordered sequences stay ordered:
the ranking *order* picks the focal concepts, so a reordering is a different
experiment even when the set matches.

Backward compatibility is exact rather than approximate. An empty
`selection_config` is **omitted from the digest**, so every directory written
before the field existed keeps the digest it was written with. A four-concept
pilot directory can never be resumed as a six-concept study: candidate scoring
alone changes from four-way to six-way, and a four-way capability unit is not
comparable to a six-way one.

### The decision is replication, not the strongest cell

The pilot's rubric took the maximum off-diagonal effect and asked whether it
beat its controls. With one focal concept that is the only thing available; with
three it is a way of reporting the luckiest cell. `ROBUSTNESS_GO` requires at
least **two of three** focal concepts to transfer in **both** directions, each
against its own matched random and external unrelated controls, with the
expected sign on at least 75% of photographs, sane activation norms, the target
moving more than the unrelated candidates, and the stated distinct-image counts
actually present. A raw difference-in-means direction matching the J-space one
downgrades to `ROBUSTNESS_WEAK_GO` rather than vetoing: it answers "did the
decomposition earn its keep", not "did transfer happen".

The external unrelated control is assigned by rotation over the non-focal
concepts in ranking order. Its only inputs are two ordered name lists, which is
what makes it impossible to have chosen after seeing how the candidates behaved.
In a forced choice among the focal concepts the only alternative is the target's
direct contrast, which is not unrelated to it at all.

### Cost, stated before it is spent

`estimate_model_passes` derives the budget from the configuration: at the
committed design, 1,152 capability + 224 activation + 576 clean-scoring + 5,184
intervention = **7,136 forward passes**, ~1 h on one L4, ~19 MB of Drive. A
"pass" is one teacher-forced forward over prompt plus one candidate sequence, so
scoring six candidates costs six passes — which is why widening the concept set
is not free and why same-modality causal cells are skipped: they cost the same
and answer a different question. The robustness notebook refuses to load Gemma
until `CONFIRM_MODEL_PASS_BUDGET` is set by hand, separately from
`RUN_MODEL_STAGES`.

Layer 38 remains late and remains the only validated layer. However well this
replicates, a final-prompt-token edit there cannot establish that an effect
precedes answer-language convergence. Spoken audio is excluded by design and
environmental audio is not tested; neither absence is evidence about either.

## Layer localization: where does the transfer first appear?

The robustness study returned `ROBUSTNESS_GO` at physical layer 38 and earned
one claim: a frozen text-calibrated J-lens at late layer 38 exposes concept
representations that causally transfer between written text and images. Layer 38
sits at ~90% of depth, close enough to the output that a final-prompt-token edit
cannot separate "concept representation" from "answer-language convergence".

`notebooks/multimodal_jspace_layer_localization_colab.ipynb` and
`jlens/mmlocalize/` ask the one follow-up that earns: **how early?** Four
predetermined physical layers — 20, 26, 32, 38 (~normalized 48, 62, 76, 90) —
two concepts that already replicated bidirectionally (cat, toilet), off-diagonal
cells only, the same frozen lens, never refitted.

### Physical is not normalized

`jlens/mmlocalize/layers.py` is the only place the conversion happens, and every
artifact carries both numbers. `LOCALIZATION_LAYERS` is immutable in membership
*and order*: `assert_immutable_layer_set` refuses a layer added after a result,
a layer dropped because it failed, and the `LAYERS=(32,)` shortcut. "Earliest"
is only meaningful along a fixed ordering.

### Eligibility is earned, and the new gate is not a laxer one

Layer 32 failed the v2 conjunction with median rank 1 and MRR ≈ 0.71 but **zero**
unique top-1 agreements, and a top-10 overlap (0.113) that lost to the
wrong-layer control (0.150). Those are one fact twice: both quantities were read
off a tie block at the maximum, where `argmax` reports the tie-break rule and a
top-10 slice is an arbitrary ten of the tied tokens.

`jlens/mmlocalize/lens_validity.py` rebuilds the gate around ties rather than
relaxing it. Every rank is reported under three conventions — optimistic
(the v2 number), pessimistic, and **midrank**, the mean over all tie-break
orders and the only one invariant to how ties break. The criterion uses midrank.
The gate **adds** three blocking clauses: a tied-at-maximum ceiling, a
wrong-layer MRR margin (the wrong-layer concern that failed layer 32, restated on
a metric ties do not corrupt), and stability across four fixed prompt folds. It
**drops** only the unique-top-1 floor, because under ties that quantity is a
property of the tie-break and not of the lens. Both gates are computed for every
layer and both are printed.

Layer 32's v2 numbers were known when the gate was written; pretending otherwise
would be worse than saying so. What is genuinely fixed in advance is everything
that decides the outcome — metrics, tie convention, thresholds, controls, folds,
and the held-out prompt set — all bound into `LayerValidityGate.digest`, which
participates in the resume fingerprint, so editing any of it invalidates stored
results rather than rescoring them.

`load_lens_for_localization` separates two claims the robustness loader bundles:
**fitted** (a Jacobian exists, so a readout is defined — required) and
**certified** (a held-out readout already passed — recorded, not required).
Without that separation the question would be unaskable, since deciding it is
Stage B's job.

### An ineligible layer is not a negative result

A layer that failed the gate was never causally tested, so "no transfer found
there" is not an observation. `assert_causally_eligible` refuses to run an
intervention stage there, its diagnostics are preserved, and the rubric routes
that case to `INCONCLUSIVE_LAYER_LOCALIZATION` — never `LATE_ONLY_SUPPORTED`,
which is a claim about layers that *were* tested and came back empty.

Layer 38 is the anchor, not a competitor: if the established result does not
reproduce on this subset, there is nothing for an earlier layer to be earlier
than, and that is inconclusive too.

### The targets are frozen before the first layer is scored

`jlens/mmlocalize/targets.py` verifies the completed run by fingerprint and then
only reads it, freezes and checksums the image set, and writes an exact
image-exclusion audit. The preferred policy takes photographs the completed run
never touched; the fallback reuses its images as a **paired within-sample**
comparison and is labelled as one everywhere it appears. The two are never
mixed, and the decision is made from availability alone, before any layer runs.
`assert_same_targets_across_layers` refuses drift: a depth contrast on different
photographs per layer is partly a contrast between photographs.

Concepts are **conditioned, not sampled**. cat and toilet are fixed because they
are the two that replicated bidirectionally, so every number answers "how early
for concepts already known to transfer", not "how many concepts transfer".

### Cost, stated before it is spent

Activation capture does **not** multiply by layers — one forward pass records all
four residuals, which is why a four-layer diagnostic costs what a one-layer study
did. Causal cost *does* multiply by eligible layers, and eligibility is unknown
until Stage B, so section 7 prints the worst case (all four eligible) for
confirmation and section 16 reprints the actual cost once the gate has decided.
`RUN_REAL_LOCALIZATION`, `RUN_MODEL_STAGES` and `CONFIRM_MODEL_PASS_BUDGET` are
all False in the committed notebook; `RUN_TEXT_RECALIBRATION` is separate again
and fits nothing — recalibration is a bounded, text-only plan published to a new
path, and the v2 artifact is never overwritten.

The verdict reports the **earliest tested layer with evidence**, never the
earliest layer in the model. Four layers cannot resolve where a signal begins: a
layer between two tested ones is untested, and anything shallower than 20 is
unexamined.

## The completed causal result is steering, not a coordinate swap

The causal transfer above uses **source-derived positive-minus-negative J-space
directions**: `δ = ReLU(mean positive code − mean negative code)` estimated from
source-modality *training* examples, mapped back through the frozen dictionary
as `v_concept = V δ`, unit-normalized, then added to or subtracted from the
residual at the final prompt token:

```
h' = h ± alpha * v_concept
```

That method, its terminology, and every number it produced stand unchanged. It
is a valid causal steering experiment.

It is **not** Anthropic's coordinate swap, and must not be described as one.
The paper's intervention takes both tokens' lens vectors as the columns of a
basis, reads the activation's coordinates in that basis with a pseudoinverse,
and exchanges them:

```
V = [v_source  v_target]
c = pinv(V) h
h_patched = h + alpha * V (sigma(c) - c)
```

Three differences matter, and none of them is cosmetic:

1. **Where the coefficient comes from.** Steering multiplies a *fixed* direction
   by a *chosen* alpha. The swap's coefficient is `c_target − c_source`, read
   off the activation itself, so an activation carrying no source content is
   barely moved.
2. **What is preserved.** The swap leaves the component orthogonal to
   `span{v_source, v_target}` unchanged exactly. Steering makes no such
   guarantee.
3. **Where it is applied.** The completed run edits the final prompt token of
   one layer. The paper's protocol edits *every* prompt position — including
   multimodal evidence positions — across a contiguous band of validated layers,
   recomputing the coordinates at each one.

The swap is implemented in `jlens/mmpilot/coordinate_swap.py`, specified in
[`coordinate_swap_protocol.md`](coordinate_swap_protocol.md), drawn in
[`assets/intervention_methods.svg`](assets/intervention_methods.svg), and
exercised by `notebooks/multimodal_jspace_coordinate_swap_mock_colab.ipynb`.
**No real coordinate-swap run exists.** Its artifacts carry an
`intervention_family` that no steering run ever wrote, and a coordinate-swap run
refuses to resume from a steering run's directory — the two families can never
share a number.

The planned identity-replacement and downstream-reasoning experiments are
separate claims from each other and from anything above. Behavioral outputs
remain text; `text`, `image` and `spoken_audio` are evidence modalities;
`spoken_audio` means spoken captions, not environmental sound.

## The completed result is candidate-conditioned

The behavioral question above listed every candidate concept in the prompt:

```
Question: which one of these is present: bird, cat, giraffe, microwave,
toilet, zebra? Answer with exactly one word.
Answer:
```

Every stage consumed that prompt — the capability gate, the captured
activations, the J-space codes, the estimated directions, and every intervention
forward pass. So the following is what the completed `THREE_MODALITY_GO` study
supports, stated exactly:

- the study used a **candidate-listed** behavioral question, and **every**
  candidate concept was present in it;
- the list was **identical across samples and modalities** and did **not**
  disclose which candidate was correct;
- source-derived positive-minus-negative estimation removes shared prompt
  components to first order, but **candidate priming remains a limitation**;
- candidate-order invariance controls **ordering** bias, not semantic priming;
- the result supports **candidate-conditioned cross-modal causal steering**;
- it does **not** establish spontaneous, unprompted concept emergence;
- the coordinate-swap follow-up will use **open prompts, with the candidates
  external to the model**.

Nothing here changes a number, a verdict, or a fingerprint. No completed
artifact was edited, and the legacy question keeps its bytes and its recorded
`gemma-it-chat-balanced-options-v1` protocol string so completed runs resume
unchanged. The protocol that replaces it for future work is specified in
[`prompt_protocol.md`](prompt_protocol.md) and implemented in
`jlens/mmpilot/prompt_protocol.py`.
