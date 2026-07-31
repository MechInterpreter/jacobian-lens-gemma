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

Then run sections 1–16 in order. `RUN_REAL_PILOT` is `False`, so the first pass
exercises every cell against the synthetic world in about a minute. Set it to
`True`, restart, and run all to do the real thing.

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
section 6 performs a bounded breadth-first search (depth 3, at most 40 candidate
files, at most 512 MiB per file) under the configured SpokenCOCO and COCO roots.
It considers JSON, JSONL, CSV, and TSV metadata only; it does not recursively
stat media trees and never downloads anything. Every candidate is printed with
size, checksum, format, top-level schema, likely role fields, record count, and
an accept/reject reason.

Synchronized sources must expose deterministic image, caption, and audio fields.
The converter rejects conflicting audio-to-caption joins and missing explicit
identifiers; filename-derived identifiers are allowed only when a known metadata
ID set validates exactly one match. Official COCO category annotations are used
when locally available. Otherwise captions are normalized to lowercase word
tokens with conservative plural handling and whole-word boundaries; substring
matches are forbidden. Each group records its annotation source.

The expanded manifest is atomically written under the persistent run directory
and reused only when the original checksum, every discovered-source checksum,
and conversion hash match. The final path is printed as `original` or
`expanded_derived`. Ranking preserves the scientific gate: two concepts, six
distinct image/groups each, four source-training positives, two held-out
positives, and six negatives. Source train/validation splits are preferred when
they satisfy those counts; otherwise stable IDs and the saved seed produce a
4/2 image-disjoint split. If two concepts do not qualify, section 7 raises
`DATASET NO-GO` before model execution and reports exact shortfalls. It does not
lower `GROUPS_PER_CONCEPT=6`.

`TINY_SMOKE=False` by default. Setting it explicitly uses two concepts and two
groups only to validate real-media plumbing; its report is marked non-scientific
and cannot contribute to the research GO/WEAK-GO/NO-GO verdict.

## Assumptions this pilot makes explicit

**Gemma audio support is not assumed.** The processor is probed at run time for
an audio keyword argument and an audio component. If either is missing,
`spoken_audio` is marked blocked, the text-image pilot completes, and the report
records audio as a NO-GO with the observed interface attached. Speech is never
transcribed as a substitute. The notebook has been executed only against the
mock backend, so whether this checkpoint's processor accepts audio is still an
open empirical question that section 5 answers on first real run.

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

- **GO** — ≥2 concepts pass the behavioral gate for both text and image; the
  lens reconstructs adequately; cross-modal J-space retrieval beats the shuffled
  control on both text→image and image→text; at least one off-diagonal
  text-image cell has the expected sign in ≥75% of its samples; that effect is
  ≥1.5× the random and unrelated-concept controls; activation norms stay in
  [0.5, 2.0]×; and the target moved more than the other candidates did.
- **WEAK GO** — representational transfer is clear but causal transfer is weak,
  inconsistent, or not separated from controls. The report names the smallest
  next experiment: hold the winning concept and layer fixed and run only that
  cell with more held-out targets and a denser alpha sweep.
- **NO-GO** — capability fails, no compatible lens exists, reconstruction is
  poor, or cross-modal structure is indistinguishable from shuffled labels.

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
