# Preprocessing the L32 convergence-resolution population

`notebooks/multimodal_jspace_l32_convergence_resolution_colab.ipynb` needs a
fresh SpokenCOCO population that is provably disjoint from every photograph,
recording and caption the completed runs already spent. Establishing that means
reading identities out of the completed runs' own artifacts on a Google Drive
mount, which is thousands of small files.

This document is about how that is done now, why it changed, and how to run the
two sessions the notebook is built for.

## What went wrong

Section 8a used to execute, in one uninterruptible go:

```python
COMPLETED_TREES_BEFORE = [run_tree_digest(d) for d in COMPLETED_RUN_DIRS]
EXCLUSION = harvest_excluded_identities(COMPLETED_RUN_DIRS, require=True)
```

That is **two** full traversals of every completed run — one to digest the whole
tree by name, size and mtime, then a second to read the identities out of unit
payloads — with the entire result held in Python memory. A real L4 session spent
more than four hours there, printed nothing, and would have lost all of it to a
runtime disconnect.

Three things were wrong, and each had to be fixed separately:

1. **the work was uncheckpointed** — an interruption cost everything;
2. **the work was silent** — there was no way to tell progress from a hang;
3. **the work was unnecessary** — most of the files read cannot contribute an
   identity that a much smaller set does not already carry, and the whole-tree
   digest re-walked tens of thousands of intervention units to protect a claim
   those units are not part of.

## What replaced it

`jlens/mmpilot/prep_cache.py`. Preprocessing is now a **deterministic,
Drive-backed preparation** keyed by a pre-model fingerprint.

### The preparation fingerprint

The cache directory is
`<PREP_CACHE_ROOT>/jlens_l32_resolution_prep_v1/prep_<digest prefix>` and the
digest binds every input upstream of the model that can change which media are
excluded or which are selected:

the harvest protocol version, the source-artifact strategy version, the identity
families, the fallback rule, the completed runs' basenames, their own recorded
fingerprints, their summary/report checksums, the cached expanded manifest's
checksum and schema version, the evidence lexicon hash, the frozen selected and
focal concepts, the sample-size rule version and plan digest, the selection
algorithm version, seed and profile version, and the four per-cell image counts.

Change any one of them and you get a different cache directory. The full digest
is stored inside and compared before anything is reused; the directory name
carries a 128-bit prefix so the deepest artifact path stays inside Windows'
260-character limit.

**No timestamped run directory is involved.** A preparation that had to be told
a run name could not be reused across sessions, which is the whole point.

### Minimal sources, and a completeness proof

`plan_sources` picks, per completed run, in this order:

1. a **bulk population manifest** that enumerates every used group, when the run
   wrote one;
2. otherwise the **activation** unit family plus the bulk run documents
   (`fingerprint.json`, `split_provenance.json`, `run_manifest.json`).

Capability units are never treated as a population source: `stage_capability`
stops at `max_capability_groups_per_concept` and scores positives only.

The activation family is a superset of the capability, J-space, direction,
intervention and readout families in `group_id`, `image_id` and `sample_id`,
because `stage_activations` iterates every group of both splits (keeping
`concept=None` negatives) while every other stage consumes activation records or
draws from `subset["splits"]["test"]` under the same `sample_id(group_id,
modality)`. That invariant is written into every completeness proof — and it is
**checked, not trusted**: the recovered distinct group and image counts are
compared against the completed run's own `split_provenance.json`.

A shortfall, or a run with no recorded population to anchor against, escalates
to a separately checkpointed **fallback scan** over the skipped families. It
never silently narrows the exclusion set; `assert_complete` refuses.

`tests/test_l32_prep_cache.py` proves the optimized harvest produces exactly the
same `ExclusionSet` digest as the legacy whole-tree harvester on real-shaped
fixtures, and includes a fixture where an identity really does hide in a skipped
family so both the failing proof and the recovering fallback are exercised.

### Checkpointing

One enumerated **source inventory** serves source integrity, identity harvesting
and the after-the-fact immutability check. The harvest then walks it in bounded
work units — **at most 25 files or 30 seconds, whichever comes first** — and
each unit is committed as one atomically replaced, checksummed gzip shard
*before* the cursor advances.

On resume:

* every committed shard is verified against its own checksum and its position in
  the cursor chain;
* a shard written just before the interruption killed the state update is
  **adopted** when it verifies and starts at the cursor — it is a durable
  checkpoint and repeating it would be work this module promised not to repeat;
* a torn shard is **quarantined** (kept on disk for inspection) and only its own
  batch, and anything after it, is recomputed;
* a source inventory that no longer matches, or state from a different
  preparation, is a **refusal** naming what moved — never a silent restart.

Stopping the runtime at any moment therefore repeats **at most one bounded
in-flight unit** and never restarts from file zero while a valid checkpoint
exists.

### Progress

`ProgressReporter` prints at least every 30 seconds: the run, the family, files
done out of total, the current shard, identities recovered this session, elapsed
time, and a remaining estimate from **measured** throughput — plus whether the
work is being computed or reused. A resume prints a `RESUMING PREPROCESSING`
banner with completed shards, remaining files, the last durable checkpoint and
what was reused. A complete cache prints `PREPROCESSING REUSED FROM DRIVE — no
source unit was read`.

### The read-only guarantee

Two mechanisms, and the stronger one is structural:

* **`assert_write_allowed` refuses** any write path that passes through a
  protected run prefix, outright. A digest can only notice afterwards that a
  completed run changed; this makes the write impossible.
* **`verify_sources_unchanged`** re-enumerates the harvested families — and only
  those — comparing name, size and mtime against sha256 content digests taken
  during the single harvest read.

This is at least as strong as the old whole-tree digest *for the artifacts the
study actually depends on*: it records the same three facts for every one of
them, adds a content hash the old scan never took (an edit that preserves size
and mtime is invisible to mtime and impossible to hide from sha256), and still
catches a *new* identity-bearing file. What it deliberately does not do is
re-walk tens of thousands of intervention units that cannot influence anything
here — the old scan's coverage of those bought no scientific strength and cost
hours.

## The two-session workflow

### Session 1 — free CPU runtime

```python
RUN_REAL_L32_CONVERGENCE_RESOLUTION = True
PREPROCESSING_ONLY = True
# every model switch stays False; PREPROCESSING_ONLY forces the gates shut anyway
```

Run sections 1–8. Section 8 harvests, checkpoints and persists the whole
preparation. **Stop it whenever you like** — at a checkpoint, at a disconnect, in
the middle of a batch. No run directory is created, no model is loaded, no
scientific artifact is written.

Re-run the notebook on a new CPU runtime as often as you need; each session
resumes at the last durable checkpoint.

### Session 2 — L4 runtime

```python
RUN_REAL_L32_CONVERGENCE_RESOLUTION = True
PREPROCESSING_ONLY = False
RUN_MODEL_STAGE = True
CONFIRM_MODEL_LOAD = True
CONFIRM_STAGE_A_BUDGET = True
```

The **same scientific configuration** must be used, or the preparation
fingerprint differs and section 8 will (correctly) build a new preparation
instead of reusing the old one.

Section 8 loads and verifies the cache — zero source-unit reads — and Stage A
starts against the persisted population. Section 11 **refuses to load Gemma**
while the preparation is incomplete, and section 8a warns loudly if it finds an
incomplete preparation on a runtime that has a GPU.

## What is reconstructed, and what is re-proven

Nothing required survives only in Python memory. A fresh process rebuilds:

| reconstructed from the cache | recomputed and checked against a digest |
|---|---|
| the exclusion set (verified against its own digest **and** its shards) | the independent pool, from `GROUPS` and the exclusion set |
| the concept ranking and the frozen-concept feasibility record | the population digest, over the reloaded subset |
| the selected population, its split provenance and its checksum | the **disjointness proof** and the pseudoreplication audit |
| the media resolution and the completeness proof | the source-inventory immutability check |

The disjointness proof is deliberately never loaded. It is the study's central
claim, and a loaded proof proves only that a file says so.

## Artifacts

Inside the preparation cache directory:

```
preparation_fingerprint.json      source_plan.json
source_inventory.json             (per stage, under harvest_minimal/ and harvest_fallback/)
harvest_state.json                (per stage)
shards/shard_*.json.gz            (per stage, checksummed)
quarantine/                       (torn shards, kept for inspection)
exclusion_set.json                exclusion_set_checksum.json
completeness_proof.json           media_resolution.json
independent_pool.json             prepared_selection.json
completed_run_timing.json         preparation_complete.json
preprocessing_report.md
```

A completed cache is immutable: recomputing it from the same inputs and getting
a different exclusion, population, pool, ranking or feasibility digest is a
refusal, not an overwrite.

## Scope

Preprocessing produces no model output. `preprocessing_report.md` says so, and
nothing in this document is evidence about Gemma, about layer 32, or about when
the native readout converges.
