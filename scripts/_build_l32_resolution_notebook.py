# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Generate ``notebooks/multimodal_jspace_l32_convergence_resolution_colab.ipynb``.

The notebook is written from source here rather than edited inside a JSON blob,
so the committed file stays output-free and byte-reproducible.

Run with ``python scripts/_build_l32_resolution_notebook.py`` after changing a
cell.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TARGET = (
    REPO_ROOT
    / "notebooks"
    / "multimodal_jspace_l32_convergence_resolution_colab.ipynb"
)

CELLS: list[tuple[str, str]] = []


def markdown(text: str) -> None:
    CELLS.append(("markdown", text.strip("\n")))


def code(text: str) -> None:
    CELLS.append(("code", text.strip("\n")))


# ============================================================ 0. front matter

markdown(
    """
# Independent layer-32 convergence resolution — a fresh SpokenCOCO population

The completed open-prompt follow-up
(`mml32_l32_followup_20260808T182717`,
fingerprint `sha256:29aa1d13...cf75b67e`) reported:

```
L32 lens integrity                 PASSED
L32 representational transfer      SUPPORTED
L32 causal transfer                WEAK
L32 native output convergence      AMBIGUOUS
pre-convergence causal transfer    INCONCLUSIVE
paired L35 reference (same open protocol)  SUPPORTED
```

Its own predeclared next step — written into
`adjacent_layer_recommendation` **before** that result was visible — was a
separately fingerprinted, independently sampled layer-32 convergence-resolution
population. This notebook is that study, and nothing else.

## The one question

> On a fresh, independently selected set of SpokenCOCO photographs, written
> captions and spoken-caption recordings, is physical layer 32 classified
> `CONVERGED`, `NOT_CONVERGED` or `AMBIGUOUS` under the already frozen native
> direct-readout criterion?

## This is not an attempt to reach NOT_CONVERGED

* the criterion thresholds are frozen and their digest
  (`sha256:abbb23e1...0d0446b5a1f`) is **checked**, never recomputed into
  agreement;
* the six candidates and the three focal concepts are frozen in ranking order
  and are never replaced after a result;
* the sample size is fixed, printed and digested before any model output is
  observed, and the stopping rule forbids adding units after a classification is
  seen;
* `AMBIGUOUS` and `CONVERGED` are reported in exactly the same words as
  `NOT_CONVERGED`.

An ambiguous layer can stay ambiguous at any *n*. A second independent sample
landing in the band is evidence that the layer genuinely sits there — it is not
a shortage of data, and nothing in this notebook treats it as one.

## What "independent" is verified to mean

Four identities must be disjoint from the completed run, and each can fail on
its own: **image id**, **synchronized group id**, **recording / audio path**,
and **caption text**. Section 8 harvests all four out of the completed runs'
own artifacts, resolves the recordings and captions the unit files do not store
by mapping the group ids back through the manifest, filters the pool *before*
selection, and then **proves** disjointness over the population that was
actually built. A study that cannot construct the required independent
population refuses; it never shrinks quietly.

## Stages

**Stage A** (always): independent capability, activations at layer 32, the
native direct readout, the three controls, the classification.

**Stage B** (conditionally): causal replication on this same fresh population.
The gate is fixed in section 2 and quoted in every artifact. It stops Stage B
in exactly the cases where the headline result is *unfavourable* to the
pre-convergence hypothesis, and those Stage-A outcomes are reported in full —
so it is an efficiency gate, not a filter.

## Not in scope

* Environmental (non-speech) audio. Nothing here measures it and nothing here
  claims anything about it.
* An Anthropic-style two-coordinate swap. `jlens/mmpilot/coordinate_swap.py`
  implements it; it needs a *contiguous* confirmed layer band, which today's
  confirmed set (32, and separately 35/38/40) does not provide. That is a
  separate future study and this notebook never renames steering as a swap.
* Fitting, refitting or modifying any lens. Nothing here is fitted.
"""
)

# =========================================================== 1. bootstrap

markdown(
    """
## 1. Bootstrap repository

Run these three cells first, in order. They use nothing but the standard
library: the repository is not importable until 1c has installed it.

Google Drive is **not** needed here.
"""
)

code(
    '''
# 1a. Bootstrap constants only. Nothing from this repository is imported yet.
REPO_URL = "https://github.com/MechInterpreter/jacobian-lens-gemma.git"
BRANCH = "experiment/spokencoco-jspace-pilot"
REPO_DIR = "/content/jacobian-lens-gemma"

print(f"repo   {REPO_URL}")
print(f"branch {BRANCH}")
print(f"target {REPO_DIR}")
'''
)

code(
    '''
# 1b. Clone or update the repository, then verify the checked-out branch.
import os
import subprocess
import sys
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
REPO_PATH = Path(os.environ.get("MMPILOT_REPO_DIR") or REPO_DIR)


def _git(*arguments, cwd=None):
    result = subprocess.run(
        ["git", *arguments], cwd=cwd, capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(arguments)} failed:\\n{result.stdout}\\n{result.stderr}"
        )
    return result.stdout.strip()


if IN_COLAB:
    if not (REPO_PATH / ".git").is_dir():
        _git("clone", "--branch", BRANCH, REPO_URL, str(REPO_PATH))
    else:
        _git("fetch", "origin", BRANCH, cwd=REPO_PATH)
        _git("checkout", BRANCH, cwd=REPO_PATH)
        _git("reset", "--hard", f"origin/{BRANCH}", cwd=REPO_PATH)

CHECKED_OUT_BRANCH = _git("rev-parse", "--abbrev-ref", "HEAD", cwd=REPO_PATH)
COMMIT = _git("rev-parse", "HEAD", cwd=REPO_PATH)
if IN_COLAB and CHECKED_OUT_BRANCH != BRANCH:
    raise RuntimeError(
        f"checked out {CHECKED_OUT_BRANCH!r}, expected {BRANCH!r} — "
        "refusing to continue against the wrong code"
    )
print(f"branch {CHECKED_OUT_BRANCH}")
print(f"commit {COMMIT}")
'''
)

code(
    '''
# 1c. Install the repository, move into it, and verify that `import jlens`
# resolves to this checkout.
if IN_COLAB:
    print("installing the repository (editable) ...")
    result = subprocess.run(
        [
            sys.executable, "-m", "pip", "install",
            "transformers==5.13.1", "-e", f"{REPO_PATH}[gemma]",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"pip install -e failed:\\n{result.stdout[-2000:]}\\n{result.stderr[-2000:]}"
        )

os.chdir(REPO_PATH)
if str(REPO_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_PATH))

try:
    import jlens
except ModuleNotFoundError as error:
    raise RuntimeError(
        f"`import jlens` is still not importable after installation: {error}"
    ) from error

if Path(jlens.__file__).resolve().parent.parent != REPO_PATH.resolve():
    raise RuntimeError(
        f"`import jlens` resolved to {jlens.__file__}, not this checkout"
    )
print(f"jlens  {jlens.__file__}")
print(f"cwd    {os.getcwd()}")
'''
)

# ============================================================== 2. switches

markdown(
    """
## 2. Switches, pins and the frozen design

Every switch is `False` in the committed notebook. Opening it starts nothing,
downloads nothing and spends nothing.

| switch | what it unlocks |
|---|---|
| `RUN_REAL_L32_CONVERGENCE_RESOLUTION` | the real Drive artifacts, the real cached manifest and the real published L32 lens instead of the deterministic MOCK world |
| `PREPROCESSING_ONLY` | stops after section 8. Preprocessing is CPU work over Drive; this is the switch that lets it run on a free CPU runtime and be resumed on an L4 |
| `RUN_MODEL_STAGE` | allows Gemma to be loaded at all |
| `CONFIRM_MODEL_LOAD` | acknowledges the ~16 GB download |
| `CONFIRM_STAGE_A_BUDGET` | acknowledges the Stage-A pass budget printed in section 10 |
| `RUN_STAGE_B_CAUSAL_REPLICATION` | adds the optional causal replication on this same fresh population |
| `CONFIRM_STAGE_B_BUDGET` | acknowledges Stage B's additional passes |

The gates are **re-derived from the raw switches inside every cell that can
spend a pass**, so editing this cell and re-running it can never leave a stale
gate behind: section 2 defines `refresh_gates()` and every such cell calls it.
`PREPROCESSING_ONLY` closes all three of them, so it can never be set alongside
a model switch and quietly spend an L4 hour anyway.

### The two-session workflow this notebook is built for

| session | runtime | switches | what happens |
|---|---|---|---|
| 1 | **free CPU** | `RUN_REAL_… = True`, `PREPROCESSING_ONLY = True` | section 8 harvests, checkpoints and persists the whole preparation to Drive. Stop it whenever you like: at most the one in-flight batch of ≤25 files is repeated. |
| 2 | **L4** | `RUN_REAL_… = True`, `PREPROCESSING_ONLY = False`, model switches on | section 8 loads and verifies the preparation cache in minutes without re-reading a single source unit, then Stage A starts. |

Nothing required survives only in Python memory: `GROUPS`, `EXCLUSION`, `POOL`,
the ranking, the frozen-concept feasibility check, the selected population, its
provenance and every digest are reconstructed from persisted artifacts in a
fresh process, and each is checked against a digest rather than trusted.

### The Stage-B rule, fixed here and not revisited

Stage B runs only when Stage A returns `L32_INDEPENDENT_NOT_CONVERGED` with
passing controls. Stage B exists to support one claim — causal transfer
*before native direct-readout convergence* — and that claim needs the
convergence half to be `NOT_CONVERGED`. Under `CONVERGED` or `AMBIGUOUS` the
causal passes cannot contribute to it however they come out.

`RUN_STAGE_B_CAUSAL_REPLICATION` remains available as an **explicit override**;
an overridden run is stamped `gate_overridden: true` in every artifact and its
causal numbers are reported as descriptive, never as the predeclared path.

### Not free parameters

`L32_PHYSICAL_LAYER`, `LENS_FITTED_SCALE`, `SELECTED_CONCEPTS` and
`FOCAL_CONCEPTS` are what the completed studies fixed. Section 5 refuses an
artifact that disagrees, and section 8 refuses a concept list that does.
"""
)

code(
    '''
# 2. Configuration. Requires section 1 (it imports from the repository).
# Nothing here mounts Drive, reads data, or loads a model.
RUN_REAL_L32_CONVERGENCE_RESOLUTION = False

# Stop after section 8 with the whole preparation persisted to Drive. This is
# the CPU-session switch: preprocessing reads thousands of small files off a
# Drive mount, which a GPU cannot make faster, and section 11 refuses to load
# Gemma until the preparation is complete and verified anyway.
PREPROCESSING_ONLY = False

RUN_MODEL_STAGE = False
CONFIRM_MODEL_LOAD = False
CONFIRM_STAGE_A_BUDGET = False
RUN_STAGE_B_CAUSAL_REPLICATION = False
CONFIRM_STAGE_B_BUDGET = False

# The one place a rebuild of the 125,198-group evidence join is permitted. Left
# False so a normal session cannot spend twenty-five minutes rediscovering a
# cache it already has; section 7 refuses instead and says which input moved.
ALLOW_MANIFEST_REBUILD = False

# ------------------------------------------------------------------ design
L32_PHYSICAL_LAYER = 32
LENS_FITTED_SCALE = 250
CAPABILITY_THRESHOLD = 0.7

# A NEW seed. The completed follow-up drew under
# "spokencoco-l32-followup-v1"; drawing again under the same seed from a
# filtered pool would be a deterministic function of that run's leftovers, not
# an independent sample.
SPLIT_SEED = "spokencoco-l32-resolution-v1"

ALPHAS = (0.0, 0.25, 0.5, 1.0)

# --------------------------------------- the extension run that published L32
# The FILENAME IS NEVER TYPED HERE. Section 5 resolves it from this run's own
# report, publication metadata and checksums.
EXTENSION_RUN_DIR = (
    "/content/drive/MyDrive/jacobian-lens-gemma/runs/rgext_real_c18f03f06e7b"
)

# ------------------------------------- completed runs: READ-ONLY, and avoided
# Every population this study must be independent of. Section 8 harvests their
# identities; section 18 proves none of them changed while this study ran.
COMPLETED_RUN_DIRS = (
    "/content/drive/MyDrive/jacobian-lens-gemma/runs/"
    "mml32_l32_followup_20260808T182717",
    "/content/drive/MyDrive/jacobian-lens-gemma/runs/"
    "mmaudio_native_audio_transfer_20260806T144822",
)
COMPLETED_FOLLOWUP_SUMMARY = "l32_followup_report.json"

# --------------------------------------------------- the cached evidence join
# The completed L32 follow-up is already a pinned, read-only dependency and is
# the run that actually persisted the verified 125,198-group manifest on the
# user's Drive. Derive the cache path from that pin instead of guessing that a
# separately migrated dataset-level cache exists.
CACHED_EXPANDED_MANIFEST = (
    f"{COMPLETED_RUN_DIRS[0]}/expanded_manifest.json"
)
EXPECTED_CACHED_GROUP_COUNT = 125198

# ------------------------------------------------- the preparation cache root
# Preprocessing artifacts are DERIVED data, never a run. The directory under
# this root is named by the preparation fingerprint alone, so a CPU session and
# an L4 session that agree on every pre-model input land in the same place
# without either of them having to name a timestamped run directory.
PREP_CACHE_ROOT = "/content/drive/MyDrive/datasets/cstf_spokencoco_derived"

# Bounded work units, per the resumability contract. Stopping the runtime at any
# moment repeats at most ONE batch of this size and never rescans from zero.
PREP_BATCH_FILES = 25
PREP_CHECKPOINT_SECONDS = 30.0
PREP_PROGRESS_SECONDS = 30.0

MODEL_REPO_ID = "google/gemma-4-E4B-it"
MODEL_REVISION = "fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd"
TRANSFORMERS_VERSION_EXPECTED = "5.13.1"
EXPECT_N_LAYERS, EXPECT_D_MODEL, EXPECT_VOCAB = 42, 2560, 262144

AUDIO_PROTOCOL_VERSION_EXPECTED = "jlens.mmpilot.native_spoken_audio.v1"
AUDIO_PROTOCOL_FINGERPRINT_EXPECTED = (
    "sha256:9ad8bcc9420a7983f6e3b75d5d7080c0e2fcf0a94a76431917fcde73ba777920"
)

SPOKENCOCO_BASE_ROOT = "/content/drive/MyDrive/datasets/cstf_spokencoco"
IMAGE_MEDIA_ROOT = "/content/drive/MyDrive/datasets/cstf_spokencoco/coco"
AUDIO_MEDIA_ROOT = "/content/drive/MyDrive/datasets/cstf_spokencoco/SpokenCOCO"
DOWNLOAD_CACHE = "/content/drive/MyDrive/datasets/cstf_spokencoco_download_cache"
MANIFEST_PATH = "/content/drive/MyDrive/datasets/spokencoco_manifest.json"
RUNS_ROOT = "/content/drive/MyDrive/jacobian-lens-gemma/runs"

# Never written into. Completed runs are evidence, not scratch.
PROTECTED_RUN_PREFIXES = (
    "mmpilot_pilot_", "mmrobust_", "mmlocalize_", "rgcalib_", "rgext_",
    "audioaudit_", "text_jlens_", "mmaudio_", "mmconv_", "mml32_l32_followup",
)

import json

from jlens.mmpilot.l32_followup import INTERVENTION_FAMILY, OPEN_PROMPT_PROTOCOL
from jlens.mmpilot.l32_resolution import (
    FOCAL_CONCEPTS as FROZEN_FOCAL_CONCEPTS,
)
from jlens.mmpilot.l32_resolution import (
    FROZEN_CRITERION_DIGEST,
    L32_RESOLUTION_PROTOCOL,
    RESOLUTION_RUN_PREFIX,
    SAMPLE_SIZE_RULE,
    STAGE_B_RULE,
    STAGE_PLAN_VERSION,
    derive_resolution_gates,
    format_resolution_gates,
    plan_sample_size,
    stage_plan,
)
from jlens.mmpilot.l32_resolution import (
    SELECTED_CONCEPTS as FROZEN_SELECTED_CONCEPTS,
)
from jlens.mmpilot.pipeline import PilotConfig
from jlens.mmpilot.selection import IMAGE_UNIQUE_MOCK_PROFILE, IMAGE_UNIQUE_PROFILE
from jlens.mmpilot.tri_modal import TriModalThresholds

SAMPLE_PLAN = plan_sample_size()
STAGE_PLAN = stage_plan()

# The sample size is the PLAN's, not a hand-set number. MOCK shrinks it to keep
# a CPU run in seconds, and its profile name is part of the fingerprint so a
# MOCK directory can never be resumed as a real one.
if RUN_REAL_L32_CONVERGENCE_RESOLUTION:
    N_TRAIN_POSITIVE_IMAGES = SAMPLE_PLAN.n_train_positive_images
    N_TEST_POSITIVE_IMAGES = SAMPLE_PLAN.n_test_positive_images
    N_TRAIN_NEGATIVE_IMAGES = SAMPLE_PLAN.n_train_negative_images
    N_TEST_NEGATIVE_IMAGES = SAMPLE_PLAN.n_test_negative_images
else:
    N_TRAIN_POSITIVE_IMAGES = N_TEST_POSITIVE_IMAGES = 2
    N_TRAIN_NEGATIVE_IMAGES = N_TEST_NEGATIVE_IMAGES = 2

SCRATCH = Path(os.environ.get("MMPILOT_SCRATCH") or "/content/l32res_scratch")
SCRATCH.mkdir(parents=True, exist_ok=True)
RESOLVED_RUNS_ROOT = Path(
    os.environ.get("MMPILOT_RUNS_ROOT")
    or (RUNS_ROOT if RUN_REAL_L32_CONVERGENCE_RESOLUTION else SCRATCH / "runs")
)
EXTENSION_RUN_DIR = os.environ.get("MMPILOT_EXTENSION_RUN_DIR") or EXTENSION_RUN_DIR
if os.environ.get("MMPILOT_COMPLETED_RUN_DIRS"):
    COMPLETED_RUN_DIRS = tuple(
        os.environ["MMPILOT_COMPLETED_RUN_DIRS"].split(os.pathsep)
    )

MODALITIES = ("text", "image", "spoken_audio")
PROFILE = (
    IMAGE_UNIQUE_PROFILE
    if RUN_REAL_L32_CONVERGENCE_RESOLUTION
    else IMAGE_UNIQUE_MOCK_PROFILE
)

# The MOCK decoder has six blocks. Its stand-in for "layer 32 published by the
# extension at its own scale" is layer 1 — the same substitution every completed
# MOCK run uses.
RESOLUTION_LAYER = (
    L32_PHYSICAL_LAYER if RUN_REAL_L32_CONVERGENCE_RESOLUTION else 1
)
LAYERS = (RESOLUTION_LAYER,)

SELECTED_CONCEPTS_EXPECTED = list(FROZEN_SELECTED_CONCEPTS)
FOCAL_CONCEPTS_EXPECTED = list(FROZEN_FOCAL_CONCEPTS)

THRESHOLDS = TriModalThresholds(
    capability_threshold=CAPABILITY_THRESHOLD,
    required_positive_images_per_cell=N_TEST_POSITIVE_IMAGES,
    required_negative_images_per_cell=N_TEST_NEGATIVE_IMAGES,
)

CONFIG = PilotConfig(
    mode=(
        "l32_convergence_resolution"
        if RUN_REAL_L32_CONVERGENCE_RESOLUTION
        else "mock"
    ),
    layers=(RESOLUTION_LAYER,),
    causal_layers=(RESOLUTION_LAYER,),
    modalities=MODALITIES,
    capability_threshold=CAPABILITY_THRESHOLD,
    alphas=tuple(ALPHAS),
    n_target_examples=N_TEST_POSITIVE_IMAGES,
    pursuit_k=25 if RUN_REAL_L32_CONVERGENCE_RESOLUTION else 8,
    pursuit_correlation_chunk_size=(
        65536 if RUN_REAL_L32_CONVERGENCE_RESOLUTION else None
    ),
    direction_top_k=16 if RUN_REAL_L32_CONVERGENCE_RESOLUTION else 4,
    n_permutations=50 if RUN_REAL_L32_CONVERGENCE_RESOLUTION else 8,
    max_capability_groups_per_concept=(
        N_TRAIN_POSITIVE_IMAGES + N_TEST_POSITIVE_IMAGES
    ),
    seed=20260809,
    subset_profile=PROFILE.name,
    image_unique_targets=True,
    min_source_positive_images=N_TRAIN_POSITIVE_IMAGES,
    min_source_negative_images=N_TRAIN_NEGATIVE_IMAGES,
    off_diagonal_causal_only=True,
)

SWITCHES = {
    name: globals()[name]
    for name in (
        "RUN_REAL_L32_CONVERGENCE_RESOLUTION",
        "PREPROCESSING_ONLY",
        "RUN_MODEL_STAGE",
        "CONFIRM_MODEL_LOAD",
        "CONFIRM_STAGE_A_BUDGET",
        "RUN_STAGE_B_CAUSAL_REPLICATION",
        "CONFIRM_STAGE_B_BUDGET",
    )
}


def _apply_gates(gates):
    globals().update(gates)
    return gates


def refresh_gates():
    """Re-derive the three gates from the raw switches, MOCK override included.

    Called at the top of every cell that can spend a model pass. A gate computed
    once and read many cells later goes stale the moment section 2 is re-run,
    which is how a notebook comes to print "skipped: not requested" directly
    below a switch the operator has plainly just set.

    The MOCK override lives here rather than in each caller so it cannot be
    applied in one cell and forgotten in the next: the three real gates protect
    a ~16 GB download and an L4-hour bill, and in MOCK the "model" is a
    few-hundred-parameter CPU stub with nothing to protect. Stage B stays opt-in
    in BOTH modes, because it is the switch that changes what is claimed.
    """
    gates = derive_resolution_gates(globals())
    if PREPROCESSING_ONLY:
        # One switch, and it closes everything a model could be loaded through.
        # Overloading the MOCK/real switch to mean "CPU session" instead would
        # make a preprocessing run indistinguishable from a MOCK run in the
        # artifacts, which is exactly the confusion this study cannot afford.
        return _apply_gates({name: False for name in gates})
    if not RUN_REAL_L32_CONVERGENCE_RESOLUTION:
        gates = {
            **gates,
            "MODEL_STAGE_ENABLED": True,
            "STAGE_A_ENABLED": True,
            # Re-derived from the Stage-B switches alone, because the real rule
            # chains Stage B onto CONFIRM_STAGE_A_BUDGET and that budget gate is
            # not in force here. Leaving it chained would report
            # STAGE_B_REQUESTED=False in a MOCK run that plainly did run Stage B.
            "STAGE_B_REQUESTED": bool(
                RUN_STAGE_B_CAUSAL_REPLICATION and CONFIRM_STAGE_B_BUDGET
            ),
        }
    return _apply_gates(gates)


GATES = refresh_gates()

print(format_resolution_gates(GATES, switches=SWITCHES))
print()
print(f"preprocessing only  {PREPROCESSING_ONLY}")
if PREPROCESSING_ONLY:
    print("  every model gate is forced closed. Section 8 runs, persists the")
    print("  whole preparation to Drive and the notebook stops there.")
print(f"prep cache root     {PREP_CACHE_ROOT}")
print(f"prep work unit      {PREP_BATCH_FILES} files or "
      f"{PREP_CHECKPOINT_SECONDS:.0f}s, whichever comes first")
print()
print(f"protocol            {L32_RESOLUTION_PROTOCOL}")
print(f"run family          {RESOLUTION_RUN_PREFIX}_*")
print(f"prompt protocol     {OPEN_PROMPT_PROTOCOL}")
print(f"intervention        {INTERVENTION_FAMILY}")
print("                    (steering, NOT the Anthropic coordinate swap)")
print(f"layer               {RESOLUTION_LAYER}  (real: {L32_PHYSICAL_LAYER})")
print(f"criterion digest    {FROZEN_CRITERION_DIGEST}")
print(f"stage plan          {STAGE_PLAN_VERSION}")
print(f"sample-size rule    {SAMPLE_SIZE_RULE.digest}")
print()
print("STAGE B RULE (fixed now, before Stage A is opened):")
print(f"  {STAGE_B_RULE}")
'''
)

# ============================================================== 3. Drive

markdown(
    """
## 3. Mount Drive and verify the configured paths (read-only)

Skipped entirely in MOCK. On the real path every configured location is checked
for existence before anything is loaded, and the completed runs are checked
**read-only**.
"""
)

code(
    '''
# 3. Mount Drive and verify the configured paths exist. Read-only.
DRIVE_STATUS = "skipped"
if RUN_REAL_L32_CONVERGENCE_RESOLUTION and IN_COLAB:
    from google.colab import drive

    drive.mount("/content/drive")
    DRIVE_STATUS = "mounted"

if RUN_REAL_L32_CONVERGENCE_RESOLUTION:
    _required = {
        "extension run": EXTENSION_RUN_DIR,
        "cached expanded manifest": CACHED_EXPANDED_MANIFEST,
        "manifest": MANIFEST_PATH,
        "image media root": IMAGE_MEDIA_ROOT,
        "audio media root": AUDIO_MEDIA_ROOT,
        "runs root": RUNS_ROOT,
    }
    for _index, _completed in enumerate(COMPLETED_RUN_DIRS):
        _required[f"completed run {_index}"] = _completed
    _missing = {
        name: path for name, path in _required.items() if not Path(path).exists()
    }
    if _missing:
        raise RuntimeError(
            "these configured paths do not exist:\\n"
            + "\\n".join(f"  {name}: {path}" for name, path in _missing.items())
        )
    for name, path in sorted(_required.items()):
        print(f"  ok  {name:26s} {path}")
    # The preparation cache is DERIVED data this notebook creates. Its parent
    # must exist; the cache itself is created by section 8 and is the one
    # location outside this study's run directory that is ever written to.
    _prep_parent = Path(PREP_CACHE_ROOT).parent
    if not _prep_parent.is_dir():
        raise RuntimeError(
            f"the preparation cache's parent directory {_prep_parent} does not "
            "exist, so preprocessing has nowhere durable to checkpoint to"
        )
    print(f"  ok  {'prep cache parent':26s} {_prep_parent}")
    print(f"      prep cache root        {PREP_CACHE_ROOT} "
          f"(exists={Path(PREP_CACHE_ROOT).is_dir()}, created if absent)")
print(f"drive: {DRIVE_STATUS}")
'''
)

# ========================================================== 4. runtime

markdown(
    """
## 4. Runtime report

Prints the accelerator, the library versions and the free Drive space. Touches
no dataset and loads no model.
"""
)

code(
    '''
# 4. Runtime report. Never touches the dataset.
import platform

import torch

TRANSFORMERS_VERSION = None
try:
    import transformers

    TRANSFORMERS_VERSION = transformers.__version__
except ModuleNotFoundError:
    pass

TORCH_VERSION = torch.__version__
print(f"python        {platform.python_version()}")
print(f"torch         {TORCH_VERSION}")
print(f"transformers  {TRANSFORMERS_VERSION}")
print(f"cuda          {torch.cuda.is_available()}")
if torch.cuda.is_available():
    _properties = torch.cuda.get_device_properties(0)
    print(f"gpu           {_properties.name} "
          f"({_properties.total_memory / 1e9:.1f} GB)")
print()
print("Section 8 (preprocessing) is CPU and Drive I/O only. A GPU makes none of")
print("it faster; section 8a says so again once it knows whether the")
print("preparation cache is already complete.")

if RUN_REAL_L32_CONVERGENCE_RESOLUTION and TRANSFORMERS_VERSION != (
    TRANSFORMERS_VERSION_EXPECTED
):
    raise RuntimeError(
        f"transformers {TRANSFORMERS_VERSION} != pinned "
        f"{TRANSFORMERS_VERSION_EXPECTED}; the completed studies were measured "
        "under the pin and a different tokenizer or processor is a different "
        "experiment"
    )
'''
)

# ================================================= 5. the published L32 lens

markdown(
    """
## 5. Resolve, verify and load the published L32 lens — **nothing is fitted**

The artifact filename is never typed into this notebook. Discovery starts from
the extension run's own report, reads the publication block's checksum for layer
32, finds the `.extension.json` sidecar that claims that layer *at scale 250*,
takes the path from the sidecar, and requires the file's own digest to equal both
the sidecar's and the report's.

Scale is part of "confirmed": layer 32 **failed** at scale 100 and **passed** at
scale 250 on its own untouched 256-prompt confirmation set. An artifact fitted at
any other scale is refused.
"""
)

code(
    '''
# 5. Resolve, verify and load the published L32 lens. NOTHING IS FITTED HERE.
from jlens.mmpilot.l32_followup import (
    discover_published_l32_lens,
    l32_expectations,
    validate_discovered_lens,
)
from jlens.mmpilot.published_lens import load_published_lenses

MOCK_D_MODEL = 24
if not RUN_REAL_L32_CONVERGENCE_RESOLUTION:
    from jlens.mmpilot.mock import build_mock_extension_run

    _mock_extension = SCRATCH / "mock_extension_run"
    if not (_mock_extension / "artifacts").is_dir():
        build_mock_extension_run(
            _mock_extension, layer=RESOLUTION_LAYER, scale=LENS_FITTED_SCALE,
            d_model=MOCK_D_MODEL,
        )
    EXTENSION_RUN_DIR = str(_mock_extension)

DISCOVERED = discover_published_l32_lens(
    EXTENSION_RUN_DIR,
    layer=RESOLUTION_LAYER,
    expected_scale=LENS_FITTED_SCALE,
    require_real_mode=True,
)
EXPECTATIONS = l32_expectations(
    model_repo_id=(
        MODEL_REPO_ID if RUN_REAL_L32_CONVERGENCE_RESOLUTION else "mock/gemma-like"
    ),
    model_revision=(
        MODEL_REVISION
        if RUN_REAL_L32_CONVERGENCE_RESOLUTION
        else "mockrevision0000000000000000000000000000"
    ),
    d_model=EXPECT_D_MODEL if RUN_REAL_L32_CONVERGENCE_RESOLUTION else MOCK_D_MODEL,
    layer=RESOLUTION_LAYER,
    scale=LENS_FITTED_SCALE,
)
L32_VALIDATION = validate_discovered_lens(DISCOVERED, EXPECTATIONS)
L32_LENSES = load_published_lenses([DISCOVERED.spec()], EXPECTATIONS)
L32_LENS = L32_LENSES.lens

print(f"resolved      {DISCOVERED.lens_path}")
print(f"checksum      {DISCOVERED.lens_checksum}")
print(f"layer         {DISCOVERED.layer}   scale {DISCOVERED.scale}")
print(f"validated     {L32_VALIDATION['passed']}")
print(f"d_model       {L32_LENSES.d_model}")
if not L32_VALIDATION["passed"]:
    raise RuntimeError(
        f"the published layer {RESOLUTION_LAYER} artifact failed validation: "
        f"{L32_VALIDATION['failed_checks']}"
    )
'''
)

# ============================================= 6. the predeclared sample size

markdown(
    """
## 6. The predeclared sample size — printed before any model output exists

The frozen bars are **0.50** (`NOT_CONVERGED`, generous scoring) and **0.90**
(`CONVERGED`, unique scoring). The gap is 0.40, so the design targets:

* a 95% **Wilson** half-width of at most **0.20** at the worst case *p* = 0.5 —
  half the gap, so a point estimate sitting on one bar has an interval that does
  not reach the other;
* at least **80%** power to observe a rate at or below 0.50 when the truth is
  0.40;
* at least **80%** power to observe a rate at or above 0.90 when the truth is
  0.95.

Wilson rather than Wald because both bars sit near the ends of [0, 1], where a
Wald interval runs past 1.0 and stops meaning anything. Exact binomial tails
rather than a normal approximation, because *n* here is in the tens.

The design is sized for **two** admissible focal concepts, not three: the
completed study already found `zebra` capability-ineligible in spoken audio under
the frozen admissibility rule, so sizing for three would be sizing for a
population we have positive evidence against. Precision at one, two and three
admissible concepts is printed, so the case that actually occurs is visible
rather than only the case that was planned for.

The stopping rule is fixed-n with **no interim look**.
"""
)

code(
    '''
# 6. The predeclared sample size and stopping rule. CPU only, no model.
from jlens.mmpilot.l32_resolution import format_sample_plan

SAMPLE_PLAN_RECORD = SAMPLE_PLAN.to_dict()
print(format_sample_plan(SAMPLE_PLAN))
print()
print(f"plan digest  {SAMPLE_PLAN_RECORD['plan_digest']}")
if not SAMPLE_PLAN.adequate:
    raise RuntimeError(
        "no rung of the predeclared image-count ladder resolves the frozen "
        "bars at the expected admissible-concept count. The design is not "
        "adequate and the study does not proceed on one that is not."
    )
if RUN_REAL_L32_CONVERGENCE_RESOLUTION and (
    N_TEST_POSITIVE_IMAGES + N_TRAIN_POSITIVE_IMAGES
    != SAMPLE_PLAN.images_per_concept
):
    raise RuntimeError(
        "the real run's per-concept image counts do not match the predeclared "
        f"plan ({SAMPLE_PLAN.images_per_concept}). The sample size is not a "
        "notebook parameter."
    )
'''
)

# ============================================ 7. cached data loading

markdown(
    """
## 7. Cached data loading — the 125,198-group join is **loaded, never rebuilt**

Rediscovering the SpokenCOCO evidence join costs twenty-five minutes of CPU on a
Colab runtime and produces the same answer every time. The old
`persist_expanded_manifest` could not help: it checks compatibility only *after*
its caller has already built the whole `ExpansionResult`, so a cache hit still
paid for the miss.

`load_expanded_manifest` is the compatibility half on its own. It verifies, in
order:

1. the derivation **schema version**;
2. the **original manifest checksum**;
3. every **source metadata checksum**;
4. the **conversion hash**, which covers the evidence rule and the
   **evidence lexicon hash**;
5. the **cached group count**.

Any failure is a refusal that names the clause and its two values — never a
silent rebuild. `ALLOW_MANIFEST_REBUILD` is the one switch that permits a
rebuild, and it is `False` in the committed notebook.

**The COCO category universe is recovered from the cache itself.** An expanded
manifest serializes each group's `concept_annotations` but not the annotation
files, so `universe_from_concept_annotations` reads the category *names* back
out of the groups. It deliberately leaves `category_ids` empty and says so: those
ids are not in the cache, and a `universe_hash` that pretended otherwise would
make the cache the authority on something it never saw.
"""
)

code(
    '''
# 7a. MOCK ONLY — manufacture the cache this study is about to load.
#
# On the real path the cache already exists on Drive and this cell does nothing.
# In MOCK it is built once, here, so section 7b exercises the *loading* path
# rather than skipping it: a cache-direct loader that is never made to load
# anything in the test suite is a cache-direct loader in name only.
from jlens.mmpilot import expansion as expansion_module
from jlens.mmpilot import manifest as manifest_module

MOCK_WORLD = None
CACHE_CONVERSION = None
if not RUN_REAL_L32_CONVERGENCE_RESOLUTION:
    from jlens.mmpilot.mock import MockWorld, build_mock_dataset

    MOCK_WORLD = MockWorld({
        "bus": ("bus", "buses"),
        "cat": ("cat", "cats"),
        "clock": ("clock", "clocks"),
        "dog": ("dog", "dogs"),
        "pizza": ("pizza", "pizzas"),
        "zebra": ("zebra", "zebras"),
    })
    # Five times the images the design needs, so the simulated completed run
    # can genuinely spend half the PHOTOGRAPHS and an independent population
    # still exists afterwards. Sizing this by groups instead of photographs is
    # what makes a mock population look adequate and then collapse: excluding a
    # photograph excludes all of its sibling captions with it.
    if not (SCRATCH / "data" / "spokencoco_manifest.json").is_file():
        build_mock_dataset(
            SCRATCH / "data",
            world=MOCK_WORLD,
            images_per_concept=5 * (N_TRAIN_POSITIVE_IMAGES + N_TEST_POSITIVE_IMAGES),
            negative_images=(N_TRAIN_NEGATIVE_IMAGES + N_TEST_NEGATIVE_IMAGES) * 6,
            captions_per_image=2,
            layout="sibling",
            visual_only_images=1,
        )
    MANIFEST_PATH = str(SCRATCH / "data" / "spokencoco_manifest.json")
    IMAGE_MEDIA_ROOT = str(SCRATCH / "data" / "coco")
    AUDIO_MEDIA_ROOT = str(SCRATCH / "data" / "SpokenCOCO")
    CACHED_EXPANDED_MANIFEST = str(SCRATCH / "cache" / "expanded_manifest.json")
    EXPECTED_CACHED_GROUP_COUNT = None
    print(f"MOCK dataset ready at {SCRATCH / 'data'}")

ORIGINAL_MANIFEST_CHECKSUM = manifest_module.manifest_checksum(MANIFEST_PATH)
IMAGE_ROOTS = [Path(IMAGE_MEDIA_ROOT)]
AUDIO_ROOTS = [Path(AUDIO_MEDIA_ROOT)]

SEARCH_ROOTS = sorted({str(r) for r in IMAGE_ROOTS + AUDIO_ROOTS if r.is_dir()})
if RUN_REAL_L32_CONVERGENCE_RESOLUTION:
    SEARCH_ROOTS = sorted({
        str(c) for c in (SPOKENCOCO_BASE_ROOT, IMAGE_MEDIA_ROOT, AUDIO_MEDIA_ROOT,
                         DOWNLOAD_CACHE)
        if Path(c).is_dir()
    })

# Discovery + checksums only. This is the cheap half; it is what the
# compatibility clauses are computed from and it never joins anything.
DISCOVERED_SOURCES = expansion_module.discover_metadata_sources(
    SEARCH_ROOTS, exclude=[MANIFEST_PATH], max_files=40, max_depth=3
)
ANNOTATION_SOURCES = [
    s for s in DISCOVERED_SOURCES if s.source_kind == "coco_object_annotation"
]
SYNC_SOURCES = [s for s in DISCOVERED_SOURCES if s.usable]
EXPECTED_SOURCE_CHECKSUMS = {s.path: s.checksum for s in SYNC_SOURCES}
print(f"metadata sources discovered {len(DISCOVERED_SOURCES)} "
      f"({len(ANNOTATION_SOURCES)} annotation)")

if not RUN_REAL_L32_CONVERGENCE_RESOLUTION and not Path(
    CACHED_EXPANDED_MANIFEST
).is_file():
    from jlens.mmpilot import evidence as evidence_module
    from jlens.mmpilot.concepts import discover_category_universe

    _universe = discover_category_universe(ANNOTATION_SOURCES)
    _evidence_config = evidence_module.config_from_specs(_universe.specs)
    CACHE_CONVERSION = {
        "converter": "jlens.mmpilot.expansion.build_expanded_manifest",
        "search_roots": SEARCH_ROOTS,
        "evidence_rule": "visual_annotation_AND_caption_lexicon",
        "evidence_lexicon_hash": _evidence_config.lexicon_hash,
        "reads_only": True, "media_redownloaded": False, "audio_transcribed": False,
    }
    _raw = json.loads(Path(MANIFEST_PATH).read_text(encoding="utf-8"))
    _baseline = manifest_module.normalize_manifest(
        _raw,
        manifest_module.inspect_manifest(_raw),
        image_roots=IMAGE_ROOTS,
        audio_roots=AUDIO_ROOTS,
        source_checksum=ORIGINAL_MANIFEST_CHECKSUM,
        min_complete_groups=1,
    )
    _expansion = expansion_module.build_expanded_manifest(
        SYNC_SOURCES,
        image_roots=IMAGE_ROOTS,
        annotation_sources=ANNOTATION_SOURCES,
        candidate_concepts=_universe.lexicon(),
        max_metadata_records=20000,
        audio_roots=AUDIO_ROOTS,
        baseline_groups=_baseline.groups,
    )
    Path(CACHED_EXPANDED_MANIFEST).parent.mkdir(parents=True, exist_ok=True)
    expansion_module.persist_expanded_manifest(
        CACHED_EXPANDED_MANIFEST,
        _expansion,
        original_checksum=ORIGINAL_MANIFEST_CHECKSUM,
        conversion=CACHE_CONVERSION,
    )
    print(f"MOCK cache written to {CACHED_EXPANDED_MANIFEST}")
'''
)

code(
    '''
# 7b. Load the cached evidence join DIRECTLY. build_expanded_manifest is not
# called on this path, and the flag below records that as a fact rather than a
# claim.
from jlens.mmpilot import evidence as evidence_module
from jlens.mmpilot.concepts import universe_from_concept_annotations
from jlens.mmpilot.expansion import (
    ExpandedManifestIncompatible,
    load_expanded_manifest,
)
from jlens.mmpilot.store import payload_checksum

if CACHE_CONVERSION is None:
    from jlens.mmpilot.concepts import discover_category_universe

    _u = discover_category_universe(ANNOTATION_SOURCES)
    CACHE_CONVERSION = {
        "converter": "jlens.mmpilot.expansion.build_expanded_manifest",
        "search_roots": SEARCH_ROOTS,
        "evidence_rule": "visual_annotation_AND_caption_lexicon",
        "evidence_lexicon_hash": evidence_module.config_from_specs(
            _u.specs
        ).lexicon_hash,
        "reads_only": True, "media_redownloaded": False, "audio_transcribed": False,
    }

CACHE_LOAD = {"build_expanded_manifest_called": False}
try:
    _cached, _record = load_expanded_manifest(
        CACHED_EXPANDED_MANIFEST,
        original_checksum=ORIGINAL_MANIFEST_CHECKSUM,
        expected_sources=EXPECTED_SOURCE_CHECKSUMS,
        conversion=CACHE_CONVERSION,
        expected_group_count=EXPECTED_CACHED_GROUP_COUNT,
        expected_lexicon_hash=CACHE_CONVERSION["evidence_lexicon_hash"],
    )
except ExpandedManifestIncompatible as error:
    if not ALLOW_MANIFEST_REBUILD:
        raise RuntimeError(
            f"{error}\\n\\nThe cache was NOT rebuilt. Rebuilding the "
            "125,198-group join takes ~25 minutes and would produce a manifest "
            "this study has no reason to believe is the one the completed runs "
            "used. Fix the configured path, or set ALLOW_MANIFEST_REBUILD=True "
            "deliberately."
        ) from error
    raise

GROUPS = _cached["groups"]
DERIVED_MANIFEST_CHECKSUM = _record["manifest_file_checksum"]
CACHE_LOAD.update({
    "path": _record["path"],
    "compatible": _record["compatible"],
    "clauses": _record["clauses"],
    "n_groups": len(GROUPS),
    "manifest_file_checksum": DERIVED_MANIFEST_CHECKSUM,
    "schema_version": _cached.get("schema_version"),
    "evidence_lexicon_hash": CACHE_CONVERSION["evidence_lexicon_hash"],
    "expected_group_count": EXPECTED_CACHED_GROUP_COUNT,
})

# The category universe comes out of the cache's own concept_annotations.
UNIVERSE = universe_from_concept_annotations(GROUPS)
EVIDENCE_CONFIG = evidence_module.config_from_specs(UNIVERSE.specs)
CONCEPT_CANDIDATES = UNIVERSE.lexicon()
CACHE_LOAD["universe_source"] = UNIVERSE.sources[0]["path"]
CACHE_LOAD["n_categories_recovered"] = len(UNIVERSE.categories)
CACHE_LOAD["category_ids_available"] = UNIVERSE.sources[0]["category_ids_available"]

print(f"cache          {CACHE_LOAD['path']}")
print(f"schema         {CACHE_LOAD['schema_version']}")
print(f"groups         {CACHE_LOAD['n_groups']:,}")
print(f"file checksum  {DERIVED_MANIFEST_CHECKSUM}")
print(f"lexicon hash   {CACHE_LOAD['evidence_lexicon_hash']}")
print(f"categories     {CACHE_LOAD['n_categories_recovered']} recovered from "
      f"persisted concept_annotations")
print(f"category ids   available={CACHE_LOAD['category_ids_available']} "
      "(not serialized in an expanded manifest; universe_hash is NOT comparable "
      "to a discovery-path hash)")
print(f"build_expanded_manifest called on this path: "
      f"{CACHE_LOAD['build_expanded_manifest_called']}")
'''
)

# ================================= 8. the fresh, verified-independent population

markdown(
    """
## 8. The fresh population — preprocessed once, resumable, **proven** afterwards

This section is CPU work over a Google Drive mount, and it is the section that
used to take four hours with nothing to show for an interruption. It is now a
**checkpointed preparation** stored under a deterministic cache directory keyed
by a pre-model fingerprint. Stop the runtime whenever you like: at most one
bounded batch of ≤25 files (or ≤30 seconds) is repeated, and the scan never
restarts from file zero while a valid checkpoint exists.

### 8a. The preparation fingerprint and where its cache lives

The cache path is a function of every input upstream of the model that can
change which media are excluded or which are selected — the harvest protocol,
the completed runs' basenames, their own fingerprints, their summary checksums,
the cached expanded manifest's checksum and schema, the evidence lexicon hash,
the frozen concepts, the sample-size rule and plan digest, the selection
seed/profile/algorithm, the identity families, the source-artifact strategy and
the fallback rule. Change any one of them and you get a different cache
directory, never a silently reused one.

No timestamped run directory is involved. A CPU session and an L4 session that
agree on those inputs land in the same cache without either having to name a run.

### 8b. Harvest the identities this population must avoid — resumably

**One** enumeration of exactly the identity-bearing artifacts, reused for source
integrity, identity harvesting and the after-the-fact immutability check. The old
code took a whole-tree name/size/mtime digest and *then* walked the tree again to
read identities.

Sources are chosen minimally and the choice is **proven**, not assumed: a bulk
population manifest when a run wrote one, otherwise the activation unit family,
whose superset property over the capability, J-space, direction, intervention and
readout families is stated in the proof *and checked* against each run's own
`split_provenance.json`. Capability units are never treated as a population
source — they are capped per concept and cover positives only. A shortfall
escalates to a separately checkpointed fallback scan of the skipped families
rather than narrowing the exclusion set.

Every batch is written as one atomically replaced, checksummed gzip shard
*before* the cursor advances. A torn shard is quarantined and only its own batch
is recomputed. Progress prints at least every 30 seconds with the run, the
family, files done, the current shard, identities recovered, elapsed time and a
remaining estimate from measured throughput — and says whether the work is being
computed or reused.

### 8c. Resolve the media the unit files do not store

Unit files carry ids, not recordings or captions. A `group_id` is a hash of
exactly `image_id|caption|audio`, so mapping the excluded ids back through the
manifest recovers both — and every sibling caption of an excluded photograph is
excluded with it. The pool is then filtered **before** any selection rule runs.

### 8d. The frozen concepts, checked individually — never re-ranked

The six candidates and the three focal concepts were fixed by the completed
studies. They are set **directly**, in frozen order. The independent pool's
ranking is computed and printed because it is informative, and it is
**descriptive only**: removing the photographs a completed run spent changes the
ranking legitimately without making any frozen concept infeasible, so refusing
unless the fresh top six equals the frozen six would refuse for the wrong reason.

What is checked instead is each frozen concept against each predeclared
requirement — distinct images, synchronized groups, source positives, held-out
positives, matched negatives. A concept that genuinely falls short is a
**refusal** with its exact shortfall printed. No concept is ever substituted,
before or after any result.

### 8e. Build the population, then prove it

Disjointness is verified over the population that was actually built, by a
different predicate than the one that did the filtering: a proof derived from the
filter proves only that the filter agrees with itself.

### 8f. Persist everything, so a fresh process needs none of this memory

`GROUPS`, `EXCLUSION`, `POOL`, the ranking, the feasibility record, the selected
population, its provenance and every digest are written to the preparation cache
and reloaded — and re-verified against their digests — by the GPU session.
"""
)

code(
    '''
# 8a. Resolve the deterministic preparation cache. NO source file is read here.
from datetime import datetime, timezone

from jlens.mmpilot import prep_cache as prep
from jlens.mmpilot.l32_resolution import (
    POPULATION_SELECTION_VERSION,
    RESOLUTION_RUN_PREFIX,
    SAMPLE_SIZE_RULE_VERSION,
    assert_fresh_run_namespace,
    independent_pool,
    resolve_excluded_media,
)

if not RUN_REAL_L32_CONVERGENCE_RESOLUTION:
    # A simulated completed run that really did spend media: half the
    # PHOTOGRAPHS, chosen deterministically by image id, with every one of their
    # groups. Spending half the *groups* instead would scatter the exclusion
    # across almost every photograph and leave a pool that cannot support any
    # design — which is the shape of the mistake, not a property of the study.
    from jlens.mmpilot.mock import build_mock_completed_run
    from jlens.mmpilot.selection import stable_rank

    _mock_completed = SCRATCH / "mock_completed_followup_run"
    # Ordered by a seeded stable rank, not by id: the mock dataset numbers its
    # photographs per concept, so taking the lowest half of the ids would spend
    # whole concepts and leave a pool that is short of concepts rather than
    # short of photographs. A hash rank spends half of *each* concept.
    _images = sorted(
        {str(g["image_id"]) for g in GROUPS},
        key=lambda image_id: stable_rank(image_id, "mock-completed-run"),
    )
    _spent_images = set(_images[: len(_images) // 2])
    _spent = [
        {**g, "split": "train" if index % 2 else "test"}
        for index, g in enumerate(
            sorted(
                (g for g in GROUPS if str(g["image_id"]) in _spent_images),
                key=lambda g: str(g["group_id"]),
            )
        )
    ]
    if not (_mock_completed / "units").is_dir():
        build_mock_completed_run(
            _mock_completed, _spent, layer=RESOLUTION_LAYER,
        )
    COMPLETED_RUN_DIRS = (str(_mock_completed),)
    PREP_CACHE_ROOT = str(SCRATCH / "prep_cache_root")

# What each completed run IS, for fingerprinting: its basename, the digest its
# own fingerprint.json records, and a checksum of each summary/report it wrote.
# Never its absolute path — a remounted Drive is not a different experiment.
COMPLETED_RUN_IDENTITIES = [
    prep.completed_run_identity(_dir) for _dir in COMPLETED_RUN_DIRS
]

PREPARATION_FINGERPRINT = prep.preparation_fingerprint(
    **prep.default_fingerprint_constants(),
    completed_run_basenames=[_e["run"] for _e in COMPLETED_RUN_IDENTITIES],
    completed_run_fingerprints=[
        _e["fingerprint_digest"] for _e in COMPLETED_RUN_IDENTITIES
    ],
    completed_summary_checksums=[
        _e["summary_checksums"] for _e in COMPLETED_RUN_IDENTITIES
    ],
    cached_expanded_manifest_checksum=DERIVED_MANIFEST_CHECKSUM,
    cache_schema_version=CACHE_LOAD["schema_version"],
    evidence_lexicon_hash=CACHE_LOAD["evidence_lexicon_hash"],
    frozen_selected_concepts=list(FROZEN_SELECTED_CONCEPTS),
    frozen_focal_concepts=list(FROZEN_FOCAL_CONCEPTS),
    sample_size_rule_version=SAMPLE_SIZE_RULE_VERSION,
    sample_size_plan_digest=SAMPLE_PLAN_RECORD["plan_digest"],
    selection_algorithm_version=POPULATION_SELECTION_VERSION,
    selection_seed=SPLIT_SEED,
    selection_profile_version=PROFILE.version,
    n_train_positive_images=N_TRAIN_POSITIVE_IMAGES,
    n_test_positive_images=N_TEST_POSITIVE_IMAGES,
    n_train_negative_images=N_TRAIN_NEGATIVE_IMAGES,
    n_test_negative_images=N_TEST_NEGATIVE_IMAGES,
)
PREP_DIR = prep.preparation_cache_dir(PREP_CACHE_ROOT, PREPARATION_FINGERPRINT)
PREP_COMPLETE_ON_ENTRY = prep.preparation_is_complete(PREP_DIR)
PROGRESS = prep.ProgressReporter(interval=PREP_PROGRESS_SECONDS)

print("PREPARATION CACHE")
print(f"  preparation version  {prep.PREPARATION_VERSION}")
print(f"  preparation digest   {PREPARATION_FINGERPRINT['preparation_digest']}")
print(f"  cache directory      {PREP_DIR}")
print(f"  already complete     {PREP_COMPLETE_ON_ENTRY is not None}")
for _entry in COMPLETED_RUN_IDENTITIES:
    print(f"  avoided run          {_entry['run']}  "
          f"fingerprint {str(_entry['fingerprint_digest'])[:23]}...  "
          f"summaries {len(_entry['summary_checksums'])}")

# Runtime safeguard. Preprocessing is Drive I/O on thousands of small files; a
# GPU makes none of it faster and an L4 hour spent here is an L4 hour wasted.
print()
print(f"CUDA present: {torch.cuda.is_available()}")
if PREP_COMPLETE_ON_ENTRY is None and torch.cuda.is_available():
    print("!" * 72)
    print("PREPROCESSING IS NOT COMPLETE AND THIS RUNTIME HAS A GPU.")
    print("A GPU provides NO benefit to section 8: it is Drive I/O over many")
    print("small files. Recommended: stop this runtime, switch to a free CPU")
    print("runtime, set PREPROCESSING_ONLY = True and let section 8 finish and")
    print("checkpoint. Section 11 will refuse to load Gemma until it has.")
    print("Everything section 8 completes here is still durable and reusable —")
    print("nothing is lost either way, only GPU time.")
    print("!" * 72)
'''
)

code(
    '''
# 8b. Harvest the exclusion identities, resumably. THIS is the long cell, and it
# is the one that checkpoints. Stop it whenever you like.
PREP = prep.run_exclusion_preparation(
    PREP_DIR,
    COMPLETED_RUN_DIRS,
    fingerprint=PREPARATION_FINGERPRINT,
    batch_files=PREP_BATCH_FILES,
    checkpoint_seconds=PREP_CHECKPOINT_SECONDS,
    progress=PROGRESS,
    protected_prefixes=PROTECTED_RUN_PREFIXES,
)
EXCLUSION = PREP["exclusion"]
COMPLETENESS = prep.assert_complete(PREP["completeness"])
SOURCE_FAMILIES = PREP["families_by_run"]

print()
print("EXCLUSION SET — harvested from the completed runs' own artifacts")
for _entry in EXCLUSION.sources:
    print(f"  {_entry['run_dir']}")
    print(f"    files read {_entry['n_files_read']:>6}   "
          f"identities {_entry['n_identities']:>6}   "
          f"unreadable {len(_entry['unreadable'])}")
for _name, _count in sorted(EXCLUSION.counts().items()):
    print(f"  {_name:18s} {_count:>7,}")
print(f"  exclusion digest   {EXCLUSION.digest}")
print(f"  content digest     {COMPLETENESS['content_digest']}")
print()
print("COMPLETENESS PROOF — the exclusion set covers the whole spent population")
for _row in COMPLETENESS["runs"]:
    print(f"  {_row['run']}")
    print(f"    strategy          {_row['strategy']}")
    print(f"    expected groups   {_row['expected_group_count']} "
          f"(from {_row['expected_from']})")
    print(f"    groups recovered  {_row['group_ids_recovered']}")
    print(f"    images recovered  {_row['image_ids_recovered']} "
          f"(expected {_row['expected_image_count']})")
    print(f"    sources read      {_row['source_artifacts']}")
    print(f"    families skipped  {_row['families_skipped']}")
    print(f"    files read        {_row['n_files_read']}")
    print(f"    complete          {_row['complete']}  shortfall {_row['shortfall']}")
print(f"  fallback scan required: {COMPLETENESS['fallback_required']}")
print(f"  missing/invalid units:  {COMPLETENESS['n_missing_or_invalid_units']}")
print()
print(f"  files computed this session: {PREP['files_computed_this_session']}")
print(f"  files reused from Drive:     {PREP['files_reused_from_drive']}")
print()
print("WHY THE SKIPPED FAMILIES CANNOT HOLD AN IDENTITY THE READ ONES MISSED:")
print(f"  {COMPLETENESS['why_skipped_families_cannot_add_identities']}")
'''
)

code(
    '''
# 8c. Recover the recordings and captions behind the excluded group ids, then
# filter the pool BEFORE any selection rule runs.
from jlens.mmpilot.store import payload_checksum

MEDIA_RESOLUTION = resolve_excluded_media(EXCLUSION, GROUPS)
print(f"excluded group ids           {MEDIA_RESOLUTION['n_excluded_group_ids']:,}")
print(f"resolved in this manifest    {MEDIA_RESOLUTION['n_resolved_in_manifest']:,}")
print(f"unresolved (still excluded)  {MEDIA_RESOLUTION['n_unresolved']:,}")
print(f"audio paths added            {MEDIA_RESOLUTION['audio_paths_added']:,}")
print(f"captions added               {MEDIA_RESOLUTION['captions_added']:,}")
print(f"  {MEDIA_RESOLUTION['sibling_expansion_note']}")

# The pool is a deterministic function of GROUPS and EXCLUSION, so it is
# recomputed in every process rather than deserialized — and its digest is
# compared against the one preprocessing recorded. Loading a pool would make
# the file the authority; recomputing and checking makes the identities the
# authority and catches a cache that has drifted from the manifest.
POOL, POOL_RECORD = independent_pool(GROUPS, EXCLUSION)
POOL_DIGEST = payload_checksum(sorted(str(_g["group_id"]) for _g in POOL))
POOL_RECORD["pool_digest"] = POOL_DIGEST

prep.atomic_write_json(
    PREP_DIR / "media_resolution.json",
    MEDIA_RESOLUTION,
    protected_prefixes=PROTECTED_RUN_PREFIXES,
)
prep.atomic_write_json(
    PREP_DIR / "independent_pool.json",
    POOL_RECORD,
    protected_prefixes=PROTECTED_RUN_PREFIXES,
)

print()
print(f"groups available             {POOL_RECORD['n_groups_available']:,}")
print(f"groups excluded              {POOL_RECORD['n_groups_excluded']:,}")
print(f"independent pool             {POOL_RECORD['n_groups_in_independent_pool']:,}")
print(f"distinct images in pool      {POOL_RECORD['n_distinct_images_in_pool']:,}")
print(f"pool digest                  {POOL_DIGEST}")
print(f"  by identity: {POOL_RECORD['excluded_by_identity']}")
if not POOL:
    raise RuntimeError(
        "the independent pool is empty: every synchronized group in the "
        "manifest was already spent by a completed run. The required "
        "independent population cannot be built and this study refuses rather "
        "than reusing media."
    )
'''
)

code(
    '''
# 8d. Rank the pool DESCRIPTIVELY, then check the FROZEN concepts one by one.
#
# The ranking decides nothing here. Excluding the photographs a completed run
# spent removes coverage unevenly, so the fresh ranking can reorder while all
# six frozen concepts stay comfortably feasible; refusing unless the fresh top
# six equals the frozen six would refuse for the wrong reason. What is checked
# is each frozen concept against each predeclared requirement.
from jlens.mmpilot.l32_resolution import (
    assert_frozen_concepts_feasible,
    format_frozen_feasibility,
    frozen_concept_feasibility,
    ranking_digest,
)
from jlens.mmpilot.selection import select_focal_concepts, unrelated_control_assignment

PREPARED = prep.load_prepared_selection(PREP_DIR)
REQUIREMENTS = expansion_module.ConceptRequirements(
    min_distinct_images=N_TRAIN_POSITIVE_IMAGES + N_TEST_POSITIVE_IMAGES,
    min_groups=N_TRAIN_POSITIVE_IMAGES + N_TEST_POSITIVE_IMAGES,
    min_train_positives=N_TRAIN_POSITIVE_IMAGES,
    min_test_positives=N_TEST_POSITIVE_IMAGES,
)
EVIDENCE_INDEX = None
SELECTION_REUSED = False

if PREPARED is not None and PREPARED.get("pool_digest") == POOL_DIGEST:
    # A fresh process reloads the ranking and feasibility record instead of
    # rebuilding the evidence index over the whole pool. The pool digest above
    # is what makes that safe: a different pool cannot reach this branch.
    RANKING = PREPARED["ranking"]
    FEASIBILITY = PREPARED["feasibility"]
    SELECTION_REUSED = True
    print(f"reusing the prepared ranking and feasibility from {PREP_DIR}")
else:
    EVIDENCE_INDEX = evidence_module.build_evidence_index(
        POOL, tuple(CONCEPT_CANDIDATES), EVIDENCE_CONFIG
    )
    RANKING = expansion_module.rank_concepts(
        POOL,
        CONCEPT_CANDIDATES,
        requirements=REQUIREMENTS,
        groups_per_concept=N_TRAIN_POSITIVE_IMAGES + N_TEST_POSITIVE_IMAGES,
        max_groups_per_image=PROFILE.max_groups_per_image,
        seed=SPLIT_SEED,
        evidence_config=EVIDENCE_CONFIG,
        profile=PROFILE,
        evidence_index=EVIDENCE_INDEX,
    )
    FEASIBILITY = None

RANKED_CONCEPTS = [row["concept"] for row in RANKING]
RANKING_DIGEST = ranking_digest(RANKING)

if RUN_REAL_L32_CONVERGENCE_RESOLUTION:
    # Set directly, in frozen order. Never re-derived from this pool.
    SELECTED_NAMES = list(SELECTED_CONCEPTS_EXPECTED)
    FOCAL_CONCEPTS = list(FOCAL_CONCEPTS_EXPECTED)
    NON_FOCAL_CONCEPTS = [c for c in SELECTED_NAMES if c not in FOCAL_CONCEPTS]
else:
    # MOCK has its own six-concept world; the frozen names do not exist in it,
    # so MOCK exercises the selection machinery and the real path exercises the
    # freeze. Neither branch ever substitutes a concept after a result.
    SELECTED_NAMES = expansion_module.select_concepts(
        RANKING, n_concepts=6, max_concepts=6, requirements=REQUIREMENTS,
    )
    FOCAL_CONCEPTS, NON_FOCAL_CONCEPTS = select_focal_concepts(
        SELECTED_NAMES, n_focal=3
    )

if FEASIBILITY is None or FEASIBILITY["frozen_selected_concepts"] != list(
    SELECTED_NAMES
):
    # Recomputed rather than trusted whenever the persisted record is not about
    # exactly these concepts. The check costs nothing and it is the difference
    # between reusing a result and inheriting one.
    FEASIBILITY = frozen_concept_feasibility(
        RANKING,
        concepts=SELECTED_NAMES,
        focal=FOCAL_CONCEPTS,
        requirements=REQUIREMENTS.to_dict(),
    )
assert_frozen_concepts_feasible(FEASIBILITY)
FEASIBILITY_DIGEST = FEASIBILITY["feasibility_digest"]

UNRELATED_CONTROLS = unrelated_control_assignment(FOCAL_CONCEPTS, NON_FOCAL_CONCEPTS)
CONFIG.concepts = tuple(SELECTED_NAMES)
CONFIG.causal_concepts = tuple(FOCAL_CONCEPTS)

print(format_frozen_feasibility(FEASIBILITY))
print()
print("SELECTION — fixed before any model result exists")
print(f"  selected (frozen order)  {SELECTED_NAMES}")
print(f"  focal concepts           {FOCAL_CONCEPTS}")
print(f"  non-focal (controls)     {NON_FOCAL_CONCEPTS}")
for _focal, _control in sorted(UNRELATED_CONTROLS.items()):
    print(f"    {_focal:12s} -> {_control}")
print(f"  ranking digest           {RANKING_DIGEST}")
print(f"  feasibility digest       {FEASIBILITY_DIGEST}")
print(f"  concepts substituted     False")
'''
)

code(
    '''
# 8e. Build the population, then PROVE it is independent and unreplicated.
from jlens.mmpilot.l32_resolution import (
    assert_one_unit_per_photograph,
    audit_population_disjointness,
    selection_digest,
)

if SELECTION_REUSED and PREPARED is not None:
    SUBSET = PREPARED["subset"]
    print(f"reusing the prepared population from {PREP_DIR}")
else:
    if EVIDENCE_INDEX is None:
        EVIDENCE_INDEX = evidence_module.build_evidence_index(
            POOL, tuple(CONCEPT_CANDIDATES), EVIDENCE_CONFIG
        )
    SUBSET = manifest_module.build_subset(
        POOL,
        {name: CONCEPT_CANDIDATES[name] for name in SELECTED_NAMES},
        groups_per_concept=N_TRAIN_POSITIVE_IMAGES + N_TEST_POSITIVE_IMAGES,
        negatives_per_concept=N_TRAIN_NEGATIVE_IMAGES + N_TEST_NEGATIVE_IMAGES,
        seed=SPLIT_SEED,
        evidence_config=EVIDENCE_CONFIG,
        profile=PROFILE,
        evidence_index=EVIDENCE_INDEX,
    )
LEAKAGE = manifest_module.check_split_leakage(SUBSET)
if not LEAKAGE["ok"]:
    raise RuntimeError(f"split leakage detected, refusing to continue: {LEAKAGE}")

# Recomputed in EVERY process, never loaded: the disjointness proof is the
# study's central claim, and a loaded proof proves only that a file says so.
DISJOINTNESS = audit_population_disjointness(SUBSET, EXCLUSION, require=True)
PSEUDOREPLICATION = assert_one_unit_per_photograph(SUBSET)
POPULATION_DIGEST = selection_digest(SUBSET)

_train = SUBSET["splits"]["train"]
_test = SUBSET["splits"]["test"]
_all_rows = _train + _test

print("INDEPENDENT POPULATION")
print(f"  synchronized groups   {PSEUDOREPLICATION['n_units']}")
print(f"  distinct images       {PSEUDOREPLICATION['n_distinct_images']}")
print(f"  distinct recordings   {PSEUDOREPLICATION['n_distinct_recordings']}")
print(f"  one group per image   {PSEUDOREPLICATION['one_group_per_image']}")
print(f"  one recording/unit    {PSEUDOREPLICATION['one_recording_per_unit']}")
print()
print("DISJOINTNESS FROM THE COMPLETED RUN — verified, not assumed")
for _family in DISJOINTNESS["families_checked"]:
    print(f"  {_family:14s} population {DISJOINTNESS['population_counts'][_family]:>6}"
          f"   excluded {DISJOINTNESS['exclusion_counts'][_family]:>7}"
          f"   overlap {DISJOINTNESS['n_overlaps'][_family]:>4}")
print(f"  disjoint: {DISJOINTNESS['disjoint']}")
print(f"  population digest {POPULATION_DIGEST}")
print()
for _concept in SELECTED_NAMES:
    _tr = len({r["image_id"] for r in _train if r["concept"] == _concept})
    _te = len({r["image_id"] for r in _test if r["concept"] == _concept})
    print(f"    {_concept:12s} source images {_tr}  held-out images {_te}")

SPLIT_PROVENANCE = {
    "seed": SPLIT_SEED,
    "profile": PROFILE.to_dict(),
    "selection_algorithm": PROFILE.representative_selection,
    "selected_concepts": list(SELECTED_NAMES),
    "focal_concepts": list(FOCAL_CONCEPTS),
    "unrelated_controls": dict(sorted(UNRELATED_CONTROLS.items())),
    "n_groups": PSEUDOREPLICATION["n_units"],
    "n_distinct_images": PSEUDOREPLICATION["n_distinct_images"],
    "n_distinct_recordings": PSEUDOREPLICATION["n_distinct_recordings"],
    "leakage": LEAKAGE,
    "population_digest": POPULATION_DIGEST,
}
SPLIT_PROVENANCE_CHECKSUM = payload_checksum(SPLIT_PROVENANCE)
print(f"\\nsplit provenance checksum {SPLIT_PROVENANCE_CHECKSUM}")
'''
)

code(
    '''
# 8f. Determinism, persistence, and the end of preprocessing.
#
# Selection has to be a function of the identities, not of the order the cache
# happened to list them in. The permutation replay re-runs the whole selection
# on a shuffled pool and compares the digests — the only way "deterministic
# selection" is checked rather than asserted. It is done once, during
# preprocessing, and its result is persisted with everything else.
import random as _random

if SELECTION_REUSED and PREPARED is not None:
    SELECTION_DETERMINISM = PREPARED["selection_determinism"]
else:
    _shuffled = list(POOL)
    _random.Random(20260809).shuffle(_shuffled)
    _shuffled_index = evidence_module.build_evidence_index(
        _shuffled, tuple(CONCEPT_CANDIDATES), EVIDENCE_CONFIG
    )
    _replay = manifest_module.build_subset(
        _shuffled,
        {name: CONCEPT_CANDIDATES[name] for name in SELECTED_NAMES},
        groups_per_concept=N_TRAIN_POSITIVE_IMAGES + N_TEST_POSITIVE_IMAGES,
        negatives_per_concept=N_TRAIN_NEGATIVE_IMAGES + N_TEST_NEGATIVE_IMAGES,
        seed=SPLIT_SEED,
        evidence_config=EVIDENCE_CONFIG,
        profile=PROFILE,
        evidence_index=_shuffled_index,
    )
    SELECTION_DETERMINISM = {
        "population_digest": POPULATION_DIGEST,
        "permuted_pool_digest": selection_digest(_replay),
        "deterministic": selection_digest(_replay) == POPULATION_DIGEST,
        "check": "identical selection under a permuted manifest order",
    }
print(f"selection determinism: {SELECTION_DETERMINISM['deterministic']}")
if not SELECTION_DETERMINISM["deterministic"]:
    raise RuntimeError(
        "the selected population depends on the order the manifest listed its "
        "groups in, so it is not reproducible from the identities alone. "
        f"{SELECTION_DETERMINISM}"
    )

PREPARED_SELECTION = prep.save_prepared_selection(
    PREP_DIR,
    {
        "preparation_digest": PREPARATION_FINGERPRINT["preparation_digest"],
        "pool_digest": POOL_DIGEST,
        "pool_record": POOL_RECORD,
        "ranking": RANKING,
        "ranking_digest": RANKING_DIGEST,
        "feasibility": FEASIBILITY,
        "requirements": REQUIREMENTS.to_dict(),
        "selected_concepts": list(SELECTED_NAMES),
        "focal_concepts": list(FOCAL_CONCEPTS),
        "non_focal_concepts": list(NON_FOCAL_CONCEPTS),
        "unrelated_controls": dict(sorted(UNRELATED_CONTROLS.items())),
        "subset": SUBSET,
        "population_digest": POPULATION_DIGEST,
        "split_provenance": SPLIT_PROVENANCE,
        "split_provenance_checksum": SPLIT_PROVENANCE_CHECKSUM,
        "selection_determinism": SELECTION_DETERMINISM,
        "media_resolution": MEDIA_RESOLUTION,
    },
    protected_prefixes=PROTECTED_RUN_PREFIXES,
)

# The completion marker is also the immutability check: recomputing this
# preparation from the same inputs and getting a different digest is a refusal,
# not an overwrite.
PREPARATION_COMPLETE = prep.finalize_preparation(
    PREP_DIR,
    {
        "preparation_digest": PREPARATION_FINGERPRINT["preparation_digest"],
        "exclusion_digest": EXCLUSION.digest,
        "population_digest": POPULATION_DIGEST,
        "pool_digest": POOL_DIGEST,
        "ranking_digest": RANKING_DIGEST,
        "frozen_concept_feasibility_digest": FEASIBILITY_DIGEST,
        "completeness_complete": COMPLETENESS["complete"],
        "content_digest": COMPLETENESS["content_digest"],
    },
    protected_prefixes=PROTECTED_RUN_PREFIXES,
)
PREPROCESSING_REPORT = prep.render_preprocessing_report({
    "cache_dir": str(PREP_DIR),
    "preparation_digest": PREPARATION_FINGERPRINT["preparation_digest"],
    "exclusion_digest": EXCLUSION.digest,
    "completeness": COMPLETENESS,
    "files_computed_this_session": PREP["files_computed_this_session"],
    "files_reused_from_drive": PREP["files_reused_from_drive"],
    "batch_files": PREP_BATCH_FILES,
    "checkpoint_seconds": PREP_CHECKPOINT_SECONDS,
})
prep.atomic_write_text(
    PREP_DIR / "preprocessing_report.md",
    PREPROCESSING_REPORT,
    protected_prefixes=PROTECTED_RUN_PREFIXES,
)

print()
print("PREPROCESSING COMPLETE AND PERSISTED")
print(f"  cache directory   {PREP_DIR}")
for _name in sorted(p.name for p in PREP_DIR.iterdir()):
    print(f"    {_name}")
print(f"  exclusion digest  {PREPARATION_COMPLETE['exclusion_digest']}")
print(f"  population digest {PREPARATION_COMPLETE['population_digest']}")
print(f"  pool digest       {PREPARATION_COMPLETE['pool_digest']}")
print()
print("A fresh Python process reconstructs GROUPS, EXCLUSION, POOL, the ranking,")
print("the feasibility record, the subset, its provenance and every fingerprint")
print("from these artifacts alone. Nothing required lives only in this kernel.")
if PREPROCESSING_ONLY:
    print()
    print("=" * 72)
    print("PREPROCESSING_ONLY IS TRUE — STOP HERE.")
    print("Every cell below is a no-op in this session. Switch to an L4 runtime,")
    print("set PREPROCESSING_ONLY = False with the same scientific configuration,")
    print("and section 8 will load and verify this cache without re-reading a")
    print("single source unit.")
    print("=" * 72)
'''
)

# ================================================= 9. the open prompt

markdown(
    """
## 9. The open prompt — candidates scored externally, never shown

The model-visible question names **no candidate**. The six candidates exist only
in the external scorer, and section 9b audits every model-visible surface
(system text, question, evidence, transcript) for any candidate name or an
enumeration that would leak the option set.

This is the same protocol the completed follow-up ran
(`mmpilot.open_entity_identification.v1`), which is what makes the two
convergence measurements comparable in kind — and this study's population is
deliberately *not* the same, which is what makes it independent.
"""
)

code(
    '''
# 9. The open prompt, its leakage audit and its fingerprint.
from jlens.mmpilot.prompt_protocol import (
    CANDIDATE_SCORING_VERSION,
    CANDIDATE_VISIBILITY_RULE,
    DEFAULT_QUESTIONS,
    OPEN_ENTITY_IDENTIFICATION,
    Evidence,
    build_protocol_prompt,
    prompt_protocol_fingerprint,
)

if OPEN_PROMPT_PROTOCOL != OPEN_ENTITY_IDENTIFICATION:
    raise RuntimeError(
        f"the module's open protocol is {OPEN_PROMPT_PROTOCOL!r} but this "
        f"notebook built {OPEN_ENTITY_IDENTIFICATION!r}"
    )

OPEN_QUESTION = DEFAULT_QUESTIONS[OPEN_ENTITY_IDENTIFICATION]
print("THE MODEL-VISIBLE QUESTION, verbatim:")
print("-" * 72)
print(OPEN_QUESTION)
print("-" * 72)
print()
print(CANDIDATE_VISIBILITY_RULE)
print(f"candidate scoring: {CANDIDATE_SCORING_VERSION}")
'''
)

code(
    '''
# 9b. The candidate-visibility audit. A leak is a refusal, in every modality.
PROMPT_RECORDS = {}
PROMPT_FINGERPRINTS = {}
_evidence_by_modality = {
    "text": Evidence(modality="text", text="a photograph of the scene"),
    "image": Evidence(modality="image", media_reference="evidence/image"),
    "spoken_audio": Evidence(
        modality="spoken_audio", media_reference="evidence/audio"
    ),
}
for _modality, _evidence in _evidence_by_modality.items():
    _built = build_protocol_prompt(
        protocol=OPEN_ENTITY_IDENTIFICATION,
        evidence=_evidence,
        external_candidates=SELECTED_NAMES,
    )
    PROMPT_RECORDS[_modality] = _built
    PROMPT_FINGERPRINTS[_modality] = prompt_protocol_fingerprint(
        _built,
        model_revision=(
            MODEL_REVISION if RUN_REAL_L32_CONVERGENCE_RESOLUTION else "mock"
        ),
        processor_revision=(
            MODEL_REVISION if RUN_REAL_L32_CONVERGENCE_RESOLUTION else "mock"
        ),
        audio_protocol_fingerprint=AUDIO_PROTOCOL_FINGERPRINT_EXPECTED,
    )
    _visibility = _built.candidate_visibility
    print(f"  {_modality:13s} candidates_in_prompt="
          f"{_visibility['candidates_in_prompt']}  "
          f"prompt_hash={_built.prompt_hash[:20]}...  "
          f"leakage_passed={_built.leakage['passed']}")
    if _visibility["candidates_in_prompt"]:
        raise RuntimeError(
            f"the {_modality} prompt contains the candidate list; this study "
            "does not run a candidate-listed prompt under an open protocol"
        )

# Order must not move the prompt hash; the SET must move the fingerprint.
_reversed = build_protocol_prompt(
    protocol=OPEN_ENTITY_IDENTIFICATION,
    evidence=_evidence_by_modality["text"],
    external_candidates=list(reversed(SELECTED_NAMES)),
)
if _reversed.prompt_hash != PROMPT_RECORDS["text"].prompt_hash:
    raise RuntimeError(
        "reversing the candidate order changed the prompt hash — the candidates "
        "are reaching the prompt"
    )
_narrower = build_protocol_prompt(
    protocol=OPEN_ENTITY_IDENTIFICATION,
    evidence=_evidence_by_modality["text"],
    external_candidates=SELECTED_NAMES[:-1],
)
if (
    prompt_protocol_fingerprint(
        _narrower, model_revision="x", processor_revision="x"
    )["prompt_protocol_digest"]
    == prompt_protocol_fingerprint(
        PROMPT_RECORDS["text"], model_revision="x", processor_revision="x"
    )["prompt_protocol_digest"]
):
    raise RuntimeError(
        "a different candidate set produced the same fingerprint; a run that "
        "scored a different set would resume from this one"
    )
print("\\ncandidate-order invariance of the prompt hash: holds")
print("candidate-set sensitivity of the fingerprint: holds")

PROMPT_PROTOCOL_DIGEST = PROMPT_FINGERPRINTS["text"]["prompt_protocol_digest"]
print(f"prompt protocol digest {PROMPT_PROTOCOL_DIGEST}")
for _concept in SELECTED_NAMES:
    if _concept.lower() in OPEN_QUESTION.lower():
        raise RuntimeError(
            f"the candidate {_concept!r} appears in the model-visible question"
        )
'''
)

# ================================================= 10. budgets and gates

markdown(
    """
## 10. Pass budgets and the confirmation gates — printed before Gemma is loaded

Stage A is capability plus activation capture. The native readout itself costs
**no model passes**: it is applied to residuals already on disk, which is why an
independent convergence study is cheap and a causal replication is not.

Stage B's budget is printed here too, so a decision about it is made against a
number rather than a feeling — even though the Stage-B *gate* is not evaluated
until Stage A has returned.
"""
)

code(
    '''
# 10. Budgets, the derived runtime, and the confirmation gates. CPU only.
from jlens.mmpilot.l32_followup import derive_seconds_per_pass
from jlens.mmpilot.tri_modal import estimate_stage_passes, format_stage_budget

_n_groups = PSEUDOREPLICATION["n_units"]
_n_capability = min(
    _n_groups, CONFIG.max_capability_groups_per_concept * len(SELECTED_NAMES)
)


def _stage(stage):
    """One stage's passes. ``n_candidate_orders=1``: an open prompt names no
    candidates, so candidate-order invariance lives in the external scorer."""
    return estimate_stage_passes(
        n_concepts=len(SELECTED_NAMES),
        n_focal_concepts=len(FOCAL_CONCEPTS),
        modalities=MODALITIES,
        layers=(RESOLUTION_LAYER,),
        causal_layers=(RESOLUTION_LAYER,),
        n_total_groups=_n_groups,
        n_capability_groups=_n_capability,
        n_targets_per_cell=CONFIG.n_target_examples,
        alphas=CONFIG.alphas,
        stage=stage,
        n_candidate_orders=1,
        d_model=(
            EXPECT_D_MODEL if RUN_REAL_L32_CONVERGENCE_RESOLUTION else MOCK_D_MODEL
        ),
    )


BUDGET_STAGE_A = _stage("A")
BUDGET_STAGE_B = _stage("B")
print(format_stage_budget(BUDGET_STAGE_A))
print()
print(format_stage_budget(BUDGET_STAGE_B))
print()
# Measured once and cached with the rest of the preprocessing. The measurement
# stats every capability, activation and intervention unit of a completed run,
# which is cheap next to reading them but is still a Drive traversal — and it is
# a budget estimate, so paying for it once per preparation is enough.
_timing_path = PREP_DIR / "completed_run_timing.json"
if _timing_path.is_file():
    TIMING = json.loads(_timing_path.read_text(encoding="utf-8"))
    print(f"timing reused from {_timing_path}")
else:
    TIMING = derive_seconds_per_pass(COMPLETED_RUN_DIRS[0])
    prep.atomic_write_json(
        _timing_path, TIMING, protected_prefixes=PROTECTED_RUN_PREFIXES
    )
if TIMING.get("available"):
    _mid = float(TIMING["median_seconds_per_unit"])
    print(f"measured from the completed run's own unit-file inter-arrival times:")
    print(f"  Stage A  {BUDGET_STAGE_A.total_passes * _mid / 3600:.2f} h at "
          f"{_mid:.2f} s/pass")
    print(f"  Stage B  {BUDGET_STAGE_B.total_passes * _mid / 3600:.2f} h at "
          f"{_mid:.2f} s/pass")
    print(f"  {TIMING['caveat']}")
else:
    _lo, _hi = 0.4, 1.8
    print(f"the completed run's timing is not measurable from this runtime "
          f"({TIMING.get('reason', 'unavailable')}), so a RANGE is quoted:")
    print(f"  Stage A  {BUDGET_STAGE_A.total_passes * _lo / 3600:.2f}-"
          f"{BUDGET_STAGE_A.total_passes * _hi / 3600:.2f} h")
    print(f"  Stage B  {BUDGET_STAGE_B.total_passes * _lo / 3600:.2f}-"
          f"{BUDGET_STAGE_B.total_passes * _hi / 3600:.2f} h")
print()
print("the native direct readout itself costs 0 model passes: it is applied to")
print("residuals already stored by the activation stage.")
'''
)

code(
    '''
# 10b. The gates. Nothing below this cell runs until they are satisfied.
GATES = refresh_gates()

if RUN_REAL_L32_CONVERGENCE_RESOLUTION:
    if RUN_MODEL_STAGE and not CONFIRM_MODEL_LOAD:
        raise RuntimeError(
            "RUN_MODEL_STAGE is True but CONFIRM_MODEL_LOAD is False (~16 GB)."
        )
    if RUN_MODEL_STAGE and not CONFIRM_STAGE_A_BUDGET:
        raise RuntimeError(
            "RUN_MODEL_STAGE is True but CONFIRM_STAGE_A_BUDGET is False. "
            f"Stage A costs {BUDGET_STAGE_A.total_passes:,} model passes."
        )
if RUN_STAGE_B_CAUSAL_REPLICATION and not CONFIRM_STAGE_B_BUDGET:
    raise RuntimeError(
        "RUN_STAGE_B_CAUSAL_REPLICATION is True but CONFIRM_STAGE_B_BUDGET is "
        f"False. Stage B costs {BUDGET_STAGE_B.total_passes:,} additional "
        "model passes."
    )

print(f"model stage enabled  {MODEL_STAGE_ENABLED}")
print(f"stage A enabled      {STAGE_A_ENABLED}")
print(f"stage B requested    {STAGE_B_REQUESTED}")
print()
print("STAGE B is additionally gated on the Stage-A outcome (section 15). The")
print("switch above only says the passes are affordable; the rule says whether")
print("they are informative.")
'''
)

# ============================================== 11. model, media, invariance

markdown(
    """
## 11. Load the model, the retrying media loaders, and the invariance gate

**Drive I/O.** Both loaders read the whole file into a byte buffer with bounded
retries before decoding. A single `OSError: [Errno 5]` two hours into a run used
to take the session with it; `jlens.mmpilot.media_io` retries the read, records
every retry in a journal that is written into the run's artifacts, and refuses
only on errnos that waiting cannot fix.

**Invariance.** Capture must be a no-op and a zero-coefficient edit must
reproduce the clean scoring, *separately in each modality*. A gate that passed on
text says nothing about whether an image or audio forward pass survives the same
hook.
"""
)

code(
    '''
# 11. Preflight, backend, retrying media loaders, invariance.
from jlens.mmpilot.media_io import MEDIA_IO_VERSION, RetryJournal, drive_media_loaders

MODEL = None
BACKEND = None
AUDIO_PROTOCOL = None
MODEL_REVISION_USED = None
PROCESSOR_REVISION_USED = None
AVAILABLE_MODALITIES = []
INVARIANCE = None
MEDIA = None
MEDIA_RETRY_JOURNAL = RetryJournal()

# Gemma is not loaded until preprocessing is complete AND verified. A 16 GB
# download in front of a scan that has not finished is an L4 hour spent on Drive
# I/O, and a model loaded against a half-built population is worse than that.
if MODEL_STAGE_ENABLED and prep.preparation_is_complete(PREP_DIR) is None:
    raise RuntimeError(
        f"preprocessing is not complete in {PREP_DIR}, so the model is not "
        "loaded. Run section 8 to completion first — on a free CPU runtime "
        "with PREPROCESSING_ONLY = True, because a GPU makes Drive I/O over "
        "thousands of small files no faster. Everything section 8 has already "
        "finished is durable and will be reused."
    )

if not MODEL_STAGE_ENABLED:
    print("skipped: RUN_MODEL_STAGE and CONFIRM_MODEL_LOAD are not both True.")
    if PREPROCESSING_ONLY:
        print("(PREPROCESSING_ONLY is True, so every model gate is forced shut.)")
    print("Nothing below this cell computes a result.")
elif RUN_REAL_L32_CONVERGENCE_RESOLUTION:
    from jlens.mmpilot.preflight import preflight
    from jlens.mmpilot.real_backend import load_real_bundle
    from jlens.mmpilot.tri_modal import assert_audio_protocol

    PREFLIGHT = preflight(
        model_repo_id=MODEL_REPO_ID,
        model_revision=MODEL_REVISION,
        expect_n_layers=EXPECT_N_LAYERS,
        expect_d_model=EXPECT_D_MODEL,
        expect_vocab=EXPECT_VOCAB,
    )
    BUNDLE = load_real_bundle(
        model_repo_id=MODEL_REPO_ID,
        model_revision=MODEL_REVISION,
        layers=tuple(CONFIG.layers),
    )
    MODEL = BUNDLE.model
    BACKEND = BUNDLE.backend
    MODEL_REVISION_USED = BUNDLE.model_revision
    PROCESSOR_REVISION_USED = BUNDLE.processor_revision
    AUDIO_PROTOCOL = assert_audio_protocol(
        BUNDLE.audio_interface,
        expected_fingerprint=AUDIO_PROTOCOL_FINGERPRINT_EXPECTED,
    )
    MEDIA = drive_media_loaders(journal=MEDIA_RETRY_JOURNAL)
else:
    from jlens.mmpilot.mock import MockPilotBackend, load_mock_media

    BACKEND = MockPilotBackend(MOCK_WORLD, supports_audio=True)
    MODEL_REVISION_USED = "mock"
    PROCESSOR_REVISION_USED = "mock"
    AUDIO_PROTOCOL = {
        "protocol_version": AUDIO_PROTOCOL_VERSION_EXPECTED,
        "protocol_fingerprint": "sha256:mock-audio-protocol",
        "matches_expected_fingerprint": False,
        "note": "MOCK: the audio protocol fingerprint is checked only on the real path",
    }
    MEDIA = {
        "load_image": load_mock_media,
        "load_audio": lambda path: (load_mock_media(path), 16000),
    }
    print("MOCK backend: three modalities, no processor, no audio tower")

if BACKEND is not None:
    from jlens.mmpilot.pipeline import available_modalities

    AVAILABLE_MODALITIES, BLOCKED_MODALITIES = available_modalities(BACKEND, CONFIG)
    print(f"available modalities {AVAILABLE_MODALITIES}")
    print(f"blocked modalities   {BLOCKED_MODALITIES}")
    print(f"media io             {MEDIA_IO_VERSION}")
    if "spoken_audio" not in AVAILABLE_MODALITIES:
        raise RuntimeError(
            "spoken_audio is unavailable; this study's criterion requires it."
        )
'''
)

code(
    '''
# 11b. The per-modality invariance gate at the resolution layer.
from jlens.mmpilot.pipeline import build_condition_inputs
from jlens.mmpilot.tri_modal import run_invariance_by_modality

if BACKEND is None:
    print("skipped: no backend")
else:
    _probe_group = SUBSET["splits"]["test"][0]
    _probe_inputs = {
        modality: build_condition_inputs(
            BACKEND, _probe_group, modality, OPEN_QUESTION, MEDIA
        )
        for modality in AVAILABLE_MODALITIES
    }
    INVARIANCE = run_invariance_by_modality(
        BACKEND, _probe_inputs, list(CONFIG.layers)
    )
    print("INVARIANCE GATE (capture no-op + zero-coefficient edit)")
    for _modality, _entry in sorted(INVARIANCE["per_modality"].items()):
        print(f"  {_modality:13s} passed={_entry['passed']}")
    print(f"  overall passed {INVARIANCE['passed']}")
    if not INVARIANCE["passed"]:
        raise RuntimeError(
            "an invariance check failed. A readout taken through a hook that is "
            "not a no-op is not the model's native readout."
        )
'''
)

# ============================================ 12. fingerprint and Stage A

markdown(
    """
## 12. The run fingerprint, then Stage A

The fingerprint binds every scientific configuration field the design depends
on — model and processor revisions, Transformers and torch versions, the audio
protocol and its fingerprint, the L32 artifact and its confirmation provenance,
the physical layer and hook site, the original and expanded manifest checksums,
the cache schema and evidence lexicon hash, **the exclusion runs and their
identity digest**, the prompt protocol and its digest, the frozen concepts, the
selection algorithm/seed/profile and the exact selected population digest, the
sample-size rule and its plan digest, the convergence criterion and its digest,
the controls and their seed, and the stage plan with its conditional Stage-B
rule.

Changing **any** of them refuses the resume rather than mixing units. That is
what makes an interrupted Colab session lose at most the in-flight unit.

The `mml32res_*` run directory is created **here**, not in section 8. A
preprocessing session must not leave an empty timestamped run directory behind
for a study it is not going to run.
"""
)

code(
    '''
# 12a. Open this study's own run directory. Never inside a completed run.
RUN_DIR = None
if PREPROCESSING_ONLY:
    print("skipped: PREPROCESSING_ONLY is True — no scientific run directory is")
    print("created by a preprocessing session.")
else:
    RUN_ID = (
        f"{RESOLUTION_RUN_PREFIX}_{CONFIG.mode}_"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    )
    RUN_DIR = Path(os.environ.get("MMPILOT_RUN_DIR") or (RESOLVED_RUNS_ROOT / RUN_ID))
    assert_fresh_run_namespace(RUN_DIR, protected_prefixes=PROTECTED_RUN_PREFIXES)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    print(f"run directory {RUN_DIR}")
'''
)

code(
    '''
# 12. Open the store under the full fingerprint.
from jlens.mmpilot.causal import CONTROL_KINDS
from jlens.mmpilot.convergence import (
    CONTROL_VARIANTS,
    CONVERGENCE_CRITERION,
    CONVERGENCE_PROTOCOL,
)
from jlens.mmpilot.jspace import CONVENTIONS
from jlens.mmpilot.l32_resolution import (
    POPULATION_SELECTION_VERSION,
    SAMPLE_SIZE_RULE_VERSION,
    resolution_fingerprint,
)
from jlens.mmpilot.pipeline import (
    scientific_fingerprint,
    stage_activations,
    stage_capability,
)
from jlens.mmpilot.store import RunFingerprint, UnitStore
from jlens.mmpilot.tri_modal import TRI_MODAL_VERDICT_VERSION, audio_capability_verdict

CONTROL_SEED = 20260809
_recorded = L32_VALIDATION["recorded"]

STORE = None
RUN_STATE = "not_opened"
RESOLUTION_FINGERPRINT = None
SELECTION_FINGERPRINT = None
FINGERPRINT = None

if RUN_DIR is None or BACKEND is None:
    print("skipped: there is no model configuration to bind a fingerprint to.")
    print("A fingerprint that recorded 'no model' would be a fingerprint another")
    print("run could resume from, so none is written at all.")
else:
    RESOLUTION_FINGERPRINT = resolution_fingerprint(
        protocol=L32_RESOLUTION_PROTOCOL,
        stage_plan_version=STAGE_PLAN_VERSION,
        conditional_stage_b_rule_digest=payload_checksum(STAGE_PLAN),
        intervention_family=INTERVENTION_FAMILY,
        model_repo_id=(
            MODEL_REPO_ID
            if RUN_REAL_L32_CONVERGENCE_RESOLUTION
            else "mock/gemma-like"
        ),
        model_revision=MODEL_REVISION_USED,
        processor_revision=PROCESSOR_REVISION_USED,
        transformers_version=TRANSFORMERS_VERSION,
        torch_version=TORCH_VERSION,
        audio_protocol_version=AUDIO_PROTOCOL["protocol_version"],
        audio_protocol_fingerprint=AUDIO_PROTOCOL["protocol_fingerprint"],
        lens_path=DISCOVERED.lens_path,
        lens_checksum=DISCOVERED.lens_checksum,
        lens_confirmation_status="passed" if L32_VALIDATION["passed"] else "failed",
        lens_confirmation_set_checksum=DISCOVERED.extension_sidecar_path,
        publication_metadata_checksum=DISCOVERED.extension_artifact_checksum,
        physical_layer=RESOLUTION_LAYER,
        hook_site=CONVENTIONS["hook_site"],
        d_model=L32_LENSES.d_model,
        residual_convention=_recorded.get("residual_convention"),
        final_prompt_token_position=CONVENTIONS["position"],
        dictionary_orientation=CONVENTIONS["dictionary"],
        dictionary_normalization=CONVENTIONS["code_orientation"],
        calibration_modality=_recorded.get("calibration_modality"),
        selected_scale=LENS_FITTED_SCALE,
        original_manifest_checksum=ORIGINAL_MANIFEST_CHECKSUM,
        expanded_manifest_checksum=DERIVED_MANIFEST_CHECKSUM,
        cache_schema_version=CACHE_LOAD["schema_version"],
        evidence_lexicon_hash=CACHE_LOAD["evidence_lexicon_hash"],
        exclusion_run_dirs=sorted(str(d) for d in COMPLETED_RUN_DIRS),
        exclusion_run_checksum=EXCLUSION.digest,
        # The whole preprocessing stage is bound in: which artifacts were read,
        # what proved the exclusion set complete, which groups survived the
        # filter, how the pool ranked and how the frozen concepts fared against
        # it. Change any preparation input and this digest moves.
        preparation_version=prep.PREPARATION_VERSION,
        preparation_digest=PREPARATION_FINGERPRINT["preparation_digest"],
        exclusion_completeness_digest=payload_checksum(COMPLETENESS),
        independent_pool_digest=POOL_DIGEST,
        concept_ranking_digest=RANKING_DIGEST,
        frozen_concept_feasibility_digest=FEASIBILITY_DIGEST,
        prompt_protocol=OPEN_PROMPT_PROTOCOL,
        prompt_hash=PROMPT_RECORDS["text"].prompt_hash,
        prompt_protocol_digest=PROMPT_PROTOCOL_DIGEST,
        selected_concepts=list(SELECTED_NAMES),
        focal_concepts=list(FOCAL_CONCEPTS),
        capability_admissible_concepts=None,
        capability_protocol=OPEN_PROMPT_PROTOCOL,
        admissibility_rule_version=TRI_MODAL_VERDICT_VERSION,
        selection_algorithm_version=POPULATION_SELECTION_VERSION,
        selection_seed=SPLIT_SEED,
        selection_profile_version=PROFILE.version,
        selected_population_digest=POPULATION_DIGEST,
        sample_size_rule_version=SAMPLE_SIZE_RULE_VERSION,
        sample_size_plan_digest=SAMPLE_PLAN_RECORD["plan_digest"],
        convergence_criterion_version=CONVERGENCE_PROTOCOL,
        convergence_criterion_digest=CONVERGENCE_CRITERION.digest,
        control_variants=list(CONTROL_VARIANTS),
        control_seed=CONTROL_SEED,
        media_io_version=MEDIA_IO_VERSION,
        jlens_version=COMMIT,
    )

    # ``capability_admissible_concepts`` is deliberately None: admissibility is
    # a RESULT of Stage A, and binding a result into the fingerprint that gates
    # Stage A's own units would make the store refuse to resume itself the
    # moment the first capability unit landed.
    SELECTION_FINGERPRINT = scientific_fingerprint(
        CONFIG,
        ranked_concepts=RANKED_CONCEPTS,
        selected_concepts=SELECTED_NAMES,
        focal_concepts=FOCAL_CONCEPTS,
        unrelated_controls=UNRELATED_CONTROLS,
        derived_cache_fingerprint=DERIVED_MANIFEST_CHECKSUM,
        split_provenance_checksum=SPLIT_PROVENANCE_CHECKSUM,
        n_train_positive_images=N_TRAIN_POSITIVE_IMAGES,
        n_train_negative_images=N_TRAIN_NEGATIVE_IMAGES,
        n_test_positive_images=N_TEST_POSITIVE_IMAGES,
        n_test_negative_images=N_TEST_NEGATIVE_IMAGES,
        verdict_version=TRI_MODAL_VERDICT_VERSION,
        prompt_protocol=OPEN_PROMPT_PROTOCOL,
        candidate_ordering_protocol="external_scorer_only_no_prompt_order.v1",
    )
    FINGERPRINT = RunFingerprint(
        mode=CONFIG.mode,
        model_repo_id=(
            MODEL_REPO_ID
            if RUN_REAL_L32_CONVERGENCE_RESOLUTION
            else "mock/gemma-like"
        ),
        model_revision=MODEL_REVISION_USED,
        processor_revision=PROCESSOR_REVISION_USED,
        layers=tuple(CONFIG.layers),
        lens_checksum=L32_LENSES.combined_checksum,
        manifest_checksum=ORIGINAL_MANIFEST_CHECKSUM,
        split_id=SPLIT_SEED,
        intervention_config={
            "alphas": list(CONFIG.alphas),
            "causal_layer": RESOLUTION_LAYER,
            "off_diagonal_causal_only": True,
            "intervention_family": INTERVENTION_FAMILY,
        },
        selection_config=SELECTION_FINGERPRINT,
        extra={"resolution_fingerprint": RESOLUTION_FINGERPRINT},
    )
    STORE = UnitStore(RUN_DIR, FINGERPRINT)
    RUN_STATE = STORE.open()
    print(f"run state              {RUN_STATE}")
    print(f"run fingerprint        {FINGERPRINT.digest}")
    print(f"resolution fingerprint {RESOLUTION_FINGERPRINT['fingerprint_digest']}")
    print(f"preparation digest     "
          f"{PREPARATION_FINGERPRINT['preparation_digest']}")
    print("  no unit from any other population, protocol, criterion or "
          "preparation")
    print("  can be reused here: the exclusion digest, the completeness proof, "
          "the")
    print("  pool digest, the ranking digest, the frozen-concept feasibility "
          "digest")
    print("  and the selected population digest are all bound into this "
          "fingerprint.")
'''
)

code(
    '''
# 12b. Stage A, part 1 — behavioral capability on the FRESH population.
CAPABILITY = None
CAPABILITY_VERDICT = None
ADMISSIBILITY = None
GATES = refresh_gates()

if BACKEND is None or not STAGE_A_ENABLED:
    print("skipped: Stage A is not enabled")
else:
    from jlens.mmpilot.admissibility import concept_admissibility

    CAPABILITY_OUTCOME, CAPABILITY = stage_capability(
        BACKEND, STORE, SUBSET, CONFIG, MEDIA,
        modalities=AVAILABLE_MODALITIES,
        questions=[OPEN_QUESTION],
    )
    print(CAPABILITY_OUTCOME.line("capability"))
    print("\\nper-concept accuracy under the open prompt (raw counts):")
    for _concept, _per_modality in sorted(CAPABILITY["per_concept"].items()):
        _cells = "  ".join(
            f"{m}={e['n_correct']}/{e['n']}"
            f"(med margin {e['median_target_margin']:.2f})"
            if e.get("median_target_margin") is not None
            else f"{m}={e['n_correct']}/{e['n']}"
            for m, e in sorted(_per_modality.items())
        )
        print(f"  {_concept:12s} {_cells}")
    _order_stable = sum(
        1 for r in CAPABILITY_OUTCOME.records if r.get("option_order_stable", True)
    )
    print(f"\\noption-order stability: {_order_stable}/"
          f"{len(CAPABILITY_OUTCOME.records)} records "
          "(an open prompt is asked once; the order lives in the scorer)")

    CAPABILITY_VERDICT = audio_capability_verdict(
        CAPABILITY,
        selected_concepts=SELECTED_NAMES,
        modalities=AVAILABLE_MODALITIES,
        thresholds=THRESHOLDS,
    )
    ADMISSIBILITY = concept_admissibility(
        list(FOCAL_CONCEPTS), capability=CAPABILITY_VERDICT
    )
    STORE.save("metric", "open_prompt_capability_verdict", CAPABILITY_VERDICT)
    print(f"\\ncapability verdict  {CAPABILITY_VERDICT['verdict']}")
    print(f"admissible focal    {ADMISSIBILITY['eligible_concepts']}")
    print(f"EXCLUDED focal      {ADMISSIBILITY['excluded_concept_names']}")
    for _entry in ADMISSIBILITY["excluded_concepts"]:
        print(f"    {_entry['concept']:12s} {_entry['rejection_reason']}")
    print("  an excluded concept stays in the raw table above, is labelled with")
    print("  the arithmetic that rejected it, enters no verdict, and is NEVER")
    print("  replaced by another concept.")
'''
)

code(
    '''
# 12c. Stage A, part 2 — final-prompt-token residuals under the same question.
ACTIVATIONS = []
GATES = refresh_gates()

if BACKEND is None or not STAGE_A_ENABLED:
    print("skipped: Stage A is not enabled")
else:
    ACTIVATION_OUTCOME = stage_activations(
        BACKEND, STORE, SUBSET, CONFIG, MEDIA,
        modalities=AVAILABLE_MODALITIES,
        retained_concepts=SELECTED_NAMES,
        model_revision=MODEL_REVISION_USED,
        question=OPEN_QUESTION,
    )
    ACTIVATIONS = ACTIVATION_OUTCOME.records
    print(ACTIVATION_OUTCOME.line("activation"))
    print(f"distinct images captured   "
          f"{len({r['image_id'] for r in ACTIVATIONS})}")
    print(f"media retries survived     {MEDIA_RETRY_JOURNAL.n_retries} "
          f"on {MEDIA_RETRY_JOURNAL.n_paths} path(s)")
'''
)

# ==================================== 13. the native readout and its controls

markdown(
    """
## 13. The native direct readout at layer 32, and its three controls

The readout is the model's **own** output head applied to the stored residual:

```
logits = lm_head(final_norm(h))            # modules called, not reimplemented
logits = softcap * tanh(logits / softcap)  # exactly as the config declares
```

restricted to the six fixed candidates. No lens, no dictionary, no J-space code,
no intervention and no learned probe takes part in this number.

The head is audited against the model's own `unembed` on device-correct probes,
one probe at a time as a singleton batch. A comparison that never *ran* leaves
`matches_model_unembed` as `None`, and reading that with a truthiness test would
silently accept it — `assert_native_head_agrees` refuses it instead.

Controls: **permuted activations**, **permuted candidate-token assignment** and
**shuffled target labels**, drawn within a modality cell so a permutation never
crosses a boundary the primary metric is reported across. A missing control
record is a refusal, not a pass.

The clean-answer reference comes from the **capability** units — the same
externally-scored argmax on the same unedited input. The completed follow-up took
it from zero-alpha intervention units because it ran the causal stage first;
Stage A deliberately does not, and the quantity is identical.
"""
)

code(
    '''
# 13. The native readout, audited against the model's own unembed.
from jlens.mmpilot.capability import candidate_token_ids
from jlens.mmpilot.convergence import (
    ConvergenceFingerprint,
    ConvergenceStore,
    NativeHead,
    audit_native_head,
    build_population,
    head_from_model,
    resolve_candidate_tokens,
)
from jlens.mmpilot.l32_followup import (
    assert_native_head_agrees,
    run_single_layer_convergence,
)
from jlens.mmpilot.l32_reporting import (
    classification_detail,
    convergence_cell_rows,
    format_classification,
    format_controls,
    format_convergence_cells,
    format_l32_criterion,
)
from jlens.mmpilot.l32_resolution import clean_predictions_from_capability

CONVERGENCE = None
CONVERGENCE_CLASSIFICATION = None
CONVERGENCE_CONTROLS = None
CONVERGENCE_CELLS = None
CONTROLS_RECORD = None
HEAD_AUDIT = None
HEAD_AGREEMENT = None
POPULATION = None
CRITERION_TEXT = format_l32_criterion(layer=RESOLUTION_LAYER)

if BACKEND is None or not ACTIVATIONS:
    print("skipped: Stage A produced no activations to read out")
else:
    print(CRITERION_TEXT)
    if RUN_REAL_L32_CONVERGENCE_RESOLUTION:
        HEAD = head_from_model(MODEL)
        HEAD_AUDIT = audit_native_head(HEAD, model=MODEL, probes=4)
        HEAD_AGREEMENT = assert_native_head_agrees(HEAD_AUDIT, required=True)
        print(f"native head vs model unembed: "
              f"matched={HEAD_AGREEMENT['matches_model_unembed']}  "
              f"max_abs_diff={HEAD_AGREEMENT['max_abs_difference_vs_model_unembed']}"
              f"  protocol={HEAD_AGREEMENT['protocol']}")
    else:
        _inner = BACKEND.hf_model
        HEAD = NativeHead(
            final_norm=_inner.model.language_model.norm,
            lm_head=_inner.lm_head,
            softcap=None,
            d_model=BACKEND.d_model,
            vocab_size=_inner.lm_head.weight.shape[0],
        )
        HEAD_AUDIT = audit_native_head(HEAD, model=None, probes=4)
        HEAD_AGREEMENT = assert_native_head_agrees(HEAD_AUDIT, required=False)
        print("MOCK: the stub has no unembed to compare against, so the")
        print("agreement check is recorded as NOT RUN, never as passed:")
        print(f"  comparison_ran={HEAD_AGREEMENT['comparison_ran']}")

    TOKENIZATION = resolve_candidate_tokens(
        candidate_token_ids(BACKEND, SELECTED_NAMES)
    )
    print(f"readout mode {TOKENIZATION['readout_mode']}")

    CLEAN_PREDICTIONS = clean_predictions_from_capability(
        CAPABILITY_OUTCOME.records
    )
    POPULATION = build_population(
        activations=ACTIVATIONS,
        clean_predictions=CLEAN_PREDICTIONS,
        capability=CAPABILITY_VERDICT,
        focal_concepts=FOCAL_CONCEPTS,
        layers=(RESOLUTION_LAYER,),
    )
    print(f"population {POPULATION['n_units']} units, "
          f"{POPULATION['n_with_clean_reference']} with a clean reference")
    print(f"  admissible   {POPULATION['admissible_concepts']}")
    print(f"  inadmissible {POPULATION['inadmissible_concepts']} (excluded, not "
          "replaced)")
'''
)

code(
    '''
# 13b. Score the population, apply the frozen criterion, run the controls.
from jlens.mmpilot.l32_resolution import assert_controls_recorded

if POPULATION is None:
    print("skipped")
else:
    CONVERGENCE_STORE = ConvergenceStore(
        RUN_DIR / "convergence",
        ConvergenceFingerprint(
            protocol=CONVERGENCE_PROTOCOL,
            completed_run_fingerprint_digest=FINGERPRINT.digest,
            completed_run_dir=str(RUN_DIR),
            model_repo_id=(
                MODEL_REPO_ID
                if RUN_REAL_L32_CONVERGENCE_RESOLUTION
                else "mock/gemma-like"
            ),
            model_revision=MODEL_REVISION_USED,
            processor_revision=PROCESSOR_REVISION_USED,
            layers=(RESOLUTION_LAYER,),
            candidate_digest=TOKENIZATION["digest"],
            readout_mode=TOKENIZATION["readout_mode"],
            head_checksum=str(HEAD_AUDIT.get("head_checksum", "")),
            criterion_digest=CONVERGENCE_CRITERION.digest,
            code_version=L32_RESOLUTION_PROTOCOL,
            extra={
                "resolution_fingerprint": RESOLUTION_FINGERPRINT[
                    "fingerprint_digest"
                ],
                "population_digest": POPULATION_DIGEST,
                "exclusion_digest": EXCLUSION.digest,
            },
        ),
    )
    CONVERGENCE_STORE_STATE = CONVERGENCE_STORE.open()
    print(f"convergence store: {CONVERGENCE_STORE_STATE}")

    CONVERGENCE = run_single_layer_convergence(
        population=POPULATION,
        head=HEAD,
        tokenization=TOKENIZATION,
        head_audit=HEAD_AUDIT,
        store=CONVERGENCE_STORE,
        layer=RESOLUTION_LAYER,
        confirmation_record=L32_VALIDATION,
        control_seed=CONTROL_SEED,
    )
    CONVERGENCE_CLASSIFICATION = CONVERGENCE["classification"]
    CONVERGENCE_CONTROLS = CONVERGENCE["controls"]
    if CONVERGENCE["criterion_digest"] != FROZEN_CRITERION_DIGEST:
        raise RuntimeError(
            "the convergence criterion's digest is not the frozen one "
            f"({CONVERGENCE['criterion_digest']} vs {FROZEN_CRITERION_DIGEST}). "
            "The thresholds are predeclared and are not revisable now."
        )

    CONVERGENCE_CELLS = convergence_cell_rows(
        CONVERGENCE["summary"], layer=RESOLUTION_LAYER
    )
    print(format_convergence_cells(CONVERGENCE_CELLS, layer=RESOLUTION_LAYER))
    print()
    CONVERGENCE_DETAIL = classification_detail(CONVERGENCE_CLASSIFICATION)
    print(format_classification(CONVERGENCE_DETAIL))
    print()
    CONTROLS_RECORD = assert_controls_recorded(
        CONVERGENCE_CONTROLS, layer=RESOLUTION_LAYER
    )
    print(format_controls(
        CONTROLS_RECORD["rows"],
        controls=CONVERGENCE_CONTROLS,
        layer=RESOLUTION_LAYER,
    ))
    print()
    print("IMAGE-LEVEL UNCERTAINTY (bootstrap over photographs, per modality)")
    _per_modality = CONVERGENCE["summary"]["per_layer"][
        str(RESOLUTION_LAYER)
    ]["per_modality"]
    for _modality, _cell in sorted(_per_modality.items()):
        _boot = _cell.get("bootstrap_clean_agreement_unique") or {}
        print(f"  {_modality:13s} n={_cell.get('n', 0):>4}  "
              f"images={_cell.get('n_distinct_images', 0):>4}  "
              f"clean_unique={_boot.get('point')}  "
              f"CI=[{_boot.get('low')}, {_boot.get('high')}]")
'''
)

# ================================================ 14. the primary verdict

markdown(
    """
## 14. The one primary verdict

Exactly one of:

| verdict | meaning |
|---|---|
| `L32_INDEPENDENT_NOT_CONVERGED` | every `NOT_CONVERGED` clause holds in every required modality, scored generously |
| `L32_INDEPENDENT_CONVERGED` | every `CONVERGED` clause holds in every required modality, scored strictly |
| `L32_INDEPENDENT_AMBIGUOUS` | the layer falls between the two frozen bars |
| `REFUSED_INVALID` | a validity clause failed, so **no** classification is reported |

Validity is established first and separately. A study whose controls did not run
has not measured `AMBIGUOUS` — it has measured nothing, and saying that is a
refusal rather than a hedged classification.
"""
)

code(
    '''
# 14. The primary convergence verdict.
from jlens.mmpilot.l32_followup import lens_integrity_verdict
from jlens.mmpilot.l32_resolution import resolution_verdict

INTEGRITY = lens_integrity_verdict(
    L32_VALIDATION,
    invariance=INVARIANCE,
    discovery=DISCOVERED.to_dict(),
    layer=RESOLUTION_LAYER,
    scale=LENS_FITTED_SCALE,
)
VERDICT = None
if CONVERGENCE is None:
    print("no convergence measurement was produced; there is nothing to classify")
else:
    VERDICT = resolution_verdict(
        integrity=INTEGRITY,
        convergence=CONVERGENCE,
        controls=CONTROLS_RECORD,
        disjointness=DISJOINTNESS,
        pseudoreplication=PSEUDOREPLICATION,
        sample_plan=SAMPLE_PLAN_RECORD,
        head_agreement=HEAD_AGREEMENT,
        admissibility=ADMISSIBILITY,
    )
    print("=" * 72)
    print(f"PRIMARY VERDICT — {VERDICT['verdict_name']}: {VERDICT['verdict']}")
    print("=" * 72)
    print()
    for _clause in VERDICT["validity_clauses"]:
        _mark = "PASS" if _clause["passed"] else "FAIL"
        print(f"  [{_mark}] {_clause['clause']}")
        print(f"         {_clause['detail']}")
    print()
    print(f"  classification        {VERDICT['classification']}")
    print(f"  criterion unchanged   {VERDICT['criterion_thresholds_unchanged']}")
    print(f"  admissible focal      {VERDICT['admissible_focal_concepts']}")
    print(f"  excluded focal        {VERDICT['inadmissible_focal_concepts']}")
    print(f"  concepts replaced     {VERDICT['concepts_replaced']}")
    print()
    print(VERDICT["rationale"])
'''
)

# ================================================ 15. Stage B

markdown(
    """
## 15. Stage B — the conditional causal replication

The gate was fixed in section 2, before Stage A was opened. It is evaluated
here, printed with every Stage-A outcome, and recorded in every artifact.

**Why conditional, and why that is not suppression.** Stage B's only purpose is
the combined claim of causal transfer *before native direct-readout
convergence*. That claim requires `NOT_CONVERGED`. Under `CONVERGED` the claim is
dead at this layer whatever the causal passes show; under `AMBIGUOUS` it is
unsupported whatever they show. The gate therefore stops Stage B exactly where
the headline result is *unfavourable* to the hypothesis — and those Stage-A
outcomes are the study's primary reported verdict. Nothing is hidden by not
spending passes on a measurement that could not move them.

The alternative (run Stage B unconditionally) was rejected on cost alone, so the
override exists: setting `RUN_STAGE_B_CAUSAL_REPLICATION` runs it anyway, and the
artifacts stamp `gate_overridden: true` so an overridden run can never be
presented as the predeclared path.

Stage B uses **this same open prompt** and the existing matched-random,
external-unrelated, specificity, activation-norm and image-independence
controls. It never uses the historical candidate-listed protocol, and it is not
a coordinate swap.
"""
)

code(
    '''
# 15. Evaluate the predeclared Stage-B gate, then run it if it is on.
from jlens.mmpilot.l32_resolution import (
    L32_INDEPENDENT_NOT_CONVERGED,
    stage_b_decision,
)

GATES = refresh_gates()
STAGE_B = stage_b_decision(
    verdict=(VERDICT or {}).get("verdict", "REFUSED_INVALID"),
    controls_passed=bool((CONTROLS_RECORD or {}).get("passed")),
    requested=bool(RUN_STAGE_B_CAUSAL_REPLICATION),
    budget_confirmed=bool(CONFIRM_STAGE_B_BUDGET),
)
print("STAGE B DECISION")
print(f"  stage A verdict   {STAGE_B['stage_a_verdict']}")
print(f"  controls passed   {STAGE_B['controls_passed']}")
print(f"  gate met          {STAGE_B['gate_met']}")
print(f"  requested         {STAGE_B['requested']}")
print(f"  budget confirmed  {STAGE_B['budget_confirmed']}")
print(f"  RUNS              {STAGE_B['runs']}")
print(f"  gate overridden   {STAGE_B['gate_overridden']}")
print()
print(STAGE_B["statement"])
print()
print(f"RULE: {STAGE_B['rule']}")
print()
print(f"WHY THIS IS AN EFFICIENCY GATE: {STAGE_B['rationale']}")
'''
)

code(
    '''
# 15b. Stage B, if it runs: source-derived J-space causal steering at L32.
#
# The same code path the completed studies used — nothing about the causal
# method is new here, and the ONLY thing that differs is the population.
CAUSAL_VERDICT = None
CAUSAL_IMAGE_LEVEL = None
CAUSAL_INDEPENDENCE = None
if not STAGE_B["runs"]:
    print("Stage B did not run. Every Stage-A result above stands unchanged.")
elif CAPABILITY_VERDICT is None or CAPABILITY_VERDICT["verdict"] != (
    "AUDIO_CAPABILITY_GO"
):
    print("Stage B skipped: the open-prompt capability gate did not pass. The")
    print("causal stage does not run on a capability the model does not have")
    print("under this question, and the design is not narrowed to rescue it.")
else:
    from jlens.mmpilot.independence import (
        audit_image_independence,
        divergence_summary,
        resolve_image_identity,
        summarize_interventions_by_image,
    )
    from jlens.mmpilot.pipeline import (
        build_dictionaries,
        stage_causal,
        stage_codes,
        stage_directions,
    )
    from jlens.mmpilot.tri_modal import causal_transfer_verdict

    _dictionaries = build_dictionaries(
        L32_LENS,
        (RESOLUTION_LAYER,),
        BACKEND,
        device=(
            "cuda"
            if (RUN_REAL_L32_CONVERGENCE_RESOLUTION and torch.cuda.is_available())
            else "cpu"
        ),
        dtype=(
            torch.float16 if RUN_REAL_L32_CONVERGENCE_RESOLUTION else torch.float32
        ),
        build_chunk_rows=32768 if RUN_REAL_L32_CONVERGENCE_RESOLUTION else None,
    )
    _code_outcome = stage_codes(
        STORE, ACTIVATIONS, _dictionaries, CONFIG,
        lens_checksum=DISCOVERED.lens_checksum,
    )
    CODES = _code_outcome.records
    print(_code_outcome.line("jspace"))

    _direction_outcome, DIRECTIONS = stage_directions(
        STORE, CODES, ACTIVATIONS, _dictionaries, CONFIG,
        concepts=SELECTED_NAMES,
        modalities=AVAILABLE_MODALITIES,
        lens_checksum=DISCOVERED.lens_checksum,
    )
    del _dictionaries
    print(_direction_outcome.line("direction"))

    _causal_outcome, INTERVENTIONS = stage_causal(
        BACKEND, STORE, SUBSET, CODES, ACTIVATIONS, DIRECTIONS, CONFIG, MEDIA,
        concepts=FOCAL_CONCEPTS,
        modalities=AVAILABLE_MODALITIES,
        all_concepts=SELECTED_NAMES,
        unrelated_controls=UNRELATED_CONTROLS,
        question=OPEN_QUESTION,
    )
    INTERVENTION_RECORDS = _causal_outcome.records
    print(_causal_outcome.line("intervention"))

    _identity = resolve_image_identity(
        [*ACTIVATIONS, *CODES, *INTERVENTION_RECORDS]
    )
    CAUSAL_IMAGE_LEVEL = summarize_interventions_by_image(
        INTERVENTION_RECORDS, _identity, group_summary=INTERVENTIONS
    )
    _divergence = divergence_summary(CAUSAL_IMAGE_LEVEL)
    CAUSAL_INDEPENDENCE = audit_image_independence(
        _identity, interventions=INTERVENTION_RECORDS, concepts=SELECTED_NAMES,
    )
    if _divergence["n_rows_pseudoreplicated_at_group_level"]:
        raise RuntimeError(
            "an intervention cell drew more than one observation from one "
            "photograph; refusing to report a pseudoreplicated causal summary."
        )

    CAUSAL_VERDICT = causal_transfer_verdict(
        CAUSAL_IMAGE_LEVEL,
        layer=RESOLUTION_LAYER,
        focal_concepts=FOCAL_CONCEPTS,
        thresholds=THRESHOLDS,
        name="L32_INDEPENDENT_CAUSAL_TRANSFER",
        capability=CAPABILITY_VERDICT,
    )
    STORE.save("metric", "stage_b_causal_verdict", CAUSAL_VERDICT)
    STORE.save("metric", "stage_b_image_independence", CAUSAL_INDEPENDENCE)
    print()
    print(f"STAGE B causal verdict: {CAUSAL_VERDICT['verdict']}")
    print(CAUSAL_VERDICT.get("rationale", ""))
    if STAGE_B["gate_overridden"]:
        print()
        print("NOTE: the predeclared gate was OVERRIDDEN. These causal numbers")
        print("are DESCRIPTIVE and do not enter the combined pre-convergence")
        print("claim, which requires a NOT_CONVERGED convergence half.")
'''
)

# ================================================ 16. synthesis

markdown(
    """
## 16. Synthesis with the completed causal evidence — kept strictly separate

The trap this section exists to close: an independent `NOT_CONVERGED` next to the
completed run's causal evidence *looks* like a combined pre-convergence claim,
and it is not one. The completed causal measurement was made on a **different
population**. Pairing its effect with this population's convergence would credit
a causal result to photographs it was never measured on.

So the synthesis states, explicitly, what additional causal replication is
required — and it is a measurement, not an argument.
"""
)

code(
    '''
# 16. The synthesis, with the completed evidence read-only and never merged.
from jlens.mmpilot.l32_resolution import (
    COMPLETED_FOLLOWUP_FINGERPRINT,
    COMPLETED_FOLLOWUP_RUN,
    causal_synthesis,
)

COMPLETED_CAUSAL = {
    "verdict": "WEAK",
    "run_dir": COMPLETED_FOLLOWUP_RUN,
    "fingerprint": COMPLETED_FOLLOWUP_FINGERPRINT,
    "prompt_protocol": OPEN_PROMPT_PROTOCOL,
    "note": "read-only historical context; never pooled with anything measured here",
}
_completed_report = Path(COMPLETED_RUN_DIRS[0]) / COMPLETED_FOLLOWUP_SUMMARY
if _completed_report.is_file():
    try:
        _payload = json.loads(_completed_report.read_text(encoding="utf-8"))
        _causal = (_payload.get("verdicts") or {}).get("L32_CAUSAL_TRANSFER") or {}
        if _causal.get("verdict"):
            COMPLETED_CAUSAL["verdict"] = str(_causal["verdict"])
            COMPLETED_CAUSAL["read_from"] = str(_completed_report)
    except (json.JSONDecodeError, OSError) as _error:
        COMPLETED_CAUSAL["read_error"] = str(_error)

SYNTHESIS = causal_synthesis(
    verdict=VERDICT or {"verdict": "REFUSED_INVALID"},
    completed_causal=COMPLETED_CAUSAL,
    stage_b={**STAGE_B, "causal_verdict": (CAUSAL_VERDICT or {}).get("verdict")},
)
print("SYNTHESIS")
print(f"  independent convergence   {SYNTHESIS['independent_convergence_verdict']}")
print(f"  completed causal (r/o)    {SYNTHESIS['completed_causal_verdict']}")
print(f"  same population           {SYNTHESIS['populations_are_the_same']}")
print(f"  combined claim            {SYNTHESIS['combined_pre_convergence_claim']}")
print()
print(SYNTHESIS["statement"])
if SYNTHESIS["additional_causal_replication_required"]:
    print()
    print("STILL REQUIRED before any pre-convergence causal claim:")
    for _item in SYNTHESIS["additional_causal_replication_required"]:
        print(f"  - {_item}")
print()
print("NEVER CLAIMED:")
for _item in SYNTHESIS["never_claimed"]:
    print(f"  - {_item}")
'''
)

# ================================================ 17. artifacts

markdown(
    """
## 17. Write the artifacts

Into this run's own `mml32res_*` directory and nowhere else:

* `l32_convergence_resolution_summary.json`
* `l32_convergence_resolution_report.md`
* `population_manifest.json`
* `disjointness_audit.json`
* `convergence_tables.json` and `convergence_controls.json`
* `run_state.json`
* `stage_b_causal_report.json` — only when Stage B ran, kept clearly separate
"""
)

code(
    '''
# 17. Assemble and write every artifact.
from jlens.mmpilot.l32_followup import assert_report_phrasing
from jlens.mmpilot.l32_resolution import build_summary, render_report

ARTIFACT_PATHS = {}
SUMMARY = None
RESUME = None
IMMUTABILITY = prep.assert_sources_unchanged(
    prep.verify_sources_unchanged(
        COMPLETED_RUN_DIRS,
        PREP["inventory"],
        SOURCE_FAMILIES,
    )
)

if RUN_DIR is None or RESOLUTION_FINGERPRINT is None:
    print("skipped: no run directory and no run fingerprint, so there is no")
    print("scientific result to write. Preprocessing artifacts are already on")
    print(f"Drive at {PREP_DIR}.")
else:
    POPULATION_MANIFEST = {
        "schema": "jlens.mmpilot.l32_resolution_population_manifest.v1",
        "selection_version": POPULATION_SELECTION_VERSION,
        "seed": SPLIT_SEED,
        "profile": PROFILE.to_dict(),
        "population_digest": POPULATION_DIGEST,
        "determinism_check": SELECTION_DETERMINISM,
        "sample_size_plan": SAMPLE_PLAN_RECORD,
        "selected_concepts": list(SELECTED_NAMES),
        "focal_concepts": list(FOCAL_CONCEPTS),
        "concept_ranking_is_descriptive_only": True,
        "concept_ranking_digest": RANKING_DIGEST,
        "frozen_concept_feasibility": FEASIBILITY,
        "unrelated_controls": dict(sorted(UNRELATED_CONTROLS.items())),
        "split_provenance": SPLIT_PROVENANCE,
        "pool": POOL_RECORD,
        "media_resolution": MEDIA_RESOLUTION,
        "preparation": {
            "cache_dir": str(PREP_DIR),
            "fingerprint": PREPARATION_FINGERPRINT,
            "completeness_proof": COMPLETENESS,
            "source_families_read": SOURCE_FAMILIES,
            "files_computed_this_session": PREP["files_computed_this_session"],
            "files_reused_from_drive": PREP["files_reused_from_drive"],
        },
        "units": [
            {
                "group_id": str(row["group_id"]),
                "image_id": str(row["image_id"]),
                "audio_path": str(row["audio_path"]),
                "caption": str(row["caption"]),
                "concept": row["concept"],
                "split": row["split"],
            }
            for row in sorted(_all_rows, key=lambda r: str(r["group_id"]))
        ],
        "media_retry_journal": MEDIA_RETRY_JOURNAL.to_dict(),
    }

    RESUME = {
        "schema": "jlens.mmpilot.l32_resolution_resume.v1",
        "run_state": RUN_STATE,
        "run_dir": str(RUN_DIR),
        "preparation_cache_dir": str(PREP_DIR),
        "convergence_store_state": globals().get("CONVERGENCE_STORE_STATE"),
        "invalid_units": list(STORE.invalid_units),
        "invalid_convergence_units": list(
            getattr(globals().get("CONVERGENCE_STORE"), "invalid_units", [])
        ),
        "units_computed": (CONVERGENCE or {}).get("units_computed"),
        "units_reused": (CONVERGENCE or {}).get("units_reused"),
        "preprocessing_files_computed": PREP["files_computed_this_session"],
        "preprocessing_files_reused": PREP["files_reused_from_drive"],
        "atomicity": (
            "every unit is written atomically with a checksum of its own payload "
            "and the run fingerprint's digest; preprocessing is committed the "
            "same way in bounded shards. An interrupted session loses at most "
            "the in-flight unit or the in-flight preprocessing batch, and a "
            "changed scientific configuration refuses the resume rather than "
            "mixing units"
        ),
    }

    SUMMARY = build_summary(
        fingerprint=RESOLUTION_FINGERPRINT,
        verdict=VERDICT
        or {"verdict": "REFUSED_INVALID", "rationale": "Stage A did not run"},
        synthesis=SYNTHESIS,
        sample_plan=SAMPLE_PLAN_RECORD,
        disjointness=DISJOINTNESS,
        pseudoreplication=PSEUDOREPLICATION,
        pool=POOL_RECORD,
        exclusion=EXCLUSION.to_dict(),
        convergence={
            k: v for k, v in (CONVERGENCE or {}).items() if k != "summary"
        },
        controls=CONTROLS_RECORD or {},
        capability=CAPABILITY_VERDICT or {},
        stage_plan_record=STAGE_PLAN,
        stage_b=STAGE_B,
        immutability=IMMUTABILITY,
        cache=CACHE_LOAD,
        resume=RESUME,
        mode=CONFIG.mode,
        preparation=POPULATION_MANIFEST["preparation"],
        frozen_concept_feasibility=FEASIBILITY,
    )
    SUMMARY["mock_proves_pipeline_only"] = not RUN_REAL_L32_CONVERGENCE_RESOLUTION
    REPORT_MARKDOWN = render_report(SUMMARY)
    PHRASING = assert_report_phrasing(REPORT_MARKDOWN)

    for _name, _payload in (
        ("l32_convergence_resolution_summary.json", SUMMARY),
        ("population_manifest.json", POPULATION_MANIFEST),
        ("disjointness_audit.json", DISJOINTNESS),
        ("frozen_concept_feasibility.json", FEASIBILITY),
        ("exclusion_completeness_proof.json", COMPLETENESS),
        ("convergence_tables.json", {
            "cells": CONVERGENCE_CELLS,
            "summary": (CONVERGENCE or {}).get("summary"),
            "classification": CONVERGENCE_CLASSIFICATION,
        }),
        ("convergence_controls.json", {
            "record": CONTROLS_RECORD,
            "raw": CONVERGENCE_CONTROLS,
        }),
        ("run_state.json", RESUME),
        ("completed_run_immutability.json", IMMUTABILITY),
    ):
        _path = RUN_DIR / _name
        _path.write_text(
            json.dumps(_payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        ARTIFACT_PATHS[_name] = str(_path)

    _report_path = RUN_DIR / "l32_convergence_resolution_report.md"
    _report_path.write_text(REPORT_MARKDOWN, encoding="utf-8")
    ARTIFACT_PATHS["l32_convergence_resolution_report.md"] = str(_report_path)

    if STAGE_B["runs"]:
        _causal_path = RUN_DIR / "stage_b_causal_report.json"
        _causal_path.write_text(
            json.dumps(
                {
                    "schema": "jlens.mmpilot.l32_resolution_stage_b_report.v1",
                    "decision": STAGE_B,
                    "intervention_family": INTERVENTION_FAMILY,
                    "prompt_protocol": OPEN_PROMPT_PROTOCOL,
                    "causal_verdict": CAUSAL_VERDICT,
                    "image_level": CAUSAL_IMAGE_LEVEL,
                    "image_independence": CAUSAL_INDEPENDENCE,
                    "separate_from_convergence": True,
                },
                indent=2, ensure_ascii=False, default=str,
            ),
            encoding="utf-8",
        )
        ARTIFACT_PATHS["stage_b_causal_report.json"] = str(_causal_path)

    print(f"phrasing check passed: {PHRASING['passed']}")
    for _name, _path in sorted(ARTIFACT_PATHS.items()):
        print(f"  {_name:44s} {_path}")
'''
)

# ================================================ 18. read-only proof

markdown(
    """
## 18. Resume state, and the proof that nothing outside this run was written

The completed runs are protected two ways, and the stronger one is structural:
every write path in `jlens.mmpilot.prep_cache` passes `assert_write_allowed`,
which **refuses** a path through a protected run prefix outright rather than
noticing afterwards that one changed.

The check that runs here is the re-enumeration half. The identity-bearing
families — and only those — are enumerated again and compared to the inventory
section 8 built, by name, size and mtime, against sha256 content digests taken
during the single harvest read. That is at least as strong as the old whole-tree
digest **for the artifacts this study actually depends on**: it covers the same
three facts for every one of them, adds a content hash the old scan never took
(an edit preserving size and mtime is invisible to mtime and impossible to hide
from sha256), and still catches a *new* identity-bearing file. What it
deliberately does not do is re-walk tens of thousands of intervention units that
cannot influence anything here — the old scan's coverage of those bought no
scientific strength and cost hours on a Drive mount.
"""
)

code(
    '''
# 18. Resume state and the read-only proof.
print("PREPROCESSING")
print(f"  cache directory       {PREP_DIR}")
print(f"  preparation digest    "
      f"{PREPARATION_FINGERPRINT['preparation_digest']}")
print(f"  files computed        {PREP['files_computed_this_session']}")
print(f"  files reused          {PREP['files_reused_from_drive']}")
print(f"  checkpoint unit       {PREP_BATCH_FILES} files or "
      f"{PREP_CHECKPOINT_SECONDS:.0f}s, whichever comes first")
print("  stopping the runtime repeats at most ONE in-flight bounded unit and")
print("  never rescans from file zero while a valid checkpoint exists.")
print()
if RESUME is None:
    print("RESUME: no scientific stage ran in this session.")
else:
    print("RESUME")
    print(f"  run state             {RESUME['run_state']}")
    print(f"  convergence store     {RESUME['convergence_store_state']}")
    print(f"  units computed        {RESUME['units_computed']}")
    print(f"  units reused          {RESUME['units_reused']}")
    print(f"  invalid units         {len(RESUME['invalid_units'])}")
    print(f"  invalid readout units {len(RESUME['invalid_convergence_units'])}")
    print(f"  {RESUME['atomicity']}")
print()
print("COMPLETED RUNS — READ-ONLY")
for _run, _families in sorted(IMMUTABILITY["families_verified"].items()):
    print(f"  unchanged  {_run}")
    print(f"             families {_families}")
print(f"  files before/after    {IMMUTABILITY['n_files_before']}/"
      f"{IMMUTABILITY['n_files_after']}")
print(f"  appeared/vanished/modified  {len(IMMUTABILITY['appeared'])}/"
      f"{len(IMMUTABILITY['vanished'])}/{len(IMMUTABILITY['modified'])}")
print(f"  method: {IMMUTABILITY['method']}")
print("  plus: every preprocessing write path refuses a protected run prefix")
print("  outright, so a completed run cannot be written to in the first place.")
print()
print("=" * 72)
if PREPROCESSING_ONLY:
    print("PREPROCESSING ONLY — no model ran and no verdict exists.")
elif not RUN_REAL_L32_CONVERGENCE_RESOLUTION:
    print("MOCK RUN — this proves the pipeline, and nothing about Gemma.")
elif SUMMARY is not None:
    print(f"PRIMARY VERDICT: {SUMMARY['primary_verdict']}")
else:
    print("NO VERDICT: the model stage did not run.")
print("=" * 72)
'''
)


def build() -> dict:
    return {
        "cells": [
            {
                "cell_type": kind,
                "metadata": {},
                "source": text.splitlines(keepends=True),
                **({"execution_count": None, "outputs": []} if kind == "code" else {}),
            }
            for kind, text in CELLS
        ],
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


if __name__ == "__main__":
    TARGET.write_text(
        json.dumps(build(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {TARGET}")
