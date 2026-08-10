# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Generate ``notebooks/research_grade_l27_l31_preconvergence_study_colab.ipynb``.

The notebook is written from source here rather than edited inside a JSON blob,
so the committed file stays output-free and byte-reproducible.

Run with ``python scripts/_build_l27_l31_preconvergence_study_notebook.py``
after changing a cell.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TARGET = (
    REPO_ROOT
    / "notebooks"
    / "research_grade_l27_l31_preconvergence_study_colab.ipynb"
)

CELLS: list[tuple[str, str]] = []


def markdown(text: str) -> None:
    CELLS.append(("markdown", text.strip("\n")))


def code(text: str) -> None:
    CELLS.append(("code", text.strip("\n")))


# ============================================================ 0. front matter

markdown(
    """
# The L27–L31 transition study — one bounded question, five predeclared layers

Three results already on the record close an interval:

```
L26   failed the frozen J-lens confirmation gate
L32   passed J-lens confirmation at scale 250,
      and was classified AMBIGUOUS twice, independently,
      under the frozen native direct-readout convergence criterion
L35   CONVERGED
```

So the open question lives strictly between 26 and 32, and this notebook asks
it once:

> Does any predeclared layer in **27, 28, 29, 30, 31** simultaneously have
> (1) a rigorously confirmed text-calibrated J-lens, (2) a clearly
> `NOT_CONVERGED` native direct readout across text, image and spoken audio,
> and (3) controlled cross-modal causal transfer?

## This is not an instruction to find a favourable layer

`ADJACENT_LENS_NO_GO`, `AMBIGUOUS_CONVERGENCE`, `CONVERGED_BEFORE_CAUSAL_TEST`,
`CAUSAL_TRANSFER_NOT_SUPPORTED` and `REFUSED_INVALID` are first-class terminal
outcomes, reported in the same words and at the same volume as support. The
things that could bend a result are all fixed before any result exists:

* the candidate set is `(27, 28, 29, 30, 31)` and is **closed** — no L33/L34, no
  widening after a table is read, no replacement layer;
* the validity gate is the **same object** the extension applied to L26 and
  L32 (`ADJACENT_GATE is EXTENSION_GATE`), not a copy with an edit;
* the layer-selection rule is "the **lowest** candidate that passes every
  frozen clause", fixed before the confirmation set is opened — never "the one
  that came closest";
* the convergence criterion digest
  `sha256:abbb23e1…0d0446b5a1f` is **checked**, never recomputed into agreement;
* the six behavioural concepts are frozen before any model output and are never
  reranked or substituted;
* the causal stage is gated on the *unfavourable* outcomes being absent, and an
  overridden causal run is stamped `DESCRIPTIVE_ONLY` in every artifact.

## What is reused and what cannot be

**Reused:** the pinned checkpoint `google/gemma-4-E4B-it` at revision
`fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd`, transformers `5.13.1`, the
text-only calibration protocol, the corpus family and prompt normalization, the
target layer, the Jacobian and final-norm conventions, the tie-aware
native-readout validity rule, the open prompt protocol
`mmpilot.open_entity_identification.v1`, the completed studies' intervention
family and their controls, and the checkpointed SpokenCOCO preparation cache.

**Not reusable:** the extension accumulator. It holds source layers
`[8, 14, 20, 26, 32, 35, 38, 40]`; none of 27–31 is in it, and an accumulator
does not acquire a layer it never captured. New Jacobian accumulation over the
five new source layers is required, and seeding from that parent is refused.

**Also not reusable:** the extension's confirmation set. It has been opened —
that is *how* L26's failure and L32's pass are known — so it is development
history here. A new untouched set is built, or the study is **blocked**.

## The hook site, written down

```
output of model.language_model.layers[l]     # post-block residual
final prompt token, where a single position is read
target layer 41, scale 250
```

Scale 250 because that is the scale at which L32 was selected and independently
confirmed. **No scale search runs.**

## Stages

| stage | runtime | model | what it does |
|---|---|---|---|
| 0 | free CPU | no | corpus provenance, untouched confirmation, SpokenCOCO exclusion harvest and independent population — all checkpointed |
| 1 | L4 | yes | fit L27–L31 at scale 250, resumable per bounded batch |
| 2 | L4 | yes | frozen validity gate on every candidate; select the **earliest** fully confirmed layer, or stop with `ADJACENT_LENS_NO_GO` |
| 3 | L4 | yes | behavioural capability and the native direct readout in three modalities |
| 4 | L4/A100 | yes | conditional cross-modal causal transfer with the full control set |

## Not in scope

* An **Anthropic two-coordinate swap**. `jlens/mmpilot/coordinate_swap.py`
  implements it; it needs a *contiguous* confirmed layer band, which today's
  confirmed set does not provide. The intervention here is additive J-space
  residual steering, the same family the open-prompt L32 follow-up used, and
  nothing in this notebook is described as a swap.
* Environmental (non-speech) audio.
* Republishing or modifying any completed run or published lens. Every one of
  them is opened read-only and checksummed before and after.
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
print(f"branch {CHECKED_OUT_BRANCH}")
print(f"commit {COMMIT}")
'''
)

code(
    '''
# 1c. Install the package (editable), then make it importable in this kernel.
if IN_COLAB:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-e", str(REPO_PATH)],
        check=True,
    )
if str(REPO_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_PATH))
os.chdir(REPO_PATH)

import jlens

print(f"jlens from {Path(jlens.__file__).parent}")
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
| `RUN_REAL_PRECONVERGENCE_STUDY` | the real Drive artifacts, the real corpus and the real checkpoint instead of the deterministic MOCK world |
| `PREPROCESSING_ONLY` | stops after Stage 0 with the whole preparation persisted to Drive. Stage 0 is CPU and Drive I/O; a GPU makes none of it faster |
| `RUN_LENS_FITTING` + `CONFIRM_FITTING_BUDGET` | Stage 1: fit L27–L31 at scale 250 |
| `RUN_UNTOUCHED_CONFIRMATION` | Stage 2: open the untouched confirmation set — **once** |
| `RUN_MODEL_STAGE` + `CONFIRM_MODEL_LOAD` | permits the ~16 GB download at all |
| `CONFIRM_STAGE_3_BUDGET` | Stage 3: capability + activations + native readout |
| `RUN_STAGE_4_CAUSAL_TRANSFER` + `CONFIRM_STAGE_4_BUDGET` | Stage 4: the conditional causal transfer |

Gates are **re-derived from the raw switches inside every cell that can spend a
pass**: section 2 defines `refresh_gates()` and every expensive cell calls it
first. A gate computed once and read ten cells later goes stale the moment this
cell is re-run, which is how a notebook comes to print "skipped: not requested"
directly under a switch the operator has just set. `PREPROCESSING_ONLY` closes
every gate, so it can never be combined with a model switch and quietly spend an
L4 hour anyway.

### The three sessions this notebook is built for

| session | runtime | switches | what happens |
|---|---|---|---|
| 1 | **free CPU** | real + `PREPROCESSING_ONLY = True` | Stage 0 harvests, checkpoints and persists everything to Drive. Stop it whenever: at most one bounded unit of ≤25 files is repeated. |
| 2 | **L4** | real, `PREPROCESSING_ONLY = False`, fitting + confirmation switches | Stage 0 loads the cache without re-reading a source unit; Stages 1–3 run. |
| 3 | **L4 or A100** | as session 2 plus the Stage-4 switches | Stage 4, only if its predeclared gate is met. |

### The completed populations this study must avoid

Three, and the third is **not** a constant. Several `mml32res_*` directories can
exist on a Drive; only you know which holds the completed convergence-resolution
study, so `COMPLETED_RESOLUTION_RUN_DIR` must be set by hand. Nothing here
discovers "the newest" one.

### Not free parameters

`ADJACENT_CANDIDATE_LAYERS`, `ADJACENT_FITTING_SCALE`, the gate, the selection
rule, the convergence criterion digest and the six frozen concepts are what the
completed studies fixed. Cells below refuse a configuration that disagrees.
"""
)

code(
    '''
# 2. Configuration. Requires section 1 (it imports from the repository).
# Nothing here mounts Drive, reads data, or loads a model.
RUN_REAL_PRECONVERGENCE_STUDY = False

# Stop after Stage 0 with the whole preparation persisted to Drive. This is the
# CPU-session switch: Stage 0 reads thousands of small files off a Drive mount,
# which a GPU cannot make faster.
PREPROCESSING_ONLY = False

# ---- Stage 1 -------------------------------------------------------------
RUN_LENS_FITTING = False
CONFIRM_FITTING_BUDGET = False
# ---- Stage 2 -------------------------------------------------------------
RUN_UNTOUCHED_CONFIRMATION = False
# ---- Stage 3 -------------------------------------------------------------
RUN_MODEL_STAGE = False
CONFIRM_MODEL_LOAD = False
CONFIRM_STAGE_3_BUDGET = False
# ---- Stage 4 -------------------------------------------------------------
RUN_STAGE_4_CAUSAL_TRANSFER = False
CONFIRM_STAGE_4_BUDGET = False

# The one place a rebuild of the SpokenCOCO evidence join is permitted. Left
# False so a normal session cannot spend twenty-five minutes rediscovering a
# cache it already has; Stage 0 refuses instead and says which input moved.
ALLOW_MANIFEST_REBUILD = False

import json

from jlens.calibration.adjacent import (
    ADJACENT_CANDIDATE_LAYERS,
    ADJACENT_CONFIRMATION_GATE,
    ADJACENT_FITTING_SCALE,
    ADJACENT_GATE,
    ADJACENT_HOOK_SITE,
    ADJACENT_PROTOCOL,
    ADJACENT_PROTOCOL_VERSION,
    ADJACENT_SELECTION_RULE,
    ADJACENT_SPLIT_SEED,
    AMBIGUOUS_UPPER_LAYER,
    CONVERGED_REFERENCE_LAYER,
    FAILED_LOWER_LAYER,
    N_CONFIRMATION_PROMPTS,
    N_DEVELOPMENT_PROMPTS,
    adjacent_budget,
    adjacent_gate_text,
)
from jlens.calibration.extension import EXTENSION_GATE
from jlens.mmpilot.l32_followup import INTERVENTION_FAMILY, OPEN_PROMPT_PROTOCOL
from jlens.mmpilot.l32_resolution import (
    FOCAL_CONCEPTS as FROZEN_FOCAL_CONCEPTS,
)
from jlens.mmpilot.l32_resolution import (
    SAMPLE_SIZE_RULE,
    SAMPLE_SIZE_RULE_VERSION,
    plan_sample_size,
)
from jlens.mmpilot.l32_resolution import (
    SELECTED_CONCEPTS as FROZEN_SELECTED_CONCEPTS,
)
from jlens.mmpilot.pipeline import PilotConfig
from jlens.mmpilot.preconvergence import (
    COMPLETED_AUDIO_TRANSFER_RUN,
    COMPLETED_FOLLOWUP_RUN,
    COORDINATE_SWAP_SCOPE,
    FROZEN_CRITERION_DIGEST,
    PRECONVERGENCE_PROTOCOL,
    PRECONVERGENCE_RUN_PREFIX,
    POPULATION_SELECTION_VERSION,
    REQUIRED_CAUSAL_CONTROLS,
    STAGE_PLAN_VERSION,
    TERMINAL_OUTCOMES,
    derive_preconvergence_gates,
    format_preconvergence_gates,
    format_stage_plan,
    stage_plan,
)
from jlens.mmpilot.selection import IMAGE_UNIQUE_MOCK_PROFILE, IMAGE_UNIQUE_PROFILE
from jlens.mmpilot.tri_modal import TriModalThresholds

# ------------------------------------------------------------------ design
CAPABILITY_THRESHOLD = 0.7

# A NEW seed. Drawing again under a completed study's seed from a filtered pool
# would be a deterministic function of that run's leftovers, not an independent
# sample.
SPLIT_SEED = "spokencoco-l27-l31-preconvergence-v1"

ALPHAS = (0.0, 0.25, 0.5, 1.0)

# ---------------------------------- the completed calibration this study reads
# Read-only, for its corpus provenance and its fit ordering ONLY. Its
# accumulator is NOT continued: it holds no layer in 27-31.
PARENT_RUN_DIR = (
    "/content/drive/MyDrive/jacobian-lens-gemma/runs/rgcalib_real_7e3736b4de8f"
)
EXTENSION_RUN_DIR = (
    "/content/drive/MyDrive/jacobian-lens-gemma/runs/rgext_real_c18f03f06e7b"
)

# ------------------------------------- completed runs: READ-ONLY, and avoided
# YOU MUST SET THE THIRD ONE. It is the completed mml32res convergence-resolution
# run, and it is never discovered: several mml32res_* directories can exist and
# picking the newest would treat a spent population as available.
COMPLETED_RESOLUTION_RUN_DIR = ""

COMPLETED_RUN_DIRS = tuple(
    path
    for path in (
        f"/content/drive/MyDrive/jacobian-lens-gemma/runs/{COMPLETED_AUDIO_TRANSFER_RUN}",
        f"/content/drive/MyDrive/jacobian-lens-gemma/runs/{COMPLETED_FOLLOWUP_RUN}",
        COMPLETED_RESOLUTION_RUN_DIR,
    )
    if path
)

CACHED_EXPANDED_MANIFEST = (
    f"/content/drive/MyDrive/jacobian-lens-gemma/runs/{COMPLETED_FOLLOWUP_RUN}"
    "/expanded_manifest.json"
)
EXPECTED_CACHED_GROUP_COUNT = 125198

PREP_CACHE_ROOT = "/content/drive/MyDrive/datasets/cstf_spokencoco_derived"
PREP_BATCH_FILES = 25
PREP_CHECKPOINT_SECONDS = 30.0
PREP_PROGRESS_SECONDS = 30.0

MODEL_REPO_ID = "google/gemma-4-E4B-it"
MODEL_REVISION = "fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd"
TRANSFORMERS_VERSION_EXPECTED = "5.13.1"
EXPECT_N_LAYERS, EXPECT_D_MODEL, EXPECT_VOCAB = 42, 2560, 262144
TARGET_LAYER = 41
MAX_SEQ_LEN = 128
SKIP_FIRST = 16
DIM_BATCH = 8

AUDIO_PROTOCOL_VERSION_EXPECTED = "jlens.mmpilot.native_spoken_audio.v1"
AUDIO_PROTOCOL_FINGERPRINT_EXPECTED = (
    "sha256:9ad8bcc9420a7983f6e3b75d5d7080c0e2fcf0a94a76431917fcde73ba777920"
)

CORPUS_PRIMARY = {
    "hf_dataset": "Salesforce/wikitext",
    "config": "wikitext-103-raw-v1",
    "split": "train",
    "revision": "b08601e04326c79dfdd32d625aee71d232d685c3",
    "revision_status": "MUST_MATCH_PARENT_RESOLVED_REVISION",
    "min_chars": 600,
    "license": "CC BY-SA 3.0",
}
ARTIFACT_FORMAT_VERSION = "jlens.calibration.artifact.v1"

SPOKENCOCO_BASE_ROOT = "/content/drive/MyDrive/datasets/cstf_spokencoco"
IMAGE_MEDIA_ROOT = "/content/drive/MyDrive/datasets/cstf_spokencoco/coco"
AUDIO_MEDIA_ROOT = "/content/drive/MyDrive/datasets/cstf_spokencoco/SpokenCOCO"
DOWNLOAD_CACHE = "/content/drive/MyDrive/datasets/cstf_spokencoco_download_cache"
MANIFEST_PATH = "/content/drive/MyDrive/datasets/spokencoco_manifest.json"
RUNS_ROOT = "/content/drive/MyDrive/jacobian-lens-gemma/runs"

# Never written into. Completed runs and published lenses are evidence.
PROTECTED_RUN_PREFIXES = (
    "mmpilot_pilot_", "mmrobust_", "mmlocalize_", "rgcalib_", "rgext_",
    "audioaudit_", "text_jlens_", "mmaudio_", "mmconv_", "mml32_", "mml32res_",
)

SAMPLE_PLAN = plan_sample_size()
STAGE_PLAN = stage_plan()

if RUN_REAL_PRECONVERGENCE_STUDY:
    N_TRAIN_POSITIVE_IMAGES = SAMPLE_PLAN.n_train_positive_images
    N_TEST_POSITIVE_IMAGES = SAMPLE_PLAN.n_test_positive_images
    N_TRAIN_NEGATIVE_IMAGES = SAMPLE_PLAN.n_train_negative_images
    N_TEST_NEGATIVE_IMAGES = SAMPLE_PLAN.n_test_negative_images
else:
    N_TRAIN_POSITIVE_IMAGES = N_TEST_POSITIVE_IMAGES = 2
    N_TRAIN_NEGATIVE_IMAGES = N_TEST_NEGATIVE_IMAGES = 2

SCRATCH = Path(os.environ.get("MMPILOT_SCRATCH") or "/content/preconv_scratch")
SCRATCH.mkdir(parents=True, exist_ok=True)
RESOLVED_RUNS_ROOT = Path(
    os.environ.get("MMPILOT_RUNS_ROOT")
    or (RUNS_ROOT if RUN_REAL_PRECONVERGENCE_STUDY else SCRATCH / "runs")
)
PARENT_RUN_DIR = os.environ.get("MMPILOT_PARENT_RUN_DIR") or PARENT_RUN_DIR
EXTENSION_RUN_DIR = os.environ.get("MMPILOT_EXTENSION_RUN_DIR") or EXTENSION_RUN_DIR
if os.environ.get("MMPILOT_COMPLETED_RUN_DIRS"):
    COMPLETED_RUN_DIRS = tuple(
        os.environ["MMPILOT_COMPLETED_RUN_DIRS"].split(os.pathsep)
    )
if os.environ.get("MMPILOT_RESOLUTION_RUN_DIR"):
    COMPLETED_RESOLUTION_RUN_DIR = os.environ["MMPILOT_RESOLUTION_RUN_DIR"]

MODALITIES = ("text", "image", "spoken_audio")
PROFILE = (
    IMAGE_UNIQUE_PROFILE
    if RUN_REAL_PRECONVERGENCE_STUDY
    else IMAGE_UNIQUE_MOCK_PROFILE
)

# The MOCK multimodal decoder has six blocks. Its stand-in for "the selected
# adjacent layer" is layer 1 — the same substitution every completed MOCK run
# uses. The MOCK *calibration* model, by contrast, has the real 42-block depth,
# so the fit and the gate run at the genuine physical layers.
MOCK_MULTIMODAL_LAYER = 1
MOCK_D_MODEL = 24

SELECTED_CONCEPTS_EXPECTED = list(FROZEN_SELECTED_CONCEPTS)
FOCAL_CONCEPTS_EXPECTED = list(FROZEN_FOCAL_CONCEPTS)

THRESHOLDS = TriModalThresholds(
    capability_threshold=CAPABILITY_THRESHOLD,
    required_positive_images_per_cell=N_TEST_POSITIVE_IMAGES,
    required_negative_images_per_cell=N_TEST_NEGATIVE_IMAGES,
)

SWITCHES = {
    name: globals()[name]
    for name in (
        "RUN_REAL_PRECONVERGENCE_STUDY",
        "PREPROCESSING_ONLY",
        "RUN_LENS_FITTING",
        "CONFIRM_FITTING_BUDGET",
        "RUN_UNTOUCHED_CONFIRMATION",
        "RUN_MODEL_STAGE",
        "CONFIRM_MODEL_LOAD",
        "CONFIRM_STAGE_3_BUDGET",
        "RUN_STAGE_4_CAUSAL_TRANSFER",
        "CONFIRM_STAGE_4_BUDGET",
    )
}


def refresh_gates():
    """Re-derive every gate from the raw switches. Called before any spend.

    The MOCK override lives here rather than in each caller so it cannot be
    applied in one cell and forgotten in the next: the real gates protect a
    ~16 GB download and L4 hours, and in MOCK the "model" is a few-hundred-
    parameter CPU stub with nothing to protect. Stage 4 stays opt-in in BOTH
    modes, because it is the switch that changes what is claimed.
    """
    gates = derive_preconvergence_gates(globals())
    if PREPROCESSING_ONLY:
        globals().update(gates)
        return gates
    if not RUN_REAL_PRECONVERGENCE_STUDY:
        gates = {
            **gates,
            "FITTING_ENABLED": True,
            "CONFIRMATION_ENABLED": True,
            "MODEL_STAGE_ENABLED": True,
            "STAGE_3_ENABLED": True,
            # Re-derived from the Stage-4 switches alone: the real rule chains
            # Stage 4 onto CONFIRM_STAGE_3_BUDGET, and leaving it chained would
            # report STAGE_4_REQUESTED=False in a MOCK run that plainly ran it.
            "STAGE_4_REQUESTED": bool(
                RUN_STAGE_4_CAUSAL_TRANSFER and CONFIRM_STAGE_4_BUDGET
            ),
        }
    globals().update(gates)
    return gates


GATES = refresh_gates()

print(format_preconvergence_gates(GATES, switches=SWITCHES))
print()
print(f"protocol            {PRECONVERGENCE_PROTOCOL}")
print(f"lens protocol       {ADJACENT_PROTOCOL_VERSION}")
print(f"run family          {PRECONVERGENCE_RUN_PREFIX}_*")
print(f"candidates          {list(ADJACENT_CANDIDATE_LAYERS)}   (closed)")
print(f"  interval floor    L{FAILED_LOWER_LAYER}  failed lens confirmation")
print(f"  interval ceiling  L{AMBIGUOUS_UPPER_LAYER}  confirmed, AMBIGUOUS twice")
print(f"  reference         L{CONVERGED_REFERENCE_LAYER}  CONVERGED")
print(f"fitting scale       {ADJACENT_FITTING_SCALE}   (no scale search)")
print(f"hook site           {ADJACENT_HOOK_SITE}")
print(f"gate                {ADJACENT_CONFIRMATION_GATE.version}")
print(f"gate digest         {ADJACENT_CONFIRMATION_GATE.digest}")
print(f"selection rule      {ADJACENT_SELECTION_RULE.digest}")
print(f"criterion digest    {FROZEN_CRITERION_DIGEST}")
print(f"prompt protocol     {OPEN_PROMPT_PROTOCOL}")
print(f"intervention        {INTERVENTION_FAMILY}")
print(f"terminal outcomes   {list(TERMINAL_OUTCOMES)}")
print()
print(COORDINATE_SWAP_SCOPE)
if PREPROCESSING_ONLY:
    print()
    print("PREPROCESSING_ONLY is True: every model gate is forced closed.")
'''
)

# ================================================================== 3. Drive

markdown(
    """
## 3. Mount Drive and verify the configured paths (read-only)

Skipped entirely in MOCK. On the real path every configured location is checked
for existence before anything is loaded, the completed runs are checked
**read-only**, and the `mml32res_*` pin is required here rather than discovered
later.
"""
)

code(
    '''
# 3. Mount Drive and verify the configured paths exist. Read-only.
from jlens.mmpilot.preconvergence import assert_completed_population_pins

DRIVE_STATUS = "skipped"
POPULATION_PINS = None

if RUN_REAL_PRECONVERGENCE_STUDY and IN_COLAB:
    from google.colab import drive

    drive.mount("/content/drive")
    DRIVE_STATUS = "mounted"

if RUN_REAL_PRECONVERGENCE_STUDY:
    POPULATION_PINS = assert_completed_population_pins(
        COMPLETED_RUN_DIRS,
        resolution_run_dir=COMPLETED_RESOLUTION_RUN_DIR,
    )
    _required = {
        "parent calibration run": PARENT_RUN_DIR,
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
    _prep_parent = Path(PREP_CACHE_ROOT).parent
    if not _prep_parent.is_dir():
        raise RuntimeError(
            f"the preparation cache's parent directory {_prep_parent} does not "
            "exist, so Stage 0 has nowhere durable to checkpoint to"
        )
    print(f"  ok  {'prep cache parent':26s} {_prep_parent}")
    print()
    print(f"pinned mml32res run  {POPULATION_PINS['pinned_resolution_run']}")
    print(f"  discovered         {POPULATION_PINS['pin_was_discovered']}")
    print(f"  defaulted          {POPULATION_PINS['pin_was_defaulted']}")
print(f"drive: {DRIVE_STATUS}")
'''
)

# =============================================================== 4. runtime

markdown(
    """
## 4. Runtime report

Prints the accelerator and the library versions. Touches no dataset and loads no
model. Stage 0 is CPU and Drive I/O only; a GPU makes none of it faster.
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

if RUN_REAL_PRECONVERGENCE_STUDY and TRANSFORMERS_VERSION != (
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

# ==================================================== 5. pre-download checks

markdown(
    """
## 5. Everything cheap that can fail, failing before the 16 GB download

Every wasted L4 start had the same shape: an hour of setup, the model download,
then a `TypeError` or an `ImportError` from a line that only executes on the
real branch. This cell binds **every** real-branch call site against the
installed signatures — 65 of them, imported from the same checked-out commit the
notebook will import from — and checks the frozen digests, in milliseconds,
without loading anything.

It also asserts the gate identity that the whole study rests on:
`ADJACENT_GATE is EXTENSION_GATE`. Not "equal field by field" — the *same
object*, so there is no copy that could drift.

Run this cell before setting any model switch. If it prints anything other than
`PASS`, the L4 would have died on it.
"""
)

code(
    '''
# 5. Pre-download contract and artifact checks. Loads nothing.
from jlens.mmpilot.preconvergence import check_preconvergence_call_contracts
from jlens.mmpilot.preflight import check_call_contracts

PREDOWNLOAD = {"schema": "jlens.mmpilot.preconvergence_predownload.v1"}

_shared = check_call_contracts()
_study = check_preconvergence_call_contracts()
PREDOWNLOAD["shared_contract_failures"] = _shared
PREDOWNLOAD["study_contract_failures"] = _study

# The gate is the extension's own object, not a copy. Identity, not equality:
# an equal copy can drift on the next edit and this cannot.
PREDOWNLOAD["gate_is_extension_gate"] = ADJACENT_GATE is EXTENSION_GATE
PREDOWNLOAD["gate_digest"] = ADJACENT_CONFIRMATION_GATE.digest
PREDOWNLOAD["criterion_digest_expected"] = FROZEN_CRITERION_DIGEST

from jlens.mmpilot.convergence import CONVERGENCE_CRITERION

PREDOWNLOAD["criterion_digest_installed"] = CONVERGENCE_CRITERION.digest
PREDOWNLOAD["criterion_digest_matches"] = (
    CONVERGENCE_CRITERION.digest == FROZEN_CRITERION_DIGEST
)
PREDOWNLOAD["candidate_layers"] = list(ADJACENT_CANDIDATE_LAYERS)
PREDOWNLOAD["fitting_scale"] = ADJACENT_FITTING_SCALE
PREDOWNLOAD["protocol_digest"] = ADJACENT_PROTOCOL.digest
PREDOWNLOAD["selection_rule_digest"] = ADJACENT_SELECTION_RULE.digest

_problems = []
if _shared:
    _problems += [f"shared contract: {row}" for row in _shared]
if _study:
    _problems += [f"study contract: {row}" for row in _study]
if not PREDOWNLOAD["gate_is_extension_gate"]:
    _problems.append(
        "ADJACENT_GATE is not the extension's gate object; a copy has been "
        "introduced and the thresholds can now drift apart"
    )
if not PREDOWNLOAD["criterion_digest_matches"]:
    _problems.append(
        "the installed convergence criterion digest "
        f"{CONVERGENCE_CRITERION.digest} != the frozen "
        f"{FROZEN_CRITERION_DIGEST}; the thresholds are predeclared and are not "
        "revisable"
    )
if tuple(ADJACENT_CANDIDATE_LAYERS) != (27, 28, 29, 30, 31):
    _problems.append(
        f"the candidate interval is {list(ADJACENT_CANDIDATE_LAYERS)}, not "
        "(27, 28, 29, 30, 31). The interval is closed and is not widened."
    )
PREDOWNLOAD["problems"] = _problems
PREDOWNLOAD["passed"] = not _problems

print(f"shared call contracts   {len(_shared)} failure(s)")
print(f"study call contracts    {len(_study)} failure(s)")
print(f"gate is extension gate  {PREDOWNLOAD['gate_is_extension_gate']}")
print(f"criterion digest        {PREDOWNLOAD['criterion_digest_matches']}")
print(f"candidate interval      {PREDOWNLOAD['candidate_layers']}")
print()
if _problems:
    for _problem in _problems:
        print(f"  FAIL  {_problem}")
    raise RuntimeError(
        "pre-download checks failed; the model load would have died on these. "
        "Nothing has been downloaded."
    )
print("PRE-DOWNLOAD CHECKS: PASS")
'''
)

# ============================================== 6. protocol, gate, stage plan

markdown(
    """
## 6. The frozen protocol, the unchanged gate, the selection rule, the plan

All four are printed before any number exists, and all four are digested into
the run fingerprint. Editing one invalidates stored results rather than
re-deciding on them.
"""
)

code(
    '''
# 6. Print the frozen design. Computes nothing.
print(ADJACENT_PROTOCOL.text())
print()
print(adjacent_gate_text())
print()
print(ADJACENT_SELECTION_RULE.text())
print()
print(format_stage_plan(STAGE_PLAN))
'''
)

# ================================ 7. Stage 0a — corpus and the fit ordering

markdown(
    """
## 7. Stage 0a — corpus provenance and the fit ordering (CPU, no model)

The completed calibration stored split **checksums**, not corpus text, so the
only way to recover the prompt list this study must fit on is to re-stream the
pinned corpus under the parent's own collection parameters — every one read from
the parent's artifacts, with its source named — and prove the result reproduces
those checksums.

The fit ordering is then the extension's, unchanged: positions `0 … 99` are the
parent's fit partition in the parent's nested order, and any further positions
come from records the parent's collection never reached, in ascending stream
index. Prompt identity is therefore shared with the completed studies even
though **the accumulator is not**.

`assert_new_source_layers` is the guardrail that says so out loud: the parent
holds `[8, 14, 20, 26, 32, 35, 38, 40]`, this study fits `[27 … 31]`, the two
sets are disjoint, and no checkpoint is seeded from the parent.
"""
)

code(
    '''
# 7. Re-stream the corpus, prove the reconstruction, build the fit ordering.
from jlens.calibration.adjacent import (
    AdjacentStore,
    adjacent_corpus_manifest,
    assert_new_source_layers,
    build_untouched_confirmation,
)
from jlens.calibration.corpus import build_records
from jlens.calibration.extension import (
    build_extension_fit_order,
    build_fresh_evaluation_splits,
    parent_collection_parameters,
    verify_fit_prefix,
    verify_reconstructed_partitions,
)
from jlens.calibration.fitting import filter_records_by_tokens
from jlens.calibration.parent import (
    ParentRequirements,
    audit_parent_run,
    discover_parent_files,
    load_parent_run,
    protected_parent_checksums,
)
from jlens.calibration.plan import build_capture_plan, normalized_depth

MOCK_PARENT = None
CALIBRATION_MODEL = None

if RUN_REAL_PRECONVERGENCE_STUDY:
    PARENT_ROOT = Path(PARENT_RUN_DIR)
    PARENT_BASELINE = 100
    ADJACENT_SCALE = ADJACENT_FITTING_SCALE
else:
    # MOCK: produce a parent by running the completed study's OWN code against a
    # tiny frozen CPU stack at the REAL depth (42 blocks), so the adjacent fit
    # below happens at the genuine physical layers 27-31.
    from jlens.calibration.extension_mock import (
        MOCK_BASELINE_SCALE,
        build_mock_parent_run,
    )
    from jlens.calibration.mock import MockCalibrationModel

    CALIBRATION_MODEL = MockCalibrationModel()
    MOCK_PARENT = build_mock_parent_run(
        SCRATCH / "mock_parent", model=CALIBRATION_MODEL
    )
    PARENT_ROOT = Path(MOCK_PARENT.root)
    PARENT_BASELINE = MOCK_BASELINE_SCALE
    ADJACENT_SCALE = 12

PARENT_INVENTORY = discover_parent_files(PARENT_ROOT)
PARENT = load_parent_run(PARENT_ROOT, baseline_scale=PARENT_BASELINE)
PARENT_PLAN = PARENT.capture_plan
PARENT_CHECKSUMS_BEFORE = protected_parent_checksums(
    PARENT_ROOT, layout=PARENT.layout
)

# THE guardrail. The parent is read for provenance; its accumulator is not
# continued, because it holds no layer in 27-31.
SOURCE_LAYERS = assert_new_source_layers(
    candidate_layers=ADJACENT_CANDIDATE_LAYERS,
    parent_source_layers=PARENT.accumulator.source_layers,
)

PARENT_REQUIREMENTS = ParentRequirements(
    model_repo_id=(
        MODEL_REPO_ID
        if RUN_REAL_PRECONVERGENCE_STUDY
        else PARENT.fingerprint["model_repo_id"]
    ),
    model_revision=(
        MODEL_REVISION
        if RUN_REAL_PRECONVERGENCE_STUDY
        else PARENT.fingerprint["model_revision"]
    ),
    tokenizer_repo_id=(
        MODEL_REPO_ID
        if RUN_REAL_PRECONVERGENCE_STUDY
        else PARENT.fingerprint["model_repo_id"]
    ),
    tokenizer_revision=PARENT.fingerprint["tokenizer_revision"],
    source_layers=tuple(PARENT.accumulator.source_layers),
    target_layer=PARENT.accumulator.target_layer,
    d_model=PARENT_PLAN["d_model"],
    hook_site="block_output",
    skip_first=PARENT_PLAN["skip_first"],
    max_seq_len=PARENT_PLAN["max_seq_len"],
    dim_batch=PARENT_PLAN["dim_batch"],
    corpus_hf_dataset=(
        CORPUS_PRIMARY["hf_dataset"] if RUN_REAL_PRECONVERGENCE_STUDY else "mock"
    ),
    corpus_config=(
        CORPUS_PRIMARY["config"] if RUN_REAL_PRECONVERGENCE_STUDY else "mock"
    ),
    corpus_split=CORPUS_PRIMARY["split"],
    estimator="jlens.fitting.fit (upstream, unmodified)",
    artifact_format_version=ARTIFACT_FORMAT_VERSION,
    baseline_scale=PARENT_BASELINE,
    expected_n_done=PARENT_BASELINE,
)
PARENT_AUDIT = audit_parent_run(PARENT, requirements=PARENT_REQUIREMENTS)

# The capture plan for THIS study. Same target layer, same positions, same
# dtype, same estimator; only the source layers move.
ADJACENT_PLAN = build_capture_plan(
    layers=tuple(ADJACENT_CANDIDATE_LAYERS),
    target_layer=(
        TARGET_LAYER if RUN_REAL_PRECONVERGENCE_STUDY else PARENT.accumulator.target_layer
    ),
    d_model=EXPECT_D_MODEL if RUN_REAL_PRECONVERGENCE_STUDY else PARENT_PLAN["d_model"],
    dim_batch=DIM_BATCH if RUN_REAL_PRECONVERGENCE_STUDY else PARENT_PLAN["dim_batch"],
    max_seq_len=(
        MAX_SEQ_LEN if RUN_REAL_PRECONVERGENCE_STUDY else PARENT_PLAN["max_seq_len"]
    ),
    skip_first=(
        SKIP_FIRST if RUN_REAL_PRECONVERGENCE_STUDY else PARENT_PLAN["skip_first"]
    ),
    n_layers=EXPECT_N_LAYERS,
)

COLLECTION = parent_collection_parameters(PARENT)

if RUN_REAL_PRECONVERGENCE_STUDY:
    from datasets import load_dataset

    CORPUS_CONFIG = dict(CORPUS_PRIMARY)
    CORPUS_ID = PARENT.corpus["corpus_id"]
    from jlens.calibration.corpus import collect_records_for_partition_quotas

    _stream = load_dataset(
        CORPUS_CONFIG["hf_dataset"],
        CORPUS_CONFIG["config"],
        split=CORPUS_CONFIG["split"],
        streaming=True,
    )
    _RECORDS, _PARTITIONS = collect_records_for_partition_quotas(
        corpus_id=CORPUS_ID,
        texts=(_record["text"] for _record in _stream),
        min_chars=COLLECTION["min_chars"],
        min_fit=COLLECTION["min_fit"],
        max_texts=COLLECTION["max_texts"],
        seed=COLLECTION["seed"],
        n_validation=COLLECTION["n_validation"],
        n_confirmation=COLLECTION["n_confirmation"],
    )
    RECONSTRUCTION = verify_reconstructed_partitions(_PARTITIONS, parent=PARENT)
    OLD_FIT = _PARTITIONS.fit
    OLD_VALIDATION = _PARTITIONS.validation
    OLD_CONFIRMATION = _PARTITIONS.confirmation

    _LAST_STREAM_INDEX = max(record.stream_index for record in _RECORDS)
    _extra_stream = load_dataset(
        CORPUS_CONFIG["hf_dataset"],
        CORPUS_CONFIG["config"],
        split=CORPUS_CONFIG["split"],
        streaming=True,
    )
    # Room for the extension's 256+256, this study's 256, and the fit tail.
    _extra_needed = COLLECTION["max_texts"] + 60 * (
        ADJACENT_SCALE + 2 * N_DEVELOPMENT_PROMPTS + 2 * N_CONFIRMATION_PROMPTS
    )
    _extra_texts = []
    for _index, _record in enumerate(_extra_stream):
        if _index >= _extra_needed:
            break
        _extra_texts.append(_record["text"])
    EXTENSION_POOL = [
        record
        for record in build_records(
            CORPUS_ID, _extra_texts, min_chars=COLLECTION["min_chars"]
        )
        if record.stream_index > _LAST_STREAM_INDEX
    ]
else:
    from jlens.calibration.extension_mock import mock_extension_pool

    CORPUS_CONFIG = dict(MOCK_PARENT.corpus_config)
    CORPUS_ID = PARENT.corpus["corpus_id"]
    RECONSTRUCTION = verify_reconstructed_partitions(
        MOCK_PARENT.partitions, parent=PARENT
    )
    OLD_FIT = MOCK_PARENT.partitions.fit
    OLD_VALIDATION = MOCK_PARENT.partitions.validation
    OLD_CONFIRMATION = MOCK_PARENT.partitions.confirmation
    EXTENSION_POOL = mock_extension_pool(MOCK_PARENT)

# Records too short to contribute a valid source position are dropped, exactly
# as the parent did. This needs a TOKENIZER, not the model: the ~16 GB weights
# are not touched here, so Stage 0 stays a CPU session.
if RUN_REAL_PRECONVERGENCE_STUDY:
    from transformers import AutoTokenizer

    STAGE0_TOKENIZER = AutoTokenizer.from_pretrained(
        MODEL_REPO_ID, revision=MODEL_REVISION, token=os.environ.get("HF_TOKEN")
    )

    def _token_count(text):
        return len(
            STAGE0_TOKENIZER(
                text,
                add_special_tokens=True,
                truncation=True,
                max_length=ADJACENT_PLAN.max_seq_len,
            )["input_ids"]
        )
else:
    STAGE0_TOKENIZER = None
    _token_count = CALIBRATION_MODEL.tokenizer.token_count

PARENT_FIT_RECORDS, DROPPED_SHORT = filter_records_by_tokens(
    OLD_FIT,
    token_count=_token_count,
    skip_first=ADJACENT_PLAN.skip_first,
    max_seq_len=ADJACENT_PLAN.max_seq_len,
)
EXTENSION_POOL, DROPPED_SHORT_POOL = filter_records_by_tokens(
    EXTENSION_POOL,
    token_count=_token_count,
    skip_first=ADJACENT_PLAN.skip_first,
    max_seq_len=ADJACENT_PLAN.max_seq_len,
)

FIT_RECORDS = build_extension_fit_order(
    PARENT_FIT_RECORDS,
    n_needed=ADJACENT_SCALE,
    extension_pool=EXTENSION_POOL,
)
FIT_PREFIX = verify_fit_prefix(
    FIT_RECORDS,
    n_parent=min(PARENT_BASELINE, ADJACENT_SCALE),
    parent_prefix_checksum=PARENT.fit_prefix_checksum(
        min(PARENT_BASELINE, ADJACENT_SCALE)
    ),
)

print(f"parent run          {PARENT_ROOT}")
print(f"  files             {PARENT_INVENTORY['n_files']}")
print(f"  audit             {'PASS' if PARENT_AUDIT['compatible'] else 'FAIL'}")
print(f"  accumulator layers{list(PARENT.accumulator.source_layers)}")
print()
print("SOURCE LAYERS — new accumulation, never a continuation")
print(f"  candidates        {SOURCE_LAYERS['candidate_layers']}")
print(f"  parent grid       {SOURCE_LAYERS['parent_source_layers']}")
print(f"  overlap           {SOURCE_LAYERS['overlap']}  disjoint="
      f"{SOURCE_LAYERS['disjoint']}")
print(f"  parent seeded     {SOURCE_LAYERS['parent_accumulator_may_be_seeded']}")
print(f"  {SOURCE_LAYERS['why']}")
print()
print(f"capture plan        layers {list(ADJACENT_PLAN.layers)} -> "
      f"L{ADJACENT_PLAN.target_layer}")
print(f"  normalized depth  {[normalized_depth(l) for l in ADJACENT_PLAN.layers]}")
print(f"  backward span     {ADJACENT_PLAN.backward_span} blocks")
print(f"  plan digest       {ADJACENT_PLAN.digest}")
print()
print(f"reconstruction      {'EXACT' if RECONSTRUCTION['all_match'] else 'FAILED'}")
for _row in RECONSTRUCTION["partitions"]:
    print(f"  {_row['partition']:<13} {_row['actual_size']:>7,} records  "
          f"matches={_row['matches']}")
print(f"extension pool      {len(EXTENSION_POOL):,} records the parent never saw")
print(f"fit ordering        {len(FIT_RECORDS):,} records at scale {ADJACENT_SCALE}")
print(f"prefix verified     {FIT_PREFIX['matches']}")
'''
)

# ================== 8. Stage 0b — development reuse and untouched confirmation

markdown(
    """
## 8. Stage 0b — the reused development set and a genuinely untouched confirmation

**Development is reused, and its role is recorded.** The extension's 256
development records were designated as development, they were used as
development, and they are used as development again here. Their checksum and
record ids go into the manifest so "reused development" is a fact on the record
rather than a claim.

**Confirmation is not reused, and cannot be.** The extension's confirmation set
has been opened — that opening is *how* L26's failure and L32's pass are known —
so for these five candidates it is development history. A new set is drawn under
a new seed and a new `|adj|` bucket tag, from corpus that nothing in this lineage
has reached, excluding by exact normalized checksum **and** by banded SimHash at
Hamming distance ≤ 3:

* every fitting prompt (the parent's fit partition and this study's fit list);
* every development prompt (the parent's validation set and the extension's);
* every previously opened confirmation prompt (the parent's and the extension's);
* every record in the prior calibration manifests named as dependencies.

Then an **independent** audit runs on the constructed set — a different predicate
from the filter that built it, because a proof derived from the filter proves
only that the filter agrees with itself.

**If 256 untouched records cannot be built, the study is blocked.** It is not
resized, no substitute corpus is reached for, and no opened set is recycled.
"""
)

code(
    '''
# 8. Reconstruct the extension's splits, then build the untouched set.
from jlens.calibration.adjacent import audit_untouched_confirmation
from jlens.mmpilot.store import payload_checksum

# The extension's own development/confirmation sets, rebuilt deterministically
# from the same pool under the same protocol. Rebuilt rather than read, so that
# what is excluded here is provably what the extension actually held.
EXTENSION_EXCLUDED = {
    "old_fit": OLD_FIT,
    "old_development": OLD_VALIDATION,
    "old_confirmation": OLD_CONFIRMATION,
    "new_fit": FIT_RECORDS,
}
EXTENSION_SPLITS = build_fresh_evaluation_splits(
    EXTENSION_POOL,
    excluded=EXTENSION_EXCLUDED,
    corpus_id=CORPUS_ID,
    n_development=N_DEVELOPMENT_PROMPTS,
    n_confirmation=N_CONFIRMATION_PROMPTS,
)
EXTENSION_SPLITS_MANIFEST = EXTENSION_SPLITS.manifest()

DEVELOPMENT_RECORDS = list(EXTENSION_SPLITS.development)
DEVELOPMENT_ROLE = {
    "source": "the early-layer extension's designated development partition",
    "role_then": "development",
    "role_now": "development",
    "reused": True,
    "size": len(DEVELOPMENT_RECORDS),
    "checksum": EXTENSION_SPLITS.checksum("development"),
    "record_ids": EXTENSION_SPLITS.record_ids("development"),
    "why_reuse_is_admissible": (
        "a development set is designated to be looked at; reusing one changes "
        "nothing about what it can support, and it supports nothing on its own"
    ),
}

DEPENDENCY_MANIFESTS = {
    "parent_corpus_manifest_checksum": PARENT.corpus.get("corpus_manifest_checksum"),
    "parent_split_checksums": dict(PARENT.split_checksums),
    "extension_splits_manifest_checksum": EXTENSION_SPLITS_MANIFEST[
        "manifest_checksum"
    ],
    "extension_development_checksum": EXTENSION_SPLITS.checksum("development"),
    "extension_confirmation_checksum": EXTENSION_SPLITS.checksum("confirmation"),
}

UNTOUCHED_EXCLUDED = {
    "parent_fit": OLD_FIT,
    "parent_development": OLD_VALIDATION,
    "parent_confirmation_opened": OLD_CONFIRMATION,
    "adjacent_fit": FIT_RECORDS,
    "extension_development_reused_here": DEVELOPMENT_RECORDS,
    "extension_confirmation_opened": list(EXTENSION_SPLITS.confirmation),
}

CONFIRMATION = build_untouched_confirmation(
    EXTENSION_POOL,
    excluded=UNTOUCHED_EXCLUDED,
    corpus_id=CORPUS_ID,
    n_confirmation=N_CONFIRMATION_PROMPTS,
    development_role=DEVELOPMENT_ROLE,
    dependency_manifests=sorted(DEPENDENCY_MANIFESTS),
)
UNTOUCHED_AUDIT = audit_untouched_confirmation(
    CONFIRMATION, excluded=UNTOUCHED_EXCLUDED
)
CONFIRMATION_MANIFEST = CONFIRMATION.manifest()

CORPUS_MANIFEST = adjacent_corpus_manifest(
    corpus_config=CORPUS_CONFIG,
    corpus_id=CORPUS_ID,
    fit_records=FIT_RECORDS,
    development_records=DEVELOPMENT_RECORDS,
    confirmation=CONFIRMATION,
    scale=ADJACENT_SCALE,
    dependency_manifests=DEPENDENCY_MANIFESTS,
)

print("DEVELOPMENT — reused, with its role and hashes recorded")
print(f"  size              {DEVELOPMENT_ROLE['size']}")
print(f"  checksum          {DEVELOPMENT_ROLE['checksum']}")
print(f"  role then / now   {DEVELOPMENT_ROLE['role_then']} / "
      f"{DEVELOPMENT_ROLE['role_now']}")
print()
print("UNTOUCHED CONFIRMATION — built, then independently proved")
print(f"  size              {CONFIRMATION_MANIFEST['size']} "
      f"(required {N_CONFIRMATION_PROMPTS}; never reduced)")
print(f"  checksum          {CONFIRMATION_MANIFEST['checksum']}")
print(f"  pool after excl.  {CONFIRMATION.n_pool:,}")
print(f"  excluded exact    {CONFIRMATION.excluded_exact}")
print(f"  excluded near     {CONFIRMATION.excluded_near}")
print(f"  pool duplicates   {CONFIRMATION.excluded_pool_duplicates}")
print()
print("INDEPENDENT AUDIT — on the constructed set, by a different predicate")
print(f"  disjoint from     {UNTOUCHED_AUDIT['required_disjoint_from']}")
print(f"  exact hits        {UNTOUCHED_AUDIT['n_exact_hits']}")
print(f"  near hits         {UNTOUCHED_AUDIT['n_near_hits']}")
print(f"  internal dupes    {UNTOUCHED_AUDIT['n_internal_duplicates']}")
print(f"  pairs compared    {UNTOUCHED_AUDIT['candidate_pairs_compared']:,}")
print(f"  UNTOUCHED         {UNTOUCHED_AUDIT['untouched']}")
print()
print(f"corpus manifest     {CORPUS_MANIFEST['corpus_manifest_checksum']}")
print(f"dependencies        {sorted(DEPENDENCY_MANIFESTS)}")
'''
)

# ========================= 9. Stage 0c — the cached SpokenCOCO evidence join

markdown(
    """
## 9. Stage 0c — the evidence join is **loaded, never rebuilt**

Rediscovering the SpokenCOCO join costs twenty-five minutes of CPU and produces
the same answer every time. `load_expanded_manifest` verifies, in order: the
derivation schema version, the original manifest checksum, every source metadata
checksum, the conversion hash (which covers the evidence rule and the lexicon
hash), and the cached group count. Any failure is a refusal naming the clause and
its two values — never a silent rebuild. `ALLOW_MANIFEST_REBUILD` is the one
switch that permits one, and it is `False` in the committed notebook.
"""
)

code(
    '''
# 9a. MOCK ONLY — manufacture the dataset, the cache, and three completed runs.
#
# On the real path this cell does nothing: the cache and the completed runs are
# already on Drive. In MOCK they are built here so section 9b exercises the
# LOADING path and section 10 exercises the real exclusion harvest and the real
# population pin check.
from jlens.mmpilot import expansion as expansion_module
from jlens.mmpilot import manifest as manifest_module

MOCK_WORLD = None
CACHE_CONVERSION = None
if not RUN_REAL_PRECONVERGENCE_STUDY:
    from jlens.mmpilot.mock import MockWorld, build_mock_dataset

    MOCK_WORLD = MockWorld({
        "bus": ("bus", "buses"),
        "cat": ("cat", "cats"),
        "clock": ("clock", "clocks"),
        "dog": ("dog", "dogs"),
        "pizza": ("pizza", "pizzas"),
        "zebra": ("zebra", "zebras"),
    })
    # Six times the photographs the design needs, so three simulated completed
    # runs can genuinely spend media and an independent population still exists.
    if not (SCRATCH / "data" / "spokencoco_manifest.json").is_file():
        build_mock_dataset(
            SCRATCH / "data",
            world=MOCK_WORLD,
            images_per_concept=8 * (N_TRAIN_POSITIVE_IMAGES + N_TEST_POSITIVE_IMAGES),
            negative_images=(N_TRAIN_NEGATIVE_IMAGES + N_TEST_NEGATIVE_IMAGES) * 8,
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
if RUN_REAL_PRECONVERGENCE_STUDY:
    SEARCH_ROOTS = sorted({
        str(c) for c in (SPOKENCOCO_BASE_ROOT, IMAGE_MEDIA_ROOT, AUDIO_MEDIA_ROOT,
                         DOWNLOAD_CACHE)
        if Path(c).is_dir()
    })

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

if not RUN_REAL_PRECONVERGENCE_STUDY and not Path(
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
# 9b. Load the cached evidence join DIRECTLY. build_expanded_manifest is not
# called on this path, and the flag below records that as a fact, not a claim.
from jlens.mmpilot import evidence as evidence_module
from jlens.mmpilot.concepts import universe_from_concept_annotations
from jlens.mmpilot.expansion import (
    ExpandedManifestIncompatible,
    load_expanded_manifest,
)

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
            f"{error}\\n\\nThe cache was NOT rebuilt. Rebuilding the join takes "
            "~25 minutes and would produce a manifest this study has no reason "
            "to believe is the one the completed runs used. Fix the configured "
            "path, or set ALLOW_MANIFEST_REBUILD=True deliberately."
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

UNIVERSE = universe_from_concept_annotations(GROUPS)
EVIDENCE_CONFIG = evidence_module.config_from_specs(UNIVERSE.specs)
CONCEPT_CANDIDATES = UNIVERSE.lexicon()
CACHE_LOAD["n_categories_recovered"] = len(UNIVERSE.categories)
CACHE_LOAD["category_ids_available"] = UNIVERSE.sources[0]["category_ids_available"]

print(f"cache          {CACHE_LOAD['path']}")
print(f"schema         {CACHE_LOAD['schema_version']}")
print(f"groups         {CACHE_LOAD['n_groups']:,}")
print(f"file checksum  {DERIVED_MANIFEST_CHECKSUM}")
print(f"lexicon hash   {CACHE_LOAD['evidence_lexicon_hash']}")
print(f"categories     {CACHE_LOAD['n_categories_recovered']} recovered")
print(f"build_expanded_manifest called on this path: "
      f"{CACHE_LOAD['build_expanded_manifest_called']}")
'''
)

# ======================= 10. Stage 0d — the independent multimodal population

markdown(
    """
## 10. Stage 0d — the independent population, checkpointed and then **proved**

This is the long part, and it is CPU and Drive I/O. It is a **checkpointed
preparation** under a deterministic cache directory keyed by a pre-model
fingerprint. Stop the runtime whenever you like: at most one bounded batch of
≤25 files (or ≤30 seconds) is repeated, and the scan never restarts from file
zero while a valid checkpoint exists.

Three completed populations are excluded, not two — the audio transfer run, the
open-prompt L32 follow-up, and the **pinned** `mml32res_*` convergence-resolution
run. Disjointness must hold on all of: `image_id`, `group_id`, audio/recording
path, caption identity, and media checksum where available.

The six behavioural concepts are **set directly, in frozen order**, before any
model output exists. The independent pool's ranking is computed and printed
because it is informative, and it is **descriptive only**: removing spent
photographs changes the ranking legitimately. What is checked is each frozen
concept against each predeclared requirement. A concept that genuinely falls
short is a refusal with its shortfall printed — never a substitution.
"""
)

code(
    '''
# 10a. Resolve the deterministic preparation cache. NO source file is read here.
from datetime import datetime, timezone

from jlens.mmpilot import prep_cache as prep
from jlens.mmpilot.l32_resolution import (
    independent_pool,
    resolve_excluded_media,
)

if not RUN_REAL_PRECONVERGENCE_STUDY:
    # Three simulated completed runs, named EXACTLY as the real ones are, so
    # the population-pin check below is genuinely exercised in MOCK rather than
    # skipped. Each spends a distinct third of the PHOTOGRAPHS; spending groups
    # instead would scatter the exclusion over nearly every photograph and leave
    # a pool that cannot support any design.
    from jlens.mmpilot.mock import build_mock_completed_run
    from jlens.mmpilot.selection import stable_rank

    _images = sorted(
        {str(g["image_id"]) for g in GROUPS},
        key=lambda image_id: stable_rank(image_id, "mock-completed-runs"),
    )
    _third = max(1, len(_images) // 6)
    _mock_runs = []
    for _index, _name in enumerate((
        COMPLETED_AUDIO_TRANSFER_RUN,
        COMPLETED_FOLLOWUP_RUN,
        "mml32res_mock_completed_20260809T000000",
    )):
        _spent_images = set(_images[_index * _third : (_index + 1) * _third])
        _spent = [
            {**g, "split": "train" if _row % 2 else "test"}
            for _row, g in enumerate(
                sorted(
                    (g for g in GROUPS if str(g["image_id"]) in _spent_images),
                    key=lambda g: str(g["group_id"]),
                )
            )
        ]
        _dir = SCRATCH / "cr" / _name
        if not (_dir / "units").is_dir():
            build_mock_completed_run(
                _dir, _spent, layer=MOCK_MULTIMODAL_LAYER,
                run_fingerprint=f"sha256:mock-completed-{_index}",
            )
        _mock_runs.append(str(_dir))
    COMPLETED_RUN_DIRS = tuple(_mock_runs)
    COMPLETED_RESOLUTION_RUN_DIR = _mock_runs[2]
    PREP_CACHE_ROOT = str(SCRATCH / "prep_cache_root")

# The pin check runs in BOTH modes. In MOCK it is checked against the simulated
# directories, which is the only way the check is ever executed by a test.
POPULATION_PINS = assert_completed_population_pins(
    COMPLETED_RUN_DIRS,
    resolution_run_dir=COMPLETED_RESOLUTION_RUN_DIR,
)

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
    sample_size_plan_digest=SAMPLE_PLAN.to_dict()["plan_digest"],
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
SAMPLE_PLAN_RECORD = SAMPLE_PLAN.to_dict()

print("PREPARATION CACHE")
print(f"  preparation version  {prep.PREPARATION_VERSION}")
print(f"  preparation digest   {PREPARATION_FINGERPRINT['preparation_digest']}")
print(f"  cache directory      {PREP_DIR}")
print(f"  already complete     {PREP_COMPLETE_ON_ENTRY is not None}")
print()
print("POPULATIONS THIS STUDY MUST BE INDEPENDENT OF")
for _entry in COMPLETED_RUN_IDENTITIES:
    print(f"  {_entry['run']}")
print(f"  pinned mml32res      {POPULATION_PINS['pinned_resolution_run']}")
print(f"  discovered/defaulted {POPULATION_PINS['pin_was_discovered']}/"
      f"{POPULATION_PINS['pin_was_defaulted']}")
print()
print(f"CUDA present: {torch.cuda.is_available()}")
if PREP_COMPLETE_ON_ENTRY is None and torch.cuda.is_available():
    print("!" * 72)
    print("STAGE 0 IS NOT COMPLETE AND THIS RUNTIME HAS A GPU.")
    print("A GPU provides NO benefit here: it is Drive I/O over many small")
    print("files. Recommended: stop this runtime, switch to a free CPU runtime,")
    print("set PREPROCESSING_ONLY = True and let Stage 0 finish and checkpoint.")
    print("Everything already completed is durable and will be reused.")
    print("!" * 72)
'''
)

code(
    '''
# 10b. Harvest the exclusion identities, resumably. THIS is the long cell.
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
          f"identities {_entry['n_identities']:>6}")
for _name, _count in sorted(EXCLUSION.counts().items()):
    print(f"  {_name:18s} {_count:>7,}")
print(f"  exclusion digest   {EXCLUSION.digest}")
print()
print("COMPLETENESS PROOF")
for _row in COMPLETENESS["runs"]:
    print(f"  {_row['run']}: strategy={_row['strategy']} "
          f"groups={_row['group_ids_recovered']}/{_row['expected_group_count']} "
          f"complete={_row['complete']}")
print(f"  fallback scan required: {COMPLETENESS['fallback_required']}")
print(f"  files computed this session: {PREP['files_computed_this_session']}")
print(f"  files reused from Drive:     {PREP['files_reused_from_drive']}")
'''
)

code(
    '''
# 10c. Recover the recordings and captions behind the excluded group ids, then
# filter the pool BEFORE any selection rule runs.
MEDIA_RESOLUTION = resolve_excluded_media(EXCLUSION, GROUPS)
POOL, POOL_RECORD = independent_pool(GROUPS, EXCLUSION)
POOL_DIGEST = payload_checksum(sorted(str(_g["group_id"]) for _g in POOL))
POOL_RECORD["pool_digest"] = POOL_DIGEST

prep.atomic_write_json(
    PREP_DIR / "media_resolution.json", MEDIA_RESOLUTION,
    protected_prefixes=PROTECTED_RUN_PREFIXES,
)
prep.atomic_write_json(
    PREP_DIR / "independent_pool.json", POOL_RECORD,
    protected_prefixes=PROTECTED_RUN_PREFIXES,
)

print(f"excluded group ids           {MEDIA_RESOLUTION['n_excluded_group_ids']:,}")
print(f"resolved in this manifest    {MEDIA_RESOLUTION['n_resolved_in_manifest']:,}")
print(f"audio paths added            {MEDIA_RESOLUTION['audio_paths_added']:,}")
print(f"captions added               {MEDIA_RESOLUTION['captions_added']:,}")
print()
print(f"groups available             {POOL_RECORD['n_groups_available']:,}")
print(f"groups excluded              {POOL_RECORD['n_groups_excluded']:,}")
print(f"independent pool             {POOL_RECORD['n_groups_in_independent_pool']:,}")
print(f"distinct images in pool      {POOL_RECORD['n_distinct_images_in_pool']:,}")
print(f"pool digest                  {POOL_DIGEST}")
if not POOL:
    raise RuntimeError(
        "the independent pool is empty: every synchronized group was already "
        "spent by a completed run. The required independent population cannot "
        "be built and this study refuses rather than reusing media."
    )
'''
)

code(
    '''
# 10d. Rank the pool DESCRIPTIVELY, then check the FROZEN concepts one by one.
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

if RUN_REAL_PRECONVERGENCE_STUDY:
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
    FEASIBILITY = frozen_concept_feasibility(
        RANKING,
        concepts=SELECTED_NAMES,
        focal=FOCAL_CONCEPTS,
        requirements=REQUIREMENTS.to_dict(),
    )
assert_frozen_concepts_feasible(FEASIBILITY)
FEASIBILITY_DIGEST = FEASIBILITY["feasibility_digest"]
UNRELATED_CONTROLS = unrelated_control_assignment(FOCAL_CONCEPTS, NON_FOCAL_CONCEPTS)

print(format_frozen_feasibility(FEASIBILITY))
print()
print("SELECTION — fixed before any model result exists")
print(f"  selected (frozen order)  {SELECTED_NAMES}")
print(f"  focal concepts           {FOCAL_CONCEPTS}")
print(f"  non-focal (controls)     {NON_FOCAL_CONCEPTS}")
print(f"  ranking digest           {RANKING_DIGEST}   (DESCRIPTIVE ONLY)")
print(f"  feasibility digest       {FEASIBILITY_DIGEST}")
print(f"  concepts substituted     False")
'''
)

code(
    '''
# 10e. Build the population, then PROVE it independent and unreplicated.
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

# Recomputed in EVERY process, never loaded: a loaded proof proves only that a
# file says so.
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
print()
print("DISJOINTNESS FROM ALL THREE SPENT POPULATIONS — verified, not assumed")
for _family in DISJOINTNESS["families_checked"]:
    print(f"  {_family:14s} population "
          f"{DISJOINTNESS['population_counts'][_family]:>6}"
          f"   excluded {DISJOINTNESS['exclusion_counts'][_family]:>7}"
          f"   overlap {DISJOINTNESS['n_overlaps'][_family]:>4}")
print(f"  disjoint: {DISJOINTNESS['disjoint']}")
print(f"  population digest {POPULATION_DIGEST}")

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
print(f"split provenance checksum {SPLIT_PROVENANCE_CHECKSUM}")
'''
)

code(
    '''
# 10f. Determinism, persistence, and the end of Stage 0.
import random as _random

if SELECTION_REUSED and PREPARED is not None:
    SELECTION_DETERMINISM = PREPARED["selection_determinism"]
else:
    _shuffled = list(POOL)
    _random.Random(20260810).shuffle(_shuffled)
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
        f"groups in, so it is not reproducible from the identities alone. "
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
        "untouched_confirmation_checksum": CONFIRMATION_MANIFEST["checksum"],
        "adjacent_corpus_manifest_checksum": CORPUS_MANIFEST[
            "corpus_manifest_checksum"
        ],
    },
    protected_prefixes=PROTECTED_RUN_PREFIXES,
)
prep.atomic_write_json(
    PREP_DIR / "untouched_confirmation_manifest.json",
    {"manifest": CONFIRMATION_MANIFEST, "audit": UNTOUCHED_AUDIT,
     "development_role": DEVELOPMENT_ROLE, "corpus": CORPUS_MANIFEST},
    protected_prefixes=PROTECTED_RUN_PREFIXES,
)
prep.atomic_write_text(
    PREP_DIR / "preprocessing_report.md",
    prep.render_preprocessing_report({
        "cache_dir": str(PREP_DIR),
        "preparation_digest": PREPARATION_FINGERPRINT["preparation_digest"],
        "exclusion_digest": EXCLUSION.digest,
        "completeness": COMPLETENESS,
        "files_computed_this_session": PREP["files_computed_this_session"],
        "files_reused_from_drive": PREP["files_reused_from_drive"],
        "batch_files": PREP_BATCH_FILES,
        "checkpoint_seconds": PREP_CHECKPOINT_SECONDS,
    }),
    protected_prefixes=PROTECTED_RUN_PREFIXES,
)

print()
print("STAGE 0 COMPLETE AND PERSISTED")
print(f"  cache directory   {PREP_DIR}")
for _name in sorted(p.name for p in PREP_DIR.iterdir()):
    print(f"    {_name}")
if PREPROCESSING_ONLY:
    print()
    print("=" * 72)
    print("PREPROCESSING_ONLY IS TRUE — STOP HERE.")
    print("Every cell below is a no-op in this session. Switch to an L4, set")
    print("PREPROCESSING_ONLY = False with the same scientific configuration,")
    print("and Stage 0 will load and verify this cache without re-reading a")
    print("single source unit.")
    print("=" * 72)
'''
)

# =============================================== 11. the open prompt and leakage

markdown(
    """
## 11. The open prompt, and the candidate-visibility leakage audit

The model-visible prompt names **no candidate**. The six candidates exist only
in the external scorer. The audit below runs and is persisted **separately for
text, image and spoken_audio**, because the surfaces differ by modality — a
transcript field exists only for audio, a media reference only for image — and
an audit that passed on text says nothing about the other two.

A leaked candidate in any modality is a **hard refusal**, not a warning.
"""
)

code(
    '''
# 11. Build the open prompt in each modality and audit it for leakage.
from jlens.mmpilot.prompt_protocol import (
    CANDIDATE_SCORING_VERSION,
    CANDIDATE_VISIBILITY_RULE,
    DEFAULT_QUESTIONS,
    OPEN_ENTITY_IDENTIFICATION,
    Evidence,
    assert_prompt_leakage_clean,
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

PROMPT_RECORDS = {}
PROMPT_FINGERPRINTS = {}
LEAKAGE_AUDIT = {
    "schema": "jlens.mmpilot.preconvergence_candidate_leakage.v1",
    "protocol": OPEN_PROMPT_PROTOCOL,
    "audited_separately_per_modality": True,
    "per_modality": {},
}
_evidence_by_modality = {
    "text": Evidence(modality="text", text="a photograph of the scene"),
    "image": Evidence(modality="image", media_reference="evidence/image"),
    "spoken_audio": Evidence(
        modality="spoken_audio",
        media_reference="evidence/audio",
        transcript=None,
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
            MODEL_REVISION if RUN_REAL_PRECONVERGENCE_STUDY else "mock"
        ),
        processor_revision=(
            MODEL_REVISION if RUN_REAL_PRECONVERGENCE_STUDY else "mock"
        ),
        audio_protocol_fingerprint=AUDIO_PROTOCOL_FINGERPRINT_EXPECTED,
    )
    # A leak is a refusal. assert_prompt_leakage_clean RAISES on a finding, so
    # this line is the enforcement and the record below is the evidence.
    _clean = assert_prompt_leakage_clean(_built.leakage)
    LEAKAGE_AUDIT["per_modality"][_modality] = {
        "prompt_hash": _built.prompt_hash,
        "candidates_in_prompt": _built.candidate_visibility["candidates_in_prompt"],
        "leakage": _built.leakage,
        "clean": _clean,
        "prompt_protocol_digest": PROMPT_FINGERPRINTS[_modality][
            "prompt_protocol_digest"
        ],
    }
    print(f"  {_modality:13s} candidates_in_prompt="
          f"{_built.candidate_visibility['candidates_in_prompt']}  "
          f"leakage_passed={_built.leakage['passed']}  "
          f"prompt_hash={_built.prompt_hash[:20]}...")
    if _built.candidate_visibility["candidates_in_prompt"]:
        raise RuntimeError(
            f"the {_modality} prompt contains the candidate list; this study "
            "does not run a candidate-listed prompt under an open protocol"
        )

for _concept in SELECTED_NAMES:
    if _concept.lower() in OPEN_QUESTION.lower():
        raise RuntimeError(
            f"the candidate {_concept!r} appears in the model-visible question"
        )

# Order must not move the prompt hash; the SET must move the fingerprint.
_reversed = build_protocol_prompt(
    protocol=OPEN_ENTITY_IDENTIFICATION,
    evidence=_evidence_by_modality["text"],
    external_candidates=list(reversed(SELECTED_NAMES)),
)
LEAKAGE_AUDIT["candidate_order_invariant"] = (
    _reversed.prompt_hash == PROMPT_RECORDS["text"].prompt_hash
)
_narrower = build_protocol_prompt(
    protocol=OPEN_ENTITY_IDENTIFICATION,
    evidence=_evidence_by_modality["text"],
    external_candidates=SELECTED_NAMES[:-1],
)
LEAKAGE_AUDIT["candidate_set_moves_fingerprint"] = (
    prompt_protocol_fingerprint(
        _narrower, model_revision="x", processor_revision="x"
    )["prompt_protocol_digest"]
    != prompt_protocol_fingerprint(
        PROMPT_RECORDS["text"], model_revision="x", processor_revision="x"
    )["prompt_protocol_digest"]
)
LEAKAGE_AUDIT["passed"] = bool(
    all(
        not entry["candidates_in_prompt"] and entry["leakage"]["passed"]
        for entry in LEAKAGE_AUDIT["per_modality"].values()
    )
    and LEAKAGE_AUDIT["candidate_order_invariant"]
    and LEAKAGE_AUDIT["candidate_set_moves_fingerprint"]
)
LEAKAGE_AUDIT["audit_digest"] = payload_checksum(LEAKAGE_AUDIT)
PROMPT_PROTOCOL_DIGEST = PROMPT_FINGERPRINTS["text"]["prompt_protocol_digest"]

print()
print(f"candidate-order invariance: {LEAKAGE_AUDIT['candidate_order_invariant']}")
print(f"candidate-set sensitivity:  "
      f"{LEAKAGE_AUDIT['candidate_set_moves_fingerprint']}")
print(f"leakage audit passed:       {LEAKAGE_AUDIT['passed']}")
print(f"audit digest                {LEAKAGE_AUDIT['audit_digest']}")
if not LEAKAGE_AUDIT["passed"]:
    raise RuntimeError(
        "the candidate-visibility audit did not pass in every modality; "
        "refusing to run an open-protocol study on a leaking prompt"
    )
'''
)

# ========================================================= 12. budgets, gates

markdown(
    """
## 12. Every budget, printed before any switch may be set

Three expensive things, three numbers, three separate confirmation switches:

| stage | cost | switch |
|---|---|---|
| 1 — fitting L27–L31 at scale 250 | forward + backward passes, printed below | `CONFIRM_FITTING_BUDGET` |
| 3 — capability + activations | model passes, printed below | `CONFIRM_STAGE_3_BUDGET` |
| 4 — causal transfer | additional model passes, printed below | `CONFIRM_STAGE_4_BUDGET` |

The native readout itself costs **zero** model passes: it is applied to
residuals already on disk, which is why measuring convergence is cheap and
measuring causal transfer is not.

The gates are re-derived here from the raw switches, and again inside every cell
below that can spend anything.
"""
)

code(
    '''
# 12. Budgets and the confirmation gates. CPU only, nothing is loaded.
from jlens.mmpilot.tri_modal import estimate_stage_passes, format_stage_budget

FITTING_BUDGET = adjacent_budget(
    scale=ADJACENT_SCALE, layers=tuple(ADJACENT_PLAN.layers),
    target_layer=ADJACENT_PLAN.target_layer,
)
from jlens.calibration.adjacent import format_adjacent_budget

print(format_adjacent_budget(FITTING_BUDGET))

_n_groups = PSEUDOREPLICATION["n_units"]
_n_capability = min(
    _n_groups, (N_TRAIN_POSITIVE_IMAGES + N_TEST_POSITIVE_IMAGES) * len(SELECTED_NAMES)
)
_budget_layer = (
    ADJACENT_CANDIDATE_LAYERS[0]
    if RUN_REAL_PRECONVERGENCE_STUDY
    else MOCK_MULTIMODAL_LAYER
)


def _stage(stage):
    """One stage's passes. ``n_candidate_orders=1``: an open prompt names no
    candidates, so candidate-order invariance lives in the external scorer."""
    return estimate_stage_passes(
        n_concepts=len(SELECTED_NAMES),
        n_focal_concepts=len(FOCAL_CONCEPTS),
        modalities=MODALITIES,
        layers=(_budget_layer,),
        causal_layers=(_budget_layer,),
        n_total_groups=_n_groups,
        n_capability_groups=_n_capability,
        n_targets_per_cell=N_TEST_POSITIVE_IMAGES,
        alphas=ALPHAS,
        stage=stage,
        n_candidate_orders=1,
        d_model=(
            EXPECT_D_MODEL if RUN_REAL_PRECONVERGENCE_STUDY else MOCK_D_MODEL
        ),
    )


BUDGET_STAGE_3 = _stage("A")
BUDGET_STAGE_4 = _stage("B")
print()
print(format_stage_budget(BUDGET_STAGE_3))
print()
print(format_stage_budget(BUDGET_STAGE_4))
print()
print("the native direct readout itself costs 0 model passes: it is applied to")
print("residuals already stored by the activation stage.")
'''
)

code(
    '''
# 12b. The gates. Nothing below this cell runs until they are satisfied.
GATES = refresh_gates()

if RUN_REAL_PRECONVERGENCE_STUDY:
    if RUN_LENS_FITTING and not CONFIRM_FITTING_BUDGET:
        raise RuntimeError(
            "RUN_LENS_FITTING is True but CONFIRM_FITTING_BUDGET is False. "
            f"Stage 1 costs {FITTING_BUDGET['n_forward_passes']:,} forward and "
            f"{FITTING_BUDGET['n_backward_passes']:,} backward passes "
            f"({FITTING_BUDGET['fit_hours_low']:.2f}-"
            f"{FITTING_BUDGET['fit_hours_high']:.2f} h on an L4)."
        )
    if RUN_MODEL_STAGE and not CONFIRM_MODEL_LOAD:
        raise RuntimeError(
            "RUN_MODEL_STAGE is True but CONFIRM_MODEL_LOAD is False (~16 GB)."
        )
    if RUN_MODEL_STAGE and CONFIRM_MODEL_LOAD and not CONFIRM_STAGE_3_BUDGET:
        raise RuntimeError(
            "the model may load but CONFIRM_STAGE_3_BUDGET is False. Stage 3 "
            f"costs {BUDGET_STAGE_3.total_passes:,} model passes."
        )
if RUN_STAGE_4_CAUSAL_TRANSFER and not CONFIRM_STAGE_4_BUDGET:
    raise RuntimeError(
        "RUN_STAGE_4_CAUSAL_TRANSFER is True but CONFIRM_STAGE_4_BUDGET is "
        f"False. Stage 4 costs {BUDGET_STAGE_4.total_passes:,} additional "
        "model passes."
    )

print(f"fitting enabled       {FITTING_ENABLED}")
print(f"confirmation enabled  {CONFIRMATION_ENABLED}")
print(f"model stage enabled   {MODEL_STAGE_ENABLED}")
print(f"stage 3 enabled       {STAGE_3_ENABLED}")
print(f"stage 4 requested     {STAGE_4_REQUESTED}")
print()
print("STAGE 4 is additionally gated on the Stage-2 and Stage-3 outcomes.")
print("The switch above only says the passes are affordable; the rule in")
print("section 19 says whether they are informative.")
'''
)

# ============================================================ 13. model load

markdown(
    """
## 13. Load the model **once** — it serves both halves of the study

`build_real_backend` returns a bundle whose `lens_model` is the
`Gemma4LensModel` the calibration code fits with, and whose `backend` is the
multimodal pilot backend Stages 3–4 use. One ~16 GB load, both halves. Loading
twice would be an hour of L4 time spent to obtain the same weights.

Section 5 has already bound every call site on this branch, so anything that
could fail on a signature has failed already, on CPU.

Stage 0 must be **complete and verified** before the download starts: a 16 GB
download in front of an unfinished scan is an L4 hour spent on Drive I/O, and a
model loaded against a half-built population is worse than that.
"""
)

code(
    '''
# 13. Preflight, backend, retrying media loaders.
from jlens.mmpilot.media_io import MEDIA_IO_VERSION, RetryJournal, drive_media_loaders

GATES = refresh_gates()

MODEL = None
BACKEND = None
BUNDLE = None
AUDIO_PROTOCOL = None
MODEL_REVISION_USED = None
PROCESSOR_REVISION_USED = None
TOKENIZER_REVISION_USED = None
AVAILABLE_MODALITIES = []
MEDIA = None
MEDIA_RETRY_JOURNAL = RetryJournal()
LOAD_INFO = None

if MODEL_STAGE_ENABLED and prep.preparation_is_complete(PREP_DIR) is None:
    raise RuntimeError(
        f"Stage 0 is not complete in {PREP_DIR}, so the model is not loaded. "
        "Run Stage 0 to completion first — on a free CPU runtime with "
        "PREPROCESSING_ONLY = True, because a GPU makes Drive I/O over "
        "thousands of small files no faster. Everything Stage 0 has already "
        "finished is durable and will be reused."
    )

if not MODEL_STAGE_ENABLED:
    print("skipped: the model gates are not open.")
    if PREPROCESSING_ONLY:
        print("(PREPROCESSING_ONLY is True, so every model gate is forced shut.)")
    print("Nothing below this cell computes a result.")
elif RUN_REAL_PRECONVERGENCE_STUDY:
    import getpass

    from jlens.mmpilot.real_backend import build_real_backend
    from jlens.mmpilot.tri_modal import assert_audio_protocol

    if not PREDOWNLOAD["passed"]:
        raise RuntimeError(
            "section 5's pre-download checks did not pass; refusing the load"
        )
    if not os.environ.get("HF_TOKEN"):
        _token = getpass.getpass("HF_TOKEN (input hidden): ").strip()
        if not _token:
            raise RuntimeError("a Hugging Face token is required for the gated repo")
        os.environ["HF_TOKEN"] = _token

    BUNDLE = build_real_backend(
        MODEL_REPO_ID,
        revision=MODEL_REVISION,
        token=os.environ["HF_TOKEN"],
        device="cuda" if torch.cuda.is_available() else "cpu",
        allow_model_load=True,
        expect_n_layers=EXPECT_N_LAYERS,
        expect_d_model=EXPECT_D_MODEL,
        expect_vocab_size=EXPECT_VOCAB,
        resolve_audio=True,
    )
    MODEL = BUNDLE.lens_model
    BACKEND = BUNDLE.backend
    LOAD_INFO = BUNDLE.load_info
    MODEL_REVISION_USED = BUNDLE.model_revision
    PROCESSOR_REVISION_USED = BUNDLE.processor_revision
    TOKENIZER_REVISION_USED = str(
        LOAD_INFO.get("tokenizer_revision") or BUNDLE.model_revision
    )
    if BUNDLE.audio_interface is None:
        raise RuntimeError(
            "the native spoken-audio path did not resolve: "
            f"{BUNDLE.audio_blocked_reason}. This study's criterion requires "
            "three modalities and cannot silently degrade to two."
        )
    AUDIO_PROTOCOL = assert_audio_protocol(
        BUNDLE.audio_interface,
        expected_fingerprint=AUDIO_PROTOCOL_FINGERPRINT_EXPECTED,
    )
    MEDIA = drive_media_loaders(journal=MEDIA_RETRY_JOURNAL)
    CALIBRATION_MODEL = MODEL
    print(f"model revision      {MODEL_REVISION_USED}")
    print(f"processor revision  {PROCESSOR_REVISION_USED}")
    print(f"audio protocol      {AUDIO_PROTOCOL['protocol_version']}")
    print(f"  fingerprint       {AUDIO_PROTOCOL['protocol_fingerprint']}")
    print(f"  matches expected  {AUDIO_PROTOCOL['matches_expected_fingerprint']}")
else:
    from jlens.calibration.extension_mock import mock_load_info
    from jlens.mmpilot.mock import MockPilotBackend, load_mock_media

    BACKEND = MockPilotBackend(MOCK_WORLD, supports_audio=True)
    MODEL = CALIBRATION_MODEL
    LOAD_INFO = mock_load_info()
    MODEL_REVISION_USED = LOAD_INFO["model_revision"]
    PROCESSOR_REVISION_USED = "mock"
    TOKENIZER_REVISION_USED = LOAD_INFO["tokenizer_revision"]
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
    print("MOCK: a 42-block CPU calibration stack for the lens, and a separate")
    print("three-modality pilot stub for the behavioural half.")

if BACKEND is not None:
    from jlens.mmpilot.pipeline import available_modalities, PilotConfig as _PilotConfig

    MULTIMODAL_LAYER = (
        None if RUN_REAL_PRECONVERGENCE_STUDY else MOCK_MULTIMODAL_LAYER
    )
    print(f"media io            {MEDIA_IO_VERSION}")
'''
)

# ========================================================= 14. Stage 1 — fit

markdown(
    """
## 14. Stage 1 — fit L27–L31 at scale 250, in one resumable run

All five candidates are fitted **together**, in one accumulator, in one pass
over the prompt list. Five separate fits would be five times the backward passes
for the same information, and would leave five checkpoints that could drift
apart.

Resume is upstream's, hardened: the accumulator is written atomically every
bounded batch and reloaded automatically, and upstream **refuses** a checkpoint
fitted with different `source_layers`, `target_layer` or `skip_first`. The run
fingerprint additionally binds the model revision, the layer set, the corpus
manifest, the prompt protocol, the target layer, `skip_first`, the dtype and the
fitting scale — so a changed configuration refuses the resume instead of mixing
checkpoints.

An interruption loses at most the in-flight bounded batch.
"""
)

code(
    '''
# 14. Open this study's run directory and fit the five candidate layers.
from jlens.calibration.fitting import ScaleNotReachedError, run_calibration
from jlens.calibration.state import CalibrationFingerprint
from jlens.calibration.scale import PLATEAU_RULE
from jlens.mmpilot.preconvergence import assert_fresh_run_namespace

GATES = refresh_gates()

MODE = "real" if RUN_REAL_PRECONVERGENCE_STUDY else "mock"
LENS_FINGERPRINT = None
RUN_DIR = None
LENS_STORE = None
LENS_RESUME_STATE = "not_opened"
FIT_RESULT = None
FIT_RECORD = None

if not FITTING_ENABLED or MODEL is None:
    print("skipped: Stage 1 is not enabled, or no model is loaded.")
else:
    LENS_FINGERPRINT = CalibrationFingerprint(
        mode=MODE,
        protocol_version=ADJACENT_PROTOCOL_VERSION,
        model_repo_id=(
            MODEL_REPO_ID if RUN_REAL_PRECONVERGENCE_STUDY else "mock/gemma-like"
        ),
        model_revision=MODEL_REVISION_USED,
        tokenizer_revision=TOKENIZER_REVISION_USED,
        capture_plan_digest=ADJACENT_PLAN.digest,
        corpus_manifest_checksum=CORPUS_MANIFEST["corpus_manifest_checksum"],
        gate_digest=ADJACENT_CONFIRMATION_GATE.digest,
        plateau_rule_digest=PLATEAU_RULE.digest,
        scale_points=(ADJACENT_SCALE,),
        artifact_format_version=ARTIFACT_FORMAT_VERSION,
        extra={
            "adjacent_protocol_digest": ADJACENT_PROTOCOL.digest,
            "selection_rule_digest": ADJACENT_SELECTION_RULE.digest,
            "candidate_layers": list(ADJACENT_CANDIDATE_LAYERS),
            "source_layer_record": SOURCE_LAYERS["source_layer_checksum"],
            "skip_first": ADJACENT_PLAN.skip_first,
            "target_layer": ADJACENT_PLAN.target_layer,
            "dtype": "bfloat16" if RUN_REAL_PRECONVERGENCE_STUDY else "float32",
            "fitting_scale": int(ADJACENT_SCALE),
            "prompt_protocol": OPEN_PROMPT_PROTOCOL,
            "untouched_confirmation_checksum": CONFIRMATION_MANIFEST["checksum"],
            "development_checksum": DEVELOPMENT_ROLE["checksum"],
            "parent_accumulator_seeded": False,
        },
    )
    RUN_DIR = Path(
        os.environ.get("MMPILOT_RUN_DIR")
        or (
            RESOLVED_RUNS_ROOT
            / f"{PRECONVERGENCE_RUN_PREFIX}_{MODE}_{LENS_FINGERPRINT.digest[7:19]}"
        )
    )
    assert_fresh_run_namespace(RUN_DIR, protected_prefixes=PROTECTED_RUN_PREFIXES)
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    LENS_STORE = AdjacentStore(RUN_DIR / "lens", LENS_FINGERPRINT)
    LENS_RESUME_STATE = LENS_STORE.open()
    LENS_STORE.save(
        "corpus_provenance",
        "manifest",
        {
            "corpus": CORPUS_MANIFEST,
            "reconstruction": RECONSTRUCTION,
            "fit_prefix_verification": FIT_PREFIX,
            "source_layers": SOURCE_LAYERS,
            "parent_audit_passed": PARENT_AUDIT["compatible"],
            "n_dropped_too_short": len(DROPPED_SHORT),
        },
    )
    LENS_STORE.save(
        "untouched_confirmation",
        "manifest",
        {
            "manifest": CONFIRMATION_MANIFEST,
            "audit": UNTOUCHED_AUDIT,
            "development_role": DEVELOPMENT_ROLE,
            "dependency_manifests": DEPENDENCY_MANIFESTS,
        },
    )

    print(f"run directory   {RUN_DIR}")
    print(f"lens fingerprint{LENS_FINGERPRINT.digest}")
    print(f"resume          {LENS_RESUME_STATE}")
    print(f"checkpoint      {LENS_STORE.checkpoint_path}")
    print(f"scale           {ADJACENT_SCALE}  layers {list(ADJACENT_PLAN.layers)}")
    print()

    _checkpoint_every = 25 if RUN_REAL_PRECONVERGENCE_STUDY else 4
    FIT_RESULT = run_calibration(
        MODEL,
        FIT_RECORDS,
        plan=ADJACENT_PLAN,
        scale_points=(ADJACENT_SCALE,),
        store=LENS_STORE,
        checkpoint_every=_checkpoint_every,
        diagnostics_every=_checkpoint_every,
    )
    _snapshot = FIT_RESULT.snapshots[ADJACENT_SCALE]
    FIT_RECORD = {
        "schema": "jlens.calibration.adjacent_fit.v1",
        "layers": list(ADJACENT_PLAN.layers),
        "target_layer": ADJACENT_PLAN.target_layer,
        "hook_site": ADJACENT_HOOK_SITE,
        "scale": int(ADJACENT_SCALE),
        "n_done": FIT_RESULT.n_done,
        "n_skipped": FIT_RESULT.n_skipped,
        "elapsed_seconds": round(FIT_RESULT.elapsed_seconds, 2),
        "checkpoint_path": FIT_RESULT.checkpoint_path,
        "checkpoint_every": _checkpoint_every,
        "snapshot": _snapshot.to_dict(),
        "parent_accumulator_seeded": False,
        "source_layers_are_new": SOURCE_LAYERS["disjoint"],
        "objective": "not_applicable_estimator_is_a_sample_mean",
        "resume_state": LENS_RESUME_STATE,
    }
    LENS_STORE.save("adjacent_fit", "record", FIT_RECORD)

    print(f"prompts fitted  {FIT_RESULT.n_done:,}   skipped {FIT_RESULT.n_skipped}")
    print(f"elapsed         {FIT_RESULT.elapsed_seconds / 3600:.4f} h")
    print(f"snapshot        {_snapshot.path}")
    print(f"  n_prompts     {_snapshot.n_prompts}")
    print(f"  checksum      {_snapshot.checksum}")
    print(f"  layers        {list(_snapshot.layers)}")
    print()
    print("RESUMABILITY: the accumulator is written atomically every "
          f"{_checkpoint_every} prompt(s).")
    print("Stopping the runtime loses at most that bounded batch; a changed")
    print("configuration refuses the resume rather than mixing checkpoints.")
'''
)

# ================================================ 15. Stage 2 — the gate

markdown(
    """
## 15. Stage 2 — development, the untouched confirmation, and the earliest layer

Development first, on the reused 256-record set, for every candidate. Development
publishes nothing and selects nothing: it exists so that a confirmation result
can be read against a prior expectation rather than in isolation.

Then the untouched set is opened **once**, and the same frozen gate is applied to
all five candidates. Every clause, every control, midrank, optimistic and
pessimistic rank, MRR, top-k, tie-at-max, fold stability and prompt coverage is
recorded for **every** candidate — including every failure.

Finally the predeclared rule runs: the **lowest** candidate that passes every
clause. If none pass, the study stops with `ADJACENT_LENS_NO_GO` and the
complete table is still written.
"""
)

code(
    '''
# 15a. The readout scorer. One forward per prompt; every variant reads the same
# activations, so the controls are matched by construction.
from jlens.calibration.gate import (
    eligible_layers,
    evaluate_calibration_layers,
    ordinary_next_token_argmax,
    select_diverse_validation_prompts,
)
from jlens.controls import control_lens, distant_layer_mapping, layer_mapped_lens
from jlens.hooks import ActivationRecorder
from jlens.mmlocalize.lens_validity import tie_aware_row


def score_readout_rows(lens, prompts, layers, target_layer):
    """Tie-aware rows for every (prompt, layer, variant)."""
    import hashlib

    from jlens.calibration.gate import CONTROL_SEED

    variants = {
        "permuted": control_lens(lens, "permuted", seed=CONTROL_SEED),
        "random": control_lens(lens, "random", seed=CONTROL_SEED),
        "wrong_layer": layer_mapped_lens(lens, distant_layer_mapping(layers)),
    }
    rows = []
    record_at = sorted({*layers, target_layer})
    for index, prompt in enumerate(prompts):
        prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()
        ids = MODEL.encode(prompt, max_length=ADJACENT_PLAN.max_seq_len)
        with torch.no_grad():
            with ActivationRecorder(MODEL.layers, at=record_at) as recorder:
                MODEL.forward(ids)
                captured = {i: recorder.activations[i].detach() for i in record_at}
            actual = MODEL.unembed(captured[target_layer][0, -1:].float())[0]
            for layer in layers:
                hidden = captured[layer][0, -1:].float()
                readouts = {
                    "j_lens": MODEL.unembed(lens.transport(hidden, layer))[0],
                    "logit_lens": MODEL.unembed(hidden)[0],
                }
                for name, control in variants.items():
                    readouts[name] = MODEL.unembed(control.transport(hidden, layer))[0]
                for name, logits in readouts.items():
                    rows.append(
                        tie_aware_row(
                            sample_index=index,
                            prompt_sha=prompt_sha,
                            layer=layer,
                            variant=name,
                            variant_logits=logits,
                            actual_logits=actual,
                        )
                    )
    return rows


# MOCK note: a tiny random stack has no interesting per-layer structure, so the
# MOCK path scores FIXTURE rows with commissioned archetypes. The rows are
# synthetic; the SCORER, the GATE, the folds and the SELECTION RULE are real.
MOCK_SCENARIO = "earliest_wins"
print(f"scoring mode: {'real readout' if RUN_REAL_PRECONVERGENCE_STUDY else 'fixtures (' + MOCK_SCENARIO + ')'}")
'''
)

code(
    '''
# 15b. Development on the reused set, for every candidate.
from jlens.calibration.adjacent import (
    ADJACENT_CONFIRMATION_PROMPT_SEED,
    adjacent_target_diversity,
)
from jlens.calibration.extension import DEVELOPMENT_PROMPT_SEED

GATES = refresh_gates()
DEVELOPMENT = None
DEVELOPMENT_PROMPTS = None
DEVELOPMENT_SELECTION = None
DEVELOPMENT_DIVERSITY = None

if LENS_STORE is None or FIT_RESULT is None:
    print("skipped: no fitted lens exists in this session.")
else:
    if RUN_REAL_PRECONVERGENCE_STUDY:
        def _target_token(prompt):
            # The model's ordinary output path only. No lens, no candidate layer.
            return ordinary_next_token_argmax(
                MODEL, prompt, max_length=ADJACENT_PLAN.max_seq_len
            )
    else:
        from jlens.calibration.extension_mock import mock_target_token

        _target_token = mock_target_token

    DEVELOPMENT_PROMPTS, DEVELOPMENT_SELECTION = select_diverse_validation_prompts(
        [record.text for record in DEVELOPMENT_RECORDS],
        n_prompts=N_DEVELOPMENT_PROMPTS,
        gate=ADJACENT_GATE,
        seed=DEVELOPMENT_PROMPT_SEED,
        target_token_for_prompt=_target_token,
    )
    DEVELOPMENT_DIVERSITY = adjacent_target_diversity(
        [row["target_token_id"] for row in DEVELOPMENT_SELECTION["prompts"]]
    )

    _stored = LENS_STORE.load("adjacent_development", f"scale{ADJACENT_SCALE}")
    if _stored is not None:
        DEVELOPMENT = {int(k): v for k, v in _stored["by_layer"].items()}
        print("reused stored development verdicts")
    else:
        if RUN_REAL_PRECONVERGENCE_STUDY:
            _rows = score_readout_rows(
                FIT_RESULT.lens_for_scale(ADJACENT_SCALE),
                DEVELOPMENT_PROMPTS,
                list(ADJACENT_PLAN.layers),
                ADJACENT_PLAN.target_layer,
            )
        else:
            from jlens.calibration.adjacent_mock import mock_adjacent_rows

            _rows = mock_adjacent_rows(
                MOCK_SCENARIO,
                stage="development",
                n_prompts=N_DEVELOPMENT_PROMPTS,
                layers=list(ADJACENT_PLAN.layers),
            )
        DEVELOPMENT = evaluate_calibration_layers(
            _rows,
            layers=list(ADJACENT_PLAN.layers),
            scale=ADJACENT_SCALE,
            stage="validation",
            gate=ADJACENT_GATE,
        )
        LENS_STORE.save(
            "adjacent_development",
            f"scale{ADJACENT_SCALE}",
            {
                "scale": ADJACENT_SCALE,
                "development_set": "REUSED from the early-layer extension",
                "development_checksum": DEVELOPMENT_ROLE["checksum"],
                "selection_checksum": DEVELOPMENT_SELECTION["selection_checksum"],
                "diversity": DEVELOPMENT_DIVERSITY,
                "by_layer": {str(k): v for k, v in DEVELOPMENT.items()},
            },
        )

    print(f"development prompts   {len(DEVELOPMENT_PROMPTS)} (reused set)")
    print(f"  distinct targets    {DEVELOPMENT_DIVERSITY['n_distinct_target_tokens']} "
          f"(floor {DEVELOPMENT_DIVERSITY['min_distinct_target_tokens']})")
    print(f"  passed diversity    {DEVELOPMENT_DIVERSITY['passed']}")
    print(f"  eligible layers     {eligible_layers(DEVELOPMENT)}")
    print()
    print("Development publishes nothing and selects nothing.")
'''
)

code(
    '''
# 15c. Open the untouched confirmation set — once — and apply the frozen gate.
from jlens.calibration.adjacent import (
    adjacent_lens_verdict,
    confirmation_table,
    format_confirmation_table,
    select_earliest_confirmed_layer,
)
from jlens.calibration.publication import ConfirmationVault

GATES = refresh_gates()
CONFIRMATION_RESULTS = None
CONFIRMATION_DIVERSITY = None
SELECTION = None
LENS_VERDICT = None
SELECTED_LAYER = None
VAULT = None

if LENS_STORE is None or DEVELOPMENT is None:
    print("skipped: there is no development result to confirm against.")
elif not CONFIRMATION_ENABLED:
    VAULT = ConfirmationVault(records=CONFIRMATION.records)
    print("RUN_UNTOUCHED_CONFIRMATION is False — the set stays locked.")
    print(f"vault status: {VAULT.status()}")
    print()
    print("'not run' is NOT 'nothing passed'. No candidate has been offered the")
    print("untouched confirmation set.")
else:
    VAULT = ConfirmationVault(records=CONFIRMATION.records)
    # The selection rule is already fixed (section 6) and is recorded against
    # the vault before it opens, so the choice cannot be made after the table.
    VAULT.unlock(
        {
            "selection_rule_digest": ADJACENT_SELECTION_RULE.digest,
            "selection_rule": ADJACENT_SELECTION_RULE.to_dict(),
            "selected_scale": int(ADJACENT_SCALE),
            "confirmation_not_consulted": True,
        }
    )
    _records = VAULT.open()

    if RUN_REAL_PRECONVERGENCE_STUDY:
        def _target_token_c(prompt):
            return ordinary_next_token_argmax(
                MODEL, prompt, max_length=ADJACENT_PLAN.max_seq_len
            )
    else:
        from jlens.calibration.extension_mock import mock_target_token as _target_token_c

    CONFIRMATION_PROMPTS, CONFIRMATION_SELECTION = select_diverse_validation_prompts(
        [record.text for record in _records],
        n_prompts=N_CONFIRMATION_PROMPTS,
        gate=ADJACENT_CONFIRMATION_GATE,
        seed=ADJACENT_CONFIRMATION_PROMPT_SEED,
        target_token_for_prompt=_target_token_c,
    )
    CONFIRMATION_DIVERSITY = adjacent_target_diversity(
        [row["target_token_id"] for row in CONFIRMATION_SELECTION["prompts"]],
        gate=ADJACENT_CONFIRMATION_GATE,
    )

    _stored = LENS_STORE.load("adjacent_confirmation", f"scale{ADJACENT_SCALE}")
    if _stored is not None:
        CONFIRMATION_RESULTS = {int(k): v for k, v in _stored["by_layer"].items()}
        print("reused stored confirmation verdicts")
    else:
        if RUN_REAL_PRECONVERGENCE_STUDY:
            _rows = score_readout_rows(
                FIT_RESULT.lens_for_scale(ADJACENT_SCALE),
                CONFIRMATION_PROMPTS,
                list(ADJACENT_PLAN.layers),
                ADJACENT_PLAN.target_layer,
            )
        else:
            from jlens.calibration.adjacent_mock import mock_adjacent_rows

            _rows = mock_adjacent_rows(
                MOCK_SCENARIO,
                stage="confirmation",
                n_prompts=N_CONFIRMATION_PROMPTS,
                layers=list(ADJACENT_PLAN.layers),
            )
        CONFIRMATION_RESULTS = evaluate_calibration_layers(
            _rows,
            layers=list(ADJACENT_PLAN.layers),
            scale=ADJACENT_SCALE,
            stage="confirmation",
            gate=ADJACENT_CONFIRMATION_GATE,
        )
        LENS_STORE.save(
            "adjacent_confirmation",
            f"scale{ADJACENT_SCALE}",
            {
                "scale": ADJACENT_SCALE,
                "confirmation_manifest": CONFIRMATION_MANIFEST,
                "untouched_audit": UNTOUCHED_AUDIT,
                "selection_checksum": CONFIRMATION_SELECTION["selection_checksum"],
                "diversity": CONFIRMATION_DIVERSITY,
                "by_layer": {str(k): v for k, v in CONFIRMATION_RESULTS.items()},
            },
        )

    SELECTION = select_earliest_confirmed_layer(
        CONFIRMATION_RESULTS,
        candidates=tuple(ADJACENT_PLAN.layers),
        development=DEVELOPMENT,
    )
    LENS_VERDICT = adjacent_lens_verdict(
        SELECTION,
        confirmation_manifest=CONFIRMATION_MANIFEST,
        untouched_audit=UNTOUCHED_AUDIT,
        source_layer_record=SOURCE_LAYERS,
    )
    SELECTED_LAYER = LENS_VERDICT["selected_layer"]
    LENS_STORE.save("layer_selection", "selection", SELECTION)
    LENS_STORE.save("layer_selection", "verdict", LENS_VERDICT)

    print(f"vault status: {VAULT.status()}")
    print(f"confirmation diversity: {CONFIRMATION_DIVERSITY['passed']} "
          f"({CONFIRMATION_DIVERSITY['n_distinct_target_tokens']} distinct)")
    print()
    print("EVERY CANDIDATE, PASS OR FAIL")
    print(format_confirmation_table(SELECTION["table"]))
    print()
    for _row in SELECTION["table"]:
        print(f"  L{_row['layer']}  controls (MRR)  {_row['controls']}")
        print(f"        folds beat all controls  {_row['fold_beats_all_controls']}")
        print(f"        prompt coverage          {_row['prompt_coverage']}")
    print()
    print("=" * 72)
    print(f"ADJACENT_LENS_VALIDITY: {LENS_VERDICT['verdict']}")
    print("=" * 72)
    for _clause in LENS_VERDICT["validity_clauses"]:
        print(f"  [{'PASS' if _clause['passed'] else 'FAIL'}] {_clause['clause']}")
        print(f"         {_clause['detail']}")
    print()
    print(f"  passing layers  {LENS_VERDICT['passing_layers']}")
    print(f"  SELECTED LAYER  {SELECTED_LAYER}")
    print()
    print(LENS_VERDICT["rationale"])
    if SELECTED_LAYER is None:
        print()
        print("STOP. Stages 3 and 4 have no layer to measure and will not run.")
        print("The interval is closed: there is no wider band to try, and no")
        print("candidate is promoted on the strength of being closest.")
'''
)

# ============================ 16. invariance and the run fingerprint

markdown(
    """
## 16. The invariance gate and the run fingerprint

**Invariance.** Capture must be a no-op and a zero-coefficient edit must
reproduce the clean scoring, *separately in each modality*. A gate that passed on
text says nothing about whether an image or an audio forward pass survives the
same hook.

**The fingerprint** binds every scientific configuration field the design depends
on: revisions, versions, the audio protocol, the fitted lens and its confirmation
provenance, the physical layer and hook site, the manifests, **the three excluded
runs and the population pin**, the preparation digest, the prompt protocol and
its leakage audit, the frozen concepts, the selection algorithm and population
digest, the sample-size rule, the convergence criterion and its digest, the
control sets and their seed. Changing any one refuses the resume rather than
mixing units — which is what makes an interrupted Colab session lose at most the
in-flight unit.
"""
)

code(
    '''
# 16a. Fix the multimodal layer and build the pilot configuration.
GATES = refresh_gates()

MULTIMODAL_LAYER = None
CONFIG = None
if SELECTED_LAYER is not None and BACKEND is not None:
    # The MOCK pilot decoder has six blocks; its stand-in for the selected
    # adjacent layer is layer 1, exactly as every completed MOCK run does. The
    # real path uses the physical layer the gate selected.
    MULTIMODAL_LAYER = (
        int(SELECTED_LAYER)
        if RUN_REAL_PRECONVERGENCE_STUDY
        else MOCK_MULTIMODAL_LAYER
    )
    CONFIG = PilotConfig(
        mode=("l27_l31_preconvergence" if RUN_REAL_PRECONVERGENCE_STUDY else "mock"),
        layers=(MULTIMODAL_LAYER,),
        causal_layers=(MULTIMODAL_LAYER,),
        modalities=MODALITIES,
        capability_threshold=CAPABILITY_THRESHOLD,
        alphas=tuple(ALPHAS),
        n_target_examples=N_TEST_POSITIVE_IMAGES,
        pursuit_k=25 if RUN_REAL_PRECONVERGENCE_STUDY else 8,
        pursuit_correlation_chunk_size=(
            65536 if RUN_REAL_PRECONVERGENCE_STUDY else None
        ),
        direction_top_k=16 if RUN_REAL_PRECONVERGENCE_STUDY else 4,
        n_permutations=50 if RUN_REAL_PRECONVERGENCE_STUDY else 8,
        max_capability_groups_per_concept=(
            N_TRAIN_POSITIVE_IMAGES + N_TEST_POSITIVE_IMAGES
        ),
        seed=20260810,
        subset_profile=PROFILE.name,
        image_unique_targets=True,
        min_source_positive_images=N_TRAIN_POSITIVE_IMAGES,
        min_source_negative_images=N_TRAIN_NEGATIVE_IMAGES,
        off_diagonal_causal_only=True,
        concepts=tuple(SELECTED_NAMES),
        causal_concepts=tuple(FOCAL_CONCEPTS),
    )
    AVAILABLE_MODALITIES, BLOCKED_MODALITIES = available_modalities(BACKEND, CONFIG)
    print(f"selected layer        {SELECTED_LAYER}")
    print(f"multimodal layer      {MULTIMODAL_LAYER}")
    print(f"available modalities  {AVAILABLE_MODALITIES}")
    print(f"blocked modalities    {BLOCKED_MODALITIES}")
    if "spoken_audio" not in AVAILABLE_MODALITIES:
        raise RuntimeError(
            "spoken_audio is unavailable; this study's criterion requires it "
            "and does not silently degrade to two modalities."
        )
else:
    print("skipped: no confirmed layer, or no backend. Stages 3-4 do not run.")
'''
)

code(
    '''
# 16b. The per-modality invariance gate at the selected layer.
from jlens.mmpilot.pipeline import build_condition_inputs
from jlens.mmpilot.tri_modal import run_invariance_by_modality

INVARIANCE = None
if CONFIG is None:
    print("skipped: no multimodal configuration")
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

code(
    '''
# 16c. The run fingerprint and the unit store.
from jlens.mmpilot.convergence import (
    CONTROL_VARIANTS,
    CONVERGENCE_CRITERION,
    CONVERGENCE_PROTOCOL,
)
from jlens.mmpilot.jspace import CONVENTIONS
from jlens.mmpilot.pipeline import scientific_fingerprint
from jlens.mmpilot.preconvergence import (
    REQUIRED_CAUSAL_CONTROLS as REQUIRED_CAUSAL,
)
from jlens.mmpilot.preconvergence import preconvergence_fingerprint
from jlens.mmpilot.store import RunFingerprint, UnitStore
from jlens.mmpilot.tri_modal import TRI_MODAL_VERDICT_VERSION

CONTROL_SEED = 20260810
STORE = None
RUN_STATE = "not_opened"
STUDY_FINGERPRINT = None
FINGERPRINT = None
LENS_SNAPSHOT = (FIT_RECORD or {}).get("snapshot") or {}

if CONFIG is None or LENS_VERDICT is None:
    print("skipped: there is no confirmed configuration to bind a fingerprint to.")
    print("A fingerprint recording 'no layer' would be one another run could")
    print("resume from, so none is written at all.")
else:
    STUDY_FINGERPRINT = preconvergence_fingerprint(
        protocol=PRECONVERGENCE_PROTOCOL,
        stage_plan_version=STAGE_PLAN_VERSION,
        stage_four_rule_digest=payload_checksum(STAGE_PLAN),
        intervention_family=INTERVENTION_FAMILY,
        adjacent_protocol_digest=ADJACENT_PROTOCOL.digest,
        candidate_layers=list(ADJACENT_CANDIDATE_LAYERS),
        fitting_scale=int(ADJACENT_SCALE),
        adjacent_gate_digest=ADJACENT_CONFIRMATION_GATE.digest,
        adjacent_selection_rule_digest=ADJACENT_SELECTION_RULE.digest,
        adjacent_run_dir=str(RUN_DIR),
        confirmation_manifest_checksum=CONFIRMATION_MANIFEST["manifest_checksum"],
        untouched_audit_checksum=UNTOUCHED_AUDIT["audit_checksum"],
        model_repo_id=(
            MODEL_REPO_ID if RUN_REAL_PRECONVERGENCE_STUDY else "mock/gemma-like"
        ),
        model_revision=MODEL_REVISION_USED,
        processor_revision=PROCESSOR_REVISION_USED,
        transformers_version=TRANSFORMERS_VERSION,
        torch_version=TORCH_VERSION,
        audio_protocol_version=AUDIO_PROTOCOL["protocol_version"],
        audio_protocol_fingerprint=AUDIO_PROTOCOL["protocol_fingerprint"],
        lens_path=LENS_SNAPSHOT.get("path"),
        lens_checksum=LENS_SNAPSHOT.get("checksum"),
        lens_confirmation_status="passed",
        physical_layer=int(SELECTED_LAYER),
        hook_site=ADJACENT_HOOK_SITE,
        d_model=int(ADJACENT_PLAN.d_model),
        residual_convention=CONVENTIONS["hook_site"],
        final_prompt_token_position=CONVENTIONS["position"],
        dictionary_orientation=CONVENTIONS["dictionary"],
        dictionary_normalization=CONVENTIONS["code_orientation"],
        calibration_modality="text-only",
        original_manifest_checksum=ORIGINAL_MANIFEST_CHECKSUM,
        expanded_manifest_checksum=DERIVED_MANIFEST_CHECKSUM,
        cache_schema_version=CACHE_LOAD["schema_version"],
        evidence_lexicon_hash=CACHE_LOAD["evidence_lexicon_hash"],
        exclusion_run_dirs=sorted(str(d) for d in COMPLETED_RUN_DIRS),
        exclusion_run_checksum=EXCLUSION.digest,
        population_pins_checksum=POPULATION_PINS["pins_checksum"],
        preparation_version=prep.PREPARATION_VERSION,
        preparation_digest=PREPARATION_FINGERPRINT["preparation_digest"],
        exclusion_completeness_digest=payload_checksum(COMPLETENESS),
        independent_pool_digest=POOL_DIGEST,
        concept_ranking_digest=RANKING_DIGEST,
        frozen_concept_feasibility_digest=FEASIBILITY_DIGEST,
        prompt_protocol=OPEN_PROMPT_PROTOCOL,
        prompt_hash=PROMPT_RECORDS["text"].prompt_hash,
        prompt_protocol_digest=PROMPT_PROTOCOL_DIGEST,
        candidate_leakage_audit_digest=LEAKAGE_AUDIT["audit_digest"],
        selected_concepts=list(SELECTED_NAMES),
        focal_concepts=list(FOCAL_CONCEPTS),
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
        required_causal_controls=list(REQUIRED_CAUSAL),
        media_io_version=MEDIA_IO_VERSION,
        jlens_version=COMMIT,
    )
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
            MODEL_REPO_ID if RUN_REAL_PRECONVERGENCE_STUDY else "mock/gemma-like"
        ),
        model_revision=MODEL_REVISION_USED,
        processor_revision=PROCESSOR_REVISION_USED,
        layers=tuple(CONFIG.layers),
        lens_checksum=str(LENS_SNAPSHOT.get("checksum")),
        manifest_checksum=ORIGINAL_MANIFEST_CHECKSUM,
        split_id=SPLIT_SEED,
        intervention_config={
            "alphas": list(CONFIG.alphas),
            "causal_layer": MULTIMODAL_LAYER,
            "off_diagonal_causal_only": True,
            "intervention_family": INTERVENTION_FAMILY,
            "required_controls": list(REQUIRED_CAUSAL),
        },
        selection_config=SELECTION_FINGERPRINT,
        extra={"preconvergence_fingerprint": STUDY_FINGERPRINT},
    )
    STORE = UnitStore(RUN_DIR, FINGERPRINT)
    RUN_STATE = STORE.open()
    print(f"run state              {RUN_STATE}")
    print(f"run fingerprint        {FINGERPRINT.digest}")
    print(f"study fingerprint      {STUDY_FINGERPRINT['fingerprint_digest']}")
    print(f"lens fingerprint       {LENS_FINGERPRINT.digest}")
    print(f"preparation digest     {PREPARATION_FINGERPRINT['preparation_digest']}")
    print("  no unit from another population, layer, protocol, criterion or")
    print("  preparation can be reused here.")
'''
)

# ================================================ 17. Stage 3 — behaviour

markdown(
    """
## 17. Stage 3 — behavioural capability and the residuals

Capability under the **open** prompt, in all three modalities, on the fresh
population. A concept that fails the frozen admissibility rule in any required
modality is measured, printed with the arithmetic that rejected it, excluded from
every verdict — and **never replaced by another concept**.

Then the final-prompt-token residual at the selected layer, under the same
question.
"""
)

code(
    '''
# 17a. Stage 3, part 1 — capability.
from jlens.mmpilot.admissibility import concept_admissibility
from jlens.mmpilot.pipeline import stage_activations, stage_capability
from jlens.mmpilot.tri_modal import audio_capability_verdict

GATES = refresh_gates()
CAPABILITY = None
CAPABILITY_OUTCOME = None
CAPABILITY_VERDICT = None
ADMISSIBILITY = None

if STORE is None or not STAGE_3_ENABLED:
    print("skipped: Stage 3 is not enabled, or no layer was confirmed.")
else:
    CAPABILITY_OUTCOME, CAPABILITY = stage_capability(
        BACKEND, STORE, SUBSET, CONFIG, MEDIA,
        modalities=AVAILABLE_MODALITIES,
        questions=[OPEN_QUESTION],
    )
    print(CAPABILITY_OUTCOME.line("capability"))
    print()
    print("per-concept accuracy under the open prompt (raw counts):")
    for _concept, _per_modality in sorted(CAPABILITY["per_concept"].items()):
        _cells = "  ".join(
            f"{m}={e['n_correct']}/{e['n']}"
            for m, e in sorted(_per_modality.items())
        )
        print(f"  {_concept:12s} {_cells}")

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
    print()
    print(f"capability verdict  {CAPABILITY_VERDICT['verdict']}")
    print(f"admissible focal    {ADMISSIBILITY['eligible_concepts']}")
    print(f"EXCLUDED focal      {ADMISSIBILITY['excluded_concept_names']}")
    for _entry in ADMISSIBILITY["excluded_concepts"]:
        print(f"    {_entry['concept']:12s} {_entry['rejection_reason']}")
    print("  an excluded concept stays in the table above, enters no verdict,")
    print("  and is NEVER replaced by another concept.")
'''
)

code(
    '''
# 17b. Stage 3, part 2 — final-prompt-token residuals at the selected layer.
GATES = refresh_gates()
ACTIVATIONS = []

if STORE is None or not STAGE_3_ENABLED or CAPABILITY is None:
    print("skipped: Stage 3 is not enabled")
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

# ============================== 18. Stage 3 — the native direct readout

markdown(
    """
## 18. The native direct readout at the selected layer, and its controls

The model's **own** output head applied to the stored residual:

```
logits = lm_head(final_norm(h))            # modules called, not reimplemented
logits = softcap * tanh(logits / softcap)  # exactly as the config declares
```

restricted to the six fixed candidates. No lens, no dictionary, no J-space code,
no intervention and no learned probe takes part in this number — which is the
whole point: a *confirmed lens* and an *unconverged native readout* have to be
two independent facts about the same layer.

The head is audited against the model's own `unembed`. A comparison that never
*ran* leaves `matches_model_unembed` as `None`, and reading that with a
truthiness test would silently accept it; `assert_native_head_agrees` refuses it.

The frozen criterion digest is **checked**, not recomputed. Classification is
exactly one of `CONVERGED`, `NOT_CONVERGED`, `AMBIGUOUS`.
"""
)

code(
    '''
# 18a. The native head, the candidates, and the population.
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
TOKENIZATION = None
HEAD = None

if STORE is None or not ACTIVATIONS:
    print("skipped: Stage 3 produced no activations to read out")
else:
    print(format_l32_criterion(layer=MULTIMODAL_LAYER))
    if RUN_REAL_PRECONVERGENCE_STUDY:
        HEAD = head_from_model(MODEL)
        HEAD_AUDIT = audit_native_head(HEAD, model=MODEL, probes=4)
        HEAD_AGREEMENT = assert_native_head_agrees(HEAD_AUDIT, required=True)
        print(f"native head vs model unembed: "
              f"matched={HEAD_AGREEMENT['matches_model_unembed']}  "
              f"max_abs_diff={HEAD_AGREEMENT['max_abs_difference_vs_model_unembed']}")
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
    CLEAN_PREDICTIONS = clean_predictions_from_capability(
        CAPABILITY_OUTCOME.records
    )
    POPULATION = build_population(
        activations=ACTIVATIONS,
        clean_predictions=CLEAN_PREDICTIONS,
        capability=CAPABILITY_VERDICT,
        focal_concepts=FOCAL_CONCEPTS,
        layers=(MULTIMODAL_LAYER,),
    )
    print(f"readout mode {TOKENIZATION['readout_mode']}")
    print(f"population   {POPULATION['n_units']} units, "
          f"{POPULATION['n_with_clean_reference']} with a clean reference")
    print(f"  admissible   {POPULATION['admissible_concepts']}")
    print(f"  inadmissible {POPULATION['inadmissible_concepts']} (excluded, not "
          "replaced)")
'''
)

code(
    '''
# 18b. Score the population, apply the frozen criterion, run the controls.
from jlens.mmpilot.l32_resolution import assert_controls_recorded
from jlens.mmpilot.preconvergence import convergence_verdict_for_layer

CONVERGENCE_STORE = None
CONVERGENCE_STORE_STATE = None
CONVERGENCE_VERDICT = None
LENS_INTEGRITY = None

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
                if RUN_REAL_PRECONVERGENCE_STUDY
                else "mock/gemma-like"
            ),
            model_revision=MODEL_REVISION_USED,
            processor_revision=PROCESSOR_REVISION_USED,
            layers=(MULTIMODAL_LAYER,),
            candidate_digest=TOKENIZATION["digest"],
            readout_mode=TOKENIZATION["readout_mode"],
            head_checksum=str(HEAD_AUDIT.get("head_checksum", "")),
            criterion_digest=CONVERGENCE_CRITERION.digest,
            code_version=PRECONVERGENCE_PROTOCOL,
            extra={
                "preconvergence_fingerprint": STUDY_FINGERPRINT[
                    "fingerprint_digest"
                ],
                "population_digest": POPULATION_DIGEST,
                "exclusion_digest": EXCLUSION.digest,
                "selected_layer": SELECTED_LAYER,
            },
        ),
    )
    CONVERGENCE_STORE_STATE = CONVERGENCE_STORE.open()
    print(f"convergence store: {CONVERGENCE_STORE_STATE}")

    _confirmation_record = dict(CONFIRMATION_RESULTS[int(SELECTED_LAYER)])
    _confirmation_record["layer"] = int(MULTIMODAL_LAYER)
    CONVERGENCE = run_single_layer_convergence(
        population=POPULATION,
        head=HEAD,
        tokenization=TOKENIZATION,
        head_audit=HEAD_AUDIT,
        store=CONVERGENCE_STORE,
        layer=MULTIMODAL_LAYER,
        confirmation_record=_confirmation_record,
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
        CONVERGENCE["summary"], layer=MULTIMODAL_LAYER
    )
    print(format_convergence_cells(CONVERGENCE_CELLS, layer=MULTIMODAL_LAYER))
    print()
    print(format_classification(classification_detail(CONVERGENCE_CLASSIFICATION)))
    print()
    CONTROLS_RECORD = assert_controls_recorded(
        CONVERGENCE_CONTROLS, layer=MULTIMODAL_LAYER
    )
    print(format_controls(
        CONTROLS_RECORD["rows"],
        controls=CONVERGENCE_CONTROLS,
        layer=MULTIMODAL_LAYER,
    ))

    from jlens.mmpilot.preconvergence import adjacent_lens_integrity

    LENS_INTEGRITY = adjacent_lens_integrity(
        layer=int(SELECTED_LAYER),
        scale=int(ADJACENT_SCALE),
        snapshot={**LENS_SNAPSHOT, "hook_site": ADJACENT_HOOK_SITE},
        confirmation_verdict=CONFIRMATION_RESULTS[int(SELECTED_LAYER)],
        invariance=INVARIANCE,
        calibration_modality="text-only",
    )
    CONVERGENCE_VERDICT = convergence_verdict_for_layer(
        layer=int(SELECTED_LAYER),
        integrity=LENS_INTEGRITY,
        convergence=CONVERGENCE,
        controls=CONTROLS_RECORD,
        disjointness=DISJOINTNESS,
        pseudoreplication=PSEUDOREPLICATION,
        sample_plan=SAMPLE_PLAN_RECORD,
        head_agreement=HEAD_AGREEMENT,
        admissibility=ADMISSIBILITY,
        leakage_audit=LEAKAGE_AUDIT,
    )
    print()
    print("=" * 72)
    print(f"NATIVE_OUTPUT_CONVERGENCE: {CONVERGENCE_VERDICT['verdict']}")
    print("=" * 72)
    for _clause in CONVERGENCE_VERDICT["validity_clauses"]:
        print(f"  [{'PASS' if _clause['passed'] else 'FAIL'}] {_clause['clause']}")
        print(f"         {_clause['detail']}")
    print()
    print(CONVERGENCE_VERDICT["rationale"])
'''
)

# ================================================ 19. Stage 4 — causal

markdown(
    """
## 19. Stage 4 — the conditional cross-modal causal transfer

The gate was fixed in section 6, before Stage 3 opened. It requires **all** of:
a candidate passed untouched lens confirmation; that same layer is
`NOT_CONVERGED` in every required modality; every convergence control passed;
behavioural capability is sufficient.

**Why conditional, and why that is not suppression.** Stage 4's only purpose is
the principal claim — causal transfer *before* native direct-readout
convergence. That claim needs the convergence half to be `NOT_CONVERGED`. Under
`CONVERGED` the claim is dead at this layer whatever the causal passes show;
under `AMBIGUOUS` it is unsupported whatever they show. The gate therefore
withholds passes exactly where the headline result is **unfavourable** to the
hypothesis, and those Stage-3 outcomes are the study's primary reported verdicts.

`RUN_STAGE_4_CAUSAL_TRANSFER` remains an **explicit override**. An overridden run
is stamped `gate_overridden: true` and `DESCRIPTIVE_ONLY` in every artifact, and
its numbers never support the principal claim.

The intervention family is the completed open-prompt follow-up's — additive
J-space residual steering — reused unchanged so the two are comparable. It is
**not** an Anthropic two-coordinate swap, and nothing here calls it one.

Required controls: matched random direction, external unrelated concept,
shuffled/permuted control, zero intervention, activation-norm sanity, target
specificity / global disruption, and image-level independent aggregation. A
missing control record is a failure, never a pass.
"""
)

code(
    '''
# 19a. Evaluate the predeclared Stage-4 gate.
from jlens.mmpilot.preconvergence import LAYER_NOT_CONVERGED, stage_four_decision

GATES = refresh_gates()
STAGE_4 = stage_four_decision(
    lens_verdict=str((LENS_VERDICT or {}).get("verdict", "ADJACENT_LENS_NO_GO")),
    convergence_verdict=str(
        (CONVERGENCE_VERDICT or {}).get("verdict", "REFUSED_INVALID")
    ),
    controls_passed=bool((CONTROLS_RECORD or {}).get("passed")),
    capability_sufficient=bool(
        (CAPABILITY_VERDICT or {}).get("verdict") == "AUDIO_CAPABILITY_GO"
    ),
    requested=bool(RUN_STAGE_4_CAUSAL_TRANSFER),
    budget_confirmed=bool(CONFIRM_STAGE_4_BUDGET),
)
print("STAGE 4 DECISION")
for _clause in STAGE_4["gate_clauses"]:
    print(f"  [{'PASS' if _clause['passed'] else 'FAIL'}] {_clause['clause']}")
    print(f"         {_clause['detail']}")
print()
print(f"  gate met          {STAGE_4['gate_met']}")
print(f"  requested         {STAGE_4['requested']}")
print(f"  budget confirmed  {STAGE_4['budget_confirmed']}")
print(f"  RUNS              {STAGE_4['runs']}")
print(f"  gate overridden   {STAGE_4['gate_overridden']}")
print(f"  evidence status   {STAGE_4['evidence_status']}")
print()
print(STAGE_4["statement"])
print()
print(f"RULE: {STAGE_4['rule']}")
print()
print(f"WHY THIS IS AN EFFICIENCY GATE: {STAGE_4['rationale']}")
'''
)

code(
    '''
# 19b. Stage 4, if it runs: source-derived J-space steering at the selected layer.
from jlens.mmpilot.preconvergence import assert_causal_controls_recorded

# Re-derived here as well as in 19a: this is the cell that actually spends the
# passes, and a gate read from an earlier cell is a gate that can be stale.
GATES = refresh_gates()
if STAGE_4["runs"] and not STAGE_4_REQUESTED:
    raise RuntimeError(
        "the Stage-4 decision says it runs, but the raw switches re-derived "
        "just now say it was not requested. Re-run the decision cell; a stale "
        "derived gate must never be what spends a pass."
    )

CAUSAL_VERDICT = None
CAUSAL_IMAGE_LEVEL = None
CAUSAL_INDEPENDENCE = None
CAUSAL_CONTROLS = None
INTERVENTIONS = None
INTERVENTION_RECORDS = []

if not STAGE_4["runs"]:
    print("Stage 4 did not run. Every Stage-2 and Stage-3 result above stands.")
elif STORE is None or not ACTIVATIONS:
    print("Stage 4 cannot run: there are no activations at the selected layer.")
else:
    from jlens.mmpilot.causal import CONTROL_KINDS
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

    if RUN_REAL_PRECONVERGENCE_STUDY:
        CAUSAL_LENS = FIT_RESULT.lens_for_scale(ADJACENT_SCALE)
    else:
        # MOCK ONLY. The pilot decoder is a different stack from the 42-block
        # calibration stack, so a lens fitted at physical layers 27-31 cannot be
        # applied to its layer 1. A synthetic identity lens stands in — the same
        # shallow-for-deep substitution every completed MOCK run makes. It
        # carries no scientific content and exists so the causal machinery is
        # exercised rather than skipped.
        from jlens.lens import JacobianLens

        CAUSAL_LENS = JacobianLens(
            jacobians={MULTIMODAL_LAYER: torch.eye(BACKEND.d_model)},
            n_prompts=ADJACENT_SCALE,
            d_model=BACKEND.d_model,
        )

    _dictionaries = build_dictionaries(
        CAUSAL_LENS,
        (MULTIMODAL_LAYER,),
        BACKEND,
        device=(
            "cuda"
            if (RUN_REAL_PRECONVERGENCE_STUDY and torch.cuda.is_available())
            else "cpu"
        ),
        dtype=(
            torch.float16 if RUN_REAL_PRECONVERGENCE_STUDY else torch.float32
        ),
        build_chunk_rows=32768 if RUN_REAL_PRECONVERGENCE_STUDY else None,
    )
    _code_outcome = stage_codes(
        STORE, ACTIVATIONS, _dictionaries, CONFIG,
        lens_checksum=str(LENS_SNAPSHOT.get("checksum")),
    )
    CODES = _code_outcome.records
    print(_code_outcome.line("jspace"))

    _direction_outcome, DIRECTIONS = stage_directions(
        STORE, CODES, ACTIVATIONS, _dictionaries, CONFIG,
        concepts=SELECTED_NAMES,
        modalities=AVAILABLE_MODALITIES,
        lens_checksum=str(LENS_SNAPSHOT.get("checksum")),
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

    # The seven required controls, each mapped to the record that establishes it.
    _kinds = {str(kind) for kind in CONTROL_KINDS}
    _observed = {
        str(row.get("control_kind"))
        for row in INTERVENTION_RECORDS
        if row.get("control_kind")
    }
    _norm_ok = all(
        row.get("activation_norm_ratio") is None
        or 0.2 <= float(row["activation_norm_ratio"]) <= 5.0
        for row in INTERVENTION_RECORDS
    )
    CAUSAL_CONTROLS = assert_causal_controls_recorded({
        "matched_random_direction": {
            "passed": "random_norm_matched" in _observed,
            "detail": f"control_kind random_norm_matched present={('random_norm_matched' in _observed)}",
        },
        "external_unrelated_concept": {
            "passed": "unrelated_concept" in _observed,
            "detail": f"unrelated controls {dict(sorted(UNRELATED_CONTROLS.items()))}",
        },
        "shuffled_permuted_control": {
            "passed": bool(INTERVENTIONS.get("permutation_test")) or (
                CONFIG.n_permutations > 0
            ),
            "detail": f"n_permutations={CONFIG.n_permutations}",
        },
        "zero_intervention": {
            "passed": 0.0 in tuple(CONFIG.alphas) and "zero" in _observed,
            "detail": f"alphas {list(CONFIG.alphas)}; zero kind present={('zero' in _observed)}",
        },
        "activation_norm_sanity": {
            "passed": _norm_ok,
            "detail": "every recorded activation-norm ratio within [0.2, 5.0]",
        },
        "target_specificity_global_disruption": {
            "passed": bool(
                any(
                    row.get("off_target_shift") is not None
                    for row in INTERVENTION_RECORDS
                )
            )
            or bool(INTERVENTIONS.get("specificity")),
            "detail": "off-target/global-disruption measurement present",
        },
        "image_level_independent_aggregation": {
            "passed": bool(CAUSAL_INDEPENDENCE.get("passed", True))
            and not _divergence["n_rows_pseudoreplicated_at_group_level"],
            "detail": (
                f"{CAUSAL_IMAGE_LEVEL.get('n_images')} photograph(s); "
                f"pseudoreplicated rows "
                f"{_divergence['n_rows_pseudoreplicated_at_group_level']}"
            ),
        },
    })

    CAUSAL_VERDICT = causal_transfer_verdict(
        CAUSAL_IMAGE_LEVEL,
        layer=MULTIMODAL_LAYER,
        focal_concepts=FOCAL_CONCEPTS,
        thresholds=THRESHOLDS,
        name="PRECONVERGENCE_CAUSAL_TRANSFER",
        capability=CAPABILITY_VERDICT,
    )
    STORE.save("metric", "stage_4_causal_verdict", CAUSAL_VERDICT)
    STORE.save("metric", "stage_4_causal_controls", CAUSAL_CONTROLS)
    STORE.save("metric", "stage_4_image_independence", CAUSAL_INDEPENDENCE)

    print()
    print("REQUIRED CAUSAL CONTROLS")
    for _row in CAUSAL_CONTROLS["rows"]:
        print(f"  [{'PASS' if _row['passed'] else 'FAIL'}] {_row['control']}")
        print(f"         {_row['detail']}")
    print(f"  all present and passing: {CAUSAL_CONTROLS['passed']}")
    print()
    print(f"STAGE 4 causal verdict: {CAUSAL_VERDICT['verdict']}")
    print(CAUSAL_VERDICT.get("rationale", ""))
    if STAGE_4["gate_overridden"]:
        print()
        print("NOTE: the predeclared gate was OVERRIDDEN. These causal numbers")
        print("are DESCRIPTIVE_ONLY and do not enter the principal claim.")
'''
)

# ================================================ 20. the five verdicts

markdown(
    """
## 20. The five verdicts and the one terminal outcome

```
ADJACENT_LENS_VALIDITY
EARLIEST_CONFIRMED_LAYER
NATIVE_OUTPUT_CONVERGENCE
THREE_MODALITY_CAUSAL_TRANSFER
PRECONVERGENCE_CAUSAL_TRANSFER
```

The principal success verdict requires **one and the same physical layer** to
have an untouched confirmed J-lens, a `NOT_CONVERGED` native direct readout, and
controlled cross-modal causal transfer — all three **on the same independent
multimodal population**. `assert_same_population` is what makes that mechanical:
convergence measured on one population is never paired with a causal effect
measured on another.
"""
)

code(
    '''
# 20. Assemble the five verdicts and the terminal outcome.
from jlens.mmpilot.preconvergence import (
    assert_same_population,
    preconvergence_verdicts,
)

SAME_POPULATION = assert_same_population(
    convergence_population_digest=(
        POPULATION_DIGEST if CONVERGENCE is not None else None
    ),
    causal_population_digest=(
        POPULATION_DIGEST if CAUSAL_VERDICT is not None else None
    ),
    convergence_layer=(MULTIMODAL_LAYER if CONVERGENCE is not None else None),
    causal_layer=(MULTIMODAL_LAYER if CAUSAL_VERDICT is not None else None),
    require=False,
)

VERDICTS = preconvergence_verdicts(
    lens_verdict=LENS_VERDICT
    or {
        "verdict": "ADJACENT_LENS_NO_GO",
        "selected_layer": None,
        "rationale": "the untouched confirmation set was never opened",
    },
    convergence=CONVERGENCE_VERDICT,
    causal=CAUSAL_VERDICT,
    causal_controls=CAUSAL_CONTROLS,
    stage_four=STAGE_4,
    same_population=SAME_POPULATION,
)

print("=" * 72)
print(f"TERMINAL OUTCOME: {VERDICTS['terminal_outcome']}")
print("=" * 72)
print()
for _name in VERDICTS["verdict_names"]:
    _entry = VERDICTS["verdicts"][_name]
    print(f"  {_name:34s} {_entry['verdict']}")
print()
print(VERDICTS["statement"])
print()
print("THE PRINCIPAL CLAIM REQUIRES:")
for _item in VERDICTS["principal_claim_requires"]:
    print(f"  - {_item}")
print()
print(f"same population   {SAME_POPULATION['same_population']}")
print(f"same layer        {SAME_POPULATION['same_layer']}")
print(f"combinable        {SAME_POPULATION['combinable']}")
print(f"  {SAME_POPULATION['why']}")
print()
print(f"every possible terminal outcome: {list(TERMINAL_OUTCOMES)}")
'''
)

# ================================================ 21. artifacts

markdown(
    """
## 21. Write the artifacts

Into this run's own `mmpre_*` directory and nowhere else.
"""
)

code(
    '''
# 21. Assemble and write every artifact.
from jlens.mmpilot.preconvergence import build_summary, render_report

ARTIFACT_PATHS = {}
SUMMARY = None
RESUME = None
IMMUTABILITY = prep.assert_sources_unchanged(
    prep.verify_sources_unchanged(
        COMPLETED_RUN_DIRS, PREP["inventory"], SOURCE_FAMILIES,
    )
)
from jlens.calibration.parent import assert_parent_unchanged
from jlens.calibration.parent import protected_parent_checksums as _parent_checksums

PARENT_IMMUTABILITY = assert_parent_unchanged(
    PARENT_CHECKSUMS_BEFORE,
    _parent_checksums(PARENT_ROOT, layout=PARENT.layout),
)

if RUN_DIR is None or STUDY_FINGERPRINT is None:
    print("skipped: no run directory and no study fingerprint, so there is no")
    print("scientific result to write. Stage 0's artifacts are already on Drive")
    print(f"at {PREP_DIR}.")
    if RUN_DIR is not None and LENS_STORE is not None:
        # Stage 1/2 ran but no layer was confirmed: the lens table is still the
        # study's result and is written on its own.
        _no_go = {
            "schema": "jlens.calibration.adjacent_lens_no_go.v1",
            "verdict": (LENS_VERDICT or {}).get("verdict", "ADJACENT_LENS_NO_GO"),
            "selection": SELECTION,
            "lens_verdict": LENS_VERDICT,
            "fit": FIT_RECORD,
            "confirmation_manifest": CONFIRMATION_MANIFEST,
            "untouched_audit": UNTOUCHED_AUDIT,
            "verdicts": VERDICTS,
        }
        _path = RUN_DIR / "adjacent_lens_table.json"
        _path.write_text(
            json.dumps(_no_go, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        ARTIFACT_PATHS["adjacent_lens_table.json"] = str(_path)
        print(f"  wrote {_path}")
else:
    POPULATION_MANIFEST = {
        "schema": "jlens.mmpilot.preconvergence_population_manifest.v1",
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
        "population_pins": POPULATION_PINS,
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
        "schema": "jlens.mmpilot.preconvergence_resume.v1",
        "run_state": RUN_STATE,
        "run_dir": str(RUN_DIR),
        "lens_run_state": LENS_RESUME_STATE,
        "lens_store_status": LENS_STORE.status_report() if LENS_STORE else None,
        "preparation_cache_dir": str(PREP_DIR),
        "convergence_store_state": CONVERGENCE_STORE_STATE,
        "invalid_units": list(STORE.invalid_units),
        "invalid_convergence_units": list(
            getattr(CONVERGENCE_STORE, "invalid_units", [])
        ),
        "units_computed": (CONVERGENCE or {}).get("units_computed"),
        "units_reused": (CONVERGENCE or {}).get("units_reused"),
        "fit_n_done": (FIT_RECORD or {}).get("n_done"),
        "fit_checkpoint": (FIT_RECORD or {}).get("checkpoint_path"),
        "fit_checkpoint_every": (FIT_RECORD or {}).get("checkpoint_every"),
        "preprocessing_files_computed": PREP["files_computed_this_session"],
        "preprocessing_files_reused": PREP["files_reused_from_drive"],
        "atomicity": (
            "the fitting accumulator is written atomically every bounded batch; "
            "every behavioural unit is written atomically with a checksum of its "
            "own payload and the run fingerprint's digest; Stage 0 is committed "
            "the same way in bounded shards. An interrupted session loses at "
            "most the in-flight batch or unit, and a changed scientific "
            "configuration refuses the resume rather than mixing them"
        ),
    }

    SUMMARY = build_summary(
        fingerprint=STUDY_FINGERPRINT,
        verdicts=VERDICTS,
        lens_verdict=LENS_VERDICT,
        confirmation_manifest=CONFIRMATION_MANIFEST,
        untouched_audit=UNTOUCHED_AUDIT,
        source_layer_record=SOURCE_LAYERS,
        fit_record=FIT_RECORD or {},
        convergence=CONVERGENCE_VERDICT,
        convergence_controls=CONTROLS_RECORD,
        capability=CAPABILITY_VERDICT or {},
        disjointness=DISJOINTNESS,
        pseudoreplication=PSEUDOREPLICATION,
        pool=POOL_RECORD,
        exclusion=EXCLUSION.to_dict(),
        population_pins=POPULATION_PINS,
        sample_plan=SAMPLE_PLAN_RECORD,
        leakage_audit=LEAKAGE_AUDIT,
        stage_plan_record=STAGE_PLAN,
        stage_four=STAGE_4,
        causal=CAUSAL_VERDICT,
        causal_controls=CAUSAL_CONTROLS,
        immutability={
            "completed_runs": IMMUTABILITY,
            "parent_calibration_run": PARENT_IMMUTABILITY,
        },
        cache=CACHE_LOAD,
        resume=RESUME,
        mode=CONFIG.mode,
        preparation=POPULATION_MANIFEST["preparation"],
        frozen_concept_feasibility=FEASIBILITY,
    )
    REPORT_MARKDOWN = render_report(SUMMARY)

    for _name, _payload in (
        ("l27_l31_preconvergence_summary.json", SUMMARY),
        ("adjacent_lens_table.json", {
            "selection": SELECTION,
            "lens_verdict": LENS_VERDICT,
            "lens_integrity": LENS_INTEGRITY,
            "fit": FIT_RECORD,
            "corpus": CORPUS_MANIFEST,
            "confirmation_manifest": CONFIRMATION_MANIFEST,
            "untouched_audit": UNTOUCHED_AUDIT,
            "development_role": DEVELOPMENT_ROLE,
            "source_layers": SOURCE_LAYERS,
        }),
        ("population_manifest.json", POPULATION_MANIFEST),
        ("disjointness_audit.json", DISJOINTNESS),
        ("candidate_leakage_audit.json", LEAKAGE_AUDIT),
        ("frozen_concept_feasibility.json", FEASIBILITY),
        ("exclusion_completeness_proof.json", COMPLETENESS),
        ("convergence_tables.json", {
            "cells": CONVERGENCE_CELLS,
            "summary": (CONVERGENCE or {}).get("summary"),
            "classification": CONVERGENCE_CLASSIFICATION,
            "verdict": CONVERGENCE_VERDICT,
        }),
        ("convergence_controls.json", {
            "record": CONTROLS_RECORD, "raw": CONVERGENCE_CONTROLS,
        }),
        ("run_state.json", RESUME),
        ("completed_run_immutability.json", {
            "completed_runs": IMMUTABILITY,
            "parent_calibration_run": PARENT_IMMUTABILITY,
        }),
    ):
        _path = RUN_DIR / _name
        _path.write_text(
            json.dumps(_payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        ARTIFACT_PATHS[_name] = str(_path)

    _report_path = RUN_DIR / "l27_l31_preconvergence_report.md"
    _report_path.write_text(REPORT_MARKDOWN, encoding="utf-8")
    ARTIFACT_PATHS["l27_l31_preconvergence_report.md"] = str(_report_path)

    if STAGE_4["runs"]:
        _causal_path = RUN_DIR / "stage_4_causal_report.json"
        _causal_path.write_text(
            json.dumps(
                {
                    "schema": "jlens.mmpilot.preconvergence_stage_four_report.v1",
                    "decision": STAGE_4,
                    "evidence_status": STAGE_4["evidence_status"],
                    "intervention_family": INTERVENTION_FAMILY,
                    "coordinate_swap_scope": COORDINATE_SWAP_SCOPE,
                    "prompt_protocol": OPEN_PROMPT_PROTOCOL,
                    "causal_verdict": CAUSAL_VERDICT,
                    "required_controls": CAUSAL_CONTROLS,
                    "image_level": CAUSAL_IMAGE_LEVEL,
                    "image_independence": CAUSAL_INDEPENDENCE,
                    "separate_from_convergence": True,
                },
                indent=2, ensure_ascii=False, default=str,
            ),
            encoding="utf-8",
        )
        ARTIFACT_PATHS["stage_4_causal_report.json"] = str(_causal_path)

    for _name, _path in sorted(ARTIFACT_PATHS.items()):
        print(f"  {_name:42s} {_path}")
'''
)

# ================================================ 22. read-only proof

markdown(
    """
## 22. Resume state, and the proof that nothing outside this run was written

Two protections, and the stronger one is structural: every write path in
`jlens.mmpilot.prep_cache` passes `assert_write_allowed`, which **refuses** a
path through a protected run prefix outright rather than noticing afterwards
that one changed. `assert_fresh_run_namespace` does the same for this study's own
run directory.

The check that runs here is the re-enumeration half — the identity-bearing
artifacts of all three completed multimodal runs, plus every resolved file of the
parent calibration run, compared by name, size, mtime and sha256 against what was
recorded before this study started.
"""
)

code(
    '''
# 22. Resume state and the read-only proof.
print("STAGE 0")
print(f"  cache directory       {PREP_DIR}")
print(f"  preparation digest    "
      f"{PREPARATION_FINGERPRINT['preparation_digest']}")
print(f"  files computed        {PREP['files_computed_this_session']}")
print(f"  files reused          {PREP['files_reused_from_drive']}")
print(f"  checkpoint unit       {PREP_BATCH_FILES} files or "
      f"{PREP_CHECKPOINT_SECONDS:.0f}s, whichever comes first")
print()
if RESUME is None:
    print("RESUME: no scientific stage completed in this session.")
    if FIT_RECORD is not None:
        print(f"  fitting reached n_done={FIT_RECORD['n_done']} and is durable at")
        print(f"  {FIT_RECORD['checkpoint_path']}")
else:
    print("RESUME")
    print(f"  run state             {RESUME['run_state']}")
    print(f"  lens run state        {RESUME['lens_run_state']}")
    print(f"  convergence store     {RESUME['convergence_store_state']}")
    print(f"  fit n_done            {RESUME['fit_n_done']}")
    print(f"  fit checkpoint every  {RESUME['fit_checkpoint_every']} prompt(s)")
    print(f"  readout computed      {RESUME['units_computed']}")
    print(f"  readout reused        {RESUME['units_reused']}")
    print(f"  invalid units         {len(RESUME['invalid_units'])}")
    print(f"  {RESUME['atomicity']}")
print()
print("COMPLETED RUNS — READ-ONLY")
for _run, _families in sorted(IMMUTABILITY["families_verified"].items()):
    print(f"  unchanged  {_run}  families {_families}")
print(f"  appeared/vanished/modified  {len(IMMUTABILITY['appeared'])}/"
      f"{len(IMMUTABILITY['vanished'])}/{len(IMMUTABILITY['modified'])}")
print()
print("PARENT CALIBRATION RUN — READ-ONLY")
print(f"  immutable             {PARENT_IMMUTABILITY['immutable']} "
      f"({PARENT_IMMUTABILITY['n_files_checked']} files re-checksummed)")
print(f"  accumulator seeded    {SOURCE_LAYERS['parent_accumulator_may_be_seeded']}")
print()
print("=" * 72)
if PREPROCESSING_ONLY:
    print("STAGE 0 ONLY — no model ran and no verdict exists.")
elif not RUN_REAL_PRECONVERGENCE_STUDY:
    print("MOCK RUN — this proves the pipeline, and nothing about Gemma, about")
    print("any layer in 27-31, or about whether pre-convergence causal transfer")
    print("exists.")
elif SUMMARY is not None:
    print(f"TERMINAL OUTCOME: {SUMMARY['primary_verdict']}")
else:
    print(f"TERMINAL OUTCOME: {VERDICTS['terminal_outcome']}")
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
