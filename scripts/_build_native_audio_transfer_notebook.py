# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Generate ``notebooks/multimodal_jspace_spokencoco_native_audio_colab.ipynb``.

Written from source rather than edited as JSON, so the committed notebook stays
output-free and byte-reproducible. Run
``python scripts/_build_native_audio_transfer_notebook.py`` after changing a
cell; a test regenerates it and fails on drift.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TARGET = (
    REPO_ROOT
    / "notebooks"
    / "multimodal_jspace_spokencoco_native_audio_colab.ipynb"
)

CELLS: list[tuple[str, str]] = []


def markdown(text: str) -> None:
    CELLS.append(("markdown", text.strip("\n")))


def code(text: str) -> None:
    CELLS.append(("code", text.strip("\n")))


markdown(
    """
# Three-modality J-space transfer — text, image and **native spoken audio**

**The question.** Using frozen, independently confirmed, *text-calibrated*
J-lenses, do semantic J-space coordinates transfer among **text**, **image**
and **spoken audio** without any learned cross-modal alignment?

The four directions this study exists for are

- `text -> spoken_audio`
- `spoken_audio -> text`
- `image -> spoken_audio`
- `spoken_audio -> image`

`text <-> image` is computed under exactly the same rules and reported beside
them as an **internal replication**. It is already a confirmed result; a
three-modality claim may never rest on it.

**SpokenCOCO's recordings are spoken captions** — a person reading the written
caption aloud. This is linguistic audio. Nothing here is evidence about
environmental sound, and no verdict may be phrased as though it were.

## What has already been established, and what has not

- The native spoken-audio input path is **engineering-ready** (`AUDIO_READY`,
  protocol `jlens.mmpilot.native_spoken_audio.v1`). The audio tower is invoked,
  placeholders match features, distinct waveforms give distinct logits, and the
  capture and zero-intervention invariances hold. **That is not evidence of
  speech understanding and not evidence of J-space transfer.**
- Three lenses passed the calibration run's untouched confirmation set at scale
  100: **layers 35, 38 and 40**. Layer 32 and every earlier tested layer
  **failed**. Layer 32 is never loaded here and is never described as validated.
- Layer 35 is the primary causal layer **only because it is the earliest
  independently confirmed one**. It is not "pre-language" and not
  "pre-convergence"; convergence timing is unresolved and this study does not
  test it.

## Nothing starts by itself

Eight switches, all `False` in the committed notebook, all set by hand. The
committed notebook runs **MOCK only**, and every expensive stage sits behind its
own printed budget and its own separate confirmation flag. Clicking "Run all"
spends nothing.
"""
)

markdown(
    """
## 1. Bootstrap repository

Run these three cells first, in order. They use nothing but the standard
library: the repository is not importable until 1c has installed it, so
anything that says `from jlens...` before then would fail with
`ModuleNotFoundError: No module named 'jlens'`.

Google Drive is **not** needed here — the package is installed and verified
before section 3 mounts anything.
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
#
# Idempotent: clones when absent, otherwise fetches the branch, checks it out,
# and resets to origin. The reset discards local edits inside the Colab
# checkout — that directory is scratch, not somewhere to keep work.
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
# resolves to this checkout. Every later cell may import from the package.
if IN_COLAB:
    print("installing the repository (editable) ...")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "transformers==5.13.1",
            "-e",
            f"{REPO_PATH}[gemma]",
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
        f"`import jlens` resolved to {jlens.__file__}, not this checkout — "
        "another installation is shadowing this checkout"
    )
print(f"jlens  {jlens.__file__}")
print(f"cwd    {os.getcwd()}")
'''
)

markdown(
    """
## 2. Configuration and the staged switches

| switch | what it unlocks |
| --- | --- |
| `RUN_REAL_AUDIO_TRANSFER` | the real Drive dataset and the three real published lenses instead of the deterministic MOCK world |
| `RUN_MODEL_STAGES` | allows Gemma to be loaded at all |
| `CONFIRM_MODEL_LOAD` | acknowledges the ~16 GB download |
| `CONFIRM_REPRESENTATION_BUDGET` | **Stage A**: capability, activations, J-space codes, six representational pairs at layers 35/38/40 |
| `RUN_L35_CAUSAL_STAGE` + `CONFIRM_L35_CAUSAL_BUDGET` | **Stage B**: the primary causal test at layer 35 |
| `RUN_L38_L40_REPLICATION` + `CONFIRM_REPLICATION_BUDGET` | **Stage C**: the same *frozen* design at layers 38 and 40 |

Every stage prints its exact forward-pass count, runtime estimate, unit count
and Drive footprint **before** its confirmation flag can do anything. No stage
runs because "Run all" was clicked.

The design is fixed here and is not adjustable from results: six capability
concepts, three focal causal concepts taken from the top of the pre-model
ranking, one synchronized group per photograph, off-diagonal cells only, and
the alpha sweep `0, 0.25, 0.5, 1.0`.
"""
)

code(
    '''
# 2. Configuration. Requires section 1 (it imports from the repository).
# Nothing here mounts Drive, reads data, or loads a model.
RUN_REAL_AUDIO_TRANSFER = False
RUN_MODEL_STAGES = False
CONFIRM_MODEL_LOAD = False
CONFIRM_REPRESENTATION_BUDGET = False
RUN_L35_CAUSAL_STAGE = False
CONFIRM_L35_CAUSAL_BUDGET = False
RUN_L38_L40_REPLICATION = False
CONFIRM_REPLICATION_BUDGET = False

# ------------------------------------------------------------------ design
N_CONCEPTS = 6
N_FOCAL_CONCEPTS = 3
REAL_LAYERS = (35, 38, 40)
REAL_PRIMARY_CAUSAL_LAYER = 35
REAL_REPLICATION_LAYERS = (38, 40)
ALPHAS = (0.0, 0.25, 0.5, 1.0)
CAPABILITY_THRESHOLD = 0.7
SPLIT_SEED = "spokencoco-native-audio-v1"

# The MOCK world is smaller on purpose: it exercises every branch, and its cell
# sizes are plumbing parameters rather than scientific ones.
N_TRAIN_POSITIVE_IMAGES = 8 if RUN_REAL_AUDIO_TRANSFER else 2
N_TEST_POSITIVE_IMAGES = 8 if RUN_REAL_AUDIO_TRANSFER else 2
N_TRAIN_NEGATIVE_IMAGES = 8 if RUN_REAL_AUDIO_TRANSFER else 2
N_TEST_NEGATIVE_IMAGES = 8 if RUN_REAL_AUDIO_TRANSFER else 2

# --------------------------------------------------------- the three lenses
# Published by the completed research-grade calibration run and confirmed on
# its untouched confirmation set. Frozen. THIS NOTEBOOK DOES NOT FIT A LENS.
PUBLISHED_LENS_DIR = (
    "/content/drive/MyDrive/jacobian-lens-gemma/runs/"
    "rgcalib_real_7e3736b4de8f/artifacts/published"
)
PUBLISHED_LENS_PINS = {
    35: (
        "lens.layer35.scale100.validated.pt",
        "sha256:64fb02d718ac48adc1bced99e2eff3c2215052ba144d5dedac05f17936a96ed1",
    ),
    38: (
        "lens.layer38.scale100.validated.pt",
        "sha256:c8508fbf2b916e5d9aaeb8711a30f76414ee16478c5f6cc321e57e2fe846d1c0",
    ),
    40: (
        "lens.layer40.scale100.validated.pt",
        "sha256:8a90f67eeb9bb5db14e6715b8bc516a899da1c3210d0662ec7fa177b5409f7d7",
    ),
}
LENS_FITTED_SCALE = 100

MODEL_REPO_ID = "google/gemma-4-E4B-it"
MODEL_REVISION = "fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd"
TRANSFORMERS_VERSION_EXPECTED = "5.13.1"
EXPECT_N_LAYERS, EXPECT_D_MODEL, EXPECT_VOCAB = 42, 2560, 262144

# ------------------------------------------------- the spoken-audio protocol
# From the completed native-audio engineering audit. AUDIO_READY is engineering
# evidence only; it says nothing about semantics or transfer.
AUDIO_PROTOCOL_VERSION_EXPECTED = "jlens.mmpilot.native_spoken_audio.v1"
AUDIO_PROTOCOL_FINGERPRINT_EXPECTED = (
    "sha256:9ad8bcc9420a7983f6e3b75d5d7080c0e2fcf0a94a76431917fcde73ba777920"
)

# --------------------------------------------------------------- the data
SPOKENCOCO_BASE_ROOT = "/content/drive/MyDrive/datasets/cstf_spokencoco"
IMAGE_MEDIA_ROOT = "/content/drive/MyDrive/datasets/cstf_spokencoco/coco"
AUDIO_MEDIA_ROOT = "/content/drive/MyDrive/datasets/cstf_spokencoco/SpokenCOCO"
DOWNLOAD_CACHE = "/content/drive/MyDrive/datasets/cstf_spokencoco_download_cache"
MANIFEST_PATH = "/content/drive/MyDrive/datasets/spokencoco_manifest.json"
RUNS_ROOT = "/content/drive/MyDrive/jacobian-lens-gemma/runs"

# The completed pilot's derived evidence is reused READ-ONLY: its expanded
# manifest is the expensive metadata join this study would otherwise redo.
COMPLETED_PILOT_RUN_DIR = (
    "/content/drive/MyDrive/jacobian-lens-gemma/runs/mmpilot_pilot_20260803T160711"
)

# Run directories that must never be written into. This study creates its own
# namespace; a completed run is evidence, not scratch.
PROTECTED_RUN_PREFIXES = (
    "mmpilot_pilot_",
    "mmrobust_",
    "mmlocalize_",
    "rgcalib_",
    "audioaudit_",
    "text_jlens_",
)

import json

from jlens.mmpilot.pipeline import PilotConfig
from jlens.mmpilot.selection import IMAGE_UNIQUE_MOCK_PROFILE, IMAGE_UNIQUE_PROFILE
from jlens.mmpilot.tri_modal import (
    ALL_PAIRS,
    AUDIO_PAIRS,
    TRI_MODAL_VERDICT_VERSION,
    TriModalThresholds,
)

SCRATCH = Path(os.environ.get("MMPILOT_SCRATCH") or "/content/mmaudio_scratch")
SCRATCH.mkdir(parents=True, exist_ok=True)
RESOLVED_RUNS_ROOT = Path(
    os.environ.get("MMPILOT_RUNS_ROOT")
    or (RUNS_ROOT if RUN_REAL_AUDIO_TRANSFER else SCRATCH / "runs")
)
COMPLETED_PILOT_RUN_DIR = (
    os.environ.get("MMPILOT_PILOT_RUN_DIR") or COMPLETED_PILOT_RUN_DIR
)

MODALITIES = ("text", "image", "spoken_audio")
# The scientific profile is `image_unique` (eight distinct images per split).
# MOCK uses the same policy at two images per split so a CPU-sized synthetic
# world still exercises every branch; the profile name is bound into the run
# fingerprint, so the two can never be mixed.
PROFILE = IMAGE_UNIQUE_PROFILE if RUN_REAL_AUDIO_TRANSFER else IMAGE_UNIQUE_MOCK_PROFILE
if RUN_REAL_AUDIO_TRANSFER and PROFILE.name != "image_unique":
    raise RuntimeError(
        f"the real path must run under the image_unique profile, not "
        f"{PROFILE.name!r}"
    )

# The MOCK world's decoder has six blocks, so its "validated layers" are 2/3/4.
LAYERS = REAL_LAYERS if RUN_REAL_AUDIO_TRANSFER else (2, 3, 4)
PRIMARY_CAUSAL_LAYER = (
    REAL_PRIMARY_CAUSAL_LAYER if RUN_REAL_AUDIO_TRANSFER else 2
)
REPLICATION_LAYERS = REAL_REPLICATION_LAYERS if RUN_REAL_AUDIO_TRANSFER else (3, 4)

THRESHOLDS = TriModalThresholds(
    capability_threshold=CAPABILITY_THRESHOLD,
    required_positive_images_per_cell=N_TEST_POSITIVE_IMAGES,
    required_negative_images_per_cell=N_TEST_NEGATIVE_IMAGES,
)

CONFIG = PilotConfig(
    mode="native_audio_transfer" if RUN_REAL_AUDIO_TRANSFER else "mock",
    layers=tuple(LAYERS),
    causal_layers=(PRIMARY_CAUSAL_LAYER,),
    modalities=MODALITIES,
    capability_threshold=CAPABILITY_THRESHOLD,
    alphas=tuple(ALPHAS),
    n_target_examples=N_TEST_POSITIVE_IMAGES,
    pursuit_k=25 if RUN_REAL_AUDIO_TRANSFER else 8,
    pursuit_correlation_chunk_size=65536 if RUN_REAL_AUDIO_TRANSFER else None,
    direction_top_k=16 if RUN_REAL_AUDIO_TRANSFER else 4,
    n_permutations=50 if RUN_REAL_AUDIO_TRANSFER else 8,
    max_capability_groups_per_concept=8,
    seed=20260805,
    # The photograph is the independent unit before anything is spent.
    subset_profile=PROFILE.name,
    image_unique_targets=True,
    min_source_positive_images=N_TRAIN_POSITIVE_IMAGES,
    min_source_negative_images=N_TRAIN_NEGATIVE_IMAGES,
    off_diagonal_causal_only=True,
)

# The stage plan. Bound into the run fingerprint — the switch *positions* are
# not, because binding them would refuse to resume Stage A's units the moment
# Stage B was turned on, which is the opposite of what resume is for. The
# observed switch positions are written into run_manifest.json instead.
STAGE_PLAN = {
    "protocol": "mmpilot.tri_modal_stage_plan.v1",
    "A": {"layers": list(LAYERS), "work": "capability+activation+jspace+representation"},
    "B": {"layers": [PRIMARY_CAUSAL_LAYER], "work": "causal", "off_diagonal_only": True},
    "C": {"layers": list(REPLICATION_LAYERS), "work": "causal", "frozen_copy_of": "B"},
}

print(f"RUN_REAL_AUDIO_TRANSFER       = {RUN_REAL_AUDIO_TRANSFER}")
print(f"RUN_MODEL_STAGES              = {RUN_MODEL_STAGES}")
print(f"CONFIRM_MODEL_LOAD            = {CONFIRM_MODEL_LOAD}")
print(f"CONFIRM_REPRESENTATION_BUDGET = {CONFIRM_REPRESENTATION_BUDGET}")
print(f"RUN_L35_CAUSAL_STAGE          = {RUN_L35_CAUSAL_STAGE}")
print(f"CONFIRM_L35_CAUSAL_BUDGET     = {CONFIRM_L35_CAUSAL_BUDGET}")
print(f"RUN_L38_L40_REPLICATION       = {RUN_L38_L40_REPLICATION}")
print(f"CONFIRM_REPLICATION_BUDGET    = {CONFIRM_REPLICATION_BUDGET}")
print()
print(f"mode        {CONFIG.mode}")
print(f"modalities  {list(MODALITIES)}")
print(f"layers      {list(LAYERS)}  primary causal L{PRIMARY_CAUSAL_LAYER}  "
      f"replication {list(REPLICATION_LAYERS)}")
print(f"alphas      {list(CONFIG.alphas)}")
print(f"pairs       {len(ALL_PAIRS)} directional ({len(AUDIO_PAIRS)} audio-related)")
print(f"profile     {PROFILE.version}  (max {PROFILE.max_groups_per_image} group per image)")
print(f"verdict     {TRI_MODAL_VERDICT_VERSION}")
print()
print("This notebook does not fit a lens and fits no cross-modal alignment.")
'''
)

markdown(
    """
## 3. Mount Google Drive

Read-only checks. This cell never creates, moves, or deletes anything inside
the dataset, and never touches a completed run directory except to read from it.
"""
)

code(
    '''
# 3. Mount Drive and verify the configured paths exist. Read-only.
PUBLISHED_LENS_PATHS = {
    layer: str(Path(PUBLISHED_LENS_DIR) / name)
    for layer, (name, _checksum) in sorted(PUBLISHED_LENS_PINS.items())
}

if RUN_REAL_AUDIO_TRANSFER and IN_COLAB:
    from google.colab import drive

    drive.mount("/content/drive", force_remount=False)

if RUN_REAL_AUDIO_TRANSFER:
    _required = [
        SPOKENCOCO_BASE_ROOT,
        IMAGE_MEDIA_ROOT,
        AUDIO_MEDIA_ROOT,
        MANIFEST_PATH,
        COMPLETED_PILOT_RUN_DIR,
        *PUBLISHED_LENS_PATHS.values(),
    ]
    missing = [path for path in _required if not Path(path).exists()]
    if missing:
        raise RuntimeError(
            f"configured path(s) do not exist: {missing}. Nothing is discovered "
            "automatically; fix the configuration in section 2."
        )
    print("all configured paths exist, including the three published lenses:")
    for _layer, _path in sorted(PUBLISHED_LENS_PATHS.items()):
        print(f"  L{_layer}  {_path}")
else:
    print("skipped: RUN_REAL_AUDIO_TRANSFER is False (the MOCK world needs no Drive)")
'''
)

markdown(
    """
## 4. Install and verify dependencies

Media codecs only. The repository itself was installed in 1c.
"""
)

code(
    '''
# 4. Media dependencies and the runtime report. Never touches the dataset.
if IN_COLAB:
    command = [sys.executable, "-m", "pip", "install", "-q",
               "pillow", "soundfile", "librosa"]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{' '.join(command)} failed:\\n{result.stderr[-2000:]}")

import torch

print(f"torch {torch.__version__}")
try:
    import transformers

    TRANSFORMERS_VERSION = str(transformers.__version__)
except ModuleNotFoundError:
    TRANSFORMERS_VERSION = "not-installed"
print(f"transformers {TRANSFORMERS_VERSION}")
if RUN_REAL_AUDIO_TRANSFER and TRANSFORMERS_VERSION != TRANSFORMERS_VERSION_EXPECTED:
    raise RuntimeError(
        f"this run is bound to transformers=={TRANSFORMERS_VERSION_EXPECTED}, "
        f"but the runtime imported {TRANSFORMERS_VERSION}. Restart the session "
        "and rerun the pinned bootstrap before loading Gemma"
    )
print(f"cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    _properties = torch.cuda.get_device_properties(0)
    print(f"gpu   {_properties.name}  {_properties.total_memory / 1e9:.1f} GB")
    print("     one L4 is the design target; nothing here needs more")
else:
    print("no GPU — sections 1-7 are CPU-only and will still run")
'''
)

markdown(
    """
## 5. Reuse the derived cache (CPU only)

The completed pilot already paid for the metadata join, the media validation and
the synchronized-evidence audit. That work depends on the dataset and the
evidence rule, not on how many modalities a study uses, so it is **reused**:
this section reads the pilot's expanded manifest and verifies its recorded
checksum of the original manifest before taking a single group from it.

If that checksum disagrees, the join is redone rather than trusted. A derived
artifact whose source has moved underneath it is not evidence about the current
dataset.

The run directory is created in a **new namespace** and is refused if it would
land inside a completed pilot, robustness, localization, calibration or
audio-audit run.
"""
)

code(
    '''
# 5. CPU ONLY. Reuse the derived evidence; create a fresh run namespace.
from datetime import datetime, timezone

from jlens.mmpilot import evidence as evidence_module
from jlens.mmpilot import expansion as expansion_module
from jlens.mmpilot import manifest as manifest_module
from jlens.mmpilot.concepts import discover_category_universe

if not RUN_REAL_AUDIO_TRANSFER:
    # The synthetic world. Six concepts, two captions per image, so the
    # one-group-per-image rule has real sibling groups to exclude, and the same
    # concept direction enters through text, "image" bytes and "audio" bytes.
    from jlens.mmpilot.mock import MockWorld, build_mock_dataset

    MOCK_CONCEPTS_6 = {
        "bus": ("bus", "buses"),
        "cat": ("cat", "cats"),
        "clock": ("clock", "clocks"),
        "dog": ("dog", "dogs"),
        "pizza": ("pizza", "pizzas"),
        "zebra": ("zebra", "zebras"),
    }
    MOCK_WORLD = MockWorld(MOCK_CONCEPTS_6)
    if not (SCRATCH / "data" / "spokencoco_manifest.json").is_file():
        build_mock_dataset(
            SCRATCH / "data",
            world=MOCK_WORLD,
            images_per_concept=N_TRAIN_POSITIVE_IMAGES + N_TEST_POSITIVE_IMAGES,
            negative_images=(N_TRAIN_NEGATIVE_IMAGES + N_TEST_NEGATIVE_IMAGES) * 2,
            captions_per_image=2,
            layout="sibling",
            visual_only_images=1,
        )
    MANIFEST_PATH = str(SCRATCH / "data" / "spokencoco_manifest.json")
    IMAGE_MEDIA_ROOT = str(SCRATCH / "data" / "coco")
    AUDIO_MEDIA_ROOT = str(SCRATCH / "data" / "SpokenCOCO")
    print(f"MOCK dataset ready at {SCRATCH / 'data'}")

MANIFEST_CHECKSUM = manifest_module.manifest_checksum(MANIFEST_PATH)
IMAGE_ROOTS = [Path(IMAGE_MEDIA_ROOT)]
AUDIO_ROOTS = [Path(AUDIO_MEDIA_ROOT)]

RUN_ID = (
    f"mmaudio_{CONFIG.mode}_"
    f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
)
RUN_DIR = Path(os.environ.get("MMPILOT_RUN_DIR") or (RESOLVED_RUNS_ROOT / RUN_ID))
_offending = [
    prefix for prefix in PROTECTED_RUN_PREFIXES if RUN_DIR.name.startswith(prefix)
] + [
    prefix
    for prefix in PROTECTED_RUN_PREFIXES
    for part in RUN_DIR.parts[:-1]
    if part.startswith(prefix)
]
if _offending:
    raise RuntimeError(
        f"{RUN_DIR} is inside a completed run namespace ({sorted(set(_offending))}). "
        "Completed runs are evidence and are never written into; this study "
        "creates its own mmaudio_* directory."
    )
RUN_DIR.mkdir(parents=True, exist_ok=True)
print(f"run directory {RUN_DIR}")

SEARCH_ROOTS = sorted({str(root) for root in IMAGE_ROOTS + AUDIO_ROOTS if root.is_dir()})
if RUN_REAL_AUDIO_TRANSFER:
    SEARCH_ROOTS = sorted(
        {
            str(candidate)
            for candidate in (
                SPOKENCOCO_BASE_ROOT, IMAGE_MEDIA_ROOT, AUDIO_MEDIA_ROOT, DOWNLOAD_CACHE
            )
            if Path(candidate).is_dir()
        }
    )
DISCOVERED = expansion_module.discover_metadata_sources(
    SEARCH_ROOTS, exclude=[MANIFEST_PATH], max_files=40, max_depth=3
)
ANNOTATION_SOURCES = [s for s in DISCOVERED if s.source_kind == "coco_object_annotation"]
SYNC_SOURCES = [s for s in DISCOVERED if s.usable]
UNIVERSE = discover_category_universe(ANNOTATION_SOURCES)
EVIDENCE_CONFIG = evidence_module.config_from_specs(UNIVERSE.specs)
CONCEPT_CANDIDATES = UNIVERSE.lexicon()
print(
    f"\\n{len(UNIVERSE.categories)} categories discovered, "
    f"{len(UNIVERSE.eligible)} eligible as concepts"
)
print(f"lexicon hash {EVIDENCE_CONFIG.lexicon_hash}")

REUSED_EXPANDED_MANIFEST = None
PILOT_EXPANDED = Path(COMPLETED_PILOT_RUN_DIR) / "expanded_manifest.json"
if PILOT_EXPANDED.is_file():
    _stored = json.loads(PILOT_EXPANDED.read_text(encoding="utf-8"))
    if _stored.get("original_manifest_checksum") == MANIFEST_CHECKSUM and _stored.get(
        "groups"
    ):
        REUSED_EXPANDED_MANIFEST = _stored
        print(f"\\nreusing the completed pilot's expanded manifest: {PILOT_EXPANDED}")
        print(f"  original manifest checksum verified: {MANIFEST_CHECKSUM}")
    else:
        print(
            f"\\n{PILOT_EXPANDED} does not match the current original manifest "
            "checksum — re-deriving rather than trusting it."
        )

if REUSED_EXPANDED_MANIFEST is not None:
    PILOT_GROUPS = REUSED_EXPANDED_MANIFEST["groups"]
    DERIVED_CACHE_FINGERPRINT = REUSED_EXPANDED_MANIFEST.get(
        "expanded_manifest_checksum"
    ) or MANIFEST_CHECKSUM
    EXPANSION_STATUS = "reused: derived join loaded from the completed pilot"
else:
    BASELINE = manifest_module.normalize_manifest(
        json.loads(Path(MANIFEST_PATH).read_text(encoding="utf-8")),
        manifest_module.inspect_manifest(
            json.loads(Path(MANIFEST_PATH).read_text(encoding="utf-8"))
        ),
        image_roots=IMAGE_ROOTS,
        audio_roots=AUDIO_ROOTS,
        source_checksum=MANIFEST_CHECKSUM,
        min_complete_groups=1,
    )
    EXPANSION = expansion_module.build_expanded_manifest(
        SYNC_SOURCES,
        image_roots=IMAGE_ROOTS,
        annotation_sources=ANNOTATION_SOURCES,
        candidate_concepts=CONCEPT_CANDIDATES,
        max_metadata_records=20000,
        audio_roots=AUDIO_ROOTS,
        baseline_groups=BASELINE.groups,
    )
    PILOT_GROUPS = EXPANSION.groups
    _payload, EXPANSION_STATUS = expansion_module.persist_expanded_manifest(
        RUN_DIR / "expanded_manifest.json",
        EXPANSION,
        original_checksum=MANIFEST_CHECKSUM,
        conversion={
            "converter": "jlens.mmpilot.expansion.build_expanded_manifest",
            "search_roots": SEARCH_ROOTS,
            "evidence_rule": "visual_annotation_AND_caption_lexicon",
            "evidence_lexicon_hash": EVIDENCE_CONFIG.lexicon_hash,
            "reads_only": True, "media_redownloaded": False, "audio_transcribed": False,
        },
    )
    DERIVED_CACHE_FINGERPRINT = _payload.get("expanded_manifest_checksum", MANIFEST_CHECKSUM)
print(EXPANSION_STATUS)
print(f"synchronized groups available: {len(PILOT_GROUPS)}")
print(f"distinct images available:     {len({g['image_id'] for g in PILOT_GROUPS})}")
print(f"derived cache fingerprint:     {DERIVED_CACHE_FINGERPRINT}")
'''
)

markdown(
    """
## 6. Select six concepts and build the unique-image subset (CPU only)

**Feasibility is re-screened; ranking order is preserved.** Every candidate is
re-scored against this study's cell sizes, in the same deterministic order the
earlier studies used — the order is what picks the focal concepts, so
re-sorting it would substitute a different, arbitrary choice.

**The three focal concepts are the first three in that order, and they are
fixed here — before Gemma is loaded.** The remaining three supply the external
unrelated controls, assigned by rotation. Nothing about this rule reads a
capability result, an activation, or a target-test example.

**If fewer than six concepts are feasible under all three modalities, this is
reported and the design is not quietly narrowed.** A concept is never replaced
after its results are seen.

One synchronized group per image, chosen by a seeded stable rank over the group
id. Because each group carries exactly one recording, one group per image also
means one recording per image — which is what keeps the audio arm's targets
recording-disjoint as well as image-disjoint.
"""
)

code(
    '''
# 6. CPU ONLY. Re-screen feasibility, take the top six in ranking order, fix
# the focal concepts and their controls, then build the one-group-per-image
# subset.
import time

from jlens.mmpilot.selection import (
    select_focal_concepts,
    unrelated_control_assignment,
)

_selection_t0 = time.perf_counter()
print(f"indexing {len(PILOT_GROUPS):,} synchronized groups once ...")
EVIDENCE_INDEX = evidence_module.build_evidence_index(
    PILOT_GROUPS, tuple(CONCEPT_CANDIDATES), EVIDENCE_CONFIG
)

REQUIREMENTS = expansion_module.ConceptRequirements(
    min_distinct_images=N_TRAIN_POSITIVE_IMAGES + N_TEST_POSITIVE_IMAGES,
    min_groups=N_TRAIN_POSITIVE_IMAGES + N_TEST_POSITIVE_IMAGES,
    min_train_positives=N_TRAIN_POSITIVE_IMAGES,
    min_test_positives=N_TEST_POSITIVE_IMAGES,
)
RANKING = expansion_module.rank_concepts(
    PILOT_GROUPS,
    CONCEPT_CANDIDATES,
    requirements=REQUIREMENTS,
    groups_per_concept=N_TRAIN_POSITIVE_IMAGES + N_TEST_POSITIVE_IMAGES,
    max_groups_per_image=PROFILE.max_groups_per_image,
    seed=SPLIT_SEED,
    evidence_config=EVIDENCE_CONFIG,
    profile=PROFILE,
    evidence_index=EVIDENCE_INDEX,
)
print("=" * 72)
print("RANKED COVERAGE AT THIS STUDY'S CELL SIZES (complete, with rejections)")
print("=" * 72)
print(expansion_module.format_ranking_table(RANKING, limit=20))

RANKED_CONCEPTS = [row["concept"] for row in RANKING]
SELECTED_NAMES = expansion_module.select_concepts(
    RANKING,
    n_concepts=N_CONCEPTS,
    max_concepts=N_CONCEPTS,
    requirements=REQUIREMENTS,
)
FOCAL_CONCEPTS, NON_FOCAL_CONCEPTS = select_focal_concepts(
    SELECTED_NAMES, n_focal=N_FOCAL_CONCEPTS
)
UNRELATED_CONTROLS = unrelated_control_assignment(FOCAL_CONCEPTS, NON_FOCAL_CONCEPTS)

print("\\n" + "=" * 72)
print("SELECTION — fixed before any model result exists")
print("=" * 72)
print(f"  selected (ranking order): {SELECTED_NAMES}")
print(f"  focal causal concepts:    {FOCAL_CONCEPTS}")
print(f"  non-focal (controls):     {NON_FOCAL_CONCEPTS}")
print("  external unrelated control assignment:")
for _focal, _control in sorted(UNRELATED_CONTROLS.items()):
    print(f"    {_focal:12s} -> {_control}")
print(
    "  rule: the i-th focal concept takes the (i mod n)-th non-focal concept, "
    "both in ranking order. No capability result, activation, or target-test "
    "example takes part."
)

CONFIG.concepts = tuple(SELECTED_NAMES)
CONFIG.causal_concepts = tuple(FOCAL_CONCEPTS)

SELECTED_CONCEPTS = {name: CONCEPT_CANDIDATES[name] for name in SELECTED_NAMES}
SUBSET = manifest_module.build_subset(
    PILOT_GROUPS,
    SELECTED_CONCEPTS,
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

_train = SUBSET["splits"]["train"]
_test = SUBSET["splits"]["test"]
_all_rows = _train + _test
N_TOTAL_GROUPS = len(_all_rows)
N_DISTINCT_IMAGES = len({row["image_id"] for row in _all_rows})
N_DISTINCT_RECORDINGS = len({str(row["audio_path"]) for row in _all_rows})
N_SIBLINGS_EXCLUDED = sum(
    (row.get("split_provenance") or {}).get("n_sibling_groups_excluded", 0)
    for row in _all_rows
)
TRAIN_IMAGES = {row["image_id"] for row in _train}
TEST_IMAGES = {row["image_id"] for row in _test}
IMAGE_OVERLAP = sorted(TRAIN_IMAGES & TEST_IMAGES)
if IMAGE_OVERLAP:
    raise RuntimeError(
        f"{len(IMAGE_OVERLAP)} image(s) appear in both the source-training and "
        f"held-out splits: {IMAGE_OVERLAP[:5]}. Source-derived directions would "
        "be estimated partly from the images they are then tested on."
    )
if N_DISTINCT_RECORDINGS != N_TOTAL_GROUPS:
    raise RuntimeError(
        f"{N_TOTAL_GROUPS} groups carry only {N_DISTINCT_RECORDINGS} distinct "
        "recordings; the audio arm would reuse a recording across cells"
    )

print("\\n" + "=" * 72)
print("UNIQUE-IMAGE SUBSET")
print("=" * 72)
print(f"  synchronized groups   {N_TOTAL_GROUPS}")
print(f"  distinct images       {N_DISTINCT_IMAGES}")
print(f"  distinct recordings   {N_DISTINCT_RECORDINGS}")
print(f"  groups per image      {N_TOTAL_GROUPS / max(1, N_DISTINCT_IMAGES):.2f}")
print(f"  sibling groups excluded at selection: {N_SIBLINGS_EXCLUDED}")
print(f"  split leakage check:  {LEAKAGE['ok']}")
print(f"  train/held-out image overlap: {len(IMAGE_OVERLAP)}")
for _concept in SELECTED_NAMES:
    _tr = len({r["image_id"] for r in _train if r["concept"] == _concept})
    _te = len({r["image_id"] for r in _test if r["concept"] == _concept})
    print(f"    {_concept:12s} train images {_tr}  held-out images {_te}")
print(
    f"    negatives     train images "
    f"{len({r['image_id'] for r in _train if not r['concept']})}  "
    f"held-out images {len({r['image_id'] for r in _test if not r['concept']})}"
)

SPLIT_PROVENANCE = {
    "seed": SPLIT_SEED,
    "profile": PROFILE.to_dict(),
    "selected_concepts": list(SELECTED_NAMES),
    "focal_concepts": list(FOCAL_CONCEPTS),
    "unrelated_controls": dict(sorted(UNRELATED_CONTROLS.items())),
    "n_groups": N_TOTAL_GROUPS,
    "n_distinct_images": N_DISTINCT_IMAGES,
    "n_distinct_recordings": N_DISTINCT_RECORDINGS,
    "train_heldout_image_overlap": IMAGE_OVERLAP,
    "leakage": LEAKAGE,
}
(RUN_DIR / "split_provenance.json").write_text(
    json.dumps(SPLIT_PROVENANCE, indent=2, default=str), encoding="utf-8"
)
from jlens.mmpilot.store import payload_checksum

SPLIT_PROVENANCE_CHECKSUM = payload_checksum(SPLIT_PROVENANCE)
print(f"\\nsplit provenance checksum {SPLIT_PROVENANCE_CHECKSUM}")
print(f"section 6 completed in {time.perf_counter() - _selection_t0:.1f} seconds")
'''
)

markdown(
    """
## 7. Staged budget — confirm each stage before it can spend anything

Every number below is derived from the configuration, not guessed. A "pass" is
one teacher-forced forward over the prompt plus one candidate sequence, so
scoring six candidates costs six passes — which is why widening the concept set
is not free and why three modalities cost half again what two did.

Read the numbers, then set the flags for **one stage at a time** in section 2
and re-run from there.
"""
)

code(
    '''
# 7. CPU ONLY. Print each stage's exact budget and derive the stage gates.
from jlens.mmpilot.tri_modal import (
    estimate_stage_passes,
    format_stage_budget,
    format_total_budget,
)

N_CAPABILITY_GROUPS = min(
    len(SELECTED_NAMES) * CONFIG.max_capability_groups_per_concept,
    len([row for row in _all_rows if row["concept"]]),
)
_budget_kwargs = dict(
    n_concepts=len(SELECTED_NAMES),
    n_focal_concepts=len(FOCAL_CONCEPTS),
    modalities=MODALITIES,
    layers=LAYERS,
    n_total_groups=N_TOTAL_GROUPS,
    n_capability_groups=N_CAPABILITY_GROUPS,
    n_targets_per_cell=N_TEST_POSITIVE_IMAGES + N_TEST_NEGATIVE_IMAGES,
    alphas=CONFIG.alphas,
    d_model=EXPECT_D_MODEL if RUN_REAL_AUDIO_TRANSFER else 24,
)
BUDGET_A = estimate_stage_passes(
    stage="A", causal_layers=(), **_budget_kwargs
)
BUDGET_B = estimate_stage_passes(
    stage="B", causal_layers=(PRIMARY_CAUSAL_LAYER,), **_budget_kwargs
)
BUDGET_C = estimate_stage_passes(
    stage="C", causal_layers=REPLICATION_LAYERS, **_budget_kwargs
)
BUDGETS = [BUDGET_A, BUDGET_B, BUDGET_C]
for _budget in BUDGETS:
    print(format_stage_budget(_budget))
    print()
print(format_total_budget(BUDGETS))
(RUN_DIR / "pass_budget.json").write_text(
    json.dumps([b.to_dict() for b in BUDGETS], indent=2, default=str), encoding="utf-8"
)

# The derived gates are never *carried* from here. Every cell that decides
# whether to spend model passes calls refresh_stage_gates(globals()) again,
# immediately before deciding, so editing section 2 and executing it is enough.
from jlens.mmpilot.stage_gates import format_stage_gates, refresh_stage_gates

STAGE_GATES = refresh_stage_gates(globals())
print()
print(format_stage_gates(STAGE_GATES, switches=globals()))
if not MODEL_STAGES_ENABLED:
    print()
    print("MODEL STAGES BLOCKED — nothing below loads a model. To proceed set,")
    print("in section 2 and in this order:")
    print()
    print("    RUN_MODEL_STAGES              = True")
    print("    CONFIRM_MODEL_LOAD            = True")
    print("    CONFIRM_REPRESENTATION_BUDGET = True   # Stage A")
    print("    RUN_REAL_AUDIO_TRANSFER       = True   # real data and real lenses")
print()
print(
    "Stage B is additionally gated on the AUDIO_CAPABILITY verdict in section "
    "11: fewer than two concepts readable in all three channels is an "
    "AUDIO CAPABILITY NO-GO and the causal stages are skipped."
)
'''
)

markdown(
    """
## 8. Load Gemma and resolve the **native** spoken-audio protocol

Nothing below runs unless `RUN_MODEL_STAGES` **and** `CONFIRM_MODEL_LOAD` are
both True.

The real branch goes through the tested package function
`jlens.mmpilot.real_backend.build_real_backend` with `resolve_audio=True`, which
probes the audio path rather than inferring it from a component being present.
`supports_audio=True` from processor inspection was true the entire time the
audio path was broken; only a probed
`jlens.mmpilot.audio.resolve_audio_interface` counts.

The resolved protocol's **fingerprint must equal the one this study is bound
to**. It covers the model and processor revisions, the transformers version,
the calling convention, the sampling rate and the placeholder token, so the
most likely cause of a mismatch is a different transformers release in the
runtime — and a study whose audio inputs mean something different from the
audited ones is a different study.
"""
)

code(
    '''
# 8a. Call-signature preflight. Binds every real-path call against the
# installed signatures on CPU, before the ~16 GB download may start.
PREFLIGHT_FAILURES = None
if not MODEL_STAGES_ENABLED:
    print("skipped: MODEL_STAGES_ENABLED is False")
else:
    from jlens.mmpilot.preflight import check_call_contracts

    PREFLIGHT_FAILURES = check_call_contracts()
    if PREFLIGHT_FAILURES:
        raise RuntimeError(
            "call-signature contracts failed:\\n  - " + "\\n  - ".join(PREFLIGHT_FAILURES)
        )
    print("call-signature contracts: all bind")
'''
)

code(
    '''
# 8b. Load the model (or the MOCK backend) and resolve the native audio path.
MODEL = None
BACKEND = None
INTERFACE = None
AUDIO_INTERFACE = None
AUDIO_PROTOCOL = None
ARCH_REPORT = None
AVAILABLE_MODALITIES = []
BLOCKED_MODALITIES = []
MODEL_REVISION_USED = None
PROCESSOR_REVISION_USED = None
if not MODEL_STAGES_ENABLED:
    print("skipped: MODEL_STAGES_ENABLED is False")
else:
    if RUN_REAL_AUDIO_TRANSFER:
        import getpass

        from jlens.mmpilot.real_backend import build_real_backend

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
        INTERFACE = BUNDLE.interface
        ARCH_REPORT = BUNDLE.architecture
        AUDIO_INTERFACE = BUNDLE.audio_interface
        MODEL_REVISION_USED = BUNDLE.model_revision
        PROCESSOR_REVISION_USED = BUNDLE.processor_revision
        if AUDIO_INTERFACE is None:
            raise RuntimeError(
                "the native spoken-audio path did not resolve: "
                f"{BUNDLE.audio_blocked_reason}. This study is about spoken "
                "audio; it does not silently degrade to text and image."
            )
        from jlens.mmpilot.tri_modal import assert_audio_protocol

        AUDIO_PROTOCOL = assert_audio_protocol(
            AUDIO_INTERFACE,
            expected_fingerprint=AUDIO_PROTOCOL_FINGERPRINT_EXPECTED,
        )
        print("resolved interface:")
        print(json.dumps(INTERFACE, indent=2, default=str))
        print(json.dumps(
            {k: ARCH_REPORT[k] for k in ("n_layers", "d_model", "vocab_size", "layout_path")
             if k in ARCH_REPORT},
            indent=2, default=str,
        ))
        print(f"model revision used: {MODEL_REVISION_USED}")
        print(f"audio protocol      {AUDIO_PROTOCOL['protocol_version']}")
        print(f"audio fingerprint   {AUDIO_PROTOCOL['protocol_fingerprint']}")
        print(f"dynamic placeholder count: "
              f"{AUDIO_PROTOCOL['dynamic_placeholder_count']}")
    else:
        from jlens.mmpilot.mock import MockPilotBackend

        BACKEND = MockPilotBackend(MOCK_WORLD, supports_audio=True)
        INTERFACE = BACKEND.interface
        MODEL_REVISION_USED = "mock"
        PROCESSOR_REVISION_USED = "mock"
        AUDIO_PROTOCOL = {
            "protocol_version": AUDIO_PROTOCOL_VERSION_EXPECTED,
            "protocol_fingerprint": "sha256:mock-audio-protocol",
            "call_convention": "mock",
            "dynamic_placeholder_count": False,
            "matches_expected_fingerprint": False,
            "note": (
                "MOCK: the native audio protocol is not resolved here. The "
                "fingerprint check runs only on the real path."
            ),
        }
        print("MOCK backend: three modalities, no processor, no audio tower")

    from jlens.mmpilot.pipeline import available_modalities

    AVAILABLE_MODALITIES, BLOCKED_MODALITIES = available_modalities(BACKEND, CONFIG)
    print(f"available modalities: {AVAILABLE_MODALITIES}")
    print(f"blocked modalities:   {BLOCKED_MODALITIES}")
    print(f"d_model {BACKEND.d_model}, {len(BACKEND.blocks)} decoder blocks")
    if "spoken_audio" not in AVAILABLE_MODALITIES:
        raise RuntimeError(
            "spoken_audio is not available. This study's primary question is "
            "about spoken audio and it does not proceed without it."
        )
'''
)

markdown(
    """
## 9. Load the three published, independently confirmed lenses

Frozen, text-calibrated, published elsewhere. **This notebook does not fit a
lens.** Each artifact is held to its own metadata record: publication status,
confirmation status, exact checksum, model id and revision, tokenizer revision,
fitted scale 100, calibration modality `text-only`, layer, `d_model` 2560, hook
site, residual convention, vector orientation, normalization convention, and
frozen.

The schema is *inspected*, not assumed: a required field that is absent is a
refusal, never a default. **Layer 32 and every earlier tested layer failed
confirmation and are refused by the same rule that accepts 35, 38 and 40.**
"""
)

code(
    '''
# 9. Validate and load the published lenses. Nothing is fitted here.
LENSES = None
LENS = None
LENS_CHECKSUMS = {}
LENS_REPORT = None
if not MODEL_STAGES_ENABLED:
    print("skipped: MODEL_STAGES_ENABLED is False")
else:
    from jlens.mmpilot.published_lens import (
        PublishedLensExpectations,
        PublishedLensSpec,
        format_lens_report,
        load_published_lenses,
    )

    if RUN_REAL_AUDIO_TRANSFER:
        LENS_SPECS = [
            PublishedLensSpec(
                layer=layer,
                path=PUBLISHED_LENS_PATHS[layer],
                expect_sha256=PUBLISHED_LENS_PINS[layer][1],
            )
            for layer in sorted(PUBLISHED_LENS_PINS)
        ]
        LENS_EXPECTATIONS = PublishedLensExpectations(
            model_repo_id=MODEL_REPO_ID,
            model_revision=MODEL_REVISION_USED,
            scale_point=LENS_FITTED_SCALE,
            d_model=EXPECT_D_MODEL,
        )
    else:
        from jlens.mmpilot.mock import build_mock_published_lenses

        _mock_lenses = build_mock_published_lenses(
            SCRATCH / "published_lenses", layers=LAYERS, d_model=BACKEND.d_model
        )
        LENS_SPECS = _mock_lenses["specs"]
        LENS_EXPECTATIONS = _mock_lenses["expectations"]
        MOCK_FAILED_LENS_SPEC = _mock_lenses["failed_spec"]

    LENSES = load_published_lenses(LENS_SPECS, LENS_EXPECTATIONS)
    LENS = LENSES.lens
    LENS_CHECKSUMS = dict(LENSES.checksums)
    LENS_REPORT = LENSES.to_dict()
    print(format_lens_report(LENSES))
    (RUN_DIR / "lens_validation.json").write_text(
        json.dumps(LENS_REPORT, indent=2, default=str), encoding="utf-8"
    )

    # The same rule that accepted the confirmed layers refuses a failed one.
    # Demonstrated here rather than asserted, because "we would have refused it"
    # is a claim and this is a check.
    if not RUN_REAL_AUDIO_TRANSFER and MOCK_FAILED_LENS_SPEC is not None:
        from jlens.mmpilot.published_lens import PublishedLensRefused

        try:
            load_published_lenses([MOCK_FAILED_LENS_SPEC], LENS_EXPECTATIONS)
        except PublishedLensRefused as _error:
            print("\\nrefusal check: a failed-confirmation artifact is rejected")
            print(f"  failed clauses: {_error.problems}")
        else:
            raise RuntimeError(
                "a failed-confirmation artifact was accepted; the refusal path "
                "is not working"
            )
'''
)

markdown(
    """
## 10. Architecture, spans and the invariance gate — in **all three** modalities

Before anything is measured: the exact residual hook site at every layer, the
text/image/audio token spans, and the final prompt position **after** the media
span. Then the two invariance checks — capturing an activation must not change
the logits, and a coefficient-zero edit must reproduce the clean scoring —
**run separately in each modality**. A gate that passed on text says nothing
about whether an image or an audio forward pass survives the same hook.

A modality whose invariance checks fail is refused causal work rather than
recorded with a weaker result.
"""
)

code(
    '''
# 10. Per-modality architecture report and invariance gate.
ARCHITECTURE = None
INVARIANCE = None
INVARIANT_MODALITIES = []
if not MODEL_STAGES_ENABLED:
    print("skipped: MODEL_STAGES_ENABLED is False")
else:
    from jlens.mmpilot.capability import build_question
    from jlens.mmpilot.pipeline import build_condition_inputs
    from jlens.mmpilot.tri_modal import (
        modality_architecture_report,
        run_invariance_by_modality,
    )

    if RUN_REAL_AUDIO_TRANSFER:
        from PIL import Image

        def _load_image(path):
            return Image.open(path).convert("RGB")

        def _load_audio(path):
            import soundfile as sf

            waveform, sample_rate = sf.read(path, dtype="float32")
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)
            return waveform, int(sample_rate)
    else:
        from jlens.mmpilot.mock import load_mock_media

        def _load_image(path):
            return load_mock_media(path)

        def _load_audio(path):
            return load_mock_media(path), 16000

    MEDIA = {"load_image": _load_image, "load_audio": _load_audio}

    _probe_question = build_question(sorted(SELECTED_NAMES))
    _probe_group = SUBSET["splits"]["train"][0]
    PROBE_INPUTS = {
        modality: build_condition_inputs(
            BACKEND, _probe_group, modality, _probe_question, MEDIA
        )
        for modality in AVAILABLE_MODALITIES
    }
    ARCHITECTURE = modality_architecture_report(
        BACKEND, PROBE_INPUTS, layers=CONFIG.layers
    )
    for _modality, _entry in sorted(ARCHITECTURE["per_modality"].items()):
        print(
            f"  {_modality:13s} prompt_len={_entry['prompt_len']:4d}  "
            f"final position={_entry['final_prompt_position']:4d}  "
            f"span={_entry['modality_token_span']}"
        )
        if _modality == "spoken_audio" and _entry.get("audio"):
            print(
                f"                placeholders="
                f"{_entry['audio'].get('n_placeholders')} "
                f"features imply {_entry['audio'].get('n_expected_from_features')}  "
                f"agree={_entry.get('placeholder_feature_counts_agree')}"
            )
    print(f"\\nhook site: {ARCHITECTURE['hook_site']}")
    print(f"read/edit: {ARCHITECTURE['read_and_edit_position']}")
    print(f"layers checked: {ARCHITECTURE['layers']} of "
          f"{ARCHITECTURE['n_decoder_blocks']} decoder blocks")

    INVARIANCE = run_invariance_by_modality(BACKEND, PROBE_INPUTS, CONFIG.layers)
    INVARIANT_MODALITIES = [
        modality
        for modality, entry in INVARIANCE["per_modality"].items()
        if entry["passed"]
    ]
    print(f"\\ninvariance gate passed in: {INVARIANT_MODALITIES}")
    (RUN_DIR / "architecture_and_invariance.json").write_text(
        json.dumps(
            {"architecture": ARCHITECTURE, "invariance": INVARIANCE},
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
'''
)

markdown(
    """
## 11. Capability gate — **verdict A, `AUDIO_CAPABILITY`**

Six-way complete candidate-sequence scoring, both option orders, the identical
question in every modality. Only the evidence channel changes: written caption,
image pixels, or the recording. No transcript reaches the audio condition, no
caption reaches the image condition, and no filename or dataset metadata
reaches any of them — checked in code, not promised in prose.

**If fewer than two concepts pass in text, image *and* spoken audio, this is an
`AUDIO_CAPABILITY_NO_GO` and the expensive stages are skipped.** A concept is
never replaced after its capability result is seen.
"""
)

code(
    '''
# 11. Behavioral gate, then verdict A.
CAPABILITY = None
CAPABILITY_VERDICT = None
STORE = None
FINGERPRINT = None
SELECTION_FINGERPRINT = None
CAPABILITY_OUTCOME = None
if not STAGE_A_ENABLED:
    print("skipped: STAGE_A_ENABLED is False")
else:
    from jlens.mmpilot.pipeline import scientific_fingerprint, stage_capability
    from jlens.mmpilot.store import RunFingerprint, UnitStore
    from jlens.mmpilot.tri_modal import (
        AUDIO_CAPABILITY_GO,
        audio_capability_verdict,
    )

    SELECTION_FINGERPRINT = scientific_fingerprint(
        CONFIG,
        ranked_concepts=RANKED_CONCEPTS,
        selected_concepts=SELECTED_NAMES,
        focal_concepts=FOCAL_CONCEPTS,
        unrelated_controls=UNRELATED_CONTROLS,
        derived_cache_fingerprint=DERIVED_CACHE_FINGERPRINT,
        split_provenance_checksum=SPLIT_PROVENANCE_CHECKSUM,
        n_train_positive_images=N_TRAIN_POSITIVE_IMAGES,
        n_train_negative_images=N_TRAIN_NEGATIVE_IMAGES,
        n_test_positive_images=N_TEST_POSITIVE_IMAGES,
        n_test_negative_images=N_TEST_NEGATIVE_IMAGES,
        verdict_version=TRI_MODAL_VERDICT_VERSION,
    )
    FINGERPRINT = RunFingerprint(
        mode=CONFIG.mode,
        model_repo_id=MODEL_REPO_ID if RUN_REAL_AUDIO_TRANSFER else "mock/gemma-like",
        model_revision=MODEL_REVISION_USED,
        processor_revision=PROCESSOR_REVISION_USED,
        layers=tuple(CONFIG.layers),
        lens_checksum=LENSES.combined_checksum,
        manifest_checksum=MANIFEST_CHECKSUM,
        split_id=SPLIT_SEED,
        intervention_config={
            "alphas": list(CONFIG.alphas),
            "direction_top_k": CONFIG.direction_top_k,
            "primary_causal_layer": PRIMARY_CAUSAL_LAYER,
            "replication_layers": list(REPLICATION_LAYERS),
            "off_diagonal_causal_only": CONFIG.off_diagonal_causal_only,
        },
        selection_config=SELECTION_FINGERPRINT,
        extra={
            "protocol": TRI_MODAL_VERDICT_VERSION,
            "modalities": list(MODALITIES),
            "pairs": list(ALL_PAIRS),
            "audio_protocol_version": AUDIO_PROTOCOL_VERSION_EXPECTED,
            "audio_protocol_fingerprint": (
                AUDIO_PROTOCOL_FINGERPRINT_EXPECTED
                if RUN_REAL_AUDIO_TRANSFER
                else AUDIO_PROTOCOL["protocol_fingerprint"]
            ),
            "transformers_version": TRANSFORMERS_VERSION,
            "per_layer_lens_checksums": {
                str(layer): value for layer, value in sorted(LENS_CHECKSUMS.items())
            },
            "derived_cache_fingerprint": DERIVED_CACHE_FINGERPRINT,
            "stage_plan": STAGE_PLAN,
        },
    )
    STORE = UnitStore(RUN_DIR, FINGERPRINT)
    print("run state:", STORE.open())
    print(f"fingerprint {FINGERPRINT.digest}")
    print(
        "  no unit from the pilot, the robustness study, the localization study "
        "or the audio audit can be reused here: the modality set, the three "
        "lens checksums, the audio protocol and the stage plan are all bound "
        "into this digest."
    )

    CAPABILITY_OUTCOME, CAPABILITY = stage_capability(
        BACKEND, STORE, SUBSET, CONFIG, MEDIA, modalities=AVAILABLE_MODALITIES
    )
    print("\\n" + CAPABILITY_OUTCOME.line("capability"))
    print("\\nper-concept accuracy (raw counts), margins and order stability:")
    for _concept, _per_modality in sorted(CAPABILITY["per_concept"].items()):
        print(f"  {_concept:12s} " + "  ".join(
            f"{m}={e['n_correct']}/{e['n']} (med margin "
            f"{(e['median_target_margin'] or 0.0):+.3f})"
            for m, e in sorted(_per_modality.items())
        ))
    _unstable = [
        r["sample_id"]
        for r in CAPABILITY_OUTCOME.records
        if not r.get("option_order_stable", True)
    ]
    print(f"\\noption-order-unstable samples: {len(_unstable)}")
    print("raw predictions (first 10):")
    for _record in CAPABILITY_OUTCOME.records[:10]:
        print(
            f"  {_record['sample_id']:38s} target={_record['concept']!s:12s} "
            f"pred={_record['prediction']!s:12s} margin="
            f"{_record['target_margin']:+.4f} order_stable="
            f"{_record.get('option_order_stable')}"
        )

    CAPABILITY_VERDICT = audio_capability_verdict(
        CAPABILITY,
        selected_concepts=SELECTED_NAMES,
        modalities=AVAILABLE_MODALITIES,
        thresholds=THRESHOLDS,
    )
    STORE.save("metric", "audio_capability_verdict", CAPABILITY_VERDICT)
    print("\\n" + "=" * 72)
    print(f"VERDICT A — AUDIO_CAPABILITY: {CAPABILITY_VERDICT['verdict']}")
    print("=" * 72)
    print(CAPABILITY_VERDICT["rationale"])
    if CAPABILITY_VERDICT["verdict"] != AUDIO_CAPABILITY_GO:
        print()
        print("The causal stages will not run. This is a capability result, not a")
        print("transfer result, and the design is not narrowed to rescue it.")
'''
)

markdown(
    """
## 12. Extract final-prompt-token activations at every validated layer

One forward pass per (group, modality), three layers captured from it, batch
size 1, moved to CPU immediately. Each (sample, modality, layer) is an atomic
checksum-bound unit written as soon as it finishes, so a disconnected runtime
loses at most the unit in flight.
"""
)

code(
    '''
# 12. Capture the final-prompt-token residual at every validated layer.
ACTIVATIONS = []
ACTIVATION_OUTCOME = None
if not STAGE_A_ENABLED:
    print("skipped: STAGE_A_ENABLED is False")
else:
    from jlens.mmpilot.pipeline import stage_activations

    ACTIVATION_OUTCOME = stage_activations(
        BACKEND, STORE, SUBSET, CONFIG, MEDIA,
        modalities=AVAILABLE_MODALITIES,
        retained_concepts=SELECTED_NAMES,
        model_revision=MODEL_REVISION_USED,
    )
    ACTIVATIONS = ACTIVATION_OUTCOME.records
    print(ACTIVATION_OUTCOME.line("activation"))
    print(f"distinct images captured: {len({r['image_id'] for r in ACTIVATIONS})}")
    for _modality in AVAILABLE_MODALITIES:
        _rows = [r for r in ACTIVATIONS if r["modality"] == _modality]
        print(f"  {_modality:13s} {len(_rows)} units over "
              f"{len({r['layer'] for r in _rows})} layers")
'''
)

markdown(
    """
## 13. J-space codes — the repository's ordinary sparse nonnegative pursuit

One frozen dictionary per layer, built from that layer's **own** published
lens, used, and released before the next is built. No basis is fitted,
rescaled, substituted, or shared between layers. The raw residual activations
are retained as the baseline.
"""
)

code(
    '''
# 13. Sparse nonnegative J-space coordinates, one layer at a time.
CODES = []
CODE_OUTCOMES = {}
if not STAGE_A_ENABLED:
    print("skipped: STAGE_A_ENABLED is False")
else:
    from jlens.mmpilot.pipeline import build_dictionaries, stage_codes

    for _layer in CONFIG.layers:
        _dictionaries = build_dictionaries(
            LENS,
            (_layer,),
            BACKEND,
            device="cuda" if (RUN_REAL_AUDIO_TRANSFER and torch.cuda.is_available()) else "cpu",
            dtype=torch.float16 if RUN_REAL_AUDIO_TRANSFER else torch.float32,
            build_chunk_rows=32768 if RUN_REAL_AUDIO_TRANSFER else None,
        )
        _outcome = stage_codes(
            STORE,
            ACTIVATIONS,
            _dictionaries,
            CONFIG,
            lens_checksum=LENS_CHECKSUMS[_layer],
        )
        CODE_OUTCOMES[_layer] = _outcome
        CODES.extend(_outcome.records)
        print(f"L{_layer}: " + _outcome.line("jspace")
              + f"  (lens {LENS_CHECKSUMS[_layer][:23]}...)")
        del _dictionaries
    print(f"\\ntotal J-space units: {len(CODES)}")
'''
)

markdown(
    """
## 14. All six directional pairs — **verdict B, `REPRESENTATIONAL_TRANSFER`**

At every validated layer, for every ordered cross-modal direction:
image-disjoint nearest-neighbour retrieval, matched-versus-mismatched
similarity, weighted sparse-support overlap, the raw-residual baseline, the
shuffled-label distribution and its p95, the exact exclusion counts, and the
number of distinct source and target images.

A candidate target is excluded whenever it shares the query's `image_id`, and
the group-level exclusion still applies underneath it — so a caption can never
retrieve its own photograph, or its own recording, through a sibling group.

The gate is **strictly beating the shuffled p95**, not a fixed additive margin:
accuracy is discrete at `1/n_queries`, so an additive margin can demand an
accuracy above 1.0. `text <-> image` is computed identically and reported as an
internal replication.
"""
)

code(
    '''
# 14. Six directional pairs at every validated layer, then verdict B.
REPRESENTATIONAL = {}
REPRESENTATIONAL_VERDICT = None
if not STAGE_A_ENABLED:
    print("skipped: STAGE_A_ENABLED is False")
else:
    from jlens.mmpilot.pipeline import stage_representational
    from jlens.mmpilot.tri_modal import representational_transfer_verdict

    for _layer in CONFIG.layers:
        REPRESENTATIONAL[_layer] = stage_representational(
            STORE, ACTIVATIONS, CODES, CONFIG,
            layer=_layer,
            modalities=AVAILABLE_MODALITIES,
        )
        print(f"\\n=== layer {_layer} ===")
        for _pair in ALL_PAIRS:
            _entry = REPRESENTATIONAL[_layer]["pairs"].get(_pair)
            if _entry is None:
                print(f"  {_pair:24s} not evaluated")
                continue
            _exclusions = _entry["exclusions"]
            print(
                f"  {_pair:24s} queries={_entry['jspace_retrieval']['n_queries']:3d}  "
                f"top1={_entry['jspace_retrieval']['top1_accuracy']:.3f}  "
                f"mrr={_entry['jspace_retrieval']['mrr']:.3f}  "
                f"shuffled_p95={_entry['shuffled_control']['p95_top1_accuracy']:.3f}"
            )
            print(
                f"      J gap {_entry['jspace_separation']['gap']:+.4f}  "
                f"raw gap {_entry['raw_residual_separation']['gap']:+.4f}  "
                f"support-overlap gap {_entry['jspace_support_overlap']['gap']:+.4f}  "
                f"raw top1 {_entry['raw_residual_retrieval']['top1_accuracy']:.3f}"
            )
            print(
                f"      src images {_entry['n_distinct_source_images']}  "
                f"tgt images {_entry['n_distinct_target_images']}  "
                f"excluded same-group {_exclusions['n_excluded_same_group']}  "
                f"extra same-image {_exclusions['n_excluded_same_image_different_group']}"
            )

    # Retrieval is pooled over every selected concept, so it cannot be filtered
    # per concept without recomputing it. The capability roster is passed in so
    # the verdict *discloses* which pooled concepts failed the behavioral gate
    # rather than leaving that invisible; it does not remove them.
    REPRESENTATIONAL_VERDICT = representational_transfer_verdict(
        REPRESENTATIONAL,
        thresholds=THRESHOLDS,
        primary_layer=PRIMARY_CAUSAL_LAYER,
        capability=CAPABILITY_VERDICT,
        pooled_concepts=SELECTED_NAMES,
    )
    STORE.save("metric", "representational_transfer_verdict", REPRESENTATIONAL_VERDICT)
    print("\\n" + "=" * 72)
    print(f"VERDICT B — REPRESENTATIONAL_TRANSFER: "
          f"{REPRESENTATIONAL_VERDICT['verdict']}")
    print("=" * 72)
    print(REPRESENTATIONAL_VERDICT["rationale"])
    print(f"\\naudio directions beating shuffled: "
          f"{REPRESENTATIONAL_VERDICT['audio_directions_beating_shuffled']}")
    print(f"text<->image replication: "
          f"{REPRESENTATIONAL_VERDICT['replication_directions_beating_shuffled']}")
    _pool = REPRESENTATIONAL_VERDICT["capability_pool_disclosure"]
    print(f"\\ncapability-ineligible concepts in this pooled retrieval: "
          f"{_pool['capability_ineligible_concepts_in_pool']}")
    print(f"  {_pool['reading']}")
'''
)

markdown(
    """
## 15. Stage A is complete — the measured budget before any causal work

Everything above is representational. Nothing causal has run, and nothing
causal starts by itself.

This cell prints what Stage A actually cost and what Stage B would cost, so the
Stage-B confirmation is made against measured numbers rather than the estimate
from section 7.
"""
)

code(
    '''
# 15. Stop point. Print the measured Stage-A cost and the Stage-B ask.
#
# The gates are re-derived from the raw switches right here, so this cell can
# never report a stage as "not requested" while section 2 says otherwise.
from jlens.mmpilot.stage_gates import format_stage_gates, refresh_stage_gates

STAGE_GATES = refresh_stage_gates(globals())
print(format_stage_gates(STAGE_GATES, switches=globals()))
print()

STAGE_A_MEASURED = None
if not STAGE_A_ENABLED:
    print("skipped: STAGE_A_ENABLED is False")
else:
    _status = STORE.status_report()
    STAGE_A_MEASURED = {
        "completed_units": _status["completed_units"],
        "capability": CAPABILITY_OUTCOME.line("capability"),
        "activation": ACTIVATION_OUTCOME.line("activation"),
        "jspace": {str(k): v.line("jspace") for k, v in sorted(CODE_OUTCOMES.items())},
        "estimated_stage_a_passes": BUDGET_A.total_passes,
        "capability_verdict": CAPABILITY_VERDICT["verdict"],
        "representational_verdict": REPRESENTATIONAL_VERDICT["verdict"],
    }
    STORE.save("metric", "stage_a_measured", STAGE_A_MEASURED)
    print("=" * 72)
    print("STAGE A COMPLETE — measured")
    print("=" * 72)
    for _stage, _count in sorted(_status["completed_units"].items()):
        print(f"  {_stage:24s} {_count}")
    print(f"\\n  estimated Stage A passes: {BUDGET_A.total_passes:,}")
    print(f"  verdict A: {CAPABILITY_VERDICT['verdict']}")
    print(f"  verdict B: {REPRESENTATIONAL_VERDICT['verdict']}")
    print()
    print(format_stage_budget(BUDGET_B))
    print()
    if not STAGE_B_REQUESTED:
        print("STAGE B NOT REQUESTED. To run the primary causal test set, in")
        print("section 2:")
        print()
        print("    RUN_L35_CAUSAL_STAGE      = True")
        print("    CONFIRM_L35_CAUSAL_BUDGET = True")
        print()
        print("and re-run from section 2. Stored Stage-A units are reused.")
'''
)

markdown(
    """
## 16. Stage B — the primary causal test at layer 35

**Verdict C, `L35_CAUSAL_TRANSFER`.**

Off-diagonal cells only. Each direction is estimated from the **source
modality's training images alone** — no target-modality activation and no
held-out example of any modality takes part — and applied at the final prompt
token of held-out targets that are distinct images and distinct recordings,
disjoint from every image the direction was estimated from.

Necessity (subtract from a held-out positive) and sufficiency (add to a matched
negative), with all four controls: zero coefficient, norm-matched random, an
external unrelated concept's direction, and the raw residual
positive-minus-negative difference. Alphas `0, 0.25, 0.5, 1.0`.

Layer 35 is primary because it is the **earliest independently confirmed**
lens. That is a statement about which lenses passed confirmation, not about
where in the decoder anything converges.

**A measured effect is not automatically evidence.** Every predeclared cell is
measured and printed, including cells for a concept the model could not read out
of one of the channels. Whether a cell may *support a claim* is decided in one
place — `jlens.mmpilot.admissibility` — and only cells whose concept passed the
behavioral capability gate in **text, image and spoken audio** are allowed to.
A concept that failed is `CAPABILITY_INELIGIBLE`: it stays in the raw table with
the arithmetic that rejected it, it counts toward no supporting total, no
bidirectional pair and no GO criterion, and it is never replaced by a substitute
concept chosen after the capability results were visible.
"""
)

code(
    '''
# 16. Source-only directions and the layer-35 causal cells, then verdict C.
#
# The gate is re-derived from the raw switches immediately before the decision.
# A gate computed in an earlier cell and left behind in the kernel is exactly
# how this cell used to print "skipped" with RUN_L35_CAUSAL_STAGE set to True.
from jlens.mmpilot.stage_gates import format_stage_gates, refresh_stage_gates

STAGE_GATES = refresh_stage_gates(globals())
print(format_stage_gates(STAGE_GATES, switches=globals()))
print()

DIRECTIONS = {}
INTERVENTIONS = {}
INTERVENTION_RECORDS = []
IMAGE_LEVEL = {}
DIVERGENCE = {}
CAUSAL_OUTCOMES = {}
DIRECTION_OUTCOMES = {}
PRIMARY_CAUSAL_VERDICT = None
CAUSAL_LAYERS_RUN = []
if not STAGE_B_REQUESTED:
    print("skipped: STAGE_B_REQUESTED is False")
elif CAPABILITY_VERDICT["verdict"] != "AUDIO_CAPABILITY_GO":
    print(
        "skipped: AUDIO CAPABILITY NO-GO. The causal stages do not run on a "
        "capability the model does not have."
    )
elif "spoken_audio" not in INVARIANT_MODALITIES:
    print(
        "skipped: the spoken-audio invariance checks did not pass, so an "
        "intervention's effect in that modality would have no interpretation."
    )
else:
    from dataclasses import replace as _replace

    from jlens.mmpilot.independence import (
        divergence_summary,
        resolve_image_identity,
        summarize_interventions_by_image,
    )
    from jlens.mmpilot.pipeline import (
        build_dictionaries,
        stage_causal,
        stage_directions,
    )
    from jlens.mmpilot.tri_modal import causal_transfer_verdict


    def run_causal_layer(layer, focal=None):
        """Directions and interventions at one layer, with its own dictionary.

        ``focal`` narrows which focal concepts model passes are spent on. It is
        an *execution* parameter only: CONFIG.causal_concepts and the run
        fingerprint keep the full predeclared focal set, so narrowing it here
        cannot make the completed Stage-A/Stage-B units unresumable, and no
        concept is ever substituted for an excluded one.
        """
        focal = list(FOCAL_CONCEPTS if focal is None else focal)
        config = _replace(CONFIG, causal_layers=(layer,))
        dictionaries = build_dictionaries(
            LENS,
            (layer,),
            BACKEND,
            device="cuda" if (RUN_REAL_AUDIO_TRANSFER and torch.cuda.is_available()) else "cpu",
            dtype=torch.float16 if RUN_REAL_AUDIO_TRANSFER else torch.float32,
            build_chunk_rows=32768 if RUN_REAL_AUDIO_TRANSFER else None,
        )
        direction_outcome, directions = stage_directions(
            STORE, CODES, ACTIVATIONS, dictionaries, config,
            concepts=SELECTED_NAMES,
            modalities=AVAILABLE_MODALITIES,
            lens_checksum=LENS_CHECKSUMS[layer],
        )
        del dictionaries
        DIRECTIONS.update(directions)
        DIRECTION_OUTCOMES[layer] = direction_outcome
        print(f"L{layer}: " + direction_outcome.line("direction"))
        for record in sorted(
            direction_outcome.records,
            key=lambda r: (r.get("concept") or "", r.get("source_modality") or "", r["kind"]),
        ):
            if record["kind"] != "source_concept":
                continue
            print(
                f"    {record['concept']:12s} {record['source_modality']:13s} "
                f"positives={record['n_source_positive_images']} images  "
                f"negatives={record['n_source_negative_images']} images  "
                f"uses_target_modality_data={record['uses_target_modality_data']}"
            )
        causal_outcome, summary = stage_causal(
            BACKEND, STORE, SUBSET, CODES, ACTIVATIONS, DIRECTIONS, config, MEDIA,
            concepts=focal,
            modalities=AVAILABLE_MODALITIES,
            all_concepts=SELECTED_NAMES,
            unrelated_controls=UNRELATED_CONTROLS,
        )
        CAUSAL_OUTCOMES[layer] = causal_outcome
        INTERVENTIONS[layer] = summary
        INTERVENTION_RECORDS.extend(causal_outcome.records)
        print(f"L{layer}: " + causal_outcome.line("intervention"))

        identity = resolve_image_identity(
            [*ACTIVATIONS, *CODES, *causal_outcome.records]
        )
        image_level = summarize_interventions_by_image(
            causal_outcome.records, identity, group_summary=summary
        )
        divergence = divergence_summary(image_level)
        STORE.save("metric", f"interventions_image_level_L{layer}", image_level)
        IMAGE_LEVEL[layer] = image_level
        DIVERGENCE[layer] = divergence
        print(
            f"    distinct images {image_level['n_distinct_images_overall']}  "
            f"groups {image_level['n_groups_overall']}  "
            f"pseudoreplicated rows "
            f"{divergence['n_rows_pseudoreplicated_at_group_level']}"
            f"/{divergence['n_rows']}"
        )
        if divergence["n_rows_pseudoreplicated_at_group_level"]:
            raise RuntimeError(
                "an intervention cell drew more than one observation from one "
                "photograph. With one group per image this cannot happen unless "
                "selection was bypassed; refusing to report a pseudoreplicated "
                "causal summary."
            )
        CAUSAL_LAYERS_RUN.append(layer)
        return image_level


    # Stage B measures every predeclared focal concept, including any that
    # failed the behavioral capability gate — those cells were declared as
    # diagnostics and they are kept. The capability result is handed to the
    # verdict so it can decide which of them may be *evidence*.
    _image_level = run_causal_layer(PRIMARY_CAUSAL_LAYER)
    PRIMARY_CAUSAL_VERDICT = causal_transfer_verdict(
        _image_level,
        layer=PRIMARY_CAUSAL_LAYER,
        focal_concepts=FOCAL_CONCEPTS,
        thresholds=THRESHOLDS,
        name=f"L{PRIMARY_CAUSAL_LAYER}_CAUSAL_TRANSFER",
        capability=CAPABILITY_VERDICT,
    )
    STORE.save("metric", "primary_causal_verdict", PRIMARY_CAUSAL_VERDICT)
    print("\\n" + "=" * 72)
    print(f"VERDICT C — L{PRIMARY_CAUSAL_LAYER}_CAUSAL_TRANSFER: "
          f"{PRIMARY_CAUSAL_VERDICT['verdict']}")
    print("=" * 72)
    print(PRIMARY_CAUSAL_VERDICT["rationale"])
    _adm = PRIMARY_CAUSAL_VERDICT["capability_admissibility"]
    print()
    print("MEASURED DIAGNOSTIC vs ADMISSIBLE EVIDENCE")
    print(f"  fixed focal concepts       {_adm['fixed_concepts']}")
    print(f"  capability-eligible        {_adm['eligible_concepts']}")
    print(f"  CAPABILITY_INELIGIBLE      {_adm['excluded_concept_names']}")
    for _entry in _adm["excluded_concepts"]:
        print(f"      {_entry['concept']}: {_entry['rejection_reason']}")
    print(f"\\naudio cells passing (measured, complete): "
          f"{PRIMARY_CAUSAL_VERDICT['audio_cells_passing']}")
    print(f"audio cells measured but INADMISSIBLE: "
          f"{PRIMARY_CAUSAL_VERDICT['audio_cells_measured_but_inadmissible']}")
    print(f"audio cells supporting a claim (admissible only): "
          f"{PRIMARY_CAUSAL_VERDICT['audio_cells_supporting_a_claim']}")
    print(f"text<->image cells passing (replication only, admissible): "
          f"{PRIMARY_CAUSAL_VERDICT['replication_cells_passing']}")
    print(f"\\n{_adm['no_post_hoc_replacement']}")
    print(f"\\n{PRIMARY_CAUSAL_VERDICT['layer_choice_note']}")
'''
)

markdown(
    """
## 17. Stage C — the same frozen design at layers 38 and 40

**Verdict D, `L38_L40_REPLICATION`.**

Behind its own confirmation flag, and only after Stage B has run. The design is
byte-identical to Stage B's — same concepts, same focal set, same controls,
same alphas, same targets. **No layer is selected on its causal outcome**; the
layers were fixed by which lenses passed confirmation, before any number here
existed.

**Passes are spent only on capability-eligible focal concepts.** The fixed focal
set does not change — it is bound into the run fingerprint, and narrowing it
would make the completed Stage-A and Stage-B units unresumable. Only the
*execution* narrows: a focal concept that failed the behavioral gate would
produce cells that could not be evidence, so measuring them again at two more
layers buys nothing. Its cells still appear in the Stage-C table, marked
`not_executed_capability_ineligible` — a different fact from "measured and
inadmissible", and recorded as a different one. The cell below prints the fixed
set, the eligible set, the excluded set with reasons, the maximum design budget
and the actual gated budget **before** spending anything, and no concept is
substituted for an excluded one.
"""
)

code(
    '''
# 17. The replication layers, then verdict D.
#
# The gate is re-derived from the raw switches immediately before the decision.
from jlens.mmpilot.stage_gates import format_stage_gates, refresh_stage_gates

STAGE_GATES = refresh_stage_gates(globals())
print(format_stage_gates(STAGE_GATES, switches=globals()))
print()

REPLICATION_VERDICT = None
LAYER_CAUSAL_VERDICTS = {}
STAGE_C_FOCAL_ADMISSIBILITY = None
STAGE_C_ELIGIBLE_FOCAL = []
BUDGET_C_GATED = None
if PRIMARY_CAUSAL_VERDICT is not None:
    LAYER_CAUSAL_VERDICTS[PRIMARY_CAUSAL_LAYER] = PRIMARY_CAUSAL_VERDICT

# The focal set is FIXED and stays fixed: the roster below only says which of
# the predeclared concepts may carry evidence. Nothing is added to replace one
# that cannot, and the design's maximum budget is printed beside the gated one
# so the reduction is visible as a consequence rather than as a smaller study.
if CAPABILITY_VERDICT is not None:
    from jlens.mmpilot.admissibility import concept_admissibility
    from jlens.mmpilot.tri_modal import format_focal_capability_gate

    STAGE_C_FOCAL_ADMISSIBILITY = concept_admissibility(
        list(FOCAL_CONCEPTS),
        capability=CAPABILITY_VERDICT,
        threshold=CAPABILITY_VERDICT.get("threshold", CAPABILITY_THRESHOLD),
    )
    STAGE_C_ELIGIBLE_FOCAL = list(STAGE_C_FOCAL_ADMISSIBILITY["eligible_concepts"])
    BUDGET_C_GATED = estimate_stage_passes(
        stage="C",
        causal_layers=REPLICATION_LAYERS,
        **{**_budget_kwargs, "n_focal_concepts": len(STAGE_C_ELIGIBLE_FOCAL)},
    )
    print(format_focal_capability_gate(
        STAGE_C_FOCAL_ADMISSIBILITY,
        max_budget=BUDGET_C,
        gated_budget=BUDGET_C_GATED,
        stage="C",
    ))
    print()

if not STAGE_C_REQUESTED:
    print("skipped: STAGE_C_REQUESTED is False")
elif PRIMARY_CAUSAL_VERDICT is None:
    print("skipped: Stage B did not run, so there is no frozen design to repeat")
elif not STAGE_C_ELIGIBLE_FOCAL:
    print(
        "skipped: no predeclared focal concept passed the behavioral capability "
        "gate in all three modalities, so every Stage-C cell would be a "
        "diagnostic. No concept is substituted to make the stage runnable."
    )
else:
    print(format_stage_budget(BUDGET_C_GATED))
    print()
    for _layer in REPLICATION_LAYERS:
        _level = run_causal_layer(_layer, focal=STAGE_C_ELIGIBLE_FOCAL)
        LAYER_CAUSAL_VERDICTS[_layer] = causal_transfer_verdict(
            _level,
            layer=_layer,
            # The verdict is still told the FULL fixed focal set, so an excluded
            # concept appears in the raw table as deliberately not executed
            # rather than vanishing from the design.
            focal_concepts=FOCAL_CONCEPTS,
            thresholds=THRESHOLDS,
            name=f"L{_layer}_CAUSAL_TRANSFER",
            capability=CAPABILITY_VERDICT,
            executed_concepts=STAGE_C_ELIGIBLE_FOCAL,
        )
        STORE.save(
            "metric", f"causal_verdict_L{_layer}", LAYER_CAUSAL_VERDICTS[_layer]
        )
        print(f"  L{_layer}: {LAYER_CAUSAL_VERDICTS[_layer]['verdict']}")

from jlens.mmpilot.tri_modal import replication_verdict

REPLICATION_VERDICT = replication_verdict(
    LAYER_CAUSAL_VERDICTS,
    primary=PRIMARY_CAUSAL_VERDICT,
    layers=REPLICATION_LAYERS,
)
if STORE is not None:
    STORE.save("metric", "replication_verdict", REPLICATION_VERDICT)
print("\\n" + "=" * 72)
print(f"VERDICT D — L38_L40_REPLICATION: {REPLICATION_VERDICT['verdict']}")
print("=" * 72)
print(REPLICATION_VERDICT["rationale"])
'''
)

markdown(
    """
## 18. Overall verdict and report — **verdict E**

A three-modality GO requires **all** of: at least two concepts readable in all
three channels; audio-related J-space structure beating the shuffled control;
at least one audio-related off-diagonal causal cell with the expected sign,
clearing its own matched random **and** external unrelated controls;
specificity rather than global candidate disruption; sane activation norms; at
least two distinct target images/recordings behind each claimed cell; the
invariance gate passing in every intervened modality; and — explicitly — a
result that is **not** text <-> image replication alone.

`WEAK GO` if representation transfers but the audio-related causality is weak
or one-directional. `NO-GO` if spoken-audio capability fails, audio J-space
structure does not beat controls, effects are non-specific, or interventions
resemble random.
"""
)

code(
    '''
# 18. Apply the rubric and write the report.
OVERALL_VERDICT = None
REPORT = None
if not STAGE_A_ENABLED:
    print("skipped: STAGE_A_ENABLED is False — no verdict is produced")
else:
    from jlens.mmpilot.tri_modal import overall_verdict, render_report

    OVERALL_VERDICT = overall_verdict(
        capability=CAPABILITY_VERDICT,
        representational=REPRESENTATIONAL_VERDICT,
        primary_causal=PRIMARY_CAUSAL_VERDICT,
        replication=REPLICATION_VERDICT,
        invariance=INVARIANCE,
        blocked_modalities=BLOCKED_MODALITIES,
        thresholds=THRESHOLDS,
    )
    REPORT = render_report(
        run_dir=str(RUN_DIR),
        capability=CAPABILITY_VERDICT,
        representational=REPRESENTATIONAL_VERDICT,
        primary_causal=PRIMARY_CAUSAL_VERDICT,
        replication=REPLICATION_VERDICT,
        overall=OVERALL_VERDICT,
        lens_report=LENS_REPORT,
        audio_protocol=AUDIO_PROTOCOL,
        budgets=[b.to_dict() for b in BUDGETS],
        resume=STORE.status_report(),
        mode=CONFIG.mode,
    )
    (RUN_DIR / "native_audio_transfer_report.md").write_text(REPORT, encoding="utf-8")
    RUN_MANIFEST = {
        "run_dir": str(RUN_DIR),
        "mode": CONFIG.mode,
        "commit": COMMIT,
        "switches": {
            "RUN_REAL_AUDIO_TRANSFER": RUN_REAL_AUDIO_TRANSFER,
            "RUN_MODEL_STAGES": RUN_MODEL_STAGES,
            "CONFIRM_MODEL_LOAD": CONFIRM_MODEL_LOAD,
            "CONFIRM_REPRESENTATION_BUDGET": CONFIRM_REPRESENTATION_BUDGET,
            "RUN_L35_CAUSAL_STAGE": RUN_L35_CAUSAL_STAGE,
            "CONFIRM_L35_CAUSAL_BUDGET": CONFIRM_L35_CAUSAL_BUDGET,
            "RUN_L38_L40_REPLICATION": RUN_L38_L40_REPLICATION,
            "CONFIRM_REPLICATION_BUDGET": CONFIRM_REPLICATION_BUDGET,
        },
        "stage_plan": STAGE_PLAN,
        "causal_layers_run": list(CAUSAL_LAYERS_RUN),
        "capability_admissibility_rule": OVERALL_VERDICT.get(
            "capability_admissibility_rule"
        ),
        "capability_admissibility": OVERALL_VERDICT.get("capability_admissibility"),
        "stage_c_focal_admissibility": STAGE_C_FOCAL_ADMISSIBILITY,
        "fingerprint_digest": FINGERPRINT.digest,
        "selection_fingerprint": SELECTION_FINGERPRINT,
        "split_provenance": SPLIT_PROVENANCE,
        "lens_validation": LENS_REPORT,
        "audio_protocol": AUDIO_PROTOCOL,
        "architecture": ARCHITECTURE,
        "invariance": INVARIANCE,
        "budgets": [b.to_dict() for b in BUDGETS],
        "verdicts": {
            "A_audio_capability": CAPABILITY_VERDICT,
            "B_representational_transfer": REPRESENTATIONAL_VERDICT,
            "C_primary_causal": PRIMARY_CAUSAL_VERDICT,
            "D_replication": REPLICATION_VERDICT,
            "E_overall": OVERALL_VERDICT,
        },
        "resume": STORE.status_report(),
    }
    (RUN_DIR / "native_audio_transfer_summary.json").write_text(
        json.dumps(RUN_MANIFEST, indent=2, default=str), encoding="utf-8"
    )
    (RUN_DIR / "run_manifest.json").write_text(
        json.dumps(
            {k: RUN_MANIFEST[k] for k in ("run_dir", "mode", "commit", "switches",
                                          "stage_plan", "causal_layers_run",
                                          "fingerprint_digest")},
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    STORE.save("metric", "three_modality_verdict", OVERALL_VERDICT)

    print("=" * 72)
    print("THE FIVE VERDICTS")
    print("=" * 72)
    print(f"  A  AUDIO_CAPABILITY               {CAPABILITY_VERDICT['verdict']}")
    print(f"  B  REPRESENTATIONAL_TRANSFER      "
          f"{REPRESENTATIONAL_VERDICT['verdict']}")
    print(f"  C  L{PRIMARY_CAUSAL_LAYER}_CAUSAL_TRANSFER            "
          f"{(PRIMARY_CAUSAL_VERDICT or {}).get('verdict', 'NOT_EVALUATED')}")
    print(f"  D  L38_L40_REPLICATION            {REPLICATION_VERDICT['verdict']}")
    print(f"  E  OVERALL_THREE_MODALITY_VERDICT {OVERALL_VERDICT['verdict']}")
    print()
    print(OVERALL_VERDICT["rationale"])
    print()
    for _name, _status in OVERALL_VERDICT["criteria_status"].items():
        print(f"  {_status:15s} {_name}")
    print()
    print(f"  {OVERALL_VERDICT['scope_limitation']}")
    print(f"  {OVERALL_VERDICT['layer_choice_note']}")
    print(f"  {OVERALL_VERDICT['intervention_limitation']}")
    print(f"  {OVERALL_VERDICT['alignment_limitation']}")
    print()
    print(f"report  {RUN_DIR / 'native_audio_transfer_report.md'}")
    print(f"summary {RUN_DIR / 'native_audio_transfer_summary.json'}")
'''
)

markdown(
    """
## 18b. Amended, capability-filtered report — no model, no recomputation

A completed run's *measurements* and a completed run's *interpretation* have
different lifetimes. When the reporting rule is corrected, the expensive units
are unaffected and must not be recomputed — so this cell re-derives verdicts C,
D and E from the stored units alone and writes a **separately versioned**
artifact beside the original.

- the original `native_audio_transfer_report.md` is never overwritten;
- the **raw-generation fingerprint is preserved**, so every completed capability
  and intervention unit stays reusable and a rerun still resumes them;
- the postprocessing rule is versioned and checksummed on its own, and the
  amended artifact binds to the run fingerprint **and** to a digest over the
  exact source units it read. Re-deriving it against changed units is refused.

Run this on a CPU runtime against an existing run directory. It loads nothing.
"""
)

code(
    '''
# 18b. Regenerate the verdicts from stored units under the admissibility rule.
AMENDED = None
AMENDED_PATHS = None
if STORE is None:
    print("skipped: there is no run state to amend")
else:
    from jlens.mmpilot.amend import (
        AMENDED_REPORT_NAME,
        POSTPROCESSING_VERSION,
        build_amended_report,
        verify_amended_binding,
        write_amended_report,
    )

    AMENDED = build_amended_report(
        STORE,
        primary_layer=PRIMARY_CAUSAL_LAYER,
        replication_layers=REPLICATION_LAYERS,
        focal_concepts=FOCAL_CONCEPTS,
        thresholds=THRESHOLDS,
        original_report_path=RUN_DIR / "native_audio_transfer_report.md",
        lens_report=LENS_REPORT,
        audio_protocol=AUDIO_PROTOCOL,
        invariance=INVARIANCE,
        blocked_modalities=BLOCKED_MODALITIES,
        mode=CONFIG.mode,
    )
    AMENDED_PATHS = write_amended_report(AMENDED, run_dir=RUN_DIR)
    # Immediately hold the artifact to the units it was computed from, so the
    # binding is exercised on the way out rather than only on some later reuse.
    print(verify_amended_binding(AMENDED["summary"], STORE))

    _amended_overall = AMENDED["verdicts"]["overall"]
    _amended_primary = AMENDED["verdicts"]["primary_causal"] or {}
    print()
    print("=" * 72)
    print(f"AMENDED VERDICTS — {POSTPROCESSING_VERSION}")
    print("=" * 72)
    print(f"  C  L{PRIMARY_CAUSAL_LAYER}_CAUSAL_TRANSFER            "
          f"{_amended_primary.get('verdict', 'NOT_EVALUATED')}")
    print(f"  D  L38_L40_REPLICATION            "
          f"{AMENDED['verdicts']['replication']['verdict']}")
    print(f"  E  OVERALL_THREE_MODALITY_VERDICT {_amended_overall['verdict']}")
    print()
    print(f"  admissible supporting cells: "
          f"{_amended_primary.get('audio_cells_supporting_a_claim')}")
    print(f"  measured but inadmissible:   "
          f"{_amended_primary.get('audio_cells_measured_but_inadmissible')}")
    print()
    print(f"  raw-generation fingerprint (unchanged) "
          f"{AMENDED['binding']['run_fingerprint_digest']}")
    print(f"  source units digest                   "
          f"{AMENDED['binding']['source_units']['combined_digest']}")
    print(f"  admissibility rule checksum           "
          f"{AMENDED['binding']['admissibility_rule_checksum']}")
    print()
    print(f"amended report  {AMENDED_PATHS['report']}")
    print(f"amended summary {AMENDED_PATHS['summary']}")
    print(f"original report kept at {RUN_DIR / 'native_audio_transfer_report.md'}")
'''
)

markdown(
    """
## 19. Resume state

Every stage writes one small checksummed JSON per unit as soon as it finishes,
so a disconnected Colab session loses at most the unit in flight. Re-running
this notebook against the same run directory reuses everything it can verify —
and refuses everything it cannot, because the modality set, the three lens
checksums, the audio protocol, the selection policy and the stage plan are all
bound into the fingerprint.
"""
)

code(
    '''
# 19. What was computed and what was reused.
if STORE is None:
    print("no run state: the model stages did not run")
    print()
    print("=" * 72)
    print("NOTHING RAN — this is the committed default.")
    print("=" * 72)
    print("To run the study, set in section 2, one stage at a time:")
    print()
    print("    RUN_REAL_AUDIO_TRANSFER       = True")
    print("    RUN_MODEL_STAGES              = True")
    print("    CONFIRM_MODEL_LOAD            = True")
    print("    CONFIRM_REPRESENTATION_BUDGET = True   # Stage A, after section 7")
    print()
    print("then, only after reading Stage A's result and Stage B's budget:")
    print()
    print("    RUN_L35_CAUSAL_STAGE          = True")
    print("    CONFIRM_L35_CAUSAL_BUDGET     = True   # Stage B")
    print()
    print("and only after that:")
    print()
    print("    RUN_L38_L40_REPLICATION       = True")
    print("    CONFIRM_REPLICATION_BUDGET    = True   # Stage C")
else:
    STATUS = STORE.status_report()
    print(f"run directory {STATUS['run_dir']}")
    print(f"state         {STATUS['status']}")
    print(f"fingerprint   {STATUS['fingerprint_digest']}")
    print("completed units:")
    for _stage, _count in sorted(STATUS["completed_units"].items()):
        print(f"  {_stage:24s} {_count}")
    if STATUS["invalid_units"]:
        print(f"invalid units (recomputed): {len(STATUS['invalid_units'])}")
    print()
    for _name, _outcome in (
        ("capability", CAPABILITY_OUTCOME),
        ("activation", ACTIVATION_OUTCOME),
    ):
        if _outcome is not None:
            print("  " + _outcome.line(_name))
    for _layer, _outcome in sorted(CODE_OUTCOMES.items()):
        print(f"  L{_layer} " + _outcome.line("jspace"))
    for _layer, _outcome in sorted(DIRECTION_OUTCOMES.items()):
        print(f"  L{_layer} " + _outcome.line("direction"))
    for _layer, _outcome in sorted(CAUSAL_OUTCOMES.items()):
        print(f"  L{_layer} " + _outcome.line("intervention"))
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
