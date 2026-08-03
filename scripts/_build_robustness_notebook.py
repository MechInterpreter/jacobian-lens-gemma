# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Generate ``notebooks/multimodal_jspace_spokencoco_robustness_colab.ipynb``.

Written from source rather than edited as JSON, so the committed notebook stays
output-free and byte-reproducible. Run
``python scripts/_build_robustness_notebook.py`` after changing a cell; a test
regenerates it and fails on drift.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TARGET = (
    REPO_ROOT / "notebooks" / "multimodal_jspace_spokencoco_robustness_colab.ipynb"
)

CELLS: list[tuple[str, str]] = []


def markdown(text: str) -> None:
    CELLS.append(("markdown", text.strip("\n")))


def code(text: str) -> None:
    CELLS.append(("code", text.strip("\n")))


markdown(
    """
# Six-concept robustness study — SpokenCOCO x Gemma 4 E4B IT

**One question.** The four-concept pilot survived image-level correction and
returned `GO_CONFIRMED_AFTER_IMAGE_DEDUP`. **Does it replicate?**

Six concepts instead of four. Eight distinct photographs per cell instead of
two. Three focal causal concepts instead of two. And — the change that matters
most — the photograph is the independent unit **from selection onward** rather
than by correction afterwards. The pilot's audit averaged repeated captions of
one image back together after the passes had been spent; this study never
spends them.

**This is not a replacement for the pilot notebook and not a framework.** One
layer (38). Two modalities. One frozen validated lens, never refitted.
Off-diagonal cells only. A three-point alpha sweep. Everything that would make
it bigger is deliberately out, because none of it is what "does this replicate"
needs.

**The decision is replication, not the strongest cell.** The pilot's rubric
took the largest off-diagonal effect and asked whether it beat its controls.
With three focal concepts that would be a way of reporting the luckiest cell.
A robustness GO here requires at least two of three focal concepts to transfer
in **both** directions, each against its own matched controls.

**Nothing starts by itself.** `RUN_REAL_ROBUSTNESS`, `RUN_MODEL_STAGES` and
`CONFIRM_MODEL_PASS_BUDGET` are all False in the committed notebook, and all
three must be set by hand. The budget cell prints the exact number of forward
passes before any of it can run.

## What this cannot tell you

- **Layer 38 is late in the decoder** and is the only validated layer. A
  final-prompt-token edit there cannot establish that an effect precedes
  answer-language convergence. However well this replicates, it is not
  evidence of pre-convergence semantics.
- **Spoken audio is excluded by design**, not by failure. Environmental audio
  is not tested at all. Neither absence is evidence about either.
- Interventions add and subtract a direction on the residual stream. That is
  **not erasure** and not **projection ablation**.
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
        [sys.executable, "-m", "pip", "install", "-e", f"{REPO_PATH}[gemma]"],
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
## 2. Configuration

Four switches, all False in the committed notebook, all set by hand:

| switch | what it does |
| --- | --- |
| `RUN_REAL_ROBUSTNESS` | use the real Drive dataset and the real lens instead of the deterministic MOCK world |
| `RUN_MODEL_STAGES` | allow Gemma to be loaded at all |
| `CONFIRM_MODEL_PASS_BUDGET` | acknowledge the printed forward-pass budget |
| `ENABLE_SPOKEN_AUDIO` | **leave False.** Spoken audio is outside this study |

`TINY_SMOKE` stays False too: this study's numbers only mean something at the
stated cell sizes, and a smoke-sized run that reported a verdict would be
reporting one about nothing.

The design is fixed here and not adjustable from results: six concepts, layer
38 only, text and image, eight distinct images in every cell, three focal
concepts taken from the top of the pre-model ranking.
"""
)

code(
    '''
# 2. Configuration. Requires section 1 (it imports from the repository).
# Nothing here mounts Drive, reads data, or loads a model.
RUN_REAL_ROBUSTNESS = False
RUN_MODEL_STAGES = False
TINY_SMOKE = False
ENABLE_SPOKEN_AUDIO = False

# Set to True only after reading the budget printed in section 7.
CONFIRM_MODEL_PASS_BUDGET = False

# ------------------------------------------------------------------ design
N_CONCEPTS = 6
N_FOCAL_CONCEPTS = 3
LAYERS = (38,)
CAUSAL_LAYERS = (38,)
ALPHAS = (0.0, 0.25, 0.5)
CAPABILITY_THRESHOLD = 0.7
N_TRAIN_POSITIVE_IMAGES = 8
N_TEST_POSITIVE_IMAGES = 8
N_TRAIN_NEGATIVE_IMAGES = 8
N_TEST_NEGATIVE_IMAGES = 8
SPLIT_SEED = "spokencoco-robustness-v1"

# ---------------------------------------------------------------- the lens
# Frozen, previously validated, never refitted here. This notebook does not
# fit a lens.
LENS_PATH = (
    "/content/drive/MyDrive/jacobian-lens-gemma/runs/"
    "text_jlens_early_layer_recalibration_v2/artifacts/lens.validated.pt"
)
LENS_EXPECT_SHA256 = (
    "sha256:4b17bf6086901e633f94d3391f5de6eccd3e735cc24cece63887505d73641c2b"
)
MODEL_REPO_ID = "google/gemma-3n-e4b-it"
MODEL_REVISION = "fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd"

# --------------------------------------------------------------- the data
SPOKENCOCO_BASE_ROOT = "/content/drive/MyDrive/datasets/cstf_spokencoco"
IMAGE_MEDIA_ROOT = "/content/drive/MyDrive/datasets/cstf_spokencoco/coco"
AUDIO_MEDIA_ROOT = "/content/drive/MyDrive/datasets/cstf_spokencoco/SpokenCOCO"
DOWNLOAD_CACHE = "/content/drive/MyDrive/datasets/cstf_spokencoco_download_cache"
MANIFEST_PATH = "/content/drive/MyDrive/datasets/spokencoco_manifest.json"
RUNS_ROOT = "/content/drive/MyDrive/jacobian-lens-gemma/runs"

# The completed pilot's derived evidence is reused read-only. Its expanded
# manifest is the metadata join this study would otherwise redo; its checksum
# of the original manifest is verified before a single group is taken from it.
COMPLETED_PILOT_RUN_DIR = (
    "/content/drive/MyDrive/jacobian-lens-gemma/runs/mmpilot_pilot_20260803T160711"
)

import json

from jlens.mmpilot.pipeline import PilotConfig
from jlens.mmpilot.robustness import ROBUSTNESS_VERDICT_VERSION, RobustnessThresholds
from jlens.mmpilot.selection import IMAGE_UNIQUE_PROFILE

SCRATCH = Path(os.environ.get("MMPILOT_SCRATCH") or "/content/mmpilot_scratch")
SCRATCH.mkdir(parents=True, exist_ok=True)
RESOLVED_RUNS_ROOT = Path(
    os.environ.get("MMPILOT_RUNS_ROOT") or (RUNS_ROOT if RUN_REAL_ROBUSTNESS else SCRATCH / "runs")
)
COMPLETED_PILOT_RUN_DIR = (
    os.environ.get("MMPILOT_PILOT_RUN_DIR") or COMPLETED_PILOT_RUN_DIR
)

MODALITIES = ("text", "image") + (("spoken_audio",) if ENABLE_SPOKEN_AUDIO else ())
PROFILE = IMAGE_UNIQUE_PROFILE
THRESHOLDS = RobustnessThresholds(
    n_concepts_required=N_CONCEPTS,
    n_focal_concepts=N_FOCAL_CONCEPTS,
    required_positive_images_per_cell=N_TEST_POSITIVE_IMAGES,
    required_negative_images_per_cell=N_TEST_NEGATIVE_IMAGES,
    capability_threshold=CAPABILITY_THRESHOLD,
)

CONFIG = PilotConfig(
    mode="robustness" if RUN_REAL_ROBUSTNESS else "mock",
    layers=tuple(LAYERS) if RUN_REAL_ROBUSTNESS else (4,),
    causal_layers=tuple(CAUSAL_LAYERS) if RUN_REAL_ROBUSTNESS else (4,),
    modalities=MODALITIES,
    capability_threshold=CAPABILITY_THRESHOLD,
    alphas=tuple(ALPHAS),
    n_target_examples=N_TEST_POSITIVE_IMAGES,
    pursuit_k=25 if RUN_REAL_ROBUSTNESS else 8,
    pursuit_correlation_chunk_size=65536 if RUN_REAL_ROBUSTNESS else None,
    direction_top_k=16 if RUN_REAL_ROBUSTNESS else 4,
    n_permutations=50 if RUN_REAL_ROBUSTNESS else 8,
    max_capability_groups_per_concept=8,
    seed=20260803,
    # The repair: the photograph is the unit before anything is spent.
    subset_profile="image_unique",
    image_unique_targets=True,
    min_source_positive_images=N_TRAIN_POSITIVE_IMAGES,
    min_source_negative_images=N_TRAIN_NEGATIVE_IMAGES,
    off_diagonal_causal_only=True,
)

print(f"RUN_REAL_ROBUSTNESS       = {RUN_REAL_ROBUSTNESS}")
print(f"RUN_MODEL_STAGES          = {RUN_MODEL_STAGES}")
print(f"CONFIRM_MODEL_PASS_BUDGET = {CONFIRM_MODEL_PASS_BUDGET}")
print(f"ENABLE_SPOKEN_AUDIO       = {ENABLE_SPOKEN_AUDIO}  (outside this study)")
print(f"TINY_SMOKE                = {TINY_SMOKE}")
print()
print(f"mode        {CONFIG.mode}")
print(f"modalities  {list(MODALITIES)}")
print(f"layers      {list(CONFIG.layers)}  causal {list(CONFIG.causal_layers)}")
print(f"alphas      {list(CONFIG.alphas)}")
print(f"profile     {PROFILE.version}  (max {PROFILE.max_groups_per_image} group per image)")
print(f"cells       8 distinct positive + 8 distinct negative images each")
print(f"verdict     {ROBUSTNESS_VERDICT_VERSION}")
print()
print("This notebook does not fit a lens.")
'''
)

markdown(
    """
## 3. Mount Google Drive

Read-only checks. This cell never creates, moves, or deletes anything inside
the dataset, and never touches the completed pilot's run directory except to
read from it.
"""
)

code(
    '''
# 3. Mount Drive and verify the configured paths exist.
if RUN_REAL_ROBUSTNESS and IN_COLAB:
    from google.colab import drive

    drive.mount("/content/drive", force_remount=False)

if RUN_REAL_ROBUSTNESS:
    missing = [
        path
        for path in (
            SPOKENCOCO_BASE_ROOT,
            IMAGE_MEDIA_ROOT,
            AUDIO_MEDIA_ROOT,
            MANIFEST_PATH,
            LENS_PATH,
            COMPLETED_PILOT_RUN_DIR,
        )
        if not Path(path).exists()
    ]
    if missing:
        raise RuntimeError(
            f"configured path(s) do not exist: {missing}. Nothing is discovered "
            "automatically; fix the configuration in section 2."
        )
    print("all configured paths exist")
else:
    print("skipped: RUN_REAL_ROBUSTNESS is False (the MOCK world needs no Drive)")
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
## 5. Load the derived cache and audit media (CPU only)

The completed pilot already paid for the metadata join, the media validation
and the synchronized-evidence audit. That work depends on the dataset and the
evidence rule, not on how many concepts a study selects, so it is **reused**:
this section reads the pilot's expanded manifest and verifies its recorded
checksum of the original manifest before taking a single group from it.

If that checksum disagrees, the audit is redone rather than trusted. A derived
artifact whose source has moved underneath it is not evidence about the
current dataset.
"""
)

code(
    '''
# 5. CPU ONLY. Reuse the derived evidence: load the completed pilot's expanded
# manifest, verify it against the current original manifest, and re-audit only
# if it does not match. No model, no GPU.
from datetime import datetime, timezone

from jlens.mmpilot import evidence as evidence_module
from jlens.mmpilot import expansion as expansion_module
from jlens.mmpilot import manifest as manifest_module
from jlens.mmpilot.concepts import discover_category_universe

if not RUN_REAL_ROBUSTNESS:
    # The synthetic world. Six concepts, two captions per image, so the
    # one-group-per-image rule has real sibling groups to exclude.
    from jlens.mmpilot.mock import MockWorld, build_mock_dataset

    MOCK_CONCEPTS_6 = {
        "bus": ("bus", "buses"),
        "cat": ("cat", "cats"),
        "clock": ("clock", "clocks"),
        "dog": ("dog", "dogs"),
        "pizza": ("pizza", "pizzas"),
        "zebra": ("zebra", "zebras"),
    }
    # The backend in section 8 must be built from *this* world. A backend with
    # a different concept set would encode the concepts it does not know as
    # single fallback tokens, and a one-token candidate beats a two-token one
    # on summed log-probability for reasons that have nothing to do with the
    # evidence.
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
MEDIA_ROOT_CONFIG = manifest_module.resolve_media_roots(
    image_roots=IMAGE_ROOTS, audio_roots=AUDIO_ROOTS
)

RUN_ID = (
    f"mmrobust_{CONFIG.mode}_"
    f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
)
RUN_DIR = Path(os.environ.get("MMPILOT_RUN_DIR") or (RESOLVED_RUNS_ROOT / RUN_ID))
RUN_DIR.mkdir(parents=True, exist_ok=True)
print(f"run directory {RUN_DIR}")

SEARCH_ROOTS = sorted({str(root) for root in IMAGE_ROOTS + AUDIO_ROOTS if root.is_dir()})
if RUN_REAL_ROBUSTNESS:
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

# --------------------------------------------------- reuse the derived join
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
print(EXPANSION_STATUS)
print(f"synchronized groups available: {len(PILOT_GROUPS)}")
print(
    f"distinct images available:     "
    f"{len({g['image_id'] for g in PILOT_GROUPS})}"
)
'''
)

markdown(
    """
## 6. Select six concepts and build the unique-image subset (CPU only)

**Feasibility is re-screened, ranking order is preserved.** A concept feasible
at the pilot's six images is not automatically feasible at sixteen, so every
candidate is re-scored against this study's requirements. The *order* is the
same deterministic score the pilot used — it is not re-sorted alphabetically,
because the order is what picks the focal concepts.

**The focal concepts are the first three in that order, and they are fixed
here — before Gemma is loaded.** The remaining three supply the external
unrelated controls, assigned by rotation. Nothing about this rule reads a
capability result, an activation, or a target-test example: a control chosen
after seeing how the candidates behave is not a control.

**If fewer than six concepts are feasible, this is a capability NO-GO for the
robustness profile.** A concept is never quietly swapped out after its results
are seen.

The subset takes **one synchronized group per image**, chosen by a seeded
stable rank over the group id. The sibling captions it excludes are recorded
on the row rather than dropped silently.
"""
)

code(
    '''
# 6. CPU ONLY. Re-screen feasibility at this study's cell sizes, take the top
# six in ranking order, fix the focal concepts and their controls, then build
# the one-group-per-image subset.
from jlens.mmpilot.selection import (
    select_focal_concepts,
    unrelated_control_assignment,
)
import time

_selection_t0 = time.perf_counter()
print(f"indexing {len(PILOT_GROUPS):,} synchronized groups once ...")
EVIDENCE_INDEX = evidence_module.build_evidence_index(
    PILOT_GROUPS, tuple(CONCEPT_CANDIDATES), EVIDENCE_CONFIG
)

ROBUSTNESS_REQUIREMENTS = expansion_module.ConceptRequirements(
    min_distinct_images=N_TRAIN_POSITIVE_IMAGES + N_TEST_POSITIVE_IMAGES,
    min_groups=N_TRAIN_POSITIVE_IMAGES + N_TEST_POSITIVE_IMAGES,
    min_train_positives=N_TRAIN_POSITIVE_IMAGES,
    min_test_positives=N_TEST_POSITIVE_IMAGES,
)
RANKING = expansion_module.rank_concepts(
    PILOT_GROUPS,
    CONCEPT_CANDIDATES,
    requirements=ROBUSTNESS_REQUIREMENTS,
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
    requirements=ROBUSTNESS_REQUIREMENTS,
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
N_SIBLINGS_EXCLUDED = sum(
    (row.get("split_provenance") or {}).get("n_sibling_groups_excluded", 0)
    for row in _all_rows
)

print("\\n" + "=" * 72)
print("UNIQUE-IMAGE SUBSET")
print("=" * 72)
print(f"  synchronized groups   {N_TOTAL_GROUPS}")
print(f"  distinct images       {N_DISTINCT_IMAGES}")
print(f"  groups per image      {N_TOTAL_GROUPS / max(1, N_DISTINCT_IMAGES):.2f}")
print(f"  sibling groups excluded at selection: {N_SIBLINGS_EXCLUDED}")
print(f"  split leakage check: {LEAKAGE['ok']}")
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
## 7. Pass and storage estimate — confirm before spending anything

Every number below is derived from the configuration, not guessed. Read them,
then set `CONFIRM_MODEL_PASS_BUDGET = True` in section 2 and re-run from there.

A "pass" is one teacher-forced forward over the prompt plus one candidate
sequence. Scoring six candidates costs six passes, which is why widening the
concept set is not free — and why this study skips same-modality causal cells:
they cost the same as the cross-modal ones and answer a different question.
"""
)

code(
    '''
# 7. CPU ONLY. Print the exact budget and refuse to continue without an
# explicit, separate confirmation.
from jlens.mmpilot.robustness import estimate_model_passes, format_budget

N_CAPABILITY_GROUPS = min(
    len(SELECTED_NAMES) * CONFIG.max_capability_groups_per_concept,
    len([row for row in _all_rows if row["concept"]]),
)
BUDGET = estimate_model_passes(
    n_concepts=len(SELECTED_NAMES),
    n_focal_concepts=len(FOCAL_CONCEPTS),
    modalities=MODALITIES,
    n_total_groups=N_TOTAL_GROUPS,
    n_capability_groups=N_CAPABILITY_GROUPS,
    n_targets_per_cell=N_TEST_POSITIVE_IMAGES + N_TEST_NEGATIVE_IMAGES,
    alphas=CONFIG.alphas,
    off_diagonal_only=CONFIG.off_diagonal_causal_only,
)
print(format_budget(BUDGET))
(RUN_DIR / "pass_budget.json").write_text(
    json.dumps(BUDGET.to_dict(), indent=2, default=str), encoding="utf-8"
)

BUDGET_CONFIRMED = bool(CONFIRM_MODEL_PASS_BUDGET)
print()
if not BUDGET_CONFIRMED:
    print("=" * 72)
    print("MODEL STAGES BLOCKED — the budget has not been confirmed.")
    print("=" * 72)
    print("To proceed, set in section 2:")
    print()
    print("    CONFIRM_MODEL_PASS_BUDGET = True")
    print("    RUN_MODEL_STAGES          = True")
    print("    RUN_REAL_ROBUSTNESS       = True   # for the real dataset and lens")
    print()
    print("then re-run from section 2. Nothing below loads a model until all")
    print("three are set.")
else:
    print("budget confirmed by CONFIRM_MODEL_PASS_BUDGET")

# The single gate every model stage below reads.
MODEL_STAGES_ENABLED = bool(RUN_MODEL_STAGES and BUDGET_CONFIRMED)
print(f"MODEL_STAGES_ENABLED = {MODEL_STAGES_ENABLED}")
'''
)

markdown(
    """
## 8. Load and audit Gemma

Nothing below this point runs unless `RUN_MODEL_STAGES` **and**
`CONFIRM_MODEL_PASS_BUDGET` are both True.
"""
)

code(
    '''
# 8. Load the model (or the MOCK backend) and record what was actually loaded.
MODEL = None
BACKEND = None
AVAILABLE_MODALITIES = []
BLOCKED_MODALITIES = []
if not MODEL_STAGES_ENABLED:
    print("skipped: MODEL_STAGES_ENABLED is False")
else:
    if RUN_REAL_ROBUSTNESS:
        import getpass

        if not os.environ.get("HF_TOKEN"):
            _token = getpass.getpass("HF_TOKEN (input hidden): ").strip()
            if not _token:
                raise RuntimeError("a Hugging Face token is required for the gated repo")
            os.environ["HF_TOKEN"] = _token
        from jlens.gemma4 import load_gemma4
        from jlens.mmpilot.backend import GemmaPilotBackend

        MODEL, PROCESSOR = load_gemma4(
            MODEL_REPO_ID,
            revision=MODEL_REVISION,
            token=os.environ["HF_TOKEN"],
            allow_model_load=True,
        )
        BACKEND = GemmaPilotBackend(MODEL, PROCESSOR)
        MODEL_REVISION_USED = MODEL_REVISION
        PROCESSOR_REVISION_USED = MODEL_REVISION
    else:
        from jlens.mmpilot.mock import MockPilotBackend

        BACKEND = MockPilotBackend(MOCK_WORLD, supports_audio=False)
        MODEL_REVISION_USED = "mock"
        PROCESSOR_REVISION_USED = "mock"

    from jlens.mmpilot.pipeline import available_modalities

    AVAILABLE_MODALITIES, BLOCKED_MODALITIES = available_modalities(BACKEND, CONFIG)
    print(f"available modalities: {AVAILABLE_MODALITIES}")
    print(f"blocked modalities:   {BLOCKED_MODALITIES}")
    print(f"d_model {BACKEND.d_model}, {len(BACKEND.blocks)} decoder blocks")
    if "spoken_audio" in AVAILABLE_MODALITIES:
        raise RuntimeError(
            "spoken audio is outside this study by design; it must not be in "
            "the configured modalities"
        )
'''
)

markdown(
    """
## 9. Capability gate

Six-way complete candidate-sequence scoring, both option orders, identical
question in both modalities. **All six concepts must pass in text and image.**
A concept that fails is reported as a robustness-profile capability NO-GO; it
is never replaced by a seventh after the fact.
"""
)

code(
    '''
# 9. Behavioral gate. Six candidates, both option orders, complete sequences.
CAPABILITY = None
STORE = None
if not MODEL_STAGES_ENABLED:
    print("skipped: MODEL_STAGES_ENABLED is False")
else:
    from jlens.mmpilot.pipeline import scientific_fingerprint, stage_capability
    from jlens.mmpilot.store import RunFingerprint, UnitStore

    if RUN_REAL_ROBUSTNESS:
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

    SELECTION_FINGERPRINT = scientific_fingerprint(
        CONFIG,
        ranked_concepts=RANKED_CONCEPTS,
        selected_concepts=SELECTED_NAMES,
        focal_concepts=FOCAL_CONCEPTS,
        unrelated_controls=UNRELATED_CONTROLS,
        derived_cache_fingerprint=MANIFEST_CHECKSUM,
        split_provenance_checksum=SPLIT_PROVENANCE_CHECKSUM,
        n_train_positive_images=N_TRAIN_POSITIVE_IMAGES,
        n_train_negative_images=N_TRAIN_NEGATIVE_IMAGES,
        n_test_positive_images=N_TEST_POSITIVE_IMAGES,
        n_test_negative_images=N_TEST_NEGATIVE_IMAGES,
        verdict_version=ROBUSTNESS_VERDICT_VERSION,
    )
    FINGERPRINT = RunFingerprint(
        mode=CONFIG.mode,
        model_repo_id=MODEL_REPO_ID if RUN_REAL_ROBUSTNESS else "mock/gemma-like",
        model_revision=MODEL_REVISION_USED,
        processor_revision=PROCESSOR_REVISION_USED,
        layers=tuple(CONFIG.layers),
        lens_checksum=LENS_EXPECT_SHA256 if RUN_REAL_ROBUSTNESS else "sha256:mock-identity-lens",
        manifest_checksum=MANIFEST_CHECKSUM,
        split_id=SPLIT_SEED,
        intervention_config={
            "alphas": list(CONFIG.alphas),
            "direction_top_k": CONFIG.direction_top_k,
            "causal_layers": list(CONFIG.causal_layers),
        },
        selection_config=SELECTION_FINGERPRINT,
    )
    STORE = UnitStore(RUN_DIR, FINGERPRINT)
    print("run state:", STORE.open())
    print(f"fingerprint {FINGERPRINT.digest}")
    print(
        "  units from the four-concept pilot can never be reused here: the "
        "selection config, the concept set and the six-way candidate scoring "
        "are all bound into this digest."
    )

    CAPABILITY_OUTCOME, CAPABILITY = stage_capability(
        BACKEND, STORE, SUBSET, CONFIG, MEDIA, modalities=AVAILABLE_MODALITIES
    )
    print("\\n" + CAPABILITY_OUTCOME.line("capability"))
    print("\\nper-concept accuracy:")
    for _concept, _per_modality in sorted(CAPABILITY["per_concept"].items()):
        print(f"  {_concept:12s} " + "  ".join(
            f"{m}={e['n_correct']}/{e['n']}" for m, e in sorted(_per_modality.items())
        ))
    RETAINED = CAPABILITY["text_image_retained_concepts"]
    print(f"\\nretained (text+image): {RETAINED}")
    CAPABILITY_FAILED = [c for c in SELECTED_NAMES if c not in RETAINED]
    if CAPABILITY_FAILED:
        print(
            f"\\nROBUSTNESS-PROFILE CAPABILITY NO-GO: {CAPABILITY_FAILED} did not "
            "clear the gate in both modalities. These concepts are NOT replaced "
            "— the design fixed its six before any model ran, and swapping one "
            "now would be selecting on the outcome."
        )
'''
)

markdown(
    """
## 10. Validate the frozen lens

The lens is checksum-pinned and validated against the model it is about to be
used with. **This notebook does not fit a lens.** If no compatible lens exists
the study stops and names the missing artifact.
"""
)

code(
    '''
# 10. Validate the frozen, previously fitted lens. Nothing is fitted here.
LENS = None
LENS_VALIDATION = None
if not MODEL_STAGES_ENABLED:
    print("skipped: MODEL_STAGES_ENABLED is False")
else:
    from jlens.mmpilot.jspace import validate_lens

    if RUN_REAL_ROBUSTNESS:
        from jlens.lens import JacobianLens
        from jlens.mmpilot.cache import file_sha256

        _checksum = file_sha256(LENS_PATH)
        if _checksum != LENS_EXPECT_SHA256:
            raise RuntimeError(
                f"lens checksum {_checksum} != pinned {LENS_EXPECT_SHA256}; "
                "refusing to use a lens other than the validated one"
            )
        LENS = JacobianLens.load(LENS_PATH)
        _lens_path, _lens_checksum = LENS_PATH, _checksum
        _expect_repo, _expect_revision = MODEL_REPO_ID, MODEL_REVISION
    else:
        from jlens.mmpilot.mock import mock_lens

        LENS = mock_lens(layers=CONFIG.layers)
        _lens_path, _lens_checksum = "mock", "sha256:mock-identity-lens"
        _expect_repo, _expect_revision = "mock/gemma-like", "mock"

    LENS_VALIDATION = validate_lens(
        LENS,
        lens_path=_lens_path,
        lens_checksum=_lens_checksum,
        layers=CONFIG.layers,
        model_repo_id=_expect_repo,
        model_revision=_expect_revision,
        expect_model_repo_id=_expect_repo,
        expect_model_revision=_expect_revision,
        expect_d_model=BACKEND.d_model,
    )
    print(json.dumps(LENS_VALIDATION, indent=2, default=str)[:1200])
    print("\\nfrozen:", LENS_VALIDATION["frozen"])
'''
)

markdown(
    """
## 11. Extract layer-38 activations

One forward pass per (group, modality), final prompt token, one layer.
"""
)

code(
    '''
# 11. Capture the final-prompt-token residual at layer 38.
ACTIVATIONS = []
if not MODEL_STAGES_ENABLED:
    print("skipped: MODEL_STAGES_ENABLED is False")
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
'''
)

markdown(
    """
## 12. Compute J-space codes

Nonnegative gradient pursuit against the frozen dictionary. No basis is fitted,
rescaled, or substituted.
"""
)

code(
    '''
# 12. Sparse nonnegative J-space coordinates for every stored activation.
CODES = []
DICTIONARIES = {}
if not MODEL_STAGES_ENABLED:
    print("skipped: MODEL_STAGES_ENABLED is False")
else:
    from jlens.mmpilot.pipeline import build_dictionaries, stage_codes

    DICTIONARIES = build_dictionaries(LENS, CONFIG.layers, BACKEND)
    CODE_OUTCOME = stage_codes(
        STORE, ACTIVATIONS, DICTIONARIES, CONFIG, lens_checksum=LENS_VALIDATION["lens_checksum"]
    )
    CODES = CODE_OUTCOME.records
    print(CODE_OUTCOME.line("jspace"))
'''
)

markdown(
    """
## 13. Image-disjoint representational tests

A candidate target is excluded whenever it shares the query's `image_id`, and
the group-level exclusion still applies underneath it. The same rule governs
retrieval, matched/mismatched separation, weighted support overlap, the
raw-residual baseline and the shuffled-label control — a report where some of
those were image-disjoint and others were not would not mean anything.

The gate is **strictly beating the shuffled p95**, not a fixed additive margin:
accuracy is discrete at `1/n_queries`, so an additive margin can demand an
accuracy above 1.0.
"""
)

code(
    '''
# 13. Cross-modal retrieval, separation, support overlap, raw baseline and the
# shuffled control — all under the same image-level exclusion rule.
REPRESENTATIONAL = None
if not MODEL_STAGES_ENABLED:
    print("skipped: MODEL_STAGES_ENABLED is False")
else:
    from jlens.mmpilot.pipeline import stage_representational

    REPRESENTATIONAL = stage_representational(
        STORE, ACTIVATIONS, CODES, CONFIG,
        layer=CONFIG.causal_layers[-1],
        modalities=AVAILABLE_MODALITIES,
    )
    for _pair, _entry in sorted(REPRESENTATIONAL["pairs"].items()):
        _exclusions = _entry["exclusions"]
        print(
            f"  {_pair:14s} queries={_entry['jspace_retrieval']['n_queries']:3d}  "
            f"top1={_entry['jspace_retrieval']['top1_accuracy']:.3f}  "
            f"mrr={_entry['jspace_retrieval']['mrr']:.3f}  "
            f"shuffled_p95={_entry['shuffled_control']['p95_top1_accuracy']:.3f}"
        )
        print(
            f"    J gap {_entry['jspace_separation']['gap']:+.4f}  "
            f"raw gap {_entry['raw_residual_separation']['gap']:+.4f}  "
            f"support-overlap gap {_entry['jspace_support_overlap']['gap']:+.4f}"
        )
        print(
            f"    eligible targets per query min/med/max "
            f"{_exclusions['eligible_targets']['min']}/"
            f"{_exclusions['eligible_targets']['median']}/"
            f"{_exclusions['eligible_targets']['max']}  "
            f"(same-group excl {_exclusions['n_excluded_same_group']}, "
            f"extra same-image excl "
            f"{_exclusions['n_excluded_same_image_different_group']})"
        )
'''
)

markdown(
    """
## 14. Estimate source-only directions

Each direction is estimated from **eight distinct training photographs** of one
modality and eight distinct matched negatives, with the two sets disjoint. No
target-modality activation and no test example of any modality takes part. A
short set refuses rather than shrinking.
"""
)

code(
    '''
# 14. Source-derived directions, from distinct training images only.
DIRECTIONS = {}
if not MODEL_STAGES_ENABLED:
    print("skipped: MODEL_STAGES_ENABLED is False")
else:
    from jlens.mmpilot.pipeline import stage_directions

    DIRECTION_OUTCOME, DIRECTIONS = stage_directions(
        STORE, CODES, ACTIVATIONS, DICTIONARIES, CONFIG,
        concepts=SELECTED_NAMES,
        modalities=AVAILABLE_MODALITIES,
        lens_checksum=LENS_VALIDATION["lens_checksum"],
    )
    print(DIRECTION_OUTCOME.line("direction"))
    for _record in sorted(
        DIRECTION_OUTCOME.records,
        key=lambda r: (r.get("concept") or "", r.get("source_modality") or "", r["kind"]),
    ):
        if _record["kind"] != "source_concept":
            continue
        print(
            f"  {_record['concept']:12s} {_record['source_modality']:6s} "
            f"positives={_record['n_source_positive_images']} images  "
            f"negatives={_record['n_source_negative_images']} images  "
            f"uses_target_modality_data={_record['uses_target_modality_data']}"
        )
'''
)

markdown(
    """
## 15. Off-diagonal causal interventions

Text-derived directions applied to image targets, and image-derived directions
applied to text targets. Necessity (subtract from a held-out positive) and
sufficiency (add to a matched negative), eight distinct photographs each, all
four controls, at the final prompt token.

Same-modality cells are skipped: they cost the same and answer a different
question. The core question here is cross-modal transfer.
"""
)

code(
    '''
# 15. The cross-modal transfer cells, with all four controls.
INTERVENTIONS = None
INTERVENTION_RECORDS = []
if not MODEL_STAGES_ENABLED:
    print("skipped: MODEL_STAGES_ENABLED is False")
else:
    from jlens.mmpilot.pipeline import stage_causal

    CAUSAL_OUTCOME, INTERVENTIONS = stage_causal(
        BACKEND, STORE, SUBSET, CODES, ACTIVATIONS, DIRECTIONS, CONFIG, MEDIA,
        concepts=FOCAL_CONCEPTS,
        modalities=AVAILABLE_MODALITIES,
        all_concepts=SELECTED_NAMES,
        unrelated_controls=UNRELATED_CONTROLS,
    )
    INTERVENTION_RECORDS = CAUSAL_OUTCOME.records
    print(CAUSAL_OUTCOME.line("intervention"))
    print(
        f"distinct target images touched: "
        f"{len({r['image_id'] for r in INTERVENTION_RECORDS})}"
    )
    print(
        f"target selection version: "
        f"{sorted({r['target_selection_version'] for r in INTERVENTION_RECORDS})}"
    )
'''
)

markdown(
    """
## 16. Aggregate at the image level

Repeated observations of one photograph are averaged **within** the image
before the cell statistic is computed. With one group per image there should be
nothing to average — and the report says so explicitly, which is the check that
the selection repair actually worked.
"""
)

code(
    '''
# 16. Image-level aggregation. With the unique-image profile this should be a
# no-op, and that is exactly what is verified.
IMAGE_LEVEL = None
if not MODEL_STAGES_ENABLED:
    print("skipped: MODEL_STAGES_ENABLED is False")
else:
    from jlens.mmpilot.independence import (
        divergence_summary,
        resolve_image_identity,
        summarize_interventions_by_image,
    )

    IDENTITY = resolve_image_identity([*ACTIVATIONS, *CODES, *INTERVENTION_RECORDS])
    IMAGE_LEVEL = summarize_interventions_by_image(
        INTERVENTION_RECORDS, IDENTITY, group_summary=INTERVENTIONS
    )
    DIVERGENCE = divergence_summary(IMAGE_LEVEL)
    STORE.save("metric", "interventions_image_level", IMAGE_LEVEL)
    print(f"groups: {IMAGE_LEVEL['n_groups_overall']}")
    print(f"distinct images: {IMAGE_LEVEL['n_distinct_images_overall']}")
    print(
        f"cells pseudoreplicated at group level: "
        f"{DIVERGENCE['n_rows_pseudoreplicated_at_group_level']} of {DIVERGENCE['n_rows']}"
    )
    print(f"max |image - group| effect: {DIVERGENCE['max_abs_divergence']:.6f}")
    if DIVERGENCE["n_rows_pseudoreplicated_at_group_level"] == 0:
        print(
            "  as designed: one synchronized group per photograph, so no "
            "aggregation was needed to make the unit honest"
        )
'''
)

markdown(
    """
## 17. Robustness verdict and report

The decision is **replication**: at least two of three focal concepts
transferring in both directions, each against its own matched controls. The
strongest single cell decides nothing.
"""
)

code(
    '''
# 17. Apply the robustness rubric and write the report.
VERDICT = None
if not MODEL_STAGES_ENABLED:
    print("skipped: MODEL_STAGES_ENABLED is False")
else:
    from jlens.mmpilot.robustness import render_report, robustness_verdict

    VERDICT = robustness_verdict(
        capability=CAPABILITY,
        representational=REPRESENTATIONAL,
        interventions=IMAGE_LEVEL,
        selected_concepts=SELECTED_NAMES,
        focal_concepts=FOCAL_CONCEPTS,
        unrelated_controls=UNRELATED_CONTROLS,
        blocked_modalities=BLOCKED_MODALITIES,
        thresholds=THRESHOLDS,
    )
    REPORT = render_report(
        run_dir=str(RUN_DIR),
        verdict=VERDICT,
        budget=BUDGET.to_dict(),
        resume=STORE.status_report(),
        mode=CONFIG.mode,
    )
    (RUN_DIR / "robustness_report.md").write_text(REPORT, encoding="utf-8")
    (RUN_DIR / "robustness_summary.json").write_text(
        json.dumps(
            {
                "verdict": VERDICT,
                "budget": BUDGET.to_dict(),
                "split_provenance": SPLIT_PROVENANCE,
                "selection_fingerprint": SELECTION_FINGERPRINT,
                "fingerprint_digest": FINGERPRINT.digest,
                "lens_validation": LENS_VALIDATION,
                "representational": REPRESENTATIONAL,
                "interventions_image_level": IMAGE_LEVEL,
                "capability": CAPABILITY,
                "resume": STORE.status_report(),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    STORE.save("metric", "robustness_verdict", VERDICT)
    print("=" * 72)
    print(f"VERDICT: {VERDICT['verdict']}")
    print("=" * 72)
    print(VERDICT["rationale"])
    print()
    for _name, _status in VERDICT["criteria_status"].items():
        print(f"  {_status:15s} {_name}")
    print(
        f"\\n  transferred both directions: "
        f"{VERDICT['concepts_transferring_both_directions']}"
    )
    print(
        f"  transferred either direction: "
        f"{VERDICT['concepts_transferring_either_direction']}"
    )
    print(f"\\n  {VERDICT['late_layer_limitation']}")
    print(f"  {VERDICT['scope_limitation']}")
    print(f"\\nreport  {RUN_DIR / 'robustness_report.md'}")
    print(f"summary {RUN_DIR / 'robustness_summary.json'}")
'''
)

markdown(
    """
## 18. Resume state

Every stage writes one small checksummed JSON per unit as soon as it finishes,
so a disconnected Colab session loses at most the unit in flight. Re-running
this notebook against the same run directory reuses everything it can verify.

Units from the four-concept pilot can never be reused here: the selection
config, the concept set and the six-way candidate scoring are all bound into
the fingerprint, and a four-way capability score is not comparable to a
six-way one.
"""
)

code(
    '''
# 18. What was computed and what was reused.
if STORE is None:
    print("no run state: the model stages did not run")
    print()
    print("=" * 72)
    print("NOTHING RAN — this is the committed default.")
    print("=" * 72)
    print("To run the study, set in section 2:")
    print()
    print("    RUN_REAL_ROBUSTNESS       = True")
    print("    RUN_MODEL_STAGES          = True")
    print("    CONFIRM_MODEL_PASS_BUDGET = True   # after reading section 7")
    print()
    print("Leave ENABLE_SPOKEN_AUDIO and TINY_SMOKE False.")
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
        ("jspace", CODE_OUTCOME),
        ("direction", DIRECTION_OUTCOME),
        ("intervention", CAUSAL_OUTCOME),
    ):
        print("  " + _outcome.line(_name))
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
