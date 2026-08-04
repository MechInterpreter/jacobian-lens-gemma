# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Generate ``notebooks/multimodal_jspace_layer_localization_colab.ipynb``.

Written from source rather than edited as JSON, so the committed notebook stays
output-free and byte-reproducible. Run
``python scripts/_build_localization_notebook.py`` after changing a cell; a test
regenerates it and fails on drift.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TARGET = (
    REPO_ROOT / "notebooks" / "multimodal_jspace_layer_localization_colab.ipynb"
)

CELLS: list[tuple[str, str]] = []


def markdown(text: str) -> None:
    CELLS.append(("markdown", text.strip("\n")))


def code(text: str) -> None:
    CELLS.append(("code", text.strip("\n")))


markdown(
    """
# Layer localization — where does the text↔image transfer first appear?

**One question.** The six-concept robustness study returned `ROBUSTNESS_GO` at
physical decoder layer 38 and earned exactly one claim: *a frozen
text-calibrated J-lens at late layer 38 exposes concept representations that
causally transfer between written text and images.* Layer 38 sits at ~90% of
this model's depth. **Does that transfer exist earlier, or only near
answer-language convergence?**

Four predetermined physical layers — **20, 26, 32, 38** (~normalized 48, 62, 76,
90) — two concepts that already replicated bidirectionally (**cat** and
**toilet**), off-diagonal cells only, one frozen lens, never refitted.

**This is not another robustness study and not a framework.** It adds no
concepts, no modalities, no layers, and no lens. Everything that would make it
bigger is out, because none of it is what "how early?" needs.

## Three things that decide whether this means anything

**Eligibility is earned, not declared.** An earlier layer does not become
testable by appearing in `LAYERS`. It must pass an independently specified,
tie-aware, text-only lens-validity gate (section 10) before any causal claim may
rest on it. A layer that fails keeps its diagnostic numbers and is **skipped
causally** — and that skip is not a negative result, because nothing was tested.

**Layer 38 is the anchor, not a competitor.** If the established result does not
reproduce on this subset, no earlier layer's number means anything: there would
be nothing to be earlier *than*. That case is `INCONCLUSIVE_LAYER_LOCALIZATION`.

**The targets are frozen before the first layer is scored** (section 6), and the
same photographs are used at every layer, so a difference between layers is a
fact about depth rather than about which pictures each layer happened to get.

**Nothing starts by itself.** `RUN_REAL_LOCALIZATION`, `RUN_MODEL_STAGES` and
`CONFIRM_MODEL_PASS_BUDGET` are all False in the committed notebook and all must
be set by hand. `RUN_TEXT_RECALIBRATION` is separate again, and even when set it
fits nothing here.

## What this cannot tell you

- It reports the **earliest tested layer with evidence**, never the earliest
  layer in the model. A layer between two tested ones is untested; anything
  shallower than 20 is unexamined.
- Localization is **conditioned on cat and toilet**, the two concepts already
  known to transfer. This is not an estimate of how many concepts transfer.
- Interventions add and subtract a direction on the residual stream at the final
  prompt token. That is **not erasure** and not **projection ablation**.
- Written text and images only. Spoken audio is excluded by design and
  environmental audio is not tested; neither absence is evidence about either.
"""
)

markdown(
    """
## 1. Bootstrap repository

Run these three cells first, in order. They use nothing but the standard
library: the repository is not importable until 1c has installed it, so anything
that says `from jlens...` before then would fail with `ModuleNotFoundError`.

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
REPO_PATH = Path(os.environ.get("MMLOCALIZE_REPO_DIR") or REPO_DIR)


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
| `RUN_REAL_LOCALIZATION` | use the real Drive dataset, the real completed run, and the real lens instead of the deterministic MOCK world |
| `RUN_MODEL_STAGES` | allow Gemma to be loaded at all |
| `CONFIRM_MODEL_PASS_BUDGET` | acknowledge the printed forward-pass budget |
| `RUN_TEXT_RECALIBRATION` | **leave False for the first run.** Prints the bounded recalibration plan; it never fits a lens here |

**The default first real run evaluates the existing frozen v2 artifact.**
Recalibration is a separate, explicit choice and cannot happen as a side effect
of anything below.

The design is fixed here and is not adjustable from results: four physical
layers, two concepts, four distinct images per role per concept, three alphas,
four control kinds.
"""
)

code(
    '''
# 2. Configuration. Requires section 1 (it imports from the repository).
# Nothing here mounts Drive, reads data, or loads a model.
RUN_REAL_LOCALIZATION = False
RUN_MODEL_STAGES = False
RUN_TEXT_RECALIBRATION = False

# Set to True only after reading the budget printed in section 7.
CONFIRM_MODEL_PASS_BUDGET = False

import json

from jlens.mmlocalize.layers import (
    LAYER_SET_VERSION,
    LOCALIZATION_LAYERS,
    MODEL_N_LAYERS,
    REFERENCE_LAYER,
    assert_immutable_layer_set,
    describe_layers,
    layer_manifest,
)
from jlens.mmlocalize.lens_validity import (
    CONTROL_SEED,
    CONTROL_VARIANTS,
    DIAGNOSTIC_VARIANTS,
    LOCALIZATION_VALIDITY_GATE,
    N_VALIDATION_PROMPTS,
    READOUT_VARIANTS,
    VALIDATION_PROMPT_SEED,
    VALIDITY_PROTOCOL,
    gate_text,
)
from jlens.mmlocalize.targets import (
    CONCEPT_CONDITIONING_LIMITATION,
    LOCALIZATION_CONCEPTS,
    N_SOURCE_NEGATIVE_IMAGES,
    N_SOURCE_POSITIVE_IMAGES,
    N_TARGET_NEGATIVE_IMAGES,
    N_TARGET_POSITIVE_IMAGES,
)
from jlens.mmlocalize.verdict import (
    LOCALIZATION_VERDICT_VERSION,
    LocalizationThresholds,
)
from jlens.mmpilot.pipeline import PilotConfig
from jlens.mmpilot.selection import IMAGE_UNIQUE_PROFILE

# ------------------------------------------------------------------ design
# The layer list is immutable. Editing it here fails immediately rather than
# quietly answering a different question.
LAYERS = (20, 26, 32, 38)
assert_immutable_layer_set(LAYERS)
CONCEPTS = LOCALIZATION_CONCEPTS          # ("cat", "toilet"), fixed
ALPHAS = (0.0, 0.25, 0.5)
CAPABILITY_THRESHOLD = 0.7
SPLIT_SEED = "spokencoco-localization-v1"

# ---------------------------------------------------------------- the lens
# Frozen, previously validated, never refitted here. Layer 38 is certified by
# its manifest; 20, 26 and 32 are fitted but NOT certified — that is what
# section 10 tests.
LENS_PATH = (
    "/content/drive/MyDrive/jacobian-lens-gemma/runs/"
    "text_jlens_early_layer_recalibration_v2/artifacts/lens.validated.pt"
)
LENS_EXPECT_SHA256 = (
    "sha256:4b17bf6086901e633f94d3391f5de6eccd3e735cc24cece63887505d73641c2b"
)
V2_RUN_DIR = (
    "/content/drive/MyDrive/jacobian-lens-gemma/runs/"
    "text_jlens_early_layer_recalibration_v2"
)
MODEL_REPO_ID = "google/gemma-4-E4B-it"
MODEL_REVISION = "fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd"
EXPECT_N_LAYERS, EXPECT_D_MODEL, EXPECT_VOCAB = 42, 2560, 262144

# The completed six-concept robustness run. Read-only, verified by fingerprint.
COMPLETED_ROBUSTNESS_RUN_DIR = (
    "/content/drive/MyDrive/jacobian-lens-gemma/runs/"
    "mmrobust_robustness_20260804T154417"
)
COMPLETED_ROBUSTNESS_FINGERPRINT = (
    "sha256:61d0f0e7eb0e2b75831817fa7b9a7f4ebb36d7f4d03fbebce669634390c4c278"
)

# --------------------------------------------------------------- the data
SPOKENCOCO_BASE_ROOT = "/content/drive/MyDrive/datasets/cstf_spokencoco"
IMAGE_MEDIA_ROOT = "/content/drive/MyDrive/datasets/cstf_spokencoco/coco"
AUDIO_MEDIA_ROOT = "/content/drive/MyDrive/datasets/cstf_spokencoco/SpokenCOCO"
DOWNLOAD_CACHE = "/content/drive/MyDrive/datasets/cstf_spokencoco_download_cache"
MANIFEST_PATH = "/content/drive/MyDrive/datasets/spokencoco_manifest.json"
RUNS_ROOT = "/content/drive/MyDrive/jacobian-lens-gemma/runs"

# Candidate pool for the held-out text prompts. The v2 run consumed the first 40
# WikiText records; a wider pool leaves the seed a real choice after those and
# any layer-32 confirmatory prompts are excluded by hash.
POOL_RECORDS = 320

SCRATCH = Path(os.environ.get("MMLOCALIZE_SCRATCH") or "/content/mmlocalize_scratch")
SCRATCH.mkdir(parents=True, exist_ok=True)
RESOLVED_RUNS_ROOT = Path(
    os.environ.get("MMLOCALIZE_RUNS_ROOT")
    or (RUNS_ROOT if RUN_REAL_LOCALIZATION else SCRATCH / "runs")
)
COMPLETED_ROBUSTNESS_RUN_DIR = (
    os.environ.get("MMLOCALIZE_COMPLETED_RUN_DIR") or COMPLETED_ROBUSTNESS_RUN_DIR
)

MODALITIES = ("text", "image")
PROFILE = IMAGE_UNIQUE_PROFILE
THRESHOLDS = LocalizationThresholds(
    concepts=tuple(CONCEPTS),
    required_positive_images_per_cell=N_TARGET_POSITIVE_IMAGES,
    required_negative_images_per_cell=N_TARGET_NEGATIVE_IMAGES,
)

CONFIG = PilotConfig(
    mode="localization" if RUN_REAL_LOCALIZATION else "mock",
    layers=tuple(LAYERS),
    # Filled in after section 10 decides which layers earned a causal claim.
    causal_layers=(),
    modalities=MODALITIES,
    causal_concepts=tuple(CONCEPTS),
    capability_threshold=CAPABILITY_THRESHOLD,
    alphas=tuple(ALPHAS),
    n_target_examples=N_TARGET_POSITIVE_IMAGES,
    pursuit_k=25 if RUN_REAL_LOCALIZATION else 8,
    pursuit_correlation_chunk_size=65536 if RUN_REAL_LOCALIZATION else None,
    direction_top_k=16 if RUN_REAL_LOCALIZATION else 4,
    n_permutations=50 if RUN_REAL_LOCALIZATION else 8,
    max_capability_groups_per_concept=N_SOURCE_POSITIVE_IMAGES
    + N_TARGET_POSITIVE_IMAGES,
    seed=20260804,
    subset_profile="image_unique",
    image_unique_targets=True,
    min_source_positive_images=N_SOURCE_POSITIVE_IMAGES,
    min_source_negative_images=N_SOURCE_NEGATIVE_IMAGES,
    off_diagonal_causal_only=True,
)

print(f"RUN_REAL_LOCALIZATION     = {RUN_REAL_LOCALIZATION}")
print(f"RUN_MODEL_STAGES          = {RUN_MODEL_STAGES}")
print(f"CONFIRM_MODEL_PASS_BUDGET = {CONFIRM_MODEL_PASS_BUDGET}")
print(f"RUN_TEXT_RECALIBRATION    = {RUN_TEXT_RECALIBRATION}  (fits nothing here)")
print()
print(describe_layers(LAYERS))
print()
print(f"mode        {CONFIG.mode}")
print(f"concepts    {list(CONCEPTS)}  (fixed; conditioned, not sampled)")
print(f"modalities  {list(MODALITIES)}")
print(f"alphas      {list(CONFIG.alphas)}")
print(f"images      source +{N_SOURCE_POSITIVE_IMAGES}/-{N_SOURCE_NEGATIVE_IMAGES}, "
      f"target +{N_TARGET_POSITIVE_IMAGES}/-{N_TARGET_NEGATIVE_IMAGES} per concept")
print(f"validity    {VALIDITY_PROTOCOL}")
print(f"verdict     {LOCALIZATION_VERDICT_VERSION}")
print()
print(CONCEPT_CONDITIONING_LIMITATION)
print()
print("This notebook does not fit a lens.")
'''
)

markdown(
    """
### 2b. The predeclared layer-validity gate

Printed **before any result-producing cell runs**, and bound into the resume
fingerprint by its digest so editing it invalidates stored results rather than
rescoring them.

Read the "why this differs from the old gate" section carefully. Layer 32's v2
failure — median rank 1, MRR ~0.71, but *zero* unique top-1 agreements — was one
fact reported twice: both quantities were being read off a tie block at the
maximum, where `argmax` reports the tie-break rule and a top-10 slice is an
arbitrary ten of the tied tokens. The new gate **adds** three blocking clauses
(a tied-at-maximum ceiling, a wrong-layer MRR margin, and fold stability) and
drops only the unique-top-1 floor, which under ties is not a property of the
lens. Both gates are computed and both are printed.
"""
)

code(
    '''
# 2b. Print the criterion before anything can be computed under it.
print(gate_text())
print()
print(f"gate digest {LOCALIZATION_VALIDITY_GATE.digest}")
print(f"controls    {list(CONTROL_VARIANTS)} (seed {CONTROL_SEED})")
print(f"diagnostic  {list(DIAGNOSTIC_VARIANTS)} — reported, never blocking")
print(f"prompts     {N_VALIDATION_PROMPTS} held out, seed {VALIDATION_PROMPT_SEED}")
'''
)

markdown(
    """
## 3. Mount Google Drive

Read-only checks. This cell never creates, moves, or deletes anything inside the
dataset, and never touches the completed robustness run except to read from it.
"""
)

code(
    '''
# 3. Mount Drive and verify the configured paths exist.
if RUN_REAL_LOCALIZATION and IN_COLAB:
    from google.colab import drive

    drive.mount("/content/drive", force_remount=False)

if RUN_REAL_LOCALIZATION:
    missing = [
        path
        for path in (
            SPOKENCOCO_BASE_ROOT,
            IMAGE_MEDIA_ROOT,
            MANIFEST_PATH,
            LENS_PATH,
            V2_RUN_DIR,
            COMPLETED_ROBUSTNESS_RUN_DIR,
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
    print("skipped: RUN_REAL_LOCALIZATION is False (the MOCK world needs no Drive)")
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
    command = [sys.executable, "-m", "pip", "install", "-q", "pillow", "soundfile"]
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
## 5. Verify the completed robustness run and reuse its evidence (CPU only)

The completed run is verified **by fingerprint** and then only read. A mismatch
refuses rather than proceeding: a localization conditioned on "some run in that
directory" would inherit a provenance nobody can state.

What is reused, read-only: the derived metadata join (the expanded manifest),
the image identities, and the split provenance. What is **not** reused: any
target image, under the fresh policy — that is the point of section 6.
"""
)

code(
    '''
# 5. CPU ONLY. Verify the completed run, collect the images it touched, and
# load the derived evidence cache. No model, no GPU, no writes into that run.
from datetime import datetime, timezone

from jlens.mmlocalize.targets import completed_run_images, verify_completed_run
from jlens.mmpilot import evidence as evidence_module
from jlens.mmpilot import expansion as expansion_module
from jlens.mmpilot import manifest as manifest_module
from jlens.mmpilot.concepts import discover_category_universe

if not RUN_REAL_LOCALIZATION:
    # The synthetic world. Six concepts so the candidate set and the external
    # unrelated controls match the completed run's protocol; cat and toilet are
    # the localization pair.
    from jlens.mmpilot.mock import MockWorld, build_mock_dataset

    MOCK_CONCEPTS_6 = {
        "cat": ("cat", "cats"),
        "toilet": ("toilet", "toilets"),
        "bus": ("bus", "buses"),
        "clock": ("clock", "clocks"),
        "dog": ("dog", "dogs"),
        "pizza": ("pizza", "pizzas"),
    }
    MOCK_WORLD = MockWorld(MOCK_CONCEPTS_6)
    if not (SCRATCH / "data" / "spokencoco_manifest.json").is_file():
        build_mock_dataset(
            SCRATCH / "data",
            world=MOCK_WORLD,
            images_per_concept=(N_SOURCE_POSITIVE_IMAGES + N_TARGET_POSITIVE_IMAGES) * 2,
            negative_images=(N_SOURCE_NEGATIVE_IMAGES + N_TARGET_NEGATIVE_IMAGES) * 4,
            captions_per_image=2,
            layout="sibling",
            visual_only_images=1,
        )
    MANIFEST_PATH = str(SCRATCH / "data" / "spokencoco_manifest.json")
    IMAGE_MEDIA_ROOT = str(SCRATCH / "data" / "coco")
    AUDIO_MEDIA_ROOT = str(SCRATCH / "data" / "SpokenCOCO")
    print(f"MOCK dataset ready at {SCRATCH / 'data'}")

# ------------------------------------------- the completed run, verified first
if RUN_REAL_LOCALIZATION:
    COMPLETED_RUN = verify_completed_run(
        COMPLETED_ROBUSTNESS_RUN_DIR,
        expect_fingerprint=COMPLETED_ROBUSTNESS_FINGERPRINT,
    )
    COMPLETED_IMAGES = completed_run_images(COMPLETED_ROBUSTNESS_RUN_DIR)
    print(f"completed run verified: {COMPLETED_RUN['run_dir']}")
    print(f"  fingerprint {COMPLETED_RUN['fingerprint']}")
    print(f"  verdict     {COMPLETED_RUN['verdict']}")
    print(f"  artifacts   {len(COMPLETED_RUN['artifact_checksums'])} checksummed")
else:
    COMPLETED_RUN = {
        "run_dir": "mock://no-completed-run",
        "fingerprint": "sha256:mock-completed-robustness-run",
        "fingerprint_matches_pin": True,
        "verdict": "ROBUSTNESS_GO",
        "artifact_checksums": {},
        "read_only": True,
    }
    COMPLETED_IMAGES = {
        "run_dir": COMPLETED_RUN["run_dir"],
        "causal_target_images": [],
        "all_images": [],
        "n_causal_target_images": 0,
        "n_all_images": 0,
    }
    print("MOCK: no completed run to verify; a synthetic record stands in")

print(
    f"images the completed run touched: {COMPLETED_IMAGES['n_all_images']} "
    f"({COMPLETED_IMAGES['n_causal_target_images']} were its causal targets)"
)

# --------------------------------------------------------- the derived cache
MANIFEST_CHECKSUM = manifest_module.manifest_checksum(MANIFEST_PATH)
IMAGE_ROOTS = [Path(IMAGE_MEDIA_ROOT)]
AUDIO_ROOTS = [Path(AUDIO_MEDIA_ROOT)]

RUN_ID = (
    f"mmlocalize_{CONFIG.mode}_"
    f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
)
RUN_DIR = Path(os.environ.get("MMLOCALIZE_RUN_DIR") or (RESOLVED_RUNS_ROOT / RUN_ID))
RUN_DIR.mkdir(parents=True, exist_ok=True)
if Path(RUN_DIR).resolve() == Path(COMPLETED_ROBUSTNESS_RUN_DIR).resolve():
    raise RuntimeError(
        "the run directory is the completed robustness run; that run is "
        "read-only and is never written into"
    )
print(f"\\nrun directory {RUN_DIR}")

SEARCH_ROOTS = sorted({str(root) for root in IMAGE_ROOTS + AUDIO_ROOTS if root.is_dir()})
if RUN_REAL_LOCALIZATION:
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

REUSED_EXPANDED_MANIFEST = None
COMPLETED_EXPANDED = Path(COMPLETED_ROBUSTNESS_RUN_DIR) / "expanded_manifest.json"
if COMPLETED_EXPANDED.is_file():
    _stored = json.loads(COMPLETED_EXPANDED.read_text(encoding="utf-8"))
    if _stored.get("original_manifest_checksum") == MANIFEST_CHECKSUM and _stored.get(
        "groups"
    ):
        REUSED_EXPANDED_MANIFEST = _stored
        print(f"reusing the completed run's expanded manifest: {COMPLETED_EXPANDED}")

if REUSED_EXPANDED_MANIFEST is not None:
    ALL_GROUPS = REUSED_EXPANDED_MANIFEST["groups"]
    EXPANSION_STATUS = "reused: derived join loaded from the completed robustness run"
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
    ALL_GROUPS = EXPANSION.groups
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
print(f"synchronized groups available: {len(ALL_GROUPS)}")
print(f"distinct images available:     {len({g['image_id'] for g in ALL_GROUPS})}")
'''
)

markdown(
    """
## 6. Freeze the localization targets (CPU only)

**This is the cell that has to run before any layer is scored.**

The target policy is decided from availability alone. Preferred:
`fresh_image_disjoint_from_completed_run` — photographs the completed robustness
run never touched, which buys an independent test rather than a re-reading of
the same images. Fallback when the cache cannot supply enough fresh validated
groups: `reused_completed_run_images_paired_within_sample`, which is a **paired
within-sample comparison** and is labelled as one everywhere it appears. The two
are never mixed.

The frozen set is checksummed, and the same photographs are used at every layer.
An exact image-exclusion audit is written beside the manifest.
"""
)

code(
    '''
# 6. CPU ONLY. Decide the target policy from availability, build the
# one-group-per-image subset, freeze the targets, and audit the exclusions.
from jlens.mmlocalize.targets import (
    POLICY_FRESH_DISJOINT,
    assert_same_targets_across_layers,
    audit_image_exclusions,
    choose_target_policy,
    format_targets,
    freeze_targets,
    target_manifest,
)
from jlens.mmpilot.selection import select_focal_concepts, unrelated_control_assignment
from jlens.mmpilot.store import payload_checksum

EXCLUDED_IMAGES = set(COMPLETED_IMAGES["all_images"])
FRESH_GROUPS = [g for g in ALL_GROUPS if str(g["image_id"]) not in EXCLUDED_IMAGES]

EVIDENCE_INDEX_FRESH = evidence_module.build_evidence_index(
    FRESH_GROUPS, tuple(CONCEPT_CANDIDATES), EVIDENCE_CONFIG
)
_fresh_positive_images = {
    concept: len(
        {
            g["image_id"]
            for g in FRESH_GROUPS
            if EVIDENCE_INDEX_FRESH.concept_of(g, concept)
        }
    )
    if hasattr(EVIDENCE_INDEX_FRESH, "concept_of")
    else 0
    for concept in CONCEPTS
}
# The evidence index API differs by version; fall back to the ranking, which is
# the same computation the subset builder performs.
_fresh_ranking = expansion_module.rank_concepts(
    FRESH_GROUPS,
    {name: CONCEPT_CANDIDATES[name] for name in CONCEPTS if name in CONCEPT_CANDIDATES}
    or CONCEPT_CANDIDATES,
    requirements=expansion_module.ConceptRequirements(
        min_distinct_images=N_SOURCE_POSITIVE_IMAGES + N_TARGET_POSITIVE_IMAGES,
        min_groups=N_SOURCE_POSITIVE_IMAGES + N_TARGET_POSITIVE_IMAGES,
        min_train_positives=N_SOURCE_POSITIVE_IMAGES,
        min_test_positives=N_TARGET_POSITIVE_IMAGES,
    ),
    groups_per_concept=N_SOURCE_POSITIVE_IMAGES + N_TARGET_POSITIVE_IMAGES,
    max_groups_per_image=PROFILE.max_groups_per_image,
    seed=SPLIT_SEED,
    evidence_config=EVIDENCE_CONFIG,
    profile=PROFILE,
    evidence_index=EVIDENCE_INDEX_FRESH,
)
_by_concept = {row["concept"]: row for row in _fresh_ranking}
FRESH_AVAILABLE = {
    concept: int(_by_concept.get(concept, {}).get("n_distinct_images", 0))
    for concept in CONCEPTS
}
FRESH_NEGATIVES_AVAILABLE = len(
    {
        g["image_id"]
        for g in FRESH_GROUPS
        if not any(
            _by_concept.get(c, {}).get("concept") == c and g["image_id"] in set()
            for c in CONCEPTS
        )
    }
)

POLICY_DECISION = choose_target_policy(
    n_available_fresh_images=FRESH_AVAILABLE,
    n_available_fresh_negatives=FRESH_NEGATIVES_AVAILABLE,
    concepts=CONCEPTS,
)
TARGET_POLICY = POLICY_DECISION["policy"]
print("=" * 72)
print("TARGET POLICY — decided from availability, before any layer result")
print("=" * 72)
print(f"  policy              {TARGET_POLICY}")
print(f"  fresh feasible      {POLICY_DECISION['fresh_targets_feasible']}")
print(f"  per-concept fresh   {POLICY_DECISION['per_concept_available']} "
      f"(need {POLICY_DECISION['required_positive_images_per_concept']} each)")
print(f"  fresh negatives     {POLICY_DECISION['available_negative_images']} "
      f"(need {POLICY_DECISION['required_negative_images']})")
if POLICY_DECISION["limitation"]:
    print()
    print("  " + POLICY_DECISION["limitation"])

# The pool the subset is built from follows the policy, and only the policy.
SUBSET_GROUPS = FRESH_GROUPS if TARGET_POLICY == POLICY_FRESH_DISJOINT else ALL_GROUPS
EVIDENCE_INDEX = evidence_module.build_evidence_index(
    SUBSET_GROUPS, tuple(CONCEPT_CANDIDATES), EVIDENCE_CONFIG
)

# Candidate scoring keeps the completed run's SIX-WAY protocol, so a capability
# number here means what it meant there — a six-way forced choice and a two-way
# one are not the same measurement. The six concepts are read out of the
# completed run rather than re-derived, because re-deriving them could quietly
# produce a different set and a different question.
COMPLETED_CONCEPTS = []
_completed_summary = Path(COMPLETED_ROBUSTNESS_RUN_DIR) / "robustness_summary.json"
if _completed_summary.is_file():
    _payload = json.loads(_completed_summary.read_text(encoding="utf-8"))
    COMPLETED_CONCEPTS = list(
        (_payload.get("verdict") or {}).get("selected_concepts")
        or (_payload.get("split_provenance") or {}).get("selected_concepts")
        or []
    )
if COMPLETED_CONCEPTS:
    _missing = [c for c in CONCEPTS if c not in COMPLETED_CONCEPTS]
    if _missing:
        raise RuntimeError(
            f"{_missing} are not among the completed run's concepts "
            f"{COMPLETED_CONCEPTS}; this study localizes that run's result and "
            "cannot substitute a concept it never established"
        )
    # cat and toilet first (they are the focal pair), the rest in the completed
    # run's own ranking order.
    SELECTED_NAMES = list(CONCEPTS) + [
        name for name in COMPLETED_CONCEPTS if name not in CONCEPTS
    ]
elif RUN_REAL_LOCALIZATION:
    raise RuntimeError(
        f"{_completed_summary} does not record the completed run's selected "
        "concepts; the six-way candidate protocol cannot be reproduced without "
        "them, and a narrower forced choice would not be comparable"
    )
else:
    SELECTED_NAMES = list(CONCEPTS) + [
        name for name in sorted(CONCEPT_CANDIDATES) if name not in CONCEPTS
    ][:4]
FOCAL_CONCEPTS, NON_FOCAL_CONCEPTS = select_focal_concepts(
    SELECTED_NAMES, n_focal=len(CONCEPTS)
)
UNRELATED_CONTROLS = unrelated_control_assignment(FOCAL_CONCEPTS, NON_FOCAL_CONCEPTS)
CONFIG.concepts = tuple(SELECTED_NAMES)
CONFIG.causal_concepts = tuple(FOCAL_CONCEPTS)
print(f"\\n  candidates scored   {SELECTED_NAMES}")
print(f"  focal (localized)   {FOCAL_CONCEPTS}")
print(f"  unrelated controls  {dict(sorted(UNRELATED_CONTROLS.items()))}")

# The external unrelated-concept control is a DIRECTION, so its concept needs
# source-training photographs of its own — without them the control is silently
# absent and every cell fails for a missing control rather than on its evidence.
# These concepts contribute training images only; they are never causal targets.
DIRECTION_CONCEPTS = list(CONCEPTS) + sorted(
    {control for control in UNRELATED_CONTROLS.values() if control not in CONCEPTS}
)
print(f"  direction concepts  {DIRECTION_CONCEPTS} "
      f"(the last {len(DIRECTION_CONCEPTS) - len(CONCEPTS)} supply unrelated "
      "controls and are never targets)")

SUBSET = manifest_module.build_subset(
    SUBSET_GROUPS,
    {name: CONCEPT_CANDIDATES[name] for name in DIRECTION_CONCEPTS},
    groups_per_concept=N_SOURCE_POSITIVE_IMAGES + N_TARGET_POSITIVE_IMAGES,
    negatives_per_concept=N_SOURCE_NEGATIVE_IMAGES + N_TARGET_NEGATIVE_IMAGES,
    seed=SPLIT_SEED,
    evidence_config=EVIDENCE_CONFIG,
    profile=PROFILE,
    evidence_index=EVIDENCE_INDEX,
)
LEAKAGE = manifest_module.check_split_leakage(SUBSET)
if not LEAKAGE["ok"]:
    raise RuntimeError(f"split leakage detected, refusing to continue: {LEAKAGE}")

_train, _test = SUBSET["splits"]["train"], SUBSET["splits"]["test"]
_all_rows = _train + _test
N_TOTAL_GROUPS = len(_all_rows)
N_DISTINCT_IMAGES = len({row["image_id"] for row in _all_rows})

TARGETS = freeze_targets(
    policy=TARGET_POLICY,
    source_positive_images={
        c: sorted({r["image_id"] for r in _train if r["concept"] == c}) for c in CONCEPTS
    },
    source_negative_images={
        c: sorted({r["image_id"] for r in _train if not r["concept"]}) for c in CONCEPTS
    },
    target_positive_images={
        c: sorted({r["image_id"] for r in _test if r["concept"] == c}) for c in CONCEPTS
    },
    target_negative_images={
        c: sorted({r["image_id"] for r in _test if not r["concept"]}) for c in CONCEPTS
    },
    completed_run_images=COMPLETED_IMAGES["all_images"],
    concepts=CONCEPTS,
)
EXCLUSION_AUDIT = audit_image_exclusions(
    TARGETS, completed_run=COMPLETED_IMAGES, n_available_images=len(
        {g["image_id"] for g in ALL_GROUPS}
    )
)
TARGET_MANIFEST = target_manifest(
    TARGETS, audit=EXCLUSION_AUDIT, completed_run=COMPLETED_RUN, layers=LAYERS
)
(RUN_DIR / "localization_target_manifest.json").write_text(
    json.dumps(TARGET_MANIFEST, indent=2, default=str), encoding="utf-8"
)
(RUN_DIR / "image_exclusion_audit.json").write_text(
    json.dumps(EXCLUSION_AUDIT, indent=2, default=str), encoding="utf-8"
)

print()
print(format_targets(TARGETS, EXCLUSION_AUDIT))
print()
print(f"  synchronized groups {N_TOTAL_GROUPS} over {N_DISTINCT_IMAGES} distinct images")
print(f"  split leakage check {LEAKAGE['ok']}")
print(f"  manifest checksum   {TARGET_MANIFEST['manifest_checksum']}")

SPLIT_PROVENANCE = {
    "seed": SPLIT_SEED,
    "profile": PROFILE.to_dict(),
    "concepts": list(CONCEPTS),
    "candidates_scored": list(SELECTED_NAMES),
    "unrelated_controls": dict(sorted(UNRELATED_CONTROLS.items())),
    "target_policy": TARGET_POLICY,
    "target_checksum": TARGETS.checksum,
    "n_groups": N_TOTAL_GROUPS,
    "n_distinct_images": N_DISTINCT_IMAGES,
    "leakage": LEAKAGE,
}
(RUN_DIR / "split_provenance.json").write_text(
    json.dumps(SPLIT_PROVENANCE, indent=2, default=str), encoding="utf-8"
)
SPLIT_PROVENANCE_CHECKSUM = payload_checksum(SPLIT_PROVENANCE)
print(f"  split provenance    {SPLIT_PROVENANCE_CHECKSUM}")
'''
)

markdown(
    """
## 7. Pass and storage estimate — confirm before spending anything

Every number below is derived from the configuration, not guessed.

Two costs are worth reading. **Activation capture does not multiply by layers**:
one forward pass records all four layers' residuals, which is why a four-layer
diagnostic is affordable at all. **Causal cost does multiply by eligible
layers**, and eligibility is not known until section 10 — so the figure printed
here assumes the worst case, all four layers eligible. Section 16 reprints the
actual cost once Stage B has decided.

Read them, then set `CONFIRM_MODEL_PASS_BUDGET = True` in section 2 and re-run
from there.
"""
)

code(
    '''
# 7. CPU ONLY. Print the exact budget (worst case on eligibility) and refuse to
# continue without an explicit, separate confirmation.
from jlens.mmlocalize.lens_validity import (
    RECALIBRATION_PLAN,
    check_recalibration_target,
    format_recalibration_plan,
)
from jlens.mmlocalize.verdict import estimate_localization_passes, format_budget

N_CAPABILITY_GROUPS = min(
    len(DIRECTION_CONCEPTS) * CONFIG.max_capability_groups_per_concept,
    len([row for row in _all_rows if row["concept"]]),
)
BUDGET = estimate_localization_passes(
    n_concepts=len(SELECTED_NAMES),
    modalities=MODALITIES,
    n_total_groups=N_TOTAL_GROUPS,
    n_capability_groups=N_CAPABILITY_GROUPS,
    n_layers_captured=len(LAYERS),
    n_eligible_causal_layers=len(LAYERS),   # worst case, until section 10 decides
    n_targets_per_cell=N_TARGET_POSITIVE_IMAGES + N_TARGET_NEGATIVE_IMAGES,
    alphas=CONFIG.alphas,
    n_validation_prompts=N_VALIDATION_PROMPTS,
    recalibration_enabled=RUN_TEXT_RECALIBRATION,
)
print(format_budget(BUDGET))
print()
print("  the causal figures above assume ALL FOUR layers pass section 10's gate.")
print("  section 16 reprints the actual cost once eligibility is known.")
(RUN_DIR / "pass_budget.json").write_text(
    json.dumps(BUDGET.to_dict(), indent=2, default=str), encoding="utf-8"
)

# ------------------------------------------------- the recalibration decision
print()
if RUN_TEXT_RECALIBRATION:
    print(format_recalibration_plan())
    RECALIBRATION_CHECK = check_recalibration_target(LENS_PATH)
    print()
    print("  RUN_TEXT_RECALIBRATION is set, but this notebook still fits nothing.")
    print("  Run the plan above in its own notebook, freeze and checksum the new")
    print("  artifact, then point LENS_PATH and LENS_EXPECT_SHA256 at it.")
else:
    RECALIBRATION_CHECK = None
    print("recalibration: DISABLED — the frozen v2 artifact is evaluated as it is.")
    print(f"  if section 10 shows it is underpowered, the bounded plan is "
          f"{RECALIBRATION_PLAN['n_fitting_prompts']} text-only prompts at layers "
          f"{RECALIBRATION_PLAN['layers']},")
    print(f"  published to a NEW path; {RECALIBRATION_PLAN['never_overwrite']}")
    print("  is never overwritten.")

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
    print("    RUN_REAL_LOCALIZATION     = True   # for the real dataset and lens")
    print()
    print("then re-run from section 2. Nothing below loads a model until all")
    print("three are set.")
else:
    print("budget confirmed by CONFIRM_MODEL_PASS_BUDGET")

MODEL_STAGES_ENABLED = bool(RUN_MODEL_STAGES and BUDGET_CONFIRMED)
print(f"MODEL_STAGES_ENABLED = {MODEL_STAGES_ENABLED}")
'''
)

markdown(
    """
## 8. Load and audit Gemma

Nothing below this point runs unless `RUN_MODEL_STAGES` **and**
`CONFIRM_MODEL_PASS_BUDGET` are both True.

**8a runs the real-path preflight before the 16 GB download.** Without loading
any weights it verifies: the layer list is the predetermined one; the lens is on
disk, matches its pin, declares a text-only calibration, names this model
revision, and has a **fitted Jacobian at every requested layer**; layer 38 is
certified by its manifest, so there is an anchor; the revision resolves on the
Hub; every call the real path will make binds against the installed signatures;
the completed run is verified; the target manifest is frozen, complete and
image-disjoint; the run fingerprint is complete; and Drive has room.

It prints `REAL PATH PREFLIGHT: PASS` or fails listing **every** problem at once.

**8b reproduces the tested load sequence** via
`jlens.mmpilot.real_backend.build_real_backend`, then runs the **invariance
gate** at every layer: activation capture must be a no-op and a zero-coefficient
intervention must not change the logits.
"""
)

code(
    '''
# 8a. REAL PATH PREFLIGHT — must pass before the 16 GB download may start.
PREFLIGHT = None
FINGERPRINT_FIELDS = {
    "model_repo_id": MODEL_REPO_ID if RUN_REAL_LOCALIZATION else "mock/gemma-like",
    "model_revision": MODEL_REVISION if RUN_REAL_LOCALIZATION else "mock",
    "processor_revision": MODEL_REVISION if RUN_REAL_LOCALIZATION else "mock",
    "lens_checksum": LENS_EXPECT_SHA256,
    "calibration_protocol": "text-only WikiText chat rendering (v2)",
    "layers": list(LAYERS),
    "validity_gate_digest": LOCALIZATION_VALIDITY_GATE.digest,
    "manifest_checksum": MANIFEST_CHECKSUM,
    "target_checksum": TARGETS.checksum,
    "source_image_ids": TARGETS.all_source_images(),
    "target_image_ids": TARGETS.all_target_images(),
    "concepts": list(CONCEPTS),
    "prompt_protocol": "canonical_and_reversed_sorted_options.v1",
    "alphas": list(CONFIG.alphas),
    "controls": list(CONTROL_VARIANTS),
    "pursuit_config": {
        "k": CONFIG.pursuit_k,
        "refine_steps": CONFIG.pursuit_refine_steps,
        "direction_top_k": CONFIG.direction_top_k,
    },
}

if not MODEL_STAGES_ENABLED:
    print("skipped: MODEL_STAGES_ENABLED is False")
elif not RUN_REAL_LOCALIZATION:
    from jlens.mmlocalize.real_path import check_localization_call_contracts

    _contract_failures = check_localization_call_contracts()
    if _contract_failures:
        raise RuntimeError(
            "call-signature contracts failed even in MOCK:\\n  - "
            + "\\n  - ".join(_contract_failures)
        )
    print(
        "MOCK path: full preflight needs the real lens and the Hub; the "
        "call-signature contracts were checked and all bind."
    )
else:
    import getpass

    from jlens.mmlocalize.real_path import format_preflight, localization_preflight

    if not os.environ.get("HF_TOKEN"):
        _token = getpass.getpass("HF_TOKEN (input hidden): ").strip()
        if not _token:
            raise RuntimeError("a Hugging Face token is required for the gated repo")
        os.environ["HF_TOKEN"] = _token

    PREFLIGHT = localization_preflight(
        model_repo_id=MODEL_REPO_ID,
        model_revision=MODEL_REVISION,
        lens_path=LENS_PATH,
        lens_expect_checksum=LENS_EXPECT_SHA256,
        layers=LAYERS,
        expect_d_model=EXPECT_D_MODEL,
        expect_n_layers=EXPECT_N_LAYERS,
        concepts=CONCEPTS,
        target_manifest=TARGET_MANIFEST,
        completed_run=COMPLETED_RUN,
        fingerprint_fields=FINGERPRINT_FIELDS,
        runs_root=RESOLVED_RUNS_ROOT,
        token=os.environ["HF_TOKEN"],
    )
    print(format_preflight(PREFLIGHT))
'''
)

code(
    '''
# 8b. Load the model (or the MOCK backend), audit it, and run the invariance
# gate at every layer under test.
MODEL = None
BACKEND = None
READOUT_MODEL = None
INVARIANCE = None
AVAILABLE_MODALITIES = []
BLOCKED_MODALITIES = []
if not MODEL_STAGES_ENABLED:
    print("skipped: MODEL_STAGES_ENABLED is False")
else:
    if RUN_REAL_LOCALIZATION:
        from jlens.mmpilot.real_backend import build_real_backend

        BUNDLE = build_real_backend(
            MODEL_REPO_ID,
            revision=MODEL_REVISION,
            token=os.environ["HF_TOKEN"],
            device="cuda" if torch.cuda.is_available() else "cpu",
            allow_model_load=True,
            expect_n_layers=EXPECT_N_LAYERS,
            expect_d_model=EXPECT_D_MODEL,
            expect_vocab_size=EXPECT_VOCAB,
        )
        MODEL = BUNDLE.lens_model
        BACKEND = BUNDLE.backend
        READOUT_MODEL = BUNDLE.lens_model      # Stage B reads out from this
        MODEL_REVISION_USED = BUNDLE.model_revision
        PROCESSOR_REVISION_USED = BUNDLE.processor_revision
        print(json.dumps(BUNDLE.interface, indent=2, default=str))
        print(f"model revision used: {MODEL_REVISION_USED}")
    else:
        from jlens.mmlocalize.mock_readout import MockReadoutModel
        from jlens.mmpilot.mock import MockPilotBackend

        # 42 blocks so the MOCK evaluates the SAME physical layers as the L4.
        BACKEND = MockPilotBackend(
            MOCK_WORLD, supports_audio=False, n_layers=MODEL_N_LAYERS
        )
        # One lens serves Stage B's readout and the J-space dictionary, so the
        # readout model must share the backend's residual width and vocabulary.
        READOUT_MODEL = MockReadoutModel(
            d_model=BACKEND.d_model,
            vocab=int(BACKEND.unembedding_weight().shape[0]),
        )
        MODEL_REVISION_USED = "mock"
        PROCESSOR_REVISION_USED = "mock"

    from jlens.mmpilot.pipeline import available_modalities

    AVAILABLE_MODALITIES, BLOCKED_MODALITIES = available_modalities(BACKEND, CONFIG)
    print(f"available modalities: {AVAILABLE_MODALITIES}")
    print(f"blocked modalities:   {BLOCKED_MODALITIES}")
    print(f"d_model {BACKEND.d_model}, {len(BACKEND.blocks)} decoder blocks")
    if "spoken_audio" in AVAILABLE_MODALITIES:
        raise RuntimeError("spoken audio is outside this study by design")

    # The invariance gate at EVERY layer under test, before anything is
    # measured: capturing must not change the forward pass, and a
    # zero-coefficient intervention must not change the logits.
    from jlens.mmpilot.backend import run_invariance_gate
    from jlens.mmpilot.capability import build_prompt, build_question

    _probe = BACKEND.build_inputs(
        prompt=build_prompt(
            build_question(sorted(SELECTED_NAMES)),
            modality="text",
            caption="a photo used only to probe the hooks",
        ),
        modality="text",
    )
    INVARIANCE = run_invariance_gate(BACKEND, _probe, list(LAYERS))
    print(f"invariance gate passed at {list(LAYERS)}: {INVARIANCE['passed']}")
'''
)

markdown(
    """
## 9. Load the frozen lens for the layers under test

`load_lens_for_localization` separates two claims the robustness study's loader
bundles together:

- **fitted** — the lens has a Jacobian here, so a readout is defined. Required
  for every requested layer; a missing Jacobian is a missing artifact, not
  something Stage B could test.
- **certified** — a held-out native readout has already passed here. Recorded,
  reported, and **not** required, because deciding that is section 10's job.

Everything else is checked exactly as the robustness study checks it: pinned
checksum, published manifest, text-only calibration, matching model revision.
"""
)

code(
    '''
# 9. Load the frozen lens. Nothing is fitted, rescaled, or substituted.
LENS = None
LENS_RECORD = None
if not MODEL_STAGES_ENABLED:
    print("skipped: MODEL_STAGES_ENABLED is False")
else:
    if RUN_REAL_LOCALIZATION:
        from jlens.mmlocalize.real_path import load_lens_for_localization

        LOCALIZATION_LENS = load_lens_for_localization(
            LENS_PATH,
            expect_checksum=LENS_EXPECT_SHA256,
            layers=LAYERS,
            model_revision=MODEL_REVISION_USED,
            expect_d_model=EXPECT_D_MODEL,
        )
        LENS = LOCALIZATION_LENS.lens
        LENS_RECORD = LOCALIZATION_LENS.to_dict()
        LENS_CHECKSUM_USED = LOCALIZATION_LENS.checksum
    else:
        from jlens.mmlocalize.mock_readout import (
            MOCK_LENS_CHECKSUM,
            mock_localization_lens,
        )

        LENS, READOUT_MODEL = mock_localization_lens(READOUT_MODEL, layers=LAYERS)
        LENS_CHECKSUM_USED = MOCK_LENS_CHECKSUM
        LENS_RECORD = {
            "lens_path": "mock",
            "lens_checksum": MOCK_LENS_CHECKSUM,
            "fitted_layers": list(LAYERS),
            "natively_validated_layers": [REFERENCE_LAYER],
            "layers_under_test": [x for x in LAYERS if x != REFERENCE_LAYER],
        }

    print(json.dumps(LENS_RECORD, indent=2, default=str))
    print()
    print(f"  fitted at            {LENS_RECORD['fitted_layers']}")
    print(f"  already certified    {LENS_RECORD['natively_validated_layers']}")
    print(f"  UNDER TEST (sec. 10) {LENS_RECORD['layers_under_test']}")
    print()
    print("  A layer under test carries no causal claim until the Stage B gate")
    print("  passes it. This notebook does not fit a lens.")
'''
)

markdown(
    """
## 10. Stage B — text-only layer validity (held-out, tie-aware)

**This is the gate that decides which layers may carry a causal claim.**

Text only, on prompts the lens has never seen — neither in v2 fitting, nor in v2
validation, nor in any layer-32 confirmatory run, all excluded by hash. No
multimodal example takes part, and no cat or toilet target example takes part.
Every layer is scored in **one** forward pass per prompt.

Every rank is reported under three conventions; the criterion uses the
**midrank**, which is the only one invariant to how ties are broken. The old
gate is computed and printed beside the new one.
"""
)

code(
    '''
# 10. Score every layer's readout against the model's own final-layer argmax,
# on held-out text, with tie-aware metrics. One forward pass per prompt covers
# all four layers.
VALIDITY = {}
VALIDITY_ROWS = []
ELIGIBLE_LAYERS = []
PROMPT_MANIFEST = None
if not MODEL_STAGES_ENABLED:
    print("skipped: MODEL_STAGES_ENABLED is False")
else:
    from jlens.controls import control_lens, wrong_layer_lens
    from jlens.hooks import ActivationRecorder
    from jlens.mmlocalize.lens_validity import (
        eligible_layers,
        evaluate_all_layers,
        tie_aware_row,
    )
    from jlens.native_readout import (
        excluded_prompt_hashes,
        prompt_sha256,
        select_confirmatory_prompts,
    )

    if RUN_REAL_LOCALIZATION:
        from jlens.examples import load_wikitext_prompts

        _v2_prompt_meta = json.loads(
            (Path(V2_RUN_DIR) / "prompt_metadata.json").read_text(encoding="utf-8")
        )
        EXCLUDED_PROMPTS = excluded_prompt_hashes(_v2_prompt_meta)
        # Any layer-32 confirmatory run's prompts are excluded too: Stage B has
        # to be independent of the previous layer-32 test, not a rerun of it.
        for _candidate in sorted(Path(RUNS_ROOT).glob("*layer32*/prompt_manifest.json")):
            for _row in json.loads(
                _candidate.read_text(encoding="utf-8")
            ).get("prompts", []):
                EXCLUDED_PROMPTS[str(_row["prompt_sha256"])] = "layer32_confirmatory"

        _instruction = "Continue this passage."

        def _render(passage):
            passage = passage.strip()
            return READOUT_MODEL.tokenizer.apply_chat_template(
                [{"role": "user", "content": f"{_instruction}\\n\\n{passage}"}],
                tokenize=False,
                add_generation_prompt=True,
            )

        POOL = [_render(text) for text in load_wikitext_prompts(POOL_RECORDS)]
    else:
        from jlens.mmlocalize.mock_readout import mock_validation_prompts

        POOL = mock_validation_prompts(POOL_RECORDS)
        EXCLUDED_PROMPTS = {prompt_sha256(p): "mock_v2_fitting" for p in POOL[:40]}

    VALIDATION_PROMPTS, PROMPT_MANIFEST = select_confirmatory_prompts(
        POOL,
        n_prompts=N_VALIDATION_PROMPTS,
        excluded=EXCLUDED_PROMPTS,
        seed=VALIDATION_PROMPT_SEED,
    )
    PROMPT_MANIFEST["modality"] = "text"
    PROMPT_MANIFEST["excluded_roles"] = sorted(set(EXCLUDED_PROMPTS.values()))
    print(
        f"{len(VALIDATION_PROMPTS)} held-out prompts from a pool of "
        f"{PROMPT_MANIFEST['pool_size']}; {PROMPT_MANIFEST['n_excluded']} excluded "
        f"as {PROMPT_MANIFEST['excluded_roles']}; overlap=0"
    )

    VARIANTS = {
        "j_lens": LENS,
        "permuted": control_lens(LENS, "permuted", seed=CONTROL_SEED),
        "random": control_lens(LENS, "random", seed=CONTROL_SEED),
        "wrong_layer": wrong_layer_lens(LENS),
    }
    _final = READOUT_MODEL.n_layers - 1
    for _index, _prompt in enumerate(VALIDATION_PROMPTS):
        _sha = PROMPT_MANIFEST["prompts"][_index]["prompt_sha256"]
        _ids = READOUT_MODEL.encode(_prompt)
        with torch.no_grad():
            with ActivationRecorder(
                READOUT_MODEL.layers, at=[*LAYERS, _final]
            ) as _recorder:
                READOUT_MODEL.forward(_ids)
            _actual = READOUT_MODEL.unembed(
                _recorder.activations[_final][0, -1].float()
            )
            for _layer in LAYERS:
                _residual = _recorder.activations[_layer][0, -1].float()
                _scored = {"logit_lens": READOUT_MODEL.unembed(_residual)}
                for _name, _variant in VARIANTS.items():
                    _scored[_name] = READOUT_MODEL.unembed(
                        _variant.transport(_residual, _layer)
                    )
                for _name in READOUT_VARIANTS:
                    VALIDITY_ROWS.append(
                        tie_aware_row(
                            sample_index=_index,
                            prompt_sha=_sha,
                            layer=_layer,
                            variant=_name,
                            variant_logits=_scored[_name],
                            actual_logits=_actual,
                        )
                    )
        del _recorder, _actual
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    VALIDITY = evaluate_all_layers(VALIDITY_ROWS, layers=LAYERS)
    ELIGIBLE_LAYERS = eligible_layers(VALIDITY)

    print("\\n" + "=" * 72)
    print("STAGE B — LAYER VALIDITY (midrank criterion; both gates reported)")
    print("=" * 72)
    for _layer in LAYERS:
        _r = VALIDITY[_layer]
        _m = _r["metrics"]["j_lens"]
        print(
            f"  L{_layer:<3d} ~{_r['metrics'] and ''}"
            f"{'ELIGIBLE  ' if _r['eligible'] else 'INELIGIBLE'}  "
            f"MRR={_m['mean_reciprocal_rank']:.4f}  "
            f"med_midrank={_m['median_midrank']:.1f}  "
            f"med_optimistic={_m['median_optimistic_rank']:.1f}  "
            f"unique_top1={_m['unique_top1_agreement']:.3f}  "
            f"tied_at_max={_m['tied_at_max_rate']:.3f}  "
            f"top10={_m['top10_inclusion']:.3f}"
        )
        print(
            f"        old gate: {'pass' if _r['legacy_gate']['passed'] else 'fail'}"
            f"   new gate: {'pass' if _r['eligible'] else 'fail'}"
            f"   gates agree: {_r['gates_agree']}"
        )
        for _check in _r["checks"]:
            print(f"        [{'ok' if _check['passed'] else 'FAIL'}] "
                  f"{_check['check']}: {_check['detail']}")
    print()
    print(f"  ELIGIBLE LAYERS: {ELIGIBLE_LAYERS}")
    print(f"  ineligible:      {[x for x in LAYERS if x not in ELIGIBLE_LAYERS]}")
    print("  An ineligible layer keeps its diagnostics and is skipped causally.")
    print("  That skip is not evidence that transfer is absent there.")

    CONFIG.causal_layers = tuple(ELIGIBLE_LAYERS)
    (RUN_DIR / "layer_validity.json").write_text(
        json.dumps(
            {
                "protocol": VALIDITY_PROTOCOL,
                "gate": LOCALIZATION_VALIDITY_GATE.to_dict(),
                "gate_digest": LOCALIZATION_VALIDITY_GATE.digest,
                "gate_text": gate_text(),
                "layer_manifest": layer_manifest(LAYERS),
                "prompt_manifest": PROMPT_MANIFEST,
                "per_layer": {str(k): v for k, v in VALIDITY.items()},
                "eligible_layers": ELIGIBLE_LAYERS,
                "fits_a_lens": False,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
'''
)

markdown(
    """
## 11. Capability confirmation on the localization subset

The completed run established six-way capability on **its** images. These are
different photographs, so capability is re-confirmed here for cat and toilet
before anything is read off them — cheaply, at the same six-way forced choice
and both option orders, so a number here means what a number there meant.
"""
)

code(
    '''
# 11. Behavioral gate on the frozen localization subset. Opens the run store.
CAPABILITY = None
STORE = None
if not MODEL_STAGES_ENABLED:
    print("skipped: MODEL_STAGES_ENABLED is False")
else:
    from jlens.mmpilot.pipeline import scientific_fingerprint, stage_capability
    from jlens.mmpilot.store import RunFingerprint, UnitStore

    if RUN_REAL_LOCALIZATION:
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
        ranked_concepts=SELECTED_NAMES,
        selected_concepts=SELECTED_NAMES,
        focal_concepts=FOCAL_CONCEPTS,
        unrelated_controls=UNRELATED_CONTROLS,
        derived_cache_fingerprint=MANIFEST_CHECKSUM,
        split_provenance_checksum=SPLIT_PROVENANCE_CHECKSUM,
        n_train_positive_images=N_SOURCE_POSITIVE_IMAGES,
        n_train_negative_images=N_SOURCE_NEGATIVE_IMAGES,
        n_test_positive_images=N_TARGET_POSITIVE_IMAGES,
        n_test_negative_images=N_TARGET_NEGATIVE_IMAGES,
        verdict_version=LOCALIZATION_VERDICT_VERSION,
    )
    # Everything the results were produced from. A change to any of it refuses
    # the resume rather than mixing artifacts.
    SELECTION_FINGERPRINT.update(
        {
            "layer_manifest": layer_manifest(LAYERS),
            "layer_set_version": LAYER_SET_VERSION,
            "validity_gate_digest": LOCALIZATION_VALIDITY_GATE.digest,
            "validity_protocol": VALIDITY_PROTOCOL,
            "eligible_causal_layers": list(ELIGIBLE_LAYERS),
            "target_policy": TARGET_POLICY,
            "target_checksum": TARGETS.checksum,
            "target_manifest_checksum": TARGET_MANIFEST["manifest_checksum"],
            "source_image_ids": TARGETS.all_source_images(),
            "target_image_ids": TARGETS.all_target_images(),
            "completed_run_fingerprint": COMPLETED_RUN["fingerprint"],
            "validation_prompt_hashes": [
                row["prompt_sha256"] for row in PROMPT_MANIFEST["prompts"]
            ],
            "localization_concepts": list(CONCEPTS),
        }
    )
    FINGERPRINT = RunFingerprint(
        mode=CONFIG.mode,
        model_repo_id=MODEL_REPO_ID if RUN_REAL_LOCALIZATION else "mock/gemma-like",
        model_revision=MODEL_REVISION_USED,
        processor_revision=PROCESSOR_REVISION_USED,
        layers=tuple(LAYERS),
        lens_checksum=LENS_CHECKSUM_USED,
        manifest_checksum=MANIFEST_CHECKSUM,
        split_id=SPLIT_SEED,
        intervention_config={
            "alphas": list(CONFIG.alphas),
            "direction_top_k": CONFIG.direction_top_k,
            "causal_layers": list(ELIGIBLE_LAYERS),
        },
        selection_config=SELECTION_FINGERPRINT,
    )
    STORE = UnitStore(RUN_DIR, FINGERPRINT)
    print("run state:", STORE.open())
    print(f"fingerprint {FINGERPRINT.digest}")
    print(
        "  units from the robustness run can never be reused here: the layer "
        "set, the validity gate, the target set and the concept pair are all "
        "bound into this digest."
    )

    CAPABILITY_OUTCOME, CAPABILITY = stage_capability(
        BACKEND, STORE, SUBSET, CONFIG, MEDIA, modalities=AVAILABLE_MODALITIES
    )
    print("\\n" + CAPABILITY_OUTCOME.line("capability"))
    for _concept, _per_modality in sorted(CAPABILITY["per_concept"].items()):
        print(f"  {_concept:12s} " + "  ".join(
            f"{m}={e['n_correct']}/{e['n']}" for m, e in sorted(_per_modality.items())
        ))
    RETAINED = CAPABILITY["text_image_retained_concepts"]
    print(f"\\nretained (text+image): {RETAINED}")
    CAPABILITY_FAILED = [c for c in CONCEPTS if c not in RETAINED]
    if CAPABILITY_FAILED:
        print(
            f"\\nLOCALIZATION CAPABILITY NO-GO: {CAPABILITY_FAILED} did not clear "
            "the gate on these photographs. The concepts are NOT replaced — they "
            "were fixed by the completed run's bidirectional replication, and "
            "swapping one now would be selecting on the outcome."
        )
'''
)

markdown(
    """
## 12. Capture activations at all four layers

**One forward pass per (group, modality) records every layer.** That is why a
four-layer diagnostic costs what a one-layer study cost, and it also guarantees
the layers see literally the same forward pass rather than four re-runs.

Same convention as the completed run: post-block residual, final prompt token.
"""
)

code(
    '''
# 12. Final-prompt-token residual at layers 20, 26, 32 and 38 — one pass each.
ACTIVATIONS = []
if not MODEL_STAGES_ENABLED:
    print("skipped: MODEL_STAGES_ENABLED is False")
else:
    from jlens.mmpilot.jspace import CONVENTIONS
    from jlens.mmpilot.pipeline import stage_activations

    print("capture convention:")
    for _key in ("hook_site", "position"):
        print(f"  {_key}: {CONVENTIONS[_key]}")

    ACTIVATION_OUTCOME = stage_activations(
        BACKEND, STORE, SUBSET, CONFIG, MEDIA,
        modalities=AVAILABLE_MODALITIES,
        # The unrelated-control concepts are captured too: their directions are
        # controls, and a control that was never estimated is not a control.
        retained_concepts=list(DIRECTION_CONCEPTS),
        model_revision=MODEL_REVISION_USED,
    )
    ACTIVATIONS = ACTIVATION_OUTCOME.records
    print("\\n" + ACTIVATION_OUTCOME.line("activation"))
    print(f"distinct images captured: {len({r['image_id'] for r in ACTIVATIONS})}")
    for _layer in LAYERS:
        _n = len([r for r in ACTIVATIONS if int(r["layer"]) == _layer])
        print(f"  L{_layer}: {_n} activations")
'''
)

markdown(
    """
## 13. Compute J-space codes

Nonnegative gradient pursuit against the frozen dictionary, one dictionary per
layer. No basis is fitted, rescaled, or substituted.
"""
)

code(
    '''
# 13. Sparse nonnegative J-space coordinates for every stored activation.
CODES = []
DICTIONARIES = {}
if not MODEL_STAGES_ENABLED:
    print("skipped: MODEL_STAGES_ENABLED is False")
else:
    from jlens.mmpilot.pipeline import build_dictionaries, stage_codes

    DICTIONARIES = build_dictionaries(
        LENS,
        LAYERS,
        BACKEND,
        device="cuda" if (RUN_REAL_LOCALIZATION and torch.cuda.is_available()) else "cpu",
        dtype=torch.float16 if RUN_REAL_LOCALIZATION else torch.float32,
        build_chunk_rows=32768 if RUN_REAL_LOCALIZATION else None,
    )
    CODE_OUTCOME = stage_codes(
        STORE, ACTIVATIONS, DICTIONARIES, CONFIG, lens_checksum=LENS_CHECKSUM_USED
    )
    CODES = CODE_OUTCOME.records
    print(CODE_OUTCOME.line("jspace"))
'''
)

markdown(
    """
## 14. Stage C — paired representational tests, per layer

Cross-modal retrieval, matched-minus-mismatched separation, weighted support
overlap, the raw-residual baseline and the shuffled-label control — computed at
**every** layer, eligible or not, because a diagnostic is worth having even
where a causal claim is not permitted.

Identical samples at every layer, so this is a paired depth comparison. A
candidate target is excluded whenever it shares the query's `image_id`, and the
photograph is the unit throughout.
"""
)

code(
    '''
# 14. Representational structure at every layer, under the image-level
# exclusion rule. Ineligible layers are measured too — only their CAUSAL stage
# is skipped.
REPRESENTATIONAL = {}
if not MODEL_STAGES_ENABLED:
    print("skipped: MODEL_STAGES_ENABLED is False")
else:
    from jlens.mmpilot.jspace import EXCLUSION_RULE_VERSION
    from jlens.mmpilot.pipeline import stage_representational

    print(f"exclusion rule: {EXCLUSION_RULE_VERSION}\\n")
    for _layer in LAYERS:
        REPRESENTATIONAL[_layer] = stage_representational(
            STORE, ACTIVATIONS, CODES, CONFIG,
            layer=_layer,
            modalities=AVAILABLE_MODALITIES,
        )
        _eligible = "eligible" if _layer in ELIGIBLE_LAYERS else "INELIGIBLE"
        print(f"L{_layer} ({_eligible}):")
        for _pair, _entry in sorted(REPRESENTATIONAL[_layer]["pairs"].items()):
            print(
                f"  {_pair:14s} queries={_entry['jspace_retrieval']['n_queries']:3d}  "
                f"top1={_entry['jspace_retrieval']['top1_accuracy']:.3f}  "
                f"mrr={_entry['jspace_retrieval']['mrr']:.3f}  "
                f"shuffled_p95={_entry['shuffled_control']['p95_top1_accuracy']:.3f}  "
                f"J gap {_entry['jspace_separation']['gap']:+.4f}  "
                f"raw gap {_entry['raw_residual_separation']['gap']:+.4f}"
            )
'''
)

markdown(
    """
## 15. Estimate source-only directions at eligible layers

Each direction comes from **four distinct training photographs** of one modality
and four distinct matched negatives, the two sets disjoint. No target-modality
activation and no test example takes part. A short set refuses rather than
shrinking.

Directions are estimated **only at eligible layers**: a direction at a layer
whose readout could not be shown to mean anything would be an object with no
interpretation.
"""
)

code(
    '''
# 15. Source-derived directions, from distinct training images only, at the
# layers that earned a causal claim.
DIRECTIONS = {}
if not MODEL_STAGES_ENABLED:
    print("skipped: MODEL_STAGES_ENABLED is False")
elif not ELIGIBLE_LAYERS:
    print(
        "skipped: no layer passed the Stage B gate, so there is no layer at "
        "which a direction would have an interpretation"
    )
else:
    from jlens.mmpilot.pipeline import stage_directions

    DIRECTION_OUTCOME, DIRECTIONS = stage_directions(
        STORE, CODES, ACTIVATIONS, DICTIONARIES, CONFIG,
        concepts=SELECTED_NAMES,
        modalities=AVAILABLE_MODALITIES,
        lens_checksum=LENS_CHECKSUM_USED,
    )
    print(DIRECTION_OUTCOME.line("direction"))
    for _record in sorted(
        DIRECTION_OUTCOME.records,
        key=lambda r: (int(r["layer"]), r.get("concept") or "", r.get("source_modality") or ""),
    ):
        if _record["kind"] != "source_concept" or _record["concept"] not in (
            DIRECTION_CONCEPTS
        ):
            continue
        _role = "focal" if _record["concept"] in CONCEPTS else "control"
        print(
            f"  L{_record['layer']:<3d} {_role:7s} {_record['concept']:8s} "
            f"{_record['source_modality']:6s} "
            f"positives={_record['n_source_positive_images']} images  "
            f"negatives={_record['n_source_negative_images']} images  "
            f"uses_target_modality_data={_record['uses_target_modality_data']}"
        )
'''
)

markdown(
    """
## 16. Stage D — off-diagonal causal interventions at eligible layers

Text-derived directions applied to image targets, and image-derived directions
applied to text targets. Necessity (subtract from a held-out positive) and
sufficiency (add to a matched negative), four distinct photographs each, all
four controls, at the final prompt token.

**The same target images at every layer** — that is what makes the depth
contrast paired, and it is asserted rather than assumed.

An ineligible layer is skipped here, and `assert_causally_eligible` is what
enforces it: a causal claim at a layer whose readout was never validated has no
interpretation.
"""
)

code(
    '''
# 16. The cross-modal transfer cells at eligible layers, with all four controls.
INTERVENTIONS = None
INTERVENTION_RECORDS = []
ACTUAL_BUDGET = None
if not MODEL_STAGES_ENABLED:
    print("skipped: MODEL_STAGES_ENABLED is False")
elif not ELIGIBLE_LAYERS:
    print("=" * 72)
    print("NO LAYER PASSED THE STAGE B GATE — the causal stage is skipped entirely.")
    print("=" * 72)
    print("This is not evidence that transfer is absent at any layer. It is the")
    print("absence of a usable readout, which is a fact about the frozen lens.")
else:
    from jlens.mmlocalize.lens_validity import assert_causally_eligible
    from jlens.mmpilot.pipeline import stage_causal

    for _layer in ELIGIBLE_LAYERS:
        assert_causally_eligible(_layer, VALIDITY)   # refuses an ineligible layer
    print(f"causally eligible: {ELIGIBLE_LAYERS}")
    print(f"skipped causally:  {[x for x in LAYERS if x not in ELIGIBLE_LAYERS]}")

    ACTUAL_BUDGET = estimate_localization_passes(
        n_concepts=len(SELECTED_NAMES),
        modalities=MODALITIES,
        n_total_groups=N_TOTAL_GROUPS,
        n_capability_groups=N_CAPABILITY_GROUPS,
        n_layers_captured=len(LAYERS),
        n_eligible_causal_layers=len(ELIGIBLE_LAYERS),
        n_targets_per_cell=N_TARGET_POSITIVE_IMAGES + N_TARGET_NEGATIVE_IMAGES,
        alphas=CONFIG.alphas,
        n_validation_prompts=N_VALIDATION_PROMPTS,
        recalibration_enabled=False,
    )
    print("\\nACTUAL budget now that eligibility is known:")
    print(format_budget(ACTUAL_BUDGET))

    CAUSAL_OUTCOME, INTERVENTIONS = stage_causal(
        BACKEND, STORE, SUBSET, CODES, ACTIVATIONS, DIRECTIONS, CONFIG, MEDIA,
        concepts=list(CONCEPTS),
        modalities=AVAILABLE_MODALITIES,
        all_concepts=SELECTED_NAMES,
        unrelated_controls=UNRELATED_CONTROLS,
    )
    INTERVENTION_RECORDS = CAUSAL_OUTCOME.records
    print("\\n" + CAUSAL_OUTCOME.line("intervention"))

    # The depth contrast is paired only if every layer got the same photographs.
    _images_by_layer = {}
    for _record in INTERVENTION_RECORDS:
        _images_by_layer.setdefault(int(_record["layer"]), set()).add(
            str(_record["image_id"])
        )
    PAIRING = assert_same_targets_across_layers(TARGETS, _images_by_layer)
    print(f"paired across layers: {PAIRING}")
'''
)

markdown(
    """
## 17. Aggregate at the image level

Repeated observations of one photograph are averaged **within** the image before
the cell statistic is computed. With one group per image there is nothing to
average — and the report says so explicitly, which is the check that the
selection policy actually held.
"""
)

code(
    '''
# 17. Image-level aggregation. The photograph is the independent unit.
IMAGE_LEVEL = None
if not MODEL_STAGES_ENABLED or not INTERVENTION_RECORDS:
    print("skipped: no interventions were run")
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
    if DIVERGENCE["n_rows_pseudoreplicated_at_group_level"] == 0:
        print(
            "  as designed: one synchronized group per photograph, so no "
            "aggregation was needed to make the unit honest"
        )
'''
)

markdown(
    """
## 18. Localization verdict and report

One of three, and the rubric refuses to force a positive earlier-layer
conclusion:

| verdict | what it takes |
| --- | --- |
| `EARLY_TRANSFER_CONFIRMED` | layer 38 reproduces **and** an earlier layer passes the validity gate, beats shuffled controls in both directions, and shows an expected-sign off-diagonal effect for cat or toilet that exceeds random and unrelated controls with sane norms |
| `LATE_ONLY_SUPPORTED` | layer 38 reproduces and the earlier layers **that were eligible** produced no controlled transfer |
| `INCONCLUSIVE_LAYER_LOCALIZATION` | layer 38 fails to reproduce, **or** no earlier layer was eligible so none was ever tested |

The last row matters: "no earlier layer could be validated" is **not**
`LATE_ONLY_SUPPORTED`. That would report an untested layer as a negative result.
"""
)

code(
    '''
# 18. Apply the localization rubric and write the report.
VERDICT = None
if not MODEL_STAGES_ENABLED:
    print("skipped: MODEL_STAGES_ENABLED is False")
else:
    from jlens.mmlocalize.verdict import localization_verdict, render_report

    VERDICT = localization_verdict(
        validity=VALIDITY,
        representational=REPRESENTATIONAL,
        interventions=IMAGE_LEVEL or {"rows": []},
        target_manifest=TARGET_MANIFEST,
        layers=LAYERS,
        reference_layer=REFERENCE_LAYER,
        concepts=CONCEPTS,
        thresholds=THRESHOLDS,
    )
    REPORT = render_report(
        run_dir=str(RUN_DIR),
        verdict=VERDICT,
        validity=VALIDITY,
        budget=(ACTUAL_BUDGET or BUDGET).to_dict(),
        resume=STORE.status_report(),
        mode=CONFIG.mode,
    )
    (RUN_DIR / "localization_report.md").write_text(REPORT, encoding="utf-8")
    (RUN_DIR / "localization_summary.json").write_text(
        json.dumps(
            {
                "verdict": VERDICT,
                "layer_validity": {str(k): v for k, v in VALIDITY.items()},
                "budget": (ACTUAL_BUDGET or BUDGET).to_dict(),
                "target_manifest": TARGET_MANIFEST,
                "image_exclusion_audit": EXCLUSION_AUDIT,
                "split_provenance": SPLIT_PROVENANCE,
                "selection_fingerprint": SELECTION_FINGERPRINT,
                "fingerprint_digest": FINGERPRINT.digest,
                "lens_record": LENS_RECORD,
                "completed_run": COMPLETED_RUN,
                "capability": CAPABILITY,
                "representational": {str(k): v for k, v in REPRESENTATIONAL.items()},
                "interventions_image_level": IMAGE_LEVEL,
                "resume": STORE.status_report(),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    STORE.save("metric", "localization_verdict", VERDICT)

    print("=" * 72)
    print(f"VERDICT: {VERDICT['verdict']}")
    print("=" * 72)
    print(VERDICT["rationale"])
    print()
    for _name, _status in VERDICT["criteria_status"].items():
        print(f"  {_status:15s} {_name}")
    print()
    print(f"  layer 38 reproduces:      {VERDICT['reference_layer_reproduces']}")
    print(f"  eligible earlier layers:  {VERDICT['eligible_earlier_layers']}")
    print(f"  earlier layers with transfer: {VERDICT['earlier_layers_transferring']}")
    print(f"  EARLIEST TESTED LAYER WITH EVIDENCE: "
          f"{VERDICT['earliest_tested_layer_with_evidence']}")
    print()
    print(f"  {VERDICT['depth_scope_limitation']}")
    print(f"  {VERDICT['concept_conditioning_limitation']}")
    if VERDICT.get("target_policy_limitation"):
        print(f"  {VERDICT['target_policy_limitation']}")
    print()
    print(f"report  {RUN_DIR / 'localization_report.md'}")
    print(f"summary {RUN_DIR / 'localization_summary.json'}")
'''
)

markdown(
    """
## 19. Resume state

Every stage writes one small checksummed JSON per unit as soon as it finishes,
so a disconnected Colab session loses at most the unit in flight. Re-running
against the same run directory reuses everything it can verify.

Units from the robustness run can never be reused here: the layer set, the
validity gate digest, the frozen target set, the validation prompt hashes and
the concept pair are all bound into the fingerprint.
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
    print("To run the study, set in section 2:")
    print()
    print("    RUN_REAL_LOCALIZATION     = True")
    print("    RUN_MODEL_STAGES          = True")
    print("    CONFIRM_MODEL_PASS_BUDGET = True   # after reading section 7")
    print()
    print("Leave RUN_TEXT_RECALIBRATION False for the first run.")
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
    ):
        print("  " + _outcome.line(_name))
    if ELIGIBLE_LAYERS:
        print("  " + DIRECTION_OUTCOME.line("direction"))
        print("  " + CAUSAL_OUTCOME.line("intervention"))
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
