"""Generate the matched multimodal J-lens experiment notebook."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "notebooks" / "multimodal_jspace_matched_jlens_colab.ipynb"
CELLS: list[tuple[str, str]] = []


def markdown(value: str) -> None:
    CELLS.append(("markdown", value.strip("\n")))


def code(value: str) -> None:
    CELLS.append(("code", value.strip("\n")))


markdown(
    r"""
# Matched-distribution J-Lens — text, image, spoken captions, and pooled

This notebook asks one clean comparative question:

> Does estimating the average decoder Jacobian on the checkpoint's real
> multimodal input distribution produce a more faithful readout and causal
> interface than estimating it on text alone?

It fits four lenses over the **same synchronized photographs/captions**:

1. text-only;
2. image-only;
3. spoken-caption-only;
4. an equal-size pooled arm assigning the same photographs evenly across the
   three modalities.

The old text-only result remains a baseline. The pooled lens is not declared
better in advance, and no arm is selected after seeing a causal result.

## Scientific boundaries

* The fitted object is still the paper's sample-mean Jacobian
  `E[d h_final / d h_l]`; no probe, adapter, classifier, or cross-modal
  projection is learned.
* Image/audio examples pass through the pinned processor and modality towers.
  Their real placeholder spans participate in the Jacobian estimator.
* Fit, cross-evaluation, and causal photographs are disjoint.
* The primary cross-evaluation endpoint is full-vocabulary fidelity to the
  model's own unrestricted next-token answer. It is not semantic accuracy.
* The causal comparison is text-lens versus pooled-lens using the paper's exact
  two-coordinate exchange, unrestricted next-token output, no answer appended,
  and no candidate list. Alpha=1 remains the exact primary exchange; a separate
  paired dose-response stage labels every other alpha as sensitivity evidence.
* Spoken audio means a human reading a caption, not environmental sound.

## Stages and resume

| stage | runtime | work | resume unit |
|---|---|---|---|
| 0 | CPU | load the cached synchronization and freeze populations | persisted plan |
| 1 | A100 recommended | fit four lenses | atomic arm accumulator |
| 2 | GPU | 4 x 3 full-vocabulary cross-evaluation | one photograph |
| 3 | GPU | gated exact-swap comparison on fresh bird/cat media | one trial |
| 3B | GPU | paired alpha dose-response on the frozen Stage-3 population | one condition |
| 3C | A100 recommended | extend pooled J to L16-L32 and run a broad paper-depth workspace development | one fit checkpoint / one trial |
| 3D | A100 or L4 | checksum-pinned fresh confirmation of the frozen development winner | one capability/trial JSON |
| 4 | CPU | write the report from stored units | no model |

Changing any model, processor, audio protocol, cache, population, order, layer,
prompt, lens, or causal setting changes the fingerprint and refuses reuse.
"""
)

markdown("## 0. Bootstrap")
code(
    r'''
import hashlib, json, os, subprocess, sys, tempfile
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
REPO_URL = "https://github.com/MechInterpreter/jacobian-lens-gemma.git"
BRANCH = "experiment/spokencoco-jspace-pilot"
REPO_DIR = Path(
    os.environ.get("JLENS_REPO_DIR")
    or ("/content/jacobian-lens-gemma" if IN_COLAB else Path.cwd())
)

if IN_COLAB:
    if not (REPO_DIR / ".git").is_dir():
        subprocess.run(
            ["git", "clone", "--branch", BRANCH, "--single-branch", REPO_URL, str(REPO_DIR)],
            check=True,
        )
    else:
        subprocess.run(["git", "-C", str(REPO_DIR), "fetch", "origin", BRANCH], check=True)
        subprocess.run(["git", "-C", str(REPO_DIR), "checkout", BRANCH], check=True)
        subprocess.run(["git", "-C", str(REPO_DIR), "pull", "--ff-only", "origin", BRANCH], check=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-e", str(REPO_DIR),
         "transformers==5.13.1", "accelerate", "soundfile", "datasets", "pillow"],
        check=True,
    )

os.chdir(REPO_DIR)
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))
COMMIT = subprocess.run(
    ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
).stdout.strip()
print("commit", COMMIT)
'''
)

markdown("## 1. Configuration — set this once")
code(
    r'''
# For a clean real run set RUN_REAL_MATCHED_JLENS=True and the desired stages.
# Stage 0 can run on CPU. Stages 1-3 need a GPU; A100 is strongly recommended
# for Stage 1. Stage 4 can run in a fresh CPU session by setting REPORT_RUN_DIR.
RUN_REAL_MATCHED_JLENS = False
RUN_STAGE0_FREEZE_PLAN = False
RUN_STAGE1_FIT_LENSES = False
RUN_STAGE2_CROSS_EVALUATE = False
RUN_STAGE3_CAUSAL_COMPARE = False
RUN_STAGE3B_ALPHA_SWEEP = False
RUN_STAGE3C_BROAD_POOLED_WORKSPACE = False
RUN_STAGE3D_FRESH_MULTIMODAL_CONFIRMATION = False
RUN_STAGE4_WRITE_REPORT = False

CONFIRM_MODEL_LOAD = False
CONFIRM_FIT_BUDGET = False
CONFIRM_CAUSAL_BUDGET = False
CONFIRM_ALPHA_SWEEP_BUDGET = False
CONFIRM_BROAD_POOLED_FIT_BUDGET = False
CONFIRM_BROAD_POOLED_CAUSAL_BUDGET = False
CONFIRM_FRESH_MULTIMODAL_CONFIRMATION_BUDGET = False
REPORT_RUN_DIR = None

MODEL_REPO_ID = "google/gemma-4-E4B-it"
MODEL_REVISION = "fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd"
EXPECT_N_LAYERS, EXPECT_D_MODEL, EXPECT_VOCAB = 42, 2560, 262144
AUDIO_PROTOCOL_FINGERPRINT = (
    "sha256:9ad8bcc9420a7983f6e3b75d5d7080c0e2fcf0a94a76431917fcde73ba777920"
)

# Scale 99 is today's bounded comparative pilot. It is approximately the
# original published scale and permits an exact 33/33/33 pooled allocation,
# but it is not called a definitive confirmation.
# Change to 250 only before Stage 0; it creates a different fingerprint/run.
# Ninety-nine makes the pooled arm exactly 33 text + 33 image + 33 audio while
# keeping every arm at the same total sample count.
N_FIT_GROUPS = 99
N_CROSS_EVAL_GROUPS = 48
# Prospective causal follow-up.  The completed 32-per-concept screen is read
# only to exclude its photographs.  Ninety-six new candidates are selected
# before any new clean answer is opened.
N_CAUSAL_CANDIDATES_PER_CONCEPT = 96
N_CAUSAL_IMAGES_PER_CELL = 8
SOURCE_LAYERS = (33, 34, 35, 36, 37, 38, 39, 40)
TARGET_LAYER = 41
DIM_BATCH = 8
SKIP_FIRST = 16
CHECKPOINT_EVERY = 10
PLAN_SEED = "matched-jlens-scale99-20260819-v1"
CAUSAL_SEED = "matched-jlens-causal-followup-20260819-v1"
EVAL_CONCEPTS = ("bird", "cat")
CONTROL_CONCEPTS = ("zebra", "giraffe")
CAUSAL_LAYERS = SOURCE_LAYERS
PRIMARY_ALPHA = 1.0
BROAD_POOLED_EARLY_LAYERS = tuple(range(16, 33))
BROAD_POOLED_LATE_LAYERS = tuple(range(33, 41))
BROAD_POOLED_BAND = tuple(range(16, 41))
BROAD_POOLED_ALPHAS = (1.0, 2.0)
BROAD_POOLED_PAIRS = (
    ("bird", "cat"), ("cat", "bird"),
    ("bird", "zebra"), ("zebra", "bird"),
    ("bird", "giraffe"), ("giraffe", "bird"),
)
BROAD_POOLED_CONCEPTS = ("bird", "cat", "zebra", "giraffe")
BROAD_POOLED_CONTROLS = ("microwave", "toilet")
BROAD_POOLED_CANDIDATES_PER_CONCEPT = 64
BROAD_POOLED_IMAGES_PER_DIRECTION = 8
BROAD_POOLED_SEED = "paper-depth-pooled-j-development-20260822-v1"

# Frozen only after Stage 3C completed. Stage 3D verifies every pin before it
# opens a fresh photograph. Nothing here is selected again in confirmation.
BROAD_DEVELOPMENT_RUN_DIR = (
    "/content/drive/MyDrive/jacobian-lens-gemma/runs/mmbroadpooledj/"
    "mmbroadpooledj_real_ee944cd22ba1"
)
EXPECTED_BROAD_DEVELOPMENT_REPORT_CHECKSUM = (
    "sha256:ec1747a78902080ac3fac5f6aa5bc105e36f49ade5b517282ad7c3673da31a42"
)
EXPECTED_BROAD_DEVELOPMENT_POPULATION_DIGEST = (
    "sha256:c3f1cdc351d4710b79937ab49a99a877647594ff475bd64b3dfd13170047f23d"
)
EXPECTED_BROAD_POOLED_LENS_CHECKSUM = (
    "sha256:3321a47db8aa7f948507e0419d80f23e97684fe3411d07f5d0d14c76b5d0ee1f"
)
CONFIRMATION_DIRECTION = ("bird", "cat")
CONFIRMATION_ALPHA = 1.0
CONFIRMATION_CANDIDATES = 64
CONFIRMATION_IMAGES = 16
CONFIRMATION_MIN_SUCCESS_RATE = 0.75
CONFIRMATION_MIN_CONTROL_MARGIN = 0.25
CONFIRMATION_FAMILYWISE_ALPHA = 0.05
CONFIRMATION_SEED = "broad-pooled-j-fresh-confirmation-20260822-v1"

# ---------------------------------------------------------------------------
# Follow-up studies. Experiment A localizes inside the validated band on the
# already spent development population (exploratory only). Experiment B tests a
# non-leg-count property on genuinely fresh media. Experiment C is a
# prospective replication test of the cat->bird development failure.
#
# None of these fits or refits anything. Each reuses the checksum-pinned pooled
# L16-L40 lens. Only the pooled arm spans that band: the text-only, image-only
# and spoken-audio-only lenses cover L33-L40, so no four-arm L16-L40 comparison
# exists here and none is claimed.
RUN_STAGE5A_BAND_LOCALIZATION = False
RUN_STAGE5B0_PROPERTY_AUDIT = False
RUN_STAGE5B1_NEW_PROPERTY_DEVELOPMENT = False
RUN_STAGE5B2_FREEZE_NEW_PROPERTY_DESIGN = False
RUN_STAGE5B3_NEW_PROPERTY_CONFIRMATION = False
RUN_STAGE5C_ASYMMETRY_REPLICATION = False
RUN_ARTIFACT_EXCLUSION_AUDIT = False

CONFIRM_LOCALIZATION_BUDGET = False
CONFIRM_NEW_PROPERTY_DEVELOPMENT_BUDGET = False
CONFIRM_NEW_PROPERTY_CONFIRMATION_BUDGET = False
CONFIRM_ASYMMETRY_BUDGET = False

# The completed Stage 3D confirmation, read only to spend its media. All 64
# candidate photographs were opened during capability screening, so all 64 are
# excluded from every later population — not only the 16 that were recruited.
FRESH_CONFIRMATION_RUN_DIR = None
EXPECTED_FRESH_CONFIRMATION_REPORT_CHECKSUM = (
    "sha256:2bb6dcc1346229573566125bc8d91c782247d55af5091f4215d98bb621472ff7"
)
FRESH_CONFIRMATION_CANDIDATES_OPENED = 64
FRESH_CONFIRMATION_IMAGES_RECRUITED = 16

# Experiment A. The band grid and its analysis rule live in
# jlens.mmpilot.multimodal_followup.localization_grid() and are frozen there
# before any sub-band outcome exists. The population is the spent broad
# development one, so every output of this stage is exploratory.
LOCALIZATION_DIRECTION = ("bird", "cat")
LOCALIZATION_ALPHA = 1.0

# Experiment B. 'body_covering' is the audited first choice: it is not
# derivable from leg count, it keeps 'bird' available, and its answers are
# ordinary one-word nouns. 'animal_sound' is the declared fallback; it refuses
# 'bird' outright because COCO birds have no single conventional sound.
# Concepts whose covering is contested (horse, cow, zebra, giraffe, elephant)
# are refused by the audit and cannot be selected here.
NEW_PROPERTY_FAMILY = "body_covering"
NEW_PROPERTY_FALLBACK_FAMILY = "animal_sound"
NEW_PROPERTY_CONCEPTS = ("bird", "cat", "sheep")
NEW_PROPERTY_DEV_DIRECTIONS = (
    ("bird", "cat"), ("cat", "bird"),
    ("bird", "sheep"), ("sheep", "bird"),
    ("cat", "sheep"), ("sheep", "cat"),
)
# Predeclared tie-break, fixed before any outcome: if development licenses
# several directions, confirmation takes the first of these that passed.
NEW_PROPERTY_DIRECTION_PRIORITY = (
    "bird->cat", "bird->sheep", "cat->sheep",
    "sheep->cat", "sheep->bird", "cat->bird",
)
NEW_PROPERTY_MAX_NEW_TOKENS = 6
NEW_PROPERTY_DEV_CANDIDATES_PER_CONCEPT = 48
NEW_PROPERTY_DEV_IMAGES_PER_DIRECTION = 8
NEW_PROPERTY_DEV_MIN_SUCCESS_RATE = 0.50
NEW_PROPERTY_DEV_MIN_CONTROL_MARGIN = 0.25
NEW_PROPERTY_MIN_CLEAN_CAPABILITY_RATE = 0.75
NEW_PROPERTY_DEV_SEED = "multimodal-new-property-development-20260823-v1"
NEW_PROPERTY_CONFIRM_CANDIDATES = 64
NEW_PROPERTY_CONFIRM_IMAGES = 16
NEW_PROPERTY_CONFIRM_MIN_SUCCESS_RATE = 0.75
NEW_PROPERTY_CONFIRM_MIN_CONTROL_MARGIN = 0.25
NEW_PROPERTY_CONFIRM_FAMILYWISE_ALPHA = 0.05
NEW_PROPERTY_CONFIRM_SEED = "multimodal-new-property-confirmation-20260823-v1"
# Written by Stage 5B2 and read by Stage 5B3. Stage 5B3 refuses to open a
# fresh photograph until this file exists and verifies.
NEW_PROPERTY_FROZEN_DESIGN_PATH = None
NEW_PROPERTY_DEVELOPMENT_RUN_DIR = None
EXPECTED_NEW_PROPERTY_DEVELOPMENT_CHECKSUM = None

# Experiment C. Same leg-count protocol and endpoint as the confirmed
# bird->cat study, run backwards on fresh cat media.
ASYMMETRY_DIRECTION = ("cat", "bird")
ASYMMETRY_CANDIDATES = 64
ASYMMETRY_IMAGES = 16
ASYMMETRY_MIN_SUCCESS_RATE = 0.75
ASYMMETRY_MIN_CONTROL_MARGIN = 0.25
ASYMMETRY_FAMILYWISE_ALPHA = 0.05
ASYMMETRY_SEED = "multimodal-asymmetry-replication-20260823-v1"
# Alpha=1 is the exact exchange. The refinement grid brackets the strongest
# stable signal in the coarse 0.5/1/2/4 sweep and never enters the alpha>=2
# regime that already produced large activation-norm inflation. The grid was
# chosen after observing that coarse result, so it is explicitly exploratory
# and any selected alpha needs fresh-population confirmation.
ALPHA_SWEEP = (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)
ALPHA_SWEEP_OUTCOME_INFORMED = True
ALPHA_SELECTION_MAX_ACTIVATION_NORM_RATIO = 1.25
ALPHA_SELECTION_MAX_UPDATE_RATIO = 0.50

# The four completed lenses are imported read-only when Stage 3 is requested
# without Stage 1.  Every report and tensor checksum is pinned below.  Changing
# the fresh causal population creates a new run fingerprint but never triggers
# refitting.
CAUSAL_LENS_SOURCE_RUN_DIR = (
    "/content/drive/MyDrive/jacobian-lens-gemma/runs/mmjlens4/"
    "mmjlens4_real_1d3b1afbd019"
)
EXPECTED_SOURCE_FINAL_REPORT_CHECKSUM = (
    "sha256:875e13a8829bfd226c637ef4522d64d4d5ef91f31adcdace4942e72e75eb1e0e"
)
EXPECTED_SOURCE_CROSS_REPORT_CHECKSUM = (
    "sha256:a8536614f6e751e65ec250016852d6d614c0bc16befbfeb502e1faa148a3c69f"
)
EXPECTED_SOURCE_CAUSAL_REPORT_CHECKSUM = (
    "sha256:3370a2de8713024235b154ade3d7531eca491fea5592d9cf6b0397b434d573df"
)
EXPECTED_SOURCE_LENS_CHECKSUMS = {
    "text": "sha256:01c2591e55eda83fb17e784bb1e35fb437ee1ccf1ba556e95269c913b9596717",
    "image": "sha256:16f0a7c6dcbc36133ed28028016020cb7e8c8a8ec4c2879e283e191b04c1ef6d",
    "spoken_audio": "sha256:2f9140e28b2dd41b6f7e8e138ef0a11507d6013b1f4e95265d8e80e213936f55",
    "pooled": "sha256:7569552f1b9137ab859fe54e5d54395920c740fea94a909c8ef43623ddb5ea0e",
}

# The completed unrestricted alpha=1 run freezes the paired population for
# Stage 3B. It is read and checksum-verified; no photograph is re-recruited.
ALPHA1_CAUSAL_SOURCE_RUN_DIR = (
    "/content/drive/MyDrive/jacobian-lens-gemma/runs/mmjlens5causal/"
    "mmjlens5causal_real_5c7833b905c3"
)
EXPECTED_ALPHA1_FINAL_REPORT_CHECKSUM = (
    "sha256:0a59304f4cb464502e611845fc6cb6ed6fed18256b5946e15b05215427e6ac50"
)
EXPECTED_ALPHA1_CAUSAL_REPORT_CHECKSUM = (
    "sha256:fb27b51b7d88a763c0451bd298bf3258225d1741afb368255bbbf487aa2ef572"
)
EXPECTED_ALPHA1_SCIENTIFIC_FINGERPRINT = (
    "sha256:5c7833b905c3b32db0c8e78eae8ea6e432f86efd136391f02ab9047f59679dd2"
)

REAL_MODE = bool(RUN_REAL_MATCHED_JLENS)
if RUN_STAGE3_CAUSAL_COMPARE and RUN_STAGE3B_ALPHA_SWEEP:
    raise RuntimeError("Run Stage 3 or Stage 3B, never both in one session")
if RUN_STAGE3C_BROAD_POOLED_WORKSPACE and RUN_STAGE3D_FRESH_MULTIMODAL_CONFIRMATION:
    raise RuntimeError("Run Stage 3C development or Stage 3D confirmation, never both")
FOLLOWUP_STAGES = {
    "5A_band_localization": RUN_STAGE5A_BAND_LOCALIZATION,
    "5B0_property_audit": RUN_STAGE5B0_PROPERTY_AUDIT,
    "5B1_new_property_development": RUN_STAGE5B1_NEW_PROPERTY_DEVELOPMENT,
    "5B2_freeze": RUN_STAGE5B2_FREEZE_NEW_PROPERTY_DESIGN,
    "5B3_new_property_confirmation": RUN_STAGE5B3_NEW_PROPERTY_CONFIRMATION,
    "5C_asymmetry_replication": RUN_STAGE5C_ASYMMETRY_REPLICATION,
}
if sum(1 for value in FOLLOWUP_STAGES.values() if value) > 1:
    raise RuntimeError(
        "run exactly one follow-up stage per session; artifacts of different "
        f"stages are never mixed (requested {[k for k, v in FOLLOWUP_STAGES.items() if v]})"
    )
if any(FOLLOWUP_STAGES.values()) and any((
    RUN_STAGE1_FIT_LENSES, RUN_STAGE2_CROSS_EVALUATE, RUN_STAGE3_CAUSAL_COMPARE,
    RUN_STAGE3B_ALPHA_SWEEP, RUN_STAGE3C_BROAD_POOLED_WORKSPACE,
    RUN_STAGE3D_FRESH_MULTIMODAL_CONFIRMATION,
)):
    raise RuntimeError(
        "a follow-up stage never shares a session with the completed study "
        "stages; their reports are immutable evidence"
    )
MODEL_STAGE = any((
    RUN_STAGE1_FIT_LENSES, RUN_STAGE2_CROSS_EVALUATE,
    RUN_STAGE3_CAUSAL_COMPARE, RUN_STAGE3B_ALPHA_SWEEP,
    RUN_STAGE3C_BROAD_POOLED_WORKSPACE,
    RUN_STAGE3D_FRESH_MULTIMODAL_CONFIRMATION,
    RUN_STAGE5A_BAND_LOCALIZATION,
    RUN_STAGE5B0_PROPERTY_AUDIT,
    RUN_STAGE5B1_NEW_PROPERTY_DEVELOPMENT,
    RUN_STAGE5B3_NEW_PROPERTY_CONFIRMATION,
    RUN_STAGE5C_ASYMMETRY_REPLICATION,
))
MODEL_ENABLED = bool(MODEL_STAGE and CONFIRM_MODEL_LOAD)
FIT_ENABLED = bool(RUN_STAGE1_FIT_LENSES and MODEL_ENABLED and CONFIRM_FIT_BUDGET)
CROSS_ENABLED = bool(RUN_STAGE2_CROSS_EVALUATE and MODEL_ENABLED)
CAUSAL_ENABLED = bool(RUN_STAGE3_CAUSAL_COMPARE and MODEL_ENABLED and CONFIRM_CAUSAL_BUDGET)
ALPHA_SWEEP_ENABLED = bool(
    RUN_STAGE3B_ALPHA_SWEEP and MODEL_ENABLED and CONFIRM_ALPHA_SWEEP_BUDGET
)
BROAD_POOLED_ENABLED = bool(
    RUN_STAGE3C_BROAD_POOLED_WORKSPACE and MODEL_ENABLED
    and CONFIRM_BROAD_POOLED_FIT_BUDGET
    and CONFIRM_BROAD_POOLED_CAUSAL_BUDGET
)
FRESH_CONFIRMATION_ENABLED = bool(
    RUN_STAGE3D_FRESH_MULTIMODAL_CONFIRMATION and MODEL_ENABLED
    and CONFIRM_FRESH_MULTIMODAL_CONFIRMATION_BUDGET
)
if REAL_MODE and MODEL_STAGE and not MODEL_ENABLED:
    print("MODEL STAGES BLOCKED: set CONFIRM_MODEL_LOAD after reading the budget")
if REAL_MODE and RUN_STAGE1_FIT_LENSES and not FIT_ENABLED:
    print("FIT BLOCKED: set CONFIRM_FIT_BUDGET after reading the budget")
if REAL_MODE and RUN_STAGE3_CAUSAL_COMPARE and not CAUSAL_ENABLED:
    print("CAUSAL BLOCKED: set CONFIRM_CAUSAL_BUDGET after reading the budget")
if REAL_MODE and RUN_STAGE3B_ALPHA_SWEEP and not ALPHA_SWEEP_ENABLED:
    print("ALPHA SWEEP BLOCKED: set CONFIRM_ALPHA_SWEEP_BUDGET after reading the budget")
if REAL_MODE and RUN_STAGE3C_BROAD_POOLED_WORKSPACE and not BROAD_POOLED_ENABLED:
    print(
        "BROAD POOLED WORKSPACE BLOCKED: confirm both its fit and causal budgets"
    )
if REAL_MODE and RUN_STAGE3D_FRESH_MULTIMODAL_CONFIRMATION and not FRESH_CONFIRMATION_ENABLED:
    print("FRESH CONFIRMATION BLOCKED: confirm its printed budget")

LOCALIZATION_ENABLED = bool(
    RUN_STAGE5A_BAND_LOCALIZATION and MODEL_ENABLED and CONFIRM_LOCALIZATION_BUDGET
)
PROPERTY_AUDIT_ENABLED = bool(RUN_STAGE5B0_PROPERTY_AUDIT and MODEL_ENABLED)
NEW_PROPERTY_DEV_ENABLED = bool(
    RUN_STAGE5B1_NEW_PROPERTY_DEVELOPMENT and MODEL_ENABLED
    and CONFIRM_NEW_PROPERTY_DEVELOPMENT_BUDGET
)
NEW_PROPERTY_FREEZE_ENABLED = bool(RUN_STAGE5B2_FREEZE_NEW_PROPERTY_DESIGN)
NEW_PROPERTY_CONFIRM_ENABLED = bool(
    RUN_STAGE5B3_NEW_PROPERTY_CONFIRMATION and MODEL_ENABLED
    and CONFIRM_NEW_PROPERTY_CONFIRMATION_BUDGET
)
ASYMMETRY_ENABLED = bool(
    RUN_STAGE5C_ASYMMETRY_REPLICATION and MODEL_ENABLED and CONFIRM_ASYMMETRY_BUDGET
)
for _name, _requested, _enabled in (
    ("STAGE 5A LOCALIZATION", RUN_STAGE5A_BAND_LOCALIZATION, LOCALIZATION_ENABLED),
    ("STAGE 5B1 NEW-PROPERTY DEVELOPMENT", RUN_STAGE5B1_NEW_PROPERTY_DEVELOPMENT, NEW_PROPERTY_DEV_ENABLED),
    ("STAGE 5B3 NEW-PROPERTY CONFIRMATION", RUN_STAGE5B3_NEW_PROPERTY_CONFIRMATION, NEW_PROPERTY_CONFIRM_ENABLED),
    ("STAGE 5C ASYMMETRY REPLICATION", RUN_STAGE5C_ASYMMETRY_REPLICATION, ASYMMETRY_ENABLED),
):
    if REAL_MODE and _requested and not _enabled:
        print(f"{_name} BLOCKED: confirm its printed budget above")
'''
)

markdown("## 2. Paths and exact budgets (no model load)")
code(
    r'''
from jlens.mmpilot.multimodal_lens import fit_budget

if REAL_MODE:
    from google.colab import drive
    drive.mount("/content/drive")
    RUNS_ROOT = Path("/content/drive/MyDrive/jacobian-lens-gemma/runs")
    EXPANDED_MANIFEST_CACHE = (
        RUNS_ROOT / "mml32_l32_followup_20260808T182717" / "expanded_manifest.json"
    )
    IMAGE_MEDIA_ROOT = Path("/content/drive/MyDrive/datasets/cstf_spokencoco/coco")
else:
    RUNS_ROOT = Path(tempfile.gettempdir()) / "jlens_matched_mock_runs"
    EXPANDED_MANIFEST_CACHE = None
    IMAGE_MEDIA_ROOT = Path("/mock/coco")

BUDGET = fit_budget(
    n_fit_groups=N_FIT_GROUPS if REAL_MODE else 3,
    n_layers=len(SOURCE_LAYERS) if REAL_MODE else 2,
    d_model=EXPECT_D_MODEL if REAL_MODE else 12,
    dim_batch=DIM_BATCH if REAL_MODE else 4,
)
print("=" * 78)
print("FIT BUDGET — four separate, resumable arms")
print("=" * 78)
for _arm, _count in BUDGET["prompts_by_arm"].items():
    print(f"  {_arm:16s} {_count:>5} processor examples")
print("  forward passes  ", BUDGET["total_prompt_forwards"])
print("  backward passes ", BUDGET["total_backward_passes"])
print("  fitted layers   ", list(SOURCE_LAYERS))
print("  checkpoint every", CHECKPOINT_EVERY, "examples; at most that in-flight prefix is recomputed")
print()
print("CROSS-EVALUATION BUDGET")
print("  model forwards  ", N_CROSS_EVAL_GROUPS * 3 if REAL_MODE else 6)
print("  readouts         ", (N_CROSS_EVAL_GROUPS if REAL_MODE else 2) * 3 * 4 * len(SOURCE_LAYERS if REAL_MODE else (1, 2)))
print()
print("CAUSAL UPPER BOUND")
print("  clean screening ", N_CAUSAL_CANDIDATES_PER_CONCEPT * 2 * 3 * 2)
print("  exact/random/unrelated trials after recruitment",
      N_CAUSAL_IMAGES_PER_CELL * 2 * 3 * 2 * 2 * 3)
print()
_sweep_rows = N_CAUSAL_IMAGES_PER_CELL * 2 * 3 * 2
_sweep_conditions = 2 * len(ALPHA_SWEEP) * 3
print("STAGE 3B PAIRED ALPHA-SWEEP BUDGET")
print("  frozen clean-capable inputs", _sweep_rows)
print("  clean forwards            ", _sweep_rows)
print("  patched forwards          ", _sweep_rows * _sweep_conditions)
print("  total forwards            ", _sweep_rows * (1 + _sweep_conditions))
print("  alphas                    ", list(ALPHA_SWEEP))
print("  resume                    one arm x alpha x condition trial")
print("  scientific role           exploratory sensitivity; alpha=1 stays primary")
print()
_broad_directions = len(BROAD_POOLED_PAIRS)
_broad_cells = (
    BROAD_POOLED_IMAGES_PER_DIRECTION * _broad_directions * 3
)
print("STAGE 3C BROAD POOLED J-LENS BUDGET")
print("  method                    ordinary J-lens; no R-lens")
print("  fitted shard              pooled L16-L32 only")
print("  reused checksum-pinned    pooled L33-L40")
print("  fitting examples          ", N_FIT_GROUPS)
print("  fitting forward passes    ", N_FIT_GROUPS)
print("  fitting backward passes   ", N_FIT_GROUPS * ((EXPECT_D_MODEL + DIM_BATCH - 1) // DIM_BATCH))
print("  checkpoint                every", CHECKPOINT_EVERY, "examples")
print("  paper-depth band          ", list(BROAD_POOLED_BAND))
print("  directions                ", list(BROAD_POOLED_PAIRS))
print("  clean capability forwards ", BROAD_POOLED_CANDIDATES_PER_CONCEPT * len(BROAD_POOLED_CONCEPTS) * 3)
print("  causal rows               ", _broad_cells)
print("  patched forwards          ", _broad_cells * (1 + 3 * len(BROAD_POOLED_ALPHAS)))
print("  primary / sensitivity     alpha=1 / alpha=2")
print("  resume                    one atomic fit prefix; then one condition JSON")
print()
print("STAGE 3D FRESH CONFIRMATION BUDGET")
print("  fitting / backward passes 0 / 0")
print("  frozen direction          ", "->".join(CONFIRMATION_DIRECTION))
print("  frozen alpha / band       ", CONFIRMATION_ALPHA, list(BROAD_POOLED_BAND))
print("  fresh candidates          ", CONFIRMATION_CANDIDATES, "bird photographs")
print("  recruited units           ", CONFIRMATION_IMAGES, "photographs x 3 modalities")
print("  capability forwards       ", CONFIRMATION_CANDIDATES * 3)
print("  intervention forwards     ", CONFIRMATION_IMAGES * 3 * 4)
print("  conditions                exact, zero, random, unrelated")
print("  primary endpoint          unrestricted next-token top1 = 4")
print("  inference                 paired exact sign tests; Holm FWER", CONFIRMATION_FAMILYWISE_ALPHA)
print("  resume                    one checksum-valid JSON per completed forward")
print()

from jlens.mmpilot.multimodal_followup import (
    followup_budget, localization_budget, localization_grid, stage_map,
)

LOCALIZATION_GRID = localization_grid()
STAGE_MAP = stage_map()
print("=" * 78)
print("FOLLOW-UP STAGE MAP — nothing below fits or refits a lens")
print("=" * 78)
for _row in STAGE_MAP["stages"]:
    print(
        f"  {_row['stage']:<3} {_row['name']:<38} {_row['label']:<24}"
        f" confirms={_row['confirms']}"
    )
for _rule in STAGE_MAP["never"]:
    print("  never:", _rule)
print()
LOCALIZATION_BUDGET = localization_budget(
    grid=LOCALIZATION_GRID,
    n_photographs=BROAD_POOLED_IMAGES_PER_DIRECTION if REAL_MODE else 2,
)
print("STAGE 5A EXPLORATORY LOCALIZATION BUDGET")
print("  population                spent broad development photographs")
print("  new media opened          ", LOCALIZATION_BUDGET["new_media_opened"])
print("  lens fits / backward      ", LOCALIZATION_BUDGET["lens_fits"], "/",
      LOCALIZATION_BUDGET["backward_passes"])
print("  bands in the frozen grid  ", LOCALIZATION_BUDGET["n_bands"])
print("  band families             ", {k: len(v) for k, v in LOCALIZATION_GRID["families"].items()})
print("  clean forwards            ", LOCALIZATION_BUDGET["clean_forwards"])
print("  patched forwards          ", LOCALIZATION_BUDGET["patched_forwards"])
print("  TOTAL NEW MODEL FORWARDS  ", LOCALIZATION_BUDGET["total_forwards"])
print("  label                     exploratory/descriptive, never confirmation")
print()
NEW_PROPERTY_DEV_BUDGET = followup_budget(
    stage="5B0+5B1",
    n_candidates=(
        NEW_PROPERTY_DEV_CANDIDATES_PER_CONCEPT * len(NEW_PROPERTY_CONCEPTS)
        if REAL_MODE else 4
    ),
    n_recruited=NEW_PROPERTY_DEV_IMAGES_PER_DIRECTION if REAL_MODE else 2,
    max_new_tokens=NEW_PROPERTY_MAX_NEW_TOKENS,
    n_directions=len(NEW_PROPERTY_DEV_DIRECTIONS),
)
print("STAGE 5B0/5B1 NEW-PROPERTY AUDIT + DEVELOPMENT BUDGET")
print("  property family           ", NEW_PROPERTY_FAMILY, "(fallback",
      NEW_PROPERTY_FALLBACK_FAMILY + ")")
print("  endpoint                  unrestricted complete generation,",
      NEW_PROPERTY_MAX_NEW_TOKENS, "tokens")
print("  directions                ", [f"{a}->{b}" for a, b in NEW_PROPERTY_DEV_DIRECTIONS])
print("  capability forwards       ", NEW_PROPERTY_DEV_BUDGET["capability_forwards"])
print("  clean forwards            ", NEW_PROPERTY_DEV_BUDGET["clean_forwards"])
print("  patched forwards          ", NEW_PROPERTY_DEV_BUDGET["patched_forwards"])
print("  TOTAL NEW MODEL FORWARDS  ", NEW_PROPERTY_DEV_BUDGET["total_forwards"])
print("  lens fits / backward      ", NEW_PROPERTY_DEV_BUDGET["lens_fits"], "/",
      NEW_PROPERTY_DEV_BUDGET["backward_passes"])
print()
NEW_PROPERTY_CONFIRM_BUDGET = followup_budget(
    stage="5B3",
    n_candidates=NEW_PROPERTY_CONFIRM_CANDIDATES if REAL_MODE else 4,
    n_recruited=NEW_PROPERTY_CONFIRM_IMAGES if REAL_MODE else 2,
    max_new_tokens=NEW_PROPERTY_MAX_NEW_TOKENS,
)
ASYMMETRY_BUDGET = followup_budget(
    stage="5C",
    n_candidates=ASYMMETRY_CANDIDATES if REAL_MODE else 4,
    n_recruited=ASYMMETRY_IMAGES if REAL_MODE else 2,
    max_new_tokens=1,
)
for _label, _budget in (
    ("STAGE 5B3 NEW-PROPERTY FRESH CONFIRMATION BUDGET", NEW_PROPERTY_CONFIRM_BUDGET),
    ("STAGE 5C ASYMMETRY REPLICATION BUDGET", ASYMMETRY_BUDGET),
):
    print(_label)
    print("  capability forwards       ", _budget["capability_forwards"])
    print("  clean forwards            ", _budget["clean_forwards"])
    print("  patched forwards          ", _budget["patched_forwards"])
    print("  TOTAL NEW MODEL FORWARDS  ", _budget["total_forwards"])
    print("  lens fits / backward      ", _budget["lens_fits"], "/",
          _budget["backward_passes"])
    print("  resume                    ", _budget["resume_unit"])
    print()
'''
)

markdown("## 3. Load the synchronization cache and freeze all populations")
code(
    r'''
from jlens.mmpilot.multimodal_lens import (
    answer_equivalence_record, build_matched_plan,
    load_completed_alpha_sweep_source, load_completed_causal_source,
    select_causal_groups,
)
from jlens.mmpilot.store import payload_checksum

if REAL_MODE:
    if not EXPANDED_MANIFEST_CACHE.is_file():
        raise FileNotFoundError(
            f"Required cache not found: {EXPANDED_MANIFEST_CACHE}. This notebook "
            "never rebuilds the 125k-group join on GPU."
        )
    _bytes = EXPANDED_MANIFEST_CACHE.read_bytes()
    MANIFEST_CHECKSUM = "sha256:" + hashlib.sha256(_bytes).hexdigest()
    _payload = json.loads(_bytes)
    GROUPS = [dict(row) for row in _payload["groups"]]
    del _payload, _bytes
else:
    from jlens.mmpilot.mock import MockWorld
    _world = MockWorld()
    GROUPS = []
    for _index in range(18):
        _concept = None
        if 10 <= _index < 14:
            _concept = "bird"
        elif 14 <= _index:
            _concept = "cat"
        _caption = (
            f"A {_concept} in a field example {_index}"
            if _concept else f"A neutral landscape with clouds example {_index}"
        )
        GROUPS.append({
            "group_id": f"mock_g{_index}", "image_id": f"mock_i{_index}",
            "caption": _caption, "concept": _concept,
            "image_path": f"/mock/images/{_index}.jpg",
            "audio_path": f"/mock/audio/{_index}.wav",
        })
    MANIFEST_CHECKSUM = payload_checksum(GROUPS)

CAUSAL_SOURCE = None
ALPHA1_SOURCE = None
SOURCE_EXCLUDED_IMAGE_IDS = []
_use_completed_lenses = bool(
    REAL_MODE
    and (RUN_STAGE3_CAUSAL_COMPARE or RUN_STAGE3B_ALPHA_SWEEP)
    and not RUN_STAGE1_FIT_LENSES
)
if _use_completed_lenses:
    CAUSAL_SOURCE = load_completed_causal_source(
        CAUSAL_LENS_SOURCE_RUN_DIR,
        expected_final_report_checksum=EXPECTED_SOURCE_FINAL_REPORT_CHECKSUM,
        expected_cross_report_checksum=EXPECTED_SOURCE_CROSS_REPORT_CHECKSUM,
        expected_causal_report_checksum=EXPECTED_SOURCE_CAUSAL_REPORT_CHECKSUM,
        expected_lens_checksums=EXPECTED_SOURCE_LENS_CHECKSUMS,
    )
    SOURCE_EXCLUDED_IMAGE_IDS = list(CAUSAL_SOURCE["excluded_image_ids"])
    print("completed lens source", CAUSAL_SOURCE["run_dir"])
    print("source digest", CAUSAL_SOURCE["source_digest"])
    print("previously screened images excluded", len(SOURCE_EXCLUDED_IMAGE_IDS))
if REAL_MODE and RUN_STAGE3B_ALPHA_SWEEP:
    if CAUSAL_SOURCE is None:
        raise RuntimeError("Stage 3B requires the checksum-pinned four-lens source")
    ALPHA1_SOURCE = load_completed_alpha_sweep_source(
        ALPHA1_CAUSAL_SOURCE_RUN_DIR,
        expected_final_report_checksum=EXPECTED_ALPHA1_FINAL_REPORT_CHECKSUM,
        expected_causal_report_checksum=EXPECTED_ALPHA1_CAUSAL_REPORT_CHECKSUM,
        expected_scientific_fingerprint=EXPECTED_ALPHA1_SCIENTIFIC_FINGERPRINT,
        expected_lens_checksums=EXPECTED_SOURCE_LENS_CHECKSUMS,
        expected_lens_source_digest=CAUSAL_SOURCE["source_digest"],
    )
    print("completed alpha=1 population", ALPHA1_SOURCE["run_dir"])
    print("alpha=1 source digest", ALPHA1_SOURCE["source_digest"])

ANSWER_EQUIVALENCE = answer_equivalence_record()
print("answer equivalence", ANSWER_EQUIVALENCE)

_fit_n = N_FIT_GROUPS if REAL_MODE else 3
_eval_n = N_CROSS_EVAL_GROUPS if REAL_MODE else 2
PLAN = build_matched_plan(
    GROUPS, n_fit_groups=_fit_n, n_eval_groups=_eval_n,
    seed=PLAN_SEED, excluded_eval_concepts=(*EVAL_CONCEPTS, *CONTROL_CONCEPTS),
)
CAUSAL_POPULATION = select_causal_groups(
    GROUPS, concepts=EVAL_CONCEPTS,
    n_per_concept=N_CAUSAL_CANDIDATES_PER_CONCEPT if REAL_MODE else 3,
    excluded_image_ids=(
        *PLAN["fit_image_ids"], *PLAN["eval_image_ids"],
        *SOURCE_EXCLUDED_IMAGE_IDS,
    ),
    seed=CAUSAL_SEED,
)
CAUSAL_POPULATION_DIGEST = payload_checksum(CAUSAL_POPULATION)
SWEEP_POPULATION = None
SWEEP_POPULATION_DIGEST = None
if ALPHA1_SOURCE is not None:
    _groups_by_id = {str(row["group_id"]): row for row in GROUPS}
    SWEEP_POPULATION = {}
    for _source, _identities in ALPHA1_SOURCE["groups_by_source"].items():
        SWEEP_POPULATION[_source] = []
        for _identity in _identities:
            _group_id = str(_identity["group_id"])
            if _group_id not in _groups_by_id:
                raise RuntimeError(f"pinned alpha-sweep group {_group_id!r} is absent")
            _group = _groups_by_id[_group_id]
            if str(_group["image_id"]) != str(_identity["image_id"]):
                raise RuntimeError(f"pinned alpha-sweep group {_group_id!r} changed image")
            SWEEP_POPULATION[_source].append(_group)
    if {name: len(rows) for name, rows in SWEEP_POPULATION.items()} != {
        "bird": 8, "cat": 8,
    }:
        raise RuntimeError("the pinned alpha-sweep population is incomplete")
    SWEEP_POPULATION_DIGEST = payload_checksum([
        {"source": source, "group_id": row["group_id"], "image_id": row["image_id"]}
        for source in EVAL_CONCEPTS for row in SWEEP_POPULATION[source]
    ])
print("manifest", MANIFEST_CHECKSUM, "groups", len(GROUPS))
print("fit images", len(PLAN["fit_image_ids"]), "eval images", len(PLAN["eval_image_ids"]))
print("fit/eval overlap", PLAN["fit_eval_image_overlap"])
print("plan", PLAN["plan_digest"])
print("causal", CAUSAL_POPULATION_DIGEST,
      {name: len(rows) for name, rows in CAUSAL_POPULATION.items()})

SCIENTIFIC_CONFIG = {
    "study": (
        "matched_multimodal_jlens.alpha_refinement_0125_to_1.v3"
        if ALPHA1_SOURCE is not None
        else "matched_multimodal_jlens.causal_followup.v1"
        if _use_completed_lenses else "matched_multimodal_jlens.v4"
    ),
    "model_repo_id": MODEL_REPO_ID,
    "model_revision": MODEL_REVISION,
    "audio_protocol_fingerprint": AUDIO_PROTOCOL_FINGERPRINT,
    "manifest_checksum": MANIFEST_CHECKSUM,
    "plan_digest": PLAN["plan_digest"],
    "causal_population_digest": (
        SWEEP_POPULATION_DIGEST or CAUSAL_POPULATION_DIGEST
    ),
    "causal_source_digest": (
        CAUSAL_SOURCE["source_digest"] if CAUSAL_SOURCE else None
    ),
    "alpha1_population_source_digest": (
        ALPHA1_SOURCE["source_digest"] if ALPHA1_SOURCE else None
    ),
    "answer_equivalence": ANSWER_EQUIVALENCE,
    "n_causal_candidates_per_concept": N_CAUSAL_CANDIDATES_PER_CONCEPT,
    "source_layers": list(SOURCE_LAYERS if REAL_MODE else (1, 2)),
    "target_layer": TARGET_LAYER if REAL_MODE else 3,
    "dim_batch": DIM_BATCH if REAL_MODE else 4,
    "skip_first": SKIP_FIRST if REAL_MODE else 2,
    "primary_alpha": PRIMARY_ALPHA,
    "alpha_sweep": list(ALPHA_SWEEP) if ALPHA1_SOURCE else [PRIMARY_ALPHA],
    "alpha_roles": {
        f"{alpha:g}": (
            "primary_exact_exchange"
            if alpha == 1.0
            else "outcome_informed_stable_range_refinement"
        )
        for alpha in (ALPHA_SWEEP if ALPHA1_SOURCE else (PRIMARY_ALPHA,))
    },
    "alpha_grid_selected_after_coarse_sweep": bool(
        ALPHA1_SOURCE and ALPHA_SWEEP_OUTCOME_INFORMED
    ),
    "causal_protocol": (
        "matched_multimodal_jlens_unrestricted_alpha_refinement.v6"
        if ALPHA1_SOURCE else "matched_multimodal_jlens_unrestricted_swap.v3"
    ),
    "clean_recruitment": "all_modalities_x_identity_and_property",
    "causal_controls": ["random", "unrelated"],
    "causal_concepts": list(EVAL_CONCEPTS),
    "control_concepts": list(CONTROL_CONCEPTS),
    "commit": COMMIT,
}
SCIENTIFIC_FINGERPRINT = payload_checksum(SCIENTIFIC_CONFIG)
_run_family = (
    "mmjlens6alpha" if ALPHA1_SOURCE is not None
    else "mmjlens5causal" if _use_completed_lenses else "mmjlens4"
)
RUN_DIR = RUNS_ROOT / _run_family / f"{_run_family}_{'real' if REAL_MODE else 'mock'}_{SCIENTIFIC_FINGERPRINT.split(':')[1][:12]}"
RUN_DIR.mkdir(parents=True, exist_ok=True)
_plan_path = RUN_DIR / "matched_population_plan.json"
if _plan_path.is_file():
    _stored = json.loads(_plan_path.read_text())
    if _stored.get("plan_digest") != PLAN["plan_digest"]:
        raise RuntimeError("run directory holds a different population plan")
else:
    _plan_path.write_text(json.dumps(PLAN, indent=2, default=str))
if CAUSAL_SOURCE is not None:
    (RUN_DIR / "causal_source_provenance.json").write_text(
        json.dumps(CAUSAL_SOURCE, indent=2, default=str)
    )
if ALPHA1_SOURCE is not None:
    (RUN_DIR / "alpha1_population_source.json").write_text(
        json.dumps(ALPHA1_SOURCE, indent=2, default=str)
    )
print("run", RUN_DIR)
print("fingerprint", SCIENTIFIC_FINGERPRINT)
'''
)

markdown("## 4. Load the pinned model and audited native processor")
code(
    r'''
BACKEND = BUNDLE = AUDIO_RECORD = None
if REAL_MODE and MODEL_ENABLED:
    import getpass
    if not os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = getpass.getpass("HF_TOKEN (input hidden): ").strip()
    from jlens.mmpilot.real_backend import build_real_backend
    from jlens.mmpilot.tri_modal import assert_audio_protocol
    BUNDLE = build_real_backend(
        MODEL_REPO_ID, revision=MODEL_REVISION, token=os.environ["HF_TOKEN"],
        device="cuda", allow_model_load=True, resolve_audio=True,
        expect_n_layers=EXPECT_N_LAYERS, expect_d_model=EXPECT_D_MODEL,
        expect_vocab_size=EXPECT_VOCAB,
    )
    if BUNDLE.audio_interface is None:
        raise RuntimeError("native spoken audio did not resolve: " + BUNDLE.audio_blocked_reason)
    AUDIO_RECORD = assert_audio_protocol(
        BUNDLE.audio_interface, expected_fingerprint=AUDIO_PROTOCOL_FINGERPRINT
    )
    BACKEND = BUNDLE.backend
elif not REAL_MODE:
    from jlens.mmpilot.mock import MockPilotBackend, MockWorld
    BACKEND = MockPilotBackend(MockWorld(), n_layers=4)
    AUDIO_RECORD = {"protocol_fingerprint": "mock-audio"}
elif MODEL_STAGE:
    print("skipped: model confirmation is false")
'''
)

markdown("## 5. Media loading and processor-input construction")
code(
    r'''
from jlens.mmpilot.media_io import RetryJournal, drive_media_loaders

if REAL_MODE:
    MEDIA = drive_media_loaders(journal=RetryJournal())
else:
    MEDIA = None

def _mock_evidence(group, modality):
    concepts = tuple(
        name for name in (*EVAL_CONCEPTS, *CONTROL_CONCEPTS)
        if name in str(group.get("caption", "")).lower().split()
    )
    return BACKEND.world.evidence(
        concepts_present=concepts, modality=modality,
        nuisance_key=f"{group['group_id']}|{modality}",
    )

def build_group_inputs(group, modality, prompt):
    if not REAL_MODE:
        evidence = _mock_evidence(group, modality)
        kwargs = {"prompt": prompt, "modality": modality}
        if modality == "image": kwargs["image"] = evidence
        if modality == "spoken_audio": kwargs["audio"] = evidence
        return BACKEND.build_inputs(**kwargs)
    if modality == "text":
        return BACKEND.build_inputs(prompt=prompt, modality="text")
    if modality == "image":
        return BACKEND.build_inputs(
            prompt=prompt, modality="image",
            image=MEDIA["load_image"](group["image_path"]),
            media_path=group["image_path"],
        )
    waveform, rate = MEDIA["load_audio"](group["audio_path"])
    return BACKEND.build_inputs(
        prompt=prompt, modality="spoken_audio", audio=waveform,
        sampling_rate=rate, media_path=group["audio_path"],
    )

def build_fit_inputs(unit):
    group = {
        "group_id": unit.group_id, "caption": unit.caption,
        "image_path": unit.image_path, "audio_path": unit.audio_path,
    }
    return build_group_inputs(group, unit.modality, unit.prompt)
'''
)

markdown("## 6. Stage 1 — fit the four lenses")
code(
    r'''
from jlens.lens import JacobianLens
from jlens.metadata import file_sha256
from jlens.mmpilot.multimodal_lens import LENS_ARMS, fit_arm, plan_units

LENSES, LENS_CHECKSUMS = {}, {}
_layers = SOURCE_LAYERS if REAL_MODE else (1, 2)
_target = TARGET_LAYER if REAL_MODE else 3
_dim_batch = DIM_BATCH if REAL_MODE else 4
_skip = SKIP_FIRST if REAL_MODE else 2
_fit_requested = FIT_ENABLED if REAL_MODE else True

def progress(row):
    if row["index"] == 1 or row["checkpoint_written"] or row["index"] == row["total"]:
        print(f"{row['arm']:16s} {row['index']:>4}/{row['total']} "
              f"{row['modality']:13s} {row['elapsed_seconds']:.1f}s "
              f"checkpoint={row['checkpoint_written']}")

for _arm in LENS_ARMS:
    _lens_path = RUN_DIR / "lenses" / f"lens.{_arm}.pt"
    _checkpoint = RUN_DIR / "lenses" / "checkpoints" / f"{_arm}.jacobian_sum.pt"
    if _lens_path.is_file():
        LENSES[_arm] = JacobianLens.load(str(_lens_path))
        print(_arm, "reused completed lens", _lens_path)
    elif CAUSAL_SOURCE is not None:
        _source_path = Path(CAUSAL_SOURCE["lens_paths"][_arm])
        LENSES[_arm] = JacobianLens.load(str(_source_path))
        print(_arm, "imported read-only", _source_path)
    elif _fit_requested:
        _units = plan_units(PLAN, _arm)
        LENSES[_arm] = fit_arm(
            BACKEND, _units, build_inputs=build_fit_inputs,
            source_layers=_layers, target_layer=_target,
            checkpoint_path=_checkpoint, arm=_arm,
            scientific_fingerprint=SCIENTIFIC_FINGERPRINT,
            dim_batch=_dim_batch, skip_first=_skip,
            checkpoint_every=CHECKPOINT_EVERY if REAL_MODE else 1,
            progress=progress,
        )
        _lens_path.parent.mkdir(parents=True, exist_ok=True)
        _temporary = _lens_path.with_suffix(".tmp.pt")
        LENSES[_arm].save(str(_temporary))
        os.replace(_temporary, _lens_path)
        print(_arm, "completed", LENSES[_arm].n_prompts, "units")
    if CAUSAL_SOURCE is not None:
        LENS_CHECKSUMS[_arm] = CAUSAL_SOURCE["lens_checksums"][_arm]
    elif _lens_path.is_file():
        LENS_CHECKSUMS[_arm] = file_sha256(str(_lens_path))

if len(LENSES) != 4 and (
    RUN_STAGE2_CROSS_EVALUATE or RUN_STAGE3_CAUSAL_COMPARE
    or RUN_STAGE3B_ALPHA_SWEEP or not REAL_MODE
):
    raise RuntimeError(
        f"stages 2-3 require all four lenses; available {sorted(LENSES)}. "
        "Finish Stage 1 first; checkpoints resume automatically."
    )
print("lens checksums", json.dumps(LENS_CHECKSUMS, indent=2))
'''
)

markdown("## 7. Open the fingerprinted unit store")
code(
    r'''
from jlens.mmpilot.store import RunFingerprint, UnitStore

FINGERPRINT = RunFingerprint(
    mode="real" if REAL_MODE else "mock",
    model_repo_id=MODEL_REPO_ID,
    model_revision=MODEL_REVISION,
    processor_revision=MODEL_REVISION,
    layers=tuple(_layers),
    lens_checksum=payload_checksum(LENS_CHECKSUMS),
    manifest_checksum=MANIFEST_CHECKSUM,
    split_id=PLAN["plan_digest"],
    intervention_config={
        "alphas": list(ALPHA_SWEEP) if ALPHA1_SOURCE else [PRIMARY_ALPHA],
        "primary_alpha": PRIMARY_ALPHA, "layers": list(_layers),
        "position_rule": "all_prompt_positions",
        "teacher_forcing": False, "candidate_list": False,
    },
    extra={
        "study_fingerprint": SCIENTIFIC_FINGERPRINT,
        "audio_protocol_fingerprint": AUDIO_PROTOCOL_FINGERPRINT if REAL_MODE else "mock-audio",
        "causal_population_digest": (
            SWEEP_POPULATION_DIGEST or CAUSAL_POPULATION_DIGEST
        ),
        "alpha1_population_source_digest": (
            ALPHA1_SOURCE["source_digest"] if ALPHA1_SOURCE else None
        ),
    },
)
STORE = UnitStore(RUN_DIR, FINGERPRINT)
print("run state", STORE.open())
print("unit fingerprint", FINGERPRINT.digest)
'''
)

markdown("## 8. Stage 2 — 4 x 3 full-vocabulary cross-evaluation")
code(
    r'''
from jlens.mmpilot.multimodal_lens import capture_eval_rows, summarize_cross_eval
from jlens.mmpilot.store import safe_key

CROSS_ROWS = []
_cross_requested = CROSS_ENABLED if REAL_MODE else True
if _cross_requested:
    for _index, _group in enumerate(PLAN["eval_groups"], 1):
        _key = safe_key("cross_eval", _group["group_id"])
        _stored = STORE.load("metric", _key)
        if _stored is None:
            _rows = capture_eval_rows(
                BACKEND, LENSES, [_group], build_inputs=build_group_inputs,
                layers=_layers,
            )
            STORE.save("metric", _key, {"rows": _rows})
            _stored = {"rows": _rows}
            _work = "computed"
        else:
            _work = "reused"
        CROSS_ROWS.extend(_stored["rows"])
        if _index == 1 or _index % 8 == 0 or _index == len(PLAN["eval_groups"]):
            print(f"cross-eval {_index}/{len(PLAN['eval_groups'])} {_work}")
    CROSS_REPORT = summarize_cross_eval(CROSS_ROWS)
    STORE.save("metric", "cross_eval_report", CROSS_REPORT)
    (RUN_DIR / "multimodal_lens_cross_eval_report.json").write_text(
        json.dumps(CROSS_REPORT, indent=2, default=str)
    )
else:
    CROSS_REPORT = STORE.load("metric", "cross_eval_report")
    if CROSS_REPORT is None and CAUSAL_SOURCE is not None:
        CROSS_REPORT = json.loads(
            Path(CAUSAL_SOURCE["cross_report_path"]).read_text(encoding="utf-8")
        )
        print("cross-evaluation imported read-only from completed source run")

if CROSS_REPORT:
    print("=" * 96)
    print("CROSS-EVALUATION — native unrestricted next-token fidelity")
    print("=" * 96)
    print(f"{'arm':16s} {'test':14s} {'L':>3s} {'top1':>7s} {'MRR':>7s} {'median':>8s}")
    for _row in CROSS_REPORT["cells"]:
        print(f"{_row['lens_arm']:16s} {_row['test_modality']:14s} "
              f"{_row['layer']:3d} {_row['top1_agreement']:7.3f} "
              f"{_row['mrr']:7.3f} {_row['median_midrank']:8.1f}")
    print("report", RUN_DIR / "multimodal_lens_cross_eval_report.json")
'''
)

markdown("## 9. Stage 3 — unrestricted exact-swap causal comparison")
code(
    r'''
from jlens.mmpilot.coordinate_swap import (
    random_two_direction_basis, resolve_concept_token,
)
from jlens.mmpilot.digit_reasoning_confirmation import resolve_digit_endpoints
from jlens.mmpilot.multimodal_lens import (
    build_swap_bases_for_lens, open_answer_matches,
    unrestricted_swap_trial,
)

CAUSAL_REPORT = None
_causal_requested = CAUSAL_ENABLED if REAL_MODE else True
if _causal_requested:
    _encode = BACKEND.encode_candidate if REAL_MODE else BACKEND.encode_token
    CONCEPT_TOKENS = {name: resolve_concept_token(_encode, name) for name in (*EVAL_CONCEPTS, *CONTROL_CONCEPTS)}
    DIGITS = (
        resolve_digit_endpoints(BACKEND)
        if REAL_MODE
        else {"token_ids": {
            "2": CONCEPT_TOKENS["bird"].token_id,
            "4": CONCEPT_TOKENS["cat"].token_id,
        }}
    )
    _answers = {"bird": "2", "cat": "4"}
    _bases = {}
    _unrelated_bases = {}
    for _arm in ("text", "pooled"):
        for _source, _target_name in (("bird", "cat"), ("cat", "bird")):
            _bases[(_arm, _source, _target_name)] = build_swap_bases_for_lens(
                LENSES[_arm], BACKEND.unembedding_weight(), layers=_layers,
                source=CONCEPT_TOKENS[_source], target=CONCEPT_TOKENS[_target_name],
            )
        _unrelated_bases[_arm] = build_swap_bases_for_lens(
            LENSES[_arm], BACKEND.unembedding_weight(), layers=_layers,
            source=CONCEPT_TOKENS[CONTROL_CONCEPTS[0]],
            target=CONCEPT_TOKENS[CONTROL_CONCEPTS[1]],
        )

    def _prompt(kind, modality, caption):
        question = (
            "What animal is present in the evidence? Answer with the animal name.\nAnswer:"
            if kind == "identity" else
            "How many legs does the animal in the evidence typically have? Answer with one digit.\nAnswer:"
        )
        return f"Caption: {caption}\n{question}" if modality == "text" else question

    # Recruit only photographs the clean model answers correctly for both
    # open endpoints in every modality.  Screening is saved separately, so a
    # disconnect never repeats a completed clean forward pass.
    _clean_rows = []
    for _source, _target_name in (("bird", "cat"), ("cat", "bird")):
        for _group in CAUSAL_POPULATION[_source]:
            for _modality in ("text", "image", "spoken_audio"):
                for _kind in ("identity", "property"):
                    _key = safe_key(
                        "causal_clean", _source, _group["group_id"],
                        _modality, _kind,
                    )
                    _stored = STORE.load("capability", _key)
                    if _stored is None:
                        _inputs = build_group_inputs(
                            _group, _modality, _prompt(_kind, _modality, _group["caption"])
                        )
                        _clean_logits = BACKEND.forward_logits(_inputs.tensors)[
                            0, _inputs.final_prompt_position
                        ].float()
                        _expected = (
                            CONCEPT_TOKENS[_source].token_id
                            if _kind == "identity"
                            else DIGITS["token_ids"][_answers[_source]]
                        )
                        _expected_surface = (
                            _source if _kind == "identity" else _answers[_source]
                        )
                        _clean_top_token_id = int(_clean_logits.argmax())
                        _clean_surface = BACKEND.decode_token(
                            _clean_top_token_id
                        ).strip()
                        _stored = {
                            "source": _source,
                            "group_id": _group["group_id"], "image_id": _group["image_id"],
                            "modality": _modality, "prompt_kind": _kind,
                            "clean_top_token_id": _clean_top_token_id,
                            "clean_surface": _clean_surface,
                            "expected_source_token_id": int(_expected),
                            "expected_surface": _expected_surface,
                            "answer_equivalence_version": ANSWER_EQUIVALENCE["version"],
                            "clean_success": (
                                open_answer_matches(
                                    _clean_surface, _expected_surface
                                )
                                if REAL_MODE else True
                            ),
                        }
                        STORE.save("capability", _key, _stored)
                        _work = "computed"
                    else:
                        _work = "reused"
                    _clean_rows.append(_stored)
                    if len(_clean_rows) == 1 or len(_clean_rows) % 24 == 0:
                        print("clean screen", len(_clean_rows), _work)

    _recruited = {}
    _required_causal_images = N_CAUSAL_IMAGES_PER_CELL if REAL_MODE else 2
    for _source in EVAL_CONCEPTS:
        _eligible = []
        for _group in CAUSAL_POPULATION[_source]:
            _group_rows = [
                row for row in _clean_rows
                if row["source"] == _source
                and row["group_id"] == _group["group_id"]
            ]
            if len(_group_rows) == 6 and all(row["clean_success"] for row in _group_rows):
                _eligible.append(_group)
        _recruited[_source] = _eligible[:_required_causal_images]
    _capability_ok = all(
        len(_recruited[name]) == _required_causal_images for name in EVAL_CONCEPTS
    )
    print("recruited", {name: len(rows) for name, rows in _recruited.items()})

    _rows = []
    if _capability_ok:
        for _source, _target_name in (("bird", "cat"), ("cat", "bird")):
            for _group in _recruited[_source]:
                for _modality in ("text", "image", "spoken_audio"):
                    for _kind in ("identity", "property"):
                        _key = safe_key(
                            "causal", _source, _target_name, _group["group_id"],
                            _modality, _kind,
                        )
                        _stored = STORE.load("intervention", _key)
                        if _stored is None:
                            _inputs = build_group_inputs(
                                _group, _modality,
                                _prompt(_kind, _modality, _group["caption"]),
                            )
                            _expected = (
                                CONCEPT_TOKENS[_target_name].token_id
                                if _kind == "identity"
                                else DIGITS["token_ids"][_answers[_target_name]]
                            )
                            _expected_surface = (
                                _target_name
                                if _kind == "identity"
                                else _answers[_target_name]
                            )
                            _record = {
                                "source": _source, "target": _target_name,
                                "group_id": _group["group_id"],
                                "image_id": _group["image_id"],
                                "modality": _modality, "prompt_kind": _kind,
                                "expected_token_id": int(_expected),
                                "expected_surface": _expected_surface,
                                "answer_equivalence_version": ANSWER_EQUIVALENCE["version"],
                                "arms": {},
                            }
                            for _arm in ("text", "pooled"):
                                _exact_bases = _bases[(_arm, _source, _target_name)]
                                _condition_bases = {
                                    "exact": _exact_bases,
                                    "random": {
                                        layer: random_two_direction_basis(
                                            basis,
                                            seed=(20260819 + layer),
                                        )
                                        for layer, basis in _exact_bases.items()
                                    },
                                    "unrelated": _unrelated_bases[_arm],
                                }
                                _record["arms"][_arm] = {}
                                for _condition, _condition_basis in _condition_bases.items():
                                    _trial = unrestricted_swap_trial(
                                        BACKEND, _inputs, bases=_condition_basis,
                                        alpha=PRIMARY_ALPHA,
                                    )
                                    _patched_surface = BACKEND.decode_token(
                                        _trial["patched_top_token_id"]
                                    ).strip()
                                    _record["arms"][_arm][_condition] = {
                                        **_trial,
                                        "patched_surface": _patched_surface,
                                        "success": open_answer_matches(
                                            _patched_surface, _expected_surface
                                        ),
                                    }
                            STORE.save("intervention", _key, _record)
                            _stored, _work = _record, "computed"
                        else:
                            _work = "reused"
                        _rows.append(_stored)
                        if len(_rows) == 1 or len(_rows) % 12 == 0:
                            print("causal", len(_rows), _work)

    _cells = []
    if _capability_ok:
        for _arm in ("text", "pooled"):
            for _source, _target_name in (("bird", "cat"), ("cat", "bird")):
                for _kind in ("identity", "property"):
                    for _modality in ("text", "image", "spoken_audio"):
                        _selected = [
                            row for row in _rows
                            if row["source"] == _source
                            and row["prompt_kind"] == _kind
                            and row["modality"] == _modality
                        ]
                        _cells.append({
                            "lens_arm": _arm,
                            "direction": f"{_source}->{_target_name}",
                            "prompt_kind": _kind, "modality": _modality,
                            "n": len(_selected),
                            **{
                                f"{condition}_success_rate": sum(
                                    row["arms"][_arm][condition]["success"]
                                    for row in _selected
                                ) / len(_selected)
                                for condition in ("exact", "random", "unrelated")
                            },
                        })
    CAUSAL_REPORT = {
        "protocol": "matched_multimodal_jlens_unrestricted_swap.v3",
        "verdict": (
            "MEASURED" if _capability_ok else "CAPABILITY_NO_GO"
        ),
        "primary_alpha": PRIMARY_ALPHA,
        "teacher_forcing_used": False,
        "candidate_list_supplied": False,
        "answer_equivalence": ANSWER_EQUIVALENCE,
        "source_run_provenance": CAUSAL_SOURCE,
        "fresh_population": {
            "candidate_count_per_concept": N_CAUSAL_CANDIDATES_PER_CONCEPT,
            "excluded_previous_screen_images": len(SOURCE_EXCLUDED_IMAGE_IDS),
            "causal_population_digest": CAUSAL_POPULATION_DIGEST,
        },
        "clean_capability_required_in_every_modality_and_endpoint": True,
        "recruited_counts": {
            name: len(rows) for name, rows in _recruited.items()
        },
        "arms_compared": ["text", "pooled"],
        "controls": ["random", "unrelated"],
        "cells": _cells,
        "clean_screen": _clean_rows,
        "rows": _rows,
    }
    CAUSAL_REPORT["report_checksum"] = payload_checksum(CAUSAL_REPORT)
    STORE.save("metric", "causal_report", CAUSAL_REPORT)
    (RUN_DIR / "multimodal_lens_causal_comparison_report.json").write_text(
        json.dumps(CAUSAL_REPORT, indent=2, default=str)
    )
else:
    CAUSAL_REPORT = STORE.load("metric", "causal_report")

if CAUSAL_REPORT:
    print("=" * 86)
    print("UNRESTRICTED EXACT-SWAP COMPARISON")
    print("=" * 86)
    for _cell in CAUSAL_REPORT["cells"]:
        print(f"{_cell['lens_arm']:8s} {_cell['direction']:10s} "
              f"{_cell['prompt_kind']:8s} {_cell['modality']:13s} "
              f"exact={_cell['exact_success_rate']:.3f} "
              f"random={_cell['random_success_rate']:.3f} "
              f"unrelated={_cell['unrelated_success_rate']:.3f} n={_cell['n']}")
    print("report", RUN_DIR / "multimodal_lens_causal_comparison_report.json")
'''
)

markdown("## 10. Stage 3B — paired alpha dose-response on the frozen population")
code(
    r'''
ALPHA_SWEEP_REPORT = None
if REAL_MODE and ALPHA_SWEEP_ENABLED:
    from jlens.mmpilot.coordinate_swap import (
        METHOD_VERSION as COORDINATE_SWAP_METHOD_VERSION,
        random_two_direction_basis,
        resolve_concept_token,
    )
    from jlens.mmpilot.digit_reasoning_confirmation import resolve_digit_endpoints
    from jlens.mmpilot.multimodal_lens import (
        build_swap_bases_for_lens, open_answer_matches,
        unrestricted_swap_trial,
    )
    from jlens.mmpilot.store import safe_key

    if ALPHA1_SOURCE is None or SWEEP_POPULATION is None:
        raise RuntimeError("Stage 3B requires the verified completed alpha=1 population")
    _encode = BACKEND.encode_candidate
    CONCEPT_TOKENS = {
        name: resolve_concept_token(_encode, name)
        for name in (*EVAL_CONCEPTS, *CONTROL_CONCEPTS)
    }
    DIGITS = resolve_digit_endpoints(BACKEND)
    _answers = {"bird": "2", "cat": "4"}
    _sweep_bases, _sweep_random, _sweep_unrelated = {}, {}, {}
    for _arm in ("text", "pooled"):
        for _source, _target_name in (("bird", "cat"), ("cat", "bird")):
            _exact = build_swap_bases_for_lens(
                LENSES[_arm], BACKEND.unembedding_weight(), layers=_layers,
                source=CONCEPT_TOKENS[_source],
                target=CONCEPT_TOKENS[_target_name],
            )
            _sweep_bases[(_arm, _source, _target_name)] = _exact
            _sweep_random[(_arm, _source, _target_name)] = {
                layer: random_two_direction_basis(
                    basis, seed=(20260820 + layer)
                )
                for layer, basis in _exact.items()
            }
        _sweep_unrelated[_arm] = build_swap_bases_for_lens(
            LENSES[_arm], BACKEND.unembedding_weight(), layers=_layers,
            source=CONCEPT_TOKENS[CONTROL_CONCEPTS[0]],
            target=CONCEPT_TOKENS[CONTROL_CONCEPTS[1]],
        )

    def _sweep_prompt(kind, modality, caption):
        question = (
            "What animal is present in the evidence? Answer with the animal name.\nAnswer:"
            if kind == "identity" else
            "How many legs does the animal in the evidence typically have? Answer with one digit.\nAnswer:"
        )
        return f"Caption: {caption}\n{question}" if modality == "text" else question

    _sweep_rows = []
    _trial_counter = 0
    _computed_counter = 0
    _reused_counter = 0
    for _source, _target_name in (("bird", "cat"), ("cat", "bird")):
        for _group in SWEEP_POPULATION[_source]:
            for _modality in ("text", "image", "spoken_audio"):
                for _kind in ("identity", "property"):
                    _expected = (
                        CONCEPT_TOKENS[_target_name].token_id
                        if _kind == "identity"
                        else DIGITS["token_ids"][_answers[_target_name]]
                    )
                    _expected_surface = (
                        _target_name if _kind == "identity" else _answers[_target_name]
                    )
                    _source_token = (
                        CONCEPT_TOKENS[_source].token_id
                        if _kind == "identity"
                        else DIGITS["token_ids"][_answers[_source]]
                    )
                    _record = {
                        "source": _source, "target": _target_name,
                        "group_id": _group["group_id"],
                        "image_id": _group["image_id"],
                        "modality": _modality, "prompt_kind": _kind,
                        "expected_token_id": int(_expected),
                        "expected_surface": _expected_surface,
                        "arms": {},
                    }
                    _specs = []
                    for _arm in ("text", "pooled"):
                        _record["arms"][_arm] = {}
                        for _alpha in ALPHA_SWEEP:
                            _alpha_key = f"a{_alpha:g}"
                            _record["arms"][_arm][_alpha_key] = {}
                            _conditions = {
                                "exact": _sweep_bases[(_arm, _source, _target_name)],
                                "random": _sweep_random[(_arm, _source, _target_name)],
                                "unrelated": _sweep_unrelated[_arm],
                            }
                            for _condition, _condition_bases in _conditions.items():
                                _key = safe_key(
                                    "alpha_sweep", _source, _target_name,
                                    _group["group_id"], _modality, _kind,
                                    _arm, _alpha_key, _condition,
                                )
                                _stored = STORE.load("intervention", _key)
                                _specs.append((
                                    _arm, _alpha, _alpha_key, _condition,
                                    _condition_bases, _key, _stored,
                                ))
                    _missing = [spec for spec in _specs if spec[-1] is None]
                    _inputs = None
                    _clean_logits = None
                    if _missing:
                        _inputs = build_group_inputs(
                            _group, _modality,
                            _sweep_prompt(
                                _kind, _modality, _group["caption"]
                            ),
                        )
                        _clean_logits = BACKEND.forward_logits(_inputs.tensors)[
                            0, _inputs.final_prompt_position
                        ].float()
                    for (
                        _arm, _alpha, _alpha_key, _condition,
                        _condition_bases, _key, _stored,
                    ) in _specs:
                        if _stored is None:
                            _trial = unrestricted_swap_trial(
                                BACKEND, _inputs, bases=_condition_bases,
                                alpha=_alpha, target_token_id=int(_expected),
                                source_token_id=int(_source_token),
                                clean_logits=_clean_logits,
                                compact_positions=True,
                            )
                            _patched_surface = BACKEND.decode_token(
                                _trial["patched_top_token_id"]
                            ).strip()
                            _stored = {
                                **_trial,
                                "patched_surface": _patched_surface,
                                "success": open_answer_matches(
                                    _patched_surface, _expected_surface
                                ),
                            }
                            STORE.save("intervention", _key, _stored)
                            _computed_counter += 1
                        else:
                            _reused_counter += 1
                        _record["arms"][_arm][_alpha_key][_condition] = _stored
                        _trial_counter += 1
                        if _trial_counter == 1 or _trial_counter % 48 == 0:
                            print(
                                "alpha trials", _trial_counter,
                                "computed", _computed_counter,
                                "reused", _reused_counter,
                            )
                    _sweep_rows.append(_record)

    _alpha1_expected = {
        (
            row["source"], row["target"], row["group_id"], row["modality"],
            row["prompt_kind"], row["lens_arm"],
        ): int(row["patched_top_token_id"])
        for row in ALPHA1_SOURCE["alpha1_exact_outcomes"]
    }
    _alpha1_parity_failures = []
    for _row in _sweep_rows:
        for _arm in ("text", "pooled"):
            _key = (
                _row["source"], _row["target"], _row["group_id"],
                _row["modality"], _row["prompt_kind"], _arm,
            )
            _observed = int(
                _row["arms"][_arm]["a1"]["exact"]["patched_top_token_id"]
            )
            if _alpha1_expected.get(_key) != _observed:
                _alpha1_parity_failures.append({
                    "key": list(_key), "expected": _alpha1_expected.get(_key),
                    "observed": _observed,
                })
    if _alpha1_parity_failures:
        raise RuntimeError(
            "alpha=1 exact outcomes do not reproduce the completed source run; "
            f"first failures: {_alpha1_parity_failures[:3]}"
        )
    print("alpha=1 exact parity with completed run", True, "outcomes", len(_alpha1_expected))

    def _mean(values):
        values = [float(value) for value in values]
        return sum(values) / len(values) if values else None

    _sweep_cells = []
    for _arm in ("text", "pooled"):
        for _alpha in ALPHA_SWEEP:
            _alpha_key = f"a{_alpha:g}"
            for _source, _target_name in (("bird", "cat"), ("cat", "bird")):
                for _kind in ("identity", "property"):
                    for _modality in ("text", "image", "spoken_audio"):
                        _selected = [
                            row for row in _sweep_rows
                            if row["source"] == _source
                            and row["prompt_kind"] == _kind
                            and row["modality"] == _modality
                        ]
                        _by_condition = {
                            condition: [
                                row["arms"][_arm][_alpha_key][condition]
                                for row in _selected
                            ]
                            for condition in ("exact", "random", "unrelated")
                        }
                        _cell = {
                            "lens_arm": _arm,
                            "alpha": float(_alpha),
                            "alpha_role": _by_condition["exact"][0]["alpha_role"],
                            "direction": f"{_source}->{_target_name}",
                            "prompt_kind": _kind,
                            "modality": _modality,
                            "n": len(_selected),
                        }
                        for _condition, _trials in _by_condition.items():
                            _cell[_condition] = {
                                "top1_success_rate": _mean(
                                    trial["success"] for trial in _trials
                                ),
                                "prediction_change_rate": _mean(
                                    trial["prediction_changed"] for trial in _trials
                                ),
                                "mean_target_logit_delta": _mean(
                                    trial["target_logit_delta"] for trial in _trials
                                ),
                                "mean_target_rank_improvement": _mean(
                                    trial["target_rank_improvement"] for trial in _trials
                                ),
                                "mean_target_probability_delta": _mean(
                                    trial["target_probability_delta"] for trial in _trials
                                ),
                                "mean_source_logit_delta": _mean(
                                    trial["source_logit_delta"] for trial in _trials
                                ),
                                "mean_kl_clean_to_patched": _mean(
                                    trial["kl_clean_to_patched"] for trial in _trials
                                ),
                                "max_activation_norm_ratio": max(
                                    trial["max_activation_norm_ratio"]
                                    for trial in _trials
                                ),
                                "max_update_to_activation_norm_ratio": max(
                                    trial["max_update_to_activation_norm_ratio"]
                                    for trial in _trials
                                ),
                            }
                        _cell["specificity"] = {
                            "exact_minus_random_target_logit_delta": (
                                _cell["exact"]["mean_target_logit_delta"]
                                - _cell["random"]["mean_target_logit_delta"]
                            ),
                            "exact_minus_unrelated_target_logit_delta": (
                                _cell["exact"]["mean_target_logit_delta"]
                                - _cell["unrelated"]["mean_target_logit_delta"]
                            ),
                            "controls_are_alpha_matched": True,
                        }
                        _sweep_cells.append(_cell)

    # Select one common alpha for both lens arms using identity evidence only.
    # Property/leg-answer rows are deliberately excluded from selection so
    # they remain a downstream diagnostic rather than the tuning objective.
    _alpha_ranking = []
    for _alpha in ALPHA_SWEEP:
        _identity_cells = [
            cell for cell in _sweep_cells
            if cell["alpha"] == float(_alpha)
            and cell["prompt_kind"] == "identity"
        ]
        _property_cells = [
            cell for cell in _sweep_cells
            if cell["alpha"] == float(_alpha)
            and cell["prompt_kind"] == "property"
        ]
        _identity_vs_random = [
            cell["specificity"]["exact_minus_random_target_logit_delta"]
            for cell in _identity_cells
        ]
        _identity_vs_unrelated = [
            cell["specificity"]["exact_minus_unrelated_target_logit_delta"]
            for cell in _identity_cells
        ]
        _max_norm_ratio = max(
            cell["exact"]["max_activation_norm_ratio"]
            for cell in _identity_cells
        )
        _max_update_ratio = max(
            cell["exact"]["max_update_to_activation_norm_ratio"]
            for cell in _identity_cells
        )
        _all_specific = all(value > 0.0 for value in _identity_vs_random) and all(
            value > 0.0 for value in _identity_vs_unrelated
        )
        _safe = (
            _max_norm_ratio <= ALPHA_SELECTION_MAX_ACTIVATION_NORM_RATIO
            and _max_update_ratio <= ALPHA_SELECTION_MAX_UPDATE_RATIO
        )
        _alpha_ranking.append({
            "alpha": float(_alpha),
            "n_identity_cells": len(_identity_cells),
            "common_to_both_lens_arms": True,
            "mean_identity_target_logit_delta": _mean(
                cell["exact"]["mean_target_logit_delta"]
                for cell in _identity_cells
            ),
            "mean_identity_exact_minus_random": _mean(_identity_vs_random),
            "mean_identity_exact_minus_unrelated": _mean(
                _identity_vs_unrelated
            ),
            "robust_identity_specificity_score": min(
                _mean(_identity_vs_random), _mean(_identity_vs_unrelated)
            ),
            "identity_top1_success_rate": _mean(
                cell["exact"]["top1_success_rate"]
                for cell in _identity_cells
            ),
            "mean_property_target_logit_delta_not_used_for_selection": _mean(
                cell["exact"]["mean_target_logit_delta"]
                for cell in _property_cells
            ),
            "property_top1_success_rate_not_used_for_selection": _mean(
                cell["exact"]["top1_success_rate"]
                for cell in _property_cells
            ),
            "all_identity_cells_beat_both_controls": _all_specific,
            "max_activation_norm_ratio": _max_norm_ratio,
            "max_update_to_activation_norm_ratio": _max_update_ratio,
            "passes_predeclared_safety_guard": _safe,
            "eligible_for_exploratory_selection": bool(_all_specific and _safe),
        })
    _eligible_alphas = sorted(
        (
            row for row in _alpha_ranking
            if row["eligible_for_exploratory_selection"]
        ),
        key=lambda row: (
            -row["robust_identity_specificity_score"],
            -row["identity_top1_success_rate"],
            row["max_activation_norm_ratio"],
            row["alpha"],
        ),
    )
    _exploratory_best_alpha = (
        _eligible_alphas[0]["alpha"] if _eligible_alphas else None
    )

    ALPHA_SWEEP_REPORT = {
        "schema": "jlens.mmpilot.matched_multimodal_alpha_sweep.v1",
        "protocol": "matched_multimodal_jlens_unrestricted_alpha_refinement.v6",
        "verdict": "EXPLORATORY_ALPHA_DOSE_RESPONSE_MEASURED",
        "scientific_fingerprint": SCIENTIFIC_FINGERPRINT,
        "method": {
            "coordinate_swap_method_version": COORDINATE_SWAP_METHOD_VERSION,
            "equation": "c=pinv(V)h; h'=h+alpha*V*(swap(c)-c)",
            "vectors": "raw rows of W_U @ J_layer; no normalization",
            "layers": list(_layers),
            "positions": "every original prompt position",
            "coordinates_recomputed_at_every_layer": True,
            "orthogonal_component_preserved": True,
            "teacher_forcing_used": False,
            "candidate_list_supplied": False,
            "tested_alphas": list(ALPHA_SWEEP),
            "paper_exact_anchor_alpha": 1.0,
            "paper_reported_double_strength_alpha_measured_in_coarse_run": 2.0,
            "multimodal_task_is_extension_not_exact_replication": True,
        },
        "alpha_roles": SCIENTIFIC_CONFIG["alpha_roles"],
        "primary_alpha_remains": PRIMARY_ALPHA,
        "alpha_selected_after_outcomes": ALPHA_SWEEP_OUTCOME_INFORMED,
        "alpha_grid_provenance": (
            "The 0.125-to-1 grid was selected after inspecting the coarse "
            "0.5/1/2/4 sweep, where alpha=0.5 had the strongest stable identity "
            "signal and alpha>=2 caused large norm inflation. It is exploratory "
            "model-characterization only; any selected alpha requires a fresh, "
            "independently frozen population."
        ),
        "exploratory_alpha_selection": {
            "selection_rule_version": (
                "mmpilot.multimodal_alpha_identity_specificity.v1"
            ),
            "rule_frozen_before_refinement_outcomes": True,
            "grid_selected_after_coarse_outcomes": True,
            "one_common_alpha_across_lens_arms": True,
            "selection_endpoint": "identity_target_logit_specificity",
            "property_endpoint_used_for_selection": False,
            "score": (
                "min(mean(exact-random target-logit delta), "
                "mean(exact-unrelated target-logit delta)) over all 12 identity "
                "cells from both lens arms"
            ),
            "eligibility": (
                "every identity cell beats both controls and the predeclared "
                "activation/update norm guards pass"
            ),
            "max_activation_norm_ratio": (
                ALPHA_SELECTION_MAX_ACTIVATION_NORM_RATIO
            ),
            "max_update_to_activation_norm_ratio": (
                ALPHA_SELECTION_MAX_UPDATE_RATIO
            ),
            "ranking": _alpha_ranking,
            "exploratory_best_alpha": _exploratory_best_alpha,
            "confirmatory_status": "REQUIRES_FRESH_POPULATION",
        },
        "population_reused_without_reselection": True,
        "alpha1_exact_outcome_parity": {
            "passed": True,
            "n_outcomes": len(_alpha1_expected),
            "failures": [],
        },
        "population_source": ALPHA1_SOURCE,
        "lens_checksums": LENS_CHECKSUMS,
        "controls": ["random", "unrelated"],
        "controls_are_intensity_matched": True,
        "graded_endpoints": [
            "target_logit_delta", "target_rank_improvement",
            "target_probability_delta", "source_logit_delta",
            "kl_clean_to_patched", "activation_norm_ratio",
            "unrestricted_top1_success",
        ],
        "cells": _sweep_cells,
        "rows": _sweep_rows,
    }
    ALPHA_SWEEP_REPORT["report_checksum"] = payload_checksum(ALPHA_SWEEP_REPORT)
    STORE.save("metric", "alpha_sweep_report", ALPHA_SWEEP_REPORT)
    _alpha_path = RUN_DIR / "multimodal_lens_alpha_sweep_report.json"
    _alpha_path.write_text(json.dumps(ALPHA_SWEEP_REPORT, indent=2, default=str))
elif REAL_MODE:
    ALPHA_SWEEP_REPORT = (
        STORE.load("metric", "alpha_sweep_report")
        if "STORE" in globals() else None
    )

if ALPHA_SWEEP_REPORT:
    print("=" * 112)
    print("PAIRED ALPHA DOSE-RESPONSE — unrestricted full-vocabulary output")
    print("=" * 112)
    print(f"{'arm':7s} {'a':>4s} {'direction':10s} {'endpoint':8s} {'modality':13s} "
          f"{'top1':>6s} {'dlogit':>9s} {'drank':>9s} {'vs-rand':>9s} {'vs-unrel':>9s}")
    for _cell in ALPHA_SWEEP_REPORT["cells"]:
        print(
            f"{_cell['lens_arm']:7s} {_cell['alpha']:4.1f} "
            f"{_cell['direction']:10s} {_cell['prompt_kind']:8s} "
            f"{_cell['modality']:13s} "
            f"{_cell['exact']['top1_success_rate']:6.3f} "
            f"{_cell['exact']['mean_target_logit_delta']:+9.3f} "
            f"{_cell['exact']['mean_target_rank_improvement']:+9.1f} "
            f"{_cell['specificity']['exact_minus_random_target_logit_delta']:+9.3f} "
            f"{_cell['specificity']['exact_minus_unrelated_target_logit_delta']:+9.3f}"
        )
    print("report", RUN_DIR / "multimodal_lens_alpha_sweep_report.json")
    print("checksum", ALPHA_SWEEP_REPORT["report_checksum"])
    print(
        "exploratory best alpha",
        ALPHA_SWEEP_REPORT["exploratory_alpha_selection"][
            "exploratory_best_alpha"
        ],
        "(fresh-population confirmation required)",
    )
    print("Alpha=1 remains primary; every other alpha is sensitivity evidence.")
'''
)

markdown("## 10C. Stage 3C — broad paper-depth pooled multimodal J-lens")
code(
    r'''
BROAD_POOLED_REPORT = None
if REAL_MODE and BROAD_POOLED_ENABLED:
    from jlens.lens import JacobianLens
    from jlens.metadata import file_sha256
    from jlens.mmpilot.coordinate_swap import (
        random_two_direction_basis, resolve_concept_token,
    )
    from jlens.mmpilot.digit_reasoning_confirmation import resolve_digit_endpoints
    from jlens.mmpilot.multimodal_lens import (
        build_swap_bases_for_lens, combine_layer_shards, fit_arm,
        load_completed_causal_source, open_answer_matches, plan_units,
        select_causal_groups, unrestricted_swap_trial,
    )
    from jlens.mmpilot.store import RunFingerprint, UnitStore, safe_key

    # The late shard is not rediscovered. Its completed report and tensor are
    # checksum-pinned before the first new backward pass.
    _broad_source = load_completed_causal_source(
        CAUSAL_LENS_SOURCE_RUN_DIR,
        expected_final_report_checksum=EXPECTED_SOURCE_FINAL_REPORT_CHECKSUM,
        expected_cross_report_checksum=EXPECTED_SOURCE_CROSS_REPORT_CHECKSUM,
        expected_causal_report_checksum=EXPECTED_SOURCE_CAUSAL_REPORT_CHECKSUM,
        expected_lens_checksums=EXPECTED_SOURCE_LENS_CHECKSUMS,
    )
    _late_path = Path(_broad_source["lens_paths"]["pooled"])
    _late_lens = JacobianLens.load(str(_late_path))
    if _broad_source.get("fit_plan_digest") != PLAN["plan_digest"]:
        raise RuntimeError(
            "the checksum-pinned late shard was not fitted on this exact "
            "99-example multimodal plan"
        )
    if _late_lens.source_layers != list(BROAD_POOLED_LATE_LAYERS):
        raise RuntimeError(
            f"pinned pooled late shard covers {_late_lens.source_layers}, not "
            f"{list(BROAD_POOLED_LATE_LAYERS)}"
        )
    if _late_lens.n_prompts != N_FIT_GROUPS or _late_lens.d_model != EXPECT_D_MODEL:
        raise RuntimeError(
            "the checksum-pinned late shard has the wrong fit count or width"
        )

    _broad_config = {
        "study": "paper_depth_pooled_multimodal_jlens_development.v1",
        "model_repo_id": MODEL_REPO_ID,
        "model_revision": MODEL_REVISION,
        "audio_protocol_fingerprint": AUDIO_PROTOCOL_FINGERPRINT,
        "manifest_checksum": MANIFEST_CHECKSUM,
        "fit_plan_digest": PLAN["plan_digest"],
        "fit_arm": "pooled",
        "fit_examples": N_FIT_GROUPS,
        "early_layers_fitted_now": list(BROAD_POOLED_EARLY_LAYERS),
        "late_layers_reused": list(BROAD_POOLED_LATE_LAYERS),
        "full_band": list(BROAD_POOLED_BAND),
        "target_layer": TARGET_LAYER,
        "late_lens_checksum": EXPECTED_SOURCE_LENS_CHECKSUMS["pooled"],
        "pairs": [list(pair) for pair in BROAD_POOLED_PAIRS],
        "concepts": list(BROAD_POOLED_CONCEPTS),
        "control_concepts": list(BROAD_POOLED_CONTROLS),
        "alphas": list(BROAD_POOLED_ALPHAS),
        "primary_alpha": 1.0,
        "position_rule": "all_original_prompt_positions",
        "teacher_forcing": False,
        "candidate_list": False,
        "population_seed": BROAD_POOLED_SEED,
        "commit": COMMIT,
    }
    _broad_digest = payload_checksum(_broad_config)
    BROAD_RUN_DIR = (
        RUNS_ROOT / "mmbroadpooledj" /
        f"mmbroadpooledj_real_{_broad_digest.split(':')[1][:12]}"
    )
    BROAD_RUN_DIR.mkdir(parents=True, exist_ok=True)
    (BROAD_RUN_DIR / "scientific_config.json").write_text(
        json.dumps(_broad_config, indent=2)
    )

    _early_path = BROAD_RUN_DIR / "lenses" / "lens.pooled.l16_l32.pt"
    _early_checkpoint = (
        BROAD_RUN_DIR / "lenses" / "checkpoints" /
        "pooled.l16_l32.jacobian_sum.pt"
    )
    if _early_path.is_file():
        _early_lens = JacobianLens.load(str(_early_path))
        print("broad pooled early shard reused", _early_path)
    else:
        _early_lens = fit_arm(
            BACKEND, plan_units(PLAN, "pooled"), build_inputs=build_fit_inputs,
            source_layers=BROAD_POOLED_EARLY_LAYERS, target_layer=TARGET_LAYER,
            checkpoint_path=_early_checkpoint, arm="pooled",
            scientific_fingerprint=_broad_digest, dim_batch=DIM_BATCH,
            skip_first=SKIP_FIRST, checkpoint_every=CHECKPOINT_EVERY,
            progress=progress,
        )
        _early_path.parent.mkdir(parents=True, exist_ok=True)
        _temporary = _early_path.with_suffix(".tmp.pt")
        _early_lens.save(str(_temporary))
        os.replace(_temporary, _early_path)
        print("broad pooled early shard completed", _early_path)

    _broad_lens = combine_layer_shards(
        [_early_lens, _late_lens], expected_layers=BROAD_POOLED_BAND
    )
    _combined_path = BROAD_RUN_DIR / "lenses" / "lens.pooled.l16_l40.pt"
    if not _combined_path.is_file():
        _temporary = _combined_path.with_suffix(".tmp.pt")
        _broad_lens.save(str(_temporary))
        os.replace(_temporary, _combined_path)
    _early_checksum = file_sha256(str(_early_path))
    _combined_checksum = file_sha256(str(_combined_path))
    print("broad pooled lens", _combined_path)
    print("  early checksum", _early_checksum)
    print("  late checksum ", EXPECTED_SOURCE_LENS_CHECKSUMS["pooled"])
    print("  full checksum ", _combined_checksum)

    _forbidden = {
        concept: tuple(other for other in BROAD_POOLED_CONCEPTS if other != concept)
        for concept in BROAD_POOLED_CONCEPTS
    }
    _broad_population = select_causal_groups(
        GROUPS, concepts=BROAD_POOLED_CONCEPTS,
        n_per_concept=BROAD_POOLED_CANDIDATES_PER_CONCEPT,
        excluded_image_ids=(
            *PLAN["fit_image_ids"], *PLAN["eval_image_ids"],
            *_broad_source["excluded_image_ids"],
        ),
        seed=BROAD_POOLED_SEED, forbidden_concepts=_forbidden,
    )
    _population_record = {
        concept: [
            {"group_id": row["group_id"], "image_id": row["image_id"]}
            for row in rows
        ]
        for concept, rows in _broad_population.items()
    }
    _population_digest = payload_checksum(_population_record)
    (BROAD_RUN_DIR / "development_population.json").write_text(
        json.dumps({
            "population": _population_record,
            "population_digest": _population_digest,
            "excluded_fit_images": len(PLAN["fit_image_ids"]),
            "excluded_eval_images": len(PLAN["eval_image_ids"]),
            "excluded_prior_screen_images": len(_broad_source["excluded_image_ids"]),
        }, indent=2)
    )

    _broad_fingerprint = RunFingerprint(
        mode="real", model_repo_id=MODEL_REPO_ID,
        model_revision=MODEL_REVISION, processor_revision=MODEL_REVISION,
        layers=tuple(BROAD_POOLED_BAND), lens_checksum=_combined_checksum,
        manifest_checksum=MANIFEST_CHECKSUM, split_id=_population_digest,
        intervention_config={
            "alphas": list(BROAD_POOLED_ALPHAS),
            "pairs": [list(pair) for pair in BROAD_POOLED_PAIRS],
            "positions": "all_original_prompt_positions",
            "controls": ["zero", "random_norm_matched", "unrelated_alpha_matched"],
        },
        extra={"study_digest": _broad_digest, "fit_plan_digest": PLAN["plan_digest"]},
    )
    _broad_store = UnitStore(BROAD_RUN_DIR, _broad_fingerprint)
    print("broad run state", _broad_store.open())

    _encode = BACKEND.encode_candidate
    _all_tokens = {
        name: resolve_concept_token(_encode, name)
        for name in (*BROAD_POOLED_CONCEPTS, *BROAD_POOLED_CONTROLS)
    }
    _digits = resolve_digit_endpoints(BACKEND)
    _legs = {"bird": "2", "cat": "4", "zebra": "4", "giraffe": "4"}
    _bases = {
        (source, target): build_swap_bases_for_lens(
            _broad_lens, BACKEND.unembedding_weight(),
            layers=BROAD_POOLED_BAND, source=_all_tokens[source],
            target=_all_tokens[target],
        )
        for source, target in BROAD_POOLED_PAIRS
    }
    _unrelated = build_swap_bases_for_lens(
        _broad_lens, BACKEND.unembedding_weight(), layers=BROAD_POOLED_BAND,
        source=_all_tokens[BROAD_POOLED_CONTROLS[0]],
        target=_all_tokens[BROAD_POOLED_CONTROLS[1]],
    )

    def _broad_prompt(modality, caption):
        question = (
            "How many legs does the animal in the evidence typically have? "
            "Answer with one digit.\nAnswer:"
        )
        return f"Caption: {caption}\n{question}" if modality == "text" else question

    # Recruitment uses only clean capability and is persisted before any swap.
    _capability_rows = []
    for _source in BROAD_POOLED_CONCEPTS:
        for _group in _broad_population[_source]:
            for _modality in ("text", "image", "spoken_audio"):
                _key = safe_key("broad_capability", _source, _group["group_id"], _modality)
                _row = _broad_store.load("capability", _key)
                if _row is None:
                    _inputs = build_group_inputs(
                        _group, _modality,
                        _broad_prompt(_modality, _group["caption"]),
                    )
                    _logits = BACKEND.forward_logits(_inputs.tensors)[
                        0, _inputs.final_prompt_position
                    ].float()
                    _surface = BACKEND.decode_token(int(_logits.argmax())).strip()
                    _row = {
                        "source": _source, "group_id": _group["group_id"],
                        "image_id": _group["image_id"], "modality": _modality,
                        "expected": _legs[_source], "generated": _surface,
                        "pass": open_answer_matches(_surface, _legs[_source]),
                    }
                    _broad_store.save("capability", _key, _row)
                    _work = "computed"
                else:
                    _work = "reused"
                _capability_rows.append(_row)
                if len(_capability_rows) == 1 or len(_capability_rows) % 48 == 0:
                    print("broad capability", len(_capability_rows), _work)

    _recruited = {}
    for _source in BROAD_POOLED_CONCEPTS:
        _eligible = []
        for _group in _broad_population[_source]:
            _rows = [
                row for row in _capability_rows
                if row["source"] == _source and row["group_id"] == _group["group_id"]
            ]
            if len(_rows) == 3 and all(row["pass"] for row in _rows):
                _eligible.append(_group)
        _recruited[_source] = _eligible[:BROAD_POOLED_IMAGES_PER_DIRECTION]
    _capability_go = all(
        len(rows) == BROAD_POOLED_IMAGES_PER_DIRECTION
        for rows in _recruited.values()
    )
    print("broad recruited", {key: len(value) for key, value in _recruited.items()})

    _trial_rows = []
    if _capability_go:
        for _source, _target_name in BROAD_POOLED_PAIRS:
            for _group in _recruited[_source]:
                for _modality in ("text", "image", "spoken_audio"):
                    _inputs = None
                    _clean_logits = None
                    _source_answer_id = int(_digits["token_ids"][_legs[_source]])
                    _target_answer_id = int(_digits["token_ids"][_legs[_target_name]])
                    _conditions = [("zero", 0.0, _bases[(_source, _target_name)])]
                    for _alpha in BROAD_POOLED_ALPHAS:
                        _exact = _bases[(_source, _target_name)]
                        _conditions.extend((
                            (f"exact_alpha{_alpha:g}", _alpha, _exact),
                            (f"random_alpha{_alpha:g}", _alpha, {
                                layer: random_two_direction_basis(
                                    basis, seed=20260822 + layer
                                ) for layer, basis in _exact.items()
                            }),
                            (f"unrelated_alpha{_alpha:g}", _alpha, _unrelated),
                        ))
                    for _condition, _alpha, _condition_bases in _conditions:
                        _key = safe_key(
                            "broad_trial", _source, _target_name,
                            _group["group_id"], _modality, _condition,
                        )
                        _stored = _broad_store.load("intervention", _key)
                        if _stored is None:
                            if _inputs is None:
                                _inputs = build_group_inputs(
                                    _group, _modality,
                                    _broad_prompt(_modality, _group["caption"]),
                                )
                                _clean_logits = BACKEND.forward_logits(_inputs.tensors)[
                                    0, _inputs.final_prompt_position
                                ].float()
                            _trial = unrestricted_swap_trial(
                                BACKEND, _inputs, bases=_condition_bases,
                                alpha=_alpha, target_token_id=_target_answer_id,
                                source_token_id=_source_answer_id,
                                clean_logits=_clean_logits, compact_positions=True,
                            )
                            _surface = BACKEND.decode_token(
                                _trial["patched_top_token_id"]
                            ).strip()
                            _stored = {
                                **_trial, "source": _source,
                                "target": _target_name,
                                "direction": f"{_source}->{_target_name}",
                                "group_id": _group["group_id"],
                                "image_id": _group["image_id"],
                                "modality": _modality, "condition": _condition,
                                "expected": _legs[_target_name],
                                "patched_surface": _surface,
                                "success": open_answer_matches(
                                    _surface, _legs[_target_name]
                                ),
                            }
                            _broad_store.save("intervention", _key, _stored)
                            _work = "computed"
                        else:
                            _work = "reused"
                        _trial_rows.append(_stored)
                        if len(_trial_rows) == 1 or len(_trial_rows) % 96 == 0:
                            print("broad trials", len(_trial_rows), _work)

    def _rate(rows):
        return sum(bool(row["success"]) for row in rows) / len(rows) if rows else 0.0

    _cells = []
    for _source, _target_name in BROAD_POOLED_PAIRS:
        for _modality in ("text", "image", "spoken_audio"):
            _selected = [
                row for row in _trial_rows
                if row["source"] == _source and row["target"] == _target_name
                and row["modality"] == _modality
            ]
            _cell = {
                "direction": f"{_source}->{_target_name}",
                "modality": _modality,
                "n": BROAD_POOLED_IMAGES_PER_DIRECTION,
            }
            for _condition in (
                "zero", "exact_alpha1", "random_alpha1", "unrelated_alpha1",
                "exact_alpha2", "random_alpha2", "unrelated_alpha2",
            ):
                _rows = [row for row in _selected if row["condition"] == _condition]
                _cell[_condition] = {
                    "success_rate": _rate(_rows),
                    "successes": sum(bool(row["success"]) for row in _rows),
                    "mean_target_logit_delta": (
                        sum(float(row["target_logit_delta"]) for row in _rows) / len(_rows)
                        if _rows else None
                    ),
                    "max_activation_norm_ratio": (
                        max(float(row["max_activation_norm_ratio"]) for row in _rows)
                        if _rows else None
                    ),
                    "max_update_to_activation_norm_ratio": (
                        max(
                            float(row["max_update_to_activation_norm_ratio"])
                            for row in _rows
                        ) if _rows else None
                    ),
                    "integrity_pass": bool(_rows) and all(
                        row["all_prompt_positions_patched"]
                        and row["layers_patched"] == list(BROAD_POOLED_BAND)
                        and float(row["max_orthogonal_residual_drift"]) <= 1e-5
                        and float(row["max_coordinate_update_error"]) <= 1e-5
                        for row in _rows
                    ),
                }
            _cells.append(_cell)

    def _direction_pass(direction, alpha):
        cells = [cell for cell in _cells if cell["direction"] == direction]
        exact = f"exact_alpha{alpha:g}"
        random = f"random_alpha{alpha:g}"
        unrelated = f"unrelated_alpha{alpha:g}"
        return len(cells) == 3 and all(
            cell[exact]["success_rate"] >= 0.50
            and cell[exact]["success_rate"] > cell["zero"]["success_rate"]
            and cell[exact]["success_rate"] > cell[random]["success_rate"]
            and cell[exact]["success_rate"] > cell[unrelated]["success_rate"]
            and cell[exact]["integrity_pass"]
            and cell[random]["integrity_pass"]
            and cell[unrelated]["integrity_pass"]
            and cell[exact]["max_activation_norm_ratio"] <= (
                1.25 if alpha == 1.0 else 1.50
            )
            and cell[exact]["max_update_to_activation_norm_ratio"] <= (
                0.50 if alpha == 1.0 else 1.00
            )
            for cell in cells
        )

    _alpha1_directions = [
        f"{source}->{target}" for source, target in BROAD_POOLED_PAIRS
        if _direction_pass(f"{source}->{target}", 1.0)
    ]
    _alpha2_directions = [
        f"{source}->{target}" for source, target in BROAD_POOLED_PAIRS
        if _direction_pass(f"{source}->{target}", 2.0)
    ]
    _verdict = (
        "BROAD_POOLED_J_DEVELOPMENT_CAPABILITY_NO_GO" if not _capability_go
        else "BROAD_POOLED_J_DEVELOPMENT_ALPHA1_GO" if _alpha1_directions
        else "BROAD_POOLED_J_DEVELOPMENT_ALPHA2_SENSITIVITY_ONLY"
        if _alpha2_directions else "BROAD_POOLED_J_DEVELOPMENT_NO_GO"
    )
    BROAD_POOLED_REPORT = {
        "schema": "jlens.mmpilot.broad_pooled_multimodal_j_workspace.v1",
        "verdict": _verdict,
        "scientific_config": _broad_config,
        "study_digest": _broad_digest,
        "population_digest": _population_digest,
        "lens_provenance": {
            "fit_distribution": "33 text + 33 image + 33 spoken_audio",
            "fit_plan_digest": PLAN["plan_digest"],
            "early_shard_checksum": _early_checksum,
            "late_shard_checksum": EXPECTED_SOURCE_LENS_CHECKSUMS["pooled"],
            "combined_checksum": _combined_checksum,
            "same_99_examples_in_both_shards": True,
        },
        "method": {
            "lens": "ordinary sample-mean J-lens",
            "r_lens_used": False,
            "equation": "c=pinv(V)h; h'=h+alpha*V*(swap(c)-c)",
            "layers": list(BROAD_POOLED_BAND),
            "positions": "every original prompt position",
            "coordinates_recomputed_at_every_layer": True,
            "teacher_forcing_used": False,
            "candidate_list_supplied": False,
            "endpoint": "unrestricted full-vocabulary next-token top1",
        },
        "capability_go": _capability_go,
        "recruited_counts": {key: len(value) for key, value in _recruited.items()},
        "alpha1_primary_passing_directions": _alpha1_directions,
        "alpha2_sensitivity_passing_directions": _alpha2_directions,
        "cells": _cells,
        "rows": _trial_rows,
        "claim_boundary": (
            "This is prospective development on a predeclared paper-depth band. "
            "Any passing direction must be repeated on a fresh frozen population "
            "before it is described as independently confirmed."
        ),
    }
    BROAD_POOLED_REPORT["report_checksum"] = payload_checksum(BROAD_POOLED_REPORT)
    _broad_store.save("metric", "broad_pooled_j_report", BROAD_POOLED_REPORT)
    _broad_report_path = BROAD_RUN_DIR / "broad_pooled_multimodal_j_workspace_report.json"
    _broad_report_path.write_text(json.dumps(BROAD_POOLED_REPORT, indent=2, default=str))
    print("=" * 96)
    print("BROAD POOLED MULTIMODAL J-LENS —", _verdict)
    print("=" * 96)
    print("alpha=1 passing directions", _alpha1_directions)
    print("alpha=2 sensitivity       ", _alpha2_directions)
    print("report", _broad_report_path)
    print("checksum", BROAD_POOLED_REPORT["report_checksum"])
elif RUN_STAGE3C_BROAD_POOLED_WORKSPACE:
    print("Stage 3C requested but blocked by model/fit/causal budget confirmations.")
'''
)

markdown("## 10D. Stage 3D — independently frozen multimodal confirmation")
code(
    r'''
FRESH_CONFIRMATION_REPORT = None
if REAL_MODE and FRESH_CONFIRMATION_ENABLED:
    from jlens.lens import JacobianLens
    from jlens.mmpilot.coordinate_swap import (
        random_two_direction_basis, resolve_concept_token,
    )
    from jlens.mmpilot.digit_reasoning_confirmation import resolve_digit_endpoints
    from jlens.mmpilot.multimodal_lens import (
        build_swap_bases_for_lens, holm_adjust,
        load_broad_pooled_development_source, load_completed_causal_source,
        open_answer_matches, paired_binary_one_sided_p,
        select_causal_groups, unrestricted_swap_trial,
    )
    from jlens.mmpilot.store import RunFingerprint, UnitStore, safe_key

    CONFIRMATION_SOURCE = load_broad_pooled_development_source(
        BROAD_DEVELOPMENT_RUN_DIR,
        expected_report_checksum=EXPECTED_BROAD_DEVELOPMENT_REPORT_CHECKSUM,
        expected_population_digest=EXPECTED_BROAD_DEVELOPMENT_POPULATION_DIGEST,
        expected_lens_checksum=EXPECTED_BROAD_POOLED_LENS_CHECKSUM,
        expected_direction=CONFIRMATION_DIRECTION,
    )
    _prior_source = load_completed_causal_source(
        CAUSAL_LENS_SOURCE_RUN_DIR,
        expected_final_report_checksum=EXPECTED_SOURCE_FINAL_REPORT_CHECKSUM,
        expected_cross_report_checksum=EXPECTED_SOURCE_CROSS_REPORT_CHECKSUM,
        expected_causal_report_checksum=EXPECTED_SOURCE_CAUSAL_REPORT_CHECKSUM,
        expected_lens_checksums=EXPECTED_SOURCE_LENS_CHECKSUMS,
    )
    print("CONFIRMATION DESIGN FROZEN BEFORE FRESH POPULATION")
    print("  source report", EXPECTED_BROAD_DEVELOPMENT_REPORT_CHECKSUM)
    print("  source lens  ", EXPECTED_BROAD_POOLED_LENS_CHECKSUM)
    print("  direction    ", "->".join(CONFIRMATION_DIRECTION))
    print("  alpha / band ", CONFIRMATION_ALPHA, list(BROAD_POOLED_BAND))
    print("  source digest", CONFIRMATION_SOURCE["source_digest"])

    _confirmation_lens = JacobianLens.load(CONFIRMATION_SOURCE["lens_path"])
    if _confirmation_lens.source_layers != list(BROAD_POOLED_BAND):
        raise RuntimeError("confirmation lens no longer covers exactly L16-L40")
    if _confirmation_lens.n_prompts != N_FIT_GROUPS:
        raise RuntimeError("confirmation lens no longer records 99 fit examples")

    _confirmation_config = {
        "study": "broad_pooled_multimodal_jlens_fresh_confirmation.v1",
        "development_source_digest": CONFIRMATION_SOURCE["source_digest"],
        "development_report_checksum": EXPECTED_BROAD_DEVELOPMENT_REPORT_CHECKSUM,
        "development_population_digest": EXPECTED_BROAD_DEVELOPMENT_POPULATION_DIGEST,
        "lens_checksum": EXPECTED_BROAD_POOLED_LENS_CHECKSUM,
        "model_repo_id": MODEL_REPO_ID,
        "model_revision": MODEL_REVISION,
        "audio_protocol_fingerprint": AUDIO_PROTOCOL_FINGERPRINT,
        "manifest_checksum": MANIFEST_CHECKSUM,
        "direction": list(CONFIRMATION_DIRECTION),
        "alpha": CONFIRMATION_ALPHA,
        "layers": list(BROAD_POOLED_BAND),
        "positions": "every original prompt position",
        "teacher_forcing": False,
        "candidate_list": False,
        "candidate_count": CONFIRMATION_CANDIDATES,
        "confirmation_images": CONFIRMATION_IMAGES,
        "min_success_rate": CONFIRMATION_MIN_SUCCESS_RATE,
        "min_control_margin": CONFIRMATION_MIN_CONTROL_MARGIN,
        "familywise_alpha": CONFIRMATION_FAMILYWISE_ALPHA,
        "seed": CONFIRMATION_SEED,
        "commit": COMMIT,
    }
    _confirmation_config_digest = payload_checksum(_confirmation_config)
    CONFIRMATION_RUN_DIR = (
        RUNS_ROOT / "mmbroadconfirm" /
        f"mmbroadconfirm_real_{_confirmation_config_digest.split(':')[1][:12]}"
    )
    CONFIRMATION_RUN_DIR.mkdir(parents=True, exist_ok=True)
    (CONFIRMATION_RUN_DIR / "scientific_config.json").write_text(
        json.dumps(_confirmation_config, indent=2)
    )

    # Exclude all opened development candidates—not only the eight winners—
    # plus every fit/evaluation and earlier causal-screen photograph.
    _excluded_confirmation_images = sorted(set(
        [*CONFIRMATION_SOURCE["excluded_image_ids"]]
        + [str(value) for value in PLAN["fit_image_ids"]]
        + [str(value) for value in PLAN["eval_image_ids"]]
        + [str(value) for value in _prior_source["excluded_image_ids"]]
    ))
    _confirmation_population = select_causal_groups(
        GROUPS, concepts=(CONFIRMATION_DIRECTION[0],),
        n_per_concept=CONFIRMATION_CANDIDATES,
        excluded_image_ids=_excluded_confirmation_images,
        seed=CONFIRMATION_SEED,
        forbidden_concepts={CONFIRMATION_DIRECTION[0]: (CONFIRMATION_DIRECTION[1],)},
    )[CONFIRMATION_DIRECTION[0]]
    _confirmation_population_record = [
        {"group_id": row["group_id"], "image_id": row["image_id"]}
        for row in _confirmation_population
    ]
    _confirmation_population_digest = payload_checksum(
        _confirmation_population_record
    )
    if set(row["image_id"] for row in _confirmation_population_record) & set(
        _excluded_confirmation_images
    ):
        raise RuntimeError("fresh confirmation overlaps an opened photograph")
    (CONFIRMATION_RUN_DIR / "fresh_population.json").write_text(json.dumps({
        "population": _confirmation_population_record,
        "population_digest": _confirmation_population_digest,
        "excluded_image_ids_digest": payload_checksum(_excluded_confirmation_images),
        "n_excluded_images": len(_excluded_confirmation_images),
        "selected_before_capability": True,
    }, indent=2))

    _confirmation_fingerprint = RunFingerprint(
        mode="real", model_repo_id=MODEL_REPO_ID,
        model_revision=MODEL_REVISION, processor_revision=MODEL_REVISION,
        layers=tuple(BROAD_POOLED_BAND),
        lens_checksum=EXPECTED_BROAD_POOLED_LENS_CHECKSUM,
        manifest_checksum=MANIFEST_CHECKSUM,
        split_id=_confirmation_population_digest,
        intervention_config={
            "direction": list(CONFIRMATION_DIRECTION),
            "alpha": CONFIRMATION_ALPHA,
            "conditions": ["exact", "zero", "random", "unrelated"],
            "positions": "all_original_prompt_positions",
        },
        extra={
            "confirmation_config_digest": _confirmation_config_digest,
            "development_source_digest": CONFIRMATION_SOURCE["source_digest"],
        },
    )
    _confirmation_store = UnitStore(
        CONFIRMATION_RUN_DIR, _confirmation_fingerprint
    )
    print("confirmation run state", _confirmation_store.open())

    _source, _target_name = CONFIRMATION_DIRECTION
    _tokens = {
        name: resolve_concept_token(BACKEND.encode_candidate, name)
        for name in (_source, _target_name, *BROAD_POOLED_CONTROLS)
    }
    _digits = resolve_digit_endpoints(BACKEND)
    _source_answer, _target_answer = "2", "4"
    _source_answer_id = int(_digits["token_ids"][_source_answer])
    _target_answer_id = int(_digits["token_ids"][_target_answer])
    _exact_bases = build_swap_bases_for_lens(
        _confirmation_lens, BACKEND.unembedding_weight(),
        layers=BROAD_POOLED_BAND, source=_tokens[_source],
        target=_tokens[_target_name],
    )
    _random_bases = {
        layer: random_two_direction_basis(basis, seed=20260822 + layer)
        for layer, basis in _exact_bases.items()
    }
    _unrelated_bases = build_swap_bases_for_lens(
        _confirmation_lens, BACKEND.unembedding_weight(),
        layers=BROAD_POOLED_BAND,
        source=_tokens[BROAD_POOLED_CONTROLS[0]],
        target=_tokens[BROAD_POOLED_CONTROLS[1]],
    )

    def _confirmation_prompt(modality, caption):
        question = (
            "How many legs does the animal in the evidence typically have? "
            "Answer with one digit.\nAnswer:"
        )
        return f"Caption: {caption}\n{question}" if modality == "text" else question

    _confirmation_capability = []
    for _group in _confirmation_population:
        for _modality in ("text", "image", "spoken_audio"):
            _key = safe_key(
                "fresh_capability", _group["group_id"], _modality
            )
            _row = _confirmation_store.load("capability", _key)
            if _row is None:
                _inputs = build_group_inputs(
                    _group, _modality,
                    _confirmation_prompt(_modality, _group["caption"]),
                )
                _logits = BACKEND.forward_logits(_inputs.tensors)[
                    0, _inputs.final_prompt_position
                ].float()
                _surface = BACKEND.decode_token(int(_logits.argmax())).strip()
                _row = {
                    "group_id": _group["group_id"],
                    "image_id": _group["image_id"],
                    "modality": _modality,
                    "expected": _source_answer,
                    "generated": _surface,
                    "pass": open_answer_matches(_surface, _source_answer),
                }
                _confirmation_store.save("capability", _key, _row)
                _work = "computed"
            else:
                _work = "reused"
            _confirmation_capability.append(_row)
            if len(_confirmation_capability) == 1 or len(_confirmation_capability) % 24 == 0:
                print("confirmation capability", len(_confirmation_capability), _work)

    _confirmation_recruited = []
    for _group in _confirmation_population:
        _rows = [
            row for row in _confirmation_capability
            if row["group_id"] == _group["group_id"]
        ]
        if len(_rows) == 3 and all(row["pass"] for row in _rows):
            _confirmation_recruited.append(_group)
        if len(_confirmation_recruited) == CONFIRMATION_IMAGES:
            break
    _confirmation_capability_go = (
        len(_confirmation_recruited) == CONFIRMATION_IMAGES
    )
    print("confirmation recruited", len(_confirmation_recruited), "/", CONFIRMATION_IMAGES)

    _confirmation_rows = []
    if _confirmation_capability_go:
        _condition_specs = (
            ("exact", 1.0, _exact_bases),
            ("zero", 0.0, _exact_bases),
            ("random", 1.0, _random_bases),
            ("unrelated", 1.0, _unrelated_bases),
        )
        for _group in _confirmation_recruited:
            for _modality in ("text", "image", "spoken_audio"):
                _inputs = None
                _clean_logits = None
                for _condition, _alpha, _bases in _condition_specs:
                    _key = safe_key(
                        "fresh_trial", _group["group_id"], _modality, _condition
                    )
                    _stored = _confirmation_store.load("intervention", _key)
                    if _stored is None:
                        if _inputs is None:
                            _inputs = build_group_inputs(
                                _group, _modality,
                                _confirmation_prompt(_modality, _group["caption"]),
                            )
                            _clean_logits = BACKEND.forward_logits(_inputs.tensors)[
                                0, _inputs.final_prompt_position
                            ].float()
                        _trial = unrestricted_swap_trial(
                            BACKEND, _inputs, bases=_bases, alpha=_alpha,
                            target_token_id=_target_answer_id,
                            source_token_id=_source_answer_id,
                            clean_logits=_clean_logits, compact_positions=True,
                        )
                        _surface = BACKEND.decode_token(
                            _trial["patched_top_token_id"]
                        ).strip()
                        _stored = {
                            **_trial, "group_id": _group["group_id"],
                            "image_id": _group["image_id"],
                            "modality": _modality, "condition": _condition,
                            "expected": _target_answer,
                            "patched_surface": _surface,
                            "success": open_answer_matches(
                                _surface, _target_answer
                            ),
                        }
                        _confirmation_store.save("intervention", _key, _stored)
                        _work = "computed"
                    else:
                        _work = "reused"
                    _confirmation_rows.append(_stored)
                    if len(_confirmation_rows) == 1 or len(_confirmation_rows) % 48 == 0:
                        print("confirmation trials", len(_confirmation_rows), _work)

    _confirmation_cells = []
    _raw_comparisons = []
    for _modality in ("text", "image", "spoken_audio"):
        _modality_rows = [
            row for row in _confirmation_rows if row["modality"] == _modality
        ]
        _by_condition = {
            condition: sorted(
                [row for row in _modality_rows if row["condition"] == condition],
                key=lambda row: row["group_id"],
            )
            for condition in ("exact", "zero", "random", "unrelated")
        }
        _exact_outcomes = [bool(row["success"]) for row in _by_condition["exact"]]
        _cell = {
            "modality": _modality,
            "n_photographs": len(_exact_outcomes),
            "exact_successes": sum(_exact_outcomes),
            "exact_success_rate": (
                sum(_exact_outcomes) / len(_exact_outcomes)
                if _exact_outcomes else 0.0
            ),
            "controls": {},
            "integrity_pass": bool(_by_condition["exact"]) and all(
                row["all_prompt_positions_patched"]
                and row["layers_patched"] == list(BROAD_POOLED_BAND)
                and float(row["max_orthogonal_residual_drift"]) <= 1e-5
                and float(row["max_coordinate_update_error"]) <= 1e-5
                for condition_rows in _by_condition.values()
                for row in condition_rows
            ),
            "max_activation_norm_ratio": max(
                (float(row["max_activation_norm_ratio"])
                 for row in _by_condition["exact"]), default=1.0
            ),
            "max_update_to_activation_norm_ratio": max(
                (float(row["max_update_to_activation_norm_ratio"])
                 for row in _by_condition["exact"]), default=0.0
            ),
        }
        for _control in ("zero", "random", "unrelated"):
            _control_outcomes = [
                bool(row["success"]) for row in _by_condition[_control]
            ]
            _test = paired_binary_one_sided_p(
                _exact_outcomes, _control_outcomes
            )
            _comparison = {
                "modality": _modality,
                "control": _control,
                **_test,
            }
            _raw_comparisons.append(_comparison)
            _control_rate = sum(_control_outcomes) / len(_control_outcomes)
            _cell["controls"][_control] = {
                "successes": sum(_control_outcomes),
                "success_rate": _control_rate,
                "exact_minus_control": _cell["exact_success_rate"] - _control_rate,
            }
        _confirmation_cells.append(_cell)
    _adjusted_comparisons = holm_adjust(_raw_comparisons)
    for _comparison in _adjusted_comparisons:
        for _cell in _confirmation_cells:
            if _cell["modality"] == _comparison["modality"]:
                _cell["controls"][_comparison["control"]]["paired_test"] = _comparison

    _confirmation_gate = {
        "capability_population_complete": _confirmation_capability_go,
        "fresh_population_disjoint": not bool(
            set(row["image_id"] for row in _confirmation_population_record)
            & set(_excluded_confirmation_images)
        ),
        "success_rate_in_every_modality": bool(_confirmation_cells) and all(
            cell["exact_success_rate"] >= CONFIRMATION_MIN_SUCCESS_RATE
            for cell in _confirmation_cells
        ),
        "control_margin_in_every_comparison": bool(_confirmation_cells) and all(
            control["exact_minus_control"] >= CONFIRMATION_MIN_CONTROL_MARGIN
            for cell in _confirmation_cells
            for control in cell["controls"].values()
        ),
        "holm_passing_in_every_comparison": bool(_adjusted_comparisons) and all(
            row["holm_adjusted_p"] <= CONFIRMATION_FAMILYWISE_ALPHA
            for row in _adjusted_comparisons
        ),
        "coordinate_integrity_in_every_modality": bool(_confirmation_cells) and all(
            cell["integrity_pass"] for cell in _confirmation_cells
        ),
        "activation_norms_sane": bool(_confirmation_cells) and all(
            cell["max_activation_norm_ratio"] <= 1.25
            and cell["max_update_to_activation_norm_ratio"] <= 0.50
            for cell in _confirmation_cells
        ),
    }
    _confirmation_verdict = (
        "FRESH_MULTIMODAL_CONFIRMATION_GO"
        if all(_confirmation_gate.values())
        else "FRESH_MULTIMODAL_CONFIRMATION_CAPABILITY_NO_GO"
        if not _confirmation_capability_go
        else "FRESH_MULTIMODAL_CONFIRMATION_NO_GO"
    )
    FRESH_CONFIRMATION_REPORT = {
        "schema": "jlens.mmpilot.broad_pooled_multimodal_confirmation.v1",
        "verdict": _confirmation_verdict,
        "scientific_config": _confirmation_config,
        "confirmation_config_digest": _confirmation_config_digest,
        "development_source": CONFIRMATION_SOURCE,
        "population_digest": _confirmation_population_digest,
        "n_fresh_candidates": len(_confirmation_population),
        "n_recruited_photographs": len(_confirmation_recruited),
        "gate": _confirmation_gate,
        "cells": _confirmation_cells,
        "paired_comparisons": _adjusted_comparisons,
        "capability_rows": _confirmation_capability,
        "rows": _confirmation_rows,
        "method": {
            "lens_refitted": False,
            "direction": "bird->cat",
            "alpha": 1.0,
            "layers": list(BROAD_POOLED_BAND),
            "positions": "every original prompt position",
            "teacher_forcing_used": False,
            "candidate_list_supplied": False,
            "endpoint": "unrestricted full-vocabulary next-token top1",
            "independent_unit": "photograph with three synchronized modalities",
            "multiple_testing": "Holm across 3 modalities x 3 controls",
        },
    }
    FRESH_CONFIRMATION_REPORT["report_checksum"] = payload_checksum(
        FRESH_CONFIRMATION_REPORT
    )
    _confirmation_store.save(
        "metric", "fresh_multimodal_confirmation", FRESH_CONFIRMATION_REPORT
    )
    _confirmation_report_path = (
        CONFIRMATION_RUN_DIR / "fresh_multimodal_confirmation_report.json"
    )
    _confirmation_report_path.write_text(
        json.dumps(FRESH_CONFIRMATION_REPORT, indent=2, default=str)
    )
    print("=" * 96)
    print("FRESH MULTIMODAL CONFIRMATION —", _confirmation_verdict)
    print("=" * 96)
    for _cell in _confirmation_cells:
        print(
            _cell["modality"],
            f"exact {_cell['exact_successes']}/{_cell['n_photographs']}",
            "controls",
            {name: value["successes"] for name, value in _cell["controls"].items()},
        )
    print("gate", _confirmation_gate)
    print("report", _confirmation_report_path)
    print("checksum", FRESH_CONFIRMATION_REPORT["report_checksum"])
elif RUN_STAGE3D_FRESH_MULTIMODAL_CONFIRMATION:
    print("Stage 3D requested but blocked by model or confirmation budget.")
'''
)

markdown("## 11. Stage 4 — final report, including null results")
code(
    r'''
if REAL_MODE and REPORT_RUN_DIR is not None:
    RUN_DIR = Path(REPORT_RUN_DIR)
    CROSS_REPORT = json.loads((RUN_DIR / "multimodal_lens_cross_eval_report.json").read_text())
    _causal_path = RUN_DIR / "multimodal_lens_causal_comparison_report.json"
    CAUSAL_REPORT = json.loads(_causal_path.read_text()) if _causal_path.is_file() else None
    _alpha_path = RUN_DIR / "multimodal_lens_alpha_sweep_report.json"
    ALPHA_SWEEP_REPORT = json.loads(_alpha_path.read_text()) if _alpha_path.is_file() else None

if RUN_STAGE4_WRITE_REPORT or not REAL_MODE:
    FINAL = {
        "schema": "jlens.mmpilot.matched_multimodal_jlens_report.v1",
        "scientific_config": SCIENTIFIC_CONFIG,
        "scientific_fingerprint": SCIENTIFIC_FINGERPRINT,
        "lens_checksums": LENS_CHECKSUMS,
        "fit_budget": BUDGET,
        "cross_evaluation": CROSS_REPORT,
        "causal_comparison": CAUSAL_REPORT,
        "alpha_dose_response": ALPHA_SWEEP_REPORT,
        "claim_boundary": (
            "A pooled lens outperforming the text lens would diagnose fitting-"
            "distribution mismatch. It would not by itself establish a shared "
            "workspace or reliable downstream recomputation; those require the "
            "separately reported unrestricted causal endpoint."
        ),
    }
    FINAL["report_checksum"] = payload_checksum(FINAL)
    _path = RUN_DIR / "matched_multimodal_jlens_report.json"
    _path.write_text(json.dumps(FINAL, indent=2, default=str))
    print("=" * 78)
    print("MATCHED MULTIMODAL J-LENS STUDY COMPLETE")
    print("=" * 78)
    print("report", _path)
    print("checksum", FINAL["report_checksum"])
    print("No verdict is promoted beyond the endpoint actually measured.")
else:
    print("Stage 4 not requested. Completed units remain resumable.")
'''
)

markdown(
    r"""
## 12. Read-only artifact and exclusion audit

Which photograph identities are already spent, and where that is recorded.

A photograph is spent the moment the model was run on it in **any** stage,
capability screening included: its clean answer has been seen, and reusing it
would let a known answer leak into a later recruitment. That is why all 64
candidates opened by the completed confirmation are excluded here and not just
the 16 that were recruited from them.

This cell loads nothing but JSON and never touches the model.
"""
)
code(
    r'''
EXCLUSION_UNIVERSE = None
SPENT_CONFIRMATION = None
if REAL_MODE and (RUN_ARTIFACT_EXCLUSION_AUDIT or any(FOLLOWUP_STAGES.values())):
    from jlens.mmpilot.multimodal_followup import (
        exclusion_universe, load_spent_confirmation_population,
    )
    from jlens.mmpilot.multimodal_lens import (
        load_broad_pooled_development_source, load_completed_causal_source,
    )

    if FRESH_CONFIRMATION_RUN_DIR is None:
        raise RuntimeError(
            "set FRESH_CONFIRMATION_RUN_DIR to the Drive folder holding "
            "fresh_multimodal_confirmation_report.json; every follow-up "
            "population must exclude all 64 photographs it opened"
        )
    SPENT_CONFIRMATION = load_spent_confirmation_population(
        FRESH_CONFIRMATION_RUN_DIR,
        expected_report_checksum=EXPECTED_FRESH_CONFIRMATION_REPORT_CHECKSUM,
        expected_candidates=FRESH_CONFIRMATION_CANDIDATES_OPENED,
        expected_recruited=FRESH_CONFIRMATION_IMAGES_RECRUITED,
    )
    _development_source = load_broad_pooled_development_source(
        BROAD_DEVELOPMENT_RUN_DIR,
        expected_report_checksum=EXPECTED_BROAD_DEVELOPMENT_REPORT_CHECKSUM,
        expected_population_digest=EXPECTED_BROAD_DEVELOPMENT_POPULATION_DIGEST,
        expected_lens_checksum=EXPECTED_BROAD_POOLED_LENS_CHECKSUM,
        expected_direction=CONFIRMATION_DIRECTION,
    )
    _prior_causal = load_completed_causal_source(
        CAUSAL_LENS_SOURCE_RUN_DIR,
        expected_final_report_checksum=EXPECTED_SOURCE_FINAL_REPORT_CHECKSUM,
        expected_cross_report_checksum=EXPECTED_SOURCE_CROSS_REPORT_CHECKSUM,
        expected_causal_report_checksum=EXPECTED_SOURCE_CAUSAL_REPORT_CHECKSUM,
        expected_lens_checksums=EXPECTED_SOURCE_LENS_CHECKSUMS,
    )
    EXCLUSION_UNIVERSE = exclusion_universe(
        fit_image_ids=[str(value) for value in PLAN["fit_image_ids"]],
        eval_image_ids=[str(value) for value in PLAN["eval_image_ids"]],
        prior_causal_image_ids=_prior_causal["excluded_image_ids"],
        broad_development_image_ids=_development_source["excluded_image_ids"],
        confirmation_candidate_image_ids=SPENT_CONFIRMATION["candidate_image_ids"],
    )
    print("=" * 78)
    print("ARTIFACT AND EXCLUSION AUDIT (read-only)")
    print("=" * 78)
    print("completed confirmation verdict ", SPENT_CONFIRMATION["verdict"])
    print("  candidates opened            ", SPENT_CONFIRMATION["n_candidates"])
    print("  capability rows              ", SPENT_CONFIRMATION["n_capability_rows"])
    print("  recruited photographs        ", SPENT_CONFIRMATION["n_recruited"])
    print("  all candidates treated spent ", SPENT_CONFIRMATION["all_candidates_spent"])
    for _source, _count in EXCLUSION_UNIVERSE["counts_by_source"].items():
        print(f"  excluded from {_source:<34} {_count}")
    print("TOTAL EXCLUDED IDENTITIES      ", EXCLUSION_UNIVERSE["n_excluded"])
    print("exclusion digest               ", EXCLUSION_UNIVERSE["exclusion_digest"])
    print("existing reports are immutable evidence and are never rewritten here")
elif RUN_ARTIFACT_EXCLUSION_AUDIT:
    print("Artifact audit runs in REAL_MODE against the Drive artifacts.")
'''
)

markdown(
    r"""
## 13. Stage 5A — exploratory band localization

**Exploratory and descriptive. Not confirmation, and not promotable.**

The confirmed study patched the whole validated band, so nothing in it says
where inside L16-L40 the effect lives. This stage varies the band and nothing
else: same pooled lens, same bird->cat direction, same alpha=1 exact exchange,
same prompt, same unrestricted next-token endpoint, same zero/random/unrelated
controls, same every-original-prompt-position rule.

Its population is the broad development population, which is already spent.
That is why the whole analysis is labelled exploratory and why no verdict from
it may be reported as evidence for a localization claim.

The grid has three families. The suffix and prefix families are nested chains,
in which start layer and band length move together — a boundary in either
family cannot be read as an onset, and the report says so in its own claim
boundary. The five-way partition is the only family whose members do not
contain one another; a passing window there is individually sufficient on this
population, and a failing window is not evidence of non-involvement.
"""
)
code(
    r'''
LOCALIZATION_REPORT = None
if REAL_MODE and LOCALIZATION_ENABLED:
    from jlens.lens import JacobianLens
    from jlens.mmpilot.coordinate_swap import (
        random_two_direction_basis, resolve_concept_token,
    )
    from jlens.mmpilot.digit_reasoning_confirmation import resolve_digit_endpoints
    from jlens.mmpilot.multimodal_followup import (
        assert_lens_reused_not_refitted, load_localization_population,
        summarize_localization,
    )
    from jlens.mmpilot.multimodal_lens import (
        build_swap_bases_for_lens, load_broad_pooled_development_source,
        open_answer_matches, unrestricted_swap_trial,
    )
    from jlens.mmpilot.store import RunFingerprint, UnitStore, safe_key

    _source_pin = load_broad_pooled_development_source(
        BROAD_DEVELOPMENT_RUN_DIR,
        expected_report_checksum=EXPECTED_BROAD_DEVELOPMENT_REPORT_CHECKSUM,
        expected_population_digest=EXPECTED_BROAD_DEVELOPMENT_POPULATION_DIGEST,
        expected_lens_checksum=EXPECTED_BROAD_POOLED_LENS_CHECKSUM,
        expected_direction=LOCALIZATION_DIRECTION,
    )
    LOCALIZATION_POPULATION = load_localization_population(
        BROAD_DEVELOPMENT_RUN_DIR,
        expected_report_checksum=EXPECTED_BROAD_DEVELOPMENT_REPORT_CHECKSUM,
        direction=LOCALIZATION_DIRECTION,
    )
    print("localization population is", LOCALIZATION_POPULATION["population_status"])
    print("  licence:", LOCALIZATION_POPULATION["reuse_licence"])
    print("  groups :", LOCALIZATION_POPULATION["n_groups"])

    _localization_config = {
        "study": "exploratory_multimodal_band_localization.v1",
        "label": "exploratory",
        "is_confirmation": False,
        "model_repo_id": MODEL_REPO_ID,
        "model_revision": MODEL_REVISION,
        "audio_protocol_fingerprint": AUDIO_PROTOCOL_FINGERPRINT,
        "manifest_checksum": MANIFEST_CHECKSUM,
        "lens_checksum": EXPECTED_BROAD_POOLED_LENS_CHECKSUM,
        "lens_refitted": False,
        "backward_passes": 0,
        "direction": list(LOCALIZATION_DIRECTION),
        "alpha": LOCALIZATION_ALPHA,
        "grid_digest": LOCALIZATION_GRID["grid_digest"],
        "population_digest": LOCALIZATION_POPULATION["population_digest"],
        "development_report_checksum": EXPECTED_BROAD_DEVELOPMENT_REPORT_CHECKSUM,
        "teacher_forcing_used": False,
        "candidate_list_supplied": False,
        "commit": COMMIT,
    }
    assert_lens_reused_not_refitted(_localization_config)
    _localization_digest = payload_checksum(_localization_config)
    LOCALIZATION_RUN_DIR = (
        RUNS_ROOT / "mmlocalizeband" /
        f"mmlocalizeband_real_{_localization_digest.split(':')[1][:12]}"
    )
    LOCALIZATION_RUN_DIR.mkdir(parents=True, exist_ok=True)
    (LOCALIZATION_RUN_DIR / "scientific_config.json").write_text(
        json.dumps(_localization_config, indent=2)
    )
    (LOCALIZATION_RUN_DIR / "frozen_localization_grid.json").write_text(
        json.dumps(LOCALIZATION_GRID, indent=2)
    )

    _lens = JacobianLens.load(_source_pin["lens_path"])
    if _lens.source_layers != list(BROAD_POOLED_BAND):
        raise RuntimeError("the pinned pooled lens no longer covers L16-L40")

    _by_group = {str(row["group_id"]): row for row in GROUPS}
    _localization_groups = []
    for _row in LOCALIZATION_POPULATION["groups"]:
        _group = _by_group.get(str(_row["group_id"]))
        if _group is None:
            raise RuntimeError(
                f"development group {_row['group_id']} is missing from the "
                "synchronization cache; the population cannot be reproduced"
            )
        _localization_groups.append(_group)

    _tokens = {
        name: resolve_concept_token(BACKEND.encode_candidate, name)
        for name in (*LOCALIZATION_DIRECTION, *BROAD_POOLED_CONTROLS)
    }
    _digits = resolve_digit_endpoints(BACKEND)
    _legs = {"bird": "2", "cat": "4", "zebra": "4", "giraffe": "4"}
    _source_name, _target_name = LOCALIZATION_DIRECTION
    _source_answer_id = int(_digits["token_ids"][_legs[_source_name]])
    _target_answer_id = int(_digits["token_ids"][_legs[_target_name]])

    def _localization_prompt(modality, caption):
        question = (
            "How many legs does the animal in the evidence typically have? "
            "Answer with one digit.\nAnswer:"
        )
        return f"Caption: {caption}\n{question}" if modality == "text" else question

    _localization_fingerprint = RunFingerprint(
        mode="real", model_repo_id=MODEL_REPO_ID, model_revision=MODEL_REVISION,
        processor_revision=MODEL_REVISION, layers=tuple(BROAD_POOLED_BAND),
        lens_checksum=EXPECTED_BROAD_POOLED_LENS_CHECKSUM,
        manifest_checksum=MANIFEST_CHECKSUM,
        split_id=LOCALIZATION_POPULATION["population_digest"],
        intervention_config={
            "grid_digest": LOCALIZATION_GRID["grid_digest"],
            "alpha": LOCALIZATION_ALPHA,
            "conditions": list(LOCALIZATION_GRID["analysis_rule"]["conditions"]),
            "positions": "all_original_prompt_positions",
        },
        extra={"study_digest": _localization_digest},
    )
    _localization_store = UnitStore(LOCALIZATION_RUN_DIR, _localization_fingerprint)
    print("localization run state", _localization_store.open())

    _localization_rows = []
    for _band in LOCALIZATION_GRID["bands"]:
        _band_layers = tuple(int(layer) for layer in _band["layers"])
        _exact_bases = build_swap_bases_for_lens(
            _lens, BACKEND.unembedding_weight(), layers=_band_layers,
            source=_tokens[_source_name], target=_tokens[_target_name],
        )
        _random_bases = {
            layer: random_two_direction_basis(basis, seed=20260823 + layer)
            for layer, basis in _exact_bases.items()
        }
        _unrelated_bases = build_swap_bases_for_lens(
            _lens, BACKEND.unembedding_weight(), layers=_band_layers,
            source=_tokens[BROAD_POOLED_CONTROLS[0]],
            target=_tokens[BROAD_POOLED_CONTROLS[1]],
        )
        _conditions = (
            ("exact", LOCALIZATION_ALPHA, _exact_bases),
            ("zero", 0.0, _exact_bases),
            ("random", LOCALIZATION_ALPHA, _random_bases),
            ("unrelated", LOCALIZATION_ALPHA, _unrelated_bases),
        )
        for _group in _localization_groups:
            for _modality in ("text", "image", "spoken_audio"):
                _inputs = None
                _clean_logits = None
                for _condition, _alpha, _bases in _conditions:
                    _key = safe_key(
                        "loc", _band["name"], _group["group_id"], _modality, _condition
                    )
                    _stored = _localization_store.load("intervention", _key)
                    if _stored is None:
                        if _inputs is None:
                            _inputs = build_group_inputs(
                                _group, _modality,
                                _localization_prompt(_modality, _group["caption"]),
                            )
                            _clean_logits = BACKEND.forward_logits(_inputs.tensors)[
                                0, _inputs.final_prompt_position
                            ].float()
                        _trial = unrestricted_swap_trial(
                            BACKEND, _inputs, bases=_bases, alpha=_alpha,
                            target_token_id=_target_answer_id,
                            source_token_id=_source_answer_id,
                            clean_logits=_clean_logits, compact_positions=True,
                        )
                        _surface = BACKEND.decode_token(
                            _trial["patched_top_token_id"]
                        ).strip()
                        _stored = {
                            **_trial, "band": _band["name"],
                            "group_id": _group["group_id"],
                            "image_id": _group["image_id"],
                            "modality": _modality, "condition": _condition,
                            "expected": _legs[_target_name],
                            "patched_surface": _surface,
                            "success": open_answer_matches(
                                _surface, _legs[_target_name]
                            ),
                        }
                        _localization_store.save("intervention", _key, _stored)
                        _work = "computed"
                    else:
                        _work = "reused"
                    _localization_rows.append(_stored)
                    if len(_localization_rows) % 96 == 0:
                        print("localization trials", len(_localization_rows), _work)

    LOCALIZATION_REPORT = summarize_localization(
        _localization_rows, grid=LOCALIZATION_GRID
    )
    LOCALIZATION_REPORT = {
        **LOCALIZATION_REPORT,
        "scientific_config": _localization_config,
        "population": LOCALIZATION_POPULATION,
        "rows": _localization_rows,
    }
    LOCALIZATION_REPORT["report_checksum"] = payload_checksum(
        {k: v for k, v in LOCALIZATION_REPORT.items() if k != "report_checksum"}
    )
    _localization_store.save("metric", "exploratory_band_localization", LOCALIZATION_REPORT)
    _localization_path = (
        LOCALIZATION_RUN_DIR / "exploratory_band_localization_report.json"
    )
    _localization_path.write_text(
        json.dumps(LOCALIZATION_REPORT, indent=2, default=str)
    )
    print("=" * 96)
    print("EXPLORATORY BAND LOCALIZATION —", LOCALIZATION_REPORT["verdict"])
    print("=" * 96)
    for _cell in LOCALIZATION_REPORT["cells"]:
        print(
            f"  {_cell['band']:<10} {_cell['modality']:<13}"
            f" exact {_cell['exact_successes']}/{_cell['n']}",
            {name: value["successes"] for name, value in _cell["controls"].items()},
        )
    print("bands carrying the effect  ", LOCALIZATION_REPORT["bands_carrying_effect"])
    print("by family                  ", LOCALIZATION_REPORT["bands_by_family"])
    print("onset layer claimed        ",
          LOCALIZATION_REPORT["claim_boundary"]["onset_layer_claimed"])
    print("onset claim                ",
          LOCALIZATION_REPORT["claim_boundary"]["onset_claim"])
    print("necessity claimed          ",
          LOCALIZATION_REPORT["claim_boundary"]["necessity_claimed"])
    print("report", _localization_path)
    print("checksum", LOCALIZATION_REPORT["report_checksum"])
elif RUN_STAGE5A_BAND_LOCALIZATION:
    print("Stage 5A requested but blocked by the model or localization budget.")
'''
)

markdown(
    r"""
## 14. Stage 5B0 + 5B1 — new-property audit and development

Leg count cannot carry a generalization claim: `bird=2`, `cat=4`, `zebra=4`,
`giraffe=4`, so bird->cat, bird->zebra and bird->giraffe all test the same
2 -> 4 answer change, and cat->zebra changes no observable answer at all.

Stage 5B0 audits a candidate property before any causal spending:

* semantic admissibility, declared per concept with a written reason. A concept
  whose correct surface answer is contested is refused — horse, cow, zebra,
  giraffe and elephant are all refused for body covering, and *bird* is refused
  for animal sound because COCO birds have no single conventional sound;
* media availability on genuinely fresh photographs;
* clean capability in text, image and spoken audio at the declared rate.

A direction survives only if both endpoints survive all three **and** their two
answers differ. The endpoint is unrestricted complete generation scored after
the fact against predeclared aliases — answers are not required to be single
tokens, and no candidate list or teacher forcing is ever supplied.

Stage 5B1 then runs the exact alpha=1 exchange over L16-L40 with the same three
controls on that fresh development population.
"""
)
code(
    r'''
PROPERTY_AUDIT_REPORT = None
NEW_PROPERTY_DEVELOPMENT_REPORT = None
if REAL_MODE and (PROPERTY_AUDIT_ENABLED or NEW_PROPERTY_DEV_ENABLED):
    from jlens.lens import JacobianLens
    from jlens.mmpilot.coordinate_swap import (
        random_two_direction_basis, resolve_concept_token,
    )
    from jlens.mmpilot.multimodal_followup import (
        PROPERTY_FAMILIES, artifact_exclusion_audit, assert_lens_reused_not_refitted,
        assert_property_pair_changes_answer, audit_property_family,
        generation_trial_row, new_property_development_verdict,
        property_answer_matches,
    )
    from jlens.mmpilot.multimodal_lens import (
        build_swap_bases_for_lens, load_broad_pooled_development_source,
        select_causal_groups,
    )
    from jlens.mmpilot.store import RunFingerprint, UnitStore, safe_key
    from jlens.mmpilot.workspace_replication import (
        unrestricted_greedy_completion, unrestricted_greedy_swap_trial,
    )

    _property = PROPERTY_FAMILIES[NEW_PROPERTY_FAMILY]
    _source_pin = load_broad_pooled_development_source(
        BROAD_DEVELOPMENT_RUN_DIR,
        expected_report_checksum=EXPECTED_BROAD_DEVELOPMENT_REPORT_CHECKSUM,
        expected_population_digest=EXPECTED_BROAD_DEVELOPMENT_POPULATION_DIGEST,
        expected_lens_checksum=EXPECTED_BROAD_POOLED_LENS_CHECKSUM,
        expected_direction=CONFIRMATION_DIRECTION,
    )
    # Every direction is checked against the property table before any media
    # are opened; a pair that does not change the answer cannot be requested.
    for _pair in NEW_PROPERTY_DEV_DIRECTIONS:
        assert_property_pair_changes_answer(NEW_PROPERTY_FAMILY, _pair[0], _pair[1])

    _dev_config = {
        "study": "multimodal_new_property_development.v1",
        "property_family": NEW_PROPERTY_FAMILY,
        "prompt": _property.question,
        "max_new_tokens": NEW_PROPERTY_MAX_NEW_TOKENS,
        "model_repo_id": MODEL_REPO_ID,
        "model_revision": MODEL_REVISION,
        "audio_protocol_fingerprint": AUDIO_PROTOCOL_FINGERPRINT,
        "manifest_checksum": MANIFEST_CHECKSUM,
        "lens_checksum": EXPECTED_BROAD_POOLED_LENS_CHECKSUM,
        "lens_refitted": False,
        "backward_passes": 0,
        "layers": list(BROAD_POOLED_BAND),
        "alpha": 1.0,
        "positions": "every original prompt position",
        "concepts": list(NEW_PROPERTY_CONCEPTS),
        "directions": [list(pair) for pair in NEW_PROPERTY_DEV_DIRECTIONS],
        "controls": ["zero", "random", "unrelated"],
        "candidates_per_concept": NEW_PROPERTY_DEV_CANDIDATES_PER_CONCEPT,
        "images_per_direction": NEW_PROPERTY_DEV_IMAGES_PER_DIRECTION,
        "min_success_rate": NEW_PROPERTY_DEV_MIN_SUCCESS_RATE,
        "min_control_margin": NEW_PROPERTY_DEV_MIN_CONTROL_MARGIN,
        "min_clean_capability_rate": NEW_PROPERTY_MIN_CLEAN_CAPABILITY_RATE,
        "exclusion_digest": EXCLUSION_UNIVERSE["exclusion_digest"],
        "seed": NEW_PROPERTY_DEV_SEED,
        "teacher_forcing_used": False,
        "candidate_list_supplied": False,
        "commit": COMMIT,
    }
    assert_lens_reused_not_refitted(_dev_config)
    _dev_digest = payload_checksum(_dev_config)
    NEW_PROPERTY_DEV_RUN_DIR = (
        RUNS_ROOT / "mmnewproperty" /
        f"mmnewpropertydev_real_{_dev_digest.split(':')[1][:12]}"
    )
    NEW_PROPERTY_DEV_RUN_DIR.mkdir(parents=True, exist_ok=True)
    (NEW_PROPERTY_DEV_RUN_DIR / "scientific_config.json").write_text(
        json.dumps(_dev_config, indent=2)
    )

    # Fresh development media: selected before any answer is opened, and
    # disjoint from every spent identity including all 64 confirmation
    # candidates.
    _forbidden = {
        concept: tuple(
            other for other in NEW_PROPERTY_CONCEPTS if other != concept
        )
        for concept in NEW_PROPERTY_CONCEPTS
    }
    _dev_population = select_causal_groups(
        GROUPS, concepts=NEW_PROPERTY_CONCEPTS,
        n_per_concept=NEW_PROPERTY_DEV_CANDIDATES_PER_CONCEPT,
        excluded_image_ids=EXCLUSION_UNIVERSE["excluded_image_ids"],
        seed=NEW_PROPERTY_DEV_SEED, forbidden_concepts=_forbidden,
    )
    _dev_flat = [row for rows in _dev_population.values() for row in rows]
    NEW_PROPERTY_DEV_EXCLUSION_AUDIT = artifact_exclusion_audit(
        _dev_flat, universe=EXCLUSION_UNIVERSE, label="new_property_development"
    )
    (NEW_PROPERTY_DEV_RUN_DIR / "development_population.json").write_text(
        json.dumps({
            "population": {
                concept: [
                    {"group_id": row["group_id"], "image_id": row["image_id"]}
                    for row in rows
                ]
                for concept, rows in _dev_population.items()
            },
            "exclusion_audit": NEW_PROPERTY_DEV_EXCLUSION_AUDIT,
            "selected_before_any_answer_opened": True,
        }, indent=2)
    )
    print("fresh development media selected; exclusion audit disjoint =",
          NEW_PROPERTY_DEV_EXCLUSION_AUDIT["disjoint"])

    _dev_fingerprint = RunFingerprint(
        mode="real", model_repo_id=MODEL_REPO_ID, model_revision=MODEL_REVISION,
        processor_revision=MODEL_REVISION, layers=tuple(BROAD_POOLED_BAND),
        lens_checksum=EXPECTED_BROAD_POOLED_LENS_CHECKSUM,
        manifest_checksum=MANIFEST_CHECKSUM,
        split_id=NEW_PROPERTY_DEV_EXCLUSION_AUDIT["audit_digest"],
        intervention_config={
            "property_family": NEW_PROPERTY_FAMILY,
            "alpha": 1.0,
            "conditions": ["exact", "zero", "random", "unrelated"],
            "positions": "all_original_prompt_positions",
            "max_new_tokens": NEW_PROPERTY_MAX_NEW_TOKENS,
        },
        extra={"study_digest": _dev_digest},
    )
    _dev_store = UnitStore(NEW_PROPERTY_DEV_RUN_DIR, _dev_fingerprint)
    print("new-property development run state", _dev_store.open())

    # ---- Stage 5B0: clean capability on the untouched model only -----------
    _capability_rows = []
    for _concept in NEW_PROPERTY_CONCEPTS:
        _answer = _property.answer_for(_concept)
        for _group in _dev_population[_concept]:
            for _modality in ("text", "image", "spoken_audio"):
                _key = safe_key("propcap", _concept, _group["group_id"], _modality)
                _row = _dev_store.load("capability", _key)
                if _row is None:
                    _inputs = build_group_inputs(
                        _group, _modality,
                        _property.prompt(_modality, _group["caption"]),
                    )
                    _clean = unrestricted_greedy_completion(
                        BACKEND, _inputs, answer=_answer.answer,
                        max_new_tokens=NEW_PROPERTY_MAX_NEW_TOKENS,
                    )
                    _row = {
                        "concept": _concept, "group_id": _group["group_id"],
                        "image_id": _group["image_id"], "modality": _modality,
                        "expected_aliases": list(_answer.aliases),
                        "generated": _clean["generated_text"],
                        "pass": property_answer_matches(
                            _clean["generated_text"], _answer
                        ),
                    }
                    _dev_store.save("capability", _key, _row)
                _capability_rows.append(_row)
                if len(_capability_rows) % 72 == 0:
                    print("property capability", len(_capability_rows))

    _capability_by_concept = {
        concept: {
            modality: (
                sum(
                    bool(row["pass"]) for row in _capability_rows
                    if row["concept"] == concept and row["modality"] == modality
                ) / max(1, sum(
                    1 for row in _capability_rows
                    if row["concept"] == concept and row["modality"] == modality
                ))
            )
            for modality in ("text", "image", "spoken_audio")
        }
        for concept in NEW_PROPERTY_CONCEPTS
    }
    PROPERTY_AUDIT_REPORT = audit_property_family(
        NEW_PROPERTY_FAMILY,
        available_media={
            concept: len(rows) for concept, rows in _dev_population.items()
        },
        min_media_per_concept=NEW_PROPERTY_DEV_CANDIDATES_PER_CONCEPT,
        clean_capability=_capability_by_concept,
        min_clean_capability_rate=NEW_PROPERTY_MIN_CLEAN_CAPABILITY_RATE,
    )
    PROPERTY_AUDIT_REPORT = {
        **PROPERTY_AUDIT_REPORT,
        "clean_capability_by_concept": _capability_by_concept,
        "capability_rows": _capability_rows,
        "fallback_family_if_no_go": NEW_PROPERTY_FALLBACK_FAMILY,
    }
    _audit_path = NEW_PROPERTY_DEV_RUN_DIR / "new_property_audit_report.json"
    _audit_path.write_text(
        json.dumps(PROPERTY_AUDIT_REPORT, indent=2, default=str)
    )
    print("=" * 96)
    print("PROPERTY AUDIT —", PROPERTY_AUDIT_REPORT["verdict"])
    print("=" * 96)
    print("family              ", PROPERTY_AUDIT_REPORT["family"])
    print("admissible concepts ", PROPERTY_AUDIT_REPORT["admissible_concepts"])
    print("refused concepts    ",
          [row["concept"] for row in PROPERTY_AUDIT_REPORT["refused_concepts"]])
    print("usable after data   ", PROPERTY_AUDIT_REPORT["usable_concepts"])
    print("clean capability    ", _capability_by_concept)
    print("candidate directions",
          [row["direction"] for row in PROPERTY_AUDIT_REPORT["candidate_directions"]])
    print("audit report", _audit_path)

    # ---- Stage 5B1: the exact exchange on the fresh development media ------
    if NEW_PROPERTY_DEV_ENABLED and PROPERTY_AUDIT_REPORT["verdict"] == "PROPERTY_AUDIT_GO":
        _lens = JacobianLens.load(_source_pin["lens_path"])
        _tokens = {
            name: resolve_concept_token(BACKEND.encode_candidate, name)
            for name in (*NEW_PROPERTY_CONCEPTS, *BROAD_POOLED_CONTROLS)
        }
        _unrelated_bases = build_swap_bases_for_lens(
            _lens, BACKEND.unembedding_weight(), layers=BROAD_POOLED_BAND,
            source=_tokens[BROAD_POOLED_CONTROLS[0]],
            target=_tokens[BROAD_POOLED_CONTROLS[1]],
        )
        _usable = set(PROPERTY_AUDIT_REPORT["usable_concepts"])
        _recruited = {}
        for _concept in NEW_PROPERTY_CONCEPTS:
            _eligible = []
            for _group in _dev_population[_concept]:
                _rows = [
                    row for row in _capability_rows
                    if row["concept"] == _concept
                    and row["group_id"] == _group["group_id"]
                ]
                if len(_rows) == 3 and all(row["pass"] for row in _rows):
                    _eligible.append(_group)
            _recruited[_concept] = _eligible[:NEW_PROPERTY_DEV_IMAGES_PER_DIRECTION]
        _capability_go = all(
            len(_recruited[concept]) == NEW_PROPERTY_DEV_IMAGES_PER_DIRECTION
            for concept in NEW_PROPERTY_CONCEPTS if concept in _usable
        ) and bool(_usable)
        print("recruited", {k: len(v) for k, v in _recruited.items()},
              "capability_go", _capability_go)

        _dev_rows = []
        if _capability_go:
            for _pair in NEW_PROPERTY_DEV_DIRECTIONS:
                _src, _tgt = _pair
                if not {_src, _tgt} <= _usable:
                    print("skipping", f"{_src}->{_tgt}", "— a concept is unusable")
                    continue
                _target_answer = _property.answer_for(_tgt)
                _exact_bases = build_swap_bases_for_lens(
                    _lens, BACKEND.unembedding_weight(), layers=BROAD_POOLED_BAND,
                    source=_tokens[_src], target=_tokens[_tgt],
                )
                _random_bases = {
                    layer: random_two_direction_basis(basis, seed=20260823 + layer)
                    for layer, basis in _exact_bases.items()
                }
                _conditions = (
                    ("exact", 1.0, _exact_bases),
                    ("zero", 0.0, _exact_bases),
                    ("random", 1.0, _random_bases),
                    ("unrelated", 1.0, _unrelated_bases),
                )
                for _group in _recruited[_src]:
                    for _modality in ("text", "image", "spoken_audio"):
                        for _condition, _alpha, _bases in _conditions:
                            _key = safe_key(
                                "proptrial", _src, _tgt, _group["group_id"],
                                _modality, _condition,
                            )
                            _stored = _dev_store.load("intervention", _key)
                            if _stored is None:
                                _inputs = build_group_inputs(
                                    _group, _modality,
                                    _property.prompt(_modality, _group["caption"]),
                                )
                                _trial = unrestricted_greedy_swap_trial(
                                    BACKEND, _inputs, bases=_bases, alpha=_alpha,
                                    answer=_target_answer.answer,
                                    max_new_tokens=NEW_PROPERTY_MAX_NEW_TOKENS,
                                )
                                _stored = generation_trial_row(
                                    _trial, group=_group, modality=_modality,
                                    condition=_condition, direction=(_src, _tgt),
                                    answer=_target_answer,
                                    layers=BROAD_POOLED_BAND,
                                )
                                _dev_store.save("intervention", _key, _stored)
                            _dev_rows.append(_stored)
                            if len(_dev_rows) % 96 == 0:
                                print("new-property trials", len(_dev_rows))

        NEW_PROPERTY_DEVELOPMENT_REPORT = new_property_development_verdict(
            _dev_rows, audit=PROPERTY_AUDIT_REPORT, layers=BROAD_POOLED_BAND,
            capability_go=_capability_go,
            min_success_rate=NEW_PROPERTY_DEV_MIN_SUCCESS_RATE,
            min_control_margin=NEW_PROPERTY_DEV_MIN_CONTROL_MARGIN,
        )
        NEW_PROPERTY_DEVELOPMENT_REPORT = {
            **NEW_PROPERTY_DEVELOPMENT_REPORT,
            "scientific_config": _dev_config,
            "exclusion_audit": NEW_PROPERTY_DEV_EXCLUSION_AUDIT,
            "recruited_counts": {k: len(v) for k, v in _recruited.items()},
            "rows": _dev_rows,
        }
        NEW_PROPERTY_DEVELOPMENT_REPORT["report_checksum"] = payload_checksum({
            k: v for k, v in NEW_PROPERTY_DEVELOPMENT_REPORT.items()
            if k != "report_checksum"
        })
        _dev_store.save(
            "metric", "new_property_development", NEW_PROPERTY_DEVELOPMENT_REPORT
        )
        _dev_path = (
            NEW_PROPERTY_DEV_RUN_DIR / "new_property_development_report.json"
        )
        _dev_path.write_text(
            json.dumps(NEW_PROPERTY_DEVELOPMENT_REPORT, indent=2, default=str)
        )
        print("=" * 96)
        print("NEW-PROPERTY DEVELOPMENT —",
              NEW_PROPERTY_DEVELOPMENT_REPORT["verdict"])
        print("=" * 96)
        print("passing directions ",
              NEW_PROPERTY_DEVELOPMENT_REPORT["passing_directions"])
        print("failure modes      ",
              NEW_PROPERTY_DEVELOPMENT_REPORT["failure_modes"])
        print("run dir            ", NEW_PROPERTY_DEV_RUN_DIR)
        print("report             ", _dev_path)
        print("checksum           ",
              NEW_PROPERTY_DEVELOPMENT_REPORT["report_checksum"])
        if NEW_PROPERTY_DEVELOPMENT_REPORT["verdict"] != "NEW_PROPERTY_DEVELOPMENT_GO":
            print("NO_GO: confirmation stays closed. Nothing is re-thresholded.")
elif RUN_STAGE5B0_PROPERTY_AUDIT or RUN_STAGE5B1_NEW_PROPERTY_DEVELOPMENT:
    print("Stage 5B0/5B1 requested but blocked by the model or development budget.")
'''
)

markdown(
    r"""
## 15. Stage 5B2 — freeze the confirmation design

CPU-only. Reads the development report by checksum, applies the predeclared
tie-break if several directions passed, and writes the frozen design artifact.

Stage 5B3 refuses to open a fresh photograph until this file exists and its own
digest verifies, so the freeze genuinely precedes the confirmation population.
"""
)
code(
    r'''
FROZEN_NEW_PROPERTY_DESIGN = None
if REAL_MODE and NEW_PROPERTY_FREEZE_ENABLED:
    from jlens.mmpilot.multimodal_followup import (
        exclusion_universe, freeze_new_property_design, load_verified_report,
    )

    if NEW_PROPERTY_DEVELOPMENT_RUN_DIR is None or EXPECTED_NEW_PROPERTY_DEVELOPMENT_CHECKSUM is None:
        raise RuntimeError(
            "set NEW_PROPERTY_DEVELOPMENT_RUN_DIR and "
            "EXPECTED_NEW_PROPERTY_DEVELOPMENT_CHECKSUM from the Stage 5B1 run"
        )
    _development = load_verified_report(
        Path(NEW_PROPERTY_DEVELOPMENT_RUN_DIR) / "new_property_development_report.json",
        expected_checksum=EXPECTED_NEW_PROPERTY_DEVELOPMENT_CHECKSUM,
        label="new-property development report",
    )
    _development["report_checksum"] = EXPECTED_NEW_PROPERTY_DEVELOPMENT_CHECKSUM
    _audit = json.loads(
        (Path(NEW_PROPERTY_DEVELOPMENT_RUN_DIR) / "new_property_audit_report.json").read_text()
    )
    _passing = list(_development.get("passing_directions") or [])
    _chosen = next(
        (name for name in NEW_PROPERTY_DIRECTION_PRIORITY if name in _passing), None
    )
    if _chosen is None:
        raise RuntimeError(
            f"development licensed no direction ({_development['verdict']}); "
            "confirmation stays closed"
        )
    print("predeclared tie-break selected", _chosen, "from", _passing)

    # Every identity Stage 5B0/5B1 opened — interventions and the clean
    # capability screen alike. Stage 5B3 recomputes this same set and refuses
    # to run if the two disagree.
    _dev_images = sorted({
        str(row["image_id"]) for row in _development.get("rows") or []
    } | {
        str(row["image_id"]) for row in _audit.get("capability_rows") or []
    })
    _confirm_exclusions = exclusion_universe(
        fit_image_ids=EXCLUSION_UNIVERSE["sources"]["fit"],
        eval_image_ids=EXCLUSION_UNIVERSE["sources"]["cross_evaluation"],
        prior_causal_image_ids=EXCLUSION_UNIVERSE["sources"]["prior_causal_screens"],
        broad_development_image_ids=EXCLUSION_UNIVERSE["sources"]["broad_development"],
        confirmation_candidate_image_ids=EXCLUSION_UNIVERSE["sources"][
            "confirmation_candidates_all_opened"
        ],
        extra_image_ids={"new_property_development_opened": _dev_images},
    )
    FROZEN_NEW_PROPERTY_DESIGN = freeze_new_property_design(
        development=_development, audit=_audit,
        direction=tuple(_chosen.split("->")),
        lens_checksum=EXPECTED_BROAD_POOLED_LENS_CHECKSUM,
        layers=BROAD_POOLED_BAND, alpha=1.0,
        exclusions=_confirm_exclusions,
        n_candidates=NEW_PROPERTY_CONFIRM_CANDIDATES,
        n_recruited=NEW_PROPERTY_CONFIRM_IMAGES,
        min_success_rate=NEW_PROPERTY_CONFIRM_MIN_SUCCESS_RATE,
        min_control_margin=NEW_PROPERTY_CONFIRM_MIN_CONTROL_MARGIN,
        min_clean_capability_rate=NEW_PROPERTY_MIN_CLEAN_CAPABILITY_RATE,
        familywise_alpha=NEW_PROPERTY_CONFIRM_FAMILYWISE_ALPHA,
        recruitment_rule=(
            "clean property capability in all three modalities, in the frozen "
            "candidate order, before any intervention runs"
        ),
        seed=NEW_PROPERTY_CONFIRM_SEED,
    )
    _design_path = (
        Path(NEW_PROPERTY_DEVELOPMENT_RUN_DIR) / "frozen_new_property_design.json"
    )
    _temporary = _design_path.with_suffix(".tmp.json")
    _temporary.write_text(
        json.dumps(FROZEN_NEW_PROPERTY_DESIGN, indent=2, default=str)
    )
    os.replace(_temporary, _design_path)
    print("=" * 96)
    print("NEW-PROPERTY CONFIRMATION DESIGN FROZEN")
    print("=" * 96)
    print("direction        ", FROZEN_NEW_PROPERTY_DESIGN["direction"])
    print("property family  ", FROZEN_NEW_PROPERTY_DESIGN["property_family"])
    print("answer aliases   ", FROZEN_NEW_PROPERTY_DESIGN["answer_aliases"])
    print("layers / alpha   ", FROZEN_NEW_PROPERTY_DESIGN["layers"][0], "-",
          FROZEN_NEW_PROPERTY_DESIGN["layers"][-1], "/",
          FROZEN_NEW_PROPERTY_DESIGN["alpha"])
    print("excluded ids     ", FROZEN_NEW_PROPERTY_DESIGN["n_excluded_identities"])
    print("design digest    ", FROZEN_NEW_PROPERTY_DESIGN["design_digest"])
    print("path             ", _design_path)
    print("Set NEW_PROPERTY_FROZEN_DESIGN_PATH to this path before Stage 5B3.")
elif RUN_STAGE5B2_FREEZE_NEW_PROPERTY_DESIGN:
    print("Stage 5B2 runs on CPU in REAL_MODE against the Drive artifacts.")
'''
)

markdown(
    r"""
## 16. Stage 5B3 — fresh new-property confirmation

Runs the frozen design and nothing else. The population is selected before any
answer is opened and excludes every spent identity, including all 64
photographs the completed bird->cat confirmation opened. Thresholds, prompt,
pair, aliases and recruitment rule come from the frozen file; none of them can
be revised after outcomes are seen, and every failure and raw generation is
preserved in the report.
"""
)
code(
    r'''
NEW_PROPERTY_CONFIRMATION_REPORT = None
if REAL_MODE and NEW_PROPERTY_CONFIRM_ENABLED:
    from jlens.lens import JacobianLens
    from jlens.mmpilot.coordinate_swap import (
        random_two_direction_basis, resolve_concept_token,
    )
    from jlens.mmpilot.multimodal_followup import (
        PROPERTY_FAMILIES, artifact_exclusion_audit, assert_design_frozen,
        confirmation_verdict, exclusion_universe, generation_trial_row,
        property_answer_matches,
    )
    from jlens.mmpilot.multimodal_lens import (
        build_swap_bases_for_lens, load_broad_pooled_development_source,
        select_causal_groups,
    )
    from jlens.mmpilot.store import RunFingerprint, UnitStore, safe_key
    from jlens.mmpilot.workspace_replication import (
        unrestricted_greedy_completion, unrestricted_greedy_swap_trial,
    )

    if NEW_PROPERTY_FROZEN_DESIGN_PATH is None:
        raise RuntimeError(
            "Stage 5B3 cannot open a fresh photograph before Stage 5B2 wrote "
            "the frozen design; set NEW_PROPERTY_FROZEN_DESIGN_PATH"
        )
    DESIGN = assert_design_frozen(NEW_PROPERTY_FROZEN_DESIGN_PATH)
    print("frozen design verified", DESIGN["design_digest"])
    print("  direction", DESIGN["direction"], "property", DESIGN["property_family"])
    print("  thresholds", DESIGN["thresholds"])

    _property = PROPERTY_FAMILIES[DESIGN["property_family"]]
    _src, _tgt = DESIGN["direction"]
    _source_answer = _property.answer_for(_src)
    _target_answer = _property.answer_for(_tgt)
    _source_pin = load_broad_pooled_development_source(
        BROAD_DEVELOPMENT_RUN_DIR,
        expected_report_checksum=EXPECTED_BROAD_DEVELOPMENT_REPORT_CHECKSUM,
        expected_population_digest=EXPECTED_BROAD_DEVELOPMENT_POPULATION_DIGEST,
        expected_lens_checksum=EXPECTED_BROAD_POOLED_LENS_CHECKSUM,
        expected_direction=CONFIRMATION_DIRECTION,
    )
    if DESIGN["lens_checksum"] != EXPECTED_BROAD_POOLED_LENS_CHECKSUM:
        raise RuntimeError("the frozen design pins a different pooled lens")

    _confirm_config = {
        "study": "multimodal_new_property_confirmation.v1",
        "design_digest": DESIGN["design_digest"],
        "model_repo_id": MODEL_REPO_ID, "model_revision": MODEL_REVISION,
        "manifest_checksum": MANIFEST_CHECKSUM,
        "audio_protocol_fingerprint": AUDIO_PROTOCOL_FINGERPRINT,
        "commit": COMMIT,
    }
    _confirm_digest = payload_checksum(_confirm_config)
    NEW_PROPERTY_CONFIRM_RUN_DIR = (
        RUNS_ROOT / "mmnewpropertyconfirm" /
        f"mmnewpropertyconfirm_real_{_confirm_digest.split(':')[1][:12]}"
    )
    NEW_PROPERTY_CONFIRM_RUN_DIR.mkdir(parents=True, exist_ok=True)
    (NEW_PROPERTY_CONFIRM_RUN_DIR / "scientific_config.json").write_text(
        json.dumps(_confirm_config, indent=2)
    )

    def _development_opened_image_ids():
        """Every identity Stage 5B0/5B1 opened, recomputed rather than retyped."""

        if NEW_PROPERTY_DEVELOPMENT_RUN_DIR is None:
            raise RuntimeError(
                "set NEW_PROPERTY_DEVELOPMENT_RUN_DIR so the development media "
                "can be excluded from the fresh confirmation population"
            )
        root = Path(NEW_PROPERTY_DEVELOPMENT_RUN_DIR)
        development = json.loads(
            (root / "new_property_development_report.json").read_text()
        )
        audit = json.loads((root / "new_property_audit_report.json").read_text())
        return sorted(
            {str(row["image_id"]) for row in development.get("rows") or []}
            | {str(row["image_id"]) for row in audit.get("capability_rows") or []}
        )

    _confirm_exclusions = exclusion_universe(
        fit_image_ids=EXCLUSION_UNIVERSE["sources"]["fit"],
        eval_image_ids=EXCLUSION_UNIVERSE["sources"]["cross_evaluation"],
        prior_causal_image_ids=EXCLUSION_UNIVERSE["sources"]["prior_causal_screens"],
        broad_development_image_ids=EXCLUSION_UNIVERSE["sources"]["broad_development"],
        confirmation_candidate_image_ids=EXCLUSION_UNIVERSE["sources"][
            "confirmation_candidates_all_opened"
        ],
        extra_image_ids={
            "new_property_development_opened": _development_opened_image_ids(),
        },
    )
    if _confirm_exclusions["exclusion_digest"] != DESIGN["exclusion_digest"]:
        raise RuntimeError(
            "the exclusion universe differs from the one frozen in the design; "
            "refusing to mix populations"
        )
    _confirm_population = select_causal_groups(
        GROUPS, concepts=(_src,),
        n_per_concept=int(DESIGN["n_candidates"]),
        excluded_image_ids=_confirm_exclusions["excluded_image_ids"],
        seed=DESIGN["seed"], forbidden_concepts={_src: (_tgt,)},
    )[_src]
    CONFIRM_EXCLUSION_AUDIT = artifact_exclusion_audit(
        _confirm_population, universe=_confirm_exclusions,
        label="new_property_confirmation",
    )
    (NEW_PROPERTY_CONFIRM_RUN_DIR / "fresh_population.json").write_text(
        json.dumps({
            "population": [
                {"group_id": row["group_id"], "image_id": row["image_id"]}
                for row in _confirm_population
            ],
            "exclusion_audit": CONFIRM_EXCLUSION_AUDIT,
            "selected_before_capability": True,
        }, indent=2)
    )
    print("fresh confirmation population disjoint =", CONFIRM_EXCLUSION_AUDIT["disjoint"])

    _confirm_fingerprint = RunFingerprint(
        mode="real", model_repo_id=MODEL_REPO_ID, model_revision=MODEL_REVISION,
        processor_revision=MODEL_REVISION, layers=tuple(DESIGN["layers"]),
        lens_checksum=DESIGN["lens_checksum"], manifest_checksum=MANIFEST_CHECKSUM,
        split_id=CONFIRM_EXCLUSION_AUDIT["audit_digest"],
        intervention_config={
            "direction": list(DESIGN["direction"]),
            "alpha": DESIGN["alpha"],
            "conditions": list(DESIGN["conditions"]),
            "positions": "all_original_prompt_positions",
            "max_new_tokens": DESIGN["max_new_tokens"],
        },
        extra={"design_digest": DESIGN["design_digest"]},
    )
    _confirm_store = UnitStore(NEW_PROPERTY_CONFIRM_RUN_DIR, _confirm_fingerprint)
    print("confirmation run state", _confirm_store.open())

    _confirm_capability = []
    for _group in _confirm_population:
        for _modality in ("text", "image", "spoken_audio"):
            _key = safe_key("npcap", _group["group_id"], _modality)
            _row = _confirm_store.load("capability", _key)
            if _row is None:
                _inputs = build_group_inputs(
                    _group, _modality,
                    _property.prompt(_modality, _group["caption"]),
                )
                _clean = unrestricted_greedy_completion(
                    BACKEND, _inputs, answer=_source_answer.answer,
                    max_new_tokens=int(DESIGN["max_new_tokens"]),
                )
                _row = {
                    "group_id": _group["group_id"], "image_id": _group["image_id"],
                    "modality": _modality,
                    "expected_aliases": list(_source_answer.aliases),
                    "generated": _clean["generated_text"],
                    "pass": property_answer_matches(
                        _clean["generated_text"], _source_answer
                    ),
                }
                _confirm_store.save("capability", _key, _row)
            _confirm_capability.append(_row)
            if len(_confirm_capability) % 48 == 0:
                print("confirmation capability", len(_confirm_capability))

    _recruited = []
    for _group in _confirm_population:
        _rows = [
            row for row in _confirm_capability
            if row["group_id"] == _group["group_id"]
        ]
        if len(_rows) == 3 and all(row["pass"] for row in _rows):
            _recruited.append(_group)
        if len(_recruited) == int(DESIGN["n_recruited"]):
            break
    _capability_go = len(_recruited) == int(DESIGN["n_recruited"])
    print("confirmation recruited", len(_recruited), "/", DESIGN["n_recruited"])

    _confirm_rows = []
    if _capability_go:
        _lens = JacobianLens.load(_source_pin["lens_path"])
        _tokens = {
            name: resolve_concept_token(BACKEND.encode_candidate, name)
            for name in (_src, _tgt, *BROAD_POOLED_CONTROLS)
        }
        _exact_bases = build_swap_bases_for_lens(
            _lens, BACKEND.unembedding_weight(), layers=tuple(DESIGN["layers"]),
            source=_tokens[_src], target=_tokens[_tgt],
        )
        _random_bases = {
            layer: random_two_direction_basis(basis, seed=20260823 + layer)
            for layer, basis in _exact_bases.items()
        }
        _unrelated_bases = build_swap_bases_for_lens(
            _lens, BACKEND.unembedding_weight(), layers=tuple(DESIGN["layers"]),
            source=_tokens[BROAD_POOLED_CONTROLS[0]],
            target=_tokens[BROAD_POOLED_CONTROLS[1]],
        )
        _conditions = (
            ("exact", float(DESIGN["alpha"]), _exact_bases),
            ("zero", 0.0, _exact_bases),
            ("random", float(DESIGN["alpha"]), _random_bases),
            ("unrelated", float(DESIGN["alpha"]), _unrelated_bases),
        )
        for _group in _recruited:
            for _modality in ("text", "image", "spoken_audio"):
                for _condition, _alpha, _bases in _conditions:
                    _key = safe_key(
                        "nptrial", _group["group_id"], _modality, _condition
                    )
                    _stored = _confirm_store.load("intervention", _key)
                    if _stored is None:
                        _inputs = build_group_inputs(
                            _group, _modality,
                            _property.prompt(_modality, _group["caption"]),
                        )
                        _trial = unrestricted_greedy_swap_trial(
                            BACKEND, _inputs, bases=_bases, alpha=_alpha,
                            answer=_target_answer.answer,
                            max_new_tokens=int(DESIGN["max_new_tokens"]),
                        )
                        _stored = generation_trial_row(
                            _trial, group=_group, modality=_modality,
                            condition=_condition, direction=(_src, _tgt),
                            answer=_target_answer, layers=DESIGN["layers"],
                        )
                        _confirm_store.save("intervention", _key, _stored)
                    _confirm_rows.append(_stored)
                    if len(_confirm_rows) % 48 == 0:
                        print("confirmation trials", len(_confirm_rows))

    NEW_PROPERTY_CONFIRMATION_REPORT = confirmation_verdict(
        _confirm_rows, design=DESIGN, capability_go=_capability_go,
        exclusion_audit=CONFIRM_EXCLUSION_AUDIT,
    )
    NEW_PROPERTY_CONFIRMATION_REPORT = {
        **NEW_PROPERTY_CONFIRMATION_REPORT,
        "scientific_config": _confirm_config,
        "frozen_design": DESIGN,
        "capability_rows": _confirm_capability,
        "n_fresh_candidates": len(_confirm_population),
        "n_recruited": len(_recruited),
    }
    NEW_PROPERTY_CONFIRMATION_REPORT["report_checksum"] = payload_checksum({
        k: v for k, v in NEW_PROPERTY_CONFIRMATION_REPORT.items()
        if k != "report_checksum"
    })
    _confirm_store.save(
        "metric", "new_property_confirmation", NEW_PROPERTY_CONFIRMATION_REPORT
    )
    _confirm_path = (
        NEW_PROPERTY_CONFIRM_RUN_DIR / "new_property_confirmation_report.json"
    )
    _confirm_path.write_text(
        json.dumps(NEW_PROPERTY_CONFIRMATION_REPORT, indent=2, default=str)
    )
    print("=" * 96)
    print("NEW-PROPERTY FRESH CONFIRMATION —",
          NEW_PROPERTY_CONFIRMATION_REPORT["verdict"])
    print("=" * 96)
    for _cell in NEW_PROPERTY_CONFIRMATION_REPORT["cells"]:
        print(
            _cell["modality"],
            f"exact {_cell['exact_successes']}/{_cell['n']}",
            {name: value["successes"] for name, value in _cell["controls"].items()},
        )
    print("gate        ", NEW_PROPERTY_CONFIRMATION_REPORT["gate"])
    print("failure mode", NEW_PROPERTY_CONFIRMATION_REPORT["failure_mode"])
    print("report", _confirm_path)
    print("checksum", NEW_PROPERTY_CONFIRMATION_REPORT["report_checksum"])
elif RUN_STAGE5B3_NEW_PROPERTY_CONFIRMATION:
    print("Stage 5B3 requested but blocked by the model or confirmation budget.")
'''
)

markdown(
    r"""
## 17. Stage 5C — prospective asymmetry replication

Cat to bird **was** tested in development: 0 successes in 24 trials at alpha=1,
against 24/24 for bird to cat. That is a recorded development observation on a
spent population, not an established property of the representation. It could
reflect model capability, prompt behaviour, coordinate quality, or concept
geometry as easily as a genuine asymmetry.

This stage runs the identical leg-count protocol backwards on fresh cat media.
A null replicates the observed difference and explains nothing about its cause.
A clear effect would show the development failure did not replicate, and the
asymmetry should then not be reported at all.
"""
)
code(
    r'''
ASYMMETRY_REPORT = None
if REAL_MODE and ASYMMETRY_ENABLED:
    from jlens.lens import JacobianLens
    from jlens.mmpilot.coordinate_swap import (
        random_two_direction_basis, resolve_concept_token,
    )
    from jlens.mmpilot.digit_reasoning_confirmation import resolve_digit_endpoints
    from jlens.mmpilot.multimodal_followup import (
        artifact_exclusion_audit, asymmetry_replication_design,
        asymmetry_replication_verdict, exclusion_universe,
    )
    from jlens.mmpilot.multimodal_lens import (
        build_swap_bases_for_lens, load_broad_pooled_development_source,
        open_answer_matches, select_causal_groups, unrestricted_swap_trial,
    )
    from jlens.mmpilot.store import RunFingerprint, UnitStore, safe_key

    _source_pin = load_broad_pooled_development_source(
        BROAD_DEVELOPMENT_RUN_DIR,
        expected_report_checksum=EXPECTED_BROAD_DEVELOPMENT_REPORT_CHECKSUM,
        expected_population_digest=EXPECTED_BROAD_DEVELOPMENT_POPULATION_DIGEST,
        expected_lens_checksum=EXPECTED_BROAD_POOLED_LENS_CHECKSUM,
        expected_direction=CONFIRMATION_DIRECTION,
    )
    ASYMMETRY_DESIGN = asymmetry_replication_design(
        lens_checksum=EXPECTED_BROAD_POOLED_LENS_CHECKSUM,
        exclusions=EXCLUSION_UNIVERSE, layers=BROAD_POOLED_BAND, alpha=1.0,
        n_candidates=ASYMMETRY_CANDIDATES, n_recruited=ASYMMETRY_IMAGES,
        min_success_rate=ASYMMETRY_MIN_SUCCESS_RATE,
        min_control_margin=ASYMMETRY_MIN_CONTROL_MARGIN,
        min_clean_capability_rate=CONFIRMATION_MIN_SUCCESS_RATE,
        familywise_alpha=ASYMMETRY_FAMILYWISE_ALPHA, seed=ASYMMETRY_SEED,
    )
    print("=" * 96)
    print("ASYMMETRY REPLICATION DESIGN FROZEN BEFORE ANY FRESH CAT IS OPENED")
    print("=" * 96)
    print("development record:",
          ASYMMETRY_DESIGN["development_record"]["accurate_statement"])
    print("direction", ASYMMETRY_DESIGN["direction"], "alpha",
          ASYMMETRY_DESIGN["alpha"], "layers", ASYMMETRY_DESIGN["layers"][0],
          "-", ASYMMETRY_DESIGN["layers"][-1])
    print("excluded identities", ASYMMETRY_DESIGN["n_excluded_identities"])
    print("design digest", ASYMMETRY_DESIGN["design_digest"])

    ASYMMETRY_RUN_DIR = (
        RUNS_ROOT / "mmasymmetry" /
        f"mmasymmetry_real_{ASYMMETRY_DESIGN['design_digest'].split(':')[1][:12]}"
    )
    ASYMMETRY_RUN_DIR.mkdir(parents=True, exist_ok=True)
    (ASYMMETRY_RUN_DIR / "frozen_asymmetry_design.json").write_text(
        json.dumps(ASYMMETRY_DESIGN, indent=2, default=str)
    )

    _src, _tgt = ASYMMETRY_DESIGN["direction"]
    _asym_population = select_causal_groups(
        GROUPS, concepts=(_src,), n_per_concept=int(ASYMMETRY_DESIGN["n_candidates"]),
        excluded_image_ids=EXCLUSION_UNIVERSE["excluded_image_ids"],
        seed=ASYMMETRY_DESIGN["seed"], forbidden_concepts={_src: (_tgt,)},
    )[_src]
    ASYMMETRY_EXCLUSION_AUDIT = artifact_exclusion_audit(
        _asym_population, universe=EXCLUSION_UNIVERSE, label="asymmetry_replication"
    )
    (ASYMMETRY_RUN_DIR / "fresh_population.json").write_text(
        json.dumps({
            "population": [
                {"group_id": row["group_id"], "image_id": row["image_id"]}
                for row in _asym_population
            ],
            "exclusion_audit": ASYMMETRY_EXCLUSION_AUDIT,
            "selected_before_capability": True,
        }, indent=2)
    )
    print("fresh cat population disjoint =", ASYMMETRY_EXCLUSION_AUDIT["disjoint"])

    _asym_fingerprint = RunFingerprint(
        mode="real", model_repo_id=MODEL_REPO_ID, model_revision=MODEL_REVISION,
        processor_revision=MODEL_REVISION, layers=tuple(ASYMMETRY_DESIGN["layers"]),
        lens_checksum=EXPECTED_BROAD_POOLED_LENS_CHECKSUM,
        manifest_checksum=MANIFEST_CHECKSUM,
        split_id=ASYMMETRY_EXCLUSION_AUDIT["audit_digest"],
        intervention_config={
            "direction": list(ASYMMETRY_DESIGN["direction"]),
            "alpha": ASYMMETRY_DESIGN["alpha"],
            "conditions": list(ASYMMETRY_DESIGN["conditions"]),
            "positions": "all_original_prompt_positions",
        },
        extra={"design_digest": ASYMMETRY_DESIGN["design_digest"]},
    )
    _asym_store = UnitStore(ASYMMETRY_RUN_DIR, _asym_fingerprint)
    print("asymmetry run state", _asym_store.open())

    _legs = {"bird": "2", "cat": "4"}
    _source_answer, _target_answer = _legs[_src], _legs[_tgt]

    def _asym_prompt(modality, caption):
        question = (
            "How many legs does the animal in the evidence typically have? "
            "Answer with one digit.\nAnswer:"
        )
        return f"Caption: {caption}\n{question}" if modality == "text" else question

    _asym_capability = []
    for _group in _asym_population:
        for _modality in ("text", "image", "spoken_audio"):
            _key = safe_key("asymcap", _group["group_id"], _modality)
            _row = _asym_store.load("capability", _key)
            if _row is None:
                _inputs = build_group_inputs(
                    _group, _modality, _asym_prompt(_modality, _group["caption"])
                )
                _logits = BACKEND.forward_logits(_inputs.tensors)[
                    0, _inputs.final_prompt_position
                ].float()
                _surface = BACKEND.decode_token(int(_logits.argmax())).strip()
                _row = {
                    "group_id": _group["group_id"], "image_id": _group["image_id"],
                    "modality": _modality, "expected": _source_answer,
                    "generated": _surface,
                    "pass": open_answer_matches(_surface, _source_answer),
                }
                _asym_store.save("capability", _key, _row)
            _asym_capability.append(_row)
            if len(_asym_capability) % 48 == 0:
                print("asymmetry capability", len(_asym_capability))

    _asym_recruited = []
    for _group in _asym_population:
        _rows = [
            row for row in _asym_capability
            if row["group_id"] == _group["group_id"]
        ]
        if len(_rows) == 3 and all(row["pass"] for row in _rows):
            _asym_recruited.append(_group)
        if len(_asym_recruited) == int(ASYMMETRY_DESIGN["n_recruited"]):
            break
    _asym_capability_go = (
        len(_asym_recruited) == int(ASYMMETRY_DESIGN["n_recruited"])
    )
    print("asymmetry recruited", len(_asym_recruited), "/",
          ASYMMETRY_DESIGN["n_recruited"])

    _asym_rows = []
    if _asym_capability_go:
        _lens = JacobianLens.load(_source_pin["lens_path"])
        _tokens = {
            name: resolve_concept_token(BACKEND.encode_candidate, name)
            for name in (_src, _tgt, *BROAD_POOLED_CONTROLS)
        }
        _digits = resolve_digit_endpoints(BACKEND)
        _exact_bases = build_swap_bases_for_lens(
            _lens, BACKEND.unembedding_weight(),
            layers=tuple(ASYMMETRY_DESIGN["layers"]),
            source=_tokens[_src], target=_tokens[_tgt],
        )
        _random_bases = {
            layer: random_two_direction_basis(basis, seed=20260823 + layer)
            for layer, basis in _exact_bases.items()
        }
        _unrelated_bases = build_swap_bases_for_lens(
            _lens, BACKEND.unembedding_weight(),
            layers=tuple(ASYMMETRY_DESIGN["layers"]),
            source=_tokens[BROAD_POOLED_CONTROLS[0]],
            target=_tokens[BROAD_POOLED_CONTROLS[1]],
        )
        _conditions = (
            ("exact", float(ASYMMETRY_DESIGN["alpha"]), _exact_bases),
            ("zero", 0.0, _exact_bases),
            ("random", float(ASYMMETRY_DESIGN["alpha"]), _random_bases),
            ("unrelated", float(ASYMMETRY_DESIGN["alpha"]), _unrelated_bases),
        )
        for _group in _asym_recruited:
            for _modality in ("text", "image", "spoken_audio"):
                _inputs = None
                _clean_logits = None
                for _condition, _alpha, _bases in _conditions:
                    _key = safe_key(
                        "asymtrial", _group["group_id"], _modality, _condition
                    )
                    _stored = _asym_store.load("intervention", _key)
                    if _stored is None:
                        if _inputs is None:
                            _inputs = build_group_inputs(
                                _group, _modality,
                                _asym_prompt(_modality, _group["caption"]),
                            )
                            _clean_logits = BACKEND.forward_logits(_inputs.tensors)[
                                0, _inputs.final_prompt_position
                            ].float()
                        _trial = unrestricted_swap_trial(
                            BACKEND, _inputs, bases=_bases, alpha=_alpha,
                            target_token_id=int(_digits["token_ids"][_target_answer]),
                            source_token_id=int(_digits["token_ids"][_source_answer]),
                            clean_logits=_clean_logits, compact_positions=True,
                        )
                        _surface = BACKEND.decode_token(
                            _trial["patched_top_token_id"]
                        ).strip()
                        _stored = {
                            **_trial, "direction": f"{_src}->{_tgt}",
                            "group_id": _group["group_id"],
                            "image_id": _group["image_id"],
                            "modality": _modality, "condition": _condition,
                            "expected": _target_answer,
                            "patched_surface": _surface,
                            "success": open_answer_matches(
                                _surface, _target_answer
                            ),
                        }
                        _asym_store.save("intervention", _key, _stored)
                    _asym_rows.append(_stored)
                    if len(_asym_rows) % 48 == 0:
                        print("asymmetry trials", len(_asym_rows))

    ASYMMETRY_REPORT = asymmetry_replication_verdict(
        _asym_rows, design=ASYMMETRY_DESIGN, capability_go=_asym_capability_go,
        exclusion_audit=ASYMMETRY_EXCLUSION_AUDIT,
    )
    ASYMMETRY_REPORT = {
        **ASYMMETRY_REPORT,
        "frozen_design": ASYMMETRY_DESIGN,
        "capability_rows": _asym_capability,
        "n_fresh_candidates": len(_asym_population),
        "n_recruited": len(_asym_recruited),
    }
    ASYMMETRY_REPORT["report_checksum"] = payload_checksum({
        k: v for k, v in ASYMMETRY_REPORT.items() if k != "report_checksum"
    })
    _asym_store.save("metric", "asymmetry_replication", ASYMMETRY_REPORT)
    _asym_path = ASYMMETRY_RUN_DIR / "asymmetry_replication_report.json"
    _asym_path.write_text(json.dumps(ASYMMETRY_REPORT, indent=2, default=str))
    print("=" * 96)
    print("ASYMMETRY REPLICATION —", ASYMMETRY_REPORT["verdict"])
    print("=" * 96)
    print("reverse successes", ASYMMETRY_REPORT["reverse_successes"], "/",
          ASYMMETRY_REPORT["reverse_trials"])
    print("cause identified ", ASYMMETRY_REPORT["cause_of_asymmetry_identified"])
    print("boundary         ", ASYMMETRY_REPORT["claim_boundary"])
    print("report", _asym_path)
    print("checksum", ASYMMETRY_REPORT["report_checksum"])
elif RUN_STAGE5C_ASYMMETRY_REPLICATION:
    print("Stage 5C requested but blocked by the model or asymmetry budget.")
'''
)

markdown(
    r"""
## 18. MOCK follow-up smoke run

Runs in MOCK mode only. It exercises the follow-up grid, store, resume gate,
freeze gate and every verdict branch without a model, a lens or a photograph.

**A green MOCK run says nothing whatsoever about Gemma 4.** No number it prints
may appear in a scientific report.
"""
)
code(
    r'''
if not REAL_MODE:
    from jlens.mmpilot.multimodal_followup import development_direction_record
    from jlens.mmpilot.multimodal_followup_mock import (
        SCENARIOS, run_mock_asymmetry_study, run_mock_localization,
        run_mock_new_property_study,
    )

    MOCK_ROOT = RUNS_ROOT / "followup_mock"
    MOCK_LOCALIZATION = run_mock_localization(MOCK_ROOT / "localization", n_photographs=2)
    print("MOCK localization verdict     ", MOCK_LOCALIZATION["verdict"])
    print("  label / confirmation        ", MOCK_LOCALIZATION["label"],
          MOCK_LOCALIZATION["is_confirmation"])
    print("  onset layer claimed         ",
          MOCK_LOCALIZATION["claim_boundary"]["onset_layer_claimed"])

    MOCK_FOLLOWUP = {}
    for _scenario in SCENARIOS:
        _result = run_mock_new_property_study(
            MOCK_ROOT / _scenario, scenario=_scenario
        )
        MOCK_FOLLOWUP[_scenario] = {
            "development": _result["development"]["verdict"],
            "confirmation": (_result["confirmation"] or {}).get("verdict"),
        }
        print(f"MOCK {_scenario:<18}", MOCK_FOLLOWUP[_scenario])
    assert len({row["development"] for row in MOCK_FOLLOWUP.values()}) == len(SCENARIOS)

    MOCK_ASYMMETRY = run_mock_asymmetry_study(MOCK_ROOT / "asymmetry")["report"]
    print("MOCK asymmetry verdict        ", MOCK_ASYMMETRY["verdict"])
    print("MOCK development record       ",
          development_direction_record()["accurate_statement"])
    print("MOCK RESULTS ARE NOT SCIENTIFIC RESULTS.")
'''
)

metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3"},
    "colab": {"name": TARGET.name, "provenance": []},
}
notebook = {
    "cells": [
        {
            "cell_type": kind,
            "metadata": {},
            "source": [line + "\n" for line in value.splitlines()],
            **({"execution_count": None, "outputs": []} if kind == "code" else {}),
        }
        for kind, value in CELLS
    ],
    "metadata": metadata,
    "nbformat": 4,
    "nbformat_minor": 5,
}
TARGET.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
print(TARGET)
