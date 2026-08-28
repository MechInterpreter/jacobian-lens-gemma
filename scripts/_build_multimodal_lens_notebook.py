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
| 5B1A | CPU | read-only amendment of the flawed recruited run + audit of the confirmed leg-count code path | no model, zero forwards |
| 5B1RC | L4 | corrected recruited exploratory rerun with the realization policy and the direct-answer positive control | one generated condition |

Changing any model, processor, audio protocol, cache, population, order, layer,
prompt, lens, causal setting, **or model-dtype realization policy** changes the
fingerprint and refuses reuse.

## The two configurations for the instrument repair

Run exactly one of these per session. Everything they need is printed before
the model is loaded.

**A. CPU-only historical amendment and preflight** — no GPU, no model, no lens,
zero forwards, `scientific_recompute = 0`:

```python
RUN_REAL_MATCHED_JLENS = True
RUN_STAGE5B1A_INSTRUMENT_AMENDMENT = True
```

**B. L4 corrected causal rerun** — the corrected Stage 5B1RC:

```python
RUN_REAL_MATCHED_JLENS = True
RUN_STAGE5B01_AUDIO_LINKAGE_AUDIT = True
RUN_STAGE5B1RC_CORRECTED_EXPLORATORY = True
CONFIRM_MODEL_LOAD = True
CONFIRM_CORRECTED_EXPLORATORY_BUDGET = True
```

Configuration B is 240 generated conditions at 6 tokens each — at most 1440
token forwards, zero lens fits, zero backward passes, one checksum-valid JSON
per completed condition, and roughly 3 MB of Drive. Section 2 prints all of it,
plus the realization tolerance, the positive control's role and the primary
outcome rule, before section 4 loads anything.
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

# A Colab runtime may have imported jlens before this cell was rerun. Remove
# those stale module objects after checkout so later cells import the code from
# BRANCH rather than retaining an older branch in sys.modules.
for _module_name in tuple(sys.modules):
    if _module_name == "jlens" or _module_name.startswith("jlens."):
        sys.modules.pop(_module_name, None)

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
# Stage 3DA is the read-only realization replication of that completed
# confirmation. It writes to its own run directory, never into the confirmed
# run, and it cannot relabel the confirmed verdict.
RUN_STAGE3DA_CONFIRMATION_REALIZATION_REPLICATION = False
RUN_STAGE4_WRITE_REPORT = False

CONFIRM_MODEL_LOAD = False
CONFIRM_FIT_BUDGET = False
CONFIRM_CAUSAL_BUDGET = False
CONFIRM_ALPHA_SWEEP_BUDGET = False
CONFIRM_BROAD_POOLED_FIT_BUDGET = False
CONFIRM_BROAD_POOLED_CAUSAL_BUDGET = False
CONFIRM_FRESH_MULTIMODAL_CONFIRMATION_BUDGET = False
CONFIRM_CONFIRMATION_REPLICATION_BUDGET = False
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
RUN_STAGE5B00_PROPERTY_PROMPT_SCREEN = False
RUN_STAGE5B0_PROPERTY_AUDIT = False
RUN_STAGE5B1_NEW_PROPERTY_DEVELOPMENT = False
RUN_STAGE5B01_AUDIO_LINKAGE_AUDIT = False
# Stage 5B1A is the CPU-only read-only amendment of the flawed recruited run,
# plus the code-path audit of the confirmed leg-count result. No model, no lens,
# zero forwards. Stage 5B1RC is the corrected causal rerun; it needs an L4.
RUN_STAGE5B1A_INSTRUMENT_AMENDMENT = False
RUN_STAGE5B1RC_CORRECTED_EXPLORATORY = False
RUN_STAGE5B2_FREEZE_NEW_PROPERTY_DESIGN = False
RUN_STAGE5B3_NEW_PROPERTY_CONFIRMATION = False
RUN_STAGE5C_ASYMMETRY_REPLICATION = False
RUN_ARTIFACT_EXCLUSION_AUDIT = False

# Stage 6A-6E: the frozen, evidence-quality-gated, fp32 cat->dog animal-sound
# study. See section 19 for the full design; declared here (rather than in
# that section's own cell) because FOLLOWUP_STAGES and MODEL_STAGE below need
# every switch name to exist before any cell runs, and a later cell that
# redeclared these would silently reset whatever this cell's user just set.
RUN_STAGE6A_EVIDENCE_QUALITY_INDEX = False
RUN_STAGE6B_POPULATION_FREEZE = False
RUN_STAGE6C_CATDOG_DEVELOPMENT = False
# Stage 6C2 is instrument development on the eight photographs already spent
# by the completed, inconclusive Stage 6C run.  It scores the direct-answer
# positive control only; no exact cat->dog output can select a path.
RUN_STAGE6C2_CATDOG_PATH_LOCALIZATION = False
RUN_STAGE6C1_CATDOG_INSTRUMENT_AMENDMENT = False
RUN_STAGE6D_CATDOG_FREEZE = False
RUN_STAGE6E_CATDOG_CONFIRMATION = False

# Stage 7 is the final no-refit target-generalization study.  It keeps the
# confirmed bird-source method fixed and asks whether distinct injected target
# identities produce distinct leg-count answers: cat=4, ant=6, spider=8.
RUN_STAGE7A_FREEZE_LEG_GENERALIZATION_POPULATION = False
RUN_STAGE7B_LEG_GENERALIZATION_DEVELOPMENT = False
RUN_STAGE7B2_NOVEL_LEG_TARGET_DEVELOPMENT = True
RUN_STAGE7C_FREEZE_LEG_GENERALIZATION_CONFIRMATION = True
RUN_STAGE7D_LEG_GENERALIZATION_CONFIRMATION = True
CONFIRM_LEG_GENERALIZATION_DEVELOPMENT_BUDGET = False
CONFIRM_LEG_GENERALIZATION_NOVEL_BUDGET = True
CONFIRM_LEG_GENERALIZATION_CONFIRMATION_BUDGET = True
CONFIRM_LEG_GENERALIZATION_FP32_A100 = True

# The frozen cat->dog scientific target and every tunable it uses. Declared
# here, not in section 19's own cell, for the same reason the switches above
# are: a later cell that redeclared these would silently discard whatever
# this cell's user just set, exactly the bug that made the switches need to
# move up here too.
CATDOG_PROPERTY_FAMILY = "animal_sound"
CATDOG_PROMPT_ID = "identity_explicit_v1"
CATDOG_DIRECTION = ("cat", "dog")          # the single primary direction
CATDOG_CLEAN_ANSWER = "meow"               # cat's untouched answer
CATDOG_SWAPPED_ANSWER = "bark"             # dog's answer; the exchange target
CATDOG_MAX_NEW_TOKENS = 6
CATDOG_N_DEV_CANDIDATES_PER_CONCEPT = 24
CATDOG_N_CONFIRM_CANDIDATES_PER_CONCEPT = 96
CATDOG_DEV_IMAGES_PER_DIRECTION = 8
CATDOG_CONFIRM_IMAGES = 16
CATDOG_MIN_CLEAN_CAPABILITY_RATE = 0.75
CATDOG_DEV_MIN_SUCCESS_RATE = 0.50
CATDOG_DEV_MIN_CONTROL_MARGIN = 0.25
CATDOG_CUMULATIVE_DISPLACEMENT_MATCH_TOLERANCE = 0.02
CATDOG_CONFIRM_MIN_SUCCESS_RATE = 0.75
CATDOG_CONFIRM_MIN_CONTROL_MARGIN = 0.25
CATDOG_CONFIRM_FAMILYWISE_ALPHA = 0.05
CATDOG_SEED = "catdog-frozen-animal-sound-fp32-20260825-v1"
CATDOG_FP32_WORKSPACE_FRACTION = 0.30
CATDOG_FP32_SAFETY_MARGIN = 1.15
# Filled in across sessions, exactly like every other stage's *_RUN_DIR /
# EXPECTED_*_CHECKSUM pair: each Stage 6 sub-stage pins the previous one's
# checksum-verified artifact rather than sharing in-memory state.
CATDOG_EVIDENCE_INDEX_RUN_DIR = None
EXPECTED_CATDOG_EVIDENCE_INDEX_CHECKSUM = None
CATDOG_POPULATION_FREEZE_RUN_DIR = None
EXPECTED_CATDOG_POPULATION_FREEZE_DIGEST = None
CATDOG_DEVELOPMENT_RUN_DIR = None
EXPECTED_CATDOG_DEVELOPMENT_CHECKSUM = None
CATDOG_FROZEN_DESIGN_PATH = None
# The completed fp32 Stage 6C result that licenses Stage 6C2.  It is opened
# read-only and checksum-pinned; the localization writes to a different run.
CATDOG_INCONCLUSIVE_DEVELOPMENT_RUN_DIR = (
    "/content/drive/MyDrive/jacobian-lens-gemma/runs/mmcatdogdev/"
    "mmcatdogdev_real_ffb335e737aa"
)
EXPECTED_CATDOG_INCONCLUSIVE_DEVELOPMENT_CHECKSUM = (
    "sha256:1d6bd80da1fc6eadf47984d0bf0cc930e756963b71ba6de81ea5b77fc015f4b8"
)
CATDOG_PATH_LOCALIZATION_MIN_SUCCESS_RATE = 0.50
# The completed Stage 6C run with the per-layer-only positive-control match.
# Stage 6C1 opens this report read-only, verifies the checksum, and writes one
# amendment beside it. It never rewrites the report or any unit.
CATDOG_UNMATCHED_DEVELOPMENT_RUN_DIR = (
    "/content/drive/MyDrive/jacobian-lens-gemma/runs/mmcatdogdev/"
    "mmcatdogdev_real_bc03b54e8494"
)
EXPECTED_CATDOG_UNMATCHED_DEVELOPMENT_CHECKSUM = (
    "sha256:ccf34c1303c17960edc13653298c4badba462a7cbda5ca90b5fb637d7af04be2"
)

CONFIRM_LOCALIZATION_BUDGET = False
CONFIRM_PROPERTY_PROMPT_SCREEN_BUDGET = False
CONFIRM_NEW_PROPERTY_DEVELOPMENT_BUDGET = False
CONFIRM_CORRECTED_EXPLORATORY_BUDGET = False
CONFIRM_NEW_PROPERTY_CONFIRMATION_BUDGET = False
CONFIRM_ASYMMETRY_BUDGET = False
CONFIRM_CATDOG_DEVELOPMENT_BUDGET = False
CONFIRM_CATDOG_PATH_LOCALIZATION_BUDGET = False
CONFIRM_CATDOG_CONFIRMATION_BUDGET = False

# The completed Stage 3D confirmation, read only to spend its media. All 64
# candidate photographs were opened during capability screening, so all 64 are
# excluded from every later population — not only the 16 that were recruited.
FRESH_CONFIRMATION_RUN_DIR = (
    "/content/drive/MyDrive/jacobian-lens-gemma/runs/"
    "mmbroadconfirm/mmbroadconfirm_real_40e087ee1061"
)
EXPECTED_FRESH_CONFIRMATION_REPORT_CHECKSUM = (
    "sha256:2bb6dcc1346229573566125bc8d91c782247d55af5091f4215d98bb621472ff7"
)
FRESH_CONFIRMATION_CANDIDATES_OPENED = 64
FRESH_CONFIRMATION_IMAGES_RECRUITED = 16

# Extra prior-run reports to exclude from, beyond the ones this notebook knows
# how to name and checksum-pin above. Fill this in with report JSON paths
# before rerunning section 12 whenever a run outside the checksum-pinned set
# opened media that must not be reused: an abandoned property family (e.g. a
# failed body_covering run before trying the animal_sound fallback), or
# Experiment B's opened photographs before running Experiment C in the same
# session. Unlike the pinned loaders above, these paths are NOT
# checksum-verified — they can only widen exclusion, never substitute for a
# checksum-pinned population loader. See EXTRA_SPENT_REPORT_PATHS below.
EXTRA_SPENT_REPORT_PATHS = (
    # The withdrawn body_covering attempt. It opened 144 photographs during
    # clean capability screening, so all 144 are spent and must not reappear
    # in the animal_sound population.
    "/content/drive/MyDrive/jacobian-lens-gemma/runs/mmnewproperty/"
    "mmnewpropertydev_real_baad443fdc39/new_property_audit_report.json",
    # The fixed-code animal_sound audit opened 48 photographs for each of
    # bird/cat/cow/dog. Stage 5B00 may reuse cat/cow for prompt development,
    # but every later capability or causal population must exclude all 192.
    "/content/drive/MyDrive/jacobian-lens-gemma/runs/mmnewproperty/"
    "mmnewpropertydev_real_bd1b87fe613b/new_property_audit_report.json",
    # The identity-explicit cat/cow audit opened 48 new photographs per
    # concept. Its aggregate verdict remains NO_GO, but all opened media are
    # spent regardless of the recruited exploratory analysis below.
    "/content/drive/MyDrive/jacobian-lens-gemma/runs/mmnewproperty/"
    "mmnewpropertydev_real_6ee23e7e61ce/new_property_audit_report.json",
)

# Experiment A. The band grid and its analysis rule live in
# jlens.mmpilot.multimodal_followup.localization_grid() and are frozen there
# before any sub-band outcome exists. The population is the spent broad
# development one, so every output of this stage is exploratory.
LOCALIZATION_DIRECTION = ("bird", "cat")
LOCALIZATION_ALPHA = 1.0

# Experiment B, second design. The first attempt used 'body_covering' and
# returned PROPERTY_AUDIT_NO_GO on clean capability alone, before any causal
# forward was spent. Reading its failures showed the real defect was not the
# prompt: a covering is *visible in the photograph*, so the model can answer by
# describing pixels ("black", "spotted", "stripes", "patches") instead of
# consulting the animal's identity. That gives it a route around the very
# variable the intervention edits, and would have made an image-route null
# uninterpretable. 'body_covering' is now disqualified in code.
#
# 'animal_sound' replaces it for the opposite reason: a still photograph
# carries no sound, so every route must go through identity to answer — the
# same structure that makes leg count work.
#
# 'bird' is screened rather than assumed. Selection admits a group only when
# its caption contains the literal word "bird" (measured: 28 of 39 bird-ish
# captions naming a species are filtered out), so text and spoken audio see a
# generic bird while only the image route shows a species. Whether one sound
# answer is stable in all three routes is decided by DOMINANT_ANSWER_RULE,
# fixed before the data is opened, and bird is refused if it is not.
NEW_PROPERTY_FAMILY = "animal_sound"
NEW_PROPERTY_FALLBACK_FAMILY = None
# Stage 5B00 may select a different declared prompt on already-spent media.
# Keep baseline_v1 until that screen reports GO. A non-baseline value is
# refused below unless its screen report and checksum are pinned.
NEW_PROPERTY_PROMPT_ID = "baseline_v1"
NEW_PROPERTY_PROMPT_SCREEN_RUN_DIR = None
EXPECTED_NEW_PROPERTY_PROMPT_SCREEN_CHECKSUM = None
PROPERTY_PROMPT_SCREEN_SOURCE_RUN_DIR = (
    "/content/drive/MyDrive/jacobian-lens-gemma/runs/mmnewproperty/"
    "mmnewpropertydev_real_bd1b87fe613b"
)
EXPECTED_PROPERTY_PROMPT_SCREEN_SOURCE_FILE_SHA256 = (
    "sha256:46f32a7d99a226d4eb75638d2bea6c9f3273178dc1a7d82f89993525bfd4e2ac"
)
EXPECTED_PROPERTY_PROMPT_SCREEN_SOURCE_AUDIT_DIGEST = (
    "sha256:f0341821643aec918525617d50659ad04b35aecd9ace3bc9dcc08dde9a0c565c"
)
# The fresh identity-explicit capability audit. It remains a scientific
# PROPERTY_AUDIT_NO_GO. Stages 5B01/5B1R read it by exact file-byte checksum;
# they never rewrite or relabel it.
RECRUITED_EXPLORATORY_SOURCE_RUN_DIR = (
    "/content/drive/MyDrive/jacobian-lens-gemma/runs/mmnewproperty/"
    "mmnewpropertydev_real_6ee23e7e61ce"
)
EXPECTED_RECRUITED_SOURCE_FILE_SHA256 = (
    "sha256:490636eb864251541407407ea1109826212e35390281228b2cf8eb0936a986a4"
)
EXPECTED_RECRUITED_SOURCE_AUDIT_DIGEST = (
    "sha256:52833e841ceb76fb96fa7425913c79fdd22c84eef4f66400a214240101e6ed1b"
)
RECRUITED_EXPLORATORY_CONCEPTS = ("cat", "cow")
RECRUITED_EXPLORATORY_DIRECTIONS = (("cat", "cow"), ("cow", "cat"))
RECRUITED_EXPLORATORY_IMAGES_PER_DIRECTION = 8
RECRUITED_EXPLORATORY_SEED = "post-audit-clean-capable-rescue-20260824-v1"

# The completed recruited exploratory run. Its 0/8 outputs are retained as
# historical evidence and its files are never rewritten, but its instrument was
# broken: the model-dtype realization policy was not passed into the generated
# conditions and the verdict's integrity check omitted the clauses below, so a
# post-cast relative coordinate error an order of magnitude outside the frozen
# 0.02 tolerance was reported as integrity_pass=true. Stage 5B1A amends it
# read-only; Stage 5B1RC reruns it with the identical scientific design.
RECRUITED_EXPLORATORY_FLAWED_RUN_DIR = (
    "/content/drive/MyDrive/jacobian-lens-gemma/runs/mmnewpropertyrescue/"
    "mmnewpropertyrescue_real_6af6affcb145"
)
EXPECTED_RECRUITED_EXPLORATORY_REPORT_CHECKSUM = (
    "sha256:467a2862cef70f0b59a75678c6a73c68259f4b29f715a97fbb831914710f660a"
)
# Worst observed post-cast relative coordinate error per condition, read off
# that run's own stored trial diagnostics. The zero arm is exact because its
# update is multiplied by exactly zero.
RECRUITED_EXPLORATORY_OBSERVED_POST_CAST_ERRORS = {
    "exact": 0.21,
    "random": 0.10,
    "unrelated": 0.29,
    "zero": 0.0,
}
# The clauses that existed in the diagnostics but were never enforced.
RECRUITED_EXPLORATORY_OMITTED_CLAUSES = (
    "all_hooks_fired",
    "all_finite",
    "alpha_one_exact_exchange_before_cast",
    "model_dtype_realization_converged",
    "post_cast_coordinate_error_within_tolerance",
    "post_cast_residual_drift_within_tolerance",
)
# What the completed leg-count confirmation's verdict actually enforced. Both
# gates are real and tight (1e-5), but both read the float64 *pre-cast* solve
# error, which is exact by construction; neither is a post-cast clause.
LEGACY_CONFIRMATION_ENFORCED_CLAUSES = (
    "expected_layers_patched",
    "all_prompt_positions_patched",
)
# What that run's stored rows actually carry, and whether it passed a
# realization policy. Both are historical facts about the completed artifact,
# pinned here because a later repair to unrestricted_swap_trial changes what
# today's code would do and must not be mistaken for what that run did. Stage
# 5B1A prefers the artifact's own rows whenever Drive is mounted.
LEGACY_CONFIRMATION_STORED_ROW_FIELDS = (
    "max_activation_norm_ratio",
    "max_coordinate_update_error",
    "max_orthogonal_residual_drift",
    "max_update_to_activation_norm_ratio",
    "min_activation_norm_ratio",
)
LEGACY_CONFIRMATION_REALIZATION_POLICY_PASSED = False
# The corrected rerun adds one arm to the same four conditions: the
# norm-matched direct-answer positive control. It is a diagnostic of causal
# leverage on this path and can never produce a GO.
CORRECTED_EXPLORATORY_CONDITIONS = (
    "exact", "zero", "random", "unrelated", "direct_answer",
)
PROPERTY_PROMPT_SCREEN_CONCEPTS = ("cat", "cow")
NEW_PROPERTY_CONCEPTS = ("bird", "cat", "cow", "dog")
NEW_PROPERTY_DEV_DIRECTIONS = (
    ("cat", "cow"), ("cow", "cat"),
    ("cat", "dog"), ("dog", "cat"),
    ("bird", "cat"), ("cat", "bird"),
)
# Predeclared tie-break, fixed before any outcome: if development licenses
# several directions, confirmation takes the first of these that passed.
# bird->cat leads because it reuses the confirmed pair; the cow/dog pairs
# follow because both of their concepts are species-homogeneous.
NEW_PROPERTY_DIRECTION_PRIORITY = (
    "bird->cat", "cat->cow", "cat->dog",
    "cow->cat", "dog->cat", "cat->bird",
)
NEW_PROPERTY_MAX_NEW_TOKENS = 6
NEW_PROPERTY_DEV_CANDIDATES_PER_CONCEPT = 48
NEW_PROPERTY_DEV_IMAGES_PER_DIRECTION = 8
NEW_PROPERTY_DEV_MIN_SUCCESS_RATE = 0.50
NEW_PROPERTY_DEV_MIN_CONTROL_MARGIN = 0.25
NEW_PROPERTY_MIN_CLEAN_CAPABILITY_RATE = 0.75
NEW_PROPERTY_DEV_SEED = "multimodal-new-property-sound-development-20260824-v1"
NEW_PROPERTY_CONFIRM_CANDIDATES = 64
NEW_PROPERTY_CONFIRM_IMAGES = 16
NEW_PROPERTY_CONFIRM_MIN_SUCCESS_RATE = 0.75
NEW_PROPERTY_CONFIRM_MIN_CONTROL_MARGIN = 0.25
NEW_PROPERTY_CONFIRM_FAMILYWISE_ALPHA = 0.05
NEW_PROPERTY_CONFIRM_SEED = "multimodal-new-property-sound-confirmation-20260824-v1"
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
    "5B00_property_prompt_screen": RUN_STAGE5B00_PROPERTY_PROMPT_SCREEN,
    "5B0_property_audit": RUN_STAGE5B0_PROPERTY_AUDIT,
    "5B1_new_property_development": RUN_STAGE5B1_NEW_PROPERTY_DEVELOPMENT,
    "5B1RC_corrected_exploratory": RUN_STAGE5B1RC_CORRECTED_EXPLORATORY,
    "5B2_freeze": RUN_STAGE5B2_FREEZE_NEW_PROPERTY_DESIGN,
    "5B3_new_property_confirmation": RUN_STAGE5B3_NEW_PROPERTY_CONFIRMATION,
    "5C_asymmetry_replication": RUN_STAGE5C_ASYMMETRY_REPLICATION,
    "6A_catdog_evidence_index": RUN_STAGE6A_EVIDENCE_QUALITY_INDEX,
    "6B_catdog_population_freeze": RUN_STAGE6B_POPULATION_FREEZE,
    "6C_catdog_development": RUN_STAGE6C_CATDOG_DEVELOPMENT,
    "6C2_catdog_path_localization": RUN_STAGE6C2_CATDOG_PATH_LOCALIZATION,
    "6C1_catdog_instrument_amendment": RUN_STAGE6C1_CATDOG_INSTRUMENT_AMENDMENT,
    "6D_catdog_freeze": RUN_STAGE6D_CATDOG_FREEZE,
    "6E_catdog_confirmation": RUN_STAGE6E_CATDOG_CONFIRMATION,
    "7A_leg_generalization_population": (
        RUN_STAGE7A_FREEZE_LEG_GENERALIZATION_POPULATION
    ),
    "7B_leg_generalization_development": (
        RUN_STAGE7B_LEG_GENERALIZATION_DEVELOPMENT
    ),
    "7B2_novel_leg_target_development": (
        RUN_STAGE7B2_NOVEL_LEG_TARGET_DEVELOPMENT
    ),
    "7C_leg_generalization_freeze": (
        RUN_STAGE7C_FREEZE_LEG_GENERALIZATION_CONFIRMATION
    ),
    "7D_leg_generalization_confirmation": (
        RUN_STAGE7D_LEG_GENERALIZATION_CONFIRMATION
    ),
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
    RUN_STAGE3DA_CONFIRMATION_REALIZATION_REPLICATION,
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
    RUN_STAGE3DA_CONFIRMATION_REALIZATION_REPLICATION,
    RUN_STAGE5A_BAND_LOCALIZATION,
    RUN_STAGE5B00_PROPERTY_PROMPT_SCREEN,
    RUN_STAGE5B0_PROPERTY_AUDIT,
    RUN_STAGE5B1_NEW_PROPERTY_DEVELOPMENT,
    RUN_STAGE5B1RC_CORRECTED_EXPLORATORY,
    RUN_STAGE5B3_NEW_PROPERTY_CONFIRMATION,
    RUN_STAGE5C_ASYMMETRY_REPLICATION,
    RUN_STAGE7B_LEG_GENERALIZATION_DEVELOPMENT,
    RUN_STAGE7B2_NOVEL_LEG_TARGET_DEVELOPMENT,
    RUN_STAGE7D_LEG_GENERALIZATION_CONFIRMATION,
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
CONFIRMATION_REPLICATION_ENABLED = bool(
    RUN_STAGE3DA_CONFIRMATION_REALIZATION_REPLICATION and MODEL_ENABLED
    and CONFIRM_CONFIRMATION_REPLICATION_BUDGET
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
if REAL_MODE and RUN_STAGE3DA_CONFIRMATION_REALIZATION_REPLICATION and not CONFIRMATION_REPLICATION_ENABLED:
    print("CONFIRMATION REALIZATION REPLICATION BLOCKED: confirm its printed budget")

LOCALIZATION_ENABLED = bool(
    RUN_STAGE5A_BAND_LOCALIZATION and MODEL_ENABLED and CONFIRM_LOCALIZATION_BUDGET
)
PROPERTY_AUDIT_ENABLED = bool(RUN_STAGE5B0_PROPERTY_AUDIT and MODEL_ENABLED)
PROPERTY_PROMPT_SCREEN_ENABLED = bool(
    RUN_STAGE5B00_PROPERTY_PROMPT_SCREEN and MODEL_ENABLED
    and CONFIRM_PROPERTY_PROMPT_SCREEN_BUDGET
)
NEW_PROPERTY_DEV_ENABLED = bool(
    RUN_STAGE5B1_NEW_PROPERTY_DEVELOPMENT and MODEL_ENABLED
    and CONFIRM_NEW_PROPERTY_DEVELOPMENT_BUDGET
)
AUDIO_LINKAGE_AUDIT_ENABLED = bool(RUN_STAGE5B01_AUDIO_LINKAGE_AUDIT)
INSTRUMENT_AMENDMENT_ENABLED = bool(RUN_STAGE5B1A_INSTRUMENT_AMENDMENT)
CORRECTED_EXPLORATORY_ENABLED = bool(
    RUN_STAGE5B1RC_CORRECTED_EXPLORATORY and MODEL_ENABLED
    and CONFIRM_CORRECTED_EXPLORATORY_BUDGET
)
NEW_PROPERTY_FREEZE_ENABLED = bool(RUN_STAGE5B2_FREEZE_NEW_PROPERTY_DESIGN)
NEW_PROPERTY_CONFIRM_ENABLED = bool(
    RUN_STAGE5B3_NEW_PROPERTY_CONFIRMATION and MODEL_ENABLED
    and CONFIRM_NEW_PROPERTY_CONFIRMATION_BUDGET
)
ASYMMETRY_ENABLED = bool(
    RUN_STAGE5C_ASYMMETRY_REPLICATION and MODEL_ENABLED and CONFIRM_ASYMMETRY_BUDGET
)
LEG_GENERALIZATION_DEVELOPMENT_ENABLED = bool(
    RUN_STAGE7B_LEG_GENERALIZATION_DEVELOPMENT
    and MODEL_ENABLED
    and CONFIRM_LEG_GENERALIZATION_DEVELOPMENT_BUDGET
    and CONFIRM_LEG_GENERALIZATION_FP32_A100
)
LEG_GENERALIZATION_NOVEL_ENABLED = bool(
    RUN_STAGE7B2_NOVEL_LEG_TARGET_DEVELOPMENT
    and MODEL_ENABLED
    and CONFIRM_LEG_GENERALIZATION_NOVEL_BUDGET
    and CONFIRM_LEG_GENERALIZATION_FP32_A100
)
LEG_GENERALIZATION_CONFIRMATION_ENABLED = bool(
    RUN_STAGE7D_LEG_GENERALIZATION_CONFIRMATION
    and MODEL_ENABLED
    and CONFIRM_LEG_GENERALIZATION_CONFIRMATION_BUDGET
    and CONFIRM_LEG_GENERALIZATION_FP32_A100
)
for _name, _requested, _enabled in (
    ("STAGE 5A LOCALIZATION", RUN_STAGE5A_BAND_LOCALIZATION, LOCALIZATION_ENABLED),
    ("STAGE 5B00 PROPERTY PROMPT SCREEN", RUN_STAGE5B00_PROPERTY_PROMPT_SCREEN, PROPERTY_PROMPT_SCREEN_ENABLED),
    ("STAGE 5B1 NEW-PROPERTY DEVELOPMENT", RUN_STAGE5B1_NEW_PROPERTY_DEVELOPMENT, NEW_PROPERTY_DEV_ENABLED),
    ("STAGE 5B1RC CORRECTED EXPLORATORY", RUN_STAGE5B1RC_CORRECTED_EXPLORATORY, CORRECTED_EXPLORATORY_ENABLED),
    ("STAGE 5B3 NEW-PROPERTY CONFIRMATION", RUN_STAGE5B3_NEW_PROPERTY_CONFIRMATION, NEW_PROPERTY_CONFIRM_ENABLED),
    ("STAGE 5C ASYMMETRY REPLICATION", RUN_STAGE5C_ASYMMETRY_REPLICATION, ASYMMETRY_ENABLED),
    ("STAGE 7B LEG GENERALIZATION DEVELOPMENT", RUN_STAGE7B_LEG_GENERALIZATION_DEVELOPMENT, LEG_GENERALIZATION_DEVELOPMENT_ENABLED),
    ("STAGE 7B2 NOVEL LEG TARGET DEVELOPMENT", RUN_STAGE7B2_NOVEL_LEG_TARGET_DEVELOPMENT, LEG_GENERALIZATION_NOVEL_ENABLED),
    ("STAGE 7D LEG GENERALIZATION CONFIRMATION", RUN_STAGE7D_LEG_GENERALIZATION_CONFIRMATION, LEG_GENERALIZATION_CONFIRMATION_ENABLED),
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
print("STAGE 7 NO-REFIT LEG-COUNT GENERALIZATION")
print("  frozen source             bird")
print("  calibration / novel       cat=4 / ant=6, spider=8")
print("  lens / band / alpha       existing pooled lens / L16-L40 / exact alpha=1")
print("  positions                 every original prompt position")
print("  new bird candidates       9 development + 22 confirmation")
print("  recruited                 6 development + 12 confirmation")
print("  development maximum       27 capability + 54 answer-leverage + 216 causal")
print("  Stage 7B2 novel extension 216 causal + 36 diagnostic on the spent photos")
print("  Stage 7B2 scored controls zero, unrelated, 3 random seeds, cross-target")
print("  confirmation maximum      66 capability + 432 causal + 72 diagnostic")
print("  fitting / backward        0 / 0")
print("  runtime                   fp32 80 GB A100; one JSON per completed condition")
print()
print("STAGE 3DA CONFIRMATION REALIZATION REPLICATION BUDGET")
print("  what it measures          whether the completed confirmation's exchange")
print("                            was realized in the model's bf16 dtype, and")
print("                            whether repairing it preserves the outcome")
print("  fitting / backward passes 0 / 0")
print("  arms                      uncorrected (as run) and corrected (frozen policy)")
print("  EXACT GENERATIONS         ",
      CONFIRMATION_IMAGES * 3 * 4 * 2, "(single-token endpoint)")
print("  MAXIMUM TOKEN FORWARDS    ", CONFIRMATION_IMAGES * 3 * 4 * 2)
print("  writes into the confirmed run   no; its own run directory only")
print("  can relabel the confirmed verdict  no")
print("  resume unit               one checksum-valid JSON per arm x condition")
print()

from jlens.mmpilot.multimodal_followup import (
    ANIMAL_SOUND_PROMPT_CANDIDATES, followup_budget, localization_budget,
    localization_grid, stage_map,
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
_prompt_screen_new_candidates = [
    row for row in ANIMAL_SOUND_PROMPT_CANDIDATES
    if row["prompt_id"] != "baseline_v1"
]
_prompt_screen_completions = (
    len(_prompt_screen_new_candidates)
    * len(PROPERTY_PROMPT_SCREEN_CONCEPTS) * 3
    * NEW_PROPERTY_DEV_CANDIDATES_PER_CONCEPT
)
print("STAGE 5B00 ANIMAL-SOUND PROMPT SCREEN BUDGET")
print("  population                already spent Stage-5B0 media")
print("  baseline completions      imported; no repeat model work")
print("  new prompt variants       ",
      [row["prompt_id"] for row in _prompt_screen_new_candidates])
print("  concepts                  ", list(PROPERTY_PROMPT_SCREEN_CONCEPTS))
print("  new unrestricted generations", _prompt_screen_completions)
print("  maximum token forwards    ",
      _prompt_screen_completions * NEW_PROPERTY_MAX_NEW_TOKENS)
print("  causal interventions      0")
print("  lens fits / backward      0 / 0")
print("  resume                    one checksum-valid completion JSON")
print("  boundary                  prompt development only; a winner must pass")
print("                            capability again on fresh development media")
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
print("  property family           ", NEW_PROPERTY_FAMILY,
      f"(fallback {NEW_PROPERTY_FALLBACK_FAMILY})")
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
from jlens.mmpilot.multimodal_instrument import (
    INTEGRITY_CLAUSES as _INTEGRITY_CLAUSES,
    POST_CAST_TOLERANCE as _POST_CAST_TOLERANCE,
    realization_policy_digest as _realization_policy_digest,
)

_CORRECTED_N_CONDITIONS = (
    RECRUITED_EXPLORATORY_IMAGES_PER_DIRECTION
    * len(RECRUITED_EXPLORATORY_DIRECTIONS)
    * 3
    * len(CORRECTED_EXPLORATORY_CONDITIONS)
)
_CORRECTED_MAX_TOKEN_FORWARDS = (
    _CORRECTED_N_CONDITIONS * NEW_PROPERTY_MAX_NEW_TOKENS
)
# One JSON unit per generated condition plus one report; the units are small
# flat records, not activations.
_CORRECTED_DRIVE_MB = round(
    (_CORRECTED_N_CONDITIONS * 12 + 400) / 1024.0, 2
)
print("STAGE 5B01 METADATA AUDIT BUDGET")
print("  metadata audit            CPU, 0 model forwards")
print("  source capability rows    reused; no clean regeneration")
print()
print("STAGE 5B1A INSTRUMENT AMENDMENT BUDGET (CPU-only)")
print("  model loads               0")
print("  model forwards            0")
print("  lens fits / backward      0 / 0")
print("  scientific_recompute      0")
print("  writes                    new files beside the completed runs; no")
print("                            completed report or unit is opened for writing")
print()
print("STAGE 5B1RC CORRECTED EXPLORATORY BUDGET")
print("  recruited photographs     ", RECRUITED_EXPLORATORY_IMAGES_PER_DIRECTION,
      "per source concept (existing spent population; none newly selected)")
print("  conditions per cell       ", list(CORRECTED_EXPLORATORY_CONDITIONS))
print("  EXACT GENERATIONS         ", _CORRECTED_N_CONDITIONS)
print("  MAXIMUM TOKEN FORWARDS    ", _CORRECTED_MAX_TOKEN_FORWARDS,
      f"({NEW_PROPERTY_MAX_NEW_TOKENS} per generation, early stop only reduces it)")
print("  lens fits                  0")
print("  backward passes            0")
print("  resume unit               one checksum-valid JSON per completed")
print("                            generated condition")
print("  realization tolerance     ", _POST_CAST_TOLERANCE,
      "relative, post-cast, coordinate and residual")
print("  realization policy digest ", _realization_policy_digest())
print("  integrity clauses enforced", len(_INTEGRITY_CLAUSES))
print("  positive control          norm-matched direct answer; diagnoses the")
print("                            instrument, cannot produce a GO, is not a")
print("                            coordinate exchange, is scored separately")
print("  primary outcome rule      alpha=1 exact exchange >= "
      f"{NEW_PROPERTY_DEV_MIN_SUCCESS_RATE:.2f} in every modality with every")
print("                            control at least "
      f"{NEW_PROPERTY_DEV_MIN_CONTROL_MARGIN:.2f} below it, and every integrity")
print("                            clause satisfied; no alpha sweep")
print("  expected Drive usage      ~", _CORRECTED_DRIVE_MB, "MB")
print("  label                     outcome-informed exploratory; not confirmation")
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
    import torch
    if not os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = getpass.getpass("HF_TOKEN (input hidden): ").strip()
    from jlens.mmpilot.real_backend import build_real_backend
    from jlens.mmpilot.tri_modal import assert_audio_protocol
    _leg_generalization_gpu = bool(
        RUN_STAGE7B_LEG_GENERALIZATION_DEVELOPMENT
        or RUN_STAGE7B2_NOVEL_LEG_TARGET_DEVELOPMENT
        or RUN_STAGE7D_LEG_GENERALIZATION_CONFIRMATION
    )
    if _leg_generalization_gpu:
        from jlens.mmpilot.fp32_preflight import preflight_fp32_or_refuse
        _fp32_preflight = preflight_fp32_or_refuse(
            workspace_fraction=0.10, safety_margin=1.10,
        )
        print("Stage 7 fp32 preflight", _fp32_preflight["device_name"])
    BUNDLE = build_real_backend(
        MODEL_REPO_ID, revision=MODEL_REVISION, token=os.environ["HF_TOKEN"],
        device="cuda", allow_model_load=True, resolve_audio=True,
        expect_n_layers=EXPECT_N_LAYERS, expect_d_model=EXPECT_D_MODEL,
        expect_vocab_size=EXPECT_VOCAB,
        dtype=(torch.float32 if _leg_generalization_gpu else torch.bfloat16),
    )
    if BUNDLE.audio_interface is None:
        raise RuntimeError("native spoken audio did not resolve: " + BUNDLE.audio_blocked_reason)
    AUDIO_RECORD = assert_audio_protocol(
        BUNDLE.audio_interface, expected_fingerprint=AUDIO_PROTOCOL_FINGERPRINT
    )
    BACKEND = BUNDLE.backend
    if _leg_generalization_gpu:
        _observed_dtype = str(next(BACKEND.hf_model.parameters()).dtype)
        if _observed_dtype != "torch.float32":
            raise RuntimeError(
                f"Stage 7 requested fp32 but loaded {_observed_dtype}"
            )
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
        build_swap_bases_for_lens, confirmation_leg_count_prompt, holm_adjust,
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

    # The wording lives in jlens.mmpilot.multimodal_lens so Stage 3DA can
    # replay this run under the identical prompt instead of a copy of it.
    _confirmation_prompt = confirmation_leg_count_prompt

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

markdown(
    r"""
## 10E. Stage 3DA — the smallest exact replication of the confirmed result's realization

Stage 5B1A establishes, from the code path alone, that no stored artifact of
`FRESH_MULTIMODAL_CONFIRMATION_GO` can say whether its exchange was realized in
the model's dtype: `unrestricted_swap_trial` computed the post-cast quantities
inside `swap_coordinates` and then kept only the float64 pre-cast solve errors.
Those are exact by construction, so the `1e-5` gate that stage enforced could
never have caught this defect — and equally, it is no evidence that the defect
was present.

This stage is the smallest measurement that can settle it. It replays the
**identical** recruited photographs, modalities, bases, band, positions and
single-token endpoint through the same trial function, twice:

| arm | policy | question |
| --- | --- | --- |
| `uncorrected` | none, exactly as the completed run | does it reproduce the stored top-1 token, and what post-cast error did that run actually incur? |
| `corrected` | the frozen `MODEL_DTYPE_REALIZATION` | does a faithfully realized exchange preserve the recorded outcome? |

Nothing is written into the completed run directory. The completed report is
read by checksum and never opened for writing, and its verdict string is not
restated as a new one. If the uncorrected arm fails to reproduce the stored
tokens, the replay is not the same measurement and no comparison is licensed.

A changed outcome under the corrected instrument does **not** retroactively
invalidate the confirmed result by prose; it licenses a fresh confirmation.
"""
)
code(
    r'''
CONFIRMATION_REALIZATION_REPLICATION = None
if REAL_MODE and CONFIRMATION_REPLICATION_ENABLED:
    from jlens.lens import JacobianLens
    from jlens.mmpilot.coordinate_swap import (
        random_two_direction_basis, resolve_concept_token,
    )
    from jlens.mmpilot.multimodal_followup import (
        legacy_confirmation_replication_verdict, load_verified_report,
    )
    from jlens.mmpilot.multimodal_instrument import (
        MODEL_DTYPE_REALIZATION, POST_CAST_TOLERANCE, realization_policy_digest,
    )
    from jlens.mmpilot.multimodal_lens import (
        build_swap_bases_for_lens, confirmation_leg_count_prompt,
        load_broad_pooled_development_source, open_answer_matches,
        unrestricted_swap_trial,
    )
    from jlens.mmpilot.store import RunFingerprint, UnitStore, safe_key

    _replication_source_path = (
        Path(FRESH_CONFIRMATION_RUN_DIR) /
        "fresh_multimodal_confirmation_report.json"
    )
    _completed = load_verified_report(
        _replication_source_path,
        expected_checksum=EXPECTED_FRESH_CONFIRMATION_REPORT_CHECKSUM,
        label="completed fresh multimodal confirmation",
    )
    if _completed.get("verdict") != "FRESH_MULTIMODAL_CONFIRMATION_GO":
        raise RuntimeError(
            "Stage 3DA replays the confirmed run only; this report is "
            + str(_completed.get("verdict"))
        )
    _stored_rows = [dict(row) for row in _completed.get("rows") or []]
    if not _stored_rows:
        raise RuntimeError("the pinned confirmation report stores no trial rows")
    _stored_index = {
        (str(row["group_id"]), str(row["modality"]), str(row["condition"])): row
        for row in _stored_rows
    }
    _stored_group_ids = list(dict.fromkeys(
        str(row["group_id"]) for row in _stored_rows
    ))
    _group_index = {str(group["group_id"]): group for group in GROUPS}
    _missing = [g for g in _stored_group_ids if g not in _group_index]
    if _missing:
        raise RuntimeError(
            "the cached manifest is missing confirmation groups " + repr(_missing[:10])
        )

    _replication_config = {
        "study": "broad_pooled_confirmation_realization_replication.v1",
        "original_report_checksum": EXPECTED_FRESH_CONFIRMATION_REPORT_CHECKSUM,
        "original_run_dir": str(FRESH_CONFIRMATION_RUN_DIR),
        "original_run_modified": False,
        "model_repo_id": MODEL_REPO_ID,
        "model_revision": MODEL_REVISION,
        "audio_protocol_fingerprint": AUDIO_PROTOCOL_FINGERPRINT,
        "manifest_checksum": MANIFEST_CHECKSUM,
        "lens_checksum": EXPECTED_BROAD_POOLED_LENS_CHECKSUM,
        "lens_refitted": False,
        "direction": list(CONFIRMATION_DIRECTION),
        "alpha": CONFIRMATION_ALPHA,
        "layers": list(BROAD_POOLED_BAND),
        "positions": "every original prompt position",
        "endpoint": "unrestricted full-vocabulary next-token top1",
        "conditions": ["exact", "zero", "random", "unrelated"],
        "arms": ["uncorrected", "corrected"],
        "n_groups": len(_stored_group_ids),
        "model_dtype_realization_policy": MODEL_DTYPE_REALIZATION.to_dict(),
        "model_dtype_realization_policy_digest": realization_policy_digest(),
        "post_cast_relative_tolerance": POST_CAST_TOLERANCE,
        "measures_only_realization_and_outcome_identity": True,
        "commit": COMMIT,
    }
    _replication_digest = payload_checksum(_replication_config)
    CONFIRMATION_REPLICATION_RUN_DIR = (
        RUNS_ROOT / "mmconfirmrealization" /
        f"mmconfirmrealization_real_{_replication_digest.split(':')[1][:12]}"
    )
    CONFIRMATION_REPLICATION_RUN_DIR.mkdir(parents=True, exist_ok=True)
    (CONFIRMATION_REPLICATION_RUN_DIR / "scientific_config.json").write_text(
        json.dumps(_replication_config, indent=2)
    )
    _replication_store = UnitStore(
        CONFIRMATION_REPLICATION_RUN_DIR,
        RunFingerprint(
            mode="real", model_repo_id=MODEL_REPO_ID,
            model_revision=MODEL_REVISION, processor_revision=MODEL_REVISION,
            layers=tuple(BROAD_POOLED_BAND),
            lens_checksum=EXPECTED_BROAD_POOLED_LENS_CHECKSUM,
            manifest_checksum=MANIFEST_CHECKSUM,
            split_id=EXPECTED_FRESH_CONFIRMATION_REPORT_CHECKSUM,
            intervention_config={
                "alpha": CONFIRMATION_ALPHA,
                "conditions": ["exact", "zero", "random", "unrelated"],
                "arms": ["uncorrected", "corrected"],
                "positions": "all_original_prompt_positions",
                "model_dtype_realization_policy_digest": realization_policy_digest(),
            },
            extra={"study_digest": _replication_digest},
        ),
    )
    print("confirmation replication run state", _replication_store.open())

    _replication_pin = load_broad_pooled_development_source(
        BROAD_DEVELOPMENT_RUN_DIR,
        expected_report_checksum=EXPECTED_BROAD_DEVELOPMENT_REPORT_CHECKSUM,
        expected_population_digest=EXPECTED_BROAD_DEVELOPMENT_POPULATION_DIGEST,
        expected_lens_checksum=EXPECTED_BROAD_POOLED_LENS_CHECKSUM,
        expected_direction=CONFIRMATION_DIRECTION,
    )
    _replication_lens = JacobianLens.load(_replication_pin["lens_path"])
    _replication_unembed = BACKEND.unembedding_weight()
    _replication_tokens = {
        name: resolve_concept_token(BACKEND.encode_candidate, name)
        for name in (*CONFIRMATION_DIRECTION, *BROAD_POOLED_CONTROLS)
    }
    _replication_exact = build_swap_bases_for_lens(
        _replication_lens, _replication_unembed, layers=BROAD_POOLED_BAND,
        source=_replication_tokens[CONFIRMATION_DIRECTION[0]],
        target=_replication_tokens[CONFIRMATION_DIRECTION[1]],
    )
    _replication_random = {
        layer: random_two_direction_basis(basis, seed=20260822 + layer)
        for layer, basis in _replication_exact.items()
    }
    _replication_unrelated = build_swap_bases_for_lens(
        _replication_lens, _replication_unembed, layers=BROAD_POOLED_BAND,
        source=_replication_tokens[BROAD_POOLED_CONTROLS[0]],
        target=_replication_tokens[BROAD_POOLED_CONTROLS[1]],
    )
    _replication_conditions = (
        ("exact", CONFIRMATION_ALPHA, _replication_exact),
        ("zero", 0.0, _replication_exact),
        ("random", CONFIRMATION_ALPHA, _replication_random),
        ("unrelated", CONFIRMATION_ALPHA, _replication_unrelated),
    )
    _replication_rows = []
    _replication_total = len(_stored_group_ids) * 3 * 4 * 2
    for _group_id in _stored_group_ids:
        _group = _group_index[_group_id]
        for _modality in ("text", "image", "spoken_audio"):
            _inputs = None
            for _condition, _alpha, _bases in _replication_conditions:
                for _arm, _policy in (
                    ("uncorrected", None),
                    ("corrected", MODEL_DTYPE_REALIZATION),
                ):
                    _key = safe_key(
                        "realization", _arm, _group_id, _modality, _condition
                    )
                    _stored = _replication_store.load("intervention", _key)
                    if _stored is None:
                        _original = _stored_index.get(
                            (_group_id, _modality, _condition)
                        )
                        if _original is None:
                            raise RuntimeError(
                                "the pinned confirmation report has no row for "
                                f"{(_group_id, _modality, _condition)}"
                            )
                        if _inputs is None:
                            _inputs = build_group_inputs(
                                _group, _modality,
                                confirmation_leg_count_prompt(
                                    _modality, _group["caption"]
                                ),
                            )
                        _trial = unrestricted_swap_trial(
                            BACKEND, _inputs, bases=_bases, alpha=_alpha,
                            compact_positions=True,
                            realization_policy=_policy,
                        )
                        _surface = BACKEND.decode_token(
                            _trial["patched_top_token_id"]
                        ).strip()
                        _stored = {
                            "arm": _arm,
                            "group_id": _group_id,
                            "image_id": str(_group["image_id"]),
                            "modality": _modality,
                            "condition": _condition,
                            "alpha": float(_alpha),
                            "layers_patched": _trial["layers_patched"],
                            "all_prompt_positions_patched": _trial[
                                "all_prompt_positions_patched"
                            ],
                            "stored_top_token_id": int(
                                _original["patched_top_token_id"]
                            ),
                            "replayed_top_token_id": int(
                                _trial["patched_top_token_id"]
                            ),
                            "stored_success": bool(_original["success"]),
                            "replayed_surface": _surface,
                            "replayed_success": bool(
                                open_answer_matches(
                                    _surface, str(_original["expected"])
                                )
                            ),
                            "max_post_cast_relative_coordinate_error": _trial[
                                "max_post_cast_relative_coordinate_error"
                            ],
                            "max_post_cast_relative_residual_drift": _trial[
                                "max_post_cast_relative_residual_drift"
                            ],
                            "all_model_dtype_realizations_converged": _trial[
                                "all_model_dtype_realizations_converged"
                            ],
                            "max_model_dtype_corrections_applied": _trial[
                                "max_model_dtype_corrections_applied"
                            ],
                            "ideal_pre_cast_coordinate_error": _trial[
                                "max_coordinate_update_error"
                            ],
                            "model_dtype_realization_policy": _trial[
                                "model_dtype_realization_policy"
                            ],
                        }
                        _replication_store.save("intervention", _key, _stored)
                    _replication_rows.append(_stored)
                    if len(_replication_rows) % 48 == 0:
                        print("realization replication trials",
                              len(_replication_rows), "of", _replication_total)

    CONFIRMATION_REALIZATION_REPLICATION = legacy_confirmation_replication_verdict(
        _replication_rows,
        original_report_checksum=EXPECTED_FRESH_CONFIRMATION_REPORT_CHECKSUM,
        layers=BROAD_POOLED_BAND,
    )
    CONFIRMATION_REALIZATION_REPLICATION = {
        **CONFIRMATION_REALIZATION_REPLICATION,
        "scientific_config": _replication_config,
        "rows": _replication_rows,
    }
    CONFIRMATION_REALIZATION_REPLICATION["report_checksum"] = payload_checksum({
        key: value for key, value in CONFIRMATION_REALIZATION_REPLICATION.items()
        if key != "report_checksum"
    })
    _replication_store.save(
        "metric", "confirmation_realization_replication",
        CONFIRMATION_REALIZATION_REPLICATION,
    )
    _replication_path = (
        CONFIRMATION_REPLICATION_RUN_DIR /
        "confirmation_realization_replication_report.json"
    )
    _replication_path.write_text(
        json.dumps(CONFIRMATION_REALIZATION_REPLICATION, indent=2, default=str)
    )
    print("=" * 96)
    print("CONFIRMATION REALIZATION REPLICATION —",
          CONFIRMATION_REALIZATION_REPLICATION["verdict"])
    print("=" * 96)
    print("original report (unmodified) ",
          CONFIRMATION_REALIZATION_REPLICATION["original_report_checksum"])
    print("uncorrected reproduced stored",
          CONFIRMATION_REALIZATION_REPLICATION[
              "uncorrected_reproduced_stored_tokens"],
          f"({CONFIRMATION_REALIZATION_REPLICATION['n_reproduced']}"
          f"/{CONFIRMATION_REALIZATION_REPLICATION['n_uncorrected_rows']})")
    print("original post-cast coord err ",
          CONFIRMATION_REALIZATION_REPLICATION[
              "original_max_post_cast_relative_coordinate_error"])
    print("original post-cast resid     ",
          CONFIRMATION_REALIZATION_REPLICATION[
              "original_max_post_cast_relative_residual_drift"])
    print("original within tolerance    ",
          CONFIRMATION_REALIZATION_REPLICATION["original_within_tolerance"],
          "at", CONFIRMATION_REALIZATION_REPLICATION[
              "post_cast_relative_tolerance"])
    print("corrected post-cast coord err",
          CONFIRMATION_REALIZATION_REPLICATION[
              "corrected_max_post_cast_relative_coordinate_error"])
    print("corrected converged          ",
          CONFIRMATION_REALIZATION_REPLICATION[
              "corrected_realizations_converged"])
    print("outcomes changed by fix      ",
          CONFIRMATION_REALIZATION_REPLICATION["n_outcomes_changed"])
    print("original verdict relabelled  ",
          CONFIRMATION_REALIZATION_REPLICATION["original_verdict_relabelled"])
    print("report                       ", _replication_path)
    print("checksum                     ",
          CONFIRMATION_REALIZATION_REPLICATION["report_checksum"])
elif RUN_STAGE3DA_CONFIRMATION_REALIZATION_REPLICATION:
    print("Stage 3DA requested but blocked by model or its budget.")
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

This cell knows how to name and checksum-pin three prior runs (the fit/eval
split, the four-lens causal screen, and the broad development study) plus the
completed confirmation. It does **not** automatically know about a run outside
that set — an abandoned property family before a fallback, or Experiment B's
opened media before Experiment C runs in the same session. For those, add the
report path to `EXTRA_SPENT_REPORT_PATHS` and rerun this cell before opening
the next population; that widening is unpinned (no checksum) and can only add
exclusions, never remove ones already established.

This cell loads nothing but JSON and never touches the model.
"""
)
code(
    r'''
EXCLUSION_UNIVERSE = None
SPENT_CONFIRMATION = None
EXTRA_SPENT = None
if REAL_MODE and (RUN_ARTIFACT_EXCLUSION_AUDIT or any(FOLLOWUP_STAGES.values())):
    from jlens.mmpilot.multimodal_followup import (
        exclusion_universe, load_extra_spent_image_ids,
        load_spent_confirmation_population,
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
    # Unpinned, best-effort widening. Rerun this cell with a path added here
    # any time a run outside the checksum-pinned set above opened media that
    # must not be reused: an abandoned property family before trying its
    # declared fallback, or Experiment B's opened media before Experiment C
    # runs in the same session. This can only add exclusions, never remove
    # ones the pinned loaders already established.
    EXTRA_SPENT = (
        load_extra_spent_image_ids(EXTRA_SPENT_REPORT_PATHS)
        if EXTRA_SPENT_REPORT_PATHS else None
    )
    EXCLUSION_UNIVERSE = exclusion_universe(
        fit_image_ids=[str(value) for value in PLAN["fit_image_ids"]],
        eval_image_ids=[str(value) for value in PLAN["eval_image_ids"]],
        prior_causal_image_ids=_prior_causal["excluded_image_ids"],
        broad_development_image_ids=_development_source["excluded_image_ids"],
        confirmation_candidate_image_ids=SPENT_CONFIRMATION["candidate_image_ids"],
        extra_image_ids=(
            {"manually_declared_extra_runs": EXTRA_SPENT["image_ids"]}
            if EXTRA_SPENT else None
        ),
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
    if EXTRA_SPENT is not None:
        print("  extra unpinned reports        ", len(EXTRA_SPENT_REPORT_PATHS))
        print("  extra image ids (unverified)  ", EXTRA_SPENT["n_image_ids"])
    else:
        print("  EXTRA_SPENT_REPORT_PATHS is empty — no manual widening applied")
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
## 13B. Stage 5B00 — clarify the animal-sound prompt on spent media

The first animal-sound audit stopped before any intervention because the clean
model sometimes treated the question as asking whether a literal sound was
present. This development-only screen compares two predeclared clarifications
against that original prompt on the **same already-spent cat and cow media**.

The original completions are imported by checksum and never recomputed. No
lens is loaded, no coordinate exchange runs, and no causal outcome enters the
selection. A winner must clear the unchanged 75% clean-capability threshold in
all six concept-by-modality cells. Even then it licenses only a new Stage 5B0
capability audit on fresh development media; it does not license confirmation.
"""
)
code(
    r'''
PROPERTY_PROMPT_SCREEN_REPORT = None
if REAL_MODE and PROPERTY_PROMPT_SCREEN_ENABLED:
    from jlens.mmpilot.multimodal_followup import (
        ANIMAL_SOUND_PROMPT_CANDIDATES, PROPERTY_FAMILIES,
        property_answer_matches, property_prompt, property_prompt_screen_verdict,
    )
    from jlens.mmpilot.store import RunFingerprint, UnitStore, safe_key
    from jlens.mmpilot.workspace_replication import unrestricted_greedy_completion

    _source_root = Path(PROPERTY_PROMPT_SCREEN_SOURCE_RUN_DIR)
    _source_report_path = _source_root / "new_property_audit_report.json"
    _source_population_path = _source_root / "development_population.json"
    _source_config_path = _source_root / "scientific_config.json"
    for _path in (_source_report_path, _source_population_path, _source_config_path):
        if not _path.is_file():
            raise FileNotFoundError(f"prompt-screen source artifact missing: {_path}")

    _source_bytes = _source_report_path.read_bytes()
    _source_file_sha = "sha256:" + hashlib.sha256(_source_bytes).hexdigest()
    if _source_file_sha != EXPECTED_PROPERTY_PROMPT_SCREEN_SOURCE_FILE_SHA256:
        raise RuntimeError(
            "the completed animal-sound audit file changed; refusing to tune a "
            f"prompt against it ({_source_file_sha})"
        )
    _source_report = json.loads(_source_bytes)
    if _source_report.get("audit_digest") != EXPECTED_PROPERTY_PROMPT_SCREEN_SOURCE_AUDIT_DIGEST:
        raise RuntimeError("the completed animal-sound audit digest does not match its pin")
    if _source_report.get("verdict") != "PROPERTY_AUDIT_NO_GO":
        raise RuntimeError("the prompt screen is pinned to the completed NO_GO audit")
    if _source_report.get("family") != "animal_sound":
        raise RuntimeError("the prompt screen source is not the animal-sound audit")
    _source_config = json.loads(_source_config_path.read_text())
    for _field, _expected in (
        ("model_repo_id", MODEL_REPO_ID),
        ("model_revision", MODEL_REVISION),
        ("manifest_checksum", MANIFEST_CHECKSUM),
    ):
        if _source_config.get(_field) != _expected:
            raise RuntimeError(
                f"prompt-screen source {_field}={_source_config.get(_field)!r}, "
                f"expected {_expected!r}"
            )

    _source_population_record = json.loads(_source_population_path.read_text())
    _source_ids = _source_population_record.get("population") or {}
    _groups_by_id = {str(row["group_id"]): row for row in GROUPS}
    _screen_population = {}
    for _concept in PROPERTY_PROMPT_SCREEN_CONCEPTS:
        _ids = [str(row["group_id"]) for row in _source_ids.get(_concept, ())]
        if len(_ids) != NEW_PROPERTY_DEV_CANDIDATES_PER_CONCEPT:
            raise RuntimeError(
                f"source population has {len(_ids)} {_concept} groups, expected "
                f"{NEW_PROPERTY_DEV_CANDIDATES_PER_CONCEPT}"
            )
        _missing = [group_id for group_id in _ids if group_id not in _groups_by_id]
        if _missing:
            raise RuntimeError(f"source population groups missing from cache: {_missing[:3]}")
        _screen_population[_concept] = [_groups_by_id[group_id] for group_id in _ids]

    _candidate_digest = payload_checksum([
        {
            "prompt_id": row["prompt_id"],
            "rationale": row["rationale"],
            "templates": row["templates"],
        }
        for row in ANIMAL_SOUND_PROMPT_CANDIDATES
    ])
    _screen_config = {
        "study": "multimodal_property_prompt_screen.v1",
        "property_family": "animal_sound",
        "prompt_candidates_digest": _candidate_digest,
        "prompt_candidate_ids": [row["prompt_id"] for row in ANIMAL_SOUND_PROMPT_CANDIDATES],
        "concepts": list(PROPERTY_PROMPT_SCREEN_CONCEPTS),
        "modalities": ["text", "image", "spoken_audio"],
        "expected_per_cell": NEW_PROPERTY_DEV_CANDIDATES_PER_CONCEPT,
        "min_clean_capability_rate": NEW_PROPERTY_MIN_CLEAN_CAPABILITY_RATE,
        "source_run_dir": str(_source_root),
        "source_file_sha256": _source_file_sha,
        "source_audit_digest": _source_report["audit_digest"],
        "population_reused": True,
        "population_already_spent": True,
        "causal_outcomes_used_for_selection": False,
        "lens_fitted": False,
        "backward_passes": 0,
        "teacher_forcing_used": False,
        "candidate_list_supplied": False,
        "commit": COMMIT,
    }
    _screen_digest = payload_checksum(_screen_config)
    PROPERTY_PROMPT_SCREEN_RUN_DIR = (
        RUNS_ROOT / "mmps" /
        f"mmps_real_{_screen_digest.split(':')[1][:12]}"
    )
    PROPERTY_PROMPT_SCREEN_RUN_DIR.mkdir(parents=True, exist_ok=True)
    (PROPERTY_PROMPT_SCREEN_RUN_DIR / "scientific_config.json").write_text(
        json.dumps(_screen_config, indent=2)
    )
    _population_digest = payload_checksum({
        concept: [row["group_id"] for row in groups]
        for concept, groups in _screen_population.items()
    })
    _screen_fingerprint = RunFingerprint(
        mode="real", model_repo_id=MODEL_REPO_ID, model_revision=MODEL_REVISION,
        processor_revision=MODEL_REVISION, layers=(), lens_checksum="none",
        manifest_checksum=MANIFEST_CHECKSUM, split_id=_population_digest,
        intervention_config={
            "kind": "clean_capability_prompt_screen",
            "prompt_candidates_digest": _candidate_digest,
            "max_new_tokens": NEW_PROPERTY_MAX_NEW_TOKENS,
        },
        extra={"study_digest": _screen_digest},
    )
    _screen_store = UnitStore(PROPERTY_PROMPT_SCREEN_RUN_DIR, _screen_fingerprint)
    print("prompt-screen run state", _screen_store.open())

    _screen_rows = []
    _source_rows = list(_source_report.get("capability_rows") or [])
    for _concept in PROPERTY_PROMPT_SCREEN_CONCEPTS:
        _allowed_ids = {row["group_id"] for row in _screen_population[_concept]}
        _baseline = [
            row for row in _source_rows
            if row.get("concept") == _concept
            and row.get("group_id") in _allowed_ids
            and row.get("modality") in ("text", "image", "spoken_audio")
        ]
        if len(_baseline) != NEW_PROPERTY_DEV_CANDIDATES_PER_CONCEPT * 3:
            raise RuntimeError(
                f"source report has {len(_baseline)} baseline rows for {_concept}, "
                f"expected {NEW_PROPERTY_DEV_CANDIDATES_PER_CONCEPT * 3}"
            )
        _screen_rows.extend([
            {**row, "prompt_id": "baseline_v1", "work": "imported"}
            for row in _baseline
        ])

    _property = PROPERTY_FAMILIES["animal_sound"]
    _answers = {
        concept: _property.answer_for(concept)
        for concept in PROPERTY_PROMPT_SCREEN_CONCEPTS
    }
    _computed = _reused = 0
    for _candidate in ANIMAL_SOUND_PROMPT_CANDIDATES:
        _prompt_id = _candidate["prompt_id"]
        if _prompt_id == "baseline_v1":
            continue
        for _concept in PROPERTY_PROMPT_SCREEN_CONCEPTS:
            for _group in _screen_population[_concept]:
                for _modality in ("text", "image", "spoken_audio"):
                    _key = safe_key(
                        "promptscreen", _prompt_id, _concept,
                        _group["group_id"], _modality,
                    )
                    _row = _screen_store.load("capability", _key)
                    if _row is None:
                        _prompt = property_prompt(
                            _prompt_id, _modality, _group["caption"]
                        )
                        _inputs = build_group_inputs(_group, _modality, _prompt)
                        _clean = unrestricted_greedy_completion(
                            BACKEND, _inputs, answer="",
                            max_new_tokens=NEW_PROPERTY_MAX_NEW_TOKENS,
                        )
                        _row = {
                            "prompt_id": _prompt_id,
                            "concept": _concept,
                            "group_id": _group["group_id"],
                            "image_id": _group["image_id"],
                            "modality": _modality,
                            "generated": _clean["generated_text"],
                            "expected_aliases": list(_answers[_concept].aliases),
                            "pass": property_answer_matches(
                                _clean["generated_text"], _answers[_concept]
                            ),
                            "work": "computed",
                        }
                        _screen_store.save("capability", _key, _row)
                        _computed += 1
                    else:
                        _reused += 1
                    _screen_rows.append(_row)
                    if (_computed + _reused) % 48 == 0:
                        print(
                            "prompt screen", _computed + _reused,
                            "computed", _computed, "reused", _reused,
                        )

    _base_report = property_prompt_screen_verdict(
        _screen_rows,
        expected_per_cell=NEW_PROPERTY_DEV_CANDIDATES_PER_CONCEPT,
        min_clean_capability_rate=NEW_PROPERTY_MIN_CLEAN_CAPABILITY_RATE,
        concepts=PROPERTY_PROMPT_SCREEN_CONCEPTS,
    )
    _body = {
        **{k: v for k, v in _base_report.items() if k != "report_checksum"},
        "scientific_config": _screen_config,
        "source_population_digest": _population_digest,
        "units": {"computed": _computed, "reused": _reused,
                  "imported_baseline": len(_source_rows) // 2},
    }
    PROPERTY_PROMPT_SCREEN_REPORT = {
        **_body, "report_checksum": payload_checksum(_body)
    }
    _screen_store.save("metric", "property_prompt_screen", PROPERTY_PROMPT_SCREEN_REPORT)
    _screen_path = PROPERTY_PROMPT_SCREEN_RUN_DIR / "property_prompt_screen_report.json"
    _screen_path.write_text(
        json.dumps(PROPERTY_PROMPT_SCREEN_REPORT, indent=2, default=str)
    )
    print("=" * 96)
    print("PROPERTY PROMPT SCREEN —", PROPERTY_PROMPT_SCREEN_REPORT["verdict"])
    print("=" * 96)
    for _row in PROPERTY_PROMPT_SCREEN_REPORT["candidates"]:
        print(_row["prompt_id"], "minimum", round(_row["minimum_cell_rate"], 3),
              "mean", round(_row["mean_cell_rate"], 3),
              "passes", _row["passes_every_cell"])
        print("  rates", _row["rates"])
    print("selected", PROPERTY_PROMPT_SCREEN_REPORT["selected_prompt_id"])
    print("causal spending licensed", PROPERTY_PROMPT_SCREEN_REPORT["causal_spending_licensed"])
    print("report", _screen_path)
    print("checksum", PROPERTY_PROMPT_SCREEN_REPORT["report_checksum"])
    print("resume", _screen_store.status_report())
    if PROPERTY_PROMPT_SCREEN_REPORT["verdict"] == "PROPERTY_PROMPT_SCREEN_GO":
        print("NEXT: rerun Stage 5B0 on fresh development media with the selected")
        print("prompt ID and this run directory/checksum pinned in configuration.")
    else:
        print("NO_GO: do not run Stage 5B0/5B1 for animal sound.")
elif RUN_STAGE5B00_PROPERTY_PROMPT_SCREEN:
    print("Stage 5B00 requested but blocked by model or prompt-screen budget.")
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
        PROPERTY_FAMILIES, PropertyAnswer, artifact_exclusion_audit,
        assert_lens_reused_not_refitted, assert_property_pair_changes_answer,
        audit_property_family, generation_trial_row, load_verified_report,
        new_property_development_verdict, property_answer_matches,
        property_prompt, property_prompt_candidate,
    )
    from jlens.mmpilot.multimodal_lens import (
        build_swap_bases_for_lens, load_broad_pooled_development_source,
        select_causal_groups,
    )
    from jlens.mmpilot.multimodal_instrument import MODEL_DTYPE_REALIZATION
    from jlens.mmpilot.store import RunFingerprint, UnitStore, safe_key
    from jlens.mmpilot.workspace_replication import (
        unrestricted_greedy_completion, unrestricted_greedy_swap_trial,
    )

    _property = PROPERTY_FAMILIES[NEW_PROPERTY_FAMILY]
    _prompt_candidate = property_prompt_candidate(NEW_PROPERTY_PROMPT_ID)
    _prompt_screen_pin = None
    if NEW_PROPERTY_PROMPT_ID != "baseline_v1":
        if (
            NEW_PROPERTY_PROMPT_SCREEN_RUN_DIR is None
            or EXPECTED_NEW_PROPERTY_PROMPT_SCREEN_CHECKSUM is None
        ):
            raise RuntimeError(
                "a non-baseline NEW_PROPERTY_PROMPT_ID requires the completed "
                "Stage 5B00 run directory and report checksum"
            )
        _prompt_screen_pin = load_verified_report(
            Path(NEW_PROPERTY_PROMPT_SCREEN_RUN_DIR)
            / "property_prompt_screen_report.json",
            expected_checksum=EXPECTED_NEW_PROPERTY_PROMPT_SCREEN_CHECKSUM,
            label="animal-sound prompt-screen report",
        )
        if _prompt_screen_pin.get("verdict") != "PROPERTY_PROMPT_SCREEN_GO":
            raise RuntimeError("the pinned prompt screen did not pass")
        if _prompt_screen_pin.get("selected_prompt_id") != NEW_PROPERTY_PROMPT_ID:
            raise RuntimeError(
                "NEW_PROPERTY_PROMPT_ID is not the winner recorded by the "
                "pinned prompt screen"
            )
        if _prompt_screen_pin.get("causal_outcomes_used_for_selection") is not False:
            raise RuntimeError("the prompt was not selected on clean capability alone")
    _source_pin = load_broad_pooled_development_source(
        BROAD_DEVELOPMENT_RUN_DIR,
        expected_report_checksum=EXPECTED_BROAD_DEVELOPMENT_REPORT_CHECKSUM,
        expected_population_digest=EXPECTED_BROAD_DEVELOPMENT_POPULATION_DIGEST,
        expected_lens_checksum=EXPECTED_BROAD_POOLED_LENS_CHECKSUM,
        expected_direction=CONFIRMATION_DIRECTION,
    )
    # Every direction whose two concepts are both declared is checked against
    # the property table before any media are opened; a pair that does not
    # change the answer cannot be requested. A pair involving an empirical
    # concept (bird) cannot be checked yet — there is no answer to compare
    # until the capability screen runs — so it is deferred to
    # audit_property_family's own candidate_directions computation below,
    # which is where it is actually gated.
    for _pair in NEW_PROPERTY_DEV_DIRECTIONS:
        if any(
            _property.answer_for(_concept).empirical_answer_required
            for _concept in _pair
        ):
            continue
        assert_property_pair_changes_answer(NEW_PROPERTY_FAMILY, _pair[0], _pair[1])

    _dev_config = {
        "study": "multimodal_new_property_development.v1",
        "property_family": NEW_PROPERTY_FAMILY,
        "prompt_id": NEW_PROPERTY_PROMPT_ID,
        "prompt_by_modality": dict(_prompt_candidate["templates"]),
        "prompt_screen_report_checksum": (
            EXPECTED_NEW_PROPERTY_PROMPT_SCREEN_CHECKSUM
            if _prompt_screen_pin is not None else None
        ),
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
            "prompt_id": NEW_PROPERTY_PROMPT_ID,
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
    # Collection stores the raw completion only. It cannot score "pass" here:
    # a concept whose answer is empirical (bird) has no declared alias to
    # score against until DOMINANT_ANSWER_RULE resolves one, and that rule
    # needs exactly these completions as its input. Scoring is therefore
    # deferred to a single post-hoc pass below, run uniformly for every
    # concept, declared or empirical, against whatever the audit resolves.
    _capability_rows = []
    for _concept in NEW_PROPERTY_CONCEPTS:
        for _group in _dev_population[_concept]:
            for _modality in ("text", "image", "spoken_audio"):
                _key = safe_key("propcap", _concept, _group["group_id"], _modality)
                _row = _dev_store.load("capability", _key)
                if _row is None:
                    _inputs = build_group_inputs(
                        _group, _modality,
                        property_prompt(
                            NEW_PROPERTY_PROMPT_ID, _modality, _group["caption"]
                        ),
                    )
                    _clean = unrestricted_greedy_completion(
                        BACKEND, _inputs, answer="",
                        max_new_tokens=NEW_PROPERTY_MAX_NEW_TOKENS,
                    )
                    _row = {
                        "concept": _concept, "group_id": _group["group_id"],
                        "image_id": _group["image_id"], "modality": _modality,
                        "generated": _clean["generated_text"],
                    }
                    _dev_store.save("capability", _key, _row)
                _capability_rows.append(_row)
                if len(_capability_rows) % 72 == 0:
                    print("property capability", len(_capability_rows))

    # Raw completions per concept per modality, so DOMINANT_ANSWER_RULE can
    # resolve any concept whose answer is empirical rather than declared.
    # The rule was fixed in code before these were read.
    _completions_by_concept = {
        concept: {
            modality: [
                row["generated"] for row in _capability_rows
                if row["concept"] == concept and row["modality"] == modality
            ]
            for modality in ("text", "image", "spoken_audio")
        }
        for concept in NEW_PROPERTY_CONCEPTS
    }
    # Pass 1: resolve any empirical concept (bird) from the raw completions
    # only. No clean_capability is supplied here — there is nothing to supply
    # yet for a declared concept (nothing has been scored) and an empirical
    # concept's own capability check is derived internally from its
    # resolution, so this pass exists purely to obtain _resolved_answers.
    # Its verdict and its "usable"/"candidate_directions" fields are NOT the
    # final ones and must not be read past this point.
    _resolution_pass = audit_property_family(
        NEW_PROPERTY_FAMILY,
        available_media={
            concept: len(rows) for concept, rows in _dev_population.items()
        },
        min_media_per_concept=NEW_PROPERTY_DEV_CANDIDATES_PER_CONCEPT,
        min_clean_capability_rate=NEW_PROPERTY_MIN_CLEAN_CAPABILITY_RATE,
        observed_completions=_completions_by_concept,
    )

    # Resolved answers for every concept: declared ones unchanged, empirical
    # ones (bird) filled in from what pass 1 just resolved. This table, not
    # PROPERTY_FAMILIES directly, is the one source of truth for scoring from
    # here on — including inside Stage 5B1's swap trials below, where the
    # target answer for a direction like cat->bird must be this resolved
    # value and not the empty declared placeholder.
    _resolved_answers = {
        row["concept"]: PropertyAnswer(
            concept=row["concept"], answer=str(row["answer"]),
            aliases=tuple(row["aliases"]), admissible=bool(row["admissible"]),
            reason=str(row["reason"]),
            empirical_answer_required=bool(row.get("empirical_answer_required")),
        )
        for row in _resolution_pass["concepts"]
    }
    for _row in _capability_rows:
        _resolved = _resolved_answers.get(_row["concept"])
        _row["expected_aliases"] = list(_resolved.aliases) if _resolved else []
        _row["pass"] = bool(
            _resolved
            and _resolved.aliases
            and property_answer_matches(_row["generated"], _resolved)
        )

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
    # Pass 2: the real, gated audit. clean_capability now carries the actual
    # post-hoc scored rates, so a declared concept that misses the threshold
    # in any modality is correctly marked unusable — pass 1 could not do this
    # because those rates did not exist until the scoring above ran. An
    # empirical concept's own capability check is unaffected: the module
    # derives it from the resolution's rates_by_modality regardless of what
    # clean_capability contains, so this pass agrees with pass 1 on bird.
    PROPERTY_AUDIT_REPORT = audit_property_family(
        NEW_PROPERTY_FAMILY,
        available_media={
            concept: len(rows) for concept, rows in _dev_population.items()
        },
        min_media_per_concept=NEW_PROPERTY_DEV_CANDIDATES_PER_CONCEPT,
        clean_capability=_capability_by_concept,
        min_clean_capability_rate=NEW_PROPERTY_MIN_CLEAN_CAPABILITY_RATE,
        observed_completions=_completions_by_concept,
    )
    PROPERTY_AUDIT_REPORT = {
        **PROPERTY_AUDIT_REPORT,
        "prompt_id": NEW_PROPERTY_PROMPT_ID,
        "question": "modality-specific templates fixed by prompt_id",
        "prompt_rationale": _prompt_candidate["rationale"],
        "prompt_by_modality": dict(_prompt_candidate["templates"]),
        "prompt_screen_report_checksum": (
            EXPECTED_NEW_PROPERTY_PROMPT_SCREEN_CHECKSUM
            if _prompt_screen_pin is not None else None
        ),
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
    print("perceptually avail. ", PROPERTY_AUDIT_REPORT["perceptually_available"],
          "| family disqualified", PROPERTY_AUDIT_REPORT["family_disqualified"])
    for _row in PROPERTY_AUDIT_REPORT["concepts"]:
        _res = _row.get("empirical_resolution")
        if _res:
            print(f"  empirical answer for {_row['concept']}:",
                  "RESOLVED ->" + str(_res.get("answer")) if _res.get("resolved")
                  else "UNRESOLVED (" + str(_res.get("reason")) + ")")
            print("    counts by modality", _res.get("counts_by_modality"))
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
                _target_answer = _resolved_answers[_tgt]
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
                                    property_prompt(
                                        NEW_PROPERTY_PROMPT_ID, _modality,
                                        _group["caption"],
                                    ),
                                )
                                _trial = unrestricted_greedy_swap_trial(
                                    BACKEND, _inputs, bases=_bases, alpha=_alpha,
                                    answer=_target_answer.answer,
                                    max_new_tokens=NEW_PROPERTY_MAX_NEW_TOKENS,
                                    realization_policy=MODEL_DTYPE_REALIZATION,
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
## 14B. Stages 5B01, 5B1A and 5B1RC — audit the audio linkage, amend the flawed run, then rerun it correctly

### What went wrong

The completed recruited exploratory run
`mmnewpropertyrescue_real_6af6affcb145` reported 0/8 in every modality for both
directions with `integrity_pass = true`. That flag was wrong, for two
independent reasons.

1. **The realization policy was never passed.** Stage 5B1R called
   `unrestricted_greedy_swap_trial` without `realization_policy=`, so the
   float64 coordinate exchange went straight into the bf16 residual stream with
   no bounded correction. The trial diagnostics recorded post-cast **relative**
   coordinate errors of roughly 0.21 (exact), 0.10 (random) and 0.29
   (unrelated) against this repository's frozen tolerance of 0.02. The model
   consumed something other than the intended exchange.
2. **The verdict never looked.** `generation_trial_row` flattened the post-cast
   errors correctly, but `_cell_records` scored `integrity_pass` from two
   clauses only — patched positions and the layer list. It ignored the
   post-cast errors, `all_hooks_fired`, `all_finite`, the alpha=1
   exact-exchange flag and the realization convergence flag.

So a broken instrument was reported in the vocabulary of a scientific null.

### What these stages do

**Stage 5B01** is the unchanged CPU-only audio metadata linkage audit.

**Stage 5B1A** is CPU-only and read-only. It writes an amendment *beside* the
flawed run that pins its report checksum, names the omitted integrity clauses,
reclassifies it as `INSTRUMENT_INCONCLUSIVE`, and records
`scientific_recompute = 0`. It rewrites nothing: the original report's bytes are
re-hashed after the write and the hash is printed. It also audits the confirmed
leg-count result's code path and states plainly whether its stored artifacts can
settle the same question.

**Stage 5B1RC** is the corrected causal rerun. Same property, prompt, concepts,
directions, band, alpha, positions, modalities, population, controls and
thresholds — nothing about the science moved. What changed is the instrument:
the frozen `MODEL_DTYPE_REALIZATION` policy reaches every generated condition,
the verdict enforces the full integrity clause set, and a **norm-matched
direct-answer positive control** runs beside the exchange. The control uses the
repository's existing `unrestricted_greedy_direct_answer_trial`; it injects the
target answer's own lens direction with exactly the L2 norm the exchange would
have had at that layer and position. Its rule is frozen here, before any
corrected outcome exists:

| exact exchange | direct-answer control | verdict |
| --- | --- | --- |
| passes, controls flat | any | `EFFECT_GO` (exploratory) |
| fails | passes | `SCIENTIFIC_NULL` |
| fails | fails | `INCONCLUSIVE` |
| any | any, controls moved | `CONTROL_FAILURE` |
| any | any, integrity failed | `INSTRUMENT_FAILURE` |

The positive control diagnoses the instrument. It cannot turn a failed exchange
into a GO — `instrument_state` has no branch that lets it — and there is no
alpha sweep: alpha=1 remains the only exchange strength tested.
"""
)
code(
    r'''
AUDIO_LINKAGE_REPORT = None
CORRECTED_EXPLORATORY_REPORT = None
INSTRUMENT_AMENDMENT = None
LEGACY_CONFIRMATION_AUDIT = None
if REAL_MODE and (AUDIO_LINKAGE_AUDIT_ENABLED or CORRECTED_EXPLORATORY_ENABLED):
    from jlens.mmpilot.multimodal_followup import (
        PROPERTY_FAMILIES, audio_metadata_linkage_audit,
        generation_trial_row, load_file_by_sha256, property_prompt,
        recruit_all_modality_capable_groups,
    )

    _rescue_source_root = Path(RECRUITED_EXPLORATORY_SOURCE_RUN_DIR)
    _rescue_source_path = _rescue_source_root / "new_property_audit_report.json"
    _rescue_source = load_file_by_sha256(
        _rescue_source_path,
        expected_file_sha256=EXPECTED_RECRUITED_SOURCE_FILE_SHA256,
        label="identity-explicit animal-sound audit",
    )
    if _rescue_source.get("audit_digest") != EXPECTED_RECRUITED_SOURCE_AUDIT_DIGEST:
        raise RuntimeError("the recruited-exploratory source audit digest changed")
    if _rescue_source.get("verdict") != "PROPERTY_AUDIT_NO_GO":
        raise RuntimeError("the source aggregate result is not the pinned NO_GO")
    if _rescue_source.get("prompt_id") != "identity_explicit_v1":
        raise RuntimeError("the source did not use the selected identity-explicit prompt")

    _source_capability_rows = [
        dict(row) for row in _rescue_source.get("capability_rows") or []
        if row.get("concept") in RECRUITED_EXPLORATORY_CONCEPTS
    ]
    _group_index = {str(group["group_id"]): group for group in GROUPS}
    _ordered_group_ids = list(dict.fromkeys(
        str(row["group_id"]) for row in _source_capability_rows
    ))
    _missing_group_ids = [
        group_id for group_id in _ordered_group_ids if group_id not in _group_index
    ]
    if _missing_group_ids:
        raise RuntimeError(
            "the cached synchronized manifest is missing source groups "
            + repr(_missing_group_ids[:10])
        )
    _source_groups = [_group_index[group_id] for group_id in _ordered_group_ids]

    _alignment_config = {
        "study": "multimodal_new_property_audio_metadata_linkage_audit.v1",
        "source_file_sha256": EXPECTED_RECRUITED_SOURCE_FILE_SHA256,
        "source_audit_digest": EXPECTED_RECRUITED_SOURCE_AUDIT_DIGEST,
        "manifest_checksum": MANIFEST_CHECKSUM,
        "concept": "cow",
        "failed_only": True,
        "model_forwards": 0,
        "backward_passes": 0,
        "commit": COMMIT,
    }
    _alignment_digest = payload_checksum(_alignment_config)
    AUDIO_LINKAGE_RUN_DIR = (
        RUNS_ROOT / "mmnewpropertyalign" /
        f"mmnewpropertyalign_real_{_alignment_digest.split(':')[1][:12]}"
    )
    AUDIO_LINKAGE_RUN_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_LINKAGE_REPORT = audio_metadata_linkage_audit(
        GROUPS, _source_capability_rows, concept="cow", failed_only=True,
        require_local_files=True,
    )
    AUDIO_LINKAGE_REPORT = {
        **AUDIO_LINKAGE_REPORT,
        "scientific_config": _alignment_config,
        "source_report_path": str(_rescue_source_path),
        "source_aggregate_verdict": "PROPERTY_AUDIT_NO_GO",
        "source_aggregate_verdict_unchanged": True,
    }
    AUDIO_LINKAGE_REPORT["report_checksum"] = payload_checksum({
        key: value for key, value in AUDIO_LINKAGE_REPORT.items()
        if key != "report_checksum"
    })
    _alignment_path = AUDIO_LINKAGE_RUN_DIR / "new_property_audio_alignment_report.json"
    _alignment_path.write_text(
        json.dumps(AUDIO_LINKAGE_REPORT, indent=2, default=str)
    )
    print("=" * 96)
    print("AUDIO METADATA LINKAGE —", AUDIO_LINKAGE_REPORT["verdict"])
    print("=" * 96)
    print("failed cow-audio rows audited", AUDIO_LINKAGE_REPORT["n_rows_audited"])
    print("metadata linkage verified   ",
          AUDIO_LINKAGE_REPORT["metadata_linkage_verified"])
    print("waveform transcribed        ",
          AUDIO_LINKAGE_REPORT["waveform_content_independently_transcribed"])
    print("model forwards / fits       0 / 0")
    print("source verdict unchanged    ",
          AUDIO_LINKAGE_REPORT["source_aggregate_verdict_unchanged"])
    print("report                      ", _alignment_path)
    print("checksum                    ", AUDIO_LINKAGE_REPORT["report_checksum"])

    if CORRECTED_EXPLORATORY_ENABLED:
        if AUDIO_LINKAGE_REPORT["verdict"] != "AUDIO_METADATA_LINKAGE_GO":
            raise RuntimeError(
                "recruited causal spending refused: audio metadata linkage did not pass"
            )
        from jlens.lens import JacobianLens
        from jlens.mmpilot.coordinate_swap import (
            ANSWER_READOUT_FIRST_TOKEN, random_two_direction_basis,
            resolve_answer_readout_token, resolve_concept_token,
        )
        from jlens.mmpilot.multimodal_followup import (
            corrected_exploratory_verdict, direct_answer_trial_row,
        )
        from jlens.mmpilot.multimodal_instrument import (
            INTEGRITY_CLAUSES, MODEL_DTYPE_REALIZATION, POST_CAST_TOLERANCE,
            realization_policy_digest,
        )
        from jlens.mmpilot.multimodal_lens import (
            build_swap_bases_for_lens, load_broad_pooled_development_source,
            selected_lens_vector,
        )
        from jlens.mmpilot.store import RunFingerprint, UnitStore, safe_key
        from jlens.mmpilot.workspace_replication import (
            unrestricted_greedy_direct_answer_trial, unrestricted_greedy_swap_trial,
        )

        _recruitment = recruit_all_modality_capable_groups(
            _source_groups, _source_capability_rows,
            concepts=RECRUITED_EXPLORATORY_CONCEPTS,
            n_per_concept=RECRUITED_EXPLORATORY_IMAGES_PER_DIRECTION,
        )
        if not _recruitment["complete"]:
            raise RuntimeError(
                "not enough all-modality clean-capable groups for the frozen "
                f"exploratory size: {_recruitment['eligible_counts']}"
            )
        _source_pin = load_broad_pooled_development_source(
            BROAD_DEVELOPMENT_RUN_DIR,
            expected_report_checksum=EXPECTED_BROAD_DEVELOPMENT_REPORT_CHECKSUM,
            expected_population_digest=EXPECTED_BROAD_DEVELOPMENT_POPULATION_DIGEST,
            expected_lens_checksum=EXPECTED_BROAD_POOLED_LENS_CHECKSUM,
            expected_direction=CONFIRMATION_DIRECTION,
        )
        _property = PROPERTY_FAMILIES["animal_sound"]
        # Resolve the direct-answer tokens *before* anything is spent.
        #
        # This is the answer *surface* ("meow"), never a swap concept, so it is
        # resolved through resolve_answer_readout_token rather than
        # resolve_concept_token: the swap's single-token requirement exists
        # because a multi-token concept has no well-defined two-coordinate
        # subspace, which is not what the direct-answer control is building. A
        # multi-token answer falls back to its first token -- the same
        # FIRST-TOKEN-ONLY convention jlens.mmpilot.convergence already uses
        # for readout scoring, labelled the same way -- and the fallback is
        # recorded on every artifact rather than hidden. cat/cow themselves
        # (the swap concepts) still go through resolve_concept_token below and
        # still refuse outright if either is ever multi-token.
        _answer_tokens = {}
        _answer_token_resolution = {}
        for _src, _tgt in RECRUITED_EXPLORATORY_DIRECTIONS:
            _answer = _property.answer_for(_tgt).answer
            _answer_tokens[_tgt] = resolve_answer_readout_token(
                BACKEND.encode_candidate, _answer
            )
            _answer_token_resolution[_tgt] = _answer_tokens[_tgt].to_dict()
            if _answer_tokens[_tgt].variant.startswith("first:"):
                print(
                    f"direct-answer control for {_tgt!r} ({_answer!r}) is not "
                    "single-token under this tokenizer; using its first token "
                    f"{_answer_tokens[_tgt].token_id} "
                    f"({ANSWER_READOUT_FIRST_TOKEN})"
                )
        _rescue_config = {
            "study": "multimodal_new_property_recruited_exploratory_corrected.v3",
            "supersedes_report_checksum": EXPECTED_RECRUITED_EXPLORATORY_REPORT_CHECKSUM,
            "source_file_sha256": EXPECTED_RECRUITED_SOURCE_FILE_SHA256,
            "source_audit_digest": EXPECTED_RECRUITED_SOURCE_AUDIT_DIGEST,
            "audio_linkage_digest": AUDIO_LINKAGE_REPORT["audit_digest"],
            "recruitment_digest": _recruitment["selection_digest"],
            "model_repo_id": MODEL_REPO_ID,
            "model_revision": MODEL_REVISION,
            "audio_protocol_fingerprint": AUDIO_PROTOCOL_FINGERPRINT,
            "manifest_checksum": MANIFEST_CHECKSUM,
            "lens_checksum": EXPECTED_BROAD_POOLED_LENS_CHECKSUM,
            "lens_refitted": False,
            "layers": list(BROAD_POOLED_BAND),
            "alpha": 1.0,
            "alpha_sweep": False,
            "positions": "every original prompt position",
            "prompt_id": "identity_explicit_v1",
            "directions": [list(pair) for pair in RECRUITED_EXPLORATORY_DIRECTIONS],
            "conditions": list(CORRECTED_EXPLORATORY_CONDITIONS),
            "images_per_direction": RECRUITED_EXPLORATORY_IMAGES_PER_DIRECTION,
            "max_new_tokens": NEW_PROPERTY_MAX_NEW_TOKENS,
            "seed": RECRUITED_EXPLORATORY_SEED,
            # --- the instrument, bound into the fingerprint
            "model_dtype_realization_policy": MODEL_DTYPE_REALIZATION.to_dict(),
            "model_dtype_realization_policy_digest": realization_policy_digest(),
            "post_cast_relative_tolerance": POST_CAST_TOLERANCE,
            "integrity_clauses_enforced": list(INTEGRITY_CLAUSES),
            "positive_control": {
                "kind": "norm_matched_direct_answer",
                "strength_match": "cumulative_band_displacement_l2",
                "cumulative_displacement_match_tolerance": POST_CAST_TOLERANCE,
                "implementation": (
                    "jlens.mmpilot.workspace_replication."
                    "unrestricted_greedy_direct_answer_trial"
                ),
                "answer_tokens": {
                    concept: token.token_id
                    for concept, token in sorted(_answer_tokens.items())
                },
                # A multi-token answer surface falls back to its first token
                # rather than refusing, unlike the swap concepts. Recorded per
                # concept so every artifact states whether an exact answer
                # token or a first-token diagnostic backs its control.
                "answer_token_resolution": dict(sorted(
                    _answer_token_resolution.items()
                )),
                "first_token_fallback_label": ANSWER_READOUT_FIRST_TOKEN,
                "alpha": 1.0,
                "can_produce_a_go": False,
                "rule_frozen_before_outcomes": True,
            },
            "thresholds": {
                "min_success_rate": NEW_PROPERTY_DEV_MIN_SUCCESS_RATE,
                "min_control_margin": NEW_PROPERTY_DEV_MIN_CONTROL_MARGIN,
            },
            "outcome_informed_stage_design": True,
            "is_confirmation": False,
            "commit": COMMIT,
        }
        _rescue_digest = payload_checksum(_rescue_config)
        CORRECTED_EXPLORATORY_RUN_DIR = (
            RUNS_ROOT / "mmnewpropertycorrected" /
            f"mmnewpropertycorrected_real_{_rescue_digest.split(':')[1][:12]}"
        )
        CORRECTED_EXPLORATORY_RUN_DIR.mkdir(parents=True, exist_ok=True)
        (CORRECTED_EXPLORATORY_RUN_DIR / "scientific_config.json").write_text(
            json.dumps(_rescue_config, indent=2)
        )
        (CORRECTED_EXPLORATORY_RUN_DIR / "recruitment.json").write_text(
            json.dumps({
                key: value for key, value in _recruitment.items()
                if key != "groups"
            }, indent=2)
        )
        _fingerprint = RunFingerprint(
            mode="real", model_repo_id=MODEL_REPO_ID,
            model_revision=MODEL_REVISION, processor_revision=MODEL_REVISION,
            layers=tuple(BROAD_POOLED_BAND),
            lens_checksum=EXPECTED_BROAD_POOLED_LENS_CHECKSUM,
            manifest_checksum=MANIFEST_CHECKSUM,
            split_id=_recruitment["selection_digest"],
            intervention_config={
                "alpha": 1.0,
                "directions": [list(pair) for pair in RECRUITED_EXPLORATORY_DIRECTIONS],
                "conditions": list(CORRECTED_EXPLORATORY_CONDITIONS),
                "positions": "all_original_prompt_positions",
                "max_new_tokens": NEW_PROPERTY_MAX_NEW_TOKENS,
                # A changed realization policy is a changed instrument, so it
                # invalidates every stored unit rather than silently mixing
                # corrected and uncorrected trials in one report.
                "model_dtype_realization_policy_digest": realization_policy_digest(),
                "positive_control": _rescue_config["positive_control"],
            },
            extra={"study_digest": _rescue_digest},
        )
        _rescue_store = UnitStore(CORRECTED_EXPLORATORY_RUN_DIR, _fingerprint)
        print("corrected exploratory run state", _rescue_store.open())

        _lens = JacobianLens.load(_source_pin["lens_path"])
        _tokens = {
            name: resolve_concept_token(BACKEND.encode_candidate, name)
            for name in (*RECRUITED_EXPLORATORY_CONCEPTS, *BROAD_POOLED_CONTROLS)
        }
        _unembed = BACKEND.unembedding_weight()
        _unrelated_bases = build_swap_bases_for_lens(
            _lens, _unembed, layers=BROAD_POOLED_BAND,
            source=_tokens[BROAD_POOLED_CONTROLS[0]],
            target=_tokens[BROAD_POOLED_CONTROLS[1]],
        )
        _rescue_rows = []
        _n_conditions = (
            len(RECRUITED_EXPLORATORY_DIRECTIONS)
            * RECRUITED_EXPLORATORY_IMAGES_PER_DIRECTION
            * 3 * len(CORRECTED_EXPLORATORY_CONDITIONS)
        )
        for _src, _tgt in RECRUITED_EXPLORATORY_DIRECTIONS:
            _target_answer = _property.answer_for(_tgt)
            _exact_bases = build_swap_bases_for_lens(
                _lens, _unembed, layers=BROAD_POOLED_BAND,
                source=_tokens[_src], target=_tokens[_tgt],
            )
            _random_bases = {
                layer: random_two_direction_basis(basis, seed=20260824 + layer)
                for layer, basis in _exact_bases.items()
            }
            # The positive control's direction is the *answer* token's own lens
            # vector at each layer, the same construction the bases use for a
            # concept token. Its magnitude is set inside the trial to the exact
            # swap's per-position update norm, so it is not free steering.
            _answer_vectors = {
                int(layer): selected_lens_vector(
                    _lens, _unembed, layer=int(layer),
                    token_id=_answer_tokens[_tgt].token_id,
                )
                for layer in BROAD_POOLED_BAND
            }
            _conditions = (
                ("exact", 1.0, _exact_bases),
                ("zero", 0.0, _exact_bases),
                ("random", 1.0, _random_bases),
                ("unrelated", 1.0, _unrelated_bases),
                ("direct_answer", 1.0, _exact_bases),
            )
            for _group in _recruitment["groups"][_src]:
                for _modality in ("text", "image", "spoken_audio"):
                    for _condition, _alpha, _bases in _conditions:
                        _key = safe_key(
                            "corrected", _src, _tgt, _group["group_id"],
                            _modality, _condition,
                        )
                        _stored = _rescue_store.load("intervention", _key)
                        if _stored is None:
                            _inputs = build_group_inputs(
                                _group, _modality,
                                property_prompt(
                                    "identity_explicit_v1", _modality,
                                    _group["caption"],
                                ),
                            )
                            if _condition == "direct_answer":
                                _trial = unrestricted_greedy_direct_answer_trial(
                                    BACKEND, _inputs, bases=_bases,
                                    answer_vectors=_answer_vectors,
                                    answer=_target_answer.answer,
                                    max_new_tokens=NEW_PROPERTY_MAX_NEW_TOKENS,
                                    realization_policy=MODEL_DTYPE_REALIZATION,
                                    alpha=1.0,
                                )
                                _stored = direct_answer_trial_row(
                                    _trial, group=_group, modality=_modality,
                                    direction=(_src, _tgt),
                                    answer=_target_answer,
                                    layers=BROAD_POOLED_BAND,
                                )
                            else:
                                _trial = unrestricted_greedy_swap_trial(
                                    BACKEND, _inputs, bases=_bases, alpha=_alpha,
                                    answer=_target_answer.answer,
                                    max_new_tokens=NEW_PROPERTY_MAX_NEW_TOKENS,
                                    realization_policy=MODEL_DTYPE_REALIZATION,
                                )
                                _stored = generation_trial_row(
                                    _trial, group=_group, modality=_modality,
                                    condition=_condition, direction=(_src, _tgt),
                                    answer=_target_answer, layers=BROAD_POOLED_BAND,
                                )
                            # one checksum-valid atomic unit per completed
                            # generated condition; a disconnect loses nothing
                            _rescue_store.save("intervention", _key, _stored)
                        _rescue_rows.append(_stored)
                        if len(_rescue_rows) == 1 or len(_rescue_rows) % 60 == 0:
                            print("corrected exploratory conditions",
                                  len(_rescue_rows), "of", _n_conditions)

        CORRECTED_EXPLORATORY_REPORT = corrected_exploratory_verdict(
            _rescue_rows, source_audit=_rescue_source,
            linkage_audit=AUDIO_LINKAGE_REPORT, recruitment=_recruitment,
            superseded_report_checksum=EXPECTED_RECRUITED_EXPLORATORY_REPORT_CHECKSUM,
            layers=BROAD_POOLED_BAND,
            min_success_rate=NEW_PROPERTY_DEV_MIN_SUCCESS_RATE,
            min_control_margin=NEW_PROPERTY_DEV_MIN_CONTROL_MARGIN,
        )
        CORRECTED_EXPLORATORY_REPORT = {
            **CORRECTED_EXPLORATORY_REPORT,
            "scientific_config": _rescue_config,
            "recruitment": {
                key: value for key, value in _recruitment.items()
                if key != "groups"
            },
            "rows": _rescue_rows,
        }
        CORRECTED_EXPLORATORY_REPORT["report_checksum"] = payload_checksum({
            key: value for key, value in CORRECTED_EXPLORATORY_REPORT.items()
            if key != "report_checksum"
        })
        _rescue_store.save(
            "metric", "corrected_new_property_exploratory",
            CORRECTED_EXPLORATORY_REPORT,
        )
        _rescue_path = (
            CORRECTED_EXPLORATORY_RUN_DIR /
            "corrected_new_property_exploratory_report.json"
        )
        _rescue_path.write_text(
            json.dumps(CORRECTED_EXPLORATORY_REPORT, indent=2, default=str)
        )
        print("=" * 96)
        print("CORRECTED RECRUITED EXPLORATORY —",
              CORRECTED_EXPLORATORY_REPORT["verdict"])
        print("=" * 96)
        print("eligible clean-capable groups", _recruitment["eligible_counts"])
        print("answer token resolution      ", _answer_token_resolution)
        print("instrument state             ",
              CORRECTED_EXPLORATORY_REPORT["instrument_state"])
        print("passing directions           ",
              CORRECTED_EXPLORATORY_REPORT["passing_directions"])
        print("failure modes                ",
              CORRECTED_EXPLORATORY_REPORT["failure_modes"])
        for _row in CORRECTED_EXPLORATORY_REPORT["directions"]:
            _control = _row["direct_answer_positive_control"]
            print(f"  {_row['direction']:<12}", _row["instrument_state"])
            for _cell in _row["cells"]:
                print(
                    f"    {_cell['modality']:<13}",
                    f"exact {_cell['exact_successes']}/{_cell['n']}",
                    "controls",
                    {k: v["successes"] for k, v in _cell["controls"].items()},
                    "post-cast ok", _cell["integrity_pass"],
                )
            print("    direct-answer positive control",
                  {k: f"{v['successes']}/{v['n']}"
                   for k, v in _control["by_modality"].items()},
                  "passed", _control["passed"])
        print("realization tolerance        ", POST_CAST_TOLERANCE)
        print("realization policy digest    ", realization_policy_digest())
        print("supersedes (not rewritten)   ",
              EXPECTED_RECRUITED_EXPLORATORY_REPORT_CHECKSUM)
        print("source aggregate verdict     PROPERTY_AUDIT_NO_GO (unchanged)")
        print("label                        exploratory; not confirmation")
        print("report                       ", _rescue_path)
        print("checksum                     ",
              CORRECTED_EXPLORATORY_REPORT["report_checksum"])
elif RUN_STAGE5B01_AUDIO_LINKAGE_AUDIT or RUN_STAGE5B1RC_CORRECTED_EXPLORATORY:
    print("Stage 5B01/5B1RC requested but its required gate is disabled.")
'''
)

markdown(
    r"""
### Stage 5B1A — the read-only historical amendment

CPU-only, no model, no lens, zero forwards. Everything below is written *beside*
the completed artifacts. `scientific_recompute` is `0`, the original verdict
strings are reproduced verbatim, and the original report's bytes are re-hashed
after the amendment is written so the printout proves nothing was touched.

The second half audits the confirmed leg-count result
(`FRESH_MULTIMODAL_CONFIRMATION_GO`). That study ran a different trial function,
`unrestricted_swap_trial`, whose enforced gate was `1e-5` on
`max_coordinate_update_error` and `max_orthogonal_residual_drift` — far tighter
than 0.02, but those two fields are the **float64 pre-cast solve** errors, which
are exact by construction and say nothing about what the bf16 residual stream
received. The post-cast quantities were computed inside `swap_coordinates` but
never persisted by that trial function. This stage reports that finding without
reaffirming or invalidating the confirmed result, and names the smallest exact
replication that could settle it.
"""
)
code(
    r'''
if INSTRUMENT_AMENDMENT_ENABLED:
    import hashlib
    import inspect

    from jlens.mmpilot.multimodal_followup import (
        instrument_defect_amendment, legacy_confirmation_realization_audit,
    )
    from jlens.mmpilot.multimodal_instrument import POST_CAST_TOLERANCE
    from jlens.mmpilot.multimodal_lens import unrestricted_swap_trial

    AMENDMENT_RUN_DIR = RUNS_ROOT / "mmamendments"
    AMENDMENT_RUN_DIR.mkdir(parents=True, exist_ok=True)

    _flawed_path = (
        Path(RECRUITED_EXPLORATORY_FLAWED_RUN_DIR) /
        "recruited_new_property_exploratory_report.json"
    )
    _bytes_before = (
        hashlib.sha256(_flawed_path.read_bytes()).hexdigest()
        if _flawed_path.is_file() else None
    )
    INSTRUMENT_AMENDMENT = instrument_defect_amendment(
        original_report_path=str(_flawed_path),
        original_report_checksum=EXPECTED_RECRUITED_EXPLORATORY_REPORT_CHECKSUM,
        original_run_name=Path(RECRUITED_EXPLORATORY_FLAWED_RUN_DIR).name,
        original_verdict="RECRUITED_NEW_PROPERTY_EXPLORATORY_NO_GO",
        omitted_clauses=RECRUITED_EXPLORATORY_OMITTED_CLAUSES,
        observed_post_cast_relative_errors=(
            RECRUITED_EXPLORATORY_OBSERVED_POST_CAST_ERRORS
        ),
        corrected_stage="5B1RC",
    )
    _amendment_path = (
        AMENDMENT_RUN_DIR /
        f"{Path(RECRUITED_EXPLORATORY_FLAWED_RUN_DIR).name}_instrument_amendment.json"
    )
    _amendment_path.write_text(
        json.dumps(INSTRUMENT_AMENDMENT, indent=2, default=str)
    )
    _bytes_after = (
        hashlib.sha256(_flawed_path.read_bytes()).hexdigest()
        if _flawed_path.is_file() else None
    )
    print("=" * 96)
    print("INSTRUMENT AMENDMENT —", INSTRUMENT_AMENDMENT["corrected_classification"])
    print("=" * 96)
    print("original run          ", INSTRUMENT_AMENDMENT["original_run_name"])
    print("original verdict      ", INSTRUMENT_AMENDMENT["original_verdict"],
          "(reproduced verbatim, unchanged)")
    print("pinned report checksum", INSTRUMENT_AMENDMENT["original_report_checksum"])
    print("omitted clauses       ",
          INSTRUMENT_AMENDMENT["omitted_integrity_clauses"])
    print("observed post-cast    ",
          INSTRUMENT_AMENDMENT["observed_post_cast_relative_errors"],
          "against tolerance", POST_CAST_TOLERANCE)
    print("scientific_recompute  ", INSTRUMENT_AMENDMENT["scientific_recompute"])
    print("original file sha256  ", _bytes_before, "->", _bytes_after,
          "(unchanged)" if _bytes_before == _bytes_after else "CHANGED — STOP")
    print("amendment             ", _amendment_path)
    print("checksum              ", INSTRUMENT_AMENDMENT["amendment_checksum"])

    # --- the confirmed leg-count result's own code path
    #
    # The question is what the *completed* run did, not what this code can do
    # now: Stage 3DA's repair added a realization_policy parameter to
    # unrestricted_swap_trial, so introspecting today's signature would answer
    # a different question and answer it wrongly. The evidence used here is the
    # completed run's own stored rows when they are reachable, and the pinned
    # historical fact otherwise.
    _confirmation_path = (
        Path(FRESH_CONFIRMATION_RUN_DIR) / "fresh_multimodal_confirmation_report.json"
    )
    _confirmation_artifact_reachable = _confirmation_path.is_file()
    if _confirmation_artifact_reachable:
        _confirmation_payload = json.loads(_confirmation_path.read_text())
        _confirmation_row_fields = sorted({
            key
            for row in _confirmation_payload.get("rows") or []
            for key in row
        })
        _historical_policy_passed = any(
            key.startswith("max_post_cast")
            or key == "model_dtype_realization_policy_supplied"
            for key in _confirmation_row_fields
        )
    else:
        _confirmation_row_fields = list(LEGACY_CONFIRMATION_STORED_ROW_FIELDS)
        _historical_policy_passed = LEGACY_CONFIRMATION_REALIZATION_POLICY_PASSED
    LEGACY_CONFIRMATION_AUDIT = legacy_confirmation_realization_audit(
        report_checksum=EXPECTED_FRESH_CONFIRMATION_REPORT_CHECKSUM,
        trial_function="jlens.mmpilot.multimodal_lens.unrestricted_swap_trial",
        realization_policy_passed=bool(_historical_policy_passed),
        stored_diagnostic_fields=_confirmation_row_fields,
        enforced_integrity_clauses=list(LEGACY_CONFIRMATION_ENFORCED_CLAUSES),
        enforced_tolerance=1e-5,
    )
    _legacy_path = AMENDMENT_RUN_DIR / "fresh_confirmation_realization_audit.json"
    _legacy_path.write_text(
        json.dumps(LEGACY_CONFIRMATION_AUDIT, indent=2, default=str)
    )
    print()
    print("=" * 96)
    print("LEGACY CONFIRMATION REALIZATION AUDIT —",
          LEGACY_CONFIRMATION_AUDIT["verdict"])
    print("=" * 96)
    print("pinned report checksum   ", LEGACY_CONFIRMATION_AUDIT["report_checksum"])
    print("trial function           ", LEGACY_CONFIRMATION_AUDIT["trial_function"])
    print("evidence source          ",
          "stored rows" if _confirmation_artifact_reachable else "pinned historical fact")
    print("realization policy passed",
          LEGACY_CONFIRMATION_AUDIT["realization_policy_passed"], "(at the time of that run)")
    print("today's trial function    ",
          "accepts realization_policy"
          if "realization_policy" in inspect.signature(unrestricted_swap_trial).parameters
          else "still cannot accept one")
    print("post-cast fields stored  ",
          LEGACY_CONFIRMATION_AUDIT["post_cast_diagnostics_stored"] or "NONE")
    print("post-cast gates enforced ",
          LEGACY_CONFIRMATION_AUDIT["post_cast_diagnostics_enforced"] or "NONE")
    print("enforced tolerance       ",
          LEGACY_CONFIRMATION_AUDIT["enforced_tolerance"],
          "on the float64 pre-cast solve error")
    print("reaffirms original       ",
          LEGACY_CONFIRMATION_AUDIT["reaffirms_original_result"])
    print("invalidates original     ",
          LEGACY_CONFIRMATION_AUDIT["invalidates_original_result"])
    print("scientific_recompute     ",
          LEGACY_CONFIRMATION_AUDIT["scientific_recompute"])
    if LEGACY_CONFIRMATION_AUDIT["required_replication"]:
        print("required replication     ",
              LEGACY_CONFIRMATION_AUDIT["required_replication"]["smallest_sufficient_design"])
    print("audit                    ", _legacy_path)
    print("checksum                 ", LEGACY_CONFIRMATION_AUDIT["audit_checksum"])
elif RUN_STAGE5B1A_INSTRUMENT_AMENDMENT:
    print("Stage 5B1A requested but disabled.")
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
    from jlens.mmpilot.multimodal_instrument import MODEL_DTYPE_REALIZATION
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

    from jlens.mmpilot.multimodal_followup import PropertyAnswer

    _property = PROPERTY_FAMILIES[DESIGN["property_family"]]
    _src, _tgt = DESIGN["direction"]
    # The frozen design's own answer_aliases are authoritative here, not the
    # static PROPERTY_FAMILIES table: for an empirical concept like bird, the
    # declared table still has an empty alias set, and the resolved answer
    # only exists in what Stage 5B2 froze.
    _source_answer = PropertyAnswer(
        concept=_src, answer=(DESIGN["answer_aliases"][_src] or [""])[0],
        aliases=tuple(DESIGN["answer_aliases"][_src]), admissible=True,
        reason="resolved and frozen by Stage 5B2",
    )
    _target_answer = PropertyAnswer(
        concept=_tgt, answer=(DESIGN["answer_aliases"][_tgt] or [""])[0],
        aliases=tuple(DESIGN["answer_aliases"][_tgt]), admissible=True,
        reason="resolved and frozen by Stage 5B2",
    )
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
                    str(DESIGN["prompt_by_modality"][_modality]).format(
                        caption=_group["caption"]
                    ),
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
                            str(DESIGN["prompt_by_modality"][_modality]).format(
                                caption=_group["caption"]
                            ),
                        )
                        _trial = unrestricted_greedy_swap_trial(
                            BACKEND, _inputs, bases=_bases, alpha=_alpha,
                            answer=_target_answer.answer,
                            max_new_tokens=int(DESIGN["max_new_tokens"]),
                            realization_policy=MODEL_DTYPE_REALIZATION,
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

    # Keyed by the mock configuration so that changing the property family or
    # direction lands in a new directory instead of colliding with a stale one
    # and tripping the resume gate.
    from jlens.mmpilot.multimodal_instrument import realization_policy_digest

    _mock_key = payload_checksum({
        "family": NEW_PROPERTY_FAMILY,
        "concepts": list(NEW_PROPERTY_CONCEPTS),
        "directions": [list(pair) for pair in NEW_PROPERTY_DEV_DIRECTIONS],
        "scenarios": list(SCENARIOS),
        # The trial-record schema and the integrity clauses it feeds are part
        # of the mock's configuration; a change to either must land in a new
        # directory rather than resume rows a newer verdict cannot score.
        "realization_policy_digest": realization_policy_digest(),
    }).split(":")[1][:12]
    MOCK_ROOT = RUNS_ROOT / "followup_mock" / _mock_key
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

markdown(
    r"""
## 19. Stages 6A-6E — a frozen, evidence-gated cat->dog animal-sound study (fp32)

The cat/cow animal-sound branch surfaced two problems independent of the
coordinate-swap instrument itself: cow's spoken-audio capability narrowly
missed its own frozen gate, and manual inspection of its 16 recruited
photographs found four of eight cow images compromised -- a distant speck, a
promotional statue, and two frames with an unlabeled competing animal a
caption named but COCO's own object detector missed. Stage 3DA additionally
showed the *previous* instrument's "every trial must converge" gate was too
strict for bf16: the leg-count replay's exact arm converged only 24/48 times
even though those converged trials showed a 92% effect. This section is a
clean restart that fixes both problems at once rather than patching either.

**What changed, concretely:**

* **fp32, not bf16.** Stage 3DA's own diagnostics showed the float64-to-model
  cast is where essentially all realization error comes from (the pre-cast
  solve error was ~4e-12; the bf16 post-cast error ran up to 0.45). fp32's
  24-bit mantissa should make that error negligible. There is no bf16
  fallback anywhere below: `jlens.mmpilot.fp32_preflight` estimates the load
  and refuses cleanly, before any weight is loaded, if the GPU cannot hold it.
* **cat->dog, not cat->cow.** Frozen as the single primary direction before
  any causal outcome opens. `dog` was already a declared, admissible
  `animal_sound` concept (`bark`/`barks`/`woof`), and its evidence pool is
  larger than cow's.
* **A frozen evidence-quality gate**
  (`jlens.mmpilot.evidence_quality`), built entirely from raw COCO instance
  and caption files -- not the lossy category-name-only projection the
  earlier pipeline used -- and applied *before* any capability screening. See
  that module's docstring for the four real compromised photographs it is
  modeled on.
* **Disjoint development and confirmation populations, frozen and persisted
  before the model loads at all** (Stage 6B), not merely excluded
  after the fact from whatever is left.

**Stages, in order (one per session, matching every earlier follow-up):**

| stage | runtime | spends a causal outcome? |
|---|---|---|
| 6A | CPU | no -- reads raw COCO files only |
| 6B | CPU | no -- freezes populations from 6A's output |
| 6C | fp32 GPU | yes -- capability audit + development (exact, controls, direct-answer) |
| 6D | CPU | no -- freezes the confirmation design from 6C's GO |
| 6E | fp32 GPU | yes -- fresh confirmation |

The earlier cat/cow reports (`RECRUITED_NEW_PROPERTY_EXPLORATORY_NO_GO` and its
Stage 5B1A amendment, `CORRECTED_RECRUITED_EXPLORATORY_INSTRUMENT_FAILURE`)
are **not** touched, rewritten, or reused by anything below.
"""
)
code(
    r'''
from jlens.mmpilot.evidence_quality import (
    DEFAULT_THRESHOLDS as CATDOG_EVIDENCE_THRESHOLDS,
)
from jlens.mmpilot.fp32_preflight import (
    estimate_fp32_inference_memory, GEMMA4_E4B_APPROX_PARAM_COUNT,
)

# Every tunable and switch above this point (the scientific target, the
# population sizes, every threshold, the fp32 budget, the six RUN_STAGE6*
# switches, the two CONFIRM_CATDOG_*_BUDGET switches, and every *_RUN_DIR /
# EXPECTED_*_CHECKSUM pin) is declared once, in section 1's configuration
# cell -- not here. FOLLOWUP_STAGES and MODEL_STAGE need the switches before
# this cell ever runs, and a redeclaration in this cell would silently
# discard whatever the config cell's user just set for any of them.
#
# Only the raw COCO annotation paths are computed here: they derive from
# IMAGE_MEDIA_ROOT, which section 3 does not set until after section 1 runs.
CATDOG_COCO_ANNOTATIONS_ROOT = IMAGE_MEDIA_ROOT / "annotations"
CATDOG_COCO_INSTANCES_PATHS = (
    CATDOG_COCO_ANNOTATIONS_ROOT / "instances_train2014.json",
    CATDOG_COCO_ANNOTATIONS_ROOT / "instances_val2014.json",
)
CATDOG_COCO_CAPTIONS_PATHS = (
    CATDOG_COCO_ANNOTATIONS_ROOT / "captions_train2014.json",
    CATDOG_COCO_ANNOTATIONS_ROOT / "captions_val2014.json",
)

if REAL_MODE:
    # Mutual exclusivity across every follow-up stage, Stage 6 included, is
    # already enforced once by FOLLOWUP_STAGES's own sum check in section 1.
    _catdog_model_stage = (
        RUN_STAGE6C_CATDOG_DEVELOPMENT
        or RUN_STAGE6C2_CATDOG_PATH_LOCALIZATION
        or RUN_STAGE6E_CATDOG_CONFIRMATION
    )
    CATDOG_MODEL_ENABLED = bool(_catdog_model_stage and CONFIRM_MODEL_LOAD)
    CATDOG_DEVELOPMENT_ENABLED = bool(
        RUN_STAGE6C_CATDOG_DEVELOPMENT and CATDOG_MODEL_ENABLED
        and CONFIRM_CATDOG_DEVELOPMENT_BUDGET
    )
    CATDOG_CONFIRMATION_ENABLED = bool(
        RUN_STAGE6E_CATDOG_CONFIRMATION and CATDOG_MODEL_ENABLED
        and CONFIRM_CATDOG_CONFIRMATION_BUDGET
    )
    CATDOG_PATH_LOCALIZATION_ENABLED = bool(
        RUN_STAGE6C2_CATDOG_PATH_LOCALIZATION and CATDOG_MODEL_ENABLED
        and CONFIRM_CATDOG_PATH_LOCALIZATION_BUDGET
    )
    if RUN_STAGE6C_CATDOG_DEVELOPMENT and not CATDOG_DEVELOPMENT_ENABLED:
        print("STAGE 6C BLOCKED: confirm CONFIRM_MODEL_LOAD and "
              "CONFIRM_CATDOG_DEVELOPMENT_BUDGET after reading the printed budget")
    if RUN_STAGE6E_CATDOG_CONFIRMATION and not CATDOG_CONFIRMATION_ENABLED:
        print("STAGE 6E BLOCKED: confirm CONFIRM_MODEL_LOAD and "
              "CONFIRM_CATDOG_CONFIRMATION_BUDGET after reading the printed budget")
    if (
        RUN_STAGE6C2_CATDOG_PATH_LOCALIZATION
        and not CATDOG_PATH_LOCALIZATION_ENABLED
    ):
        print("STAGE 6C2 BLOCKED: confirm CONFIRM_MODEL_LOAD and "
              "CONFIRM_CATDOG_PATH_LOCALIZATION_BUDGET after reading the "
              "printed budget")

    _catdog_est = estimate_fp32_inference_memory(
        workspace_fraction=CATDOG_FP32_WORKSPACE_FRACTION,
        safety_margin=CATDOG_FP32_SAFETY_MARGIN,
    )
    print("STAGE 6 CAT->DOG FROZEN STUDY -- CONFIGURATION")
    print("  direction                 ", "->".join(CATDOG_DIRECTION),
          f"({CATDOG_CLEAN_ANSWER} -> {CATDOG_SWAPPED_ANSWER})")
    print("  property / prompt         ", CATDOG_PROPERTY_FAMILY, CATDOG_PROMPT_ID)
    print("  layers / alpha            ", list(BROAD_POOLED_BAND), 1.0)
    print("  model dtype               fp32 (no bf16 fallback)")
    print("  fp32 estimate             ",
          f"{_catdog_est.raw_total_gib:.1f} GiB raw, "
          f"{_catdog_est.required_gib:.1f} GiB required with "
          f"{CATDOG_FP32_SAFETY_MARGIN:.2f}x safety margin")
    print("  params (bf16-checkpoint-derived)", f"{GEMMA4_E4B_APPROX_PARAM_COUNT:,}")
    print("  evidence-quality thresholds", CATDOG_EVIDENCE_THRESHOLDS.to_dict())
    print("  dev / confirm candidates  ",
          CATDOG_N_DEV_CANDIDATES_PER_CONCEPT, "/",
          CATDOG_N_CONFIRM_CANDIDATES_PER_CONCEPT, "per concept")
    print("  dev / confirm recruited   ",
          CATDOG_DEV_IMAGES_PER_DIRECTION, "/", CATDOG_CONFIRM_IMAGES)
    print("  primary outcome rule       exact >=",
          CATDOG_DEV_MIN_SUCCESS_RATE, "in every modality, every control >=",
          CATDOG_DEV_MIN_CONTROL_MARGIN, "below it; direct-answer diagnostic "
           "only, cannot substitute for the primary effect")
    print("  direct-answer strength     cumulative band-displacement L2, tolerance",
          CATDOG_CUMULATIVE_DISPLACEMENT_MATCH_TOLERANCE)
    print("  direct-answer calibration  ",
          CATDOG_DEV_IMAGES_PER_DIRECTION * 3,
          "extra outcome-blind forwards (one per development control trial)")
    from jlens.mmpilot.catdog_localization import frozen_grid_record
    _catdog_path_grid = frozen_grid_record()
    _catdog_path_conditions = CATDOG_DEV_IMAGES_PER_DIRECTION * sum(
        len(_path["applicable_modalities"])
        for _path in _catdog_path_grid["paths"]
    )
    print("  Stage 6C2 localization     ", _catdog_path_conditions,
          "direct-answer conditions; <=",
          _catdog_path_conditions * (CATDOG_MAX_NEW_TOKENS + 1),
          "forwards; no exact-swap outcome is scored")
    print("  zero fitting, zero backward passes in every Stage 6 cell")
'''
)

markdown(
    r"""
### Stage 6A -- the frozen evidence-quality index (CPU only, no model)

Reads the raw ``instances_{split}.json`` / ``captions_{split}.json`` files
directly (bbox area and every caption, not the lossy category-name
projection) and scores every image whose only detected animal is a cat or a
dog against the five frozen criteria in
`jlens.mmpilot.evidence_quality`. Nothing here is informed by any model
output, causal or otherwise.
"""
)
code(
    r'''
CATDOG_EVIDENCE_INDEX = None
if REAL_MODE and RUN_STAGE6A_EVIDENCE_QUALITY_INDEX:
    from jlens.mmpilot.evidence_quality import build_clean_evidence_index

    for _path in (*CATDOG_COCO_INSTANCES_PATHS, *CATDOG_COCO_CAPTIONS_PATHS):
        if not Path(_path).is_file():
            raise RuntimeError(f"missing raw COCO annotation file: {_path}")

    CATDOG_EVIDENCE_INDEX = build_clean_evidence_index(
        CATDOG_COCO_INSTANCES_PATHS, CATDOG_COCO_CAPTIONS_PATHS,
        targets=CATDOG_DIRECTION, thresholds=CATDOG_EVIDENCE_THRESHOLDS,
    )
    _index_config = {
        "study": "catdog_evidence_quality_index.v1",
        "targets": list(CATDOG_DIRECTION),
        "thresholds_digest": CATDOG_EVIDENCE_THRESHOLDS.digest,
        "commit": COMMIT,
    }
    _index_digest = payload_checksum(_index_config)
    CATDOG_EVIDENCE_INDEX_RUN_DIR = (
        RUNS_ROOT / "mmcatdogevidence" /
        f"mmcatdogevidence_real_{_index_digest.split(':')[1][:12]}"
    )
    CATDOG_EVIDENCE_INDEX_RUN_DIR.mkdir(parents=True, exist_ok=True)
    _index_path = CATDOG_EVIDENCE_INDEX_RUN_DIR / "catdog_evidence_quality_index.json"
    _index_path.write_text(
        json.dumps({**CATDOG_EVIDENCE_INDEX, "scientific_config": _index_config},
                   indent=2, default=str)
    )
    print("=" * 96)
    print("CAT->DOG EVIDENCE QUALITY INDEX")
    print("=" * 96)
    print("candidates scored  ", CATDOG_EVIDENCE_INDEX["n_candidates_scored"])
    print("approved            ", CATDOG_EVIDENCE_INDEX["n_approved"])
    print("rejected, by cause  ", CATDOG_EVIDENCE_INDEX["rejected_counts"])
    print("report              ", _index_path)
    print("checksum            ", CATDOG_EVIDENCE_INDEX["index_checksum"])
elif RUN_STAGE6A_EVIDENCE_QUALITY_INDEX:
    print("Stage 6A requested but REAL_MODE is off.")
'''
)

markdown(
    r"""
### Stage 6B -- freeze disjoint development and confirmation populations (CPU only, before any model load)

Intersects Stage 6A's approved photographs with the synchronized manifest
(requiring image, caption, *and* audio all present), then partitions them by
seeded stable hash into disjoint development and confirmation pools.
Disjointness is verified and persisted, not assumed. Nothing below this cell
has seen a model output yet.
"""
)
code(
    r'''
CATDOG_POPULATION_FREEZE = None
if REAL_MODE and RUN_STAGE6B_POPULATION_FREEZE:
    from jlens.mmpilot.evidence_quality import (
        filter_synchronized_groups, freeze_disjoint_populations,
    )

    if CATDOG_EVIDENCE_INDEX_RUN_DIR is None or EXPECTED_CATDOG_EVIDENCE_INDEX_CHECKSUM is None:
        raise RuntimeError(
            "Stage 6B requires CATDOG_EVIDENCE_INDEX_RUN_DIR and its expected "
            "checksum; run Stage 6A first, in a prior session"
        )
    _index_path = (
        Path(CATDOG_EVIDENCE_INDEX_RUN_DIR) / "catdog_evidence_quality_index.json"
    )
    _index_payload = json.loads(_index_path.read_text(encoding="utf-8"))
    _recorded = _index_payload.get("index_checksum")
    _recomputed = payload_checksum({
        k: v for k, v in _index_payload.items()
        if k not in ("index_checksum", "scientific_config")
    })
    if _recorded != EXPECTED_CATDOG_EVIDENCE_INDEX_CHECKSUM or _recomputed != _recorded:
        raise RuntimeError(
            f"evidence index checksum mismatch: recorded={_recorded!r} "
            f"recomputed={_recomputed!r} expected="
            f"{EXPECTED_CATDOG_EVIDENCE_INDEX_CHECKSUM!r}"
        )

    if EXCLUSION_UNIVERSE is None:
        raise RuntimeError(
            "Stage 6B needs EXCLUSION_UNIVERSE (Section 12) so any cat "
            "photograph already spent by the cat/cow study is excluded here "
            "too; ensure RUN_STAGE6B_POPULATION_FREEZE is set before Section "
            "12 runs, or rerun Section 12 first"
        )
    _already_spent = {str(v) for v in EXCLUSION_UNIVERSE["excluded_image_ids"]}
    _clean_groups_by_concept = {}
    for _concept in CATDOG_DIRECTION:
        _synced = filter_synchronized_groups(_index_payload, GROUPS, target=_concept)
        _fresh = [g for g in _synced if str(g["image_id"]) not in _already_spent]
        print("synchronized clean-evidence groups for", _concept, ":", len(_synced),
              "->", len(_fresh), "after excluding", len(_synced) - len(_fresh),
              "already spent by prior studies")
        _clean_groups_by_concept[_concept] = _fresh

    CATDOG_POPULATION_FREEZE = freeze_disjoint_populations(
        _clean_groups_by_concept,
        n_dev_per_concept=CATDOG_N_DEV_CANDIDATES_PER_CONCEPT,
        n_confirm_per_concept=CATDOG_N_CONFIRM_CANDIDATES_PER_CONCEPT,
        seed=CATDOG_SEED,
    )
    _freeze_config = {
        "study": "catdog_disjoint_population_freeze.v1",
        "evidence_index_checksum": EXPECTED_CATDOG_EVIDENCE_INDEX_CHECKSUM,
        "manifest_checksum": MANIFEST_CHECKSUM,
        "seed": CATDOG_SEED,
        "commit": COMMIT,
    }
    _freeze_run_digest = payload_checksum(_freeze_config)
    CATDOG_POPULATION_FREEZE_RUN_DIR = (
        RUNS_ROOT / "mmcatdogfreeze" /
        f"mmcatdogfreeze_real_{_freeze_run_digest.split(':')[1][:12]}"
    )
    CATDOG_POPULATION_FREEZE_RUN_DIR.mkdir(parents=True, exist_ok=True)
    _freeze_path = (
        CATDOG_POPULATION_FREEZE_RUN_DIR / "catdog_population_freeze.json"
    )
    _freeze_path.write_text(
        json.dumps({**CATDOG_POPULATION_FREEZE, "scientific_config": _freeze_config},
                   indent=2, default=str)
    )
    print("=" * 96)
    print("CAT->DOG DISJOINT POPULATION FREEZE")
    print("=" * 96)
    print("development photographs  ", CATDOG_POPULATION_FREEZE["n_development"])
    print("confirmation photographs ", CATDOG_POPULATION_FREEZE["n_confirmation"])
    print("disjoint                 ", CATDOG_POPULATION_FREEZE["disjoint"])
    print("frozen before model load ",
          CATDOG_POPULATION_FREEZE["frozen_before_model_load"])
    print("report                   ", _freeze_path)
    print("checksum                 ", CATDOG_POPULATION_FREEZE["freeze_digest"])
elif RUN_STAGE6B_POPULATION_FREEZE:
    print("Stage 6B requested but REAL_MODE is off.")
'''
)

markdown(
    r"""
### Stage 6C -- fp32 preflight, capability audit, and development

The fp32 preflight runs **before** `build_real_backend` is ever called.
Capability screening uses `recruit_all_modality_capable_groups`'s existing
rule (the untouched model must answer correctly in text, image *and* spoken
audio, for the same photograph) restricted to the pre-frozen development pool
only. Recruitment never sees a causal outcome. Development then runs the
exact cat->dog exchange plus all three controls plus the norm-matched
direct-answer "bark" positive control, entirely in fp32, saving one
checksum-valid unit per completed condition.
"""
)
code(
    r'''
CATDOG_DEVELOPMENT_REPORT = None
CATDOG_MODEL_LOADED_DTYPE = None
if REAL_MODE and RUN_STAGE6C_CATDOG_DEVELOPMENT:
    if CATDOG_POPULATION_FREEZE_RUN_DIR is None or EXPECTED_CATDOG_POPULATION_FREEZE_DIGEST is None:
        raise RuntimeError(
            "Stage 6C requires CATDOG_POPULATION_FREEZE_RUN_DIR and its "
            "expected digest; run Stage 6B first, in a prior session"
        )
    _freeze_path = (
        Path(CATDOG_POPULATION_FREEZE_RUN_DIR) / "catdog_population_freeze.json"
    )
    _freeze_payload = json.loads(_freeze_path.read_text(encoding="utf-8"))
    _recorded = _freeze_payload.get("freeze_digest")
    _recomputed = payload_checksum({
        k: v for k, v in _freeze_payload.items()
        if k not in ("freeze_digest", "scientific_config", "development", "confirmation")
    })
    if _recorded != EXPECTED_CATDOG_POPULATION_FREEZE_DIGEST or _recomputed != _recorded:
        raise RuntimeError(
            f"population freeze digest mismatch: recorded={_recorded!r} "
            f"recomputed={_recomputed!r} expected="
            f"{EXPECTED_CATDOG_POPULATION_FREEZE_DIGEST!r}"
        )
    if not _freeze_payload.get("disjoint"):
        raise RuntimeError("the frozen populations are not disjoint; refusing")

    import torch

    from jlens.mmpilot.fp32_preflight import preflight_fp32_or_refuse

    CATDOG_FP32_PREFLIGHT = preflight_fp32_or_refuse(
        workspace_fraction=CATDOG_FP32_WORKSPACE_FRACTION,
        safety_margin=CATDOG_FP32_SAFETY_MARGIN,
    )
    print("fp32 preflight passed:", CATDOG_FP32_PREFLIGHT["device_name"],
          f"{CATDOG_FP32_PREFLIGHT['free_gib']:.1f} GiB free, "
          f"{CATDOG_FP32_PREFLIGHT['required_gib']:.1f} GiB required")

    from jlens.mmpilot.real_backend import build_real_backend

    _bundle = build_real_backend(
        MODEL_REPO_ID, revision=MODEL_REVISION, allow_model_load=True,
        resolve_audio=True, dtype=torch.float32,
    )
    BACKEND = _bundle.backend
    CATDOG_MODEL_LOADED_DTYPE = _bundle.load_info.get("dtype")
    print("model loaded in dtype   ", CATDOG_MODEL_LOADED_DTYPE)
elif RUN_STAGE6C_CATDOG_DEVELOPMENT:
    print("Stage 6C requested but REAL_MODE is off.")
'''
)

code(
    r'''
if REAL_MODE and RUN_STAGE6C_CATDOG_DEVELOPMENT and CATDOG_MODEL_LOADED_DTYPE is not None:
    _freeze_path = (
        Path(CATDOG_POPULATION_FREEZE_RUN_DIR) / "catdog_population_freeze.json"
    )
    _freeze_payload = json.loads(_freeze_path.read_text(encoding="utf-8"))
    _dev_pool = _freeze_payload["development"]

    from jlens.lens import JacobianLens
    from jlens.mmpilot.coordinate_swap import (
        random_two_direction_basis, resolve_answer_readout_token,
        resolve_concept_token,
    )
    from jlens.mmpilot.multimodal_followup import (
        PROPERTY_FAMILIES, direct_answer_trial_row,
        generation_trial_row, new_property_development_verdict,
        property_answer_matches, property_prompt,
        recruit_all_modality_capable_groups,
    )
    from jlens.mmpilot.multimodal_instrument import (
        INSTRUMENT_VERSION, MODEL_DTYPE_REALIZATION, POST_CAST_TOLERANCE,
    )
    from jlens.mmpilot.multimodal_lens import (
        build_swap_bases_for_lens, load_broad_pooled_development_source,
        selected_lens_vector,
    )
    from jlens.mmpilot.store import RunFingerprint, UnitStore, safe_key
    from jlens.mmpilot.workspace_replication import (
        unrestricted_greedy_direct_answer_trial, unrestricted_greedy_swap_trial,
    )

    _property = PROPERTY_FAMILIES[CATDOG_PROPERTY_FAMILY]
    if CATDOG_CUMULATIVE_DISPLACEMENT_MATCH_TOLERANCE != POST_CAST_TOLERANCE:
        raise RuntimeError(
            "the cumulative band-displacement match must use the unchanged "
            f"frozen 0.02 tolerance, not "
            f"{CATDOG_CUMULATIVE_DISPLACEMENT_MATCH_TOLERANCE}"
        )
    _src, _tgt = CATDOG_DIRECTION
    _clean_answer = _property.answer_for(_src).answer
    _swap_answer = _property.answer_for(_tgt).answer
    if _clean_answer != CATDOG_CLEAN_ANSWER or _swap_answer != CATDOG_SWAPPED_ANSWER:
        raise RuntimeError(
            f"declared answers ({_clean_answer!r} -> {_swap_answer!r}) do not "
            f"match the frozen scientific target ({CATDOG_CLEAN_ANSWER!r} -> "
            f"{CATDOG_SWAPPED_ANSWER!r})"
        )

    _dev_config = {
        "study": "catdog_frozen_animal_sound_development.v1",
        "population_freeze_digest": EXPECTED_CATDOG_POPULATION_FREEZE_DIGEST,
        "model_repo_id": MODEL_REPO_ID, "model_revision": MODEL_REVISION,
        "model_dtype": "float32",
        "audio_protocol_fingerprint": AUDIO_PROTOCOL_FINGERPRINT,
        "manifest_checksum": MANIFEST_CHECKSUM,
        "lens_checksum": EXPECTED_BROAD_POOLED_LENS_CHECKSUM,
        "lens_refitted": False,
        "direction": list(CATDOG_DIRECTION), "alpha": 1.0,
        "layers": list(BROAD_POOLED_BAND),
        "positions": "every original prompt position",
        "prompt_id": CATDOG_PROMPT_ID,
        "conditions": ["exact", "zero", "random", "unrelated", "direct_answer"],
        "instrument_version": INSTRUMENT_VERSION,
        "direct_answer_strength_match": "cumulative_band_displacement_l2",
        "cumulative_displacement_match_tolerance": (
            CATDOG_CUMULATIVE_DISPLACEMENT_MATCH_TOLERANCE
        ),
        "images_per_direction": CATDOG_DEV_IMAGES_PER_DIRECTION,
        "max_new_tokens": CATDOG_MAX_NEW_TOKENS,
        "seed": CATDOG_SEED,
        "model_dtype_realization_policy_digest": (
            payload_checksum(MODEL_DTYPE_REALIZATION.to_dict())
        ),
        "outcome_informed_stage_design": False,
        "is_confirmation": False,
        "commit": COMMIT,
    }
    _dev_digest = payload_checksum(_dev_config)
    CATDOG_DEVELOPMENT_RUN_DIR = (
        RUNS_ROOT / "mmcatdogdev" / f"mmcatdogdev_real_{_dev_digest.split(':')[1][:12]}"
    )
    CATDOG_DEVELOPMENT_RUN_DIR.mkdir(parents=True, exist_ok=True)
    (CATDOG_DEVELOPMENT_RUN_DIR / "scientific_config.json").write_text(
        json.dumps(_dev_config, indent=2)
    )
    _dev_store = UnitStore(
        CATDOG_DEVELOPMENT_RUN_DIR,
        RunFingerprint(
            mode="real", model_repo_id=MODEL_REPO_ID, model_revision=MODEL_REVISION,
            processor_revision=MODEL_REVISION, layers=tuple(BROAD_POOLED_BAND),
            lens_checksum=EXPECTED_BROAD_POOLED_LENS_CHECKSUM,
            manifest_checksum=MANIFEST_CHECKSUM,
            split_id=EXPECTED_CATDOG_POPULATION_FREEZE_DIGEST,
            intervention_config={
                "alpha": 1.0, "direction": list(CATDOG_DIRECTION),
                "conditions": _dev_config["conditions"], "dtype": "float32",
                "positions": "all_original_prompt_positions",
                "max_new_tokens": CATDOG_MAX_NEW_TOKENS,
                "instrument_version": INSTRUMENT_VERSION,
                "direct_answer_strength_match": "cumulative_band_displacement_l2",
                "cumulative_displacement_match_tolerance": (
                    CATDOG_CUMULATIVE_DISPLACEMENT_MATCH_TOLERANCE
                ),
            },
            extra={"study_digest": _dev_digest},
        ),
    )
    print("catdog development run state", _dev_store.open())

    # --- capability: clean, untouched model only, restricted to the frozen
    # development pool. Recruitment never inspects a causal outcome.
    _dev_capability = []
    for _concept in CATDOG_DIRECTION:
        for _group in _dev_pool[_concept]:
            for _modality in ("text", "image", "spoken_audio"):
                _key = safe_key("catdogcap", _group["group_id"], _modality)
                _row = _dev_store.load("capability", _key)
                if _row is None:
                    _answer = _property.answer_for(_concept)
                    _inputs = build_group_inputs(
                        _group, _modality,
                        property_prompt(CATDOG_PROMPT_ID, _modality, _group["caption"]),
                    )
                    from jlens.mmpilot.workspace_replication import (
                        unrestricted_greedy_completion,
                    )
                    _completion = unrestricted_greedy_completion(
                        BACKEND, _inputs, answer=_answer.answer,
                        max_new_tokens=CATDOG_MAX_NEW_TOKENS,
                    )
                    _row = {
                        "concept": _concept, "group_id": _group["group_id"],
                        "image_id": _group["image_id"], "modality": _modality,
                        "generated": _completion["generated_text"],
                        "pass": bool(property_answer_matches(
                            _completion["generated_text"], _answer
                        )),
                    }
                    _dev_store.save("capability", _key, _row)
                _dev_capability.append(_row)

    _dev_recruitment = recruit_all_modality_capable_groups(
        [g for _c in CATDOG_DIRECTION for g in _dev_pool[_c]], _dev_capability,
        concepts=CATDOG_DIRECTION, n_per_concept=CATDOG_DEV_IMAGES_PER_DIRECTION,
    )
    print("dev eligible clean-capable groups", _dev_recruitment["eligible_counts"])
    if not _dev_recruitment["complete"]:
        raise RuntimeError(
            "not enough all-modality clean-capable development photographs: "
            f"{_dev_recruitment['eligible_counts']}"
        )

    _lens = JacobianLens.load(
        load_broad_pooled_development_source(
            BROAD_DEVELOPMENT_RUN_DIR,
            expected_report_checksum=EXPECTED_BROAD_DEVELOPMENT_REPORT_CHECKSUM,
            expected_population_digest=EXPECTED_BROAD_DEVELOPMENT_POPULATION_DIGEST,
            expected_lens_checksum=EXPECTED_BROAD_POOLED_LENS_CHECKSUM,
            expected_direction=CONFIRMATION_DIRECTION,
        )["lens_path"]
    )
    _unembed = BACKEND.unembedding_weight()
    _tokens = {
        name: resolve_concept_token(BACKEND.encode_candidate, name)
        for name in (*CATDOG_DIRECTION, *BROAD_POOLED_CONTROLS)
    }
    _exact_bases = build_swap_bases_for_lens(
        _lens, _unembed, layers=BROAD_POOLED_BAND,
        source=_tokens[_src], target=_tokens[_tgt],
    )
    _random_bases = {
        layer: random_two_direction_basis(basis, seed=20260825 + layer)
        for layer, basis in _exact_bases.items()
    }
    _unrelated_bases = build_swap_bases_for_lens(
        _lens, _unembed, layers=BROAD_POOLED_BAND,
        source=_tokens[BROAD_POOLED_CONTROLS[0]], target=_tokens[BROAD_POOLED_CONTROLS[1]],
    )
    _answer_token = resolve_answer_readout_token(BACKEND.encode_candidate, _swap_answer)
    if _answer_token.variant.startswith("first:"):
        print(f"direct-answer control for {_tgt!r} ({_swap_answer!r}) is not "
              f"single-token; using its first token {_answer_token.token_id}")
    _answer_vectors = {
        int(layer): selected_lens_vector(
            _lens, _unembed, layer=int(layer), token_id=_answer_token.token_id,
        )
        for layer in BROAD_POOLED_BAND
    }

    _conditions = (
        ("exact", 1.0, _exact_bases), ("zero", 0.0, _exact_bases),
        ("random", 1.0, _random_bases), ("unrelated", 1.0, _unrelated_bases),
        ("direct_answer", 1.0, _exact_bases),
    )
    _dev_rows = []
    _n_dev_conditions = CATDOG_DEV_IMAGES_PER_DIRECTION * 3 * len(_conditions)
    for _group in _dev_recruitment["groups"][_src]:
        for _modality in ("text", "image", "spoken_audio"):
            for _condition, _alpha, _bases in _conditions:
                _key = safe_key(
                    "catdogdev", _src, _tgt, _group["group_id"], _modality, _condition,
                )
                _stored = _dev_store.load("intervention", _key)
                if _stored is None:
                    _inputs = build_group_inputs(
                        _group, _modality,
                        property_prompt(CATDOG_PROMPT_ID, _modality, _group["caption"]),
                    )
                    if _condition == "direct_answer":
                        _trial = unrestricted_greedy_direct_answer_trial(
                            BACKEND, _inputs, bases=_bases,
                            answer_vectors=_answer_vectors, answer=_swap_answer,
                            max_new_tokens=CATDOG_MAX_NEW_TOKENS,
                            realization_policy=MODEL_DTYPE_REALIZATION, alpha=1.0,
                        )
                        _stored = direct_answer_trial_row(
                            _trial, group=_group, modality=_modality,
                            direction=(_src, _tgt),
                            answer=_property.answer_for(_tgt),
                            layers=BROAD_POOLED_BAND,
                        )
                    else:
                        _trial = unrestricted_greedy_swap_trial(
                            BACKEND, _inputs, bases=_bases, alpha=_alpha,
                            answer=_swap_answer, max_new_tokens=CATDOG_MAX_NEW_TOKENS,
                            realization_policy=MODEL_DTYPE_REALIZATION,
                        )
                        _stored = generation_trial_row(
                            _trial, group=_group, modality=_modality,
                            condition=_condition, direction=(_src, _tgt),
                            answer=_property.answer_for(_tgt), layers=BROAD_POOLED_BAND,
                        )
                    _dev_store.save("intervention", _key, _stored)
                _dev_rows.append(_stored)
                if len(_dev_rows) % 40 == 0:
                    print("development conditions", len(_dev_rows), "of", _n_dev_conditions)

    # Freshness against every prior study (including the cat/cow photos
    # already spent) was already established in Stage 6B, which intersected
    # the evidence-quality-approved pool against EXCLUSION_UNIVERSE before
    # freezing development and confirmation apart from each other.
    CATDOG_DEVELOPMENT_REPORT = new_property_development_verdict(
        _dev_rows,
        audit={"family": CATDOG_PROPERTY_FAMILY, "audit_digest": _dev_digest},
        layers=BROAD_POOLED_BAND, capability_go=_dev_recruitment["complete"],
        min_success_rate=CATDOG_DEV_MIN_SUCCESS_RATE,
        min_control_margin=CATDOG_DEV_MIN_CONTROL_MARGIN,
        post_cast_tolerance=CATDOG_CUMULATIVE_DISPLACEMENT_MATCH_TOLERANCE,
    )
    CATDOG_DEVELOPMENT_REPORT = {
        **CATDOG_DEVELOPMENT_REPORT, "scientific_config": _dev_config,
        "recruitment": {k: v for k, v in _dev_recruitment.items() if k != "groups"},
        "rows": _dev_rows,
    }
    CATDOG_DEVELOPMENT_REPORT["report_checksum"] = payload_checksum({
        k: v for k, v in CATDOG_DEVELOPMENT_REPORT.items() if k != "report_checksum"
    })
    _dev_store.save("metric", "catdog_development", CATDOG_DEVELOPMENT_REPORT)
    _dev_path = CATDOG_DEVELOPMENT_RUN_DIR / "catdog_development_report.json"
    _dev_path.write_text(json.dumps(CATDOG_DEVELOPMENT_REPORT, indent=2, default=str))
    print("=" * 96)
    print("CAT->DOG DEVELOPMENT --", CATDOG_DEVELOPMENT_REPORT["verdict"])
    print("=" * 96)
    for _row in CATDOG_DEVELOPMENT_REPORT["directions"]:
        print(f"  {_row['direction']:<12}", _row["instrument_state"],
              "passed" if _row["passed"] else "not passed")
        for _cell in _row["cells"]:
            print(f"    {_cell['modality']:<13}",
                  f"exact {_cell['exact_successes']}/{_cell['n']}",
                  "controls", {k: v["successes"] for k, v in _cell["controls"].items()},
                  "integrity", _cell["integrity_pass"])
        _control = _row["direct_answer_positive_control"]
        print("    direct-answer control",
              {k: f"{v['successes']}/{v['n']}" for k, v in _control["by_modality"].items()},
              "passed", _control["passed"])
    print("report   ", _dev_path)
    print("checksum ", CATDOG_DEVELOPMENT_REPORT["report_checksum"])
elif RUN_STAGE6C_CATDOG_DEVELOPMENT:
    print("Stage 6C requested but blocked; confirm its printed budget.")
'''
)

markdown(
    r"""
### Stage 6C2 -- outcome-blind direct-answer path localization (spent data)

The completed fp32 Stage 6C run is checksum-pinned and opened read-only.  Its
exact exchange and its cumulatively matched direct-answer control both scored
0/8 in text, image and spoken audio on L16-L40/all prompt positions, so that
path was **inconclusive**, not a scientific null.

This repair uses only the already-spent eight cat photographs and scores only
the direct ``bark`` positive control over a small grid frozen in
`jlens.mmpilot.catdog_localization`.  The exact cat-to-dog generations are
never rerun, loaded into the localization table, or used for selection.  A GO
here is instrument development only: the selected path must next face the real
alpha=1 exchange on different development photographs before confirmation.
Every condition is an atomic resume unit; no lens is fitted.
"""
)
code(
    r'''
CATDOG_PATH_LOCALIZATION_SOURCE = None
CATDOG_PATH_LOCALIZATION_REPORT = None
CATDOG_PATH_MODEL_LOADED_DTYPE = None
if REAL_MODE and RUN_STAGE6C2_CATDOG_PATH_LOCALIZATION:
    from jlens.mmpilot.catdog_localization import (
        frozen_grid_record, verify_inconclusive_source_report,
    )

    _source_path = (
        Path(CATDOG_INCONCLUSIVE_DEVELOPMENT_RUN_DIR) /
        "catdog_development_report.json"
    )
    if not _source_path.is_file():
        raise RuntimeError(f"missing completed catdog development report: {_source_path}")
    _source_payload = json.loads(_source_path.read_text(encoding="utf-8"))
    CATDOG_PATH_LOCALIZATION_SOURCE = verify_inconclusive_source_report(
        _source_payload,
        expected_checksum=EXPECTED_CATDOG_INCONCLUSIVE_DEVELOPMENT_CHECKSUM,
        expected_model_revision=MODEL_REVISION,
        expected_lens_checksum=EXPECTED_BROAD_POOLED_LENS_CHECKSUM,
    )
    CATDOG_PATH_LOCALIZATION_GRID = frozen_grid_record()
    print("CAT->DOG PATH LOCALIZATION -- SOURCE VERIFIED")
    print("  source report ", _source_path)
    print("  checksum      ", CATDOG_PATH_LOCALIZATION_SOURCE["report_checksum"])
    print("  spent groups  ", len(CATDOG_PATH_LOCALIZATION_SOURCE["group_ids"]))
    print("  grid digest   ", CATDOG_PATH_LOCALIZATION_GRID["grid_digest"])
    print("  bands         ", CATDOG_PATH_LOCALIZATION_GRID["bands"])
    print("  policies      ", CATDOG_PATH_LOCALIZATION_GRID["position_rules"])
    print("  exact outputs used for selection: False")
elif RUN_STAGE6C2_CATDOG_PATH_LOCALIZATION:
    print("Stage 6C2 requested but REAL_MODE is off.")
'''
)

code(
    r'''
if (
    REAL_MODE
    and RUN_STAGE6C2_CATDOG_PATH_LOCALIZATION
    and CATDOG_PATH_LOCALIZATION_ENABLED
):
    import torch
    from jlens.mmpilot.fp32_preflight import preflight_fp32_or_refuse

    CATDOG_PATH_FP32_PREFLIGHT = preflight_fp32_or_refuse(
        workspace_fraction=CATDOG_FP32_WORKSPACE_FRACTION,
        safety_margin=CATDOG_FP32_SAFETY_MARGIN,
    )
    print("fp32 preflight passed:", CATDOG_PATH_FP32_PREFLIGHT["device_name"],
          f"{CATDOG_PATH_FP32_PREFLIGHT['free_gib']:.1f} GiB free, "
          f"{CATDOG_PATH_FP32_PREFLIGHT['required_gib']:.1f} GiB required")

    from jlens.mmpilot.real_backend import build_real_backend

    _bundle = build_real_backend(
        MODEL_REPO_ID, revision=MODEL_REVISION, allow_model_load=True,
        resolve_audio=True, dtype=torch.float32,
    )
    BACKEND = _bundle.backend
    CATDOG_PATH_MODEL_LOADED_DTYPE = _bundle.load_info.get("dtype")
    print("model loaded in dtype", CATDOG_PATH_MODEL_LOADED_DTYPE)
elif RUN_STAGE6C2_CATDOG_PATH_LOCALIZATION:
    print("Stage 6C2 is pinned but blocked; confirm its printed budget.")
'''
)

code(
    r'''
if (
    REAL_MODE
    and RUN_STAGE6C2_CATDOG_PATH_LOCALIZATION
    and CATDOG_PATH_LOCALIZATION_ENABLED
    and CATDOG_PATH_MODEL_LOADED_DTYPE is not None
):
    from jlens.lens import JacobianLens
    from jlens.mmpilot.catdog_localization import (
        CATDOG_PATH_POSITION_POLICIES, summarize_path_localization,
    )
    from jlens.mmpilot.coordinate_swap import (
        resolve_answer_readout_token, resolve_concept_token,
    )
    from jlens.mmpilot.multimodal_followup import (
        PROPERTY_FAMILIES, direct_answer_trial_row, property_prompt,
    )
    from jlens.mmpilot.multimodal_instrument import (
        INSTRUMENT_VERSION, MODEL_DTYPE_REALIZATION, POST_CAST_TOLERANCE,
    )
    from jlens.mmpilot.multimodal_lens import (
        build_swap_bases_for_lens, load_broad_pooled_development_source,
        selected_lens_vector,
    )
    from jlens.mmpilot.store import RunFingerprint, UnitStore, safe_key
    from jlens.mmpilot.workspace_replication import (
        unrestricted_greedy_direct_answer_trial,
    )

    _source_group_ids = set(CATDOG_PATH_LOCALIZATION_SOURCE["group_ids"])
    _group_by_id = {str(group["group_id"]): group for group in GROUPS}
    _missing_groups = sorted(_source_group_ids - set(_group_by_id))
    if _missing_groups:
        raise RuntimeError(
            "the current synchronized manifest cannot reconstruct the pinned "
            f"spent development groups: {_missing_groups}"
        )
    _localization_groups = [
        _group_by_id[group_id]
        for group_id in CATDOG_PATH_LOCALIZATION_SOURCE["group_ids"]
    ]

    _localization_config = {
        "study": "catdog_direct_answer_path_localization.v1",
        "source_report_checksum": (
            EXPECTED_CATDOG_INCONCLUSIVE_DEVELOPMENT_CHECKSUM
        ),
        "source_run_dir": str(CATDOG_INCONCLUSIVE_DEVELOPMENT_RUN_DIR),
        "source_groups": list(CATDOG_PATH_LOCALIZATION_SOURCE["group_ids"]),
        "grid": CATDOG_PATH_LOCALIZATION_GRID,
        "minimum_success_rate": CATDOG_PATH_LOCALIZATION_MIN_SUCCESS_RATE,
        "model_repo_id": MODEL_REPO_ID,
        "model_revision": MODEL_REVISION,
        "model_dtype": "float32",
        "audio_protocol_fingerprint": AUDIO_PROTOCOL_FINGERPRINT,
        "manifest_checksum": MANIFEST_CHECKSUM,
        "lens_checksum": EXPECTED_BROAD_POOLED_LENS_CHECKSUM,
        "lens_refitted": False,
        "direction": list(CATDOG_DIRECTION),
        "answer": CATDOG_SWAPPED_ANSWER,
        "alpha": 1.0,
        "conditions": ["direct_answer"],
        "exact_exchange_outcomes_used_for_selection": False,
        "population_status": "already_spent_development_only",
        "instrument_version": INSTRUMENT_VERSION,
        "post_cast_tolerance": POST_CAST_TOLERANCE,
        "max_new_tokens": CATDOG_MAX_NEW_TOKENS,
        "is_confirmation": False,
        "can_establish_catdog_causal_transfer": False,
        "commit": COMMIT,
    }
    _localization_digest = payload_checksum(_localization_config)
    CATDOG_PATH_LOCALIZATION_RUN_DIR = (
        RUNS_ROOT / "mmcatdogloc" /
        f"mmcatdogloc_real_{_localization_digest.split(':')[1][:12]}"
    )
    CATDOG_PATH_LOCALIZATION_RUN_DIR.mkdir(parents=True, exist_ok=True)
    (CATDOG_PATH_LOCALIZATION_RUN_DIR / "scientific_config.json").write_text(
        json.dumps(_localization_config, indent=2), encoding="utf-8"
    )
    (CATDOG_PATH_LOCALIZATION_RUN_DIR / "frozen_path_grid.json").write_text(
        json.dumps(CATDOG_PATH_LOCALIZATION_GRID, indent=2), encoding="utf-8"
    )
    _localization_store = UnitStore(
        CATDOG_PATH_LOCALIZATION_RUN_DIR,
        RunFingerprint(
            mode="real", model_repo_id=MODEL_REPO_ID,
            model_revision=MODEL_REVISION, processor_revision=MODEL_REVISION,
            layers=tuple(BROAD_POOLED_BAND),
            lens_checksum=EXPECTED_BROAD_POOLED_LENS_CHECKSUM,
            manifest_checksum=MANIFEST_CHECKSUM,
            split_id=EXPECTED_CATDOG_INCONCLUSIVE_DEVELOPMENT_CHECKSUM,
            intervention_config={
                "alpha": 1.0,
                "direction": list(CATDOG_DIRECTION),
                "condition": "direct_answer_only",
                "dtype": "float32",
                "grid_digest": CATDOG_PATH_LOCALIZATION_GRID["grid_digest"],
                "max_new_tokens": CATDOG_MAX_NEW_TOKENS,
                "instrument_version": INSTRUMENT_VERSION,
                "post_cast_tolerance": POST_CAST_TOLERANCE,
            },
            extra={"study_digest": _localization_digest},
        ),
    )
    print("catdog path-localization run state", _localization_store.open())

    _lens = JacobianLens.load(
        load_broad_pooled_development_source(
            BROAD_DEVELOPMENT_RUN_DIR,
            expected_report_checksum=EXPECTED_BROAD_DEVELOPMENT_REPORT_CHECKSUM,
            expected_population_digest=EXPECTED_BROAD_DEVELOPMENT_POPULATION_DIGEST,
            expected_lens_checksum=EXPECTED_BROAD_POOLED_LENS_CHECKSUM,
            expected_direction=CONFIRMATION_DIRECTION,
        )["lens_path"]
    )
    _unembed = BACKEND.unembedding_weight()
    _src, _tgt = CATDOG_DIRECTION
    _tokens = {
        name: resolve_concept_token(BACKEND.encode_candidate, name)
        for name in CATDOG_DIRECTION
    }
    _answer_token = resolve_answer_readout_token(
        BACKEND.encode_candidate, CATDOG_SWAPPED_ANSWER
    )
    _target_answer = PROPERTY_FAMILIES[CATDOG_PROPERTY_FAMILY].answer_for(_tgt)

    _bases_by_band = {}
    _answers_by_band = {}
    for _band_list in CATDOG_PATH_LOCALIZATION_GRID["bands"]:
        _band = tuple(map(int, _band_list))
        _bases_by_band[_band] = build_swap_bases_for_lens(
            _lens, _unembed, layers=_band,
            source=_tokens[_src], target=_tokens[_tgt],
        )
        _answers_by_band[_band] = {
            layer: selected_lens_vector(
                _lens, _unembed, layer=layer, token_id=_answer_token.token_id,
            )
            for layer in _band
        }

    _localization_rows = []
    # Cells the policy leaves undefined are never run, so they are not counted.
    _total = len(_localization_groups) * sum(
        len(_path["applicable_modalities"])
        for _path in CATDOG_PATH_LOCALIZATION_GRID["paths"]
    )
    _computed = 0
    _reused = 0
    for _group in _localization_groups:
        for _modality in ("text", "image", "spoken_audio"):
            for _band_list in CATDOG_PATH_LOCALIZATION_GRID["bands"]:
                _band = tuple(map(int, _band_list))
                _band_name = f"L{_band[0]}_L{_band[-1]}"
                for _policy in CATDOG_PATH_LOCALIZATION_GRID["position_rules"]:
                    _applied_rule = CATDOG_PATH_POSITION_POLICIES[_policy][_modality]
                    if _applied_rule is None:
                        # Undefined cell, not a skipped one: a text prompt has
                        # no distinct evidence token span. Running anything
                        # here would mean substituting a different rule under
                        # this policy's name.
                        continue
                    _key = safe_key(
                        "catdogpath", _group["group_id"], _modality,
                        _band_name, _policy,
                    )
                    _stored = _localization_store.load("intervention", _key)
                    if _stored is None:
                        _inputs = build_group_inputs(
                            _group, _modality,
                            property_prompt(
                                CATDOG_PROMPT_ID, _modality, _group["caption"]
                            ),
                        )
                        _trial = unrestricted_greedy_direct_answer_trial(
                            BACKEND, _inputs, bases=_bases_by_band[_band],
                            answer_vectors=_answers_by_band[_band],
                            answer=CATDOG_SWAPPED_ANSWER,
                            max_new_tokens=CATDOG_MAX_NEW_TOKENS,
                            position_rule=_applied_rule,
                            realization_policy=MODEL_DTYPE_REALIZATION,
                            alpha=1.0,
                        )
                        _stored = direct_answer_trial_row(
                            _trial, group=_group, modality=_modality,
                            direction=CATDOG_DIRECTION, answer=_target_answer,
                            layers=_band,
                        )
                        _stored = {
                            **_stored,
                            "position_rule": _policy,
                            "applied_position_rule": _applied_rule,
                            "source_report_checksum": (
                                EXPECTED_CATDOG_INCONCLUSIVE_DEVELOPMENT_CHECKSUM
                            ),
                            "selection_signal": "direct_answer_only",
                        }
                        _localization_store.save("intervention", _key, _stored)
                        _computed += 1
                    else:
                        _reused += 1
                    _localization_rows.append(_stored)
                    if len(_localization_rows) == 1 or len(_localization_rows) % 40 == 0:
                        print("path localization", len(_localization_rows), "of", _total,
                              "computed", _computed, "reused", _reused)

    CATDOG_PATH_LOCALIZATION_REPORT = summarize_path_localization(
        _localization_rows,
        source_report_checksum=EXPECTED_CATDOG_INCONCLUSIVE_DEVELOPMENT_CHECKSUM,
        grid=CATDOG_PATH_LOCALIZATION_GRID,
        expected_group_ids=CATDOG_PATH_LOCALIZATION_SOURCE["group_ids"],
        minimum_success_rate=CATDOG_PATH_LOCALIZATION_MIN_SUCCESS_RATE,
        post_cast_tolerance=POST_CAST_TOLERANCE,
    )
    CATDOG_PATH_LOCALIZATION_REPORT = {
        **CATDOG_PATH_LOCALIZATION_REPORT,
        "scientific_config": _localization_config,
        "rows": _localization_rows,
    }
    CATDOG_PATH_LOCALIZATION_REPORT["report_checksum"] = payload_checksum({
        key: value for key, value in CATDOG_PATH_LOCALIZATION_REPORT.items()
        if key != "report_checksum"
    })
    _localization_store.save(
        "metric", "catdog_path_localization", CATDOG_PATH_LOCALIZATION_REPORT
    )
    _localization_path = (
        CATDOG_PATH_LOCALIZATION_RUN_DIR /
        "catdog_direct_answer_path_localization_report.json"
    )
    _localization_path.write_text(
        json.dumps(CATDOG_PATH_LOCALIZATION_REPORT, indent=2, default=str),
        encoding="utf-8",
    )
    print("=" * 96)
    print("CAT->DOG DIRECT-ANSWER PATH LOCALIZATION --",
          CATDOG_PATH_LOCALIZATION_REPORT["verdict"])
    print("=" * 96)
    print("selected path", CATDOG_PATH_LOCALIZATION_REPORT["selected_path"])
    print("scientific grade instrument development only")
    print("exact cat->dog outcomes used for selection False")
    print("report  ", _localization_path)
    print("checksum", CATDOG_PATH_LOCALIZATION_REPORT["report_checksum"])
elif RUN_STAGE6C2_CATDOG_PATH_LOCALIZATION:
    print("Stage 6C2 requested but blocked; confirm its printed budget.")
'''
)

markdown(
    r"""
### Stage 6C1 -- read-only amendment of the cumulatively unmatched run (CPU)

Pins the completed Stage 6C report by checksum and writes one amendment beside
it. The report and every stored unit remain untouched; `scientific_recompute`
is zero. This stage records why its `SCIENTIFIC_NULL` is now `INCONCLUSIVE` and
does not predict the corrected rerun's outcome.
"""
)
code(
    r'''
CATDOG_INSTRUMENT_AMENDMENT = None
CATDOG_INSTRUMENT_AMENDMENT_PATH = None
if REAL_MODE and RUN_STAGE6C1_CATDOG_INSTRUMENT_AMENDMENT:
    from jlens.mmpilot.multimodal_followup import (
        direct_answer_matching_defect_amendment,
    )

    _original_path = (
        Path(CATDOG_UNMATCHED_DEVELOPMENT_RUN_DIR) /
        "catdog_development_report.json"
    )
    _original = json.loads(_original_path.read_text(encoding="utf-8"))
    _recorded = _original.get("report_checksum")
    _recomputed = payload_checksum({
        key: value for key, value in _original.items()
        if key != "report_checksum"
    })
    if (
        _recorded != EXPECTED_CATDOG_UNMATCHED_DEVELOPMENT_CHECKSUM
        or _recomputed != _recorded
    ):
        raise RuntimeError(
            "the completed unmatched Stage 6C report failed its checksum pin: "
            f"recorded={_recorded!r} recomputed={_recomputed!r} expected="
            f"{EXPECTED_CATDOG_UNMATCHED_DEVELOPMENT_CHECKSUM!r}"
        )
    _states = list((_original.get("instrument_states") or {}).values())
    if _original.get("verdict") != "NEW_PROPERTY_DEVELOPMENT_NO_GO" or _states != [
        "SCIENTIFIC_NULL"
    ]:
        raise RuntimeError(
            "the pinned report no longer has the classification this amendment "
            f"corrects: verdict={_original.get('verdict')!r}, states={_states!r}"
        )
    _candidate = direct_answer_matching_defect_amendment(
        original_report_path=str(_original_path),
        original_report_checksum=EXPECTED_CATDOG_UNMATCHED_DEVELOPMENT_CHECKSUM,
        original_run_name=Path(CATDOG_UNMATCHED_DEVELOPMENT_RUN_DIR).name,
        original_verdict=_original["verdict"],
        observed_direct_to_exact_ratios={
            "minimum": 2.3, "median": 3.1, "maximum": 4.7,
        },
        n_trials=24,
        corrected_stage="6C",
    )
    CATDOG_INSTRUMENT_AMENDMENT_PATH = (
        Path(CATDOG_UNMATCHED_DEVELOPMENT_RUN_DIR) /
        "catdog_direct_answer_matching_amendment.json"
    )
    if CATDOG_INSTRUMENT_AMENDMENT_PATH.exists():
        CATDOG_INSTRUMENT_AMENDMENT = json.loads(
            CATDOG_INSTRUMENT_AMENDMENT_PATH.read_text(encoding="utf-8")
        )
        _existing_checksum = CATDOG_INSTRUMENT_AMENDMENT.get("amendment_checksum")
        _existing_recomputed = payload_checksum({
            key: value for key, value in CATDOG_INSTRUMENT_AMENDMENT.items()
            if key != "amendment_checksum"
        })
        if (
            _existing_checksum != _existing_recomputed
            or CATDOG_INSTRUMENT_AMENDMENT.get("original_report_checksum")
            != EXPECTED_CATDOG_UNMATCHED_DEVELOPMENT_CHECKSUM
            or CATDOG_INSTRUMENT_AMENDMENT.get(
                "observed_direct_to_exact_cumulative_displacement_ratio"
            ) != _candidate[
                "observed_direct_to_exact_cumulative_displacement_ratio"
            ]
            or CATDOG_INSTRUMENT_AMENDMENT.get("omitted_integrity_clauses")
            != _candidate["omitted_integrity_clauses"]
            or CATDOG_INSTRUMENT_AMENDMENT.get("corrected_classification")
            != "INCONCLUSIVE"
            or CATDOG_INSTRUMENT_AMENDMENT.get("scientific_recompute") != 0
        ):
            raise RuntimeError(
                "an incompatible or corrupt catdog amendment already exists; "
                "refusing to overwrite it"
            )
    else:
        CATDOG_INSTRUMENT_AMENDMENT = _candidate
        CATDOG_INSTRUMENT_AMENDMENT_PATH.write_text(
            json.dumps(CATDOG_INSTRUMENT_AMENDMENT, indent=2), encoding="utf-8"
        )
    print("CAT->DOG COMPLETED RUN AMENDMENT -- INCONCLUSIVE")
    print("scientific recompute", CATDOG_INSTRUMENT_AMENDMENT["scientific_recompute"])
    print("original modified   ", CATDOG_INSTRUMENT_AMENDMENT["original_report_modified"])
    print("path                ", CATDOG_INSTRUMENT_AMENDMENT_PATH)
    print("checksum            ", CATDOG_INSTRUMENT_AMENDMENT["amendment_checksum"])
elif RUN_STAGE6C1_CATDOG_INSTRUMENT_AMENDMENT:
    print("Stage 6C1 requested but REAL_MODE is off.")
'''
)

markdown(
    r"""
### Stage 6D -- freeze the confirmation design (CPU only)

Refuses unless Stage 6C returned `NEW_PROPERTY_DEVELOPMENT_GO` for
`cat->dog` specifically.
"""
)
code(
    r'''
CATDOG_FROZEN_DESIGN = None
if REAL_MODE and RUN_STAGE6D_CATDOG_FREEZE:
    from jlens.mmpilot.multimodal_followup import freeze_new_property_design

    if EXCLUSION_UNIVERSE is None:
        raise RuntimeError(
            "Stage 6D needs EXCLUSION_UNIVERSE (Section 12); ensure "
            "RUN_STAGE6D_CATDOG_FREEZE is set before Section 12 runs"
        )

    if CATDOG_DEVELOPMENT_RUN_DIR is None or EXPECTED_CATDOG_DEVELOPMENT_CHECKSUM is None:
        raise RuntimeError(
            "Stage 6D requires CATDOG_DEVELOPMENT_RUN_DIR and its expected "
            "checksum; run Stage 6C first, in a prior session"
        )
    _dev_path = Path(CATDOG_DEVELOPMENT_RUN_DIR) / "catdog_development_report.json"
    _dev_payload = json.loads(_dev_path.read_text(encoding="utf-8"))
    _recorded = _dev_payload.get("report_checksum")
    _recomputed = payload_checksum({
        k: v for k, v in _dev_payload.items() if k != "report_checksum"
    })
    if _recorded != EXPECTED_CATDOG_DEVELOPMENT_CHECKSUM or _recomputed != _recorded:
        raise RuntimeError(
            f"development report checksum mismatch: recorded={_recorded!r} "
            f"recomputed={_recomputed!r} expected="
            f"{EXPECTED_CATDOG_DEVELOPMENT_CHECKSUM!r}"
        )
    # The direct-answer positive control is diagnostic everywhere else in this
    # codebase -- it can license reading a failed exchange as a null but can
    # never itself gate a GO. This study's frozen design asks more of it: the
    # control must actually have worked, on every modality, for development to
    # be trusted at all. A primary effect that passed while the diagnostic
    # meant to sanity-check the whole causal path failed would be exactly the
    # kind of result that needs the extra scrutiny this refusal forces before
    # any fresh confirmation photograph is opened.
    _cat_dog_direction_name = f"{CATDOG_DIRECTION[0]}->{CATDOG_DIRECTION[1]}"
    _cat_dog_row = next(
        (row for row in _dev_payload["directions"]
         if row["direction"] == _cat_dog_direction_name),
        None,
    )
    if _cat_dog_row is None:
        raise RuntimeError(
            f"the development report has no row for {_cat_dog_direction_name!r}"
        )
    if not _cat_dog_row["direct_answer_positive_control"]["passed"]:
        raise RuntimeError(
            "Stage 6D refuses to freeze: development's own frozen requirement "
            "is that the norm-matched direct-answer control passes, not only "
            "that the exact exchange does. It did not pass here: "
            f"{_cat_dog_row['direct_answer_positive_control']['by_modality']}"
        )

    from jlens.mmpilot.multimodal_followup import (
        PROPERTY_FAMILIES, property_prompt,
    )

    _property = PROPERTY_FAMILIES[CATDOG_PROPERTY_FAMILY]
    _src, _tgt = CATDOG_DIRECTION
    _audit_stub = {
        "family": CATDOG_PROPERTY_FAMILY,
        "audit_digest": _dev_payload["scientific_config"]["population_freeze_digest"],
        "concepts": [
            {"concept": _src, "answer": _property.answer_for(_src).answer,
             "aliases": list(_property.answer_for(_src).aliases),
             "admissible": bool(_property.answer_for(_src).admissible)},
            {"concept": _tgt, "answer": _property.answer_for(_tgt).answer,
             "aliases": list(_property.answer_for(_tgt).aliases),
             "admissible": bool(_property.answer_for(_tgt).admissible)},
        ],
        "prompt_id": CATDOG_PROMPT_ID,
        "question": _property.question,
        "prompt_by_modality": {
            modality: property_prompt(CATDOG_PROMPT_ID, modality, "{caption}")
            for modality in ("text", "image", "spoken_audio")
        },
        "answer_normalization": "casefold_whitespace",
    }
    CATDOG_FROZEN_DESIGN = freeze_new_property_design(
        development=_dev_payload, audit=_audit_stub, direction=CATDOG_DIRECTION,
        lens_checksum=EXPECTED_BROAD_POOLED_LENS_CHECKSUM,
        layers=BROAD_POOLED_BAND, alpha=1.0,
        exclusions=EXCLUSION_UNIVERSE,
        n_candidates=CATDOG_N_CONFIRM_CANDIDATES_PER_CONCEPT,
        n_recruited=CATDOG_CONFIRM_IMAGES,
        min_success_rate=CATDOG_CONFIRM_MIN_SUCCESS_RATE,
        min_control_margin=CATDOG_CONFIRM_MIN_CONTROL_MARGIN,
        min_clean_capability_rate=CATDOG_MIN_CLEAN_CAPABILITY_RATE,
        familywise_alpha=CATDOG_CONFIRM_FAMILYWISE_ALPHA,
        recruitment_rule=(
            "clean property capability in all three modalities, from the "
            "population frozen disjoint from development in Stage 6B"
        ),
        seed=CATDOG_SEED,
    )
    CATDOG_FROZEN_DESIGN_PATH = (
        Path(CATDOG_DEVELOPMENT_RUN_DIR).parent / "catdog_frozen_design.json"
    )
    CATDOG_FROZEN_DESIGN_PATH.write_text(
        json.dumps(CATDOG_FROZEN_DESIGN, indent=2, default=str)
    )
    print("=" * 96)
    print("CAT->DOG DESIGN FROZEN")
    print("=" * 96)
    print("direction        ", "->".join(CATDOG_FROZEN_DESIGN["direction"]))
    print("answer aliases   ", CATDOG_FROZEN_DESIGN["answer_aliases"])
    print("thresholds       ", CATDOG_FROZEN_DESIGN["thresholds"])
    print("path             ", CATDOG_FROZEN_DESIGN_PATH)
    print("digest           ", CATDOG_FROZEN_DESIGN["design_digest"])
elif RUN_STAGE6D_CATDOG_FREEZE:
    print("Stage 6D requested but REAL_MODE is off.")
'''
)

markdown(
    r"""
### Stage 6E -- fresh confirmation (fp32 GPU)

Sources its population exclusively from Stage 6B's pre-frozen confirmation
pool -- never a newly selected photograph, never one from development. No
part of the frozen design changes here.
"""
)
code(
    r'''
CATDOG_CONFIRMATION_REPORT = None
CATDOG_CONFIRM_MODEL_LOADED = False
if REAL_MODE and RUN_STAGE6E_CATDOG_CONFIRMATION:
    if CATDOG_FROZEN_DESIGN_PATH is None:
        raise RuntimeError(
            "Stage 6E cannot open a fresh photograph before Stage 6D wrote "
            "the frozen design; set CATDOG_FROZEN_DESIGN_PATH"
        )
    if CATDOG_POPULATION_FREEZE_RUN_DIR is None or EXPECTED_CATDOG_POPULATION_FREEZE_DIGEST is None:
        raise RuntimeError("Stage 6E requires the Stage 6B population freeze pin")
    _freeze_payload = json.loads(
        (Path(CATDOG_POPULATION_FREEZE_RUN_DIR) / "catdog_population_freeze.json")
        .read_text(encoding="utf-8")
    )
    if payload_checksum({
        k: v for k, v in _freeze_payload.items()
        if k not in ("freeze_digest", "scientific_config", "development", "confirmation")
    }) != EXPECTED_CATDOG_POPULATION_FREEZE_DIGEST:
        raise RuntimeError("population freeze digest mismatch")

    import torch

    from jlens.mmpilot.fp32_preflight import preflight_fp32_or_refuse

    CATDOG_CONFIRM_FP32_PREFLIGHT = preflight_fp32_or_refuse(
        workspace_fraction=CATDOG_FP32_WORKSPACE_FRACTION,
        safety_margin=CATDOG_FP32_SAFETY_MARGIN,
    )
    print("fp32 preflight passed:", CATDOG_CONFIRM_FP32_PREFLIGHT["device_name"])

    from jlens.mmpilot.real_backend import build_real_backend

    _bundle = build_real_backend(
        MODEL_REPO_ID, revision=MODEL_REVISION, allow_model_load=True,
        resolve_audio=True, dtype=torch.float32,
    )
    BACKEND = _bundle.backend
    CATDOG_CONFIRM_MODEL_LOADED = True
    print("model loaded in dtype   ", _bundle.load_info.get("dtype"))
elif RUN_STAGE6E_CATDOG_CONFIRMATION:
    print("Stage 6E requested but REAL_MODE is off.")
'''
)

code(
    r'''
if REAL_MODE and RUN_STAGE6E_CATDOG_CONFIRMATION and CATDOG_CONFIRM_MODEL_LOADED:
    from jlens.mmpilot.multimodal_followup import (
        assert_design_frozen, confirmation_verdict, direct_answer_trial_row,
        generation_trial_row, property_answer_matches,
        recruit_all_modality_capable_groups,
    )

    _design = assert_design_frozen(CATDOG_FROZEN_DESIGN_PATH)
    _freeze_payload = json.loads(
        (Path(CATDOG_POPULATION_FREEZE_RUN_DIR) / "catdog_population_freeze.json")
        .read_text(encoding="utf-8")
    )
    _confirm_pool = _freeze_payload["confirmation"]

    from jlens.lens import JacobianLens
    from jlens.mmpilot.coordinate_swap import (
        random_two_direction_basis, resolve_concept_token,
    )
    from jlens.mmpilot.multimodal_instrument import MODEL_DTYPE_REALIZATION
    from jlens.mmpilot.multimodal_lens import (
        build_swap_bases_for_lens, load_broad_pooled_development_source,
    )
    from jlens.mmpilot.store import RunFingerprint, UnitStore, safe_key
    from jlens.mmpilot.workspace_replication import (
        unrestricted_greedy_completion, unrestricted_greedy_swap_trial,
    )

    _src, _tgt = _design["direction"]
    _confirm_config = {
        "study": "catdog_frozen_animal_sound_confirmation.v1",
        "design_digest": _design["design_digest"],
        "population_freeze_digest": EXPECTED_CATDOG_POPULATION_FREEZE_DIGEST,
        "model_repo_id": MODEL_REPO_ID, "model_revision": MODEL_REVISION,
        "model_dtype": "float32",
        "commit": COMMIT,
    }
    _confirm_digest = payload_checksum(_confirm_config)
    CATDOG_CONFIRMATION_RUN_DIR = (
        RUNS_ROOT / "mmcatdogconfirm" /
        f"mmcatdogconfirm_real_{_confirm_digest.split(':')[1][:12]}"
    )
    CATDOG_CONFIRMATION_RUN_DIR.mkdir(parents=True, exist_ok=True)
    (CATDOG_CONFIRMATION_RUN_DIR / "scientific_config.json").write_text(
        json.dumps(_confirm_config, indent=2)
    )
    _confirm_store = UnitStore(
        CATDOG_CONFIRMATION_RUN_DIR,
        RunFingerprint(
            mode="real", model_repo_id=MODEL_REPO_ID, model_revision=MODEL_REVISION,
            processor_revision=MODEL_REVISION, layers=tuple(_design["layers"]),
            lens_checksum=_design["lens_checksum"], manifest_checksum=MANIFEST_CHECKSUM,
            split_id=_design["design_digest"],
            intervention_config={
                "alpha": _design["alpha"], "conditions": _design["conditions"],
                "dtype": "float32", "positions": "all_original_prompt_positions",
                "max_new_tokens": _design["max_new_tokens"],
            },
            extra={"study_digest": _confirm_digest},
        ),
    )
    print("catdog confirmation run state", _confirm_store.open())

    _confirm_capability = []
    for _concept in (_src, _tgt):
        for _group in _confirm_pool[_concept]:
            _key = safe_key("catdogconfcap", _group["group_id"])
            for _modality in ("text", "image", "spoken_audio"):
                _sub_key = f"{_key}__{_modality}"
                _row = _confirm_store.load("capability", _sub_key)
                if _row is None:
                    _prompt = str(_design["prompt_by_modality"][_modality]).format(
                        caption=_group["caption"]
                    )
                    _inputs = build_group_inputs(_group, _modality, _prompt)
                    _target_word = (
                        _design["answer_aliases"][_src][0]
                        if _concept == _src else _design["answer_aliases"][_tgt][0]
                    )
                    _completion = unrestricted_greedy_completion(
                        BACKEND, _inputs, answer=_target_word,
                        max_new_tokens=int(_design["max_new_tokens"]),
                    )
                    _expected_aliases = (
                        _design["answer_aliases"][_src]
                        if _concept == _src else _design["answer_aliases"][_tgt]
                    )
                    _row = {
                        "concept": _concept, "group_id": _group["group_id"],
                        "image_id": _group["image_id"], "modality": _modality,
                        "generated": _completion["generated_text"],
                        "pass": bool(property_answer_matches(
                            _completion["generated_text"],
                            {"aliases": _expected_aliases},
                        )),
                    }
                    _confirm_store.save("capability", _sub_key, _row)
                _confirm_capability.append(_row)

    _confirm_recruitment = recruit_all_modality_capable_groups(
        [g for _c in (_src, _tgt) for g in _confirm_pool[_c]], _confirm_capability,
        concepts=(_src, _tgt), n_per_concept=CATDOG_CONFIRM_IMAGES,
    )
    print("confirm eligible clean-capable groups", _confirm_recruitment["eligible_counts"])
    _capability_go = bool(_confirm_recruitment["complete"])

    _confirm_rows = []
    if _capability_go:
        _lens = JacobianLens.load(
            load_broad_pooled_development_source(
                BROAD_DEVELOPMENT_RUN_DIR,
                expected_report_checksum=EXPECTED_BROAD_DEVELOPMENT_REPORT_CHECKSUM,
                expected_population_digest=EXPECTED_BROAD_DEVELOPMENT_POPULATION_DIGEST,
                expected_lens_checksum=EXPECTED_BROAD_POOLED_LENS_CHECKSUM,
                expected_direction=CONFIRMATION_DIRECTION,
            )["lens_path"]
        )
        _unembed = BACKEND.unembedding_weight()
        _tokens = {
            name: resolve_concept_token(BACKEND.encode_candidate, name)
            for name in (_src, _tgt, *BROAD_POOLED_CONTROLS)
        }
        _exact_bases = build_swap_bases_for_lens(
            _lens, _unembed, layers=_design["layers"],
            source=_tokens[_src], target=_tokens[_tgt],
        )
        _random_bases = {
            layer: random_two_direction_basis(basis, seed=20260825 + layer)
            for layer, basis in _exact_bases.items()
        }
        _unrelated_bases = build_swap_bases_for_lens(
            _lens, _unembed, layers=_design["layers"],
            source=_tokens[BROAD_POOLED_CONTROLS[0]],
            target=_tokens[BROAD_POOLED_CONTROLS[1]],
        )
        _conditions = (
            ("exact", _design["alpha"], _exact_bases), ("zero", 0.0, _exact_bases),
            ("random", _design["alpha"], _random_bases),
            ("unrelated", _design["alpha"], _unrelated_bases),
        )
        _target_answer_row = {
            "aliases": _design["answer_aliases"][_tgt],
        }
        _n_confirm_conditions = CATDOG_CONFIRM_IMAGES * 3 * len(_conditions)
        for _group in _confirm_recruitment["groups"][_src]:
            for _modality in ("text", "image", "spoken_audio"):
                for _condition, _alpha, _bases in _conditions:
                    _key = safe_key(
                        "catdogconf", _src, _tgt, _group["group_id"], _modality, _condition,
                    )
                    _stored = _confirm_store.load("intervention", _key)
                    if _stored is None:
                        _prompt = str(_design["prompt_by_modality"][_modality]).format(
                            caption=_group["caption"]
                        )
                        _inputs = build_group_inputs(_group, _modality, _prompt)
                        _answer_word = _design["answer_aliases"][_tgt][0]
                        _trial = unrestricted_greedy_swap_trial(
                            BACKEND, _inputs, bases=_bases, alpha=_alpha,
                            answer=_answer_word,
                            max_new_tokens=int(_design["max_new_tokens"]),
                            realization_policy=MODEL_DTYPE_REALIZATION,
                        )
                        _stored = generation_trial_row(
                            _trial, group=_group, modality=_modality,
                            condition=_condition, direction=(_src, _tgt),
                            answer=_target_answer_row, layers=_design["layers"],
                        )
                        _confirm_store.save("intervention", _key, _stored)
                    _confirm_rows.append(_stored)
                    if len(_confirm_rows) % 40 == 0:
                        print("confirmation conditions", len(_confirm_rows),
                              "of", _n_confirm_conditions)

    _confirm_exclusion_audit = {
        "disjoint": True,
        "population_freeze_digest": EXPECTED_CATDOG_POPULATION_FREEZE_DIGEST,
        "development_and_confirmation_share_no_image": True,
    }
    CATDOG_CONFIRMATION_REPORT = confirmation_verdict(
        _confirm_rows, design=_design, capability_go=_capability_go,
        exclusion_audit=_confirm_exclusion_audit,
    )
    CATDOG_CONFIRMATION_REPORT = {
        **CATDOG_CONFIRMATION_REPORT, "scientific_config": _confirm_config,
        "recruitment": {
            k: v for k, v in _confirm_recruitment.items() if k != "groups"
        },
    }
    CATDOG_CONFIRMATION_REPORT["report_checksum"] = payload_checksum({
        k: v for k, v in CATDOG_CONFIRMATION_REPORT.items() if k != "report_checksum"
    })
    _confirm_store.save("metric", "catdog_confirmation", CATDOG_CONFIRMATION_REPORT)
    _confirm_path = CATDOG_CONFIRMATION_RUN_DIR / "catdog_confirmation_report.json"
    _confirm_path.write_text(
        json.dumps(CATDOG_CONFIRMATION_REPORT, indent=2, default=str)
    )
    print("=" * 96)
    print("CAT->DOG FRESH CONFIRMATION --", CATDOG_CONFIRMATION_REPORT["verdict"])
    print("=" * 96)
    for _cell in CATDOG_CONFIRMATION_REPORT["cells"]:
        print(_cell["modality"], f"exact {_cell['exact_successes']}/{_cell['n']}",
              "controls", {k: v["successes"] for k, v in _cell["controls"].items()})
    print("gate     ", CATDOG_CONFIRMATION_REPORT.get("gate"))
    print("report   ", _confirm_path)
    print("checksum ", CATDOG_CONFIRMATION_REPORT["report_checksum"])
elif RUN_STAGE6E_CATDOG_CONFIRMATION:
    print("Stage 6E requested but blocked; confirm its printed budget.")
'''
)

markdown(
    r"""
## 20. Stage 7: no-refit target generalization with distinct answers

This is the final extension of the confirmed bird-to-cat method, not a new
lens search.  The source evidence is always a fresh bird photograph, its
caption, or its spoken caption.  The exact alpha-one exchange targets cat,
ant, or spider under the already confirmed pooled L16-L40 lens.  Their
downstream answers are 4, 6, and 8, so success on a novel target cannot be
explained by merely deleting the bird coordinate.

Stage 7A freezes disjoint development and confirmation photographs on CPU.
Stage 7B checks clean bird capability and runs the cat calibration.

Cat alone cannot carry the claim.  Four is the answer most animals get, so a
perturbation that merely disturbs the bird evidence can land on it by accident,
which is what the pilot saw: in images the exact and the random exchange
produced four on exactly the same five photographs.  Ant and spider are the
repair, because six and eight are not where a disturbed model falls.

Stage 7B2 therefore runs ant and spider on the six already-spent development
photographs, against zero, unrelated and three independently seeded random
controls, checksum-pinned to the completed cat run and recomputing none of it.
It adds one control that costs no forward pass and that the four-legged design
could not express: the exchange toward the *other* identity, rescored against
this one.  A generic push cannot pass it, because passing requires the answer to
follow which identity was inserted.  Direct answer-coordinate leverage is
recorded as a diagnostic and never gates; its only job is to tell a capacity
limit apart from a transfer failure if a target comes back null.

Stage 7C freezes every passing novel target without opening confirmation
outputs.  Stage 7D tests them once on the untouched 22-photograph population:
the first target in the predeclared priority order gates the verdict inside a
Holm family of three modalities by six controls, and any further target is
Holm-corrected in its own family as supporting evidence, so carrying it costs
the primary no familywise power.  No stage fits a lens or searches alpha,
layers, or positions.
"""
)
code(
    r'''
from jlens.mmpilot.leg_count_generalization import frozen_design, novel_frozen_design

LEG_GENERALIZATION_DESIGN = frozen_design()
# The novel-target extension is frozen in its own object.  The completed cat run
# embeds the digest above, and a finished run must not be retroactively rescored
# under a design that changed after it.
NOVEL_LEG_DESIGN = novel_frozen_design()
if NOVEL_LEG_DESIGN["novel_targets"] != ["ant", "spider"]:
    raise RuntimeError("the frozen novel targets changed")
if NOVEL_LEG_DESIGN["target_answers"] != {"ant": "6", "spider": "8"}:
    raise RuntimeError("the frozen novel answers changed")
if tuple(NOVEL_LEG_DESIGN["layers"]) != tuple(BROAD_POOLED_BAND):
    raise RuntimeError("the novel extension must reuse the confirmed band")
LEG_GENERALIZATION_SOURCE = "bird"
LEG_GENERALIZATION_TARGET_ANSWERS = {"cat": "4", "ant": "6", "spider": "8"}
LEG_GENERALIZATION_LAYERS = tuple(range(16, 41))
LEG_GENERALIZATION_ALPHA = 1.0
if LEG_GENERALIZATION_DESIGN["source"] != LEG_GENERALIZATION_SOURCE:
    raise RuntimeError("the frozen Stage 7 source changed")
if LEG_GENERALIZATION_DESIGN["target_answers"] != LEG_GENERALIZATION_TARGET_ANSWERS:
    raise RuntimeError("the frozen Stage 7 target answers changed")
if tuple(LEG_GENERALIZATION_DESIGN["layers"]) != LEG_GENERALIZATION_LAYERS:
    raise RuntimeError("the frozen Stage 7 layer band changed")
if LEG_GENERALIZATION_LAYERS != tuple(BROAD_POOLED_BAND):
    raise RuntimeError("Stage 7 must reuse the confirmed L16-L40 band")
LEG_GENERALIZATION_ROOT = RUNS_ROOT / "mmleggeneralization"
LEG_GENERALIZATION_POPULATION_PATH = (
    LEG_GENERALIZATION_ROOT / "frozen_population.json"
)
LEG_GENERALIZATION_DEVELOPMENT_ROOT = LEG_GENERALIZATION_ROOT / "development"
LEG_GENERALIZATION_DEVELOPMENT_REPORT_PATH = (
    LEG_GENERALIZATION_DEVELOPMENT_ROOT / "leg_count_generalization_development_report.json"
)
LEG_GENERALIZATION_NOVEL_REPORT_PATH = (
    LEG_GENERALIZATION_DEVELOPMENT_ROOT
    / "leg_count_novel_target_development_report.json"
)
# Pin for the completed Stage 7B cat run.  Only the prefix was carried off that
# finished run, so the prefix is what is checked, together with the report's own
# internal self-consistency, its lens and its population digest.  Paste the full
# digest below to harden the pin.  Do not invent the remaining characters: a
# wrong full digest aborts the A100 session on the very check meant to guard it.
EXPECTED_STAGE7B_CAT_REPORT_CHECKSUM_PREFIX = "sha256:30f2246b"
EXPECTED_STAGE7B_CAT_REPORT_CHECKSUM = None
LEG_GENERALIZATION_CONFIRMATION_DESIGN_PATH = (
    LEG_GENERALIZATION_ROOT / "confirmation_design.json"
)
LEG_GENERALIZATION_CONFIRMATION_ROOT = LEG_GENERALIZATION_ROOT / "confirmation"
LEG_GENERALIZATION_CONFIRMATION_REPORT_PATH = (
    LEG_GENERALIZATION_CONFIRMATION_ROOT
    / "fresh_multimodal_leg_count_generalization_report.json"
)

if RUN_STAGE7A_FREEZE_LEG_GENERALIZATION_POPULATION:
    from jlens.mmpilot.multimodal_followup import artifact_exclusion_audit
    from jlens.mmpilot.multimodal_lens import select_causal_groups

    if EXCLUSION_UNIVERSE is None:
        raise RuntimeError("Stage 7A requires the completed exclusion audit")
    _selected = select_causal_groups(
        GROUPS,
        concepts=("bird",),
        n_per_concept=31,
        excluded_image_ids=EXCLUSION_UNIVERSE["excluded_image_ids"],
        seed="multimodal-distinct-leg-count-generalization-20260827-v1",
        forbidden_concepts={"bird": ("cat", "ant", "spider")},
    )["bird"]
    _population_audit = artifact_exclusion_audit(
        _selected, universe=EXCLUSION_UNIVERSE,
        label="leg_count_generalization_frozen_population",
    )
    _population_body = {
        "version": "mmpilot.multimodal_leg_count_population.v1",
        "design_digest": LEG_GENERALIZATION_DESIGN["design_digest"],
        "manifest_checksum": MANIFEST_CHECKSUM,
        "exclusion_digest": EXCLUSION_UNIVERSE["exclusion_digest"],
        "population_audit": _population_audit,
        "selected_before_model_outputs": True,
        "development": [
            {"group_id": str(row["group_id"]), "image_id": str(row["image_id"])}
            for row in _selected[:9]
        ],
        "confirmation": [
            {"group_id": str(row["group_id"]), "image_id": str(row["image_id"])}
            for row in _selected[9:]
        ],
        "fitting_performed": False,
        "backward_passes": 0,
    }
    _population_body["population_digest"] = payload_checksum(_population_body)
    LEG_GENERALIZATION_ROOT.mkdir(parents=True, exist_ok=True)
    if LEG_GENERALIZATION_POPULATION_PATH.is_file():
        _existing = json.loads(
            LEG_GENERALIZATION_POPULATION_PATH.read_text(encoding="utf-8")
        )
        if _existing != _population_body:
            raise RuntimeError("the frozen Stage 7 population changed")
    else:
        LEG_GENERALIZATION_POPULATION_PATH.write_text(
            json.dumps(_population_body, indent=2), encoding="utf-8"
        )
    print("LEG-COUNT GENERALIZATION POPULATION FROZEN")
    print("  development candidates", len(_population_body["development"]))
    print("  confirmation candidates", len(_population_body["confirmation"]))
    print("  overlap", bool(
        {row["image_id"] for row in _population_body["development"]}
        & {row["image_id"] for row in _population_body["confirmation"]}
    ))
    print("  population digest", _population_body["population_digest"])
    print("  model forwards 0; fitting 0; backward passes 0")
    print("STOP THIS CPU RUNTIME. Run Stage 7B alone on an 80 GB A100.")
'''
)

code(
    r'''
def _leg_population_groups(split):
    if not LEG_GENERALIZATION_POPULATION_PATH.is_file():
        raise RuntimeError("run Stage 7A before any Stage 7 model work")
    population = json.loads(
        LEG_GENERALIZATION_POPULATION_PATH.read_text(encoding="utf-8")
    )
    expected = population.get("population_digest")
    body = {key: value for key, value in population.items() if key != "population_digest"}
    if expected != payload_checksum(body):
        raise RuntimeError("the frozen Stage 7 population failed its checksum")
    index = {str(group["group_id"]): group for group in GROUPS}
    rows = []
    for record in population[split]:
        group = index.get(str(record["group_id"]))
        if group is None or str(group["image_id"]) != str(record["image_id"]):
            raise RuntimeError(f"cannot reconstruct frozen group {record}")
        rows.append(group)
    return population, rows


def _leg_integrity(trial, layers):
    return bool(
        trial.get("all_prompt_positions_patched")
        and list(trial.get("layers_patched") or []) == list(layers)
        and trial.get("all_model_dtype_realizations_converged")
        and float(trial.get("max_activation_norm_ratio", 99.0)) <= 1.25
        and float(trial.get("max_update_to_activation_norm_ratio", 99.0)) <= 0.50
    )


def _compact_leg_trial(trial, *, group, modality, target, condition, expected):
    return {
        "group_id": str(group["group_id"]),
        "image_id": str(group["image_id"]),
        "modality": str(modality),
        "source": "bird",
        "target": str(target),
        "expected": str(expected),
        "condition": str(condition),
        "patched_top_token_id": int(trial["patched_top_token_id"]),
        "patched_surface": str(trial["patched_surface"]),
        "success": bool(trial["success"]),
        "integrity_pass": _leg_integrity(trial, BROAD_POOLED_BAND),
        "layers_patched": list(trial["layers_patched"]),
        "all_prompt_positions_patched": bool(trial["all_prompt_positions_patched"]),
        "all_model_dtype_realizations_converged": bool(
            trial.get("all_model_dtype_realizations_converged")
        ),
        "max_activation_norm_ratio": float(trial["max_activation_norm_ratio"]),
        "max_update_to_activation_norm_ratio": float(
            trial["max_update_to_activation_norm_ratio"]
        ),
        "max_post_cast_relative_coordinate_error": float(
            trial.get("max_post_cast_relative_coordinate_error", 0.0)
        ),
    }
'''
)

code(
    r'''
LEG_GENERALIZATION_DEVELOPMENT_REPORT = None
if REAL_MODE and LEG_GENERALIZATION_DEVELOPMENT_ENABLED:
    from jlens.lens import JacobianLens
    from jlens.mmpilot.coordinate_swap import (
        random_two_direction_basis, resolve_concept_token,
    )
    from jlens.mmpilot.leg_count_generalization import (
        CONDITIONS as LEG_CONDITIONS,
        MODALITIES as LEG_MODALITIES,
        NUMBER_WORDS as LEG_NUMBER_WORDS,
        TARGET_ANSWERS as LEG_TARGET_ANSWERS,
        development_report, leg_count_answer_matches,
    )
    from jlens.mmpilot.multimodal_instrument import MODEL_DTYPE_REALIZATION
    from jlens.mmpilot.multimodal_lens import (
        build_swap_bases_for_lens, confirmation_leg_count_prompt,
        load_broad_pooled_development_source,
        unrestricted_swap_trial,
    )
    from jlens.mmpilot.store import RunFingerprint, UnitStore, safe_key

    _population, _development_candidates = _leg_population_groups("development")
    _source_pin = load_broad_pooled_development_source(
        BROAD_DEVELOPMENT_RUN_DIR,
        expected_report_checksum=EXPECTED_BROAD_DEVELOPMENT_REPORT_CHECKSUM,
        expected_population_digest=EXPECTED_BROAD_DEVELOPMENT_POPULATION_DIGEST,
        expected_lens_checksum=EXPECTED_BROAD_POOLED_LENS_CHECKSUM,
        expected_direction=CONFIRMATION_DIRECTION,
    )
    _lens = JacobianLens.load(_source_pin["lens_path"])
    if _lens.source_layers != list(BROAD_POOLED_BAND):
        raise RuntimeError("Stage 7 requires the checksum-pinned L16-L40 lens")
    _development_config = {
        "study": "multimodal_distinct_leg_count_generalization_development.v2",
        "design_digest": LEG_GENERALIZATION_DESIGN["design_digest"],
        "population_digest": _population["population_digest"],
        "lens_checksum": EXPECTED_BROAD_POOLED_LENS_CHECKSUM,
        "model_repo_id": MODEL_REPO_ID,
        "model_revision": MODEL_REVISION,
        "model_dtype": "float32",
        "targets": dict(LEG_TARGET_ANSWERS),
        "layers": list(BROAD_POOLED_BAND),
        "alpha": 1.0,
        "positions": "every_original_prompt_position",
        "answer_matching": "digit_or_english_number_word.v1",
        "answer_basis_surfaces": dict(LEG_NUMBER_WORDS),
        "answer_leverage_role": "reported_diagnostic_not_an_admission_gate",
        "spending_order": "cat_calibration_then_ant_and_spider_if_cat_passes",
        "fresh_confirmation_opened": False,
        "fitting_performed": False,
        "commit": COMMIT,
    }
    _development_digest = payload_checksum(_development_config)
    _development_run_root = (
        LEG_GENERALIZATION_DEVELOPMENT_ROOT
        / f"legdev_real_{_development_digest.removeprefix('sha256:')[:12]}"
    )
    _development_run_root.mkdir(parents=True, exist_ok=True)
    (_development_run_root / "scientific_config.json").write_text(
        json.dumps(_development_config, indent=2), encoding="utf-8"
    )
    _development_store = UnitStore(
        _development_run_root,
        RunFingerprint(
            mode="real", model_repo_id=MODEL_REPO_ID,
            model_revision=MODEL_REVISION, processor_revision=MODEL_REVISION,
            layers=tuple(BROAD_POOLED_BAND),
            lens_checksum=EXPECTED_BROAD_POOLED_LENS_CHECKSUM,
            manifest_checksum=MANIFEST_CHECKSUM,
            split_id=payload_checksum(_population["development"]),
            intervention_config={
                "design_digest": LEG_GENERALIZATION_DESIGN["design_digest"],
                "targets": dict(LEG_TARGET_ANSWERS),
                "conditions": list(LEG_CONDITIONS),
                "alpha": 1.0, "positions": "all_original_prompt_positions",
                "dtype": "float32",
            },
            extra={"study_digest": _development_digest},
        ),
    )
    print("leg-generalization development run", _development_store.open())
    print("lens reused", EXPECTED_BROAD_POOLED_LENS_CHECKSUM)
    print("fitting performed False; backward passes 0")

    _capability_rows = []
    for _group in _development_candidates:
        for _modality in LEG_MODALITIES:
            _key = safe_key("legdevcap", _group["group_id"], _modality)
            _row = _development_store.load("capability", _key)
            if _row is None:
                _inputs = build_group_inputs(
                    _group, _modality,
                    confirmation_leg_count_prompt(_modality, _group["caption"]),
                )
                _logits = BACKEND.forward_logits(_inputs.tensors)[
                    0, _inputs.final_prompt_position
                ].float()
                _surface = BACKEND.decode_token(int(_logits.argmax())).strip()
                _row = {
                    "group_id": str(_group["group_id"]),
                    "image_id": str(_group["image_id"]),
                    "modality": _modality, "expected": "2",
                    "generated": _surface,
                    "pass": leg_count_answer_matches(_surface, "2"),
                }
                _development_store.save("capability", _key, _row)
            _capability_rows.append(_row)
            if len(_capability_rows) == 1 or len(_capability_rows) % 24 == 0:
                print("leg development capability", len(_capability_rows), "of", 27)
    _recruited = []
    for _group in _development_candidates:
        _rows = [
            row for row in _capability_rows
            if row["group_id"] == str(_group["group_id"])
        ]
        if len(_rows) == 3 and all(row["pass"] for row in _rows):
            _recruited.append(_group)
        if len(_recruited) == 6:
            break
    if len(_recruited) != 6:
        raise RuntimeError(
            f"Stage 7 development recruited {len(_recruited)}/6 clean bird groups"
        )
    print("leg development recruited", len(_recruited), "/ 6")

    _unembed = BACKEND.unembedding_weight()
    _concept_names = ("bird", *LEG_TARGET_ANSWERS, *BROAD_POOLED_CONTROLS)
    _concept_tokens = {
        name: resolve_concept_token(BACKEND.encode_candidate, name)
        for name in _concept_names
    }
    _answer_tokens = {
        answer: resolve_concept_token(BACKEND.encode_candidate, word)
        for answer, word in LEG_NUMBER_WORDS.items()
    }
    _concept_bases = {
        target: build_swap_bases_for_lens(
            _lens, _unembed, layers=BROAD_POOLED_BAND,
            source=_concept_tokens["bird"], target=_concept_tokens[target],
        )
        for target in LEG_TARGET_ANSWERS
    }
    _answer_bases = {
        target: build_swap_bases_for_lens(
            _lens, _unembed, layers=BROAD_POOLED_BAND,
            source=_answer_tokens["2"],
            target=_answer_tokens[LEG_TARGET_ANSWERS[target]],
        )
        for target in LEG_TARGET_ANSWERS
    }
    _random_bases = {
        target: {
            layer: random_two_direction_basis(
                basis, seed=20260827 + 1000 * index + layer
            )
            for layer, basis in _concept_bases[target].items()
        }
        for index, target in enumerate(LEG_TARGET_ANSWERS)
    }
    _unrelated_bases = build_swap_bases_for_lens(
        _lens, _unembed, layers=BROAD_POOLED_BAND,
        source=_concept_tokens[BROAD_POOLED_CONTROLS[0]],
        target=_concept_tokens[BROAD_POOLED_CONTROLS[1]],
    )

    def _run_leg_trial(group, modality, target, condition, alpha, bases, prefix):
        key = safe_key(prefix, group["group_id"], modality, target, condition)
        stored = _development_store.load("intervention", key)
        if stored is not None:
            return stored, "reused"
        inputs = build_group_inputs(
            group, modality,
            confirmation_leg_count_prompt(modality, group["caption"]),
        )
        clean_logits = BACKEND.forward_logits(inputs.tensors)[
            0, inputs.final_prompt_position
        ].float()
        trial = unrestricted_swap_trial(
            BACKEND, inputs, bases=bases, alpha=alpha,
            target_token_id=int(_answer_tokens[LEG_TARGET_ANSWERS[target]].token_id),
            source_token_id=int(_answer_tokens["2"].token_id),
            clean_logits=clean_logits, compact_positions=True,
            realization_policy=MODEL_DTYPE_REALIZATION,
        )
        surface = BACKEND.decode_token(int(trial["patched_top_token_id"])).strip()
        trial = {
            **trial, "patched_surface": surface,
            "success": leg_count_answer_matches(
                surface, LEG_TARGET_ANSWERS[target]
            ),
        }
        stored = _compact_leg_trial(
            trial, group=group, modality=modality, target=target,
            condition=condition, expected=LEG_TARGET_ANSWERS[target],
        )
        _development_store.save("intervention", key, stored)
        return stored, "computed"

    _leverage_rows = []
    for _target in LEG_TARGET_ANSWERS:
        for _group in _recruited:
            for _modality in LEG_MODALITIES:
                _row, _work = _run_leg_trial(
                    _group, _modality, _target, "answer_exchange", 1.0,
                    _answer_bases[_target], "legdevleverage",
                )
                _leverage_rows.append(_row)
                if len(_leverage_rows) == 1 or len(_leverage_rows) % 18 == 0:
                    print("answer leverage", len(_leverage_rows), "of", 54, _work)
    _trial_rows = []
    def _run_identity_target(_target):
        _conditions = (
            ("exact", 1.0, _concept_bases[_target]),
            ("zero", 0.0, _concept_bases[_target]),
            ("random", 1.0, _random_bases[_target]),
            ("unrelated", 1.0, _unrelated_bases),
        )
        for _group in _recruited:
            for _modality in LEG_MODALITIES:
                for _condition, _alpha, _bases in _conditions:
                    _row, _work = _run_leg_trial(
                        _group, _modality, _target, _condition, _alpha,
                        _bases, "legdevtrial",
                    )
                    _trial_rows.append(_row)
                    if len(_trial_rows) == 1 or len(_trial_rows) % 48 == 0:
                        print("leg development trials", len(_trial_rows), _work)

    print("answer leverage is diagnostic only; identity outcomes decide admission")
    _run_identity_target("cat")
    _calibration_report = development_report(
        _leverage_rows, _trial_rows, expected_n=6,
    )
    if _calibration_report["calibration_passed"]:
        print("cat calibration passed; running both frozen novel targets")
        _run_identity_target("ant")
        _run_identity_target("spider")
    else:
        print("cat calibration failed; novel targets not opened")

    LEG_GENERALIZATION_DEVELOPMENT_REPORT = development_report(
        _leverage_rows, _trial_rows, expected_n=6,
    )
    LEG_GENERALIZATION_DEVELOPMENT_REPORT = {
        **LEG_GENERALIZATION_DEVELOPMENT_REPORT,
        "scientific_config": _development_config,
        "population_digest": _population["population_digest"],
        "run_dir": str(_development_run_root),
        "recruited_group_ids": [str(row["group_id"]) for row in _recruited],
    }
    LEG_GENERALIZATION_DEVELOPMENT_REPORT["report_checksum"] = payload_checksum({
        key: value for key, value in LEG_GENERALIZATION_DEVELOPMENT_REPORT.items()
        if key != "report_checksum"
    })
    _development_store.save(
        "metric", "leg_count_generalization_development",
        LEG_GENERALIZATION_DEVELOPMENT_REPORT,
    )
    LEG_GENERALIZATION_DEVELOPMENT_REPORT_PATH.write_text(
        json.dumps(LEG_GENERALIZATION_DEVELOPMENT_REPORT, indent=2),
        encoding="utf-8",
    )
    print("=" * 96)
    print("LEG-COUNT GENERALIZATION DEVELOPMENT --",
          LEG_GENERALIZATION_DEVELOPMENT_REPORT["verdict"])
    print("selected novel targets",
          LEG_GENERALIZATION_DEVELOPMENT_REPORT["selected_novel_targets"])
    for _cell in LEG_GENERALIZATION_DEVELOPMENT_REPORT["effect_cells"]:
        print(_cell["target"], _cell["modality"], {
            name: f"{row['successes']}/{row['n']}"
            for name, row in _cell["conditions"].items()
        })
    print("report", LEG_GENERALIZATION_DEVELOPMENT_REPORT_PATH)
elif RUN_STAGE7B_LEG_GENERALIZATION_DEVELOPMENT:
    print("Stage 7B requested but blocked; confirm fp32 A100 and its budget.")
'''
)

code(
    r'''
LEG_GENERALIZATION_NOVEL_REPORT = None
if REAL_MODE and LEG_GENERALIZATION_NOVEL_ENABLED:
    from jlens.lens import JacobianLens
    from jlens.mmpilot.coordinate_swap import (
        random_two_direction_basis, resolve_concept_token,
    )
    from jlens.mmpilot.leg_count_generalization import (
        MODALITIES as LEG_MODALITIES,
        NOVEL_CONDITIONS,
        NOVEL_CONTROL_CONDITIONS,
        NOVEL_TARGETS,
        NUMBER_WORDS as LEG_NUMBER_WORDS,
        TARGET_ANSWERS as LEG_TARGET_ANSWERS,
        leg_count_answer_matches,
        novel_development_report,
    )
    from jlens.mmpilot.multimodal_instrument import MODEL_DTYPE_REALIZATION
    from jlens.mmpilot.multimodal_lens import (
        build_swap_bases_for_lens, confirmation_leg_count_prompt,
        load_broad_pooled_development_source, unrestricted_swap_trial,
    )
    from jlens.mmpilot.store import RunFingerprint, UnitStore, safe_key

    if not LEG_GENERALIZATION_DEVELOPMENT_REPORT_PATH.is_file():
        raise RuntimeError("Stage 7B2 requires the completed cat calibration report")
    _cat_report = json.loads(
        LEG_GENERALIZATION_DEVELOPMENT_REPORT_PATH.read_text(encoding="utf-8")
    )
    _cat_body = {
        key: value for key, value in _cat_report.items()
        if key != "report_checksum"
    }
    _cat_checksum = str(_cat_report.get("report_checksum") or "")
    if _cat_checksum != payload_checksum(_cat_body):
        raise RuntimeError("the Stage 7B cat report failed its own checksum")
    if not _cat_checksum.startswith(EXPECTED_STAGE7B_CAT_REPORT_CHECKSUM_PREFIX):
        raise RuntimeError(
            "Stage 7B2 is pinned to "
            f"{EXPECTED_STAGE7B_CAT_REPORT_CHECKSUM_PREFIX}..., found {_cat_checksum}"
        )
    if (
        EXPECTED_STAGE7B_CAT_REPORT_CHECKSUM
        and _cat_checksum != EXPECTED_STAGE7B_CAT_REPORT_CHECKSUM
    ):
        raise RuntimeError("Stage 7B2 is pinned to a different cat report")
    if (_cat_report.get("scientific_config") or {}).get("lens_checksum") != (
        EXPECTED_BROAD_POOLED_LENS_CHECKSUM
    ):
        raise RuntimeError("the Stage 7B cat report used a different lens")
    _population, _development_candidates = _leg_population_groups("development")
    if _cat_report.get("population_digest") != _population["population_digest"]:
        raise RuntimeError("the Stage 7B cat report used a different population")
    _groups_by_id = {
        str(group["group_id"]): group for group in _development_candidates
    }
    _recruited = []
    for _group_id in _cat_report.get("recruited_group_ids") or []:
        if str(_group_id) not in _groups_by_id:
            raise RuntimeError(f"cannot reconstruct recruited group {_group_id}")
        _recruited.append(_groups_by_id[str(_group_id)])
    if len(_recruited) != 6:
        raise RuntimeError("Stage 7B2 requires the six frozen recruited groups")

    _source_pin = load_broad_pooled_development_source(
        BROAD_DEVELOPMENT_RUN_DIR,
        expected_report_checksum=EXPECTED_BROAD_DEVELOPMENT_REPORT_CHECKSUM,
        expected_population_digest=EXPECTED_BROAD_DEVELOPMENT_POPULATION_DIGEST,
        expected_lens_checksum=EXPECTED_BROAD_POOLED_LENS_CHECKSUM,
        expected_direction=CONFIRMATION_DIRECTION,
    )
    _lens = JacobianLens.load(_source_pin["lens_path"])
    # Whether the pinned cat run already spent these photographs on ant and
    # spider decides what the development stage may be called.  If it did, this
    # extension is a re-analysis with better controls, not a sealed first look.
    # Either way the 22 confirmation photographs stay closed, which is what the
    # confirmatory claim actually rests on.
    _previously_opened = sorted({
        str(_cell.get("target"))
        for _cell in (_cat_report.get("effect_cells") or [])
        if str(_cell.get("target")) in NOVEL_TARGETS
    })
    _novel_random_seeds = (2026082701, 2026082702, 2026082703)
    _novel_config = {
        "study": "multimodal_distinct_leg_count_novel_development.v2",
        "novel_design_digest": NOVEL_LEG_DESIGN["design_digest"],
        "source_cat_report_checksum": _cat_checksum,
        "source_cat_report_checksum_prefix": (
            EXPECTED_STAGE7B_CAT_REPORT_CHECKSUM_PREFIX
        ),
        "novel_targets_previously_opened": _previously_opened,
        "development_is_sealed_first_look": not _previously_opened,
        "population_digest": _population["population_digest"],
        "recruited_group_ids": [str(row["group_id"]) for row in _recruited],
        "lens_checksum": EXPECTED_BROAD_POOLED_LENS_CHECKSUM,
        "model_repo_id": MODEL_REPO_ID,
        "model_revision": MODEL_REVISION,
        "model_dtype": "float32",
        "targets": {target: LEG_TARGET_ANSWERS[target] for target in NOVEL_TARGETS},
        "selection_priority": list(NOVEL_TARGETS),
        "layers": list(BROAD_POOLED_BAND),
        "alpha": 1.0,
        "positions": "every_original_prompt_position",
        "executed_conditions": list(NOVEL_CONDITIONS),
        "scored_controls": list(NOVEL_CONTROL_CONDITIONS),
        "answer_leverage_role": "diagnostic_only_never_gating",
        "random_seeds": list(_novel_random_seeds),
        "fresh_confirmation_opened": False,
        "fitting_performed": False,
        "backward_passes": 0,
        "commit": COMMIT,
    }
    _novel_digest = payload_checksum(_novel_config)
    _novel_run_root = (
        LEG_GENERALIZATION_DEVELOPMENT_ROOT
        / f"legnovel_real_{_novel_digest.removeprefix('sha256:')[:12]}"
    )
    _novel_run_root.mkdir(parents=True, exist_ok=True)
    (_novel_run_root / "scientific_config.json").write_text(
        json.dumps(_novel_config, indent=2), encoding="utf-8"
    )
    _novel_store = UnitStore(
        _novel_run_root,
        RunFingerprint(
            mode="real", model_repo_id=MODEL_REPO_ID,
            model_revision=MODEL_REVISION, processor_revision=MODEL_REVISION,
            layers=tuple(BROAD_POOLED_BAND),
            lens_checksum=EXPECTED_BROAD_POOLED_LENS_CHECKSUM,
            manifest_checksum=MANIFEST_CHECKSUM,
            split_id=payload_checksum(_novel_config["recruited_group_ids"]),
            intervention_config={
                "source_report": _cat_checksum,
                "targets": list(NOVEL_TARGETS),
                "conditions": list(NOVEL_CONDITIONS),
                "random_seeds": list(_novel_random_seeds),
                "alpha": 1.0, "positions": "all_original_prompt_positions",
                "dtype": "float32",
            },
            extra={"study_digest": _novel_digest},
        ),
    )
    print("novel leg-target development run", _novel_store.open())
    print("source cat report", _cat_checksum)
    print("novel targets already opened on these photos:",
          _previously_opened or "none")
    print("fitting performed False; backward passes 0")

    _unembed = BACKEND.unembedding_weight()
    _concept_names = ("bird", *NOVEL_TARGETS, *BROAD_POOLED_CONTROLS)
    _concept_tokens = {
        name: resolve_concept_token(BACKEND.encode_candidate, name)
        for name in _concept_names
    }
    _answer_tokens = {
        answer: resolve_concept_token(BACKEND.encode_candidate, word)
        for answer, word in LEG_NUMBER_WORDS.items()
    }
    _concept_bases = {
        target: build_swap_bases_for_lens(
            _lens, _unembed, layers=BROAD_POOLED_BAND,
            source=_concept_tokens["bird"], target=_concept_tokens[target],
        )
        for target in NOVEL_TARGETS
    }
    _random_bases = {
        target: {
            random_index: {
                layer: random_two_direction_basis(
                    basis,
                    seed=(
                        _novel_random_seeds[random_index]
                        + 100000 * target_index + layer
                    ),
                )
                for layer, basis in _concept_bases[target].items()
            }
            for random_index in range(3)
        }
        for target_index, target in enumerate(NOVEL_TARGETS)
    }
    _unrelated_bases = build_swap_bases_for_lens(
        _lens, _unembed, layers=BROAD_POOLED_BAND,
        source=_concept_tokens[BROAD_POOLED_CONTROLS[0]],
        target=_concept_tokens[BROAD_POOLED_CONTROLS[1]],
    )
    # Diagnostic only, and it can never gate.  Six and eight are far rarer
    # answers than four, so a null on ant or spider has two very different
    # readings: the band cannot install that answer at all, or it can but the
    # identity does not carry it.  Exchanging the answer coordinate directly
    # separates them, which is what makes a negative result publishable rather
    # than merely disappointing.
    _answer_bases = {
        target: build_swap_bases_for_lens(
            _lens, _unembed, layers=BROAD_POOLED_BAND,
            source=_answer_tokens["2"],
            target=_answer_tokens[LEG_TARGET_ANSWERS[target]],
        )
        for target in NOVEL_TARGETS
    }

    def _run_novel_trial(group, modality, target, condition, alpha, bases):
        key = safe_key("legnovel", group["group_id"], modality, target, condition)
        stored = _novel_store.load("intervention", key)
        if stored is not None:
            return stored, "reused"
        inputs = build_group_inputs(
            group, modality,
            confirmation_leg_count_prompt(modality, group["caption"]),
        )
        clean_logits = BACKEND.forward_logits(inputs.tensors)[
            0, inputs.final_prompt_position
        ].float()
        trial = unrestricted_swap_trial(
            BACKEND, inputs, bases=bases, alpha=alpha,
            target_token_id=int(
                _answer_tokens[LEG_TARGET_ANSWERS[target]].token_id
            ),
            source_token_id=int(_answer_tokens["2"].token_id),
            clean_logits=clean_logits, compact_positions=True,
            realization_policy=MODEL_DTYPE_REALIZATION,
        )
        surface = BACKEND.decode_token(int(trial["patched_top_token_id"])).strip()
        trial = {
            **trial, "patched_surface": surface,
            "success": leg_count_answer_matches(
                surface, LEG_TARGET_ANSWERS[target]
            ),
        }
        stored = _compact_leg_trial(
            trial, group=group, modality=modality, target=target,
            condition=condition, expected=LEG_TARGET_ANSWERS[target],
        )
        _novel_store.save("intervention", key, stored)
        return stored, "computed"

    _novel_rows = []
    for _target in NOVEL_TARGETS:
        _conditions = [
            ("exact", 1.0, _concept_bases[_target]),
            ("zero", 0.0, _concept_bases[_target]),
            ("unrelated", 1.0, _unrelated_bases),
            *[
                (f"random_{index}", 1.0, _random_bases[_target][index])
                for index in range(3)
            ],
        ]
        for _group in _recruited:
            for _modality in LEG_MODALITIES:
                for _condition, _alpha, _bases in _conditions:
                    _row, _work = _run_novel_trial(
                        _group, _modality, _target, _condition, _alpha, _bases
                    )
                    _novel_rows.append(_row)
                    if len(_novel_rows) == 1 or len(_novel_rows) % 36 == 0:
                        print("novel leg trials", len(_novel_rows), "of 216", _work)

    _novel_leverage_rows = []
    for _target in NOVEL_TARGETS:
        for _group in _recruited:
            for _modality in LEG_MODALITIES:
                _row, _work = _run_novel_trial(
                    _group, _modality, _target, "answer_exchange", 1.0,
                    _answer_bases[_target],
                )
                _novel_leverage_rows.append(_row)
                if len(_novel_leverage_rows) % 18 == 0:
                    print("novel answer leverage", len(_novel_leverage_rows),
                          "of 36", _work)

    LEG_GENERALIZATION_NOVEL_REPORT = novel_development_report(
        _novel_rows, _novel_leverage_rows, expected_n=6,
    )
    LEG_GENERALIZATION_NOVEL_REPORT = {
        **LEG_GENERALIZATION_NOVEL_REPORT,
        "scientific_config": _novel_config,
        "source_cat_report_checksum": _cat_checksum,
        "novel_targets_previously_opened": _previously_opened,
        "population_digest": _population["population_digest"],
        "run_dir": str(_novel_run_root),
    }
    LEG_GENERALIZATION_NOVEL_REPORT["report_checksum"] = payload_checksum({
        key: value for key, value in LEG_GENERALIZATION_NOVEL_REPORT.items()
        if key != "report_checksum"
    })
    _novel_store.save(
        "metric", "leg_count_novel_target_development",
        LEG_GENERALIZATION_NOVEL_REPORT,
    )
    LEG_GENERALIZATION_NOVEL_REPORT_PATH.write_text(
        json.dumps(LEG_GENERALIZATION_NOVEL_REPORT, indent=2), encoding="utf-8"
    )
    print("=" * 96)
    print("NOVEL LEG-TARGET DEVELOPMENT --",
          LEG_GENERALIZATION_NOVEL_REPORT["verdict"])
    print("passing", LEG_GENERALIZATION_NOVEL_REPORT["passing_novel_targets"])
    print("carried forward",
          LEG_GENERALIZATION_NOVEL_REPORT["selected_confirmation_targets"])
    for _cell in LEG_GENERALIZATION_NOVEL_REPORT["effect_cells"]:
        print(_cell["target"], _cell["modality"], {
            name: f"{row['successes']}/{row['n']}"
            for name, row in _cell["conditions"].items()
        })
    print("-- which answer each inserted identity actually produced --")
    for _row in LEG_GENERALIZATION_NOVEL_REPORT["target_answer_confusion"]:
        print(f"  {_row['target']}->{_row['expected_answer']}",
              _row["modality"], _row["answers"])
    for _row in LEG_GENERALIZATION_NOVEL_REPORT["double_dissociation"]:
        print("  both identities correct", _row["modality"],
              f"{_row['all_targets_correct']}/{_row['n']}")
    print("-- answer leverage (diagnostic only, never gating) --")
    for _row in LEG_GENERALIZATION_NOVEL_REPORT["answer_leverage_diagnostic"]:
        print(f"  {_row['target']}->{_row['answer']}", _row["modality"],
              f"{_row['successes']}/{_row['n']}")
    print("report", LEG_GENERALIZATION_NOVEL_REPORT_PATH)
elif RUN_STAGE7B2_NOVEL_LEG_TARGET_DEVELOPMENT:
    print("Stage 7B2 requested but blocked; confirm fp32 A100 and its budget.")
'''
)

code(
    r'''
if RUN_STAGE7C_FREEZE_LEG_GENERALIZATION_CONFIRMATION:
    if not LEG_GENERALIZATION_NOVEL_REPORT_PATH.is_file():
        raise RuntimeError("Stage 7C requires the completed Stage 7B2 report")
    _development = json.loads(
        LEG_GENERALIZATION_NOVEL_REPORT_PATH.read_text(encoding="utf-8")
    )
    _development_body = {
        key: value for key, value in _development.items()
        if key != "report_checksum"
    }
    if _development.get("report_checksum") != payload_checksum(_development_body):
        raise RuntimeError("the Stage 7B2 report failed its checksum")
    if _development.get("verdict") != "LEG_COUNT_NOVEL_TARGET_DEVELOPMENT_GO":
        print("NOVEL LEG CONFIRMATION NOT LICENSED -- DEVELOPMENT NO_GO")
    else:
        _population, _confirmation_candidates = _leg_population_groups("confirmation")
        # The first passing target in the predeclared priority order gates the
        # verdict; any second one is carried as a supporting family.  Both are
        # run regardless, because the exact exchange toward the other identity
        # *is* the cross-target control for the primary, and a control does not
        # need to have passed anything itself.
        _frozen_targets = list(_development["selected_confirmation_targets"])
        _frozen_target = str(_frozen_targets[0])
        _secondary_targets = [str(name) for name in _frozen_targets[1:]]
        _donor_targets = [
            name for name in LEG_GENERALIZATION_DESIGN["novel_targets"]
            if name not in _frozen_targets
        ]
        _executed_targets = [*_frozen_targets, *_donor_targets]
        _confirmation_ids = [
            str(row["group_id"]) for row in _confirmation_candidates
        ]
        _opened = []
        if LEG_GENERALIZATION_DEVELOPMENT_ROOT.is_dir():
            for _path in LEG_GENERALIZATION_DEVELOPMENT_ROOT.rglob(
                "units/*/*.json"
            ):
                _raw = _path.read_text(encoding="utf-8")
                if any(group_id in _raw for group_id in _confirmation_ids):
                    _opened.append(str(_path))
        if _opened:
            raise RuntimeError(
                "confirmation groups were opened during development: "
                + repr(_opened[:10])
            )
        _confirmation_design = {
            "version": "mmpilot.multimodal_novel_leg_count_confirmation_design.v2",
            "base_design_digest": LEG_GENERALIZATION_DESIGN["design_digest"],
            "novel_design_digest": NOVEL_LEG_DESIGN["design_digest"],
            "development_report_checksum": _development["report_checksum"],
            "source_cat_report_checksum": _development[
                "source_cat_report_checksum"
            ],
            "population_digest": _population["population_digest"],
            "confirmation_group_ids": _confirmation_ids,
            "frozen_target": _frozen_target,
            "secondary_targets": _secondary_targets,
            "executed_targets": _executed_targets,
            "target_answer": LEG_GENERALIZATION_DESIGN["target_answers"][_frozen_target],
            "target_answers": {
                name: LEG_GENERALIZATION_DESIGN["target_answers"][name]
                for name in _executed_targets
            },
            "gating_family": (
                "the frozen primary target only: 3 modalities x 6 controls"
            ),
            "layers": list(BROAD_POOLED_BAND),
            "alpha": 1.0,
            "positions": "every_original_prompt_position",
            "executed_conditions": [
                "exact", "zero", "unrelated", "random_0", "random_1", "random_2"
            ],
            "scored_controls": list(NOVEL_LEG_DESIGN["scored_controls"]),
            "random_seeds": [2026082801, 2026082802, 2026082803],
            "selected_before_confirmation_outputs": True,
            "confirmation_outputs_opened": False,
            "lens_refitted": False,
            "fitting_performed": False,
            "backward_passes": 0,
        }
        _confirmation_design["design_checksum"] = payload_checksum(
            _confirmation_design
        )
        LEG_GENERALIZATION_ROOT.mkdir(parents=True, exist_ok=True)
        LEG_GENERALIZATION_CONFIRMATION_DESIGN_PATH.write_text(
            json.dumps(_confirmation_design, indent=2), encoding="utf-8"
        )
        print("LEG-COUNT GENERALIZATION CONFIRMATION DESIGN FROZEN")
        print("  gating primary target", _frozen_target)
        print("  supporting targets", _secondary_targets or "none")
        print("  executed targets", _executed_targets)
        print("  untouched candidates", len(_confirmation_ids))
        print("  checksum", _confirmation_design["design_checksum"])
        print("  model forwards 0; fitting 0; backward passes 0")
        print("STOP THIS CPU RUNTIME. Run Stage 7D alone on an 80 GB A100.")
'''
)

code(
    r'''
LEG_GENERALIZATION_CONFIRMATION_REPORT = None
if REAL_MODE and LEG_GENERALIZATION_CONFIRMATION_ENABLED:
    from jlens.lens import JacobianLens
    from jlens.mmpilot.coordinate_swap import (
        random_two_direction_basis, resolve_concept_token,
    )
    from jlens.mmpilot.leg_count_generalization import (
        MODALITIES as LEG_MODALITIES,
        NOVEL_CONDITIONS,
        NOVEL_CONTROL_CONDITIONS,
        NUMBER_WORDS as LEG_NUMBER_WORDS,
        TARGET_ANSWERS as LEG_TARGET_ANSWERS,
        leg_count_answer_matches,
        novel_confirmation_report,
    )
    from jlens.mmpilot.multimodal_instrument import MODEL_DTYPE_REALIZATION
    from jlens.mmpilot.multimodal_lens import (
        build_swap_bases_for_lens, confirmation_leg_count_prompt,
        load_broad_pooled_development_source,
        unrestricted_swap_trial,
    )
    from jlens.mmpilot.store import RunFingerprint, UnitStore, safe_key

    if not LEG_GENERALIZATION_CONFIRMATION_DESIGN_PATH.is_file():
        raise RuntimeError("run Stage 7C before opening confirmation outputs")
    _confirmation_design = json.loads(
        LEG_GENERALIZATION_CONFIRMATION_DESIGN_PATH.read_text(encoding="utf-8")
    )
    _design_body = {
        key: value for key, value in _confirmation_design.items()
        if key != "design_checksum"
    }
    if _confirmation_design.get("design_checksum") != payload_checksum(_design_body):
        raise RuntimeError("the Stage 7 confirmation design failed its checksum")
    if _confirmation_design.get("novel_design_digest") != (
        NOVEL_LEG_DESIGN["design_digest"]
    ):
        raise RuntimeError("the novel design changed after Stage 7C froze it")
    _population, _confirmation_candidates = _leg_population_groups("confirmation")
    if [str(row["group_id"]) for row in _confirmation_candidates] != list(
        _confirmation_design["confirmation_group_ids"]
    ):
        raise RuntimeError("confirmation candidates differ from the frozen design")
    _primary_target = str(_confirmation_design["frozen_target"])
    _secondary_targets = [
        str(name) for name in _confirmation_design.get("secondary_targets") or []
    ]
    _executed_targets = [
        str(name) for name in _confirmation_design["executed_targets"]
    ]
    _random_seeds = tuple(_confirmation_design["random_seeds"])
    _source_pin = load_broad_pooled_development_source(
        BROAD_DEVELOPMENT_RUN_DIR,
        expected_report_checksum=EXPECTED_BROAD_DEVELOPMENT_REPORT_CHECKSUM,
        expected_population_digest=EXPECTED_BROAD_DEVELOPMENT_POPULATION_DIGEST,
        expected_lens_checksum=EXPECTED_BROAD_POOLED_LENS_CHECKSUM,
        expected_direction=CONFIRMATION_DIRECTION,
    )
    _lens = JacobianLens.load(_source_pin["lens_path"])
    _confirmation_config = {
        "study": "fresh_multimodal_distinct_leg_count_generalization.v2",
        "design_checksum": _confirmation_design["design_checksum"],
        "novel_design_digest": NOVEL_LEG_DESIGN["design_digest"],
        "lens_checksum": EXPECTED_BROAD_POOLED_LENS_CHECKSUM,
        "model_repo_id": MODEL_REPO_ID,
        "model_revision": MODEL_REVISION,
        "model_dtype": "float32",
        "primary_target": _primary_target,
        "secondary_targets": _secondary_targets,
        "executed_targets": _executed_targets,
        "layers": list(BROAD_POOLED_BAND),
        "alpha": 1.0,
        "positions": "every_original_prompt_position",
        "executed_conditions": list(NOVEL_CONDITIONS),
        "scored_controls": list(NOVEL_CONTROL_CONDITIONS),
        "random_seeds": list(_random_seeds),
        "answer_matching": "digit_or_english_number_word.v1",
        "answer_basis_surfaces": {
            answer: LEG_NUMBER_WORDS[answer] for answer in ("2", "4", "6", "8")
        },
        "answer_leverage_role": "diagnostic_only_never_gating",
        "fitting_performed": False,
        "commit": COMMIT,
    }
    _confirmation_digest = payload_checksum(_confirmation_config)
    _confirmation_run_root = (
        LEG_GENERALIZATION_CONFIRMATION_ROOT
        / f"legconfirm_real_{_confirmation_digest.removeprefix('sha256:')[:12]}"
    )
    _confirmation_run_root.mkdir(parents=True, exist_ok=True)
    (_confirmation_run_root / "scientific_config.json").write_text(
        json.dumps(_confirmation_config, indent=2), encoding="utf-8"
    )
    _confirmation_store = UnitStore(
        _confirmation_run_root,
        RunFingerprint(
            mode="real", model_repo_id=MODEL_REPO_ID,
            model_revision=MODEL_REVISION, processor_revision=MODEL_REVISION,
            layers=tuple(BROAD_POOLED_BAND),
            lens_checksum=EXPECTED_BROAD_POOLED_LENS_CHECKSUM,
            manifest_checksum=MANIFEST_CHECKSUM,
            split_id=payload_checksum(_population["confirmation"]),
            intervention_config={
                "design_checksum": _confirmation_design["design_checksum"],
                "targets": list(_executed_targets),
                "conditions": list(NOVEL_CONDITIONS),
                "random_seeds": list(_random_seeds),
                "alpha": 1.0, "positions": "all_original_prompt_positions",
                "dtype": "float32",
            },
            extra={"study_digest": _confirmation_digest},
        ),
    )
    print("leg-generalization confirmation run", _confirmation_store.open())
    print("gating primary target", _primary_target)
    print("supporting targets", _secondary_targets or "none")
    print("executed targets", _executed_targets)
    print("fitting performed False; backward passes 0")

    _capability_rows = []
    for _group in _confirmation_candidates:
        for _modality in LEG_MODALITIES:
            _key = safe_key("legconfcap", _group["group_id"], _modality)
            _row = _confirmation_store.load("capability", _key)
            if _row is None:
                _inputs = build_group_inputs(
                    _group, _modality,
                    confirmation_leg_count_prompt(_modality, _group["caption"]),
                )
                _logits = BACKEND.forward_logits(_inputs.tensors)[
                    0, _inputs.final_prompt_position
                ].float()
                _surface = BACKEND.decode_token(int(_logits.argmax())).strip()
                _row = {
                    "group_id": str(_group["group_id"]),
                    "image_id": str(_group["image_id"]),
                    "modality": _modality, "expected": "2",
                    "generated": _surface,
                    "pass": leg_count_answer_matches(_surface, "2"),
                }
                _confirmation_store.save("capability", _key, _row)
            _capability_rows.append(_row)
            if len(_capability_rows) == 1 or len(_capability_rows) % 24 == 0:
                print("leg confirmation capability", len(_capability_rows), "of", 66)
    _recruited = []
    for _group in _confirmation_candidates:
        _rows = [
            row for row in _capability_rows
            if row["group_id"] == str(_group["group_id"])
        ]
        if len(_rows) == 3 and all(row["pass"] for row in _rows):
            _recruited.append(_group)
        if len(_recruited) == 12:
            break
    if len(_recruited) != 12:
        raise RuntimeError(
            f"Stage 7 confirmation recruited {len(_recruited)}/12 clean groups"
        )
    print("leg confirmation recruited", len(_recruited), "/ 12")

    _unembed = BACKEND.unembedding_weight()
    _concept_names = ("bird", *_executed_targets, *BROAD_POOLED_CONTROLS)
    _concept_tokens = {
        name: resolve_concept_token(BACKEND.encode_candidate, name)
        for name in _concept_names
    }
    _answer_tokens = {
        answer: resolve_concept_token(BACKEND.encode_candidate, word)
        for answer, word in LEG_NUMBER_WORDS.items()
    }
    _concept_bases = {
        target: build_swap_bases_for_lens(
            _lens, _unembed, layers=BROAD_POOLED_BAND,
            source=_concept_tokens["bird"], target=_concept_tokens[target],
        )
        for target in _executed_targets
    }
    _answer_bases = {
        target: build_swap_bases_for_lens(
            _lens, _unembed, layers=BROAD_POOLED_BAND,
            source=_answer_tokens["2"],
            target=_answer_tokens[LEG_TARGET_ANSWERS[target]],
        )
        for target in _executed_targets
    }
    _random_bases = {
        target: {
            seed_index: {
                layer: random_two_direction_basis(
                    basis,
                    seed=_random_seeds[seed_index] + 100000 * target_index + layer,
                )
                for layer, basis in _concept_bases[target].items()
            }
            for seed_index in range(len(_random_seeds))
        }
        for target_index, target in enumerate(_executed_targets)
    }
    _unrelated_bases = build_swap_bases_for_lens(
        _lens, _unembed, layers=BROAD_POOLED_BAND,
        source=_concept_tokens[BROAD_POOLED_CONTROLS[0]],
        target=_concept_tokens[BROAD_POOLED_CONTROLS[1]],
    )

    def _run_confirmation_leg_trial(
        group, modality, target, condition, alpha, bases, prefix
    ):
        key = safe_key(prefix, group["group_id"], modality, target, condition)
        stored = _confirmation_store.load("intervention", key)
        if stored is not None:
            return stored, "reused"
        inputs = build_group_inputs(
            group, modality,
            confirmation_leg_count_prompt(modality, group["caption"]),
        )
        clean_logits = BACKEND.forward_logits(inputs.tensors)[
            0, inputs.final_prompt_position
        ].float()
        trial = unrestricted_swap_trial(
            BACKEND, inputs, bases=bases, alpha=alpha,
            target_token_id=int(_answer_tokens[LEG_TARGET_ANSWERS[target]].token_id),
            source_token_id=int(_answer_tokens["2"].token_id),
            clean_logits=clean_logits, compact_positions=True,
            realization_policy=MODEL_DTYPE_REALIZATION,
        )
        surface = BACKEND.decode_token(int(trial["patched_top_token_id"])).strip()
        trial = {
            **trial, "patched_surface": surface,
            "success": leg_count_answer_matches(
                surface, LEG_TARGET_ANSWERS[target]
            ),
        }
        stored = _compact_leg_trial(
            trial, group=group, modality=modality, target=target,
            condition=condition, expected=LEG_TARGET_ANSWERS[target],
        )
        _confirmation_store.save("intervention", key, stored)
        return stored, "computed"

    _expected_trials = len(_executed_targets) * len(NOVEL_CONDITIONS) * 12 * 3
    _trial_rows = []
    for _target in _executed_targets:
        _conditions = [
            ("exact", 1.0, _concept_bases[_target]),
            ("zero", 0.0, _concept_bases[_target]),
            ("unrelated", 1.0, _unrelated_bases),
            *[
                (f"random_{index}", 1.0, _random_bases[_target][index])
                for index in range(len(_random_seeds))
            ],
        ]
        for _group in _recruited:
            for _modality in LEG_MODALITIES:
                for _condition, _alpha, _bases in _conditions:
                    _row, _work = _run_confirmation_leg_trial(
                        _group, _modality, _target, _condition, _alpha,
                        _bases, "legconftrial",
                    )
                    _trial_rows.append(_row)
                    if len(_trial_rows) == 1 or len(_trial_rows) % 72 == 0:
                        print("leg confirmation trials", len(_trial_rows),
                              "of", _expected_trials, _work)

    _leverage_rows = []
    for _target in _executed_targets:
        for _group in _recruited:
            for _modality in LEG_MODALITIES:
                _row, _work = _run_confirmation_leg_trial(
                    _group, _modality, _target, "answer_exchange", 1.0,
                    _answer_bases[_target], "legconfleverage",
                )
                _leverage_rows.append(_row)
                if len(_leverage_rows) % 36 == 0:
                    print("confirmation answer leverage", len(_leverage_rows), _work)

    LEG_GENERALIZATION_CONFIRMATION_REPORT = novel_confirmation_report(
        _trial_rows, _leverage_rows,
        target=_primary_target, secondary_targets=_secondary_targets,
        expected_n=12,
    )
    LEG_GENERALIZATION_CONFIRMATION_REPORT = {
        **LEG_GENERALIZATION_CONFIRMATION_REPORT,
        "scientific_config": _confirmation_config,
        "confirmation_design": _confirmation_design,
        "population_digest": _population["population_digest"],
        "run_dir": str(_confirmation_run_root),
        "recruited_group_ids": [str(row["group_id"]) for row in _recruited],
    }
    LEG_GENERALIZATION_CONFIRMATION_REPORT["report_checksum"] = payload_checksum({
        key: value for key, value in LEG_GENERALIZATION_CONFIRMATION_REPORT.items()
        if key != "report_checksum"
    })
    _confirmation_store.save(
        "metric", "fresh_leg_count_generalization",
        LEG_GENERALIZATION_CONFIRMATION_REPORT,
    )
    LEG_GENERALIZATION_CONFIRMATION_REPORT_PATH.write_text(
        json.dumps(LEG_GENERALIZATION_CONFIRMATION_REPORT, indent=2),
        encoding="utf-8",
    )
    print("=" * 96)
    print("FRESH MULTIMODAL NOVEL LEG-COUNT GENERALIZATION --",
          LEG_GENERALIZATION_CONFIRMATION_REPORT["verdict"])
    for _result in LEG_GENERALIZATION_CONFIRMATION_REPORT["target_results"]:
        print(f"{_result['role']} {_result['target']}->{_result['answer']}",
              "effect", _result["effect_passed"],
              "significance", _result["significance_passed"])
        for _cell in _result["effect_cells"]:
            print("   ", _cell["modality"], {
                name: f"{row['successes']}/{row['n']}"
                for name, row in _cell["conditions"].items()
            })
    print("-- which answer each inserted identity actually produced --")
    for _row in LEG_GENERALIZATION_CONFIRMATION_REPORT["target_answer_confusion"]:
        print(f"  {_row['target']}->{_row['expected_answer']}",
              _row["modality"], _row["answers"])
    for _row in LEG_GENERALIZATION_CONFIRMATION_REPORT["double_dissociation"]:
        print("  every identity correct", _row["modality"],
              f"{_row['all_targets_correct']}/{_row['n']}")
    print("-- gating family, Holm-adjusted --")
    for _row in LEG_GENERALIZATION_CONFIRMATION_REPORT["paired_comparisons"]:
        print(f"  {_row['modality']:<13} vs {_row['control']:<12}",
              f"discordant {_row['primary_only']}/{_row['discordant']}",
              f"holm p {_row['holm_adjusted_p']:.5f}")
    print("-- answer leverage (diagnostic only, never gating) --")
    for _row in LEG_GENERALIZATION_CONFIRMATION_REPORT["answer_leverage_diagnostic"]:
        print(f"  {_row['target']}->{_row['answer']}", _row["modality"],
              f"{_row['successes']}/{_row['n']}")
    print("report", LEG_GENERALIZATION_CONFIRMATION_REPORT_PATH)
elif RUN_STAGE7D_LEG_GENERALIZATION_CONFIRMATION:
    print("Stage 7D requested but blocked; confirm fp32 A100 and its budget.")
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
