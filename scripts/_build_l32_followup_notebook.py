# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Generate ``notebooks/multimodal_jspace_spokencoco_l32_followup_colab.ipynb``.

The notebook is written from source here rather than edited inside a JSON blob,
so the committed file stays output-free and byte-reproducible.

Run with ``python scripts/_build_l32_followup_notebook.py`` after changing a
cell.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TARGET = (
    REPO_ROOT
    / "notebooks"
    / "multimodal_jspace_spokencoco_l32_followup_colab.ipynb"
)

CELLS: list[tuple[str, str]] = []


def markdown(text: str) -> None:
    CELLS.append(("markdown", text.strip("\n")))


def code(text: str) -> None:
    CELLS.append(("code", text.strip("\n")))


# ============================================================ 0. front matter

markdown(
    """
# Layer 32 confirmatory follow-up — J-space causal transfer under an open prompt

The research-grade early-layer extension confirmed **physical layer 32** at
scale 250 on its own untouched 256-prompt confirmation set:

```
verdict          EARLY_LAYER_CALIBRATION_GO
L32              MRR 0.4058, median midrank 3.50, top-10 inclusion 0.574
L26              failed
invalid units    0
parent run       immutable
```

This notebook asks four questions of that lens, and nothing else:

1. Does controlled J-space causal transfer replicate at L32 across **text**,
   **image** and **spoken audio**?
2. Does it survive the same image-independent, capability-admissible controls?
3. Has the model's **native direct output readout** already converged at L32?
4. Can transfer be claimed **before native direct-readout convergence**?

A negative or inconclusive answer is a result. No threshold, concept, prompt,
sample, alpha or control is tuned after an outcome is visible.

## The intervention this notebook runs

**Source-derived J-space causal steering.** A concept direction is estimated
from *source-modality training examples only* —

```
delta = ReLU(mean positive J-code - mean matched-negative J-code)
v     = V delta,  normalized to unit L2
h'    = h +- alpha * v * (mean clean activation norm)
```

— and applied at the final prompt token of a **held-out target-modality**
example.

**This is not the Anthropic two-coordinate swap.** That intervention measures
its coefficient from the activation itself (`c = pinv(V) h`;
`h + alpha V (sigma(c) - c)`) and the paper applies it over a **contiguous band**
of intermediate layers. The repository implements it in full
(`jlens/mmpilot/coordinate_swap.py`), and it is not usable here: the confirmed
layers are 32, and separately 35/38/40, which is not a contiguous band, so
`build_layer_band` refuses every band that exists today. A real coordinate-swap
study is a **separate follow-up**. Nothing in this notebook renames steering as
a swap, and `intervention_family` is written into every artifact so the two can
never resume from each other's directories.

## What is new here, and what is read-only

Newly measured under the **open** prompt protocol: everything at layer 32.

Read-only historical context: the completed three-modality study
(`mmaudio_native_audio_transfer_20260806T144822`), which ran under
`gemma-it-chat-balanced-options-v1` — a question that **lists all six
candidates**. Its numbers are not paired with anything measured here, and
section 6 refuses a cross-layer comparison unless the paired L35 reference is
run under *this* protocol.

## Not in scope

Layers 33 and 34 are neither fitted nor tested. Section 18 gives the conditional
rule for whether they should be, decided before the result is visible.
"""
)

# =========================================================== 1. bootstrap

markdown(
    """
## 1. Bootstrap repository

Run these three cells first, in order. They use nothing but the standard
library: the repository is not importable until 1c has installed it.

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

# ============================================================ 2. switches

markdown(
    """
## 2. Switches, pins and the design

Every switch is `False` in the committed notebook. Opening it starts nothing,
downloads nothing and spends nothing.

| switch | what it unlocks |
|---|---|
| `RUN_REAL_L32_FOLLOWUP` | the real Drive artifacts and the real published L32 lens instead of the deterministic MOCK world |
| `RUN_MODEL_STAGES` | allows Gemma to be loaded at all |
| `CONFIRM_MODEL_LOAD` | acknowledges the ~16 GB download |
| `CONFIRM_CAUSAL_BUDGET` | acknowledges the L32 pass budget printed in section 10 |
| `RUN_PAIRED_L35_REFERENCE` | adds the **paired L35 reference condition** under this same open protocol |
| `CONFIRM_PAIRED_REFERENCE_BUDGET` | acknowledges the reference's additional passes |

The real run needs `RUN_REAL_L32_FOLLOWUP`, `RUN_MODEL_STAGES`,
`CONFIRM_MODEL_LOAD` and `CONFIRM_CAUSAL_BUDGET` all `True`. The paired
reference needs both of its own switches on top; section 6 explains when it is
methodologically required, **before** section 10 asks you to confirm its cost.

`L32_PHYSICAL_LAYER` and `LENS_FITTED_SCALE` are not free parameters. They are
what the extension confirmed, and section 5 refuses an artifact that disagrees.
"""
)

code(
    '''
# 2. Configuration. Requires section 1 (it imports from the repository).
# Nothing here mounts Drive, reads data, or loads a model.
RUN_REAL_L32_FOLLOWUP = False
RUN_MODEL_STAGES = False
CONFIRM_MODEL_LOAD = False
CONFIRM_CAUSAL_BUDGET = False
RUN_PAIRED_L35_REFERENCE = False
CONFIRM_PAIRED_REFERENCE_BUDGET = False

# ------------------------------------------------------------------ design
N_CONCEPTS = 6
N_FOCAL_CONCEPTS = 3
L32_PHYSICAL_LAYER = 32
REFERENCE_PHYSICAL_LAYER = 35
LENS_FITTED_SCALE = 250
ALPHAS = (0.0, 0.25, 0.5, 1.0)
CAPABILITY_THRESHOLD = 0.7
SPLIT_SEED = "spokencoco-l32-followup-v1"

N_TRAIN_POSITIVE_IMAGES = 8 if RUN_REAL_L32_FOLLOWUP else 2
N_TEST_POSITIVE_IMAGES = 8 if RUN_REAL_L32_FOLLOWUP else 2
N_TRAIN_NEGATIVE_IMAGES = 8 if RUN_REAL_L32_FOLLOWUP else 2
N_TEST_NEGATIVE_IMAGES = 8 if RUN_REAL_L32_FOLLOWUP else 2

# --------------------------------------- the extension run that published L32
# The FILENAME IS NEVER TYPED HERE. Section 5 resolves it from this run's own
# report, publication metadata and checksums.
EXTENSION_RUN_DIR = (
    "/content/drive/MyDrive/jacobian-lens-gemma/runs/rgext_real_c18f03f06e7b"
)

# --------------------------------- the scale-100 lens for the L35 reference
# Used ONLY when RUN_PAIRED_L35_REFERENCE is True. Layer 35 was confirmed by
# the scale-100 calibration run and is loaded under ITS scale, not this one.
REFERENCE_LENS_DIR = (
    "/content/drive/MyDrive/jacobian-lens-gemma/runs/"
    "rgcalib_real_7e3736b4de8f/artifacts/published"
)
REFERENCE_LENS_FILE = "lens.layer35.scale100.validated.pt"
REFERENCE_LENS_SHA256 = (
    "sha256:64fb02d718ac48adc1bced99e2eff3c2215052ba144d5dedac05f17936a96ed1"
)
REFERENCE_LENS_FITTED_SCALE = 100

# --------------------------------------------- the completed, read-only study
COMPLETED_TRANSFER_RUN_DIR = (
    "/content/drive/MyDrive/jacobian-lens-gemma/runs/"
    "mmaudio_native_audio_transfer_20260806T144822"
)
COMPLETED_PILOT_RUN_DIR = (
    "/content/drive/MyDrive/jacobian-lens-gemma/runs/mmpilot_pilot_20260803T160711"
)

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
    "audioaudit_", "text_jlens_", "mmaudio_", "mmconv_",
)

import json

from jlens.mmpilot.l32_followup import (
    INTERVENTION_FAMILY,
    L32_FOLLOWUP_PROTOCOL,
    OPEN_PROMPT_PROTOCOL,
    SELECTED_SCALE,
)
from jlens.mmpilot.pipeline import PilotConfig
from jlens.mmpilot.selection import IMAGE_UNIQUE_MOCK_PROFILE, IMAGE_UNIQUE_PROFILE
from jlens.mmpilot.tri_modal import ALL_PAIRS, TriModalThresholds

if RUN_REAL_L32_FOLLOWUP:
    # The real path never reinterprets what "layer 32 at scale 250" means.
    if L32_PHYSICAL_LAYER != 32 or LENS_FITTED_SCALE != SELECTED_SCALE:
        raise RuntimeError(
            f"the real path is pinned to physical layer 32 at scale "
            f"{SELECTED_SCALE}; got layer {L32_PHYSICAL_LAYER} at scale "
            f"{LENS_FITTED_SCALE}"
        )

SCRATCH = Path(os.environ.get("MMPILOT_SCRATCH") or "/content/l32_scratch")
SCRATCH.mkdir(parents=True, exist_ok=True)
RESOLVED_RUNS_ROOT = Path(
    os.environ.get("MMPILOT_RUNS_ROOT")
    or (RUNS_ROOT if RUN_REAL_L32_FOLLOWUP else SCRATCH / "runs")
)
EXTENSION_RUN_DIR = os.environ.get("MMPILOT_EXTENSION_RUN_DIR") or EXTENSION_RUN_DIR
COMPLETED_TRANSFER_RUN_DIR = (
    os.environ.get("MMPILOT_COMPLETED_RUN_DIR") or COMPLETED_TRANSFER_RUN_DIR
)

MODALITIES = ("text", "image", "spoken_audio")
PROFILE = IMAGE_UNIQUE_PROFILE if RUN_REAL_L32_FOLLOWUP else IMAGE_UNIQUE_MOCK_PROFILE

# The MOCK decoder has six blocks. Its stand-in for "layer 32 published by the
# extension at its own scale" is layer 1, and for "layer 35 published at scale
# 100" is layer 2 — exactly the substitution the completed MOCK runs use.
CAUSAL_LAYER = L32_PHYSICAL_LAYER if RUN_REAL_L32_FOLLOWUP else 1
REFERENCE_LAYER = REFERENCE_PHYSICAL_LAYER if RUN_REAL_L32_FOLLOWUP else 2
LAYERS = (CAUSAL_LAYER,)

THRESHOLDS = TriModalThresholds(
    capability_threshold=CAPABILITY_THRESHOLD,
    required_positive_images_per_cell=N_TEST_POSITIVE_IMAGES,
    required_negative_images_per_cell=N_TEST_NEGATIVE_IMAGES,
)

CONFIG = PilotConfig(
    mode="l32_followup" if RUN_REAL_L32_FOLLOWUP else "mock",
    layers=(CAUSAL_LAYER,),
    causal_layers=(CAUSAL_LAYER,),
    modalities=MODALITIES,
    capability_threshold=CAPABILITY_THRESHOLD,
    alphas=tuple(ALPHAS),
    n_target_examples=N_TEST_POSITIVE_IMAGES,
    pursuit_k=25 if RUN_REAL_L32_FOLLOWUP else 8,
    pursuit_correlation_chunk_size=65536 if RUN_REAL_L32_FOLLOWUP else None,
    direction_top_k=16 if RUN_REAL_L32_FOLLOWUP else 4,
    n_permutations=50 if RUN_REAL_L32_FOLLOWUP else 8,
    max_capability_groups_per_concept=8,
    seed=20260808,
    subset_profile=PROFILE.name,
    image_unique_targets=True,
    min_source_positive_images=N_TRAIN_POSITIVE_IMAGES,
    min_source_negative_images=N_TRAIN_NEGATIVE_IMAGES,
    off_diagonal_causal_only=True,
)

# The three real gates exist to protect a ~16 GB download and an L4-hour bill.
# In MOCK the "model" is a few-hundred-parameter CPU stub, there is nothing to
# download and nothing to spend, so the deterministic MOCK runs its stages
# without them. The REAL path requires all three, and section 10b refuses a
# half-set combination in either mode.
REAL_GATES_SATISFIED = (
    RUN_MODEL_STAGES and CONFIRM_MODEL_LOAD and CONFIRM_CAUSAL_BUDGET
)
MODEL_STAGES_ENABLED = REAL_GATES_SATISFIED if RUN_REAL_L32_FOLLOWUP else True

# The paired reference is opt-in in BOTH modes: it is the condition that makes a
# cross-layer statement possible, and running it silently would hide that the
# statement needed it.
PAIRED_REFERENCE_ENABLED = (
    MODEL_STAGES_ENABLED
    and RUN_PAIRED_L35_REFERENCE
    and CONFIRM_PAIRED_REFERENCE_BUDGET
)

print(f"RUN_REAL_L32_FOLLOWUP           = {RUN_REAL_L32_FOLLOWUP}")
print(f"RUN_MODEL_STAGES                = {RUN_MODEL_STAGES}")
print(f"CONFIRM_MODEL_LOAD              = {CONFIRM_MODEL_LOAD}")
print(f"CONFIRM_CAUSAL_BUDGET           = {CONFIRM_CAUSAL_BUDGET}")
print(f"RUN_PAIRED_L35_REFERENCE        = {RUN_PAIRED_L35_REFERENCE}")
print(f"CONFIRM_PAIRED_REFERENCE_BUDGET = {CONFIRM_PAIRED_REFERENCE_BUDGET}")
print()
print(f"model stages enabled            {MODEL_STAGES_ENABLED}")
print(f"paired L35 reference enabled    {PAIRED_REFERENCE_ENABLED}")
print()
print(f"protocol           {L32_FOLLOWUP_PROTOCOL}")
print(f"prompt protocol    {OPEN_PROMPT_PROTOCOL}")
print(f"intervention       {INTERVENTION_FAMILY}")
print(f"                   (steering, NOT the Anthropic coordinate swap)")
print(f"causal layer       {CAUSAL_LAYER}  (real: {L32_PHYSICAL_LAYER})")
print(f"reference layer    {REFERENCE_LAYER}  (real: {REFERENCE_PHYSICAL_LAYER})")
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
if RUN_REAL_L32_FOLLOWUP and IN_COLAB:
    from google.colab import drive

    drive.mount("/content/drive")
    DRIVE_STATUS = "mounted"

if RUN_REAL_L32_FOLLOWUP:
    _required = {
        "extension run": EXTENSION_RUN_DIR,
        "completed transfer run": COMPLETED_TRANSFER_RUN_DIR,
        "manifest": MANIFEST_PATH,
        "image media root": IMAGE_MEDIA_ROOT,
        "audio media root": AUDIO_MEDIA_ROOT,
        "runs root": RUNS_ROOT,
    }
    _missing = {name: path for name, path in _required.items() if not Path(path).exists()}
    if _missing:
        raise RuntimeError(
            "these configured paths do not exist:\\n  "
            + "\\n  ".join(f"{name}: {path}" for name, path in sorted(_missing.items()))
        )
    for _name, _path in sorted(_required.items()):
        print(f"  ok  {_name:24s} {_path}")
    if RUN_PAIRED_L35_REFERENCE:
        _ref = Path(REFERENCE_LENS_DIR) / REFERENCE_LENS_FILE
        if not _ref.is_file():
            raise RuntimeError(f"the paired-reference lens is missing: {_ref}")
        print(f"  ok  {'reference lens':24s} {_ref}")
else:
    print("skipped: RUN_REAL_L32_FOLLOWUP is False (the MOCK world needs no Drive)")
print(f"\\ndrive: {DRIVE_STATUS}")
'''
)

# ========================================================= 4. environment

markdown(
    """
## 4. Runtime and versions

Recorded, and on the real path pinned. A different Transformers version is a
different tokenizer, a different processor and potentially a different audio
placeholder convention.
"""
)

code(
    '''
# 4. Runtime report. Never touches the dataset.
import platform

import torch

try:
    import transformers

    TRANSFORMERS_VERSION = transformers.__version__
except ImportError:
    TRANSFORMERS_VERSION = None

ENVIRONMENT = {
    "python": platform.python_version(),
    "platform": platform.platform(),
    "torch": torch.__version__,
    "transformers": TRANSFORMERS_VERSION,
    "cuda_available": torch.cuda.is_available(),
    "device_name": (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    ),
    "commit": COMMIT,
}
if RUN_REAL_L32_FOLLOWUP and TRANSFORMERS_VERSION != TRANSFORMERS_VERSION_EXPECTED:
    raise RuntimeError(
        f"transformers {TRANSFORMERS_VERSION} != pinned "
        f"{TRANSFORMERS_VERSION_EXPECTED}"
    )
print(json.dumps(ENVIRONMENT, indent=2))
'''
)

# =============================================== 5. artifact discovery

markdown(
    """
## 5. Resolve the published L32 lens — from metadata, not from a filename

**The artifact filename is never assumed.** The chain, every link checked:

1. `artifacts/early_layer_extension_report.json` declares the schema this code
   knows, `mode == "real"`, verdict `EARLY_LAYER_CALIBRATION_GO`, selected scale
   250, zero invalid units, and a parent-immutability proof;
2. its `publication` block lists layer 32 and carries a checksum for it;
3. **exactly one** `*.extension.json` sidecar under `artifacts/published`
   claims layer 32 at scale 250, `validated: true`, status
   `PUBLISHED_VALIDATED_EARLY_LAYER`. Two candidates is a refusal, not a
   sort-order decision;
4. the lens path comes from *that sidecar*, resolved inside the published
   directory (a run copied between Drive mounts keeps its bytes and loses its
   absolute paths);
5. the file's own sha256 must equal **both** the sidecar's `lens_checksum`
   **and** the report's `published_checksums["32"]`;
6. the sidecar recomputes its own `artifact_checksum`.

Then the ordinary published-artifact validation runs — schema completeness,
frozen, validated, model id and revision, tokenizer revision, `d_model`, hook
site, residual convention, vector orientation, normalization convention,
calibration modality `text-only`, and **fitted scale 250**.

**Scale is part of "confirmed".** Layer 32 failed at scale 100 and passed at
scale 250. The scale-100 module defaults say so, and this study states its own
expectations rather than editing that record.
"""
)

code(
    '''
# 5. Resolve, verify and load the published L32 lens. NOTHING IS FITTED HERE.
from jlens.mmpilot.l32_followup import (
    ArtifactDiscoveryRefused,
    discover_published_l32_lens,
    l32_expectations,
    validate_discovered_lens,
)
from jlens.mmpilot.published_lens import (
    PublishedLensRefused,
    format_lens_report,
    load_published_lenses,
)

if RUN_REAL_L32_FOLLOWUP:
    RESOLVED_EXTENSION_RUN_DIR = EXTENSION_RUN_DIR
    MOCK_EXTENSION = None
else:
    from jlens.mmpilot.mock import MOCK_D_MODEL, build_mock_extension_run

    MOCK_EXTENSION = build_mock_extension_run(
        SCRATCH / "extension_run",
        layer=CAUSAL_LAYER,
        scale=LENS_FITTED_SCALE,
        d_model=MOCK_D_MODEL,
    )
    RESOLVED_EXTENSION_RUN_DIR = MOCK_EXTENSION["root"]

DISCOVERED = discover_published_l32_lens(
    RESOLVED_EXTENSION_RUN_DIR,
    layer=CAUSAL_LAYER,
    expected_scale=LENS_FITTED_SCALE,
)
L32_EXPECTATIONS = l32_expectations(
    model_repo_id=MODEL_REPO_ID if RUN_REAL_L32_FOLLOWUP else "mock/gemma-like",
    model_revision=(
        MODEL_REVISION
        if RUN_REAL_L32_FOLLOWUP
        else "mockrevision0000000000000000000000000000"
    ),
    d_model=EXPECT_D_MODEL if RUN_REAL_L32_FOLLOWUP else MOCK_D_MODEL,
    layer=CAUSAL_LAYER,
    scale=LENS_FITTED_SCALE,
)
L32_VALIDATION = validate_discovered_lens(DISCOVERED, L32_EXPECTATIONS)
L32_LENSES = load_published_lenses([DISCOVERED.spec()], L32_EXPECTATIONS)
L32_LENS = L32_LENSES.lens
L32_LENS_CHECKSUM = DISCOVERED.lens_checksum

print("ARTIFACT DISCOVERY — resolved, never assumed")
print(f"  extension run    {RESOLVED_EXTENSION_RUN_DIR}")
print(f"  report           {Path(DISCOVERED.report_path).name}")
print(f"  sidecars scanned {DISCOVERED.discovery_evidence['extension_sidecars_scanned']}")
print(f"  matched          {DISCOVERED.discovery_evidence['matching_sidecars']}")
print(f"  resolved lens    {Path(DISCOVERED.lens_path).name}")
print(f"  file sha256      {DISCOVERED.lens_checksum}")
print(f"  report agrees    "
      f"{DISCOVERED.discovery_evidence['report_checksum_for_layer'] == DISCOVERED.lens_checksum}")
print(f"  scale            {DISCOVERED.scale}")
print()
print(format_lens_report(L32_LENSES))
'''
)

markdown(
    """
### 5b. The refusals, exercised rather than asserted

"We would have refused a wrong artifact" is a claim. This cell makes the
refusals happen, in MOCK, on artifacts written to be broken: a lens whose bytes
no longer match its recorded digest (a partial Drive sync), a report that
disagrees with the sidecar, a sidecar that fails its own checksum, two sidecars
claiming one layer, a sidecar for the wrong layer or the wrong scale, and an
artifact recorded `validated: false`.
"""
)

code(
    '''
# 5b. Watch every discovery refusal happen. MOCK only — nothing to break on Drive.
DISCOVERY_REFUSALS = {}
if RUN_REAL_L32_FOLLOWUP:
    print("skipped: the refusal drills run against deliberately broken MOCK "
          "artifacts and are never written near a real run")
else:
    from jlens.mmpilot.mock import MOCK_EXTENSION_DEFECTS

    for _defect in MOCK_EXTENSION_DEFECTS:
        _built = build_mock_extension_run(
            SCRATCH / "defects" / _defect,
            layer=CAUSAL_LAYER,
            scale=LENS_FITTED_SCALE,
            d_model=MOCK_D_MODEL,
            corrupt=_defect,
        )
        try:
            _found = discover_published_l32_lens(
                _built["root"], layer=CAUSAL_LAYER, expected_scale=LENS_FITTED_SCALE
            )
            validate_discovered_lens(_found, _built["expectations"])
        except (ArtifactDiscoveryRefused, PublishedLensRefused) as _error:
            DISCOVERY_REFUSALS[_defect] = type(_error).__name__
            print(f"  refused  {_defect:20s} {type(_error).__name__}")
        else:
            raise RuntimeError(
                f"a {_defect!r} artifact was accepted; the refusal path is broken"
            )

    for _name, _kwargs in (
        ("mock_mode_report", {"mode": "mock"}),
        ("no_go_verdict", {"verdict": "EARLY_LAYER_CALIBRATION_NO_GO"}),
        ("nothing_published", {"publish": False}),
    ):
        _built = build_mock_extension_run(
            SCRATCH / "defects" / _name,
            layer=CAUSAL_LAYER, scale=LENS_FITTED_SCALE, d_model=MOCK_D_MODEL,
            **_kwargs,
        )
        try:
            discover_published_l32_lens(
                _built["root"], layer=CAUSAL_LAYER, expected_scale=LENS_FITTED_SCALE
            )
        except ArtifactDiscoveryRefused as _error:
            DISCOVERY_REFUSALS[_name] = type(_error).__name__
            print(f"  refused  {_name:20s} {type(_error).__name__}")
        else:
            raise RuntimeError(f"a {_name!r} report was accepted")

    print(f"\\n{len(DISCOVERY_REFUSALS)} refusal paths exercised")
'''
)

# ======================================== 6. comparability, BEFORE the budget

markdown(
    """
## 6. Comparability with the completed L35 study — decided before any pass

The completed three-modality run asked:

```
Question: which one of these is present: bird, cat, giraffe, microwave,
toilet, zebra? Answer with exactly one word.
Answer:
```

Every candidate — including anything a transfer would move *toward* — was in the
model's own input. That result is **valid**, as candidate-conditioned cross-modal
causal steering. It is not a result about an open question.

This study asks under `mmpilot.open_entity_identification.v1`:

```
What is present in the evidence? Answer with its name.
Answer:
```

The domain-neutral protocol is the correct one, not a convenience: the six-concept
SpokenCOCO set contains `toilet` and `microwave`, and
`open_animal_identification.v1` refuses a non-animal candidate outright rather
than measuring what the model says when asked for an animal that is not there.

**These are different measurements.** The cell below reads the completed run's
*own recorded protocol* — not the constant in the current code — and decides
whether a paired L35 reference is required. When it is:

* the completed L35 numbers stay available as **read-only historical context**;
* they are never differenced against, pooled with, or ranked beside anything
  measured here;
* any cross-layer comparison is **refused** until `RUN_PAIRED_L35_REFERENCE`
  measures L35 under this protocol, on identical samples, targets, controls and
  scoring.

This is stated here, before section 10 asks you to confirm what the reference
costs.
"""
)

code(
    '''
# 6. Read the completed run read-only and decide comparability.
from jlens.mmpilot.l32_followup import (
    COMPLETED_STUDY_PROMPT_PROTOCOL,
    PairedReferenceRequired,
    assert_paired_reference_available,
    prompt_protocol_comparability,
)

COMPLETED_SUMMARY = None
COMPLETED_SUMMARY_CHECKSUM = None
if RUN_REAL_L32_FOLLOWUP:
    from jlens.metadata import file_sha256

    _candidates = [
        Path(COMPLETED_TRANSFER_RUN_DIR)
        / "native_audio_transfer_summary_capability_filtered_v2.json",
        Path(COMPLETED_TRANSFER_RUN_DIR) / "native_audio_transfer_summary.json",
    ]
    _found = [path for path in _candidates if path.is_file()]
    if not _found:
        raise RuntimeError(
            f"no completed-run summary under {COMPLETED_TRANSFER_RUN_DIR}. The "
            "comparability decision is read from the completed artifacts, never "
            "assumed."
        )
    COMPLETED_SUMMARY_PATH = _found[0]
    COMPLETED_SUMMARY = json.loads(
        COMPLETED_SUMMARY_PATH.read_text(encoding="utf-8")
    )
    COMPLETED_SUMMARY_CHECKSUM = file_sha256(str(COMPLETED_SUMMARY_PATH))
    print(f"read-only: {COMPLETED_SUMMARY_PATH.name}")
    print(f"checksum:  {COMPLETED_SUMMARY_CHECKSUM}")
else:
    # The MOCK stands in for the completed run's own record: it says, as the
    # real artifact does, that the run was scored under the candidate-listed
    # protocol. The comparability decision must come out the same way.
    COMPLETED_SUMMARY = {
        "run_dir": "mock/completed_three_modality_run",
        "selection_fingerprint": {
            "capability_protocol": COMPLETED_STUDY_PROMPT_PROTOCOL,
            "n_candidates_scored": N_CONCEPTS,
        },
        "verdicts": {"E_overall": {"verdict": "THREE_MODALITY_GO"}},
    }
    COMPLETED_SUMMARY_CHECKSUM = "sha256:mock-completed-summary"
    print("MOCK: synthetic completed-run record, candidate-listed protocol")

COMPARABILITY = prompt_protocol_comparability(COMPLETED_SUMMARY)
print()
print("=" * 72)
print("PROMPT-PROTOCOL COMPARABILITY")
print("=" * 72)
print(f"  completed run asked under   {COMPARABILITY['completed_run_protocol']}")
print(f"  this study asks under       {COMPARABILITY['new_protocol']}")
print(f"  protocols match             {COMPARABILITY['protocols_match']}")
print(f"  paired reference required   {COMPARABILITY['paired_reference_required']}")
print()
print(COMPARABILITY["statement"])

if COMPARABILITY["paired_reference_required"]:
    print()
    print("CONSEQUENCE FOR SECTION 10's BUDGET:")
    print("  A numerical L32-vs-L35 comparison needs RUN_PAIRED_L35_REFERENCE.")
    print("  Without it this notebook still produces every L32 verdict — verdict E")
    print("  is a statement about layer 32 alone and never needs the reference —")
    print("  but it REFUSES to print a cross-layer comparison.")

# Demonstrated, not asserted: the refusal fires when the comparison is asked
# for without the reference.
try:
    assert_paired_reference_available(COMPARABILITY, paired_reference_ran=False)
    CROSS_LAYER_ALLOWED_WITHOUT_REFERENCE = True
except PairedReferenceRequired as _error:
    CROSS_LAYER_ALLOWED_WITHOUT_REFERENCE = False
    print()
    print("refusal check: a cross-layer comparison without the reference is refused")
'''
)

# ============================================== 7. the open prompt protocol

markdown(
    """
## 7. The open prompt — candidates are scored, never shown

The neutral question and the scored candidate answers are **separate objects**.
Only the question is ever built into a prompt; the candidate strings go to the
external teacher-forced scorer and nowhere else.

Recorded and verified below: protocol name and version, the exact model-visible
prompt, its hash, the candidate-visibility audit, the candidate token ids, the
complete-sequence scoring version, and the prompt-protocol digest.

**Two invariants that pull opposite ways, both checked.** Candidate *order* must
not move the prompt hash — the candidates are not in the prompt. The candidate
*set* must move the fingerprint — a run that scored a different set measured a
different thing.

Candidate-order invariance of the *prompt* is trivially satisfied here and is
recorded as such; it is not claimed as a control, because there is no order in
the prompt to control for. The completed study's canonical/reversed option pair
does not apply.

**The evidence keeps its semantics.** The caption, the photograph and the
recording still contain the concept — that is the evidence. Nothing is stripped
from them. What is absent is the *candidate list in the question*.
"""
)

code(
    '''
# 7. Build the open prompt, audit candidate visibility, fingerprint it.
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
'''
)

markdown(
    """
The candidate set is not decided yet — it comes out of the data in section 8,
in the ranking order that fixes the focal concepts. Section 9 then runs the
candidate-visibility audit against that set. Nothing about the *question*
depends on it, which is the property under test.
"""
)

# ============================================================= 8. the data

markdown(
    """
## 8. The frozen SpokenCOCO selection design

Reused exactly where it is scientifically compatible:

* one synchronized group per photograph — so one recording per photograph too;
* image-disjoint and synchronized-group-disjoint train/test partitions;
* source-training images disjoint from held-out target images;
* the photograph/recording is the independent causal unit;
* no sibling-caption pseudoreplication;
* exact media checksums;
* text, image and spoken audio kept separate — **no transcript ever substitutes
  for the recording**.

Where the prior source and target media can be identified from the completed
run's artifacts, they are reused so the layer comparison is paired. That is
**verified from the artifacts**, not assumed: the cell prints how many of the
prior sample ids were matched, and says plainly when it could not match them.
"""
)

code(
    '''
# 8. Build the subset under the frozen selection design. CPU only.
from datetime import datetime, timezone

from jlens.mmpilot import manifest as manifest_module

MOCK_WORLD = None
if not RUN_REAL_L32_FOLLOWUP:
    from jlens.mmpilot.mock import MockWorld, build_mock_dataset

    MOCK_WORLD = MockWorld({
        "bus": ("bus", "buses"),
        "cat": ("cat", "cats"),
        "clock": ("clock", "clocks"),
        "dog": ("dog", "dogs"),
        "pizza": ("pizza", "pizzas"),
        "zebra": ("zebra", "zebras"),
    })
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

ORIGINAL_MANIFEST_CHECKSUM = manifest_module.manifest_checksum(MANIFEST_PATH)
IMAGE_ROOTS = [Path(IMAGE_MEDIA_ROOT)]
AUDIO_ROOTS = [Path(AUDIO_MEDIA_ROOT)]

RUN_ID = (
    f"mml32_{CONFIG.mode}_"
    f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
)
RUN_DIR = Path(os.environ.get("MMPILOT_RUN_DIR") or (RESOLVED_RUNS_ROOT / RUN_ID))
_offending = sorted({
    prefix
    for prefix in PROTECTED_RUN_PREFIXES
    for part in RUN_DIR.parts
    if part.startswith(prefix)
})
if _offending:
    raise RuntimeError(
        f"{RUN_DIR} is inside a completed run namespace ({_offending}). "
        "Completed runs are evidence and are never written into; this study "
        "creates its own mml32_* directory."
    )
RUN_DIR.mkdir(parents=True, exist_ok=True)
print(f"run directory {RUN_DIR}")
print(f"original manifest checksum {ORIGINAL_MANIFEST_CHECKSUM}")
'''
)

code(
    '''
# 8b. Derive the evidence join, rank concepts, and take the top six IN ORDER.
from jlens.mmpilot import evidence as evidence_module
from jlens.mmpilot import expansion as expansion_module
from jlens.mmpilot.concepts import discover_category_universe
from jlens.mmpilot.selection import select_focal_concepts, unrelated_control_assignment
from jlens.mmpilot.store import payload_checksum

SEARCH_ROOTS = sorted({str(r) for r in IMAGE_ROOTS + AUDIO_ROOTS if r.is_dir()})
if RUN_REAL_L32_FOLLOWUP:
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
UNIVERSE = discover_category_universe(ANNOTATION_SOURCES)
EVIDENCE_CONFIG = evidence_module.config_from_specs(UNIVERSE.specs)
CONCEPT_CANDIDATES = UNIVERSE.lexicon()

_raw = json.loads(Path(MANIFEST_PATH).read_text(encoding="utf-8"))
BASELINE = manifest_module.normalize_manifest(
    _raw,
    manifest_module.inspect_manifest(_raw),
    image_roots=IMAGE_ROOTS,
    audio_roots=AUDIO_ROOTS,
    source_checksum=ORIGINAL_MANIFEST_CHECKSUM,
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
GROUPS = EXPANSION.groups
_payload, EXPANSION_STATUS = expansion_module.persist_expanded_manifest(
    RUN_DIR / "expanded_manifest.json",
    EXPANSION,
    original_checksum=ORIGINAL_MANIFEST_CHECKSUM,
    conversion={
        "converter": "jlens.mmpilot.expansion.build_expanded_manifest",
        "search_roots": SEARCH_ROOTS,
        "evidence_rule": "visual_annotation_AND_caption_lexicon",
        "evidence_lexicon_hash": EVIDENCE_CONFIG.lexicon_hash,
        "reads_only": True, "media_redownloaded": False, "audio_transcribed": False,
    },
)
DERIVED_MANIFEST_CHECKSUM = _payload.get(
    "expanded_manifest_checksum", ORIGINAL_MANIFEST_CHECKSUM
)
print(EXPANSION_STATUS)
print(f"synchronized groups {len(GROUPS)}")
print(f"derived manifest checksum {DERIVED_MANIFEST_CHECKSUM}")

EVIDENCE_INDEX = evidence_module.build_evidence_index(
    GROUPS, tuple(CONCEPT_CANDIDATES), EVIDENCE_CONFIG
)
REQUIREMENTS = expansion_module.ConceptRequirements(
    min_distinct_images=N_TRAIN_POSITIVE_IMAGES + N_TEST_POSITIVE_IMAGES,
    min_groups=N_TRAIN_POSITIVE_IMAGES + N_TEST_POSITIVE_IMAGES,
    min_train_positives=N_TRAIN_POSITIVE_IMAGES,
    min_test_positives=N_TEST_POSITIVE_IMAGES,
)
RANKING = expansion_module.rank_concepts(
    GROUPS,
    CONCEPT_CANDIDATES,
    requirements=REQUIREMENTS,
    groups_per_concept=N_TRAIN_POSITIVE_IMAGES + N_TEST_POSITIVE_IMAGES,
    max_groups_per_image=PROFILE.max_groups_per_image,
    seed=SPLIT_SEED,
    evidence_config=EVIDENCE_CONFIG,
    profile=PROFILE,
    evidence_index=EVIDENCE_INDEX,
)
RANKED_CONCEPTS = [row["concept"] for row in RANKING]
SELECTED_NAMES = expansion_module.select_concepts(
    RANKING, n_concepts=N_CONCEPTS, max_concepts=N_CONCEPTS,
    requirements=REQUIREMENTS,
)
FOCAL_CONCEPTS, NON_FOCAL_CONCEPTS = select_focal_concepts(
    SELECTED_NAMES, n_focal=N_FOCAL_CONCEPTS
)
UNRELATED_CONTROLS = unrelated_control_assignment(FOCAL_CONCEPTS, NON_FOCAL_CONCEPTS)
CONFIG.concepts = tuple(SELECTED_NAMES)
CONFIG.causal_concepts = tuple(FOCAL_CONCEPTS)

print()
print("SELECTION — fixed before any model result exists")
print(f"  selected (ranking order) {SELECTED_NAMES}")
print(f"  focal causal concepts    {FOCAL_CONCEPTS}")
print(f"  non-focal (controls)     {NON_FOCAL_CONCEPTS}")
for _focal, _control in sorted(UNRELATED_CONTROLS.items()):
    print(f"    {_focal:12s} -> {_control}")
'''
)

code(
    '''
# 8c. The one-group-per-image subset, its leakage check and its disjointness.
SUBSET = manifest_module.build_subset(
    GROUPS,
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
IMAGE_OVERLAP = sorted(
    {r["image_id"] for r in _train} & {r["image_id"] for r in _test}
)
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

print("UNIQUE-IMAGE SUBSET")
print(f"  synchronized groups   {N_TOTAL_GROUPS}")
print(f"  distinct images       {N_DISTINCT_IMAGES}")
print(f"  distinct recordings   {N_DISTINCT_RECORDINGS}")
print(f"  groups per image      {N_TOTAL_GROUPS / max(1, N_DISTINCT_IMAGES):.2f}")
print(f"  sibling groups excluded at selection: {N_SIBLINGS_EXCLUDED}")
print(f"  split leakage check   {LEAKAGE['ok']}")
print(f"  train/held-out image overlap {len(IMAGE_OVERLAP)}")
for _concept in SELECTED_NAMES:
    _tr = len({r["image_id"] for r in _train if r["concept"] == _concept})
    _te = len({r["image_id"] for r in _test if r["concept"] == _concept})
    print(f"    {_concept:12s} train images {_tr}  held-out images {_te}")

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
SPLIT_PROVENANCE_CHECKSUM = payload_checksum(SPLIT_PROVENANCE)
print(f"\\nsplit provenance checksum {SPLIT_PROVENANCE_CHECKSUM}")
'''
)

code(
    '''
# 8d. Are the prior source/target media reusable? Verified, never assumed.
PAIRING_AUDIT = {
    "checked": False, "prior_sample_ids": 0, "matched": 0,
    "paired_media": False, "note": "",
}
_prior = (COMPLETED_SUMMARY or {}).get("selection_fingerprint") or {}
_prior_ids = set()
for _key in ("source_sample_ids", "target_sample_ids"):
    _value = _prior.get(_key)
    if isinstance(_value, (list, tuple)):
        _prior_ids.update(str(item) for item in _value)

_ours = {
    str(group["group_id"])
    for split in ("train", "test")
    for group in SUBSET["splits"][split]
}
if _prior_ids:
    PAIRING_AUDIT.update({
        "checked": True,
        "prior_sample_ids": len(_prior_ids),
        "matched": len(_prior_ids & _ours),
        "paired_media": _prior_ids <= _ours,
    })
    PAIRING_AUDIT["note"] = (
        "the prior study's media are all present in this subset, so the layer "
        "comparison is paired at the media level"
        if PAIRING_AUDIT["paired_media"]
        else "the prior media are only partly recoverable; the comparison is "
             "NOT paired at the media level and is reported as such"
    )
else:
    PAIRING_AUDIT["note"] = (
        "the completed run's artifacts record no per-sample source/target ids, "
        "so prior media cannot be verified as reused. This is reported, not "
        "assumed away: the paired L35 reference (section 16) runs on THIS "
        "subset, which is what makes it paired."
    )
print(f"prior sample ids found : {PAIRING_AUDIT['prior_sample_ids']}")
print(f"matched in this subset : {PAIRING_AUDIT['matched']}")
print(f"paired at media level  : {PAIRING_AUDIT['paired_media']}")
print(PAIRING_AUDIT["note"])
'''
)

# ================================================= 9. capability admissibility

markdown(
    """
## 9. Capability admissibility — the predeclared gate, not a new one

Only concepts that passed the completed study's **three-modality behavioral
capability gate** may support a claim here. The prior exclusion is preserved:
`zebra` cleared the gate in text (8/8) and image (8/8) but not in spoken audio
(5/8 against a 70% threshold), and it is `CAPABILITY_INELIGIBLE`.

An excluded concept:

* stays in the raw tables as the predeclared diagnostic it is;
* is labelled, with the arithmetic that rejected it;
* never enters a supporting-cell list or a verdict;
* and is **never replaced** by another concept after results are seen.

The gate is re-measured under *this* prompt protocol in section 12 — a
capability result under a candidate-listed question does not transfer to an open
one — but the *concept set* and the *rule* are the completed study's, fixed
before anything runs here.
"""
)

code(
    '''
# 9a. The candidate-visibility audit and the prompt-protocol fingerprint.
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
        model_revision=MODEL_REVISION if RUN_REAL_L32_FOLLOWUP else "mock",
        processor_revision=MODEL_REVISION if RUN_REAL_L32_FOLLOWUP else "mock",
        audio_protocol_fingerprint=AUDIO_PROTOCOL_FINGERPRINT_EXPECTED,
    )

print("candidate visibility, per modality:")
for _modality, _built in PROMPT_RECORDS.items():
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

# Order must not move the prompt hash.
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
print("\\ncandidate-order invariance of the prompt hash: holds")
print("  (recorded as a property, NOT claimed as a control: an open prompt has")
print("   no candidate order to control for)")

# The candidate SET must move the fingerprint.
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
print("candidate-set sensitivity of the fingerprint: holds")

print(f"\\nscoring          {CANDIDATE_SCORING_VERSION}")
print(f"visibility rule  {CANDIDATE_VISIBILITY_RULE}")
print(f"digest (text)    {PROMPT_FINGERPRINTS['text']['prompt_protocol_digest']}")
'''
)

code(
    '''
# 9b. Print the predeclared admissibility rule and the prior exclusions.
from jlens.mmpilot.admissibility import (
    CLAIM_ADMISSIBILITY_RULE_VERSION,
    PRINCIPAL_REQUIRED_MODALITIES,
    admissibility_rule_record,
)

ADMISSIBILITY_RULE = admissibility_rule_record(
    threshold=CAPABILITY_THRESHOLD, principal_three_modality=True
)
PRIOR_INELIGIBLE = []
_prior_verdicts = (COMPLETED_SUMMARY or {}).get("verdicts") or {}
for _name, _verdict in sorted(_prior_verdicts.items()):
    _adm = (_verdict or {}).get("capability_admissibility") or {}
    PRIOR_INELIGIBLE.extend(_adm.get("excluded_concept_names") or [])
PRIOR_INELIGIBLE = sorted(set(PRIOR_INELIGIBLE))

print(f"rule version        {CLAIM_ADMISSIBILITY_RULE_VERSION}")
print(f"required modalities {list(PRINCIPAL_REQUIRED_MODALITIES)}")
print(f"threshold           {CAPABILITY_THRESHOLD}")
print(f"prior ineligible    {PRIOR_INELIGIBLE or '(none recorded in the artifacts)'}")
print()
print(ADMISSIBILITY_RULE.get("no_post_hoc_replacement", ""))
'''
)

# ================================================== 10. budgets and gates

markdown(
    """
## 10. Exact pass budgets — printed before a model is loaded

Three numbers: the **L32-only** run, any **paired L35 reference**, and the
**total**.

The L4 runtime estimate is **derived from the completed native-audio run's own
unit-file inter-arrival times**, not from an assumed 0.5 s/pass. Those files are
the only timing metadata the completed run left behind; gaps above a cutoff are
dropped as runtime disconnects, so the number estimates *marginal* per-pass cost
and understates total elapsed wall clock. When the completed run is not
reachable from this runtime — which is the MOCK case, and the case on a fresh
Colab session that has not mounted Drive — a **range** is printed instead and
labelled as one. No point estimate is invented.
"""
)

code(
    '''
# 10. Budgets, the derived runtime, and the confirmation gates.
from jlens.mmpilot.l32_followup import (
    FollowupBudget,
    derive_seconds_per_pass,
    format_budget,
)
from jlens.mmpilot.tri_modal import estimate_stage_passes

_n_groups = N_TOTAL_GROUPS
_n_capability = min(
    _n_groups, CONFIG.max_capability_groups_per_concept * len(SELECTED_NAMES)
)


def _stage(layer, stage):
    """One stage's passes at one layer.

    ``n_candidate_orders=1`` because the open prompt is asked once: the
    canonical/reversed option pair the completed study ran exists to control
    candidate *order in the prompt*, and this prompt has no candidates in it.
    """
    return estimate_stage_passes(
        n_concepts=len(SELECTED_NAMES),
        n_focal_concepts=len(FOCAL_CONCEPTS),
        modalities=MODALITIES,
        layers=(layer,),
        causal_layers=(layer,),
        n_total_groups=_n_groups,
        n_capability_groups=_n_capability,
        n_targets_per_cell=CONFIG.n_target_examples,
        alphas=CONFIG.alphas,
        stage=stage,
        n_candidate_orders=1,
        d_model=EXPECT_D_MODEL if RUN_REAL_L32_FOLLOWUP else MOCK_D_MODEL,
    )


BUDGET_L32_A = _stage(CAUSAL_LAYER, "A")
BUDGET_L32_B = _stage(CAUSAL_LAYER, "B")
BUDGET_L35_A = _stage(REFERENCE_LAYER, "A")
BUDGET_L35_B = _stage(REFERENCE_LAYER, "B")

# The reference does NOT re-run capability: the behavioral gate is a property
# of the prompt and the evidence, not of a layer, and its units are already
# stored. It does re-capture activations, because a residual at layer 35 is a
# different tensor from one at layer 32.
REFERENCE_PASSES = BUDGET_L35_A.activation_passes + BUDGET_L35_B.total_passes

BUDGET = FollowupBudget(
    l32_only_passes=BUDGET_L32_A.total_passes + BUDGET_L32_B.total_passes,
    paired_l35_passes=REFERENCE_PASSES if RUN_PAIRED_L35_REFERENCE else 0,
    l32_units={
        **BUDGET_L32_A.estimated_units,
        **{
            key: BUDGET_L32_A.estimated_units.get(key, 0) + value
            for key, value in BUDGET_L32_B.estimated_units.items()
        },
    },
    paired_l35_units=dict(BUDGET_L35_B.estimated_units),
    estimated_drive_bytes=(
        BUDGET_L32_A.estimated_drive_bytes
        + BUDGET_L32_B.estimated_drive_bytes
        + (
            BUDGET_L35_A.estimated_drive_bytes + BUDGET_L35_B.estimated_drive_bytes
            if RUN_PAIRED_L35_REFERENCE
            else 0
        )
    ),
)
print(f"L32 capability   {BUDGET_L32_A.capability_passes:>8,} passes")
print(f"L32 activations  {BUDGET_L32_A.activation_passes:>8,} passes")
print(f"L32 causal clean {BUDGET_L32_B.causal_clean_passes:>8,} passes")
print(f"L32 causal edit  {BUDGET_L32_B.causal_intervention_passes:>8,} passes")
print(f"L32 causal cells {BUDGET_L32_B.n_causal_cells} "
      f"(off-diagonal only), {BUDGET_L32_B.n_conditions_per_target} conditions "
      f"per target")
print()
TIMING = derive_seconds_per_pass(COMPLETED_TRANSFER_RUN_DIR)
print(format_budget(
    BUDGET, TIMING, paired_reference_planned=RUN_PAIRED_L35_REFERENCE
))
'''
)

code(
    '''
# 10b. The gates. Nothing below this cell runs until they are all satisfied.
if COMPARABILITY["paired_reference_required"] and not RUN_PAIRED_L35_REFERENCE:
    print("NOTE — the paired L35 reference is METHODOLOGICALLY REQUIRED for any")
    print("cross-layer comparison, and it is NOT planned in this run.")
    print("  * every L32 verdict (A-E) is still produced; verdict E is about")
    print("    layer 32 alone and never needs the reference;")
    print("  * section 17 will REFUSE to print an L32-vs-L35 comparison.")
    print("  To measure it, set RUN_PAIRED_L35_REFERENCE and")
    print("  CONFIRM_PAIRED_REFERENCE_BUDGET after reading the budget above.")
    print()

if RUN_PAIRED_L35_REFERENCE and not CONFIRM_PAIRED_REFERENCE_BUDGET:
    raise RuntimeError(
        f"RUN_PAIRED_L35_REFERENCE is True but CONFIRM_PAIRED_REFERENCE_BUDGET "
        f"is False. The reference costs {BUDGET.paired_l35_passes:,} additional "
        "model passes; confirm the budget printed above."
    )
if RUN_MODEL_STAGES and not CONFIRM_MODEL_LOAD:
    raise RuntimeError(
        "RUN_MODEL_STAGES is True but CONFIRM_MODEL_LOAD is False (~16 GB)."
    )
if RUN_MODEL_STAGES and not CONFIRM_CAUSAL_BUDGET:
    raise RuntimeError(
        f"RUN_MODEL_STAGES is True but CONFIRM_CAUSAL_BUDGET is False. The "
        f"L32 run costs {BUDGET.l32_only_passes:,} model passes."
    )

print(f"model stages enabled         {MODEL_STAGES_ENABLED}")
print(f"paired L35 reference enabled {PAIRED_REFERENCE_ENABLED}")
'''
)

# =================================================== 11. model and invariance

markdown(
    """
## 11. Load the model, then the invariance gate — in **all three** modalities

Capture must be a no-op and a zero-coefficient edit must reproduce the clean
scoring, **separately in each modality**. A gate that passed on text says nothing
about whether an image or audio forward pass survives the same hook. A modality
whose checks fail is refused causal work rather than recorded with a weaker
result.
"""
)

code(
    '''
# 11. Preflight, backend, invariance.
MODEL = None
BACKEND = None
AUDIO_PROTOCOL = None
MODEL_REVISION_USED = None
PROCESSOR_REVISION_USED = None
AVAILABLE_MODALITIES = []
INVARIANCE = None
INVARIANT_MODALITIES = []
if not MODEL_STAGES_ENABLED:
    print("skipped: RUN_MODEL_STAGES, CONFIRM_MODEL_LOAD and CONFIRM_CAUSAL_BUDGET")
    print("are not all True. Nothing below this cell computes a result.")
else:
    if RUN_REAL_L32_FOLLOWUP:
        import getpass

        from jlens.mmpilot.preflight import check_call_contracts
        from jlens.mmpilot.real_backend import build_real_backend
        from jlens.mmpilot.tri_modal import assert_audio_protocol

        _failures = check_call_contracts()
        if _failures:
            raise RuntimeError(
                "call-signature contracts failed:\\n  - " + "\\n  - ".join(_failures)
            )
        print("call-signature contracts: all bind")

        if not os.environ.get("HF_TOKEN"):
            _token = getpass.getpass("HF_TOKEN (input hidden): ").strip()
            if not _token:
                raise RuntimeError("a Hugging Face token is required")
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
        MODEL_REVISION_USED = BUNDLE.model_revision
        PROCESSOR_REVISION_USED = BUNDLE.processor_revision
        if BUNDLE.audio_interface is None:
            raise RuntimeError(
                "the native spoken-audio path did not resolve: "
                f"{BUNDLE.audio_blocked_reason}. This study is about spoken "
                "audio and does not silently degrade to text and image."
            )
        AUDIO_PROTOCOL = assert_audio_protocol(
            BUNDLE.audio_interface,
            expected_fingerprint=AUDIO_PROTOCOL_FINGERPRINT_EXPECTED,
        )
    else:
        from jlens.mmpilot.mock import MockPilotBackend

        BACKEND = MockPilotBackend(MOCK_WORLD, supports_audio=True)
        MODEL_REVISION_USED = "mock"
        PROCESSOR_REVISION_USED = "mock"
        AUDIO_PROTOCOL = {
            "protocol_version": AUDIO_PROTOCOL_VERSION_EXPECTED,
            "protocol_fingerprint": "sha256:mock-audio-protocol",
            "matches_expected_fingerprint": False,
            "note": "MOCK: the audio protocol fingerprint is checked only on the real path",
        }
        print("MOCK backend: three modalities, no processor, no audio tower")

    from jlens.mmpilot.pipeline import available_modalities

    AVAILABLE_MODALITIES, BLOCKED_MODALITIES = available_modalities(BACKEND, CONFIG)
    print(f"available modalities {AVAILABLE_MODALITIES}")
    print(f"blocked modalities   {BLOCKED_MODALITIES}")
    if "spoken_audio" not in AVAILABLE_MODALITIES:
        raise RuntimeError(
            "spoken_audio is unavailable; this study's question is about it."
        )
'''
)

code(
    '''
# 11b. Media loaders, then the per-modality invariance gate at layer 32.
from jlens.mmpilot.pipeline import build_condition_inputs
from jlens.mmpilot.tri_modal import run_invariance_by_modality

MEDIA = None
if BACKEND is not None:
    if RUN_REAL_L32_FOLLOWUP:
        from PIL import Image

        def _load_image(path):
            return Image.open(path).convert("RGB")

        def _load_audio(path):
            import soundfile as sf

            waveform, sample_rate = sf.read(path, dtype="float32")
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)
            return waveform, int(sample_rate)

        MEDIA = {"load_image": _load_image, "load_audio": _load_audio}
    else:
        from jlens.mmpilot.mock import load_mock_media

        MEDIA = {
            "load_image": load_mock_media,
            "load_audio": lambda path: (load_mock_media(path), 16000),
        }

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
    INVARIANT_MODALITIES = [
        modality
        for modality, entry in INVARIANCE["per_modality"].items()
        if entry["passed"]
    ]
    print("INVARIANCE GATE (capture no-op + zero-coefficient edit)")
    for _modality, _entry in sorted(INVARIANCE["per_modality"].items()):
        print(f"  {_modality:13s} passed={_entry['passed']}")
    print(f"  overall passed {INVARIANCE['passed']}")
    if not INVARIANCE["passed"]:
        raise RuntimeError(
            "an invariance check failed. An intervention's effect in a modality "
            "whose hook is not a no-op has no interpretation."
        )
else:
    print("skipped: no backend")
'''
)

# ===================================== 12. fingerprint, capability, activations

markdown(
    """
## 12. Run fingerprint, behavioral capability under the open prompt, activations

The fingerprint binds every scientific configuration field the brief lists —
model and revision, processor revision, Transformers version, audio protocol and
its fingerprint, the L32 artifact path and checksum, physical layer, `d_model`,
hook site, residual convention, final-prompt-token position, dictionary
orientation and normalization, calibration modality, selected scale,
confirmation status and its provenance, publication metadata, the original and
derived manifest checksums, subset and split provenance, the prompt protocol and
its hash and digest, the selected and admissible concepts, the source and target
sample ids, the controls, the alphas, the direction normalization, and the
convergence criterion version.

A change to **any** of them refuses the resume rather than mixing rows.

Capability is re-measured here because an open question is a different ask than a
candidate-listed one. The concept set and the admissibility rule are unchanged.
"""
)

code(
    '''
# 12. Open the store under the full fingerprint, then measure capability.
from jlens.mmpilot.causal import CONTROL_KINDS
from jlens.mmpilot.convergence import CONVERGENCE_CRITERION, CONVERGENCE_PROTOCOL
from jlens.mmpilot.jspace import CONVENTIONS
from jlens.mmpilot.l32_followup import followup_fingerprint
from jlens.mmpilot.pipeline import (
    scientific_fingerprint,
    stage_activations,
    stage_capability,
)
from jlens.mmpilot.store import RunFingerprint, UnitStore
from jlens.mmpilot.tri_modal import TRI_MODAL_VERDICT_VERSION, audio_capability_verdict

SOURCE_SAMPLE_IDS = sorted(
    str(g["group_id"]) for g in SUBSET["splits"]["train"]
)
TARGET_SAMPLE_IDS = sorted(
    str(g["group_id"]) for g in SUBSET["splits"]["test"]
)
_recorded = L32_VALIDATION["recorded"]

FOLLOWUP_FINGERPRINT = followup_fingerprint(
    protocol=L32_FOLLOWUP_PROTOCOL,
    intervention_family=INTERVENTION_FAMILY,
    model_repo_id=MODEL_REPO_ID if RUN_REAL_L32_FOLLOWUP else "mock/gemma-like",
    model_revision=MODEL_REVISION_USED,
    processor_revision=PROCESSOR_REVISION_USED,
    transformers_version=TRANSFORMERS_VERSION,
    audio_protocol_version=AUDIO_PROTOCOL["protocol_version"],
    audio_protocol_fingerprint=AUDIO_PROTOCOL["protocol_fingerprint"],
    lens_path=DISCOVERED.lens_path,
    lens_checksum=DISCOVERED.lens_checksum,
    physical_layer=CAUSAL_LAYER,
    d_model=L32_LENSES.d_model,
    hook_site=CONVENTIONS["hook_site"],
    residual_convention=_recorded.get("residual_convention"),
    final_prompt_token_position=CONVENTIONS["position"],
    dictionary_orientation=CONVENTIONS["dictionary"],
    dictionary_normalization=CONVENTIONS["code_orientation"],
    calibration_modality=_recorded.get("calibration_modality"),
    selected_scale=LENS_FITTED_SCALE,
    confirmation_status="passed" if L32_VALIDATION["passed"] else "failed",
    confirmation_set_checksum=DISCOVERED.extension_sidecar_path,
    publication_metadata_checksum=DISCOVERED.extension_artifact_checksum,
    original_manifest_checksum=ORIGINAL_MANIFEST_CHECKSUM,
    derived_manifest_checksum=DERIVED_MANIFEST_CHECKSUM,
    subset_provenance_checksum=SPLIT_PROVENANCE_CHECKSUM,
    split_provenance_checksum=SPLIT_PROVENANCE_CHECKSUM,
    prompt_protocol=OPEN_PROMPT_PROTOCOL,
    prompt_protocol_version=OPEN_PROMPT_PROTOCOL,
    prompt_hash=PROMPT_RECORDS["text"].prompt_hash,
    prompt_protocol_digest=PROMPT_FINGERPRINTS["text"]["prompt_protocol_digest"],
    selected_concepts=list(SELECTED_NAMES),
    capability_admissible_concepts=list(FOCAL_CONCEPTS),
    source_sample_ids=SOURCE_SAMPLE_IDS,
    target_sample_ids=TARGET_SAMPLE_IDS,
    controls=list(CONTROL_KINDS),
    alphas=list(CONFIG.alphas),
    direction_normalization=CONVENTIONS["direction_normalization"],
    convergence_criterion_version=CONVERGENCE_PROTOCOL,
    convergence_criterion_digest=CONVERGENCE_CRITERION.digest,
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
    model_repo_id=MODEL_REPO_ID if RUN_REAL_L32_FOLLOWUP else "mock/gemma-like",
    model_revision=MODEL_REVISION_USED,
    processor_revision=PROCESSOR_REVISION_USED,
    layers=tuple(CONFIG.layers),
    lens_checksum=L32_LENSES.combined_checksum,
    manifest_checksum=ORIGINAL_MANIFEST_CHECKSUM,
    split_id=SPLIT_SEED,
    intervention_config={
        "alphas": list(CONFIG.alphas),
        "direction_top_k": CONFIG.direction_top_k,
        "causal_layer": CAUSAL_LAYER,
        "off_diagonal_causal_only": True,
        "intervention_family": INTERVENTION_FAMILY,
    },
    selection_config=SELECTION_FINGERPRINT,
    extra={"followup_fingerprint": FOLLOWUP_FINGERPRINT},
)
STORE = UnitStore(RUN_DIR, FINGERPRINT)
print("run state:", STORE.open())
print(f"run fingerprint      {FINGERPRINT.digest}")
print(f"followup fingerprint {FOLLOWUP_FINGERPRINT['fingerprint_digest']}")
print("  no unit from any candidate-listed run can be reused here: the prompt")
print("  protocol, its hash and its digest are all bound into this fingerprint.")
'''
)

code(
    '''
# 12b. Behavioral capability under the OPEN prompt, then the admissibility gate.
CAPABILITY = None
CAPABILITY_VERDICT = None
CAPABILITY_OUTCOME = None
if BACKEND is None:
    print("skipped: no backend")
else:
    CAPABILITY_OUTCOME, CAPABILITY = stage_capability(
        BACKEND, STORE, SUBSET, CONFIG, MEDIA,
        modalities=AVAILABLE_MODALITIES,
        questions=[OPEN_QUESTION],
    )
    print(CAPABILITY_OUTCOME.line("capability"))
    print("\\nper-concept accuracy under the open prompt:")
    for _concept, _per_modality in sorted(CAPABILITY["per_concept"].items()):
        print(f"  {_concept:12s} " + "  ".join(
            f"{m}={e['n_correct']}/{e['n']}"
            for m, e in sorted(_per_modality.items())
        ))
    CAPABILITY_VERDICT = audio_capability_verdict(
        CAPABILITY,
        selected_concepts=SELECTED_NAMES,
        modalities=AVAILABLE_MODALITIES,
        thresholds=THRESHOLDS,
    )
    STORE.save("metric", "open_prompt_capability_verdict", CAPABILITY_VERDICT)
    print(f"\\ncapability verdict: {CAPABILITY_VERDICT['verdict']}")
    print(CAPABILITY_VERDICT["rationale"])
'''
)

code(
    '''
# 12c. Final-prompt-token residuals at layer 32, under the same open question.
ACTIVATIONS = []
if BACKEND is None:
    print("skipped: no backend")
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
    print(f"distinct images captured: {len({r['image_id'] for r in ACTIVATIONS})}")
'''
)

# ================================= 13. codes and representational diagnostics

markdown(
    """
## 13. J-space codes and the representational diagnostics — **verdict B**

One frozen dictionary built from layer 32's **own** published lens, used, and
released. No basis is fitted, rescaled or substituted.

All six directed modality pairs get image-disjoint nearest-neighbour retrieval,
matched-versus-mismatched separation, weighted sparse-support overlap, the raw
residual baseline, and the shuffled-label distribution with its p95.

**Retrieval excludes the same image identity, not merely the same synchronized
group.** SpokenCOCO ships several captions per photograph and the subset keeps
more than one, so group-level exclusion alone would let a caption reach its own
photograph through a sibling caption.

These are **supporting evidence, not causal evidence**, and verdict B is pooled
over concepts rather than capability-filtered — it discloses the ineligible
concepts instead, and is never sufficient on its own.
"""
)

code(
    '''
# 13. Codes and the six directed pairs at layer 32. Verdict B.
CODES = []
REPRESENTATIONAL = {}
REPRESENTATIONAL_VERDICT = None
if BACKEND is None:
    print("skipped: no backend")
else:
    from jlens.mmpilot.pipeline import (
        build_dictionaries,
        stage_codes,
        stage_representational,
    )
    from jlens.mmpilot.tri_modal import representational_transfer_verdict

    _dictionaries = build_dictionaries(
        L32_LENS,
        (CAUSAL_LAYER,),
        BACKEND,
        device=(
            "cuda" if (RUN_REAL_L32_FOLLOWUP and torch.cuda.is_available()) else "cpu"
        ),
        dtype=torch.float16 if RUN_REAL_L32_FOLLOWUP else torch.float32,
        build_chunk_rows=32768 if RUN_REAL_L32_FOLLOWUP else None,
    )
    _code_outcome = stage_codes(
        STORE, ACTIVATIONS, _dictionaries, CONFIG, lens_checksum=L32_LENS_CHECKSUM
    )
    CODES = _code_outcome.records
    del _dictionaries
    print(f"L{CAUSAL_LAYER}: " + _code_outcome.line("jspace"))

    REPRESENTATIONAL[CAUSAL_LAYER] = stage_representational(
        STORE, ACTIVATIONS, CODES, CONFIG,
        layer=CAUSAL_LAYER,
        modalities=AVAILABLE_MODALITIES,
    )
    from jlens.mmpilot.tri_modal import representational_rows

    print(f"\\n=== layer {CAUSAL_LAYER}: all six directed pairs ===")
    for _row in representational_rows(REPRESENTATIONAL[CAUSAL_LAYER]):
        if not _row["evaluated"]:
            print(f"  {_row['pair']:26s} NOT EVALUATED — {_row['reason']}")
            continue
        print(
            f"  {_row['pair']:26s} n={_row['n_queries']:3d}  "
            f"jspace_top1={_row['jspace_top1']:.3f}  "
            f"raw_top1={_row['raw_top1']:.3f}  "
            f"shuffled_p95={_row['shuffled_p95']:.3f}  "
            f"beats={_row['beats_shuffled']}"
        )
        print(
            f"  {'':26s} separation gap jspace={_row['jspace_separation_gap']:+.3f} "
            f"raw={_row['raw_separation_gap']:+.3f}  "
            f"support_overlap_gap={_row['support_overlap_gap']:+.3f}  "
            f"excluded_same_group={_row['n_excluded_same_group']} "
            f"excluded_same_image_other_group="
            f"{_row['n_excluded_same_image_different_group']}"
        )

    REPRESENTATIONAL_VERDICT = representational_transfer_verdict(
        REPRESENTATIONAL,
        thresholds=THRESHOLDS,
        primary_layer=CAUSAL_LAYER,
        capability=CAPABILITY_VERDICT,
        pooled_concepts=SELECTED_NAMES,
    )
    STORE.save("metric", "l32_representational_verdict", REPRESENTATIONAL_VERDICT)
    print(f"\\nVERDICT B — L32_REPRESENTATIONAL_TRANSFER: "
          f"{REPRESENTATIONAL_VERDICT['verdict']}")
    print(REPRESENTATIONAL_VERDICT["rationale"])
    print("\\n  supporting evidence only; never causal evidence, and never")
    print("  sufficient on its own.")
'''
)

# ================================================== 14. the causal test

markdown(
    """
## 14. The causal test at layer 32 — **verdict C**

Directions are estimated from **source-modality training examples only**:
`ReLU(mean positive code − mean matched-negative code)`, reconstructed through
the frozen dictionary and normalized by the documented convention. No target
activation and no test example of any modality enters a direction.

Every available off-diagonal source→target pair is tested on held-out positive
and matched-negative target examples, with the small alpha sweep and all four
controls — zero, norm-matched random, external unrelated concept, and the raw
residual positive-minus-negative — through the same code path.

Recorded separately: **necessity** (subtraction from a positive) and
**sufficiency** (addition to a matched negative); intended score change, margin
change, unrelated-output disruption, prediction change, and activation-norm
change. Aggregation is by distinct photograph/recording, never by caption group.

**Subtraction is directional subtraction on a residual stream.** It is not
erasure and not projection ablation, and nothing here is described as either.
"""
)

code(
    '''
# 14. Source-only directions, the off-diagonal causal cells, verdict C.
DIRECTIONS = {}
INTERVENTIONS = None
INTERVENTION_RECORDS = []
IMAGE_LEVEL = None
INDEPENDENCE = None
CAUSAL_VERDICT = None
CAUSAL_BREAKDOWN = None
if BACKEND is None:
    print("skipped: no backend")
elif CAPABILITY_VERDICT is None or CAPABILITY_VERDICT["verdict"] != "AUDIO_CAPABILITY_GO":
    print("skipped: the open-prompt capability gate did not pass. The causal")
    print("stage does not run on a capability the model does not have under")
    print("this question, and the design is not narrowed to rescue it.")
else:
    from jlens.mmpilot.independence import (
        audit_image_independence,
        divergence_summary,
        resolve_image_identity,
        summarize_interventions_by_image,
    )
    from jlens.mmpilot.pipeline import stage_causal, stage_directions
    from jlens.mmpilot.l32_reporting import (
        causal_cell_breakdown,
        format_causal_breakdown,
    )
    from jlens.mmpilot.tri_modal import causal_transfer_verdict

    _dictionaries = build_dictionaries(
        L32_LENS,
        (CAUSAL_LAYER,),
        BACKEND,
        device=(
            "cuda" if (RUN_REAL_L32_FOLLOWUP and torch.cuda.is_available()) else "cpu"
        ),
        dtype=torch.float16 if RUN_REAL_L32_FOLLOWUP else torch.float32,
        build_chunk_rows=32768 if RUN_REAL_L32_FOLLOWUP else None,
    )
    _direction_outcome, DIRECTIONS = stage_directions(
        STORE, CODES, ACTIVATIONS, _dictionaries, CONFIG,
        concepts=SELECTED_NAMES,
        modalities=AVAILABLE_MODALITIES,
        lens_checksum=L32_LENS_CHECKSUM,
    )
    del _dictionaries
    print(f"L{CAUSAL_LAYER}: " + _direction_outcome.line("direction"))
    for _record in sorted(
        _direction_outcome.records,
        key=lambda r: (r.get("concept") or "", r.get("source_modality") or ""),
    ):
        if _record["kind"] != "source_concept":
            continue
        print(f"    {_record['concept']:12s} {_record['source_modality']:13s} "
              f"pos={_record['n_source_positive_images']} img  "
              f"neg={_record['n_source_negative_images']} img  "
              f"uses_target_modality_data={_record['uses_target_modality_data']}")

    _causal_outcome, INTERVENTIONS = stage_causal(
        BACKEND, STORE, SUBSET, CODES, ACTIVATIONS, DIRECTIONS, CONFIG, MEDIA,
        concepts=FOCAL_CONCEPTS,
        modalities=AVAILABLE_MODALITIES,
        all_concepts=SELECTED_NAMES,
        unrelated_controls=UNRELATED_CONTROLS,
        question=OPEN_QUESTION,
    )
    INTERVENTION_RECORDS = _causal_outcome.records
    print(f"L{CAUSAL_LAYER}: " + _causal_outcome.line("intervention"))

    _identity = resolve_image_identity(
        [*ACTIVATIONS, *CODES, *INTERVENTION_RECORDS]
    )
    IMAGE_LEVEL = summarize_interventions_by_image(
        INTERVENTION_RECORDS, _identity, group_summary=INTERVENTIONS
    )
    _divergence = divergence_summary(IMAGE_LEVEL)
    INDEPENDENCE = audit_image_independence(
        _identity,
        interventions=INTERVENTION_RECORDS,
        concepts=SELECTED_NAMES,
    )
    STORE.save("metric", "image_independence_audit", INDEPENDENCE)
    STORE.save("metric", f"interventions_image_level_L{CAUSAL_LAYER}", IMAGE_LEVEL)
    print(f"    distinct images {IMAGE_LEVEL['n_distinct_images_overall']}  "
          f"groups {IMAGE_LEVEL['n_groups_overall']}  "
          f"pseudoreplicated rows "
          f"{_divergence['n_rows_pseudoreplicated_at_group_level']}"
          f"/{_divergence['n_rows']}")
    if _divergence["n_rows_pseudoreplicated_at_group_level"]:
        raise RuntimeError(
            "an intervention cell drew more than one observation from one "
            "photograph; refusing to report a pseudoreplicated causal summary."
        )

    CAUSAL_VERDICT = causal_transfer_verdict(
        IMAGE_LEVEL,
        layer=CAUSAL_LAYER,
        focal_concepts=FOCAL_CONCEPTS,
        thresholds=THRESHOLDS,
        name="L32_CAUSAL_TRANSFER",
        capability=CAPABILITY_VERDICT,
    )
    STORE.save("metric", "l32_causal_verdict", CAUSAL_VERDICT)
    print("\\n" + "=" * 72)
    print(f"VERDICT C — L32_CAUSAL_TRANSFER: {CAUSAL_VERDICT['verdict']}")
    print("=" * 72)
    print(CAUSAL_VERDICT["rationale"])
    _adm = CAUSAL_VERDICT["capability_admissibility"]
    print()
    print(f"  fixed focal concepts   {_adm['fixed_concepts']}")
    print(f"  capability-eligible    {_adm['eligible_concepts']}")
    print(f"  CAPABILITY_INELIGIBLE  {_adm['excluded_concept_names']}")
    print(f"\\n{_adm['no_post_hoc_replacement']}")

    # Verdict C is decided on the AUDIO arm; verdict E counts every admissible
    # off-diagonal cell. Both numbers are right and they differ by the
    # text<->image replication cells, so the buckets are printed rather than
    # left to look like a contradiction.
    print()
    CAUSAL_BREAKDOWN = causal_cell_breakdown(CAUSAL_VERDICT)
    print(format_causal_breakdown(CAUSAL_BREAKDOWN, layer=CAUSAL_LAYER))
'''
)

# ============================================ 15. native convergence audit

markdown(
    """
## 15. Native output-convergence audit at layer 32 — **verdict D**

The model's **own** output head, on the same activations the causal analysis
used:

```
logits = lm_head(final_norm(h))            # the live modules, called
logits = softcap * tanh(logits / softcap)  # the model's declared cap
```

No lens, no dictionary, no J-space code, no intervention, no learned probe.

**Agreement with the model's own `unembed` is required before proceeding**, to
float tolerance, probe by probe as singleton batches — comparing a batched
`unembed` against a row-wise stack compares GEMM shapes, not readouts.

The criterion is the **already predeclared two-sided rule** and its thresholds
are not touched — but it is *printed for this audit*. The historical
`CRITERION_TEXT` constant is titled "L35 / L38 / L40" and names layer 35 as the
primary throughout; it is the completed three-modality study's protocol, it is
the wrong document for a single-layer L32 audit, and it is left unedited because
it is that run's record. `format_l32_criterion` states the same thresholds and
the same digest under the correct scope, and says plainly that **no later-layer
trajectory clause is applied here** — that clause needs a second layer and
cannot be evaluated, or claimed, from one point.

Controls are the existing meaningful ones: permuted activations, permuted
candidate-token assignments, and shuffled target labels. All three are printed
individually with the field each was compared on, and a variant that produced
**no record refuses the cell rather than counting as a pass** —
`summarize_controls` skips a variant with no rows, which would otherwise leave
`all_controls_passed` True with the control never having run.

> A literal wrong-layer output head is **not** technically meaningful here:
> Gemma 4 has exactly one final normalization module and one unembedding, shared
> by every layer, so there is no layer-specific readout component to misapply.
> The permuted-activation control is its substitute — same head, different
> sample's residual.

Interpreting layer 32 at all requires the artifact's **passing validation
record**. The scale-100 refusal still stands on its own evidence; what clears it
is the scale-250 confirmation, passed in as a record rather than as a flag.
"""
)

code(
    '''
# 15. The native readout at layer 32, audited against the model's own unembed.
#
# The criterion printed here is written for THIS audit. The historical
# CRITERION_TEXT constant is titled "L35 / L38 / L40" and names layer 35 as the
# primary throughout; it is the completed three-modality study's protocol and is
# deliberately left untouched, because editing it would rewrite that run's
# record. Same thresholds, same digest — only the scope statement differs.
from jlens.mmpilot.convergence import (
    ConvergenceFingerprint,
    ConvergenceStore,
    NativeHead,
    audit_native_head,
    build_population,
    clean_predictions_from_interventions,
    head_from_model,
    resolve_candidate_tokens,
)
from jlens.mmpilot.capability import candidate_token_ids
from jlens.mmpilot.l32_followup import (
    assert_native_head_agrees,
    run_single_layer_convergence,
)
from jlens.mmpilot.l32_reporting import (
    L32_REPORTING_VERSION,
    classification_detail,
    control_rows,
    convergence_cell_rows,
    format_classification,
    format_controls,
    format_convergence_cells,
    format_l32_criterion,
)

CONVERGENCE = None
CONVERGENCE_CLASSIFICATION = None
CONVERGENCE_CONTROLS = None
CONVERGENCE_CELLS = None
CONVERGENCE_DETAIL = None
CONTROL_ROWS = None
HEAD_AUDIT = None
HEAD_AGREEMENT = None
L32_CRITERION_TEXT = format_l32_criterion(layer=CAUSAL_LAYER)
if BACKEND is None or not INTERVENTION_RECORDS:
    print("skipped: the causal stage produced no clean references to score against")
else:
    print(L32_CRITERION_TEXT)

    if RUN_REAL_L32_FOLLOWUP:
        HEAD = head_from_model(MODEL)
        # audit_native_head raises if the comparison RUNS and the gap is wide;
        # assert_native_head_agrees additionally refuses the case where it never
        # ran, which would leave matches_model_unembed as None.
        HEAD_AUDIT = audit_native_head(HEAD, model=MODEL, probes=4)
        HEAD_AGREEMENT = assert_native_head_agrees(HEAD_AUDIT, required=True)
        print(f"native head vs model unembed: matched="
              f"{HEAD_AGREEMENT['matches_model_unembed']}  "
              f"max_abs_diff={HEAD_AGREEMENT['max_abs_difference_vs_model_unembed']}  "
              f"protocol={HEAD_AGREEMENT['protocol']}")
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

    CLEAN_PREDICTIONS = clean_predictions_from_interventions(INTERVENTION_RECORDS)
    POPULATION = build_population(
        activations=ACTIVATIONS,
        clean_predictions=CLEAN_PREDICTIONS,
        capability=CAPABILITY_VERDICT,
        focal_concepts=FOCAL_CONCEPTS,
        layers=(CAUSAL_LAYER,),
    )
    print(f"population {POPULATION['n_units']} units, "
          f"{POPULATION['n_with_clean_reference']} with a clean reference")
    print(f"  admissible   {POPULATION['admissible_concepts']}")
    print(f"  inadmissible {POPULATION['inadmissible_concepts']}")
'''
)

code(
    '''
# 15b. Score the population, apply the frozen criterion, run the controls.
if BACKEND is None or not INTERVENTION_RECORDS:
    print("skipped")
else:
    CONVERGENCE_STORE = ConvergenceStore(
        RUN_DIR / "convergence",
        ConvergenceFingerprint(
            protocol=CONVERGENCE_PROTOCOL,
            completed_run_fingerprint_digest=FINGERPRINT.digest,
            completed_run_dir=str(RUN_DIR),
            model_repo_id=(
                MODEL_REPO_ID if RUN_REAL_L32_FOLLOWUP else "mock/gemma-like"
            ),
            model_revision=MODEL_REVISION_USED,
            processor_revision=PROCESSOR_REVISION_USED,
            layers=(CAUSAL_LAYER,),
            candidate_digest=TOKENIZATION["digest"],
            readout_mode=TOKENIZATION["readout_mode"],
            head_checksum=str(HEAD_AUDIT.get("head_checksum", "")),
            criterion_digest=CONVERGENCE_CRITERION.digest,
            code_version=L32_FOLLOWUP_PROTOCOL,
            extra={"followup_fingerprint": FOLLOWUP_FINGERPRINT["fingerprint_digest"]},
        ),
    )
    CONVERGENCE = run_single_layer_convergence(
        population=POPULATION,
        head=HEAD,
        tokenization=TOKENIZATION,
        head_audit=HEAD_AUDIT,
        store=CONVERGENCE_STORE,
        layer=CAUSAL_LAYER,
        # The scale-250 confirmation, passed as the record it is. Nothing else
        # clears layer 32 for interpretation.
        confirmation_record=L32_VALIDATION,
    )
    CONVERGENCE_CLASSIFICATION = CONVERGENCE["classification"]
    CONVERGENCE_CONTROLS = CONVERGENCE["controls"]
    if not CONVERGENCE["criterion_thresholds_unchanged"]:
        raise RuntimeError(
            "the convergence criterion's thresholds differ from the frozen "
            "ones. They are predeclared and are not revisable now."
        )

    print("=" * 72)
    print(f"VERDICT D — L32_NATIVE_OUTPUT_CONVERGENCE: "
          f"{CONVERGENCE_CLASSIFICATION['classification']}")
    print("=" * 72)

    # Every field below is read by the name summarize_cell actually stores it
    # under. The first version of this cell asked for `unique_top1_rate` and
    # `median_entropy`, which have never existed, and printed None for both —
    # indistinguishable from a measurement that came back empty.
    CONVERGENCE_CELLS = convergence_cell_rows(
        CONVERGENCE["summary"], layer=CAUSAL_LAYER
    )
    print(format_convergence_cells(CONVERGENCE_CELLS, layer=CAUSAL_LAYER))
    print()

    # classify_layer has no `decided_by` field; it records which clauses failed
    # on each side. AMBIGUOUS means both lists are non-empty.
    CONVERGENCE_DETAIL = classification_detail(CONVERGENCE_CLASSIFICATION)
    print(format_classification(CONVERGENCE_DETAIL))
    print()

    # The variants live under per_layer[layer]["controls"][variant]. Reading one
    # level too shallow printed None for all three. A variant that produced no
    # rows is skipped by summarize_controls, leaving `all_controls_passed` True
    # with the control never having run — so completeness is asserted first.
    CONTROL_ROWS = control_rows(CONVERGENCE_CONTROLS, layer=CAUSAL_LAYER)
    print(format_controls(
        CONTROL_ROWS, controls=CONVERGENCE_CONTROLS, layer=CAUSAL_LAYER
    ))
'''
)

# ======================================== 16. the paired L35 reference

markdown(
    """
## 16. The paired L35 reference — same protocol, same samples, same everything

Runs only when `RUN_PAIRED_L35_REFERENCE` **and**
`CONFIRM_PAIRED_REFERENCE_BUDGET` are both `True`.

This is the **smallest necessary** reference condition: layer 35 under *this*
open prompt, on *these* samples, with *these* targets, controls and scoring. It
exists so an L32-vs-L35 statement can be made at all — the completed study's
numbers were produced under a different question and cannot serve.

The layer-35 lens is loaded under **its own** scale (100), because that is what
confirmed it. Two layers fitted at different scales are not one lens, and
`load_published_lenses` refuses to assemble them as one.
"""
)

code(
    '''
# 16. The paired L35 reference under the same open protocol.
REFERENCE_CAUSAL_VERDICT = None
REFERENCE_IMAGE_LEVEL = None
REFERENCE_LENS_REPORT = None
if not PAIRED_REFERENCE_ENABLED:
    print("skipped: the paired L35 reference is not enabled.")
    print("  Set RUN_PAIRED_L35_REFERENCE and CONFIRM_PAIRED_REFERENCE_BUDGET")
    print("  to measure it. Without it, section 17 refuses a cross-layer")
    print("  comparison and every L32 verdict is still produced.")
else:
    from dataclasses import replace as _replace

    from jlens.mmpilot.published_lens import PublishedLensExpectations, PublishedLensSpec

    if RUN_REAL_L32_FOLLOWUP:
        _ref_specs = [PublishedLensSpec(
            layer=REFERENCE_LAYER,
            path=str(Path(REFERENCE_LENS_DIR) / REFERENCE_LENS_FILE),
            expect_sha256=REFERENCE_LENS_SHA256,
        )]
        _ref_expectations = PublishedLensExpectations(
            model_repo_id=MODEL_REPO_ID,
            model_revision=MODEL_REVISION_USED,
            scale_point=REFERENCE_LENS_FITTED_SCALE,
            d_model=EXPECT_D_MODEL,
        )
    else:
        from jlens.mmpilot.mock import build_mock_published_lenses

        _mock_ref = build_mock_published_lenses(
            SCRATCH / "reference_lens",
            layers=(REFERENCE_LAYER,),
            d_model=BACKEND.d_model,
            scale=REFERENCE_LENS_FITTED_SCALE,
        )
        _ref_specs = _mock_ref["specs"]
        _ref_expectations = _mock_ref["expectations"]

    REFERENCE_LENSES = load_published_lenses(_ref_specs, _ref_expectations)
    REFERENCE_LENS_REPORT = REFERENCE_LENSES.to_dict()
    print(format_lens_report(REFERENCE_LENSES))

    _ref_config = _replace(
        CONFIG, layers=(REFERENCE_LAYER,), causal_layers=(REFERENCE_LAYER,)
    )
    _ref_activations = stage_activations(
        BACKEND, STORE, SUBSET, _ref_config, MEDIA,
        modalities=AVAILABLE_MODALITIES,
        retained_concepts=SELECTED_NAMES,
        model_revision=MODEL_REVISION_USED,
        question=OPEN_QUESTION,
    ).records
    _ref_dictionaries = build_dictionaries(
        REFERENCE_LENSES.lens, (REFERENCE_LAYER,), BACKEND,
        device=(
            "cuda" if (RUN_REAL_L32_FOLLOWUP and torch.cuda.is_available()) else "cpu"
        ),
        dtype=torch.float16 if RUN_REAL_L32_FOLLOWUP else torch.float32,
    )
    _ref_codes = stage_codes(
        STORE, _ref_activations, _ref_dictionaries, _ref_config,
        lens_checksum=REFERENCE_LENSES.checksums[REFERENCE_LAYER],
    ).records
    _ref_direction_outcome, _ref_directions = stage_directions(
        STORE, _ref_codes, _ref_activations, _ref_dictionaries, _ref_config,
        concepts=SELECTED_NAMES,
        modalities=AVAILABLE_MODALITIES,
        lens_checksum=REFERENCE_LENSES.checksums[REFERENCE_LAYER],
    )
    del _ref_dictionaries
    _ref_causal_outcome, _ref_summary = stage_causal(
        BACKEND, STORE, SUBSET, _ref_codes, _ref_activations, _ref_directions,
        _ref_config, MEDIA,
        concepts=FOCAL_CONCEPTS,
        modalities=AVAILABLE_MODALITIES,
        all_concepts=SELECTED_NAMES,
        unrelated_controls=UNRELATED_CONTROLS,
        question=OPEN_QUESTION,
    )
    _ref_identity = resolve_image_identity(
        [*_ref_activations, *_ref_codes, *_ref_causal_outcome.records]
    )
    REFERENCE_IMAGE_LEVEL = summarize_interventions_by_image(
        _ref_causal_outcome.records, _ref_identity, group_summary=_ref_summary
    )
    REFERENCE_CAUSAL_VERDICT = causal_transfer_verdict(
        REFERENCE_IMAGE_LEVEL,
        layer=REFERENCE_LAYER,
        focal_concepts=FOCAL_CONCEPTS,
        thresholds=THRESHOLDS,
        name=f"L{REFERENCE_PHYSICAL_LAYER}_PAIRED_REFERENCE_CAUSAL_TRANSFER",
        capability=CAPABILITY_VERDICT,
    )
    STORE.save("metric", "paired_l35_reference_verdict", REFERENCE_CAUSAL_VERDICT)
    print(f"\\nPAIRED L{REFERENCE_PHYSICAL_LAYER} REFERENCE: "
          f"{REFERENCE_CAUSAL_VERDICT['verdict']}")
    print(REFERENCE_CAUSAL_VERDICT["rationale"])
'''
)

# ================================================= 17. verdicts and report

markdown(
    """
## 17. The five verdicts

* **A** `L32_LENS_INTEGRITY` — discovery, validation, scale binding, invariance.
* **B** `L32_REPRESENTATIONAL_TRANSFER` — supporting evidence only.
* **C** `L32_CAUSAL_TRANSFER` — the causal result on admissible cells.
* **D** `L32_NATIVE_OUTPUT_CONVERGENCE` — the frozen two-sided criterion.
* **E** `PRE_CONVERGENCE_CAUSAL_TRANSFER` — the combined claim.

**E is a statement about layer 32 alone**, and it does not need the paired
reference. Its clauses are predeclared: integrity passed, capability valid, at
least one capability-admissible off-diagonal effect with the expected sign,
above its matched-random and external-unrelated controls, specific rather than
global disruption, activation norms sane, evidence on at least **two** distinct
photographs/recordings, layer 32 classified `NOT_CONVERGED`, and controls
passing.

The `CONVERGED` branch is checked **first**, so it can never be masked. An
`AMBIGUOUS` layer 32 can only yield `INCONCLUSIVE`.

The completed study's cross-layer rule (`convergence_verdict`, which needs a
*later* validated layer to be more converged) is deliberately **not** used: this
run audits one layer, and applying a trajectory rule to a single point would be
inventing a comparison.

The report says **"before native direct-readout convergence"**. It never says
"pre-linguistic", "language-free", or "before language exists" — a weak native
readout is a fact about the readout, not about whether linguistic information is
present, and `assert_report_phrasing` refuses a report that claims otherwise.
"""
)

code(
    '''
# 17. Assemble the five verdicts, check the phrasing, write the report.
from jlens.mmpilot.l32_followup import (
    CONVERGENCE_PHRASE,
    adjacent_layer_recommendation,
    assert_report_phrasing,
    lens_integrity_verdict,
    pre_convergence_verdict,
    separate_measured_from_historical,
)

VERDICT_A = lens_integrity_verdict(
    L32_VALIDATION,
    invariance=INVARIANCE,
    discovery=DISCOVERED.to_dict(),
    layer=CAUSAL_LAYER,
    scale=LENS_FITTED_SCALE,
)
VERDICT_B = REPRESENTATIONAL_VERDICT
VERDICT_C = CAUSAL_VERDICT
VERDICT_D = CONVERGENCE_CLASSIFICATION
VERDICT_E = None

if VERDICT_C is not None and VERDICT_D is not None:
    VERDICT_E = pre_convergence_verdict(
        integrity=VERDICT_A,
        causal=VERDICT_C,
        convergence=VERDICT_D,
        controls=CONVERGENCE_CONTROLS or {},
        capability=CAPABILITY_VERDICT or {},
    )

print("=" * 72)
print("THE FIVE VERDICTS")
print("=" * 72)
print(f"  A  L32_LENS_INTEGRITY               {VERDICT_A['verdict']}")
print(f"  B  L32_REPRESENTATIONAL_TRANSFER    "
      f"{(VERDICT_B or {}).get('verdict', 'NOT_EVALUATED')}")
print(f"  C  L32_CAUSAL_TRANSFER              "
      f"{(VERDICT_C or {}).get('verdict', 'NOT_EVALUATED')}")
print(f"  D  L32_NATIVE_OUTPUT_CONVERGENCE    "
      f"{(VERDICT_D or {}).get('classification', 'NOT_EVALUATED')}")
print(f"  E  PRE_CONVERGENCE_CAUSAL_TRANSFER  "
      f"{(VERDICT_E or {}).get('verdict', 'NOT_EVALUATED')}")
if VERDICT_E is not None:
    print()
    print(VERDICT_E["rationale"])
    print()
    for _clause in VERDICT_E["clauses"]:
        print(f"  {'PASS' if _clause['passed'] else 'FAIL'}  "
              f"{_clause['clause']:52s} {_clause['detail']}")
'''
)

code(
    '''
# 17b. The cross-layer comparison — produced only when it is licensed.
CROSS_LAYER = None
try:
    assert_paired_reference_available(
        COMPARABILITY, paired_reference_ran=REFERENCE_CAUSAL_VERDICT is not None
    )
    CROSS_LAYER = {
        "licensed": True,
        "l32": (VERDICT_C or {}).get("verdict"),
        f"l{REFERENCE_PHYSICAL_LAYER}_paired_reference": (
            REFERENCE_CAUSAL_VERDICT or {}
        ).get("verdict"),
        "same_prompt_protocol": True,
        "same_samples": True,
        "note": (
            "both layers were measured under the same open protocol on the same "
            "samples, targets, controls and scoring, so this comparison is paired"
        ),
    }
    print("CROSS-LAYER COMPARISON (licensed — the reference was measured)")
    print(json.dumps(CROSS_LAYER, indent=2))
except PairedReferenceRequired as _error:
    CROSS_LAYER = {"licensed": False, "reason": str(_error)}
    print("CROSS-LAYER COMPARISON: REFUSED")
    print(_error)

MEASURED_VS_HISTORICAL = separate_measured_from_historical(
    measured={
        "prompt_protocol": OPEN_PROMPT_PROTOCOL,
        "layer": CAUSAL_LAYER,
        "causal": (VERDICT_C or {}).get("verdict"),
        "representational": (VERDICT_B or {}).get("verdict"),
        "convergence": (VERDICT_D or {}).get("classification"),
        "paired_reference": (REFERENCE_CAUSAL_VERDICT or {}).get("verdict"),
    },
    historical={
        "run_dir": COMPLETED_TRANSFER_RUN_DIR,
        "summary_checksum": COMPLETED_SUMMARY_CHECKSUM,
        "prompt_protocol": COMPARABILITY["completed_run_protocol"],
        "overall_verdict": (
            ((COMPLETED_SUMMARY or {}).get("verdicts") or {}).get("E_overall") or {}
        ).get("verdict"),
    },
    comparability=COMPARABILITY,
)
print()
print(f"newly measured vs read-only historical: "
      f"may_be_compared_numerically="
      f"{MEASURED_VS_HISTORICAL['may_be_compared_numerically']}")
print(MEASURED_VS_HISTORICAL["note"])
'''
)

# ============================================= 18. recommendation and report

markdown(
    """
## 18. The conditional next step — and the report

Layers 33 and 34 are **not** fitted or tested here. Whether they should be is
decided by the rule below, which was fixed before any result existed:

| state | next step |
|---|---|
| L32 causal **and** `NOT_CONVERGED` | no adjacent sweep required for the core claim; L33/L34 are optional localization |
| L32 noncausal while the **matched** L35 reference is causal | fit L33/L34 — the effect is bracketed between them |
| L32 `CONVERGED` | L33/L34 are later still and cannot establish an earlier result; investigate layers **earlier** than 32 |
| L32 `AMBIGUOUS` | the smallest convergence-resolution study; more samples, not more layers |
"""
)

code(
    '''
# 18. The conditional recommendation, the phrasing check, and the report.
RECOMMENDATION = adjacent_layer_recommendation(
    causal_verdict=(VERDICT_C or {}).get("verdict", "NOT_EVALUATED"),
    classification=(VERDICT_D or {}).get("classification", "NOT_EVALUATED"),
    reference_causal=(REFERENCE_CAUSAL_VERDICT or {}).get("verdict"),
)
print("=" * 72)
print("NEXT STEP")
print("=" * 72)
print(f"  recommendation {RECOMMENDATION['recommendation']}")
print(f"  fit L33/L34    {RECOMMENDATION['fit_l33_l34']}")
print()
print(RECOMMENDATION["rationale"])

REPORT = {
    "schema": "jlens.mmpilot.l32_followup_report.v1",
    "protocol": L32_FOLLOWUP_PROTOCOL,
    "mode": CONFIG.mode,
    "commit": COMMIT,
    "run_dir": str(RUN_DIR),
    "intervention_family": INTERVENTION_FAMILY,
    "intervention_note": (
        "source-derived J-space causal steering (h' = h +- alpha * v_concept). "
        "NOT the Anthropic two-coordinate swap, which measures its coefficient "
        "from the activation and requires a contiguous confirmed layer band."
    ),
    "environment": ENVIRONMENT,
    "artifact_discovery": DISCOVERED.to_dict(),
    "lens_validation": L32_VALIDATION,
    "discovery_refusals_exercised": DISCOVERY_REFUSALS,
    "prompt_protocol": {
        modality: fingerprint
        for modality, fingerprint in PROMPT_FINGERPRINTS.items()
    },
    "model_visible_prompt": OPEN_QUESTION,
    "comparability": COMPARABILITY,
    "media_pairing_audit": PAIRING_AUDIT,
    "split_leakage": LEAKAGE,
    "image_independence_audit": INDEPENDENCE,
    "split_provenance": SPLIT_PROVENANCE,
    "original_manifest_checksum": ORIGINAL_MANIFEST_CHECKSUM,
    "derived_manifest_checksum": DERIVED_MANIFEST_CHECKSUM,
    "admissibility_rule": ADMISSIBILITY_RULE,
    "budget": BUDGET.to_dict(),
    "timing": TIMING,
    "invariance": INVARIANCE,
    "native_head_audit": HEAD_AUDIT,
    "native_head_agreement": HEAD_AGREEMENT,
    "reporting_version": L32_REPORTING_VERSION,
    "l32_criterion_text": L32_CRITERION_TEXT,
    # The per-modality table, the clause detail and the flattened controls are
    # stored, not just printed, so a reader never has to re-derive them from the
    # rows to see what was measured.
    "convergence_cells": CONVERGENCE_CELLS,
    "classification_detail": CONVERGENCE_DETAIL,
    "controls": CONTROL_ROWS,
    "causal_breakdown": CAUSAL_BREAKDOWN,
    "convergence": (
        {k: v for k, v in CONVERGENCE.items() if k != "summary"}
        if CONVERGENCE
        else None
    ),
    "convergence_summary": (CONVERGENCE or {}).get("summary"),
    "followup_fingerprint": FOLLOWUP_FINGERPRINT,
    "run_fingerprint_digest": FINGERPRINT.digest,
    "representational": REPRESENTATIONAL,
    "verdicts": {
        "A_lens_integrity": VERDICT_A,
        "B_representational_transfer": VERDICT_B,
        "C_causal_transfer": VERDICT_C,
        "D_native_output_convergence": VERDICT_D,
        "E_pre_convergence_causal_transfer": VERDICT_E,
    },
    "paired_reference": {
        "enabled": PAIRED_REFERENCE_ENABLED,
        "lens": REFERENCE_LENS_REPORT,
        "verdict": REFERENCE_CAUSAL_VERDICT,
    },
    "cross_layer_comparison": CROSS_LAYER,
    "measured_vs_historical": MEASURED_VS_HISTORICAL,
    "next_step": RECOMMENDATION,
    "convergence_phrase": CONVERGENCE_PHRASE,
    "switches": {
        "RUN_REAL_L32_FOLLOWUP": RUN_REAL_L32_FOLLOWUP,
        "RUN_MODEL_STAGES": RUN_MODEL_STAGES,
        "CONFIRM_MODEL_LOAD": CONFIRM_MODEL_LOAD,
        "CONFIRM_CAUSAL_BUDGET": CONFIRM_CAUSAL_BUDGET,
        "RUN_PAIRED_L35_REFERENCE": RUN_PAIRED_L35_REFERENCE,
        "CONFIRM_PAIRED_REFERENCE_BUDGET": CONFIRM_PAIRED_REFERENCE_BUDGET,
    },
    "resume": STORE.status_report() if STORE is not None else None,
    "mock_proves_pipeline_only": CONFIG.mode != "l32_followup",
}
PHRASING = assert_report_phrasing(json.dumps(REPORT, default=str))
REPORT["phrasing_check"] = PHRASING

(RUN_DIR / "l32_followup_report.json").write_text(
    json.dumps(REPORT, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
)
print(f"\\nreport {RUN_DIR / 'l32_followup_report.json'}")
print(f"phrasing check passed: {PHRASING['passed']}  "
      f"(required phrase present: {PHRASING['required_phrase_present']})")
'''
)

code(
    '''
# 18b. Resume state, and the proof that nothing outside this run was written.
_status = STORE.status_report() if STORE is not None else {}
print("RESUME")
print(f"  status        {_status.get('status')}")
print(f"  fingerprint   {_status.get('fingerprint_digest')}")
print(f"  units         {_status.get('completed_units')}")
print(f"  invalid units {len(_status.get('invalid_units') or [])}")
if _status.get("invalid_units"):
    raise RuntimeError(
        f"torn units found: {_status['invalid_units']}. A checksum-invalid unit "
        "is treated as missing, never as data."
    )

print()
print("WRITTEN")
print(f"  {RUN_DIR}")
print("NOT WRITTEN (read-only evidence)")
for _path in (RESOLVED_EXTENSION_RUN_DIR, COMPLETED_TRANSFER_RUN_DIR):
    print(f"  {_path}")

if CONFIG.mode != "l32_followup":
    print()
    print("MOCK RUN COMPLETE. Pipeline behaviour only — no evidence about Gemma 4,")
    print("about layer 32, about causal transfer, or about when the native")
    print("readout converges.")
else:
    print()
    print("Send back l32_followup_report.json, the section 13 pair table, the")
    print("section 14 causal table, and the section 15 convergence table.")
'''
)

# ============================================ 19. the reporting amendment

markdown(
    """
## 19. Reporting amendment for an already-completed run — **CPU only**

Sections 1–18 above now print correctly, so a **new** run needs nothing from
this section. It exists for runs that completed under the earlier, defective
output: a run whose numbers are right and whose display was not.

Set `AMEND_COMPLETED_RUN_DIR` to the finished run and run **sections 1, 2, 4 and
19 only**. No Drive dataset is read, no model is loaded, no unit is computed, no
run directory is created.

What it does:

* reads `l32_followup_report.json` **read-only** and checksums it;
* re-derives the per-modality table, the classification and the controls from
  the run's own checksum-validated readout rows, using the **same** frozen
  functions at the **same** criterion digest — and **refuses** if the re-derived
  classification differs from the recorded one, because then it would not be a
  reporting pass;
* binds the result to the original report's checksum, the run fingerprint, a
  source-unit digest and the reporting version;
* writes `l32_followup_report_reporting_v2.json` and `.md` **beside** the
  original, atomically.

The original report and every stored unit stay byte-identical. **No verdict
moves.** Layer 32 stays whatever it was measured to be.
"""
)

code(
    '''
# 19. CPU-only reporting amendment. Requires sections 1, 2 and 4 only.
#
# Set this to a completed run directory and run this cell. Leave it empty and
# the cell does nothing.
AMEND_COMPLETED_RUN_DIR = ""

AMEND_COMPLETED_RUN_DIR = (
    os.environ.get("MMPILOT_AMEND_L32_RUN_DIR") or AMEND_COMPLETED_RUN_DIR
)
# Optional: pin the exact bytes the amendment was reviewed against.
AMEND_EXPECTED_REPORT_SHA256 = ""

AMENDMENT = None
AMENDMENT_PATHS = None
if not AMEND_COMPLETED_RUN_DIR:
    print("skipped: AMEND_COMPLETED_RUN_DIR is empty.")
    print()
    print("Sections 1-18 already print the corrected output, so a run completed")
    print("with this notebook needs no amendment. Point this at a run that")
    print("finished under the earlier output to re-render it.")
else:
    from jlens.mmpilot.l32_reporting import (
        build_reporting_amendment,
        render_amendment_markdown,
        write_reporting_amendment,
    )

    AMENDMENT = build_reporting_amendment(
        AMEND_COMPLETED_RUN_DIR,
        layer=L32_PHYSICAL_LAYER,
        expected_report_checksum=AMEND_EXPECTED_REPORT_SHA256 or None,
    )
    AMENDMENT_PATHS = write_reporting_amendment(AMEND_COMPLETED_RUN_DIR, AMENDMENT)

    print(render_amendment_markdown(AMENDMENT))
    print()
    print("=" * 72)
    print("WRITTEN")
    for _name, _path in sorted(AMENDMENT_PATHS.items()):
        print(f"  {_name:9s} {_path}")
    print("UNCHANGED")
    print(f"  {AMENDMENT['amends']['original_report']}  "
          f"{AMENDMENT['amends']['original_report_checksum']}")
    print("  every stored scientific unit")
    print()
    for _name, _value in AMENDMENT["verdicts_unchanged"].items():
        print(f"  {_name:38s} {_value}")
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
