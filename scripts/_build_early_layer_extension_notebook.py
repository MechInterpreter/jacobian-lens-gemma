# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Generate ``notebooks/research_grade_early_layer_jlens_extension_colab.ipynb``.

Written from source rather than edited as JSON, so the committed notebook stays
output-free and byte-reproducible. Run
``python scripts/_build_early_layer_extension_notebook.py`` after changing a
cell; a test regenerates it and fails on drift.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TARGET = (
    REPO_ROOT
    / "notebooks"
    / "research_grade_early_layer_jlens_extension_colab.ipynb"
)

CELLS: list[tuple[str, str]] = []


def markdown(text: str) -> None:
    CELLS.append(("markdown", text.strip("\n")))


def code(text: str) -> None:
    CELLS.append(("code", text.strip("\n")))


# ============================================================== introduction

markdown(
    """
# Early-layer J-lens extension — continuing one accumulator, replacing every held-out set

**One question.** The completed scale-100 calibration validated physical layers
**35, 38 and 40** on an untouched confirmation set and **failed at 32 and 26**.
Layer 32 failed closely enough to be worth more calibration data. The completed
multimodal experiment has since established controlled three-modality causal
transfer at 35/38/40, and the native-output convergence audit found those layers
already converged onto the clean answer. *Does layer 32 — or, if it earns it,
layer 26 — become a validated readout under substantially stronger calibration?*

## The one thing you must understand before running anything

**The scale-100 fitting accumulator is reusable. The scale-100 confirmation set
is not.**

`J_l = E[∂h_final/∂h_l]` is a population mean estimated by a running average
over a deterministically ordered prompt list (`jlens/fitting.py`, upstream and
unmodified). The fitting loop reads **no** validation or confirmation result, so
there is no code path by which a held-out number could have influenced the
accumulator. Continuing it from 100 prompts to 250 and 1,000 is a longer fit —
not a fit informed by its own evaluation. The continuation is **bit-identical**
to a fresh nested fit over the same ordering, because the additions happen in
the same sequence, and a test asserts that rather than assuming it.

The old confirmation set has been **opened**, its verdict has been **read**, and
that verdict is why this extension exists. A set whose result already influenced
the decision to run a larger scale cannot be that scale's untouched endpoint. It
is not reused, not relabelled and not reset — it is **excluded** from every new
split, and the exclusion is counted and checksummed. The old development set
goes with it, for the same reason.

## What this notebook does and does not touch

- The parent run `rgcalib_real_7e3736b4de8f` is opened **read-only**. Every
  resolved parent file is checksummed before and after, and the run proves
  byte-for-byte that it changed nothing.
- The existing **L35 / L38 / L40 published lenses are unchanged.** This
  extension neither overwrites, replaces nor republishes them.
- Calibration and validation are **text-only**, on the same pinned WikiText
  revision. No SpokenCOCO, no image, no audio, no cross-modal adapter.
- The estimator is unchanged. There is no optimizer and no loss.

**Nothing starts by itself.** `RUN_REAL_EARLY_LAYER_EXTENSION`,
`RUN_MODEL_STAGES`, `CONFIRM_PARENT_IMPORT`, `CONFIRM_250_BUDGET`,
`CONFIRM_1K_BUDGET`, `RUN_FRESH_DEVELOPMENT`, `RUN_FINAL_CONFIRMATION` and
`PUBLISH_VALIDATED_EARLY_LENSES` are all `False` in the committed notebook and
must each be set by hand. Running every cell as committed performs a
deterministic MOCK run that touches no model, no Hub, no Drive and no corpus.

**MOCK success proves pipeline behaviour only.** It is not evidence about
Gemma 4, about layer 32, about layer 26, or about whether a validated earlier
readout exists.

Frozen configuration:
[`configs/research_grade_early_layer_jlens_extension_v1.json`](../configs/research_grade_early_layer_jlens_extension_v1.json).
The completed study's configuration is **not modified**.
"""
)

# ================================================================= 0. bootstrap

markdown(
    """
## 0. Bootstrap repository

Clone or update the checkout and make `import jlens` resolve to it. Nothing from
the repository is imported until the last cell of this section.
"""
)

code(
    '''
# 0a. Bootstrap constants only. Nothing from this repository is imported yet.
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
# 0b. Clone or update the repository, then verify the checked-out branch.
#
# Idempotent: clones when absent, otherwise fetches the branch, checks it out,
# and resets to origin. The reset discards local edits inside the Colab
# checkout — that directory is scratch, not somewhere to keep work.
import os
import subprocess
import sys
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
REPO_PATH = Path(os.environ.get("RGEXT_REPO_DIR") or REPO_DIR)


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
# 0c. Install the repository, move into it, and verify that `import jlens`
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

# ============================================================ 1. configuration

markdown(
    """
## 1. Configuration and staged switches

Every switch is `False`. They are staged so that "I want to import the parent",
"I have read what 250 costs", "I have read what 1,000 costs", "run development",
"open the confirmation set" and "publish" are six separate decisions rather than
one.

The scale schedule is **nested**: 1,000 cannot be confirmed without 250, because
the larger lens *is* the smaller accumulator continued. Scale 100 is the
parent's and is descriptive only — it is scored on the fresh development set so
the plateau rule has three points, and it can never be newly confirmed.
"""
)

code(
    '''
# 1. Configuration. Requires section 0 (it imports from the repository).
# Nothing here mounts Drive, reads the parent run, or loads a model.

# ---- run switches --------------------------------------------------------
RUN_REAL_EARLY_LAYER_EXTENSION = False
RUN_MODEL_STAGES = False
CONFIRM_PARENT_IMPORT = False
RUN_FRESH_DEVELOPMENT = False
RUN_FINAL_CONFIRMATION = False
PUBLISH_VALIDATED_EARLY_LENSES = False
# ---- budget confirmations (read section 11 first) ------------------------
CONFIRM_250_BUDGET = False
CONFIRM_1K_BUDGET = False

# ---- where the completed scale-100 run lives -----------------------------
PARENT_RUN_DIR = (
    "/content/drive/MyDrive/jacobian-lens-gemma/runs/rgcalib_real_7e3736b4de8f"
)

import json
from pathlib import Path

from jlens.calibration.extension import (
    BASELINE_SCALE,
    DEVELOPMENT_SCALE_POINTS,
    EARLY_LAYERS_OF_INTEREST,
    EXTENSION_CONFIRMATION_GATE,
    EXTENSION_GATE,
    EXTENSION_PROTOCOL,
    EXTENSION_SCALE_POINTS,
    EXTENSION_SELECTION_RULE,
    EXTENSION_SPLIT_SEED,
    N_CONFIRMATION_PROMPTS,
    N_DEVELOPMENT_PROMPTS,
    PRIMARY_EARLY_LAYER,
    PUBLISHABLE_LAYERS,
    SECONDARY_EARLY_LAYER,
    extension_gate_text,
)
from jlens.calibration.plan import build_capture_plan, normalized_depth
from jlens.calibration.scale import PLATEAU_RULE

CONFIG_PATH = Path("configs/research_grade_early_layer_jlens_extension_v1.json")
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
PROTOCOL_VERSION = CONFIG["protocol_version"]

LAYERS = tuple(CONFIG["sites"]["source_layers"])
TARGET_LAYER = CONFIG["sites"]["target_layer"]
MAX_SEQ_LEN = 128
SKIP_FIRST = CONFIG["positions"]["skip_first"]
DIM_BATCH = 8

if PROTOCOL_VERSION != EXTENSION_PROTOCOL.version:
    raise RuntimeError(
        "the frozen config and the module disagree about the protocol version; "
        "refusing to answer a different question than the one that was frozen"
    )
if tuple(CONFIG["scale"]["candidate_scales"]) != tuple(EXTENSION_SCALE_POINTS):
    raise RuntimeError(
        "the frozen config and the module disagree about the candidate scales"
    )
if tuple(CONFIG["scale_selection"]["earlier_layers"]) != tuple(
    EARLY_LAYERS_OF_INTEREST
):
    raise RuntimeError(
        "the frozen config and the module disagree about the earlier layers"
    )

# ---- the nested budget confirmations -------------------------------------
_CONFIRMED = [
    scale
    for scale, confirmed in zip(
        EXTENSION_SCALE_POINTS, (CONFIRM_250_BUDGET, CONFIRM_1K_BUDGET)
    )
    if confirmed
]
if _CONFIRMED and _CONFIRMED != list(EXTENSION_SCALE_POINTS[: len(_CONFIRMED)]):
    raise RuntimeError(
        f"budget confirmations {_CONFIRMED} are not a nested prefix of "
        f"{list(EXTENSION_SCALE_POINTS)}. The 1,000-prompt lens IS the "
        "250-prompt accumulator continued, so 1,000 cannot be run without 250."
    )
ACTIVE_SCALE_POINTS = tuple(_CONFIRMED)

MODEL_STAGES_ENABLED = bool(
    RUN_REAL_EARLY_LAYER_EXTENSION
    and RUN_MODEL_STAGES
    and CONFIRM_PARENT_IMPORT
    and ACTIVE_SCALE_POINTS
)
MODE = "real" if MODEL_STAGES_ENABLED else "mock"

PLAN = build_capture_plan(
    layers=LAYERS,
    target_layer=TARGET_LAYER,
    d_model=CONFIG["model"]["expect_d_model"],
    dim_batch=DIM_BATCH,
    max_seq_len=MAX_SEQ_LEN,
    skip_first=SKIP_FIRST,
    n_layers=CONFIG["model"]["expect_n_layers"],
)

print(f"protocol           {PROTOCOL_VERSION}")
print(f"protocol digest    {EXTENSION_PROTOCOL.digest}")
print(f"mode               {MODE}")
print(f"layers             {list(LAYERS)}")
print(f"normalized depth   {[normalized_depth(l) for l in LAYERS]}")
print(f"target layer       L{TARGET_LAYER}   plan digest {PLAN.digest}")
print(f"baseline scale     {BASELINE_SCALE} (descriptive only, never re-confirmed)")
print(f"candidate scales   {list(EXTENSION_SCALE_POINTS)}  active {list(ACTIVE_SCALE_POINTS)}")
print(f"development scored {list(DEVELOPMENT_SCALE_POINTS)}")
print(f"primary / stretch  L{PRIMARY_EARLY_LAYER} / L{SECONDARY_EARLY_LAYER}")
print(f"publication targets{list(PUBLISHABLE_LAYERS)}")
print(f"gate               {EXTENSION_GATE.version}")
print(f"gate digest        {EXTENSION_GATE.digest}")
print(f"selection rule     {EXTENSION_SELECTION_RULE.digest}")
print(f"plateau digest     {PLATEAU_RULE.digest}")
print(f"development/confirmation prompts   {N_DEVELOPMENT_PROMPTS} / {N_CONFIRMATION_PROMPTS}")
print(f"fresh split seed   {EXTENSION_SPLIT_SEED}")
if not MODEL_STAGES_ENABLED:
    print()
    print("MOCK MODE — no Gemma, no Hub, no Drive, no corpus download.")
    print("This proves pipeline behaviour only. It is not evidence about Gemma,")
    print("about layer 32, about layer 26, or about whether an earlier lens exists.")
'''
)

# ================================================================== 2. drive

markdown(
    """
## 2. Optional mount Google Drive

Mounted only for a real run: the parent run lives there, and so does the
extension's own run directory. The extension **never writes inside the parent**.

Put the HuggingFace cache on Drive too — at ~16 GB per model download, re-pulling
Gemma every session is the second-largest hidden cost of this study.
"""
)

code(
    '''
# 2. Optional Drive mount. Skipped entirely in MOCK mode.
import tempfile

DRIVE_ROOT = None
if MODEL_STAGES_ENABLED and IN_COLAB:
    from google.colab import drive  # noqa: PLC0415

    drive.mount("/content/drive")
    DRIVE_ROOT = Path("/content/drive/MyDrive/jacobian-lens-gemma")
    DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(DRIVE_ROOT / "hf_cache"))
    print(f"drive    {DRIVE_ROOT}")
    print(f"HF_HOME  {os.environ['HF_HOME']}")
else:
    DRIVE_ROOT = Path(tempfile.mkdtemp(prefix="rgext_mock_"))
    print(f"MOCK run directory (temporary, not Drive): {DRIVE_ROOT}")

RUN_ROOT = DRIVE_ROOT / "runs"
RUN_ROOT.mkdir(parents=True, exist_ok=True)
print(f"runs     {RUN_ROOT}")
'''
)

# ============================================================ 3. dependencies

markdown(
    """
## 3. Install and verify dependencies

Versions are recorded into every artifact. In MOCK mode only `torch` is
required — `transformers`, `datasets` and `huggingface_hub` are never imported.
"""
)

code(
    '''
# 3. Dependencies. In MOCK mode nothing beyond torch is imported.
import platform

import torch

ENVIRONMENT = {
    "python": platform.python_version(),
    "platform": platform.platform(),
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "device_name": (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    ),
    "local_commit": COMMIT,
}

if MODEL_STAGES_ENABLED:
    if IN_COLAB:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "datasets"],
            check=False,
        )
    import transformers  # noqa: PLC0415

    from jlens.metadata import UPSTREAM_COMMIT, environment_manifest  # noqa: PLC0415

    ENVIRONMENT = environment_manifest()
    ENVIRONMENT["upstream_commit"] = UPSTREAM_COMMIT
    if not torch.cuda.is_available():
        raise RuntimeError(
            "a real continuation needs a GPU runtime; this one has no CUDA device"
        )
else:
    from jlens.metadata import UPSTREAM_COMMIT  # noqa: PLC0415

    ENVIRONMENT["upstream_commit"] = UPSTREAM_COMMIT
    ENVIRONMENT["transformers"] = "not imported (MOCK)"

for key, value in ENVIRONMENT.items():
    print(f"  {key:<20} {value}")
'''
)

# ========================================================== 4. authentication

markdown(
    """
## 4. Authentication

Gemma 4 is a gated repository. A token is needed only for a real run; MOCK mode
never contacts the Hub.
"""
)

code(
    '''
# 4. Authentication. No-op in MOCK mode.
HF_TOKEN = None
if MODEL_STAGES_ENABLED:
    if IN_COLAB:
        try:
            from google.colab import userdata  # noqa: PLC0415

            HF_TOKEN = userdata.get("HF_TOKEN")
        except Exception:
            HF_TOKEN = None
    HF_TOKEN = HF_TOKEN or os.environ.get("HF_TOKEN")
    if not HF_TOKEN:
        raise RuntimeError(
            "no HF_TOKEN found. google/gemma-4-E4B-it is gated: add HF_TOKEN to "
            "the Colab secrets panel or export it into the environment."
        )
    os.environ["HF_TOKEN"] = HF_TOKEN
    print("HF token present")
else:
    print("MOCK mode: no authentication, no Hub contact.")
'''
)

# ================================================ 5. protocol, gate, selection

markdown(
    """
## 5. Frozen protocol, gate and selection rule

Printed **before** any result-producing cell runs. All three digests are
checksummed into the run fingerprint: editing a threshold invalidates stored
results rather than rescoring them, which is what makes "predeclared" mean
something operationally.

The gate is the completed study's gate carried over threshold for threshold, at
a larger sample and with a **stricter** target-diversity floor. Nothing is
loosened to let layer 32 or layer 26 pass.
"""
)

code(
    '''
# 5. The frozen protocol, the gate, the selection rule, and the plateau rule.
print(EXTENSION_PROTOCOL.text())
print()
print(extension_gate_text())
print()
print(EXTENSION_SELECTION_RULE.text())
print()
print(PLATEAU_RULE.text())
'''
)

code(
    '''
# 5b. What is reused from the parent, and what is deliberately destroyed.
print("WHAT CROSSES THE BOUNDARY FROM THE COMPLETED SCALE-100 RUN")
print()
print("  REUSED   the fitting accumulator {jacobian_sum, n_done}")
print("           " + CONFIG["evidentiary_boundary"]["why_the_accumulator_is_reusable"])
print()
print("  SPENT    the old confirmation set")
print("           " + CONFIG["evidentiary_boundary"]["old_confirmation_set_reasoning"])
print("           handling: " + CONFIG["evidentiary_boundary"]["old_confirmation_set_handling"])
print()
print("  SPENT    the old development set")
print("           " + CONFIG["evidentiary_boundary"]["old_development_set_handling"])
print()
print("  ESTIMATOR")
print("    J_l = E[dh_final / dh_l], a running mean over prompts.")
print("    Implementation: jlens.fitting.fit (upstream Anthropic code, unmodified).")
print(f"    Upstream commit: {ENVIRONMENT.get('upstream_commit')}")
print("    objective   " + CONFIG["continuation"]["optimizer"])
print("    Continuing the accumulator is a LONGER FIT, not a different object.")
'''
)

# =================================================== 6. parent-run audit

markdown(
    """
## 6. Parent-run audit and read-only import

Filenames and schema are **resolved, not assumed**: the directory is walked, the
roles this extension needs are matched against what is actually there, and a
missing artifact stops the run with the role named, the paths searched, and the
files that do exist.

Every stored unit is verified against its own checksum *and* against the parent
run's fingerprint before its contents are believed. Then twenty-four blocking
checks establish that the accumulator may be continued at all — including
`n_done == 100`, the layer grid, the hook site, `d_model`, the model and
tokenizer revisions, the corpus identity, and that no prompt was dropped before
the parent fitted (without which the parent's fitted prompt identity is not
recoverable).

The read-only provenance manifest is written into the **extension's** directory,
never the parent's.
"""
)

code(
    '''
# 6. Resolve, read and audit the parent run. Nothing here writes to it.
from jlens.calibration.parent import (
    ParentImportRefused,
    ParentRequirements,
    audit_parent_run,
    discover_parent_files,
    format_parent_audit,
    load_parent_run,
    parent_provenance_manifest,
    protected_parent_checksums,
)

MOCK_PARENT = None
MODEL = None

if MODEL_STAGES_ENABLED:
    PARENT_ROOT = Path(PARENT_RUN_DIR)
    PARENT_BASELINE = BASELINE_SCALE
else:
    # MOCK: produce a parent by running the completed study's OWN code against a
    # tiny frozen CPU stack — real store, real fit, real checkpoint, real units.
    from jlens.calibration.extension_mock import (  # noqa: PLC0415
        MOCK_BASELINE_SCALE,
        build_mock_parent_run,
    )
    from jlens.calibration.mock import MockCalibrationModel  # noqa: PLC0415

    MODEL = MockCalibrationModel()
    MOCK_PARENT = build_mock_parent_run(RUN_ROOT / "mock_parent", model=MODEL)
    PARENT_ROOT = Path(MOCK_PARENT.root)
    PARENT_BASELINE = MOCK_BASELINE_SCALE

INVENTORY = discover_parent_files(PARENT_ROOT)
PARENT = load_parent_run(PARENT_ROOT, baseline_scale=PARENT_BASELINE)

if MODEL_STAGES_ENABLED:
    _plan_for_requirements = PLAN
else:
    _plan_for_requirements = MOCK_PARENT.plan

REQUIREMENTS = ParentRequirements(
    model_repo_id=CONFIG["model"]["repo_id"],
    model_revision=CONFIG["model"]["revision"],
    tokenizer_repo_id=CONFIG["model"]["tokenizer_repo_id"],
    tokenizer_revision=CONFIG["model"]["tokenizer_revision"],
    source_layers=tuple(_plan_for_requirements.layers),
    target_layer=_plan_for_requirements.target_layer,
    d_model=_plan_for_requirements.d_model,
    hook_site=CONFIG["sites"]["source_site"],
    skip_first=_plan_for_requirements.skip_first,
    max_seq_len=_plan_for_requirements.max_seq_len,
    dim_batch=_plan_for_requirements.dim_batch,
    corpus_hf_dataset=(
        CONFIG["corpus"]["primary"]["hf_dataset"] if MODEL_STAGES_ENABLED else "mock"
    ),
    corpus_config=(
        CONFIG["corpus"]["primary"]["config"] if MODEL_STAGES_ENABLED else "mock"
    ),
    corpus_split=CONFIG["corpus"]["primary"]["split"],
    estimator=CONFIG["continuation"]["estimator_implementation"],
    artifact_format_version=CONFIG["artifact_format_version"],
    baseline_scale=PARENT_BASELINE,
    expected_n_done=PARENT_BASELINE,
)

PARENT_AUDIT = audit_parent_run(PARENT, requirements=REQUIREMENTS)
PARENT_CHECKSUMS_BEFORE = protected_parent_checksums(
    PARENT_ROOT, layout=PARENT.layout
)

print(format_parent_audit(PARENT_AUDIT))
print()
print(f"parent files       {INVENTORY['n_files']} "
      f"({INVENTORY['total_bytes'] / 2**20:.1f} MiB)")
print(f"protected files    {PARENT_CHECKSUMS_BEFORE['n_files']} checksummed before the run")
print(f"optional absent    {PARENT.layout['optional_absent']}")
print(f"old confirmation vault  {PARENT.confirmation_vault_status or '<no run report>'}")
'''
)

# ============================================ 7. model architecture and hooks

markdown(
    """
## 7. Model architecture and hook audit

Verifies every assumption the estimator depends on before GPU time is committed:
42 layers, `d_model` 2560, vocab 262144, frozen parameters, tied unembedding, and
the residual site the lens reads. Identical to the completed study's audit —
this extension changes nothing about capture.

In MOCK mode this is the same tiny frozen CPU stack the mock parent was fitted
on, at the real depth and the real layer grid.
"""
)

code(
    '''
# 7. Load (or reuse) the model and audit the architecture and the hook site.
if MODEL_STAGES_ENABLED:
    from jlens.gemma4 import load_gemma4, verify_architecture  # noqa: PLC0415

    MODEL, LOAD_INFO = load_gemma4(
        CONFIG["model"]["repo_id"],
        revision=CONFIG["model"]["revision"],
        dtype=getattr(torch, CONFIG["model"]["dtype"]),
        device_map=CONFIG["model"]["device_map"],
        allow_model_load=True,
        token=HF_TOKEN,
    )
    ARCHITECTURE = verify_architecture(
        MODEL,
        expect_n_layers=CONFIG["model"]["expect_n_layers"],
        expect_d_model=CONFIG["model"]["expect_d_model"],
        expect_vocab_size=CONFIG["model"]["expect_vocab_size"],
    ).to_dict()
    ACTIVE_PLAN = PLAN
else:
    from jlens.calibration.extension_mock import mock_load_info  # noqa: PLC0415

    LOAD_INFO = mock_load_info()
    ARCHITECTURE = {
        "model_class": type(MODEL).__name__,
        "n_layers": MODEL.n_layers,
        "d_model": MODEL.d_model,
        "vocab_size": MODEL.vocab_size,
        "params_frozen": not any(p.requires_grad for p in MODEL.parameters()),
        "warnings": ["MOCK model — proves interface and pipeline only"],
    }
    ACTIVE_PLAN = MOCK_PARENT.plan

if tuple(ACTIVE_PLAN.layers) != tuple(PARENT.accumulator.source_layers):
    raise RuntimeError(
        f"the capture plan fits {list(ACTIVE_PLAN.layers)} but the parent "
        f"accumulator holds {list(PARENT.accumulator.source_layers)}; upstream "
        "would refuse this checkpoint and so does this notebook"
    )

print(f"model class     {ARCHITECTURE['model_class']}")
print(f"layers/width    {ARCHITECTURE['n_layers']} / {ARCHITECTURE['d_model']}")
print(f"vocab           {ARCHITECTURE['vocab_size']}")
print(f"params frozen   {ARCHITECTURE['params_frozen']}")
print(f"revision        {LOAD_INFO['model_revision']}")
for _warning in ARCHITECTURE.get("warnings", []):
    print(f"  note: {_warning}")
print()
print(f"capture plan    layers {list(ACTIVE_PLAN.layers)} -> L{ACTIVE_PLAN.target_layer}")
print(f"  per prompt    1 forward + {ACTIVE_PLAN.backward_passes_per_prompt} backward passes")
print(f"  backward span {ACTIVE_PLAN.backward_span} blocks (the cost driver)")
print("  capture is UNCHANGED from the parent run; the accumulator agrees.")
'''
)

# ================================== 8. reconstruct the fit ordering

markdown(
    """
## 8. Reconstruct the fit ordering and verify the parent prefix

The parent stored split **checksums**, not corpus text. So the only way to
recover its fit ordering is to re-stream the pinned corpus under the parent's own
collection parameters — every one of which is read from the parent's artifacts,
with its source named — and prove the result reproduces those checksums.

Then the extension's own fit ordering is declared:

1. positions `0 … 99` are the parent's fit partition in the parent's nested
   order, so **every prefix the parent fitted is a prefix of this list**;
2. any further positions come from records the parent's collection never
   reached, in ascending stream index.

Clause 2 is why the ordering is declared here rather than taken from
`build_partitions`: re-running the nested-hash order over a *longer* record set
can insert a late record ahead of an early one, silently breaking the nesting the
continuation depends on. The pinned-prefix ordering is stable under stream
extension; the pure hash ordering is not.

Finally the first 100 records are checksummed against the parent's fit-prompt
manifest. **The 100 prompts are skipped only after that agreement.**
"""
)

code(
    '''
# 8. Re-stream the corpus, prove the reconstruction, and build the fit ordering.
from jlens.calibration.corpus import collect_records_for_partition_quotas
from jlens.calibration.extension import (
    build_extension_fit_order,
    parent_collection_parameters,
    verify_fit_prefix,
    verify_reconstructed_partitions,
)
from jlens.calibration.fitting import filter_records_by_tokens

COLLECTION = parent_collection_parameters(PARENT)
LARGEST_SCALE = max(ACTIVE_SCALE_POINTS) if ACTIVE_SCALE_POINTS else max(
    EXTENSION_SCALE_POINTS
)

if MODEL_STAGES_ENABLED:
    from datasets import load_dataset  # noqa: PLC0415

    CORPUS_CONFIG = dict(CONFIG["corpus"]["primary"])
    CORPUS_ID = PARENT.corpus["corpus_id"]
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

    # Records the parent's collection never reached, for the fresh sets and for
    # any fit positions beyond the parent's own partition.
    from jlens.calibration.corpus import build_records  # noqa: PLC0415

    _LAST_STREAM_INDEX = max(record.stream_index for record in _RECORDS)
    _extra_stream = load_dataset(
        CORPUS_CONFIG["hf_dataset"],
        CORPUS_CONFIG["config"],
        split=CORPUS_CONFIG["split"],
        streaming=True,
    )
    _extra_needed = COLLECTION["max_texts"] + 40 * (
        LARGEST_SCALE + N_DEVELOPMENT_PROMPTS + N_CONFIRMATION_PROMPTS
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
    _token_count = lambda text: int(  # noqa: E731
        MODEL.encode(text, max_length=ACTIVE_PLAN.max_seq_len).shape[1]
    )
else:
    from jlens.calibration.extension_mock import mock_extension_pool  # noqa: PLC0415

    CORPUS_CONFIG = dict(MOCK_PARENT.corpus_config)
    CORPUS_ID = PARENT.corpus["corpus_id"]
    RECONSTRUCTION = verify_reconstructed_partitions(
        MOCK_PARENT.partitions, parent=PARENT
    )
    OLD_FIT = MOCK_PARENT.partitions.fit
    OLD_VALIDATION = MOCK_PARENT.partitions.validation
    OLD_CONFIRMATION = MOCK_PARENT.partitions.confirmation
    EXTENSION_POOL = mock_extension_pool(MOCK_PARENT)
    _token_count = MODEL.tokenizer.token_count
    from jlens.calibration.extension_mock import (  # noqa: PLC0415
        MOCK_DEVELOPMENT_SCALES,
        MOCK_EXTENSION_SCALES,
    )

    LARGEST_SCALE = MOCK_EXTENSION_SCALES[-1]

# Drop records too short to contribute a valid source position, exactly as the
# parent did. The parent recorded zero drops, which is what makes its split
# prefix and its fitted prefix the same 100 prompts.
PARENT_FIT_RECORDS, DROPPED_SHORT = filter_records_by_tokens(
    OLD_FIT,
    token_count=_token_count,
    skip_first=ACTIVE_PLAN.skip_first,
    max_seq_len=ACTIVE_PLAN.max_seq_len,
)

FIT_RECORDS = build_extension_fit_order(
    PARENT_FIT_RECORDS,
    n_needed=LARGEST_SCALE,
    extension_pool=EXTENSION_POOL,
)
PREFIX = verify_fit_prefix(
    FIT_RECORDS,
    n_parent=PARENT_BASELINE,
    parent_prefix_checksum=PARENT.fit_prefix_checksum(PARENT_BASELINE),
)

print(f"collection params  {COLLECTION}")
print(f"reconstruction     {'EXACT' if RECONSTRUCTION['all_match'] else 'FAILED'}")
for _row in RECONSTRUCTION["partitions"]:
    print(f"  {_row['partition']:<13} {_row['actual_size']:>7,} records  "
          f"{_row['actual_checksum']}  matches={_row['matches']}")
print(f"dropped too short  {len(DROPPED_SHORT)} (parent recorded "
      f"{PARENT.corpus_manifest.get('n_dropped_too_short')})")
print(f"extension pool     {len(EXTENSION_POOL):,} records the parent never collected")
print(f"fit ordering       {len(FIT_RECORDS):,} records "
      f"({CONFIG['continuation']['fit_order_protocol']})")
print()
print(f"parent prefix      {PREFIX['parent_prefix_checksum']}")
print(f"reconstructed      {PREFIX['reconstructed_prefix_checksum']}")
print(f"MATCHES            {PREFIX['matches']}  -> skipping the first "
      f"{PARENT_BASELINE} prompts is authorized: {PREFIX['skip_authorized']}")
'''
)

# =========================================== 9. fresh development/confirmation

markdown(
    """
## 9. Completely fresh development and confirmation sets

Drawn from records the parent's collection **never reached**, under a new split
seed, a new bucket layout and a new ordering hash. Then everything old is
excluded anyway — by exact normalized checksum and by banded SimHash at Hamming
distance ≤ 3 — because a guard that is only ever exercised by the thing that
makes it unnecessary is not a guard:

- every old fit record,
- every old development record,
- every old confirmation record,
- every new fit record through the largest candidate scale.

An independent cross-split leakage audit then runs on the *constructed* sets,
against each other and against every excluded set. A hit **raises**.

**Sizes are never reduced.** If the corpus cannot fill 256 + 256, the run stops
and asks for more corpus.
"""
)

code(
    '''
# 9. Build the fresh sets, then audit them independently.
from jlens.calibration.extension import (
    audit_fresh_split_leakage,
    build_fresh_evaluation_splits,
)

EXCLUDED = {
    "old_fit": OLD_FIT,
    "old_development": OLD_VALIDATION,
    "old_confirmation": OLD_CONFIRMATION,
    "new_fit": FIT_RECORDS,
}

SPLITS = build_fresh_evaluation_splits(
    EXTENSION_POOL,
    excluded=EXCLUDED,
    corpus_id=CORPUS_ID,
    n_development=N_DEVELOPMENT_PROMPTS,
    n_confirmation=N_CONFIRMATION_PROMPTS,
)
SPLIT_LEAKAGE = audit_fresh_split_leakage(SPLITS, excluded=EXCLUDED)
SPLITS_MANIFEST = SPLITS.manifest()

print(f"pool after exclusions   {SPLITS.n_pool:,}")
print(f"excluded (exact)        {SPLITS.excluded_exact}")
print(f"excluded (near dup)     {SPLITS.excluded_near}")
print(f"excluded (pool dup)     {SPLITS.excluded_pool_duplicates}")
print()
for _name in ("development", "confirmation"):
    print(f"  {_name:<13} {len(SPLITS.get(_name)):>5} records  {SPLITS.checksum(_name)}")
print()
print(f"cross-split audit       {'CLEAN' if SPLIT_LEAKAGE['ok'] else 'FAILED'} — "
      f"{SPLIT_LEAKAGE['n_exact_hits']} exact, {SPLIT_LEAKAGE['n_near_hits']} near, "
      f"{SPLIT_LEAKAGE['candidate_pairs_compared']:,} candidate pairs compared")
print(f"splits manifest         {SPLITS_MANIFEST['manifest_checksum']}")
print()
print("The old development and confirmation sets are EXCLUDED, not reused.")
print("The old confirmation result is what motivated this extension; a set whose")
print("verdict already influenced that decision cannot be this scale's endpoint.")
'''
)

# =================================================== 10. target diversity

markdown(
    """
## 10. Target-token diversity audit

Targets are the frozen model's **own final-layer argmax** at the last prompt
position. No lens and no candidate layer is consulted —
`select_diverse_validation_prompts` has no parameter through which one could be
supplied, which makes "confirmation selection must not consult any J-lens result"
a property of the signature rather than a promise.

The floor is **stricter** than the completed study's: ≥ 32 distinct targets over
256 prompts (was ≥ 24 over 128), with the same 25% single-target ceiling. It
**refuses** rather than lowering either threshold.
"""
)

code(
    '''
# 10. Choose the held-out prompt sets, stratified on the model's own output.
from jlens.calibration.extension import (
    CONFIRMATION_PROMPT_SEED,
    DEVELOPMENT_PROMPT_SEED,
    audit_extension_target_diversity,
)
from jlens.calibration.gate import (
    ordinary_next_token_argmax,
    select_diverse_validation_prompts,
)

if MODEL_STAGES_ENABLED:
    def _target_token(prompt):
        # The model's ordinary output path only. No lens, no candidate layer.
        return ordinary_next_token_argmax(MODEL, prompt, max_length=MAX_SEQ_LEN)
else:
    from jlens.calibration.extension_mock import mock_target_token  # noqa: PLC0415

    _target_token = mock_target_token

DEVELOPMENT_PROMPTS, DEVELOPMENT_SELECTION = select_diverse_validation_prompts(
    [record.text for record in SPLITS.development],
    n_prompts=N_DEVELOPMENT_PROMPTS,
    gate=EXTENSION_GATE,
    seed=DEVELOPMENT_PROMPT_SEED,
    target_token_for_prompt=_target_token,
)
CONFIRMATION_PROMPTS, CONFIRMATION_SELECTION = select_diverse_validation_prompts(
    [record.text for record in SPLITS.confirmation],
    n_prompts=N_CONFIRMATION_PROMPTS,
    gate=EXTENSION_CONFIRMATION_GATE,
    seed=CONFIRMATION_PROMPT_SEED,
    target_token_for_prompt=_target_token,
)
DIVERSITY = audit_extension_target_diversity(
    [row["target_token_id"] for row in DEVELOPMENT_SELECTION["prompts"]]
)
CONFIRMATION_DIVERSITY = audit_extension_target_diversity(
    [row["target_token_id"] for row in CONFIRMATION_SELECTION["prompts"]]
)

print(f"development prompts   {len(DEVELOPMENT_PROMPTS)}")
print(f"  distinct targets    {DIVERSITY['n_distinct_target_tokens']} "
      f"(floor {DIVERSITY['min_distinct_target_tokens']})")
print(f"  max target share    {DIVERSITY['max_target_token_share']:.1%} "
      f"(ceiling {DIVERSITY['max_target_token_share_allowed']:.0%})")
print(f"  passed              {DIVERSITY['passed']}")
print(f"  selection checksum  {DEVELOPMENT_SELECTION['selection_checksum']}")
print()
print(f"confirmation prompts  {len(CONFIRMATION_PROMPTS)} (held back; see section 17)")
print(f"  distinct targets    {CONFIRMATION_DIVERSITY['n_distinct_target_tokens']}")
print(f"  max target share    {CONFIRMATION_DIVERSITY['max_target_token_share']:.1%}")
print(f"  passed              {CONFIRMATION_DIVERSITY['passed']}")
print(f"  selection checksum  {CONFIRMATION_SELECTION['selection_checksum']}")
print(f"  selected by J-lens performance: "
      f"{CONFIRMATION_SELECTION['selected_by_jlens_performance']}")
'''
)

# ==================================================================== 11. budget

markdown(
    """
## 11. Compute and storage budget

**Read this before setting any budget switch.** The anchor is the operator's own
observation: the scale-100 fit took **≈ 7.1 minutes on one L4** with this exact
layer grid and capture plan. Everything below is that number multiplied out,
linearly in prompts.

**This is extrapolation, not measurement.** One observation, one runtime, one
prompt-length distribution. The structure of the estimator supports linearity —
one forward plus a fixed 320 backward passes per prompt — but a shared cloud L4
is not a fixed-speed device. The high end of the band is the planning number.

Rows are **incremental, not cumulative**: the parent's first 100 prompts are
already in the accumulator and are never refitted. That is the whole economy of
the continuation.
"""
)

code(
    '''
# 11. The budget. Nothing here runs the model.
from jlens.calibration.extension import extension_budget, format_extension_budget

BUDGET = extension_budget(
    plan=PLAN,
    scale_points=EXTENSION_SCALE_POINTS,
    baseline_scale=BASELINE_SCALE,
    n_development=N_DEVELOPMENT_PROMPTS,
    n_confirmation=N_CONFIRMATION_PROMPTS,
    observed_minutes=CONFIG["budget"]["anchor_minutes"],
)
print(format_extension_budget(BUDGET))
print()
print("HONEST SUMMARY")
_r250 = BUDGET["rows"][0]
_r1k = BUDGET["rows"][1]
print(f"  reaching   250 costs {_r250['incremental_prompts']} more prompts "
      f"(~{_r250['l4_hours_central']:.2f} h central, "
      f"{_r250['l4_hours_low']:.2f}-{_r250['l4_hours_high']:.2f} h)")
print(f"  reaching 1,000 costs {_r1k['incremental_prompts']} more prompts "
      f"(~{_r1k['l4_hours_central']:.2f} h central, "
      f"{_r1k['l4_hours_low']:.2f}-{_r1k['l4_hours_high']:.2f} h)")
print(f"  total from the parent's 100: "
      f"{sum(r['incremental_prompts'] for r in BUDGET['rows'])} prompts, "
      f"~{sum(r['l4_hours_central'] for r in BUDGET['rows']):.1f} h central")
print(f"  Drive       ~{BUDGET['storage_bytes']['total'] / 2**30:.2f} GiB")
print("  A fit parallelizes exactly across runtimes via JacobianLens.merge().")
print("  This extension authorizes no scale beyond 1,000.")
'''
)

# ================================================= 12. real-run confirmation

markdown(
    """
## 12. Explicit real-run confirmation

The gate between reading the budget and spending it. Every switch must be set by
hand, and this cell prints exactly what this execution will do either way.
"""
)

code(
    '''
# 12. What this execution will actually do.
print(f"RUN_REAL_EARLY_LAYER_EXTENSION   {RUN_REAL_EARLY_LAYER_EXTENSION}")
print(f"RUN_MODEL_STAGES                 {RUN_MODEL_STAGES}")
print(f"CONFIRM_PARENT_IMPORT            {CONFIRM_PARENT_IMPORT}")
print(f"CONFIRM_250_BUDGET               {CONFIRM_250_BUDGET}")
print(f"CONFIRM_1K_BUDGET                {CONFIRM_1K_BUDGET}")
print(f"RUN_FRESH_DEVELOPMENT            {RUN_FRESH_DEVELOPMENT}")
print(f"RUN_FINAL_CONFIRMATION           {RUN_FINAL_CONFIRMATION}")
print(f"PUBLISH_VALIDATED_EARLY_LENSES   {PUBLISH_VALIDATED_EARLY_LENSES}")
print()
if MODEL_STAGES_ENABLED and not RUN_FRESH_DEVELOPMENT:
    raise RuntimeError(
        "RUN_FRESH_DEVELOPMENT is False but the model stages are enabled. "
        "Scoring 256 held-out prompts is a deliberate act; set it before the "
        "continuation starts rather than discovering the omission after the "
        "expensive cell has run."
    )
if MODEL_STAGES_ENABLED:
    _row = [r for r in BUDGET["rows"] if r["scale"] == max(ACTIVE_SCALE_POINTS)][0]
    print("REAL EXTENSION ENABLED")
    print(f"  parent run     {PARENT.root}")
    print(f"  continuing     n_done={PARENT.accumulator.n_done} -> "
          f"{list(ACTIVE_SCALE_POINTS)}")
    print(f"  incremental    {max(ACTIVE_SCALE_POINTS) - BASELINE_SCALE} prompts")
    print(f"  estimated      {_row['l4_hours_central']:.2f} h central "
          f"({_row['l4_hours_low']:.2f}-{_row['l4_hours_high']:.2f} h) for the last step")
    print(f"  Drive          ~{BUDGET['storage_bytes']['total'] / 2**30:.2f} GiB")
    print("  Resume is automatic; a disconnect loses at most one checkpoint interval")
    print("  and never the parent's 100 prompts.")
else:
    print("MOCK RUN — nothing above is spent.")
    print("  No model was downloaded, no corpus streamed, no Drive written.")
    print("  The parent below was produced by this repository's own code against a")
    print("  tiny CPU stack. Every verdict is a fixture. It proves the pipeline")
    print("  runs, and it proves nothing whatsoever about Gemma 4.")
'''
)

# ================================================ 13. continue the fit

markdown(
    """
## 13. Continue the fit from the parent accumulator

The parent checkpoint is **copied** into the extension run — byte for byte,
checksum-verified — and the parent file is opened read-only. Upstream `fit` then
skips the first 100 prompts (its own `next_idx` mechanism) and consumes the rest
of the deterministic ordering, snapshotting at 250 and at 1,000.

Resume is upstream's, hardened: a checkpoint fitted with different layers, target
or `skip_first`, or one that has fallen behind the parent's prompt count, is
**refused** rather than mixed.

In MOCK the continuation is additionally proved **bit-identical** to a fresh
nested fit by refitting from scratch and comparing tensor by tensor. That proof
is unaffordable at 1,000 prompts on an L4 and is what the prefix checksum stands
in for there.
"""
)

code(
    '''
# 13. Open the extension run, seed the checkpoint, and continue the fit.
from jlens.calibration.extension import (
    EXTENSION_FIT_ORDER_PROTOCOL,
    ExtensionStore,
    extension_corpus_manifest,
    seed_extension_checkpoint,
)
from jlens.calibration.fitting import run_calibration
from jlens.calibration.state import CalibrationFingerprint

if MODEL_STAGES_ENABLED:
    SCALES = tuple(ACTIVE_SCALE_POINTS)
    DEV_SCALES = (BASELINE_SCALE, *SCALES)
else:
    SCALES = MOCK_EXTENSION_SCALES
    DEV_SCALES = MOCK_DEVELOPMENT_SCALES

CORPUS_MANIFEST = extension_corpus_manifest(
    corpus_config=CORPUS_CONFIG,
    corpus_id=CORPUS_ID,
    fit_records=FIT_RECORDS,
    splits=SPLITS,
    scale_points=SCALES,
    parent=PARENT,
    fit_prefix_verification=PREFIX,
)

FINGERPRINT = CalibrationFingerprint(
    mode=MODE,
    protocol_version=PROTOCOL_VERSION,
    model_repo_id=LOAD_INFO["model_repo_id"],
    model_revision=LOAD_INFO["model_revision"],
    tokenizer_revision=LOAD_INFO["tokenizer_revision"],
    capture_plan_digest=ACTIVE_PLAN.digest,
    corpus_manifest_checksum=CORPUS_MANIFEST["corpus_manifest_checksum"],
    gate_digest=EXTENSION_GATE.digest,
    plateau_rule_digest=PLATEAU_RULE.digest,
    scale_points=tuple(SCALES),
    artifact_format_version=CONFIG["artifact_format_version"],
    extra={
        "extension_protocol_digest": EXTENSION_PROTOCOL.digest,
        "selection_rule_digest": EXTENSION_SELECTION_RULE.digest,
        "parent_fingerprint_digest": PARENT.fingerprint_digest,
        "parent_accumulator_checksum": PARENT.accumulator.checksum,
        "fit_order_protocol": EXTENSION_FIT_ORDER_PROTOCOL,
        "fresh_splits_checksum": SPLITS_MANIFEST["manifest_checksum"],
    },
)
RUN_DIR = RUN_ROOT / f"rgext_{MODE}_{FINGERPRINT.digest[7:19]}"
STORE = ExtensionStore(RUN_DIR, FINGERPRINT)
RESUME_STATUS = STORE.open()

PROVENANCE = parent_provenance_manifest(
    PARENT,
    PARENT_AUDIT,
    immutability=PARENT_CHECKSUMS_BEFORE,
    extension_protocol_version=PROTOCOL_VERSION,
    extension_run_dir=str(RUN_DIR),
)
STORE.save("parent_import", "provenance", PROVENANCE)
(RUN_DIR / "artifacts").mkdir(parents=True, exist_ok=True)
(RUN_DIR / "artifacts" / "parent_provenance.json").write_text(
    json.dumps(PROVENANCE, indent=2, ensure_ascii=False, default=str),
    encoding="utf-8",
)
STORE.save(
    "fresh_splits",
    "manifest",
    {
        "splits": SPLITS_MANIFEST,
        "leakage_audit": SPLIT_LEAKAGE,
        "development_selection": DEVELOPMENT_SELECTION,
        "confirmation_selection_checksum": CONFIRMATION_SELECTION[
            "selection_checksum"
        ],
        "development_diversity": DIVERSITY,
        "confirmation_diversity": CONFIRMATION_DIVERSITY,
        "reconstruction": RECONSTRUCTION,
        "fit_prefix_verification": PREFIX,
        "corpus": CORPUS_MANIFEST,
    },
)

SEED = seed_extension_checkpoint(PARENT.accumulator, STORE.checkpoint_path)
print(f"run dir        {RUN_DIR}")
print(f"fingerprint    {FINGERPRINT.digest}")
print(f"resume         {RESUME_STATUS}")
print(f"checkpoint     {SEED['action']} from parent {SEED['parent_checkpoint_checksum']}")
print(f"               n_done={SEED['n_done']} next_idx={SEED['next_idx']}  "
      f"parent written: {SEED['parent_written']}")
'''
)

code(
    '''
# 13b. The continuation itself. This is the only expensive cell in the notebook.
CONTINUATION = run_calibration(
    MODEL,
    FIT_RECORDS,
    plan=ACTIVE_PLAN,
    scale_points=SCALES,
    store=STORE,
    checkpoint_every=25 if MODEL_STAGES_ENABLED else 4,
    diagnostics_every=25 if MODEL_STAGES_ENABLED else 4,
)

CONTINUATION_RECORD = {
    "fit_order_protocol": EXTENSION_FIT_ORDER_PROTOCOL,
    "parent_checkpoint_checksum": PARENT.accumulator.checksum,
    "parent_n_done": PARENT.accumulator.n_done,
    "prefix_verification": PREFIX,
    "seed": SEED,
    "n_done": CONTINUATION.n_done,
    "n_skipped": CONTINUATION.n_skipped,
    "scale_points": list(SCALES),
    "snapshots": {
        str(scale): snapshot.to_dict()
        for scale, snapshot in sorted(CONTINUATION.snapshots.items())
    },
    "elapsed_seconds": round(CONTINUATION.elapsed_seconds, 2),
    "parent_written": False,
    "objective": "not_applicable_estimator_is_a_sample_mean",
}

# MOCK only: prove bit-identity with a fresh nested fit. Unaffordable for real.
EQUIVALENCE = None
if not MODEL_STAGES_ENABLED:
    from jlens.calibration.extension import (  # noqa: PLC0415
        verify_continuation_equals_fresh_fit,
    )

    EQUIVALENCE = verify_continuation_equals_fresh_fit(
        MODEL,
        FIT_RECORDS,
        plan=ACTIVE_PLAN,
        scale=max(SCALES),
        continued_checkpoint=STORE.checkpoint_path,
        scratch_dir=RUN_DIR / "scratch",
    )
    CONTINUATION_RECORD["equivalence_to_fresh_fit"] = EQUIVALENCE
STORE.save("continuation", "record", CONTINUATION_RECORD)

print(f"prompts fitted   {CONTINUATION.n_done:,} "
      f"(the parent's {PARENT.accumulator.n_done} were skipped, not refitted)")
print(f"skipped          {CONTINUATION.n_skipped}")
print(f"elapsed          {CONTINUATION.elapsed_seconds / 3600:.3f} h")
print()
print("scale snapshots")
for _scale, _snapshot in sorted(CONTINUATION.snapshots.items()):
    print(f"  {_scale:>7,}  n_prompts={_snapshot.n_prompts:<7,} {_snapshot.checksum}")
if EQUIVALENCE is not None:
    print()
    print(f"continuation == fresh nested fit at {max(SCALES)}: "
          f"{EQUIVALENCE['bit_identical']} (bit-identical, {len(EQUIVALENCE['layers_compared'])} layers)")
else:
    print()
    print("Bit-identity with a fresh nested fit is proved in MOCK and in the test")
    print("suite; at this scale the guarantee rests on the verified prompt prefix")
    print("and upstream's own resume path.")
'''
)

# ============================================== 14. development evaluation

markdown(
    """
## 14. Evaluate the fresh development set at every scale

Native readout on 256 held-out prompts the lens never saw, against three controls
and one diagnostic, scored with the project's standard tie-aware scorer.

Scored at **100, 250 and 1,000**. The 100-prompt point is the parent's lens read
read-only and is **descriptive**: it exists so the plateau rule has three points
and can report something measured rather than reaching `PLATEAU_REACHED` by
construction. It can never be newly confirmed.

Continuous metrics are printed for **every** layer at **every** scale, pass or
fail.
"""
)

code(
    '''
# 14a. The readout. One forward per prompt; every variant reads the same
# activations, so controls are matched by construction.
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
        ids = MODEL.encode(prompt, max_length=ACTIVE_PLAN.max_seq_len)
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


def lens_for_scale(scale):
    """The lens at one development scale point.

    The baseline scale is the PARENT's snapshot, opened read-only. It is a
    descriptive comparison point and is never re-confirmed.
    """
    from jlens.lens import JacobianLens

    if int(scale) == int(PARENT_BASELINE):
        return JacobianLens.load(
            str(Path(PARENT.root) / PARENT.layout["required"]["baseline_lens"])
        )
    return CONTINUATION.lens_for_scale(int(scale))
'''
)

code(
    '''
# 14b. Development verdicts at every scale.
#
# MOCK note: a tiny random stack has no interesting per-layer structure, so the
# MOCK path scores FIXTURE rows with known archetypes instead. The rows are
# synthetic; the SCORER, the GATE, the folds and the comparison are the real ones.
from jlens.calibration.gate import eligible_layers, evaluate_calibration_layers

MOCK_SCENARIO = "l32_late_pass"

# RUN_FRESH_DEVELOPMENT is enforced in section 12, before the expensive cell.
DEVELOPMENT = {}
for _index, _scale in enumerate(DEV_SCALES):
    _key = f"scale{_scale}"
    _stored = STORE.load("validation", _key)
    if _stored is not None:
        DEVELOPMENT[_scale] = {int(k): v for k, v in _stored["by_layer"].items()}
        print(f"scale {_scale:>7,}  reused stored development verdicts")
        continue
    if MODEL_STAGES_ENABLED:
        _rows = score_readout_rows(
            lens_for_scale(_scale),
            DEVELOPMENT_PROMPTS,
            list(ACTIVE_PLAN.layers),
            ACTIVE_PLAN.target_layer,
        )
    else:
        from jlens.calibration.extension_mock import (  # noqa: PLC0415
            mock_extension_rows,
        )

        _rows = mock_extension_rows(
            MOCK_SCENARIO,
            stage="development",
            scale=_scale,
            scale_index=_index,
            n_prompts=N_DEVELOPMENT_PROMPTS,
        )
    DEVELOPMENT[_scale] = evaluate_calibration_layers(
        _rows,
        layers=list(ACTIVE_PLAN.layers),
        scale=_scale,
        stage="validation",
        gate=EXTENSION_GATE,
    )
    STORE.save(
        "validation",
        _key,
        {
            "scale": _scale,
            "descriptive_only": bool(int(_scale) == int(PARENT_BASELINE)),
            "by_layer": {str(k): v for k, v in DEVELOPMENT[_scale].items()},
        },
    )
    _tag = "  (descriptive)" if int(_scale) == int(PARENT_BASELINE) else ""
    print(f"scale {_scale:>7,}  eligible {eligible_layers(DEVELOPMENT[_scale])}{_tag}")

GRID_LAYERS = sorted(DEVELOPMENT[DEV_SCALES[-1]])
print()
print(f"{'layer':>6} {'role':>12} {'scale':>8} {'pass':>6} {'MRR':>8} {'midrank':>9} "
      f"{'opt':>7} {'tie@max':>8} {'top10':>7} {'distinct':>9}")
for _layer in GRID_LAYERS:
    _role = (
        "PRIMARY" if _layer == PRIMARY_EARLY_LAYER
        else "stretch" if _layer == SECONDARY_EARLY_LAYER
        else "published" if _layer in (35, 38, 40)
        else "descriptive"
    )
    for _scale in DEV_SCALES:
        _v = DEVELOPMENT[_scale][_layer]
        _m = _v["metrics"]["j_lens"]
        print(f"{_layer:>6} {_role:>12} {_scale:>8,} "
              f"{('PASS' if _v['passed'] else 'fail'):>6} "
              f"{_m['mean_reciprocal_rank']:>8.4f} {_m['median_midrank']:>9.2f} "
              f"{_m['median_optimistic_rank']:>7.1f} {_m['tied_at_max_rate']:>8.3f} "
              f"{_m['top10_inclusion']:>7.3f} "
              f"{_v['target_diversity']['n_distinct_target_tokens']:>9}")
'''
)

# ====================================== 15. scale comparison and plateau

markdown(
    """
## 15. Compare the scale points and apply the plateau rule

Does more calibration data buy anything at the earlier layers, and has it stopped
buying? The `opt` column beside `midrank` is the whole reason the gate is
tie-aware: a layer can read rank 1 optimistically and sit halfway down a tie
block.

The plateau rule is the completed study's, **unchanged**, and its digest is bound
into the run fingerprint.
"""
)

code(
    '''
# 15. The comparison table, its deltas, and the plateau verdict.
from jlens.calibration.scale import compare_scales, evaluate_plateau

COMPARISON = compare_scales(DEVELOPMENT, layers=GRID_LAYERS)
PLATEAU = evaluate_plateau(COMPARISON)
STORE.save(
    "scale_comparison",
    "comparison",
    {"comparison": COMPARISON, "plateau": PLATEAU},
)

print("eligible layers by scale (development only)")
for _scale, _layers in COMPARISON["eligible_by_scale"].items():
    _tag = "  descriptive" if int(_scale) == int(PARENT_BASELINE) else ""
    print(f"  {int(_scale):>7,}  {_layers}{_tag}")
print()
if COMPARISON["deltas"]:
    print(f"{'layer':>6} {'step':>18} {'dMRR':>9} {'midrank drop':>14} "
          f"{'dTie@max':>10}  pass")
    for _row in COMPARISON["deltas"]:
        _step = f"{_row['from_scale']:,}->{_row['to_scale']:,}"
        _before = "PASS" if _row["passed_before"] else "fail"
        _after = "PASS" if _row["passed_after"] else "fail"
        print(f"{_row['layer']:>6} {_step:>18} "
              f"{_row['delta_mrr']:>+9.4f} "
              f"{_row['median_midrank_relative_drop']:>+13.1%} "
              f"{_row['delta_tied_at_max_rate']:>+10.4f}  "
              f"{_before}->{_after}")
print()
print(f"plateau verdict        {PLATEAU['verdict']}")
print(f"extension justified    {PLATEAU['extension_justified']}")
print(f"runs automatically     {PLATEAU.get('runs_automatically', False)}")
for _clause in PLATEAU["clauses"]:
    print(f"  [{'pass' if _clause['passed'] else 'FAIL'}] {_clause['clause']}")
    print(f"         {_clause['detail']}")
print()
print("This rule is diagnostic. It authorizes no scale beyond 1,000 and starts")
print("nothing by itself.")
'''
)

# ============================================ 16. freeze the scale selection

markdown(
    """
## 16. Freeze the scale selection

Written atomically **before** the confirmation set is touched. The vault that
holds the confirmation prompts will not open without this payload, and the
payload asserts `confirmation_not_consulted` — so "the confirmation set
influenced nothing" is a property of the code, not a claim in a document.

The rule was fixed in section 5, before any number existed. The descriptive
baseline scale is reported and is **not selectable**.
"""
)

code(
    '''
# 16. Apply the predeclared selection rule and record the choice atomically.
from jlens.calibration.extension import select_extension_scale

SELECTION = select_extension_scale(
    COMPARISON,
    plateau=PLATEAU,
    rule=EXTENSION_SELECTION_RULE,
    candidate_scales=SCALES,
)
STORE.save("scale_selection", "selection", SELECTION)

print(f"selected scale             {SELECTION['selected_scale']:,}")
print(f"clause applied             {SELECTION['clause_applied']}")
print(f"reason                     {SELECTION['reason']}")
print(f"candidate scales           {SELECTION['candidate_scales']}")
print(f"descriptive (not selectable) {SELECTION['descriptive_scales_not_selectable']}")
print(f"eligible earlier by scale  {SELECTION['eligible_earlier_by_scale']}")
print(f"plateau reached            {SELECTION['plateau_reached']} "
      f"(clause informative: {SELECTION['plateau_clause_informative']})")
print(f"confirmation consulted     {not SELECTION['confirmation_not_consulted']}")
print(f"multimodal consulted       {not SELECTION['multimodal_outcomes_not_consulted']}")
print(f"selection checksum         {SELECTION['selection_checksum']}")
'''
)

# ================================================== 17. final confirmation

markdown(
    """
## 17. Run the untouched final confirmation

The fresh confirmation set is held in a vault that refuses to release it until a
scale selection has been recorded against it. It is unlocked **exactly once**,
only the selected scale is evaluated, and the same frozen gate applies.

Every layer's result is recorded, including every failure. No layer is retried at
another scale after this, and no threshold changes after this.
"""
)

code(
    '''
# 17. Open the fresh vault (only against a recorded selection) and confirm.
from jlens.calibration.publication import ConfirmationVault

VAULT = ConfirmationVault(records=SPLITS.confirmation)
CONFIRMATION = None

if not RUN_FINAL_CONFIRMATION:
    print("RUN_FINAL_CONFIRMATION is False — the fresh confirmation set stays locked.")
    print(f"vault status: {VAULT.status()}")
    print()
    print("Note: 'not run' is NOT the same as 'nothing passed'. No layer has been")
    print("offered the fresh confirmation set.")
else:
    VAULT.unlock(SELECTION)
    _records = VAULT.open()
    _scale = SELECTION["selected_scale"]
    _stored = STORE.load("confirmation", f"scale{_scale}")
    if _stored is not None:
        CONFIRMATION = {int(k): v for k, v in _stored["by_layer"].items()}
        print("reused stored confirmation verdicts")
    else:
        if MODEL_STAGES_ENABLED:
            _rows = score_readout_rows(
                lens_for_scale(_scale),
                CONFIRMATION_PROMPTS,
                list(ACTIVE_PLAN.layers),
                ACTIVE_PLAN.target_layer,
            )
        else:
            from jlens.calibration.extension_mock import (  # noqa: PLC0415
                mock_extension_rows,
            )

            _rows = mock_extension_rows(
                MOCK_SCENARIO,
                stage="confirmation",
                scale=_scale,
                scale_index=list(DEV_SCALES).index(_scale),
                n_prompts=N_CONFIRMATION_PROMPTS,
            )
        CONFIRMATION = evaluate_calibration_layers(
            _rows,
            layers=list(ACTIVE_PLAN.layers),
            scale=_scale,
            stage="confirmation",
            gate=EXTENSION_CONFIRMATION_GATE,
        )
        STORE.save(
            "confirmation",
            f"scale{_scale}",
            {
                "scale": _scale,
                "selection": SELECTION,
                "by_layer": {str(k): v for k, v in CONFIRMATION.items()},
            },
        )
    print(f"vault status: {VAULT.status()}")
    print()
    print(f"{'layer':>6} {'pass':>6} {'MRR':>8} {'midrank':>9} {'top10':>7}  failed clauses")
    for _layer in sorted(CONFIRMATION):
        _v = CONFIRMATION[_layer]
        _m = _v["metrics"]["j_lens"]
        print(f"{_layer:>6} {('PASS' if _v['passed'] else 'fail'):>6} "
              f"{_m['mean_reciprocal_rank']:>8.4f} {_m['median_midrank']:>9.2f} "
              f"{_m['top10_inclusion']:>7.3f}  {_v['failed_checks'] or ''}")
'''
)

# ======================================================== 18. publication

markdown(
    """
## 18. Publish validated earlier lenses

One artifact per **earlier** layer that passed the fresh confirmation gate.
Publication is refused for a layer with no confirmation verdict, a layer whose
verdict failed, a development verdict offered in place of a confirmation one, a
layer outside the publication targets (L26 and L32), a destination outside the
extension's own run, and any destination inside the parent run.

**The existing L35 / L38 / L40 publications are untouched.** Their confirmation
numbers here are reported and nothing is written for them.

**Fitting completing is not a reason to publish anything.**
"""
)

code(
    '''
# 18. Publish, or explain precisely why nothing was published.
from jlens.calibration.extension import publish_early_layer
from jlens.calibration.publication import PublicationRefused, record_failed_layer

PUBLISHED, FAILED, PUBLICATION = [], [], None

if not PUBLISH_VALIDATED_EARLY_LENSES:
    print("PUBLISH_VALIDATED_EARLY_LENSES is False — nothing is written.")
elif CONFIRMATION is None:
    print("No confirmation results exist. Publication requires the untouched")
    print("confirmation set; a lens is never published because fitting finished.")
else:
    _scale = SELECTION["selected_scale"]
    _lens = CONTINUATION.lens_for_scale(_scale)
    for _layer in sorted(CONFIRMATION):
        _verdict = CONFIRMATION[_layer]
        _development = DEVELOPMENT[_scale][_layer]
        if _layer not in PUBLISHABLE_LAYERS:
            print(f"  L{_layer}: not a publication target "
                  f"{list(PUBLISHABLE_LAYERS)}; result recorded, nothing written")
            continue
        if not _verdict["passed"]:
            FAILED.append(
                record_failed_layer(
                    layer=_layer,
                    scale=_scale,
                    confirmation_verdict=_verdict,
                    validation_verdict=_development,
                )
            )
            continue
        try:
            PUBLISHED.append(
                publish_early_layer(
                    layer=_layer,
                    scale=_scale,
                    lens=_lens,
                    destination=STORE.published_path(_layer, _scale),
                    confirmation_verdict=_verdict,
                    development_verdict=_development,
                    vault=VAULT,
                    parent=PARENT,
                    parent_audit=PARENT_AUDIT,
                    continuation=CONTINUATION_RECORD,
                    splits=SPLITS,
                    selection=SELECTION,
                    extension_run_dir=RUN_DIR,
                    load_info=LOAD_INFO,
                    corpus_manifest=CORPUS_MANIFEST,
                    capture_plan=ACTIVE_PLAN.to_dict(),
                    fitting_diagnostics=CONTINUATION.to_dict(),
                    environment=ENVIRONMENT,
                )
            )
        except PublicationRefused as error:
            print(f"  refused L{_layer}: {error}")
            FAILED.append(
                record_failed_layer(
                    layer=_layer,
                    scale=_scale,
                    confirmation_verdict=_verdict,
                    validation_verdict=_development,
                )
            )
    PUBLICATION = {
        "n_published": len(PUBLISHED),
        "n_failed": len(FAILED),
        "published_layers": sorted(int(a["physical_layer"]) for a in PUBLISHED),
        "failed_layers": sorted(int(a["physical_layer"]) for a in FAILED),
        "published_checksums": {
            str(a["physical_layer"]): a["lens_checksum"] for a in PUBLISHED
        },
        "failed_layers_marked_validated": False,
        "publication_targets": list(PUBLISHABLE_LAYERS),
        "existing_publications_unchanged": [35, 38, 40],
    }
    STORE.save("publication", f"scale{_scale}", PUBLICATION)
    print(f"published {PUBLICATION['n_published']} earlier layer(s): "
          f"{PUBLICATION['published_layers']}")
    print(f"failed    {PUBLICATION['n_failed']} layer(s): "
          f"{PUBLICATION['failed_layers']} (diagnostics kept, validated=false)")
    print(f"existing L35/L38/L40 publications unchanged: "
          f"{PUBLICATION['existing_publications_unchanged']}")
'''
)

# ========================================= 19. verdict, report, immutability

markdown(
    """
## 19. Early-layer verdict, report, and the parent-immutability proof

The verdict keys on the fresh confirmation set alone. `GO` requires L32 or L26 to
have passed the frozen gate on data nothing has seen. Anything else is
`EARLY_LAYER_CALIBRATION_NO_GO`, stated plainly.

Finally the parent run is checksummed again and compared byte for byte with the
checksums taken in section 6. This extension never opens a parent file for
writing; the proof is what makes that auditable rather than asserted.
"""
)

code(
    '''
# 19. The verdict, the report, and the proof that the parent is untouched.
from jlens.calibration.extension import early_layer_verdict
from jlens.calibration.parent import assert_parent_unchanged

VERDICT = None
if CONFIRMATION is not None:
    VERDICT = early_layer_verdict(
        CONFIRMATION,
        scale=SELECTION["selected_scale"],
        selection=SELECTION,
        development=DEVELOPMENT[SELECTION["selected_scale"]],
    )
    STORE.save("early_layer_verdict", "verdict", VERDICT)

PARENT_CHECKSUMS_AFTER = protected_parent_checksums(PARENT_ROOT, layout=PARENT.layout)
IMMUTABILITY = assert_parent_unchanged(PARENT_CHECKSUMS_BEFORE, PARENT_CHECKSUMS_AFTER)
STORE.save("parent_import", "immutability_proof", IMMUTABILITY)

RESUME = STORE.status_report()
REPORT = {
    "schema": "jlens.calibration.early_layer_extension_report.v1",
    "mode": MODE,
    "protocol_version": PROTOCOL_VERSION,
    "protocol": EXTENSION_PROTOCOL.to_dict(),
    "protocol_digest": EXTENSION_PROTOCOL.digest,
    "fingerprint_digest": FINGERPRINT.digest,
    "gate_digest": EXTENSION_GATE.digest,
    "selection_rule_digest": EXTENSION_SELECTION_RULE.digest,
    "parent_provenance": PROVENANCE,
    "parent_immutability_proof": IMMUTABILITY,
    "corpus": CORPUS_MANIFEST,
    "fresh_splits": SPLITS_MANIFEST,
    "fresh_split_leakage_audit": SPLIT_LEAKAGE,
    "development_diversity": DIVERSITY,
    "confirmation_diversity": CONFIRMATION_DIVERSITY,
    "budget": BUDGET,
    "continuation": CONTINUATION_RECORD,
    "scale_comparison": COMPARISON,
    "plateau": PLATEAU,
    "scale_selection": SELECTION,
    "confirmation_vault": VAULT.status(),
    "confirmation": (
        {str(k): v for k, v in CONFIRMATION.items()} if CONFIRMATION else None
    ),
    "early_layer_verdict": VERDICT,
    "publication": PUBLICATION,
    "resume": RESUME,
    "environment": ENVIRONMENT,
    "mock_proves_pipeline_only": MODE != "real",
}

REPORT_DIR = RUN_DIR / "artifacts"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
(REPORT_DIR / "early_layer_extension_report.json").write_text(
    json.dumps(REPORT, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
)

print(f"parent immutable   {IMMUTABILITY['immutable']} "
      f"({IMMUTABILITY['n_files_checked']} files re-checksummed)")
print(f"report             {REPORT_DIR / 'early_layer_extension_report.json'}")
print(f"resume             {RESUME['status']}")
print(f"checkpoint present {RESUME['checkpoint_present']}")
print(f"units              {RESUME['completed_units']}")
print(f"invalid units      {len(RESUME['invalid_units'])}")
print()
if VERDICT is None:
    print("NO VERDICT — the fresh confirmation set was never opened.")
    print("This is NOT the same as 'nothing passed'.")
else:
    print(f"VERDICT            {VERDICT['verdict']}")
    print(f"  earlier layers passing confirmation: "
          f"{VERDICT['earlier_layers_passing_confirmation']}")
    print(f"  {VERDICT['statement']}")
print()
if MODE != "real":
    print("MOCK RUN COMPLETE. Pipeline behaviour only — no evidence about Gemma 4,")
    print("about layer 32, about layer 26, or about whether a validated earlier")
    print("readout exists.")
else:
    print("Send back artifacts/early_layer_extension_report.json plus the")
    print("development table, the selection block and the confirmation table.")
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
