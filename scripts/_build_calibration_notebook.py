# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Generate ``notebooks/research_grade_multilayer_jlens_calibration_colab.ipynb``.

Written from source rather than edited as JSON, so the committed notebook stays
output-free and byte-reproducible. Run
``python scripts/_build_calibration_notebook.py`` after changing a cell; a test
regenerates it and fails on drift.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TARGET = (
    REPO_ROOT
    / "notebooks"
    / "research_grade_multilayer_jlens_calibration_colab.ipynb"
)

CELLS: list[tuple[str, str]] = []


def markdown(text: str) -> None:
    CELLS.append(("markdown", text.strip("\n")))


def code(text: str) -> None:
    CELLS.append(("code", text.strip("\n")))


# ============================================================== introduction

markdown(
    """
# Research-grade multilayer text-only J-lens calibration

**One question.** The v2 lens was fitted on **32 prompts** and validated at
layer 38 alone; layers 20, 26 and 32 failed. Anthropic's own lenses use
**1,000 sequences**. *Was the earlier-layer failure caused by inadequate
calibration, or does it persist under substantially stronger calibration?*

Eight physical layers — **8, 14, 20, 26, 32, 35, 38, 40** — three nested corpus
scales, one frozen tie-aware gate, one untouched confirmation set. Text only.

## Read this before running anything

**There is no fitting objective and no optimizer.** `J_l = E[∂h_final/∂h_l]` is
a population mean estimated by a running average
(`jlens/fitting.py`, upstream and unmodified). Nothing is minimized; nothing
converges in the optimization sense. Increasing the prompt count reduces
**estimator variance** and does nothing else. Any document that gives this
workflow an "optimizer configuration" is describing a different object.

**The scale points are free.** Because the estimator is a running mean over a
deterministically ordered list, the accumulator at prompt 1,000 *is* the 1k
lens. The 100/250/1k lenses are snapshots of **one** accumulator, so all three
cost exactly as much as the 1k lens alone.

**The primary source predicts a plateau immediately.** The upstream README
states: *"The paper's lenses use 1000 sequences of 128 tokens from a
pretraining-like corpus. Quality saturates quickly (§9.3); ~100 prompts is
usable."* Our first scale point tests that usable-scale claim, while the
1,000-prompt endpoint matches the paper's production scale. **If earlier layers
fail at 1k, there is no source-backed reason to expect larger runs
to rescue them** — and that is a real answer, not a disappointing one.

**The budget is staged and section 9 says so in numbers.** The 100- and
250-prompt checkpoints test the upstream saturation claim before the
paper-matched 1,000-prompt endpoint commits ~23 L4-hours. Read section 9 before
setting any switch.

**Nothing starts by itself.** `RUN_REAL_CALIBRATION`, `RUN_MODEL_STAGES`, every
`CONFIRM_*_BUDGET`, `RUN_FINAL_CONFIRMATION` and
`PUBLISH_VALIDATED_LENSES` are all False in the committed notebook and must each
be set by hand. Opening this notebook and running every cell performs a
deterministic MOCK run and touches no model, no Hub, no Drive and no corpus.

**MOCK success proves pipeline behaviour only.** It is not evidence about
Gemma 4, about any layer, or about whether a research-grade lens exists.

Reference: [`docs/research_grade_jlens_methodology.md`](../docs/research_grade_jlens_methodology.md)
and [`docs/research_grade_jlens_calibration_protocol.md`](../docs/research_grade_jlens_calibration_protocol.md).
"""
)

# ================================================================= 0. bootstrap

markdown(
    """
## 0. Bootstrap repository

Clone or update the checkout and make `import jlens` resolve to it. Nothing
from the repository is imported until the last cell of this section.
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
REPO_PATH = Path(os.environ.get("RGCALIB_REPO_DIR") or REPO_DIR)


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
## 1. Configuration

Every switch is False. Setting one is a deliberate act, and the budget switches
are separate from the run switches so that "I want to do this" and "I have read
what it costs" are two different decisions.

The scale schedule is derived from the budget confirmations and is **nested**:
you cannot confirm 250 without also confirming 100, or 1,000 without both,
because every larger lens is the smaller accumulator continued.
"""
)

code(
    '''
# 1. Configuration. Requires section 0 (it imports from the repository).
# Nothing here mounts Drive, reads data, or loads a model.

# ---- run switches --------------------------------------------------------
RUN_REAL_CALIBRATION = False
RUN_MODEL_STAGES = False
RUN_FINAL_CONFIRMATION = False
PUBLISH_VALIDATED_LENSES = False
# ---- budget confirmations (read section 9 first) -------------------------
CONFIRM_100_BUDGET = False
CONFIRM_250_BUDGET = False
CONFIRM_1K_BUDGET = False

import json
from pathlib import Path

from jlens.calibration.baseline import baseline_manifest, format_baseline_manifest
from jlens.calibration.gate import (
    CALIBRATION_GATE,
    CALIBRATION_VALIDITY_PROTOCOL,
    CONFIRMATION_PROMPT_SEED,
    CONTROL_SEED,
    N_CONFIRMATION_PROMPTS,
    N_VALIDATION_PROMPTS,
    VALIDATION_PROMPT_SEED,
    gate_text,
)
from jlens.calibration.plan import (
    CALIBRATION_LAYERS,
    SCALE_POINTS,
    build_capture_plan,
    normalized_depth,
)
from jlens.calibration.scale import PLATEAU_RULE

CONFIG_PATH = Path("configs/research_grade_jlens_calibration_v1.json")
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
PROTOCOL_VERSION = CONFIG["protocol_version"]

# ---- the frozen grid -----------------------------------------------------
LAYERS = CALIBRATION_LAYERS
TARGET_LAYER = CONFIG["sites"]["target_layer"]
MAX_SEQ_LEN = CONFIG["fitting"]["max_seq_len"]
SKIP_FIRST = CONFIG["positions"]["skip_first"]
DIM_BATCH = CONFIG["fitting"]["dim_batch"]

if tuple(CONFIG["sites"]["source_layers"]) != tuple(LAYERS):
    raise RuntimeError(
        "the frozen config and the module disagree about the layer grid; "
        "refusing to answer a different question than the one that was frozen"
    )

# ---- the nested scale schedule ------------------------------------------
_CONFIRMED = [
    scale
    for scale, confirmed in zip(
        SCALE_POINTS, (CONFIRM_100_BUDGET, CONFIRM_250_BUDGET, CONFIRM_1K_BUDGET)
    )
    if confirmed
]
if _CONFIRMED and _CONFIRMED != list(SCALE_POINTS[: len(_CONFIRMED)]):
    raise RuntimeError(
        f"budget confirmations {_CONFIRMED} are not a nested prefix of "
        f"{list(SCALE_POINTS)}. The larger lens IS the smaller accumulator "
        "continued, so a larger scale cannot be run without the smaller one."
    )
ACTIVE_SCALE_POINTS = tuple(_CONFIRMED)

MODEL_STAGES_ENABLED = bool(
    RUN_REAL_CALIBRATION and RUN_MODEL_STAGES and ACTIVE_SCALE_POINTS
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

print(f"protocol          {PROTOCOL_VERSION}")
print(f"mode              {MODE}")
print(f"layers            {list(LAYERS)}")
print(f"normalized depth  {[normalized_depth(l) for l in LAYERS]}")
print(f"target layer      L{TARGET_LAYER}   plan digest {PLAN.digest}")
print(f"scale points      configured {list(SCALE_POINTS)}  active {list(ACTIVE_SCALE_POINTS)}")
print(f"gate              {CALIBRATION_VALIDITY_PROTOCOL}")
print(f"gate digest       {CALIBRATION_GATE.digest}")
print(f"plateau digest    {PLATEAU_RULE.digest}")
print(f"validation/confirmation prompts   {N_VALIDATION_PROMPTS} / {N_CONFIRMATION_PROMPTS}")
if not MODEL_STAGES_ENABLED:
    print()
    print("MOCK MODE — no Gemma, no Hub, no Drive, no corpus download.")
    print("This proves pipeline behaviour only. It is not evidence about Gemma.")
'''
)

# ================================================================== 2. drive

markdown(
    """
## 2. Optional mount Google Drive

Only mounted for a real run, and only to hold the run directory. The 210 MiB
accumulator checkpoint and the scale snapshots live here so that a disconnected
session loses at most one checkpoint interval.

**Put the HuggingFace cache on Drive too.** At ~16 GB per model download and 1–3
sessions, re-downloading Gemma every session is the second-largest hidden cost
in this study.
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
    # Keep the 16 GB model cache off the ephemeral runtime disk.
    os.environ.setdefault("HF_HOME", str(DRIVE_ROOT / "hf_cache"))
    print(f"drive    {DRIVE_ROOT}")
    print(f"HF_HOME  {os.environ['HF_HOME']}")
else:
    DRIVE_ROOT = Path(tempfile.mkdtemp(prefix="rgcalib_mock_"))
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
            "a real calibration needs a GPU runtime; this one has no CUDA device"
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

# ====================================================== 5. methodology + gate

markdown(
    """
## 5. Methodology and frozen protocol

Printed **before** any result-producing cell runs. Everything below is
checksummed into the run fingerprint: editing a threshold invalidates stored
results rather than rescoring them, which is what makes "predeclared" mean
something operationally.

The baseline manifest documents what this project already established. The
calibration reads **no** multimodal run result; the v2 lens is listed as
readable only as a comparison baseline.
"""
)

code(
    '''
# 5. The frozen protocol, the gate, the plateau rule, and prior evidence.
print(gate_text())
print()
print(PLATEAU_RULE.text())
'''
)

code(
    '''
# 5b. What the estimator is — and what it is not.
print("ESTIMATOR")
print("  J_l = E[dh_final / dh_l], estimated by a running mean over prompts.")
print("  Implementation: jlens.fitting.fit (upstream Anthropic code, unmodified).")
print(f"  Upstream commit: {ENVIRONMENT.get('upstream_commit')}")
print()
print("  fitting objective   " + CONFIG["fitting"]["objective"])
print("  optimizer           " + CONFIG["fitting"]["optimizer"])
print("  schedule            " + CONFIG["fitting"]["schedule"])
print("  " + CONFIG["fitting"]["why_not_applicable"])
print()
print("  Consequence 1: more prompts reduces estimator VARIANCE and nothing else.")
print("  Consequence 2: scale points nest exactly, so 100/250/1k cost as much")
print("                 as 1k alone (snapshots of one accumulator).")
print("  Consequence 3: the upstream README says quality saturates quickly and")
print("                 ~100 prompts is usable; that is our first point.")
print()

BASELINE = baseline_manifest()
print(format_baseline_manifest(BASELINE))
'''
)

# =================================================== 6. corpus and splits

markdown(
    """
## 6. Corpus audit and deterministic splits

Assignment is a stable hash of the record ID, not a position, so partitions are
independent by construction and a record's partition never depends on how many
records were drawn.

Duplicate control is two-layered: a sha256 over normalized text for exact
duplicates, and a banded 64-bit SimHash over word 3-grams for near duplicates.
A cross-partition hit **raises** — dropping the offending record would silently
change the split that every downstream checksum is bound to.
"""
)

code(
    '''
# 6. Stream (or synthesize) the corpus, split it, and audit for leakage.
from jlens.calibration.corpus import (
    audit_leakage,
    collect_records_for_partition_quotas,
    corpus_manifest,
    scale_nesting_audit,
)

CORPUS_CONFIG = dict(CONFIG["corpus"]["primary"])
SPLIT_SEED = CONFIG["splits"]["split_seed"]
PLANNED_SCALES = ACTIVE_SCALE_POINTS or SCALE_POINTS

if MODEL_STAGES_ENABLED:
    from datasets import load_dataset  # noqa: PLC0415

    CORPUS_ID = f"{CORPUS_CONFIG['config']}/{CORPUS_CONFIG['split']}"
    # A hard safety bound, not a target count. Hash partitioning is stochastic
    # in count, so collection continues until every exact quota is satisfied.
    _max_texts = max(10_000, 8 * (max(PLANNED_SCALES) + N_VALIDATION_PROMPTS + N_CONFIRMATION_PROMPTS))
    _stream = load_dataset(
        CORPUS_CONFIG["hf_dataset"],
        CORPUS_CONFIG["config"],
        split=CORPUS_CONFIG["split"],
        streaming=True,
    )
    _texts = (_record["text"] for _record in _stream)
    CORPUS_CONFIG["revision_status"] = "RESOLVE_AND_RECORD_ON_FIRST_REAL_RUN"
else:
    from jlens.calibration.mock import mock_corpus_texts  # noqa: PLC0415

    CORPUS_ID = "mock/train"
    CORPUS_CONFIG = {
        "hf_dataset": "mock",
        "config": "mock",
        "split": "train",
        "revision": "mock",
        "revision_status": "MOCK_NO_CORPUS_WAS_DOWNLOADED",
        "min_chars": 100,
        "license": "n/a",
    }
    _max_texts = 10_000
    _texts = mock_corpus_texts(_max_texts)

RECORDS, PARTITIONS = collect_records_for_partition_quotas(
    corpus_id=CORPUS_ID,
    texts=_texts,
    min_chars=CORPUS_CONFIG["min_chars"],
    min_fit=max(PLANNED_SCALES),
    max_texts=_max_texts,
    seed=SPLIT_SEED,
    n_validation=N_VALIDATION_PROMPTS,
    n_confirmation=N_CONFIRMATION_PROMPTS,
)

# MOCK uses small synthetic equivalents of 100/250/1k; nesting is what is tested.
if MODEL_STAGES_ENABLED:
    SCALES = tuple(PLANNED_SCALES)
else:
    from jlens.calibration.mock import MOCK_SCALE_POINTS  # noqa: PLC0415

    SCALES = MOCK_SCALE_POINTS

LEAKAGE = audit_leakage(PARTITIONS, scale_points=SCALES)
NESTING = scale_nesting_audit(PARTITIONS.fit, SCALES)
CORPUS_MANIFEST = corpus_manifest(
    PARTITIONS, corpus_config=CORPUS_CONFIG, scale_points=SCALES
)

print(f"corpus            {CORPUS_ID}")
print(f"revision          {CORPUS_CONFIG['revision']}  ({CORPUS_CONFIG['revision_status']})")
print(f"records kept      {len(RECORDS):,}")
print(f"exact dups dropped {len(PARTITIONS.dropped_exact_duplicates):,}")
for _name in ("fit", "validation", "confirmation"):
    print(f"  {_name:<13} {len(PARTITIONS.get(_name)):>7,}  {PARTITIONS.checksum(_name)}")
print(f"scale points      {list(SCALES)}")
print(f"nesting exact     {NESTING['nested']}")
print(
    f"leakage audit     {'CLEAN' if LEAKAGE['ok'] else 'FAILED'} — "
    f"{LEAKAGE['n_exact_hits']} exact, {LEAKAGE['n_near_hits']} near, "
    f"{LEAKAGE['candidate_pairs_compared']:,} candidate pairs compared"
)
'''
)

# ================================================== 7. target diversity audit

markdown(
    """
## 7. Target-token diversity audit

The 32-prompt v2 validation accidentally contained only **seven** distinct model
output targets. Over 128 prompts this workflow requires **≥ 24 distinct targets
and no single target above 25% of prompts**, and it **refuses** rather than
lowering either threshold.

Targets are the frozen model's own final-layer argmax. No lens and no candidate
layer is consulted — `select_diverse_validation_prompts` has no parameter
through which one could be supplied.
"""
)

code(
    '''
# 7. Choose the held-out sets, stratified on the model's own output target.
from jlens.calibration.gate import (
    audit_target_diversity,
    ordinary_next_token_argmax,
    select_diverse_validation_prompts,
)

if MODEL_STAGES_ENABLED:
    def _target_token(prompt):
        # The model's ordinary output path only. No lens, no candidate layer.
        return ordinary_next_token_argmax(
            MODEL, prompt, max_length=MAX_SEQ_LEN
        )
else:
    def _target_token(prompt):
        return int(prompt.split()[1]) % 40

if MODEL_STAGES_ENABLED:
    # Deferred: the model does not exist until section 8.
    VALIDATION_PROMPTS = VALIDATION_MANIFEST = None
    CONFIRMATION_PROMPTS = CONFIRMATION_MANIFEST = None
    DIVERSITY = {"deferred_until_model_is_loaded": True}
    print("Real run: target discovery needs the model; deferred to section 8.")
else:
    VALIDATION_PROMPTS, VALIDATION_MANIFEST = select_diverse_validation_prompts(
        [record.text for record in PARTITIONS.validation],
        n_prompts=N_VALIDATION_PROMPTS,
        seed=VALIDATION_PROMPT_SEED,
        target_token_for_prompt=_target_token,
    )
    CONFIRMATION_PROMPTS, CONFIRMATION_MANIFEST = select_diverse_validation_prompts(
        [record.text for record in PARTITIONS.confirmation],
        n_prompts=N_CONFIRMATION_PROMPTS,
        seed=CONFIRMATION_PROMPT_SEED,
        target_token_for_prompt=_target_token,
    )
    DIVERSITY = VALIDATION_MANIFEST["diversity"]
    print(f"validation prompts    {len(VALIDATION_PROMPTS)}")
    print(f"  distinct targets    {DIVERSITY['n_distinct_target_tokens']} "
          f"(floor {DIVERSITY['min_distinct_target_tokens']})")
    print(f"  max target share    {DIVERSITY['max_target_token_share']:.1%} "
          f"(ceiling {DIVERSITY['max_target_token_share_allowed']:.0%})")
    print(f"  passed              {DIVERSITY['passed']}")
    print(f"confirmation prompts  {len(CONFIRMATION_PROMPTS)} "
          f"(held back; see section 16)")
    print(f"  selection checksum  {VALIDATION_MANIFEST['selection_checksum']}")
'''
)

# ================================================ 8. model + architecture audit

markdown(
    """
## 8. Model architecture and hook audit

Verifies every assumption the estimator depends on before any GPU time is
committed: dense routing, 42 layers, `d_model` 2560, vocab 262144, frozen
parameters, tied unembedding, and the residual site the lens reads.

In MOCK mode this is a tiny frozen CPU stack with the same interface — real
autograd, real hooks, real multilayer capture.
"""
)

code(
    '''
# 8. Load (or mock) the model and audit the architecture and the hook site.
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
else:
    from jlens.calibration.mock import MockCalibrationModel  # noqa: PLC0415

    MODEL = MockCalibrationModel()
    LOAD_INFO = {
        "model_repo_id": "mock",
        "model_revision": "0" * 40,
        "tokenizer_repo_id": "mock",
        "tokenizer_revision": "0" * 40,
        "dtype": "float32",
        "device_map": "cpu",
    }
    ARCHITECTURE = {
        "model_class": type(MODEL).__name__,
        "n_layers": MODEL.n_layers,
        "d_model": MODEL.d_model,
        "vocab_size": MODEL.vocab_size,
        "params_frozen": not any(p.requires_grad for p in MODEL.parameters()),
        "warnings": ["MOCK model — proves interface and pipeline only"],
    }

print(f"model class     {ARCHITECTURE['model_class']}")
print(f"layers/width    {ARCHITECTURE['n_layers']} / {ARCHITECTURE['d_model']}")
print(f"vocab           {ARCHITECTURE['vocab_size']}")
print(f"params frozen   {ARCHITECTURE['params_frozen']}")
print(f"revision        {LOAD_INFO['model_revision']}")
for _warning in ARCHITECTURE.get("warnings", []):
    print(f"  note: {_warning}")

# The MOCK grid runs on the tiny stack; the real grid is the frozen one.
if MODEL_STAGES_ENABLED:
    ACTIVE_PLAN = PLAN
else:
    # The mock stack has the real depth (42 blocks) at a tiny width, so MOCK
    # exercises the actual layer grid rather than a stand-in for it.
    ACTIVE_PLAN = build_capture_plan(
        layers=LAYERS,
        target_layer=TARGET_LAYER,
        d_model=MODEL.d_model,
        dim_batch=8,
        max_seq_len=48,
        skip_first=4,
        n_layers=MODEL.n_layers,
    )
print()
print(f"capture plan    layers {list(ACTIVE_PLAN.layers)} -> L{ACTIVE_PLAN.target_layer}")
print(f"  per prompt    1 forward + {ACTIVE_PLAN.backward_passes_per_prompt} backward passes")
print(f"  backward span {ACTIVE_PLAN.backward_span} blocks (the cost driver)")
print(f"  all layers captured in one pass: {ACTIVE_PLAN.to_dict()['all_layers_captured_in_one_pass']}")
'''
)

code(
    '''
# 8b. Real runs discover their held-out targets now that the model exists.
if MODEL_STAGES_ENABLED:
    VALIDATION_PROMPTS, VALIDATION_MANIFEST = select_diverse_validation_prompts(
        [record.text for record in PARTITIONS.validation],
        n_prompts=N_VALIDATION_PROMPTS,
        seed=VALIDATION_PROMPT_SEED,
        target_token_for_prompt=_target_token,
    )
    CONFIRMATION_PROMPTS, CONFIRMATION_MANIFEST = select_diverse_validation_prompts(
        [record.text for record in PARTITIONS.confirmation],
        n_prompts=N_CONFIRMATION_PROMPTS,
        seed=CONFIRMATION_PROMPT_SEED,
        target_token_for_prompt=_target_token,
    )
    DIVERSITY = VALIDATION_MANIFEST["diversity"]
    print(f"validation distinct targets {DIVERSITY['n_distinct_target_tokens']}, "
          f"max share {DIVERSITY['max_target_token_share']:.1%}, "
          f"passed {DIVERSITY['passed']}")
else:
    print("MOCK: held-out sets were already chosen in section 7.")
'''
)

# ==================================================================== 9. budget

markdown(
    """
## 9. Compute and storage budget

**Read this before setting any switch.** Every number is extrapolated from one
real measurement on this project's own hardware — 100 prompts × 128 tokens ×
7 layers in 9,665.4 s on an NVIDIA L4 (`docs/pilot_report.md`) — so the
uncertainty band is wide and stated.

The scale rows are **cumulative, not additive**: reaching 1,000 also produces
the 100- and 250-prompt lenses.
"""
)

code(
    '''
# 9. The budget. Nothing here runs the model.
from jlens.calibration.plan import estimate_budget, format_budget

BUDGET = estimate_budget(
    PLAN,
    scale_points=SCALE_POINTS,
    n_validation=N_VALIDATION_PROMPTS,
    n_confirmation=N_CONFIRMATION_PROMPTS,
)
print(format_budget(BUDGET, PLAN))
print()
print("HONEST SUMMARY")
print("  100 prompts is the upstream usable-scale benchmark (~2-3 hours).")
print("  250 prompts tests whether earlier layers are still improving (~6 hours).")
print("  1,000 prompts matches the paper's construction scale (~23 hours).")
print("  This two-week protocol authorizes no scale beyond 1,000.")
print("  A fit parallelizes exactly across runtimes via JacobianLens.merge(),")
print("  so N concurrent L4s divide the wall time by N.")
'''
)

# ================================================= 10. real-run confirmation

markdown(
    """
## 10. Explicit real-run confirmation

The gate between reading the budget and spending it. Both the run switch and
the matching budget switch must be set by hand, and this cell prints exactly
what will happen either way.
"""
)

code(
    '''
# 10. What this execution will actually do.
print(f"RUN_REAL_CALIBRATION          {RUN_REAL_CALIBRATION}")
print(f"RUN_MODEL_STAGES              {RUN_MODEL_STAGES}")
print(f"CONFIRM_100_BUDGET            {CONFIRM_100_BUDGET}")
print(f"CONFIRM_250_BUDGET            {CONFIRM_250_BUDGET}")
print(f"CONFIRM_1K_BUDGET             {CONFIRM_1K_BUDGET}")
print(f"RUN_FINAL_CONFIRMATION        {RUN_FINAL_CONFIRMATION}")
print(f"PUBLISH_VALIDATED_LENSES      {PUBLISH_VALIDATED_LENSES}")
print()
if MODEL_STAGES_ENABLED:
    _hours = BUDGET.row(max(ACTIVE_SCALE_POINTS))
    print("REAL CALIBRATION ENABLED")
    print(f"  scale points   {list(ACTIVE_SCALE_POINTS)}")
    print(f"  estimated      {_hours['fitting_hours_central']:.1f} h central "
          f"({_hours['fitting_hours_low']:.1f}-{_hours['fitting_hours_high']:.1f} h)")
    print(f"  sessions       {_hours['colab_sessions_central']}-{_hours['colab_sessions_high']}")
    print(f"  Drive          ~{BUDGET.drive_bytes_total / 2**30:.2f} GiB")
    print("  Resume is automatic; a disconnect loses at most one checkpoint interval.")
else:
    print("MOCK RUN — nothing above is spent.")
    print("  No model was downloaded, no corpus streamed, no Drive written.")
    print("  Every result below is a fixture. It proves the pipeline runs,")
    print("  and it proves nothing whatsoever about Gemma 4.")
'''
)

# ====================================== 11. collect / resume calibration data

markdown(
    """
## 11. Collect or resume multilayer calibration data

The run directory, the fingerprint, and the prompt list.

Prompts are filtered by **token** length first, using the tokenizer alone. That
matters: upstream `fit` skips prompts too short to contribute a valid source
position, which would make `n_done` drift below the prompt index and turn a
"1,000-prompt lens" into a lens fitted on some unknown number below 1,000.
Filtering first makes the scale points exact.
"""
)

code(
    '''
# 11. Open the run directory under a fingerprint, and prepare the fit list.
from jlens.calibration.fitting import filter_records_by_tokens
from jlens.calibration.state import CalibrationFingerprint, CalibrationStore

if MODEL_STAGES_ENABLED:
    def _token_count(text):
        return int(MODEL.encode(text, max_length=ACTIVE_PLAN.max_seq_len).shape[1])
else:
    def _token_count(text):
        return MODEL.tokenizer.token_count(text)

FIT_RECORDS, DROPPED_SHORT = filter_records_by_tokens(
    PARTITIONS.fit,
    token_count=_token_count,
    skip_first=ACTIVE_PLAN.skip_first,
    max_seq_len=ACTIVE_PLAN.max_seq_len,
)

FINGERPRINT = CalibrationFingerprint(
    mode=MODE,
    protocol_version=PROTOCOL_VERSION,
    model_repo_id=LOAD_INFO["model_repo_id"],
    model_revision=LOAD_INFO["model_revision"],
    tokenizer_revision=LOAD_INFO["tokenizer_revision"],
    capture_plan_digest=ACTIVE_PLAN.digest,
    corpus_manifest_checksum=CORPUS_MANIFEST["corpus_manifest_checksum"],
    gate_digest=CALIBRATION_GATE.digest,
    plateau_rule_digest=PLATEAU_RULE.digest,
    scale_points=tuple(SCALES),
    artifact_format_version=CONFIG["artifact_format_version"],
)
RUN_DIR = RUN_ROOT / f"rgcalib_{MODE}_{FINGERPRINT.digest[7:19]}"
STORE = CalibrationStore(RUN_DIR, FINGERPRINT)
RESUME_STATUS = STORE.open()

STORE.save(
    "corpus",
    "manifest",
    {
        "corpus": CORPUS_MANIFEST,
        "leakage_audit": LEAKAGE,
        "scale_nesting_audit": NESTING,
        "n_fit_records_after_token_filter": len(FIT_RECORDS),
        "n_dropped_too_short": len(DROPPED_SHORT),
        "validation_selection": VALIDATION_MANIFEST,
        "confirmation_selection_checksum": CONFIRMATION_MANIFEST["selection_checksum"],
    },
)

print(f"run dir       {RUN_DIR}")
print(f"fingerprint   {FINGERPRINT.digest}")
print(f"resume        {RESUME_STATUS}")
print(f"fit records   {len(FIT_RECORDS):,} usable "
      f"({len(DROPPED_SHORT):,} too short in tokens, dropped before fitting)")
print(f"largest scale {max(SCALES):,}")
if len(FIT_RECORDS) < max(SCALES):
    raise RuntimeError(
        f"only {len(FIT_RECORDS)} usable fit records for a scale point of "
        f"{max(SCALES)}; stream more corpus records"
    )
'''
)

# ================================================ 12. fit / resume per layer

markdown(
    """
## 12. Fit or resume per-layer lenses

One call to the **unmodified upstream** `jlens.fitting.fit`. All layers come
from a single forward and a single set of backward passes per prompt; each layer
gets its own `[d_model, d_model]` matrix and none are shared.

The nested scale snapshots are written from the shared accumulator as `n_done`
reaches each point, so the three lenses cost as much as the largest alone. A
test asserts each snapshot is bit-identical to a standalone fit on the same
prefix.

Resume is upstream's: the accumulator checkpoint is atomic and reloaded
automatically, and a checkpoint fitted with different layers, target or
`skip_first` is **refused**.
"""
)

code(
    '''
# 12. The fit. This is the only expensive cell in the notebook.
from jlens.calibration.fitting import run_calibration

CALIBRATION = run_calibration(
    MODEL,
    FIT_RECORDS,
    plan=ACTIVE_PLAN,
    scale_points=SCALES,
    store=STORE,
    checkpoint_every=25 if MODEL_STAGES_ENABLED else 4,
    diagnostics_every=25 if MODEL_STAGES_ENABLED else 4,
)

print(f"prompts fitted   {CALIBRATION.n_done:,}")
print(f"skipped          {CALIBRATION.n_skipped}")
print(f"elapsed          {CALIBRATION.elapsed_seconds / 3600:.2f} h")
print(f"checkpoint       {CALIBRATION.checkpoint_path}")
print()
print("scale snapshots")
for _scale, _snapshot in sorted(CALIBRATION.snapshots.items()):
    print(f"  {_scale:>7,}  n_prompts={_snapshot.n_prompts:<7,} {_snapshot.checksum}")

if CALIBRATION.diagnostics:
    print()
    print("per-layer convergence (last prompt; mean_relative_change should fall ~1/n)")
    for _layer, _entry in sorted(
        CALIBRATION.diagnostics[-1]["layers"].items(), key=lambda kv: int(kv[0])
    ):
        print(f"  L{_layer:<3} ||J||={_entry['prompt_jacobian_norm']:>10.4f}  "
              f"mean||J||={_entry['running_mean_norm']:>10.4f}  "
              f"d_mean={_entry['mean_relative_change']}  finite={_entry['finite']}")
'''
)

# ===================================================== 13. validate per scale

markdown(
    """
## 13. Validate each layer at the current scale

Native readout on held-out prompts the lens never saw, against three controls
and one diagnostic, scored with the project's standard tie-aware scorer.

Controls: row-permuted `J_l` (primary), norm-matched random, and wrong-layer via
`distant_layer_mapping` — **not** the deprecated cyclic control, which conflates
adjacent and distant substitutions. The ordinary logit lens is a diagnostic and
blocks nothing.
"""
)

code(
    '''
# 13a. The readout. One forward per prompt; every variant reads the same
# activations, so controls are matched by construction.
from jlens.controls import control_lens, distant_layer_mapping, layer_mapped_lens
from jlens.hooks import ActivationRecorder
from jlens.mmlocalize.lens_validity import tie_aware_row


def score_readout_rows(lens, prompts, layers, target_layer):
    """Tie-aware rows for every (prompt, layer, variant)."""
    import hashlib

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
            # The model's own prediction at the last position. The softcap is
            # strictly monotonic, so every rank-based metric here is identical
            # pre- and post-cap.
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
'''
)

code(
    '''
# 13b. Validate every scale point on the development set.
#
# MOCK note: a tiny random stack has no interesting per-layer structure, so the
# MOCK path scores FIXTURE rows with four known archetypes instead — a
# degenerate layer, an early layer that improves with scale, a layer that looks
# perfect on optimistic rank and fails tie-aware validation, and a late layer
# that always passes. The rows are synthetic; the SCORER and the GATE are the
# real ones.
from jlens.calibration.gate import eligible_layers, evaluate_calibration_layers

VALIDATION = {}
for _index, _scale in enumerate(SCALES):
    _key = f"scale{_scale}"
    _stored = STORE.load("validation", _key)
    if _stored is not None:
        VALIDATION[_scale] = {int(k): v for k, v in _stored["by_layer"].items()}
        print(f"scale {_scale:>7,}  reused stored validation")
        continue
    if MODEL_STAGES_ENABLED:
        _lens = CALIBRATION.lens_for_scale(_scale)
        _rows = score_readout_rows(
            _lens, VALIDATION_PROMPTS, list(ACTIVE_PLAN.layers), ACTIVE_PLAN.target_layer
        )
        _layers = list(ACTIVE_PLAN.layers)
    else:
        from jlens.calibration.mock import MOCK_LAYERS, mock_validation_rows  # noqa: PLC0415

        _rows = mock_validation_rows(
            scale=_scale, scale_index=_index, n_prompts=N_VALIDATION_PROMPTS
        )
        _layers = list(MOCK_LAYERS)
    VALIDATION[_scale] = evaluate_calibration_layers(
        _rows, layers=_layers, scale=_scale, stage="validation"
    )
    STORE.save(
        "validation",
        _key,
        {"scale": _scale, "by_layer": {str(k): v for k, v in VALIDATION[_scale].items()}},
    )
    print(f"scale {_scale:>7,}  eligible {eligible_layers(VALIDATION[_scale])}")

GRID_LAYERS = sorted(VALIDATION[SCALES[-1]])
print()
print(f"{'layer':>6} {'scale':>8} {'pass':>6} {'MRR':>8} {'midrank':>9} "
      f"{'opt':>7} {'tie@max':>8} {'top10':>7}")
for _layer in GRID_LAYERS:
    for _scale in SCALES:
        _v = VALIDATION[_scale][_layer]
        _m = _v["metrics"]["j_lens"]
        print(f"{_layer:>6} {_scale:>8,} {('PASS' if _v['passed'] else 'fail'):>6} "
              f"{_m['mean_reciprocal_rank']:>8.4f} {_m['median_midrank']:>9.2f} "
              f"{_m['median_optimistic_rank']:>7.1f} {_m['tied_at_max_rate']:>8.3f} "
              f"{_m['top10_inclusion']:>7.3f}")
'''
)

# ==================================================== 14. compare the scales

markdown(
    """
## 14. Compare the scale points

Does more calibration data buy anything at layers earlier than 38? The `opt`
column beside `midrank` is the whole reason the gate is tie-aware: a layer can
read rank 1 optimistically and sit halfway down a tie block.
"""
)

code(
    '''
# 14. The scale comparison table and its deltas.
from jlens.calibration.scale import compare_scales

COMPARISON = compare_scales(VALIDATION, layers=GRID_LAYERS)
STORE.save("scale_comparison", "comparison", COMPARISON)

print("eligible layers by scale")
for _scale, _layers in COMPARISON["eligible_by_scale"].items():
    print(f"  {int(_scale):>7,}  {_layers}")
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
'''
)

# ======================================================== 15. plateau rule

markdown(
    """
## 15. Apply the predeclared plateau rule

The rule was fixed before any number existed and its digest is bound into the
run fingerprint. Here it reports whether improvement is still visible at the
paper-matched 1,000-prompt endpoint. It never authorizes a larger run.
"""
)

code(
    '''
# 15. The plateau verdict and the scale selection.
from jlens.calibration.scale import evaluate_plateau, select_scale

PLATEAU = evaluate_plateau(COMPARISON)
print(f"verdict                {PLATEAU['verdict']}")
print(f"extension justified    {PLATEAU['extension_justified']}")
print(f"runs automatically     {PLATEAU['runs_automatically']}")
print()
for _clause in PLATEAU["clauses"]:
    print(f"  [{'pass' if _clause['passed'] else 'FAIL'}] {_clause['clause']}")
    print(f"         {_clause['detail']}")

SELECTION = select_scale(COMPARISON)
print()
print(f"selected scale         {SELECTION['selected_scale']:,}")
print(f"reason                 {SELECTION['reason']}")
print(f"eligible at selection  {SELECTION['eligible_at_selected_scale']}")

if PLATEAU["extension_justified"]:
    print()
    print("The estimator is still improving at the final 1,000-prompt endpoint.")
    print("That is recorded as a limitation; this two-week protocol authorizes")
    print("no larger scale and will not start one automatically.")
'''
)

# ================================================== 16. final confirmation

markdown(
    """
## 16. Run the untouched final confirmation

The confirmation set is held in a vault that refuses to release it until a scale
selection has been recorded against it. That makes "the confirmation set
influenced nothing" a property of the code, not a claim in a document.

A layer must pass the **same gate, unchanged**, on these 128 prompts to be
published.
"""
)

code(
    '''
# 16. Open the vault (only against a recorded selection) and confirm.
from jlens.calibration.publication import ConfirmationVault

VAULT = ConfirmationVault(records=PARTITIONS.confirmation)
CONFIRMATION = None

if not RUN_FINAL_CONFIRMATION:
    print("RUN_FINAL_CONFIRMATION is False — the confirmation set stays locked.")
    print(f"vault status: {VAULT.status()}")
    print()
    print("Note: 'not run' is NOT the same as 'nothing passed'. No layer has")
    print("been offered the confirmation set.")
else:
    VAULT.unlock(SELECTION)
    _records = VAULT.open()
    _scale = SELECTION["selected_scale"]
    _stored = STORE.load("confirmation", f"scale{_scale}")
    if _stored is not None:
        CONFIRMATION = {int(k): v for k, v in _stored["by_layer"].items()}
        print("reused stored confirmation")
    else:
        if MODEL_STAGES_ENABLED:
            _lens = CALIBRATION.lens_for_scale(_scale)
            _rows = score_readout_rows(
                _lens,
                CONFIRMATION_PROMPTS,
                list(ACTIVE_PLAN.layers),
                ACTIVE_PLAN.target_layer,
            )
            _layers = list(ACTIVE_PLAN.layers)
        else:
            from jlens.calibration.mock import MOCK_LAYERS, mock_validation_rows  # noqa: PLC0415

            _rows = mock_validation_rows(
                scale=_scale,
                scale_index=len(SCALES) - 1,
                n_prompts=N_CONFIRMATION_PROMPTS,
            )
            _layers = list(MOCK_LAYERS)
        CONFIRMATION = evaluate_calibration_layers(
            _rows,
            layers=_layers,
            scale=_scale,
            stage="confirmation",
            gate=CALIBRATION_GATE.for_confirmation(),
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
    for _layer in sorted(CONFIRMATION):
        _v = CONFIRMATION[_layer]
        print(f"  L{_layer:<3} {('PASS' if _v['passed'] else 'fail')}  "
              f"{_v['failed_checks'] or ''}")
'''
)

# ======================================================== 17. publication

markdown(
    """
## 17. Publish passing frozen lens artifacts

One artifact per passing layer, each carrying the full provenance record.
Publication is refused for a layer with no confirmation verdict, a layer whose
confirmation verdict failed, a development verdict offered in place of a
confirmation one, and any attempt to write over a frozen completed-run artifact.

**Fitting completing is not a reason to publish anything.**
"""
)

code(
    '''
# 17. Publish, or explain precisely why nothing was published.
from jlens.calibration.publication import (
    PublicationRefused,
    publication_summary,
    publish_layer,
    record_failed_layer,
)

PUBLISHED, FAILED, PUBLICATION = [], [], None

if not PUBLISH_VALIDATED_LENSES:
    print("PUBLISH_VALIDATED_LENSES is False — nothing is written.")
elif CONFIRMATION is None:
    print("No confirmation results exist. Publication requires the untouched")
    print("confirmation set; a lens is never published because fitting finished.")
else:
    _scale = SELECTION["selected_scale"]
    _lens = CALIBRATION.lens_for_scale(_scale)
    _fitted = set(_lens.source_layers)
    for _layer in sorted(CONFIRMATION):
        _verdict = CONFIRMATION[_layer]
        _validation = VALIDATION[_scale][_layer]
        if not _verdict["passed"] or _layer not in _fitted:
            FAILED.append(
                record_failed_layer(
                    layer=_layer,
                    scale=_scale,
                    confirmation_verdict=_verdict,
                    validation_verdict=_validation,
                )
            )
            continue
        try:
            PUBLISHED.append(
                publish_layer(
                    layer=_layer,
                    scale=_scale,
                    lens=_lens,
                    destination=STORE.published_path(_layer, _scale),
                    confirmation_verdict=_verdict,
                    validation_verdict=_validation,
                    vault=VAULT,
                    load_info=LOAD_INFO,
                    corpus_manifest=CORPUS_MANIFEST,
                    capture_plan=ACTIVE_PLAN.to_dict(),
                    fitting_diagnostics=CALIBRATION.to_dict(),
                    environment=ENVIRONMENT,
                    protocol_version=PROTOCOL_VERSION,
                )
            )
        except PublicationRefused as error:
            print(f"  refused L{_layer}: {error}")
            FAILED.append(
                record_failed_layer(
                    layer=_layer,
                    scale=_scale,
                    confirmation_verdict=_verdict,
                    validation_verdict=_validation,
                )
            )
    PUBLICATION = publication_summary(PUBLISHED, FAILED)
    STORE.save("publication", f"scale{_scale}", PUBLICATION)
    print(f"published {PUBLICATION['n_published']} layer(s): "
          f"{PUBLICATION['published_layers']}")
    print(f"failed    {PUBLICATION['n_failed']} layer(s): "
          f"{PUBLICATION['failed_layers']} (diagnostics kept, validated=false)")
'''
)

# ============================================================== 18. report

markdown(
    """
## 18. Generate report and resume status

The report separates what was measured, what a predeclared rule decided, and
what is still unknown — and it says outright what the run does not establish.
"""
)

code(
    '''
# 18. Write the report and print the resume state.
from jlens.calibration.report import (
    calibration_report_markdown,
    calibration_report_payload,
)

RESUME = STORE.status_report()
REPORT_PAYLOAD = calibration_report_payload(
    fingerprint_digest=FINGERPRINT.digest,
    protocol_version=PROTOCOL_VERSION,
    capture_plan=ACTIVE_PLAN.to_dict(),
    budget=BUDGET.to_dict(),
    corpus_manifest=CORPUS_MANIFEST,
    leakage_audit=LEAKAGE,
    nesting_audit=NESTING,
    diversity=DIVERSITY,
    comparison=COMPARISON,
    plateau=PLATEAU,
    selection=SELECTION,
    confirmation=(
        {str(k): v for k, v in CONFIRMATION.items()} if CONFIRMATION else None
    ),
    publication=PUBLICATION,
    vault_status=VAULT.status(),
    resume_status=RESUME,
    baseline=BASELINE,
    mode=MODE,
)
REPORT_MARKDOWN = calibration_report_markdown(
    REPORT_PAYLOAD, gate_text=gate_text(), plateau_text=PLATEAU_RULE.text()
)

REPORT_DIR = RUN_DIR / "artifacts"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
(REPORT_DIR / "calibration_report.md").write_text(REPORT_MARKDOWN, encoding="utf-8")
(REPORT_DIR / "calibration_report.json").write_text(
    json.dumps(REPORT_PAYLOAD, indent=2, ensure_ascii=False, default=str),
    encoding="utf-8",
)

print(f"report        {REPORT_DIR / 'calibration_report.md'}")
print(f"checksum      {REPORT_PAYLOAD['report_checksum']}")
print(f"resume        {RESUME['status']}")
print(f"checkpoint    {RESUME['checkpoint_present']}")
print(f"units         {RESUME['completed_units']}")
print(f"invalid units {len(RESUME['invalid_units'])}")
print()
if MODE != "real":
    print("MOCK RUN COMPLETE. Pipeline behaviour only — no evidence about Gemma 4,")
    print("about any layer, or about whether a research-grade lens exists.")
else:
    print("Send back artifacts/calibration_report.md plus the scale table.")
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
