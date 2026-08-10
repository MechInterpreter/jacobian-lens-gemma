# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Generate ``notebooks/archive/completed_studies/gemma4_native_spoken_audio_feasibility_colab.ipynb``.

Written from source rather than edited as JSON, so the committed notebook stays
output-free and byte-reproducible. Run
``python scripts/_build_audio_feasibility_notebook.py`` after changing a cell;
``tests/test_mmpilot_audio_notebook.py`` regenerates it and fails on drift.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TARGET = (
    REPO_ROOT / "notebooks" / "archive" / "completed_studies" / "gemma4_native_spoken_audio_feasibility_colab.ipynb"
)

CELLS: list[tuple[str, str]] = []


def markdown(text: str) -> None:
    CELLS.append(("markdown", text.strip("\n")))


def code(text: str) -> None:
    CELLS.append(("code", text.strip("\n")))


markdown(
    """
# Native spoken audio on Gemma 4 E4B IT — feasibility, not science

**One question.** Can a real SpokenCOCO waveform be presented to the pinned
checkpoint through **its own native audio pathway** — reaching the audio tower,
producing the required placeholder span, changing the model's output, and
leaving the capture and intervention harness undisturbed?

**This notebook does not run the multimodal J-space experiment.** It fits no
lens, estimates no direction, and measures no transfer. `AUDIO READY` here means
*the plumbing works*. It is **not** evidence that spoken-audio J-space transfer
works, and it never becomes that evidence by being repeated.

## What was actually wrong

The pilot recorded `spoken_audio` as blocked because the *"processor/model
produced audio features but zero audio placeholder tokens"*. That is real and
reproducible — and it is not a statement that the checkpoint lacks audio:

```python
processor(text="Answer with exactly one word.", audio=waveform)
# input_ids: (1, 11)  input_features: (1, 99, 128)  audio placeholders: 0
```

`Gemma4Processor` expands `<|audio|>` tokens that are **already in the text**.
A bare text prompt has none, so the features are computed and then scattered
into nothing. No error is raised — `validate_inputs` checks image-token counts
and has no audio equivalent, and `_check_special_mm_tokens` compares zero to
zero. The mismatch only surfaces inside `Gemma4Model.forward` as
`Audio features and audio tokens do not match, tokens: 0, features: N`.

The supported native path is the **chat-template audio content block**, which
renders the placeholder so the processor can expand it against the features it
just computed. That is the whole repair, and
`jlens/mmpilot/audio.py` implements it centrally.

## Three switches, all False

| switch | what it does |
| --- | --- |
| `RUN_REAL_AUDIO_AUDIT` | use real SpokenCOCO waveforms from Drive instead of the deterministic MOCK world |
| `RUN_MODEL_STAGE` | allow the real Gemma checkpoint to be loaded at all |
| `CONFIRM_MODEL_LOAD` | acknowledge the ~16 GB download explicitly |

At the committed defaults, opening and running this notebook top to bottom loads
no model, mounts no Drive, reads no media, and exercises the **same audit code**
against a mock world. That MOCK result proves the notebook runs and proves
nothing about Gemma.

## The three verdicts

- **`AUDIO READY`** — every required check passed. Native spoken audio is
  technically usable for a future study. Engineering evidence only.
- **`AUDIO BLOCKED`** — the pinned checkpoint, processor or Transformers version
  cannot support the required path. Reported with the exact broken link.
- **`AUDIO INVALID`** — an input *was* built and then failed a check that makes
  later numbers untrustworthy: a hook that moved the logits, a tower that never
  fired, a recording indistinguishable from silence, a transcript in the prompt.
  This is the dangerous state, because it looks like success.

## Fingerprint consequences

Any change to the model id, model revision, processor revision, Transformers
version, audio protocol, placeholder convention or hook-position convention
**must** change the run fingerprint. The resolved protocol carries its own
digest for exactly this; section 13 prints it. **If enabling audio ever required
changing the checkpoint or revision, the current text-calibrated lens could not
be reused** and a new lens would have to be fitted first.
"""
)


# ------------------------------------------------------------------ 1. boot

markdown(
    """
## 1. Bootstrap repository

Three cells, standard library only: the repository is not importable until 1c
has installed it. Google Drive is not needed here — nothing is mounted until
section 3, and then only for a real run.
"""
)

code(
    """
# 1a. Bootstrap constants only. Nothing from this repository is imported yet.
REPO_URL = "https://github.com/MechInterpreter/jacobian-lens-gemma.git"
BRANCH = "experiment/spokencoco-jspace-pilot"
REPO_DIR = "/content/jacobian-lens-gemma"

print(f"repo   {REPO_URL}")
print(f"branch {BRANCH}")
print(f"target {REPO_DIR}")
"""
)

code(
    """
# 1b. Clone or update the repository, then verify the checked-out branch.
#
# Idempotent: clones when absent, otherwise fetches, checks out and resets to
# origin. The reset discards local edits inside the Colab checkout — that
# directory is scratch, not somewhere to keep work.
import os
import subprocess
import sys
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
REPO_PATH = Path(os.environ.get("AUDIOAUDIT_REPO_DIR") or REPO_DIR)


def _git(*arguments, cwd=None):
    result = subprocess.run(["git", *arguments], cwd=cwd, capture_output=True, text=True)
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
"""
)

code(
    """
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
"""
)


# ---------------------------------------------------------------- 2. config

markdown(
    """
## 2. Configuration

All three switches are False in the committed notebook and must be set by hand.
Nothing below mounts Drive, downloads a model, or reads media while they are.

The model id and revision are **pinned and not adjustable from results**.
Changing either invalidates the text-calibrated J-lens, which was validated
against this revision alone — see section 13.
"""
)

code(
    """
# 2. Configuration. Requires section 1 (it imports from the repository).
RUN_REAL_AUDIO_AUDIT = False
RUN_MODEL_STAGE = False

# Set to True only after reading the download note printed below.
CONFIRM_MODEL_LOAD = False

import json

from jlens.mmpilot.audio import (
    AUDIO_PROTOCOL_VERSION,
    CALL_CONVENTION,
    CONTENT_BLOCK_SCHEMA,
    PROBE_DURATIONS_S,
    REASONS,
)
from jlens.mmpilot.audio_audit import AUDIT_VERSION, REQUIRED_CHECKS
from jlens.mmpilot.capability import build_prompt, build_question

# --- the pin. Not adjustable from results.
MODEL_REPO_ID = "google/gemma-4-E4B-it"
MODEL_REVISION = "fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd"
EXPECT_N_LAYERS, EXPECT_D_MODEL, EXPECT_VOCAB = 42, 2560, 262144

# --- what the audit measures at. Two mid/late layers is enough to show the
# harness is undisturbed; this notebook estimates nothing, so the choice
# carries no scientific weight.
AUDIT_LAYERS = (21, 38)

# --- Drive locations, used only when RUN_REAL_AUDIO_AUDIT is True.
AUDIO_MEDIA_ROOT = "/content/drive/MyDrive/datasets/cstf_spokencoco/SpokenCOCO"
RUNS_ROOT = "/content/drive/MyDrive/jacobian-lens-gemma/runs"

# --- the probe question, used by section 7 before any model exists. It carries
# no caption, as every spoken-audio prompt must: the recording is the only
# evidence. Section 9 rebuilds it from the candidates actually selected, so the
# options asked about and the sequences scored are the same set.
PROBE_CONCEPTS = ("cat", "dog")
QUESTION = build_question(sorted(PROBE_CONCEPTS))
AUDIT_PROMPT = build_prompt(QUESTION, modality="spoken_audio")

MODE = "real" if RUN_REAL_AUDIO_AUDIT else "mock"
print(f"mode                 {MODE}")
print(f"RUN_REAL_AUDIO_AUDIT {RUN_REAL_AUDIO_AUDIT}")
print(f"RUN_MODEL_STAGE      {RUN_MODEL_STAGE}")
print(f"CONFIRM_MODEL_LOAD   {CONFIRM_MODEL_LOAD}")
print(f"audit version        {AUDIT_VERSION}")
print(f"audio protocol       {AUDIO_PROTOCOL_VERSION}")
print(f"call convention      {CALL_CONVENTION}")
print(f"content block        {CONTENT_BLOCK_SCHEMA}")
print(f"probe durations (s)  {list(PROBE_DURATIONS_S)}")
print(f"prompt               {AUDIT_PROMPT!r}")
print()
print("required checks:")
for _name in REQUIRED_CHECKS:
    print(f"  - {_name}")
print()
print("refusal codes this audit can report:")
print("  " + ", ".join(REASONS))

if RUN_MODEL_STAGE and not CONFIRM_MODEL_LOAD:
    raise RuntimeError(
        "RUN_MODEL_STAGE=True needs CONFIRM_MODEL_LOAD=True as well. Loading "
        f"{MODEL_REPO_ID} downloads roughly 16 GB and needs a GPU runtime; say "
        "so explicitly rather than having it happen as a side effect."
    )
if RUN_REAL_AUDIO_AUDIT and not RUN_MODEL_STAGE:
    print(
        "\\nnote: RUN_REAL_AUDIO_AUDIT=True with RUN_MODEL_STAGE=False inspects "
        "the real processor and resolves the real protocol on CPU, and stops "
        "before anything that needs weights. That is a useful first run."
    )
"""
)


# ----------------------------------------------------------------- 3. drive

markdown(
    """
## 3. Mount Google Drive

Only for a real run, and only to **read** waveforms and to write this audit's
own new directory. No completed pilot, robustness or localization run is opened,
modified, or written into — section 13 refuses a runs root that looks like one.
"""
)

code(
    """
# 3. Mount Drive. Skipped entirely in MOCK.
import tempfile
from pathlib import Path

DRIVE_MOUNTED = False
if RUN_REAL_AUDIO_AUDIT:
    if IN_COLAB:
        from google.colab import drive

        drive.mount("/content/drive")
        DRIVE_MOUNTED = True
    RESOLVED_RUNS_ROOT = Path(RUNS_ROOT)
    RESOLVED_AUDIO_ROOT = Path(AUDIO_MEDIA_ROOT)
    print(f"audio root {RESOLVED_AUDIO_ROOT}  exists={RESOLVED_AUDIO_ROOT.is_dir()}")
    print(f"runs root  {RESOLVED_RUNS_ROOT}")
    if not RESOLVED_AUDIO_ROOT.is_dir():
        raise RuntimeError(
            f"{RESOLVED_AUDIO_ROOT} not found. Fix AUDIO_MEDIA_ROOT in section 2; "
            "this notebook never downloads media."
        )
else:
    SCRATCH = Path(os.environ.get("AUDIOAUDIT_SCRATCH") or tempfile.mkdtemp(prefix="audioaudit_"))
    RESOLVED_RUNS_ROOT = SCRATCH / "runs"
    RESOLVED_AUDIO_ROOT = SCRATCH / "audio"
    RESOLVED_RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    RESOLVED_AUDIO_ROOT.mkdir(parents=True, exist_ok=True)
    print("MOCK: no Drive mounted")
    print(f"scratch runs root {RESOLVED_RUNS_ROOT}")
"""
)


# ---------------------------------------------------------- 4. dependencies

markdown(
    """
## 4. Install and verify dependencies

The Transformers version is part of the audio protocol's fingerprint, so it is
printed and recorded rather than assumed. `soundfile` is needed only to decode
real waveforms.
"""
)

code(
    """
# 4. Dependencies. Versions are recorded because they bind the protocol.
import importlib

ENVIRONMENT = {"python": sys.version.split()[0]}

if IN_COLAB and RUN_REAL_AUDIO_AUDIT:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "soundfile"],
        capture_output=True,
        text=True,
    )

for _package in ("torch", "transformers", "numpy", "soundfile"):
    try:
        _module = importlib.import_module(_package)
        ENVIRONMENT[_package] = getattr(_module, "__version__", "unknown")
    except ImportError:
        ENVIRONMENT[_package] = "not installed"

import torch

ENVIRONMENT["cuda_available"] = bool(torch.cuda.is_available())
ENVIRONMENT["device"] = "cuda" if torch.cuda.is_available() else "cpu"
ENVIRONMENT["model_repo_id"] = MODEL_REPO_ID
ENVIRONMENT["model_revision"] = MODEL_REVISION
ENVIRONMENT["mode"] = MODE
print(json.dumps(ENVIRONMENT, indent=2))

if RUN_REAL_AUDIO_AUDIT and ENVIRONMENT["soundfile"] == "not installed":
    raise RuntimeError(
        "soundfile is required to decode SpokenCOCO waveforms. This notebook "
        "does not substitute transcripts for audio under any circumstances."
    )
if RUN_MODEL_STAGE and not ENVIRONMENT["cuda_available"]:
    print(
        "\\nwarning: no GPU. The model stage will run on CPU, which is slow but "
        "correct — every check here is about inputs and invariance, not speed."
    )
"""
)


# --------------------------------------------------------- 5. authenticate

markdown(
    """
## 5. Authenticate

Needed only to download the checkpoint. The pinned repository is public at this
revision, so a token is optional; supply one if your runtime is rate limited.
The token is read interactively and never written to disk or into any artifact.
"""
)

code(
    """
# 5. Hugging Face authentication. Skipped unless the model will be loaded.
HF_TOKEN = None
if RUN_MODEL_STAGE:
    HF_TOKEN = os.environ.get("HF_TOKEN") or None
    if HF_TOKEN is None and IN_COLAB:
        try:
            from google.colab import userdata

            HF_TOKEN = userdata.get("HF_TOKEN")
        except Exception:
            HF_TOKEN = None
    print("token supplied:", HF_TOKEN is not None)
    print("(the pinned revision is public; a token only lifts rate limits)")
else:
    print("skipped: RUN_MODEL_STAGE is False, nothing is downloaded")
"""
)


# ---------------------------------------------- 6. inspect model + processor

markdown(
    """
## 6. Inspect the pinned model and processor

Reported by inspection, not from documentation: the processor class, its
components, its accepted call parameters, the audio token id, and whether an
audio tower is configured.

**Component presence is not support.** This checkpoint reports
`supports_audio=True` here *and still* produced zero placeholder tokens under
the old calling convention. Section 7 is what settles it.

This cell needs no weights: the processor and config are small files.
"""
)

code(
    """
# 6. Processor and config, by inspection. No weights are downloaded here.
from jlens.mmpilot.backend import resolve_processor_interface

PROCESSOR = HF_CONFIG = None
INTERFACE = {}

if RUN_REAL_AUDIO_AUDIT:
    from transformers import AutoConfig, AutoProcessor

    PROCESSOR = AutoProcessor.from_pretrained(
        MODEL_REPO_ID, revision=MODEL_REVISION, token=HF_TOKEN
    )
    HF_CONFIG = AutoConfig.from_pretrained(
        MODEL_REPO_ID, revision=MODEL_REVISION, token=HF_TOKEN
    )
else:
    from jlens.mmpilot.mock import MockAudioProcessor, MockAudioTokenizer, mock_audio_config

    PROCESSOR = MockAudioProcessor()
    PROCESSOR.tokenizer = MockAudioTokenizer()
    HF_CONFIG = mock_audio_config()

INTERFACE = resolve_processor_interface(PROCESSOR, HF_CONFIG)
print(json.dumps(INTERFACE, indent=2, default=str))
print()
print("presence is not support:")
print(f"  supports_audio (components exist) = {INTERFACE['supports_audio']}")
print("  whether an audio input can actually be built is settled in section 7")
"""
)


# ------------------------------------------------------- 7. resolve protocol

markdown(
    """
## 7. Resolve the native audio protocol

`resolve_audio_interface` does not read documentation. It pushes a **generated**
waveform through the real processor at two durations and requires a verified
placeholder span each time, or it raises `SpokenAudioUnsupportedError` naming
which link broke.

This cell also reproduces the original failure side by side, so the difference
between the two conventions is visible rather than asserted.
"""
)

code(
    """
# 7. Resolve the protocol by probing it. Still no weights.
from jlens.mmpilot.audio import (
    SpokenAudioUnsupportedError,
    encode_audio_prompt,
    probe_waveform,
    resolve_audio_interface,
    verify_audio_encoding,
)

SAMPLING_RATE = int(getattr(PROCESSOR.feature_extractor, "sampling_rate", 16000))
AUDIO_INTERFACE = None
AUDIO_BLOCKED_REASON = ""

# --- the old convention, reproduced. Features, and no placeholders.
_probe = probe_waveform(1.0, SAMPLING_RATE)
_bare = PROCESSOR(text=AUDIT_PROMPT, audio=_probe, return_tensors="pt")
_bare_placeholders = int((_bare["input_ids"] == HF_CONFIG.audio_token_id).sum())
print("bare processor(text=..., audio=...) call:")
print(f"  input_ids            {tuple(_bare['input_ids'].shape)}")
print(f"  input_features       {tuple(_bare['input_features'].shape)}")
print(f"  audio placeholders   {_bare_placeholders}   <-- the blocker")
print("  no exception raised  <-- why this was not caught earlier")

# --- the supported convention.
_templated = encode_audio_prompt(PROCESSOR, AUDIT_PROMPT, _probe)
_templated_placeholders = int((_templated["input_ids"] == HF_CONFIG.audio_token_id).sum())
print()
print("chat-template audio content block:")
print(f"  input_ids            {tuple(_templated['input_ids'].shape)}")
print(f"  input_features       {tuple(_templated['input_features'].shape)}")
print(f"  audio placeholders   {_templated_placeholders}")

print()
try:
    AUDIO_INTERFACE = resolve_audio_interface(
        PROCESSOR,
        HF_CONFIG,
        model_repo_id=MODEL_REPO_ID if RUN_REAL_AUDIO_AUDIT else "mock/gemma-like",
        model_revision=MODEL_REVISION if RUN_REAL_AUDIO_AUDIT else "mock-rev",
        processor_revision=MODEL_REVISION if RUN_REAL_AUDIO_AUDIT else "mock-rev",
    )
    print("PROTOCOL RESOLVED")
    print(json.dumps(AUDIO_INTERFACE.to_record(), indent=2, default=str))
except SpokenAudioUnsupportedError as error:
    AUDIO_BLOCKED_REASON = str(error)
    print("PROTOCOL NOT RESOLVED —", error.reason)
    print(error)
    print()
    print(
        "Sections 8-12 cannot run. Section 13 will emit AUDIO BLOCKED with this "
        "reason and the nearest supported configuration."
    )
"""
)


# ------------------------------------------------------------ 8. real audio

markdown(
    """
## 8. Load one or two real SpokenCOCO waveforms

Two **different** recordings, because one cannot show that the model
distinguishes them. Decoded to float32 mono at the feature extractor's own rate;
a rate mismatch is refused rather than resampled, since passing an ndarray at
the wrong rate silently reinterprets the recording's pitch and duration.

**No filename, transcript or caption is ever handed to the model.** The paths
below are used to open files and are then dropped.
"""
)

code(
    """
# 8. Two different recordings. Generated in MOCK; read from Drive in a real run.
WAVEFORMS = []

if AUDIO_INTERFACE is None:
    print("skipped: the protocol did not resolve in section 7")
elif RUN_REAL_AUDIO_AUDIT:
    import soundfile as sf

    _candidates = sorted(RESOLVED_AUDIO_ROOT.rglob("*.wav"))[:2]
    if len(_candidates) < 2:
        raise RuntimeError(
            f"found {len(_candidates)} .wav file(s) under {RESOLVED_AUDIO_ROOT}; "
            "two different recordings are required. Nothing is downloaded and no "
            "transcript is substituted."
        )
    for _index, _path in enumerate(_candidates):
        _samples, _rate = sf.read(str(_path), dtype="float32")
        if _samples.ndim > 1:
            _samples = _samples.mean(axis=1)
        if int(_rate) != SAMPLING_RATE:
            raise RuntimeError(
                f"{_path.name} is {_rate} Hz but the feature extractor expects "
                f"{SAMPLING_RATE} Hz. Resample on load; this notebook refuses to "
                "reinterpret the rate silently."
            )
        # Label by position, never by filename: the label is printed, and a
        # semantic filename in a printed label is one step from a prompt.
        WAVEFORMS.append((f"recording_{_index}", _samples))
        print(
            f"recording_{_index}: {_samples.shape[0]} samples, "
            f"{_samples.shape[0] / SAMPLING_RATE:.2f} s, {_samples.dtype}"
        )
else:
    for _index, _seconds in enumerate((1.0, 1.5)):
        WAVEFORMS.append((f"recording_{_index}", probe_waveform(_seconds, SAMPLING_RATE, seed=_index + 1)))
        print(f"recording_{_index}: MOCK generated, {_seconds:.2f} s")

print()
print(f"{len(WAVEFORMS)} distinct recording(s) ready; silence is added by the audit")
"""
)


# ----------------------------------------------- 9. model + the audit itself

markdown(
    """
## 9. Verify placeholder and audio-tower behavior

This is where the model is loaded, if it is loaded at all.

The audit runs **once**, in this cell, and produces every measurement sections
9–12 report. Running it once is deliberate: the numbers printed below and the
numbers written into the report are then necessarily the same numbers.

Sections 9–12 each display the checks they own.
"""
)

code(
    """
# 9. Load the model (gated), then run the audit once.
from jlens.mmpilot.audio_audit import run_audio_audit

BACKEND = None
AUDIT = None
CANDIDATE_IDS = {}

if AUDIO_INTERFACE is None:
    print("skipped: the protocol did not resolve in section 7")
elif not RUN_MODEL_STAGE:
    print(
        "RUN_MODEL_STAGE is False — no model is loaded. Sections 9-12 need one; "
        "section 13 will report what was established without it."
    )
elif RUN_REAL_AUDIO_AUDIT:
    from jlens.mmpilot.real_backend import build_real_backend

    BUNDLE = build_real_backend(
        MODEL_REPO_ID,
        revision=MODEL_REVISION,
        token=HF_TOKEN,
        device=ENVIRONMENT["device"],
        allow_model_load=CONFIRM_MODEL_LOAD,
        expect_n_layers=EXPECT_N_LAYERS,
        expect_d_model=EXPECT_D_MODEL,
        expect_vocab_size=EXPECT_VOCAB,
        resolve_audio=True,
    )
    BACKEND = BUNDLE.backend
    AUDIO_INTERFACE = BUNDLE.audio_interface or AUDIO_INTERFACE
    AUDIO_BLOCKED_REASON = BUNDLE.audio_blocked_reason or AUDIO_BLOCKED_REASON
    print(json.dumps(BUNDLE.architecture, indent=2, default=str))
else:
    from jlens.mmpilot.mock import build_mock_audio_backend

    BACKEND, PROCESSOR, AUDIO_INTERFACE = build_mock_audio_backend()
    print("MOCK backend ready")

if BACKEND is not None:
    from jlens.mmpilot.audio_audit import select_scoring_candidates

    # Chosen by measuring token lengths against the live tokenizer, never by
    # assuming them. The pilot's behavioral concepts are single tokens under
    # this tokenizer (" cat" -> [5866], " dog" -> [4799]) and so cannot
    # exercise complete-sequence scoring at all.
    CANDIDATE_IDS = select_scoring_candidates(BACKEND)
    print("scoring candidates (selected after tokenization):")
    for _name, _ids in CANDIDATE_IDS.items():
        print(f"  {_name:16s} n_tokens={len(_ids)} ids={_ids}")

    # Ask about the options actually being scored. Still no caption.
    QUESTION = build_question(sorted(CANDIDATE_IDS))
    AUDIT_PROMPT = build_prompt(QUESTION, modality="spoken_audio")
    print(f"\\naudit prompt: {AUDIT_PROMPT!r}")

    _layers = [layer for layer in AUDIT_LAYERS if layer < BACKEND.n_layers] or [
        BACKEND.n_layers - 1
    ]
    AUDIT = run_audio_audit(
        BACKEND,
        prompt=AUDIT_PROMPT,
        waveforms=WAVEFORMS,
        sampling_rate=SAMPLING_RATE,
        layers=_layers,
        candidate_ids=CANDIDATE_IDS,
        # The recording is the only evidence: nothing from the dataset may
        # appear in the prompt.
        forbidden_text=[
            *(str(label) for label, _ in WAVEFORMS),
            "caption",
            "transcript",
            ".wav",
        ],
        mode=MODE,
        environment=ENVIRONMENT,
    )
    _by_name = {check.name: check for check in AUDIT.checks}
    for _name in ("placeholder_span", "placeholder_feature_agreement",
                  "final_prompt_position", "audio_tower_invoked"):
        _check = _by_name.get(_name)
        if _check is not None:
            print(f"\\n{_name}: {'PASS' if _check.passed else 'FAIL'}")
            print("  " + json.dumps(_check.detail, default=str)[:400])
"""
)


# --------------------------------------------------- 10. does the audio matter

markdown(
    """
## 10. Real waveform vs silence vs a different waveform

A placeholder span proves the *text* was built correctly. These two checks are
what prove the **recording** reached the model and mattered: the same prompt,
the same token count, and a different acoustic input has to move the logits.
"""
)

code(
    """
# 10. The recording has to make a difference.
if AUDIT is None:
    print("skipped: no audit was run")
else:
    _by_name = {check.name: check for check in AUDIT.checks}
    for _name in ("waveform_differs_from_silence", "waveforms_differ_from_each_other"):
        _check = _by_name[_name]
        print(f"{_name}: {'PASS' if _check.passed else 'FAIL'}")
        print(f"  max |Δlogit| = {_check.detail['max_abs_logit_diff']:.6g} "
              f"(> {_check.detail['tolerance']:g} required)")
    if not all(_by_name[n].passed for n in
               ("waveform_differs_from_silence", "waveforms_differ_from_each_other")):
        print(
            "\\nA recording that does not move the logits means the audio channel "
            "is decorative. That is AUDIO INVALID, not AUDIO READY."
        )
"""
)


# --------------------------------------------------------- 11. answer scoring

markdown(
    """
## 11. Complete candidate-sequence scoring

Two different questions live here, and only the first is asked.

**Scoring validity** — does the mechanism correctly score *complete* candidate
token sequences? Every candidate's whole sequence is scored by teacher-forced
conditional log likelihood, and the check requires finite per-token and
aggregate terms, an aggregate equal to the sum of its own per-token terms, at
least one genuinely multi-token candidate, no prefix-degenerate pair, and a
score that does not move when the candidate order changes.

**Behavioral capability** — can Gemma recognize a concept from the recording?
**Not measured here, and not measurable here.** The waveform was never selected
to be about any candidate, so which candidate scores highest is meaningless. It
is printed, clearly labelled, and is **not** a criterion. That question belongs
to the SpokenCOCO experiment.

> Candidates are selected in section 9 by measuring token lengths against the
> live tokenizer. An earlier version of this audit used the pilot's behavioral
> concepts, which Gemma encodes as single tokens — complete-sequence scoring ran
> correctly and returned finite scores, and the audit still reported FAIL
> because its fixture could not exercise the multi-token path. The rule now
> separates "the scorer works" from "the fixture was capable of showing it".
"""
)

code(
    """
# 11. Whole-sequence scoring on an audio input. Validity only.
if AUDIT is None:
    print("skipped: no audit was run")
else:
    _check = {check.name: check for check in AUDIT.checks}["candidate_sequence_scoring"]
    _detail = _check.detail
    print(f"candidate_sequence_scoring: {'PASS' if _check.passed else 'FAIL'}")
    print(f"rule:                 {_detail['rule']}")
    print(f"measures:             {_detail['measures']}")
    print(f"tokens per candidate: {_detail['n_tokens_per_candidate']}")
    print(f"order-invariance max |delta|: {_detail['order_invariance_max_abs_delta']:.3e}")
    for _name, _row in sorted(_detail["scores"].items()):
        print(
            f"  {_name:16s} sum={_row['sum_logprob']:+.4f} "
            f"mean={_row['mean_logprob']:+.4f} n_tokens={_row['n_tokens']}"
        )
        print(f"    ids={_row['token_ids']}")
        print(f"    per-token={[round(v, 4) for v in _row.get('token_logprobs', [])]}")
    for _failure in _detail.get("failures", []):
        print(f"  FAILURE: {_failure}")
    print()
    print("SCORING VALIDITY is what passed or failed above: whether complete")
    print("token sequences are scored correctly.")
    print()
    print(f"highest-scoring candidate: {_detail['reported_only_argmax']!r}")
    print("  ^ REPORTED ONLY, NOT A CRITERION. The recording was not selected to")
    print("    be about any candidate, so this says nothing. Whether Gemma can")
    print("    recognize a concept from speech is BEHAVIORAL CAPABILITY, which")
    print("    this notebook does not measure and the SpokenCOCO experiment does.")
"""
)


# ------------------------------------------------- 12. activations, invariance

markdown(
    """
## 12. Activation capture and invariance

Three things, all about the harness rather than the model:

- the residual at the **true final prompt token**, captured after audio
  expansion moved that position;
- a capture hook must not change the logits;
- a coefficient-zero edit must reproduce the clean result exactly.

If any of these fails on an audio input, every later audio number is
untrustworthy — which is what `AUDIO INVALID` says.
"""
)

code(
    """
# 12. Capture at the final prompt token, and the two invariance checks.
ACTIVATIONS = {}
if AUDIT is None:
    print("skipped: no audit was run")
else:
    from jlens.mmpilot.jspace import capture_final_prompt_activations

    _built = BACKEND.build_inputs(
        prompt=AUDIT_PROMPT,
        modality="spoken_audio",
        audio=WAVEFORMS[0][1],
        sampling_rate=SAMPLING_RATE,
    )
    _layers = [layer for layer in AUDIT_LAYERS if layer < BACKEND.n_layers] or [
        BACKEND.n_layers - 1
    ]
    ACTIVATIONS = capture_final_prompt_activations(BACKEND, _built, _layers)
    print(f"audio span            {_built.modality_token_range}")
    print(f"prompt_len            {_built.prompt_len}")
    print(f"final prompt position {_built.final_prompt_position}  "
          f"(after the span, not inside it)")
    for _layer, _vector in sorted(ACTIVATIONS.items()):
        print(f"  layer {_layer:3d}: shape {tuple(_vector.shape)} "
              f"norm {float(_vector.norm()):.4f}")

    print()
    _by_name = {check.name: check for check in AUDIT.checks}
    for _name in ("capture_noop", "zero_intervention"):
        _check = _by_name[_name]
        print(f"{_name}: {'PASS' if _check.passed else 'FAIL'}")
        print("  " + json.dumps(_check.detail, default=str)[:300])
"""
)


# ------------------------------------------------------------- 13. the report

markdown(
    """
## 13. AUDIO READY / AUDIO BLOCKED / AUDIO INVALID

The verdict is computed in `jlens.mmpilot.audio_audit.verdict_from_checks`, not
written by hand here, and a required check that never ran counts as a failure
rather than a silent pass.

The report is written to a **new** audit run directory. Completed pilot,
robustness, localization and J-space runs are refused as write targets outright.
"""
)

code(
    """
# 13. Emit the verdict and write the report.
from jlens.mmpilot.audio_audit import (
    AUDIO_BLOCKED,
    AUDIO_INVALID,
    AUDIO_READY,
    AudioAuditResult,
    Check,
    new_audit_run_dir,
    write_audit_report,
)

if AUDIT is None:
    # Nothing could be measured. That is BLOCKED, never INVALID.
    AUDIT = AudioAuditResult(
        verdict=AUDIO_BLOCKED,
        checks=[Check("protocol_resolved", passed=AUDIO_INTERFACE is not None, blocking=True)],
        audio_interface=AUDIO_INTERFACE.to_record() if AUDIO_INTERFACE else None,
        blocked_reason=AUDIO_BLOCKED_REASON
        or "no model stage ran, so no audio behavior was measured",
        environment=ENVIRONMENT,
        mode=MODE,
    )

RUN_DIR = new_audit_run_dir(RESOLVED_RUNS_ROOT, mode=MODE)
WRITTEN = write_audit_report(RUN_DIR, AUDIT)

print("=" * 72)
print(AUDIT.verdict)
print("=" * 72)
if AUDIT.failed:
    print("failed checks:", ", ".join(AUDIT.failed))
if AUDIT.blocked_reason:
    print("blocked because:", AUDIT.blocked_reason)
print()
print(f"run directory {RUN_DIR}")
print(f"json          {WRITTEN['json']}")
print(f"markdown      {WRITTEN['markdown']}")

if AUDIT.verdict == AUDIO_READY:
    print()
    print("AUDIO READY is engineering evidence only. It establishes that native")
    print("spoken audio is technically usable. It is NOT evidence that J-space")
    print("concept representations transfer to or from spoken audio — that")
    print("experiment has not been run.")
elif AUDIT.verdict == AUDIO_INVALID:
    print()
    print("AUDIO INVALID: an input was built and then failed a check that makes")
    print("later numbers untrustworthy. Do not run the study.")

print()
print("-" * 72)
print("fingerprint consequences")
print("-" * 72)
print("Any change to the model id, model revision, processor revision,")
print("Transformers version, audio protocol, placeholder convention or")
print("hook-position convention MUST change the run fingerprint.")
if AUDIO_INTERFACE is not None:
    print()
    print(f"audio protocol fingerprint: {AUDIO_INTERFACE.protocol_fingerprint}")
    print("Bind it into RunFingerprint.extra as 'audio_protocol_fingerprint' so a")
    print("run recorded under one protocol refuses to resume under another.")
print()
print(f"J-lens compatibility: the validated text-calibrated lens was fitted and")
print(f"validated against model revision {MODEL_REVISION}. This audit does not")
print("change the checkpoint, so that lens remains applicable. If enabling audio")
print("ever required a different checkpoint or revision, the lens could NOT be")
print("reused and a new one would have to be fitted first.")
"""
)


# ------------------------------------------------------------- 14. resume

markdown(
    """
## 14. Resume and status

This audit is a single short pass, so there is nothing to resume in the sense
the pilot means it. What is worth showing is what exists on disk, what this run
bound itself to, and what the next step is.
"""
)

code(
    """
# 14. Status: what exists, what it was bound to, and what comes next.
print("audit run directories under the runs root:")
_existing = sorted(p.name for p in Path(RESOLVED_RUNS_ROOT).glob("audioaudit_*"))
for _name in _existing:
    print(f"  {_name}")
print(f"  ({len(_existing)} total)")

print()
print("this run:")
print(f"  mode      {MODE}")
print(f"  verdict   {AUDIT.verdict}")
print(f"  directory {RUN_DIR}")
print(f"  checksum  {AUDIT.to_dict()['report_checksum']}")

print()
print("completed scientific runs were neither read nor modified by this notebook.")

print()
print("next:")
if MODE == "mock":
    print("  This was a MOCK run. It proves the notebook executes and proves")
    print("  nothing about Gemma. For the real answer set, in section 2:")
    print("      RUN_REAL_AUDIO_AUDIT = True")
    print("      RUN_MODEL_STAGE      = True")
    print("      CONFIRM_MODEL_LOAD   = True")
    print("  on an L4 (or any 24 GB GPU). Expect ~10 minutes, dominated by the")
    print("  ~16 GB model download; the audit itself is a few dozen forward passes.")
elif AUDIT.verdict == AUDIO_READY:
    print("  Send back audio_audit.md. Native spoken audio is technically usable;")
    print("  designing the scientific study is a separate decision.")
else:
    print("  Send back audio_audit.md. It names the exact broken link and the")
    print("  nearest supported configuration.")
"""
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
