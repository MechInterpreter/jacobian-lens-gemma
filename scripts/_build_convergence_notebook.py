# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Generate ``notebooks/multimodal_jspace_output_convergence_audit_colab.ipynb``.

Written from source rather than edited as JSON, so the committed notebook stays
output-free and byte-reproducible. Run
``python scripts/_build_convergence_notebook.py`` after changing a cell; a test
regenerates it and fails on drift.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TARGET = (
    REPO_ROOT
    / "notebooks"
    / "multimodal_jspace_output_convergence_audit_colab.ipynb"
)

CELLS: list[tuple[str, str]] = []


def markdown(text: str) -> None:
    CELLS.append(("markdown", text.strip("\n")))


def code(text: str) -> None:
    CELLS.append(("code", text.strip("\n")))


markdown(
    """
# Output-convergence timing audit — does L35 transfer precede the answer?

**The completed result this follows up.** The native spoken-audio run
`mmaudio_native_audio_transfer_20260806T144822` established
`THREE_MODALITY_GO`: behavioral capability in text, image and spoken audio,
cross-modal J-space structure, controlled causal transfer at **layer 35**,
replication at **38** and **40**, one synchronized group per photograph, no
image-level pseudoreplication, and capability-ineligible `zebra` cells excluded
from every principal claim. That run is **finished and immutable**. This
notebook reruns none of it.

**The gap it left.** Its own module docstring says so outright: layer 35 is
primary *because it is the earliest independently confirmed lens, and for no
other reason* — "not 'pre-language', not 'pre-convergence' ... convergence
timing is unresolved here". A final-prompt-token residual intervention shows
that editing layer 35 changes the answer. It does not show that layer 35
*precedes* the answer.

**The narrow question here.** At each validated layer — L35, L38, L40 — how far
has the clean final-prompt-token residual already converged onto the model's own
final candidate answer, measured through the model's **own frozen output head**?
Then: compare that trajectory with the already-completed causal evidence at the
same layers.

## What is measured, exactly

For each **stored** clean residual `h` at an audited layer:

```
logits = lm_head(final_norm(h))            # the live modules, called, not reimplemented
logits = softcap * tanh(logits / softcap)  # if the config declares a softcap
```

restricted to the six fixed behavioral candidates. No lens, no dictionary, no
J-space code, no intervention, and **no learned probe** enters this number. That
is what makes the readout *native*: it is the model's own output pathway,
applied one layer early.

Section 8 does not assume the RMSNorm convention — it *detects* which one the
live module implements and checks the whole path against the model's own
`unembed`. A hand-rolled `x / rms * w` is exactly the standard-logit-lens
mistake this architecture punishes.

## The interpretation boundary

A weak direct readout means the representation has **not converged under the
predeclared criterion in this notebook**. It is *not* proof that linguistic
information is absent, and it says nothing about what a nonlinear decoder or a
trained probe could recover. Every artifact written here carries that sentence.
The permitted phrasing is "before native direct-readout convergence" and
"before obvious output-answer convergence under the predeclared criterion".
Never "pre-linguistic". Never "language-free".

## The criterion is two-sided, on purpose

There is an obvious way to cheat an audit like this: set the "converged" bar
high, watch layer 35 fail it, and call the failure evidence of pre-convergence.
So there are **two** bars with a deliberate gap. A layer is `CONVERGED` only
above the upper one and `NOT_CONVERGED` only below the lower one; anything
between is `AMBIGUOUS` and can produce nothing but
`INCONCLUSIVE_CONVERGENCE_TIMING`. The claim-supporting state has to be earned
against a bar of its own. Section 4 prints the full rule before any
result-producing cell runs.

## Nothing starts by itself

Two switches, both `False` in the committed notebook, both set by hand:

| switch | what it unlocks |
| --- | --- |
| `RUN_REAL_CONVERGENCE_AUDIT` | the real completed run on Drive instead of the deterministic MOCK world |
| `CONFIRM_MODEL_LOAD` | acknowledges loading Gemma **once**, for its output head only |

Clicking "Run all" on the committed notebook runs the MOCK world and spends
nothing. This audit executes **zero model forward passes** even on the real
path: it reads stored activations and applies two frozen modules to them.
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
# 1c. Install the repository and verify that `import jlens` resolves here.
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
        f"`import jlens` resolved to {jlens.__file__}, not this checkout"
    )
print(f"jlens  {jlens.__file__}")
print(f"cwd    {os.getcwd()}")
'''
)

markdown(
    """
## 2. The two switches, and the pins the real path refuses to guess

`RUN_REAL_CONVERGENCE_AUDIT = False` runs the MOCK world. `CONFIRM_MODEL_LOAD`
is separate because loading Gemma is the only expensive thing this notebook can
do, and it does it **once**, for two frozen modules.

Some pins are recorded in this repository and are filled in below: the model
repo and revision, the three published lens checksums, and the spoken-audio
protocol fingerprint. Two pins are properties of the *completed run* and are
deliberately left empty:

- `EXPECTED_RUN_FINGERPRINT_DIGEST` — read it from the completed run's
  `run_manifest.json` (`fingerprint_digest`).
- `EXPECTED_PROCESSOR_REVISION` — read it from the completed run's
  `fingerprint.json` (`processor_revision`).

Section 6 prints what the run records so you can compare, but **printing is not
verifying**: the audit compares the run against what *you* pinned, so a value
copied out of the run and pasted back in proves nothing. Paste them from your
own record of the run.
"""
)

code(
    '''
# 2. Switches and pins. Both switches are False in the committed notebook.
RUN_REAL_CONVERGENCE_AUDIT = False
CONFIRM_MODEL_LOAD = False

# Optional, secondary, and unable to change any verdict.
RUN_SECONDARY_PROBE = False

# ---------------------------------------------------------- the completed run
COMPLETED_RUN_DIR = (
    "/content/drive/MyDrive/jacobian-lens-gemma/runs/"
    "mmaudio_native_audio_transfer_20260806T144822"
)
# Where this audit writes. OUTSIDE the completed run, always.
AUDIT_ROOT = "/content/drive/MyDrive/jacobian-lens-gemma/audits"
AUDIT_NAME = "output_convergence_v1"

# ----------------------------------------------- pins recorded in this repo
MODEL_REPO_ID = "google/gemma-4-E4B-it"
MODEL_REVISION = "fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd"
EXPECT_N_LAYERS, EXPECT_D_MODEL, EXPECT_VOCAB = 42, 2560, 262144

PUBLISHED_LENS_CHECKSUMS = {
    35: "sha256:64fb02d718ac48adc1bced99e2eff3c2215052ba144d5dedac05f17936a96ed1",
    38: "sha256:c8508fbf2b916e5d9aaeb8711a30f76414ee16478c5f6cc321e57e2fe846d1c0",
    40: "sha256:8a90f67eeb9bb5db14e6715b8bc516a899da1c3210d0662ec7fa177b5409f7d7",
}
AUDIO_PROTOCOL_VERSION_EXPECTED = "jlens.mmpilot.native_spoken_audio.v1"
AUDIO_PROTOCOL_FINGERPRINT_EXPECTED = (
    "sha256:9ad8bcc9420a7983f6e3b75d5d7080c0e2fcf0a94a76431917fcde73ba777920"
)

# -------------------------------- pins that belong to the run, not the repo
EXPECTED_RUN_FINGERPRINT_DIGEST = ""
EXPECTED_PROCESSOR_REVISION = ""

# Environment overrides, so a real run does not require editing the notebook.
EXPECTED_RUN_FINGERPRINT_DIGEST = (
    os.environ.get("MMCONV_RUN_FINGERPRINT") or EXPECTED_RUN_FINGERPRINT_DIGEST
)
EXPECTED_PROCESSOR_REVISION = (
    os.environ.get("MMCONV_PROCESSOR_REVISION") or EXPECTED_PROCESSOR_REVISION
)

print(f"RUN_REAL_CONVERGENCE_AUDIT = {RUN_REAL_CONVERGENCE_AUDIT}")
print(f"CONFIRM_MODEL_LOAD         = {CONFIRM_MODEL_LOAD}")
print(f"RUN_SECONDARY_PROBE        = {RUN_SECONDARY_PROBE}")
print()
if not RUN_REAL_CONVERGENCE_AUDIT:
    print("MOCK mode: no Drive, no Gemma, no completed run is opened.")
'''
)

code(
    '''
# 3. Import the audit. Nothing here computes a result.
from jlens.mmpilot.convergence import (
    AUDITED_LAYERS,
    CONVERGENCE_CRITERION,
    CONVERGENCE_PROTOCOL,
    CRITERION_TEXT,
    INTERPRETATION_BOUNDARY,
    LENS_INVALID_LAYERS,
    MODALITIES,
    PRIMARY_LAYER,
    WRONG_LAYER_CONTROL_NOTE,
    ConvergenceFingerprint,
    ConvergenceStore,
    assert_lens_valid_layer,
    assert_run_unchanged,
    audit_native_head,
    build_population,
    clean_predictions_from_interventions,
    head_from_model,
    protected_file_checksums,
    resolve_candidate_tokens,
    run_convergence_audit,
    verify_completed_run,
)
from jlens.mmpilot.published_lens import (
    CONFIRMED_LAYERS,
    FAILED_CONFIRMATION_LAYERS,
    combined_lens_checksum,
)

LAYERS = tuple(AUDITED_LAYERS)
COMBINED_LENS_CHECKSUM = combined_lens_checksum(PUBLISHED_LENS_CHECKSUMS)

# The audited layers must be exactly the layers the calibration run confirmed.
# A layer that failed confirmation is never audited and never interpreted.
if tuple(sorted(LAYERS)) != tuple(sorted(CONFIRMED_LAYERS)):
    raise RuntimeError(
        f"audited layers {LAYERS} are not the confirmed layers {CONFIRMED_LAYERS}"
    )
for _layer in LAYERS:
    assert_lens_valid_layer(_layer)

print(f"protocol         {CONVERGENCE_PROTOCOL}")
print(f"audited layers   {list(LAYERS)}  (lens confirmation: PASS)")
print(f"never audited    {list(FAILED_CONFIRMATION_LAYERS)}  (failed confirmation)")
print(f"primary layer    L{PRIMARY_LAYER}  (the completed causal result's layer)")
print(f"modalities       {list(MODALITIES)}")
print(f"combined lens    {COMBINED_LENS_CHECKSUM}")
'''
)

markdown(
    """
## 4. The predeclared criterion, printed before any result exists

This cell runs before anything is measured, in MOCK and on the real path alike.
The criterion's digest is bound into the audit fingerprint, so editing the rule
invalidates stored rows instead of silently rescoring them under a rule they
were not produced by.
"""
)

code(
    '''
# 4. The rule, in full, before any result-producing cell.
print(CRITERION_TEXT)
print("=" * 72)
print(f"criterion digest {CONVERGENCE_CRITERION.digest}")
print("=" * 72)
'''
)

markdown(
    """
## 5. MOCK — the deterministic synthetic world

Runs whenever `RUN_REAL_CONVERGENCE_AUDIT` is `False`, which is the committed
default. It builds a synthetic *completed run* with the same `UnitStore` layout
and artifact names as the real one, a tiny output head that uses Gemma's
`(1 + weight)` RMSNorm convention, and a signal strength per layer that is the
only knob.

Five worlds are run, plus one that removes the causal support. The matrix below
is the point: it shows the branches being exercised rather than asserting they
are.

In particular `flat_weak` — every layer weak, so layer 35 is `NOT_CONVERGED` and
yet no later layer is clearly more converged — must **not** produce
`PRE_CONVERGENCE_TRANSFER_SUPPORTED`. A weak direct readout on its own is not
the claim.
"""
)

code(
    '''
# 5. The MOCK world, and the full verdict matrix.
import tempfile

MOCK_RESULT = None
MOCK_MATRIX = None

if not RUN_REAL_CONVERGENCE_AUDIT:
    from jlens.mmpilot.convergence_mock import (
        MOCK_MODES,
        MockWorldSpec,
        mock_verdict_matrix,
        run_mock_convergence_audit,
    )

    MOCK_ROOT = Path(tempfile.mkdtemp(prefix="mmconv_mock_"))
    MOCK_RESULT = run_mock_convergence_audit(
        MOCK_ROOT / "completed_run",
        MOCK_ROOT / "audit",
        spec=MockWorldSpec(mode="pre_convergence"),
        run_probe=True,
        bootstrap_resamples=400,
    )
    print("MOCK world 'pre_convergence'")
    print(f"  verdict          {MOCK_RESULT['verdict']['verdict']}")
    print(f"  readout mode     {MOCK_RESULT['tokenization']['readout_mode']}")
    print(
        f"  norm convention  "
        f"{MOCK_RESULT['head_audit']['norm_weight_convention']} "
        f"(softcap {MOCK_RESULT['head_audit']['final_logit_softcapping']})"
    )
    print(f"  controls passed  {MOCK_RESULT['controls']['all_controls_passed']}")
    print(f"  run unchanged    {MOCK_RESULT['immutability']['unchanged']}")
    print()
    for _row in MOCK_RESULT["table"]:
        print(
            f"  L{_row['layer']:<3} {str(_row['convergence_classification']):<14} "
            f"agreement={_row['clean_agreement_unique']:.3f} "
            f"rank={_row['median_target_rank']:.1f} "
            f"causal={_row['causal_transfer_verdict']}"
        )
    print()
    MOCK_MATRIX = mock_verdict_matrix(
        MOCK_ROOT / "matrix", bootstrap_resamples=200
    )
    print("verdict matrix — every branch, exercised:")
    for _mode, _verdict in MOCK_MATRIX.items():
        print(f"  {_mode:<40s} {_verdict}")
    print()
    assert MOCK_MATRIX["pre_convergence"] == "PRE_CONVERGENCE_TRANSFER_SUPPORTED"
    assert MOCK_MATRIX["converged_early"] == "TRANSFER_AT_OR_AFTER_CONVERGENCE"
    # The two that matter most: a weak readout alone, and a weak readout without
    # the causal result, must both refuse the claim.
    assert MOCK_MATRIX["flat_weak"] == "INCONCLUSIVE_CONVERGENCE_TIMING"
    assert (
        MOCK_MATRIX["pre_convergence_without_causal_support"]
        == "INCONCLUSIVE_CONVERGENCE_TIMING"
    )
    print("MOCK assertions passed: a weak direct readout alone cannot produce")
    print("PRE_CONVERGENCE_TRANSFER_SUPPORTED.")
else:
    print("real mode requested; the MOCK world is skipped")
'''
)

markdown(
    """
## 6. Stage 1 — provenance and integrity of the completed run

Mount Drive, open the **named** completed run, and refuse anything that does not
match a pin. Nothing is created, nothing is discovered by scanning, and no value
is defaulted: a missing field fails the check it belongs to.

The protected files are checksummed **before** the audit and again at the end
(section 10). The completed run must be byte-identical across the whole session.
"""
)

code(
    '''
# 6a. Mount Drive and take the "before" checksums of the protected files.
COMPLETED_RUN = None
CHECKSUMS_BEFORE = None

if RUN_REAL_CONVERGENCE_AUDIT:
    if IN_COLAB:
        from google.colab import drive

        drive.mount("/content/drive")
    COMPLETED_RUN = Path(COMPLETED_RUN_DIR)
    if not (COMPLETED_RUN / "fingerprint.json").is_file():
        raise RuntimeError(
            f"{COMPLETED_RUN} has no fingerprint.json. This audit opens one "
            "explicitly named completed run and refuses to create or discover "
            "anything."
        )
    CHECKSUMS_BEFORE = protected_file_checksums(COMPLETED_RUN)
    print(f"completed run {COMPLETED_RUN}")
    for _name, _checksum in sorted(CHECKSUMS_BEFORE.items()):
        print(f"  {_name:<58s} {_checksum or '(absent)'}")
else:
    print("MOCK mode: no Drive is mounted and no completed run is opened")
'''
)

code(
    '''
# 6b. Refuse to proceed without the two run-specific pins.
#
# Printed below for comparison only. Printing is not verifying: the audit
# compares the run against what YOU pinned, so a value copied out of the run and
# pasted back proves nothing. Paste them from your own record of the run.
if RUN_REAL_CONVERGENCE_AUDIT:
    import json as _json

    RUN_FINGERPRINT_PAYLOAD = _json.loads(
        (COMPLETED_RUN / "fingerprint.json").read_text(encoding="utf-8")
    )
    print("the run records:")
    print(f"  model_repo_id       {RUN_FINGERPRINT_PAYLOAD.get('model_repo_id')}")
    print(f"  model_revision      {RUN_FINGERPRINT_PAYLOAD.get('model_revision')}")
    print(f"  processor_revision  {RUN_FINGERPRINT_PAYLOAD.get('processor_revision')}")
    print(f"  layers              {RUN_FINGERPRINT_PAYLOAD.get('layers')}")
    print()
    _missing = [
        name
        for name, value in (
            ("EXPECTED_RUN_FINGERPRINT_DIGEST", EXPECTED_RUN_FINGERPRINT_DIGEST),
            ("EXPECTED_PROCESSOR_REVISION", EXPECTED_PROCESSOR_REVISION),
        )
        if not value
    ]
    if _missing:
        raise RuntimeError(
            "refusing to audit an unpinned run. Set "
            + ", ".join(_missing)
            + " in section 2 (or via the MMCONV_RUN_FINGERPRINT / "
            "MMCONV_PROCESSOR_REVISION environment variables). Read the "
            "fingerprint digest from the completed run's run_manifest.json and "
            "the processor revision from its fingerprint.json, and paste them "
            "from your own record — not from the values printed above."
        )
'''
)

code(
    '''
# 6c. Verify every Stage-1 precondition. The first mismatch stops the audit.
INTEGRITY = None

if RUN_REAL_CONVERGENCE_AUDIT:
    COMPLETED_SUMMARY_PATH = (
        COMPLETED_RUN / "native_audio_transfer_summary_capability_filtered_v2.json"
    )
    if not COMPLETED_SUMMARY_PATH.is_file():
        COMPLETED_SUMMARY_PATH = COMPLETED_RUN / "native_audio_transfer_summary.json"
    COMPLETED_SUMMARY = _json.loads(
        COMPLETED_SUMMARY_PATH.read_text(encoding="utf-8")
    )
    INTEGRITY = verify_completed_run(
        run_dir=COMPLETED_RUN,
        fingerprint_payload=RUN_FINGERPRINT_PAYLOAD,
        expected_fingerprint_digest=EXPECTED_RUN_FINGERPRINT_DIGEST,
        expected_model_repo_id=MODEL_REPO_ID,
        expected_model_revision=MODEL_REVISION,
        expected_processor_revision=EXPECTED_PROCESSOR_REVISION,
        expected_audio_protocol_version=AUDIO_PROTOCOL_VERSION_EXPECTED,
        expected_audio_protocol_fingerprint=AUDIO_PROTOCOL_FINGERPRINT_EXPECTED,
        expected_lens_checksums=PUBLISHED_LENS_CHECKSUMS,
        expected_combined_lens_checksum=COMBINED_LENS_CHECKSUM,
        summary=COMPLETED_SUMMARY,
        layers=LAYERS,
    )
    INTEGRITY["immutability"] = {"checksums": dict(CHECKSUMS_BEFORE)}
    print(f"verdict source  {COMPLETED_SUMMARY_PATH.name}")
    for _check in INTEGRITY["checks"]:
        print(f"  {'PASS' if _check['passed'] else 'FAIL'}  {_check['check']}")
    print()
    print(f"completed run's own verdict: {INTEGRITY['completed_overall_verdict']}")
    print("that verdict is read and never revisited by this audit")
'''
)

markdown(
    """
## 7. Stage 2 — the frozen evaluation population

The population is the completed run's, taken as-is. Its stored `activation`
units *are* the clean final-prompt-token residuals, so nothing is re-extracted
and no media is loaded. The model's clean final answer per sample is recovered
from the zero-alpha intervention units, where the edit is a no-op by
construction.

Concepts, examples, modalities, candidate set and split are **not** reselected.
`zebra` stays in as an explicitly labelled capability-ineligible diagnostic and
enters no principal number and no verdict clause.
"""
)

code(
    '''
# 7. Rebuild the evaluation population from stored units. No model, no media.
POPULATION = None
TOKENIZATION = None

if RUN_REAL_CONVERGENCE_AUDIT:
    from jlens.mmpilot.amend_open import open_existing_store

    COMPLETED_STORE = open_existing_store(
        COMPLETED_RUN, expected_fingerprint=EXPECTED_RUN_FINGERPRINT_DIGEST
    )
    CAPABILITY = COMPLETED_STORE.load("metric", "capability_summary")
    if CAPABILITY is None:
        raise RuntimeError(
            "the completed run has no capability_summary metric unit; the "
            "admissibility rule cannot be applied and no claim can be made"
        )
    FOCAL_CONCEPTS = list(
        COMPLETED_SUMMARY.get("focal_concepts")
        or (COMPLETED_SUMMARY.get("verdicts", {}).get("C_primary_causal", {}) or {}).get(
            "focal_concepts"
        )
        or []
    )
    if not FOCAL_CONCEPTS:
        raise RuntimeError(
            "the completed run's summary does not record its fixed focal "
            "concepts; refusing to invent a concept set"
        )

    TOKENIZATION = resolve_candidate_tokens(CAPABILITY["candidate_token_ids"])
    POPULATION = build_population(
        activations=list(COMPLETED_STORE.load_all("activation").values()),
        clean_predictions=clean_predictions_from_interventions(
            COMPLETED_STORE.load_all("intervention").values()
        ),
        capability=CAPABILITY,
        focal_concepts=FOCAL_CONCEPTS,
        layers=LAYERS,
    )
    print(f"fixed focal concepts       {FOCAL_CONCEPTS}")
    print(f"capability-admissible      {POPULATION['admissible_concepts']}")
    print(f"capability-ineligible      {POPULATION['inadmissible_concepts']} "
          "(descriptive only)")
    print(f"units                      {POPULATION['n_units']}")
    print(f"with a clean reference     {POPULATION['n_with_clean_reference']}")
    print()
    print(f"candidates                 {TOKENIZATION['candidates']}")
    print(f"all single-token           {TOKENIZATION['all_candidates_single_token']}")
    print(f"readout mode               {TOKENIZATION['readout_mode']}")
    print()
    print(TOKENIZATION["scoring_note"])
'''
)

markdown(
    """
## 8. Stage 3 — load Gemma **once**, for two frozen modules

This is the only expensive cell, and it is gated by `CONFIRM_MODEL_LOAD`. The
model is loaded to obtain the final normalization module and the unembedding,
which are then **called** on stored activations. There is no forward pass, no
generation and no tokenization of any prompt.

`audit_native_head` does not assume a formula. It pushes probes through the live
norm, compares the result against both RMSNorm conventions, records which one
matches, and checks the whole path against the model's own `unembed`. A
disagreement stops the audit rather than being reported as a small numerical
difference.
"""
)

code(
    '''
# 8. The frozen output head, audited rather than assumed.
HEAD = None
HEAD_AUDIT = None

if RUN_REAL_CONVERGENCE_AUDIT:
    if not CONFIRM_MODEL_LOAD:
        raise RuntimeError(
            "CONFIRM_MODEL_LOAD is False. The native readout needs the model's "
            "own final norm and unembedding; they cannot be reconstructed "
            "safely from recorded artifacts, because the RMSNorm weight "
            "convention is implementation-specific and getting it wrong is "
            "silent. Set CONFIRM_MODEL_LOAD = True to load Gemma once "
            "(~16 GB download, no forward pass is executed)."
        )
    import torch
    from jlens.gemma4 import load_gemma4

    MODEL, LOAD_INFO = load_gemma4(
        MODEL_REPO_ID,
        revision=MODEL_REVISION,
        dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else None,
        allow_model_load=True,
    )
    for _parameter in MODEL._hf_model.parameters():
        _parameter.requires_grad_(False)
    MODEL._hf_model.eval()

    if LOAD_INFO["model_revision"] != MODEL_REVISION:
        raise RuntimeError(
            f"loaded revision {LOAD_INFO['model_revision']} != pinned "
            f"{MODEL_REVISION}"
        )
    HEAD = head_from_model(MODEL)
    if (HEAD.d_model, HEAD.vocab_size) != (EXPECT_D_MODEL, EXPECT_VOCAB):
        raise RuntimeError(
            f"head is {HEAD.vocab_size} x {HEAD.d_model}, expected "
            f"{EXPECT_VOCAB} x {EXPECT_D_MODEL}"
        )
    HEAD_AUDIT = audit_native_head(HEAD, model=MODEL)

    print(f"readout            {HEAD_AUDIT['readout_expression']}")
    print(f"final norm         {HEAD_AUDIT['final_norm_class']}")
    print(f"norm convention    {HEAD_AUDIT['norm_weight_convention']}")
    print(f"  residuals        {HEAD_AUDIT['norm_convention_residuals']}")
    print(f"softcap            {HEAD_AUDIT['final_logit_softcapping']}")
    print(f"matches unembed    {HEAD_AUDIT['matches_model_unembed']} "
          f"(max |diff| {HEAD_AUDIT['max_abs_difference_vs_model_unembed']})")
    print(f"head checksum      {HEAD_AUDIT['head_checksum']}")
'''
)

markdown(
    """
## 9. Stages 4–7 — score, apply the criterion, combine, and decide

One pass over the stored activations. Every row is written atomically and
checksummed, so a disconnected session resumes rather than restarting, and a
fingerprint that does not match is refused rather than mixed.

The controls run through the same code path with exactly one thing changed
each: shuffled labels, permuted candidate-to-token assignment, and permuted
activations. There is no layer-specific readout component in Gemma to misapply,
so the literal wrong-layer control is not technically meaningful here — the
permuted-activation control is its substitute, and the artifacts say so.

The frozen causal verdicts are **read** from the completed run's
capability-filtered summary. Nothing about the causal result is recomputed.
"""
)

code(
    '''
# 9. Run the audit and write the versioned artifacts.
RESULT = None
AUDIT_STORE = None

if RUN_REAL_CONVERGENCE_AUDIT:
    AUDIT_DIR = Path(AUDIT_ROOT) / AUDIT_NAME
    if str(AUDIT_DIR.resolve()).startswith(str(COMPLETED_RUN.resolve())):
        raise RuntimeError(
            f"the audit directory {AUDIT_DIR} is inside the completed run; "
            "this audit never writes into the run it audits"
        )
    FINGERPRINT = ConvergenceFingerprint(
        protocol=CONVERGENCE_PROTOCOL,
        completed_run_fingerprint_digest=EXPECTED_RUN_FINGERPRINT_DIGEST,
        completed_run_dir=str(COMPLETED_RUN),
        model_repo_id=MODEL_REPO_ID,
        model_revision=MODEL_REVISION,
        processor_revision=EXPECTED_PROCESSOR_REVISION,
        layers=tuple(int(x) for x in LAYERS),
        candidate_digest=TOKENIZATION["digest"],
        readout_mode=TOKENIZATION["readout_mode"],
        head_checksum=str(HEAD_AUDIT["head_checksum"]),
        criterion_digest=CONVERGENCE_CRITERION.digest,
        code_version=COMMIT,
        extra={"combined_lens_checksum": COMBINED_LENS_CHECKSUM},
    )
    AUDIT_STORE = ConvergenceStore(AUDIT_DIR, FINGERPRINT)
    print("audit state:", AUDIT_STORE.open())
    print(f"audit fingerprint {FINGERPRINT.digest}")
    print()

    RESULT = run_convergence_audit(
        population=POPULATION,
        head=HEAD,
        tokenization=TOKENIZATION,
        head_audit=HEAD_AUDIT,
        integrity=INTEGRITY,
        completed_summary=COMPLETED_SUMMARY,
        store=AUDIT_STORE,
        criterion=CONVERGENCE_CRITERION,
        layers=LAYERS,
        run_probe=RUN_SECONDARY_PROBE,
    )
    print(f"rows computed {RESULT['units_computed']}  reused {RESULT['units_reused']}")
'''
)

code(
    '''
# 9b. The layer table and the verdict, printed.
_ACTIVE = RESULT if RESULT is not None else MOCK_RESULT
if _ACTIVE is not None:
    print("=" * 78)
    print("LAYER TABLE — native convergence beside frozen causal evidence")
    print("=" * 78)
    print(
        f"{'layer':<7}{'lens':<7}{'convergence':<16}{'agree':<8}{'rank':<7}"
        f"{'causal':<13}{'concepts'}"
    )
    for _row in _ACTIVE["table"]:
        _agree = _row["clean_agreement_unique"]
        _rank = _row["median_target_rank"]
        print(
            f"L{_row['layer']:<6}"
            f"{'PASS' if _row['lens_validity_gate_passed'] else 'FAIL':<7}"
            f"{str(_row['convergence_classification']):<16}"
            f"{('-' if _agree is None else format(_agree, '.3f')):<8}"
            f"{('-' if _rank is None else format(_rank, '.1f')):<7}"
            f"{str(_row['causal_transfer_verdict']):<13}"
            f"{_row['causal_concepts_supporting'] or '-'}"
        )
    print()
    print("per-modality agreement (a layer converged in one channel only is not")
    print("a converged layer):")
    for _row in _ACTIVE["table"]:
        _cells = "  ".join(
            f"{_m}="
            + (
                "-"
                if _row.get(f"clean_agreement_unique_{_m}") is None
                else format(_row[f"clean_agreement_unique_{_m}"], ".3f")
            )
            for _m in MODALITIES
        )
        print(f"  L{_row['layer']:<4} {_cells}")
    print()
    print("=" * 78)
    print(f"VERDICT: {_ACTIVE['verdict']['verdict']}")
    print("=" * 78)
    print(_ACTIVE["verdict"]["rationale"])
    print()
    for _check in _ACTIVE["verdict"]["checks"]:
        print(f"  {'PASS' if _check['passed'] else 'FAIL'}  {_check['check']}")
    print()
    print(INTERPRETATION_BOUNDARY)
'''
)

code(
    '''
# 9c. Sensitivity at fixed alternative thresholds. The primary rule is unchanged.
if _ACTIVE is not None:
    print("sensitivity — how each layer would be classified under fixed")
    print("alternative thresholds. None of these can change the verdict above.")
    print()
    for _entry in _ACTIVE["sensitivity"]["variants"]:
        _classes = "  ".join(
            f"L{_layer}={_entry['classifications'].get(str(_layer), '-')}"
            for _layer in AUDITED_LAYERS
        )
        print(
            f"  {_entry['variant']:<26s} "
            f"converged>={_entry['converged_min_clean_agreement']:.2f} "
            f"not-converged<={_entry['not_converged_max_clean_agreement']:.2f}   "
            f"{_classes}"
        )
    print()
    print("controls:")
    for _layer_key in sorted(_ACTIVE["controls"]["per_layer"], key=int):
        for _name, _control in sorted(
            _ACTIVE["controls"]["per_layer"][_layer_key]["controls"].items()
        ):
            print(
                f"  L{_layer_key} {_name:<28s} "
                f"{'PASS' if _control['passed'] else 'FAIL'}  {_control['reason']}"
            )
    print()
    print(WRONG_LAYER_CONTROL_NOTE)
'''
)

markdown(
    """
## 10. Immutability of the completed run, and what was written

The protected files are checksummed again and compared with section 6a. A file
that appeared counts as a difference too: this audit writes nothing into the run
it audits.
"""
)

code(
    '''
# 10. Prove the completed run is byte-identical, then report.
if RUN_REAL_CONVERGENCE_AUDIT:
    IMMUTABILITY = assert_run_unchanged(
        CHECKSUMS_BEFORE, protected_file_checksums(COMPLETED_RUN)
    )
    print(f"completed run unchanged: {IMMUTABILITY['unchanged']}")
    for _name in IMMUTABILITY["checked_files"]:
        print(f"  {_name}")
    print()
    _STATUS = AUDIT_STORE.status_report()
    print(f"audit directory  {_STATUS['audit_dir']}")
    print(f"state            {_STATUS['status']}")
    print(f"fingerprint      {_STATUS['fingerprint_digest']}")
    print(f"stored rows      {_STATUS['stored_units']}")
    if _STATUS["invalid_units"]:
        print(f"invalid rows     {len(_STATUS['invalid_units'])} (recomputed)")
    print()
    print("artifacts written:")
    for _path in RESULT["artifacts"]:
        print(f"  {_path}")
elif MOCK_RESULT is not None:
    print(f"MOCK completed run unchanged: {MOCK_RESULT['immutability']['unchanged']}")
    print(f"MOCK artifacts: {len(MOCK_RESULT['artifacts'])} files under")
    print(f"  {MOCK_RESULT['store'].root}")
    print()
    print("=" * 72)
    print("MOCK ONLY — this is the committed default and it spends nothing.")
    print("=" * 72)
    print("To run the real audit against the completed run, set in section 2:")
    print()
    print("    RUN_REAL_CONVERGENCE_AUDIT      = True")
    print("    CONFIRM_MODEL_LOAD              = True")
    print("    EXPECTED_RUN_FINGERPRINT_DIGEST = <from run_manifest.json>")
    print("    EXPECTED_PROCESSOR_REVISION     = <from fingerprint.json>")
    print()
    print("The audit executes zero model forward passes. Gemma is loaded once,")
    print("for its final norm and unembedding only.")
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
