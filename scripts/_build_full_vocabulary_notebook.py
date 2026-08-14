"""Generate the full-vocabulary causal validation Colab notebook.

Deterministic: running this script twice writes byte-identical output, and
``tests/test_full_vocabulary_notebook.py`` fails if the checked-in notebook
drifts from it.

This builder writes a **new** notebook. Every completed study's notebook and
builder are untouched — retroactively rewriting what a completed notebook meant
is exactly what an endpoint correction must not do. The superseded terminology
is marked in ``notebooks/README.md`` and answered by the endpoint audit and the
versioned amendments.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = (
    ROOT
    / "notebooks"
    / "multimodal_jspace_full_vocabulary_causal_validation_colab.ipynb"
)
CELLS: list[tuple[str, str]] = []


def markdown(value: str) -> None:
    CELLS.append(("markdown", value.strip("\n")))


def code(value: str) -> None:
    CELLS.append(("code", value.strip("\n")))


markdown(
    r"""
# The unrestricted next-token endpoint — final causal validation (Gemma 4 E4B)

Every completed behavioral and causal result in this repository was scored by
teacher-forcing a **predeclared candidate set** and taking the argmax over that
set. That is a valid forced-choice preference and a valid conditional
likelihood. It is not the model's output, and it is not the endpoint
Anthropic's systematic evaluation reports.

Theirs:

* run the original prompt with **no answer candidate appended**;
* apply the intervention during that forward pass;
* inspect the **complete next-token distribution** at the final prompt position;
* record the global argmax token;
* record the target token's rank across the **entire** vocabulary;
* count a success only when the target-appropriate token is **global rank 1**.

This notebook implements that endpoint exactly, and re-evaluates two frozen
study families under it.

## What this is, and what it is not

| | |
|---|---|
| Family A | the completed L33-L40 follow-up's design and **its exact population**, rescored on the unrestricted endpoint |
| Family B | the canonical three-modality run's own concepts, media, layers and controls, rescored on the unrestricted endpoint |
| This notebook is | a **measurement-correction rerun on the same populations** |
| This notebook is **not** | an independent replication, a new alpha search, a new concept search, a layer search, a population redraw, or a lens refit |
| Completed reports | read-only, checksum-verified, never modified. `scientific_recompute = 0` |
| L32 | excluded categorically. There is no configuration of this notebook that admits it |

The completed L33-L40 result
(`L33_L40_VALIDATED_BAND_FOLLOWUP_ALPHA2_SENSITIVITY_ONLY`) is **immutable**. It
is relabelled by a versioned amendment as a *restricted-candidate preference*
result, and its numbers are unchanged.

## The three verdict families, which can never overwrite each other

1. **unrestricted output** — the target token is the unique global argmax;
2. **conditional log-probability** — the target answer's likelihood moved
   against its controls;
3. **cross-modal conjunction** — our extension, and only when (1) holds in
   every modality.

α=2 is a prespecified sensitivity condition and is **never** primary evidence.
A restricted-candidate preference, a positive margin without global rank 1, a
greedy completion, a direct-answer-arm success, one modality, or a pooled
direction can none of them produce a full-vocabulary GO.

## The stages

| stage | runtime | what it does |
|---|---|---|
| 0 | **CPU** | The endpoint-semantics audit, the claim ledger, and the versioned amendments. No model, no media. |
| 1 | **GPU** | Verify provenance, reuse the exact populations, resolve the answer tokens, print the pass budget, then score. Blocked until stage 0 passes and both confirmations are set by hand. |
| 2 | **CPU** | Aggregate and judge from the saved units alone. No model, no media. |

Run with every switch `False` and the notebook performs a complete MOCK run on
CPU — no Drive, no model, no download, no spend. **A green MOCK run is evidence
about this code and about nothing else.**
"""
)

markdown("## 0. Colab bootstrap")
code(
    r'''
import json, os, subprocess, sys
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
         "transformers==5.13.1", "accelerate", "soundfile", "datasets"],
        check=True,
    )

os.chdir(REPO_DIR)
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

commit = subprocess.run(
    ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
).stdout.strip()
print("repository", REPO_DIR)
print("branch    ", BRANCH)
print("commit    ", commit)
'''
)

markdown(
    """
## 1. Configuration — three stages, two confirmations

`RUN_ENDPOINT_AUDIT_CPU` and `RUN_FINAL_REPORT_CPU` never load a model.
`RUN_FULL_VOCAB_CAUSAL_GPU` additionally requires `CONFIRM_MODEL_LOAD` and
`CONFIRM_PASS_BUDGET`, both set by hand after reading the printed budget in
section 6.

With all switches `False` the notebook performs a complete MOCK run on CPU.
"""
)
code(
    r'''
# ---- stage switches (all False = full MOCK run on CPU) -------------------
RUN_ENDPOINT_AUDIT_CPU = False      # CPU: audit, claim ledger, amendments. No model.
RUN_FULL_VOCAB_CAUSAL_GPU = False   # GPU: the unrestricted scoring. Needs both confirmations.
RUN_FINAL_REPORT_CPU = False        # CPU: aggregate and judge from saved units. No model.

# ---- explicit confirmations for the GPU stage ---------------------------
CONFIRM_MODEL_LOAD = False          # set True only after reading section 6
CONFIRM_PASS_BUDGET = False         # set True only after reading section 6

REAL_MODE = bool(
    RUN_ENDPOINT_AUDIT_CPU or RUN_FULL_VOCAB_CAUSAL_GPU or RUN_FINAL_REPORT_CPU
)
GPU_STAGE = bool(RUN_FULL_VOCAB_CAUSAL_GPU and CONFIRM_MODEL_LOAD and CONFIRM_PASS_BUDGET)
if RUN_FULL_VOCAB_CAUSAL_GPU and not GPU_STAGE:
    print("NOTE: RUN_FULL_VOCAB_CAUSAL_GPU is set but a confirmation is missing.")
    print("      Sections 7-10 will refuse. Read section 6 first.")

# ---- pinned model identity ----------------------------------------------
MODEL_REPO_ID = "google/gemma-4-E4B-it"
MODEL_REVISION = "fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd"
TRANSFORMERS_VERSION_EXPECTED = "5.13.1"
EXPECT_N_LAYERS, EXPECT_D_MODEL, EXPECT_VOCAB = 42, 2560, 262144
AUDIO_PROTOCOL_FINGERPRINT = (
    "sha256:9ad8bcc9420a7983f6e3b75d5d7080c0e2fcf0a94a76431917fcde73ba777920"
)

# ---- Drive paths (real mode only; never globbed, never "latest") --------
DRIVE_ROOT = Path("/content/drive/MyDrive/jacobian-lens-gemma")
BAND_FOLLOWUP_RUN_DIR = DRIVE_ROOT / "runs" / "mmband33" / "band3340_real_2a72bda9b4ba"
CANONICAL_AUDIO_RUN_DIR = (
    DRIVE_ROOT / "runs" / "mmaudio_native_audio_transfer_20260806T144822"
)
CORRECTED_RUN_DIR = DRIVE_ROOT / "runs" / "mmband" / "bandcorr_real_eb5b00f135e4"
EXPANDED_MANIFEST_CACHE = (
    DRIVE_ROOT / "runs" / "mml32_l32_followup_20260808T182717"
    / "expanded_manifest.json"
)
FULL_VOCAB_RUN_ROOT = DRIVE_ROOT / "runs" / "mmfv"
AUDIT_RUN_ROOT = DRIVE_ROOT / "runs" / "mmfv_audit"

# Completed run directories this notebook must never write into.
PROTECTED_RUN_DIRS = (
    BAND_FOLLOWUP_RUN_DIR, CANONICAL_AUDIO_RUN_DIR, CORRECTED_RUN_DIR,
)

print("mode                        ", "REAL" if REAL_MODE else "MOCK")
print("endpoint audit (CPU)        ", RUN_ENDPOINT_AUDIT_CPU)
print("full-vocab causal (GPU)     ", RUN_FULL_VOCAB_CAUSAL_GPU, " gated:", GPU_STAGE)
print("final report (CPU)          ", RUN_FINAL_REPORT_CPU)
'''
)

markdown(
    """
## 1b. Mount Drive and verify real-mode inputs

Every real stage mounts Drive itself. The mount is idempotent, so reopening a
notebook or reconnecting to a runtime does not require a manual setup cell.
Family A's immutable source report, corrected lens run and expanded manifest
are checked before any model can load or any forward pass can be spent.
"""
)
code(
    r'''
if REAL_MODE and IN_COLAB:
    from google.colab import drive

    drive.mount("/content/drive", force_remount=False)

if REAL_MODE:
    _my_drive = Path("/content/drive/MyDrive")
    if not _my_drive.is_dir():
        raise RuntimeError(
            "Google Drive is not mounted at /content/drive/MyDrive. The notebook "
            "attempted its own mount; reconnect Drive access and rerun this cell."
        )

    _required_real_inputs = {}
    if RUN_ENDPOINT_AUDIT_CPU:
        _required_real_inputs.update({
            "completed L33-L40 follow-up run": BAND_FOLLOWUP_RUN_DIR,
            "canonical three-modality run": CANONICAL_AUDIO_RUN_DIR,
        })
    if GPU_STAGE:
        _required_real_inputs.update({
            "completed L33-L40 follow-up report": (
                BAND_FOLLOWUP_RUN_DIR
                / "l33_l40_validated_band_followup_report.json"
            ),
            "corrected L33-L40 lens run": CORRECTED_RUN_DIR,
            "expanded SpokenCOCO manifest": EXPANDED_MANIFEST_CACHE,
        })

    _missing_real_inputs = {
        name: path
        for name, path in _required_real_inputs.items()
        if not path.exists()
    }
    if _missing_real_inputs:
        raise RuntimeError(
            "Required real-mode inputs are missing after Drive mounted:\n"
            + "\n".join(
                f"  {name}: {path}"
                for name, path in _missing_real_inputs.items()
            )
            + "\nRefusing before model load or scientific spending."
        )

    print("Drive mounted               ", _my_drive)
    for _name, _path in _required_real_inputs.items():
        print(f"  verified {_name}: {_path}")
else:
    print("MOCK mode: Drive is neither mounted nor read")
'''
)

markdown(
    """
## 2. The frozen design

Family A's design is the completed follow-up's, unchanged in every respect
except the endpoint: the same four suffix bands ending at L40, the same two
arms, the same seven conditions, the same three modalities, the same two
readouts, the same directed pairs, the same α roles.

`full_vocab_design_record` refuses a band topology that is not the completed
study's, and refuses any band containing L32.
"""
)
code(
    r'''
from jlens.mmpilot.full_vocab_study import (
    FULL_VOCAB_PROTOCOL_VERSION, FULL_VOCAB_RUN_PREFIX, FULL_VOCAB_STUDY_NAME,
    GREEDY_SUBSET, PASS_CAP, REQUIRED_PINS, FullVocabRefused, FullVocabThresholds,
    full_vocab_design_record,
)
from jlens.mmpilot.full_vocabulary import (
    ENDPOINT_UNRESTRICTED_NEXT_TOKEN, FULL_VOCAB_SCORING_VERSION,
    OUTPUT_HEAD_CONVENTION,
)

THRESHOLDS = FullVocabThresholds()
DESIGN = full_vocab_design_record(thresholds=THRESHOLDS)

print("study                       ", FULL_VOCAB_STUDY_NAME)
print("protocol                    ", FULL_VOCAB_PROTOCOL_VERSION)
print("primary endpoint            ", ENDPOINT_UNRESTRICTED_NEXT_TOKEN)
print("scoring version             ", FULL_VOCAB_SCORING_VERSION)
print("bands                       ", DESIGN["band_keys"])
print("excluded layer              ", DESIGN["excluded_layer"], "(categorically)")
print("arms                        ", DESIGN["arms"])
print("conditions                  ", DESIGN["conditions"])
print("modalities                  ", DESIGN["modalities"])
print("readouts                    ", DESIGN["readouts"], " primary:", DESIGN["primary_readout"])
print("directed pairs              ", DESIGN["directed_pairs"])
print("alpha roles                 ", DESIGN["alpha_roles"])
print("greedy subset               ", GREEDY_SUBSET)
print("design digest               ", DESIGN["design_digest"])
print("threshold digest            ", DESIGN["threshold_digest"])
print()
print("PROVENANCE PINS (a pin that is empty refuses its family):")
for _name, _row in REQUIRED_PINS.items():
    print(f"  {_name:44s} {'SET' if _row['value'] else 'EMPTY'}  family {_row['family']}")
'''
)

markdown(
    """
## 3. Stage 0 — the endpoint-semantics audit (CPU, no model)

Every active scientific module is classified into exactly one endpoint class,
and every row is traced to **the function that computes the field** rather than
to the report's prose. `verify_registry` imports each module and resolves each
function, so the ledger cannot drift from the code.

The scan is the blocking clause: it greps the modules that compute
restricted-candidate or conditional-log-probability endpoints for the four
prohibited descriptions — "full-vocabulary", "global top-1", "the model's
output", "paper-comparable" — and fails the audit if one appears without an
explicit endpoint qualifier. Modules that genuinely measure the full vocabulary
(the J-lens calibration, the native readout, the convergence audit) are entitled
to those words and are out of scope by construction.
"""
)
code(
    r'''
from jlens.mmpilot.endpoint_audit import (
    AUDITED_ENDPOINTS, EndpointAuditFailed, endpoint_audit_files,
    endpoint_audit_record, scanned_modules, verify_registry,
)
from jlens.mmpilot.full_vocab_study import REQUIRED_PINS

RESOLVED = verify_registry()
AUDIT = endpoint_audit_record(
    repo_root=REPO_DIR,
    report_checksums={
        name: row["value"] for name, row in REQUIRED_PINS.items() if row["value"]
    },
)
LEDGER = AUDIT["claim_ledger"]

print("endpoints audited           ", AUDIT["n_endpoints_audited"])
print("functions resolved          ", len(RESOLVED))
print("modules in the blocking scan", len(scanned_modules()))
print("prohibited-phrase hits      ", AUDIT["source_scan"]["n_hits"])
print("unqualified overclaims      ", AUDIT["source_scan"]["n_overclaims"])
print("audit passed                ", AUDIT["passed"])
print("audit digest                ", AUDIT["audit_digest"])
print("claim ledger digest         ", AUDIT["claim_ledger_digest"])
print()
for _name, _count in LEDGER["by_endpoint_class"].items():
    print(f"  {_name:44s} {_count}")
print()
for _name, _count in LEDGER["by_survival"].items():
    print(f"  {_name:44s} {_count}")
print()
print("claims requiring full-vocabulary revalidation:")
for _claim in LEDGER["requiring_revalidation"]:
    print("  -", _claim)

if not AUDIT["passed"]:
    for _row in AUDIT["source_scan"]["overclaims"]:
        print(f"  OVERCLAIM {_row['path']}:{_row['line']} ({_row['pattern']}) {_row['text']}")
    raise EndpointAuditFailed(
        "the active package still describes a restricted-candidate result as an "
        "unrestricted one. Every later stage is blocked until that is corrected."
    )
'''
)

markdown(
    """
### 3b. Write the audit artifacts

`endpoint_semantics_audit.json`, `endpoint_semantics_audit.md` and the
machine-readable claim ledger, into a **new** audit run directory. In MOCK mode
they go to a local scratch directory; nothing is written into any completed run.
"""
)
code(
    r'''
import tempfile

AUDIT_DIR = (
    AUDIT_RUN_ROOT / f"audit_{AUDIT['audit_digest'].split(':')[1][:12]}"
    if RUN_ENDPOINT_AUDIT_CPU
    else Path(tempfile.mkdtemp(prefix="jlens_endpoint_audit_mock_"))
)
for _protected in PROTECTED_RUN_DIRS:
    if _protected == AUDIT_DIR or _protected in AUDIT_DIR.parents:
        raise RuntimeError(
            f"the audit directory {AUDIT_DIR} is inside the completed run "
            f"{_protected}; refusing to write beside an immutable artifact"
        )
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

_LF = chr(10)
for _name, _text in endpoint_audit_files(AUDIT).items():
    # An explicit LF on purpose: a platform-translated newline would make the
    # same audit differ between machines, and the committed artifacts would
    # never match a Colab run's.
    (AUDIT_DIR / _name).write_text(_text, encoding="utf-8", newline=_LF)
print("audit artifacts written to", AUDIT_DIR)
for _name in endpoint_audit_files(AUDIT):
    print("  ", _name)
'''
)

markdown(
    """
### 3c. The versioned amendments

Two amendments, written **beside** the originals in the new audit run
directory. Each binds the original report's path and checksum, the original run
fingerprint, the endpoint-audit digest, the claim-ledger digest and — when the
completed run's units are reachable — a source-unit digest.

`scientific_recompute` is `0`. The original verdict strings are reproduced
verbatim and the corrected labels sit beside them; no historical GO becomes a
new numerical verdict through prose.
"""
)
code(
    r'''
from jlens.mmpilot.endpoint_amend import (
    BAND_FOLLOWUP_TERMINOLOGY, THREE_MODALITY_TERMINOLOGY,
    build_endpoint_amendment, source_unit_digest_from_disk, verify_amendment_binding,
    write_endpoint_amendment,
)
from jlens.mmpilot.full_vocab_study import (
    BAND_FOLLOWUP_FINGERPRINT_PIN, BAND_FOLLOWUP_REPORT_CHECKSUM_PIN,
    BAND_FOLLOWUP_RUN_NAME, CANONICAL_AUDIO_GENERATION_FINGERPRINT_PIN,
    CANONICAL_AUDIO_REPORT_CHECKSUM_PIN, CANONICAL_AUDIO_RUN_NAME,
    CONTROLLED_TARGET_LOGPROB_EFFECT, FULL_VOCAB_NOT_EVALUATED,
    RESTRICTED_CANDIDATE_PREFERENCE_GO,
)
from jlens.mmpilot.full_vocabulary import (
    ENDPOINT_CONDITIONAL_LOGPROB, ENDPOINT_RESTRICTED_CANDIDATE,
)

AMENDMENTS = {}

_band_units = (
    source_unit_digest_from_disk(BAND_FOLLOWUP_RUN_DIR)
    if RUN_ENDPOINT_AUDIT_CPU
    else None
)
AMENDMENTS["l33_l40_validated_band_followup_report"] = build_endpoint_amendment(
    name="l33_l40_validated_band_followup_report",
    study="L33-L40 validated-band follow-up",
    original_report_path=str(
        BAND_FOLLOWUP_RUN_DIR / "l33_l40_validated_band_followup_report.json"
    ),
    original_report_checksum=BAND_FOLLOWUP_REPORT_CHECKSUM_PIN,
    original_run_name=BAND_FOLLOWUP_RUN_NAME,
    original_run_fingerprint=BAND_FOLLOWUP_FINGERPRINT_PIN,
    original_verdict="L33_L40_VALIDATED_BAND_FOLLOWUP_ALPHA2_SENSITIVITY_ONLY",
    original_endpoint_class=ENDPOINT_RESTRICTED_CANDIDATE,
    endpoint_audit_digest=AUDIT["audit_digest"],
    claim_ledger_digest=AUDIT["claim_ledger_digest"],
    terminology=BAND_FOLLOWUP_TERMINOLOGY,
    corrected_labels=[RESTRICTED_CANDIDATE_PREFERENCE_GO, FULL_VOCAB_NOT_EVALUATED],
    source_unit_digest=_band_units,
    written_utc="2026-08-13T00:00:00+00:00" if not REAL_MODE else None,
)

AMENDMENTS["native_audio_transfer_three_modality_verdict"] = build_endpoint_amendment(
    name="native_audio_transfer_three_modality_verdict",
    study="native spoken-audio transfer (canonical three-modality run)",
    original_report_path=str(
        CANONICAL_AUDIO_RUN_DIR / "native_audio_transfer_summary.json"
    ),
    # The canonical run's *report* checksum pin is deliberately empty, so the
    # amendment binds to the raw-generation fingerprint, which is recorded in
    # docs/three_modality_claim_admissibility.md. The amendment says which.
    original_report_checksum=(
        CANONICAL_AUDIO_REPORT_CHECKSUM_PIN
        or CANONICAL_AUDIO_GENERATION_FINGERPRINT_PIN
    ),
    original_run_name=CANONICAL_AUDIO_RUN_NAME,
    original_run_fingerprint=CANONICAL_AUDIO_GENERATION_FINGERPRINT_PIN,
    original_verdict="THREE_MODALITY_GO",
    original_endpoint_class=ENDPOINT_CONDITIONAL_LOGPROB,
    endpoint_audit_digest=AUDIT["audit_digest"],
    claim_ledger_digest=AUDIT["claim_ledger_digest"],
    terminology=THREE_MODALITY_TERMINOLOGY,
    corrected_labels=[CONTROLLED_TARGET_LOGPROB_EFFECT, FULL_VOCAB_NOT_EVALUATED],
    source_unit_digest=(
        source_unit_digest_from_disk(
            CANONICAL_AUDIO_RUN_DIR, stages=("capability", "intervention")
        )
        if RUN_ENDPOINT_AUDIT_CPU
        else None
    ),
    written_utc="2026-08-13T00:00:00+00:00" if not REAL_MODE else None,
)

for _name, _amendment in AMENDMENTS.items():
    verify_amendment_binding(
        _amendment,
        original_report_checksum=_amendment["original_report_checksum"],
        original_run_fingerprint=_amendment["original_run_fingerprint"],
        endpoint_audit_digest=AUDIT["audit_digest"],
    )
    _placed = write_endpoint_amendment(AUDIT_DIR, _amendment)
    print(f"{_placed['status']:8s} {_name}")
    print(f"         original verdict, verbatim: {_amendment['original_verdict']}")
    print(f"         corrected labels:           {_amendment['corrected_labels']}")
    print(f"         scientific_recompute:       {_amendment['scientific_recompute']}")
    print(f"         terminology changes:        {_amendment['n_terminology_changes']}")
'''
)

markdown(
    """
## 4. Provenance — configured, never discovered (real mode only)

The completed artifacts are named by explicit paths and verified against the
pins in `jlens.mmpilot.full_vocab_study`. There is no globbing and no
newest-first sort anywhere in this section.

`CANONICAL_AUDIO_REPORT_CHECKSUM` is deliberately empty in the repository: the
canonical run's report checksum is not recorded anywhere here, and the one
number this study may not accept is a checksum printed by the run it is
verifying. **Family B refuses here** until an operator writes the out-of-band
value into that pin. That refusal is the specified behaviour.
"""
)
code(
    r'''
from jlens.mmpilot.full_vocab_study import (
    read_band_followup_report, read_canonical_audio_provenance,
    read_historical_prompt_hashes, reuse_completed_population,
)

BAND_REPORT = None
BAND_REPORT_PATH = None
POPULATION_REUSE = None
HISTORICAL_PROMPTS = None
AUDIO_PROVENANCE = None
FAMILY_B_REFUSAL = None

if GPU_STAGE:
    BAND_REPORT_PATH, BAND_REPORT = read_band_followup_report(BAND_FOLLOWUP_RUN_DIR)
    print("family A source report      ", BAND_REPORT_PATH)
    print("  checksum                  ", BAND_REPORT["report_checksum"])
    print("  original verdict          ", BAND_REPORT["followup_verdict"]["verdict"])

    POPULATION_REUSE = reuse_completed_population(BAND_REPORT)
    print("  population groups reused  ", POPULATION_REUSE["n_groups"])
    print("  distinct images           ", POPULATION_REUSE["n_distinct_images"])
    print("  redrawn / enlarged        ",
          POPULATION_REUSE["redrawn"], "/", POPULATION_REUSE["enlarged"])
    print("  reuse digest              ", POPULATION_REUSE["population_reuse_digest"])

    HISTORICAL_PROMPTS = read_historical_prompt_hashes(BAND_FOLLOWUP_RUN_DIR)
    print("  historical prompt hashes  ", HISTORICAL_PROMPTS["n_prompt_hashes"])

    try:
        AUDIO_PROVENANCE = read_canonical_audio_provenance(CANONICAL_AUDIO_RUN_DIR)
        print("family B source run        ", AUDIO_PROVENANCE["run_dir"])
    except FullVocabRefused as error:
        FAMILY_B_REFUSAL = str(error)
        print()
        print("FAMILY B REFUSED (this is the specified behaviour, not a failure):")
        print(" ", FAMILY_B_REFUSAL)
        print()
        print("  Family B will be reported as NOT_EVALUATED. Family A is unaffected.")
else:
    print("skipped: the GPU stage is not gated on")
'''
)

markdown(
    """
## 5. Answer tokens — resolved at runtime, before any model weight loads

The unrestricted endpoint is a statement about **one** vocabulary row. `two`,
`four`, `bird` and `cat` must each encode as exactly one next token in this
prompt context; if any of them does not, the reasoning experiment is refused
here, before model spending.

Family B's concepts are *not* required to be single tokens — `microwave` may
well be several in this vocabulary. A multi-token concept is recorded as
unsupported for the unrestricted endpoint and reported separately as a
sequence-likelihood diagnostic. It is never truncated to its first token, and it
is never replaced by a different concept.

Loading a tokenizer is not loading Gemma. This cell runs before the model does.
"""
)
code(
    r'''
from jlens.mmpilot.full_vocab_study import (
    REQUIRED_SINGLE_TOKEN_ANSWERS, resolve_study_tokens,
)

TOKENS = None
PROCESSOR_BUNDLE = None

if GPU_STAGE:
    import getpass
    from jlens.mmpilot.real_backend import build_processor_backend

    if not os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = getpass.getpass("HF_TOKEN (input hidden): ").strip()
    # This API has no model-loading entry point: processor and tokenizer only.
    PROCESSOR_BUNDLE = build_processor_backend(
        MODEL_REPO_ID, revision=MODEL_REVISION, token=os.environ["HF_TOKEN"],
    )
    _tokenizer_backend = PROCESSOR_BUNDLE.backend

    _three_modality_concepts = (
        ("bird", "cat", "giraffe", "microwave", "toilet", "zebra")
        if AUDIO_PROVENANCE else ()
    )
    TOKENS = resolve_study_tokens(
        _tokenizer_backend,
        three_modality_concepts=_three_modality_concepts,
    )
    print("family A answers single-token:", TOKENS["family_a_supported"])
    for _name, _row in sorted(TOKENS["reasoning"]["supported"].items()):
        print(f"  {_name:10s} -> token {_row['token_id']:7d}  surface {_row['surface']!r}")
    if TOKENS["family_b_unsupported_for_unrestricted_endpoint"]:
        print()
        print("family B concepts UNSUPPORTED for the unrestricted endpoint")
        print("(reported separately as sequence-likelihood diagnostics, never truncated):")
        for _name in TOKENS["family_b_unsupported_for_unrestricted_endpoint"]:
            print("  -", _name, TOKENS["three_modality"]["unsupported"][_name]["token_ids"])
    print("token digest                ", TOKENS["token_digest"])
else:
    print("skipped: the GPU stage is not gated on")
'''
)

markdown(
    """
## 6. The pass budget — printed before any model weight loads

Every count is derived from the reused population and the frozen design. The
unrestricted endpoint costs **one forward pass per trial**, not one per
candidate, which is why a faithful rerun of the completed 10,752-candidate-pass
design fits under the 5,000-pass cap.

If the derived total exceeds the cap, this cell stops the notebook, names the
factor causing the excess, and prints the smallest lossless reduction. No
condition is ever silently dropped.

Read the block, then set `CONFIRM_PASS_BUDGET = True` and
`CONFIRM_MODEL_LOAD = True` in section 1 and re-run from there.
"""
)
code(
    r'''
from jlens.mmpilot.full_vocab_study import (
    family_a_trials, format_pass_budget, full_vocab_pass_budget,
)
from jlens.mmpilot.full_vocab_mock import MOCK_N_IMAGES, mock_population_reuse

_reuse = POPULATION_REUSE if POPULATION_REUSE else mock_population_reuse()
TRIAL_PLAN = family_a_trials(_reuse, DESIGN)
_clean_trials = [row for row in TRIAL_PLAN if row["kind"] == "clean"]
_intervention_trials = [row for row in TRIAL_PLAN if row["kind"] == "intervention"]
_a_cells = len(_clean_trials) // len(DESIGN["readouts"])

# The greedy subset is predeclared in GREEDY_SUBSET and costed here.
GREEDY_TRIALS = [
    row for row in TRIAL_PLAN
    if row["modality"] == GREEDY_SUBSET["modality"]
    and row["readout"] == GREEDY_SUBSET["readout"]
    and (
        row["condition"] in GREEDY_SUBSET["conditions"]
        and (row["band"] is None or row["band"] == GREEDY_SUBSET["band"])
        and (row["arm"] in (None, GREEDY_SUBSET["arm"]))
    )
]

# Family B's factors come from the canonical run's own artifacts, or are zero.
if AUDIO_PROVENANCE:
    _amended = AUDIO_PROVENANCE["amended_summary"]
    _claim_cells = list(
        ((_amended.get("primary_causal") or {}).get("audio_cells_supporting_a_claim"))
        or ()
    )
    _b_cells = len(_claim_cells)
    _b_samples = int((_amended.get("thresholds") or {}).get("min_images_per_cell", 8))
    _b_layers = 1 + len(list((_amended.get("replication") or {}).get("layers") or ()))
    _b_conditions = 5   # source_concept + zero + random + unrelated + raw_residual
    _b_clean = _b_cells * _b_samples
else:
    _claim_cells, _b_cells, _b_samples, _b_layers, _b_conditions, _b_clean = [], 0, 0, 0, 0, 0

BUDGET = full_vocab_pass_budget(
    a_cells=_a_cells,
    a_bands=len(DESIGN["bands"]),
    a_arms=len(DESIGN["arms"]),
    a_conditions=len(DESIGN["conditions"]),
    a_readouts=len(DESIGN["readouts"]),
    b_claim_supporting_cells=_b_cells,
    b_samples_per_cell=_b_samples,
    b_layers=_b_layers,
    b_conditions=_b_conditions,
    b_clean_inputs=_b_clean,
    greedy_trials=len(GREEDY_TRIALS),
    greedy_max_new_tokens=int(GREEDY_SUBSET["max_new_tokens"]),
    cap=PASS_CAP,
)
print(format_pass_budget(BUDGET))
print()
print("planned trials              ",
      {"clean": len(_clean_trials), "intervention": len(_intervention_trials),
       "greedy": len(GREEDY_TRIALS)})
if not BUDGET["within_cap"]:
    raise FullVocabRefused(
        "the derived pass budget exceeds the cap. Stopping before any model "
        "loads. The driving factor and the smallest lossless reduction are "
        "printed above; choose one explicitly rather than dropping a condition."
    )
'''
)

markdown(
    """
## 7. Stage 1 — load the model and rebuild the historical inputs (GPU)

The population is the completed run's, by group id. The captions the text
condition needs are not in the report, so each prompt is rebuilt from the pinned
manifest and then **proven** to be the historical one:
`assert_prompt_reconstruction` compares the rebuilt prompt hash against the hash
the completed run recorded for that `(group, modality, readout)` and refuses on
any difference. A rerun that cannot prove it is scoring the same prompt is not a
measurement correction.

The J-lens vectors come from the same corrected artifacts the completed
follow-up used, one per band layer, re-checksummed on load.
"""
)
code(
    r'''
BACKEND = None
STORE = None
RUN_DIR = None
FINGERPRINT_CONFIG = None
LENS_CHECKSUMS = {}
MEDIA_CHECKSUMS = {}
HOOK_INTEGRITY = None
RECONSTRUCTION_PROOFS = []

if GPU_STAGE:
    import hashlib
    import torch
    from jlens.lens import JacobianLens
    from jlens.mmpilot.coordinate_swap import (
        METHOD_VERSION, build_swap_basis_from_vectors, random_two_direction_basis,
        resolve_concept_token,
    )
    from jlens.mmpilot.full_vocab_study import (
        FULL_VOCAB_UNIT_FAMILY, assert_prompt_reconstruction, full_vocab_fingerprint,
    )
    from jlens.mmpilot.media_io import RetryJournal, drive_media_loaders
    from jlens.mmpilot.prompt_protocol import (
        Evidence, HIDDEN_ANIMAL_LEGS, OPEN_ANIMAL_IDENTIFICATION, build_backend_inputs,
        build_protocol_prompt, concept_spec, prompt_protocol_fingerprint,
    )
    from jlens.mmpilot.real_backend import build_real_backend
    from jlens.mmpilot.store import RunFingerprint, UnitStore
    from jlens.mmpilot.tri_modal import assert_audio_protocol
    from jlens.mmpilot.validated_band_followup import (
        discover_corrected_band_lenses, read_corrected_validation_report,
    )

    _existing_bundle = globals().get("_bundle")
    if _existing_bundle is None:
        _bundle = build_real_backend(
            MODEL_REPO_ID, revision=MODEL_REVISION, token=os.environ["HF_TOKEN"],
            device="cuda", allow_model_load=True, resolve_audio=True,
            expect_n_layers=EXPECT_N_LAYERS, expect_d_model=EXPECT_D_MODEL,
            expect_vocab_size=EXPECT_VOCAB,
        )
        print("model bundle                loaded")
    else:
        _architecture = dict(_existing_bundle.architecture)
        _expected = {
            "n_layers": EXPECT_N_LAYERS,
            "d_model": EXPECT_D_MODEL,
            "vocab_size": EXPECT_VOCAB,
        }
        _observed = {key: _architecture.get(key) for key in _expected}
        if (
            _existing_bundle.model_revision != MODEL_REVISION
            or _observed != _expected
        ):
            raise RuntimeError(
                "an already-loaded model bundle has different identity or "
                f"architecture: revision={_existing_bundle.model_revision!r}, "
                f"architecture={_observed!r}; refusing to reuse it"
            )
        _bundle = _existing_bundle
        print("model bundle                reused from this runtime")
    if _bundle.audio_interface is None:
        raise RuntimeError(
            "native spoken audio did not resolve: " + _bundle.audio_blocked_reason
        )
    AUDIO_RECORD = assert_audio_protocol(
        _bundle.audio_interface,
        expected_fingerprint=AUDIO_PROTOCOL_FINGERPRINT,
    )
    BACKEND = _bundle.backend

    # --- the same validated L33-L40 artifacts the completed follow-up used
    _corrected_path, _corrected = read_corrected_validation_report(
        CORRECTED_RUN_DIR,
        expected_model_repo_id=MODEL_REPO_ID,
        expected_model_revision=_bundle.model_revision,
    )
    CORRECTED_ARTIFACTS, ARTIFACT_DISCOVERY = discover_corrected_band_lenses(
        CORRECTED_RUN_DIR,
        report=_corrected,
    )
    LENS_CHECKSUMS = {
        int(layer): artifact.lens_checksum
        for layer, artifact in CORRECTED_ARTIFACTS.items()
    }
    print("validated band artifacts   ", sorted(LENS_CHECKSUMS))

    _names = ("bird", "cat", "two", "four", "zebra", "giraffe")
    CONCEPT_TOKENS = {
        name: resolve_concept_token(BACKEND.encode_candidate, name) for name in _names
    }
    _unembedding = BACKEND.unembedding_weight()
    _rows = {
        name: _unembedding[token.token_id].detach().float().cpu()
        for name, token in CONCEPT_TOKENS.items()
    }
    TOKEN_VECTORS, _files = {}, {}
    for _layer in sorted({layer for band in DESIGN["bands"] for layer in band}):
        _source = CORRECTED_ARTIFACTS[_layer]
        if _source.lens_path not in _files:
            _files[_source.lens_path] = JacobianLens.load(_source.lens_path)
        _jacobian = _files[_source.lens_path].jacobians[
            _source.layer_key_in_file
        ].detach().float().cpu()
        TOKEN_VECTORS[_layer] = {name: row @ _jacobian for name, row in _rows.items()}
        del _jacobian
    del _files, _unembedding, _rows

    def selected_bases(layers, source_name, target_name):
        return {
            layer: build_swap_basis_from_vectors(
                TOKEN_VECTORS[layer][source_name], TOKEN_VECTORS[layer][target_name],
                layer=layer, source=CONCEPT_TOKENS[source_name],
                target=CONCEPT_TOKENS[target_name],
            )
            for layer in layers
        }

    # --- the historical captions, by group id, from the pinned manifest
    _raw = EXPANDED_MANIFEST_CACHE.read_bytes()
    MANIFEST_FILE_CHECKSUM = "sha256:" + hashlib.sha256(_raw).hexdigest()
    _by_group = {
        str(row["group_id"]): row for row in json.loads(_raw)["groups"]
    }
    del _raw
    GROUPS = {}
    for _row in POPULATION_REUSE["groups"]:
        _gid = str(_row["group_id"])
        if _gid not in _by_group:
            raise FullVocabRefused(
                f"group {_gid} is in the completed population but not in the "
                "pinned manifest; the historical input cannot be reconstructed"
            )
        GROUPS[_gid] = {**_by_group[_gid], **_row}
    print("historical groups rebuilt  ", len(GROUPS))

    MEDIA = drive_media_loaders(journal=RetryJournal())
    IDENTITY_ANSWERS = ("bird", "cat")
    PROPERTY_ANSWERS = ("two", "four")
    ANSWER_TOKEN_IDS = dict(TOKENS["reasoning"]["token_ids"])

    def load_evidence(group, modality):
        if modality == "text":
            return Evidence(modality="text", text=group["caption"])
        if modality == "image":
            return Evidence(
                modality="image", media=MEDIA["load_image"](group["image_path"]),
                media_reference=group["image_path"],
            )
        waveform, rate = MEDIA["load_audio"](group["audio_path"])
        return Evidence(
            modality="spoken_audio", media=waveform, sampling_rate=rate,
            media_reference=group["audio_path"], transcript=group["caption"],
        )

    def make_input(group, modality, source, target, readout, evidence=None):
        evidence = evidence or load_evidence(group, modality)
        protocol = (
            OPEN_ANIMAL_IDENTIFICATION if readout == "identity" else HIDDEN_ANIMAL_LEGS
        )
        answers = IDENTITY_ANSWERS if readout == "identity" else PROPERTY_ANSWERS
        built = build_protocol_prompt(
            protocol=protocol, evidence=evidence, external_candidates=answers,
            source=concept_spec(source), target=concept_spec(target),
            encode_candidate=BACKEND.encode_candidate,
        )
        offline = group["caption"] if modality != "text" else None
        return built, build_backend_inputs(BACKEND, built, transcript=offline)

    FINGERPRINT_CONFIG = full_vocab_fingerprint(
        design=DESIGN,
        endpoint_audit_digest=AUDIT["audit_digest"],
        band_followup_report_checksum=BAND_REPORT["report_checksum"],
        band_followup_fingerprint=BAND_FOLLOWUP_FINGERPRINT_PIN,
        canonical_audio_provenance=AUDIO_PROVENANCE,
        population_reuse=POPULATION_REUSE,
        lens_checksums=LENS_CHECKSUMS,
        media_checksums=None,
        model_repo_id=MODEL_REPO_ID,
        model_revision=_bundle.model_revision,
        processor_revision=_bundle.processor_revision,
        transformers_version=TRANSFORMERS_VERSION_EXPECTED,
        audio_protocol_fingerprint=AUDIO_PROTOCOL_FINGERPRINT,
        prompt_protocol=None,
        tokens=TOKENS,
        coordinate_swap_method_version=METHOD_VERSION,
        thresholds=THRESHOLDS.to_dict(),
        seeds={"random_control_seed_intermediate": 20260813,
               "random_control_seed_answer": 20261813},
        output_head_convention=OUTPUT_HEAD_CONVENTION,
    )
    _digest = FINGERPRINT_CONFIG["full_vocab_fingerprint_digest"]
    RUN_DIR = FULL_VOCAB_RUN_ROOT / f"{FULL_VOCAB_RUN_PREFIX}_real_{_digest.split(':')[1][:12]}"
    for _protected in PROTECTED_RUN_DIRS:
        if _protected == RUN_DIR or _protected in RUN_DIR.parents:
            raise RuntimeError(
                f"the new run directory {RUN_DIR} is inside the completed run "
                f"{_protected}; refusing to write beside an immutable artifact"
            )
    STORE = UnitStore(
        RUN_DIR,
        RunFingerprint(
            mode=FULL_VOCAB_UNIT_FAMILY,
            model_repo_id=MODEL_REPO_ID,
            model_revision=_bundle.model_revision,
            processor_revision=_bundle.processor_revision,
            layers=tuple(sorted({l for band in DESIGN["bands"] for l in band})),
            lens_checksum=_digest,
            manifest_checksum=MANIFEST_FILE_CHECKSUM,
            split_id=POPULATION_REUSE["population_reuse_digest"],
            intervention_config=FINGERPRINT_CONFIG,
            selection_config={
                "population_reuse_digest": POPULATION_REUSE["population_reuse_digest"],
                "group_ids": POPULATION_REUSE["group_ids"],
            },
            extra={
                "study": FULL_VOCAB_STUDY_NAME,
                "endpoint": ENDPOINT_UNRESTRICTED_NEXT_TOKEN,
                "endpoint_audit_digest": AUDIT["audit_digest"],
                "completed_runs_read": "read-only",
            },
        ),
    )
    print("run directory              ", RUN_DIR)
    print("store                      ", STORE.open())
    print("fingerprint digest         ", _digest)
else:
    print("skipped: the GPU stage is not gated on")
'''
)

markdown(
    """
## 8. Clean unrestricted scoring, then every intervention and control

One forward pass per trial. `score_unrestricted_next_token` receives **no
candidate list** — the target and source token ids go in only to read ranks back
out of a distribution that already exists — and asserts that nothing was
appended to `input_ids`.

Each trial is saved as its own atomically written, checksum-valid unit, so a
disconnect loses at most the forward pass currently executing. Computed and
reused counts print every 25 trials.

The restricted-candidate scores are recorded beside the primary endpoint as a
clearly labelled secondary diagnostic, so the two endpoints can be compared on
the same population.
"""
)
code(
    r'''
CLEAN_UNITS = {}
FV_RECORDS = []

if GPU_STAGE:
    from jlens.mmpilot.capability import score_candidate_sequences
    from jlens.mmpilot.coordinate_swap import coordinate_swap_band
    from jlens.mmpilot.full_vocab_study import FULL_VOCAB_UNIT_STORE_STAGE
    from jlens.mmpilot.full_vocabulary import (
        restricted_candidate_top1, score_unrestricted_next_token,
        unrestricted_trial_record,
    )
    from jlens.mmpilot.validated_band_followup import assert_band_hook_integrity

    _decode = BACKEND.decode_token
    _computed = _reused = 0

    def _answers_for(readout, source, target):
        if readout == "identity":
            return source, target
        return ("two" if source == "bird" else "four",
                "two" if target == "bird" else "four")

    # ---- clean, then intervened. The clean unit for an input must exist
    #      before its edits are scored.
    _inputs_cache = {}

    def _input_for(group_id, modality, readout, source, target):
        key = (group_id, modality, readout)
        if key not in _inputs_cache:
            built, inputs = make_input(
                GROUPS[group_id], modality, source, target, readout
            )
            RECONSTRUCTION_PROOFS.append(
                assert_prompt_reconstruction(
                    group_id=group_id, modality=modality, readout=readout,
                    rebuilt_prompt_hash=built.prompt_hash,
                    historical=HISTORICAL_PROMPTS,
                )
            )
            _inputs_cache[key] = (built, inputs)
        return _inputs_cache[key]

    for _trial in TRIAL_PLAN:
        _stored = STORE.load(FULL_VOCAB_UNIT_STORE_STAGE, _trial["key"])
        if _stored is not None:
            if _trial["kind"] == "clean":
                CLEAN_UNITS[
                    (_trial["group_id"], _trial["modality"], _trial["readout"])
                ] = _stored
            _reused += 1
            continue
        _source, _target = _trial["source"], _trial["target"]
        _src_answer, _tgt_answer = _answers_for(_trial["readout"], _source, _target)
        _built, _inputs = _input_for(
            _trial["group_id"], _trial["modality"], _trial["readout"], _source, _target
        )
        _named = {
            "target": ANSWER_TOKEN_IDS[_tgt_answer],
            "source": ANSWER_TOKEN_IDS[_src_answer],
        }
        _answer_ids = {
            name: BACKEND.encode_candidate(f" {name}")
            for name in (_src_answer, _tgt_answer)
        }

        if _trial["kind"] == "clean":
            _scored = score_unrestricted_next_token(
                BACKEND, _inputs, target_token_ids=_named, top_k=10,
                decode=_decode, expected_vocab_size=EXPECT_VOCAB,
            )
            _restricted = restricted_candidate_top1(
                score_candidate_sequences(BACKEND, _inputs, _answer_ids), _tgt_answer
            )
            _record = unrestricted_trial_record(
                _scored, trial_kind="clean", condition="clean",
                modality=_trial["modality"], readout=_trial["readout"],
                source_answer=_src_answer, target_answer=_tgt_answer,
                source_token_id=_named["source"], target_token_id=_named["target"],
                source_concept=_source, target_concept=_target,
                group_id=_trial["group_id"], image_id=_trial["image_id"],
                prompt_hash=_built.prompt_hash,
                media_checksum=_inputs.media_checksum,
                restricted=_restricted,
                model_pins={"repo_id": MODEL_REPO_ID,
                            "revision": _bundle.model_revision,
                            "vocab_size": EXPECT_VOCAB},
            )
            CLEAN_UNITS[(_trial["group_id"], _trial["modality"], _trial["readout"])] = _record
            STORE.save(FULL_VOCAB_UNIT_STORE_STAGE, _trial["key"], _record)
            _computed += 1
        else:
            _layers = tuple(_trial["band"])
            _arm = _trial["arm"]
            _condition = _trial["condition"]
            _banks = {
                "intermediate": selected_bases(_layers, _source, _target),
                "answer": selected_bases(
                    _layers,
                    "two" if _source == "bird" else "four",
                    "two" if _target == "bird" else "four",
                ),
                "unrelated": selected_bases(_layers, "zebra", "giraffe"),
            }
            _base_arm = _banks[_arm]
            if _condition.startswith("unrelated_"):
                _bases = _banks["unrelated"]
            elif _condition.startswith("random_"):
                _bases = {
                    layer: random_two_direction_basis(
                        basis, seed=20260813 + layer + _layers[0]
                        + (0 if _arm == "intermediate" else 1000)
                    )
                    for layer, basis in _base_arm.items()
                }
            else:
                _bases = _base_arm

            with coordinate_swap_band(
                BACKEND.blocks, _bases, alpha=float(_trial["alpha"]),
                prompt_len=_inputs.prompt_len,
                position_rule=DESIGN.get("position_rule", "all_prompt_positions"),
                evidence_span=_inputs.modality_token_range,
                record_coordinates=False,
            ) as _stats:
                _scored = score_unrestricted_next_token(
                    BACKEND, _inputs, target_token_ids=_named, top_k=10,
                    decode=_decode, expected_vocab_size=EXPECT_VOCAB,
                )
                _restricted_scores = score_candidate_sequences(
                    BACKEND, _inputs, _answer_ids
                )
            _patched = sorted(
                int(layer) for layer, row in _stats.items() if row["n_forward_passes"]
            )
            HOOK_INTEGRITY = assert_band_hook_integrity(
                {
                    "layers_patched": _patched,
                    "n_positions_patched": _stats[_patched[0]]["n_positions"],
                    "n_candidate_positions_skipped":
                        _stats[_patched[0]]["n_candidate_positions_skipped"],
                    "layer_stats": {str(k): _stats[k] for k in sorted(_stats)},
                },
                band=_layers, prompt_len=_inputs.prompt_len,
                expected_forward_passes=1 + len(_answer_ids),
            )
            _clean = CLEAN_UNITS[
                (_trial["group_id"], _trial["modality"], _trial["readout"])
            ]
            _record = unrestricted_trial_record(
                _scored, trial_kind="intervention", condition=_condition,
                arm=_arm, band=list(_layers), alpha=float(_trial["alpha"]),
                modality=_trial["modality"], readout=_trial["readout"],
                source_answer=_src_answer, target_answer=_tgt_answer,
                source_token_id=_named["source"], target_token_id=_named["target"],
                source_concept=_source, target_concept=_target,
                group_id=_trial["group_id"], image_id=_trial["image_id"],
                prompt_hash=_built.prompt_hash,
                media_checksum=_inputs.media_checksum,
                hook_integrity=HOOK_INTEGRITY,
                restricted=restricted_candidate_top1(_restricted_scores, _tgt_answer),
                clean=_clean,
                model_pins={"repo_id": MODEL_REPO_ID,
                            "revision": _bundle.model_revision,
                            "vocab_size": EXPECT_VOCAB},
            )
            _record["band_key"] = _trial["band_key"]
            STORE.save(FULL_VOCAB_UNIT_STORE_STAGE, _trial["key"], _record)
            _computed += 1
        if _computed % 25 == 0 or _computed == 1:
            print(f"trials {_computed:,} computed  {_reused:,} reused")

    print("scoring complete", {"computed": _computed, "reused": _reused})
    print("prompt reconstructions proven", len(RECONSTRUCTION_PROOFS))
    FV_RECORDS = [
        row for row in STORE.load_all(FULL_VOCAB_UNIT_STORE_STAGE).values()
        if row.get("status") == "complete"
    ]
else:
    print("skipped: the GPU stage is not gated on")
'''
)

markdown(
    """
## 9. The predeclared greedy demonstration (secondary)

Deterministic, temperature 0, a fixed small budget, over the subset frozen in
`GREEDY_SUBSET` and costed in section 6. It shows what the model writes. It
never overrides, replaces or stands in for the full-vocabulary next-token rank,
and no verdict is derived from it.
"""
)
code(
    r'''
GREEDY_RECORDS = []

if GPU_STAGE:
    from jlens.mmpilot.coordinate_swap import coordinate_swap_band
    from jlens.mmpilot.full_vocabulary import greedy_generate

    for _trial in GREEDY_TRIALS:
        _key = _trial["key"] + "--greedy"
        if STORE.has("metric", _key):
            GREEDY_RECORDS.append(STORE.load("metric", _key))
            continue
        _source, _target = _trial["source"], _trial["target"]
        _tgt_answer = "two" if _target == "bird" else "four"
        _built, _inputs = _input_for(
            _trial["group_id"], _trial["modality"], _trial["readout"], _source, _target
        )
        if _trial["kind"] == "clean":
            _record = greedy_generate(
                BACKEND, _inputs, max_new_tokens=int(GREEDY_SUBSET["max_new_tokens"]),
                decode=BACKEND.decode_token, answer=_tgt_answer,
            )
        else:
            _bases = selected_bases(tuple(_trial["band"]), _source, _target)
            with coordinate_swap_band(
                BACKEND.blocks, _bases, alpha=float(_trial["alpha"]),
                prompt_len=_inputs.prompt_len,
                evidence_span=_inputs.modality_token_range,
            ):
                _record = greedy_generate(
                    BACKEND, _inputs,
                    max_new_tokens=int(GREEDY_SUBSET["max_new_tokens"]),
                    decode=BACKEND.decode_token, answer=_tgt_answer,
                )
        _record.update({
            "group_id": _trial["group_id"], "modality": _trial["modality"],
            "readout": _trial["readout"], "condition": _trial["condition"],
            "band": _trial["band"], "arm": _trial["arm"],
        })
        STORE.save("metric", _key, _record)
        GREEDY_RECORDS.append(_record)
    print("greedy demonstrations      ", len(GREEDY_RECORDS))
    print("  (secondary; no verdict is derived from these)")
else:
    print("skipped: the GPU stage is not gated on")
'''
)

markdown(
    """
## 10. Family B — the canonical three-modality causal endpoint

Read the completed run explicitly, reuse its exact concepts, media, layers,
directions, α values, controls and prompts, and re-score each historically
claim-supporting cell under the unrestricted endpoint. Both the full-vocabulary
rank and the historical conditional-log-probability metric are recorded for
every cell; neither substitutes for the other.

When the provenance pin is empty or an input cannot be reconstructed from an
immutable artifact, this family is **refused explicitly** and reported as
`NOT_EVALUATED`. It is never approximated.
"""
)
code(
    r'''
FAMILY_B_RECORDS = []
FAMILY_B_EVALUATED = False

if GPU_STAGE and AUDIO_PROVENANCE:
    raise NotImplementedError(
        "Family B's execution path is reached only once "
        "CANONICAL_AUDIO_REPORT_CHECKSUM is pinned and the canonical run is "
        "mounted. Until then read_canonical_audio_provenance refuses above and "
        "this cell is not entered, which is the specified behaviour."
    )
elif GPU_STAGE:
    print("family B: NOT EVALUATED")
    print(" ", FAMILY_B_REFUSAL)
else:
    print("skipped: the GPU stage is not gated on")
'''
)

markdown(
    """
## 11. MOCK — the commissioned cases

Eight causal worlds and five provenance worlds, each with a bounded verdict
predeclared in `jlens.mmpilot.full_vocab_mock`. The records go through the real
`unrestricted_trial_record`, the real `summarize_full_vocab_cells` and the real
`unrestricted_reasoning_verdict`.

| case | required verdict |
|---|---|
| `full_vocab_go` | `FULL_VOCAB_REASONING_ALPHA1_GO` |
| `alpha2_only` | `FULL_VOCAB_REASONING_ALPHA2_ONLY` — never promoted |
| `restricted_only` | `FULL_VOCAB_REASONING_NO_GO` — **the defect being corrected** |
| `direct_answer_only` | `FULL_VOCAB_REASONING_NO_GO` — a positive control alone proves nothing |
| `asymmetric_direction` | GO, with one direction only, visible per direction |
| `control_failure` | `FULL_VOCAB_REASONING_NO_GO` |
| `capability_no_go` | `CAPABILITY_NO_GO` — no causal trial ran |
| `null` | `FULL_VOCAB_REASONING_NO_GO` |

**A green MOCK run is evidence about this code and about nothing else.**
"""
)
code(
    r'''
MOCK_RESULTS = {}
if not REAL_MODE:
    from jlens.mmpilot.full_vocab_mock import (
        CAUSAL_SCENARIOS, PROVENANCE_SCENARIOS, mock_band_followup_report,
        mock_full_vocab_records, mock_population_reuse,
    )
    from jlens.mmpilot.full_vocab_study import (
        conditional_logprob_verdict, cross_modal_conjunction, format_verdicts,
        read_band_followup_report, reuse_completed_population,
        summarize_full_vocab_cells, unrestricted_reasoning_verdict,
    )

    print(f"  {'case':<22} {'verdict':<36} {'as required':>11}")
    for _key, _scenario in CAUSAL_SCENARIOS.items():
        _records = mock_full_vocab_records(
            _scenario, bands=DESIGN["bands"], conditions=DESIGN["conditions"],
            arms=DESIGN["arms"],
        )
        _cells = summarize_full_vocab_cells(_records, thresholds=THRESHOLDS)
        _verdict = unrestricted_reasoning_verdict(
            _cells, bands=DESIGN["band_keys"], modalities=DESIGN["modalities"],
            directed_pairs=DESIGN["directed_pairs"], thresholds=THRESHOLDS,
            capability_sufficient=_scenario.capability_sufficient,
            causal_stage_ran=_scenario.causal_stage_ran,
        )
        _ok = _verdict["verdict"] == _scenario.expected_verdict
        MOCK_RESULTS[_key] = {
            "verdict": _verdict["verdict"],
            "expected": _scenario.expected_verdict,
            "as_required": _ok,
            "per_direction": _verdict.get("per_direction"),
        }
        print(f"  {_key:<22} {_verdict['verdict']:<36} {str(_ok):>11}")
        if not _ok:
            raise RuntimeError(
                f"MOCK case {_key!r} returned {_verdict['verdict']!r}, "
                f"required {_scenario.expected_verdict!r}"
            )

    print()
    print("  provenance worlds (through the real reader, on a temp directory):")
    import tempfile as _tempfile
    from jlens.mmpilot.validated_band_followup import FOLLOWUP_REPORT_NAME

    _fp = "sha256:" + "b" * 64
    _mock_report = mock_band_followup_report(fingerprint_pin=_fp)
    _pin = _mock_report["report_checksum"]
    for _key, _scenario in PROVENANCE_SCENARIOS.items():
        _mutated = _scenario.mutate(dict(_mock_report))
        with _tempfile.TemporaryDirectory() as _tmp:
            (Path(_tmp) / FOLLOWUP_REPORT_NAME).write_text(
                json.dumps(_mutated), encoding="utf-8"
            )
            try:
                _path, _read = read_band_followup_report(
                    _tmp, expected_report_checksum=_pin, expected_fingerprint=_fp
                )
                reuse_completed_population(_read)
                _refused = False
            except FullVocabRefused:
                _refused = True
        _ok = _refused == _scenario.expect_refusal
        print(f"  {_key:<22} refused={str(_refused):<6} "
              f"required={str(_scenario.expect_refusal):<6} {_ok}")
        if not _ok:
            raise RuntimeError(
                f"provenance case {_key!r} did not behave as required: {_scenario.because}"
            )

    # One end-to-end MOCK verdict trio, printed exactly as the real one is.
    _records = mock_full_vocab_records(
        CAUSAL_SCENARIOS["full_vocab_go"], bands=DESIGN["bands"],
        conditions=DESIGN["conditions"], arms=DESIGN["arms"],
    )
    MOCK_CELLS = summarize_full_vocab_cells(_records, thresholds=THRESHOLDS)
    MOCK_UNRESTRICTED = unrestricted_reasoning_verdict(
        MOCK_CELLS, bands=DESIGN["band_keys"], modalities=DESIGN["modalities"],
        directed_pairs=DESIGN["directed_pairs"], thresholds=THRESHOLDS,
    )
    MOCK_CONDITIONAL = conditional_logprob_verdict(None, evaluated=False)
    MOCK_CONJUNCTION = cross_modal_conjunction(MOCK_UNRESTRICTED, MOCK_CONDITIONAL)
    print()
    print(format_verdicts(MOCK_UNRESTRICTED, MOCK_CONDITIONAL, MOCK_CONJUNCTION))
    FV_RECORDS = _records
else:
    print("skipped: real mode")
'''
)

markdown(
    """
## 12. Stage 2 — the final report (CPU, no model, no media)

Every verdict is reconstructed from the saved units alone. The three verdict
families are computed separately, stored under separate keys, and printed
separately: an unrestricted-output verdict and a conditional-log-probability
verdict can never overwrite each other, and the cross-modal conjunction is
computed from both rather than asserted.
"""
)
code(
    r'''
from jlens.mmpilot.full_vocab_study import (
    conditional_logprob_verdict, cross_modal_conjunction, format_verdicts,
    full_vocab_report, summarize_full_vocab_cells, unrestricted_reasoning_verdict,
)

REPORT = None
if RUN_FINAL_REPORT_CPU or not REAL_MODE:
    if RUN_FINAL_REPORT_CPU:
        from jlens.mmpilot.store import payload_checksum
        _units_dir = RUN_DIR / "units" / "intervention" if RUN_DIR else None
        if _units_dir is None or not _units_dir.exists():
            raise FullVocabRefused(
                "RUN_FINAL_REPORT_CPU needs a completed run directory. Set "
                "RUN_DIR to the mmfv_real_* run you want to report on."
            )
        FV_RECORDS = []
        for _path in sorted(_units_dir.glob("*.json")):
            _stored = json.loads(_path.read_text(encoding="utf-8"))
            if _stored.get("unit_checksum") != payload_checksum(_stored["payload"]):
                raise FullVocabRefused(f"{_path} failed its own unit checksum")
            if _stored["payload"].get("status") == "complete":
                FV_RECORDS.append(_stored["payload"])
        print("units read (no model loaded)", len(FV_RECORDS))

    CELLS = summarize_full_vocab_cells(FV_RECORDS, thresholds=THRESHOLDS)
    UNRESTRICTED = unrestricted_reasoning_verdict(
        CELLS, bands=DESIGN["band_keys"], modalities=DESIGN["modalities"],
        directed_pairs=DESIGN["directed_pairs"], thresholds=THRESHOLDS,
        capability_sufficient=True, causal_stage_ran=bool(FV_RECORDS),
    )
    CONDITIONAL = conditional_logprob_verdict(
        None, evaluated=bool(FAMILY_B_EVALUATED)
    )
    CONJUNCTION = cross_modal_conjunction(UNRESTRICTED, CONDITIONAL)

    REPORT = full_vocab_report(
        mode="real" if REAL_MODE else "mock",
        design=DESIGN,
        fingerprint=FINGERPRINT_CONFIG,
        endpoint_audit=AUDIT,
        population_reuse=POPULATION_REUSE or mock_population_reuse(),
        band_followup_provenance={
            "run_name": BAND_FOLLOWUP_RUN_NAME,
            "report_checksum": BAND_FOLLOWUP_REPORT_CHECKSUM_PIN,
            "fingerprint": BAND_FOLLOWUP_FINGERPRINT_PIN,
            "read_or_modified": "read-only",
        },
        canonical_audio_provenance=AUDIO_PROVENANCE,
        tokens=TOKENS or {},
        budget=BUDGET,
        cells=CELLS,
        unrestricted=UNRESTRICTED,
        conditional=CONDITIONAL,
        conjunction=CONJUNCTION,
        greedy=GREEDY_RECORDS,
        hook_integrity=HOOK_INTEGRITY or {},
        fitting_audit={"fitting_performed": False, "backward_passes": 0},
        run_dir=str(RUN_DIR) if RUN_DIR else None,
    )
    print(format_verdicts(UNRESTRICTED, CONDITIONAL, CONJUNCTION))
    print()
    print("report checksum            ", REPORT["report_checksum"])
    print("completed reports modified ", REPORT["completed_reports_modified"])
    print("scientific recompute       ", REPORT["scientific_recompute_of_completed_runs"])
    if RUN_FINAL_REPORT_CPU and RUN_DIR:
        (RUN_DIR / "full_vocabulary_causal_validation_report.json").write_text(
            json.dumps(REPORT, indent=2, sort_keys=True, default=str), encoding="utf-8"
        )
        print("report written to", RUN_DIR)
else:
    print("skipped: RUN_FINAL_REPORT_CPU is False")
'''
)

markdown(
    """
## 13. The boundary

* This was a **measurement correction on the same populations**, not an
  independent replication. Any result here inherits every limitation of the
  populations it reuses, including the candidate-conditioned prompt limitation
  recorded in `docs/prompt_protocol.md`.
* α=2 is a prespecified sensitivity condition. It is never primary evidence and
  is never described as one.
* The completed `L33_L40_VALIDATED_BAND_FOLLOWUP_ALPHA2_SENSITIVITY_ONLY` and
  `BAND_CORRECTED_CONTROL_NO_GO` verdicts are unchanged. They are relabelled as
  restricted-candidate results by a versioned amendment, and their numbers are
  identical.
* L32 remains excluded. This notebook supports no claim about a band beginning
  at L32 or about any layer earlier than 33.
* SpokenCOCO recordings are spoken captions — linguistic audio. Nothing here is
  evidence about environmental sound.
* The model outputs text; image and audio are input modalities only.

**After this experiment the research phase ends.** No further alpha search,
concept search, layer search, population redraw or lens refit follows from any
outcome here. What follows is writing, figures, limitations and release.
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
