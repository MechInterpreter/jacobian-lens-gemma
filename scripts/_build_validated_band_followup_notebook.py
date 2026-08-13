"""Generate the L33-L40 validated-band causal follow-up Colab notebook.

Deterministic: running this script twice writes byte-identical output, and
``tests/test_validated_band_followup_notebook.py`` fails if the checked-in
notebook drifts from it.

This builder writes a **new** notebook. The completed L32-L40 band-swap
notebook and its builder are untouched, because retroactively changing what
that notebook means is exactly what a follow-up must not do.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = (
    ROOT / "notebooks" / "multimodal_jspace_anthropic_band33_40_swap_colab.ipynb"
)
CELLS: list[tuple[str, str]] = []


def markdown(value: str) -> None:
    CELLS.append(("markdown", value.strip("\n")))


def code(value: str) -> None:
    CELLS.append(("code", value.strip("\n")))


markdown(
    r"""
# The validated band L33-L40 — a prospective causal follow-up (Gemma 4 E4B)

Anthropic's **primary** swap protocol, over the physical band an independent
text-only lens validation actually validated:

`c = pinv(V) h`, `h_patched = h + alpha * V (sigma(c) - c)`

* at **every physical layer of the contiguous band L33-L40**, all eight hooks
  installed at once,
* with the coordinates **recomputed at each layer** from that layer's own
  activation and its own validated `W_U @ J_l`,
* at **every original prompt position**, never at a teacher-forced candidate
  token,
* `alpha = 1` as the primary exact exchange and `alpha = 2` as a separately
  labelled fixed double-strength sensitivity condition, each against its own
  intensity-matched controls,
* scored by the top-1 trial definition: **the downstream target answer becomes
  top-1**.

## Read this before anything else

The corrected independent lens validation
(`bandcorr_real_eb5b00f135e4`) scored all nine physical layers 32-40 on one
untouched confirmation population under the corrected fixed-universe
wrong-layer control. **L32 failed** the frozen coverage/nondegeneracy clause —
tied-at-max 0.51171875 against a frozen ceiling of 0.50 — and L33-L40 passed.

| | |
|---|---|
| The originally planned full L32-L40 study | remains **`BAND_CORRECTED_CONTROL_NO_GO`**, permanently |
| Its stage 3 | correctly remained blocked, and stays blocked |
| This notebook | is a **prospective causal follow-up** over L33-L40 |
| What selected L33-L40 | the completed text-only lens validation, and nothing else |
| What did not select it | any causal outcome; none existed for any band |

A favourable result here may support causal exchange across the validated
L33-L40 band under this protocol. It is **never** relabelled as confirmation of
the failed L32-L40 design, it supports no claim about a band beginning at L32,
and it supports no claim about any layer earlier than 33.

L32 failed only one clause — it passed MRR-vs-noise, MRR-vs-wrong-layer,
rank/top-10 and fold stability. That does not make it validated, and it does
not authorise changing the tie ceiling, re-drawing the population until L32
passes, or slipping L32 into a band. There is no configuration of this notebook
that admits it.

## What this notebook does and does not touch

The completed corrected run is **read-only selection evidence**. It is
checksummed before the first read and again after the last one, and the
preflight refuses if a single byte moved. Nothing is refitted: eight validated
scale-250 matrices are read from that run's own publication directory, resolved
through its own report and sidecars, and re-checksummed against pinned values.
There is no fitting entry point in this notebook, and the preflight proves it.

## The stages

| stage | runtime | what it does |
|---|---|---|
| A | **CPU** | Verify the corrected report, resolve and re-checksum the eight L33-L40 artifacts, prove the completed run unchanged, freeze the design, print the pass budget. No model. |
| B | **GPU** | The band swap: clean behavioural screen, then `coordinate_swap_band` over the four predeclared suffix bands. Blocked until stage A prints an admission. |
| C | **CPU** | Aggregate, judge, and the intermediate-versus-answer timing comparison. No model, no media. |

Run with every switch `False` and the notebook performs a complete MOCK run on
CPU — no Drive, no model, no download, no spend. **A green MOCK run is evidence
about this code and about nothing else.**
"""
)

markdown("## 0. Colab bootstrap")
code(
    r'''
import os, subprocess, sys
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
## 1. Configuration — three switches, cleanly separated

`RUN_PREFLIGHT_CPU` and `RUN_REPORT_CPU` never load a model. `RUN_CAUSAL_GPU`
additionally requires `CONFIRM_MODEL_LOAD` and `CONFIRM_PASS_BUDGET`, both set
by hand after reading the printed budget in section 6.

With all three `False` the notebook performs a complete MOCK run on CPU.
"""
)
code(
    r'''
# ---- stage switches (all False = full MOCK run on CPU) -------------------
RUN_PREFLIGHT_CPU = False   # CPU: verify the corrected run, freeze the design. No model.
RUN_CAUSAL_GPU = False      # GPU: the L33-L40 contiguous-band causal run.
RUN_REPORT_CPU = False      # CPU: aggregate, judge, timing. No model, no media.

# Set this to a finished stage-B run directory to run stage C on its own.
# Leave it None to analyse the run this session produced.
REPORT_RUN_DIR = None

# ---- explicit spend confirmations (read section 6 first) ----------------
CONFIRM_MODEL_LOAD = False
CONFIRM_PASS_BUDGET = False

REAL_MODE = any((RUN_PREFLIGHT_CPU, RUN_CAUSAL_GPU, RUN_REPORT_CPU))
MODE = "real" if REAL_MODE else "mock"

# ---- the frozen follow-up design ----------------------------------------
from jlens.mmpilot.band_swap import (
    BAND_CONDITIONS, PRIMARY_ALPHA, SECONDARY_ALPHA, BandSwapThresholds,
)
from jlens.mmpilot.coordinate_swap import PRIMARY_POSITION_RULE
from jlens.mmpilot.validated_band_followup import (
    EXCLUDED_FAILED_LAYER, EXCLUDED_LAYER_REASON, FOLLOWUP_BAND_END,
    FOLLOWUP_BAND_START, FOLLOWUP_PRIMARY_BAND, FOLLOWUP_PROTOCOL_VERSION,
    FOLLOWUP_STUDY_NAME, FOLLOWUP_SUFFIX_STARTS, ORIGINAL_RUN_NAME,
    ORIGINAL_VERDICT, format_followup_boundary,
)

# Band STARTS, not a sampled layer grid. The band beginning at 33 patches every
# physical layer 33..40; reading this tuple as the patched set would report
# layers as patched that no hook ever touched.
BAND_START_LAYERS = FOLLOWUP_SUFFIX_STARTS   # (33, 35, 38, 40)
BAND_END_LAYER = FOLLOWUP_BAND_END
POSITION_RULE = PRIMARY_POSITION_RULE
ALPHAS = (PRIMARY_ALPHA, SECONDARY_ALPHA)

PAIR_CONCEPTS = ("bird", "cat")        # 2 legs versus 4 legs
CONTROL_CONCEPTS = ("zebra", "giraffe")
POPULATION_CONCEPTS = (*PAIR_CONCEPTS, *CONTROL_CONCEPTS)
MODALITIES = ("text", "image", "spoken_audio")
PAPER_COMPARABLE_MODALITY = "text"
READOUT_ARMS = ("identity", "property")
CANDIDATE_IMAGES_PER_CONCEPT = 24
MAX_ANALYSIS_IMAGES_PER_CELL = 8
MIN_ANALYSIS_IMAGES_PER_CELL = 4

# A new seed for a new population. The completed causal runs' photographs are
# excluded outright; this seed decides the order of what remains.
SELECTION_SEED = "l33-l40-validated-band-followup-gemma-v1"
RANDOM_CONTROL_SEED_INTERMEDIATE = 20260813
RANDOM_CONTROL_SEED_ANSWER = 20261813

THRESHOLDS = BandSwapThresholds(
    min_images=MIN_ANALYSIS_IMAGES_PER_CELL,
    min_target_top1_rate=0.50,
    control_top1_margin=0.0,
    max_identity_flip_rate_answer_arm=0.25,
)

# ---- identity of the model this study is about --------------------------
MODEL_REPO_ID = "google/gemma-4-E4B-it"
MODEL_REVISION = "fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd"
TRANSFORMERS_VERSION_EXPECTED = "5.13.1"
EXPECT_N_LAYERS, EXPECT_D_MODEL, EXPECT_VOCAB = 42, 2560, 262144
AUDIO_PROTOCOL_FINGERPRINT = (
    "sha256:9ad8bcc9420a7983f6e3b75d5d7080c0e2fcf0a94a76431917fcde73ba777920"
)

print(format_followup_boundary())
print()
print("mode                        ", MODE)
print("study                       ", FOLLOWUP_STUDY_NAME)
print("protocol                    ", FOLLOWUP_PROTOCOL_VERSION)
print("primary band (patched)      ", list(FOLLOWUP_PRIMARY_BAND))
print("suffix band STARTS          ", list(BAND_START_LAYERS), "-> end", BAND_END_LAYER)
print("excluded failed layer       ", f"L{EXCLUDED_FAILED_LAYER}")
print("original L32-L40 verdict    ", ORIGINAL_VERDICT, f"({ORIGINAL_RUN_NAME})")
print("position rule               ", POSITION_RULE)
print("alphas                      ", ALPHAS, "(1 = exchange, 2 = sensitivity)")
print("conditions                  ", BAND_CONDITIONS)
print("threshold digest            ", THRESHOLDS.digest)
for name, value in (
    ("RUN_PREFLIGHT_CPU", RUN_PREFLIGHT_CPU),
    ("RUN_CAUSAL_GPU", RUN_CAUSAL_GPU),
    ("RUN_REPORT_CPU", RUN_REPORT_CPU),
    ("CONFIRM_MODEL_LOAD", CONFIRM_MODEL_LOAD),
    ("CONFIRM_PASS_BUDGET", CONFIRM_PASS_BUDGET),
):
    print(f"{name:<28} {value}")
if not REAL_MODE:
    print()
    print("MOCK RUN: no Drive, no model, no download, nothing spent.")
'''
)

markdown(
    """
## 2. The runs this notebook reads, and the ones it must never write into

Every path is **explicitly configured**. Nothing globs for "the latest run":
selection evidence that depends on directory sort order can change without
anyone editing it.

`CORRECTED_RUN_DIR` is the completed corrected validation. It is read for two
things — the report that selected L33-L40, and the eight artifacts that define
the exchanged coordinates — and it is proved byte-for-byte unchanged afterwards.

The completed causal runs are read for exactly one thing: every photograph they
spent is excluded from this study's population. Those populations were examined
while their successors were designed, so drawing from them again would select
images already known to be capability-valid.
"""
)
code(
    r'''
import json

RUNS_ROOT = Path("/content/drive/MyDrive/jacobian-lens-gemma/runs")

# The completed corrected validation — explicitly named, never discovered.
CORRECTED_RUN_DIR = RUNS_ROOT / "mmband" / ORIGINAL_RUN_NAME

EXPANDED_MANIFEST_CACHE = (
    RUNS_ROOT / "mml32_l32_followup_20260808T182717" / "expanded_manifest.json"
)
PRIOR_EXCLUSION_SET = Path(
    "/content/drive/MyDrive/datasets/cstf_spokencoco_derived/"
    "jlens_l32_resolution_prep_v1/"
    "prep_020ebbe6f832aece5ece6cb8bee994ca/exclusion_set.json"
)

# Read-only. A directory whose report does not match its pinned checksum is
# refused rather than trusted — refusing to guess which photographs it spent.
COMPLETED_CAUSAL_RUNS = {
    "single_layer_v1": (
        RUNS_ROOT / "mmpaper_real_24be1d028bf1",
        "paper_reasoning_swap_report.json",
        "sha256:a60f3336bf8acdc98dc1a434698104eaa98b3192c44f43fa5ab21212826ae397",
    ),
    "single_layer_v2": (
        RUNS_ROOT / "mmpaper2_real_04ab55235502",
        "paper_reasoning_swap_v2_report.json",
        "sha256:b64ce3cec51371769b908d14342fbf42f64a6dccb82f8d235ad81d643815ddc6",
    ),
    "alpha2_capability_screen": (
        RUNS_ROOT / "mmpaperconfirm_real_6b0745c08d84",
        "paper_reasoning_swap_alpha2_confirmation_report.json",
        "sha256:37d32605b24984f09c0dfccaab7c7ea98e217bef82412bd28576384b22f23c11",
    ),
}

# Nothing this notebook writes may land inside any of these.
PROTECTED_RUN_DIRS = (
    CORRECTED_RUN_DIR,
    RUNS_ROOT / "rgcalib_real_7e3736b4de8f",
    RUNS_ROOT / "rgext_real_c18f03f06e7b",
    RUNS_ROOT / "mmband" / "bandlens_real_de9338ec2a6e",
    *(directory for directory, _, _ in COMPLETED_CAUSAL_RUNS.values()),
)

FOLLOWUP_RUN_ROOT = RUNS_ROOT / "mmband33"

if REAL_MODE:
    if IN_COLAB:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
    required = [CORRECTED_RUN_DIR]
    if RUN_CAUSAL_GPU:
        required += [EXPANDED_MANIFEST_CACHE, PRIOR_EXCLUSION_SET]
        required += [directory for directory, _, _ in COMPLETED_CAUSAL_RUNS.values()]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "configured artifact(s) missing; refusing before any model load:\n  "
            + "\n  ".join(missing)
        )
    if RUN_CAUSAL_GPU:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("stage B requires a GPU runtime; use L4 or A100")
    print("every configured artifact is present")
    print("corrected run  ", CORRECTED_RUN_DIR)
    print("follow-up runs ", FOLLOWUP_RUN_ROOT)
else:
    print("MOCK: no Drive mounted, no artifact read.")
'''
)

markdown(
    """
## 3. Stage A — the mandatory preflight

CPU only. No model is loaded anywhere in this section, and the notebook stops
here rather than three hours in if any clause fails.

1. Read the corrected report from the **explicitly configured** run directory.
2. Verify its pinned `report_checksum`, and that it recomputes over its own body.
3. Verify schema, `mode=real`, protocol digest, universe checksum,
   confirmation-manifest checksum, model identity and revision, scale 250, hook
   convention, fit-prefix checksum and the frozen gate digest.
4. Verify `layers_passing == [33..40]` exactly, that 32 is among
   `layers_failing`, that the largest admissible contiguous band is exactly
   `[33, 40]`, that the report did **not** unblock the original stage 3, and
   that no threshold was changed and no matrix refitted.
5. Resolve each L33-L40 artifact through the report's own publication directory
   and sidecars — no filename is guessed.
6. Recompute every file checksum and match the eight pinned values.
7. Verify every artifact is scale 250, text-only, model/revision matched,
   independently confirmed on the same manifest, and shares one capture
   geometry and estimator.
8. Refuse L32 categorically.
9. Refuse any missing, duplicate, inconsistent, unpublished or mismatched
   artifact.
10. Checksum the completed run before and after, and prove it unchanged.
"""
)
code(
    r'''
from jlens.mmpilot.validated_band_followup import (
    EXPECTED_ARTIFACT_CHECKSUMS, EXPECTED_CONFIRMATION_MANIFEST_CHECKSUM,
    EXPECTED_PROTOCOL_DIGEST, EXPECTED_REPORT_CHECKSUM, EXPECTED_UNIVERSE_CHECKSUM,
    FollowupRefused, assert_corrected_run_unmodified, assert_followup_band,
    assert_no_fitting_entry_point, corrected_run_digest,
    discover_corrected_band_lenses, followup_preflight_record,
    format_followup_preflight, read_corrected_validation_report,
)

PREFLIGHT = None
CORRECTED_REPORT = None
CORRECTED_ARTIFACTS = {}
ARTIFACT_DISCOVERY = {}
LENS_CHECKSUMS = {}
IMMUTABILITY = None
FITTING_AUDIT = None

if RUN_PREFLIGHT_CPU or RUN_CAUSAL_GPU:
    _digest_before = corrected_run_digest(CORRECTED_RUN_DIR)

    _report_path, CORRECTED_REPORT = read_corrected_validation_report(
        CORRECTED_RUN_DIR,
        expected_report_checksum=EXPECTED_REPORT_CHECKSUM,
        expected_protocol_digest=EXPECTED_PROTOCOL_DIGEST,
        expected_universe_checksum=EXPECTED_UNIVERSE_CHECKSUM,
        expected_confirmation_manifest_checksum=(
            EXPECTED_CONFIRMATION_MANIFEST_CHECKSUM
        ),
        expected_model_repo_id=MODEL_REPO_ID,
        expected_model_revision=MODEL_REVISION,
    )
    ADMISSION = assert_followup_band(CORRECTED_REPORT)
    CORRECTED_ARTIFACTS, ARTIFACT_DISCOVERY = discover_corrected_band_lenses(
        CORRECTED_RUN_DIR,
        report=CORRECTED_REPORT,
        layers=FOLLOWUP_PRIMARY_BAND,
        expected_checksums=EXPECTED_ARTIFACT_CHECKSUMS,
    )
    LENS_CHECKSUMS = {
        layer: source.lens_checksum for layer, source in CORRECTED_ARTIFACTS.items()
    }

    # Every read of the completed run is done. Prove it is unchanged.
    IMMUTABILITY = assert_corrected_run_unmodified(
        _digest_before, corrected_run_digest(CORRECTED_RUN_DIR)
    )

    PREFLIGHT = followup_preflight_record(
        report_path=_report_path,
        report=CORRECTED_REPORT,
        admission=ADMISSION,
        discovery=ARTIFACT_DISCOVERY,
        immutability=IMMUTABILITY,
        corrected_run_dir=CORRECTED_RUN_DIR,
    )
    print(format_followup_preflight(PREFLIGHT))
    print()
    print("=" * 78)
    print("ORIGINAL L32-L40 VERDICT REMAINS  ", ORIGINAL_VERDICT)
    print("NEW FOLLOW-UP BAND                 L33-L40")
    print("SELECTION SOURCE                   lens validation only (text-only)")
    print("NO CAUSAL OUTCOME SELECTED THE BAND")
    print("NO FITTING WILL OCCUR")
    print("=" * 78)
else:
    print("skipped: RUN_PREFLIGHT_CPU and RUN_CAUSAL_GPU are both False")

# The claim that nothing is fitted, made checkable. Every module this notebook
# touches is searched for a calibration or fitting entry point.
import jlens.mmpilot.band_swap as _band_swap_module
import jlens.mmpilot.coordinate_swap as _coordinate_swap_module
import jlens.mmpilot.validated_band_followup as _followup_module

FITTING_AUDIT = assert_no_fitting_entry_point(
    _followup_module, _band_swap_module, _coordinate_swap_module
)
print()
print("fitting audit:", FITTING_AUDIT["no_fitting_entry_point_is_reachable"],
      " backward passes", FITTING_AUDIT["backward_passes"])
'''
)

markdown(
    """
## 4. MOCK — the preflight against commissioned cases

The real reader, the real admission clauses, the real discovery and the real
immutability proof, run over a synthetic corrected run written to a temporary
directory. Sixteen commissioned cases: one that must be admitted and fifteen
that must be refused, including a report in which **L32 also passed** — which
would be the originally planned confirmatory band and is a different study with
a different predeclaration.

**A green MOCK run is evidence about this code and about nothing else.**
"""
)
code(
    r'''
PREFLIGHT_MOCK_RESULTS = {}
if not REAL_MODE:
    import tempfile as _tempfile

    from jlens.mmpilot.validated_band_followup import (
        FollowupRefused as _FollowupRefused,
        assert_corrected_run_unmodified as _assert_unmodified,
        assert_followup_band as _assert_band,
        corrected_run_digest as _run_digest,
        discover_corrected_band_lenses as _discover,
        read_corrected_validation_report as _read_report,
    )
    from jlens.mmpilot.validated_band_followup_mock import (
        PREFLIGHT_SCENARIOS, mock_corrected_run,
    )

    def _run_mock_preflight(pins, *, require_real_mode):
        _path, _report = _read_report(
            pins["run_dir"],
            expected_report_checksum=pins["expected_report_checksum"],
            expected_protocol_digest=pins["expected_protocol_digest"],
            expected_universe_checksum=pins["expected_universe_checksum"],
            expected_confirmation_manifest_checksum=(
                pins["expected_confirmation_manifest_checksum"]
            ),
            expected_model_repo_id=pins["expected_model_repo_id"],
            expected_model_revision=pins["expected_model_revision"],
            require_real_mode=require_real_mode,
        )
        _assert_band(_report)
        return _discover(
            pins["run_dir"],
            report=_report,
            expected_checksums=pins["expected_artifact_checksums"],
        )

    print(f"  {'case':<32} {'refused':>8}  {'as required':>11}")
    for _key, _scenario in PREFLIGHT_SCENARIOS.items():
        # ignore_cleanup_errors: the fixture is written, read and checksummed
        # inside the block, and a virus scanner briefly holding a handle after
        # it must not turn a passed scenario into a failed run.
        with _tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as _tmp:
            _pins = mock_corrected_run(_tmp, scenario=_key)
            _before = _run_digest(_pins["run_dir"])
            try:
                # The fixture writes mode="mock", so the real reader refuses it
                # unless told otherwise. The mock_mode_report case is the one
                # that does not tell it otherwise.
                _run_mock_preflight(
                    _pins, require_real_mode=(_key == "mock_mode_report")
                )
                _refused, _why = False, None
            except _FollowupRefused as _error:
                _refused, _why = True, str(_error).splitlines()[0]
            _unchanged = _assert_unmodified(_before, _run_digest(_pins["run_dir"]))
            PREFLIGHT_MOCK_RESULTS[_key] = {
                "refused": _refused,
                "must_be_refused": _scenario.must_be_refused,
                "as_required": _refused == _scenario.must_be_refused,
                "fixture_unchanged_by_reading": _unchanged["identical"],
                "reason": _why,
            }
        _row = PREFLIGHT_MOCK_RESULTS[_key]
        print(f"  {_key:<32} {str(_refused):>8}  {str(_row['as_required']):>11}")
    assert all(row["as_required"] for row in PREFLIGHT_MOCK_RESULTS.values())
    assert all(
        row["fixture_unchanged_by_reading"] for row in PREFLIGHT_MOCK_RESULTS.values()
    )
    print()
    print("MOCK preflight complete: real reader, real clauses, real refusals.")
    print("L32 is refused in every case in which it appears.")
else:
    print("skipped: real mode")
'''
)

markdown(
    """
## 5. Freeze the design and the four predeclared bands

The primary band is the whole validated range. The comparison bands are its
suffixes, with the starts predeclared **before** any causal number exists:

`[33..40]`, `[35..40]`, `[38..40]`, `[40..40]`

This is the earlier intended study's comparison topology, adjusted in exactly
one place because the validated primary band starts at 33 rather than 32.
Nothing is added or removed after seeing a causal result, and no all-layer
sweep is substituted for it.

`build_band` refuses any band containing a layer that is not in the admitted
set, `assert_contiguous` refuses a sampled grid described as a range, and
`followup_design_record` refuses a band containing L32 or one that leaves the
validated primary band.
"""
)
code(
    r'''
from jlens.mmpilot.band_swap import (
    BandDesignRefused, assert_contiguous, band_key, build_band,
    predeclare_suffix_bands,
)
from jlens.mmpilot.coordinate_swap import LayerBandError
from jlens.mmpilot.validated_band_followup import followup_design_record

DESIGN, PRIMARY_BAND, SUFFIX_BANDS, BAND_KEYS = None, None, (), ()

# The sampled grid is not a band, and this is where that is enforced.
try:
    assert_contiguous((33, 35, 38, 40), what="the predeclared band START list")
    raise AssertionError("unreachable: a sampled grid must never validate as a band")
except BandDesignRefused as _refusal:
    print("sanity check —", _refusal)

# L32 can never enter a band admitted here, whatever a report says. Two
# independent refusals: the lens gate has no artifact for it, and the design
# record refuses it even when a caller hands one over.
try:
    build_band(32, BAND_END_LAYER, usable_layers=FOLLOWUP_PRIMARY_BAND,
               n_layers=EXPECT_N_LAYERS)
    raise AssertionError("unreachable: L32 has no validated lens for a band")
except LayerBandError as _refusal:
    print("sanity check —", str(_refusal).splitlines()[0])
try:
    followup_design_record(
        primary_band=build_band(
            32, BAND_END_LAYER,
            usable_layers=(32, *FOLLOWUP_PRIMARY_BAND), n_layers=EXPECT_N_LAYERS,
        ),
        suffix_bands=(), admission={}, discovery={},
    )
    raise AssertionError("unreachable: L32 must never enter the follow-up band")
except FollowupRefused as _refusal:
    print("sanity check —", str(_refusal).splitlines()[0])
print()

_usable = tuple(FOLLOWUP_PRIMARY_BAND)
if not REAL_MODE:
    _admission_record = {
        "layers_passing": list(FOLLOWUP_PRIMARY_BAND),
        "layers_failing": [EXCLUDED_FAILED_LAYER],
        "followup_band": list(FOLLOWUP_PRIMARY_BAND),
        "mock": "synthetic admission; proves the pipeline and nothing else",
    }
    _discovery_record = {
        "discovery_checksum": "sha256:mock-discovery",
        "lens_checksums": {
            str(layer): f"sha256:mock-corrected-L{layer}"
            for layer in FOLLOWUP_PRIMARY_BAND
        },
        "mock": True,
    }
    LENS_CHECKSUMS = {
        layer: f"sha256:mock-corrected-L{layer}" for layer in FOLLOWUP_PRIMARY_BAND
    }
else:
    _admission_record = ADMISSION if PREFLIGHT else {}
    _discovery_record = ARTIFACT_DISCOVERY

if PREFLIGHT is not None or not REAL_MODE:
    PRIMARY_BAND = build_band(
        FOLLOWUP_BAND_START, BAND_END_LAYER,
        usable_layers=_usable, n_layers=EXPECT_N_LAYERS,
    )
    SUFFIX_BANDS = predeclare_suffix_bands(
        starts=BAND_START_LAYERS, end=BAND_END_LAYER,
        usable_layers=_usable, n_layers=EXPECT_N_LAYERS,
    )
    BAND_KEYS = tuple(band_key(band) for band in SUFFIX_BANDS)
    DESIGN = followup_design_record(
        primary_band=PRIMARY_BAND,
        suffix_bands=SUFFIX_BANDS,
        admission=_admission_record,
        discovery=_discovery_record,
        position_rule=POSITION_RULE,
        alphas=ALPHAS,
        conditions=BAND_CONDITIONS,
    )
    print("primary band      ", list(PRIMARY_BAND.layers))
    print("predeclared bands ", list(BAND_KEYS))
    for _band in SUFFIX_BANDS:
        print(f"  {band_key(_band):>6}  patches {list(_band.layers)}")
    print("design digest     ", DESIGN["design_digest"])
    print()
    print(" ", DESIGN["sampled_start_list_is_not_the_patched_layers"])
else:
    print("skipped: stage A did not run, so no design is frozen")
'''
)

markdown(
    """
## 6. The exact pass budget — printed before any model can load

Derived from the configuration, never hard-coded. A *candidate pass* is one
teacher-forced scored forward pass.

* **clean** = 2 concepts x 24 photographs x 3 modalities x 2 readouts x 2 candidates
* **intervention** = (2 x 3 x 8) analysis cells x 4 bands x 2 arms x 7 conditions
  x 2 readouts x 2 candidates

Installing eight hooks costs no more forward passes than installing one, so a
band trial costs what a single-layer trial costs. If the derived total is not
the expected design budget the notebook stops and names the factor rather than
loading a model against a budget nobody checked.

The wall-time band below is an **extrapolation**, derived from 0.9–2.2 s per
candidate pass. The completed L32–L40 notebook asserted 2–5 h in prose for the
identical 11,328-pass workload; neither number is a measurement, and the first
fifty trials of a real run will tell you more than either.
"""
)
code(
    r'''
from jlens.mmpilot.validated_band_followup import (
    followup_pass_budget, format_followup_pass_budget,
)

PASS_BUDGET = followup_pass_budget(
    n_pair_concepts=len(PAIR_CONCEPTS),
    n_modalities=len(MODALITIES),
    n_readouts=len(READOUT_ARMS),
    n_candidates_per_readout=len(PAIR_CONCEPTS),
    candidate_images_per_concept=CANDIDATE_IMAGES_PER_CONCEPT,
    max_analysis_images_per_cell=MAX_ANALYSIS_IMAGES_PER_CELL,
    n_bands=len(BAND_START_LAYERS),
    n_arms=2,
    n_conditions=len(BAND_CONDITIONS),
    band_layer_counts=[
        BAND_END_LAYER - int(start) + 1 for start in sorted(BAND_START_LAYERS)
    ],
)
print(format_followup_pass_budget(PASS_BUDGET))

if not PASS_BUDGET["matches_expected_design"]:
    raise RuntimeError(
        "the derived pass budget is not the expected design budget:\n"
        f"  derived  clean={PASS_BUDGET['clean_candidate_passes']} "
        f"intervention={PASS_BUDGET['intervention_candidate_passes']} "
        f"total={PASS_BUDGET['total']}\n"
        f"  expected clean={PASS_BUDGET['expected_clean_candidate_passes']} "
        f"intervention={PASS_BUDGET['expected_intervention_candidate_passes']} "
        f"total={PASS_BUDGET['expected_total']}\n"
        f"  factors  {PASS_BUDGET['factors']}\n"
        "Name the factor that changed before allowing a model to load."
    )
if RUN_CAUSAL_GPU and not (CONFIRM_MODEL_LOAD and CONFIRM_PASS_BUDGET):
    raise RuntimeError(
        "stage B needs CONFIRM_MODEL_LOAD and CONFIRM_PASS_BUDGET set by hand"
    )
'''
)

markdown(
    """
## 7. Stage B — the contiguous-band coordinate swap over L33-L40

GPU. `coordinate_swap_band` installs one hook per physical band layer, all at
once, for the whole scored forward pass. Each layer reads its own activation,
builds `c` from its own validated lens vectors, and patches every original
prompt position; positions at or beyond `prompt_len` are teacher-forced
candidate tokens and are never touched.

Two refusals guard every stored trial: `band_trial_record` refuses a trial whose
hooks fired at the wrong *set* of layers, and `assert_band_hook_integrity`
refuses one whose hooks fired an uneven number of times or at the wrong
positions.

**The population is fresh and image-disjoint.** Every photograph and group spent
by the three completed causal runs is excluded, along with the prior exclusion
set. One photograph is the independent unit and one synchronized
group/recording is kept per photograph. The clean behavioural screen runs
first, and the causal stage may use only predeclared capability-eligible cells
under the existing frozen rule — no concept or sample is replaced based on an
outcome.

Both arms are preserved: `intermediate` exchanges the animal-identity
coordinates, `answer` exchanges the corresponding two/four leg-answer
coordinates. Conditions: `swap_alpha1` (primary), `swap_alpha2` (sensitivity),
`zero`, and norm-matched random and unrelated-pair controls **at both alphas**,
so alpha=2 is never compared against an alpha=1 baseline.

The spoken-audio transcript is passed to `build_backend_inputs` **only** so it
can prove the transcript is absent from every backend argument. It never
reaches the model.
"""
)
code(
    r'''
from jlens.mmpilot.band_swap import BAND_INTERVENTION_FAMILY, CONDITION_ALPHA, band_trial_record
from jlens.mmpilot.validated_band_followup import (
    FOLLOWUP_INTERVENTION_FAMILY, assert_band_hook_integrity, followup_fingerprint,
)

SWAP_STORE = None
SWAP_RUN_DIR = None
BAND_RECORDS = []
POPULATION = None
EXCLUSION = {}
MEDIA_CHECKSUMS = {}
CAUSAL_SELECTION = None
CAUSAL_STAGE_RAN = False
CAPABILITY_SUFFICIENT = True
HOOK_INTEGRITY = None
FINGERPRINT_CONFIG = None
DIRECTED_PAIRS = []

if RUN_CAUSAL_GPU:
    import getpass, torch
    from jlens.mmpilot.capability import prediction_and_margin, score_candidate_sequences
    from jlens.mmpilot.coordinate_swap import (
        METHOD_VERSION, assert_open_prompt_protocol, build_swap_basis_from_vectors,
        random_two_direction_basis, resolve_concept_token, run_swap_condition,
    )
    from jlens.mmpilot.evidence import EvidenceConfig
    from jlens.mmpilot.media_io import RetryJournal, drive_media_loaders
    from jlens.mmpilot.paper_reasoning_swap import (
        hidden_animal_population, select_capability_eligible_samples,
    )
    from jlens.mmpilot.prompt_protocol import (
        Evidence, HIDDEN_ANIMAL_LEGS, OPEN_ANIMAL_IDENTIFICATION, PromptLeakageError,
        assert_property_contrast, build_backend_inputs, build_protocol_prompt,
        concept_spec, leg_count_surfaces, prompt_protocol_fingerprint, resolve_leg_count,
    )
    from jlens.mmpilot.real_backend import build_real_backend
    from jlens.mmpilot.store import RunFingerprint, UnitStore, payload_checksum, safe_key
    from jlens.mmpilot.tri_modal import assert_audio_protocol
    from jlens.lens import JacobianLens

    if PREFLIGHT is None or DESIGN is None:
        raise RuntimeError(
            "stage B is blocked: the section 3 preflight did not admit the "
            "L33-L40 band in this session. The causal stage runs only behind a "
            "printed admission from the completed corrected validation."
        )
    if sorted(PRIMARY_BAND.layers) != sorted(PREFLIGHT["admission"]["layers_passing"]):
        raise RuntimeError(
            f"the frozen primary band {sorted(PRIMARY_BAND.layers)} is not the set "
            f"of layers the corrected validation passed "
            f"{PREFLIGHT['admission']['layers_passing']}"
        )

    if not os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = getpass.getpass("HF_TOKEN (input hidden): ").strip()
    _bundle = build_real_backend(
        MODEL_REPO_ID, revision=MODEL_REVISION, token=os.environ["HF_TOKEN"],
        device="cuda", allow_model_load=True, resolve_audio=True,
        expect_n_layers=EXPECT_N_LAYERS, expect_d_model=EXPECT_D_MODEL,
        expect_vocab_size=EXPECT_VOCAB,
    )
    if _bundle.audio_interface is None:
        raise RuntimeError("native spoken audio did not resolve: " + _bundle.audio_blocked_reason)
    AUDIO_RECORD = assert_audio_protocol(
        _bundle.audio_interface, expected_fingerprint=AUDIO_PROTOCOL_FINGERPRINT
    )
    BACKEND = _bundle.backend

    # --- per-layer J-lens vectors, each from the corrected artifact the
    #     preflight resolved and re-checksummed. Every band layer gets its own.
    TOKEN_NAMES = (*POPULATION_CONCEPTS, "two", "four")
    TOKENS = {name: resolve_concept_token(BACKEND.encode_candidate, name) for name in TOKEN_NAMES}
    _unembedding = BACKEND.unembedding_weight()
    _rows = {
        name: _unembedding[token.token_id].detach().float().cpu()
        for name, token in TOKENS.items()
    }
    TOKEN_VECTORS = {}
    _loaded_files = {}
    for _layer in PRIMARY_BAND.layers:
        _source = CORRECTED_ARTIFACTS[_layer]
        if _source.lens_path not in _loaded_files:
            _loaded_files[_source.lens_path] = JacobianLens.load(_source.lens_path)
        _jacobian = _loaded_files[_source.lens_path].jacobians[_source.layer_key_in_file]
        _jacobian = _jacobian.detach().float().cpu()
        TOKEN_VECTORS[_layer] = {name: row @ _jacobian for name, row in _rows.items()}
        del _jacobian
    del _loaded_files, _unembedding, _rows
    if sorted(TOKEN_VECTORS) != sorted(PRIMARY_BAND.layers):
        raise RuntimeError("a band layer has no coordinate vectors of its own")
    print("lens vectors built, one basis source per band layer:", sorted(TOKEN_VECTORS))

    def selected_bases(layers, source_name, target_name):
        return {
            layer: build_swap_basis_from_vectors(
                TOKEN_VECTORS[layer][source_name], TOKEN_VECTORS[layer][target_name],
                layer=layer, source=TOKENS[source_name], target=TOKENS[target_name],
            )
            for layer in layers
        }

    # --- the fresh population, built after the design is frozen
    _raw = EXPANDED_MANIFEST_CACHE.read_bytes()
    MANIFEST_FILE_CHECKSUM = "sha256:" + __import__("hashlib").sha256(_raw).hexdigest()
    _manifest = json.loads(_raw)
    _prior = json.loads(PRIOR_EXCLUSION_SET.read_bytes())
    _excluded_images = {str(v) for v in _prior.get("image_ids", [])}
    _excluded_groups = {str(v) for v in _prior.get("group_ids", [])}

    _spent = {}
    for _name, (_dir, _report_name, _pin) in COMPLETED_CAUSAL_RUNS.items():
        _completed = json.loads((_dir / _report_name).read_text(encoding="utf-8"))
        if _completed.get("report_checksum") != _pin:
            raise RuntimeError(
                f"{_name}: {_dir / _report_name} does not match its pinned report "
                "checksum; refusing to guess which photographs it spent"
            )
        _images, _groups = set(), set()
        for _path in sorted((_dir / "units" / "capability").glob("*.json")):
            _row = json.loads(_path.read_text(encoding="utf-8"))
            _pl = _row.get("payload") if isinstance(_row.get("payload"), dict) else _row
            if _pl.get("image_id"):
                _images.add(str(_pl["image_id"]))
            if _pl.get("group_id"):
                _groups.add(str(_pl["group_id"]))
        if not _images:
            raise RuntimeError(f"{_name}: no capability units found in {_dir}")
        _spent[_name] = {
            "report_checksum": _pin, "n_images": len(_images),
            "image_ids": sorted(_images), "group_ids": sorted(_groups),
        }
        _excluded_images |= _images
        _excluded_groups |= _groups
        print(f"  excluded {len(_images):3d} images spent by {_name}")

    EXCLUSION = {
        "prior_exclusion_set": str(PRIOR_EXCLUSION_SET),
        "prior_exclusion_checksum": "sha256:" + __import__("hashlib").sha256(
            PRIOR_EXCLUSION_SET.read_bytes()
        ).hexdigest(),
        "completed_causal_runs": _spent,
        "n_excluded_images": len(_excluded_images),
        "n_excluded_groups": len(_excluded_groups),
    }
    EXCLUSION["exclusion_digest"] = payload_checksum(EXCLUSION)
    print("  exclusion digest", EXCLUSION["exclusion_digest"])

    _eligible = [
        row for row in _manifest["groups"]
        if str(row.get("image_id")) not in _excluded_images
        and str(row.get("group_id")) not in _excluded_groups
    ]
    POPULATION = hidden_animal_population(
        _eligible,
        concept_names=PAIR_CONCEPTS,
        evidence_config=EvidenceConfig(
            lexicon={name: (name,) for name in POPULATION_CONCEPTS},
            coco_categories={name: (name,) for name in POPULATION_CONCEPTS},
            require_visual_evidence=True, require_caption_evidence=False,
        ),
        images_per_concept=CANDIDATE_IMAGES_PER_CONCEPT,
        seed=SELECTION_SEED,
    )
    del _manifest, _raw, _eligible
    _chosen_images = {str(row["image_id"]) for row in POPULATION["groups"]}
    _chosen_groups = {str(row["group_id"]) for row in POPULATION["groups"]}
    if _chosen_images & _excluded_images or _chosen_groups & _excluded_groups:
        raise RuntimeError("the new population overlaps the exclusion set")
    if len(_chosen_groups) != len(_chosen_images):
        raise RuntimeError(
            "more than one synchronized group was kept for a photograph; one "
            "photograph is the independent unit"
        )
    EXCLUSION["zero_overlap_proof"] = {
        "n_population_images": len(_chosen_images),
        "n_population_groups": len(_chosen_groups),
        "overlap_with_excluded_images": sorted(_chosen_images & _excluded_images),
        "overlap_with_excluded_groups": sorted(_chosen_groups & _excluded_groups),
        "one_group_per_photograph": True,
    }
    print("population digest", POPULATION["population_digest"])
    print("population images", POPULATION["n_distinct_images"],
          " zero overlap with any completed causal run")

    DIRECTED_PAIRS = []
    for _source, _target in (PAIR_CONCEPTS, tuple(reversed(PAIR_CONCEPTS))):
        _contrast = assert_property_contrast(_source, _target)
        DIRECTED_PAIRS.append({
            "source": _source, "target": _target,
            "source_property_value": _contrast["source_value"],
            "target_property_value": _contrast["target_value"],
        })

    MEDIA = drive_media_loaders(journal=RetryJournal())
    IDENTITY_CANDIDATES = PAIR_CONCEPTS
    PROPERTY_CANDIDATES = ("two", "four")
    CANDIDATE_IDS = {
        "identity": {n: BACKEND.encode_candidate(f" {n}") for n in IDENTITY_CANDIDATES},
        "property": {n: BACKEND.encode_candidate(f" {n}") for n in PROPERTY_CANDIDATES},
    }

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
        protocol = OPEN_ANIMAL_IDENTIFICATION if readout == "identity" else HIDDEN_ANIMAL_LEGS
        candidates = IDENTITY_CANDIDATES if readout == "identity" else PROPERTY_CANDIDATES
        built = build_protocol_prompt(
            protocol=protocol, evidence=evidence, external_candidates=candidates,
            source=concept_spec(source), target=concept_spec(target),
            encode_candidate=BACKEND.encode_candidate,
        )
        # The transcript is passed only so build_backend_inputs can prove it is
        # absent from every backend argument. It never reaches the model.
        offline = group["caption"] if modality != "text" else None
        return built, build_backend_inputs(BACKEND, built, transcript=offline)

    _probe_group = POPULATION["groups"][0]
    PROMPT_PROTOCOLS = []
    for _readout, _protocol in (
        ("identity", OPEN_ANIMAL_IDENTIFICATION), ("property", HIDDEN_ANIMAL_LEGS)
    ):
        _built, _ = make_input(
            _probe_group, "text", _probe_group["concept"],
            next(n for n in PAIR_CONCEPTS if n != _probe_group["concept"]), _readout,
        )
        PROMPT_PROTOCOLS.append({
            "readout": _readout,
            "protocol": _protocol,
            **prompt_protocol_fingerprint(
                _built,
                model_revision=_bundle.model_revision,
                processor_revision=_bundle.processor_revision,
                audio_protocol_fingerprint=AUDIO_RECORD["protocol_fingerprint"],
            ),
        })
        assert_open_prompt_protocol(PROMPT_PROTOCOLS[-1])

    FINGERPRINT_CONFIG = followup_fingerprint(
        design=DESIGN,
        preflight=PREFLIGHT,
        lens_checksums=LENS_CHECKSUMS,
        model_repo_id=MODEL_REPO_ID,
        model_revision=_bundle.model_revision,
        processor_revision=_bundle.processor_revision,
        transformers_version=TRANSFORMERS_VERSION_EXPECTED,
        audio_protocol_fingerprint=AUDIO_RECORD["protocol_fingerprint"],
        prompt_protocol=PROMPT_PROTOCOLS,
        candidate_token_ids=CANDIDATE_IDS,
        directed_pairs=DIRECTED_PAIRS,
        population=POPULATION,
        exclusion=EXCLUSION,
        thresholds=THRESHOLDS.to_dict(),
        seeds={
            "selection_seed": SELECTION_SEED,
            "random_control_seed_intermediate": RANDOM_CONTROL_SEED_INTERMEDIATE,
            "random_control_seed_answer": RANDOM_CONTROL_SEED_ANSWER,
            "bootstrap_seed": THRESHOLDS.bootstrap_seed,
        },
        readout_arms=READOUT_ARMS,
        coordinate_swap_method_version=METHOD_VERSION,
        scoring_rule=(
            "teacher-forced complete-sequence scoring of the predeclared external "
            "candidates; success is the target answer being top-1"
        ),
    )
    _fingerprint = RunFingerprint(
        mode=FOLLOWUP_INTERVENTION_FAMILY,
        model_repo_id=MODEL_REPO_ID,
        model_revision=_bundle.model_revision,
        processor_revision=_bundle.processor_revision,
        layers=tuple(PRIMARY_BAND.layers),
        lens_checksum=FINGERPRINT_CONFIG["followup_fingerprint_digest"],
        manifest_checksum=MANIFEST_FILE_CHECKSUM,
        split_id=SELECTION_SEED,
        intervention_config=FINGERPRINT_CONFIG,
        selection_config={
            "population_digest": POPULATION["population_digest"],
            "exclusion_digest": EXCLUSION["exclusion_digest"],
            "pair_concepts": list(PAIR_CONCEPTS),
            "control_concepts": list(CONTROL_CONCEPTS),
            "modalities": list(MODALITIES),
        },
        extra={
            "study": FOLLOWUP_STUDY_NAME,
            "protocol_version": FOLLOWUP_PROTOCOL_VERSION,
            "design_digest": DESIGN["design_digest"],
            "corrected_report_checksum": PREFLIGHT["corrected_report_checksum"],
            "original_l32_l40_verdict": ORIGINAL_VERDICT,
            "completed_runs_read": "read-only",
        },
    )
    SWAP_RUN_DIR = FOLLOWUP_RUN_ROOT / f"band3340_real_{_fingerprint.digest.split(':')[1][:12]}"
    for _protected in PROTECTED_RUN_DIRS:
        if _protected == SWAP_RUN_DIR or _protected in SWAP_RUN_DIR.parents:
            raise RuntimeError(
                f"{SWAP_RUN_DIR} is inside the completed run {_protected}; this "
                "study never writes into completed calibration, validation or "
                "causal evidence"
            )
    SWAP_STORE = UnitStore(SWAP_RUN_DIR, _fingerprint)
    print("follow-up run", SWAP_RUN_DIR)
    print("run state    ", SWAP_STORE.open())

    # --- clean behavioural screen, before any causal spending
    _computed = _reused = 0
    for group in POPULATION["groups"]:
        source = group["concept"]
        target = next(name for name in PAIR_CONCEPTS if name != source)
        for modality in MODALITIES:
            for readout in READOUT_ARMS:
                key = safe_key("b3340-clean", group["group_id"], modality, readout)
                if SWAP_STORE.has("capability", key):
                    _reused += 1
                    continue
                source_answer = (
                    source if readout == "identity"
                    else leg_count_surfaces(resolve_leg_count(source))[0]
                )
                try:
                    built, inputs = make_input(group, modality, source, target, readout)
                except PromptLeakageError as error:
                    SWAP_STORE.save("capability", key, {
                        "status": "leakage_rejected", "group_id": group["group_id"],
                        "image_id": group["image_id"], "source": source, "target": target,
                        "modality": modality, "readout": readout,
                        "source_answer": source_answer, "prediction": None,
                        "correct": False, "rejection": str(error),
                    })
                    _computed += 1
                    continue
                scores = score_candidate_sequences(BACKEND, inputs, CANDIDATE_IDS[readout])
                verdict = prediction_and_margin(scores, source_answer)
                SWAP_STORE.save("capability", key, {
                    "group_id": group["group_id"], "image_id": group["image_id"],
                    "source": source, "target": target, "modality": modality,
                    "readout": readout, "source_answer": source_answer,
                    "prediction": verdict["prediction"], "correct": verdict["correct"],
                    "scores": scores, "prompt_len": inputs.prompt_len,
                    "prompt": built.to_dict(),
                    # The recording/photograph this cell was actually scored on,
                    # by the bytes the processor saw rather than by its path.
                    "media_reference": (
                        group["audio_path"] if modality == "spoken_audio"
                        else group["image_path"] if modality == "image" else None
                    ),
                    "media_checksum": inputs.media_checksum,
                })
                _computed += 1
                if (_computed + _reused) % 50 == 0:
                    print(f"clean screen {_computed:,} computed  {_reused:,} reused")
    CLEAN_UNITS = list(SWAP_STORE.load_all("capability").values())
    MEDIA_CHECKSUMS = {
        f"{row['group_id']}|{row['modality']}": {
            "image_id": row["image_id"],
            "media_reference": row.get("media_reference"),
            "media_checksum": row.get("media_checksum"),
        }
        for row in CLEAN_UNITS
        if row.get("media_checksum")
    }
    # Recorded, never bound into the fingerprint: the checksums are discovered
    # while the screen runs, and a fingerprint that could only be computed after
    # the screen would make the run unresumable.
    SWAP_STORE.save("metric", "followup_media_checksums", MEDIA_CHECKSUMS)
    print("media checksums recorded for", len(MEDIA_CHECKSUMS), "photograph/recording cells")
    CAUSAL_SELECTION = select_capability_eligible_samples(
        CLEAN_UNITS, concepts=PAIR_CONCEPTS, modalities=MODALITIES,
        max_images_per_cell=MAX_ANALYSIS_IMAGES_PER_CELL,
        min_images_per_cell=MIN_ANALYSIS_IMAGES_PER_CELL, seed=SELECTION_SEED,
    )
    SWAP_STORE.save("metric", "followup_capability_selection", CAUSAL_SELECTION)
    print(f"clean screen: {_computed} computed, {_reused} reused")
    for row in CAUSAL_SELECTION["cells"]:
        print(f"  {row['concept']:6s} {row['modality']:12s} "
              f"eligible={row['n_eligible']:2d} selected={row['n_selected']:2d} "
              f"sufficient={row['sufficient']}")
    CAPABILITY_SUFFICIENT = bool(CAUSAL_SELECTION["all_cells_sufficient"])
    CAUSAL_STAGE_RAN = True
    print("  all cells sufficient", CAPABILITY_SUFFICIENT)

    if not CAPABILITY_SUFFICIENT:
        print()
        print("CAUSAL STAGE STOPPED: clean capability was insufficient.")
        print("This is not a null causal result and is never reported as one.")
        print("No concept and no sample is replaced to make a cell sufficient.")
    else:
        _clean_by_key = {
            (r["group_id"], r["modality"], r["readout"]): r for r in CLEAN_UNITS
        }
        _selected = {k: set(v) for k, v in CAUSAL_SELECTION["selected_group_ids"].items()}
        _computed = _reused = 0
        for group in POPULATION["groups"]:
            source = group["concept"]
            target = next(name for name in PAIR_CONCEPTS if name != source)
            source_property = leg_count_surfaces(resolve_leg_count(source))[0]
            target_property = leg_count_surfaces(resolve_leg_count(target))[0]
            for modality in MODALITIES:
                if group["group_id"] not in _selected[f"{source}|{modality}"]:
                    continue
                evidence = None
                built_by_readout, inputs_by_readout = {}, {}
                for band in SUFFIX_BANDS:
                    layers = tuple(band.layers)
                    banks = {
                        "intermediate": selected_bases(layers, source, target),
                        "answer": selected_bases(
                            layers,
                            "two" if resolve_leg_count(source) == 2 else "four",
                            "two" if resolve_leg_count(target) == 2 else "four",
                        ),
                        "unrelated": selected_bases(layers, *CONTROL_CONCEPTS),
                    }
                    banks["random_intermediate"] = {
                        layer: random_two_direction_basis(
                            basis, seed=RANDOM_CONTROL_SEED_INTERMEDIATE + layer + band.start
                        )
                        for layer, basis in banks["intermediate"].items()
                    }
                    banks["random_answer"] = {
                        layer: random_two_direction_basis(
                            basis, seed=RANDOM_CONTROL_SEED_ANSWER + layer + band.start
                        )
                        for layer, basis in banks["answer"].items()
                    }
                    for arm in ("intermediate", "answer"):
                        for condition in BAND_CONDITIONS:
                            for readout in READOUT_ARMS:
                                key = safe_key(
                                    "b3340-swap", group["group_id"], modality,
                                    band_key(layers), arm, condition, readout,
                                )
                                if SWAP_STORE.has("intervention", key):
                                    _reused += 1
                                    continue
                                if evidence is None:
                                    evidence = load_evidence(group, modality)
                                if readout not in inputs_by_readout:
                                    _built, _inputs = make_input(
                                        group, modality, source, target, readout,
                                        evidence=evidence,
                                    )
                                    built_by_readout[readout] = _built
                                    inputs_by_readout[readout] = _inputs
                                inputs = inputs_by_readout[readout]
                                clean = _clean_by_key[(group["group_id"], modality, readout)]
                                target_answer = target if readout == "identity" else target_property
                                source_answer = source if readout == "identity" else source_property
                                if condition.startswith("unrelated_"):
                                    bases = banks["unrelated"]
                                elif condition.startswith("random_"):
                                    bases = banks[f"random_{arm}"]
                                else:
                                    bases = banks[arm]
                                result = run_swap_condition(
                                    BACKEND, inputs, bases=bases,
                                    alpha=CONDITION_ALPHA[condition],
                                    candidate_ids=CANDIDATE_IDS[readout],
                                    target_concept=target_answer,
                                    clean_scores=clean["scores"],
                                    position_rule=POSITION_RULE,
                                    record_coordinates=False,
                                )
                                HOOK_INTEGRITY = assert_band_hook_integrity(
                                    result, band=layers, prompt_len=inputs.prompt_len,
                                    expected_forward_passes=len(CANDIDATE_IDS[readout]),
                                )
                                SWAP_STORE.save("intervention", key, band_trial_record(
                                    result, band=layers, arm=arm, condition=condition,
                                    modality=modality, source=source, target=target,
                                    source_answer=source_answer, target_answer=target_answer,
                                    readout=readout, group_id=group["group_id"],
                                    image_id=group["image_id"],
                                    prompt_hash=built_by_readout[readout].prompt_hash,
                                ))
                                _computed += 1
                                if _computed % 50 == 0 or _computed == 1:
                                    print(f"band trials {_computed:,} computed  {_reused:,} reused")
        print("band swap complete", {"computed": _computed, "reused": _reused})
        SWAP_STORE.save("metric", "followup_hook_integrity", HOOK_INTEGRITY or {})
    BAND_RECORDS = [
        row for row in SWAP_STORE.load_all("intervention").values()
        if row.get("status") == "complete"
    ]
else:
    print("skipped: RUN_CAUSAL_GPU is False")
'''
)

markdown(
    """
## 8. MOCK — stage B against the synthetic world

The same `run_swap_condition`, the same `band_trial_record`, the same conditions
and the same aggregation, against `jlens.mmpilot.coordinate_swap_mock`: evidence
injected at layer 0, carry blocks, a broadcast at layer 4, and a reasoning layer
at 5 that computes the legs answer from whichever identity coordinates reach it.

Read the identity numbers with the parity note in mind: this world's carry
blocks nearly commute with the exchange, so an even-length band lands the
*identity* readout almost back on the source. The downstream answer does not,
because it is written once at the reasoning layer. That is precisely why
identity is a diagnostic and the downstream answer is the endpoint.

`assert_band_hook_integrity` runs here too, against the real hook records the
real synthetic forward pass produced.
"""
)
code(
    r'''
if not REAL_MODE:
    from jlens.mmpilot.band_swap_mock import mock_band_grid, run_mock_band_trials
    from jlens.mmpilot.coordinate_swap_mock import SwapMockBackend
    from jlens.mmpilot.validated_band_followup import assert_band_hook_integrity

    MOCK_BACKEND = SwapMockBackend()
    BAND_RECORDS = run_mock_band_trials(
        MOCK_BACKEND, modalities=MODALITIES, n_images=4
    )
    BAND_KEYS = tuple(band_key(band) for band in mock_band_grid())
    DIRECTED_PAIRS = [{"source": "bird", "target": "cat"}]
    CAPABILITY_SUFFICIENT = True
    CAUSAL_STAGE_RAN = True
    print("mock band trials", len(BAND_RECORDS))
    print("mock bands       ", list(BAND_KEYS))
    print("  (the MOCK world is 8 layers deep; the real bands are L33-L40)")

    # One real hook-integrity check over a real synthetic forward pass.
    from jlens.mmpilot.coordinate_swap import run_swap_condition as _run_swap
    from jlens.mmpilot.coordinate_swap_mock import (
        PROPERTY_CANDIDATES as _MOCK_PROPS, PROPERTY_QUESTION as _MOCK_Q,
        mock_bases, mock_concept_tokens,
    )
    from jlens.mmpilot.capability import score_candidate_sequences as _score
    _tokens = mock_concept_tokens(MOCK_BACKEND)
    _band = mock_band_grid()[0]
    _cands = {n: MOCK_BACKEND.encode_candidate(f" {n}") for n in _MOCK_PROPS}
    _inputs = MOCK_BACKEND.build_inputs(
        prompt=_MOCK_Q, modality="text", concept="bird", nuisance_key="hook-check"
    )
    _clean = _score(MOCK_BACKEND, _inputs, _cands)
    _result = _run_swap(
        MOCK_BACKEND, _inputs,
        bases=mock_bases(MOCK_BACKEND.world, layers=_band,
                         source=_tokens["bird"], target=_tokens["cat"]),
        alpha=1.0, candidate_ids=_cands, target_concept="four",
        clean_scores=_clean, record_coordinates=False,
    )
    MOCK_HOOK_INTEGRITY = assert_band_hook_integrity(
        _result, band=_band, prompt_len=_inputs.prompt_len,
        expected_forward_passes=len(_cands),
    )
    print("hook integrity   ", MOCK_HOOK_INTEGRITY)
else:
    print("skipped: real mode")
'''
)

markdown(
    """
## 9. MOCK — the commissioned causal cases

Six cases, each with a bounded verdict predeclared in
`jlens.mmpilot.validated_band_followup_mock.CAUSAL_SCENARIOS`. The records go
through the real `band_trial_record`, the real `summarize_band_cells`, the real
`band_reasoning_verdict` and the real re-labelling.

| case | required verdict |
|---|---|
| `favorable` | GO, and timing **inconclusive** — a favourable endpoint licenses no ordering claim on its own |
| `null` | NULL |
| `control_failure` | NULL — a primary rate its own matched controls match is not a result |
| `alpha2_only` | ALPHA2_SENSITIVITY_ONLY — never promoted to the alpha=1 primary |
| `asymmetric_direction` | GO, with the intermediate arm consumed earlier in **one direction only** |
| `capability_no_go` | CAPABILITY_NO_GO — no causal trial ran, so this is not a null result |

**A green MOCK run is evidence about this code and about nothing else.**
"""
)
code(
    r'''
CAUSAL_MOCK_RESULTS = {}
if not REAL_MODE:
    from jlens.mmpilot.band_swap import band_reasoning_verdict as _reasoning
    from jlens.mmpilot.band_swap import summarize_band_cells as _summarize
    from jlens.mmpilot.validated_band_followup import (
        followup_onset_timing as _timing, followup_verdict as _verdict,
    )
    from jlens.mmpilot.validated_band_followup_mock import (
        CAUSAL_SCENARIOS, MOCK_DIRECTED_PAIRS, MOCK_MODALITIES, mock_band_keys,
        mock_followup_records,
    )

    _keys = mock_band_keys()
    print(f"  {'case':<24} {'verdict':<56} {'as required':>11}")
    for _key, _scenario in CAUSAL_SCENARIOS.items():
        _records = mock_followup_records(_scenario)
        if _records:
            _cells = _summarize(_records, thresholds=THRESHOLDS)
            _reason = _reasoning(
                _cells, bands=_keys, directed_pairs=MOCK_DIRECTED_PAIRS,
                modalities=MOCK_MODALITIES,
                paper_comparable_modality=PAPER_COMPARABLE_MODALITY,
                thresholds=THRESHOLDS,
            )
        else:
            _reason = None
        _v = _verdict(
            _reason, capability_sufficient=_scenario.capability_sufficient
        )
        _t = (
            _timing(
                _reason, bands=_keys, directed_pairs=MOCK_DIRECTED_PAIRS,
                modalities=MOCK_MODALITIES, condition="swap_alpha1",
                modality=PAPER_COMPARABLE_MODALITY,
            )
            if _reason is not None
            else None
        )
        CAUSAL_MOCK_RESULTS[_key] = {
            "verdict": _v["verdict"],
            "expected_verdict": _scenario.expected_verdict,
            "timing": None if _t is None else _t["verdict"],
            "expected_timing": _scenario.expected_timing,
            "per_direction": None if _t is None else _t["per_direction"],
            "as_required": (
                _v["verdict"] == _scenario.expected_verdict
                and (None if _t is None else _t["verdict"]) == _scenario.expected_timing
            ),
        }
        _row = CAUSAL_MOCK_RESULTS[_key]
        print(f"  {_key:<24} {_row['verdict']:<56} {str(_row['as_required']):>11}")
    assert all(row["as_required"] for row in CAUSAL_MOCK_RESULTS.values())

    print()
    print("asymmetric_direction, each direction reported before any pooled summary:")
    for _row in CAUSAL_MOCK_RESULTS["asymmetric_direction"]["per_direction"]:
        print(f"  {_row['pair']:<12} intermediate deepest="
              f"{_row['intermediate_deepest_effective_start']}  answer deepest="
              f"{_row['answer_deepest_effective_start']}  "
              f"licensed={_row['licensed_separation']}")
    print()
    print("control_failure is NULL, not a reported rate:",
          CAUSAL_MOCK_RESULTS["control_failure"]["verdict"])
    print("capability_no_go is not a null causal result:",
          CAUSAL_MOCK_RESULTS["capability_no_go"]["verdict"])
else:
    print("skipped: real mode")
'''
)

markdown(
    """
## 10. Stage C — aggregate and judge

Image-level aggregation with a seeded bootstrap interval, then the verdict.

* **Primary:** the fraction of trials where the target-appropriate downstream
  answer is top-1, at alpha=1, beating every intensity-matched control.
* **Sensitivity:** alpha=2, compared against alpha=2-matched controls only, and
  reported separately. It is never interchangeable with the alpha=1 evidence.
* **Secondary:** target rank, log-prob, margin and the bootstrap interval. A
  positive margin that does not reach top-1 is `partial_movement_not_top1` and
  is never counted as a success.
* **Diagnostic:** identity replacement. It cannot produce a GO on its own.
* Text-only is the paper-comparable result; image and spoken audio are reported
  separately, and the tri-modal conjunction is labelled as our extension.
  SpokenCOCO tests linguistic spoken captions, not environmental sound.
"""
)
code(
    r'''
from jlens.mmpilot.band_swap import band_reasoning_verdict, summarize_band_cells
from jlens.mmpilot.validated_band_followup import (
    followup_verdict, read_followup_units,
)

CELLS, REASONING, VERDICT, REPORT_CONTEXT = [], None, None, None

if RUN_REPORT_CPU and not BAND_RECORDS and REPORT_RUN_DIR:
    # Stage C on its own: re-read a finished run's units, checksum by checksum,
    # with no model, no processor and no media. The bands and directed pairs
    # come from that run's report, so the analysis is over the design that was
    # frozen rather than over whatever is on disk.
    BAND_RECORDS, REPORT_CONTEXT = read_followup_units(REPORT_RUN_DIR)
    BAND_KEYS = tuple(REPORT_CONTEXT["band_keys"])
    DIRECTED_PAIRS = list(REPORT_CONTEXT["directed_pairs"])
    CAPABILITY_SUFFICIENT = bool(REPORT_CONTEXT["capability_sufficient"])
    CAUSAL_STAGE_RAN = True
    print("stage C re-analysis of", REPORT_CONTEXT["run_dir"])
    print("  complete units", REPORT_CONTEXT["n_units"],
          " invalid", REPORT_CONTEXT["n_invalid_units"])
    print("  bands         ", list(BAND_KEYS))

if BAND_RECORDS:
    CELLS = summarize_band_cells(BAND_RECORDS, thresholds=THRESHOLDS)
    REASONING = band_reasoning_verdict(
        CELLS, bands=BAND_KEYS, directed_pairs=DIRECTED_PAIRS,
        modalities=MODALITIES,
        paper_comparable_modality=PAPER_COMPARABLE_MODALITY,
        thresholds=THRESHOLDS,
    )
VERDICT = followup_verdict(
    REASONING,
    causal_stage_ran=bool(CAUSAL_STAGE_RAN),
    capability_sufficient=bool(CAPABILITY_SUFFICIENT and REASONING is not None),
    capability_selection=CAUSAL_SELECTION,
)
print("=" * 78)
print("VERDICT", VERDICT["verdict"])
print("=" * 78)
if REASONING is not None:
    print("alpha=1 primary passing bands   :",
          REASONING["paper_comparable"]["passing_bands"])
    print("alpha=2 sensitivity passing bands:",
          REASONING["alpha2_sensitivity"]["passing_bands"],
          " (sensitivity evidence, not primary)")
    print("tri-modal extension passing bands:",
          REASONING["modality_extension"]["tri_modal_passing_bands"])
    print()
    print(f"{'band':>6} {'arm':<13} {'condition':<12} {'modality':<13} "
          f"{'top1':>6} {'95% CI':>13} {'rank':>6} {'margin':>9} {'partial':>8}")
    for cell in CELLS:
        if cell["readout"] != "property" or cell["condition"] not in ("swap_alpha1", "swap_alpha2"):
            continue
        interval = cell["target_top1_bootstrap"]
        ci = f"[{interval['lower']:.2f},{interval['upper']:.2f}]"
        rank = "-" if cell["mean_target_rank"] is None else f"{cell['mean_target_rank']:.2f}"
        print(f"{cell['band_key']:>6} {cell['arm']:<13} {cell['condition']:<12} "
              f"{cell['modality']:<13} {cell['target_top1_rate']:>6.2f} {ci:>13} "
              f"{rank:>6} {cell['mean_target_margin_change']:>+9.3f} "
              f"{cell['partial_movement_rate']:>8.2f}")
else:
    print(VERDICT.get("why", "no band trials to aggregate"))
'''
)

markdown(
    """
## 11. Stage C — intermediate versus answer, over the four frozen suffix starts

The same operator, twice: once exchanging the inferred animal's coordinates and
once exchanging the leg-count answer's, over the identical bands, with the
identical controls, and **direction-matched** — the same `source -> target` pair
has to carry both arms.

Each direction is reported on its own **before** any pooled summary. Two
statistics per arm, both printed:

* `earliest_effective_start` — the first band start that works. Over nested
  suffix bands this is monotone-degenerate for an arm carried by the deepest
  layers, which is why it is not the classification statistic.
* `deepest_effective_start` — the last band start that still works.

The suffix bands are nested by construction, and **nesting alone licenses no
monotonicity claim**. The observed effective starts and their controls are
reported exactly as measured, and when no licensed separation exists the verdict
is an explicit inconclusive. The discarded native direct-readout convergence
gate is not used.
"""
)
code(
    r'''
from jlens.mmpilot.validated_band_followup import followup_onset_timing

TIMING = None
if REASONING is not None and (RUN_REPORT_CPU or not REAL_MODE):
    TIMING = followup_onset_timing(
        REASONING, bands=BAND_KEYS, directed_pairs=DIRECTED_PAIRS,
        modalities=MODALITIES, condition="swap_alpha1",
        modality=PAPER_COMPARABLE_MODALITY,
    )
    print("=" * 78)
    print("TIMING", TIMING["verdict"])
    print("=" * 78)
    print("per direction, before any pooled summary:")
    for row in TIMING["per_direction"]:
        print(f"  {row['pair']}")
        print(f"    intermediate effective starts {row['intermediate_effective_starts']}")
        print(f"    answer effective starts       {row['answer_effective_starts']}")
        print(f"    intermediate earliest/deepest {row['intermediate_earliest_effective_start']}"
              f" / {row['intermediate_deepest_effective_start']}")
        print(f"    answer earliest/deepest       {row['answer_earliest_effective_start']}"
              f" / {row['answer_deepest_effective_start']}")
        print(f"    licensed separation           {row['licensed_separation']}")
    print()
    print("pooled:", TIMING["pooled_summary"]["classifications"])
    print()
    print(" ", TIMING["monotonicity_not_asserted_from_nesting"])
    print(" ", TIMING["why_not_earliest"])
    print("  band starts are not exact physical onsets:",
          TIMING["band_starts_are_not_exact_physical_onsets"])
    print("  native direct-readout convergence gate used:",
          TIMING["native_direct_readout_convergence_gate_used"])
elif REASONING is not None:
    print("skipped: set RUN_REPORT_CPU=True for the CPU-only timing stage")
else:
    print("skipped: no reasoning verdict")
'''
)

markdown(
    """
## 12. Report, resume status, and what to send back
"""
)
code(
    r'''
from jlens.mmpilot.store import payload_checksum
from jlens.mmpilot.validated_band_followup import (
    FOLLOWUP_REPORT_NAME, followup_report, format_followup_verdict,
)

REPORT = followup_report(
    mode=MODE,
    preflight=PREFLIGHT or {},
    design=DESIGN or {},
    fingerprint=FINGERPRINT_CONFIG,
    population=POPULATION,
    exclusion=EXCLUSION,
    media_checksums=MEDIA_CHECKSUMS,
    capability_selection=CAUSAL_SELECTION,
    capability_sufficient=bool(CAUSAL_STAGE_RAN and CAPABILITY_SUFFICIENT),
    directed_pairs=DIRECTED_PAIRS,
    band_keys=list(BAND_KEYS),
    thresholds=THRESHOLDS.to_dict(),
    cells=CELLS,
    reasoning=REASONING,
    verdict=VERDICT,
    timing=TIMING,
    budget=PASS_BUDGET,
    hook_integrity=HOOK_INTEGRITY,
    fitting_audit=FITTING_AUDIT,
    immutability=IMMUTABILITY,
    resume=None if SWAP_STORE is None else SWAP_STORE.status_report(),
    run_dir=str(SWAP_RUN_DIR) if SWAP_RUN_DIR else None,
    reanalysis_of=REPORT_CONTEXT,
)
if SWAP_STORE is not None:
    _path = SWAP_RUN_DIR / FOLLOWUP_REPORT_NAME
    _tmp = _path.with_suffix(".json.tmp")
    _tmp.write_text(json.dumps(REPORT, indent=2, default=str), encoding="utf-8")
    os.replace(_tmp, _path)
    SWAP_STORE.save("metric", "l33_l40_validated_band_followup_report", REPORT)
    print("report", _path)
print("report checksum", REPORT["report_checksum"])
print()
print(format_followup_verdict(VERDICT, TIMING))

if SWAP_STORE is not None:
    print()
    print("resume status")
    print(json.dumps(SWAP_STORE.status_report(), indent=2, default=str))

print()
if MODE == "real":
    print("Send back:")
    print(" ", FOLLOWUP_REPORT_NAME, "  (stages A-C)")
    print("  the printed REPORTING BOUNDARY block")
    print("  the printed PREFLIGHT block, including the eight artifact checksums")
    print("  the printed PASS BUDGET block")
    print("  the printed VERDICT table and the TIMING per-direction block")
    print("  the resume status above")
    print()
    print("The completed corrected validation in", ORIGINAL_RUN_NAME, "is unchanged")
    print("and stays where it is. The originally planned L32-L40 study remains")
    print(" ", ORIGINAL_VERDICT + ". This is a prospective causal follow-up over the")
    print("validated band L33-L40, not that study's confirmation.")
else:
    print("MOCK RUN COMPLETE — pipeline behaviour only.")
    print("No scientific claim about Gemma 4, about layers 33-40, about L32,")
    print("about any modality, or about the workspace hypothesis is made or")
    print("implied. A green MOCK run is evidence about this code and about")
    print("nothing else.")
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
