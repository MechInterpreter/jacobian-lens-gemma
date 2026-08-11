"""Generate the paper-faithful contiguous-band coordinate-swap Colab notebook.

Deterministic: running this script twice writes byte-identical output, and
``tests/test_anthropic_band_swap_notebook.py`` fails if the checked-in notebook
drifts from it.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "notebooks" / "multimodal_jspace_anthropic_band_swap_colab.ipynb"
CELLS: list[tuple[str, str]] = []


def markdown(value: str) -> None:
    CELLS.append(("markdown", value.strip("\n")))


def code(value: str) -> None:
    CELLS.append(("code", value.strip("\n")))


markdown(
    r"""
# Anthropic's contiguous-band J-lens coordinate swap — Gemma 4 E4B

The paper's **primary** swap protocol, run as the paper runs it:

`c = pinv(V) h`, `h_patched = h + alpha * V (sigma(c) - c)`

* applied at **every original prompt position**,
* at **every physical layer of a contiguous band**, all hooks installed at once,
* with the coordinates **recomputed at each layer** from that layer's own
  activation and its own `W_U @ J_l`,
* never at a teacher-forced candidate token,
* `alpha = 1` as the primary exact exchange and `alpha = 2` as a separately
  labelled fixed double-strength extrapolation — never swept per sample,
* scored by Anthropic's own trial definition: **the downstream target answer
  becomes top-1**.

## What this notebook is, and what the completed run was

The completed study
(`notebooks/multimodal_jspace_anthropic_reasoning_swap_colab.ipynb`) applied one
exchange at one confirmed layer per forward pass, on the sampled grid
32/35/38/40. Its run directories, reports and notebook are historical evidence
and **nothing here reads, rewrites or resumes them**.

Its justification for refusing a band does not carry: exact exchange is an
involution, so applying *one fixed update* twice cancels — but a band clamp is
not that. Each band layer has its own J-lens basis and re-reads its own
coordinates, so two layers cancel only insofar as their bases and coordinates
agree, which in a real transformer they do not.

What a band does require is that **every physical layer inside it** has an
independently confirmed, pinned lens at one scale. `[32, 35, 38, 40]` is four
confirmed layers, not the range 32-40, and `assert_contiguous` refuses to call
it one. So this notebook fits and confirms the five layers Gemma is missing
before it clamps anything.

## The five stages, each resumable

| stage | runtime | what it does |
|---|---|---|
| 0 | CPU | Read the completed artifacts, print the L32-L40 lens table, freeze the bands. Refuses before any model load if a pin is missing. |
| 1 | GPU | Fit the missing physical layers 33, 34, 36, 37, 39 at scale 250 on the identical frozen corpus ordering. Atomic checkpoint every 25 prompts. |
| 2 | GPU | Confirm them on **third-generation** held-out sets under the same frozen gate, and publish only what passes. |
| 3 | GPU | The band swap itself: `coordinate_swap_band`, alpha=1 primary, alpha=2 secondary, intensity-matched controls. |
| 4 | CPU | Intermediate-versus-answer timing over the predeclared suffix bands. |

Run with every switch False and the notebook executes the whole pipeline
against a synthetic CPU world — no Drive, no model, no download, no spend. **A
green MOCK run is evidence about this code and about nothing else.**
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
## 1. Configuration — the only switches

Every real switch defaults to `False`, and with all of them False the notebook
performs a complete MOCK run on CPU. Stage 1 additionally requires
`CONFIRM_FIT_BUDGET`, stage 3 additionally requires `CONFIRM_PASS_BUDGET`, and
both require `CONFIRM_MODEL_LOAD`.
"""
)
code(
    r'''
# ---- stage switches (all False = full MOCK run on CPU) -------------------
RUN_STAGE0_PREFLIGHT = False           # CPU: read Drive artifacts, freeze the design
RUN_STAGE1_FIT_MISSING_LENSES = False  # GPU: fit L33/34/36/37/39 at scale 250
RUN_STAGE2_CONFIRM_AND_PUBLISH = False # GPU: untouched confirmation + publication
RUN_STAGE3_BAND_SWAP = False           # GPU: the contiguous-band causal run
RUN_STAGE4_TIMING = False              # CPU: intermediate-vs-answer onset

# Stage 4 re-analyses a finished stage-3 run without a model or any media. Set
# this to that run directory to run it on its own; leave it None to analyse the
# run this session just produced.
STAGE4_SWAP_RUN_DIR = None

# ---- explicit spend confirmations (read sections 5 and 9 first) ----------
CONFIRM_MODEL_LOAD = False
CONFIRM_FIT_BUDGET = False
CONFIRM_PASS_BUDGET = False

REAL_MODE = any((
    RUN_STAGE0_PREFLIGHT, RUN_STAGE1_FIT_MISSING_LENSES,
    RUN_STAGE2_CONFIRM_AND_PUBLISH, RUN_STAGE3_BAND_SWAP, RUN_STAGE4_TIMING,
))
MODE = "real" if REAL_MODE else "mock"

# ---- the frozen band design ---------------------------------------------
from jlens.mmpilot.band_lens import (
    BAND_INTERIOR_LAYERS, BAND_SCALE, BAND_TARGET_LAYER, BAND_WINDOW,
)
from jlens.mmpilot.band_swap import (
    BAND_CONDITIONS, BAND_SWAP_VERSION, PRIMARY_ALPHA, SECONDARY_ALPHA,
    BandSwapThresholds,
)
from jlens.mmpilot.coordinate_swap import PRIMARY_POSITION_RULE

BAND_START_LAYERS = (32, 35, 38, 40)   # band STARTS, not a sampled layer grid
BAND_END_LAYER = BAND_WINDOW[1]
POSITION_RULE = PRIMARY_POSITION_RULE
ALPHAS = (PRIMARY_ALPHA, SECONDARY_ALPHA)

PAIR_CONCEPTS = ("bird", "cat")        # 2 legs versus 4 legs
CONTROL_CONCEPTS = ("zebra", "giraffe")
POPULATION_CONCEPTS = (*PAIR_CONCEPTS, *CONTROL_CONCEPTS)
MODALITIES = ("text", "image", "spoken_audio")
PAPER_COMPARABLE_MODALITY = "text"
CANDIDATE_IMAGES_PER_CONCEPT = 24
MAX_ANALYSIS_IMAGES_PER_CELL = 8
MIN_ANALYSIS_IMAGES_PER_CELL = 4
SELECTION_SEED = "anthropic-contiguous-band-swap-gemma-v1"

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

# ---- completed runs this notebook reads, never writes --------------------
RUNS_ROOT = Path("/content/drive/MyDrive/jacobian-lens-gemma/runs")
PARENT_CALIBRATION_RUN_DIR = RUNS_ROOT / "rgcalib_real_7e3736b4de8f"
EXTENSION_RUN_DIR = RUNS_ROOT / "rgext_real_c18f03f06e7b"
EXPANDED_MANIFEST_CACHE = (
    RUNS_ROOT / "mml32_l32_followup_20260808T182717" / "expanded_manifest.json"
)
PRIOR_EXCLUSION_SET = Path(
    "/content/drive/MyDrive/datasets/cstf_spokencoco_derived/"
    "jlens_l32_resolution_prep_v1/"
    "prep_020ebbe6f832aece5ece6cb8bee994ca/exclusion_set.json"
)
PROTECTED_RUN_DIRS = (PARENT_CALIBRATION_RUN_DIR, EXTENSION_RUN_DIR)

print("mode                        ", MODE)
print("study version               ", BAND_SWAP_VERSION)
print("band window                 ", BAND_WINDOW, "scale", BAND_SCALE)
print("band starts (contiguous)    ", BAND_START_LAYERS, "-> end", BAND_END_LAYER)
print("interior layers to fit      ", BAND_INTERIOR_LAYERS)
print("position rule               ", POSITION_RULE)
print("alphas                      ", ALPHAS, "(1 = exchange, 2 = extrapolation)")
print("conditions                  ", BAND_CONDITIONS)
print("threshold digest            ", THRESHOLDS.digest)
for name, value in (
    ("RUN_STAGE0_PREFLIGHT", RUN_STAGE0_PREFLIGHT),
    ("RUN_STAGE1_FIT_MISSING_LENSES", RUN_STAGE1_FIT_MISSING_LENSES),
    ("RUN_STAGE2_CONFIRM_AND_PUBLISH", RUN_STAGE2_CONFIRM_AND_PUBLISH),
    ("RUN_STAGE3_BAND_SWAP", RUN_STAGE3_BAND_SWAP),
    ("RUN_STAGE4_TIMING", RUN_STAGE4_TIMING),
    ("CONFIRM_MODEL_LOAD", CONFIRM_MODEL_LOAD),
    ("CONFIRM_FIT_BUDGET", CONFIRM_FIT_BUDGET),
    ("CONFIRM_PASS_BUDGET", CONFIRM_PASS_BUDGET),
):
    print(f"{name:<32} {value}")
if not REAL_MODE:
    print()
    print("MOCK RUN: no Drive, no model, no download, nothing spent.")
'''
)

markdown(
    """
## 2. Mount Drive and fail fast on every pin

Nothing is mounted and nothing is read in MOCK mode. In real mode every
configured artifact must exist **before** a model is loaded; a missing pin
stops the run here rather than three hours in.
"""
)
code(
    r'''
import json

if REAL_MODE:
    if IN_COLAB:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
    required = [EXTENSION_RUN_DIR]
    if RUN_STAGE1_FIT_MISSING_LENSES or RUN_STAGE2_CONFIRM_AND_PUBLISH:
        required.append(PARENT_CALIBRATION_RUN_DIR)
    if RUN_STAGE3_BAND_SWAP:
        required += [EXPANDED_MANIFEST_CACHE, PRIOR_EXCLUSION_SET]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "configured artifact(s) missing; refusing before any model load:\n  "
            + "\n  ".join(missing)
        )
    if (RUN_STAGE1_FIT_MISSING_LENSES or RUN_STAGE2_CONFIRM_AND_PUBLISH
            or RUN_STAGE3_BAND_SWAP):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("stages 1-3 require a GPU runtime; use L4 or A100")
    print("every configured artifact is present")
else:
    print("MOCK: no Drive mounted, no artifact read.")
'''
)

markdown(
    """
## 3. Stage 0 — the lens inventory for every physical layer 32-40

Read-only. For each physical layer in the band window this prints the lens
path, its checksum, the scale and corpus it was fitted on, the confirmation set
it was judged on, the verdict, and whether it may enter a band. Layers with no
artifact get a row too, with the reason — the point of the table is that the
gaps are visible rather than inferred.

Three chains produce it, none of which assumes a filename:

* **L32** — the early-layer extension's published scale-250 artifact, resolved
  through its own report and re-checksummed.
* **L35 / L38 / L40** — present in that run's scale-250 **snapshot** with
  recorded confirmation verdicts from the same fresh 256-prompt set. They were
  scored and not published, because the extension's publication targets were
  L26 and L32. (Their separately published artifacts are scale *100*: a
  different fit, and a band may not mix scales.)
* **L33 / L34 / L36 / L37 / L39** — this study's own published artifacts, once
  stages 1 and 2 have produced them.
"""
)
code(
    r'''
from jlens.mmpilot.band_lens import (
    BandLensRefused, BandLensSource,
    discover_extension_scale250_lenses, discover_published_band_lenses,
)
from jlens.mmpilot.band_swap import (
    BandLensRow, format_lens_inventory, lens_inventory,
)

BAND_RUN_ROOT = RUNS_ROOT / "mmband"
LENS_ROWS, LENS_SOURCES, DISCOVERY = [], {}, {}

if REAL_MODE:
    LENS_ROWS, LENS_SOURCES, DISCOVERY = discover_extension_scale250_lenses(
        EXTENSION_RUN_DIR, snapshot_layers=(35, 38, 40), published_layers=(32,),
        scale=BAND_SCALE,
    )
    _interior_dirs = sorted(BAND_RUN_ROOT.glob("bandlens_real_*")) if BAND_RUN_ROOT.is_dir() else []
    for _dir in _interior_dirs:
        try:
            _rows, _sources, _evidence = discover_published_band_lenses(
                _dir, layers=BAND_INTERIOR_LAYERS, scale=BAND_SCALE
            )
        except BandLensRefused as _error:
            print(f"  interior run {_dir.name}: {_error}")
            continue
        for _row in _rows:
            _existing = next((r for r in LENS_ROWS if r.layer == _row.layer), None)
            if (
                _row.usable
                and _existing is not None
                and _existing.usable
                and _existing.lens_checksum != _row.lens_checksum
            ):
                raise RuntimeError(
                    f"two runs publish a usable lens for layer {_row.layer} with "
                    f"different checksums ({_existing.lens_checksum} from "
                    f"{_existing.provenance}, {_row.lens_checksum} from "
                    f"{_dir.name}). Which artifact a band rests on is not decided "
                    "by directory sort order; point the notebook at one run."
                )
            if _row.usable or _existing is None:
                LENS_ROWS = [r for r in LENS_ROWS if r.layer != _row.layer] + [_row]
        LENS_SOURCES.update(_sources)
        DISCOVERY[f"interior_{_dir.name}"] = _evidence
    _mock_note = None
else:
    from jlens.mmpilot.band_swap_mock import MOCK_BAND_USABLE_LAYERS, MOCK_BAND_WINDOW
    from jlens.mmpilot.coordinate_swap_mock import MOCK_LENS_CHECKSUM
    BAND_WINDOW = MOCK_BAND_WINDOW
    LENS_ROWS = [
        BandLensRow(
            layer=_layer,
            lens_path=f"<synthetic mock world>/L{_layer}",
            lens_checksum=f"{MOCK_LENS_CHECKSUM}:L{_layer}",
            fit_scale=BAND_SCALE,
            fit_corpus="mock: one frozen synthetic dictionary at every layer",
            validation_set="mock: no held-out set exists",
            confirmation_verdict="MOCK — not a confirmation",
            usable=True,
            reason="synthetic world; proves the pipeline and nothing else",
            provenance="mock",
        )
        for _layer in MOCK_BAND_USABLE_LAYERS
    ]
    _mock_note = "MOCK inventory: every row is synthetic and says so."

INVENTORY = lens_inventory(
    LENS_ROWS, layer_range=BAND_WINDOW, required_scale=BAND_SCALE
)
print(format_lens_inventory(INVENTORY))
if _mock_note:
    print()
    print(_mock_note)
'''
)

markdown(
    """
## 4. Stage 0 — freeze the bands before any model result exists

The primary band is the full contiguous range, and the comparison bands are its
suffixes. Every physical layer inside every band must be usable or the design
is refused — bands are never trimmed to fit the lenses on hand, and the sampled
grid is never relabelled as a range.

If the interior lenses do not exist yet the refusal here is the expected
outcome: it names the missing layers, and stages 1 and 2 are what produce them.
"""
)
code(
    r'''
from jlens.mmpilot.band_swap import (
    BandDesignRefused, assert_contiguous, band_design_record, band_key,
    build_band, largest_admissible_band, predeclare_suffix_bands,
)
from jlens.mmpilot.coordinate_swap import LayerBandError

if not REAL_MODE:
    from jlens.mmpilot.band_swap_mock import MOCK_BAND_END, MOCK_BAND_STARTS
    BAND_START_LAYERS = MOCK_BAND_STARTS
    BAND_END_LAYER = MOCK_BAND_END

USABLE_LAYERS = tuple(INVENTORY["usable_layers"])
DESIGN, PRIMARY_BAND, SUFFIX_BANDS, BAND_KEYS = None, None, (), ()
BAND_DESIGN_BLOCKED = None

# The sampled grid is not a band, and this is where that is enforced.
try:
    assert_contiguous((32, 35, 38, 40), what="the completed study's sampled grid")
    raise AssertionError("unreachable: a sampled grid must never validate as a band")
except BandDesignRefused as _refusal:
    print("sanity check —", _refusal)
print()

try:
    PRIMARY_BAND = build_band(
        BAND_START_LAYERS[0], BAND_END_LAYER,
        usable_layers=USABLE_LAYERS, n_layers=EXPECT_N_LAYERS if REAL_MODE else None,
    )
    SUFFIX_BANDS = predeclare_suffix_bands(
        starts=BAND_START_LAYERS, end=BAND_END_LAYER,
        usable_layers=USABLE_LAYERS, n_layers=EXPECT_N_LAYERS if REAL_MODE else None,
    )
    BAND_KEYS = tuple(band_key(band) for band in SUFFIX_BANDS)
    DESIGN = band_design_record(
        inventory=INVENTORY,
        primary_band=PRIMARY_BAND,
        suffix_bands=SUFFIX_BANDS,
        position_rule=POSITION_RULE,
        alphas=ALPHAS,
        conditions=BAND_CONDITIONS,
    )
    print("primary band        ", list(PRIMARY_BAND.layers))
    print("predeclared bands   ", [list(band.layers) for band in SUFFIX_BANDS])
    print("band keys           ", list(BAND_KEYS))
    print("design digest       ", DESIGN["design_digest"])
except LayerBandError as _error:
    BAND_DESIGN_BLOCKED = str(_error)
    _admissible = largest_admissible_band(
        USABLE_LAYERS, low=BAND_WINDOW[0], high=BAND_WINDOW[1]
    )
    print("BAND DESIGN BLOCKED")
    print(_error)
    print()
    print("largest admissible contiguous band right now:", _admissible)
    print("missing physical layers:",
          sorted(set(range(BAND_WINDOW[0], BAND_WINDOW[1] + 1)) - set(USABLE_LAYERS)))
    print("Run stages 1 and 2 to fit and confirm them. Stage 3 stays blocked.")
'''
)

markdown(
    """
## 5. Stage 1 — the budget for fitting the missing layers

Five layers in **one** pass. The upstream estimator differentiates to every
requested source layer from a single forward and a single set of backward
passes, and it starts the graph at the shallowest requested layer — so this fit
traverses far fewer blocks per backward pass than the extension's did.

Two estimates are printed. The span-scaled one is a model of that saving; the
unscaled one applies the operator's original observation with no discount and
is the number to plan with.
"""
)
code(
    r'''
from jlens.mmpilot.band_lens import band_capture_plan, band_fit_budget, format_band_fit_budget

BAND_PLAN = band_capture_plan(
    layers=BAND_INTERIOR_LAYERS,
    target_layer=BAND_TARGET_LAYER,
    d_model=EXPECT_D_MODEL,
    n_layers=EXPECT_N_LAYERS,
)
FIT_BUDGET = band_fit_budget(plan=BAND_PLAN, scale=BAND_SCALE)
print(format_band_fit_budget(FIT_BUDGET))
print()
print("capture plan digest", BAND_PLAN.digest)
if RUN_STAGE1_FIT_MISSING_LENSES and not (CONFIRM_MODEL_LOAD and CONFIRM_FIT_BUDGET):
    raise RuntimeError(
        "stage 1 needs CONFIRM_MODEL_LOAD and CONFIRM_FIT_BUDGET set by hand"
    )
'''
)

markdown(
    """
## 6. Stage 1 — reconstruct the exact fit ordering, then fit

The extension stored split **checksums**, not corpus text, so the only way to
fit the missing layers on the same 250 prompts is to re-stream the pinned
corpus under the parent's own collection parameters, rebuild the ordering, and
**prove** it reproduces both the parent's 100-prompt prefix checksum and the
extension's 250-prompt prefix checksum. Only then does the fit start.

This is a fresh accumulator, not a continuation: the extension's checkpoint
holds `jacobian_sum` for a different layer grid, upstream refuses that resume,
and appending five new layers to it would claim 250 prompts for matrices that
saw none. `assert_corpus_equivalence` records exactly that, and refuses the run
if the extension's artifacts cannot establish the corpus.

The accumulator is written atomically every **25 prompts**; a disconnect loses
at most 25 prompts of fitting and never the verified ordering.
"""
)
code(
    r'''
from jlens.mmpilot.band_lens import (
    BAND_LENS_PROTOCOL, BAND_SPLIT_SEED, BandLensStore,
    assert_corpus_equivalence, band_scale_selection,
)

MODEL = None
BAND_STORE = None
BAND_RUN_DIR = None
FIT_RECORDS = ()
EQUIVALENCE = None
CONTINUATION = None
EXTENSION_REPORT = None
PARENT = None

if RUN_STAGE1_FIT_MISSING_LENSES or RUN_STAGE2_CONFIRM_AND_PUBLISH:
    import torch
    from jlens.calibration.corpus import build_records, collect_records_for_partition_quotas
    from jlens.calibration.extension import (
        EXTENSION_GATE, build_extension_fit_order, parent_collection_parameters,
        verify_fit_prefix, verify_reconstructed_partitions,
    )
    from jlens.calibration.fitting import filter_records_by_tokens, run_calibration
    from jlens.calibration.parent import (
        ParentRequirements, audit_parent_run, load_parent_run,
        parent_provenance_manifest, protected_parent_checksums,
    )
    from jlens.calibration.state import CalibrationFingerprint
    from jlens.gemma4 import load_gemma4, verify_architecture
    from jlens.mmpilot.l32_followup import read_extension_report
    from jlens.mmpilot.store import payload_checksum

    _report_path, EXTENSION_REPORT = read_extension_report(EXTENSION_RUN_DIR)
    EQUIVALENCE = assert_corpus_equivalence(
        extension_report=EXTENSION_REPORT,
        plan=BAND_PLAN,
        scale=BAND_SCALE,
        model_repo_id=MODEL_REPO_ID,
        model_revision=MODEL_REVISION,
    )
    print("corpus equivalence  ", EQUIVALENCE["passed"], EQUIVALENCE["equivalence_checksum"])
    print("  corpus            ", EQUIVALENCE["extension_corpus_id"],
          "@", EQUIVALENCE["extension_corpus_revision"])
    print("  250-prompt prefix ", EQUIVALENCE["extension_fit_prefix_checksum"])

    PARENT = load_parent_run(PARENT_CALIBRATION_RUN_DIR, baseline_scale=100)
    PARENT_CHECKSUMS_BEFORE = protected_parent_checksums(
        PARENT_CALIBRATION_RUN_DIR, layout=PARENT.layout
    )
    COLLECTION = parent_collection_parameters(PARENT)

    import getpass
    if not os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = getpass.getpass("HF_TOKEN (input hidden): ").strip()
    MODEL, LOAD_INFO = load_gemma4(
        MODEL_REPO_ID, revision=MODEL_REVISION, dtype=torch.bfloat16,
        device_map="cuda", allow_model_load=True, token=os.environ["HF_TOKEN"],
    )
    ARCHITECTURE = verify_architecture(
        MODEL, expect_n_layers=EXPECT_N_LAYERS, expect_d_model=EXPECT_D_MODEL,
        expect_vocab_size=EXPECT_VOCAB,
    ).to_dict()

    from datasets import load_dataset
    _corpus = dict(EXTENSION_REPORT["corpus"])
    _stream = load_dataset(
        "Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True
    )
    _records, _partitions = collect_records_for_partition_quotas(
        corpus_id=_corpus["corpus_id"],
        texts=(row["text"] for row in _stream),
        min_chars=COLLECTION["min_chars"],
        min_fit=COLLECTION["min_fit"],
        max_texts=COLLECTION["max_texts"],
        seed=COLLECTION["seed"],
        n_validation=COLLECTION["n_validation"],
        n_confirmation=COLLECTION["n_confirmation"],
    )
    RECONSTRUCTION = verify_reconstructed_partitions(_partitions, parent=PARENT)
    OLD_FIT, OLD_VALIDATION, OLD_CONFIRMATION = (
        _partitions.fit, _partitions.validation, _partitions.confirmation
    )
    # The pool and the fit ordering are rebuilt at the extension's OWN largest
    # scale, read from its report — not at this band's 250 — because that is
    # what determines the record pool it drew its held-out sets from. Getting
    # this wrong would exclude the wrong records in section 7, and section 7
    # proves the reconstruction rather than trusting it.
    LARGEST_EXTENSION_SCALE = max(int(point) for point in _corpus["scale_points"])
    _last_index = max(record.stream_index for record in _records)
    _extra_stream = load_dataset(
        "Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True
    )
    _needed = COLLECTION["max_texts"] + 40 * (LARGEST_EXTENSION_SCALE + 512)
    _texts = []
    for _index, _row in enumerate(_extra_stream):
        if _index >= _needed:
            break
        _texts.append(_row["text"])
    EXTENSION_POOL = [
        record
        for record in build_records(_corpus["corpus_id"], _texts, min_chars=COLLECTION["min_chars"])
        if record.stream_index > _last_index
    ]
    _token_count = lambda text: int(MODEL.encode(text, max_length=BAND_PLAN.max_seq_len).shape[1])
    PARENT_FIT_RECORDS, DROPPED_SHORT = filter_records_by_tokens(
        OLD_FIT, token_count=_token_count,
        skip_first=BAND_PLAN.skip_first, max_seq_len=BAND_PLAN.max_seq_len,
    )
    EXTENSION_FIT_RECORDS = build_extension_fit_order(
        PARENT_FIT_RECORDS, n_needed=LARGEST_EXTENSION_SCALE,
        extension_pool=EXTENSION_POOL,
    )
    # This band's fit list is a prefix of it: the same prompts, in the same
    # order, stopping at the scale the confirmed band layers were fitted at.
    FIT_RECORDS = EXTENSION_FIT_RECORDS[:BAND_SCALE]
    PREFIX_100 = verify_fit_prefix(
        FIT_RECORDS, n_parent=100,
        parent_prefix_checksum=PARENT.fit_prefix_checksum(100),
    )
    _recomputed_250 = payload_checksum([record.to_dict() for record in FIT_RECORDS[:BAND_SCALE]])
    if _recomputed_250 != EQUIVALENCE["extension_fit_prefix_checksum"]:
        raise RuntimeError(
            "the reconstructed 250-prompt fit ordering does not match the "
            f"extension's recorded prefix checksum (rebuilt {_recomputed_250}, "
            f"recorded {EQUIVALENCE['extension_fit_prefix_checksum']}). Refusing: "
            "these five layers would be fitted on a different 250 prompts than "
            "the layers they must sit beside."
        )
    print("largest extension scale", LARGEST_EXTENSION_SCALE)
    print("parent 100-prefix   ", PREFIX_100["matches"])
    print("extension 250-prefix", True, _recomputed_250)

    FINGERPRINT = CalibrationFingerprint(
        mode="real",
        protocol_version=BAND_LENS_PROTOCOL,
        model_repo_id=LOAD_INFO["model_repo_id"],
        model_revision=LOAD_INFO["model_revision"],
        tokenizer_revision=LOAD_INFO["tokenizer_revision"],
        capture_plan_digest=BAND_PLAN.digest,
        corpus_manifest_checksum=EQUIVALENCE["equivalence_checksum"],
        gate_digest=EXTENSION_GATE.digest,
        plateau_rule_digest="not_applicable_single_scale",
        scale_points=(BAND_SCALE,),
        artifact_format_version="jlens.calibration.artifact.v1",
        extra={
            "band_window": list(BAND_WINDOW),
            "interior_layers": list(BAND_INTERIOR_LAYERS),
            "extension_run": str(EXTENSION_RUN_DIR),
            "extension_fit_prefix_checksum": EQUIVALENCE["extension_fit_prefix_checksum"],
            "band_split_seed": BAND_SPLIT_SEED,
            "seeded_from_extension_accumulator": False,
        },
    )
    BAND_RUN_DIR = BAND_RUN_ROOT / f"bandlens_real_{FINGERPRINT.digest[7:19]}"
    BAND_STORE = BandLensStore(BAND_RUN_DIR, FINGERPRINT)
    print("band lens run       ", BAND_RUN_DIR)
    print("resume              ", BAND_STORE.open())
    BAND_STORE.save("band_preflight", "equivalence", EQUIVALENCE)
    BAND_STORE.save("band_preflight", "provenance", parent_provenance_manifest(
        PARENT,
        audit_parent_run(PARENT, requirements=ParentRequirements(
            model_repo_id=MODEL_REPO_ID, model_revision=MODEL_REVISION,
            tokenizer_repo_id=MODEL_REPO_ID, tokenizer_revision=MODEL_REVISION,
            source_layers=tuple(PARENT.accumulator.source_layers),
            target_layer=BAND_TARGET_LAYER, d_model=EXPECT_D_MODEL,
            hook_site="block_output", skip_first=BAND_PLAN.skip_first,
            max_seq_len=BAND_PLAN.max_seq_len, dim_batch=BAND_PLAN.dim_batch,
            corpus_hf_dataset="Salesforce/wikitext", corpus_config="wikitext-103-raw-v1",
            corpus_split="train",
            estimator="jlens.fitting.fit (upstream, unmodified)",
            artifact_format_version="jlens.calibration.artifact.v1",
            baseline_scale=100, expected_n_done=100,
        )),
        immutability=PARENT_CHECKSUMS_BEFORE,
        extension_protocol_version=BAND_LENS_PROTOCOL,
        extension_run_dir=str(BAND_RUN_DIR),
    ))

if RUN_STAGE1_FIT_MISSING_LENSES:
    CONTINUATION = run_calibration(
        MODEL, FIT_RECORDS, plan=BAND_PLAN, scale_points=(BAND_SCALE,),
        store=BAND_STORE, checkpoint_every=25, diagnostics_every=25,
    )
    BAND_STORE.save("band_fit", "record", {
        "protocol": BAND_LENS_PROTOCOL,
        "n_done": CONTINUATION.n_done,
        "n_skipped": CONTINUATION.n_skipped,
        "elapsed_seconds": round(CONTINUATION.elapsed_seconds, 2),
        "snapshots": {
            str(scale): snapshot.to_dict()
            for scale, snapshot in sorted(CONTINUATION.snapshots.items())
        },
        "fresh_accumulator": True,
        "seeded_from_extension_accumulator": False,
        "checkpoint_every": 25,
        "capture_plan": BAND_PLAN.to_dict(),
    })
    print("fitted prompts      ", CONTINUATION.n_done)
    for _scale, _snapshot in sorted(CONTINUATION.snapshots.items()):
        print(f"  scale {_scale}: {_snapshot.checksum}")
elif not REAL_MODE:
    print("skipped: stage 1 runs the MOCK fit in section 8")
else:
    print("skipped: RUN_STAGE1_FIT_MISSING_LENSES is False")
'''
)

markdown(
    """
## 7. Stage 2 — third-generation held-out sets, confirmation, publication

The extension's fresh development and confirmation sets have been **opened and
read**; their verdicts are on the record and were used to publish L32. So they
are development history for this run and are excluded from it, alongside the
parent's fit, development and confirmation records and every prompt in this
fit. The exclusion list is only trustworthy if the reconstruction that produced
it is proved to be the extension's own draw, which
`verify_reconstructed_extension_splits` does before anything is excluded.

The gate is the frozen one, unchanged: `EXTENSION_GATE` on development and
`EXTENSION_CONFIRMATION_GATE` on confirmation — the same rule the confirmed
band layers passed. Nothing is loosened, and the layer list and pass criteria
are fixed before the confirmation vault opens. A layer publishes only if it
passes; a band exists only if every physical layer in it does.
"""
)
code(
    r'''
from jlens.calibration.extension import (
    EXTENSION_CONFIRMATION_GATE, EXTENSION_GATE, build_fresh_evaluation_splits,
)
from jlens.mmpilot.band_lens import (
    band_layer_verdict, build_band_evaluation_splits, publish_band_layer,
    verify_reconstructed_extension_splits,
)

BAND_SPLITS = None
BAND_CONFIRMATION = None
BAND_DEVELOPMENT = None
BAND_VERDICT = None
BAND_PUBLICATION = None

if RUN_STAGE2_CONFIRM_AND_PUBLISH:
    from jlens.calibration.gate import (
        evaluate_calibration_layers, ordinary_next_token_argmax,
        select_diverse_validation_prompts,
    )
    from jlens.calibration.publication import (
        ConfirmationVault, PublicationRefused, record_failed_layer,
    )
    from jlens.controls import control_lens, distant_layer_mapping, layer_mapped_lens
    from jlens.hooks import ActivationRecorder
    from jlens.mmlocalize.lens_validity import tie_aware_row
    from jlens.mmpilot.band_lens import (
        BAND_CONFIRMATION_PROMPT_SEED, BAND_DEVELOPMENT_PROMPT_SEED,
    )

    # Rebuild the extension's own fresh sets, prove they are the same draw, then
    # exclude them.
    _extension_excluded = {
        "old_fit": OLD_FIT, "old_development": OLD_VALIDATION,
        "old_confirmation": OLD_CONFIRMATION, "new_fit": EXTENSION_FIT_RECORDS,
    }
    EXTENSION_SPLITS = build_fresh_evaluation_splits(
        EXTENSION_POOL, excluded=_extension_excluded,
        corpus_id=EXTENSION_REPORT["corpus"]["corpus_id"],
    )
    EXTENSION_SPLIT_PROOF = verify_reconstructed_extension_splits(
        EXTENSION_SPLITS, extension_report=EXTENSION_REPORT
    )
    print("extension sets rebuilt and verified:", EXTENSION_SPLIT_PROOF["all_match"])

    BAND_EXCLUDED = {
        **_extension_excluded,
        "extension_development": EXTENSION_SPLITS.development,
        "extension_confirmation": EXTENSION_SPLITS.confirmation,
    }
    BAND_SPLITS, BAND_SPLIT_LEAKAGE = build_band_evaluation_splits(
        EXTENSION_POOL, excluded=BAND_EXCLUDED,
        corpus_id=EXTENSION_REPORT["corpus"]["corpus_id"],
    )
    BAND_STORE.save("band_splits", "manifest", {
        "splits": BAND_SPLITS.manifest(),
        "leakage_audit": BAND_SPLIT_LEAKAGE,
        "extension_split_proof": EXTENSION_SPLIT_PROOF,
        "generation": "third: parent sets and extension sets both excluded",
    })
    for _name in ("development", "confirmation"):
        print(f"  {_name:<13} {len(BAND_SPLITS.get(_name)):>4} records  "
              f"{BAND_SPLITS.checksum(_name)}")
    print("  leakage audit", "CLEAN" if BAND_SPLIT_LEAKAGE["ok"] else "FAILED")

    _target_token = lambda prompt: ordinary_next_token_argmax(
        MODEL, prompt, max_length=BAND_PLAN.max_seq_len
    )
    DEV_PROMPTS, DEV_SELECTION = select_diverse_validation_prompts(
        [record.text for record in BAND_SPLITS.development],
        n_prompts=EXTENSION_GATE.n_prompts, gate=EXTENSION_GATE,
        seed=BAND_DEVELOPMENT_PROMPT_SEED, target_token_for_prompt=_target_token,
    )
    CONF_PROMPTS, CONF_SELECTION = select_diverse_validation_prompts(
        [record.text for record in BAND_SPLITS.confirmation],
        n_prompts=EXTENSION_CONFIRMATION_GATE.n_prompts,
        gate=EXTENSION_CONFIRMATION_GATE,
        seed=BAND_CONFIRMATION_PROMPT_SEED, target_token_for_prompt=_target_token,
    )

    def score_readout_rows(lens, prompts, layers, target_layer):
        """Tie-aware rows for every (prompt, layer, variant). One forward each."""
        import hashlib
        from jlens.calibration.gate import CONTROL_SEED
        variants = {
            "permuted": control_lens(lens, "permuted", seed=CONTROL_SEED),
            "random": control_lens(lens, "random", seed=CONTROL_SEED),
            "wrong_layer": layer_mapped_lens(lens, distant_layer_mapping(layers)),
        }
        rows, record_at = [], sorted({*layers, target_layer})
        for index, prompt in enumerate(prompts):
            prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()
            ids = MODEL.encode(prompt, max_length=BAND_PLAN.max_seq_len)
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
                        rows.append(tie_aware_row(
                            sample_index=index, prompt_sha=prompt_sha, layer=layer,
                            variant=name, variant_logits=logits, actual_logits=actual,
                        ))
        return rows

    BAND_LENS_OBJECT = CONTINUATION.lens_for_scale(BAND_SCALE)
    _stored = BAND_STORE.load("validation", f"scale{BAND_SCALE}")
    if _stored is not None:
        BAND_DEVELOPMENT = {int(k): v for k, v in _stored["by_layer"].items()}
    else:
        BAND_DEVELOPMENT = evaluate_calibration_layers(
            score_readout_rows(
                BAND_LENS_OBJECT, DEV_PROMPTS, list(BAND_PLAN.layers), BAND_TARGET_LAYER
            ),
            layers=list(BAND_PLAN.layers), scale=BAND_SCALE, stage="validation",
            gate=EXTENSION_GATE,
        )
        BAND_STORE.save("validation", f"scale{BAND_SCALE}", {
            "scale": BAND_SCALE,
            "by_layer": {str(k): v for k, v in BAND_DEVELOPMENT.items()},
        })

    SELECTION = band_scale_selection(scale=BAND_SCALE, equivalence=EQUIVALENCE)
    VAULT = ConfirmationVault(records=BAND_SPLITS.confirmation)
    VAULT.unlock(SELECTION)
    _records = VAULT.open()
    _stored = BAND_STORE.load("confirmation", f"scale{BAND_SCALE}")
    if _stored is not None:
        BAND_CONFIRMATION = {int(k): v for k, v in _stored["by_layer"].items()}
    else:
        BAND_CONFIRMATION = evaluate_calibration_layers(
            score_readout_rows(
                BAND_LENS_OBJECT, CONF_PROMPTS, list(BAND_PLAN.layers), BAND_TARGET_LAYER
            ),
            layers=list(BAND_PLAN.layers), scale=BAND_SCALE, stage="confirmation",
            gate=EXTENSION_CONFIRMATION_GATE,
        )
        BAND_STORE.save("confirmation", f"scale{BAND_SCALE}", {
            "scale": BAND_SCALE, "selection": SELECTION,
            "by_layer": {str(k): v for k, v in BAND_CONFIRMATION.items()},
        })

    PUBLISHED, FAILED = [], []
    for _layer in sorted(BAND_CONFIRMATION):
        _verdict = BAND_CONFIRMATION[_layer]
        if not _verdict["passed"]:
            FAILED.append(record_failed_layer(
                layer=_layer, scale=BAND_SCALE, confirmation_verdict=_verdict,
                validation_verdict=BAND_DEVELOPMENT[_layer],
            ))
            continue
        try:
            PUBLISHED.append(publish_band_layer(
                layer=_layer, scale=BAND_SCALE, lens=BAND_LENS_OBJECT,
                destination=BAND_STORE.published_path(_layer, BAND_SCALE),
                confirmation_verdict=_verdict,
                development_verdict=BAND_DEVELOPMENT[_layer],
                vault=VAULT, splits=BAND_SPLITS, selection=SELECTION,
                equivalence=EQUIVALENCE, band_run_dir=BAND_RUN_DIR,
                protected_run_dirs=PROTECTED_RUN_DIRS,
                load_info=LOAD_INFO,
                corpus_manifest={
                    "corpus_id": EQUIVALENCE["extension_corpus_id"],
                    "revision": EQUIVALENCE["extension_corpus_revision"],
                    "revision_status": "MATCHES_EXTENSION_RESOLVED_REVISION",
                    "corpus_manifest_checksum": EQUIVALENCE["equivalence_checksum"],
                    "splits": {"checksums": {
                        "fit": EQUIVALENCE["extension_fit_prefix_checksum"],
                        "validation": BAND_SPLITS.checksum("development"),
                        "confirmation": BAND_SPLITS.checksum("confirmation"),
                    }},
                },
                capture_plan=BAND_PLAN.to_dict(),
                fitting_diagnostics=CONTINUATION.to_dict(),
                environment={"torch": torch.__version__, "mode": "real"},
            ))
        except PublicationRefused as _error:
            print(f"  refused L{_layer}: {_error}")
            FAILED.append(record_failed_layer(
                layer=_layer, scale=BAND_SCALE, confirmation_verdict=_verdict,
                validation_verdict=BAND_DEVELOPMENT[_layer],
            ))
    BAND_PUBLICATION = {
        "n_published": len(PUBLISHED), "n_failed": len(FAILED),
        "published_layers": sorted(int(a["physical_layer"]) for a in PUBLISHED),
        "failed_layers": sorted(int(a["physical_layer"]) for a in FAILED),
        "published_checksums": {
            str(a["physical_layer"]): a["lens_checksum"] for a in PUBLISHED
        },
        "publication_targets": list(BAND_INTERIOR_LAYERS),
    }
    BAND_STORE.save("band_publication", f"scale{BAND_SCALE}", BAND_PUBLICATION)
    BAND_VERDICT = band_layer_verdict(
        BAND_CONFIRMATION, scale=BAND_SCALE, selection=SELECTION,
        development=BAND_DEVELOPMENT,
        interior_layers=BAND_INTERIOR_LAYERS,
        already_confirmed_layers=[
            row["layer"] for row in INVENTORY["rows"]
            if row["usable"] and row["layer"] not in BAND_INTERIOR_LAYERS
        ],
        window=BAND_WINDOW,
    )
    BAND_STORE.save("band_verdict", "verdict", BAND_VERDICT)
    _report = {
        "schema": "jlens.mmpilot.band_interior_lens_report.v1",
        "mode": "real",
        "protocol": BAND_LENS_PROTOCOL,
        "fingerprint_digest": BAND_STORE.fingerprint.digest,
        "corpus_equivalence": EQUIVALENCE,
        "fresh_splits": BAND_SPLITS.manifest(),
        "fresh_split_leakage_audit": BAND_SPLIT_LEAKAGE,
        "extension_split_proof": EXTENSION_SPLIT_PROOF,
        "development": {str(k): v for k, v in BAND_DEVELOPMENT.items()},
        "confirmation": {str(k): v for k, v in BAND_CONFIRMATION.items()},
        "publication": BAND_PUBLICATION,
        "band_verdict": BAND_VERDICT,
        "budget": FIT_BUDGET,
        "resume": BAND_STORE.status_report(),
    }
    _report_dir = BAND_RUN_DIR / "artifacts"
    _report_dir.mkdir(parents=True, exist_ok=True)
    _tmp = _report_dir / "band_interior_lens_report.json.tmp"
    _tmp.write_text(json.dumps(_report, indent=2, default=str), encoding="utf-8")
    os.replace(_tmp, _report_dir / "band_interior_lens_report.json")

    print()
    print("VERDICT", BAND_VERDICT["verdict"])
    print("  interior passing   ", BAND_VERDICT["interior_layers_passing"])
    print("  interior failing   ", BAND_VERDICT["interior_layers_failing"])
    print("  full band available", BAND_VERDICT["full_band_available"])
    print("  largest admissible ", BAND_VERDICT["largest_admissible_contiguous_band"])
    print("  ", BAND_VERDICT["statement"])
    print("  report", _report_dir / "band_interior_lens_report.json")
    print()
    print("Re-run section 3 after this: the inventory picks the new lenses up and")
    print("the band design in section 4 stops being blocked.")
else:
    print("skipped: RUN_STAGE2_CONFIRM_AND_PUBLISH is False")
'''
)

markdown(
    """
## 8. MOCK — stages 1 and 2 against a synthetic stack

Runs this repository's own calibration code (real store, real
`run_calibration`, real upstream `fit`, real gate, real publication path)
against a tiny CPU model, for the two commissioned cases: every interior layer
passes, and one interior layer fails. The second is the one worth watching —
the pipeline must report a reduced admissible sub-band rather than widening the
band to whatever it has.
"""
)
code(
    r'''
MOCK_LENS_RESULTS = {}
if not REAL_MODE:
    import tempfile
    from jlens.calibration.corpus import build_records
    from jlens.calibration.fitting import filter_records_by_tokens, run_calibration
    from jlens.calibration.mock import MockCalibrationModel, mock_corpus_texts
    from jlens.calibration.state import CalibrationFingerprint
    from jlens.mmpilot.band_lens import (
        BAND_SCALE as _BAND_SCALE, BandLensStore, band_capture_plan, band_layer_verdict,
    )
    from jlens.mmpilot.band_swap_mock import (
        BAND_MOCK_LAYERS, BAND_MOCK_SCENARIOS, band_mock_confirmation,
    )

    MOCK_ROOT = Path(tempfile.gettempdir()) / "jlens_band_swap_mock"
    MOCK_ROOT.mkdir(parents=True, exist_ok=True)

    # Stage 1, for real, on a tiny stack: the real store, the real
    # run_calibration, the real upstream fit, over the real interior layer grid.
    _model = MockCalibrationModel()
    _plan = band_capture_plan(
        layers=BAND_MOCK_LAYERS, d_model=_model.d_model, n_layers=_model.n_layers,
        max_seq_len=48, skip_first=4,
    )
    _records, _ = filter_records_by_tokens(
        build_records("mock/train", mock_corpus_texts(60), min_chars=100),
        token_count=_model.tokenizer.token_count, skip_first=4, max_seq_len=48,
    )
    _mock_fingerprint = CalibrationFingerprint(
        mode="mock", protocol_version=BAND_LENS_PROTOCOL,
        model_repo_id="mock", model_revision="mock", tokenizer_revision="mock",
        capture_plan_digest=_plan.digest, corpus_manifest_checksum="sha256:mock-corpus",
        gate_digest=EXTENSION_GATE.digest, plateau_rule_digest="not_applicable_single_scale",
        scale_points=(12,), artifact_format_version="jlens.calibration.artifact.v1",
    )
    _mock_store = BandLensStore(
        MOCK_ROOT / f"bandlens_mock_{_mock_fingerprint.digest[7:19]}", _mock_fingerprint
    )
    print("mock band-lens run", _mock_store.root, _mock_store.open())
    MOCK_FIT = run_calibration(
        _model, _records, plan=_plan, scale_points=(12,), store=_mock_store,
        checkpoint_every=4, diagnostics_every=4,
    )
    print(f"  fitted {MOCK_FIT.n_done} prompts over layers {list(_plan.layers)}")
    for _scale, _snapshot in sorted(MOCK_FIT.snapshots.items()):
        print(f"  scale {_scale}: {_snapshot.checksum}")
    _resumed = run_calibration(
        _model, _records, plan=_plan, scale_points=(12,), store=_mock_store,
        checkpoint_every=4, diagnostics_every=4,
    )
    print("  re-running reused the completed snapshot:",
          _resumed.snapshots[12].checksum == MOCK_FIT.snapshots[12].checksum)

    for _key, _scenario in BAND_MOCK_SCENARIOS.items():
        _development = band_mock_confirmation(
            _scenario, scale=_BAND_SCALE, n_prompts=EXTENSION_GATE.n_prompts,
            stage="development", layers=BAND_MOCK_LAYERS,
        )
        _confirmation = band_mock_confirmation(
            _scenario, scale=_BAND_SCALE, n_prompts=EXTENSION_CONFIRMATION_GATE.n_prompts,
            stage="confirmation", layers=BAND_MOCK_LAYERS,
        )
        _verdict = band_layer_verdict(
            _confirmation, scale=_BAND_SCALE,
            selection={"selection_checksum": "sha256:mock-selection"},
            development=_development, interior_layers=BAND_MOCK_LAYERS,
            already_confirmed_layers=(32, 35, 38, 40), window=(32, 40),
        )
        MOCK_LENS_RESULTS[_key] = _verdict
        print(f"{_key:<20} {_verdict['verdict']:<26} "
              f"passing={_verdict['interior_layers_passing']} "
              f"band={_verdict['largest_admissible_contiguous_band']}")
        assert _verdict["verdict"] == _scenario.expected_verdict, _key
    print()
    print("MOCK lens stages complete. Real gate, real scorer, fixture rows.")
else:
    print("skipped: real mode")
'''
)

markdown(
    """
## 9. Stage 3 — the pass budget for the band swap

Every band is a full clamp: `4 bands x 2 arms x 7 conditions x 2 readouts x 3
modalities x 2 directions x 8 photographs`, each scoring the predeclared
candidate set. Installing nine hooks costs no more forward passes than
installing one, so a band trial costs what a single-layer trial costs.
"""
)
code(
    r'''
BAND_PASS_BUDGET = None
if REAL_MODE:
    _n_bands = max(1, len(SUFFIX_BANDS))
    _clean = (
        len(PAIR_CONCEPTS) * CANDIDATE_IMAGES_PER_CONCEPT * len(MODALITIES) * 2 * 2
    )
    _cells = len(PAIR_CONCEPTS) * len(MODALITIES) * MAX_ANALYSIS_IMAGES_PER_CELL
    _intervention = _cells * _n_bands * 2 * len(BAND_CONDITIONS) * 2 * 2
    BAND_PASS_BUDGET = {
        "clean_candidate_passes": _clean,
        "intervention_candidate_passes": _intervention,
        "total": _clean + _intervention,
        "bands": _n_bands,
        "conditions": len(BAND_CONDITIONS),
        "hooks_per_trial": [len(band.layers) for band in SUFFIX_BANDS],
    }
    print("PASS BUDGET")
    for _key, _value in BAND_PASS_BUDGET.items():
        print(f"  {_key:<32} {_value}")
    print("  expected L4 wall time: roughly 2-5 hours; A100 is usually faster")
    if RUN_STAGE3_BAND_SWAP and not (CONFIRM_MODEL_LOAD and CONFIRM_PASS_BUDGET):
        raise RuntimeError(
            "stage 3 needs CONFIRM_MODEL_LOAD and CONFIRM_PASS_BUDGET set by hand"
        )
else:
    print("skipped: MOCK")
'''
)

markdown(
    """
## 10. Stage 3 — the contiguous-band coordinate swap

`coordinate_swap_band` installs one hook per physical band layer, all at once,
for the whole scored forward pass. Each layer reads its own activation, builds
`c` from its own lens vectors, and patches every original prompt position;
positions at or beyond `prompt_len` are teacher-forced candidate tokens and are
never touched. `band_trial_record` refuses to store a trial whose hooks did not
fire at exactly the requested band.

Conditions: `swap_alpha1` (primary), `swap_alpha2` (fixed double strength),
`zero`, and norm-matched random and unrelated-pair controls **at both alphas**,
so alpha=2 is never compared against an alpha=1 baseline.
"""
)
code(
    r'''
from jlens.mmpilot.band_swap import (
    BAND_INTERVENTION_FAMILY, CONDITION_ALPHA, band_swap_fingerprint, band_trial_record,
)

SWAP_STORE = None
SWAP_RUN_DIR = None
BAND_RECORDS = []

if RUN_STAGE3_BAND_SWAP:
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

    if DESIGN is None:
        raise RuntimeError(
            "stage 3 is blocked: no admissible contiguous band exists yet.\n"
            + str(BAND_DESIGN_BLOCKED)
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

    # --- per-layer J-lens vectors, each from the artifact the inventory named
    TOKEN_NAMES = (*POPULATION_CONCEPTS, "two", "four")
    TOKENS = {name: resolve_concept_token(BACKEND.encode_candidate, name) for name in TOKEN_NAMES}
    _unembedding = BACKEND.unembedding_weight()
    _rows = {
        name: _unembedding[token.token_id].detach().float().cpu()
        for name, token in TOKENS.items()
    }
    TOKEN_VECTORS, LENS_CHECKSUMS = {}, {}
    _loaded_files = {}
    for _layer in PRIMARY_BAND.layers:
        _source = LENS_SOURCES[_layer]
        if _source.path not in _loaded_files:
            _loaded_files[_source.path] = JacobianLens.load(_source.path)
        _jacobian = _loaded_files[_source.path].jacobians[_source.layer_key_in_file]
        _jacobian = _jacobian.detach().float().cpu()
        TOKEN_VECTORS[_layer] = {name: row @ _jacobian for name, row in _rows.items()}
        LENS_CHECKSUMS[_layer] = _source.checksum
        del _jacobian
    del _loaded_files, _unembedding, _rows
    print("lens vectors built for", sorted(TOKEN_VECTORS))

    def selected_bases(layers, source_name, target_name):
        return {
            layer: build_swap_basis_from_vectors(
                TOKEN_VECTORS[layer][source_name], TOKEN_VECTORS[layer][target_name],
                layer=layer, source=TOKENS[source_name], target=TOKENS[target_name],
            )
            for layer in layers
        }

    # --- the population: same hidden-animal protocol as the completed study
    _raw = EXPANDED_MANIFEST_CACHE.read_bytes()
    MANIFEST_FILE_CHECKSUM = "sha256:" + __import__("hashlib").sha256(_raw).hexdigest()
    _payload = json.loads(_raw)
    _exclusion = json.loads(PRIOR_EXCLUSION_SET.read_bytes())
    _excluded_images = {str(v) for v in _exclusion.get("image_ids", [])}
    _excluded_groups = {str(v) for v in _exclusion.get("group_ids", [])}
    _eligible = [
        row for row in _payload["groups"]
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
    del _payload, _raw, _eligible
    print("population digest", POPULATION["population_digest"])

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
        offline = group["caption"] if modality != "text" else None
        return built, build_backend_inputs(BACKEND, built, transcript=offline)

    # The protocol identifiers, plus the fingerprint of one actually-built
    # prompt per protocol, so what was asked is bound beside what was patched.
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
    SWAP_FINGERPRINT_CONFIG = band_swap_fingerprint(
        design=DESIGN,
        lens_checksums=LENS_CHECKSUMS,
        model_repo_id=MODEL_REPO_ID,
        model_revision=_bundle.model_revision,
        processor_revision=_bundle.processor_revision,
        transformers_version=TRANSFORMERS_VERSION_EXPECTED,
        audio_protocol_fingerprint=AUDIO_RECORD["protocol_fingerprint"],
        prompt_protocol=PROMPT_PROTOCOLS,
        directed_pairs=DIRECTED_PAIRS,
        sample_identities={
            "population_digest": POPULATION["population_digest"],
            "selection_seed": SELECTION_SEED,
            "candidate_images_per_concept": CANDIDATE_IMAGES_PER_CONCEPT,
            "max_analysis_images_per_cell": MAX_ANALYSIS_IMAGES_PER_CELL,
            "min_analysis_images_per_cell": MIN_ANALYSIS_IMAGES_PER_CELL,
        },
        thresholds=THRESHOLDS.to_dict(),
        coordinate_swap_method_version=METHOD_VERSION,
        fitting_manifest={"band_lens_run": str(BAND_RUN_DIR or ""), "scale": BAND_SCALE},
        validation_manifest={"inventory_digest": INVENTORY["inventory_digest"]},
        scoring_rule=(
            "teacher-forced complete-sequence scoring of the predeclared external "
            "candidates; success is the target answer being top-1"
        ),
    )
    _fingerprint = RunFingerprint(
        mode="anthropic_contiguous_band_coordinate_swap",
        model_repo_id=MODEL_REPO_ID,
        model_revision=_bundle.model_revision,
        processor_revision=_bundle.processor_revision,
        layers=tuple(PRIMARY_BAND.layers),
        lens_checksum=SWAP_FINGERPRINT_CONFIG["band_fingerprint_digest"],
        manifest_checksum=MANIFEST_FILE_CHECKSUM,
        split_id=SELECTION_SEED,
        intervention_config=SWAP_FINGERPRINT_CONFIG,
        selection_config={
            "population_digest": POPULATION["population_digest"],
            "pair_concepts": list(PAIR_CONCEPTS),
            "control_concepts": list(CONTROL_CONCEPTS),
            "modalities": list(MODALITIES),
        },
        extra={
            "study": BAND_SWAP_VERSION,
            "design_digest": DESIGN["design_digest"],
            "inventory_digest": INVENTORY["inventory_digest"],
            "completed_single_layer_run_read": False,
        },
    )
    SWAP_RUN_DIR = RUNS_ROOT / f"mmband_real_{_fingerprint.digest.split(':')[1][:12]}"
    SWAP_STORE = UnitStore(SWAP_RUN_DIR, _fingerprint)
    print("band swap run", SWAP_RUN_DIR)
    print("run state    ", SWAP_STORE.open())

    # --- clean behavioural screen, before any causal spending
    _computed = _reused = 0
    for group in POPULATION["groups"]:
        source = group["concept"]
        target = next(name for name in PAIR_CONCEPTS if name != source)
        for modality in MODALITIES:
            for readout in ("identity", "property"):
                key = safe_key("band-clean", group["group_id"], modality, readout)
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
                })
                _computed += 1
    CLEAN_UNITS = list(SWAP_STORE.load_all("capability").values())
    CAUSAL_SELECTION = select_capability_eligible_samples(
        CLEAN_UNITS, concepts=PAIR_CONCEPTS, modalities=MODALITIES,
        max_images_per_cell=MAX_ANALYSIS_IMAGES_PER_CELL,
        min_images_per_cell=MIN_ANALYSIS_IMAGES_PER_CELL, seed=SELECTION_SEED,
    )
    SWAP_STORE.save("metric", "band_capability_selection", CAUSAL_SELECTION)
    print(f"clean screen: {_computed} computed, {_reused} reused")
    for row in CAUSAL_SELECTION["cells"]:
        print(f"  {row['concept']:6s} {row['modality']:12s} "
              f"eligible={row['n_eligible']:2d} selected={row['n_selected']:2d} "
              f"sufficient={row['sufficient']}")
    print("  all cells sufficient", CAUSAL_SELECTION["all_cells_sufficient"])

    if not CAUSAL_SELECTION["all_cells_sufficient"]:
        print("CAUSAL STAGE STOPPED: clean capability was insufficient.")
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
                        layer: random_two_direction_basis(basis, seed=20260811 + layer + band.start)
                        for layer, basis in banks["intermediate"].items()
                    }
                    banks["random_answer"] = {
                        layer: random_two_direction_basis(basis, seed=20261811 + layer + band.start)
                        for layer, basis in banks["answer"].items()
                    }
                    for arm in ("intermediate", "answer"):
                        for condition in BAND_CONDITIONS:
                            for readout in ("identity", "property"):
                                key = safe_key(
                                    "band-swap", group["group_id"], modality,
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
    BAND_RECORDS = [
        row for row in SWAP_STORE.load_all("intervention").values()
        if row.get("status") == "complete"
    ]
else:
    print("skipped: RUN_STAGE3_BAND_SWAP is False")
'''
)

markdown(
    """
## 11. MOCK — stage 3 against the synthetic world

The same `run_swap_condition`, the same `band_trial_record`, the same
conditions and the same aggregation. The world is
`jlens.mmpilot.coordinate_swap_mock`: evidence injected at layer 0, carry
blocks, a broadcast at layer 4, and a reasoning layer at 5 that computes the
legs answer from whichever identity coordinates reach it.

Read the identity numbers with the parity note in mind: this world's carry
blocks nearly commute with the exchange, so an even-length band lands the
*identity* readout almost back on the source. The downstream answer does not
move back, because it is written once at the reasoning layer. That is precisely
why identity is a diagnostic and the downstream answer is the endpoint.
"""
)
code(
    r'''
if not REAL_MODE:
    from jlens.mmpilot.band_swap_mock import mock_band_grid, run_mock_band_trials
    from jlens.mmpilot.coordinate_swap_mock import SwapMockBackend

    MOCK_BACKEND = SwapMockBackend()
    BAND_RECORDS = run_mock_band_trials(
        MOCK_BACKEND, modalities=MODALITIES, n_images=4
    )
    BAND_KEYS = tuple(band_key(band) for band in mock_band_grid())
    DIRECTED_PAIRS = [{"source": "bird", "target": "cat"}]
    print("mock band trials", len(BAND_RECORDS))
    print("bands            ", list(BAND_KEYS))
else:
    print("skipped: real mode")
'''
)

markdown(
    """
## 12. Stage 3 — aggregate and judge

Image-level aggregation with a seeded bootstrap interval, then the verdict.

* **Primary:** the fraction of trials where the target-appropriate downstream
  answer is top-1, at alpha=1, beating every intensity-matched control.
* **Secondary:** target rank, log-prob, margin, and the bootstrap interval.
  A positive margin that does not reach top-1 is reported as
  `partial_movement_not_top1` and is never counted as a success.
* **Diagnostic:** identity replacement. It cannot produce a reasoning GO on its
  own, and the verdict does not read it as a success clause.
* Text-only is the paper-comparable result; image and spoken audio are reported
  separately, and the tri-modal conjunction is labelled as our extension.
"""
)
code(
    r'''
from jlens.mmpilot.band_swap import (
    band_reasoning_verdict, read_band_units, summarize_band_cells,
)

CELLS, REASONING, STAGE4_CONTEXT = [], None, None
if RUN_STAGE4_TIMING and not BAND_RECORDS and STAGE4_SWAP_RUN_DIR:
    # Stage 4 on its own: re-read a finished run's units, checksum by checksum,
    # with no model, no processor and no media. The bands and directed pairs
    # come from that run's report, so the analysis is over the design that was
    # frozen rather than over whatever is on disk.
    BAND_RECORDS, STAGE4_CONTEXT = read_band_units(STAGE4_SWAP_RUN_DIR)
    BAND_KEYS = tuple(STAGE4_CONTEXT["band_keys"])
    DIRECTED_PAIRS = list(STAGE4_CONTEXT["directed_pairs"])
    print("stage 4 re-analysis of", STAGE4_CONTEXT["run_dir"])
    print("  complete units", STAGE4_CONTEXT["n_units"],
          " invalid", STAGE4_CONTEXT["n_invalid_units"])
    print("  bands         ", list(BAND_KEYS))

if BAND_RECORDS:
    CELLS = summarize_band_cells(BAND_RECORDS, thresholds=THRESHOLDS)
    REASONING = band_reasoning_verdict(
        CELLS, bands=BAND_KEYS, directed_pairs=DIRECTED_PAIRS,
        modalities=MODALITIES,
        paper_comparable_modality=PAPER_COMPARABLE_MODALITY,
        thresholds=THRESHOLDS,
    )
    print("=" * 72)
    print("VERDICT", REASONING["verdict"])
    print("=" * 72)
    print("paper-comparable (text, alpha=1) passing bands:",
          REASONING["paper_comparable"]["passing_bands"])
    print("tri-modal extension passing bands            :",
          REASONING["modality_extension"]["tri_modal_passing_bands"])
    print("alpha=2 sensitivity passing bands            :",
          REASONING["alpha2_sensitivity"]["passing_bands"])
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
    print("no band trials to aggregate")
'''
)

markdown(
    """
## 13. Stage 4 — intermediate versus answer, over the same suffix bands

The same operator, twice: once exchanging the inferred animal's coordinates and
once exchanging the leg-count answer's, over the identical bands, with the
identical controls, and **direction-matched** — the same `source -> target`
pair has to carry both arms.

Two statistics per arm, both printed:

* `earliest_effective_start` — the first band start that works. Over nested
  suffix bands this is monotone-degenerate for an arm carried by the deepest
  layers, which is exactly why it is not the classification statistic.
* `deepest_effective_start` — the last band start that still works: the depth
  by which the representation has been consumed. An intermediate concept that
  stops being editable *before* the answer does is evidence that it was
  computed first and then used.

A band start is not an exact physical-layer onset, and every payload says so.
"""
)
code(
    r'''
from jlens.mmpilot.band_swap import band_onset_timing

TIMING = None
if REASONING is not None and (RUN_STAGE4_TIMING or not REAL_MODE):
    TIMING = band_onset_timing(
        REASONING, bands=BAND_KEYS, directed_pairs=DIRECTED_PAIRS,
        modalities=MODALITIES, condition="swap_alpha1",
        modality=PAPER_COMPARABLE_MODALITY,
    )
    print("=" * 72)
    print("TIMING", TIMING["verdict"])
    print("=" * 72)
    for row in TIMING["pairs"]:
        print(f"  {row['pair']}")
        print(f"    effective band starts  {row['effective_band_starts']}")
        print(f"    earliest effective     {row['earliest_effective_start']}")
        print(f"    deepest effective      {row['deepest_effective_start']}")
        print(f"    classification         {row['classification']}")
    print()
    print(" ", TIMING["why_not_earliest"])
    print("  band starts are not exact physical onsets:",
          TIMING["band_starts_are_not_exact_physical_onsets"])
    print("  discarded native direct-readout convergence gate used:",
          TIMING["native_direct_readout_convergence_gate_used"])
elif REASONING is not None:
    print("skipped: set RUN_STAGE4_TIMING=True for the CPU-only timing stage")
else:
    print("skipped: no reasoning verdict")
'''
)

markdown(
    """
## 14. Report, resume status, and what to send back
"""
)
code(
    r'''
REPORT = None
if REASONING is not None:
    from jlens.mmpilot.store import payload_checksum

    REPORT = {
        "schema": "jlens.mmpilot.anthropic_band_swap_report.v1",
        "mode": MODE,
        "study_version": BAND_SWAP_VERSION,
        "intervention_family": BAND_INTERVENTION_FAMILY,
        "design": DESIGN,
        "lens_inventory": INVENTORY,
        "lens_discovery": DISCOVERY,
        "band_lens_run": str(BAND_RUN_DIR) if BAND_RUN_DIR else None,
        "band_lens_verdict": BAND_VERDICT,
        "swap_run": str(SWAP_RUN_DIR) if SWAP_RUN_DIR else None,
        "stage4_reanalysis_of": STAGE4_CONTEXT,
        "directed_pairs": DIRECTED_PAIRS,
        "band_keys": list(BAND_KEYS),
        "thresholds": THRESHOLDS.to_dict(),
        "threshold_digest": THRESHOLDS.digest,
        "cells": CELLS,
        "reasoning_verdict": REASONING,
        "timing": TIMING,
        "method_statement": (
            "one exact two-coordinate exchange at every physical layer of a "
            "contiguous band, installed simultaneously, recomputed per layer from "
            "that layer's own activation, applied at every original prompt "
            "position and never at a teacher-forced candidate token"
        ),
        "anthropic_comparison": (
            "the equation, the all-position clamp, the contiguous band, the "
            "alpha=1 primary and alpha=2 secondary conditions and the top-1 trial "
            "definition are replicated; the model, the dataset, the modalities and "
            "the band grid are ours"
        ),
        "completed_single_layer_run_read_or_modified": False,
        "mock_proves_pipeline_only": MODE != "real",
        "public_method_reference": (
            "https://transformer-circuits.pub/2026/workspace/index.html"
            "#technical-details-of-j-lens-use-cases"
        ),
    }
    REPORT["report_checksum"] = payload_checksum(REPORT)
    if SWAP_STORE is not None:
        _path = SWAP_RUN_DIR / "anthropic_band_swap_report.json"
        _tmp = _path.with_suffix(".json.tmp")
        _tmp.write_text(json.dumps(REPORT, indent=2, default=str), encoding="utf-8")
        os.replace(_tmp, _path)
        SWAP_STORE.save("metric", "anthropic_band_swap_report", REPORT)
        print("report", _path)
    print("report checksum", REPORT["report_checksum"])

for _store, _label in ((BAND_STORE, "band lens run"), (SWAP_STORE, "band swap run")):
    if _store is not None:
        print()
        print(_label)
        print(json.dumps(_store.status_report(), indent=2, default=str))

print()
if MODE == "real":
    print("Send back:")
    print("  artifacts/band_interior_lens_report.json   (stages 1-2)")
    print("  anthropic_band_swap_report.json           (stages 3-4)")
    print("  the inventory table, the verdict block and the timing block above")
else:
    print("MOCK RUN COMPLETE — pipeline behaviour only.")
    print("No scientific claim about Gemma 4, about layers 33-39, about any")
    print("modality, and about the workspace hypothesis is made or implied.")
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
