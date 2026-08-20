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
RUN_STAGE4_WRITE_REPORT = False

CONFIRM_MODEL_LOAD = False
CONFIRM_FIT_BUDGET = False
CONFIRM_CAUSAL_BUDGET = False
CONFIRM_ALPHA_SWEEP_BUDGET = False
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
# Alpha=1 is the exact exchange. Alpha=2 is Anthropic's reported double-strength
# condition. Alpha=0.5 and alpha=4 are explicitly exploratory interpolation and
# extrapolation diagnostics; none may replace alpha=1 as the primary result.
ALPHA_SWEEP = (0.5, 1.0, 2.0, 4.0)

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
MODEL_STAGE = any((
    RUN_STAGE1_FIT_LENSES, RUN_STAGE2_CROSS_EVALUATE,
    RUN_STAGE3_CAUSAL_COMPARE, RUN_STAGE3B_ALPHA_SWEEP,
))
MODEL_ENABLED = bool(MODEL_STAGE and CONFIRM_MODEL_LOAD)
FIT_ENABLED = bool(RUN_STAGE1_FIT_LENSES and MODEL_ENABLED and CONFIRM_FIT_BUDGET)
CROSS_ENABLED = bool(RUN_STAGE2_CROSS_EVALUATE and MODEL_ENABLED)
CAUSAL_ENABLED = bool(RUN_STAGE3_CAUSAL_COMPARE and MODEL_ENABLED and CONFIRM_CAUSAL_BUDGET)
ALPHA_SWEEP_ENABLED = bool(
    RUN_STAGE3B_ALPHA_SWEEP and MODEL_ENABLED and CONFIRM_ALPHA_SWEEP_BUDGET
)
if REAL_MODE and MODEL_STAGE and not MODEL_ENABLED:
    print("MODEL STAGES BLOCKED: set CONFIRM_MODEL_LOAD after reading the budget")
if REAL_MODE and RUN_STAGE1_FIT_LENSES and not FIT_ENABLED:
    print("FIT BLOCKED: set CONFIRM_FIT_BUDGET after reading the budget")
if REAL_MODE and RUN_STAGE3_CAUSAL_COMPARE and not CAUSAL_ENABLED:
    print("CAUSAL BLOCKED: set CONFIRM_CAUSAL_BUDGET after reading the budget")
if REAL_MODE and RUN_STAGE3B_ALPHA_SWEEP and not ALPHA_SWEEP_ENABLED:
    print("ALPHA SWEEP BLOCKED: set CONFIRM_ALPHA_SWEEP_BUDGET after reading the budget")
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
        "matched_multimodal_jlens.alpha_dose_response.v1"
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
        "0.5": "exploratory_interpolation",
        "1.0": "primary_exact_exchange",
        "2.0": "paper_reported_double_strength_sensitivity",
        "4.0": "exploratory_high_extrapolation",
    } if ALPHA1_SOURCE else {"1.0": "primary_exact_exchange"},
    "causal_protocol": (
        "matched_multimodal_jlens_unrestricted_alpha_sweep.v4"
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

    ALPHA_SWEEP_REPORT = {
        "schema": "jlens.mmpilot.matched_multimodal_alpha_sweep.v1",
        "protocol": "matched_multimodal_jlens_unrestricted_alpha_sweep.v4",
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
            "paper_comparable_alphas": [1.0, 2.0],
            "multimodal_task_is_extension_not_exact_replication": True,
        },
        "alpha_roles": SCIENTIFIC_CONFIG["alpha_roles"],
        "primary_alpha_remains": PRIMARY_ALPHA,
        "alpha_selected_after_outcomes": False,
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
    print("Alpha=1 remains primary; every other alpha is sensitivity evidence.")
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
