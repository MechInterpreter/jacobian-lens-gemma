"""Generate the paper-first workspace replication and confirmation notebook."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "notebooks" / "multimodal_jspace_workspace_replication_colab.ipynb"
CELLS: list[tuple[str, str]] = []


def markdown(value: str) -> None:
    CELLS.append(("markdown", value.strip("\n")))


def code(value: str) -> None:
    CELLS.append(("code", value.strip("\n")))


markdown(
    r"""
# Loading-first matched J/R-lens replication and multimodal confirmation

This is the shortest fail-closed path from the completed work to an interpretable
result. It uses **one causal implementation only**: Anthropic's two-coordinate
exchange in `jlens.mmpilot.coordinate_swap`.

The order is mandatory:

0. **Matched R-lens fit or verified reuse.** The early and late studies used the
   same frozen 99-example text and pooled multimodal fit plans with the same
   audited dense R-lens rules. In `combined_r_l27_l40` mode, their disjoint
   L27-L32 and L33-L40 matrices are checksum-verified and concatenated into one
   contiguous L27-L40 lens without fitting or averaging. Other modes retain the
   original resumable fitting path.
1. **Clean text loading, then text replication.** Compare the published text
   J-lens, matched text/pooled J-lenses, and matched text/pooled R-lenses on the
   same clean task activations. Select the instrument and contiguous band using
   final-prompt-token source loading only, before a causal hook exists. Then
   reproduce the paper's text-only implicit two-hop task
   (`spider → ant`, expected downstream answer `8 → 6`) and its France/China
   flexible-function family. The default modes use exact `alpha=1`; the
   prospective combined-band mode uses the paper's reported double-strength
   `alpha=2` intervention as primary and keeps `alpha=1` as sensitivity.
   Because this checkpoint is instruction-tuned, a generic answer-free user
   instruction is followed by the literal fragment as an assistant prefill
   with `continue_final_message=True`. Clean capability is measured for every
   task before any intervention is allowed to run.
   The answer must be the final lexical item; fixed digit/number-word
   equivalents are accepted and explicit negation is rejected.
   Gemma tokenizes the paper's digit outputs as whitespace + digit, so success
   is the complete answer from unrestricted two-token greedy generation—not a
   one-token prefix, candidate score, or teacher-forced likelihood.
1b. **Legacy alpha=1 diagnostic.** In alpha=1 modes, audit the actual
   post-cast exchange and iteratively correct only its two-coordinate component
   until the BF16 tensor consumed by Gemma meets the frozen 2% tolerance. Then
   test exactly the already selected clean-loading band at exact `alpha=1`
   against zero, random, unrelated, and norm-matched direct-answer controls.
   Causal outcomes are therefore unable to select a different layer or band.
   A passing development result still requires fresh confirmation.
2. **Clean multimodal source loading.** On development media, compare the
   matched pooled J and pooled R instruments at each layer and prompt position.
   The instrument, pair, and position rule are selected from these clean rows;
   no multimodal causal outcome is consulted.
3. **Freeze the design.** Require the clean-loading-selected multimodal
   instrument and band to remain available, and freeze their artifact identity,
   pair, and modality-specific position rule. Alpha roles are fixed by the
   study mode before clean or causal outcomes are opened.
4. **Fresh confirmation.** Open new photographs/recordings, prove zero overlap,
   and test both swap directions with unrestricted identity and downstream
   property outputs against zero, random, unrelated, and norm-matched
   direct-answer controls. Every two-token condition is saved atomically.

If the text replication or clean-loading gate fails, later causal spending is
blocked. No threshold, pair, layer, position, or alpha may be changed after its
gate has seen results. R-Lens changes how each per-layer map is fitted; it does
not change Anthropic's later two-coordinate, all-position, contiguous-band clamp.
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

markdown("## 1. Configuration — change switches, not scientific constants")
code(
    r'''
RUN_REAL_WORKSPACE_REPLICATION = False
RUN_STAGE0_FIT_MATCHED_R_LENSES = False
RUN_STAGE1_TEXT_REPLICATION = False
RUN_STAGE1B_TEXT_DIAGNOSTIC = False
RUN_STAGE2_MULTIMODAL_LOADING_DEVELOPMENT = False
RUN_STAGE3_FREEZE_DESIGN = False
RUN_STAGE4_FRESH_CONFIRMATION = False
RUN_STAGE5_WRITE_REPORT = False

# Tokenizer only: no weights, no GPU, ~a few MB. Run this before any paid
# session -- TEXT_CONCEPT_TOKENS is otherwise built after the model is
# resident, so one multi-token concept wastes the whole session.
RUN_TASK_TOKEN_PREFLIGHT = False

CONFIRM_MODEL_LOAD = False
CONFIRM_R_LENS_FIT_BUDGET = False
CONFIRM_TEXT_DIAGNOSTIC_BUDGET = False
CONFIRM_DEVELOPMENT_BUDGET = False
CONFIRM_CONFIRMATION_BUDGET = False

MODEL_REPO_ID = "google/gemma-4-E4B-it"
MODEL_REVISION = "fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd"
STUDY_LAYER_WINDOW = "late_jr_l33_l40"
# "frozen_v1" reproduces the completed runs byte for byte. "expanded_v1"
# adds the implicit-two-hop rows (n=1 -> n=14). "expanded_v2" drops the three
# rows that failed the clean-capability gate (n=1 -> n=11) and is the one to
# use. Either changes text_task_digest
# and therefore the fingerprint, so it can never resume into a frozen run.
TEXT_TASK_SET = "frozen_v1"
SCIENTIFIC_IMPLEMENTATION_COMMIT = (
    "c6b5dc144051a13ae163c89d2bfb5a0f955e9288"
)
EXPECT_N_LAYERS, EXPECT_D_MODEL, EXPECT_VOCAB = 42, 2560, 262144
AUDIO_PROTOCOL_FINGERPRINT = (
    "sha256:9ad8bcc9420a7983f6e3b75d5d7080c0e2fcf0a94a76431917fcde73ba777920"
)
if STUDY_LAYER_WINDOW == "late_jr_l33_l40":
    LAYERS = tuple(range(33, 41))
    SCIENTIFIC_IMPLEMENTATION_ID = SCIENTIFIC_IMPLEMENTATION_COMMIT
elif STUDY_LAYER_WINDOW == "early_r_l27_l32":
    LAYERS = tuple(range(27, 33))
    SCIENTIFIC_IMPLEMENTATION_ID = "loading-first-early-r-l27-l32.v1"
elif STUDY_LAYER_WINDOW == "combined_r_l27_l40":
    LAYERS = tuple(range(27, 41))
    SCIENTIFIC_IMPLEMENTATION_ID = "paper-band-combined-r-l27-l40-alpha2.v1"
elif STUDY_LAYER_WINDOW == "mid_r_l21_l29":
    # The paper reindexes depth to [0,100] and puts the workspace at ~38-92,
    # which on 42 layers is L16-L39; it applies the swap across a band of
    # *intermediate* layers. L21-L29 is depth 50-69, the middle of that range.
    # Every band tested so far missed it: L33-40 is depth 79-95 (two layers
    # past the workspace end) and L30-32 is depth 71-76 but only three layers,
    # which left the positive control at 4/17. Nine layers is also odd-length,
    # so the involution parity hazard does not apply.
    LAYERS = tuple(range(21, 30))
    SCIENTIFIC_IMPLEMENTATION_ID = "paper-band-mid-r-l21-l29.v1"
else:
    raise ValueError(
        "STUDY_LAYER_WINDOW must be 'late_jr_l33_l40', 'early_r_l27_l32', "
        "'combined_r_l27_l40', or 'mid_r_l21_l29'"
    )
TEXT_PRIMARY_ALPHA = 2.0 if STUDY_LAYER_WINDOW == "combined_r_l27_l40" else 1.0
# Instrument-power override, set from the POSITIVE CONTROL only.
# direct_answer_norm_matched measures whether a perturbation of this magnitude
# can move the output at all, with no reference to whether the swap succeeds.
# A short band delivers less total perturbation -- L30-32 (3 layers) passed the
# control on 4/17 tasks vs 15/17 for L33-40 (8 layers) -- so an implicit null
# read off an underpowered band is uninterpretable. Raise this until the control
# passes a majority, then read the swap. Selecting alpha on the SWAP outcome
# instead would violate causal_outcome_may_select_band and is not what this is
# for. None keeps the window's derived value.
TEXT_PRIMARY_ALPHA_OVERRIDE = None
if TEXT_PRIMARY_ALPHA_OVERRIDE is not None:
    TEXT_PRIMARY_ALPHA = float(TEXT_PRIMARY_ALPHA_OVERRIDE)
TEXT_DIAGNOSTIC_RANDOM_SEED = 20260820
MULTIMODAL_PRIMARY_ALPHA = TEXT_PRIMARY_ALPHA
MULTIMODAL_SENSITIVITY_ALPHA = (
    1.0 if STUDY_LAYER_WINDOW == "combined_r_l27_l40" else 0.75
)
CANDIDATE_PAIRS = (("bird", "cat"), ("bird", "zebra"), ("bird", "giraffe"))
CONTROL_CONCEPTS = ("microwave", "toilet")
# The unrelated comparison for implicit_two_hop, used by BOTH the clean loading
# capture (source_advantage) and the intervention's unrelated basis. They must
# agree, and every name here must reach TEXT_CONCEPT_TOKENS or
# capture_source_loading refuses for a missing lens vector.
# (zebra, giraffe) cannot serve the expanded set: zebra is a source concept
# there, giraffe reads as yellow/orange -- two swapped color answers -- and both
# are African, which contaminates the continent rows. frozen_v1 keeps the
# original pair so the completed runs stay re-derivable.
IMPLICIT_UNRELATED_CONCEPTS = (
    ("zebra", "giraffe") if TEXT_TASK_SET == "frozen_v1" else CONTROL_CONCEPTS
)
DEVELOPMENT_IMAGES_PER_SOURCE = 8
CONFIRMATION_IMAGES_PER_SOURCE = 8
MIN_SOURCE_ADVANTAGE = 0.0
MIN_SOURCE_COSINE = 0.0
R_LENS_CHECKPOINT_EVERY = 5
# Cotangent rows per backward pass, and therefore the replicated forward batch.
# Purely a memory/speed tradeoff: the full d_model x d_model Jacobian is
# assembled either way, so halving this is mathematically identical and only
# doubles the number of backward passes. Drop to 4 (or 2) if the pooled arm
# OOMs -- its image units have far longer sequences than the text arm's.
R_LENS_DIM_BATCH = 8
EVIDENCE_POSITION_MARGIN = 0.0
MIN_CONFIRMATION_SUCCESS_RATE = 0.50
CONFIRMATION_FAMILYWISE_ALPHA = 0.05
DEVELOPMENT_SEED = "paper-first-loading-development-20260820-v1"
CONFIRMATION_SEED = "paper-first-fresh-confirmation-20260820-v1"
PROMPT_PROTOCOL = "mmpilot.implicit_animal_property_open_output.v1"

# The fitter allocates and frees large transient tensors every backward pass;
# expandable_segments lets the caching allocator reuse those blocks instead of
# fragmenting. Set before torch initializes CUDA to take effect.
import os as _os
_os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

REAL_MODE = bool(RUN_REAL_WORKSPACE_REPLICATION)
MODEL_STAGE = any((
    RUN_STAGE0_FIT_MATCHED_R_LENSES,
    RUN_STAGE1_TEXT_REPLICATION,
    RUN_STAGE1B_TEXT_DIAGNOSTIC,
    RUN_STAGE2_MULTIMODAL_LOADING_DEVELOPMENT,
    RUN_STAGE4_FRESH_CONFIRMATION,
))
if RUN_STAGE0_FIT_MATCHED_R_LENSES and not CONFIRM_R_LENS_FIT_BUDGET:
    print("R-LENS FIT BLOCKED: set CONFIRM_R_LENS_FIT_BUDGET after reading the budget")
if MODEL_STAGE and not CONFIRM_MODEL_LOAD:
    print("MODEL STAGES BLOCKED: set CONFIRM_MODEL_LOAD after reading the budget")
if RUN_STAGE1B_TEXT_DIAGNOSTIC and not CONFIRM_TEXT_DIAGNOSTIC_BUDGET:
    print("TEXT DIAGNOSTIC BLOCKED: set CONFIRM_TEXT_DIAGNOSTIC_BUDGET")
if RUN_STAGE2_MULTIMODAL_LOADING_DEVELOPMENT and not CONFIRM_DEVELOPMENT_BUDGET:
    print("DEVELOPMENT BLOCKED: set CONFIRM_DEVELOPMENT_BUDGET")
if RUN_STAGE4_FRESH_CONFIRMATION and not CONFIRM_CONFIRMATION_BUDGET:
    print("CONFIRMATION BLOCKED: set CONFIRM_CONFIRMATION_BUDGET")
if (
    STUDY_LAYER_WINDOW == "combined_r_l27_l40"
    and RUN_STAGE0_FIT_MATCHED_R_LENSES
):
    raise RuntimeError(
        "combined_r_l27_l40 reuses the two completed R-lens shards; "
        "RUN_STAGE0_FIT_MATCHED_R_LENSES must remain False"
    )
'''
)

markdown("## 2. Paths, prior evidence, and pass budget — no model load")
code(
    r'''
if REAL_MODE:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    RUNS_ROOT = Path("/content/drive/MyDrive/jacobian-lens-gemma/runs")
    CORRECTED_RUN_DIR = RUNS_ROOT / "mmband" / "bandcorr_real_eb5b00f135e4"
    MATCHED_LENS_RUN_DIR = RUNS_ROOT / "mmjlens4" / "mmjlens4_real_1d3b1afbd019"
    EARLY_R_LENS_RUN_DIR = (
        RUNS_ROOT / "mmworkspace" / "mmworkspace_real_62ef81c904c5"
    )
    LATE_R_LENS_RUN_DIR = (
        RUNS_ROOT / "mmworkspace" / "mmworkspace_real_711a374adc77"
    )
    EXPANDED_MANIFEST_CACHE = (
        RUNS_ROOT / "mml32_l32_followup_20260808T182717" / "expanded_manifest.json"
    )
    MATCHED_FIT_PLAN_PATH = MATCHED_LENS_RUN_DIR / "matched_population_plan.json"
else:
    RUNS_ROOT = Path(tempfile.gettempdir()) / "workspace_replication_mock"
    CORRECTED_RUN_DIR = MATCHED_LENS_RUN_DIR = EXPANDED_MANIFEST_CACHE = None
    MATCHED_FIT_PLAN_PATH = None
    EARLY_R_LENS_RUN_DIR = LATE_R_LENS_RUN_DIR = None

if TEXT_TASK_SET not in ("frozen_v1", "expanded_v1", "expanded_v2"):
    raise ValueError(f"unknown TEXT_TASK_SET {TEXT_TASK_SET!r}")
_wr = __import__(
    "jlens.mmpilot.workspace_replication",
    fromlist=[
        "anthropic_text_tasks",
        "anthropic_text_tasks_expanded_v1",
        "anthropic_text_tasks_expanded_v2",
    ],
)
TEXT_TASKS = {
    "frozen_v1": _wr.anthropic_text_tasks,
    "expanded_v1": _wr.anthropic_text_tasks_expanded_v1,
    "expanded_v2": _wr.anthropic_text_tasks_expanded_v2,
}[TEXT_TASK_SET]()
print("TEXT TASK SET", TEXT_TASK_SET, "digest", _wr.text_task_digest(TEXT_TASKS))
TEXT_DIAGNOSTIC_BANDS = __import__(
    "jlens.mmpilot.workspace_replication", fromlist=["text_diagnostic_bands"]
).text_diagnostic_bands(LAYERS)
TEXT_DIAGNOSTIC_CONDITIONS = __import__(
    "jlens.mmpilot.workspace_replication",
    fromlist=["TEXT_DIAGNOSTIC_CONDITIONS"],
).TEXT_DIAGNOSTIC_CONDITIONS
print("TEXT TASKS", len(TEXT_TASKS))
print("  unrestricted generation passes", len(TEXT_TASKS) * 2 * 4)
print("  clean source-loading passes", len(TEXT_TASKS))
print("  Stage-1 total passes", len(TEXT_TASKS) * 2 * 4 + len(TEXT_TASKS))
print("TEXT DIAGNOSTIC UPPER BOUND — actual run uses exactly one clean-loading-selected band")
print("  old singleton/suffix grid shown only as a strict cost ceiling",
      len(TEXT_DIAGNOSTIC_BANDS), [list(band) for band in TEXT_DIAGNOSTIC_BANDS])
print("  conditions per task/band", list(TEXT_DIAGNOSTIC_CONDITIONS))
print("  unrestricted forward passes",
      len(TEXT_TASKS) * len(TEXT_DIAGNOSTIC_BANDS)
      * len(TEXT_DIAGNOSTIC_CONDITIONS) * 2)
print("  derived from same-run Stage 1",
      len(TEXT_TASKS) * 3 * 2,
      "(full-band exact/random/unrelated; no repeat forwards)")
print("  newly computed forward passes",
      len(TEXT_TASKS) * len(TEXT_DIAGNOSTIC_BANDS)
      * len(TEXT_DIAGNOSTIC_CONDITIONS) * 2 - len(TEXT_TASKS) * 3 * 2)
print("  backward passes 0")
print("  resume one two-token condition; maximum completed work lost 0")
print("DEVELOPMENT UPPER BOUND")
print("  clean loading forwards", len(CANDIDATE_PAIRS) * DEVELOPMENT_IMAGES_PER_SOURCE * 3)
print("  no intervention forwards in Stage 2")
print("FRESH CONFIRMATION UPPER BOUND")
print("  clean generation forwards",
      CONFIRMATION_IMAGES_PER_SOURCE * 3 * 2 * 2 * 2)
print("  six causal/control conditions, two directions, two-token generation",
      CONFIRMATION_IMAGES_PER_SOURCE * 3 * 2 * 2 * 6 * 2)
print("RESUME UNIT: one two-token condition JSON; completed work lost = 0")
print("MATCHED R-LENS FIT — same frozen 99 examples as each J-Lens arm")
print("  arms text + pooled; layers", list(LAYERS))
print("  each arm 99 processor examples, one forward + 320 backward passes/example")
print("  atomic checkpoint every", R_LENS_CHECKPOINT_EVERY, "examples")
print("  disconnect loses at most one incomplete checkpoint batch")
if STUDY_LAYER_WINDOW == "combined_r_l27_l40":
    print("COMBINED PAPER BAND")
    print("  reuses completed R-lens shards; fitting passes 0")
    print("  contiguous clamp", list(LAYERS), "at every original prompt position")
    print("  primary alpha", TEXT_PRIMARY_ALPHA, "(paper's double-strength swap)")
    print("  sensitivity alpha", MULTIMODAL_SENSITIVITY_ALPHA)

if REAL_MODE:
    _required_paths = [
        CORRECTED_RUN_DIR, MATCHED_LENS_RUN_DIR,
        EXPANDED_MANIFEST_CACHE, MATCHED_FIT_PLAN_PATH,
    ]
    if STUDY_LAYER_WINDOW == "combined_r_l27_l40":
        _required_paths.extend((EARLY_R_LENS_RUN_DIR, LATE_R_LENS_RUN_DIR))
    missing = [path for path in _required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("configured artifact(s) missing:\n  " + "\n  ".join(map(str, missing)))

COMBINED_R_SOURCE_PROVENANCE = None
if REAL_MODE and STUDY_LAYER_WINDOW == "combined_r_l27_l40":
    from jlens.metadata import file_sha256
    from jlens.mmpilot.store import payload_checksum
    from jlens.relprop import R_LENS_METHOD

    _source_rows = []
    _plan_digests = set()
    for _role, _root, _expected_layers in (
        ("early", EARLY_R_LENS_RUN_DIR, list(range(27, 33))),
        ("late", LATE_R_LENS_RUN_DIR, list(range(33, 41))),
    ):
        _config = json.loads((_root / "scientific_config.json").read_text())
        _inventory = json.loads((_root / "r_lens_inventory.json").read_text())
        if _config.get("model_repo_id") != MODEL_REPO_ID:
            raise RuntimeError(f"{_role} R-lens model repo does not match")
        if _config.get("model_revision") != MODEL_REVISION:
            raise RuntimeError(f"{_role} R-lens model revision does not match")
        if _config.get("r_lens_method_digest") != R_LENS_METHOD.digest:
            raise RuntimeError(f"{_role} R-lens method digest does not match")
        if list(map(int, _config.get("layers") or ())) != _expected_layers:
            raise RuntimeError(
                f"{_role} R-lens layers are {_config.get('layers')}, "
                f"not {_expected_layers}"
            )
        _plan_digests.add(str(_config.get("population_plan_digest")))
        _arms = {}
        for _arm in ("text", "pooled"):
            _record = dict(_inventory["lenses"][_arm])
            _path = Path(_record["path"])
            if not _path.is_file():
                raise FileNotFoundError(f"missing {_role} {_arm} R-lens: {_path}")
            _actual = file_sha256(str(_path))
            if _actual != _record["checksum"]:
                raise RuntimeError(
                    f"{_role} {_arm} R-lens checksum changed: "
                    f"{_actual} != {_record['checksum']}"
                )
            if int(_record["n_prompts"]) != 99:
                raise RuntimeError(
                    f"{_role} {_arm} R-lens used {_record['n_prompts']} prompts, not 99"
                )
            _arms[_arm] = {
                "path": str(_path), "checksum": _actual,
                "n_prompts": int(_record["n_prompts"]),
            }
        _source_rows.append({
            "role": _role, "run_dir": str(_root),
            "layers": _expected_layers,
            "population_plan_digest": _config.get("population_plan_digest"),
            "method_digest": _config.get("r_lens_method_digest"),
            "arms": _arms,
        })
    if len(_plan_digests) != 1:
        raise RuntimeError(
            "early and late R-lens shards were not fitted on the same frozen plan"
        )
    _combined_payload = {
        "version": "mmpilot.combined_r_lens_band.v1",
        "layers": list(LAYERS),
        "same_frozen_fit_plan": True,
        "same_method": True,
        "no_refitting": True,
        "sources": _source_rows,
    }
    COMBINED_R_SOURCE_PROVENANCE = {
        **_combined_payload,
        "source_digest": payload_checksum(_combined_payload),
    }
    print(json.dumps(COMBINED_R_SOURCE_PROVENANCE, indent=2))
'''
)

markdown("## 2b. Task-set token preflight — tokenizer only, no weights, no GPU")
code(
    r'''
if RUN_TASK_TOKEN_PREFLIGHT:
    from transformers import AutoTokenizer

    from jlens.mmpilot.workspace_replication import (
        assert_task_set_resolvable, task_set_token_preflight,
    )

    # Tokenizer alone at the pinned revision. This is the same encoder the
    # backend exposes as encode_candidate, so a concept that resolves here
    # resolves in Stage 1 too.
    _tok = AutoTokenizer.from_pretrained(MODEL_REPO_ID, revision=MODEL_REVISION)

    def _encode(text):
        return _tok.encode(text, add_special_tokens=False)

    TASK_TOKEN_PREFLIGHT = task_set_token_preflight(
        TEXT_TASKS, _encode,
        extra_concepts=("Japan", "Brazil", *CONTROL_CONCEPTS),
    )
    print("task set", TEXT_TASK_SET, "digest", TASK_TOKEN_PREFLIGHT["task_digest"])
    print("tasks", TASK_TOKEN_PREFLIGHT["n_tasks"],
          TASK_TOKEN_PREFLIGHT["tasks_by_family"])
    print("concepts", TASK_TOKEN_PREFLIGHT["n_concepts"],
          "single-token", TASK_TOKEN_PREFLIGHT["all_single_token"],
          "collision-free", TASK_TOKEN_PREFLIGHT["no_collisions"])
    for _name, _why in sorted(TASK_TOKEN_PREFLIGHT["unresolvable"].items()):
        print("  UNRESOLVABLE", _name, "--", _why.splitlines()[0])
    for _row in TASK_TOKEN_PREFLIGHT["collisions"]:
        print("  COLLISION", _row)
    print("checksum", TASK_TOKEN_PREFLIGHT["preflight_checksum"])

    # Refuses here, on CPU, rather than after the 16 GB download.
    assert_task_set_resolvable(TASK_TOKEN_PREFLIGHT)
    print("PREFLIGHT PASSED — every concept is one token; no role collides")
    print("Still unproven: whether Gemma answers each clean prompt correctly.")
    print("That is Stage 1's capability gate, and it needs the model.")
else:
    TASK_TOKEN_PREFLIGHT = None
    print("token preflight skipped: set RUN_TASK_TOKEN_PREFLIGHT to check the task set")
'''
)

markdown("## 3. Load the manifest, exclude prior causal media, and freeze development/confirmation pools")
code(
    r'''
GROUPS = []
PRIOR_EXCLUDED_IMAGES = set()
PRIOR_EXCLUDED_GROUPS = set()
if REAL_MODE:
    from jlens.mmpilot.multimodal_lens import load_completed_causal_source, select_causal_groups
    from jlens.mmpilot.store import payload_checksum

    raw = EXPANDED_MANIFEST_CACHE.read_bytes()
    MANIFEST_CHECKSUM = "sha256:" + hashlib.sha256(raw).hexdigest()
    GROUPS = [dict(row) for row in json.loads(raw)["groups"]]
    del raw
    prior = load_completed_causal_source(
        MATCHED_LENS_RUN_DIR,
        expected_final_report_checksum="sha256:875e13a8829bfd226c637ef4522d64d4d5ef91f31adcdace4942e72e75eb1e0e",
        expected_cross_report_checksum="sha256:a8536614f6e751e65ec250016852d6d614c0bc16befbfeb502e1faa148a3c69f",
        expected_causal_report_checksum="sha256:3370a2de8713024235b154ade3d7531eca491fea5592d9cf6b0397b434d573df",
        expected_lens_checksums={
            "text": "sha256:01c2591e55eda83fb17e784bb1e35fb437ee1ccf1ba556e95269c913b9596717",
            "image": "sha256:16f0a7c6dcbc36133ed28028016020cb7e8c8a8ec4c2879e283e191b04c1ef6d",
            "spoken_audio": "sha256:2f9140e28b2dd41b6f7e8e138ef0a11507d6013b1f4e95265d8e80e213936f55",
            "pooled": "sha256:7569552f1b9137ab859fe54e5d54395920c740fea94a909c8ef43623ddb5ea0e",
        },
    )
    PRIOR_EXCLUDED_IMAGES |= set(prior["excluded_image_ids"])
    _workspace_plan_sources = []
    _workspace_roots_to_exclude = (
        (EARLY_R_LENS_RUN_DIR, LATE_R_LENS_RUN_DIR)
        if STUDY_LAYER_WINDOW == "combined_r_l27_l40"
        else ()
    )
    for _workspace_root in _workspace_roots_to_exclude:
        if _workspace_root is None:
            continue
        _workspace_plan_path = _workspace_root / "population_plan.json"
        if not _workspace_plan_path.is_file():
            if STUDY_LAYER_WINDOW == "combined_r_l27_l40":
                raise FileNotFoundError(
                    "the combined-band source run has no population_plan.json: "
                    f"{_workspace_plan_path}"
                )
            continue
        _workspace_plan = json.loads(_workspace_plan_path.read_text())
        _spent_rows = [
            *(_workspace_plan.get("development") or ()),
            *(_workspace_plan.get("confirmation") or ()),
        ]
        _spent_images = {
            str(row["image_id"]) for row in _spent_rows if row.get("image_id")
        }
        _spent_groups = {
            str(row["group_id"]) for row in _spent_rows if row.get("group_id")
        }
        PRIOR_EXCLUDED_IMAGES |= _spent_images
        PRIOR_EXCLUDED_GROUPS |= _spent_groups
        _workspace_plan_sources.append({
            "run_dir": str(_workspace_root),
            "plan_digest": _workspace_plan.get("plan_digest"),
            "excluded_images": len(_spent_images),
            "excluded_groups": len(_spent_groups),
        })
    source_names = sorted({name for pair in CANDIDATE_PAIRS for name in pair})
    DEV_POOL = select_causal_groups(
        GROUPS, concepts=source_names,
        n_per_concept=DEVELOPMENT_IMAGES_PER_SOURCE,
        excluded_image_ids=sorted(PRIOR_EXCLUDED_IMAGES), seed=DEVELOPMENT_SEED,
    )
    DEV_GROUPS = [{**row, "concept": name} for name in source_names for row in DEV_POOL[name]]
    DEV_IMAGE_IDS = {str(row["image_id"]) for row in DEV_GROUPS}
    DEV_GROUP_IDS = {str(row["group_id"]) for row in DEV_GROUPS}
    CONFIRM_POOL = select_causal_groups(
        GROUPS, concepts=source_names,
        n_per_concept=CONFIRMATION_IMAGES_PER_SOURCE,
        excluded_image_ids=sorted(PRIOR_EXCLUDED_IMAGES | DEV_IMAGE_IDS),
        seed=CONFIRMATION_SEED,
    )
    CONFIRM_GROUPS = [{**row, "concept": name} for name in source_names for row in CONFIRM_POOL[name]]
    from jlens.mmpilot.workspace_replication import assert_fresh_population
    FRESHNESS = assert_fresh_population(
        CONFIRM_GROUPS,
        forbidden_image_ids=sorted(PRIOR_EXCLUDED_IMAGES | DEV_IMAGE_IDS),
        forbidden_group_ids=sorted(DEV_GROUP_IDS),
    )
    POPULATION_PLAN = {
        "manifest_checksum": MANIFEST_CHECKSUM,
        "prior_workspace_population_sources": _workspace_plan_sources,
        "prior_workspace_excluded_image_ids": sorted(PRIOR_EXCLUDED_IMAGES),
        "prior_workspace_excluded_group_ids": sorted(PRIOR_EXCLUDED_GROUPS),
        "development": [{"group_id": r["group_id"], "image_id": r["image_id"], "concept": r.get("concept")} for r in DEV_GROUPS],
        "confirmation": [{"group_id": r["group_id"], "image_id": r["image_id"], "concept": r.get("concept")} for r in CONFIRM_GROUPS],
        "freshness": FRESHNESS,
    }
    POPULATION_PLAN["plan_digest"] = payload_checksum(POPULATION_PLAN)
    print("development images", len(DEV_IMAGE_IDS))
    print("confirmation images", len({str(r['image_id']) for r in CONFIRM_GROUPS}))
    print("freshness", FRESHNESS)
else:
    MANIFEST_CHECKSUM = "mock"
    DEV_GROUPS = CONFIRM_GROUPS = []
    DEV_IMAGE_IDS = DEV_GROUP_IDS = set()
    POPULATION_PLAN = {"plan_digest": "mock"}
'''
)

markdown("## 4. Fingerprinted run and atomic unit store")
code(
    r'''
from jlens.mmpilot.store import RunFingerprint, UnitStore, payload_checksum
from jlens.mmpilot.workspace_replication import (
    PROTOCOL_VERSION, TEXT_ANSWER_MATCH_RULE, TEXT_COMPLETION_INSTRUCTION,
    TEXT_DIAGNOSTIC_VERSION, TEXT_POST_CAST_MAX_RELATIVE_ERROR,
    TEXT_INPUT_PROTOCOL_VERSION, TEXT_MAX_NEW_TOKENS, TEXT_OUTPUT_ENDPOINT_VERSION,
    TEXT_MODEL_DTYPE_REALIZATION,
    text_task_digest,
)
from jlens.relprop import R_LENS_METHOD

SCIENTIFIC_CONFIG = {
    "protocol": PROTOCOL_VERSION,
    "text_input_protocol": TEXT_INPUT_PROTOCOL_VERSION,
    "text_completion_instruction": TEXT_COMPLETION_INSTRUCTION,
    "text_answer_match_rule": TEXT_ANSWER_MATCH_RULE,
    "output_endpoint": TEXT_OUTPUT_ENDPOINT_VERSION,
    "max_new_tokens": TEXT_MAX_NEW_TOKENS,
    "model_repo_id": MODEL_REPO_ID, "model_revision": MODEL_REVISION,
    "study_layer_window": STUDY_LAYER_WINDOW,
    "audio_protocol_fingerprint": AUDIO_PROTOCOL_FINGERPRINT,
    "layers": list(LAYERS), "text_task_digest": text_task_digest(TEXT_TASKS),
    "text_diagnostic": {
        "version": TEXT_DIAGNOSTIC_VERSION,
        "bands": "exactly_one_clean_loading_selected_contiguous_band",
        "causal_outcome_may_select_band": False,
        "conditions": list(TEXT_DIAGNOSTIC_CONDITIONS),
        "alpha": TEXT_PRIMARY_ALPHA,
        "random_seed": TEXT_DIAGNOSTIC_RANDOM_SEED,
        "post_cast_max_relative_coordinate_error": TEXT_POST_CAST_MAX_RELATIVE_ERROR,
        "model_dtype_realization": TEXT_MODEL_DTYPE_REALIZATION.to_dict(),
        "primary_endpoint": "unrestricted_greedy_complete_answer",
        "teacher_forcing_used": False,
        "selection_is_development_only": True,
        "fresh_confirmation_required": True,
    },
    "candidate_pairs": [list(pair) for pair in CANDIDATE_PAIRS],
    "control_concepts": list(CONTROL_CONCEPTS),
    "implicit_unrelated_concepts": list(IMPLICIT_UNRELATED_CONCEPTS),
    "text_task_set": TEXT_TASK_SET,
    "primary_alpha": TEXT_PRIMARY_ALPHA,
    "primary_alpha_overridden": TEXT_PRIMARY_ALPHA_OVERRIDE is not None,
    "sensitivity_alpha": MULTIMODAL_SENSITIVITY_ALPHA,
    "population_plan_digest": POPULATION_PLAN["plan_digest"],
    "min_source_advantage": MIN_SOURCE_ADVANTAGE,
    "min_source_cosine": MIN_SOURCE_COSINE,
    "loading_first_selection": "mmpilot.loading_first_instrument_selection.v1",
    "r_lens_method": R_LENS_METHOD.to_dict(),
    "r_lens_method_digest": R_LENS_METHOD.digest,
    "r_lens_fit_arms": ["text", "pooled"],
    "r_lens_checkpoint_every": R_LENS_CHECKPOINT_EVERY,
    "combined_r_source_provenance": COMBINED_R_SOURCE_PROVENANCE,
    "evidence_position_margin": EVIDENCE_POSITION_MARGIN,
    "minimum_confirmation_success_rate": MIN_CONFIRMATION_SUCCESS_RATE,
    "confirmation_familywise_alpha": CONFIRMATION_FAMILYWISE_ALPHA,
    # Reporting-only amendments must not strand expensive checksum-valid
    # scientific units in a new run namespace. This is the exact commit whose
    # model-facing implementation generated the existing run.
    "commit": SCIENTIFIC_IMPLEMENTATION_ID,
}
FINGERPRINT_DIGEST = payload_checksum(SCIENTIFIC_CONFIG)
RUN_DIR = RUNS_ROOT / "mmworkspace" / f"mmworkspace_{'real' if REAL_MODE else 'mock'}_{FINGERPRINT_DIGEST.split(':')[-1][:12]}"
RUN_DIR.mkdir(parents=True, exist_ok=True)
FINGERPRINT = RunFingerprint(
    mode="real" if REAL_MODE else "mock", model_repo_id=MODEL_REPO_ID,
    model_revision=MODEL_REVISION, processor_revision=MODEL_REVISION,
    layers=LAYERS,
    lens_checksum=payload_checksum({
        "corrected_text_run": str(CORRECTED_RUN_DIR),
        "matched_lens_run": str(MATCHED_LENS_RUN_DIR),
    }),
    manifest_checksum=MANIFEST_CHECKSUM,
    split_id=POPULATION_PLAN["plan_digest"],
    intervention_config=SCIENTIFIC_CONFIG,
)
STORE = UnitStore(RUN_DIR, FINGERPRINT)
STORE.open()
(RUN_DIR / "scientific_config.json").write_text(json.dumps(SCIENTIFIC_CONFIG, indent=2))
(RUN_DIR / "population_plan.json").write_text(json.dumps(POPULATION_PLAN, indent=2, default=str))
print("run", RUN_DIR)
print("resume", STORE.status_report())
'''
)

markdown("## 5. Load Gemma and the two independently sourced lens families")
code(
    r'''
BACKEND = None
TEXT_TOKEN_VECTORS = {}
MATCHED_LENSES = {}
R_LENSES = {}
R_LENS_SOURCE_PATHS = {}
INSTRUMENT_VECTORS = {}
if REAL_MODE and MODEL_STAGE and CONFIRM_MODEL_LOAD:
    import getpass, torch
    from jlens.lens import JacobianLens
    from jlens.mmpilot.real_backend import build_real_backend
    from jlens.mmpilot.tri_modal import assert_audio_protocol
    from jlens.mmpilot.validated_band_followup import (
        discover_corrected_band_lenses, read_corrected_validation_report,
    )
    if not os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = getpass.getpass("HF_TOKEN (input hidden): ").strip()
    BUNDLE = build_real_backend(
        MODEL_REPO_ID, revision=MODEL_REVISION, token=os.environ["HF_TOKEN"],
        device="cuda", allow_model_load=True, resolve_audio=True,
        expect_n_layers=EXPECT_N_LAYERS, expect_d_model=EXPECT_D_MODEL,
        expect_vocab_size=EXPECT_VOCAB,
    )
    if BUNDLE.audio_interface is None:
        raise RuntimeError("native spoken audio did not resolve: " + BUNDLE.audio_blocked_reason)
    assert_audio_protocol(BUNDLE.audio_interface, expected_fingerprint=AUDIO_PROTOCOL_FINGERPRINT)
    BACKEND = BUNDLE.backend
    from jlens.mmpilot.workspace_replication import semantic_answer_concept
    names = sorted(
        {task.source for task in TEXT_TASKS}
        | {task.target for task in TEXT_TASKS}
        | {semantic_answer_concept(task.clean_answer) for task in TEXT_TASKS}
        | {semantic_answer_concept(task.swapped_answer) for task in TEXT_TASKS}
        | {"Japan", "Brazil"}
        | set(CONTROL_CONCEPTS)
        | set(IMPLICIT_UNRELATED_CONCEPTS)
    )
    from jlens.mmpilot.coordinate_swap import resolve_concept_token
    TEXT_CONCEPT_TOKENS = {name: resolve_concept_token(BACKEND.encode_candidate, name) for name in names}
    for arm in ("text", "image", "spoken_audio", "pooled"):
        MATCHED_LENSES[arm] = JacobianLens.load(str(MATCHED_LENS_RUN_DIR / "lenses" / f"lens.{arm}.pt"))
    from jlens.mmpilot.multimodal_lens import (
        fit_arm, plan_units, selected_lens_vector,
    )

    def _vectors_for(lens):
        return {
            layer: {
                name: selected_lens_vector(
                    lens, BACKEND.unembedding_weight(),
                    layer=layer, token_id=token.token_id,
                )
                for name, token in TEXT_CONCEPT_TOKENS.items()
            }
            for layer in LAYERS
        }

    if STUDY_LAYER_WINDOW == "late_jr_l33_l40":
        corrected_path, corrected = read_corrected_validation_report(
            CORRECTED_RUN_DIR, expected_model_repo_id=MODEL_REPO_ID,
            expected_model_revision=BUNDLE.model_revision,
        )
        CORRECTED_ARTIFACTS, _ = discover_corrected_band_lenses(
            CORRECTED_RUN_DIR, report=corrected, layers=LAYERS,
        )
        unembedding = BACKEND.unembedding_weight()
        rows = {
            name: unembedding[token.token_id].detach().float().cpu()
            for name, token in TEXT_CONCEPT_TOKENS.items()
        }
        loaded = {}
        for layer in LAYERS:
            source = CORRECTED_ARTIFACTS[layer]
            loaded.setdefault(
                source.lens_path, JacobianLens.load(source.lens_path)
            )
            jacobian = loaded[source.lens_path].jacobians[
                source.layer_key_in_file
            ].detach().float().cpu()
            TEXT_TOKEN_VECTORS[layer] = {
                name: row @ jacobian for name, row in rows.items()
            }
        del loaded, rows, unembedding
        INSTRUMENT_VECTORS["published_text_j"] = TEXT_TOKEN_VECTORS
        for _arm in ("text", "pooled"):
            if set(LAYERS).issubset(MATCHED_LENSES[_arm].jacobians):
                INSTRUMENT_VECTORS[f"matched_{_arm}_j"] = _vectors_for(
                    MATCHED_LENSES[_arm]
                )
    elif STUDY_LAYER_WINDOW in ("early_r_l27_l32", "mid_r_l21_l29"):
        print(
            STUDY_LAYER_WINDOW,
            "R-lens mode: late J-lens artifacts are historical controls "
            "only and are not applied outside their fitted layer grid",
        )
    else:
        print(
            "combined R-lens mode: joining the completed L27-L32 and L33-L40 "
            "R-lens shards into one contiguous L27-L40 intervention band"
        )
        from jlens.mmpilot.loading_first import combine_disjoint_layer_lenses

        for _arm in ("text", "pooled"):
            _shards = []
            for _source in COMBINED_R_SOURCE_PROVENANCE["sources"]:
                _shards.append(
                    JacobianLens.load(_source["arms"][_arm]["path"])
                )
            R_LENSES[_arm] = combine_disjoint_layer_lenses(
                _shards, expected_layers=LAYERS
            )
            print(
                "R-lens", _arm, "combined without refitting",
                R_LENSES[_arm].source_layers,
            )

    _fit_plan = json.loads(MATCHED_FIT_PLAN_PATH.read_text())
    from jlens.mmpilot.media_io import RetryJournal, drive_media_loaders
    _fit_media = drive_media_loaders(journal=RetryJournal())

    def _build_r_fit_inputs(unit):
        if unit.modality == "text":
            return BACKEND.build_inputs(prompt=unit.prompt, modality="text")
        if unit.modality == "image":
            return BACKEND.build_inputs(
                prompt=unit.prompt, modality="image",
                image=_fit_media["load_image"](unit.image_path),
                media_path=unit.image_path,
            )
        waveform, rate = _fit_media["load_audio"](unit.audio_path)
        return BACKEND.build_inputs(
            prompt=unit.prompt, modality="spoken_audio", audio=waveform,
            sampling_rate=rate, media_path=unit.audio_path,
        )

    from jlens.relprop import audit_dense_relprop_architecture, dense_relprop_backward
    R_LENS_ARCHITECTURE_AUDIT = audit_dense_relprop_architecture(
        BACKEND.hf_model.model.language_model
    )
    for _arm in ("text", "pooled"):
        _path = RUN_DIR / "r_lenses" / f"lens.{_arm}.pt"
        _early_path = (
            EARLY_R_LENS_RUN_DIR / "r_lenses" / f"lens.{_arm}.pt"
            if EARLY_R_LENS_RUN_DIR is not None
            else None
        )
        if STUDY_LAYER_WINDOW == "combined_r_l27_l40":
            pass
        elif _path.is_file():
            R_LENSES[_arm] = JacobianLens.load(str(_path))
            R_LENS_SOURCE_PATHS[_arm] = _path
            print("R-lens", _arm, "reused", _path)
        elif (
            STUDY_LAYER_WINDOW == "early_r_l27_l32"
            and _early_path is not None
            and _early_path.is_file()
        ):
            # The standalone L27-32 window has its own completed shard from
            # mmworkspace_real_62ef81c904c5 -- the same one combined_r_l27_l40
            # joins without refitting. A fresh TEXT_TASK_SET changes this run's
            # fingerprint and therefore RUN_DIR, so the RUN_DIR-local reuse
            # check above never finds it; without this branch every task-set
            # change would silently re-trigger a full Stage 0 fit.
            _candidate = JacobianLens.load(str(_early_path))
            if not set(LAYERS).issubset(_candidate.jacobians):
                raise RuntimeError(
                    f"{_early_path} does not cover layers {list(LAYERS)}; "
                    "refusing to reuse an incompatible R-lens shard"
                )
            R_LENSES[_arm] = _candidate
            R_LENS_SOURCE_PATHS[_arm] = _early_path
            print("R-lens", _arm, "reused from", _early_path)
        elif RUN_STAGE0_FIT_MATCHED_R_LENSES and CONFIRM_R_LENS_FIT_BUDGET:
            _units = plan_units(_fit_plan, _arm)
            R_LENSES[_arm] = fit_arm(
                BACKEND, _units, build_inputs=_build_r_fit_inputs,
                source_layers=LAYERS, target_layer=41,
                checkpoint_path=(
                    RUN_DIR / "r_lenses" / "checkpoints" /
                    f"{_arm}.jacobian_sum.pt"
                ),
                arm=_arm,
                scientific_fingerprint=FINGERPRINT_DIGEST + ":r_lens:" + _arm,
                dim_batch=R_LENS_DIM_BATCH, skip_first=16,
                checkpoint_every=R_LENS_CHECKPOINT_EVERY,
                backward_context=lambda: dense_relprop_backward(
                    BACKEND.hf_model.model.language_model
                ),
                progress=lambda row: print(
                    f"R-lens {row['arm']} {row['index']}/{row['total']} "
                    f"checkpoint={row['checkpoint_written']}"
                ) if (
                    row["index"] == 1 or row["checkpoint_written"]
                    or row["index"] == row["total"]
                ) else None,
            )
            _path.parent.mkdir(parents=True, exist_ok=True)
            _tmp = _path.with_suffix(".tmp.pt")
            R_LENSES[_arm].save(str(_tmp))
            os.replace(_tmp, _path)
            R_LENS_SOURCE_PATHS[_arm] = _path
            print("R-lens", _arm, "completed", _path)
        if _arm in R_LENSES:
            INSTRUMENT_VECTORS[f"matched_{_arm}_r"] = _vectors_for(R_LENSES[_arm])
    if R_LENSES:
        from jlens.metadata import file_sha256
        if STUDY_LAYER_WINDOW == "combined_r_l27_l40":
            _inventory = {
                "method_digest": R_LENS_METHOD.digest,
                "architecture_audit_checksum": R_LENS_ARCHITECTURE_AUDIT[
                    "audit_checksum"
                ],
                "combined_without_refitting": True,
                "combined_source_digest": COMBINED_R_SOURCE_PROVENANCE[
                    "source_digest"
                ],
                "layers": list(LAYERS),
                "lenses": {
                    arm: {
                        "n_prompts": lens.n_prompts,
                        "source_artifacts": [
                            source["arms"][arm]
                            for source in COMBINED_R_SOURCE_PROVENANCE["sources"]
                        ],
                    }
                    for arm, lens in sorted(R_LENSES.items())
                },
            }
        else:
            _inventory = {
                "method_digest": R_LENS_METHOD.digest,
                "architecture_audit_checksum": R_LENS_ARCHITECTURE_AUDIT[
                    "audit_checksum"
                ],
                "lenses": {
                    # The shard may live in this RUN_DIR (freshly fitted) or in
                    # the completed early run (reused without refitting), so the
                    # inventory records where it actually came from rather than
                    # assuming RUN_DIR.
                    arm: {
                        "path": str(R_LENS_SOURCE_PATHS[arm]),
                        "checksum": file_sha256(str(R_LENS_SOURCE_PATHS[arm])),
                        "n_prompts": lens.n_prompts,
                        "reused_without_refitting": (
                            R_LENS_SOURCE_PATHS[arm].parent.parent != RUN_DIR
                        ),
                    }
                    for arm, lens in sorted(R_LENSES.items())
                },
            }
        _inventory["inventory_checksum"] = payload_checksum(_inventory)
        _inventory_path = RUN_DIR / "r_lens_inventory.json"
        if _inventory_path.is_file():
            _recorded = json.loads(_inventory_path.read_text())
            if _recorded != _inventory:
                raise RuntimeError(
                    "R-lens artifacts changed after this run was opened; "
                    "refusing to reuse loading or intervention units"
                )
        else:
            _inventory_path.write_text(json.dumps(_inventory, indent=2))
    print("validated text layers", sorted(TEXT_TOKEN_VECTORS))
    print("matched lenses", sorted(MATCHED_LENSES))
    print("loading-first instruments", sorted(INSTRUMENT_VECTORS))
elif MODEL_STAGE:
    print("skipped: model load is not confirmed")
'''
)

markdown("## 6. Stage 1 — paper-task text replication and source-loading audit")
code(
    r'''
TEXT_VERDICT = STORE.load("metric", "text_replication_verdict")
LOADING_FIRST_SELECTION = STORE.load("metric", "loading_first_selection")
ACTIVE_TEXT_LAYERS = tuple(LAYERS)
SELECTED_INSTRUMENT = None
if LOADING_FIRST_SELECTION is not None:
    SELECTED_INSTRUMENT = LOADING_FIRST_SELECTION.get("selected_instrument")
    ACTIVE_TEXT_LAYERS = tuple(LOADING_FIRST_SELECTION.get("selected_band") or LAYERS)
    if SELECTED_INSTRUMENT in INSTRUMENT_VECTORS:
        TEXT_TOKEN_VECTORS = INSTRUMENT_VECTORS[SELECTED_INSTRUMENT]
if REAL_MODE and RUN_STAGE1_TEXT_REPLICATION and CONFIRM_MODEL_LOAD:
    from jlens.mmpilot.coordinate_swap import (
        build_swap_basis_from_vectors, random_two_direction_basis,
    )
    from jlens.mmpilot.store import safe_key
    from jlens.mmpilot.workspace_replication import (
        TEXT_MAX_NEW_TOKENS, build_assistant_prefill_completion_inputs,
        capture_source_loading, text_capability_verdict,
        semantic_answer_concept,
        text_replication_verdict,
        unrestricted_greedy_completion, unrestricted_greedy_swap_trial,
    )

    # Capability is a distinct first gate. No basis is built and no hook is
    # installed unless every clean completion succeeds.
    capability_rows = []
    for task in TEXT_TASKS:
        key = safe_key("text-paper-capability", task.task_id)
        stored = STORE.load("capability", key)
        if stored is None:
            inputs = build_assistant_prefill_completion_inputs(BACKEND, task.prompt)
            if not (
                inputs.route.get("chat_template_used") is True
                and inputs.route.get("continue_final_message") is True
            ):
                raise RuntimeError("Stage 1 requires the assistant-prefill route")
            clean = unrestricted_greedy_completion(
                BACKEND, inputs, answer=task.clean_answer,
                max_new_tokens=TEXT_MAX_NEW_TOKENS,
                diagnostic_token_ids={
                    "swapped_answer_head": TEXT_CONCEPT_TOKENS[
                        semantic_answer_concept(task.swapped_answer)
                    ].token_id
                },
            )
            stored = {
                "task_id": task.task_id,
                "task": task.to_dict(),
                "input_route": dict(inputs.route),
                "clean_correct": bool(clean["answer_match"]),
                "clean": clean,
                "intervention_ran": False,
            }
            STORE.save("capability", key, stored)
            work = "computed"
        else:
            work = "reused"
        capability_rows.append(stored)
        print(
            "capability", task.task_id, work,
            "expected", repr(task.clean_answer),
            "generated", repr(stored["clean"]["generated_text"]),
            "pass", stored["clean_correct"],
        )

    TEXT_CAPABILITY = text_capability_verdict(capability_rows, tasks=TEXT_TASKS)
    STORE.save("metric", "text_capability_verdict", TEXT_CAPABILITY)
    print(json.dumps(TEXT_CAPABILITY, indent=2))

    # This is intentionally before the first causal hook.  Every candidate
    # instrument is judged on the same clean assistant-prefill activations;
    # intervention outcomes cannot influence the instrument or band.
    from jlens.mmpilot.loading_first import select_loading_instrument
    loading_by_instrument = {}
    for instrument, vectors in sorted(INSTRUMENT_VECTORS.items()):
        instrument_rows = []
        for task in TEXT_TASKS:
            key = safe_key("loading-first", instrument, task.task_id)
            stored = STORE.load("activation", key)
            if stored is None:
                inputs = build_assistant_prefill_completion_inputs(
                    BACKEND, task.prompt
                )
                controls = (
                    IMPLICIT_UNRELATED_CONCEPTS
                    if task.family == "implicit_two_hop"
                    else ("Japan", "Brazil")
                )
                stored = {
                    "instrument": instrument,
                    "task_id": task.task_id,
                    "rows": capture_source_loading(
                        BACKEND, inputs, vectors_by_layer=vectors,
                        source=task.source, target=task.target,
                        unrelated=controls, sample_id=task.task_id,
                        modality="text",
                    ),
                    "causal_result_consulted": False,
                }
                STORE.save("activation", key, stored)
                work = "computed"
            else:
                work = "reused"
            instrument_rows.extend(stored["rows"])
            print("clean loading", instrument, task.task_id, work)
        loading_by_instrument[instrument] = instrument_rows

    LOADING_FIRST_SELECTION = select_loading_instrument(
        loading_by_instrument,
        tasks=[task.task_id for task in TEXT_TASKS],
        layers=LAYERS,
        position_class="final_prompt_token",
        min_source_cosine=MIN_SOURCE_COSINE,
        min_source_advantage=MIN_SOURCE_ADVANTAGE,
    )
    STORE.save("metric", "loading_first_selection", LOADING_FIRST_SELECTION)
    (RUN_DIR / "loading_first_selection.json").write_text(
        json.dumps(LOADING_FIRST_SELECTION, indent=2, default=str)
    )
    SELECTED_INSTRUMENT = LOADING_FIRST_SELECTION["selected_instrument"]
    ACTIVE_TEXT_LAYERS = tuple(LOADING_FIRST_SELECTION["selected_band"])
    if SELECTED_INSTRUMENT is not None:
        TEXT_TOKEN_VECTORS = INSTRUMENT_VECTORS[SELECTED_INSTRUMENT]
    print(json.dumps(LOADING_FIRST_SELECTION, indent=2))

    text_rows = []
    if (
        TEXT_CAPABILITY["causal_spending_licensed"]
        and LOADING_FIRST_SELECTION["verdict"]
        == "LOADING_FIRST_INSTRUMENT_GO"
    ):
        for task in TEXT_TASKS:
            key = safe_key("text-paper", task.task_id)
            stored = STORE.load("intervention", key)
            if stored is not None:
                text_rows.append(stored)
                print(
                    task.task_id, "reused", "swapped answer generated",
                    stored.get(
                        "exact_primary_swapped_answer_generated",
                        stored.get("exact_alpha1_swapped_answer_generated"),
                    ),
                )
                continue

            clean_record = next(
                row for row in capability_rows if row["task_id"] == task.task_id
            )
            inputs = build_assistant_prefill_completion_inputs(BACKEND, task.prompt)
            bases = {
                layer: build_swap_basis_from_vectors(
                    TEXT_TOKEN_VECTORS[layer][task.source], TEXT_TOKEN_VECTORS[layer][task.target],
                    layer=layer, source=TEXT_CONCEPT_TOKENS[task.source], target=TEXT_CONCEPT_TOKENS[task.target],
                ) for layer in ACTIVE_TEXT_LAYERS
            }
            random_bases = {layer: random_two_direction_basis(basis, seed=20260820 + layer) for layer, basis in bases.items()}
            # The unrelated basis must be inert for the queried property.
            # (zebra, giraffe) is not: giraffe reads as yellow/orange and both
            # are African, so it could produce a color or continent swapped
            # answer by itself and fail the control spuriously -- and zebra is
            # a source concept in the expanded set. CONTROL_CONCEPTS is the
            # study's own declared control_concepts, inert for legs/color/continent.
            controls = (
                IMPLICIT_UNRELATED_CONCEPTS
                if task.family == "implicit_two_hop"
                else ("Japan", "Brazil")
            )
            unrelated = {
                layer: build_swap_basis_from_vectors(
                    TEXT_TOKEN_VECTORS[layer][controls[0]], TEXT_TOKEN_VECTORS[layer][controls[1]],
                    layer=layer, source=TEXT_CONCEPT_TOKENS[controls[0]], target=TEXT_CONCEPT_TOKENS[controls[1]],
                ) for layer in ACTIVE_TEXT_LAYERS
            }
            exact = unrestricted_greedy_swap_trial(
                BACKEND, inputs, bases=bases, alpha=TEXT_PRIMARY_ALPHA,
                answer=task.swapped_answer,
                max_new_tokens=TEXT_MAX_NEW_TOKENS,
                diagnostic_token_ids={
                    "swapped_answer_head": TEXT_CONCEPT_TOKENS[
                        semantic_answer_concept(task.swapped_answer)
                    ].token_id
                },
                realization_policy=TEXT_MODEL_DTYPE_REALIZATION,
            )
            random = unrestricted_greedy_swap_trial(
                BACKEND, inputs, bases=random_bases, alpha=TEXT_PRIMARY_ALPHA,
                answer=task.swapped_answer,
                max_new_tokens=TEXT_MAX_NEW_TOKENS,
                diagnostic_token_ids={
                    "swapped_answer_head": TEXT_CONCEPT_TOKENS[
                        semantic_answer_concept(task.swapped_answer)
                    ].token_id
                },
                realization_policy=TEXT_MODEL_DTYPE_REALIZATION,
            )
            unrelated_result = unrestricted_greedy_swap_trial(
                BACKEND, inputs, bases=unrelated, alpha=TEXT_PRIMARY_ALPHA,
                answer=task.swapped_answer,
                max_new_tokens=TEXT_MAX_NEW_TOKENS,
                diagnostic_token_ids={
                    "swapped_answer_head": TEXT_CONCEPT_TOKENS[
                        semantic_answer_concept(task.swapped_answer)
                    ].token_id
                },
                realization_policy=TEXT_MODEL_DTYPE_REALIZATION,
            )
            stored = {
                "task_id": task.task_id, "task": task.to_dict(),
                "input_route": dict(inputs.route),
                "output_endpoint": "unrestricted_greedy_complete_answer",
                "clean_correct": bool(clean_record["clean_correct"]),
                "primary_alpha": TEXT_PRIMARY_ALPHA,
                "exact_primary_swapped_answer_generated": bool(exact["answer_match"]),
                "random_primary_swapped_answer_generated": bool(random["answer_match"]),
                "unrelated_primary_swapped_answer_generated": bool(
                    unrelated_result["answer_match"]
                ),
                "clean": clean_record["clean"],
                "exact": exact, "random": random, "unrelated": unrelated_result,
                "loading_rows": [
                    row
                    for row in loading_by_instrument[SELECTED_INSTRUMENT]
                    if row["sample_id"] == task.task_id
                ],
                "selected_instrument": SELECTED_INSTRUMENT,
                "selected_band": list(ACTIVE_TEXT_LAYERS),
            }
            if TEXT_PRIMARY_ALPHA == 1.0:
                stored.update({
                    "exact_alpha1_swapped_answer_generated": bool(
                        exact["answer_match"]
                    ),
                    "random_swapped_answer_generated": bool(
                        random["answer_match"]
                    ),
                    "unrelated_swapped_answer_generated": bool(
                        unrelated_result["answer_match"]
                    ),
                })
            STORE.save("intervention", key, stored)
            text_rows.append(stored)
            print(
                task.task_id, "computed",
                "swapped answer generated",
                stored["exact_primary_swapped_answer_generated"],
            )
        TEXT_VERDICT = text_replication_verdict(
            text_rows, primary_alpha=TEXT_PRIMARY_ALPHA, task_set=TEXT_TASKS
        )
    else:
        _blocked_verdict = (
            "TEXT_LOADING_FIRST_NO_GO"
            if TEXT_CAPABILITY["causal_spending_licensed"]
            else "TEXT_PAPER_CAPABILITY_NO_GO"
        )
        TEXT_VERDICT = {
            **{
                key: value
                for key, value in TEXT_CAPABILITY.items()
                if key != "report_checksum"
            },
            "verdict": _blocked_verdict,
            "multimodal_stage_licensed": False,
            "interventions_run": False,
            "loading_first_selection": LOADING_FIRST_SELECTION,
        }
        TEXT_VERDICT["report_checksum"] = payload_checksum(TEXT_VERDICT)
    STORE.save("metric", "text_replication_verdict", TEXT_VERDICT)
    print(json.dumps(TEXT_VERDICT, indent=2))
elif RUN_STAGE1_TEXT_REPLICATION:
    print("skipped: real mode/model confirmation required")
'''
)

markdown("## 6b. Bounded text diagnostic — audited single layers and suffix bands")
code(
    r'''
TEXT_DIAGNOSTIC_REPORT = STORE.load("metric", "text_swap_diagnostic_report")
if (
    REAL_MODE
    and RUN_STAGE1B_TEXT_DIAGNOSTIC
    and CONFIRM_MODEL_LOAD
    and CONFIRM_TEXT_DIAGNOSTIC_BUDGET
    # An explicit TEXT_PRIMARY_ALPHA_OVERRIDE is a deliberate power calibration
    # and must reach the diagnostic; without this clause setting the override
    # would silently skip Stage 1B, which is how the alpha=2 combined run
    # produced a bare verdict with no integrity audit at all.
    and (TEXT_PRIMARY_ALPHA == 1.0 or TEXT_PRIMARY_ALPHA_OVERRIDE is not None)
):
    from jlens.mmpilot.coordinate_swap import (
        build_swap_basis_from_vectors, random_two_direction_basis,
    )
    from jlens.mmpilot.store import safe_key
    from jlens.mmpilot.workspace_replication import (
        build_assistant_prefill_completion_inputs, semantic_answer_concept,
        text_capability_verdict,
        text_swap_diagnostic_report, unrestricted_greedy_direct_answer_trial,
        unrestricted_greedy_swap_trial,
    )

    # Reconstruct the clean gate from checksum-valid units. The diagnostic
    # never runs if capability did not pass in this fingerprinted run.
    diagnostic_clean_rows = []
    for task in TEXT_TASKS:
        stored = STORE.load(
            "capability", safe_key("text-paper-capability", task.task_id)
        )
        if stored is None:
            raise RuntimeError(
                "Stage 6b needs Stage 1 capability units from this run. "
                "Enable RUN_STAGE1_TEXT_REPLICATION and rerun from the top."
            )
        diagnostic_clean_rows.append(stored)
    diagnostic_capability = text_capability_verdict(
        diagnostic_clean_rows, tasks=TEXT_TASKS
    )
    if not diagnostic_capability["causal_spending_licensed"]:
        raise RuntimeError("text capability did not license the diagnostic")

    diagnostic_records = []
    computed = reused = derived = 0
    TEXT_DIAGNOSTIC_BANDS = (tuple(ACTIVE_TEXT_LAYERS),)
    total = (
        len(TEXT_TASKS) * len(TEXT_DIAGNOSTIC_BANDS)
        * len(TEXT_DIAGNOSTIC_CONDITIONS)
    )
    for task in TEXT_TASKS:
        baseline = STORE.load(
            "intervention", safe_key("text-paper", task.task_id)
        )
        if baseline is None:
            raise RuntimeError(
                "Stage 6b needs the Stage 1 full-band intervention unit from "
                "this run. Enable RUN_STAGE1_TEXT_REPLICATION and rerun."
            )
        inputs = build_assistant_prefill_completion_inputs(BACKEND, task.prompt)
        answer_name = semantic_answer_concept(task.swapped_answer)
        diagnostic_tokens = {
            "swapped_answer_head": TEXT_CONCEPT_TOKENS[answer_name].token_id
        }
        bases_all = {
            layer: build_swap_basis_from_vectors(
                TEXT_TOKEN_VECTORS[layer][task.source],
                TEXT_TOKEN_VECTORS[layer][task.target],
                layer=layer,
                source=TEXT_CONCEPT_TOKENS[task.source],
                target=TEXT_CONCEPT_TOKENS[task.target],
            )
            for layer in ACTIVE_TEXT_LAYERS
        }
        random_all = {
            layer: random_two_direction_basis(
                basis, seed=TEXT_DIAGNOSTIC_RANDOM_SEED + layer
            )
            for layer, basis in bases_all.items()
        }
        control_names = (
            IMPLICIT_UNRELATED_CONCEPTS
            if task.family == "implicit_two_hop"
            else ("Japan", "Brazil")
        )
        unrelated_all = {
            layer: build_swap_basis_from_vectors(
                TEXT_TOKEN_VECTORS[layer][control_names[0]],
                TEXT_TOKEN_VECTORS[layer][control_names[1]],
                layer=layer,
                source=TEXT_CONCEPT_TOKENS[control_names[0]],
                target=TEXT_CONCEPT_TOKENS[control_names[1]],
            )
            for layer in ACTIVE_TEXT_LAYERS
        }
        answer_vectors_all = {
            layer: TEXT_TOKEN_VECTORS[layer][answer_name]
            for layer in ACTIVE_TEXT_LAYERS
        }

        for band in TEXT_DIAGNOSTIC_BANDS:
            band_name = "L" + "-".join(map(str, band))
            bases = {layer: bases_all[layer] for layer in band}
            random_bases = {layer: random_all[layer] for layer in band}
            unrelated_bases = {layer: unrelated_all[layer] for layer in band}
            answer_vectors = {layer: answer_vectors_all[layer] for layer in band}
            for condition in TEXT_DIAGNOSTIC_CONDITIONS:
                key = safe_key(
                    "text-diagnostic", task.task_id, band_name, condition
                )
                stored = STORE.load("intervention", key)
                if stored is None:
                    work = "computed"
                    baseline_field = {
                        "exact_alpha1": "exact",
                        "random_alpha1": "random",
                        "unrelated_alpha1": "unrelated",
                    }.get(condition)
                    if (
                        tuple(band) == tuple(ACTIVE_TEXT_LAYERS)
                        and baseline_field is not None
                    ):
                        result = dict(baseline[baseline_field])
                        derived += 1
                        work = "derived_from_stage1"
                    elif condition == "exact_alpha1":
                        result = unrestricted_greedy_swap_trial(
                            BACKEND, inputs, bases=bases, alpha=1.0,
                            answer=task.swapped_answer,
                            diagnostic_token_ids=diagnostic_tokens,
                            realization_policy=TEXT_MODEL_DTYPE_REALIZATION,
                        )
                    elif condition == "zero":
                        result = unrestricted_greedy_swap_trial(
                            BACKEND, inputs, bases=bases, alpha=0.0,
                            answer=task.swapped_answer,
                            diagnostic_token_ids=diagnostic_tokens,
                            realization_policy=TEXT_MODEL_DTYPE_REALIZATION,
                        )
                    elif condition == "random_alpha1":
                        result = unrestricted_greedy_swap_trial(
                            BACKEND, inputs, bases=random_bases, alpha=1.0,
                            answer=task.swapped_answer,
                            diagnostic_token_ids=diagnostic_tokens,
                            realization_policy=TEXT_MODEL_DTYPE_REALIZATION,
                        )
                    elif condition == "unrelated_alpha1":
                        result = unrestricted_greedy_swap_trial(
                            BACKEND, inputs, bases=unrelated_bases, alpha=1.0,
                            answer=task.swapped_answer,
                            diagnostic_token_ids=diagnostic_tokens,
                            realization_policy=TEXT_MODEL_DTYPE_REALIZATION,
                        )
                    elif condition == "direct_answer_norm_matched":
                        result = unrestricted_greedy_direct_answer_trial(
                            BACKEND, inputs, bases=bases,
                            answer_vectors=answer_vectors,
                            answer=task.swapped_answer,
                            diagnostic_token_ids=diagnostic_tokens,
                            realization_policy=TEXT_MODEL_DTYPE_REALIZATION,
                            alpha=TEXT_PRIMARY_ALPHA,
                        )
                    else:
                        raise RuntimeError(f"unknown diagnostic condition {condition}")
                    stored = {
                        "version": TEXT_DIAGNOSTIC_VERSION,
                        "development_only": True,
                        "task_id": task.task_id,
                        "task": task.to_dict(),
                        "band": list(band),
                        "band_key": band_name,
                        "condition": condition,
                        "result": result,
                    }
                    STORE.save("intervention", key, stored)
                    if work == "computed":
                        computed += 1
                else:
                    reused += 1
                    work = "reused"
                diagnostic_records.append(stored)
                done = computed + reused + derived
                if done == 1 or done % 25 == 0 or done == total:
                    print(
                        f"diagnostic {done}/{total} {work} "
                        f"task={task.task_id} band={band_name} "
                        f"condition={condition} generated="
                        f"{stored['result']['answer_match']}"
                    )

    TEXT_DIAGNOSTIC_REPORT = text_swap_diagnostic_report(
        diagnostic_records,
        clean_rows=diagnostic_clean_rows,
        layers=ACTIVE_TEXT_LAYERS,
        bands=TEXT_DIAGNOSTIC_BANDS,
        task_set=TEXT_TASKS,
    )
    STORE.save("metric", "text_swap_diagnostic_report", TEXT_DIAGNOSTIC_REPORT)
    diagnostic_path = RUN_DIR / "text_swap_diagnostic_report.json"
    diagnostic_path.write_text(
        json.dumps(TEXT_DIAGNOSTIC_REPORT, indent=2, default=str)
    )
    print()
    print("=" * 78)
    print("TEXT SWAP DIAGNOSTIC", TEXT_DIAGNOSTIC_REPORT["verdict"])
    print("=" * 78)
    for row in TEXT_DIAGNOSTIC_REPORT["bands"]:
        print(
            f"  {row['band']} exact={row['exact_successes']}/{len(TEXT_TASKS)} "
            f"implicit={row['implicit_two_hop_success']} "
            f"flexible={row['flexible_function_success_rate']:.3f} "
            f"direct={row['direct_answer_positive_control_rate']:.3f} "
            f"controls={row['matched_controls_pass']} "
            f"coordinate_audit={row['coordinate_audits_pass']} "
            f"all_integrity={row['all_condition_integrity_pass']} "
            f"eligible={row['eligible_for_fresh_confirmation']}"
        )
    print("selected", TEXT_DIAGNOSTIC_REPORT["selected_band_for_fresh_confirmation"])
    print("units", {"computed": computed, "derived": derived, "reused": reused})
    print("report", diagnostic_path)
    print("checksum", TEXT_DIAGNOSTIC_REPORT["report_checksum"])
    print("resume", STORE.status_report())
elif RUN_STAGE1B_TEXT_DIAGNOSTIC:
    print(
        "skipped: Stage 6b needs real mode, model confirmation, its budget "
        "gate, and either alpha=1 or an explicit TEXT_PRIMARY_ALPHA_OVERRIDE"
    )
'''
)

markdown("## 7. Stage 2 — development-only multimodal source loading (no interventions)")
code(
    r'''
LOCALIZATION = STORE.load("metric", "loading_localization")
PAIR_SELECTION = STORE.load("metric", "loading_pair_selection")
MULTIMODAL_INSTRUMENT_SELECTION = STORE.load(
    "metric", "multimodal_instrument_selection"
)
TEXT_CAUSAL_GATE_MET = bool(
    TEXT_VERDICT is not None
    and TEXT_VERDICT.get("verdict") == "TEXT_PAPER_REPLICATION_GO"
) or bool(
    TEXT_DIAGNOSTIC_REPORT is not None
    and TEXT_DIAGNOSTIC_REPORT.get("verdict")
    == "TEXT_DIAGNOSTIC_ALPHA1_CANDIDATE_FOUND"
)
if (
    REAL_MODE
    and RUN_STAGE2_MULTIMODAL_LOADING_DEVELOPMENT
    and CONFIRM_MODEL_LOAD
    and CONFIRM_DEVELOPMENT_BUDGET
    and TEXT_CAUSAL_GATE_MET
):
    from jlens.mmpilot.coordinate_swap import resolve_concept_token
    from jlens.mmpilot.media_io import RetryJournal, drive_media_loaders
    from jlens.mmpilot.multimodal_lens import selected_lens_vector
    from jlens.mmpilot.store import safe_key
    from jlens.mmpilot.workspace_replication import (
        capture_source_loading, freeze_loading_localization,
        select_pair_from_loading, summarize_loading,
    )
    MEDIA = drive_media_loaders(journal=RetryJournal())
    names = sorted({name for pair in CANDIDATE_PAIRS for name in pair} | set(CONTROL_CONCEPTS))
    TOKENS = {name: resolve_concept_token(BACKEND.encode_candidate, name) for name in names}
    _pooled_candidates = {}
    if set(LAYERS).issubset(MATCHED_LENSES["pooled"].jacobians):
        _pooled_candidates["matched_pooled_j"] = MATCHED_LENSES["pooled"]
    if "pooled" in R_LENSES:
        _pooled_candidates["matched_pooled_r"] = R_LENSES["pooled"]
    if not _pooled_candidates:
        raise RuntimeError(
            "no pooled lens covers the configured study layer window"
        )
    MULTIMODAL_VECTORS = {
        instrument: {
            layer: {
                name: selected_lens_vector(
                    lens, BACKEND.unembedding_weight(),
                    layer=layer, token_id=token.token_id,
                ) for name, token in TOKENS.items()
            } for layer in LAYERS
        }
        for instrument, lens in _pooled_candidates.items()
    }
    def build_inputs(group, modality):
        question = "Identify the animal from the evidence internally, then answer: How many legs does that animal typically have? Answer with one digit:\nAnswer:"
        if modality == "text":
            return BACKEND.build_inputs(prompt=f"Caption: {group['caption']}\n{question}", modality="text")
        if modality == "image":
            return BACKEND.build_inputs(prompt=question, modality="image", image=MEDIA["load_image"](group["image_path"]), media_path=group["image_path"])
        waveform, rate = MEDIA["load_audio"](group["audio_path"])
        return BACKEND.build_inputs(prompt=question, modality="spoken_audio", audio=waveform, sampling_rate=rate, media_path=group["audio_path"])
    loading_by_instrument = {name: [] for name in MULTIMODAL_VECTORS}
    by_concept = {}
    for group in DEV_GROUPS:
        by_concept.setdefault(str(group.get("concept")), []).append(group)
    for instrument, vectors in sorted(MULTIMODAL_VECTORS.items()):
      for source, target in CANDIDATE_PAIRS:
        for group in by_concept.get(source, []):
            for modality in ("text", "image", "spoken_audio"):
                key = safe_key(
                    "loading", instrument, source, target,
                    group["group_id"], modality,
                )
                stored = STORE.load("activation", key)
                if stored is None:
                    inputs = build_inputs(group, modality)
                    stored = {
                        "source": source, "target": target,
                        "group_id": group["group_id"], "image_id": group["image_id"],
                        "modality": modality,
                        "rows": capture_source_loading(
                            BACKEND, inputs, vectors_by_layer=vectors,
                            source=source, target=target, unrelated=CONTROL_CONCEPTS,
                            sample_id=f"{group['group_id']}:{modality}", modality=modality,
                        ),
                    }
                    STORE.save("activation", key, stored)
                    work = "computed"
                else:
                    work = "reused"
                loading_by_instrument[instrument].extend(stored["rows"])
                print("loading", instrument, source, target, group["group_id"], modality, work)
    _instrument_rows = []
    for instrument, loading_rows in sorted(loading_by_instrument.items()):
        pair_selection = select_pair_from_loading(
            loading_rows, candidate_pairs=CANDIDATE_PAIRS,
            required_modalities=("text", "image", "spoken_audio"),
        )
        selected_source, selected_target = pair_selection["selected_pair"]
        selected_rows = [
            row for row in loading_rows
            if row["source"] == selected_source
            and row["target"] == selected_target
        ]
        localization = freeze_loading_localization(
            selected_rows,
            required_modalities=("text", "image", "spoken_audio"),
            candidate_layers=LAYERS,
            min_source_advantage=MIN_SOURCE_ADVANTAGE,
            evidence_position_margin=EVIDENCE_POSITION_MARGIN,
        )
        _instrument_rows.append({
            "instrument": instrument,
            "loading_report": summarize_loading(loading_rows),
            "pair_selection": pair_selection,
            "localization": localization,
            "band_length": len(localization["selected_band"]),
            "pair_score": next(
                row["weakest_modality_score"]
                for row in pair_selection["ranking"]
                if [row["source"], row["target"]]
                == pair_selection["selected_pair"]
            ),
            "causal_result_consulted": False,
        })
    _instrument_rows.sort(
        key=lambda row: (
            -row["band_length"], -row["pair_score"], row["instrument"]
        )
    )
    _selected = next(
        (row for row in _instrument_rows if row["band_length"] > 0), None
    )
    MULTIMODAL_INSTRUMENT_SELECTION = {
        "version": "mmpilot.loading_first_multimodal_instrument.v1",
        "ranking": _instrument_rows,
        "selected_instrument": _selected["instrument"] if _selected else None,
        "verdict": "MULTIMODAL_INSTRUMENT_GO" if _selected else "MULTIMODAL_INSTRUMENT_NO_GO",
        "causal_result_consulted": False,
    }
    MULTIMODAL_INSTRUMENT_SELECTION["selection_digest"] = payload_checksum(
        MULTIMODAL_INSTRUMENT_SELECTION
    )
    LOADING_REPORT = _selected["loading_report"] if _selected else None
    PAIR_SELECTION = _selected["pair_selection"] if _selected else None
    LOCALIZATION = _selected["localization"] if _selected else None
    STORE.save("metric", "loading_report", LOADING_REPORT)
    STORE.save("metric", "loading_pair_selection", PAIR_SELECTION)
    STORE.save("metric", "loading_localization", LOCALIZATION)
    STORE.save(
        "metric", "multimodal_instrument_selection",
        MULTIMODAL_INSTRUMENT_SELECTION,
    )
    print(json.dumps(MULTIMODAL_INSTRUMENT_SELECTION, indent=2))
    print(json.dumps(PAIR_SELECTION, indent=2))
    print(json.dumps(LOCALIZATION, indent=2))
elif RUN_STAGE2_MULTIMODAL_LOADING_DEVELOPMENT:
    print(
        "Stage 2 did not spend: it requires a passing predeclared text swap "
        "or audited alpha=1 diagnostic candidate, plus the model and budget gates."
    )
'''
)

markdown("## 8. Stage 3 — freeze the causal design before confirmation media are opened")
code(
    r'''
CONFIRMATION_DESIGN = STORE.load("metric", "confirmation_design")
if RUN_STAGE3_FREEZE_DESIGN:
    from jlens.mmpilot.workspace_replication import freeze_confirmation_design
    if TEXT_VERDICT is None:
        TEXT_VERDICT = STORE.load("metric", "text_replication_verdict")
    if LOCALIZATION is None:
        LOCALIZATION = STORE.load("metric", "loading_localization")
    if PAIR_SELECTION is None:
        PAIR_SELECTION = STORE.load("metric", "loading_pair_selection")
    TEXT_DIAGNOSTIC_REPORT = TEXT_DIAGNOSTIC_REPORT or STORE.load(
        "metric", "text_swap_diagnostic_report"
    )
    if (
        not TEXT_CAUSAL_GATE_MET
        or LOCALIZATION is None
        or PAIR_SELECTION is None
    ):
        print("Stage 3 did not freeze: its text/loading gates are not met.")
    else:
        _design_kwargs = {
            "localization": LOCALIZATION,
            "pair": PAIR_SELECTION["selected_pair"],
            "alpha": MULTIMODAL_PRIMARY_ALPHA,
            "sensitivity_alpha": MULTIMODAL_SENSITIVITY_ALPHA,
            "prompt_protocol": PROMPT_PROTOCOL,
            "development_population_digest": POPULATION_PLAN["plan_digest"],
        }
        if (
            TEXT_DIAGNOSTIC_REPORT is not None
            and TEXT_DIAGNOSTIC_REPORT.get("verdict")
            == "TEXT_DIAGNOSTIC_ALPHA1_CANDIDATE_FOUND"
        ):
            _design_kwargs["text_diagnostic"] = TEXT_DIAGNOSTIC_REPORT
        else:
            _design_kwargs["text_verdict"] = TEXT_VERDICT
        CONFIRMATION_DESIGN = freeze_confirmation_design(**_design_kwargs)
        CONFIRMATION_DESIGN["forbidden_development_image_ids"] = sorted(DEV_IMAGE_IDS)
        CONFIRMATION_DESIGN["forbidden_development_group_ids"] = sorted(DEV_GROUP_IDS)
        CONFIRMATION_DESIGN["forbidden_prior_image_ids"] = sorted(PRIOR_EXCLUDED_IMAGES)
        CONFIRMATION_DESIGN["multimodal_instrument"] = (
            MULTIMODAL_INSTRUMENT_SELECTION["selected_instrument"]
        )
        CONFIRMATION_DESIGN["multimodal_instrument_selection_digest"] = (
            MULTIMODAL_INSTRUMENT_SELECTION["selection_digest"]
        )
        CONFIRMATION_DESIGN["design_digest"] = payload_checksum(
            {k: v for k, v in CONFIRMATION_DESIGN.items() if k != "design_digest"}
        )
        STORE.save("metric", "confirmation_design", CONFIRMATION_DESIGN)
        (RUN_DIR / "frozen_confirmation_design.json").write_text(
            json.dumps(CONFIRMATION_DESIGN, indent=2)
        )
        print(json.dumps(CONFIRMATION_DESIGN, indent=2))
'''
)

markdown("## 9. Stage 4 — untouched multimodal confirmation with unrestricted output")
code(
    r'''
CONFIRMATION_REPORT = STORE.load("metric", "fresh_confirmation_report")
if REAL_MODE and RUN_STAGE4_FRESH_CONFIRMATION and CONFIRM_MODEL_LOAD and CONFIRM_CONFIRMATION_BUDGET:
    from jlens.mmpilot.coordinate_swap import random_two_direction_basis, resolve_concept_token
    from jlens.mmpilot.media_io import RetryJournal, drive_media_loaders
    from jlens.mmpilot.multimodal_lens import (
        build_swap_bases_for_lens, selected_lens_vector,
    )
    from jlens.mmpilot.store import safe_key
    from jlens.mmpilot.workspace_replication import (
        TEXT_MAX_NEW_TOKENS, TEXT_MODEL_DTYPE_REALIZATION,
        assert_fresh_population, semantic_answer_concept,
        unrestricted_greedy_completion, unrestricted_greedy_swap_trial,
        unrestricted_greedy_direct_answer_trial,
    )
    if CONFIRMATION_DESIGN is None:
        CONFIRMATION_DESIGN = STORE.load("metric", "confirmation_design")
    if CONFIRMATION_DESIGN is None:
        print("Stage 4 did not spend: Stage 3 produced no frozen design.")
    else:
        if MULTIMODAL_INSTRUMENT_SELECTION is None:
            MULTIMODAL_INSTRUMENT_SELECTION = STORE.load(
                "metric", "multimodal_instrument_selection"
            )
        _selected_multimodal = MULTIMODAL_INSTRUMENT_SELECTION[
            "selected_instrument"
        ]
        if _selected_multimodal == "matched_pooled_j":
            SELECTED_MULTIMODAL_LENS = MATCHED_LENSES["pooled"]
        elif _selected_multimodal == "matched_pooled_r" and "pooled" in R_LENSES:
            SELECTED_MULTIMODAL_LENS = R_LENSES["pooled"]
        else:
            raise RuntimeError(
                "the frozen multimodal instrument is unavailable in this session"
            )
        source, target = CONFIRMATION_DESIGN["pair"]
        directions = ((source, target), (target, source))
        freshness = assert_fresh_population(
            CONFIRM_GROUPS,
            forbidden_image_ids=sorted(set(
                CONFIRMATION_DESIGN["forbidden_development_image_ids"]
            ) | set(CONFIRMATION_DESIGN["forbidden_prior_image_ids"])),
            forbidden_group_ids=CONFIRMATION_DESIGN["forbidden_development_group_ids"],
        )
        MEDIA = drive_media_loaders(journal=RetryJournal())
        answers = {"bird": "2", "cat": "4", "zebra": "4", "giraffe": "4"}
        if answers[source] == answers[target]:
            raise RuntimeError("selected pair has no downstream leg-count contrast")
        token_names = {
            source, target, *CONTROL_CONCEPTS,
            semantic_answer_concept(answers[source]),
            semantic_answer_concept(answers[target]),
        }
        TOKENS = {
            name: resolve_concept_token(BACKEND.encode_candidate, name)
            for name in token_names
        }
        band = CONFIRMATION_DESIGN["layer_band"]
        direction_assets = {}
        for direction_index, (direction_source, direction_target) in enumerate(directions):
            exact_bases = build_swap_bases_for_lens(
                SELECTED_MULTIMODAL_LENS, BACKEND.unembedding_weight(), layers=band,
                source=TOKENS[direction_source], target=TOKENS[direction_target],
            )
            direction_assets[(direction_source, direction_target)] = {
                "exact": exact_bases,
                "random": {
                    layer: random_two_direction_basis(
                        basis, seed=20260821 + 100 * direction_index + layer
                    ) for layer, basis in exact_bases.items()
                },
                "unrelated": build_swap_bases_for_lens(
                    SELECTED_MULTIMODAL_LENS, BACKEND.unembedding_weight(),
                    layers=band, source=TOKENS[CONTROL_CONCEPTS[0]],
                    target=TOKENS[CONTROL_CONCEPTS[1]],
                ),
                "answer_vectors": {
                    layer: selected_lens_vector(
                        SELECTED_MULTIMODAL_LENS, BACKEND.unembedding_weight(),
                        layer=layer,
                        token_id=TOKENS[
                            semantic_answer_concept(answers[direction_target])
                        ].token_id,
                    ) for layer in band
                },
            }
        question_identity = "Identify the animal from the evidence internally. Answer with only the animal name:\nAnswer:"
        question_property = "Identify the animal from the evidence internally, then answer: How many legs does that animal typically have? Answer with one digit:\nAnswer:"
        def build_inputs(group, modality, kind):
            question = question_identity if kind == "identity" else question_property
            if modality == "text":
                return BACKEND.build_inputs(prompt=f"Caption: {group['caption']}\n{question}", modality="text")
            if modality == "image":
                return BACKEND.build_inputs(prompt=question, modality="image", image=MEDIA["load_image"](group["image_path"]), media_path=group["image_path"])
            waveform, rate = MEDIA["load_audio"](group["audio_path"])
            return BACKEND.build_inputs(prompt=question, modality="spoken_audio", audio=waveform, sampling_rate=rate, media_path=group["audio_path"])
        rows = []
        _primary_alpha = float(CONFIRMATION_DESIGN["primary_alpha"])
        _sensitivity_alpha = CONFIRMATION_DESIGN.get("sensitivity_alpha")
        _primary_tag = (
            "alpha1" if _primary_alpha == 1.0
            else "alpha2" if _primary_alpha == 2.0
            else "primary"
        )
        _sensitivity_tag = (
            "alpha075" if _sensitivity_alpha == 0.75
            else "alpha1" if _sensitivity_alpha == 1.0
            else "sensitivity"
        )
        _exact_key = f"exact_{_primary_tag}"
        _random_key = f"random_{_primary_tag}"
        _unrelated_key = f"unrelated_{_primary_tag}"
        _sensitivity_key = f"exact_{_sensitivity_tag}"
        condition_specs = (
            (_exact_key, "exact", _primary_alpha),
            ("zero", "exact", 0.0),
            (_random_key, "random", _primary_alpha),
            (_unrelated_key, "unrelated", _primary_alpha),
            (_sensitivity_key, "exact", _sensitivity_alpha),
            ("direct_answer_norm_matched", "direct", None),
        )
        completed = computed = reused = 0
        for direction_source, direction_target in directions:
            assets = direction_assets[(direction_source, direction_target)]
            selected_groups = [
                row for row in CONFIRM_GROUPS
                if str(row.get("concept")) == direction_source
            ][:CONFIRMATION_IMAGES_PER_SOURCE]
            for group in selected_groups:
                for modality in ("text", "image", "spoken_audio"):
                    for kind in ("identity", "property"):
                        inputs = build_inputs(group, modality, kind)
                        source_answer = direction_source if kind == "identity" else answers[direction_source]
                        target_answer = direction_target if kind == "identity" else answers[direction_target]
                        clean_key = safe_key(
                            "fresh-confirm-clean", direction_source,
                            direction_target, group["group_id"], modality, kind,
                        )
                        clean_unit = STORE.load("capability", clean_key)
                        if clean_unit is None:
                            clean_unit = unrestricted_greedy_completion(
                                BACKEND, inputs, answer=source_answer,
                                max_new_tokens=TEXT_MAX_NEW_TOKENS,
                            )
                            STORE.save("capability", clean_key, clean_unit)
                        conditions = {}
                        for name, basis_name, alpha in condition_specs:
                            key = safe_key(
                                "fresh-confirm", direction_source,
                                direction_target, group["group_id"], modality,
                                kind, name,
                            )
                            trial = STORE.load("intervention", key)
                            if trial is None:
                                if basis_name == "direct":
                                    trial = unrestricted_greedy_direct_answer_trial(
                                        BACKEND, inputs, bases=assets["exact"],
                                        answer_vectors=assets["answer_vectors"],
                                        answer=target_answer,
                                        position_rule=CONFIRMATION_DESIGN[
                                            "position_rule_by_modality"
                                        ][modality],
                                        realization_policy=TEXT_MODEL_DTYPE_REALIZATION,
                                    )
                                else:
                                    trial = unrestricted_greedy_swap_trial(
                                        BACKEND, inputs, bases=assets[basis_name],
                                        alpha=alpha, answer=target_answer,
                                        max_new_tokens=TEXT_MAX_NEW_TOKENS,
                                        position_rule=CONFIRMATION_DESIGN[
                                            "position_rule_by_modality"
                                        ][modality],
                                        realization_policy=TEXT_MODEL_DTYPE_REALIZATION,
                                    )
                                trial = {
                                    **trial, "patched_surface": trial["generated_text"],
                                    "success": bool(trial["answer_match"]),
                                }
                                STORE.save("intervention", key, trial)
                                computed += 1
                                work = "computed"
                            else:
                                reused += 1
                                work = "reused"
                            conditions[name] = trial
                            completed += 1
                            if completed == 1 or completed % 25 == 0:
                                print("confirmation", completed, work, direction_source, direction_target, modality, kind, name)
                        rows.append({
                            "group_id": group["group_id"], "image_id": group["image_id"],
                            "source": direction_source, "target": direction_target,
                            "direction": f"{direction_source}->{direction_target}",
                            "modality": modality, "prompt_kind": kind,
                            "source_answer": source_answer, "target_answer": target_answer,
                            "output_endpoint": "unrestricted_greedy_complete_answer",
                            "clean_surface": clean_unit["generated_text"],
                            "clean_correct": bool(clean_unit["answer_match"]),
                            "clean": clean_unit, "conditions": conditions,
                        })
        def condition_integrity(trial):
            diagnostics = dict(trial.get("intervention_diagnostics") or {})
            return bool(
                diagnostics.get("all_hooks_fired")
                and diagnostics.get("all_finite")
                and diagnostics.get("all_model_dtype_realizations_converged", True)
            )
        cells = []
        for direction_source, direction_target in directions:
            direction_name = f"{direction_source}->{direction_target}"
            for kind in ("identity", "property"):
                for modality in ("text", "image", "spoken_audio"):
                    cell = [
                        row for row in rows
                        if row["direction"] == direction_name
                        and row["prompt_kind"] == kind
                        and row["modality"] == modality
                    ]
                    _cell_payload = {
                        "direction": direction_name, "prompt_kind": kind,
                        "modality": modality, "n": len(cell),
                        "clean_capability": sum(row["clean_correct"] for row in cell) / len(cell),
                        "primary_alpha": _primary_alpha,
                        "primary_success": sum(row["conditions"][_exact_key]["success"] for row in cell) / len(cell),
                        "sensitivity_alpha": _sensitivity_alpha,
                        "sensitivity_success": sum(row["conditions"][_sensitivity_key]["success"] for row in cell) / len(cell),
                        "zero_success": sum(row["conditions"]["zero"]["success"] for row in cell) / len(cell),
                        "random_primary_success": sum(row["conditions"][_random_key]["success"] for row in cell) / len(cell),
                        "unrelated_primary_success": sum(row["conditions"][_unrelated_key]["success"] for row in cell) / len(cell),
                        "direct_answer_success": sum(row["conditions"]["direct_answer_norm_matched"]["success"] for row in cell) / len(cell),
                        "all_condition_integrity": all(
                            condition_integrity(trial)
                            for row in cell for trial in row["conditions"].values()
                        ),
                    }
                    if _primary_alpha == 1.0:
                        _cell_payload["alpha1_success"] = _cell_payload[
                            "primary_success"
                        ]
                        _cell_payload["random_alpha1_success"] = _cell_payload[
                            "random_primary_success"
                        ]
                        _cell_payload["unrelated_alpha1_success"] = _cell_payload[
                            "unrelated_primary_success"
                        ]
                    if _sensitivity_alpha == 0.75:
                        _cell_payload["alpha075_success"] = _cell_payload[
                            "sensitivity_success"
                        ]
                    cells.append(_cell_payload)
        from jlens.mmpilot.workspace_replication import holm_adjust, paired_binary_superiority
        paired, raw_p = {}, {}
        for direction_source, direction_target in directions:
            direction_name = f"{direction_source}->{direction_target}"
            for kind in ("identity", "property"):
                kind_rows = [row for row in rows if row["direction"] == direction_name and row["prompt_kind"] == kind]
                for control in ("zero", _random_key, _unrelated_key):
                    key = f"{direction_name}_{kind}_exact_vs_{control}"
                    paired[key] = paired_binary_superiority(
                        [row["conditions"][_exact_key]["success"] for row in kind_rows],
                        [row["conditions"][control]["success"] for row in kind_rows],
                    )
                    raw_p[key] = paired[key]["one_sided_exact_p"]
        adjusted = holm_adjust(raw_p)
        for key in paired:
            paired[key]["holm_p"] = adjusted[key]
            paired[key]["passed"] = adjusted[key] <= CONFIRMATION_FAMILYWISE_ALPHA
        paired_text, raw_text_p = {}, {}
        for direction_source, direction_target in directions:
            direction_name = f"{direction_source}->{direction_target}"
            for kind in ("identity", "property"):
                text_rows = [
                    row for row in rows
                    if row["direction"] == direction_name
                    and row["prompt_kind"] == kind
                    and row["modality"] == "text"
                ]
                for control in ("zero", _random_key, _unrelated_key):
                    key = f"{direction_name}_{kind}_text_exact_vs_{control}"
                    paired_text[key] = paired_binary_superiority(
                        [row["conditions"][_exact_key]["success"] for row in text_rows],
                        [row["conditions"][control]["success"] for row in text_rows],
                    )
                    raw_text_p[key] = paired_text[key]["one_sided_exact_p"]
        adjusted_text = holm_adjust(raw_text_p)
        for key in paired_text:
            paired_text[key]["holm_p"] = adjusted_text[key]
            paired_text[key]["passed"] = (
                adjusted_text[key] <= CONFIRMATION_FAMILYWISE_ALPHA
            )
        integrity = all(cell["all_condition_integrity"] for cell in cells)
        capability = all(cell["clean_capability"] >= 0.75 for cell in cells)
        positive_control = all(cell["direct_answer_success"] >= 0.50 for cell in cells)
        primary = all(
            cell["primary_success"] >= MIN_CONFIRMATION_SUCCESS_RATE
            and cell["primary_success"] > max(
                cell["zero_success"], cell["random_primary_success"],
                cell["unrelated_primary_success"],
            ) for cell in cells
        ) and all(row["passed"] for row in paired.values())
        downstream = all(
            cell["primary_success"] >= MIN_CONFIRMATION_SUCCESS_RATE
            for cell in cells if cell["prompt_kind"] == "property"
        )
        text_cells = [cell for cell in cells if cell["modality"] == "text"]
        text_primary = all(
            cell["primary_success"] >= MIN_CONFIRMATION_SUCCESS_RATE
            and cell["primary_success"] > max(
                cell["zero_success"], cell["random_primary_success"],
                cell["unrelated_primary_success"],
            )
            for cell in text_cells
        ) and all(row["passed"] for row in paired_text.values())
        text_verdict = (
            "FRESH_TEXT_CAUSAL_RECOMPUTATION_GO"
            if integrity and capability and positive_control and text_primary
            else "FRESH_TEXT_CAUSAL_RECOMPUTATION_NO_GO"
        )
        verdict = (
            "FRESH_MULTIMODAL_ENGINEERING_NO_GO" if not integrity
            else "FRESH_MULTIMODAL_CAPABILITY_NO_GO" if not capability
            else "FRESH_MULTIMODAL_POSITIVE_CONTROL_NO_GO" if not positive_control
            else "FRESH_MULTIMODAL_DOWNSTREAM_RECOMPUTATION_GO"
            if primary and downstream
            else "FRESH_MULTIMODAL_CONFIRMATION_NO_GO"
        )
        CONFIRMATION_REPORT = {
            "version": "mmpilot.fresh_multimodal_confirmation.v2",
            "verdict": verdict, "text_only_verdict": text_verdict,
            "design": CONFIRMATION_DESIGN, "freshness": freshness,
            "cells": cells, "paired_tests": paired,
            "paired_text_tests": paired_text,
            "familywise_alpha": CONFIRMATION_FAMILYWISE_ALPHA,
            "minimum_cell_success_rate": MIN_CONFIRMATION_SUCCESS_RATE,
            "both_directions_tested": True,
            "engineering_integrity_passed": integrity,
            "clean_capability_passed": capability,
            "positive_control_passed": positive_control,
            "rows": rows, "teacher_forcing_used": False,
            "candidate_list_supplied": False,
            "primary_alpha": _primary_alpha,
            "primary_alpha_is_paper_double_strength": _primary_alpha == 2.0,
            "alpha1_is_primary": _primary_alpha == 1.0,
            "sensitivity_alpha": _sensitivity_alpha,
            "atomic_resume_unit": "one two-token condition",
            "maximum_completed_forward_passes_lost_on_disconnect": 0,
        }
        CONFIRMATION_REPORT["report_checksum"] = payload_checksum(CONFIRMATION_REPORT)
        STORE.save("metric", "fresh_confirmation_report", CONFIRMATION_REPORT)
        (RUN_DIR / "fresh_multimodal_confirmation_report.json").write_text(json.dumps(CONFIRMATION_REPORT, indent=2, default=str))
        print(json.dumps({k: v for k, v in CONFIRMATION_REPORT.items() if k != "rows"}, indent=2))
elif RUN_STAGE4_FRESH_CONFIRMATION:
    print("Stage 4 did not spend: real mode/model/budget gate is not met.")
'''
)

markdown("## 10. Stage 5 — write the integrated report and stop")
code(
    r'''
if RUN_STAGE5_WRITE_REPORT:
    TEXT_VERDICT = TEXT_VERDICT or STORE.load("metric", "text_replication_verdict")
    TEXT_DIAGNOSTIC_REPORT = TEXT_DIAGNOSTIC_REPORT or STORE.load(
        "metric", "text_swap_diagnostic_report"
    )
    PAIR_SELECTION = PAIR_SELECTION or STORE.load("metric", "loading_pair_selection")
    LOCALIZATION = LOCALIZATION or STORE.load("metric", "loading_localization")
    CONFIRMATION_DESIGN = CONFIRMATION_DESIGN or STORE.load("metric", "confirmation_design")
    CONFIRMATION_REPORT = CONFIRMATION_REPORT or STORE.load("metric", "fresh_confirmation_report")
    LOADING_FIRST_SELECTION = LOADING_FIRST_SELECTION or STORE.load(
        "metric", "loading_first_selection"
    )
    MULTIMODAL_INSTRUMENT_SELECTION = (
        MULTIMODAL_INSTRUMENT_SELECTION
        or STORE.load("metric", "multimodal_instrument_selection")
    )
    FINAL = {
        "schema": "jlens.mmpilot.paper_first_workspace_study.v3",
        "scientific_config": SCIENTIFIC_CONFIG,
        "population_plan": POPULATION_PLAN,
        "text_replication": TEXT_VERDICT,
        "text_swap_diagnostic": TEXT_DIAGNOSTIC_REPORT,
        "text_loading_first_instrument": LOADING_FIRST_SELECTION,
        "multimodal_loading_first_instrument": MULTIMODAL_INSTRUMENT_SELECTION,
        "r_lens_inventory": (
            json.loads((RUN_DIR / "r_lens_inventory.json").read_text())
            if (RUN_DIR / "r_lens_inventory.json").is_file()
            else None
        ),
        "loading_pair_selection": PAIR_SELECTION,
        "loading_localization": LOCALIZATION,
        "frozen_confirmation_design": CONFIRMATION_DESIGN,
        "fresh_confirmation": CONFIRMATION_REPORT,
        "claims": {
            "text_only_causal_recomputation_supported": bool(
                CONFIRMATION_REPORT
                and CONFIRMATION_REPORT.get("text_only_verdict")
                == "FRESH_TEXT_CAUSAL_RECOMPUTATION_GO"
            ),
            "strong_multimodal_causal_recomputation_supported": bool(
                CONFIRMATION_REPORT
                and CONFIRMATION_REPORT.get("verdict")
                == "FRESH_MULTIMODAL_DOWNSTREAM_RECOMPUTATION_GO"
            ),
        },
        "claim_boundary": (
            "A multimodal downstream-recomputation claim is licensed only by "
            "FRESH_MULTIMODAL_DOWNSTREAM_RECOMPUTATION_GO. Development loading "
            "and alpha=.75 sensitivity cannot substitute for it."
        ),
    }
    FINAL["report_checksum"] = payload_checksum(FINAL)
    path = RUN_DIR / "paper_first_workspace_study_report.json"
    path.write_text(json.dumps(FINAL, indent=2, default=str))
    print("report", path)
    print("checksum", FINAL["report_checksum"])
    print("resume", STORE.status_report())
'''
)


def _cell(cell_type: str, source: str) -> dict:
    body = [line + "\n" for line in source.splitlines()]
    if body:
        body[-1] = body[-1].rstrip("\n")
    result = {"cell_type": cell_type, "metadata": {}, "source": body}
    if cell_type == "code":
        result.update({"execution_count": None, "outputs": []})
    return result


notebook = {
    "cells": [_cell(kind, source) for kind, source in CELLS],
    "metadata": {
        "accelerator": "GPU",
        "colab": {"name": TARGET.name, "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
TARGET.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(TARGET)
