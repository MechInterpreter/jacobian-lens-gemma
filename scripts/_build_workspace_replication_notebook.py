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
# Paper-first J-lens replication, source loading, and fresh multimodal confirmation

This is the shortest fail-closed path from the completed work to an interpretable
result. It uses **one causal implementation only**: Anthropic's two-coordinate
exchange in `jlens.mmpilot.coordinate_swap`.

The order is mandatory:

1. **Text replication.** Reproduce the paper's text-only implicit two-hop task
   (`spider → ant`, expected downstream answer `8 → 6`) and its France/China
   flexible-function family with the validated text lens and exact `alpha=1`.
   Gemma tokenizes the paper's digit outputs as whitespace + digit, so success
   is the complete answer from unrestricted two-token greedy generation—not a
   one-token prefix, candidate score, or teacher-forced likelihood.
2. **Clean source loading.** On development media, measure whether the source
   concept is actually visible through the matched pooled lens at each layer and
   prompt position. Causal outcomes do not exist yet.
3. **Freeze the design.** Choose the concept pair, contiguous layer band, and
   position rule from clean loading only. `alpha=1` remains primary; `alpha=.75`
   is a labelled interpolation sensitivity.
4. **Fresh confirmation.** Open new photographs/recordings, prove zero overlap,
   and test unrestricted identity and downstream property outputs against random
   and unrelated controls.

If the text replication or clean-loading gate fails, later causal spending is
blocked. No threshold, pair, layer, position, or alpha may be changed after its
gate has seen results.
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
RUN_STAGE1_TEXT_REPLICATION = False
RUN_STAGE2_MULTIMODAL_LOADING_DEVELOPMENT = False
RUN_STAGE3_FREEZE_DESIGN = False
RUN_STAGE4_FRESH_CONFIRMATION = False
RUN_STAGE5_WRITE_REPORT = False

CONFIRM_MODEL_LOAD = False
CONFIRM_DEVELOPMENT_BUDGET = False
CONFIRM_CONFIRMATION_BUDGET = False

MODEL_REPO_ID = "google/gemma-4-E4B-it"
MODEL_REVISION = "fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd"
EXPECT_N_LAYERS, EXPECT_D_MODEL, EXPECT_VOCAB = 42, 2560, 262144
AUDIO_PROTOCOL_FINGERPRINT = (
    "sha256:9ad8bcc9420a7983f6e3b75d5d7080c0e2fcf0a94a76431917fcde73ba777920"
)
LAYERS = tuple(range(33, 41))
TEXT_PRIMARY_ALPHA = 1.0
MULTIMODAL_PRIMARY_ALPHA = 1.0
MULTIMODAL_SENSITIVITY_ALPHA = 0.75
CANDIDATE_PAIRS = (("bird", "cat"), ("bird", "zebra"), ("bird", "giraffe"))
CONTROL_CONCEPTS = ("microwave", "toilet")
DEVELOPMENT_IMAGES_PER_SOURCE = 8
CONFIRMATION_IMAGES_PER_SOURCE = 8
MIN_SOURCE_ADVANTAGE = 0.0
EVIDENCE_POSITION_MARGIN = 0.0
MIN_CONFIRMATION_SUCCESS_RATE = 0.50
CONFIRMATION_FAMILYWISE_ALPHA = 0.05
DEVELOPMENT_SEED = "paper-first-loading-development-20260820-v1"
CONFIRMATION_SEED = "paper-first-fresh-confirmation-20260820-v1"
PROMPT_PROTOCOL = "mmpilot.implicit_animal_property_open_output.v1"

REAL_MODE = bool(RUN_REAL_WORKSPACE_REPLICATION)
MODEL_STAGE = any((
    RUN_STAGE1_TEXT_REPLICATION,
    RUN_STAGE2_MULTIMODAL_LOADING_DEVELOPMENT,
    RUN_STAGE4_FRESH_CONFIRMATION,
))
if MODEL_STAGE and not CONFIRM_MODEL_LOAD:
    print("MODEL STAGES BLOCKED: set CONFIRM_MODEL_LOAD after reading the budget")
if RUN_STAGE2_MULTIMODAL_LOADING_DEVELOPMENT and not CONFIRM_DEVELOPMENT_BUDGET:
    print("DEVELOPMENT BLOCKED: set CONFIRM_DEVELOPMENT_BUDGET")
if RUN_STAGE4_FRESH_CONFIRMATION and not CONFIRM_CONFIRMATION_BUDGET:
    print("CONFIRMATION BLOCKED: set CONFIRM_CONFIRMATION_BUDGET")
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
    EXPANDED_MANIFEST_CACHE = (
        RUNS_ROOT / "mml32_l32_followup_20260808T182717" / "expanded_manifest.json"
    )
else:
    RUNS_ROOT = Path(tempfile.gettempdir()) / "workspace_replication_mock"
    CORRECTED_RUN_DIR = MATCHED_LENS_RUN_DIR = EXPANDED_MANIFEST_CACHE = None

TEXT_TASKS = __import__(
    "jlens.mmpilot.workspace_replication", fromlist=["anthropic_text_tasks"]
).anthropic_text_tasks()
print("TEXT TASKS", len(TEXT_TASKS))
print("  unrestricted generation passes", len(TEXT_TASKS) * 2 * 4)
print("  clean source-loading passes", len(TEXT_TASKS))
print("  Stage-1 total passes", len(TEXT_TASKS) * 2 * 4 + len(TEXT_TASKS))
print("DEVELOPMENT UPPER BOUND")
print("  clean loading forwards", len(CANDIDATE_PAIRS) * DEVELOPMENT_IMAGES_PER_SOURCE * 3)
print("  no intervention forwards in Stage 2")
print("FRESH CONFIRMATION UPPER BOUND")
print("  clean generation forwards", CONFIRMATION_IMAGES_PER_SOURCE * 3 * 2 * 2)
print("  exact/random/unrelated x alpha1 plus alpha=.75 sensitivity",
      CONFIRMATION_IMAGES_PER_SOURCE * 3 * 2 * 4 * 2)
print("RESUME UNIT: one text task, loading sample, or causal condition JSON")

if REAL_MODE:
    missing = [path for path in (CORRECTED_RUN_DIR, MATCHED_LENS_RUN_DIR, EXPANDED_MANIFEST_CACHE) if not path.exists()]
    if missing:
        raise FileNotFoundError("configured artifact(s) missing:\n  " + "\n  ".join(map(str, missing)))
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
    PROTOCOL_VERSION, TEXT_MAX_NEW_TOKENS, TEXT_OUTPUT_ENDPOINT_VERSION,
    text_task_digest,
)

SCIENTIFIC_CONFIG = {
    "protocol": PROTOCOL_VERSION,
    "output_endpoint": TEXT_OUTPUT_ENDPOINT_VERSION,
    "max_new_tokens": TEXT_MAX_NEW_TOKENS,
    "model_repo_id": MODEL_REPO_ID, "model_revision": MODEL_REVISION,
    "audio_protocol_fingerprint": AUDIO_PROTOCOL_FINGERPRINT,
    "layers": list(LAYERS), "text_task_digest": text_task_digest(TEXT_TASKS),
    "candidate_pairs": [list(pair) for pair in CANDIDATE_PAIRS],
    "control_concepts": list(CONTROL_CONCEPTS),
    "primary_alpha": 1.0, "sensitivity_alpha": 0.75,
    "population_plan_digest": POPULATION_PLAN["plan_digest"],
    "min_source_advantage": MIN_SOURCE_ADVANTAGE,
    "evidence_position_margin": EVIDENCE_POSITION_MARGIN,
    "minimum_confirmation_success_rate": MIN_CONFIRMATION_SUCCESS_RATE,
    "confirmation_familywise_alpha": CONFIRMATION_FAMILYWISE_ALPHA,
    "commit": COMMIT,
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
    corrected_path, corrected = read_corrected_validation_report(
        CORRECTED_RUN_DIR, expected_model_repo_id=MODEL_REPO_ID,
        expected_model_revision=BUNDLE.model_revision,
    )
    CORRECTED_ARTIFACTS, _ = discover_corrected_band_lenses(
        CORRECTED_RUN_DIR, report=corrected, layers=LAYERS,
    )
    names = sorted({task.source for task in TEXT_TASKS} | {task.target for task in TEXT_TASKS} | {"zebra", "giraffe", "Japan", "Brazil"})
    from jlens.mmpilot.coordinate_swap import resolve_concept_token
    TEXT_CONCEPT_TOKENS = {name: resolve_concept_token(BACKEND.encode_candidate, name) for name in names}
    unembedding = BACKEND.unembedding_weight()
    rows = {name: unembedding[token.token_id].detach().float().cpu() for name, token in TEXT_CONCEPT_TOKENS.items()}
    loaded = {}
    for layer in LAYERS:
        source = CORRECTED_ARTIFACTS[layer]
        loaded.setdefault(source.lens_path, JacobianLens.load(source.lens_path))
        jacobian = loaded[source.lens_path].jacobians[source.layer_key_in_file].detach().float().cpu()
        TEXT_TOKEN_VECTORS[layer] = {name: row @ jacobian for name, row in rows.items()}
    del loaded, rows, unembedding
    for arm in ("text", "image", "spoken_audio", "pooled"):
        MATCHED_LENSES[arm] = JacobianLens.load(str(MATCHED_LENS_RUN_DIR / "lenses" / f"lens.{arm}.pt"))
    print("validated text layers", sorted(TEXT_TOKEN_VECTORS))
    print("matched lenses", sorted(MATCHED_LENSES))
elif MODEL_STAGE:
    print("skipped: model load is not confirmed")
'''
)

markdown("## 6. Stage 1 — paper-task text replication and source-loading audit")
code(
    r'''
TEXT_VERDICT = STORE.load("metric", "text_replication_verdict")
if REAL_MODE and RUN_STAGE1_TEXT_REPLICATION and CONFIRM_MODEL_LOAD:
    from jlens.mmpilot.coordinate_swap import (
        build_swap_basis_from_vectors, random_two_direction_basis,
    )
    from jlens.mmpilot.store import safe_key
    from jlens.mmpilot.workspace_replication import (
        TEXT_MAX_NEW_TOKENS, capture_source_loading, text_replication_verdict,
        unrestricted_greedy_completion, unrestricted_greedy_swap_trial,
    )
    text_rows = []
    for task in TEXT_TASKS:
        key = safe_key("text-paper", task.task_id)
        stored = STORE.load("intervention", key)
        if stored is None:
            inputs = BACKEND.build_inputs(prompt=task.prompt, modality="text")
            clean = unrestricted_greedy_completion(
                BACKEND, inputs, answer=task.clean_answer,
                max_new_tokens=TEXT_MAX_NEW_TOKENS,
            )
            bases = {
                layer: build_swap_basis_from_vectors(
                    TEXT_TOKEN_VECTORS[layer][task.source], TEXT_TOKEN_VECTORS[layer][task.target],
                    layer=layer, source=TEXT_CONCEPT_TOKENS[task.source], target=TEXT_CONCEPT_TOKENS[task.target],
                ) for layer in LAYERS
            }
            random_bases = {layer: random_two_direction_basis(basis, seed=20260820 + layer) for layer, basis in bases.items()}
            controls = ("zebra", "giraffe") if task.family == "implicit_two_hop" else ("Japan", "Brazil")
            unrelated = {
                layer: build_swap_basis_from_vectors(
                    TEXT_TOKEN_VECTORS[layer][controls[0]], TEXT_TOKEN_VECTORS[layer][controls[1]],
                    layer=layer, source=TEXT_CONCEPT_TOKENS[controls[0]], target=TEXT_CONCEPT_TOKENS[controls[1]],
                ) for layer in LAYERS
            }
            exact = unrestricted_greedy_swap_trial(
                BACKEND, inputs, bases=bases, alpha=1.0,
                answer=task.swapped_answer,
                max_new_tokens=TEXT_MAX_NEW_TOKENS,
            )
            random = unrestricted_greedy_swap_trial(
                BACKEND, inputs, bases=random_bases, alpha=1.0,
                answer=task.swapped_answer,
                max_new_tokens=TEXT_MAX_NEW_TOKENS,
            )
            unrelated_result = unrestricted_greedy_swap_trial(
                BACKEND, inputs, bases=unrelated, alpha=1.0,
                answer=task.swapped_answer,
                max_new_tokens=TEXT_MAX_NEW_TOKENS,
            )
            loading = capture_source_loading(
                BACKEND, inputs, vectors_by_layer=TEXT_TOKEN_VECTORS,
                source=task.source, target=task.target, unrelated=controls,
                sample_id=task.task_id, modality="text",
            )
            stored = {
                "task_id": task.task_id, "task": task.to_dict(),
                "output_endpoint": "unrestricted_greedy_complete_answer",
                "clean_correct": bool(clean["answer_match"]),
                "exact_alpha1_swapped_answer_generated": bool(exact["answer_match"]),
                "random_swapped_answer_generated": bool(random["answer_match"]),
                "unrelated_swapped_answer_generated": bool(unrelated_result["answer_match"]),
                "clean": clean,
                "exact": exact, "random": random, "unrelated": unrelated_result,
                "loading_rows": loading,
            }
            STORE.save("intervention", key, stored)
            work = "computed"
        else:
            work = "reused"
        text_rows.append(stored)
        print(
            task.task_id, work,
            "clean", stored["clean_correct"],
            "swapped answer generated",
            stored["exact_alpha1_swapped_answer_generated"],
        )
    TEXT_VERDICT = text_replication_verdict(text_rows)
    STORE.save("metric", "text_replication_verdict", TEXT_VERDICT)
    print(json.dumps(TEXT_VERDICT, indent=2))
elif RUN_STAGE1_TEXT_REPLICATION:
    print("skipped: real mode/model confirmation required")
'''
)

markdown("## 7. Stage 2 — development-only multimodal source loading (no interventions)")
code(
    r'''
LOCALIZATION = STORE.load("metric", "loading_localization")
PAIR_SELECTION = STORE.load("metric", "loading_pair_selection")
if REAL_MODE and RUN_STAGE2_MULTIMODAL_LOADING_DEVELOPMENT and CONFIRM_MODEL_LOAD and CONFIRM_DEVELOPMENT_BUDGET:
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
    MATCHED_VECTORS = {
        layer: {
            name: selected_lens_vector(
                MATCHED_LENSES["pooled"], BACKEND.unembedding_weight(),
                layer=layer, token_id=token.token_id,
            ) for name, token in TOKENS.items()
        } for layer in LAYERS
    }
    def build_inputs(group, modality):
        question = "Identify the animal from the evidence internally, then answer: How many legs does that animal typically have? Answer with one digit:\nAnswer:"
        if modality == "text":
            return BACKEND.build_inputs(prompt=f"Caption: {group['caption']}\n{question}", modality="text")
        if modality == "image":
            return BACKEND.build_inputs(prompt=question, modality="image", image=MEDIA["load_image"](group["image_path"]), media_path=group["image_path"])
        waveform, rate = MEDIA["load_audio"](group["audio_path"])
        return BACKEND.build_inputs(prompt=question, modality="spoken_audio", audio=waveform, sampling_rate=rate, media_path=group["audio_path"])
    loading_rows = []
    by_concept = {}
    for group in DEV_GROUPS:
        by_concept.setdefault(str(group.get("concept")), []).append(group)
    for source, target in CANDIDATE_PAIRS:
        for group in by_concept.get(source, []):
            for modality in ("text", "image", "spoken_audio"):
                key = safe_key("loading", source, target, group["group_id"], modality)
                stored = STORE.load("activation", key)
                if stored is None:
                    inputs = build_inputs(group, modality)
                    stored = {
                        "source": source, "target": target,
                        "group_id": group["group_id"], "image_id": group["image_id"],
                        "modality": modality,
                        "rows": capture_source_loading(
                            BACKEND, inputs, vectors_by_layer=MATCHED_VECTORS,
                            source=source, target=target, unrelated=CONTROL_CONCEPTS,
                            sample_id=f"{group['group_id']}:{modality}", modality=modality,
                        ),
                    }
                    STORE.save("activation", key, stored)
                    work = "computed"
                else:
                    work = "reused"
                loading_rows.extend(stored["rows"])
                print("loading", source, target, group["group_id"], modality, work)
    LOADING_REPORT = summarize_loading(loading_rows)
    PAIR_SELECTION = select_pair_from_loading(
        loading_rows, candidate_pairs=CANDIDATE_PAIRS,
        required_modalities=("text", "image", "spoken_audio"),
    )
    selected_source, selected_target = PAIR_SELECTION["selected_pair"]
    selected_rows = [row for row in loading_rows if row["source"] == selected_source and row["target"] == selected_target]
    LOCALIZATION = freeze_loading_localization(
        selected_rows, required_modalities=("text", "image", "spoken_audio"),
        candidate_layers=LAYERS, min_source_advantage=MIN_SOURCE_ADVANTAGE,
        evidence_position_margin=EVIDENCE_POSITION_MARGIN,
    )
    STORE.save("metric", "loading_report", LOADING_REPORT)
    STORE.save("metric", "loading_pair_selection", PAIR_SELECTION)
    STORE.save("metric", "loading_localization", LOCALIZATION)
    print(json.dumps(PAIR_SELECTION, indent=2))
    print(json.dumps(LOCALIZATION, indent=2))
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
    CONFIRMATION_DESIGN = freeze_confirmation_design(
        text_verdict=TEXT_VERDICT, localization=LOCALIZATION,
        pair=PAIR_SELECTION["selected_pair"], alpha=MULTIMODAL_PRIMARY_ALPHA,
        sensitivity_alpha=MULTIMODAL_SENSITIVITY_ALPHA,
        prompt_protocol=PROMPT_PROTOCOL,
        development_population_digest=POPULATION_PLAN["plan_digest"],
    )
    CONFIRMATION_DESIGN["forbidden_development_image_ids"] = sorted(DEV_IMAGE_IDS)
    CONFIRMATION_DESIGN["forbidden_development_group_ids"] = sorted(DEV_GROUP_IDS)
    CONFIRMATION_DESIGN["forbidden_prior_image_ids"] = sorted(PRIOR_EXCLUDED_IMAGES)
    CONFIRMATION_DESIGN["design_digest"] = payload_checksum(
        {k: v for k, v in CONFIRMATION_DESIGN.items() if k != "design_digest"}
    )
    STORE.save("metric", "confirmation_design", CONFIRMATION_DESIGN)
    (RUN_DIR / "frozen_confirmation_design.json").write_text(json.dumps(CONFIRMATION_DESIGN, indent=2))
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
    from jlens.mmpilot.multimodal_lens import build_swap_bases_for_lens
    from jlens.mmpilot.store import safe_key
    from jlens.mmpilot.workspace_replication import (
        TEXT_MAX_NEW_TOKENS, assert_fresh_population,
        unrestricted_greedy_completion, unrestricted_greedy_swap_trial,
    )
    if CONFIRMATION_DESIGN is None:
        CONFIRMATION_DESIGN = STORE.load("metric", "confirmation_design")
    if CONFIRMATION_DESIGN is None:
        raise RuntimeError("Stage 4 requires the frozen Stage-3 design")
    source, target = CONFIRMATION_DESIGN["pair"]
    freshness = assert_fresh_population(
        CONFIRM_GROUPS,
        forbidden_image_ids=sorted(set(
            CONFIRMATION_DESIGN["forbidden_development_image_ids"]
        ) | set(CONFIRMATION_DESIGN["forbidden_prior_image_ids"])),
        forbidden_group_ids=CONFIRMATION_DESIGN["forbidden_development_group_ids"],
    )
    MEDIA = drive_media_loaders(journal=RetryJournal())
    TOKENS = {name: resolve_concept_token(BACKEND.encode_candidate, name) for name in (source, target, *CONTROL_CONCEPTS)}
    answers = {"bird": "2", "cat": "4", "zebra": "4", "giraffe": "4"}
    if answers[source] == answers[target]:
        raise RuntimeError("selected pair has no downstream leg-count contrast")
    band = CONFIRMATION_DESIGN["layer_band"]
    exact_bases = build_swap_bases_for_lens(
        MATCHED_LENSES["pooled"], BACKEND.unembedding_weight(), layers=band,
        source=TOKENS[source], target=TOKENS[target],
    )
    random_bases = {layer: random_two_direction_basis(basis, seed=20260821 + layer) for layer, basis in exact_bases.items()}
    unrelated_bases = build_swap_bases_for_lens(
        MATCHED_LENSES["pooled"], BACKEND.unembedding_weight(), layers=band,
        source=TOKENS[CONTROL_CONCEPTS[0]], target=TOKENS[CONTROL_CONCEPTS[1]],
    )
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
    selected_groups = [row for row in CONFIRM_GROUPS if str(row.get("concept")) == source][:CONFIRMATION_IMAGES_PER_SOURCE]
    rows = []
    for group in selected_groups:
        for modality in ("text", "image", "spoken_audio"):
            for kind in ("identity", "property"):
                inputs = build_inputs(group, modality, kind)
                source_answer = source if kind == "identity" else answers[source]
                target_answer = target if kind == "identity" else answers[target]
                clean = unrestricted_greedy_completion(
                    BACKEND, inputs, answer=source_answer,
                    max_new_tokens=TEXT_MAX_NEW_TOKENS,
                )
                key = safe_key("fresh-confirm", group["group_id"], modality, kind)
                stored = STORE.load("intervention", key)
                if stored is None:
                    conditions = {}
                    for name, bases, alpha in (
                        ("exact_alpha1", exact_bases, 1.0),
                        ("random_alpha1", random_bases, 1.0),
                        ("unrelated_alpha1", unrelated_bases, 1.0),
                        ("exact_alpha075", exact_bases, 0.75),
                    ):
                        trial = unrestricted_greedy_swap_trial(
                            BACKEND, inputs, bases=bases, alpha=alpha,
                            answer=target_answer,
                            max_new_tokens=TEXT_MAX_NEW_TOKENS,
                            position_rule=CONFIRMATION_DESIGN[
                                "position_rule_by_modality"
                            ][modality],
                        )
                        conditions[name] = {
                            **trial,
                            "patched_surface": trial["generated_text"],
                            "success": bool(trial["answer_match"]),
                        }
                    stored = {
                        "group_id": group["group_id"], "image_id": group["image_id"],
                        "source": source, "target": target, "modality": modality,
                        "prompt_kind": kind, "source_answer": source_answer,
                        "target_answer": target_answer,
                        "output_endpoint": "unrestricted_greedy_complete_answer",
                        "clean_surface": clean["generated_text"],
                        "clean_correct": bool(clean["answer_match"]),
                        "clean": clean,
                        "conditions": conditions,
                    }
                    STORE.save("intervention", key, stored)
                    work = "computed"
                else:
                    work = "reused"
                rows.append(stored)
                print("confirmation", len(rows), work, group["group_id"], modality, kind)
    cells = []
    for kind in ("identity", "property"):
        for modality in ("text", "image", "spoken_audio"):
            cell = [row for row in rows if row["prompt_kind"] == kind and row["modality"] == modality]
            cells.append({
                "prompt_kind": kind, "modality": modality, "n": len(cell),
                "clean_capability": sum(row["clean_correct"] for row in cell) / len(cell),
                "alpha1_success": sum(row["conditions"]["exact_alpha1"]["success"] for row in cell) / len(cell),
                "alpha075_success": sum(row["conditions"]["exact_alpha075"]["success"] for row in cell) / len(cell),
                "random_alpha1_success": sum(row["conditions"]["random_alpha1"]["success"] for row in cell) / len(cell),
                "unrelated_alpha1_success": sum(row["conditions"]["unrelated_alpha1"]["success"] for row in cell) / len(cell),
            })
    from jlens.mmpilot.workspace_replication import holm_adjust, paired_binary_superiority
    paired = {}
    raw_p = {}
    for kind in ("identity", "property"):
        kind_rows = [row for row in rows if row["prompt_kind"] == kind]
        for control in ("random_alpha1", "unrelated_alpha1"):
            key = f"{kind}_exact_vs_{control}"
            paired[key] = paired_binary_superiority(
                [row["conditions"]["exact_alpha1"]["success"] for row in kind_rows],
                [row["conditions"][control]["success"] for row in kind_rows],
            )
            raw_p[key] = paired[key]["one_sided_exact_p"]
    adjusted = holm_adjust(raw_p)
    for key in paired:
        paired[key]["holm_p"] = adjusted[key]
        paired[key]["passed"] = adjusted[key] <= CONFIRMATION_FAMILYWISE_ALPHA
    capability = all(cell["clean_capability"] >= 0.75 for cell in cells)
    primary = all(
        cell["alpha1_success"] >= MIN_CONFIRMATION_SUCCESS_RATE
        and cell["alpha1_success"] > max(
            cell["random_alpha1_success"], cell["unrelated_alpha1_success"]
        ) for cell in cells
    ) and all(row["passed"] for row in paired.values())
    downstream = all(
        cell["alpha1_success"] >= MIN_CONFIRMATION_SUCCESS_RATE
        for cell in cells if cell["prompt_kind"] == "property"
    )
    verdict = (
        "FRESH_MULTIMODAL_DOWNSTREAM_RECOMPUTATION_GO"
        if capability and primary and downstream
        else "FRESH_MULTIMODAL_CONFIRMATION_NO_GO"
    )
    CONFIRMATION_REPORT = {
        "version": "mmpilot.fresh_multimodal_confirmation.v1",
        "verdict": verdict, "design": CONFIRMATION_DESIGN,
        "freshness": freshness, "cells": cells, "paired_tests": paired,
        "familywise_alpha": CONFIRMATION_FAMILYWISE_ALPHA,
        "minimum_cell_success_rate": MIN_CONFIRMATION_SUCCESS_RATE,
        "rows": rows,
        "teacher_forcing_used": False, "candidate_list_supplied": False,
        "alpha1_is_primary": True, "alpha075_is_sensitivity_only": True,
    }
    CONFIRMATION_REPORT["report_checksum"] = payload_checksum(CONFIRMATION_REPORT)
    STORE.save("metric", "fresh_confirmation_report", CONFIRMATION_REPORT)
    (RUN_DIR / "fresh_multimodal_confirmation_report.json").write_text(json.dumps(CONFIRMATION_REPORT, indent=2, default=str))
    print(json.dumps({k: v for k, v in CONFIRMATION_REPORT.items() if k != "rows"}, indent=2))
'''
)

markdown("## 10. Stage 5 — write the integrated report and stop")
code(
    r'''
if RUN_STAGE5_WRITE_REPORT:
    TEXT_VERDICT = TEXT_VERDICT or STORE.load("metric", "text_replication_verdict")
    PAIR_SELECTION = PAIR_SELECTION or STORE.load("metric", "loading_pair_selection")
    LOCALIZATION = LOCALIZATION or STORE.load("metric", "loading_localization")
    CONFIRMATION_DESIGN = CONFIRMATION_DESIGN or STORE.load("metric", "confirmation_design")
    CONFIRMATION_REPORT = CONFIRMATION_REPORT or STORE.load("metric", "fresh_confirmation_report")
    FINAL = {
        "schema": "jlens.mmpilot.paper_first_workspace_study.v1",
        "scientific_config": SCIENTIFIC_CONFIG,
        "population_plan": POPULATION_PLAN,
        "text_replication": TEXT_VERDICT,
        "loading_pair_selection": PAIR_SELECTION,
        "loading_localization": LOCALIZATION,
        "frozen_confirmation_design": CONFIRMATION_DESIGN,
        "fresh_confirmation": CONFIRMATION_REPORT,
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
