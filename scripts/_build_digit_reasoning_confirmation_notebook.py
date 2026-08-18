"""Generate the prospective digit-endpoint reasoning confirmation notebook."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = (
    ROOT
    / "notebooks"
    / "multimodal_jspace_digit_reasoning_confirmation_colab.ipynb"
)
CELLS: list[tuple[str, str]] = []


def markdown(value: str) -> None:
    CELLS.append(("markdown", value.strip("\n")))


def code(value: str) -> None:
    CELLS.append(("code", value.strip("\n")))


markdown(
    r"""
# Exact-swap confirmation — unrestricted digit output (Gemma 4 E4B)

The completed α=2 digit study remains immutable: it established that doubling
the exchange is destructive in Gemma 4. This separately fingerprinted v2 study
freezes a new population and asks the canonical paper-comparable question:

> Does a two-coordinate J-lens exchange of the represented animal identity make
> the target animal's leg-count digit the unique top token in Gemma's complete
> next-token distribution?

The answer is never appended. No candidate list decides the output. Every
primary score is checked against a one-token deterministic greedy continuation.

## Frozen design

* contiguous independently validated band **L33–L40**;
* `bird → cat` (`2 → 4`) and `cat → bird` (`4 → 2`);
* text, image and native spoken-caption audio;
* paper-style two-coordinate exchange at every original prompt position;
* **α=1 is the primary and only nonzero strength**: the exact exchange;
* zero, norm-matched α=1 random and α=1 unrelated-coordinate controls;
* a direct digit-coordinate exchange is a required positive control;
* a norm-matched random α=1 direct-answer control is also required;
* at least eight distinct fresh photographs per direction×modality cell;
* every scored or generated forward pass is saved atomically and resumes by
  checksum-valid unit key. A disconnect loses no completed trial.

## Stages

| stage | runtime | purpose |
|---|---|---|
| 0 | CPU | Verify the eight lens artifacts and freeze the protocol. |
| 1 | L4/A100 | Select fresh media, screen clean digit capability, and run the swaps. |
| 2 | CPU | Judge only from saved units and write the final report. |

No lens is fitted, there are no backward passes, and α>1 is never run.
"""
)

markdown("## 0. Bootstrap")
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
print("commit", subprocess.run(
    ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
).stdout.strip())
'''
)

markdown("## 1. Configuration")
code(
    r'''
# CPU preflight, GPU confirmation, CPU report. A clean end-to-end GPU session
# sets all three stage switches and both confirmations True.
RUN_PREFLIGHT_CPU = False
RUN_DIGIT_CONFIRMATION_GPU = False
RUN_FINAL_REPORT_CPU = False
CONFIRM_MODEL_LOAD = False
CONFIRM_PASS_BUDGET = False

# For a report-only resumed session, point this at the completed run. Leave
# None when the GPU stage runs in this kernel.
REPORT_RUN_DIR = None

REAL_MODE = any((RUN_PREFLIGHT_CPU, RUN_DIGIT_CONFIRMATION_GPU, RUN_FINAL_REPORT_CPU))
GPU_STAGE = bool(
    RUN_DIGIT_CONFIRMATION_GPU and CONFIRM_MODEL_LOAD and CONFIRM_PASS_BUDGET
)
if RUN_DIGIT_CONFIRMATION_GPU and not GPU_STAGE:
    print("GPU stage requested but blocked: set both confirmations after reading the budget.")

MODEL_REPO_ID = "google/gemma-4-E4B-it"
MODEL_REVISION = "fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd"
TRANSFORMERS_VERSION_EXPECTED = "5.13.1"
EXPECT_N_LAYERS, EXPECT_D_MODEL, EXPECT_VOCAB = 42, 2560, 262144
AUDIO_PROTOCOL_FINGERPRINT = (
    "sha256:9ad8bcc9420a7983f6e3b75d5d7080c0e2fcf0a94a76431917fcde73ba777920"
)

PAIR_CONCEPTS = ("bird", "cat")
CONTROL_CONCEPTS = ("zebra", "giraffe")
POPULATION_CONCEPTS = (*PAIR_CONCEPTS, *CONTROL_CONCEPTS)
CANDIDATE_IMAGES_PER_CONCEPT = 24
ANALYSIS_IMAGES_PER_CELL = 8
SELECTION_SEED = "alpha1-exact-swap-independent-20260817-v2"
RANDOM_CONTROL_SEED_INTERMEDIATE = 2026081701
RANDOM_CONTROL_SEED_ANSWER = 2026081702

from jlens.mmpilot.digit_reasoning_confirmation import (
    CONFIRMATION_BAND, CONFIRMATION_CONDITIONS,
    DIGIT_CONFIRMATION_PROTOCOL_VERSION, DigitConfirmationThresholds,
    confirmation_design, confirmation_pass_budget,
)

THRESHOLDS = DigitConfirmationThresholds(
    min_images_per_cell=ANALYSIS_IMAGES_PER_CELL,
    min_primary_success_rate_per_cell=0.50,
    min_positive_control_rate_per_cell=0.50,
    familywise_alpha=0.05,
    bootstrap_samples=10_000,
    bootstrap_seed=20260813,
)
DESIGN = confirmation_design(thresholds=THRESHOLDS)
BUDGET = confirmation_pass_budget(
    n_images_per_direction=ANALYSIS_IMAGES_PER_CELL,
    capability_candidate_images_per_direction=CANDIDATE_IMAGES_PER_CONCEPT,
)

print("protocol       ", DIGIT_CONFIRMATION_PROTOCOL_VERSION)
print("band           ", list(CONFIRMATION_BAND))
print("endpoint       ", DESIGN["concept_to_digit_answer"])
print("exact alpha    ", DESIGN["primary_alpha"])
print("conditions     ", CONFIRMATION_CONDITIONS)
print("threshold      ", DESIGN["threshold_digest"])
print("forward passes ", BUDGET["total_forward_passes"])
print("backward passes", BUDGET["backward_passes"])
print("resume loss    ", BUDGET["maximum_completed_work_lost_on_disconnect"])
'''
)

markdown("## 2. Explicit immutable inputs")
code(
    r'''
RUNS_ROOT = Path("/content/drive/MyDrive/jacobian-lens-gemma/runs")
CORRECTED_RUN_DIR = RUNS_ROOT / "mmband" / "bandcorr_real_eb5b00f135e4"
EXPANDED_MANIFEST_CACHE = (
    RUNS_ROOT / "mml32_l32_followup_20260808T182717" / "expanded_manifest.json"
)
PRIOR_EXCLUSION_SET = Path(
    "/content/drive/MyDrive/datasets/cstf_spokencoco_derived/"
    "jlens_l32_resolution_prep_v1/"
    "prep_020ebbe6f832aece5ece6cb8bee994ca/exclusion_set.json"
)

# Every population whose causal output has already been inspected is excluded.
# Report checksums are pins, not values learned from the directory being checked.
COMPLETED_CAUSAL_RUNS = {
    "single_layer_v1": (
        RUNS_ROOT / "mmpaper_real_24be1d028bf1", "paper_reasoning_swap_report.json",
        "sha256:a60f3336bf8acdc98dc1a434698104eaa98b3192c44f43fa5ab21212826ae397",
    ),
    "single_layer_v2": (
        RUNS_ROOT / "mmpaper2_real_04ab55235502", "paper_reasoning_swap_v2_report.json",
        "sha256:b64ce3cec51371769b908d14342fbf42f64a6dccb82f8d235ad81d643815ddc6",
    ),
    "alpha2_capability_screen": (
        RUNS_ROOT / "mmpaperconfirm_real_6b0745c08d84",
        "paper_reasoning_swap_alpha2_confirmation_report.json",
        "sha256:37d32605b24984f09c0dfccaab7c7ea98e217bef82412bd28576384b22f23c11",
    ),
    "alpha2_independent_confirmation": (
        RUNS_ROOT / "mmpaperconfirm_real_a496d5ad7f18",
        "paper_reasoning_swap_alpha2_confirmation_report.json",
        "sha256:a81d0190ef41a140c04f571124e3f4e06ff785a81bd00e78c7d139c7967ccd4f",
    ),
    "validated_band_followup": (
        RUNS_ROOT / "mmband33" / "band3340_real_2a72bda9b4ba",
        "l33_l40_validated_band_followup_report.json",
        "sha256:f808ac89236c640269698d18c999412e0164533349b69a4d9960cdcc1ce263cb",
    ),
    "word_endpoint_full_vocab": (
        RUNS_ROOT / "mmfv" / "mmfv_real_bfb07903e961",
        "full_vocabulary_causal_validation_report.json",
        "sha256:669a821c689b742ae6e5bfdead6207f0bb058f569e794a50289b703b5f586e08",
    ),
    "alpha2_digit_confirmation": (
        RUNS_ROOT / "mmdigitconfirm" / "mmdigitconfirm_real_68c182bfc025",
        "digit_reasoning_confirmation_report.json",
        "sha256:8c6188ffe36d006d942395e7bbe3e708180a65041c5db599b6cc23f2bfcff043",
    ),
}
DIGIT_RUN_ROOT = RUNS_ROOT / "mmalpha1confirm"

if REAL_MODE and IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)

if REAL_MODE:
    required = [CORRECTED_RUN_DIR]
    if RUN_DIGIT_CONFIRMATION_GPU:
        required += [EXPANDED_MANIFEST_CACHE, PRIOR_EXCLUSION_SET]
        required += [directory for directory, _, _ in COMPLETED_CAUSAL_RUNS.values()]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("configured immutable input(s) missing:\n  " + "\n  ".join(missing))
    print("all configured immutable inputs exist")
else:
    print("dry configuration: no Drive read and no model loaded")
'''
)

markdown("## 3. CPU preflight — eight validated lenses, no fitting")
code(
    r'''
PREFLIGHT = None
CORRECTED_REPORT = None
CORRECTED_ARTIFACTS = {}
LENS_CHECKSUMS = {}

if RUN_PREFLIGHT_CPU or RUN_DIGIT_CONFIRMATION_GPU:
    from jlens.mmpilot.validated_band_followup import (
        assert_corrected_run_unmodified, assert_followup_band,
        corrected_run_digest, discover_corrected_band_lenses,
        read_corrected_validation_report,
    )
    _before = corrected_run_digest(CORRECTED_RUN_DIR)
    _report_path, CORRECTED_REPORT = read_corrected_validation_report(
        CORRECTED_RUN_DIR,
        expected_model_repo_id=MODEL_REPO_ID,
        expected_model_revision=MODEL_REVISION,
    )
    _admission = assert_followup_band(CORRECTED_REPORT)
    CORRECTED_ARTIFACTS, _discovery = discover_corrected_band_lenses(
        CORRECTED_RUN_DIR, report=CORRECTED_REPORT, layers=CONFIRMATION_BAND
    )
    _immutability = assert_corrected_run_unmodified(
        _before, corrected_run_digest(CORRECTED_RUN_DIR)
    )
    LENS_CHECKSUMS = {
        layer: artifact.lens_checksum
        for layer, artifact in sorted(CORRECTED_ARTIFACTS.items())
    }
    PREFLIGHT = {
        "report_path": str(_report_path),
        "report_checksum": CORRECTED_REPORT["report_checksum"],
        "admission": _admission,
        "artifact_discovery": _discovery,
        "lens_checksums": {str(k): v for k, v in LENS_CHECKSUMS.items()},
        "corrected_run_unchanged": _immutability["identical"],
        "layers": sorted(LENS_CHECKSUMS),
        "fitting_performed": False,
        "backward_passes": 0,
    }
    if PREFLIGHT["layers"] != list(CONFIRMATION_BAND):
        raise RuntimeError("preflight did not resolve every physical layer L33-L40")
    print("PREFLIGHT PASS")
    print("  corrected report", PREFLIGHT["report_checksum"])
    print("  lenses          ", PREFLIGHT["layers"])
    for layer, checksum in LENS_CHECKSUMS.items():
        print(f"    L{layer}: {checksum}")
    print("  run unchanged   ", PREFLIGHT["corrected_run_unchanged"])
    print("  fitting/backward", PREFLIGHT["fitting_performed"], PREFLIGHT["backward_passes"])
else:
    print("skipped: preflight not requested")
'''
)

markdown("## 4. Budget gate — read before loading the model")
code(
    r'''
print("=" * 78)
print("FINAL CONFIRMATION PASS BUDGET")
print("=" * 78)
for key, value in BUDGET.items():
    if key != "budget_digest":
        print(f"  {key:<46} {value}")
print("  budget_digest", BUDGET["budget_digest"])
print()
print("Set RUN_DIGIT_CONFIRMATION_GPU, CONFIRM_MODEL_LOAD and")
print("CONFIRM_PASS_BUDGET True to authorize these passes.")
'''
)

markdown("## 5. GPU confirmation — fresh population, atomic units")
code(
    r'''
STORE = None
RUN_DIR = None
POPULATION = None
EXCLUSION = None
ENDPOINT = None
CAPABILITY = {"all_cells_sufficient": False, "cells": []}
FINGERPRINT_CONFIG = None
CAUSAL_STAGE_RAN = False

if GPU_STAGE:
    import getpass, hashlib, torch
    from jlens.lens import JacobianLens
    from jlens.mmpilot.coordinate_swap import (
        PRIMARY_POSITION_RULE, assert_open_prompt_protocol,
        build_swap_basis_from_vectors,
        coordinate_swap_band, random_two_direction_basis, resolve_concept_token,
    )
    from jlens.mmpilot.digit_reasoning_confirmation import (
        DIGIT_ANSWERS, DIGIT_CONFIRMATION_RUN_PREFIX,
        confirmation_fingerprint, confirmation_trial_key, resolve_digit_endpoints,
    )
    from jlens.mmpilot.evidence import EvidenceConfig
    from jlens.mmpilot.full_vocabulary import (
        greedy_generate, score_unrestricted_next_token, unrestricted_trial_record,
    )
    from jlens.mmpilot.media_io import RetryJournal, drive_media_loaders
    from jlens.mmpilot.paper_reasoning_swap import hidden_animal_population
    from jlens.mmpilot.prompt_protocol import (
        Evidence, HIDDEN_ANIMAL_LEGS, build_backend_inputs, build_protocol_prompt,
        concept_spec, prompt_protocol_fingerprint,
    )
    from jlens.mmpilot.real_backend import build_processor_backend, build_real_backend
    from jlens.mmpilot.selection import stable_rank
    from jlens.mmpilot.store import RunFingerprint, UnitStore, payload_checksum
    from jlens.mmpilot.tri_modal import assert_audio_protocol

    if PREFLIGHT is None:
        raise RuntimeError("GPU stage requires the preflight to pass in this kernel")
    if not torch.cuda.is_available():
        raise RuntimeError("GPU confirmation requires an L4 or A100 runtime")
    if not os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = getpass.getpass("HF_TOKEN (input hidden): ").strip()

    # Token endpoint freezes before the fresh population is opened and before
    # model weights load. This loader has no model-loading entry point.
    _processor_bundle = build_processor_backend(
        MODEL_REPO_ID, revision=MODEL_REVISION, token=os.environ["HF_TOKEN"]
    )
    ENDPOINT = resolve_digit_endpoints(_processor_bundle.backend)
    print("DIGIT ENDPOINT FROZEN BEFORE POPULATION")
    print("  token ids", ENDPOINT["token_ids"], "decoded", ENDPOINT["decoded"])

    # Build the complete spent-media exclusion before selecting a photograph.
    _prior = json.loads(PRIOR_EXCLUSION_SET.read_text(encoding="utf-8"))
    _excluded_images = {str(v) for v in _prior.get("image_ids", [])}
    _excluded_groups = {str(v) for v in _prior.get("group_ids", [])}
    _spent = {}
    for _name, (_directory, _report_name, _pin) in COMPLETED_CAUSAL_RUNS.items():
        _report = json.loads((_directory / _report_name).read_text(encoding="utf-8"))
        if _report.get("report_checksum") != _pin:
            raise RuntimeError(f"{_name}: completed report checksum does not match its pin")
        _images, _groups = set(), set()
        # Prefer the compact population manifest carried by the completed
        # report.  If an older report omitted it, capability units cover the
        # sampled population and are far cheaper to audit than thousands of
        # intervention units.  Intervention units are a last-resort fallback
        # only; this keeps a clean Colab start from spending hours enumerating
        # Drive files before the model is loaded.
        _recorded_population = list(_report.get("population_groups", []))
        _recorded_population.extend(
            list((_report.get("population") or {}).get("groups", []))
        )
        for _row in _recorded_population:
            if _row.get("image_id"):
                _images.add(str(_row["image_id"]))
            if _row.get("group_id"):
                _groups.add(str(_row["group_id"]))

        _stages_read = []
        for _stage in (() if _images else ("capability",)):
            _stages_read.append(_stage)
            for _path in sorted((_directory / "units" / _stage).glob("*.json")):
                _stored = json.loads(_path.read_text(encoding="utf-8"))
                _payload = _stored.get("payload") if isinstance(_stored.get("payload"), dict) else _stored
                if _stored.get("payload") is not None and _stored.get("unit_checksum") != payload_checksum(_payload):
                    raise RuntimeError(f"checksum-invalid spent unit: {_path}")
                if _payload.get("image_id"):
                    _images.add(str(_payload["image_id"]))
                if _payload.get("group_id"):
                    _groups.add(str(_payload["group_id"]))

        if not _images:
            _stage = "intervention"
            _stages_read.append(_stage)
            for _path in sorted((_directory / "units" / _stage).glob("*.json")):
                _stored = json.loads(_path.read_text(encoding="utf-8"))
                _payload = _stored.get("payload") if isinstance(_stored.get("payload"), dict) else _stored
                if _stored.get("payload") is not None and _stored.get("unit_checksum") != payload_checksum(_payload):
                    raise RuntimeError(f"checksum-invalid spent unit: {_path}")
                if _payload.get("image_id"):
                    _images.add(str(_payload["image_id"]))
                if _payload.get("group_id"):
                    _groups.add(str(_payload["group_id"]))
        if not _images:
            raise RuntimeError(f"{_name}: no spent image identity could be recovered")
        _excluded_images |= _images
        _excluded_groups |= _groups
        _spent[_name] = {
            "report_checksum": _pin,
            "n_images": len(_images),
            "n_groups": len(_groups),
            "identity_source": (
                "report.population"
                if _recorded_population
                else "+".join(_stages_read)
            ),
        }
        print(f"  excluded {len(_images):3d} images spent by {_name}")
    EXCLUSION = {
        "prior_exclusion_checksum": "sha256:" + hashlib.sha256(PRIOR_EXCLUSION_SET.read_bytes()).hexdigest(),
        "completed_causal_runs": _spent,
        "n_excluded_images": len(_excluded_images),
        "n_excluded_groups": len(_excluded_groups),
    }
    EXCLUSION["exclusion_digest"] = payload_checksum(EXCLUSION)

    _manifest_bytes = EXPANDED_MANIFEST_CACHE.read_bytes()
    _manifest = json.loads(_manifest_bytes)
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
            require_visual_evidence=True,
            require_caption_evidence=False,
        ),
        images_per_concept=CANDIDATE_IMAGES_PER_CONCEPT,
        seed=SELECTION_SEED,
    )
    _population_images = {str(row["image_id"]) for row in POPULATION["groups"]}
    _population_groups = {str(row["group_id"]) for row in POPULATION["groups"]}
    if _population_images & _excluded_images or _population_groups & _excluded_groups:
        raise RuntimeError("fresh population overlaps a completed causal population")
    if len(_population_images) != len(_population_groups):
        raise RuntimeError("the population is not one synchronized group per photograph")
    print("fresh population", POPULATION["population_digest"], len(_population_images), "images")
    del _manifest, _eligible

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

    # Every band layer uses its own independently validated Jacobian matrix.
    _token_names = (*PAIR_CONCEPTS, *CONTROL_CONCEPTS, "2", "4")
    TOKENS = {name: resolve_concept_token(BACKEND.encode_candidate, name) for name in _token_names}
    for _digit in ("2", "4"):
        if TOKENS[_digit].token_id != ENDPOINT["token_ids"][_digit]:
            raise RuntimeError(
                f"coordinate token for {_digit} disagrees with frozen output endpoint"
            )
    _unembedding = BACKEND.unembedding_weight()
    _rows = {name: _unembedding[token.token_id].detach().float().cpu() for name, token in TOKENS.items()}
    TOKEN_VECTORS, _loaded = {}, {}
    for _layer in CONFIRMATION_BAND:
        _source = CORRECTED_ARTIFACTS[_layer]
        if _source.lens_path not in _loaded:
            _loaded[_source.lens_path] = JacobianLens.load(_source.lens_path)
        _jacobian = _loaded[_source.lens_path].jacobians[_source.layer_key_in_file].detach().float().cpu()
        TOKEN_VECTORS[_layer] = {name: row @ _jacobian for name, row in _rows.items()}
    del _loaded, _unembedding, _rows

    def selected_bases(source_name, target_name):
        return {
            layer: build_swap_basis_from_vectors(
                TOKEN_VECTORS[layer][source_name], TOKEN_VECTORS[layer][target_name],
                layer=layer, source=TOKENS[source_name], target=TOKENS[target_name],
            )
            for layer in CONFIRMATION_BAND
        }

    def intervention_diagnostics(stats):
        by_layer = {}
        for layer, layer_stats in sorted(stats.items()):
            swap = layer_stats["swap"]
            before = [float(value) for value in swap["activation_norm_before"]]
            after = [float(value) for value in swap["activation_norm_after"]]
            update = [float(value) for value in swap["update_norm"]]
            if any(value <= 0.0 for value in before):
                raise RuntimeError(f"L{layer}: zero activation norm in intervention")
            by_layer[str(layer)] = {
                "max_update_to_activation_ratio": max(
                    delta / base for delta, base in zip(update, before)
                ),
                "max_after_to_before_ratio": max(
                    new / base for new, base in zip(after, before)
                ),
                "min_after_to_before_ratio": min(
                    new / base for new, base in zip(after, before)
                ),
                "max_coordinate_update_error": float(
                    swap["max_coordinate_update_error"]
                ),
                "alpha_one_is_exact_exchange": bool(
                    swap["alpha_one_is_exact_exchange"]
                ),
            }
        return {
            "by_layer": by_layer,
            "all_finite": all(
                torch.isfinite(torch.tensor(value)).all().item()
                for row in by_layer.values()
                for value in row.values()
                if isinstance(value, (int, float))
            ),
            "max_coordinate_update_error": max(
                row["max_coordinate_update_error"] for row in by_layer.values()
            ),
            "all_layers_are_exact_alpha_one_exchange": all(
                row["alpha_one_is_exact_exchange"] for row in by_layer.values()
            ),
        }

    MEDIA = drive_media_loaders(journal=RetryJournal())

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

    def make_input(group, modality, source, target, evidence=None):
        evidence = evidence or load_evidence(group, modality)
        built = build_protocol_prompt(
            protocol=HIDDEN_ANIMAL_LEGS, evidence=evidence,
            external_candidates=("2", "4"), source=concept_spec(source),
            target=concept_spec(target), encode_candidate=BACKEND.encode_candidate,
        )
        offline = group["caption"] if modality != "text" else None
        return built, build_backend_inputs(BACKEND, built, transcript=offline)

    _probe = POPULATION["groups"][0]
    PROMPT_PROTOCOLS = []
    for _modality in ("text", "image", "spoken_audio"):
        _built, _inputs = make_input(
            _probe, _modality, _probe["concept"],
            next(name for name in PAIR_CONCEPTS if name != _probe["concept"]),
        )
        PROMPT_PROTOCOLS.append(prompt_protocol_fingerprint(
            _built, model_revision=_bundle.model_revision,
            processor_revision=_bundle.processor_revision,
            audio_protocol_fingerprint=AUDIO_RECORD["protocol_fingerprint"],
        ))
        assert_open_prompt_protocol(PROMPT_PROTOCOLS[-1])

    FINGERPRINT_CONFIG = confirmation_fingerprint(
        design=DESIGN, endpoint=ENDPOINT, population=POPULATION,
        exclusion=EXCLUSION, lens_checksums=LENS_CHECKSUMS,
        model_pins={
            "repo_id": MODEL_REPO_ID, "model_revision": _bundle.model_revision,
            "processor_revision": _bundle.processor_revision,
            "transformers_version": TRANSFORMERS_VERSION_EXPECTED,
        },
        audio_protocol_fingerprint=AUDIO_RECORD["protocol_fingerprint"],
        prompt_protocol=PROMPT_PROTOCOLS,
        seeds={
            "selection": SELECTION_SEED,
            "random_intermediate": RANDOM_CONTROL_SEED_INTERMEDIATE,
            "random_answer": RANDOM_CONTROL_SEED_ANSWER,
            "bootstrap": THRESHOLDS.bootstrap_seed,
        },
    )
    _digest = FINGERPRINT_CONFIG["confirmation_fingerprint_digest"]
    _fingerprint = RunFingerprint(
        mode="digit_reasoning_confirmation", model_repo_id=MODEL_REPO_ID,
        model_revision=_bundle.model_revision,
        processor_revision=_bundle.processor_revision,
        layers=CONFIRMATION_BAND, lens_checksum=_digest,
        manifest_checksum="sha256:" + hashlib.sha256(_manifest_bytes).hexdigest(),
        split_id=SELECTION_SEED, intervention_config=FINGERPRINT_CONFIG,
        selection_config={
            "population_digest": POPULATION["population_digest"],
            "exclusion_digest": EXCLUSION["exclusion_digest"],
        },
        extra={"protocol_version": DIGIT_CONFIRMATION_PROTOCOL_VERSION,
               "endpoint_digest": ENDPOINT["endpoint_digest"]},
    )
    RUN_DIR = DIGIT_RUN_ROOT / f"{DIGIT_CONFIRMATION_RUN_PREFIX}_real_{_digest.split(':')[1][:12]}"
    STORE = UnitStore(RUN_DIR, _fingerprint)
    print("run directory", RUN_DIR)
    print("resume       ", STORE.open())
    STORE.save("metric", "digit_confirmation_protocol", {
        "kind": "digit_confirmation_protocol",
        "endpoint": ENDPOINT,
        "fingerprint": FINGERPRINT_CONFIG,
        "population": POPULATION,
        "exclusion": EXCLUSION,
        "budget": BUDGET,
    })

    def score_distribution(inputs, source, target):
        return score_unrestricted_next_token(
            BACKEND, inputs,
            target_token_ids={
                "source": ENDPOINT["token_ids"][DIGIT_ANSWERS[source]],
                "target": ENDPOINT["token_ids"][DIGIT_ANSWERS[target]],
            },
            top_k=20, expected_vocab_size=EXPECT_VOCAB,
        )

    # Clean capability: unrestricted source digit plus actual one-token greedy.
    _groups_by_id = {str(row["group_id"]): row for row in POPULATION["groups"]}
    _computed = _reused = 0
    for group in POPULATION["groups"]:
        source = group["concept"]
        target = next(name for name in PAIR_CONCEPTS if name != source)
        for modality in DESIGN["modalities"]:
            _score_key = confirmation_trial_key(
                group_id=group["group_id"], modality=modality,
                arm="clean", condition="clean", kind="score",
            )
            _greedy_key = confirmation_trial_key(
                group_id=group["group_id"], modality=modality,
                arm="clean", condition="clean", kind="greedy",
            )
            if STORE.has("capability", _score_key) and STORE.has("capability", _greedy_key):
                _reused += 2
                continue
            built, inputs = make_input(group, modality, source, target)
            if not STORE.has("capability", _score_key):
                scored = score_distribution(inputs, source, target)
                source_row = scored["named_tokens"]["source"]
                STORE.save("capability", _score_key, {
                    "kind": "clean_score", "score_key": _score_key,
                    "group_id": group["group_id"], "image_id": group["image_id"],
                    "source_concept": source, "target_concept": target,
                    "modality": modality, "prompt_hash": built.prompt_hash,
                    "media_checksum": inputs.media_checksum,
                    "global_argmax_token_id": scored["global_argmax_token_id"],
                    "global_argmax_token": scored["global_argmax_token"],
                    "source_token_id": ENDPOINT["token_ids"][DIGIT_ANSWERS[source]],
                    "source_is_unique_global_top1": source_row["is_unique_maximum"],
                    "source_rank": source_row["rank"],
                    "target_rank": scored["named_tokens"]["target"]["rank"],
                    "target_logprob": scored["named_tokens"]["target"]["logprob"],
                    "scored": scored,
                })
                _computed += 1
            if not STORE.has("capability", _greedy_key):
                generated = greedy_generate(BACKEND, inputs, max_new_tokens=1,
                                            answer=DIGIT_ANSWERS[source])
                STORE.save("capability", _greedy_key, {
                    "kind": "clean_greedy", "score_key": _score_key,
                    "group_id": group["group_id"], "image_id": group["image_id"],
                    "source_concept": source, "target_concept": target,
                    "modality": modality, **generated,
                })
                _computed += 1
            if (_computed + _reused) % 25 == 0:
                print("capability", _computed, "computed", _reused, "reused")

    _cap_units = list(STORE.load_all("capability").values())
    _scores = {row["score_key"]: row for row in _cap_units if row.get("kind") == "clean_score"}
    _greedy = {row["score_key"]: row for row in _cap_units if row.get("kind") == "clean_greedy"}
    _selected, _cap_cells = {}, []
    for source in PAIR_CONCEPTS:
        for modality in DESIGN["modalities"]:
            eligible = [
                row for row in _scores.values()
                if row["source_concept"] == source and row["modality"] == modality
                and row["source_is_unique_global_top1"]
                and row["score_key"] in _greedy
                and _greedy[row["score_key"]]["exact_answer_match"]
                and _greedy[row["score_key"]]["generated_token_ids"][0]
                    == row["global_argmax_token_id"]
            ]
            eligible.sort(key=lambda row: stable_rank(
                f"{row['group_id']}|{modality}", SELECTION_SEED
            ))
            chosen = eligible[:ANALYSIS_IMAGES_PER_CELL]
            key = f"{source}|{modality}"
            _selected[key] = [row["group_id"] for row in chosen]
            _cap_cells.append({
                "source": source, "modality": modality,
                "n_eligible": len(eligible), "n_selected": len(chosen),
                "sufficient": len(chosen) == ANALYSIS_IMAGES_PER_CELL,
                "selected_group_ids": list(_selected[key]),
            })
    CAPABILITY = {
        "endpoint": "unrestricted source digit plus one-token greedy parity",
        "cells": _cap_cells, "selected_group_ids": _selected,
        "all_cells_sufficient": all(row["sufficient"] for row in _cap_cells),
        "selection_seed": SELECTION_SEED,
    }
    CAPABILITY["capability_digest"] = payload_checksum(CAPABILITY)
    STORE.save("metric", "digit_capability_selection", CAPABILITY)
    print("CAPABILITY", CAPABILITY["all_cells_sufficient"])
    for row in _cap_cells:
        print(" ", row)

    if CAPABILITY["all_cells_sufficient"]:
        _score_by_group_modality = {
            (row["group_id"], row["modality"]): row for row in _scores.values()
        }
        _computed = _reused = 0
        for source in PAIR_CONCEPTS:
            target = next(name for name in PAIR_CONCEPTS if name != source)
            for modality in DESIGN["modalities"]:
                for group_id in _selected[f"{source}|{modality}"]:
                    group = _groups_by_id[group_id]
                    built, inputs = make_input(group, modality, source, target)
                    clean = _score_by_group_modality[(group_id, modality)]
                    banks = {
                        "intermediate": selected_bases(source, target),
                        "answer": selected_bases(DIGIT_ANSWERS[source], DIGIT_ANSWERS[target]),
                        "unrelated": selected_bases(*CONTROL_CONCEPTS),
                    }
                    banks["random_intermediate"] = {
                        layer: random_two_direction_basis(
                            basis, seed=RANDOM_CONTROL_SEED_INTERMEDIATE + layer
                        ) for layer, basis in banks["intermediate"].items()
                    }
                    banks["random_answer"] = {
                        layer: random_two_direction_basis(
                            basis, seed=RANDOM_CONTROL_SEED_ANSWER + layer
                        ) for layer, basis in banks["answer"].items()
                    }
                    for arm in ("intermediate", "answer"):
                        for condition in DESIGN["arm_conditions"][arm]:
                            _score_key = confirmation_trial_key(
                                group_id=group_id, modality=modality, arm=arm,
                                condition=condition, kind="score",
                            )
                            if STORE.has("intervention", _score_key):
                                _reused += 1
                            else:
                                if condition.startswith("random"):
                                    bases = banks[f"random_{arm}"]
                                elif condition.startswith("unrelated"):
                                    bases = banks["unrelated"]
                                else:
                                    bases = banks[arm]
                                alpha = 0.0 if condition == "zero" else 1.0
                                with coordinate_swap_band(
                                    BACKEND.blocks, bases, alpha=alpha,
                                    prompt_len=inputs.prompt_len,
                                    position_rule=PRIMARY_POSITION_RULE,
                                    evidence_span=inputs.modality_token_range,
                                    record_coordinates=False,
                                ) as stats:
                                    scored = score_distribution(inputs, source, target)
                                fired = sorted(layer for layer, row in stats.items()
                                               if row["n_forward_passes"] == 1)
                                if fired != list(CONFIRMATION_BAND):
                                    raise RuntimeError(f"hook integrity failure: fired {fired}")
                                record = unrestricted_trial_record(
                                    scored, trial_kind="trial", condition=condition,
                                    arm=arm, band=CONFIRMATION_BAND, alpha=alpha,
                                    modality=modality, readout="property",
                                    source_answer=DIGIT_ANSWERS[source],
                                    target_answer=DIGIT_ANSWERS[target],
                                    source_token_id=ENDPOINT["token_ids"][DIGIT_ANSWERS[source]],
                                    target_token_id=ENDPOINT["token_ids"][DIGIT_ANSWERS[target]],
                                    source_concept=source, target_concept=target,
                                    group_id=group_id, image_id=group["image_id"],
                                    prompt_hash=built.prompt_hash,
                                    media_checksum=inputs.media_checksum,
                                    hook_integrity={
                                        "layers_expected": list(CONFIRMATION_BAND),
                                        "layers_fired_once": fired,
                                        "all_original_prompt_positions": all(
                                            row["positions"] == list(range(inputs.prompt_len))
                                            for row in stats.values()
                                        ),
                                        "intervention_diagnostics": intervention_diagnostics(stats),
                                    },
                                    clean=clean,
                                )
                                record["score_key"] = _score_key
                                STORE.save("intervention", _score_key, record)
                                _computed += 1

                            # Primary treatment, positive control and one zero
                            # receive an independently stored one-token greedy run.
                            wants_greedy = (
                                condition == "swap_alpha1"
                                or (condition == "zero" and arm == "intermediate")
                            )
                            if wants_greedy:
                                _greedy_key = confirmation_trial_key(
                                    group_id=group_id, modality=modality, arm=arm,
                                    condition=condition, kind="greedy",
                                )
                                if STORE.has("intervention", _greedy_key):
                                    _reused += 1
                                else:
                                    if condition == "zero":
                                        bases, alpha = banks[arm], 0.0
                                    else:
                                        bases, alpha = banks[arm], 1.0
                                    with coordinate_swap_band(
                                        BACKEND.blocks, bases, alpha=alpha,
                                        prompt_len=inputs.prompt_len,
                                        position_rule=PRIMARY_POSITION_RULE,
                                        evidence_span=inputs.modality_token_range,
                                        record_coordinates=False,
                                    ) as stats:
                                        generated = greedy_generate(
                                            BACKEND, inputs, max_new_tokens=1,
                                            answer=DIGIT_ANSWERS[target],
                                        )
                                    STORE.save("intervention", _greedy_key, {
                                        "kind": "trial_greedy", "score_key": _score_key,
                                        "group_id": group_id, "image_id": group["image_id"],
                                        "source_concept": source, "target_concept": target,
                                        "modality": modality, "arm": arm,
                                        "condition": condition, **generated,
                                    })
                                    _computed += 1
                            if _computed % 25 == 0 and _computed:
                                print("trials", _computed, "computed", _reused, "reused")
        CAUSAL_STAGE_RAN = True
        print("confirmation complete", {"computed": _computed, "reused": _reused})
    else:
        print("CAPABILITY NO-GO: no intervention pass was spent and no sample was replaced.")
else:
    print("skipped: GPU stage is not enabled")
'''
)

markdown("## 6. Final report from saved units only")
code(
    r'''
from jlens.mmpilot.digit_reasoning_confirmation import (
    DIGIT_CONFIRMATION_REPORT_NAME, aggregate_confirmation,
    confirmation_report, confirmation_verdict, format_confirmation_verdict,
)
from jlens.mmpilot.store import payload_checksum

REPORT = None
AGGREGATION = None
VERDICT = None

def load_valid_units(root, stage, expected_fingerprint_digest):
    payloads, invalid = [], []
    for path in sorted((Path(root) / "units" / stage).glob("*.json")):
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
            payload = stored["payload"]
            if (
                stored.get("unit_checksum") != payload_checksum(payload)
                or stored.get("fingerprint_digest") != expected_fingerprint_digest
            ):
                invalid.append(str(path))
            else:
                payloads.append(payload)
        except Exception:
            invalid.append(str(path))
    if invalid:
        raise RuntimeError(f"checksum-invalid/torn units: {invalid[:5]}")
    return payloads

_report_root = Path(REPORT_RUN_DIR) if REPORT_RUN_DIR else RUN_DIR
if RUN_FINAL_REPORT_CPU:
    if _report_root is None:
        raise RuntimeError("set REPORT_RUN_DIR for a report-only session")
    _fingerprint_payload = json.loads(
        (_report_root / "fingerprint.json").read_text(encoding="utf-8")
    )
    _fingerprint_payload.pop("written_utc", None)
    _expected_unit_fingerprint = payload_checksum(_fingerprint_payload)
    _cap_units = load_valid_units(
        _report_root, "capability", _expected_unit_fingerprint
    )
    _metric_units = load_valid_units(
        _report_root, "metric", _expected_unit_fingerprint
    )
    _intervention_units = load_valid_units(
        _report_root, "intervention", _expected_unit_fingerprint
    )
    if not CAPABILITY.get("cells"):
        CAPABILITY = next(
            (row for row in _metric_units if row.get("capability_digest")),
            {"all_cells_sufficient": False, "cells": []},
        )
    _protocol_unit = next(
        (row for row in _metric_units if row.get("kind") == "digit_confirmation_protocol"),
        {},
    )
    if ENDPOINT is None:
        ENDPOINT = _protocol_unit.get("endpoint")
    if FINGERPRINT_CONFIG is None:
        FINGERPRINT_CONFIG = _protocol_unit.get("fingerprint")
    if POPULATION is None:
        POPULATION = _protocol_unit.get("population")
    if EXCLUSION is None:
        EXCLUSION = _protocol_unit.get("exclusion")
    _scores = {
        row["score_key"]: dict(row)
        for row in _intervention_units
        if row.get("trial_kind") == "trial"
    }
    _greedy = {
        row["score_key"]: row
        for row in _intervention_units
        if row.get("kind") == "trial_greedy"
    }
    for key, row in _scores.items():
        if key in _greedy:
            generated = _greedy[key]
            row["greedy_generated_token_ids"] = generated["generated_token_ids"]
            row["greedy_exact_target_match"] = generated["exact_answer_match"]
            row["greedy_first_token_equals_global_argmax"] = (
                generated["generated_token_ids"][0] == row["global_argmax_token_id"]
            )
        elif row["condition"] == "swap_alpha1":
            row["greedy_first_token_equals_global_argmax"] = False
    AGGREGATION = (
        aggregate_confirmation(list(_scores.values()), thresholds=THRESHOLDS)
        if _scores else None
    )
    VERDICT = confirmation_verdict(
        AGGREGATION, capability=CAPABILITY, thresholds=THRESHOLDS,
        causal_stage_ran=bool(_scores),
    )
    print(format_confirmation_verdict(VERDICT))
    if AGGREGATION:
        print()
        print("DIRECTION × MODALITY PRIMARY CELLS")
        for row in AGGREGATION["cells"]:
            if row["arm"] == "intermediate" and row["condition"] == "swap_alpha1":
                print(f"  {row['source']}->{row['target']} {row['modality']:<12} "
                      f"{row['successes']}/{row['n']} = {row['success_rate']:.3f}")
        print("PAIRED CONTROLS")
        for name, row in AGGREGATION["paired_primary_vs_controls"].items():
            print(f"  {name:<18} diff={row['mean_difference']:+.3f} "
                  f"CI={row['ci95']} Holm-p={row['holm_adjusted_pvalue']:.4g} "
                  f"pass={row['passed']}")
        print("PAIRED DIRECT-ANSWER CONTROLS")
        for name, row in AGGREGATION["paired_direct_answer_vs_controls"].items():
            print(f"  {name:<18} diff={row['mean_difference']:+.3f} "
                  f"CI={row['ci95']} Holm-p={row['holm_adjusted_pvalue']:.4g} "
                  f"pass={row['passed']}")
    _resume = {
        "run_dir": str(_report_root),
        "n_capability_units": len(_cap_units),
        "n_intervention_units": len(_intervention_units),
        "invalid_units": [],
        "atomic_unit_resume": True,
    }
    REPORT = confirmation_report(
        design=DESIGN, endpoint=ENDPOINT or {"missing": True},
        fingerprint=FINGERPRINT_CONFIG or {"missing": True},
        population=POPULATION or {"missing": True},
        exclusion=EXCLUSION or {"missing": True},
        capability=CAPABILITY, aggregation=AGGREGATION, verdict=VERDICT,
        budget=BUDGET, resume=_resume,
    )
    (_report_root / DIGIT_CONFIRMATION_REPORT_NAME).write_text(
        json.dumps(REPORT, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    print("report", _report_root / DIGIT_CONFIRMATION_REPORT_NAME)
    print("checksum", REPORT["report_checksum"])
else:
    print("skipped: RUN_FINAL_REPORT_CPU is False")
'''
)

markdown(
    r"""
## 7. Interpretation boundary

Only `DIGIT_REASONING_THREE_MODALITY_CAUSAL_GO` licenses the strong claim printed
in the design. Every direction and modality must pass, the α=1 direct-answer arm
must validate the endpoint against zero and matched random controls, and the
paired identity exchange must beat all three α=1 controls after Holm correction
with a bootstrap interval above zero.

Any other verdict is reported as measured. No alpha, endpoint, layer, sample,
threshold or concept is changed after this run. The completed α=2 result remains
an immutable result about extrapolation and is never pooled here. Image and audio are input
modalities; Gemma's output is text. SpokenCOCO is spoken linguistic captions,
not environmental sound.
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
