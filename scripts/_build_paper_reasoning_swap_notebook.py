"""Generate the real paper-style multimodal coordinate-swap Colab notebook."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "notebooks" / "multimodal_jspace_anthropic_reasoning_swap_colab.ipynb"
CELLS: list[tuple[str, str]] = []


def markdown(value: str) -> None:
    CELLS.append(("markdown", value.strip("\n")))


def code(value: str) -> None:
    CELLS.append(("code", value.strip("\n")))


markdown(
    r"""
# Paper-style hidden-intermediate J-lens swap — text, image, spoken audio

This is the apples-to-apples causal test from Anthropic's *Verbalizable
Representations Form a Global Workspace in Language Models*, adapted to the
confirmed Gemma 4 E4B J-lenses and SpokenCOCO.

For the same hidden-animal evidence, it compares:

1. **intermediate swap:** exchange the source and counterfactual animal's two
   J-lens coordinates;
2. **answer swap:** exchange the corresponding leg-count answer coordinates.

Both arms use the exact operation

`h' = h + V (sigma(pinv(V) h) - pinv(V) h)`

at **every original prompt position**. Each forward pass performs this exchange
at exactly **one independently confirmed physical layer**. This matters because
the exchange is an involution: repeating it at several layers can undo an
earlier exchange. Each tested layer reads and exchanges its own current
coordinates once. The target entity and answer are absent from the
model-visible prompt; candidates exist only in the external teacher-forced
scorer.

The old source-derived steering result and the bespoke native-readout
`NOT_CONVERGED` gate are not used. The primary alpha=1 verdict asks whether the
intermediate-coordinate intervention first changes both the hidden identity
and its downstream answer at a shallower tested layer than the direct
answer-coordinate intervention. Alpha=2 is a separately labelled sensitivity
analysis, not a replacement for the primary test.

## Honest scope

Anthropic reports 25 evenly spaced layer samples. Gemma currently has
independently confirmed J-lenses at physical layers 32, 35, 38 and 40. This
notebook tests layers 32, 35, 38 and 40 independently. It does **not** pretend
the unmeasured physical layers between them were tested, and it localizes onset
only among those four points.

## Safe execution

Opening the notebook spends nothing. Set the three switches in section 2 and
run top-to-bottom on an L4 or A100. The 125,198-group evidence join is loaded
from one pinned cache and never rebuilt. Every clean score and every
intervention/readout condition is checksum-valid and atomically saved to Drive.
Clean behavior is screened first on 24 fresh candidates per concept/modality;
only a predeclared stable sample of up to eight correctly solved images per
cell enters the expensive causal stage. If any cell has fewer than four, the
causal stage stops before spending those passes.
Rerunning the same notebook resumes; changing any scientific input refuses the
old directory instead of mixing results.
"""
)

markdown("## 1. Colab bootstrap")
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
         "transformers==5.13.1", "accelerate", "soundfile"],
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

markdown("## 2. Configuration — the only switches")
code(
    r'''
RUN_REAL_PAPER_SWAP = False
CONFIRM_MODEL_LOAD = False
CONFIRM_PASS_BUDGET = False

# Frozen v2 scientific design.  The completed v1 run is read only and its
# images are excluded below.
SAMPLED_LAYERS = (32, 35, 38, 40)
PAIR_CONCEPTS = ("bird", "cat")       # 2 legs versus 4 legs
CONTROL_CONCEPTS = ("zebra", "giraffe")
POPULATION_CONCEPTS = (*PAIR_CONCEPTS, *CONTROL_CONCEPTS)
CANDIDATE_IMAGES_PER_CONCEPT = 24
MAX_ANALYSIS_IMAGES_PER_CELL = 8
MIN_ANALYSIS_IMAGES_PER_CELL = 4
MODALITIES = ("text", "image", "spoken_audio")
POSITION_RULE = "all_prompt_positions"
CONDITIONS = (
    "swap_alpha1", "swap_alpha2", "zero",
    "random_alpha1", "random_alpha2",
    "unrelated_alpha1", "unrelated_alpha2",
    "position_descriptive",
)
SELECTION_SEED = "anthropic-hidden-animal-swap-gemma-v2-independent"

MODEL_REPO_ID = "google/gemma-4-E4B-it"
MODEL_REVISION = "fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd"
TRANSFORMERS_VERSION_EXPECTED = "5.13.1"
EXPECT_N_LAYERS, EXPECT_D_MODEL, EXPECT_VOCAB = 42, 2560, 262144
AUDIO_PROTOCOL_FINGERPRINT = (
    "sha256:9ad8bcc9420a7983f6e3b75d5d7080c0e2fcf0a94a76431917fcde73ba777920"
)

RUNS_ROOT = Path("/content/drive/MyDrive/jacobian-lens-gemma/runs")
EXPANDED_MANIFEST_CACHE = Path(
    "/content/drive/MyDrive/jacobian-lens-gemma/runs/"
    "mml32_l32_followup_20260808T182717/expanded_manifest.json"
)
PRIOR_EXCLUSION_SET = Path(
    "/content/drive/MyDrive/datasets/cstf_spokencoco_derived/"
    "jlens_l32_resolution_prep_v1/"
    "prep_020ebbe6f832aece5ece6cb8bee994ca/exclusion_set.json"
)
COMPLETED_V1_RUN_DIR = Path(
    "/content/drive/MyDrive/jacobian-lens-gemma/runs/"
    "mmpaper_real_24be1d028bf1"
)
COMPLETED_V1_REPORT_CHECKSUM = (
    "sha256:a60f3336bf8acdc98dc1a434698104eaa98b3192c44f43fa5ab21212826ae397"
)
EXTENSION_RUN_DIR = Path(
    "/content/drive/MyDrive/jacobian-lens-gemma/runs/rgext_real_c18f03f06e7b"
)
PUBLISHED_LENS_DIR = Path(
    "/content/drive/MyDrive/jacobian-lens-gemma/runs/"
    "rgcalib_real_7e3736b4de8f/artifacts/published"
)
LATE_LENS_PINS = {
    35: ("lens.layer35.scale100.validated.pt", "sha256:64fb02d718ac48adc1bced99e2eff3c2215052ba144d5dedac05f17936a96ed1"),
    38: ("lens.layer38.scale100.validated.pt", "sha256:c8508fbf2b916e5d9aaeb8711a30f76414ee16478c5f6cc321e57e2fe846d1c0"),
    40: ("lens.layer40.scale100.validated.pt", "sha256:8a90f67eeb9bb5db14e6715b8bc516a899da1c3210d0662ec7fa177b5409f7d7"),
}

from jlens.mmpilot.paper_reasoning_swap import PaperSwapV2Thresholds
THRESHOLDS = PaperSwapV2Thresholds(
    min_images=4,
    min_target_flip_rate=0.50,
    min_joint_intermediate_rate=0.50,
    max_answer_identity_flip_rate=0.25,
    min_margin_gain=0.0,
    control_margin=0.0,
)

print("RUN_REAL_PAPER_SWAP", RUN_REAL_PAPER_SWAP)
print("CONFIRM_MODEL_LOAD ", CONFIRM_MODEL_LOAD)
print("CONFIRM_PASS_BUDGET", CONFIRM_PASS_BUDGET)
print("layers", SAMPLED_LAYERS, "conditions", CONDITIONS)
print("candidate images/concept", CANDIDATE_IMAGES_PER_CONCEPT)
print("analysis images/cell", MAX_ANALYSIS_IMAGES_PER_CELL)
print("threshold digest", THRESHOLDS.digest)
'''
)

markdown("## 3. Mount Drive and fail fast on every pin")
code(
    r'''
if RUN_REAL_PAPER_SWAP:
    if IN_COLAB:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
    missing = [
        str(path) for path in (
            EXPANDED_MANIFEST_CACHE, PRIOR_EXCLUSION_SET,
            EXTENSION_RUN_DIR, PUBLISHED_LENS_DIR, COMPLETED_V1_RUN_DIR,
        )
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("configured artifact(s) missing:\n  " + "\n  ".join(missing))
    if not __import__("torch").cuda.is_available():
        raise RuntimeError("real execution requires a GPU runtime; use L4 or A100")
else:
    print("SAFE DEFAULT: no Drive mounted, no model loaded, no forward pass run.")
'''
)

markdown("## 4. Load the cached evidence join and freeze the hidden population")
code(
    r'''
import hashlib, json

POPULATION = None
MANIFEST_FILE_CHECKSUM = None
EXCLUSION_FILE_CHECKSUM = None
if RUN_REAL_PAPER_SWAP:
    from jlens.mmpilot.evidence import EvidenceConfig
    from jlens.mmpilot.paper_reasoning_swap import hidden_animal_population
    from jlens.mmpilot.store import payload_checksum

    raw_bytes = EXPANDED_MANIFEST_CACHE.read_bytes()
    MANIFEST_FILE_CHECKSUM = "sha256:" + hashlib.sha256(raw_bytes).hexdigest()
    payload = json.loads(raw_bytes)
    if payload.get("n_groups") != 125198 or not isinstance(payload.get("groups"), list):
        raise RuntimeError(
            f"expected the pinned 125,198-group cache, got n_groups={payload.get('n_groups')}"
        )
    exclusion_bytes = PRIOR_EXCLUSION_SET.read_bytes()
    EXCLUSION_FILE_CHECKSUM = "sha256:" + hashlib.sha256(exclusion_bytes).hexdigest()
    exclusion = json.loads(exclusion_bytes)
    excluded_images = {str(value) for value in exclusion.get("image_ids", [])}
    excluded_groups = {str(value) for value in exclusion.get("group_ids", [])}

    # The completed v1 population was examined while designing v2.  Its pair
    # images are therefore spent and must not enter this confirmatory run.
    v1_report_path = COMPLETED_V1_RUN_DIR / "paper_reasoning_swap_report.json"
    v1_report = json.loads(v1_report_path.read_text(encoding="utf-8"))
    if v1_report.get("report_checksum") != COMPLETED_V1_REPORT_CHECKSUM:
        raise RuntimeError(
            "completed v1 paper report checksum mismatch; refusing to guess "
            "which population was previously examined"
        )
    v1_capability_files = sorted(
        (COMPLETED_V1_RUN_DIR / "units" / "capability").glob("*.json")
    )
    if not v1_capability_files:
        raise RuntimeError("completed v1 run has no capability units")
    v1_images = set()
    v1_groups = set()
    for path in v1_capability_files:
        row = json.loads(path.read_text(encoding="utf-8"))
        payload_row = row.get("payload") if isinstance(row.get("payload"), dict) else row
        if payload_row.get("image_id"):
            v1_images.add(str(payload_row["image_id"]))
        if payload_row.get("group_id"):
            v1_groups.add(str(payload_row["group_id"]))
    if len(v1_images) != 12:
        raise RuntimeError(
            f"expected 12 spent pair images in completed v1, found {len(v1_images)}"
        )
    excluded_images.update(v1_images)
    excluded_groups.update(v1_groups)
    EXCLUSION_FILE_CHECKSUM = payload_checksum({
        "base_exclusion_file_checksum": EXCLUSION_FILE_CHECKSUM,
        "completed_v1_report_checksum": COMPLETED_V1_REPORT_CHECKSUM,
        "completed_v1_image_ids": sorted(v1_images),
        "completed_v1_group_ids": sorted(v1_groups),
    })
    eligible_groups = [
        row for row in payload["groups"]
        if str(row.get("image_id")) not in excluded_images
        and str(row.get("group_id")) not in excluded_groups
    ]
    evidence_config = EvidenceConfig(
        lexicon={name: (name,) for name in POPULATION_CONCEPTS},
        coco_categories={name: (name,) for name in POPULATION_CONCEPTS},
        require_visual_evidence=True,
        require_caption_evidence=False,
    )
    POPULATION = hidden_animal_population(
        eligible_groups,
        concept_names=POPULATION_CONCEPTS,
        evidence_config=evidence_config,
        images_per_concept=CANDIDATE_IMAGES_PER_CONCEPT,
        seed=SELECTION_SEED,
    )
    selected_images = {str(row["image_id"]) for row in POPULATION["groups"]}
    selected_groups = {str(row["group_id"]) for row in POPULATION["groups"]}
    if selected_images & excluded_images or selected_groups & excluded_groups:
        raise RuntimeError("the new population overlaps the prior exclusion set")
    del payload, raw_bytes, exclusion_bytes, eligible_groups
    print("manifest checksum", MANIFEST_FILE_CHECKSUM)
    print("prior exclusion checksum", EXCLUSION_FILE_CHECKSUM)
    print("prior excluded images/groups", len(excluded_images), len(excluded_groups))
    print("completed v1 pair images excluded", len(v1_images))
    print("population digest", POPULATION["population_digest"])
    print("groups/images", POPULATION["n_groups"], POPULATION["n_distinct_images"])
    print("coverage")
    for name, row in POPULATION["coverage"].items():
        print(f"  {name:10s} eligible={row['eligible_distinct_images']:4d} selected={row['selected_distinct_images']}")
    print("leakage rejections", POPULATION["rejections"])
    assert POPULATION["one_group_per_image"]
else:
    print("skipped")
'''
)

markdown("## 5. Resolve and verify the four independently confirmed lenses")
code(
    r'''
LENS_SETS = {}
LENS_CHECKSUMS = {}
if RUN_REAL_PAPER_SWAP:
    from jlens.mmpilot.l32_followup import (
        discover_published_l32_lens, l32_expectations, validate_discovered_lens,
    )
    from jlens.mmpilot.published_lens import (
        PublishedLensExpectations, PublishedLensSpec, combined_lens_checksum,
        format_lens_report, load_published_lenses,
    )

    discovered = discover_published_l32_lens(
        EXTENSION_RUN_DIR, layer=32, expected_scale=250
    )
    l32_expect = l32_expectations(
        model_repo_id=MODEL_REPO_ID, model_revision=MODEL_REVISION,
        d_model=EXPECT_D_MODEL, layer=32, scale=250,
    )
    validate_discovered_lens(discovered, l32_expect)
    LENS_SETS[32] = load_published_lenses([discovered.spec()], l32_expect)

    late_specs = [
        PublishedLensSpec(
            layer=layer, path=str(PUBLISHED_LENS_DIR / filename), expect_sha256=checksum
        )
        for layer, (filename, checksum) in sorted(LATE_LENS_PINS.items())
    ]
    late_expect = PublishedLensExpectations(
        model_repo_id=MODEL_REPO_ID, model_revision=MODEL_REVISION,
        scale_point=100, d_model=EXPECT_D_MODEL,
    )
    late = load_published_lenses(late_specs, late_expect)
    for layer in late.layers:
        LENS_SETS[layer] = late
    LENS_CHECKSUMS = {
        32: LENS_SETS[32].checksums[32],
        **{layer: late.checksums[layer] for layer in late.layers},
    }
    if tuple(sorted(LENS_CHECKSUMS)) != SAMPLED_LAYERS:
        raise RuntimeError(f"confirmed lens grid mismatch: {sorted(LENS_CHECKSUMS)}")
    COMBINED_LENS_CHECKSUM = combined_lens_checksum(LENS_CHECKSUMS)
    print("confirmed layers", sorted(LENS_CHECKSUMS))
    for layer in SAMPLED_LAYERS:
        print(f"  L{layer}: {LENS_CHECKSUMS[layer]}")
    print("combined", COMBINED_LENS_CHECKSUM)
else:
    COMBINED_LENS_CHECKSUM = None
    print("skipped")
'''
)

markdown("## 6. Freeze the layer ranges, directed pair, prompts, and budget")
code(
    r'''
BAND_RECORD = None
DIRECTED_PAIRS = None
if RUN_REAL_PAPER_SWAP:
    from jlens.mmpilot.paper_reasoning_swap import independent_layer_record
    from jlens.mmpilot.prompt_protocol import (
        HIDDEN_ANIMAL_LEGS, OPEN_ANIMAL_IDENTIFICATION,
        assert_property_contrast,
    )
    BAND_RECORD = independent_layer_record(
        SAMPLED_LAYERS, validated_layers=tuple(LENS_CHECKSUMS)
    )
    DIRECTED_PAIRS = []
    for source, target in (PAIR_CONCEPTS, tuple(reversed(PAIR_CONCEPTS))):
        contrast = assert_property_contrast(source, target)
        DIRECTED_PAIRS.append({
            "source": source, "target": target,
            "source_property_value": contrast["source_value"],
            "target_property_value": contrast["target_value"],
        })
    print("independent single-layer interventions", BAND_RECORD["bands"])
    print("repeated exchange forbidden", BAND_RECORD["repeated_exchange_forbidden"])
    print("directed pairs", DIRECTED_PAIRS)
    print("identity protocol", OPEN_ANIMAL_IDENTIFICATION)
    print("property protocol", HIDDEN_ANIMAL_LEGS)
    n_candidate_images = len(PAIR_CONCEPTS) * CANDIDATE_IMAGES_PER_CONCEPT
    clean_passes = n_candidate_images * len(MODALITIES) * 2 * 2
    n_selected_cells = (
        len(PAIR_CONCEPTS) * len(MODALITIES) * MAX_ANALYSIS_IMAGES_PER_CELL
    )
    intervention_passes = (
        n_selected_cells * len(BAND_RECORD["bands"])
        * 2 * len(CONDITIONS) * 2 * 2
    )
    TOTAL_PASSES = clean_passes + intervention_passes
    print("\nPASS BUDGET")
    print("  clean candidate passes       ", f"{clean_passes:,}")
    print("  intervention candidate passes", f"{intervention_passes:,}")
    print("  TOTAL                        ", f"{TOTAL_PASSES:,}")
    print("  expected L4 wall time: roughly 2–5 hours; A100 is usually faster")
    print("  capability screening runs first; causal work is skipped if any")
    print("  concept/modality cell has fewer than", MIN_ANALYSIS_IMAGES_PER_CELL, "clean images")
    if not (CONFIRM_MODEL_LOAD and CONFIRM_PASS_BUDGET):
        print("\nBLOCKED: set both confirmation switches in section 2.")
else:
    TOTAL_PASSES = 0
    print("skipped")
'''
)

markdown("## 7. Load Gemma and verify native spoken audio")
code(
    r'''
BACKEND = None
MODEL_REVISION_USED = PROCESSOR_REVISION_USED = None
if RUN_REAL_PAPER_SWAP and CONFIRM_MODEL_LOAD and CONFIRM_PASS_BUDGET:
    import getpass, torch
    from jlens.mmpilot.real_backend import build_real_backend
    from jlens.mmpilot.tri_modal import assert_audio_protocol

    if not os.environ.get("HF_TOKEN"):
        token = getpass.getpass("HF_TOKEN (input hidden): ").strip()
        if not token:
            raise RuntimeError("HF_TOKEN is required for the gated checkpoint")
        os.environ["HF_TOKEN"] = token
    bundle = build_real_backend(
        MODEL_REPO_ID, revision=MODEL_REVISION, token=os.environ["HF_TOKEN"],
        device="cuda", allow_model_load=True, resolve_audio=True,
        expect_n_layers=EXPECT_N_LAYERS, expect_d_model=EXPECT_D_MODEL,
        expect_vocab_size=EXPECT_VOCAB,
    )
    if bundle.audio_interface is None:
        raise RuntimeError("native spoken audio did not resolve: " + bundle.audio_blocked_reason)
    audio_record = assert_audio_protocol(
        bundle.audio_interface, expected_fingerprint=AUDIO_PROTOCOL_FINGERPRINT
    )
    BACKEND = bundle.backend
    MODEL_REVISION_USED = bundle.model_revision
    PROCESSOR_REVISION_USED = bundle.processor_revision
    print("model", MODEL_REVISION_USED, "device", bundle.device)
    print("audio", audio_record["protocol_fingerprint"])
else:
    print("skipped: real execution and both confirmations are required")
'''
)

markdown("## 8. Build raw J-lens atoms, exact swap bases, and open the resumable run")
code(
    r'''
STORE = None
TOKEN_VECTORS = {}
BASES = {}
if BACKEND is not None:
    import torch
    from jlens.mmpilot.coordinate_swap import (
        METHOD_VERSION,
        build_swap_basis_from_vectors,
        random_two_direction_basis,
        resolve_concept_token,
    )
    from jlens.mmpilot.paper_reasoning_swap import PAPER_REASONING_SWAP_V2_VERSION
    from jlens.mmpilot.store import RunFingerprint, UnitStore, payload_checksum

    if METHOD_VERSION != "jlens.mmpilot.coordinate_swap.v1":
        raise RuntimeError(f"coordinate-swap algebra drifted: {METHOD_VERSION}")

    token_names = (*POPULATION_CONCEPTS, "two", "four")
    TOKENS = {name: resolve_concept_token(BACKEND.encode_candidate, name) for name in token_names}
    unembedding = BACKEND.unembedding_weight()
    unembedding_rows = {
        name: unembedding[token.token_id].detach().float().cpu()
        for name, token in TOKENS.items()
    }
    for layer in SAMPLED_LAYERS:
        jacobian = LENS_SETS[layer].lens.jacobians[layer].detach().float().cpu()
        TOKEN_VECTORS[layer] = {
            name: row @ jacobian for name, row in unembedding_rows.items()
        }
        del jacobian
    del unembedding, unembedding_rows
    print(
        "selected lens vectors built",
        {layer: sorted(rows) for layer, rows in TOKEN_VECTORS.items()},
    )

    def selected_bases(layers, source_name, target_name):
        return {
            layer: build_swap_basis_from_vectors(
                TOKEN_VECTORS[layer][source_name],
                TOKEN_VECTORS[layer][target_name],
                layer=layer,
                source=TOKENS[source_name],
                target=TOKENS[target_name],
            )
            for layer in layers
        }

    for pair in DIRECTED_PAIRS:
        source, target = pair["source"], pair["target"]
        source_answer = "two" if pair["source_property_value"] == 2 else "four"
        target_answer = "two" if pair["target_property_value"] == 2 else "four"
        pair_key = f"{source}->{target}"
        BASES[pair_key] = {}
        for band in BAND_RECORD["bands"]:
            band = tuple(band)
            entity = selected_bases(band, source, target)
            answer = selected_bases(band, source_answer, target_answer)
            unrelated = selected_bases(
                band, CONTROL_CONCEPTS[0], CONTROL_CONCEPTS[1]
            )
            BASES[pair_key][band[0]] = {
                "intermediate": entity,
                "answer": answer,
                "unrelated": unrelated,
                "random_intermediate": {
                    layer: random_two_direction_basis(basis, seed=20260810 + layer + band[0])
                    for layer, basis in entity.items()
                },
                "random_answer": {
                    layer: random_two_direction_basis(basis, seed=20261810 + layer + band[0])
                    for layer, basis in answer.items()
                },
            }

    fingerprint = RunFingerprint(
        mode="paper_reasoning_coordinate_swap",
        model_repo_id=MODEL_REPO_ID,
        model_revision=MODEL_REVISION_USED,
        processor_revision=PROCESSOR_REVISION_USED,
        layers=SAMPLED_LAYERS,
        lens_checksum=COMBINED_LENS_CHECKSUM,
        manifest_checksum=MANIFEST_FILE_CHECKSUM,
        split_id=SELECTION_SEED,
        intervention_config={
            "method": PAPER_REASONING_SWAP_V2_VERSION,
            "coordinate_algebra": METHOD_VERSION,
            "family": "anthropic_independent_single_layer_coordinate_swap",
            "bands": BAND_RECORD["bands"],
            "position_rule": POSITION_RULE,
            "conditions": list(CONDITIONS),
            "arms": ["intermediate", "answer"],
            "primary_alpha": 1.0,
            "sensitivity_alpha": 2.0,
            "one_exchange_per_forward_pass": True,
            "repeated_exchange_forbidden": True,
            "position_descriptive_is_blocking": False,
        },
        selection_config={
            "population_digest": POPULATION["population_digest"],
            "prior_exclusion_checksum": EXCLUSION_FILE_CHECKSUM,
            "population_concepts": list(POPULATION_CONCEPTS),
            "pair_concepts": list(PAIR_CONCEPTS),
            "control_concepts": list(CONTROL_CONCEPTS),
            "candidate_images_per_concept": CANDIDATE_IMAGES_PER_CONCEPT,
            "max_analysis_images_per_cell": MAX_ANALYSIS_IMAGES_PER_CELL,
            "min_analysis_images_per_cell": MIN_ANALYSIS_IMAGES_PER_CELL,
            "capability_selection_seed": SELECTION_SEED,
        },
        extra={
            "study": PAPER_REASONING_SWAP_V2_VERSION,
            "audio_protocol_fingerprint": AUDIO_PROTOCOL_FINGERPRINT,
            "per_layer_lens_checksums": {str(k): v for k, v in sorted(LENS_CHECKSUMS.items())},
            "band_digest": BAND_RECORD["digest"],
            "threshold_digest": THRESHOLDS.digest,
            "completed_v1_report_checksum": COMPLETED_V1_REPORT_CHECKSUM,
            "prompt_protocols": ["mmpilot.open_animal_identification.v1", "mmpilot.hidden_animal_legs.v1"],
            "target_and_candidates_absent_from_prompt": True,
        },
    )
    RUN_DIR = RUNS_ROOT / f"mmpaper2_real_{fingerprint.digest.split(':')[1][:12]}"
    STORE = UnitStore(RUN_DIR, fingerprint)
    print("run directory", RUN_DIR)
    print("run state    ", STORE.open())
    print("fingerprint  ", fingerprint.digest)
    print("RESUME: every completed clean/readout condition is reused after checksum validation")
else:
    RUN_DIR = None
    print("skipped")
'''
)

markdown("## 9. Input helper — transcript remains audit-only")
code(
    r'''
if STORE is not None:
    from jlens.mmpilot.media_io import RetryJournal, drive_media_loaders
    from jlens.mmpilot.prompt_protocol import (
        Evidence, HIDDEN_ANIMAL_LEGS, OPEN_ANIMAL_IDENTIFICATION,
        build_backend_inputs, build_protocol_prompt, concept_spec,
        leg_count_surfaces, resolve_leg_count,
    )
    MEDIA_JOURNAL = RetryJournal()
    MEDIA = drive_media_loaders(journal=MEDIA_JOURNAL)
    IDENTITY_CANDIDATES = PAIR_CONCEPTS
    PROPERTY_CANDIDATES = ("two", "four")
    CANDIDATE_IDS = {
        "identity": {name: BACKEND.encode_candidate(f" {name}") for name in IDENTITY_CANDIDATES},
        "property": {name: BACKEND.encode_candidate(f" {name}") for name in PROPERTY_CANDIDATES},
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
        # Text intentionally carries the caption as its evidence.  The offline
        # transcript-leakage guard applies only to non-text modalities, where
        # that caption must remain absent from the model-visible inputs.
        offline_transcript = group["caption"] if modality != "text" else None
        inputs = build_backend_inputs(
            BACKEND, built, transcript=offline_transcript
        )
        return built, inputs

    print("candidate ids", CANDIDATE_IDS)
    if not all(len(ids) == 1 for mapping in CANDIDATE_IDS.values() for ids in mapping.values()):
        print("NOTE: complete sequence scoring is supported, but swap concepts themselves were separately required single-token.")
else:
    print("skipped")
'''
)

markdown("## 10. Clean behavioral gate — atomically saved per image/modality/readout")
code(
    r'''
CAUSAL_SELECTION = None
if STORE is not None:
    from jlens.mmpilot.capability import prediction_and_margin, score_candidate_sequences
    from jlens.mmpilot.paper_reasoning_swap import select_capability_eligible_samples
    from jlens.mmpilot.store import safe_key

    source_groups = [row for row in POPULATION["groups"] if row["concept"] in PAIR_CONCEPTS]
    computed = reused = 0
    for group in source_groups:
        source = group["concept"]
        target = next(name for name in PAIR_CONCEPTS if name != source)
        for modality in MODALITIES:
            for readout in ("identity", "property"):
                key = safe_key("paper-clean", group["group_id"], modality, readout)
                if STORE.has("capability", key):
                    reused += 1
                    continue
                built, inputs = make_input(group, modality, source, target, readout)
                scores = score_candidate_sequences(BACKEND, inputs, CANDIDATE_IDS[readout])
                source_answer = (
                    source if readout == "identity"
                    else leg_count_surfaces(resolve_leg_count(source))[0]
                )
                verdict = prediction_and_margin(scores, source_answer)
                STORE.save("capability", key, {
                    "group_id": group["group_id"], "image_id": group["image_id"],
                    "source": source, "target": target, "modality": modality,
                    "readout": readout, "source_answer": source_answer,
                    "prediction": verdict["prediction"], "correct": verdict["correct"],
                    "scores": scores, "prompt": built.to_dict(),
                    "prompt_len": inputs.prompt_len,
                })
                computed += 1
                print(f"clean {computed:3d} computed  {reused:3d} reused  {source}:{modality}:{readout}")
    clean_units = list(STORE.load_all("capability").values())
    for source in PAIR_CONCEPTS:
        for modality in MODALITIES:
            rows = [r for r in clean_units if r["source"] == source and r["modality"] == modality]
            print(source, modality, {kind: sum(r["correct"] for r in rows if r["readout"] == kind) for kind in ("identity", "property")})
    CAUSAL_SELECTION = select_capability_eligible_samples(
        clean_units,
        concepts=PAIR_CONCEPTS,
        modalities=MODALITIES,
        max_images_per_cell=MAX_ANALYSIS_IMAGES_PER_CELL,
        min_images_per_cell=MIN_ANALYSIS_IMAGES_PER_CELL,
        seed=SELECTION_SEED,
    )
    STORE.save("metric", "paper_v2_capability_selection", CAUSAL_SELECTION)
    print("\nCAPABILITY-FROZEN CAUSAL POPULATION")
    for row in CAUSAL_SELECTION["cells"]:
        print(
            f"  {row['concept']:6s} {row['modality']:12s} "
            f"eligible={row['n_eligible']:2d} selected={row['n_selected']:2d} "
            f"sufficient={row['sufficient']}"
        )
    print("  all cells sufficient", CAUSAL_SELECTION["all_cells_sufficient"])
    print("  selection digest    ", CAUSAL_SELECTION["digest"])
    print("  swap results consulted", CAUSAL_SELECTION["swap_results_consulted"])
    if not CAUSAL_SELECTION["all_cells_sufficient"]:
        print("CAUSAL STAGE STOPPED BEFORE INTERVENTIONS: clean capability was insufficient.")
else:
    print("skipped")
'''
)

markdown("## 11. Exact intermediate and answer swaps — smart-save after every condition")
code(
    r'''
if STORE is not None and CAUSAL_SELECTION["all_cells_sufficient"]:
    from jlens.mmpilot.coordinate_swap import run_swap_condition
    from jlens.mmpilot.store import safe_key

    clean_by_key = {
        (row["group_id"], row["modality"], row["readout"]): row
        for row in STORE.load_all("capability").values()
    }
    selected_by_cell = {
        key: set(group_ids)
        for key, group_ids in CAUSAL_SELECTION["selected_group_ids"].items()
    }
    computed = reused = 0
    source_groups = [row for row in POPULATION["groups"] if row["concept"] in PAIR_CONCEPTS]
    for group in source_groups:
        source = group["concept"]
        target = next(name for name in PAIR_CONCEPTS if name != source)
        pair_key = f"{source}->{target}"
        source_property = leg_count_surfaces(resolve_leg_count(source))[0]
        target_property = leg_count_surfaces(resolve_leg_count(target))[0]
        for modality in MODALITIES:
            if group["group_id"] not in selected_by_cell[f"{source}|{modality}"]:
                continue
            evidence = None
            inputs_by_readout = {}
            built_by_readout = {}
            for band in BAND_RECORD["bands"]:
                if len(band) != 1:
                    raise RuntimeError(
                        "v2 forbids repeated coordinate exchange in one forward pass"
                    )
                start = band[0]
                for arm in ("intermediate", "answer"):
                    for condition in CONDITIONS:
                        for readout in ("identity", "property"):
                            key = safe_key(
                                "paper-swap", group["group_id"], modality, start,
                                arm, condition, readout,
                            )
                            if STORE.has("intervention", key):
                                reused += 1
                                continue
                            if evidence is None:
                                evidence = load_evidence(group, modality)
                            if readout not in inputs_by_readout:
                                built, inputs = make_input(
                                    group, modality, source, target, readout, evidence=evidence
                                )
                                built_by_readout[readout] = built
                                inputs_by_readout[readout] = inputs
                            inputs = inputs_by_readout[readout]
                            clean = clean_by_key[(group["group_id"], modality, readout)]
                            target_answer = target if readout == "identity" else target_property
                            source_answer = source if readout == "identity" else source_property
                            bank = BASES[pair_key][start]
                            if condition.startswith("unrelated_"):
                                alpha = 2.0 if condition.endswith("alpha2") else 1.0
                                bases, position = bank["unrelated"], POSITION_RULE
                            elif condition.startswith("random_"):
                                bases = bank[f"random_{arm}"]
                                alpha = 2.0 if condition.endswith("alpha2") else 1.0
                                position = POSITION_RULE
                            else:
                                bases = bank[arm]
                                alpha = {
                                    "swap_alpha1": 1.0,
                                    "swap_alpha2": 2.0,
                                    "zero": 0.0,
                                    "position_descriptive": 1.0,
                                }[condition]
                                position = (
                                    "final_prompt_token_only"
                                    if condition == "position_descriptive"
                                    else POSITION_RULE
                                )
                            result = run_swap_condition(
                                BACKEND, inputs, bases=bases, alpha=alpha,
                                candidate_ids=CANDIDATE_IDS[readout],
                                target_concept=target_answer, clean_scores=clean["scores"],
                                position_rule=position, record_coordinates=False,
                            )
                            STORE.save("intervention", key, {
                                "status": "complete", "group_id": group["group_id"],
                                "image_id": group["image_id"], "source": source, "target": target,
                                "source_answer": source_answer, "target_answer": target_answer,
                                "modality": modality, "start_layer": start, "band": list(band),
                                "arm": arm, "condition": condition, "alpha": alpha,
                                "readout": readout,
                                "prompt_hash": built_by_readout[readout].prompt_hash,
                                **result,
                            })
                            computed += 1
                            if computed % 10 == 0 or computed == 1:
                                print(f"intervention {computed:,} computed  {reused:,} reused")
    print("intervention complete", {"computed": computed, "reused": reused})
elif STORE is not None:
    print("skipped: the clean capability gate failed before causal spending")
else:
    print("skipped")
'''
)

markdown("## 12. Image-level aggregation and the paper-style onset verdict")
code(
    r'''
RESULT = None
if STORE is not None:
    import os
    from jlens.mmpilot.paper_reasoning_swap import (
        VERDICT_V2_VERSION,
        paper_onset_verdict_v2,
        summarize_cells,
    )
    from jlens.mmpilot.store import payload_checksum

    records = [
        row for row in STORE.load_all("intervention").values()
        if row.get("status") == "complete"
    ]
    CELLS = summarize_cells(records)
    if CAUSAL_SELECTION["all_cells_sufficient"]:
        RESULT = paper_onset_verdict_v2(
            CELLS,
            layers=SAMPLED_LAYERS,
            directed_pairs=DIRECTED_PAIRS,
            modalities=MODALITIES,
            thresholds=THRESHOLDS,
        )
    else:
        RESULT = {
            "version": VERDICT_V2_VERSION,
            "verdict": "PAPER_STYLE_CAPABILITY_NO_GO",
            "primary_onsets": {"intermediate": None, "answer": None},
            "sensitivity_onsets": {"intermediate": None, "answer": None},
            "tested_layers": list(SAMPLED_LAYERS),
            "thresholds": THRESHOLDS.__dict__,
            "threshold_digest": THRESHOLDS.digest,
            "reason": (
                "at least one concept/modality cell had too few independently "
                "selected images with both clean identity and property answers correct"
            ),
        }
    RESULT.update({
        "schema": "mmpilot.paper_reasoning_swap_report.v2",
        "run_dir": str(RUN_DIR),
        "run_fingerprint": STORE.fingerprint.digest,
        "population_digest": POPULATION["population_digest"],
        "prior_exclusion_checksum": EXCLUSION_FILE_CHECKSUM,
        "completed_v1_report_checksum": COMPLETED_V1_REPORT_CHECKSUM,
        "capability_selection": CAUSAL_SELECTION,
        "band_record": BAND_RECORD,
        "directed_pairs": DIRECTED_PAIRS,
        "lens_checksums": {str(k): v for k, v in sorted(LENS_CHECKSUMS.items())},
        "cells_full": CELLS,
        "method_statement": (
            "one exact two-coordinate exchange at one independently confirmed "
            "physical layer per forward pass, applied at every original prompt "
            "position; intermediate and answer arms differ only in which token "
            "pair defines V"
        ),
        "anthropic_comparison": (
            "the exact two-coordinate equation and intermediate-vs-answer "
            "confound comparison are replicated; independent one-layer "
            "localization is a Gemma adaptation because only four physical "
            "layers have confirmed lenses; model, dataset and modalities also differ"
        ),
        "public_method_reference": (
            "https://transformer-circuits.pub/2026/workspace/index.html"
            "#technical-details-of-j-lens-use-cases"
        ),
    })
    RESULT["report_checksum"] = payload_checksum(RESULT)

    report_path = RUN_DIR / "paper_reasoning_swap_v2_report.json"
    tmp = report_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(RESULT, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, report_path)
    STORE.save("metric", "paper_reasoning_onset_verdict_v2", RESULT)

    print("=" * 72)
    print("VERDICT", RESULT["verdict"])
    print("=" * 72)
    print("primary alpha=1 onsets", RESULT["primary_onsets"])
    print("alpha=2 sensitivity  ", RESULT["sensitivity_onsets"])
    print("tested layers        ", RESULT["tested_layers"])
    print("report            ", report_path)
    print("checksum          ", RESULT["report_checksum"])
    print()
    print("This verdict does not use the old native direct-readout convergence gate.")
    print("It compares the paper's two causal interventions directly.")
    print("The final-token-only position diagnostic is reported but cannot veto a result.")
else:
    print("skipped")
'''
)

markdown("## 13. Resume status and what to send back")
code(
    r'''
if STORE is not None:
    print(json.dumps(STORE.status_report(), indent=2))
    print("\nRerunning with the same configuration resumes checksum-valid units.")
    print("A changed lens, cache, pair, layer grid, threshold, prompt protocol,")
    print("audio protocol, condition set, or model revision changes the fingerprint")
    print("and is refused rather than mixed.")
    print("\nSend back paper_reasoning_swap_v2_report.json and the verdict block above.")
else:
    print("To run: set all three section-2 switches True and use an L4 or A100.")
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
    TARGET.write_text(json.dumps(build(), indent=1) + "\n", encoding="utf-8")
    print(f"wrote {TARGET}")
