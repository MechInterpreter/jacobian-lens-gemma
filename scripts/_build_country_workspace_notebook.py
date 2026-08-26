"""Build the prospective country-fact multimodal workspace notebook."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "notebooks" / "multimodal_country_workspace_generalization_colab.ipynb"
CELLS: list[tuple[str, str]] = []


def markdown(value: str) -> None:
    CELLS.append(("markdown", value.strip("\n")))


def code(value: str) -> None:
    CELLS.append(("code", value.strip("\n")))


markdown(
    r"""
# Prospective multimodal coordinate-swap generalization: country facts

This notebook is one clean attempt to broaden the independently confirmed
bird-to-cat leg-count result. It asks whether an **exact exchange of country
identity coordinates** changes two downstream facts, capital and continent,
for two independent country pairs across text, flag images, and spoken country
names.

The benchmark is separate from SpokenCOCO because SpokenCOCO contains COCO
objects, not country identities. The image source contains 15–20 independently
generated and image-verified flag renderings per country. Spoken evidence is a
deterministic WAV rendering of the country name; it is speech input, not a
transcript substitution. The transcript is retained only as provenance.

## Claim ladder

* `PARTIAL_GO`: at least one direction passes both facts in all modalities.
* `BIDIRECTIONAL_GO`: both directions of one pair pass both facts.
* `GENERALIZED_GO`: at least one direction from each of two independent pairs
  passes both facts.
* `FULL_GRID_GO`: every predeclared direction and fact passes.

No finite experiment proves universal generalization. Each verdict names the
measured scope.

## Integrity

* One pooled multimodal J-lens is fitted on 99 examples from eleven countries
  that never appear in evaluation.
* Development and confirmation contain different images and audio files.
* Path selection uses only clean capability and a direct-answer positive
  control. Exact identity-swap outputs are unavailable while paths are chosen.
* Alpha 1 is the exact two-coordinate exchange. No alpha sweep is performed.
* Output is unrestricted greedy generation. There is no candidate list and no
  teacher forcing.
* Every fit checkpoint and every completed forward condition is persisted to
  Drive. A disconnect resumes instead of restarting.

Use an **80 GB A100**. The model is intentionally loaded in fp32; the notebook
refuses a smaller device rather than silently falling back to bf16.
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
'''
)

markdown("## 9. Stage 2: clean capability and outcome-blind path localization")
code(
    r'''
CAPABILITY_REPORT = LOCALIZATION_REPORT = None
CAPABILITY_PATH = RUN_DIR / "country_capability_report.json"
LOCALIZATION_PATH = RUN_DIR / "country_direct_answer_localization_report.json"

def _write_report(path, report):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, path)

if CAPABILITY_PATH.is_file():
    CAPABILITY_REPORT = json.loads(CAPABILITY_PATH.read_text(encoding="utf-8"))
if LOCALIZATION_PATH.is_file():
    LOCALIZATION_REPORT = json.loads(LOCALIZATION_PATH.read_text(encoding="utf-8"))

if RUN_STAGE2_CAPABILITY_AND_LOCALIZATION:
    if not (MODEL_ENABLED and CONFIRM_LOCALIZATION_BUDGET and LENS is not None):
        raise RuntimeError("Stage 2 requires the completed lens and localization budget confirmation")
    from jlens.mmpilot.country_workspace import (
        capability_report, direct_answer_localization_report,
    )
    development_rows = sorted(
        [row for row in MEDIA_ROWS if row["study_split"] == "development"],
        key=lambda row: row["unit_id"],
    )
    capability_rows = []
    for row in development_rows:
        for modality in MODALITIES:
            for property_name in ("identity", *PROPERTIES):
                expected = row["country"] if property_name == "identity" else fact(row["country"], property_name)
                key = safe_key("country_capability", row["unit_id"], modality, property_name)
                stored = STORE.load("capability", key)
                if stored is None:
                    inputs = build_task_inputs(row, modality, property_name)
                    result = unrestricted_greedy_completion(
                        BACKEND, inputs, answer=expected, max_new_tokens=MAX_NEW_TOKENS,
                    )
                    stored = {
                        **result, "unit_id": row["unit_id"], "country": row["country"],
                        "modality": modality, "property": property_name,
                        "expected": expected,
                        "success": generated_success(result, expected, property_name),
                    }
                    STORE.save("capability", key, stored)
                    work = "computed"
                else:
                    work = "reused"
                capability_rows.append(stored)
                if len(capability_rows) == 1 or len(capability_rows) % 24 == 0:
                    print("capability", len(capability_rows), work)
    CAPABILITY_REPORT = capability_report(capability_rows)
    _write_report(CAPABILITY_PATH, CAPABILITY_REPORT)
    print("capability verdict", CAPABILITY_REPORT["verdict"])

    if CAPABILITY_REPORT["verdict"] == "COUNTRY_CAPABILITY_GO":
        tokens = concept_tokens((*EVAL_COUNTRIES, *CONTROL_COUNTRIES))
        unembed = BACKEND.unembedding_weight()
        localization_rows = []
        for property_name in PROPERTIES:
            for source, target in DIRECTIONS:
                source_rows = [
                    row for row in development_rows if row["country"] == source
                ][:N_LOCALIZATION_PER_COUNTRY]
                expected = fact(target, property_name)
                for band in PATH_BANDS:
                    bases = build_swap_bases_for_lens(
                        LENS, unembed, layers=band, source=tokens[source], target=tokens[target]
                    )
                    vectors = answer_vectors(LENS, expected, band)
                    for row in source_rows:
                        for modality in MODALITIES:
                            key = safe_key(
                                "country_direct", property_name, source, target,
                                band[0], band[-1], row["unit_id"], modality,
                            )
                            stored = STORE.load("intervention", key)
                            if stored is None:
                                inputs = build_task_inputs(row, modality, property_name)
                                result = unrestricted_greedy_direct_answer_trial(
                                    BACKEND, inputs, bases=bases, answer_vectors=vectors,
                                    answer=expected, max_new_tokens=MAX_NEW_TOKENS,
                                    position_rule="all_prompt_positions",
                                    realization_policy=TEXT_MODEL_DTYPE_REALIZATION, alpha=1.0,
                                )
                                stored = {
                                    **result, "unit_id": row["unit_id"],
                                    "source": source, "target": target,
                                    "direction": f"{source}->{target}",
                                    "property": property_name, "modality": modality,
                                    "condition": "direct_answer", "expected": expected,
                                    "success": answer_matches(result["generated_text"], expected),
                                    "integrity_pass": diagnostic_integrity(result),
                                }
                                STORE.save("intervention", key, stored)
                                work = "computed"
                            else:
                                work = "reused"
                            localization_rows.append(stored)
                            if len(localization_rows) == 1 or len(localization_rows) % 48 == 0:
                                print("direct localization", len(localization_rows), work)
        LOCALIZATION_REPORT = direct_answer_localization_report(localization_rows)
        _write_report(LOCALIZATION_PATH, LOCALIZATION_REPORT)
        print("localization verdict", LOCALIZATION_REPORT["verdict"])
        print("selected paths", LOCALIZATION_REPORT["selected_paths"])
    else:
        print("Localization not run: clean capability did not pass.")
elif CAPABILITY_REPORT is not None:
    print("reused capability", CAPABILITY_REPORT["verdict"])
    if LOCALIZATION_REPORT is not None:
        print("reused localization", LOCALIZATION_REPORT["verdict"])
'''
)

markdown("## 10. Stage 3: exact alpha-one development")
code(
    r'''
DEVELOPMENT_REPORT = None
DEVELOPMENT_PATH = RUN_DIR / "country_exact_swap_development_report.json"
DESIGN_PATH = RUN_DIR / "country_confirmation_design.json"
if DEVELOPMENT_PATH.is_file():
    DEVELOPMENT_REPORT = json.loads(DEVELOPMENT_PATH.read_text(encoding="utf-8"))

if RUN_STAGE3_DEVELOPMENT_SWAP:
    if not (MODEL_ENABLED and CONFIRM_DEVELOPMENT_BUDGET and LENS is not None):
        raise RuntimeError("Stage 3 requires the completed lens and development budget confirmation")
    if CAPABILITY_REPORT is None:
        raise RuntimeError("Stage 3 requires the completed capability report")
    if CAPABILITY_REPORT["verdict"] != "COUNTRY_CAPABILITY_GO":
        print("DEVELOPMENT NOT LICENSED: clean capability did not pass")
    elif LOCALIZATION_REPORT is None:
        print("DEVELOPMENT NOT LICENSED: no localization report exists")
    elif LOCALIZATION_REPORT["verdict"] != "COUNTRY_DIRECT_PATHS_GO":
        print("DEVELOPMENT NOT LICENSED: direct-answer localization did not pass")
    else:
        from jlens.mmpilot.country_workspace import causal_report, freeze_confirmation_design
        development_rows = sorted(
            [row for row in MEDIA_ROWS if row["study_split"] == "development"],
            key=lambda row: row["unit_id"],
        )
        tokens = concept_tokens((*EVAL_COUNTRIES, *CONTROL_COUNTRIES))
        unembed = BACKEND.unembedding_weight()
        trial_rows = []
        for property_name in PROPERTIES:
            band = tuple(LOCALIZATION_REPORT["selected_paths"][property_name]["band"])
            unrelated = build_swap_bases_for_lens(
                LENS, unembed, layers=band,
                source=tokens[CONTROL_COUNTRIES[0]], target=tokens[CONTROL_COUNTRIES[1]],
            )
            for source, target in DIRECTIONS:
                exact = build_swap_bases_for_lens(
                    LENS, unembed, layers=band, source=tokens[source], target=tokens[target]
                )
                random_bases = {
                    layer: random_two_direction_basis(
                        basis, seed=RANDOM_SEED + layer + 1000 * DIRECTIONS.index((source, target))
                    ) for layer, basis in exact.items()
                }
                expected = fact(target, property_name)
                source_rows = [row for row in development_rows if row["country"] == source]
                for row in source_rows:
                    for modality in MODALITIES:
                        for condition, alpha, bases in (
                            ("exact", 1.0, exact), ("zero", 0.0, exact),
                            ("random", 1.0, random_bases), ("unrelated", 1.0, unrelated),
                        ):
                            key = safe_key(
                                "country_development", property_name, source, target,
                                row["unit_id"], modality, condition,
                            )
                            stored = STORE.load("intervention", key)
                            if stored is None:
                                inputs = build_task_inputs(row, modality, property_name)
                                result = unrestricted_greedy_swap_trial(
                                    BACKEND, inputs, bases=bases, alpha=alpha,
                                    answer=expected, max_new_tokens=MAX_NEW_TOKENS,
                                    position_rule="all_prompt_positions",
                                    realization_policy=TEXT_MODEL_DTYPE_REALIZATION,
                                )
                                stored = {
                                    **result, "unit_id": row["unit_id"],
                                    "source": source, "target": target,
                                    "direction": f"{source}->{target}",
                                    "property": property_name, "modality": modality,
                                    "condition": condition, "expected": expected,
                                    "success": answer_matches(result["generated_text"], expected),
                                    "integrity_pass": diagnostic_integrity(
                                        result, exact=(condition == "exact")
                                    ),
                                }
                                STORE.save("intervention", key, stored)
                                work = "computed"
                            else:
                                work = "reused"
                            trial_rows.append(stored)
                            if len(trial_rows) == 1 or len(trial_rows) % 48 == 0:
                                print("development", len(trial_rows), work)
        DEVELOPMENT_REPORT = causal_report(
            trial_rows, stage="development", expected_n=N_DEVELOPMENT_PER_COUNTRY
        )
        _write_report(DEVELOPMENT_PATH, DEVELOPMENT_REPORT)
        print("development verdict", DEVELOPMENT_REPORT["verdict"])
        print("passing directions", DEVELOPMENT_REPORT["passing_directions_both_properties"])
        if DEVELOPMENT_REPORT["passing_directions_both_properties"]:
            design = freeze_confirmation_design(
                protocol=PROTOCOL, media_validation=PREPARED["media_validation"],
                capability=CAPABILITY_REPORT, localization=LOCALIZATION_REPORT,
                development=DEVELOPMENT_REPORT,
            )
            _write_report(DESIGN_PATH, design)
            print("confirmation design frozen", design["design_checksum"])
        else:
            print("confirmation remains unopened: no direction passed both properties")
elif DEVELOPMENT_REPORT is not None:
    print("reused development", DEVELOPMENT_REPORT["verdict"])
'''
)

markdown("## 11. Stage 4: fresh confirmation")
code(
    r'''
CONFIRMATION_REPORT = None
CONFIRMATION_PATH = RUN_DIR / "country_fresh_confirmation_report.json"
if CONFIRMATION_PATH.is_file():
    CONFIRMATION_REPORT = json.loads(CONFIRMATION_PATH.read_text(encoding="utf-8"))

if RUN_STAGE4_FRESH_CONFIRMATION:
    if not (MODEL_ENABLED and CONFIRM_CONFIRMATION_BUDGET and LENS is not None):
        raise RuntimeError("Stage 4 requires the completed lens and confirmation budget confirmation")
    if not DESIGN_PATH.is_file():
        print("CONFIRMATION NOT LICENSED: no frozen development winner")
    else:
        from jlens.mmpilot.country_workspace import causal_report
        design = json.loads(DESIGN_PATH.read_text(encoding="utf-8"))
        selected_directions = list(design["directions"])
        confirmation_rows = sorted(
            [row for row in MEDIA_ROWS if row["study_split"] == "confirmation"],
            key=lambda row: row["unit_id"],
        )
        selected_sources = {direction.split("->", 1)[0] for direction in selected_directions}
        clean_rows = []
        for row in confirmation_rows:
            if row["country"] not in selected_sources:
                continue
            for modality in MODALITIES:
                for property_name in PROPERTIES:
                    expected = fact(row["country"], property_name)
                    key = safe_key("country_confirm_capability", row["unit_id"], modality, property_name)
                    stored = STORE.load("capability", key)
                    if stored is None:
                        inputs = build_task_inputs(row, modality, property_name)
                        result = unrestricted_greedy_completion(
                            BACKEND, inputs, answer=expected, max_new_tokens=MAX_NEW_TOKENS,
                        )
                        stored = {
                            **result, "unit_id": row["unit_id"], "country": row["country"],
                            "modality": modality, "property": property_name,
                            "expected": expected,
                            "success": answer_matches(result["generated_text"], expected),
                        }
                        STORE.save("capability", key, stored)
                        work = "computed"
                    else:
                        work = "reused"
                    clean_rows.append(stored)
                    if len(clean_rows) == 1 or len(clean_rows) % 24 == 0:
                        print("confirmation capability", len(clean_rows), work)
        clean_cells = []
        for country in sorted(selected_sources):
            for property_name in PROPERTIES:
                for modality in MODALITIES:
                    cell = [
                        row for row in clean_rows
                        if row["country"] == country
                        and row["property"] == property_name
                        and row["modality"] == modality
                    ]
                    successes = sum(bool(row["success"]) for row in cell)
                    rate = successes / len(cell) if cell else 0.0
                    clean_cells.append({
                        "country": country, "property": property_name,
                        "modality": modality, "n": len(cell),
                        "successes": successes, "rate": rate,
                        "passed": (
                            len(cell) == N_CONFIRMATION_PER_COUNTRY
                            and rate >= MIN_CAPABILITY_RATE
                        ),
                    })
        clean_gate = bool(clean_cells) and all(cell["passed"] for cell in clean_cells)
        print("confirmation clean capability", clean_gate, clean_cells)
        if not clean_gate:
            CONFIRMATION_REPORT = {
                "version": PROTOCOL["version"],
                "verdict": "COUNTRY_CONFIRMATION_CAPABILITY_NO_GO",
                "design": design, "capability_rows": clean_rows,
                "capability_cells": clean_cells,
                "confirmation_interventions_run": False,
            }
            CONFIRMATION_REPORT["report_checksum"] = payload_checksum(CONFIRMATION_REPORT)
            _write_report(CONFIRMATION_PATH, CONFIRMATION_REPORT)
        else:
            tokens = concept_tokens((*EVAL_COUNTRIES, *CONTROL_COUNTRIES))
            unembed = BACKEND.unembedding_weight()
            trial_rows = []
            for direction in selected_directions:
                source, target = direction.split("->", 1)
                source_rows = [row for row in confirmation_rows if row["country"] == source]
                for property_name in PROPERTIES:
                    band = tuple(design["selected_paths"][property_name]["band"])
                    exact = build_swap_bases_for_lens(
                        LENS, unembed, layers=band, source=tokens[source], target=tokens[target]
                    )
                    random_bases = {
                        layer: random_two_direction_basis(
                            basis, seed=RANDOM_SEED + 50000 + layer + 1000 * selected_directions.index(direction)
                        ) for layer, basis in exact.items()
                    }
                    unrelated = build_swap_bases_for_lens(
                        LENS, unembed, layers=band,
                        source=tokens[CONTROL_COUNTRIES[0]], target=tokens[CONTROL_COUNTRIES[1]],
                    )
                    expected = fact(target, property_name)
                    for row in source_rows:
                        for modality in MODALITIES:
                            for condition, alpha, bases in (
                                ("exact", 1.0, exact), ("zero", 0.0, exact),
                                ("random", 1.0, random_bases), ("unrelated", 1.0, unrelated),
                            ):
                                key = safe_key(
                                    "country_confirmation", property_name, source, target,
                                    row["unit_id"], modality, condition,
                                )
                                stored = STORE.load("intervention", key)
                                if stored is None:
                                    inputs = build_task_inputs(row, modality, property_name)
                                    result = unrestricted_greedy_swap_trial(
                                        BACKEND, inputs, bases=bases, alpha=alpha,
                                        answer=expected, max_new_tokens=MAX_NEW_TOKENS,
                                        position_rule="all_prompt_positions",
                                        realization_policy=TEXT_MODEL_DTYPE_REALIZATION,
                                    )
                                    stored = {
                                        **result, "unit_id": row["unit_id"],
                                        "source": source, "target": target,
                                        "direction": direction, "property": property_name,
                                        "modality": modality, "condition": condition,
                                        "expected": expected,
                                        "success": answer_matches(result["generated_text"], expected),
                                        "integrity_pass": diagnostic_integrity(
                                            result, exact=(condition == "exact")
                                        ),
                                    }
                                    STORE.save("intervention", key, stored)
                                    work = "computed"
                                else:
                                    work = "reused"
                                trial_rows.append(stored)
                                if len(trial_rows) == 1 or len(trial_rows) % 48 == 0:
                                    print("confirmation", len(trial_rows), work)
            CONFIRMATION_REPORT = causal_report(
                trial_rows, stage="confirmation", expected_n=N_CONFIRMATION_PER_COUNTRY,
                frozen_directions=selected_directions,
            )
            CONFIRMATION_REPORT["design"] = design
            CONFIRMATION_REPORT["capability_rows"] = clean_rows
            CONFIRMATION_REPORT["capability_cells"] = clean_cells
            body = {k: v for k, v in CONFIRMATION_REPORT.items() if k != "report_checksum"}
            CONFIRMATION_REPORT["report_checksum"] = payload_checksum(body)
            _write_report(CONFIRMATION_PATH, CONFIRMATION_REPORT)
            print("=" * 88)
            print("FRESH COUNTRY-FACT CONFIRMATION", CONFIRMATION_REPORT["verdict"])
            print("passing", CONFIRMATION_REPORT["passing_directions_both_properties"])
            print("bidirectional", CONFIRMATION_REPORT["bidirectional_pairs_both_properties"])
            print("two-pair generalized", CONFIRMATION_REPORT["generalized_across_two_pairs"])
            print("report", CONFIRMATION_PATH)
            print("checksum", CONFIRMATION_REPORT["report_checksum"])
elif CONFIRMATION_REPORT is not None:
    print("reused confirmation", CONFIRMATION_REPORT["verdict"])
'''
)

markdown("## 12. Final report and handoff")
code(
    r'''
if RUN_STAGE5_WRITE_REPORT:
    FINAL = {
        "version": PROTOCOL["version"],
        "scientific_config": SCIENTIFIC_CONFIG,
        "protocol": PROTOCOL,
        "media_validation": PREPARED["media_validation"],
        "lens_path": str(LENS_PATH),
        "lens_exists": LENS_PATH.is_file(),
        "capability": CAPABILITY_REPORT,
        "localization": LOCALIZATION_REPORT,
        "development": DEVELOPMENT_REPORT,
        "confirmation": CONFIRMATION_REPORT,
        "headline_verdict": (
            CONFIRMATION_REPORT.get("verdict") if CONFIRMATION_REPORT
            else DEVELOPMENT_REPORT.get("verdict") if DEVELOPMENT_REPORT
            else LOCALIZATION_REPORT.get("verdict") if LOCALIZATION_REPORT
            else CAPABILITY_REPORT.get("verdict") if CAPABILITY_REPORT
            else "NOT_RUN"
        ),
        "claim_boundary": (
            "The confirmed bird-to-cat leg-count result is a separate completed "
            "study. This benchmark broadens it only to country pairs and facts "
            "that pass fresh confirmation; a development result is never called "
            "confirmed and no finite grid is called universal."
        ),
    }
    FINAL["report_checksum"] = payload_checksum(FINAL)
    final_path = RUN_DIR / "country_workspace_generalization_report.json"
    _write_report(final_path, FINAL)
    print("FINAL", FINAL["headline_verdict"])
    print("report", final_path)
    print("checksum", FINAL["report_checksum"])
print("resume status", json.dumps(STORE.status_report(), indent=2))
print("Send back country_workspace_generalization_report.json and, if present,")
print("country_fresh_confirmation_report.json.")
'''
)
markdown("## 5. Open the fingerprinted run")
code(
    r'''
from jlens.mmpilot.store import RunFingerprint, UnitStore, payload_checksum, safe_key
PROTOCOL = PREPARED["protocol"]
SCIENTIFIC_CONFIG = {
    "study": PROTOCOL["version"],
    "protocol_digest": PROTOCOL["protocol_digest"],
    "dataset_revision": PREPARED["dataset_revision"],
    "population_digest": PREPARED["population_digest"],
    "model_repo_id": MODEL_REPO_ID,
    "model_revision": MODEL_REVISION,
    "model_dtype": "float32",
    "audio_protocol_fingerprint": AUDIO_PROTOCOL_FINGERPRINT,
    "max_new_tokens": MAX_NEW_TOKENS,
    "random_seed": RANDOM_SEED,
    # Engineering-only control-flow repairs do not create a new scientific
    # run. Any scientific change must deliberately update this frozen pin.
    "commit": SCIENTIFIC_IMPLEMENTATION_COMMIT,
}
SCIENTIFIC_DIGEST = payload_checksum(SCIENTIFIC_CONFIG)
RUN_DIR = RUNS_ROOT / f"mmcountry_real_{SCIENTIFIC_DIGEST.split(':')[1][:12]}"
RUN_DIR.mkdir(parents=True, exist_ok=True)
(RUN_DIR / "scientific_config.json").write_text(
    json.dumps(SCIENTIFIC_CONFIG, indent=2), encoding="utf-8"
)
FINGERPRINT = RunFingerprint(
    mode="real", model_repo_id=MODEL_REPO_ID, model_revision=MODEL_REVISION,
    processor_revision=MODEL_REVISION, layers=tuple(LAYERS),
    lens_checksum="country-pooled-fit-bound-in-run",
    manifest_checksum=PREPARED["population_digest"], split_id=PREPARED["population_digest"],
    intervention_config={
        "alpha": 1.0, "position_rule": "all_prompt_positions",
        "directions": [list(pair) for pair in DIRECTIONS],
        "properties": list(PROPERTIES), "path_bands": [list(band) for band in PATH_BANDS],
        "controls": ["zero", "random", "unrelated", "direct_answer"],
    },
    extra={"scientific_digest": SCIENTIFIC_DIGEST, "dtype": "float32"},
)
STORE = UnitStore(RUN_DIR, FINGERPRINT)
print("run directory", RUN_DIR)
print("resume", STORE.open())
'''
)

markdown("## 6. Load Gemma 4 E4B in fp32")
code(
    r'''
BACKEND = BUNDLE = None
MODEL_ENABLED = (
    REAL_MODE and MODEL_STAGE and CONFIRM_MODEL_LOAD and CONFIRM_FP32_A100
)
if MODEL_ENABLED:
    import getpass
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA runtime is required")
    name = torch.cuda.get_device_name(0)
    total_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
    if "A100" not in name or total_gib < 70:
        raise RuntimeError(
            f"fp32 study requires an 80 GB A100; found {name} with {total_gib:.1f} GiB"
        )
    if not os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = getpass.getpass("HF_TOKEN (input hidden): ").strip()
    from jlens.mmpilot.real_backend import build_real_backend
    from jlens.mmpilot.tri_modal import assert_audio_protocol
    BUNDLE = build_real_backend(
        MODEL_REPO_ID, revision=MODEL_REVISION, token=os.environ["HF_TOKEN"],
        device="cuda", allow_model_load=True, resolve_audio=True,
        expect_n_layers=EXPECT_N_LAYERS, expect_d_model=EXPECT_D_MODEL,
        expect_vocab_size=EXPECT_VOCAB, dtype=torch.float32,
    )
    if BUNDLE.audio_interface is None:
        raise RuntimeError("native audio unavailable: " + BUNDLE.audio_blocked_reason)
    assert_audio_protocol(
        BUNDLE.audio_interface, expected_fingerprint=AUDIO_PROTOCOL_FINGERPRINT
    )
    BACKEND = BUNDLE.backend
    observed_dtype = str(next(BACKEND.hf_model.parameters()).dtype)
    if observed_dtype != "torch.float32":
        raise RuntimeError(f"requested fp32 but loaded {observed_dtype}")
    print("loaded", name, f"{total_gib:.1f} GiB", observed_dtype)
elif MODEL_STAGE:
    print("model stage requested but confirmation flags are false")
'''
)

markdown("## 7. Input, answer, vector, and integrity helpers")
code(
    r'''
import random
if MODEL_STAGE:
    import soundfile as sf
    from PIL import Image
else:
    sf = Image = None
from jlens.mmpilot.coordinate_swap import (
    random_two_direction_basis, resolve_concept_token,
)
from jlens.mmpilot.country_workspace import (
    CONTROL_COUNTRIES, FACTS, answer_matches, assistant_prefill, fact,
    identity_matches, speech_evidence, text_evidence,
)
from jlens.mmpilot.multimodal_lens import (
    build_swap_bases_for_lens, selected_lens_vector,
)
from jlens.mmpilot.workspace_replication import (
    TEXT_MODEL_DTYPE_REALIZATION, build_multimodal_assistant_prefill_inputs,
    unrestricted_greedy_completion, unrestricted_greedy_direct_answer_trial,
    unrestricted_greedy_swap_trial,
)

MEDIA_ROWS = [dict(row) for row in PREPARED["rows"]]

def _load_image(row):
    return Image.open(row["image_path"]).convert("RGB")

def _load_audio(row):
    waveform, rate = sf.read(row["audio_path"], dtype="float32", always_2d=False)
    if getattr(waveform, "ndim", 1) == 2:
        waveform = waveform.mean(axis=1)
    if int(rate) != 16000:
        raise RuntimeError(f"prepared audio is {rate} Hz, not 16000")
    return waveform, int(rate)

def build_task_inputs(row, modality, property_name):
    kwargs = {
        "backend": BACKEND, "modality": modality,
        "assistant_prefill": assistant_prefill(property_name),
    }
    if modality == "text":
        kwargs["caption"] = text_evidence(row["country"])
    elif modality == "image":
        kwargs["image"] = _load_image(row)
        kwargs["media_path"] = row["image_path"]
    else:
        waveform, rate = _load_audio(row)
        kwargs.update(audio=waveform, sampling_rate=rate, media_path=row["audio_path"])
    return build_multimodal_assistant_prefill_inputs(**kwargs)

def generated_success(result, expected, property_name):
    if property_name == "identity":
        return identity_matches(result["generated_text"], expected)
    return answer_matches(result["generated_text"], expected)

def diagnostic_integrity(result, *, exact=False):
    diag = result.get("intervention_diagnostics") or {}
    return bool(
        diag.get("all_hooks_fired") and diag.get("all_finite")
        and diag.get("all_model_dtype_realizations_converged")
        and diag.get("all_cumulative_band_displacements_complete")
        and (not exact or diag.get("all_layers_are_exact_alpha_one_exchange_before_cast"))
    )

def concept_tokens(names):
    return {name: resolve_concept_token(BACKEND.encode_candidate, name) for name in names}

def answer_vectors(lens, answer, band):
    token = resolve_concept_token(BACKEND.encode_candidate, answer)
    unembed = BACKEND.unembedding_weight()
    return {
        layer: selected_lens_vector(
            lens, unembed, layer=layer, token_id=token.token_id
        ) for layer in band
    }
'''
)

markdown("## 8. Stage 1: fit one pooled multimodal J-lens")
code(
    r'''
LENS = None
LENS_PATH = RUN_DIR / "lenses" / "lens.pooled.country.l16_l40.pt"
LENS_PROVENANCE_PATH = RUN_DIR / "lenses" / "lens.pooled.country.l16_l40.json"
if LENS_PATH.is_file():
    from jlens.lens import JacobianLens
    from jlens.mmpilot.backend import file_checksum
    if not LENS_PROVENANCE_PATH.is_file():
        raise RuntimeError("completed lens has no checksum-bound provenance record")
    lens_provenance = json.loads(LENS_PROVENANCE_PATH.read_text(encoding="utf-8"))
    if file_checksum(str(LENS_PATH)) != lens_provenance.get("lens_checksum"):
        raise RuntimeError("completed lens checksum does not match its provenance record")
    LENS = JacobianLens.load(str(LENS_PATH))
    if LENS.source_layers != list(LAYERS) or LENS.n_prompts != 99 or LENS.d_model != EXPECT_D_MODEL:
        raise RuntimeError("completed lens has the wrong layers, population size, or width")
    print("reused completed lens", LENS_PATH)
elif RUN_STAGE1_FIT_POOLED_LENS:
    if not (MODEL_ENABLED and CONFIRM_LENS_FIT_BUDGET):
        raise RuntimeError("Stage 1 requires model, fp32 A100, and fit-budget confirmation")
    from jlens.mmpilot.multimodal_lens import FitUnit, fit_arm, fitting_prompt

    fit_rows = sorted(
        [row for row in MEDIA_ROWS if row["study_split"] == "fit"],
        key=lambda row: row["unit_id"],
    )
    units = []
    for row in fit_rows:
        for modality in MODALITIES:
            units.append(FitUnit(
                unit_id=f"{row['unit_id']}:{modality}", group_id=row["unit_id"],
                image_id=row["image_checksum"], modality=modality,
                caption=text_evidence(row["country"]), image_path=row["image_path"],
                audio_path=row["audio_path"],
                prompt=fitting_prompt(modality, text_evidence(row["country"])),
            ))
    if len(units) != 99:
        raise RuntimeError(f"frozen pooled fit requires 99 units, got {len(units)}")

    def build_fit_input(unit):
        row = next(row for row in fit_rows if row["unit_id"] == unit.group_id)
        if unit.modality == "text":
            return BACKEND.build_inputs(prompt=unit.prompt, modality="text")
        if unit.modality == "image":
            return BACKEND.build_inputs(
                prompt=unit.prompt, modality="image", image=_load_image(row),
                media_path=row["image_path"],
            )
        waveform, rate = _load_audio(row)
        return BACKEND.build_inputs(
            prompt=unit.prompt, modality="spoken_audio", audio=waveform,
            sampling_rate=rate, media_path=row["audio_path"],
        )

    def progress(info):
        if info["index"] == 1 or info["checkpoint_written"] or info["index"] == info["total"]:
            print("pooled", info["index"], "/", info["total"], info["modality"],
                  f"{info['elapsed_seconds']:.1f}s", "checkpoint", info["checkpoint_written"])

    checkpoint = RUN_DIR / "lenses" / "checkpoints" / "pooled.jacobian_sum.pt"
    LENS = fit_arm(
        BACKEND, units, build_inputs=build_fit_input, source_layers=LAYERS,
        target_layer=TARGET_LAYER, checkpoint_path=checkpoint, arm="pooled",
        scientific_fingerprint=SCIENTIFIC_DIGEST, dim_batch=DIM_BATCH,
        skip_first=SKIP_FIRST, checkpoint_every=CHECKPOINT_EVERY, progress=progress,
    )
    LENS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = LENS_PATH.with_suffix(".tmp.pt")
    # Preserve the fp32 fit. Saving with JacobianLens.save's fp16 default would
    # make a resumed run scientifically different from the fitting session.
    LENS.save(str(temporary), dtype=torch.float32)
    os.replace(temporary, LENS_PATH)
    from jlens.mmpilot.backend import file_checksum
    lens_provenance = {
        "lens_checksum": file_checksum(str(LENS_PATH)),
        "scientific_digest": SCIENTIFIC_DIGEST,
        "source_layers": list(LENS.source_layers),
        "target_layer": TARGET_LAYER,
        "n_prompts": LENS.n_prompts,
        "d_model": LENS.d_model,
        "stored_dtype": "float32",
    }
    lens_provenance["provenance_checksum"] = payload_checksum(lens_provenance)
    provenance_tmp = LENS_PROVENANCE_PATH.with_name(
        LENS_PROVENANCE_PATH.name + ".tmp"
    )
    provenance_tmp.write_text(
        json.dumps(lens_provenance, indent=2), encoding="utf-8"
    )
    os.replace(provenance_tmp, LENS_PROVENANCE_PATH)
    print("lens completed", LENS_PATH)
    print("lens checksum", lens_provenance["lens_checksum"])
elif any((RUN_STAGE2_CAPABILITY_AND_LOCALIZATION, RUN_STAGE3_DEVELOPMENT_SWAP, RUN_STAGE4_FRESH_CONFIRMATION)):
    raise RuntimeError("No completed lens exists. Run Stage 1 first.")
'''
)

markdown("## 0. Bootstrap continuation")
code(
    r'''
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
         "transformers==5.13.1", "accelerate", "datasets", "pillow",
         "soundfile", "pytesseract"],
        check=True,
    )
    if not Path("/usr/bin/espeak-ng").is_file() or not Path("/usr/bin/tesseract").is_file():
        subprocess.run(["apt-get", "update", "-qq"], check=True)
        subprocess.run(["apt-get", "install", "-y", "-qq", "espeak-ng", "tesseract-ocr"], check=True)
for name in tuple(sys.modules):
    if name == "jlens" or name.startswith("jlens."):
        sys.modules.pop(name, None)
os.chdir(REPO_DIR)
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))
COMMIT = subprocess.run(
    ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
).stdout.strip()
print("commit", COMMIT)
'''
)

markdown("## 1. Configuration")
code(
    r'''
# Stage 0 is CPU preparation. Stages 1-4 require an 80 GB A100.
RUN_REAL_COUNTRY_WORKSPACE = False
RUN_STAGE0_PREPARE_DATA = False
RUN_STAGE1_FIT_POOLED_LENS = False
RUN_STAGE2_CAPABILITY_AND_LOCALIZATION = False
RUN_STAGE3_DEVELOPMENT_SWAP = False
RUN_STAGE4_FRESH_CONFIRMATION = False
RUN_STAGE5_WRITE_REPORT = False

CONFIRM_MODEL_LOAD = False
CONFIRM_FP32_A100 = False
CONFIRM_LENS_FIT_BUDGET = False
CONFIRM_LOCALIZATION_BUDGET = False
CONFIRM_DEVELOPMENT_BUDGET = False
CONFIRM_CONFIRMATION_BUDGET = False

MODEL_REPO_ID = "google/gemma-4-E4B-it"
MODEL_REVISION = "fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd"
SCIENTIFIC_IMPLEMENTATION_COMMIT = (
    "09283b7e3ba98fe49a21a284327e4eac2edf4d86"
)
EXPECT_N_LAYERS, EXPECT_D_MODEL, EXPECT_VOCAB = 42, 2560, 262144
AUDIO_PROTOCOL_FINGERPRINT = (
    "sha256:9ad8bcc9420a7983f6e3b75d5d7080c0e2fcf0a94a76431917fcde73ba777920"
)
DATA_ROOT = Path("/content/drive/MyDrive/datasets/jlens_country_workspace_v1")
RUNS_ROOT = Path("/content/drive/MyDrive/jacobian-lens-gemma/runs/mmcountry")
PREPARED_POPULATION_PATH = DATA_ROOT / "prepared_population.json"
PREPARATION_COMPLETE_PATH = DATA_ROOT / "preparation_complete.json"
CHECKPOINT_EVERY = 5
DIM_BATCH = 8
SKIP_FIRST = 16
MAX_NEW_TOKENS = 6
RANDOM_SEED = 20260826

REAL_MODE = bool(RUN_REAL_COUNTRY_WORKSPACE)
ANY_STAGE = any((
    RUN_STAGE0_PREPARE_DATA,
    RUN_STAGE1_FIT_POOLED_LENS,
    RUN_STAGE2_CAPABILITY_AND_LOCALIZATION,
    RUN_STAGE3_DEVELOPMENT_SWAP,
    RUN_STAGE4_FRESH_CONFIRMATION,
    RUN_STAGE5_WRITE_REPORT,
))
if not ANY_STAGE:
    DATA_ROOT = Path(tempfile.gettempdir()) / "jlens_country_workspace_disabled"
    RUNS_ROOT = DATA_ROOT / "runs"
    PREPARED_POPULATION_PATH = DATA_ROOT / "prepared_population.json"
    PREPARATION_COMPLETE_PATH = DATA_ROOT / "preparation_complete.json"
MODEL_STAGE = any((
    RUN_STAGE1_FIT_POOLED_LENS,
    RUN_STAGE2_CAPABILITY_AND_LOCALIZATION,
    RUN_STAGE3_DEVELOPMENT_SWAP,
    RUN_STAGE4_FRESH_CONFIRMATION,
))
if MODEL_STAGE and not CONFIRM_MODEL_LOAD:
    print("MODEL BLOCKED: set CONFIRM_MODEL_LOAD=True after reading the budget")
if MODEL_STAGE and not CONFIRM_FP32_A100:
    print("FP32 BLOCKED: set CONFIRM_FP32_A100=True on an 80 GB A100")
'''
)

markdown("## 2. Frozen protocol and budget, printed before model load")
code(
    r'''
from jlens.mmpilot.country_workspace import (
    DIRECTIONS, EVAL_COUNTRIES, FIT_COUNTRIES, LAYERS, MODALITIES,
    N_CONFIRMATION_PER_COUNTRY, N_DEVELOPMENT_PER_COUNTRY,
    N_FIT_PER_COUNTRY, N_LOCALIZATION_PER_COUNTRY, PATH_BANDS,
    PROPERTIES, TARGET_LAYER, benchmark_spec,
)

print("COUNTRY WORKSPACE BUDGET")
print("  fit population", len(FIT_COUNTRIES) * N_FIT_PER_COUNTRY, "identities x 3 modalities = 99 examples")
print("  fit work", 99, "forwards +", 99 * ((EXPECT_D_MODEL + DIM_BATCH - 1) // DIM_BATCH), "backward passes")
print("  clean capability", len(EVAL_COUNTRIES) * N_DEVELOPMENT_PER_COUNTRY * len(MODALITIES) * 3, "complete generations")
print("  direct localization", len(DIRECTIONS) * len(PROPERTIES) * len(MODALITIES) * len(PATH_BANDS) * N_LOCALIZATION_PER_COUNTRY, "conditions")
print("  development", len(DIRECTIONS) * len(PROPERTIES) * len(MODALITIES) * 4 * N_DEVELOPMENT_PER_COUNTRY, "conditions")
print("  confirmation maximum", len(DIRECTIONS) * len(PROPERTIES) * len(MODALITIES) * 4 * N_CONFIRMATION_PER_COUNTRY, "conditions")
print("  model dtype float32; A100 80 GB required; no fallback")
print("  resume fit checkpoint every", CHECKPOINT_EVERY, "examples; causal resume unit one condition")
print("  expected wall time after CPU prep: approximately 8-16 A100 hours if every stage is licensed")
print("  Stage 2 or 3 can stop the study early without opening confirmation")
'''
)

markdown("## 3. Mount Drive")
code(
    r'''
if IN_COLAB and ANY_STAGE:
    from google.colab import drive
    drive.mount("/content/drive")
DATA_ROOT.mkdir(parents=True, exist_ok=True)
RUNS_ROOT.mkdir(parents=True, exist_ok=True)
'''
)

markdown("## 4. Stage 0, CPU only: prepare and seal the dataset")
code(
    r'''
PREPARED = None
from jlens.mmpilot.store import payload_checksum
if RUN_STAGE0_PREPARE_DATA:
    import subprocess
    import numpy as np
    import pytesseract
    import soundfile as sf
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from PIL import Image
    from jlens.mmpilot.country_workspace import (
        DATASET_ID, DATASET_REVISION, EVAL_COUNTRIES, FACTS, FIT_COUNTRIES,
        CountryMediaRow,
        N_CONFIRMATION_PER_COUNTRY, N_DEVELOPMENT_PER_COUNTRY,
        N_FIT_PER_COUNTRY, benchmark_spec, normalize_surface, speech_evidence,
        validate_media_plan,
    )
    def _sha(path):
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return "sha256:" + digest.hexdigest()

    def _atomic_json(path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary, path)

    if PREPARATION_COMPLETE_PATH.is_file() and PREPARED_POPULATION_PATH.is_file():
        PREPARED = json.loads(PREPARED_POPULATION_PATH.read_text(encoding="utf-8"))
        complete = json.loads(PREPARATION_COMPLETE_PATH.read_text(encoding="utf-8"))
        prepared_body = {
            key: value for key, value in PREPARED.items()
            if key != "prepared_checksum"
        }
        if PREPARED.get("prepared_checksum") != payload_checksum(prepared_body):
            raise RuntimeError("prepared population failed its own checksum")
        if complete.get("prepared_checksum") != PREPARED.get("prepared_checksum"):
            raise RuntimeError("preparation completion marker has the wrong checksum")
        if complete.get("population_digest") != PREPARED.get("population_digest"):
            raise RuntimeError("prepared population checksum does not match completion marker")
        expected_protocol = benchmark_spec(
            dataset_revision=str(PREPARED.get("dataset_revision") or "")
        )
        if PREPARED.get("protocol", {}).get("protocol_digest") != expected_protocol["protocol_digest"]:
            raise RuntimeError("prepared population belongs to a different frozen protocol")
        print("Stage 0 reused completed preparation", PREPARED_POPULATION_PATH)
    else:
        dataset_revision = HfApi().dataset_info(
            DATASET_ID, revision=DATASET_REVISION
        ).sha
        if dataset_revision != DATASET_REVISION:
            raise RuntimeError(
                f"dataset revision resolved to {dataset_revision}, not the "
                f"frozen {DATASET_REVISION}"
            )
        protocol = benchmark_spec(dataset_revision=dataset_revision)
        dataset = load_dataset(
            DATASET_ID, revision=dataset_revision, streaming=True
        )
        available = []
        for split_name, split in dataset.items():
            for index, record in enumerate(split):
                country = str(record.get("country") or "")
                if country not in {*FIT_COUNTRIES, *EVAL_COUNTRIES}:
                    continue
                seed = str(record.get("seed") or index)
                rank = hashlib.sha256(f"country-workspace-v1|{country}|{split_name}|{seed}|{index}".encode()).hexdigest()
                available.append((country, rank, split_name, index, seed, record["image"]))
        by_country = {}
        for country in (*FIT_COUNTRIES, *EVAL_COUNTRIES):
            by_country[country] = sorted(
                [row for row in available if row[0] == country], key=lambda row: row[1]
            )
        shortages = {
            country: len(rows) for country, rows in by_country.items()
            if len(rows) < (N_FIT_PER_COUNTRY if country in FIT_COUNTRIES else N_DEVELOPMENT_PER_COUNTRY + N_CONFIRMATION_PER_COUNTRY)
        }
        if shortages:
            raise RuntimeError(f"dataset does not support the frozen design: {shortages}")

        rows = []
        image_root = DATA_ROOT / "images"
        audio_root = DATA_ROOT / "audio"
        image_root.mkdir(parents=True, exist_ok=True)
        audio_root.mkdir(parents=True, exist_ok=True)
        prep_unit_root = DATA_ROOT / "prep_units" / dataset_revision
        prep_unit_root.mkdir(parents=True, exist_ok=True)
        voices = ("en-us", "en-gb", "en")
        forbidden_ocr = {
            normalize_surface(value)
            for value in (*FIT_COUNTRIES, *EVAL_COUNTRIES)
        } | {
            normalize_surface(value)
            for mapping in FACTS.values()
            for value in mapping.values()
        }

        def _ocr_has_forbidden(text):
            padded = f" {normalize_surface(text)} "
            return sorted(
                token for token in forbidden_ocr
                if token and f" {token} " in padded
            )

        for country in (*FIT_COUNTRIES, *EVAL_COUNTRIES):
            if country in FIT_COUNTRIES:
                needed = N_FIT_PER_COUNTRY
            else:
                needed = N_DEVELOPMENT_PER_COUNTRY + N_CONFIRMATION_PER_COUNTRY
            selected_count = 0
            for source in by_country[country]:
                if selected_count >= needed:
                    break
                ordinal = selected_count
                study_split = (
                    "fit" if country in FIT_COUNTRIES
                    else "development" if ordinal < N_DEVELOPMENT_PER_COUNTRY
                    else "confirmation"
                )
                _, rank, source_split, source_index, source_seed, image = source
                unit_id = hashlib.sha256(f"{country}|{rank}|{study_split}".encode()).hexdigest()[:20]
                prep_unit_path = prep_unit_root / f"{unit_id}.json"
                if prep_unit_path.is_file():
                    candidate = json.loads(prep_unit_path.read_text(encoding="utf-8"))
                    candidate_body = {
                        key: value for key, value in candidate.items()
                        if key != "prep_unit_checksum"
                    }
                    if candidate.get("prep_unit_checksum") == payload_checksum(candidate_body):
                        if Path(candidate["image_path"]).is_file() and Path(candidate["audio_path"]).is_file():
                            if _sha(Path(candidate["image_path"])) == candidate["image_checksum"] and _sha(Path(candidate["audio_path"])) == candidate["audio_checksum"]:
                                rows.append(candidate_body)
                                selected_count += 1
                                print("prepared", len(rows), country, study_split, "reused")
                                continue
                image_path = image_root / f"{unit_id}.png"
                if not image_path.is_file():
                    image.convert("RGB").save(image_path, format="PNG")
                ocr_text = pytesseract.image_to_string(Image.open(image_path).convert("RGB"))
                ocr_hits = _ocr_has_forbidden(ocr_text)
                if ocr_hits:
                    print("rejected OCR", country, source_seed, ocr_hits)
                    continue
                voice = voices[ordinal % len(voices)]
                # A unique speaking rate per within-country item prevents two
                # study units from carrying byte-identical audio evidence.
                speed = 135 + 3 * ordinal
                pitch = 35 + (7 * ordinal) % 30
                speech = speech_evidence(country)
                audio_path = audio_root / f"{unit_id}.wav"
                if not audio_path.is_file():
                    raw_path = audio_root / f"{unit_id}.raw.wav"
                    subprocess.run([
                        "espeak-ng", "-v", voice, "-s", str(speed), "-p", str(pitch),
                        "-w", str(raw_path), speech,
                    ], check=True, capture_output=True)
                    waveform, rate = sf.read(raw_path, dtype="float32", always_2d=False)
                    if waveform.ndim == 2:
                        waveform = waveform.mean(axis=1)
                    old = np.arange(len(waveform), dtype=np.float64) / float(rate)
                    new_len = max(1, int(round(len(waveform) * 16000.0 / float(rate))))
                    new = np.arange(new_len, dtype=np.float64) / 16000.0
                    waveform = np.interp(new, old, waveform).astype("float32")
                    peak = float(np.max(np.abs(waveform))) if len(waveform) else 0.0
                    if peak > 0.98:
                        waveform = waveform * (0.98 / peak)
                    sf.write(audio_path, waveform, 16000, subtype="FLOAT")
                    raw_path.unlink(missing_ok=True)
                prepared_row = CountryMediaRow(
                    unit_id=unit_id, country=country, source_split=source_split,
                    source_index=int(source_index), source_seed=source_seed,
                    image_path=str(image_path), image_checksum=_sha(image_path),
                    audio_path=str(audio_path), audio_checksum=_sha(audio_path),
                    speech_text=speech, speech_voice=voice, speech_speed=speed,
                    speech_pitch=pitch, study_split=study_split, ocr_text=ocr_text,
                ).to_dict()
                _atomic_json(prep_unit_path, {
                    **prepared_row,
                    "prep_unit_checksum": payload_checksum(prepared_row),
                })
                rows.append(prepared_row)
                selected_count += 1
                print("prepared", len(rows), country, study_split, "computed")
            if selected_count != needed:
                raise RuntimeError(
                    f"{country!r} has only {selected_count} OCR-clean images; "
                    f"the frozen design requires {needed}"
                )
        validation = validate_media_plan(rows)
        if not validation["passed"]:
            raise RuntimeError("prepared media failed validation: " + json.dumps(validation["problems"], indent=2))
        PREPARED = {
            "version": protocol["version"], "protocol": protocol,
            "dataset_revision": dataset_revision, "rows": rows,
            "population_digest": validation["population_digest"],
            "media_validation": validation,
            "confirmation_outputs_opened": False,
        }
        PREPARED["prepared_checksum"] = payload_checksum(PREPARED)
        _atomic_json(PREPARED_POPULATION_PATH, PREPARED)
        _atomic_json(PREPARATION_COMPLETE_PATH, {
            "prepared_checksum": PREPARED["prepared_checksum"],
            "population_digest": PREPARED["population_digest"],
            "dataset_revision": dataset_revision,
        })
        print("Stage 0 completed", PREPARED_POPULATION_PATH)
else:
    if not PREPARED_POPULATION_PATH.is_file():
        if ANY_STAGE:
            raise RuntimeError("Run Stage 0 on CPU first; no prepared population exists")
        disabled_protocol = benchmark_spec(dataset_revision="disabled")
        PREPARED = {
            "version": disabled_protocol["version"],
            "protocol": disabled_protocol,
            "dataset_revision": "disabled",
            "rows": [],
            "population_digest": payload_checksum([]),
            "media_validation": {"passed": False, "population_digest": payload_checksum([])},
            "confirmation_outputs_opened": False,
        }
        PREPARED["prepared_checksum"] = payload_checksum(PREPARED)
        print("workflow disabled: no data, model, or scientific result opened")
    else:
        PREPARED = json.loads(PREPARED_POPULATION_PATH.read_text(encoding="utf-8"))
print("dataset revision", PREPARED["dataset_revision"])
print("population digest", PREPARED["population_digest"])
print("media validation", PREPARED["media_validation"]["passed"])
'''
)


def build() -> dict:
    blocks: list[tuple[int, list[tuple[str, str]]]] = []
    current: list[tuple[str, str]] = []
    current_order = -1
    for kind, source in CELLS:
        if kind == "markdown":
            first = source.splitlines()[0].strip()
            if first.startswith("## ") and first[3:4].isdigit():
                if current:
                    blocks.append((current_order, current))
                current = []
                current_order = int(first[3:].split(".", 1)[0])
        current.append((kind, source))
    if current:
        blocks.append((current_order, current))
    ordered = [
        item
        for _order, block in sorted(blocks, key=lambda row: row[0])
        for item in block
    ]
    cells = []
    for kind, source in ordered:
        cell = {
            "cell_type": kind,
            "metadata": {},
            "source": [line + "\n" for line in source.splitlines()],
        }
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"gpuType": "A100", "provenance": []},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.x"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    TARGET.write_text(json.dumps(build(), indent=1), encoding="utf-8")
    print(TARGET)
