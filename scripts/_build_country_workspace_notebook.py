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
* The inherited animal-domain prompt is never used. A country-specific prompt
  is checksum-bound before any corrected output is generated.
* The existing pooled lens must first identify the source country from clean
  activations. If either that readout or a direct country-identity exchange
  fails, the notebook may fit one task-matched pooled multimodal J-lens on the
  already frozen 99-example fit split and repeat the same validation.
* The layer band is selected only by whether an exact country-identity exchange
  changes the generated country name across two independent pairs. Capital and
  continent outputs remain hidden until that band is frozen.
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

def _verified_parent_report(name, expected_checksum):
    path = PARENT_V2_RUN_DIR / name
    if not path.is_file():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    body = {key: value for key, value in report.items() if key != "report_checksum"}
    computed_checksum = payload_checksum(body)
    if (
        report.get("report_checksum") != expected_checksum
        or computed_checksum != expected_checksum
    ):
        raise RuntimeError(
            f"refusing changed parent report {path}: "
            f"recorded={report.get('report_checksum')} "
            f"computed={computed_checksum} expected={expected_checksum}"
        )
    return report

if CAPABILITY_PATH.is_file():
    CAPABILITY_REPORT = json.loads(CAPABILITY_PATH.read_text(encoding="utf-8"))
if LOCALIZATION_PATH.is_file():
    LOCALIZATION_REPORT = json.loads(LOCALIZATION_PATH.read_text(encoding="utf-8"))

if CAPABILITY_REPORT is None and RUN_STAGE2_CAPABILITY_AND_LOCALIZATION:
    CAPABILITY_REPORT = _verified_parent_report(
        "country_capability_report.json", PARENT_CAPABILITY_CHECKSUM
    )
    if CAPABILITY_REPORT is not None:
        _write_report(CAPABILITY_PATH, CAPABILITY_REPORT)
        print("reused checksum-pinned parent capability", CAPABILITY_REPORT["verdict"])
if LOCALIZATION_REPORT is None and RUN_STAGE2_CAPABILITY_AND_LOCALIZATION:
    LOCALIZATION_REPORT = _verified_parent_report(
        "country_direct_answer_localization_report.json",
        PARENT_LOCALIZATION_CHECKSUM,
    )
    if LOCALIZATION_REPORT is not None:
        _write_report(LOCALIZATION_PATH, LOCALIZATION_REPORT)
        print("reused checksum-pinned parent localization", LOCALIZATION_REPORT["verdict"])

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

    if CAPABILITY_REPORT.get("generalization_ready"):
        eligible_direction_names = set(CAPABILITY_REPORT["eligible_directions"])
        tokens = concept_tokens((*EVAL_COUNTRIES, *CONTROL_COUNTRIES))
        unembed = BACKEND.unembedding_weight()
        localization_rows = []
        for property_name in PROPERTIES:
            for source, target in DIRECTIONS:
                if f"{source}->{target}" not in eligible_direction_names:
                    continue
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
        print("Localization not run: clean source capability did not cover two pairs.")
elif CAPABILITY_REPORT is not None:
    print("reused capability", CAPABILITY_REPORT["verdict"])
    if LOCALIZATION_REPORT is not None:
        print("reused localization", LOCALIZATION_REPORT["verdict"])
'''
)

markdown("## 9.5 Corrected country prompt, lens validation, and identity-band calibration")
code(
    r'''
CORRECTED_CAPABILITY_PATH = RUN_DIR / "country_corrected_prompt_capability_report.json"
LENS_VALIDATION_PATH = RUN_DIR / "country_identity_lens_validation_report.json"
IDENTITY_CALIBRATION_PATH = RUN_DIR / "country_identity_band_calibration_report.json"
ACTIVE_LENS_PATH = RUN_DIR / "country_active_lens.json"
TASK_LENS_PATH = RUN_DIR / "lenses" / "lens.pooled.country_identity.l16_l40.pt"
TASK_LENS_PROVENANCE_PATH = RUN_DIR / "lenses" / "lens.pooled.country_identity.l16_l40.json"
BALANCED_LENS_PATH = RUN_DIR / "lenses" / "lens.pooled.country_balanced_tasks.l16_l40.pt"
BALANCED_LENS_PROVENANCE_PATH = RUN_DIR / "lenses" / "lens.pooled.country_balanced_tasks.l16_l40.json"

LENS_VALIDATION_REPORT = None
IDENTITY_CALIBRATION_REPORT = None
ACTIVE_LENS = LENS
ACTIVE_LENS_LABEL = "original_pooled_j"
if CORRECTED_CAPABILITY_PATH.is_file():
    CAPABILITY_REPORT = json.loads(CORRECTED_CAPABILITY_PATH.read_text(encoding="utf-8"))
if LENS_VALIDATION_PATH.is_file():
    LENS_VALIDATION_REPORT = json.loads(LENS_VALIDATION_PATH.read_text(encoding="utf-8"))
if IDENTITY_CALIBRATION_PATH.is_file():
    IDENTITY_CALIBRATION_REPORT = json.loads(IDENTITY_CALIBRATION_PATH.read_text(encoding="utf-8"))
if MODEL_STAGE and ACTIVE_LENS_PATH.is_file():
    _active = json.loads(ACTIVE_LENS_PATH.read_text(encoding="utf-8"))
    ACTIVE_LENS_LABEL = _active["label"]
    if ACTIVE_LENS_LABEL == "task_matched_pooled_j":
        from jlens.lens import JacobianLens
        from jlens.mmpilot.backend import file_checksum
        if file_checksum(str(TASK_LENS_PATH)) != _active["lens_checksum"]:
            raise RuntimeError("task-matched active lens checksum changed")
        ACTIVE_LENS = JacobianLens.load(str(TASK_LENS_PATH))
    elif ACTIVE_LENS_LABEL == "balanced_task_pooled_j":
        from jlens.lens import JacobianLens
        from jlens.mmpilot.backend import file_checksum
        if file_checksum(str(BALANCED_LENS_PATH)) != _active["lens_checksum"]:
            raise RuntimeError("balanced-task active lens checksum changed")
        ACTIVE_LENS = JacobianLens.load(str(BALANCED_LENS_PATH))

if RUN_STAGE2_DEBUG_COUNTRY_INSTRUMENT or RUN_STAGE2C_FIT_BALANCED_TASK_LENS:
    if not (
        MODEL_ENABLED and CONFIRM_IDENTITY_CALIBRATION_BUDGET and LENS is not None
    ):
        raise RuntimeError(
            "corrected country debugging requires the model, existing lens, "
            "and identity-calibration budget confirmation"
        )
    import math
    import torch
    from jlens.hooks import ActivationRecorder
    from jlens.lens import JacobianLens
    from jlens.mmpilot.backend import file_checksum
    from jlens.mmpilot.country_workspace import (
        capability_report, identity_band_calibration_report,
        identity_lens_validation_report,
    )
    from jlens.mmpilot.multimodal_lens import FitUnit, fit_arm

    development_rows = sorted(
        [row for row in MEDIA_ROWS if row["study_split"] == "development"],
        key=lambda row: row["unit_id"],
    )

    # The old capability report used the inherited animal instruction and is
    # never reused. These are fresh outputs under COUNTRY_COMPLETION_INSTRUCTION.
    capability_rows = []
    for row in development_rows:
        for modality in MODALITIES:
            for property_name in ("identity", *PROPERTIES):
                expected = row["country"] if property_name == "identity" else fact(row["country"], property_name)
                key = safe_key("country_v4_capability", row["unit_id"], modality, property_name)
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
                        "country_instruction": COUNTRY_COMPLETION_INSTRUCTION,
                    }
                    STORE.save("capability", key, stored)
                    work = "computed"
                else:
                    work = "reused"
                capability_rows.append(stored)
                if len(capability_rows) == 1 or len(capability_rows) % 24 == 0:
                    print("corrected capability", len(capability_rows), work)
    CAPABILITY_REPORT = capability_report(capability_rows)
    _capability_body = {
        key: value for key, value in CAPABILITY_REPORT.items()
        if key != "report_checksum"
    }
    _capability_body.update({
        "country_instruction": COUNTRY_COMPLETION_INSTRUCTION,
        "supersedes_animal_instruction_run": True,
    })
    CAPABILITY_REPORT = {
        **_capability_body, "report_checksum": payload_checksum(_capability_body)
    }
    _write_report(CORRECTED_CAPABILITY_PATH, CAPABILITY_REPORT)
    print("corrected capability verdict", CAPABILITY_REPORT["verdict"])

    candidate_names = (*EVAL_COUNTRIES, *CONTROL_COUNTRIES)
    candidate_tokens = concept_tokens(candidate_names)
    unembed = BACKEND.unembedding_weight()

    def validate_identity_lens(lens, label):
        vectors = {
            layer: {
                country: selected_lens_vector(
                    lens, unembed, layer=layer,
                    token_id=candidate_tokens[country].token_id,
                )
                for country in candidate_names
            }
            for layer in LAYERS
        }
        eligible_sources = set(CAPABILITY_REPORT["eligible_countries"])
        rows = []
        source_rows = [
            row for row in development_rows if row["country"] in eligible_sources
        ]
        for row in source_rows:
            for modality in MODALITIES:
                key = safe_key("country_lens_readout", label, row["unit_id"], modality)
                stored = STORE.load("activation", key)
                if stored is None:
                    inputs = build_task_inputs(row, modality, "identity")
                    with ActivationRecorder(BACKEND.blocks, at=LAYERS) as recorder:
                        BACKEND.forward_logits(inputs.tensors)
                    per_layer = []
                    for layer in LAYERS:
                        h = recorder.activations[layer][0, inputs.final_prompt_position].detach().float().cpu()
                        scores = {
                            country: float(h.dot(vectors[layer][country]))
                            for country in candidate_names
                        }
                        source_score = scores[row["country"]]
                        others = [
                            score for country, score in scores.items()
                            if country != row["country"]
                        ]
                        per_layer.append({
                            "layer": layer,
                            "scores": scores,
                            "source_is_sole_top1": bool(
                                source_score > max(others, default=-math.inf)
                            ),
                            "all_finite": all(math.isfinite(score) for score in scores.values()),
                        })
                    stored = {
                        "unit_id": row["unit_id"], "country": row["country"],
                        "modality": modality, "lens_label": label,
                        "per_layer": per_layer,
                    }
                    STORE.save("activation", key, stored)
                    work = "computed"
                else:
                    work = "reused"
                rows.extend({
                    "unit_id": stored["unit_id"], "country": stored["country"],
                    "modality": stored["modality"], "lens_label": label, **item,
                } for item in stored["per_layer"])
                if len(rows) == len(LAYERS) or len(rows) % (len(LAYERS) * 12) == 0:
                    print("lens identity readout", label, len(rows), work)
        expected_n = len(source_rows)
        report = identity_lens_validation_report(
            rows, expected_n_per_modality=expected_n,
        )
        _body = {key: value for key, value in report.items() if key != "report_checksum"}
        _body["lens_label"] = label
        _body["lens_checksum"] = (
            file_checksum(str(BALANCED_LENS_PATH))
            if label == "balanced_task_pooled_j"
            else file_checksum(str(TASK_LENS_PATH))
            if label == "task_matched_pooled_j"
            else PARENT_LENS_CHECKSUM
        )
        return {**_body, "report_checksum": payload_checksum(_body)}

    def calibrate_identity_band(lens, label):
        eligible_direction_names = set(CAPABILITY_REPORT["eligible_directions"])
        rows = []
        for source, target in DIRECTIONS:
            direction = f"{source}->{target}"
            if direction not in eligible_direction_names:
                continue
            source_rows = [row for row in development_rows if row["country"] == source]
            for band in PATH_BANDS:
                exact = build_swap_bases_for_lens(
                    lens, unembed, layers=band,
                    source=candidate_tokens[source], target=candidate_tokens[target],
                )
                random_bases = {
                    layer: random_two_direction_basis(
                        basis,
                        seed=RANDOM_SEED + layer + 1000 * DIRECTIONS.index((source, target)),
                    )
                    for layer, basis in exact.items()
                }
                unrelated = build_swap_bases_for_lens(
                    lens, unembed, layers=band,
                    source=candidate_tokens[CONTROL_COUNTRIES[0]],
                    target=candidate_tokens[CONTROL_COUNTRIES[1]],
                )
                for row in source_rows:
                    for modality in MODALITIES:
                        for condition, alpha, bases in (
                            ("exact", 1.0, exact), ("zero", 0.0, exact),
                            ("random", 1.0, random_bases),
                            ("unrelated", 1.0, unrelated),
                        ):
                            key = safe_key(
                                "country_identity_calibration", label, source, target,
                                band[0], band[-1], row["unit_id"], modality, condition,
                            )
                            stored = STORE.load("intervention", key)
                            if stored is None:
                                inputs = build_task_inputs(row, modality, "identity")
                                result = unrestricted_greedy_swap_trial(
                                    BACKEND, inputs, bases=bases, alpha=alpha,
                                    answer=target, max_new_tokens=MAX_NEW_TOKENS,
                                    position_rule="all_prompt_positions",
                                    realization_policy=TEXT_MODEL_DTYPE_REALIZATION,
                                )
                                stored = {
                                    **result, "unit_id": row["unit_id"],
                                    "source": source, "target": target,
                                    "direction": direction, "modality": modality,
                                    "condition": condition, "expected": target,
                                    "success": identity_matches(result["generated_text"], target),
                                    "integrity_pass": diagnostic_integrity(
                                        result, exact=(condition == "exact")
                                    ),
                                }
                                STORE.save("intervention", key, stored)
                                work = "computed"
                            else:
                                work = "reused"
                            rows.append(stored)
                            if len(rows) == 1 or len(rows) % 96 == 0:
                                print("identity calibration", label, len(rows), work)
        report = identity_band_calibration_report(
            rows,
            eligible_directions=sorted(eligible_direction_names),
            expected_n=N_DEVELOPMENT_PER_COUNTRY,
        )
        _body = {key: value for key, value in report.items() if key != "report_checksum"}
        _body["lens_label"] = label
        _body["lens_checksum"] = (
            file_checksum(str(BALANCED_LENS_PATH))
            if label == "balanced_task_pooled_j"
            else file_checksum(str(TASK_LENS_PATH))
            if label == "task_matched_pooled_j"
            else PARENT_LENS_CHECKSUM
        )
        return {**_body, "report_checksum": payload_checksum(_body)}

    ACTIVE_LENS = LENS
    ACTIVE_LENS_LABEL = "original_pooled_j"
    IDENTITY_CALIBRATION_REPORT = None
    LENS_VALIDATION_REPORT = validate_identity_lens(ACTIVE_LENS, ACTIVE_LENS_LABEL)
    _write_report(
        RUN_DIR / "country_identity_lens_validation.original_pooled_j.json",
        LENS_VALIDATION_REPORT,
    )
    if LENS_VALIDATION_REPORT["verdict"] == "COUNTRY_IDENTITY_LENS_VALIDATION_GO":
        IDENTITY_CALIBRATION_REPORT = calibrate_identity_band(
            ACTIVE_LENS, ACTIVE_LENS_LABEL
        )
        _write_report(
            RUN_DIR / "country_identity_band_calibration.original_pooled_j.json",
            IDENTITY_CALIBRATION_REPORT,
        )

    needs_refit = (
        LENS_VALIDATION_REPORT["verdict"] != "COUNTRY_IDENTITY_LENS_VALIDATION_GO"
        or IDENTITY_CALIBRATION_REPORT is None
        or IDENTITY_CALIBRATION_REPORT["verdict"]
        != "COUNTRY_IDENTITY_BAND_CALIBRATION_GO"
    )
    if needs_refit and RUN_STAGE2_REFIT_TASK_MATCHED_LENS_IF_NEEDED:
        if not CONFIRM_TASK_MATCHED_REFIT_BUDGET:
            raise RuntimeError("task-matched refit was requested without budget confirmation")
        fit_rows = sorted(
            [row for row in MEDIA_ROWS if row["study_split"] == "fit"],
            key=lambda row: row["unit_id"],
        )
        units = [
            FitUnit(
                unit_id=f"{row['unit_id']}:{modality}:country_identity",
                group_id=row["unit_id"], image_id=row["image_checksum"],
                modality=modality, caption=text_evidence(row["country"]),
                image_path=row["image_path"], audio_path=row["audio_path"],
                prompt="country_identity_task_matched_assistant_prefill",
            )
            for row in fit_rows for modality in MODALITIES
        ]
        if len(units) != 99:
            raise RuntimeError(f"task-matched country fit requires 99 units, got {len(units)}")
        def task_fit_inputs(unit):
            row = next(item for item in fit_rows if item["unit_id"] == unit.group_id)
            return build_task_inputs(row, unit.modality, "identity")
        checkpoint = RUN_DIR / "lenses" / "checkpoints" / "country_identity.jacobian_sum.pt"
        if TASK_LENS_PATH.is_file():
            ACTIVE_LENS = JacobianLens.load(str(TASK_LENS_PATH))
        else:
            ACTIVE_LENS = fit_arm(
                BACKEND, units, build_inputs=task_fit_inputs,
                source_layers=LAYERS, target_layer=TARGET_LAYER,
                checkpoint_path=checkpoint, arm="pooled",
                scientific_fingerprint=payload_checksum({
                    "base": SCIENTIFIC_DIGEST,
                    "fit": "country_identity_task_matched_assistant_prefill",
                }),
                dim_batch=DIM_BATCH, skip_first=SKIP_FIRST,
                checkpoint_every=CHECKPOINT_EVERY,
                progress=lambda info: print(
                    "task-matched fit", info["index"], "/", info["total"],
                    info["modality"], "checkpoint", info["checkpoint_written"],
                ) if (
                    info["index"] == 1 or info["checkpoint_written"]
                    or info["index"] == info["total"]
                ) else None,
            )
            TASK_LENS_PATH.parent.mkdir(parents=True, exist_ok=True)
            temporary = TASK_LENS_PATH.with_suffix(".tmp.pt")
            ACTIVE_LENS.save(str(temporary), dtype=torch.float32)
            os.replace(temporary, TASK_LENS_PATH)
        task_provenance = {
            "lens_checksum": file_checksum(str(TASK_LENS_PATH)),
            "source_layers": list(LAYERS), "target_layer": TARGET_LAYER,
            "n_prompts": 99, "stored_dtype": "float32",
            "fit_prompt": "country_identity_task_matched_assistant_prefill",
            "country_instruction": COUNTRY_COMPLETION_INSTRUCTION,
        }
        task_provenance["provenance_checksum"] = payload_checksum(task_provenance)
        _write_report(TASK_LENS_PROVENANCE_PATH, task_provenance)
        ACTIVE_LENS_LABEL = "task_matched_pooled_j"
        IDENTITY_CALIBRATION_REPORT = None
        LENS_VALIDATION_REPORT = validate_identity_lens(ACTIVE_LENS, ACTIVE_LENS_LABEL)
        _write_report(
            RUN_DIR / "country_identity_lens_validation.task_matched_pooled_j.json",
            LENS_VALIDATION_REPORT,
        )
        if LENS_VALIDATION_REPORT["verdict"] == "COUNTRY_IDENTITY_LENS_VALIDATION_GO":
            IDENTITY_CALIBRATION_REPORT = calibrate_identity_band(
                ACTIVE_LENS, ACTIVE_LENS_LABEL
            )
            _write_report(
                RUN_DIR / "country_identity_band_calibration.task_matched_pooled_j.json",
                IDENTITY_CALIBRATION_REPORT,
            )

    if RUN_STAGE2C_FIT_BALANCED_TASK_LENS:
        if not CONFIRM_BALANCED_TASK_FINAL_FIT_BUDGET:
            raise RuntimeError(
                "the final balanced-task fit was requested without budget confirmation"
            )
        fit_rows = sorted(
            [row for row in MEDIA_ROWS if row["study_split"] == "fit"],
            key=lambda row: row["unit_id"],
        )
        task_names = ("identity", "capital", "continent")
        balanced_plan = []
        for row_index, row in enumerate(fit_rows):
            for modality_index, modality in enumerate(MODALITIES):
                property_name = task_names[(row_index + modality_index) % len(task_names)]
                balanced_plan.append((row, modality, property_name))
        balance = {
            modality: {
                property_name: sum(
                    1 for _row, item_modality, item_property in balanced_plan
                    if item_modality == modality and item_property == property_name
                )
                for property_name in task_names
            }
            for modality in MODALITIES
        }
        if len(balanced_plan) != 99 or any(
            count != 11 for per_modality in balance.values()
            for count in per_modality.values()
        ):
            raise RuntimeError(
                f"the final fit is not exactly balanced across tasks/modalities: {balance}"
            )
        units = [
            FitUnit(
                unit_id=f"{row['unit_id']}:{modality}:{property_name}",
                group_id=row["unit_id"], image_id=row["image_checksum"],
                modality=modality, caption=text_evidence(row["country"]),
                image_path=row["image_path"], audio_path=row["audio_path"],
                prompt=f"country_balanced_task:{property_name}",
            )
            for row, modality, property_name in balanced_plan
        ]
        plan_by_unit = {
            unit.unit_id: (row, property_name)
            for unit, (row, _modality, property_name) in zip(units, balanced_plan)
        }
        def balanced_fit_inputs(unit):
            row, property_name = plan_by_unit[unit.unit_id]
            return build_task_inputs(row, unit.modality, property_name)
        checkpoint = (
            RUN_DIR / "lenses" / "checkpoints"
            / "country_balanced_tasks.jacobian_sum.pt"
        )
        if BALANCED_LENS_PATH.is_file():
            ACTIVE_LENS = JacobianLens.load(str(BALANCED_LENS_PATH))
            print("final balanced-task lens reused; no fitting performed")
        else:
            ACTIVE_LENS = fit_arm(
                BACKEND, units, build_inputs=balanced_fit_inputs,
                source_layers=LAYERS, target_layer=TARGET_LAYER,
                checkpoint_path=checkpoint, arm="pooled",
                scientific_fingerprint=payload_checksum({
                    "base": SCIENTIFIC_DIGEST,
                    "fit": "country_balanced_identity_capital_continent.v1",
                    "balance": balance,
                    "n_prompts": len(units),
                    "no_further_fit_fallback": True,
                }),
                dim_batch=DIM_BATCH, skip_first=SKIP_FIRST,
                checkpoint_every=CHECKPOINT_EVERY,
                progress=lambda info: print(
                    "FINAL balanced fit", info["index"], "/", info["total"],
                    info["modality"], "checkpoint", info["checkpoint_written"],
                ) if (
                    info["index"] == 1 or info["checkpoint_written"]
                    or info["index"] == info["total"]
                ) else None,
            )
            BALANCED_LENS_PATH.parent.mkdir(parents=True, exist_ok=True)
            temporary = BALANCED_LENS_PATH.with_suffix(".tmp.pt")
            ACTIVE_LENS.save(str(temporary), dtype=torch.float32)
            os.replace(temporary, BALANCED_LENS_PATH)
        balanced_provenance = {
            "lens_checksum": file_checksum(str(BALANCED_LENS_PATH)),
            "source_layers": list(LAYERS), "target_layer": TARGET_LAYER,
            "n_prompts": 99, "stored_dtype": "float32",
            "fit_tasks": list(task_names), "per_modality_task_counts": balance,
            "fit_countries": sorted({row["country"] for row in fit_rows}),
            "excluded_evaluation_countries": list(EVAL_COUNTRIES),
            "checkpoint_every": CHECKPOINT_EVERY,
            "no_further_fit_fallback": True,
        }
        balanced_provenance["provenance_checksum"] = payload_checksum(
            balanced_provenance
        )
        _write_report(BALANCED_LENS_PROVENANCE_PATH, balanced_provenance)
        ACTIVE_LENS_LABEL = "balanced_task_pooled_j"
        IDENTITY_CALIBRATION_REPORT = None
        LENS_VALIDATION_REPORT = validate_identity_lens(
            ACTIVE_LENS, ACTIVE_LENS_LABEL
        )
        _write_report(
            RUN_DIR / "country_identity_lens_validation.balanced_task_pooled_j.json",
            LENS_VALIDATION_REPORT,
        )
        if LENS_VALIDATION_REPORT["verdict"] == "COUNTRY_IDENTITY_LENS_VALIDATION_GO":
            IDENTITY_CALIBRATION_REPORT = calibrate_identity_band(
                ACTIVE_LENS, ACTIVE_LENS_LABEL
            )
            _write_report(
                RUN_DIR / "country_identity_band_calibration.balanced_task_pooled_j.json",
                IDENTITY_CALIBRATION_REPORT,
            )
        print("FINAL FIT COMPLETE; NO ADDITIONAL FIT FALLBACK IS IMPLEMENTED")

    _write_report(LENS_VALIDATION_PATH, LENS_VALIDATION_REPORT)
    if IDENTITY_CALIBRATION_REPORT is not None:
        _write_report(IDENTITY_CALIBRATION_PATH, IDENTITY_CALIBRATION_REPORT)
    active_record = {
        "label": ACTIVE_LENS_LABEL,
        "lens_path": str(
            BALANCED_LENS_PATH
            if ACTIVE_LENS_LABEL == "balanced_task_pooled_j"
            else TASK_LENS_PATH
            if ACTIVE_LENS_LABEL == "task_matched_pooled_j"
            else LENS_PATH
        ),
        "lens_checksum": (
            file_checksum(str(BALANCED_LENS_PATH))
            if ACTIVE_LENS_LABEL == "balanced_task_pooled_j"
            else file_checksum(str(TASK_LENS_PATH))
            if ACTIVE_LENS_LABEL == "task_matched_pooled_j"
            else PARENT_LENS_CHECKSUM
        ),
        "lens_validation_checksum": LENS_VALIDATION_REPORT["report_checksum"],
        "identity_calibration_checksum": (
            IDENTITY_CALIBRATION_REPORT.get("report_checksum")
            if IDENTITY_CALIBRATION_REPORT else None
        ),
    }
    active_record["record_checksum"] = payload_checksum(active_record)
    _write_report(ACTIVE_LENS_PATH, active_record)
    print("identity lens verdict", LENS_VALIDATION_REPORT["verdict"])
    print("eligible identity layers", LENS_VALIDATION_REPORT["eligible_layers"])
    print(
        "identity calibration verdict",
        IDENTITY_CALIBRATION_REPORT["verdict"] if IDENTITY_CALIBRATION_REPORT else "NOT_RUN",
    )
    print(
        "selected identity band",
        IDENTITY_CALIBRATION_REPORT.get("selected") if IDENTITY_CALIBRATION_REPORT else None,
    )
elif LENS_VALIDATION_REPORT is not None:
    print("reused identity lens validation", LENS_VALIDATION_REPORT["verdict"])
    if IDENTITY_CALIBRATION_REPORT is not None:
        print("reused identity calibration", IDENTITY_CALIBRATION_REPORT["verdict"])

LENS = ACTIVE_LENS
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
    if not CAPABILITY_REPORT.get("generalization_ready"):
        print("DEVELOPMENT NOT LICENSED: clean source capability did not cover two pairs")
    elif LENS_VALIDATION_REPORT is None or (
        LENS_VALIDATION_REPORT.get("verdict")
        != "COUNTRY_IDENTITY_LENS_VALIDATION_GO"
    ):
        print("DEVELOPMENT NOT LICENSED: country identity lens validation did not pass")
    elif IDENTITY_CALIBRATION_REPORT is None or (
        IDENTITY_CALIBRATION_REPORT.get("verdict")
        != "COUNTRY_IDENTITY_BAND_CALIBRATION_GO"
    ):
        print("DEVELOPMENT NOT LICENSED: identity-swap band calibration did not pass")
    else:
        from jlens.mmpilot.country_workspace import (
            causal_report, freeze_identity_calibrated_confirmation_design,
        )
        development_rows = sorted(
            [row for row in MEDIA_ROWS if row["study_split"] == "development"],
            key=lambda row: row["unit_id"],
        )
        tokens = concept_tokens((*EVAL_COUNTRIES, *CONTROL_COUNTRIES))
        unembed = BACKEND.unembedding_weight()
        trial_rows = []
        eligible_direction_names = set(CAPABILITY_REPORT["eligible_directions"])
        selected_identity_band = tuple(
            IDENTITY_CALIBRATION_REPORT["selected"]["band"]
        )
        for property_name in PROPERTIES:
            band = selected_identity_band
            unrelated = build_swap_bases_for_lens(
                LENS, unembed, layers=band,
                source=tokens[CONTROL_COUNTRIES[0]], target=tokens[CONTROL_COUNTRIES[1]],
            )
            for source, target in DIRECTIONS:
                if f"{source}->{target}" not in eligible_direction_names:
                    continue
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
        if DEVELOPMENT_REPORT["generalized_across_two_pairs"]:
            design = freeze_identity_calibrated_confirmation_design(
                protocol=PROTOCOL, media_validation=PREPARED["media_validation"],
                capability=CAPABILITY_REPORT,
                lens_validation=LENS_VALIDATION_REPORT,
                identity_calibration=IDENTITY_CALIBRATION_REPORT,
                development=DEVELOPMENT_REPORT,
            )
            _write_report(DESIGN_PATH, design)
            print("confirmation design frozen", design["design_checksum"])
        else:
            print(
                "confirmation remains unopened: development did not pass both "
                "properties across two independent pairs"
            )
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

markdown("## 12. Prospective France-to-China downstream follow-up")
code(
    r'''
FRANCE_CHINA_DEVELOPMENT_PATH = RUN_DIR / "france_china_downstream_development_report.json"
FRANCE_CHINA_DESIGN_PATH = RUN_DIR / "france_china_confirmation_design.json"
FRANCE_CHINA_CONFIRMATION_PATH = RUN_DIR / "france_china_fresh_confirmation_report.json"
FRANCE_CHINA_DEVELOPMENT = (
    json.loads(FRANCE_CHINA_DEVELOPMENT_PATH.read_text(encoding="utf-8"))
    if FRANCE_CHINA_DEVELOPMENT_PATH.is_file() else None
)
FRANCE_CHINA_CONFIRMATION = (
    json.loads(FRANCE_CHINA_CONFIRMATION_PATH.read_text(encoding="utf-8"))
    if FRANCE_CHINA_CONFIRMATION_PATH.is_file() else None
)
from jlens.mmpilot.backend import file_checksum

def _france_china_trials(rows, *, stage, expected_n):
    from jlens.mmpilot.country_workspace import france_china_followup_report
    tokens = concept_tokens((*EVAL_COUNTRIES, *CONTROL_COUNTRIES))
    unembed = BACKEND.unembedding_weight()
    exact = build_swap_bases_for_lens(
        LENS, unembed, layers=LAYERS, source=tokens["France"], target=tokens["China"]
    )
    random_bases = {
        layer: random_two_direction_basis(basis, seed=RANDOM_SEED + 70000 + layer)
        for layer, basis in exact.items()
    }
    unrelated = build_swap_bases_for_lens(
        LENS, unembed, layers=LAYERS,
        source=tokens[CONTROL_COUNTRIES[0]], target=tokens[CONTROL_COUNTRIES[1]],
    )
    trial_rows = []
    active_lens_checksum = file_checksum(str(BALANCED_LENS_PATH))
    for property_name in PROPERTIES:
        expected = fact("China", property_name)
        for row in rows:
            for modality in MODALITIES:
                for condition, alpha, bases in (
                    ("exact", 1.0, exact), ("zero", 0.0, exact),
                    ("random", 1.0, random_bases), ("unrelated", 1.0, unrelated),
                ):
                    key = safe_key(
                        "france_china_followup", stage, property_name,
                        ACTIVE_LENS_LABEL, active_lens_checksum,
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
                            "source": "France", "target": "China",
                            "direction": "France->China", "property": property_name,
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
                        print("France-to-China", stage, len(trial_rows), work)
    return france_china_followup_report(
        trial_rows, stage=stage, expected_n=expected_n
    )

if RUN_STAGE3B_FRANCE_CHINA_DOWNSTREAM_DEVELOPMENT:
    if not (
        MODEL_ENABLED and CONFIRM_FRANCE_CHINA_DEVELOPMENT_BUDGET
        and LENS is not None
    ):
        raise RuntimeError(
            "France-to-China development requires the model, existing task-matched "
            "lens, and its budget confirmation"
        )
    if ACTIVE_LENS_LABEL != "balanced_task_pooled_j":
        raise RuntimeError("the follow-up is pinned to the final balanced-task pooled J-lens")
    if LENS_VALIDATION_REPORT is None or IDENTITY_CALIBRATION_REPORT is None:
        raise RuntimeError("the follow-up requires the stored lens-validation artifacts")
    full_band = next(
        (candidate for candidate in IDENTITY_CALIBRATION_REPORT.get("candidates", ())
         if tuple(candidate.get("band", ())) == LAYERS), None,
    )
    if full_band is None or "France->China" not in full_band.get("passing_directions", ()):
        raise RuntimeError(
            "the stored calibration does not pin France-to-China at L16-L40"
        )
    development_rows = sorted(
        [row for row in MEDIA_ROWS if row["study_split"] == "development" and row["country"] == "France"],
        key=lambda row: row["unit_id"],
    )
    FRANCE_CHINA_DEVELOPMENT = _france_china_trials(
        development_rows, stage="development", expected_n=N_DEVELOPMENT_PER_COUNTRY
    )
    _write_report(FRANCE_CHINA_DEVELOPMENT_PATH, FRANCE_CHINA_DEVELOPMENT)
    print("FRANCE-TO-CHINA DEVELOPMENT", FRANCE_CHINA_DEVELOPMENT["verdict"])
    print("report", FRANCE_CHINA_DEVELOPMENT_PATH)
    if FRANCE_CHINA_DEVELOPMENT["verdict"] == "COUNTRY_FRANCE_CHINA_DEVELOPMENT_GO":
        from jlens.mmpilot.country_workspace import freeze_france_china_followup_design
        design = freeze_france_china_followup_design(
            protocol=PROTOCOL, media_validation=PREPARED["media_validation"],
            capability=CAPABILITY_REPORT, lens_validation=LENS_VALIDATION_REPORT,
            identity_calibration=IDENTITY_CALIBRATION_REPORT,
            lens_checksum=file_checksum(str(BALANCED_LENS_PATH)),
            development=FRANCE_CHINA_DEVELOPMENT,
        )
        _write_report(FRANCE_CHINA_DESIGN_PATH, design)
        print("fresh confirmation licensed", design["design_checksum"])
    else:
        print("fresh confirmation remains unopened")
elif FRANCE_CHINA_DEVELOPMENT is not None:
    print("reused France-to-China development", FRANCE_CHINA_DEVELOPMENT["verdict"])

if RUN_STAGE4B_FRANCE_CHINA_FRESH_CONFIRMATION:
    if not (
        MODEL_ENABLED and CONFIRM_FRANCE_CHINA_CONFIRMATION_BUDGET
        and LENS is not None
    ):
        raise RuntimeError(
            "France-to-China confirmation requires the model, existing lens, "
            "and its budget confirmation"
        )
    if not FRANCE_CHINA_DESIGN_PATH.is_file():
        print("FRANCE-TO-CHINA CONFIRMATION NOT LICENSED: no frozen design")
    else:
        design = json.loads(FRANCE_CHINA_DESIGN_PATH.read_text(encoding="utf-8"))
        confirmation_rows = sorted(
            [row for row in MEDIA_ROWS if row["study_split"] == "confirmation" and row["country"] == "France"],
            key=lambda row: row["unit_id"],
        )
        clean_rows = []
        for property_name in PROPERTIES:
            expected = fact("France", property_name)
            for row in confirmation_rows:
                for modality in MODALITIES:
                    key = safe_key("france_china_capability", property_name, row["unit_id"], modality)
                    stored = STORE.load("capability", key)
                    if stored is None:
                        result = unrestricted_greedy_completion(
                            BACKEND, build_task_inputs(row, modality, property_name),
                            answer=expected, max_new_tokens=MAX_NEW_TOKENS,
                        )
                        stored = {
                            **result, "unit_id": row["unit_id"], "country": "France",
                            "property": property_name, "modality": modality,
                            "expected": expected,
                            "success": answer_matches(result["generated_text"], expected),
                        }
                        STORE.save("capability", key, stored)
                    clean_rows.append(stored)
        clean_cells = []
        for property_name in PROPERTIES:
            for modality in MODALITIES:
                cell = [row for row in clean_rows if row["property"] == property_name and row["modality"] == modality]
                successes = sum(bool(row["success"]) for row in cell)
                rate = successes / len(cell) if cell else 0.0
                clean_cells.append({
                    "property": property_name, "modality": modality,
                    "n": len(cell), "successes": successes, "rate": rate,
                    "passed": len(cell) == N_CONFIRMATION_PER_COUNTRY and rate >= MIN_CAPABILITY_RATE,
                })
        if not clean_cells or not all(cell["passed"] for cell in clean_cells):
            FRANCE_CHINA_CONFIRMATION = {
                "version": "country.france_china_followup.v1",
                "verdict": "COUNTRY_FRANCE_CHINA_CONFIRMATION_CAPABILITY_NO_GO",
                "design": design, "capability_rows": clean_rows,
                "capability_cells": clean_cells, "interventions_run": False,
                "parent_two_pair_verdict_unchanged": True,
            }
            FRANCE_CHINA_CONFIRMATION["report_checksum"] = payload_checksum(FRANCE_CHINA_CONFIRMATION)
        else:
            FRANCE_CHINA_CONFIRMATION = _france_china_trials(
                confirmation_rows, stage="confirmation",
                expected_n=N_CONFIRMATION_PER_COUNTRY,
            )
            FRANCE_CHINA_CONFIRMATION["design"] = design
            FRANCE_CHINA_CONFIRMATION["capability_rows"] = clean_rows
            FRANCE_CHINA_CONFIRMATION["capability_cells"] = clean_cells
            body = {k: v for k, v in FRANCE_CHINA_CONFIRMATION.items() if k != "report_checksum"}
            FRANCE_CHINA_CONFIRMATION["report_checksum"] = payload_checksum(body)
        _write_report(FRANCE_CHINA_CONFIRMATION_PATH, FRANCE_CHINA_CONFIRMATION)
        print("FRANCE-TO-CHINA FRESH CONFIRMATION", FRANCE_CHINA_CONFIRMATION["verdict"])
        print("report", FRANCE_CHINA_CONFIRMATION_PATH)
elif FRANCE_CHINA_CONFIRMATION is not None:
    print("reused France-to-China confirmation", FRANCE_CHINA_CONFIRMATION["verdict"])

print("PARENT TWO-PAIR COUNTRY VERDICT REMAINS UNCHANGED")
'''
)

markdown("## 13. Final report and handoff")
code(
    r'''
if RUN_STAGE5_WRITE_REPORT:
    FINAL = {
        "version": PROTOCOL["version"],
        "scientific_config": SCIENTIFIC_CONFIG,
        "protocol": PROTOCOL,
        "media_validation": PREPARED["media_validation"],
        "lens_path": str(
            BALANCED_LENS_PATH
            if ACTIVE_LENS_LABEL == "balanced_task_pooled_j"
            else TASK_LENS_PATH
            if ACTIVE_LENS_LABEL == "task_matched_pooled_j"
            else LENS_PATH
        ),
        "lens_exists": bool(LENS is not None),
        "capability": CAPABILITY_REPORT,
        "localization": LOCALIZATION_REPORT,
        "identity_lens_validation": LENS_VALIDATION_REPORT,
        "identity_band_calibration": IDENTITY_CALIBRATION_REPORT,
        "active_lens_label": ACTIVE_LENS_LABEL,
        "development": DEVELOPMENT_REPORT,
        "confirmation": CONFIRMATION_REPORT,
        "france_china_downstream_development": FRANCE_CHINA_DEVELOPMENT,
        "france_china_fresh_confirmation": FRANCE_CHINA_CONFIRMATION,
        "causal_site_screen": CAUSAL_SITE_SCREEN,
        "restricted_swap_development": RESTRICTED_SWAP_DEVELOPMENT,
        "headline_verdict": (
            CONFIRMATION_REPORT.get("verdict") if CONFIRMATION_REPORT
            else DEVELOPMENT_REPORT.get("verdict") if DEVELOPMENT_REPORT
            else IDENTITY_CALIBRATION_REPORT.get("verdict") if IDENTITY_CALIBRATION_REPORT
            else LENS_VALIDATION_REPORT.get("verdict") if LENS_VALIDATION_REPORT
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
print("Also send france_china_downstream_development_report.json and, if present,")
print("france_china_fresh_confirmation_report.json.")
print("For the no-refit diagnosis, send country_causal_site_plan.json,")
print("country_causal_site_screen_report.json and, if licensed,")
print("country_localized_development_report.json.")
'''
)
markdown("## 5. Open the fingerprinted run")
code(
    r'''
from jlens.mmpilot.store import RunFingerprint, UnitStore, payload_checksum, safe_key
BASE_PROTOCOL = PREPARED["protocol"]
PROTOCOL_AMENDMENT = {
    "version": "mmpilot.country_prompt_lens_band_debug.v4",
    "base_protocol_digest": BASE_PROTOCOL["protocol_digest"],
    "country_completion_instruction": COUNTRY_COMPLETION_INSTRUCTION,
    "band_selection_endpoint": "exact_country_identity_swap_only",
    "band_selection_candidates": [list(band) for band in PATH_BANDS],
    "downstream_facts_hidden_during_band_selection": True,
    "conditional_refit": (
        "one_task_matched_pooled_multimodal_j_lens_after_existing_lens_no_go"
    ),
}
PROTOCOL_AMENDMENT["amendment_digest"] = payload_checksum(PROTOCOL_AMENDMENT)
PROTOCOL = {
    **BASE_PROTOCOL,
    "effective_amendment": PROTOCOL_AMENDMENT,
    "effective_protocol_digest": payload_checksum({
        "base_protocol_digest": BASE_PROTOCOL["protocol_digest"],
        "amendment_digest": PROTOCOL_AMENDMENT["amendment_digest"],
    }),
}
SCIENTIFIC_CONFIG = {
    "study": PROTOCOL["version"],
    "base_protocol_digest": PROTOCOL["protocol_digest"],
    "protocol_amendment_digest": PROTOCOL_AMENDMENT["amendment_digest"],
    "effective_protocol_digest": PROTOCOL["effective_protocol_digest"],
    "dataset_revision": PREPARED["dataset_revision"],
    "population_digest": PREPARED["population_digest"],
    "model_repo_id": MODEL_REPO_ID,
    "model_revision": MODEL_REVISION,
    "model_dtype": "float32",
    "audio_protocol_fingerprint": AUDIO_PROTOCOL_FINGERPRINT,
    "max_new_tokens": MAX_NEW_TOKENS,
    "random_seed": RANDOM_SEED,
    "scientific_implementation_id": SCIENTIFIC_IMPLEMENTATION_ID,
    "capability_gate_version": "mmpilot.country_direction_capability.v2",
    "parent_v1_run_dir": str(PARENT_V1_RUN_DIR),
    "parent_lens_checksum": PARENT_LENS_CHECKSUM,
    "parent_v2_run_dir": str(PARENT_V2_RUN_DIR),
    "parent_capability_checksum": PARENT_CAPABILITY_CHECKSUM,
    "parent_localization_checksum": PARENT_LOCALIZATION_CHECKSUM,
    "country_completion_instruction": COUNTRY_COMPLETION_INSTRUCTION,
    "band_selection_endpoint": "exact_country_identity_swap_only",
    "band_selection_candidates": [list(band) for band in PATH_BANDS],
    "downstream_facts_hidden_during_band_selection": True,
    "task_matched_refit_rule": "only_after_existing_lens_readout_or_identity_swap_no_go",
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
        "controls": ["zero", "random", "unrelated"],
        "instrument_sequence": [
            "corrected_clean_capability", "identity_lens_validation",
            "identity_band_calibration", "downstream_fact_development",
            "fresh_confirmation",
        ],
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
    CONTROL_COUNTRIES, COUNTRY_COMPLETION_INSTRUCTION, FACTS, answer_matches,
    MIN_CAPABILITY_RATE, assistant_prefill, fact, identity_matches,
    speech_evidence, text_evidence,
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
    return build_multimodal_assistant_prefill_inputs(
        **kwargs, instruction=COUNTRY_COMPLETION_INSTRUCTION
    )

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
if MODEL_STAGE and LENS_PATH.is_file():
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
elif MODEL_STAGE and (PARENT_V1_RUN_DIR / "lenses" / LENS_PATH.name).is_file():
    from jlens.lens import JacobianLens
    from jlens.mmpilot.backend import file_checksum
    parent_lens_path = PARENT_V1_RUN_DIR / "lenses" / LENS_PATH.name
    parent_provenance_path = (
        PARENT_V1_RUN_DIR / "lenses" / LENS_PROVENANCE_PATH.name
    )
    if not parent_provenance_path.is_file():
        raise RuntimeError("the v1 parent lens has no provenance record")
    lens_provenance = json.loads(
        parent_provenance_path.read_text(encoding="utf-8")
    )
    actual_parent_checksum = file_checksum(str(parent_lens_path))
    if actual_parent_checksum != PARENT_LENS_CHECKSUM:
        raise RuntimeError(
            "the v1 parent lens does not match the frozen checksum"
        )
    if lens_provenance.get("lens_checksum") != PARENT_LENS_CHECKSUM:
        raise RuntimeError("the v1 parent provenance has the wrong checksum")
    LENS_PATH = parent_lens_path
    LENS_PROVENANCE_PATH = parent_provenance_path
    LENS = JacobianLens.load(str(LENS_PATH))
    if LENS.source_layers != list(LAYERS) or LENS.n_prompts != 99 or LENS.d_model != EXPECT_D_MODEL:
        raise RuntimeError("the v1 parent lens has the wrong scientific shape")
    print("reused checksum-pinned v1 lens", LENS_PATH)
    print("lens checksum", actual_parent_checksum)
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
elif any((
    RUN_STAGE2_CAPABILITY_AND_LOCALIZATION,
    RUN_STAGE3_DEVELOPMENT_SWAP,
    RUN_STAGE4_FRESH_CONFIRMATION,
    RUN_STAGE3B_FRANCE_CHINA_DOWNSTREAM_DEVELOPMENT,
    RUN_STAGE4B_FRANCE_CHINA_FRESH_CONFIRMATION,
    RUN_STAGE6B_CAUSAL_SITE_SCREEN,
    RUN_STAGE6C_RESTRICTED_SWAP_DEVELOPMENT,
)):
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
RUN_STAGE2_DEBUG_COUNTRY_INSTRUMENT = False
RUN_STAGE2_REFIT_TASK_MATCHED_LENS_IF_NEEDED = False
RUN_STAGE2C_FIT_BALANCED_TASK_LENS = False
RUN_STAGE3_DEVELOPMENT_SWAP = False
RUN_STAGE4_FRESH_CONFIRMATION = False
RUN_STAGE3B_FRANCE_CHINA_DOWNSTREAM_DEVELOPMENT = False
RUN_STAGE4B_FRANCE_CHINA_FRESH_CONFIRMATION = False
RUN_STAGE6A_CPU_CAUSAL_SITE_PLAN = False
RUN_STAGE6B_CAUSAL_SITE_SCREEN = False
RUN_STAGE6C_RESTRICTED_SWAP_DEVELOPMENT = False
RUN_STAGE5_WRITE_REPORT = False

CONFIRM_MODEL_LOAD = False
CONFIRM_FP32_A100 = False
CONFIRM_LENS_FIT_BUDGET = False
CONFIRM_LOCALIZATION_BUDGET = False
CONFIRM_IDENTITY_CALIBRATION_BUDGET = False
CONFIRM_TASK_MATCHED_REFIT_BUDGET = False
CONFIRM_BALANCED_TASK_FINAL_FIT_BUDGET = False
CONFIRM_DEVELOPMENT_BUDGET = False
CONFIRM_CONFIRMATION_BUDGET = False
CONFIRM_FRANCE_CHINA_DEVELOPMENT_BUDGET = False
CONFIRM_FRANCE_CHINA_CONFIRMATION_BUDGET = False
CONFIRM_CAUSAL_SITE_SCREEN_BUDGET = False
CONFIRM_RESTRICTED_SWAP_BUDGET = False

MODEL_REPO_ID = "google/gemma-4-E4B-it"
MODEL_REVISION = "fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd"
SCIENTIFIC_IMPLEMENTATION_ID = "country-prompt-lens-band-debug-v4.20260826"
PARENT_V1_RUN_DIR = Path(
    "/content/drive/MyDrive/jacobian-lens-gemma/runs/mmcountry/"
    "mmcountry_real_91055b9ab807"
)
PARENT_LENS_CHECKSUM = (
    "sha256:abfae7fdd9fb2cb66afe4c3d6ae0211a1b727001c882844e966883fcc0284ebe"
)
PARENT_V2_RUN_DIR = Path(
    "/content/drive/MyDrive/jacobian-lens-gemma/runs/mmcountry/"
    "mmcountry_real_5428299c4217"
)
PARENT_CAPABILITY_CHECKSUM = (
    "sha256:cd90a36c6625f708f72f7458a4407832b9c4ac65a0562d46063d0d7df2dee30e"
)
PARENT_LOCALIZATION_CHECKSUM = (
    "sha256:b0a61df35ba7fba9b4cf37e00f07d2ae092ce83a51e798ffb7c97f3605936ac4"
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
    RUN_STAGE2_DEBUG_COUNTRY_INSTRUMENT,
    RUN_STAGE2_REFIT_TASK_MATCHED_LENS_IF_NEEDED,
    RUN_STAGE2C_FIT_BALANCED_TASK_LENS,
    RUN_STAGE3_DEVELOPMENT_SWAP,
    RUN_STAGE4_FRESH_CONFIRMATION,
    RUN_STAGE3B_FRANCE_CHINA_DOWNSTREAM_DEVELOPMENT,
    RUN_STAGE4B_FRANCE_CHINA_FRESH_CONFIRMATION,
    RUN_STAGE6A_CPU_CAUSAL_SITE_PLAN,
    RUN_STAGE6B_CAUSAL_SITE_SCREEN,
    RUN_STAGE6C_RESTRICTED_SWAP_DEVELOPMENT,
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
    RUN_STAGE2_DEBUG_COUNTRY_INSTRUMENT,
    RUN_STAGE2_REFIT_TASK_MATCHED_LENS_IF_NEEDED,
    RUN_STAGE2C_FIT_BALANCED_TASK_LENS,
    RUN_STAGE3_DEVELOPMENT_SWAP,
    RUN_STAGE4_FRESH_CONFIRMATION,
    RUN_STAGE3B_FRANCE_CHINA_DOWNSTREAM_DEVELOPMENT,
    RUN_STAGE4B_FRANCE_CHINA_FRESH_CONFIRMATION,
    RUN_STAGE6B_CAUSAL_SITE_SCREEN,
    RUN_STAGE6C_RESTRICTED_SWAP_DEVELOPMENT,
))
if MODEL_STAGE and not CONFIRM_MODEL_LOAD:
    print("MODEL BLOCKED: set CONFIRM_MODEL_LOAD=True after reading the budget")
if MODEL_STAGE and not CONFIRM_FP32_A100:
    print("FP32 BLOCKED: set CONFIRM_FP32_A100=True on an 80 GB A100")
if RUN_STAGE2_DEBUG_COUNTRY_INSTRUMENT and not CONFIRM_IDENTITY_CALIBRATION_BUDGET:
    print("IDENTITY CALIBRATION BLOCKED: confirm its forward-pass budget")
if (
    RUN_STAGE2_REFIT_TASK_MATCHED_LENS_IF_NEEDED
    and not CONFIRM_TASK_MATCHED_REFIT_BUDGET
):
    print("TASK-MATCHED REFIT BLOCKED: confirm its backward-pass budget")
if RUN_STAGE2C_FIT_BALANCED_TASK_LENS and not CONFIRM_BALANCED_TASK_FINAL_FIT_BUDGET:
    print("FINAL BALANCED FIT BLOCKED: confirm its one-time backward-pass budget")
if RUN_STAGE6B_CAUSAL_SITE_SCREEN and not CONFIRM_CAUSAL_SITE_SCREEN_BUDGET:
    print("CAUSAL-SITE SCREEN BLOCKED: confirm its forward-pass budget")
if RUN_STAGE6C_RESTRICTED_SWAP_DEVELOPMENT and not CONFIRM_RESTRICTED_SWAP_BUDGET:
    print("RESTRICTED SWAP BLOCKED: confirm its forward-pass budget")
'''
)

markdown("## 2. Frozen protocol and budget, printed before model load")
code(
    r'''
from jlens.mmpilot.country_workspace import (
    COUNTRY_COMPLETION_INSTRUCTION, DIRECTIONS, EVAL_COUNTRIES, FIT_COUNTRIES,
    LAYERS, MODALITIES,
    N_CONFIRMATION_PER_COUNTRY, N_DEVELOPMENT_PER_COUNTRY,
    N_FIT_PER_COUNTRY, N_LOCALIZATION_PER_COUNTRY, PATH_BANDS,
    PROPERTIES, TARGET_LAYER, benchmark_spec,
)

print("COUNTRY WORKSPACE BUDGET")
print("  fit population", len(FIT_COUNTRIES) * N_FIT_PER_COUNTRY, "identities x 3 modalities = 99 examples")
print("  fit work", 99, "forwards +", 99 * ((EXPECT_D_MODEL + DIM_BATCH - 1) // DIM_BATCH), "backward passes")
print("  clean capability", len(EVAL_COUNTRIES) * N_DEVELOPMENT_PER_COUNTRY * len(MODALITIES) * 3, "complete generations")
print("  direct localization", len(DIRECTIONS) * len(PROPERTIES) * len(MODALITIES) * len(PATH_BANDS) * N_LOCALIZATION_PER_COUNTRY, "conditions")
print("  corrected clean capability", len(EVAL_COUNTRIES) * N_DEVELOPMENT_PER_COUNTRY * len(MODALITIES) * 3, "complete generations")
print("  identity lens readout", 3 * N_DEVELOPMENT_PER_COUNTRY * len(MODALITIES), "forwards plus cheap six-country projections")
print("  identity band calibration maximum", 3 * len(PATH_BANDS) * len(MODALITIES) * 4 * N_DEVELOPMENT_PER_COUNTRY, "conditions per lens")
print("  conditional task-matched refit", 99, "forwards plus", 99 * ((EXPECT_D_MODEL + DIM_BATCH - 1) // DIM_BATCH), "backward passes")
print("  FINAL balanced-task pooled fit", 99, "forwards plus", 99 * ((EXPECT_D_MODEL + DIM_BATCH - 1) // DIM_BATCH), "backward passes; exactly one fit arm")
print("    balance: 33 identity, 33 capital, 33 continent; 11 of each per modality")
print("  development", len(DIRECTIONS) * len(PROPERTIES) * len(MODALITIES) * 4 * N_DEVELOPMENT_PER_COUNTRY, "conditions")
print("  confirmation maximum", len(DIRECTIONS) * len(PROPERTIES) * len(MODALITIES) * 4 * N_CONFIRMATION_PER_COUNTRY, "conditions")
print("  France-to-China follow-up development", len(PROPERTIES) * len(MODALITIES) * 4 * N_DEVELOPMENT_PER_COUNTRY, "conditions")
print("  France-to-China follow-up confirmation", len(PROPERTIES) * len(MODALITIES) * 4 * N_CONFIRMATION_PER_COUNTRY, "conditions plus clean capability")
print("  no-refit causal-site screen", len(PATH_BANDS) * 2 * len(PROPERTIES) * len(MODALITIES) * 4, "development conditions")
print("    selection reads actual-target-state and direct-answer controls only")
print("  localized full-state replication", (N_DEVELOPMENT_PER_COUNTRY - 1) * len(PROPERTIES) * len(MODALITIES) * 3, "development conditions")
print("  localized exact J-lens exchange", (N_DEVELOPMENT_PER_COUNTRY - 1) * len(PROPERTIES) * len(MODALITIES) * 4, "development conditions")
print("    combined localized development", (N_DEVELOPMENT_PER_COUNTRY - 1) * len(PROPERTIES) * len(MODALITIES) * 7, "conditions")
print("  diagnostic fitting/backward passes 0; fresh confirmation examples opened 0")
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

markdown("## 12.5 No-refit causal-site diagnosis")
code(
    r'''
from jlens.mmpilot.backend import file_checksum
from jlens.mmpilot.country_activation_patch import (
    ACTIVATION_PATCH_VERSION, LOCALIZED_DEVELOPMENT_VERSION,
    PATCH_SITES, SCREEN_CONDITIONS,
    capture_activation_sites, causal_site_screen_report, patch_position,
    localized_development_report, single_position_inputs,
    state_validated_selection,
    unrestricted_greedy_activation_patch_trial,
)

CAUSAL_SITE_PLAN_PATH = RUN_DIR / "country_causal_site_plan.json"
CAUSAL_SITE_ROOT = RUN_DIR / "diagnostics" / "country_causal_site_v1"
CAUSAL_SITE_SCREEN_PATH = CAUSAL_SITE_ROOT / "country_causal_site_screen_report.json"
LOCALIZED_DEVELOPMENT_ROOT = (
    RUN_DIR / "diagnostics" / "country_localized_development_v1"
)
RESTRICTED_SWAP_PATH = (
    LOCALIZED_DEVELOPMENT_ROOT / "country_localized_development_report.json"
)

_development_by_country = {
    country: sorted(
        [
            row for row in MEDIA_ROWS
            if row["study_split"] == "development" and row["country"] == country
        ],
        key=lambda row: row["unit_id"],
    )
    for country in ("France", "China")
}
_italy_fit_rows = sorted(
    [row for row in MEDIA_ROWS if row["study_split"] == "fit" and row["country"] == "Italy"],
    key=lambda row: row["unit_id"],
)
if any(len(rows) != N_DEVELOPMENT_PER_COUNTRY for rows in _development_by_country.values()):
    raise RuntimeError("the causal-site diagnostic requires all frozen development rows")
if not _italy_fit_rows:
    raise RuntimeError("the causal-site diagnostic requires its frozen unrelated Italy donor")

_screen_source_rows = _development_by_country["France"][:1]
_restricted_source_rows = _development_by_country["France"][1:]
_target_donor_row = _development_by_country["China"][0]
_unrelated_donor_row = _italy_fit_rows[0]
_causal_plan_body = {
    "version": "mmpilot.country_causal_site_plan.v1",
    "purpose": "locate causal leverage before another fit or any fresh confirmation",
    "source": "France", "target": "China", "unrelated_donor": "Italy",
    "properties": list(PROPERTIES), "modalities": list(MODALITIES),
    "bands": [list(band) for band in PATH_BANDS], "sites": list(PATCH_SITES),
    "screen_conditions": list(SCREEN_CONDITIONS),
    "screen_source_units": [row["unit_id"] for row in _screen_source_rows],
    "restricted_swap_units": [row["unit_id"] for row in _restricted_source_rows],
    "target_donor_unit": _target_donor_row["unit_id"],
    "unrelated_donor_unit": _unrelated_donor_row["unit_id"],
    "selection_reads_coordinate_swap_outcomes": False,
    "fresh_confirmation_opened": False,
    "fitting_performed": False, "backward_passes": 0,
    "screen_generation_conditions": len(PATH_BANDS) * len(PATCH_SITES)
        * len(PROPERTIES) * len(MODALITIES) * len(SCREEN_CONDITIONS),
    "restricted_generation_conditions_if_licensed": len(_restricted_source_rows)
        * len(PROPERTIES) * len(MODALITIES) * 4,
}
_causal_site_requested = any((
    RUN_STAGE6A_CPU_CAUSAL_SITE_PLAN,
    RUN_STAGE6B_CAUSAL_SITE_SCREEN,
    RUN_STAGE6C_RESTRICTED_SWAP_DEVELOPMENT,
))
if _causal_site_requested:
    if not BALANCED_LENS_PATH.is_file():
        raise RuntimeError(
            "the no-refit diagnostic requires the completed balanced-task lens"
        )
    _causal_plan_body["balanced_lens_checksum"] = file_checksum(
        str(BALANCED_LENS_PATH)
    )
_causal_plan_body["plan_digest"] = payload_checksum(_causal_plan_body)

if RUN_STAGE6A_CPU_CAUSAL_SITE_PLAN:
    _write_report(CAUSAL_SITE_PLAN_PATH, _causal_plan_body)
    print("CAUSAL-SITE PLAN FROZEN")
    print("  plan", CAUSAL_SITE_PLAN_PATH)
    print("  digest", _causal_plan_body["plan_digest"])
    print("  screen", _causal_plan_body["screen_generation_conditions"], "generation conditions")
    print("  original restricted J-lens follow-up", _causal_plan_body["restricted_generation_conditions_if_licensed"], "conditions")
    print("  a state-valid-screen amendment may add 54 full-state conditions")
    print("  fitting 0; backward passes 0; fresh confirmation opened False")

CAUSAL_SITE_SCREEN = (
    json.loads(CAUSAL_SITE_SCREEN_PATH.read_text(encoding="utf-8"))
    if CAUSAL_SITE_SCREEN_PATH.is_file() else None
)
RESTRICTED_SWAP_DEVELOPMENT = (
    json.loads(RESTRICTED_SWAP_PATH.read_text(encoding="utf-8"))
    if RESTRICTED_SWAP_PATH.is_file() else None
)

if RUN_STAGE6B_CAUSAL_SITE_SCREEN or RUN_STAGE6C_RESTRICTED_SWAP_DEVELOPMENT:
    if not MODEL_ENABLED:
        raise RuntimeError("causal-site GPU stages require the loaded fp32 model")
    if ACTIVE_LENS_LABEL != "balanced_task_pooled_j":
        raise RuntimeError("causal-site diagnosis is pinned to the final balanced-task lens")
    _balanced_checksum = _causal_plan_body["balanced_lens_checksum"]
    _patch_fingerprint = RunFingerprint(
        mode="real", model_repo_id=MODEL_REPO_ID, model_revision=MODEL_REVISION,
        processor_revision=MODEL_REVISION, layers=tuple(LAYERS),
        lens_checksum=_balanced_checksum,
        manifest_checksum=PREPARED["population_digest"],
        split_id=payload_checksum({
            "screen": _causal_plan_body["screen_source_units"],
            "restricted": _causal_plan_body["restricted_swap_units"],
        }),
        intervention_config={
            "version": ACTIVATION_PATCH_VERSION,
            "bands": [list(band) for band in PATH_BANDS],
            "sites": list(PATCH_SITES),
            "screen_conditions": list(SCREEN_CONDITIONS),
            "restricted_conditions": ["exact", "zero", "random", "unrelated"],
            "selection_reads_coordinate_swap_outcomes": False,
            "fresh_confirmation_opened": False,
        },
        extra={
            "parent_scientific_digest": SCIENTIFIC_DIGEST,
            "plan_digest": _causal_plan_body["plan_digest"],
            "fitting_performed": False,
        },
    )
    PATCH_STORE = UnitStore(CAUSAL_SITE_ROOT, _patch_fingerprint)
    print("causal-site run", CAUSAL_SITE_ROOT)
    print("resume", PATCH_STORE.open())

if RUN_STAGE6B_CAUSAL_SITE_SCREEN:
    if not CONFIRM_CAUSAL_SITE_SCREEN_BUDGET:
        raise RuntimeError("causal-site screen budget is not confirmed")
    if CAUSAL_SITE_PLAN_PATH.is_file():
        _stored_plan = json.loads(CAUSAL_SITE_PLAN_PATH.read_text(encoding="utf-8"))
        if _stored_plan.get("plan_digest") != _causal_plan_body["plan_digest"]:
            raise RuntimeError("the stored causal-site plan differs from this run")
    else:
        _write_report(CAUSAL_SITE_PLAN_PATH, _causal_plan_body)

    _country_tokens = concept_tokens(("France", "China", "Italy"))
    _capture_cache = {}

    def _captured(row, property_name, modality):
        cache_key = (row["unit_id"], property_name, modality)
        if cache_key not in _capture_cache:
            inputs = build_task_inputs(row, modality, property_name)
            positions = {
                site: patch_position(
                    inputs, site,
                    country_token_id=(
                        _country_tokens[row["country"]].token_id
                        if modality == "text" else None
                    ),
                )
                for site in PATCH_SITES
            }
            _capture_cache[cache_key] = {
                "inputs": inputs,
                "positions": positions,
                "activations": capture_activation_sites(
                    BACKEND, inputs, layers=LAYERS, positions=positions,
                ),
            }
        return _capture_cache[cache_key]

    _screen_rows = []
    _unembed = BACKEND.unembedding_weight()
    for source_row in _screen_source_rows:
        for property_name in PROPERTIES:
            expected = fact("China", property_name)
            for modality in MODALITIES:
                source_capture = _captured(source_row, property_name, modality)
                target_capture = _captured(_target_donor_row, property_name, modality)
                unrelated_capture = _captured(_unrelated_donor_row, property_name, modality)
                for band in PATH_BANDS:
                    exact_bases = build_swap_bases_for_lens(
                        ACTIVE_LENS, _unembed, layers=band,
                        source=_country_tokens["France"], target=_country_tokens["China"],
                    )
                    direct_vectors = answer_vectors(ACTIVE_LENS, expected, band)
                    for site in PATCH_SITES:
                        source_position = source_capture["positions"][site]
                        site_inputs = single_position_inputs(
                            source_capture["inputs"], source_position
                        )
                        donor_conditions = {
                            "target_state": target_capture["activations"][site],
                            "self_state": source_capture["activations"][site],
                            "unrelated_state": unrelated_capture["activations"][site],
                        }
                        for condition in SCREEN_CONDITIONS:
                            key = safe_key(
                                "country_causal_site_v1", property_name, modality,
                                band[0], band[-1], site, source_row["unit_id"], condition,
                            )
                            stored = PATCH_STORE.load("intervention", key)
                            if stored is None:
                                if condition == "direct_answer":
                                    result = unrestricted_greedy_direct_answer_trial(
                                        BACKEND, site_inputs, bases=exact_bases,
                                        answer_vectors=direct_vectors, answer=expected,
                                        max_new_tokens=MAX_NEW_TOKENS,
                                        position_rule="evidence_span_only",
                                        realization_policy=TEXT_MODEL_DTYPE_REALIZATION,
                                        alpha=1.0,
                                    )
                                    integrity = diagnostic_integrity(result)
                                else:
                                    donor = {
                                        layer: donor_conditions[condition][layer]
                                        for layer in band
                                    }
                                    result = unrestricted_greedy_activation_patch_trial(
                                        BACKEND, source_capture["inputs"],
                                        donor_by_layer=donor,
                                        source_position=source_position,
                                        answer=expected, max_new_tokens=MAX_NEW_TOKENS,
                                    )
                                    patch_diag = result["activation_patch_diagnostics"]
                                    integrity = bool(
                                        patch_diag["all_hooks_fired"]
                                        and patch_diag["all_finite"]
                                    )
                                stored = {
                                    **result, "unit_id": source_row["unit_id"],
                                    "source": "France", "target": "China",
                                    "property": property_name, "modality": modality,
                                    "condition": condition, "site": site,
                                    "layers_patched": list(band), "expected": expected,
                                    "success": answer_matches(result["generated_text"], expected),
                                    "integrity_pass": integrity,
                                }
                                PATCH_STORE.save("intervention", key, stored)
                                work = "computed"
                            else:
                                work = "reused"
                            _screen_rows.append(stored)
                            if len(_screen_rows) == 1 or len(_screen_rows) % 48 == 0:
                                print("causal-site screen", len(_screen_rows), work)
    CAUSAL_SITE_SCREEN = causal_site_screen_report(
        _screen_rows, bands=PATH_BANDS, expected_n=1,
        properties=PROPERTIES, modalities=MODALITIES,
    )
    _write_report(CAUSAL_SITE_SCREEN_PATH, CAUSAL_SITE_SCREEN)
    print("CAUSAL-SITE SCREEN", CAUSAL_SITE_SCREEN["verdict"])
    print("selected", CAUSAL_SITE_SCREEN["selected"])
    print("selection read coordinate-swap outcomes", CAUSAL_SITE_SCREEN["selection_used_coordinate_swap_outcomes"])
    print("report", CAUSAL_SITE_SCREEN_PATH)

if RUN_STAGE6C_RESTRICTED_SWAP_DEVELOPMENT:
    if not CONFIRM_RESTRICTED_SWAP_BUDGET:
        raise RuntimeError("localized development budget is not confirmed")
    if CAUSAL_SITE_SCREEN is None:
        raise RuntimeError("run or load the causal-site screen first")
    _state_choice = state_validated_selection(CAUSAL_SITE_SCREEN)
    print("SOURCE SCREEN VERDICT UNCHANGED", CAUSAL_SITE_SCREEN["verdict"])
    print("STATE-VALIDATED PATH", _state_choice["verdict"])
    if _state_choice["selected"] is None:
        print("LOCALIZED DEVELOPMENT NOT LICENSED: no state-valid path")
    else:
        selection = _state_choice["selected"]
        band = tuple(map(int, selection["band"]))
        site = selection["site"]
        _country_tokens = concept_tokens((*EVAL_COUNTRIES, *CONTROL_COUNTRIES))
        _unembed = BACKEND.unembedding_weight()
        exact_bases = build_swap_bases_for_lens(
            ACTIVE_LENS, _unembed, layers=band,
            source=_country_tokens["France"], target=_country_tokens["China"],
        )
        random_bases = {
            layer: random_two_direction_basis(
                basis, seed=RANDOM_SEED + 91000 + layer
            )
            for layer, basis in exact_bases.items()
        }
        unrelated_bases = build_swap_bases_for_lens(
            ACTIVE_LENS, _unembed, layers=band,
            source=_country_tokens[CONTROL_COUNTRIES[0]],
            target=_country_tokens[CONTROL_COUNTRIES[1]],
        )
        _restricted_target_rows = _development_by_country["China"][1:]
        _restricted_unrelated_rows = _italy_fit_rows[1:1 + len(_restricted_source_rows)]
        if not (
            len(_restricted_source_rows)
            == len(_restricted_target_rows)
            == len(_restricted_unrelated_rows)
        ):
            raise RuntimeError("localized development donor pairing is incomplete")
        _localized_fingerprint = RunFingerprint(
            mode="real", model_repo_id=MODEL_REPO_ID,
            model_revision=MODEL_REVISION, processor_revision=MODEL_REVISION,
            layers=band, lens_checksum=_balanced_checksum,
            manifest_checksum=PREPARED["population_digest"],
            split_id=payload_checksum({
                "source": [row["unit_id"] for row in _restricted_source_rows],
                "target": [row["unit_id"] for row in _restricted_target_rows],
                "unrelated": [row["unit_id"] for row in _restricted_unrelated_rows],
            }),
            intervention_config={
                "version": LOCALIZED_DEVELOPMENT_VERSION,
                "selection": selection,
                "full_state_conditions": [
                    "target_state", "self_state", "unrelated_state",
                ],
                "coordinate_conditions": [
                    "exact", "zero", "random", "unrelated",
                ],
                "alpha": 1.0,
                "position_rule": "one_state_validated_position",
                "fresh_confirmation_opened": False,
            },
            extra={
                "source_screen_checksum": CAUSAL_SITE_SCREEN["report_checksum"],
                "state_selection_checksum": _state_choice["record_checksum"],
                "fitting_performed": False, "backward_passes": 0,
            },
        )
        LOCALIZED_STORE = UnitStore(
            LOCALIZED_DEVELOPMENT_ROOT, _localized_fingerprint
        )
        print("localized development run", LOCALIZED_DEVELOPMENT_ROOT)
        print("resume", LOCALIZED_STORE.open())

        def _localized_capture(row, property_name, modality):
            inputs = build_task_inputs(row, modality, property_name)
            position = patch_position(
                inputs, site,
                country_token_id=(
                    _country_tokens[row["country"]].token_id
                    if modality == "text" else None
                ),
            )
            captured = capture_activation_sites(
                BACKEND, inputs, layers=band, positions={site: position},
            )
            return inputs, position, captured[site]

        _state_rows = []
        _coordinate_rows = []
        for row_index, source_row in enumerate(_restricted_source_rows):
            target_row = _restricted_target_rows[row_index]
            unrelated_row = _restricted_unrelated_rows[row_index]
            for property_name in PROPERTIES:
                expected = fact("China", property_name)
                for modality in MODALITIES:
                    inputs, position, source_state = _localized_capture(
                        source_row, property_name, modality
                    )
                    _, _, target_state = _localized_capture(
                        target_row, property_name, modality
                    )
                    _, _, unrelated_state = _localized_capture(
                        unrelated_row, property_name, modality
                    )
                    for condition, donor_state in (
                        ("target_state", target_state),
                        ("self_state", source_state),
                        ("unrelated_state", unrelated_state),
                    ):
                        key = safe_key(
                            "country_localized_state_v1",
                            CAUSAL_SITE_SCREEN["report_checksum"],
                            property_name, modality, source_row["unit_id"], condition,
                        )
                        stored = LOCALIZED_STORE.load("intervention", key)
                        if stored is None:
                            result = unrestricted_greedy_activation_patch_trial(
                                BACKEND, inputs, donor_by_layer=donor_state,
                                source_position=position, answer=expected,
                                max_new_tokens=MAX_NEW_TOKENS,
                            )
                            patch_diag = result["activation_patch_diagnostics"]
                            stored = {
                                **result, "unit_id": source_row["unit_id"],
                                "donor_unit_id": (
                                    target_row["unit_id"]
                                    if condition == "target_state"
                                    else source_row["unit_id"]
                                    if condition == "self_state"
                                    else unrelated_row["unit_id"]
                                ),
                                "source": "France", "target": "China",
                                "property": property_name, "modality": modality,
                                "condition": condition, "site": site,
                                "layers_patched": list(band), "expected": expected,
                                "success": answer_matches(
                                    result["generated_text"], expected
                                ),
                                "integrity_pass": bool(
                                    patch_diag["all_hooks_fired"]
                                    and patch_diag["all_finite"]
                                ),
                            }
                            LOCALIZED_STORE.save("intervention", key, stored)
                            work = "computed"
                        else:
                            work = "reused"
                        _state_rows.append(stored)
                        if len(_state_rows) == 1 or len(_state_rows) % 18 == 0:
                            print("localized state", len(_state_rows), work)

                    site_inputs = single_position_inputs(inputs, position)
                    for condition, alpha, bases in (
                        ("exact", 1.0, exact_bases),
                        ("zero", 0.0, exact_bases),
                        ("random", 1.0, random_bases),
                        ("unrelated", 1.0, unrelated_bases),
                    ):
                        key = safe_key(
                            "country_restricted_swap_v1",
                            CAUSAL_SITE_SCREEN["report_checksum"],
                            property_name, modality, source_row["unit_id"], condition,
                        )
                        stored = LOCALIZED_STORE.load("intervention", key)
                        if stored is None:
                            result = unrestricted_greedy_swap_trial(
                                BACKEND, site_inputs, bases=bases, alpha=alpha,
                                answer=expected, max_new_tokens=MAX_NEW_TOKENS,
                                position_rule="evidence_span_only",
                                realization_policy=TEXT_MODEL_DTYPE_REALIZATION,
                            )
                            stored = {
                                **result, "unit_id": source_row["unit_id"],
                                "source": "France", "target": "China",
                                "property": property_name, "modality": modality,
                                "condition": condition, "site": site,
                                "expected": expected,
                                "success": answer_matches(result["generated_text"], expected),
                                "integrity_pass": diagnostic_integrity(
                                    result, exact=(condition == "exact")
                                ),
                            }
                            LOCALIZED_STORE.save("intervention", key, stored)
                            work = "computed"
                        else:
                            work = "reused"
                        _coordinate_rows.append(stored)
                        if (
                            len(_coordinate_rows) == 1
                            or len(_coordinate_rows) % 24 == 0
                        ):
                            print("localized J-lens", len(_coordinate_rows), work)
        RESTRICTED_SWAP_DEVELOPMENT = localized_development_report(
            _state_rows, _coordinate_rows,
            expected_n=len(_restricted_source_rows),
            properties=PROPERTIES, modalities=MODALITIES,
            selection={**selection, "selection_record": _state_choice},
        )
        _write_report(RESTRICTED_SWAP_PATH, RESTRICTED_SWAP_DEVELOPMENT)
        print("LOCALIZED DEVELOPMENT", RESTRICTED_SWAP_DEVELOPMENT["verdict"])
        for cell in RESTRICTED_SWAP_DEVELOPMENT["full_state_arm"]["cells"]:
            target = cell["conditions"]["target_state"]
            print(
                "state", cell["property"], cell["modality"],
                f"target {target['successes']}/{target['n']}",
                "controls", {
                    name: cell["conditions"][name]["successes"]
                    for name in ("self_state", "unrelated_state")
                },
            )
        for cell in RESTRICTED_SWAP_DEVELOPMENT["j_lens_coordinate_arm"]["cells"]:
            exact = cell["conditions"]["exact"]
            print(
                "J-lens", cell["property"], cell["modality"],
                f"exact {exact['successes']}/{exact['n']}",
                "controls", {
                    name: cell["conditions"][name]["successes"]
                    for name in ("zero", "random", "unrelated")
                },
            )
        print("report", RESTRICTED_SWAP_PATH)

if RUN_STAGE6A_CPU_CAUSAL_SITE_PLAN and not (
    RUN_STAGE6B_CAUSAL_SITE_SCREEN or RUN_STAGE6C_RESTRICTED_SWAP_DEVELOPMENT
):
    print("CPU PLAN COMPLETE. Stop this runtime before the fp32 A100 stages.")
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
