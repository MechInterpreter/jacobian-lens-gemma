# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Deterministic export of completed run artifacts into explorer bundles.

The Gemma 4 Multimodal J-Lens Explorer (``explorer/``) is a static browser
application: everything it shows comes from one normalized JSON bundle
(schema ``jlens.explorer.bundle.v1``, described by
``schemas/explorer_bundle.schema.json``). This module turns the completed
J-space run's artifacts into such a bundle without mutating any source file,
and merges later bundles (causal smoke run, multimodal capture) into it.

Determinism contract: :func:`build_text_bundle` on the same inputs returns
the same bundle, and :func:`canonical_json` renders it to identical bytes —
the creation timestamp is derived from the *source run's* recorded
``written_utc``, never from the wall clock, ordering of every array is
explicitly sorted, and JSON is dumped with sorted keys.

Path safety: exported bundles never contain absolute local paths (the source
records carry Colab ``/content/drive/...`` paths in their provenance; those
are reduced to run-relative identifiers). :func:`assert_no_absolute_paths`
enforces this and is run on every export.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Iterable, Sequence
from typing import Any

from jlens.metadata import file_sha256

EXPORTER_VERSION = "1.0.0"
SCHEMA_VERSION = "1.0.0"
BUNDLE_SCHEMA = "jlens.explorer.bundle.v1"

#: Representative demo subset: covers every category, both formats, and
#: deliberately includes weak examples (counting/association/chat items whose
#: layer-38 J-lens rank is poor) alongside strong ones. Not cherry-picked.
DEFAULT_DEMO_SLUGS: tuple[str, ...] = (
    "factual-gold-symbol",
    "factual-canberra",
    "factual-shakespeare",
    "factual-photosynthesis",
    "multihop-madrid-language",
    "multihop-eiffel-capital",
    "multihop-sushi-currency",
    "association-storm",
    "association-desert",
    "association-doctor",
    "antonym-tall-short",
    "antonym-early-late",
    "counting-week-days",
    "counting-days-four",
    "syntactic-alphabet",
    "syntactic-roses-blue",
    "syntactic-numbered-list",
    "chat-factual-canberra",
    "chat-antonym-early",
    "chat-counting-triangle",
)

_ABSOLUTE_PATH_PATTERN = re.compile(
    r"(^|[\s\"'=(])(/content/|/home/|/Users/|[A-Za-z]:[\\/](?:Users|content|home)[\\/])"
)

#: Threshold for the descriptive high-frequency (nuisance-candidate) flag:
#: an atom selected across at least this many distinct prompts in the source
#: run's k=10/16/25 sweep is flagged. Descriptive bookkeeping, not a claim.
HIGH_FREQUENCY_MIN_DISTINCT_PROMPTS = 10


class ExportError(ValueError):
    """Raised when source artifacts are malformed or inconsistent."""


# --------------------------------------------------------------- loading


def _load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_run_artifacts(
    run_dir: str, *, k: int = 10, layers: Sequence[int] | None = None
) -> dict:
    """Load a completed jspace_pursuit run's artifacts read-only.

    Returns a dict with ``run_metadata``, ``cones`` (list over all layers),
    ``trajectories``, ``eval_results``, resolved ``layers``, and
    ``artifact_fingerprints`` (repo-relative path -> sha256). Raises
    :class:`ExportError` on malformed artifacts; missing *optional*
    artifacts (trajectories, eval results) produce warnings collected in
    ``warnings``.
    """
    meta_path = os.path.join(run_dir, "run_metadata.json")
    if not os.path.isfile(meta_path):
        raise ExportError(f"{run_dir}: not a run directory (no run_metadata.json)")
    run_metadata = _load_json(meta_path)
    if run_metadata.get("mode") != "jspace_pursuit":
        raise ExportError(
            f"{meta_path}: expected mode 'jspace_pursuit', got "
            f"{run_metadata.get('mode')!r}"
        )
    run_layers = list(
        run_metadata.get("config", {}).get("decomposition", {}).get("layers", [])
    )
    layers = list(layers) if layers is not None else run_layers
    unknown = sorted(set(layers) - set(run_layers))
    if unknown:
        raise ExportError(f"layers {unknown} not in the run's layers {run_layers}")

    fingerprints: dict[str, str] = {"run_metadata.json": file_sha256(meta_path)}
    warnings: list[str] = []

    cones: list[dict] = []
    for layer in sorted(layers):
        rel = f"artifacts/cones/cones_layer{layer}_k{k}.json"
        path = os.path.join(run_dir, rel)
        if not os.path.isfile(path):
            raise ExportError(f"missing cone artifact: {path}")
        records = _load_json(path)
        if not isinstance(records, list):
            raise ExportError(f"{path}: expected a list of cone records")
        for index, record in enumerate(records):
            if record.get("schema") != "jlens.cones.record.v1":
                raise ExportError(
                    f"{path}[{index}]: unexpected schema {record.get('schema')!r}"
                )
            record["_source_artifact"] = rel
            record["_source_index"] = index
        cones.extend(records)
        fingerprints[rel] = file_sha256(path)

    trajectories: list[dict] = []
    traj_rel = f"artifacts/trajectories_k{k}.json"
    traj_path = os.path.join(run_dir, traj_rel)
    if os.path.isfile(traj_path):
        trajectories = _load_json(traj_path)
        for index, record in enumerate(trajectories):
            if record.get("schema") != "jlens.cones.transition.v1":
                raise ExportError(
                    f"{traj_path}[{index}]: unexpected schema "
                    f"{record.get('schema')!r}"
                )
        fingerprints[traj_rel] = file_sha256(traj_path)
    else:
        warnings.append(f"optional artifact missing: {traj_rel} (no trajectories)")

    eval_results: dict | None = None
    eval_rel = "artifacts/eval_v2_results.json"
    eval_path = os.path.join(run_dir, eval_rel)
    if os.path.isfile(eval_path):
        eval_results = _load_json(eval_path)
        fingerprints[eval_rel] = file_sha256(eval_path)
    else:
        warnings.append(
            f"optional artifact missing: {eval_rel} (no rank/overlap records)"
        )

    return {
        "run_metadata": run_metadata,
        "cones": cones,
        "trajectories": trajectories,
        "eval_results": eval_results,
        "layers": sorted(layers),
        "k": k,
        "artifact_fingerprints": fingerprints,
        "warnings": warnings,
    }


def load_prompt_texts(prompts_path: str) -> dict[str, dict]:
    """Slug -> {text, is_pre_template} from an eval_prompts_v2-style file."""
    data = _load_json(prompts_path)
    texts: dict[str, dict] = {}
    for entry in data.get("plain", []):
        texts[entry["slug"]] = {"text": entry["text"], "is_pre_template": False}
    for entry in data.get("chat", []):
        texts[entry["slug"]] = {"text": entry["user"], "is_pre_template": True}
    return texts


def load_atom_frequencies(analysis_dir: str) -> dict[int, dict]:
    """token_id -> frequency metadata from an analysis report's
    ``atom_frequencies.csv`` (descriptive nuisance-candidate bookkeeping)."""
    import csv

    path = os.path.join(analysis_dir, "atom_frequencies.csv")
    if not os.path.isfile(path):
        return {}
    table: dict[int, dict] = {}
    with open(path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            token_id = int(row["token_id"])
            distinct = int(row["n_distinct_prompts"])
            table[token_id] = {
                "total_selections": int(row["total_selections"]),
                "n_distinct_prompts": distinct,
                "high_frequency": distinct >= HIGH_FREQUENCY_MIN_DISTINCT_PROMPTS,
            }
    return table


# ------------------------------------------------------------ identities


def example_id(modality: str, slug: str | None, prompt_hash: str) -> str:
    """Stable example ID: ``<modality>:<slug>:<prompt_hash>``."""
    return f"{modality}:{slug or 'unnamed'}:{prompt_hash}"


def _strength(eval_example: dict | None, positions: Sequence[int]) -> dict | None:
    """Descriptive strong/weak tag from measured layer-38 J-lens rank of the
    model's top-1 at the example's last position (rank 0 = exact top-1
    agreement -> strong; rank > 1000 -> weak; else middling)."""
    if not eval_example:
        return None
    layer38 = eval_example.get("layers", {}).get("38", {}).get("jlens")
    if not layer38:
        return None
    ranks = layer38.get("rank_of_model_top1")
    if not ranks:
        return None
    try:
        index = list(eval_example["positions"]).index(-1)
    except ValueError:
        index = len(ranks) - 1
    rank = ranks[index]
    tag = "strong" if rank == 0 else ("weak" if rank > 1000 else "middling")
    return {
        "tag": tag,
        "basis": (
            f"measured layer-38 J-lens rank of the model top-1 at position -1 "
            f"is {rank} (0=strong, >1000=weak)"
        ),
    }


# --------------------------------------------------------------- building


def _cone_key(record: dict) -> tuple[str, int, int]:
    return (record["prompt_hash"], int(record["position"]), int(record["layer"]))


def _reduced_source_provenance(record: dict, run_id: str) -> dict:
    """Cone provenance without absolute local paths."""
    src = record.get("run_provenance", {})
    return {
        "run_id": run_id,
        "artifact": record["_source_artifact"],
        "record_index": record["_source_index"],
        "config_fingerprint": src.get("config_fingerprint"),
        "lens_fingerprint": src.get("lens_fingerprint"),
        "model_revision": src.get("model_revision"),
        "local_commit": src.get("local_commit"),
        "upstream_commit": src.get("upstream_commit"),
        "algorithm_settings": record.get("algorithm_settings"),
        "dictionary_provenance": record.get("dictionary_provenance"),
    }


def _concentration(coefficients: Sequence[float]) -> dict | None:
    total = float(sum(c for c in coefficients if c > 0))
    if total <= 0:
        return None
    shares = [c / total for c in coefficients if c > 0]
    return {
        "herfindahl": sum(s * s for s in shares),
        "top1_share": max(shares),
        "n_nonzero": len(shares),
    }


def _cone_to_bundle_record(
    record: dict,
    *,
    example: dict,
    run_id: str,
    atom_frequencies: dict[int, dict],
) -> dict:
    position = int(record["position"])
    output = example["model_output"].get(str(position), {})
    model_top1_id = output.get("model_top1_id")
    coefficients = list(record["coefficients"])
    atoms = []
    total = float(sum(c for c in coefficients if c > 0))
    for token_id, label, coeff in zip(
        record["selected_token_ids"],
        record["selected_labels"],
        coefficients,
        strict=True,
    ):
        atoms.append(
            {
                "token_id": int(token_id),
                "label": label,
                "coefficient": float(coeff),
                "is_output_token": model_top1_id is not None
                and int(token_id) == int(model_top1_id),
                "is_effective": coeff > 0,
                "coefficient_share": (coeff / total) if total > 0 and coeff > 0 else None,
                "nuisance": atom_frequencies.get(int(token_id)),
            }
        )
    return {
        "example_id": example["example_id"],
        "layer": int(record["layer"]),
        "position": position,
        "requested_k": int(record["requested_k"]),
        "n_selected": int(record["n_selected"]),
        "data_status": "measured",
        "selected_atoms": atoms,
        "coefficient_sum": total if total > 0 else None,
        "top_coefficient": max(coefficients) if coefficients else None,
        "concentration": _concentration(coefficients),
        "reconstruction": dict(record["reconstruction"]),
        "cone_signature_digest": record.get("cone_signature", {}).get("digest"),
        "source_provenance": _reduced_source_provenance(record, run_id),
    }


def _cone_to_pursuit_trace(record: dict, *, example: dict) -> dict:
    """Playback trace from what the run actually recorded: selection order
    and residual-norm history. Per-step coefficients were NOT recorded and
    are exported as unavailable, never fabricated."""
    history = list(record["stopping"]["residual_norm_history"])
    n_iterations = int(record["stopping"]["n_iterations"])
    target_norm = float(record["reconstruction"]["target_norm"])
    selected = list(record["selected_token_ids"])
    labels = list(record["selected_labels"])
    coefficients = list(record["coefficients"])
    if len(history) != n_iterations + 1:
        raise ExportError(
            f"cone {record['prompt_slug']} L{record['layer']} P{record['position']}: "
            f"residual history length {len(history)} != n_iterations+1 "
            f"({n_iterations + 1})"
        )
    if len(selected) < n_iterations:
        raise ExportError(
            f"cone {record['prompt_slug']} L{record['layer']} P{record['position']}: "
            f"{len(selected)} selected atoms for {n_iterations} iterations"
        )
    steps = []
    for step in range(1, n_iterations + 1):
        residual_norm = float(history[step])
        relative = residual_norm / target_norm if target_norm > 0 else 0.0
        steps.append(
            {
                "step": step,
                "added_token_id": int(selected[step - 1]),
                "added_label": labels[step - 1],
                "support_after": [int(t) for t in selected[:step]],
                "residual_norm": residual_norm,
                "relative_residual": relative,
                "explained_fraction": 1.0 - relative * relative,
                "coefficients_after": None,
                "final_coefficient_zero": coefficients[step - 1] == 0.0,
            }
        )
    return {
        "example_id": example["example_id"],
        "layer": int(record["layer"]),
        "position": int(record["position"]),
        "requested_k": int(record["requested_k"]),
        "n_iterations": n_iterations,
        "stop_reason": record["stopping"]["stop_reason"],
        "data_status": "measured",
        "per_step_coefficients_available": False,
        "initial_residual_norm": float(history[0]),
        "target_norm": target_norm,
        "steps": steps,
    }


def _labelled(ids: Iterable[int], labels: dict[int, str]) -> list[dict]:
    return [
        {"token_id": int(i), "label": labels.get(int(i), "")} for i in sorted(ids)
    ]


def build_text_bundle(
    run_dir: str,
    *,
    prompts_path: str | None = None,
    analysis_dir: str | None = None,
    k: int = 10,
    layers: Sequence[int] | None = None,
    slugs: Sequence[str] | None = None,
    implementation_commit: str | None = None,
) -> tuple[dict, list[str]]:
    """Build a measured text bundle from a completed jspace run.

    Returns ``(bundle, warnings)``. The bundle is deterministic: same inputs,
    same output (creation time is the source run's recorded ``written_utc``).
    """
    run = load_run_artifacts(run_dir, k=k, layers=layers)
    warnings = list(run["warnings"])
    meta = run["run_metadata"]
    run_id = meta["run_id"]
    lens_fingerprint = meta.get("lens_verification", {}).get("file_sha256")
    model_repo = meta.get("config", {}).get("model", {}).get("repo_id")
    model_revision = meta.get("load_info", {}).get("model_revision") or meta.get(
        "config", {}
    ).get("model", {}).get("revision")

    prompt_texts = load_prompt_texts(prompts_path) if prompts_path else {}
    if not prompt_texts:
        warnings.append("no prompts file given; prompt_text will be null")
    atom_frequencies = load_atom_frequencies(analysis_dir) if analysis_dir else {}
    if analysis_dir and not atom_frequencies:
        warnings.append(
            f"analysis dir {analysis_dir} has no atom_frequencies.csv; "
            "nuisance metadata omitted"
        )

    eval_examples: dict[str, dict] = {}
    if run["eval_results"]:
        for entry in run["eval_results"].get("results", {}).get("examples", []):
            eval_examples[entry["slug"]] = entry
    eval_top_k = (
        run["eval_results"].get("results", {}).get("top_k")
        if run["eval_results"]
        else None
    )

    # Group capture metadata by slug -> positions.
    capture_by_slug: dict[str, list[dict]] = {}
    for item in meta.get("capture_meta", []):
        capture_by_slug.setdefault(item["slug"], []).append(item)
    selected_slugs = list(slugs) if slugs is not None else sorted(capture_by_slug)
    missing = sorted(set(selected_slugs) - set(capture_by_slug))
    if missing:
        raise ExportError(f"slugs not present in the run: {missing}")

    cones_by_key = { _cone_key(r): r for r in run["cones"] }

    examples: list[dict] = []
    layer_records: list[dict] = []
    bundle_cones: list[dict] = []
    pursuit_traces: list[dict] = []
    trajectories: list[dict] = []

    for slug in sorted(selected_slugs):
        items = sorted(capture_by_slug[slug], key=lambda i: i["position"])
        first = items[0]
        prompt_hash = first["prompt_hash"]
        eid = example_id("text", slug, prompt_hash)
        positions = [int(i["position"]) for i in items]
        text_entry = prompt_texts.get(slug)
        model_output = {
            str(int(i["position"])): {
                "input_token_id": int(i["input_token_id"]),
                "input_token": i["input_token"],
                "model_top1_id": int(i["model_top1_id"]),
                "model_top1_token": i["model_top1_token"],
                "model_topk": None,
            }
            for i in items
        }
        example = {
            "example_id": eid,
            "prompt_slug": slug,
            "prompt_hash": prompt_hash,
            "category": first["category"],
            "format": first["format"],
            "modality": "text",
            "display_title": (text_entry or {}).get("text") or slug,
            "prompt_text": (text_entry or {}).get("text"),
            "data_status": "measured",
            "seq_len": int(first["seq_len"]),
            "selected_positions": positions,
            "model_output": model_output,
            "strength": _strength(eval_examples.get(slug), positions),
            "selection_reason": None,
            "input": {
                "text": {
                    "token_ids": None,
                    "token_labels": None,
                    "positions_available": positions,
                    "special_token_flags": None,
                    "prompt_text_is_pre_template": (
                        text_entry["is_pre_template"] if text_entry else None
                    ),
                    "tokenization_available": False,
                },
                "image": None,
                "audio": None,
            },
        }
        examples.append(example)

        eval_example = eval_examples.get(slug)
        for item in items:
            position = int(item["position"])
            for layer in run["layers"]:
                cone = cones_by_key.get((prompt_hash, position, layer))
                rank = None
                overlap = None
                eval_metadata = None
                if eval_example:
                    per_layer = eval_example.get("layers", {}).get(str(layer), {})
                    jl = per_layer.get("jlens")
                    if jl:
                        try:
                            pos_index = list(eval_example["positions"]).index(position)
                            rank = int(jl["rank_of_model_top1"][pos_index])
                        except (ValueError, IndexError):
                            rank = None
                        overlap = jl.get("topk_overlap_with_model")
                        eval_metadata = {
                            "top_k": eval_top_k,
                            "rank_convention": "0 = argmax",
                            "overlap_aggregation": (
                                "topk_overlap_with_model is aggregated over the "
                                "example's positions in the source eval"
                            ),
                            "readout": "pre-softcap W_U norm(J h)",
                        }
                recon = cone["reconstruction"] if cone else {}
                layer_records.append(
                    {
                        "example_id": eid,
                        "layer": layer,
                        "position": position,
                        "source_site": "block_output",
                        "data_status": "measured",
                        "input_token_id": int(item["input_token_id"]),
                        "input_token": item["input_token"],
                        "model_topk": None,
                        "jlens_topk": None,
                        "rank_of_model_top1": rank,
                        "topk_overlap_with_model": overlap,
                        "eval_metadata": eval_metadata,
                        "target_activation_norm": recon.get("target_norm"),
                        "residual_norm": recon.get("residual_norm"),
                        "relative_residual": recon.get("relative_residual"),
                        "explained_fraction": recon.get("explained_fraction"),
                    }
                )
                if cone:
                    bundle_cones.append(
                        _cone_to_bundle_record(
                            cone,
                            example=example,
                            run_id=run_id,
                            atom_frequencies=atom_frequencies,
                        )
                    )
                    pursuit_traces.append(_cone_to_pursuit_trace(cone, example=example))
                else:
                    warnings.append(
                        f"no k={k} cone for {slug} position {position} layer {layer}"
                    )

        # Trajectories for this example (labels resolved from cone records).
        for transition in run["trajectories"]:
            if (
                transition["prompt_hash"] != prompt_hash
                or int(transition["position"]) not in positions
            ):
                continue
            position = int(transition["position"])
            cone_from = cones_by_key.get(
                (prompt_hash, position, int(transition["layer_from"]))
            )
            cone_to = cones_by_key.get(
                (prompt_hash, position, int(transition["layer_to"]))
            )
            labels: dict[int, str] = {}
            for cone in (cone_from, cone_to):
                if cone:
                    labels.update(
                        zip(
                            (int(t) for t in cone["effective_token_ids"]),
                            cone["effective_labels"],
                            strict=True,
                        )
                    )
            ids_from = (
                {int(t) for t in cone_from["effective_token_ids"]} if cone_from else set()
            )
            ids_to = (
                {int(t) for t in cone_to["effective_token_ids"]} if cone_to else set()
            )
            retained = ids_from & ids_to
            top1 = model_output.get(str(position), {}).get("model_top1_id")
            trajectories.append(
                {
                    "example_id": eid,
                    "position": position,
                    "layer_from": int(transition["layer_from"]),
                    "layer_to": int(transition["layer_to"]),
                    "retained_atoms": _labelled(retained, labels),
                    "entered_atoms": _labelled(
                        transition["entered_token_ids"],
                        dict(
                            zip(
                                (int(t) for t in transition["entered_token_ids"]),
                                transition["entered_labels"],
                                strict=True,
                            )
                        ),
                    ),
                    "exited_atoms": _labelled(
                        transition["exited_token_ids"],
                        dict(
                            zip(
                                (int(t) for t in transition["exited_token_ids"]),
                                transition["exited_labels"],
                                strict=True,
                            )
                        ),
                    ),
                    "jaccard": float(transition["active_set_overlap"]["jaccard"]),
                    "weighted_similarity": float(transition["weighted_similarity"]),
                    "explained_fraction_from": transition["explained_fraction_from"],
                    "explained_fraction_to": transition["explained_fraction_to"],
                    "delta_explained_fraction": transition["delta_explained_fraction"],
                    "output_token_persistence": (
                        {
                            "in_from": int(top1) in ids_from,
                            "in_to": int(top1) in ids_to,
                        }
                        if top1 is not None
                        else None
                    ),
                    "data_status": "measured",
                }
            )

    bundle = {
        "schema": BUNDLE_SCHEMA,
        "provenance": {
            "schema_version": SCHEMA_VERSION,
            "exporter_version": EXPORTER_VERSION,
            "source_run_ids": [run_id],
            "source_artifact_fingerprints": dict(
                sorted(run["artifact_fingerprints"].items())
            ),
            "lens_fingerprint": lens_fingerprint,
            "model_repo_id": model_repo,
            "model_revision": model_revision,
            "implementation_commit": implementation_commit,
            "created_utc": meta.get("written_utc"),
            "data_status": "measured",
            "modalities_present": ["text"],
            "merged_bundles": [],
            "notes": (
                "Exported from the completed J-space gradient-pursuit run on the "
                "frozen pilot lens. Per-step pursuit coefficients and per-layer "
                "J-lens top-k lists were not persisted by that run and are "
                "exported as unavailable, not fabricated."
            ),
        },
        "examples": sorted(examples, key=lambda e: e["example_id"]),
        "layer_records": sorted(
            layer_records, key=lambda r: (r["example_id"], r["layer"], r["position"])
        ),
        "cones": sorted(
            bundle_cones, key=lambda r: (r["example_id"], r["layer"], r["position"])
        ),
        "pursuit_traces": sorted(
            pursuit_traces, key=lambda r: (r["example_id"], r["layer"], r["position"])
        ),
        "trajectories": sorted(
            trajectories,
            key=lambda r: (r["example_id"], r["position"], r["layer_from"]),
        ),
        "causal_records": [],
    }
    assert_no_absolute_paths(bundle)
    return bundle, warnings


# ------------------------------------------------ notebook bundle assembly


def make_provenance(
    *,
    source_run_ids: Sequence[str],
    model_repo_id: str,
    model_revision: str,
    created_utc: str,
    data_status: str,
    modalities_present: Sequence[str],
    exporter_version: str = EXPORTER_VERSION,
    lens_fingerprint: str | None = None,
    source_artifact_fingerprints: dict[str, str] | None = None,
    implementation_commit: str | None = None,
    notes: str = "",
) -> dict:
    """Bundle provenance block for notebook-produced bundles (causal smoke
    run, multimodal capture)."""
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "exporter_version": exporter_version,
        "source_run_ids": sorted(source_run_ids),
        "source_artifact_fingerprints": dict(
            sorted((source_artifact_fingerprints or {}).items())
        ),
        "model_repo_id": model_repo_id,
        "model_revision": model_revision,
        "implementation_commit": implementation_commit,
        "created_utc": created_utc,
        "data_status": data_status,
        "modalities_present": sorted(modalities_present),
        "merged_bundles": [],
        "notes": notes,
    }
    if lens_fingerprint is not None:
        provenance["lens_fingerprint"] = lens_fingerprint
    return provenance


def assemble_bundle(
    *,
    provenance: dict,
    examples: Sequence[dict] = (),
    layer_records: Sequence[dict] = (),
    cones: Sequence[dict] = (),
    pursuit_traces: Sequence[dict] = (),
    trajectories: Sequence[dict] = (),
    causal_records: Sequence[dict] = (),
    causal_baseline_parity: dict | None = None,
    notes: str | None = None,
) -> dict:
    """Assemble a bundle with canonical section ordering and path checks —
    the single constructor notebooks use so their bundles match exactly what
    :func:`build_text_bundle` produces structurally."""
    bundle = {
        "schema": BUNDLE_SCHEMA,
        "provenance": provenance,
        "examples": sorted(examples, key=lambda e: e["example_id"]),
        "layer_records": sorted(
            layer_records, key=lambda r: (r["example_id"], r["layer"], r["position"])
        ),
        "cones": sorted(
            cones, key=lambda r: (r["example_id"], r["layer"], r["position"])
        ),
        "pursuit_traces": sorted(
            pursuit_traces, key=lambda r: (r["example_id"], r["layer"], r["position"])
        ),
        "trajectories": sorted(
            trajectories,
            key=lambda r: (r["example_id"], r["position"], r["layer_from"]),
        ),
        "causal_records": sorted(causal_records, key=lambda r: r["condition_id"]),
    }
    if causal_baseline_parity is not None:
        bundle["causal_baseline_parity"] = causal_baseline_parity
    if notes is not None:
        bundle["notes"] = notes
    assert_no_absolute_paths(bundle)
    return bundle


def intervention_to_causal_record(record: dict) -> dict:
    """Project a ``jlens.interventions.record.v1`` JSONL record onto the
    bundle's ``causalRecord`` shape (dropping backend-internal fields)."""
    keys = (
        "condition_id",
        "example_id",
        "layer",
        "position",
        "target_kind",
        "atom_token_id",
        "atom_label",
        "multiplier",
        "status",
        "norm_preserving",
        "delta_norm",
        "activation_norm",
        "delta_to_activation_ratio",
        "target_token_id",
        "target_token",
        "target_logit_before",
        "target_logit_after",
        "target_logit_delta",
        "target_rank_before",
        "target_rank_after",
        "target_prob_before",
        "target_prob_after",
        "top1_before",
        "top1_after",
        "top10_before",
        "top10_after",
        "top10_overlap",
        "kl_divergence_after_vs_before",
        "completion_before",
        "completion_after",
        "control_family",
        "matched_target_condition_id",
        "provenance",
    )
    return {key: record.get(key) for key in keys if key in record}


# ---------------------------------------------------------------- merging


def merge_bundles(base: dict, extra: dict) -> dict:
    """Merge ``extra`` (e.g. a causal or multimodal bundle) into ``base``.

    Records are deduplicated by their stable identity (example_id /
    condition_id / record coordinates); on collision the *extra* record wins,
    which is the documented replacement workflow for fixtures superseded by
    measured runs. Provenance of the merged bundle is appended to
    ``merged_bundles`` and modality support is unioned. Returns a new dict;
    neither input is mutated.
    """
    for bundle in (base, extra):
        if bundle.get("schema") != BUNDLE_SCHEMA:
            raise ExportError(
                f"cannot merge: schema {bundle.get('schema')!r} != {BUNDLE_SCHEMA}"
            )
    merged = json.loads(json.dumps(base))
    incoming = json.loads(json.dumps(extra))

    def merge_section(section: str, key) -> None:
        combined = {key(r): r for r in merged.get(section, [])}
        combined.update({key(r): r for r in incoming.get(section, [])})
        merged[section] = [combined[k] for k in sorted(combined)]

    merge_section("examples", lambda r: r["example_id"])
    merge_section(
        "layer_records", lambda r: (r["example_id"], r["layer"], r["position"])
    )
    merge_section("cones", lambda r: (r["example_id"], r["layer"], r["position"]))
    merge_section(
        "pursuit_traces", lambda r: (r["example_id"], r["layer"], r["position"])
    )
    merge_section(
        "trajectories",
        lambda r: (r["example_id"], r["position"], r["layer_from"], r["layer_to"]),
    )
    merge_section("causal_records", lambda r: r["condition_id"])

    if "causal_baseline_parity" in incoming:
        merged["causal_baseline_parity"] = incoming["causal_baseline_parity"]

    prov = merged["provenance"]
    extra_prov = incoming.get("provenance", {})
    prov["merged_bundles"] = list(prov.get("merged_bundles", [])) + [extra_prov]
    prov["modalities_present"] = sorted(
        set(prov.get("modalities_present", []))
        | set(extra_prov.get("modalities_present", []))
    )
    prov["source_run_ids"] = sorted(
        set(prov.get("source_run_ids", [])) | set(extra_prov.get("source_run_ids", []))
    )
    # The merged bundle keeps the weakest data status so a fixture can never
    # masquerade as measured after a merge.
    order = {"measured": 0, "imported": 1, "synthetic_fixture": 2}
    statuses = [
        prov.get("data_status", "measured"),
        extra_prov.get("data_status", "measured"),
    ]
    prov["data_status"] = max(statuses, key=lambda s: order.get(s, 2))
    assert_no_absolute_paths(merged)
    return merged


# ------------------------------------------------------------- rendering


def assert_no_absolute_paths(obj: Any, *, _path: str = "$") -> None:
    """Raise :class:`ExportError` if any string in ``obj`` looks like an
    absolute local filesystem path (Colab, home directories, Windows user
    paths). Exported browser data must be machine-independent."""
    if isinstance(obj, str):
        if _ABSOLUTE_PATH_PATTERN.search(obj):
            raise ExportError(f"absolute local path leaked into bundle at {_path}: {obj!r}")
    elif isinstance(obj, dict):
        for key, value in obj.items():
            assert_no_absolute_paths(value, _path=f"{_path}.{key}")
    elif isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            assert_no_absolute_paths(value, _path=f"{_path}[{index}]")


def canonical_json(bundle: dict) -> str:
    """Deterministic rendering: sorted keys, compact separators, ensure_ascii
    off (labels keep their unicode), trailing newline."""
    return (
        json.dumps(bundle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    )


def validate_bundle(bundle: dict, schema_path: str) -> None:
    """Validate against the explorer bundle JSON Schema (jsonschema must be
    installed; it is part of the dev extra)."""
    import jsonschema

    schema = _load_json(schema_path)
    jsonschema.validate(instance=bundle, schema=schema)


def write_bundle(bundle: dict, out_path: str, *, schema_path: str | None = None) -> str:
    """Validate (when a schema is given), render canonically, write, and
    return the bundle's sha256 fingerprint."""
    assert_no_absolute_paths(bundle)
    if schema_path:
        validate_bundle(bundle, schema_path)
    rendered = canonical_json(bundle)
    directory = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(directory, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(rendered)
    return "sha256:" + hashlib.sha256(rendered.encode("utf-8")).hexdigest()
