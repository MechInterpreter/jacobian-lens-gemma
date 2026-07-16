# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Deterministic offline analysis of a completed ``jspace_pursuit`` run.

Consumes the artifacts a run directory already contains (cone records,
trajectories, candidate-ignition records, recurring signatures,
``eval_v2_results.json``, ``run_metadata.json``) and produces derived
summaries: an integrity report, per-(layer, k, format, category, position)
metric tables, marginal-k gains, cross-k active-set stability,
layer-transition summaries, atom frequency/enrichment tables,
similarity-group reports, and an evaluation-control summary.

Everything here is **read-only with respect to the run directory** — no
function in this module writes inside it, repairs records, or loads a model.
Outputs go to a separate report directory (see
``scripts/analyze_jspace.py``). All orderings and tie-breaks are
deterministic so repeated runs produce byte-identical outputs.
"""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from jlens.cones import CONE_RECORD_SCHEMA, TRANSITION_SCHEMA, cone_signature
from jlens.ignition import IGNITION_RECORD_SCHEMA
from jlens.similarity import (
    atom_enrichment,
    atom_selection_frequencies,
    coefficient_map,
    inverse_frequency_weights,
    jaccard_similarity,
    output_token_recurrence,
    similarity_groups,
    sparse_cosine,
    threshold_sensitivity,
    top_m_atoms,
    weighted_jaccard,
)

#: Fields that identify one decomposed activation across k values.
MATCH_FIELDS = ("prompt_hash", "prompt_slug", "format", "layer", "position")

_EVAL_VARIANTS = (
    "jlens",
    "logit_lens",
    "permuted",
    "random",
    "adjacent_layer",
    "distant_layer",
    "shuffled_layer",
)


# ------------------------------------------------------------------- loading


@dataclass
class RunArtifacts:
    """In-memory view of one run directory (read-only)."""

    run_dir: str
    metadata: dict
    cones: dict[tuple[int, int], list[dict]]  # (layer, k) -> records
    trajectories: dict[int, list[dict]]  # k -> transition records
    ignition: dict[int, list[dict]]  # k -> candidate records
    signatures: dict[int, list[dict]]  # k -> recurring-signature rows
    evaluation: dict  # eval_v2_results.json payload
    layers: list[int] = field(default_factory=list)
    k_values: list[int] = field(default_factory=list)

    def all_cone_records(self) -> list[dict]:
        """Every cone record, ordered by (layer, k, file order)."""
        out: list[dict] = []
        for key in sorted(self.cones):
            out.extend(self.cones[key])
        return out

    def capture_meta_index(self) -> dict[tuple, dict]:
        """capture_meta entries keyed by (prompt_hash, position)."""
        return {
            (m["prompt_hash"], m["position"]): m
            for m in self.metadata.get("capture_meta", [])
        }


def _read_json(path: str):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_run(run_dir: str) -> RunArtifacts:
    """Load a completed jspace run directory. Raises ``FileNotFoundError``
    for missing required artifacts; malformed JSON propagates as-is (the
    integrity checker reports it — nothing is silently repaired)."""
    metadata = _read_json(os.path.join(run_dir, "run_metadata.json"))
    layers = list(metadata["config"]["decomposition"]["layers"])
    k_values = list(metadata["config"]["decomposition"]["k_values"])
    artifacts = os.path.join(run_dir, "artifacts")

    cones: dict[tuple[int, int], list[dict]] = {}
    for layer in layers:
        for k in k_values:
            path = os.path.join(artifacts, "cones", f"cones_layer{layer}_k{k}.json")
            cones[(layer, k)] = _read_json(path)

    trajectories = {
        k: _read_json(os.path.join(artifacts, f"trajectories_k{k}.json"))
        for k in k_values
    }
    ignition = {
        k: _read_json(os.path.join(artifacts, f"ignition_candidates_k{k}.json"))
        for k in k_values
    }
    signatures = {
        k: _read_json(os.path.join(artifacts, f"recurring_signatures_k{k}.json"))
        for k in k_values
    }
    evaluation = _read_json(os.path.join(artifacts, "eval_v2_results.json"))
    return RunArtifacts(
        run_dir=run_dir,
        metadata=metadata,
        cones=cones,
        trajectories=trajectories,
        ignition=ignition,
        signatures=signatures,
        evaluation=evaluation,
        layers=layers,
        k_values=k_values,
    )


# ----------------------------------------------------------------- integrity


def _is_finite_number(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def check_cone_record(record: Mapping, expected_layer: int, expected_k: int) -> list[str]:
    """Structural and numerical checks for one cone record. Returns a list
    of human-readable problems (empty = clean)."""
    problems: list[str] = []
    where = (
        f"{record.get('prompt_slug')}/{record.get('format')}/L{expected_layer}"
        f"/k{expected_k}/pos{record.get('position')}"
    )
    if record.get("schema") != CONE_RECORD_SCHEMA:
        problems.append(f"{where}: bad schema {record.get('schema')!r}")
        return problems
    if record.get("layer") != expected_layer:
        problems.append(f"{where}: layer {record.get('layer')} != file layer")
    if record.get("requested_k") != expected_k:
        problems.append(f"{where}: requested_k {record.get('requested_k')} != file k")

    ids = record["effective_token_ids"]
    coeffs = record["effective_coefficients"]
    if len(ids) != len(set(ids)):
        problems.append(f"{where}: duplicate token ids in effective set")
    if len(ids) != len(coeffs) or len(ids) != len(record["effective_labels"]):
        problems.append(f"{where}: misaligned effective ids/coeffs/labels")
    for c in coeffs:
        if not _is_finite_number(c):
            problems.append(f"{where}: nonfinite coefficient {c!r}")
            break
        if c <= 0:
            problems.append(f"{where}: non-positive effective coefficient {c!r}")
            break
    if record["n_selected"] > record["requested_k"]:
        problems.append(f"{where}: n_selected exceeds requested_k")
    if len(record["selected_token_ids"]) != record["n_selected"]:
        problems.append(f"{where}: selected_token_ids length != n_selected")

    recon = record["reconstruction"]
    for name in ("target_norm", "residual_norm", "relative_residual", "explained_fraction"):
        if not _is_finite_number(recon.get(name)):
            problems.append(f"{where}: nonfinite reconstruction.{name}")
    if _is_finite_number(recon.get("residual_norm")) and _is_finite_number(
        recon.get("target_norm")
    ):
        if recon["residual_norm"] > recon["target_norm"] * (1 + 1e-6):
            problems.append(f"{where}: residual_norm exceeds target_norm")

    stopping = record["stopping"]
    history = stopping["residual_norm_history"]
    if len(history) != stopping["n_iterations"] + 1:
        problems.append(f"{where}: history length != n_iterations + 1")
    for prev, curr in zip(history, history[1:], strict=False):
        if curr > prev * (1 + 1e-6):
            problems.append(f"{where}: residual history not non-increasing")
            break
    if stopping["stop_reason"] == "max_atoms" and record["n_selected"] != record["requested_k"]:
        problems.append(f"{where}: stop_reason max_atoms but n_selected < requested_k")
    if (
        stopping["stop_reason"] == "no_positive_correlation"
        and record["n_selected"] >= record["requested_k"]
    ):
        problems.append(f"{where}: no_positive_correlation stop with full selection")

    expected_signature = cone_signature(ids)
    if expected_signature["digest"] != record["cone_signature"]["digest"]:
        problems.append(f"{where}: cone_signature digest mismatch")
    return problems


def check_integrity(art: RunArtifacts) -> dict:
    """Full integrity report for the run (machine-readable). Nothing is
    repaired; every anomaly is reported where found."""
    issues: list[str] = []
    notes: list[str] = []
    meta = art.metadata
    config = meta["config"]

    expected_activations = meta.get("n_activations_per_layer")
    capture_index = art.capture_meta_index()

    counts: dict[str, int] = {}
    duplicate_keys: list[str] = []
    for (layer, k), records in sorted(art.cones.items()):
        counts[f"cones_layer{layer}_k{k}"] = len(records)
        if expected_activations is not None and len(records) != expected_activations:
            issues.append(
                f"cones_layer{layer}_k{k}: {len(records)} records, expected "
                f"{expected_activations}"
            )
        seen: set[tuple] = set()
        for record in records:
            key = tuple(record.get(f) for f in MATCH_FIELDS)
            if key in seen:
                duplicate_keys.append(f"cones_layer{layer}_k{k}: duplicate {key}")
            seen.add(key)
            issues.extend(check_cone_record(record, layer, k))
            capture = capture_index.get((record["prompt_hash"], record["position"]))
            if capture is None:
                issues.append(
                    f"L{layer}/k{k}/{record.get('prompt_slug')}: prompt_hash not in "
                    "capture_meta"
                )
            else:
                if capture["slug"] != record.get("prompt_slug") or capture[
                    "format"
                ] != record.get("format"):
                    issues.append(
                        f"L{layer}/k{k}/{record.get('prompt_slug')}: slug/format "
                        "disagrees with capture_meta"
                    )
                if capture["model_top1_id"] != record["run_provenance"].get(
                    "model_top1_id"
                ):
                    issues.append(
                        f"L{layer}/k{k}/{record.get('prompt_slug')}/pos"
                        f"{record['position']}: model_top1_id disagrees with "
                        "capture_meta"
                    )
    issues.extend(duplicate_keys)

    # Trajectory / ignition / signature counts per k.
    n_prompt_positions = len(capture_index) or None
    n_transitions_expected = (
        n_prompt_positions * (len(art.layers) - 1) if n_prompt_positions else None
    )
    for k in art.k_values:
        counts[f"trajectories_k{k}"] = len(art.trajectories[k])
        counts[f"ignition_candidates_k{k}"] = len(art.ignition[k])
        counts[f"recurring_signatures_k{k}"] = len(art.signatures[k])
        if n_transitions_expected is not None:
            for name, rows in (
                (f"trajectories_k{k}", art.trajectories[k]),
                (f"ignition_candidates_k{k}", art.ignition[k]),
            ):
                if len(rows) != n_transitions_expected:
                    issues.append(
                        f"{name}: {len(rows)} records, expected {n_transitions_expected}"
                    )
        for row in art.trajectories[k]:
            if row.get("schema") != TRANSITION_SCHEMA:
                issues.append(f"trajectories_k{k}: bad schema {row.get('schema')!r}")
                break
        for row in art.ignition[k]:
            if row.get("schema") != IGNITION_RECORD_SCHEMA:
                issues.append(
                    f"ignition_candidates_k{k}: bad schema {row.get('schema')!r}"
                )
                break
        signature_total = sum(row["count"] for row in art.signatures[k])
        cone_total = sum(len(art.cones[(layer, k)]) for layer in art.layers)
        if signature_total != cone_total:
            issues.append(
                f"recurring_signatures_k{k}: counts sum to {signature_total}, "
                f"expected {cone_total}"
            )

    # Lens / model provenance cross-checks.
    lens_check = meta.get("lens_verification", {})
    if lens_check.get("file_sha256") != config["lens"]["expect_file_sha256"]:
        issues.append("lens_verification.file_sha256 != config expectation")
    load_info = meta.get("load_info", {})
    if load_info.get("model_revision") != config["model"]["revision"]:
        issues.append("load_info.model_revision != config.model.revision")

    # Known provenance inconsistency: the static config is recorded verbatim,
    # so allow_model_load=false coexists with a real load_info block. Report,
    # never repair (see jlens.metadata.execution_record for the forward fix).
    model_loaded = bool(load_info.get("model_repo_id"))
    if model_loaded and config["model"].get("allow_model_load") is False:
        notes.append(
            "config.model.allow_model_load is false but load_info shows the model "
            "was loaded: the config block records the *static* notebook YAML, and "
            "the run resolved allow-model-load at execution time without writing "
            "the resolved value back. Future runs should record "
            "jlens.metadata.execution_record(...) alongside the static config."
        )

    # Evaluation payload consistency.
    eval_results = art.evaluation.get("results", {})
    if sorted(eval_results.get("layers", [])) != sorted(art.layers):
        issues.append("eval layers != decomposition layers")
    if eval_results.get("n_prompts") != (
        n_prompt_positions // 2 if n_prompt_positions else None
    ):
        notes.append(
            f"eval n_prompts={eval_results.get('n_prompts')} vs "
            f"{n_prompt_positions} captured prompt/position pairs"
        )

    return {
        "schema": "jlens.analysis.integrity.v1",
        "run_id": meta.get("run_id"),
        "run_dir_recorded": meta.get("run_dir"),
        "model_revision": load_info.get("model_revision"),
        "local_commit": meta.get("environment", {}).get("local_commit"),
        "lens_fingerprint": lens_check.get("file_sha256"),
        "layers": art.layers,
        "k_values": art.k_values,
        "record_counts": counts,
        "n_capture_positions": n_prompt_positions,
        "issues": issues,
        "notes": notes,
        "clean": not issues,
    }


# ------------------------------------------------------------ record metrics


def record_metrics(record: Mapping) -> dict:
    """Scalar metrics for one cone record (pure; no rounding)."""
    coeffs = list(record["effective_coefficients"])
    ids = list(record["effective_token_ids"])
    total = sum(coeffs)
    largest = max(coeffs) if coeffs else 0.0
    herfindahl = (
        sum((c / total) ** 2 for c in coeffs) if total > 0 else 0.0
    )
    top1 = record.get("run_provenance", {}).get("model_top1_id")
    included = top1 is not None and int(top1) in ids
    share = (
        coeffs[ids.index(int(top1))] / total if included and total > 0 else 0.0
    )
    history = record["stopping"]["residual_norm_history"]
    return {
        "explained_fraction": record["reconstruction"]["explained_fraction"],
        "relative_residual": record["reconstruction"]["relative_residual"],
        "target_norm": record["reconstruction"]["target_norm"],
        "residual_norm": record["reconstruction"]["residual_norm"],
        "coefficient_sum": total,
        "largest_coefficient": largest,
        "largest_coefficient_share": (largest / total) if total > 0 else 0.0,
        "herfindahl": herfindahl,
        "active_set_size": len(ids),
        "n_iterations": record["stopping"]["n_iterations"],
        "residual_decrease": history[0] - history[-1],
        "residual_decrease_relative": (
            (history[0] - history[-1]) / history[0] if history[0] > 0 else 0.0
        ),
        "output_token_included": included,
        "output_token_share": share,
        "output_token_is_top": bool(
            included and coeffs and ids[coeffs.index(largest)] == int(top1)
        ),
        "stop_reason": record["stopping"]["stop_reason"],
    }


_DISTRIBUTION_METRICS = (
    "explained_fraction",
    "relative_residual",
    "target_norm",
    "residual_norm",
    "coefficient_sum",
    "largest_coefficient",
    "largest_coefficient_share",
    "herfindahl",
)


def _distribution(values: Sequence[float]) -> dict:
    values = list(values)
    if not values:
        return {"mean": None, "median": None, "std": None, "q1": None, "q3": None}
    if len(values) == 1:
        return {
            "mean": values[0],
            "median": values[0],
            "std": 0.0,
            "q1": values[0],
            "q3": values[0],
        }
    quartiles = statistics.quantiles(values, n=4, method="inclusive")
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "std": statistics.stdev(values),
        "q1": quartiles[0],
        "q3": quartiles[2],
    }


def _category(record: Mapping) -> str:
    return record.get("run_provenance", {}).get("category", "unknown")


def _metric_row(layer, k, fmt, category, position, records: list[dict]) -> dict:
    metrics = [record_metrics(r) for r in records]
    row: dict = {
        "layer": layer,
        "k": k,
        "format": fmt,
        "category": category,
        "position": position,
        "n": len(records),
    }
    for name in _DISTRIBUTION_METRICS:
        dist = _distribution([m[name] for m in metrics])
        for stat, value in dist.items():
            row[f"{name}_{stat}"] = value
    included = [m for m in metrics if m["output_token_included"]]
    row["output_token_inclusion_rate"] = (
        len(included) / len(metrics) if metrics else None
    )
    row["output_token_share_when_present"] = (
        statistics.fmean([m["output_token_share"] for m in included])
        if included
        else None
    )
    row["output_token_top_rate"] = (
        sum(m["output_token_is_top"] for m in metrics) / len(metrics)
        if metrics
        else None
    )
    row["active_set_size_mean"] = statistics.fmean(
        [m["active_set_size"] for m in metrics]
    )
    row["n_iterations_mean"] = statistics.fmean([m["n_iterations"] for m in metrics])
    row["residual_decrease_relative_mean"] = statistics.fmean(
        [m["residual_decrease_relative"] for m in metrics]
    )
    stops = Counter(m["stop_reason"] for m in metrics)
    row["stop_reasons"] = ";".join(f"{k_}={v}" for k_, v in sorted(stops.items()))
    return row


def metrics_table(art: RunArtifacts) -> list[dict]:
    """Aggregate metric rows per (layer, k) under several stratifications:
    overall, by format, by position, by category, and by format x position.
    Wildcard strata are recorded as ``"all"``."""
    rows: list[dict] = []
    for (layer, k), records in sorted(art.cones.items()):
        cuts: list[tuple[str, str | int, list[dict]]] = []
        cuts.append(("all", "all", "all", records))
        for fmt in sorted({r["format"] for r in records}):
            cuts.append((fmt, "all", "all", [r for r in records if r["format"] == fmt]))
        for pos in sorted({r["position"] for r in records}):
            cuts.append(
                ("all", "all", pos, [r for r in records if r["position"] == pos])
            )
        for cat in sorted({_category(r) for r in records}):
            cuts.append(
                ("all", cat, "all", [r for r in records if _category(r) == cat])
            )
        for fmt in sorted({r["format"] for r in records}):
            for pos in sorted({r["position"] for r in records}):
                subset = [
                    r for r in records if r["format"] == fmt and r["position"] == pos
                ]
                if subset:
                    cuts.append((fmt, "all", pos, subset))
        for fmt, cat, pos, subset in cuts:
            rows.append(_metric_row(layer, k, fmt, cat, pos, subset))
    return rows


# -------------------------------------------------------------- k comparison


def record_key(record: Mapping) -> tuple:
    return tuple(record[f] for f in MATCH_FIELDS)


def match_across_k(art: RunArtifacts) -> dict[tuple, dict[int, dict]]:
    """Records for the same activation keyed by MATCH_FIELDS, then by k.
    Raises on duplicate keys within one (layer, k) file."""
    matched: dict[tuple, dict[int, dict]] = defaultdict(dict)
    for (layer, k), records in sorted(art.cones.items()):
        for record in records:
            key = record_key(record)
            if k in matched[key]:
                raise ValueError(f"duplicate record for {key} at k={k}")
            matched[key][k] = record
    return dict(matched)


def k_marginal_gains(art: RunArtifacts) -> list[dict]:
    """Paired marginal reconstruction changes between k values, per layer
    and format (plus 'all')."""
    matched = match_across_k(art)
    pairs = [
        (a, b)
        for i, a in enumerate(art.k_values)
        for b in art.k_values[i + 1 :]
    ]
    rows: list[dict] = []
    for layer in art.layers:
        for fmt in ("all", "plain", "chat"):
            for k_small, k_large in pairs:
                deltas_ef, rel_ef, deltas_res, rel_res = [], [], [], []
                for key, by_k in sorted(matched.items()):
                    record = by_k.get(k_small)
                    other = by_k.get(k_large)
                    if record is None or other is None or record["layer"] != layer:
                        continue
                    if fmt != "all" and record["format"] != fmt:
                        continue
                    ef_a = record["reconstruction"]["explained_fraction"]
                    ef_b = other["reconstruction"]["explained_fraction"]
                    res_a = record["reconstruction"]["residual_norm"]
                    res_b = other["reconstruction"]["residual_norm"]
                    deltas_ef.append(ef_b - ef_a)
                    if ef_a > 0:
                        rel_ef.append((ef_b - ef_a) / ef_a)
                    deltas_res.append(res_b - res_a)
                    if res_a > 0:
                        rel_res.append((res_b - res_a) / res_a)
                if not deltas_ef:
                    continue
                rows.append(
                    {
                        "layer": layer,
                        "format": fmt,
                        "k_from": k_small,
                        "k_to": k_large,
                        "n": len(deltas_ef),
                        "mean_delta_explained_fraction": statistics.fmean(deltas_ef),
                        "median_delta_explained_fraction": statistics.median(deltas_ef),
                        "mean_relative_gain_explained_fraction": (
                            statistics.fmean(rel_ef) if rel_ef else None
                        ),
                        "mean_delta_residual_norm": statistics.fmean(deltas_res),
                        "mean_relative_change_residual_norm": (
                            statistics.fmean(rel_res) if rel_res else None
                        ),
                        "fraction_improved": sum(d > 0 for d in deltas_ef)
                        / len(deltas_ef),
                    }
                )
    return rows


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Spearman rank correlation with average ranks (None for n < 2 or a
    constant series)."""
    n = len(xs)
    if n < 2:
        return None

    def ranks(values: Sequence[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: (values[i], i))
        result = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and values[order[j + 1]] == values[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for idx in order[i : j + 1]:
                result[idx] = avg
            i = j + 1
        return result

    rx, ry = ranks(xs), ranks(ys)
    mean_x, mean_y = statistics.fmean(rx), statistics.fmean(ry)
    cov = sum((a - mean_x) * (b - mean_y) for a, b in zip(rx, ry, strict=True))
    var_x = sum((a - mean_x) ** 2 for a in rx)
    var_y = sum((b - mean_y) ** 2 for b in ry)
    if var_x == 0 or var_y == 0:
        return None
    return cov / math.sqrt(var_x * var_y)


def cross_k_stability(art: RunArtifacts) -> list[dict]:
    """Per (layer, k-pair, format): support containment, Jaccard, weighted
    Jaccard, sparse cosine, rank stability of shared atoms, output-token
    persistence, and top-atom persistence — averaged over matched
    activations."""
    matched = match_across_k(art)
    pairs = [
        (a, b)
        for i, a in enumerate(art.k_values)
        for b in art.k_values[i + 1 :]
    ]
    rows: list[dict] = []
    for layer in art.layers:
        for fmt in ("all", "plain", "chat"):
            for k_small, k_large in pairs:
                acc = defaultdict(list)
                for key, by_k in sorted(matched.items()):
                    small = by_k.get(k_small)
                    large = by_k.get(k_large)
                    if small is None or large is None or small["layer"] != layer:
                        continue
                    if fmt != "all" and small["format"] != fmt:
                        continue
                    ids_s = set(small["effective_token_ids"])
                    ids_l = set(large["effective_token_ids"])
                    map_s, map_l = coefficient_map(small), coefficient_map(large)
                    acc["containment"].append(
                        len(ids_s & ids_l) / len(ids_s) if ids_s else 1.0
                    )
                    acc["jaccard"].append(jaccard_similarity(ids_s, ids_l))
                    acc["weighted_jaccard"].append(weighted_jaccard(map_s, map_l))
                    acc["cosine"].append(sparse_cosine(map_s, map_l))
                    shared = sorted(ids_s & ids_l)
                    rho = _spearman(
                        [map_s[i] for i in shared], [map_l[i] for i in shared]
                    )
                    if rho is not None:
                        acc["shared_atom_rank_correlation"].append(rho)
                    top1 = small.get("run_provenance", {}).get("model_top1_id")
                    if top1 is not None and int(top1) in ids_s:
                        acc["output_token_persistence"].append(
                            1.0 if int(top1) in ids_l else 0.0
                        )
                    top_atom = top_m_atoms(small, 1)
                    if top_atom:
                        acc["top_atom_persistence"].append(
                            1.0 if top_atom[0] in ids_l else 0.0
                        )
                        top_atom_l = top_m_atoms(large, 1)
                        acc["top_atom_still_top"].append(
                            1.0 if top_atom_l and top_atom_l[0] == top_atom[0] else 0.0
                        )
                if not acc["jaccard"]:
                    continue
                row = {
                    "layer": layer,
                    "format": fmt,
                    "k_small": k_small,
                    "k_large": k_large,
                    "n": len(acc["jaccard"]),
                }
                for name in (
                    "containment",
                    "jaccard",
                    "weighted_jaccard",
                    "cosine",
                    "shared_atom_rank_correlation",
                    "output_token_persistence",
                    "top_atom_persistence",
                    "top_atom_still_top",
                ):
                    values = acc[name]
                    row[f"mean_{name}"] = (
                        statistics.fmean(values) if values else None
                    )
                rows.append(row)
    return rows


# ----------------------------------------------------------- transitions


def _slug_category_index(art: RunArtifacts) -> dict[str, str]:
    return {
        m["slug"]: m["category"] for m in art.metadata.get("capture_meta", [])
    }


def transition_summary(art: RunArtifacts) -> list[dict]:
    """Candidate-ignition signal distributions per (k, transition) under
    format/position/category cuts. Signals stay separate; no composite is
    computed (candidate ignition diagnostics — NOT validated ignition)."""
    categories = _slug_category_index(art)
    rows: list[dict] = []
    for k in art.k_values:
        records = art.ignition[k]
        by_transition: dict[tuple[int, int], list[dict]] = defaultdict(list)
        for record in records:
            by_transition[(record["layer_from"], record["layer_to"])].append(record)
        for transition in sorted(by_transition):
            group = by_transition[transition]
            cuts: list[tuple[str, str, list[dict]]] = [("all", "all", group)]
            for fmt in sorted({g["format"] for g in group}):
                cuts.append(
                    (fmt, "all", [g for g in group if g["format"] == fmt])
                )
            for pos in sorted({g["position"] for g in group}):
                cuts.append(
                    ("all", str(pos), [g for g in group if g["position"] == pos])
                )
            for cat in sorted(
                {categories.get(g.get("prompt_slug"), "unknown") for g in group}
            ):
                cuts.append(
                    (
                        "all",
                        f"category:{cat}",
                        [
                            g
                            for g in group
                            if categories.get(g.get("prompt_slug"), "unknown") == cat
                        ],
                    )
                )
            for fmt, cut, subset in cuts:
                if not subset:
                    continue
                signals = [g["signals"] for g in subset]
                aligned = [
                    s["output_alignment_to"]
                    for s in signals
                    if s.get("output_alignment_to") is not None
                ]
                rows.append(
                    {
                        "k": k,
                        "layer_from": transition[0],
                        "layer_to": transition[1],
                        "format": fmt,
                        "cut": cut,
                        "n": len(subset),
                        "mean_delta_explained_fraction": statistics.fmean(
                            s["delta_explained_fraction"] for s in signals
                        ),
                        "median_delta_explained_fraction": statistics.median(
                            s["delta_explained_fraction"] for s in signals
                        ),
                        "mean_active_set_jaccard": statistics.fmean(
                            s["active_set_jaccard"] for s in signals
                        ),
                        "median_active_set_jaccard": statistics.median(
                            s["active_set_jaccard"] for s in signals
                        ),
                        "mean_weighted_similarity": statistics.fmean(
                            s["weighted_similarity"] for s in signals
                        ),
                        "mean_delta_herfindahl": statistics.fmean(
                            s["delta_herfindahl"] for s in signals
                        ),
                        "mean_top1_share_to": statistics.fmean(
                            s["top1_share_to"] for s in signals
                        ),
                        "mean_persistence_length": statistics.fmean(
                            s["persistence_length_from_here"] for s in signals
                        ),
                        "output_alignment_rate_to": (
                            statistics.fmean(
                                1.0 if a["in_active_set"] else 0.0 for a in aligned
                            )
                            if aligned
                            else None
                        ),
                        "mean_output_coefficient_share_to": (
                            statistics.fmean(a["coefficient_share"] for a in aligned)
                            if aligned
                            else None
                        ),
                    }
                )
    return rows


def stabilization_robustness(art: RunArtifacts) -> list[dict]:
    """Per-trajectory transition stability recomputed from the cone records,
    with two extra variants the trajectory artifacts cannot provide: a
    frequency-adjusted weighted similarity (inverse-selection-frequency
    weights per k) and a similarity with the record's own model top-1 output
    token removed. Used to test whether the apparent final-transition
    (35->38) stabilization survives frequency adjustment and is not driven
    purely by shared output-token inclusion.

    'Stabilization' here means sparse-coordinate stability across a
    fitted-layer transition — explicitly not validated ignition.
    """
    categories = _slug_category_index(art)
    rows: list[dict] = []
    last_from = art.layers[-2] if len(art.layers) >= 2 else None
    for k in art.k_values:
        records = [r for layer in art.layers for r in art.cones[(layer, k)]]
        freqs = atom_selection_frequencies(records, strata=())
        weights = inverse_frequency_weights(freqs["overall"], len(records))
        by_key: dict[tuple, dict[int, dict]] = defaultdict(dict)
        for record in records:
            by_key[
                (record["prompt_hash"], record["format"], record["position"])
            ][record["layer"]] = record
        for key in sorted(by_key):
            by_layer = by_key[key]
            for layer_from, layer_to in zip(art.layers, art.layers[1:], strict=False):
                prev = by_layer.get(layer_from)
                curr = by_layer.get(layer_to)
                if prev is None or curr is None:
                    continue
                map_prev, map_curr = coefficient_map(prev), coefficient_map(curr)
                adj_prev = {t: c * weights.get(t, 1.0) for t, c in map_prev.items()}
                adj_curr = {t: c * weights.get(t, 1.0) for t, c in map_curr.items()}
                top1 = prev.get("run_provenance", {}).get("model_top1_id")
                no_out_prev = {t: c for t, c in map_prev.items() if t != top1}
                no_out_curr = {t: c for t, c in map_curr.items() if t != top1}
                rows.append(
                    {
                        "k": k,
                        "layer_from": layer_from,
                        "layer_to": layer_to,
                        "is_final_transition": layer_from == last_from,
                        "format": prev["format"],
                        "position": prev["position"],
                        "category": categories.get(prev.get("prompt_slug"), "unknown"),
                        "prompt_slug": prev.get("prompt_slug"),
                        "jaccard": jaccard_similarity(map_prev, map_curr),
                        "weighted_similarity": weighted_jaccard(map_prev, map_curr),
                        "cosine_similarity": sparse_cosine(map_prev, map_curr),
                        "frequency_adjusted_similarity": weighted_jaccard(
                            adj_prev, adj_curr
                        ),
                        "similarity_without_output_token": weighted_jaccard(
                            no_out_prev, no_out_curr
                        ),
                        "delta_explained_fraction": (
                            curr["reconstruction"]["explained_fraction"]
                            - prev["reconstruction"]["explained_fraction"]
                        ),
                    }
                )
    rows.sort(
        key=lambda r: (
            r["k"],
            r["layer_from"],
            r["format"],
            r["position"],
            r["prompt_slug"] or "",
        )
    )
    return rows


# ----------------------------------------------------------------- eval


def _eval_examples(art: RunArtifacts) -> list[dict]:
    return art.evaluation["results"]["examples"]


def eval_control_summary(art: RunArtifacts) -> list[dict]:
    """Recompute aggregate rank statistics for every lens variant from the
    per-example eval records, cut by layer x format (plus 'all'), category,
    and recorded position. Mirrors jlens.evaluation.aggregate_ranks
    definitions (rank 0 = argmax; hit@k = rank < k; MRR = mean 1/(rank+1))."""
    examples = _eval_examples(art)
    rows: list[dict] = []
    layer_keys = sorted({int(l) for e in examples for l in e["layers"]})
    for layer in layer_keys:
        cuts: list[tuple[str, str, str]] = [("all", "all", "all")]
        cuts += [
            (fmt, "all", "all") for fmt in sorted({e["format"] for e in examples})
        ]
        cuts += [
            ("all", cat, "all") for cat in sorted({e["category"] for e in examples})
        ]
        positions = sorted({p for e in examples for p in e["positions"]})
        cuts += [("all", "all", str(pos)) for pos in positions]
        cuts += [
            (fmt, "all", str(pos))
            for fmt in sorted({e["format"] for e in examples})
            for pos in positions
        ]
        for fmt, cat, pos in cuts:
            for variant in _EVAL_VARIANTS:
                ranks: list[int] = []
                overlaps: list[float] = []
                for example in examples:
                    if fmt != "all" and example["format"] != fmt:
                        continue
                    if cat != "all" and example["category"] != cat:
                        continue
                    data = example["layers"].get(str(layer))
                    if data is None or variant not in data:
                        continue
                    variant_data = data[variant]
                    for position, rank in zip(
                        example["positions"],
                        variant_data["rank_of_model_top1"],
                        strict=True,
                    ):
                        if pos != "all" and str(position) != pos:
                            continue
                        ranks.append(rank)
                    if pos == "all":
                        overlaps.append(variant_data["topk_overlap_with_model"])
                if not ranks:
                    continue
                rows.append(
                    {
                        "layer": layer,
                        "format": fmt,
                        "category": cat,
                        "position": pos,
                        "variant": variant,
                        "n_ranks": len(ranks),
                        "median_rank": statistics.median(ranks),
                        "mean_reciprocal_rank": statistics.fmean(
                            1.0 / (r + 1.0) for r in ranks
                        ),
                        "hit_rate@1": statistics.fmean(r < 1 for r in ranks),
                        "hit_rate@5": statistics.fmean(r < 5 for r in ranks),
                        "hit_rate@10": statistics.fmean(r < 10 for r in ranks),
                        "mean_topk_overlap": (
                            statistics.fmean(overlaps) if overlaps else None
                        ),
                    }
                )
    return rows


def eval_control_collisions(art: RunArtifacts) -> dict:
    """Where layer-mapped controls use the same source Jacobian, their
    per-example ranks must coincide; report observed identity rates against
    what the recorded mapping implies."""
    provenance = art.evaluation["results"]["provenance"]

    def mapping_of(variant: str) -> dict[int, int]:
        return {
            m["applied_at_layer"]: m["jacobian_fitted_at_layer"]
            for m in provenance.get(variant, {}).get("mapping", [])
        }

    mapped = {
        v: mapping_of(v)
        for v in ("adjacent_layer", "distant_layer", "shuffled_layer")
    }
    examples = _eval_examples(art)
    layer_keys = sorted({int(l) for e in examples for l in e["layers"]})
    pairs = [
        ("distant_layer", "shuffled_layer"),
        ("adjacent_layer", "shuffled_layer"),
        ("adjacent_layer", "distant_layer"),
    ]
    report: dict[str, list] = {"schema": "jlens.analysis.control_collisions.v1", "pairs": []}
    for va, vb in pairs:
        for layer in layer_keys:
            expected_identical = (
                mapped[va].get(layer) is not None
                and mapped[va].get(layer) == mapped[vb].get(layer)
            )
            identical = 0
            total = 0
            for example in examples:
                data = example["layers"].get(str(layer))
                if data is None or va not in data or vb not in data:
                    continue
                total += 1
                if (
                    data[va]["rank_of_model_top1"]
                    == data[vb]["rank_of_model_top1"]
                ):
                    identical += 1
            report["pairs"].append(
                {
                    "variant_a": va,
                    "variant_b": vb,
                    "layer": layer,
                    "source_layer_a": mapped[va].get(layer),
                    "source_layer_b": mapped[vb].get(layer),
                    "expected_identical": expected_identical,
                    "identical_examples": identical,
                    "total_examples": total,
                    "consistent": (
                        (identical == total) if expected_identical else identical < total
                    ),
                }
            )
    # Fitted-J reuse: a layer-mapped control that borrows layer m's Jacobian
    # is the fitted lens of layer m applied off-layer; flag mappings whose
    # source is a decomposition neighbor (interpretation aid, not an error).
    report["adjacent_source_notes"] = [
        {
            "variant": variant,
            "applied_at_layer": applied,
            "jacobian_fitted_at_layer": source,
        }
        for variant, mapping in sorted(mapped.items())
        for applied, source in sorted(mapping.items())
        if abs(applied - source) <= 3
    ]
    return report


# ----------------------------------------------------- atoms and similarity


def atom_frequency_table(art: RunArtifacts) -> list[dict]:
    """Per-atom selection counts: overall and by layer, k, format, and
    position, plus the number of distinct prompts the atom appears for."""
    records = art.all_cone_records()
    overall = atom_selection_frequencies(records, strata=())
    by_layer = atom_selection_frequencies(records, strata=("layer",))
    by_k = atom_selection_frequencies(records, strata=("requested_k",))
    by_format = atom_selection_frequencies(records, strata=("format",))
    by_position = atom_selection_frequencies(records, strata=("position",))
    prompts: dict[int, set] = defaultdict(set)
    for record in records:
        for token_id in record["effective_token_ids"]:
            prompts[int(token_id)].add(record["prompt_hash"])
    counts = overall["overall"]
    rows = []
    for token_id in sorted(counts, key=lambda t: (-counts[t], t)):
        row = {
            "token_id": token_id,
            "label": overall["labels"].get(token_id, ""),
            "total_selections": counts[token_id],
            "n_distinct_prompts": len(prompts[token_id]),
        }
        for layer in art.layers:
            row[f"layer_{layer}"] = by_layer["by_stratum"].get((layer,), {}).get(
                token_id, 0
            )
        for k in art.k_values:
            row[f"k_{k}"] = by_k["by_stratum"].get((k,), {}).get(token_id, 0)
        for fmt in ("plain", "chat"):
            row[f"format_{fmt}"] = by_format["by_stratum"].get((fmt,), {}).get(
                token_id, 0
            )
        for pos in (-2, -1):
            row[f"position_{pos}"] = by_position["by_stratum"].get((pos,), {}).get(
                token_id, 0
            )
        rows.append(row)
    return rows


def similarity_report(
    art: RunArtifacts,
    *,
    metric: str = "weighted_jaccard",
    thresholds: Sequence[float] = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    detail_threshold: float = 0.5,
    max_members: int = 20,
) -> dict:
    """Similarity-group analysis per k: threshold sensitivity for the raw
    and frequency-adjusted metric, and non-singleton group membership at
    ``detail_threshold``. Groups are similarity groups, not concept
    clusters."""
    payload: dict = {
        "schema": "jlens.analysis.similarity_groups.v1",
        "metric": metric,
        "strata": ["layer", "requested_k", "format", "position"],
        "detail_threshold": detail_threshold,
        "note": (
            "Connected components under pairwise similarity >= threshold, "
            "within (layer, k, format, position) strata. Similarity groups, "
            "not validated concept clusters."
        ),
        "by_k": {},
    }
    for k in art.k_values:
        records = [
            r
            for layer in art.layers
            for r in art.cones[(layer, k)]
        ]
        freqs = atom_selection_frequencies(records, strata=())
        weights = inverse_frequency_weights(freqs["overall"], len(records))
        sensitivity_raw = threshold_sensitivity(
            records, metric=metric, thresholds=thresholds
        )
        sensitivity_adjusted = threshold_sensitivity(
            records, metric=metric, thresholds=thresholds, atom_weights=weights
        )
        groups = similarity_groups(
            records, metric=metric, threshold=detail_threshold
        )
        non_singleton = [
            {
                "stratum": g["stratum"],
                "size": g["size"],
                "members": [
                    {
                        "prompt_slug": records[i]["prompt_slug"],
                        "format": records[i]["format"],
                        "position": records[i]["position"],
                        "top_atoms": [
                            {"token_id": t, "label": lbl}
                            for t, lbl in zip(
                                top_m_atoms(records[i], 3),
                                [
                                    records[i]["effective_labels"][
                                        records[i]["effective_token_ids"].index(t)
                                    ]
                                    for t in top_m_atoms(records[i], 3)
                                ],
                                strict=True,
                            )
                        ],
                    }
                    for i in g["record_indices"][:max_members]
                ],
            }
            for g in groups
            if g["size"] > 1
        ]
        payload["by_k"][str(k)] = {
            "n_records": len(records),
            "threshold_sensitivity_raw": sensitivity_raw,
            "threshold_sensitivity_frequency_adjusted": sensitivity_adjusted,
            "non_singleton_groups_at_detail_threshold": non_singleton,
            "output_token_recurrence_top": output_token_recurrence(records)[:20],
        }
    return payload


# ------------------------------------------------------------------- writing


def _format_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.8g}"
    return str(value)


def write_csv(rows: Sequence[Mapping], path: str) -> None:
    """Write dict rows as UTF-8 CSV with deterministic column order (first
    row's key order) and fixed float formatting."""
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write("")
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _format_value(row.get(k)) for k in fieldnames})


def write_json(payload, path: str) -> None:
    """Deterministic pretty JSON (sorted keys, UTF-8, trailing newline)."""
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=1, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
