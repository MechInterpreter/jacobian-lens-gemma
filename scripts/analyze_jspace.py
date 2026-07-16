# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Deterministic offline analysis of a completed jspace_pursuit run.

    python scripts/analyze_jspace.py \
        --run-dir runs/jspace_20260716T170808536780_e4118850fb70 \
        [--out-dir reports/<run_id>] [--lens path/to/lens.pt]

Reads the run's artifacts (never modifying them), and writes derived
summaries into a separate report directory:

- integrity_summary.json          (Phase: artifact inventory + checks)
- metrics_by_layer_k_format.csv   (per layer/k/format/category/position)
- k_marginal_gains.csv            (paired k10->k16->k25 gains)
- cross_k_stability.csv           (active-set stability across k)
- transition_metrics.csv          (per-transition candidate signals)
- candidate_ignition_summary.csv  (per-trajectory transition stability,
                                   incl. frequency-adjusted and
                                   output-token-excluded variants)
- atom_frequencies.csv            (per-atom selection counts)
- atom_enrichment.csv             (observed-vs-expected by layer/format)
- similarity_groups.json          (threshold-sensitivity + groups)
- evaluation_control_summary.csv  (all lens variants, recomputed)
- evaluation_control_collisions.json
- analysis_summary.json           (headline numbers; includes lens-matrix
                                   statistics when --lens is given)

CPU-only; no model download, no GPU, no mutation of the run directory.
Running twice produces byte-identical outputs.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jlens.jspace_analysis import (
    RunArtifacts,
    atom_frequency_table,
    check_integrity,
    cross_k_stability,
    eval_control_collisions,
    eval_control_summary,
    k_marginal_gains,
    load_run,
    metrics_table,
    record_metrics,
    similarity_report,
    stabilization_robustness,
    transition_summary,
    write_csv,
    write_json,
)
from jlens.similarity import atom_enrichment, atom_selection_frequencies


def lens_matrix_statistics(lens_path: str) -> dict:
    """CPU-only per-layer statistics of the fitted lens matrices (Frobenius
    norm, spectral norm, singular-value summaries, effective-rank proxies,
    row/column norm quartiles). Loads only the lens artifact — never the
    model."""
    import torch

    checkpoint = torch.load(lens_path, map_location="cpu", weights_only=True)
    stats: dict[str, dict] = {}
    for layer in checkpoint["source_layers"]:
        J = checkpoint["J"][layer].float()
        singular = torch.linalg.svdvals(J)
        shares = singular / singular.sum()
        entropy_rank = torch.exp(
            -(shares * (shares + 1e-30).log()).sum()
        ).item()
        participation_ratio = (singular.sum() ** 2 / (singular**2).sum()).item()
        rows = J.norm(dim=1)
        cols = J.norm(dim=0)

        def quartiles(t: "torch.Tensor") -> list[float]:
            q = torch.quantile(t, torch.tensor([0.25, 0.5, 0.75]))
            return [float(x) for x in q]

        stats[str(layer)] = {
            "frobenius_norm": float(J.norm()),
            "spectral_norm": float(singular[0]),
            "singular_values": {
                "top1": float(singular[0]),
                "top10": float(singular[9]),
                "top100": float(singular[99]),
                "median": float(singular.median()),
            },
            "effective_rank_entropy": entropy_rank,
            "effective_rank_participation_ratio": participation_ratio,
            "all_finite": bool(torch.isfinite(J).all()),
            "row_norm_quartiles": quartiles(rows),
            "row_norm_max": float(rows.max()),
            "col_norm_quartiles": quartiles(cols),
            "diagonal_mass": float(J.diagonal().abs().sum() / J.abs().sum()),
        }
    return {
        "lens_path": os.path.basename(lens_path),
        "source_layers": list(checkpoint["source_layers"]),
        "n_prompts": checkpoint["n_prompts"],
        "per_layer": stats,
        "note": (
            "Statistics of the fitted J_l matrices only. The J-space atoms "
            "(rows of W_U @ J_l) additionally require Gemma's unembedding "
            "matrix, which is not available without loading the model; "
            "atom-level norms are therefore not recomputable offline."
        ),
    }


def headline_summary(art: RunArtifacts, integrity: dict) -> dict:
    """Compact machine-readable headline numbers for the run report."""
    summary: dict = {
        "schema": "jlens.analysis.summary.v1",
        "run_id": art.metadata.get("run_id"),
        "model_revision": art.metadata.get("load_info", {}).get("model_revision"),
        "lens_fingerprint": art.metadata.get("lens_verification", {}).get(
            "file_sha256"
        ),
        "layers": art.layers,
        "k_values": art.k_values,
        "integrity_clean": integrity["clean"],
        "n_integrity_issues": len(integrity["issues"]),
        "explained_fraction_mean": {},
        "output_token_inclusion_rate": {},
        "final_transition_weighted_similarity_mean": {},
        "exact_signature_repeats": {},
    }
    for (layer, k), records in sorted(art.cones.items()):
        metrics = [record_metrics(r) for r in records]
        summary["explained_fraction_mean"][f"layer{layer}_k{k}"] = statistics.fmean(
            m["explained_fraction"] for m in metrics
        )
        for fmt in ("plain", "chat"):
            subset = [
                m
                for m, r in zip(metrics, records, strict=True)
                if r["format"] == fmt
            ]
            if subset:
                summary["output_token_inclusion_rate"][
                    f"layer{layer}_k{k}_{fmt}"
                ] = statistics.fmean(m["output_token_included"] for m in subset)
    last_from = art.layers[-2] if len(art.layers) >= 2 else None
    for k in art.k_values:
        final = [
            t["weighted_similarity"]
            for t in art.trajectories[k]
            if t["layer_from"] == last_from
        ]
        if final:
            summary["final_transition_weighted_similarity_mean"][
                f"k{k}"
            ] = statistics.fmean(final)
        summary["exact_signature_repeats"][f"k{k}"] = sum(
            1 for row in art.signatures[k] if row["count"] > 1
        )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="completed jspace run dir")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="report output dir (default: reports/<run_id>)",
    )
    parser.add_argument(
        "--lens",
        default=None,
        help="optional path to the frozen pilot lens.pt for J-matrix "
        "statistics (CPU-only; no model load)",
    )
    args = parser.parse_args(argv)

    art = load_run(args.run_dir)
    run_id = art.metadata.get("run_id", os.path.basename(os.path.normpath(args.run_dir)))
    out_dir = args.out_dir or os.path.join("reports", run_id)
    os.makedirs(out_dir, exist_ok=True)

    integrity = check_integrity(art)
    write_json(integrity, os.path.join(out_dir, "integrity_summary.json"))

    write_csv(metrics_table(art), os.path.join(out_dir, "metrics_by_layer_k_format.csv"))
    write_csv(k_marginal_gains(art), os.path.join(out_dir, "k_marginal_gains.csv"))
    write_csv(cross_k_stability(art), os.path.join(out_dir, "cross_k_stability.csv"))
    write_csv(transition_summary(art), os.path.join(out_dir, "transition_metrics.csv"))
    write_csv(
        stabilization_robustness(art),
        os.path.join(out_dir, "candidate_ignition_summary.csv"),
    )
    write_csv(atom_frequency_table(art), os.path.join(out_dir, "atom_frequencies.csv"))

    records = art.all_cone_records()
    frequencies_by_layer = atom_selection_frequencies(records, strata=("layer",))
    frequencies_by_format = atom_selection_frequencies(records, strata=("format",))
    enrichment_rows = []
    for frequencies in (frequencies_by_layer, frequencies_by_format):
        for row in atom_enrichment(frequencies):
            flattened = {
                "stratum": ";".join(
                    f"{key}={value}" for key, value in row["stratum"].items()
                ),
                **{key: value for key, value in row.items() if key != "stratum"},
            }
            enrichment_rows.append(flattened)
    # Keep the table light: only atoms selected at least 3 times in-stratum.
    enrichment_rows = [r for r in enrichment_rows if r["observed"] >= 3]
    write_csv(enrichment_rows, os.path.join(out_dir, "atom_enrichment.csv"))

    write_json(
        similarity_report(art), os.path.join(out_dir, "similarity_groups.json")
    )
    write_csv(
        eval_control_summary(art),
        os.path.join(out_dir, "evaluation_control_summary.csv"),
    )
    write_json(
        eval_control_collisions(art),
        os.path.join(out_dir, "evaluation_control_collisions.json"),
    )

    summary = headline_summary(art, integrity)
    if args.lens:
        summary["lens_matrix_statistics"] = lens_matrix_statistics(args.lens)
    write_json(summary, os.path.join(out_dir, "analysis_summary.json"))

    print(f"wrote analysis outputs to {out_dir}")
    print(f"integrity: {'clean' if integrity['clean'] else 'ISSUES FOUND'} "
          f"({len(integrity['issues'])} issues, {len(integrity['notes'])} notes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
