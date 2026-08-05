# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Markdown and JSON reporting for a calibration run.

The report is written so that a reader who did not run it can tell three things
apart: what was measured, what was decided by a predeclared rule, and what is
still unknown. In particular it never says a lens is validated because fitting
completed, and it prints the confirmation state explicitly so that "we have not
run confirmation yet" cannot be mistaken for "nothing passed".
"""

from __future__ import annotations

from collections.abc import Mapping

from jlens.mmpilot.store import payload_checksum

__all__ = ["calibration_report_markdown", "calibration_report_payload"]


def calibration_report_payload(
    *,
    fingerprint_digest: str,
    protocol_version: str,
    capture_plan: Mapping,
    budget: Mapping,
    corpus_manifest: Mapping,
    leakage_audit: Mapping,
    nesting_audit: Mapping,
    diversity: Mapping,
    comparison: Mapping | None,
    plateau: Mapping | None,
    selection: Mapping | None,
    confirmation: Mapping | None,
    publication: Mapping | None,
    vault_status: Mapping,
    resume_status: Mapping,
    baseline: Mapping,
    mode: str,
) -> dict:
    """The machine-readable run record."""
    payload = {
        "schema": "jlens.calibration.report.v1",
        "mode": mode,
        "protocol_version": protocol_version,
        "fingerprint_digest": fingerprint_digest,
        "capture_plan": dict(capture_plan),
        "budget": dict(budget),
        "corpus": dict(corpus_manifest),
        "leakage_audit": dict(leakage_audit),
        "scale_nesting_audit": dict(nesting_audit),
        "target_diversity": dict(diversity),
        "scale_comparison": dict(comparison) if comparison else None,
        "plateau": dict(plateau) if plateau else None,
        "scale_selection": dict(selection) if selection else None,
        "confirmation": dict(confirmation) if confirmation else None,
        "publication": dict(publication) if publication else None,
        "confirmation_vault": dict(vault_status),
        "resume": dict(resume_status),
        "baseline_manifest": dict(baseline),
        "objective": "not_applicable_estimator_is_a_sample_mean",
        "mock_proves_pipeline_only": mode != "real",
    }
    payload["report_checksum"] = payload_checksum(payload)
    return payload


def _scale_table(comparison: Mapping) -> list[str]:
    lines = [
        "| layer | depth | scale | passed | MRR | median midrank | opt | pess | "
        "top-10 incl | tied@max | vs wrong-layer |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in comparison["rows"]:
        lines.append(
            f"| {row['layer']} | {round(100 * row['layer'] / 42)} | {row['scale']:,} | "
            f"{'PASS' if row['passed'] else 'fail'} | "
            f"{row['mean_reciprocal_rank']:.4f} | {row['median_midrank']:.2f} | "
            f"{row['median_optimistic_rank']:.1f} | "
            f"{row['median_pessimistic_rank']:.1f} | "
            f"{row['top10_inclusion']:.3f} | {row['tied_at_max_rate']:.3f} | "
            f"{row['margin_over_wrong_layer']:+.4f} |"
        )
    return lines


def _delta_table(comparison: Mapping) -> list[str]:
    lines = [
        "| layer | step | ΔMRR | midrank drop | Δtied@max | pass before → after |",
        "|---|---|---|---|---|---|",
    ]
    for row in comparison["deltas"]:
        lines.append(
            f"| {row['layer']} | {row['from_scale']:,}→{row['to_scale']:,} | "
            f"{row['delta_mrr']:+.4f} | "
            f"{row['median_midrank_relative_drop']:+.1%} | "
            f"{row['delta_tied_at_max_rate']:+.4f} | "
            f"{'PASS' if row['passed_before'] else 'fail'} → "
            f"{'PASS' if row['passed_after'] else 'fail'} |"
        )
    return lines


def calibration_report_markdown(
    payload: Mapping,
    *,
    gate_text: str,
    plateau_text: str,
) -> str:
    """The human-readable report."""
    mode = payload["mode"]
    lines: list[str] = [
        "# Research-grade multilayer J-lens calibration — run report",
        "",
        f"- **Mode:** `{mode}`",
        f"- **Protocol:** `{payload['protocol_version']}`",
        f"- **Fingerprint:** `{payload['fingerprint_digest']}`",
        f"- **Report checksum:** `{payload['report_checksum']}`",
        "",
    ]

    if mode != "real":
        lines += [
            "> **MOCK RUN.** This proves pipeline behaviour only. It loaded no",
            "> Gemma, contacted no Hub, mounted no Drive, and downloaded no corpus.",
            "> Nothing here is evidence about Gemma 4, about any layer, or about",
            "> whether a research-grade lens exists.",
            "",
        ]

    plan = payload["capture_plan"]
    lines += [
        "## Configuration",
        "",
        f"- Layers `{plan['layers']}` (normalized depth `{plan['normalized_depth']}`)",
        f"- Target layer `{plan['target_layer']}`, site `block_output`, "
        f"`skip_first={plan['skip_first']}`, `max_seq_len={plan['max_seq_len']}`",
        f"- Per prompt: 1 forward + {plan['backward_passes_per_prompt']} backward "
        f"passes; backward span {plan['backward_span']} blocks",
        f"- All {len(plan['layers'])} layers captured in one pass: "
        f"`{plan['all_layers_captured_in_one_pass']}`",
        f"- Per-layer parameters: {plan['per_layer_parameters']}",
        "",
        "**There is no fitting objective and no optimizer.** `J_l` is a population",
        "mean estimated by a running average; increasing the prompt count reduces",
        "estimator variance and does nothing else.",
        "",
        "## Corpus and splits",
        "",
        f"- Corpus `{payload['corpus'].get('corpus_id')}` "
        f"(revision `{payload['corpus'].get('revision')}`, "
        f"status `{payload['corpus'].get('revision_status')}`)",
        f"- Modality `{payload['corpus'].get('modality')}`; SpokenCOCO used: "
        f"`{payload['corpus'].get('spokencoco_used')}`",
    ]
    splits = payload["corpus"].get("splits", {})
    for name, size in (splits.get("sizes") or {}).items():
        lines.append(
            f"- `{name}`: {size:,} records, checksum "
            f"`{(splits.get('checksums') or {}).get(name)}`"
        )
    leakage = payload["leakage_audit"]
    nesting = payload["scale_nesting_audit"]
    lines += [
        f"- Leakage audit: **{'clean' if leakage.get('ok') else 'FAILED'}** — "
        f"{leakage.get('n_exact_hits', 0)} exact and "
        f"{leakage.get('n_near_hits', 0)} near duplicates across partitions, "
        f"{leakage.get('candidate_pairs_compared', 0)} candidate pairs compared",
        f"- Scale nesting: **{'exact' if nesting.get('nested') else 'BROKEN'}**",
        "",
        "## Target-token diversity",
        "",
        f"- {payload['target_diversity'].get('n_distinct_target_tokens')} distinct "
        f"targets over {payload['target_diversity'].get('n_prompts')} prompts "
        f"(floor {payload['target_diversity'].get('min_distinct_target_tokens')})",
        f"- Largest single-target share "
        f"{payload['target_diversity'].get('max_target_token_share', 0):.1%} "
        f"(ceiling "
        f"{payload['target_diversity'].get('max_target_token_share_allowed', 0):.0%})",
        f"- Passed: **{payload['target_diversity'].get('passed')}**",
        "",
        "## Budget",
        "",
        "Scale rows are **cumulative, not additive** — the points are nested.",
        "",
        "| scale | hours (central) | range | ~12h sessions |",
        "|---|---|---|---|",
    ]
    for row in payload["budget"].get("per_scale", []):
        lines.append(
            f"| {row['scale']:,} | {row['fitting_hours_central']:.1f} | "
            f"{row['fitting_hours_low']:.1f}–{row['fitting_hours_high']:.1f} | "
            f"{row['colab_sessions_central']}–{row['colab_sessions_high']} |"
        )
    lines += [
        "",
        f"Drive total ≈ {payload['budget'].get('drive_gib_total', 0):.2f} GiB.",
        "",
        "## Predeclared gate",
        "",
        "```",
        gate_text.strip(),
        "```",
        "",
    ]

    comparison = payload.get("scale_comparison")
    if comparison:
        lines += ["## Scale comparison", "", *_scale_table(comparison), ""]
        if comparison.get("deltas"):
            lines += ["### Change between scale points", "", *_delta_table(comparison), ""]
        lines += [
            "Eligible layers by scale: "
            + ", ".join(
                f"`{scale}` → {layers}"
                for scale, layers in comparison["eligible_by_scale"].items()
            ),
            "",
        ]
    else:
        lines += ["## Scale comparison", "", "_Not run._", ""]

    lines += ["## Plateau rule", "", "```", plateau_text.strip(), "```", ""]
    plateau = payload.get("plateau")
    if plateau:
        lines += [
            f"**Verdict: `{plateau['verdict']}`** "
            f"(extension justified: `{plateau['extension_justified']}`)",
            "",
        ]
        for clause in plateau.get("clauses", []):
            lines.append(
                f"- `{clause['clause']}`: "
                f"**{'pass' if clause['passed'] else 'fail'}** — {clause['detail']}"
            )
        lines.append("")
    else:
        lines += ["_Not evaluated._", ""]

    selection = payload.get("scale_selection")
    lines += ["## Scale selection", ""]
    if selection:
        lines += [
            f"- Selected scale: **{selection['selected_scale']:,}**",
            f"- Rule: {selection['selection_rule']}",
            f"- Reason: {selection['reason']}",
            f"- Confirmation consulted: `{not selection['confirmation_not_consulted']}`",
            "",
        ]
    else:
        lines += ["_Not selected._", ""]

    vault = payload["confirmation_vault"]
    lines += [
        "## Final confirmation",
        "",
        f"- Vault locked: `{vault.get('locked')}`; opened: `{vault.get('opened')}`",
        f"- Records held: {vault.get('n_records')}",
    ]
    confirmation = payload.get("confirmation")
    if confirmation:
        lines += ["", "| layer | passed | failed clauses |", "|---|---|---|"]
        for layer, verdict in sorted(confirmation.items(), key=lambda kv: int(kv[0])):
            lines.append(
                f"| {layer} | {'PASS' if verdict['passed'] else 'fail'} | "
                f"{', '.join(verdict['failed_checks']) or '—'} |"
            )
        lines.append("")
    else:
        lines += [
            "",
            "_Confirmation was not run. This is **not** the same as 'no layer",
            "passed' — no layer has been offered the confirmation set._",
            "",
        ]

    publication = payload.get("publication")
    lines += ["## Publication", ""]
    if publication:
        lines += [
            f"- Published: {publication['n_published']} "
            f"layer(s) {publication['published_layers']}",
            f"- Failed: {publication['n_failed']} "
            f"layer(s) {publication['failed_layers']}",
            "- Failed layers keep their diagnostics and are never marked validated.",
            "",
        ]
        for layer, checksum in publication.get("published_checksums", {}).items():
            lines.append(f"  - L{layer}: `{checksum}`")
        lines.append("")
    else:
        lines += ["_Nothing published._", ""]

    lines += [
        "## Resume state",
        "",
        f"- Status: `{payload['resume'].get('status')}`",
        f"- Checkpoint present: `{payload['resume'].get('checkpoint_present')}`",
        f"- Completed units: `{payload['resume'].get('completed_units')}`",
        f"- Invalid units dropped: {len(payload['resume'].get('invalid_units', []))}",
        "",
        "## What this run does not establish",
        "",
        "- A passing gate is a statement about **text-only native readout**, not",
        "  about causal transfer, not about J-space, and not about any modality",
        "  other than text.",
        "- Earlier-layer causal transfer is **not** claimed here under any outcome.",
        "  It would require an independently confirmed lens *and* the multimodal",
        "  causal experiment actually being run.",
        "- MOCK success proves pipeline behaviour only.",
        "",
    ]
    return "\n".join(lines)
