# SPDX-License-Identifier: Apache-2.0
"""No-refit evidence-sealing diagnostic for the country workspace study.

The source evidence is available while the final prompt state is encoded.  At
the declared bottleneck, later attention from that state is prevented from
returning to the earlier prompt prefix.  Newly generated tokens are likewise
prevented from reading that prefix, so they must use the encoded final state.

This is a development diagnostic, not the original paper intervention.  It
does not fit a lens and it never opens the untouched confirmation population.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any

import torch

from jlens.mmpilot.store import payload_checksum

EVIDENCE_SEAL_VERSION = "mmpilot.country_evidence_seal.v1"
SEALED_DEVELOPMENT_VERSION = "mmpilot.country_sealed_evidence_development.v1"
MATCHED_SCAFFOLD_VERSION = "mmpilot.country_matched_scaffold_development.v1"
MATCHED_CONFIRMATION_VERSION = "mmpilot.country_matched_scaffold_confirmation.v1"
SEALED_STATE_CONDITIONS = ("clean_sealed", "target_state", "unrelated_state")
SEALED_COORDINATE_CONDITIONS = ("exact", "zero", "random", "unrelated")


class CountryEvidenceSealRefused(RuntimeError):
    """The evidence seal cannot be applied or audited unambiguously."""


def evidence_positions(inputs, *, country_token_id: int | None = None) -> tuple[int, ...]:
    """Resolve the complete evidence span without guessing prompt positions."""

    prompt_len = int(inputs.prompt_len)
    span = getattr(inputs, "modality_token_range", None)
    if span is not None:
        start, end = map(int, span)
        if not 0 <= start < end <= prompt_len:
            raise CountryEvidenceSealRefused(
                f"invalid modality token range [{start}, {end}) for prompt {prompt_len}"
            )
        return tuple(range(start, end))

    if country_token_id is None:
        raise CountryEvidenceSealRefused(
            "text evidence sealing requires the frozen country token id"
        )
    input_ids = inputs.tensors.get("input_ids")
    if not torch.is_tensor(input_ids) or input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise CountryEvidenceSealRefused(
            "text evidence sealing requires rank-two, single-example input_ids"
        )
    matches = (
        input_ids[0, :prompt_len].detach().cpu() == int(country_token_id)
    ).nonzero(as_tuple=True)[0].tolist()
    if len(matches) != 1:
        raise CountryEvidenceSealRefused(
            f"country token {country_token_id} occurs {len(matches)} times; "
            "refusing an ambiguous evidence span"
        )
    return (int(matches[0]),)


@contextmanager
def seal_evidence_attention(
    blocks: Sequence[torch.nn.Module],
    *,
    evidence_token_positions: Sequence[int],
    prompt_len: int,
    bottleneck_layer: int,
):
    """Seal direct evidence access after encoding and during generation.

    Prompt tokens can read evidence below ``bottleneck_layer``.  From the
    bottleneck onward, the final prompt token cannot read it again.  Generated
    tokens are blocked from reading evidence at every layer.  The latter is
    necessary because generation recomputes the full sequence without a KV
    cache and could otherwise recover the source identity in early layers.
    """

    prompt_len = int(prompt_len)
    bottleneck_layer = int(bottleneck_layer)
    evidence = tuple(sorted({int(position) for position in evidence_token_positions}))
    if prompt_len < 1:
        raise CountryEvidenceSealRefused("the evidence seal requires a nonempty prompt")
    if not evidence or any(not 0 <= position < prompt_len for position in evidence):
        raise CountryEvidenceSealRefused(
            f"invalid evidence positions {list(evidence)} for prompt length {prompt_len}"
        )
    if any(position >= prompt_len - 1 for position in evidence):
        raise CountryEvidenceSealRefused(
            "the evidence span overlaps the final prompt bottleneck token"
        )
    if not 0 <= bottleneck_layer < len(blocks):
        raise CountryEvidenceSealRefused(
            f"bottleneck layer {bottleneck_layer} is outside {len(blocks)} blocks"
        )

    # Sealing only the raw media span would leave an indirect bypass through
    # later prompt tokens that had already attended to it.  The final prompt
    # token is therefore the sole permitted carrier across the bottleneck.
    sealed_prefix = tuple(range(prompt_len - 1))
    stats: dict[int, dict[str, Any]] = {}
    with ExitStack() as stack:
        for layer, block in enumerate(blocks):
            row = {
                "layer": layer,
                "bottleneck_layer": bottleneck_layer,
                "n_forward_passes": 0,
                "n_masked_query_key_pairs": 0,
                "all_masks_rank_four": True,
                "all_masks_finite_or_negative_infinity": True,
                "n_base_masks_materialized": 0,
                "prompt_endpoint_sealed": layer >= bottleneck_layer,
                "generated_tokens_sealed": True,
                "evidence_token_positions": list(evidence),
                "sealed_prefix_positions": list(sealed_prefix),
            }
            stats[layer] = row

            def make_hook(layer_index: int, record: dict):
                def hook(module, args, kwargs):
                    hidden = args[0] if args else kwargs.get("hidden_states")
                    mask = kwargs.get("attention_mask")
                    if not torch.is_tensor(hidden) or hidden.ndim != 3:
                        raise CountryEvidenceSealRefused(
                            f"layer {layer_index} hidden state is not [batch, seq, d_model]"
                        )
                    sequence_length = int(hidden.shape[1])
                    if mask is None:
                        attention = getattr(module, "self_attn", None)
                        implementation = getattr(
                            getattr(module, "config", None),
                            "_attn_implementation",
                            None,
                        )
                        if implementation != "sdpa":
                            raise CountryEvidenceSealRefused(
                                f"layer {layer_index} omitted its mask under unsupported "
                                f"attention implementation {implementation!r}"
                            )
                        positions = torch.arange(
                            sequence_length, device=hidden.device
                        )
                        query = positions[:, None]
                        key = positions[None, :]
                        allowed = torch.ones_like(query == key)
                        if bool(getattr(attention, "is_causal", True)):
                            allowed = key <= query
                        if bool(getattr(attention, "is_sliding", False)):
                            window = int(getattr(attention, "sliding_window", 0) or 0)
                            if window < 1:
                                raise CountryEvidenceSealRefused(
                                    f"layer {layer_index} declares sliding attention "
                                    "without a positive window"
                                )
                            allowed = allowed & (key > query - window)
                        mask = allowed[None, None].expand(
                            int(hidden.shape[0]), 1, -1, -1
                        ).clone()
                        record["n_base_masks_materialized"] += 1
                    if (
                        not torch.is_tensor(mask)
                        or mask.ndim != 4
                        or not (mask.is_floating_point() or mask.dtype == torch.bool)
                    ):
                        raise CountryEvidenceSealRefused(
                            f"layer {layer_index} attention mask must be rank-four "
                            "floating-point or boolean"
                        )
                    if sequence_length < prompt_len:
                        raise CountryEvidenceSealRefused(
                            f"layer {layer_index} sequence is shorter than the frozen prompt"
                        )
                    if int(mask.shape[-2]) != sequence_length or int(mask.shape[-1]) < sequence_length:
                        raise CountryEvidenceSealRefused(
                            f"layer {layer_index} mask shape {tuple(mask.shape)} does not "
                            f"cover sequence length {sequence_length}"
                        )

                    # Below the bottleneck only newly generated query tokens are
                    # sealed.  At and above it the final prompt token is sealed too.
                    query_start = (
                        prompt_len - 1 if layer_index >= bottleneck_layer else prompt_len
                    )
                    new_mask = mask.clone()
                    n_queries = max(0, sequence_length - query_start)
                    if n_queries:
                        blocked_value = (
                            False
                            if new_mask.dtype == torch.bool
                            else torch.finfo(new_mask.dtype).min
                        )
                        new_mask[
                            ..., query_start:sequence_length, list(sealed_prefix)
                        ] = blocked_value
                    record["n_forward_passes"] += 1
                    record["n_masked_query_key_pairs"] += (
                        n_queries * len(sealed_prefix)
                    )
                    finite_or_neg_inf = bool(
                        new_mask.dtype == torch.bool
                        or torch.isfinite(new_mask).all()
                        or torch.isneginf(new_mask)
                        .logical_or(torch.isfinite(new_mask))
                        .all()
                    )
                    record["all_masks_finite_or_negative_infinity"] = bool(
                        record["all_masks_finite_or_negative_infinity"]
                        and finite_or_neg_inf
                    )
                    return args, {**kwargs, "attention_mask": new_mask}

                return hook

            handle = block.register_forward_pre_hook(
                make_hook(layer, row), with_kwargs=True
            )
            stack.callback(handle.remove)
        yield stats


def _seal_summary(stats: Mapping[int, Mapping], expected_forward_passes: int) -> dict:
    by_layer = {str(layer): dict(stats[layer]) for layer in sorted(stats)}
    all_hooks = bool(by_layer) and all(
        int(row["n_forward_passes"]) == int(expected_forward_passes)
        for row in by_layer.values()
    )
    return {
        "version": EVIDENCE_SEAL_VERSION,
        "by_layer": by_layer,
        "expected_forward_passes": int(expected_forward_passes),
        "all_hooks_fired": all_hooks,
        "all_masks_rank_four": all(
            bool(row["all_masks_rank_four"]) for row in by_layer.values()
        ),
        "all_masks_finite_or_negative_infinity": all(
            bool(row["all_masks_finite_or_negative_infinity"])
            for row in by_layer.values()
        ),
        "generated_tokens_sealed_at_every_layer": all(
            bool(row["generated_tokens_sealed"]) for row in by_layer.values()
        ),
    }


def _with_seal_diagnostics(result: Mapping, stats: Mapping[int, Mapping]) -> dict:
    expected = int(result["n_forward_passes"])
    return {**dict(result), "evidence_seal_diagnostics": _seal_summary(stats, expected)}


@torch.no_grad()
def unrestricted_greedy_sealed_completion(
    backend,
    inputs,
    *,
    evidence_token_positions: Sequence[int],
    bottleneck_layer: int,
    answer: str,
    max_new_tokens: int,
) -> dict:
    from jlens.mmpilot.workspace_replication import unrestricted_greedy_completion

    with seal_evidence_attention(
        backend.blocks,
        evidence_token_positions=evidence_token_positions,
        prompt_len=int(inputs.prompt_len),
        bottleneck_layer=int(bottleneck_layer),
    ) as stats:
        result = unrestricted_greedy_completion(
            backend, inputs, answer=str(answer), max_new_tokens=int(max_new_tokens)
        )
    return _with_seal_diagnostics(result, stats)


@torch.no_grad()
def unrestricted_greedy_sealed_activation_patch_trial(
    backend,
    inputs,
    *,
    donor_by_layer: Mapping[int, torch.Tensor],
    source_position: int,
    evidence_token_positions: Sequence[int],
    bottleneck_layer: int,
    answer: str,
    max_new_tokens: int,
) -> dict:
    from jlens.mmpilot.country_activation_patch import (
        unrestricted_greedy_activation_patch_trial,
    )

    with seal_evidence_attention(
        backend.blocks,
        evidence_token_positions=evidence_token_positions,
        prompt_len=int(inputs.prompt_len),
        bottleneck_layer=int(bottleneck_layer),
    ) as stats:
        result = unrestricted_greedy_activation_patch_trial(
            backend,
            inputs,
            donor_by_layer=donor_by_layer,
            source_position=int(source_position),
            answer=str(answer),
            max_new_tokens=int(max_new_tokens),
        )
    return _with_seal_diagnostics(result, stats)


@torch.no_grad()
def unrestricted_greedy_sealed_swap_trial(
    backend,
    inputs,
    *,
    bases: Mapping,
    alpha: float,
    evidence_token_positions: Sequence[int],
    bottleneck_layer: int,
    answer: str,
    max_new_tokens: int,
    position_rule: str,
    realization_policy,
) -> dict:
    from jlens.mmpilot.workspace_replication import unrestricted_greedy_swap_trial

    with seal_evidence_attention(
        backend.blocks,
        evidence_token_positions=evidence_token_positions,
        prompt_len=int(inputs.prompt_len),
        bottleneck_layer=int(bottleneck_layer),
    ) as stats:
        result = unrestricted_greedy_swap_trial(
            backend,
            inputs,
            bases=bases,
            alpha=float(alpha),
            answer=str(answer),
            max_new_tokens=int(max_new_tokens),
            position_rule=str(position_rule),
            realization_policy=realization_policy,
        )
    return _with_seal_diagnostics(result, stats)


@torch.no_grad()
def unrestricted_greedy_sealed_scaffolded_swap_trial(
    backend,
    inputs,
    *,
    scaffold_by_layer: Mapping[int, torch.Tensor],
    source_position: int,
    bases: Mapping,
    alpha: float,
    evidence_token_positions: Sequence[int],
    bottleneck_layer: int,
    answer: str,
    max_new_tokens: int,
    position_rule: str,
    realization_policy,
) -> dict:
    """Restore one matched source state, then vary only swap coordinates.

    Forward hooks run in registration order.  The source-state scaffold is
    registered first; the coordinate hook is registered inside it and thus
    receives the restored source activation.  Every coordinate condition uses
    the identical scaffold and evidence seal.
    """

    from jlens.mmpilot.country_activation_patch import activation_patch_band
    from jlens.mmpilot.workspace_replication import unrestricted_greedy_swap_trial

    with seal_evidence_attention(
        backend.blocks,
        evidence_token_positions=evidence_token_positions,
        prompt_len=int(inputs.prompt_len),
        bottleneck_layer=int(bottleneck_layer),
    ) as seal_stats:
        with activation_patch_band(
            backend.blocks,
            scaffold_by_layer,
            source_position=int(source_position),
            prompt_len=int(inputs.prompt_len),
        ) as scaffold_stats:
            result = unrestricted_greedy_swap_trial(
                backend,
                inputs,
                bases=bases,
                alpha=float(alpha),
                answer=str(answer),
                max_new_tokens=int(max_new_tokens),
                position_rule=str(position_rule),
                realization_policy=realization_policy,
            )
    expected = int(result["n_forward_passes"])
    scaffold_by_layer_record = {
        str(layer): dict(scaffold_stats[layer]) for layer in sorted(scaffold_stats)
    }
    scaffold_diagnostics = {
        "hook_order": "source_state_scaffold_then_coordinate_exchange",
        "expected_forward_passes": expected,
        "by_layer": scaffold_by_layer_record,
        "all_hooks_fired": bool(scaffold_by_layer_record)
        and all(
            int(row["n_forward_passes"]) == expected
            for row in scaffold_by_layer_record.values()
        ),
        "all_finite": bool(scaffold_by_layer_record)
        and all(bool(row["all_finite"]) for row in scaffold_by_layer_record.values()),
    }
    return {
        **_with_seal_diagnostics(result, seal_stats),
        "matched_scaffold_version": MATCHED_SCAFFOLD_VERSION,
        "matched_scaffold_diagnostics": scaffold_diagnostics,
    }


def sealed_integrity(result: Mapping, *, require_intervention: bool = False) -> bool:
    seal = result.get("evidence_seal_diagnostics") or {}
    passed = bool(
        seal.get("all_hooks_fired")
        and seal.get("all_masks_rank_four")
        and seal.get("all_masks_finite_or_negative_infinity")
        and seal.get("generated_tokens_sealed_at_every_layer")
    )
    if not require_intervention:
        return passed
    intervention = result.get("activation_patch_diagnostics")
    if intervention is not None:
        return bool(
            passed
            and intervention.get("all_hooks_fired")
            and intervention.get("all_finite")
        )
    swap = result.get("intervention_diagnostics") or {}
    scaffold = result.get("matched_scaffold_diagnostics")
    scaffold_passed = bool(
        scaffold is None
        or (scaffold.get("all_hooks_fired") and scaffold.get("all_finite"))
    )
    return bool(
        passed
        and scaffold_passed
        and swap.get("all_hooks_fired")
        and swap.get("all_finite")
    )


def _summary(rows: Sequence[Mapping], condition: str) -> dict:
    selected = [row for row in rows if row.get("condition") == condition]
    successes = sum(bool(row.get("success")) for row in selected)
    return {
        "n": len(selected),
        "successes": successes,
        "rate": successes / len(selected) if selected else 0.0,
        "integrity_pass": bool(selected)
        and all(bool(row.get("integrity_pass")) for row in selected),
    }


def sealed_development_report(
    state_rows: Sequence[Mapping],
    coordinate_rows: Sequence[Mapping],
    *,
    expected_n: int,
    properties: Sequence[str],
    modalities: Sequence[str],
    band: Sequence[int],
) -> dict:
    """Score the bottleneck diagnostic while keeping both arms separate."""

    state_records = [dict(row) for row in state_rows]
    coordinate_records = [dict(row) for row in coordinate_rows]
    state_cells = []
    coordinate_cells = []
    for property_name in map(str, properties):
        for modality in map(str, modalities):
            state_subset = [
                row for row in state_records
                if row.get("property") == property_name and row.get("modality") == modality
            ]
            state_conditions = {
                name: _summary(state_subset, name) for name in SEALED_STATE_CONDITIONS
            }
            state_complete = all(
                row["n"] == int(expected_n) for row in state_conditions.values()
            )
            state_pass = bool(
                state_complete
                and state_conditions["clean_sealed"]["rate"] == 1.0
                and state_conditions["target_state"]["rate"] >= 0.75
                and state_conditions["unrelated_state"]["rate"] == 0.0
                and all(row["integrity_pass"] for row in state_conditions.values())
            )
            state_cells.append({
                "property": property_name,
                "modality": modality,
                "conditions": state_conditions,
                "passed": state_pass,
            })

            coordinate_subset = [
                row for row in coordinate_records
                if row.get("property") == property_name and row.get("modality") == modality
            ]
            coordinate_conditions = {
                name: _summary(coordinate_subset, name)
                for name in SEALED_COORDINATE_CONDITIONS
            }
            exact = coordinate_conditions["exact"]
            coordinate_complete = all(
                row["n"] == int(expected_n) for row in coordinate_conditions.values()
            )
            margins = {
                name: exact["rate"] - coordinate_conditions[name]["rate"]
                for name in ("zero", "random", "unrelated")
            }
            coordinate_pass = bool(
                coordinate_complete
                and exact["rate"] >= 0.75
                and min(margins.values()) >= 0.25
                and all(row["integrity_pass"] for row in coordinate_conditions.values())
            )
            coordinate_cells.append({
                "property": property_name,
                "modality": modality,
                "conditions": coordinate_conditions,
                "margins": margins,
                "passed": coordinate_pass,
            })

    state_passed = bool(state_cells) and all(row["passed"] for row in state_cells)
    coordinate_passed = bool(coordinate_cells) and all(
        row["passed"] for row in coordinate_cells
    )
    if state_passed and coordinate_passed:
        verdict = "COUNTRY_SEALED_EVIDENCE_BOTH_GO"
    elif state_passed:
        verdict = "COUNTRY_SEALED_EVIDENCE_STATE_ONLY_GO"
    elif coordinate_passed:
        verdict = "COUNTRY_SEALED_EVIDENCE_JLENS_ONLY_GO"
    else:
        verdict = "COUNTRY_SEALED_EVIDENCE_NO_GO"
    body = {
        "version": SEALED_DEVELOPMENT_VERSION,
        "verdict": verdict,
        "stage": "development",
        "method_role": "controlled_bottleneck_diagnostic_not_original_paper_method",
        "fitting_performed": False,
        "backward_passes": 0,
        "fresh_confirmation_opened": False,
        "band": list(map(int, band)),
        "bottleneck_layer": int(tuple(map(int, band))[0]),
        "state_arm": {"passed": state_passed, "cells": state_cells, "rows": state_records},
        "j_lens_coordinate_arm": {
            "passed": coordinate_passed,
            "cells": coordinate_cells,
            "rows": coordinate_records,
        },
    }
    return {**body, "report_checksum": payload_checksum(body)}


def matched_scaffold_report(
    state_rows: Sequence[Mapping],
    coordinate_rows: Sequence[Mapping],
    *,
    expected_n: int,
    properties: Sequence[str],
    modalities: Sequence[str],
    band: Sequence[int],
) -> dict:
    """Score the fair bottleneck test with one identical source scaffold."""

    state_records = [dict(row) for row in state_rows]
    coordinate_records = [dict(row) for row in coordinate_rows]
    state_cells = []
    coordinate_cells = []
    for property_name in map(str, properties):
        for modality in map(str, modalities):
            state_subset = [
                row
                for row in state_records
                if row.get("property") == property_name
                and row.get("modality") == modality
            ]
            state_conditions = {
                name: _summary(state_subset, name)
                for name in ("self_scaffold", "target_state", "unrelated_state")
            }
            state_pass = bool(
                all(row["n"] == int(expected_n) for row in state_conditions.values())
                and state_conditions["self_scaffold"]["rate"] == 1.0
                and state_conditions["target_state"]["rate"] >= 0.75
                and state_conditions["unrelated_state"]["rate"] == 0.0
                and all(row["integrity_pass"] for row in state_conditions.values())
            )
            state_cells.append(
                {
                    "property": property_name,
                    "modality": modality,
                    "conditions": state_conditions,
                    "passed": state_pass,
                }
            )

            coordinate_subset = [
                row
                for row in coordinate_records
                if row.get("property") == property_name
                and row.get("modality") == modality
            ]
            conditions = {
                name: _summary(coordinate_subset, name)
                for name in SEALED_COORDINATE_CONDITIONS
            }
            exact = conditions["exact"]
            margins = {
                name: exact["rate"] - conditions[name]["rate"]
                for name in ("zero", "random", "unrelated")
            }
            coordinate_pass = bool(
                all(row["n"] == int(expected_n) for row in conditions.values())
                and exact["rate"] >= 0.75
                and min(margins.values()) >= 0.25
                and all(row["integrity_pass"] for row in conditions.values())
            )
            coordinate_cells.append(
                {
                    "property": property_name,
                    "modality": modality,
                    "conditions": conditions,
                    "margins": margins,
                    "passed": coordinate_pass,
                }
            )

    state_passed = bool(state_cells) and all(row["passed"] for row in state_cells)
    coordinate_passed = bool(coordinate_cells) and all(
        row["passed"] for row in coordinate_cells
    )
    if state_passed and coordinate_passed:
        verdict = "COUNTRY_MATCHED_SCAFFOLD_BOTH_GO"
    elif state_passed:
        verdict = "COUNTRY_MATCHED_SCAFFOLD_STATE_ONLY_GO"
    elif coordinate_passed:
        verdict = "COUNTRY_MATCHED_SCAFFOLD_JLENS_ONLY_GO"
    else:
        verdict = "COUNTRY_MATCHED_SCAFFOLD_NO_GO"
    body = {
        "version": MATCHED_SCAFFOLD_VERSION,
        "verdict": verdict,
        "stage": "development",
        "method_role": "matched_state_scaffold_bottleneck_diagnostic",
        "fitting_performed": False,
        "backward_passes": 0,
        "fresh_confirmation_opened": False,
        "only_coordinate_condition_varies_after_source_scaffold": True,
        "band": list(map(int, band)),
        "state_arm": {
            "passed": state_passed,
            "cells": state_cells,
            "rows": state_records,
        },
        "j_lens_coordinate_arm": {
            "passed": coordinate_passed,
            "cells": coordinate_cells,
            "rows": coordinate_records,
        },
    }
    return {**body, "report_checksum": payload_checksum(body)}


def audit_unopened_confirmation_outputs(
    runs_root: str | Path,
    confirmation_unit_ids: Sequence[str],
    *,
    manifest_checksum: str | None = None,
    split_id: str | None = None,
    on_candidate_scanned: Callable[[int, int, Path, int], None] | None = None,
) -> dict:
    """Find stored generated outputs tied to any proposed confirmation unit.

    Args:
        on_candidate_scanned: Optional progress callback invoked after each
            candidate file is read, as ``(index, total, path, file_bytes)``
            (1-indexed). This function returns nothing until every candidate
            is scanned, which for a large report is otherwise indistinguishable
            from being hung; callers that print from this run it live rather
            than reporting progress after the fact.
    """

    wanted = {str(value) for value in confirmation_unit_ids}
    if not wanted:
        raise CountryEvidenceSealRefused("confirmation freshness audit needs unit ids")
    output_keys = {
        "generated_text",
        "generated_token_ids",
        "answer_match",
        "success",
    }
    findings = []

    def walk(value, path: Path) -> None:
        if isinstance(value, Mapping):
            unit_id = str(value.get("unit_id") or "")
            if unit_id in wanted and any(key in value for key in output_keys):
                findings.append(
                    {
                        "path": str(path),
                        "unit_id": unit_id,
                        "output_keys": sorted(output_keys.intersection(value)),
                    }
                )
            for child in value.values():
                walk(child, path)
        elif isinstance(value, list):
            for child in value:
                walk(child, path)

    root = Path(runs_root)
    candidate_paths: list[Path]
    fingerprints_scanned = 0
    candidate_stores: list[str] = []
    # Every run this study writes binds its fingerprint to the same
    # manifest_checksum (the population digest), so a bounded scan that
    # matches on it should find every prior store belonging to this
    # population -- *provided* that digest never changed across sessions.
    # It is not re-derived here (that would require reopening the population
    # itself), so this cannot detect a session that ran under a different
    # digest and would therefore fall outside the bounded scan. Every
    # fingerprint this audit actually reads is recorded below regardless of
    # whether it matched, specifically so a second, unexpected
    # manifest_checksum among them is visible in the printed report rather
    # than silently narrowing what got scanned.
    unmatched_fingerprints: list[dict] = []
    if manifest_checksum is not None and split_id is not None and root.is_dir():
        fingerprint_paths = sorted(
            {
                *root.glob("*/fingerprint.json"),
                *root.glob("*/diagnostics/*/fingerprint.json"),
            }
        )
        stores = []
        for fingerprint_path in fingerprint_paths:
            fingerprints_scanned += 1
            try:
                fingerprint = json.loads(
                    fingerprint_path.read_text(encoding="utf-8")
                )
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            if (
                fingerprint.get("manifest_checksum") == manifest_checksum
                or fingerprint.get("split_id") == split_id
            ):
                stores.append(fingerprint_path.parent)
            else:
                unmatched_fingerprints.append(
                    {
                        "path": str(fingerprint_path),
                        "manifest_checksum": fingerprint.get("manifest_checksum"),
                        "split_id": fingerprint.get("split_id"),
                    }
                )
        candidate_stores = [str(path) for path in stores]
        candidate_paths = sorted(
            {
                path
                for store in stores
                for path in (
                    *store.glob("*.json"),
                    *store.glob("units/*/*.json"),
                )
                if path.name != "fingerprint.json"
            }
        )
    else:
        candidate_paths = sorted(root.rglob("*.json")) if root.is_dir() else []

    # Checked as raw bytes first, not decoded text: a full UTF-8 decode of a
    # multi-hundred-MB report is real, avoidable CPU cost paid on every
    # candidate file whether or not it turns out to mention any confirmation
    # id. The unit ids are plain ASCII, so a byte-level substring search finds
    # every occurrence a text search would; decoding only happens for a file
    # that already matched and therefore needs walking.
    wanted_bytes = [unit_id.encode("utf-8") for unit_id in wanted]
    json_files_read = 0
    candidate_bytes_read = 0
    largest_candidate_file_bytes = 0
    total_candidates = len(candidate_paths)

    # Reading thousands of small files one at a time over a networked mount
    # (Drive FUSE) is dominated by per-file round-trip latency, not bytes --
    # a single population's own run can easily produce several thousand
    # candidates, since every diagnostic sub-stage of one busy pipeline run
    # shares that population's digest. The reads are independent and
    # I/O-bound (each releases the GIL), so a thread pool turns thousands of
    # serial round-trips into a bounded number of concurrent ones without
    # changing which bytes are read or what is checked in them.
    def _read_bytes(path: Path) -> bytes | None:
        try:
            return path.read_bytes()
        except OSError:
            return None

    with ThreadPoolExecutor(max_workers=32) as pool:
        prefetched = list(pool.map(_read_bytes, candidate_paths))

    for index, (path, raw_bytes) in enumerate(
        zip(candidate_paths, prefetched), start=1
    ):
        if raw_bytes is None:
            continue
        candidate_bytes_read += len(raw_bytes)
        largest_candidate_file_bytes = max(largest_candidate_file_bytes, len(raw_bytes))
        if on_candidate_scanned is not None:
            on_candidate_scanned(index, total_candidates, path, len(raw_bytes))
        if not any(needle in raw_bytes for needle in wanted_bytes):
            continue
        try:
            payload = json.loads(raw_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        json_files_read += 1
        walk(payload, path)
    body = {
        "version": "mmpilot.country_confirmation_freshness_audit.v1",
        "runs_root": str(root),
        "confirmation_unit_ids": sorted(wanted),
        "n_json_findings": len(findings),
        "findings": findings,
        "fresh": not findings,
        "audit_strategy": (
            "bounded_fingerprint_first_v1"
            if manifest_checksum is not None and split_id is not None
            else "recursive_compatibility_v1"
        ),
        "fingerprints_scanned": fingerprints_scanned,
        "candidate_stores": candidate_stores,
        "candidate_json_files": len(candidate_paths),
        "matching_json_files_read": json_files_read,
        "candidate_bytes_read": candidate_bytes_read,
        "largest_candidate_file_bytes": largest_candidate_file_bytes,
        # Fingerprints this scan actually opened but excluded, because their
        # manifest_checksum/split_id disagreed with today's. A run legitimately
        # belonging to this population should never appear here; if one does,
        # this population's digest was not stable across sessions and the
        # bounded scan cannot be trusted without reconciling it by hand.
        "unmatched_fingerprints": unmatched_fingerprints,
        "distinct_unmatched_manifest_checksums": sorted(
            {
                entry["manifest_checksum"]
                for entry in unmatched_fingerprints
                if entry["manifest_checksum"] is not None
            }
        ),
    }
    return {**body, "audit_checksum": payload_checksum(body)}


def matched_scaffold_confirmation_report(
    state_rows: Sequence[Mapping],
    coordinate_rows: Sequence[Mapping],
    *,
    expected_n: int,
    primary_modalities: Sequence[str] = ("image", "spoken_audio"),
    secondary_modalities: Sequence[str] = ("text",),
    familywise_alpha: float = 0.05,
) -> dict:
    """Confirm the frozen pooled continent signal on untouched examples."""

    from jlens.mmpilot.multimodal_lens import (
        holm_adjust,
        paired_binary_one_sided_p,
    )

    state_records = [dict(row) for row in state_rows]
    coordinate_records = [dict(row) for row in coordinate_rows]
    all_modalities = tuple(map(str, (*primary_modalities, *secondary_modalities)))
    state_cells = []
    coordinate_cells = []
    for modality in all_modalities:
        state_subset = [row for row in state_records if row.get("modality") == modality]
        for condition in ("self_scaffold", "target_state", "unrelated_state"):
            unit_ids = [
                str(row.get("unit_id"))
                for row in state_subset
                if row.get("condition") == condition
            ]
            if len(unit_ids) != int(expected_n) or len(set(unit_ids)) != int(expected_n):
                raise CountryEvidenceSealRefused(
                    f"confirmation {modality}/{condition} is not exactly "
                    f"{expected_n} distinct units"
                )
        state_conditions = {
            name: _summary(state_subset, name)
            for name in ("self_scaffold", "target_state", "unrelated_state")
        }
        state_cells.append({"modality": modality, "conditions": state_conditions})
        coordinate_subset = [
            row for row in coordinate_records if row.get("modality") == modality
        ]
        for condition in SEALED_COORDINATE_CONDITIONS:
            unit_ids = [
                str(row.get("unit_id"))
                for row in coordinate_subset
                if row.get("condition") == condition
            ]
            if len(unit_ids) != int(expected_n) or len(set(unit_ids)) != int(expected_n):
                raise CountryEvidenceSealRefused(
                    f"confirmation {modality}/{condition} is not exactly "
                    f"{expected_n} distinct units"
                )
        conditions = {
            name: _summary(coordinate_subset, name)
            for name in SEALED_COORDINATE_CONDITIONS
        }
        coordinate_cells.append({"modality": modality, "conditions": conditions})

    primary_set = set(map(str, primary_modalities))
    primary_state = [row for row in state_records if row.get("modality") in primary_set]
    primary_coordinate = [
        row for row in coordinate_records if row.get("modality") in primary_set
    ]
    exact_by_key = {
        (row["unit_id"], row["modality"]): bool(row.get("success"))
        for row in primary_coordinate
        if row.get("condition") == "exact"
    }
    comparisons = []
    for control_name in ("zero", "random", "unrelated"):
        control_by_key = {
            (row["unit_id"], row["modality"]): bool(row.get("success"))
            for row in primary_coordinate
            if row.get("condition") == control_name
        }
        if set(exact_by_key) != set(control_by_key):
            raise CountryEvidenceSealRefused(
                f"confirmation {control_name} outcomes are not paired to exact"
            )
        ordered = sorted(exact_by_key)
        comparisons.append(
            {
                "control": control_name,
                **paired_binary_one_sided_p(
                    [exact_by_key[key] for key in ordered],
                    [control_by_key[key] for key in ordered],
                ),
            }
        )
    adjusted = holm_adjust(comparisons)

    pooled_exact = _summary(primary_coordinate, "exact")
    pooled_controls = {
        name: _summary(primary_coordinate, name)
        for name in ("zero", "random", "unrelated")
    }
    modality_presence = {}
    for modality in map(str, primary_modalities):
        subset = [row for row in primary_coordinate if row.get("modality") == modality]
        exact = _summary(subset, "exact")
        controls = {name: _summary(subset, name) for name in pooled_controls}
        modality_presence[modality] = {
            "exact": exact,
            "controls": controls,
            "passed": bool(
                exact["n"] == int(expected_n)
                and exact["rate"] >= 0.25
                and exact["rate"] > max(row["rate"] for row in controls.values())
                and exact["integrity_pass"]
                and all(row["integrity_pass"] for row in controls.values())
            ),
        }
    state_gate = bool(primary_state) and all(
        cell["conditions"]["self_scaffold"]["n"] == int(expected_n)
        and cell["conditions"]["target_state"]["n"] == int(expected_n)
        and cell["conditions"]["unrelated_state"]["n"] == int(expected_n)
        and cell["conditions"]["self_scaffold"]["rate"] == 1.0
        and cell["conditions"]["target_state"]["rate"] >= 0.75
        and cell["conditions"]["unrelated_state"]["rate"] == 0.0
        and all(row["integrity_pass"] for row in cell["conditions"].values())
        for cell in state_cells
        if cell["modality"] in primary_set
    )
    pooled_gate = bool(
        pooled_exact["n"] == int(expected_n) * len(primary_set)
        and pooled_exact["rate"] >= 0.50
        and all(
            row["n"] == pooled_exact["n"]
            and pooled_exact["rate"] - row["rate"] >= 0.25
            for row in pooled_controls.values()
        )
        and pooled_exact["integrity_pass"]
        and all(row["integrity_pass"] for row in pooled_controls.values())
    )
    statistics_gate = bool(adjusted) and all(
        float(row["holm_adjusted_p"]) <= float(familywise_alpha)
        for row in adjusted
    )
    passed = bool(
        state_gate
        and pooled_gate
        and statistics_gate
        and all(row["passed"] for row in modality_presence.values())
    )
    body = {
        "version": MATCHED_CONFIRMATION_VERSION,
        "verdict": (
            "COUNTRY_MATCHED_SCAFFOLD_FRESH_CONFIRMATION_GO"
            if passed
            else "COUNTRY_MATCHED_SCAFFOLD_FRESH_CONFIRMATION_NO_GO"
        ),
        "stage": "fresh_confirmation",
        "property": "continent",
        "direction": "France->China",
        "primary_modalities": list(map(str, primary_modalities)),
        "secondary_modalities": list(map(str, secondary_modalities)),
        "pooled_primary": {
            "exact": pooled_exact,
            "controls": pooled_controls,
            "paired_comparisons": adjusted,
        },
        "primary_modality_presence": modality_presence,
        "state_gate_passed": state_gate,
        "pooled_effect_gate_passed": pooled_gate,
        "familywise_statistics_gate_passed": statistics_gate,
        "familywise_alpha": float(familywise_alpha),
        "state_cells": state_cells,
        "coordinate_cells": coordinate_cells,
        "state_rows": state_records,
        "coordinate_rows": coordinate_records,
        "fitting_performed": False,
        "backward_passes": 0,
    }
    return {**body, "report_checksum": payload_checksum(body)}


__all__ = [
    "EVIDENCE_SEAL_VERSION",
    "MATCHED_SCAFFOLD_VERSION",
    "MATCHED_CONFIRMATION_VERSION",
    "SEALED_COORDINATE_CONDITIONS",
    "SEALED_DEVELOPMENT_VERSION",
    "SEALED_STATE_CONDITIONS",
    "CountryEvidenceSealRefused",
    "evidence_positions",
    "audit_unopened_confirmation_outputs",
    "matched_scaffold_confirmation_report",
    "matched_scaffold_report",
    "seal_evidence_attention",
    "sealed_development_report",
    "sealed_integrity",
    "unrestricted_greedy_sealed_activation_patch_trial",
    "unrestricted_greedy_sealed_completion",
    "unrestricted_greedy_sealed_scaffolded_swap_trial",
    "unrestricted_greedy_sealed_swap_trial",
]
