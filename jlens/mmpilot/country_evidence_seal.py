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

from collections.abc import Mapping, Sequence
from contextlib import ExitStack, contextmanager
from typing import Any

import torch

from jlens.mmpilot.store import payload_checksum

EVIDENCE_SEAL_VERSION = "mmpilot.country_evidence_seal.v1"
SEALED_DEVELOPMENT_VERSION = "mmpilot.country_sealed_evidence_development.v1"
MATCHED_SCAFFOLD_VERSION = "mmpilot.country_matched_scaffold_development.v1"
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


__all__ = [
    "EVIDENCE_SEAL_VERSION",
    "MATCHED_SCAFFOLD_VERSION",
    "SEALED_COORDINATE_CONDITIONS",
    "SEALED_DEVELOPMENT_VERSION",
    "SEALED_STATE_CONDITIONS",
    "CountryEvidenceSealRefused",
    "evidence_positions",
    "matched_scaffold_report",
    "seal_evidence_attention",
    "sealed_development_report",
    "sealed_integrity",
    "unrestricted_greedy_sealed_activation_patch_trial",
    "unrestricted_greedy_sealed_completion",
    "unrestricted_greedy_sealed_scaffolded_swap_trial",
    "unrestricted_greedy_sealed_swap_trial",
]
