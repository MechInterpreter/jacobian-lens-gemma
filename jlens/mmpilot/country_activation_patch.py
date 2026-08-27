# SPDX-License-Identifier: Apache-2.0
"""No-refit causal-site diagnostics for the country workspace study.

The country study established that a broad J-lens coordinate exchange can
change the generated country name.  That does not establish that the exchange
replaced the intermediate state used to retrieve a capital or continent.  This
module provides a stricter diagnostic: copy a *real clean activation* from a
matched target-country run into a source-country run at a small, declared site.

The activation patch is only a development-time localization instrument.  It
is never itself reported as a J-lens coordinate-swap result, and its outcomes
cannot open the untouched confirmation population.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from typing import Any

import torch

from jlens.hooks import ActivationRecorder
from jlens.mmpilot.store import payload_checksum

ACTIVATION_PATCH_VERSION = "mmpilot.country_activation_patch.v1"
CAUSAL_SITE_SCREEN_VERSION = "mmpilot.country_causal_site_screen.v1"
RESTRICTED_SWAP_VERSION = "mmpilot.country_restricted_swap_development.v1"
LOCALIZED_DEVELOPMENT_VERSION = "mmpilot.country_localized_development.v1"
PATCH_SITES = ("evidence_endpoint", "final_prompt_token")
SCREEN_CONDITIONS = (
    "target_state",
    "self_state",
    "unrelated_state",
    "direct_answer",
)
RESTRICTED_SWAP_CONDITIONS = ("exact", "zero", "random", "unrelated")


class CountryActivationPatchRefused(RuntimeError):
    """A causal-site diagnostic is ambiguous or violates its frozen design."""


def patch_position(inputs, site: str, *, country_token_id: int | None = None) -> int:
    """Resolve one semantically declared prompt position.

    ``final_prompt_token`` is the answer-ready assistant-prefill endpoint.
    ``evidence_endpoint`` is the last image/audio placeholder token, or the
    unique explicit country token in a text prompt.  Refusing ambiguity is
    important: guessing a text occurrence would turn localization into an
    undocumented hyperparameter search.
    """

    site = str(site)
    if site not in PATCH_SITES:
        raise CountryActivationPatchRefused(
            f"unknown patch site {site!r}; known sites are {PATCH_SITES}"
        )
    prompt_len = int(inputs.prompt_len)
    if prompt_len < 1:
        raise CountryActivationPatchRefused("a patch input has an empty prompt")
    if site == "final_prompt_token":
        return prompt_len - 1

    span = getattr(inputs, "modality_token_range", None)
    if span is not None:
        start, end = map(int, span)
        if not 0 <= start < end <= prompt_len:
            raise CountryActivationPatchRefused(
                f"invalid modality token range [{start}, {end}) for prompt {prompt_len}"
            )
        return end - 1

    if country_token_id is None:
        raise CountryActivationPatchRefused(
            "text evidence-endpoint localization requires the frozen country token id"
        )
    input_ids = inputs.tensors.get("input_ids")
    if not torch.is_tensor(input_ids) or input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise CountryActivationPatchRefused(
            "text evidence-endpoint localization requires [1, seq] input_ids"
        )
    positions = (
        input_ids[0, :prompt_len].detach().cpu() == int(country_token_id)
    ).nonzero(as_tuple=True)[0].tolist()
    if len(positions) != 1:
        raise CountryActivationPatchRefused(
            f"country token {country_token_id} occurs {len(positions)} times; "
            "refusing an ambiguous evidence position"
        )
    return int(positions[0])


def single_position_inputs(inputs, position: int):
    """Return a BuiltInputs view whose evidence span is exactly one position."""

    position = int(position)
    if not 0 <= position < int(inputs.prompt_len):
        raise CountryActivationPatchRefused(
            f"patch position {position} is outside prompt length {inputs.prompt_len}"
        )
    return replace(inputs, modality_token_range=[position, position + 1])


@torch.no_grad()
def capture_activation_sites(
    backend,
    inputs,
    *,
    layers: Sequence[int],
    positions: Mapping[str, int],
) -> dict[str, dict[int, torch.Tensor]]:
    """Capture small fp32 activation slices from one clean donor forward."""

    ordered_layers = tuple(sorted({int(layer) for layer in layers}))
    if not ordered_layers:
        raise CountryActivationPatchRefused("activation capture needs at least one layer")
    resolved_positions = {str(name): int(value) for name, value in positions.items()}
    if not resolved_positions:
        raise CountryActivationPatchRefused("activation capture needs a declared site")
    with ActivationRecorder(backend.blocks, at=ordered_layers) as recorder:
        backend.forward_logits(inputs.tensors)
    captured: dict[str, dict[int, torch.Tensor]] = {}
    for name, position in resolved_positions.items():
        by_layer = {}
        for layer in ordered_layers:
            activation = recorder.activations[layer]
            if activation.ndim != 3 or activation.shape[0] != 1:
                raise CountryActivationPatchRefused(
                    f"layer {layer} donor activation is not [1, seq, d_model]"
                )
            if not 0 <= position < int(activation.shape[1]):
                raise CountryActivationPatchRefused(
                    f"site {name!r} position {position} is outside layer {layer} "
                    f"sequence length {activation.shape[1]}"
                )
            value = activation[0, position].detach().float().cpu()
            if not bool(torch.isfinite(value).all()):
                raise CountryActivationPatchRefused(
                    f"site {name!r} donor activation at layer {layer} is non-finite"
                )
            by_layer[layer] = value
        captured[name] = by_layer
    return captured


@contextmanager
def activation_patch_band(
    blocks: Sequence[torch.nn.Module],
    donor_by_layer: Mapping[int, torch.Tensor],
    *,
    source_position: int,
    prompt_len: int,
):
    """Replace one source position with a clean donor state across a band."""

    layers = tuple(sorted(map(int, donor_by_layer)))
    if not layers:
        raise CountryActivationPatchRefused("an activation patch needs a layer band")
    if tuple(range(layers[0], layers[-1] + 1)) != layers:
        raise CountryActivationPatchRefused(
            f"activation-patch layers must be contiguous, got {list(layers)}"
        )
    source_position = int(source_position)
    if not 0 <= source_position < int(prompt_len):
        raise CountryActivationPatchRefused(
            f"source position {source_position} is outside prompt length {prompt_len}"
        )

    stats: dict[int, dict[str, Any]] = {}
    with ExitStack() as stack:
        for layer in layers:
            donor = donor_by_layer[layer].detach().float().cpu()
            if donor.ndim != 1 or not bool(torch.isfinite(donor).all()):
                raise CountryActivationPatchRefused(
                    f"layer {layer} donor must be one finite d_model vector"
                )
            row = {
                "layer": layer,
                "source_position": source_position,
                "n_forward_passes": 0,
                "all_finite": True,
                "max_update_to_activation_ratio": 0.0,
                "max_after_to_before_activation_ratio": 0.0,
            }
            stats[layer] = row

            def make_hook(layer_index: int, donor_value: torch.Tensor, record: dict):
                def hook(module, hook_inputs, output):
                    is_tensor = torch.is_tensor(output)
                    hidden = output if is_tensor else output[0]
                    if hidden.ndim != 3 or hidden.shape[0] < 1:
                        raise CountryActivationPatchRefused(
                            f"layer {layer_index} output is not [batch, seq, d_model]"
                        )
                    if int(hidden.shape[1]) < int(prompt_len):
                        raise CountryActivationPatchRefused(
                            f"layer {layer_index} sequence is shorter than the frozen prompt"
                        )
                    if int(hidden.shape[2]) != int(donor_value.numel()):
                        raise CountryActivationPatchRefused(
                            f"layer {layer_index} donor width {donor_value.numel()} does "
                            f"not match d_model {hidden.shape[2]}"
                        )
                    before = hidden[0, source_position].detach().float()
                    replacement = donor_value.to(device=hidden.device, dtype=hidden.dtype)
                    new_hidden = hidden.clone()
                    new_hidden[0, source_position] = replacement
                    realized = new_hidden[0, source_position].detach().float()
                    update_ratio = float(
                        (realized - before).norm() / before.norm().clamp_min(1.0)
                    )
                    activation_ratio = float(
                        realized.norm() / before.norm().clamp_min(1.0)
                    )
                    finite = bool(torch.isfinite(realized).all())
                    record["n_forward_passes"] += 1
                    record["all_finite"] = bool(record["all_finite"] and finite)
                    record["max_update_to_activation_ratio"] = max(
                        float(record["max_update_to_activation_ratio"]), update_ratio
                    )
                    record["max_after_to_before_activation_ratio"] = max(
                        float(record["max_after_to_before_activation_ratio"]),
                        activation_ratio,
                    )
                    if is_tensor:
                        return new_hidden
                    return (new_hidden, *tuple(output)[1:])

                return hook

            handle = blocks[layer].register_forward_hook(make_hook(layer, donor, row))
            stack.callback(handle.remove)
        yield stats


@torch.no_grad()
def unrestricted_greedy_activation_patch_trial(
    backend,
    inputs,
    *,
    donor_by_layer: Mapping[int, torch.Tensor],
    source_position: int,
    answer: str,
    max_new_tokens: int,
    diagnostic_token_ids: Mapping[str, int] | None = None,
) -> dict:
    """Patch one declared position on every generation pass and generate freely."""

    from jlens.mmpilot.workspace_replication import unrestricted_greedy_completion

    with activation_patch_band(
        backend.blocks,
        donor_by_layer,
        source_position=int(source_position),
        prompt_len=int(inputs.prompt_len),
    ) as stats:
        generated = unrestricted_greedy_completion(
            backend,
            inputs,
            answer=str(answer),
            max_new_tokens=int(max_new_tokens),
            diagnostic_token_ids=diagnostic_token_ids,
        )
    expected_passes = int(generated["n_forward_passes"])
    by_layer = {
        str(layer): dict(stats[layer]) for layer in sorted(stats)
    }
    all_hooks = all(
        int(row["n_forward_passes"]) == expected_passes
        for row in by_layer.values()
    )
    all_finite = all(bool(row["all_finite"]) for row in by_layer.values())
    return {
        **generated,
        "activation_patch_version": ACTIVATION_PATCH_VERSION,
        "layers_patched": sorted(map(int, donor_by_layer)),
        "source_position": int(source_position),
        "all_prompt_positions_patched": False,
        "activation_patch_diagnostics": {
            "by_layer": by_layer,
            "all_hooks_fired": all_hooks,
            "all_finite": all_finite,
            "expected_forward_passes": expected_passes,
            "max_update_to_activation_ratio": max(
                (float(row["max_update_to_activation_ratio"]) for row in by_layer.values()),
                default=0.0,
            ),
        },
    }


def _summary(rows: Sequence[Mapping], condition: str) -> dict:
    selected = [row for row in rows if row.get("condition") == condition]
    return {
        "n": len(selected),
        "successes": sum(bool(row.get("success")) for row in selected),
        "rate": (
            sum(bool(row.get("success")) for row in selected) / len(selected)
            if selected
            else 0.0
        ),
        "integrity_pass": bool(selected)
        and all(bool(row.get("integrity_pass")) for row in selected),
    }


def causal_site_screen_report(
    rows: Sequence[Mapping],
    *,
    bands: Sequence[Sequence[int]],
    expected_n: int,
    properties: Sequence[str],
    modalities: Sequence[str],
) -> dict:
    """Select a path using positive controls, never exact-swap outcomes."""

    records = [dict(row) for row in rows]
    if any(row.get("condition") not in SCREEN_CONDITIONS for row in records):
        raise CountryActivationPatchRefused(
            "the causal-site screen contains an undeclared condition"
        )
    candidates = []
    for band in (tuple(map(int, item)) for item in bands):
        for site in PATCH_SITES:
            cells = []
            for property_name in map(str, properties):
                for modality in map(str, modalities):
                    subset = [
                        row
                        for row in records
                        if tuple(map(int, row.get("layers_patched") or ())) == band
                        and row.get("site") == site
                        and row.get("property") == property_name
                        and row.get("modality") == modality
                    ]
                    conditions = {
                        condition: _summary(subset, condition)
                        for condition in SCREEN_CONDITIONS
                    }
                    complete = all(
                        item["n"] == int(expected_n) for item in conditions.values()
                    )
                    passed = (
                        complete
                        and conditions["target_state"]["rate"] == 1.0
                        and conditions["direct_answer"]["rate"] == 1.0
                        and conditions["self_state"]["rate"] == 0.0
                        and conditions["unrelated_state"]["rate"] == 0.0
                        and all(item["integrity_pass"] for item in conditions.values())
                    )
                    cells.append(
                        {
                            "property": property_name,
                            "modality": modality,
                            "conditions": conditions,
                            "complete": complete,
                            "passed": passed,
                        }
                    )
            candidate_passed = len(cells) == len(properties) * len(modalities) and all(
                cell["passed"] for cell in cells
            )
            candidates.append(
                {
                    "path_id": f"L{band[0]}-{band[-1]}:{site}",
                    "band": list(band),
                    "site": site,
                    "passed": candidate_passed,
                    "cells": cells,
                }
            )
    passing = [candidate for candidate in candidates if candidate["passed"]]
    passing.sort(key=lambda row: (len(row["band"]), row["band"], row["site"]))
    selected = None
    if passing:
        selected = {
            key: passing[0][key] for key in ("path_id", "band", "site")
        }
    body = {
        "version": CAUSAL_SITE_SCREEN_VERSION,
        "verdict": (
            "COUNTRY_CAUSAL_SITE_SCREEN_GO"
            if selected is not None
            else "COUNTRY_CAUSAL_SITE_SCREEN_NO_GO"
        ),
        "selection_used_coordinate_swap_outcomes": False,
        "selection_used_fresh_confirmation": False,
        "expected_n": int(expected_n),
        "conditions": list(SCREEN_CONDITIONS),
        "selected": selected,
        "candidates": candidates,
        "rows": records,
    }
    return {**body, "report_checksum": payload_checksum(body)}


def state_validated_selection(screen: Mapping) -> dict:
    """Derive a path using only clean-state patches and negative controls.

    This is intentionally separate from the original conjunctive screen.  The
    original verdict remains unchanged: its direct-answer arm failed.  This
    derived development choice answers a narrower question, namely where a
    real target-country state has causal leverage.  It never reads an exact
    J-lens coordinate-swap outcome.
    """

    if screen.get("selection_used_coordinate_swap_outcomes") is not False:
        raise CountryActivationPatchRefused(
            "the source screen does not prove outcome-blind path selection"
        )
    passing = []
    for candidate in screen.get("candidates") or ():
        cells = candidate.get("cells") or ()
        state_pass = bool(cells)
        for cell in cells:
            conditions = cell.get("conditions") or {}
            target = conditions.get("target_state") or {}
            self_state = conditions.get("self_state") or {}
            unrelated = conditions.get("unrelated_state") or {}
            state_pass = bool(
                state_pass
                and target.get("n") == screen.get("expected_n")
                and target.get("rate") == 1.0
                and target.get("integrity_pass") is True
                and self_state.get("n") == screen.get("expected_n")
                and self_state.get("rate") == 0.0
                and self_state.get("integrity_pass") is True
                and unrelated.get("n") == screen.get("expected_n")
                and unrelated.get("rate") == 0.0
                and unrelated.get("integrity_pass") is True
            )
        if state_pass:
            passing.append(
                {
                    key: candidate[key]
                    for key in ("path_id", "band", "site")
                }
            )
    passing.sort(key=lambda row: (len(row["band"]), row["band"], row["site"]))
    body = {
        "version": "mmpilot.country_state_validated_selection.v1",
        "source_screen_checksum": screen.get("report_checksum"),
        "source_screen_verdict_unchanged": screen.get("verdict"),
        "selection_used_coordinate_swap_outcomes": False,
        "selection_used_direct_answer_outcomes": False,
        "fresh_confirmation_opened": False,
        "passing_paths": passing,
        "selected": passing[0] if passing else None,
        "verdict": (
            "COUNTRY_STATE_VALIDATED_PATH_GO"
            if passing
            else "COUNTRY_STATE_VALIDATED_PATH_NO_GO"
        ),
    }
    return {**body, "record_checksum": payload_checksum(body)}


def restricted_swap_report(
    rows: Sequence[Mapping],
    *,
    expected_n: int,
    properties: Sequence[str],
    modalities: Sequence[str],
    band: Sequence[int],
    site: str,
) -> dict:
    """Score the exact exchange on development rows held out from screening."""

    records = [dict(row) for row in rows]
    band = tuple(map(int, band))
    cells = []
    for property_name in map(str, properties):
        for modality in map(str, modalities):
            subset = [
                row
                for row in records
                if row.get("property") == property_name
                and row.get("modality") == modality
            ]
            conditions = {
                condition: _summary(subset, condition)
                for condition in RESTRICTED_SWAP_CONDITIONS
            }
            exact = conditions["exact"]
            margins = {
                name: exact["rate"] - conditions[name]["rate"]
                for name in ("zero", "random", "unrelated")
            }
            complete = all(item["n"] == int(expected_n) for item in conditions.values())
            passed = (
                complete
                and exact["rate"] >= 0.75
                and min(margins.values()) >= 0.25
                and all(item["integrity_pass"] for item in conditions.values())
            )
            cells.append(
                {
                    "property": property_name,
                    "modality": modality,
                    "conditions": conditions,
                    "margins": margins,
                    "complete": complete,
                    "passed": passed,
                }
            )
    passed = len(cells) == len(properties) * len(modalities) and all(
        cell["passed"] for cell in cells
    )
    body = {
        "version": RESTRICTED_SWAP_VERSION,
        "verdict": (
            "COUNTRY_RESTRICTED_SWAP_DEVELOPMENT_GO"
            if passed
            else "COUNTRY_RESTRICTED_SWAP_DEVELOPMENT_NO_GO"
        ),
        "stage": "development",
        "fresh_confirmation_opened": False,
        "direction": "France->China",
        "band": list(band),
        "site": str(site),
        "expected_n": int(expected_n),
        "cells": cells,
        "rows": records,
    }
    return {**body, "report_checksum": payload_checksum(body)}


def localized_development_report(
    state_rows: Sequence[Mapping],
    coordinate_rows: Sequence[Mapping],
    *,
    expected_n: int,
    properties: Sequence[str],
    modalities: Sequence[str],
    selection: Mapping,
) -> dict:
    """Report full-state transfer and J-lens exchange as distinct arms."""

    state_records = [dict(row) for row in state_rows]
    state_cells = []
    for property_name in map(str, properties):
        for modality in map(str, modalities):
            subset = [
                row
                for row in state_records
                if row.get("property") == property_name
                and row.get("modality") == modality
            ]
            conditions = {
                condition: _summary(subset, condition)
                for condition in ("target_state", "self_state", "unrelated_state")
            }
            target = conditions["target_state"]
            margins = {
                name: target["rate"] - conditions[name]["rate"]
                for name in ("self_state", "unrelated_state")
            }
            complete = all(
                item["n"] == int(expected_n) for item in conditions.values()
            )
            passed = bool(
                complete
                and target["rate"] >= 0.75
                and min(margins.values()) >= 0.25
                and all(item["integrity_pass"] for item in conditions.values())
            )
            state_cells.append(
                {
                    "property": property_name,
                    "modality": modality,
                    "conditions": conditions,
                    "margins": margins,
                    "complete": complete,
                    "passed": passed,
                }
            )
    state_passed = bool(state_cells) and all(cell["passed"] for cell in state_cells)
    coordinate = restricted_swap_report(
        coordinate_rows,
        expected_n=expected_n,
        properties=properties,
        modalities=modalities,
        band=selection["band"],
        site=selection["site"],
    )
    coordinate_passed = (
        coordinate["verdict"] == "COUNTRY_RESTRICTED_SWAP_DEVELOPMENT_GO"
    )
    if state_passed and coordinate_passed:
        verdict = "COUNTRY_LOCALIZED_DEVELOPMENT_BOTH_GO"
    elif state_passed:
        verdict = "COUNTRY_LOCALIZED_DEVELOPMENT_STATE_ONLY_GO"
    elif coordinate_passed:
        verdict = "COUNTRY_LOCALIZED_DEVELOPMENT_JLENS_ONLY_GO"
    else:
        verdict = "COUNTRY_LOCALIZED_DEVELOPMENT_NO_GO"
    body = {
        "version": LOCALIZED_DEVELOPMENT_VERSION,
        "verdict": verdict,
        "stage": "development",
        "fresh_confirmation_opened": False,
        "fitting_performed": False,
        "backward_passes": 0,
        "selection": dict(selection),
        "full_state_arm": {
            "passed": state_passed,
            "cells": state_cells,
            "rows": state_records,
        },
        "j_lens_coordinate_arm": coordinate,
    }
    return {**body, "report_checksum": payload_checksum(body)}


__all__ = [
    "ACTIVATION_PATCH_VERSION",
    "CAUSAL_SITE_SCREEN_VERSION",
    "LOCALIZED_DEVELOPMENT_VERSION",
    "PATCH_SITES",
    "RESTRICTED_SWAP_CONDITIONS",
    "RESTRICTED_SWAP_VERSION",
    "SCREEN_CONDITIONS",
    "CountryActivationPatchRefused",
    "activation_patch_band",
    "capture_activation_sites",
    "causal_site_screen_report",
    "localized_development_report",
    "patch_position",
    "restricted_swap_report",
    "single_position_inputs",
    "state_validated_selection",
    "unrestricted_greedy_activation_patch_trial",
]
