# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Held-out lens evaluation with named controls and aggregate statistics.

Supersedes the single-prompt control table of the smoke/pilot notebooks
(see ``docs/pilot_report.md``, "Audit of the wrong-layer control") for
future evaluations. Three things change:

1. **Named controls with recorded provenance.** The ambiguous cyclic
   "wrong_layer" control is replaced by explicitly named layer-mapping
   controls (``adjacent_layer``, ``distant_layer``, ``shuffled_layer``);
   every control's exact layer mapping and seed is recorded in the output,
   and tests verify each control lens holds exactly the matrix it claims.
   The primary destructive row-permuted control, the scale-matched random
   control, and the ordinary logit-lens baseline are preserved unchanged.
2. **Aggregate statistics.** Per-example ranks are aggregated into median
   rank, mean reciprocal rank, and top-1/5/10 hit rates — per layer, per
   variant, per prompt category, and per format — instead of a single
   combined row.
3. **A categorized evaluation set.** ``configs/prompts/eval_prompts_v2.json``
   spans several small task categories; plain-text and chat-formatted
   prompts are always evaluated and aggregated separately. It is a
   deterministic held-out probe set, not a comprehensive benchmark.

The completed smoke and pilot artifacts are not reinterpreted or altered by
this module.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from jlens.controls import (
    adjacent_layer_mapping,
    control_lens,
    distant_layer_mapping,
    layer_mapped_lens,
    mapping_provenance,
    ranks_of_targets,
    shuffled_layer_mapping,
    topk_overlap,
)
from jlens.lens import JacobianLens
from jlens.protocol import LensModel


@dataclass(frozen=True)
class ControlVariant:
    """One evaluation variant: a lens (or the identity baseline) plus the
    provenance needed to state exactly which matrices it applies."""

    name: str
    lens: JacobianLens
    use_jacobian: bool = True
    #: Serializable record of how this variant's matrices were derived.
    provenance: dict = field(default_factory=dict)


def build_control_suite(
    lens: JacobianLens, *, control_seed: int
) -> dict[str, ControlVariant]:
    """The named evaluation suite for a fitted lens.

    Variants:
        - ``jlens`` — the fitted lens itself.
        - ``logit_lens`` — identity transport (``use_jacobian=False``);
          the ordinary logit-lens baseline.
        - ``permuted`` — row-permuted fitted ``J`` (primary destructive
          control; preserved from the smoke/pilot protocol).
        - ``random`` — scale-matched i.i.d. Gaussian (preserved).
        - ``adjacent_layer`` — each layer transported with the nearest other
          fitted layer's ``J`` (weak mismatch).
        - ``distant_layer`` — each layer transported with the farthest
          fitted layer's ``J`` (strong mismatch).
        - ``shuffled_layer`` — a seeded derangement of the fitted layers
          (mixed mismatch, no systematic cyclic artifact).

    Layer-mapping controls require >= 2 fitted layers and are omitted (not
    silently approximated) otherwise.
    """
    suite: dict[str, ControlVariant] = {
        "jlens": ControlVariant(
            "jlens", lens, provenance={"kind": "fitted", "transport": "J_l @ h"}
        ),
        "logit_lens": ControlVariant(
            "logit_lens",
            lens,
            use_jacobian=False,
            provenance={"kind": "identity", "transport": "h (no J)"},
        ),
        "permuted": ControlVariant(
            "permuted",
            control_lens(lens, "permuted", seed=control_seed),
            provenance={
                "kind": "row_permuted_fitted_J",
                "seed": control_seed,
                "note": "primary destructive control; layer l uses seed+l",
            },
        ),
        "random": ControlVariant(
            "random",
            control_lens(lens, "random", seed=control_seed),
            provenance={
                "kind": "scale_matched_random",
                "seed": control_seed,
                "note": "iid Gaussian matched to ||J_l||_F; layer l uses seed+l",
            },
        ),
    }
    if len(lens.source_layers) >= 2:
        for name, mapping in (
            ("adjacent_layer", adjacent_layer_mapping(lens.source_layers)),
            ("distant_layer", distant_layer_mapping(lens.source_layers)),
            (
                "shuffled_layer",
                shuffled_layer_mapping(lens.source_layers, seed=control_seed),
            ),
        ):
            provenance: dict[str, Any] = {
                "kind": "layer_mapped_fitted_J",
                "mapping": mapping_provenance(mapping),
            }
            if name == "shuffled_layer":
                provenance["seed"] = control_seed
            suite[name] = ControlVariant(
                name, layer_mapped_lens(lens, mapping), provenance=provenance
            )
    return suite


def load_eval_prompts_v2(path: str, tokenizer: Any) -> list[dict]:
    """Flatten an eval_prompts_v2 file into evaluation rows.

    Returns ``[{slug, category, format, text, positions}]``. Chat entries
    are rendered through the tokenizer's chat template at load time; plain
    and chat rows carry their format so downstream aggregation never mixes
    them.
    """
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    rows: list[dict] = []
    for entry in payload.get("plain", []):
        rows.append(
            {
                "slug": entry["slug"],
                "category": entry["category"],
                "format": "plain",
                "text": entry["text"],
                "positions": entry.get("positions", [-1]),
            }
        )
    for entry in payload.get("chat", []):
        messages = []
        if entry.get("system"):
            messages.append({"role": "system", "content": entry["system"]})
        messages.append({"role": "user", "content": entry["user"]})
        if entry.get("assistant_prefill"):
            messages.append(
                {"role": "assistant", "content": entry["assistant_prefill"]}
            )
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, continue_final_message=True
            )
        else:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        rows.append(
            {
                "slug": entry["slug"],
                "category": entry["category"],
                "format": "chat",
                "text": text,
                "positions": entry.get("positions", [-1]),
            }
        )
    return rows


def aggregate_ranks(ranks: Sequence[int]) -> dict:
    """Aggregate statistics over per-example ranks (0 = argmax).

    Returns ``n``, ``median_rank``, ``mean_reciprocal_rank`` (mean of
    ``1/(rank+1)``), and ``hit_rate@{1,5,10}`` (fraction with rank < k).
    """
    if len(ranks) == 0:
        return {
            "n": 0,
            "median_rank": None,
            "mean_reciprocal_rank": None,
            "hit_rate@1": None,
            "hit_rate@5": None,
            "hit_rate@10": None,
        }
    import statistics

    tensor = torch.as_tensor(list(ranks), dtype=torch.float64)
    return {
        "n": len(ranks),
        "median_rank": float(statistics.median(list(ranks))),
        "mean_reciprocal_rank": float((1.0 / (tensor + 1.0)).mean()),
        "hit_rate@1": float((tensor < 1).float().mean()),
        "hit_rate@5": float((tensor < 5).float().mean()),
        "hit_rate@10": float((tensor < 10).float().mean()),
    }


@torch.no_grad()
def capture_residuals(
    model: LensModel,
    text: str,
    *,
    layers: Sequence[int],
    positions: Sequence[int],
    max_seq_len: int = 512,
) -> tuple[dict[int, torch.Tensor], torch.Tensor, torch.Tensor]:
    """One forward pass; returns ``({layer: [n_positions, d_model]},
    model_logits [n_positions, vocab], input_ids [1, seq_len])`` at the
    requested positions. Used both by :func:`evaluate_suite` and by the
    decomposition workflow (which decomposes exactly these residuals)."""
    from jlens.hooks import ActivationRecorder

    final_layer = model.n_layers - 1
    record_at = sorted(set(layers) | {final_layer})
    input_ids = model.encode(text, max_length=max_seq_len)
    with ActivationRecorder(model.layers, at=record_at) as recorder:
        model.forward(input_ids)
        activations = {i: recorder.activations[i].detach() for i in record_at}

    def select(layer: int) -> torch.Tensor:
        full = activations[layer][0]  # [seq_len, d_model]
        return full[list(positions)].float()

    residuals = {layer: select(layer) for layer in layers}
    model_logits = model.unembed(select(final_layer)).float().cpu()
    return residuals, model_logits, input_ids.cpu()


def evaluate_suite(
    model: LensModel,
    suite: dict[str, ControlVariant],
    prompts: Sequence[dict],
    *,
    layers: Sequence[int],
    top_k: int = 10,
    max_seq_len: int = 512,
) -> dict:
    """Evaluate every variant on every prompt/position and aggregate.

    Per (prompt, position, layer, variant) this records the top-``top_k``
    overlap with the model's real output logits and the rank of the model's
    actual top-1 token. Aggregation is per (format, layer, variant) and per
    (format, category, layer, variant); plain and chat formats are never
    combined. Control provenance is embedded in the returned payload.

    Returns a JSON-serializable dict with keys ``examples`` (per-example
    records), ``aggregates``, ``aggregates_by_category``, ``provenance``,
    ``layers``, ``top_k``.
    """
    layers = list(layers)
    examples: list[dict] = []
    # (format, layer, variant) -> [ranks]; (format, category, layer, variant) -> [ranks]
    rank_bins: dict[tuple, list[int]] = {}
    overlap_bins: dict[tuple, list[float]] = {}

    for prompt in prompts:
        positions = list(prompt["positions"])
        residuals, model_logits, _ = capture_residuals(
            model, prompt["text"], layers=layers, positions=positions,
            max_seq_len=max_seq_len,
        )
        # One forward per prompt; every variant transports the same captured
        # residuals, so differences are attributable to the matrices alone.
        per_variant_logits: dict[str, dict[int, torch.Tensor]] = {}
        for name, variant in suite.items():
            per_variant_logits[name] = {}
            for layer in layers:
                residual = residuals[layer]
                if variant.use_jacobian:
                    residual = variant.lens.transport(residual, layer)
                per_variant_logits[name][layer] = (
                    model.unembed(residual).float().cpu()
                )
        model_top1 = model_logits.argmax(-1)  # [n_positions]

        record: dict[str, Any] = {
            "slug": prompt["slug"],
            "category": prompt.get("category", "uncategorized"),
            "format": prompt["format"],
            "positions": positions,
            "model_top1_ids": [int(t) for t in model_top1],
            "layers": {},
        }
        for layer in layers:
            layer_record: dict[str, Any] = {}
            for name in suite:
                logits = per_variant_logits[name][layer]
                ranks = [int(r) for r in ranks_of_targets(logits, model_top1)]
                overlap = topk_overlap(logits, model_logits, top_k)
                layer_record[name] = {
                    "topk_overlap_with_model": round(overlap, 4),
                    "rank_of_model_top1": ranks,
                }
                for pos_idx in range(len(positions)):
                    key = (prompt["format"], layer, name)
                    rank_bins.setdefault(key, []).append(ranks[pos_idx])
                    overlap_bins.setdefault(key, []).append(overlap)
                    cat_key = (
                        prompt["format"],
                        prompt.get("category", "uncategorized"),
                        layer,
                        name,
                    )
                    rank_bins.setdefault(cat_key, []).append(ranks[pos_idx])
            record["layers"][layer] = layer_record
        examples.append(record)

    def _bin_payload(key: tuple) -> dict:
        payload = aggregate_ranks(rank_bins[key])
        if key in overlap_bins:
            payload["mean_topk_overlap"] = round(
                float(torch.tensor(overlap_bins[key]).mean()), 4
            )
        return payload

    aggregates: dict[str, dict] = {}
    aggregates_by_category: dict[str, dict] = {}
    for key in sorted(rank_bins, key=str):
        if len(key) == 3:
            fmt, layer, name = key
            aggregates.setdefault(fmt, {}).setdefault(str(layer), {})[name] = (
                _bin_payload(key)
            )
        else:
            fmt, category, layer, name = key
            aggregates_by_category.setdefault(fmt, {}).setdefault(
                category, {}
            ).setdefault(str(layer), {})[name] = _bin_payload(key)

    return {
        "layers": layers,
        "top_k": top_k,
        "n_prompts": len(list(prompts)),
        "examples": examples,
        "aggregates": aggregates,
        "aggregates_by_category": aggregates_by_category,
        "provenance": {
            name: variant.provenance for name, variant in suite.items()
        },
        "notes": (
            "rank_of_model_top1 uses 0 = argmax. Plain and chat formats are "
            "aggregated separately. This is a deterministic held-out probe "
            "set, not a comprehensive benchmark."
        ),
    }
