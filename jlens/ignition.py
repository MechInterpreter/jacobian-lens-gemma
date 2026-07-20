# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Candidate-ignition diagnostics over cone trajectories.

**This module does not detect ignition.** The paper's ignition phenomenon —
a sharp commitment to one interpretation around its models' workspace-onset
layer — was established there with evidence (interventions, ambiguous-input
sweeps, cross-model replication) this project has not produced for Gemma.
Everything here is an explicitly labeled *candidate* diagnostic: observable
layer-transition signals that would be *consistent with* an ignition-like
transition and are worth inspecting, no more.

Signals kept separately visible per transition
(:func:`candidate_ignition_signals`):

- ``delta_explained_fraction`` — sharp improvement in sparse
  reconstruction quality between adjacent fitted layers;
- ``active_set_jaccard`` / ``weighted_similarity`` — active-set stability;
- ``delta_herfindahl`` / ``top1_share_to`` — concentration into fewer
  dominant coefficients;
- ``output_alignment_to`` — whether (and with what coefficient share) the
  model's actual top-1 output token is in the active set after the
  transition (alignment with downstream output);
- ``persistence_length_from_here`` — how many consecutive subsequent
  transitions keep active-set Jaccard at or above a threshold.

Semantic coherence of decoded labels is intentionally **not** scored
automatically — no defensible label-coherence metric exists in this
codebase, so labels are exported for human inspection instead.

An optional composite (:func:`heuristic_candidate_score`) is provided for
exploratory ranking only: it is **disabled by default**, its weights are
explicit arguments recorded in the output, and it must not be read as a
scientifically validated ignition score.

Alternative explanations that any apparent transition must be weighed
against (see also ``docs/jspace_decomposition.md``):

1. **Lens quality varies by layer** — the pilot lens's readout fidelity is
   strong only at layers 28–38; a mid-stack "jump" may reflect lens error
   shrinking, not representation change.
2. **Tokenization granularity** — multi-token words and non-English tokens
   change what an active set can look like across positions.
3. **Fitting-corpus bias** — the lens was fitted on WikiText plain text;
   chat-formatted activations show template-token artifacts (pilot report,
   qualitative section).
4. **Ordinary late-layer vocabulary alignment** — near the unembedding
   every reasonable readout (including the plain logit lens) converges to
   the output distribution; convergence at layers 35–38 is expected and is
   not ignition evidence.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from jlens.cones import TRANSITION_SCHEMA

IGNITION_RECORD_SCHEMA = "jlens.ignition.candidate.v1"

#: Exploratory composite weights — exposed, arbitrary, and NOT validated.
DEFAULT_HEURISTIC_WEIGHTS: dict[str, float] = {
    "delta_explained_fraction": 1.0,
    "active_set_jaccard": 1.0,
    "top1_share_to": 0.5,
    "output_alignment_share": 0.5,
}


def candidate_ignition_signals(
    transitions: Sequence[dict],
    records_by_layer: dict[int, dict],
    *,
    model_top1_id: int | None = None,
    persistence_jaccard_threshold: float = 0.5,
) -> list[dict]:
    """Per-transition candidate signals for one (prompt, position)
    trajectory.

    Args:
        transitions: Output of :func:`jlens.cones.cone_trajectory` for one
            prompt/position (already layer-ordered).
        records_by_layer: The trajectory's cone records keyed by layer
            (used for output-alignment coefficient shares).
        model_top1_id: The model's actual top-1 *output* token id at this
            position, if known; enables the output-alignment signal.
        persistence_jaccard_threshold: Jaccard level treated as "stable"
            when measuring persistence run-lengths.

    Returns:
        One ``jlens.ignition.candidate.v1`` dict per transition. Every
        signal stays separately visible; no composite is added here.
    """
    for transition in transitions:
        if transition.get("schema") != TRANSITION_SCHEMA:
            raise ValueError(
                f"expected {TRANSITION_SCHEMA} transitions, got "
                f"{transition.get('schema')!r}"
            )
    signals: list[dict] = []
    jaccards = [t["active_set_overlap"]["jaccard"] for t in transitions]
    for index, transition in enumerate(transitions):
        # Persistence: consecutive transitions (from this one onward) whose
        # active sets stay at/above the stability threshold.
        run_length = 0
        for jaccard in jaccards[index:]:
            if jaccard >= persistence_jaccard_threshold:
                run_length += 1
            else:
                break

        alignment: dict | None = None
        if model_top1_id is not None:
            record_to = records_by_layer[transition["layer_to"]]
            ids = record_to["effective_token_ids"]
            coeffs = record_to["effective_coefficients"]
            total = sum(coeffs)
            share = 0.0
            if int(model_top1_id) in ids and total > 0:
                share = coeffs[ids.index(int(model_top1_id))] / total
            alignment = {
                "model_top1_id": int(model_top1_id),
                "in_active_set": int(model_top1_id) in ids,
                "coefficient_share": share,
            }

        signals.append(
            {
                "schema": IGNITION_RECORD_SCHEMA,
                "label": "candidate ignition diagnostics — NOT validated ignition",
                "prompt_hash": transition["prompt_hash"],
                "prompt_slug": transition.get("prompt_slug"),
                "format": transition["format"],
                "position": transition["position"],
                "layer_from": transition["layer_from"],
                "layer_to": transition["layer_to"],
                "signals": {
                    "delta_explained_fraction": transition[
                        "delta_explained_fraction"
                    ],
                    "explained_fraction_to": transition["explained_fraction_to"],
                    "active_set_jaccard": transition["active_set_overlap"][
                        "jaccard"
                    ],
                    "weighted_similarity": transition["weighted_similarity"],
                    "delta_herfindahl": (
                        transition["concentration_to"]["herfindahl"]
                        - transition["concentration_from"]["herfindahl"]
                    ),
                    "top1_share_to": transition["concentration_to"]["top1_share"],
                    "persistence_length_from_here": run_length,
                    "output_alignment_to": alignment,
                },
                "entered_labels": transition["entered_labels"],
                "exited_labels": transition["exited_labels"],
                "caveats": (
                    "Weigh against: layer-dependent lens quality, "
                    "tokenization effects, WikiText fitting-corpus bias, and "
                    "ordinary late-layer vocabulary alignment."
                ),
            }
        )
    return signals


def heuristic_candidate_score(
    signal_record: dict,
    *,
    weights: dict[str, float] | None = None,
    enabled: bool = False,
) -> dict | None:
    """Optional exploratory composite over one transition's signals.

    Disabled by default (returns ``None``); when enabled, returns
    ``{"score", "weights", "components", "label"}`` with every component and
    weight exposed. The score is an unvalidated ranking heuristic for
    deciding what to look at first — never a measurement of ignition.
    """
    if not enabled:
        return None
    weights = dict(weights if weights is not None else DEFAULT_HEURISTIC_WEIGHTS)
    signals = signal_record["signals"]
    components: dict[str, float] = {}
    for name, weight in weights.items():
        if name == "output_alignment_share":
            alignment = signals.get("output_alignment_to")
            value = alignment["coefficient_share"] if alignment else 0.0
        else:
            value = signals.get(name)
            if value is None:
                raise KeyError(f"unknown signal {name!r} in weights")
        components[name] = float(value) * weight
    return {
        "label": "HEURISTIC exploratory composite — not a validated ignition score",
        "score": sum(components.values()),
        "weights": weights,
        "components": components,
    }


def export_transition_records(signal_records: Sequence[dict], path: str) -> None:
    """Write candidate-ignition records as pretty JSON (schema-checked),
    suitable for a future UI to consume."""
    for record in signal_records:
        if record.get("schema") != IGNITION_RECORD_SCHEMA:
            raise ValueError(
                f"not a candidate-ignition record: {record.get('schema')!r}"
            )
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(list(signal_records), handle, indent=2, ensure_ascii=False)


def load_transition_records(path: str) -> list[dict]:
    """Load records written by :func:`export_transition_records`."""
    with open(path, encoding="utf-8") as handle:
        records = json.load(handle)
    for record in records:
        if record.get("schema") != IGNITION_RECORD_SCHEMA:
            raise ValueError(
                f"{path}: unexpected record schema {record.get('schema')!r}"
            )
    return records
