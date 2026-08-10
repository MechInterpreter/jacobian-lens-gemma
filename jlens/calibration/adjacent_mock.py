# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Commissioned synthetic outcomes for the L27–L31 adjacent-layer study.

A tiny random transformer has no interesting per-layer structure, so a MOCK run
that scored its real activations would report five indistinguishable layers and
prove nothing about the selection rule. These fixtures instead place the target
token at a **known** rank per layer per stage and hand the rows to the *real*
tie-aware scorer, the *real* gate and the *real* selection rule. The rows are
synthetic; every decision made about them is the production one.

Three scenarios, and the middle one is the point:

``earliest_wins``
    L27 and L28 fail, L29 and L31 pass. The rule must select **29** — the lowest
    passer — and not 31, which has the prettier numbers. This is the scenario
    that would catch a "best-looking layer" implementation.

``none_pass``
    Every candidate fails the untouched confirmation. ``ADJACENT_LENS_NO_GO`` is
    a first-class outcome and the complete table is still recorded.

``development_only``
    Every candidate passes development and then fails confirmation. This is why
    development cannot publish anything.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

from jlens.calibration.adjacent import ADJACENT_CANDIDATE_LAYERS
from jlens.calibration.extension_mock import CONTROL_RANK, _place_at_rank
from jlens.mmlocalize.lens_validity import tie_aware_row

__all__ = [
    "ADJACENT_SCENARIOS",
    "MOCK_TARGET_MODULUS",
    "MOCK_VOCAB",
    "AdjacentLayerBehaviour",
    "AdjacentScenario",
    "mock_adjacent_rows",
]

#: 64 possible targets over 256 prompts clears the gate's 32-token floor with
#: room to spare and keeps the largest single-target share far under 25%.
MOCK_TARGET_MODULUS = 64
MOCK_VOCAB = 512


@dataclass(frozen=True)
class AdjacentLayerBehaviour:
    """A known statistical case for one candidate layer, per stage.

    The knob is the exact optimistic rank the target is placed at: ``1`` is a
    unique argmax, and a tie block of ``t`` gives optimistic rank 1 with midrank
    ``(t + 1) / 2`` — which is how a layer that "looks like rank 1" fails the
    tie-aware gate.
    """

    development: int
    confirmation: int
    development_tie_block: int = 1
    confirmation_tie_block: int = 1


@dataclass(frozen=True)
class AdjacentScenario:
    """One commissioned outcome, with what the selection rule must do."""

    key: str
    label: str
    expected_verdict: str
    expected_selected_layer: int | None
    layers: dict[int, AdjacentLayerBehaviour] = field(default_factory=dict)
    note: str = ""


def _passes() -> AdjacentLayerBehaviour:
    return AdjacentLayerBehaviour(development=1, confirmation=1)


def _fails() -> AdjacentLayerBehaviour:
    return AdjacentLayerBehaviour(development=200, confirmation=195)


ADJACENT_SCENARIOS: dict[str, AdjacentScenario] = {
    "earliest_wins": AdjacentScenario(
        key="earliest_wins",
        label="L29 and L31 pass confirmation; the rule must select 29",
        expected_verdict="ADJACENT_LENS_GO",
        expected_selected_layer=29,
        note=(
            "L31 is given a *better* margin than L29 on purpose. An "
            "implementation that ranked the passers instead of ordering them "
            "would select 31 and would be caught here"
        ),
        layers={
            27: _fails(),
            28: AdjacentLayerBehaviour(
                development=1, confirmation=1,
                development_tie_block=64, confirmation_tie_block=64,
            ),
            29: AdjacentLayerBehaviour(development=1, confirmation=2),
            30: _fails(),
            31: _passes(),
        },
    ),
    "none_pass": AdjacentScenario(
        key="none_pass",
        label="every candidate fails the untouched confirmation",
        expected_verdict="ADJACENT_LENS_NO_GO",
        expected_selected_layer=None,
        note=(
            "the honest negative. The interval is closed, so there is no wider "
            "band to reach for and no 'closest' layer to promote"
        ),
        layers={layer: _fails() for layer in ADJACENT_CANDIDATE_LAYERS},
    ),
    "development_only": AdjacentScenario(
        key="development_only",
        label="every candidate passes development and fails confirmation",
        expected_verdict="ADJACENT_LENS_NO_GO",
        expected_selected_layer=None,
        note=(
            "why development cannot publish. The development set is reused "
            "history; only the untouched set decides"
        ),
        layers={
            layer: AdjacentLayerBehaviour(development=1, confirmation=150)
            for layer in ADJACENT_CANDIDATE_LAYERS
        },
    ),
}


def _scores(
    *,
    layer: int,
    stage: str,
    prompt_index: int,
    variant: str,
    target: int,
    rank: int,
    tie_block: int,
    vocab: int,
    n_distinct_targets: int,
) -> torch.Tensor:
    seed = int(
        hashlib.sha256(
            f"adj|{layer}|{stage}|{prompt_index}|{variant}".encode()
        ).hexdigest()[:8],
        16,
    )
    generator = torch.Generator().manual_seed(seed)
    scores = torch.randn(int(vocab), generator=generator)
    if variant != "j_lens":
        rank, tie_block = CONTROL_RANK[variant], 1
    _place_at_rank(scores, target, rank)
    if tie_block > 1:
        offset = int(n_distinct_targets) + 1
        peak = float(scores.max()) + 1.0
        partners = {
            offset + (prompt_index * 7 + step * 13) % (int(vocab) - offset)
            for step in range(int(tie_block) - 1)
        }
        partners.discard(target)
        for index in list(partners)[: int(tie_block) - 1]:
            scores[index] = peak
        scores[target] = peak
    return scores


def mock_adjacent_rows(
    scenario: AdjacentScenario | str,
    *,
    stage: str,
    n_prompts: int,
    layers: Sequence[int] = ADJACENT_CANDIDATE_LAYERS,
    n_distinct_targets: int = MOCK_TARGET_MODULUS,
    vocab: int = MOCK_VOCAB,
    variants: Sequence[str] = (
        "j_lens",
        "permuted",
        "random",
        "wrong_layer",
        "logit_lens",
    ),
) -> list[dict]:
    """Rows for one stage, scored by the **real** tie-aware scorer.

    Args:
        stage: ``"development"`` or ``"confirmation"``. The stage enters the row
            seed, so a confirmation row is never a development row relabelled.
    """
    if isinstance(scenario, str):
        scenario = ADJACENT_SCENARIOS[scenario]
    if stage not in ("development", "confirmation"):
        raise ValueError(f"unknown stage {stage!r}; expected development/confirmation")

    rows: list[dict] = []
    for layer in layers:
        behaviour = scenario.layers[int(layer)]
        if stage == "development":
            rank, tie_block = behaviour.development, behaviour.development_tie_block
        else:
            rank, tie_block = behaviour.confirmation, behaviour.confirmation_tie_block
        for prompt_index in range(int(n_prompts)):
            target = prompt_index % int(n_distinct_targets)
            actual = torch.full((int(vocab),), -10.0)
            actual[target] = 10.0
            prompt_sha = hashlib.sha256(
                f"adj|{scenario.key}|{stage}|{prompt_index}".encode()
            ).hexdigest()
            for variant in variants:
                rows.append(
                    tie_aware_row(
                        sample_index=prompt_index,
                        prompt_sha=prompt_sha,
                        layer=int(layer),
                        variant=variant,
                        variant_logits=_scores(
                            layer=int(layer),
                            stage=stage,
                            prompt_index=prompt_index,
                            variant=variant,
                            target=target,
                            rank=rank,
                            tie_block=tie_block,
                            vocab=int(vocab),
                            n_distinct_targets=int(n_distinct_targets),
                        ),
                        actual_logits=actual,
                    )
                )
    return rows
