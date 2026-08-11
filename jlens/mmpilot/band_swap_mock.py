# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""MOCK mode for the contiguous-band study — the band, not the data.

Two halves, both built on code that already exists.

**The causal half** reuses :mod:`jlens.mmpilot.coordinate_swap_mock` unchanged.
That world already has what a band study needs and nothing else does: an
evidence injection at layer 0, carry blocks, a broadcast at layer 4, and a
*reasoning* layer at 5 that computes the legs answer from whatever identity
coordinates arrive there. So a band that starts before layer 5 can change the
downstream answer by changing the identity, and a band that starts at or after
it cannot — while the answer coordinates themselves are editable right up to
the last layer. That is exactly the ordering Stage 4 measures, and it is a
property of the stack rather than a number written into a fixture.

The suffix bands are :data:`MOCK_BAND_STARTS` over the contiguous window
:data:`MOCK_BAND_WINDOW`. Read the parity note before reading a MOCK identity
number: the world's carry blocks nearly commute with the exchange, so the
*identity* readout at the end of an even-length band lands almost back on the
source. The property answer does not, because it is written once at the
reasoning layer from the coordinates that arrived there. This is why identity
is a diagnostic and the downstream answer is the endpoint — in the MOCK by
construction, in the paper by argument.

**The lens half** reuses :mod:`jlens.calibration.extension_mock`: its real
mock parent run, its real corpus pool, and its real tie-aware row synthesiser,
with band-interior layer behaviours added. The scorer, the gate, the folds,
the splits and the publication path are the real ones; only the rank structure
of the rows is a fixture.

**A green MOCK run proves the pipeline runs. It is evidence about Gemma 4,
about layers 33-39, and about the workspace hypothesis in exactly no respect.**
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from jlens.calibration.extension_mock import ExtensionScenario, LayerBehaviour
from jlens.mmpilot.band_lens import BAND_INTERIOR_LAYERS
from jlens.mmpilot.band_swap import (
    BAND_CONDITIONS,
    CONDITION_ALPHA,
    band_trial_record,
)
from jlens.mmpilot.coordinate_swap_mock import (
    IDENTITY_CANDIDATES,
    PROPERTY_CANDIDATES,
    REASONING_LAYER,
    mock_bases,
)

__all__ = [
    "BAND_MOCK_LAYERS",
    "BAND_MOCK_SCENARIOS",
    "MOCK_BAND_END",
    "MOCK_BAND_STARTS",
    "MOCK_BAND_USABLE_LAYERS",
    "MOCK_BAND_WINDOW",
    "mock_band_grid",
    "run_mock_band_trials",
]

#: The MOCK's contiguous band window. It spans the reasoning layer, so some
#: tested bands begin before the answer is computed and some after.
MOCK_BAND_WINDOW: tuple[int, int] = (2, 7)
MOCK_BAND_END = MOCK_BAND_WINDOW[1]

#: Predeclared suffix-band starts, the MOCK's stand-in for 32/35/38/40. Two
#: begin before :data:`~jlens.mmpilot.coordinate_swap_mock.REASONING_LAYER` and
#: two at or after it, which is what makes the arms separable at all.
MOCK_BAND_STARTS: tuple[int, ...] = (
    MOCK_BAND_WINDOW[0],
    REASONING_LAYER - 1,
    REASONING_LAYER,
    MOCK_BAND_END,
)

#: What the MOCK pretends passed confirmation for the band window — the whole
#: contiguous range, because a band needs every layer in it. The narrower
#: :data:`~jlens.mmpilot.coordinate_swap_mock.MOCK_VALIDATED_LAYERS` is left
#: exactly as it is and is what the "interior lens missing" refusal is tested
#: against.
MOCK_BAND_USABLE_LAYERS: tuple[int, ...] = tuple(range(2, MOCK_BAND_END + 1))


def mock_band_grid() -> tuple[tuple[int, ...], ...]:
    """The MOCK suffix bands, as ordered layer tuples."""
    return tuple(
        tuple(range(start, MOCK_BAND_END + 1)) for start in MOCK_BAND_STARTS
    )


def run_mock_band_trials(
    backend,
    *,
    source: str = "bird",
    target: str = "cat",
    source_answer: str = "two",
    target_answer: str = "four",
    modalities: Sequence[str] = ("text", "image", "spoken_audio"),
    conditions: Sequence[str] = BAND_CONDITIONS,
    n_images: int = 4,
    unrelated: tuple[str, str] = ("dog", "car"),
) -> list[dict]:
    """Every band trial the MOCK stage stores, in the real record shape.

    One record per band x arm x condition x modality x image x readout, built
    by the same :func:`jlens.mmpilot.band_swap.band_trial_record` the real run
    uses, from the same
    :func:`jlens.mmpilot.coordinate_swap.run_swap_condition`. The synthetic
    part is the world; the intervention, the scoring and the record are not.
    """
    from jlens.mmpilot.capability import score_candidate_sequences
    from jlens.mmpilot.coordinate_swap import (
        random_two_direction_basis,
        run_swap_condition,
    )
    from jlens.mmpilot.coordinate_swap_mock import (
        IDENTITY_QUESTION,
        PROPERTY_QUESTION,
        mock_concept_tokens,
    )

    tokens = mock_concept_tokens(backend)
    candidate_ids = {
        "identity": {
            name: backend.encode_candidate(f" {name}") for name in IDENTITY_CANDIDATES
        },
        "property": {
            name: backend.encode_candidate(f" {name}") for name in PROPERTY_CANDIDATES
        },
    }
    answers = {"identity": (source, target), "property": (source_answer, target_answer)}
    questions = {"identity": IDENTITY_QUESTION, "property": PROPERTY_QUESTION}

    records: list[dict] = []
    for band in mock_band_grid():
        banks = {
            "intermediate": mock_bases(
                backend.world,
                layers=band,
                source=tokens[source],
                target=tokens[target],
            ),
            "answer": mock_bases(
                backend.world,
                layers=band,
                source=tokens[source_answer],
                target=tokens[target_answer],
            ),
        }
        banks["unrelated"] = mock_bases(
            backend.world,
            layers=band,
            source=tokens[unrelated[0]],
            target=tokens[unrelated[1]],
        )
        banks["random_intermediate"] = {
            layer: random_two_direction_basis(basis, seed=911 + layer)
            for layer, basis in banks["intermediate"].items()
        }
        banks["random_answer"] = {
            layer: random_two_direction_basis(basis, seed=977 + layer)
            for layer, basis in banks["answer"].items()
        }
        for modality in modalities:
            for index in range(int(n_images)):
                image_id = f"mock-image-{index}"
                for readout, question in questions.items():
                    inputs = backend.build_inputs(
                        prompt=question,
                        modality=str(modality),
                        concept=source,
                        nuisance_key=image_id,
                    )
                    clean = score_candidate_sequences(
                        backend, inputs, candidate_ids[readout]
                    )
                    for condition in conditions:
                        for arm in ("intermediate", "answer"):
                            if condition.startswith("unrelated_"):
                                chosen = banks["unrelated"]
                            elif condition.startswith("random_"):
                                chosen = banks[f"random_{arm}"]
                            else:
                                chosen = banks[arm]
                            result = run_swap_condition(
                                backend,
                                inputs,
                                bases=chosen,
                                alpha=CONDITION_ALPHA[condition],
                                candidate_ids=candidate_ids[readout],
                                target_concept=answers[readout][1],
                                clean_scores=clean,
                                record_coordinates=False,
                            )
                            records.append(
                                band_trial_record(
                                    result,
                                    band=band,
                                    arm=arm,
                                    condition=condition,
                                    modality=str(modality),
                                    source=source,
                                    target=target,
                                    source_answer=answers[readout][0],
                                    target_answer=answers[readout][1],
                                    readout=readout,
                                    group_id=f"mock-group-{index}",
                                    image_id=image_id,
                                    prompt_hash=inputs.prompt_hash,
                                )
                            )
    return records


# ------------------------------------------------------- the lens-fit half

#: The layers a MOCK band-interior fit runs over. The real interior layers, at
#: the mock stack's depth — :class:`~jlens.calibration.mock.MockCalibrationModel`
#: has the real 42-block layout at a tiny width, so no renumbering is needed.
BAND_MOCK_LAYERS: tuple[int, ...] = BAND_INTERIOR_LAYERS


def _passing() -> LayerBehaviour:
    return LayerBehaviour(development=(1, 1, 1), confirmation=1)


def _failing() -> LayerBehaviour:
    return LayerBehaviour(development=(210, 200, 190), confirmation=195)


#: The two commissioned cases. The pipeline must publish a full band in one and
#: report a reduced admissible sub-band in the other — never quietly widen the
#: band to the layers it happens to have.
BAND_MOCK_SCENARIOS: dict[str, ExtensionScenario] = {
    "all_interior_pass": ExtensionScenario(
        key="all_interior_pass",
        label="every interior layer passes fresh confirmation; the full band exists",
        expected_verdict="BAND_INTERIOR_LENS_GO",
        expected_selected_scale=None,
        expected_published_layers=BAND_MOCK_LAYERS,
        note="the case the band study needs; nothing about it is evidence for Gemma",
        layers={layer: _passing() for layer in BAND_MOCK_LAYERS},
    ),
    "one_interior_fails": ExtensionScenario(
        key="one_interior_fails",
        label="L36 fails confirmation; the band is reduced, not widened",
        expected_verdict="BAND_INTERIOR_LENS_NO_GO",
        expected_selected_scale=None,
        expected_published_layers=(33, 34, 37, 39),
        note=(
            "the honest partial result: 32-34 is the largest admissible "
            "contiguous run, and it is chosen from layer geometry with no causal "
            "outcome in existence"
        ),
        layers={
            **{layer: _passing() for layer in BAND_MOCK_LAYERS},
            36: _failing(),
        },
    ),
}


def band_mock_confirmation(
    scenario: ExtensionScenario | str,
    *,
    layers: Sequence[int] = BAND_MOCK_LAYERS,
    scale: int,
    n_prompts: int,
    stage: str = "confirmation",
    scale_index: int = 0,
) -> Mapping[int, dict]:
    """Per-layer verdicts for a MOCK band stage, scored by the real gate."""
    from jlens.calibration.extension import (
        EXTENSION_CONFIRMATION_GATE,
        EXTENSION_GATE,
    )
    from jlens.calibration.extension_mock import mock_extension_rows
    from jlens.calibration.gate import evaluate_calibration_layers

    if isinstance(scenario, str):
        scenario = BAND_MOCK_SCENARIOS[scenario]
    rows = mock_extension_rows(
        scenario,
        stage=stage,
        scale=int(scale),
        scale_index=int(scale_index),
        n_prompts=int(n_prompts),
        layers=layers,
    )
    gate = EXTENSION_CONFIRMATION_GATE if stage == "confirmation" else EXTENSION_GATE
    return evaluate_calibration_layers(
        rows,
        layers=list(layers),
        scale=int(scale),
        stage="confirmation" if stage == "confirmation" else "validation",
        gate=gate,
    )
