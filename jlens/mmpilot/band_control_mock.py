# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""MOCK mode for the corrected wrong-layer control.

Three commissioned cases, chosen so a green run demonstrates the pipeline
*distinguishing* outcomes rather than saying yes to everything:

============================  ================================================
``all_nine_pass``             every physical layer 32-40 clears the frozen gate
                              on the one new population. Full-band GO, and the
                              causal stage is unblocked.
``one_interior_fails``        L36 fails. There is no full band; the largest
                              contiguous passing sub-band is reported from
                              layer geometry alone.
``previously_confirmed_fails``L35 — a layer the *earlier* run confirmed —
                              fails under the common population. There is still
                              no full band. This is the case that matters most:
                              the corrected protocol re-evaluates all nine
                              layers together, so a layer's old verdict cannot
                              carry it.
============================  ================================================

The rows are fixtures with controlled rank structure; the scorer
(:func:`jlens.mmlocalize.lens_validity.tie_aware_row`), the gate
(:data:`jlens.calibration.extension.EXTENSION_CONFIRMATION_GATE`), the folds,
the manifest binding and the verdict are the real ones.

**A green MOCK run proves pipeline behaviour and nothing else.** It is not
evidence about Gemma 4, about layers 32-40, about the corrected control's
outcome on real data, or about the workspace hypothesis.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from jlens.calibration.extension_mock import ExtensionScenario, LayerBehaviour
from jlens.mmpilot.band_control import (
    BAND_SCORING_LAYERS,
    CORRECTED_BAND_GO,
    CORRECTED_BAND_NO_GO,
    CORRECTED_GATE,
    CORRECTED_SCALE,
    FIXED_CONTROL_UNIVERSE,
    WRONG_LAYER_MAPPING,
    ControlUniverse,
    LensSnapshot,
    build_control_universe,
    evaluate_corrected_layers,
)

__all__ = [
    "CORRECTED_MOCK_LAYERS",
    "CORRECTED_MOCK_MANIFEST_CHECKSUM",
    "CORRECTED_MOCK_SCENARIOS",
    "mock_control_universe",
    "mock_corrected_confirmation",
    "mock_lens_snapshots",
]

#: The MOCK scores the real nine.
CORRECTED_MOCK_LAYERS: tuple[int, ...] = BAND_SCORING_LAYERS

#: One synthetic manifest checksum, shared by every layer in a scenario —
#: which is the point: nine layers, one population, one manifest.
CORRECTED_MOCK_MANIFEST_CHECKSUM = "sha256:mock-corrected-confirmation-manifest"


def _passing() -> LayerBehaviour:
    return LayerBehaviour(development=(1, 1, 1), confirmation=1)


def _failing() -> LayerBehaviour:
    return LayerBehaviour(development=(210, 200, 190), confirmation=195)


CORRECTED_MOCK_SCENARIOS: dict[str, ExtensionScenario] = {
    "all_nine_pass": ExtensionScenario(
        key="all_nine_pass",
        label="every physical layer 32-40 passes the corrected confirmation",
        expected_verdict=CORRECTED_BAND_GO,
        expected_selected_scale=None,
        expected_published_layers=CORRECTED_MOCK_LAYERS,
        note=(
            "the only case in which the full band is admissible and stage 3 is "
            "unblocked; nothing about it is evidence for Gemma"
        ),
        layers={layer: _passing() for layer in CORRECTED_MOCK_LAYERS},
    ),
    "one_interior_fails": ExtensionScenario(
        key="one_interior_fails",
        label="L36 fails; the band is reduced from geometry, never widened",
        expected_verdict=CORRECTED_BAND_NO_GO,
        expected_selected_scale=None,
        expected_published_layers=(32, 33, 34, 35, 37, 38, 39, 40),
        note=(
            "32-35 and 37-40 are both four layers long, so the frozen tie rule "
            "takes the shallowest start; no causal outcome is consulted and none "
            "exists at this stage"
        ),
        layers={
            **{layer: _passing() for layer in CORRECTED_MOCK_LAYERS},
            36: _failing(),
        },
    ),
    "previously_confirmed_fails": ExtensionScenario(
        key="previously_confirmed_fails",
        label="L35, confirmed by the earlier run, fails on the common population",
        expected_verdict=CORRECTED_BAND_NO_GO,
        expected_selected_scale=None,
        expected_published_layers=(32, 33, 34, 36, 37, 38, 39, 40),
        note=(
            "the case the corrected protocol exists for: an old confirmation "
            "verdict does not carry a layer through a new population, so a "
            "previously confirmed layer can fail and the full band is refused"
        ),
        layers={
            **{layer: _passing() for layer in CORRECTED_MOCK_LAYERS},
            35: _failing(),
        },
    ),
}


def mock_corrected_confirmation(
    scenario: ExtensionScenario | str,
    *,
    layers: Sequence[int] = CORRECTED_MOCK_LAYERS,
    scale: int = CORRECTED_SCALE,
    n_prompts: int | None = None,
    stage: str = "confirmation",
    manifest_checksum: str = CORRECTED_MOCK_MANIFEST_CHECKSUM,
    scale_index: int = 0,
) -> Mapping[int, dict]:
    """Per-layer verdicts for a MOCK corrected stage, scored by the real gate."""
    from jlens.calibration.extension_mock import mock_extension_rows

    if isinstance(scenario, str):
        scenario = CORRECTED_MOCK_SCENARIOS[scenario]
    size = int(CORRECTED_GATE.n_prompts if n_prompts is None else n_prompts)
    rows = mock_extension_rows(
        scenario,
        stage=stage,
        scale=int(scale),
        scale_index=int(scale_index),
        n_prompts=size,
        layers=layers,
    )
    return evaluate_corrected_layers(
        rows,
        manifest_checksum=str(manifest_checksum),
        layers=list(layers),
        scale=int(scale),
        stage="confirmation" if stage == "confirmation" else "validation",
        gate=CORRECTED_GATE,
    )


def mock_lens_snapshots(
    *,
    scale: int = CORRECTED_SCALE,
    extension_layers: Sequence[int] = (8, 14, 20, 26, 32, 35, 38, 40),
    interior_layers: Sequence[int] = (33, 34, 36, 37, 39),
) -> list[LensSnapshot]:
    """Two synthetic snapshots that agree on every equivalence clause.

    The real pair are the extension's broad scale-250 snapshot and the
    band-interior run's scale-250 snapshot. These carry the same fields with
    obviously synthetic values, so :func:`build_control_universe` runs its real
    refusals against a real shape.
    """
    common = {
        "scale": int(scale),
        "model_repo_id": "google/gemma-4-E4B-it",
        "model_revision": "mock-revision",
        "target_layer": 41,
        "hook_site": "block_output",
        "residual_convention": "pre-final-norm residual, the exact input to block l+1",
        "d_model": 2560,
        "corpus_id": "mock/train",
        "corpus_revision": "mock-corpus-revision",
        "fit_prefix_checksum": "sha256:mock-fit-prefix",
        "estimator": "jlens.fitting.fit (upstream, unmodified)",
        "capture_geometry": {
            "target_layer": 41,
            "d_model": 2560,
            "dim_batch": 8,
            "max_seq_len": 128,
            "skip_first": 16,
            "n_layers": 42,
        },
        "hook_site_source": "mock fixture",
    }
    return [
        LensSnapshot(
            name="extension_scale250_snapshot",
            path="<mock>/lens.scale250.pt",
            file_checksum="sha256:mock-extension-snapshot",
            layers=tuple(int(layer) for layer in extension_layers),
            matrix_checksums={
                int(layer): f"sha256:mock-matrix-L{int(layer)}"
                for layer in extension_layers
            },
            **common,
        ),
        LensSnapshot(
            name="band_interior_scale250_snapshot",
            path="<mock>/lens.band.scale250.pt",
            file_checksum="sha256:mock-interior-snapshot",
            layers=tuple(int(layer) for layer in interior_layers),
            matrix_checksums={
                int(layer): f"sha256:mock-matrix-L{int(layer)}"
                for layer in interior_layers
            },
            **common,
        ),
    ]


def mock_control_universe(
    *, universe: Sequence[int] = FIXED_CONTROL_UNIVERSE
) -> ControlUniverse:
    """The merged MOCK universe, built by the real merge and its real refusals."""
    built = build_control_universe(mock_lens_snapshots(), universe=universe)
    assert built.mapping == WRONG_LAYER_MAPPING, (
        "the MOCK universe produced a different wrong-layer mapping than the "
        "frozen one; the fixture, not the protocol, is wrong"
    )
    return built
