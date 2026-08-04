# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Physical decoder layers, normalized depth, and the immutable layer list.

Two numbering systems appear in this work and they are easy to confuse:

* a **physical layer** indexes ``model.language_model.layers`` — the only thing
  a hook, a Jacobian, or an intervention can be attached to;
* a **normalized layer** is a percentage of model depth, which is how
  cross-model comparisons are usually reported.

In this 42-layer Gemma the four layers under test are physical 20, 26, 32 and 38
— normalized 48, 62, 76 and 90. Writing "layer 32" without saying which system
you mean is a bug waiting to be committed, so every artifact this package writes
carries both, and :func:`layer_ref` is the only place the conversion happens.

:data:`LOCALIZATION_LAYERS` is frozen. Adding a layer after seeing a result, or
dropping one that failed, converts a predetermined comparison into a chosen one;
:func:`assert_immutable_layer_set` is what refuses that, and the tuple is bound
into the run fingerprint so a directory built over one layer set can never be
resumed under another.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

#: Decoder depth of ``google/gemma-4-E4B-it``. ``verify_architecture`` holds the
#: loaded checkpoint to this, so a mismatch fails before any layer is captured.
MODEL_N_LAYERS = 42

#: The four physical layers, fixed before any result exists. Layer 38 is the
#: positive reference (it produced ``ROBUSTNESS_GO``); 32 is the main candidate
#: earlier layer; 20 and 26 are earlier controls that failed the v2 validation
#: outright and are carried so the depth series has a floor.
LOCALIZATION_LAYERS: tuple[int, ...] = (20, 26, 32, 38)

#: The layer whose result is already established. If it does not reproduce on
#: the localization subset, the run is inconclusive regardless of what the
#: earlier layers do — there would be nothing to localize.
REFERENCE_LAYER = 38

#: Bound into the run fingerprint. Editing :data:`LOCALIZATION_LAYERS` must
#: invalidate stored results rather than silently extend them.
LAYER_SET_VERSION = "mmlocalize.layers.20-26-32-38.v1"


class ImmutableLayerSetError(RuntimeError):
    """The predetermined layer list was changed.

    Raised rather than warned. The comparison this package reports is "among
    these four layers, which is earliest"; a list edited after a result exists
    answers a different question with the same words.
    """


@dataclass(frozen=True)
class LayerRef:
    """One layer in both numbering systems, plus its role in this study."""

    physical: int
    normalized: int
    n_model_layers: int
    role: str

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        return f"physical L{self.physical} (~normalized {self.normalized})"


def normalized_depth(physical: int, *, n_model_layers: int = MODEL_N_LAYERS) -> int:
    """Percentage of model depth for ``physical``, rounded to an integer.

    ``round(100 * physical / n_model_layers)``. For this 42-layer Gemma that
    maps 20 -> 48, 26 -> 62, 32 -> 76 and 38 -> 90, which are the normalized
    labels the surrounding literature uses for these layers.
    """
    if n_model_layers <= 0:
        raise ValueError(f"n_model_layers must be positive, got {n_model_layers}")
    if not 0 <= physical <= n_model_layers:
        raise ValueError(
            f"physical layer {physical} is outside a {n_model_layers}-layer model"
        )
    return int(round(100.0 * physical / n_model_layers))


def layer_ref(physical: int, *, n_model_layers: int = MODEL_N_LAYERS) -> LayerRef:
    """A :class:`LayerRef` for ``physical``, with its role in this study."""
    if physical == REFERENCE_LAYER:
        role = "positive reference (validated; produced ROBUSTNESS_GO)"
    elif physical in LOCALIZATION_LAYERS:
        role = "earlier layer under test (eligibility must be earned)"
    else:
        role = "outside the predetermined layer set"
    return LayerRef(
        physical=int(physical),
        normalized=normalized_depth(physical, n_model_layers=n_model_layers),
        n_model_layers=int(n_model_layers),
        role=role,
    )


def earlier_layers(layers: Sequence[int] = LOCALIZATION_LAYERS) -> list[int]:
    """Every layer strictly earlier than the reference, in depth order."""
    return sorted(int(layer) for layer in layers if int(layer) < REFERENCE_LAYER)


def assert_immutable_layer_set(layers: Sequence[int]) -> tuple[int, ...]:
    """Return ``layers`` as a tuple, or refuse if it is not the fixed set.

    Order matters as well as membership: the report reads the series shallow to
    deep, and "earliest layer with evidence" is only meaningful along that
    ordering.

    Raises:
        ImmutableLayerSetError: On any addition, removal, or reordering.
    """
    requested = tuple(int(layer) for layer in layers)
    if requested != LOCALIZATION_LAYERS:
        added = sorted(set(requested) - set(LOCALIZATION_LAYERS))
        removed = sorted(set(LOCALIZATION_LAYERS) - set(requested))
        raise ImmutableLayerSetError(
            f"the layer set is fixed at {list(LOCALIZATION_LAYERS)} and was given "
            f"{list(requested)} (added {added}, removed {removed}). Layers are not "
            "added or dropped after results exist — that turns a predetermined "
            "comparison into a chosen one."
        )
    return requested


def describe_layers(
    layers: Sequence[int] = LOCALIZATION_LAYERS,
    *,
    n_model_layers: int = MODEL_N_LAYERS,
) -> str:
    """The block the notebook prints before anything is computed."""
    lines = [
        f"PREDETERMINED LAYER SET — {LAYER_SET_VERSION}",
        f"  model depth: {n_model_layers} decoder blocks",
        "",
        "  physical  normalized  role",
    ]
    for layer in layers:
        ref = layer_ref(layer, n_model_layers=n_model_layers)
        lines.append(f"  L{ref.physical:<8d} ~{ref.normalized:<10d} {ref.role}")
    lines += [
        "",
        "  Physical and normalized numbers are never interchanged. Every artifact",
        "  this run writes carries both.",
        "  Layers are not added or removed after a result is seen.",
    ]
    return "\n".join(lines)


def layer_manifest(
    layers: Sequence[int] = LOCALIZATION_LAYERS,
    *,
    n_model_layers: int = MODEL_N_LAYERS,
) -> dict:
    """Serialisable record of the layer set, for the run fingerprint."""
    return {
        "version": LAYER_SET_VERSION,
        "n_model_layers": int(n_model_layers),
        "physical_layers": [int(layer) for layer in layers],
        "reference_layer": REFERENCE_LAYER,
        "earlier_layers": earlier_layers(layers),
        "layers": [
            layer_ref(layer, n_model_layers=n_model_layers).to_dict() for layer in layers
        ],
        "mapping_rule": "normalized = round(100 * physical / n_model_layers)",
    }


__all__ = [
    "LAYER_SET_VERSION",
    "LOCALIZATION_LAYERS",
    "MODEL_N_LAYERS",
    "REFERENCE_LAYER",
    "ImmutableLayerSetError",
    "LayerRef",
    "assert_immutable_layer_set",
    "describe_layers",
    "earlier_layers",
    "layer_manifest",
    "layer_ref",
    "normalized_depth",
]
