# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Depth localization of the confirmed text-image causal transfer.

The six-concept robustness study returned ``ROBUSTNESS_GO`` at physical decoder
layer 38 and earned exactly one claim: *a frozen text-calibrated J-lens at late
layer 38 exposes concept representations that causally transfer between written
text and images.* Layer 38 sits at 90% of this model's depth, close enough to
the output that a final-prompt-token edit there cannot distinguish "concept
representation" from "answer-language convergence".

This package asks the one follow-up that result earns: **how early does the
transfer appear?** Four predetermined physical layers (20, 26, 32, 38), two
concepts that already replicated bidirectionally (cat and toilet), off-diagonal
cells only. Layer 38 is the positive reference — if it does not reproduce on the
localization subset, nothing else here means anything.

It is not another robustness study and not a framework. Nothing is added to the
design after a result is seen.

Three separations keep the question honest:

**Eligibility is earned, not assumed.** An earlier layer does not become
testable by being written into a layer list. It must pass an independently
specified text-only lens-validity gate (:mod:`~jlens.mmlocalize.lens_validity`)
before any causal claim may rest on it. A layer that fails keeps its diagnostic
numbers and is skipped causally.

**The targets are frozen before the first layer result exists.**
:mod:`~jlens.mmlocalize.targets` fixes and fingerprints the image set, and the
same images are used at every layer, so a difference between layers is a fact
about depth rather than about which photographs each layer happened to get.

**The earliest tested layer is not the earliest layer.** The rubric in
:mod:`~jlens.mmlocalize.verdict` names this limit in every verdict it produces.
"""

from jlens.mmlocalize.layers import (
    LOCALIZATION_LAYERS,
    MODEL_N_LAYERS,
    REFERENCE_LAYER,
    LayerRef,
    describe_layers,
    layer_ref,
    normalized_depth,
)

__all__ = [
    "LOCALIZATION_LAYERS",
    "MODEL_N_LAYERS",
    "REFERENCE_LAYER",
    "LayerRef",
    "describe_layers",
    "layer_ref",
    "normalized_depth",
]
