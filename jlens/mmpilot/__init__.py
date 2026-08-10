# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Minimal multimodal J-space transfer pilot (SpokenCOCO).

This package is deliberately small. It exists so the Colab notebook
``notebooks/archive/completed_studies/multimodal_jspace_spokencoco_pilot_colab.ipynb``
can be tested on CPU without a GPU, a model download, or Google Drive. Everything model-facing
goes through a narrow protocol (:mod:`jlens.mmpilot.backend`) with two
implementations: the real Gemma 4 E4B one built in the notebook, and the
deterministic synthetic one in :mod:`jlens.mmpilot.mock`.

Scope: does a **frozen, text-calibrated** J-lens expose J-space coordinates
that transfer across ``text`` / ``image`` / ``spoken_audio`` evidence in the
same model? Nothing here decodes activations to language, learns a cross-modal
mapping, or trains anything.

SpokenCOCO carries written captions, their spoken readings, and the COCO
image. So the pilot tests transfer among *visual*, *written-linguistic*, and
*spoken-linguistic* evidence — **not** environmental audio.
"""

from jlens.mmpilot.store import IncompatibleStateError, RunFingerprint, UnitStore

MODALITIES = ("text", "image", "spoken_audio")

__all__ = ["MODALITIES", "IncompatibleStateError", "RunFingerprint", "UnitStore"]
