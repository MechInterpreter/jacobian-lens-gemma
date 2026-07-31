# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""J-space language autoencoder: can a sparse J-lens cone be *verbalized* by a
frozen Gemma and *geometrically reconstructed* from that verbalization?

The cycle under test (see ``docs/jspace_language_autoencoder.md``)::

    q  ──cone adapter──►  memory  ──frozen Gemma──►  phrase
    ▲                                                  │
    └────────  frozen phrase reconstructor  ◄──────────┘
                          q_hat

Only two modules are ever trained: :class:`~jlens.autoencoder.adapter.ConeAdapter`
and :class:`~jlens.autoencoder.reconstructor.PhraseReconstructor`. Gemma, the
tokenizer, the fitted lens, the J-space dictionary, and the pursuit are frozen
inputs — the package asserts this rather than assuming it.

This is a **feasibility study**. Every entry point either produces a measured
number with provenance or raises; nothing returns a placeholder, and the
GO/NO-GO report has no code path that suppresses a negative verdict.

Separate from, and non-interfering with, the completed generative-validation
experiment in :mod:`jlens.generative`.
"""

from jlens.autoencoder.config import (
    AdapterConfig,
    AutoencoderConfig,
    DatasetConfig,
    EvaluationConfig,
    PreferenceConfig,
    ReconstructorConfig,
    load_autoencoder_config,
)
from jlens.autoencoder.errors import AutoencoderError

__all__ = [
    "AdapterConfig",
    "AutoencoderConfig",
    "AutoencoderError",
    "DatasetConfig",
    "EvaluationConfig",
    "PreferenceConfig",
    "ReconstructorConfig",
    "load_autoencoder_config",
]
