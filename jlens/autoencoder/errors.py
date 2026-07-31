# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""The single error type the autoencoder raises.

One type on purpose: every failure in this package is "the experiment cannot
proceed honestly from here", and callers (scripts, notebook cells) should treat
them identically — abort and report, never degrade to a placeholder.
"""

from __future__ import annotations


class AutoencoderError(ValueError):
    """Invalid request, failed invariant, or unmet precondition."""
