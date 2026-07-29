# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""The model interface the lens is typed against.

Any model can be plugged in by implementing these members.
:func:`jlens.hf.from_hf` is the HuggingFace adapter; ``tests/tiny.py`` is a
minimal from-scratch example.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import torch
from torch import nn


class LensModel(Protocol):
    """What the lens needs from a model.

    Attributes:
        n_layers: Number of residual blocks.
        d_model: Residual-stream width.
        layers: The residual blocks, indexable by integer; what
            :class:`~jlens.hooks.ActivationRecorder` hooks.
        tokenizer: Tokenizer used by the visualisation helpers; must provide
            ``decode(token_ids) -> str``. Fitting and :meth:`apply` never
            touch it.
    """

    n_layers: int
    d_model: int
    layers: Sequence[nn.Module]
    tokenizer: Any

    def encode(self, text: str, *, max_length: int = ...) -> torch.Tensor:
        """Tokenize ``text`` to ``input_ids`` of shape ``[1, seq_len]`` on the
        model's input device."""
        ...

    def forward(self, input_ids: torch.Tensor) -> Any:
        """Run the residual stack on ``input_ids`` (no LM head). Must build an
        autograd graph through :attr:`layers` when grad is enabled, and must be
        deterministic across batch elements (eval mode, dropout off) — the
        fitting estimator replicates the prompt along the batch axis.

        Whether the returned ``last_hidden_state`` has the final norm applied
        is implementation-defined (HuggingFace applies it; ``tests/tiny.py``
        does not), so it is **not** a portable residual-stream tensor. Read
        residual-stream activations from hooks on :attr:`layers`, and get
        model predictions from :meth:`logits_from_ids`."""
        ...

    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        """Map a **pre-final-norm** residual-stream tensor ``[..., d_model]``
        to logits ``[..., vocab_size]`` (final norm + LM head).

        The input must be a residual-stream activation, i.e. a block output as
        captured by :class:`~jlens.hooks.ActivationRecorder`. Passing an
        already-normed tensor (such as a HuggingFace
        ``forward(...).last_hidden_state``) applies the final norm twice and
        silently corrupts the logits."""
        ...

    def logits_from_ids(
        self, input_ids: torch.Tensor, *, n_last: int | None = None
    ) -> torch.Tensor:
        """Logits for ``input_ids`` through the model's own output pathway —
        what the model itself would predict, with the final norm, LM head, and
        any logit post-processing applied exactly once. Forward hooks on
        :attr:`layers` must still fire, so activation interventions apply.

        ``n_last`` restricts the result to the final ``n_last`` positions,
        returning ``[batch, n_last, vocab]`` (``None`` gives the full
        ``[batch, seq, vocab]``). Implementations should run the LM head on the
        corresponding hidden-state slice rather than slicing afterwards: that is
        what ``generate()`` does, and matching the head's GEMM shape is what
        makes the two paths numerically equivalent in reduced precision."""
        ...
