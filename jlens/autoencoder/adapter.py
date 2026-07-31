# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""The cone adapter: ``q`` → continuous memory Gemma can read.

The adapter is the *only* thing that stands between a J-space cone and Gemma's
decoder, and its input signature is the experiment's central leakage guard:
:meth:`ConeAdapter.forward` takes one tensor, ``q``. There is no argument
through which a phrase, token id, atom index, example id, or source prompt could
arrive, so "the adapter cheated" is not a hypothesis that needs testing — it is
structurally unavailable.

Scale: the memory is written into Gemma's input-embedding stream, so it is
calibrated to that stream's measured RMS (Gemma folds a ``sqrt(d_model)`` factor
into the embedding module itself). The calibration constant is measured at
construction time from real token embeddings and stored in the checkpoint, not
assumed.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from jlens.autoencoder.checkpoints import parameter_counts
from jlens.autoencoder.conditioning import (
    ConditioningBackend,
    assert_gemma_frozen,
    assert_no_frozen_parameters_in_optimizer,
)
from jlens.autoencoder.config import AdapterConfig
from jlens.autoencoder.errors import AutoencoderError
from jlens.autoencoder.prompting import VerbalizerPrompt
from jlens.autoencoder.verbalizer import assert_no_gemma_gradients, sequence_logprobs


class ConeAdapter(nn.Module):
    """``q [B, d_model]`` → ``memory [B, M, d_model]``.

    ``q``'s norm is activation magnitude, not meaning, and varies by orders of
    magnitude across occurrences of the same phrase; the input ``LayerNorm``
    removes it while preserving direction. Each output slot is then renormalized
    to a fixed RMS so the memory sits at the same magnitude as a real token
    embedding regardless of what the MLP happens to emit early in training —
    without this, the first optimizer steps either whisper (and Gemma ignores the
    memory) or shout (and Gemma emits noise), and neither failure is about
    J-space.
    """

    def __init__(
        self,
        *,
        d_model: int,
        config: AdapterConfig,
        target_rms: float,
    ) -> None:
        super().__init__()
        if target_rms <= 0:
            raise AutoencoderError(f"target_rms must be > 0, got {target_rms}")
        self.d_model = int(d_model)
        self.n_memory_tokens = int(config.n_memory_tokens)
        self.config = config
        self.register_buffer(
            "target_rms",
            torch.tensor(float(target_rms) * float(config.memory_rms_scale)),
        )
        hidden = int(config.hidden_dim)
        self.input_norm = nn.LayerNorm(self.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(self.d_model, hidden),
            nn.GELU(),
            nn.Dropout(float(config.dropout)),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.to_memory = nn.Linear(hidden, self.n_memory_tokens * self.d_model)
        self.slot_bias = nn.Parameter(torch.zeros(self.n_memory_tokens, self.d_model))
        nn.init.normal_(self.to_memory.weight, std=0.02)
        nn.init.zeros_(self.to_memory.bias)

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """The **only** input is ``q``. See the module docstring."""
        if q.ndim == 1:
            q = q.unsqueeze(0)
        if q.ndim != 2 or q.shape[-1] != self.d_model:
            raise AutoencoderError(
                f"q must be [B, {self.d_model}] or [{self.d_model}], got {tuple(q.shape)}"
            )
        if not bool(torch.isfinite(q).all()):
            raise AutoencoderError("q contains NaN/Inf")
        hidden = self.mlp(self.input_norm(q.float()))
        memory = self.to_memory(hidden).view(-1, self.n_memory_tokens, self.d_model)
        memory = memory + self.slot_bias
        rms = memory.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
        return memory / rms * self.target_rms

    def describe(self) -> dict:
        return {
            "class": type(self).__name__,
            "d_model": self.d_model,
            "n_memory_tokens": self.n_memory_tokens,
            "target_rms": float(self.target_rms),
            "input_signature": "forward(q) — q only, no metadata channel",
            **parameter_counts(self),
        }


def zero_memory_like(adapter: ConeAdapter, *, batch: int = 1) -> torch.Tensor:
    """An all-zero memory of the adapter's shape — the zero-memory baseline."""
    return torch.zeros(int(batch), adapter.n_memory_tokens, adapter.d_model)


def warm_start_targets(
    dataset,
    indices: Sequence[int],
    phrase_targets: dict[str, list[int]],
) -> list[list[int]]:
    """Teacher-forcing targets for ``indices``: phrase ids plus end-of-turn.

    The end-of-turn token is part of the target on purpose (the brief requires
    it): an adapter trained only on phrase tokens learns what to say but not
    when to stop, and every beam then runs to the length cap.
    """
    targets: list[list[int]] = []
    for index in indices:
        phrase = dataset.records[index]["phrase"]
        if phrase not in phrase_targets:
            raise AutoencoderError(f"no teacher-forcing target for phrase {phrase!r}")
        targets.append(list(phrase_targets[phrase]))
    return targets


def train_adapter_warm_start(
    model,
    adapter: ConeAdapter,
    dataset,
    prompt: VerbalizerPrompt,
    *,
    config: AdapterConfig,
    conditioner: ConditioningBackend,
    phrase_targets: dict[str, list[int]],
    pad_token_id: int,
    device: torch.device | str = "cpu",
    split: str = "train",
    log: object = None,
    on_epoch=None,
    start_epoch: int = 0,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[ConeAdapter, dict]:
    """Supervised warm start: teacher-forced phrase cross-entropy through frozen
    Gemma.

    Only ``adapter``'s parameters are optimized; this is asserted against the
    optimizer's actual parameter identities before the first step, and Gemma is
    re-checked for gradients after every backward pass.
    """
    assert_gemma_frozen(model, where="adapter warm start (before)")
    adapter = adapter.to(device)
    adapter.train()
    if optimizer is None:
        optimizer = torch.optim.AdamW(
            adapter.parameters(),
            lr=float(config.learning_rate),
            weight_decay=float(config.weight_decay),
        )
    optimizer_report = assert_no_frozen_parameters_in_optimizer(
        optimizer, frozen_modules=[model], trainable=adapter
    )
    indices = dataset.indices_for_split(split)
    if not indices:
        raise AutoencoderError(f"split {split!r} has no records to warm start on")
    cones = dataset.cones
    generator = torch.Generator().manual_seed(int(config.seed))
    history: list[dict] = []
    for epoch in range(int(start_epoch), int(config.epochs)):
        order = [indices[i] for i in torch.randperm(len(indices), generator=generator).tolist()]
        epoch_loss = 0.0
        epoch_tokens = 0
        n_batches = 0
        for start in range(0, len(order), int(config.batch_size)):
            batch_indices = order[start : start + int(config.batch_size)]
            q = cones[batch_indices].to(device)
            memory = adapter(q)
            targets = warm_start_targets(dataset, batch_indices, phrase_targets)
            scored = sequence_logprobs(
                model,
                prompt,
                memory,
                targets,
                conditioner=conditioner,
                pad_token_id=pad_token_id,
            )
            loss = -(scored["total"] / scored["n_tokens"].clamp_min(1)).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            assert_no_gemma_gradients(model, where="adapter warm start backward")
            if config.grad_clip > 0:
                nn.utils.clip_grad_norm_(adapter.parameters(), float(config.grad_clip))
            optimizer.step()
            epoch_loss += float(loss.detach())
            epoch_tokens += int(scored["n_tokens"].sum())
            n_batches += 1
        metrics = {
            "epoch": epoch,
            "loss": epoch_loss / max(1, n_batches),
            "mean_target_nll": epoch_loss / max(1, n_batches),
            "n_target_tokens": epoch_tokens,
            "n_batches": n_batches,
        }
        history.append(metrics)
        if log is not None:
            log.info("adapter warm start epoch %d loss=%.4f", epoch, metrics["loss"])
        if on_epoch is not None:
            on_epoch(epoch, metrics, optimizer)
    adapter.eval()
    assert_gemma_frozen(model, where="adapter warm start (after)")
    summary = {
        "split": split,
        "n_records": len(indices),
        "history": history,
        "optimizer": optimizer_report,
        "adapter": adapter.describe(),
    }
    return adapter, summary
