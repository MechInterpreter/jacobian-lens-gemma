# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Reading and writing through frozen Gemma with a conditioned memory.

Three entry points, all sharing the same rules:

* logits always come from ``model.logits_from_ids`` — the model's own head, so
  the final norm and Gemma's logit softcap run exactly once, by the library.
  Never ``unembed(forward(ids).last_hidden_state)``, which norms twice.
* generation is **uncached**: every step is a full forward pass. The KV cache is
  a speed optimization that would need its own equivalence proof against this
  path before any result could rest on it.
* the beam batch is evaluated in one forward pass per step, at equal length,
  with finished beams frozen rather than extended.

:func:`sequence_logprobs` is differentiable — it is how gradients reach the
adapter — while :func:`beam_search` is not, by construction.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from jlens.autoencoder.conditioning import ConditioningBackend
from jlens.autoencoder.errors import AutoencoderError
from jlens.autoencoder.prompting import VerbalizerPrompt


def _device_of(model) -> torch.device:
    embed = getattr(model, "_embed_tokens", None)
    if embed is None:
        raise AutoencoderError("model does not expose _embed_tokens")
    return embed.weight.device


def assert_no_gemma_gradients(model, *, where: str = "after backward") -> dict:
    """Fail if any Gemma parameter accumulated a gradient.

    ``requires_grad=False`` should make this impossible, which is exactly why it
    is worth asserting: if it ever fires, something re-enabled grads, and every
    "Gemma is frozen" claim in the run would be false.
    """
    hf_model = getattr(model, "_hf_model", model)
    with_grad = [n for n, p in hf_model.named_parameters() if p.grad is not None]
    if with_grad:
        raise AutoencoderError(
            f"{where}: {len(with_grad)} Gemma parameter(s) hold gradients "
            f"(first: {with_grad[0]}); the model must remain frozen"
        )
    return {"checked": where, "gemma_parameters_with_grad": 0}


def _padded_sequences(
    sequences: Sequence[Sequence[int]], *, pad_token_id: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = [len(s) for s in sequences]
    if not lengths or min(lengths) < 1:
        raise AutoencoderError("every scored sequence must have at least one token")
    width = max(lengths)
    ids = torch.full((len(sequences), width), int(pad_token_id), dtype=torch.long, device=device)
    mask = torch.zeros(len(sequences), width, dtype=torch.bool, device=device)
    for row, sequence in enumerate(sequences):
        ids[row, : len(sequence)] = torch.tensor(
            [int(t) for t in sequence], dtype=torch.long, device=device
        )
        mask[row, : len(sequence)] = True
    return ids, mask


def sequence_logprobs(
    model,
    prompt: VerbalizerPrompt,
    memory: torch.Tensor,
    sequences: Sequence[Sequence[int]],
    *,
    conditioner: ConditioningBackend,
    pad_token_id: int,
) -> dict:
    """Teacher-forced log-probability of each sequence under the memory.

    **Differentiable** with respect to ``memory`` (and therefore the adapter);
    Gemma contributes no gradients because none of its parameters require them.

    Args:
        memory: ``[M, d]`` (shared by every sequence) or ``[B, M, d]``.
        sequences: token id lists, typically ``phrase ++ <end_of_turn>``.

    Returns ``{"total": [B], "mean": [B], "n_tokens": [B], "per_token": [B, T]}``
    where padded positions contribute exactly zero.

    Padding sits strictly after each sequence's real tokens and attention is
    causal, so a padded position can never influence a scored one — which is why
    a single ``attention_mask`` of ones is correct here despite the padding.
    """
    device = _device_of(model)
    batch = len(sequences)
    if batch == 0:
        raise AutoencoderError("sequence_logprobs called with no sequences")
    target_ids, target_mask = _padded_sequences(
        sequences, pad_token_id=pad_token_id, device=device
    )
    width = target_ids.shape[1]
    prompt_ids = prompt.input_ids(batch=batch, device=device)
    ids = torch.cat([prompt_ids, target_ids], dim=1)
    if memory.ndim == 2:
        memory = memory.unsqueeze(0)
    if memory.shape[0] not in (1, batch):
        raise AutoencoderError(
            f"memory batch {memory.shape[0]} matches neither 1 nor the sequence "
            f"batch {batch}"
        )
    span = (prompt.memory_start, prompt.memory_end)
    with conditioner.conditioned(model, memory=memory, span=span):
        # n_last = width + 1 covers the prompt-final position (which predicts the
        # first target token) through the second-to-last target position.
        logits = model.logits_from_ids(ids, n_last=width + 1)[:, :-1, :]
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    gathered = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    per_token = gathered * target_mask
    n_tokens = target_mask.sum(dim=-1)
    total = per_token.sum(dim=-1)
    return {
        "total": total,
        "mean": total / n_tokens.clamp_min(1),
        "n_tokens": n_tokens,
        "per_token": per_token,
    }


@dataclass(frozen=True)
class Candidate:
    """One generated phrase candidate."""

    token_ids: tuple[int, ...]
    text: str
    logprob: float
    mean_logprob: float
    n_tokens: int
    finished: bool
    beam_rank: int

    def to_dict(self) -> dict:
        return {
            "token_ids": list(self.token_ids),
            "text": self.text,
            "sequence_logprob": self.logprob,
            "mean_token_logprob": self.mean_logprob,
            "n_tokens": self.n_tokens,
            "finished": self.finished,
            "beam_rank": self.beam_rank,
        }


@torch.no_grad()
def beam_search(
    model,
    prompt: VerbalizerPrompt,
    memory: torch.Tensor,
    *,
    conditioner: ConditioningBackend,
    beam_width: int,
    max_new_tokens: int,
    stop_token_ids: Sequence[int],
    pad_token_id: int,
) -> list[Candidate]:
    """Uncached batched beam search over the frozen model.

    All beams stay at equal length; a beam that emits a stop token is marked
    finished, its score frozen, and it is padded rather than extended. Its
    trailing pad positions are never scored again, so they cannot change its
    ranking.

    Returns candidates sorted by total sequence log-probability, descending.
    Stop tokens are excluded from the returned ids and from the token count, so
    ``n_tokens`` is the length of the phrase itself.
    """
    if beam_width < 1:
        raise AutoencoderError(f"beam_width must be >= 1, got {beam_width}")
    if max_new_tokens < 1:
        raise AutoencoderError(f"max_new_tokens must be >= 1, got {max_new_tokens}")
    device = _device_of(model)
    stop = {int(t) for t in stop_token_ids}
    if not stop:
        raise AutoencoderError("beam_search needs at least one stop token id")
    if memory.ndim == 2:
        memory = memory.unsqueeze(0)
    if memory.shape[0] != 1:
        raise AutoencoderError(
            "beam_search conditions one cone at a time; pass a [M, d] or [1, M, d] memory"
        )
    prompt_row = prompt.input_ids(batch=1, device=device)
    tokens = torch.zeros((1, 0), dtype=torch.long, device=device)
    scores = torch.zeros(1, dtype=torch.float32, device=device)
    finished = torch.zeros(1, dtype=torch.bool, device=device)
    span = (prompt.memory_start, prompt.memory_end)

    with conditioner.conditioned(model, memory=memory, span=span):
        for _step in range(int(max_new_tokens)):
            n_beams = tokens.shape[0]
            ids = torch.cat([prompt_row.expand(n_beams, -1), tokens], dim=1)
            logits = model.logits_from_ids(ids, n_last=1)[:, -1, :].float()
            log_p = torch.log_softmax(logits, dim=-1)
            if bool(finished.any()):
                # A finished beam has exactly one zero-cost continuation (pad),
                # so its total score is frozen and it cannot be re-ranked by
                # anything it "says" after stopping.
                log_p[finished] = float("-inf")
                log_p[finished, int(pad_token_id)] = 0.0
            total = scores.unsqueeze(1) + log_p
            flat = total.reshape(-1)
            k = min(int(beam_width), int(flat.numel()))
            top_scores, top_indices = flat.topk(k)
            vocab = log_p.shape[-1]
            beam_index = torch.div(top_indices, vocab, rounding_mode="floor")
            token_index = top_indices % vocab
            tokens = torch.cat([tokens[beam_index], token_index.unsqueeze(1)], dim=1)
            scores = top_scores
            newly_stopped = torch.tensor(
                [int(t) in stop for t in token_index.tolist()],
                dtype=torch.bool,
                device=device,
            )
            finished = finished[beam_index] | newly_stopped
            if bool(finished.all()):
                break

    candidates: list[Candidate] = []
    for rank in range(tokens.shape[0]):
        raw = [int(t) for t in tokens[rank].tolist()]
        trimmed: list[int] = []
        is_finished = False
        for token in raw:
            if token in stop:
                is_finished = True
                break
            if token == int(pad_token_id):
                break
            trimmed.append(token)
        text = model.tokenizer.decode(trimmed, skip_special_tokens=True).strip()
        logprob = float(scores[rank])
        candidates.append(
            Candidate(
                token_ids=tuple(trimmed),
                text=text,
                logprob=logprob,
                mean_logprob=logprob / max(1, len(trimmed)),
                n_tokens=len(trimmed),
                finished=bool(is_finished),
                beam_rank=rank,
            )
        )
    return sorted(candidates, key=lambda c: (-c.logprob, c.beam_rank))


@torch.no_grad()
def greedy_phrase(
    model,
    prompt: VerbalizerPrompt,
    memory: torch.Tensor,
    *,
    conditioner: ConditioningBackend,
    max_new_tokens: int,
    stop_token_ids: Sequence[int],
    pad_token_id: int,
) -> Candidate:
    """Beam search with width 1 — the greedy decode, kept as one code path so
    greedy and beam results can never diverge for structural reasons."""
    return beam_search(
        model,
        prompt,
        memory,
        conditioner=conditioner,
        beam_width=1,
        max_new_tokens=max_new_tokens,
        stop_token_ids=stop_token_ids,
        pad_token_id=pad_token_id,
    )[0]
