# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""The verbalizer prompt: one constant instruction plus spliced memory slots.

Two properties matter and are enforced here rather than assumed:

1. **The instruction is constant and carries no concept clue.** One string per
   prompt id, identical for every cone, screened by
   :func:`jlens.generative.assert_clean_prompt` against the priming vocabulary
   (``internal`` / ``representation`` / ``concept`` / ``label`` / ``value``)
   that produced this repository's earlier confounded "Internal Concept"
   decodes. All semantic content must come from ``q``.

2. **The memory occupies a known, exact token span.** The prompt is rendered
   through the tokenizer's own chat template with a literal sentinel at the
   start of the user content, split on that sentinel, and the two halves
   tokenized separately; ``n_memory_tokens`` filler ids are spliced between
   them. The span ``[memory_start, memory_end)`` is therefore exact by
   construction, not located by a search that could match the wrong tokens.

The brief's literal example wording is available as
``verbalizer-brief-literal`` but is **not** a default: it contains two of the
forbidden terms, and this repository has already observed an instruction-tuned
Gemma echoing exactly that vocabulary back. Selecting it is an explicit,
recorded choice.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from jlens.autoencoder.errors import AutoencoderError
from jlens.generative import (
    assert_clean_prompt,
    chat_control_token_ids,
    token_strings,
)

#: Literal marker placed at the start of the user content and replaced by the
#: memory slots. Chosen so no natural prompt text contains it.
VERBALIZER_MEMORY_SENTINEL = "@@JMEM@@"

#: The constant instructions. The task is supplied by the wording; the *content*
#: must come only from the injected memory.
VERBALIZER_INSTRUCTIONS: dict[str, str] = {
    "verbalizer-default": (
        "The memory above stands for one specific thing. "
        "Reply with only its shortest specific name."
    ),
    "verbalizer-paraphrase-a": (
        "Given the memory above, reply with only the shortest specific name "
        "for what it stands for."
    ),
    "verbalizer-paraphrase-b": (
        "Answer with only the shortest specific name for the thing the memory "
        "above stands for."
    ),
}

#: The brief's literal example. Retained verbatim for reproducibility, kept out
#: of :data:`VERBALIZER_INSTRUCTIONS` because it reintroduces known priming
#: vocabulary; :func:`resolve_instruction` accepts it and records the choice.
PRIMING_RISK_INSTRUCTIONS: dict[str, str] = {
    "verbalizer-brief-literal": (
        "Identify the concept represented by the supplied internal memory. "
        "Reply with only its shortest specific name."
    ),
}

DEFAULT_PROMPT_ID = "verbalizer-default"

#: Gemma's assistant end-of-turn token id, as fixed by the brief. Resolved from
#: the tokenizer at run time; the constant is the cross-check, not the source.
GEMMA_END_OF_TURN_ID = 106

DEFAULT_MAX_LENGTH = 512

# Import-time guard: a default instruction can never silently acquire priming
# vocabulary. The brief-literal one is deliberately not checked.
for _prompt_id, _text in VERBALIZER_INSTRUCTIONS.items():
    assert_clean_prompt(_text, prompt_id=_prompt_id)
del _prompt_id, _text


def is_default_prompt_id(prompt_id: str) -> bool:
    """Whether ``prompt_id`` names a screened, priming-free instruction."""
    return prompt_id in VERBALIZER_INSTRUCTIONS


def resolve_instruction(prompt_id: str) -> str:
    """Instruction text for ``prompt_id``. Unknown ids raise rather than
    falling back to the default — a typo must not silently change the prompt
    every record in a run was produced under."""
    if prompt_id in VERBALIZER_INSTRUCTIONS:
        return VERBALIZER_INSTRUCTIONS[prompt_id]
    if prompt_id in PRIMING_RISK_INSTRUCTIONS:
        return PRIMING_RISK_INSTRUCTIONS[prompt_id]
    raise AutoencoderError(
        f"unknown verbalizer prompt id {prompt_id!r}; known ids are "
        f"{sorted(VERBALIZER_INSTRUCTIONS)} (default) and "
        f"{sorted(PRIMING_RISK_INSTRUCTIONS)} (priming-risk, opt-in)"
    )


def resolve_end_of_turn_id(tokenizer, *, expected: int | None = GEMMA_END_OF_TURN_ID) -> dict:
    """Resolve the assistant end-of-turn token id from the tokenizer.

    Returns ``{"end_of_turn_id", "eos_token_id", "stop_token_ids",
    "matches_expected"}``. The tokenizer is authoritative; ``expected`` only
    records whether the pinned checkpoint's documented id (106) was observed, so
    a tokenizer change is *visible* rather than silently followed.
    """
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if not callable(convert):
        raise AutoencoderError(
            "tokenizer cannot convert '<end_of_turn>' to an id; generation would "
            "have no turn-terminating stop token"
        )
    unk = getattr(tokenizer, "unk_token_id", None)
    end_of_turn = convert("<end_of_turn>")
    if not isinstance(end_of_turn, int) or end_of_turn < 0 or end_of_turn == unk:
        raise AutoencoderError(
            f"tokenizer does not know '<end_of_turn>' (got {end_of_turn!r}); "
            f"generation would never stop on a turn boundary"
        )
    eos = getattr(tokenizer, "eos_token_id", None)
    stop = [int(end_of_turn)]
    if isinstance(eos, int) and int(eos) != int(end_of_turn):
        stop.append(int(eos))
    return {
        "end_of_turn_id": int(end_of_turn),
        "eos_token_id": int(eos) if isinstance(eos, int) else None,
        "stop_token_ids": stop,
        "matches_expected": None if expected is None else int(end_of_turn) == int(expected),
        "expected_end_of_turn_id": expected,
    }


def _tokenize(tokenizer, text: str, *, max_length: int) -> list[int]:
    encoded = tokenizer(
        text,
        return_tensors=None,
        truncation=True,
        max_length=int(max_length),
        add_special_tokens=False,
    )
    ids = encoded.input_ids if hasattr(encoded, "input_ids") else encoded["input_ids"]
    if ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return [int(t) for t in ids]


def _filler_token_id(tokenizer) -> int:
    """Id placed in the memory slots. Its embedding is *overwritten*, never
    added to, so the choice is cosmetic — but it is recorded so a stored prompt
    is reproducible byte for byte."""
    for attribute in ("pad_token_id", "bos_token_id", "eos_token_id"):
        value = getattr(tokenizer, attribute, None)
        if isinstance(value, int) and value >= 0:
            return int(value)
    raise AutoencoderError("tokenizer has no pad/bos/eos id to use as a memory filler")


@dataclass(frozen=True)
class VerbalizerPrompt:
    """One fully described verbalizer prompt with an exact memory span."""

    prompt_id: str
    instruction: str
    rendered_prompt: str
    left_text: str
    right_text: str
    token_ids: tuple[int, ...]
    token_strings: tuple[str, ...]
    memory_start: int
    memory_end: int
    filler_token_id: int
    structure: dict

    @property
    def prompt_len(self) -> int:
        return len(self.token_ids)

    @property
    def n_memory_tokens(self) -> int:
        return self.memory_end - self.memory_start

    def input_ids(self, *, batch: int = 1, device=None) -> torch.Tensor:
        """``[batch, prompt_len]`` ids (the same prompt repeated per row)."""
        row = torch.tensor(list(self.token_ids), dtype=torch.long, device=device)
        return row.unsqueeze(0).expand(int(batch), -1).contiguous()

    def to_debug_dict(self) -> dict:
        return {
            "prompt_id": self.prompt_id,
            "instruction": self.instruction,
            "is_default_prompt": is_default_prompt_id(self.prompt_id),
            "rendered_prompt": self.rendered_prompt,
            "prompt_token_ids": list(self.token_ids),
            "prompt_tokens": list(self.token_strings),
            "prompt_len": self.prompt_len,
            "memory_span": [self.memory_start, self.memory_end],
            "n_memory_tokens": self.n_memory_tokens,
            "filler_token_id": self.filler_token_id,
            "structure": dict(self.structure),
        }


def build_verbalizer_prompt(
    tokenizer,
    *,
    n_memory_tokens: int,
    prompt_id: str = DEFAULT_PROMPT_ID,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> VerbalizerPrompt:
    """Render the chat prompt and splice ``n_memory_tokens`` memory slots in.

    The rendered chat string is
    ``<bos><start_of_turn>user\\n@@JMEM@@\\n<INSTRUCTION><end_of_turn>\\n<start_of_turn>model\\n``;
    the sentinel is removed and replaced by the memory slots at the token level.
    """
    if n_memory_tokens < 1:
        raise AutoencoderError(f"n_memory_tokens must be >= 1, got {n_memory_tokens}")
    instruction = resolve_instruction(prompt_id)
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_chat_template):
        raise AutoencoderError(
            "the verbalizer requires a tokenizer with apply_chat_template; the "
            "pinned checkpoint is instruction-tuned and must be prompted as a chat"
        )
    user_content = f"{VERBALIZER_MEMORY_SENTINEL}\n{instruction}"
    rendered = apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    if rendered.count(VERBALIZER_MEMORY_SENTINEL) != 1:
        raise AutoencoderError(
            f"memory sentinel appears {rendered.count(VERBALIZER_MEMORY_SENTINEL)} "
            f"time(s) in the rendered prompt, expected exactly 1: {rendered!r}"
        )
    left_text, right_text = rendered.split(VERBALIZER_MEMORY_SENTINEL)
    left_ids = _tokenize(tokenizer, left_text, max_length=max_length)
    right_ids = _tokenize(tokenizer, right_text, max_length=max_length)
    bos = getattr(tokenizer, "bos_token_id", None)
    if isinstance(bos, int) and (not left_ids or left_ids[0] != int(bos)):
        left_ids = [int(bos), *left_ids]
    filler = _filler_token_id(tokenizer)
    ids = [*left_ids, *([filler] * int(n_memory_tokens)), *right_ids]
    if not ids:
        raise AutoencoderError(f"verbalizer prompt {prompt_id!r} tokenized to nothing")

    structure = _check_structure(tokenizer, rendered, ids, right_text)
    return VerbalizerPrompt(
        prompt_id=prompt_id,
        instruction=instruction,
        rendered_prompt=rendered,
        left_text=left_text,
        right_text=right_text,
        token_ids=tuple(ids),
        token_strings=tuple(token_strings(tokenizer, ids)),
        memory_start=len(left_ids),
        memory_end=len(left_ids) + int(n_memory_tokens),
        filler_token_id=filler,
        structure=structure,
    )


def _check_structure(tokenizer, rendered: str, ids: list[int], right_text: str) -> dict:
    """The chat rendering must have exactly the shape generation assumes."""
    problems: list[str] = []
    bos = getattr(tokenizer, "bos_token_id", None)
    n_bos = None
    if isinstance(bos, int):
        n_bos = sum(1 for t in ids if int(t) == int(bos))
        if n_bos != 1:
            problems.append(f"expected exactly 1 BOS token, found {n_bos}")
        elif int(ids[0]) != int(bos):
            problems.append("BOS is present but not at position 0")
    n_start = rendered.count("<start_of_turn>")
    n_end = rendered.count("<end_of_turn>")
    markers_checked = bool(n_start or n_end)
    if markers_checked:
        if n_start != 2:
            problems.append(
                f"expected 2 <start_of_turn> markers (user turn + generation "
                f"prefix), found {n_start}"
            )
        if n_end != 1:
            problems.append(f"expected 1 <end_of_turn> marker, found {n_end}")
        if "<start_of_turn>model" not in right_text:
            problems.append(
                "the model-generation prefix is not after the memory slots; the "
                "memory would not be attended by the answer"
            )
    if not right_text.strip():
        problems.append("nothing follows the memory slots (no instruction, no prefix)")
    if problems:
        raise AutoencoderError(
            "verbalizer prompt is malformed:\n  "
            + "\n  ".join(problems)
            + f"\nrendered prompt: {rendered!r}"
        )
    return {
        "n_bos_tokens": n_bos,
        "n_start_of_turn_markers": n_start,
        "n_end_of_turn_markers": n_end,
        "turn_markers_checked": markers_checked,
    }


def phrase_target_ids(
    tokenizer,
    prompt: VerbalizerPrompt,
    phrase: str,
    *,
    end_of_turn_id: int,
    include_end_of_turn: bool = True,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> dict:
    """Teacher-forcing target ids for ``phrase`` as the assistant continuation.

    Derived **contextually** — as what appending the phrase adds beyond the
    prompt's own trailing text — so the segmentation is the one the model
    actually sees after ``<start_of_turn>model``, not the phrase tokenized in
    isolation. The memory splice cannot affect this: it happens strictly before
    ``right_text``, and only the seam at the end of ``right_text`` matters.

    The end-of-turn token is appended as the final target (the brief requires it
    in training targets), so the adapter learns to *stop*.
    """
    text = str(phrase)
    if not text.strip():
        raise AutoencoderError("phrase is empty")
    tail = _tokenize(tokenizer, prompt.right_text, max_length=max_length)
    joint = _tokenize(tokenizer, prompt.right_text + text, max_length=max_length)
    if joint[: len(tail)] != tail:
        shared = 0
        for a, b in zip(joint, tail, strict=False):
            if a != b:
                break
            shared += 1
        raise AutoencoderError(
            f"appending phrase {text!r} re-tokenized the prompt tail (agree on "
            f"{shared}/{len(tail)} tokens), so the continuation cannot be sliced "
            f"off safely"
        )
    continuation = joint[len(tail) :]
    if not continuation:
        raise AutoencoderError(f"phrase {text!r} added no tokens after the prompt")
    control = chat_control_token_ids(tokenizer)
    kept = [t for t in continuation if t not in control]
    excluded = [t for t in continuation if t in control]
    if not kept:
        raise AutoencoderError(
            f"phrase {text!r} produced only control tokens {excluded} after the prompt"
        )
    target = list(kept)
    if include_end_of_turn:
        target.append(int(end_of_turn_id))
    return {
        "phrase": text,
        "phrase_token_ids": kept,
        "phrase_token_strings": token_strings(tokenizer, kept),
        "n_phrase_tokens": len(kept),
        "target_token_ids": target,
        "excluded_token_ids": excluded,
        "decoded_phrase": tokenizer.decode(kept, skip_special_tokens=True),
        "includes_end_of_turn": bool(include_end_of_turn),
    }
