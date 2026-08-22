# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Paper-first replication and source-loading localization.

This module is deliberately small.  It does not implement another intervention:
all causal edits still go through :mod:`jlens.mmpilot.coordinate_swap`.  Its job
is to keep the order of evidence honest:

1. reproduce the paper's text-only task with the exact alpha=1 coordinate swap;
2. measure whether the clean residual actually loads on the source J-lens row;
3. choose a contiguous layer band and a prompt-position rule from that clean
   loading measurement only;
4. freeze that choice before a fresh multimodal population is opened.

The functions here are pure apart from :func:`capture_source_loading`, which
performs one no-gradient model forward pass.  They are therefore suitable for
unit tests and for a resumable notebook orchestrator.
"""

from __future__ import annotations

import math
import re
import statistics
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass

import torch
from torch import nn

from jlens.hooks import ActivationRecorder
from jlens.mmpilot.coordinate_swap import (
    MODEL_DTYPE_REALIZATION_VERSION,
    PRIMARY_POSITION_RULE,
    ModelDtypeRealizationPolicy,
    SwapBasis,
    coordinate_swap_band,
    read_coordinates,
    resolve_positions,
    swap_coordinates,
    tensor_checksum,
)
from jlens.mmpilot.store import payload_checksum

PROTOCOL_VERSION = "mmpilot.paper_first_workspace_replication.v7"
TEXT_INPUT_PROTOCOL_VERSION = "mmpilot.assistant_prefill_completion.v1"
TEXT_OUTPUT_ENDPOINT_VERSION = "mmpilot.unrestricted_greedy_semantic_head_answer.v2"
TEXT_MAX_NEW_TOKENS = 2
MULTIMODAL_INPUT_PROTOCOL_VERSION = (
    "mmpilot.multimodal_assistant_prefill_completion.v1"
)
MULTIMODAL_COMPLETION_INSTRUCTION = (
    "Use only the supplied evidence. Complete the assistant's factual sentence "
    "directly with the requested answer. Do not restart, explain, or list choices."
)
MULTIMODAL_MAX_NEW_TOKENS = 4
TEXT_COMPLETION_INSTRUCTION = (
    "Complete the assistant's factual sentence by continuing it directly. "
    "Do not restart or explain the sentence."
)
TEXT_ANSWER_MATCH_RULE = (
    "after NFKC/case/whitespace/punctuation normalization, the expected answer "
    "or its fixed digit/English-number-word equivalent must be the final lexical "
    "item; explicit negation markers anywhere in the completion reject the match"
)
_NUMBER_WORDS = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
    "10": "ten",
}
_NEGATION_MARKERS = frozenset({"not", "no", "non", "never", "except", "outside"})
LOADING_VERSION = "mmpilot.clean_source_loading.v1"
LOCALIZATION_VERSION = "mmpilot.loading_only_localization.v1"
CONFIRMATION_VERSION = "mmpilot.fresh_multimodal_confirmation.v2"
TEXT_DIAGNOSTIC_VERSION = "mmpilot.paper_first_text_swap_diagnostic.v2"
TEXT_DIAGNOSTIC_CONDITIONS = (
    "exact_alpha1",
    "zero",
    "random_alpha1",
    "unrelated_alpha1",
    "direct_answer_norm_matched",
)
TEXT_POST_CAST_MAX_RELATIVE_ERROR = 0.02
TEXT_MODEL_DTYPE_REALIZATION = ModelDtypeRealizationPolicy(
    max_corrections=8,
    relative_coordinate_tolerance=TEXT_POST_CAST_MAX_RELATIVE_ERROR,
    relative_residual_tolerance=TEXT_POST_CAST_MAX_RELATIVE_ERROR,
    minimum_scale=1.0,
)


class WorkspaceReplicationRefused(RuntimeError):
    """The requested claim is not licensed by the recorded evidence."""


@dataclass(frozen=True)
class TextReplicationTask:
    """One predeclared text-only use case from the paper's task family."""

    task_id: str
    family: str
    prompt: str
    source: str
    target: str
    clean_answer: str
    swapped_answer: str
    implicit_intermediate: bool

    def to_dict(self) -> dict:
        return asdict(self)


def anthropic_text_tasks() -> tuple[TextReplicationTask, ...]:
    """Frozen text tasks, defined before any Gemma result is read.

    ``spider_to_ant_legs`` is the downstream-recomputation task.  The country
    rows are the paper's flexible-function family: one concept replacement is
    read through several downstream questions.  They are useful replication
    targets but do not by themselves establish recomputation.
    """

    return (
        TextReplicationTask(
            "spider_to_ant_legs",
            "implicit_two_hop",
            "The number of legs on the animal that spins webs is",
            "spider",
            "ant",
            "8",
            "6",
            True,
        ),
        TextReplicationTask(
            "france_to_china_capital",
            "flexible_function",
            "The capital of France is",
            "France",
            "China",
            "Paris",
            "Beijing",
            False,
        ),
        TextReplicationTask(
            "france_to_china_language",
            "flexible_function",
            "The main language of France is",
            "France",
            "China",
            "French",
            "Chinese",
            False,
        ),
        TextReplicationTask(
            "france_to_china_continent",
            "flexible_function",
            "France is located in",
            "France",
            "China",
            "Europe",
            "Asia",
            False,
        ),
        TextReplicationTask(
            "china_to_france_capital",
            "flexible_function",
            "The capital of China is",
            "China",
            "France",
            "Beijing",
            "Paris",
            False,
        ),
        TextReplicationTask(
            "china_to_france_language",
            "flexible_function",
            "The main language of China is",
            "China",
            "France",
            "Chinese",
            "French",
            False,
        ),
        TextReplicationTask(
            "china_to_france_continent",
            "flexible_function",
            "China is located in",
            "China",
            "France",
            "Asia",
            "Europe",
            False,
        ),
    )


IMPLICIT_TWO_HOP_EXPANSION_VERSION = "mmpilot.implicit_two_hop_expansion.v1"


def implicit_two_hop_expansion_v1() -> tuple[TextReplicationTask, ...]:
    """Additional downstream-recomputation tasks, defined before any Gemma result.

    :func:`anthropic_text_tasks` contains exactly **one** ``implicit_two_hop``
    row, and its own docstring says the country rows "do not by themselves
    establish recomputation".  A downstream-recomputation claim therefore rested
    on ``n = 1``.  These rows raise that denominator.

    Every row obeys the same three rules as ``spider_to_ant_legs``:

    * the source is named **nowhere** in the prompt -- it is reachable only
      through a description, so answering at all requires the intermediate;
    * source and target have *different* values of the queried property, so a
      successful swap is visible in the answer rather than in a tie;
    * both concepts and both answers are single common words, because
      :func:`jlens.mmpilot.coordinate_swap.resolve_concept_token` refuses a
      multi-token concept and the endpoint decodes only
      ``TEXT_MAX_NEW_TOKENS`` tokens.

    Three answer families are used on purpose.  A set made only of leg counts
    could be passed by an intervention that merely perturbs a number, which is
    weaker than recomputation; colour and continent answers cannot be reached
    that way.

    Nothing here is admissible until it survives the clean-capability gate --
    :func:`text_capability_verdict` drops any row Gemma does not already answer
    correctly, and that check must run before this set is frozen.
    """

    return (
        # --- leg counts -------------------------------------------------
        TextReplicationTask(
            "bee_to_spider_legs", "implicit_two_hop",
            "The number of legs on the insect that makes honey is",
            "bee", "spider", "6", "8", True,
        ),
        TextReplicationTask(
            "dog_to_bird_legs", "implicit_two_hop",
            "The number of legs on the animal that barks is",
            "dog", "bird", "4", "2", True,
        ),
        TextReplicationTask(
            "cat_to_spider_legs", "implicit_two_hop",
            "The number of legs on the animal that purrs is",
            "cat", "spider", "4", "8", True,
        ),
        TextReplicationTask(
            "elephant_to_ant_legs", "implicit_two_hop",
            "The number of legs on the animal with a trunk is",
            "elephant", "ant", "4", "6", True,
        ),
        TextReplicationTask(
            "bird_to_cow_legs", "implicit_two_hop",
            "The number of legs on the animal covered in feathers is",
            "bird", "cow", "2", "4", True,
        ),
        TextReplicationTask(
            "cow_to_bee_legs", "implicit_two_hop",
            "The number of legs on the farm animal that produces milk is",
            "cow", "bee", "4", "6", True,
        ),
        TextReplicationTask(
            "horse_to_bird_legs", "implicit_two_hop",
            "The number of legs on the animal that neighs is",
            "horse", "bird", "4", "2", True,
        ),
        # --- colors -----------------------------------------------------
        TextReplicationTask(
            "apple_to_banana_color", "implicit_two_hop",
            "The color of the fruit that keeps the doctor away is",
            "apple", "banana", "red", "yellow", True,
        ),
        TextReplicationTask(
            "banana_to_apple_color", "implicit_two_hop",
            "The color of the fruit that monkeys are said to love is",
            "banana", "apple", "yellow", "red", True,
        ),
        TextReplicationTask(
            "carrot_to_tomato_color", "implicit_two_hop",
            "The color of the vegetable that rabbits are said to love is",
            "carrot", "tomato", "orange", "red", True,
        ),
        TextReplicationTask(
            "grape_to_banana_color", "implicit_two_hop",
            "The color of the fruit that is pressed to make wine is",
            "grape", "banana", "purple", "yellow", True,
        ),
        # --- continents -------------------------------------------------
        TextReplicationTask(
            "zebra_to_panda_continent", "implicit_two_hop",
            "The continent home to the animal with black and white stripes is",
            "zebra", "panda", "Africa", "Asia", True,
        ),
        TextReplicationTask(
            "lion_to_panda_continent", "implicit_two_hop",
            "The continent home to the large cat with a mane is",
            "lion", "panda", "Africa", "Asia", True,
        ),
    )


#: Rows dropped from v1 after the clean-capability gate, with the reason.
#:
#: This is a revision made *after* seeing data, and it is recorded rather than
#: quietly applied.  What it was made on is clean capability -- whether Gemma
#: answers the un-intervened prompt correctly -- which is **not** the causal
#: outcome, so it does not select on the result being tested.  An item the model
#: cannot answer clean cannot test recomputation: there is no first hop to swap.
#:
#: All three failed the same way.  Gemma spends its first token on " typically",
#: and ``TEXT_MAX_NEW_TOKENS`` is 2, so only one content token is ever observed --
#: and that token is independently wrong.  Raising the token budget is not the
#: fix: it is part of the endpoint protocol
#: (``mmpilot.unrestricted_greedy_semantic_head_answer.v2``) and every completed
#: run is comparable only at 2.
CLEAN_CAPABILITY_EXCLUSIONS_V2: dict[str, str] = {
    "cat_to_spider_legs": (
        'generated " typically nine" -- "the animal that purrs" pulls the '
        "nine-lives association rather than a leg count"
    ),
    "elephant_to_ant_legs": (
        'generated " typically eight" -- the continuation is a height, not a '
        "leg count"
    ),
    "grape_to_banana_color": (
        'generated " typically red" -- red grapes are common, so the expected '
        '"purple" was contestable and the row is not a clean two-hop item'
    ),
}


def implicit_two_hop_expansion_v2() -> tuple[TextReplicationTask, ...]:
    """:func:`implicit_two_hop_expansion_v1` minus the rows that fail clean.

    v1 is left intact so the excluded rows stay inspectable and the reason for
    each is auditable next to it.  Ten rows survive, taking implicit_two_hop
    from n = 1 to n = 11.
    """

    return tuple(
        task
        for task in implicit_two_hop_expansion_v1()
        if task.task_id not in CLEAN_CAPABILITY_EXCLUSIONS_V2
    )


def anthropic_text_tasks_expanded_v2() -> tuple[TextReplicationTask, ...]:
    """The frozen paper set plus :func:`implicit_two_hop_expansion_v2`."""

    return anthropic_text_tasks() + implicit_two_hop_expansion_v2()


def anthropic_text_tasks_expanded_v1() -> tuple[TextReplicationTask, ...]:
    """The frozen paper set plus :func:`implicit_two_hop_expansion_v1`.

    ``anthropic_text_tasks`` is deliberately **not** modified: the completed
    alpha=2 run must stay byte-for-byte re-derivable, and its
    :func:`text_task_digest` is bound into that run's fingerprint.  Using this
    function instead changes the digest, so a run over the expanded set gets its
    own run directory and can never resume into the completed one.
    """

    return anthropic_text_tasks() + implicit_two_hop_expansion_v1()


def task_set_token_preflight(
    tasks: Sequence[TextReplicationTask],
    encode: Callable[[str], Sequence[int]],
    *,
    extra_concepts: Sequence[str] = ("zebra", "giraffe", "Japan", "Brazil"),
) -> dict:
    """Check every concept a task set needs resolves to one token. CPU only.

    The notebook builds ``TEXT_CONCEPT_TOKENS`` *after* the model is on the GPU,
    so one multi-token concept currently fails a paid session.  This runs the
    same resolution against a tokenizer alone and reports **every** failure at
    once rather than raising on the first, so a single pass fixes the whole set.

    Returns a report; it never raises for an unresolvable concept.  Call
    :func:`assert_task_set_resolvable` to turn the report into a refusal.
    """

    from jlens.mmpilot.coordinate_swap import (
        MultiTokenConceptError,
        resolve_concept_token,
    )

    selected = tuple(tasks)
    names = sorted(
        {task.source for task in selected}
        | {task.target for task in selected}
        | {semantic_answer_concept(task.clean_answer) for task in selected}
        | {semantic_answer_concept(task.swapped_answer) for task in selected}
        | set(map(str, extra_concepts))
    )
    resolved: dict[str, dict] = {}
    unresolvable: dict[str, str] = {}
    for name in names:
        try:
            resolved[name] = resolve_concept_token(encode, name).to_dict()
        except MultiTokenConceptError as error:
            unresolvable[name] = str(error)

    collisions = []
    for task in selected:
        pairs = (
            ("source_target", task.source, task.target),
            (
                "clean_swapped_answer",
                semantic_answer_concept(task.clean_answer),
                semantic_answer_concept(task.swapped_answer),
            ),
        )
        for kind, left, right in pairs:
            if left in resolved and right in resolved:
                if resolved[left]["token_id"] == resolved[right]["token_id"]:
                    collisions.append(
                        {
                            "task_id": task.task_id,
                            "kind": kind,
                            "left": left,
                            "right": right,
                            "token_id": resolved[left]["token_id"],
                        }
                    )

    families: dict[str, int] = {}
    for task in selected:
        families[task.family] = families.get(task.family, 0) + 1

    payload = {
        "version": IMPLICIT_TWO_HOP_EXPANSION_VERSION,
        "n_tasks": len(selected),
        "tasks_by_family": families,
        "n_concepts": len(names),
        "unresolvable": unresolvable,
        "collisions": collisions,
        "all_single_token": not unresolvable,
        "no_collisions": not collisions,
        "passed": not unresolvable and not collisions,
        "task_digest": text_task_digest(selected),
        "resolved": resolved,
    }
    return {**payload, "preflight_checksum": payload_checksum(payload)}


def assert_task_set_resolvable(report) -> None:
    """Refuse a task set whose concepts cannot be intervened on."""

    if report.get("unresolvable"):
        listed = ", ".join(sorted(report["unresolvable"]))
        raise WorkspaceReplicationRefused(
            f"these concepts are not single tokens and cannot be swapped: {listed}. "
            "Replace the task rows that use them; truncating a multi-token concept "
            "would intervene on a different concept."
        )
    if report.get("collisions"):
        listed = ", ".join(
            f"{row['task_id']}:{row['kind']}({row['left']}/{row['right']})"
            for row in report["collisions"]
        )
        raise WorkspaceReplicationRefused(
            f"these rows resolve two distinct roles to one token: {listed}. "
            "There is nothing to exchange and no visible answer change."
        )


def text_task_digest(tasks: Sequence[TextReplicationTask] | None = None) -> str:
    selected = anthropic_text_tasks() if tasks is None else tuple(tasks)
    return payload_checksum(
        {
            "version": PROTOCOL_VERSION,
            "input_protocol": TEXT_INPUT_PROTOCOL_VERSION,
            "completion_instruction": TEXT_COMPLETION_INSTRUCTION,
            "tasks": [task.to_dict() for task in selected],
        }
    )


def semantic_answer_concept(answer: str) -> str:
    """The single-token semantic head used only for causal diagnostics.

    Free generation remains the primary endpoint.  This helper merely maps a
    digit to its English number word so Gemma's whitespace-plus-digit
    tokenization cannot prevent a full-vocabulary, non-teacher-forced trace or
    a direct-answer positive control.  Every non-digit answer is unchanged.
    """

    normalized = str(answer).strip()
    return _NUMBER_WORDS.get(normalized, normalized)


def swapped_answer_diagnostic_tokens(
    answer: str,
    concept_tokens: Mapping[str, object],
    encode: Callable[[str], Sequence[int]] | None = None,
    *,
    allow_missing_head: bool = False,
) -> dict[str, int]:
    """Diagnostic token ids for a swapped answer, covering both surface forms.

    ``semantic_answer_concept`` maps a digit answer to its English word so the
    concept resolves to a single leading-space token (``" four"``). But Gemma
    can answer the digit path instead, and ``" 4"`` tokenizes as **two** tokens
    (``" "`` then ``"4"``) -- so on that path the word-form probe observes
    neither decoded step and the recorded log-probability is about a token the
    model never considered emitting.

    ``bird_to_cow_legs`` is the worked example: the swap flipped the completion
    to ``" 4"`` (a correct recomputation, matched by
    :func:`completion_answer_matches`, which knows digit/word equivalence) while
    the word-form probe reported a *decrease* of 2.32 nats.

    So numeric answers also get ``swapped_answer_digit``, the bare digit token,
    and any analysis should take the **max** over the two forms. Non-numeric
    answers are unchanged, and the ``swapped_answer_head`` key keeps its exact
    previous meaning so completed runs stay comparable on that field.
    """

    head = semantic_answer_concept(answer)
    if head in concept_tokens:
        out = {"swapped_answer_head": int(concept_tokens[head].token_id)}
    elif allow_missing_head:
        # This trace is optional: unrestricted generated text remains the
        # endpoint. Large confirmation sets can contain multi-token answer
        # heads even when source and target are valid one-token coordinates.
        out = {}
    else:
        raise KeyError(head)
    bare = str(answer).strip()
    if bare in _NUMBER_WORDS and encode is not None:
        ids = list(encode(bare))
        if len(ids) == 1:
            out["swapped_answer_digit"] = int(ids[0])
    return out


def best_swapped_answer_logprob(result: Mapping, step: int | None = None) -> float | None:
    """Best log-probability across every recorded swapped-answer surface form.

    Takes the max over ``swapped_answer_head`` and ``swapped_answer_digit`` so a
    digit-path answer is not scored on a word-form probe. ``step`` selects one
    decoded position; ``None`` maximizes over all of them.
    """

    best = None
    for entry in result.get("full_vocabulary_diagnostic_trace") or ():
        if step is not None and int(entry.get("step", -1)) != int(step):
            continue
        for name, token in (entry.get("tokens") or {}).items():
            if not str(name).startswith("swapped_answer"):
                continue
            value = float(token["logprob"])
            if best is None or value > best:
                best = value
    return best


def text_diagnostic_bands(layers: Sequence[int]) -> tuple[tuple[int, ...], ...]:
    """Predeclare every singleton and every suffix band, without duplicates.

    The completed L33--L40 run tested only the full eight-layer band.  These
    development-only conditions distinguish a localized effect from repeated
    exchange/cancellation.  Any selected condition still requires a fresh
    confirmation before it can support a claim.
    """

    ordered = tuple(sorted(set(map(int, layers))))
    if not ordered:
        raise WorkspaceReplicationRefused("text diagnostics need at least one layer")
    if ordered != tuple(range(ordered[0], ordered[-1] + 1)):
        raise WorkspaceReplicationRefused(
            f"text diagnostic layers must be contiguous, got {list(ordered)}"
        )
    candidates = [(layer,) for layer in ordered]
    candidates.extend(tuple(ordered[index:]) for index in range(len(ordered) - 1))
    unique: list[tuple[int, ...]] = []
    for band in candidates:
        if band not in unique:
            unique.append(band)
    return tuple(unique)


def build_assistant_prefill_completion_inputs(backend, prompt: str):
    """Adapt a literal completion task to Gemma's instruction interface.

    Gemma 4 E4B-IT restarts or echoes sentence fragments passed either as user
    messages or as raw untemplated text.  The architecture-appropriate
    adaptation gives a generic, answer-free completion instruction as the user
    turn and places the paper's literal fragment in an assistant turn with
    ``continue_final_message=True``.  The generated tokens therefore continue
    the existing fragment rather than answer a new chat turn.
    """

    from jlens.mmpilot.backend import BuiltInputs, text_hash

    processor = getattr(backend, "processor", None)
    device = getattr(backend, "device", None)
    if processor is None or device is None:
        raise WorkspaceReplicationRefused(
            "assistant-prefill completion requires the real backend's processor and device"
        )
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": TEXT_COMPLETION_INSTRUCTION}],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": str(prompt)}],
        },
    ]
    apply_chat_template = getattr(processor, "apply_chat_template", None)
    if not callable(apply_chat_template):
        raise WorkspaceReplicationRefused(
            "assistant-prefill completion requires the pinned chat template"
        )
    encoded = apply_chat_template(
        messages,
        continue_final_message=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    tensors = {
        key: (value.to(device) if torch.is_tensor(value) else value)
        for key, value in dict(encoded).items()
    }
    tensors.setdefault("use_cache", False)
    input_ids = tensors.get("input_ids")
    if not torch.is_tensor(input_ids) or input_ids.ndim != 2:
        raise WorkspaceReplicationRefused(
            "the assistant-prefill route did not produce rank-two input_ids"
        )
    return BuiltInputs(
        tensors=tensors,
        prompt_len=int(input_ids.shape[1]),
        modality="text",
        prompt_hash=text_hash(str(prompt)),
        route={
            "route": "assistant_prefill_completion",
            "chat_template_used": True,
            "continue_final_message": True,
            "input_protocol": TEXT_INPUT_PROTOCOL_VERSION,
        },
        modality_token_range=None,
    )


def build_multimodal_assistant_prefill_inputs(
    backend,
    *,
    modality: str,
    assistant_prefill: str,
    caption: str | None = None,
    image=None,
    audio=None,
    sampling_rate: int | None = None,
    media_path: str | None = None,
):
    """Build one answer-neutral completion input for any supported modality.

    This is the multimodal analogue of
    :func:`build_assistant_prefill_completion_inputs`.  The evidence remains in
    the user turn while the assistant turn ends in a factual sentence stem such
    as ``"The number of legs on the animal in the evidence is"``.  The model
    generates the answer itself: no candidate list, answer token, or
    teacher-forced continuation is supplied.

    Using the same assistant-prefill route in every modality is scientifically
    important.  Gemma otherwise spends the two-token text/image endpoint on a
    conversational restart (``"The animal"``), while its native audio template
    often emits the requested digit immediately.  That is a protocol artifact,
    not a modality comparison.
    """

    from jlens.mmpilot.backend import (
        BuiltInputs,
        contiguous_token_range,
        file_checksum,
        text_hash,
    )

    modality = str(modality)
    if modality not in {"text", "image", "spoken_audio"}:
        raise WorkspaceReplicationRefused(
            f"unknown multimodal completion modality {modality!r}"
        )
    processor = getattr(backend, "processor", None)
    device = getattr(backend, "device", None)
    if processor is None or device is None:
        raise WorkspaceReplicationRefused(
            "multimodal assistant-prefill completion requires the real "
            "backend's processor and device"
        )
    apply_chat_template = getattr(processor, "apply_chat_template", None)
    if not callable(apply_chat_template):
        raise WorkspaceReplicationRefused(
            "multimodal assistant-prefill completion requires the pinned chat "
            "template"
        )

    instruction = MULTIMODAL_COMPLETION_INSTRUCTION
    user_content: list[dict] = []
    audio_record = None
    modality_token_id = None
    if modality == "text":
        if not str(caption or "").strip():
            raise WorkspaceReplicationRefused(
                "the text completion route requires a non-empty caption"
            )
        user_content.append(
            {
                "type": "text",
                "text": f"Evidence caption: {str(caption).strip()}\n{instruction}",
            }
        )
    elif modality == "image":
        if image is None:
            raise WorkspaceReplicationRefused(
                "the image completion route requires an image"
            )
        user_content.extend(
            (
                {"type": "image", "image": image},
                {"type": "text", "text": instruction},
            )
        )
        modality_token_id = getattr(backend, "interface", {}).get(
            "image_token_id"
        )
    else:
        from jlens.mmpilot.audio import (
            audio_content_block,
            prepare_waveform,
            verify_audio_encoding,
        )

        resolved = getattr(backend, "audio_interface", None)
        if resolved is None or audio is None:
            raise WorkspaceReplicationRefused(
                "the spoken-audio completion route requires the resolved native "
                "audio interface and a waveform"
            )
        prepared = prepare_waveform(
            audio,
            int(sampling_rate) if sampling_rate is not None else resolved.sampling_rate,
            expected_rate=resolved.sampling_rate,
        )
        user_content.extend(
            (
                audio_content_block(prepared.samples),
                {"type": "text", "text": instruction},
            )
        )
        modality_token_id = int(resolved.audio_token_id)
        audio_record = {"waveform": prepared.to_dict()}

    messages = [
        {"role": "user", "content": user_content},
        {
            "role": "assistant",
            "content": [{"type": "text", "text": str(assistant_prefill)}],
        },
    ]
    encoded = apply_chat_template(
        messages,
        continue_final_message=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    tensors = {
        key: (value.to(device) if torch.is_tensor(value) else value)
        for key, value in dict(encoded).items()
    }
    tensors.setdefault("use_cache", False)
    input_ids = tensors.get("input_ids")
    if not torch.is_tensor(input_ids) or input_ids.ndim != 2:
        raise WorkspaceReplicationRefused(
            "the multimodal assistant-prefill route did not produce rank-two "
            "input_ids"
        )
    modality_range = contiguous_token_range(input_ids, modality_token_id)
    if modality == "spoken_audio":
        verified = verify_audio_encoding(
            encoded, audio_token_id=int(modality_token_id)
        )
        modality_range = list(verified["audio_token_span"])
        audio_record = {**verified, **dict(audio_record or {})}
    route = {
        "route": "multimodal_assistant_prefill_completion",
        "chat_template_used": True,
        "continue_final_message": True,
        "input_protocol": MULTIMODAL_INPUT_PROTOCOL_VERSION,
        "answer_prefill_is_answer_neutral": True,
        "candidate_list_supplied": False,
        "teacher_forcing_used": False,
    }
    return BuiltInputs(
        tensors=tensors,
        prompt_len=int(input_ids.shape[1]),
        modality=modality,
        prompt_hash=text_hash(
            f"{modality}|{caption or ''}|{instruction}|{assistant_prefill}"
        ),
        media_checksum=file_checksum(media_path) if media_path else None,
        route=route,
        modality_token_range=modality_range,
        audio=audio_record,
    )


#: Gemma control tokens that can terminate a completion. They are markup, not
#: content, and must be removed before the final lexical item is identified --
#: the normalizer strips ``<``, ``|`` and ``>`` as punctuation, so ``<turn|>``
#: otherwise survives as the bare word ``"turn"`` and *becomes* the final item.
#: That silently rejected correct answers: ``" Paris<turn|>"`` on
#: ``china_to_france_capital`` is exactly the swapped answer and scored False.
_CONTROL_TOKEN_WORDS = frozenset({"turn", "eos", "bos", "pad", "unk", "mask"})
_CONTROL_TOKEN = re.compile(r"<\|?[a-z_]+\|?>", re.IGNORECASE)


def completion_answer_matches(generated: str, answer: str) -> bool:
    """Match an unrestricted completion by its final semantic head.

    Instruction models often produce a short modifier before the answer
    (``Western Europe``, ``Mandarin Chinese``) and spell a digit as a number
    word.  This rule is deliberately narrower than substring matching: the
    answer must be the final lexical item, and a negated mention never counts.

    Control tokens such as ``<turn|>`` are stripped first. They are generation
    terminators rather than content, and leaving them in makes ``"turn"`` the
    final lexical item and rejects an otherwise-correct answer.
    """

    from jlens.mmpilot.full_vocabulary import normalize_generated_text

    generated_words = re.findall(
        r"\w+",
        normalize_generated_text(_CONTROL_TOKEN.sub(" ", str(generated))),
        flags=re.UNICODE,
    )
    # Defence in depth: if a control token survived in a shape the pattern above
    # does not cover, drop it from the tail rather than scoring it as content.
    while len(generated_words) > 1 and generated_words[-1] in _CONTROL_TOKEN_WORDS:
        generated_words.pop()
    wanted_words = re.findall(
        r"\w+", normalize_generated_text(str(answer)), flags=re.UNICODE
    )
    if not generated_words or not wanted_words:
        return False
    if any(word in _NEGATION_MARKERS for word in generated_words):
        return False
    wanted = wanted_words[-1]
    aliases = {wanted}
    if wanted in _NUMBER_WORDS:
        aliases.add(_NUMBER_WORDS[wanted])
    reverse = {word: digit for digit, word in _NUMBER_WORDS.items()}
    if wanted in reverse:
        aliases.add(reverse[wanted])
    return generated_words[-1] in aliases


@torch.no_grad()
def unrestricted_greedy_completion(
    backend,
    inputs,
    *,
    answer: str,
    max_new_tokens: int = TEXT_MAX_NEW_TOKENS,
    diagnostic_token_ids: Mapping[str, int] | None = None,
) -> dict:
    """Generate a complete answer without candidates or teacher forcing.

    Anthropic's tokenizer represents the paper's digit answers as one next
    token. Gemma 4 represents the same continuation as two tokens (a whitespace
    token followed by the digit), so a one-row global-argmax endpoint is not
    defined for this model. This endpoint preserves the literal prompt and
    answer while greedily generating the complete token sequence. Every token
    is selected from the full vocabulary; no answer token is appended.
    """

    from jlens.mmpilot.capability import _extend_tensors
    from jlens.mmpilot.full_vocabulary import (
        normalize_generated_text,
        token_decoder,
    )

    budget = int(max_new_tokens)
    if budget < 1:
        raise WorkspaceReplicationRefused("max_new_tokens must be positive")
    decoder = token_decoder(backend)
    tensors = dict(inputs.tensors)
    input_ids = tensors.get("input_ids")
    if not torch.is_tensor(input_ids) or input_ids.ndim != 2:
        raise WorkspaceReplicationRefused(
            "unrestricted complete-answer generation requires rank-two input_ids"
        )
    if int(input_ids.shape[1]) != int(inputs.prompt_len):
        raise WorkspaceReplicationRefused(
            "generation must begin at the untouched prompt boundary"
        )

    diagnostic_ids = {
        str(name): int(token_id)
        for name, token_id in dict(diagnostic_token_ids or {}).items()
    }
    generated: list[int] = []
    diagnostic_trace: list[dict] = []
    for step_index in range(budget):
        step_tensors = (
            tensors
            if not generated
            else _extend_tensors(tensors, int(inputs.prompt_len), generated)
        )
        logits = backend.forward_logits(step_tensors)
        step = logits[0, -1].float()
        if not bool(torch.isfinite(step).all()):
            raise WorkspaceReplicationRefused(
                "non-finite logits during unrestricted generation"
            )
        top_id = int(step.argmax())
        log_normalizer = torch.logsumexp(step, dim=-1)
        token_rows = {}
        for name, token_id in diagnostic_ids.items():
            if not 0 <= token_id < int(step.shape[0]):
                raise WorkspaceReplicationRefused(
                    f"diagnostic token {name!r} id {token_id} is outside a "
                    f"vocabulary of size {int(step.shape[0])}"
                )
            token_logit = step[token_id]
            token_rows[name] = {
                "token_id": token_id,
                "logit": float(token_logit),
                "logprob": float(token_logit - log_normalizer),
                "rank": int((step > token_logit).sum()) + 1,
                "is_global_top1": token_id == top_id,
                "margin_to_global_top1": float(token_logit - step[top_id]),
            }
        diagnostic_trace.append(
            {
                "step": int(step_index),
                "selected_token_id": top_id,
                "tokens": token_rows,
            }
        )
        generated.append(top_id)

    text = "".join(decoder(token_id) for token_id in generated)
    return {
        "endpoint_version": TEXT_OUTPUT_ENDPOINT_VERSION,
        "endpoint": "unrestricted_greedy_complete_answer",
        "teacher_forcing_used": False,
        "candidate_list_supplied": False,
        "temperature": 0.0,
        "do_sample": False,
        "max_new_tokens": budget,
        "n_forward_passes": budget,
        "generated_token_ids": generated,
        "generated_text": text,
        "normalized_generated_text": normalize_generated_text(text),
        "answer": str(answer),
        "normalized_answer": normalize_generated_text(str(answer)),
        "answer_match_rule": TEXT_ANSWER_MATCH_RULE,
        "answer_match": bool(completion_answer_matches(text, str(answer))),
        "full_vocabulary_diagnostic_is_teacher_forced": False,
        "full_vocabulary_diagnostic_trace": diagnostic_trace,
    }


@torch.no_grad()
def unrestricted_greedy_swap_trial(
    backend,
    inputs,
    *,
    bases: Mapping,
    alpha: float,
    answer: str,
    max_new_tokens: int = TEXT_MAX_NEW_TOKENS,
    position_rule: str = "all_prompt_positions",
    diagnostic_token_ids: Mapping[str, int] | None = None,
    realization_policy: ModelDtypeRealizationPolicy | None = None,
) -> dict:
    """Run the paper's swap and freely generate the complete answer.

    Hooks remain active for every greedy decoding step but patch only the
    pre-existing prompt positions. Thus the generated answer is observed, not
    supplied, while the intervention is identical on each recomputed forward.
    """

    with coordinate_swap_band(
        backend.blocks,
        bases,
        alpha=float(alpha),
        prompt_len=int(inputs.prompt_len),
        position_rule=str(position_rule),
        evidence_span=getattr(inputs, "modality_token_range", None),
        record_coordinates=False,
        realization_policy=realization_policy,
    ) as stats:
        generated = unrestricted_greedy_completion(
            backend,
            inputs,
            answer=str(answer),
            max_new_tokens=int(max_new_tokens),
            diagnostic_token_ids=diagnostic_token_ids,
        )

    positions = {
        str(layer): list(stats[layer].get("positions") or [])
        for layer in sorted(stats)
    }
    expected = list(range(int(inputs.prompt_len)))
    return {
        **generated,
        "alpha": float(alpha),
        "alpha_role": "exact_exchange" if float(alpha) == 1.0 else "nonexact",
        "position_rule": str(position_rule),
        "layers_patched": sorted(int(layer) for layer in stats),
        "positions_patched": positions,
        "all_prompt_positions_patched": all(
            layer_positions == expected for layer_positions in positions.values()
        )
        if str(position_rule) == "all_prompt_positions"
        else None,
        "hook_forward_passes_by_layer": {
            str(layer): int(stats[layer].get("n_forward_passes") or 0)
            for layer in sorted(stats)
        },
        "intervention_diagnostics": summarize_swap_diagnostics(stats),
    }


def _finite_numbers(value) -> bool:
    if isinstance(value, Mapping):
        return all(_finite_numbers(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_numbers(item) for item in value)
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return True


def summarize_swap_diagnostics(stats: Mapping[int, Mapping]) -> dict:
    """Compact the real hook audit without persisting activation-sized arrays."""

    by_layer = {}
    for layer in sorted(map(int, stats)):
        row = dict(stats[layer])
        history = list(row.get("swap_history") or ())
        if not history and row.get("swap"):
            history = [dict(row["swap"])]
        update_ratios = []
        activation_ratios = []
        for record in history:
            updates = list(record.get("update_norm") or ())
            before = list(record.get("activation_norm_before") or ())
            after = list(record.get("activation_norm_after") or ())
            update_ratios.extend(
                float(update) / max(float(norm), 1e-12)
                for update, norm in zip(updates, before, strict=True)
            )
            activation_ratios.extend(
                float(end) / max(float(start), 1e-12)
                for start, end in zip(before, after, strict=True)
            )
        basis = dict(row.get("basis") or {})
        basis_diagnostics = dict(basis.get("diagnostics") or {})
        by_layer[str(layer)] = {
            "n_forward_passes": int(row.get("n_forward_passes") or 0),
            "n_swap_records": len(history),
            "n_positions": int(row.get("n_positions") or 0),
            "condition_number": basis_diagnostics.get("condition_number"),
            "numerical_rank": basis_diagnostics.get("numerical_rank"),
            "all_finite": bool(history) and all(_finite_numbers(record) for record in history),
            "all_alpha_one_exact_before_cast": bool(history)
            and all(bool(record.get("alpha_one_is_exact_exchange")) for record in history),
            "all_model_dtype_realizations_converged": bool(history)
            and all(
                bool(record.get("model_dtype_realization_converged", True))
                for record in history
            ),
            "max_model_dtype_corrections_applied": max(
                (
                    int(record.get("model_dtype_corrections_applied") or 0)
                    for record in history
                ),
                default=0,
            ),
            "max_ideal_coordinate_error": max(
                (float(record["max_coordinate_update_error"]) for record in history),
                default=None,
            ),
            "max_post_cast_coordinate_error": max(
                (
                    float(record["max_post_cast_coordinate_update_error"])
                    for record in history
                ),
                default=None,
            ),
            "max_post_cast_relative_coordinate_error": max(
                (
                    float(record["max_post_cast_relative_coordinate_update_error"])
                    for record in history
                ),
                default=None,
            ),
            "max_post_cast_relative_residual_drift": max(
                (
                    float(record["max_post_cast_relative_orthogonal_residual_drift"])
                    for record in history
                ),
                default=None,
            ),
            "max_update_to_activation_ratio": max(update_ratios, default=None),
            "min_after_to_before_activation_ratio": min(
                activation_ratios, default=None
            ),
            "max_after_to_before_activation_ratio": max(
                activation_ratios, default=None
            ),
        }
    layers = list(by_layer.values())
    relative_errors = [
        float(row["max_post_cast_relative_coordinate_error"])
        for row in layers
        if row["max_post_cast_relative_coordinate_error"] is not None
    ]
    residual_drifts = [
        float(row["max_post_cast_relative_residual_drift"])
        for row in layers
        if row["max_post_cast_relative_residual_drift"] is not None
    ]
    payload = {
        "version": TEXT_DIAGNOSTIC_VERSION,
        "by_layer": by_layer,
        "all_hooks_fired": bool(layers)
        and all(
            row["n_forward_passes"] == TEXT_MAX_NEW_TOKENS
            and row["n_swap_records"] == TEXT_MAX_NEW_TOKENS
            for row in layers
        ),
        "all_finite": bool(layers) and all(row["all_finite"] for row in layers),
        "all_layers_are_exact_alpha_one_exchange_before_cast": bool(layers)
        and all(row["all_alpha_one_exact_before_cast"] for row in layers),
        "all_model_dtype_realizations_converged": bool(layers)
        and all(row["all_model_dtype_realizations_converged"] for row in layers),
        "model_dtype_realization_version": MODEL_DTYPE_REALIZATION_VERSION,
        "max_post_cast_relative_coordinate_error": max(
            relative_errors, default=None
        ),
        "max_post_cast_relative_residual_drift": max(
            residual_drifts, default=None
        ),
        "post_cast_error_threshold": TEXT_POST_CAST_MAX_RELATIVE_ERROR,
        "post_cast_coordinate_audit_passed": bool(relative_errors)
        and max(relative_errors) <= TEXT_POST_CAST_MAX_RELATIVE_ERROR,
        "post_cast_residual_audit_passed": bool(residual_drifts)
        and max(residual_drifts) <= TEXT_POST_CAST_MAX_RELATIVE_ERROR,
        "post_cast_audit_passed": bool(relative_errors)
        and bool(residual_drifts)
        and max(relative_errors) <= TEXT_POST_CAST_MAX_RELATIVE_ERROR
        and max(residual_drifts) <= TEXT_POST_CAST_MAX_RELATIVE_ERROR,
    }
    return {**payload, "diagnostic_checksum": payload_checksum(payload)}


@contextmanager
def _norm_matched_direct_answer_band(
    blocks: Sequence[nn.Module],
    bases: Mapping[int, SwapBasis],
    answer_vectors: Mapping[int, torch.Tensor],
    *,
    prompt_len: int,
    position_rule: str = PRIMARY_POSITION_RULE,
    evidence_span: Sequence[int] | None = None,
    realization_policy: ModelDtypeRealizationPolicy | None = None,
    alpha: float = 1.0,
):
    """Add the answer direction with each position's exact-swap update norm.

    This is a positive control, not a coordinate swap.  Its update has the
    same post-cast L2 norm as the exchange would have had **at ``alpha``** at
    that layer and position, so a failure cannot be dismissed as comparing
    interventions of unrelated intensity.

    ``alpha`` must be the same strength the swap arm uses.  It was previously
    pinned to 1.0, which silently unmatched the control whenever the swap ran
    at any other strength -- an alpha=2 run then compared a doubled swap
    against an alpha=1 control and its pass rate measured nothing.
    """

    if set(map(int, bases)) != set(map(int, answer_vectors)):
        raise WorkspaceReplicationRefused(
            "direct-answer vectors must cover exactly the coordinate-swap band"
        )
    stats: dict[int, dict] = {}
    with ExitStack() as stack:
        for layer in sorted(map(int, bases)):
            basis = bases[layer]
            trainable = [
                name
                for name, parameter in blocks[layer].named_parameters()
                if parameter.requires_grad
            ]
            if trainable:
                raise WorkspaceReplicationRefused(
                    f"direct-answer control found trainable parameters at layer "
                    f"{layer}: {trainable}"
                )
            answer = answer_vectors[layer].detach().to(torch.float64).flatten()
            if answer.numel() != basis.d_model or not bool(torch.isfinite(answer).all()):
                raise WorkspaceReplicationRefused(
                    f"invalid direct-answer vector at layer {layer}"
                )
            norm = float(answer.norm())
            if norm == 0.0:
                raise WorkspaceReplicationRefused(
                    f"zero direct-answer vector at layer {layer}"
                )
            unit_answer = answer / norm
            row = {
                "layer": layer,
                "n_forward_passes": 0,
                "positions": None,
                "history": [],
                "answer_vector_norm": norm,
                "answer_vector_checksum": tensor_checksum(answer),
            }
            stats[layer] = row

            def make_hook(
                layer_index: int,
                layer_basis: SwapBasis,
                layer_unit_answer: torch.Tensor,
                layer_row: dict,
            ):
                def hook(module, inputs, output):
                    is_tensor = torch.is_tensor(output)
                    hidden = output if is_tensor else output[0]
                    positions = resolve_positions(
                        position_rule,
                        prompt_len=int(prompt_len),
                        seq_len=int(hidden.shape[1]),
                        evidence_span=evidence_span,
                    )
                    selected = hidden[0, positions]
                    exact_after_cast, _ = swap_coordinates(
                        selected,
                        layer_basis.V,
                        alpha=float(alpha),
                        realization_policy=realization_policy,
                    )
                    matched_norm = (
                        exact_after_cast.detach().float() - selected.detach().float()
                    ).norm(dim=-1)
                    direction = layer_unit_answer.to(
                        device=hidden.device, dtype=torch.float32
                    )
                    requested_norm = matched_norm.clone()
                    correction_history = []
                    max_corrections = (
                        0
                        if realization_policy is None
                        else int(realization_policy.max_corrections)
                    )
                    for _correction_index in range(max_corrections + 1):
                        proposed = (
                            selected.detach().float()
                            + requested_norm[:, None] * direction
                        )
                        patched = proposed.to(hidden.dtype)
                        actual_norm = (
                            patched.detach().float() - selected.detach().float()
                        ).norm(dim=-1)
                        relative_match_error = (
                            (actual_norm - matched_norm).abs()
                            / matched_norm.clamp_min(1.0)
                        )
                        error_max = float(relative_match_error.max())
                        correction_history.append(error_max)
                        if realization_policy is None or error_max <= float(
                            realization_policy.relative_coordinate_tolerance
                        ):
                            break
                        requested_norm = requested_norm * (
                            matched_norm / actual_norm.clamp_min(1e-12)
                        )
                    new_hidden = hidden.clone()
                    new_hidden[0, positions] = patched
                    layer_row["n_forward_passes"] += 1
                    layer_row["positions"] = list(positions)
                    layer_row["history"].append(
                        {
                            "n_positions": len(positions),
                            "all_finite": bool(
                                torch.isfinite(patched).all()
                                and torch.isfinite(relative_match_error).all()
                            ),
                            "max_relative_norm_match_error": float(
                                relative_match_error.max()
                            ),
                            "model_dtype_realization_converged": bool(
                                realization_policy is None
                                or float(relative_match_error.max())
                                <= float(
                                    realization_policy.relative_coordinate_tolerance
                                )
                            ),
                            "model_dtype_corrections_applied": (
                                len(correction_history) - 1
                            ),
                            "max_update_to_activation_ratio": float(
                                (
                                    actual_norm
                                    / selected.detach().float().norm(dim=-1).clamp_min(1.0)
                                ).max()
                            ),
                        }
                    )
                    if is_tensor:
                        return new_hidden
                    return (new_hidden, *tuple(output)[1:])

                return hook

            handle = blocks[layer].register_forward_hook(
                make_hook(layer, basis, unit_answer, row)
            )
            stack.callback(handle.remove)
        yield stats


def summarize_direct_answer_diagnostics(stats: Mapping[int, Mapping]) -> dict:
    by_layer = {}
    for layer in sorted(map(int, stats)):
        row = dict(stats[layer])
        history = list(row.get("history") or ())
        by_layer[str(layer)] = {
            "n_forward_passes": int(row.get("n_forward_passes") or 0),
            "n_records": len(history),
            "all_finite": bool(history)
            and all(bool(record.get("all_finite")) for record in history),
            "max_relative_norm_match_error": max(
                (
                    float(record["max_relative_norm_match_error"])
                    for record in history
                ),
                default=None,
            ),
            "all_model_dtype_realizations_converged": bool(history)
            and all(
                bool(record.get("model_dtype_realization_converged", True))
                for record in history
            ),
            "max_model_dtype_corrections_applied": max(
                (
                    int(record.get("model_dtype_corrections_applied") or 0)
                    for record in history
                ),
                default=0,
            ),
            "max_update_to_activation_ratio": max(
                (
                    float(record["max_update_to_activation_ratio"])
                    for record in history
                ),
                default=None,
            ),
        }
    payload = {
        "version": TEXT_DIAGNOSTIC_VERSION,
        "control": "direct_answer_norm_matched_to_exact_swap_per_position",
        "by_layer": by_layer,
        "all_hooks_fired": bool(by_layer)
        and all(
            row["n_forward_passes"] == TEXT_MAX_NEW_TOKENS
            and row["n_records"] == TEXT_MAX_NEW_TOKENS
            for row in by_layer.values()
        ),
        "all_finite": bool(by_layer)
        and all(row["all_finite"] for row in by_layer.values()),
        "all_model_dtype_realizations_converged": bool(by_layer)
        and all(
            row["all_model_dtype_realizations_converged"]
            for row in by_layer.values()
        ),
    }
    return {**payload, "diagnostic_checksum": payload_checksum(payload)}


@torch.no_grad()
def unrestricted_greedy_direct_answer_trial(
    backend,
    inputs,
    *,
    bases: Mapping[int, SwapBasis],
    answer_vectors: Mapping[int, torch.Tensor],
    answer: str,
    max_new_tokens: int = TEXT_MAX_NEW_TOKENS,
    position_rule: str = PRIMARY_POSITION_RULE,
    diagnostic_token_ids: Mapping[str, int] | None = None,
    realization_policy: ModelDtypeRealizationPolicy | None = None,
    alpha: float = 1.0,
) -> dict:
    with _norm_matched_direct_answer_band(
        backend.blocks,
        bases,
        answer_vectors,
        prompt_len=int(inputs.prompt_len),
        position_rule=str(position_rule),
        evidence_span=getattr(inputs, "modality_token_range", None),
        realization_policy=realization_policy,
        alpha=float(alpha),
    ) as stats:
        generated = unrestricted_greedy_completion(
            backend,
            inputs,
            answer=str(answer),
            max_new_tokens=int(max_new_tokens),
            diagnostic_token_ids=diagnostic_token_ids,
        )
    positions = {
        str(layer): list(stats[layer].get("positions") or [])
        for layer in sorted(stats)
    }
    expected = list(range(int(inputs.prompt_len)))
    return {
        **generated,
        "condition": "direct_answer_norm_matched",
        "position_rule": str(position_rule),
        "layers_patched": sorted(map(int, stats)),
        "positions_patched": positions,
        "all_prompt_positions_patched": all(
            layer_positions == expected for layer_positions in positions.values()
        ),
        "hook_forward_passes_by_layer": {
            str(layer): int(stats[layer].get("n_forward_passes") or 0)
            for layer in sorted(stats)
        },
        "intervention_diagnostics": summarize_direct_answer_diagnostics(stats),
    }


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    # Activations are captured on the model device while published/matched
    # lens vectors may be retained on CPU.  This diagnostic is tiny and is not
    # part of the model forward, so normalize both operands to CPU explicitly
    # before taking their dot product.  A dtype-only ``to`` preserves the
    # original devices and fails for the real CUDA/CPU pairing.
    left = a.detach().to(device="cpu", dtype=torch.float64).flatten()
    right = b.detach().to(device="cpu", dtype=torch.float64).flatten()
    denominator = float(left.norm() * right.norm())
    if denominator == 0.0:
        return 0.0
    value = float(left.dot(right)) / denominator
    if not math.isfinite(value):
        raise WorkspaceReplicationRefused("a clean source-loading cosine is non-finite")
    return value


@torch.no_grad()
def capture_source_loading(
    backend,
    inputs,
    *,
    vectors_by_layer: Mapping[int, Mapping[str, torch.Tensor]],
    source: str,
    target: str,
    unrelated: Sequence[str] = (),
    sample_id: str,
    modality: str,
) -> list[dict]:
    """Measure clean J-lens loading at every original prompt position.

    This is an observation-only forward pass.  No hook returns a replacement,
    no candidate token is appended, and no causal outcome is available to the
    caller while localization is chosen.
    """

    layers = tuple(sorted(int(layer) for layer in vectors_by_layer))
    if not layers:
        raise WorkspaceReplicationRefused("source loading needs at least one layer")
    required = {source, target, *map(str, unrelated)}
    missing = {
        layer: sorted(required - set(vectors_by_layer[layer]))
        for layer in layers
        if required - set(vectors_by_layer[layer])
    }
    if missing:
        raise WorkspaceReplicationRefused(f"missing lens vectors: {missing}")

    with ActivationRecorder(backend.blocks, at=layers) as recorder:
        backend.forward_logits(inputs.tensors)
    evidence_span = inputs.modality_token_range
    evidence_positions = (
        set(range(int(evidence_span[0]), int(evidence_span[1])))
        if evidence_span is not None
        else set()
    )
    rows: list[dict] = []
    for layer in layers:
        activation = recorder.activations[layer].detach()[0, : inputs.prompt_len]
        source_vector = vectors_by_layer[layer][source]
        target_vector = vectors_by_layer[layer][target]
        V = torch.stack((source_vector, target_vector), dim=1)
        coordinates = read_coordinates(activation, V)
        for position in range(inputs.prompt_len):
            h = activation[position]
            unrelated_cosines = {
                name: _cosine(h, vectors_by_layer[layer][name])
                for name in unrelated
            }
            source_cosine = _cosine(h, source_vector)
            target_cosine = _cosine(h, target_vector)
            control = max([target_cosine, *unrelated_cosines.values()])
            rows.append(
                {
                    "version": LOADING_VERSION,
                    "sample_id": str(sample_id),
                    "modality": str(modality),
                    "layer": int(layer),
                    "position": int(position),
                    "position_class": (
                        "final_prompt_token"
                        if position == inputs.final_prompt_position
                        else "evidence"
                        if position in evidence_positions
                        else "non_evidence"
                    ),
                    "source": str(source),
                    "target": str(target),
                    "source_cosine": source_cosine,
                    "target_cosine": target_cosine,
                    "unrelated_cosines": unrelated_cosines,
                    "source_advantage": source_cosine - control,
                    "source_coordinate": float(coordinates[position, 0]),
                    "target_coordinate": float(coordinates[position, 1]),
                    "prompt_len": int(inputs.prompt_len),
                    "evidence_span": list(evidence_span) if evidence_span else None,
                    "causal_result_consulted": False,
                }
            )
    return rows


def _median(values: Sequence[float]) -> float:
    if not values:
        raise WorkspaceReplicationRefused("cannot summarize an empty loading cell")
    return float(statistics.median(map(float, values)))


def summarize_loading(rows: Sequence[Mapping]) -> dict:
    """Aggregate loading without discarding the per-position measurements."""

    if not rows:
        raise WorkspaceReplicationRefused("no clean source-loading rows were supplied")
    grouped: dict[tuple[str, int, str], list[Mapping]] = {}
    for row in rows:
        grouped.setdefault(
            (str(row["modality"]), int(row["layer"]), str(row["position_class"])),
            [],
        ).append(row)
    cells = []
    for (modality, layer, position_class), cell in sorted(grouped.items()):
        cells.append(
            {
                "modality": modality,
                "layer": layer,
                "position_class": position_class,
                "n": len(cell),
                "n_samples": len({str(row["sample_id"]) for row in cell}),
                "median_source_cosine": _median(
                    [float(row["source_cosine"]) for row in cell]
                ),
                "median_target_cosine": _median(
                    [float(row["target_cosine"]) for row in cell]
                ),
                "median_source_advantage": _median(
                    [float(row["source_advantage"]) for row in cell]
                ),
            }
        )
    payload = {
        "version": LOADING_VERSION,
        "n_rows": len(rows),
        "n_samples": len({str(row["sample_id"]) for row in rows}),
        "cells": cells,
        "causal_result_consulted": False,
    }
    return {**payload, "report_checksum": payload_checksum(payload)}


def _runs(layers: Sequence[int]) -> list[tuple[int, ...]]:
    runs: list[list[int]] = []
    for layer in sorted(set(map(int, layers))):
        if not runs or layer != runs[-1][-1] + 1:
            runs.append([layer])
        else:
            runs[-1].append(layer)
    return [tuple(run) for run in runs]


def freeze_loading_localization(
    rows: Sequence[Mapping],
    *,
    required_modalities: Sequence[str],
    candidate_layers: Sequence[int],
    min_source_advantage: float = 0.0,
    evidence_position_margin: float = 0.0,
) -> dict:
    """Choose a band and position rule using clean loading and nothing else.

    A layer is admissible only when its median source advantage is above the
    frozen margin in *every* required modality.  The longest contiguous run is
    used; ties choose the deeper run deterministically.  For image/audio, an
    evidence-only rule is chosen only when evidence positions beat non-evidence
    positions by the frozen margin in every modality that has an evidence span.
    Otherwise the literal paper rule (all prompt positions) remains in force.
    """

    required = tuple(map(str, required_modalities))
    layers = tuple(sorted(set(map(int, candidate_layers))))
    if not required or not layers:
        raise WorkspaceReplicationRefused("localization needs modalities and layers")
    by_cell: dict[tuple[str, int], list[Mapping]] = {}
    for row in rows:
        if int(row["layer"]) in layers and str(row["modality"]) in required:
            by_cell.setdefault((str(row["modality"]), int(row["layer"])), []).append(row)

    layer_evidence = []
    eligible = []
    for layer in layers:
        medians = {}
        complete = True
        for modality in required:
            cell = by_cell.get((modality, layer), [])
            if not cell:
                complete = False
                medians[modality] = None
            else:
                medians[modality] = _median(
                    [float(row["source_advantage"]) for row in cell]
                )
        passed = complete and all(
            float(medians[modality]) > float(min_source_advantage)
            for modality in required
        )
        if passed:
            eligible.append(layer)
        layer_evidence.append(
            {"layer": layer, "median_source_advantage": medians, "passed": passed}
        )

    runs = _runs(eligible)
    selected = max(runs, key=lambda run: (len(run), run[-1]), default=())

    position_evidence = []
    position_rule_by_modality = {modality: "all_prompt_positions" for modality in required}
    for modality in required:
        evidence_values = [
            float(row["source_advantage"])
            for row in rows
            if str(row["modality"]) == modality
            and int(row["layer"]) in selected
            and str(row["position_class"]) == "evidence"
        ]
        non_evidence_values = [
            float(row["source_advantage"])
            for row in rows
            if str(row["modality"]) == modality
            and int(row["layer"]) in selected
            and str(row["position_class"]) in {"non_evidence", "final_prompt_token"}
        ]
        if not evidence_values:
            continue
        evidence_median = _median(evidence_values)
        non_evidence_median = _median(non_evidence_values)
        passed = evidence_median >= non_evidence_median + float(evidence_position_margin)
        if selected and passed:
            position_rule_by_modality[modality] = "evidence_span_only"
        position_evidence.append(
            {
                "modality": modality,
                "median_evidence_advantage": evidence_median,
                "median_non_evidence_advantage": non_evidence_median,
                "passed": passed,
            }
        )
    unique_rules = sorted(set(position_rule_by_modality.values()))
    position_rule = unique_rules[0] if len(unique_rules) == 1 else "modality_specific"
    verdict = "LOADING_LOCALIZATION_GO" if selected else "LOADING_LOCALIZATION_NO_GO"
    payload = {
        "version": LOCALIZATION_VERSION,
        "verdict": verdict,
        "candidate_layers": list(layers),
        "required_modalities": list(required),
        "min_source_advantage": float(min_source_advantage),
        "evidence_position_margin": float(evidence_position_margin),
        "layer_evidence": layer_evidence,
        "eligible_layers": eligible,
        "contiguous_runs": [list(run) for run in runs],
        "selected_band": list(selected),
        "position_rule": position_rule,
        "position_rule_by_modality": position_rule_by_modality,
        "position_evidence": position_evidence,
        "causal_result_consulted": False,
        "selection_depended_on_causal_outcome": False,
    }
    return {**payload, "design_digest": payload_checksum(payload)}


def select_pair_from_loading(
    rows: Sequence[Mapping],
    *,
    candidate_pairs: Sequence[Sequence[str]],
    required_modalities: Sequence[str],
) -> dict:
    """Select a concept pair from clean source loading, never causal response.

    The score is the weakest modality's median source advantage, pooled across
    layers and positions.  This rewards a source representation that is
    already visible in every channel instead of a pair that happened to react
    strongly to an intervention.
    """

    pairs = [tuple(map(str, pair)) for pair in candidate_pairs]
    if any(len(pair) != 2 or pair[0] == pair[1] for pair in pairs):
        raise WorkspaceReplicationRefused(f"invalid candidate pairs: {pairs}")
    modalities = tuple(map(str, required_modalities))
    ranking = []
    for source, target in pairs:
        per_modality = {}
        for modality in modalities:
            values = [
                float(row["source_advantage"])
                for row in rows
                if str(row.get("source")) == source
                and str(row.get("target")) == target
                and str(row.get("modality")) == modality
            ]
            per_modality[modality] = _median(values) if values else None
        complete = all(value is not None for value in per_modality.values())
        score = (
            min(float(value) for value in per_modality.values())
            if complete
            else float("-inf")
        )
        ranking.append(
            {
                "source": source,
                "target": target,
                "per_modality_median_source_advantage": per_modality,
                "weakest_modality_score": score,
                "complete": complete,
            }
        )
    ranking.sort(
        key=lambda row: (
            -float(row["weakest_modality_score"]),
            str(row["source"]),
            str(row["target"]),
        )
    )
    if not ranking or not ranking[0]["complete"]:
        raise WorkspaceReplicationRefused(
            "no candidate pair has loading measurements in every required modality"
        )
    payload = {
        "version": LOCALIZATION_VERSION,
        "selection_rule": "maximize the weakest modality's median clean source advantage",
        "required_modalities": list(modalities),
        "ranking": ranking,
        "selected_pair": [ranking[0]["source"], ranking[0]["target"]],
        "causal_result_consulted": False,
        "selection_depended_on_causal_outcome": False,
    }
    return {**payload, "selection_digest": payload_checksum(payload)}


def text_capability_verdict(
    rows: Sequence[Mapping], *, tasks: Sequence[TextReplicationTask] | None = None
) -> dict:
    """Decide whether causal spending is licensed for the text tasks.

    ``tasks`` defaults to :func:`anthropic_text_tasks` so the frozen runs stay
    re-derivable.  It **must** be passed when the run uses a different set:
    verdicts computed over a hardcoded set silently ignore every extra row, so
    a task Gemma cannot answer would never block causal spending.
    """

    selected = anthropic_text_tasks() if tasks is None else tuple(tasks)
    expected = {task.task_id for task in selected}
    by_task = {str(row["task_id"]): dict(row) for row in rows}
    missing = sorted(expected - set(by_task))
    passed = not missing and all(
        bool(by_task[task_id].get("clean_correct")) for task_id in expected
    )
    payload = {
        "version": PROTOCOL_VERSION,
        "verdict": (
            "TEXT_PAPER_CAPABILITY_GO" if passed else "TEXT_PAPER_CAPABILITY_NO_GO"
        ),
        "all_clean_answers_correct": passed,
        "missing_tasks": missing,
        "causal_spending_licensed": passed,
        "interventions_run": False,
    }
    return {**payload, "report_checksum": payload_checksum(payload)}


def text_replication_verdict(
    rows: Sequence[Mapping],
    *,
    primary_alpha: float = 1.0,
    task_set: Sequence[TextReplicationTask] | None = None,
) -> dict:
    """Gate later stages on the paper task, not a multimodal hope.

    ``task_set`` defaults to :func:`anthropic_text_tasks`.  Pass the run's own
    set whenever it differs, or every added row is silently dropped from both
    rates.

    The implicit family is scored as a **rate** over its members rather than by
    reading ``spider_to_ant_legs`` alone.  With the frozen set that family has
    exactly one member, so ``rate >= 0.5`` reduces to the previous "the single
    implicit task must swap" and the frozen verdicts are unchanged.
    """

    selected = anthropic_text_tasks() if task_set is None else tuple(task_set)
    tasks = {task.task_id: task for task in selected}
    by_task = {str(row["task_id"]): dict(row) for row in rows}
    missing = sorted(set(tasks) - set(by_task))
    clean = not missing and all(bool(by_task[name].get("clean_correct")) for name in tasks)
    def _swapped(row: Mapping) -> bool:
        return bool(
            row.get(
                "exact_primary_swapped_answer_generated",
                row.get(
                    "exact_alpha1_swapped_answer_generated",
                    row.get("exact_alpha1_target_top1"),
                ),
            )
        )

    implicit_rows = [
        by_task[name]
        for name, task in tasks.items()
        if task.family == "implicit_two_hop" and name in by_task
    ]
    implicit_rate = (
        sum(_swapped(row) for row in implicit_rows) / len(implicit_rows)
        if implicit_rows
        else 0.0
    )
    flexible_rows = [
        by_task[name] for name, task in tasks.items() if task.family == "flexible_function"
    ]
    flexible_rate = (
        sum(
            bool(
                row.get(
                    "exact_primary_swapped_answer_generated",
                    row.get(
                        "exact_alpha1_swapped_answer_generated",
                        row.get("exact_alpha1_target_top1"),
                    ),
                )
            )
            for row in flexible_rows
        )
        / len(flexible_rows)
        if flexible_rows
        else 0.0
    )
    controls = not missing and all(
        not bool(
            row.get(
                "random_primary_swapped_answer_generated",
                row.get(
                    "random_swapped_answer_generated",
                    row.get("random_target_top1"),
                ),
            )
        )
        and not bool(
            row.get(
                "unrelated_primary_swapped_answer_generated",
                row.get(
                    "unrelated_swapped_answer_generated",
                    row.get("unrelated_target_top1"),
                ),
            )
        )
        for row in by_task.values()
    )
    passed = (
        clean and implicit_rate >= 0.5 and flexible_rate >= 0.5 and controls
    )
    payload = {
        "version": PROTOCOL_VERSION,
        "verdict": "TEXT_PAPER_REPLICATION_GO" if passed else "TEXT_PAPER_REPLICATION_NO_GO",
        "primary_alpha": float(primary_alpha),
        "primary_alpha_role": (
            "exact_coordinate_exchange"
            if float(primary_alpha) == 1.0
            else "double_strength_coordinate_exchange"
            if float(primary_alpha) == 2.0
            else "amplified_coordinate_exchange"
        ),
        "all_clean_answers_correct": clean,
        "output_endpoint": "unrestricted_greedy_complete_answer",
        "implicit_two_hop_swapped_answer_rate": implicit_rate,
        "n_implicit_two_hop_tasks": len(implicit_rows),
        "n_flexible_function_tasks": len(flexible_rows),
        "flexible_function_swapped_answer_rate": flexible_rate,
        "matched_controls_pass": controls,
        "missing_tasks": missing,
        "multimodal_stage_licensed": passed,
    }
    return {**payload, "report_checksum": payload_checksum(payload)}


def _best_diagnostic_logprob(result: Mapping, name: str) -> float | None:
    values = []
    for step in result.get("full_vocabulary_diagnostic_trace") or ():
        token = dict(step.get("tokens") or {}).get(str(name))
        if token is not None:
            values.append(float(token["logprob"]))
    return max(values) if values else None


def text_swap_diagnostic_report(
    records: Sequence[Mapping],
    *,
    clean_rows: Sequence[Mapping],
    layers: Sequence[int],
    bands: Sequence[Sequence[int]] | None = None,
    task_set: Sequence[TextReplicationTask] | None = None,
) -> dict:
    """Summarize the predeclared layer/band diagnostic without claiming confirmation.

    ``task_set`` defaults to :func:`anthropic_text_tasks`; pass the run's own set
    or the diagnostic silently reports on a subset of what was measured.
    """

    selected_tasks = anthropic_text_tasks() if task_set is None else tuple(task_set)
    tasks = {task.task_id: task for task in selected_tasks}
    layer_set = set(map(int, layers))
    if bands is None:
        normalized_bands = text_diagnostic_bands(layers)
    else:
        normalized_bands = tuple(tuple(map(int, band)) for band in bands)
        if (
            not normalized_bands
            or any(not band for band in normalized_bands)
            or len(set(normalized_bands)) != len(normalized_bands)
            or any(not set(band).issubset(layer_set) for band in normalized_bands)
            or any(tuple(sorted(set(band))) != band for band in normalized_bands)
            or any(
                any(
                    right != left + 1
                    for left, right in zip(band, band[1:], strict=False)
                )
                for band in normalized_bands
            )
        ):
            raise WorkspaceReplicationRefused(
                "diagnostic bands must be unique, nonempty, contiguous, "
                "sorted subsets of the declared layers"
            )
    bands = normalized_bands
    expected = {
        (task_id, tuple(band), condition)
        for task_id in tasks
        for band in bands
        for condition in TEXT_DIAGNOSTIC_CONDITIONS
    }
    indexed = {
        (
            str(row.get("task_id")),
            tuple(map(int, row.get("band") or ())),
            str(row.get("condition")),
        ): dict(row)
        for row in records
    }
    missing = sorted(
        {
            f"{task}|{'-'.join(map(str, band))}|{condition}"
            for task, band, condition in expected - set(indexed)
        }
    )
    clean_by_task = {str(row["task_id"]): dict(row) for row in clean_rows}
    band_rows = []
    for band in bands:
        exact_successes = 0
        flexible_successes = 0
        implicit_success = False
        direct_successes = 0
        control_successes = {name: 0 for name in ("zero", "random_alpha1", "unrelated_alpha1")}
        coordinate_audit_passes = []
        direct_integrity_passes = []
        control_integrity_passes = []
        all_integrity_passes = []
        task_rows = []
        for task_id, task in tasks.items():
            cell = {
                condition: indexed.get((task_id, tuple(band), condition))
                for condition in TEXT_DIAGNOSTIC_CONDITIONS
            }
            exact_record = cell["exact_alpha1"]
            exact_result = dict((exact_record or {}).get("result") or {})
            exact_success = bool(exact_result.get("answer_match"))
            exact_successes += int(exact_success)
            if task.implicit_intermediate:
                implicit_success = exact_success
            elif exact_success:
                flexible_successes += 1
            direct_result = dict(
                (cell["direct_answer_norm_matched"] or {}).get("result") or {}
            )
            direct_successes += int(bool(direct_result.get("answer_match")))
            control_results = {}
            control_integrity = {}
            for control in control_successes:
                result = dict((cell[control] or {}).get("result") or {})
                success = bool(result.get("answer_match"))
                control_successes[control] += int(success)
                control_results[control] = success
                control_diagnostics = dict(
                    result.get("intervention_diagnostics") or {}
                )
                control_integrity[control] = bool(
                    control_diagnostics.get("all_hooks_fired")
                    and control_diagnostics.get("all_finite")
                )
            direct_diagnostics = dict(
                direct_result.get("intervention_diagnostics") or {}
            )
            direct_layer_rows = list(
                dict(direct_diagnostics.get("by_layer") or {}).values()
            )
            direct_integrity = bool(
                direct_diagnostics.get("all_hooks_fired")
                and direct_diagnostics.get("all_finite")
                and direct_diagnostics.get(
                    "all_model_dtype_realizations_converged", True
                )
                and direct_layer_rows
                and max(
                    float(row.get("max_relative_norm_match_error", math.inf))
                    for row in direct_layer_rows
                )
                <= TEXT_POST_CAST_MAX_RELATIVE_ERROR
            )
            diagnostics = dict(exact_result.get("intervention_diagnostics") or {})
            audit_pass = bool(
                diagnostics.get("all_hooks_fired")
                and diagnostics.get("all_finite")
                and diagnostics.get(
                    "all_layers_are_exact_alpha_one_exchange_before_cast"
                )
                and diagnostics.get(
                    "all_model_dtype_realizations_converged", True
                )
                and diagnostics.get("post_cast_audit_passed")
            )
            all_condition_integrity = bool(
                audit_pass
                and direct_integrity
                and all(control_integrity.values())
            )
            coordinate_audit_passes.append(audit_pass)
            direct_integrity_passes.append(direct_integrity)
            control_integrity_passes.append(all(control_integrity.values()))
            all_integrity_passes.append(all_condition_integrity)
            clean_result = dict((clean_by_task.get(task_id) or {}).get("clean") or {})
            clean_logprob = _best_diagnostic_logprob(
                clean_result, "swapped_answer_head"
            )
            exact_logprob = _best_diagnostic_logprob(
                exact_result, "swapped_answer_head"
            )
            task_rows.append(
                {
                    "task_id": task_id,
                    "family": task.family,
                    "exact_swapped_answer_generated": exact_success,
                    "direct_answer_swapped_answer_generated": bool(
                        direct_result.get("answer_match")
                    ),
                    "control_swapped_answer_generated": control_results,
                    "coordinate_audit_passed": audit_pass,
                    "control_integrity": control_integrity,
                    "direct_answer_integrity": direct_integrity,
                    "all_condition_integrity_passed": all_condition_integrity,
                    "swapped_answer_head_best_logprob_change": (
                        exact_logprob - clean_logprob
                        if exact_logprob is not None and clean_logprob is not None
                        else None
                    ),
                }
            )
        complete = all(
            (task_id, tuple(band), condition) in indexed
            for task_id in tasks
            for condition in TEXT_DIAGNOSTIC_CONDITIONS
        )
        flexible_rate = flexible_successes / 6.0
        controls_pass = all(count == 0 for count in control_successes.values())
        coordinate_audits_pass = bool(coordinate_audit_passes) and all(
            coordinate_audit_passes
        )
        direct_answer_integrity_pass = bool(direct_integrity_passes) and all(
            direct_integrity_passes
        )
        control_integrity_pass = bool(control_integrity_passes) and all(
            control_integrity_passes
        )
        all_condition_integrity_pass = bool(all_integrity_passes) and all(
            all_integrity_passes
        )
        positive_control_rate = direct_successes / len(tasks)
        eligible = bool(
            complete
            and all_condition_integrity_pass
            and controls_pass
            and implicit_success
            and flexible_rate >= 0.5
            and positive_control_rate >= 0.5
        )
        band_rows.append(
            {
                "band": list(band),
                "complete": complete,
                "coordinate_audits_pass": coordinate_audits_pass,
                "direct_answer_integrity_pass": direct_answer_integrity_pass,
                "control_integrity_pass": control_integrity_pass,
                "all_condition_integrity_pass": all_condition_integrity_pass,
                "implicit_two_hop_success": implicit_success,
                "flexible_function_success_rate": flexible_rate,
                "exact_successes": exact_successes,
                "direct_answer_positive_control_rate": positive_control_rate,
                "control_success_counts": control_successes,
                "matched_controls_pass": controls_pass,
                "eligible_for_fresh_confirmation": eligible,
                "tasks": task_rows,
            }
        )
    candidates = [row for row in band_rows if row["eligible_for_fresh_confirmation"]]
    candidates.sort(
        key=lambda row: (
            -int(row["exact_successes"]),
            -float(row["direct_answer_positive_control_rate"]),
            len(row["band"]),
            -int(row["band"][0]),
        )
    )
    bands_with_integrity_failures = [
        row["band"]
        for row in band_rows
        if row["complete"] and not row["all_condition_integrity_pass"]
    ]
    selected = candidates[0]["band"] if candidates and not missing else None
    verdict = (
        "TEXT_DIAGNOSTIC_INCOMPLETE"
        if missing
        else "TEXT_DIAGNOSTIC_ALPHA1_CANDIDATE_FOUND"
        if selected
        else "TEXT_DIAGNOSTIC_ENGINEERING_NO_GO"
        if len(bands_with_integrity_failures) == len(band_rows)
        else "TEXT_DIAGNOSTIC_NO_ALPHA1_CANDIDATE"
    )
    payload = {
        "version": TEXT_DIAGNOSTIC_VERSION,
        "verdict": verdict,
        "development_only": True,
        "primary_endpoint": "unrestricted_greedy_complete_answer",
        "teacher_forcing_used": False,
        "tested_alpha": 1.0,
        "tested_layers": list(map(int, layers)),
        "tested_bands": [list(band) for band in bands],
        "conditions": list(TEXT_DIAGNOSTIC_CONDITIONS),
        "post_cast_relative_error_threshold": TEXT_POST_CAST_MAX_RELATIVE_ERROR,
        "missing_units": missing,
        "bands_with_integrity_failures": bands_with_integrity_failures,
        "band_integrity_is_evaluated_independently": True,
        "bands": band_rows,
        "selected_band_for_fresh_confirmation": selected,
        "selection_rule": (
            "eligible requires audited exact alpha=1, implicit success, at least "
            "3/6 flexible-function successes, zero random/unrelated successes, "
            "and >=50% norm-matched direct-answer success; ties maximize exact "
            "then direct-answer successes, then prefer the shortest deeper band"
        ),
        "fresh_confirmation_required": selected is not None,
        "multimodal_stage_licensed": False,
    }
    return {**payload, "report_checksum": payload_checksum(payload)}


def freeze_confirmation_design(
    *,
    text_verdict: Mapping | None = None,
    text_diagnostic: Mapping | None = None,
    localization: Mapping,
    pair: Sequence[str],
    alpha: float = 1.0,
    sensitivity_alpha: float | None = 0.75,
    prompt_protocol: str,
    development_population_digest: str,
    prospective_loading_followup: Mapping | None = None,
) -> dict:
    """Freeze the confirmatory design before any fresh media are opened."""

    if text_diagnostic is not None:
        if text_diagnostic.get("verdict") != "TEXT_DIAGNOSTIC_ALPHA1_CANDIDATE_FOUND":
            raise WorkspaceReplicationRefused(
                "the audited text-only alpha=1 localization found no candidate; "
                "multimodal confirmation is blocked"
            )
        text_band = tuple(
            map(int, text_diagnostic.get("selected_band_for_fresh_confirmation") or ())
        )
        if not text_band:
            raise WorkspaceReplicationRefused(
                "the text diagnostic names no band for fresh confirmation"
            )
        text_evidence = {
            "version": text_diagnostic.get("version"),
            "verdict": text_diagnostic.get("verdict"),
            "report_checksum": text_diagnostic.get("report_checksum"),
            "selected_band": list(text_band),
        }
    else:
        if not text_verdict or text_verdict.get("verdict") not in {
            "TEXT_PAPER_REPLICATION_GO",
            "L21_TEXT_CONFIRMATION_GO",
        }:
            raise WorkspaceReplicationRefused(
                "the text-only paper replication did not pass; multimodal "
                "confirmation is blocked"
            )
        text_band = tuple(map(int, localization.get("selected_band") or ()))
        if prospective_loading_followup is not None:
            text_band = tuple(
                map(
                    int,
                    prospective_loading_followup.get("selected_band") or (),
                )
            )
            if not text_band:
                raise WorkspaceReplicationRefused(
                    "the prospective loading follow-up names no layer band"
                )
        text_evidence = {
            "version": text_verdict.get("version"),
            "verdict": text_verdict.get("verdict"),
            "report_checksum": text_verdict.get("report_checksum"),
            "selected_band": list(text_band),
        }
    if prospective_loading_followup is None:
        if localization.get("verdict") != "LOADING_LOCALIZATION_GO":
            raise WorkspaceReplicationRefused(
                "clean source loading did not license a layer band"
            )
    else:
        if localization.get("verdict") != "LOADING_LOCALIZATION_NO_GO":
            raise WorkspaceReplicationRefused(
                "the prospective follow-up must preserve the completed "
                "loading-localization NO_GO"
            )
        if prospective_loading_followup.get("causal_result_consulted") is not False:
            raise WorkspaceReplicationRefused(
                "the prospective follow-up was not frozen outcome-blind"
            )
        if prospective_loading_followup.get("multimodal_causal_outcomes_opened") is not False:
            raise WorkspaceReplicationRefused(
                "multimodal outcomes were opened before the follow-up froze"
            )
    eligible_layers = set(
        map(
            int,
            localization.get("eligible_layers")
            or localization.get("selected_band")
            or (),
        )
    )
    if (
        prospective_loading_followup is None
        and not set(text_band).issubset(eligible_layers)
    ):
        raise WorkspaceReplicationRefused(
            "the text-selected band is not cleanly source-loaded in every "
            f"required modality: band={list(text_band)}, eligible="
            f"{sorted(eligible_layers)}"
        )
    names = tuple(map(str, pair))
    if len(names) != 2 or names[0] == names[1]:
        raise WorkspaceReplicationRefused(f"confirmation needs two concepts, got {names}")
    payload = {
        "version": CONFIRMATION_VERSION,
        "pair": list(names),
        "primary_alpha": float(alpha),
        "primary_alpha_role": (
            "exact_coordinate_exchange"
            if float(alpha) == 1.0
            else "double_strength_coordinate_exchange"
            if float(alpha) == 2.0
            else "amplified_coordinate_exchange"
        ),
        "sensitivity_alpha": (
            float(sensitivity_alpha) if sensitivity_alpha is not None else None
        ),
        "sensitivity_alpha_role": (
            None
            if sensitivity_alpha is None
            else "exact_coordinate_exchange_sensitivity"
            if float(sensitivity_alpha) == 1.0
            else "interpolation_not_primary"
            if 0.0 < float(sensitivity_alpha) < 1.0
            else "amplified_sensitivity_not_primary"
        ),
        "layer_band": list(text_band),
        "text_causal_evidence": text_evidence,
        "position_rule": str(localization["position_rule"]),
        "position_rule_by_modality": dict(
            localization.get("position_rule_by_modality")
            or {"text": str(localization["position_rule"])}
        ),
        "prospective_loading_followup": (
            None
            if prospective_loading_followup is None
            else dict(prospective_loading_followup)
        ),
        "loading_gate_overridden": prospective_loading_followup is not None,
        "prompt_protocol": str(prompt_protocol),
        "development_population_digest": str(development_population_digest),
        "fresh_population_required": True,
        "development_images_forbidden": True,
        "teacher_forcing_used": False,
        "candidate_list_supplied": False,
        "selection_depended_on_causal_outcome": False,
    }
    return {**payload, "design_digest": payload_checksum(payload)}


def assert_fresh_population(
    confirmation_groups: Sequence[Mapping],
    *,
    forbidden_image_ids: Sequence[str],
    forbidden_group_ids: Sequence[str] = (),
) -> dict:
    """Prove that confirmation media were absent from development and prior runs."""

    images = [str(row["image_id"]) for row in confirmation_groups]
    groups = [str(row["group_id"]) for row in confirmation_groups]
    image_overlap = sorted(set(images) & set(map(str, forbidden_image_ids)))
    group_overlap = sorted(set(groups) & set(map(str, forbidden_group_ids)))
    if image_overlap or group_overlap:
        raise WorkspaceReplicationRefused(
            f"confirmation population is not fresh: images={image_overlap}, groups={group_overlap}"
        )
    payload = {
        "version": CONFIRMATION_VERSION,
        "n_groups": len(groups),
        "n_distinct_groups": len(set(groups)),
        "n_distinct_images": len(set(images)),
        "image_overlap": image_overlap,
        "group_overlap": group_overlap,
        "fresh": True,
    }
    return {**payload, "population_digest": payload_checksum(payload)}


def paired_binary_superiority(
    treatment: Sequence[bool], control: Sequence[bool]
) -> dict:
    """Exact one-sided paired sign test for two binary causal conditions."""

    if len(treatment) != len(control) or not treatment:
        raise WorkspaceReplicationRefused(
            "paired binary superiority needs two equal, non-empty sequences"
        )
    wins = sum(bool(a) and not bool(b) for a, b in zip(treatment, control, strict=True))
    losses = sum(not bool(a) and bool(b) for a, b in zip(treatment, control, strict=True))
    discordant = wins + losses
    pvalue = (
        sum(math.comb(discordant, k) for k in range(wins, discordant + 1))
        / (2**discordant)
        if discordant
        else 1.0
    )
    return {
        "n": len(treatment),
        "treatment_rate": sum(map(bool, treatment)) / len(treatment),
        "control_rate": sum(map(bool, control)) / len(control),
        "wins": wins,
        "losses": losses,
        "ties": len(treatment) - discordant,
        "one_sided_exact_p": float(pvalue),
    }


def holm_adjust(pvalues: Mapping[str, float]) -> dict[str, float]:
    """Holm step-down family-wise correction, returned in original keys."""

    ordered = sorted((float(value), str(name)) for name, value in pvalues.items())
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for index, (value, name) in enumerate(ordered):
        running = max(running, min(1.0, value * (total - index)))
        adjusted[name] = running
    return {str(name): adjusted[str(name)] for name in pvalues}


__all__ = [
    "CONFIRMATION_VERSION",
    "LOADING_VERSION",
    "LOCALIZATION_VERSION",
    "PROTOCOL_VERSION",
    "TEXT_INPUT_PROTOCOL_VERSION",
    "MULTIMODAL_INPUT_PROTOCOL_VERSION",
    "MULTIMODAL_COMPLETION_INSTRUCTION",
    "MULTIMODAL_MAX_NEW_TOKENS",
    "TEXT_COMPLETION_INSTRUCTION",
    "TEXT_ANSWER_MATCH_RULE",
    "TEXT_MAX_NEW_TOKENS",
    "TEXT_OUTPUT_ENDPOINT_VERSION",
    "TEXT_DIAGNOSTIC_CONDITIONS",
    "TEXT_DIAGNOSTIC_VERSION",
    "TEXT_POST_CAST_MAX_RELATIVE_ERROR",
    "TEXT_MODEL_DTYPE_REALIZATION",
    "TextReplicationTask",
    "WorkspaceReplicationRefused",
    "anthropic_text_tasks",
    "assert_fresh_population",
    "build_assistant_prefill_completion_inputs",
    "build_multimodal_assistant_prefill_inputs",
    "capture_source_loading",
    "completion_answer_matches",
    "freeze_confirmation_design",
    "freeze_loading_localization",
    "holm_adjust",
    "paired_binary_superiority",
    "select_pair_from_loading",
    "semantic_answer_concept",
    "summarize_direct_answer_diagnostics",
    "summarize_loading",
    "summarize_swap_diagnostics",
    "text_diagnostic_bands",
    "text_replication_verdict",
    "text_swap_diagnostic_report",
    "text_capability_verdict",
    "text_task_digest",
    "unrestricted_greedy_completion",
    "unrestricted_greedy_direct_answer_trial",
    "unrestricted_greedy_swap_trial",
]
