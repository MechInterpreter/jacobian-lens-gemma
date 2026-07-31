# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""The behavioral gate: can the model read the concept out of each channel?

Nothing about activations means anything for a concept the model cannot
recognize behaviorally, so this runs first. The question text and the answer
candidates are byte-identical across modalities; only the evidence channel
changes — written caption, image pixels, or the spoken recording.

Answers are scored, not generated: every candidate's **complete token
sequence** is scored by teacher-forced conditional log likelihood. Scoring
only the first token would silently compare prefixes ("bus" vs "bu|s"), so the
sequence sum is the decision statistic and the per-candidate token ids are
recorded next to it.

The same scoring function is reused under intervention in
:mod:`jlens.mmpilot.causal`, so clean and edited runs are measured the same way.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from jlens.mmpilot.backend import BuiltInputs, PilotBackend

#: Identical for every modality. The evidence is what changes, never the ask.
DEFAULT_QUESTION = (
    "Question: which one of these is present: {options}? "
    "Answer with exactly one word.\nAnswer:"
)


def build_question(concepts: Sequence[str], template: str = DEFAULT_QUESTION) -> str:
    """The shared question, with candidates listed in a fixed order."""
    return template.format(options=", ".join(sorted(concepts)))


def build_prompt(question: str, *, modality: str, caption: str | None = None) -> str:
    """Prompt text for one condition.

    The text condition carries the written caption; the image and
    spoken-audio conditions carry no transcript and no caption — the media is
    the only evidence. File names and dataset metadata are never included.
    """
    if modality == "text":
        if not caption:
            raise ValueError("the text condition needs a written caption")
        return f"Caption: {caption.strip()}\n{question}"
    if modality in ("image", "spoken_audio"):
        return question
    raise ValueError(f"unknown modality {modality!r}")


def candidate_token_ids(
    backend: PilotBackend, candidates: Sequence[str]
) -> dict[str, list[int]]:
    """Token ids for each candidate answer (leading space, no specials).

    The leading space matters: after ``Answer:`` the model continues with
    ``" dog"``, not ``"dog"``.
    """
    out: dict[str, list[int]] = {}
    for candidate in candidates:
        ids = backend.encode_candidate(f" {candidate}")
        if not ids:
            raise ValueError(f"candidate {candidate!r} encoded to zero tokens")
        out[candidate] = [int(i) for i in ids]
    return out


def _extend_tensors(tensors: Mapping, prompt_len: int, candidate: Sequence[int]) -> dict:
    """Append candidate token ids, extending only the text-length tensors.

    Media tensors (``pixel_values``, ``input_features``, their masks) are
    indexed by pixels or audio frames, not by text position, so they pass
    through untouched. Some media masks can coincidentally have the prompt
    width, so only explicitly named decoder-sequence fields are extended.
    """
    extended = dict(tensors)
    input_ids = tensors["input_ids"]
    suffix = torch.tensor(
        [list(candidate)], dtype=torch.long, device=input_ids.device
    )
    extended["input_ids"] = torch.cat([input_ids, suffix], dim=1)

    # Only named decoder-sequence fields may be extended. Media masks such as
    # Gemma 4's input_features_mask can coincidentally have prompt_len columns.
    for key in ("attention_mask", "token_type_ids", "mm_token_type_ids"):
        value = tensors.get(key)
        if not torch.is_tensor(value) or value.ndim != 2 or value.shape[1] != prompt_len:
            continue
        fill = 1 if key == "attention_mask" else 0
        pad = torch.full(
            (value.shape[0], len(candidate)), fill, dtype=value.dtype, device=value.device
        )
        extended[key] = torch.cat([value, pad], dim=1)

    position_ids = tensors.get("position_ids")
    if (
        torch.is_tensor(position_ids)
        and position_ids.ndim == 2
        and position_ids.shape[1] == prompt_len
    ):
        offsets = torch.arange(
            1,
            len(candidate) + 1,
            dtype=position_ids.dtype,
            device=position_ids.device,
        ).unsqueeze(0)
        extended["position_ids"] = torch.cat(
            [position_ids, position_ids[:, -1:] + offsets], dim=1
        )
    return extended


def score_candidate_sequences(
    backend: PilotBackend,
    inputs: BuiltInputs,
    candidate_ids: Mapping[str, Sequence[int]],
) -> dict[str, dict]:
    """Teacher-forced conditional log likelihood of each complete candidate.

    One forward pass per candidate (batch size 1). Returns, per candidate,
    ``sum_logprob`` (the decision statistic), ``mean_logprob`` (length-
    normalized, reported for context), ``n_tokens``, and ``token_ids``.
    """
    scores: dict[str, dict] = {}
    for candidate, ids in candidate_ids.items():
        tensors = _extend_tensors(inputs.tensors, inputs.prompt_len, ids)
        logits = backend.forward_logits(tensors)
        if logits.shape[1] != inputs.prompt_len + len(ids):
            raise RuntimeError(
                f"logits length {logits.shape[1]} does not match "
                f"prompt+candidate length {inputs.prompt_len + len(ids)}; the "
                "processor expanded tokens inside the model — refusing to score"
            )
        log_probs = torch.log_softmax(logits[0].float(), dim=-1)
        total = 0.0
        for offset, token_id in enumerate(ids):
            total += float(log_probs[inputs.prompt_len - 1 + offset, int(token_id)])
        scores[candidate] = {
            "sum_logprob": total,
            "mean_logprob": total / len(ids),
            "n_tokens": len(ids),
            "token_ids": [int(i) for i in ids],
        }
    return scores


def prediction_and_margin(scores: Mapping[str, Mapping], target: str) -> dict:
    """Argmax candidate and the target's margin over the best alternative."""
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1]["sum_logprob"], kv[0]))
    prediction = ranked[0][0]
    others = [value["sum_logprob"] for name, value in scores.items() if name != target]
    margin = scores[target]["sum_logprob"] - (max(others) if others else 0.0)
    return {
        "prediction": prediction,
        "correct": prediction == target,
        "target_score": scores[target]["sum_logprob"],
        "target_margin": margin,
        "ranking": [name for name, _ in ranked],
    }


def capability_record(
    backend: PilotBackend,
    inputs: BuiltInputs,
    *,
    sample_id: str,
    concept: str,
    group_id: str,
    candidate_ids: Mapping[str, Sequence[int]],
) -> dict:
    """One capability-gate result, in the shape the unit store persists."""
    scores = score_candidate_sequences(backend, inputs, candidate_ids)
    verdict = prediction_and_margin(scores, concept)
    return {
        "sample_id": sample_id,
        "group_id": group_id,
        "concept": concept,
        "modality": inputs.modality,
        "prompt_hash": inputs.prompt_hash,
        "media_checksum": inputs.media_checksum,
        "prompt_len": inputs.prompt_len,
        "modality_token_range": inputs.modality_token_range,
        "candidate_token_ids": {k: list(v) for k, v in candidate_ids.items()},
        "candidate_scores": scores,
        "all_candidates_single_token": all(len(v) == 1 for v in candidate_ids.values()),
        **verdict,
    }


def capability_summary(
    records: Sequence[Mapping], *, threshold: float = 0.7, modalities: Sequence[str]
) -> dict:
    """Per-(concept, modality) accuracy and which concepts survive the gate.

    A concept is retained only if it clears ``threshold`` in **every**
    available modality. Modalities that were blocked upstream are absent from
    ``modalities`` and cannot veto a concept — they are reported as blocked
    instead.

    The pilot is small enough that a percentage over 4–8 samples is unstable,
    so the per-sample margins are carried through for the notebook to print.
    """
    cells: dict[tuple[str, str], list[Mapping]] = {}
    for record in records:
        cells.setdefault((record["concept"], record["modality"]), []).append(record)

    accuracy: dict[str, dict[str, dict]] = {}
    for (concept, modality), rows in sorted(cells.items()):
        correct = sum(1 for row in rows if row["correct"])
        margins = sorted(float(row["target_margin"]) for row in rows)
        accuracy.setdefault(concept, {})[modality] = {
            "n": len(rows),
            "n_correct": correct,
            "accuracy": correct / len(rows) if rows else 0.0,
            "median_target_margin": margins[len(margins) // 2] if margins else None,
            "min_target_margin": margins[0] if margins else None,
            "passed": bool(rows) and correct / len(rows) >= threshold,
        }

    retained = sorted(
        concept
        for concept, per_modality in accuracy.items()
        if all(
            per_modality.get(modality, {}).get("passed", False) for modality in modalities
        )
    )
    text_image_retained = sorted(
        concept
        for concept, per_modality in accuracy.items()
        if all(
            per_modality.get(modality, {}).get("passed", False)
            for modality in ("text", "image")
            if modality in modalities
        )
    )
    return {
        "threshold": threshold,
        "modalities_evaluated": list(modalities),
        "per_concept": accuracy,
        "retained_concepts": retained,
        "text_image_retained_concepts": text_image_retained,
        "n_records": len(records),
    }
