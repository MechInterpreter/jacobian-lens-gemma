# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""The narrow model interface the pilot runs against.

Everything downstream — capability scoring, activation capture, interventions
— talks to a :class:`PilotBackend`. There are two implementations: the real
Gemma 4 E4B one here, and the deterministic synthetic one in
:mod:`jlens.mmpilot.mock`. That split is what lets the whole pipeline be
tested on CPU without a 16 GB download.

Nothing in this module assumes the checkpoint supports audio. The processor
interface is *resolved by inspection* (:func:`resolve_processor_interface`)
and an unsupported modality raises :class:`ModalityUnsupportedError`, which the
notebook records as a blocked condition. Speech is never substituted with its
transcript.

Two invariance checks live here because they are properties of the hook
mechanics rather than of any experiment: :func:`check_capture_noop` (a capture
hook must not change logits) and :func:`check_zero_intervention` (a
coefficient-zero edit must reproduce the clean result). The notebook refuses
to intervene unless both pass.
"""

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import torch
from torch import nn

from jlens.hooks import ActivationRecorder
from jlens.interventions import residual_intervention

#: Modality names used everywhere in the pilot. ``spoken_audio`` is a *spoken
#: caption*, i.e. linguistic audio — never environmental sound.
MODALITIES = ("text", "image", "spoken_audio")


class ModalityUnsupportedError(RuntimeError):
    """The loaded checkpoint/processor cannot take this evidence channel."""


class InvarianceError(RuntimeError):
    """A no-op hook or a zero-coefficient edit changed the model's output."""


@dataclass
class BuiltInputs:
    """One prepared forward-pass input.

    Attributes:
        tensors: Keyword arguments for the model's ``forward``.
        prompt_len: Number of prompt tokens; the final prompt token — the
            position everything is measured and edited at — is
            ``prompt_len - 1``.
        modality: One of :data:`MODALITIES`.
        prompt_hash: ``sha256`` of the prompt text (first 16 hex chars).
        media_checksum: ``sha256`` of the image/audio bytes, or ``None`` for text.
        route: How the inputs were built (processor call vs chat template).
        modality_token_range: ``[start, end)`` of the modality's token span
            when the processor exposes one contiguous run, else ``None``.
    """

    tensors: dict
    prompt_len: int
    modality: str
    prompt_hash: str = ""
    media_checksum: str | None = None
    route: dict = field(default_factory=dict)
    modality_token_range: list[int] | None = None

    @property
    def final_prompt_position(self) -> int:
        return self.prompt_len - 1


@runtime_checkable
class PilotBackend(Protocol):
    """What the pilot needs from a model. Deliberately tiny."""

    d_model: int
    n_layers: int

    @property
    def blocks(self) -> Sequence[nn.Module]:
        """Decoder blocks, for hooks and interventions."""

    def supports(self, modality: str) -> bool: ...

    def build_inputs(
        self,
        *,
        prompt: str,
        modality: str,
        image: Any = None,
        audio: Any = None,
        sampling_rate: int | None = None,
    ) -> BuiltInputs: ...

    def forward_logits(self, tensors: Mapping) -> torch.Tensor:
        """``[1, seq, vocab]`` logits for one prepared input."""

    def encode_candidate(self, text: str) -> list[int]:
        """Token ids for a candidate continuation (no BOS, no specials)."""

    def unembedding_weight(self) -> torch.Tensor:
        """``[vocab, d_model]`` ``W_U`` — read-only, for the J-space dictionary."""


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def file_checksum(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


# ------------------------------------------------------- processor probing


def resolve_processor_interface(processor: Any, hf_config: Any) -> dict:
    """Report, by inspection, which evidence channels the processor accepts.

    Naming and documentation are not evidence: this looks at the processor's
    actual ``__call__`` signature and its component sub-processors. The result
    is recorded verbatim in the run manifest, and ``supports_audio=False`` is a
    hard block on the spoken-audio condition rather than a reason to fall back
    to transcripts.
    """
    call_params = list(inspect.signature(processor.__call__).parameters)
    audio_kwarg = next(
        (k for k in ("audio", "audios", "raw_audio", "raw_speech") if k in call_params),
        None,
    )
    components = {
        name: type(getattr(processor, name)).__name__
        for name in (
            "tokenizer",
            "image_processor",
            "feature_extractor",
            "audio_processor",
            "video_processor",
        )
        if getattr(processor, name, None) is not None
    }
    has_audio_component = any(
        key in components for key in ("feature_extractor", "audio_processor")
    )
    return {
        "processor_class": type(processor).__name__,
        "components": components,
        "call_parameters": call_params,
        "audio_kwarg": audio_kwarg,
        "supports_image": "image_processor" in components
        and any(k in call_params for k in ("images", "image")),
        "supports_audio": bool(audio_kwarg) and has_audio_component,
        "has_chat_template": getattr(
            getattr(processor, "tokenizer", None), "chat_template", None
        )
        is not None
        or getattr(processor, "chat_template", None) is not None,
        "image_token_id": getattr(hf_config, "image_token_id", None),
        "audio_token_id": getattr(hf_config, "audio_token_id", None),
        "audio_tower_present": hasattr(hf_config, "audio_config"),
        "vision_tower_present": hasattr(hf_config, "vision_config"),
    }


def contiguous_token_range(input_ids: torch.Tensor, token_id: int | None) -> list[int] | None:
    """``[start, end)`` of a contiguous run of ``token_id``; ``None`` otherwise.

    Non-contiguous or absent runs are recorded as ``None`` rather than guessed.
    """
    if token_id is None:
        return None
    hits = (input_ids[0] == int(token_id)).nonzero().flatten().tolist()
    if not hits or hits[-1] - hits[0] + 1 != len(hits):
        return None
    return [int(hits[0]), int(hits[-1]) + 1]


# ------------------------------------------------------------ real backend


class GemmaPilotBackend:
    """:class:`PilotBackend` over a loaded ``Gemma4ForConditionalGeneration``.

    Args:
        hf_model: The loaded multimodal model (frozen by the caller).
        processor: ``AutoProcessor`` for the same revision.
        interface: Result of :func:`resolve_processor_interface`.
        device: Where inputs are moved before the forward pass.
    """

    def __init__(
        self,
        hf_model: nn.Module,
        processor: Any,
        interface: Mapping,
        *,
        device: torch.device | str = "cuda",
    ) -> None:
        self.hf_model = hf_model
        self.processor = processor
        self.interface = dict(interface)
        self.device = torch.device(device)
        self.tokenizer = getattr(processor, "tokenizer", processor)
        text_config = hf_model.config.get_text_config()
        self.d_model = int(text_config.hidden_size)
        self.n_layers = int(text_config.num_hidden_layers)

    @property
    def blocks(self) -> Sequence[nn.Module]:
        return self.hf_model.model.language_model.layers

    def supports(self, modality: str) -> bool:
        if modality == "text":
            return True
        if modality == "image":
            return bool(self.interface["supports_image"])
        if modality == "spoken_audio":
            return bool(self.interface["supports_audio"])
        raise ValueError(f"unknown modality {modality!r}")

    def unembedding_weight(self) -> torch.Tensor:
        return self.hf_model.lm_head.weight

    def encode_candidate(self, text: str) -> list[int]:
        return list(
            self.tokenizer(text, add_special_tokens=False, return_tensors=None)["input_ids"]
        )

    def build_inputs(
        self,
        *,
        prompt: str,
        modality: str,
        image: Any = None,
        audio: Any = None,
        sampling_rate: int | None = None,
        media_path: str | None = None,
    ) -> BuiltInputs:
        if not self.supports(modality):
            raise ModalityUnsupportedError(
                f"the loaded processor does not accept {modality!r} "
                f"(interface: {self.interface})"
            )
        kwargs: dict[str, Any] = {"text": prompt, "return_tensors": "pt"}
        if image is not None:
            kwargs["images"] = image
        if audio is not None:
            kwargs[self.interface["audio_kwarg"]] = audio
            if sampling_rate is not None and "sampling_rate" in self.interface["call_parameters"]:
                kwargs["sampling_rate"] = sampling_rate
        if not self.interface["has_chat_template"]:
            encoded = self.processor(**kwargs)
            route = {"route": "processor_call", "kwargs": sorted(k for k in kwargs if k != "text")}
        else:
            content: list[dict] = []
            if image is not None:
                content.append({"type": "image", "image": image})
            if audio is not None:
                content.append({"type": "audio", "audio": audio})
            content.append({"type": "text", "text": prompt})
            encoded = self.processor.apply_chat_template(
                [{"role": "user", "content": content}],
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            route = {"route": "chat_template", "add_generation_prompt": True}

        tensors = {
            key: (value.to(self.device) if torch.is_tensor(value) else value)
            for key, value in dict(encoded).items()
        }
        tensors.setdefault("use_cache", False)
        input_ids = tensors["input_ids"]
        token_id = (
            self.interface["image_token_id"]
            if modality == "image"
            else self.interface["audio_token_id"]
            if modality == "spoken_audio"
            else None
        )
        return BuiltInputs(
            tensors=tensors,
            prompt_len=int(input_ids.shape[1]),
            modality=modality,
            prompt_hash=text_hash(prompt),
            media_checksum=file_checksum(media_path) if media_path else None,
            route=route,
            modality_token_range=contiguous_token_range(input_ids, token_id),
        )

    @torch.no_grad()
    def forward_logits(self, tensors: Mapping) -> torch.Tensor:
        output = self.hf_model(**dict(tensors))
        return output.logits.float()


# ------------------------------------------------------- invariance checks


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max())


def check_capture_noop(
    backend: PilotBackend, inputs: BuiltInputs, layers: Sequence[int], *, tolerance: float = 1e-4
) -> dict:
    """A capture hook must not perturb the forward pass.

    Runs the same input with and without an :class:`ActivationRecorder` on
    ``layers`` and compares logits.

    Raises:
        InvarianceError: If the maximum absolute logit difference exceeds
            ``tolerance``. Capturing is unsafe if this fails, so nothing
            downstream should run.
    """
    clean = backend.forward_logits(inputs.tensors)
    with ActivationRecorder(backend.blocks, at=list(layers)) as recorder:
        hooked = backend.forward_logits(inputs.tensors)
        captured = sorted(recorder.activations)
    diff = _max_abs_diff(clean, hooked)
    report = {
        "check": "capture_noop",
        "layers": list(layers),
        "captured_layers": captured,
        "max_abs_logit_diff": diff,
        "tolerance": tolerance,
        "passed": diff <= tolerance,
    }
    if not report["passed"]:
        raise InvarianceError(
            f"capture hook changed logits by {diff:.3e} (> {tolerance:.1e}); "
            "refusing to capture or intervene"
        )
    return report


def check_zero_intervention(
    backend: PilotBackend,
    inputs: BuiltInputs,
    layer: int,
    *,
    position: int | None = None,
    tolerance: float = 1e-4,
) -> dict:
    """A coefficient-zero edit must reproduce the clean forward pass exactly.

    Exercises the real intervention path (hook installed, delta present,
    ``multiplier=0``) so a bug in indexing or dtype handling shows up here
    rather than as a "small effect" later.

    Raises:
        InvarianceError: If logits move by more than ``tolerance``.
    """
    position = inputs.final_prompt_position if position is None else position
    clean = backend.forward_logits(inputs.tensors)
    delta = torch.ones(backend.d_model, dtype=torch.float32)
    with residual_intervention(
        backend.blocks, layer, position=position, delta=delta, multiplier=0.0
    ) as stats:
        edited = backend.forward_logits(inputs.tensors)
    diff = _max_abs_diff(clean, edited)
    report = {
        "check": "zero_coefficient_intervention",
        "layer": layer,
        "position": position,
        "resolved_position": stats["resolved_position"],
        "activation_norm": stats["activation_norm"],
        "max_abs_logit_diff": diff,
        "tolerance": tolerance,
        "passed": diff <= tolerance,
    }
    if not report["passed"]:
        raise InvarianceError(
            f"zero-coefficient intervention changed logits by {diff:.3e} "
            f"(> {tolerance:.1e}); refusing to intervene"
        )
    return report


def run_invariance_gate(
    backend: PilotBackend,
    inputs: BuiltInputs,
    layers: Sequence[int],
    *,
    tolerance: float = 1e-4,
) -> dict:
    """Both mandatory checks. Any failure raises :class:`InvarianceError`."""
    capture = check_capture_noop(backend, inputs, layers, tolerance=tolerance)
    zero = [
        check_zero_intervention(backend, inputs, layer, tolerance=tolerance)
        for layer in layers
    ]
    return {"capture_noop": capture, "zero_intervention": zero, "passed": True}
