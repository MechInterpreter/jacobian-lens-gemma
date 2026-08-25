# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""The real-model construction path, as package code instead of notebook code.

Three L4 starts died on the same class of defect: a notebook cell that only
executes when ``RUN_REAL_ROBUSTNESS`` is True calling an API that had drifted —
a renamed class, a return value unpacked as something it is not, a required
argument never passed. MOCK execution cannot catch those, because the MOCK
branch never runs the real branch's lines; string-matching tests cannot catch
them, because the strings look fine.

The repair is structural: the real branch is now a thin call into this module,
and this module is exercised by tests that monkeypatch only the two hooks that
touch the network (:func:`_loader` and :func:`_processor_factory`) and then run
the **actual construction code** — the same unpacking, the same freezing, the
same interface resolution, the same :class:`GemmaPilotBackend` constructor the
L4 will hit.

The sequence reproduced here is the four-concept pilot notebook's real path,
which completed a full run against the real checkpoint. It is copied, not
reinvented:

1. ``load_gemma4`` → ``(Gemma4LensModel, load_info)`` — **not** a processor.
2. Freeze every parameter of the wrapped HF model.
3. ``verify_architecture`` against the expected depth/width/vocab.
4. ``AutoProcessor.from_pretrained`` at the **resolved** revision.
5. ``resolve_processor_interface`` by inspection of the processor.
6. ``GemmaPilotBackend(hf_model, processor, interface, device=...)``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from jlens.mmpilot.backend import GemmaPilotBackend, resolve_processor_interface

#: Gemma 4 E4B IT. ``verify_architecture`` hard-fails on any mismatch.
EXPECT_N_LAYERS = 42
EXPECT_D_MODEL = 2560
EXPECT_VOCAB = 262144


class LensManifestError(RuntimeError):
    """The published lens's manifest is missing or names a different setup."""


def _loader():
    """Import hook: the ~16 GB model loader. Patched by the fake-path tests."""
    from jlens.gemma4 import load_gemma4, verify_architecture

    return load_gemma4, verify_architecture


def _processor_factory():
    """Import hook: the processor loader. Patched by the fake-path tests."""
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained


@dataclass
class RealBackendBundle:
    """Everything section 8 needs to record about what was actually loaded."""

    backend: GemmaPilotBackend
    lens_model: Any
    processor: Any
    interface: dict
    load_info: dict
    architecture: dict
    model_revision: str
    processor_revision: str
    device: str
    #: The probed native spoken-audio protocol, when ``resolve_audio`` was
    #: requested and succeeded. ``None`` leaves ``spoken_audio`` blocked.
    audio_interface: Any = None
    #: Why audio is blocked, when it is. ``""`` when audio was not requested.
    audio_blocked_reason: str = ""


@dataclass
class ProcessorOnlyBundle:
    """A processor and tokenization-only backend, with no model weights."""

    backend: Any
    processor: Any
    processor_revision: str


class TokenizerOnlyBackend:
    """The token methods required by endpoint preflight, and nothing else."""

    def __init__(self, processor: Any) -> None:
        self.processor = processor
        self.tokenizer = getattr(processor, "tokenizer", processor)

    def encode_candidate(self, text: str) -> list[int]:
        return list(
            self.tokenizer(
                text, add_special_tokens=False, return_tensors=None
            )["input_ids"]
        )

    def decode_token(self, token_id: int) -> str:
        return str(
            self.tokenizer.decode([int(token_id)], skip_special_tokens=False)
        )


def build_processor_backend(
    repo_id: str,
    *,
    revision: str,
    token: str | None = None,
) -> ProcessorOnlyBundle:
    """Load only the pinned processor/tokenizer; never call the model loader."""
    processor = _processor_factory()(repo_id, revision=revision, token=token)
    return ProcessorOnlyBundle(
        backend=TokenizerOnlyBackend(processor),
        processor=processor,
        processor_revision=str(revision),
    )


def build_real_backend(
    repo_id: str,
    *,
    revision: str,
    token: str | None = None,
    device: str | None = None,
    allow_model_load: bool = False,
    expect_n_layers: int = EXPECT_N_LAYERS,
    expect_d_model: int = EXPECT_D_MODEL,
    expect_vocab_size: int = EXPECT_VOCAB,
    resolve_audio: bool = False,
    dtype: torch.dtype = torch.bfloat16,
) -> RealBackendBundle:
    """Load, audit, and wrap the real Gemma 4 checkpoint.

    Args:
        allow_model_load: Passed through to :func:`jlens.gemma4.load_gemma4`,
            which refuses the ~16 GB download without it. Deliberately not
            defaulted to True here: the notebook states it explicitly, so the
            guard is visible at the call site rather than buried.
        resolve_audio: Probe the native spoken-audio protocol with
            :func:`jlens.mmpilot.audio.resolve_audio_interface` and attach it to
            the backend. Defaults to False, which leaves ``spoken_audio``
            blocked. Every completed run so far was text-and-image; turning this
            on changes what the backend will accept, so it is stated at the call
            site rather than inferred from the checkpoint having an audio tower.
        dtype: Model weight dtype. Defaults to ``torch.bfloat16`` (the
            checkpoint's native dtype and every study before the cat->dog
            animal-sound one). Passing ``torch.float32`` here is the whole of
            what "run in fp32" means; there is no bf16 fallback anywhere in
            this function if an fp32 load fails or does not fit — a caller
            that wants fp32 must clear
            :func:`jlens.mmpilot.fp32_preflight.preflight_fp32_or_refuse`
            first and pass the same dtype it checked.

    Raises:
        RuntimeError: From ``load_gemma4`` when ``allow_model_load`` is False.
        ValueError: From ``verify_architecture`` on any depth/width/vocab
            mismatch or a trainable parameter.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    load_gemma4, verify_architecture = _loader()

    lens_model, load_info = load_gemma4(
        repo_id,
        revision=revision,
        dtype=dtype,
        device_map=device if device != "cpu" else None,
        allow_model_load=allow_model_load,
        token=token,
    )
    hf_model = lens_model._hf_model
    for parameter in hf_model.parameters():
        parameter.requires_grad_(False)

    architecture = verify_architecture(
        lens_model,
        expect_n_layers=expect_n_layers,
        expect_d_model=expect_d_model,
        expect_vocab_size=expect_vocab_size,
    ).to_dict()

    resolved_revision = str(load_info["model_revision"])
    processor = _processor_factory()(
        repo_id, revision=resolved_revision, token=token
    )
    interface = resolve_processor_interface(processor, hf_model.config)

    audio_interface = None
    audio_blocked_reason = ""
    if resolve_audio:
        from jlens.mmpilot.audio import (
            SpokenAudioUnsupportedError,
            resolve_audio_interface,
        )

        try:
            audio_interface = resolve_audio_interface(
                processor,
                hf_model.config,
                model_repo_id=repo_id,
                model_revision=resolved_revision,
                processor_revision=resolved_revision,
            )
        except SpokenAudioUnsupportedError as exc:
            # A blocked channel is a recorded result, never a crashed run: the
            # text-and-image study must still be able to proceed.
            audio_blocked_reason = str(exc)

    backend = GemmaPilotBackend(
        hf_model,
        processor,
        interface,
        device=device,
        audio_interface=audio_interface,
    )
    return RealBackendBundle(
        backend=backend,
        lens_model=lens_model,
        processor=processor,
        interface=interface,
        load_info=dict(load_info),
        architecture=architecture,
        model_revision=resolved_revision,
        processor_revision=resolved_revision,
        device=device,
        audio_interface=audio_interface,
        audio_blocked_reason=audio_blocked_reason,
    )


# ------------------------------------------------------------ the frozen lens


@dataclass
class ValidatedLens:
    """A frozen lens plus the native validation that published it."""

    lens: Any
    path: str
    checksum: str
    native_validation: dict = field(default_factory=dict)


def load_validated_lens(
    lens_path: str | Path,
    *,
    expect_checksum: str,
    layers: tuple[int, ...] | list[int],
    model_revision: str,
) -> ValidatedLens:
    """Load a published lens and hold it to its own manifest.

    ``lens.validated.pt`` is only published next to a
    ``validated_lens_manifest.json`` recording the checksum, the model revision
    it was validated against, and the layers that passed the held-out native
    readout. All of that is re-checked here, because a lens used with a model
    it was never validated for answers a different question than the one the
    manifest certifies.

    Raises:
        LensManifestError: Missing lens, checksum mismatch against the pin,
            missing manifest, or a manifest naming a different status,
            checksum, revision, or layer set.
    """
    from jlens.lens import JacobianLens
    from jlens.metadata import file_sha256

    lens_path = Path(lens_path)
    if not lens_path.is_file():
        raise LensManifestError(
            f"the frozen validated lens was not found at {lens_path}. This "
            "study does not fit a lens; point LENS_PATH at the published "
            "artifact."
        )
    checksum = file_sha256(str(lens_path))
    if checksum != expect_checksum:
        raise LensManifestError(
            f"lens checksum {checksum} != pinned {expect_checksum}; refusing "
            "to use a lens other than the validated one"
        )
    manifest_path = lens_path.parent / "validated_lens_manifest.json"
    if not manifest_path.is_file():
        raise LensManifestError(
            f"missing {manifest_path}: a lens without its validation manifest "
            "is an unvalidated lens"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    problems = []
    if manifest.get("status") != "validated_text_only":
        problems.append(f"manifest status is {manifest.get('status')!r}")
    if manifest.get("lens_checksum") != checksum:
        problems.append("manifest checksum does not match the lens file")
    if manifest.get("model_revision") != model_revision:
        problems.append(
            f"manifest model revision {manifest.get('model_revision')!r} does "
            f"not match the loaded model revision {model_revision!r}"
        )
    passing = set(manifest.get("native_readout_layers_passing", []))
    if not set(int(layer) for layer in layers) <= passing:
        problems.append(
            f"requested layers {sorted(int(x) for x in layers)}; natively "
            f"validated layers {sorted(passing)}"
        )
    if problems:
        raise LensManifestError(
            "incompatible published lens: " + "; ".join(problems)
        )
    lens = JacobianLens.load(str(lens_path))
    return ValidatedLens(
        lens=lens,
        path=str(lens_path),
        checksum=checksum,
        native_validation=manifest,
    )


__all__ = [
    "EXPECT_D_MODEL",
    "EXPECT_N_LAYERS",
    "EXPECT_VOCAB",
    "LensManifestError",
    "ProcessorOnlyBundle",
    "RealBackendBundle",
    "TokenizerOnlyBackend",
    "ValidatedLens",
    "build_processor_backend",
    "build_real_backend",
    "load_validated_lens",
]
