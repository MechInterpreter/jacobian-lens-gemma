# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""The native spoken-audio input path, resolved by probe rather than by belief.

Why this module exists
======================

The pilot recorded ``spoken_audio`` as blocked with the reason *"processor/model
produced audio features but zero audio placeholder tokens"*. That is a real,
reproducible state of the pinned checkpoint — and it is **not** a statement that
the checkpoint lacks audio. It is what ``Gemma4Processor`` does when the text it
is given contains no audio placeholder:

    >>> processor(text="Answer yes or no.", audio=waveform)   # doctest: +SKIP
    {'input_ids': (1, 11), 'input_features': (1, 99, 128), ...}   # 0 placeholders

``_process_audio`` runs and returns real mel features, but
:meth:`~transformers.ProcessorMixin.get_text_with_replacements` only expands
occurrences of ``processor.audio_token`` that are *already in the text*. Nothing
raises: ``Gemma4Processor.validate_inputs`` checks image-token/image count
consistency but has no audio equivalent, and ``_check_special_mm_tokens`` only
compares text-side and id-side counts, which are both zero. The mismatch
surfaces much later, inside ``Gemma4Model.forward``, as
``Audio features and audio tokens do not match, tokens: 0, features: N``.

So the failure was a *calling-convention* defect, not a capability limit. The
supported native path is the chat-template content block, which renders
``<|audio|>`` and lets the processor expand it into
``<|audio>`` + N × ``<|audio|>`` + ``<audio|>``, with N derived from the audio's
own duration.

What this module guarantees
===========================

:func:`resolve_audio_interface` never concludes anything from a component being
*present*. It pushes a tiny generated waveform through the real processor and
requires the placeholder span to actually appear, at two different durations, or
it raises :class:`SpokenAudioUnsupportedError` naming which link in the chain
broke. :func:`verify_audio_encoding` then re-checks every prepared input against
the features it was built from, so the three fatal states — features without
placeholders, placeholders without features, and a count mismatch between them —
are refusals here rather than a confusing error inside the model.

Nothing in this module transcribes, captions, resamples, or renames anything.
The waveform and the shared question are the only things the model ever sees.
"""

from __future__ import annotations

import hashlib
import inspect
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: Bumped whenever the resolved protocol's *meaning* changes. It is part of
#: :attr:`ResolvedAudioInterface.protocol_fingerprint`, so a run bound to one
#: protocol version refuses to resume under another.
AUDIO_PROTOCOL_VERSION = "jlens.mmpilot.native_spoken_audio.v1"

#: The modality name, fixed. Never "audio", never "speech".
SPOKEN_AUDIO = "spoken_audio"

#: The only supported convention. A content block, not a bare processor call:
#: the bare call is exactly what produced zero placeholders.
CALL_CONVENTION = "chat_template_audio_content_block"

#: The content-block schema handed to ``apply_chat_template``. The waveform goes
#: under ``"audio"`` as a float32 mono ndarray — never a path, never a URL, so
#: no filename can reach the model.
CONTENT_BLOCK_SCHEMA = {"type": "audio", "audio": "<float32 mono 1-D ndarray>"}

#: Tensor keys the feature extractor is expected to produce.
FEATURE_KEYS = ("input_features", "input_features_mask")

#: Durations, in seconds, of the probe waveforms. Two distinct lengths, because
#: one length cannot show that placeholder count tracks the audio.
PROBE_DURATIONS_S = (0.5, 1.0)

#: The probe's question. Deliberately free of any concept, caption or filename.
PROBE_PROMPT = "Answer with exactly one word."


class SpokenAudioUnsupportedError(RuntimeError):
    """The native spoken-audio path is not usable, with the exact broken link.

    Attributes:
        reason: A stable machine-readable code (see :data:`REASONS`), so the
            feasibility report can distinguish "this checkpoint has no audio
            tower" from "this call built an input the model would reject".
        detail: Everything observed, for the report.
    """

    def __init__(self, reason: str, message: str, **detail: Any) -> None:
        super().__init__(f"[{reason}] {message}")
        self.reason = reason
        self.detail = dict(detail)


#: Every refusal code this module can raise. Listed so the notebook's report can
#: enumerate them rather than string-matching messages.
REASONS = (
    "no_audio_call_parameter",
    "no_feature_extractor",
    "no_audio_token",
    "no_audio_token_id",
    "no_chat_template",
    "no_audio_tower",
    "sampling_rate_mismatch",
    "invalid_waveform",
    "features_without_placeholders",
    "placeholders_without_features",
    "no_audio_pathway_engaged",
    "placeholder_feature_count_mismatch",
    "non_contiguous_placeholder_span",
    "probe_failed",
)


# --------------------------------------------------------------- the waveform


@dataclass(frozen=True)
class PreparedWaveform:
    """A waveform in exactly the form the feature extractor expects."""

    samples: np.ndarray
    sampling_rate: int
    n_samples: int
    duration_s: float
    n_channels_in: int
    dtype_in: str
    checksum: str

    def to_dict(self) -> dict:
        return {
            "sampling_rate": self.sampling_rate,
            "n_samples": self.n_samples,
            "duration_s": round(self.duration_s, 6),
            "n_channels_in": self.n_channels_in,
            "dtype_in": self.dtype_in,
            "dtype_out": "float32",
            "ndim_out": 1,
            "checksum": self.checksum,
        }


def prepare_waveform(
    waveform: Any, sampling_rate: int, *, expected_rate: int
) -> PreparedWaveform:
    """Coerce ``waveform`` to float32 mono at ``expected_rate``, or refuse.

    Multi-channel input is averaged down to mono, which is a documented
    reduction of the same recording. A **sample-rate mismatch is refused, not
    resampled**: ``load_audio`` passes an ndarray through untouched, so a 22 kHz
    array handed to a 16 kHz feature extractor is silently reinterpreted as a
    slower, lower-pitched 16 kHz recording. That is a change to the evidence,
    and it would be invisible in every downstream number.

    Raises:
        SpokenAudioUnsupportedError: On a rate mismatch, an empty waveform, more
            than two dimensions, or any non-finite sample.
    """
    array = np.asarray(waveform)
    dtype_in = str(array.dtype)
    if array.ndim > 2:
        raise SpokenAudioUnsupportedError(
            "invalid_waveform",
            f"waveform has {array.ndim} dimensions; expected 1 (mono) or 2 "
            "(channels), so this is not a single recording",
            shape=list(array.shape),
        )
    n_channels_in = 1 if array.ndim == 1 else int(min(array.shape))
    if array.ndim == 2:
        # soundfile returns (frames, channels); some loaders transpose it.
        axis = 1 if array.shape[0] >= array.shape[1] else 0
        array = array.mean(axis=axis)
    array = np.ascontiguousarray(array, dtype=np.float32)
    if array.size == 0:
        raise SpokenAudioUnsupportedError(
            "invalid_waveform", "waveform is empty; there is no evidence to present"
        )
    if not np.isfinite(array).all():
        raise SpokenAudioUnsupportedError(
            "invalid_waveform",
            "waveform contains NaN or infinity; the mel filterbank would "
            "propagate it into every audio soft token",
            n_nonfinite=int((~np.isfinite(array)).sum()),
        )
    if int(sampling_rate) != int(expected_rate):
        raise SpokenAudioUnsupportedError(
            "sampling_rate_mismatch",
            f"waveform is {sampling_rate} Hz but the feature extractor expects "
            f"{expected_rate} Hz. This is refused rather than resampled: the "
            "processor passes an ndarray through untouched, so the mismatch "
            "would silently reinterpret the recording's pitch and duration. "
            "Resample to "
            f"{expected_rate} Hz when loading the file.",
            given_rate=int(sampling_rate),
            expected_rate=int(expected_rate),
        )
    return PreparedWaveform(
        samples=array,
        sampling_rate=int(expected_rate),
        n_samples=int(array.size),
        duration_s=float(array.size) / float(expected_rate),
        n_channels_in=n_channels_in,
        dtype_in=dtype_in,
        checksum="sha256:" + hashlib.sha256(array.tobytes()).hexdigest()[:32],
    )


def probe_waveform(duration_s: float, sampling_rate: int, *, seed: int = 0) -> np.ndarray:
    """A tiny deterministic tone+noise waveform. Generated, never a real file.

    Used by :func:`resolve_audio_interface` and by the CPU tests, so no real
    SpokenCOCO media is ever needed to establish that the path works.
    """
    n = max(1, int(round(duration_s * sampling_rate)))
    t = np.arange(n, dtype=np.float32) / float(sampling_rate)
    tone = 0.3 * np.sin(2.0 * math.pi * 440.0 * t, dtype=np.float32)
    noise = 0.02 * np.random.default_rng(seed).standard_normal(n).astype(np.float32)
    return np.ascontiguousarray(tone + noise, dtype=np.float32)


def silence_waveform(duration_s: float, sampling_rate: int) -> np.ndarray:
    """Digital silence of the same shape — the null evidence a probe compares to."""
    return np.zeros(max(1, int(round(duration_s * sampling_rate))), dtype=np.float32)


def audio_content_block(samples: np.ndarray) -> dict:
    """The one content block the model is ever given for spoken audio.

    Carries the waveform itself. No ``path``, no ``url``, no transcript, no
    caption — the evidence channel and nothing else.
    """
    return {"type": "audio", "audio": samples}


# ------------------------------------------------------------- placeholders


def expected_placeholder_count(features_mask_row: Any) -> int:
    """Placeholders the audio tower's output will need for one recording.

    Recomputed from the feature mask with the same two stride-2 convolution
    reductions the tower applies, which is the quantity ``Gemma4Model.forward``
    compares against ``audio_mask.sum()``. Computing it here is what lets a
    mismatch be a refusal at input-build time instead of a
    ``Audio features and audio tokens do not match`` deep inside the model.
    """
    mask = np.asarray(
        features_mask_row.detach().cpu().numpy()
        if hasattr(features_mask_row, "detach")
        else features_mask_row
    ).astype(bool)
    t = len(mask)
    for _ in range(2):
        t_out = (t + 2 - 3) // 2 + 1
        mask = mask[::2][:t_out]
        t = len(mask)
    return int(mask.sum())


def placeholder_span(input_ids: Any, audio_token_id: int) -> list[int] | None:
    """``[start, end)`` of the contiguous audio placeholder run, else ``None``."""
    ids = np.asarray(
        input_ids.detach().cpu().numpy() if hasattr(input_ids, "detach") else input_ids
    ).reshape(-1)
    hits = np.nonzero(ids == int(audio_token_id))[0].tolist()
    if not hits or hits[-1] - hits[0] + 1 != len(hits):
        return None
    return [int(hits[0]), int(hits[-1]) + 1]


def verify_audio_encoding(encoded: Mapping, *, audio_token_id: int) -> dict:
    """Hold one prepared audio input to the invariant the model will enforce.

    Raises:
        SpokenAudioUnsupportedError: With ``features_without_placeholders`` (the
            state that blocked this pilot), ``placeholders_without_features``,
            ``no_audio_pathway_engaged``, ``placeholder_feature_count_mismatch``
            or ``non_contiguous_placeholder_span``.
    """
    input_ids = encoded["input_ids"]
    ids = np.asarray(
        input_ids.detach().cpu().numpy() if hasattr(input_ids, "detach") else input_ids
    ).reshape(1, -1)
    n_placeholders = int((ids == int(audio_token_id)).sum())
    features = encoded.get("input_features")
    features_mask = encoded.get("input_features_mask")
    has_features = features is not None and features_mask is not None

    if has_features and n_placeholders == 0:
        raise SpokenAudioUnsupportedError(
            "features_without_placeholders",
            "the processor produced audio features but the text carried no "
            f"audio placeholder token (id {audio_token_id}). This is the exact "
            "state that blocked the pilot: the prompt must contain the audio "
            "token for the processor to expand it, which the chat-template "
            "audio content block is what supplies. The model would reject this "
            "input with 'Audio features and audio tokens do not match, "
            "tokens: 0'.",
            n_placeholders=0,
            feature_shape=list(getattr(features, "shape", [])),
        )
    if n_placeholders and not has_features:
        raise SpokenAudioUnsupportedError(
            "placeholders_without_features",
            f"the text carries {n_placeholders} audio placeholder token(s) but "
            "no audio features were produced; the placeholders would be "
            "embedded as literal special tokens rather than as the recording",
            n_placeholders=n_placeholders,
            keys=sorted(encoded.keys()),
        )
    if not n_placeholders and not has_features:
        raise SpokenAudioUnsupportedError(
            "no_audio_pathway_engaged",
            "neither audio features nor audio placeholder tokens are present; "
            "the recording never entered the model",
            keys=sorted(encoded.keys()),
        )

    expected = expected_placeholder_count(features_mask[0])
    if expected != n_placeholders:
        raise SpokenAudioUnsupportedError(
            "placeholder_feature_count_mismatch",
            f"{n_placeholders} audio placeholder token(s) but the feature mask "
            f"implies {expected} audio soft token(s); the model's "
            "masked_scatter would fail on the size mismatch",
            n_placeholders=n_placeholders,
            n_expected=expected,
        )
    span = placeholder_span(ids, audio_token_id)
    if span is None:
        raise SpokenAudioUnsupportedError(
            "non_contiguous_placeholder_span",
            "the audio placeholder tokens are not one contiguous run, so there "
            "is no single audio span to record or reason about",
            n_placeholders=n_placeholders,
        )
    return {
        "n_placeholders": n_placeholders,
        "n_expected_from_features": expected,
        "audio_token_span": span,
        "feature_keys": sorted(k for k in FEATURE_KEYS if k in encoded),
        "feature_shapes": {
            key: list(encoded[key].shape) for key in FEATURE_KEYS if key in encoded
        },
        "prompt_len": int(ids.shape[1]),
        "final_prompt_position": int(ids.shape[1]) - 1,
    }


# ------------------------------------------------------------- the encoding


def encode_audio_prompt(
    processor: Any,
    prompt: str,
    samples: np.ndarray,
    *,
    add_generation_prompt: bool = True,
) -> Any:
    """Build one spoken-audio input through the supported native path.

    The audio arrives as a chat-template content block, so the template renders
    the audio token and the processor expands it against the features it just
    computed. This is the whole repair.
    """
    return processor.apply_chat_template(
        [
            {
                "role": "user",
                "content": [
                    audio_content_block(samples),
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        add_generation_prompt=add_generation_prompt,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )


# -------------------------------------------------------- resolved interface


@dataclass(frozen=True)
class ResolvedAudioInterface:
    """Everything a run must record to have used the native audio path.

    This is the "resolved audio interface object": it is written into the run
    manifest, and its :attr:`protocol_fingerprint` binds stored results to the
    exact protocol that produced them.
    """

    protocol_version: str
    call_convention: str
    audio_kwarg: str
    content_block_schema: dict
    chat_template_convention: str
    sampling_rate: int
    waveform_dtype: str
    waveform_ndim: int
    audio_token: str
    audio_token_id: int
    boa_token: str | None
    eoa_token: str | None
    feature_keys: tuple[str, ...]
    processor_class: str
    feature_extractor_class: str
    audio_tower_present: bool
    model_repo_id: str
    model_revision: str
    processor_revision: str
    transformers_version: str
    probes: tuple[dict, ...] = ()
    dynamic_placeholder_count: bool = False
    notes: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "protocol_version": self.protocol_version,
            "call_convention": self.call_convention,
            "audio_kwarg": self.audio_kwarg,
            "content_block_schema": dict(self.content_block_schema),
            "chat_template_convention": self.chat_template_convention,
            "sampling_rate": self.sampling_rate,
            "waveform_dtype": self.waveform_dtype,
            "waveform_ndim": self.waveform_ndim,
            "audio_token": self.audio_token,
            "audio_token_id": self.audio_token_id,
            "boa_token": self.boa_token,
            "eoa_token": self.eoa_token,
            "feature_keys": list(self.feature_keys),
            "processor_class": self.processor_class,
            "feature_extractor_class": self.feature_extractor_class,
            "audio_tower_present": self.audio_tower_present,
            "model_repo_id": self.model_repo_id,
            "model_revision": self.model_revision,
            "processor_revision": self.processor_revision,
            "transformers_version": self.transformers_version,
            "probes": [dict(probe) for probe in self.probes],
            "dynamic_placeholder_count": self.dynamic_placeholder_count,
            "notes": dict(self.notes),
        }

    def to_record(self) -> dict:
        """:meth:`to_dict` plus the digest, for writing into an artifact.

        The digest is *carried* here, not derivable from the record: it is
        computed over the configuration fields only, while the record also holds
        the probe evidence. Recompute it from a
        :class:`ResolvedAudioInterface`, never from this dictionary.
        """
        return {**self.to_dict(), "protocol_fingerprint": self.protocol_fingerprint}

    @property
    def protocol_fingerprint(self) -> str:
        """``sha256`` over the protocol, excluding incidental probe numbers.

        The fingerprint covers what a *later* run would have to match: the model
        and processor revisions, the transformers version, the call and chat
        conventions, the sample rate, and the placeholder token. Probe token
        counts are evidence, not configuration, so they stay out of it — two
        machines that resolve the same protocol must agree.
        """
        from jlens.mmpilot.store import payload_checksum

        payload = self.to_dict()
        for key in ("probes", "notes", "dynamic_placeholder_count"):
            payload.pop(key, None)
        return payload_checksum(payload)


def _transformers_version() -> str:
    try:
        import transformers

        return str(transformers.__version__)
    except Exception:  # pragma: no cover - transformers is a hard dependency
        return "unknown"


def resolve_audio_interface(
    processor: Any,
    hf_config: Any,
    *,
    model_repo_id: str,
    model_revision: str,
    processor_revision: str,
    probe_durations_s: Sequence[float] = PROBE_DURATIONS_S,
    require_audio_tower: bool = True,
) -> ResolvedAudioInterface:
    """Resolve the native audio protocol *by running it*, or refuse.

    Presence is never taken as support. Every attribute below is checked, and
    then a generated waveform is pushed through the real processor at two
    durations and required to produce a verified placeholder span. Only then is
    a :class:`ResolvedAudioInterface` returned.

    No model weights are needed, so this runs on CPU in a second and is what the
    feasibility notebook calls before deciding whether to load Gemma at all.

    Raises:
        SpokenAudioUnsupportedError: Naming which link broke. The message states
            the nearest supported configuration where one exists.
    """
    call_params = list(inspect.signature(processor.__call__).parameters)
    audio_kwarg = next(
        (k for k in ("audio", "audios", "raw_audio", "raw_speech") if k in call_params),
        None,
    )
    if audio_kwarg is None:
        raise SpokenAudioUnsupportedError(
            "no_audio_call_parameter",
            f"{type(processor).__name__}.__call__ has no audio parameter "
            f"(parameters: {call_params}); this processor cannot take a "
            "waveform at all",
            call_parameters=call_params,
        )

    feature_extractor = getattr(processor, "feature_extractor", None) or getattr(
        processor, "audio_processor", None
    )
    if feature_extractor is None:
        raise SpokenAudioUnsupportedError(
            "no_feature_extractor",
            f"{type(processor).__name__} exposes no feature_extractor or "
            "audio_processor, so no waveform can be turned into features. If "
            "the checkpoint's processor_config.json has no 'feature_extractor' "
            "section, the audio components were not published with it.",
        )

    audio_token = getattr(processor, "audio_token", None)
    if not audio_token:
        raise SpokenAudioUnsupportedError(
            "no_audio_token",
            "the processor has no audio_token, so there is no placeholder for "
            "the audio features to be scattered into",
        )
    audio_token_id = getattr(processor, "audio_token_id", None)
    if audio_token_id is None:
        audio_token_id = getattr(hf_config, "audio_token_id", None)
    if audio_token_id is None:
        raise SpokenAudioUnsupportedError(
            "no_audio_token_id",
            "neither the processor nor the model config defines audio_token_id; "
            "the audio span could not be located in input_ids even if it were "
            "produced",
        )

    has_chat_template = getattr(processor, "chat_template", None) is not None or (
        getattr(getattr(processor, "tokenizer", None), "chat_template", None) is not None
    )
    if not has_chat_template:
        raise SpokenAudioUnsupportedError(
            "no_chat_template",
            "the processor has no chat template, so the audio content block "
            "cannot be rendered into the audio placeholder token. A bare "
            "processor(text=..., audio=...) call is not a substitute: it "
            "produces features with zero placeholders and the model rejects it.",
        )

    audio_tower_present = hasattr(hf_config, "audio_config") and (
        getattr(hf_config, "audio_config", None) is not None
    )
    if require_audio_tower and not audio_tower_present:
        raise SpokenAudioUnsupportedError(
            "no_audio_tower",
            f"{model_repo_id}@{model_revision} has no audio_config, so the "
            "checkpoint carries no audio tower. Audio placeholders would have "
            "nothing to be filled from. A different checkpoint would be "
            "required, which invalidates the text-calibrated lens.",
        )

    sampling_rate = int(getattr(feature_extractor, "sampling_rate", 16_000))

    probes: list[dict] = []
    for index, duration in enumerate(probe_durations_s):
        samples = probe_waveform(duration, sampling_rate, seed=index)
        prepared = prepare_waveform(samples, sampling_rate, expected_rate=sampling_rate)
        try:
            encoded = encode_audio_prompt(processor, PROBE_PROMPT, prepared.samples)
        except SpokenAudioUnsupportedError:
            raise
        except Exception as exc:
            raise SpokenAudioUnsupportedError(
                "probe_failed",
                f"the processor raised {type(exc).__name__} while building a "
                f"{duration:g} s probe input through "
                f"{CALL_CONVENTION}: {exc}",
                duration_s=duration,
            ) from exc
        verified = verify_audio_encoding(encoded, audio_token_id=int(audio_token_id))
        probes.append({"waveform": prepared.to_dict(), **verified})

    counts = [probe["n_placeholders"] for probe in probes]
    dynamic = len(set(counts)) > 1

    return ResolvedAudioInterface(
        protocol_version=AUDIO_PROTOCOL_VERSION,
        call_convention=CALL_CONVENTION,
        audio_kwarg=audio_kwarg,
        content_block_schema=dict(CONTENT_BLOCK_SCHEMA),
        chat_template_convention=(
            "apply_chat_template(conversation, add_generation_prompt=True, "
            "tokenize=True, return_dict=True, return_tensors='pt')"
        ),
        sampling_rate=sampling_rate,
        waveform_dtype="float32",
        waveform_ndim=1,
        audio_token=str(audio_token),
        audio_token_id=int(audio_token_id),
        boa_token=getattr(processor, "boa_token", None),
        eoa_token=getattr(processor, "eoa_token", None),
        feature_keys=tuple(probes[0]["feature_keys"]) if probes else FEATURE_KEYS,
        processor_class=type(processor).__name__,
        feature_extractor_class=type(feature_extractor).__name__,
        audio_tower_present=bool(audio_tower_present),
        model_repo_id=str(model_repo_id),
        model_revision=str(model_revision),
        processor_revision=str(processor_revision),
        transformers_version=_transformers_version(),
        probes=tuple(probes),
        dynamic_placeholder_count=dynamic,
        notes={
            "placeholder_counts": counts,
            "why_not_bare_processor_call": (
                "processor(text=<prompt>, audio=<waveform>) returns features "
                "with zero placeholder tokens and raises nothing; the model "
                "then fails with 'Audio features and audio tokens do not "
                "match'. The content block is what inserts the placeholder."
            ),
        },
    )


# ------------------------------------------------------------ tower + leakage


def audio_tower_module(hf_model: Any) -> Any | None:
    """The audio tower submodule, or ``None`` if this model has none."""
    for path in (("model", "audio_tower"), ("audio_tower",)):
        node = hf_model
        for name in path:
            node = getattr(node, name, None)
            if node is None:
                break
        if node is not None:
            return node
    return None


def check_audio_tower_invoked(backend: Any, inputs: Any) -> dict:
    """Confirm the forward pass actually ran the audio tower.

    A placeholder span proves the *text* was built correctly. Only a fired hook
    on the tower proves the recording was encoded rather than ignored.

    Raises:
        SpokenAudioUnsupportedError: If the model has no audio tower, or the
            tower was not entered during the forward pass.
    """
    tower = audio_tower_module(getattr(backend, "hf_model", backend))
    if tower is None:
        raise SpokenAudioUnsupportedError(
            "no_audio_tower",
            "the loaded model exposes no audio tower module, so audio "
            "placeholders cannot be filled from a recording",
        )
    seen: dict[str, Any] = {"invoked": False}

    def _hook(_module, _args, output):
        seen["invoked"] = True
        tensor = getattr(output, "last_hidden_state", None)
        if tensor is None and hasattr(output, "shape"):
            tensor = output
        if tensor is not None and hasattr(tensor, "shape"):
            seen["output_shape"] = list(tensor.shape)

    handle = tower.register_forward_hook(_hook)
    try:
        backend.forward_logits(inputs.tensors)
    finally:
        handle.remove()
    if not seen["invoked"]:
        raise SpokenAudioUnsupportedError(
            "no_audio_pathway_engaged",
            "the audio tower was never entered during the forward pass; the "
            "recording did not reach the model even though the input carried "
            "audio features",
        )
    return {
        "check": "audio_tower_invoked",
        "tower_class": type(tower).__name__,
        "invoked": True,
        "output_shape": seen.get("output_shape"),
    }


def assert_no_text_leakage(prompt: str, *, forbidden: Sequence[str]) -> dict:
    """Refuse a spoken-audio prompt that carries a transcript or a filename.

    The spoken-audio condition's whole claim is that the recording is the only
    evidence. A caption or a file stem in the prompt would make the condition
    answerable from text, and the resulting number would be about the caption.

    Raises:
        ValueError: Naming the leaked fragment.
    """
    haystack = prompt.casefold()
    leaked = sorted(
        {
            str(item)
            for item in forbidden
            if str(item).strip() and str(item).strip().casefold() in haystack
        }
    )
    if leaked:
        raise ValueError(
            "the spoken_audio prompt leaks non-audio evidence "
            f"{leaked!r}; the recording must be the only evidence. Captions, "
            "transcripts, file names and dataset ids are never included."
        )
    return {"check": "no_text_leakage", "n_checked": len(list(forbidden)), "leaked": []}


__all__ = [
    "AUDIO_PROTOCOL_VERSION",
    "CALL_CONVENTION",
    "CONTENT_BLOCK_SCHEMA",
    "FEATURE_KEYS",
    "PROBE_DURATIONS_S",
    "PROBE_PROMPT",
    "REASONS",
    "SPOKEN_AUDIO",
    "PreparedWaveform",
    "ResolvedAudioInterface",
    "SpokenAudioUnsupportedError",
    "assert_no_text_leakage",
    "audio_content_block",
    "audio_tower_module",
    "check_audio_tower_invoked",
    "encode_audio_prompt",
    "expected_placeholder_count",
    "placeholder_span",
    "prepare_waveform",
    "probe_waveform",
    "resolve_audio_interface",
    "silence_waveform",
    "verify_audio_encoding",
]
