# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Deterministic MOCK world: a synthetic dataset and a synthetic model.

MOCK mode exists so the whole pipeline — manifest normalization, capability
scoring, activation capture, pursuit, retrieval, direction estimation,
interventions, controls, resume, reporting — can be exercised on CPU in
seconds, with no Gemma, no Drive, and no media codecs.

The synthetic world is built around a *known shared concept direction*: each
concept has a fixed vector that enters the model identically whether the
evidence arrived as written caption, as "image" bytes, or as "audio" bytes.
On top of it sit a per-modality offset and a per-sample nuisance component, so
retrieval and direction estimation have something to fail at — a sample's
activation is never just its concept vector, and modality is itself a strong
direction that the analysis must not mistake for concept identity.

The mock is not tuned to make J-space beat the raw-residual baseline. In a
world this simple the raw difference-in-means is a very good estimator, and it
often wins; that comparison is reported rather than arranged.

**A passing MOCK run proves the pipeline runs. It is never evidence about
Gemma.** Every artifact a MOCK run writes is labelled ``mode="mock"``.
"""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from jlens.lens import JacobianLens
from jlens.mmpilot.backend import BuiltInputs, file_checksum, text_hash
from jlens.mmpilot.manifest import caption_mentions
from jlens.mmpilot.store import safe_key

MOCK_D_MODEL = 24
MOCK_VOCAB = 64
MOCK_N_LAYERS = 6
MOCK_LAYERS = (2, 4)

#: Token ids the mock reserves. Answer tokens start at :data:`_ANSWER_BASE`.
_PAD, _EOS, _BOS, _ANSWER_SUFFIX = 0, 1, 2, 7
_ANSWER_BASE = 8
_TEXT_BASE = 24

#: Amplitudes: concept signal, per-modality offset, per-sample nuisance.
CONCEPT_AMPLITUDE = 3.0
MODALITY_AMPLITUDE = 0.7
NUISANCE_AMPLITUDE = 0.7
READOUT_GAIN = 1.0

MOCK_CONCEPTS: dict[str, tuple[str, ...]] = {
    "bus": ("bus", "buses"),
    "cat": ("cat", "cats"),
    "dog": ("dog", "dogs"),
    "pizza": ("pizza", "pizzas"),
}

#: Categories the mock's ``instances_*.json`` declares but that no image is a
#: usable positive for. Real COCO declares 80 and most are infeasible on any
#: given local subset; discovery has to find them and the ranking has to reject
#: them for stated reasons rather than never having considered them.
#:
#: Chosen to exercise every branch of the lexical policy in
#: :mod:`jlens.mmpilot.concepts`: an excluded colour term (``orange``), an
#: alias-only category (``remote``), an ambiguous one (``train``), a phrase
#: category that collides with a concept (``hot dog`` vs ``dog``), and the
#: ubiquitous one.
DEFAULT_EXTRA_CATEGORIES: tuple[str, ...] = (
    "hot dog",
    "orange",
    "person",
    "remote",
    "train",
)

#: The category COCO annotates on most images. Not a usable concept.
UBIQUITOUS_CATEGORY = "person"


def _seeded_vector(key: str, d_model: int) -> torch.Tensor:
    """A deterministic unit vector for ``key`` (stable across machines)."""
    seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") % (2**31 - 1)
    generator = torch.Generator().manual_seed(seed)
    vector = torch.randn(d_model, generator=generator, dtype=torch.float32)
    return vector / vector.norm()


class MockWorld:
    """The generative story behind the synthetic data.

    Evidence for a sample is ``concept + modality offset + nuisance``. The
    concept term is identical across modalities — that shared direction is
    exactly what the pilot is supposed to find.
    """

    def __init__(
        self,
        concepts: Mapping[str, Sequence[str]] = MOCK_CONCEPTS,
        *,
        d_model: int = MOCK_D_MODEL,
        seed: str = "mmpilot-mock",
    ) -> None:
        self.concepts = {name: tuple(words) for name, words in concepts.items()}
        self.d_model = d_model
        self.seed = seed

    def concept_vector(self, concept: str) -> torch.Tensor:
        return _seeded_vector(f"{self.seed}|concept|{concept}", self.d_model)

    def modality_vector(self, modality: str) -> torch.Tensor:
        return _seeded_vector(f"{self.seed}|modality|{modality}", self.d_model)

    def evidence(
        self, *, concepts_present: Sequence[str], modality: str, nuisance_key: str
    ) -> torch.Tensor:
        vector = MODALITY_AMPLITUDE * self.modality_vector(modality)
        for concept in sorted(concepts_present):
            vector = vector + CONCEPT_AMPLITUDE * self.concept_vector(concept)
        return vector + NUISANCE_AMPLITUDE * _seeded_vector(
            f"{self.seed}|nuisance|{nuisance_key}", self.d_model
        )

    def concepts_in(self, caption: str) -> list[str]:
        return sorted(
            name
            for name, words in self.concepts.items()
            if caption_mentions(caption, words)
        )


# ------------------------------------------------------------ media on disk


def _write_vector(path: Path, vector: torch.Tensor) -> None:
    """Store a vector as raw float32 bytes under a media file extension.

    The mock's "decoder" is :func:`load_mock_media`. Nothing here pretends to
    be a real JPEG or WAV; the point is that media arrives as opaque bytes that
    the backend turns into evidence, exactly as a real tower would.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"MOCKMEDIA" + struct.pack(f"<{len(vector)}f", *vector.tolist()))


def load_mock_media(path: str | Path) -> torch.Tensor:
    raw = Path(path).read_bytes()
    if not raw.startswith(b"MOCKMEDIA"):
        raise ValueError(f"{path} is not mock media")
    payload = raw[len(b"MOCKMEDIA") :]
    return torch.tensor(struct.unpack(f"<{len(payload) // 4}f", payload), dtype=torch.float32)


def build_mock_dataset(
    root: str | Path,
    *,
    world: MockWorld | None = None,
    images_per_concept: int = 6,
    negative_images: int = 12,
    captions_per_image: int = 2,
    speakers: Sequence[str] = ("spk-a", "spk-b", "spk-c"),
    layout: str = "flat",
    manifest_records: int | None = None,
    visual_only_images: int = 0,
    extra_categories: Sequence[str] | None = None,
) -> dict:
    """Write a synthetic SpokenCOCO-shaped dataset and manifest.

    The manifest deliberately uses a *nested* shape (one record per image with
    a list of caption entries) and field names the pilot code must discover
    rather than assume.

    ``layout`` picks the on-disk arrangement:

    - ``"flat"`` — images and audio under one root, the simple case.
    - ``"sibling"`` — the arrangement the real Drive dataset uses: images under
      ``coco/train2014/`` addressed as ``train2014/...jpg``, recordings under
      ``SpokenCOCO/wavs/train/`` addressed as ``wavs/train/...wav``. The two
      modalities are only resolvable from *different* roots, which is what
      broke the single-root assumption.

    ``visual_only_images`` adds images per concept that carry the COCO object
    annotation while their captions never name the concept — the SpokenCOCO
    pattern that broke the first real run. Their *evidence* still contains the
    concept (the picture really does show it), so a labelling rule that reads
    only the annotation will happily select them and then mark the text arm
    wrong. They must be rejected as synchronized positives.

    ``extra_categories`` are declared in the written ``instances_*.json``
    without being world concepts, so the discovered candidate universe is wider
    than the set of concepts that can actually work — which is the situation on
    real COCO. Defaults to :data:`DEFAULT_EXTRA_CATEGORIES`.

    Returns ``{"root", "manifest_path", "world", "image_root", "audio_root",
    "object_annotations", "visual_only_image_ids", "declared_categories"}``.
    """
    if layout not in ("flat", "sibling"):
        raise ValueError(f"unknown layout {layout!r}")
    world = world or MockWorld()
    root = Path(root)
    image_root = root / "coco" if layout == "sibling" else root
    audio_root = root / "SpokenCOCO" if layout == "sibling" else root

    records: list[dict] = []
    #: Ground-truth visual labels: what is in the picture, caption or not.
    object_annotations: dict[str, list[str]] = {}
    visual_only_image_ids: list[str] = []

    def add_image(
        image_id: str,
        concepts_present: Sequence[str],
        index: int,
        *,
        name_in_caption: bool = True,
    ) -> None:
        if layout == "sibling":
            image_rel = f"train2014/COCO_train2014_{image_id}.jpg"
        else:
            image_rel = f"images/{image_id}.jpg"
        _write_vector(
            image_root / image_rel,
            world.evidence(
                concepts_present=concepts_present,
                modality="image",
                nuisance_key=f"{image_id}|image",
            ),
        )
        object_annotations[image_id] = sorted(concepts_present)
        if concepts_present and not name_in_caption:
            visual_only_image_ids.append(image_id)
        entries = []
        for caption_index in range(captions_per_image):
            if not name_in_caption:
                # The picture contains the concept; the sentence does not name
                # it. This is the group that must never become a positive.
                subject = "small creature"
            else:
                subject = " and a ".join(concepts_present) if concepts_present else "table"
            caption = (
                f"a photo number {caption_index} showing a {subject} "
                f"near a window in scene {index}"
            )
            if layout == "sibling":
                audio_rel = f"wavs/train/{image_id}_{caption_index}.wav"
            else:
                audio_rel = f"audio/{image_id}_{caption_index}.wav"
            _write_vector(
                audio_root / audio_rel,
                world.evidence(
                    concepts_present=concepts_present,
                    modality="spoken_audio",
                    nuisance_key=f"{caption}|spoken_audio",
                ),
            )
            entries.append(
                {
                    "uttid": f"{image_id}_{caption_index}",
                    "wav": audio_rel,
                    "text": caption,
                    "speaker": speakers[(index + caption_index) % len(speakers)],
                }
            )
        records.append({"image": image_rel, "image_id": image_id, "captions": entries})

    index = 0
    for concept in sorted(world.concepts):
        for _ in range(images_per_concept):
            add_image(f"img{index:04d}", [concept], index)
            index += 1
        for _ in range(visual_only_images):
            add_image(f"img{index:04d}", [concept], index, name_in_caption=False)
            index += 1
    for _ in range(negative_images):
        add_image(f"img{index:04d}", [], index)
        index += 1

    manifest_path = root / "spokencoco_manifest.json"
    full_records = records
    if manifest_records is not None:
        # Reproduces the real situation: a small hand-made manifest sitting
        # next to SpokenCOCO's own far larger annotation file.
        records = records[:manifest_records]
    manifest_path.write_text(
        json.dumps({"split": "pilot", "data": records}, indent=2), encoding="utf-8"
    )
    # COCO object annotations describe the *picture*, so they are written from
    # the world's ground truth and never from the caption text. That is what
    # makes a visual-only group possible at all.
    #
    # The file declares MORE categories than the world's concepts, exactly as
    # real COCO does: 80 categories are defined and only a few of them will
    # turn out to be feasible here. Discovery has to read the universe off this
    # file rather than assume the concepts it happens to find groups for.
    if extra_categories is None:
        extra_categories = DEFAULT_EXTRA_CATEGORIES
    category_names = sorted({*world.concepts, *extra_categories})
    category_ids = {name: index + 1 for index, name in enumerate(category_names)}
    # `person` is annotated on most images, as in COCO. It is not a usable
    # concept — it discriminates nothing — and it is what the co-occurrence
    # statistic and the exclusion policy exist to handle.
    if UBIQUITOUS_CATEGORY in category_names:
        for position, image_id in enumerate(sorted(object_annotations)):
            if position % 4:
                object_annotations[image_id] = sorted(
                    {*object_annotations[image_id], UBIQUITOUS_CATEGORY}
                )
    instances_path = image_root / "annotations" / "instances_train2014.json"
    instances_path.parent.mkdir(parents=True, exist_ok=True)
    instances = [
        {"image_id": image_id, "category_id": category_ids[name], "id": position + 1}
        for position, (image_id, name) in enumerate(
            (image_id, name)
            for image_id in sorted(object_annotations)
            for name in object_annotations[image_id]
        )
    ]
    instances_path.write_text(
        json.dumps(
            {
                "categories": [
                    {
                        "id": category_ids[name],
                        "name": name,
                        "supercategory": "mock",
                    }
                    for name in category_names
                ],
                "annotations": instances,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    annotation_path = None
    if layout == "sibling":
        (root / "cstf_dataset_marker.json").write_text(
            json.dumps({"dataset": "cstf_spokencoco", "mock": True}), encoding="utf-8"
        )
        annotation_path = audio_root / "SpokenCOCO_train.json"
        annotation_path.parent.mkdir(parents=True, exist_ok=True)
        annotation_path.write_text(
            json.dumps({"data": full_records}, indent=2), encoding="utf-8"
        )
        # A COCO caption file: discoverable, but carrying no audio, so it must
        # be reported and then rejected as unable to form synchronized groups.
        coco_annotations = image_root / "annotations"
        coco_annotations.mkdir(parents=True, exist_ok=True)
        (coco_annotations / "captions_train2014.json").write_text(
            json.dumps(
                {
                    "images": [
                        {"id": index, "file_name": record["image"].split("/")[-1]}
                        for index, record in enumerate(full_records)
                    ],
                    "annotations": [
                        {"image_id": index, "id": index, "caption": entry["text"]}
                        for index, record in enumerate(full_records)
                        for entry in record["captions"][:1]
                    ],
                }
            ),
            encoding="utf-8",
        )
    return {
        "root": str(root),
        "manifest_path": str(manifest_path),
        "annotation_path": str(annotation_path) if annotation_path else None,
        "instances_path": str(instances_path),
        "declared_categories": category_names,
        "object_annotations": object_annotations,
        "visual_only_image_ids": sorted(visual_only_image_ids),
        "world": world,
        "layout": layout,
        "image_root": str(image_root),
        "audio_root": str(audio_root),
        "n_records_in_manifest": len(records),
        "n_records_total": len(full_records),
    }


def attach_object_annotations(groups: Sequence[Mapping], built: Mapping) -> list[dict]:
    """Copies of ``groups`` carrying the mock world's ground-truth COCO labels.

    The real pipeline gets these from ``instances_*.json`` via
    :func:`jlens.mmpilot.expansion.attach_concept_annotations`. Tests that work
    straight from a normalized manifest use this instead of re-parsing the file.
    """
    labels = built["object_annotations"]
    out = []
    for group in groups:
        found = sorted(labels.get(str(group["image_id"]), []))
        out.append(
            {
                **dict(group),
                "concept_annotations": found,
                "annotation_source": "coco_object_annotation" if found else "none",
                "synchronized_group_id": group.get(
                    "synchronized_group_id", group["group_id"]
                ),
            }
        )
    return out


# ----------------------------------------------------------------- the model


class _MockBlock(nn.Module):
    """Block 0 broadcasts position 0 to every position (a stand-in for
    attention reaching the evidence); later blocks are the identity, so the
    Jacobian from any later layer to the output is exactly ``I``."""

    def __init__(self, broadcast: bool) -> None:
        super().__init__()
        self.broadcast = broadcast

    def forward(self, hidden: torch.Tensor, **_kwargs) -> torch.Tensor:
        if self.broadcast:
            return hidden + hidden[:, :1]
        return hidden


class _MockTextModel(nn.Module):
    def __init__(self, d_model: int, vocab: int, n_layers: int) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(7)
        self.embed_tokens = nn.Embedding(vocab, d_model)
        with torch.no_grad():
            self.embed_tokens.weight.copy_(
                torch.randn(vocab, d_model, generator=generator) * 0.05
            )
        self.layers = nn.ModuleList(
            _MockBlock(broadcast=(i == 0)) for i in range(n_layers)
        )
        self.norm = nn.Identity()

    def forward(self, input_ids: torch.Tensor, evidence: torch.Tensor | None = None, **_kw):
        hidden = self.embed_tokens(input_ids)
        if evidence is not None:
            hidden = hidden.clone()
            hidden[:, 0] = hidden[:, 0] + evidence.reshape(hidden.shape[0], -1)
        for block in self.layers:
            hidden = block(hidden)
        return SimpleNamespace(last_hidden_state=self.norm(hidden))


class _MockModel(nn.Module):
    def __init__(self, d_model: int, vocab: int, n_layers: int) -> None:
        super().__init__()
        self.language_model = _MockTextModel(d_model, vocab, n_layers)
        self.vision_tower = nn.Linear(2, 2)
        self.audio_tower = nn.Linear(2, 2)


class MockGemmaLike(nn.Module):
    """A model with Gemma 4's module layout and analytically clean readout.

    ``lm_head`` rows for the answer tokens *are* the concept vectors (times a
    gain), and the final norm is the identity, so the logit of concept ``c``
    is exactly ``<h_last, c_vec> * gain``. That makes the expected sign of
    every intervention derivable rather than empirical.
    """

    def __init__(self, world: MockWorld, *, n_layers: int = MOCK_N_LAYERS) -> None:
        super().__init__()
        self.world = world
        d_model, vocab = world.d_model, MOCK_VOCAB
        self.model = _MockModel(d_model, vocab, n_layers)
        self.lm_head = nn.Linear(d_model, vocab, bias=False)
        generator = torch.Generator().manual_seed(11)
        with torch.no_grad():
            # Every row is unit norm, including the concept rows. Atom norms in
            # a real W_U are broadly comparable, and sparse-code cosine compares
            # coefficients on *raw* atoms — a mock with one giant atom would
            # make every code look alike for reasons the real model does not have.
            rows = torch.randn(vocab, d_model, generator=generator)
            rows = rows / rows.norm(dim=-1, keepdim=True)
            self.lm_head.weight.copy_(rows * READOUT_GAIN)
            for index, concept in enumerate(sorted(world.concepts)):
                self.lm_head.weight[_ANSWER_BASE + index] = (
                    world.concept_vector(concept) * READOUT_GAIN
                )
            self.lm_head.weight[_ANSWER_SUFFIX] = torch.zeros(d_model)
        self.config = SimpleNamespace(
            get_text_config=lambda: SimpleNamespace(
                hidden_size=d_model, num_hidden_layers=n_layers, vocab_size=vocab
            ),
            image_token_id=None,
            audio_token_id=None,
        )
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, input_ids=None, evidence=None, **_kwargs):
        hidden = self.model.language_model(input_ids=input_ids, evidence=evidence)
        return SimpleNamespace(logits=self.lm_head(hidden.last_hidden_state))


class MockPilotBackend:
    """:class:`~jlens.mmpilot.backend.PilotBackend` over :class:`MockGemmaLike`.

    ``supports_audio`` is configurable so the notebook's blocked-modality path
    (spoken audio unavailable → text/image pilot, audio reported NO-GO) is
    testable without needing a checkpoint that actually lacks audio.
    """

    def __init__(
        self,
        world: MockWorld | None = None,
        *,
        supports_audio: bool = True,
        n_layers: int = MOCK_N_LAYERS,
    ) -> None:
        self.world = world or MockWorld()
        self.hf_model = MockGemmaLike(self.world, n_layers=n_layers)
        self.d_model = self.world.d_model
        self.n_layers = n_layers
        self._supports_audio = supports_audio
        self.answer_tokens = {
            concept: _ANSWER_BASE + index
            for index, concept in enumerate(sorted(self.world.concepts))
        }
        self.interface = {
            "processor_class": "MockProcessor",
            "components": {"tokenizer": "MockTokenizer", "image_processor": "MockImage"},
            "call_parameters": ["text", "images", "audio", "return_tensors"],
            "audio_kwarg": "audio" if supports_audio else None,
            "supports_image": True,
            "supports_audio": supports_audio,
            "has_chat_template": False,
            "image_token_id": None,
            "audio_token_id": None,
            "audio_tower_present": supports_audio,
            "vision_tower_present": True,
        }

    @property
    def blocks(self):
        return self.hf_model.model.language_model.layers

    def supports(self, modality: str) -> bool:
        if modality == "text":
            return True
        if modality == "image":
            return True
        if modality == "spoken_audio":
            return self._supports_audio
        raise ValueError(f"unknown modality {modality!r}")

    def unembedding_weight(self) -> torch.Tensor:
        return self.hf_model.lm_head.weight

    def encode_candidate(self, text: str) -> list[int]:
        stripped = text.strip()
        if stripped in self.answer_tokens:
            # Deliberately two tokens: the pilot must score complete sequences.
            return [self.answer_tokens[stripped], _ANSWER_SUFFIX]
        return self._tokenize(stripped)

    def _tokenize(self, text: str) -> list[int]:
        ids = []
        for word in text.split():
            digest = hashlib.sha256(word.encode()).digest()
            ids.append(_TEXT_BASE + digest[0] % (MOCK_VOCAB - _TEXT_BASE))
        return ids or [_PAD]

    def build_inputs(
        self,
        *,
        prompt: str,
        modality: str,
        image=None,
        audio=None,
        sampling_rate: int | None = None,
        media_path: str | None = None,
    ) -> BuiltInputs:
        if not self.supports(modality):
            from jlens.mmpilot.backend import ModalityUnsupportedError

            raise ModalityUnsupportedError(f"mock backend has no {modality!r} channel")
        if modality == "text":
            caption = ""
            for line in prompt.splitlines():
                if line.startswith("Caption:"):
                    caption = line[len("Caption:") :].strip()
            evidence = self.world.evidence(
                concepts_present=self.world.concepts_in(caption),
                modality="text",
                nuisance_key=f"{caption}|text",
            )
        else:
            media = image if modality == "image" else audio
            if media is None:
                raise ValueError(f"{modality} condition needs its media")
            evidence = torch.as_tensor(media, dtype=torch.float32)
        ids = [_BOS, *self._tokenize(prompt)]
        return BuiltInputs(
            tensors={
                "input_ids": torch.tensor([ids], dtype=torch.long),
                "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
                "evidence": evidence.reshape(1, 1, -1),
            },
            prompt_len=len(ids),
            modality=modality,
            prompt_hash=text_hash(prompt),
            media_checksum=file_checksum(media_path) if media_path else None,
            route={"route": "mock"},
            modality_token_range=None,
        )

    @torch.no_grad()
    def forward_logits(self, tensors: Mapping) -> torch.Tensor:
        payload = {k: v for k, v in dict(tensors).items() if k != "use_cache"}
        return self.hf_model(**payload).logits.float()


# ------------------------------------------------- the mock audio processor


#: Mock stand-ins for the pinned checkpoint's real audio tokens.
MOCK_AUDIO_TOKEN_ID = 5
MOCK_BOA_TOKEN_ID = 3
MOCK_EOA_TOKEN_ID = 4


#: Feature width the mock feature extractor emits per frame.
MOCK_AUDIO_FEATURE_DIM = 8


class MockAudioFeatureExtractor:
    """Frame arithmetic and eight per-frame statistics — not a filterbank.

    The statistics are real functions of the samples, so two different
    recordings produce different features and silence produces a distinct one.
    A feature extractor that returned zeros would make "waveform differs from
    silence" pass for the wrong reason.
    """

    def __init__(self, sampling_rate: int = 16_000, hop_length: int = 160) -> None:
        self.sampling_rate = sampling_rate
        self.hop_length = hop_length

    def n_frames(self, n_samples: int) -> int:
        return max(1, n_samples // self.hop_length - 1)

    def __call__(self, waveform) -> torch.Tensor:
        samples = torch.as_tensor(waveform, dtype=torch.float32).reshape(-1)
        hop = self.hop_length
        n_frames = self.n_frames(int(samples.numel()))
        rows = []
        for index in range(n_frames):
            frame = samples[index * hop : index * hop + 2 * hop]
            if frame.numel() == 0:
                frame = samples[:1]
            half = max(1, frame.numel() // 2)
            rows.append(
                torch.stack(
                    [
                        frame.mean(),
                        frame.abs().mean(),
                        frame.pow(2).mean(),
                        frame.max(),
                        frame.min(),
                        frame[:half].mean(),
                        frame[half:].mean() if frame[half:].numel() else frame.mean(),
                        frame.abs().max(),
                    ]
                )
            )
        return torch.stack(rows).unsqueeze(0)


class MockAudioProcessor:
    """A processor that reproduces ``Gemma4Processor``'s audio contract exactly.

    Two things about the real processor have to be reproduced, because they are
    what the repair is about:

    1. ``apply_chat_template`` renders an ``{"type": "audio"}`` content block as
       the audio token, and only *then* does the processor expand it against the
       features. Audio passed without that token yields features and **zero**
       placeholders, silently.
    2. The number of placeholders is derived from the feature mask through two
       stride-2 reductions, so it tracks the recording's duration.

    The ``emit_*`` flags exist so every refusal in
    :mod:`jlens.mmpilot.audio` can be exercised deterministically on CPU,
    without needing a checkpoint that is broken in that particular way.
    """

    chat_template = "{# mock #}"

    def __init__(
        self,
        *,
        sampling_rate: int = 16_000,
        emit_placeholders: bool = True,
        emit_features: bool = True,
        placeholder_delta: int = 0,
        contiguous: bool = True,
        renders_audio_token: bool = True,
    ) -> None:
        self.feature_extractor = MockAudioFeatureExtractor(sampling_rate)
        self.audio_token = "<|audio|>"
        self.audio_token_id = MOCK_AUDIO_TOKEN_ID
        self.boa_token = "<|audio>"
        self.eoa_token = "<audio|>"
        self.tokenizer = SimpleNamespace(chat_template=self.chat_template)
        self.emit_placeholders = emit_placeholders
        self.emit_features = emit_features
        self.placeholder_delta = placeholder_delta
        self.contiguous = contiguous
        self.renders_audio_token = renders_audio_token

    def __call__(self, images=None, text=None, videos=None, audio=None, **kwargs):
        """The bare call. Note it does **not** insert a placeholder for you."""
        return self._encode(text or "", audio, placeholder_in_text=False)

    def apply_chat_template(
        self,
        conversation,
        *,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        **kwargs,
    ):
        blocks = list(conversation[0]["content"])
        audio = next(
            (block["audio"] for block in blocks if block.get("type") == "audio"), None
        )
        prompt = "".join(
            str(block.get("text", "")) for block in blocks if block.get("type") == "text"
        )
        return self._encode(
            prompt, audio, placeholder_in_text=audio is not None and self.renders_audio_token
        )

    def _placeholder_count(self, n_frames: int) -> int:
        t = n_frames
        for _ in range(2):
            t = min((t + 2 - 3) // 2 + 1, len(range(0, t, 2)))
        return max(0, t)

    def _encode(self, prompt: str, audio, *, placeholder_in_text: bool) -> dict:
        prompt_ids = [_BOS, *(_TEXT_BASE + (index % 8) for index in range(len(prompt.split())))]
        encoded: dict[str, torch.Tensor] = {}
        placeholders: list[int] = []
        if audio is not None:
            n_samples = int(torch.as_tensor(audio).numel())
            n_frames = self.feature_extractor.n_frames(n_samples)
            if self.emit_features:
                encoded["input_features"] = self.feature_extractor(audio)
                encoded["input_features_mask"] = torch.ones(1, n_frames, dtype=torch.bool)
            if placeholder_in_text and self.emit_placeholders:
                count = max(0, self._placeholder_count(n_frames) + self.placeholder_delta)
                placeholders = [
                    MOCK_BOA_TOKEN_ID,
                    *([MOCK_AUDIO_TOKEN_ID] * count),
                    MOCK_EOA_TOKEN_ID,
                ]
                if not self.contiguous and count >= 2:
                    placeholders.insert(1 + count // 2, _TEXT_BASE)
        ids = [prompt_ids[0], *placeholders, *prompt_ids[1:]]
        encoded["input_ids"] = torch.tensor([ids], dtype=torch.long)
        encoded["attention_mask"] = torch.ones(1, len(ids), dtype=torch.long)
        return encoded


def mock_audio_config(*, audio_tower: bool = True) -> SimpleNamespace:
    """A config shaped like the checkpoint's, for the audio resolver."""
    return SimpleNamespace(
        audio_token_id=MOCK_AUDIO_TOKEN_ID,
        image_token_id=None,
        **(
            {"audio_config": SimpleNamespace(hidden_size=MOCK_AUDIO_FEATURE_DIM)}
            if audio_tower
            else {}
        ),
    )


class MockAudioTokenizer:
    """The tokenizer surface :class:`~jlens.mmpilot.backend.GemmaPilotBackend`
    actually uses: ``__call__`` with ``add_special_tokens``, plus a template.

    Words are split into two-character pieces, so an answer like ``" cat"``
    encodes to **more than one token**. That is deliberate, and matches
    :meth:`MockPilotBackend.encode_candidate`: a mock whose every candidate is a
    single token would let first-token scoring pass every test the pilot has.
    """

    chat_template = "{# mock #}"

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        ids = []
        for word in str(text).split():
            for start in range(0, len(word), 2):
                piece = word[start : start + 2]
                ids.append(
                    _TEXT_BASE + (sum(map(ord, piece)) % (MOCK_VOCAB - _TEXT_BASE))
                )
        return {"input_ids": ids or [_PAD]}


class MockAudioTower(nn.Module):
    """Projects frames to residual width, with the same two stride-2 reductions
    the placeholder count assumes — so it returns exactly as many vectors as
    there are placeholders, which is the invariant the real model enforces."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Linear(MOCK_AUDIO_FEATURE_DIM, d_model, bias=False)
        with torch.no_grad():
            generator = torch.Generator().manual_seed(23)
            self.proj.weight.copy_(
                torch.randn(d_model, MOCK_AUDIO_FEATURE_DIM, generator=generator)
            )

    def forward(self, input_features, input_features_mask=None, **_kwargs):
        hidden = self.proj(input_features)
        for _ in range(2):
            t_out = (hidden.shape[1] + 2 - 3) // 2 + 1
            hidden = hidden[:, ::2][:, :t_out]
        return hidden


class _AudioBlock(nn.Module):
    """Block 0 mixes every position, so the final prompt token really depends
    on the audio span; later blocks are the identity."""

    def __init__(self, broadcast: bool) -> None:
        super().__init__()
        self.broadcast = broadcast

    def forward(self, hidden: torch.Tensor, **_kwargs) -> torch.Tensor:
        if self.broadcast:
            return hidden + hidden.mean(dim=1, keepdim=True)
        return hidden


class _AudioLanguageModel(nn.Module):
    def __init__(self, d_model: int, n_layers: int) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(5)
        self.embed_tokens = nn.Embedding(MOCK_VOCAB, d_model)
        with torch.no_grad():
            self.embed_tokens.weight.copy_(
                torch.randn(MOCK_VOCAB, d_model, generator=generator) * 0.05
            )
        self.layers = nn.ModuleList(
            _AudioBlock(broadcast=(index == 0)) for index in range(n_layers)
        )


class _AudioInner(nn.Module):
    def __init__(self, d_model: int, n_layers: int) -> None:
        super().__init__()
        self.language_model = _AudioLanguageModel(d_model, n_layers)
        self.audio_tower = MockAudioTower(d_model)


class MockAudioGemmaLike(nn.Module):
    """Gemma 4's module layout with a **working** audio tower.

    The tower's output is scattered into the placeholder positions and the
    tokens/features counts are checked exactly as ``Gemma4Model.forward`` checks
    them, so a mismatched input fails here for the same reason and with the same
    message it would fail on the real model.
    """

    def __init__(self, *, d_model: int = MOCK_D_MODEL, n_layers: int = 4) -> None:
        super().__init__()
        self.model = _AudioInner(d_model, n_layers)
        self.lm_head = nn.Linear(d_model, MOCK_VOCAB, bias=False)
        with torch.no_grad():
            generator = torch.Generator().manual_seed(13)
            rows = torch.randn(MOCK_VOCAB, d_model, generator=generator)
            self.lm_head.weight.copy_(rows / rows.norm(dim=-1, keepdim=True))
        self.config = SimpleNamespace(
            get_text_config=lambda: SimpleNamespace(
                hidden_size=d_model, num_hidden_layers=n_layers, vocab_size=MOCK_VOCAB
            ),
            image_token_id=None,
            audio_token_id=MOCK_AUDIO_TOKEN_ID,
            audio_config=SimpleNamespace(hidden_size=MOCK_AUDIO_FEATURE_DIM),
        )
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        input_features=None,
        input_features_mask=None,
        **_kwargs,
    ):
        hidden = self.model.language_model.embed_tokens(input_ids)
        if input_features is not None:
            features = self.model.audio_tower(input_features, input_features_mask)
            mask = input_ids == MOCK_AUDIO_TOKEN_ID
            if int(mask.sum()) != int(features.shape[1]):
                raise ValueError(
                    "Audio features and audio tokens do not match, tokens: "
                    f"{int(mask.sum())}, features: {int(features.shape[1])}"
                )
            hidden = hidden.clone()
            hidden[mask] = features.reshape(-1, hidden.shape[-1]).to(hidden.dtype)
        for block in self.model.language_model.layers:
            hidden = block(hidden)
        return SimpleNamespace(logits=self.lm_head(hidden))


def build_mock_audio_backend(**processor_kwargs):
    """``(backend, processor, resolved)`` — the MOCK spoken-audio world.

    A real :class:`~jlens.mmpilot.backend.GemmaPilotBackend` over a processor
    that reproduces ``Gemma4Processor``'s audio contract and a model that really
    consumes the tower's output. The feasibility notebook's MOCK branch runs the
    *same* audit code against this as its real branch runs against Gemma, so the
    two cannot drift.
    """
    from jlens.mmpilot.audio import resolve_audio_interface
    from jlens.mmpilot.backend import GemmaPilotBackend, resolve_processor_interface

    processor = MockAudioProcessor(**processor_kwargs)
    processor.tokenizer = MockAudioTokenizer()
    model = MockAudioGemmaLike()
    interface = resolve_processor_interface(processor, model.config)
    resolved = resolve_audio_interface(
        processor,
        model.config,
        model_repo_id="mock/gemma-like",
        model_revision="mock-rev",
        processor_revision="mock-rev",
    )
    backend = GemmaPilotBackend(
        model, processor, interface, device="cpu", audio_interface=resolved
    )
    return backend, processor, resolved


MOCK_LENS_CHECKSUM = "sha256:mock-identity-lens"


def run_mock_pilot(
    dataset_root: str | Path,
    run_dir: str | Path,
    *,
    supports_audio: bool = True,
    n_permutations: int = 20,
) -> dict:
    """Run every stage against the synthetic world and return the artifacts.

    This is what the notebook's MOCK path calls, and what the end-to-end test
    asserts on, so the two can never drift apart. Calling it twice on the same
    ``run_dir`` exercises resume.
    """
    import json as _json

    import torch as _torch

    from jlens.mmpilot import concepts as concepts_module
    from jlens.mmpilot import evidence as evidence_module
    from jlens.mmpilot import expansion as expansion_module
    from jlens.mmpilot import manifest as manifest_module
    from jlens.mmpilot import pipeline as pipeline_module
    from jlens.mmpilot.jspace import validate_lens
    from jlens.mmpilot.store import RunFingerprint, UnitStore

    dataset_root, run_dir = Path(dataset_root), Path(run_dir)
    if not (dataset_root / "spokencoco_manifest.json").is_file():
        build_mock_dataset(dataset_root)
    manifest_path = dataset_root / "spokencoco_manifest.json"
    payload = _json.loads(manifest_path.read_text(encoding="utf-8"))

    schema = manifest_module.inspect_manifest(payload)
    normalized = manifest_module.normalize_manifest(
        payload,
        schema,
        media_roots=[dataset_root],
        source_checksum=manifest_module.manifest_checksum(manifest_path),
        min_complete_groups=8,
    )
    # Visual evidence: read the COCO object annotations off disk, exactly as
    # the real path does. Without them nothing is a valid positive.
    instances_path = dataset_root / "annotations" / "instances_train2014.json"
    annotation_sources = expansion_module.discover_metadata_sources(
        [instances_path.parent], max_files=8, max_depth=1
    )
    groups = [dict(group) for group in normalized.groups]
    expansion_module.attach_concept_annotations(
        groups,
        [s for s in annotation_sources if s.source_kind == "coco_object_annotation"],
    )

    # The candidate universe comes from the annotation file, not from
    # MOCK_CONCEPTS: the mock exercises the same discovery the real path uses,
    # so the two cannot drift apart.
    universe = concepts_module.discover_category_universe(
        [s for s in annotation_sources if s.source_kind == "coco_object_annotation"]
    )
    candidate_concepts = universe.lexicon()

    # The CPU evidence audit, run in MOCK too so the notebook's audit stage and
    # this path can never drift apart.
    evidence_config = evidence_module.config_from_specs(universe.specs)
    audit = evidence_module.audit_groups(
        groups,
        config=evidence_config,
        source_checksums=evidence_module.source_checksums(
            [manifest_path, instances_path]
        ),
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    conversion = {"converter": "jlens.mmpilot.mock.run_mock_pilot", "mode": "mock"}
    evidence_module.persist_evidence_audit(
        run_dir / "synchronized_evidence_audit.json", audit, conversion=conversion
    )
    evidence_module.persist_synchronized_manifest(
        run_dir / "synchronized_evidence_manifest.json",
        groups,
        audit,
        original_checksum=normalized.source_checksum,
        conversion=conversion,
    )
    ranking = expansion_module.rank_concepts(
        groups,
        candidate_concepts,
        groups_per_concept=6,
        evidence_config=evidence_config,
    )
    evidence_module.persist_concept_ranking(
        run_dir / "concept_ranking.json",
        ranking,
        audit,
        requirements=expansion_module.ConceptRequirements().to_dict(),
        conversion=conversion,
    )
    # Every feasible concept, not the pilot's top two: this path exists to
    # exercise the pipeline, and a richer world gives retrieval and the
    # intervention controls something to fail at. The *selection* is still the
    # ranking's, so discovery and ranking are exercised rather than bypassed.
    selected_names = expansion_module.select_concepts(
        ranking,
        n_concepts=2,
        max_concepts=len(ranking),
        total_synchronized_records=len(groups),
    )
    selected_concepts = {name: candidate_concepts[name] for name in selected_names}

    subset = manifest_module.build_subset(
        groups,
        selected_concepts,
        groups_per_concept=6,
        negatives_per_concept=6,
        evidence_config=evidence_config,
    )
    leakage = manifest_module.check_split_leakage(subset)
    if not leakage["ok"]:
        raise RuntimeError(f"mock split leaked: {leakage}")

    backend = MockPilotBackend(supports_audio=supports_audio)
    lens = mock_lens()
    config = pipeline_module.PilotConfig(
        mode="mock",
        layers=MOCK_LAYERS,
        causal_layers=(MOCK_LAYERS[-1],),
        concepts=tuple(sorted(selected_concepts)),
        pursuit_k=8,
        pursuit_correlation_chunk_size=None,
        direction_top_k=4,
        n_permutations=n_permutations,
        n_target_examples=2,
    )
    store = UnitStore(
        run_dir,
        RunFingerprint(
            mode="mock",
            model_repo_id="mock/gemma-like",
            model_revision="mock-rev",
            processor_revision="mock-rev",
            layers=tuple(config.layers),
            lens_checksum=MOCK_LENS_CHECKSUM,
            manifest_checksum=normalized.source_checksum,
            split_id=subset["provenance"]["seed"],
            intervention_config={
                "alphas": list(config.alphas),
                "direction_top_k": config.direction_top_k,
            },
        ),
    )
    status = store.open()
    (run_dir / "derived_manifest.json").write_text(
        _json.dumps(normalized.to_dict(), indent=2), encoding="utf-8"
    )

    media = {
        "load_image": load_mock_media,
        "load_audio": lambda path: (load_mock_media(path), 16000),
    }
    available, blocked = pipeline_module.available_modalities(backend, config)

    probe = backend.build_inputs(
        prompt="Question: which one of these is present: dog? Answer:", modality="text"
    )
    from jlens.mmpilot.backend import run_invariance_gate

    invariance = run_invariance_gate(backend, probe, list(config.layers))

    capability_outcome, capability = pipeline_module.stage_capability(
        backend, store, subset, config, media, modalities=available
    )
    lens_validation = validate_lens(
        lens,
        lens_path=str(run_dir / "mock_lens.pt"),
        lens_checksum=MOCK_LENS_CHECKSUM,
        layers=config.layers,
        model_repo_id="mock/gemma-like",
        model_revision="mock-rev",
        expect_model_repo_id="mock/gemma-like",
        expect_model_revision="mock-rev",
        expect_d_model=MOCK_D_MODEL,
        expect_checksum=MOCK_LENS_CHECKSUM,
    )
    activation_outcome = pipeline_module.stage_activations(
        backend,
        store,
        subset,
        config,
        media,
        modalities=available,
        retained_concepts=capability["retained_concepts"],
        model_revision="mock-rev",
    )
    dictionaries = pipeline_module.build_dictionaries(
        lens, config.layers, backend, dtype=_torch.float32, build_chunk_rows=None
    )
    code_outcome = pipeline_module.stage_codes(
        store,
        activation_outcome.records,
        dictionaries,
        config,
        lens_checksum=MOCK_LENS_CHECKSUM,
    )
    control_outcome, reconstruction_control = pipeline_module.stage_reconstruction_control(
        store,
        activation_outcome.records,
        dictionaries,
        config,
        lens_checksum=MOCK_LENS_CHECKSUM,
        primary_layer=config.layers[-1],
    )
    representational = pipeline_module.stage_representational(
        store,
        activation_outcome.records,
        code_outcome.records,
        config,
        layer=config.layers[-1],
        modalities=available,
    )
    direction_outcome, directions = pipeline_module.stage_directions(
        store,
        code_outcome.records,
        activation_outcome.records,
        dictionaries,
        config,
        concepts=capability["retained_concepts"],
        modalities=available,
        lens_checksum=MOCK_LENS_CHECKSUM,
    )
    causal_outcome, interventions = pipeline_module.stage_causal(
        backend,
        store,
        subset,
        code_outcome.records,
        activation_outcome.records,
        directions,
        config,
        media,
        concepts=capability["retained_concepts"][:1],
        modalities=available,
        all_concepts=capability["retained_concepts"],
    )
    markdown, summary = pipeline_module.stage_report(
        store,
        config=config,
        capability=capability,
        lens_validation=lens_validation,
        codes=code_outcome.records,
        representational=representational,
        interventions=interventions,
        invariance=invariance,
        reconstruction_control=reconstruction_control,
        blocked_modalities=blocked,
        manifest_audit=normalized.audit,
    )
    return {
        "status": status,
        "store": store,
        "config": config,
        "normalized": normalized,
        "groups": groups,
        "evidence_audit": audit,
        "evidence_config": evidence_config,
        "ranking": ranking,
        "subset": subset,
        "leakage": leakage,
        "available_modalities": available,
        "blocked_modalities": blocked,
        "invariance": invariance,
        "capability": capability,
        "lens_validation": lens_validation,
        "representational": representational,
        "reconstruction_control": reconstruction_control,
        "interventions": interventions,
        "directions": directions,
        "markdown": markdown,
        "summary": summary,
        "outcomes": {
            "capability": capability_outcome,
            "activation": activation_outcome,
            "jspace": code_outcome,
            "reconstruction_control": control_outcome,
            "direction": direction_outcome,
            "intervention": causal_outcome,
        },
    }


def mock_lens(layers: Sequence[int] = MOCK_LAYERS, *, d_model: int = MOCK_D_MODEL) -> JacobianLens:
    """The frozen "text-calibrated" lens for the mock.

    Identity Jacobians are the *correct* ones for :class:`MockGemmaLike`
    (every block after the first is the identity and the final norm is too),
    so the mock's J-space is the real J-space of that model rather than a
    convenient fiction.
    """
    return JacobianLens(
        jacobians={int(layer): torch.eye(d_model) for layer in layers},
        n_prompts=100,
        d_model=d_model,
    )


#: The three-layer mock stand-in for the published layer-35/38/40 artifacts.
MOCK_PUBLISHED_LAYERS: tuple[int, ...] = (2, 3, 4)

#: A mock layer that must be refused: it stands in for layer 32, which was
#: tested and failed confirmation.
MOCK_FAILED_CONFIRMATION_LAYER = 1


def build_mock_published_lenses(
    root: str | Path,
    *,
    layers: Sequence[int] = MOCK_PUBLISHED_LAYERS,
    d_model: int = MOCK_D_MODEL,
    model_repo_id: str = "mock/gemma-like",
    model_revision: str = "mockrevision0000000000000000000000000000",
    scale: int = 100,
    include_failed_layer: bool = True,
) -> dict:
    """Write per-layer published artifacts in the real calibration's format.

    One ``.pt`` per layer plus the ``.json`` sidecar
    :func:`jlens.calibration.publication.build_artifact` writes, so the MOCK run
    exercises :func:`jlens.mmpilot.published_lens.load_published_lenses` — the
    schema inspection, the checksum agreement, the confirmation clause — rather
    than skipping straight to an in-memory lens.

    ``include_failed_layer`` also writes a layer whose confirmation *failed*, in
    the shape :func:`jlens.calibration.publication.record_failed_layer` produces.
    Nothing loads it; it exists so a test can point the loader at it and watch
    the refusal happen.

    Returns ``{"specs", "failed_spec", "expectations", "root"}`` with the specs
    in the shape the notebook's configuration uses.
    """
    from jlens.metadata import file_sha256
    from jlens.mmpilot.published_lens import (
        EXPECTED_ARTIFACT_FORMAT,
        PublishedLensExpectations,
        PublishedLensSpec,
    )
    from jlens.mmpilot.store import payload_checksum

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    def write(layer: int, *, passed: bool) -> dict:
        destination = root / f"lens.layer{layer}.scale{scale}.validated.pt"
        JacobianLens(
            jacobians={int(layer): torch.eye(d_model)},
            n_prompts=scale,
            d_model=d_model,
        ).save(str(destination))
        checksum = file_sha256(str(destination))
        artifact = {
            "artifact_format_version": EXPECTED_ARTIFACT_FORMAT,
            "protocol_version": "mock.calibration.v1",
            "frozen": True,
            "validated": bool(passed),
            "model_repo_id": model_repo_id,
            "model_revision": model_revision,
            "tokenizer_repo_id": model_repo_id,
            "tokenizer_revision": model_revision,
            "physical_layer": int(layer),
            "normalized_depth": round(int(layer) / MOCK_N_LAYERS, 4),
            "d_model": int(d_model),
            "target_layer": MOCK_N_LAYERS - 1,
            "hook_site": "block_output",
            "residual_convention": (
                "pre-final-norm residual after the block; the exact input to "
                "block l+1"
            ),
            "vector_orientation": (
                "J_l maps a layer-l residual into the final-layer basis; applied "
                "as residual @ J_l.T"
            ),
            "normalization_convention": "readout is lm_head(final_norm(J_l @ h))",
            "logit_softcap": 30.0,
            "calibration_modality": "text-only",
            "spokencoco_used": False,
            "multimodal_data_used": False,
            "cross_modal_alignment": False,
            "modality_specific_lens": False,
            "corpus_id": "mock/wikitext-like",
            "corpus_revision": "mock",
            "n_fitting_prompts": int(scale),
            "scale_point": int(scale),
            "gate_digest": "sha256:mock-gate",
            "confirmation_protocol": "mock-confirmation-v1",
            "confirmation_failed_checks": [] if passed else ["mock_rank_criterion"],
            "confirmation_metrics": {"mean_reciprocal_rank": 0.9 if passed else 0.1},
            "validation_protocol": "mock-validation-v1",
            "validation_failed_checks": [],
            "estimator": "jlens.fitting.fit (upstream, unmodified)",
            "objective": "not_applicable_estimator_is_a_sample_mean",
            "lens_path": str(destination),
            "lens_checksum": checksum,
        }
        if not passed:
            artifact["published"] = False
            artifact["reason"] = "did not pass the confirmation gate"
        artifact["artifact_checksum"] = payload_checksum(artifact)
        destination.with_suffix(".json").write_text(
            json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return {"layer": int(layer), "path": str(destination), "expect_sha256": checksum}

    specs = [
        PublishedLensSpec(**write(int(layer), passed=True)) for layer in sorted(layers)
    ]
    failed_spec = (
        PublishedLensSpec(**write(MOCK_FAILED_CONFIRMATION_LAYER, passed=False))
        if include_failed_layer
        else None
    )
    expectations = PublishedLensExpectations(
        model_repo_id=model_repo_id,
        model_revision=model_revision,
        scale_point=int(scale),
        d_model=int(d_model),
        confirmed_layers=tuple(int(layer) for layer in sorted(layers)),
        failed_confirmation_layers=(MOCK_FAILED_CONFIRMATION_LAYER,),
    )
    return {
        "root": str(root),
        "specs": specs,
        "failed_spec": failed_spec,
        "expectations": expectations,
    }


#: The MOCK stand-in for physical layer 32 — an *earlier* layer than the mock
#: published set (2/3/4), published under the extension's own schema at its own
#: scale. Its whole job is to let the discovery chain run on CPU.
MOCK_EXTENSION_LAYER = 1
MOCK_EXTENSION_SCALE = 250

#: The deliberate defects :func:`build_mock_extension_run` can write, so a test
#: watches a refusal happen rather than asserting that it would.
MOCK_EXTENSION_DEFECTS: tuple[str, ...] = (
    "lens_bytes",
    "report_checksum",
    "extension_checksum",
    "duplicate",
    "wrong_layer",
    "wrong_scale",
    "unconfirmed",
)


def build_mock_extension_run(
    root: str | Path,
    *,
    layer: int = MOCK_EXTENSION_LAYER,
    scale: int = MOCK_EXTENSION_SCALE,
    d_model: int = MOCK_D_MODEL,
    model_repo_id: str = "mock/gemma-like",
    model_revision: str = "mockrevision0000000000000000000000000000",
    mode: str = "real",
    verdict: str = "EARLY_LAYER_CALIBRATION_GO",
    publish: bool = True,
    corrupt: str | None = None,
) -> dict:
    """Write a mock early-layer extension run, in the real run's exact layout.

    Produces ``artifacts/early_layer_extension_report.json`` plus, under
    ``artifacts/published``, the lens, the base ``.json`` sidecar
    :func:`jlens.calibration.publication.build_artifact` writes, and the
    ``.extension.json`` sidecar
    :func:`jlens.calibration.extension.publish_early_layer` writes beside it.
    That is the entire input to
    :func:`jlens.mmpilot.l32_followup.discover_published_l32_lens`, so the MOCK
    exercises discovery instead of skipping to a hard-coded path.

    Args:
        corrupt: One defect from :data:`MOCK_EXTENSION_DEFECTS`, or ``None``.
            ``"lens_bytes"`` rewrites the file after both sidecars are written —
            exactly the state a partial Drive sync leaves behind.
    """
    from jlens.calibration.publication import ARTIFACT_FORMAT_VERSION
    from jlens.metadata import file_sha256
    from jlens.mmpilot.l32_followup import (
        EXTENSION_ARTIFACT_SCHEMA,
        EXTENSION_REPORT_SCHEMA,
        PUBLISHED_STATUS,
        l32_expectations,
    )
    from jlens.mmpilot.store import payload_checksum

    if corrupt is not None and corrupt not in MOCK_EXTENSION_DEFECTS:
        raise ValueError(
            f"unknown defect {corrupt!r}; known defects are {MOCK_EXTENSION_DEFECTS}"
        )

    root = Path(root)
    published = root / "artifacts" / "published"
    published.mkdir(parents=True, exist_ok=True)

    recorded_layer = 99 if corrupt == "wrong_layer" else int(layer)
    recorded_scale = 100 if corrupt == "wrong_scale" else int(scale)
    validated = corrupt != "unconfirmed"

    destination = (
        published / f"lens.early.layer{int(layer)}.scale{int(scale)}.validated.pt"
    )
    JacobianLens(
        jacobians={int(layer): torch.eye(d_model)},
        n_prompts=int(scale),
        d_model=int(d_model),
    ).save(str(destination))
    checksum = file_sha256(str(destination))

    confirmation_metrics = {
        "mean_reciprocal_rank": 0.4058,
        "median_midrank": 3.5,
        "top_10_inclusion": 0.574,
        "n_confirmation_prompts": 256,
    }
    base = {
        "artifact_format_version": ARTIFACT_FORMAT_VERSION,
        "protocol_version": "research-grade-early-layer-jlens-extension-v1",
        "frozen": True,
        "validated": bool(validated),
        "model_repo_id": model_repo_id,
        "model_revision": model_revision,
        "tokenizer_repo_id": model_repo_id,
        "tokenizer_revision": model_revision,
        "physical_layer": int(layer),
        "normalized_depth": round(int(layer) / MOCK_N_LAYERS, 4),
        "d_model": int(d_model),
        "target_layer": MOCK_N_LAYERS - 1,
        "hook_site": "block_output",
        "residual_convention": (
            "pre-final-norm residual after the block; the exact input to block l+1"
        ),
        "vector_orientation": (
            "J_l maps a layer-l residual into the final-layer basis; applied as "
            "residual @ J_l.T"
        ),
        "normalization_convention": "readout is lm_head(final_norm(J_l @ h))",
        "logit_softcap": 30.0,
        "calibration_modality": "text-only",
        "spokencoco_used": False,
        "multimodal_data_used": False,
        "cross_modal_alignment": False,
        "modality_specific_lens": False,
        "corpus_id": "mock/wikitext-like",
        "corpus_revision": "mock",
        "confirmation_split_checksum": "sha256:mock-confirmation-split",
        "n_fitting_prompts": int(scale),
        "scale_point": int(scale),
        "gate_digest": "sha256:mock-extension-gate",
        "confirmation_protocol": "mock-extension-confirmation-v1",
        "confirmation_failed_checks": [] if validated else ["mock_rank_criterion"],
        "confirmation_metrics": dict(confirmation_metrics),
        "validation_protocol": "mock-extension-development-v1",
        "validation_failed_checks": [],
        "estimator": "jlens.fitting.fit (upstream, unmodified)",
        "objective": "not_applicable_estimator_is_a_sample_mean",
        "lens_path": str(destination),
        "lens_checksum": checksum,
    }
    base["artifact_checksum"] = payload_checksum(base)
    destination.with_suffix(".json").write_text(
        json.dumps(base, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    def write_extension_sidecar(path: Path) -> dict:
        payload = {
            "schema": EXTENSION_ARTIFACT_SCHEMA,
            "protocol_version": "research-grade-early-layer-jlens-extension-v1",
            "protocol_digest": "sha256:mock-extension-protocol",
            "frozen": True,
            "validated": bool(validated),
            "publication_status": PUBLISHED_STATUS,
            "physical_layer": int(recorded_layer),
            "scale_point": int(recorded_scale),
            "parent_run_root": str(root / "parent"),
            "parent_accumulator_checksum": "sha256:mock-accumulator",
            "old_confirmation_set_reused": False,
            "old_development_set_reused": False,
            "model_repo_id": model_repo_id,
            "model_revision": model_revision,
            "n_fitting_prompts": int(recorded_scale),
            "fit_order_protocol": "parent-prefix-pinned-nested-order-v1",
            "development_split_checksum": "sha256:mock-development-split",
            "confirmation_split_checksum": "sha256:mock-confirmation-split",
            "fresh_splits_manifest_checksum": "sha256:mock-fresh-splits",
            "confirmation_metrics": dict(confirmation_metrics),
            "confirmation_failed_checks": list(base["confirmation_failed_checks"]),
            "lens_path": str(destination),
            "lens_checksum": checksum,
            "base_artifact_checksum": base["artifact_checksum"],
            "existing_publications_unchanged": [35, 38, 40],
            "parent_run_written": False,
            "calibration_modality": "text-only",
            "spokencoco_used": False,
            "multimodal_data_used": False,
        }
        payload["artifact_checksum"] = (
            "sha256:deliberately-wrong"
            if corrupt == "extension_checksum"
            else payload_checksum(payload)
        )
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return payload

    extension = write_extension_sidecar(destination.with_suffix(".extension.json"))
    if corrupt == "duplicate":
        write_extension_sidecar(
            published
            / f"lens.early.layer{int(layer)}.scale{int(scale)}.copy.extension.json"
        )

    report_checksum = (
        "sha256:deliberately-wrong" if corrupt == "report_checksum" else checksum
    )
    report = {
        "schema": EXTENSION_REPORT_SCHEMA,
        "mode": mode,
        "protocol_version": "research-grade-early-layer-jlens-extension-v1",
        "fingerprint_digest": "sha256:mock-extension-fingerprint",
        "gate_digest": "sha256:mock-extension-gate",
        "parent_immutability_proof": {"immutable": True, "n_files_checked": 7},
        "scale_selection": {
            "selected_scale": int(scale),
            "selection_checksum": "sha256:mock-selection",
        },
        "early_layer_verdict": {
            "verdict": verdict,
            "earlier_layers_passing_confirmation": [int(layer)] if publish else [],
            "statement": "mock verdict; proves pipeline behaviour only",
        },
        "publication": {
            "n_published": 1 if publish else 0,
            "n_failed": 0 if publish else 1,
            "published_layers": [int(layer)] if publish else [],
            "failed_layers": [] if publish else [int(layer)],
            "published_checksums": (
                {str(int(layer)): report_checksum} if publish else {}
            ),
            "failed_layers_marked_validated": False,
        },
        "resume": {"status": "resuming", "invalid_units": []},
        "mock_proves_pipeline_only": mode != "real",
    }
    report_path = root / "artifacts" / "early_layer_extension_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if corrupt == "lens_bytes":
        JacobianLens(
            jacobians={int(layer): torch.eye(d_model) * 2.0},
            n_prompts=int(scale),
            d_model=int(d_model),
        ).save(str(destination))

    return {
        "root": str(root),
        "report_path": str(report_path),
        "published_dir": str(published),
        "lens_path": str(destination),
        "lens_checksum": checksum,
        "base_sidecar": base,
        "extension_sidecar": extension,
        "report": report,
        "expectations": l32_expectations(
            model_repo_id=model_repo_id,
            model_revision=model_revision,
            d_model=int(d_model),
            layer=int(layer),
            scale=int(scale),
        ),
    }


# ------------------------------------------- a completed run to be independent of

#: The unit stages a mock completed run writes into. Enough for
#: :func:`jlens.mmpilot.l32_resolution.harvest_excluded_identities` to have to
#: walk more than one shape, which is the behaviour that matters.
MOCK_COMPLETED_STAGES: tuple[str, ...] = ("capability", "activation")


def build_mock_completed_run(
    root: str | Path,
    groups: Sequence[Mapping],
    *,
    modalities: Sequence[str] = ("text", "image", "spoken_audio"),
    layer: int = MOCK_EXTENSION_LAYER,
    run_fingerprint: str = "sha256:mock-completed-followup",
    write_split_provenance: bool = True,
) -> dict:
    """Write a completed-run directory whose media a later study must avoid.

    The point is not to simulate a whole study — it is to reproduce the *shape*
    the exclusion harvest has to read: per-unit JSON under ``units/<stage>/``
    with the identity nested inside a ``payload``, a ``fingerprint.json``
    carrying ``source_sample_ids`` / ``target_sample_ids``, and a
    ``split_provenance.json`` that carries none of them. A harvest that only
    knew one of those three would pass a test written against that one and lose
    identities on the real directory.

    Nothing here is scientific. The unit payloads carry identities and nothing
    a verdict could be computed from.
    """
    root = Path(root)
    groups = [dict(group) for group in groups]
    written = 0
    for stage in MOCK_COMPLETED_STAGES:
        stage_dir = root / "units" / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        for group in groups:
            for modality in modalities:
                identifier = f"{group['group_id']}::{modality}"
                payload = {
                    "sample_id": identifier,
                    "group_id": str(group["group_id"]),
                    "image_id": str(group["image_id"]),
                    "modality": modality,
                    "split": str(group.get("split", "test")),
                    "layer": int(layer),
                }
                path = stage_dir / f"{safe_key(identifier, stage)}.json"
                path.write_text(
                    json.dumps(
                        {
                            "schema": "jlens.mmpilot.unit.v1",
                            "stage": stage,
                            "key": path.stem,
                            "fingerprint_digest": run_fingerprint,
                            "payload": payload,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                written += 1

    root.mkdir(parents=True, exist_ok=True)
    (root / "fingerprint.json").write_text(
        json.dumps(
            {
                "protocol": "mmpilot.l32_open_prompt_followup.v1",
                "fingerprint_digest": run_fingerprint,
                "source_sample_ids": sorted(
                    str(g["group_id"]) for g in groups if g.get("split") == "train"
                ),
                "target_sample_ids": sorted(
                    str(g["group_id"]) for g in groups if g.get("split") != "train"
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if write_split_provenance:
        (root / "split_provenance.json").write_text(
            json.dumps(
                {
                    "seed": "mock-completed",
                    "n_groups": len(groups),
                    "n_distinct_images": len({str(g["image_id"]) for g in groups}),
                    "note": "carries counts, deliberately no identities",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return {
        "root": str(root),
        "n_units": written,
        "n_groups": len(groups),
        "group_ids": sorted(str(g["group_id"]) for g in groups),
        "image_ids": sorted({str(g["image_id"]) for g in groups}),
        "audio_paths": sorted({str(g["audio_path"]) for g in groups if g.get("audio_path")}),
        "captions": sorted({str(g["caption"]) for g in groups if g.get("caption")}),
    }
