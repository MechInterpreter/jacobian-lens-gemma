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
