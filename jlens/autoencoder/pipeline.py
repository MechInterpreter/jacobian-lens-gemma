# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Shared setup for every stage of the experiment.

The four scripts (dataset / reconstructor / adapter / evaluation) all need the
same frozen inputs assembled the same way: the model (real or mock), the
checksum-verified lens, the layer-14 J-space dictionary, the constant verbalizer
prompt, the stop tokens, and the contextual phrase tokenizer. Assembling them in
one place is what makes "the reconstructor and the adapter saw the same phrase
segmentation" a fact rather than a hope.
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

from jlens.autoencoder.conditioning import (
    SoftPrefixConditioner,
    assert_gemma_frozen,
    measure_memory_scale,
)
from jlens.autoencoder.config import AutoencoderConfig
from jlens.autoencoder.errors import AutoencoderError
from jlens.autoencoder.prompting import (
    DEFAULT_PROMPT_ID,
    VerbalizerPrompt,
    build_verbalizer_prompt,
    phrase_target_ids,
    resolve_end_of_turn_id,
)
from jlens.lens import JacobianLens
from jlens.pursuit import JSpaceDictionary

logger = logging.getLogger("jspace_language_autoencoder")

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


@dataclass
class Stack:
    """Everything frozen, assembled and verified."""

    model: object
    lens: JacobianLens
    dictionary: JSpaceDictionary
    prompt: VerbalizerPrompt
    conditioner: SoftPrefixConditioner
    source_layer: int
    stop_token_ids: list[int]
    pad_token_id: int
    end_of_turn: dict
    memory_scale: dict
    documents: list[str] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)
    is_mock: bool = False

    @property
    def tokenizer(self):
        return self.model.tokenizer

    @property
    def d_model(self) -> int:
        return int(self.model.d_model)

    def phrase_token_ids(self, phrase: str) -> list[int]:
        """The phrase's ids **as the assistant continuation of the constant
        prompt** — the single segmentation used by the dataset, the
        reconstructor, and the adapter."""
        return phrase_target_ids(
            self.tokenizer,
            self.prompt,
            phrase,
            end_of_turn_id=self.end_of_turn["end_of_turn_id"],
            include_end_of_turn=False,
        )["phrase_token_ids"]

    def phrase_target_ids(self, phrase: str) -> list[int]:
        """Teacher-forcing target: phrase ids plus the end-of-turn token."""
        return phrase_target_ids(
            self.tokenizer,
            self.prompt,
            phrase,
            end_of_turn_id=self.end_of_turn["end_of_turn_id"],
            include_end_of_turn=True,
        )["target_token_ids"]

    def token_id_map(self, phrases: Sequence[str]) -> dict[str, list[int]]:
        return {phrase: self.phrase_token_ids(phrase) for phrase in dict.fromkeys(phrases)}

    def target_id_map(self, phrases: Sequence[str]) -> dict[str, list[int]]:
        return {phrase: self.phrase_target_ids(phrase) for phrase in dict.fromkeys(phrases)}


def load_verified_lens(config: AutoencoderConfig, *, runs_root: str) -> tuple[JacobianLens, dict]:
    """Load the frozen lens and verify its checksum before anything uses it."""
    path = os.path.join(runs_root, config.lens.run_dir_name, config.lens.artifact_relpath)
    if not os.path.isfile(path):
        raise AutoencoderError(
            f"lens artifact not found at {path}; set lens.run_dir_name / "
            f"lens.artifact_relpath, or point --runs-root at the directory holding "
            f"the pilot run"
        )
    observed = file_sha256(path)
    expected = config.lens.expect_file_sha256
    if expected is not None and observed != expected:
        raise AutoencoderError(
            f"lens checksum mismatch at {path}: expected {expected}, got {observed}. "
            f"Refusing to run against an unverified lens."
        )
    lens = JacobianLens.load(path)
    layer = config.dataset.source_layer
    if layer not in lens.jacobians:
        raise AutoencoderError(
            f"the loaded lens has no layer {layer} (fitted layers: {lens.source_layers})"
        )
    return lens, {
        "lens_path": path,
        "lens_sha256": observed,
        "lens_sha256_verified": expected is not None,
        "lens_source_layers": list(lens.source_layers),
        "lens_n_prompts": lens.n_prompts,
        "lens_d_model": lens.d_model,
    }


def build_stack(
    config: AutoencoderConfig,
    *,
    mock: bool,
    allow_model_load: bool = False,
    device_map: str | None = None,
    runs_root: str = "runs",
    n_mock_documents: int = 96,
) -> Stack:
    """Assemble the frozen stack, real or mock.

    ``mock=True`` is the CPU smoke path: no checkpoint, no download, no lens
    file — the same code paths with a deterministic stand-in
    (:mod:`jlens.autoencoder.mocks`).
    """
    if mock:
        from jlens.autoencoder.mocks import build_mock_stack

        mock_stack = build_mock_stack(
            source_layer=config.dataset.source_layer, n_documents=n_mock_documents
        )
        model = mock_stack.model
        lens = mock_stack.lens
        documents = mock_stack.documents
        provenance = {
            "mock": True,
            "note": "deterministic CPU mock; no Gemma checkpoint was loaded",
        }
    else:
        from jlens.gemma4 import load_gemma4, verify_architecture

        model, load_info = load_gemma4(
            config.model.repo_id,
            revision=config.model.revision,
            dtype=_DTYPES[config.model.dtype],
            device_map=device_map,
            allow_model_load=allow_model_load or config.model.allow_model_load,
            token=os.environ.get("HF_TOKEN"),
        )
        report = verify_architecture(
            model,
            expect_n_layers=config.model.expect_n_layers,
            expect_d_model=config.model.expect_d_model,
            expect_vocab_size=config.model.expect_vocab_size,
        )
        lens, lens_info = load_verified_lens(config, runs_root=runs_root)
        documents = []
        provenance = {
            "mock": False,
            "load_info": load_info,
            "architecture": report.to_dict(),
            **lens_info,
        }
    assert_gemma_frozen(model, where="stack construction")

    layer = int(config.dataset.source_layer)
    dictionary = JSpaceDictionary.from_lens(
        lens,
        layer,
        model._lm_head.weight,
        device="cpu" if mock else model._lm_head.weight.device,
        dtype=_DTYPES.get(config.pursuit.atoms_dtype, torch.float32),
        build_chunk_rows=config.pursuit.build_chunk_rows,
    )
    end_of_turn = resolve_end_of_turn_id(model.tokenizer)
    prompt = build_verbalizer_prompt(
        model.tokenizer,
        n_memory_tokens=config.adapter.n_memory_tokens,
        prompt_id=DEFAULT_PROMPT_ID,
    )
    pad = getattr(model.tokenizer, "pad_token_id", None)
    if not isinstance(pad, int):
        pad = end_of_turn["end_of_turn_id"]
    memory_scale = measure_memory_scale(model, prompt.input_ids(batch=1))
    return Stack(
        model=model,
        lens=lens,
        dictionary=dictionary,
        prompt=prompt,
        conditioner=SoftPrefixConditioner(),
        source_layer=layer,
        stop_token_ids=list(end_of_turn["stop_token_ids"]),
        pad_token_id=int(pad),
        end_of_turn=end_of_turn,
        memory_scale=memory_scale,
        documents=documents,
        provenance={
            **provenance,
            "dictionary": dict(dictionary.provenance),
            "end_of_turn": end_of_turn,
            "memory_scale": memory_scale,
            "verbalizer_prompt": prompt.to_debug_dict(),
        },
        is_mock=bool(mock),
    )


def resolve_documents(
    config: AutoencoderConfig, stack: Stack, *, n_documents: int | None = None
) -> list[str]:
    """Corpus documents for the configured source."""
    if config.dataset.corpus == "mock":
        if not stack.documents:
            from jlens.autoencoder.mocks import mock_corpus

            return mock_corpus(n_documents=n_documents or 96)
        return stack.documents
    from jlens.autoencoder.dataset import load_wikitext_documents

    wanted = n_documents or config.dataset.max_documents
    documents = load_wikitext_documents(
        wanted, min_chars=config.dataset.min_document_chars
    )
    if not documents:
        raise AutoencoderError(
            "WikiText-103 returned no documents at the configured minimum length"
        )
    return documents
