# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""End-to-end mock test of the J-space decomposition pipeline: the exact
sequence the Colab notebook's real-model cells run, on the Gemma4-shaped
CPU mock. Validates orchestration and output schemas — NOT real-model
decomposition quality."""

import json

import pytest
import torch

from jlens.cones import (
    cone_trajectory,
    load_cone_records,
    make_cone_record,
    recurring_signatures,
    save_cone_records,
)
from jlens.evaluation import capture_residuals
from jlens.fitting import fit
from jlens.gemma4 import Gemma4LensModel
from jlens.ignition import candidate_ignition_signals, export_transition_records
from jlens.metadata import prompt_hashes
from jlens.pursuit import JSpaceDictionary, PursuitSettings, gradient_pursuit

from .mock_gemma4 import MockGemma4ForConditionalGeneration, MockTokenizer


@pytest.fixture(scope="module")
def pipeline():
    hf_model = MockGemma4ForConditionalGeneration(n_layers=6, d_model=8, vocab=32)
    model = Gemma4LensModel(hf_model, MockTokenizer())
    lens = fit(
        model,
        ["abcdefghij klmnop qrstuv " * 3, "zyxw vutsrq ponmlkj " * 3],
        source_layers=[1, 2, 3],
        dim_batch=4,
        max_seq_len=48,
    )
    return model, lens


def test_full_decomposition_pipeline(pipeline, tmp_path):
    model, lens = pipeline
    layers = [1, 2, 3]
    k_values = [1, 3]
    prompt = {"slug": "mock", "category": "factual", "format": "plain",
              "text": "abcdefghij klmnop qrstuv " * 2, "positions": [-2, -1]}

    # Capture residuals (one forward), as notebook cell 8 does.
    residuals, model_logits, input_ids = capture_residuals(
        model, prompt["text"], layers=layers, positions=prompt["positions"],
        max_seq_len=64,
    )
    assert set(residuals) == set(layers)
    assert all(r.shape == (2, 8) for r in residuals.values())
    model_top1 = model_logits.argmax(-1)
    row_hash = prompt_hashes([prompt["text"]])[0]

    meta = []
    for i, pos in enumerate(prompt["positions"]):
        tok = int(input_ids[0, pos])
        meta.append({
            "position": pos,
            "input_token_id": tok,
            "input_token": model.tokenizer.decode([tok]),
            "model_top1_id": int(model_top1[i]),
        })

    # Dictionary from the frozen lens + the model's real unembedding weight
    # (notebook cell 10); the lens matrices must be left untouched.
    lens_before = {l: J.clone() for l, J in lens.jacobians.items()}
    run_provenance = {"run_id": "mock", "lens_fingerprint": "sha256:mock",
                      "model_revision": "mock"}
    cones_dir = tmp_path / "cones"
    cones_dir.mkdir()
    for layer in layers:
        dictionary = JSpaceDictionary.from_lens(
            lens, layer, model._lm_head.weight
        )
        assert dictionary.n_atoms == 32 and dictionary.d_model == 8
        for k in k_values:
            settings = PursuitSettings(k=k, correlation_chunk_size=16)
            result = gradient_pursuit(residuals[layer], dictionary, settings)
            records = []
            for m, record in zip(meta, result.to_records(), strict=True):
                labels = [model.tokenizer.decode([i]) for i in record["token_ids"]]
                records.append(make_cone_record(
                    record, decoded_labels=labels, layer=layer,
                    position=m["position"],
                    input_token_id=m["input_token_id"],
                    input_token=m["input_token"],
                    prompt_hash=row_hash, prompt_slug=prompt["slug"],
                    prompt_format=prompt["format"],
                    run_provenance=run_provenance,
                ))
            save_cone_records(
                records, str(cones_dir / f"cones_layer{layer:02d}_k{k:02d}.json")
            )
    for layer in layers:  # frozen lens untouched
        torch.testing.assert_close(lens.jacobians[layer], lens_before[layer])

    # Trajectories + candidate-ignition (notebook cell 11), per k.
    for k in k_values:
        all_records = []
        for layer in layers:
            all_records.extend(load_cone_records(
                str(cones_dir / f"cones_layer{layer:02d}_k{k:02d}.json")
            ))
        assert len(all_records) == len(layers) * 2
        for position in (-2, -1):
            per_pos = [r for r in all_records if r["position"] == position]
            transitions = cone_trajectory(per_pos)
            assert len(transitions) == len(layers) - 1
            by_layer = {r["layer"]: r for r in per_pos}
            top1 = next(m["model_top1_id"] for m in meta
                        if m["position"] == position)
            signals = candidate_ignition_signals(
                transitions, by_layer, model_top1_id=top1
            )
            assert len(signals) == len(transitions)
            out = tmp_path / f"ignition_k{k}_pos{abs(position)}.json"
            export_transition_records(signals, str(out))
            json.load(open(out, encoding="utf-8"))
        table = recurring_signatures(all_records)
        assert sum(row["count"] for row in table) == len(all_records)


def test_pipeline_deterministic(pipeline):
    """Repeated capture + pursuit is bit-identical (eval-mode mock model,
    deterministic pursuit)."""
    model, lens = pipeline
    prompt_text = "abcdefghij klmnop qrstuv"
    dictionary = JSpaceDictionary.from_lens(lens, 2, model._lm_head.weight)

    def run():
        residuals, _, _ = capture_residuals(
            model, prompt_text, layers=[2], positions=[-1], max_seq_len=48
        )
        result = gradient_pursuit(
            residuals[2], dictionary, PursuitSettings(k=3)
        )
        return result.to_records()

    assert json.dumps(run(), sort_keys=True) == json.dumps(run(), sort_keys=True)
