# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""End-to-end test of scripts/run_generative_validation.py on the
Gemma4-shaped mock (load_gemma4 monkeypatched; no network). Validates the
exact code path the real GPU run takes: gates, per-layer dictionary +
pursuit, the full condition battery, records, and the go/no-go artifacts."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

from jlens.lens import JacobianLens

from .mock_gemma4 import MockGemma4ForConditionalGeneration, MockTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent


def _import_runner():
    spec = importlib.util.spec_from_file_location(
        "run_generative_validation",
        REPO_ROOT / "scripts" / "run_generative_validation.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MockWithGenerate(MockGemma4ForConditionalGeneration):
    """Mock plus an independent greedy ``generate`` (uncached loop through
    the same modules) for the runner's greedy-equivalence gate."""

    def generate(self, *, input_ids, attention_mask=None, max_new_tokens, do_sample):
        assert do_sample is False
        ids = input_ids
        cap = self.config.get_text_config().final_logit_softcapping
        for _ in range(max_new_tokens):
            with torch.no_grad():
                hidden = self.model.language_model(input_ids=ids).last_hidden_state
                logits = self.lm_head(self.model.language_model.norm(hidden))[0, -1]
                logits = cap * torch.tanh(logits / cap)
            next_id = int(logits.argmax())
            ids = torch.cat([ids, torch.tensor([[next_id]])], dim=1)
        return ids


def _mock_load_gemma4(*args, **kwargs):
    from jlens.gemma4 import Gemma4LensModel

    if not kwargs.get("allow_model_load"):
        raise RuntimeError("allow_model_load")
    model = Gemma4LensModel(MockWithGenerate(), MockTokenizer())
    load_info = {
        "model_repo_id": "mock/gemma4",
        "model_revision": "f" * 40,
        "tokenizer_repo_id": "mock/gemma4",
        "tokenizer_revision": "f" * 40,
        "tokenizer_class": "MockTokenizer",
        "dtype": "torch.float32",
        "device_map": "None",
        "transformers_version": "mock",
        "torch_version": "mock",
    }
    return model, load_info


@pytest.fixture()
def experiment(tmp_path):
    """Config + benchmark + mock lens artifact retargeted at the 6-layer/d8
    mock, laid out the way the runner expects under --runs-root."""
    yaml = pytest.importorskip("yaml")
    with open(
        REPO_ROOT / "configs" / "gemma_generative_validation.yaml", encoding="utf-8"
    ) as handle:
        config = yaml.safe_load(handle)

    runs_root = tmp_path / "runs"
    lens_dir = runs_root / "pilot_mock" / "artifacts"
    lens_dir.mkdir(parents=True)
    torch.manual_seed(7)
    lens = JacobianLens(
        {1: torch.randn(8, 8), 3: torch.randn(8, 8)}, n_prompts=1, d_model=8
    )
    lens.save(str(lens_dir / "lens.pt"))

    manifest = {
        "version": 1,
        "dev": [
            {
                "example_id": "mock-a",
                "category": "compound_word",
                "source_prompt": "A house for birds is called a",
                "control_prompt": "A tank for fish is called an",
                "target_phrase": " birdhouse",
                "target_token_ids": None,
                "extraction_position": -1,
                "manual_subcone": {"1": [0]},
            },
            {
                "example_id": "mock-b",
                "category": "noun_phrase",
                "source_prompt": "A region light cannot escape is called a",
                "control_prompt": None,
                "target_phrase": " black hole",
                "target_token_ids": None,
                "extraction_position": -1,
                "manual_subcone": None,
            },
        ],
        "heldout": [],
    }
    manifest_path = tmp_path / "benchmark.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    config["model"].update(
        allow_model_load=False,
        expect_n_layers=6,
        expect_d_model=8,
        expect_vocab_size=32,
    )
    config["lens"].update(
        run_dir_name="pilot_mock",
        artifact_relpath="artifacts/lens.pt",
        expect_file_sha256=None,
        expect_model_revision="f" * 40,
        expect_source_layers=[1, 3],
        expect_n_prompts=1,
    )
    config["benchmark"].update(manifest_path=str(manifest_path), split="dev")
    config["pursuit"].update(k=4, correlation_chunk_size=None)
    config["steering"].update(
        layers=[1, 3],
        strength_ratios=[1.0],
        mass_thresholds=[0.6],
        wrong_position_anchor=1,
    )
    config["decode"].update(max_new_tokens=3, ratios=[1.0])
    config["neutral_prompts"] = config["neutral_prompts"][:2]
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return {
        "config_path": str(config_path),
        "runs_root": str(runs_root),
        "config": config,
    }


def test_runner_end_to_end_on_mock(experiment, monkeypatch):
    runner = _import_runner()
    monkeypatch.setattr(runner, "load_gemma4", _mock_load_gemma4)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_generative_validation.py",
            "--config",
            experiment["config_path"],
            "--allow-model-load",
            "--runs-root",
            experiment["runs_root"],
        ],
    )
    runner.main()

    run_dirs = [
        p
        for p in Path(experiment["runs_root"]).iterdir()
        if p.name.startswith("generative_")
    ]
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    for name in (
        "run_started.json",
        "resolved_config.json",
        "summary.md",
        "run_metadata.json",
    ):
        assert (run_dir / name).is_file()
    artifacts = run_dir / "artifacts"
    for name in (
        "gates.json",
        "pursuits.json",
        "records.jsonl",
        "summary_by_condition.json",
        "calibration.json",
        "gonogo.json",
    ):
        assert (artifacts / name).is_file(), name

    gates = json.loads((artifacts / "gates.json").read_text(encoding="utf-8"))
    assert gates["zero_parity_ok"] is True
    assert gates["greedy_equivalence_ok"] is True

    records = [
        json.loads(line)
        for line in (artifacts / "records.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert records
    conditions = {r["vector_condition"] for r in records}
    assert {
        "none",
        "zero",
        "full_cone",
        "mass_subcone",
        "unrelated_cone",
        "random_matched_norm",
        "shuffled",
        "sign_reversed",
        "wrong_layer",
        "wrong_position",
        "raw_activation",
        "activation_diff",
        "manual_subcone",
    } <= conditions

    for record in records:
        assert record["schema"] == "jlens.generative.record.v1"
        assert record["total_logprob"] is not None
        assert record["provenance"]["active_token_ids"]
        assert record["provenance"]["active_coefficients"]

    # The no-hook baseline is exactly the model distribution: zero KL, and
    # the zero-vector hook reproduces it (delta over zero is 0 by defn).
    none_records = [r for r in records if r["vector_condition"] == "none"]
    assert all(
        abs(r["kl_divergence_from_baseline"]) < 1e-9 for r in none_records
    )
    zero_records = [r for r in records if r["vector_condition"] == "zero"]
    for zero, none in zip(zero_records, none_records, strict=True):
        assert zero["total_logprob"] == pytest.approx(
            none["total_logprob"], abs=1e-5
        )

    # Norm scaling: measured ratio equals the requested ratio at the default
    # anchor (the receiving-norm the vector was scaled against).
    for record in records:
        if record["requested_ratio"] and record["vector_condition"] not in (
            "wrong_position",
        ):
            assert record["measured_ratio"] == pytest.approx(
                record["requested_ratio"], rel=1e-4
            )

    # Site controls really moved the site.
    wrong_layer = [r for r in records if r["vector_condition"] == "wrong_layer"]
    assert wrong_layer
    assert all(r["injection_layer"] != r["source_layer"] for r in wrong_layer)
    wrong_position = [
        r for r in records if r["vector_condition"] == "wrong_position"
    ]
    assert wrong_position
    assert all(r["injection_anchor"] == 1 for r in wrong_position)

    # manual_subcone ran only where the manifest provided indices (layer 1
    # of mock-a) and recorded the subset.
    manual = [r for r in records if r["vector_condition"] == "manual_subcone"]
    assert manual
    assert all(r["example_id"] == "mock-a" for r in manual)
    assert all(r["source_layer"] == 1 for r in manual)
    assert all(r["vector_meta"]["subset_indices"] == [0] for r in manual)

    # mass_subcone recorded its threshold; activation_diff only ran for the
    # example with a control prompt.
    mass = [r for r in records if r["vector_condition"] == "mass_subcone"]
    assert all(r["vector_meta"]["mass_threshold"] == 0.6 for r in mass)
    diff = [r for r in records if r["vector_condition"] == "activation_diff"]
    assert diff and all(r["example_id"] == "mock-a" for r in diff)

    # Specificity diffs resolved against the matched unrelated-cone record.
    full = [r for r in records if r["vector_condition"] == "full_cone"]
    assert any(r["delta_logprob_vs_unrelated"] is not None for r in full)
    assert all(r["delta_logprob_vs_zero"] is not None for r in full)

    # Decodes happened for the configured conditions and recorded text.
    decoded = [r for r in records if r.get("generated_token_ids") is not None]
    assert decoded
    assert all(len(r["generated_token_ids"]) <= 3 for r in decoded)
    assert all(r["stop_reason"] in ("max_new_tokens", "eos") for r in decoded)

    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["mode"] == "generative_validation"
    assert metadata["gonogo"] is not None
    assert metadata["calibration"] is not None
    assert metadata["n_records"] == len(records)
