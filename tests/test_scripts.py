# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""End-to-end pipeline test: fit_gemma.py -> apply_gemma.py on the
Gemma4-shaped mock (load_gemma4 monkeypatched; no network, no transformers).
This validates the exact code path the real Colab run will take."""

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from .mock_gemma4 import MockGemma4ForConditionalGeneration, MockTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent


def _import_script(name: str):
    spec = importlib.util.spec_from_file_location(
        name, REPO_ROOT / "scripts" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _mock_load_gemma4(*args, **kwargs):
    from jlens.gemma4 import Gemma4LensModel

    if not kwargs.get("allow_model_load"):
        raise RuntimeError("allow_model_load")
    model = Gemma4LensModel(
        MockGemma4ForConditionalGeneration(), MockTokenizer(auto_bos=False)
    )
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
def tmp_config(tmp_path):
    """The shipped microsmoke config, retargeted at the 6-layer/d8 mock."""
    with open(REPO_ROOT / "configs" / "gemma_text_microsmoke.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["model"].update(
        allow_model_load=True, expect_n_layers=6, expect_d_model=8,
        expect_vocab_size=32,
    )
    config["sites"].update(source_layers=[1, 3], target_layer=5)
    config["fitting"].update(
        prompts_path=str(REPO_ROOT / "configs/prompts/fit_prompts.json"),
        n_prompts=2, max_seq_len=48, dim_batch=4,
    )
    config["eval"]["prompts_path"] = str(REPO_ROOT / "configs/prompts/eval_prompts.json")
    config["paths"] = {
        "checkpoint": str(tmp_path / "ckpt.pt"),
        "output_dir": str(tmp_path / "out"),
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path, tmp_path


def test_fit_then_apply_pipeline(tmp_config, monkeypatch):
    config_path, tmp_path = tmp_config

    fit_gemma = _import_script("fit_gemma")
    monkeypatch.setattr(fit_gemma, "load_gemma4", _mock_load_gemma4)
    assert fit_gemma.main(["--config", str(config_path)]) == 0

    out = tmp_path / "out"
    assert (out / "lens.pt").exists()
    meta = json.loads((out / "fit_metadata.json").read_text(encoding="utf-8"))
    assert meta["n_prompts_fitted"] == 2
    assert meta["lens_reload_ok"] is True
    assert len(meta["prompt_hashes"]) == 2
    assert meta["architecture_report"]["d_model"] == 8
    assert meta["architecture_report"]["params_frozen"] is True
    assert meta["probe"]["all_finite"] is True
    assert meta["probe"]["jacobian_shapes"] == {"1": [8, 8], "3": [8, 8]} or meta[
        "probe"
    ]["jacobian_shapes"] == {1: [8, 8], 3: [8, 8]}
    assert meta["load_info"]["model_revision"] == "f" * 40
    assert meta["environment"]["upstream_commit"].startswith("581d398")

    apply_gemma = _import_script("apply_gemma")
    monkeypatch.setattr(apply_gemma, "load_gemma4", _mock_load_gemma4)
    assert (
        apply_gemma.main(
            ["--config", str(config_path), "--lens", str(out / "lens.pt"),
             "--allow-model-load"]
        )
        == 0
    )
    results = json.loads((out / "apply_results.json").read_text(encoding="utf-8"))
    assert results["layers"] == [1, 3]
    formats = {r["format"] for r in results["results"]}
    assert formats == {"plain", "chat"}  # evaluated separately, both present
    first = results["results"][0]
    layer_data = first["layers"][str(1)] if str(1) in first["layers"] else first["layers"][1]
    readout = layer_data["readouts"][-1]
    assert len(readout["jlens_topk"]) == 10
    entry = readout["jlens_topk"][0]
    assert {"token", "id", "logit_pre", "logit_capped"} <= set(entry)
    metrics = layer_data["metrics"]
    assert {"jlens", "logit_lens", "permuted", "random", "wrong_layer"} <= set(metrics)
    assert (out / "apply_summary.md").read_text(encoding="utf-8").startswith("# J-lens")


def test_fit_refuses_chat_fitting_corpus(tmp_config, monkeypatch, tmp_path):
    """Adjustment 2: the fitting corpus must be plain text; a non-plain_text
    prompts file is rejected before any model load."""
    config_path, _ = tmp_config
    bad = tmp_path / "bad_prompts.json"
    bad.write_text(
        json.dumps({"format": "chat", "prompts": ["hi"]}), encoding="utf-8"
    )
    fit_gemma = _import_script("fit_gemma")
    monkeypatch.setattr(fit_gemma, "load_gemma4", _mock_load_gemma4)

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["fitting"]["prompts_path"] = str(bad)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="plain_text"):
        fit_gemma.main(["--config", str(config_path)])
