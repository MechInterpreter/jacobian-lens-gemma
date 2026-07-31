# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""End-to-end CPU smoke test: the four stage scripts, in order, on the mock.

This is the exact code path the L4 pilot takes, minus the checkpoint — the
scripts' own ``main()`` functions, the shipped config, the real dataset builder,
the real training loops, the real beam search, and the real GO/NO-GO report.

It asserts the *plumbing*, not the science: the mock has no semantics, so the
verdict is expected to be NO-GO and the test says so explicitly rather than
lowering a threshold to manufacture a pass.
"""

import importlib.util
import json
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "jspace_language_autoencoder.yaml"


def _import_script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(script_name: str, argv: list[str]) -> int:
    """Invoke a stage script's own ``main()`` with the given command line."""
    import sys

    module = _import_script(script_name)
    saved = sys.argv
    sys.argv = [script_name, *argv]
    try:
        return module.main()
    finally:
        sys.argv = saved


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("jlang_e2e")


@pytest.fixture(scope="module")
def pipeline(run_dir):
    """Run the four stages once; every test below reads the artifacts."""
    base = ["--config", str(CONFIG_PATH), "--smoke", "--output-dir", str(run_dir)]
    return {
        "dataset": _run("build_jspace_language_dataset", [*base, "--benchmark"]),
        "reconstructor": _run("train_phrase_reconstructor", [*base, "--ignore-gate"]),
        "adapter": _run("train_cone_adapter", base),
        "evaluation": _run("evaluate_jspace_language", base),
    }


def _read(run_dir: Path, relative: str) -> dict:
    return json.loads((run_dir / relative).read_text(encoding="utf-8"))


def test_every_stage_completes(pipeline):
    assert pipeline["dataset"] == 0
    # The reconstructor gate is expected to fail on a semantics-free mock; the
    # run continues only because --ignore-gate was passed explicitly.
    assert pipeline["reconstructor"] == 0
    assert pipeline["adapter"] == 0
    # 4 = NO-GO, which is a result, not a crash.
    assert pipeline["evaluation"] in (0, 4)


def test_dataset_artifacts_exist_and_are_consistent(pipeline, run_dir):
    manifest = _read(run_dir, "dataset/manifest.json")
    assert manifest["n_records"] > 0
    assert manifest["n_phrases"] >= 3
    assert manifest["records_sha256"].startswith("sha256:")
    records = [
        json.loads(line)
        for line in (run_dir / "dataset" / "records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) == manifest["n_records"]
    assert {r["split"] for r in records} == {"train", "val", "heldout"}
    assert all(r["source_layer"] == 14 for r in records)
    assert all(2 <= r["n_phrase_tokens"] <= 6 for r in records)


def test_benchmark_was_measured_before_the_build(pipeline, run_dir):
    benchmark = _read(run_dir, "artifacts/benchmark.json")
    assert benchmark["measured"]["seconds_per_occurrence"] > 0
    assert benchmark["projection"]["estimated_storage_mb"] > 0


def test_leakage_report_is_clean(pipeline, run_dir):
    leakage = _read(run_dir, "artifacts/leakage.json")
    assert leakage["clean"] is True
    assert leakage["violations"] == []


def test_reconstructor_gate_is_recorded_with_its_verdict(pipeline, run_dir):
    gate = _read(run_dir, "artifacts/reconstructor_gate.json")
    assert gate["verdict"] in ("GO", "NO-GO")
    assert gate["ignored"] is True
    assert {c["name"] for c in gate["criteria"]} == {
        "auroc_correct_vs_distractor",
        "top5_retrieval",
    }


def test_reconstructor_is_frozen_in_its_checkpoint(pipeline, run_dir):
    payload = torch.load(run_dir / "reconstructor.pt", map_location="cpu", weights_only=False)
    assert payload["metadata"]["kind"] == "reconstructor"
    assert payload["metadata"]["extra"]["frozen_after_training"] is True
    # The checkpoint stores an already-frozen module, so its live counts read
    # 0 trainable; the count from *before* freezing is what proves it was
    # trained, and it is recorded separately.
    assert payload["metadata"]["trainable_parameters"] == 0
    assert payload["metadata"]["frozen_parameters"] > 0
    assert payload["metadata"]["extra"]["training_summary"]["trainable_parameters"] > 0


def test_adapter_records_which_reconstructor_it_was_trained_against(pipeline, run_dir):
    adapter = torch.load(run_dir / "adapter.pt", map_location="cpu", weights_only=False)
    reconstructor = torch.load(
        run_dir / "reconstructor.pt", map_location="cpu", weights_only=False
    )
    assert (
        adapter["metadata"]["extra"]["reconstructor_sha256"]
        == reconstructor["metadata"]["state_dict_sha256"]
    )


def test_resume_points_were_written_every_epoch(pipeline, run_dir):
    resume_points = sorted(run_dir.glob("adapter_epoch*.pt"))
    assert resume_points, "no resume checkpoint was written"
    payload = torch.load(resume_points[-1], map_location="cpu", weights_only=False)
    assert "optimizer_state" in payload
    assert "rng_state" in payload
    assert payload["metadata"]["epoch"] is not None


def test_a_resume_point_is_discoverable_and_loadable(pipeline, run_dir):
    """What makes a terminated Colab runtime cost one epoch instead of the whole
    stage: the newest valid resume point is found without being named."""
    from jlens.autoencoder.checkpoints import find_resume_checkpoint, load_checkpoint

    path = find_resume_checkpoint(str(run_dir), kind="adapter")
    assert path is not None
    payload = load_checkpoint(path, expect_kind="adapter")
    assert payload["metadata"]["kind"] == "adapter"
    assert int(payload["metadata"]["epoch"]) >= 0


def test_a_corrupt_resume_point_is_skipped_not_crashed_on(pipeline, run_dir, tmp_path):
    from jlens.autoencoder.checkpoints import find_resume_checkpoint

    (tmp_path / "adapter_epoch999.pt").write_bytes(b"not a checkpoint")
    assert find_resume_checkpoint(str(tmp_path), kind="adapter") is None


def test_evaluation_covers_every_baseline(pipeline, run_dir):
    from jlens.autoencoder.baselines import BASELINE_IDS

    evaluation = _read(run_dir, "artifacts/evaluation.json")
    assert set(evaluation["baselines"]) == set(BASELINE_IDS)
    assert evaluation["split"] == "heldout"
    assert evaluation["resources"]["model_forward_calls"] > 0


def test_gonogo_report_is_complete_and_honest(pipeline, run_dir):
    report = _read(run_dir, "artifacts/gonogo.json")
    assert report["verdict"] in ("GO", "NO-GO")
    assert [c["id"] for c in report["criteria"]] == [1, 2, 3, 4, 5, 6, 7]
    if report["verdict"] == "NO-GO":
        assert report["failed_criteria"]
        assert report["failure_attribution"]["primary"] in (
            "phrase_reconstructor",
            "cone_adapter",
            "decoding_interface",
            "cone_information_loss",
            "insufficient_data",
            "prompt_dependence",
        )
    assert report["artifact_identity"]["reconstructor_sha256"].startswith("sha256:")
    assert report["artifact_identity"]["adapter_sha256"].startswith("sha256:")


def test_summary_markdown_states_the_verdict(pipeline, run_dir):
    summary = (run_dir / "summary.md").read_text(encoding="utf-8")
    report = _read(run_dir, "artifacts/gonogo.json")
    assert f"**Verdict: {report['verdict']}**" in summary


def test_stage_metadata_distinguishes_configured_from_actual_model_loading(
    pipeline, run_dir
):
    for name in (
        "dataset_metadata.json",
        "reconstructor_metadata.json",
        "adapter_metadata.json",
        "evaluation_metadata.json",
    ):
        metadata = _read(run_dir, name)
        assert metadata["execution"]["model_loaded"] is False  # mock stack
        assert metadata["stack_provenance"]["mock"] is True
        assert metadata["config_fingerprint"].startswith("sha256:")


def test_verbalizer_prompt_is_identical_across_stages(pipeline, run_dir):
    """One constant instruction, byte-identical everywhere. If two stages
    disagreed on the prompt, every cross-stage number would be incomparable."""
    prompts = {
        name: _read(run_dir, name)["stack_provenance"]["verbalizer_prompt"]
        for name in (
            "dataset_metadata.json",
            "reconstructor_metadata.json",
            "adapter_metadata.json",
            "evaluation_metadata.json",
        )
    }
    rendered = {json.dumps(p["rendered_prompt"]) for p in prompts.values()}
    spans = {tuple(p["memory_span"]) for p in prompts.values()}
    assert len(rendered) == 1, prompts
    assert len(spans) == 1, prompts
    assert all(p["is_default_prompt"] for p in prompts.values())


def test_no_phrase_label_can_reach_the_adapter(pipeline, run_dir):
    """Structural leakage check on the artifact that was actually trained: the
    adapter takes only ``q``, and its stored tensors are exactly the layers of a
    freshly constructed adapter — no extra table that could hold phrase text."""
    import inspect

    from jlens.autoencoder.adapter import ConeAdapter
    from jlens.autoencoder.config import AdapterConfig

    assert list(inspect.signature(ConeAdapter.forward).parameters) == ["self", "q"]
    payload = torch.load(run_dir / "adapter.pt", map_location="cpu", weights_only=False)
    fresh = ConeAdapter(
        d_model=32,
        config=AdapterConfig(n_memory_tokens=2, hidden_dim=32),
        target_rms=1.0,
    )
    assert set(payload["state_dict"]) == set(fresh.state_dict())
