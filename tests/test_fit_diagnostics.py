# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Tests for jlens.fit_diagnostics and the fit(on_prompt=...) hook (CPU,
tiny model, no network)."""

import json
import statistics

import pytest
import torch

from jlens.fit_diagnostics import (
    DIAGNOSTIC_ROW_SCHEMA,
    PromptDiagnosticsRecorder,
)
from jlens.fitting import fit
from jlens.metadata import prompt_hashes

from .tiny import TinyDecoder

PROMPTS = [
    "the quick brown fox jumps over the lazy dog again and again",
    "a completely different sentence with its own token statistics here",
    "yet another prompt so the running average has something to average",
    "the final prompt in the tiny corpus closes out the fitting loop",
]


def run_fit(tmp_path, prompts, *, diag_name="diag.jsonl", ckpt=None, resume=True,
            expected_hashes=None):
    model = TinyDecoder(n_layers=4, d_model=8)
    recorder = PromptDiagnosticsRecorder(
        str(tmp_path / diag_name), expected_prompt_hashes=expected_hashes
    )
    lens = fit(
        model,
        prompts,
        source_layers=[2],
        target_layer=3,
        dim_batch=4,
        max_seq_len=64,
        skip_first=4,
        checkpoint_path=str(tmp_path / ckpt) if ckpt else None,
        checkpoint_every=2,
        resume=resume,
        on_prompt=recorder.on_prompt,
    )
    return lens, recorder


def test_recorder_records_every_prompt(tmp_path):
    lens, recorder = run_fit(tmp_path, PROMPTS)
    assert lens.n_prompts == len(PROMPTS)
    assert len(recorder.rows) == len(PROMPTS)
    hashes = prompt_hashes(PROMPTS)
    for index, row in enumerate(recorder.rows):
        assert row["schema"] == DIAGNOSTIC_ROW_SCHEMA
        assert row["prompt_index"] == index
        assert row["prompt_hash"] == hashes[index]
        assert row["status"] == "ok"
        entry = row["per_layer"]["2"]
        assert entry["frobenius_norm"] > 0
        assert entry["max_abs"] > 0
        assert entry["all_finite"] is True
        assert entry["running_mean_frobenius_norm"] > 0
        assert 0 < entry["contribution_fraction_of_sum_norm"] <= 1.0


def test_welford_matches_statistics(tmp_path):
    _, recorder = run_fit(tmp_path, PROMPTS)
    norms = [r["frobenius_norm_max_over_layers"] for r in recorder.rows]
    last = recorder.rows[-1]
    assert last["running_norm_mean"] == pytest.approx(statistics.fmean(norms))
    assert last["running_norm_variance"] == pytest.approx(
        statistics.variance(norms)
    )
    assert recorder.rows[0]["running_norm_variance"] is None


def test_alignment_and_relative_change_fields(tmp_path):
    _, recorder = run_fit(tmp_path, PROMPTS)
    first = recorder.rows[0]["per_layer"]["2"]
    assert first["alignment_cosine_with_running_mean"] is None
    assert first["running_mean_relative_change"] is None
    later = recorder.rows[-1]["per_layer"]["2"]
    assert -1.0 <= later["alignment_cosine_with_running_mean"] <= 1.0
    assert later["running_mean_relative_change"] > 0
    # Adding the n-th prompt moves the mean by at most ~1/n of the ratio.
    assert later["running_mean_relative_change"] < 5.0


def test_rows_persist_as_jsonl(tmp_path):
    _, recorder = run_fit(tmp_path, PROMPTS)
    lines = (tmp_path / "diag.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(PROMPTS)
    assert json.loads(lines[0])["prompt_index"] == 0


def test_skipped_prompt_recorded_with_reason(tmp_path):
    prompts = [PROMPTS[0], "abc", PROMPTS[1]]  # "abc" too short for skip_first=4
    lens, recorder = run_fit(tmp_path, prompts)
    assert lens.n_prompts == 2
    assert [r["status"] for r in recorder.rows] == ["ok", "skipped", "ok"]
    assert "too short" in recorder.rows[1]["reason"]
    assert recorder.rows[1]["prompt_hash"] == prompt_hashes(["abc"])[0]


def test_expected_hash_mismatch_raises(tmp_path):
    wrong = ["0" * 16] * len(PROMPTS)
    with pytest.raises(ValueError, match="does not match the reference"):
        run_fit(tmp_path, PROMPTS, expected_hashes=wrong)


def test_expected_hashes_accept_matching_corpus(tmp_path):
    _, recorder = run_fit(tmp_path, PROMPTS, expected_hashes=prompt_hashes(PROMPTS))
    assert len(recorder.rows) == len(PROMPTS)


def test_resume_continues_file_without_duplicates(tmp_path):
    model = TinyDecoder(n_layers=4, d_model=8)
    path = tmp_path / "diag.jsonl"
    ckpt = tmp_path / "ckpt.pt"
    recorder_a = PromptDiagnosticsRecorder(str(path))
    fit(model, PROMPTS[:2], source_layers=[2], target_layer=3, dim_batch=4,
        max_seq_len=64, skip_first=4, checkpoint_path=str(ckpt),
        checkpoint_every=1, on_prompt=recorder_a.on_prompt)
    assert len(recorder_a.rows) == 2

    # Resume over the full corpus: fit skips prompts 0-1 via its checkpoint,
    # and a fresh recorder on the same file continues from row 2.
    recorder_b = PromptDiagnosticsRecorder(str(path))
    assert len(recorder_b.rows) == 2
    fit(model, PROMPTS, source_layers=[2], target_layer=3, dim_batch=4,
        max_seq_len=64, skip_first=4, checkpoint_path=str(ckpt),
        checkpoint_every=1, resume=True, on_prompt=recorder_b.on_prompt)
    assert [r["prompt_index"] for r in recorder_b.rows] == [0, 1, 2, 3]
    norms = [r["frobenius_norm_max_over_layers"] for r in recorder_b.rows]
    assert recorder_b.rows[-1]["running_norm_mean"] == pytest.approx(
        statistics.fmean(norms)
    )


def test_non_advancing_index_raises(tmp_path):
    _, recorder = run_fit(tmp_path, PROMPTS)
    with pytest.raises(ValueError, match="does not advance"):
        recorder.on_prompt({
            "prompt_index": 0,
            "status": "skipped",
            "prompt": "x",
            "reason": "test",
            "elapsed_seconds": 0.0,
            "per_prompt_jacobians": None,
            "jacobian_sum": {},
            "n_done": 4,
            "checkpoint_path": None,
            "checkpoint_written": False,
        })


def test_reload_rejects_foreign_schema(tmp_path):
    path = tmp_path / "diag.jsonl"
    path.write_text('{"schema": "something.else", "prompt_index": 0}\n',
                    encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected row schema"):
        PromptDiagnosticsRecorder(str(path))


def synthetic_event(index, norm, *, n_done, running_sum):
    J = torch.zeros(4, 4)
    J[0, 0] = norm
    running_sum += J
    return {
        "prompt_index": index,
        "status": "ok",
        "prompt": f"prompt-{index}",
        "seq_len": 10,
        "n_valid_positions": 5,
        "elapsed_seconds": 1.0,
        "per_prompt_jacobians": {21: J},
        "jacobian_sum": {21: running_sum},
        "n_done": n_done,
        "checkpoint_path": None,
        "checkpoint_written": False,
    }, running_sum


def test_summary_dominance_and_stabilization(tmp_path):
    recorder = PromptDiagnosticsRecorder(str(tmp_path / "d.jsonl"))
    running = torch.zeros(4, 4)
    norms = [1.0, 1.0, 90.0, 1.0]  # one dominant prompt
    for index, norm in enumerate(norms):
        event, running = synthetic_event(
            index, norm, n_done=index + 1, running_sum=running.clone()
        )
        recorder.on_prompt(event)
    summary = recorder.summary(top_n=2, tail_window=2)
    assert summary["n_prompts_ok"] == 4
    assert summary["all_finite"] is True
    assert summary["top_prompts_by_norm"][0]["prompt_index"] == 2
    assert summary["top1_share_of_total_norm_mass"] == pytest.approx(90 / 93)
    assert summary["per_prompt_norm"]["max_over_median"] == pytest.approx(90.0)
    assert summary["running_mean_frobenius_norm"]["final"] == pytest.approx(93 / 4)
    recorder.write_summary(str(tmp_path / "summary.json"))
    assert json.load(open(tmp_path / "summary.json"))["n_prompts_ok"] == 4


def test_write_csv_flattens_rows(tmp_path):
    import csv

    _, recorder = run_fit(tmp_path, [PROMPTS[0], "abc", PROMPTS[1]])
    out = tmp_path / "diag.csv"
    recorder.write_csv(str(out))
    rows = list(csv.DictReader(open(out, encoding="utf-8")))
    assert len(rows) == 3  # one line per prompt x layer (single layer)
    assert rows[0]["layer"] == "2"
    assert rows[1]["status"] == "skipped"
    assert float(rows[0]["frobenius_norm"]) > 0


def test_summary_empty_recorder(tmp_path):
    recorder = PromptDiagnosticsRecorder(str(tmp_path / "d.jsonl"))
    summary = recorder.summary()
    assert summary["n_prompts_ok"] == 0
