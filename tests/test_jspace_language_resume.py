# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Interruption and resume, proved rather than asserted.

Every claim this repository makes about resumability is tested here by actually
interrupting the thing and comparing the result to an uninterrupted run:

* dataset — resumed shards produce a byte-identical ``records.jsonl`` and
  bitwise-identical tensors;
* reconstructor / warm start / preference — resumed parameters hash to the same
  value as the uninterrupted run's, at several different interruption points;
* evaluation — the aggregated report is identical after shards are deleted and
  corrupted;
* configuration mismatch — refused, with nothing overwritten;
* corrupt checkpoints — skipped in favour of the previous valid one;
* completion markers — never written on an interrupted stage;
* idempotency — a completed stage does no expensive work when rerun.

Everything runs on the deterministic CPU mock. No Gemma, no downloads, no GPU.

Interruptions are simulated by an :class:`InterruptGuard` subclass whose
``should_stop`` flips after a chosen number of checks. That is exactly what a
real signal does — the handler only ever sets a flag — so these tests exercise
the same code path a SIGINT takes, without the flakiness of racing a signal
against a training loop.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

from jlens.autoencoder.adapter import ConeAdapter, train_adapter_warm_start
from jlens.autoencoder.checkpoints import (
    find_training_checkpoint,
    load_checkpoint,
    load_training_checkpoint,
    restore_training_state,
    save_training_checkpoint,
    state_dict_sha256,
)
from jlens.autoencoder.config import AutoencoderConfig
from jlens.autoencoder.dataset import JSpaceLanguageDataset, build_dataset, save_dataset
from jlens.autoencoder.dataset_shards import (
    build_dataset_sharded,
    shard_path,
    validate_shards,
)
from jlens.autoencoder.errors import AutoencoderError
from jlens.autoencoder.pipeline import build_stack, resolve_documents
from jlens.autoencoder.preference import BeamCache, train_preference
from jlens.autoencoder.reconstructor import (
    PhraseEmbedder,
    PhraseReconstructor,
    build_reconstructor_training,
    train_reconstructor,
)
from jlens.autoencoder.state import (
    InterruptGuard,
    StageInterrupted,
    StageState,
    check_compatible,
    stage_identity,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "jspace_language_autoencoder.yaml"

#: Small enough to run in seconds, large enough that every split is non-empty
#: and every loop runs several batches (so "interrupt mid-epoch" is meaningful).
SMOKE_CONFIG = {
    "dataset": {
        "mode": "smoke",
        "corpus": "mock",
        "n_phrases": 16,
        "occurrences_per_phrase": 2,
        "min_context_tokens": 3,
        "max_context_tokens": 48,
        "max_documents": 96,
        "min_document_chars": 40,
        "capture_batch_size": 4,
    },
    "reconstructor": {
        "hidden_dim": 32,
        "n_heads": 4,
        "n_layers": 1,
        "epochs": 4,
        "batch_size": 2,
        "n_distractors": 3,
        # Non-zero on purpose: dropout consumes the global RNG stream, so a
        # resume that failed to restore it would show up as differing weights.
        "dropout": 0.2,
    },
    "adapter": {
        "n_memory_tokens": 2,
        "hidden_dim": 32,
        "epochs": 3,
        "batch_size": 2,
        "beam_width": 3,
        "max_new_tokens": 6,
    },
    "preference": {
        "epochs": 2,
        "batch_size": 2,
        "n_unrelated_cones": 3,
        "max_pairs_per_example": 2,
    },
    "evaluation": {
        "beam_width": 3,
        "max_new_tokens": 6,
        "n_unrelated_cones": 3,
        "n_distractors": 3,
    },
}


class StopAfter(InterruptGuard):
    """A guard that requests a stop after ``n`` checks.

    Subclasses :class:`InterruptGuard` rather than mocking it so the loops under
    test use the real interface. ``should_stop`` is what a signal handler's flag
    drives, so flipping it here reproduces a SIGINT deterministically.
    """

    def __init__(self, n: int, stage: str = "test"):
        super().__init__(stage)
        self.n = int(n)
        self.checks = 0

    def __enter__(self):  # no signal handlers: pytest owns the main thread
        return self

    def __exit__(self, *exc):
        return False

    def should_stop(self) -> bool:
        self.checks += 1
        return self.checks > self.n


# ------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def config() -> AutoencoderConfig:
    return AutoencoderConfig.from_dict(dict(SMOKE_CONFIG))


@pytest.fixture(scope="module")
def stack(config):
    return build_stack(config, mock=True)


@pytest.fixture(scope="module")
def documents(config, stack):
    return resolve_documents(config, stack)


@pytest.fixture(scope="module")
def dataset(config, stack, documents, tmp_path_factory):
    """A dataset built the ordinary (non-incremental) way, for training tests."""
    directory = tmp_path_factory.mktemp("dataset")
    result = build_dataset(
        stack.model,
        stack.dictionary,
        documents,
        dataset_config=config.dataset,
        pursuit_config=config.pursuit,
        phrase_token_ids=stack.phrase_token_ids,
    )
    save_dataset(str(directory), result)
    loaded = JSpaceLanguageDataset.load(str(directory))
    loaded.assert_usable()
    return loaded


@pytest.fixture(scope="module")
def embedder(config, stack):
    return PhraseEmbedder(stack.model, max_phrase_tokens=config.reconstructor.max_phrase_tokens)


@pytest.fixture(scope="module")
def token_map(dataset, stack, config):
    phrases = list(
        dict.fromkeys(
            [r["phrase"] for r in dataset.records]
            + list(config.evaluation.confabulation_attractors)
        )
    )
    return stack.token_id_map(phrases)


@pytest.fixture(scope="module")
def target_map(dataset, stack, config):
    phrases = list(
        dict.fromkeys(
            [r["phrase"] for r in dataset.records]
            + list(config.evaluation.confabulation_attractors)
        )
    )
    return stack.target_id_map(phrases)


def identity_for(config, stage, run_dir):
    return stage_identity(config, stage=stage, run_dir=str(run_dir), run_id="test-run")


def checkpoint_writer(module, optimizer, generator, *, stage, kind, directory, identity, prefix):
    """The same checkpoint callback the scripts install, minus the state file."""

    def checkpoint(*, reason, epoch, batch_index, global_step, order, partial, metrics, history):
        path = str(Path(directory) / f"{prefix}{reason}_{epoch:03d}_{batch_index:05d}.pt")
        save_training_checkpoint(
            path,
            module,
            kind=kind,
            stage=stage,
            identity=identity,
            reason=reason,
            config={},
            epoch=epoch,
            batch_index=batch_index,
            global_step=global_step,
            optimizer=optimizer,
            sampler_order=order,
            generators={"sampler": generator},
            metrics=metrics,
            history=history,
            extra={"partial_epoch": partial},
        )
        return path

    return checkpoint


def restore_position(payload, module, optimizer, generator):
    """Turn a loaded checkpoint into the kwargs a resumed loop needs."""
    report = restore_training_state(
        payload, module, optimizer=optimizer, generators={"sampler": generator}
    )
    epoch_done = report["reason"] == "epoch_complete"
    return {
        "start_epoch": report["epoch"] + 1 if epoch_done else report["epoch"],
        "start_batch": 0 if epoch_done else report["batch_index"],
        "global_step": report["global_step"],
        "history": report["history"],
        "resume_order": None if epoch_done else report["sampler_order"],
        "resume_partial": (
            {} if epoch_done else (payload["metadata"].get("extra") or {}).get("partial_epoch") or {}
        ),
    }


# ------------------------------------------------- 1. dataset interruption


def _records_bytes(result) -> str:
    return "\n".join(json.dumps(r, sort_keys=True, ensure_ascii=False) for r in result.records)


def test_dataset_resume_skips_completed_records_and_matches_clean_build(
    config, stack, documents, tmp_path
):
    """Interrupt after a few shards; the resumed dataset equals a clean build."""
    clean = build_dataset(
        stack.model,
        stack.dictionary,
        documents,
        dataset_config=config.dataset,
        pursuit_config=config.pursuit,
        phrase_token_ids=stack.phrase_token_ids,
        provenance={"mock": True},
    )
    shard_dir = tmp_path / "shards"
    kwargs = dict(
        dataset_config=config.dataset,
        pursuit_config=config.pursuit,
        phrase_token_ids=stack.phrase_token_ids,
        provenance={"mock": True},
        shard_dir=str(shard_dir),
    )
    with pytest.raises(StageInterrupted):
        build_dataset_sharded(
            stack.model, stack.dictionary, documents, guard=StopAfter(2, "dataset"), **kwargs
        )
    partial = validate_shards(str(shard_dir))
    assert len(partial["valid"]) == 2, "completed shards must survive the interruption"
    assert not partial["invalid"], "no half-written shard may be left behind"

    resumed = build_dataset_sharded(stack.model, stack.dictionary, documents, **kwargs)
    assert resumed.stats["sharded_build"]["n_reused_shards"] == 2, "completed work was redone"
    assert _records_bytes(resumed) == _records_bytes(clean)
    assert torch.equal(resumed.activations, clean.activations)
    assert torch.equal(resumed.cones, clean.cones)
    assert resumed.phrase_token_ids == clean.phrase_token_ids


def test_dataset_rebuild_is_idempotent_and_byte_stable(config, stack, documents, tmp_path):
    """Rerunning a finished build recomputes nothing and produces the same bytes."""
    shard_dir = tmp_path / "shards"
    kwargs = dict(
        dataset_config=config.dataset,
        pursuit_config=config.pursuit,
        phrase_token_ids=stack.phrase_token_ids,
        shard_dir=str(shard_dir),
    )
    first = build_dataset_sharded(stack.model, stack.dictionary, documents, **kwargs)
    second = build_dataset_sharded(stack.model, stack.dictionary, documents, **kwargs)
    n_shards = first.stats["sharded_build"]["n_shards"]
    assert second.stats["sharded_build"]["n_reused_shards"] == n_shards
    assert _records_bytes(second) == _records_bytes(first)
    assert torch.equal(second.cones, first.cones)


def test_partial_shard_without_sidecar_is_never_treated_as_complete(
    config, stack, documents, tmp_path
):
    """A ``.pt`` whose sidecar never landed is a write that did not finish."""
    shard_dir = tmp_path / "shards"
    kwargs = dict(
        dataset_config=config.dataset,
        pursuit_config=config.pursuit,
        phrase_token_ids=stack.phrase_token_ids,
        shard_dir=str(shard_dir),
    )
    clean = build_dataset_sharded(stack.model, stack.dictionary, documents, **kwargs)
    sidecar = Path(shard_path(str(shard_dir), 1)).with_suffix(".json")
    sidecar.unlink()
    assert 1 not in validate_shards(str(shard_dir))["valid"]
    rebuilt = build_dataset_sharded(stack.model, stack.dictionary, documents, **kwargs)
    assert _records_bytes(rebuilt) == _records_bytes(clean)
    assert torch.equal(rebuilt.cones, clean.cones)


def test_corrupt_shard_is_ignored_and_recomputed(config, stack, documents, tmp_path):
    shard_dir = tmp_path / "shards"
    kwargs = dict(
        dataset_config=config.dataset,
        pursuit_config=config.pursuit,
        phrase_token_ids=stack.phrase_token_ids,
        shard_dir=str(shard_dir),
    )
    clean = build_dataset_sharded(stack.model, stack.dictionary, documents, **kwargs)
    Path(shard_path(str(shard_dir), 0)).write_bytes(b"not a torch file")
    found = validate_shards(str(shard_dir))
    assert 0 not in found["valid"]
    assert any("shard_000000" in entry["path"] for entry in found["invalid"])
    rebuilt = build_dataset_sharded(stack.model, stack.dictionary, documents, **kwargs)
    assert _records_bytes(rebuilt) == _records_bytes(clean)


# ------------------------------------------- 2. reconstructor interruption


def _train_reconstructor_clean(dataset, embedder, config, stack, token_map):
    model, optimizer, generator = build_reconstructor_training(dataset, config=config.reconstructor)
    return train_reconstructor(
        dataset,
        embedder,
        config=config.reconstructor,
        source_layer=stack.source_layer,
        phrase_token_ids=token_map,
        model=model,
        optimizer=optimizer,
        generator=generator,
    )


@pytest.mark.parametrize("stop_after", [3, 7, 11, 14])
def test_reconstructor_resume_matches_uninterrupted_training(
    dataset, embedder, config, stack, token_map, tmp_path, stop_after
):
    """Interrupt mid-epoch at four different points; the result never moves.

    Parameters *and* the per-epoch history must match: identical weights with a
    history whose interrupted epoch averages over the tail only would be a
    resume that silently changed what the run reports.
    """
    clean, clean_summary = _train_reconstructor_clean(
        dataset, embedder, config, stack, token_map
    )
    clean_sha = state_dict_sha256(clean)

    directory = tmp_path / f"ckpt{stop_after}"
    directory.mkdir()
    identity = identity_for(config, "reconstructor", tmp_path)
    model, optimizer, generator = build_reconstructor_training(dataset, config=config.reconstructor)
    with pytest.raises(StageInterrupted) as excinfo:
        train_reconstructor(
            dataset,
            embedder,
            config=config.reconstructor,
            source_layer=stack.source_layer,
            phrase_token_ids=token_map,
            model=model,
            optimizer=optimizer,
            generator=generator,
            checkpoint=checkpoint_writer(
                model,
                optimizer,
                generator,
                stage="reconstructor",
                kind="reconstructor",
                directory=directory,
                identity=identity,
                prefix="reconstructor_",
            ),
            guard=StopAfter(stop_after, "reconstructor"),
        )
    path = excinfo.value.checkpoint_path
    assert path and Path(path).is_file(), "an interruption must leave a checkpoint"
    assert "keyboard_interrupt" in Path(path).name, "interrupted checkpoints need distinct names"

    resumed_model, resumed_optimizer, resumed_generator = build_reconstructor_training(
        dataset, config=config.reconstructor, seed_global=False
    )
    payload = load_training_checkpoint(
        path, expect_kind="reconstructor", expect_stage="reconstructor"
    )
    position = restore_position(payload, resumed_model, resumed_optimizer, resumed_generator)
    resumed, resumed_summary = train_reconstructor(
        dataset,
        embedder,
        config=config.reconstructor,
        source_layer=stack.source_layer,
        phrase_token_ids=token_map,
        model=resumed_model,
        optimizer=resumed_optimizer,
        generator=resumed_generator,
        **position,
    )
    assert state_dict_sha256(resumed) == clean_sha
    assert resumed_summary["global_step"] == clean_summary["global_step"]
    assert [h["loss"] for h in resumed_summary["history"]] == pytest.approx(
        [h["loss"] for h in clean_summary["history"]], abs=0.0
    )


def test_reconstructor_checkpoint_restores_optimizer_and_rng(
    dataset, embedder, config, stack, token_map, tmp_path
):
    """The restore reports what it actually restored, and it is not nothing."""
    directory = tmp_path / "ckpt"
    directory.mkdir()
    identity = identity_for(config, "reconstructor", tmp_path)
    model, optimizer, generator = build_reconstructor_training(dataset, config=config.reconstructor)
    with pytest.raises(StageInterrupted):
        train_reconstructor(
            dataset,
            embedder,
            config=config.reconstructor,
            source_layer=stack.source_layer,
            phrase_token_ids=token_map,
            model=model,
            optimizer=optimizer,
            generator=generator,
            checkpoint=checkpoint_writer(
                model, optimizer, generator,
                stage="reconstructor", kind="reconstructor",
                directory=directory, identity=identity, prefix="reconstructor_",
            ),
            guard=StopAfter(5, "reconstructor"),
        )
    path, _ = find_training_checkpoint(
        str(directory), kind="reconstructor", stage="reconstructor", prefix="reconstructor_"
    )
    payload = load_training_checkpoint(path)
    fresh, fresh_optimizer, fresh_generator = build_reconstructor_training(
        dataset, config=config.reconstructor, seed_global=False
    )
    report = restore_training_state(
        payload, fresh, optimizer=fresh_optimizer, generators={"sampler": fresh_generator}
    )
    assert report["optimizer_restored"] is True
    assert "sampler" in report["generators_restored"]
    assert report["rng_restored"]["torch_cpu"] is True
    assert report["rng_restored"]["python"] is True
    assert report["sampler_order"], "the epoch's shuffle must be stored, not re-derived"
    assert fresh_optimizer.state_dict()["state"], "optimizer moments must come back"


# ------------------------------------------------- 3. adapter warm start


def new_adapter_and_optimizer(stack, dataset, config):
    """A freshly initialized adapter at a *fixed* RNG position, plus its optimizer.

    The seed is the point. ``ConeAdapter`` draws its initial weights from the
    global stream, so two adapters built at different points in a test session
    start from different weights and no amount of correct resuming would make
    their trajectories agree. Seeding here isolates the thing under test — the
    resume — from where in the file the comparison run happened to be built.
    """
    torch.manual_seed(int(config.adapter.seed))
    adapter = ConeAdapter(
        d_model=dataset.d_model,
        config=config.adapter,
        target_rms=stack.memory_scale["embedding_rms"],
    )
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=float(config.adapter.learning_rate),
        weight_decay=float(config.adapter.weight_decay),
    )
    generator = torch.Generator().manual_seed(int(config.adapter.seed))
    return adapter, optimizer, generator


def _warm_start_clean(stack, dataset, config, target_map):
    adapter, optimizer, generator = new_adapter_and_optimizer(stack, dataset, config)
    return train_adapter_warm_start(
        stack.model,
        adapter,
        dataset,
        stack.prompt,
        config=config.adapter,
        conditioner=stack.conditioner,
        phrase_targets=target_map,
        pad_token_id=stack.pad_token_id,
        optimizer=optimizer,
        generator=generator,
    )


@pytest.mark.parametrize("stop_after", [2, 5, 9])
def test_adapter_warm_start_resume_is_deterministic(
    stack, dataset, config, target_map, tmp_path, stop_after
):
    """Warm start resumes to the same weights an uninterrupted run reaches."""
    clean, clean_summary = _warm_start_clean(stack, dataset, config, target_map)
    clean_sha = state_dict_sha256(clean)

    directory = tmp_path / f"warm{stop_after}"
    directory.mkdir()
    identity = identity_for(config, "adapter_warm", tmp_path)
    adapter, optimizer, generator = new_adapter_and_optimizer(stack, dataset, config)
    common = dict(
        config=config.adapter,
        conditioner=stack.conditioner,
        phrase_targets=target_map,
        pad_token_id=stack.pad_token_id,
    )
    with pytest.raises(StageInterrupted) as excinfo:
        train_adapter_warm_start(
            stack.model,
            adapter,
            dataset,
            stack.prompt,
            optimizer=optimizer,
            generator=generator,
            checkpoint=checkpoint_writer(
                adapter, optimizer, generator,
                stage="adapter_warm", kind="adapter",
                directory=directory, identity=identity, prefix="adapter_warm_",
            ),
            guard=StopAfter(stop_after, "adapter_warm"),
            **common,
        )
    payload = load_training_checkpoint(
        excinfo.value.checkpoint_path, expect_kind="adapter", expect_stage="adapter_warm"
    )
    resumed_adapter, resumed_optimizer, resumed_generator = new_adapter_and_optimizer(
        stack, dataset, config
    )
    position = restore_position(payload, resumed_adapter, resumed_optimizer, resumed_generator)
    resumed, resumed_summary = train_adapter_warm_start(
        stack.model,
        resumed_adapter,
        dataset,
        stack.prompt,
        optimizer=resumed_optimizer,
        generator=resumed_generator,
        **position,
        **common,
    )
    assert state_dict_sha256(resumed) == clean_sha
    assert [h["loss"] for h in resumed_summary["history"]] == pytest.approx(
        [h["loss"] for h in clean_summary["history"]], abs=0.0
    )


def test_legacy_adapter_epoch_checkpoints_are_still_written(tmp_path):
    """The pre-existing ``adapter_epoch*.pt`` location must keep working."""
    run_dir = tmp_path / "run"
    base = ["--config", str(CONFIG_PATH), "--smoke", "--output-dir", str(run_dir)]
    assert _run("build_jspace_language_dataset", base) == 0
    assert _run("train_phrase_reconstructor", [*base, "--ignore-gate"]) == 0
    assert _run("train_cone_adapter", [*base, "--skip-preference"]) == 0
    legacy = sorted(run_dir.glob("adapter_epoch*.pt"))
    assert legacy, "adapter_epoch*.pt is the historical resume point and must remain"
    payload = load_checkpoint(str(legacy[-1]), expect_kind="adapter")
    assert payload["metadata"]["extra"]["stage"] == "warm_start"
    assert "optimizer_state" in payload


# ------------------------------------------------ 4. preference training


def _preference_inputs(stack, dataset, config, token_map, embedder):
    """A frozen reconstructor and a warm-started adapter, built once."""
    reconstructor = PhraseReconstructor(d_model=dataset.d_model, config=config.reconstructor)
    reconstructor.eval()
    for parameter in reconstructor.parameters():
        parameter.requires_grad_(False)
    warm = ConeAdapter(
        d_model=dataset.d_model,
        config=config.adapter,
        target_rms=stack.memory_scale["embedding_rms"],
    )
    return reconstructor, warm


def _train_preference(stack, dataset, config, embedder, reconstructor, adapter, **extra):
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=float(config.preference.learning_rate)
    )
    generator = torch.Generator().manual_seed(int(config.preference.seed))
    extra.setdefault("optimizer", optimizer)
    extra.setdefault("generator", generator)
    return (
        train_preference(
            stack.model,
            adapter,
            reconstructor,
            embedder,
            dataset,
            stack.prompt,
            config=config.preference,
            conditioner=stack.conditioner,
            stop_token_ids=stack.stop_token_ids,
            pad_token_id=stack.pad_token_id,
            beam_width=config.adapter.beam_width,
            max_new_tokens=config.adapter.max_new_tokens,
            source_layer=stack.source_layer,
            accept_recon_min=config.evaluation.accept_recon_min,
            accept_margin_min=config.evaluation.accept_margin_min,
            **extra,
        ),
        extra["optimizer"],
        extra["generator"],
    )


def test_preference_resume_matches_uninterrupted_training(
    stack, dataset, config, embedder, token_map, tmp_path
):
    """Interrupt preference training mid-epoch; the resumed adapter matches."""
    reconstructor, warm = _preference_inputs(stack, dataset, config, token_map, embedder)
    reference_state = {k: v.clone() for k, v in warm.state_dict().items()}

    import copy

    (clean, clean_summary), _, _ = _train_preference(
        stack, dataset, config, embedder, reconstructor, copy.deepcopy(warm),
        reference_state=reference_state,
    )
    clean_sha = state_dict_sha256(clean)

    directory = tmp_path / "pref"
    directory.mkdir()
    identity = identity_for(config, "adapter_preference", tmp_path)
    adapter = copy.deepcopy(warm)
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=float(config.preference.learning_rate)
    )
    generator = torch.Generator().manual_seed(int(config.preference.seed))
    with pytest.raises(StageInterrupted) as excinfo:
        _train_preference(
            stack, dataset, config, embedder, reconstructor, adapter,
            optimizer=optimizer,
            generator=generator,
            reference_state=reference_state,
            checkpoint=checkpoint_writer(
                adapter, optimizer, generator,
                stage="adapter_preference", kind="adapter",
                directory=directory, identity=identity, prefix="adapter_preference_",
            ),
            guard=StopAfter(3, "adapter_preference"),
        )
    payload = load_training_checkpoint(
        excinfo.value.checkpoint_path,
        expect_kind="adapter",
        expect_stage="adapter_preference",
    )
    resumed_adapter = copy.deepcopy(warm)
    resumed_optimizer = torch.optim.AdamW(
        resumed_adapter.parameters(), lr=float(config.preference.learning_rate)
    )
    resumed_generator = torch.Generator().manual_seed(int(config.preference.seed))
    position = restore_position(
        payload, resumed_adapter, resumed_optimizer, resumed_generator
    )
    (resumed, resumed_summary), _, _ = _train_preference(
        stack, dataset, config, embedder, reconstructor, resumed_adapter,
        optimizer=resumed_optimizer,
        generator=resumed_generator,
        reference_state=reference_state,
        **position,
    )
    assert state_dict_sha256(resumed) == clean_sha
    assert [h["loss"] for h in resumed_summary["history"]] == pytest.approx(
        [h["loss"] for h in clean_summary["history"]], abs=0.0
    )


def test_preference_discovery_never_picks_up_a_warm_start_checkpoint(
    stack, dataset, config, tmp_path
):
    """The two stages' checkpoints must not be confusable, by name or by content."""
    directory = tmp_path / "checkpoints"
    directory.mkdir()
    adapter = ConeAdapter(
        d_model=dataset.d_model,
        config=config.adapter,
        target_rms=stack.memory_scale["embedding_rms"],
    )
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(1)
    for stage, prefix in (("adapter_warm", "adapter_warm_"), ):
        save_training_checkpoint(
            str(directory / f"{prefix}epoch000.pt"),
            adapter,
            kind="adapter",
            stage=stage,
            identity=identity_for(config, stage, tmp_path),
            reason="epoch_complete",
            config={},
            epoch=0,
            batch_index=0,
            global_step=1,
            optimizer=optimizer,
            generators={"sampler": generator},
        )
    # Nothing for the preference stage exists yet.
    found, _ = find_training_checkpoint(
        str(directory), kind="adapter", stage="adapter_preference",
        prefix="adapter_preference_",
    )
    assert found is None, "warm-start state must not be discoverable as preference state"

    # Even pointed straight at it, loading refuses on the recorded stage.
    with pytest.raises(AutoencoderError, match="expected stage 'adapter_preference'"):
        load_training_checkpoint(
            str(directory / "adapter_warm_epoch000.pt"),
            expect_kind="adapter",
            expect_stage="adapter_preference",
        )


def test_preference_stage_preserves_the_warm_start_artifact(tmp_path):
    """Preference training must not overwrite ``adapter_warm.pt``."""
    run_dir = tmp_path / "run"
    base = ["--config", str(CONFIG_PATH), "--smoke", "--output-dir", str(run_dir)]
    assert _run("build_jspace_language_dataset", base) == 0
    assert _run("train_phrase_reconstructor", [*base, "--ignore-gate"]) == 0
    assert _run("train_cone_adapter", [*base, "--skip-preference"]) == 0
    warm_sha = load_checkpoint(str(run_dir / "adapter_warm.pt"), expect_kind="adapter")[
        "metadata"
    ]["state_dict_sha256"]
    assert _run("train_cone_adapter", base) == 0
    after = load_checkpoint(str(run_dir / "adapter_warm.pt"), expect_kind="adapter")["metadata"][
        "state_dict_sha256"
    ]
    assert after == warm_sha, "the warm-start artifact was overwritten by preference training"
    final = load_checkpoint(str(run_dir / "adapter.pt"), expect_kind="adapter")["metadata"]
    assert final["extra"]["adapter_warm_sha256"] == warm_sha
    assert (run_dir / "checkpoints" / "adapter_preference_epoch000.pt").is_file()


def test_beam_cache_reuses_generations_for_an_unchanged_adapter(tmp_path):
    """A cache hit requires the same adapter weights, not merely the same q."""
    from jlens.autoencoder.preference import beam_cache_key
    from jlens.autoencoder.verbalizer import Candidate

    cache = BeamCache(str(tmp_path / "beams"), capacity=8)
    q = torch.arange(6, dtype=torch.float32)
    candidates = [
        Candidate(token_ids=(7, 8), text="a b", logprob=-1.0, mean_logprob=-0.5,
                  n_tokens=2, finished=True, beam_rank=0)
    ]
    key = dict(q=q, prompt_id="verbalizer-default", beam_width=3,
               max_new_tokens=6, stop_token_ids=[5])
    first = beam_cache_key(adapter_sha256="sha256:aaa", **key)
    cache.put(first, candidates)
    assert [c.to_dict() for c in cache.get(first)] == [c.to_dict() for c in candidates]
    moved = beam_cache_key(adapter_sha256="sha256:bbb", **key)
    assert cache.get(moved) is None, "a stepped adapter must miss the cache"


# ------------------------------------------------------- 5. evaluation


def test_evaluation_reuses_shards_and_aggregates_identically(tmp_path):
    """Delete and corrupt shards, resume, and get the same report back."""
    run_dir = tmp_path / "run"
    base = ["--config", str(CONFIG_PATH), "--smoke", "--output-dir", str(run_dir)]
    assert _run("build_jspace_language_dataset", base) == 0
    assert _run("train_phrase_reconstructor", [*base, "--ignore-gate"]) == 0
    assert _run("train_cone_adapter", base) == 0
    assert _run("evaluate_jspace_language", base) in (0, 4)

    full = json.loads((run_dir / "artifacts" / "evaluation.json").read_text(encoding="utf-8"))
    full_report = json.loads((run_dir / "artifacts" / "gonogo.json").read_text(encoding="utf-8"))
    full_summary = (run_dir / "summary.md").read_text(encoding="utf-8")

    shards = run_dir / "evaluation_shards"
    removed = 0
    for kind in ("baseline", "robustness"):
        for index, path in enumerate(sorted((shards / kind).glob("*.json"))):
            if index % 3 == 0:
                path.unlink()
                removed += 1
    assert removed, "the fixture must actually remove some shards"
    victim = sorted((shards / "baseline").glob("*.json"))[0]
    victim.write_text("{ truncated", encoding="utf-8")

    # Exactly what a real interruption leaves: no marker, status interrupted.
    state = StageState(str(run_dir), "evaluation")
    Path(state.complete_path).unlink()
    state.write(status="interrupted")

    assert _run("evaluate_jspace_language", base) in (0, 4)
    again = json.loads((run_dir / "artifacts" / "evaluation.json").read_text(encoding="utf-8"))
    again_report = json.loads((run_dir / "artifacts" / "gonogo.json").read_text(encoding="utf-8"))

    assert again["sharding"]["reused_shards"] > 0, "valid shards must be reused"
    assert again["sharding"]["computed_shards"] > 0, "missing shards must be recomputed"
    assert again["per_record"] == full["per_record"]
    assert again["baselines"] == full["baselines"]
    assert again["cross_cone_swap"] == full["cross_cone_swap"]
    # ``resources`` is the measured cost of what actually ran and is excluded on
    # purpose; every result in the report is compared above and below.
    assert {k: v for k, v in again_report.items() if k != "resources"} == {
        k: v for k, v in full_report.items() if k != "resources"
    }
    assert (run_dir / "summary.md").read_text(encoding="utf-8") == full_summary


def test_evaluation_is_idempotent_and_does_no_work_when_complete(tmp_path):
    run_dir = tmp_path / "run"
    base = ["--config", str(CONFIG_PATH), "--smoke", "--output-dir", str(run_dir)]
    for script, extra in [
        ("build_jspace_language_dataset", []),
        ("train_phrase_reconstructor", ["--ignore-gate"]),
        ("train_cone_adapter", []),
    ]:
        assert _run(script, [*base, *extra]) == 0
    assert _run("evaluate_jspace_language", base) in (0, 4)
    before = (run_dir / "artifacts" / "evaluation.json").read_bytes()
    before_summary = (run_dir / "summary.md").read_bytes()
    metadata = json.loads(
        (run_dir / "evaluation_metadata.json").read_text(encoding="utf-8")
    )

    assert _run("evaluate_jspace_language", base) in (0, 4)
    assert (run_dir / "artifacts" / "evaluation.json").read_bytes() == before
    assert (run_dir / "summary.md").read_bytes() == before_summary
    # A skipped stage does not rewrite its metadata, which is how "no expensive
    # work happened" is observable from outside.
    assert json.loads(
        (run_dir / "evaluation_metadata.json").read_text(encoding="utf-8")
    )["written_utc"] == metadata["written_utc"]


def test_completed_stages_are_skipped_on_rerun(tmp_path):
    """Rerunning the whole pipeline recomputes nothing and changes no artifact."""
    run_dir = tmp_path / "run"
    base = ["--config", str(CONFIG_PATH), "--smoke", "--output-dir", str(run_dir)]
    stages = [
        ("build_jspace_language_dataset", []),
        ("train_phrase_reconstructor", ["--ignore-gate"]),
        ("train_cone_adapter", []),
    ]
    for script, extra in stages:
        assert _run(script, [*base, *extra]) == 0
    fingerprints = {
        name: (run_dir / name).read_bytes()
        for name in ("reconstructor.pt", "adapter.pt", "adapter_warm.pt")
    }
    records = (run_dir / "dataset" / "records.jsonl").read_bytes()
    for script, extra in stages:
        assert _run(script, [*base, *extra]) == 0
    for name, blob in fingerprints.items():
        assert (run_dir / name).read_bytes() == blob, f"{name} was rewritten on a skipped rerun"
    assert (run_dir / "dataset" / "records.jsonl").read_bytes() == records
    for stage in ("dataset", "reconstructor", "adapter_warm", "adapter_preference"):
        assert StageState(str(run_dir), stage).status == "complete"


# ---------------------------------------------- 6. configuration mismatch


def test_semantic_config_change_is_refused_and_cannot_be_overridden(config, tmp_path):
    """Changing the layer, revision, or lens is never a resumable change."""
    stored = identity_for(config, "reconstructor", tmp_path).to_dict()
    for field, payload in [
        ("source_layer", {"dataset": {**SMOKE_CONFIG["dataset"], "source_layer": 21}}),
        ("model_revision", {"model": {"revision": "deadbeef"}}),
        ("lens_sha256", {"lens": {"expect_file_sha256": "sha256:" + "0" * 64}}),
        ("pursuit", {"pursuit": {"k": 5}}),
        ("split_policy", {"dataset": {**SMOKE_CONFIG["dataset"], "split_salt": "other"}}),
        ("architecture", {"reconstructor": {**SMOKE_CONFIG["reconstructor"], "hidden_dim": 64}}),
    ]:
        merged = {**SMOKE_CONFIG, **payload}
        if field == "source_layer":
            merged["lens"] = {"expect_source_layers": [21]}
        changed = AutoencoderConfig.from_dict(merged)
        current = identity_for(changed, "reconstructor", tmp_path)
        strict = check_compatible(stored, current)
        assert not strict.compatible, f"{field} must not be resumable"
        # The override exists for cadence and epoch counts, never for these.
        waived = check_compatible(stored, current, allow_nonsemantic_drift=True)
        assert not waived.compatible, f"--allow-config-drift must not waive {field}"
        assert any(field in m for m in waived.semantic_mismatches), waived.semantic_mismatches


def test_nonsemantic_change_is_refused_by_default_and_waivable(config, tmp_path):
    stored = identity_for(config, "reconstructor", tmp_path).to_dict()
    changed = AutoencoderConfig.from_dict(
        {**SMOKE_CONFIG, "reconstructor": {**SMOKE_CONFIG["reconstructor"], "epochs": 9}}
    )
    current = identity_for(changed, "reconstructor", tmp_path)
    assert not check_compatible(stored, current).compatible
    waived = check_compatible(stored, current, allow_nonsemantic_drift=True)
    assert waived.compatible and waived.overridden
    assert not waived.semantic_mismatches


def test_incompatible_state_is_refused_without_overwriting_artifacts(tmp_path):
    """A changed layer stops the stage and leaves every artifact untouched."""
    run_dir = tmp_path / "run"
    base = ["--config", str(CONFIG_PATH), "--smoke", "--output-dir", str(run_dir)]
    assert _run("build_jspace_language_dataset", base) == 0
    records = (run_dir / "dataset" / "records.jsonl").read_bytes()
    tensors = (run_dir / "dataset" / "tensors.pt").read_bytes()

    # Rewrite the stored identity as if the run had been built at another layer.
    state = StageState(str(run_dir), "dataset")
    stored = state.read()
    stored["identity"]["source_layer"] = 21
    stored["identity"]["dataset_identity"]["source_layer"] = 21
    from jlens.autoencoder.state import atomic_write_json

    atomic_write_json(state.state_path, stored)
    Path(state.complete_path).unlink()
    state.write(status="interrupted", detail={"note": "identity rewritten by the test"})

    with pytest.raises(AutoencoderError, match="incompatible|refusing"):
        _run("build_jspace_language_dataset", base)
    assert (run_dir / "dataset" / "records.jsonl").read_bytes() == records
    assert (run_dir / "dataset" / "tensors.pt").read_bytes() == tensors


# ----------------------------------------- 7. corrupt / partial checkpoints


def test_corrupt_latest_checkpoint_falls_back_to_the_previous_valid_one(
    dataset, config, stack, tmp_path
):
    directory = tmp_path / "checkpoints"
    directory.mkdir()
    identity = identity_for(config, "reconstructor", tmp_path)
    model = PhraseReconstructor(d_model=dataset.d_model, config=config.reconstructor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for epoch in (0, 1):
        save_training_checkpoint(
            str(directory / f"reconstructor_epoch{epoch:03d}.pt"),
            model,
            kind="reconstructor",
            stage="reconstructor",
            identity=identity,
            reason="epoch_complete",
            config={},
            epoch=epoch,
            batch_index=0,
            global_step=epoch + 1,
            optimizer=optimizer,
        )
    (directory / "reconstructor_epoch001.pt").write_bytes(b"truncated")
    path, rejected = find_training_checkpoint(
        str(directory), kind="reconstructor", stage="reconstructor", prefix="reconstructor_"
    )
    assert path is not None and path.endswith("epoch000.pt")
    assert len(rejected) == 1 and "epoch001" in rejected[0]["path"]
    assert rejected[0]["reason"], "a rejected candidate must say why"


def test_checkpoint_with_tampered_weights_is_rejected_on_load(dataset, config, tmp_path):
    """Validation is by recomputed checksum, not by 'it deserialized'."""
    path = tmp_path / "reconstructor_epoch000.pt"
    model = PhraseReconstructor(d_model=dataset.d_model, config=config.reconstructor)
    save_training_checkpoint(
        str(path),
        model,
        kind="reconstructor",
        stage="reconstructor",
        identity=identity_for(config, "reconstructor", tmp_path),
        reason="epoch_complete",
        config={},
        epoch=0,
        batch_index=0,
        global_step=1,
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    key = sorted(payload["state_dict"])[0]
    payload["state_dict"][key] = payload["state_dict"][key] + 1.0
    torch.save(payload, path)
    with pytest.raises(AutoencoderError, match="checksum"):
        load_training_checkpoint(str(path))


def test_in_flight_temporary_files_are_never_candidates(tmp_path):
    from jlens.autoencoder.state import iter_valid_files

    (tmp_path / "adapter_warm_epoch000.pt").write_bytes(b"x")
    (tmp_path / "adapter_warm_epoch001.pt.tmp.1234").write_bytes(b"y")
    names = [Path(p).name for p in iter_valid_files(str(tmp_path), suffix=".pt")]
    assert names == ["adapter_warm_epoch000.pt"]


# ------------------------------------------------- 8. interruption hygiene


def test_interrupted_stage_writes_no_completion_marker(
    dataset, embedder, config, stack, token_map, tmp_path
):
    run_dir = tmp_path / "run"
    (run_dir / "state" / "reconstructor").mkdir(parents=True)
    directory = run_dir / "checkpoints"
    directory.mkdir()
    state = StageState(str(run_dir), "reconstructor")
    identity = identity_for(config, "reconstructor", run_dir)
    state.write(status="in_progress", identity=identity)

    model, optimizer, generator = build_reconstructor_training(dataset, config=config.reconstructor)
    with pytest.raises(StageInterrupted) as excinfo:
        train_reconstructor(
            dataset,
            embedder,
            config=config.reconstructor,
            source_layer=stack.source_layer,
            phrase_token_ids=token_map,
            model=model,
            optimizer=optimizer,
            generator=generator,
            checkpoint=checkpoint_writer(
                model, optimizer, generator,
                stage="reconstructor", kind="reconstructor",
                directory=directory, identity=identity, prefix="reconstructor_",
            ),
            guard=StopAfter(4, "reconstructor"),
        )
    state.write(status="interrupted", checkpoint_path=excinfo.value.checkpoint_path,
                reason="keyboard_interrupt")
    assert not Path(state.complete_path).exists(), "an interrupted stage is not complete"
    assert state.status == "interrupted"
    assert not (run_dir / "reconstructor.pt").exists(), (
        "the final artifact must only appear after the stage completes"
    )
    assert Path(excinfo.value.checkpoint_path).is_file()


def test_interrupt_guard_only_sets_a_flag(tmp_path):
    """The handler must not serialize anything; the loop checkpoints, not it."""
    import signal as signal_module

    guard = InterruptGuard("dataset", signals=[])
    assert guard.should_stop() is False
    guard._handle(getattr(signal_module, "SIGINT", 2), None)
    assert guard.should_stop() is True
    assert guard.reason == "keyboard_interrupt"
    with pytest.raises(StageInterrupted) as excinfo:
        guard.check(checkpoint_path="/tmp/x.pt")
    assert excinfo.value.checkpoint_path == "/tmp/x.pt"
    assert excinfo.value.stage == "dataset"


def test_stage_state_reports_completion_only_when_the_marker_exists(config, tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "state" / "dataset").mkdir(parents=True)
    state = StageState(str(run_dir), "dataset")
    identity = identity_for(config, "dataset", run_dir)
    assert state.status == "not_started"
    state.write(status="in_progress", identity=identity)
    assert state.status == "in_progress"
    # A process killed between the state write and the marker write.
    state.write(status="complete", identity=identity)
    assert state.status == "interrupted", (
        "a completion claim with no marker behind it must not read as complete"
    )
    artifact = run_dir / "thing.txt"
    artifact.write_text("x", encoding="utf-8")
    state.mark_complete(identity=identity, artifacts={"thing": str(artifact)})
    assert state.status == "complete"
    assert state.validate_artifacts()[0] is True
    artifact.write_text("y", encoding="utf-8")
    valid, problems = state.validate_artifacts()
    assert not valid and "checksum changed" in problems[0]


# -------------------------------------------------------------- CLI surface


def _import_script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(script_name: str, argv: list[str]) -> int:
    module = _import_script(script_name)
    saved = sys.argv
    sys.argv = [script_name, *argv]
    try:
        return module.main()
    finally:
        sys.argv = saved


@pytest.mark.parametrize(
    "script",
    [
        "build_jspace_language_dataset",
        "train_phrase_reconstructor",
        "train_cone_adapter",
        "evaluate_jspace_language",
    ],
)
def test_every_script_exposes_the_shared_resume_flags(script):
    module = _import_script(script)
    saved = sys.argv
    sys.argv = [script, "--config", str(CONFIG_PATH), "--help"]
    try:
        with pytest.raises(SystemExit):
            module.parse_args()
    finally:
        sys.argv = saved
    # Parsing the flags is the real check: --help only proves they are printed.
    sys.argv = [
        script,
        "--config",
        str(CONFIG_PATH),
        "--run-dir",
        "/tmp/x",
        "--no-resume",
        "--force",
        "--checkpoint-every-steps",
        "5",
        "--status-only",
        "--validate-state",
    ]
    try:
        args = module.parse_args()
    finally:
        sys.argv = saved
    assert args.output_dir == "/tmp/x"
    assert args.resume is False
    assert args.force is True
    assert args.checkpoint_every_steps == 5
    assert args.status_only is True
    assert args.validate_state is True


def test_status_only_reports_without_doing_work(tmp_path, capsys):
    run_dir = tmp_path / "run"
    base = ["--config", str(CONFIG_PATH), "--smoke", "--output-dir", str(run_dir)]
    assert _run("build_jspace_language_dataset", [*base, "--status-only"]) == 0
    out = capsys.readouterr().out
    assert "not_started" in out
    assert not (run_dir / "dataset" / "records.jsonl").exists()


def test_validate_state_reports_without_doing_work(tmp_path, capsys):
    run_dir = tmp_path / "run"
    base = ["--config", str(CONFIG_PATH), "--smoke", "--output-dir", str(run_dir)]
    assert _run("build_jspace_language_dataset", base) == 0
    capsys.readouterr()
    assert _run("build_jspace_language_dataset", [*base, "--validate-state"]) == 0
    out = capsys.readouterr().out
    assert "valid:" in out and "records.jsonl" in out


def test_stage_announcements_use_the_four_defined_sentences(tmp_path, capsys):
    run_dir = tmp_path / "run"
    base = ["--config", str(CONFIG_PATH), "--smoke", "--output-dir", str(run_dir)]
    assert _run("build_jspace_language_dataset", base) == 0
    assert "starting new stage" in capsys.readouterr().out
    assert _run("build_jspace_language_dataset", base) == 0
    assert "already complete; skipping" in capsys.readouterr().out
