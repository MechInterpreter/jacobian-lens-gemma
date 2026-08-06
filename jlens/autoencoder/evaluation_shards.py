# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Incremental, idempotent held-out evaluation.

Evaluation is the most expensive stage per unit of output: eight baselines and
three paraphrases, each a full beam generation, over every held-out record. It
is also the stage where an interruption used to cost the most, because
:func:`~jlens.autoencoder.evaluation.evaluate_baselines` returned once, at the
end, or not at all.

This module persists **one atomic shard per unit of work** —
``(record, baseline)``, ``(prompt, record)``, and the confabulation probe — each
carrying enough provenance to say what produced it. On rerun, shards whose
provenance still matches are reused and only the missing combinations are
computed. Aggregation is a pure function of the shards, so ``evaluation.json``,
``gonogo.json``, and ``summary.md`` come out the same whether the run was
interrupted five times or none.

**Identity of a shard** is the hash of everything that could change its value:
the adapter and reconstructor weights, the dataset record's cone, the decoding
settings, the acceptance thresholds, and the reward configuration. Change the
beam width and every generation shard is invalidated; change only the GO/NO-GO
*thresholds* and none are, because thresholds are applied during aggregation —
which is why a threshold sweep does not re-run a single beam.

**The one ordering subtlety.** ``sample_unrelated_cones`` draws from a generator
that advances once per record, so a record's unrelated cones depend on how many
records preceded it. The draws are cheap and the generations are not, so the
resumed run *replays every draw in order* and skips only the expensive part.
That is what makes a resumed evaluation numerically identical to an
uninterrupted one rather than merely similar.
"""

from __future__ import annotations

import os
import time
from collections.abc import Sequence

import torch

from jlens.autoencoder.baselines import BASELINE_IDS, run_baselines
from jlens.autoencoder.checkpoints import state_dict_sha256
from jlens.autoencoder.config import EvaluationConfig
from jlens.autoencoder.errors import AutoencoderError
from jlens.autoencoder.evaluation import (
    build_confabulation_probe,
    count_model_passes,
    summarize_results,
)
from jlens.autoencoder.inference import exact_match, first_token_rank, verbalize_cone
from jlens.autoencoder.preference import (
    normalize_candidate_text,
    sample_unrelated_cones,
)
from jlens.autoencoder.prompting import build_verbalizer_prompt
from jlens.autoencoder.state import (
    InterruptGuard,
    StageInterrupted,
    atomic_write_json,
    read_json_if_valid,
    sha256_of_json,
)

SHARD_SCHEMA = "jlens.autoencoder.evaluation.shard.v1"

#: Shard families. Each lives in its own subdirectory so a configuration change
#: that invalidates one (say, the paraphrase list) leaves the others alone.
SHARD_KINDS = ("baseline", "robustness", "confabulation")


def _generation_identity(
    *,
    config: EvaluationConfig,
    reward_config,
    adapter_sha256: str,
    reconstructor_sha256: str,
    source_layer: int,
) -> dict:
    """What every generation shard must agree on to be reusable.

    Notably absent: the ``gate_*`` thresholds. Those are applied when the report
    is assembled, so moving one re-aggregates instead of regenerating.
    """
    return {
        "adapter_sha256": adapter_sha256,
        "reconstructor_sha256": reconstructor_sha256,
        "source_layer": int(source_layer),
        "beam_width": int(config.beam_width),
        "max_new_tokens": int(config.max_new_tokens),
        "n_unrelated_cones": int(config.n_unrelated_cones),
        "accept_recon_min": float(config.accept_recon_min),
        "accept_margin_min": float(config.accept_margin_min),
        "seed": int(config.seed),
        "reward": {
            "weight_reconstruction": float(reward_config.weight_reconstruction),
            "weight_margin": float(reward_config.weight_margin),
            "brevity_penalty": float(reward_config.brevity_penalty),
            "brevity_target_tokens": int(reward_config.brevity_target_tokens),
            "duplicate_penalty": float(reward_config.duplicate_penalty),
            "length_normalize": bool(reward_config.length_normalize),
        },
    }


class EvaluationShardStore:
    """Reads and writes evaluation shards under ``root``.

    A shard is valid only if its recorded ``identity_sha256`` matches the current
    one. That is the whole invalidation rule: no timestamps, no manual cache
    busting, and no way for a changed adapter to be scored against a stale beam.

    Writing and reuse are separate switches. ``--force`` / ``--no-resume`` mean
    "do this work again", not "and be unrecoverable while doing it": those runs
    still persist shards (so an interruption mid-restart is not total loss) but
    start from none of them.
    """

    def __init__(self, root: str, *, identity: dict, enabled: bool = True, reuse: bool = True):
        self.root = root
        self.identity = dict(identity)
        self.identity_sha256 = sha256_of_json(identity)
        self.enabled = bool(enabled)
        self.reuse = bool(reuse)
        self.reused = 0
        self.computed = 0
        self.invalidated = 0

    def _dir(self, kind: str) -> str:
        if kind not in SHARD_KINDS:
            raise AutoencoderError(f"unknown shard kind {kind!r}; known are {list(SHARD_KINDS)}")
        return os.path.join(self.root, kind)

    def path(self, kind: str, name: str) -> str:
        return os.path.join(self._dir(kind), f"{name}.json")

    def read(self, kind: str, name: str) -> dict | None:
        if not self.enabled or not self.reuse:
            return None
        payload = read_json_if_valid(self.path(kind, name))
        if not payload or payload.get("schema") != SHARD_SCHEMA:
            return None
        if payload.get("identity_sha256") != self.identity_sha256:
            self.invalidated += 1
            return None
        self.reused += 1
        return payload

    def write(self, kind: str, name: str, result: dict, *, provenance: dict) -> str:
        payload = {
            "schema": SHARD_SCHEMA,
            "kind": kind,
            "name": name,
            "identity_sha256": self.identity_sha256,
            "identity": self.identity,
            "provenance": dict(provenance),
            "result": result,
        }
        self.computed += 1
        if not self.enabled:
            return ""
        return atomic_write_json(self.path(kind, name), payload)

    def stats(self) -> dict:
        return {
            "enabled": self.enabled,
            "reuse": self.reuse,
            "root": os.path.abspath(self.root) if self.enabled else None,
            "identity_sha256": self.identity_sha256,
            "reused_shards": self.reused,
            "computed_shards": self.computed,
            "invalidated_shards": self.invalidated,
        }


def _record_name(index: int) -> str:
    return f"record_{int(index):06d}"


def evaluate_baselines_sharded(
    model,
    adapter,
    reconstructor,
    embedder,
    dataset,
    prompt,
    *,
    config: EvaluationConfig,
    reward_config,
    conditioner,
    stop_token_ids: Sequence[int],
    pad_token_id: int,
    source_layer: int,
    dictionary=None,
    split: str = "heldout",
    baselines: Sequence[str] = BASELINE_IDS,
    device: torch.device | str = "cpu",
    limit: int | None = None,
    log: object = None,
    store: EvaluationShardStore | None = None,
    guard: InterruptGuard | None = None,
) -> dict:
    """Every baseline on every record of ``split``, reusing valid shards.

    Returns the same document shape as
    :func:`~jlens.autoencoder.evaluation.evaluate_baselines`. Only the missing
    ``(record, baseline)`` combinations are generated: ``run_baselines`` is
    called with the missing subset, which keeps its own sharing intact (the
    adapter beam feeds both ``adapter_raw_beam`` and ``adapter_reranked``, and is
    generated once when either is missing).
    """
    indices = dataset.indices_for_split(split)
    if limit is not None:
        indices = indices[: int(limit)]
    if not indices:
        raise AutoencoderError(f"split {split!r} has no records to evaluate")
    generator = torch.Generator().manual_seed(int(config.seed))
    per_baseline: dict[str, list[tuple[dict, str]]] = {b: [] for b in baselines}
    per_record: list[dict] = []
    passes_total = {"n_calls": 0, "n_rows": 0, "n_positions": 0}
    elapsed_total = 0.0
    for index in indices:
        record = dataset.records[index]
        # Drawn for every record, computed or reused: this generator's position
        # is part of the result, so skipping the draw for a cached record would
        # change the cones every later record sees.
        unrelated = sample_unrelated_cones(
            dataset, index, n=config.n_unrelated_cones, generator=generator
        )
        cached: dict[str, dict] = {}
        missing: list[str] = []
        for baseline_id in baselines:
            shard = (
                None
                if store is None
                else store.read("baseline", f"{_record_name(index)}__{baseline_id}")
            )
            if shard is None:
                missing.append(baseline_id)
            else:
                cached[baseline_id] = shard["result"]
                # Read once. The cost this shard recorded when it was computed
                # is the cost of the work it stands in for, so the aggregate
                # totals come out the same as an uninterrupted run's.
                recorded = (shard.get("provenance") or {}).get("model_passes") or {}
                for key in passes_total:
                    passes_total[key] += float(recorded.get(key, 0))
                elapsed_total += float((shard.get("provenance") or {}).get("wall_seconds") or 0.0)
        first_token_shard = None if store is None else store.read("baseline", f"{_record_name(index)}__first_token")
        if missing or first_token_shard is None:
            if guard is not None and guard.should_stop():
                raise StageInterrupted(
                    f"evaluation: stopped before record {index}",
                    stage="evaluation",
                    checkpoint_path=None if store is None else store.root,
                )
        results: dict[str, dict] = dict(cached)
        if missing:
            started = time.perf_counter()
            with count_model_passes(model) as passes:
                fresh = run_baselines(
                    model,
                    adapter,
                    reconstructor,
                    embedder,
                    dataset,
                    prompt,
                    index,
                    conditioner=conditioner,
                    unrelated_cones=unrelated,
                    reward_config=reward_config,
                    stop_token_ids=stop_token_ids,
                    pad_token_id=pad_token_id,
                    beam_width=config.beam_width,
                    max_new_tokens=config.max_new_tokens,
                    source_layer=source_layer,
                    accept_recon_min=config.accept_recon_min,
                    accept_margin_min=config.accept_margin_min,
                    dictionary=dictionary,
                    baselines=missing,
                    seed=int(config.seed),
                    device=device,
                )
            record_elapsed = time.perf_counter() - started
            elapsed_total += record_elapsed
            for key in passes_total:
                passes_total[key] += float(passes[key])
            # ``run_baselines`` shares one adapter beam between two baselines, so
            # per-baseline cost is not separable. The measured total is divided
            # evenly across the baselines this call produced: the split is
            # arbitrary, the sum it reconstructs is not.
            share = {k: v / max(1, len(missing)) for k, v in passes.items()}
            for baseline_id, result in fresh.items():
                results[baseline_id] = result
                if store is not None:
                    store.write(
                        "baseline",
                        f"{_record_name(index)}__{baseline_id}",
                        result,
                        provenance={
                            "record_index": int(index),
                            "phrase": record["phrase"],
                            "phrase_id": record["phrase_id"],
                            "cone_sha256": record.get("cone_sha256"),
                            "baseline_id": baseline_id,
                            "split": split,
                            "model_passes": {k: float(v) for k, v in share.items()},
                            "wall_seconds": record_elapsed / max(1, len(missing)),
                        },
                    )
        for baseline_id in baselines:
            if baseline_id not in results:
                raise AutoencoderError(
                    f"baseline {baseline_id!r} missing for record {index} after evaluation"
                )
            per_baseline[baseline_id].append((results[baseline_id], record["phrase"]))

        if first_token_shard is not None:
            first_token = first_token_shard["result"]
            provenance = first_token_shard.get("provenance") or {}
            for key in passes_total:
                passes_total[key] += float((provenance.get("model_passes") or {}).get(key, 0))
        else:
            first_token = None
            started = time.perf_counter()
            with count_model_passes(model) as first_token_passes:
                if record["phrase_token_ids"]:
                    with torch.no_grad():
                        memory = adapter(dataset.cones[index].unsqueeze(0).to(device))
                    first_token = first_token_rank(
                        model,
                        prompt,
                        memory,
                        int(record["phrase_token_ids"][0]),
                        conditioner=conditioner,
                    )
            elapsed_total += time.perf_counter() - started
            for key in passes_total:
                passes_total[key] += float(first_token_passes[key])
            if store is not None:
                store.write(
                    "baseline",
                    f"{_record_name(index)}__first_token",
                    first_token,
                    provenance={
                        "record_index": int(index),
                        "split": split,
                        "model_passes": {k: float(v) for k, v in first_token_passes.items()},
                    },
                )
        per_record.append(
            {
                "record_index": index,
                "phrase": record["phrase"],
                "phrase_id": record["phrase_id"],
                "first_token": first_token,
                "results": {
                    baseline_id: {
                        "verdict": result["verdict"],
                        "top_text": (
                            result["top_candidate"]["text"]
                            if result.get("top_candidate")
                            else None
                        ),
                        "top_cosine": (
                            result["top_candidate"]["cosine"]
                            if result.get("top_candidate")
                            else None
                        ),
                        "top_explained_fraction": (
                            result["top_candidate"]["explained_fraction"]
                            if result.get("top_candidate")
                            else None
                        ),
                        "exact_top1": exact_match(result, record["phrase"], top_k=1),
                    }
                    for baseline_id, result in results.items()
                },
            }
        )
        if log is not None:
            log.info("evaluated record %d (%s)", index, record["phrase"])
    summaries = {
        baseline_id: summarize_results(entries)
        for baseline_id, entries in per_baseline.items()
        if entries
    }
    first_token_ranks = [
        r["first_token"]["first_token_rank"] for r in per_record if r["first_token"]
    ]
    return {
        "split": split,
        "n_records": len(indices),
        "baselines": summaries,
        "per_record": per_record,
        "mean_first_token_rank": (
            sum(first_token_ranks) / len(first_token_ranks) if first_token_ranks else None
        ),
        "first_token_top1_rate": (
            sum(1 for r in first_token_ranks if r == 0) / len(first_token_ranks)
            if first_token_ranks
            else None
        ),
        "resources": {
            # Summed from per-shard records rather than measured over this
            # process, so a resumed run still reports the cost of the whole
            # evaluation rather than of its last leg.
            #
            # This is the one block that a resumed run may report differently
            # from an uninterrupted one, and it should: regenerating a subset of
            # a record's baselines does not cost the same as generating all of
            # them (the adapter beam is shared between two of them). Every
            # *result* in this document is identical either way; the cost of
            # arriving at it is a measurement of what actually ran.
            "wall_seconds": round(elapsed_total, 3),
            "model_forward_calls": int(round(passes_total["n_calls"])),
            "model_scored_rows": int(round(passes_total["n_rows"])),
            "model_scored_positions": int(round(passes_total["n_positions"])),
            "peak_cuda_memory_gb": (
                round(torch.cuda.max_memory_allocated() / 2**30, 3)
                if torch.cuda.is_available()
                else None
            ),
            "accounting": (
                "summed over per-record result shards; after a resume this is the "
                "cost actually incurred, which need not equal an uninterrupted run's"
            ),
        },
        "sharding": None if store is None else store.stats(),
    }


def evaluate_prompt_robustness_sharded(
    model,
    adapter,
    reconstructor,
    embedder,
    dataset,
    *,
    config: EvaluationConfig,
    reward_config,
    conditioner,
    stop_token_ids: Sequence[int],
    pad_token_id: int,
    source_layer: int,
    n_memory_tokens: int,
    split: str = "heldout",
    device: torch.device | str = "cpu",
    limit: int | None = None,
    store: EvaluationShardStore | None = None,
    guard: InterruptGuard | None = None,
) -> dict:
    """The paraphrase sweep, one shard per ``(prompt_id, record)``.

    Same generator-replay rule as the baseline pass: the draw happens for every
    record of every prompt, the generation only for the ones without a shard.
    """
    indices = dataset.indices_for_split(split)
    if limit is not None:
        indices = indices[: int(limit)]
    generator = torch.Generator().manual_seed(int(config.seed) + 5)
    per_prompt: dict[str, dict] = {}
    answers: dict[str, list[str]] = {}
    for prompt_id in config.paraphrase_prompt_ids:
        prompt = build_verbalizer_prompt(
            model.tokenizer, n_memory_tokens=n_memory_tokens, prompt_id=prompt_id
        )
        entries: list[tuple[dict, str]] = []
        texts: list[str] = []
        for index in indices:
            record = dataset.records[index]
            unrelated = sample_unrelated_cones(
                dataset, index, n=config.n_unrelated_cones, generator=generator
            )
            name = f"{prompt_id}__{_record_name(index)}"
            shard = None if store is None else store.read("robustness", name)
            if shard is not None:
                result = shard["result"]
            else:
                if guard is not None and guard.should_stop():
                    raise StageInterrupted(
                        f"evaluation: stopped before robustness {name}",
                        stage="evaluation",
                        checkpoint_path=None if store is None else store.root,
                    )
                result = verbalize_cone(
                    model,
                    adapter,
                    reconstructor,
                    embedder,
                    dataset.cones[index],
                    prompt,
                    conditioner=conditioner,
                    unrelated_cones=unrelated,
                    reward_config=reward_config,
                    stop_token_ids=stop_token_ids,
                    pad_token_id=pad_token_id,
                    beam_width=config.beam_width,
                    max_new_tokens=config.max_new_tokens,
                    source_layer=source_layer,
                    accept_recon_min=config.accept_recon_min,
                    accept_margin_min=config.accept_margin_min,
                    device=device,
                )
                if store is not None:
                    store.write(
                        "robustness",
                        name,
                        result,
                        provenance={
                            "prompt_id": prompt_id,
                            "record_index": int(index),
                            "phrase": record["phrase"],
                            "phrase_id": record["phrase_id"],
                            "cone_sha256": record.get("cone_sha256"),
                            "split": split,
                        },
                    )
            entries.append((result, record["phrase"]))
            texts.append(
                normalize_candidate_text(result["top_candidate"]["text"])
                if result.get("top_candidate")
                else ""
            )
        per_prompt[prompt_id] = summarize_results(entries)
        answers[prompt_id] = texts
    prompt_ids = list(config.paraphrase_prompt_ids)
    agreement = None
    if len(prompt_ids) > 1 and indices:
        reference = answers[prompt_ids[0]]
        matches = [
            all(answers[p][i] == reference[i] for p in prompt_ids[1:])
            for i in range(len(reference))
        ]
        agreement = sum(1 for m in matches if m) / len(matches)
    return {
        "split": split,
        "n_records": len(indices),
        "per_prompt": per_prompt,
        "cross_prompt_agreement": agreement,
        "top1_spread": (
            max(m["exact_match_top1"] for m in per_prompt.values())
            - min(m["exact_match_top1"] for m in per_prompt.values())
            if per_prompt
            else None
        ),
    }


def build_confabulation_probe_sharded(
    reconstructor,
    embedder,
    dataset,
    *,
    config: EvaluationConfig,
    phrase_token_ids: dict[str, list[int]],
    source_layer: int,
    split: str = "heldout",
    device: torch.device | str = "cpu",
    store: EvaluationShardStore | None = None,
) -> dict:
    """The attractor probe, cached as a single shard.

    One shard rather than one per attractor: the probe scores all attractors
    against all cones in a single batched pass, so splitting it would multiply
    the work it is supposed to save.
    """
    name = f"attractors__{split}"
    shard = None if store is None else store.read("confabulation", name)
    if shard is not None:
        return shard["result"]
    result = build_confabulation_probe(
        reconstructor,
        embedder,
        dataset,
        config=config,
        phrase_token_ids=phrase_token_ids,
        source_layer=source_layer,
        split=split,
        device=device,
    )
    if store is not None:
        store.write(
            "confabulation",
            name,
            result,
            provenance={
                "split": split,
                "attractors": list(config.confabulation_attractors),
                "n_records": len(dataset.indices_for_split(split)),
            },
        )
    return result


def expected_shard_names(
    dataset,
    *,
    config: EvaluationConfig,
    split: str,
    baselines: Sequence[str] = BASELINE_IDS,
    limit: int | None = None,
) -> dict[str, list[str]]:
    """Every shard a complete evaluation must have, by kind.

    Aggregation refuses to mark the stage complete unless all of these exist, so
    "evaluation.json was written" cannot come to mean "most of the records ran".
    """
    indices = dataset.indices_for_split(split)
    if limit is not None:
        indices = indices[: int(limit)]
    baseline_names = [
        f"{_record_name(i)}__{baseline_id}" for i in indices for baseline_id in baselines
    ]
    baseline_names += [f"{_record_name(i)}__first_token" for i in indices]
    return {
        "baseline": sorted(baseline_names),
        "robustness": sorted(
            f"{prompt_id}__{_record_name(i)}"
            for prompt_id in config.paraphrase_prompt_ids
            for i in indices
        ),
        "confabulation": [f"attractors__{split}"],
    }


def missing_shards(
    store: EvaluationShardStore, expected: dict[str, list[str]], *, kinds: Sequence[str] | None = None
) -> dict[str, list[str]]:
    """Which expected shards are absent or invalid, by kind."""
    wanted = list(kinds) if kinds is not None else list(expected)
    report: dict[str, list[str]] = {}
    for kind in wanted:
        missing = [
            name
            for name in expected.get(kind, [])
            if read_json_if_valid(store.path(kind, name)) is None
            or (read_json_if_valid(store.path(kind, name)) or {}).get("identity_sha256")
            != store.identity_sha256
        ]
        if missing:
            report[kind] = missing
    return report


def evaluation_identity(
    *,
    config: EvaluationConfig,
    reward_config,
    adapter,
    reconstructor,
    source_layer: int,
) -> dict:
    """Build the generation identity from the live modules."""
    return _generation_identity(
        config=config,
        reward_config=reward_config,
        adapter_sha256=state_dict_sha256(adapter),
        reconstructor_sha256=state_dict_sha256(reconstructor),
        source_layer=source_layer,
    )
