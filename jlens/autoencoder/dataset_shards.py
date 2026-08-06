# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Incremental, resumable construction of the (phrase, cone) dataset.

:func:`jlens.autoencoder.dataset.build_dataset` mines, captures, decomposes, and
returns — all or nothing. On an L4 that is hours of pursuit that a disconnected
Colab runtime takes with it. This module does the same work in **shards** that
are written as they complete, so a rebuild resumes instead of restarting.

The build is split into a cheap deterministic plan and an expensive body:

* **Plan** — mining and context tokenization. No model forward passes, so it is
  re-derived on every run rather than cached, and its hash is what a shard is
  keyed against. If the corpus or the mining policy changed, the plan hash
  changes and the old shards are refused instead of quietly reused.
* **Body** — for each chunk: capture the layer-``L`` activation before each
  occurrence, then one batched pursuit over the chunk.

The chunk boundary is *exactly* the pursuit batching of the non-incremental
builder (``dataset.capture_batch_size`` occurrences of the kept list, in mined
order). That is not a coincidence, it is the requirement: batched pursuit
reduces over the batch dimension, and re-cutting the batches would perturb
floating-point results. A shard groups a whole number of chunks, so shard size
is configurable without ever moving a batch boundary.

Splits are assigned in :func:`assemble` over the complete kept set, matching
:func:`~jlens.autoencoder.dataset.assign_splits`' contract that assignment
depends on the *set* of phrases. That is why splits cannot be written into
shards: until the last shard exists, the set is not known.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import torch

from jlens.autoencoder.config import SPLITS, DatasetConfig, PursuitConfig
from jlens.autoencoder.dataset import (
    DATASET_RECORD_SCHEMA,
    DatasetBuildResult,
    PhraseOccurrence,
    assign_splits,
    capture_source_activation,
    context_token_ids,
    mine_phrase_occurrences,
    normalize_phrase,
)
from jlens.autoencoder.errors import AutoencoderError
from jlens.autoencoder.state import (
    InterruptGuard,
    StageInterrupted,
    atomic_torch_save,
    atomic_write_json,
    iter_valid_files,
    read_json_if_valid,
    sha256_of_file,
    sha256_of_json,
    utcnow,
)
from jlens.generative import tensor_sha256, weighted_reconstruction
from jlens.pursuit import JSpaceDictionary, PursuitSettings, gradient_pursuit

SHARD_SCHEMA = "jlens.autoencoder.dataset.shard.v1"
PLAN_SCHEMA = "jlens.autoencoder.dataset.plan.v1"

#: Timing fields. Measurements of *the run*, not of the data, so they are the
#: one part of a rebuilt manifest that legitimately differs after an
#: interruption — the records and tensors do not.
TIMING_KEYS = ("capture_seconds", "pursuit_seconds", "seconds_per_occurrence")


@dataclass
class PlannedUnit:
    """One occurrence that survived context filtering, ready to be captured."""

    occurrence_index: int
    occurrence: PhraseOccurrence
    context_ids: list[int]

    @property
    def unit_id(self) -> str:
        """Stable identity of this occurrence: phrase, document, and span.

        Position-independent on purpose — a shard keyed by list index would be
        silently reusable after a mining change that shifted everything by one.
        """
        return (
            f"{self.occurrence.phrase_id}:{self.occurrence.document_sha256[7:23]}:"
            f"{self.occurrence.char_start}-{self.occurrence.char_end}"
        )


@dataclass
class BuildPlan:
    """The deterministic, model-free part of a build."""

    units: list[PlannedUnit]
    chunks: list[list[int]]
    mining_stats: dict
    rejected: list[dict] = field(default_factory=list)
    n_mined: int = 0

    @property
    def fingerprint(self) -> str:
        return sha256_of_json(
            {
                "schema": PLAN_SCHEMA,
                "unit_ids": [unit.unit_id for unit in self.units],
                "context_lengths": [len(unit.context_ids) for unit in self.units],
                "chunks": self.chunks,
            }
        )

    def candidate_phrase_ids(self) -> list[str]:
        return sorted({unit.occurrence.phrase_id for unit in self.units})

    def expected_occurrences_per_phrase(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for unit in self.units:
            counts[unit.occurrence.phrase_id] = counts.get(unit.occurrence.phrase_id, 0) + 1
        return dict(sorted(counts.items()))


def plan_build(
    model,
    documents: Sequence[str],
    *,
    dataset_config: DatasetConfig,
    phrase_token_ids: Callable[[str], list[int]],
    max_phrases: int | None = None,
) -> BuildPlan:
    """Mine and context-tokenize; produce the chunk schedule. No model passes.

    Occurrences rejected for insufficient preceding context are recorded in
    ``rejected`` rather than dropped silently — the non-incremental builder
    ``continue``s past them, and a resumable build has to be able to say why a
    mined occurrence produced no record.
    """
    occurrences, mining_stats = mine_phrase_occurrences(
        documents,
        config=dataset_config,
        phrase_token_ids=phrase_token_ids,
        max_phrases=max_phrases,
    )
    units: list[PlannedUnit] = []
    rejected: list[dict] = []
    for index, occurrence in enumerate(occurrences):
        try:
            ids = context_token_ids(
                model.tokenizer,
                occurrence.context_text,
                max_context_tokens=dataset_config.max_context_tokens,
                min_context_tokens=dataset_config.min_context_tokens,
            )
        except AutoencoderError as exc:
            rejected.append(
                {
                    "occurrence_index": index,
                    "phrase_id": occurrence.phrase_id,
                    "phrase": occurrence.phrase,
                    "reason": "insufficient_context",
                    "error": str(exc),
                }
            )
            continue
        units.append(PlannedUnit(occurrence_index=index, occurrence=occurrence, context_ids=ids))
    if not units:
        raise AutoencoderError(
            "every mined occurrence was rejected for insufficient preceding "
            "context; lower dataset.min_context_tokens"
        )
    batch = max(1, int(dataset_config.capture_batch_size))
    chunks = [list(range(start, min(start + batch, len(units)))) for start in range(0, len(units), batch)]
    return BuildPlan(
        units=units,
        chunks=chunks,
        mining_stats=mining_stats,
        rejected=rejected,
        n_mined=len(occurrences),
    )


def chunks_per_shard(dataset_config: DatasetConfig, shard_size: int | None) -> int:
    """How many pursuit chunks a shard holds.

    ``shard_size`` is expressed in occurrences because that is what an operator
    thinks in, but it is snapped up to a whole number of ``capture_batch_size``
    chunks: a shard that ended mid-chunk would force the pursuit batch to be
    re-cut on resume, and the results would no longer be bit-identical to an
    uninterrupted run.
    """
    batch = max(1, int(dataset_config.capture_batch_size))
    if shard_size is None:
        return 1
    if int(shard_size) < 1:
        raise AutoencoderError(f"dataset shard size must be >= 1, got {shard_size}")
    return max(1, int(round(float(shard_size) / batch)))


# ------------------------------------------------------------------- storage


def shard_path(shard_dir: str, shard_index: int) -> str:
    return os.path.join(shard_dir, f"shard_{shard_index:06d}.pt")


def _shard_sidecar(path: str) -> str:
    return os.path.splitext(path)[0] + ".json"


def write_shard(
    shard_dir: str,
    shard_index: int,
    *,
    plan_fingerprint: str,
    chunk_indices: list[int],
    unit_ids: list[str],
    activations: torch.Tensor,
    cones: torch.Tensor,
    pursuit_records: list[dict],
    context_lengths: list[int],
    capture_seconds: float,
    pursuit_seconds: float,
) -> dict:
    """Write one shard atomically, then its sidecar; return the sidecar.

    Order matters: the ``.pt`` lands first and the ``.json`` (which carries its
    checksum) second, so a shard with a sidecar is always a complete shard. A
    ``.pt`` with no sidecar is treated as absent and recomputed.
    """
    path = shard_path(shard_dir, shard_index)
    payload = {
        "schema": SHARD_SCHEMA,
        "shard_index": int(shard_index),
        "plan_fingerprint": plan_fingerprint,
        "chunk_indices": list(chunk_indices),
        "unit_ids": list(unit_ids),
        "activations": activations,
        "cones": cones,
        "pursuit_records": list(pursuit_records),
        "context_lengths": list(context_lengths),
        "capture_seconds": float(capture_seconds),
        "pursuit_seconds": float(pursuit_seconds),
    }
    atomic_torch_save(payload, path)
    sidecar = {
        "schema": SHARD_SCHEMA,
        "shard_index": int(shard_index),
        "plan_fingerprint": plan_fingerprint,
        "chunk_indices": list(chunk_indices),
        "unit_ids": list(unit_ids),
        "n_units": len(unit_ids),
        "written_utc": utcnow(),
        "shard_sha256": sha256_of_file(path),
    }
    atomic_write_json(_shard_sidecar(path), sidecar)
    return sidecar


def load_shard(path: str, *, plan_fingerprint: str | None = None) -> dict | None:
    """Load and validate one shard, or return ``None`` if it is unusable.

    Validated three ways: the sidecar must exist (the write completed), the
    file checksum must match it (the bytes are intact), and the plan fingerprint
    must agree (the shard belongs to *this* build).
    """
    sidecar = read_json_if_valid(_shard_sidecar(path))
    if not sidecar or sidecar.get("schema") != SHARD_SCHEMA:
        return None
    if not os.path.isfile(path):
        return None
    if sidecar.get("shard_sha256") and sha256_of_file(path) != sidecar["shard_sha256"]:
        return None
    if plan_fingerprint is not None and sidecar.get("plan_fingerprint") != plan_fingerprint:
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:  # noqa: BLE001 - an unreadable shard is recomputed
        return None
    if not isinstance(payload, dict) or payload.get("schema") != SHARD_SCHEMA:
        return None
    if list(payload.get("unit_ids") or []) != list(sidecar.get("unit_ids") or []):
        return None
    return payload


def validate_shards(shard_dir: str, *, plan_fingerprint: str | None = None) -> dict:
    """Report which shards in ``shard_dir`` are usable, and why the rest are not."""
    valid: dict[int, dict] = {}
    invalid: list[dict] = []
    for path in iter_valid_files(shard_dir, suffix=".pt"):
        payload = load_shard(path, plan_fingerprint=plan_fingerprint)
        if payload is None:
            invalid.append({"path": path, "reason": "missing sidecar, checksum, or plan mismatch"})
            continue
        valid[int(payload["shard_index"])] = payload
    return {"valid": valid, "invalid": invalid}


# --------------------------------------------------------------------- build


def _pursue_chunk(
    activations: torch.Tensor,
    dictionary: JSpaceDictionary,
    settings: PursuitSettings,
) -> tuple[list[dict], list[torch.Tensor]]:
    chunk = activations.to(dictionary.device)
    result = gradient_pursuit(chunk, dictionary, settings)
    records: list[dict] = []
    cones: list[torch.Tensor] = []
    for record in result.to_records():
        cone = weighted_reconstruction(
            dictionary.atoms, record["token_ids"], record["coefficients"]
        )
        cones.append(cone.detach().float().cpu())
        records.append(record)
    del chunk, result
    return records, cones


def build_dataset_sharded(
    model,
    lens_dictionary: JSpaceDictionary,
    documents: Sequence[str],
    *,
    dataset_config: DatasetConfig,
    pursuit_config: PursuitConfig,
    phrase_token_ids: Callable[[str], list[int]],
    shard_dir: str,
    provenance: dict | None = None,
    progress: Callable[[str], None] | None = None,
    max_phrases: int | None = None,
    shard_size: int | None = None,
    guard: InterruptGuard | None = None,
    on_progress: Callable[[dict], None] | None = None,
) -> DatasetBuildResult:
    """Build incrementally into ``shard_dir``, reusing whatever is already there.

    Raises :class:`~jlens.autoencoder.state.StageInterrupted` when ``guard``
    reports a stop request. The in-flight shard is discarded (never half
    written); everything already on disk stays.
    """
    layer = int(dataset_config.source_layer)
    if layer != lens_dictionary.layer:
        raise AutoencoderError(
            f"dataset.source_layer={layer} but the dictionary was built for layer "
            f"{lens_dictionary.layer}"
        )
    plan = plan_build(
        model,
        documents,
        dataset_config=dataset_config,
        phrase_token_ids=phrase_token_ids,
        max_phrases=max_phrases,
    )
    fingerprint = plan.fingerprint
    group = chunks_per_shard(dataset_config, shard_size)
    shards = [plan.chunks[i : i + group] for i in range(0, len(plan.chunks), group)]
    os.makedirs(shard_dir, exist_ok=True)

    existing = validate_shards(shard_dir, plan_fingerprint=fingerprint)
    settings = PursuitSettings(
        k=pursuit_config.k,
        normalize_atoms=pursuit_config.normalize_atoms,
        refine_steps=pursuit_config.refine_steps,
        tol_relative_residual=pursuit_config.tol_relative_residual,
        correlation_chunk_size=pursuit_config.correlation_chunk_size,
    )
    device = next(iter(model.layers.parameters())).device
    reused = 0

    def emit_progress(completed: int) -> None:
        if on_progress is None:
            return
        done = sorted(existing["valid"])
        on_progress(
            {
                    "plan_fingerprint": fingerprint,
                    "expected_units": len(plan.units),
                    "completed_units": sum(
                        len(existing["valid"][index]["unit_ids"]) for index in done
                    ),
                    "completed_unit_ids": [
                        unit_id
                        for index in done
                        for unit_id in existing["valid"][index]["unit_ids"]
                    ],
                    "expected_shards": len(shards),
                    "completed_shards": completed,
                    "candidate_phrase_ids": plan.candidate_phrase_ids(),
                    "expected_occurrences_per_phrase": plan.expected_occurrences_per_phrase(),
                    "failed_units": plan.rejected,
                    "corpus_cursor": {
                        "documents_scanned": plan.mining_stats.get("documents_scanned"),
                        "n_documents": len(documents),
                    },
                    "shard_hashes": {
                        str(index): (
                            read_json_if_valid(_shard_sidecar(shard_path(shard_dir, index))) or {}
                        ).get("shard_sha256")
                        for index in done
                    },
            }
        )

    for shard_index, chunk_group in enumerate(shards):
        if shard_index in existing["valid"]:
            reused += 1
            continue
        if guard is not None and guard.should_stop():
            emit_progress(shard_index)
            raise StageInterrupted(
                f"dataset: stopped after {shard_index}/{len(shards)} shards",
                stage="dataset",
                checkpoint_path=shard_dir,
            )
        activations: list[torch.Tensor] = []
        cones: list[torch.Tensor] = []
        records: list[dict] = []
        context_lengths: list[int] = []
        unit_ids: list[str] = []
        flat_indices: list[int] = []
        capture_seconds = 0.0
        pursuit_seconds = 0.0
        for chunk in chunk_group:
            started = time.perf_counter()
            chunk_activations: list[torch.Tensor] = []
            for unit_index in chunk:
                unit = plan.units[unit_index]
                tensor = torch.tensor([unit.context_ids], dtype=torch.long, device=device)
                chunk_activations.append(capture_source_activation(model, tensor, layer).cpu())
                context_lengths.append(len(unit.context_ids))
                unit_ids.append(unit.unit_id)
                flat_indices.append(unit_index)
            capture_seconds += time.perf_counter() - started
            stacked = torch.stack(chunk_activations)
            started = time.perf_counter()
            chunk_records, chunk_cones = _pursue_chunk(stacked, lens_dictionary, settings)
            pursuit_seconds += time.perf_counter() - started
            activations.append(stacked)
            cones.extend(chunk_cones)
            records.extend(chunk_records)
        write_shard(
            shard_dir,
            shard_index,
            plan_fingerprint=fingerprint,
            chunk_indices=list(flat_indices),
            unit_ids=unit_ids,
            activations=torch.cat(activations),
            cones=torch.stack(cones),
            pursuit_records=records,
            context_lengths=context_lengths,
            capture_seconds=capture_seconds,
            pursuit_seconds=pursuit_seconds,
        )
        existing["valid"][shard_index] = load_shard(
            shard_path(shard_dir, shard_index), plan_fingerprint=fingerprint
        )
        emit_progress(shard_index + 1)
        if progress is not None:
            progress(
                f"dataset shard {shard_index + 1}/{len(shards)} "
                f"({len(unit_ids)} occurrences) written"
            )
    emit_progress(len(shards))
    if reused and progress is not None:
        progress(f"reused {reused}/{len(shards)} shards from a previous run")
    return assemble(
        plan,
        shard_dir,
        dataset_config=dataset_config,
        pursuit_config=pursuit_config,
        phrase_token_ids=phrase_token_ids,
        provenance=provenance,
        n_shards=len(shards),
        n_reused_shards=reused,
    )


def assemble(
    plan: BuildPlan,
    shard_dir: str,
    *,
    dataset_config: DatasetConfig,
    pursuit_config: PursuitConfig,
    phrase_token_ids: Callable[[str], list[int]],
    provenance: dict | None = None,
    n_shards: int | None = None,
    n_reused_shards: int = 0,
) -> DatasetBuildResult:
    """Deterministically combine shards into the same result a clean build gives.

    Pure and idempotent: the only inputs are the plan and the shards, so running
    it twice over unchanged shards produces byte-identical records and tensors.
    Splits are computed here because they depend on the complete kept set.
    """
    found = validate_shards(shard_dir, plan_fingerprint=plan.fingerprint)
    ordered = [found["valid"][index] for index in sorted(found["valid"])]
    if n_shards is not None and len(ordered) != n_shards:
        missing = sorted(set(range(n_shards)) - set(found["valid"]))
        raise AutoencoderError(
            f"cannot assemble the dataset: {len(missing)} shard(s) missing or invalid "
            f"({missing[:8]}{'...' if len(missing) > 8 else ''}). Rerun the dataset "
            f"stage to compute them."
        )
    unit_ids: list[str] = []
    flat_indices: list[int] = []
    activations: list[torch.Tensor] = []
    cones: list[torch.Tensor] = []
    pursuit_records: list[dict] = []
    context_lengths: list[int] = []
    capture_seconds = 0.0
    pursuit_seconds = 0.0
    for payload in ordered:
        unit_ids.extend(payload["unit_ids"])
        flat_indices.extend(int(i) for i in payload["chunk_indices"])
        activations.append(payload["activations"])
        cones.append(payload["cones"])
        pursuit_records.extend(payload["pursuit_records"])
        context_lengths.extend(int(n) for n in payload["context_lengths"])
        capture_seconds += float(payload.get("capture_seconds") or 0.0)
        pursuit_seconds += float(payload.get("pursuit_seconds") or 0.0)
    if not unit_ids:
        raise AutoencoderError(f"no valid dataset shards under {shard_dir}")
    expected_ids = [unit.unit_id for unit in plan.units][: len(unit_ids)]
    if unit_ids != expected_ids:
        raise AutoencoderError(
            "shard unit order does not match the build plan; the corpus or mining "
            "policy changed under an existing shard directory"
        )

    activation_tensor = torch.cat(activations)
    cone_tensor = torch.cat(cones)
    kept = [plan.units[index].occurrence for index in flat_indices]
    splits = assign_splits(
        [occurrence.phrase for occurrence in kept],
        salt=dataset_config.split_salt,
        val_fraction=dataset_config.val_fraction,
        heldout_fraction=dataset_config.heldout_fraction,
    )
    records: list[dict] = []
    token_ids_per_record: list[list[int]] = []
    for index, occurrence in enumerate(kept):
        ids = phrase_token_ids(occurrence.phrase)
        token_ids_per_record.append(ids)
        activation = activation_tensor[index]
        cone = cone_tensor[index]
        pursuit_record = pursuit_records[index]
        records.append(
            {
                "schema": DATASET_RECORD_SCHEMA,
                "record_index": index,
                "phrase": occurrence.phrase,
                "phrase_normalized": normalize_phrase(occurrence.phrase),
                "phrase_id": occurrence.phrase_id,
                "phrase_token_ids": list(ids),
                "n_phrase_tokens": len(ids),
                "split": splits[normalize_phrase(occurrence.phrase)],
                "source": occurrence.to_dict(),
                "context_token_len": context_lengths[index],
                "source_layer": int(dataset_config.source_layer),
                "source_activation_norm": float(activation.norm()),
                "source_activation_sha256": tensor_sha256(activation),
                "cone_norm": float(cone.norm()),
                "cone_sha256": tensor_sha256(cone),
                "active_token_ids": pursuit_record["token_ids"],
                "active_coefficients": pursuit_record["coefficients"],
                "n_active_atoms": pursuit_record["n_selected"],
                "pursuit_explained_fraction": pursuit_record["explained_fraction"],
                "pursuit_relative_residual": pursuit_record["relative_residual"],
                "pursuit_stop_reason": pursuit_record["stop_reason"],
                "provenance": dict(provenance or {}),
            }
        )
    stats = {
        **plan.mining_stats,
        "n_occurrences_mined": plan.n_mined,
        "n_occurrences_captured": len(kept),
        "n_phrases_captured": len({r["phrase_id"] for r in records}),
        "capture_seconds": round(capture_seconds, 3),
        "pursuit_seconds": round(pursuit_seconds, 3),
        "seconds_per_occurrence": round(
            (capture_seconds + pursuit_seconds) / max(1, len(kept)), 4
        ),
        "d_model": int(activation_tensor.shape[1]),
        "split_counts": {
            split: sum(1 for r in records if r["split"] == split) for split in SPLITS
        },
        "phrase_split_counts": {
            split: len({r["phrase_id"] for r in records if r["split"] == split})
            for split in SPLITS
        },
        "sharded_build": {
            "n_shards": len(ordered),
            "n_reused_shards": int(n_reused_shards),
            "plan_fingerprint": plan.fingerprint,
            "shard_dir": os.path.abspath(shard_dir),
        },
    }
    return DatasetBuildResult(
        records=records,
        activations=activation_tensor,
        cones=cone_tensor,
        phrase_token_ids=token_ids_per_record,
        stats=stats,
    )
