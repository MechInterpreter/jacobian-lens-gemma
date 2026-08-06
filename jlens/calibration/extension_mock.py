# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Deterministic MOCK mode for the early-layer extension.

No Gemma, no Hub, no Drive, no corpus download — and, unlike a stub, no
simulation of the parts that matter. Built in two halves, following
:mod:`jlens.calibration.mock`:

**Half one — the parent run is real.** :func:`build_mock_parent_run` produces a
parent by *running the completed study's own code*: the real
:class:`~jlens.calibration.state.CalibrationStore`, the real
:func:`~jlens.calibration.fitting.run_calibration`, and therefore the real
upstream :func:`jlens.fitting.fit` against
:class:`~jlens.calibration.mock.MockCalibrationModel`. The resulting directory
has the real layout, the real unit checksums, a real ``jacobian_sum`` checkpoint
and a real scale snapshot. The parent audit and the continuation are then
exercised against an artifact of the same kind they will meet on Drive, at a
width where a full refit costs seconds — which is what makes
:func:`~jlens.calibration.extension.verify_continuation_equals_fresh_fit`
affordable here and unaffordable in a real run.

**Half two — the verdicts meet known cases.** Development and confirmation rows
are synthesised with controlled rank and tie structure and then scored by the
**real** :func:`jlens.mmlocalize.lens_validity.tie_aware_row`, so the scoring,
the gate, the folds, the scale comparison and the selection rule are the real
ones and only the inputs are fixtures.

The commissioned scenarios, each of which the pipeline must handle differently:

===========================  =================================================
``l32_late_pass``            L32 fails at 250, passes development at 1,000,
                             then passes fresh confirmation. GO, published.
``l32_confirmation_fail``    L32 passes development at both scales — so the
                             *smaller* scale is selected — and then fails the
                             untouched confirmation. NO-GO, nothing published,
                             and the layer is never marked validated.
``no_early_layer``           L32 and L26 both fail at 1,000. The predeclared
                             rule still takes 1,000 to one honest confirmation.
                             NO-GO.
``old_confirmation_leak``    An old confirmation record is planted in a new
                             split; the cross-split audit refuses it.
``continuation_mismatch``    A continued accumulator that does not equal a
                             fresh nested fit; the equivalence check refuses it.
===========================  =================================================

**A green MOCK run proves pipeline behaviour and nothing else.** It is not
evidence about Gemma 4, about layer 32, about layer 26, or about whether a
validated earlier readout exists.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import torch

from jlens.calibration.corpus import (
    CorpusRecord,
    audit_leakage,
    build_records,
    collect_records_for_partition_quotas,
    corpus_manifest,
    scale_nesting_audit,
)
from jlens.calibration.extension import (
    EXTENSION_GATE,
    FreshEvaluationSplits,
)
from jlens.calibration.fitting import filter_records_by_tokens, run_calibration
from jlens.calibration.gate import CALIBRATION_GATE, select_diverse_validation_prompts
from jlens.calibration.mock import MockCalibrationModel, mock_corpus_texts
from jlens.calibration.plan import build_capture_plan
from jlens.calibration.scale import PLATEAU_RULE
from jlens.calibration.state import CalibrationFingerprint, CalibrationStore
from jlens.mmlocalize.lens_validity import tie_aware_row
from jlens.mmpilot.store import payload_checksum

__all__ = [
    "EXTENSION_SCENARIOS",
    "MOCK_BASELINE_SCALE",
    "MOCK_CORPUS_ID",
    "MOCK_DEVELOPMENT_SCALES",
    "MOCK_EXTENSION_SCALES",
    "MOCK_LAYERS",
    "MOCK_PARENT_PROTOCOL_VERSION",
    "MOCK_TARGET_MODULUS",
    "ExtensionScenario",
    "LayerBehaviour",
    "MockParent",
    "build_mock_parent_run",
    "corrupt_continued_checkpoint",
    "mock_capture_plan",
    "mock_extension_environment",
    "mock_extension_gate_sizes",
    "mock_extension_pool",
    "mock_extension_rows",
    "mock_load_info",
    "mock_target_token",
    "plant_old_confirmation_record",
]

MOCK_CORPUS_ID = "mock/train"
MOCK_PARENT_PROTOCOL_VERSION = (
    "research-grade-multilayer-text-jlens-calibration-v2-feasible-scales"
)

#: Small synthetic equivalents of 100 / 250 / 1,000. Nesting and continuation
#: behaviour are what is under test, not the absolute numbers.
MOCK_BASELINE_SCALE = 8
MOCK_EXTENSION_SCALES = (12, 20)

#: The scales the MOCK development set is scored at: the descriptive baseline
#: plus the two candidates, mirroring
#: :data:`jlens.calibration.extension.DEVELOPMENT_SCALE_POINTS`.
MOCK_DEVELOPMENT_SCALES = (MOCK_BASELINE_SCALE, *MOCK_EXTENSION_SCALES)

#: How many corpus texts the mock stream offers. Enough that the parent's
#: collection stops well short of the end, leaving a genuinely untouched region
#: for the fresh evaluation sets.
MOCK_STREAM_TEXTS = 4_000

#: Target-token modulus for the mock target-discovery callable. 64 possible
#: targets over 256 prompts clears the extension's stricter 32-token floor with
#: room to spare, and keeps the largest single-target share far below the 25%
#: ceiling.
MOCK_TARGET_MODULUS = 64

MOCK_VOCAB = 512
MOCK_LAYERS = (8, 14, 20, 26, 32, 35, 38, 40)

#: Fixed optimistic rank for each control at every layer and scale, as in
#: :mod:`jlens.calibration.mock`: deep enough that a real readout clears the
#: ratio and margin clauses, and identical across layers so a control number
#: never depends on which layer it sits under.
CONTROL_RANK = {
    "permuted": 250,
    "random": 300,
    "wrong_layer": 200,
    "logit_lens": 220,
}


def mock_target_token(prompt: str) -> int:
    """The 'model's own output target' for a mock prompt.

    Reads the record index out of the synthetic text. Deterministic, and — like
    the real :func:`~jlens.calibration.gate.ordinary_next_token_argmax` — it has
    no parameter through which a J-lens or a candidate layer could be supplied.
    """
    parts = str(prompt).split()
    try:
        index = int(parts[1])
    except (IndexError, ValueError):
        index = int(hashlib.sha256(str(prompt).encode()).hexdigest()[:8], 16)
    return index % MOCK_TARGET_MODULUS


def mock_capture_plan(model: MockCalibrationModel):
    """The mock stack's capture plan: the real layer grid at a tiny width."""
    return build_capture_plan(
        layers=MOCK_LAYERS,
        target_layer=41,
        d_model=model.d_model,
        dim_batch=8,
        max_seq_len=48,
        skip_first=4,
        n_layers=model.n_layers,
    )


# --------------------------------------------------------------- the parent


@dataclass(frozen=True)
class MockParent:
    """A synthetic parent run, produced by the completed study's own code."""

    root: str
    model: MockCalibrationModel
    plan: object
    records: tuple[CorpusRecord, ...]
    partitions: object
    fit_records: tuple[CorpusRecord, ...]
    baseline_scale: int
    last_stream_index: int
    fingerprint: CalibrationFingerprint
    corpus_config: dict

    @property
    def fit_prefix_checksum(self) -> str:
        return payload_checksum(
            [record.to_dict() for record in self.fit_records[: self.baseline_scale]]
        )


def _mock_corpus_config() -> dict:
    return {
        "hf_dataset": "mock",
        "config": "mock",
        "split": "train",
        "revision": "mock-revision",
        "revision_status": "MOCK_NO_CORPUS_WAS_DOWNLOADED",
        "min_chars": 100,
        "license": "n/a",
    }


def build_mock_parent_run(
    root: str | os.PathLike[str],
    *,
    baseline_scale: int = MOCK_BASELINE_SCALE,
    n_validation: int = 32,
    n_confirmation: int = 32,
    model: MockCalibrationModel | None = None,
    confirmation_opened: bool = True,
) -> MockParent:
    """Produce a complete parent calibration run on disk.

    Runs the real fitting path, writes the real units, and finishes with a run
    report whose confirmation vault records that the confirmation set was opened
    — because that is the fact the extension exists to respect.
    """
    root = Path(root)
    model = model or MockCalibrationModel()
    plan = mock_capture_plan(model)
    corpus_config = _mock_corpus_config()

    texts = mock_corpus_texts(MOCK_STREAM_TEXTS)
    records, partitions = collect_records_for_partition_quotas(
        corpus_id=MOCK_CORPUS_ID,
        texts=texts,
        min_chars=corpus_config["min_chars"],
        min_fit=int(baseline_scale),
        max_texts=MOCK_STREAM_TEXTS,
        n_validation=int(n_validation),
        n_confirmation=int(n_confirmation),
    )
    scales = (int(baseline_scale),)

    fit_records, dropped = filter_records_by_tokens(
        partitions.fit,
        token_count=model.tokenizer.token_count,
        skip_first=plan.skip_first,
        max_seq_len=plan.max_seq_len,
    )
    leakage = audit_leakage(partitions, scale_points=scales)
    nesting = scale_nesting_audit(partitions.fit, scales)
    manifest = corpus_manifest(
        partitions, corpus_config=corpus_config, scale_points=scales
    )

    # A small parent uses a small held-out set; the floors scale with it. These
    # are the *parent's* numbers, reproduced only so the fixture has the right
    # shape — the extension's own floors are the frozen ones in
    # :data:`jlens.calibration.extension.EXTENSION_GATE` and are never softened.
    parent_gate = replace(
        CALIBRATION_GATE, n_prompts=int(n_validation), min_distinct_target_tokens=8
    )
    _validation_prompts, validation_manifest = select_diverse_validation_prompts(
        [record.text for record in partitions.validation],
        n_prompts=int(n_validation),
        gate=parent_gate,
        seed=20260805,
        target_token_for_prompt=mock_target_token,
    )
    _confirmation_prompts, confirmation_manifest = select_diverse_validation_prompts(
        [record.text for record in partitions.confirmation],
        n_prompts=int(n_confirmation),
        gate=replace(parent_gate, n_prompts=int(n_confirmation)),
        seed=20260806,
        target_token_for_prompt=mock_target_token,
    )

    fingerprint = CalibrationFingerprint(
        mode="mock",
        protocol_version=MOCK_PARENT_PROTOCOL_VERSION,
        model_repo_id="google/gemma-4-E4B-it",
        model_revision="fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd",
        tokenizer_revision="fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd",
        capture_plan_digest=plan.digest,
        corpus_manifest_checksum=manifest["corpus_manifest_checksum"],
        gate_digest=CALIBRATION_GATE.digest,
        plateau_rule_digest=PLATEAU_RULE.digest,
        scale_points=scales,
        artifact_format_version="jlens.calibration.artifact.v1",
    )
    store = CalibrationStore(root, fingerprint)
    store.open()
    store.save(
        "corpus",
        "manifest",
        {
            "corpus": manifest,
            "leakage_audit": leakage,
            "scale_nesting_audit": nesting,
            "n_fit_records_after_token_filter": len(fit_records),
            "n_dropped_too_short": len(dropped),
            "validation_selection": validation_manifest,
            "confirmation_selection_checksum": confirmation_manifest[
                "selection_checksum"
            ],
        },
    )

    run_calibration(
        model,
        fit_records,
        plan=plan,
        scale_points=scales,
        store=store,
        checkpoint_every=4,
        diagnostics_every=4,
    )

    # The parent's own verdicts, recorded exactly as the completed study did.
    # Read for provenance only: this extension never re-scores them.
    store.save(
        "validation",
        f"scale{int(baseline_scale)}",
        {
            "scale": int(baseline_scale),
            "by_layer": {
                str(layer): {"layer": layer, "passed": layer in (35, 38, 40)}
                for layer in MOCK_LAYERS
            },
        },
    )
    store.save(
        "confirmation",
        f"scale{int(baseline_scale)}",
        {
            "scale": int(baseline_scale),
            "by_layer": {
                str(layer): {"layer": layer, "passed": layer in (35, 38, 40)}
                for layer in MOCK_LAYERS
            },
        },
    )

    report = {
        "schema": "jlens.calibration.report.v1",
        "mode": "mock",
        "protocol_version": MOCK_PARENT_PROTOCOL_VERSION,
        "fingerprint_digest": fingerprint.digest,
        "confirmation_vault": {
            "locked": not confirmation_opened,
            "opened": bool(confirmation_opened),
            "n_records": len(partitions.confirmation),
            "selected_scale": int(baseline_scale),
            "selection_checksum": "sha256:mock-parent-selection",
        },
    }
    artifacts = root / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "calibration_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return MockParent(
        root=str(root),
        model=model,
        plan=plan,
        records=tuple(records),
        partitions=partitions,
        fit_records=tuple(fit_records),
        baseline_scale=int(baseline_scale),
        last_stream_index=max(record.stream_index for record in records),
        fingerprint=fingerprint,
        corpus_config=corpus_config,
    )


def mock_extension_pool(parent: MockParent, *, min_chars: int = 100) -> list[CorpusRecord]:
    """Records the parent's collection never reached.

    The parent stops streaming the moment its quotas are met, so everything past
    that point is corpus the parent has not seen in any partition. That region —
    not a reshuffle of the parent's own records — is where the fresh evaluation
    sets come from.
    """
    texts = mock_corpus_texts(MOCK_STREAM_TEXTS)
    everything = build_records(MOCK_CORPUS_ID, texts, min_chars=min_chars)
    return [
        record
        for record in everything
        if record.stream_index > parent.last_stream_index
    ]


# ------------------------------------------------------------- the scenarios


@dataclass(frozen=True)
class LayerBehaviour:
    """A known statistical case for one layer, per stage.

    The knob is the exact optimistic rank the target is placed at, as in
    :class:`jlens.calibration.mock.LayerArchetype`: ``1`` is a unique argmax, and
    a tie block of ``t`` gives optimistic rank 1 with midrank ``(t + 1) / 2``.

    ``development`` is indexed by :data:`MOCK_DEVELOPMENT_SCALES` — the
    descriptive baseline first, then the two candidate scales.
    """

    development: tuple[int, ...]
    confirmation: int
    development_tie_block: tuple[int, ...] = (1, 1, 1)
    confirmation_tie_block: int = 1


@dataclass(frozen=True)
class ExtensionScenario:
    """One commissioned synthetic outcome, with what the pipeline must do."""

    key: str
    label: str
    expected_verdict: str
    expected_selected_scale: int | None
    expected_published_layers: tuple[int, ...]
    layers: dict[int, LayerBehaviour] = field(default_factory=dict)
    note: str = ""


def _passing(confirmation: int = 1) -> LayerBehaviour:
    return LayerBehaviour(development=(1, 1, 1), confirmation=confirmation)


def _failing() -> LayerBehaviour:
    return LayerBehaviour(development=(210, 200, 190), confirmation=195)


def _degenerate() -> LayerBehaviour:
    return LayerBehaviour(
        development=(1, 1, 1),
        confirmation=1,
        development_tie_block=(64, 64, 64),
        confirmation_tie_block=64,
    )


#: The established late layers behave the same way in every scenario: they pass.
#: They are here so the table is not three rows wide, and so a scenario cannot
#: accidentally test "everything fails" and call it a distinguishing result.
def _late_layers() -> dict[int, LayerBehaviour]:
    return {35: _passing(), 38: _passing(), 40: _passing()}


EXTENSION_SCENARIOS: dict[str, ExtensionScenario] = {
    "l32_late_pass": ExtensionScenario(
        key="l32_late_pass",
        label="L32 fails at 250, passes development at 1,000, passes confirmation",
        expected_verdict="EARLY_LAYER_CALIBRATION_GO",
        expected_selected_scale=MOCK_EXTENSION_SCALES[-1],
        expected_published_layers=(32,),
        note=(
            "the case the extension exists to detect: the earlier layer's "
            "failure was estimator variance, and it survives the untouched "
            "confirmation set. The plateau rule sees L32 still climbing, which "
            "is why the baseline scale is scored — with two points it could not"
        ),
        layers={
            8: _degenerate(),
            14: _failing(),
            20: _failing(),
            26: LayerBehaviour(development=(200, 150, 40), confirmation=40),
            32: LayerBehaviour(development=(200, 60, 1), confirmation=1),
            **_late_layers(),
        },
    ),
    "l32_confirmation_fail": ExtensionScenario(
        key="l32_confirmation_fail",
        label="L32 passes development at both scales and fails fresh confirmation",
        expected_verdict="EARLY_LAYER_CALIBRATION_NO_GO",
        expected_selected_scale=MOCK_EXTENSION_SCALES[0],
        expected_published_layers=(),
        note=(
            "the reason confirmation is untouched. The smaller scale is selected "
            "on development alone, and the layer is not published, not marked "
            "validated, and not retried at another scale"
        ),
        layers={
            8: _degenerate(),
            14: _failing(),
            20: _failing(),
            26: LayerBehaviour(development=(200, 150, 60), confirmation=60),
            32: LayerBehaviour(development=(60, 1, 1), confirmation=60),
            **_late_layers(),
        },
    ),
    "no_early_layer": ExtensionScenario(
        key="no_early_layer",
        label="L26 and L32 both fail at 1,000 — the honest negative",
        expected_verdict="EARLY_LAYER_CALIBRATION_NO_GO",
        expected_selected_scale=MOCK_EXTENSION_SCALES[-1],
        expected_published_layers=(),
        note=(
            "no earlier layer passes development at any candidate scale, so the "
            "predeclared rule still takes the largest scale to one honest "
            "confirmation and reports what it says"
        ),
        layers={
            8: _degenerate(),
            14: _failing(),
            20: _failing(),
            26: LayerBehaviour(development=(200, 150, 60), confirmation=60),
            32: LayerBehaviour(development=(150, 60, 40), confirmation=40),
            **_late_layers(),
        },
    ),
}


def _place_at_rank(scores: torch.Tensor, target: int, rank: int) -> None:
    """Set ``scores[target]`` so exactly ``rank - 1`` other tokens beat it."""
    others = torch.cat([scores[:target], scores[target + 1 :]])
    ordered = torch.sort(others, descending=True).values
    rank = max(1, min(int(rank), int(ordered.numel())))
    if rank == 1:
        scores[target] = float(ordered[0]) + 1.0
    else:
        scores[target] = (float(ordered[rank - 2]) + float(ordered[rank - 1])) / 2.0


def _scores(
    *,
    layer: int,
    stage: str,
    stage_index: int,
    prompt_index: int,
    variant: str,
    target: int,
    rank: int,
    tie_block: int,
    vocab: int,
    n_distinct_targets: int,
) -> torch.Tensor:
    seed = int(
        hashlib.sha256(
            f"ext|{layer}|{stage}|{stage_index}|{prompt_index}|{variant}".encode()
        ).hexdigest()[:8],
        16,
    )
    generator = torch.Generator().manual_seed(seed)
    scores = torch.randn(int(vocab), generator=generator)
    if variant != "j_lens":
        rank, tie_block = CONTROL_RANK[variant], 1
    _place_at_rank(scores, target, rank)
    if tie_block > 1:
        offset = int(n_distinct_targets) + 1
        peak = float(scores.max()) + 1.0
        partners = {
            offset + (prompt_index * 7 + step * 13) % (int(vocab) - offset)
            for step in range(int(tie_block) - 1)
        }
        partners.discard(target)
        for index in list(partners)[: int(tie_block) - 1]:
            scores[index] = peak
        scores[target] = peak
    return scores


def mock_extension_rows(
    scenario: ExtensionScenario | str,
    *,
    stage: str,
    scale: int,
    scale_index: int = 0,
    n_prompts: int,
    layers: Sequence[int] = MOCK_LAYERS,
    n_distinct_targets: int = MOCK_TARGET_MODULUS,
    vocab: int = MOCK_VOCAB,
    variants: Sequence[str] = (
        "j_lens",
        "permuted",
        "random",
        "wrong_layer",
        "logit_lens",
    ),
) -> list[dict]:
    """Rows for one stage at one scale, scored by the **real** tie-aware scorer.

    Args:
        stage: ``"development"`` or ``"confirmation"``. The stage is part of the
            row seed, so a confirmation row is never a development row wearing a
            different label.
        scale_index: Which development scale point this is. Ignored for
            confirmation, which has a single behaviour per layer.
    """
    if isinstance(scenario, str):
        scenario = EXTENSION_SCENARIOS[scenario]
    if stage not in ("development", "confirmation"):
        raise ValueError(f"unknown stage {stage!r}; expected development/confirmation")

    rows: list[dict] = []
    for layer in layers:
        behaviour = scenario.layers[int(layer)]
        if stage == "development":
            rank = behaviour.development[int(scale_index)]
            tie_block = behaviour.development_tie_block[int(scale_index)]
        else:
            rank = behaviour.confirmation
            tie_block = behaviour.confirmation_tie_block
        for prompt_index in range(int(n_prompts)):
            target = prompt_index % int(n_distinct_targets)
            actual = torch.full((int(vocab),), -10.0)
            actual[target] = 10.0
            prompt_sha = hashlib.sha256(
                f"ext|{scenario.key}|{stage}|{scale}|{prompt_index}".encode()
            ).hexdigest()
            for variant in variants:
                rows.append(
                    tie_aware_row(
                        sample_index=prompt_index,
                        prompt_sha=prompt_sha,
                        layer=int(layer),
                        variant=variant,
                        variant_logits=_scores(
                            layer=int(layer),
                            stage=stage,
                            stage_index=int(scale_index),
                            prompt_index=prompt_index,
                            variant=variant,
                            target=target,
                            rank=rank,
                            tie_block=tie_block,
                            vocab=int(vocab),
                            n_distinct_targets=int(n_distinct_targets),
                        ),
                        actual_logits=actual,
                    )
                )
    return rows


# ------------------------------------------------- the two refusal scenarios


def plant_old_confirmation_record(
    splits: FreshEvaluationSplits, old_confirmation: Sequence[CorpusRecord]
) -> FreshEvaluationSplits:
    """A fresh split with an old confirmation record smuggled into it.

    The builder in :mod:`jlens.calibration.extension` excludes such a record
    before bucketing, so this constructs the state the builder prevents in order
    to prove the independent cross-split audit *also* refuses it. A guard that
    is only ever exercised by the thing that makes it unnecessary is not a guard.
    """
    if not old_confirmation:
        raise ValueError("no old confirmation records to plant")
    contaminated = (*splits.confirmation[:-1], old_confirmation[0])
    return FreshEvaluationSplits(
        development=splits.development,
        confirmation=contaminated,
        corpus_id=splits.corpus_id,
        seed=splits.seed,
        n_pool=splits.n_pool,
        excluded_exact=dict(splits.excluded_exact),
        excluded_near=dict(splits.excluded_near),
        excluded_pool_duplicates=splits.excluded_pool_duplicates,
        n_near_comparisons=splits.n_near_comparisons,
    )


def corrupt_continued_checkpoint(
    path: str | os.PathLike[str], *, layer: int | None = None, delta: float = 1e-3
) -> dict:
    """Perturb a continued accumulator so it no longer equals a fresh fit.

    Used to prove the equivalence check is load-bearing. A continuation that
    silently differs from the fit its prompt count claims is the one failure
    mode a checksum on the prompt list cannot catch, so it gets its own test
    with its own deliberate corruption.
    """
    path = Path(path)
    state = torch.load(str(path), map_location="cpu", weights_only=True)
    target = int(layer) if layer is not None else sorted(state["jacobian_sum"])[0]
    state["jacobian_sum"][target] = state["jacobian_sum"][target] + float(delta)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    torch.save(state, temporary)
    os.replace(temporary, path)
    return {"corrupted_layer": target, "delta": float(delta), "path": str(path)}


def mock_extension_environment() -> dict:
    """The environment block recorded in MOCK artifacts."""
    return {
        "python": "mock",
        "torch": torch.__version__,
        "transformers": "not imported (MOCK)",
        "cuda_available": False,
        "mock": True,
    }


def mock_load_info() -> dict:
    """Model identity for a MOCK run, shaped like the real loader's output."""
    return {
        "model_repo_id": "google/gemma-4-E4B-it",
        "model_revision": "fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd",
        "tokenizer_repo_id": "google/gemma-4-E4B-it",
        "tokenizer_revision": "fa62d88df2e6df5efa9d26ad6b3beaea2765f0cd",
        "dtype": "float32",
        "device_map": "cpu",
    }


def mock_extension_gate_sizes() -> Mapping[str, int]:
    """The sample sizes the MOCK run evaluates at — the real ones."""
    return {
        "development": EXTENSION_GATE.n_prompts,
        "confirmation": EXTENSION_GATE.n_prompts,
    }
