# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Stage functions the notebook calls, so the notebook holds no untested logic.

Each stage is resumable in the same way: build the unit key, skip if the store
already holds a checksum-valid unit, otherwise compute and save immediately.
Nothing batches work it cannot afford to lose.

The stages are model-agnostic — they take a
:class:`~jlens.mmpilot.backend.PilotBackend` and a media loader, so the MOCK
end-to-end test and the real Colab pilot run the *same* code.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field

from jlens.mmpilot.backend import ModalityUnsupportedError, PilotBackend
from jlens.mmpilot.capability import (
    balanced_capability_record,
    build_ordered_questions,
    build_prompt,
    build_question,
    candidate_token_ids,
    capability_record,
    capability_summary,
    score_candidate_sequences,
)
from jlens.mmpilot.causal import (
    DEFAULT_ALPHAS,
    condition_key,
    estimate_concept_direction,
    random_control_direction,
    raw_residual_direction,
    run_condition,
    summarize_interventions,
    unit_direction_tensor,
)
from jlens.mmpilot.jspace import (
    activation_record,
    activation_tensor,
    build_dictionary,
    capture_final_prompt_activations,
    code_map,
    jspace_code,
    representational_report,
)
from jlens.mmpilot.reconstruction import (
    ReconstructionControlConfig,
    reconstruction_control_record,
    summarize_reconstruction_controls,
)
from jlens.mmpilot.report import code_statistics, gonogo_report
from jlens.mmpilot.store import UnitStore, safe_key
from jlens.pursuit import PursuitSettings

#: ``(load_image(path), load_audio(path) -> (data, sampling_rate))``
MediaLoader = Mapping[str, Callable]


@dataclass
class PilotConfig:
    """Everything the pilot needs that is not the data or the model."""

    mode: str = "mock"
    layers: tuple[int, ...] = (35, 38)
    causal_layers: tuple[int, ...] = (38,)
    modalities: tuple[str, ...] = ("text", "image", "spoken_audio")
    concepts: tuple[str, ...] = ()
    causal_concepts: tuple[str, ...] = ()
    capability_threshold: float = 0.7
    pursuit_k: int = 25
    pursuit_refine_steps: int = 2
    pursuit_correlation_chunk_size: int | None = 65536
    direction_top_k: int | None = 16
    alphas: tuple[float, ...] = DEFAULT_ALPHAS
    n_target_examples: int = 2
    n_permutations: int = 50
    seed: int = 20260731
    max_capability_groups_per_concept: int = 8

    def pursuit_settings(self) -> PursuitSettings:
        return PursuitSettings(
            k=self.pursuit_k,
            normalize_atoms=True,
            refine_steps=self.pursuit_refine_steps,
            tol_relative_residual=0.0,
            correlation_chunk_size=self.pursuit_correlation_chunk_size,
        )

    def to_dict(self) -> dict:
        payload = asdict(self)
        for key in ("layers", "causal_layers", "modalities", "concepts",
                    "causal_concepts", "alphas"):
            payload[key] = list(payload[key])
        return payload


@dataclass
class StageOutcome:
    """What a stage did, so the notebook can print resume behaviour honestly."""

    computed: int = 0
    reused: int = 0
    skipped: list[str] = field(default_factory=list)
    records: list[dict] = field(default_factory=list)

    def line(self, name: str) -> str:
        return (
            f"{name}: {self.computed} computed, {self.reused} reused"
            + (f", {len(self.skipped)} skipped" if self.skipped else "")
        )


def sample_id(group_id: str, modality: str) -> str:
    return f"{group_id}:{modality}"


def derived_seed(*parts: object) -> int:
    """A reproducible 31-bit seed from ``parts``.

    Python's ``hash`` is salted per process, so a resumed run would draw
    different "random" controls than the run it is resuming. This does not.
    """
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode()).digest()
    return int.from_bytes(digest[:4], "big") % (2**31 - 1)


def _build_inputs_for(
    backend: PilotBackend,
    group: Mapping,
    modality: str,
    question: str,
    media: MediaLoader,
):
    prompt = build_prompt(question, modality=modality, caption=group["caption"])
    if modality == "text":
        return backend.build_inputs(prompt=prompt, modality=modality)
    if modality == "image":
        return backend.build_inputs(
            prompt=prompt,
            modality=modality,
            image=media["load_image"](group["image_path"]),
            media_path=group["image_path"],
        )
    data, sampling_rate = media["load_audio"](group["audio_path"])
    return backend.build_inputs(
        prompt=prompt,
        modality=modality,
        audio=data,
        sampling_rate=sampling_rate,
        media_path=group["audio_path"],
    )


def available_modalities(backend: PilotBackend, config: PilotConfig) -> tuple[list[str], list[str]]:
    """``(available, blocked)`` — blocked modalities are reported, never faked."""
    available = [m for m in config.modalities if backend.supports(m)]
    blocked = [m for m in config.modalities if not backend.supports(m)]
    return available, blocked


# ------------------------------------------------------------------ stage 3


def stage_capability(
    backend: PilotBackend,
    store: UnitStore,
    subset: Mapping,
    config: PilotConfig,
    media: MediaLoader,
    *,
    modalities: Sequence[str],
) -> tuple[StageOutcome, dict]:
    """Score every positive group in every available modality."""
    concepts = list(config.concepts) or sorted(
        {g["concept"] for g in subset["splits"]["train"] if g["concept"]}
    )
    questions = build_ordered_questions(concepts)
    candidates = candidate_token_ids(backend, concepts)
    groups = [
        group
        for split in ("train", "test")
        for group in subset["splits"][split]
        if group["concept"] in concepts
    ]
    per_concept: dict[str, int] = {}
    outcome = StageOutcome()
    for group in groups:
        seen = per_concept.get(group["concept"], 0)
        if seen >= config.max_capability_groups_per_concept:
            continue
        per_concept[group["concept"]] = seen + 1
        for modality in modalities:
            identifier = sample_id(group["group_id"], modality)
            key = safe_key(identifier)
            cached = store.load("capability", key)
            if cached is not None:
                outcome.reused += 1
                outcome.records.append(cached)
                continue
            ordered_records = []
            try:
                for question in questions:
                    inputs = _build_inputs_for(backend, group, modality, question, media)
                    ordered_records.append(
                        capability_record(
                            backend,
                            inputs,
                            sample_id=identifier,
                            concept=group["concept"],
                            group_id=group["group_id"],
                            candidate_ids=candidates,
                        )
                    )
            except ModalityUnsupportedError:
                outcome.skipped.append(identifier)
                continue
            record = balanced_capability_record(
                ordered_records, concept=group["concept"]
            )
            record["split"] = group["split"]
            record["image_id"] = group["image_id"]
            store.save("capability", key, record)
            outcome.computed += 1
            outcome.records.append(record)

    summary = capability_summary(
        outcome.records, threshold=config.capability_threshold, modalities=modalities
    )
    summary["question"] = questions[0]
    summary["questions"] = questions
    summary["candidate_token_ids"] = {k: list(v) for k, v in candidates.items()}
    store.save("metric", "capability_summary", summary)
    return outcome, summary


# ------------------------------------------------------------------ stage 5


def stage_activations(
    backend: PilotBackend,
    store: UnitStore,
    subset: Mapping,
    config: PilotConfig,
    media: MediaLoader,
    *,
    modalities: Sequence[str],
    retained_concepts: Sequence[str],
    model_revision: str,
) -> StageOutcome:
    """Capture the final-prompt-token residual for retained concepts and
    matched negatives, one atomic unit per (sample, modality, layer)."""
    question = build_question(list(config.concepts) or list(retained_concepts))
    outcome = StageOutcome()
    for split in ("train", "test"):
        for group in subset["splits"][split]:
            if group["concept"] is not None and group["concept"] not in retained_concepts:
                continue
            for modality in modalities:
                identifier = sample_id(group["group_id"], modality)
                keys = {layer: safe_key(identifier, f"L{layer}") for layer in config.layers}
                cached = {
                    layer: store.load("activation", key) for layer, key in keys.items()
                }
                if all(value is not None for value in cached.values()):
                    outcome.reused += len(cached)
                    outcome.records.extend(cached.values())
                    continue
                try:
                    inputs = _build_inputs_for(backend, group, modality, question, media)
                except ModalityUnsupportedError:
                    outcome.skipped.append(identifier)
                    continue
                activations = capture_final_prompt_activations(
                    backend, inputs, list(config.layers)
                )
                for layer, activation in activations.items():
                    record = activation_record(
                        activation,
                        sample_id=identifier,
                        group_id=group["group_id"],
                        image_id=group["image_id"],
                        concept=group["concept"],
                        modality=modality,
                        split=group["split"],
                        layer=layer,
                        inputs=inputs,
                        model_revision=model_revision,
                    )
                    store.save("activation", keys[layer], record)
                    outcome.computed += 1
                    outcome.records.append(record)
    return outcome


# ------------------------------------------------------------------ stage 6


def build_dictionaries(lens, layers: Sequence[int], backend: PilotBackend, **kwargs) -> dict:
    """One frozen dictionary per layer. Callers should ``del`` each before
    building the next on a memory-tight GPU."""
    weight = backend.unembedding_weight()
    return {int(layer): build_dictionary(lens, int(layer), weight, **kwargs) for layer in layers}


def stage_codes(
    store: UnitStore,
    activations: Sequence[Mapping],
    dictionaries: Mapping[int, object],
    config: PilotConfig,
    *,
    lens_checksum: str,
) -> StageOutcome:
    """Sparse nonnegative J-space coordinates for every stored activation."""
    settings = config.pursuit_settings()
    outcome = StageOutcome()
    for record in activations:
        layer = int(record["layer"])
        if layer not in dictionaries:
            continue
        key = safe_key(record["sample_id"], f"L{layer}")
        cached = store.load("jspace", key)
        if cached is not None:
            outcome.reused += 1
            outcome.records.append(cached)
            continue
        code = jspace_code(
            activation_tensor(record),
            dictionaries[layer],
            settings,
            activation_checksum=record["activation_checksum"],
            lens_checksum=lens_checksum,
        )
        code.update(
            {
                "sample_id": record["sample_id"],
                "group_id": record["group_id"],
                "image_id": record["image_id"],
                "concept": record["concept"],
                "modality": record["modality"],
                "split": record["split"],
            }
        )
        store.save("jspace", key, code)
        outcome.computed += 1
        outcome.records.append(code)
    return outcome


# --------------------------------------------------- stage 6b: random control


def stage_reconstruction_control(
    store: UnitStore,
    activations: Sequence[Mapping],
    dictionaries: Mapping[int, object],
    config: PilotConfig,
    *,
    lens_checksum: str,
    control_config: ReconstructionControlConfig | None = None,
    primary_layer: int | None = None,
) -> tuple[StageOutcome, dict]:
    """Compare the frozen lens against matched random directions.

    Held-out **text** activations only: the lens was calibrated on text, so
    text is where "does this basis beat noise" is a fair question, and keeping
    the diagnostic small is deliberate.

    This replaces the pilot's old absolute reconstruction gate. Explaining a
    small share of an activation is expected of a sparse workspace — the
    published J-space result puts the median around 6-7% — so the criterion is
    excess over a matched control, not an absolute level.
    """
    control_config = control_config or ReconstructionControlConfig()
    settings = config.pursuit_settings()
    outcome = StageOutcome()
    per_layer: dict[int, int] = {}
    candidates = sorted(
        (
            record
            for record in activations
            if record["modality"] == "text"
            and record["split"] == "test"
            and int(record["layer"]) in dictionaries
        ),
        key=lambda record: (int(record["layer"]), record["sample_id"]),
    )
    for record in candidates:
        layer = int(record["layer"])
        seen = per_layer.get(layer, 0)
        if seen >= control_config.max_samples_per_layer:
            continue
        per_layer[layer] = seen + 1
        key = safe_key(record["sample_id"], f"L{layer}", "control")
        cached = store.load("reconstruction_control", key)
        if cached is not None:
            outcome.reused += 1
            outcome.records.append(cached)
            continue
        computed = reconstruction_control_record(
            activation_tensor(record),
            dictionaries[layer],
            settings,
            config=control_config,
            sample_id=record["sample_id"],
            layer=layer,
            modality=record["modality"],
            split=record["split"],
            activation_checksum=record["activation_checksum"],
            lens_checksum=lens_checksum,
        )
        store.save("reconstruction_control", key, computed)
        outcome.computed += 1
        outcome.records.append(computed)

    summary = summarize_reconstruction_controls(
        outcome.records,
        config=control_config,
        primary_layer=primary_layer if primary_layer is not None else config.layers[-1],
    )
    store.save("metric", "reconstruction_control", summary)
    return outcome, summary


# ------------------------------------------------------------------ stage 7


def stage_representational(
    store: UnitStore,
    activations: Sequence[Mapping],
    codes: Sequence[Mapping],
    config: PilotConfig,
    *,
    layer: int,
    modalities: Sequence[str],
) -> dict:
    """Retrieval, separation, support overlap, raw baseline, shuffled control."""
    by_sample = {
        record["sample_id"]: record
        for record in activations
        if int(record["layer"]) == layer
    }
    samples = []
    for code in codes:
        if int(code["layer"]) != layer:
            continue
        activation = by_sample.get(code["sample_id"])
        if activation is None:
            continue
        samples.append(
            {
                "sample_id": code["sample_id"],
                "group_id": code["group_id"],
                "image_id": code["image_id"],
                "concept": code["concept"],
                "modality": code["modality"],
                "code": code_map(code),
                "activation": activation["activation"],
            }
        )
    report = representational_report(
        samples,
        modalities=modalities,
        n_permutations=config.n_permutations,
        seed=config.seed,
    )
    report["layer"] = layer
    store.save("metric", safe_key("representational", f"L{layer}"), report)
    return report


# ------------------------------------------------------------------ stage 8


def stage_directions(
    store: UnitStore,
    codes: Sequence[Mapping],
    activations: Sequence[Mapping],
    dictionaries: Mapping[int, object],
    config: PilotConfig,
    *,
    concepts: Sequence[str],
    modalities: Sequence[str],
    lens_checksum: str,
) -> tuple[StageOutcome, dict]:
    """Estimate every source-derived direction from *training* examples only."""
    outcome = StageOutcome()
    directions: dict[tuple, dict] = {}
    activation_index = {
        (record["sample_id"], int(record["layer"])): record for record in activations
    }
    for layer in config.causal_layers:
        if layer not in dictionaries:
            continue
        for source_modality in modalities:
            train = [
                code
                for code in codes
                if int(code["layer"]) == layer
                and code["modality"] == source_modality
                and code["split"] == "train"
            ]
            negatives = [code for code in train if not code["concept"]]
            if not negatives:
                continue
            for concept in concepts:
                positives = [code for code in train if code["concept"] == concept]
                if not positives:
                    continue
                for kind in ("source_concept", "raw_residual_difference"):
                    key = safe_key(concept, source_modality, f"L{layer}", kind)
                    cached = store.load("direction", key)
                    if cached is not None:
                        outcome.reused += 1
                        directions[(concept, source_modality, layer, kind)] = cached
                        continue
                    if kind == "source_concept":
                        record = estimate_concept_direction(
                            [code_map(code) for code in positives],
                            [code_map(code) for code in negatives],
                            dictionaries[layer],
                            concept=concept,
                            source_modality=source_modality,
                            layer=layer,
                            positive_ids=[code["sample_id"] for code in positives],
                            negative_ids=[code["sample_id"] for code in negatives],
                            lens_checksum=lens_checksum,
                            top_k=config.direction_top_k,
                        )
                    else:
                        record = raw_residual_direction(
                            [
                                activation_index[(code["sample_id"], layer)]["activation"]
                                for code in positives
                            ],
                            [
                                activation_index[(code["sample_id"], layer)]["activation"]
                                for code in negatives
                            ],
                            concept=concept,
                            source_modality=source_modality,
                            layer=layer,
                        )
                    store.save("direction", key, record)
                    outcome.computed += 1
                    directions[(concept, source_modality, layer, kind)] = record
    outcome.records = list(directions.values())
    return outcome, directions


# ------------------------------------------------------------------ stage 9


def stage_causal(
    backend: PilotBackend,
    store: UnitStore,
    subset: Mapping,
    codes: Sequence[Mapping],
    activations: Sequence[Mapping],
    directions: Mapping,
    config: PilotConfig,
    media: MediaLoader,
    *,
    concepts: Sequence[str],
    modalities: Sequence[str],
    all_concepts: Sequence[str],
) -> tuple[StageOutcome, dict]:
    """The source x target transfer matrix, with all four controls."""
    question = build_question(list(config.concepts) or list(all_concepts))
    candidates = candidate_token_ids(backend, list(config.concepts) or list(all_concepts))
    groups = {group["group_id"]: group for group in subset["splits"]["test"]}
    outcome = StageOutcome()

    norm_samples: dict[tuple[str, int], list[float]] = {}
    for record in activations:
        norm_samples.setdefault(
            (record["modality"], int(record["layer"])), []
        ).append(float(record["norm"]))
    reference_norms = {
        key: sum(values) / len(values) for key, values in norm_samples.items()
    }

    for layer in config.causal_layers:
        for concept in concepts:
            for source_modality in modalities:
                base = directions.get((concept, source_modality, layer, "source_concept"))
                if base is None:
                    continue
                # Prefer a control concept outside the focal causal set. In a
                # two-way forced choice, the only alternative is not actually
                # unrelated: it is the target's direct contrast.
                unrelated_concept = next(
                    (other for other in all_concepts if other not in concepts),
                    next((other for other in all_concepts if other != concept), None),
                )
                variants: dict[str, dict] = {"source_concept": base, "zero": base}
                raw = directions.get(
                    (concept, source_modality, layer, "raw_residual_difference")
                )
                if raw:
                    variants["raw_residual_difference"] = raw
                if unrelated_concept:
                    other = directions.get(
                        (unrelated_concept, source_modality, layer, "source_concept")
                    )
                    if other:
                        variants["unrelated_concept"] = other
                for target_modality in modalities:
                    targets = _select_targets(
                        codes, groups, concept, target_modality, config.n_target_examples
                    )
                    for group, sign in targets:
                        identifier = sample_id(group["group_id"], target_modality)
                        try:
                            inputs = _build_inputs_for(
                                backend, group, target_modality, question, media
                            )
                        except ModalityUnsupportedError:
                            outcome.skipped.append(identifier)
                            continue
                        clean = score_candidate_sequences(backend, inputs, candidates)
                        reference = reference_norms.get((target_modality, layer), 1.0)
                        # A fresh random draw per target sample: one fixed draw
                        # would carry whatever accidental alignment it happens
                        # to have with the concept into every cell, which is
                        # exactly the bias the control is meant to remove.
                        conditions = dict(variants)
                        conditions["random_norm_matched"] = random_control_direction(
                            backend.d_model,
                            seed=derived_seed(
                                config.seed,
                                concept,
                                source_modality,
                                target_modality,
                                layer,
                                identifier,
                            ),
                            matched_to=f"{concept}|{source_modality}|L{layer}|{identifier}",
                        )
                        for control_kind, direction in conditions.items():
                            alphas = (0.0,) if control_kind == "zero" else tuple(
                                a for a in config.alphas if a > 0
                            )
                            for alpha in alphas:
                                key = condition_key(
                                    concept=concept,
                                    source_modality=source_modality,
                                    target_modality=target_modality,
                                    layer=layer,
                                    control_kind=control_kind,
                                    alpha=alpha,
                                    sample_id=identifier,
                                )
                                cached = store.load("intervention", key)
                                if cached is not None:
                                    outcome.reused += 1
                                    outcome.records.append(cached)
                                    continue
                                record = run_condition(
                                    backend,
                                    inputs,
                                    layer=layer,
                                    unit_direction=unit_direction_tensor(direction),
                                    alpha=alpha,
                                    reference_norm=reference,
                                    sign=sign,
                                    candidate_ids=candidates,
                                    target_concept=concept,
                                    clean_scores=clean,
                                )
                                record.update(
                                    {
                                        "concept": concept,
                                        "source_modality": source_modality,
                                        "target_modality": target_modality,
                                        "control_kind": control_kind,
                                        "sample_id": identifier,
                                        "group_id": group["group_id"],
                                        # The independent statistical unit. Two
                                        # synchronized groups can carry the same
                                        # photograph, so a group-level count of
                                        # image targets over-reports n.
                                        "image_id": group["image_id"],
                                        "target_is_positive": sign < 0,
                                        "direction_checksum": direction["direction_checksum"],
                                    }
                                )
                                store.save("intervention", key, record)
                                outcome.computed += 1
                                outcome.records.append(record)

    summary = summarize_interventions(outcome.records)
    store.save("metric", "intervention_summary", summary)
    return outcome, summary


def _select_targets(
    codes: Sequence[Mapping],
    groups: Mapping[str, Mapping],
    concept: str,
    modality: str,
    n_each: int,
) -> list[tuple[Mapping, int]]:
    """Held-out positives (subtract, ``sign=-1``) and matched negatives (add, ``+1``)."""
    positives: list[tuple[Mapping, int]] = []
    negatives: list[tuple[Mapping, int]] = []
    seen: set[str] = set()
    for code in sorted(codes, key=lambda c: c["sample_id"]):
        if code["split"] != "test" or code["modality"] != modality:
            continue
        group = groups.get(code["group_id"])
        if group is None or group["group_id"] in seen:
            continue
        if code["concept"] == concept and len(positives) < n_each:
            positives.append((group, -1))
            seen.add(group["group_id"])
        elif code["concept"] is None and len(negatives) < n_each:
            negatives.append((group, +1))
            seen.add(group["group_id"])
    return positives + negatives


# ----------------------------------------------------------------- stage 10


def stage_report(
    store: UnitStore,
    *,
    config: PilotConfig,
    capability: Mapping,
    lens_validation: Mapping | None,
    codes: Sequence[Mapping],
    representational: Mapping,
    interventions: Mapping,
    invariance: Mapping | None,
    blocked_modalities: Sequence[str],
    reconstruction_control: Mapping | None = None,
    manifest_audit: Mapping | None = None,
) -> tuple[str, dict]:
    """Write ``report.md`` and ``summary.json`` into the run directory."""
    stats = code_statistics(codes)
    markdown, summary = gonogo_report(
        mode=config.mode,
        run_dir=str(store.root),
        capability=capability,
        lens_validation=lens_validation,
        code_stats=stats,
        representational=representational,
        interventions=interventions,
        invariance=invariance,
        reconstruction_control=reconstruction_control,
        blocked_modalities=blocked_modalities,
        manifest_audit=manifest_audit,
    )
    summary["config"] = config.to_dict()
    summary["resume"] = store.status_report()
    store.save("metric", "gonogo", summary)
    (store.root / "report.md").write_text(markdown, encoding="utf-8")
    (store.root / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    return markdown, summary


__all__ = [
    "MediaLoader",
    "PilotConfig",
    "StageOutcome",
    "available_modalities",
    "build_dictionaries",
    "sample_id",
    "stage_activations",
    "stage_capability",
    "stage_causal",
    "stage_codes",
    "stage_reconstruction_control",
    "stage_directions",
    "stage_report",
    "stage_representational",
]
