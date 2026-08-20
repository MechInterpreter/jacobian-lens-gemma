# SPDX-License-Identifier: Apache-2.0
"""Modality-matched Jacobian-lens fitting and cross evaluation.

The upstream fitter accepts strings and tokenizes them with the language-model
tokenizer.  That is exactly right for a text-only J-lens and exactly wrong for
the experiment in this module: image and spoken-audio examples must pass
through the checkpoint's real processor and modality towers before the
decoder Jacobian is measured.

This module keeps the upstream estimator unchanged.  It only replaces the
``model.encode(prompt)`` boundary with a :class:`~jlens.mmpilot.backend.BuiltInputs`
object produced by the audited multimodal backend.  For every valid decoder
position it estimates the same average ``d h_final / d h_l`` and stores the
same fp32 ``[d_model, d_model]`` matrices.

Four equal-size arms are supported by design: text, image, spoken audio, and a
pooled arm whose photographs are assigned as evenly as possible across the
three modalities.  Each arm has its own fingerprint-bound checkpoint.  A
changed population, order, media checksum, processor protocol, layer grid, or
estimator configuration refuses resume instead of mixing accumulators.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from jlens.fitting import valid_position_mask
from jlens.hooks import ActivationRecorder
from jlens.lens import JacobianLens
from jlens.metadata import file_sha256
from jlens.mmpilot.coordinate_swap import (
    ConceptToken,
    SwapBasis,
    build_swap_basis_from_vectors,
    coordinate_swap_band,
)
from jlens.mmpilot.store import payload_checksum

MODALITIES = ("text", "image", "spoken_audio")
LENS_ARMS = ("text", "image", "spoken_audio", "pooled")
ESTIMATOR_VERSION = "mmpilot.multimodal_average_jacobian.v1"
PLAN_VERSION = "mmpilot.matched_multimodal_lens_plan.v1"
CROSS_EVAL_VERSION = "mmpilot.multimodal_lens_cross_eval.v1"
PRIMARY_POSITION_RULE = "all_prompt_positions"
ANSWER_EQUIVALENCE_VERSION = (
    "mmpilot.open_answer_equivalence.casefold_whitespace.v1"
)
CAUSAL_SOURCE_VERSION = "mmpilot.matched_multimodal_causal_source.v1"


class MultimodalLensRefused(RuntimeError):
    """The requested run would mix or mislabel scientific inputs."""


def normalize_open_answer_surface(value: str) -> str:
    """Normalize only tokenizer/case aliases declared before causal sampling.

    This deliberately does not remove punctuation, singularize words, or map
    species names onto a parent category.  It licenses only Unicode
    normalization, case folding, and whitespace equivalence, which are
    properties of surface realization rather than semantic relabeling.
    """

    normalized = unicodedata.normalize("NFKC", str(value)).casefold()
    return " ".join(normalized.split())


def open_answer_matches(observed: str, expected: str) -> bool:
    """Whether two one-token answer surfaces are equivalent under v1."""

    return normalize_open_answer_surface(observed) == normalize_open_answer_surface(
        expected
    )


def answer_equivalence_record() -> dict:
    """Machine-readable boundary for the prospective causal follow-up."""

    payload = {
        "version": ANSWER_EQUIVALENCE_VERSION,
        "unicode_normalization": "NFKC",
        "case_sensitive": False,
        "whitespace_rule": "strip_and_collapse",
        "punctuation_removed": False,
        "semantic_aliases": [],
        "plural_or_taxonomy_mapping": False,
    }
    return {**payload, "protocol_digest": payload_checksum(payload)}


def _verified_payload(path: Path, *, expected_checksum: str, label: str) -> dict:
    if not path.is_file():
        raise MultimodalLensRefused(f"missing {label}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MultimodalLensRefused(f"could not read {label}: {path}") from exc
    recorded = str(payload.get("report_checksum") or "")
    body = {key: value for key, value in payload.items() if key != "report_checksum"}
    recomputed = payload_checksum(body)
    if recorded != expected_checksum or recomputed != expected_checksum:
        raise MultimodalLensRefused(
            f"{label} checksum mismatch: recorded={recorded!r}, "
            f"recomputed={recomputed!r}, expected={expected_checksum!r}"
        )
    return payload


def load_completed_causal_source(
    run_dir: str | Path,
    *,
    expected_final_report_checksum: str,
    expected_cross_report_checksum: str,
    expected_causal_report_checksum: str,
    expected_lens_checksums: Mapping[str, str],
) -> dict:
    """Verify the completed four-lens run reused by a fresh causal follow-up.

    The old clean-screen photographs are harvested from the completed causal
    report and become mandatory exclusions.  Lens files are read in place and
    checksum-pinned; they are never refitted or copied into the new run.
    """

    root = Path(run_dir)
    final = _verified_payload(
        root / "matched_multimodal_jlens_report.json",
        expected_checksum=expected_final_report_checksum,
        label="completed matched-multimodal report",
    )
    causal = _verified_payload(
        root / "multimodal_lens_causal_comparison_report.json",
        expected_checksum=expected_causal_report_checksum,
        label="completed causal report",
    )
    cross_path = root / "multimodal_lens_cross_eval_report.json"
    _cross = _verified_payload(
        cross_path,
        expected_checksum=expected_cross_report_checksum,
        label="completed cross-evaluation report",
    )
    if causal.get("verdict") != "CAPABILITY_NO_GO":
        raise MultimodalLensRefused(
            "the pinned source causal report is not the completed "
            "CAPABILITY_NO_GO run"
        )
    if (final.get("causal_comparison") or {}).get("report_checksum") != (
        expected_causal_report_checksum
    ):
        raise MultimodalLensRefused(
            "the final report does not embed the pinned causal report"
        )
    if (final.get("cross_evaluation") or {}).get("report_checksum") != (
        expected_cross_report_checksum
    ):
        raise MultimodalLensRefused(
            "the final report does not embed the pinned cross-evaluation report"
        )

    recorded_checksums = dict(final.get("lens_checksums") or {})
    if set(recorded_checksums) != set(LENS_ARMS):
        raise MultimodalLensRefused(
            f"the source report does not record all four lens arms: "
            f"{sorted(recorded_checksums)}"
        )
    lens_paths: dict[str, str] = {}
    for arm in LENS_ARMS:
        expected = str(expected_lens_checksums.get(arm) or "")
        recorded = str(recorded_checksums.get(arm) or "")
        path = root / "lenses" / f"lens.{arm}.pt"
        observed = file_sha256(str(path)) if path.is_file() else "missing"
        if not expected or recorded != expected or observed != expected:
            raise MultimodalLensRefused(
                f"source lens {arm!r} is not checksum-pinned: "
                f"recorded={recorded!r}, observed={observed!r}, "
                f"expected={expected!r}"
            )
        lens_paths[arm] = str(path)

    clean_screen = list(causal.get("clean_screen") or [])
    excluded_image_ids = sorted(
        {
            str(row.get("image_id"))
            for row in clean_screen
            if str(row.get("image_id") or "").strip()
        }
    )
    if not excluded_image_ids:
        raise MultimodalLensRefused(
            "the completed causal report records no screened image identities"
        )
    payload = {
        "version": CAUSAL_SOURCE_VERSION,
        "run_dir": str(root),
        "source_scientific_fingerprint": final.get("scientific_fingerprint"),
        "final_report_checksum": expected_final_report_checksum,
        "cross_report_checksum": expected_cross_report_checksum,
        "cross_report_path": str(cross_path),
        "causal_report_checksum": expected_causal_report_checksum,
        "lens_checksums": recorded_checksums,
        "lens_paths": lens_paths,
        "excluded_image_ids": excluded_image_ids,
        "n_excluded_images": len(excluded_image_ids),
    }
    return {**payload, "source_digest": payload_checksum(payload)}


@dataclass(frozen=True)
class FitUnit:
    """One processor input in a frozen fitting order."""

    unit_id: str
    group_id: str
    image_id: str
    modality: str
    caption: str
    image_path: str
    audio_path: str
    prompt: str

    def to_dict(self) -> dict:
        return asdict(self)


def _stable_rank(value: str, seed: str) -> str:
    return hashlib.sha256(f"{seed}|{value}".encode()).hexdigest()


def _surface_contains(text: str, concept: str) -> bool:
    words = "".join(ch.lower() if ch.isalnum() else " " for ch in text).split()
    needle = "".join(ch.lower() if ch.isalnum() else " " for ch in concept).split()
    if not needle:
        return False
    width = len(needle)
    return any(words[i : i + width] == needle for i in range(len(words) - width + 1))


def fitting_prompt(modality: str, caption: str) -> str:
    """A long, candidate-free prompt whose evidence channel is explicit.

    Text receives the written caption.  Image and audio receive no transcript;
    their media is supplied separately to the processor.  The common suffix
    keeps the textual register comparable and makes the text sequence long
    enough for the estimator's frozen attention-sink exclusion.
    """

    suffix = (
        "Examine the evidence carefully and form a concise internal summary of "
        "its main subject, setting, and action. Do not use a candidate list. "
        "Prepare to continue with one descriptive word.\nSummary:"
    )
    if modality == "text":
        return f"Evidence is a written caption.\nCaption: {caption.strip()}\n{suffix}"
    if modality == "image":
        return f"Evidence is the attached image.\n{suffix}"
    if modality == "spoken_audio":
        return f"Evidence is the attached spoken recording.\n{suffix}"
    raise ValueError(f"unknown modality {modality!r}")


def evaluation_prompt(modality: str, caption: str) -> str:
    """Open, candidate-free prompt for full-vocabulary cross evaluation."""

    question = (
        "What is the main subject of the evidence? Answer with one word.\nAnswer:"
    )
    if modality == "text":
        return f"Caption: {caption.strip()}\n{question}"
    if modality in ("image", "spoken_audio"):
        return question
    raise ValueError(f"unknown modality {modality!r}")


def _concept_fields(group: Mapping) -> list[str]:
    values: list[str] = []
    # ExpandedManifest v3 stores audited COCO object labels under
    # ``concept_annotations``.  The shorter names are retained for normalized
    # manifests and MOCK fixtures.
    for key in (
        "concept",
        "concepts",
        "categories",
        "category_names",
        "concept_annotations",
    ):
        value = group.get(key)
        if isinstance(value, str):
            values.append(value)
        elif isinstance(value, Sequence):
            values.extend(str(item) for item in value)
    return values


def _eligible_group(group: Mapping, excluded_concepts: Sequence[str]) -> bool:
    required = ("group_id", "image_id", "caption", "image_path", "audio_path")
    if any(not str(group.get(key) or "").strip() for key in required):
        return False
    haystacks = [str(group.get("caption") or ""), *_concept_fields(group)]
    return not any(
        _surface_contains(haystack, concept)
        for concept in excluded_concepts
        for haystack in haystacks
    )


def build_matched_plan(
    groups: Sequence[Mapping],
    *,
    n_fit_groups: int,
    n_eval_groups: int,
    seed: str,
    excluded_eval_concepts: Sequence[str] = (),
) -> dict:
    """Freeze matched, image-disjoint fit and evaluation populations.

    One recording is retained per photograph before ranking.  Every retained
    fit photograph contributes exactly one unit to each unimodal arm and one
    unit to the pooled arm.  The pooled assignment cycles across modalities,
    so it has the same number of fitting examples as every comparator while
    remaining balanced to within one example.  The evaluation photographs are
    the next ranked photographs and never enter any fit arm.
    """

    if n_fit_groups < 1 or n_eval_groups < 1:
        raise ValueError("n_fit_groups and n_eval_groups must be positive")
    by_image: dict[str, list[dict]] = {}
    for raw in groups:
        group = dict(raw)
        if not _eligible_group(group, excluded_eval_concepts):
            continue
        by_image.setdefault(str(group["image_id"]), []).append(group)

    representatives: list[dict] = []
    for siblings in by_image.values():
        representatives.append(
            min(
                siblings,
                key=lambda row: _stable_rank(str(row["group_id"]), f"{seed}|sibling"),
            )
        )
    ordered = sorted(
        representatives,
        key=lambda row: _stable_rank(str(row["image_id"]), f"{seed}|image"),
    )
    needed = n_fit_groups + n_eval_groups
    if len(ordered) < needed:
        raise MultimodalLensRefused(
            f"only {len(ordered)} concept-neutral distinct photographs are "
            f"available, but the frozen plan needs {needed}"
        )
    fit_groups = ordered[:n_fit_groups]
    eval_groups = ordered[n_fit_groups:needed]

    def unit(group: Mapping, modality: str) -> FitUnit:
        group_id = str(group["group_id"])
        return FitUnit(
            unit_id=f"{group_id}:{modality}",
            group_id=group_id,
            image_id=str(group["image_id"]),
            modality=modality,
            caption=str(group["caption"]),
            image_path=str(group["image_path"]),
            audio_path=str(group["audio_path"]),
            prompt=fitting_prompt(modality, str(group["caption"])),
        )

    per_modality = {
        modality: [unit(group, modality) for group in fit_groups]
        for modality in MODALITIES
    }
    # One view per photograph keeps the pooled arm sample-count matched to all
    # three unimodal arms.  Cycling modalities makes every prefix maximally
    # balanced without selecting examples from model results.
    pooled = [
        per_modality[MODALITIES[index % len(MODALITIES)]][index]
        for index in range(n_fit_groups)
    ]
    arms = {**per_modality, "pooled": pooled}
    payload = {
        "version": PLAN_VERSION,
        "seed": seed,
        "excluded_eval_concepts": list(excluded_eval_concepts),
        "n_fit_groups": n_fit_groups,
        "n_eval_groups": n_eval_groups,
        "fit_image_ids": [str(group["image_id"]) for group in fit_groups],
        "eval_image_ids": [str(group["image_id"]) for group in eval_groups],
        "fit_groups": fit_groups,
        "eval_groups": eval_groups,
        "arms": {name: [row.to_dict() for row in rows] for name, rows in arms.items()},
        "pooled_modality_counts": {
            modality: sum(row.modality == modality for row in pooled)
            for modality in MODALITIES
        },
        "one_group_per_image": True,
        "fit_eval_image_overlap": sorted(
            {str(g["image_id"]) for g in fit_groups}
            & {str(g["image_id"]) for g in eval_groups}
        ),
    }
    payload["plan_digest"] = payload_checksum(payload)
    return payload


def plan_units(plan: Mapping, arm: str) -> list[FitUnit]:
    if arm not in LENS_ARMS:
        raise ValueError(f"unknown lens arm {arm!r}")
    return [FitUnit(**row) for row in plan["arms"][arm]]


def select_causal_groups(
    groups: Sequence[Mapping],
    *,
    concepts: Sequence[str],
    n_per_concept: int,
    excluded_image_ids: Sequence[str],
    seed: str,
) -> dict[str, list[dict]]:
    """Freeze fresh, distinct photographs for the unrestricted causal stage."""

    excluded = {str(value) for value in excluded_image_ids}
    result: dict[str, list[dict]] = {}
    for concept in concepts:
        by_image: dict[str, list[dict]] = {}
        for raw in groups:
            group = dict(raw)
            image_id = str(group.get("image_id") or "")
            if not image_id or image_id in excluded:
                continue
            if any(
                not str(group.get(key) or "").strip()
                for key in ("group_id", "caption", "image_path", "audio_path")
            ):
                continue
            labelled = any(
                _surface_contains(value, concept) for value in _concept_fields(group)
            )
            mentioned = _surface_contains(str(group.get("caption") or ""), concept)
            if not (labelled and mentioned):
                continue
            by_image.setdefault(image_id, []).append(group)
        representatives = [
            min(
                siblings,
                key=lambda row: _stable_rank(
                    str(row["group_id"]), f"{seed}|{concept}|sibling"
                ),
            )
            for siblings in by_image.values()
        ]
        ordered = sorted(
            representatives,
            key=lambda row: _stable_rank(
                str(row["image_id"]), f"{seed}|{concept}|image"
            ),
        )
        if len(ordered) < n_per_concept:
            raise MultimodalLensRefused(
                f"concept {concept!r} has only {len(ordered)} fresh, caption-"
                f"supported photographs; {n_per_concept} are required"
            )
        result[str(concept)] = ordered[:n_per_concept]
    overlap = set()
    seen: set[str] = set()
    for rows in result.values():
        for row in rows:
            image_id = str(row["image_id"])
            if image_id in seen:
                overlap.add(image_id)
            seen.add(image_id)
    if overlap:
        raise MultimodalLensRefused(
            f"causal concepts share selected photographs {sorted(overlap)}"
        )
    return result


def _replicate_tensors(tensors: Mapping[str, Any], batch_size: int) -> dict:
    """Repeat one processor example without changing any feature value."""

    out: dict[str, Any] = {}
    for key, value in dict(tensors).items():
        if not torch.is_tensor(value):
            out[key] = value
            continue
        if value.ndim == 0:
            out[key] = value
        elif value.shape[0] == 1:
            out[key] = value.expand(batch_size, *value.shape[1:]).contiguous()
        else:
            raise MultimodalLensRefused(
                f"processor tensor {key!r} has batch dimension {value.shape[0]}, "
                "expected one example before Jacobian replication"
            )
    out.setdefault("use_cache", False)
    return out


def jacobian_for_built_inputs(
    backend,
    inputs,
    source_layers: Sequence[int],
    *,
    target_layer: int,
    dim_batch: int = 8,
    skip_first: int = 16,
    backward_context: Callable[[], object] | None = None,
) -> tuple[dict[int, torch.Tensor], int, int]:
    """The upstream estimator over real processor tensors.

    Media placeholders and their decoder positions participate exactly as they
    do in an ordinary multimodal forward.  No transcript is accepted here;
    that invariant is enforced by the backend that built ``inputs``.
    """

    source_layers = sorted({int(layer) for layer in source_layers})
    if not source_layers or source_layers[-1] >= int(target_layer):
        raise ValueError("source layers must be nonempty and precede target_layer")
    if int(target_layer) >= int(backend.n_layers):
        raise ValueError("target_layer is outside the decoder")
    seq_len = int(inputs.prompt_len)
    mask = valid_position_mask(seq_len, skip_first=skip_first)
    valid = mask.nonzero(as_tuple=True)[0]
    n_valid = int(valid.numel())
    d_model = int(backend.d_model)
    jacobians = {
        layer: torch.zeros(d_model, d_model, dtype=torch.float32)
        for layer in source_layers
    }
    n_passes = math.ceil(d_model / dim_batch)
    tensors = _replicate_tensors(inputs.tensors, dim_batch)

    with (
        backward_context() if backward_context is not None else nullcontext(),
        ActivationRecorder(
            backend.blocks,
            at=[*source_layers, int(target_layer)],
            start_graph_at=min(source_layers),
        ) as recorder,
        torch.enable_grad(),
    ):
        backend.hf_model(**tensors)
        target = recorder.activations[int(target_layer)]
        sources = [recorder.activations[layer] for layer in source_layers]
        valid_target = valid.to(target.device)
        batch = torch.arange(dim_batch, device=target.device)
        cotangent = torch.zeros_like(target)
        for pass_index, start in enumerate(range(0, d_model, dim_batch)):
            width = min(dim_batch, d_model - start)
            cotangent.zero_()
            cotangent[
                batch[:width, None], valid_target[None, :], start + batch[:width, None]
            ] = 1.0
            grads = torch.autograd.grad(
                outputs=target,
                inputs=sources,
                grad_outputs=cotangent,
                retain_graph=pass_index < n_passes - 1,
            )
            for layer, grad in zip(source_layers, grads, strict=True):
                positions = valid.to(grad.device)
                rows = grad[:width, positions, :].float().mean(dim=1)
                jacobians[layer][start : start + width] = rows.cpu()
            del grads
    return jacobians, seq_len, n_valid


def _atomic_torch_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def fit_arm(
    backend,
    units: Sequence[FitUnit],
    *,
    build_inputs: Callable[[FitUnit], Any],
    source_layers: Sequence[int],
    target_layer: int,
    checkpoint_path: str | Path,
    arm: str,
    scientific_fingerprint: str,
    dim_batch: int = 8,
    skip_first: int = 16,
    checkpoint_every: int = 5,
    backward_context: Callable[[], object] | None = None,
    progress: Callable[[dict], None] | None = None,
) -> JacobianLens:
    """Fit one arm with an atomic, fingerprint-bound accumulator."""

    if arm not in LENS_ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    layers = sorted({int(layer) for layer in source_layers})
    record_digest = payload_checksum([unit.to_dict() for unit in units])
    contract = {
        "estimator_version": ESTIMATOR_VERSION,
        "scientific_fingerprint": scientific_fingerprint,
        "arm": arm,
        "record_digest": record_digest,
        "source_layers": layers,
        "target_layer": int(target_layer),
        "dim_batch": int(dim_batch),
        "skip_first": int(skip_first),
        "d_model": int(backend.d_model),
    }
    contract_digest = payload_checksum(contract)
    checkpoint = Path(checkpoint_path)
    if checkpoint.is_file():
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state.get("contract_digest") != contract_digest:
            raise MultimodalLensRefused(
                f"checkpoint {checkpoint} belongs to a different arm, "
                "population, processor protocol, layer grid, or estimator; "
                "refusing to mix accumulators"
            )
        sums = state["jacobian_sum"]
        n_done = int(state["n_done"])
        next_index = int(state["next_index"])
    else:
        sums = {
            layer: torch.zeros(backend.d_model, backend.d_model, dtype=torch.float32)
            for layer in layers
        }
        n_done = 0
        next_index = 0

    def save() -> None:
        _atomic_torch_save(
            {
                "contract": contract,
                "contract_digest": contract_digest,
                "jacobian_sum": sums,
                "n_done": n_done,
                "next_index": next_index,
            },
            checkpoint,
        )

    for index, unit in enumerate(units):
        if index < next_index:
            continue
        started = time.perf_counter()
        built = build_inputs(unit)
        contribution, seq_len, n_valid = jacobian_for_built_inputs(
            backend,
            built,
            layers,
            target_layer=target_layer,
            dim_batch=dim_batch,
            skip_first=skip_first,
            backward_context=backward_context,
        )
        for layer in layers:
            sums[layer] += contribution[layer]
        n_done += 1
        next_index = index + 1
        wrote = next_index % max(1, checkpoint_every) == 0
        if wrote:
            save()
        if progress is not None:
            progress(
                {
                    "arm": arm,
                    "index": next_index,
                    "total": len(units),
                    "unit_id": unit.unit_id,
                    "modality": unit.modality,
                    "seq_len": seq_len,
                    "n_valid_positions": n_valid,
                    "n_done": n_done,
                    "elapsed_seconds": time.perf_counter() - started,
                    "checkpoint_written": wrote,
                }
            )
    save()
    if n_done != len(units):
        raise MultimodalLensRefused(
            f"arm {arm} accumulated {n_done} units for a {len(units)}-unit plan"
        )
    return JacobianLens(
        jacobians={layer: sums[layer] / n_done for layer in layers},
        n_prompts=n_done,
        d_model=int(backend.d_model),
    )


def selected_lens_vector(
    lens: JacobianLens, unembedding_weight: torch.Tensor, *, layer: int, token_id: int
) -> torch.Tensor:
    """One row of ``W_U @ J_l`` without materializing the full dictionary."""

    row = unembedding_weight[int(token_id)].detach().float().cpu()
    return row @ lens.jacobians[int(layer)].detach().float().cpu()


def build_swap_bases_for_lens(
    lens: JacobianLens,
    unembedding_weight: torch.Tensor,
    *,
    layers: Sequence[int],
    source: ConceptToken,
    target: ConceptToken,
) -> dict[int, SwapBasis]:
    return {
        int(layer): build_swap_basis_from_vectors(
            selected_lens_vector(
                lens, unembedding_weight, layer=int(layer), token_id=source.token_id
            ),
            selected_lens_vector(
                lens, unembedding_weight, layer=int(layer), token_id=target.token_id
            ),
            layer=int(layer),
            source=source,
            target=target,
        )
        for layer in layers
    }


@torch.no_grad()
def capture_eval_rows(
    backend,
    lenses: Mapping[str, JacobianLens],
    eval_groups: Sequence[Mapping],
    *,
    build_inputs: Callable[[Mapping, str, str], Any],
    layers: Sequence[int],
) -> list[dict]:
    """Full-vocabulary native-answer fidelity for every arm x modality x layer."""

    rows: list[dict] = []
    norm = backend.hf_model.model.language_model.norm
    head = backend.hf_model.lm_head
    for group in eval_groups:
        for modality in MODALITIES:
            prompt = evaluation_prompt(modality, str(group["caption"]))
            built = build_inputs(group, modality, prompt)
            with ActivationRecorder(backend.blocks, at=list(layers)) as recorder:
                clean_logits = backend.forward_logits(built.tensors)[
                    0, built.final_prompt_position
                ].float()
            clean_token = int(clean_logits.argmax())
            for arm, lens in lenses.items():
                for layer in layers:
                    h = recorder.activations[int(layer)][
                        0, built.final_prompt_position
                    ].detach().float().cpu()
                    transported = lens.transport(h, int(layer))
                    device = head.weight.device
                    lens_logits = head(
                        norm(transported.to(device=device, dtype=head.weight.dtype))
                    )[0].float() if transported.ndim == 2 else head(
                        norm(transported.to(device=device, dtype=head.weight.dtype))
                    ).float()
                    target_score = lens_logits[clean_token]
                    optimistic_rank = 1 + int((lens_logits > target_score).sum())
                    pessimistic_rank = int((lens_logits >= target_score).sum())
                    rows.append(
                        {
                            "version": CROSS_EVAL_VERSION,
                            "group_id": str(group["group_id"]),
                            "image_id": str(group["image_id"]),
                            "test_modality": modality,
                            "lens_arm": arm,
                            "layer": int(layer),
                            "clean_token_id": clean_token,
                            "lens_top_token_id": int(lens_logits.argmax()),
                            "top1_agreement": int(lens_logits.argmax()) == clean_token,
                            "optimistic_rank": optimistic_rank,
                            "pessimistic_rank": pessimistic_rank,
                            "midrank": (optimistic_rank + pessimistic_rank) / 2.0,
                            "reciprocal_midrank": 2.0
                            / (optimistic_rank + pessimistic_rank),
                        }
                    )
    return rows


def summarize_cross_eval(rows: Sequence[Mapping], *, shuffle_seed: int = 20260819) -> dict:
    """Summarize the frozen 4x3 matrix and a shuffled-target control."""

    grouped: dict[tuple[str, str, int], list[dict]] = {}
    for raw in rows:
        row = dict(raw)
        grouped.setdefault(
            (str(row["lens_arm"]), str(row["test_modality"]), int(row["layer"])),
            [],
        ).append(row)
    cells: list[dict] = []
    for (arm, modality, layer), cell in sorted(grouped.items()):
        n = len(cell)
        cells.append(
            {
                "lens_arm": arm,
                "test_modality": modality,
                "layer": layer,
                "n": n,
                "top1_agreement": sum(bool(r["top1_agreement"]) for r in cell) / n,
                "mrr": sum(float(r["reciprocal_midrank"]) for r in cell) / n,
                "median_midrank": float(
                    torch.tensor([float(r["midrank"]) for r in cell]).median()
                ),
            }
        )
    # The control permutes target ownership within each modality/layer.  The
    # full per-token logits are deliberately not persisted, so this control is
    # expressed as the top-1 agreement expected under a derangement of the
    # recorded native targets.
    controls: list[dict] = []
    for (arm, modality, layer), cell in sorted(grouped.items()):
        ordered = sorted(cell, key=lambda r: (str(r["image_id"]), str(r["group_id"])))
        targets = [int(r["clean_token_id"]) for r in ordered]
        shuffled = targets[:]
        random.Random(f"{shuffle_seed}|{arm}|{modality}|{layer}").shuffle(shuffled)
        controls.append(
            {
                "lens_arm": arm,
                "test_modality": modality,
                "layer": layer,
                "kind": "shuffled_native_target_top1",
                "top1_agreement": sum(
                    int(r["lens_top_token_id"]) == target
                    for r, target in zip(ordered, shuffled, strict=True)
                )
                / len(ordered),
            }
        )
    payload = {
        "version": CROSS_EVAL_VERSION,
        "n_rows": len(rows),
        "cells": cells,
        "controls": controls,
        "primary_comparison": "pooled_vs_text_on_image_and_spoken_audio",
        "interpretation": (
            "native-answer fidelity measures whether each frozen Jacobian map "
            "transports a modality-induced residual toward the model's own "
            "unrestricted next-token answer; it is not semantic accuracy"
        ),
    }
    payload["report_checksum"] = payload_checksum(payload)
    return payload


@torch.no_grad()
def unrestricted_swap_trial(
    backend,
    inputs,
    *,
    bases: Mapping[int, SwapBasis],
    alpha: float = 1.0,
) -> dict:
    """One exact exchange scored on the unrestricted next-token distribution."""

    clean = backend.forward_logits(inputs.tensors)[0, inputs.final_prompt_position].float()
    with coordinate_swap_band(
        backend.blocks,
        bases,
        alpha=float(alpha),
        prompt_len=inputs.prompt_len,
        position_rule=PRIMARY_POSITION_RULE,
        evidence_span=inputs.modality_token_range,
        record_coordinates=False,
    ) as stats:
        patched = backend.forward_logits(inputs.tensors)[
            0, inputs.final_prompt_position
        ].float()
    return {
        "alpha": float(alpha),
        "teacher_forcing_used": False,
        "candidate_list_supplied": False,
        "clean_top_token_id": int(clean.argmax()),
        "patched_top_token_id": int(patched.argmax()),
        "prediction_changed": int(clean.argmax()) != int(patched.argmax()),
        "layers_patched": sorted(int(layer) for layer in stats),
        "positions_patched": {
            str(layer): list(stats[layer].get("positions") or []) for layer in stats
        },
    }


def fit_budget(
    *, n_fit_groups: int, n_layers: int, d_model: int = 2560, dim_batch: int = 8
) -> dict:
    """Exact forward/backward counts printed before model load."""

    per_prompt_backward = math.ceil(d_model / dim_batch)
    per_arm = {
        "text": n_fit_groups,
        "image": n_fit_groups,
        "spoken_audio": n_fit_groups,
        "pooled": n_fit_groups,
    }
    total_prompts = sum(per_arm.values())
    return {
        "n_fit_groups": n_fit_groups,
        "n_layers": n_layers,
        "dim_batch": dim_batch,
        "per_prompt_forward": 1,
        "per_prompt_backward": per_prompt_backward,
        "prompts_by_arm": per_arm,
        "total_prompt_forwards": total_prompts,
        "total_backward_passes": total_prompts * per_prompt_backward,
        "checkpoint_contract": (
            "one atomic accumulator per arm; at most checkpoint_every newly "
            "completed units are recomputed after a disconnect"
        ),
    }


__all__ = [
    "CROSS_EVAL_VERSION",
    "ESTIMATOR_VERSION",
    "FitUnit",
    "LENS_ARMS",
    "MODALITIES",
    "MultimodalLensRefused",
    "build_matched_plan",
    "build_swap_bases_for_lens",
    "capture_eval_rows",
    "evaluation_prompt",
    "fit_arm",
    "fit_budget",
    "fitting_prompt",
    "jacobian_for_built_inputs",
    "plan_units",
    "selected_lens_vector",
    "select_causal_groups",
    "summarize_cross_eval",
    "unrestricted_swap_trial",
]
