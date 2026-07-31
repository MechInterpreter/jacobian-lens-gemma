# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Frozen-lens validation, activation capture, J-space codes, and the
representational tests.

The lens is a *previously fitted, text-calibrated* artifact. It is validated
against the model it is about to be used with and then never touched again:
nothing here fits, refits, rescales, or substitutes a basis. If no compatible
lens exists the pilot stops — a random basis or an SAE dictionary would answer
a different question.

Given the frozen dictionary ``V`` (rows of ``W_U @ J_l``, the repository's
existing J-space convention) each final-prompt-token activation ``h`` is
decomposed by the repository's validated nonnegative gradient pursuit into
``h ≈ V s``, ``s >= 0``. The representational tests then ask whether those
coordinates carry concept identity across modalities better than the raw
residual does, and better than shuffled labels.

Retrieval never lets a group retrieve its own synchronized twin: the written
caption, the recording of that caption, and the image share content by
construction, so scoring them against each other would measure the dataset's
pairing rather than the model's representation.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Mapping, Sequence

import torch

from jlens.hooks import ActivationRecorder
from jlens.lens import JacobianLens
from jlens.mmpilot.backend import BuiltInputs, PilotBackend
from jlens.mmpilot.store import payload_checksum
from jlens.pursuit import JSpaceDictionary, PursuitSettings, gradient_pursuit
from jlens.similarity import sparse_cosine, weighted_jaccard

#: The one documented convention for the whole pilot. Recorded in every
#: artifact so a later run cannot silently disagree.
CONVENTIONS = {
    "hook_site": "output of model.language_model.layers[l] (post-block residual)",
    "position": "final prompt token (prompt_len - 1)",
    "dictionary": "rows of W_U @ J_l (jlens.pursuit.JSpaceDictionary.from_lens)",
    "final_norm_weight_folded": False,
    "code_orientation": "h ~= V s with s >= 0; coefficients on raw atoms",
    "direction_normalization": (
        "unit L2, then scaled by the mean L2 norm of the target-modality clean "
        "activations; alpha multiplies that scaled vector"
    ),
}


class LensIncompatibleError(RuntimeError):
    """The available lens artifact cannot be used with this model/layers."""


# ----------------------------------------------------------- lens validation


def validate_lens(
    lens: JacobianLens,
    *,
    lens_path: str,
    lens_checksum: str,
    layers: Sequence[int],
    model_repo_id: str,
    model_revision: str,
    expect_model_repo_id: str,
    expect_model_revision: str,
    expect_d_model: int,
    expect_checksum: str | None = None,
) -> dict:
    """Check every property the pilot depends on, and record it.

    Raises:
        LensIncompatibleError: On any mismatch — wrong model, wrong revision,
            wrong width, an unfitted layer, a non-finite Jacobian, or a
            checksum that disagrees with the pin. There is no fallback path:
            the notebook reports NO-GO and names the missing artifact.
    """
    problems: list[str] = []
    if expect_checksum and lens_checksum != expect_checksum:
        problems.append(f"lens checksum {lens_checksum} != pinned {expect_checksum}")
    if model_repo_id != expect_model_repo_id:
        problems.append(f"model repo {model_repo_id} != lens repo {expect_model_repo_id}")
    if model_revision != expect_model_revision:
        problems.append(
            f"model revision {model_revision} != lens revision {expect_model_revision}"
        )
    if lens.d_model != expect_d_model:
        problems.append(f"lens d_model {lens.d_model} != model d_model {expect_d_model}")
    missing = [layer for layer in layers if layer not in lens.jacobians]
    if missing:
        problems.append(f"layers {missing} were never fitted (have {lens.source_layers})")
    for layer, jacobian in lens.jacobians.items():
        if not bool(torch.isfinite(jacobian).all()):
            problems.append(f"non-finite Jacobian at layer {layer}")
        if jacobian.shape != (lens.d_model, lens.d_model):
            problems.append(f"layer {layer} Jacobian is {tuple(jacobian.shape)}")
    if problems:
        raise LensIncompatibleError(
            "frozen lens is not compatible with this run:\n  - " + "\n  - ".join(problems)
        )
    return {
        "lens_path": lens_path,
        "lens_checksum": lens_checksum,
        "model_repo_id": model_repo_id,
        "model_revision": model_revision,
        "d_model": lens.d_model,
        "fitted_layers": list(lens.source_layers),
        "layers_used": list(layers),
        "n_prompts_fitted_on": lens.n_prompts,
        "calibration_modality": "text",
        "frozen": True,
        "conventions": dict(CONVENTIONS),
    }


# --------------------------------------------------------- activation capture


def tensor_checksum(tensor: torch.Tensor) -> str:
    """``sha256:`` over the float32 bytes — exact, dtype- and order-stable."""
    contiguous = tensor.detach().to(torch.float32).cpu().contiguous()
    return "sha256:" + hashlib.sha256(contiguous.numpy().tobytes()).hexdigest()


@torch.no_grad()
def capture_final_prompt_activations(
    backend: PilotBackend, inputs: BuiltInputs, layers: Sequence[int]
) -> dict[int, torch.Tensor]:
    """Residual at the final prompt token for each layer, moved to CPU at once.

    Raises:
        RuntimeError: If the hooked sequence length disagrees with the
            prompt length the inputs reported. That would mean the model
            expanded modality tokens internally, so ``prompt_len - 1`` is not
            the final prompt token — better to stop than to measure the wrong
            position.
    """
    with ActivationRecorder(backend.blocks, at=list(layers)) as recorder:
        backend.forward_logits(inputs.tensors)
        captured = {layer: recorder.activations[layer] for layer in layers}
    out: dict[int, torch.Tensor] = {}
    for layer, hidden in captured.items():
        if hidden.shape[1] != inputs.prompt_len:
            raise RuntimeError(
                f"layer {layer} saw sequence length {hidden.shape[1]} but the "
                f"inputs report prompt_len={inputs.prompt_len}; the final-prompt "
                "position cannot be resolved reliably"
            )
        out[layer] = hidden[0, inputs.prompt_len - 1].detach().float().cpu().clone()
    return out


def activation_record(
    activation: torch.Tensor,
    *,
    sample_id: str,
    group_id: str,
    image_id: str,
    concept: str | None,
    modality: str,
    split: str,
    layer: int,
    inputs: BuiltInputs,
    model_revision: str,
) -> dict:
    """One atomic (sample, modality, layer) activation unit."""
    return {
        "sample_id": sample_id,
        "group_id": group_id,
        "image_id": image_id,
        "concept": concept,
        "modality": modality,
        "split": split,
        "layer": layer,
        "hook_site": CONVENTIONS["hook_site"],
        "position": CONVENTIONS["position"],
        "shape": list(activation.shape),
        "norm": float(activation.norm()),
        "activation_checksum": tensor_checksum(activation),
        "activation": [float(x) for x in activation.tolist()],
        "prompt_hash": inputs.prompt_hash,
        "media_checksum": inputs.media_checksum,
        "prompt_len": inputs.prompt_len,
        "model_revision": model_revision,
    }


def activation_tensor(record: Mapping) -> torch.Tensor:
    return torch.tensor(record["activation"], dtype=torch.float32)


# --------------------------------------------------------------- J-space codes


def build_dictionary(
    lens: JacobianLens,
    layer: int,
    unembedding_weight: torch.Tensor,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float16,
    build_chunk_rows: int | None = 32768,
) -> JSpaceDictionary:
    """The frozen layer dictionary ``V``.

    fp16 storage and chunked construction keep the ~262k x 2560 atom matrix
    inside an L4 next to a bf16 Gemma 4 E4B; correlations are still computed
    in float32 by the pursuit.
    """
    return JSpaceDictionary.from_lens(
        lens,
        layer,
        unembedding_weight,
        final_norm_weight=None,
        device=device,
        dtype=dtype,
        build_chunk_rows=build_chunk_rows,
    )


def jspace_code(
    activation: torch.Tensor,
    dictionary: JSpaceDictionary,
    settings: PursuitSettings,
    *,
    activation_checksum: str,
    lens_checksum: str,
) -> dict:
    """Decompose one activation into nonnegative J-space coordinates."""
    target = activation.to(device=dictionary.device, dtype=torch.float32).unsqueeze(0)
    result = gradient_pursuit(target, dictionary, settings)
    record = result.to_records()[0]
    support = record["token_ids"]
    coefficients = record["coefficients"]
    return {
        "layer": dictionary.layer,
        "support": support,
        "coefficients": coefficients,
        "n_active": sum(1 for c in coefficients if c > 0),
        "sparsity": (
            1.0 - sum(1 for c in coefficients if c > 0) / dictionary.n_atoms
        ),
        "reconstruction_error": record["residual_norm"],
        "relative_residual": record["relative_residual"],
        "explained_fraction": record["explained_fraction"],
        "target_norm": record["target_norm"],
        "convergence_status": record["stop_reason"],
        "activation_checksum": activation_checksum,
        "lens_checksum": lens_checksum,
        "config_hash": payload_checksum(
            {"settings": record["settings"], "conventions": CONVENTIONS}
        ),
    }


def code_map(record: Mapping) -> dict[int, float]:
    """``{atom id: coefficient}`` for the strictly positive coefficients."""
    return {
        int(token): float(coefficient)
        for token, coefficient in zip(
            record["support"], record["coefficients"], strict=True
        )
        if coefficient > 0
    }


def reconstruct_direction(
    code: Mapping[int, float], dictionary: JSpaceDictionary
) -> torch.Tensor:
    """``V s`` for a sparse code — the residual-space image of J-space coords."""
    out = torch.zeros(dictionary.d_model, dtype=torch.float32, device=dictionary.device)
    for atom, coefficient in code.items():
        out = out + float(coefficient) * dictionary.atoms[int(atom)].float()
    return out


# ------------------------------------------------------- representational tests


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denominator = float(a.norm()) * float(b.norm())
    return float(a.dot(b)) / denominator if denominator > 0 else 0.0


def jspace_similarity(a: Mapping[int, float], b: Mapping[int, float]) -> float:
    """Cosine between two sparse codes (reuses :mod:`jlens.similarity`)."""
    return float(sparse_cosine(a, b))


def support_overlap(a: Mapping[int, float], b: Mapping[int, float]) -> float:
    """Coefficient-weighted Jaccard over the two supports."""
    return float(weighted_jaccard(a, b))


def _pairs(
    sources: Sequence[Mapping], targets: Sequence[Mapping]
) -> list[tuple[Mapping, list[Mapping]]]:
    """Each source with its admissible targets (never its own image/group)."""
    out = []
    for source in sources:
        admissible = [
            target
            for target in targets
            if target["group_id"] != source["group_id"]
            and target["image_id"] != source["image_id"]
        ]
        if admissible:
            out.append((source, admissible))
    return out


def retrieval_metrics(
    sources: Sequence[Mapping],
    targets: Sequence[Mapping],
    similarity,
    *,
    labels: Mapping[str, str] | None = None,
) -> dict:
    """Nearest-neighbour retrieval accuracy from ``sources`` into ``targets``.

    ``labels`` overrides each target's concept (used by the shuffled control).
    Ties break on ``sample_id`` so results are reproducible.
    """
    hits = 0
    reciprocal = 0.0
    chance = 0.0
    evaluated = _pairs(sources, targets)
    for source, admissible in evaluated:
        def label(item: Mapping) -> str:
            return labels[item["sample_id"]] if labels else item["concept"]

        ranked = sorted(
            admissible,
            key=lambda target: (-similarity(source, target), target["sample_id"]),
        )
        same = [i for i, target in enumerate(ranked) if label(target) == source["concept"]]
        if same and same[0] == 0:
            hits += 1
        if same:
            reciprocal += 1.0 / (same[0] + 1)
        chance += len(same) / len(ranked)
    n = len(evaluated)
    return {
        "n_queries": n,
        "top1_accuracy": hits / n if n else 0.0,
        "mrr": reciprocal / n if n else 0.0,
        "chance_top1": chance / n if n else 0.0,
    }


def separation_metrics(
    sources: Sequence[Mapping], targets: Sequence[Mapping], similarity
) -> dict:
    """Matched-concept vs mismatched-concept similarity across modalities."""
    matched: list[float] = []
    mismatched: list[float] = []
    for source, admissible in _pairs(sources, targets):
        for target in admissible:
            value = similarity(source, target)
            (matched if target["concept"] == source["concept"] else mismatched).append(value)
    return {
        "n_matched_pairs": len(matched),
        "n_mismatched_pairs": len(mismatched),
        "matched_mean": _mean(matched),
        "mismatched_mean": _mean(mismatched),
        "gap": _mean(matched) - _mean(mismatched),
        "cohens_d": _cohens_d(matched, mismatched),
    }


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    mean_a, mean_b = _mean(a), _mean(b)
    var_a = sum((x - mean_a) ** 2 for x in a) / (len(a) - 1)
    var_b = sum((x - mean_b) ** 2 for x in b) / (len(b) - 1)
    pooled = math.sqrt((var_a + var_b) / 2)
    return (mean_a - mean_b) / pooled if pooled > 0 else 0.0


def shuffled_label_control(
    sources: Sequence[Mapping],
    targets: Sequence[Mapping],
    similarity,
    *,
    n_permutations: int = 50,
    seed: int = 20260731,
) -> dict:
    """Retrieval accuracy under permuted target labels (deterministic)."""
    rng = random.Random(seed)
    accuracies = []
    concepts = [target["concept"] for target in targets]
    for _ in range(n_permutations):
        permuted = concepts[:]
        rng.shuffle(permuted)
        labels = {
            target["sample_id"]: label
            for target, label in zip(targets, permuted, strict=True)
        }
        accuracies.append(
            retrieval_metrics(sources, targets, similarity, labels=labels)["top1_accuracy"]
        )
    accuracies.sort()
    index = min(len(accuracies) - 1, int(0.95 * len(accuracies)))
    return {
        "n_permutations": n_permutations,
        "mean_top1_accuracy": _mean(accuracies),
        "p95_top1_accuracy": accuracies[index] if accuracies else 0.0,
    }


def representational_report(
    samples: Sequence[Mapping],
    *,
    modalities: Sequence[str],
    n_permutations: int = 50,
    seed: int = 20260731,
) -> dict:
    """All ordered modality pairs, in J-space and in raw residual space.

    Each ``samples`` entry needs ``sample_id``, ``group_id``, ``image_id``,
    ``concept``, ``modality``, ``code`` (``{atom: coefficient}``) and
    ``activation`` (a list of floats). Only positives (a non-null concept)
    take part.
    """
    positives = [s for s in samples if s.get("concept")]
    by_modality = {
        modality: [s for s in positives if s["modality"] == modality]
        for modality in modalities
    }
    vectors = {
        s["sample_id"]: torch.tensor(s["activation"], dtype=torch.float32)
        for s in positives
    }

    def jspace(a: Mapping, b: Mapping) -> float:
        return jspace_similarity(a["code"], b["code"])

    def overlap(a: Mapping, b: Mapping) -> float:
        return support_overlap(a["code"], b["code"])

    def raw(a: Mapping, b: Mapping) -> float:
        return _cosine(vectors[a["sample_id"]], vectors[b["sample_id"]])

    pairs: dict[str, dict] = {}
    for source_modality in modalities:
        for target_modality in modalities:
            if source_modality == target_modality:
                continue
            sources = by_modality[source_modality]
            targets = by_modality[target_modality]
            if not sources or not targets:
                continue
            key = f"{source_modality}->{target_modality}"
            pairs[key] = {
                "n_sources": len(sources),
                "n_targets": len(targets),
                "jspace_retrieval": retrieval_metrics(sources, targets, jspace),
                "jspace_separation": separation_metrics(sources, targets, jspace),
                "jspace_support_overlap": separation_metrics(sources, targets, overlap),
                "raw_residual_retrieval": retrieval_metrics(sources, targets, raw),
                "raw_residual_separation": separation_metrics(sources, targets, raw),
                "shuffled_control": shuffled_label_control(
                    sources, targets, jspace, n_permutations=n_permutations, seed=seed
                ),
            }
    return {
        "conventions": dict(CONVENTIONS),
        "modalities": list(modalities),
        "n_positive_samples": len(positives),
        "pairs": pairs,
        "note": (
            "Retrieval and clustering are representational evidence only. "
            "Causal transfer is tested separately and is not implied by these "
            "numbers."
        ),
    }
