# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""A deterministic synthetic world for the output-convergence audit.

The audit's real path reads a completed run off Drive and loads Gemma once for
its output head. Neither is available on a CPU test runner, and neither should
be needed to know whether the *logic* is right. So this module builds the whole
thing out of small tensors: a completed run directory with the same
:class:`~jlens.mmpilot.store.UnitStore` layout, the same artifact names and the
same verdict shapes, and a tiny output head with Gemma's normalization
convention.

Nothing here loads Gemma, downloads anything, or touches a real run.

What the world is tuned to prove
--------------------------------

The readout signal at each layer is a single knob (:class:`MockWorldSpec`
``layer_strength``), so the audit's three verdict branches can each be reached
on demand:

``pre_convergence``
    Layer 35 carries no candidate signal, layer 40 carries a lot. The readout
    at 35 sits near the six-candidate chance rate and at 40 is essentially
    perfect. With the frozen causal evidence saying SUPPORTED at 35, this is
    the only configuration that may produce
    :data:`~jlens.mmpilot.convergence.PRE_CONVERGENCE_TRANSFER_SUPPORTED`.

``converged_early``
    Every layer carries a lot of signal, so layer 35 clears the converged bar
    and the verdict must be
    :data:`~jlens.mmpilot.convergence.TRANSFER_AT_OR_AFTER_CONVERGENCE` — checked
    first, and therefore not maskable.

``ambiguous``
    Layer 35 lands between the two bars.

``flat_weak``
    **The important one.** Every layer is weak, so layer 35 is NOT_CONVERGED and
    yet no later layer is clearly more converged. This is the configuration that
    proves a weak direct readout *on its own* cannot produce the
    pre-convergence verdict — which is exactly the failure mode a study like
    this invites.

``degenerate``
    The readout answers the same candidate for every sample. A failed readout,
    not a fact about the representation, and it must yield
    :data:`~jlens.mmpilot.convergence.INCONCLUSIVE_CONVERGENCE_TIMING`.

``causal_supported=False`` is orthogonal to all of the above: it rewrites the
synthetic frozen causal evidence to UNSUPPORTED, so a perfectly good
pre-convergence trajectory still cannot carry the claim.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn

from jlens.mmpilot.convergence import (
    AUDITED_LAYERS,
    CONVERGENCE_PROTOCOL,
    NativeHead,
)
from jlens.mmpilot.jspace import CONVENTIONS, tensor_checksum
from jlens.mmpilot.store import RunFingerprint, UnitStore, payload_checksum, safe_key

#: Small enough to be instant, wide enough that six candidate directions are not
#: forced to overlap.
MOCK_D_MODEL = 16
MOCK_VOCAB = 40

#: The behavioral candidate set: six words, exactly as the completed run used.
MOCK_CANDIDATES: tuple[str, ...] = (
    "bus",
    "cat",
    "dog",
    "pizza",
    "toilet",
    "zebra",
)

#: The predeclared focal set. ``zebra`` is kept because the real study kept it:
#: it failed the spoken-audio capability gate and must survive as a labelled
#: diagnostic without touching a principal number.
MOCK_FOCAL_CONCEPTS: tuple[str, ...] = ("cat", "toilet", "zebra")

#: The concept the capability gate rejects, and the channel it fails in.
MOCK_INELIGIBLE_CONCEPT = "zebra"
MOCK_INELIGIBLE_MODALITY = "spoken_audio"

MOCK_MODALITIES: tuple[str, ...] = ("text", "image", "spoken_audio")

MOCK_MODEL_REPO_ID = "mock/gemma-like"
MOCK_MODEL_REVISION = "0" * 40
MOCK_PROCESSOR_REVISION = "0" * 40
MOCK_AUDIO_PROTOCOL_VERSION = "jlens.mmpilot.native_spoken_audio.v1"
MOCK_AUDIO_PROTOCOL_FINGERPRINT = "sha256:mock-audio-protocol"
MOCK_LENS_CHECKSUMS: dict[int, str] = {
    35: "sha256:mock-lens-L35",
    38: "sha256:mock-lens-L38",
    40: "sha256:mock-lens-L40",
}
MOCK_COMBINED_LENS_CHECKSUM = "sha256:mock-lens-combined"
MOCK_MANIFEST_CHECKSUM = "sha256:mock-manifest"
MOCK_SPLIT_ID = "mock-split-0"

#: Layer signal strengths per world mode. A strength of 0 leaves the residual
#: pure noise, so the readout falls to the six-candidate chance rate.
MOCK_MODES: dict[str, dict[int, float]] = {
    "pre_convergence": {35: 0.0, 38: 1.15, 40: 9.0},
    "converged_early": {35: 9.0, 38: 9.0, 40: 9.0},
    "ambiguous": {35: 1.15, 38: 1.3, 40: 1.5},
    "flat_weak": {35: 0.0, 38: 0.0, 40: 0.0},
    "degenerate": {35: 0.0, 38: 0.0, 40: 0.0},
}


class MockRMSNorm(nn.Module):
    """RMSNorm with Gemma's ``(1 + weight)`` convention.

    Written this way on purpose: it lets
    :func:`~jlens.mmpilot.convergence.audit_native_head` prove it can tell the
    two RMSNorm conventions apart on a module it was not told about, which is
    the check standing between this audit and a silently wrong logit lens.
    """

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(d_model))
        self.eps = float(eps)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        normed = hidden * torch.rsqrt(
            hidden.pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        return normed * (1.0 + self.weight)


@dataclass(frozen=True)
class MockWorldSpec:
    """Everything that decides what the synthetic world looks like."""

    mode: str = "pre_convergence"
    seed: int = 20260806
    images_per_concept: int = 5
    softcap: float | None = 30.0
    multi_token_candidate: str | None = None
    causal_supported: bool = True
    layers: tuple[int, ...] = AUDITED_LAYERS
    concepts: tuple[str, ...] = MOCK_FOCAL_CONCEPTS
    candidates: tuple[str, ...] = MOCK_CANDIDATES
    layer_strength: dict[int, float] = field(default_factory=dict)

    def strengths(self) -> dict[int, float]:
        if self.layer_strength:
            return {int(k): float(v) for k, v in self.layer_strength.items()}
        if self.mode not in MOCK_MODES:
            raise ValueError(
                f"unknown mock mode {self.mode!r}; known modes are {sorted(MOCK_MODES)}"
            )
        return dict(MOCK_MODES[self.mode])


def mock_candidate_token_ids(spec: MockWorldSpec) -> dict[str, list[int]]:
    """Candidate token ids, one per candidate unless a multi-token one is asked for.

    ``multi_token_candidate`` gives that candidate a second token so the
    first-token-only diagnostic path can be exercised without pretending a
    hidden state produced a sequence score.
    """
    ids: dict[str, list[int]] = {}
    for index, candidate in enumerate(sorted(spec.candidates)):
        token = 10 + index
        ids[candidate] = (
            [token, 30 + index] if candidate == spec.multi_token_candidate else [token]
        )
    return ids


def build_mock_head(spec: MockWorldSpec) -> tuple[NativeHead, dict[str, int]]:
    """A tiny frozen output head plus the candidate direction each token reads.

    Row ``t`` of the unembedding for candidate ``k``'s first token is the ``k``-th
    basis direction, so a residual pointed at direction ``k`` reads out as
    candidate ``k``. Every other row is small noise, which keeps a full-vocabulary
    argmax meaningful without letting it dominate.
    """
    generator = torch.Generator().manual_seed(spec.seed)
    weight = 0.02 * torch.randn(MOCK_VOCAB, MOCK_D_MODEL, generator=generator)
    token_ids = mock_candidate_token_ids(spec)
    directions: dict[str, int] = {}
    for index, candidate in enumerate(sorted(spec.candidates)):
        first_token = token_ids[candidate][0]
        weight[first_token] = 0.0
        weight[first_token, index] = 4.0
        directions[candidate] = index

    lm_head = nn.Linear(MOCK_D_MODEL, MOCK_VOCAB, bias=False)
    with torch.no_grad():
        lm_head.weight.copy_(weight)
    lm_head.weight.requires_grad_(False)

    norm = MockRMSNorm(MOCK_D_MODEL)
    norm.weight.requires_grad_(False)
    head = NativeHead(
        final_norm=norm,
        lm_head=lm_head,
        softcap=spec.softcap,
        d_model=MOCK_D_MODEL,
        vocab_size=MOCK_VOCAB,
    )
    return head, directions


def mock_activation(
    *,
    spec: MockWorldSpec,
    concept: str,
    directions: Mapping[str, int],
    modality: str,
    layer: int,
    image_index: int,
) -> torch.Tensor:
    """One synthetic clean final-prompt-token residual.

    ``strength * e_concept + noise``. The noise seed folds in the sample's whole
    identity, so the same world always produces the same activation and two
    different samples never accidentally share one.
    """
    if spec.mode == "degenerate":
        # Every sample identical, so the readout answers the same candidate
        # every time. A failed measurement, not a weak representation.
        constant = torch.zeros(MOCK_D_MODEL)
        constant[directions[sorted(spec.candidates)[0]]] = 5.0
        return constant

    key = f"{concept}|{modality}|{layer}|{image_index}|{spec.seed}"
    # Seeded from a checksum rather than ``hash()``: Python's string hash is
    # salted per process, which would make the "deterministic" world different
    # on every run.
    generator = torch.Generator().manual_seed(
        int(payload_checksum(key).split(":")[1][:8], 16)
    )
    noise = torch.randn(MOCK_D_MODEL, generator=generator)
    strength = spec.strengths()[int(layer)]
    activation = noise.clone()
    activation[directions[concept]] += float(strength)
    return activation


def _capability_summary(spec: MockWorldSpec) -> dict:
    """The completed run's capability table: two concepts pass, one fails audio."""
    per_concept: dict[str, dict] = {}
    for concept in spec.concepts:
        per_concept[concept] = {}
        for modality in MOCK_MODALITIES:
            fails = (
                concept == MOCK_INELIGIBLE_CONCEPT
                and modality == MOCK_INELIGIBLE_MODALITY
            )
            n, n_correct = 8, (5 if fails else 8)
            per_concept[concept][modality] = {
                "n": n,
                "n_correct": n_correct,
                "accuracy": n_correct / n,
                "median_target_margin": 1.5,
                "min_target_margin": 0.4,
                "passed": not fails,
            }
    retained = sorted(
        concept
        for concept in spec.concepts
        if all(per_concept[concept][m]["passed"] for m in MOCK_MODALITIES)
    )
    return {
        "threshold": 0.7,
        "modalities_evaluated": list(MOCK_MODALITIES),
        "per_concept": per_concept,
        "retained_concepts": retained,
        "text_image_retained_concepts": sorted(spec.concepts),
        "n_records": len(spec.concepts) * len(MOCK_MODALITIES) * 8,
        "candidate_token_ids": mock_candidate_token_ids(spec),
        "question": "Question: which one of these is present: "
        + ", ".join(sorted(spec.candidates))
        + "? Answer with exactly one word.\nAnswer:",
    }


def _causal_cell(
    *, layer: int, concept: str, pair: str, supported: bool, admissible: bool
) -> dict:
    """One synthetic image-level causal cell, in the completed run's shape."""
    effect = 0.62 if supported else 0.02
    return {
        "layer": int(layer),
        "concept": concept,
        "pair": pair,
        "audio_related": "spoken_audio" in pair,
        "evaluated": True,
        "execution_status": "measured",
        "alpha": 1.0,
        "mean_signed_target_effect": effect,
        "fraction_expected_sign": 1.0 if supported else 0.5,
        "mean_activation_norm_ratio": 1.04,
        "mean_abs_unrelated_change": 0.08,
        "n_distinct_images": 6,
        "n_positive_images": 3,
        "n_negative_images": 3,
        "meets_claim_image_floor": bool(supported),
        "random_control": 0.05,
        "unrelated_control": 0.04,
        "raw_residual_control": 0.10,
        "jspace_beats_raw_direction": True,
        "passes": bool(supported),
        "reasons": [] if supported else ["effect +0.0200 is not positive enough"],
        "capability_admissible": bool(admissible),
        "capability_label": "CAPABILITY_ELIGIBLE" if admissible else "CAPABILITY_INELIGIBLE",
        "capability_rejection_reason": (
            None
            if admissible
            else f"{concept}: spoken_audio capability 5/8 = 62.5% < 70.0%"
        ),
        "counted_toward_verdict": bool(supported and admissible),
    }


def _causal_verdict(spec: MockWorldSpec, *, layer: int, name: str) -> dict:
    """One layer's synthetic causal verdict, read (not recomputed) by the audit."""
    supported = bool(spec.causal_supported)
    admissible = [c for c in spec.concepts if c != MOCK_INELIGIBLE_CONCEPT]
    pairs = (
        "text->spoken_audio",
        "spoken_audio->text",
        "image->spoken_audio",
        "spoken_audio->image",
    )
    cells = [
        _causal_cell(
            layer=layer,
            concept=concept,
            pair=pair,
            supported=supported,
            admissible=concept != MOCK_INELIGIBLE_CONCEPT,
        )
        for concept in spec.concepts
        for pair in pairs
    ]
    supporting = [
        {"concept": cell["concept"], "pair": cell["pair"]}
        for cell in cells
        if cell["counted_toward_verdict"] and cell["meets_claim_image_floor"]
    ]
    return {
        "schema": "jlens.mmpilot.causal_transfer_verdict.v2",
        "name": name,
        "verdict": "SUPPORTED" if supported else "UNSUPPORTED",
        "rationale": (
            f"{len(supporting)} capability-admissible audio-related cells at "
            f"layer {layer} cleared their controls"
            if supported
            else f"no admissible audio-related cell at layer {layer} cleared its controls"
        ),
        "layer": int(layer),
        "focal_concepts": list(spec.concepts),
        "cells": cells,
        "audio_cells_supporting_a_claim": supporting,
        "audio_cells_passing_admissible": supporting,
        "audio_cells_measured_but_inadmissible": [
            {"concept": cell["concept"], "pair": cell["pair"]}
            for cell in cells
            if not cell["capability_admissible"]
        ],
        "capability_admissibility": {
            "eligible_concepts": admissible,
            "excluded_concept_names": [MOCK_INELIGIBLE_CONCEPT],
        },
    }


def build_mock_completed_run(
    run_dir: str | Path, spec: MockWorldSpec | None = None
) -> dict:
    """Write a synthetic completed three-modality run and return what to expect of it.

    The directory that comes out is byte-shaped like the real one: a
    :class:`~jlens.mmpilot.store.UnitStore` with ``activation``, ``intervention``
    and ``metric`` units, the run's own report and summary, and the
    capability-filtered amended pair. The audit then reads it with exactly the
    code it uses on Drive.
    """
    spec = spec or MockWorldSpec()
    root = Path(run_dir)
    head, directions = build_mock_head(spec)
    capability = _capability_summary(spec)

    fingerprint = RunFingerprint(
        mode="mock",
        model_repo_id=MOCK_MODEL_REPO_ID,
        model_revision=MOCK_MODEL_REVISION,
        processor_revision=MOCK_PROCESSOR_REVISION,
        layers=tuple(int(x) for x in spec.layers),
        lens_checksum=MOCK_COMBINED_LENS_CHECKSUM,
        manifest_checksum=MOCK_MANIFEST_CHECKSUM,
        split_id=MOCK_SPLIT_ID,
        intervention_config={"alphas": [0.0, 0.5, 1.0], "primary_causal_layer": 35},
        extra={
            "protocol": "mmpilot.tri_modal_transfer_verdict.v1",
            "modalities": list(MOCK_MODALITIES),
            "audio_protocol_version": MOCK_AUDIO_PROTOCOL_VERSION,
            "audio_protocol_fingerprint": MOCK_AUDIO_PROTOCOL_FINGERPRINT,
            "per_layer_lens_checksums": {
                str(layer): MOCK_LENS_CHECKSUMS[layer] for layer in spec.layers
            },
        },
    )
    store = UnitStore(root, fingerprint)
    store.open()

    for concept in spec.concepts:
        for image_index in range(spec.images_per_concept):
            group_id = f"grp-{concept}-{image_index:02d}"
            image_id = f"img-{concept}-{image_index:02d}"
            split = "test" if image_index % 2 == 0 else "train"
            for modality in MOCK_MODALITIES:
                identifier = f"{group_id}::{modality}"
                for layer in spec.layers:
                    activation = mock_activation(
                        spec=spec,
                        concept=concept,
                        directions=directions,
                        modality=modality,
                        layer=int(layer),
                        image_index=image_index,
                    )
                    store.save(
                        "activation",
                        safe_key(identifier, f"L{int(layer)}"),
                        {
                            "sample_id": identifier,
                            "group_id": group_id,
                            "image_id": image_id,
                            "concept": concept,
                            "modality": modality,
                            "split": split,
                            "layer": int(layer),
                            "hook_site": CONVENTIONS["hook_site"],
                            "position": CONVENTIONS["position"],
                            "shape": list(activation.shape),
                            "norm": float(activation.norm()),
                            "activation_checksum": tensor_checksum(activation),
                            "activation": [float(x) for x in activation.tolist()],
                            "prompt_hash": f"sha256:mock-{identifier}",
                            "media_checksum": f"sha256:mock-media-{image_id}",
                            "prompt_len": 32,
                            "model_revision": MOCK_MODEL_REVISION,
                        },
                    )
                # The zero-alpha control unit: the clean, unedited run. Its
                # ``clean_prediction`` is the reference answer the audit's
                # agreement metric is measured against.
                store.save(
                    "intervention",
                    safe_key(concept, "zero", f"L{int(spec.layers[0])}", identifier),
                    {
                        "sample_id": identifier,
                        "group_id": group_id,
                        "image_id": image_id,
                        "concept": concept,
                        "source_modality": "text",
                        "target_modality": modality,
                        "layer": int(spec.layers[0]),
                        "control_kind": "zero",
                        "alpha": 0.0,
                        "clean_prediction": concept,
                        "prediction": concept,
                        "prediction_changed": False,
                        "activation_norm_ratio": 1.0,
                    },
                )

    store.save("metric", "capability_summary", capability)
    store.save(
        "metric",
        "audio_capability_verdict",
        {
            "verdict": "AUDIO_CAPABILITY_GO",
            "per_concept": capability["per_concept"],
            "threshold": capability["threshold"],
            "retained_concepts": capability["retained_concepts"],
            "concepts_passing_all_three": capability["retained_concepts"],
        },
    )

    verdicts = {
        "A_audio_capability": {
            "verdict": "AUDIO_CAPABILITY_GO",
            "per_concept": capability["per_concept"],
            "threshold": capability["threshold"],
            "retained_concepts": capability["retained_concepts"],
        },
        "B_representational_transfer": {"verdict": "SUPPORTED"},
        "C_primary_causal": _causal_verdict(spec, layer=35, name="L35_CAUSAL_TRANSFER"),
        "D_replication": {
            "verdict": "SUPPORTED" if spec.causal_supported else "UNSUPPORTED",
            "per_layer": {
                str(layer): _causal_verdict(
                    spec, layer=int(layer), name=f"L{int(layer)}_CAUSAL_TRANSFER"
                )
                for layer in spec.layers
                if int(layer) != 35
            },
        },
        "E_overall": {
            "verdict": (
                "THREE_MODALITY_GO" if spec.causal_supported else "THREE_MODALITY_NO_GO"
            ),
            "rationale": "synthetic completed-run verdict for the MOCK audit",
        },
    }
    summary = {
        "run_dir": str(root),
        "mode": "mock",
        "commit": "mock",
        "fingerprint_digest": fingerprint.digest,
        "lens_validation": {
            "layers": [int(x) for x in spec.layers],
            "checksums": {str(k): v for k, v in sorted(MOCK_LENS_CHECKSUMS.items())},
            "combined_checksum": MOCK_COMBINED_LENS_CHECKSUM,
            "confirmation_status": {
                "35": "PASS",
                "38": "PASS",
                "40": "PASS",
                "32": "FAILED_UNTOUCHED_CONFIRMATION",
            },
        },
        "audio_protocol": {
            "protocol_version": MOCK_AUDIO_PROTOCOL_VERSION,
            "protocol_fingerprint": MOCK_AUDIO_PROTOCOL_FINGERPRINT,
        },
        "verdicts": verdicts,
        "resume": store.status_report(),
    }
    for name, payload in (
        ("native_audio_transfer_summary.json", summary),
        ("native_audio_transfer_summary_capability_filtered_v2.json", {
            "schema": "jlens.mmpilot.amended_report_summary.v1",
            "binding": {"run_fingerprint_digest": fingerprint.digest},
            "primary_layer": 35,
            "replication_layers": [
                int(x) for x in spec.layers if int(x) != 35
            ],
            "focal_concepts": list(spec.concepts),
            "verdicts": verdicts,
        }),
        ("run_manifest.json", {
            "run_dir": str(root),
            "mode": "mock",
            "commit": "mock",
            "fingerprint_digest": fingerprint.digest,
        }),
    ):
        (root / name).write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )
    for name in (
        "native_audio_transfer_report.md",
        "native_audio_transfer_report_capability_filtered_v2.md",
    ):
        (root / name).write_text(
            f"# Synthetic completed run ({name})\n\n"
            "Written by jlens.mmpilot.convergence_mock. The audit must never "
            "modify this file.\n",
            encoding="utf-8",
        )

    return {
        "run_dir": str(root),
        "spec": spec,
        "store": store,
        "fingerprint": fingerprint,
        "head": head,
        "directions": directions,
        "capability": capability,
        "summary": summary,
        "candidate_token_ids": mock_candidate_token_ids(spec),
        "expectations": {
            "expected_fingerprint_digest": fingerprint.digest,
            "expected_model_repo_id": MOCK_MODEL_REPO_ID,
            "expected_model_revision": MOCK_MODEL_REVISION,
            "expected_processor_revision": MOCK_PROCESSOR_REVISION,
            "expected_audio_protocol_version": MOCK_AUDIO_PROTOCOL_VERSION,
            "expected_audio_protocol_fingerprint": MOCK_AUDIO_PROTOCOL_FINGERPRINT,
            "expected_lens_checksums": {
                int(layer): MOCK_LENS_CHECKSUMS[int(layer)] for layer in spec.layers
            },
            "expected_combined_lens_checksum": MOCK_COMBINED_LENS_CHECKSUM,
        },
    }


def run_mock_convergence_audit(
    completed_run_dir: str | Path,
    audit_dir: str | Path,
    *,
    spec: MockWorldSpec | None = None,
    run_probe: bool = True,
    bootstrap_resamples: int = 400,
) -> dict:
    """Build (or reuse) the synthetic run and audit it end to end, no model loaded.

    This is what the notebook's MOCK path calls and what the tests assert on, so
    the two cannot drift. Calling it twice on the same ``audit_dir`` exercises
    resume.

    ``bootstrap_resamples`` is lowered from the real default only to keep the
    CPU test fast; the criterion is otherwise the production one, untouched.
    """
    from jlens.mmpilot.convergence import (
        CONVERGENCE_CRITERION,
        ConvergenceFingerprint,
        ConvergenceStore,
        assert_run_unchanged,
        audit_native_head,
        build_population,
        clean_predictions_from_interventions,
        protected_file_checksums,
        resolve_candidate_tokens,
        run_convergence_audit,
        verify_completed_run,
    )

    spec = spec or MockWorldSpec()
    completed_run_dir = Path(completed_run_dir)
    # Idempotent: the builder writes the same bytes for the same spec, so a
    # second call reopens the run rather than producing a different one.
    built = build_mock_completed_run(completed_run_dir, spec)
    store = built["store"]

    before = protected_file_checksums(completed_run_dir)
    integrity = verify_completed_run(
        run_dir=completed_run_dir,
        fingerprint_payload=json.loads(
            (completed_run_dir / "fingerprint.json").read_text(encoding="utf-8")
        ),
        summary=built["summary"],
        layers=spec.layers,
        **built["expectations"],
    )

    tokenization = resolve_candidate_tokens(built["candidate_token_ids"])
    head_audit = audit_native_head(built["head"])
    population = build_population(
        activations=list(store.load_all("activation").values()),
        clean_predictions=clean_predictions_from_interventions(
            store.load_all("intervention").values()
        ),
        capability=built["capability"],
        focal_concepts=spec.concepts,
        layers=spec.layers,
    )

    criterion = CONVERGENCE_CRITERION
    if bootstrap_resamples != criterion.bootstrap_resamples:
        criterion = type(criterion)(
            **{
                **criterion.to_dict(),
                "required_modalities": criterion.required_modalities,
                "bootstrap_resamples": int(bootstrap_resamples),
            }
        )

    audit_store = ConvergenceStore(
        audit_dir,
        ConvergenceFingerprint(
            protocol=CONVERGENCE_PROTOCOL,
            completed_run_fingerprint_digest=built["fingerprint"].digest,
            completed_run_dir=str(completed_run_dir),
            model_repo_id=MOCK_MODEL_REPO_ID,
            model_revision=MOCK_MODEL_REVISION,
            processor_revision=MOCK_PROCESSOR_REVISION,
            layers=tuple(int(x) for x in spec.layers),
            candidate_digest=tokenization["digest"],
            readout_mode=tokenization["readout_mode"],
            head_checksum=str(head_audit["head_checksum"]),
            criterion_digest=criterion.digest,
            code_version=CONVERGENCE_PROTOCOL,
            extra={"mode": "mock", "world_mode": spec.mode},
        ),
    )
    audit_store.open()

    integrity["immutability"] = {"checksums": dict(before)}
    result = run_convergence_audit(
        population=population,
        head=built["head"],
        tokenization=tokenization,
        head_audit=head_audit,
        integrity=integrity,
        completed_summary=built["summary"],
        store=audit_store,
        criterion=criterion,
        layers=spec.layers,
        run_probe=run_probe,
    )
    result["immutability"] = assert_run_unchanged(
        before, protected_file_checksums(completed_run_dir)
    )
    result["integrity"] = integrity
    result["population"] = population
    result["tokenization"] = tokenization
    result["head_audit"] = head_audit
    result["store"] = audit_store
    result["completed_run_dir"] = str(completed_run_dir)
    return result


def mock_verdict_matrix(
    tmp_root: str | Path,
    *,
    modes: Sequence[str] = tuple(MOCK_MODES),
    bootstrap_resamples: int = 200,
) -> dict[str, str]:
    """Run every world mode and return ``{mode: verdict}``.

    The notebook prints this so a reader can see the branches being exercised
    rather than being told they are.
    """
    root = Path(tmp_root)
    out: dict[str, str] = {}
    for mode in modes:
        spec = MockWorldSpec(mode=mode)
        result = run_mock_convergence_audit(
            root / f"completed-{mode}",
            root / f"audit-{mode}",
            spec=spec,
            run_probe=False,
            bootstrap_resamples=bootstrap_resamples,
        )
        out[mode] = result["verdict"]["verdict"]
    spec = MockWorldSpec(mode="pre_convergence", causal_supported=False)
    result = run_mock_convergence_audit(
        root / "completed-no-causal",
        root / "audit-no-causal",
        spec=spec,
        run_probe=False,
        bootstrap_resamples=bootstrap_resamples,
    )
    out["pre_convergence_without_causal_support"] = result["verdict"]["verdict"]
    return out


__all__ = [
    "MOCK_CANDIDATES",
    "MOCK_D_MODEL",
    "MOCK_FOCAL_CONCEPTS",
    "MOCK_INELIGIBLE_CONCEPT",
    "MOCK_MODALITIES",
    "MOCK_MODES",
    "MOCK_VOCAB",
    "MockRMSNorm",
    "MockWorldSpec",
    "build_mock_completed_run",
    "build_mock_head",
    "mock_activation",
    "mock_candidate_token_ids",
    "mock_verdict_matrix",
    "run_mock_convergence_audit",
]
