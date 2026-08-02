# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Native J-lens readout scoring and the layer-32 confirmatory validation.

The *native readout* asks the only question a lens has to answer on its own
terms: transport the residual at a fitted source layer with ``J_l``, unembed
it, and see whether the resulting distribution agrees with what the model
actually predicts at the final layer. Nothing here fits a lens — every entry
point consumes a frozen :class:`~jlens.lens.JacobianLens` and refuses to
modify it.

Why this module exists
----------------------

The v2 early-layer recalibration ran a native readout inline in its notebook
and published ``lens.validated.pt``. Layer 32 scored a perfect median rank of
1.0 and an MRR of 1.0 on eight held-out prompts, yet the run's conjunction
criterion failed it: mean top-10 overlap (0.113) did not beat the wrong-layer
control (0.150). Eight prompts is too small a sample to settle that, and
relabelling layer 32 as validated after the fact would be exactly the move the
original criterion existed to prevent.

So layer 32 gets a *new* test, specified before it runs: more held-out
prompts, primary metrics fixed in advance, and top-10 overlap demoted to a
reported secondary that cannot veto a primary-metric pass. Overlap is a
similarity statistic over a 262k-token vocabulary whose tail is dominated by
near-ties; it is informative, but it is not the claim under test. The claim is
"the layer-32 J-lens reads out what the model actually predicts", and that is
what top-1 agreement, MRR and median rank measure directly.

The scoring logic is shared with the original notebook's protocol rather than
re-derived, so a confirmatory number and a v2 number mean the same thing.

Predeclared criterion
---------------------

:data:`CONFIRMATORY_CRITERION` and :data:`CRITERION_TEXT` state the pass rule.
They are module constants so the notebook can print the criterion before any
result-producing cell runs, and so a criterion edit changes the resume
fingerprint and forces a recomputation instead of silently rescoring old rows.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

# Only pure serialisation helpers are reused from the multimodal pilot package
# (canonical JSON, payload checksums, filesystem-safe keys, and its resume
# error type). No SpokenCOCO or multimodal *data* is touched by this module —
# see :func:`verify_saved_lens`, which refuses a non-text calibration.
from jlens.mmpilot.store import (
    IncompatibleStateError,
    canonical_json,
    payload_checksum,
    safe_key,
)

__all__ = [
    "CONFIRMATORY_CRITERION",
    "CONFIRMATORY_LAYER",
    "CONFIRMATORY_PROMPT_SEED",
    "CONFIRMATORY_PROTOCOL",
    "CONTROL_SEED",
    "CONTROL_VARIANTS",
    "CRITERION_TEXT",
    "DIAGNOSTIC_VARIANTS",
    "N_CONFIRMATORY_PROMPTS",
    "PRIMARY_METRICS",
    "SECONDARY_METRICS",
    "VERDICT_NO_GO",
    "VERDICT_VALIDATED",
    "ConfirmatoryCriterion",
    "ConfirmatoryFingerprint",
    "ConfirmatoryStore",
    "IncompatibleStateError",
    "PromptOverlapError",
    "build_readout_variants",
    "confirmatory_report_markdown",
    "evaluate_confirmatory",
    "excluded_prompt_hashes",
    "native_readout_row",
    "prompt_sha256",
    "select_confirmatory_prompts",
    "summarize_rows",
    "summarize_variant",
    "verify_saved_lens",
]

#: The one layer this workflow evaluates. Layer 38 already passed the v2 gate
#: and is not retested here; layers 20 and 26 failed outright.
CONFIRMATORY_LAYER = 32

#: Held-out prompts in the confirmatory set. Four times the v2 sample.
N_CONFIRMATORY_PROMPTS = 32

#: Seed governing which eligible WikiText passages become the confirmatory
#: held-out set. Fixed and recorded in the prompt manifest; it participates in
#: the resume fingerprint, so changing it invalidates stored results.
CONFIRMATORY_PROMPT_SEED = 20260802

#: Seed for the randomised control lenses. Matches the v2 run's
#: ``eval.control_seed`` so a control here is the same construction as there.
CONTROL_SEED = 1234

#: Protocol tag written into every artifact. Distinct from the v2 tag: these
#: results must never be mistaken for the original validation's.
CONFIRMATORY_PROTOCOL = "layer32-confirmatory-native-jlens-readout-v1"

#: Randomised or mismatched J-lens controls. A pass must beat all of them.
CONTROL_VARIANTS = ("permuted", "random", "wrong_layer")

#: Reported for context but never blocking: the ordinary logit lens is neither
#: randomised nor wrong-layer, so it cannot answer "would any matrix do this?".
#: A logit lens that reads out well at layer 32 is a fact about the residual
#: stream, not evidence against the J-lens.
DIAGNOSTIC_VARIANTS = ("logit_lens",)

#: Every variant scored, in report order.
READOUT_VARIANTS = ("j_lens", *CONTROL_VARIANTS, *DIAGNOSTIC_VARIANTS)

#: Metrics that can pass or fail the run.
PRIMARY_METRICS = ("top1_agreement", "mean_reciprocal_rank", "median_target_rank")

#: Reported, never blocking.
SECONDARY_METRICS = ("mean_top10_overlap",)

VERDICT_VALIDATED = "VALIDATED_FOR_MULTIMODAL_FOLLOWUP"
VERDICT_NO_GO = "LAYER32_CONFIRMATORY_NO_GO"


class PromptOverlapError(RuntimeError):
    """Raised when a candidate held-out prompt was already used to fit the
    lens or to run the original validation. Never downgraded to a warning:
    an overlapping prompt turns a held-out test into a training-set readout."""


# --------------------------------------------------------------------- criterion


@dataclass(frozen=True)
class ConfirmatoryCriterion:
    """The pass rule, fixed before results exist.

    Attributes:
        n_prompts: Exact number of held-out prompts required. A short run is
            not a weaker pass; it is not a result.
        min_top1_agreement: Floor on the J-lens's top-1 agreement with the
            model's own final-layer argmax.
        control_top1_margin: How far every control's top-1 agreement must sit
            *below* the J-lens's before the controls count as beaten. Without
            a margin, "0.75 versus 0.74" would read as a pass.
        min_distinct_target_tokens: Degeneracy floor. If every prompt's target
            is the same token, agreement measures the token, not the lens.
        controls: Control variants that must all be beaten.
    """

    n_prompts: int = N_CONFIRMATORY_PROMPTS
    min_top1_agreement: float = 0.75
    control_top1_margin: float = 0.25
    min_distinct_target_tokens: int = 2
    controls: tuple[str, ...] = CONTROL_VARIANTS
    primary_metrics: tuple[str, ...] = PRIMARY_METRICS
    secondary_metrics: tuple[str, ...] = SECONDARY_METRICS

    def to_dict(self) -> dict:
        payload = asdict(self)
        for key in ("controls", "primary_metrics", "secondary_metrics"):
            payload[key] = list(payload[key])
        return payload

    @property
    def digest(self) -> str:
        """Checksum of the criterion, bound into the resume fingerprint."""
        return payload_checksum(self.to_dict())


#: The single criterion this workflow runs under.
CONFIRMATORY_CRITERION = ConfirmatoryCriterion()

#: Human-readable statement of the criterion, printed before any results.
CRITERION_TEXT = f"""\
PREDECLARED PASS CRITERION — layer {CONFIRMATORY_LAYER} confirmatory native readout
Fixed before any result-producing cell runs. Not revisable after seeing results.

Primary metrics (can pass or fail the run):
  1. J-lens top-1 agreement with the model's actual final-layer top-1 token
     is at least {CONFIRMATORY_CRITERION.min_top1_agreement:.2f}, over exactly
     {CONFIRMATORY_CRITERION.n_prompts} held-out prompts.
  2. J-lens mean reciprocal rank of the actual top-1 token strictly exceeds
     the row-permuted, norm-matched random, and wrong-layer controls.
  3. J-lens median rank of the actual top-1 token is strictly better (lower)
     than all three controls.
  4. No control reaches comparable top-1 agreement: every control's top-1
     agreement is at most the J-lens's minus {CONFIRMATORY_CRITERION.control_top1_margin:.2f}.
  5. All reported values are finite, every variant is scored on all
     {CONFIRMATORY_CRITERION.n_prompts} prompts, and the targets span at least
     {CONFIRMATORY_CRITERION.min_distinct_target_tokens} distinct tokens.

Secondary metric (reported, never blocking):
  - mean top-10 overlap with the model's final top-10.

The ordinary logit lens is reported as a diagnostic only. It is not a
randomised or wrong-layer J-lens control and cannot block a pass.

Verdict is exactly one of:
  {VERDICT_VALIDATED}
  {VERDICT_NO_GO}
"""


# ----------------------------------------------------------------------- prompts


def prompt_sha256(prompt: str) -> str:
    """Full sha256 hex of a rendered prompt.

    The v2 run wrote full digests into ``prompt_metadata.json``; overlap
    checks compare against those, so this must stay untruncated.
    """
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def excluded_prompt_hashes(prompt_metadata: dict) -> dict[str, str]:
    """Every prompt hash the confirmatory set must avoid, mapped to its role.

    Reads the v2 run's ``prompt_metadata.json``: ``fit_hashes`` are the 32
    prompts the lens was fitted on, ``heldout_hashes`` the 8 the original
    validation used. Both are disqualifying — refitting is not the only way to
    leak, and reusing the original held-out prompts would make the
    confirmatory run a rerun rather than an independent test.
    """
    excluded: dict[str, str] = {}
    for key, role in (("fit_hashes", "v2_fitting"), ("heldout_hashes", "v2_validation")):
        for digest in prompt_metadata.get(key, ()):
            excluded[str(digest)] = role
    if not excluded:
        raise ValueError(
            "prompt metadata carries neither fit_hashes nor heldout_hashes; "
            "cannot prove the confirmatory prompts are new"
        )
    return excluded


def _overlap_role(digest: str, excluded: dict[str, str]) -> str | None:
    """Role of ``digest`` in the excluded set, matching truncated digests too.

    ``jlens.metadata.prompt_hashes`` writes 16-character digests while the v2
    notebook wrote full ones. Comparing prefixes as well means a manifest in
    either form is still caught.
    """
    if digest in excluded:
        return excluded[digest]
    for stored, role in excluded.items():
        if stored and (digest.startswith(stored) or stored.startswith(digest)):
            return role
    return None


def select_confirmatory_prompts(
    pool: Sequence[str],
    *,
    n_prompts: int = N_CONFIRMATORY_PROMPTS,
    excluded: dict[str, str],
    seed: int = CONFIRMATORY_PROMPT_SEED,
) -> tuple[list[str], dict]:
    """Choose ``n_prompts`` genuinely new held-out prompts from ``pool``.

    ``pool`` is the rendered candidate list in its fixed source order. Every
    candidate whose hash appears in ``excluded`` is dropped, the survivors are
    shuffled with ``seed``, and the first ``n_prompts`` are taken. Selection is
    a pure function of ``(pool, excluded, seed, n_prompts)``.

    Returns the prompts and a manifest recording the seed, the pool size, what
    was dropped, and every selected hash.

    Raises:
        PromptOverlapError: If the eligible pool is smaller than ``n_prompts``,
            or if a selected prompt still collides with an excluded hash. The
            second check is redundant by construction and kept anyway: it is
            the assertion that actually protects the claim.
    """
    if n_prompts <= 0:
        raise ValueError(f"n_prompts must be positive, got {n_prompts}")

    hashed = [(prompt, prompt_sha256(prompt)) for prompt in pool]
    eligible: list[tuple[str, str]] = []
    dropped: list[dict] = []
    for index, (prompt, digest) in enumerate(hashed):
        role = _overlap_role(digest, excluded)
        if role is None:
            eligible.append((prompt, digest))
        else:
            dropped.append({"pool_index": index, "prompt_sha256": digest, "role": role})

    if len(eligible) < n_prompts:
        raise PromptOverlapError(
            f"only {len(eligible)} of {len(pool)} candidate prompts are new "
            f"({len(dropped)} overlap the v2 fitting/validation sets); "
            f"{n_prompts} are required. Widen the candidate pool."
        )

    order = torch.randperm(len(eligible), generator=torch.Generator().manual_seed(seed))
    selected = [eligible[int(i)] for i in order[:n_prompts]]

    for _prompt, digest in selected:
        role = _overlap_role(digest, excluded)
        if role is not None:
            raise PromptOverlapError(
                f"selected prompt {digest[:16]} overlaps the {role} set; refusing to "
                "report a held-out result on a prompt the lens has already seen"
            )

    digests = [digest for _, digest in selected]
    if len(set(digests)) != len(digests):
        raise PromptOverlapError(
            "the confirmatory set contains duplicate prompts; 32 copies of one "
            "passage is not 32 held-out prompts"
        )

    manifest = {
        "protocol": CONFIRMATORY_PROTOCOL,
        "seed": seed,
        "n_prompts": n_prompts,
        "pool_size": len(pool),
        "eligible_after_exclusion": len(eligible),
        "n_excluded": len(dropped),
        "excluded": dropped,
        "selection_rule": (
            "drop every candidate whose sha256 appears in the v2 fitting or v2 "
            "validation sets, shuffle the survivors with the fixed seed, take "
            "the first n_prompts"
        ),
        "prompts": [
            {"index": i, "prompt_sha256": digest, "n_chars": len(prompt)}
            for i, (prompt, digest) in enumerate(selected)
        ],
    }
    return [prompt for prompt, _ in selected], manifest


# ---------------------------------------------------------------- lens verification


def verify_saved_lens(
    lens: Any,
    *,
    lens_path: str | os.PathLike[str],
    expected_checksum: str,
    manifest: dict,
    expected_model_repo_id: str,
    expected_model_revision: str,
    expected_d_model: int,
    expected_hook_site: str,
    layer: int = CONFIRMATORY_LAYER,
) -> dict:
    """Check every precondition before the confirmatory run may proceed.

    Verifies the artifact's bytes, the model it was fitted against, the
    residual width, the hook site, that ``layer`` is actually fitted, that the
    calibration was text-only, and that the original validation manifest is
    present and consistent. Returns a serialisable record of what was
    observed; raises :class:`ValueError` on the first mismatch.

    The lens is only read. Nothing in this function writes to ``lens_path``.
    """
    from jlens.metadata import file_sha256

    observed_checksum = file_sha256(str(lens_path))
    checks: list[dict] = []

    def require(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(ok), "detail": detail})
        if not ok:
            raise ValueError(f"{name} failed: {detail}")

    require(
        "lens_checksum",
        observed_checksum == expected_checksum,
        f"expected {expected_checksum}, file is {observed_checksum}",
    )
    require(
        "manifest_checksum_agrees",
        manifest.get("lens_checksum") == expected_checksum,
        f"manifest records {manifest.get('lens_checksum')!r}",
    )
    require(
        "manifest_status_is_text_only",
        manifest.get("status") == "validated_text_only",
        f"manifest status is {manifest.get('status')!r}, expected 'validated_text_only'",
    )
    protocol = str(manifest.get("prompt_protocol", ""))
    require(
        "calibration_modality_is_text",
        "text-only" in protocol,
        f"manifest prompt_protocol {protocol!r} does not declare a text-only calibration",
    )
    require(
        "no_multimodal_calibration_data",
        not any(
            marker in protocol.lower()
            for marker in ("spokencoco", "coco", "image", "audio", "multimodal")
        ),
        f"manifest prompt_protocol {protocol!r} references non-text data",
    )
    require(
        "model_revision",
        manifest.get("model_revision") == expected_model_revision,
        f"manifest records {manifest.get('model_revision')!r}, expected {expected_model_revision!r}",
    )
    require(
        "layer_is_fitted",
        layer in list(lens.source_layers),
        f"layer {layer} is not among the fitted layers {list(lens.source_layers)}",
    )
    require(
        "layer_in_manifest",
        layer in list(manifest.get("source_layers", ())),
        f"layer {layer} is not among the manifest's layers {manifest.get('source_layers')}",
    )
    require(
        "hidden_dimension",
        int(lens.d_model) == int(expected_d_model),
        f"lens d_model is {lens.d_model}, expected {expected_d_model}",
    )
    require(
        "hook_site",
        expected_hook_site == "block_output",
        f"hook site is {expected_hook_site!r}; the lens was fitted at block_output",
    )
    require(
        "original_native_validation_recorded",
        bool(manifest.get("native_validation_path")),
        "manifest does not point at the original native_readout_validation.json",
    )

    return {
        "lens_path": str(lens_path),
        "lens_checksum": observed_checksum,
        "model_repo_id": expected_model_repo_id,
        "model_revision": expected_model_revision,
        "d_model": int(lens.d_model),
        "hook_site": expected_hook_site,
        "fitted_source_layers": list(lens.source_layers),
        "layer_under_test": layer,
        "calibration_protocol": protocol,
        "n_fitting_prompts": int(getattr(lens, "n_prompts", 0)),
        "checks": checks,
    }


# ------------------------------------------------------------------------ scoring


def build_readout_variants(lens: Any, *, seed: int = CONTROL_SEED) -> dict[str, Any]:
    """The J-lens plus its three randomised/mismatched controls.

    Constructed exactly as the v2 run constructed them, from the same frozen
    lens and the same seed, so a control number here is comparable to a
    control number there. ``wrong_layer`` is the historical cyclic shift the
    v2 run used; :func:`jlens.controls.wrong_layer_mapping` records which
    fitted Jacobian ends up applied at layer 32.
    """
    from jlens.controls import control_lens, wrong_layer_lens

    return {
        "j_lens": lens,
        "permuted": control_lens(lens, "permuted", seed=seed),
        "random": control_lens(lens, "random", seed=seed),
        "wrong_layer": wrong_layer_lens(lens),
    }


def native_readout_row(
    *,
    sample_index: int,
    prompt_sha: str,
    layer: int,
    variant: str,
    variant_logits: torch.Tensor,
    actual_logits: torch.Tensor,
    top_k: int = 10,
) -> dict:
    """Score one (prompt, variant) pair against the model's own prediction.

    ``actual_logits`` are the model's real final-layer logits at the last
    position; ``variant_logits`` are what the variant reads out from the
    layer-``layer`` residual at that same position. The target is the model's
    actual argmax — the lens is judged against the model, not against a label.

    Rank is 1-based: rank 1 means the variant's argmax *is* the model's.
    """
    actual = actual_logits.detach().float().flatten().cpu()
    scores = variant_logits.detach().float().flatten().cpu()
    if scores.shape != actual.shape:
        raise ValueError(
            f"variant logits {tuple(scores.shape)} do not match actual logits "
            f"{tuple(actual.shape)}"
        )

    target = int(actual.argmax())
    rank = int((scores > scores[target]).sum()) + 1
    actual_top = set(actual.topk(top_k).indices.tolist())
    variant_top = set(scores.topk(top_k).indices.tolist())
    return {
        "sample": int(sample_index),
        "prompt_sha256": prompt_sha,
        "layer": int(layer),
        "variant": variant,
        "target_token_id": target,
        "predicted_token_id": int(scores.argmax()),
        "top1_agreement": bool(int(scores.argmax()) == target),
        "target_rank": rank,
        "reciprocal_rank": 1.0 / rank,
        "top10_overlap": len(actual_top & variant_top) / float(top_k),
    }


def summarize_variant(rows: Sequence[dict]) -> dict:
    """Aggregate one variant's rows into the primary and secondary metrics."""
    if not rows:
        raise ValueError("cannot summarize an empty row set")
    return {
        "n_prompts": len(rows),
        "top1_agreement": statistics.fmean(
            1.0 if row["top1_agreement"] else 0.0 for row in rows
        ),
        "mean_reciprocal_rank": statistics.fmean(row["reciprocal_rank"] for row in rows),
        "median_target_rank": float(statistics.median(row["target_rank"] for row in rows)),
        "mean_top10_overlap": statistics.fmean(row["top10_overlap"] for row in rows),
    }


def summarize_rows(
    rows: Iterable[dict], *, variants: Sequence[str] = READOUT_VARIANTS
) -> dict[str, dict]:
    """Per-variant metrics for every variant in ``variants``."""
    rows = list(rows)
    metrics: dict[str, dict] = {}
    for variant in variants:
        selected = [row for row in rows if row["variant"] == variant]
        if not selected:
            raise ValueError(f"no rows for variant {variant!r}")
        metrics[variant] = summarize_variant(selected)
    return metrics


# ---------------------------------------------------------------------- verdict


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def evaluate_confirmatory(
    rows: Sequence[dict],
    *,
    criterion: ConfirmatoryCriterion = CONFIRMATORY_CRITERION,
    layer: int = CONFIRMATORY_LAYER,
) -> dict:
    """Apply the predeclared criterion and return the verdict.

    Every check is reported with its own pass flag so a failure names the
    clause that failed. Top-10 overlap appears in the metrics and in the
    report but in no check: a secondary metric cannot veto a primary pass.
    """
    rows = [row for row in rows if int(row["layer"]) == int(layer)]
    metrics = summarize_rows(rows)
    jlens_metrics = metrics["j_lens"]
    controls = {name: metrics[name] for name in criterion.controls}

    target_tokens = {
        row["target_token_id"] for row in rows if row["variant"] == "j_lens"
    }
    all_values_finite = all(
        _finite(value)
        for variant in metrics.values()
        for key, value in variant.items()
        if key != "n_prompts"
    )
    full_coverage = all(
        variant["n_prompts"] == criterion.n_prompts for variant in metrics.values()
    )

    checks: list[dict] = [
        {
            "check": "prompt_count",
            "primary": True,
            "passed": jlens_metrics["n_prompts"] == criterion.n_prompts,
            "detail": (
                f"{jlens_metrics['n_prompts']} scored prompts, "
                f"criterion requires exactly {criterion.n_prompts}"
            ),
        },
        {
            "check": "finite_and_nondegenerate",
            "primary": True,
            "passed": bool(
                all_values_finite
                and full_coverage
                and len(target_tokens) >= criterion.min_distinct_target_tokens
            ),
            "detail": (
                f"all metrics finite={all_values_finite}, every variant scored on "
                f"{criterion.n_prompts} prompts={full_coverage}, distinct target "
                f"tokens={len(target_tokens)} "
                f"(need >= {criterion.min_distinct_target_tokens})"
            ),
        },
        {
            "check": "top1_agreement_floor",
            "primary": True,
            "passed": jlens_metrics["top1_agreement"] >= criterion.min_top1_agreement,
            "detail": (
                f"J-lens top-1 agreement {jlens_metrics['top1_agreement']:.4f} "
                f"vs floor {criterion.min_top1_agreement:.2f}"
            ),
        },
        {
            "check": "mrr_exceeds_all_controls",
            "primary": True,
            "passed": all(
                jlens_metrics["mean_reciprocal_rank"] > control["mean_reciprocal_rank"]
                for control in controls.values()
            ),
            "detail": (
                f"J-lens MRR {jlens_metrics['mean_reciprocal_rank']:.5f} vs "
                + ", ".join(
                    f"{name}={control['mean_reciprocal_rank']:.5f}"
                    for name, control in controls.items()
                )
            ),
        },
        {
            "check": "median_rank_better_than_all_controls",
            "primary": True,
            "passed": all(
                jlens_metrics["median_target_rank"] < control["median_target_rank"]
                for control in controls.values()
            ),
            "detail": (
                f"J-lens median rank {jlens_metrics['median_target_rank']:.1f} vs "
                + ", ".join(
                    f"{name}={control['median_target_rank']:.1f}"
                    for name, control in controls.items()
                )
            ),
        },
        {
            "check": "no_control_reaches_comparable_top1",
            "primary": True,
            "passed": all(
                control["top1_agreement"]
                <= jlens_metrics["top1_agreement"] - criterion.control_top1_margin
                for control in controls.values()
            ),
            "detail": (
                f"controls must sit at or below "
                f"{jlens_metrics['top1_agreement'] - criterion.control_top1_margin:.4f}; "
                + ", ".join(
                    f"{name}={control['top1_agreement']:.4f}"
                    for name, control in controls.items()
                )
            ),
        },
    ]

    passed = all(check["passed"] for check in checks)
    return {
        "protocol": CONFIRMATORY_PROTOCOL,
        "layer": int(layer),
        "verdict": VERDICT_VALIDATED if passed else VERDICT_NO_GO,
        "passed": passed,
        "criterion": criterion.to_dict(),
        "criterion_digest": criterion.digest,
        "criterion_text": CRITERION_TEXT,
        "primary_metrics": list(criterion.primary_metrics),
        "secondary_metrics": list(criterion.secondary_metrics),
        "secondary_metrics_are_non_blocking": True,
        "diagnostic_variants": list(DIAGNOSTIC_VARIANTS),
        "checks": checks,
        "failed_checks": [c["check"] for c in checks if not c["passed"]],
        "metrics": metrics,
        "n_distinct_target_tokens": len(target_tokens),
    }


def confirmatory_report_markdown(verdict: dict, *, prompt_manifest: dict, lens_record: dict) -> str:
    """A short human-readable report of the run, for the run directory."""
    metrics = verdict["metrics"]
    layer = verdict["layer"]
    lines = [
        f"# Layer {layer} confirmatory native-readout validation",
        "",
        f"**Verdict: {verdict['verdict']}**",
        "",
        f"- protocol: `{verdict['protocol']}`",
        f"- lens: `{lens_record['lens_path']}`",
        f"- lens checksum: `{lens_record['lens_checksum']}`",
        f"- model revision: `{lens_record['model_revision']}`",
        f"- fitted source layers: {lens_record['fitted_source_layers']}",
        f"- held-out prompts: {prompt_manifest['n_prompts']} "
        f"(seed {prompt_manifest['seed']}, {prompt_manifest['n_excluded']} candidates "
        "excluded as v2 fitting/validation prompts)",
        f"- criterion digest: `{verdict['criterion_digest']}`",
        "",
        "No lens was fitted. The saved v2 artifact was read and never modified.",
        "",
        "## Metrics",
        "",
        "| variant | role | top-1 agreement | MRR | median rank | top-10 overlap* |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    roles = {
        "j_lens": "under test",
        "permuted": "control",
        "random": "control",
        "wrong_layer": "control",
        "logit_lens": "diagnostic",
    }
    for name in READOUT_VARIANTS:
        row = metrics[name]
        lines.append(
            f"| `{name}` | {roles.get(name, '')} | {row['top1_agreement']:.4f} | "
            f"{row['mean_reciprocal_rank']:.5f} | {row['median_target_rank']:.1f} | "
            f"{row['mean_top10_overlap']:.3f} |"
        )
    lines += [
        "",
        "\\* secondary metric: reported, but it cannot veto a primary-metric pass.",
        "",
        "## Predeclared criterion",
        "",
        "```",
        verdict["criterion_text"].rstrip(),
        "```",
        "",
        "## Checks",
        "",
    ]
    for check in verdict["checks"]:
        mark = "PASS" if check["passed"] else "FAIL"
        lines.append(f"- **{mark}** `{check['check']}` — {check['detail']}")
    lines += ["", "## Follow-up", ""]
    if verdict["passed"]:
        lines.append(
            f"Layer {layer} is independently confirmed on a fresh held-out set and "
            "may join layer 38 in the four-concept multimodal follow-up. The "
            "original validated-lens manifest is unchanged; the confirmation is "
            "published as a separate manifest."
        )
    else:
        lines.append(
            f"Layer {layer} did not clear the predeclared criterion. Keep only "
            "layer 38 for the four-concept follow-up. These results are preserved "
            "for the record; the original validated-lens manifest is unchanged."
        )
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------- store


@dataclass(frozen=True)
class ConfirmatoryFingerprint:
    """What a confirmatory run's stored results were produced from.

    A resume is only safe when all of this is unchanged. The prompt hashes are
    included in full: a set of 32 prompts that merely *counts* the same is not
    the same held-out set.
    """

    protocol: str
    lens_checksum: str
    model_repo_id: str
    model_revision: str
    prompt_protocol: str
    prompt_seed: int
    prompt_hashes: tuple[str, ...]
    layer: int
    controls: tuple[str, ...]
    control_seed: int
    criterion_digest: str
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["prompt_hashes"] = list(self.prompt_hashes)
        payload["controls"] = list(self.controls)
        return payload

    @property
    def digest(self) -> str:
        return payload_checksum(self.to_dict())

    def differences(self, other: ConfirmatoryFingerprint | dict) -> list[str]:
        """Field-by-field differences, so a refusal says what actually moved."""
        theirs = other.to_dict() if isinstance(other, ConfirmatoryFingerprint) else dict(other)
        mine = self.to_dict()
        out: list[str] = []
        for key in sorted(set(mine) | set(theirs)):
            a, b = mine.get(key, "<absent>"), theirs.get(key, "<absent>")
            if canonical_json(a) == canonical_json(b):
                continue
            if key == "prompt_hashes":
                out.append(
                    f"prompt_hashes: stored {len(b) if isinstance(b, list) else b} "
                    f"hashes, requested {len(a) if isinstance(a, list) else a} "
                    "(the held-out set changed)"
                )
            else:
                out.append(f"{key}: stored={b!r} requested={a!r}")
        return out


class ConfirmatoryStore:
    """Atomic per-prompt results under ``root``, gated by a fingerprint.

    Same two rules as the multimodal pilot's store, for the same reason (Colab
    disconnects mid-run): every result file carries a checksum of its own
    payload, and the run directory records what its results came from. A torn
    or edited file is treated as missing; a fingerprint mismatch is refused
    rather than silently mixed.

    Attributes:
        status: ``"starting"`` or ``"resuming"``, set by :meth:`open`.
        invalid_units: Result paths that failed their checksum and were dropped.
    """

    SCHEMA = "jlens.native_readout.confirmatory.v1"

    def __init__(
        self, root: str | os.PathLike[str], fingerprint: ConfirmatoryFingerprint
    ) -> None:
        self.root = Path(root)
        self.fingerprint = fingerprint
        self.status: str | None = None
        self.invalid_units: list[str] = []

    @property
    def fingerprint_path(self) -> Path:
        return self.root / "fingerprint.json"

    @property
    def results_dir(self) -> Path:
        return self.root / "prompt_results"

    def open(self) -> str:
        """Create or validate the run directory.

        Returns ``"starting"`` for a fresh directory, ``"resuming"`` when the
        stored fingerprint matches.

        Raises:
            IncompatibleStateError: If the stored fingerprint disagrees — the
                lens, model revision, prompt protocol or set, layer, controls,
                seed, or criterion changed. The message lists every differing
                field.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.fingerprint_path.is_file():
            self._write_json(self.fingerprint_path, self.fingerprint.to_dict())
            self.status = "starting"
            return self.status
        stored = json.loads(self.fingerprint_path.read_text(encoding="utf-8"))
        stored.pop("written_utc", None)
        if payload_checksum(stored) != self.fingerprint.digest:
            diffs = "\n  ".join(self.fingerprint.differences(stored))
            raise IncompatibleStateError(
                f"{self.root} holds confirmatory results from a different "
                f"configuration; refusing to reuse them.\n  {diffs}\n"
                "Point the run at a new directory (or delete this one yourself)."
            )
        self.status = "resuming"
        return self.status

    def result_path(self, sample_index: int, prompt_sha: str) -> Path:
        return self.results_dir / f"{safe_key(f'prompt-{sample_index:03d}', prompt_sha[:16])}.json"

    def save_result(self, sample_index: int, prompt_sha: str, payload: dict) -> Path:
        """Atomically write one prompt's rows. Returns the path written."""
        path = self.result_path(sample_index, prompt_sha)
        self._write_json(
            path,
            {
                "schema": self.SCHEMA,
                "sample": int(sample_index),
                "prompt_sha256": prompt_sha,
                "fingerprint_digest": self.fingerprint.digest,
                "unit_checksum": payload_checksum(payload),
                "payload": payload,
            },
        )
        return path

    def load_result(self, sample_index: int, prompt_sha: str) -> dict | None:
        """One prompt's payload, or ``None`` if absent, torn, checksum-invalid,
        or bound to a different fingerprint."""
        path = self.result_path(sample_index, prompt_sha)
        if not path.is_file():
            return None
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            payload = record["payload"]
            valid = (
                record.get("unit_checksum") == payload_checksum(payload)
                and record.get("fingerprint_digest") == self.fingerprint.digest
                and record.get("prompt_sha256") == prompt_sha
            )
        except (json.JSONDecodeError, KeyError, TypeError, UnicodeDecodeError):
            valid, payload = False, None
        if not valid:
            self.invalid_units.append(str(path))
            return None
        return payload

    def has_result(self, sample_index: int, prompt_sha: str) -> bool:
        return self.load_result(sample_index, prompt_sha) is not None

    def write_artifact(self, name: str, payload: dict | str) -> Path:
        """Atomically write an aggregate artifact (JSON dict or Markdown text)."""
        path = self.root / name
        if isinstance(payload, str):
            self._write_text(path, payload)
        else:
            self._write_json(path, payload)
        return path

    def status_report(self, *, expected: int = N_CONFIRMATORY_PROMPTS) -> dict:
        present = len(list(self.results_dir.glob("*.json"))) if self.results_dir.is_dir() else 0
        return {
            "run_dir": str(self.root),
            "status": self.status or "unopened",
            "fingerprint_digest": self.fingerprint.digest,
            "stored_prompt_results": present,
            "expected_prompt_results": expected,
            "invalid_units": list(self.invalid_units),
        }

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        ConfirmatoryStore._write_text(
            path, json.dumps(payload, indent=2, ensure_ascii=False, default=str)
        )

    @staticmethod
    def _write_text(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
