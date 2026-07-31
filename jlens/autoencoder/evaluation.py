# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Decisive evaluation on concept-disjoint held-out phrases, and the GO/NO-GO
report.

Everything here runs on the ``heldout`` split — phrases neither the
reconstructor nor the adapter has ever seen, in any occurrence. Training-split
numbers are computed too, but only as *diagnostics for attributing a failure*
(a model that scores well on train and at chance on held-out has a data
problem, not an architecture problem); they never enter a gate.

The report has exactly one code path. There is no branch that suppresses,
softens, or omits a NO-GO.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from contextlib import contextmanager

import torch

from jlens.autoencoder.baselines import (
    BASELINE_IDS,
    CONTROL_BASELINE_IDS,
    confabulation_probe,
    run_baselines,
)
from jlens.autoencoder.config import EvaluationConfig
from jlens.autoencoder.errors import AutoencoderError
from jlens.autoencoder.geometry import auroc, mean_or_none
from jlens.autoencoder.inference import (
    exact_match,
    first_token_rank,
    substring_recovery,
    verbalize_cone,
)
from jlens.autoencoder.preference import (
    normalize_candidate_text,
    sample_unrelated_cones,
)
from jlens.autoencoder.prompting import build_verbalizer_prompt

EVALUATION_SCHEMA = "jlens.autoencoder.evaluation.v1"
GONOGO_SCHEMA = "jlens.autoencoder.gonogo.v1"

#: The failure modes a NO-GO is attributed to. Exactly one is chosen, and the
#: evidence for the choice is reported alongside it.
FAILURE_MODES = (
    "phrase_reconstructor",
    "cone_adapter",
    "decoding_interface",
    "cone_information_loss",
    "insufficient_data",
    "prompt_dependence",
)


@contextmanager
def count_model_passes(model):
    """Count forward passes and scored rows through the model's own head.

    Wraps ``logits_from_ids`` for the duration of the block. Counting rather
    than estimating: "model passes" is a headline cost number in the report, and
    a formula derived from beam width and length silently stops being true the
    moment a beam finishes early.
    """
    original = model.logits_from_ids
    stats = {"n_calls": 0, "n_rows": 0, "n_positions": 0}

    def counted(input_ids, *, n_last=None):
        stats["n_calls"] += 1
        stats["n_rows"] += int(input_ids.shape[0])
        stats["n_positions"] += int(input_ids.shape[0] * input_ids.shape[1])
        return original(input_ids, n_last=n_last)

    model.logits_from_ids = counted  # type: ignore[method-assign]
    try:
        yield stats
    finally:
        model.logits_from_ids = original  # type: ignore[method-assign]


def summarize_results(
    results: Sequence[tuple[dict, str]], *, top_k: int = 5
) -> dict:
    """Aggregate one baseline's per-record inference results.

    ``results`` is ``[(inference_result, correct_phrase), ...]``.

    The AUROC here separates **correct from incorrect candidate phrases** by
    reconstruction cosine, pooled over every candidate of every record — the
    question "does the reconstruction score know which of the things Gemma said
    was right?", which is distinct from the reconstructor's own retrieval gate.
    """
    if not results:
        raise AutoencoderError("summarize_results called with no results")
    exact_1: list[float] = []
    exact_k: list[float] = []
    substrings: list[float] = []
    beam_contains: list[float] = []
    explained: list[float] = []
    accepted_flags: list[bool] = []
    accepted_correct: list[bool] = []
    abstained: list[bool] = []
    positive_scores: list[float] = []
    negative_scores: list[float] = []
    for result, phrase in results:
        exact_1.append(1.0 if exact_match(result, phrase, top_k=1) else 0.0)
        exact_k.append(1.0 if exact_match(result, phrase, top_k=top_k) else 0.0)
        substrings.append(substring_recovery(result, phrase, top_k=1))
        beam_contains.append(
            1.0 if exact_match(result, phrase, top_k=result["n_candidates"]) else 0.0
        )
        top = result.get("top_candidate")
        if top is not None:
            explained.append(float(top["explained_fraction"]))
        accepted = bool(result.get("accepted"))
        accepted_flags.append(accepted)
        abstained.append(not accepted)
        if accepted:
            accepted_correct.append(exact_match(result, phrase, top_k=1))
        target = normalize_candidate_text(phrase)
        for entry in result["candidates"]:
            score = float(entry["cosine"])
            if normalize_candidate_text(entry["text"]) == target:
                positive_scores.append(score)
            else:
                negative_scores.append(score)
    n = len(results)
    return {
        "n_records": n,
        "exact_match_top1": sum(exact_1) / n,
        f"exact_match_top{top_k}": sum(exact_k) / n,
        "mean_substring_recovery": sum(substrings) / n,
        "beam_contains_correct": sum(beam_contains) / n,
        "mean_explained_fraction": mean_or_none(explained),
        "acceptance_rate": sum(1 for a in accepted_flags if a) / n,
        "abstention_rate": sum(1 for a in abstained if a) / n,
        "acceptance_precision": (
            sum(1 for c in accepted_correct if c) / len(accepted_correct)
            if accepted_correct
            else None
        ),
        "unfiltered_precision": sum(exact_1) / n,
        "auroc_correct_vs_incorrect_candidates": auroc(positive_scores, negative_scores),
        "n_correct_candidates": len(positive_scores),
        "n_incorrect_candidates": len(negative_scores),
    }


def evaluate_baselines(
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
) -> dict:
    """Run every baseline on every record of ``split`` and summarize."""
    indices = dataset.indices_for_split(split)
    if limit is not None:
        indices = indices[: int(limit)]
    if not indices:
        raise AutoencoderError(f"split {split!r} has no records to evaluate")
    generator = torch.Generator().manual_seed(int(config.seed))
    per_baseline: dict[str, list[tuple[dict, str]]] = {b: [] for b in baselines}
    per_record: list[dict] = []
    started = time.perf_counter()
    with count_model_passes(model) as passes:
        for index in indices:
            record = dataset.records[index]
            unrelated = sample_unrelated_cones(
                dataset, index, n=config.n_unrelated_cones, generator=generator
            )
            results = run_baselines(
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
                baselines=baselines,
                seed=int(config.seed),
                device=device,
            )
            for baseline_id, result in results.items():
                per_baseline[baseline_id].append((result, record["phrase"]))
            first_token = None
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
    elapsed = time.perf_counter() - started
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
            "wall_seconds": round(elapsed, 3),
            "model_forward_calls": passes["n_calls"],
            "model_scored_rows": passes["n_rows"],
            "model_scored_positions": passes["n_positions"],
            "peak_cuda_memory_gb": (
                round(torch.cuda.max_memory_allocated() / 2**30, 3)
                if torch.cuda.is_available()
                else None
            ),
        },
    }


def evaluate_prompt_robustness(
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
) -> dict:
    """Top-1 recovery under each paraphrase of the constant instruction.

    A result that only survives one specific wording is a result about that
    wording. Agreement between prompts is reported alongside per-prompt accuracy
    because they fail differently: low accuracy everywhere is a weak adapter,
    high accuracy with low agreement is prompt dependence.
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


def evaluate_cross_cone_swap(evaluation: dict) -> dict:
    """Does the output follow the cone or the prompt?

    Every record saw the same constant instruction, so if the adapter's answers
    track the *phrases* rather than being constant across cones, the memory is
    doing the work. Reported as: how many distinct top-1 answers appeared, and
    how often a record's answer matched a *different* record's phrase.
    """
    rows = evaluation["per_record"]
    answers = [
        (row["results"].get("adapter_reranked") or {}).get("top_text") or ""
        for row in rows
    ]
    normalized = [normalize_candidate_text(a) for a in answers]
    phrases = [normalize_candidate_text(row["phrase"]) for row in rows]
    phrase_set = set(phrases)
    matched_own = sum(1 for a, p in zip(normalized, phrases, strict=True) if a == p)
    matched_other = sum(
        1
        for a, p in zip(normalized, phrases, strict=True)
        if a != p and a in phrase_set
    )
    return {
        "n_records": len(rows),
        "n_distinct_answers": len(set(normalized)),
        "answer_is_constant": len(set(normalized)) <= 1,
        "matched_own_phrase": matched_own / max(1, len(rows)),
        "matched_another_phrase": matched_other / max(1, len(rows)),
    }


def gonogo_report(
    *,
    reconstructor_metrics: dict,
    evaluation: dict,
    config: EvaluationConfig,
    leakage_report: dict,
    confabulation: dict | None = None,
    prompt_robustness: dict | None = None,
    diagnostics: dict | None = None,
) -> dict:
    """The verdict, with every criterion's observed value next to its threshold."""
    baselines = evaluation.get("baselines", {})
    reranked = baselines.get("adapter_reranked")
    raw_beam = baselines.get("adapter_raw_beam")
    zero = baselines.get("zero_memory")
    if reranked is None:
        raise AutoencoderError(
            "gonogo_report needs the adapter_reranked baseline; run the full "
            "evaluation before reporting"
        )

    observed_auroc = reconstructor_metrics.get("auroc_correct_vs_distractor")
    observed_top5 = reconstructor_metrics.get("top5_retrieval")
    beam_gain = (
        reranked["beam_contains_correct"] - zero["beam_contains_correct"]
        if zero is not None
        else None
    )
    rerank_gain = (
        reranked["exact_match_top1"] - raw_beam["exact_match_top1"]
        if raw_beam is not None
        else None
    )
    precision_gain = (
        reranked["acceptance_precision"] - reranked["unfiltered_precision"]
        if reranked.get("acceptance_precision") is not None
        else None
    )
    control_rates = {
        baseline_id: baselines[baseline_id]["acceptance_rate"]
        for baseline_id in CONTROL_BASELINE_IDS
        if baseline_id in baselines
    }
    worst_control = max(control_rates.values(), default=None)
    control_limit = (
        reranked["acceptance_rate"] * float(config.gate_control_acceptance_ratio_max)
    )
    n_leaks = len(leakage_report.get("violations", []))
    confabulation_clean = True if confabulation is None else bool(confabulation.get("clean"))

    criteria = [
        {
            "id": 1,
            "name": "reconstructor_auroc",
            "observed": observed_auroc,
            "threshold": float(config.gate_auroc_min),
            "passed": observed_auroc is not None and observed_auroc >= config.gate_auroc_min,
        },
        {
            "id": 2,
            "name": "reconstructor_top5_retrieval",
            "observed": observed_top5,
            "threshold": float(config.gate_top5_min),
            "passed": observed_top5 is not None and observed_top5 >= config.gate_top5_min,
        },
        {
            "id": 3,
            "name": "beam_contains_correct_vs_zero_memory",
            "observed": beam_gain,
            "threshold": float(config.gate_beam_gain_min),
            "passed": beam_gain is not None and beam_gain >= config.gate_beam_gain_min,
        },
        {
            "id": 4,
            "name": "reranking_improves_top1",
            "observed": rerank_gain,
            "threshold": 0.0,
            "passed": rerank_gain is not None and rerank_gain > 0.0,
        },
        {
            "id": 5,
            "name": "acceptance_precision_gain",
            "observed": precision_gain,
            "threshold": float(config.gate_precision_gain_min),
            "passed": precision_gain is not None
            and precision_gain >= config.gate_precision_gain_min,
        },
        {
            "id": 6,
            "name": "controls_rejected",
            "observed": worst_control,
            "threshold": control_limit,
            "passed": worst_control is not None and worst_control <= control_limit,
            "detail": control_rates,
        },
        {
            "id": 7,
            "name": "no_leakage_detected",
            "observed": n_leaks,
            "threshold": 0,
            "passed": n_leaks == 0 and confabulation_clean,
            "detail": {
                "split_leakage_violations": n_leaks,
                "confabulation_probe_clean": confabulation_clean,
            },
        },
    ]
    passed = all(c["passed"] for c in criteria)
    report = {
        "schema": GONOGO_SCHEMA,
        "verdict": "GO" if passed else "NO-GO",
        "passed": bool(passed),
        "split": evaluation.get("split"),
        "n_records": evaluation.get("n_records"),
        "criteria": criteria,
        "failed_criteria": [c["name"] for c in criteria if not c["passed"]],
        "baselines": baselines,
        "prompt_robustness": prompt_robustness,
        "confabulation_probe": confabulation,
    }
    if not passed:
        report["failure_attribution"] = attribute_failure(
            criteria,
            reconstructor_metrics=reconstructor_metrics,
            evaluation=evaluation,
            prompt_robustness=prompt_robustness,
            diagnostics=diagnostics or {},
        )
    return report


def attribute_failure(
    criteria: Sequence[dict],
    *,
    reconstructor_metrics: dict,
    evaluation: dict,
    prompt_robustness: dict | None,
    diagnostics: dict,
) -> dict:
    """Attribute a NO-GO to exactly one primary failure mode, with evidence.

    The order below is the causal order of the pipeline: an upstream failure
    makes every downstream number meaningless, so the first stage that failed is
    the one reported, and the rest are listed as consequences rather than as
    independent findings.
    """
    failed = {c["name"] for c in criteria if not c["passed"]}
    baselines = evaluation.get("baselines", {})
    reranked = baselines.get("adapter_reranked", {})
    zero = baselines.get("zero_memory", {})
    oracle = baselines.get("naive_token_average", {})
    train_retrieval = diagnostics.get("train_top5_retrieval")
    evidence: dict = {
        "failed_criteria": sorted(failed),
        "heldout_top5_retrieval": reconstructor_metrics.get("top5_retrieval"),
        "train_top5_retrieval": train_retrieval,
        "adapter_beam_contains_correct": reranked.get("beam_contains_correct"),
        "zero_memory_beam_contains_correct": zero.get("beam_contains_correct"),
        "oracle_token_average_top1": oracle.get("exact_match_top1"),
        "cross_prompt_agreement": (
            prompt_robustness.get("cross_prompt_agreement") if prompt_robustness else None
        ),
    }

    if {"reconstructor_auroc", "reconstructor_top5_retrieval"} & failed:
        # The reconstructor is upstream of everything. Whether the problem is the
        # reconstructor itself, the data, or the cone depends on how it did on
        # phrases it *was* trained on.
        if train_retrieval is not None and train_retrieval >= 0.8:
            mode = "insufficient_data"
            reason = (
                "the reconstructor fits training phrases but does not generalize to "
                "concept-disjoint phrases: too few distinct phrases to learn a "
                "phrase-independent map"
            )
        elif train_retrieval is not None and train_retrieval < 0.5:
            mode = "cone_information_loss"
            reason = (
                "the reconstructor cannot separate phrases even on its own training "
                "phrases: the k=10 cone at layer 14 does not appear to carry "
                "phrase-identifying information"
            )
        else:
            mode = "phrase_reconstructor"
            reason = (
                "the reconstructor is the first stage to fail and its training-split "
                "performance is intermediate; the map itself is the limiting factor"
            )
    elif "beam_contains_correct_vs_zero_memory" in failed:
        if (zero.get("beam_contains_correct") or 0.0) >= (
            reranked.get("beam_contains_correct") or 0.0
        ):
            mode = "cone_adapter"
            reason = (
                "the adapter's memory does not move generation beyond the "
                "zero-memory baseline: q is not reaching the decoder in a usable form"
            )
        else:
            mode = "cone_adapter"
            reason = (
                "the adapter helps but not by the required margin over zero memory"
            )
    elif "reranking_improves_top1" in failed:
        mode = "phrase_reconstructor"
        reason = (
            "generation clears the zero-memory bar but the reconstruction score does "
            "not order candidates better than Gemma's own likelihood: the reward "
            "signal is not informative"
        )
    elif "controls_rejected" in failed:
        mode = "phrase_reconstructor"
        reason = (
            "shuffled/unrelated/zero cones are accepted at a comparable rate: the "
            "acceptance test measures fluency, not cone identity"
        )
    elif "no_leakage_detected" in failed:
        mode = "insufficient_data"
        reason = (
            "a leakage or confabulation-attractor check failed; no performance number "
            "in this run can be interpreted until the dataset is rebuilt"
        )
    elif prompt_robustness is not None and (
        prompt_robustness.get("cross_prompt_agreement") is not None
        and prompt_robustness["cross_prompt_agreement"] < 0.5
    ):
        mode = "prompt_dependence"
        reason = "answers do not survive paraphrasing the constant instruction"
    else:
        mode = "cone_adapter"
        reason = "no upstream stage failed outright; the adapter is the limiting stage"
    return {
        "primary": mode,
        "known_modes": list(FAILURE_MODES),
        "reason": reason,
        "evidence": evidence,
        "note": (
            "downstream criteria that also failed are consequences of the primary "
            "mode and are not separate findings"
        ),
    }


def build_confabulation_probe(
    reconstructor,
    embedder,
    dataset,
    *,
    config: EvaluationConfig,
    phrase_token_ids: dict[str, list[int]],
    source_layer: int,
    split: str = "heldout",
    device: torch.device | str = "cpu",
) -> dict:
    """Run the attractor probe against the split's cones."""
    indices = dataset.indices_for_split(split)
    return confabulation_probe(
        reconstructor,
        embedder,
        list(config.confabulation_attractors),
        phrase_token_ids,
        dataset.cones[indices],
        source_layer=source_layer,
        accept_recon_min=config.accept_recon_min,
        device=device,
    )


def render_markdown(report: dict) -> str:
    """A compact human-readable summary of the GO/NO-GO report."""
    lines = [
        "# J-space language autoencoder — GO/NO-GO",
        "",
        f"**Verdict: {report['verdict']}** "
        f"(split `{report.get('split')}`, {report.get('n_records')} records)",
        "",
        "| # | criterion | observed | threshold | passed |",
        "|---|---|---|---|---|",
    ]
    for criterion in report["criteria"]:
        observed = criterion["observed"]
        observed_text = "n/a" if observed is None else f"{observed:.4g}"
        threshold = criterion["threshold"]
        threshold_text = "n/a" if threshold is None else f"{threshold:.4g}"
        lines.append(
            f"| {criterion['id']} | {criterion['name']} | {observed_text} | "
            f"{threshold_text} | {'yes' if criterion['passed'] else '**no**'} |"
        )
    if not report["passed"]:
        attribution = report.get("failure_attribution", {})
        lines += [
            "",
            f"**Primary failure mode: `{attribution.get('primary')}`**",
            "",
            str(attribution.get("reason", "")),
        ]
    lines += ["", "## Baselines", "", "| baseline | top-1 | top-5 | beam contains | accept rate | accept precision |", "|---|---|---|---|---|---|"]
    for baseline_id, metrics in report.get("baselines", {}).items():
        precision = metrics.get("acceptance_precision")
        lines.append(
            f"| `{baseline_id}` | {metrics['exact_match_top1']:.3f} | "
            f"{metrics.get('exact_match_top5', float('nan')):.3f} | "
            f"{metrics['beam_contains_correct']:.3f} | "
            f"{metrics['acceptance_rate']:.3f} | "
            f"{'n/a' if precision is None else f'{precision:.3f}'} |"
        )
    return "\n".join(lines) + "\n"
