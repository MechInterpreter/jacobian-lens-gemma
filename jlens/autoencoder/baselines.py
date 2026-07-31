# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""The eight baselines the adapter has to beat to mean anything.

All of them run through the *identical* prompt, beam width, stop tokens,
reconstructor, and reward. Only the memory differs — so a gap between two rows
of the results table is a fact about the memory, not about the harness.

Two of them are deliberately *not* fair comparisons and are labelled as such:

* ``naive_token_average`` needs the phrase's own token ids, so it is an
  **oracle** reference for "how much could constituent-token J-vectors alone
  say?", not a member of the adapter's pipeline. If the adapter cannot beat it,
  the cycle carries no more than the tokens already visible in the phrase.
* ``jlens_token_clues`` does not generate at all: it is the ordinary J-lens
  readout — the pursuit's selected atoms as token strings, ranked by
  coefficient. It is the "you did not need any of this machinery" baseline.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from jlens.autoencoder.errors import AutoencoderError
from jlens.autoencoder.geometry import unit
from jlens.autoencoder.inference import verbalize_cone
from jlens.autoencoder.preference import (
    normalize_candidate_text,
    reconstruct_candidates,
    score_candidates,
)
from jlens.autoencoder.verbalizer import Candidate
from jlens.generative import shuffled_coordinates, tensor_sha256

#: Baseline ids in the order the brief lists them.
BASELINE_IDS = (
    "zero_memory",
    "shuffled_q",
    "unrelated_q",
    "sign_reversed_q",
    "naive_token_average",
    "jlens_token_clues",
    "adapter_raw_beam",
    "adapter_reranked",
)

#: Baselines whose memory is derived from a *perturbed* cone. These are the
#: control battery: a positive result requires the adapter to beat them.
CONTROL_BASELINE_IDS = ("zero_memory", "shuffled_q", "unrelated_q", "sign_reversed_q")

#: Baselines that use information the pipeline is not allowed to use.
ORACLE_BASELINE_IDS = ("naive_token_average",)


def baseline_cone(
    baseline_id: str,
    q: torch.Tensor,
    *,
    dataset=None,
    index: int | None = None,
    dictionary=None,
    phrase_token_ids: Sequence[int] | None = None,
    seed: int = 0,
) -> tuple[torch.Tensor | None, dict]:
    """The substituted cone for a baseline, or ``None`` for ``zero_memory``.

    Returns ``(cone, meta)``; ``meta`` records how the cone was derived and its
    fingerprint, so a baseline row in the results is traceable to the exact
    vector it used.
    """
    if baseline_id == "zero_memory":
        return None, {"derivation": "no memory (zeros written into the slots)"}
    if baseline_id == "shuffled_q":
        cone = shuffled_coordinates(q, seed=int(seed))
        return cone, {
            "derivation": "coordinate permutation of q (norm and value multiset preserved)",
            "cone_sha256": tensor_sha256(cone),
        }
    if baseline_id == "sign_reversed_q":
        cone = -q.float()
        return cone, {"derivation": "-q", "cone_sha256": tensor_sha256(cone)}
    if baseline_id == "unrelated_q":
        if dataset is None or index is None:
            raise AutoencoderError("unrelated_q needs the dataset and a record index")
        split = dataset.records[index]["split"]
        phrase_id = dataset.records[index]["phrase_id"]
        pool = [
            i
            for i in dataset.indices_for_split(split)
            if dataset.records[i]["phrase_id"] != phrase_id
        ]
        if not pool:
            raise AutoencoderError("no unrelated cone available in this split")
        generator = torch.Generator().manual_seed(int(seed))
        donor = pool[int(torch.randint(len(pool), (1,), generator=generator))]
        cone = dataset.cones[donor]
        return cone, {
            "derivation": "another phrase's cone from the same split",
            "donor_record_index": donor,
            "donor_phrase_id": dataset.records[donor]["phrase_id"],
            "cone_sha256": tensor_sha256(cone),
        }
    if baseline_id == "naive_token_average":
        if dictionary is None or phrase_token_ids is None:
            raise AutoencoderError(
                "naive_token_average needs the J-space dictionary and the phrase's token ids"
            )
        ids = [int(t) for t in phrase_token_ids]
        if not ids:
            raise AutoencoderError("naive_token_average needs at least one phrase token")
        cone = dictionary.atoms[torch.tensor(ids, dtype=torch.long)].float().mean(dim=0).cpu()
        return cone, {
            "derivation": "mean of the J-lens vectors of the phrase's own tokens (ORACLE)",
            "oracle": True,
            "n_tokens_averaged": len(ids),
            "cone_sha256": tensor_sha256(cone),
        }
    raise AutoencoderError(f"{baseline_id!r} does not define a substituted cone")


def token_clue_result(
    reconstructor,
    embedder,
    record: dict,
    q: torch.Tensor,
    unrelated_cones: torch.Tensor,
    *,
    reward_config,
    tokenizer,
    source_layer: int,
    accept_recon_min: float,
    accept_margin_min: float,
    top_k: int,
    device: torch.device | str = "cpu",
) -> dict:
    """The ordinary J-lens readout as a candidate list: active atoms as token
    strings, ranked by pursuit coefficient.

    Scored through the same reconstructor and reward as everything else, so the
    comparison is like-for-like even though no generation happened. The
    "sequence log-probability" is reported as the coefficient's share of the
    cone's total coefficient mass — an honest stand-in, labelled as such, not a
    fabricated model score.
    """
    token_ids = [int(t) for t in record["active_token_ids"]]
    coefficients = [float(c) for c in record["active_coefficients"]]
    pairs = sorted(
        ((c, t) for c, t in zip(coefficients, token_ids, strict=True) if c > 0),
        key=lambda item: (-item[0], item[1]),
    )[: int(top_k)]
    if not pairs:
        raise AutoencoderError("record has no positive pursuit coefficients")
    total = sum(c for c, _ in pairs)
    candidates: list[Candidate] = []
    for rank, (coefficient, token_id) in enumerate(pairs):
        text = tokenizer.decode([token_id], skip_special_tokens=True).strip()
        share = coefficient / total if total > 0 else 0.0
        candidates.append(
            Candidate(
                token_ids=(token_id,),
                text=text,
                logprob=float(torch.log(torch.tensor(max(share, 1e-12)))),
                mean_logprob=float(torch.log(torch.tensor(max(share, 1e-12)))),
                n_tokens=1,
                finished=True,
                beam_rank=rank,
            )
        )
    q_hats = reconstruct_candidates(
        reconstructor, embedder, candidates, source_layer=source_layer, device=device
    )
    scored = score_candidates(
        candidates,
        q,
        q_hats,
        unrelated_cones,
        config=reward_config,
        accept_recon_min=accept_recon_min,
        accept_margin_min=accept_margin_min,
    )
    ranked = sorted(range(len(scored)), key=lambda i: (-scored[i].reward, i))
    entries = []
    for final_rank, index in enumerate(ranked):
        item = scored[index]
        entries.append(
            {
                **item.to_dict(),
                "final_rank": final_rank,
                "raw_beam_rank": item.candidate.beam_rank,
                "q_hat_norm": float(item.q_hat.norm()),
                "q_hat_sha256": tensor_sha256(item.q_hat),
            }
        )
    best = entries[0]
    return {
        "schema": "jlens.autoencoder.inference.v1",
        "verdict": "accepted" if best["accepted"] else "abstained_below_threshold",
        "accepted": bool(best["accepted"]),
        "n_candidates": len(entries),
        "n_flat_runners_up": 0,
        "flat_margin": 0.0,
        "accept_recon_min": float(accept_recon_min),
        "accept_margin_min": float(accept_margin_min),
        "q_norm": float(q.norm()),
        "q_sha256": tensor_sha256(q),
        "memory_rms": None,
        "sequence_logprob_semantics": "log of the atom's share of total coefficient mass (not a model score)",
        "top_candidate": best,
        "candidates": entries,
    }


def raw_beam_view(result: dict) -> dict:
    """The same generation, ranked by Gemma's own sequence log-probability.

    ``adapter_raw_beam`` and ``adapter_reranked`` share one generation pass on
    purpose: any difference between them is then attributable to the ranking
    alone, with zero sampling noise between the two.
    """
    entries = sorted(result["candidates"], key=lambda e: (e["raw_beam_rank"],))
    renumbered = [{**entry, "final_rank": rank} for rank, entry in enumerate(entries)]
    best = renumbered[0] if renumbered else None
    # Acceptance uses the same thresholds as the reranked view, applied to
    # *this* view's top candidate. Anything else would make the two rows differ
    # by more than the ranking, which is the one thing they must not do.
    accepted = bool(best is not None and best["accepted"])
    return {
        **result,
        "ranking": "raw_gemma_sequence_logprob",
        "verdict": "accepted" if accepted else "abstained_below_threshold",
        "accepted": accepted,
        "top_candidate": best,
        "candidates": renumbered,
    }


def run_baselines(
    model,
    adapter,
    reconstructor,
    embedder,
    dataset,
    prompt,
    index: int,
    *,
    conditioner,
    unrelated_cones: torch.Tensor,
    reward_config,
    stop_token_ids: Sequence[int],
    pad_token_id: int,
    beam_width: int,
    max_new_tokens: int,
    source_layer: int,
    accept_recon_min: float,
    accept_margin_min: float,
    dictionary=None,
    baselines: Sequence[str] = BASELINE_IDS,
    seed: int = 0,
    device: torch.device | str = "cpu",
) -> dict[str, dict]:
    """Run the requested baselines for one dataset record.

    ``adapter_reranked`` is the pipeline's own answer; the other rows are what it
    has to beat. Every row is a full inference result, so the evaluator can apply
    identical metrics to all of them.
    """
    unknown = [b for b in baselines if b not in BASELINE_IDS]
    if unknown:
        raise AutoencoderError(f"unknown baseline(s) {unknown}; known are {list(BASELINE_IDS)}")
    record = dataset.records[index]
    q = dataset.cones[index]
    results: dict[str, dict] = {}

    def generate(memory_source: torch.Tensor | None, meta: dict) -> dict:
        memory = None
        if memory_source is None:
            memory = torch.zeros(1, adapter.n_memory_tokens, adapter.d_model, device=device)
        else:
            with torch.no_grad():
                memory = adapter(memory_source.unsqueeze(0).to(device))
        result = verbalize_cone(
            model,
            adapter,
            reconstructor,
            embedder,
            q,
            prompt,
            conditioner=conditioner,
            unrelated_cones=unrelated_cones,
            reward_config=reward_config,
            stop_token_ids=stop_token_ids,
            pad_token_id=pad_token_id,
            beam_width=beam_width,
            max_new_tokens=max_new_tokens,
            source_layer=source_layer,
            accept_recon_min=accept_recon_min,
            accept_margin_min=accept_margin_min,
            device=device,
            memory=memory,
        )
        result["memory_meta"] = meta
        return result

    adapter_result: dict | None = None
    for baseline_id in baselines:
        if baseline_id in ("adapter_raw_beam", "adapter_reranked"):
            if adapter_result is None:
                adapter_result = generate(q, {"derivation": "adapter(q)"})
            if baseline_id == "adapter_reranked":
                results[baseline_id] = {**adapter_result, "ranking": "reconstructor_reward"}
            else:
                results[baseline_id] = raw_beam_view(adapter_result)
            continue
        if baseline_id == "jlens_token_clues":
            results[baseline_id] = token_clue_result(
                reconstructor,
                embedder,
                record,
                q,
                unrelated_cones,
                reward_config=reward_config,
                tokenizer=model.tokenizer,
                source_layer=source_layer,
                accept_recon_min=accept_recon_min,
                accept_margin_min=accept_margin_min,
                top_k=beam_width,
                device=device,
            )
            results[baseline_id]["memory_meta"] = {
                "derivation": "no memory; ordinary J-lens atom readout"
            }
            continue
        cone, meta = baseline_cone(
            baseline_id,
            q,
            dataset=dataset,
            index=index,
            dictionary=dictionary,
            phrase_token_ids=record["phrase_token_ids"],
            seed=seed + index,
        )
        results[baseline_id] = generate(cone, meta)
    return results


def confabulation_probe(
    reconstructor,
    embedder,
    attractors: Sequence[str],
    phrase_token_ids: dict[str, list[int]],
    cones: torch.Tensor,
    *,
    source_layer: int,
    accept_recon_min: float,
    device: torch.device | str = "cpu",
) -> dict:
    """Do famous attractor phrases score highly against *unrelated* cones?

    ``black hole`` / ``photosynthesis`` / ``quantum entanglement`` /
    ``Great Barrier Reef`` are the phrases a language-model-ish reconstructor
    reaches for. If any of them clears the acceptance threshold against cones
    that have nothing to do with it, the reconstructor is scoring fluency rather
    than geometry, and every acceptance number in the run is suspect.
    """
    missing = [a for a in attractors if a not in phrase_token_ids]
    if missing:
        raise AutoencoderError(f"no token ids for attractor phrase(s) {missing}")
    with torch.no_grad():
        embeddings, mask = embedder.batch([phrase_token_ids[a] for a in attractors])
        vectors = reconstructor(embeddings.to(device), mask.to(device), source_layer).cpu()
    similarity = unit(cones.float()) @ unit(vectors).T  # [N, A] cosines
    per_attractor = []
    for column, attractor in enumerate(attractors):
        scores = similarity[:, column]
        per_attractor.append(
            {
                "phrase": attractor,
                "mean_cosine_vs_unrelated_cones": float(scores.mean()),
                "max_cosine_vs_unrelated_cones": float(scores.max()),
                "fraction_above_accept_threshold": float(
                    (scores >= float(accept_recon_min)).float().mean()
                ),
            }
        )
    worst = max(per_attractor, key=lambda item: item["fraction_above_accept_threshold"])
    return {
        "n_cones_probed": int(cones.shape[0]),
        "accept_recon_min": float(accept_recon_min),
        "per_attractor": per_attractor,
        "worst_attractor": worst["phrase"],
        "worst_fraction_above_threshold": worst["fraction_above_accept_threshold"],
        # 10% is the tolerance: a handful of unrelated cones landing near a
        # famous phrase is noise; a fifth of them is the reconstructor scoring
        # fluency.
        "clean": bool(worst["fraction_above_accept_threshold"] <= 0.10),
    }


def normalized_texts(result: dict, *, top_k: int) -> list[str]:
    """Normalized candidate surfaces for the top-``k`` entries."""
    return [normalize_candidate_text(e["text"]) for e in result["candidates"][: int(top_k)]]
