# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Run the generative J-cone steering validation on Gemma 4 E4B.

    python scripts/run_generative_validation.py \
        --config configs/gemma_generative_validation.yaml \
        --allow-model-load [--device-map cuda] [--split dev] [--smoke]

For every benchmark example x steering layer, this captures the source
activation at the established block_output site, runs a fresh k=10
nonnegative gradient pursuit against the layer's J-space dictionary
(rows of ``W_U @ J_l``), builds the weighted reconstruction
``q_C = sum a_i v_i`` plus the full control battery, injects each vector
(norm-scaled relative to the receiving activation) into neutral
verbalization prompts under the configured steering schedules, and records
teacher-forced multi-token target log-probabilities, KL from baseline, and
(for the configured subset) uncached greedy decodes.

Hard gates before any steering: architecture verification, lens fingerprint,
zero-vector logit parity, and manual-uncached vs ``generate()`` greedy
equivalence (both decoded tokens and first-step log-prob values). Any gate
failure aborts the run.

Outputs (in the timestamped run directory): ``run_started.json``,
``resolved_config.json``, ``artifacts/gates.json``,
``artifacts/pursuits.json``, ``artifacts/records.jsonl``,
``artifacts/summary_by_condition.json``, ``artifacts/calibration.json``
(dev split), ``artifacts/gonogo.json``, ``summary.md``,
``run_metadata.json``.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jlens.gemma4 import load_gemma4, verify_architecture
from jlens.generative import (
    DEFAULT_AGREEMENT_TOP_K,
    MAX_TARGET_TOKENS,
    MIN_TARGET_TOKENS,
    NATURAL_SCALE_MATCHED_CONDITIONS,
    NEUTRAL_PROMPTS,
    GenerativeError,
    SteeringSchedule,
    SteeringSpec,
    _next_token_logprobs,
    build_condition_vector,
    choose_calibration,
    compare_next_token_distributions,
    condition_scaling_mode,
    first_token_distribution,
    gonogo_report,
    greedy_decode,
    kl_from_baseline,
    load_benchmark,
    make_generative_record,
    natural_scale_gonogo_report,
    natural_scale_verdicts,
    per_example_verdicts,
    scale_to_norm,
    scale_to_ratio,
    select_split_examples,
    summarize_by_condition,
    target_logprob,
    validate_target_tokens,
)
from jlens.hooks import ActivationRecorder
from jlens.interventions import append_record
from jlens.lens import JacobianLens
from jlens.metadata import (
    config_fingerprint,
    environment_manifest,
    execution_record,
    file_sha256,
    load_generative_config,
    write_metadata,
)
from jlens.pursuit import JSpaceDictionary, PursuitSettings, gradient_pursuit

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
logger = logging.getLogger("generative_validation")

_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float16": torch.float16,
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--allow-model-load", action="store_true")
    parser.add_argument("--device-map", default=os.environ.get("JLENS_DEVICE_MAP"))
    parser.add_argument("--split", choices=("dev", "heldout"), default=None)
    parser.add_argument(
        "--runs-root",
        default=os.environ.get("JLENS_RUNS_ROOT", "runs"),
        help="Directory containing the pilot run (lens artifact) and receiving "
        "the new run directory.",
    )
    parser.add_argument("--limit-examples", type=int, default=None)
    parser.add_argument(
        "--layers", type=int, nargs="+", default=None, help="Override steering layers"
    )
    parser.add_argument(
        "--ratios", type=float, nargs="+", default=None, help="Override strength ratios"
    )
    parser.add_argument("--no-decode", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Minimal end-to-end pass: 1 example, 1 layer, 1 ratio, 1 neutral "
        "prompt, 4-token decodes.",
    )
    return parser.parse_args()


def derived_seed(base_seed: int, *parts: object) -> int:
    """Deterministic 31-bit seed from a base seed and condition coordinates."""
    payload = json.dumps([int(base_seed), *[str(p) for p in parts]])
    return int(hashlib.sha256(payload.encode()).hexdigest()[:8], 16) % (2**31)


def resolve_target_ids(tokenizer, phrase: str) -> list[int]:
    """Token ids of the target phrase as a continuation (no BOS/specials)."""
    ids = tokenizer(phrase, add_special_tokens=False, return_tensors=None).input_ids
    if not ids:
        raise GenerativeError(f"target phrase {phrase!r} tokenized to nothing")
    return [int(t) for t in ids]


def capture_activation(model, prompt: str, layer: int, position: int) -> torch.Tensor:
    """Residual at the block_output site of ``layer`` at ``position``
    (float32, detached, on CPU is fine — pursuit moves it)."""
    ids = model.encode(prompt)
    with ActivationRecorder(model.layers, at=[layer]) as recorder:
        with torch.no_grad():
            model.forward(ids)
    hidden = recorder.activations[layer]
    seq_len = hidden.shape[1]
    resolved = position if position >= 0 else seq_len + position
    if not (0 <= resolved < seq_len):
        raise GenerativeError(
            f"extraction position {position} out of range for length {seq_len}"
        )
    return hidden[0, resolved].detach().float()


def receiving_norm(model, input_ids: torch.Tensor, layer: int, position: int) -> float:
    """Norm of the unsteered residual at (layer, position) of the prompt."""
    with ActivationRecorder(model.layers, at=[layer]) as recorder:
        with torch.no_grad():
            model.forward(input_ids)
    return float(recorder.activations[layer][0, position].detach().float().norm())


def run_gates(model, config: dict, run_dir: str) -> dict:
    """Zero-parity and greedy-equivalence gates; abort on failure.

    Every logit here is read through the model's own head rather than
    ``unembed(forward(ids).last_hidden_state)`` — the latter applies the final
    norm twice (HuggingFace text models norm before returning
    ``last_hidden_state``) and scores a path the model never takes. All reads
    also pass ``n_last=1``, matching the ``logits_to_keep=1`` that
    ``GenerationMixin`` sets, so the LM head runs the same GEMM shape as
    ``generate()``; a full-sequence head accumulates in a different order and in
    bfloat16 that alone exceeds any reasonable tolerance.

    Checks:

    1. Zero-delta parity: a zero steering vector must reproduce baseline logits.
    2. Greedy token equivalence: the manual uncached decoder and a fully
       determinized ``generate()`` must emit the same tokens.
    3. First-step distribution agreement: the decoder's own readout must agree
       with ``generate()``'s logits on the argmax, on the top-k ranking, on
       top-k log-probabilities, and in total variation — so (2) cannot pass on a
       coincidental argmax tie while the two paths actually disagree.

    (3) deliberately does **not** gate on a max-over-vocabulary log-probability
    difference. See :func:`jlens.generative.compare_next_token_distributions`:
    that statistic is reported by the deepest tail of a 262k-token vocabulary,
    where it sits at the bfloat16 quantization floor (one ULP near Gemma's
    ``|logit| = 30`` softcap bound is already 0.0625) and where no token carries
    enough probability mass to affect a decode or a target score. It is recorded
    as a diagnostic instead.
    """
    tol = float(config["parity"]["max_abs_logit_diff_tol"])
    top_k = int(config["parity"].get("greedy_top_k", DEFAULT_AGREEMENT_TOP_K))
    topk_tol = float(
        config["parity"].get("max_abs_logprob_diff_topk_tol", tol)
    )
    tv_tol = float(config["parity"].get("max_total_variation_tol", 0.02))
    prompt = NEUTRAL_PROMPTS[config["neutral_prompts"][0]]
    ids = model.encode(prompt)
    layer = int(config["steering"]["layers"][0])
    with torch.no_grad():
        baseline = model.logits_from_ids(ids, n_last=1)[0, -1].float()
    spec = SteeringSpec(
        layer=layer,
        delta=torch.zeros(model.d_model),
        schedule=SteeringSchedule("constant"),
    )
    with spec.context(model.layers, prompt_len=ids.shape[1]):
        with torch.no_grad():
            hooked = model.logits_from_ids(ids, n_last=1)[0, -1].float()
    max_diff = float((baseline - hooked).abs().max())

    n_check = 8
    tokenizer = model.tokenizer
    eos_ids = [int(tokenizer.eos_token_id)] if tokenizer.eos_token_id is not None else []
    manual = greedy_decode(model, ids, max_new_tokens=n_check, eos_token_ids=eos_ids)
    hf_model = model._hf_model
    attention_mask = torch.ones_like(ids)
    # The checkpoint's stored generation config (e.g. Gemma 4 E4B-it defaults
    # to do_sample=True, top_k=64, top_p=0.95) must be fully overridden here,
    # not just do_sample: num_beams and use_cache also have to match the
    # manual decoder (uncached, single sequence) or this is not actually
    # testing greedy equivalence. output_logits gives us generate()'s own
    # pre-processing logits so the gate can compare values, not just argmax.
    generated = hf_model.generate(
        input_ids=ids,
        attention_mask=attention_mask,
        max_new_tokens=n_check,
        do_sample=False,
        num_beams=1,
        use_cache=False,
        eos_token_id=eos_ids or None,
        return_dict_in_generate=True,
        output_logits=True,
    )
    sequences = generated.sequences
    hf_tokens = [int(t) for t in sequences[0, ids.shape[1] :]]
    manual_tokens = manual.token_ids[: len(hf_tokens)]
    greedy_equal = manual_tokens == hf_tokens[: len(manual_tokens)]

    # Distribution-level check on the first step: token equality alone can pass
    # by luck (a near-tie, or a degenerate distribution with one dominant token)
    # while the two paths compute materially different logits. Read the manual
    # side through _next_token_logprobs — the decoder's *own* readout — so this
    # constrains the path greedy_decode actually takes rather than re-deriving a
    # correct one alongside it.
    with torch.no_grad():
        manual_first = _next_token_logprobs(model, ids, n_last=1)[-1].float()
    generate_first = torch.log_softmax(
        generated.logits[0][0].float().to(manual_first.device), dim=-1
    )
    agreement = compare_next_token_distributions(
        manual_first, generate_first, top_k=top_k
    )
    first_step_ok = (
        agreement["argmax_agrees"]
        and agreement["top_k_sets_agree"]
        and agreement["max_abs_logprob_diff_topk"] <= topk_tol
        and agreement["total_variation"] <= tv_tol
    )

    gates = {
        "zero_parity_max_abs_logit_diff": max_diff,
        "zero_parity_tol": tol,
        "zero_parity_ok": max_diff <= tol,
        "greedy_manual_tokens": manual.token_ids,
        "greedy_generate_tokens": hf_tokens,
        "greedy_equivalence_ok": greedy_equal,
        # Combined first-step agreement. Thresholds apply to the top-k
        # log-probabilities and to total variation; the full-vocabulary max and
        # mean inside `greedy_first_step_agreement` are diagnostics only (a
        # bfloat16 tail artifact, not a correctness signal).
        "greedy_first_step_agreement": agreement,
        "greedy_first_step_topk_tol": topk_tol,
        "greedy_first_step_total_variation_tol": tv_tol,
        "greedy_first_step_ok": first_step_ok,
    }
    with open(
        os.path.join(run_dir, "artifacts", "gates.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(gates, handle, indent=2)
    if not gates["zero_parity_ok"]:
        raise GenerativeError(
            f"zero-vector parity gate failed: max |diff| {max_diff} > {tol}"
        )
    if not gates["greedy_equivalence_ok"]:
        raise GenerativeError(
            f"greedy equivalence gate failed: manual {manual_tokens} vs "
            f"generate() {hf_tokens}"
        )
    if not gates["greedy_first_step_ok"]:
        raise GenerativeError(
            f"greedy first-step agreement gate failed: decode readout vs "
            f"generate() argmax_agrees={agreement['argmax_agrees']}, "
            f"top{agreement['top_k']}_sets_agree={agreement['top_k_sets_agree']}, "
            f"top-k max |dlogp| {agreement['max_abs_logprob_diff_topk']:.4g} "
            f"(tol {topk_tol}), total variation "
            f"{agreement['total_variation']:.4g} (tol {tv_tol}); the decode path "
            f"and the model's own head disagree on decision-relevant tokens. "
            f"Full-vocab max |dlogp| (diagnostic, not gated) "
            f"{agreement['max_abs_logprob_diff_full_vocab']:.4g}"
        )
    logger.info(
        "gates passed: parity %.2g, greedy tokens match, first-step top-%d "
        "max |dlogp| %.2g, total variation %.2g (full-vocab max %.2g, diagnostic)",
        max_diff,
        agreement["top_k"],
        agreement["max_abs_logprob_diff_topk"],
        agreement["total_variation"],
        agreement["max_abs_logprob_diff_full_vocab"],
    )
    return gates


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    config = load_generative_config(args.config)

    overrides: dict = {}
    if args.split:
        overrides["benchmark.split"] = args.split
        config["benchmark"]["split"] = args.split
    if args.layers:
        overrides["steering.layers"] = args.layers
        config["steering"]["layers"] = list(args.layers)
    if args.ratios:
        overrides["steering.strength_ratios"] = args.ratios
        config["steering"]["strength_ratios"] = [float(r) for r in args.ratios]
    if args.no_decode:
        overrides["decode.generate"] = False
        config["decode"]["generate"] = False
    if args.smoke:
        overrides["smoke"] = True
        config["steering"]["layers"] = config["steering"]["layers"][:1]
        config["steering"]["strength_ratios"] = [1.0]
        config["steering"]["mass_thresholds"] = config["steering"]["mass_thresholds"][:1]
        config["neutral_prompts"] = config["neutral_prompts"][:1]
        config["decode"]["max_new_tokens"] = 4
        config["decode"]["ratios"] = [1.0]
        args.limit_examples = 1
    if args.limit_examples:
        overrides["limit_examples"] = args.limit_examples

    manifest_path = config["benchmark"]["manifest_path"]
    if not os.path.isabs(manifest_path):
        manifest_path = os.path.join(REPO_ROOT, manifest_path)
    manifest = load_benchmark(manifest_path)
    # Donors for the unrelated-cone control come from the *same* split, always.
    # A development run that borrowed a held-out example's cone would leak
    # held-out information into calibration.
    examples, donor_only_ids = select_split_examples(
        manifest, config["benchmark"]["split"], limit=args.limit_examples
    )

    fingerprint = config_fingerprint(config)
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    run_id = f"generative_{timestamp}_{fingerprint.split(':')[1][:12]}"
    run_dir = os.path.join(args.runs_root, run_id)
    os.makedirs(os.path.join(run_dir, "artifacts"), exist_ok=False)
    started = time.time()
    with open(os.path.join(run_dir, "run_started.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_id": run_id,
                "config_path": args.config,
                "config_fingerprint": fingerprint,
                "overrides": overrides,
                "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            },
            f,
            indent=2,
        )
    with open(os.path.join(run_dir, "resolved_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # ---------------------------------------------------------------- model
    allow = args.allow_model_load or os.environ.get("JLENS_ALLOW_GEMMA") == "1"
    model, load_info = load_gemma4(
        config["model"]["repo_id"],
        revision=config["model"]["revision"],
        dtype=_DTYPES[config["model"]["dtype"]],
        device_map=args.device_map or config["model"]["device_map"],
        allow_model_load=allow,
        token=os.environ.get("HF_TOKEN"),
    )
    report = verify_architecture(
        model,
        expect_n_layers=config["model"]["expect_n_layers"],
        expect_d_model=config["model"]["expect_d_model"],
        expect_vocab_size=config["model"]["expect_vocab_size"],
    )

    # ----------------------------------------------------------------- lens
    lens_path = os.path.join(
        args.runs_root,
        config["lens"]["run_dir_name"],
        config["lens"]["artifact_relpath"],
    )
    lens_sha = file_sha256(lens_path)
    expect_sha = config["lens"]["expect_file_sha256"]
    if expect_sha and lens_sha != expect_sha:
        raise GenerativeError(
            f"lens fingerprint mismatch: {lens_path} is {lens_sha}, config "
            f"expects {expect_sha}"
        )
    lens = JacobianLens.load(lens_path)
    missing = set(config["steering"]["layers"]) - set(lens.source_layers)
    if missing:
        raise GenerativeError(f"steering layers {sorted(missing)} not fitted: {lens}")

    gates = run_gates(model, config, run_dir)

    tokenizer = model.tokenizer
    # Target token counts are a property of the pinned checkpoint's vocabulary,
    # so this can only be checked now that the real tokenizer is loaded. Aborts
    # the run (listing every offender) rather than silently scoring a
    # single-token target, which tests no multi-token scoring and makes all
    # three steering schedules produce identical target log-probabilities.
    target_bounds = config["benchmark"].get("target_token_bounds") or {}
    resolved_targets = validate_target_tokens(
        examples,
        tokenizer,
        resolve=resolve_target_ids,
        min_tokens=int(target_bounds.get("min", MIN_TARGET_TOKENS)),
        max_tokens=int(target_bounds.get("max", MAX_TARGET_TOKENS)),
    )
    with open(
        os.path.join(run_dir, "artifacts", "targets.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(resolved_targets, handle, indent=2, ensure_ascii=False)
    logger.info(
        "target tokenization validated for %d example(s): %s",
        len(resolved_targets),
        {k: v["n_target_tokens"] for k, v in resolved_targets.items()},
    )
    eos_ids = [int(tokenizer.eos_token_id)] if tokenizer.eos_token_id is not None else []
    base_seed = int(config["eval"]["control_seed"])
    steering_layers = [int(layer) for layer in config["steering"]["layers"]]
    ratios = [float(r) for r in config["steering"]["strength_ratios"]]
    schedules = [
        SteeringSchedule(entry["kind"], decay=float(entry.get("decay", 0.5)))
        for entry in config["steering"]["schedules"]
    ]
    conditions = list(config["steering"]["conditions"])
    mass_thresholds = [float(t) for t in config["steering"]["mass_thresholds"]]
    neutral_ids = list(config["neutral_prompts"])
    decode_cfg = config["decode"]
    records_path = os.path.join(run_dir, "artifacts", "records.jsonl")
    provenance = {
        "run_id": run_id,
        "config_fingerprint": fingerprint,
        "lens_path": lens_path,
        "lens_fingerprint": lens_sha,
        "model_revision": load_info["model_revision"],
        "benchmark_split": config["benchmark"]["split"],
        "local_commit": os.popen("git rev-parse HEAD").read().strip() or None,
    }

    def should_decode(condition: str, ratio: float | None, kind: str) -> bool:
        return (
            bool(decode_cfg["generate"])
            and condition in decode_cfg["generate_conditions"]
            and (ratio is None or float(ratio) in [float(r) for r in decode_cfg["ratios"]])
            and kind in decode_cfg["schedule_kinds"]
        )

    pursuit_settings = PursuitSettings(
        k=int(config["pursuit"]["k"]),
        normalize_atoms=bool(config["pursuit"]["normalize_atoms"]),
        refine_steps=int(config["pursuit"]["refine_steps"]),
        tol_relative_residual=float(config["pursuit"]["tol_relative_residual"]),
        correlation_chunk_size=config["pursuit"]["correlation_chunk_size"],
    )
    atoms_dtype = _DTYPES[config["pursuit"]["atoms_dtype"]]
    all_records: list[dict] = []
    pursuit_log: list[dict] = []

    for layer in steering_layers:
        logger.info("building J-space dictionary for layer %d", layer)
        dictionary = JSpaceDictionary.from_lens(
            lens,
            layer,
            model._lm_head.weight,
            device=model.input_device,
            dtype=atoms_dtype,
            build_chunk_rows=16384,
        )

        # Pursuits for every example at this layer (also feeds the
        # unrelated-cone control from the next example's active set).
        per_example: dict[str, dict] = {}
        for example in examples:
            h_source = capture_activation(
                model,
                example["source_prompt"],
                layer,
                int(example["extraction_position"]),
            )
            result = gradient_pursuit(h_source, dictionary, pursuit_settings)
            record = result.to_records()[0]
            active_ids = result.active_token_ids(0)
            id_to_coeff = dict(
                zip(record["token_ids"], record["coefficients"], strict=True)
            )
            active_coeffs = [id_to_coeff[t] for t in active_ids]
            if not active_ids:
                raise GenerativeError(
                    f"{example['example_id']}: pursuit selected no active "
                    f"generators at layer {layer} "
                    f"(stop: {record['stop_reason']})"
                )
            h_control = None
            if example.get("control_prompt"):
                h_control = capture_activation(
                    model,
                    example["control_prompt"],
                    layer,
                    int(example["extraction_position"]),
                )
            per_example[example["example_id"]] = {
                "example": example,
                "h_source": h_source,
                "h_control": h_control,
                "active_ids": active_ids,
                "active_coeffs": active_coeffs,
                "active_labels": [tokenizer.decode([t]) for t in active_ids],
                "pursuit_record": record,
            }
            pursuit_log.append(
                {
                    "example_id": example["example_id"],
                    "layer": layer,
                    "active_token_ids": active_ids,
                    "active_labels": per_example[example["example_id"]]["active_labels"],
                    "active_coefficients": active_coeffs,
                    "explained_fraction": record["explained_fraction"],
                    "stop_reason": record["stop_reason"],
                }
            )

        example_ids = [e["example_id"] for e in examples]
        wrong_layer = max(
            (l for l in steering_layers if l != layer),
            key=lambda l: abs(l - layer),
            default=None,
        )

        for index, example in enumerate(examples):
            if example["example_id"] in donor_only_ids:
                continue
            entry = per_example[example["example_id"]]
            # Donor is the next example in this same-split run list, so the
            # unrelated cone never comes from the other split.
            donor = per_example[example_ids[(index + 1) % len(example_ids)]]
            # The natural-scale reference: this example's own full-cone
            # reconstruction, unscaled. Depends only on this (layer, example)'s
            # active generators, not on the neutral prompt, so it is computed
            # once here rather than per prompt/schedule. natural_scale's own
            # injected delta is exactly this vector, so its norm doubles as
            # the target norm every natural-scale-matched control below is
            # rescaled to.
            natural_full_cone_norm = float(
                build_condition_vector(
                    "full_cone",
                    atoms=dictionary.atoms,
                    token_ids=entry["active_ids"],
                    coefficients=entry["active_coeffs"],
                ).delta.float().norm()
            )
            target = resolved_targets[example["example_id"]]
            target_ids = target["target_token_ids"]
            target_strings = target["target_token_strings"]
            example_conditions = list(conditions)
            manual_map = example.get("manual_subcone") or {}
            if str(layer) in manual_map and "manual_subcone" not in example_conditions:
                example_conditions.append("manual_subcone")

            for prompt_id in neutral_ids:
                prompt_text = NEUTRAL_PROMPTS[prompt_id]
                prompt_ids_tensor = model.encode(prompt_text)
                prompt_len = prompt_ids_tensor.shape[1]
                anchor_default = prompt_len - 1
                norms = {
                    (layer, anchor_default): receiving_norm(
                        model, prompt_ids_tensor, layer, anchor_default
                    )
                }
                if wrong_layer is not None:
                    norms[(wrong_layer, anchor_default)] = receiving_norm(
                        model, prompt_ids_tensor, wrong_layer, anchor_default
                    )
                wrong_anchor = int(config["steering"]["wrong_position_anchor"])
                norms[(layer, wrong_anchor)] = receiving_norm(
                    model, prompt_ids_tensor, layer, wrong_anchor
                )
                log_p_baseline = first_token_distribution(model, prompt_ids_tensor)

                zero_total: float | None = None
                unrelated_totals: dict[tuple[float, str], float] = {}

                # Loop variables are bound as keyword defaults so each
                # closure snapshots the current iteration's values (B023).
                def emit(
                    condition: str,
                    *,
                    delta: torch.Tensor | None,
                    vector_meta: dict,
                    ratio: float | None,
                    schedule: SteeringSchedule,
                    injection_layer: int,
                    anchor: int | None,
                    seed: int | None,
                    scale_info: dict | None,
                    example=example,
                    entry=entry,
                    layer=layer,
                    prompt_id=prompt_id,
                    prompt_ids_tensor=prompt_ids_tensor,
                    target_ids=target_ids,
                    target_strings=target_strings,
                    norms=norms,
                    anchor_default=anchor_default,
                    log_p_baseline=log_p_baseline,
                    unrelated_totals=unrelated_totals,
                ) -> dict:
                    nonlocal zero_total
                    spec = (
                        None
                        if delta is None
                        else SteeringSpec(
                            layer=injection_layer,
                            delta=delta,
                            schedule=schedule,
                            anchor=anchor,
                        )
                    )
                    scoring = target_logprob(
                        model, prompt_ids_tensor, target_ids, steering=spec
                    )
                    log_p = first_token_distribution(
                        model, prompt_ids_tensor, steering=spec
                    )
                    kl = kl_from_baseline(log_p, log_p_baseline)
                    hook_stats = {
                        "delta_norm": (
                            None if delta is None else float(delta.float().norm())
                        ),
                        "anchor_activation_norm": norms.get(
                            (
                                injection_layer,
                                anchor if anchor is not None else anchor_default,
                            )
                        ),
                        "measured_ratio": None,
                    }
                    if (
                        hook_stats["delta_norm"] is not None
                        and hook_stats["anchor_activation_norm"]
                    ):
                        hook_stats["measured_ratio"] = (
                            hook_stats["delta_norm"]
                            / hook_stats["anchor_activation_norm"]
                        )
                    decode_result = None
                    exact = substring = None
                    if should_decode(condition, ratio, schedule.kind):
                        decode_result = greedy_decode(
                            model,
                            prompt_ids_tensor,
                            max_new_tokens=int(decode_cfg["max_new_tokens"]),
                            steering=spec,
                            eos_token_ids=eos_ids,
                        )
                        exact = decode_result.token_ids[: len(target_ids)] == target_ids
                        substring = (
                            example["target_phrase"].strip().lower()
                            in decode_result.text.lower()
                        )
                    total = scoring["total_logprob"]
                    if condition == "zero":
                        zero_total = total
                    if condition == "unrelated_cone" and ratio is not None:
                        unrelated_totals[(float(ratio), schedule.kind)] = total
                    record = make_generative_record(
                        run_id=run_id,
                        example_id=example["example_id"],
                        condition=condition,
                        source_layer=layer,
                        injection_layer=injection_layer,
                        source_position=int(example["extraction_position"]),
                        injection_anchor=(
                            anchor if anchor is not None else anchor_default
                        ),
                        schedule=schedule,
                        neutral_prompt_id=prompt_id,
                        strength_ratio=ratio,
                        vector_meta={**vector_meta, **(scale_info or {})},
                        hook_stats=hook_stats,
                        scoring=scoring,
                        decode=decode_result,
                        delta_vs_zero=(
                            None if zero_total is None or condition in ("none", "zero")
                            else total - zero_total
                        ),
                        delta_vs_unrelated=(
                            total
                            - unrelated_totals[(float(ratio), schedule.kind)]
                            if ratio is not None
                            and condition
                            not in ("none", "zero", "unrelated_cone")
                            and (float(ratio), schedule.kind) in unrelated_totals
                            else None
                        ),
                        kl_divergence=kl,
                        target_phrase=example["target_phrase"],
                        target_token_strings=target_strings,
                        target_recovered_exact=exact,
                        target_recovered_substring=substring,
                        seed=seed,
                        provenance={
                            **provenance,
                            "active_token_ids": entry["active_ids"],
                            "active_labels": entry["active_labels"],
                            "active_coefficients": entry["active_coeffs"],
                            "pursuit_explained_fraction": entry["pursuit_record"][
                                "explained_fraction"
                            ],
                            "category": example["category"],
                        },
                    )
                    append_record(records_path, record)
                    all_records.append(record)
                    return record

                # Baselines first: none, then zero (defines delta_vs_zero),
                # then the unrelated cone at every ratio (defines
                # specificity), then everything else.
                emit(
                    "none",
                    delta=None,
                    vector_meta={"hooked": False},
                    ratio=None,
                    schedule=schedules[0],
                    injection_layer=layer,
                    anchor=None,
                    seed=None,
                    scale_info={"scaling_mode": condition_scaling_mode("none")},
                )
                zero_vector = build_condition_vector("zero", d_model=model.d_model)
                emit(
                    "zero",
                    delta=zero_vector.delta,
                    vector_meta=zero_vector.meta,
                    ratio=None,
                    schedule=schedules[0],
                    injection_layer=layer,
                    anchor=None,
                    seed=None,
                    scale_info={"scaling_mode": condition_scaling_mode("zero")},
                )

                def build_unscaled(
                    condition: str,
                    threshold: float | None = None,
                    *,
                    example=example,
                    entry=entry,
                    donor=donor,
                    layer=layer,
                    prompt_id=prompt_id,
                    dictionary=dictionary,
                    manual_map=manual_map,
                ):
                    seed = derived_seed(
                        base_seed, example["example_id"], condition, layer, prompt_id
                    )
                    kwargs: dict = {"seed": seed}
                    # Natural-scale-matched conditions (natural_unrelated_cone,
                    # natural_shuffled, natural_sign_reversed,
                    # natural_mass_subcone) build their *direction* with
                    # exactly the same inputs as their ratio-scaled
                    # counterpart — only the final scaling step differs (below,
                    # scale_to_norm against natural_full_cone_norm instead of
                    # scale_to_ratio against the receiving activation).
                    if condition in (
                        "full_cone",
                        "natural_scale",
                        "shuffled",
                        "natural_shuffled",
                        "sign_reversed",
                        "natural_sign_reversed",
                        "wrong_layer",
                        "wrong_position",
                    ):
                        kwargs.update(
                            atoms=dictionary.atoms,
                            token_ids=entry["active_ids"],
                            coefficients=entry["active_coeffs"],
                        )
                    elif condition in ("mass_subcone", "natural_mass_subcone"):
                        kwargs.update(
                            atoms=dictionary.atoms,
                            token_ids=entry["active_ids"],
                            coefficients=entry["active_coeffs"],
                            mass_threshold=threshold,
                        )
                    elif condition == "manual_subcone":
                        kwargs.update(
                            atoms=dictionary.atoms,
                            token_ids=entry["active_ids"],
                            coefficients=entry["active_coeffs"],
                            manual_indices=manual_map[str(layer)],
                        )
                    elif condition in ("unrelated_cone", "natural_unrelated_cone"):
                        kwargs.update(
                            atoms=dictionary.atoms,
                            unrelated_token_ids=donor["active_ids"],
                            unrelated_coefficients=donor["active_coeffs"],
                        )
                    elif condition in (
                        "random_matched_norm",
                        "natural_random_matched_norm",
                    ):
                        # Unit-norm raw direction; the final scaling step below
                        # (scale_to_ratio or scale_to_norm) sets the actual
                        # injected magnitude, exactly as for every other
                        # condition here.
                        kwargs.update(d_model=model.d_model, match_norm=1.0)
                    elif condition == "raw_activation":
                        kwargs.update(raw_activation=entry["h_source"])
                    elif condition == "activation_diff":
                        if entry["h_control"] is None:
                            return None, None
                        kwargs.update(
                            raw_activation=entry["h_source"],
                            control_activation=entry["h_control"],
                        )
                    built = build_condition_vector(condition, **kwargs)
                    return built, seed

                sweep = [
                    c for c in example_conditions if c not in ("none", "zero")
                ]
                # Unrelated first so specificity diffs resolve.
                sweep.sort(key=lambda c: c != "unrelated_cone")
                for condition in sweep:
                    thresholds = (
                        mass_thresholds
                        if condition in ("mass_subcone", "natural_mass_subcone")
                        else [None]
                    )
                    for threshold in thresholds:
                        built, seed = build_unscaled(condition, threshold)
                        if built is None:
                            continue
                        injection_layer = (
                            wrong_layer if condition == "wrong_layer" else layer
                        )
                        if injection_layer is None:
                            continue
                        anchor = (
                            int(config["steering"]["wrong_position_anchor"])
                            if condition == "wrong_position"
                            else None
                        )
                        norm_key = (
                            injection_layer,
                            anchor if anchor is not None else anchor_default,
                        )
                        # natural_scale is the cone at its own norm: no
                        # rescaling, no ratio sweep. Its measured_ratio is the
                        # observable that says where the cone naturally sits
                        # relative to the receiving activation.
                        if condition == "natural_scale":
                            receiving = norms[norm_key]
                            for schedule in schedules:
                                emit(
                                    condition,
                                    delta=built.delta,
                                    vector_meta=built.meta,
                                    ratio=None,
                                    schedule=schedule,
                                    injection_layer=injection_layer,
                                    anchor=anchor,
                                    seed=seed,
                                    scale_info={
                                        "scaling_mode": condition_scaling_mode(
                                            condition
                                        ),
                                        "unscaled": True,
                                        "natural_delta_norm": natural_full_cone_norm,
                                        "receiving_activation_norm": receiving,
                                        "natural_ratio": (
                                            natural_full_cone_norm / receiving
                                            if receiving
                                            else None
                                        ),
                                        # Same value as natural_delta_norm
                                        # above (natural_scale's own delta IS
                                        # the reference every natural-scale-
                                        # matched control is scaled to) —
                                        # added under the shared field name so
                                        # a reader can look up one key across
                                        # natural_scale and its matched
                                        # controls without special-casing the
                                        # condition. natural_delta_norm is
                                        # kept for backward compatibility.
                                        "reference_natural_cone_norm": (
                                            natural_full_cone_norm
                                        ),
                                    },
                                )
                            continue
                        # Natural-scale-matched controls: same direction as
                        # their ratio-scaled counterpart, rescaled to the
                        # example's own natural_full_cone_norm instead of a
                        # ratio of the receiving activation, and swept across
                        # every schedule natural_scale itself runs under (not
                        # just the first) — that per-schedule match is what
                        # makes "does the correct cone beat this control"
                        # answerable at all: at one fixed ratio, a schedule
                        # that merely reinjects more often could look like a
                        # direction effect when it is really a magnitude one.
                        if condition in NATURAL_SCALE_MATCHED_CONDITIONS:
                            scaled, norm_info = scale_to_norm(
                                built.delta, target_norm=natural_full_cone_norm
                            )
                            scale_info = {
                                "scaling_mode": condition_scaling_mode(condition),
                                "raw_delta_norm": norm_info["raw_delta_norm"],
                                "scale_factor": norm_info["scale_factor"],
                                "scaled_delta_norm": norm_info["scaled_delta_norm"],
                                "reference_natural_cone_norm": natural_full_cone_norm,
                                "receiving_activation_norm": norms[norm_key],
                            }
                            for schedule in schedules:
                                emit(
                                    condition,
                                    delta=scaled,
                                    vector_meta=built.meta,
                                    ratio=None,
                                    schedule=schedule,
                                    injection_layer=injection_layer,
                                    anchor=anchor,
                                    seed=seed,
                                    scale_info=scale_info,
                                )
                            continue
                        for ratio in ratios:
                            scaled, scale_info = scale_to_ratio(
                                built.delta,
                                activation_norm=norms[norm_key],
                                ratio=ratio,
                            )
                            scale_info = {
                                **scale_info,
                                "scaling_mode": condition_scaling_mode(condition),
                            }
                            condition_schedules = (
                                schedules
                                if condition == "full_cone"
                                else schedules[:1]
                            )
                            for schedule in condition_schedules:
                                emit(
                                    condition,
                                    delta=scaled,
                                    vector_meta=built.meta,
                                    ratio=ratio,
                                    schedule=schedule,
                                    injection_layer=injection_layer,
                                    anchor=anchor,
                                    seed=seed,
                                    scale_info=scale_info,
                                )
        del dictionary
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------- analysis
    artifacts = os.path.join(run_dir, "artifacts")
    with open(os.path.join(artifacts, "pursuits.json"), "w", encoding="utf-8") as f:
        json.dump(pursuit_log, f, indent=2, ensure_ascii=False)
    summary = summarize_by_condition(all_records)
    with open(
        os.path.join(artifacts, "summary_by_condition.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2)

    calibration = None
    gonogo = None
    try:
        calibration = choose_calibration(all_records)
        with open(
            os.path.join(artifacts, "calibration.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(calibration, f, indent=2)
        verdicts = per_example_verdicts(
            all_records,
            source_layer=calibration["source_layer"],
            ratio=calibration["ratio"],
            schedule_kind=calibration["schedule_kind"],
        )
        gonogo = gonogo_report(verdicts)
        with open(os.path.join(artifacts, "gonogo.json"), "w", encoding="utf-8") as f:
            json.dump({"verdicts": verdicts, "report": gonogo}, f, indent=2)
    except GenerativeError as exc:
        logger.warning("analysis incomplete: %s", exc)

    # Ratio-scaled controls (GONOGO_CONTROLS) are injected at a different norm
    # than natural_scale, so beating them at a calibrated ratio cannot show
    # whether a low-strength gain is specific to the correct cone's direction.
    # This is the matched-norm answer to that question: same delta norm, every
    # schedule, per (example, layer). Absent (natural_scale_gonogo stays None)
    # whenever the run didn't configure the natural-scale-matched conditions —
    # e.g. --smoke, or a config that only requests the ratio-scaled battery —
    # exactly like the ratio-scaled gonogo block above is absent when nothing
    # scored full_cone.
    natural_scale_gonogo = None
    try:
        natural_verdicts = natural_scale_verdicts(all_records)
        natural_scale_gonogo = natural_scale_gonogo_report(natural_verdicts)
        with open(
            os.path.join(artifacts, "natural_scale_comparison.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                {"verdicts": natural_verdicts, "report": natural_scale_gonogo},
                f,
                indent=2,
            )
    except GenerativeError as exc:
        logger.warning("natural-scale comparison incomplete: %s", exc)

    lines = [
        f"# Generative J-cone validation — {run_id}",
        "",
        f"- split: {config['benchmark']['split']}",
        f"- examples: {[e['example_id'] for e in examples]}",
        f"- layers: {steering_layers}, ratios: {ratios}",
        f"- records: {len(all_records)}",
        f"- gates: parity diff {gates['zero_parity_max_abs_logit_diff']:.3g}, "
        f"greedy equivalence {gates['greedy_equivalence_ok']}, "
        f"first-step top-{gates['greedy_first_step_agreement']['top_k']} "
        f"max |dlogp| "
        f"{gates['greedy_first_step_agreement']['max_abs_logprob_diff_topk']:.3g}, "
        f"total variation "
        f"{gates['greedy_first_step_agreement']['total_variation']:.3g}",
        "",
        "## Per-condition summary",
        "",
        "| condition | mode | n | mean d(logP) vs zero | mean specificity | exact | substring |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in summary:
        def fmt(value):
            return "-" if value is None else f"{value:.3f}"

        lines.append(
            f"| {row['vector_condition']} | {row['scaling_mode']} | "
            f"{row['n_records']} | "
            f"{fmt(row['mean_delta_vs_zero'])} | "
            f"{fmt(row['mean_specificity_vs_unrelated'])} | "
            f"{fmt(row['exact_recovery_rate'])} | "
            f"{fmt(row['substring_recovery_rate'])} |"
        )
    if calibration:
        lines += [
            "",
            f"## Calibration: layer {calibration['source_layer']}, "
            f"ratio {calibration['ratio']}, {calibration['schedule_kind']} "
            f"(mean d = {calibration['mean_delta_vs_zero']:.3f})",
        ]
    if gonogo:
        lines += [
            "",
            f"## Go/no-go: **{'GO' if gonogo['go'] else 'NO-GO'}** "
            f"({gonogo['n_passing_joint_criterion']}/{gonogo['n_examples']} "
            f"examples pass; recovery {gonogo['recovery_fraction']:.0%})",
        ]
    if natural_scale_gonogo:
        lines += [
            "",
            f"## Natural-scale comparison (matched delta norm, every schedule): "
            f"**{'GO' if natural_scale_gonogo['go'] else 'NO-GO'}** "
            f"({natural_scale_gonogo['n_passing']}/{natural_scale_gonogo['n_points']} "
            f"(example, layer, schedule) points beat zero and every "
            f"natural-scale-matched control)",
        ]
    with open(os.path.join(run_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    write_metadata(
        os.path.join(run_dir, "run_metadata.json"),
        {
            "run_id": run_id,
            "mode": "generative_validation",
            "config": config,
            "config_fingerprint": fingerprint,
            "overrides": overrides,
            "load_info": load_info,
            "architecture_report": report.to_dict(),
            "execution": execution_record(
                configured_allow_model_load=bool(
                    config["model"]["allow_model_load"]
                ),
                resolved_allow_model_load=allow,
                model_loaded=True,
                override_source=(
                    "cli:--allow-model-load" if args.allow_model_load else "env:JLENS_ALLOW_GEMMA"
                ),
            ),
            "environment": environment_manifest(),
            "gates": gates,
            "n_records": len(all_records),
            "wall_seconds": round(time.time() - started, 1),
            "calibration": calibration,
            "gonogo": gonogo,
            "natural_scale_gonogo": natural_scale_gonogo,
        },
    )
    logger.info("done: %d records in %s", len(all_records), run_dir)


if __name__ == "__main__":
    main()
