# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Fit a Jacobian lens on Gemma 4 E4B from an experiment config.

Usage (real execution only under the explicit flag):

    python scripts/fit_gemma.py --config configs/gemma_text_microsmoke.yaml \
        --allow-model-load [--device-map cuda] [--dim-batch 4] [--no-resume]

Steps: validate config -> resolve immutable model revision -> load model ->
verify architecture -> memory/runtime probe at the configured dim_batch ->
fit (checkpointed) -> save lens + metadata. Everything recorded under
``paths.output_dir``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jlens.fitting import fit
from jlens.gemma4 import load_gemma4, probe_fit_cost, verify_architecture
from jlens.metadata import (
    config_fingerprint,
    environment_manifest,
    load_config,
    prompt_hashes,
    write_metadata,
)

logger = logging.getLogger("fit_gemma")

_DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32, "float16": torch.float16}


def load_fit_prompts(config: dict) -> list[str]:
    fitting = config["fitting"]
    if fitting["prompt_source"] == "file":
        with open(fitting["prompts_path"], encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("format") != "plain_text":
            raise ValueError(
                f"{fitting['prompts_path']}: fitting corpus must be plain_text "
                "(chat-templated prompts are evaluation-only)"
            )
        prompts = payload["prompts"][: fitting["n_prompts"]]
    else:  # wikitext
        from jlens.examples import load_wikitext_prompts

        prompts = load_wikitext_prompts(n_prompts=fitting["n_prompts"])
    if len(prompts) < fitting["n_prompts"]:
        raise ValueError(
            f"requested {fitting['n_prompts']} prompts, got {len(prompts)}"
        )
    return prompts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--allow-model-load",
        action="store_true",
        help="explicit opt-in to the ~16 GB model load (overrides config false)",
    )
    parser.add_argument("--device-map", default=None, help="override model.device_map")
    parser.add_argument("--dim-batch", type=int, default=None, help="override fitting.dim_batch")
    parser.add_argument("--no-resume", action="store_true", help="discard any existing checkpoint")
    parser.add_argument("--skip-probe", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("jlens").setLevel(logging.INFO)

    config = load_config(args.config)
    fingerprint = config_fingerprint(config)
    allow = args.allow_model_load or config["model"]["allow_model_load"]
    device_map = args.device_map or config["model"]["device_map"]
    dim_batch = args.dim_batch or config["fitting"]["dim_batch"]
    output_dir = config["paths"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    torch.manual_seed(config["fitting"]["seed"])
    prompts = load_fit_prompts(config)
    logger.info("mode=%s prompts=%d fingerprint=%s", config["mode"], len(prompts), fingerprint)

    model, load_info = load_gemma4(
        config["model"]["repo_id"],
        revision=config["model"]["revision"],
        tokenizer_repo_id=config["model"]["tokenizer_repo_id"],
        tokenizer_revision=config["model"]["tokenizer_revision"],
        dtype=_DTYPES[config["model"]["dtype"]],
        device_map=device_map,
        allow_model_load=allow,
    )
    report = verify_architecture(
        model,
        expect_n_layers=config["model"]["expect_n_layers"],
        expect_d_model=config["model"]["expect_d_model"],
        expect_vocab_size=config["model"]["expect_vocab_size"],
    )
    logger.info("architecture verified: %s", report.model_class)

    probe = None
    if not args.skip_probe:
        probe = probe_fit_cost(
            model,
            prompts[0],
            config["sites"]["source_layers"],
            dim_batch=dim_batch,
            max_seq_len=min(48, config["fitting"]["max_seq_len"]),
        )
        logger.info("probe: %s", probe)
        if not probe["all_finite"]:
            raise RuntimeError("probe produced non-finite Jacobians; aborting")
        n_passes = -(-config["model"]["expect_d_model"] // dim_batch)
        logger.info(
            "rough estimate: ~%.0f s/prompt x %d prompts at full seq len "
            "(probe: %d backward passes at seq %d took %.0f s)",
            probe["wall_seconds"] * config["fitting"]["max_seq_len"] / max(probe["seq_len"], 1),
            len(prompts),
            n_passes,
            probe["seq_len"],
            probe["wall_seconds"],
        )

    start = time.perf_counter()
    lens = fit(
        model,
        prompts,
        source_layers=config["sites"]["source_layers"],
        target_layer=config["sites"]["target_layer"],
        dim_batch=dim_batch,
        max_seq_len=config["fitting"]["max_seq_len"],
        skip_first=config["positions"]["skip_first"],
        checkpoint_path=config["paths"]["checkpoint"],
        checkpoint_every=config["fitting"]["checkpoint_every"],
        resume=not args.no_resume,
    )
    runtime = time.perf_counter() - start

    lens_path = os.path.join(output_dir, "lens.pt")
    lens.save(lens_path)
    reloaded_ok = len(type(lens).load(lens_path).source_layers) == len(lens.source_layers)

    write_metadata(
        os.path.join(output_dir, "fit_metadata.json"),
        {
            "config_path": os.path.abspath(args.config),
            "config": config,
            "config_fingerprint": fingerprint,
            "overrides": {"dim_batch": dim_batch, "device_map": str(device_map)},
            "load_info": load_info,
            "architecture_report": report.to_dict(),
            "probe": probe,
            "prompt_hashes": prompt_hashes(prompts),
            "n_prompts_fitted": lens.n_prompts,
            "fit_runtime_seconds": round(runtime, 1),
            "peak_cuda_memory_gb": (
                round(torch.cuda.max_memory_allocated() / 2**30, 2)
                if torch.cuda.is_available()
                else None
            ),
            "lens_path": os.path.abspath(lens_path),
            "lens_reload_ok": reloaded_ok,
            "environment": environment_manifest(),
        },
    )
    logger.info("saved %s (fitted on %d prompts, %.0f s)", lens_path, lens.n_prompts, runtime)
    return 0


if __name__ == "__main__":
    sys.exit(main())
