# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Apply a fitted Gemma 4 Jacobian lens to the evaluation prompt set, with
logit-lens baseline and negative controls.

    python scripts/apply_gemma.py --config configs/gemma_text_smoke.yaml \
        --lens artifacts/smoke/lens.pt --allow-model-load [--device-map cuda]

For every eval prompt (plain-text and chat-templated evaluated separately) and
every fitted layer this reports, at the configured positions:

- J-lens top-k tokens (pre-softcap and softcapped logit values; rankings are
  identical because the cap is monotonic),
- logit-lens (identity transport) top-k,
- permuted-J control (primary), scale-matched random control, wrong-layer
  control,
- top-k overlap with the model's real output and the rank of the model's
  top-1 token under each lens.

Writes ``apply_results.json`` and a human-readable ``apply_summary.md``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jlens.controls import (
    control_lens,
    ranks_of_targets,
    topk_overlap,
    wrong_layer_lens,
)
from jlens.gemma4 import apply_dual, load_gemma4, verify_architecture
from jlens.lens import JacobianLens
from jlens.metadata import (
    config_fingerprint,
    environment_manifest,
    load_config,
    write_metadata,
)

logger = logging.getLogger("apply_gemma")

_DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32, "float16": torch.float16}


def load_eval_prompts(path: str, tokenizer) -> list[dict]:
    """Flatten the eval prompt file into [{slug, format, text, positions}],
    rendering chat entries with the tokenizer's chat template."""
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    prompts: list[dict] = []
    for entry in payload.get("plain", []):
        prompts.append(
            {
                "slug": entry["slug"],
                "format": "plain",
                "text": entry["text"],
                "positions": entry.get("positions", [-1]),
            }
        )
    for entry in payload.get("chat", []):
        messages = []
        if entry.get("system"):
            messages.append({"role": "system", "content": entry["system"]})
        messages.append({"role": "user", "content": entry["user"]})
        if entry.get("assistant_prefill"):
            messages.append({"role": "assistant", "content": entry["assistant_prefill"]})
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, continue_final_message=True
            )
        else:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        prompts.append(
            {
                "slug": entry["slug"],
                "format": "chat",
                "text": text,
                "positions": entry.get("positions", [-1]),
            }
        )
    return prompts


def topk_readout(logits_pre: torch.Tensor, logits_capped: torch.Tensor, tokenizer, k: int):
    """[{token, id, logit_pre, logit_capped}] for the top-k of one position."""
    top = logits_pre.topk(k)
    return [
        {
            "token": tokenizer.decode([int(idx)]),
            "id": int(idx),
            "logit_pre": round(float(logits_pre[idx]), 3),
            "logit_capped": round(float(logits_capped[idx]), 3),
        }
        for idx in top.indices
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--lens", required=True, help="path to a fitted lens.pt")
    parser.add_argument("--allow-model-load", action="store_true")
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--layers", type=int, nargs="*", default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    config = load_config(args.config)
    top_k = config["eval"]["top_k"]
    control_seed = config["eval"]["control_seed"]
    output_dir = config["paths"]["output_dir"]

    lens = JacobianLens.load(args.lens)
    layers = args.layers or lens.source_layers
    controls = {
        "permuted": control_lens(lens, "permuted", seed=control_seed),
        "random": control_lens(lens, "random", seed=control_seed),
    }
    if len(lens.source_layers) >= 2:
        controls["wrong_layer"] = wrong_layer_lens(lens)

    model, load_info = load_gemma4(
        config["model"]["repo_id"],
        revision=config["model"]["revision"],
        tokenizer_repo_id=config["model"]["tokenizer_repo_id"],
        tokenizer_revision=config["model"]["tokenizer_revision"],
        dtype=_DTYPES[config["model"]["dtype"]],
        device_map=args.device_map or config["model"]["device_map"],
        allow_model_load=args.allow_model_load or config["model"]["allow_model_load"],
    )
    report = verify_architecture(
        model,
        expect_n_layers=config["model"]["expect_n_layers"],
        expect_d_model=config["model"]["expect_d_model"],
        expect_vocab_size=config["model"]["expect_vocab_size"],
    )

    prompts = load_eval_prompts(config["eval"]["prompts_path"], model.tokenizer)
    results = []
    for prompt in prompts:
        positions = prompt["positions"]
        lens_logits, model_logits, input_ids = apply_dual(
            lens, model, prompt["text"], layers=layers, positions=positions
        )
        logit_lens_logits, _, _ = apply_dual(
            lens, model, prompt["text"], layers=layers, positions=positions,
            use_jacobian=False,
        )
        control_logits = {
            name: c.apply(model, prompt["text"], layers=layers, positions=positions)[0]
            for name, c in controls.items()
        }
        model_top1 = model_logits["pre"].argmax(-1)  # [n_positions]

        per_layer = {}
        for layer in layers:
            variants = {
                "jlens": lens_logits[layer]["pre"],
                "logit_lens": logit_lens_logits[layer]["pre"],
                **{name: logits[layer].float() for name, logits in control_logits.items()},
            }
            per_layer[layer] = {
                "readouts": [
                    {
                        "position": pos,
                        "jlens_topk": topk_readout(
                            lens_logits[layer]["pre"][i],
                            lens_logits[layer]["capped"][i],
                            model.tokenizer,
                            top_k,
                        ),
                        "logit_lens_topk": topk_readout(
                            logit_lens_logits[layer]["pre"][i],
                            logit_lens_logits[layer]["capped"][i],
                            model.tokenizer,
                            top_k,
                        ),
                    }
                    for i, pos in enumerate(positions)
                ],
                "metrics": {
                    name: {
                        "topk_overlap_with_model": round(
                            topk_overlap(logits, model_logits["pre"], top_k), 4
                        ),
                        "rank_of_model_top1": [
                            int(r) for r in ranks_of_targets(logits, model_top1)
                        ],
                    }
                    for name, logits in variants.items()
                },
            }
        results.append(
            {
                "slug": prompt["slug"],
                "format": prompt["format"],
                "seq_len": int(input_ids.shape[1]),
                "positions": positions,
                "model_top1_tokens": [
                    model.tokenizer.decode([int(t)]) for t in model_top1
                ],
                "layers": per_layer,
            }
        )
        logger.info("evaluated %s (%s)", prompt["slug"], prompt["format"])

    payload = {
        "config_fingerprint": config_fingerprint(config),
        "lens_path": os.path.abspath(args.lens),
        "lens_n_prompts": lens.n_prompts,
        "layers": list(layers),
        "top_k": top_k,
        "control_seed": control_seed,
        "load_info": load_info,
        "architecture_report": report.to_dict(),
        "results": results,
        "environment": environment_manifest(),
        "notes": (
            "Rankings are identical for pre-softcap and softcapped logits "
            "(the cap is monotonic); logit values differ. Plain-text and "
            "chat-format prompts are evaluated separately; the fitted corpus "
            "was plain text only."
        ),
    }
    write_metadata(os.path.join(output_dir, "apply_results.json"), payload)
    _write_summary(os.path.join(output_dir, "apply_summary.md"), payload)
    logger.info("wrote %s", os.path.join(output_dir, "apply_results.json"))
    return 0


def _write_summary(path: str, payload: dict) -> None:
    lines = [
        "# J-lens vs logit lens vs controls — Gemma 4 E4B",
        "",
        f"- lens: `{payload['lens_path']}` (fitted on {payload['lens_n_prompts']} prompts)",
        f"- config fingerprint: `{payload['config_fingerprint']}`",
        f"- model: {payload['load_info']['model_repo_id']} @ {payload['load_info']['model_revision']}",
        "",
    ]
    for result in payload["results"]:
        lines.append(f"## {result['slug']} ({result['format']})")
        lines.append(f"model top-1 at positions {result['positions']}: "
                     f"{result['model_top1_tokens']}")
        lines.append("")
        lines.append("| layer | J-lens top-3 (last pos) | logit-lens top-3 (last pos) "
                     "| overlap J / logit / permuted / random |")
        lines.append("|---|---|---|---|")
        for layer, data in result["layers"].items():
            last = data["readouts"][-1]
            top3 = lambda r: " ".join(repr(t["token"]) for t in r[:3])  # noqa: E731
            metrics = data["metrics"]
            overlap = " / ".join(
                f"{metrics[name]['topk_overlap_with_model']:.2f}"
                for name in ("jlens", "logit_lens", "permuted", "random")
                if name in metrics
            )
            lines.append(
                f"| {layer} | {top3(last['jlens_topk'])} | "
                f"{top3(last['logit_lens_topk'])} | {overlap} |"
            )
        lines.append("")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
