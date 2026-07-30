# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Check benchmark target tokenization against the pinned Gemma tokenizer.

    python scripts/validate_benchmark_targets.py \
        --config configs/gemma_generative_validation.yaml [--split dev]

Loads the **tokenizer only** (a few MB, no model weights, no GPU), resolves
every target phrase, and applies the same 2-6 token requirement the run
enforces via :func:`jlens.generative.validate_target_tokens`. Run this before
spending GPU time: a target that tokenizes to one token cannot test multi-token
scoring, and forces prompt_only / constant / decaying to identical target
log-probabilities.

Targets are resolved exactly as the run resolves them: **contextually**, as the
assistant continuation of each configured receiver prompt at the config's
``receiver.format`` (see :func:`jlens.generative.contextual_target_token_ids`).
A phrase can segment differently after ``<start_of_turn>model`` than it does
standalone, so a standalone check would validate something the run does not use.

Prints a per-example table (token count, ids, per-token strings) for every
receiver prompt and split, then exits non-zero if any example in the checked
splits violates the requirement — so it works as a pre-run gate in a shell
pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jlens.gemma4 import resolve_revision
from jlens.generative import (
    MAX_TARGET_TOKENS,
    MIN_TARGET_TOKENS,
    GenerativeError,
    contextual_target_resolver,
    contextual_target_token_ids,
    encode_receiver_prompt,
    load_benchmark,
    receiver_format_from_config,
    resolve_neutral_prompt,
    validate_target_tokens,
)
from jlens.metadata import load_generative_config

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--split",
        choices=("dev", "heldout", "both"),
        default="both",
        help="Which split(s) must satisfy the requirement (default: both).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_generative_config(args.config)

    manifest_path = config["benchmark"]["manifest_path"]
    if not os.path.isabs(manifest_path):
        manifest_path = os.path.join(REPO_ROOT, manifest_path)
    manifest = load_benchmark(manifest_path)

    bounds = config["benchmark"].get("target_token_bounds") or {}
    min_tokens = int(bounds.get("min", MIN_TARGET_TOKENS))
    max_tokens = int(bounds.get("max", MAX_TARGET_TOKENS))

    import transformers

    repo_id = config["model"]["repo_id"]
    revision = resolve_revision(
        repo_id, config["model"]["revision"], token=os.environ.get("HF_TOKEN")
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        repo_id, revision=revision, token=os.environ.get("HF_TOKEN")
    )
    receiver_format = receiver_format_from_config(config)
    prompt_ids = list(config["neutral_prompts"])
    print(f"tokenizer: {repo_id}@{revision} ({type(tokenizer).__name__})")
    print(f"receiver format: {receiver_format}; prompts: {prompt_ids}")
    print(f"requirement: {min_tokens}-{max_tokens} tokens per target\n")

    receivers = {
        prompt_id: encode_receiver_prompt(
            tokenizer,
            resolve_neutral_prompt(prompt_id),
            prompt_id=prompt_id,
            receiver_format=receiver_format,
        )
        for prompt_id in prompt_ids
    }

    splits = ("dev", "heldout") if args.split == "both" else (args.split,)
    failures: dict[str, str] = {}
    report: dict[str, dict] = {}

    for prompt_id, receiver in receivers.items():
        anchor = receiver.steering_anchor
        print(
            f"=== receiver {prompt_id} (prompt_len {anchor['prompt_len']}, "
            f"anchor {anchor['anchor_index']} = "
            f"{anchor['anchor_token_string']!r}) ==="
        )
        print(f"  rendered: {receiver.rendered_prompt!r}")
        for split in ("dev", "heldout"):
            print(f"  --- {split} ---")
            for example in manifest[split]:
                info = contextual_target_token_ids(
                    tokenizer, receiver, example["target_phrase"]
                )
                ids = info["target_token_ids"]
                ok = min_tokens <= len(ids) <= max_tokens
                mark = "ok " if ok else "BAD"
                print(
                    f"    [{mark}] {example['example_id']:34s} n={len(ids)}  "
                    f"{example['target_phrase']!r} -> "
                    f"{info['target_token_strings']} {ids}"
                )
                report.setdefault(prompt_id, {})[example["example_id"]] = {
                    "split": split,
                    "target_phrase": example["target_phrase"],
                    "satisfies_requirement": ok,
                    **info,
                }
            print()

    # Authoritative pass/fail via the same function the run uses, so this script
    # can never disagree with the run's gate.
    for prompt_id, receiver in receivers.items():
        for split in splits:
            try:
                validate_target_tokens(
                    manifest[split],
                    tokenizer,
                    resolve=contextual_target_resolver(receiver),
                    min_tokens=min_tokens,
                    max_tokens=max_tokens,
                    use_manifest_ids=False,
                )
            except GenerativeError as exc:
                failures[f"{prompt_id}/{split}"] = str(exc)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if failures:
        for key, message in failures.items():
            print(f"\nFAILED [{key}]: {message}", file=sys.stderr)
        return 1
    print(
        f"\nAll targets in {', '.join(splits)} satisfy the requirement under "
        f"every configured receiver prompt ({receiver_format})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
