# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Export a completed run into a static explorer bundle.

    python scripts/export_explorer_bundle.py \
        --run-dir runs/jspace_20260716T170808536780_e4118850fb70 \
        --out explorer/public/data/text_demo.json \
        [--prompts configs/prompts/eval_prompts_v2.json] \
        [--analysis-dir reports/<run_id>] \
        [--k 10] [--layers 14,21,28,35,38] \
        [--slugs slug1,slug2 | --demo-set | --all] \
        [--merge path/to/other_bundle.json ...] \
        [--schema schemas/explorer_bundle.schema.json]

Deterministic: the same inputs produce byte-identical output (creation time
comes from the source run's metadata, not the wall clock). Source artifacts
are read-only. ``--merge`` folds later bundles (causal smoke run, multimodal
capture) into the export; merging is how measured records replace fixtures.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jlens.explorer_export import (
    DEFAULT_DEMO_SLUGS,
    build_text_bundle,
    merge_bundles,
    write_bundle,
)
from jlens.metadata import local_git_commit

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SCHEMA = os.path.join(REPO_ROOT, "schemas", "explorer_bundle.schema.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="completed jspace run directory")
    parser.add_argument("--out", required=True, help="output bundle path (.json)")
    parser.add_argument(
        "--prompts",
        default=os.path.join(REPO_ROOT, "configs", "prompts", "eval_prompts_v2.json"),
        help="prompt-text source (eval_prompts_v2.json)",
    )
    parser.add_argument(
        "--analysis-dir", default=None, help="optional analysis report directory"
    )
    parser.add_argument("--k", type=int, default=10, help="cone sparsity to export")
    parser.add_argument(
        "--layers", default=None, help="comma-separated layer subset (default: all)"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--slugs", default=None, help="comma-separated prompt slugs")
    group.add_argument(
        "--demo-set",
        action="store_true",
        help=f"export the {len(DEFAULT_DEMO_SLUGS)}-example representative demo subset",
    )
    group.add_argument("--all", action="store_true", help="export every prompt (default)")
    parser.add_argument(
        "--merge",
        action="append",
        default=[],
        metavar="BUNDLE_JSON",
        help="merge another explorer bundle (causal / multimodal) into the export",
    )
    parser.add_argument("--schema", default=DEFAULT_SCHEMA, help="bundle schema path")
    parser.add_argument(
        "--no-validate", action="store_true", help="skip schema validation"
    )
    args = parser.parse_args(argv)

    layers = (
        [int(x) for x in args.layers.split(",")] if args.layers else None
    )
    if args.slugs:
        slugs = [s.strip() for s in args.slugs.split(",") if s.strip()]
    elif args.demo_set:
        slugs = list(DEFAULT_DEMO_SLUGS)
    else:
        slugs = None

    bundle, warnings = build_text_bundle(
        args.run_dir,
        prompts_path=args.prompts if os.path.isfile(args.prompts) else None,
        analysis_dir=args.analysis_dir,
        k=args.k,
        layers=layers,
        slugs=slugs,
        implementation_commit=local_git_commit(REPO_ROOT),
    )
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)

    for merge_path in args.merge:
        with open(merge_path, encoding="utf-8") as handle:
            extra = json.load(handle)
        bundle = merge_bundles(bundle, extra)
        print(f"merged: {merge_path}", file=sys.stderr)

    fingerprint = write_bundle(
        bundle, args.out, schema_path=None if args.no_validate else args.schema
    )
    print(
        f"wrote {args.out} ({len(bundle['examples'])} examples, "
        f"{len(bundle['cones'])} cones, {len(bundle['trajectories'])} transitions, "
        f"{len(bundle['causal_records'])} causal records) {fingerprint}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
