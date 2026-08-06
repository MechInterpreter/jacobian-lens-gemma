# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Build the (phrase, cone) dataset for the J-space language autoencoder.

    python scripts/build_jspace_language_dataset.py \
        --config configs/jspace_language_autoencoder.yaml \
        --allow-model-load --device-map cuda \
        --runs-root <DRIVE>/runs --output-dir <DRIVE>/runs/jlang_pilot

    # CPU smoke, no checkpoint, no downloads:
    python scripts/build_jspace_language_dataset.py \
        --config configs/jspace_language_autoencoder.yaml --smoke \
        --output-dir /tmp/jlang_smoke

Mines cohesive 2-6 Gemma-token noun phrases and named entities from
WikiText-103 (or the deterministic mock corpus), collects several natural
occurrences of each, captures the layer-14 activation immediately before each
occurrence begins, runs the frozen k=10 nonnegative pursuit against the layer's
J-space dictionary, and writes records, tensors, and a manifest.

``--benchmark`` runs one small slice first and prints measured seconds,
projected wall time, projected storage, and peak GPU memory **before** the full
build is attempted, as the experiment brief requires. ``--benchmark-only`` stops
there.

**Resume.** The build writes one atomic shard per pursuit chunk into
``<output-dir>/dataset/shards/`` as it goes. Rerunning validates the shards
already there and computes only the missing ones; the final ``records.jsonl`` /
``tensors.pt`` / ``manifest.json`` are assembled deterministically from the
complete set, so an interrupted build produces the same dataset as an
uninterrupted one. ``--dataset-shard-size`` sets how many occurrences a shard
holds (snapped up to a whole number of ``dataset.capture_batch_size`` chunks, so
pursuit batching — and therefore the numbers — never changes).
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jlens.autoencoder.dataset import (  # noqa: E402
    JSpaceLanguageDataset,
    assert_no_split_leakage,
    benchmark_build,
    save_dataset,
)
from jlens.autoencoder.dataset_shards import (  # noqa: E402
    build_dataset_sharded,
    validate_shards,
)
from jlens.autoencoder.errors import AutoencoderError  # noqa: E402
from jlens.autoencoder.pipeline import resolve_documents  # noqa: E402
from jlens.autoencoder.runner import (  # noqa: E402
    add_common_args,
    build_identity,
    build_stack_from_args,
    configure_logging,
    handle_status_flags,
    prepare_run_dir,
    resolve_config,
    run_stage,
    stage_metadata,
    write_json,
)
from jlens.autoencoder.state import dataset_shards_dir, sha256_of_file  # noqa: E402

STAGE = "dataset"
SCRIPT = "build_jspace_language_dataset"


def parse_args() -> argparse.Namespace:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Measure one small slice and print projected cost before building",
    )
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--benchmark-phrases", type=int, default=4)
    parser.add_argument(
        "--limit-phrases", type=int, default=None, help="Override dataset.n_phrases"
    )
    parser.add_argument(
        "--dataset-shard-size",
        type=int,
        default=None,
        help="Occurrences per build shard (default: dataset.capture_batch_size)",
    )
    return parser.parse_args()


def validate_only(run_dir: str) -> int:
    """Report the shard and artifact state of the dataset stage, then stop."""
    shard_dir = dataset_shards_dir(run_dir)
    found = validate_shards(shard_dir)
    print(f"dataset shards under {shard_dir}")
    print(f"  valid:   {len(found['valid'])}")
    print(f"  invalid: {len(found['invalid'])}")
    for entry in found["invalid"]:
        print(f"    ! {os.path.basename(entry['path'])}: {entry['reason']}")
    for name in ("records.jsonl", "tensors.pt", "manifest.json"):
        path = os.path.join(run_dir, "dataset", name)
        status = sha256_of_file(path)[:23] if os.path.isfile(path) else "missing"
        print(f"  {name:<16} {status}")
    return 0


def main() -> int:
    args = parse_args()
    log = configure_logging(args.log_level)
    config = resolve_config(args)
    run_dir = prepare_run_dir(args)
    early = handle_status_flags(args, run_dir)
    if early is not None:
        return early
    if args.validate_state:
        return validate_only(run_dir)

    identity = build_identity(
        config,
        args,
        stage=STAGE,
        run_dir=run_dir,
        stage_config={
            "shard_size": args.dataset_shard_size,
            "limit_phrases": args.limit_phrases,
        },
    )

    def body(*, guard, plan, state) -> int:
        log.info("building dataset into %s", run_dir)
        stack = build_stack_from_args(config, args)
        documents = resolve_documents(config, stack)
        log.info("corpus: %d documents (%s)", len(documents), config.dataset.corpus)

        benchmark = None
        if args.benchmark or args.benchmark_only:
            benchmark = benchmark_build(
                stack.model,
                stack.dictionary,
                documents,
                dataset_config=config.dataset,
                pursuit_config=config.pursuit,
                phrase_token_ids=stack.phrase_token_ids,
                n_phrases=int(args.benchmark_phrases),
            )
            write_json(os.path.join(run_dir, "artifacts", "benchmark.json"), benchmark)
            measured = benchmark["measured"]
            projection = benchmark["projection"]
            log.info(
                "benchmark: %.4f s/occurrence, peak CUDA %s GB",
                measured["seconds_per_occurrence"],
                measured["peak_cuda_memory_gb"],
            )
            log.info(
                "projected full build: %s occurrences, %.2f minutes, %.2f MB "
                "(linear extrapolation from the measured slice)",
                projection["planned_occurrences"],
                projection["estimated_wall_minutes"],
                projection["estimated_storage_mb"],
            )
            if args.benchmark_only:
                write_json(
                    os.path.join(run_dir, "dataset_metadata.json"),
                    stage_metadata(
                        "dataset_benchmark", config, args, stack, payload={"benchmark": benchmark}
                    ),
                )
                return 0

        dataset_dir = os.path.join(run_dir, "dataset")
        shard_dir = dataset_shards_dir(run_dir)
        if not args.resume or args.force:
            # A restart re-derives the plan, whose fingerprint no longer matches
            # the old shards; they are left on disk and simply ignored rather
            # than deleted, so nothing computed is ever destroyed by a flag.
            log.info("not resuming: existing shards under %s will be ignored", shard_dir)

        result = build_dataset_sharded(
            stack.model,
            stack.dictionary,
            documents,
            dataset_config=config.dataset,
            pursuit_config=config.pursuit,
            phrase_token_ids=stack.phrase_token_ids,
            shard_dir=shard_dir,
            provenance={
                "source_layer": stack.source_layer,
                "pursuit_k": config.pursuit.k,
                "dictionary": dict(stack.dictionary.provenance),
                "model_repo_id": config.model.repo_id,
                "model_revision": config.model.revision,
                "mock": stack.is_mock,
            },
            progress=log.info,
            max_phrases=args.limit_phrases,
            shard_size=args.dataset_shard_size,
            guard=guard,
            on_progress=state.write_progress,
        )
        manifest = save_dataset(
            dataset_dir,
            result,
            manifest_extra={
                "config_fingerprint": config.fingerprint(),
                "source_layer": stack.source_layer,
                "pursuit_k": config.pursuit.k,
                "corpus": config.dataset.corpus,
                "mock": stack.is_mock,
            },
        )
        dataset = JSpaceLanguageDataset.load(dataset_dir)
        dataset.assert_usable()
        leakage = assert_no_split_leakage(
            dataset,
            salt=config.dataset.split_salt,
            val_fraction=config.dataset.val_fraction,
            heldout_fraction=config.dataset.heldout_fraction,
        )
        write_json(os.path.join(run_dir, "artifacts", "leakage.json"), leakage)
        write_json(
            os.path.join(run_dir, "dataset_metadata.json"),
            stage_metadata(
                "dataset",
                config,
                args,
                stack,
                payload={
                    "dataset_dir": dataset_dir,
                    "manifest": manifest,
                    "leakage": leakage,
                    "benchmark": benchmark,
                    "shard_dir": shard_dir,
                },
            ),
        )
        state.mark_complete(
            identity=identity,
            artifacts={
                "records": os.path.join(dataset_dir, "records.jsonl"),
                "tensors": os.path.join(dataset_dir, "tensors.pt"),
                "manifest": os.path.join(dataset_dir, "manifest.json"),
            },
            detail={
                "n_records": len(dataset),
                "n_phrases": manifest["n_phrases"],
                "phrase_split_counts": result.stats["phrase_split_counts"],
                "sharded_build": result.stats["sharded_build"],
            },
        )
        log.info(
            "wrote %d records / %d phrases (%s) to %s",
            len(dataset),
            manifest["n_phrases"],
            result.stats["phrase_split_counts"],
            dataset_dir,
        )
        return 0

    return run_stage(
        STAGE, args=args, run_dir=run_dir, identity=identity, script=SCRIPT, body=body, log=log
    )


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AutoencoderError as exc:
        print(f"ABORT: {exc}", file=sys.stderr)
        sys.exit(2)
