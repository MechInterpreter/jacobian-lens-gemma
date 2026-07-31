# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Train the phrase reconstructor and run the gate that decides whether the
verbalizer is trained at all.

    python scripts/train_phrase_reconstructor.py \
        --config configs/jspace_language_autoencoder.yaml \
        --allow-model-load --device-map cuda \
        --output-dir <DRIVE>/runs/jlang_pilot

The reconstructor is trained on **training-split phrase prototypes only** and is
frozen permanently on the way out. The gate is evaluated on the **validation**
split (concept-disjoint from training) and, if it fails, this script exits
non-zero: the adapter must not be trained against a reconstructor that cannot
tell correct phrases from hard distractors. ``--ignore-gate`` records the
override explicitly rather than hiding it.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from jlens.autoencoder.checkpoints import save_checkpoint  # noqa: E402
from jlens.autoencoder.dataset import JSpaceLanguageDataset  # noqa: E402
from jlens.autoencoder.errors import AutoencoderError  # noqa: E402
from jlens.autoencoder.reconstructor import (  # noqa: E402
    PhraseEmbedder,
    evaluate_reconstructor,
    reconstructor_gate,
    train_reconstructor,
)
from jlens.autoencoder.runner import (  # noqa: E402
    add_common_args,
    build_stack_from_args,
    configure_logging,
    prepare_run_dir,
    require_file,
    resolve_config,
    stage_metadata,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--dataset-dir", default=None, help="Defaults to <output-dir>/dataset")
    parser.add_argument("--gate-split", default="val", choices=("val", "heldout"))
    parser.add_argument(
        "--ignore-gate",
        action="store_true",
        help="Continue past a failed gate (recorded in the artifacts)",
    )
    return parser.parse_args()


def phrase_universe(dataset: JSpaceLanguageDataset, config) -> list[str]:
    """Every phrase the reconstructor must be able to embed: the dataset's own
    phrases plus the confabulation attractors used as universal distractors."""
    phrases = [record["phrase"] for record in dataset.records]
    return list(dict.fromkeys([*phrases, *config.evaluation.confabulation_attractors]))


def main() -> int:
    args = parse_args()
    log = configure_logging(args.log_level)
    config = resolve_config(args)
    run_dir = prepare_run_dir(args)
    dataset_dir = args.dataset_dir or os.path.join(run_dir, "dataset")
    require_file(os.path.join(dataset_dir, "records.jsonl"), what="dataset records")

    stack = build_stack_from_args(config, args)
    dataset = JSpaceLanguageDataset.load(dataset_dir)
    dataset.assert_usable()
    device = torch.device("cuda" if torch.cuda.is_available() and not stack.is_mock else "cpu")

    phrases = phrase_universe(dataset, config)
    token_map = stack.token_id_map(phrases)
    embedder = PhraseEmbedder(
        stack.model, max_phrase_tokens=config.reconstructor.max_phrase_tokens
    )

    def on_epoch(epoch: int, metrics: dict) -> None:
        log.info("epoch %d: loss=%.4f cos=%.4f", epoch, metrics["loss"], metrics["train_prototype_cosine"])

    model, summary = train_reconstructor(
        dataset,
        embedder,
        config=config.reconstructor,
        source_layer=stack.source_layer,
        phrase_token_ids=token_map,
        device=device,
        log=log,
        on_epoch=on_epoch,
    )

    metrics = {}
    for split in ("train", "val", "heldout"):
        metrics[split] = evaluate_reconstructor(
            model,
            dataset,
            embedder,
            split=split,
            config=config.reconstructor,
            source_layer=stack.source_layer,
            phrase_token_ids=token_map,
            device=device,
            extra_distractors=config.evaluation.confabulation_attractors,
        )
        log.info(
            "%s: auroc=%s top5=%.3f explained=%.4f margin=%.4f",
            split,
            metrics[split]["auroc_correct_vs_distractor"],
            metrics[split]["top5_retrieval"],
            metrics[split]["mean_explained_fraction"],
            metrics[split]["mean_specificity_margin"],
        )
    gate = reconstructor_gate(metrics[args.gate_split], config=config.reconstructor)
    gate["ignored"] = bool(args.ignore_gate)
    write_json(os.path.join(run_dir, "artifacts", "reconstructor_metrics.json"), metrics)
    write_json(os.path.join(run_dir, "artifacts", "reconstructor_gate.json"), gate)

    checkpoint_path = os.path.join(run_dir, "reconstructor.pt")
    checkpoint_metadata = save_checkpoint(
        checkpoint_path,
        model,
        kind="reconstructor",
        config=dataclasses.asdict(config.reconstructor),
        metrics={split: {
            k: v for k, v in values.items() if k not in ("per_record", "prototype_dispersion")
        } for split, values in metrics.items()},
        extra={
            "source_layer": stack.source_layer,
            "d_model": dataset.d_model,
            "training_summary": {k: v for k, v in summary.items() if k != "history"},
            "history": summary["history"],
            "gate": gate,
            "phrase_token_ids": token_map,
            "frozen_after_training": True,
        },
    )
    write_json(
        os.path.join(run_dir, "reconstructor_metadata.json"),
        stage_metadata(
            "reconstructor",
            config,
            args,
            stack,
            payload={
                "checkpoint": checkpoint_path,
                "checkpoint_metadata": checkpoint_metadata,
                "gate": gate,
                "n_train_phrases": summary["n_train_phrases"],
                "trainable_parameters": summary["trainable_parameters"],
            },
        ),
    )
    log.info("%s — %s", gate["verdict"], gate["message"])
    if not gate["passed"] and not args.ignore_gate:
        log.error(
            "STOPPING before the verbalizer is trained. Rerun with --ignore-gate "
            "only if you intend to record a run whose reconstructor gate failed."
        )
        return 3
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AutoencoderError as exc:
        print(f"ABORT: {exc}", file=sys.stderr)
        sys.exit(2)
