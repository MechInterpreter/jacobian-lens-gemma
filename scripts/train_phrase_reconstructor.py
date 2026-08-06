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

**Resume** is at *batch* granularity. Every epoch writes
``checkpoints/reconstructor_epoch<NN>.pt``, and ``--checkpoint-every-steps N``
adds periodic ones inside an epoch; both carry the optimizer, every RNG stream,
the sampler order, and the epoch's running loss sums, so a resumed run continues
on the same trajectory and reports the same history. An interruption writes
``checkpoints/reconstructor_interrupt_*.pt`` — a distinct name — and exits 130.
The final ``reconstructor.pt`` is written only after the stage completes, so its
presence always means "training finished".
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from jlens.autoencoder.checkpoints import (  # noqa: E402
    find_training_checkpoint,
    load_training_checkpoint,
    restore_training_state,
    save_checkpoint,
    save_training_checkpoint,
)
from jlens.autoencoder.dataset import JSpaceLanguageDataset  # noqa: E402
from jlens.autoencoder.errors import AutoencoderError  # noqa: E402
from jlens.autoencoder.reconstructor import (  # noqa: E402
    PhraseEmbedder,
    build_reconstructor_training,
    evaluate_reconstructor,
    reconstructor_gate,
    train_reconstructor,
)
from jlens.autoencoder.runner import (  # noqa: E402
    add_common_args,
    build_identity,
    build_stack_from_args,
    configure_logging,
    handle_status_flags,
    prepare_run_dir,
    require_file,
    resolve_config,
    run_stage,
    stage_metadata,
    write_json,
)
from jlens.autoencoder.state import checkpoints_dir  # noqa: E402

STAGE = "reconstructor"
SCRIPT = "train_phrase_reconstructor"
CHECKPOINT_PREFIX = "reconstructor_"


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


def dataset_manifest_sha(dataset_dir: str) -> str | None:
    from jlens.autoencoder.state import read_json_if_valid

    manifest = read_json_if_valid(os.path.join(dataset_dir, "manifest.json")) or {}
    return manifest.get("records_sha256")


def validate_only(run_dir: str) -> int:
    path, rejected = find_training_checkpoint(
        checkpoints_dir(run_dir), kind="reconstructor", stage=STAGE, prefix=CHECKPOINT_PREFIX
    )
    print(f"reconstructor checkpoints under {checkpoints_dir(run_dir)}")
    print(f"  latest valid: {path or '(none)'}")
    for entry in rejected:
        print(f"    ! {os.path.basename(entry['path'])}: {entry['reason']}")
    final = os.path.join(run_dir, "reconstructor.pt")
    print(f"  reconstructor.pt: {'present' if os.path.isfile(final) else 'missing'}")
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
    dataset_dir = args.dataset_dir or os.path.join(run_dir, "dataset")
    require_file(os.path.join(dataset_dir, "records.jsonl"), what="dataset records")

    identity = build_identity(
        config,
        args,
        stage=STAGE,
        run_dir=run_dir,
        dataset_manifest_sha256=dataset_manifest_sha(dataset_dir),
        stage_config=dataclasses.asdict(config.reconstructor),
    )

    def body(*, guard, plan, state) -> int:
        stack = build_stack_from_args(config, args)
        dataset = JSpaceLanguageDataset.load(dataset_dir)
        dataset.assert_usable()
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not stack.is_mock else "cpu"
        )

        phrases = phrase_universe(dataset, config)
        token_map = stack.token_id_map(phrases)
        embedder = PhraseEmbedder(
            stack.model, max_phrase_tokens=config.reconstructor.max_phrase_tokens
        )

        resume_path = None
        if plan.action == "resume":
            resume_path, rejected = find_training_checkpoint(
                checkpoints_dir(run_dir),
                kind="reconstructor",
                stage=STAGE,
                prefix=CHECKPOINT_PREFIX,
            )
            for entry in rejected:
                log.warning("skipping unusable checkpoint %s: %s", entry["path"], entry["reason"])
        # ``seed_global`` only when starting fresh: on a resume the global RNG
        # comes from the checkpoint, and reseeding here would silently move the
        # dropout stream the restored trajectory depends on.
        model, optimizer, generator = build_reconstructor_training(
            dataset, config=config.reconstructor, device=device, seed_global=resume_path is None
        )
        start_epoch = start_batch = global_step = 0
        history: list[dict] = []
        resume_order = None
        resume_partial: dict = {}
        if resume_path:
            payload = load_training_checkpoint(
                resume_path, expect_kind="reconstructor", expect_stage=STAGE
            )
            report = restore_training_state(
                payload, model, optimizer=optimizer, generators={"sampler": generator}
            )
            history = report["history"]
            global_step = report["global_step"]
            if report["reason"] == "epoch_complete":
                start_epoch = report["epoch"] + 1
            else:
                start_epoch = report["epoch"]
                start_batch = report["batch_index"]
                resume_order = report["sampler_order"]
                resume_partial = (payload["metadata"].get("extra") or {}).get(
                    "partial_epoch"
                ) or {}
            print(
                f"[{STAGE}] resuming from {resume_path} at epoch {start_epoch} "
                f"batch {start_batch} (step {global_step})",
                flush=True,
            )
            if start_epoch >= int(config.reconstructor.epochs):
                log.info("training already complete at epoch %d; scoring and freezing", start_epoch)

        def checkpoint(*, reason, epoch, batch_index, global_step, order, partial, metrics, history):
            name = (
                f"{CHECKPOINT_PREFIX}epoch{epoch:03d}.pt"
                if reason == "epoch_complete"
                else f"{CHECKPOINT_PREFIX}"
                + ("interrupt" if reason == "keyboard_interrupt" else "step")
                + f"_{epoch:03d}_{batch_index:05d}.pt"
            )
            path = os.path.join(checkpoints_dir(run_dir), name)
            save_training_checkpoint(
                path,
                model,
                kind="reconstructor",
                stage=STAGE,
                identity=identity,
                reason=reason,
                config=dataclasses.asdict(config.reconstructor),
                epoch=epoch,
                batch_index=batch_index,
                global_step=global_step,
                optimizer=optimizer,
                sampler_order=order,
                generators={"sampler": generator},
                metrics=metrics,
                history=history,
                extra={"partial_epoch": partial, "d_model": dataset.d_model},
            )
            state.write(
                status="in_progress",
                identity=identity,
                checkpoint_path=path,
                reason=reason,
            )
            state.write_progress(
                {
                    "completed_units": int(global_step),
                    "expected_units": None,
                    "epoch": int(epoch),
                    "batch_index": int(batch_index),
                    "granularity": "optimizer step",
                }
            )
            return path

        def on_epoch(epoch: int, metrics: dict) -> None:
            log.info(
                "epoch %d: loss=%.4f cos=%.4f",
                epoch,
                metrics["loss"],
                metrics["train_prototype_cosine"],
            )

        model, summary = train_reconstructor(
            dataset,
            embedder,
            config=config.reconstructor,
            source_layer=stack.source_layer,
            phrase_token_ids=token_map,
            device=device,
            log=log,
            on_epoch=on_epoch,
            model=model,
            optimizer=optimizer,
            generator=generator,
            start_epoch=start_epoch,
            start_batch=start_batch,
            global_step=global_step,
            history=history,
            resume_order=resume_order,
            resume_partial=resume_partial,
            checkpoint=checkpoint,
            checkpoint_every_steps=int(args.checkpoint_every_steps or 0),
            guard=guard,
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
                    "resume_granularity": summary["resume_granularity"],
                },
            ),
        )
        state.mark_complete(
            identity=identity,
            artifacts={"reconstructor": checkpoint_path},
            detail={
                "gate": gate,
                "global_step": summary["global_step"],
                "state_dict_sha256": checkpoint_metadata["state_dict_sha256"],
            },
        )
        log.info("%s — %s", gate["verdict"], gate["message"])
        if not gate["passed"] and not args.ignore_gate:
            log.error(
                "STOPPING before the verbalizer is trained. Rerun with --ignore-gate "
                "only if you intend to record a run whose reconstructor gate failed."
            )
            return 3
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
