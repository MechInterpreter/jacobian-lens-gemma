# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Train the cone adapter: supervised warm start, then reconstructor-guided
preference optimization.

    python scripts/train_cone_adapter.py \
        --config configs/jspace_language_autoencoder.yaml \
        --allow-model-load --device-map cuda \
        --output-dir <DRIVE>/runs/jlang_pilot

Gemma, the tokenizer, the lens, and the reconstructor are frozen throughout;
only the adapter is optimized, and the optimizer's parameter identities are
checked against the frozen modules before the first step.

Resume: every epoch writes ``adapter_epoch<NN>.pt`` with optimizer and RNG state.
``--resume`` (default on) continues from the newest valid resume point, so a
terminated Colab runtime costs at most one epoch.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from jlens.autoencoder.adapter import (  # noqa: E402
    ConeAdapter,
    train_adapter_warm_start,
)
from jlens.autoencoder.checkpoints import (  # noqa: E402
    find_resume_checkpoint,
    load_checkpoint,
    restore_module,
    restore_rng,
    save_checkpoint,
)
from jlens.autoencoder.dataset import JSpaceLanguageDataset  # noqa: E402
from jlens.autoencoder.errors import AutoencoderError  # noqa: E402
from jlens.autoencoder.preference import train_preference  # noqa: E402
from jlens.autoencoder.reconstructor import (  # noqa: E402
    PhraseEmbedder,
    PhraseReconstructor,
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
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--reconstructor", default=None, help="Defaults to <output-dir>/reconstructor.pt")
    parser.add_argument("--skip-preference", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--require-gate",
        action="store_true",
        help="Refuse to train if the reconstructor gate did not pass",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log = configure_logging(args.log_level)
    config = resolve_config(args)
    run_dir = prepare_run_dir(args)
    dataset_dir = args.dataset_dir or os.path.join(run_dir, "dataset")
    reconstructor_path = args.reconstructor or os.path.join(run_dir, "reconstructor.pt")
    require_file(os.path.join(dataset_dir, "records.jsonl"), what="dataset records")
    require_file(reconstructor_path, what="reconstructor checkpoint")

    stack = build_stack_from_args(config, args)
    dataset = JSpaceLanguageDataset.load(dataset_dir)
    dataset.assert_usable()
    device = torch.device("cuda" if torch.cuda.is_available() and not stack.is_mock else "cpu")

    payload = load_checkpoint(reconstructor_path, expect_kind="reconstructor")
    gate = payload["metadata"]["extra"].get("gate", {})
    if args.require_gate and not gate.get("passed"):
        raise AutoencoderError(
            f"the reconstructor gate did not pass ({gate.get('verdict')}); refusing "
            f"to train the verbalizer against it"
        )
    reconstructor = PhraseReconstructor(d_model=dataset.d_model, config=config.reconstructor)
    reconstructor_metadata = restore_module(reconstructor, payload)
    reconstructor.eval()
    for parameter in reconstructor.parameters():
        parameter.requires_grad_(False)
    reconstructor = reconstructor.to(device)
    log.info(
        "reconstructor restored (%s), gate=%s",
        reconstructor_metadata["state_dict_sha256"][:19],
        gate.get("verdict"),
    )

    phrases = list(
        dict.fromkeys(
            [r["phrase"] for r in dataset.records] + list(config.evaluation.confabulation_attractors)
        )
    )
    token_map = stack.token_id_map(phrases)
    target_map = stack.target_id_map(phrases)
    embedder = PhraseEmbedder(
        stack.model, max_phrase_tokens=config.reconstructor.max_phrase_tokens
    )

    adapter = ConeAdapter(
        d_model=dataset.d_model,
        config=config.adapter,
        target_rms=stack.memory_scale["embedding_rms"],
    ).to(device)
    log.info("adapter: %s", adapter.describe())

    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=float(config.adapter.learning_rate),
        weight_decay=float(config.adapter.weight_decay),
    )
    start_epoch = 0
    if not args.no_resume:
        resume_path = find_resume_checkpoint(run_dir, kind="adapter")
        if resume_path:
            resume_payload = load_checkpoint(resume_path, expect_kind="adapter")
            restore_module(adapter, resume_payload)
            if "optimizer_state" in resume_payload:
                optimizer.load_state_dict(resume_payload["optimizer_state"])
            restore_rng(resume_payload)
            start_epoch = int(resume_payload["metadata"].get("epoch") or 0) + 1
            log.info("resuming warm start from %s at epoch %d", resume_path, start_epoch)

    def on_epoch(epoch: int, metrics: dict, opt: torch.optim.Optimizer) -> None:
        save_checkpoint(
            os.path.join(run_dir, f"adapter_epoch{epoch:03d}.pt"),
            adapter,
            kind="adapter",
            config=dataclasses.asdict(config.adapter),
            metrics=metrics,
            extra={"stage": "warm_start", "target_rms": float(adapter.target_rms)},
            optimizer=opt,
            epoch=epoch,
        )

    warm_summary: dict = {"skipped": True}
    if start_epoch < int(config.adapter.epochs):
        adapter, warm_summary = train_adapter_warm_start(
            stack.model,
            adapter,
            dataset,
            stack.prompt,
            config=config.adapter,
            conditioner=stack.conditioner,
            phrase_targets=target_map,
            pad_token_id=stack.pad_token_id,
            device=device,
            log=log,
            on_epoch=on_epoch,
            start_epoch=start_epoch,
            optimizer=optimizer,
        )
    else:
        log.info("warm start already complete at epoch %d; skipping", start_epoch)
    warm_path = os.path.join(run_dir, "adapter_warm.pt")
    save_checkpoint(
        warm_path,
        adapter,
        kind="adapter",
        config=dataclasses.asdict(config.adapter),
        metrics={"warm_start": warm_summary.get("history", [])},
        extra={
            "stage": "warm_start_final",
            "reconstructor_sha256": reconstructor_metadata["state_dict_sha256"],
            "target_rms": float(adapter.target_rms),
        },
    )

    preference_summary: dict = {"skipped": True}
    if not args.skip_preference:
        def on_preference_epoch(epoch: int, metrics: dict, opt: torch.optim.Optimizer) -> None:
            save_checkpoint(
                os.path.join(run_dir, f"adapter_preference_epoch{epoch:03d}.pt"),
                adapter,
                kind="adapter",
                config=dataclasses.asdict(config.preference),
                metrics=metrics,
                extra={"stage": "preference"},
                optimizer=opt,
                epoch=epoch,
            )

        adapter, preference_summary = train_preference(
            stack.model,
            adapter,
            reconstructor,
            embedder,
            dataset,
            stack.prompt,
            config=config.preference,
            conditioner=stack.conditioner,
            stop_token_ids=stack.stop_token_ids,
            pad_token_id=stack.pad_token_id,
            beam_width=config.adapter.beam_width,
            max_new_tokens=config.adapter.max_new_tokens,
            source_layer=stack.source_layer,
            accept_recon_min=config.evaluation.accept_recon_min,
            accept_margin_min=config.evaluation.accept_margin_min,
            device=device,
            log=log,
            on_epoch=on_preference_epoch,
        )

    adapter_path = os.path.join(run_dir, "adapter.pt")
    checkpoint_metadata = save_checkpoint(
        adapter_path,
        adapter,
        kind="adapter",
        config=dataclasses.asdict(config.adapter),
        metrics={
            "warm_start": warm_summary.get("history", []),
            "preference": preference_summary.get("history", []),
        },
        extra={
            "stage": "final",
            "reconstructor_sha256": reconstructor_metadata["state_dict_sha256"],
            "reconstructor_gate": gate,
            "target_rms": float(adapter.target_rms),
            "memory_scale": stack.memory_scale,
            "phrase_token_ids_sample": dict(list(token_map.items())[:5]),
        },
    )
    write_json(
        os.path.join(run_dir, "artifacts", "adapter_training.json"),
        {"warm_start": warm_summary, "preference": preference_summary},
    )
    write_json(
        os.path.join(run_dir, "adapter_metadata.json"),
        stage_metadata(
            "adapter",
            config,
            args,
            stack,
            payload={
                "checkpoint": adapter_path,
                "warm_start_checkpoint": warm_path,
                "checkpoint_metadata": checkpoint_metadata,
                "reconstructor_checkpoint": reconstructor_path,
                "reconstructor_sha256": reconstructor_metadata["state_dict_sha256"],
                "adapter": adapter.describe(),
            },
        ),
    )
    log.info("adapter written to %s", adapter_path)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AutoencoderError as exc:
        print(f"ABORT: {exc}", file=sys.stderr)
        sys.exit(2)
