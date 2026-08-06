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

**Resume.** Warm start and preference are *two separate stages* with separate
state, separate checkpoints, and separate completion markers:

* warm start writes ``adapter_epoch<NN>.pt`` (in the run root, as before) and
  ``checkpoints/adapter_warm_*.pt``;
* preference writes ``checkpoints/adapter_preference_*.pt`` and never reads a
  warm-start checkpoint — the stage name is recorded in the checkpoint and
  checked on load, so the two cannot be confused even by filename.

The warm-start artifact ``adapter_warm.pt`` is preserved and is what the
preference stage anchors its DPO reference policy to, so a resumed preference
run optimizes against the same reference as an uninterrupted one.
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
    find_training_checkpoint,
    load_checkpoint,
    load_training_checkpoint,
    restore_module,
    restore_training_state,
    save_checkpoint,
    save_training_checkpoint,
    state_dict_sha256,
)
from jlens.autoencoder.dataset import JSpaceLanguageDataset  # noqa: E402
from jlens.autoencoder.errors import AutoencoderError  # noqa: E402
from jlens.autoencoder.preference import BeamCache, train_preference  # noqa: E402
from jlens.autoencoder.reconstructor import (  # noqa: E402
    PhraseEmbedder,
    PhraseReconstructor,
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
from jlens.autoencoder.state import checkpoints_dir, read_json_if_valid  # noqa: E402

WARM_STAGE = "adapter_warm"
PREFERENCE_STAGE = "adapter_preference"
SCRIPT = "train_cone_adapter"
WARM_PREFIX = "adapter_warm_"
PREFERENCE_PREFIX = "adapter_preference_"


def parse_args() -> argparse.Namespace:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument(
        "--reconstructor", default=None, help="Defaults to <output-dir>/reconstructor.pt"
    )
    parser.add_argument("--skip-preference", action="store_true")
    parser.add_argument(
        "--require-gate",
        action="store_true",
        help="Refuse to train if the reconstructor gate did not pass",
    )
    parser.add_argument(
        "--force-stage",
        default=None,
        choices=(WARM_STAGE, PREFERENCE_STAGE),
        help="Restart only this sub-stage (--force restarts both)",
    )
    parser.add_argument(
        "--beam-cache",
        dest="beam_cache",
        action="store_true",
        default=True,
        help="Cache preference beams by (q, adapter weights, decoding, prompt) so an "
        "interruption during candidate generation is not repaid on resume (default)",
    )
    parser.add_argument("--no-beam-cache", dest="beam_cache", action="store_false")
    parser.add_argument("--beam-cache-capacity", type=int, default=512)
    return parser.parse_args()


def dataset_manifest_sha(dataset_dir: str) -> str | None:
    manifest = read_json_if_valid(os.path.join(dataset_dir, "manifest.json")) or {}
    return manifest.get("records_sha256")


def stage_args(args: argparse.Namespace, stage: str) -> argparse.Namespace:
    """``args`` as this sub-stage sees it.

    ``--force`` restarts both sub-stages; ``--force-stage`` restarts exactly one,
    which is what you want when preference training needs redoing but the warm
    start — the expensive half — does not.
    """
    view = argparse.Namespace(**vars(args))
    forced = bool(args.force) or args.force_stage == stage
    view.force = forced
    return view


def validate_only(run_dir: str) -> int:
    for stage, prefix in ((WARM_STAGE, WARM_PREFIX), (PREFERENCE_STAGE, PREFERENCE_PREFIX)):
        path, rejected = find_training_checkpoint(
            checkpoints_dir(run_dir), kind="adapter", stage=stage, prefix=prefix
        )
        print(f"{stage} checkpoints ({prefix}*)")
        print(f"  latest valid: {path or '(none)'}")
        for entry in rejected:
            print(f"    ! {os.path.basename(entry['path'])}: {entry['reason']}")
    for name in ("adapter_warm.pt", "adapter.pt"):
        path = os.path.join(run_dir, name)
        print(f"  {name:<18} {'present' if os.path.isfile(path) else 'missing'}")
    return 0


class Sub:
    """The objects both sub-stages share, built once."""

    def __init__(self, args, config, log, run_dir, dataset_dir, reconstructor_path):
        self.args = args
        self.config = config
        self.log = log
        self.run_dir = run_dir
        self.stack = build_stack_from_args(config, args)
        self.dataset = JSpaceLanguageDataset.load(dataset_dir)
        self.dataset.assert_usable()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not self.stack.is_mock else "cpu"
        )
        payload = load_checkpoint(reconstructor_path, expect_kind="reconstructor")
        self.gate = payload["metadata"]["extra"].get("gate", {})
        if args.require_gate and not self.gate.get("passed"):
            raise AutoencoderError(
                f"the reconstructor gate did not pass ({self.gate.get('verdict')}); "
                f"refusing to train the verbalizer against it"
            )
        self.reconstructor = PhraseReconstructor(
            d_model=self.dataset.d_model, config=config.reconstructor
        )
        self.reconstructor_metadata = restore_module(self.reconstructor, payload)
        self.reconstructor.eval()
        for parameter in self.reconstructor.parameters():
            parameter.requires_grad_(False)
        self.reconstructor = self.reconstructor.to(self.device)
        self.reconstructor_sha = self.reconstructor_metadata["state_dict_sha256"]
        log.info(
            "reconstructor restored (%s), gate=%s",
            self.reconstructor_sha[:19],
            self.gate.get("verdict"),
        )
        phrases = list(
            dict.fromkeys(
                [r["phrase"] for r in self.dataset.records]
                + list(config.evaluation.confabulation_attractors)
            )
        )
        self.token_map = self.stack.token_id_map(phrases)
        self.target_map = self.stack.target_id_map(phrases)
        self.embedder = PhraseEmbedder(
            self.stack.model, max_phrase_tokens=config.reconstructor.max_phrase_tokens
        )

    def new_adapter(self) -> ConeAdapter:
        return ConeAdapter(
            d_model=self.dataset.d_model,
            config=self.config.adapter,
            target_rms=self.stack.memory_scale["embedding_rms"],
        ).to(self.device)


def _checkpoint_writer(*, stage, prefix, run_dir, identity, adapter, optimizer, generator, config, state):
    """A ``checkpoint(...)`` callback for one sub-stage.

    The stage name goes into both the filename and the checkpoint body; the body
    is what discovery checks, so renaming a file cannot make a warm-start
    checkpoint load as preference state.
    """

    def checkpoint(*, reason, epoch, batch_index, global_step, order, partial, metrics, history):
        suffix = {
            "epoch_complete": f"epoch{epoch:03d}",
            "keyboard_interrupt": f"interrupt_{epoch:03d}_{batch_index:05d}",
            "periodic": f"step_{epoch:03d}_{batch_index:05d}",
            "stage_complete": f"final_{epoch:03d}",
        }[reason]
        path = os.path.join(checkpoints_dir(run_dir), f"{prefix}{suffix}.pt")
        save_training_checkpoint(
            path,
            adapter,
            kind="adapter",
            stage=stage,
            identity=identity,
            reason=reason,
            config=config,
            epoch=epoch,
            batch_index=batch_index,
            global_step=global_step,
            optimizer=optimizer,
            sampler_order=order,
            generators={"sampler": generator},
            metrics=metrics,
            history=history,
            extra={"partial_epoch": partial, "stage": stage},
        )
        state.write(status="in_progress", identity=identity, checkpoint_path=path, reason=reason)
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

    return checkpoint


def _restore(prefix, stage, run_dir, adapter, optimizer, generator, log, plan):
    """Find and load this sub-stage's newest valid checkpoint, if resuming."""
    blank = {
        "path": None,
        "start_epoch": 0,
        "start_batch": 0,
        "global_step": 0,
        "history": [],
        "order": None,
        "partial": {},
    }
    if plan.action != "resume":
        return blank
    path, rejected = find_training_checkpoint(
        checkpoints_dir(run_dir), kind="adapter", stage=stage, prefix=prefix
    )
    for entry in rejected:
        log.warning("skipping unusable checkpoint %s: %s", entry["path"], entry["reason"])
    if not path:
        return blank
    payload = load_training_checkpoint(path, expect_kind="adapter", expect_stage=stage)
    report = restore_training_state(
        payload, adapter, optimizer=optimizer, generators={"sampler": generator}
    )
    resumed = dict(blank)
    resumed["path"] = path
    resumed["history"] = report["history"]
    resumed["global_step"] = report["global_step"]
    if report["reason"] == "epoch_complete":
        resumed["start_epoch"] = report["epoch"] + 1
    else:
        resumed["start_epoch"] = report["epoch"]
        resumed["start_batch"] = report["batch_index"]
        resumed["order"] = report["sampler_order"]
        resumed["partial"] = (payload["metadata"].get("extra") or {}).get("partial_epoch") or {}
    print(
        f"[{stage}] resuming from {path} at epoch {resumed['start_epoch']} "
        f"batch {resumed['start_batch']} (step {resumed['global_step']})",
        flush=True,
    )
    return resumed


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
    reconstructor_path = args.reconstructor or os.path.join(run_dir, "reconstructor.pt")
    require_file(os.path.join(dataset_dir, "records.jsonl"), what="dataset records")
    require_file(reconstructor_path, what="reconstructor checkpoint")

    sub = Sub(args, config, log, run_dir, dataset_dir, reconstructor_path)
    manifest_sha = dataset_manifest_sha(dataset_dir)
    warm_path = os.path.join(run_dir, "adapter_warm.pt")
    adapter_path = os.path.join(run_dir, "adapter.pt")
    warm_identity = build_identity(
        config,
        args,
        stage=WARM_STAGE,
        run_dir=run_dir,
        dataset_manifest_sha256=manifest_sha,
        reconstructor_sha256=sub.reconstructor_sha,
        stage_config=dataclasses.asdict(config.adapter),
    )

    warm_summary: dict = {"skipped": True}
    adapter = sub.new_adapter()
    log.info("adapter: %s", adapter.describe())
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=float(config.adapter.learning_rate),
        weight_decay=float(config.adapter.weight_decay),
    )
    generator = torch.Generator().manual_seed(int(config.adapter.seed))

    def warm_body(*, guard, plan, state) -> int:
        nonlocal warm_summary, adapter
        resumed = _restore(WARM_PREFIX, WARM_STAGE, run_dir, adapter, optimizer, generator, log, plan)

        def on_epoch(epoch: int, metrics: dict, opt) -> None:
            # The historical resume-point location. Kept so an existing run
            # directory keeps working and nothing that read adapter_epoch*.pt
            # stops finding it.
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

        if resumed["start_epoch"] < int(config.adapter.epochs):
            adapter, warm_summary = train_adapter_warm_start(
                sub.stack.model,
                adapter,
                sub.dataset,
                sub.stack.prompt,
                config=config.adapter,
                conditioner=sub.stack.conditioner,
                phrase_targets=sub.target_map,
                pad_token_id=sub.stack.pad_token_id,
                device=sub.device,
                log=log,
                on_epoch=on_epoch,
                start_epoch=resumed["start_epoch"],
                start_batch=resumed["start_batch"],
                global_step=resumed["global_step"],
                optimizer=optimizer,
                generator=generator,
                history=resumed["history"],
                resume_order=resumed["order"],
                resume_partial=resumed["partial"],
                checkpoint=_checkpoint_writer(
                    stage=WARM_STAGE,
                    prefix=WARM_PREFIX,
                    run_dir=run_dir,
                    identity=warm_identity,
                    adapter=adapter,
                    optimizer=optimizer,
                    generator=generator,
                    config=dataclasses.asdict(config.adapter),
                    state=state,
                ),
                checkpoint_every_steps=int(args.checkpoint_every_steps or 0),
                guard=guard,
            )
        else:
            log.info("warm start already complete at epoch %d; skipping", resumed["start_epoch"])
            warm_summary = {"skipped": True, "history": resumed["history"]}
        save_checkpoint(
            warm_path,
            adapter,
            kind="adapter",
            config=dataclasses.asdict(config.adapter),
            metrics={"warm_start": warm_summary.get("history", [])},
            extra={
                "stage": "warm_start_final",
                "reconstructor_sha256": sub.reconstructor_sha,
                "target_rms": float(adapter.target_rms),
            },
        )
        state.mark_complete(
            identity=warm_identity,
            artifacts={"adapter_warm": warm_path},
            detail={
                "history": warm_summary.get("history", []),
                "state_dict_sha256": state_dict_sha256(adapter),
            },
        )
        return 0

    code = run_stage(
        WARM_STAGE,
        args=stage_args(args, WARM_STAGE),
        run_dir=run_dir,
        identity=warm_identity,
        script=SCRIPT,
        body=warm_body,
        log=log,
    )
    if code != 0:
        return code
    # Whether the warm start ran here or was skipped as complete, the preference
    # stage must start from the warm-start artifact — not from whatever this
    # process happens to hold.
    require_file(warm_path, what="warm-start adapter")
    warm_payload = load_checkpoint(warm_path, expect_kind="adapter")
    restore_module(adapter, warm_payload)
    warm_sha = warm_payload["metadata"]["state_dict_sha256"]
    reference_state = {k: v.clone() for k, v in warm_payload["state_dict"].items()}

    preference_identity = build_identity(
        config,
        args,
        stage=PREFERENCE_STAGE,
        run_dir=run_dir,
        dataset_manifest_sha256=manifest_sha,
        reconstructor_sha256=sub.reconstructor_sha,
        adapter_warm_sha256=warm_sha,
        stage_config=dataclasses.asdict(config.preference),
    )
    preference_summary: dict = {"skipped": True}
    preference_ran = False

    def write_final_adapter() -> dict:
        return save_checkpoint(
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
                "reconstructor_sha256": sub.reconstructor_sha,
                "reconstructor_gate": sub.gate,
                "adapter_warm_sha256": warm_sha,
                "target_rms": float(adapter.target_rms),
                "memory_scale": sub.stack.memory_scale,
                "phrase_token_ids_sample": dict(list(sub.token_map.items())[:5]),
            },
        )

    def preference_body(*, guard, plan, state) -> int:
        nonlocal preference_summary, adapter, preference_ran
        preference_ran = True
        preference_optimizer = torch.optim.AdamW(
            adapter.parameters(), lr=float(config.preference.learning_rate)
        )
        preference_generator = torch.Generator().manual_seed(int(config.preference.seed))
        resumed = _restore(
            PREFERENCE_PREFIX,
            PREFERENCE_STAGE,
            run_dir,
            adapter,
            preference_optimizer,
            preference_generator,
            log,
            plan,
        )

        def on_preference_epoch(epoch: int, metrics: dict, opt) -> None:
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

        if resumed["start_epoch"] < int(config.preference.epochs):
            adapter, preference_summary = train_preference(
                sub.stack.model,
                adapter,
                sub.reconstructor,
                sub.embedder,
                sub.dataset,
                sub.stack.prompt,
                config=config.preference,
                conditioner=sub.stack.conditioner,
                stop_token_ids=sub.stack.stop_token_ids,
                pad_token_id=sub.stack.pad_token_id,
                beam_width=config.adapter.beam_width,
                max_new_tokens=config.adapter.max_new_tokens,
                source_layer=sub.stack.source_layer,
                accept_recon_min=config.evaluation.accept_recon_min,
                accept_margin_min=config.evaluation.accept_margin_min,
                device=sub.device,
                log=log,
                on_epoch=on_preference_epoch,
                start_epoch=resumed["start_epoch"],
                start_batch=resumed["start_batch"],
                global_step=resumed["global_step"],
                optimizer=preference_optimizer,
                generator=preference_generator,
                history=resumed["history"],
                resume_order=resumed["order"],
                resume_partial=resumed["partial"],
                reference_state=reference_state,
                beam_cache=BeamCache(
                    os.path.join(run_dir, "checkpoints", "beam_cache"),
                    capacity=int(args.beam_cache_capacity),
                    enabled=bool(args.beam_cache),
                ),
                checkpoint=_checkpoint_writer(
                    stage=PREFERENCE_STAGE,
                    prefix=PREFERENCE_PREFIX,
                    run_dir=run_dir,
                    identity=preference_identity,
                    adapter=adapter,
                    optimizer=preference_optimizer,
                    generator=preference_generator,
                    config=dataclasses.asdict(config.preference),
                    state=state,
                ),
                checkpoint_every_steps=int(args.checkpoint_every_steps or 0),
                guard=guard,
            )
        else:
            log.info(
                "preference training already complete at epoch %d; skipping",
                resumed["start_epoch"],
            )
            preference_summary = {"skipped": True, "history": resumed["history"]}
        # adapter.pt is written *inside* the stage, before the marker, so a
        # completion marker never names an artifact that does not exist yet.
        write_final_adapter()
        state.mark_complete(
            identity=preference_identity,
            artifacts={"adapter": adapter_path},
            detail={
                "history": preference_summary.get("history", []),
                "state_dict_sha256": state_dict_sha256(adapter),
            },
        )
        return 0

    if not args.skip_preference:
        code = run_stage(
            PREFERENCE_STAGE,
            args=stage_args(args, PREFERENCE_STAGE),
            run_dir=run_dir,
            identity=preference_identity,
            script=SCRIPT,
            body=preference_body,
            log=log,
        )
        if code != 0:
            return code

    if args.skip_preference or not preference_ran:
        # Either preference training was not asked for, or it was skipped as
        # already complete. In the second case ``adapter`` still holds the
        # warm-start weights, and the trained adapter is the one on disk —
        # loading it back is what keeps a skipped rerun from silently
        # overwriting a preference-trained adapter with the warm-start one.
        if not preference_ran and os.path.isfile(adapter_path):
            restore_module(adapter, load_checkpoint(adapter_path, expect_kind="adapter"))
            log.info("preference stage already complete; restored %s", adapter_path)
        else:
            write_final_adapter()
    checkpoint_metadata = load_checkpoint(adapter_path, expect_kind="adapter")["metadata"]
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
            sub.stack,
            payload={
                "checkpoint": adapter_path,
                "warm_start_checkpoint": warm_path,
                "warm_start_sha256": warm_sha,
                "checkpoint_metadata": checkpoint_metadata,
                "reconstructor_checkpoint": reconstructor_path,
                "reconstructor_sha256": sub.reconstructor_sha,
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
