# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Decisive evaluation on concept-disjoint held-out phrases, and the GO/NO-GO
report.

    python scripts/evaluate_jspace_language.py \
        --config configs/jspace_language_autoencoder.yaml \
        --allow-model-load --device-map cuda \
        --output-dir <DRIVE>/runs/jlang_pilot

Runs all eight baselines on the held-out split, the paraphrase-robustness sweep,
the cross-cone swap check, the confabulation-attractor probe, and the leakage
re-derivation, then writes ``artifacts/evaluation.json``,
``artifacts/gonogo.json``, and ``summary.md``.

The exit code carries the verdict: 0 for GO, 4 for NO-GO. A NO-GO is a result,
not an error — the artifacts are written either way.

**Resume.** Every expensive unit — each ``(record, baseline)``, each
``(paraphrase, record)``, and the attractor probe — is persisted as its own
atomic shard under ``<output-dir>/evaluation_shards/``. A rerun validates the
shards against the current adapter, reconstructor, and decoding settings, reuses
the ones that still apply, and generates only what is missing. Aggregation is a
pure function of the shards and refuses to mark the stage complete unless every
expected combination is present, so ``evaluation.json`` / ``gonogo.json`` /
``summary.md`` are the same whether the run was interrupted or not.
``--no-evaluation-sharding`` restores the old single-pass behaviour.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from jlens.autoencoder.adapter import ConeAdapter  # noqa: E402
from jlens.autoencoder.checkpoints import load_checkpoint, restore_module  # noqa: E402
from jlens.autoencoder.dataset import (  # noqa: E402
    JSpaceLanguageDataset,
    assert_no_split_leakage,
)
from jlens.autoencoder.errors import AutoencoderError  # noqa: E402
from jlens.autoencoder.evaluation import (  # noqa: E402
    evaluate_cross_cone_swap,
    gonogo_report,
    render_markdown,
)
from jlens.autoencoder.evaluation_shards import (  # noqa: E402
    EvaluationShardStore,
    build_confabulation_probe_sharded,
    evaluate_baselines_sharded,
    evaluate_prompt_robustness_sharded,
    evaluation_identity,
    expected_shard_names,
    missing_shards,
)
from jlens.autoencoder.reconstructor import (  # noqa: E402
    PhraseEmbedder,
    PhraseReconstructor,
    evaluate_reconstructor,
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
    write_text,
)
from jlens.autoencoder.state import (  # noqa: E402
    evaluation_shards_dir,
    read_json_if_valid,
)

STAGE = "evaluation"
SCRIPT = "evaluate_jspace_language"


def parse_args() -> argparse.Namespace:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--reconstructor", default=None)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--split", default="heldout", choices=("heldout", "val"))
    parser.add_argument("--limit-records", type=int, default=None)
    parser.add_argument(
        "--skip-robustness",
        action="store_true",
        help="Skip the paraphrase sweep (it re-runs generation once per prompt)",
    )
    parser.add_argument(
        "--evaluation-sharding",
        dest="evaluation_sharding",
        action="store_true",
        default=True,
        help="Persist one result shard per unit of work so evaluation resumes (default)",
    )
    parser.add_argument(
        "--no-evaluation-sharding", dest="evaluation_sharding", action="store_false"
    )
    return parser.parse_args()


def load_pair(run_dir, dataset, config, args, device):
    """Restore the reconstructor/adapter pair and refuse a mismatched one."""
    reconstructor_path = args.reconstructor or os.path.join(run_dir, "reconstructor.pt")
    adapter_path = args.adapter or os.path.join(run_dir, "adapter.pt")
    require_file(reconstructor_path, what="reconstructor checkpoint")
    require_file(adapter_path, what="adapter checkpoint")

    reconstructor_payload = load_checkpoint(reconstructor_path, expect_kind="reconstructor")
    reconstructor = PhraseReconstructor(d_model=dataset.d_model, config=config.reconstructor)
    reconstructor_metadata = restore_module(reconstructor, reconstructor_payload)
    reconstructor.eval().to(device)
    for parameter in reconstructor.parameters():
        parameter.requires_grad_(False)

    adapter_payload = load_checkpoint(adapter_path, expect_kind="adapter")
    adapter = ConeAdapter(
        d_model=dataset.d_model,
        config=config.adapter,
        target_rms=float(adapter_payload["metadata"]["extra"].get("target_rms") or 1.0),
    )
    adapter_metadata = restore_module(adapter, adapter_payload)
    adapter.eval().to(device)

    recorded_reconstructor = adapter_payload["metadata"]["extra"].get("reconstructor_sha256")
    if (
        recorded_reconstructor is not None
        and recorded_reconstructor != reconstructor_metadata["state_dict_sha256"]
    ):
        raise AutoencoderError(
            f"this adapter was trained against reconstructor "
            f"{recorded_reconstructor}, but {reconstructor_path} hashes to "
            f"{reconstructor_metadata['state_dict_sha256']}. Evaluating the pair "
            f"would measure a cycle that never existed."
        )
    return (
        reconstructor,
        reconstructor_metadata,
        reconstructor_path,
        adapter,
        adapter_metadata,
        adapter_path,
    )


def validate_only(run_dir: str) -> int:
    root = evaluation_shards_dir(run_dir)
    print(f"evaluation shards under {root}")
    total = 0
    for kind in ("baseline", "robustness", "confabulation"):
        directory = os.path.join(root, kind)
        names = (
            [n for n in sorted(os.listdir(directory)) if n.endswith(".json") and ".tmp." not in n]
            if os.path.isdir(directory)
            else []
        )
        good = sum(1 for n in names if read_json_if_valid(os.path.join(directory, n)))
        total += good
        print(f"  {kind:<15} {good} valid / {len(names)} files")
    print(f"  total valid shards: {total}")
    for name in ("evaluation.json", "gonogo.json"):
        path = os.path.join(run_dir, "artifacts", name)
        print(f"  {name:<18} {'present' if os.path.isfile(path) else 'missing'}")
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
    manifest = read_json_if_valid(os.path.join(dataset_dir, "manifest.json")) or {}

    stack = build_stack_from_args(config, args)
    dataset = JSpaceLanguageDataset.load(dataset_dir)
    dataset.assert_usable()
    device = torch.device("cuda" if torch.cuda.is_available() and not stack.is_mock else "cpu")
    (
        reconstructor,
        reconstructor_metadata,
        reconstructor_path,
        adapter,
        adapter_metadata,
        adapter_path,
    ) = load_pair(run_dir, dataset, config, args, device)

    identity = build_identity(
        config,
        args,
        stage=STAGE,
        run_dir=run_dir,
        dataset_manifest_sha256=manifest.get("records_sha256"),
        reconstructor_sha256=reconstructor_metadata["state_dict_sha256"],
        stage_config={
            **{
                "split": args.split,
                "limit_records": args.limit_records,
                "skip_robustness": bool(args.skip_robustness),
            },
            "adapter_sha256": adapter_metadata["state_dict_sha256"],
        },
    )
    verdict_code = {"code": 0}

    def body(*, guard, plan, state) -> int:
        phrases = list(
            dict.fromkeys(
                [r["phrase"] for r in dataset.records]
                + list(config.evaluation.confabulation_attractors)
            )
        )
        token_map = stack.token_id_map(phrases)
        embedder = PhraseEmbedder(
            stack.model, max_phrase_tokens=config.reconstructor.max_phrase_tokens
        )
        store = EvaluationShardStore(
            evaluation_shards_dir(run_dir),
            identity=evaluation_identity(
                config=config.evaluation,
                reward_config=config.preference,
                adapter=adapter,
                reconstructor=reconstructor,
                source_layer=stack.source_layer,
            ),
            enabled=bool(args.evaluation_sharding),
            # A restart recomputes everything but still *writes* shards, so an
            # interruption during the restart is not itself unrecoverable.
            # Existing shards are never deleted; a later resume picks up
            # whatever this run wrote.
            reuse=plan.action != "restart",
        )
        expected = expected_shard_names(
            dataset, config=config.evaluation, split=args.split, limit=args.limit_records
        )
        if store.enabled and store.reuse:
            outstanding = missing_shards(store, expected)
            total = sum(len(v) for v in expected.values())
            log.info(
                "evaluation shards: %d of %d present",
                total - sum(len(v) for v in outstanding.values()),
                total,
            )

        leakage = assert_no_split_leakage(
            dataset,
            salt=config.dataset.split_salt,
            val_fraction=config.dataset.val_fraction,
            heldout_fraction=config.dataset.heldout_fraction,
        )

        log.info("evaluating baselines on the %s split", args.split)
        evaluation = evaluate_baselines_sharded(
            stack.model,
            adapter,
            reconstructor,
            embedder,
            dataset,
            stack.prompt,
            config=config.evaluation,
            reward_config=config.preference,
            conditioner=stack.conditioner,
            stop_token_ids=stack.stop_token_ids,
            pad_token_id=stack.pad_token_id,
            source_layer=stack.source_layer,
            dictionary=stack.dictionary,
            split=args.split,
            device=device,
            limit=args.limit_records,
            log=log,
            store=store if store.enabled else None,
            guard=guard,
        )
        evaluation["cross_cone_swap"] = evaluate_cross_cone_swap(evaluation)
        state.write_progress(
            {
                "completed_units": store.reused + store.computed,
                "expected_units": sum(len(v) for v in expected.values()),
                "granularity": "result shard",
            }
        )

        robustness = None
        if not args.skip_robustness:
            log.info(
                "paraphrase robustness sweep over %s", config.evaluation.paraphrase_prompt_ids
            )
            robustness = evaluate_prompt_robustness_sharded(
                stack.model,
                adapter,
                reconstructor,
                embedder,
                dataset,
                config=config.evaluation,
                reward_config=config.preference,
                conditioner=stack.conditioner,
                stop_token_ids=stack.stop_token_ids,
                pad_token_id=stack.pad_token_id,
                source_layer=stack.source_layer,
                n_memory_tokens=config.adapter.n_memory_tokens,
                split=args.split,
                device=device,
                limit=args.limit_records,
                store=store if store.enabled else None,
                guard=guard,
            )

        confabulation = build_confabulation_probe_sharded(
            reconstructor,
            embedder,
            dataset,
            config=config.evaluation,
            phrase_token_ids=token_map,
            source_layer=stack.source_layer,
            split=args.split,
            device=device,
            store=store if store.enabled else None,
        )
        reconstructor_metrics = {}
        for split in ("train", args.split):
            reconstructor_metrics[split] = evaluate_reconstructor(
                reconstructor,
                dataset,
                embedder,
                split=split,
                config=config.reconstructor,
                source_layer=stack.source_layer,
                phrase_token_ids=token_map,
                device=device,
                extra_distractors=config.evaluation.confabulation_attractors,
            )
        report = gonogo_report(
            reconstructor_metrics=reconstructor_metrics[args.split],
            evaluation=evaluation,
            config=config.evaluation,
            leakage_report=leakage,
            confabulation=confabulation,
            prompt_robustness=robustness,
            diagnostics={
                "train_top5_retrieval": reconstructor_metrics["train"]["top5_retrieval"],
                "train_auroc": reconstructor_metrics["train"]["auroc_correct_vs_distractor"],
            },
        )
        report["cross_cone_swap"] = evaluation["cross_cone_swap"]
        report["resources"] = evaluation["resources"]
        report["artifact_identity"] = {
            "reconstructor_sha256": reconstructor_metadata["state_dict_sha256"],
            "adapter_sha256": adapter_metadata["state_dict_sha256"],
            "dataset_manifest": dataset.manifest.get("records_sha256"),
        }

        evaluation_path = write_json(
            os.path.join(run_dir, "artifacts", "evaluation.json"), evaluation
        )
        gonogo_path = write_json(os.path.join(run_dir, "artifacts", "gonogo.json"), report)
        write_json(
            os.path.join(run_dir, "artifacts", "reconstructor_metrics_eval.json"),
            reconstructor_metrics,
        )
        if robustness is not None:
            write_json(os.path.join(run_dir, "artifacts", "prompt_robustness.json"), robustness)
        summary_path = write_text(
            os.path.join(run_dir, "summary.md"), render_markdown(report)
        )
        write_json(
            os.path.join(run_dir, "evaluation_metadata.json"),
            stage_metadata(
                "evaluation",
                config,
                args,
                stack,
                payload={
                    "verdict": report["verdict"],
                    "failed_criteria": report["failed_criteria"],
                    "artifact_identity": report["artifact_identity"],
                    "sharding": store.stats(),
                },
            ),
        )
        if store.enabled:
            # "Complete" has to mean every expected combination ran. Checked
            # after aggregation so a partially-shard-backed report can still be
            # written and inspected — it just does not get a completion marker.
            kinds = ["baseline", "confabulation"] + ([] if args.skip_robustness else ["robustness"])
            outstanding = missing_shards(store, expected, kinds=kinds)
            if outstanding:
                raise AutoencoderError(
                    "evaluation is not complete: missing "
                    + ", ".join(f"{len(v)} {k} shard(s)" for k, v in outstanding.items())
                )
        state.mark_complete(
            identity=identity,
            artifacts={
                "evaluation": evaluation_path,
                "gonogo": gonogo_path,
                "summary": summary_path,
            },
            detail={
                "verdict": report["verdict"],
                "split": args.split,
                "sharding": store.stats(),
            },
        )
        print(render_markdown(report))
        log.info("verdict: %s", report["verdict"])
        verdict_code["code"] = 0 if report["passed"] else 4
        return 0

    code = run_stage(
        STAGE, args=args, run_dir=run_dir, identity=identity, script=SCRIPT, body=body, log=log
    )
    if code != 0:
        return code
    if verdict_code["code"] == 0 and not os.path.isfile(
        os.path.join(run_dir, "artifacts", "gonogo.json")
    ):
        return 0
    if verdict_code["code"] == 0:
        # The stage was skipped as complete: the verdict is the one on disk, not
        # a fresh pass, so it is read back rather than assumed to be GO.
        stored = read_json_if_valid(os.path.join(run_dir, "artifacts", "gonogo.json")) or {}
        if stored and not stored.get("passed", True):
            return 4
    return verdict_code["code"]


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AutoencoderError as exc:
        print(f"ABORT: {exc}", file=sys.stderr)
        sys.exit(2)
