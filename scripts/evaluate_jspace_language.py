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
    build_confabulation_probe,
    evaluate_baselines,
    evaluate_cross_cone_swap,
    evaluate_prompt_robustness,
    gonogo_report,
    render_markdown,
)
from jlens.autoencoder.reconstructor import (  # noqa: E402
    PhraseEmbedder,
    PhraseReconstructor,
    evaluate_reconstructor,
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
    write_text,
)


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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log = configure_logging(args.log_level)
    config = resolve_config(args)
    run_dir = prepare_run_dir(args)
    dataset_dir = args.dataset_dir or os.path.join(run_dir, "dataset")
    reconstructor_path = args.reconstructor or os.path.join(run_dir, "reconstructor.pt")
    adapter_path = args.adapter or os.path.join(run_dir, "adapter.pt")
    require_file(os.path.join(dataset_dir, "records.jsonl"), what="dataset records")
    require_file(reconstructor_path, what="reconstructor checkpoint")
    require_file(adapter_path, what="adapter checkpoint")

    stack = build_stack_from_args(config, args)
    dataset = JSpaceLanguageDataset.load(dataset_dir)
    dataset.assert_usable()
    device = torch.device("cuda" if torch.cuda.is_available() and not stack.is_mock else "cpu")

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

    phrases = list(
        dict.fromkeys(
            [r["phrase"] for r in dataset.records] + list(config.evaluation.confabulation_attractors)
        )
    )
    token_map = stack.token_id_map(phrases)
    embedder = PhraseEmbedder(
        stack.model, max_phrase_tokens=config.reconstructor.max_phrase_tokens
    )

    leakage = assert_no_split_leakage(
        dataset,
        salt=config.dataset.split_salt,
        val_fraction=config.dataset.val_fraction,
        heldout_fraction=config.dataset.heldout_fraction,
    )

    log.info("evaluating baselines on the %s split", args.split)
    evaluation = evaluate_baselines(
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
    )
    evaluation["cross_cone_swap"] = evaluate_cross_cone_swap(evaluation)

    robustness = None
    if not args.skip_robustness:
        log.info("paraphrase robustness sweep over %s", config.evaluation.paraphrase_prompt_ids)
        robustness = evaluate_prompt_robustness(
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
        )

    confabulation = build_confabulation_probe(
        reconstructor,
        embedder,
        dataset,
        config=config.evaluation,
        phrase_token_ids=token_map,
        source_layer=stack.source_layer,
        split=args.split,
        device=device,
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

    write_json(os.path.join(run_dir, "artifacts", "evaluation.json"), evaluation)
    write_json(os.path.join(run_dir, "artifacts", "gonogo.json"), report)
    write_json(
        os.path.join(run_dir, "artifacts", "reconstructor_metrics_eval.json"),
        reconstructor_metrics,
    )
    if robustness is not None:
        write_json(os.path.join(run_dir, "artifacts", "prompt_robustness.json"), robustness)
    write_text(os.path.join(run_dir, "summary.md"), render_markdown(report))
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
            },
        ),
    )
    print(render_markdown(report))
    log.info("verdict: %s", report["verdict"])
    return 0 if report["passed"] else 4


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AutoencoderError as exc:
        print(f"ABORT: {exc}", file=sys.stderr)
        sys.exit(2)
