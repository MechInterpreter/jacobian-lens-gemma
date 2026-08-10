# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Run one preparation in a **separate** Python process.

``tests/test_l32_prep_cache.py`` uses this to prove resumability the only way
that actually proves it: the interrupted session and the resuming session are
different interpreters, so nothing survives in memory between them.

Usage::

    python tests/_prep_cache_harness.py --cache DIR --runs A[os.pathsep]B \\
        [--batch N] [--abort-after K] [--on-corrupt quarantine|refuse] \\
        [--no-fallback] [--finalize] [--salt TEXT]

``--abort-after K`` raises after the *K*-th file-level progress line, which
lands inside a batch rather than between two of them — the interruption that
matters. Prints one JSON object on stdout.

The leading underscore keeps pytest from collecting this file as a test module.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jlens.mmpilot import prep_cache as prep  # noqa: E402


class _Aborted(RuntimeError):
    """The simulated runtime disconnect."""


def build_fingerprint(run_dirs, *, salt: str = "") -> dict:
    """A deterministic preparation fingerprint over these completed runs.

    Every field the module owns comes from :func:`default_fingerprint_constants`;
    the rest are fixed literals so a test can change exactly one thing and see
    exactly one digest move. ``salt`` rides on the selection seed, which is a
    real preparation input.
    """
    identities = [prep.completed_run_identity(run_dir) for run_dir in run_dirs]
    return prep.preparation_fingerprint(
        **prep.default_fingerprint_constants(),
        completed_run_basenames=[entry["run"] for entry in identities],
        completed_run_fingerprints=[
            entry["fingerprint_digest"] for entry in identities
        ],
        completed_summary_checksums=[
            entry["summary_checksums"] for entry in identities
        ],
        cached_expanded_manifest_checksum="sha256:manifest",
        cache_schema_version="expanded_manifest.v1",
        evidence_lexicon_hash="sha256:lexicon",
        frozen_selected_concepts=["zebra", "cat", "toilet", "giraffe", "bird",
                                  "microwave"],
        frozen_focal_concepts=["zebra", "cat", "toilet"],
        sample_size_rule_version="mmpilot.l32_resolution_sample_size.v1",
        sample_size_plan_digest="sha256:plan",
        selection_algorithm_version=(
            "mmpilot.l32_resolution_independent_population.v1"
        ),
        selection_seed=f"spokencoco-l32-resolution-v1{salt}",
        selection_profile_version="image_unique.v1",
        n_train_positive_images=6,
        n_test_positive_images=6,
        n_train_negative_images=6,
        n_test_negative_images=6,
    )


def _reporter(abort_after: int | None) -> prep.ProgressReporter:
    """Every line is recorded by the reporter itself; the printer only counts.

    Keeping a second list here would double-count each line, and the tests
    reason about how many files were read from exactly that count.
    """
    seen = {"n": 0}

    def printer(line: str) -> None:
        if "work=computed" in line:
            seen["n"] += 1
            if abort_after is not None and seen["n"] >= abort_after:
                raise _Aborted(f"simulated disconnect after {seen['n']} file(s)")

    return prep.ProgressReporter(interval=0.0, printer=printer)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--batch", type=int, default=prep.DEFAULT_BATCH_FILES)
    parser.add_argument("--abort-after", type=int, default=None)
    parser.add_argument("--on-corrupt", default="quarantine")
    parser.add_argument("--no-fallback", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--salt", default="")
    arguments = parser.parse_args()

    run_dirs = [Path(part) for part in arguments.runs.split(os.pathsep) if part]
    fingerprint = build_fingerprint(run_dirs, salt=arguments.salt)
    reporter = _reporter(arguments.abort_after)

    try:
        record = prep.run_exclusion_preparation(
            arguments.cache,
            run_dirs,
            fingerprint=fingerprint,
            batch_files=arguments.batch,
            progress=reporter,
            on_corrupt=arguments.on_corrupt,
            allow_fallback=not arguments.no_fallback,
        )
    except _Aborted as error:
        print(
            json.dumps(
                {
                    "ok": False,
                    "aborted": True,
                    "error": str(error),
                    "preparation_digest": fingerprint["preparation_digest"],
                    "progress_lines": reporter.lines,
                }
            )
        )
        return 3
    except Exception as error:  # noqa: BLE001 - reported, not swallowed
        print(
            json.dumps(
                {
                    "ok": False,
                    "aborted": False,
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc()[-3000:],
                }
            )
        )
        return 1

    if arguments.finalize:
        prep.finalize_preparation(
            arguments.cache,
            {
                "preparation_digest": fingerprint["preparation_digest"],
                "exclusion_digest": record["exclusion"].digest,
                "completeness_complete": record["completeness"]["complete"],
            },
        )

    print(
        json.dumps(
            {
                "ok": True,
                "aborted": False,
                "preparation_digest": fingerprint["preparation_digest"],
                "cache_dir": record["cache_dir"],
                "exclusion_digest": record["exclusion"].digest,
                "exclusion_counts": record["exclusion"].counts(),
                "files_computed": record["files_computed_this_session"],
                "files_reused": record["files_reused_from_drive"],
                "reused_complete_cache": record["reused_complete_cache"],
                "complete": record["completeness"]["complete"],
                "fallback_required": record["completeness"]["fallback_required"],
                "n_shards": record["completeness"]["n_shards"],
                "n_files_read": record["completeness"]["n_files_read"],
                "files_by_family": record["completeness"]["files_by_family"],
                "families_by_run": record["families_by_run"],
                "quarantined": (record["minimal_state"] or {}).get("quarantined", []),
                "cursor": (record["minimal_state"] or {}).get("cursor"),
                "runs": record["completeness"]["runs"],
                "progress_lines": reporter.lines,
            },
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
