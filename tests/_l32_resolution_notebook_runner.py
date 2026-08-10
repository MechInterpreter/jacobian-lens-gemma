# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Execute the L32 convergence-resolution notebook's code cells in a clean
interpreter.

Run as a subprocess by ``tests/test_l32_resolution_notebook.py`` from a working
directory outside the repository and with ``PYTHONPATH`` cleared, so ``jlens``
is *not* importable when the first cell runs — the notebook's section-1
bootstrap has to make the package importable itself.

Usage::

    python tests/_l32_resolution_notebook_runner.py <notebook.ipynb> [NAME=True ...]

Each ``NAME=True`` argument rewrites a committed ``NAME = False`` switch before
execution, the same edit a user makes by hand.

``--stop-after N`` executes only the first ``N`` code cells, which is how the
interruption/resume test produces a genuinely half-finished run directory
instead of a simulated one.

Prints one JSON object on stdout; exits non-zero on the first failing cell.

The leading underscore keeps pytest from collecting this file as a test module.
"""

import json
import re
import sys
import traceback
from pathlib import Path

#: Names lifted out of the notebook namespace when the run succeeds. Keep this
#: to plain JSON-safe values: the point is to assert on the notebook's own
#: conclusions, not to re-export its objects.
REPORTED = (
    "RESOLUTION_LAYER",
    "SELECTED_NAMES",
    "FOCAL_CONCEPTS",
    "OPEN_QUESTION",
    "POPULATION_DIGEST",
    "STAGE_A_ENABLED",
    "STAGE_B_REQUESTED",
    "RUN_STATE",
    "MODEL_STAGE_ENABLED",
)


def main() -> int:
    arguments = sys.argv[1:]
    stop_after = None
    if "--stop-after" in arguments:
        index = arguments.index("--stop-after")
        stop_after = int(arguments[index + 1])
        arguments = arguments[:index] + arguments[index + 2 :]

    notebook_path = Path(arguments[0])
    overrides = dict(argument.split("=", 1) for argument in arguments[1:])
    payload = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_cells = [cell for cell in payload["cells"] if cell["cell_type"] == "code"]
    if stop_after is not None:
        code_cells = code_cells[:stop_after]

    if "jlens" in sys.modules:
        print(json.dumps({"ok": False, "error": "jlens was already imported"}))
        return 2

    namespace: dict = {"__name__": "__notebook__"}
    for index, cell in enumerate(code_cells):
        source = "".join(cell["source"])
        for name, value in overrides.items():
            source = re.sub(
                rf"^{re.escape(name)} = False$",
                f"{name} = {value}",
                source,
                flags=re.MULTILINE,
            )
        try:
            exec(compile(source, f"<cell {index}>", "exec"), namespace)  # noqa: S102
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            print(
                json.dumps(
                    {
                        "ok": False,
                        "cell": index,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc()[-4000:],
                    }
                )
            )
            return 1

    result = {"ok": True, "n_cells": len(code_cells)}
    for name in REPORTED:
        if name in namespace:
            result[name] = namespace[name]

    for name, keys in (
        ("VERDICT", ("verdict", "classification", "failed_validity_clauses")),
        ("STAGE_B", ("runs", "gate_met", "gate_overridden", "stage_a_verdict")),
        (
            "SYNTHESIS",
            (
                "combined_pre_convergence_claim",
                "populations_are_the_same",
                "additional_causal_replication_required",
            ),
        ),
        ("DISJOINTNESS", ("disjoint", "failed_families", "n_overlaps")),
        (
            "PSEUDOREPLICATION",
            ("passed", "n_units", "n_distinct_images", "n_distinct_recordings"),
        ),
        ("CACHE_LOAD", ("build_expanded_manifest_called", "compatible", "n_groups",
                        "category_ids_available", "n_categories_recovered")),
        ("SELECTION_DETERMINISM", ("deterministic",)),
        ("CONTROLS_RECORD", ("passed", "missing_or_empty", "failing")),
        ("PHRASING", ("passed",)),
        ("IMMUTABILITY", ("unchanged",)),
        ("SAMPLE_PLAN_RECORD", ("images_per_concept", "plan_digest",
                                "adequate_at_expected_admissible_count")),
        ("POOL_RECORD", ("n_groups_excluded", "n_groups_in_independent_pool")),
        ("HEAD_AGREEMENT", ("passed", "comparison_ran")),
        ("RESUME", ("run_state", "units_computed", "units_reused")),
        (
            "PREP",
            ("files_computed_this_session", "files_reused_from_drive",
             "reused_complete_cache"),
        ),
        (
            "COMPLETENESS",
            ("complete", "fallback_required", "n_files_read", "n_shards",
             "unit_families_skipped"),
        ),
        (
            "FEASIBILITY",
            ("all_feasible", "infeasible_concepts", "descriptive_top_n",
             "descriptive_ranking_matches_frozen_order",
             "concepts_reselected_from_this_pool"),
        ),
        ("IMMUTABILITY", ("unchanged", "appeared", "vanished", "modified")),
    ):
        value = namespace.get(name)
        if isinstance(value, dict):
            for key in keys:
                if key in value:
                    result[f"{name}.{key}"] = value[key]

    summary = namespace.get("SUMMARY")
    if isinstance(summary, dict):
        result["summary_schema"] = summary.get("schema")
        result["primary_verdict"] = summary.get("primary_verdict")
        result["mock_proves_pipeline_only"] = summary.get("mock_proves_pipeline_only")
        result["run_dir"] = str((summary.get("resume") or {}).get("run_dir"))
        result["concepts_replaced_after_results"] = summary.get(
            "concepts_replaced_after_results"
        )

    exclusion = namespace.get("EXCLUSION")
    if exclusion is not None:
        result["exclusion_digest"] = exclusion.digest
        result["exclusion_counts"] = exclusion.counts()

    paths = namespace.get("ARTIFACT_PATHS")
    if isinstance(paths, dict):
        result["artifact_names"] = sorted(paths)

    if namespace.get("PREP_DIR") is not None:
        result["PREP_DIR"] = str(namespace["PREP_DIR"])
    fingerprint = namespace.get("PREPARATION_FINGERPRINT")
    if isinstance(fingerprint, dict):
        result["preparation_digest"] = fingerprint.get("preparation_digest")
    resolution = namespace.get("RESOLUTION_FINGERPRINT")
    if isinstance(resolution, dict):
        result["resolution_fingerprint_digest"] = resolution.get("fingerprint_digest")
    if namespace.get("POOL_DIGEST") is not None:
        result["POOL_DIGEST"] = namespace["POOL_DIGEST"]
    if namespace.get("RANKING_DIGEST") is not None:
        result["RANKING_DIGEST"] = namespace["RANKING_DIGEST"]
    if namespace.get("FEASIBILITY_DIGEST") is not None:
        result["FEASIBILITY_DIGEST"] = namespace["FEASIBILITY_DIGEST"]

    print(json.dumps(result, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
