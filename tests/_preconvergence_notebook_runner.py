# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Execute the L27-L31 pre-convergence notebook's code cells in a clean
interpreter.

Run as a subprocess by ``tests/test_preconvergence_notebook.py`` from a working
directory outside the repository and with ``PYTHONPATH`` cleared, so ``jlens``
is *not* importable when the first cell runs — the notebook's section-1
bootstrap has to make the package importable itself.

Usage::

    python tests/_preconvergence_notebook_runner.py <notebook.ipynb> [NAME=True ...]

Each ``NAME=True`` argument rewrites a committed ``NAME = False`` switch before
execution, the same edit a user makes by hand.

``--stop-after N`` executes only the first ``N`` code cells, which is how the
interruption/resume tests produce genuinely half-finished state instead of a
simulated one.

Prints one JSON object on stdout; exits non-zero on the first failing cell.

The leading underscore keeps pytest from collecting this file as a test module.
"""

import json
import re
import sys
import traceback
from pathlib import Path

#: Names lifted out of the notebook namespace when the run succeeds. Plain
#: JSON-safe values only: the point is to assert on the notebook's own
#: conclusions, not to re-export its objects.
REPORTED = (
    "SELECTED_LAYER",
    "MULTIMODAL_LAYER",
    "ADJACENT_SCALE",
    "SELECTED_NAMES",
    "FOCAL_CONCEPTS",
    "OPEN_QUESTION",
    "POPULATION_DIGEST",
    "POOL_DIGEST",
    "RANKING_DIGEST",
    "MODE",
    "RUN_STATE",
    "LENS_RESUME_STATE",
    "FITTING_ENABLED",
    "CONFIRMATION_ENABLED",
    "MODEL_STAGE_ENABLED",
    "STAGE_3_ENABLED",
    "STAGE_4_REQUESTED",
    "MOCK_SCENARIO",
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
        ("PREDOWNLOAD", ("passed", "gate_is_extension_gate",
                         "criterion_digest_matches", "shared_contract_failures",
                         "study_contract_failures")),
        ("SOURCE_LAYERS", ("disjoint", "overlap",
                           "parent_accumulator_may_be_seeded")),
        ("UNTOUCHED_AUDIT", ("untouched", "n_exact_hits", "n_near_hits",
                             "n_internal_duplicates", "required_disjoint_from")),
        ("CONFIRMATION_MANIFEST", ("size", "checksum", "protocol")),
        ("DEVELOPMENT_ROLE", ("reused", "role_then", "role_now", "checksum")),
        ("POPULATION_PINS", ("pinned_resolution_run", "pin_was_discovered",
                             "pin_was_defaulted", "n_excluded_runs")),
        ("SELECTION", ("passing_layers", "selected_layer", "candidates",
                       "evaluated_layers", "verdict")),
        ("LENS_VERDICT", ("verdict", "selected_layer",
                          "failed_validity_clauses")),
        ("FIT_RECORD", ("n_done", "scale", "layers",
                        "parent_accumulator_seeded", "checkpoint_every")),
        ("CONVERGENCE_VERDICT", ("verdict", "classification",
                                 "failed_validity_clauses",
                                 "criterion_thresholds_unchanged")),
        ("CONTROLS_RECORD", ("passed", "missing_or_empty", "failing")),
        ("STAGE_4", ("runs", "gate_met", "gate_overridden", "evidence_status",
                     "failed_gate_clauses")),
        ("CAUSAL_CONTROLS", ("passed", "missing_or_empty", "failing")),
        ("SAME_POPULATION", ("same_population", "same_layer", "combinable")),
        ("VERDICTS", ("terminal_outcome",)),
        ("LEAKAGE_AUDIT", ("passed", "candidate_order_invariant",
                           "candidate_set_moves_fingerprint", "audit_digest")),
        ("DISJOINTNESS", ("disjoint", "failed_families", "n_overlaps")),
        ("PSEUDOREPLICATION", ("passed", "n_units", "n_distinct_images",
                               "n_distinct_recordings")),
        ("SELECTION_DETERMINISM", ("deterministic",)),
        ("PREP", ("files_computed_this_session", "files_reused_from_drive")),
        ("COMPLETENESS", ("complete", "fallback_required")),
        ("CACHE_LOAD", ("build_expanded_manifest_called", "compatible",
                        "n_groups")),
        ("RESUME", ("run_state", "units_computed", "units_reused",
                    "fit_n_done", "lens_run_state")),
        ("IMMUTABILITY", ("unchanged", "appeared", "vanished", "modified")),
        ("PARENT_IMMUTABILITY", ("immutable", "n_files_checked")),
        ("FEASIBILITY", ("all_feasible", "infeasible_concepts")),
        ("INVARIANCE", ("passed",)),
        ("HEAD_AGREEMENT", ("passed", "comparison_ran")),
        ("CAPABILITY_VERDICT", ("verdict",)),
        ("FITTING_BUDGET", ("scale", "layers", "n_forward_passes",
                            "n_backward_passes")),
    ):
        value = namespace.get(name)
        if isinstance(value, dict):
            for key in keys:
                if key in value:
                    result[f"{name}.{key}"] = value[key]

    verdicts = namespace.get("VERDICTS")
    if isinstance(verdicts, dict):
        for vname, entry in (verdicts.get("verdicts") or {}).items():
            result[f"verdict.{vname}"] = entry.get("verdict")

    summary = namespace.get("SUMMARY")
    if isinstance(summary, dict):
        result["summary_schema"] = summary.get("schema")
        result["primary_verdict"] = summary.get("primary_verdict")
        result["mock_proves_pipeline_only"] = summary.get("mock_proves_pipeline_only")
        result["concepts_replaced_after_results"] = summary.get(
            "concepts_replaced_after_results"
        )
        result["run_dir"] = str((summary.get("resume") or {}).get("run_dir"))

    exclusion = namespace.get("EXCLUSION")
    if exclusion is not None:
        result["exclusion_digest"] = exclusion.digest
        result["exclusion_counts"] = exclusion.counts()

    paths = namespace.get("ARTIFACT_PATHS")
    if isinstance(paths, dict):
        result["artifact_names"] = sorted(paths)

    table = (namespace.get("SELECTION") or {}).get("table")
    if isinstance(table, list):
        result["confirmation_table"] = [
            {"layer": row["layer"], "passed": row["passed"],
             "failed_clauses": row["failed_clauses"]}
            for row in table
        ]

    if namespace.get("PREP_DIR") is not None:
        result["PREP_DIR"] = str(namespace["PREP_DIR"])
    if namespace.get("RUN_DIR") is not None:
        result["RUN_DIR"] = str(namespace["RUN_DIR"])
    fingerprint = namespace.get("PREPARATION_FINGERPRINT")
    if isinstance(fingerprint, dict):
        result["preparation_digest"] = fingerprint.get("preparation_digest")
    study = namespace.get("STUDY_FINGERPRINT")
    if isinstance(study, dict):
        result["study_fingerprint_digest"] = study.get("fingerprint_digest")
    lens_fingerprint = namespace.get("LENS_FINGERPRINT")
    if lens_fingerprint is not None:
        result["lens_fingerprint_digest"] = lens_fingerprint.digest

    print(json.dumps(result, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
