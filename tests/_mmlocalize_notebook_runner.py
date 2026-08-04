# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Execute the layer-localization notebook's code cells in a clean interpreter.

Run as a subprocess by ``tests/test_mmlocalize_notebook.py`` from a working
directory outside the repository and with ``PYTHONPATH`` cleared, so ``jlens``
is *not* importable when the first cell runs — the notebook's section-1
bootstrap has to make the package importable itself.

Usage::

    python tests/_mmlocalize_notebook_runner.py <notebook.ipynb> [NAME=True ...]

Each ``NAME=True`` argument rewrites a committed ``NAME = False`` switch, the
same edit a user makes by hand. Without any, the committed defaults run, which
is how "opening the notebook starts nothing" is tested.

Prints one JSON object on stdout; exits non-zero on the first failing cell.

The leading underscore keeps pytest from collecting this file as a test module.
"""

import json
import os
import re
import sys
import traceback
from pathlib import Path


def main() -> int:
    notebook_path = Path(sys.argv[1])
    overrides = dict(argument.split("=", 1) for argument in sys.argv[2:])
    payload = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_cells = [cell for cell in payload["cells"] if cell["cell_type"] == "code"]

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
                        "traceback": traceback.format_exc()[-5000:],
                    }
                )
            )
            return 1

    import jlens

    verdict = namespace.get("VERDICT")
    store = namespace.get("STORE")
    validity = namespace.get("VALIDITY") or {}
    targets = namespace.get("TARGETS")
    interventions = namespace.get("INTERVENTION_RECORDS") or []
    print(
        json.dumps(
            {
                "ok": True,
                "n_code_cells": len(code_cells),
                "cwd": os.getcwd(),
                "jlens_file": jlens.__file__,
                "repo_path": str(namespace["REPO_PATH"]),
                "commit": namespace["COMMIT"],
                "overrides": overrides,
                "run_real": namespace["RUN_REAL_LOCALIZATION"],
                "run_model_stages": namespace["RUN_MODEL_STAGES"],
                "run_text_recalibration": namespace["RUN_TEXT_RECALIBRATION"],
                "budget_confirmed": namespace["CONFIRM_MODEL_PASS_BUDGET"],
                "model_stages_enabled": namespace["MODEL_STAGES_ENABLED"],
                "model_is_none": namespace["MODEL"] is None,
                "invariance_passed": (namespace.get("INVARIANCE") or {}).get("passed"),
                "preflight_ran": namespace.get("PREFLIGHT") is not None,
                "run_dir": str(namespace["RUN_DIR"]),
                # ---- the layer set and the frozen targets, before any result
                "layers": list(namespace["LAYERS"]),
                "concepts": list(namespace["CONCEPTS"]),
                "reference_layer": namespace["REFERENCE_LAYER"],
                "target_policy": namespace["TARGET_POLICY"],
                "target_checksum": targets.checksum if targets else None,
                "target_manifest_checksum": namespace["TARGET_MANIFEST"][
                    "manifest_checksum"
                ],
                "n_target_images": len(targets.all_target_images()) if targets else 0,
                "n_source_images": len(targets.all_source_images()) if targets else 0,
                "exclusion_audit": namespace["EXCLUSION_AUDIT"],
                "selected_concepts": list(namespace["SELECTED_NAMES"]),
                "focal_concepts": list(namespace["FOCAL_CONCEPTS"]),
                "unrelated_controls": namespace["UNRELATED_CONTROLS"],
                "n_groups": namespace["N_TOTAL_GROUPS"],
                "n_distinct_images": namespace["N_DISTINCT_IMAGES"],
                "leakage_ok": namespace["LEAKAGE"]["ok"],
                "budget": namespace["BUDGET"].to_dict(),
                "actual_budget": (
                    namespace["ACTUAL_BUDGET"].to_dict()
                    if namespace.get("ACTUAL_BUDGET")
                    else None
                ),
                # ---- Stage B
                "eligible_layers": list(namespace.get("ELIGIBLE_LAYERS") or []),
                "validity_status": {
                    str(layer): result["status"] for layer, result in validity.items()
                },
                "validity_gate_digest": (
                    next(iter(validity.values()))["gate_digest"] if validity else None
                ),
                "legacy_gate_passed": {
                    str(layer): result["legacy_gate"]["passed"]
                    for layer, result in validity.items()
                },
                "rank_conventions": (
                    next(iter(validity.values()))["rank_conventions_reported"]
                    if validity
                    else None
                ),
                "n_validation_rows": len(namespace.get("VALIDITY_ROWS") or []),
                # ---- results
                "verdict": (verdict or {}).get("verdict"),
                "criteria_status": (verdict or {}).get("criteria_status"),
                "earliest_layer_with_evidence": (verdict or {}).get(
                    "earliest_tested_layer_with_evidence"
                ),
                "reference_reproduces": (verdict or {}).get(
                    "reference_layer_reproduces"
                ),
                "paired_comparison_rows": len(
                    (verdict or {}).get("paired_layer_comparison") or []
                ),
                "intervention_layers": sorted(
                    {int(row["layer"]) for row in interventions}
                ),
                "intervention_fields": sorted(interventions[0]) if interventions else [],
                "resume_status": (store.status_report() if store else {}).get("status"),
                "completed_units": (store.status_report() if store else {}).get(
                    "completed_units"
                ),
                "fingerprint_digest": (
                    namespace["FINGERPRINT"].digest if store else None
                ),
                "selection_fingerprint": (
                    namespace.get("SELECTION_FINGERPRINT") if store else None
                ),
            },
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
