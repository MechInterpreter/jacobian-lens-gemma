# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Execute the robustness notebook's code cells in a clean interpreter.

Run as a subprocess by ``tests/test_mmpilot_robustness_notebook.py`` from a
working directory outside the repository and with ``PYTHONPATH`` cleared, so
``jlens`` is *not* importable when the first cell runs — the notebook's
section-1 bootstrap has to make the package importable itself.

Usage::

    python tests/_mmpilot_robustness_notebook_runner.py <notebook.ipynb> [NAME=True ...]

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
                        "traceback": traceback.format_exc()[-4000:],
                    }
                )
            )
            return 1

    import jlens

    verdict = namespace.get("VERDICT")
    store = namespace.get("STORE")
    image_level = namespace.get("IMAGE_LEVEL") or {}
    subset = namespace.get("SUBSET") or {}
    rows = [row for split in subset.get("splits", {}).values() for row in split]
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
                "run_real": namespace["RUN_REAL_ROBUSTNESS"],
                "run_model_stages": namespace["RUN_MODEL_STAGES"],
                "budget_confirmed": namespace["CONFIRM_MODEL_PASS_BUDGET"],
                "model_stages_enabled": namespace["MODEL_STAGES_ENABLED"],
                "spoken_audio_enabled": namespace["ENABLE_SPOKEN_AUDIO"],
                "model_is_none": namespace["MODEL"] is None,
                "invariance_passed": (namespace.get("INVARIANCE") or {}).get("passed"),
                "preflight_ran": namespace.get("PREFLIGHT") is not None,
                "run_dir": str(namespace["RUN_DIR"]),
                # Selection, all decided before any model ran.
                "ranked_concepts": namespace["RANKED_CONCEPTS"],
                "selected_concepts": namespace["SELECTED_NAMES"],
                "focal_concepts": namespace["FOCAL_CONCEPTS"],
                "non_focal_concepts": namespace["NON_FOCAL_CONCEPTS"],
                "unrelated_controls": namespace["UNRELATED_CONTROLS"],
                "n_groups": namespace["N_TOTAL_GROUPS"],
                "n_distinct_images": namespace["N_DISTINCT_IMAGES"],
                "n_siblings_excluded": namespace["N_SIBLINGS_EXCLUDED"],
                "leakage_ok": namespace["LEAKAGE"]["ok"],
                "budget": namespace["BUDGET"].to_dict(),
                # Results, when the model stages ran.
                "verdict": (verdict or {}).get("verdict"),
                "criteria_status": (verdict or {}).get("criteria_status"),
                "bidirectional": (verdict or {}).get(
                    "concepts_transferring_both_directions"
                ),
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
                "n_pseudoreplicated_rows": (
                    namespace["DIVERGENCE"]["n_rows_pseudoreplicated_at_group_level"]
                    if image_level
                    else None
                ),
                "subset_rows_with_sibling_provenance": sum(
                    1
                    for row in rows
                    if "excluded_sibling_group_ids" in (row.get("split_provenance") or {})
                ),
                "intervention_fields": sorted(interventions[0]) if interventions else [],
            },
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
