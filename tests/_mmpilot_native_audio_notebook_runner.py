# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Execute the native-audio transfer notebook's code cells in a clean interpreter.

Run as a subprocess by ``tests/test_mmpilot_native_audio_notebook.py`` from a
working directory outside the repository and with ``PYTHONPATH`` cleared, so
``jlens`` is *not* importable when the first cell runs — the notebook's
section-1 bootstrap has to make the package importable itself.

Usage::

    python tests/_mmpilot_native_audio_notebook_runner.py <notebook.ipynb> [NAME=True ...]

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

    store = namespace.get("STORE")
    status = store.status_report() if store else {}
    representational = namespace.get("REPRESENTATIONAL") or {}
    capability_verdict = namespace.get("CAPABILITY_VERDICT") or {}
    representational_verdict = namespace.get("REPRESENTATIONAL_VERDICT") or {}
    primary_causal = namespace.get("PRIMARY_CAUSAL_VERDICT") or {}
    replication = namespace.get("REPLICATION_VERDICT") or {}
    overall = namespace.get("OVERALL_VERDICT") or {}
    lens_report = namespace.get("LENS_REPORT") or {}
    architecture = namespace.get("ARCHITECTURE") or {}
    invariance = namespace.get("INVARIANCE") or {}
    interventions = namespace.get("INTERVENTION_RECORDS") or []
    subset = namespace.get("SUBSET") or {}
    rows = [row for split in subset.get("splits", {}).values() for row in split]

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
                # ---- switches and gates
                "run_real": namespace["RUN_REAL_AUDIO_TRANSFER"],
                "run_model_stages": namespace["RUN_MODEL_STAGES"],
                "confirm_model_load": namespace["CONFIRM_MODEL_LOAD"],
                "confirm_representation_budget": namespace[
                    "CONFIRM_REPRESENTATION_BUDGET"
                ],
                "run_l35_causal_stage": namespace["RUN_L35_CAUSAL_STAGE"],
                "confirm_l35_causal_budget": namespace["CONFIRM_L35_CAUSAL_BUDGET"],
                "run_l38_l40_replication": namespace["RUN_L38_L40_REPLICATION"],
                "confirm_replication_budget": namespace["CONFIRM_REPLICATION_BUDGET"],
                "model_stages_enabled": namespace["MODEL_STAGES_ENABLED"],
                "stage_a_enabled": namespace["STAGE_A_ENABLED"],
                "stage_b_requested": namespace["STAGE_B_REQUESTED"],
                "stage_c_requested": namespace["STAGE_C_REQUESTED"],
                "model_is_none": namespace["MODEL"] is None,
                "run_dir": str(namespace["RUN_DIR"]),
                # ---- selection, all decided before any model ran
                "ranked_concepts": namespace["RANKED_CONCEPTS"],
                "selected_concepts": namespace["SELECTED_NAMES"],
                "focal_concepts": namespace["FOCAL_CONCEPTS"],
                "non_focal_concepts": namespace["NON_FOCAL_CONCEPTS"],
                "unrelated_controls": namespace["UNRELATED_CONTROLS"],
                "modalities": list(namespace["MODALITIES"]),
                "layers": list(namespace["LAYERS"]),
                "primary_causal_layer": namespace["PRIMARY_CAUSAL_LAYER"],
                "replication_layers": list(namespace["REPLICATION_LAYERS"]),
                "n_groups": namespace["N_TOTAL_GROUPS"],
                "n_distinct_images": namespace["N_DISTINCT_IMAGES"],
                "n_distinct_recordings": namespace["N_DISTINCT_RECORDINGS"],
                "n_siblings_excluded": namespace["N_SIBLINGS_EXCLUDED"],
                "train_heldout_image_overlap": namespace["IMAGE_OVERLAP"],
                "leakage_ok": namespace["LEAKAGE"]["ok"],
                "budgets": [b.to_dict() for b in namespace["BUDGETS"]],
                # ---- results
                "available_modalities": namespace["AVAILABLE_MODALITIES"],
                "blocked_modalities": namespace["BLOCKED_MODALITIES"],
                "lens_layers": lens_report.get("layers"),
                "lens_checksums": lens_report.get("checksums"),
                "lens_combined_checksum": lens_report.get("combined_checksum"),
                "architecture_passed": architecture.get("passed"),
                "architecture_modalities": sorted(
                    architecture.get("per_modality", {})
                ),
                "invariance_passed": invariance.get("passed"),
                "invariance_modalities": invariance.get("modalities"),
                "representational_layers": sorted(int(k) for k in representational),
                "representational_pairs": {
                    str(layer): sorted(entry.get("pairs", {}))
                    for layer, entry in representational.items()
                },
                "capability_verdict": capability_verdict.get("verdict"),
                "concepts_passing_all_three": capability_verdict.get(
                    "concepts_passing_all_three"
                ),
                "representational_verdict": representational_verdict.get("verdict"),
                "representational_rows": len(representational_verdict.get("rows", [])),
                "audio_directions_beating_shuffled": representational_verdict.get(
                    "audio_directions_beating_shuffled"
                ),
                "primary_causal_verdict": primary_causal.get("verdict"),
                "primary_causal_layer_in_verdict": primary_causal.get("layer"),
                "primary_causal_cells": len(primary_causal.get("cells", [])),
                "causal_layers_run": namespace["CAUSAL_LAYERS_RUN"],
                "replication_verdict": replication.get("verdict"),
                "replication_per_layer": replication.get("per_layer"),
                "overall_verdict": overall.get("verdict"),
                "criteria_status": overall.get("criteria_status"),
                "intervention_fields": sorted(interventions[0]) if interventions else [],
                "intervention_pairs": sorted(
                    {
                        f"{r['source_modality']}->{r['target_modality']}"
                        for r in interventions
                    }
                ),
                "intervention_control_kinds": sorted(
                    {r["control_kind"] for r in interventions}
                ),
                "intervention_alphas": sorted(
                    {float(r["alpha"]) for r in interventions}
                ),
                "resume_status": status.get("status"),
                "completed_units": status.get("completed_units"),
                "fingerprint_digest": (
                    namespace["FINGERPRINT"].digest if store else None
                ),
                "selection_fingerprint": namespace.get("SELECTION_FINGERPRINT"),
                "subset_rows_with_sibling_provenance": sum(
                    1
                    for row in rows
                    if "excluded_sibling_group_ids" in (row.get("split_provenance") or {})
                ),
                "report_written": (
                    Path(namespace["RUN_DIR"]) / "native_audio_transfer_report.md"
                ).is_file(),
            },
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
