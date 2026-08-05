# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Execute the calibration notebook's code cells in a clean interpreter.

Run as a subprocess by ``tests/test_calibration_notebook.py`` from a working
directory outside the repository and with ``PYTHONPATH`` cleared, so ``jlens``
is *not* importable when the first cell runs — the notebook's section-0
bootstrap has to make the package importable itself.

Usage::

    python tests/_calibration_notebook_runner.py <notebook.ipynb> [NAME=True ...]

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

    store = namespace.get("STORE")
    calibration = namespace.get("CALIBRATION")
    validation = namespace.get("VALIDATION") or {}
    confirmation = namespace.get("CONFIRMATION")
    publication = namespace.get("PUBLICATION")
    scales = list(namespace.get("SCALES") or ())

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
                # ---- switches, before anything else
                "mode": namespace["MODE"],
                "run_real": namespace["RUN_REAL_CALIBRATION"],
                "run_model_stages": namespace["RUN_MODEL_STAGES"],
                "model_stages_enabled": namespace["MODEL_STAGES_ENABLED"],
                "confirm_100": namespace["CONFIRM_100_BUDGET"],
                "confirm_250": namespace["CONFIRM_250_BUDGET"],
                "confirm_1k": namespace["CONFIRM_1K_BUDGET"],
                "run_final_confirmation": namespace["RUN_FINAL_CONFIRMATION"],
                "publish": namespace["PUBLISH_VALIDATED_LENSES"],
                "active_scale_points": list(namespace["ACTIVE_SCALE_POINTS"]),
                # ---- the frozen design, before any result
                "layers": list(namespace["LAYERS"]),
                "target_layer": namespace["TARGET_LAYER"],
                "protocol_version": namespace["PROTOCOL_VERSION"],
                "plan_digest": namespace["PLAN"].digest,
                "active_plan_layers": list(namespace["ACTIVE_PLAN"].layers),
                "gate_digest": namespace["CALIBRATION_GATE"].digest,
                "plateau_digest": namespace["PLATEAU_RULE"].digest,
                # ---- corpus and splits
                "corpus_id": namespace["CORPUS_ID"],
                "split_sizes": {
                    name: len(namespace["PARTITIONS"].get(name))
                    for name in ("fit", "validation", "confirmation")
                },
                "split_checksums": {
                    name: namespace["PARTITIONS"].checksum(name)
                    for name in ("fit", "validation", "confirmation")
                },
                "leakage_ok": namespace["LEAKAGE"]["ok"],
                "leakage_exact_hits": namespace["LEAKAGE"]["n_exact_hits"],
                "leakage_near_hits": namespace["LEAKAGE"]["n_near_hits"],
                "nesting_exact": namespace["NESTING"]["nested"],
                "scales": scales,
                "n_dropped_short": len(namespace["DROPPED_SHORT"]),
                # ---- diversity
                "diversity_passed": namespace["DIVERSITY"].get("passed"),
                "n_distinct_targets": namespace["DIVERSITY"].get(
                    "n_distinct_target_tokens"
                ),
                "max_target_share": namespace["DIVERSITY"].get(
                    "max_target_token_share"
                ),
                # ---- model
                "architecture": namespace["ARCHITECTURE"],
                "model_class": type(namespace["MODEL"]).__name__,
                # ---- budget
                "budget": namespace["BUDGET"].to_dict(),
                # ---- fit
                "run_dir": str(namespace["RUN_DIR"]),
                "fingerprint_digest": namespace["FINGERPRINT"].digest,
                "resume_status_at_open": namespace["RESUME_STATUS"],
                "n_fitted": calibration.n_done if calibration else None,
                "snapshot_scales": sorted(calibration.snapshots) if calibration else [],
                "snapshot_n_prompts": (
                    {
                        str(scale): snapshot.n_prompts
                        for scale, snapshot in sorted(calibration.snapshots.items())
                    }
                    if calibration
                    else {}
                ),
                "diagnostic_layers": (
                    sorted(int(k) for k in calibration.diagnostics[-1]["layers"])
                    if calibration and calibration.diagnostics
                    else []
                ),
                # ---- verdicts
                "eligible_by_scale": namespace["COMPARISON"]["eligible_by_scale"],
                "n_comparison_rows": len(namespace["COMPARISON"]["rows"]),
                "n_delta_rows": len(namespace["COMPARISON"]["deltas"]),
                "plateau_verdict": namespace["PLATEAU"]["verdict"],
                "plateau_extension_justified": namespace["PLATEAU"][
                    "extension_justified"
                ],
                "plateau_runs_automatically": namespace["PLATEAU"][
                    "runs_automatically"
                ],
                "selected_scale": namespace["SELECTION"]["selected_scale"],
                "validation_scales": sorted(validation),
                # ---- confirmation and publication
                "vault_status": namespace["VAULT"].status(),
                "confirmation_ran": confirmation is not None,
                "confirmation_passed": (
                    {str(k): v["passed"] for k, v in confirmation.items()}
                    if confirmation
                    else None
                ),
                "n_published": (publication or {}).get("n_published"),
                "published_layers": (publication or {}).get("published_layers"),
                "failed_layers": (publication or {}).get("failed_layers"),
                # ---- report and resume
                "report_checksum": namespace["REPORT_PAYLOAD"]["report_checksum"],
                "report_has_mock_banner": "MOCK RUN" in namespace["REPORT_MARKDOWN"],
                "resume_status": (store.status_report() if store else {}).get("status"),
                "completed_units": (store.status_report() if store else {}).get(
                    "completed_units"
                ),
                "checkpoint_present": (store.status_report() if store else {}).get(
                    "checkpoint_present"
                ),
            },
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
