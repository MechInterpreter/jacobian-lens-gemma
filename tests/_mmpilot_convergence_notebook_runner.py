# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Execute the output-convergence notebook's code cells in a clean interpreter.

Run as a subprocess by ``tests/test_mmpilot_convergence_notebook.py`` from a
working directory outside the repository and with ``PYTHONPATH`` cleared, so
``jlens`` is *not* importable when the first cell runs — the notebook's
section-1 bootstrap has to make the package importable itself.

Usage::

    python tests/_mmpilot_convergence_notebook_runner.py <notebook.ipynb> [NAME=True ...]

Each ``NAME=True`` argument rewrites a committed ``NAME = False`` switch, the
same edit a user makes by hand. Without any, the committed defaults run, which
is how "opening the notebook loads no model and opens no completed run" is
tested.

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

    mock = namespace.get("MOCK_RESULT") or {}
    result = namespace.get("RESULT") or {}
    active = result or mock
    store = mock.get("store")
    # transformers and the Gemma adapter must never be imported on the MOCK
    # path: "no model is loaded" has to be observable, not asserted.
    forbidden_modules = sorted(
        name
        for name in sys.modules
        if name == "transformers"
        or name.startswith("transformers.")
        or name.startswith("jlens.gemma4")
        or name.startswith("jlens.mmpilot.real_backend")
    )
    run_dir = Path(mock["completed_run_dir"]) if mock.get("completed_run_dir") else None

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
                # ---- switches
                "run_real": namespace["RUN_REAL_CONVERGENCE_AUDIT"],
                "confirm_model_load": namespace["CONFIRM_MODEL_LOAD"],
                "run_secondary_probe": namespace["RUN_SECONDARY_PROBE"],
                "forbidden_modules_imported": forbidden_modules,
                "model_is_none": namespace.get("MODEL") is None,
                "head_is_none": namespace.get("HEAD") is None,
                "completed_run_opened": namespace.get("COMPLETED_RUN") is not None,
                # ---- the fixed design, decided before any measurement
                "layers": list(namespace["LAYERS"]),
                "primary_layer": namespace["PRIMARY_LAYER"],
                "modalities": list(namespace["MODALITIES"]),
                "lens_invalid_layers": list(namespace["LENS_INVALID_LAYERS"]),
                "combined_lens_checksum": namespace["COMBINED_LENS_CHECKSUM"],
                "criterion_digest": namespace["CONVERGENCE_CRITERION"].digest,
                "criterion_text_present": bool(namespace["CRITERION_TEXT"]),
                # ---- MOCK results
                "mock_verdict": (mock.get("verdict") or {}).get("verdict"),
                "mock_matrix": namespace.get("MOCK_MATRIX"),
                "mock_controls_passed": (mock.get("controls") or {}).get(
                    "all_controls_passed"
                ),
                "mock_run_unchanged": (mock.get("immutability") or {}).get("unchanged"),
                "mock_readout_mode": (mock.get("tokenization") or {}).get(
                    "readout_mode"
                ),
                "mock_norm_convention": (mock.get("head_audit") or {}).get(
                    "norm_weight_convention"
                ),
                "mock_probe_ran": (mock.get("probe") or {}).get("ran"),
                "mock_probe_determines_verdict": (mock.get("probe") or {}).get(
                    "determines_verdict"
                ),
                "mock_admissible_concepts": (mock.get("population") or {}).get(
                    "admissible_concepts"
                ),
                "mock_inadmissible_concepts": (mock.get("population") or {}).get(
                    "inadmissible_concepts"
                ),
                # ---- artifacts
                "table_layers": [row["layer"] for row in (active.get("table") or [])],
                "table_columns": sorted((active.get("table") or [{}])[0]),
                # The measured trajectory itself, so determinism can be asserted
                # on results rather than on a fingerprint that (correctly) binds
                # the temporary run directory it was audited from.
                "table_agreements": [
                    row["clean_agreement_unique"] for row in (active.get("table") or [])
                ],
                "table_classifications": [
                    row["convergence_classification"]
                    for row in (active.get("table") or [])
                ],
                "audit_dir": str(store.root) if store is not None else None,
                "audit_entries": sorted(path.name for path in store.root.iterdir())
                if store is not None
                else [],
                "audit_fingerprint": store.fingerprint.digest
                if store is not None
                else None,
                "audit_resume_status": store.status_report().get("status")
                if store is not None
                else None,
                "completed_run_entries": sorted(
                    path.name for path in run_dir.iterdir()
                )
                if run_dir is not None and run_dir.is_dir()
                else [],
                "verdict_checks": [
                    check["check"] for check in (active.get("verdict") or {}).get("checks", [])
                ],
                "interpretation_boundary_in_verdict": bool(
                    (active.get("verdict") or {}).get("interpretation_boundary")
                ),
            },
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
