# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Execute the pilot notebook's code cells in a clean interpreter.

Run as a subprocess by ``tests/test_mmpilot_notebook.py`` from a working
directory outside the repository and with ``PYTHONPATH`` cleared, so ``jlens``
is *not* importable when the first cell runs. That is the whole point: the
notebook's section-0 bootstrap has to make the package importable itself, which
is what regressed when section 1 imported ``jlens`` before anything installed it.

Usage: ``python tests/_mmpilot_notebook_runner.py <notebook.ipynb>``.
Prints one JSON object on stdout; exits non-zero on the first failing cell.

The leading underscore keeps pytest from collecting this file as a test module.
"""

import json
import os
import sys
import traceback
from pathlib import Path


def main() -> int:
    notebook_path = Path(sys.argv[1])
    payload = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_cells = [cell for cell in payload["cells"] if cell["cell_type"] == "code"]

    if "jlens" in sys.modules:
        print(json.dumps({"ok": False, "error": "jlens was already imported"}))
        return 2

    namespace: dict = {"__name__": "__notebook__"}
    for index, cell in enumerate(code_cells):
        source = "".join(cell["source"])
        try:
            exec(compile(source, f"<cell {index}>", "exec"), namespace)  # noqa: S102
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            print(
                json.dumps(
                    {
                        "ok": False,
                        "cell": index,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc()[-3000:],
                    }
                )
            )
            return 1

    import jlens

    summary = namespace["SUMMARY"]
    status = namespace["STATUS"]
    print(
        json.dumps(
            {
                "ok": True,
                "n_code_cells": len(code_cells),
                "cwd": os.getcwd(),
                "jlens_file": jlens.__file__,
                "repo_path": str(namespace["REPO_PATH"]),
                "checked_out_branch": namespace["CHECKED_OUT_BRANCH"],
                "commit": namespace["COMMIT"],
                "run_real_pilot": namespace["RUN_REAL_PILOT"],
                "model_is_none": namespace["MODEL"] is None,
                "recommendation": summary["recommendation"],
                "scientific_evidence": summary["scientific_evidence"],
                "leakage_ok": namespace["LEAKAGE"]["ok"],
                "invariance_passed": namespace["INVARIANCE"]["passed"],
                "resume_status": status["status"],
                "n_interventions": status["completed_units"]["intervention"],
                "run_dir": status["run_dir"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
