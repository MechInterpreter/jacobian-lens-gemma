# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Execute the full-vocabulary notebook's code cells in a clean interpreter.

Run as a subprocess by ``tests/test_full_vocabulary_notebook.py`` from a working
directory outside the repository and with ``PYTHONPATH`` cleared, so ``jlens`` is
*not* importable when the first cell runs — the notebook's bootstrap has to make
the package importable itself.

Usage::

    python tests/_full_vocabulary_notebook_runner.py <notebook.ipynb> [NAME=True ...]

Each ``NAME=True`` argument rewrites a committed ``NAME = False`` switch before
execution, the same edit a user makes by hand.

Prints one JSON object on stdout; exits non-zero on the first failing cell.

The leading underscore keeps pytest from collecting this file as a test module.
"""

import json
import os
import re
import sys
import traceback
from pathlib import Path


def _cuda_initialized() -> bool:
    torch = sys.modules.get("torch")
    if torch is None:
        return False
    try:
        return bool(torch.cuda.is_initialized())
    except Exception:  # noqa: BLE001 - a probe, never a failure
        return False


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
                        "traceback": traceback.format_exc()[-3000:],
                    }
                )
            )
            return 1

    import jlens

    mock_results = namespace.get("MOCK_RESULTS") or {}
    report = namespace.get("REPORT") or {}
    audit = namespace.get("AUDIT") or {}
    budget = namespace.get("BUDGET") or {}
    print(
        json.dumps(
            {
                "ok": True,
                "n_code_cells": len(code_cells),
                "cwd": os.getcwd(),
                "jlens_file": jlens.__file__,
                "overrides": overrides,
                "real_mode": namespace.get("REAL_MODE"),
                "gpu_stage": namespace.get("GPU_STAGE"),
                "audit_passed": audit.get("passed"),
                "audit_digest": audit.get("audit_digest"),
                "n_overclaims": (audit.get("source_scan") or {}).get("n_overclaims"),
                "n_amendments": len(namespace.get("AMENDMENTS") or {}),
                "budget_total": budget.get("total"),
                "budget_within_cap": budget.get("within_cap"),
                "mock_results": {
                    key: {
                        "verdict": value["verdict"],
                        "expected": value["expected"],
                        "as_required": value["as_required"],
                    }
                    for key, value in mock_results.items()
                },
                "report_mode": report.get("mode"),
                "unrestricted_verdict": (
                    report.get("unrestricted_output_verdict") or {}
                ).get("verdict"),
                "conditional_verdict": (
                    report.get("conditional_logprob_verdict") or {}
                ).get("verdict"),
                "conjunction_verdict": (
                    report.get("cross_modal_conjunction") or {}
                ).get("verdict"),
                # The two things a reviewer checks first.
                "loaded_gemma": "transformers" in sys.modules,
                "torch_cuda_initialized": _cuda_initialized(),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
