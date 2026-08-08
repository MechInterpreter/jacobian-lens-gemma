# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Execute the L32 follow-up notebook's code cells in a clean interpreter.

Run as a subprocess by ``tests/test_l32_followup_notebook.py`` from a working
directory outside the repository and with ``PYTHONPATH`` cleared, so ``jlens``
is *not* importable when the first cell runs — the notebook's section-1
bootstrap has to make the package importable itself.

Usage::

    python tests/_l32_followup_notebook_runner.py <notebook.ipynb> [NAME=True ...]

Each ``NAME=True`` argument rewrites a committed ``NAME = False`` switch before
execution, the same edit a user makes by hand.

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
    "CAUSAL_LAYER",
    "REFERENCE_LAYER",
    "CROSS_LAYER_ALLOWED_WITHOUT_REFERENCE",
    "DISCOVERY_REFUSALS",
    "MODEL_STAGES_ENABLED",
    "PAIRED_REFERENCE_ENABLED",
    "OPEN_QUESTION",
    "SELECTED_NAMES",
    "FOCAL_CONCEPTS",
)


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

    result = {"ok": True, "n_cells": len(code_cells)}
    for name in REPORTED:
        if name in namespace:
            result[name] = namespace[name]

    for name in (
        "VERDICT_A",
        "VERDICT_B",
        "VERDICT_C",
        "VERDICT_D",
        "VERDICT_E",
    ):
        value = namespace.get(name)
        if isinstance(value, dict):
            result[name] = value.get("verdict") or value.get("classification")

    for name, key in (
        ("COMPARABILITY", "paired_reference_required"),
        ("CROSS_LAYER", "licensed"),
        ("RECOMMENDATION", "recommendation"),
        ("PHRASING", "passed"),
    ):
        value = namespace.get(name)
        if isinstance(value, dict):
            result[f"{name}.{key}"] = value.get(key)

    report = namespace.get("REPORT")
    if isinstance(report, dict):
        result["report_schema"] = report.get("schema")
        result["intervention_family"] = report.get("intervention_family")
        result["mock_proves_pipeline_only"] = report.get("mock_proves_pipeline_only")
        result["run_dir"] = str(report.get("run_dir"))

    print(json.dumps(result, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
