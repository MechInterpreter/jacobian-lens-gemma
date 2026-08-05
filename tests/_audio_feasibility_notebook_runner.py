# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Execute the audio-feasibility notebook's code cells in a clean interpreter.

Run as a subprocess by ``tests/test_mmpilot_audio_notebook.py`` from a working
directory outside the repository and with ``PYTHONPATH`` cleared, so ``jlens``
is *not* importable when the first cell runs — the notebook's section-1
bootstrap has to make the package importable itself.

Usage::

    python tests/_audio_feasibility_notebook_runner.py <notebook.ipynb> [NAME=True ...]

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

    audit = namespace.get("AUDIT")
    interface = namespace.get("AUDIO_INTERFACE")
    print(
        json.dumps(
            {
                "ok": True,
                "n_code_cells": len(code_cells),
                "cwd": os.getcwd(),
                "jlens_file": jlens.__file__,
                "overrides": overrides,
                "mode": namespace["MODE"],
                "run_real": namespace["RUN_REAL_AUDIO_AUDIT"],
                "run_model_stage": namespace["RUN_MODEL_STAGE"],
                "confirm_model_load": namespace["CONFIRM_MODEL_LOAD"],
                "processor_class": namespace["INTERFACE"]["processor_class"],
                "supports_audio_components": namespace["INTERFACE"]["supports_audio"],
                "protocol_resolved": interface is not None,
                "protocol_fingerprint": (
                    interface.protocol_fingerprint if interface is not None else None
                ),
                "placeholder_counts": (
                    list(interface.notes["placeholder_counts"])
                    if interface is not None
                    else None
                ),
                "dynamic_placeholder_count": (
                    interface.dynamic_placeholder_count if interface is not None else None
                ),
                "call_convention": (
                    interface.call_convention if interface is not None else None
                ),
                "n_waveforms": len(namespace["WAVEFORMS"]),
                "backend_is_none": namespace["BACKEND"] is None,
                "verdict": audit.verdict if audit is not None else None,
                "failed_checks": audit.failed if audit is not None else None,
                "check_names": (
                    [check.name for check in audit.checks] if audit is not None else []
                ),
                "activation_layers": sorted(namespace.get("ACTIVATIONS") or {}),
                "run_dir": str(namespace["RUN_DIR"]),
                "report_checksum": (
                    audit.to_dict()["report_checksum"] if audit is not None else None
                ),
                "written": namespace["WRITTEN"]["json"],
                "environment": namespace["ENVIRONMENT"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
