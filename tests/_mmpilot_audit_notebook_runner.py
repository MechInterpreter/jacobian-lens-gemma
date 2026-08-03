# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Execute the image-independence audit notebook's cells in a clean interpreter.

Run as a subprocess by ``tests/test_mmpilot_image_audit_notebook.py`` from a
working directory outside the repository and with ``PYTHONPATH`` cleared, so
``jlens`` is *not* importable when the first cell runs. The notebook's
section-0 bootstrap has to make the package importable itself.

Usage::

    python tests/_mmpilot_audit_notebook_runner.py <notebook.ipynb> [NAME=True ...]

Each ``NAME=True`` argument rewrites a committed ``NAME = False`` switch, the
same edit a user makes by hand. Without it the committed defaults run, which is
how the "nothing happens by accident" path is tested.

``MMPILOT_AUDIT_RUN_DIR`` points the notebook at a run directory and
``MMPILOT_AUDIT_EXPECT_FINGERPRINT`` overrides the pinned run fingerprint.

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
                        "traceback": traceback.format_exc()[-3000:],
                    }
                )
            )
            return 1

    import jlens

    audit = namespace.get("AUDIT")
    # `transformers` is what loads a checkpoint and `jlens.gemma4` is what
    # wraps one. Neither may be imported on this path, let alone called.
    modules = sorted(
        name
        for name in sys.modules
        if name.startswith(("transformers", "jlens.gemma4"))
    )
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
                "run_audit": namespace["RUN_IMAGE_INDEPENDENCE_AUDIT"],
                "run_dir": str(namespace["COMPLETED_RUN_DIR"]),
                "audit_is_none": audit is None,
                "verdict": (audit or {}).get("verdict"),
                "original_recommendation": (audit or {}).get("original_recommendation"),
                "model_loaded": (audit or {}).get("model_loaded"),
                "resume_status": ((audit or {}).get("resume") or {}).get("status"),
                "reused": ((audit or {}).get("resume") or {}).get("reused"),
                "computed": ((audit or {}).get("resume") or {}).get("computed"),
                "artifacts": (audit or {}).get("artifacts"),
                "all_originals_unchanged": (
                    (audit or {}).get("preservation") or {}
                ).get("all_unchanged"),
                "n_groups": ((audit or {}).get("audit") or {}).get("n_groups"),
                "n_distinct_images": ((audit or {}).get("audit") or {}).get(
                    "n_distinct_images"
                ),
                # Nothing model-shaped may be imported on this path.
                "model_modules_imported": modules,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
