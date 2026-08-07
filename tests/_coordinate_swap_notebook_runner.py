# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Execute the coordinate-swap MOCK notebook's code cells in a clean interpreter.

Run as a subprocess by ``tests/test_coordinate_swap_notebook.py`` from a working
directory outside the repository and with ``PYTHONPATH`` cleared, so ``jlens``
is *not* importable when the first cell runs — the notebook's section-1
bootstrap has to make the package importable itself.

Usage::

    python tests/_coordinate_swap_notebook_runner.py <notebook.ipynb> [NAME=True ...]

Each ``NAME=True`` argument rewrites a committed ``NAME = False`` switch before
execution, the same edit a user makes by hand. The only switch this notebook
has is ``RUN_REAL_COORDINATE_SWAP``, and flipping it is expected to *raise* —
there is no real experiment to run yet.

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

    summary = namespace.get("SUMMARY") or {}
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
                "overrides": overrides,
                "run_real_coordinate_swap": namespace["RUN_REAL_COORDINATE_SWAP"],
                "mode": namespace["MODE"],
                "status": namespace.get("STATUS"),
                "summary": summary,
                # The two things a reviewer checks first.
                "loaded_gemma": "transformers" in sys.modules,
                "torch_cuda_initialized": _cuda_initialized(),
            }
        )
    )
    return 0


def _cuda_initialized() -> bool:
    import torch

    return bool(torch.cuda.is_initialized())


if __name__ == "__main__":
    raise SystemExit(main())
