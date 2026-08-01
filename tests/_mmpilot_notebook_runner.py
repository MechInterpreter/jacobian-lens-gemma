# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Execute the pilot notebook's code cells in a clean interpreter.

Run as a subprocess by ``tests/test_mmpilot_notebook.py`` from a working
directory outside the repository and with ``PYTHONPATH`` cleared, so ``jlens``
is *not* importable when the first cell runs. That is the whole point: the
notebook's section-0 bootstrap has to make the package importable itself, which
is what regressed when section 1 imported ``jlens`` before anything installed it.

Usage::

    python tests/_mmpilot_notebook_runner.py <notebook.ipynb> [NAME=True ...]

Each ``NAME=True`` argument rewrites a committed ``NAME = False`` switch in the
notebook source before execution — the same edit a user makes by hand, made
visibly rather than through a back door the notebook itself would have to
honour. Without it the committed defaults run, which is how the CPU-only path
is tested.

Two environment variables redirect the notebook's Drive-shaped paths at a
temporary directory, so the derived-data cache can be exercised end to end
without a Drive mount: ``MMPILOT_CACHE_ROOT`` and ``MMPILOT_CACHE_STAGING``.

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

    summary = namespace.get("SUMMARY")
    status = namespace.get("STATUS")
    stats = namespace["AUDIT_STATS"]
    universe = namespace["UNIVERSE"]
    ranking = namespace["RANKING"]
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
                "run_real_pilot": namespace["RUN_REAL_PILOT"],
                "run_model_stages": namespace["RUN_MODEL_STAGES"],
                "model_is_none": namespace["MODEL"] is None,
                "selected_concepts": sorted(namespace["SELECTED_NAMES"]),
                "n_valid_positives": stats["n_valid_synchronized_positives"],
                "n_rejected": stats["n_rejected"],
                "rejection_counts": namespace["REJECTION_COUNTS"]["total"],
                "lexicon_hash": namespace["EVIDENCE_CONFIG"].lexicon_hash,
                # Discovery: what the candidate universe actually was.
                "discovered_categories": list(universe.categories),
                "eligible_categories": list(universe.eligible),
                "excluded_categories": list(universe.excluded),
                "universe_hash": universe.universe_hash,
                "lexical_hash": universe.lexical_hash,
                "ranked_concepts": [row["concept"] for row in ranking],
                "n_feasible": sum(1 for row in ranking if row["feasible"]),
                # Cache behaviour.
                "cache_state": namespace["CACHE_STATE"],
                "cache_hit": namespace["CACHE_HIT"],
                "cache_dir": str(namespace["DERIVED_CACHE"].directory),
                "cache_fingerprint": namespace["CACHE_FINGERPRINT"].digest,
                "cache_published": namespace["CACHE_PUBLISH"].get("published"),
                "run_dir": str(namespace["RUN_DIR"]),
                "leakage_ok": namespace["LEAKAGE"]["ok"],
                "recommendation": (summary or {}).get("recommendation"),
                "scientific_evidence": (summary or {}).get("scientific_evidence"),
                "not_evaluated": sorted((summary or {}).get("not_evaluated", {})),
                "invariance_passed": (namespace["INVARIANCE"] or {}).get("passed"),
                "resume_status": (status or {}).get("status"),
                "n_interventions": (status or {}).get("completed_units", {}).get(
                    "intervention"
                ),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
