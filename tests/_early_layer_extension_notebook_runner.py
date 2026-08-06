# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Execute the early-layer extension notebook's code cells in a clean interpreter.

Run as a subprocess by ``tests/test_early_layer_extension_notebook.py`` from a
working directory outside the repository and with ``PYTHONPATH`` cleared, so
``jlens`` is *not* importable when the first cell runs — the notebook's section-0
bootstrap has to make the package importable itself.

Usage::

    python tests/_early_layer_extension_notebook_runner.py <notebook.ipynb> [NAME=True ...]

Each ``NAME=True`` argument rewrites a committed ``NAME = False`` switch, the
same edit a user makes by hand. ``MOCK_SCENARIO=<key>`` selects one of the
commissioned synthetic outcomes. Without any argument the committed defaults
run, which is how "opening the notebook starts nothing" is tested.

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
    scenario = overrides.pop("MOCK_SCENARIO", None)
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
        if scenario is not None:
            source = re.sub(
                r'^MOCK_SCENARIO = "[^"]+"$',
                f'MOCK_SCENARIO = "{scenario}"',
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
    continuation = namespace.get("CONTINUATION")
    development = namespace.get("DEVELOPMENT") or {}
    confirmation = namespace.get("CONFIRMATION")
    publication = namespace.get("PUBLICATION")
    verdict = namespace.get("VERDICT")
    splits = namespace["SPLITS"]
    parent = namespace["PARENT"]

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
                "scenario": namespace.get("MOCK_SCENARIO"),
                # ---- switches, before anything else
                "mode": namespace["MODE"],
                "run_real": namespace["RUN_REAL_EARLY_LAYER_EXTENSION"],
                "run_model_stages": namespace["RUN_MODEL_STAGES"],
                "confirm_parent_import": namespace["CONFIRM_PARENT_IMPORT"],
                "model_stages_enabled": namespace["MODEL_STAGES_ENABLED"],
                "confirm_250": namespace["CONFIRM_250_BUDGET"],
                "confirm_1k": namespace["CONFIRM_1K_BUDGET"],
                "run_fresh_development": namespace["RUN_FRESH_DEVELOPMENT"],
                "run_final_confirmation": namespace["RUN_FINAL_CONFIRMATION"],
                "publish": namespace["PUBLISH_VALIDATED_EARLY_LENSES"],
                "active_scale_points": list(namespace["ACTIVE_SCALE_POINTS"]),
                # ---- the frozen design, before any result
                "protocol_version": namespace["PROTOCOL_VERSION"],
                "protocol_digest": namespace["EXTENSION_PROTOCOL"].digest,
                "gate_digest": namespace["EXTENSION_GATE"].digest,
                "gate_version": namespace["EXTENSION_GATE"].version,
                "gate_n_prompts": namespace["EXTENSION_GATE"].n_prompts,
                "gate_min_distinct_targets": namespace[
                    "EXTENSION_GATE"
                ].min_distinct_target_tokens,
                "selection_rule_digest": namespace["EXTENSION_SELECTION_RULE"].digest,
                "layers": list(namespace["LAYERS"]),
                "active_plan_layers": list(namespace["ACTIVE_PLAN"].layers),
                "publishable_layers": list(namespace["PUBLISHABLE_LAYERS"]),
                "scales": list(namespace["SCALES"]),
                "development_scales": list(namespace["DEV_SCALES"]),
                # ---- the parent
                "parent_root": parent.root,
                "parent_fingerprint_digest": parent.fingerprint_digest,
                "parent_accumulator_checksum": parent.accumulator.checksum,
                "parent_n_done": parent.accumulator.n_done,
                "parent_audit_compatible": namespace["PARENT_AUDIT"]["compatible"],
                "parent_audit_failed_checks": namespace["PARENT_AUDIT"][
                    "failed_checks"
                ],
                "parent_confirmation_vault": parent.confirmation_vault_status,
                "parent_immutable": namespace["IMMUTABILITY"]["immutable"],
                "parent_files_checked": namespace["IMMUTABILITY"]["n_files_checked"],
                "reconstruction_all_match": namespace["RECONSTRUCTION"]["all_match"],
                "prefix_matches": namespace["PREFIX"]["matches"],
                "prefix_skip_authorized": namespace["PREFIX"]["skip_authorized"],
                # ---- the fresh splits
                "fit_records": len(namespace["FIT_RECORDS"]),
                "extension_pool": len(namespace["EXTENSION_POOL"]),
                "split_sizes": {
                    name: len(splits.get(name))
                    for name in ("development", "confirmation")
                },
                "split_checksums": {
                    name: splits.checksum(name)
                    for name in ("development", "confirmation")
                },
                "excluded_exact": splits.excluded_exact,
                "excluded_near": splits.excluded_near,
                "split_leakage_ok": namespace["SPLIT_LEAKAGE"]["ok"],
                "split_leakage_pairs": namespace["SPLIT_LEAKAGE"][
                    "candidate_pairs_compared"
                ],
                "diversity_passed": namespace["DIVERSITY"]["passed"],
                "n_distinct_targets": namespace["DIVERSITY"][
                    "n_distinct_target_tokens"
                ],
                "max_target_share": namespace["DIVERSITY"]["max_target_token_share"],
                "confirmation_diversity_passed": namespace[
                    "CONFIRMATION_DIVERSITY"
                ]["passed"],
                "confirmation_selected_by_jlens": namespace[
                    "CONFIRMATION_SELECTION"
                ]["selected_by_jlens_performance"],
                # ---- the continuation
                "run_dir": str(namespace["RUN_DIR"]),
                "fingerprint_digest": namespace["FINGERPRINT"].digest,
                "resume_status_at_open": namespace["RESUME_STATUS"],
                "seed_action": namespace["SEED"]["action"],
                "seed_parent_written": namespace["SEED"]["parent_written"],
                "n_fitted": continuation.n_done if continuation else None,
                "snapshot_scales": (
                    sorted(continuation.snapshots) if continuation else []
                ),
                "snapshot_n_prompts": (
                    {
                        str(scale): snapshot.n_prompts
                        for scale, snapshot in sorted(continuation.snapshots.items())
                    }
                    if continuation
                    else {}
                ),
                "equivalence": namespace.get("EQUIVALENCE"),
                # ---- verdicts
                "eligible_by_scale": namespace["COMPARISON"]["eligible_by_scale"],
                "plateau_verdict": namespace["PLATEAU"]["verdict"],
                "selected_scale": namespace["SELECTION"]["selected_scale"],
                "selection_clause": namespace["SELECTION"]["clause_applied"],
                "selection_confirmation_not_consulted": namespace["SELECTION"][
                    "confirmation_not_consulted"
                ],
                "development_scales_scored": sorted(development),
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
                "verdict": (verdict or {}).get("verdict"),
                "verdict_early_layers": (verdict or {}).get(
                    "earlier_layers_passing_confirmation"
                ),
                "verdict_statement": (verdict or {}).get("statement"),
                # ---- resume
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
