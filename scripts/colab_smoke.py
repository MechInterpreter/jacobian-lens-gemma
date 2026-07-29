# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Lightweight remote smoke test for the generative validation workflow.

Designed to be the first thing run on a fresh Colab CLI session::

    colab exec -s jlens -f scripts/colab_smoke.py

Checks, without loading Gemma (no HF token needed, no large downloads):

1. Python/torch/transformers versions and GPU visibility.
2. The repository imports and its config/benchmark artifacts validate.
3. The generative test suite passes on the CPU mock (the exact code path
   the real run takes, minus the checkpoint).

Exits non-zero on any failure, so ``colab run``/``colab exec`` propagate it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


def main() -> int:
    import torch
    import transformers

    report = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "hf_token_present": bool(os.environ.get("HF_TOKEN")),
        "repo_root": REPO_ROOT,
    }

    from jlens.generative import load_benchmark
    from jlens.metadata import load_generative_config

    config = load_generative_config(
        os.path.join(REPO_ROOT, "configs", "gemma_generative_validation.yaml")
    )
    manifest = load_benchmark(config["benchmark"]["manifest_path"])
    report["config_ok"] = True
    report["benchmark_examples"] = {
        "dev": len(manifest["dev"]),
        "heldout": len(manifest["heldout"]),
    }

    tests = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_generative.py",
            "tests/test_generative_runner.py",
            "-q",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    report["tests_passed"] = tests.returncode == 0
    report["tests_tail"] = tests.stdout.strip().splitlines()[-1:] + (
        tests.stderr.strip().splitlines()[-3:] if tests.returncode != 0 else []
    )

    print(json.dumps(report, indent=2))
    if not report["tests_passed"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
