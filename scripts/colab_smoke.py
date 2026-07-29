# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Lightweight remote smoke test for the generative validation workflow.

Designed to be the first thing run on a fresh Colab CLI session::

    colab exec -s jlens -f scripts/colab_smoke.py

``colab exec -f`` execs the script's *source* inside a live Jupyter kernel
rather than running it as a file, so ``__file__`` is undefined there and the
kernel's working directory need not be the repo checkout. REPO_ROOT is
therefore resolved from ``__file__`` when available (local runs, a plain
``python scripts/colab_smoke.py``) and otherwise located at the checkout
``colab console`` clones under ``/content`` (see
docs/generative_validation.md), with the branch verified/switched to match.

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

REPO_URL = "https://github.com/MechInterpreter/jacobian-lens-gemma.git"
EXPECTED_BRANCH = "experiment/generative-jlens-validation"
CONTENT_DIR = "/content"
REPO_DIRNAME = "jacobian-lens-gemma"


def resolve_repo_root(file_dunder: str | None, content_dir: str = CONTENT_DIR) -> str:
    """Locate the repository root.

    ``file_dunder`` is the caller's own ``__file__`` if it has one (``None``
    under ``colab exec -f``, whose source is exec'd with no ``__file__`` in
    its globals). When it is ``None``, fall back to the pushed checkout at
    ``<content_dir>/jacobian-lens-gemma``.
    """
    if file_dunder is not None:
        return os.path.dirname(os.path.dirname(os.path.abspath(file_dunder)))
    candidate = os.path.join(content_dir, REPO_DIRNAME)
    if os.path.isdir(os.path.join(candidate, "jlens")):
        return candidate
    raise RuntimeError(
        f"__file__ is undefined (running under colab exec) and no repo "
        f"checkout was found at {candidate!r}; clone it first: "
        f"git clone --branch {EXPECTED_BRANCH} {REPO_URL} {candidate}"
    )


def ensure_branch(repo_root: str, branch: str = EXPECTED_BRANCH) -> str:
    """Check out ``branch`` in ``repo_root`` if it isn't already current;
    return the branch actually checked out."""
    current = subprocess.run(
        ["git", "-C", repo_root, "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if current == branch:
        return current
    subprocess.run(["git", "-C", repo_root, "fetch", "origin", branch], check=True)
    subprocess.run(["git", "-C", repo_root, "checkout", branch], check=True)
    return branch


REPO_ROOT = resolve_repo_root(globals().get("__file__"))
sys.path.insert(0, REPO_ROOT)


def main() -> int:
    import torch
    import transformers

    branch = ensure_branch(REPO_ROOT)

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
        "branch": branch,
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
