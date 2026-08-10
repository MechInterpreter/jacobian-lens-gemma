# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Generate ``notebooks/archive/engineering_audits/mmpilot_image_independence_audit_colab.ipynb``.

The notebook is small and every cell matters, so it is written from source
here rather than edited by hand in a JSON blob: that keeps the committed
notebook output-free and byte-reproducible.

Run with ``python scripts/_build_audit_notebook.py`` after changing a cell.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TARGET = REPO_ROOT / "notebooks" / "archive" / "engineering_audits" / "mmpilot_image_independence_audit_colab.ipynb"

CELLS: list[tuple[str, str]] = []


def markdown(text: str) -> None:
    CELLS.append(("markdown", text.strip("\n")))


def code(text: str) -> None:
    CELLS.append(("code", text.strip("\n")))


markdown(
    """
# Image-independence audit of a completed SpokenCOCO pilot run — CPU only

**One question.** How much of a completed run's verdict survives once a
*photograph*, rather than a synchronized caption/audio group, is the
independent unit?

SpokenCOCO gives one COCO image several written captions and a spoken reading
of each. The pilot's subset builder keeps more than one of them, so **one image
can enter a run as two synchronized groups**. Those two groups are one
photograph seen twice. Two things follow, and this notebook does both:

1. **Representation.** A cross-modal query must not retrieve its own
   photograph. Excluding only the query's exact group does not achieve that
   when a sibling group carries the same image, so the corrected rule excludes
   on `image_id` and supersedes the group rule.
2. **Causation.** Intervention units are stored per group. Averaging them flat
   counts one image twice — pseudoreplication. The corrected aggregation
   averages within image first and reports `n` in photographs.

**This notebook never loads Gemma.** No model, no processor, no Hugging Face
token, no GPU, no media. It reads the saved per-unit JSON of a finished run and
rearranges it. That is the whole reason it fits on a free CPU runtime.

**Nothing original is overwritten.** `report.md`, `summary.json`, the unit
files and the manifests are read-only inputs. Every output is a new versioned
artifact, and the originals' checksums are recorded before and after so the
claim is checkable rather than asserted.

**The original verdict is not privileged.** The amended verdict is one of
`GO_CONFIRMED_AFTER_IMAGE_DEDUP`, `WEAK_GO_AFTER_IMAGE_DEDUP`,
`NO_GO_AFTER_IMAGE_DEDUP` or `AUDIT_BLOCKED`, computed by the same rubric from
the corrected numbers.
"""
)

markdown(
    """
## 0. Colab bootstrap

Run these three cells first, in order. They use nothing but the standard
library: the repository is not importable until 0c has installed it, so
anything that says `from jlens...` before then would fail with
`ModuleNotFoundError: No module named 'jlens'`.

Google Drive is **not** needed here — the package is installed and verified
before section 2 mounts anything.
"""
)

code(
    '''
# 0a. Bootstrap constants only. Nothing from this repository is imported yet.
REPO_URL = "https://github.com/MechInterpreter/jacobian-lens-gemma.git"
BRANCH = "experiment/spokencoco-jspace-pilot"
REPO_DIR = "/content/jacobian-lens-gemma"

print(f"repo   {REPO_URL}")
print(f"branch {BRANCH}")
print(f"target {REPO_DIR}")
'''
)

code(
    '''
# 0b. Clone or update the repository, then verify the checked-out branch.
#
# Idempotent: clones when absent, otherwise fetches the branch, checks it out,
# and resets to origin. The reset discards local edits inside the Colab
# checkout — that directory is scratch, not somewhere to keep work.
import os
import subprocess
import sys
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
REPO_PATH = Path(os.environ.get("MMPILOT_REPO_DIR") or REPO_DIR)


def _git(*arguments, cwd=None):
    result = subprocess.run(
        ["git", *arguments], cwd=cwd, capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(arguments)} failed:\\n{result.stdout}\\n{result.stderr}"
        )
    return result.stdout.strip()


if IN_COLAB:
    if not (REPO_PATH / ".git").is_dir():
        _git("clone", "--branch", BRANCH, REPO_URL, str(REPO_PATH))
    else:
        _git("fetch", "origin", BRANCH, cwd=REPO_PATH)
        _git("checkout", BRANCH, cwd=REPO_PATH)
        _git("reset", "--hard", f"origin/{BRANCH}", cwd=REPO_PATH)

CHECKED_OUT_BRANCH = _git("rev-parse", "--abbrev-ref", "HEAD", cwd=REPO_PATH)
COMMIT = _git("rev-parse", "HEAD", cwd=REPO_PATH)
if IN_COLAB and CHECKED_OUT_BRANCH != BRANCH:
    raise RuntimeError(
        f"checked out {CHECKED_OUT_BRANCH!r}, expected {BRANCH!r} — "
        "refusing to continue against the wrong code"
    )
print(f"branch {CHECKED_OUT_BRANCH}")
print(f"commit {COMMIT}")
'''
)

code(
    '''
# 0c. Install the repository, move into it, and verify that `import jlens`
# resolves to this checkout. Every later cell may import from the package.
if IN_COLAB:
    print("installing the repository (editable) ...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", str(REPO_PATH)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"pip install -e failed:\\n{result.stdout[-2000:]}\\n{result.stderr[-2000:]}"
        )

os.chdir(REPO_PATH)
if str(REPO_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_PATH))

try:
    import jlens
except ModuleNotFoundError as error:
    raise RuntimeError(
        f"`import jlens` is still not importable after installation: {error}"
    ) from error

if Path(jlens.__file__).resolve().parent.parent != REPO_PATH.resolve():
    raise RuntimeError(
        f"`import jlens` resolved to {jlens.__file__}, not this checkout — "
        "another installation is shadowing this checkout"
    )
print(f"jlens  {jlens.__file__}")
print(f"cwd    {os.getcwd()}")
'''
)

markdown(
    """
## 1. Configuration

`RUN_IMAGE_INDEPENDENCE_AUDIT` is **False** in the committed notebook. Running
every cell as it ships does nothing but bootstrap and print what to set. Flip
it to True by hand to audit the run named below.

`COMPLETED_RUN_DIR` names the finished run explicitly. Nothing is discovered,
globbed, or guessed — auditing the wrong directory would produce a confident
verdict about a run nobody asked about.

`EXPECTED_RUN_FINGERPRINT` pins the run's identity. A mismatch stops the audit
rather than proceeding against artifacts from a different configuration. Set it
to `""` to skip the check (only sensible for a MOCK run).
"""
)

code(
    '''
# 1. Configuration. Requires section 0 (it imports from the repository).
# Nothing here mounts Drive, reads data, or loads anything.
#
# Flip RUN_IMAGE_INDEPENDENCE_AUDIT to True by hand to run the audit.
RUN_IMAGE_INDEPENDENCE_AUDIT = False

# The completed run to audit, named explicitly.
COMPLETED_RUN_DIR = (
    "/content/drive/MyDrive/jacobian-lens-gemma/runs/mmpilot_pilot_20260803T160711"
)
EXPECTED_RUN_FINGERPRINT = (
    "sha256:728eedcd6bac6bab2faf78c2c7861d369a3ddf91e53b1226ff50b635cdb97f5b"
)

# Optional. The pilot writes its subset into the derived cache rather than the
# run directory, so this is usually left empty: the saved units already carry
# group_id, image_id, concept, split and modality. Point it at a
# `pilot_subset.json` to fold the subset in as an extra identity cross-check.
SUBSET_PATH = ""

# None takes the layer the original representational report was computed at.
SELECTED_LAYER = None

# The amended rubric. A GO after dedup requires every concept in the run to
# have passed behaviorally, not the pilot's minimum of two.
REQUIRED_BEHAVIORAL_CONCEPTS = 4
MIN_DISTINCT_IMAGES = 2

from jlens.mmpilot.image_audit import ARTIFACTS, VerdictConfig

# Test hooks. A run directory and a fingerprint may be redirected from the
# environment so the CPU path can be exercised against a MOCK run.
COMPLETED_RUN_DIR = os.environ.get("MMPILOT_AUDIT_RUN_DIR") or COMPLETED_RUN_DIR
if "MMPILOT_AUDIT_EXPECT_FINGERPRINT" in os.environ:
    EXPECTED_RUN_FINGERPRINT = os.environ["MMPILOT_AUDIT_EXPECT_FINGERPRINT"]

VERDICT_CONFIG = VerdictConfig(
    required_behavioral_concepts=REQUIRED_BEHAVIORAL_CONCEPTS,
    min_distinct_images=MIN_DISTINCT_IMAGES,
)

print(f"RUN_IMAGE_INDEPENDENCE_AUDIT = {RUN_IMAGE_INDEPENDENCE_AUDIT}")
print(f"run directory   {COMPLETED_RUN_DIR}")
print(f"expected run fp {EXPECTED_RUN_FINGERPRINT or '(check disabled)'}")
print(f"layer           {SELECTED_LAYER if SELECTED_LAYER is not None else 'from the original report'}")
print(f"verdict config  {VERDICT_CONFIG.to_dict()}")
print("\\nartifacts this audit will write, relative to the run directory:")
for _name, _relative in ARTIFACTS.items():
    print(f"  {_name:16s} {_relative}")
print(
    "\\nno model, no processor, no Hugging Face token and no GPU are used at "
    "any point in this notebook."
)
'''
)

markdown(
    """
## 2. Mount Google Drive

Read-only. This cell never creates, moves, or deletes anything inside the run
directory — the audit's own outputs are written in section 3, and only to new
paths.
"""
)

code(
    '''
# 2. Mount Drive and check the run directory is readable. Nothing is written.
if RUN_IMAGE_INDEPENDENCE_AUDIT and IN_COLAB:
    from google.colab import drive

    drive.mount("/content/drive", force_remount=False)

RUN_PATH = Path(COMPLETED_RUN_DIR)
if not RUN_IMAGE_INDEPENDENCE_AUDIT:
    print("skipped: RUN_IMAGE_INDEPENDENCE_AUDIT is False")
elif not RUN_PATH.is_dir():
    raise RuntimeError(
        f"{RUN_PATH} is not a directory. Point COMPLETED_RUN_DIR at the "
        "finished run you mean to audit; nothing is discovered automatically."
    )
else:
    _expected = ["fingerprint.json", "summary.json", "report.md", "units"]
    _missing = [name for name in _expected if not (RUN_PATH / name).exists()]
    if _missing:
        raise RuntimeError(
            f"{RUN_PATH} is missing {_missing}. That is not a completed pilot "
            "run directory."
        )
    print(f"run directory OK: {RUN_PATH}")
    for _name in _expected:
        print(f"  {_name}")
'''
)

markdown(
    """
## 3. Run the audit (CPU only)

Resumable and fingerprinted. The audit fingerprint binds the original run
fingerprint, the subset checksum, the expanded-manifest checksum, the lens
checksum, the selected layer, and the version of every rule applied — image
identity, representational exclusion, causal aggregation, and the verdict
configuration. A rerun under a different fingerprint is **refused**, never
merged: mixing two rules would report a number nobody computed.

Expected runtime: seconds to a couple of minutes. It is bounded by reading a
few hundred small JSON files off Drive, not by computation.
"""
)

code(
    '''
# 3. CPU ONLY. Resolve image identity, audit the dependence, recompute the
# image-disjoint representational metrics, re-aggregate the saved interventions
# at the image level, and write the versioned artifacts.
#
# No model is loaded. No activation, code, direction or intervention is
# recomputed — every number here comes from artifacts already on disk.
AUDIT = None
if not RUN_IMAGE_INDEPENDENCE_AUDIT:
    print("skipped: RUN_IMAGE_INDEPENDENCE_AUDIT is False")
else:
    from jlens.mmpilot.image_audit import load_run, run_image_independence_audit

    if EXPECTED_RUN_FINGERPRINT:
        _loaded = load_run(
            RUN_PATH,
            subset_path=SUBSET_PATH or None,
            layer=SELECTED_LAYER,
        )
        if _loaded.run_fingerprint.digest != EXPECTED_RUN_FINGERPRINT:
            raise RuntimeError(
                f"{RUN_PATH} has run fingerprint {_loaded.run_fingerprint.digest}, "
                f"not the pinned {EXPECTED_RUN_FINGERPRINT}.\\n"
                "Refusing to audit a run other than the one named.\\n"
                "Set EXPECTED_RUN_FINGERPRINT = \\"\\" only if you are sure."
            )
        print(f"run fingerprint verified: {_loaded.run_fingerprint.digest}")

    AUDIT = run_image_independence_audit(
        RUN_PATH,
        subset_path=SUBSET_PATH or None,
        layer=SELECTED_LAYER,
        config=VERDICT_CONFIG,
    )
    _resume = AUDIT["resume"]
    print(f"\\naudit state: {_resume['status']}")
    print(f"  fingerprint: {_resume['fingerprint_digest']}")
    print(f"  reused:      {_resume['reused'] or 'none'}")
    print(f"  computed:    {_resume['computed'] or 'none'}")
    if _resume["invalid_artifacts"]:
        print(f"  invalid (recomputed): {_resume['invalid_artifacts']}")
    print(f"  model loaded: {AUDIT['model_loaded']}")
'''
)

markdown(
    """
## 4. The amended verdict and the artifacts

Read the dependence numbers before the verdict. If `n_distinct_images` is far
below `n_groups`, the run's apparent sample size was partly one photograph
counted more than once, and the image-level column is the one that means
something.

**What this audit does not settle.**

- The validated layer is late in the decoder. A final-prompt-token edit there
  cannot establish that an effect precedes answer-language convergence, so no
  claim of transfer before convergence is made here, corrected or not.
- Interventions add and subtract a direction on the residual stream. That is
  **not erasure** and not **projection ablation**, and nothing in the corrected
  numbers should be described as either.
- This audit re-reads a completed run. It corrects how that run's evidence is
  counted; it cannot repair how the run selected its targets in the first
  place. Where a cell rests on one photograph, the honest reading is one
  observation — not a smaller effect.
- Distinct captions of one image stay visible per cell as descriptive detail.
  They are never counted as independent image observations.
"""
)

code(
    '''
# 4. Print the amended verdict, what it rests on, and where everything went.
if AUDIT is None:
    print("=" * 72)
    print("NOTHING RAN — this is the committed default.")
    print("=" * 72)
    print("To audit the completed run, set in section 1:")
    print()
    print("    RUN_IMAGE_INDEPENDENCE_AUDIT = True")
    print()
    print(f"and confirm COMPLETED_RUN_DIR points at the run you mean:")
    print(f"    {COMPLETED_RUN_DIR}")
    print()
    print("Then run every cell. Expected runtime: seconds to a couple of")
    print("minutes. No GPU, no Hugging Face token, and no model are needed.")
else:
    _audit = AUDIT["audit"] or {}
    print("=" * 72)
    print("SAME-IMAGE DEPENDENCE")
    print("=" * 72)
    print(f"  synchronized groups            {_audit.get('n_groups')}")
    print(f"  distinct images                {_audit.get('n_distinct_images')}")
    print(f"  images entering as >1 group    {_audit.get('n_images_with_multiple_groups')}")
    print(f"  concepts affected              {_audit.get('concepts_affected')}")
    print(f"  train/test image overlap       {_audit.get('train_test_image_overlap') or 'none'}")
    print(f"  sibling groups crossing splits {len(_audit.get('sibling_groups_crossing_splits') or [])}")
    print(f"  hard failures                  {_audit.get('hard_failures') or 'none'}")

    if AUDIT["ok"]:
        print("\\n" + "=" * 72)
        print("CORRECTED REPRESENTATION (image-disjoint)")
        print("=" * 72)
        for _pair, _entry in sorted(AUDIT["representational"]["pairs"].items()):
            _exclusions = _entry["exclusions"]
            print(
                f"  {_pair:24s} queries={_entry['jspace_retrieval']['n_queries']:3d}  "
                f"top1={_entry['jspace_retrieval']['top1_accuracy']:.3f}  "
                f"mrr={_entry['jspace_retrieval']['mrr']:.3f}  "
                f"shuffled_p95={_entry['shuffled_control']['p95_top1_accuracy']:.3f}"
            )
            print(
                f"    excluded: same group={_exclusions['n_excluded_same_group']}  "
                f"additionally same image={_exclusions['n_excluded_same_image_different_group']}  "
                f"eligible targets min/med/max="
                f"{_exclusions['eligible_targets']['min']}/"
                f"{_exclusions['eligible_targets']['median']}/"
                f"{_exclusions['eligible_targets']['max']}"
            )

        print("\\n" + "=" * 72)
        print("CORRECTED CAUSATION (image is the independent unit)")
        print("=" * 72)
        for _row in AUDIT["interventions_image_level"]["rows"]:
            if not _row["off_diagonal"] or _row["control_kind"] != "source_concept":
                continue
            _group = (_row.get("group_level") or {}).get("mean_signed_target_effect")
            print(
                f"  {_row['concept']:10s} {_row['pair']:14s} a={_row['alpha']:<5g} "
                f"groups={_row['n_groups']:2d} images={_row['n_distinct_images']:2d} "
                f"(+{_row['n_positive_images']}/-{_row['n_negative_images']})  "
                f"image={_row['mean_signed_target_effect']:+.4f}  "
                f"group={'n/a' if _group is None else format(_group, '+.4f')}  "
                f"sign={_row['fraction_expected_sign']:.2f}"
            )
        _divergence = AUDIT["divergence"]
        print(
            f"\\n  cells pseudoreplicated at group level: "
            f"{_divergence['n_rows_pseudoreplicated_at_group_level']} of {_divergence['n_rows']}"
        )
        print(f"  max |image - group| effect: {_divergence['max_abs_divergence']:.4f}")

        _replication = AUDIT["replication"]
        print("\\n" + "=" * 72)
        print("REPLICATION")
        print("=" * 72)
        print(f"  by concept:   {_replication['by_concept']}")
        print(f"  by direction: {_replication['by_direction']}")

    print("\\n" + "=" * 72)
    print(f"AMENDED VERDICT: {AUDIT['verdict']}")
    print("=" * 72)
    print(f"  original recommendation (unchanged on disk): {AUDIT['original_recommendation']}")
    print(f"  {AUDIT['rationale']}")
    if AUDIT["ok"]:
        for _name, _status in AUDIT["summary"]["verdict"]["criteria_status"].items():
            print(f"    {_status:15s} {_name}")
        print(f"\\n  {AUDIT['summary']['verdict']['late_layer_limitation']}")

    print("\\n" + "=" * 72)
    print("ARTIFACTS WRITTEN")
    print("=" * 72)
    for _name, _path in AUDIT["artifacts"].items():
        _checksum = AUDIT["artifact_checksums"].get(
            str(Path(_path).relative_to(RUN_PATH)).replace("\\\\", "/")
        )
        print(f"  {_name:16s} {_path}")
        if _checksum:
            print(f"                   {_checksum}")

    print("\\n" + "=" * 72)
    print("PRESERVATION — the original artifacts were read, never written")
    print("=" * 72)
    for _name, _unchanged in sorted(AUDIT["preservation"]["unchanged"].items()):
        print(f"  {'unchanged' if _unchanged else 'CHANGED':10s} {_name}")
    print(f"\\n  all originals unchanged: {AUDIT['preservation']['all_unchanged']}")
    print(f"  model loaded:            {AUDIT['model_loaded']}")
'''
)


def build() -> dict:
    return {
        "cells": [
            {
                "cell_type": kind,
                "metadata": {},
                "source": (
                    text.splitlines(keepends=True)
                    if kind == "markdown"
                    else text.splitlines(keepends=True)
                ),
                **({"execution_count": None, "outputs": []} if kind == "code" else {}),
            }
            for kind, text in CELLS
        ],
        "metadata": {
            "accelerator": "None",
            "colab": {"provenance": [], "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


if __name__ == "__main__":
    TARGET.write_text(
        json.dumps(build(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {TARGET}")
