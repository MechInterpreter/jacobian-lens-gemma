"""Generate the reproducible evidence-analysis notebook."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "mats_research_evidence_analysis.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(True),
    }


cells = [
    md("""# MATS research evidence analysis\n\nThis notebook reads archived JSON reports and generates application figures. It does **not** recompute scientific results, refit a lens, load Gemma, or inspect raw outcome units. The purpose is reproducible reporting: every plotted value comes from a named completed report."""),
    md("""## Evidence boundary\n\nThe default evidence root is `tmp/research_evidence_20260819`, populated from the archived Drive run ZIPs. Change `EVIDENCE_ROOT` only to another extraction of the same completed runs. Missing reports cause a hard failure rather than a partial chart."""),
    code("""from pathlib import Path\nimport sys\n\nREPO_ROOT = Path.cwd()\nif not (REPO_ROOT / 'scripts' / 'mats_research_evidence.py').is_file():\n    REPO_ROOT = REPO_ROOT.parent\nsys.path.insert(0, str(REPO_ROOT))\n\nEVIDENCE_ROOT = REPO_ROOT / 'tmp' / 'research_evidence_20260819'\nOUTPUT_DIR = REPO_ROOT / 'reports' / 'mats_application' / 'figures'\nprint('evidence root:', EVIDENCE_ROOT)\nprint('output dir:   ', OUTPUT_DIR)"""),
    md("""## Generate figures and a machine-readable evidence summary\n\nThe chart builder reads: corrected lens confirmation, tri-modal capability, independent L32 convergence, the L27–31 J-/R-lens verdict, and the final exact α=1 unrestricted-output confirmation."""),
    code("""from scripts.mats_research_evidence import build_figures\n\nSUMMARY = build_figures(EVIDENCE_ROOT, OUTPUT_DIR)\nSUMMARY['verified_results']"""),
    code("""from IPython.display import Image, display\n\nfor name, path in SUMMARY['figures'].items():\n    print(name, path)\n    display(Image(filename=path))"""),
    md("""## Interpretation guardrails\n\n- `AUDIO_CAPABILITY_GO` is a behavioral screen under a restricted six-candidate question.\n- The tri-modal causal result is a controlled target-log-probability effect; it is not unrestricted model output.\n- L32 was **AMBIGUOUS**, not converged and not demonstrably not-converged.\n- The validated contiguous band is L33–L40, selected after lens validation; its α=2 result is sensitivity evidence.\n- The final α=1 study used unrestricted greedy next-token output. Its hidden intermediate arm did not replicate strongly enough for the desired claim, while the bird→cat direct-answer control showed the intervention machinery could move the endpoint."""),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(OUT)
