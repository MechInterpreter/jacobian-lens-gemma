# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Regenerate the committed endpoint-semantics audit artifacts.

CPU only. Imports no model and reads no run directory: the audit is a statement
about *this repository's source*, so it is fully determined by the source and by
the provenance pins that are currently set.

    python scripts/write_endpoint_audit.py

Writes ``reports/endpoint_audit/``:

* ``endpoint_semantics_audit.json`` — the classification and the source scan;
* ``endpoint_semantics_audit.md`` — the readable half;
* ``endpoint_claim_ledger.json`` — the machine-readable claim ledger;
* ``endpoint_amendments/*.json`` / ``*.md`` — the two versioned amendments, in
  their reviewable form: bound to the pinned report checksums and run
  fingerprints, with a fixed timestamp so the committed copy is deterministic.

The amendments written **here** are the reviewable copies. The ones that bind to
a completed run's own units are written by the notebook's stage 0 into a new
audit run directory beside that run — never into the completed run itself.

``tests/test_endpoint_audit.py`` regenerates everything and fails if a committed
file has drifted, so the audit can never quietly go stale against the code it
describes. Exits non-zero if the audit itself fails — that is, if the active
package describes a restricted-candidate result as an unrestricted one.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jlens.mmpilot.endpoint_amend import (  # noqa: E402
    AMENDMENT_DIRNAME,
    BAND_FOLLOWUP_TERMINOLOGY,
    THREE_MODALITY_TERMINOLOGY,
    amendment_filename,
    build_endpoint_amendment,
    render_amendment_markdown,
)
from jlens.mmpilot.endpoint_audit import (  # noqa: E402
    endpoint_audit_files,
    endpoint_audit_record,
)
from jlens.mmpilot.full_vocab_study import (  # noqa: E402
    BAND_FOLLOWUP_FINGERPRINT_PIN,
    BAND_FOLLOWUP_REPORT_CHECKSUM_PIN,
    BAND_FOLLOWUP_RUN_NAME,
    CANONICAL_AUDIO_GENERATION_FINGERPRINT_PIN,
    CANONICAL_AUDIO_REPORT_CHECKSUM_PIN,
    CANONICAL_AUDIO_RUN_NAME,
    CONTROLLED_TARGET_LOGPROB_EFFECT,
    FULL_VOCAB_NOT_EVALUATED,
    REQUIRED_PINS,
    RESTRICTED_CANDIDATE_PREFERENCE_GO,
)
from jlens.mmpilot.full_vocabulary import (  # noqa: E402
    ENDPOINT_CONDITIONAL_LOGPROB,
    ENDPOINT_RESTRICTED_CANDIDATE,
)

TARGET_DIR = ROOT / "reports" / "endpoint_audit"

#: Fixed so the committed amendments are byte-deterministic. The notebook's
#: stage 0 stamps the real time when it writes into an audit run directory.
REVIEW_COPY_UTC = "2026-08-13T00:00:00+00:00"


def _amendments(record: dict) -> dict[str, dict]:
    """The two amendments, in their reviewable, fixed-timestamp form."""
    band = build_endpoint_amendment(
        name="l33_l40_validated_band_followup_report",
        study="L33-L40 validated-band follow-up",
        original_report_path=(
            f"runs/mmband33/{BAND_FOLLOWUP_RUN_NAME}/"
            "l33_l40_validated_band_followup_report.json"
        ),
        original_report_checksum=BAND_FOLLOWUP_REPORT_CHECKSUM_PIN,
        original_run_name=BAND_FOLLOWUP_RUN_NAME,
        original_run_fingerprint=BAND_FOLLOWUP_FINGERPRINT_PIN,
        original_verdict="L33_L40_VALIDATED_BAND_FOLLOWUP_ALPHA2_SENSITIVITY_ONLY",
        original_endpoint_class=ENDPOINT_RESTRICTED_CANDIDATE,
        endpoint_audit_digest=record["audit_digest"],
        claim_ledger_digest=record["claim_ledger_digest"],
        terminology=BAND_FOLLOWUP_TERMINOLOGY,
        corrected_labels=[
            RESTRICTED_CANDIDATE_PREFERENCE_GO,
            FULL_VOCAB_NOT_EVALUATED,
        ],
        written_utc=REVIEW_COPY_UTC,
    )
    audio = build_endpoint_amendment(
        name="native_audio_transfer_three_modality_verdict",
        study="native spoken-audio transfer (canonical three-modality run)",
        original_report_path=(
            f"runs/mmaudio/{CANONICAL_AUDIO_RUN_NAME}/"
            "native_audio_transfer_summary.json"
        ),
        # The canonical run's *report* checksum pin is deliberately empty, so
        # this binds to the raw-generation fingerprint, which is recorded in
        # docs/three_modality_claim_admissibility.md.
        original_report_checksum=(
            CANONICAL_AUDIO_REPORT_CHECKSUM_PIN
            or CANONICAL_AUDIO_GENERATION_FINGERPRINT_PIN
        ),
        original_run_name=CANONICAL_AUDIO_RUN_NAME,
        original_run_fingerprint=CANONICAL_AUDIO_GENERATION_FINGERPRINT_PIN,
        original_verdict="THREE_MODALITY_GO",
        original_endpoint_class=ENDPOINT_CONDITIONAL_LOGPROB,
        endpoint_audit_digest=record["audit_digest"],
        claim_ledger_digest=record["claim_ledger_digest"],
        terminology=THREE_MODALITY_TERMINOLOGY,
        corrected_labels=[CONTROLLED_TARGET_LOGPROB_EFFECT, FULL_VOCAB_NOT_EVALUATED],
        written_utc=REVIEW_COPY_UTC,
    )
    return {row["name"]: row for row in (band, audio)}


def build() -> tuple[dict, dict[str, str]]:
    """The audit record and every file it renders to, by relative path."""
    import json

    record = endpoint_audit_record(
        repo_root=ROOT,
        report_checksums={
            name: row["value"] for name, row in REQUIRED_PINS.items() if row["value"]
        },
    )
    files = dict(endpoint_audit_files(record))
    for name, amendment in _amendments(record).items():
        stem = f"{AMENDMENT_DIRNAME}/{amendment_filename(name)}"
        files[f"{stem}.json"] = (
            json.dumps(amendment, indent=2, sort_keys=True) + "\n"
        )
        files[f"{stem}.md"] = render_amendment_markdown(amendment)
    return record, files


def main() -> int:
    record, files = build()
    for name, text in files.items():
        path = TARGET_DIR / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
        print(f"wrote {path}")
    print()
    print("audit digest       ", record["audit_digest"])
    print("claim ledger digest", record["claim_ledger_digest"])
    print("endpoints audited  ", record["n_endpoints_audited"])
    print("unqualified overclaims", record["source_scan"]["n_overclaims"])
    print("passed             ", record["passed"])
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
