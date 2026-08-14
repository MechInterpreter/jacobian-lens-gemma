"""The committed endpoint-semantics audit artifacts stay current with the code.

An audit that describes a repository it no longer matches is worse than no
audit, so the committed artifacts are regenerated here and compared. If this
fails, run::

    python scripts/write_endpoint_audit.py

and commit the result — after checking that the change is the one you meant.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from jlens.mmpilot.endpoint_audit import (
    AUDIT_JSON_NAME,
    AUDIT_MARKDOWN_NAME,
    ENDPOINT_CLASSES,
    LEDGER_JSON_NAME,
    SURVIVAL_CLASSES,
    AuditedEndpoint,
    EndpointAuditFailed,
    claim_ledger,
    scan_active_sources,
    scanned_modules,
)
from jlens.mmpilot.full_vocabulary import (
    ENDPOINT_CONDITIONAL_LOGPROB,
    ENDPOINT_RESTRICTED_CANDIDATE,
    ENDPOINT_UNRESTRICTED_NEXT_TOKEN,
)

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "reports" / "endpoint_audit"
WRITER = ROOT / "scripts" / "write_endpoint_audit.py"


@pytest.fixture(scope="module")
def generated() -> dict[str, str]:
    spec = importlib.util.spec_from_file_location("_endpoint_audit_writer", WRITER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _, files = module.build()
    return files


def test_committed_artifacts_are_current(generated):
    """Every committed file, including the two reviewable amendments."""
    for name in (AUDIT_JSON_NAME, AUDIT_MARKDOWN_NAME, LEDGER_JSON_NAME):
        assert name in generated
    assert sum(1 for name in generated if name.endswith(".json")) >= 4
    for name, text in generated.items():
        path = ARTIFACTS / name
        assert path.exists(), f"missing {path}; run scripts/write_endpoint_audit.py"
        assert path.read_text(encoding="utf-8") == text, (
            f"{name} has drifted from the code it describes; "
            "run scripts/write_endpoint_audit.py"
        )


def test_no_stray_file_sits_beside_the_committed_artifacts(generated):
    on_disk = {
        path.relative_to(ARTIFACTS).as_posix()
        for path in ARTIFACTS.rglob("*")
        if path.is_file()
    }
    assert on_disk == set(generated)


def test_the_committed_amendments_recompute_nothing():
    for path in (ARTIFACTS / "endpoint_amendments").glob("*.json"):
        amendment = json.loads(path.read_text(encoding="utf-8"))
        assert amendment["scientific_recompute"] == 0
        assert amendment["scientific_numbers_unchanged"] is True
        assert amendment["original_report_modified"] is False
        assert amendment["original_units_modified"] is False
        assert amendment["verdict_changed_by_prose"] is False
        assert amendment["full_vocabulary_status"] == "FULL_VOCABULARY_NOT_EVALUATED"
        assert amendment["original_verdict_is_reproduced_verbatim"] is True


def test_the_committed_audit_passes():
    record = json.loads((ARTIFACTS / AUDIT_JSON_NAME).read_text(encoding="utf-8"))
    assert record["passed"] is True
    assert record["source_scan"]["n_overclaims"] == 0
    assert record["scientific_recompute"] == 0
    assert record["completed_reports_untouched"] is True


def test_the_committed_ledger_traces_every_claim():
    ledger = json.loads((ARTIFACTS / LEDGER_JSON_NAME).read_text(encoding="utf-8"))
    assert ledger["n_claims"] == len(ledger["rows"])
    for row in ledger["rows"]:
        assert row["computing_function"].count(".") >= 2
        assert row["endpoint_class"] in ENDPOINT_CLASSES
        assert row["survival"] in SURVIVAL_CLASSES
        assert row["candidate_universe"]
        assert row["justified_interpretation"]
        assert isinstance(row["tokens_appended"], bool)
        assert isinstance(row["global_vocabulary_considered"], bool)
        assert isinstance(row["generation_occurred"], bool)


def test_every_restricted_claim_names_prohibited_wording():
    ledger = json.loads((ARTIFACTS / LEDGER_JSON_NAME).read_text(encoding="utf-8"))
    for row in ledger["rows"]:
        if row["endpoint_class"] in (
            ENDPOINT_RESTRICTED_CANDIDATE,
            ENDPOINT_CONDITIONAL_LOGPROB,
        ):
            assert row["prohibited_wording"], row["claim"]
            assert row["survival"] != "survives_unchanged" or row["notes"]


def test_unrestricted_claims_survive_unchanged():
    ledger = json.loads((ARTIFACTS / LEDGER_JSON_NAME).read_text(encoding="utf-8"))
    for row in ledger["rows"]:
        if row["endpoint_class"] == ENDPOINT_UNRESTRICTED_NEXT_TOKEN:
            assert row["survival"] == "survives_unchanged", row["claim"]
            assert row["requires_full_vocabulary_revalidation"] is False
            assert row["tokens_appended"] is False


def test_a_row_cannot_claim_an_endpoint_it_did_not_measure():
    with pytest.raises(ValueError, match="classified unrestricted"):
        AuditedEndpoint(
            claim="x",
            study="y",
            source_run=None,
            report_checksum_pin=None,
            module="jlens.mmpilot.full_vocabulary",
            function="score_unrestricted_next_token",
            report_fields=(),
            endpoint_class=ENDPOINT_UNRESTRICTED_NEXT_TOKEN,
            candidate_universe="two candidates",
            tokens_appended=True,
            global_vocabulary_considered=False,
            generation_occurred=False,
            justified_interpretation="none",
            prohibited_wording=(),
            requires_full_vocabulary_revalidation=False,
            survival="survives_unchanged",
        )
    with pytest.raises(ValueError, match="classified restricted-candidate"):
        AuditedEndpoint(
            claim="x",
            study="y",
            source_run=None,
            report_checksum_pin=None,
            module="jlens.mmpilot.capability",
            function="prediction_and_margin",
            report_fields=(),
            endpoint_class=ENDPOINT_RESTRICTED_CANDIDATE,
            candidate_universe="two candidates",
            tokens_appended=True,
            global_vocabulary_considered=True,
            generation_occurred=False,
            justified_interpretation="none",
            prohibited_wording=(),
            requires_full_vocabulary_revalidation=False,
            survival="survives_unchanged",
        )


def test_the_registry_refuses_a_function_that_does_not_exist():
    row = AuditedEndpoint(
        claim="x",
        study="y",
        source_run=None,
        report_checksum_pin=None,
        module="jlens.mmpilot.capability",
        function="a_function_that_was_deleted",
        report_fields=(),
        endpoint_class=ENDPOINT_RESTRICTED_CANDIDATE,
        candidate_universe="two candidates",
        tokens_appended=True,
        global_vocabulary_considered=False,
        generation_occurred=False,
        justified_interpretation="none",
        prohibited_wording=("x",),
        requires_full_vocabulary_revalidation=True,
        survival="survives_with_narrower_wording",
    )
    from jlens.mmpilot.endpoint_audit import verify_registry

    with pytest.raises(EndpointAuditFailed, match="does not exist"):
        verify_registry([row])


def test_the_scan_is_derived_from_the_registry_not_a_hand_list():
    modules = scanned_modules()
    assert "jlens.mmpilot.capability" in modules
    assert "jlens.mmpilot.band_swap" in modules
    assert "jlens.mmpilot.tri_modal" in modules
    # Modules that genuinely measure the full vocabulary are out of scope.
    assert "jlens.mmpilot.convergence" not in modules
    assert "jlens.calibration.gate" not in modules
    assert "jlens.native_readout" not in modules


def test_an_unqualified_overclaim_would_fail_the_scan(tmp_path):
    module_dir = tmp_path / "jlens" / "mmpilot"
    module_dir.mkdir(parents=True)
    (module_dir / "fake.py").write_text(
        '"""A docstring.\n\nThis reports the global top-1 over the whole thing.\n"""\n',
        encoding="utf-8",
    )
    scan = scan_active_sources(tmp_path, modules=("jlens.mmpilot.fake",))
    assert scan["passed"] is False
    assert scan["n_overclaims"] == 1
    assert scan["overclaims"][0]["pattern"] == "global_top1"


def test_a_qualified_mention_passes_the_scan(tmp_path):
    module_dir = tmp_path / "jlens" / "mmpilot"
    module_dir.mkdir(parents=True)
    (module_dir / "fake.py").write_text(
        '"""A docstring.\n\nThis is a restricted-candidate rate. It is NOT a '
        'global top-1.\n"""\n',
        encoding="utf-8",
    )
    scan = scan_active_sources(tmp_path, modules=("jlens.mmpilot.fake",))
    assert scan["passed"] is True
    assert scan["n_hits"] == 1
    assert scan["qualified_hits"][0]["classification"] == "qualified"


def test_a_missing_module_is_reported_not_silently_skipped(tmp_path):
    scan = scan_active_sources(tmp_path, modules=("jlens.mmpilot.nope",))
    assert scan["modules_not_found"] == ["jlens.mmpilot.nope"]


def test_the_ledger_digest_covers_the_configured_checksums():
    without = claim_ledger()
    with_pins = claim_ledger(
        report_checksums={"BAND_FOLLOWUP_REPORT_CHECKSUM": "sha256:" + "a" * 64}
    )
    assert without["ledger_digest"] != with_pins["ledger_digest"]
