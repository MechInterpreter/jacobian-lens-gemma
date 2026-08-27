# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the no-refit country evidence seal."""

from __future__ import annotations

import json

import pytest
import torch

from jlens.mmpilot.backend import BuiltInputs
from jlens.mmpilot.coordinate_swap import (
    ConceptToken,
    build_swap_basis_from_vectors,
    coordinate_swap_band,
)
from jlens.mmpilot.country_activation_patch import activation_patch_band
from jlens.mmpilot.country_evidence_seal import (
    CountryEvidenceSealRefused,
    audit_unopened_confirmation_outputs,
    evidence_positions,
    matched_scaffold_confirmation_report,
    matched_scaffold_report,
    seal_evidence_attention,
    sealed_development_report,
)


class _MaskBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seen: list[torch.Tensor] = []
        self.config = type("Config", (), {"_attn_implementation": "sdpa"})()
        self.self_attn = type(
            "Attention",
            (),
            {"is_causal": True, "is_sliding": False, "sliding_window": None},
        )()

    def forward(self, hidden_states, *, attention_mask):
        self.seen.append(attention_mask.detach().clone())
        return hidden_states


def _inputs(ids, *, span=None) -> BuiltInputs:
    return BuiltInputs(
        tensors={"input_ids": torch.tensor([ids])},
        prompt_len=len(ids),
        modality="image" if span else "text",
        modality_token_range=span,
    )


def test_evidence_positions_use_full_soft_span_or_unique_text_token() -> None:
    assert evidence_positions(_inputs([1, 2, 3, 4], span=[1, 3])) == (1, 2)
    assert evidence_positions(_inputs([1, 9, 3]), country_token_id=9) == (1,)
    with pytest.raises(CountryEvidenceSealRefused, match="occurs 2 times"):
        evidence_positions(_inputs([9, 2, 9]), country_token_id=9)


def test_seal_allows_encoding_then_blocks_rereading_and_generation() -> None:
    blocks = torch.nn.ModuleList([_MaskBlock() for _ in range(4)])
    base_prompt_mask = torch.zeros((1, 1, 4, 4), dtype=torch.float32)
    base_generation_mask = torch.zeros((1, 1, 5, 5), dtype=torch.float32)
    prompt_hidden = torch.zeros((1, 4, 2))
    generation_hidden = torch.zeros((1, 5, 2))
    floor = torch.finfo(torch.float32).min

    with seal_evidence_attention(
        blocks,
        evidence_token_positions=(1, 2),
        prompt_len=4,
        bottleneck_layer=2,
    ) as stats:
        for block in blocks:
            block(prompt_hidden, attention_mask=base_prompt_mask)
        for block in blocks:
            block(generation_hidden, attention_mask=base_generation_mask)

    # Before L2, the final prompt token can still encode evidence.
    assert torch.equal(blocks[0].seen[0], base_prompt_mask)
    # Generated tokens can never directly reread evidence.
    assert blocks[0].seen[1][0, 0, 4, 1].item() == floor
    assert blocks[0].seen[1][0, 0, 4, 2].item() == floor
    assert blocks[0].seen[1][0, 0, 4, 0].item() == floor
    assert blocks[0].seen[1][0, 0, 4, 3].item() == 0.0
    # At and after L2, final-prompt and generated queries are both sealed.
    assert blocks[2].seen[0][0, 0, 3, 1].item() == floor
    assert blocks[2].seen[1][0, 0, 3, 2].item() == floor
    assert blocks[2].seen[1][0, 0, 4, 2].item() == floor
    assert all(row["n_forward_passes"] == 2 for row in stats.values())
    assert all(not block._forward_pre_hooks for block in blocks)


def test_seal_materializes_sdpa_causal_mask_when_transformers_omits_it() -> None:
    blocks = torch.nn.ModuleList([_MaskBlock() for _ in range(2)])
    hidden = torch.zeros((1, 4, 2))
    with seal_evidence_attention(
        blocks,
        evidence_token_positions=(1,),
        prompt_len=4,
        bottleneck_layer=1,
    ) as stats:
        for block in blocks:
            block(hidden, attention_mask=None)
    early, late = blocks[0].seen[0], blocks[1].seen[0]
    assert early.dtype == torch.bool
    assert early[0, 0, 3, 1].item() is True
    assert late[0, 0, 3, 1].item() is False
    assert late[0, 0, 3, 0].item() is False
    assert late[0, 0, 3, 3].item() is True
    assert late[0, 0, 0, 3].item() is False
    assert all(row["n_base_masks_materialized"] == 1 for row in stats.values())


def test_matched_scaffold_hook_runs_before_coordinate_exchange() -> None:
    block = torch.nn.Identity()
    blocks = torch.nn.ModuleList([block])
    source = ConceptToken("France", 1, " France", " {}")
    target = ConceptToken("China", 2, " China", " {}")
    basis = build_swap_basis_from_vectors(
        torch.tensor([1.0, 0.0]),
        torch.tensor([0.0, 1.0]),
        layer=0,
        source=source,
        target=target,
    )
    hidden = torch.tensor([[[0.0, 0.0], [9.0, 9.0]]])
    with activation_patch_band(
        blocks,
        {0: torch.tensor([1.0, 0.0])},
        source_position=1,
        prompt_len=2,
    ):
        with coordinate_swap_band(
            blocks,
            {0: basis},
            alpha=1.0,
            prompt_len=2,
            position_rule="evidence_span_only",
            evidence_span=[1, 2],
            require_frozen=False,
        ):
            output = block(hidden)
    assert output[0, 1].tolist() == pytest.approx([0.0, 1.0], abs=1e-6)


def _rows(conditions, *, exact_success=True):
    rows = []
    for property_name in ("capital", "continent"):
        for modality in ("text", "image", "spoken_audio"):
            for _ in range(3):
                for condition in conditions:
                    success = condition in ("clean_sealed", "target_state")
                    if condition == "exact":
                        success = exact_success
                    rows.append({
                        "property": property_name,
                        "modality": modality,
                        "condition": condition,
                        "success": success,
                        "integrity_pass": True,
                    })
    return rows


def test_sealed_report_separates_state_and_jlens_claims() -> None:
    state = _rows(("clean_sealed", "target_state", "unrelated_state"))
    coordinate = _rows(("exact", "zero", "random", "unrelated"))
    report = sealed_development_report(
        state,
        coordinate,
        expected_n=3,
        properties=("capital", "continent"),
        modalities=("text", "image", "spoken_audio"),
        band=tuple(range(24, 32)),
    )
    assert report["verdict"] == "COUNTRY_SEALED_EVIDENCE_BOTH_GO"
    assert report["fitting_performed"] is False
    assert report["backward_passes"] == 0
    assert report["fresh_confirmation_opened"] is False


def test_sealed_report_can_return_state_only() -> None:
    state = _rows(("clean_sealed", "target_state", "unrelated_state"))
    coordinate = _rows(
        ("exact", "zero", "random", "unrelated"), exact_success=False
    )
    report = sealed_development_report(
        state,
        coordinate,
        expected_n=3,
        properties=("capital", "continent"),
        modalities=("text", "image", "spoken_audio"),
        band=tuple(range(24, 32)),
    )
    assert report["verdict"] == "COUNTRY_SEALED_EVIDENCE_STATE_ONLY_GO"


def test_matched_scaffold_report_requires_source_capability_and_exact_effect() -> None:
    state = _rows(("self_scaffold", "target_state", "unrelated_state"))
    for row in state:
        row["success"] = row["condition"] in ("self_scaffold", "target_state")
    coordinate = _rows(("exact", "zero", "random", "unrelated"))
    report = matched_scaffold_report(
        state,
        coordinate,
        expected_n=3,
        properties=("capital", "continent"),
        modalities=("text", "image", "spoken_audio"),
        band=tuple(range(24, 32)),
    )
    assert report["verdict"] == "COUNTRY_MATCHED_SCAFFOLD_BOTH_GO"
    assert report["only_coordinate_condition_varies_after_source_scaffold"] is True
    assert report["fitting_performed"] is False
    assert report["fresh_confirmation_opened"] is False


def test_freshness_audit_detects_only_generated_confirmation_outputs(tmp_path) -> None:
    (tmp_path / "design.json").write_text(
        '{"unit_id":"fresh-1","selected":true}', encoding="utf-8"
    )
    clean = audit_unopened_confirmation_outputs(tmp_path, ("fresh-1",))
    assert clean["fresh"] is True
    (tmp_path / "unit.json").write_text(
        '{"unit_id":"fresh-1","generated_text":"Asia"}', encoding="utf-8"
    )
    spent = audit_unopened_confirmation_outputs(tmp_path, ("fresh-1",))
    assert spent["fresh"] is False
    assert spent["findings"][0]["unit_id"] == "fresh-1"


def test_freshness_audit_uses_bounded_fingerprint_first_scan(tmp_path) -> None:
    candidate = tmp_path / "run-1" / "diagnostics" / "confirmation"
    candidate.mkdir(parents=True)
    (candidate / "fingerprint.json").write_text(
        '{"manifest_checksum":"population-1","split_id":"other"}',
        encoding="utf-8",
    )
    units = candidate / "units" / "intervention"
    units.mkdir(parents=True)
    (units / "one.json").write_text(
        '{"payload":{"unit_id":"fresh-1","generated_text":"Asia"}}',
        encoding="utf-8",
    )
    irrelevant = tmp_path / "unbounded" / "deep" / "tree"
    irrelevant.mkdir(parents=True)
    (irrelevant / "hidden.json").write_text(
        '{"unit_id":"fresh-1","generated_text":"ignored"}', encoding="utf-8"
    )
    report = audit_unopened_confirmation_outputs(
        tmp_path,
        ("fresh-1",),
        manifest_checksum="population-1",
        split_id="confirmation-1",
    )
    assert report["fresh"] is False
    assert report["audit_strategy"] == "bounded_fingerprint_first_v1"
    assert report["fingerprints_scanned"] == 1
    assert report["candidate_json_files"] == 1


def test_freshness_audit_surfaces_a_store_under_a_different_population(
    tmp_path,
) -> None:
    """A run with a different manifest_checksum is excluded, not hidden."""
    matching = tmp_path / "run-1"
    matching.mkdir()
    (matching / "fingerprint.json").write_text(
        '{"manifest_checksum":"population-1","split_id":"other"}',
        encoding="utf-8",
    )
    (matching / "report.json").write_text(
        '{"unit_id":"unrelated","generated_text":"x"}', encoding="utf-8"
    )
    stale = tmp_path / "run-2"
    stale.mkdir()
    (stale / "fingerprint.json").write_text(
        '{"manifest_checksum":"population-0-stale","split_id":"stale-split"}',
        encoding="utf-8",
    )
    report = audit_unopened_confirmation_outputs(
        tmp_path,
        ("fresh-1",),
        manifest_checksum="population-1",
        split_id="confirmation-1",
    )
    assert report["fresh"] is True
    assert report["fingerprints_scanned"] == 2
    assert len(report["unmatched_fingerprints"]) == 1
    assert report["unmatched_fingerprints"][0]["manifest_checksum"] == (
        "population-0-stale"
    )
    assert report["distinct_unmatched_manifest_checksums"] == [
        "population-0-stale"
    ]


def test_freshness_audit_scans_bytes_without_a_full_text_decode(tmp_path) -> None:
    """A large non-matching candidate is read once, in bytes, and cleared."""
    candidate = tmp_path / "run-1"
    candidate.mkdir()
    (candidate / "fingerprint.json").write_text(
        '{"manifest_checksum":"population-1","split_id":"other"}',
        encoding="utf-8",
    )
    padding = json.dumps({"note": "x" * 2_000_000, "unit_id": "not-a-match"})
    (candidate / "big_report.json").write_text(padding, encoding="utf-8")
    report = audit_unopened_confirmation_outputs(
        tmp_path,
        ("fresh-1",),
        manifest_checksum="population-1",
        split_id="confirmation-1",
    )
    assert report["fresh"] is True
    assert report["matching_json_files_read"] == 0
    assert report["candidate_bytes_read"] >= len(padding)
    assert report["largest_candidate_file_bytes"] >= len(padding)


def test_freshness_audit_reports_progress_through_a_callback(tmp_path) -> None:
    """A caller can print live progress instead of waiting on a silent call."""
    candidate = tmp_path / "run-1"
    candidate.mkdir()
    (candidate / "fingerprint.json").write_text(
        '{"manifest_checksum":"population-1","split_id":"other"}',
        encoding="utf-8",
    )
    (candidate / "one.json").write_text(
        '{"unit_id":"fresh-1","generated_text":"Asia"}', encoding="utf-8"
    )
    (candidate / "two.json").write_text(
        '{"unit_id":"unrelated"}', encoding="utf-8"
    )
    calls = []
    audit_unopened_confirmation_outputs(
        tmp_path,
        ("fresh-1",),
        manifest_checksum="population-1",
        split_id="confirmation-1",
        on_candidate_scanned=lambda index, total, path, size: calls.append(
            (index, total, path.name, size)
        ),
    )
    assert len(calls) == 2
    assert {call[2] for call in calls} == {"one.json", "two.json"}
    assert all(call[1] == 2 for call in calls)
    assert {call[0] for call in calls} == {1, 2}


def test_fresh_confirmation_report_accepts_pooled_cross_modal_effect() -> None:
    state = []
    coordinate = []
    for modality in ("image", "spoken_audio", "text"):
        for index in range(14):
            for condition in ("self_scaffold", "target_state", "unrelated_state"):
                state.append({
                    "unit_id": f"{modality}-{index}",
                    "modality": modality,
                    "condition": condition,
                    "success": condition != "unrelated_state",
                    "integrity_pass": True,
                })
            for condition in ("exact", "zero", "random", "unrelated"):
                coordinate.append({
                    "unit_id": f"{modality}-{index}",
                    "modality": modality,
                    "condition": condition,
                    "success": condition == "exact" and (
                        modality == "text" or index < 10
                    ),
                    "integrity_pass": True,
                })
    report = matched_scaffold_confirmation_report(
        state, coordinate, expected_n=14
    )
    assert report["verdict"] == "COUNTRY_MATCHED_SCAFFOLD_FRESH_CONFIRMATION_GO"
    assert report["pooled_primary"]["exact"]["successes"] == 20
    assert report["familywise_statistics_gate_passed"] is True
