# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the no-refit country evidence seal."""

from __future__ import annotations

import pytest
import torch

from jlens.mmpilot.backend import BuiltInputs
from jlens.mmpilot.country_evidence_seal import (
    CountryEvidenceSealRefused,
    evidence_positions,
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
