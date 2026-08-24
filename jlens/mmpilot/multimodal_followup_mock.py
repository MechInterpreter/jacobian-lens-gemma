# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""MOCK worlds for the multimodal follow-ups — the protocol, not the model.

Nothing here touches Gemma 4, a lens tensor, or a photograph. A scripted world
decides each trial's outcome from the scenario, the band and the condition, and
the *real* store, resume gate, verdict functions and claim boundaries run on
top of it. That is the whole point: a MOCK run exercises the plumbing that a
real run would otherwise discover is broken eight L4-hours in.

**A green MOCK run is evidence about this repository's control flow and about
Gemma 4 in exactly no respect.** No number produced here may appear in a
scientific report.

The four study scenarios are deliberately distinguishable end to end:

``favorable``
    the exchange moves the answer and no control does — a development GO.
``null``
    nothing moves the answer — a NO_GO with intact controls.
``control_failure``
    the exchange *and* the controls move the answer — a NO_GO that must not be
    reported as a null, because it indicts the instrument rather than the
    hypothesis.
``capability_no_go``
    the clean model cannot answer the question often enough to recruit a
    population, so no intervention runs at all.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from jlens.mmpilot.multimodal_followup import (
    CONTROL_CONDITIONS,
    MODALITIES,
    REQUIRED_CONDITIONS,
    VALIDATED_BAND,
    artifact_exclusion_audit,
    asymmetry_replication_design,
    asymmetry_replication_verdict,
    audit_property_family,
    confirmation_verdict,
    exclusion_universe,
    freeze_new_property_design,
    localization_grid,
    new_property_development_verdict,
    summarize_localization,
)
from jlens.mmpilot.multimodal_instrument import (
    MODEL_DTYPE_REALIZATION,
    realization_policy_digest,
)
from jlens.mmpilot.store import RunFingerprint, UnitStore, payload_checksum, safe_key

__all__ = [
    "MOCK_LENS_CHECKSUM",
    "MockDisconnect",
    "SCENARIOS",
    "mock_groups",
    "run_mock_asymmetry_study",
    "run_mock_localization",
    "run_mock_new_property_study",
]

#: A stand-in for the pooled L16-L40 lens checksum. It is a real sha256 shape
#: so the "pin the lens by checksum" guard is genuinely exercised.
MOCK_LENS_CHECKSUM = "sha256:" + "0" * 64

SCENARIOS = ("favorable", "null", "control_failure", "capability_no_go")

#: In the MOCK world the effect lives in the late layers. A band carries it
#: only if it contains this layer, which makes suffix bands separate from
#: prefix bands and gives the partition family something to find.
MOCK_EFFECT_LAYER = 34


class MockDisconnect(RuntimeError):
    """A simulated Colab disconnect, raised between two units of work."""


def mock_groups(prefix: str, count: int) -> list[dict]:
    """Deterministic synthetic photograph groups."""

    return [
        {
            "group_id": f"{prefix}-g{index:03d}",
            "image_id": f"{prefix}-i{index:03d}",
            "caption": f"a photograph of subject {index}",
            "image_path": f"/mock/{prefix}/{index:03d}.jpg",
            "audio_path": f"/mock/{prefix}/{index:03d}.wav",
        }
        for index in range(int(count))
    ]


def _clean_answer(scenario: str, index: int, source_answer: str) -> str:
    """The clean model's answer. Only one candidate is capable under NO_GO."""

    if scenario == "capability_no_go" and int(index) > 0:
        return "i don't know"
    return source_answer


def _patched_answer(
    scenario: str,
    condition: str,
    *,
    source_answer: str,
    target_answer: str,
    layers: Sequence[int] = VALIDATED_BAND,
) -> str:
    if condition == "zero":
        return source_answer
    if scenario == "favorable":
        return target_answer if condition == "exact" else source_answer
    if scenario == "control_failure":
        return target_answer if condition != "zero" else source_answer
    if scenario == "localization":
        return (
            target_answer
            if condition == "exact" and MOCK_EFFECT_LAYER in set(map(int, layers))
            else source_answer
        )
    return source_answer


def _trial_record(
    *,
    group: Mapping,
    modality: str,
    condition: str,
    layers: Sequence[int],
    prompt_len: int,
    generated: str,
    expected: str,
) -> dict:
    return {
        "group_id": str(group["group_id"]),
        "image_id": str(group["image_id"]),
        "modality": str(modality),
        "condition": str(condition),
        "alpha": 0.0 if condition == "zero" else 1.0,
        "layers_patched": list(map(int, layers)),
        "all_prompt_positions_patched": True,
        "positions_patched": {
            str(layer): list(range(prompt_len)) for layer in map(int, layers)
        },
        "max_activation_norm_ratio": 1.05,
        "max_update_to_activation_norm_ratio": 0.20,
        # The mock stands in for a *faithfully realized* intervention, so it
        # emits the complete diagnostic shape the real trial path produces.
        # Emitting a partial one made the verdict's integrity check look
        # satisfied by rows that had never been measured.
        "max_orthogonal_residual_drift": 0.0,
        "max_coordinate_update_error": 0.0,
        "all_hooks_fired": True,
        "all_finite": True,
        "all_layers_are_exact_alpha_one_exchange_before_cast": (
            condition != "zero"
        ),
        "all_model_dtype_realizations_converged": True,
        "post_cast_audit_passed": True,
        "model_dtype_realization_policy": MODEL_DTYPE_REALIZATION.to_dict(),
        "max_model_dtype_corrections_applied": 0,
        "teacher_forcing_used": False,
        "candidate_list_supplied": False,
        "generated_text": generated,
        "expected": expected,
        "success": generated == expected,
    }


def _store(root: Path, *, config: Mapping, split_id: str) -> UnitStore:
    fingerprint = RunFingerprint(
        mode="mock",
        model_repo_id="mock/gemma-4-E4B-it",
        model_revision="mock-revision",
        processor_revision="mock-revision",
        layers=tuple(map(int, config["layers"])),
        lens_checksum=str(config["lens_checksum"]),
        manifest_checksum="sha256:" + "1" * 64,
        split_id=split_id,
        intervention_config={
            "conditions": list(REQUIRED_CONDITIONS),
            "alpha": config.get("alpha", 1.0),
            "positions": "all_original_prompt_positions",
        },
        extra={"study_digest": payload_checksum(dict(config))},
    )
    store = UnitStore(root, fingerprint)
    store.open()
    return store


def run_mock_localization(
    root: str | Path,
    *,
    n_photographs: int = 4,
    interrupt_after: int | None = None,
) -> dict:
    """Exercise Experiment A's grid, store, resume and exploratory labelling."""

    grid = localization_grid()
    groups = mock_groups("locdev", n_photographs)
    config = {
        "study": "mock_localization",
        "layers": list(VALIDATED_BAND),
        "lens_checksum": MOCK_LENS_CHECKSUM,
        "grid_digest": grid["grid_digest"],
    }
    store = _store(Path(root), config=config, split_id=payload_checksum(groups))
    rows: list[dict] = []
    computed = 0
    for band in grid["bands"]:
        for group in groups:
            for modality in MODALITIES:
                for condition in REQUIRED_CONDITIONS:
                    key = safe_key(
                        "loc", band["name"], group["group_id"], modality, condition
                    )
                    stored = store.load("intervention", key)
                    if stored is None:
                        if interrupt_after is not None and computed >= interrupt_after:
                            raise MockDisconnect(
                                f"simulated disconnect after {computed} units"
                            )
                        stored = {
                            "band": band["name"],
                            **_trial_record(
                                group=group,
                                modality=modality,
                                condition=condition,
                                layers=band["layers"],
                                prompt_len=6,
                                generated=_patched_answer(
                                    "localization",
                                    condition,
                                    source_answer="2",
                                    target_answer="4",
                                    layers=band["layers"],
                                ),
                                expected="4",
                            ),
                        }
                        store.save("intervention", key, stored)
                        computed += 1
                    rows.append(stored)
    report = summarize_localization(rows, grid=grid)
    report = {**report, "n_units_computed_this_session": computed}
    (Path(root) / "exploratory_band_localization_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    return report


def run_mock_new_property_study(
    root: str | Path,
    *,
    scenario: str = "favorable",
    family: str = "animal_sound",
    direction: Sequence[str] = ("cat", "cow"),
    n_candidates: int = 16,
    n_recruited: int = 8,
    interrupt_after: int | None = None,
    open_confirmation: bool = True,
) -> dict:
    """Run Stage B0-B3 in the MOCK world, including the freeze gate."""

    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    source, target = str(direction[0]), str(direction[1])

    prior = exclusion_universe(
        fit_image_ids=[f"fit-i{index:03d}" for index in range(4)],
        broad_development_image_ids=[f"broaddev-i{index:03d}" for index in range(4)],
        confirmation_candidate_image_ids=[
            f"confcand-i{index:03d}" for index in range(64)
        ],
    )

    audit = audit_property_family(
        family,
        available_media={source: 999, target: 999},
        min_media_per_concept=int(n_candidates),
        clean_capability={
            concept: dict.fromkeys(MODALITIES, 1.0) for concept in (source, target)
        },
    )
    by_concept = {row["concept"]: row for row in audit["concepts"]}
    if source not in by_concept or target not in by_concept:
        raise ValueError("the mock direction is not declared by this family")
    source_answer = str(by_concept[source]["answer"])
    target_answer = str(by_concept[target]["answer"])

    development = _mock_stage(
        root / "development",
        scenario=scenario,
        prefix="newpropdev",
        source_answer=source_answer,
        target_answer=target_answer,
        direction=(source, target),
        n_candidates=n_candidates,
        n_recruited=n_recruited,
        interrupt_after=interrupt_after,
        exclusions=prior,
        label="development",
    )
    development_report = new_property_development_verdict(
        development["rows"],
        audit=audit,
        capability_go=development["capability_go"],
    )
    result = {
        "scenario": scenario,
        "audit": audit,
        "development": development_report,
        "development_units_computed": development["computed"],
        "confirmation": None,
        "frozen_design": None,
    }
    if development_report["verdict"] != "NEW_PROPERTY_DEVELOPMENT_GO":
        return result
    if not open_confirmation:
        return result

    spent = exclusion_universe(
        fit_image_ids=prior["sources"]["fit"],
        broad_development_image_ids=prior["sources"]["broad_development"],
        confirmation_candidate_image_ids=prior["sources"][
            "confirmation_candidates_all_opened"
        ],
        extra_image_ids={
            "new_property_development": [
                str(row["image_id"]) for row in development["population"]
            ]
        },
    )
    design = freeze_new_property_design(
        development=development_report,
        audit=audit,
        direction=(source, target),
        lens_checksum=MOCK_LENS_CHECKSUM,
        exclusions=spent,
        n_candidates=n_candidates,
        n_recruited=n_recruited,
        min_success_rate=0.75,
        min_control_margin=0.25,
        min_clean_capability_rate=0.75,
        familywise_alpha=0.05,
        recruitment_rule="clean property capability in all three modalities",
        seed="mock-new-property-confirmation",
    )
    design_path = root / "frozen_new_property_design.json"
    design_path.write_text(
        json.dumps(design, indent=2, default=str), encoding="utf-8"
    )
    confirmation = _mock_stage(
        root / "confirmation",
        scenario=scenario,
        prefix="newpropconf",
        source_answer=source_answer,
        target_answer=target_answer,
        direction=(source, target),
        n_candidates=n_candidates,
        n_recruited=n_recruited,
        interrupt_after=None,
        exclusions=spent,
        label="confirmation",
    )
    result["frozen_design"] = design
    result["confirmation"] = confirmation_verdict(
        confirmation["rows"],
        design=design,
        capability_go=confirmation["capability_go"],
        exclusion_audit=confirmation["exclusion_audit"],
    )
    return result


def run_mock_asymmetry_study(
    root: str | Path,
    *,
    scenario: str = "null",
    n_candidates: int = 16,
    n_recruited: int = 8,
) -> dict:
    """Run Experiment C in the MOCK world under an already-frozen design."""

    root = Path(root)
    spent = exclusion_universe(
        confirmation_candidate_image_ids=[
            f"confcand-i{index:03d}" for index in range(64)
        ],
    )
    design = asymmetry_replication_design(
        lens_checksum=MOCK_LENS_CHECKSUM,
        exclusions=spent,
        n_candidates=n_candidates,
        n_recruited=n_recruited,
    )
    stage = _mock_stage(
        root / "asymmetry",
        scenario=scenario,
        prefix="asym",
        source_answer="4",
        target_answer="2",
        direction=("cat", "bird"),
        n_candidates=n_candidates,
        n_recruited=n_recruited,
        interrupt_after=None,
        exclusions=spent,
        label="asymmetry",
    )
    return {
        "design": design,
        "report": asymmetry_replication_verdict(
            stage["rows"],
            design=design,
            capability_go=stage["capability_go"],
            exclusion_audit=stage["exclusion_audit"],
        ),
    }


def _mock_stage(
    root: Path,
    *,
    scenario: str,
    prefix: str,
    source_answer: str,
    target_answer: str,
    direction: Sequence[str],
    n_candidates: int,
    n_recruited: int,
    interrupt_after: int | None,
    exclusions: Mapping,
    label: str,
) -> dict:
    """One capability screen plus one intervention grid, atomically resumable."""

    root.mkdir(parents=True, exist_ok=True)
    groups = mock_groups(prefix, n_candidates)
    audit = artifact_exclusion_audit(groups, universe=exclusions, label=label)
    config = {
        "study": f"mock_{label}",
        "layers": list(VALIDATED_BAND),
        "lens_checksum": MOCK_LENS_CHECKSUM,
        "direction": list(direction),
        "scenario": scenario,
        # The trial-record schema is part of what these units *are*: a unit
        # written before the integrity clauses existed cannot be scored by a
        # verdict that enforces them. Binding the policy digest sends a schema
        # change to a new directory instead of resuming stale rows into a
        # spurious NO_GO.
        "realization_policy_digest": realization_policy_digest(),
    }
    store = _store(root, config=config, split_id=audit["audit_digest"])
    computed = 0

    capability: list[dict] = []
    for index, group in enumerate(groups):
        for modality in MODALITIES:
            key = safe_key("cap", group["group_id"], modality)
            stored = store.load("capability", key)
            if stored is None:
                if interrupt_after is not None and computed >= interrupt_after:
                    raise MockDisconnect(f"simulated disconnect after {computed} units")
                generated = _clean_answer(scenario, index, source_answer)
                stored = {
                    "group_id": str(group["group_id"]),
                    "image_id": str(group["image_id"]),
                    "modality": modality,
                    "expected": source_answer,
                    "generated": generated,
                    "pass": generated == source_answer,
                }
                store.save("capability", key, stored)
                computed += 1
            capability.append(stored)

    recruited: list[dict] = []
    for group in groups:
        rows = [
            row for row in capability if row["group_id"] == str(group["group_id"])
        ]
        if len(rows) == len(MODALITIES) and all(row["pass"] for row in rows):
            recruited.append(group)
        if len(recruited) == int(n_recruited):
            break
    capability_go = len(recruited) == int(n_recruited)

    rows: list[dict] = []
    if capability_go:
        for group in recruited:
            for modality in MODALITIES:
                for condition in REQUIRED_CONDITIONS:
                    key = safe_key("trial", group["group_id"], modality, condition)
                    stored = store.load("intervention", key)
                    if stored is None:
                        if interrupt_after is not None and computed >= interrupt_after:
                            raise MockDisconnect(
                                f"simulated disconnect after {computed} units"
                            )
                        stored = {
                            "direction": f"{direction[0]}->{direction[1]}",
                            **_trial_record(
                                group=group,
                                modality=modality,
                                condition=condition,
                                layers=VALIDATED_BAND,
                                prompt_len=6,
                                generated=_patched_answer(
                                    scenario,
                                    condition,
                                    source_answer=source_answer,
                                    target_answer=target_answer,
                                ),
                                expected=target_answer,
                            ),
                        }
                        store.save("intervention", key, stored)
                        computed += 1
                    rows.append(stored)
    return {
        "population": groups,
        "recruited": recruited,
        "capability_rows": capability,
        "capability_go": capability_go,
        "rows": rows,
        "computed": computed,
        "exclusion_audit": audit,
        "controls": list(CONTROL_CONDITIONS),
    }
