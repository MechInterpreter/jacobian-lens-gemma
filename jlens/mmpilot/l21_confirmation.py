"""Prospective confirmation of the exploratory layer-21 R-lens result.

The completed L21 run is discovery evidence.  This module deliberately keeps
that evidence separate from a new text population and from the later fresh
multimodal population.  It contains no model hooks; the existing audited
coordinate-swap implementation remains the only intervention implementation.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from jlens.metadata import file_sha256
from jlens.mmpilot.store import payload_checksum
from jlens.mmpilot.workspace_replication import (
    TextReplicationTask,
    WorkspaceReplicationRefused,
    holm_adjust,
    paired_binary_superiority,
)

DISCOVERY_FINGERPRINT = (
    "sha256:35def71a33c66564b2cf3ad550b80441fccb3feffa039ea34ace72e1581be223"
)
DISCOVERY_BAND = (21,)
DISCOVERY_INSTRUMENT = "matched_text_r"
PROBE_SWAP_VERSION = "mmpilot.l21_probe_swap_confirmation.v1"
MULTIMODAL_VERSION = "mmpilot.l21_fresh_multimodal_confirmation.v1"


@dataclass(frozen=True)
class L21TextThresholds:
    """Frozen thresholds for the independent text confirmation."""

    min_eligible_tasks: int = 30
    min_exact_successes: int = 5
    min_exact_success_rate: float = 0.15
    min_success_categories: int = 2
    familywise_alpha: float = 0.05

    def __post_init__(self) -> None:
        if self.min_eligible_tasks < 1 or self.min_exact_successes < 1:
            raise ValueError("text confirmation counts must be positive")
        if not 0.0 < self.min_exact_success_rate <= 1.0:
            raise ValueError("min_exact_success_rate must lie in (0, 1]")
        if self.min_success_categories < 1:
            raise ValueError("min_success_categories must be positive")
        if not 0.0 < self.familywise_alpha < 1.0:
            raise ValueError("familywise_alpha must lie in (0, 1)")


@dataclass(frozen=True)
class L21MultimodalThresholds:
    """Frozen pooled endpoint for trimodal downstream recomputation."""

    min_clean_capability_rate: float = 0.75
    min_property_success_rate_per_modality: float = 0.25
    min_property_successes_per_modality: int = 4
    familywise_alpha: float = 0.05

    def __post_init__(self) -> None:
        for value in (
            self.min_clean_capability_rate,
            self.min_property_success_rate_per_modality,
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError("multimodal rates must lie in [0, 1]")
        if self.min_property_successes_per_modality < 1:
            raise ValueError("minimum multimodal successes must be positive")
        if not 0.0 < self.familywise_alpha < 1.0:
            raise ValueError("familywise_alpha must lie in (0, 1)")


def probe_swap_tasks(path: str | Path) -> tuple[TextReplicationTask, ...]:
    """Load Anthropic's checked-in 90-item probe-swap population.

    Every row is an implicit two-hop task: the prompt describes but does not
    name the intermediate, and the intervention exchanges it for ``swap_to``.
    The category is retained in ``family`` for a predeclared diversity check.
    """

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    items = list(payload.get("items") or ())
    if len(items) != 90:
        raise WorkspaceReplicationRefused(
            f"probe-swap population has {len(items)} rows, expected 90"
        )
    tasks = []
    seen = set()
    for item in items:
        task_id = f"probe_swap__{item['name']}"
        if task_id in seen:
            raise WorkspaceReplicationRefused(f"duplicate probe-swap task {task_id}")
        seen.add(task_id)
        prompt = str(item["prompt"])
        source_name = str(item["intermediate"])
        target_name = str(item["swap_to"])
        if source_name.casefold() in prompt.casefold() or target_name.casefold() in prompt.casefold():
            raise WorkspaceReplicationRefused(
                f"{task_id} names its hidden source or swap target in the prompt"
            )
        tasks.append(
            TextReplicationTask(
                task_id=task_id,
                family=str(item["category"]),
                prompt=prompt,
                source=source_name,
                target=target_name,
                clean_answer=str(item["answer"]),
                swapped_answer=str(item["swap_answer"]),
                implicit_intermediate=True,
            )
        )
    return tuple(tasks)


def assert_disjoint_from_discovery(
    tasks: Sequence[TextReplicationTask],
    discovery_tasks: Sequence[TextReplicationTask],
) -> dict:
    """Refuse exact task-id or prompt reuse from the exploratory population."""

    old_ids = {task.task_id for task in discovery_tasks}
    old_prompts = {task.prompt.strip().casefold() for task in discovery_tasks}
    id_overlap = sorted(old_ids & {task.task_id for task in tasks})
    prompt_overlap = sorted(
        task.task_id
        for task in tasks
        if task.prompt.strip().casefold() in old_prompts
    )
    if id_overlap or prompt_overlap:
        raise WorkspaceReplicationRefused(
            f"text confirmation overlaps discovery: ids={id_overlap}, prompts={prompt_overlap}"
        )
    payload = {
        "version": PROBE_SWAP_VERSION,
        "n_confirmation_tasks": len(tasks),
        "n_discovery_tasks": len(discovery_tasks),
        "task_id_overlap": id_overlap,
        "prompt_overlap": prompt_overlap,
        "disjoint": True,
    }
    return {**payload, "disjointness_digest": payload_checksum(payload)}


def discover_l21_run(
    runs_root: str | Path,
    *,
    expected_fingerprint: str = DISCOVERY_FINGERPRINT,
) -> dict:
    """Find exactly one completed run carrying the verified L21 discovery."""

    root = Path(runs_root)
    candidates = []
    for selection_path in sorted(
        root.glob("mmworkspace/mmworkspace_real_*/units/metric/loading_first_selection.json")
    ):
        unit = json.loads(selection_path.read_text(encoding="utf-8"))
        if str(unit.get("fingerprint_digest")) != str(expected_fingerprint):
            continue
        selection = dict(unit.get("payload") or {})
        if tuple(map(int, selection.get("selected_band") or ())) != DISCOVERY_BAND:
            continue
        if selection.get("selected_instrument") != DISCOVERY_INSTRUMENT:
            continue
        run_dir = selection_path.parents[2]
        candidates.append((run_dir, selection_path, unit, selection))
    if len(candidates) != 1:
        raise WorkspaceReplicationRefused(
            "expected exactly one frozen L21 discovery run with fingerprint "
            f"{expected_fingerprint}, found {[str(row[0]) for row in candidates]}"
        )
    run_dir, selection_path, unit, selection = candidates[0]
    inventory_path = run_dir / "r_lens_inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    lenses = dict(inventory.get("lenses") or {})
    artifacts = {}
    for arm in ("text", "pooled"):
        record = dict(lenses.get(arm) or {})
        lens_path = Path(str(record.get("path") or (run_dir / "r_lenses" / f"lens.{arm}.pt")))
        if not lens_path.is_file():
            fallback = run_dir / "r_lenses" / f"lens.{arm}.pt"
            lens_path = fallback
        if not lens_path.is_file():
            raise WorkspaceReplicationRefused(f"missing frozen L21 {arm} R-lens")
        checksum = file_sha256(str(lens_path))
        recorded = record.get("checksum")
        if recorded and checksum != recorded:
            raise WorkspaceReplicationRefused(
                f"frozen L21 {arm} R-lens checksum changed"
            )
        artifacts[arm] = {"path": str(lens_path), "checksum": checksum}
    payload = {
        "version": PROBE_SWAP_VERSION,
        "run_dir": str(run_dir),
        "fingerprint_digest": str(unit["fingerprint_digest"]),
        "selection_path": str(selection_path),
        "selection_checksum": selection.get("selection_digest"),
        "selected_band": list(DISCOVERY_BAND),
        "selected_instrument": DISCOVERY_INSTRUMENT,
        "artifacts": artifacts,
        "discovery_outcome_not_recomputed": True,
    }
    return {**payload, "source_digest": payload_checksum(payload)}


def task_level_loading_admission(
    rows: Sequence[Mapping],
    *,
    task_ids: Sequence[str],
    layer: int = 21,
    min_source_cosine: float = 0.0,
    min_source_advantage: float = 0.0,
) -> dict:
    """Admit each task independently using clean final-token loading only."""

    expected = tuple(map(str, task_ids))
    indexed = {}
    for row in rows:
        if int(row.get("layer", -1)) != int(layer):
            continue
        if str(row.get("position_class")) != "final_prompt_token":
            continue
        indexed[str(row.get("sample_id"))] = dict(row)
    missing = sorted(set(expected) - set(indexed))
    records = []
    for task_id in expected:
        row = indexed.get(task_id)
        passed = bool(
            row is not None
            and float(row["source_cosine"]) > float(min_source_cosine)
            and float(row["source_advantage"]) > float(min_source_advantage)
        )
        records.append(
            {
                "task_id": task_id,
                "source_cosine": None if row is None else float(row["source_cosine"]),
                "source_advantage": None if row is None else float(row["source_advantage"]),
                "passed": passed,
            }
        )
    payload = {
        "version": PROBE_SWAP_VERSION,
        "layer": int(layer),
        "position_class": "final_prompt_token",
        "strict_task_level_gate": True,
        "min_source_cosine_exclusive": float(min_source_cosine),
        "min_source_advantage_exclusive": float(min_source_advantage),
        "missing_tasks": missing,
        "eligible_task_ids": [row["task_id"] for row in records if row["passed"]],
        "ineligible_task_ids": [row["task_id"] for row in records if not row["passed"]],
        "records": records,
        "causal_outcome_consulted": False,
    }
    return {**payload, "admission_digest": payload_checksum(payload)}


def l21_text_confirmation_verdict(
    rows: Sequence[Mapping],
    *,
    eligible_tasks: Sequence[TextReplicationTask],
    thresholds: L21TextThresholds | None = None,
) -> dict:
    """Predeclared paired-control verdict for the fresh text population."""

    thresholds = thresholds or L21TextThresholds()
    tasks = {task.task_id: task for task in eligible_tasks}
    indexed = {str(row["task_id"]): dict(row) for row in rows}
    missing = sorted(set(tasks) - set(indexed))
    ordered = [indexed[name] for name in tasks if name in indexed]
    endpoint_fields = (
        "exact_primary_swapped_answer_generated",
        "zero_swapped_answer_generated",
        "random_swapped_answer_generated",
        "unrelated_swapped_answer_generated",
    )
    malformed = {
        str(row.get("task_id")): [name for name in endpoint_fields if name not in row]
        for row in ordered
        if any(name not in row for name in endpoint_fields)
    }
    if malformed:
        raise WorkspaceReplicationRefused(
            "L21 confirmation rows are missing frozen endpoint fields: "
            f"{malformed}"
        )
    integrity = not missing and all(bool(row.get("integrity_passed")) for row in ordered)
    exact = [
        bool(row["exact_primary_swapped_answer_generated"])
        for row in ordered
    ]
    controls = {
        "zero": [bool(row.get("zero_swapped_answer_generated")) for row in ordered],
        "random": [bool(row.get("random_swapped_answer_generated")) for row in ordered],
        "unrelated": [bool(row.get("unrelated_swapped_answer_generated")) for row in ordered],
    }
    paired = {
        name: paired_binary_superiority(exact, values)
        for name, values in controls.items()
    } if ordered else {}
    adjusted = holm_adjust(
        {name: record["one_sided_exact_p"] for name, record in paired.items()}
    ) if paired else {}
    for name, record in paired.items():
        record["holm_p"] = adjusted[name]
        record["passed"] = adjusted[name] <= thresholds.familywise_alpha
    successes = sum(exact)
    rate = successes / len(exact) if exact else 0.0
    success_categories = sorted(
        {
            tasks[row["task_id"]].family
            for row, success in zip(ordered, exact, strict=True)
            if success
        }
    )
    passed = bool(
        integrity
        and len(ordered) >= thresholds.min_eligible_tasks
        and successes >= thresholds.min_exact_successes
        and rate >= thresholds.min_exact_success_rate
        and len(success_categories) >= thresholds.min_success_categories
        and paired
        and all(record["passed"] for record in paired.values())
    )
    payload = {
        "version": PROBE_SWAP_VERSION,
        "verdict": "L21_TEXT_CONFIRMATION_GO" if passed else "L21_TEXT_CONFIRMATION_NO_GO",
        "discovery_layer": 21,
        "primary_alpha": 1.0,
        "exact_coordinate_exchange": True,
        "primary_endpoint_field": "exact_primary_swapped_answer_generated",
        "n_eligible_tasks": len(ordered),
        "n_exact_successes": successes,
        "exact_success_rate": rate,
        "success_categories": success_categories,
        "paired_controls": paired,
        "integrity_passed": integrity,
        "missing_tasks": missing,
        "thresholds": asdict(thresholds),
        "teacher_forcing_used": False,
        "candidate_list_supplied": False,
        "multimodal_stage_licensed": passed,
    }
    return {**payload, "report_checksum": payload_checksum(payload)}


def l21_multimodal_confirmation_verdict(
    rows: Sequence[Mapping],
    *,
    thresholds: L21MultimodalThresholds | None = None,
) -> dict:
    """Pool directions within each modality while retaining every cell."""

    thresholds = thresholds or L21MultimodalThresholds()
    selected = [dict(row) for row in rows]
    modalities = ("text", "image", "spoken_audio")
    property_rows = [row for row in selected if row.get("prompt_kind") == "property"]
    integrity = bool(selected) and all(bool(row.get("integrity_passed")) for row in selected)
    modality_records = []
    pooled_primary = []
    pooled_controls = {name: [] for name in ("zero", "random", "unrelated")}
    for modality in modalities:
        cells = [row for row in property_rows if row.get("modality") == modality]
        primary = [bool(row.get("primary_success")) for row in cells]
        controls = {
            name: [bool(row.get(f"{name}_success")) for row in cells]
            for name in pooled_controls
        }
        successes = sum(primary)
        rate = successes / len(primary) if primary else 0.0
        capability = (
            sum(bool(row.get("clean_correct")) for row in cells) / len(cells)
            if cells else 0.0
        )
        modality_records.append(
            {
                "modality": modality,
                "n": len(cells),
                "clean_capability_rate": capability,
                "primary_successes": successes,
                "primary_success_rate": rate,
                "control_success_rates": {
                    name: (sum(values) / len(values) if values else 0.0)
                    for name, values in controls.items()
                },
                "passed": bool(
                    cells
                    and capability >= thresholds.min_clean_capability_rate
                    and successes >= thresholds.min_property_successes_per_modality
                    and rate >= thresholds.min_property_success_rate_per_modality
                    and all(rate > (sum(values) / len(values)) for values in controls.values())
                ),
            }
        )
        pooled_primary.extend(primary)
        for name, values in controls.items():
            pooled_controls[name].extend(values)
    paired = {
        name: paired_binary_superiority(pooled_primary, values)
        for name, values in pooled_controls.items()
    } if pooled_primary else {}
    adjusted = holm_adjust(
        {name: record["one_sided_exact_p"] for name, record in paired.items()}
    ) if paired else {}
    for name, record in paired.items():
        record["holm_p"] = adjusted[name]
        record["passed"] = adjusted[name] <= thresholds.familywise_alpha
    passed = bool(
        integrity
        and all(record["passed"] for record in modality_records)
        and paired
        and all(record["passed"] for record in paired.values())
    )
    payload = {
        "version": MULTIMODAL_VERSION,
        "verdict": (
            "L21_TRIMODAL_DOWNSTREAM_RECOMPUTATION_GO"
            if passed else "L21_TRIMODAL_DOWNSTREAM_RECOMPUTATION_NO_GO"
        ),
        "modalities": modality_records,
        "pooled_paired_controls": paired,
        "integrity_passed": integrity,
        "thresholds": asdict(thresholds),
        "teacher_forcing_used": False,
        "candidate_list_supplied": False,
    }
    return {**payload, "report_checksum": payload_checksum(payload)}


__all__ = [
    "DISCOVERY_BAND",
    "DISCOVERY_FINGERPRINT",
    "DISCOVERY_INSTRUMENT",
    "L21MultimodalThresholds",
    "L21TextThresholds",
    "MULTIMODAL_VERSION",
    "PROBE_SWAP_VERSION",
    "assert_disjoint_from_discovery",
    "discover_l21_run",
    "l21_multimodal_confirmation_verdict",
    "l21_text_confirmation_verdict",
    "probe_swap_tasks",
    "task_level_loading_admission",
]
