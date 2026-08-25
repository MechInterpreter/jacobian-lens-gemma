"""Outcome-blind causal-path localization for the cat-to-dog follow-up.

The completed fp32 development run established that neither the exact
cat-to-dog exchange nor its displacement-matched direct-answer positive
control moved the answer on the frozen L16-L40/all-prompt-position path.  A
failed positive control makes that result inconclusive: it does not separate a
failed identity exchange from a path that cannot influence the requested
answer at all.

This module defines the one permitted repair.  It searches a small, frozen
grid using only the direct-answer positive control on the already-spent
development photographs.  Exact-exchange generations are never scored or
used for selection.  A selected path is development-only and must be tested
with the real alpha=1 exchange on different development photographs before an
untouched confirmation population can be opened.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

from .store import payload_checksum

CATDOG_PATH_LOCALIZATION_VERSION = (
    "mmpilot.catdog_direct_answer_path_localization.v1"
)

# Frozen before this stage opens any new direct-answer output.
#
# Five sliding 8-layer windows stepping by 4, which together cover every layer
# in [16, 40], plus the full band as an internal control.  Overlapping windows
# localize leverage without the boundary artefact a disjoint partition
# introduces: a mechanism straddling a partition edge shows up weakly in two
# adjacent cells and strongly in neither.  The last window starts at 33 rather
# than continuing the stride to 32, because a 32-39 window would omit layer 40
# while overlapping 28-35 almost entirely.
#
# The full band at all prompt positions is retained deliberately, even though
# the checksum-pinned source run already measured it at 0/8 in every modality:
# reproducing that null here is the run's own consistency check.  If this
# control disagrees with the run it is diagnosing, the disagreement is reported
# instead of a selection.
CATDOG_PATH_BANDS: tuple[tuple[int, ...], ...] = (
    tuple(range(16, 24)),
    tuple(range(20, 28)),
    tuple(range(24, 32)),
    tuple(range(28, 36)),
    tuple(range(33, 41)),
    tuple(range(16, 41)),
)

#: The full validated band, named once so the control cannot drift from it.
CATDOG_CONTROL_BAND: tuple[int, ...] = tuple(range(16, 41))

# The first rule is the paper-faithful primary intervention.  The other two
# are explicitly diagnostic position controls.  ``non_evidence`` is omitted
# because it is almost the complement of the other rules and would add cost
# without resolving a distinct scientific ambiguity here.
CATDOG_PATH_POSITION_RULES: tuple[str, ...] = (
    "all_prompt_positions",
    "modality_evidence",
    "final_prompt_token_only",
)

CATDOG_PATH_POSITION_POLICIES: dict[str, dict[str, str]] = {
    "all_prompt_positions": {
        modality: "all_prompt_positions"
        for modality in ("text", "image", "spoken_audio")
    },
    # Gemma's text route carries the caption as ordinary prompt text and does
    # not expose a distinct modality span; images and native audio do expose
    # exact media-token spans.  Text is therefore recorded as **not
    # applicable** rather than falling back to all_prompt_positions.
    #
    # The fallback was the original implementation and it was wrong twice
    # over.  It made the text cell a byte-identical rerun of the
    # all_prompt_positions path's text cell -- same band, same photographs,
    # same rule, deterministic model -- so it burned one forward per
    # photograph per band to recompute a number already in hand.  Worse, a
    # path passing "in all three modalities" would have had its text arm
    # measured under a *different rule than the one named*, which is exactly
    # the kind of substitution a reader of the verdict cannot see.  ``None``
    # means the cell is undefined, and :func:`summarize_path_localization`
    # refuses to select any path that has one.
    "modality_evidence": {
        "text": None,
        "image": "evidence_span_only",
        "spoken_audio": "evidence_span_only",
    },
    "final_prompt_token_only": {
        modality: "final_prompt_token_only"
        for modality in ("text", "image", "spoken_audio")
    },
}


class CatDogLocalizationRefused(RuntimeError):
    """Raised when a localization input cannot establish its provenance."""


def applied_position_rule(position_rule: str, modality: str) -> str | None:
    """The rule actually applied for this cell, or ``None`` when undefined.

    ``None`` is a real state, not a missing value: a text prompt has no
    distinct evidence token span, so ``modality_evidence`` cannot be run for
    it. Callers skip those cells rather than substituting another rule.
    """
    return CATDOG_PATH_POSITION_POLICIES[str(position_rule)].get(str(modality))


def frozen_grid_record() -> dict:
    """Return the complete, checksum-bound direct-answer search grid."""

    modalities = ("text", "image", "spoken_audio")
    paths = []
    for band in CATDOG_PATH_BANDS:
        for rule in CATDOG_PATH_POSITION_RULES:
            applicable = [
                modality for modality in modalities
                if applied_position_rule(rule, modality) is not None
            ]
            paths.append(
                {
                    "band": list(band),
                    "n_layers": len(band),
                    "position_rule": rule,
                    "applied_position_rule_by_modality": {
                        modality: applied_position_rule(rule, modality)
                        for modality in modalities
                    },
                    "applicable_modalities": applicable,
                    # A path missing any modality can never satisfy the
                    # all-three-modalities selection rule, so it is recorded
                    # as diagnostic information and never selected.
                    "selectable": len(applicable) == len(modalities),
                    "is_control_band": tuple(band) == CATDOG_CONTROL_BAND
                    and rule == "all_prompt_positions",
                }
            )
    body = {
        "version": CATDOG_PATH_LOCALIZATION_VERSION,
        "bands": [list(band) for band in CATDOG_PATH_BANDS],
        "control_band": list(CATDOG_CONTROL_BAND),
        "position_rules": list(CATDOG_PATH_POSITION_RULES),
        "position_policy_by_modality": CATDOG_PATH_POSITION_POLICIES,
        "paths": paths,
        "n_paths": len(paths),
        "n_selectable_paths": sum(1 for path in paths if path["selectable"]),
        "alpha": 1.0,
        "selection_signal": "direct_answer_positive_control_only",
        "exact_exchange_outcomes_used_for_selection": False,
        "source_full_band_all_positions_repeated": True,
        "source_full_band_result": "0/8 in every modality",
        "control_band_purpose": (
            "the full band at all prompt positions is rerun as this run's own "
            "consistency check against the source it diagnoses; a disagreement "
            "is reported instead of a selection"
        ),
        "interpretation": (
            "instrument development on already-spent photographs; never a "
            "cat-to-dog causal result"
        ),
    }
    return {**body, "grid_digest": payload_checksum(body)}


def verify_inconclusive_source_report(
    report: Mapping,
    *,
    expected_checksum: str,
    expected_model_revision: str,
    expected_lens_checksum: str,
) -> dict:
    """Verify the exact completed run that licenses path localization."""

    body = {key: value for key, value in report.items() if key != "report_checksum"}
    recorded = str(report.get("report_checksum") or "")
    recomputed = payload_checksum(body)
    problems: list[str] = []
    if recorded != str(expected_checksum):
        problems.append(f"report checksum {recorded!r} != pin {expected_checksum!r}")
    if recomputed != recorded:
        problems.append(f"report checksum recomputes to {recomputed!r}")

    config = report.get("scientific_config") or {}
    expected_config = {
        "model_revision": str(expected_model_revision),
        "model_dtype": "float32",
        "lens_checksum": str(expected_lens_checksum),
        "direction": ["cat", "dog"],
        "alpha": 1.0,
        "layers": list(range(16, 41)),
        "positions": "every original prompt position",
        "prompt_id": "identity_explicit_v1",
        "outcome_informed_stage_design": False,
        "is_confirmation": False,
    }
    for key, value in expected_config.items():
        if config.get(key) != value:
            problems.append(f"scientific_config[{key!r}]={config.get(key)!r}, expected {value!r}")

    directions = list(report.get("directions") or ())
    if len(directions) != 1 or directions[0].get("direction") != "cat->dog":
        problems.append("source report is not the single cat->dog development result")
    else:
        direction = directions[0]
        if direction.get("instrument_state") != "INCONCLUSIVE":
            problems.append("source instrument state is not INCONCLUSIVE")
        direct = direction.get("direct_answer_positive_control") or {}
        if direct.get("passed") is not False:
            problems.append("source direct-answer control did not fail")
        rates = direct.get("by_modality") or {}
        for modality in ("text", "image", "spoken_audio"):
            row = rates.get(modality) or {}
            if int(row.get("n") or 0) != 8 or int(row.get("successes") or 0) != 0:
                problems.append(
                    f"source direct-answer {modality} is not the frozen 0/8 result"
                )

    rows = list(report.get("rows") or ())
    group_ids = sorted(
        {
            str(row.get("group_id"))
            for row in rows
            if row.get("condition") == "direct_answer" and row.get("group_id")
        }
    )
    if len(group_ids) != 8:
        problems.append(f"source report identifies {len(group_ids)} groups, expected 8")
    recruitment = report.get("recruitment") or {}
    recruited_cat_ids = sorted(
        str(row.get("group_id"))
        for row in ((recruitment.get("selected") or {}).get("cat") or ())
        if row.get("group_id")
    )
    if recruitment.get("complete") is not True or recruited_cat_ids != group_ids:
        problems.append(
            "source direct-answer groups do not exactly match its clean-capability "
            "recruited cat population"
        )

    if problems:
        raise CatDogLocalizationRefused(
            "the completed cat-to-dog run does not match the pinned "
            "inconclusive source:\n  - " + "\n  - ".join(problems)
        )
    return {
        "verified": True,
        "report_checksum": recorded,
        "group_ids": group_ids,
        "source_instrument_state": "INCONCLUSIVE",
        "source_direct_answer_result": {
            modality: {"successes": 0, "n": 8}
            for modality in ("text", "image", "spoken_audio")
        },
    }


def summarize_path_localization(
    rows: Sequence[Mapping],
    *,
    source_report_checksum: str,
    grid: Mapping,
    expected_group_ids: Sequence[str],
    minimum_success_rate: float = 0.50,
    post_cast_tolerance: float = 0.02,
) -> dict:
    """Score only direct-answer rows and deterministically select one path."""

    if not 0.0 <= float(minimum_success_rate) <= 1.0:
        raise CatDogLocalizationRefused("minimum_success_rate must be in [0, 1]")
    expected_ids = {str(value) for value in expected_group_ids}
    expected_modalities = ("text", "image", "spoken_audio")
    expected_bands = {tuple(map(int, band)) for band in grid.get("bands") or ()}
    expected_rules = tuple(map(str, grid.get("position_rules") or ()))
    if not expected_ids or not expected_bands or not expected_rules:
        raise CatDogLocalizationRefused("localization grid and source groups must be non-empty")

    grouped: dict[tuple[tuple[int, ...], str, str], list[Mapping]] = defaultdict(list)
    for row in rows:
        if row.get("condition") != "direct_answer":
            raise CatDogLocalizationRefused(
                "path localization accepts direct-answer rows only; exact-swap "
                "outcomes cannot enter selection"
            )
        band = tuple(map(int, row.get("layers_patched") or ()))
        rule = str(row.get("position_rule") or "")
        modality = str(row.get("modality") or "")
        if band not in expected_bands or rule not in expected_rules:
            raise CatDogLocalizationRefused(
                f"row lies outside the frozen grid: band={band}, rule={rule!r}"
            )
        grouped[(band, rule, modality)].append(row)

    candidates = []
    for band in (tuple(map(int, value)) for value in grid["bands"]):
        for rule in expected_rules:
            cells = []
            for modality in expected_modalities:
                if applied_position_rule(rule, modality) is None:
                    # Undefined, not merely absent: a text prompt has no
                    # evidence token span. Recorded so the gap is visible,
                    # and it makes the path unselectable below.
                    cells.append(
                        {
                            "modality": modality,
                            "applicable": False,
                            "applied_position_rule": None,
                            "reason": (
                                "a text prompt exposes no distinct evidence "
                                "token span; resolve_positions refuses rather "
                                "than guessing which tokens are the evidence"
                            ),
                            "n": 0,
                            "successes": 0,
                            "success_rate": None,
                            "complete": True,
                            "integrity_pass": True,
                            "passed": None,
                        }
                    )
                    continue
                cell_rows = grouped.get((band, rule, modality), [])
                ids = [str(row.get("group_id")) for row in cell_rows]
                complete = len(cell_rows) == len(expected_ids) and set(ids) == expected_ids
                integrity = complete and all(
                    row.get("all_hooks_fired") is True
                    and row.get("all_finite") is True
                    and row.get("all_model_dtype_realizations_converged") is True
                    and isinstance(
                        row.get("max_relative_cumulative_band_displacement_match_error"),
                        (int, float),
                    )
                    and float(
                        row["max_relative_cumulative_band_displacement_match_error"]
                    )
                    <= float(post_cast_tolerance)
                    and row.get("teacher_forcing_used") is False
                    and row.get("candidate_list_supplied") is False
                    for row in cell_rows
                )
                successes = sum(bool(row.get("success")) for row in cell_rows)
                rate = successes / len(cell_rows) if cell_rows else 0.0
                cells.append(
                    {
                        "modality": modality,
                        "applicable": True,
                        "applied_position_rule": applied_position_rule(
                            rule, modality
                        ),
                        "n": len(cell_rows),
                        "successes": successes,
                        "success_rate": rate,
                        "complete": complete,
                        "integrity_pass": integrity,
                        "passed": complete
                        and integrity
                        and rate >= float(minimum_success_rate),
                    }
                )
            live = [cell for cell in cells if cell["applicable"]]
            selectable = len(live) == len(expected_modalities)
            candidates.append(
                {
                    "band": list(band),
                    "n_layers": len(band),
                    "position_rule": rule,
                    "is_control_band": tuple(band) == CATDOG_CONTROL_BAND
                    and rule == "all_prompt_positions",
                    "selectable": selectable,
                    "cells": cells,
                    "minimum_modality_rate": min(
                        (cell["success_rate"] for cell in live), default=0.0
                    ),
                    "pooled_successes": sum(cell["successes"] for cell in live),
                    "pooled_n": sum(cell["n"] for cell in live),
                    # A path with an undefined modality cannot satisfy the
                    # all-three-modalities rule, however well its defined
                    # cells score. It is reported, never selected.
                    "passed": bool(
                        selectable and live and all(cell["passed"] for cell in live)
                    ),
                }
            )

    passing = [row for row in candidates if row["passed"]]
    rule_priority = {rule: index for index, rule in enumerate(expected_rules)}
    # Selection is fixed before outputs: maximize the worst-modality rate,
    # then pooled successes; prefer the paper-faithful position rule, then the
    # shortest/deepest band to avoid claiming more layers than demonstrated.
    passing.sort(
        key=lambda row: (
            -float(row["minimum_modality_rate"]),
            -int(row["pooled_successes"]),
            rule_priority[row["position_rule"]],
            len(row["band"]),
            -min(row["band"]),
        )
    )
    selected = passing[0] if passing else None

    # The full band at all prompt positions is this run's consistency check
    # against the source it diagnoses, which measured 0/8 in every modality.
    # If it disagrees, the two runs are not measuring the same thing and no
    # selection from this grid can be trusted.
    control = next(
        (row for row in candidates if row.get("is_control_band")), None
    )
    control_live = (
        [] if control is None
        else [cell for cell in control["cells"] if cell["applicable"]]
    )
    control_reproduces_source_null = (
        None if not control_live
        else all(cell["success_rate"] == 0.0 for cell in control_live)
    )
    if control_reproduces_source_null is False:
        selected = None

    body = {
        "version": CATDOG_PATH_LOCALIZATION_VERSION,
        "verdict": (
            "CATDOG_DIRECT_ANSWER_PATH_LOCALIZATION_CONTROL_DISAGREES"
            if control_reproduces_source_null is False
            else "CATDOG_DIRECT_ANSWER_PATH_LOCALIZATION_GO"
            if selected
            else "CATDOG_DIRECT_ANSWER_PATH_LOCALIZATION_NO_GO"
        ),
        "control_band_reproduces_source_null": control_reproduces_source_null,
        "n_paths_searched": len(candidates),
        "n_selectable_paths": sum(
            1 for row in candidates if row.get("selectable")
        ),
        "source_report_checksum": str(source_report_checksum),
        "grid": dict(grid),
        "minimum_success_rate": float(minimum_success_rate),
        "post_cast_tolerance": float(post_cast_tolerance),
        "n_source_groups": len(expected_ids),
        "selection_uses_exact_exchange_outcomes": False,
        "scientific_grade": "instrument_development_only",
        "can_establish_catdog_causal_transfer": False,
        "selected_path": (
            None
            if selected is None
            else {
                "band": selected["band"],
                "position_rule": selected["position_rule"],
                "minimum_modality_rate": selected["minimum_modality_rate"],
                "pooled_successes": selected["pooled_successes"],
                "pooled_n": selected["pooled_n"],
            }
        ),
        "candidates": candidates,
        "next_step": (
            "freeze this path and test the exact alpha=1 cat-to-dog exchange "
            "on different development photographs"
            if selected
            else "stop the animal-sound exchange; no common causal path was found"
        ),
        "multiplicity_disclosure": (
            "this selection is the best of n_paths_searched searched paths on "
            "already-spent photographs; quote that count wherever the selection "
            "is quoted"
        ),
    }
    return {**body, "report_checksum": payload_checksum(body)}


__all__ = [
    "CATDOG_CONTROL_BAND",
    "CATDOG_PATH_BANDS",
    "CATDOG_PATH_LOCALIZATION_VERSION",
    "applied_position_rule",
    "CATDOG_PATH_POSITION_RULES",
    "CATDOG_PATH_POSITION_POLICIES",
    "CatDogLocalizationRefused",
    "frozen_grid_record",
    "summarize_path_localization",
    "verify_inconclusive_source_report",
]
