# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Decide whether a completed swap trial measured anything at all.

Why this module exists
======================

The recruited animal-sound exploratory run
(``mmnewpropertyrescue_real_6af6affcb145``) printed ``integrity_pass = true``
for every cell and reported a clean 0/8 null in all three modalities. It was
not a null. Two independent omissions made the printed integrity flag
meaningless:

1. **The realization policy was never passed.** Stage 5B1R called
   :func:`jlens.mmpilot.workspace_replication.unrestricted_greedy_swap_trial`
   without ``realization_policy=``. The float64 coordinate exchange was
   therefore cast straight to the model's bf16 residual stream with no bounded
   correction, and the trial diagnostics recorded post-cast **relative**
   coordinate errors up to ~0.21 (exact), ~0.10 (random) and ~0.29
   (unrelated). The repository's own frozen tolerance is 0.02. The intervention
   the model actually consumed was not the intervention that was intended.

2. **The verdict ignored the evidence it already had.**
   :func:`jlens.mmpilot.multimodal_followup.generation_trial_row` faithfully
   flattened ``max_post_cast_relative_coordinate_error`` and
   ``max_post_cast_relative_residual_drift`` onto every row, but
   ``_cell_records`` scored ``integrity_pass`` from two clauses only -- all
   prompt positions patched, and the expected layer list. It never read the
   post-cast errors, ``all_hooks_fired``, ``all_finite``,
   ``all_layers_are_exact_alpha_one_exchange_before_cast`` or
   ``all_model_dtype_realizations_converged``.

So a broken instrument was reported with the vocabulary of a scientific null.

What this module fixes
======================

:func:`trial_integrity` scores **one** trial row against the complete clause
set, and :func:`cell_integrity` aggregates a cell. Three rules make the result
trustworthy:

* **A missing diagnostic fails.** ``dict.get(key) or 0.0`` turns an absent
  measurement into a passing one; that is exactly how the defect survived. A
  clause whose evidence is not present is recorded in ``missing_fields`` and
  fails.
* **Clauses are conditional on the arm, not skipped silently.** The
  ``alpha == 1`` exact-exchange-before-cast clause applies to alpha=1 arms and
  is recorded as not-applicable for the alpha=0 parity arm, never as a pass.
* **Instrument failure has its own verdict.** :data:`INSTRUMENT_STATES` keeps
  ``SCIENTIFIC_NULL`` and ``INSTRUMENT_FAILURE`` apart, and
  :func:`instrument_state` refuses to emit the first when the second applies.

The direct-answer positive control
==================================

A valid instrument is necessary but not sufficient to read a null. If an exact
coordinate exchange changes nothing, the remaining question is whether *any*
intervention on this path could have. The repository's existing norm-matched
direct-answer control
(:func:`jlens.mmpilot.workspace_replication.unrestricted_greedy_direct_answer_trial`)
answers it, and :func:`instrument_state` consumes its outcome:

===========================  =========================  =====================
primary exact exchange       direct-answer control      state
===========================  =========================  =====================
passes, controls flat        (any)                      ``EFFECT_GO``
fails                        passes                     ``SCIENTIFIC_NULL``
fails                        fails                      ``INCONCLUSIVE``
(any)                        (any), controls moved      ``CONTROL_FAILURE``
(any)                        (any), integrity failed    ``INSTRUMENT_FAILURE``
===========================  =========================  =====================

The direct-answer arm is a diagnostic. It can license reading a null and it can
withhold that licence. It can never turn a failed primary exchange into a GO,
and :func:`instrument_state` has no branch that lets it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from jlens.mmpilot.store import payload_checksum
from jlens.mmpilot.workspace_replication import (
    TEXT_MODEL_DTYPE_REALIZATION,
    TEXT_POST_CAST_MAX_RELATIVE_ERROR,
)

__all__ = [
    "INSTRUMENT_VERSION",
    "INSTRUMENT_STATES",
    "INTEGRITY_CLAUSES",
    "MODEL_DTYPE_REALIZATION",
    "POST_CAST_TOLERANCE",
    "InstrumentIntegrityRefused",
    "cell_integrity",
    "instrument_state",
    "realization_policy_digest",
    "trial_integrity",
]

INSTRUMENT_VERSION = "mmpilot.multimodal_swap_instrument_integrity.v2"

#: The one tolerance, imported rather than restated. A second literal is how a
#: gate quietly loosens; see
#: :data:`jlens.mmpilot.workspace_replication.TEXT_POST_CAST_MAX_RELATIVE_ERROR`.
POST_CAST_TOLERANCE = TEXT_POST_CAST_MAX_RELATIVE_ERROR

#: The frozen policy, re-exported under the name this study uses so a notebook
#: cannot construct a looser one by hand.
MODEL_DTYPE_REALIZATION = TEXT_MODEL_DTYPE_REALIZATION

#: Every clause a generated trial must satisfy before its outcome is scored.
#: Ordered as they are reported. ``alpha_one_exact_exchange_before_cast`` is
#: not-applicable on the alpha=0 parity arm and required everywhere else.
INTEGRITY_CLAUSES: tuple[str, ...] = (
    "expected_layers_patched",
    "all_prompt_positions_patched",
    "all_hooks_fired",
    "all_finite",
    "alpha_one_exact_exchange_before_cast",
    "model_dtype_realization_converged",
    "post_cast_coordinate_error_within_tolerance",
    "post_cast_residual_drift_within_tolerance",
    "cumulative_band_displacement_norm_matched",
    "activation_norm_ratio_within_limit",
    "update_to_activation_ratio_within_limit",
    "no_teacher_forcing",
    "no_candidate_list",
)

#: Mutually exclusive outcomes of one scored direction. ``SCIENTIFIC_NULL`` is
#: reachable only through a valid instrument *and* a passing positive control.
INSTRUMENT_STATES: tuple[str, ...] = (
    "EFFECT_GO",
    "SCIENTIFIC_NULL",
    "CONTROL_FAILURE",
    "INSTRUMENT_FAILURE",
    "INCONCLUSIVE",
)


class InstrumentIntegrityRefused(RuntimeError):
    """The integrity question was asked in a way that cannot be answered."""


def realization_policy_digest(policy=MODEL_DTYPE_REALIZATION) -> str:
    """``sha256:`` over the policy payload, for the run fingerprint.

    Binding this into the fingerprint is what makes the corrected rerun refuse
    to resume from the flawed run's units instead of silently mixing them.
    """
    return payload_checksum(policy.to_dict())


def _number(row: Mapping, key: str) -> float | None:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _flag(row: Mapping, key: str) -> bool | None:
    value = row.get(key)
    return bool(value) if isinstance(value, bool) else None


def trial_integrity(
    row: Mapping,
    *,
    layers: Sequence[int],
    tolerance: float = POST_CAST_TOLERANCE,
    max_activation_norm_ratio: float = 1.25,
    max_update_ratio: float = 0.50,
) -> dict:
    """Score one flattened trial row against :data:`INTEGRITY_CLAUSES`.

    Args:
        row: A row from
            :func:`jlens.mmpilot.multimodal_followup.generation_trial_row` or
            :func:`jlens.mmpilot.multimodal_followup.direct_answer_trial_row`.
        layers: The exact layer list the trial was supposed to patch.
        tolerance: Relative tolerance for the post-cast coordinate error,
            orthogonal residual drift, and cumulative direct-answer strength
            match. Defaults to the frozen 0.02.

    Returns:
        ``{"passed": bool, "clauses": {...}, "failed_clauses": [...],
        "missing_fields": [...], "not_applicable": [...]}``. A clause backed by
        an absent field is **False**, and its field name appears in
        ``missing_fields``.

    Raises:
        InstrumentIntegrityRefused: If ``layers`` is empty, which would make
            the layer clause vacuously true.
    """
    expected_layers = [int(layer) for layer in layers]
    if not expected_layers:
        raise InstrumentIntegrityRefused(
            "an integrity check needs the expected layer list; an empty band "
            "would pass every trial vacuously"
        )
    tolerance = float(tolerance)
    missing: list[str] = []
    not_applicable: list[str] = []
    clauses: dict[str, bool] = {}

    def flag_clause(clause: str, field: str) -> None:
        value = _flag(row, field)
        if value is None:
            missing.append(field)
            clauses[clause] = False
        else:
            clauses[clause] = value

    def bound_clause(clause: str, field: str, limit: float) -> None:
        value = _number(row, field)
        if value is None:
            missing.append(field)
            clauses[clause] = False
        else:
            clauses[clause] = value <= limit

    patched = row.get("layers_patched")
    clauses["expected_layers_patched"] = (
        isinstance(patched, (list, tuple))
        and [int(layer) for layer in patched] == expected_layers
    )
    if patched is None:
        missing.append("layers_patched")

    flag_clause("all_prompt_positions_patched", "all_prompt_positions_patched")
    flag_clause("all_hooks_fired", "all_hooks_fired")
    flag_clause("all_finite", "all_finite")

    # The alpha=0 parity arm runs the identical code path with a multiplication
    # by exactly zero, so "the alpha=1 exchange was exact before the cast" is
    # not a statement about it. Recorded as not-applicable rather than passed.
    alpha = _number(row, "alpha")
    if str(row.get("condition")) == "direct_answer":
        not_applicable.append("alpha_one_exact_exchange_before_cast")
        clauses["alpha_one_exact_exchange_before_cast"] = True
    elif alpha is None:
        missing.append("alpha")
        clauses["alpha_one_exact_exchange_before_cast"] = False
    elif alpha == 1.0:
        flag_clause(
            "alpha_one_exact_exchange_before_cast",
            "all_layers_are_exact_alpha_one_exchange_before_cast",
        )
    else:
        not_applicable.append("alpha_one_exact_exchange_before_cast")
        clauses["alpha_one_exact_exchange_before_cast"] = True

    flag_clause(
        "model_dtype_realization_converged",
        "all_model_dtype_realizations_converged",
    )

    # The direct-answer control has no coordinate exchange to audit; what it
    # must realize in model dtype is the *matched update norm*, and its own
    # field carries that. Both arms are gated at the same tolerance.
    if str(row.get("condition")) == "direct_answer":
        bound_clause(
            "post_cast_coordinate_error_within_tolerance",
            "max_relative_norm_match_error",
            tolerance,
        )
        not_applicable.append("post_cast_residual_drift_within_tolerance")
        clauses["post_cast_residual_drift_within_tolerance"] = True
        bound_clause(
            "cumulative_band_displacement_norm_matched",
            "max_relative_cumulative_band_displacement_match_error",
            tolerance,
        )
    else:
        bound_clause(
            "post_cast_coordinate_error_within_tolerance",
            "max_coordinate_update_error",
            tolerance,
        )
        bound_clause(
            "post_cast_residual_drift_within_tolerance",
            "max_orthogonal_residual_drift",
            tolerance,
        )
        not_applicable.append("cumulative_band_displacement_norm_matched")
        clauses["cumulative_band_displacement_norm_matched"] = True

    bound_clause(
        "activation_norm_ratio_within_limit",
        "max_activation_norm_ratio",
        float(max_activation_norm_ratio),
    )
    bound_clause(
        "update_to_activation_ratio_within_limit",
        "max_update_to_activation_norm_ratio",
        float(max_update_ratio),
    )

    teacher_forced = _flag(row, "teacher_forcing_used")
    candidates = _flag(row, "candidate_list_supplied")
    if teacher_forced is None:
        missing.append("teacher_forcing_used")
    if candidates is None:
        missing.append("candidate_list_supplied")
    clauses["no_teacher_forcing"] = teacher_forced is False
    clauses["no_candidate_list"] = candidates is False

    failed = [name for name in INTEGRITY_CLAUSES if not clauses[name]]
    return {
        "version": INSTRUMENT_VERSION,
        "passed": not failed,
        "clauses": {name: clauses[name] for name in INTEGRITY_CLAUSES},
        "failed_clauses": failed,
        "missing_fields": sorted(set(missing)),
        "not_applicable": not_applicable,
        "tolerance": tolerance,
    }


def cell_integrity(
    rows: Sequence[Mapping],
    *,
    layers: Sequence[int],
    tolerance: float = POST_CAST_TOLERANCE,
    max_activation_norm_ratio: float = 1.25,
    max_update_ratio: float = 0.50,
) -> dict:
    """Aggregate :func:`trial_integrity` over every trial in one cell.

    An empty cell fails. A cell whose rows were never measured is not a cell
    with a clean instrument; it is a cell with no evidence.
    """
    scored = [
        trial_integrity(
            row,
            layers=layers,
            tolerance=tolerance,
            max_activation_norm_ratio=max_activation_norm_ratio,
            max_update_ratio=max_update_ratio,
        )
        for row in rows
    ]
    failed: dict[str, int] = {}
    missing: set[str] = set()
    for record in scored:
        for clause in record["failed_clauses"]:
            failed[clause] = failed.get(clause, 0) + 1
        missing.update(record["missing_fields"])
    return {
        "version": INSTRUMENT_VERSION,
        "n_trials": len(scored),
        "passed": bool(scored) and all(record["passed"] for record in scored),
        "n_failing_trials": sum(1 for record in scored if not record["passed"]),
        "failed_clause_counts": dict(sorted(failed.items())),
        "missing_fields": sorted(missing),
        "tolerance": float(tolerance),
    }


def instrument_state(
    *,
    integrity_passed: bool,
    controls_moved: bool,
    effect_present: bool,
    direct_answer_available: bool,
    direct_answer_passed: bool | None,
) -> str:
    """Name one direction's outcome, keeping a broken instrument out of science.

    Args:
        integrity_passed: Every clause of :data:`INTEGRITY_CLAUSES` held for
            every trial in every cell of this direction.
        controls_moved: A control condition produced the target answer often
            enough to breach the predeclared margin.
        effect_present: The exact alpha=1 exchange cleared the predeclared
            success rate in **every** modality.
        direct_answer_available: The norm-matched direct-answer positive
            control was actually run for this direction.
        direct_answer_passed: Its outcome, or ``None`` when it was not run.

    Returns:
        One of :data:`INSTRUMENT_STATES`.

    The ordering is deliberate and is the whole point of the function. Broken
    instruments are named first, so no combination of later flags can produce
    ``SCIENTIFIC_NULL`` from a run whose intervention was not realized. A
    passing direct-answer arm is required to *read* a null and is incapable of
    creating a GO: ``effect_present`` is the only route to ``EFFECT_GO``.
    """
    if not integrity_passed:
        return "INSTRUMENT_FAILURE"
    if controls_moved:
        return "CONTROL_FAILURE"
    if effect_present:
        return "EFFECT_GO"
    if not direct_answer_available or direct_answer_passed is None:
        return "INCONCLUSIVE"
    return "SCIENTIFIC_NULL" if direct_answer_passed else "INCONCLUSIVE"
