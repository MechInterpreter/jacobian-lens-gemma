# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Derived stage gates that cannot go stale.

The notebook has eight raw switches in section 2 and four derived gates
(``MODEL_STAGES_ENABLED``, ``STAGE_A_ENABLED``, ``STAGE_B_REQUESTED``,
``STAGE_C_REQUESTED``). The derived gates used to be computed once, in the
budget cell, and read many cells later. A Jupyter kernel keeps whatever was
executed last, so editing section 2 and executing it — the documented way to
turn a stage on — left the *raw* switches correct and the *derived* gates
stale. Section 15 then printed ``skipped: STAGE_B_REQUESTED is False`` while
``RUN_L35_CAUSAL_STAGE`` was plainly True in the cell above it.

The fix is not a repair cell. Every cell that decides whether to spend model
passes recomputes its gate from the raw switches immediately beforehand, so the
only way to get a stale gate is to not run the cell at all.

:func:`refresh_stage_gates` reads the raw switches out of a namespace, derives
the gates, writes them **back** into that namespace, and returns them. Calling
it at the top of the Stage B and Stage C cells makes the derived names
unconditionally current at the moment they matter.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping

#: The switches a human sets by hand in section 2. Nothing else is an input.
RAW_SWITCHES: tuple[str, ...] = (
    "RUN_REAL_AUDIO_TRANSFER",
    "RUN_MODEL_STAGES",
    "CONFIRM_MODEL_LOAD",
    "CONFIRM_REPRESENTATION_BUDGET",
    "RUN_L35_CAUSAL_STAGE",
    "CONFIRM_L35_CAUSAL_BUDGET",
    "RUN_L38_L40_REPLICATION",
    "CONFIRM_REPLICATION_BUDGET",
)

#: The names :func:`derive_stage_gates` produces, in dependency order.
DERIVED_GATES: tuple[str, ...] = (
    "MODEL_STAGES_ENABLED",
    "STAGE_A_ENABLED",
    "STAGE_B_REQUESTED",
    "STAGE_C_REQUESTED",
)

STAGE_GATE_RULE_VERSION = "mmpilot.stage_gates.v1"


class MissingStageSwitch(RuntimeError):
    """A raw switch is absent from the namespace the gates were derived from.

    Refused rather than defaulted to False: a missing switch means section 2 was
    never executed in this kernel, and quietly reporting "not requested" is the
    same failure mode in a different disguise.
    """


def derive_stage_gates(switches: Mapping[str, object]) -> dict[str, bool]:
    """Derive the four gates from the eight raw switches. Pure.

    Raises:
        MissingStageSwitch: If any name in :data:`RAW_SWITCHES` is absent.
    """
    missing = [name for name in RAW_SWITCHES if name not in switches]
    if missing:
        raise MissingStageSwitch(
            "the stage gates cannot be derived because these raw switches are "
            f"not defined: {missing}. Execute section 2 in this kernel first — "
            "the gates are derived from the switches every time they are used, "
            "never carried over from an earlier execution."
        )
    values = {name: bool(switches[name]) for name in RAW_SWITCHES}
    model_stages = values["RUN_MODEL_STAGES"] and values["CONFIRM_MODEL_LOAD"]
    stage_a = model_stages and values["CONFIRM_REPRESENTATION_BUDGET"]
    stage_b = (
        stage_a
        and values["RUN_L35_CAUSAL_STAGE"]
        and values["CONFIRM_L35_CAUSAL_BUDGET"]
    )
    stage_c = (
        stage_b
        and values["RUN_L38_L40_REPLICATION"]
        and values["CONFIRM_REPLICATION_BUDGET"]
    )
    return {
        "MODEL_STAGES_ENABLED": bool(model_stages),
        "STAGE_A_ENABLED": bool(stage_a),
        "STAGE_B_REQUESTED": bool(stage_b),
        "STAGE_C_REQUESTED": bool(stage_c),
    }


def refresh_stage_gates(namespace: MutableMapping[str, object]) -> dict[str, bool]:
    """Recompute the gates from ``namespace``'s raw switches and write them back.

    Call this with ``globals()`` at the top of any cell that decides whether to
    spend model passes. After it returns, the four derived names in the
    namespace are exactly what the raw switches imply — whatever they held
    before, and regardless of which cell last assigned them.
    """
    gates = derive_stage_gates(namespace)
    namespace.update(gates)
    return gates


def format_stage_gates(gates: Mapping[str, object], *, switches: Mapping[str, object]) -> str:
    """The block a cell prints to show the gates it just re-derived."""
    lines = [
        "=" * 72,
        f"STAGE GATES (re-derived now from the raw switches — {STAGE_GATE_RULE_VERSION})",
        "=" * 72,
    ]
    lines += [f"  {name:31s} {bool(switches[name])}" for name in RAW_SWITCHES]
    lines.append("")
    lines += [f"  {name:31s} {bool(gates[name])}" for name in DERIVED_GATES]
    return "\n".join(lines)


__all__ = [
    "DERIVED_GATES",
    "RAW_SWITCHES",
    "STAGE_GATE_RULE_VERSION",
    "MissingStageSwitch",
    "derive_stage_gates",
    "format_stage_gates",
    "refresh_stage_gates",
]
