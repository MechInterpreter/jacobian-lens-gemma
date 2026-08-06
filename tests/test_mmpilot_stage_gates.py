# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Derived stage gates, and the staleness that used to survive a re-execution.

The observed failure: section 2 was edited so ``RUN_L35_CAUSAL_STAGE`` and
``CONFIRM_L35_CAUSAL_BUDGET`` were True, and that cell was executed — and
section 15 still printed ``skipped: STAGE_B_REQUESTED is False``, because
``STAGE_B_REQUESTED`` had been computed in the budget cell and left behind in
the kernel.

These tests do not check "the gates are correct once". They check that a
namespace already holding *wrong* derived values cannot keep them.
"""

import ast
import json
from pathlib import Path

import pytest

from jlens.mmpilot.stage_gates import (
    DERIVED_GATES,
    RAW_SWITCHES,
    MissingStageSwitch,
    derive_stage_gates,
    format_stage_gates,
    refresh_stage_gates,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = (
    REPO_ROOT / "notebooks" / "multimodal_jspace_spokencoco_native_audio_colab.ipynb"
)

ALL_OFF = dict.fromkeys(RAW_SWITCHES, False)
STAGE_B_ON = {
    **ALL_OFF,
    "RUN_MODEL_STAGES": True,
    "CONFIRM_MODEL_LOAD": True,
    "CONFIRM_REPRESENTATION_BUDGET": True,
    "RUN_L35_CAUSAL_STAGE": True,
    "CONFIRM_L35_CAUSAL_BUDGET": True,
}
STAGE_C_ON = {
    **STAGE_B_ON,
    "RUN_L38_L40_REPLICATION": True,
    "CONFIRM_REPLICATION_BUDGET": True,
}


def test_the_committed_defaults_request_nothing():
    gates = derive_stage_gates(ALL_OFF)
    assert set(gates) == set(DERIVED_GATES)
    assert not any(gates.values())


def test_each_stage_needs_every_switch_below_it():
    assert derive_stage_gates(STAGE_B_ON)["STAGE_B_REQUESTED"] is True
    assert derive_stage_gates(STAGE_B_ON)["STAGE_C_REQUESTED"] is False
    assert derive_stage_gates(STAGE_C_ON)["STAGE_C_REQUESTED"] is True
    # Stage B without its Stage A prerequisite is not requested.
    partial = {**STAGE_B_ON, "CONFIRM_REPRESENTATION_BUDGET": False}
    assert derive_stage_gates(partial)["STAGE_B_REQUESTED"] is False
    # Nor without its own confirmation.
    unconfirmed = {**STAGE_B_ON, "CONFIRM_L35_CAUSAL_BUDGET": False}
    assert derive_stage_gates(unconfirmed)["STAGE_B_REQUESTED"] is False


def test_a_stale_stage_b_gate_cannot_survive_changed_raw_switches():
    # A kernel that derived the gates while everything was off, then had
    # section 2 edited and re-executed. The raw switches are current; the
    # derived names are the stale ones the old bug read.
    namespace = {**ALL_OFF, **{name: False for name in DERIVED_GATES}}
    namespace.update(STAGE_B_ON)
    assert namespace["STAGE_B_REQUESTED"] is False  # stale, and wrong

    gates = refresh_stage_gates(namespace)

    assert gates["STAGE_B_REQUESTED"] is True
    assert namespace["STAGE_B_REQUESTED"] is True
    assert namespace["STAGE_A_ENABLED"] is True
    assert namespace["MODEL_STAGES_ENABLED"] is True


def test_a_stale_stage_c_gate_cannot_survive_changed_raw_switches():
    namespace = {**ALL_OFF, **{name: False for name in DERIVED_GATES}}
    namespace.update(STAGE_C_ON)
    assert namespace["STAGE_C_REQUESTED"] is False  # stale, and wrong

    refresh_stage_gates(namespace)

    assert namespace["STAGE_C_REQUESTED"] is True


def test_staleness_is_corrected_downward_too():
    # The dangerous direction is a gate that says False while the switches say
    # True. The reverse — a leftover True after the switches were turned back
    # off — would spend model passes nobody confirmed, so it is corrected as
    # well.
    namespace = {**ALL_OFF, **dict.fromkeys(DERIVED_GATES, True)}
    refresh_stage_gates(namespace)
    assert not any(namespace[name] for name in DERIVED_GATES)


def test_refreshing_is_idempotent_and_depends_on_nothing_but_the_raw_switches():
    namespace = {**STAGE_C_ON, **dict.fromkeys(DERIVED_GATES, False)}
    first = refresh_stage_gates(namespace)
    second = refresh_stage_gates(namespace)
    assert first == second
    # Derived names are outputs, never inputs: deleting them changes nothing.
    for name in DERIVED_GATES:
        namespace.pop(name)
    assert refresh_stage_gates(namespace) == first


def test_a_missing_raw_switch_is_refused_rather_than_defaulted_to_false():
    incomplete = dict(ALL_OFF)
    incomplete.pop("CONFIRM_L35_CAUSAL_BUDGET")
    with pytest.raises(MissingStageSwitch, match="CONFIRM_L35_CAUSAL_BUDGET"):
        derive_stage_gates(incomplete)
    with pytest.raises(MissingStageSwitch, match="Execute section 2"):
        refresh_stage_gates(incomplete)


def test_the_printed_block_shows_both_the_switches_and_what_they_imply():
    text = format_stage_gates(derive_stage_gates(STAGE_B_ON), switches=STAGE_B_ON)
    assert "re-derived now from the raw switches" in text
    for name in (*RAW_SWITCHES, *DERIVED_GATES):
        assert name in text


# ------------------------------------------------------- the notebook itself


def _code_cells():
    payload = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in payload["cells"] if c["cell_type"] == "code"]


def _cell_index(cells, needle):
    return next(i for i, source in enumerate(cells) if needle in source)


def test_every_cell_that_can_spend_model_passes_re_derives_its_own_gate():
    cells = _code_cells()
    stage_b = cells[_cell_index(cells, "STAGE_B_REQUESTED is False")]
    stage_c = cells[_cell_index(cells, "STAGE_C_REQUESTED is False")]
    for source, gate in ((stage_b, "STAGE_B_REQUESTED"), (stage_c, "STAGE_C_REQUESTED")):
        assert "refresh_stage_gates(globals())" in source
        # The refresh happens before the decision, not after it.
        assert source.index("refresh_stage_gates(globals())") < source.index(
            f"if not {gate}"
        )


def test_no_cell_computes_a_derived_gate_by_hand():
    # One rule, in one tested place. An expression written into a cell is how
    # the gate and the documented switch semantics drifted apart before.
    for source in _code_cells():
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            targets = {t.id for t in node.targets if isinstance(t, ast.Name)}
            if targets & set(DERIVED_GATES):
                raise AssertionError(
                    f"{sorted(targets & set(DERIVED_GATES))} is assigned by hand; "
                    "derive it with refresh_stage_gates instead"
                )


def test_the_notebook_never_needs_a_manual_repair_cell():
    source = "\n".join(_code_cells())
    for banned in ("repair cell", "re-run section 7", "manually set STAGE_"):
        assert banned not in source
