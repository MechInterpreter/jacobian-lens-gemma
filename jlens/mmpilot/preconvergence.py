# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""The L27–L31 pre-convergence transition study: multimodal half.

:mod:`jlens.calibration.adjacent` decides whether any of the five predeclared
candidates has a *confirmed* J-lens. This module decides what happens next, and
it exists mainly to stop four specific ways of getting a favourable answer for
the wrong reason.

**1. Combining halves that are about different photographs.** The principal
claim needs one and the same physical layer to have a confirmed lens, a
``NOT_CONVERGED`` native readout, and controlled cross-modal causal transfer —
all three on the *same* independent multimodal population.
:func:`preconvergence_verdicts` refuses to assemble a claim out of a convergence
result from one population and a causal result from another, and
:func:`assert_same_population` is what makes that mechanical rather than
remembered.

**2. Reusing spent media.** Three completed populations are already spent, and
the third of them cannot be named as a constant because it was selected by the
operator: :func:`assert_completed_population_pins` requires the ``mml32res_*``
run directory to be *set explicitly* and refuses to discover "the newest" one.

**3. Letting a stale gate spend an L4 hour.** Every derived gate is a pure
function of the raw switches (:func:`derive_preconvergence_gates`) and is
re-derived inside each expensive cell (:func:`refresh_preconvergence_gates`).

**4. Renaming steering as a coordinate swap.** The intervention family here is
the completed open-prompt follow-up's additive J-space steering, reused
unchanged for comparability. The Anthropic two-coordinate swap is a different
operation, it needs a contiguous confirmed layer band, and
:data:`COORDINATE_SWAP_SCOPE` says so in every artifact.

Every terminal outcome in :data:`TERMINAL_OUTCOMES` is first-class. ``NO_GO``,
``AMBIGUOUS``, ``CONVERGED`` and ``REFUSED_INVALID`` are reported in the same
words and at the same volume as support.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path

from jlens.calibration.adjacent import (
    ADJACENT_CANDIDATE_LAYERS,
    ADJACENT_FITTING_SCALE,
    ADJACENT_HOOK_SITE,
    ADJACENT_LENS_GO,
    ADJACENT_LENS_NO_GO,
    ADJACENT_PROTOCOL,
    AMBIGUOUS_UPPER_LAYER,
    CONVERGED_REFERENCE_LAYER,
    FAILED_LOWER_LAYER,
)
from jlens.mmpilot.convergence import (
    AMBIGUOUS,
    CONTROL_VARIANTS,
    CONVERGED,
    CONVERGENCE_CRITERION,
    NOT_CONVERGED,
)
from jlens.mmpilot.l32_followup import (
    INTERVENTION_FAMILY,
    OPEN_PROMPT_PROTOCOL,
)
from jlens.mmpilot.l32_resolution import FROZEN_CRITERION_DIGEST
from jlens.mmpilot.store import payload_checksum

__all__ = [
    "COMPLETED_AUDIO_TRANSFER_RUN",
    "adjacent_lens_integrity",
    "check_preconvergence_call_contracts",
    "preconvergence_call_contracts",
    "COMPLETED_FOLLOWUP_RUN",
    "COORDINATE_SWAP_SCOPE",
    "FROZEN_CRITERION_DIGEST",
    "PRECONVERGENCE_PROTOCOL",
    "PRECONVERGENCE_RAW_SWITCHES",
    "PRECONVERGENCE_RUN_PREFIX",
    "REQUIRED_CAUSAL_CONTROLS",
    "REQUIRED_MODALITIES",
    "STAGE_FOUR_RULE",
    "STAGE_PLAN_VERSION",
    "TERMINAL_OUTCOMES",
    "VERDICT_NAMES",
    "PinNotSet",
    "PopulationsDiffer",
    "PreconvergenceRefused",
    "assert_causal_controls_recorded",
    "assert_completed_population_pins",
    "assert_fresh_run_namespace",
    "assert_same_population",
    "build_summary",
    "convergence_verdict_for_layer",
    "derive_preconvergence_gates",
    "format_preconvergence_gates",
    "format_stage_plan",
    "preconvergence_fingerprint",
    "preconvergence_verdicts",
    "refresh_preconvergence_gates",
    "render_report",
    "stage_four_decision",
    "stage_plan",
]

# ------------------------------------------------------------------ versions

PRECONVERGENCE_PROTOCOL = "mmpilot.l27_l31_preconvergence_study.v1"

#: Run directories this study creates. Distinct from every completed family
#: (``rgcal_``, ``rgext_``, ``mmaudio_``, ``mmrobust_``, ``mmlocalize_``,
#: ``mmconv_``, ``mml32_``, ``mml32res_``), so a resume cannot land in one.
PRECONVERGENCE_RUN_PREFIX = "mmpre"

STAGE_PLAN_VERSION = "mmpilot.l27_l31_preconvergence_stage_plan.v1"
POPULATION_SELECTION_VERSION = "mmpilot.l27_l31_independent_population.v1"

REQUIRED_MODALITIES: tuple[str, ...] = ("text", "image", "spoken_audio")

# ------------------------------------------------ the populations to avoid

#: Named because they are fixed facts of the record.
COMPLETED_AUDIO_TRANSFER_RUN = "mmaudio_native_audio_transfer_20260806T144822"
COMPLETED_FOLLOWUP_RUN = "mml32_l32_followup_20260808T182717"

#: The prefix of the third spent population. **Not** a run name: which
#: ``mml32res_*`` run is the completed convergence-resolution study is a fact
#: only the operator holds, and this study will not guess it.
COMPLETED_RESOLUTION_RUN_PREFIX = "mml32res"

COORDINATE_SWAP_SCOPE = (
    "OUT OF SCOPE. The intervention family here is additive J-space residual "
    "steering — the same family the open-prompt L32 follow-up used, reused "
    "unchanged so the two are comparable. An Anthropic two-coordinate swap is a "
    "different operation requiring a CONTIGUOUS confirmed layer band, which does "
    "not exist today. Nothing in this study is described as a swap."
)

#: The causal controls the protocol requires. A missing record is a failure.
REQUIRED_CAUSAL_CONTROLS: tuple[str, ...] = (
    "matched_random_direction",
    "external_unrelated_concept",
    "shuffled_permuted_control",
    "zero_intervention",
    "activation_norm_sanity",
    "target_specificity_global_disruption",
    "image_level_independent_aggregation",
)

# ------------------------------------------------------------------ verdicts

VERDICT_NAMES: tuple[str, ...] = (
    "ADJACENT_LENS_VALIDITY",
    "EARLIEST_CONFIRMED_LAYER",
    "NATIVE_OUTPUT_CONVERGENCE",
    "THREE_MODALITY_CAUSAL_TRANSFER",
    "PRECONVERGENCE_CAUSAL_TRANSFER",
)

PRECONVERGENCE_SUPPORTED = "PRECONVERGENCE_CAUSAL_TRANSFER_SUPPORTED"
CONVERGED_BEFORE_CAUSAL_TEST = "CONVERGED_BEFORE_CAUSAL_TEST"
AMBIGUOUS_CONVERGENCE = "AMBIGUOUS_CONVERGENCE"
CAUSAL_TRANSFER_NOT_SUPPORTED = "CAUSAL_TRANSFER_NOT_SUPPORTED"
REFUSED_INVALID = "REFUSED_INVALID"

#: Every terminal outcome, and no others. Each is reported identically.
TERMINAL_OUTCOMES: tuple[str, ...] = (
    PRECONVERGENCE_SUPPORTED,
    ADJACENT_LENS_NO_GO,
    CONVERGED_BEFORE_CAUSAL_TEST,
    AMBIGUOUS_CONVERGENCE,
    CAUSAL_TRANSFER_NOT_SUPPORTED,
    REFUSED_INVALID,
)

#: Native-convergence verdict values.
LAYER_CONVERGED = "LAYER_CONVERGED"
LAYER_NOT_CONVERGED = "LAYER_NOT_CONVERGED"
LAYER_AMBIGUOUS = "LAYER_AMBIGUOUS"

_CLASSIFICATION_TO_VERDICT = {
    CONVERGED: LAYER_CONVERGED,
    NOT_CONVERGED: LAYER_NOT_CONVERGED,
    AMBIGUOUS: LAYER_AMBIGUOUS,
}

TRANSFER_SUPPORTED = "SUPPORTED"
TRANSFER_NOT_SUPPORTED = "NOT_SUPPORTED"
TRANSFER_NOT_EVALUATED = "NOT_EVALUATED"
TRANSFER_DESCRIPTIVE_ONLY = "DESCRIPTIVE_ONLY"


class PreconvergenceRefused(RuntimeError):
    """A precondition of the pre-convergence study does not hold."""


class PinNotSet(PreconvergenceRefused):
    """A completed population the study must avoid was not pinned explicitly."""


class PopulationsDiffer(PreconvergenceRefused):
    """Two halves of the principal claim are about different populations."""


# ---------------------------------------------------------------- the pins


def assert_completed_population_pins(
    run_dirs: Sequence[str | os.PathLike[str]],
    *,
    resolution_run_dir: str | os.PathLike[str] | None,
    required_basenames: Sequence[str] = (
        COMPLETED_AUDIO_TRANSFER_RUN,
        COMPLETED_FOLLOWUP_RUN,
    ),
    resolution_prefix: str = COMPLETED_RESOLUTION_RUN_PREFIX,
) -> dict:
    """Require every spent population to be named, and the pinned one to be set.

    The two named runs are facts of the record and are checked by basename. The
    convergence-resolution run is different in kind: several ``mml32res_*``
    directories can exist on a Drive (a preprocessing session, an abandoned
    attempt, the completed study), and only the operator knows which one holds
    the completed result. Discovering "the newest" would silently pick a
    directory whose media this study then treats as unspent.

    Raises:
        PinNotSet: If ``resolution_run_dir`` is empty, does not start with the
            expected prefix, or is not among ``run_dirs``; or if either named
            run is absent from ``run_dirs``.
    """
    basenames = [Path(str(directory)).name for directory in run_dirs]
    missing = [name for name in required_basenames if name not in basenames]
    if missing:
        raise PinNotSet(
            f"these completed populations are not in the exclusion list: "
            f"{missing}. Every population already consumed must be excluded; "
            "one omitted run means a photograph this study calls independent "
            "was already spent."
        )

    if not resolution_run_dir or not str(resolution_run_dir).strip():
        raise PinNotSet(
            "the completed convergence-resolution run is not pinned. Set "
            "COMPLETED_RESOLUTION_RUN_DIR to the exact "
            f"{resolution_prefix}_* directory that holds the completed study.\n"
            "It is not discovered and not defaulted: several such directories "
            "can exist, and picking the newest would quietly treat a spent "
            "population as available."
        )
    pinned = Path(str(resolution_run_dir))
    if not pinned.name.startswith(resolution_prefix):
        raise PinNotSet(
            f"the pinned convergence-resolution run {pinned.name!r} does not "
            f"start with {resolution_prefix!r}. Refusing rather than guessing "
            "which directory was meant."
        )
    if pinned.name not in basenames:
        raise PinNotSet(
            f"the pinned run {pinned.name!r} is not in the exclusion list "
            f"{basenames}. A pin that is not excluded excludes nothing."
        )

    payload = {
        "schema": "jlens.mmpilot.preconvergence_population_pins.v1",
        "required_named_runs": list(required_basenames),
        "pinned_resolution_run": pinned.name,
        "pinned_resolution_run_dir": str(pinned),
        "pin_was_discovered": False,
        "pin_was_defaulted": False,
        "excluded_run_basenames": sorted(basenames),
        "n_excluded_runs": len(basenames),
        "why_the_pin_is_manual": (
            "several mml32res_* directories can exist; only the operator knows "
            "which holds the completed study, and choosing the newest would "
            "treat a spent population as available"
        ),
    }
    payload["pins_checksum"] = payload_checksum(payload)
    return payload


def assert_fresh_run_namespace(
    run_dir: str | os.PathLike[str], *, protected_prefixes: Sequence[str]
) -> str:
    """Refuse a run directory inside any completed run's namespace.

    Raises:
        PreconvergenceRefused: When the path passes through a protected prefix,
            or when its own name does not start with
            :data:`PRECONVERGENCE_RUN_PREFIX`.
    """
    root = Path(run_dir)
    offending = sorted(
        {
            prefix
            for prefix in protected_prefixes
            for part in root.parts
            if part.startswith(prefix)
        }
    )
    if offending:
        raise PreconvergenceRefused(
            f"{root} is inside a completed run namespace ({offending}). "
            "Completed runs are evidence and are never written into; this study "
            f"creates its own {PRECONVERGENCE_RUN_PREFIX}_* directory."
        )
    if not root.name.startswith(PRECONVERGENCE_RUN_PREFIX):
        raise PreconvergenceRefused(
            f"{root.name!r} is not a {PRECONVERGENCE_RUN_PREFIX}_* directory. "
            "A run that does not name its own family can be resumed by the "
            "wrong study."
        )
    return str(root)


# ------------------------------------------------------------- stage gates

#: The switches a human sets by hand. Nothing else is an input to a gate.
PRECONVERGENCE_RAW_SWITCHES: tuple[str, ...] = (
    "RUN_REAL_PRECONVERGENCE_STUDY",
    "PREPROCESSING_ONLY",
    "RUN_LENS_FITTING",
    "CONFIRM_FITTING_BUDGET",
    "RUN_UNTOUCHED_CONFIRMATION",
    "RUN_MODEL_STAGE",
    "CONFIRM_MODEL_LOAD",
    "CONFIRM_STAGE_3_BUDGET",
    "RUN_STAGE_4_CAUSAL_TRANSFER",
    "CONFIRM_STAGE_4_BUDGET",
)

PRECONVERGENCE_DERIVED_GATES: tuple[str, ...] = (
    "FITTING_ENABLED",
    "CONFIRMATION_ENABLED",
    "MODEL_STAGE_ENABLED",
    "STAGE_3_ENABLED",
    "STAGE_4_REQUESTED",
)

GATE_RULE_VERSION = "mmpilot.l27_l31_preconvergence_gates.v1"


def derive_preconvergence_gates(switches: Mapping[str, object]) -> dict[str, bool]:
    """Derive every gate from the raw switches. Pure, and total.

    A Jupyter kernel keeps whatever ran last, so a gate computed once and read
    ten cells later goes stale the moment the switch cell is re-run. Each
    expensive cell calls :func:`refresh_preconvergence_gates` immediately before
    it can spend anything.

    Raises:
        MissingStageSwitch: If any raw switch is absent, which means the switch
            cell was never executed in this kernel. Defaulting to ``False``
            would print "not requested" for a switch just set by hand.
    """
    from jlens.mmpilot.stage_gates import MissingStageSwitch

    missing = [name for name in PRECONVERGENCE_RAW_SWITCHES if name not in switches]
    if missing:
        raise MissingStageSwitch(
            "the pre-convergence stage gates cannot be derived because these "
            f"raw switches are not defined: {missing}. Execute the switch cell "
            "in this kernel first."
        )
    values = {name: bool(switches[name]) for name in PRECONVERGENCE_RAW_SWITCHES}

    if values["PREPROCESSING_ONLY"]:
        # One switch, and it closes everything that could load a model or spend
        # a fitting hour. Overloading the MOCK/real switch to mean "CPU session"
        # instead would make a preparation run indistinguishable from a MOCK run
        # in the artifacts.
        return dict.fromkeys(PRECONVERGENCE_DERIVED_GATES, False)

    fitting = values["RUN_LENS_FITTING"] and values["CONFIRM_FITTING_BUDGET"]
    confirmation = values["RUN_UNTOUCHED_CONFIRMATION"]
    model_stage = values["RUN_MODEL_STAGE"] and values["CONFIRM_MODEL_LOAD"]
    stage_3 = model_stage and values["CONFIRM_STAGE_3_BUDGET"]
    stage_4 = (
        stage_3
        and values["RUN_STAGE_4_CAUSAL_TRANSFER"]
        and values["CONFIRM_STAGE_4_BUDGET"]
    )
    return {
        "FITTING_ENABLED": bool(fitting),
        "CONFIRMATION_ENABLED": bool(confirmation),
        "MODEL_STAGE_ENABLED": bool(model_stage),
        "STAGE_3_ENABLED": bool(stage_3),
        "STAGE_4_REQUESTED": bool(stage_4),
    }


def refresh_preconvergence_gates(namespace) -> dict[str, bool]:
    """Recompute the gates from ``namespace`` and write them back into it."""
    gates = derive_preconvergence_gates(namespace)
    namespace.update(gates)
    return gates


def format_preconvergence_gates(gates: Mapping, *, switches: Mapping) -> str:
    """The block a cell prints to show the gates it just re-derived."""
    lines = [
        "=" * 72,
        f"STAGE GATES (re-derived now — {GATE_RULE_VERSION})",
        "=" * 72,
    ]
    lines += [
        f"  {name:38s} {bool(switches[name])}" for name in PRECONVERGENCE_RAW_SWITCHES
    ]
    lines.append("")
    lines += [
        f"  {name:38s} {bool(gates[name])}" for name in PRECONVERGENCE_DERIVED_GATES
    ]
    return "\n".join(lines)


# -------------------------------------------------------------- stage plan

STAGE_FOUR_RULE = (
    "Stage 4 runs only when ALL of: (a) a candidate layer passed untouched lens "
    "confirmation; (b) that same layer is NOT_CONVERGED in every required "
    "modality under the frozen criterion; (c) every convergence control passed; "
    "(d) behavioral capability is sufficient under the open prompt. Otherwise it "
    "is skipped. An explicit override may run it anyway; overridden causal "
    "results are labelled DESCRIPTIVE_ONLY and never support the principal claim."
)

STAGE_FOUR_RATIONALE = (
    "Stage 4's only purpose is the principal claim — controlled cross-modal "
    "causal transfer BEFORE native direct-readout convergence. That claim needs "
    "the convergence half to be NOT_CONVERGED. Under CONVERGED it is dead at "
    "this layer whatever the causal passes show; under AMBIGUOUS it is "
    "unsupported whatever they show. The gate therefore withholds passes exactly "
    "where the headline result is UNFAVOURABLE to the hypothesis, and those "
    "outcomes are reported in full. It is an efficiency gate, not a filter."
)


def stage_plan() -> dict:
    """The five stages and their gates, fixed before any of them opens."""
    return {
        "schema": "jlens.mmpilot.preconvergence_stage_plan.v1",
        "stage_plan_version": STAGE_PLAN_VERSION,
        "stages": [
            {
                "stage": 0,
                "name": "cpu_preparation",
                "runs": "always",
                "loads_model": False,
                "contents": [
                    "corpus and split provenance for the adjacent fit",
                    "untouched confirmation construction and its audit",
                    "SpokenCOCO exclusion harvest over every spent population",
                    "independent multimodal population preparation, checkpointed",
                ],
            },
            {
                "stage": 1,
                "name": "adjacent_layer_lens_fitting",
                "runs": "gated on FITTING_ENABLED",
                "loads_model": True,
                "contents": [
                    f"fit source layers {list(ADJACENT_CANDIDATE_LAYERS)} at scale "
                    f"{ADJACENT_FITTING_SCALE} in one resumable run",
                    "atomic checkpoint per bounded batch; resume from the last "
                    "completed prompt",
                ],
            },
            {
                "stage": 2,
                "name": "development_and_untouched_confirmation",
                "runs": "gated on CONFIRMATION_ENABLED",
                "loads_model": True,
                "contents": [
                    "the frozen tie-aware validity gate on every candidate",
                    "selection of the EARLIEST fully confirmed layer",
                ],
                "stops_with": ADJACENT_LENS_NO_GO,
            },
            {
                "stage": 3,
                "name": "capability_and_native_convergence",
                "runs": "gated on STAGE_3_ENABLED",
                "loads_model": True,
                "contents": [
                    "behavioral capability under the open prompt in three modalities",
                    "final-prompt-token residual capture at the selected layer",
                    "the model's own final norm and unembedding",
                    f"classification under the frozen criterion "
                    f"({FROZEN_CRITERION_DIGEST})",
                    "the three convergence controls",
                ],
                "produces": [LAYER_CONVERGED, LAYER_NOT_CONVERGED, LAYER_AMBIGUOUS],
            },
            {
                "stage": 4,
                "name": "cross_modal_causal_transfer",
                "runs": "conditionally",
                "loads_model": True,
                "condition": STAGE_FOUR_RULE,
                "contents": [
                    f"{INTERVENTION_FAMILY} at the selected layer on the same "
                    "independent population",
                    f"required controls {list(REQUIRED_CAUSAL_CONTROLS)}",
                ],
                "forbidden": [
                    "the historical candidate-listed prompt protocol",
                    "describing additive steering as a two-coordinate swap",
                ],
            },
        ],
        "conditional_rule": STAGE_FOUR_RULE,
        "rationale": STAGE_FOUR_RATIONALE,
        "efficiency_gate_not_suppression": True,
        "all_stage_3_outcomes_reported": True,
        "intervention_family": INTERVENTION_FAMILY,
        "coordinate_swap_scope": COORDINATE_SWAP_SCOPE,
        "prompt_protocol": OPEN_PROMPT_PROTOCOL,
        "required_modalities": list(REQUIRED_MODALITIES),
    }


def format_stage_plan(plan: Mapping) -> str:
    """The plan printed before any expensive stage can be enabled."""
    lines = [
        "=" * 72,
        f"STAGE PLAN — {plan['stage_plan_version']}",
        "=" * 72,
    ]
    for stage in plan["stages"]:
        lines.append(
            f"  Stage {stage['stage']}  {stage['name']:38s} "
            f"model={stage['loads_model']}  {stage['runs']}"
        )
        for item in stage["contents"]:
            lines.append(f"           - {item}")
    lines += ["", f"  STAGE 4 RULE: {plan['conditional_rule']}", ""]
    lines.append(f"  WHY IT IS AN EFFICIENCY GATE: {plan['rationale']}")
    lines += ["", f"  COORDINATE SWAP: {plan['coordinate_swap_scope']}"]
    return "\n".join(lines)


def stage_four_decision(
    *,
    lens_verdict: str,
    convergence_verdict: str,
    controls_passed: bool,
    capability_sufficient: bool,
    requested: bool,
    budget_confirmed: bool,
) -> dict:
    """Whether Stage 4 runs, and under the gate or as an override."""
    clauses = [
        {
            "clause": "a_candidate_passed_untouched_lens_confirmation",
            "passed": lens_verdict == ADJACENT_LENS_GO,
            "detail": str(lens_verdict),
        },
        {
            "clause": "same_layer_not_converged_in_every_required_modality",
            "passed": convergence_verdict == LAYER_NOT_CONVERGED,
            "detail": str(convergence_verdict),
        },
        {
            "clause": "all_convergence_controls_passed",
            "passed": bool(controls_passed),
            "detail": f"controls_passed={bool(controls_passed)}",
        },
        {
            "clause": "sufficient_behavioral_capability",
            "passed": bool(capability_sufficient),
            "detail": f"capability_sufficient={bool(capability_sufficient)}",
        },
    ]
    gate_met = all(clause["passed"] for clause in clauses)
    runs = bool(requested and budget_confirmed)
    overridden = bool(runs and not gate_met)
    return {
        "schema": "jlens.mmpilot.preconvergence_stage_four_decision.v1",
        "stage_plan_version": STAGE_PLAN_VERSION,
        "gate_clauses": clauses,
        "failed_gate_clauses": [c["clause"] for c in clauses if not c["passed"]],
        "gate_met": gate_met,
        "requested": bool(requested),
        "budget_confirmed": bool(budget_confirmed),
        "runs": runs,
        "gate_overridden": overridden,
        "evidence_status": (
            TRANSFER_DESCRIPTIVE_ONLY if overridden else "PREDECLARED" if runs else None
        ),
        "rule": STAGE_FOUR_RULE,
        "rationale": STAGE_FOUR_RATIONALE,
        "statement": (
            "Stage 4 runs under the predeclared gate."
            if runs and gate_met
            else (
                "Stage 4 runs as an EXPLICIT OVERRIDE of the predeclared gate. "
                "Its causal numbers are DESCRIPTIVE_ONLY and never support the "
                "principal claim."
                if runs
                else (
                    "Stage 4 does not run. "
                    + (
                        "The gate is met but the run was not requested or its "
                        "budget was not confirmed."
                        if gate_met
                        else "The gate is not met: "
                        + str([c["clause"] for c in clauses if not c["passed"]])
                    )
                )
            )
        ),
    }


# --------------------------------------------- the native-convergence verdict


def convergence_verdict_for_layer(
    *,
    layer: int,
    integrity: Mapping,
    convergence: Mapping,
    controls: Mapping,
    disjointness: Mapping,
    pseudoreplication: Mapping,
    sample_plan: Mapping,
    head_agreement: Mapping,
    admissibility: Mapping,
    leakage_audit: Mapping,
) -> dict:
    """``NATIVE_OUTPUT_CONVERGENCE`` at the selected layer, or a refusal.

    Validity is established first and separately: a study whose controls did not
    run has not observed ``AMBIGUOUS``, it has observed nothing, and saying so is
    :data:`REFUSED_INVALID` rather than a hedged classification.

    The layer is read from the measurement, never from a module constant — the
    MOCK decoder stands in for a deep layer at a shallow index, and reading
    ``per_layer["29"]`` there would report a degenerate cell for one that is
    perfectly well populated.
    """
    classification_block = convergence.get("classification") or {}
    classification = classification_block.get("classification")
    criterion_digest = convergence.get("criterion_digest")
    admissible = list(admissibility.get("eligible_concepts") or [])
    summary = convergence.get("summary") or {}
    measured_layer = int(convergence.get("layer", layer))
    per_layer = (summary.get("per_layer") or {}).get(str(measured_layer)) or {}
    per_modality = per_layer.get("per_modality") or {}

    degenerate = [
        modality
        for modality in CONVERGENCE_CRITERION.required_modalities
        if int((per_modality.get(modality) or {}).get("n_distinct_predictions", 0))
        < CONVERGENCE_CRITERION.min_distinct_predictions
    ]
    undersized = [
        modality
        for modality in CONVERGENCE_CRITERION.required_modalities
        if int((per_modality.get(modality) or {}).get("n", 0))
        < CONVERGENCE_CRITERION.min_samples_per_cell
    ]

    clauses = [
        {
            "clause": "lens_integrity_passed",
            "passed": integrity.get("verdict") == "PASSED",
            "detail": str(integrity.get("verdict")),
        },
        {
            "clause": "native_head_agrees_with_model_unembed",
            "passed": bool(head_agreement.get("passed")),
            "detail": (
                f"matches={head_agreement.get('matches_model_unembed')!r} "
                f"comparison_ran={head_agreement.get('comparison_ran')!r}"
            ),
        },
        {
            "clause": "criterion_digest_unchanged",
            "passed": criterion_digest == FROZEN_CRITERION_DIGEST,
            "detail": f"{criterion_digest} vs frozen {FROZEN_CRITERION_DIGEST}",
        },
        {
            "clause": "population_disjoint_from_every_spent_population",
            "passed": bool(disjointness.get("disjoint")),
            "detail": str(disjointness.get("failed_families")),
        },
        {
            "clause": "one_unit_per_photograph",
            "passed": bool(pseudoreplication.get("passed")),
            "detail": (
                f"{pseudoreplication.get('n_units')} unit(s) on "
                f"{pseudoreplication.get('n_distinct_images')} photograph(s)"
            ),
        },
        {
            "clause": "sample_size_predeclared",
            "passed": bool(sample_plan.get("plan_digest")),
            "detail": str(sample_plan.get("plan_digest")),
        },
        {
            "clause": "no_candidate_leaked_into_any_model_visible_prompt",
            "passed": bool(leakage_audit.get("passed")),
            "detail": str(leakage_audit.get("per_modality")),
        },
        {
            "clause": "controls_present_and_passing",
            "passed": bool(controls.get("passed")),
            "detail": (
                f"missing={controls.get('missing_or_empty')} "
                f"failing={controls.get('failing')}"
            ),
        },
        {
            "clause": "outputs_finite_and_nondegenerate",
            "passed": not degenerate and not undersized,
            "detail": (
                f"single-prediction modalities {degenerate}, undersized cells "
                f"{undersized}; finiteness is enforced upstream by "
                "direct_readout_row, which refuses a non-finite row"
            ),
        },
        {
            "clause": "at_least_one_admissible_focal_concept",
            "passed": bool(admissible),
            "detail": f"admissible {admissible}",
        },
    ]
    failed = [entry["clause"] for entry in clauses if not entry["passed"]]

    if failed:
        verdict = REFUSED_INVALID
        rationale = (
            "No classification is reported. "
            f"{len(failed)} validity clause(s) did not hold: {failed}. A study "
            "whose measurement is not established has not observed AMBIGUOUS; "
            "it has observed nothing."
        )
    else:
        verdict = _CLASSIFICATION_TO_VERDICT.get(classification, REFUSED_INVALID)
        if verdict == REFUSED_INVALID:
            rationale = (
                f"the frozen criterion returned {classification!r}, which is not "
                f"one of {sorted(_CLASSIFICATION_TO_VERDICT)}"
            )
        elif verdict == LAYER_NOT_CONVERGED:
            rationale = (
                f"Physical layer {layer} is {NOT_CONVERGED} in every required "
                f"modality {list(REQUIRED_MODALITIES)} under the frozen "
                "criterion, on an independently selected population. A causal "
                "effect measured here would occur before native direct-readout "
                "convergence."
            )
        elif verdict == LAYER_CONVERGED:
            rationale = (
                f"Physical layer {layer} meets every {CONVERGED} clause in every "
                "required modality under the frozen criterion. Any causal effect "
                "at this layer occurs at or after native convergence, so the "
                "pre-convergence claim cannot be made here at all."
            )
        else:
            rationale = (
                f"Physical layer {layer} falls between the two frozen bars. An "
                "ambiguous layer is a finding, not a shortage of data: the "
                "thresholds were fixed before this measurement and are not "
                "revisable now that it is visible."
            )

    payload = {
        "schema": "jlens.mmpilot.preconvergence_convergence_verdict.v1",
        "verdict_name": "NATIVE_OUTPUT_CONVERGENCE",
        "verdict": verdict,
        "layer": int(layer),
        "measured_layer": measured_layer,
        "classification": classification,
        "criterion_digest": criterion_digest,
        "criterion_thresholds_unchanged": criterion_digest == FROZEN_CRITERION_DIGEST,
        "required_modalities": list(REQUIRED_MODALITIES),
        "validity_clauses": clauses,
        "failed_validity_clauses": failed,
        "admissible_focal_concepts": admissible,
        "inadmissible_focal_concepts": list(
            admissibility.get("excluded_concept_names") or []
        ),
        "concepts_replaced": False,
        "rationale": rationale,
    }
    payload["verdict_checksum"] = payload_checksum(payload)
    return payload


# ------------------------------------------------------------ lens integrity


def adjacent_lens_integrity(
    *,
    layer: int,
    scale: int,
    snapshot: Mapping,
    confirmation_verdict: Mapping,
    invariance: Mapping | None,
    calibration_modality: str,
) -> dict:
    """Did the layer's own freshly fitted lens survive every integrity clause?

    The completed studies discovered a *published* artifact and validated it
    against its sidecar. This study fitted the lens itself, in this run, so
    there is no sidecar to discover and pretending otherwise would be
    ceremony. What is checked instead is what actually decides the reading: the
    snapshot is the one that was fitted at the declared scale, the layer is the
    one that was confirmed, the calibration was text-only, and the capture hook
    is a no-op in every modality.
    """
    checks = [
        {
            "check": "snapshot_checksum_recorded",
            "passed": bool(snapshot.get("checksum")),
            "detail": str(snapshot.get("checksum")),
        },
        {
            "check": "snapshot_prompt_count_matches_scale",
            "passed": int(snapshot.get("n_prompts", -1)) == int(scale),
            "detail": f"n_prompts={snapshot.get('n_prompts')!r} required {scale}",
        },
        {
            "check": "confirmation_passed_for_this_layer",
            "passed": bool(confirmation_verdict.get("passed"))
            and int(confirmation_verdict.get("layer", -1)) == int(layer),
            "detail": (
                f"layer={confirmation_verdict.get('layer')!r} "
                f"passed={confirmation_verdict.get('passed')!r} "
                f"failed={confirmation_verdict.get('failed_checks')!r}"
            ),
        },
        {
            "check": "layer_is_a_predeclared_candidate",
            "passed": int(layer) in tuple(int(x) for x in ADJACENT_CANDIDATE_LAYERS),
            "detail": f"{layer} in {list(ADJACENT_CANDIDATE_LAYERS)}",
        },
        {
            "check": "text_only_calibration",
            "passed": calibration_modality == "text-only",
            "detail": str(calibration_modality),
        },
        {
            "check": "hook_site_recorded",
            "passed": bool(snapshot.get("hook_site")),
            "detail": str(snapshot.get("hook_site") or ADJACENT_HOOK_SITE),
        },
    ]
    if invariance is None:
        checks.append(
            {
                "check": "capture_and_zero_coefficient_invariance",
                "passed": False,
                "detail": (
                    "no invariance record: a missing gate is a refusal, not a "
                    "default"
                ),
            }
        )
    else:
        checks.append(
            {
                "check": "capture_and_zero_coefficient_invariance",
                "passed": bool(invariance.get("passed")),
                "detail": f"modalities {invariance.get('modalities')}",
            }
        )
    failed = [entry["check"] for entry in checks if not entry["passed"]]
    payload = {
        "schema": "jlens.mmpilot.preconvergence_lens_integrity.v1",
        "verdict_name": "ADJACENT_LENS_INTEGRITY",
        "verdict": "PASSED" if not failed else "REFUSED",
        "layer": int(layer),
        "scale": int(scale),
        "checks": checks,
        "failed_checks": failed,
        "hook_site": ADJACENT_HOOK_SITE,
    }
    payload["integrity_checksum"] = payload_checksum(payload)
    return payload


# ------------------------------------------------------------ causal controls


def assert_causal_controls_recorded(
    controls: Mapping,
    *,
    required: Sequence[str] = REQUIRED_CAUSAL_CONTROLS,
) -> dict:
    """A missing causal control is a failure, never a pass.

    ``controls`` maps control name to a record. A record that is absent, empty,
    or carries ``passed`` falsely fails; a record that merely says "not
    applicable" is treated as missing, because an inapplicable control is one
    that was not run.
    """
    rows: list[dict] = []
    missing: list[str] = []
    failing: list[str] = []
    for name in required:
        record = controls.get(name)
        present = bool(record)
        passed = bool((record or {}).get("passed")) if isinstance(record, Mapping) else False
        if not present:
            missing.append(name)
        elif not passed:
            failing.append(name)
        rows.append(
            {
                "control": name,
                "present": present,
                "passed": passed,
                "detail": (record or {}).get("detail")
                if isinstance(record, Mapping)
                else None,
            }
        )
    payload = {
        "schema": "jlens.mmpilot.preconvergence_causal_controls.v1",
        "required": list(required),
        "rows": rows,
        "missing_or_empty": missing,
        "failing": failing,
        "passed": not missing and not failing,
        "why": (
            "an absent control record is indistinguishable from a control that "
            "was never run, and a study cannot report a controlled effect on "
            "controls it cannot show it ran"
        ),
    }
    payload["controls_checksum"] = payload_checksum(payload)
    return payload


# ------------------------------------------------------ the same-population rule


def assert_same_population(
    *,
    convergence_population_digest: str | None,
    causal_population_digest: str | None,
    convergence_layer: int | None,
    causal_layer: int | None,
    require: bool = True,
) -> dict:
    """The two halves of the principal claim must be about the same thing.

    Raises:
        PopulationsDiffer: When ``require`` and either the population digests or
            the layers disagree. Combining them anyway is precisely the error
            that makes a pre-convergence claim look established when it is two
            unrelated measurements side by side.
    """
    same_population = bool(
        convergence_population_digest
        and causal_population_digest
        and convergence_population_digest == causal_population_digest
    )
    same_layer = bool(
        convergence_layer is not None
        and causal_layer is not None
        and int(convergence_layer) == int(causal_layer)
    )
    payload = {
        "schema": "jlens.mmpilot.preconvergence_same_population.v1",
        "convergence_population_digest": convergence_population_digest,
        "causal_population_digest": causal_population_digest,
        "convergence_layer": convergence_layer,
        "causal_layer": causal_layer,
        "same_population": same_population,
        "same_layer": same_layer,
        "combinable": bool(same_population and same_layer),
        "why": (
            "the principal claim is about one layer on one population; pairing "
            "convergence measured on one population with a causal effect "
            "measured on another credits an effect to photographs it was never "
            "measured on"
        ),
    }
    payload["same_population_checksum"] = payload_checksum(payload)
    if require and not payload["combinable"]:
        raise PopulationsDiffer(
            "the convergence half and the causal half are not about the same "
            f"layer and population (same_population={same_population}, "
            f"same_layer={same_layer}). Refusing to combine them."
        )
    return payload


# ----------------------------------------------------------- the five verdicts


def preconvergence_verdicts(
    *,
    lens_verdict: Mapping,
    convergence: Mapping | None,
    causal: Mapping | None,
    causal_controls: Mapping | None,
    stage_four: Mapping,
    same_population: Mapping | None,
) -> dict:
    """The five verdicts and the one terminal outcome, assembled in one place.

    The ordering is not cosmetic. Lens validity gates the layer; the layer gates
    the convergence reading; convergence and the same-population proof together
    gate whether a causal result may enter the principal claim at all.
    """
    lens = str(lens_verdict.get("verdict"))
    selected_layer = lens_verdict.get("selected_layer")

    convergence_verdict = (
        str(convergence.get("verdict")) if convergence else TRANSFER_NOT_EVALUATED
    )
    causal_verdict_raw = str((causal or {}).get("verdict") or TRANSFER_NOT_EVALUATED)
    overridden = bool(stage_four.get("gate_overridden"))
    controls_ok = bool((causal_controls or {}).get("passed"))

    if not stage_four.get("runs"):
        three_modality = TRANSFER_NOT_EVALUATED
        three_modality_detail = str(stage_four.get("statement"))
    elif overridden:
        three_modality = TRANSFER_DESCRIPTIVE_ONLY
        three_modality_detail = (
            f"measured {causal_verdict_raw!r} under an explicit override of the "
            "predeclared gate; descriptive only"
        )
    elif not controls_ok:
        three_modality = TRANSFER_NOT_SUPPORTED
        three_modality_detail = (
            "required causal controls are missing or failing: "
            f"missing={(causal_controls or {}).get('missing_or_empty')} "
            f"failing={(causal_controls or {}).get('failing')}"
        )
    elif causal_verdict_raw == TRANSFER_SUPPORTED:
        three_modality = TRANSFER_SUPPORTED
        three_modality_detail = (
            "controlled cross-modal transfer on off-diagonal cells, image-level "
            "aggregated, above every required control"
        )
    else:
        three_modality = TRANSFER_NOT_SUPPORTED
        three_modality_detail = f"measured {causal_verdict_raw!r}"

    combinable = bool((same_population or {}).get("combinable"))

    # --- the principal claim ------------------------------------------------
    if lens == ADJACENT_LENS_NO_GO:
        terminal = ADJACENT_LENS_NO_GO
        principal = TRANSFER_NOT_EVALUATED
        statement = (
            f"No layer in {list(ADJACENT_CANDIDATE_LAYERS)} has a confirmed "
            "J-lens on an untouched confirmation set. The study stops here: "
            "there is no layer to measure convergence or causal transfer at, and "
            "the interval is closed, so there is no wider band to try."
        )
    elif convergence_verdict == REFUSED_INVALID or convergence is None:
        terminal = REFUSED_INVALID
        principal = TRANSFER_NOT_EVALUATED
        statement = (
            f"Layer {selected_layer} has a confirmed lens, but the native "
            "convergence measurement is not established "
            f"({(convergence or {}).get('failed_validity_clauses')}). No "
            "classification and no causal claim is reported."
        )
    elif convergence_verdict == LAYER_CONVERGED:
        terminal = CONVERGED_BEFORE_CAUSAL_TEST
        principal = TRANSFER_NOT_EVALUATED
        statement = (
            f"Layer {selected_layer} has a confirmed lens and is {CONVERGED} "
            "under the frozen criterion. Any causal effect at this layer occurs "
            "at or after native direct-readout convergence, so the "
            "pre-convergence claim cannot be made here. The question moves to "
            f"layers shallower than {selected_layer}, and L{FAILED_LOWER_LAYER} "
            "has already failed lens confirmation."
        )
    elif convergence_verdict == LAYER_AMBIGUOUS:
        terminal = AMBIGUOUS_CONVERGENCE
        principal = TRANSFER_NOT_EVALUATED
        statement = (
            f"Layer {selected_layer} has a confirmed lens and falls between the "
            "frozen bars, exactly as L32 did twice. No pre-convergence claim "
            "follows from an ambiguous layer, and the criterion is not revisable "
            "now that the result is visible."
        )
    elif three_modality == TRANSFER_SUPPORTED and combinable:
        terminal = PRECONVERGENCE_SUPPORTED
        principal = TRANSFER_SUPPORTED
        statement = (
            f"One and the same physical layer ({selected_layer}) has an untouched "
            f"confirmed J-lens, is {NOT_CONVERGED} in every required modality "
            "under the frozen criterion, and shows controlled cross-modal causal "
            "transfer — all three on the same independent population."
        )
    elif three_modality == TRANSFER_NOT_EVALUATED:
        terminal = CAUSAL_TRANSFER_NOT_SUPPORTED
        principal = TRANSFER_NOT_EVALUATED
        statement = (
            f"Layer {selected_layer} has a confirmed lens and is {NOT_CONVERGED} "
            "under the frozen criterion, which is the half this study was built "
            "to establish. Causal transfer was NOT measured on this population, "
            "so the principal claim is not established. That is a missing "
            "measurement, not a negative result."
        )
    elif three_modality == TRANSFER_DESCRIPTIVE_ONLY:
        terminal = CAUSAL_TRANSFER_NOT_SUPPORTED
        principal = TRANSFER_DESCRIPTIVE_ONLY
        statement = (
            f"Layer {selected_layer} is {NOT_CONVERGED} with a confirmed lens, "
            "but the causal stage ran as an explicit override of its predeclared "
            "gate. Overridden causal results are descriptive and never support "
            "the principal claim."
        )
    elif not combinable:
        terminal = REFUSED_INVALID
        principal = TRANSFER_NOT_SUPPORTED
        statement = (
            "The convergence half and the causal half are not about the same "
            f"layer and population ({same_population}). They are reported "
            "separately and are never combined."
        )
    else:
        terminal = CAUSAL_TRANSFER_NOT_SUPPORTED
        principal = TRANSFER_NOT_SUPPORTED
        statement = (
            f"Layer {selected_layer} is {NOT_CONVERGED} with a confirmed lens, "
            f"and causal transfer on this same population was {three_modality}. "
            "The pre-convergence claim is not supported at this layer."
        )

    verdicts = {
        "ADJACENT_LENS_VALIDITY": {
            "verdict": lens,
            "detail": str(lens_verdict.get("rationale")),
        },
        "EARLIEST_CONFIRMED_LAYER": {
            "verdict": (
                f"L{selected_layer}" if selected_layer is not None else "NONE"
            ),
            "detail": (
                "the lowest candidate passing every frozen confirmation clause"
                if selected_layer is not None
                else f"no candidate in {list(ADJACENT_CANDIDATE_LAYERS)} passed"
            ),
        },
        "NATIVE_OUTPUT_CONVERGENCE": {
            "verdict": convergence_verdict,
            "detail": str((convergence or {}).get("rationale")),
        },
        "THREE_MODALITY_CAUSAL_TRANSFER": {
            "verdict": three_modality,
            "detail": three_modality_detail,
        },
        "PRECONVERGENCE_CAUSAL_TRANSFER": {
            "verdict": principal,
            "detail": statement,
        },
    }

    payload = {
        "schema": "jlens.mmpilot.preconvergence_verdicts.v1",
        "protocol": PRECONVERGENCE_PROTOCOL,
        "verdict_names": list(VERDICT_NAMES),
        "verdicts": verdicts,
        "selected_layer": selected_layer,
        "terminal_outcome": terminal,
        "terminal_outcomes_available": list(TERMINAL_OUTCOMES),
        "principal_claim_requires": [
            "one and the same physical layer",
            "untouched confirmed J-lens validity",
            f"{NOT_CONVERGED} native direct readout in "
            f"{list(REQUIRED_MODALITIES)}",
            "controlled cross-modal causal transfer",
            "all three on the same independent multimodal population",
        ],
        "same_population": same_population,
        "stage_four": stage_four,
        "statement": statement,
        "coordinate_swap_scope": COORDINATE_SWAP_SCOPE,
    }
    payload["verdicts_checksum"] = payload_checksum(payload)
    return payload


# --------------------------------------------------- pre-download contracts

#: A stand-in bound to parameters during signature checks. Never called.
_ARG = object()


def preconvergence_call_contracts() -> list[tuple]:
    """Every external call this study's **real** branch makes, with its kwargs.

    ``inspect.signature(...).bind`` raises on a renamed keyword, a removed
    parameter or a newly required one — precisely the drift that has been
    reaching the L4 instead of CI, because a MOCK run never executes the real
    branch's lines and a string-matching test never notices that a name is
    gone.

    Every entry is imported at call time from the same checked-out commit the
    notebook imports from, so a symbol that does not exist fails here, on CPU,
    before the ~16 GB download.
    """
    from jlens.calibration.adjacent import (
        adjacent_budget,
        adjacent_lens_verdict,
        assert_new_source_layers,
        audit_untouched_confirmation,
        build_untouched_confirmation,
        select_earliest_confirmed_layer,
    )
    from jlens.calibration.corpus import (
        build_records,
        collect_records_for_partition_quotas,
    )
    from jlens.calibration.extension import (
        build_extension_fit_order,
        build_fresh_evaluation_splits,
        parent_collection_parameters,
        verify_fit_prefix,
        verify_reconstructed_partitions,
    )
    from jlens.calibration.fitting import filter_records_by_tokens, run_calibration
    from jlens.calibration.gate import (
        evaluate_calibration_layers,
        ordinary_next_token_argmax,
        select_diverse_validation_prompts,
    )
    from jlens.calibration.parent import (
        ParentRequirements,
        audit_parent_run,
        load_parent_run,
        protected_parent_checksums,
    )
    from jlens.calibration.plan import build_capture_plan
    from jlens.calibration.state import CalibrationFingerprint
    from jlens.mmpilot.admissibility import concept_admissibility
    from jlens.mmpilot.capability import candidate_token_ids
    from jlens.mmpilot.convergence import (
        ConvergenceFingerprint,
        ConvergenceStore,
        audit_native_head,
        build_population,
        head_from_model,
        resolve_candidate_tokens,
    )
    from jlens.mmpilot.expansion import load_expanded_manifest
    from jlens.mmpilot.independence import (
        audit_image_independence,
        resolve_image_identity,
        summarize_interventions_by_image,
    )
    from jlens.mmpilot.l32_followup import (
        assert_native_head_agrees,
        run_single_layer_convergence,
    )
    from jlens.mmpilot.l32_resolution import (
        assert_controls_recorded,
        audit_population_disjointness,
        clean_predictions_from_capability,
        independent_pool,
        resolve_excluded_media,
    )
    from jlens.mmpilot.manifest import build_subset
    from jlens.mmpilot.media_io import drive_media_loaders
    from jlens.mmpilot.pipeline import (
        available_modalities,
        build_condition_inputs,
        build_dictionaries,
        scientific_fingerprint,
        stage_activations,
        stage_capability,
        stage_causal,
        stage_codes,
        stage_directions,
    )
    from jlens.mmpilot.prep_cache import run_exclusion_preparation
    from jlens.mmpilot.prompt_protocol import (
        build_protocol_prompt,
        prompt_protocol_fingerprint,
    )
    from jlens.mmpilot.real_backend import build_real_backend
    from jlens.mmpilot.store import RunFingerprint, UnitStore
    from jlens.mmpilot.tri_modal import (
        assert_audio_protocol,
        audio_capability_verdict,
        causal_transfer_verdict,
        estimate_stage_passes,
        run_invariance_by_modality,
    )

    return [
        # ---- the model load itself
        (
            "build_real_backend",
            build_real_backend,
            (_ARG,),
            {
                "revision": _ARG,
                "token": _ARG,
                "device": _ARG,
                "allow_model_load": True,
                "expect_n_layers": _ARG,
                "expect_d_model": _ARG,
                "expect_vocab_size": _ARG,
                "resolve_audio": True,
            },
        ),
        (
            "assert_audio_protocol",
            assert_audio_protocol,
            (_ARG,),
            {"expected_fingerprint": _ARG},
        ),
        # ---- Stage 0: corpus provenance and the fit ordering
        ("load_parent_run", load_parent_run, (_ARG,), {"baseline_scale": _ARG}),
        (
            "ParentRequirements",
            ParentRequirements,
            (),
            {
                "model_repo_id": _ARG,
                "model_revision": _ARG,
                "tokenizer_repo_id": _ARG,
                "tokenizer_revision": _ARG,
                "source_layers": _ARG,
                "target_layer": _ARG,
                "d_model": _ARG,
                "hook_site": _ARG,
                "skip_first": _ARG,
                "max_seq_len": _ARG,
                "dim_batch": _ARG,
                "corpus_hf_dataset": _ARG,
                "corpus_config": _ARG,
                "corpus_split": _ARG,
                "estimator": _ARG,
                "artifact_format_version": _ARG,
                "baseline_scale": _ARG,
                "expected_n_done": _ARG,
            },
        ),
        ("audit_parent_run", audit_parent_run, (_ARG,), {"requirements": _ARG}),
        (
            "protected_parent_checksums",
            protected_parent_checksums,
            (_ARG,),
            {"layout": _ARG},
        ),
        ("parent_collection_parameters", parent_collection_parameters, (_ARG,), {}),
        (
            "collect_records_for_partition_quotas",
            collect_records_for_partition_quotas,
            (),
            {
                "corpus_id": _ARG,
                "texts": _ARG,
                "min_chars": _ARG,
                "min_fit": _ARG,
                "max_texts": _ARG,
                "seed": _ARG,
                "n_validation": _ARG,
                "n_confirmation": _ARG,
            },
        ),
        ("build_records", build_records, (_ARG, _ARG), {"min_chars": _ARG}),
        (
            "verify_reconstructed_partitions",
            verify_reconstructed_partitions,
            (_ARG,),
            {"parent": _ARG},
        ),
        (
            "filter_records_by_tokens",
            filter_records_by_tokens,
            (_ARG,),
            {"token_count": _ARG, "skip_first": _ARG, "max_seq_len": _ARG},
        ),
        (
            "build_extension_fit_order",
            build_extension_fit_order,
            (_ARG,),
            {"n_needed": _ARG, "extension_pool": _ARG},
        ),
        (
            "verify_fit_prefix",
            verify_fit_prefix,
            (_ARG,),
            {"n_parent": _ARG, "parent_prefix_checksum": _ARG},
        ),
        (
            "build_fresh_evaluation_splits",
            build_fresh_evaluation_splits,
            (_ARG,),
            {
                "excluded": _ARG,
                "corpus_id": _ARG,
                "n_development": _ARG,
                "n_confirmation": _ARG,
            },
        ),
        (
            "build_untouched_confirmation",
            build_untouched_confirmation,
            (_ARG,),
            {
                "excluded": _ARG,
                "corpus_id": _ARG,
                "n_confirmation": _ARG,
                "development_role": _ARG,
                "dependency_manifests": _ARG,
            },
        ),
        (
            "audit_untouched_confirmation",
            audit_untouched_confirmation,
            (_ARG,),
            {"excluded": _ARG},
        ),
        (
            "assert_new_source_layers",
            assert_new_source_layers,
            (),
            {"candidate_layers": _ARG, "parent_source_layers": _ARG},
        ),
        # ---- Stage 1: fitting
        (
            "build_capture_plan",
            build_capture_plan,
            (),
            {
                "layers": _ARG,
                "target_layer": _ARG,
                "d_model": _ARG,
                "dim_batch": _ARG,
                "max_seq_len": _ARG,
                "skip_first": _ARG,
                "n_layers": _ARG,
            },
        ),
        (
            "CalibrationFingerprint",
            CalibrationFingerprint,
            (),
            {
                "mode": _ARG,
                "protocol_version": _ARG,
                "model_repo_id": _ARG,
                "model_revision": _ARG,
                "tokenizer_revision": _ARG,
                "capture_plan_digest": _ARG,
                "corpus_manifest_checksum": _ARG,
                "gate_digest": _ARG,
                "plateau_rule_digest": _ARG,
                "scale_points": _ARG,
                "artifact_format_version": _ARG,
                "extra": _ARG,
            },
        ),
        (
            "run_calibration",
            run_calibration,
            (_ARG, _ARG),
            {
                "plan": _ARG,
                "scale_points": _ARG,
                "store": _ARG,
                "checkpoint_every": _ARG,
                "diagnostics_every": _ARG,
            },
        ),
        ("adjacent_budget", adjacent_budget, (), {"scale": _ARG, "layers": _ARG}),
        # ---- Stage 2: development, confirmation, selection
        (
            "ordinary_next_token_argmax",
            ordinary_next_token_argmax,
            (_ARG, _ARG),
            {"max_length": _ARG},
        ),
        (
            "select_diverse_validation_prompts",
            select_diverse_validation_prompts,
            (_ARG,),
            {
                "n_prompts": _ARG,
                "gate": _ARG,
                "seed": _ARG,
                "target_token_for_prompt": _ARG,
            },
        ),
        (
            "evaluate_calibration_layers",
            evaluate_calibration_layers,
            (_ARG,),
            {"layers": _ARG, "scale": _ARG, "stage": _ARG, "gate": _ARG},
        ),
        (
            "select_earliest_confirmed_layer",
            select_earliest_confirmed_layer,
            (_ARG,),
            {"candidates": _ARG, "development": _ARG},
        ),
        (
            "adjacent_lens_verdict",
            adjacent_lens_verdict,
            (_ARG,),
            {
                "confirmation_manifest": _ARG,
                "untouched_audit": _ARG,
                "source_layer_record": _ARG,
            },
        ),
        # ---- Stage 0/3: the multimodal population
        (
            "load_expanded_manifest",
            load_expanded_manifest,
            (_ARG,),
            {
                "original_checksum": _ARG,
                "expected_sources": _ARG,
                "conversion": _ARG,
                "expected_group_count": _ARG,
                "expected_lexicon_hash": _ARG,
            },
        ),
        (
            "run_exclusion_preparation",
            run_exclusion_preparation,
            (_ARG, _ARG),
            {
                "fingerprint": _ARG,
                "batch_files": _ARG,
                "checkpoint_seconds": _ARG,
                "progress": _ARG,
                "protected_prefixes": _ARG,
            },
        ),
        ("resolve_excluded_media", resolve_excluded_media, (_ARG, _ARG), {}),
        ("independent_pool", independent_pool, (_ARG, _ARG), {}),
        (
            "build_subset",
            build_subset,
            (_ARG, _ARG),
            {
                "groups_per_concept": _ARG,
                "negatives_per_concept": _ARG,
                "seed": _ARG,
                "evidence_config": _ARG,
                "profile": _ARG,
                "evidence_index": _ARG,
            },
        ),
        (
            "audit_population_disjointness",
            audit_population_disjointness,
            (_ARG, _ARG),
            {"require": True},
        ),
        # ---- the open prompt
        (
            "build_protocol_prompt",
            build_protocol_prompt,
            (),
            {"protocol": _ARG, "evidence": _ARG, "external_candidates": _ARG},
        ),
        (
            "prompt_protocol_fingerprint",
            prompt_protocol_fingerprint,
            (_ARG,),
            {
                "model_revision": _ARG,
                "processor_revision": _ARG,
                "audio_protocol_fingerprint": _ARG,
            },
        ),
        # ---- Stage 3
        ("drive_media_loaders", drive_media_loaders, (), {"journal": _ARG}),
        ("available_modalities", available_modalities, (_ARG, _ARG), {}),
        (
            "build_condition_inputs",
            build_condition_inputs,
            (_ARG, _ARG, _ARG, _ARG, _ARG),
            {},
        ),
        (
            "run_invariance_by_modality",
            run_invariance_by_modality,
            (_ARG, _ARG, _ARG),
            {},
        ),
        (
            "scientific_fingerprint",
            scientific_fingerprint,
            (_ARG,),
            {
                "ranked_concepts": _ARG,
                "selected_concepts": _ARG,
                "focal_concepts": _ARG,
                "unrelated_controls": _ARG,
                "derived_cache_fingerprint": _ARG,
                "split_provenance_checksum": _ARG,
                "n_train_positive_images": _ARG,
                "n_train_negative_images": _ARG,
                "n_test_positive_images": _ARG,
                "n_test_negative_images": _ARG,
                "verdict_version": _ARG,
                "prompt_protocol": _ARG,
                "candidate_ordering_protocol": _ARG,
            },
        ),
        (
            "RunFingerprint",
            RunFingerprint,
            (),
            {
                "mode": _ARG,
                "model_repo_id": _ARG,
                "model_revision": _ARG,
                "processor_revision": _ARG,
                "layers": _ARG,
                "lens_checksum": _ARG,
                "manifest_checksum": _ARG,
                "split_id": _ARG,
                "intervention_config": _ARG,
                "selection_config": _ARG,
                "extra": _ARG,
            },
        ),
        ("UnitStore", UnitStore, (_ARG, _ARG), {}),
        (
            "stage_capability",
            stage_capability,
            (_ARG, _ARG, _ARG, _ARG, _ARG),
            {"modalities": _ARG, "questions": _ARG},
        ),
        (
            "audio_capability_verdict",
            audio_capability_verdict,
            (_ARG,),
            {"selected_concepts": _ARG, "modalities": _ARG, "thresholds": _ARG},
        ),
        (
            "concept_admissibility",
            concept_admissibility,
            (_ARG,),
            {"capability": _ARG},
        ),
        (
            "stage_activations",
            stage_activations,
            (_ARG, _ARG, _ARG, _ARG, _ARG),
            {
                "modalities": _ARG,
                "retained_concepts": _ARG,
                "model_revision": _ARG,
                "question": _ARG,
            },
        ),
        ("head_from_model", head_from_model, (_ARG,), {}),
        ("audit_native_head", audit_native_head, (_ARG,), {"model": _ARG, "probes": _ARG}),
        (
            "assert_native_head_agrees",
            assert_native_head_agrees,
            (_ARG,),
            {"required": True},
        ),
        ("candidate_token_ids", candidate_token_ids, (_ARG, _ARG), {}),
        ("resolve_candidate_tokens", resolve_candidate_tokens, (_ARG,), {}),
        (
            "clean_predictions_from_capability",
            clean_predictions_from_capability,
            (_ARG,),
            {},
        ),
        (
            "build_population",
            build_population,
            (),
            {
                "activations": _ARG,
                "clean_predictions": _ARG,
                "capability": _ARG,
                "focal_concepts": _ARG,
                "layers": _ARG,
            },
        ),
        (
            "ConvergenceFingerprint",
            ConvergenceFingerprint,
            (),
            {
                "protocol": _ARG,
                "completed_run_fingerprint_digest": _ARG,
                "completed_run_dir": _ARG,
                "model_repo_id": _ARG,
                "model_revision": _ARG,
                "processor_revision": _ARG,
                "layers": _ARG,
                "candidate_digest": _ARG,
                "readout_mode": _ARG,
                "head_checksum": _ARG,
                "criterion_digest": _ARG,
                "code_version": _ARG,
                "extra": _ARG,
            },
        ),
        ("ConvergenceStore", ConvergenceStore, (_ARG, _ARG), {}),
        (
            "run_single_layer_convergence",
            run_single_layer_convergence,
            (),
            {
                "population": _ARG,
                "head": _ARG,
                "tokenization": _ARG,
                "head_audit": _ARG,
                "store": _ARG,
                "layer": _ARG,
                "confirmation_record": _ARG,
                "control_seed": _ARG,
            },
        ),
        (
            "assert_controls_recorded",
            assert_controls_recorded,
            (_ARG,),
            {"layer": _ARG},
        ),
        # ---- Stage 4
        (
            "build_dictionaries",
            build_dictionaries,
            (_ARG, _ARG, _ARG),
            {"device": _ARG, "dtype": _ARG, "build_chunk_rows": _ARG},
        ),
        ("stage_codes", stage_codes, (_ARG, _ARG, _ARG, _ARG), {"lens_checksum": _ARG}),
        (
            "stage_directions",
            stage_directions,
            (_ARG, _ARG, _ARG, _ARG, _ARG),
            {"concepts": _ARG, "modalities": _ARG, "lens_checksum": _ARG},
        ),
        (
            "stage_causal",
            stage_causal,
            (_ARG, _ARG, _ARG, _ARG, _ARG, _ARG, _ARG, _ARG),
            {
                "concepts": _ARG,
                "modalities": _ARG,
                "all_concepts": _ARG,
                "unrelated_controls": _ARG,
                "question": _ARG,
            },
        ),
        ("resolve_image_identity", resolve_image_identity, (_ARG,), {}),
        (
            "summarize_interventions_by_image",
            summarize_interventions_by_image,
            (_ARG, _ARG),
            {"group_summary": _ARG},
        ),
        (
            "audit_image_independence",
            audit_image_independence,
            (_ARG,),
            {"interventions": _ARG, "concepts": _ARG},
        ),
        (
            "causal_transfer_verdict",
            causal_transfer_verdict,
            (_ARG,),
            {
                "layer": _ARG,
                "focal_concepts": _ARG,
                "thresholds": _ARG,
                "name": _ARG,
                "capability": _ARG,
            },
        ),
        # ---- budgets
        (
            "estimate_stage_passes",
            estimate_stage_passes,
            (),
            {
                "n_concepts": _ARG,
                "n_focal_concepts": _ARG,
                "modalities": _ARG,
                "layers": _ARG,
                "causal_layers": _ARG,
                "n_total_groups": _ARG,
                "n_capability_groups": _ARG,
                "n_targets_per_cell": _ARG,
                "alphas": _ARG,
                "stage": _ARG,
                "n_candidate_orders": _ARG,
                "d_model": _ARG,
            },
        ),
    ]


def check_preconvergence_call_contracts() -> list[str]:
    """Bind every real-branch call against the installed signature.

    Returns the list of failures, empty when everything binds. An unresolvable
    *import* is reported as a failure too rather than propagating, so the
    pre-download cell can print all of them at once instead of one per run.
    """
    import inspect

    try:
        contracts = preconvergence_call_contracts()
    except ImportError as error:  # a symbol that does not exist at all
        return [f"import: {error}"]

    failures = []
    for name, target, args, kwargs in contracts:
        try:
            inspect.signature(target).bind(*args, **kwargs)
        except TypeError as exc:
            failures.append(f"{name}: {exc}")
    return failures


# ------------------------------------------------------------- fingerprint

PRECONVERGENCE_FINGERPRINT_FIELDS: tuple[str, ...] = (
    "protocol",
    "stage_plan_version",
    "stage_four_rule_digest",
    "intervention_family",
    "adjacent_protocol_digest",
    "candidate_layers",
    "fitting_scale",
    "adjacent_gate_digest",
    "adjacent_selection_rule_digest",
    "adjacent_run_dir",
    "confirmation_manifest_checksum",
    "untouched_audit_checksum",
    "model_repo_id",
    "model_revision",
    "processor_revision",
    "transformers_version",
    "torch_version",
    "audio_protocol_version",
    "audio_protocol_fingerprint",
    "lens_path",
    "lens_checksum",
    "lens_confirmation_status",
    "physical_layer",
    "hook_site",
    "d_model",
    "residual_convention",
    "final_prompt_token_position",
    "dictionary_orientation",
    "dictionary_normalization",
    "calibration_modality",
    "original_manifest_checksum",
    "expanded_manifest_checksum",
    "cache_schema_version",
    "evidence_lexicon_hash",
    "exclusion_run_dirs",
    "exclusion_run_checksum",
    "population_pins_checksum",
    "preparation_version",
    "preparation_digest",
    "exclusion_completeness_digest",
    "independent_pool_digest",
    "concept_ranking_digest",
    "frozen_concept_feasibility_digest",
    "prompt_protocol",
    "prompt_hash",
    "prompt_protocol_digest",
    "candidate_leakage_audit_digest",
    "selected_concepts",
    "focal_concepts",
    "capability_protocol",
    "admissibility_rule_version",
    "selection_algorithm_version",
    "selection_seed",
    "selection_profile_version",
    "selected_population_digest",
    "sample_size_rule_version",
    "sample_size_plan_digest",
    "convergence_criterion_version",
    "convergence_criterion_digest",
    "control_variants",
    "control_seed",
    "required_causal_controls",
    "media_io_version",
    "jlens_version",
)


def preconvergence_fingerprint(**fields) -> dict:
    """Bind every scientific configuration field, and refuse a missing one.

    Raises:
        PreconvergenceRefused: If any required field is missing or an unknown
            one is supplied. A ``None`` default would let a run under one
            configuration resume from another's directory the first time a field
            was forgotten.
    """
    missing = [
        name for name in PRECONVERGENCE_FINGERPRINT_FIELDS if name not in fields
    ]
    unknown = sorted(set(fields) - set(PRECONVERGENCE_FINGERPRINT_FIELDS))
    if missing or unknown:
        raise PreconvergenceRefused(
            "the pre-convergence fingerprint is incomplete or over-specified: "
            f"missing {missing}, unknown {unknown}"
        )
    payload = {name: fields[name] for name in PRECONVERGENCE_FINGERPRINT_FIELDS}
    return {**payload, "fingerprint_digest": payload_checksum(payload)}


# ------------------------------------------------------------ summary/report


def build_summary(
    *,
    fingerprint: Mapping,
    verdicts: Mapping,
    lens_verdict: Mapping,
    confirmation_manifest: Mapping,
    untouched_audit: Mapping,
    source_layer_record: Mapping,
    fit_record: Mapping,
    convergence: Mapping | None,
    convergence_controls: Mapping | None,
    capability: Mapping,
    disjointness: Mapping,
    pseudoreplication: Mapping,
    pool: Mapping,
    exclusion: Mapping,
    population_pins: Mapping,
    sample_plan: Mapping,
    leakage_audit: Mapping,
    stage_plan_record: Mapping,
    stage_four: Mapping,
    causal: Mapping | None,
    causal_controls: Mapping | None,
    immutability: Mapping,
    cache: Mapping,
    resume: Mapping,
    mode: str,
    preparation: Mapping | None = None,
    frozen_concept_feasibility: Mapping | None = None,
) -> dict:
    """``l27_l31_preconvergence_summary.json``, assembled in one place."""
    summary = {
        "schema": "jlens.mmpilot.preconvergence_summary.v1",
        "protocol": PRECONVERGENCE_PROTOCOL,
        "mode": str(mode),
        "adjacent_protocol": ADJACENT_PROTOCOL.to_dict(),
        "adjacent_protocol_digest": ADJACENT_PROTOCOL.digest,
        "interval": {
            "candidates": list(ADJACENT_CANDIDATE_LAYERS),
            "floor": FAILED_LOWER_LAYER,
            "ceiling": AMBIGUOUS_UPPER_LAYER,
            "converged_reference": CONVERGED_REFERENCE_LAYER,
            "closed": True,
        },
        "fingerprint": dict(fingerprint),
        "primary_verdict": verdicts.get("terminal_outcome"),
        "verdicts": verdicts.get("verdicts"),
        "terminal_outcomes_available": list(TERMINAL_OUTCOMES),
        "principal_claim_requires": verdicts.get("principal_claim_requires"),
        "same_population": verdicts.get("same_population"),
        "statement": verdicts.get("statement"),
        "lens": {
            "verdict": dict(lens_verdict),
            "confirmation_manifest": dict(confirmation_manifest),
            "untouched_audit": dict(untouched_audit),
            "source_layers": dict(source_layer_record),
            "fit": dict(fit_record),
        },
        "convergence": dict(convergence) if convergence else None,
        "convergence_controls": dict(convergence_controls or {}),
        "capability": dict(capability),
        "population": {
            "disjointness": dict(disjointness),
            "pseudoreplication": dict(pseudoreplication),
            "pool": dict(pool),
            "exclusion": dict(exclusion),
            "pins": dict(population_pins),
            "sample_plan": dict(sample_plan),
            "frozen_concept_feasibility": dict(frozen_concept_feasibility or {}),
        },
        "prompt": {
            "protocol": OPEN_PROMPT_PROTOCOL,
            "candidate_leakage_audit": dict(leakage_audit),
            "candidates_are_external_scoring_choices_only": True,
        },
        "stage_plan": dict(stage_plan_record),
        "stage_four": dict(stage_four),
        "causal": dict(causal) if causal else None,
        "causal_controls": dict(causal_controls or {}),
        "intervention_family": INTERVENTION_FAMILY,
        "coordinate_swap_scope": COORDINATE_SWAP_SCOPE,
        "preparation": dict(preparation or {}),
        "cache": dict(cache),
        "completed_runs_unchanged": dict(immutability),
        "resume": dict(resume),
        "concepts_replaced_after_results": False,
        "thresholds_changed_after_results": False,
        "mock_proves_pipeline_only": mode != "real",
    }
    summary["summary_checksum"] = payload_checksum(summary)
    return summary


def render_report(summary: Mapping) -> str:
    """``l27_l31_preconvergence_report.md``."""
    verdicts = summary.get("verdicts") or {}
    lines = [
        "# L27-L31 pre-convergence transition study",
        "",
        f"Protocol `{summary.get('protocol')}` — mode **{summary.get('mode')}**.",
        "",
        f"Candidate interval {summary['interval']['candidates']}, closed: "
        f"L{summary['interval']['floor']} failed lens confirmation, "
        f"L{summary['interval']['ceiling']} is confirmed but AMBIGUOUS, "
        f"L{summary['interval']['converged_reference']} is CONVERGED.",
        "",
        "## Verdicts",
        "",
        "| verdict | value |",
        "|---|---|",
    ]
    for name in VERDICT_NAMES:
        entry = verdicts.get(name) or {}
        lines.append(f"| `{name}` | **{entry.get('verdict')}** |")
    lines += [
        "",
        f"**Terminal outcome: `{summary.get('primary_verdict')}`**",
        "",
        str(summary.get("statement")),
        "",
        "## What the principal claim requires",
        "",
    ]
    for item in summary.get("principal_claim_requires") or []:
        lines.append(f"* {item}")
    lines += [
        "",
        "## Detail",
        "",
    ]
    for name in VERDICT_NAMES:
        entry = verdicts.get(name) or {}
        lines += [f"### `{name}` — {entry.get('verdict')}", "", str(entry.get("detail")), ""]
    lines += [
        "## Scope",
        "",
        summary.get("coordinate_swap_scope", COORDINATE_SWAP_SCOPE),
        "",
        f"Intervention family: `{summary.get('intervention_family')}` "
        "(additive J-space residual steering).",
        "",
        f"Prompt protocol: `{(summary.get('prompt') or {}).get('protocol')}` — the "
        "model-visible prompt names no candidate; candidates are external scoring "
        "choices only.",
        "",
        f"Convergence controls: {list(CONTROL_VARIANTS)}. "
        f"Causal controls: {list(REQUIRED_CAUSAL_CONTROLS)}.",
        "",
    ]
    if summary.get("mock_proves_pipeline_only"):
        lines += [
            "> **MOCK run.** This proves pipeline behaviour and nothing about "
            "Gemma, about any layer in 27-31, or about whether a "
            "pre-convergence causal effect exists.",
            "",
        ]
    return "\n".join(lines)
