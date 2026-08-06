# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""Which measured cells are allowed to *support a claim*, decided in one place.

The three-modality study measures a causal cell for every (focal concept,
ordered modality pair) it predeclared, and it does that **before** it knows
whether the model could read that concept out of every channel behaviorally.
Both facts are fine on their own. Putting them together without a gate is not:
a concept the model cannot recognize in spoken audio has an audio activation,
an audio direction, and an intervention effect, all of them perfectly real
measurements — and none of them evidence that a *concept* transferred, because
the premise that the concept was present in that channel was never established.

That is exactly what happened. ``zebra`` cleared the behavioral capability gate
in text (8/8) and image (8/8) but not in spoken audio (5/8 = 62.5% against a 70%
threshold), and its ``spoken_audio -> text`` and ``spoken_audio -> image`` cells
were nevertheless counted as scientific support for the principal claim.

**The rule.** A causal cell may support a claim only when the concept passed the
behavioral capability gate in every modality that claim involves. For the
principal three-modality verdict the required set is the stronger, simpler one:
**text, image and spoken audio, all three**, regardless of which pair the cell
happens to be. A concept that fails is ``CAPABILITY_INELIGIBLE``:

* it **stays** in the raw tables as the predeclared diagnostic it is;
* it is labelled, with the counts and the arithmetic that rejected it;
* it never enters a supporting-cell list, a bidirectional summary, a GO or a
  WEAK GO;
* and it is **never replaced** by another concept after the fact. Swapping in a
  concept chosen once the capability results were visible would make the
  selection depend on the outcome, which is the failure this whole study is
  built to avoid.

Nothing here erases, hides, or reinterprets a measurement. It decides what a
measurement is allowed to be evidence *for*.

This module is the single authority for that decision. Scattering the same
conditional across six report sites is how the original omission survived
review; there is one function, it is versioned, and its version is written into
every artifact that used it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from jlens.mmpilot.store import payload_checksum

#: Bound into every decision and into the amended report. Changing the rule's
#: *meaning* changes this string, so an artifact can always say which rule
#: produced it.
CLAIM_ADMISSIBILITY_RULE_VERSION = "mmpilot.claim_admissibility.v1"

#: The principal verdict's requirement: all three channels, for every cell.
PRINCIPAL_THREE_MODALITY_MODE = "principal_three_modality"
#: The weaker per-claim requirement: only the modalities the cell itself spans.
#: Reported as a diagnostic beside the principal decision, never as the gate for
#: the three-modality verdict.
PAIR_SPECIFIC_MODE = "pair_specific"

PRINCIPAL_REQUIRED_MODALITIES: tuple[str, ...] = ("text", "image", "spoken_audio")

#: Accuracy is a ratio of small integers and the threshold is a decimal literal,
#: so ``6/8 >= 0.7`` must not turn on the last bit of a float.
_TOLERANCE = 1e-9

DEFAULT_CAPABILITY_THRESHOLD = 0.7

#: The label a rejected concept carries everywhere it appears.
CAPABILITY_INELIGIBLE = "CAPABILITY_INELIGIBLE"
CAPABILITY_ELIGIBLE = "CAPABILITY_ELIGIBLE"


class CapabilityTableMissing(ValueError):
    """No per-concept capability table was supplied where one is required.

    Raised rather than defaulting to "admissible", because a silent default is
    the bug this module exists to make impossible.
    """


def resolve_capability_table(capability: Mapping | None) -> tuple[dict, float | None]:
    """Accept any of the shapes the capability result travels in.

    :func:`jlens.mmpilot.capability.capability_summary` and
    :func:`jlens.mmpilot.tri_modal.audio_capability_verdict` both expose
    ``per_concept`` and ``threshold``; a bare ``{concept: {modality: ...}}``
    mapping is accepted too, so a caller holding only the table does not have to
    wrap it.

    Raises:
        CapabilityTableMissing: When ``capability`` is None or carries no
            per-concept entries at all.
    """
    if capability is None:
        raise CapabilityTableMissing(
            "a per-concept capability table is required before any cell may "
            "support a claim; there is no admissible-by-default path"
        )
    if "per_concept" in capability:
        per_concept = dict(capability.get("per_concept") or {})
        threshold = capability.get("threshold")
        return per_concept, (None if threshold is None else float(threshold))
    per_concept = {
        str(concept): dict(entry)
        for concept, entry in capability.items()
        if isinstance(entry, Mapping)
    }
    if not per_concept:
        raise CapabilityTableMissing(
            f"the supplied capability mapping has no per-concept entries: "
            f"{sorted(capability)[:8]}"
        )
    return per_concept, None


def _observation(entry: Mapping | None, threshold: float) -> dict:
    """One (concept, modality) capability cell, re-derived from its counts."""
    if not entry:
        return {
            "present": False,
            "n": None,
            "n_correct": None,
            "accuracy": None,
            "passed": False,
            "stored_passed": None,
        }
    n = entry.get("n")
    n_correct = entry.get("n_correct")
    if n is not None and n_correct is not None and int(n) > 0:
        accuracy = int(n_correct) / int(n)
    else:
        accuracy = entry.get("accuracy")
        accuracy = None if accuracy is None else float(accuracy)
    passed = accuracy is not None and accuracy + _TOLERANCE >= threshold
    return {
        "present": True,
        "n": None if n is None else int(n),
        "n_correct": None if n_correct is None else int(n_correct),
        "accuracy": accuracy,
        "passed": bool(passed),
        # What the run *recorded* at the time, kept beside what this rule
        # derives now. They agree unless the threshold changed, and a
        # disagreement is worth seeing rather than silently resolving.
        "stored_passed": (
            None if entry.get("passed") is None else bool(entry.get("passed"))
        ),
    }


def _describe(modality: str, observation: Mapping, threshold: float) -> str:
    if not observation["present"]:
        return f"no capability record for {modality}"
    if observation["accuracy"] is None:
        return f"{modality} capability has no accuracy to test"
    if observation["n"] is not None and observation["n_correct"] is not None:
        counts = f"{observation['n_correct']}/{observation['n']} = "
    else:
        counts = ""
    return (
        f"{modality} capability {counts}"
        f"{observation['accuracy']:.1%} < {threshold:.1%}"
    )


def claim_admissibility(
    *,
    concept: str,
    source_modality: str | None,
    target_modality: str | None,
    capability: Mapping | None,
    threshold: float | None = None,
    principal_three_modality: bool = True,
    required_modalities: Sequence[str] | None = None,
) -> dict:
    """Decide whether one cell may support a claim, and say exactly why.

    Args:
        concept: The focal concept the cell intervenes on.
        source_modality: Where the direction was estimated from. ``None`` for a
            concept-level (pair-independent) question.
        target_modality: Where the effect was measured. ``None`` as above.
        capability: The per-concept capability result — a summary, a verdict, or
            a bare table (see :func:`resolve_capability_table`).
        threshold: The behavioral gate. Defaults to the one recorded in
            ``capability``, then to :data:`DEFAULT_CAPABILITY_THRESHOLD`.
        principal_three_modality: When True (the default and the mode the
            three-modality verdict uses), the concept must pass in text, image
            **and** spoken audio no matter which pair this cell is. When False,
            only the modalities this cell spans are required.
        required_modalities: Override the required set explicitly. Used by the
            tests and by callers asking a narrower question; it does not change
            the recorded ``rule_version``, only ``required_modalities``.

    Returns:
        A JSON-serializable decision carrying ``admissible``, the concept, the
        source and target modality, the required modalities, the observed counts
        and accuracies, the threshold, the rejection reason (``None`` when
        admissible) and the rule version.
    """
    per_concept, recorded_threshold = resolve_capability_table(capability)
    if threshold is not None:
        resolved_threshold = float(threshold)
        threshold_source = "argument"
    elif recorded_threshold is not None:
        resolved_threshold = float(recorded_threshold)
        threshold_source = "capability_record"
    else:
        resolved_threshold = DEFAULT_CAPABILITY_THRESHOLD
        threshold_source = "default"

    mode = (
        PRINCIPAL_THREE_MODALITY_MODE
        if principal_three_modality
        else PAIR_SPECIFIC_MODE
    )
    if required_modalities is not None:
        required = tuple(dict.fromkeys(str(m) for m in required_modalities))
    elif principal_three_modality:
        required = PRINCIPAL_REQUIRED_MODALITIES
    else:
        required = tuple(
            dict.fromkeys(
                str(modality)
                for modality in (source_modality, target_modality)
                if modality
            )
        )

    entry = per_concept.get(concept) or {}
    observed = {
        modality: _observation(entry.get(modality), resolved_threshold)
        for modality in required
    }
    # Every modality the concept has a record for, so the diagnostic shows the
    # channels that passed as well as the one that did not.
    observed_all = {
        modality: _observation(value, resolved_threshold)
        for modality, value in sorted(entry.items())
        if isinstance(value, Mapping)
    }

    missing = [m for m in required if not observed[m]["present"]]
    failing = [m for m in required if observed[m]["present"] and not observed[m]["passed"]]
    concept_known = bool(entry)

    if not concept_known:
        admissible = False
        reason = (
            f"{concept!r} has no capability record at all, so no claim about it "
            "can be admissible"
        )
    elif missing or failing:
        admissible = False
        parts = [_describe(m, observed[m], resolved_threshold) for m in (*missing, *failing)]
        scope = (
            "required for the principal three-modality claim"
            if principal_three_modality
            else f"required by this cell's pair {source_modality}->{target_modality}"
        )
        reason = f"{concept}: " + "; ".join(parts) + f" ({scope})"
    else:
        admissible = True
        reason = None

    pair = (
        f"{source_modality}->{target_modality}"
        if source_modality and target_modality
        else None
    )
    return {
        "schema": "jlens.mmpilot.claim_admissibility.v1",
        "rule_version": CLAIM_ADMISSIBILITY_RULE_VERSION,
        "rule_mode": mode,
        "admissible": admissible,
        "label": CAPABILITY_ELIGIBLE if admissible else CAPABILITY_INELIGIBLE,
        "concept": concept,
        "source_modality": source_modality,
        "target_modality": target_modality,
        "pair": pair,
        "required_modalities": list(required),
        "threshold": resolved_threshold,
        "threshold_source": threshold_source,
        "observed": observed,
        "observed_all_modalities": observed_all,
        "missing_modalities": missing,
        "failing_modalities": failing,
        "rejection_reason": reason,
        "post_hoc_replacement_allowed": False,
    }


# ------------------------------------------------------------- concept rosters


def concept_admissibility(
    concepts: Sequence[str],
    *,
    capability: Mapping | None,
    threshold: float | None = None,
    principal_three_modality: bool = True,
    required_modalities: Sequence[str] | None = None,
) -> dict:
    """The eligible/ineligible split over a **fixed** concept list.

    ``concepts`` is the predeclared set. It is never recomputed from the
    capability result, and nothing here can add a concept that was not asked
    for — an ineligible concept leaves a hole, it does not create a vacancy.
    """
    decisions = {
        concept: claim_admissibility(
            concept=concept,
            source_modality=None,
            target_modality=None,
            capability=capability,
            threshold=threshold,
            principal_three_modality=principal_three_modality,
            required_modalities=required_modalities,
        )
        for concept in concepts
    }
    eligible = [c for c in concepts if decisions[c]["admissible"]]
    excluded = [
        {
            "concept": c,
            "label": CAPABILITY_INELIGIBLE,
            "rejection_reason": decisions[c]["rejection_reason"],
            "failing_modalities": decisions[c]["failing_modalities"],
            "missing_modalities": decisions[c]["missing_modalities"],
            "observed": decisions[c]["observed"],
        }
        for c in concepts
        if not decisions[c]["admissible"]
    ]
    return {
        "schema": "jlens.mmpilot.concept_admissibility.v1",
        "rule_version": CLAIM_ADMISSIBILITY_RULE_VERSION,
        "rule_mode": decisions[concepts[0]]["rule_mode"] if concepts else None,
        "fixed_concepts": list(concepts),
        "eligible_concepts": eligible,
        "excluded_concepts": excluded,
        "excluded_concept_names": [entry["concept"] for entry in excluded],
        "decisions": decisions,
        "no_post_hoc_replacement": (
            "The fixed set was predeclared. An ineligible concept is excluded "
            "from every verdict and is not replaced — replacing it would make "
            "the selection depend on the capability result."
        ),
    }


def annotate_causal_cells(
    cells: Sequence[Mapping],
    *,
    capability: Mapping | None,
    threshold: float | None = None,
    principal_three_modality: bool = True,
) -> list[dict]:
    """Label every measured cell without removing or altering one.

    Adds, per cell:

    ``capability_admissible``
        Whether the concept cleared the gate in every required modality.
    ``capability_rejection_reason``
        The arithmetic that rejected it, or ``None``.
    ``counted_toward_verdict``
        ``capability_admissible and passes`` — the only flag any verdict reads.
    ``capability_decision``
        The full serialized decision, so a reader never has to trust the
        summary fields.
    ``capability_decision_pair_specific``
        The same question asked of this cell's own two modalities. Reported as
        a diagnostic; the principal verdict does not use it.
    """
    out: list[dict] = []
    for cell in cells:
        concept = cell["concept"]
        pair = str(cell.get("pair") or "")
        source, _, target = pair.partition("->")
        decision = claim_admissibility(
            concept=concept,
            source_modality=source or None,
            target_modality=target or None,
            capability=capability,
            threshold=threshold,
            principal_three_modality=principal_three_modality,
        )
        pair_specific = claim_admissibility(
            concept=concept,
            source_modality=source or None,
            target_modality=target or None,
            capability=capability,
            threshold=threshold,
            principal_three_modality=False,
        )
        out.append(
            {
                **cell,
                "capability_admissible": decision["admissible"],
                "capability_label": decision["label"],
                "capability_rejection_reason": decision["rejection_reason"],
                "capability_required_modalities": decision["required_modalities"],
                "counted_toward_verdict": bool(
                    decision["admissible"] and cell.get("passes")
                ),
                "capability_decision": decision,
                "capability_decision_pair_specific": pair_specific,
            }
        )
    return out


# ------------------------------------------------------------------ the rule


def admissibility_rule_record(
    *,
    threshold: float,
    principal_three_modality: bool = True,
    required_modalities: Sequence[str] | None = None,
) -> dict:
    """The rule itself, checksummed, so an amended artifact can bind to it.

    This is deliberately separate from the raw-generation fingerprint. The units
    were produced under one contract; the report is produced under another, and
    only the second one changed here. Versioning them together would refuse a
    completed run merely because the reporting rule was corrected.
    """
    required = list(
        required_modalities
        if required_modalities is not None
        else (
            PRINCIPAL_REQUIRED_MODALITIES
            if principal_three_modality
            else ("<pair-specific>",)
        )
    )
    payload = {
        "rule_version": CLAIM_ADMISSIBILITY_RULE_VERSION,
        "rule_mode": (
            PRINCIPAL_THREE_MODALITY_MODE
            if principal_three_modality
            else PAIR_SPECIFIC_MODE
        ),
        "capability_threshold": float(threshold),
        "required_modalities": required,
        "statement": (
            "A causal cell may support a scientific claim only when the concept "
            "passed the behavioral capability gate in every modality involved in "
            "that claim. For the principal three-modality verdict the required "
            "set is text, image and spoken_audio."
        ),
        "failed_concept_policy": [
            "remains in raw tables as a predeclared diagnostic",
            "labelled CAPABILITY_INELIGIBLE",
            "cannot count toward supporting-cell totals",
            "cannot count toward bidirectional transfer",
            "cannot satisfy GO or WEAK GO criteria",
            "cannot be replaced post hoc by another concept",
        ],
    }
    return {**payload, "rule_checksum": payload_checksum(payload)}


__all__ = [
    "CAPABILITY_ELIGIBLE",
    "CAPABILITY_INELIGIBLE",
    "CLAIM_ADMISSIBILITY_RULE_VERSION",
    "DEFAULT_CAPABILITY_THRESHOLD",
    "PAIR_SPECIFIC_MODE",
    "PRINCIPAL_REQUIRED_MODALITIES",
    "PRINCIPAL_THREE_MODALITY_MODE",
    "CapabilityTableMissing",
    "admissibility_rule_record",
    "annotate_causal_cells",
    "claim_admissibility",
    "concept_admissibility",
    "resolve_capability_table",
]
