# Gemma 4 adaptation — new in this fork (github.com/anthropics/jacobian-lens is upstream).
# SPDX-License-Identifier: Apache-2.0
"""What the model is allowed to see, and what only the scorer may see.

The completed three-modality causal result asked its behavioral question by
**listing every candidate in the prompt**::

    Question: which one of these is present: bird, cat, giraffe, microwave,
    toilet, zebra? Answer with exactly one word.
    Answer:

That question is still exactly where it was
(:func:`jlens.mmpilot.capability.build_question`), it still produces the same
bytes, and the result it produced remains valid — as a *candidate-conditioned*
identification result. Every candidate concept was introduced into the prompt,
so the model's J-space representation of each of them may have been primed by
the ask itself. Source-derived positive-minus-negative estimation subtracts the
shared prompt components to first order and candidate-order invariance rules out
ordering bias, but neither removes semantic priming.

Anthropic's strongest Global Workspace interventions do not list candidates. The
internal-reasoning example — *"The number of legs on the animal that spins webs
is ..."* — never writes ``spider`` and never writes ``ant``; the flexible
generalization example — *"The capital of France is ..."* — contains the source
but not the swap target. This module is what lets the planned coordinate-swap
study ask that way: **the question and the scored candidates are separate
objects**, and only the question is ever built into a prompt.

Four protocols, and the difference between them is a claim boundary
==================================================================

``mmpilot.candidate_listed_identification.v1``
    The legacy, completed protocol. Every candidate appears in the prompt.
    Supports a *candidate-conditioned* identification claim and nothing
    stronger. Its bytes, its defaults and its recorded protocol string
    (:data:`LEGACY_CAPABILITY_PROMPT_PROTOCOL`) are frozen — completed runs
    resume unchanged.

``mmpilot.open_identification.v1``
    No candidate list in the instruction. The source identity may occur
    naturally in written evidence, and that is *recorded* rather than hidden.
    The swap target must not occur anywhere the model can see, and for spoken
    audio must not occur in the offline transcript either.

``mmpilot.open_downstream_property.v1``
    Every open-identification rule, plus: no entity candidates and no answer
    choices in the question, and the target's property answer must not appear in
    the model-visible prompt. Whether the *source's* property answer occurs
    naturally is recorded.

``mmpilot.hidden_intermediate.v1``
    Neither entity label nor any registered alias may appear in any
    model-visible text, and for spoken audio neither may appear in the offline
    transcript. The entity has to be inferred from a description. This is the
    only protocol that supports a multi-hop reasoning claim, and it is refused
    outright rather than silently downgraded when the audit cannot clear it.

An open prompt is **not** hidden-intermediate reasoning merely because the
candidate list is absent. ``open_identification`` still lets the source appear
in the evidence; ``hidden_intermediate`` does not.

The leakage audit is deterministic
==================================

Unicode normalization, case folding, punctuation and whitespace normalization,
whole-token matching, and **registered aliases**. No language model is asked
whether a prompt "hints at" a concept — that check would be unreliable in
exactly the cases that matter. The cost is stated in :data:`AUDIT_LIMITS`: a
paraphrase nobody registered is not detected, and the audit says so rather than
implying coverage it does not have.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from jlens.mmpilot.backend import text_hash
from jlens.mmpilot.coordinate_swap import PROMPT_BOUNDARY_RULE
from jlens.mmpilot.store import payload_checksum

# ------------------------------------------------------------------ versions

#: The legacy, completed protocol. Its identifier is new; the *prompt* it names
#: is byte-for-byte the one :mod:`jlens.mmpilot.capability` has always built.
CANDIDATE_LISTED_IDENTIFICATION = "mmpilot.candidate_listed_identification.v1"
OPEN_IDENTIFICATION = "mmpilot.open_identification.v1"
OPEN_DOWNSTREAM_PROPERTY = "mmpilot.open_downstream_property.v1"
HIDDEN_INTERMEDIATE = "mmpilot.hidden_intermediate.v1"

PROTOCOLS: tuple[str, ...] = (
    CANDIDATE_LISTED_IDENTIFICATION,
    OPEN_IDENTIFICATION,
    OPEN_DOWNSTREAM_PROPERTY,
    HIDDEN_INTERMEDIATE,
)

#: Protocols whose instruction carries no candidate list.
OPEN_PROTOCOLS: tuple[str, ...] = (
    OPEN_IDENTIFICATION,
    OPEN_DOWNSTREAM_PROPERTY,
    HIDDEN_INTERMEDIATE,
)

#: The string completed runs recorded as ``prompt_protocol`` /
#: ``capability_protocol``. Re-exported, never redefined: changing it would
#: change :func:`jlens.mmpilot.pipeline.scientific_fingerprint` and refuse every
#: completed run's resume.
LEGACY_CAPABILITY_PROMPT_PROTOCOL = "gemma-it-chat-balanced-options-v1"

AUDIT_VERSION = "jlens.mmpilot.prompt_leakage_audit.v1"
CLAIM_RULE_VERSION = "mmpilot.prompt_protocol_claim_admissibility.v1"

#: What the external scorer does, named so a fingerprint binds it.
CANDIDATE_SCORING_VERSION = (
    "jlens.mmpilot.capability.score_candidate_sequences."
    "complete_teacher_forced_sequence.v1"
)

#: The one sentence that defines "external".
CANDIDATE_VISIBILITY_RULE = (
    "candidate answer strings are supplied only to the external teacher-forced "
    "scorer; they are never interpolated into the model-visible prompt, and the "
    "prompt hash is therefore independent of the candidate set and of its "
    "enumeration order"
)

MODALITIES: tuple[str, ...] = ("text", "image", "spoken_audio")


class PromptProtocolError(ValueError):
    """An invalid protocol request, evidence object, or candidate set."""


class PromptLeakageError(PromptProtocolError):
    """The registered deterministic audit refused this prompt.

    Raised instead of downgrading to a weaker protocol. A ``hidden_intermediate``
    prompt whose transcript names the source is not an ``open_identification``
    prompt — it is a prompt that failed, and the caller has to fix the evidence.
    """


# ------------------------------------------------------------------ questions

#: The primary open identification question. It names no animal, lists nothing,
#: and is byte-identical across the three evidence modalities.
OPEN_IDENTIFICATION_QUESTION = (
    "What animal is present in the evidence? Answer with the animal name.\n"
    "Answer:"
)

#: The downstream-property question. No entity candidates, no numeric choices.
OPEN_PROPERTY_QUESTION = (
    "How many legs does the animal typically have? Answer with a number.\n"
    "Answer:"
)

#: ``hidden_intermediate`` asks the same property question. What changes is the
#: evidence: the entity is described rather than named, and the audit refuses
#: both entity labels everywhere the model can see and in the transcript.
HIDDEN_INTERMEDIATE_QUESTION = OPEN_PROPERTY_QUESTION

DEFAULT_QUESTIONS: dict[str, str] = {
    OPEN_IDENTIFICATION: OPEN_IDENTIFICATION_QUESTION,
    OPEN_DOWNSTREAM_PROPERTY: OPEN_PROPERTY_QUESTION,
    HIDDEN_INTERMEDIATE: HIDDEN_INTERMEDIATE_QUESTION,
}

#: How written evidence is joined to the question. The image and spoken-audio
#: conditions carry no evidence text at all — the media is the evidence.
OPEN_TEXT_EVIDENCE_TEMPLATE = "Evidence: {evidence}\n{question}"


# ------------------------------------------------------------- normalization

_COMBINING = re.compile("[̀-ͯ]")
_NON_WORD = re.compile(r"[^\w\s]", re.UNICODE)

NORMALIZATION_RULE = (
    "NFKC -> casefold -> NFKC, underscores and every non-word character mapped "
    "to a space, whitespace collapsed; matches are whole-token only, so 'cat' "
    "does not match 'concatenate' and plural forms must be registered aliases"
)

#: Stated, not implied.
AUDIT_LIMITS: tuple[str, ...] = (
    "deterministic string matching only — no model is asked whether a prompt "
    "implies a concept",
    "a paraphrase, hypernym, or description that is not a registered alias is "
    "NOT detected",
    "image pixels are never read; an image containing rendered text naming the "
    "target is outside this audit and must be excluded by the image audit",
    "a spoken-audio transcript is only as good as the transcription; an "
    "unregistered mis-transcription of the target is not detected",
    "morphological variants (plurals, possessives, compounds) are detected only "
    "when registered as aliases",
)


def normalize(text: object) -> str:
    """Case-folded, punctuation-stripped, whitespace-collapsed NFKC text.

    The full caseless-matching sequence — ``NFKC(casefold(NFKC(x)))`` — because
    a single pass is not idempotent for every script, and a prompt that differs
    from its audited form only by a composed character is the same prompt.
    """
    value = unicodedata.normalize("NFKC", str(text))
    value = unicodedata.normalize("NFKC", value.casefold())
    value = _COMBINING.sub("", unicodedata.normalize("NFD", value))
    value = unicodedata.normalize("NFKC", value)
    value = value.replace("_", " ")
    value = _NON_WORD.sub(" ", value)
    return " ".join(value.split())


def contains_surface(haystack: str, surface: str) -> bool:
    """Whole-token containment of ``surface`` in already-normalized ``haystack``."""
    needle = normalize(surface)
    if not needle:
        return False
    return re.search(rf"(?<!\S){re.escape(needle)}(?!\S)", haystack) is not None


#: A rendered candidate list: three or more short comma-separated items closed
#: by ``?`` or ``.``, optionally with a final ``or``. This is the *shape* of
#: "bird, cat, giraffe, microwave, toilet, zebra?" and is detected even when the
#: items are not the registered candidates.
_ENUMERATION = re.compile(
    r"(?:[\w][\w\- ]{0,24},\s*){2,}(?:or\s+)?[\w][\w\- ]{0,24}\s*[?.]"
)


def looks_like_candidate_enumeration(text: str) -> bool:
    """Whether ``text`` renders a comma-separated list of answer options."""
    return _ENUMERATION.search(str(text)) is not None


# ------------------------------------------------------------------ concepts


@dataclass(frozen=True)
class ConceptSpec:
    """A concept and every written form the audit is entitled to refuse.

    Attributes:
        name: The concept as the experiment names it (``"bird"``).
        aliases: Registered additional surface forms — plurals, spellings,
            common synonyms. The audit detects **only** what is registered here;
            see :data:`AUDIT_LIMITS`.
    """

    name: str
    aliases: tuple[str, ...] = ()

    @property
    def surface_forms(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for form in (self.name, *self.aliases):
            key = normalize(form)
            if key:
                seen.setdefault(key, None)
        return tuple(seen)

    def to_dict(self) -> dict:
        return {"name": self.name, "aliases": list(self.aliases)}


#: Registered aliases for the concepts the planned bird -> cat study names.
#: Deliberately conservative: plurals only. A wrong alias would refuse a clean
#: prompt, and a missing one is a stated limit rather than a silent pass.
DEFAULT_ALIASES: dict[str, tuple[str, ...]] = {
    "bird": ("birds",),
    "cat": ("cats",),
    "spider": ("spiders",),
    "ant": ("ants",),
}


def concept_spec(name: str, aliases: Sequence[str] | None = None) -> ConceptSpec:
    """A :class:`ConceptSpec`, defaulting to :data:`DEFAULT_ALIASES`."""
    registered = tuple(aliases) if aliases is not None else DEFAULT_ALIASES.get(name, ())
    return ConceptSpec(name=str(name), aliases=tuple(str(a) for a in registered))


def aliases_checksum(*concepts: ConceptSpec | None) -> str:
    """Checksum over every registered surface form, so a fingerprint binds them."""
    payload = {
        spec.name: sorted(spec.surface_forms) for spec in concepts if spec is not None
    }
    return payload_checksum(payload)


def _surfaces(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


# ------------------------------------------------------------------ evidence


@dataclass(frozen=True)
class Evidence:
    """One evidence channel, split by *who may see what*.

    Attributes:
        modality: ``text``, ``image`` or ``spoken_audio``.
        text: Model-visible written evidence. ``text`` modality only; a caption
            is legitimate evidence and may naturally contain the source.
        media: The opaque media payload handed to the backend — a decoded image,
            or a waveform array. Never inspected here.
        media_reference: Path or dataset id. **Audit only.** It is checksummed
            and checked for exposure; it is never written into a prompt.
        media_checksum: Recorded provenance for the media bytes.
        transcript: The spoken caption's written form. **Audit only, always.**
            It is never passed to a backend and never joined to a prompt.
        sampling_rate: Passed through to the backend for ``spoken_audio``.
    """

    modality: str
    text: str | None = None
    media: Any = None
    media_reference: str | None = None
    media_checksum: str | None = None
    transcript: str | None = None
    sampling_rate: int | None = None

    def __post_init__(self) -> None:
        if self.modality not in MODALITIES:
            raise PromptProtocolError(
                f"unknown modality {self.modality!r}; known: {MODALITIES}"
            )
        if self.modality == "text":
            if not (self.text or "").strip():
                raise PromptProtocolError(
                    "the text condition needs written evidence; there is no other "
                    "channel for it to arrive through"
                )
            if self.transcript is not None:
                raise PromptProtocolError(
                    "a transcript belongs to the spoken_audio condition. In the "
                    "text condition the written evidence is model-visible and is "
                    "the `text` field."
                )
        else:
            if self.text is not None:
                raise PromptProtocolError(
                    f"the {self.modality} condition carries no model-visible text: "
                    "the media is the only evidence. A caption here would make the "
                    "condition answerable without the media."
                )

    def to_dict(self) -> dict:
        """Provenance, never content. The transcript is recorded as a hash."""
        return {
            "modality": self.modality,
            "has_visible_text_evidence": bool(self.text),
            "media_reference_checksum": (
                text_hash(self.media_reference) if self.media_reference else None
            ),
            "media_checksum": self.media_checksum,
            "transcript_present": self.transcript is not None,
            "transcript_hash": (
                text_hash(self.transcript) if self.transcript is not None else None
            ),
            "transcript_is_model_visible": False,
        }


# ------------------------------------------------------------- leakage audit

LEAKAGE_CATEGORIES: tuple[str, ...] = (
    "instruction_candidate_leakage",
    "source_in_visible_evidence",
    "target_in_visible_evidence",
    "source_in_audio_transcript",
    "target_in_audio_transcript",
    "property_answer_in_prompt",
    "semantic_filename_exposure",
    "candidate_enumeration_detected",
)

REFUSE, RECORD = "refuse", "record"

#: Per-protocol policy. ``refuse`` fails the audit; ``record`` reports the
#: finding and passes. There is no third action that quietly rewrites the
#: protocol.
LEAKAGE_POLICY: dict[str, dict[str, str]] = {
    CANDIDATE_LISTED_IDENTIFICATION: {
        "instruction_candidate_leakage": RECORD,
        "candidate_enumeration_detected": RECORD,
        "source_in_visible_evidence": RECORD,
        "target_in_visible_evidence": RECORD,
        "source_in_audio_transcript": RECORD,
        "target_in_audio_transcript": RECORD,
        "property_answer_in_prompt": RECORD,
        "semantic_filename_exposure": REFUSE,
    },
    OPEN_IDENTIFICATION: {
        "instruction_candidate_leakage": REFUSE,
        "candidate_enumeration_detected": REFUSE,
        "source_in_visible_evidence": RECORD,
        "target_in_visible_evidence": REFUSE,
        "source_in_audio_transcript": RECORD,
        "target_in_audio_transcript": REFUSE,
        "property_answer_in_prompt": RECORD,
        "semantic_filename_exposure": REFUSE,
    },
    OPEN_DOWNSTREAM_PROPERTY: {
        "instruction_candidate_leakage": REFUSE,
        "candidate_enumeration_detected": REFUSE,
        "source_in_visible_evidence": RECORD,
        "target_in_visible_evidence": REFUSE,
        "source_in_audio_transcript": RECORD,
        "target_in_audio_transcript": REFUSE,
        "property_answer_in_prompt": REFUSE,
        "semantic_filename_exposure": REFUSE,
    },
    HIDDEN_INTERMEDIATE: {
        "instruction_candidate_leakage": REFUSE,
        "candidate_enumeration_detected": REFUSE,
        "source_in_visible_evidence": REFUSE,
        "target_in_visible_evidence": REFUSE,
        "source_in_audio_transcript": REFUSE,
        "target_in_audio_transcript": REFUSE,
        "property_answer_in_prompt": REFUSE,
        "semantic_filename_exposure": REFUSE,
    },
}


def _finding(
    category: str,
    action: str,
    matches: Sequence[Mapping],
    *,
    applicable: bool = True,
    auditable: bool = True,
    recorded_matches: Sequence[Mapping] = (),
    note: str = "",
) -> dict:
    """One audited category, with the status the policy assigns it."""
    detected = bool(matches)
    if not applicable:
        status = "not_applicable"
    elif not auditable:
        status = "unauditable"
    elif detected:
        status = "violation" if action == REFUSE else "recorded"
    else:
        status = "clean"
    return {
        "category": category,
        "action": action,
        "applicable": bool(applicable),
        "auditable": bool(auditable),
        "detected": detected,
        "status": status,
        "matches": [dict(match) for match in matches],
        "recorded_matches": [dict(match) for match in recorded_matches],
        "note": note,
    }


def _scan(scopes: Mapping[str, str], surfaces: Sequence[str]) -> list[dict]:
    """Every ``(surface, scope)`` hit, in a stable order."""
    hits: list[dict] = []
    for scope in sorted(scopes):
        haystack = scopes[scope]
        if not haystack:
            continue
        for surface in surfaces:
            if contains_surface(haystack, surface):
                hits.append({"surface": surface, "scope": scope})
    return hits


def audit_prompt_leakage(
    *,
    protocol: str,
    modality: str,
    instruction: str,
    visible_evidence_text: str | None = None,
    transcript: str | None = None,
    source: ConceptSpec | None = None,
    target: ConceptSpec | None = None,
    external_candidates: Sequence[str] = (),
    property_answers: Mapping[str, Any] | None = None,
    media_reference: str | None = None,
) -> dict:
    """Run the registered deterministic audit and report every category.

    Args:
        protocol: One of :data:`PROTOCOLS`; selects the policy.
        modality: One of :data:`MODALITIES`. Transcript categories are
            ``not_applicable`` outside ``spoken_audio``.
        instruction: The question text, without the evidence.
        visible_evidence_text: Written evidence the model can see, or ``None``.
        transcript: The spoken caption's written form. Read **here** and nowhere
            else. A ``spoken_audio`` condition whose transcript is missing is
            reported ``unauditable``, and that fails any protocol whose policy
            refuses transcript leakage — an unchecked transcript is not a clean
            one.
        source / target: The concept specs whose registered surface forms are
            searched for. Audit metadata; they never enter a prompt.
        external_candidates: The strings the scorer will score. Searched for in
            the instruction, which is what makes "external" checkable.
        property_answers: ``{"source": ..., "target": ...}``; each value is a
            string or a sequence of registered surface forms for that answer.
        media_reference: A path or dataset id, searched for in the visible text.

    Returns:
        The audit record. ``passed`` is True when no category has status
        ``violation`` or ``unauditable``.

    Never raises for a finding — a finding is the result. Use
    :func:`assert_prompt_leakage_clean` to turn a failing record into a refusal.
    """
    if protocol not in LEAKAGE_POLICY:
        raise PromptProtocolError(f"unknown prompt protocol {protocol!r}; known: {PROTOCOLS}")
    if modality not in MODALITIES:
        raise PromptProtocolError(f"unknown modality {modality!r}; known: {MODALITIES}")

    policy = LEAKAGE_POLICY[protocol]
    instruction_normalized = normalize(instruction)
    evidence_normalized = normalize(visible_evidence_text or "")
    transcript_normalized = normalize(transcript or "")
    visible = {"instruction": instruction_normalized, "visible_evidence": evidence_normalized}
    transcript_scope = {"audio_transcript": transcript_normalized}

    is_audio = modality == "spoken_audio"
    transcript_auditable = (not is_audio) or transcript is not None

    findings: dict[str, dict] = {}

    candidates = [str(candidate) for candidate in external_candidates]
    findings["instruction_candidate_leakage"] = _finding(
        "instruction_candidate_leakage",
        policy["instruction_candidate_leakage"],
        _scan({"instruction": instruction_normalized}, candidates),
        note="external candidate strings occurring in the model-visible question",
    )
    enumeration_matches: list[dict] = []
    if looks_like_candidate_enumeration(instruction):
        enumeration_matches.append({"surface": "<comma-separated option list>", "scope": "instruction"})
    named = findings["instruction_candidate_leakage"]["matches"]
    if len(named) >= 2:
        enumeration_matches.append(
            {"surface": "|".join(sorted(hit["surface"] for hit in named)), "scope": "instruction"}
        )
    findings["candidate_enumeration_detected"] = _finding(
        "candidate_enumeration_detected",
        policy["candidate_enumeration_detected"],
        enumeration_matches,
        note="a rendered list of answer options in the instruction",
    )

    source_forms = source.surface_forms if source else ()
    target_forms = target.surface_forms if target else ()
    findings["source_in_visible_evidence"] = _finding(
        "source_in_visible_evidence",
        policy["source_in_visible_evidence"],
        _scan(visible, source_forms),
        applicable=source is not None,
        note="the source concept in text the model can see",
    )
    findings["target_in_visible_evidence"] = _finding(
        "target_in_visible_evidence",
        policy["target_in_visible_evidence"],
        _scan(visible, target_forms),
        applicable=target is not None,
        note="the swap target in text the model can see",
    )
    findings["source_in_audio_transcript"] = _finding(
        "source_in_audio_transcript",
        policy["source_in_audio_transcript"],
        _scan(transcript_scope, source_forms),
        applicable=is_audio and source is not None,
        auditable=transcript_auditable,
        note="offline transcript only; never model-visible",
    )
    findings["target_in_audio_transcript"] = _finding(
        "target_in_audio_transcript",
        policy["target_in_audio_transcript"],
        _scan(transcript_scope, target_forms),
        applicable=is_audio and target is not None,
        auditable=transcript_auditable,
        note="offline transcript only; never model-visible",
    )

    answers = dict(property_answers or {})
    target_answer_forms = _surfaces(answers.get("target"))
    source_answer_forms = _surfaces(answers.get("source"))
    # The source's own answer is one of the answer choices, and it is explicitly
    # permitted to occur naturally ("standing on two legs" in a bird caption).
    # So it is excluded from the refused set and recorded separately; the target
    # answer and every *other* choice stay refused.
    permitted = {normalize(form) for form in source_answer_forms}
    refused_forms = [
        form
        for form in (*target_answer_forms, *candidates)
        if normalize(form) not in permitted
    ]
    refused_answer_hits = _scan(visible, refused_forms)
    source_answer_hits = _scan(visible, source_answer_forms)
    if policy["property_answer_in_prompt"] == REFUSE and protocol == HIDDEN_INTERMEDIATE:
        # A source property answer in the prompt trivially reveals the
        # intermediate the protocol exists to hide.
        refused_answer_hits = [*refused_answer_hits, *source_answer_hits]
        source_answer_hits = []
    findings["property_answer_in_prompt"] = _finding(
        "property_answer_in_prompt",
        policy["property_answer_in_prompt"],
        refused_answer_hits,
        applicable=bool(answers),
        recorded_matches=source_answer_hits,
        note="the target property answer, or any answer choice, in the visible prompt",
    )

    reference_forms: list[str] = []
    if media_reference:
        stem = str(media_reference).replace("\\", "/").rsplit("/", 1)[-1]
        reference_forms = [form for form in {stem, stem.rsplit(".", 1)[0]} if form]
    findings["semantic_filename_exposure"] = _finding(
        "semantic_filename_exposure",
        policy["semantic_filename_exposure"],
        _scan(visible, reference_forms),
        applicable=bool(reference_forms),
        note="a media file name or dataset id reaching the model-visible prompt",
    )

    violations = sorted(
        name for name, row in findings.items() if row["status"] == "violation"
    )
    unauditable = sorted(
        name
        for name, row in findings.items()
        if row["status"] == "unauditable" and row["action"] == REFUSE
    )
    return {
        "schema": "jlens.mmpilot.prompt_leakage_audit.v1",
        "audit_version": AUDIT_VERSION,
        "protocol": protocol,
        "modality": modality,
        "policy": dict(policy),
        "normalization": NORMALIZATION_RULE,
        "limits": list(AUDIT_LIMITS),
        "registered_aliases_checksum": aliases_checksum(source, target),
        "source_concept": source.to_dict() if source else None,
        "target_concept": target.to_dict() if target else None,
        "external_candidates": sorted(candidates),
        "transcript_audited": bool(is_audio and transcript is not None),
        "transcript_is_model_visible": False,
        "findings": findings,
        "violations": violations,
        "unauditable": unauditable,
        "recorded": sorted(
            name for name, row in findings.items() if row["status"] == "recorded"
        ),
        "source_in_visible_evidence": findings["source_in_visible_evidence"]["detected"],
        "source_property_answer_present": bool(
            findings["property_answer_in_prompt"]["recorded_matches"]
        ),
        "passed": not violations and not unauditable,
    }


def assert_prompt_leakage_clean(record: Mapping) -> dict:
    """Return ``record``, or refuse it. Never downgrades the protocol.

    Raises:
        PromptLeakageError: If any category the protocol refuses was detected,
            or if a category it refuses could not be audited at all.
    """
    if record.get("passed"):
        return dict(record)
    lines: list[str] = []
    for name in record.get("violations", ()):
        finding = record["findings"][name]
        surfaces = ", ".join(
            f"{hit['surface']!r} in {hit['scope']}" for hit in finding["matches"]
        )
        lines.append(f"{name}: {surfaces}")
    for name in record.get("unauditable", ()):
        lines.append(
            f"{name}: no transcript was supplied for a spoken_audio condition, so "
            "this check could not run. An unchecked transcript is not a clean one."
        )
    raise PromptLeakageError(
        f"prompt protocol {record.get('protocol')!r} refuses this prompt:\n  - "
        + "\n  - ".join(lines)
        + "\n\nThe protocol is not downgraded to a weaker one. Fix the evidence, "
        "the question, or the registered aliases."
    )


# ------------------------------------------------------------- built prompt


@dataclass(frozen=True)
class BuiltPrompt:
    """One protocol-checked prompt, with the model side and the scorer side apart.

    Attributes:
        protocol: The protocol identifier this was built under.
        modality: The evidence channel.
        question: The neutral instruction, exactly as it entered the prompt.
        question_hash: ``sha256`` prefix of :attr:`question`.
        model_visible_prompt: Everything the model reads as text. For ``image``
            and ``spoken_audio`` this **is** :attr:`question`.
        media_input: The opaque media payload for the backend, or ``None``.
        prompt_hash: ``sha256`` prefix of :attr:`model_visible_prompt`. Independent
            of the candidate set for every open protocol, because the candidates
            are not in the prompt.
        prompt_token_ids: Tokenization of :attr:`model_visible_prompt`, when a
            tokenizer was supplied. **Not** the model's final sequence: a
            processor expands image and audio placeholders, so the authoritative
            prompt/candidate boundary is
            :attr:`jlens.mmpilot.backend.BuiltInputs.prompt_len` and nothing here.
        prompt_len: ``len(prompt_token_ids)``, with the same caveat.
        external_candidates: The strings the scorer will score, in the order
            given. Order never reaches the prompt.
        external_candidate_token_ids: Complete token sequences per candidate.
        candidate_visibility: The rule and the checked fact that it held.
        leakage: The full :func:`audit_prompt_leakage` record.
        protocol_version: Same as :attr:`protocol`; named separately because that
            is what artifacts call the field.
    """

    protocol: str
    modality: str
    question: str
    question_hash: str
    model_visible_prompt: str
    media_input: Any
    prompt_hash: str
    prompt_token_ids: tuple[int, ...] | None
    prompt_len: int | None
    external_candidates: tuple[str, ...]
    external_candidate_token_ids: dict[str, list[int]] | None
    candidate_visibility: dict
    leakage: dict
    evidence: dict
    source_concept: dict | None
    target_concept: dict | None
    property_answers: dict
    sampling_rate: int | None = None
    media_reference: str | None = None
    media_checksum: str | None = None
    transcript_hash: str | None = None
    audit_only_fields: tuple[str, ...] = (
        "source_concept",
        "target_concept",
        "property_answers",
        "media_reference",
        "transcript_hash",
    )

    @property
    def protocol_version(self) -> str:
        return self.protocol

    @property
    def candidates_are_external(self) -> bool:
        return bool(self.candidate_visibility.get("candidates_in_prompt") is False)

    def to_dict(self) -> dict:
        """JSON-safe record. Carries no transcript and no media payload."""
        return {
            "protocol": self.protocol,
            "protocol_version": self.protocol_version,
            "modality": self.modality,
            "question": self.question,
            "question_hash": self.question_hash,
            "model_visible_prompt": self.model_visible_prompt,
            "prompt_hash": self.prompt_hash,
            "prompt_token_ids": (
                list(self.prompt_token_ids) if self.prompt_token_ids is not None else None
            ),
            "prompt_len": self.prompt_len,
            "prompt_len_authority": (
                "jlens.mmpilot.backend.BuiltInputs.prompt_len (a processor expands "
                "image/audio placeholders; prompt_len here counts text tokens only)"
            ),
            "external_candidates": list(self.external_candidates),
            "external_candidate_token_ids": (
                {k: list(v) for k, v in self.external_candidate_token_ids.items()}
                if self.external_candidate_token_ids is not None
                else None
            ),
            "candidate_visibility": dict(self.candidate_visibility),
            "leakage_audit": dict(self.leakage),
            "evidence": dict(self.evidence),
            "source_concept": self.source_concept,
            "target_concept": self.target_concept,
            "property_answers": dict(self.property_answers),
            "media_checksum": self.media_checksum,
            "transcript_hash": self.transcript_hash,
            "audit_only_fields": list(self.audit_only_fields),
        }


def _normalize_candidates(candidates: Sequence[str] | Mapping[str, Any]) -> tuple[str, ...]:
    names = tuple(str(name) for name in candidates)
    if not names:
        raise PromptProtocolError(
            "an external candidate set is required: the question deliberately "
            "names no answers, so something has to tell the scorer what to score"
        )
    if len(set(names)) != len(names):
        raise PromptProtocolError(f"duplicate external candidates: {names}")
    return names


def build_protocol_prompt(
    *,
    protocol: str,
    evidence: Evidence,
    external_candidates: Sequence[str] | Mapping[str, Any],
    question: str | None = None,
    source: ConceptSpec | None = None,
    target: ConceptSpec | None = None,
    property_answers: Mapping[str, Any] | None = None,
    encode_candidate: Callable[[str], Sequence[int]] | None = None,
    encode_prompt: Callable[[str], Sequence[int]] | None = None,
    legacy_candidate_list: Sequence[str] | None = None,
    strict: bool = True,
) -> BuiltPrompt:
    """Build one prompt under ``protocol``, with the candidates kept outside it.

    Args:
        protocol: One of :data:`PROTOCOLS`.
        evidence: The evidence channel. Its ``transcript`` is read by the audit
            and by nothing else.
        external_candidates: What the scorer will score. Never interpolated.
        question: The neutral instruction. Defaults to this protocol's template
            (:data:`DEFAULT_QUESTIONS`). For
            :data:`CANDIDATE_LISTED_IDENTIFICATION` the default is the legacy
            question built by :func:`jlens.mmpilot.capability.build_question`
            from ``legacy_candidate_list`` — byte-for-byte what completed runs
            used.
        source / target: Concept specs, for auditing only.
        property_answers: ``{"source": ..., "target": ...}`` surface forms.
        encode_candidate: ``str -> [token ids]``. Supplied, every candidate's
            **complete** token sequence is recorded here and handed to the
            scorer; the leading space is added exactly as
            :func:`jlens.mmpilot.capability.candidate_token_ids` adds it.
        encode_prompt: ``str -> [token ids]`` for the model-visible text.
        legacy_candidate_list: Candidate-listed protocol only. The concepts the
            legacy question enumerates.
        strict: Refuse a prompt the audit rejects. Turning it off returns the
            record instead of raising and is for *inspecting* a refusal, never
            for running one.

    Raises:
        PromptProtocolError: On an unknown protocol, a candidate list supplied to
            an open protocol, or an open question that interpolates a candidate.
        PromptLeakageError: When ``strict`` and the audit refuses.
    """
    if protocol not in PROTOCOLS:
        raise PromptProtocolError(f"unknown prompt protocol {protocol!r}; known: {PROTOCOLS}")
    names = _normalize_candidates(external_candidates)

    if protocol == CANDIDATE_LISTED_IDENTIFICATION:
        if question is None:
            from jlens.mmpilot.capability import build_question

            if not legacy_candidate_list:
                raise PromptProtocolError(
                    "the candidate-listed protocol builds its question from the "
                    "candidate list; pass legacy_candidate_list (or the exact "
                    "question) so the legacy bytes are reproduced rather than "
                    "reinvented"
                )
            question = build_question(list(legacy_candidate_list))
    else:
        if legacy_candidate_list is not None:
            raise PromptProtocolError(
                f"{protocol} never renders a candidate list; legacy_candidate_list "
                "is only meaningful for "
                f"{CANDIDATE_LISTED_IDENTIFICATION}"
            )
        if question is None:
            question = DEFAULT_QUESTIONS[protocol]
    if protocol in (OPEN_DOWNSTREAM_PROPERTY, HIDDEN_INTERMEDIATE) and not property_answers:
        raise PromptProtocolError(
            f"{protocol} scores a downstream property, so the source and target "
            "property answers must be declared (property_answers={'source': ..., "
            "'target': ...}). Without them the audit has nothing to refuse and "
            "would pass by omission."
        )

    question = str(question)
    if protocol in OPEN_PROTOCOLS:
        interpolated = sorted(
            name for name in names if contains_surface(normalize(question), name)
        )
        if interpolated:
            raise PromptProtocolError(
                f"the {protocol} question names the external candidate(s) "
                f"{interpolated}. The whole point of an open prompt is that the "
                "answers exist only in the scorer; refusing to build it."
            )

    if evidence.modality == "text":
        model_visible_prompt = (
            f"Caption: {evidence.text.strip()}\n{question}"
            if protocol == CANDIDATE_LISTED_IDENTIFICATION
            else OPEN_TEXT_EVIDENCE_TEMPLATE.format(
                evidence=evidence.text.strip(), question=question
            )
        )
        visible_evidence_text = evidence.text.strip()
    else:
        model_visible_prompt = question
        visible_evidence_text = None

    leakage = audit_prompt_leakage(
        protocol=protocol,
        modality=evidence.modality,
        instruction=question,
        visible_evidence_text=visible_evidence_text,
        transcript=evidence.transcript,
        source=source,
        target=target,
        external_candidates=names,
        property_answers=property_answers,
        media_reference=evidence.media_reference,
    )
    if strict:
        assert_prompt_leakage_clean(leakage)

    token_ids = (
        {name: [int(i) for i in encode_candidate(f" {name}")] for name in names}
        if encode_candidate is not None
        else None
    )
    if token_ids is not None:
        empty = sorted(name for name, ids in token_ids.items() if not ids)
        if empty:
            raise PromptProtocolError(f"candidate(s) {empty} encoded to zero tokens")

    prompt_ids = (
        tuple(int(i) for i in encode_prompt(model_visible_prompt))
        if encode_prompt is not None
        else None
    )

    return BuiltPrompt(
        protocol=protocol,
        modality=evidence.modality,
        question=question,
        question_hash=text_hash(question),
        model_visible_prompt=model_visible_prompt,
        media_input=evidence.media,
        prompt_hash=text_hash(model_visible_prompt),
        prompt_token_ids=prompt_ids,
        prompt_len=None if prompt_ids is None else len(prompt_ids),
        external_candidates=names,
        external_candidate_token_ids=token_ids,
        candidate_visibility={
            "rule": CANDIDATE_VISIBILITY_RULE,
            "candidates_in_prompt": protocol == CANDIDATE_LISTED_IDENTIFICATION,
            "candidates_are_external": protocol != CANDIDATE_LISTED_IDENTIFICATION,
            "scoring_version": CANDIDATE_SCORING_VERSION,
            "prompt_boundary_rule": PROMPT_BOUNDARY_RULE,
        },
        leakage=leakage,
        evidence=evidence.to_dict(),
        source_concept=source.to_dict() if source else None,
        target_concept=target.to_dict() if target else None,
        property_answers={
            key: list(_surfaces(value)) for key, value in dict(property_answers or {}).items()
        },
        sampling_rate=evidence.sampling_rate,
        media_reference=evidence.media_reference,
        media_checksum=evidence.media_checksum,
        transcript_hash=(
            text_hash(evidence.transcript) if evidence.transcript is not None else None
        ),
    )


# --------------------------------------------------------- the backend edge

#: Keyword names a backend must never receive. The transcript is the dangerous
#: one: it is the whole spoken-audio condition's claim that it stays offline.
FORBIDDEN_BACKEND_KWARGS: tuple[str, ...] = (
    "transcript",
    "caption",
    "candidates",
    "candidate_ids",
    "manifest",
    "metadata",
    "target_concept",
    "answer",
)


def backend_input_kwargs(built: BuiltPrompt, *, transcript: str | None = None) -> dict:
    """Exactly the arguments :meth:`PilotBackend.build_inputs` may receive.

    Args:
        transcript: The offline transcript, passed **only** so this function can
            prove it is absent from every value it returns. It is never put into
            the result.

    Raises:
        PromptProtocolError: If the transcript would reach the model, or if a
            forbidden keyword somehow appears.
    """
    kwargs: dict[str, Any] = {
        "prompt": built.model_visible_prompt,
        "modality": built.modality,
    }
    if built.modality == "image":
        kwargs["image"] = built.media_input
    elif built.modality == "spoken_audio":
        kwargs["audio"] = built.media_input
        if built.sampling_rate is not None:
            kwargs["sampling_rate"] = int(built.sampling_rate)
    if built.media_reference and built.modality != "text":
        # Read for a byte checksum by the backend; never rendered into a prompt.
        kwargs["media_path"] = built.media_reference

    forbidden = sorted(set(kwargs) & set(FORBIDDEN_BACKEND_KWARGS))
    if forbidden:  # pragma: no cover - structurally unreachable, checked anyway
        raise PromptProtocolError(f"forbidden backend kwargs {forbidden}")
    if transcript and transcript.strip():
        haystack = normalize(
            " ".join(str(value) for value in kwargs.values() if isinstance(value, str))
        )
        if contains_surface(haystack, transcript):
            raise PromptProtocolError(
                "the spoken-audio transcript reached the backend arguments. The "
                "recording is the only evidence; the transcript is read by the "
                "offline leakage audit and by nothing else."
            )
    return kwargs


def build_backend_inputs(backend: Any, built: BuiltPrompt, *, transcript: str | None = None):
    """Build the backend's :class:`~jlens.mmpilot.backend.BuiltInputs` for ``built``.

    The single place a protocol-checked prompt crosses into the model, so the
    check that the transcript did not cross with it happens exactly once.
    """
    return backend.build_inputs(**backend_input_kwargs(built, transcript=transcript))


# ------------------------------------------------------------ fingerprinting


def prompt_protocol_fingerprint(
    built: BuiltPrompt,
    *,
    model_revision: str,
    processor_revision: str,
    audio_protocol_fingerprint: str | None = None,
) -> dict:
    """The prompt half of a future open-prompt run's fingerprint.

    Binds everything that decides *what was asked and what was scored*. Two
    properties matter and are tested:

    * the **prompt hash** does not depend on the candidate enumeration order —
      the candidates are not in the prompt;
    * the **fingerprint** does depend on the candidate set and on its token ids —
      a run that scored a different set measured a different thing.
    """
    candidate_ids = built.external_candidate_token_ids or {}
    payload = {
        "prompt_protocol_version": built.protocol_version,
        "question_template": built.question,
        "question_hash": built.question_hash,
        "prompt_hash": built.prompt_hash,
        "candidate_visibility_rule": CANDIDATE_VISIBILITY_RULE,
        "candidates_in_prompt": bool(built.candidate_visibility["candidates_in_prompt"]),
        "leakage_audit_version": built.leakage["audit_version"],
        "leakage_audit_passed": bool(built.leakage["passed"]),
        "source_concept": (built.source_concept or {}).get("name"),
        "target_concept": (built.target_concept or {}).get("name"),
        "registered_aliases_checksum": built.leakage["registered_aliases_checksum"],
        "external_candidates": sorted(built.external_candidates),
        "external_candidate_token_ids": {
            name: [int(i) for i in candidate_ids[name]] for name in sorted(candidate_ids)
        },
        "candidate_scoring_version": CANDIDATE_SCORING_VERSION,
        "prompt_boundary_rule": PROMPT_BOUNDARY_RULE,
        "modality": built.modality,
        "model_revision": str(model_revision),
        "processor_revision": str(processor_revision),
        "audio_protocol_fingerprint": (
            audio_protocol_fingerprint if built.modality == "spoken_audio" else None
        ),
        "property_answers": {
            key: sorted(value) for key, value in sorted(built.property_answers.items())
        },
    }
    return {**payload, "prompt_protocol_digest": payload_checksum(payload)}


# ------------------------------------------------------- claim admissibility

#: The strongest claim each protocol can ever reach, before any control is
#: considered. Nothing in this module raises a claim above its protocol's entry.
MAXIMUM_CLAIM: dict[str, str] = {
    CANDIDATE_LISTED_IDENTIFICATION: "candidate_conditioned_identification",
    OPEN_IDENTIFICATION: "open_cross_modal_identification",
    OPEN_DOWNSTREAM_PROPERTY: "downstream_property_recomputation",
    HIDDEN_INTERMEDIATE: "hidden_intermediate_multi_hop_reasoning",
}

#: What each protocol may never support, stated so a report cannot imply it.
EXCLUDED_CLAIMS: dict[str, tuple[str, ...]] = {
    CANDIDATE_LISTED_IDENTIFICATION: (
        "spontaneous unprompted concept emergence",
        "open cross-modal identification",
        "downstream property recomputation",
        "multi-hop reasoning",
    ),
    OPEN_IDENTIFICATION: (
        "downstream property recomputation",
        "multi-hop reasoning",
    ),
    OPEN_DOWNSTREAM_PROPERTY: ("multi-hop reasoning",),
    HIDDEN_INTERMEDIATE: (),
}


def protocol_claim_admissibility(
    *,
    protocol: str,
    leakage: Mapping | None,
    mode: str,
    identity_replacement_passed: bool | None = None,
    direct_answer_control_passed: bool | None = None,
    direct_answer_onset_control_passed: bool | None = None,
) -> dict:
    """The strongest claim a *future* result under ``protocol`` may support.

    This decides admissibility from **predeclared** facts only: which protocol
    was used, whether the registered audit cleared it, and whether named
    controls passed. It never reads an effect size, and there is deliberately no
    path by which a claim is raised after a result is seen — a stronger claim
    requires a stronger protocol, run again.

    Args:
        protocol: One of :data:`PROTOCOLS`.
        leakage: The audit record from :func:`audit_prompt_leakage`.
        mode: ``"mock"`` makes every claim inadmissible, whatever else passed.
        identity_replacement_passed: Whether the identity condition actually
            replaced the identity. Required by
            :data:`OPEN_DOWNSTREAM_PROPERTY`.
        direct_answer_control_passed: Whether inserting the answer's own lens
            vector failed to reproduce the effect. Required by
            :data:`OPEN_DOWNSTREAM_PROPERTY`.
        direct_answer_onset_control_passed: The onset version of the same
            control. Required by :data:`HIDDEN_INTERMEDIATE`.

    Returns:
        ``{"admissible", "maximum_claim", "granted_claim", "reasons", ...}``.
        ``granted_claim`` is ``None`` whenever ``admissible`` is False.
    """
    if protocol not in PROTOCOLS:
        raise PromptProtocolError(f"unknown prompt protocol {protocol!r}; known: {PROTOCOLS}")

    reasons: list[str] = []
    requirements: dict[str, Any] = {"protocol": protocol}

    if str(mode).lower() != "real":
        reasons.append(
            f"mode is {mode!r}: no MOCK or synthetic result supports any "
            "scientific claim, regardless of which checks passed"
        )

    audit_passed = bool((leakage or {}).get("passed"))
    requirements["leakage_audit_passed"] = audit_passed
    if leakage is None:
        reasons.append("no leakage-audit record was supplied")
    elif leakage.get("protocol") != protocol:
        reasons.append(
            f"the leakage record was produced under {leakage.get('protocol')!r}, "
            f"not {protocol!r}"
        )
    elif not audit_passed:
        reasons.append(
            "the registered deterministic leakage audit refused this prompt: "
            f"{sorted(leakage.get('violations', ())) or sorted(leakage.get('unauditable', ()))}"
        )

    if protocol == OPEN_DOWNSTREAM_PROPERTY:
        requirements["identity_replacement_passed"] = identity_replacement_passed
        requirements["direct_answer_control_passed"] = direct_answer_control_passed
        if not identity_replacement_passed:
            reasons.append(
                "downstream recomputation requires the identity condition to have "
                "replaced the identity on the same evidence; without it a property "
                "change has no established cause"
            )
        if not direct_answer_control_passed:
            reasons.append(
                "the direct-answer-vector control did not pass: if inserting the "
                "answer's own lens vector moves the answer as well, the effect is "
                "a shortcut rather than recomputation"
            )
    if protocol == HIDDEN_INTERMEDIATE:
        requirements["direct_answer_onset_control_passed"] = (
            direct_answer_onset_control_passed
        )
        if not direct_answer_onset_control_passed:
            reasons.append(
                "multi-hop reasoning requires the direct-answer onset control to "
                "have passed"
            )
        for name in ("source_in_visible_evidence", "target_in_visible_evidence",
                     "source_in_audio_transcript", "target_in_audio_transcript"):
            finding = ((leakage or {}).get("findings") or {}).get(name) or {}
            if finding.get("status") == "violation":
                reasons.append(f"{name} was detected; both entity names must be absent")

    admissible = not reasons
    return {
        "schema": "jlens.mmpilot.prompt_protocol_claim_admissibility.v1",
        "rule_version": CLAIM_RULE_VERSION,
        "protocol": protocol,
        "mode": mode,
        "maximum_claim": MAXIMUM_CLAIM[protocol],
        "granted_claim": MAXIMUM_CLAIM[protocol] if admissible else None,
        "admissible": admissible,
        "excluded_claims": list(EXCLUDED_CLAIMS[protocol]),
        "requirements": requirements,
        "reasons": reasons,
        "automatic_upgrade_forbidden": (
            "A claim is never raised after a result is seen. A stronger claim "
            "requires a stronger protocol and a new run."
        ),
    }


def claim_admissibility_rule_record() -> dict:
    """The protocol -> claim rule itself, checksummed, for artifacts to bind."""
    payload = {
        "rule_version": CLAIM_RULE_VERSION,
        "leakage_audit_version": AUDIT_VERSION,
        "candidate_scoring_version": CANDIDATE_SCORING_VERSION,
        "maximum_claim": dict(MAXIMUM_CLAIM),
        "excluded_claims": {k: list(v) for k, v in EXCLUDED_CLAIMS.items()},
        "leakage_policy": {k: dict(v) for k, v in LEAKAGE_POLICY.items()},
        "statements": [
            "A candidate_listed_identification result supports only a "
            "candidate-conditioned claim.",
            "An open_identification result supports open cross-modal "
            "identification only when the target leakage checks pass.",
            "An open_downstream_property result supports downstream "
            "recomputation only when identity replacement also succeeds and the "
            "direct-answer controls pass.",
            "A hidden_intermediate result supports multi-hop reasoning only when "
            "both entity names are absent under the registered deterministic "
            "audit and the direct-answer onset control passes.",
            "No MOCK result supports any scientific claim.",
        ],
    }
    return {**payload, "rule_checksum": payload_checksum(payload)}


__all__ = [
    "AUDIT_LIMITS",
    "AUDIT_VERSION",
    "CANDIDATE_LISTED_IDENTIFICATION",
    "CANDIDATE_SCORING_VERSION",
    "CANDIDATE_VISIBILITY_RULE",
    "CLAIM_RULE_VERSION",
    "DEFAULT_ALIASES",
    "DEFAULT_QUESTIONS",
    "EXCLUDED_CLAIMS",
    "FORBIDDEN_BACKEND_KWARGS",
    "HIDDEN_INTERMEDIATE",
    "HIDDEN_INTERMEDIATE_QUESTION",
    "LEAKAGE_CATEGORIES",
    "LEAKAGE_POLICY",
    "LEGACY_CAPABILITY_PROMPT_PROTOCOL",
    "MAXIMUM_CLAIM",
    "MODALITIES",
    "NORMALIZATION_RULE",
    "OPEN_DOWNSTREAM_PROPERTY",
    "OPEN_IDENTIFICATION",
    "OPEN_IDENTIFICATION_QUESTION",
    "OPEN_PROPERTY_QUESTION",
    "OPEN_PROTOCOLS",
    "OPEN_TEXT_EVIDENCE_TEMPLATE",
    "PROTOCOLS",
    "BuiltPrompt",
    "ConceptSpec",
    "Evidence",
    "PromptLeakageError",
    "PromptProtocolError",
    "aliases_checksum",
    "assert_prompt_leakage_clean",
    "audit_prompt_leakage",
    "backend_input_kwargs",
    "build_backend_inputs",
    "build_protocol_prompt",
    "claim_admissibility_rule_record",
    "concept_spec",
    "contains_surface",
    "looks_like_candidate_enumeration",
    "normalize",
    "prompt_protocol_fingerprint",
    "protocol_claim_admissibility",
]
